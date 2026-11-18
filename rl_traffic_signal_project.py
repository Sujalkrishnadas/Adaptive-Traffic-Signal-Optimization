#!/usr/bin/env python3
"""
RL Project 6 - Adaptive Traffic Signal Optimization (Smart Cities)
===================================================================
Single-intersection, simulation-based study.

Pipeline
  1. Generate synthetic datasets  : 10,000 training scenario records + 3,000 unseen test
                                    scenarios (1,000 per difficulty level).
  2. Environment                  : vectorised intersection simulator (phase + green-time).
  3. Algorithms (all from the PDF): Q-Learning, SARSA, DQN, DDQN, Dueling DQN, PPO, A2C
  4. Baselines / proposed         : RL baselines = Q-Learning, DQN
                                    Proposed     = Dueling Double DQN + Prioritized Replay (D3QN-PER)
                                    Non-learning references = Fixed-Time, Max-Pressure
  5. Ablations (3)                : Proposed minus {Dueling, Double, PER}
  6. Protocol                     : 1,000 training episodes, 5 seeds, 1,000 unseen test
                                    scenarios per level, 3 difficulty levels, 5 metrics
  7. Outputs                      : CSVs, figures, Excel summary sheet, markdown report

Usage
  python rl_traffic_signal_project.py                 # full protocol (long)
  python rl_traffic_signal_project.py --quick         # smoke test (minutes)
  python rl_traffic_signal_project.py --jobs 4        # run (algo, seed) jobs in parallel
  python rl_traffic_signal_project.py --out results   # output folder

NOTE: all findings are SIMULATION-BASED and must not be presented as proof of real-world
performance without experimental validation.
"""
import argparse, os, time, json, copy
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------------------------
CFG = dict(
    n_train_records=10_000,       # PDF minimum synthetic records for project 6
    n_test_per_level=1_000,       # PDF: 1,000 unseen test scenarios (per difficulty level)
    train_episodes=1_000,         # PDF: 1,000 training episodes
    seeds=[0, 1, 2, 3, 4],        # PDF: 5 independent seeds
    horizon=3600.0,               # seconds per episode (1 hour)
    green_options=[10, 20, 30],   # green time choices (s)
    yellow=3.0, startup_lost=2.0, # lost time when switching phase
    sat_flow=0.5,                 # veh/s/approach during green (1800 veh/h)
    gamma=0.98,
    reward_scale=2000.0,
)
LEVELS = ["Low", "Medium", "High"]
LOAD_FACTOR = {0: 0.50, 1: 0.75, 2: 0.95}   # demand relative to ~0.2 veh/s/approach capacity
N_ACTIONS = 6                                # {NS,EW} x {10,20,30 s}
OBS_DIM = 11

RL_ALGOS = ["QLearning", "SARSA", "DQN", "DDQN", "DuelingDQN", "PPO", "A2C"]
PROPOSED = "D3QN_PER"
ABLATIONS = ["Abl_noDueling", "Abl_noDouble", "Abl_noPER"]
HEURISTICS = ["FixedTime", "MaxPressure"]
BASELINES = ["QLearning", "DQN"]             # the 2 RL baselines required by the PDF
ALL_LEARNERS = RL_ALGOS + [PROPOSED] + ABLATIONS
ALL_METHODS = HEURISTICS + ALL_LEARNERS


# ----------------------------------------------------------------------------------------
# 1. SYNTHETIC DATASET
# ----------------------------------------------------------------------------------------
def hour_profile(h):
    """Daily demand profile with AM (8h) and PM (17h) peaks."""
    return 0.55 + 0.45 * np.exp(-0.5 * ((h - 8) / 1.8) ** 2) + 0.55 * np.exp(-0.5 * ((h - 17.5) / 2.0) ** 2)


def generate_scenarios(n, seed, levels_balanced=True):
    rng = np.random.default_rng(seed)
    diff = np.arange(n) % 3 if levels_balanced else rng.integers(0, 3, n)
    rng.shuffle(diff)
    hour = rng.integers(0, 24, n)
    day = rng.choice(["weekday", "weekend"], n, p=[0.71, 0.29])
    weather = rng.choice(["clear", "rain", "fog"], n, p=[0.7, 0.2, 0.1])
    sat_factor = np.select([weather == "clear", weather == "rain"], [1.0, 0.9], 0.85)
    day_f = np.where(day == "weekend", 0.85, 1.0)
    base = np.array([LOAD_FACTOR[d] for d in diff]) * 0.2 * day_f * hour_profile(hour) / hour_profile(np.arange(24)).mean()
    imb = rng.uniform(0.6, 1.4, n)                         # NS vs EW imbalance
    dirs = np.stack([imb, imb, 2 - imb, 2 - imb], 1)       # N, S, E, W
    noise = rng.normal(1, 0.1, (n, 4))
    lam = np.clip(base[:, None] * dirs * noise, 0.01, 0.32)
    return pd.DataFrame(dict(
        record_id=np.arange(n), hour=hour, day_type=day, weather=weather,
        difficulty=diff, difficulty_name=[LEVELS[d] for d in diff],
        lam_N=lam[:, 0], lam_S=lam[:, 1], lam_E=lam[:, 2], lam_W=lam[:, 3],   # veh/s
        surge_amp=rng.uniform(0.0, 0.5, n), surge_phase=rng.uniform(0, 2 * np.pi, n),
        sat_factor=sat_factor))


def scen_to_arrays(df):
    return dict(lam=df[["lam_N", "lam_S", "lam_E", "lam_W"]].values.astype(np.float64),
                amp=df["surge_amp"].values, ph=df["surge_phase"].values,
                sat=df["sat_factor"].values, diff=df["difficulty"].values)


# ----------------------------------------------------------------------------------------
# 2. VECTORISED ENVIRONMENT
# ----------------------------------------------------------------------------------------
class TrafficEnv:
    """N independent intersections. Action = phase (NS/EW) x green duration."""

    def __init__(self, sc, seed):
        self.sc, self.n = sc, len(sc["amp"])
        self.rng = np.random.default_rng(seed)
        self.H = CFG["horizon"]
        self.durs = np.array(CFG["green_options"], dtype=float)
        self.reset()

    def reset(self):
        n = self.n
        self.q = np.zeros((n, 4)); self.t = np.zeros(n); self.phase = np.zeros(n, dtype=int)
        self.last_arr = np.zeros((n, 4))
        self.wait = np.zeros(n); self.arrived = np.zeros(n); self.served = np.zeros(n)
        self.switches = np.zeros(n); self.qsum_time = np.zeros(n)
        return self.obs()

    def active(self):
        return self.t < self.H

    def obs(self):
        o = np.zeros((self.n, OBS_DIM), dtype=np.float32)
        o[:, 0:4] = np.minimum(self.q / 40.0, 3.0)
        o[:, 4:8] = np.minimum(self.last_arr / 0.3, 2.0)
        o[np.arange(self.n), 8 + self.phase] = 1.0
        o[:, 10] = self.t / self.H
        return o

    def tab_state(self, o=None):
        """Discrete state for tabular methods: (NS queue bin, EW queue bin, phase) -> 72 states."""
        edges = np.array([5, 10, 20, 40, 80])
        ns = np.digitize(self.q[:, 0] + self.q[:, 1], edges)
        ew = np.digitize(self.q[:, 2] + self.q[:, 3], edges)
        return (ns * 6 + ew) * 2 + self.phase

    def step(self, a):
        act = self.active()
        a = np.asarray(a); n = self.n
        ph_new = a // 3; dur = self.durs[a % 3]
        sw = (ph_new != self.phase)
        T = dur + np.where(sw, CFG["yellow"], 0.0)
        eff = dur - np.where(sw, CFG["startup_lost"], 0.0)
        mult = 1.0 + self.sc["amp"] * np.sin(2 * np.pi * self.t / self.H + self.sc["ph"])
        arr = self.rng.poisson(self.sc["lam"] * mult[:, None] * T[:, None]).astype(float)
        cap = CFG["sat_flow"] * self.sc["sat"] * eff
        green = np.zeros((n, 4), dtype=bool)
        green[ph_new == 0, 0:2] = True; green[ph_new == 1, 2:4] = True
        srv = np.where(green, np.minimum(self.q + arr, cap[:, None]), 0.0)
        q_new = self.q + arr - srv
        vehsec = T * 0.5 * (self.q + q_new).sum(1)            # integral of total queue over step
        reward = -vehsec / CFG["reward_scale"]
        m = act
        self.wait += np.where(m, vehsec, 0); self.arrived += np.where(m, arr.sum(1), 0)
        self.served += np.where(m, srv.sum(1), 0); self.switches += np.where(m & sw, 1, 0)
        self.qsum_time += np.where(m, T, 0)
        self.q = np.where(m[:, None], q_new, self.q)
        self.last_arr = np.where(m[:, None], arr / T[:, None], self.last_arr)
        self.phase = np.where(m, ph_new, self.phase)
        self.t = self.t + np.where(m, T, 0)
        return self.obs(), np.where(m, reward, 0.0), ~self.active(), m

    def metrics(self):
        """Per-scenario metrics (Little's law used for per-vehicle waiting time)."""
        tt = np.maximum(self.t, 1.0)
        return dict(
            avg_wait_s=self.wait / np.maximum(self.arrived, 1),        # s per vehicle
            avg_queue=self.wait / tt / 4.0,                             # veh per approach
            veh_served_per_hr=self.served * 3600.0 / tt,                # veh/h
            throughput_ratio=self.served / np.maximum(self.arrived, 1), # served / arrived
            switches=self.switches)


# ----------------------------------------------------------------------------------------
# 3. HEURISTIC CONTROLLERS (non-learning reference)
# ----------------------------------------------------------------------------------------
class FixedTime:
    name = "FixedTime"
    def act(self, env, o): return (1 - env.phase) * 3 + 1               # alternate, 20 s green

class MaxPressure:
    name = "MaxPressure"
    def act(self, env, o):
        ns, ew = env.q[:, 0] + env.q[:, 1], env.q[:, 2] + env.q[:, 3]
        ph = (ew > ns).astype(int); qq = np.maximum(ns, ew)
        d = np.where(qq < 8, 0, np.where(qq < 20, 1, 2))
        return ph * 3 + d


# ----------------------------------------------------------------------------------------
# 4. LEARNING AGENTS
# ----------------------------------------------------------------------------------------
def eps_at(ep, total, e0=1.0, e1=0.05, frac=0.6):
    return max(e1, e0 - (e0 - e1) * ep / max(1, frac * total))


class TabularAgent:
    """Q-Learning (off-policy) and SARSA (on-policy)."""
    def __init__(self, kind, seed):
        self.kind, self.rng = kind, np.random.default_rng(seed)
        self.Q = np.zeros((72, N_ACTIONS)); self.alpha, self.gamma = 0.1, CFG["gamma"]

    def act(self, env, o, eps=0.0):
        s = env.tab_state(); a = self.Q[s].argmax(1)
        if eps > 0:
            r = self.rng.random(len(a)) < eps; a = np.where(r, self.rng.integers(0, N_ACTIONS, len(a)), a)
        return a

    def train(self, sampler, episodes, log):
        for ep in range(episodes):
            env = sampler(1); env.reset(); eps = eps_at(ep, episodes); R = 0.0
            s = env.tab_state()[0]; a = self.act(env, None, eps)[0]
            while True:
                _, r, done, _ = env.step([a]); s2 = env.tab_state()[0]; R += r[0]
                a2 = self.act(env, None, eps)[0]
                nxt = self.Q[s2, a2] if self.kind == "SARSA" else self.Q[s2].max()
                target = r[0] + (0 if done[0] else self.gamma * nxt)
                self.Q[s, a] += self.alpha * (target - self.Q[s, a])
                s, a = s2, a2
                if done[0]: break
            log(ep, R, env.metrics()["avg_wait_s"][0], eps)


def build_mlp(i, o, h=128):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


def make_qnet(dueling):
    import torch, torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(OBS_DIM, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
            if dueling:
                self.v = nn.Linear(128, 1); self.adv = nn.Linear(128, N_ACTIONS)
            else:
                self.out = nn.Linear(128, N_ACTIONS)
        def forward(self, x):
            h = self.body(x)
            if dueling:
                ad = self.adv(h); return self.v(h) + ad - ad.mean(1, keepdim=True)
            return self.out(h)
    return Net()


class DQNAgent:
    """DQN family: vanilla / double / dueling / prioritized replay (PER as an extension)."""
    def __init__(self, seed, double=False, dueling=False, per=False,
                 lr=1e-3, batch=64, buf=50_000, train_every=4, target_every=250, warmup=1000):
        import torch
        self.torch = torch; torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.double, self.per = double, per
        self.q, self.tq = make_qnet(dueling), make_qnet(dueling)
        self.tq.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.cap, self.batch = buf, batch
        self.train_every, self.target_every, self.warmup = train_every, target_every, warmup
        self.S = np.zeros((buf, OBS_DIM), np.float32); self.S2 = np.zeros_like(self.S)
        self.A = np.zeros(buf, np.int64); self.R = np.zeros(buf, np.float32); self.D = np.zeros(buf, np.float32)
        self.P = np.ones(buf, np.float64); self.n = 0; self.ptr = 0; self.steps = 0; self.upd = 0
        self.alpha_per, self.beta = 0.6, 0.4

    def act(self, env, o, eps=0.0):
        with self.torch.no_grad():
            a = self.q(self.torch.from_numpy(o)).argmax(1).numpy()
        if eps > 0:
            r = self.rng.random(len(a)) < eps; a = np.where(r, self.rng.integers(0, N_ACTIONS, len(a)), a)
        return a

    def store(self, s, a, r, s2, d):
        i = self.ptr
        self.S[i], self.A[i], self.R[i], self.S2[i], self.D[i] = s, a, r, s2, d
        self.P[i] = self.P[:self.n].max() if self.n else 1.0
        self.ptr = (self.ptr + 1) % self.cap; self.n = min(self.n + 1, self.cap)

    def learn(self):
        torch = self.torch
        if self.per:
            p = self.P[:self.n] ** self.alpha_per; p /= p.sum()
            idx = self.rng.choice(self.n, self.batch, p=p)
            w = (self.n * p[idx]) ** (-self.beta); w = torch.tensor(w / w.max(), dtype=torch.float32)
        else:
            idx = self.rng.integers(0, self.n, self.batch); w = torch.ones(self.batch)
        s, s2 = torch.from_numpy(self.S[idx]), torch.from_numpy(self.S2[idx])
        a, r, d = torch.from_numpy(self.A[idx]), torch.from_numpy(self.R[idx]), torch.from_numpy(self.D[idx])
        qsa = self.q(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            if self.double:
                a2 = self.q(s2).argmax(1, keepdim=True); nq = self.tq(s2).gather(1, a2).squeeze(1)
            else:
                nq = self.tq(s2).max(1)[0]
            y = r + CFG["gamma"] * (1 - d) * nq
        td = y - qsa
        loss = (w * torch.nn.functional.smooth_l1_loss(qsa, y, reduction="none")).mean()
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0); self.opt.step()
        if self.per: self.P[idx] = np.abs(td.detach().numpy()) + 1e-3
        self.upd += 1
        if self.upd % self.target_every == 0: self.tq.load_state_dict(self.q.state_dict())

    def train(self, sampler, episodes, log):
        for ep in range(episodes):
            env = sampler(1); o = env.reset(); eps = eps_at(ep, episodes); R = 0.0
            if self.per: self.beta = min(1.0, 0.4 + 0.6 * ep / episodes)
            while True:
                a = self.act(env, o, eps)
                o2, r, done, _ = env.step(a); R += r[0]
                self.store(o[0], a[0], r[0], o2[0], float(done[0]))
                o = o2; self.steps += 1
                if self.n >= self.warmup and self.steps % self.train_every == 0: self.learn()
                if done[0]: break
            log(ep, R, env.metrics()["avg_wait_s"][0], eps)


class ActorCritic:
    """Shared-body discrete actor-critic used by A2C (n-step) and PPO (clipped)."""
    def __init__(self, seed, kind):
        import torch, torch.nn as nn
        self.torch = torch; torch.manual_seed(seed); self.kind = kind
        self.rng = np.random.default_rng(seed)
        # separate actor and critic networks (a shared body let the large value loss swamp the policy)
        self.actor = nn.Sequential(nn.Linear(OBS_DIM, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, N_ACTIONS))
        self.critic = nn.Sequential(nn.Linear(OBS_DIM, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        self.vscale = 20.0                      # critic predicts return / vscale
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.params = params
        self.opt = torch.optim.Adam(params, lr=2.5e-4 if kind == "A2C" else 3e-4)

    def fwd(self, x):
        return self.actor(x), self.critic(x).squeeze(-1) * self.vscale

    def act(self, env, o, eps=0.0, sample=False):
        torch = self.torch
        with torch.no_grad():
            lg, _ = self.fwd(torch.from_numpy(o))
            if sample:
                return torch.distributions.Categorical(logits=lg).sample().numpy()
            return lg.argmax(1).numpy()

    def _gae(self, rs, vs, last_v, lam=0.95):
        """Generalised advantage estimation for one finished (terminal) trajectory."""
        g = CFG["gamma"]; T = len(rs); adv = np.zeros(T, np.float32); gae = 0.0; nxt_v = last_v
        for t in reversed(range(T)):
            nonterm = 0.0 if t == T - 1 else 1.0          # episode ends at T-1 (time limit)
            delta = rs[t] + g * nxt_v * nonterm - vs[t]
            gae = delta + g * lam * nonterm * gae
            adv[t] = gae; nxt_v = vs[t]
        return adv

    def _loss(self, s, a, adv, ret, old_lp=None, clip=0.2):
        torch = self.torch
        lg, v = self.fwd(s); dist = torch.distributions.Categorical(logits=lg)
        lp = dist.log_prob(a); ent = dist.entropy().mean()
        if old_lp is None:
            pl = -(lp * adv).mean()
        else:
            ratio = torch.exp(lp - old_lp)
            pl = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
        return pl + 0.5 * (((ret - v) / self.vscale) ** 2).mean() - 0.02 * ent

    def train(self, sampler, episodes, log):
        torch = self.torch
        if self.kind == "A2C": self._train_a2c(sampler, episodes, log, torch)
        else: self._train_ppo(sampler, episodes, log, torch)

    def _train_a2c(self, sampler, episodes, log, torch, nstep=40):
        for ep in range(episodes):
            env = sampler(1); o = env.reset(); R = 0.0; S, A, Rw = [], [], []
            while True:
                a = self.act(env, o, sample=True); o2, r, done, _ = env.step(a); R += r[0]
                S.append(o[0]); A.append(a[0]); Rw.append(r[0]); o = o2
                if len(S) == nstep or done[0]:
                    s = torch.tensor(np.array(S)); aa = torch.tensor(np.array(A))
                    with torch.no_grad():
                        _, vs = self.fwd(s); _, lv = self.fwd(torch.from_numpy(o))
                    # n-step discounted returns
                    ret = np.zeros(len(S), np.float32); g = 0.0 if done[0] else float(lv[0])
                    for t in reversed(range(len(S))): g = Rw[t] + CFG["gamma"] * g; ret[t] = g
                    ret_t = torch.tensor(ret); adv = ret_t - vs
                    if len(adv) > 1: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    loss = self._loss(s, aa, adv, ret_t)
                    self.opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.params, 0.5); self.opt.step()
                    S, A, Rw = [], [], []
                if done[0]: break
            log(ep, R, env.metrics()["avg_wait_s"][0], float("nan"))

    def _train_ppo(self, sampler, episodes, log, torch, n_env=2, epochs=8, mb=128):
        ep = 0
        while ep < episodes:
            k = min(n_env, episodes - ep); env = sampler(k); o = env.reset()
            tr = [dict(s=[], a=[], r=[], lp=[], v=[]) for _ in range(k)]; Rtot = np.zeros(k)
            while True:
                with torch.no_grad():
                    lg, v = self.fwd(torch.from_numpy(o)); dist = torch.distributions.Categorical(logits=lg)
                    a = dist.sample(); lp = dist.log_prob(a)
                o2, r, done, m = env.step(a.numpy()); Rtot += r
                for i in range(k):
                    if m[i]:
                        tr[i]["s"].append(o[i]); tr[i]["a"].append(a[i].item()); tr[i]["r"].append(r[i])
                        tr[i]["lp"].append(lp[i].item()); tr[i]["v"].append(v[i].item())
                o = o2
                if done.all(): break
            S, A, LP, ADV, RET = [], [], [], [], []
            for i in range(k):
                vs = np.array(tr[i]["v"], np.float32)
                adv = self._gae(tr[i]["r"], vs, 0.0)
                S += tr[i]["s"]; A += tr[i]["a"]; LP += tr[i]["lp"]; ADV.append(adv); RET.append(adv + vs)
            S = torch.tensor(np.array(S)); A = torch.tensor(A); LP = torch.tensor(LP, dtype=torch.float32)
            ADV = torch.tensor(np.concatenate(ADV)); RET = torch.tensor(np.concatenate(RET))
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
            for _ in range(epochs):
                perm = torch.randperm(len(S))
                for j in range(0, len(S), mb):
                    ix = perm[j:j + mb]
                    loss = self._loss(S[ix], A[ix], ADV[ix], RET[ix], LP[ix])
                    self.opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.params, 0.5); self.opt.step()
            mt = env.metrics()["avg_wait_s"]
            for i in range(k): log(ep + i, Rtot[i], mt[i], float("nan"))
            ep += k


def make_agent(name, seed):
    if name == "QLearning": return TabularAgent("Q", seed)
    if name == "SARSA": return TabularAgent("SARSA", seed)
    if name == "DQN": return DQNAgent(seed)
    if name == "DDQN": return DQNAgent(seed, double=True)
    if name == "DuelingDQN": return DQNAgent(seed, dueling=True)
    if name == PROPOSED: return DQNAgent(seed, double=True, dueling=True, per=True)
    if name == "Abl_noDueling": return DQNAgent(seed, double=True, dueling=False, per=True)
    if name == "Abl_noDouble": return DQNAgent(seed, double=False, dueling=True, per=True)
    if name == "Abl_noPER": return DQNAgent(seed, double=True, dueling=True, per=False)
    if name in ("PPO", "A2C"): return ActorCritic(seed, name)
    raise ValueError(name)


# ----------------------------------------------------------------------------------------
# 5. TRAIN / EVALUATE ONE (method, seed) JOB
# ----------------------------------------------------------------------------------------
def evaluate(policy, test_df, seed):
    """Run policy greedily on all unseen test scenarios; returns per-level metric means."""
    rows = []
    for lv in range(3):
        sub = test_df[test_df.difficulty == lv]
        env = TrafficEnv(scen_to_arrays(sub), seed=10_000 + seed * 7 + lv)   # same noise stream per seed/level
        o = env.reset()
        while env.active().any():
            if hasattr(policy, "name"): a = policy.act(env, o)          # heuristic
            else: a = policy.act(env, o, 0.0)
            o, _, _, _ = env.step(a)
        m = env.metrics()
        rows.append(dict(level=LEVELS[lv], n_scenarios=len(sub), **{k: float(v.mean()) for k, v in m.items()}))
    return rows


def run_job(args):
    method, seed, train_df, test_df, episodes = args
    try:
        import torch; torch.set_num_threads(1)
    except Exception:
        pass
    t0 = time.time(); np.random.seed(seed); curve = []
    arrays = scen_to_arrays(train_df); nrec = len(train_df); rng = np.random.default_rng(seed + 123)

    def sampler(k):  # draw k random scenario records from the 10,000-record dataset
        idx = rng.integers(0, nrec, k)
        sc = {key: val[idx] for key, val in arrays.items()}
        return TrafficEnv(sc, seed=int(rng.integers(1 << 30)))

    if method in HEURISTICS:
        policy = FixedTime() if method == "FixedTime" else MaxPressure()
    else:
        policy = make_agent(method, seed)
        policy.train(sampler, episodes, lambda ep, R, w, e: curve.append((ep, R, w, e)))
    rows = evaluate(policy, test_df, seed)
    for r in rows: r.update(method=method, seed=seed, train_time_s=time.time() - t0)
    print(f"[done] {method:14s} seed={seed} {time.time() - t0:6.1f}s", flush=True)
    cv = pd.DataFrame(curve, columns=["episode", "ep_reward", "ep_avg_wait", "epsilon"])
    cv["method"] = method; cv["seed"] = seed
    return rows, cv


# ----------------------------------------------------------------------------------------
# 6. ANALYSIS, FIGURES, EXCEL SUMMARY SHEET, REPORT
# ----------------------------------------------------------------------------------------
METRICS = ["avg_wait_s", "avg_queue", "veh_served_per_hr", "throughput_ratio", "switches"]
NICE = {"avg_wait_s": "Avg waiting time (s/veh)", "avg_queue": "Avg queue length (veh/approach)",
        "veh_served_per_hr": "Vehicles served per hour", "throughput_ratio": "Throughput (served/arrived)",
        "switches": "Phase switches / episode"}
LOWER_BETTER = {"avg_wait_s": True, "avg_queue": True, "veh_served_per_hr": False,
                "throughput_ratio": False, "switches": True}


def analyse(raw, curves, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy import stats
    # mean +- std over seeds
    g = raw.groupby(["level", "method"])[METRICS].agg(["mean", "std"]).round(3)
    g.columns = [f"{a}_{b}" for a, b in g.columns]; g = g.reset_index()
    g.to_csv(f"{out}/summary_by_level.csv", index=False)
    overall = raw.groupby(["method", "seed"])[METRICS].mean().groupby("method").agg(["mean", "std"]).round(3)
    overall.columns = [f"{a}_{b}" for a, b in overall.columns]; overall = overall.reset_index()
    overall.to_csv(f"{out}/summary_overall.csv", index=False)

    # significance: proposed vs each other method (Welch t-test on per-seed overall avg wait)
    per_seed = raw.groupby(["method", "seed"])["avg_wait_s"].mean().unstack(0)
    sig = []
    if PROPOSED in per_seed:
        for m in per_seed.columns:
            if m == PROPOSED: continue
            x, y = per_seed[PROPOSED].dropna(), per_seed[m].dropna()
            if len(x) > 1 and len(y) > 1 and (len(set(y)) > 1 or len(set(x)) > 1):
                t, p = stats.ttest_ind(x, y, equal_var=False)
            else:
                t, p = np.nan, np.nan
            sig.append(dict(comparison=f"{PROPOSED} vs {m}", mean_diff_wait_s=round(x.mean() - y.mean(), 3),
                            t_stat=round(t, 3), p_value=round(p, 4) if p == p else np.nan,
                            significant_0_05=bool(p < 0.05) if p == p else False))
    sig = pd.DataFrame(sig); sig.to_csv(f"{out}/significance_tests.csv", index=False)

    # ablation table
    abl = overall[overall.method.isin([PROPOSED] + ABLATIONS)].copy()
    abl.to_csv(f"{out}/ablation_table.csv", index=False)

    # figures
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for m in [x for x in ALL_LEARNERS if x in curves.method.unique()]:
        c = curves[curves.method == m].groupby("episode")["ep_avg_wait"].mean()
        ax[0].plot(c.rolling(25, min_periods=1).mean().values, label=m, lw=2 if m == PROPOSED else 1)
        r = curves[curves.method == m].groupby("episode")["ep_reward"].mean()
        ax[1].plot(r.rolling(25, min_periods=1).mean().values, label=m, lw=2 if m == PROPOSED else 1)
    ax[0].set_title("Training: avg waiting time (25-ep moving avg, mean of seeds)"); ax[0].set_xlabel("episode"); ax[0].set_ylabel("s / vehicle")
    ax[1].set_title("Training: episode return"); ax[1].set_xlabel("episode"); ax[1].legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(f"{out}/fig_training_curves.png", dpi=150); plt.close()

    for metric in ["avg_wait_s", "avg_queue", "veh_served_per_hr", "throughput_ratio"]:
        piv_m = raw.groupby(["level", "method"])[metric].mean().unstack(0)[LEVELS]
        piv_s = raw.groupby(["level", "method"])[metric].std().unstack(0)[LEVELS]
        order = [m for m in ALL_METHODS if m in piv_m.index]
        piv_m, piv_s = piv_m.loc[order], piv_s.loc[order]
        piv_m.plot(kind="bar", yerr=piv_s, figsize=(12, 4.5), capsize=2)
        plt.title(NICE[metric] + " by difficulty (mean +- std over seeds)"); plt.ylabel(NICE[metric])
        plt.xticks(rotation=40, ha="right"); plt.tight_layout()
        plt.savefig(f"{out}/fig_{metric}.png", dpi=150); plt.close()

    # best method per level/metric (RL only and incl. heuristics)
    best = []
    for lv in LEVELS:
        for mt in METRICS[:4]:
            d = g[g.level == lv].set_index("method")[f"{mt}_mean"]
            d = d.sort_values(ascending=LOWER_BETTER[mt])
            best.append(dict(level=lv, metric=NICE[mt], best_method=d.index[0], value=d.iloc[0]))
    best = pd.DataFrame(best)

    # Excel summary sheet
    with pd.ExcelWriter(f"{out}/SUMMARY_SHEET.xlsx") as xw:
        pd.DataFrame(dict(
            item=["Project", "Sector", "Train records", "Unseen test scenarios", "Training episodes",
                  "Seeds", "Difficulty levels", "RL baselines", "Proposed", "Ablations", "Metrics",
                  "Validity note"],
            value=["6 - Adaptive Traffic Signal Optimization", "Smart Cities",
                   CFG["n_train_records"], f"{CFG['n_test_per_level']} x 3 levels", CFG["train_episodes_used"],
                   len(CFG["seeds_used"]), "Low / Medium / High", ", ".join(BASELINES), PROPOSED,
                   ", ".join(ABLATIONS), "avg waiting time, queue length, vehicles served/hr, throughput (+switches)",
                   "Simulation-based findings only; not proof of real-world performance"])
        ).to_excel(xw, sheet_name="Protocol", index=False)
        overall.to_excel(xw, sheet_name="Overall", index=False); g.to_excel(xw, sheet_name="ByLevel", index=False)
        abl.to_excel(xw, sheet_name="Ablation", index=False); sig.to_excel(xw, sheet_name="SignificanceTests", index=False)
        best.to_excel(xw, sheet_name="BestPerLevel", index=False); raw.to_excel(xw, sheet_name="RawResults", index=False)

    # markdown report
    ov = overall.set_index("method")
    rl_only = ov.loc[[m for m in ALL_LEARNERS if m in ov.index]]
    bst = rl_only["avg_wait_s_mean"].idxmin()
    lines = ["# Project 6 - Adaptive Traffic Signal Optimization: Auto-generated Analysis", "",
             "> Simulation-based findings only; not proof of real-world performance.", "",
             f"* Training records: {CFG['n_train_records']:,}; test scenarios: {CFG['n_test_per_level']:,} per level x 3 levels",
             f"* Training episodes: {CFG['train_episodes_used']}; seeds: {len(CFG['seeds_used'])}", "",
             "## Overall results (mean over seeds and levels)", "", overall.to_markdown(index=False), "",
             f"**Best RL method by average waiting time:** {bst} "
             f"({rl_only.loc[bst, 'avg_wait_s_mean']:.2f} s/veh).", "",
             "## Per-level results", "", g.to_markdown(index=False), "",
             "## Ablation (proposed D3QN-PER minus one component)", "", abl.to_markdown(index=False), "",
             "## Significance (Welch t-test on per-seed avg waiting time)", "", sig.to_markdown(index=False), "",
             "## Best method per level and metric", "", best.to_markdown(index=False), ""]
    open(f"{out}/REPORT.md", "w").write("\n".join(lines))
    return overall


# ----------------------------------------------------------------------------------------
# 7. MAIN
# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_project6")
    ap.add_argument("--quick", action="store_true", help="smoke test: 60 episodes, 2 seeds, 100 tests/level")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--methods", nargs="*", default=None, help="subset of methods")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="skip training: merge raw_test_results/training_curves of these result folders "
                         "(later folders override earlier ones for the same method) and re-run analysis into --out")
    a = ap.parse_args()
    if a.merge:
        raws, cvs = {}, {}
        for d in a.merge:
            r = pd.read_csv(f"{d}/raw_test_results.csv"); c = pd.read_csv(f"{d}/training_curves.csv")
            for m in r.method.unique(): raws[m] = r[r.method == m]
            for m in c.method.unique(): cvs[m] = c[c.method == m]
        raw = pd.concat(raws.values(), ignore_index=True); curves = pd.concat(cvs.values(), ignore_index=True)
        cfgf = json.load(open(f"{a.merge[0]}/config_used.json"))["cfg"]
        CFG["train_episodes_used"], CFG["seeds_used"] = cfgf.get("train_episodes_used"), cfgf.get("seeds_used")
        os.makedirs(a.out, exist_ok=True)
        raw.to_csv(f"{a.out}/raw_test_results.csv", index=False); curves.to_csv(f"{a.out}/training_curves.csv", index=False)
        analyse(raw, curves, a.out); print(f"Merged results saved in {a.out}/"); return

    episodes = a.episodes or (60 if a.quick else CFG["train_episodes"])
    seeds = CFG["seeds"][: (a.seeds or (2 if a.quick else 5))]
    if a.quick: CFG["n_test_per_level"] = 100
    CFG["train_episodes_used"], CFG["seeds_used"] = episodes, seeds
    os.makedirs(f"{a.out}/data", exist_ok=True)

    # datasets
    train_df = generate_scenarios(CFG["n_train_records"], seed=2024)
    test_df = generate_scenarios(3 * CFG["n_test_per_level"], seed=777)      # disjoint seed => unseen
    train_df.to_csv(f"{a.out}/data/train_scenarios_10000.csv", index=False)
    test_df.to_csv(f"{a.out}/data/test_scenarios_unseen.csv", index=False)
    print(f"Datasets: train={len(train_df)} test={len(test_df)}  episodes={episodes} seeds={seeds}")

    methods = a.methods or ALL_METHODS
    jobs = [(m, s, train_df, test_df, episodes) for m in methods for s in (seeds if m not in HEURISTICS else seeds[:1])]
    results, curves = [], []
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            for rows, cv in ex.map(run_job, jobs): results += rows; curves.append(cv)
    else:
        for j in jobs:
            rows, cv = run_job(j); results += rows; curves.append(cv)

    raw = pd.DataFrame(results)
    # heuristics are deterministic wrt policy -> replicate over seeds' noise streams for fair comparison
    extra = []
    for m in [x for x in methods if x in HEURISTICS]:
        for s in seeds[1:]:
            pol = FixedTime() if m == "FixedTime" else MaxPressure()
            for r in evaluate(pol, test_df, s): r.update(method=m, seed=s, train_time_s=0.0); extra.append(r)
    raw = pd.concat([raw, pd.DataFrame(extra)], ignore_index=True)
    curves = pd.concat([c for c in curves if len(c)], ignore_index=True) if any(len(c) for c in curves) else pd.DataFrame(
        columns=["episode", "ep_reward", "ep_avg_wait", "epsilon", "method", "seed"])
    raw.to_csv(f"{a.out}/raw_test_results.csv", index=False)
    curves.to_csv(f"{a.out}/training_curves.csv", index=False)
    overall = analyse(raw, curves, a.out)
    print("\n=== OVERALL (mean over levels & seeds) ===")
    print(overall[["method", "avg_wait_s_mean", "avg_queue_mean", "veh_served_per_hr_mean", "throughput_ratio_mean"]].to_string(index=False))
    json.dump(dict(cfg={k: v for k, v in CFG.items()}, methods=methods), open(f"{a.out}/config_used.json", "w"), indent=2, default=str)
    print(f"\nAll outputs saved in: {a.out}/")


if __name__ == "__main__":
    main()
