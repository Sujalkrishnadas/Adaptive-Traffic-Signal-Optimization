# Adaptive Traffic Signal Optimization using Reinforcement Learning

RL Mini Research Project 6 (Smart Cities). A simulation-based study of adaptive signal control at a single four-way intersection, comparing tabular and deep RL algorithms against fixed-time and max-pressure controllers.

> **Validity note:** all results are simulation-based findings. They are not proof of real-world performance without experimental validation.

## What is in this repo

| Path | Description |
|---|---|
| `rl_traffic_signal_project.py` | Complete code: dataset generation, simulator, agents, training, evaluation, analysis |
| `requirements.txt` | Python dependencies |
| `results_final/data/train_scenarios_10000.csv` | 10,000 synthetic training scenario records |
| `results_final/data/test_scenarios_unseen.csv` | 3,000 unseen test scenarios (1,000 per difficulty level) |
| `results_final/SUMMARY_SHEET.xlsx` | Summary sheet (protocol, results, ablation, significance tests) |
| `results_final/REPORT.md` | Auto-generated result tables |
| `results_final/*.csv`, `fig_*.png` | Raw results, summaries, training curves and figures |

## Problem setup

- **Environment:** one intersection with four approaches (N, S, E, W) and Poisson arrivals.
- **Actions (6):** choose the phase (NS or EW) and the green time (10, 20 or 30 s). Switching phase costs 5 s of lost time.
- **Reward:** negative total vehicle waiting time during the step.
- **Difficulty levels:** Low, Medium, High (demand relative to capacity).

## Methods

- **RL algorithms:** Q-Learning, SARSA, DQN, DDQN, Dueling DQN, PPO, A2C
- **RL baselines:** Q-Learning, DQN
- **Proposed:** Dueling Double DQN with Prioritized Experience Replay (D3QN-PER)
- **Ablations:** the proposed method without Dueling, without Double, or without PER
- **Non-learning references:** Fixed-Time, Max-Pressure

## Experimental protocol

1,000 training episodes, 5 random seeds, 1,000 unseen test scenarios per level, 3 difficulty levels. Metrics: average waiting time, average queue length, vehicles served per hour, throughput (served/arrived), plus phase switches.

## Results (mean over seeds and difficulty levels)

| Method | Avg wait (s/veh) | Avg queue | Veh served/hr | Throughput |
|---|---|---|---|---|
| Abl_noPER | 46.3 | 9.42 | 1963 | 0.981 |
| Abl_noDueling | 51.3 | 10.66 | 1952 | 0.978 |
| DuelingDQN | 51.5 | 10.69 | 1946 | 0.976 |
| D3QN_PER (proposed) | 52.3 | 10.81 | 1947 | 0.976 |
| PPO | 53.8 | 10.95 | 1954 | 0.978 |
| Abl_noDouble | 54.0 | 11.24 | 1935 | 0.973 |
| DQN | 54.1 | 11.06 | 1941 | 0.974 |
| DDQN | 55.1 | 11.35 | 1935 | 0.972 |
| MaxPressure | 55.4 | 11.59 | 1945 | 0.976 |
| A2C | 76.0 | 15.80 | 1909 | 0.964 |
| QLearning | 80.4 | 16.43 | 1893 | 0.957 |
| SARSA | 88.3 | 18.32 | 1874 | 0.952 |
| FixedTime | 118.3 | 23.56 | 1848 | 0.941 |

Notes:
- The deep value-based methods and PPO roughly halve the waiting time of Fixed-Time.
- In this setup the proposed D3QN-PER did **not** outperform its no-PER ablation, and differences within the 46-55 s group are small. See `results_final/significance_tests.csv` before claiming any method is better.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python rl_traffic_signal_project.py --quick       # smoke test (minutes)
python rl_traffic_signal_project.py --jobs 4      # full protocol (hours)
```

Useful options: `--out DIR`, `--methods M1 M2 ...`, `--episodes N`, `--seeds N`, and `--merge DIR1 DIR2 --out DIR` to merge results from separate runs.

Outputs are written to `results_project6/` by default (CSV files, figures, `SUMMARY_SHEET.xlsx`, `REPORT.md`). The code runs on CPU.

## Author

Sujalkrishnadas
