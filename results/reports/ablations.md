# Ablations (post-freeze, OOS 2019-01 → 2026-08, frozen hyperparameters, 5 seeds)

Ablations never replace the headline (PLAN.md §14). LongConv runs with the hyperparameters tuned for the LSTM.

| ablation | net SR | 95% CI | gross SR | turnover/day | beta | seed SR mean |
|---|---|---|---|---|---|---|
| headline: LSTM, 15 features, K=5 | -0.37 | [-1.11, +0.34] | 0.26 | 0.24 | 0.16 | -0.46 |
| LongConv instead of LSTM (paper's sequence model) | -1.54 | [-2.29, -0.81] | -0.03 | 0.74 | 0.12 | -1.47 |
| drop past-return features | -0.44 | [-1.12, +0.18] | 0.12 | 0.30 | 0.14 | -0.50 |
| drop risk features | -0.16 | [-0.84, +0.47] | 0.47 | 0.30 | 0.15 | -0.27 |
| drop trading-friction features | -0.25 | [-1.00, +0.48] | 0.44 | 0.32 | 0.17 | -0.40 |
| K = 10 factors | -0.63 | [-1.37, +0.12] | 0.13 | 0.43 | 0.11 | -0.76 |
| K = 20 factors | -1.17 | [-1.80, -0.52] | 0.06 | 0.55 | 0.03 | -1.18 |
