# Attention-Based Statistical Arbitrage

A simplified, honest reproduction of Epstein, Wang, Choi & Pelger, *Attention Factors for Statistical Arbitrage* ([arXiv:2510.11616](https://arxiv.org/abs/2510.11616), ICAIF '25), built on free data.

- **Stage 1:** cross-sectional attention. K learned factor queries attend over 15 price/volume characteristics of each stock; the attention weights are the factor portfolios.
- **Stage 2:** an LSTM reads each stock's recent factor residuals and outputs residual-portfolio weights.
- **Joint training:** both stages are trained together on net-of-cost Sharpe plus explained variance.
- **Evaluation:** a walk-forward out-of-sample backtest from 2019-01-02 to 2026-08-31, on the 200 most liquid point-in-time S&P 500 members, delisted names included.

**Full results:** [results/reports/final_report.md](results/reports/final_report.md) · bias checks: [bias_checks.md](results/reports/bias_checks.md) · build spec: [PLAN.md](PLAN.md) · every judgment call: [DECISIONS_LOG.md](DECISIONS_LOG.md)

## Headline (out-of-sample, 1,927 trading days)

| Strategy | Net Sharpe [95% CI] | Gross Sharpe | Turnover/day | Market β |
|---|---|---|---|---|
| **Attention + LSTM** (5-seed ensemble) | **−0.37** [−1.11, +0.34] | 0.26 | 0.24 | +0.16 |
| PCA + LSTM (two-step benchmark) | −0.11 [−0.74, +0.49] | 0.33 | 0.30 | +0.23 |
| Cointegration pairs trading | −0.18 [−0.89, +0.55] | 0.23 | 0.03 | +0.01 |
| 5-day reversal | −0.36 [−1.13, +0.36] | 0.38 | 0.63 | +0.14 |
| SPY buy-and-hold (reference) | 0.93 | | | |

Costs follow the paper: 5 bps per unit traded plus 1 bp/day on short positions.

- **Costs erase the edge.** Every strategy earns a small gross edge and loses money net. With 7.7 years of data the Sharpe standard error is ≈ 0.36, so none of the strategies is distinguishable from zero or from the others.
- **The paper's central claim does not replicate in this setting.** Joint attention factors do not beat the two-step PCA benchmark: the difference is −0.26, CI [−1.12, +0.53].
- **The model worked early, then broke.**
  - The attention model was positive in 2019–2021 and negative every year after.
  - Its five learned factors collapsed to nearly the same market-like portfolio (mean pairwise cosine similarity 0.94), unlike the paper's industry factors.
  - It kept a significant market β of +0.16, so it is *not* market-neutral.
- **Survivorship bias, quantified.** The same frozen model on today's S&P 500 constituents scores −0.02 instead of −0.37, a +0.35 overstatement.
- **No look-ahead.** Scrambling all data after a date leaves the trained models' earlier weights bit-identical; pure-noise data gives nothing (mean Sharpe −0.11); a planted signal is learned (Sharpe 16).

## What makes the data pipeline trustworthy

- **Point-in-time membership:** S&P 500 membership from [fja05680/sp500](https://github.com/fja05680/sp500), pinned to a commit. All 871 membership spells since 2010 are resolved to a price series or explicitly marked unrecoverable; 99.9% of 2019+ member-days have data.
- **Delisted names:** these come from Tiingo, since Yahoo has no data for delisted stocks.
  - Reused tickers (for example `STI`, `APC`, `WRK`) are fetched by Tiingo permaTicker, because the ticker alone returns the current owner.
  - Renames are handled by a verified alias table (FB→META, ANTM→ELV, EQR→VMRK, …).
- **Halts:** halts are not tradable, and a stale-price rule stops a strategy from "selling" SVB at its frozen $106.
- **Frozen before testing:** the configuration was frozen on 2017–2018 validation (git tag `frozen-v1`) before any 2019+ data was used. Every out-of-sample run is logged with its commit in [results/oos_runs.csv](results/oos_runs.csv).

## Reproduce

```bash
make setup                 # venv in ~/.venvs (the repo drive is exFAT) + pinned deps
cp .env.example .env       # add a free Tiingo API token
make data features tensors # downloads (Tiingo ~4 h, rate-limited and resumable) and preprocessing
make test-all              # 80 tests, including slow planted-signal controls
make tune                  # 2011–2016 train / 2017–2018 validation
make oos checks ablations report
```

`configs/frozen.yaml` is committed, so `make oos` reproduces the out-of-sample runs without re-freezing. Raw data stays local (gitignored): Yahoo and Tiingo data are for personal use only.

## Deviations from the paper

| Paper | This build | Why |
|---|---|---|
| CRSP, 500 largest by market cap, 1998–2021 | yfinance + Tiingo, top 200 by dollar volume, 2019–2026 OOS | free data |
| 39 Chen–Pelger–Zhu characteristics | 15 price/volume characteristics | no Compustat; the paper finds price-based features drive results |
| LongConv sequence model | LSTM (LongConv run as an ablation) | design choice; the paper calls the sequence model interchangeable |
| K = 30, λ_Var = 100 | K = 5, λ_Var = 1 (chosen on validation) | smaller cross-section; λ_Var = 100 validated worst |

The full list is in [PLAN.md §2.2](PLAN.md) and [DECISIONS_LOG.md](DECISIONS_LOG.md).

## Layout

```
src/statarb/data/        membership, Yahoo/Tiingo downloads, ticker resolution, panel, universe, coverage
src/statarb/features.py  15 causal characteristics + rank normalization
src/statarb/tensors.py   holding-day tensors (inputs dated ≤ t−1)
src/statarb/models/      attention factors, LSTM, LongConv, joint model
src/statarb/train.py     joint Sharpe + explained-variance training
src/statarb/backtest.py  engine (halts, delistings, costs) and metrics
src/statarb/baselines/   reversal, cointegration pairs trading
src/statarb/walkforward.py, bias_checks.py, report.py
```
