# Attention-Based Statistical Arbitrage

A simplified reproduction of Epstein, Wang, Choi & Pelger, *Attention Factors for Statistical Arbitrage* ([arXiv:2510.11616](https://arxiv.org/abs/2510.11616)), built on free data (yfinance + Tiingo).

- **Stage 1:** a cross-sectional attention model over price/volume characteristics learns latent factor portfolios.
- **Stage 2:** an LSTM over each stock's 30-day factor residuals outputs residual-portfolio weights.
- **Training:** both stages are trained jointly on net-of-cost Sharpe.
- **Evaluation:** walk-forward out-of-sample backtest 2019–2026 on the 200 most liquid point-in-time S&P 500 members, delisted names included. Benchmarked against cointegration pairs trading and a two-step PCA model.

**Status:** under construction. The build spec is [PLAN.md](PLAN.md); results will appear here when the build is finished.

## Setup

```bash
make setup                 # creates .venv and installs pinned dependencies
cp .env.example .env       # then add your Tiingo API token (needed from Phase 1 on)
make test
```
