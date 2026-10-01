"""Backtest engine and performance metrics (PLAN.md §9.1–9.2).

Every strategy hands the engine *target* weights indexed by holding day t (decided at the close
of t−1, held to the close of t). The engine turns targets into positions under the halt and
delisting rules of §7.2, charges costs and returns daily P&L:

* an asset that wasn't tradable at the close of t−1 keeps its previous position (frozen);
* returns across a halt are booked on the first valid day after it (built into the panel);
* after a permanent end (no further valid close, or a gap > ``gap_max_days``) a position earns 0
  and is removed at no cost from day t_last+2 — a cash-out at the last close;
* cost_t = tc·Σ|pos_t − pos_{t−1}| + short·Σ max(−pos_t, 0), the paper's formula (no drift).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from statarb.data.panel import load_panel, wide
from statarb.data.yahoo import load_prices, master_calendar
from statarb.paths import Paths

ANNUAL = 252


@dataclass
class Market:
    """Everything the engine needs, as dates × assets matrices on the master calendar."""

    calendar: pd.DatetimeIndex
    ret: pd.DataFrame       # total return, NaN when not valid
    valid: pd.DataFrame     # tradable at that close
    spy: pd.Series          # SPY daily total return (for beta)


def load_market(cfg: dict, paths: Paths) -> Market:
    calendar = master_calendar(cfg, paths)
    panel = load_panel(paths, ["ret", "valid"])
    ret = wide(panel, "ret").reindex(calendar)
    valid = wide(panel, "valid").reindex(calendar).eq(True)
    spy_px = load_prices(paths, "SPY")["Adj Close"].reindex(calendar)
    return Market(calendar, ret, valid, spy_px / spy_px.shift(1) - 1)


def dead_mask(valid: np.ndarray, gap_max_days: int) -> np.ndarray:
    """True on days t ≥ t_last+2 inside a permanent end (series stops, or a gap > gap_max_days)."""
    T, A = valid.shape
    dead = np.zeros((T, A), dtype=bool)
    for a in range(A):
        idx = np.flatnonzero(valid[:, a])
        if len(idx) == 0:
            continue
        nxt = np.append(idx[1:], T)  # next valid index (T = never)
        ends = np.flatnonzero((nxt - idx - 1 > gap_max_days) | ((nxt == T) & (idx < T - 1)))
        for e in ends:
            dead[idx[e] + 2: nxt[e], a] = True
    return dead


def run_backtest(
    weights: pd.DataFrame,
    market: Market,
    tc_bps: float = 5.0,
    short_bps: float = 1.0,
    exec_lag: int = 0,
    gap_max_days: int = 60,
    return_positions: bool = False,
):
    """Daily results for target ``weights`` (holding days × assets; missing = 0)."""
    days = weights.index
    cal_idx = market.calendar.get_indexer(days)
    if (cal_idx < 1).any():
        raise ValueError("every holding day needs a previous trading day on the calendar")
    cols = market.ret.columns
    W = weights.reindex(columns=cols).fillna(0.0).to_numpy(dtype=float)
    if exec_lag:
        W = np.vstack([np.zeros((exec_lag, W.shape[1])), W[:-exec_lag]])
    R = market.ret.to_numpy(dtype=float)
    V = market.valid.to_numpy(dtype=bool)
    D = dead_mask(V, gap_max_days)

    out = np.zeros((len(days), 7))
    positions = np.zeros_like(W) if return_positions else None
    prev = np.zeros(len(cols))
    for k, i in enumerate(cal_idx):
        pos = np.where(V[i - 1], W[k], prev)
        removed = D[i] & (pos != 0)
        pos[removed] = 0.0
        change = np.abs(pos - prev)
        change[removed] = 0.0
        gross = float(pos @ np.nan_to_num(R[i]))
        turnover = change.sum()
        short = np.clip(-pos, 0, None).sum()
        cost = tc_bps / 1e4 * turnover + short_bps / 1e4 * short
        out[k] = (gross, cost, gross - cost, turnover, short, np.abs(pos).sum(), pos.sum())
        if positions is not None:
            positions[k] = pos
        prev = pos

    result = pd.DataFrame(out, index=days,
                          columns=["gross", "cost", "net", "turnover", "short", "gross_exposure", "net_exposure"])
    if return_positions:
        return result, pd.DataFrame(positions, index=days, columns=cols)
    return result


# ----------------------------------------------------------------------------- metrics

def sharpe(x: np.ndarray | pd.Series) -> float:
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(ANNUAL)) if sd > 0 else 0.0


def sharpe_se(x) -> float:
    """Lo (2002) standard error of the annualized Sharpe ratio, iid approximation."""
    x = np.asarray(x, dtype=float)
    sr_d = sharpe(x) / np.sqrt(ANNUAL)
    return float(np.sqrt(ANNUAL) * np.sqrt((1 + 0.5 * sr_d ** 2) / len(x)))


def stationary_bootstrap_indices(T: int, n: int, block: int, seed: int = 0) -> np.ndarray:
    """Politis–Romano stationary bootstrap: ``n`` index paths of length ``T``, mean block ``block``."""
    rng = np.random.default_rng(seed)
    idx = np.empty((n, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, n)
    restart = rng.random((n, T)) < 1.0 / block
    fresh = rng.integers(0, T, (n, T))
    for t in range(1, T):
        idx[:, t] = np.where(restart[:, t], fresh[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def _sharpe_rows(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=1, ddof=1)
    return np.where(sd > 0, x.mean(axis=1) / np.where(sd > 0, sd, 1) * np.sqrt(ANNUAL), 0.0)


def bootstrap_sharpe_ci(x, n: int = 2000, block: int = 21, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    draws = _sharpe_rows(x[stationary_bootstrap_indices(len(x), n, block, seed)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def bootstrap_sharpe_diff(a, b, n: int = 2000, block: int = 21, seed: int = 0) -> tuple[float, float, float]:
    """Sharpe(a) − Sharpe(b) with a 95% CI; both series resampled on the same days."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    idx = stationary_bootstrap_indices(len(a), n, block, seed)
    diff = _sharpe_rows(a[idx]) - _sharpe_rows(b[idx])
    return sharpe(a) - sharpe(b), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def max_drawdown(net) -> float:
    wealth = np.cumprod(1 + np.asarray(net, dtype=float))
    return float((wealth / np.maximum.accumulate(wealth) - 1).min())


def market_beta(net: pd.Series, spy: pd.Series, lags: int = 5) -> tuple[float, float]:
    """OLS beta of daily returns on SPY with a Newey–West t-statistic."""
    import statsmodels.api as sm

    both = pd.concat([net, spy], axis=1, keys=["y", "m"]).dropna()
    fit = sm.OLS(both["y"], sm.add_constant(both["m"])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.params["m"]), float(fit.tvalues["m"])


def per_year_sharpe(series: pd.Series) -> pd.Series:
    return series.groupby(series.index.year).apply(sharpe)


def summarize(result: pd.DataFrame, spy: pd.Series, bcfg: dict) -> dict:
    """The metric row reported for every strategy (PLAN.md §9.2)."""
    net, gross = result["net"], result["gross"]
    lo, hi = bootstrap_sharpe_ci(net, bcfg["bootstrap_n"], bcfg["bootstrap_block"])
    beta, beta_t = market_beta(net, spy.reindex(net.index), bcfg["nw_lags"])
    mdd = max_drawdown(net)
    ann = net.mean() * ANNUAL
    return {
        "start": net.index[0].date(), "end": net.index[-1].date(), "days": len(net),
        "sr_gross": sharpe(gross), "sr_net": sharpe(net), "sr_net_se": sharpe_se(net),
        "sr_net_ci_lo": lo, "sr_net_ci_hi": hi,
        "mu_gross": gross.mean() * ANNUAL, "mu_net": ann, "vol": net.std(ddof=1) * np.sqrt(ANNUAL),
        "max_dd": mdd, "calmar": ann / abs(mdd) if mdd < 0 else np.nan, "hit_rate": float((net > 0).mean()),
        "turnover": result["turnover"].mean(), "gross_exposure": result["gross_exposure"].mean(),
        "net_exposure": result["net_exposure"].mean(), "short_exposure": result["short"].mean(),
        "beta": beta, "beta_t": beta_t,
    }
