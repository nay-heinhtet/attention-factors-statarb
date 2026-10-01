"""Classical cointegration pairs trading (PLAN.md §9.4).

Every January and July: test all same-sector pairs in the universe for Engle–Granger
cointegration on the previous 252 days of log prices, keep the best ``max_pairs`` with sensible
half-lives and hedge ratios, then trade each pair's spread z-score for six months with fixed
formation parameters. Decisions at the close of t−1 set the position for day t, and the
weights go through the same engine and costs as every other strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class PairFit:
    y: str
    x: str
    alpha: float
    beta: float
    mu: float
    sigma: float
    pvalue: float
    halflife: float


def fit_pair(log_y: np.ndarray, log_x: np.ndarray) -> PairFit | None:
    """Engle–Granger on log prices (``y`` regressed on ``x``), hedge ratio, spread stats, half-life."""
    from statsmodels.tsa.stattools import coint

    pvalue = float(coint(log_y, log_x, trend="c", autolag="aic")[1])
    beta, alpha = np.polyfit(log_x, log_y, 1)
    spread = log_y - alpha - beta * log_x
    phi = np.polyfit(spread[:-1], np.diff(spread), 1)[0]
    halflife = -np.log(2) / np.log(1 + phi) if -1 < phi < 0 else np.inf
    return PairFit("", "", float(alpha), float(beta), float(spread.mean()), float(spread.std(ddof=1)), pvalue, float(halflife))


def formation(log_prices: pd.DataFrame, sectors: pd.Series, pcfg: dict) -> list[PairFit]:
    """Candidate pairs (both orientations; the lower p-value wins) that pass the filters, best first."""
    fits = []
    for sector, members in sectors.groupby(sectors):
        if sector == "Unknown":
            continue
        for a, b in combinations(sorted(members.index), 2):
            ya, yb = log_prices[a].to_numpy(), log_prices[b].to_numpy()
            fab, fba = fit_pair(ya, yb), fit_pair(yb, ya)
            best, (y, x) = (fab, (a, b)) if fab.pvalue <= fba.pvalue else (fba, (b, a))
            best.y, best.x = y, x
            lo_hl, hi_hl = pcfg["halflife"]
            lo_b, hi_b = pcfg["beta"]
            if best.pvalue < pcfg["pvalue_max"] and lo_hl <= best.halflife <= hi_hl and lo_b <= best.beta <= hi_b:
                fits.append(best)
    return sorted(fits, key=lambda f: f.pvalue)


def select(fits: list[PairFit], max_pairs: int, per_asset: int) -> list[PairFit]:
    chosen, uses = [], {}
    for f in fits:
        if uses.get(f.y, 0) < per_asset and uses.get(f.x, 0) < per_asset:
            chosen.append(f)
            uses[f.y] = uses.get(f.y, 0) + 1
            uses[f.x] = uses.get(f.x, 0) + 1
        if len(chosen) == max_pairs:
            break
    return chosen


def pair_states(z: np.ndarray, tradable: np.ndarray, entry: float, exit_: float, stop: float) -> np.ndarray:
    """Spread position after each close: +1 long spread, −1 short spread, 0 flat.

    Long spread opens at z ≤ −entry and closes once z ≥ −exit (reverted) or z ≤ −stop (stop-loss);
    short spread mirrors it. A stop-loss disables the pair; so does a non-tradable leg.
    """
    states = np.zeros(len(z))
    state, disabled = 0, False
    for t, (zt, ok) in enumerate(zip(z, tradable)):
        if disabled or not ok or np.isnan(zt):
            state = 0
            disabled = disabled or not ok
        elif state == 0:
            state = -1 if zt >= entry else (1 if zt <= -entry else 0)
        elif state == 1:
            if zt <= -stop:
                state, disabled = 0, True
            elif zt >= -exit_:
                state = 0
        elif state == -1:
            if zt >= stop:
                state, disabled = 0, True
            elif zt <= exit_:
                state = 0
        states[t] = state
    return states


def pairs_weights(
    close_adj: pd.DataFrame,
    valid: pd.DataFrame,
    universe: pd.DataFrame,
    sectors: pd.Series,
    start,
    end,
    pcfg: dict,
    cache: dict | None = None,
) -> pd.DataFrame:
    """Target weights for holding days in [start, end]. ``cache`` reuses formation results."""
    calendar = close_adj.index
    log_px = np.log(close_adj.where(valid.eq(True)).ffill(limit=5))
    days = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    form_days = [d for d in days if d.month in (1, 7) and (d == days[0] or days[days.get_loc(d) - 1].month != d.month)]
    if days[0] not in form_days:  # start mid-period: begin with the most recent formation
        form_days.insert(0, days[0])
    weights = pd.DataFrame(0.0, index=days, columns=close_adj.columns)
    g = 1.0 / pcfg["max_pairs"]

    for k, f_day in enumerate(form_days):
        period = days[(days >= f_day) & (days < (form_days[k + 1] if k + 1 < len(form_days) else days[-1] + pd.Timedelta(days=1)))]
        pos = calendar.get_loc(f_day)
        window = log_px.iloc[pos - pcfg["formation_days"]: pos]
        month_members = universe.loc[universe["month"] == f_day.to_period("M").to_timestamp(), "asset_id"]
        month_members = [a for a in month_members if a in valid.columns]
        ok = valid.iloc[pos - pcfg["formation_days"]: pos][month_members].eq(True).sum() >= pcfg["min_valid_days"]
        names = ok.index[ok]
        key = f_day
        if cache is not None and key in cache:
            fits = cache[key]
        else:
            fits = formation(window[names].ffill().bfill(), sectors.reindex(names).fillna("Unknown"), pcfg)
            if cache is not None:
                cache[key] = fits
        chosen = select(fits, pcfg["max_pairs"], pcfg["max_pairs_per_asset"])

        prev_days = calendar[calendar.get_indexer(period) - 1]  # decision closes t−1
        for fit in chosen:
            s = log_px.loc[prev_days, fit.y].to_numpy() - fit.alpha - fit.beta * log_px.loc[prev_days, fit.x].to_numpy()
            z = (s - fit.mu) / fit.sigma
            tradable = valid.loc[prev_days, [fit.y, fit.x]].eq(True).all(axis=1).to_numpy()
            state = pair_states(z, tradable, pcfg["entry_z"], pcfg["exit_z"], pcfg["stop_z"])
            weights.loc[period, fit.y] += state * g / (1 + fit.beta)
            weights.loc[period, fit.x] -= state * g * fit.beta / (1 + fit.beta)
        log.info("pairs %s: %d candidates pass filters, %d traded", f_day.date(), len(fits), len(chosen))
    return weights
