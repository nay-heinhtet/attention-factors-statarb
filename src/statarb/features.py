"""The 15 price/volume characteristics (PLAN.md §8), computed causally on the master calendar.

Every value at date t uses only data dated ≤ t: rolling windows are trailing, nothing is centered,
back-filled or normalized with full-sample statistics. ``tests/test_features.py`` checks this by
recomputing on truncated data. Cross-sectional rank normalization happens later, per holding day
and within that day's universe (:func:`rank_normalize`, used by tensors.py).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from statarb.data.panel import load_panel, wide
from statarb.data.yahoo import load_prices, master_calendar
from statarb.paths import project_paths

log = logging.getLogger(__name__)


def feature_names(cfg: dict, apply_drops: bool = True) -> list[str]:
    """Characteristic names in config order; ``drop_groups`` (ablations) are removed by default."""
    drops = set(cfg["features"]["drop_groups"]) if apply_drops else set()
    return [name for group, names in cfg["features"]["groups"].items() if group not in drops for name in names]


def compute_features(
    close_adj: pd.DataFrame,
    valid: pd.DataFrame,
    ret: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    market_ret: pd.Series,
    fcfg: dict,
) -> dict[str, pd.DataFrame]:
    """All characteristics as dates × assets frames. ``market_ret`` is SPY's daily return."""
    frac = fcfg["min_periods_frac"]

    def roll(frame: pd.DataFrame, window: int):
        return frame.rolling(window, min_periods=math.ceil(frac * window))

    valid = valid.eq(True)
    price = close_adj.where(valid).ffill(limit=fcfg["ffill_days"])
    r = ret.where(valid)
    dv = dollar_volume.where(valid)
    m = pd.DataFrame(np.repeat(market_ret.to_numpy()[:, None], r.shape[1], axis=1), index=r.index, columns=r.columns)
    m = m.where(r.notna())  # pairwise-complete moments for beta

    out = {
        "ret_1d": r,
        "ret_5d": price / price.shift(5) - 1,
        "ret_21d": price / price.shift(21) - 1,
        "mom_12_2": price.shift(21) / price.shift(252) - 1,
        "mom_12_7": price.shift(126) / price.shift(252) - 1,
        "mom_36_13": price.shift(252) / price.shift(756) - 1,
        "rel_52w_high": price / roll(price, 252).max(),
        "vol_5d": r.rolling(5, min_periods=4).std(),
        "vol_63d": roll(r, 63).std(),
    }
    w = roll(r, 252)
    mean_r, mean_m = w.mean(), roll(m, 252).mean()
    cov = roll(r * m, 252).mean() - mean_r * mean_m
    var = roll(m * m, 252).mean() - mean_m ** 2
    beta = cov / var.where(var > 0)
    out["beta_252d"] = beta
    out["ivol_63d"] = roll(r - beta * m, 63).std()
    out["dvol_21d"] = roll(dv, 21).mean()
    out["dvol_ratio"] = dv.rolling(5, min_periods=4).mean() / roll(dv, 252).mean()
    hl = np.log(high / low).where(valid & high.gt(0) & low.gt(0))
    out["hl_range_21d"] = roll(hl, 21).mean()
    out["amihud_21d"] = roll(r.abs() / dv, 21).mean()
    return {name: f.ffill(limit=fcfg["ffill_days"]) for name, f in out.items()}


def rank_normalize(values: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional ranks mapped to [−0.5, 0.5]; missing values → 0, the median (PLAN.md §8).

    ``values`` holds one day's raw characteristics for that day's universe (rows = assets).
    """
    n = values.notna().sum()
    ranks = values.rank(method="average")
    scaled = (ranks - 1).div((n - 1).where(n > 1)) - 0.5
    return scaled.fillna(0.0)


def to_long(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    stacked = pd.concat({name: f.stack() for name, f in features.items()}, axis=1)
    stacked.index.names = ["date", "asset_id"]
    return stacked.dropna(how="all").astype("float32").reset_index()


def run(cfg: dict, args) -> None:
    paths = project_paths(cfg)
    calendar = master_calendar(cfg, paths)
    fields = ["close_adj", "valid", "ret", "high", "low", "dollar_volume"]
    panel = load_panel(paths, fields)
    frames = {f: wide(panel, f).reindex(calendar) for f in fields}
    spy = load_prices(paths, "SPY")["Adj Close"].reindex(calendar)
    market_ret = spy / spy.shift(1) - 1

    features = compute_features(**frames, market_ret=market_ret, fcfg=cfg["features"])
    long = to_long({name: features[name] for name in feature_names(cfg, apply_drops=False)})
    long.to_parquet(paths.processed / "features.parquet", index=False)
    coverage = long.drop(columns=["date", "asset_id"]).notna().mean().round(3)
    log.info("features: %d rows (%d assets); non-missing share per feature:\n%s",
             len(long), long["asset_id"].nunique(), coverage.to_string())
