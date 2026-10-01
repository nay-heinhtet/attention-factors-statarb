"""Monthly point-in-time universe: the 200 most liquid eligible S&P 500 members (PLAN.md §7.3).

The universe for month m is formed on f_m, the last trading day of month m−1, using only data
dated ≤ f_m. ``--kind survivor`` instead uses today's constituents (Yahoo only) for every month,
which reproduces the survivorship-biased universe a naive yfinance pipeline would use.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from statarb.data.membership import load_snapshots, members_on, membership_path
from statarb.data.panel import load_panel, wide
from statarb.data.yahoo import master_calendar
from statarb.paths import project_paths

log = logging.getLogger(__name__)


def formation_dates(calendar: pd.DatetimeIndex, first_month: str, last_month: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(month start, formation date = last trading day before that month) for each month."""
    out = []
    for month in pd.date_range(first_month, last_month, freq="MS"):
        prior = calendar[calendar < month]
        out.append((month, prior[-1]))
    return out


def form_universe(
    valid: pd.DataFrame,
    close_raw: pd.DataFrame,
    ret: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    members: Callable[[pd.Timestamp], set[str]],
    formations: list[tuple[pd.Timestamp, pd.Timestamp]],
    drop: set[str],
    ucfg: dict,
) -> pd.DataFrame:
    """Top-``size`` eligible members per month by median dollar volume (all inputs dates × assets)."""
    valid_returns = ret.notna().rolling(ucfg["history_days"], min_periods=1).sum()
    dv = dollar_volume.rolling(ucfg["dv_window"], min_periods=ucfg["dv_min_obs"]).median()
    rows = []
    for month, f in formations:
        cand = sorted((members(f) - drop) & set(valid.columns))
        ok = (
            valid.loc[f, cand].eq(True)
            & close_raw.loc[f, cand].ge(ucfg["min_price"])
            & valid_returns.loc[f, cand].ge(ucfg["history_min_valid"])
            & dv.loc[f, cand].notna()
        )
        eligible = pd.DataFrame({"asset_id": ok.index[ok], "dv63": dv.loc[f, ok.index[ok]].to_numpy()})
        top = eligible.sort_values(["dv63", "asset_id"], ascending=[False, True]).head(ucfg["size"])
        if len(top) < ucfg["size"]:
            log.warning("%s: only %d eligible assets", month.strftime("%Y-%m"), len(top))
        rows.append(top.assign(month=month, formation_date=f, rank=np.arange(1, len(top) + 1)))
    return pd.concat(rows, ignore_index=True)[["month", "formation_date", "asset_id", "rank", "dv63"]]


def load_universe(paths, kind: str = "pit") -> pd.DataFrame:
    return pd.read_parquet(paths.processed / f"universe_{kind}.parquet")


def daily_universe(universe: pd.DataFrame, valid: pd.DataFrame, start, end) -> pd.DataFrame:
    """U_t for each holding day t in [start, end]: month m's universe ∩ assets valid at close t−1.

    ``valid`` is the dates × assets validity matrix on the master calendar.
    """
    days = valid.index[(valid.index >= pd.Timestamp(start)) & (valid.index <= pd.Timestamp(end))]
    member = pd.DataFrame(False, index=days, columns=valid.columns)
    months = days.to_period("M")
    for month, group in universe.groupby(universe["month"].dt.to_period("M")):
        cols = [a for a in group["asset_id"] if a in member.columns]
        member.loc[months == month, cols] = True
    prev_valid = valid.shift(1).reindex(days).eq(True)
    return member & prev_valid


def duplicate_pairs(universe: pd.DataFrame, ret: pd.DataFrame, window: int, threshold: float) -> pd.DataFrame:
    """Pairs in the same month's universe whose trailing return correlation exceeds ``threshold``."""
    found = {}
    for (month, f), group in universe.groupby(["month", "formation_date"]):
        r = ret.loc[:f, group["asset_id"]].tail(window)
        corr = r.corr(min_periods=window // 2).to_numpy()
        i, j = np.where(np.triu(corr, 1) > threshold)
        for a, b in zip(r.columns[i], r.columns[j]):
            found.setdefault((a, b), (month, float(corr[r.columns.get_loc(a), r.columns.get_loc(b)])))
    return pd.DataFrame([(a, b, m, c) for (a, b), (m, c) in found.items()],
                        columns=["asset_a", "asset_b", "first_month", "corr"])


def return_review(universe: pd.DataFrame, ret: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Daily returns beyond ±threshold for assets while they are in the universe."""
    hits = []
    for month, group in universe.groupby("month"):
        r = ret.loc[month: month + pd.offsets.MonthEnd(0), group["asset_id"]]
        big = r.stack()
        big = big[big.abs() > threshold]
        hits += [(d, a, v) for (d, a), v in big.items()]
    return pd.DataFrame(hits, columns=["date", "asset_id", "ret"]).sort_values(["date", "asset_id"], ignore_index=True)


def run(cfg: dict, args) -> None:
    paths = project_paths(cfg)
    kind, ucfg = args.kind, cfg["universe"]
    calendar = master_calendar(cfg, paths)
    panel = load_panel(paths, ["valid", "close_raw", "ret", "dollar_volume"])
    valid, close_raw, ret, dollar_volume = (wide(panel, c) for c in ["valid", "close_raw", "ret", "dollar_volume"])
    drop = set(pd.read_csv(paths.meta / "dual_class_drop.csv")["asset_id"])

    if kind == "pit":
        member_daily = pd.read_parquet(paths.processed / "member_daily.parquet")
        by_date = member_daily.groupby("date")["asset_id"].agg(set)
        members = lambda f: by_date.get(f, set())  # noqa: E731
    else:
        snapshot = members_on(load_snapshots(membership_path(paths)), ucfg["survivor_snapshot"])
        resolution = pd.read_csv(paths.processed / "resolution.csv", parse_dates=["sub_start", "sub_end"])
        today = resolution[resolution["member_ticker"].isin(snapshot) & resolution["sub_end"].ge(ucfg["survivor_snapshot"])]
        fixed = {a for a in today["asset_id"].dropna() if a.startswith("yahoo:")}
        members = lambda f: fixed  # noqa: E731
        log.info("survivor universe: %d of today's %d constituents, Yahoo only", len(fixed), len(snapshot))

    data_end = pd.Timestamp(cfg["dates"]["data_end"])
    formations = formation_dates(calendar, cfg["dates"]["first_holding_day"][:7], data_end.strftime("%Y-%m"))
    universe = form_universe(valid, close_raw, ret, dollar_volume, members, formations, drop, ucfg)
    universe.to_parquet(paths.processed / f"universe_{kind}.parquet", index=False)
    sizes = universe.groupby("month").size()
    log.info("universe_%s: %d months, size %d–%d, %d distinct assets, %d Tiingo-sourced slots",
             kind, len(sizes), sizes.min(), sizes.max(), universe["asset_id"].nunique(),
             int(universe["asset_id"].str.startswith("tiingo:").sum()))

    if kind == "pit":
        dups = duplicate_pairs(universe, ret, ucfg["history_days"], ucfg["dup_corr"])
        dups.to_csv(paths.processed / "duplicate_pairs.csv", index=False)
        review = return_review(universe, ret, cfg["data"]["return_review_threshold"])
        review.to_csv(paths.processed / "return_review.csv", index=False)
        log.info("%d near-duplicate pairs (corr > %.2f); %d returns beyond ±%.0f%% to review",
                 len(dups), ucfg["dup_corr"], len(review), 100 * cfg["data"]["return_review_threshold"])
