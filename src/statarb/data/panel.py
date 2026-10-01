"""Aligned daily panel of every resolved asset on the master calendar (PLAN.md §7.1).

Returns are total returns from consecutive *valid* closes, so the whole move across a halt is
booked on the first valid day after it. Gaps longer than ``data.gap_max_days`` are treated as a
fresh start (no return on resumption); the backtest engine treats them as permanent ends.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from statarb.data import tiingo, yahoo
from statarb.paths import Paths, project_paths

log = logging.getLogger(__name__)

FIELDS = {
    "yahoo": {"close_adj": "Adj Close", "close_raw": "Close", "high": "High", "low": "Low", "volume": "Volume"},
    "tiingo": {"close_adj": "adjClose", "close_raw": "close", "high": "adjHigh", "low": "adjLow", "volume": "volume"},
}
RESOLVED = ["yahoo", "tiingo", "alias"]


def stale_days(close_raw: pd.Series, run_days: int) -> pd.Series:
    """True on the repeat days of a run of ≥ ``run_days`` identical closes.

    Data vendors carry the last price through trading halts (SVB: $106.04 from 2023-03-09 to
    03-27). Those days were not tradable, so they must not count as valid closes.
    """
    same = close_raw.eq(close_raw.shift(1))
    run_len = same.groupby((~same).cumsum()).transform("size")
    return same & run_len.ge(run_days)


def asset_frame(raw: pd.DataFrame, source: str, calendar: pd.DatetimeIndex, gap_max_days: int,
                stale_run_days: int = 3) -> pd.DataFrame:
    """One asset's raw download → panel columns on the master calendar."""
    mapping = FIELDS[source]
    frame = raw[list(mapping.values())].rename(columns={v: k for k, v in mapping.items()})
    frame = frame[~frame.index.duplicated(keep="last")].reindex(calendar).astype("float64")
    frame.index.name = "date"
    frame["valid"] = frame["close_adj"].gt(0) & ~stale_days(frame["close_raw"], stale_run_days)
    adj = frame["close_adj"].where(frame["valid"])
    frame["ret"] = (adj / adj.ffill().shift(1) - 1).where(frame["valid"])

    pos = pd.Series(np.arange(len(frame), dtype=float), index=frame.index)
    last_valid = pos.where(frame["valid"]).ffill().shift(1)
    frame.loc[(pos - last_valid - 1) > gap_max_days, "ret"] = np.nan

    frame["dollar_volume"] = (frame["close_raw"] * frame["volume"]).where(frame["volume"] > 0)
    return frame


def apply_price_fixes(panel: pd.DataFrame, fixes: pd.DataFrame) -> pd.DataFrame:
    """Set ``ret`` to NaN at confirmed data errors listed in price_fixes.csv."""
    if fixes.empty:
        return panel
    keys = pd.MultiIndex.from_arrays([pd.to_datetime(fixes["date"]), fixes["asset_id"]])
    hit = pd.MultiIndex.from_arrays([panel["date"], panel["asset_id"]]).isin(keys)
    panel.loc[hit, "ret"] = np.nan
    log.info("price fixes: %d returns set to NaN", int(hit.sum()))
    return panel


def membership_rows(resolution: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """(date, asset_id) for every trading day an asset is an index member."""
    parts = []
    for row in resolution[resolution["status"].isin(RESOLVED)].itertuples(index=False):
        days = calendar[(calendar >= row.sub_start) & (calendar <= row.sub_end)]
        parts.append(pd.DataFrame({"date": days, "asset_id": row.asset_id}))
    return pd.concat(parts, ignore_index=True).drop_duplicates(ignore_index=True)


def load_panel(paths: Paths, columns: list[str] | None = None) -> pd.DataFrame:
    cols = None if columns is None else ["date", "asset_id", *columns]
    return pd.read_parquet(paths.processed / "panel.parquet", columns=cols)


def wide(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    """Dates × assets matrix of one panel field."""
    return panel.pivot(index="date", columns="asset_id", values=field)


def run(cfg: dict, args) -> None:
    paths = project_paths(cfg)
    calendar = yahoo.master_calendar(cfg, paths)
    resolution = pd.read_csv(paths.processed / "resolution.csv", parse_dates=["sub_start", "sub_end"])
    assets = sorted(resolution.loc[resolution["status"].isin(RESOLVED), "asset_id"].unique())

    frames = []
    for asset_id in assets:
        source, symbol = asset_id.split(":", 1)
        raw = yahoo.load_prices(paths, symbol) if source == "yahoo" else tiingo.load_prices(paths, symbol)
        frame = asset_frame(raw, source, calendar, cfg["data"]["gap_max_days"],
                            cfg["data"]["stale_run_days"]).reset_index()
        frame.insert(1, "asset_id", asset_id)
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    fixes = pd.read_csv(paths.meta / "price_fixes.csv", dtype=str)
    panel = apply_price_fixes(panel, fixes)
    panel.to_parquet(paths.processed / "panel.parquet", index=False)

    members = membership_rows(resolution, calendar)
    members.to_parquet(paths.processed / "member_daily.parquet", index=False)
    log.info("panel: %d assets × %d days (%s → %s), %d valid rows; %d member-days",
             len(assets), len(calendar), calendar[0].date(), calendar[-1].date(),
             int(panel["valid"].sum()), len(members))
