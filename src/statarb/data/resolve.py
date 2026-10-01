"""Map every membership spell to exactly one price series, or mark it unrecoverable (PLAN.md §6.4).

Run it after `yahoo`; spells that need Tiingo come out as ``pending_tiingo`` and their symbols are
written to ``tiingo_needed.csv``. Run `tiingo`, then this step again, until nothing is pending.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from statarb.data import tiingo, yahoo
from statarb.data.membership import additions, load_snapshots, membership_path
from statarb.paths import Paths, project_paths

log = logging.getLogger(__name__)

DAY = pd.Timedelta(days=1)


def load_aliases(path) -> pd.DataFrame:
    aliases = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("valid_from", "valid_to"):
        aliases[col] = pd.to_datetime(aliases[col].replace("", None))
    return aliases


def split_range(start: pd.Timestamp, end: pd.Timestamp, alias_rows: pd.DataFrame) -> list[tuple]:
    """Cut [start, end] at the alias validity boundaries → [(seg_start, seg_end, alias_row | None)]."""
    cuts = {start, end + DAY}
    for row in alias_rows.itertuples(index=False):
        if pd.notna(row.valid_from) and start < row.valid_from <= end:
            cuts.add(row.valid_from)
        if pd.notna(row.valid_to) and start <= row.valid_to < end:
            cuts.add(row.valid_to + DAY)
    bounds = sorted(cuts)
    segments = []
    for seg_start, nxt in zip(bounds[:-1], bounds[1:]):
        seg_end = nxt - DAY
        match = None
        for row in alias_rows.itertuples(index=False):
            if (pd.isna(row.valid_from) or row.valid_from <= seg_start) and (pd.isna(row.valid_to) or seg_end <= row.valid_to):
                match = row
                break
        segments.append((seg_start, seg_end, match))
    return segments


def check_coverage(valid_dates: pd.DatetimeIndex, days: pd.DatetimeIndex, cov_min: float, tol: int) -> tuple[bool, float, int]:
    """Does a series cover the trading ``days``? Returns (ok, coverage, longest internal gap in days)."""
    if len(days) == 0:
        return True, 1.0, 0
    have = days.isin(valid_dates)
    coverage = float(have.mean())
    if not have.any():
        return False, 0.0, len(days)
    idx = np.flatnonzero(have)
    first, last = idx[0], idx[-1]
    gaps = np.diff(idx) - 1
    longest_gap = int(gaps.max()) if len(gaps) else 0
    ok = coverage >= cov_min and first <= tol and last >= len(days) - 1 - tol
    return ok, coverage, longest_gap


def _overlaps(listings: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return bool(((listings["startDate"] <= end) & (listings["endDate"].isna() | (listings["endDate"] >= start))).any())


def resolve_spells(
    spells: pd.DataFrame,
    aliases: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    valid_dates: Callable[[str, str], pd.DatetimeIndex | None],
    tiingo_listings: Callable[[str], pd.DataFrame],
    cfg: dict,
) -> tuple[pd.DataFrame, set[str]]:
    """Resolve each spell's sub-ranges. ``valid_dates(source, symbol)`` returns None if not cached.

    Returns the resolution table and the Tiingo symbols that still need downloading.
    """
    d = cfg["data"]
    since, data_end = pd.Timestamp(d["spells_from"]), pd.Timestamp(cfg["dates"]["data_end"])
    cov_min, tol, gap_review = d["coverage_min"], d["edge_tolerance_days"], d["review_gap_days"]
    rows, needed = [], set()

    for spell in spells.itertuples(index=False):
        start = max(spell.spell_start, since)
        end = data_end if pd.isna(spell.spell_end) else min(spell.spell_end - DAY, data_end)
        if end < start:
            continue
        ticker = spell.member_ticker
        for seg_start, seg_end, alias in split_range(start, end, aliases[aliases["member_ticker"] == ticker]):
            days = calendar[(calendar >= seg_start) & (calendar <= seg_end)]
            row = dict(member_ticker=ticker, spell_start=spell.spell_start, spell_end=spell.spell_end,
                       sub_start=seg_start, sub_end=seg_end, status="unresolved", asset_id="",
                       coverage=np.nan, review=False, note="")
            if alias is not None:
                row["note"] = alias.note
                if alias.source == "none":
                    row["status"] = "none"
                    rows.append(row)
                    continue
                candidates = [(alias.source, alias.symbol, "alias")]
                row["review"] = alias.confidence == "verify"
            else:
                candidates = [("yahoo", yahoo.yahoo_symbol(ticker), "yahoo"),
                              ("tiingo", ticker, "tiingo"), ("tiingo", ticker + "Q", "tiingo")]

            for source, symbol, status in candidates:
                reused = False
                # Alias rows are authoritative (they may name a Tiingo permaTicker, which isn't in
                # the ticker list); default Tiingo candidates must be listed for these dates.
                if source == "tiingo" and status != "alias":
                    listings = tiingo_listings(symbol)
                    if not _overlaps(listings, seg_start, seg_end):
                        continue
                    reused = len(listings) > 1
                dates = valid_dates(source, symbol)
                if dates is None:
                    if source == "tiingo":
                        needed.add(symbol)
                        row.update(status="pending_tiingo", asset_id=f"tiingo:{symbol}")
                        break
                    continue
                ok, coverage, gap = check_coverage(dates, days, cov_min, tol)
                if ok:
                    row.update(status=status, asset_id=f"{source}:{symbol}", coverage=round(coverage, 4),
                               review=row["review"] or reused or gap > gap_review)
                    break
            rows.append(row)

    return pd.DataFrame(rows), needed


def _rename_hints(table: pd.DataFrame, adds: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    """For unresolved rows: tickers added to the index within ±3 trading days of the spell end."""
    hints = pd.Series("", index=table.index)
    for i, row in table[table["status"] == "unresolved"].iterrows():
        if pd.isna(row["spell_end"]):
            continue
        pos = calendar.searchsorted(row["spell_end"])
        lo, hi = calendar[max(pos - 3, 0)], calendar[min(pos + 3, len(calendar) - 1)]
        near = adds[(adds["date"] >= lo) & (adds["date"] <= hi)]["ticker"]
        hints[i] = " ".join(near)
    return hints


def run(cfg: dict, args) -> None:
    paths: Paths = project_paths(cfg)
    spells = pd.read_parquet(paths.processed / "spells.parquet")
    aliases = load_aliases(paths.meta / "ticker_aliases.csv")
    calendar = yahoo.master_calendar(cfg, paths)
    supported = tiingo.load_supported(paths)
    by_ticker = {t: g for t, g in supported.groupby("ticker")}
    empty = supported.iloc[:0]
    cache: dict[tuple[str, str], pd.DatetimeIndex | None] = {}
    tiingo_missing = set(tiingo.missing_path(paths).read_text().split()) if tiingo.missing_path(paths).exists() else set()

    def valid_dates(source: str, symbol: str) -> pd.DatetimeIndex | None:
        if source == "tiingo" and symbol in tiingo_missing:
            return pd.DatetimeIndex([])  # Tiingo answered with no data: fails validation, not "pending"
        if (source, symbol) not in cache:
            prices = yahoo.load_prices(paths, symbol) if source == "yahoo" else tiingo.load_prices(paths, symbol)
            col = "Adj Close" if source == "yahoo" else "adjClose"
            cache[(source, symbol)] = None if prices is None else prices.index[prices[col] > 0]
        return cache[(source, symbol)]

    table, needed = resolve_spells(spells, aliases, calendar, valid_dates,
                                   lambda s: by_ticker.get(s.upper(), empty), cfg)
    adds = additions(load_snapshots(membership_path(paths)))
    table["hints"] = _rename_hints(table, adds, calendar)
    table.to_csv(paths.processed / "resolution.csv", index=False)
    pd.DataFrame({"symbol": sorted(needed)}).to_csv(paths.processed / "tiingo_needed.csv", index=False)

    counts = table["status"].value_counts().to_dict()
    log.info("resolution: %s; %d rows flagged for review", counts, int(table["review"].sum()))
    log.info("%d Tiingo symbols still to download", len(needed))
    for row in table[table["status"] == "unresolved"].itertuples(index=False):
        log.info("unresolved %-6s %s → %s  hints: %s", row.member_ticker, row.sub_start.date(),
                 row.sub_end.date(), row.hints or "-")
