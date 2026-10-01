"""Point-in-time S&P 500 membership (fja05680/sp500 at a pinned commit) and membership spells."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from statarb.paths import Paths, project_paths

log = logging.getLogger(__name__)

RAW_URL = "https://raw.githubusercontent.com/fja05680/sp500/{sha}/{file}"


def membership_path(paths: Paths) -> Path:
    return paths.raw / "membership" / "sp500_membership.csv"


def download_membership(cfg: dict, paths: Paths) -> Path:
    """Fetch the membership file at the pinned commit; a cached copy is reused."""
    path = membership_path(paths)
    if path.exists():
        return path
    url = RAW_URL.format(sha=cfg["data"]["membership_sha"], file=quote(cfg["data"]["membership_file"]))
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    log.info("downloaded membership file (%d bytes)", len(resp.content))
    return path


def load_snapshots(path: Path) -> pd.DataFrame:
    """One row per snapshot: effective ``date`` and ``tickers`` (frozenset), sorted by date."""
    raw = pd.read_csv(path)
    snapshots = pd.DataFrame({
        "date": pd.to_datetime(raw["date"]),
        "tickers": [frozenset(t.strip() for t in s.split(",") if t.strip()) for s in raw["tickers"]],
    })
    return snapshots.sort_values("date", ignore_index=True)


def members_on(snapshots: pd.DataFrame, date) -> frozenset[str]:
    """Index members on ``date``: the latest snapshot dated on or before it."""
    i = snapshots["date"].searchsorted(pd.Timestamp(date), side="right") - 1
    return snapshots["tickers"].iloc[i] if i >= 0 else frozenset()


def build_spells(snapshots: pd.DataFrame, since=None) -> pd.DataFrame:
    """Maximal runs of consecutive snapshots that contain each ticker.

    ``spell_end`` is the first snapshot date the ticker is absent (exclusive), NaT while still a
    member. With ``since``, only spells that are still open on or after that date are kept.
    """
    opened: dict[str, pd.Timestamp] = {}
    rows = []
    prev: frozenset[str] = frozenset()
    for date, tickers in zip(snapshots["date"], snapshots["tickers"]):
        for ticker in tickers - prev:
            opened[ticker] = date
        for ticker in prev - tickers:
            rows.append((ticker, opened.pop(ticker), date))
        prev = tickers
    rows += [(ticker, start, pd.NaT) for ticker, start in opened.items()]
    spells = pd.DataFrame(rows, columns=["member_ticker", "spell_start", "spell_end"])
    spells["spell_end"] = pd.to_datetime(spells["spell_end"])
    if since is not None:
        spells = spells[spells["spell_end"].isna() | (spells["spell_end"] > pd.Timestamp(since))]
    return spells.sort_values(["member_ticker", "spell_start"], ignore_index=True)


def additions(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Tickers added to the index at each snapshot after the first (``date``, ``ticker``)."""
    rows = []
    for prev, cur, date in zip(snapshots["tickers"], snapshots["tickers"].iloc[1:], snapshots["date"].iloc[1:]):
        rows += [(date, ticker) for ticker in sorted(cur - prev)]
    return pd.DataFrame(rows, columns=["date", "ticker"])


def run(cfg: dict, args) -> None:
    paths = project_paths(cfg)
    snapshots = load_snapshots(download_membership(cfg, paths))
    spells = build_spells(snapshots, since=cfg["data"]["spells_from"])
    paths.processed.mkdir(parents=True, exist_ok=True)
    spells.to_parquet(paths.processed / "spells.parquet", index=False)
    log.info(
        "%d snapshots %s → %s; %d spells (%d tickers) overlap %s+",
        len(snapshots), snapshots["date"].iloc[0].date(), snapshots["date"].iloc[-1].date(),
        len(spells), spells["member_ticker"].nunique(), cfg["data"]["spells_from"],
    )
