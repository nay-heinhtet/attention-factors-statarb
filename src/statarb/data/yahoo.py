"""Daily prices from Yahoo via yfinance, cached one parquet file per symbol (PLAN.md §6.2)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from statarb.paths import Paths, project_paths

log = logging.getLogger(__name__)

COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def yahoo_symbol(member_ticker: str) -> str:
    """Yahoo spells share classes with a dash: BRK.B → BRK-B."""
    return member_ticker.replace(".", "-")


def cache_path(paths: Paths, symbol: str) -> Path:
    return paths.raw / "yahoo" / f"{symbol}.parquet"


def missing_path(paths: Paths) -> Path:
    return paths.raw / "yahoo_missing.txt"


def load_prices(paths: Paths, symbol: str) -> pd.DataFrame | None:
    path = cache_path(paths, symbol)
    return pd.read_parquet(path) if path.exists() else None


def master_calendar(cfg: dict, paths: Paths) -> pd.DatetimeIndex:
    """Trading days = SPY's dates on Yahoo between download_start and data_end."""
    spy = load_prices(paths, "SPY")
    if spy is None:
        raise FileNotFoundError("SPY is not cached yet; run `python -m statarb yahoo` first")
    dates = spy.index[spy["Adj Close"].notna()]
    return dates[(dates >= cfg["dates"]["download_start"]) & (dates <= cfg["dates"]["data_end"])]


def _fetch(chunk: list[str], start: str, end_exclusive: str) -> pd.DataFrame:
    import yfinance as yf

    for attempt in range(5):
        try:
            return yf.download(chunk, start=start, end=end_exclusive, auto_adjust=False, actions=False,
                               group_by="ticker", threads=True, progress=False)
        except Exception as exc:  # yfinance raises rate-limit and network errors
            log.warning("yahoo chunk failed (%s), retry %d/5 in 60 s", type(exc).__name__, attempt + 1)
            time.sleep(60)
    raise RuntimeError(f"yahoo download failed 5 times for {chunk[:3]}…")


def _split(frame: pd.DataFrame, symbol: str, start: str, end: str) -> pd.DataFrame | None:
    if not isinstance(frame.columns, pd.MultiIndex) or symbol not in frame.columns.get_level_values(0):
        return None
    prices = frame[symbol].reindex(columns=COLUMNS)
    prices = prices[(prices.index >= start) & (prices.index <= end)].dropna(how="all")
    if prices["Adj Close"].notna().sum() == 0:
        return None
    prices.index = pd.DatetimeIndex(prices.index).normalize().rename("date")
    return prices


def download(symbols: list[str], cfg: dict, paths: Paths) -> None:
    """Fetch uncached symbols in chunks; symbols that come back empty twice go to yahoo_missing.txt."""
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    start, end = cfg["dates"]["download_start"], cfg["dates"]["data_end"]
    end_exclusive = str((pd.Timestamp(end) + pd.Timedelta(days=1)).date())
    chunk_size = cfg["data"]["yahoo_chunk"]
    known_missing = set(missing_path(paths).read_text().split()) if missing_path(paths).exists() else set()
    todo = [s for s in symbols if not cache_path(paths, s).exists() and s not in known_missing]
    (paths.raw / "yahoo").mkdir(parents=True, exist_ok=True)
    log.info("yahoo: %d symbols requested, %d to download", len(symbols), len(todo))

    for attempt in (1, 2):
        empty = []
        for i in range(0, len(todo), chunk_size):
            chunk = todo[i:i + chunk_size]
            frame = _fetch(chunk, start, end_exclusive)
            for symbol in chunk:
                prices = _split(frame, symbol, start, end)
                if prices is None:
                    empty.append(symbol)
                else:
                    prices.to_parquet(cache_path(paths, symbol))
            log.info("yahoo pass %d: %d/%d done", attempt, min(i + chunk_size, len(todo)), len(todo))
            time.sleep(3)
        todo = empty
        if not todo:
            break
        if attempt == 1:
            log.info("yahoo: retrying %d empty symbols once", len(todo))
            time.sleep(30)

    missing_path(paths).write_text("\n".join(sorted(known_missing | set(todo))) + "\n")
    log.info("yahoo: %d symbols have no Yahoo data", len(known_missing | set(todo)))


def symbols_to_download(paths: Paths) -> list[str]:
    """Every member ticker (Yahoo spelling), every Yahoo alias target, and SPY."""
    spells = pd.read_parquet(paths.processed / "spells.parquet")
    aliases = pd.read_csv(paths.meta / "ticker_aliases.csv", dtype=str)
    symbols = {yahoo_symbol(t) for t in spells["member_ticker"]}
    symbols |= set(aliases.loc[aliases["source"] == "yahoo", "symbol"])
    symbols.add("SPY")
    return sorted(symbols)


def run(cfg: dict, args) -> None:
    paths = project_paths(cfg)
    download(symbols_to_download(paths), cfg, paths)
