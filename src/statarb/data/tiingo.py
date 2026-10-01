"""Daily prices for delisted names from Tiingo's free tier (PLAN.md §6.3).

The free tier allows 50 requests/hour, so requests are paced (75 s apart) and every symbol is
cached on arrival; an interrupted run resumes where it stopped. The API key is read from the
TIINGO_API_KEY environment variable or the repo's .env and only ever sent in the request header.
"""

from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from statarb.data import yahoo
from statarb.paths import REPO_ROOT, Paths, project_paths

log = logging.getLogger(__name__)

SUPPORTED_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
PRICES_URL = "https://api.tiingo.com/tiingo/daily/{symbol}/prices"
CROSS_CHECK = ["AAPL", "JPM", "XOM", "JNJ", "PG"]


def supported_path(paths: Paths) -> Path:
    return paths.raw / "tiingo_supported_tickers.csv"


def cache_path(paths: Paths, symbol: str) -> Path:
    return paths.raw / "tiingo" / f"{symbol}.parquet"


def missing_path(paths: Paths) -> Path:
    return paths.raw / "tiingo_missing.txt"


def load_supported(paths: Paths) -> pd.DataFrame:
    """Tiingo's public ticker list (downloaded once): ticker (upper case), startDate, endDate."""
    path = supported_path(paths)
    if not path.exists():
        resp = requests.get(SUPPORTED_URL, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(zf.read("supported_tickers.csv"))
        log.info("downloaded Tiingo supported-ticker list")
    listing = pd.read_csv(path, dtype=str, keep_default_na=False)
    listing["ticker"] = listing["ticker"].str.upper()
    for col in ("startDate", "endDate"):
        listing[col] = pd.to_datetime(listing[col].replace("", None), errors="coerce")
    return listing


def load_prices(paths: Paths, symbol: str) -> pd.DataFrame | None:
    path = cache_path(paths, symbol)
    return pd.read_parquet(path) if path.exists() else None


def read_api_key() -> str:
    key = os.environ.get("TIINGO_API_KEY", "").strip()
    env_file = REPO_ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "TIINGO_API_KEY":
                key = value.strip().strip("'\"")
    if not key:
        raise RuntimeError("TIINGO_API_KEY is not set (environment or .env)")
    return key


class TiingoClient:
    def __init__(self, key: str, min_interval: float):
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Token {key}", "Content-Type": "application/json"})
        self.min_interval = min_interval
        self._last = 0.0

    def prices(self, symbol: str, start: str, end: str) -> pd.DataFrame | None:
        """Daily prices, or None if Tiingo has nothing for the symbol."""
        for attempt in range(4):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            resp = self.session.get(PRICES_URL.format(symbol=symbol),
                                    params={"startDate": start, "endDate": end, "format": "json"}, timeout=60)
            self._last = time.monotonic()
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return _to_frame(data)
                log.warning("tiingo %s: unexpected payload %s", symbol, str(data)[:120])
                return None
            if resp.status_code == 404:
                return None
            if resp.status_code == 429 or "limit" in resp.text.lower():
                log.warning("tiingo rate limit hit; sleeping 1 hour (attempt %d/4)", attempt + 1)
                time.sleep(3600)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"tiingo {symbol}: still rate limited after 4 attempts")


def _to_frame(records: list[dict]) -> pd.DataFrame | None:
    if not records:
        return None
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_localize(None).dt.normalize()
    return frame.set_index("date").sort_index()


def download(symbols: list[str], cfg: dict, paths: Paths) -> None:
    known_missing = set(missing_path(paths).read_text().split()) if missing_path(paths).exists() else set()
    todo = [s for s in dict.fromkeys(symbols) if not cache_path(paths, s).exists() and s not in known_missing]
    interval = cfg["data"]["tiingo_seconds_between_requests"]
    log.info("tiingo: %d symbols to download, ~%.1f h at one request per %d s",
             len(todo), len(todo) * interval / 3600, interval)
    if not todo:
        return
    (paths.raw / "tiingo").mkdir(parents=True, exist_ok=True)
    client = TiingoClient(read_api_key(), interval)
    for i, symbol in enumerate(todo, 1):
        prices = client.prices(symbol, cfg["dates"]["download_start"], cfg["dates"]["data_end"])
        if prices is None:
            with open(missing_path(paths), "a") as fh:
                fh.write(symbol + "\n")
            log.info("tiingo %d/%d %s: no data", i, len(todo), symbol)
        else:
            prices.to_parquet(cache_path(paths, symbol))
            log.info("tiingo %d/%d %s: %d rows %s → %s", i, len(todo), symbol, len(prices),
                     prices.index[0].date(), prices.index[-1].date())


def cross_check(paths: Paths, since: str = "2010-01-01") -> pd.DataFrame:
    """Daily returns of a few large caps must agree between Yahoo and Tiingo."""
    rows = []
    for symbol in CROSS_CHECK:
        y = yahoo.load_prices(paths, symbol)["Adj Close"]
        t = load_prices(paths, symbol)["adjClose"]
        both = pd.concat([y, t], axis=1, keys=["yahoo", "tiingo"]).dropna()
        rets = both[both.index >= since].pct_change().dropna()
        diff = (rets["yahoo"] - rets["tiingo"]).abs()
        rows.append(dict(symbol=symbol, days=len(rets), corr=rets["yahoo"].corr(rets["tiingo"]),
                         median_abs_diff_bp=float(np.median(diff)) * 1e4, max_abs_diff_bp=float(diff.max()) * 1e4))
    table = pd.DataFrame(rows)
    table["ok"] = (table["corr"] > 0.999) & (table["median_abs_diff_bp"] < 1.0)
    return table


def run(cfg: dict, args) -> None:
    paths = project_paths(cfg)
    needed = pd.read_csv(paths.processed / "tiingo_needed.csv")["symbol"].tolist()
    download(CROSS_CHECK + needed, cfg, paths)
    table = cross_check(paths)
    table.to_csv(paths.processed / "source_consistency.csv", index=False)
    log.info("Yahoo vs Tiingo consistency:\n%s", table.to_string(index=False))
    if not table["ok"].all():
        raise RuntimeError("Yahoo and Tiingo returns disagree; see source_consistency.csv")
