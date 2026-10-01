"""Sector per asset, for pair formation and factor interpretation (PLAN.md §6.5).

Yahoo-listed assets use yfinance's ``info["sector"]`` (cached). Delisted Tiingo assets have no
such lookup, so their sectors live in data/meta/sectors_manual.csv. These are current
classifications, not point-in-time ones.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from statarb.paths import project_paths

log = logging.getLogger(__name__)

SECTORS = [
    "Technology", "Healthcare", "Financial Services", "Consumer Cyclical", "Consumer Defensive",
    "Industrials", "Energy", "Utilities", "Real Estate", "Basic Materials", "Communication Services",
]


def _fetch_sector(symbol: str) -> str:
    import yfinance as yf

    for attempt in range(3):
        try:
            return yf.Ticker(symbol).info.get("sector") or ""
        except Exception as exc:  # rate limits surface as generic exceptions
            log.warning("sector %s failed (%s), retry in 30 s", symbol, type(exc).__name__)
            time.sleep(30)
    return ""


def run(cfg: dict, args) -> None:
    paths = project_paths(cfg)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    resolution = pd.read_csv(paths.processed / "resolution.csv", parse_dates=["sub_end"])
    resolved = resolution[resolution["status"].isin(["yahoo", "tiingo", "alias"])]
    assets = sorted(resolved["asset_id"].unique())

    cache_file = paths.raw / "sectors_yahoo.csv"
    cache = pd.read_csv(cache_file, dtype=str, keep_default_na=False) if cache_file.exists() \
        else pd.DataFrame(columns=["symbol", "sector"])
    todo = [a.split(":", 1)[1] for a in assets if a.startswith("yahoo:")]
    todo = [s for s in todo if s not in set(cache["symbol"])]
    log.info("sectors: %d Yahoo lookups to do", len(todo))
    for i, symbol in enumerate(todo, 1):
        cache.loc[len(cache)] = [symbol, _fetch_sector(symbol)]
        time.sleep(1)
        if i % 25 == 0 or i == len(todo):
            cache.to_csv(cache_file, index=False)
            log.info("sectors: %d/%d", i, len(todo))

    sectors = {f"yahoo:{s}": sec for s, sec in zip(cache["symbol"], cache["sector"])}
    manual = pd.read_csv(paths.meta / "sectors_manual.csv", dtype=str, keep_default_na=False)
    sectors.update(zip(manual["asset_id"], manual["sector"]))
    unknown_names = sorted(set(manual["sector"]) - set(SECTORS) - {"Unknown"})
    if unknown_names:
        raise ValueError(f"sectors_manual.csv uses non-standard sector names: {unknown_names}")

    out = pd.DataFrame({"asset_id": assets})
    out["sector"] = out["asset_id"].map(sectors).replace("", None).fillna("Unknown")
    out.to_csv(paths.processed / "sectors.csv", index=False)

    recent = set(resolved.loc[resolved["sub_end"] >= "2018-01-01", "asset_id"])
    gaps = sorted(a for a in recent if a.startswith("tiingo:") and a not in set(manual["asset_id"]))
    log.info("sectors: %d assets, %d Unknown", len(out), int((out["sector"] == "Unknown").sum()))
    if gaps:
        log.warning("add to data/meta/sectors_manual.csv (Tiingo assets active from 2018): %s", " ".join(gaps))
