"""Survivorship coverage report (PLAN.md §7.4) and the SVB/First Republic halt check (§7.2).

For each year: index member-days by how they were resolved, member-days with no price data, and
how many universe slots are filled by Tiingo-only (delisted) names, i.e. names a yfinance-only
pipeline would silently miss.
"""

from __future__ import annotations

import logging

import pandas as pd

from statarb.data.membership import load_snapshots, membership_path
from statarb.data.panel import load_panel
from statarb.data.yahoo import master_calendar
from statarb.paths import project_paths

log = logging.getLogger(__name__)

HALT_CASES = {"tiingo:SIVBQ": "SVB Financial", "tiingo:FRCB": "First Republic"}


def member_days_by_status(resolution: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Member-days per year, split by resolution status."""
    years = pd.Series(calendar.year, index=calendar)
    rows = []
    for row in resolution.itertuples(index=False):
        days = years[(calendar >= row.sub_start) & (calendar <= row.sub_end)]
        source = row.asset_id.split(":")[0] if isinstance(row.asset_id, str) and row.status != "none" else "none"
        label = "unrecoverable" if row.status == "none" else ("alias" if row.status == "alias" else source)
        rows += [(y, label, n) for y, n in days.value_counts().items()]
    table = pd.DataFrame(rows, columns=["year", "via", "days"]).pivot_table(
        index="year", columns="via", values="days", aggfunc="sum", fill_value=0)
    return table.reindex(columns=["yahoo", "tiingo", "alias", "unrecoverable"], fill_value=0)


def halt_check(panel: pd.DataFrame) -> list[str]:
    """Describe how each failed bank's last trading days and post-halt prices appear in the data."""
    lines = []
    for asset_id, name in HALT_CASES.items():
        s = panel[panel["asset_id"] == asset_id].set_index("date")
        s = s[(s.index >= "2023-02-15") & (s.index <= "2023-06-30")]
        valid = s[s["valid"]]
        gaps = valid.index.to_series().diff().dt.days
        big = valid.loc[valid["ret"].abs() > 0.3, "ret"]
        lines.append(f"{name} ({asset_id}): {len(valid)} valid days Feb 15–Jun 30 2023; "
                     f"largest calendar gap {int(gaps.max())} days; moves beyond ±30%: "
                     + (", ".join(f"{d.date()} {r:+.1%}" for d, r in big.items()) or "none"))
    return lines


def run(cfg: dict, args) -> None:
    paths = project_paths(cfg)
    calendar = master_calendar(cfg, paths)
    resolution = pd.read_csv(paths.processed / "resolution.csv", parse_dates=["sub_start", "sub_end"])

    table = member_days_by_status(resolution, calendar)
    snapshots = load_snapshots(membership_path(paths))
    pit = pd.Series([len(t) for t in snapshots["tickers"]], index=snapshots["date"])
    table.insert(0, "member_days", [int(pit.reindex(calendar[calendar.year == y], method="ffill").sum()) for y in table.index])
    table["pct_with_data"] = (1 - table["unrecoverable"] / table[["yahoo", "tiingo", "alias", "unrecoverable"]].sum(axis=1)).round(4)

    universe = pd.read_parquet(paths.processed / "universe_pit.parquet")
    slots = universe.assign(year=universe["month"].dt.year, tiingo=universe["asset_id"].str.startswith("tiingo:"))
    per_year = slots.groupby("year").agg(universe_slots=("asset_id", "size"), tiingo_slots=("tiingo", "sum"))
    table = table.join(per_year)
    table["tiingo_slot_share"] = (table["tiingo_slots"] / table["universe_slots"]).round(4)
    table.to_csv(paths.processed / "coverage_report.csv")

    gone = resolution[(resolution["status"] == "none") & (resolution["sub_end"] >= "2010-01-01")]
    gone[["member_ticker", "sub_start", "sub_end", "note"]].to_csv(paths.processed / "unrecoverable.csv", index=False)

    halts = halt_check(load_panel(paths, ["valid", "ret"]))
    (paths.processed / "halt_check.txt").write_text("\n".join(halts) + "\n")
    log.info("coverage by year:\n%s", table.to_string())
    log.info("%d unrecoverable spells (%d overlap 2019+)", len(gone), int((gone["sub_end"] >= "2019-01-01").sum()))
    for line in halts:
        log.info(line)
