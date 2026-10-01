import pandas as pd

from statarb.config import load_config
from statarb.data.resolve import check_coverage, load_aliases, resolve_spells, split_range

CAL = pd.bdate_range("2010-01-01", "2026-08-31")
EMPTY_LISTING = pd.DataFrame({"startDate": pd.to_datetime([]), "endDate": pd.to_datetime([])})


def aliases(text: str, tmp_path) -> pd.DataFrame:
    path = tmp_path / "aliases.csv"
    path.write_text("member_ticker,valid_from,valid_to,source,symbol,confidence,note\n" + text)
    return load_aliases(path)


def test_series_starting_late_is_rejected():
    days = CAL[(CAL >= "2013-12-23") & (CAL <= "2022-06-08")]
    reused_ticker = CAL[CAL >= "2025-06-26"]  # the FB case: ticker now belongs to an ETF
    assert not check_coverage(reused_ticker, days, 0.95, 5)[0]
    assert check_coverage(CAL, days, 0.95, 5)[0]


def test_low_coverage_rejected_and_gap_reported():
    days = CAL[:400]
    sparse = days[::2]
    assert not check_coverage(sparse, days, 0.95, 5)[0]
    with_gap = days.delete(range(50, 62))  # 97% coverage, one 12-day hole
    ok, coverage, gap = check_coverage(with_gap, days, 0.95, 5)
    assert ok and gap == 12


def test_split_range_at_alias_boundaries(tmp_path):
    rows = aliases("IR,,2020-02-28,yahoo,TT,verify,old IR\nIR,2020-03-02,,yahoo,IR,verify,new IR\n", tmp_path)
    segments = split_range(pd.Timestamp("2010-11-17"), pd.Timestamp("2026-08-31"), rows)
    assert [(s.date().isoformat(), e.date().isoformat(), a.symbol if a else None) for s, e, a in segments] == [
        ("2010-11-17", "2020-02-28", "TT"),
        ("2020-02-29", "2020-03-01", None),
        ("2020-03-02", "2026-08-31", "IR"),
    ]


def test_resolve_prefers_alias_then_yahoo_then_tiingo(tmp_path):
    cfg = load_config()
    spells = pd.DataFrame({
        "member_ticker": ["FB", "AAPL", "ATVI"],
        "spell_start": pd.to_datetime(["2013-12-23", "2005-01-01", "2015-08-31"]),
        "spell_end": pd.to_datetime(["2022-06-09", None, "2023-10-18"]),
    })
    available = {("yahoo", "META"): CAL[CAL >= "2012-05-18"], ("yahoo", "AAPL"): CAL,
                 ("yahoo", "FB"): CAL[CAL >= "2025-06-26"]}
    listings = {"ATVI": pd.DataFrame({"startDate": pd.to_datetime(["1993-10-25"]), "endDate": pd.to_datetime(["2023-10-13"])})}

    table, needed = resolve_spells(spells, aliases("FB,,,yahoo,META,high,rename\n", tmp_path), CAL,
                                   lambda src, sym: available.get((src, sym)),
                                   lambda sym: listings.get(sym, EMPTY_LISTING), cfg)
    status = dict(zip(table["member_ticker"], zip(table["status"], table["asset_id"])))
    assert status["FB"] == ("alias", "yahoo:META")
    assert status["AAPL"] == ("yahoo", "yahoo:AAPL")
    assert status["ATVI"] == ("pending_tiingo", "tiingo:ATVI")
    assert needed == {"ATVI"}


def test_alias_none_is_explicitly_unrecoverable(tmp_path):
    spells = pd.DataFrame({"member_ticker": ["ARNC"], "spell_start": pd.to_datetime(["1996-01-02"]),
                           "spell_end": pd.to_datetime(["2016-06-01"])})
    table, _ = resolve_spells(spells, aliases("ARNC,,,none,,high,no data\n", tmp_path), CAL,
                              lambda *a: None, lambda s: EMPTY_LISTING, load_config())
    assert table["status"].tolist() == ["none"]


def test_ticker_with_no_data_anywhere_stays_unresolved(tmp_path):
    spells = pd.DataFrame({"member_ticker": ["ZZZ"], "spell_start": pd.to_datetime(["2015-01-01"]),
                           "spell_end": pd.to_datetime([None])})
    table, needed = resolve_spells(spells, aliases("", tmp_path), CAL, lambda *a: None,
                                   lambda s: EMPTY_LISTING, load_config())
    assert table["status"].tolist() == ["unresolved"] and not needed
