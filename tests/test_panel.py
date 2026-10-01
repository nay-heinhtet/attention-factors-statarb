import numpy as np
import pandas as pd
import pytest

from statarb.data.panel import asset_frame, stale_days

CAL = pd.bdate_range("2020-01-01", periods=10)


def raw(closes, volumes=None) -> pd.DataFrame:
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "Adj Close": closes, "Close": closes, "High": closes * 1.01, "Low": closes * 0.99,
        "Volume": np.full(n, 1000.0) if volumes is None else volumes,
    }, index=CAL[:n])


def test_halt_return_is_booked_on_resumption():
    frame = asset_frame(raw([100, np.nan, np.nan, 50]), "yahoo", CAL[:4], gap_max_days=60)
    assert frame["valid"].tolist() == [True, False, False, True]
    assert np.isnan(frame["ret"].iloc[0]) and frame["ret"].iloc[3] == pytest.approx(-0.5)


def test_gap_longer_than_limit_restarts_without_return():
    frame = asset_frame(raw([100, np.nan, np.nan, 50]), "yahoo", CAL[:4], gap_max_days=1)
    assert np.isnan(frame["ret"].iloc[3])


def test_stale_halt_prices_are_not_tradable():
    # SVB pattern: the vendor repeats the last close through the halt, then the collapse prints.
    frame = asset_frame(raw([10, 11, 11, 11, 0.04]), "yahoo", CAL[:5], gap_max_days=60)
    assert frame["valid"].tolist() == [True, True, False, False, True]
    assert frame["ret"].iloc[4] == pytest.approx(0.04 / 11 - 1)


def test_two_equal_closes_are_normal_trading():
    assert not stale_days(pd.Series([10.0, 10.0, 11.0]), 3).any()


def test_dollar_volume_and_zero_volume():
    frame = asset_frame(raw([10, 12], volumes=np.array([500.0, 0.0])), "yahoo", CAL[:2], gap_max_days=60)
    assert frame["dollar_volume"].iloc[0] == 5000.0 and np.isnan(frame["dollar_volume"].iloc[1])


def test_tiingo_uses_adjusted_high_low():
    t = pd.DataFrame({"adjClose": [10.0, 11.0], "close": [20.0, 22.0], "adjHigh": [10.5, 11.5], "adjLow": [9.5, 10.5],
                      "high": [21.0, 23.0], "low": [19.0, 21.0], "volume": [100.0, 100.0]}, index=CAL[:2])
    frame = asset_frame(t, "tiingo", CAL[:2], gap_max_days=60)
    assert frame["high"].tolist() == [10.5, 11.5] and frame["close_raw"].tolist() == [20.0, 22.0]
