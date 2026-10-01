import numpy as np
import pandas as pd
import pytest

from statarb.backtest import (Market, bootstrap_sharpe_ci, dead_mask, max_drawdown, run_backtest, sharpe,
                              sharpe_se)

CAL = pd.bdate_range("2020-01-01", periods=6)


def market(ret: dict, valid: dict | None = None) -> Market:
    r = pd.DataFrame(ret, index=CAL[: len(next(iter(ret.values())))])
    v = r.notna() if valid is None else pd.DataFrame(valid, index=r.index)
    return Market(r.index, r.where(v), v, pd.Series(0.0, index=r.index))


def test_hand_computed_example():
    m = market({"A": [0, 0.01, 0.02, -0.01], "B": [0, -0.02, 0.01, 0.03], "C": [0, 0.0, 0.05, 0.0]})
    w = pd.DataFrame({"A": [0.5, 0.5, 0.0], "B": [-0.5, -0.25, -0.5], "C": [0.0, 0.25, 0.5]}, index=CAL[1:4])
    res = run_backtest(w, m, tc_bps=10, short_bps=2)
    assert res["gross"].tolist() == pytest.approx([0.5 * 0.01 + 0.5 * 0.02, 0.5 * 0.02 - 0.25 * 0.01 + 0.25 * 0.05,
                                                   -0.5 * 0.03 + 0.0])
    assert res["turnover"].tolist() == pytest.approx([1.0, 0.5, 1.0])
    assert res["cost"].tolist() == pytest.approx([1e-3 * 1.0 + 2e-4 * 0.5, 1e-3 * 0.5 + 2e-4 * 0.25,
                                                  1e-3 * 1.0 + 2e-4 * 0.5])
    assert (res["net"] == res["gross"] - res["cost"]).all()


def test_alignment_cheating_vs_honest():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=2000)
    r = pd.DataFrame(rng.normal(0, 0.01, (2000, 20)), index=dates)
    m = Market(dates, r, r.notna(), pd.Series(0.0, index=dates))
    cheat = np.sign(r).iloc[1:] / 20          # uses the return it is about to earn
    honest = np.sign(r).shift(1).iloc[1:] / 20  # uses yesterday's return
    assert sharpe(run_backtest(cheat, m, 0, 0)["net"]) > 10
    assert abs(sharpe(run_backtest(honest.fillna(0), m, 0, 0)["net"])) < 1.5


def test_position_frozen_through_halt():
    # B halts on days 2-3 (no valid close); its 3-day return books on day 4.
    m = market({"A": [0, 0.0, 0.0, 0.0, 0.0], "B": [0, 0.01, np.nan, np.nan, -0.5]},
               valid={"A": [True] * 5, "B": [True, True, False, False, True]})
    w = pd.DataFrame({"A": [0.0] * 4, "B": [1.0, 0.0, 0.0, 0.0]}, index=CAL[1:5])
    res, pos = run_backtest(w, m, 0, 0, return_positions=True)
    assert pos["B"].tolist() == [1.0, 0.0, 0.0, 0.0]  # still tradable at day 1's close → can exit before the halt
    w2 = pd.DataFrame({"A": [0.0] * 4, "B": [1.0, 1.0, 0.0, 0.0]}, index=CAL[1:5])
    res2, pos2 = run_backtest(w2, m, 0, 0, return_positions=True)
    assert pos2["B"].tolist() == [1.0, 1.0, 1.0, 1.0]  # held into the halt: stuck until it trades again
    assert res2["gross"].iloc[3] == pytest.approx(-0.5)  # the collapse is booked


def test_permanent_end_cash_out_without_cost():
    m = market({"A": [0, 0.01, 0.02, np.nan, np.nan, np.nan]},
               valid={"A": [True, True, True, False, False, False]})
    w = pd.DataFrame({"A": [1.0] * 5}, index=CAL[1:6])
    res, pos = run_backtest(w, m, tc_bps=10, short_bps=0, return_positions=True)
    assert pos["A"].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]  # held on t_last+1, removed from t_last+2
    assert res["gross"].tolist() == pytest.approx([0.01, 0.02, 0.0, 0.0, 0.0])
    assert res["turnover"].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0])  # removal is free


def test_dead_mask_long_gap():
    valid = np.array([True, True, False, False, False, True])[:, None]
    assert dead_mask(valid, gap_max_days=2)[:, 0].tolist() == [False, False, False, True, True, False]
    assert not dead_mask(valid, gap_max_days=3).any()


def test_exec_lag_shifts_targets_by_one_day():
    m = market({"A": [0, 0.01, 0.02, 0.03]})
    w = pd.DataFrame({"A": [1.0, 0.0, 0.0]}, index=CAL[1:4])
    assert run_backtest(w, m, 0, 0, exec_lag=1)["gross"].tolist() == pytest.approx([0.0, 0.02, 0.0])


def test_metrics():
    x = np.array([0.01, -0.005, 0.002, 0.004, -0.001] * 100)
    assert sharpe(x) == pytest.approx(x.mean() / x.std(ddof=1) * np.sqrt(252))
    lo, hi = bootstrap_sharpe_ci(x, n=500)
    assert lo < sharpe(x) < hi and sharpe_se(x) > 0
    assert max_drawdown([0.1, -0.5, 0.2]) == pytest.approx(-0.5)
