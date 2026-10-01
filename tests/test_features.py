import numpy as np
import pandas as pd
import pytest

from statarb.config import load_config
from statarb.features import compute_features, feature_names, rank_normalize

FCFG = load_config()["features"]


def inputs(close: pd.DataFrame, market: pd.Series, volume: float = 1e6) -> dict:
    valid = close.notna()
    ret = (close / close.ffill().shift(1) - 1).where(valid)
    return dict(close_adj=close, valid=valid, ret=ret, high=close * 1.01, low=close * 0.99,
                dollar_volume=close * volume, market_ret=market, fcfg=FCFG)


def test_known_values_on_geometric_path():
    dates = pd.bdate_range("2015-01-01", periods=900)
    g = 0.001
    close = pd.DataFrame({"A": 100 * (1 + g) ** np.arange(900)}, index=dates)
    market = pd.Series(np.random.default_rng(0).normal(0, 0.01, 900), index=dates)
    f = compute_features(**inputs(close, market))
    t = dates[-1]
    assert f["ret_5d"].at[t, "A"] == pytest.approx((1 + g) ** 5 - 1)
    assert f["mom_12_2"].at[t, "A"] == pytest.approx((1 + g) ** 231 - 1)
    assert f["mom_36_13"].at[t, "A"] == pytest.approx((1 + g) ** 504 - 1)
    assert f["rel_52w_high"].at[t, "A"] == pytest.approx(1.0)
    assert f["vol_63d"].at[t, "A"] == pytest.approx(0.0, abs=1e-12)
    assert f["hl_range_21d"].at[t, "A"] == pytest.approx(np.log(1.01 / 0.99))
    assert np.isnan(f["mom_36_13"].iloc[700]["A"])  # needs 756 days of history


def test_beta_and_idiosyncratic_vol():
    dates = pd.bdate_range("2015-01-01", periods=400)
    market = pd.Series(np.random.default_rng(1).normal(0, 0.01, 400), index=dates)
    close = pd.DataFrame({"A": 50 * (1 + 2 * market.fillna(0)).cumprod()}, index=dates)
    f = compute_features(**inputs(close, market))
    assert f["beta_252d"].iloc[-1]["A"] == pytest.approx(2.0, rel=1e-6)
    assert f["ivol_63d"].iloc[-1]["A"] == pytest.approx(0.0, abs=1e-8)


def random_panel(seed: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-01", periods=1100)
    market = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    rets = rng.normal(0, 0.02, (len(dates), 30)) + market.to_numpy()[:, None]
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates, columns=[f"S{i}" for i in range(30)])
    close = close.mask(rng.random(close.shape) < 0.02)       # scattered missing days
    close.iloc[:300, :5] = np.nan                             # late listings
    return close, market


@pytest.mark.parametrize("cut", [400, 800, 1000, 1050, 1099])
def test_truncation_invariance(cut):
    close, market = random_panel()
    full = compute_features(**inputs(close, market))
    t_star = close.index[cut]
    part = compute_features(**inputs(close.loc[:t_star], market.loc[:t_star]))
    for name in feature_names(load_config(), apply_drops=False):
        pd.testing.assert_frame_equal(full[name].loc[:t_star], part[name], check_freq=False, obj=name)


def test_future_perturbation_does_not_move_the_past():
    close, market = random_panel()
    t_star = close.index[700]
    shocked = close.copy()
    shocked.loc[shocked.index > t_star] *= 3.0
    a, b = compute_features(**inputs(close, market)), compute_features(**inputs(shocked, market))
    for name in a:
        pd.testing.assert_frame_equal(a[name].loc[:t_star], b[name].loc[:t_star], check_freq=False, obj=name)


def test_rank_normalize():
    values = pd.DataFrame({"x": [3.0, 1.0, 2.0, np.nan, 5.0], "y": [1.0, 1.0, 1.0, 1.0, 1.0],
                           "z": [np.nan, np.nan, 7.0, np.nan, np.nan]})
    out = rank_normalize(values)
    assert out["x"].tolist() == pytest.approx([1 / 6, -0.5, -1 / 6, 0.0, 0.5])
    assert (out["y"] == 0.0).all()  # all ties → median
    assert (out["z"] == 0.0).all()  # fewer than 2 values → median
    assert out.abs().max().max() <= 0.5


def test_rank_normalize_uses_only_given_rows():
    universe = pd.DataFrame({"x": [10.0, 20.0, 30.0]}, index=["a", "b", "c"])
    bigger = pd.concat([universe, pd.DataFrame({"x": [-100.0, 100.0]}, index=["d", "e"])])
    assert rank_normalize(universe)["x"].tolist() == [-0.5, 0.0, 0.5]
    assert rank_normalize(bigger).loc[["a", "b", "c"], "x"].tolist() != [-0.5, 0.0, 0.5]
