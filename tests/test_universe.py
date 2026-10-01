import numpy as np
import pandas as pd
import pytest

from statarb.config import load_config
from statarb.data.universe import form_universe, formation_dates

DATES = pd.bdate_range("2019-01-01", "2020-12-31")
ASSETS = [f"yahoo:A{i:03d}" for i in range(260)]


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    ret = pd.DataFrame(rng.normal(0, 0.01, (len(DATES), len(ASSETS))), index=DATES, columns=ASSETS)
    close = 50 * (1 + ret).cumprod()
    level = np.tile(np.arange(1, len(ASSETS) + 1) * 1e6, (len(DATES), 1))  # asset i is i-th most liquid
    dv = pd.DataFrame(level * rng.uniform(0.9, 1.1, (len(DATES), 1)), index=DATES, columns=ASSETS)
    valid = close.notna()
    close.loc[:, "yahoo:A259"] = 4.0          # most liquid, but below the $5 filter
    ret.loc[: "2020-06-30", "yahoo:A258"] = np.nan  # second most liquid, but too little history
    return valid, close, ret, dv


def run(frames, months=("2020-09", "2020-12"), drop=frozenset(), truncate_after=None):
    valid, close, ret, dv = frames
    if truncate_after is not None:
        valid, close, ret, dv = (f.loc[:truncate_after] for f in frames)
    formations = formation_dates(DATES, *months)
    ucfg = load_config()["universe"]
    return form_universe(valid, close, ret, dv, lambda f: set(ASSETS), formations, set(drop), ucfg)


def test_size_ranking_and_filters(frames):
    u = run(frames)
    assert (u.groupby("month").size() == 200).all()
    picked = set(u.loc[u["month"] == "2020-12-01", "asset_id"])
    assert "yahoo:A259" not in picked  # $5 filter
    assert "yahoo:A258" not in picked  # 240-of-252 valid returns
    assert "yahoo:A257" in picked and "yahoo:A000" not in picked  # ranked by dollar volume
    assert u.loc[u["month"] == "2020-12-01", "rank"].tolist() == list(range(1, 201))


def test_dual_class_drop(frames):
    u = run(frames, drop={"yahoo:A257"})
    assert "yahoo:A257" not in set(u["asset_id"])


def test_formation_uses_last_trading_day_of_prior_month():
    (month, f), = formation_dates(DATES, "2020-03", "2020-03")
    assert month == pd.Timestamp("2020-03-01") and f == pd.Timestamp("2020-02-28")


def test_truncation_invariance(frames):
    full = run(frames, months=("2020-10", "2020-10"))
    cut = run(frames, months=("2020-10", "2020-10"), truncate_after="2020-09-30")
    pd.testing.assert_frame_equal(full, cut)
