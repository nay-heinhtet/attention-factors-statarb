import json

import pandas as pd

from statarb.walkforward import grid, override, select


def table(rows):
    return pd.DataFrame([{"params": json.dumps(p, sort_keys=True), "sr_net": s} for p, s in rows])


def test_best_wins_when_clearly_ahead():
    t = table([({"K": 5, "lr": 0.001}, 1.0), ({"K": 10, "lr": 0.003}, 0.5)])
    assert select(t, [["lr", 0.003], ["K", 10]], margin=0.2) == {"K": 5, "lr": 0.001}


def test_tie_break_prefers_paper_defaults_within_margin():
    t = table([({"K": 5, "lr": 0.001}, 1.00), ({"K": 20, "lr": 0.003}, 0.85), ({"K": 10, "lr": 0.003}, 0.82)])
    assert select(t, [["lr", 0.003], ["K", 10]], margin=0.2) == {"K": 10, "lr": 0.003}


def test_selection_uses_the_mean_over_seeds():
    t = table([({"K": 5}, 2.0), ({"K": 5}, -1.0), ({"K": 10}, 0.6), ({"K": 10}, 0.6)])
    assert select(t, [], margin=0.0) == {"K": 10}


def test_grid_and_override():
    assert len(grid({"K": [5, 10, 20], "lr": [1e-3, 3e-3]})) == 6
    assert override("pca", {"K": 5}) == {"model": {"factors": "pca", "K": 5}}
    assert override("reversal", {"lookback": 21}) == {"reversal": {"lookback": 21}}


def test_test_years_cover_2019_to_data_end_with_8_year_windows():
    from statarb.config import load_config
    from statarb.walkforward import test_years

    years = test_years(load_config())
    assert years[0] == (2019, "2011-01-01", "2018-12-31", "2019-01-01", "2019-12-31")
    assert years[-1] == (2026, "2018-01-01", "2025-12-31", "2026-01-01", "2026-08-31")
    assert len(years) == 8


def test_ensemble_renormalizes():
    from statarb.walkforward import ensemble

    a = pd.DataFrame({"x": [0.5, 1.0], "y": [-0.5, 0.0]})
    b = pd.DataFrame({"x": [0.5, -1.0], "y": [-0.5, 0.0]})
    e = ensemble([a, b])
    assert e.loc[0].tolist() == [0.5, -0.5] and e.loc[1].tolist() == [0.0, 0.0]
