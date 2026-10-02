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
