import numpy as np
import pytest
import torch

from statarb.train import explained_variance, portfolio_stats


def test_portfolio_stats_match_numpy():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(6, 10))
    w /= np.abs(w).sum(1, keepdims=True)
    R = rng.normal(0, 0.02, (6, 10))
    stats = portfolio_stats(torch.tensor(w), torch.tensor(R), tc_bps=5, short_bps=1)
    gross = (w[1:] * R[1:]).sum(1)
    turnover = np.abs(w[1:] - w[:-1]).sum(1)
    short = np.clip(-w[1:], 0, None).sum(1)
    assert stats["gross"].numpy() == pytest.approx(gross)
    assert stats["turnover"].numpy() == pytest.approx(turnover)
    assert stats["net"].numpy() == pytest.approx(gross - 5e-4 * turnover - 1e-4 * short)


def test_leaving_the_universe_counts_as_a_sale():
    w = torch.tensor([[0.3, -0.7, 0.0], [0.0, -0.5, 0.5]])
    assert portfolio_stats(w, torch.zeros(2, 3), 5, 1)["turnover"].item() == pytest.approx(0.3 + 0.2 + 0.5)


def test_explained_variance_matches_numpy_with_mask():
    rng = np.random.default_rng(1)
    R = rng.normal(0, 0.02, (60, 4))
    eps = 0.5 * R + rng.normal(0, 0.005, (60, 4))
    ok = np.ones((60, 4), bool)
    ok[:50, 3] = False  # asset 3 has only 10 observations → excluded (min_days 20)
    got = explained_variance(torch.tensor(eps), torch.tensor(R), torch.tensor(ok), min_days=20)
    want = np.mean([1 - eps[:, i].var() / R[:, i].var() for i in range(3)])
    assert got.item() == pytest.approx(want, rel=1e-6)
