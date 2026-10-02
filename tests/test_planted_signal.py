"""Positive and negative controls for the whole training loop (slow: trains real models)."""

import numpy as np
import pandas as pd
import pytest
import torch

from statarb.backtest import sharpe
from statarb.config import deep_merge, load_config
from statarb.tensors import ArbData
from statarb.train import fit, predict

N, T, LOOKBACK = 60, 1500, 30


def synthetic_data(phi: float, seed: int = 0) -> ArbData:
    """Returns from 3 factors with characteristic-driven loadings plus AR(1) residuals."""
    rng = np.random.default_rng(seed)
    chars = rng.normal(size=(N, 5))
    loadings = chars @ rng.normal(size=(5, 3)) * 0.5 + 1.0
    factors = rng.normal(0, 0.01, (T, 3))
    e = np.zeros((T, N))
    for t in range(1, T):
        e[t] = phi * e[t - 1] + rng.normal(0, 0.01, N)
    R = factors @ loadings.T + e
    ranks = (chars.argsort(0).argsort(0) / (N - 1) - 0.5).astype(np.float32)
    calendar = pd.bdate_range("2000-01-03", periods=T)
    hold = np.arange(LOOKBACK + 1, T)
    return ArbData(calendar[hold], hold, [f"s{i}" for i in range(N)], [f"c{j}" for j in range(5)],
                   torch.tensor(R, dtype=torch.float32), torch.tensor(np.broadcast_to(ranks, (len(hold), N, 5)).copy()),
                   torch.ones(len(hold), N, dtype=torch.bool), None)


def validation_gross_sharpe(phi: float) -> float:
    cfg = deep_merge(load_config(), {
        "features": {"groups": {"synthetic": [f"c{j}" for j in range(5)]},
                     "drop_groups": ["past_returns", "risk", "frictions"]},  # keep only the synthetic group
        "model": {"K": 3}, "train": {"epochs": 10, "device": "cpu"},
    })
    data = synthetic_data(phi)
    split = data.holding_days[1000]
    model, scale, _ = fit(data, cfg, data.holding_days[0], split, seed=0, device="cpu")
    w = predict(model, data, cfg, scale, split, data.holding_days[-1], device="cpu")
    R = data.R[data.hold_cal_idx[data.day_range(split, data.holding_days[-1])]].numpy()
    return sharpe((w.to_numpy() * R).sum(1))


@pytest.mark.slow
def test_model_learns_a_planted_residual_reversal():
    assert validation_gross_sharpe(phi=-0.3) > 2.0


@pytest.mark.slow
def test_model_finds_nothing_in_pure_noise():
    assert abs(validation_gross_sharpe(phi=0.0)) < 2.5
