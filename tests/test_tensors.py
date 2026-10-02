import numpy as np
import pandas as pd
import torch

from statarb.tensors import build_tensors, pca_weights
from statarb.train import make_batch

CAL = pd.bdate_range("2018-01-01", periods=320)
ASSETS = [f"a{i}" for i in range(25)]


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    ret = pd.DataFrame(rng.normal(0, 0.02, (len(CAL), len(ASSETS))), index=CAL, columns=ASSETS)
    feats = {f"f{j}": pd.DataFrame(rng.normal(size=(len(CAL), len(ASSETS))), index=CAL, columns=ASSETS)
             for j in range(3)}
    U = pd.DataFrame(True, index=CAL[280:], columns=ASSETS)
    U.iloc[:, :3] = False
    return ret, feats, U


def test_features_at_t_come_from_t_minus_1():
    ret, feats, U = synthetic()
    base = build_tensors(CAL, ret, feats, U, k_max=3)
    k = 10                                  # holding day index; its calendar date is CAL[290]
    t = U.index[k]
    shocked = {n: f.copy() for n, f in feats.items()}
    for f in shocked.values():
        f.loc[f.index >= t] = np.random.default_rng(9).normal(size=f.loc[f.index >= t].shape)
    moved = build_tensors(CAL, ret, shocked, U, k_max=3)
    assert torch.equal(base.Xn[k], moved.Xn[k])
    assert not torch.equal(base.Xn[k + 1], moved.Xn[k + 1])


def test_pca_weights_use_only_the_past_252_days():
    ret, feats, U = synthetic()
    base = build_tensors(CAL, ret, feats, U, k_max=3)
    k = 10
    shocked = ret.copy()
    shocked.loc[shocked.index >= U.index[k]] *= 5
    moved = build_tensors(CAL, shocked, feats, U, k_max=3)
    assert torch.equal(base.pca_wF[k], moved.pca_wF[k])


def test_residual_window_is_rows_t_minus_30_to_t_minus_1():
    ret, feats, U = synthetic()
    data = build_tensors(CAL, ret, feats, U, k_max=3)
    h = np.array([5, 6, 7])
    batch = make_batch(data, h, 30, [0, 1, 2], "cpu")
    c = data.hold_cal_idx[6]
    expected = torch.nan_to_num(data.R[c - 30: c]) * data.U[6][None, :]
    assert torch.equal(batch["R_hist"][1], expected)
    assert torch.equal(batch["R_t"][1], torch.nan_to_num(data.R[c]) * data.U[6])
    assert (batch["X"][1][~data.U[6]] == 0).all()


def test_pca_eigenportfolios_are_orthonormal_after_rescaling():
    rng = np.random.default_rng(0)
    window = rng.normal(0, 0.02, (252, 15)) + rng.normal(0, 0.01, (252, 1))
    w = pca_weights(window, 4)
    v = w * window.std(axis=0, ddof=1)  # undo the 1/σ scaling → eigenvectors
    assert np.allclose(v @ v.T, np.eye(4), atol=1e-8)
