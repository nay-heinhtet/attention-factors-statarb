import torch

from statarb.config import load_config
from statarb.models.arbitrage import ArbitrageModel
from statarb.models.attention_factor import AttentionFactors, loadings, map_to_assets, residuals


def softmax_weights(B=4, K=5, A=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(B, K, A, generator=g, dtype=torch.float64) * 2, dim=-1)


def test_residuals_have_no_factor_exposure():
    wF = softmax_weights()
    R = torch.randn(4, 40, dtype=torch.float64) * 0.02
    beta = loadings(wF, 1e-6)
    eps = residuals(R, wF, beta)
    F = torch.einsum("bka,ba->bk", wF, R)
    assert torch.einsum("bka,ba->bk", wF, eps).norm() / F.norm() < 1e-2


def test_mapping_matches_explicit_projection():
    wF = softmax_weights(B=2, K=3, A=12)
    beta = loadings(wF, 1e-3)
    w_port = torch.randn(2, 12, dtype=torch.float64)
    fast = map_to_assets(w_port, wF, beta)
    for b in range(2):
        omega_eps = torch.eye(12, dtype=torch.float64) - beta[b].T @ wF[b]   # I − βᵀ ω^F
        assert torch.allclose(fast[b], omega_eps.T @ w_port[b], atol=1e-6)


def test_residual_window_matches_per_day_projection():
    wF = softmax_weights(B=2, K=3, A=12)
    beta = loadings(wF, 1e-3)
    R_hist = torch.randn(2, 30, 12, dtype=torch.float64)
    window = residuals(R_hist, wF, beta)
    for b in range(2):
        for t in (0, 17, 29):
            assert torch.allclose(window[b, t], residuals(R_hist[b, t][None], wF[b][None], beta[b][None])[0])


def model_and_inputs(seed=0, A=30):
    cfg = load_config()
    torch.manual_seed(seed)
    model = ArbitrageModel(15, cfg["model"]).eval()
    X = torch.rand(3, A, 15) - 0.5
    U = torch.ones(3, A, dtype=torch.bool)
    U[:, -5:] = False
    R_hist = torch.randn(3, 30, A) * 0.02 * U[:, None, :]
    R_t = torch.randn(3, A) * 0.02 * U
    return model, X, U, R_hist, R_t


def test_assets_outside_universe_get_zero_weight_and_factor_weight():
    model, X, U, R_hist, R_t = model_and_inputs()
    out = model(X, U, R_hist, R_t, scale=0.015)
    assert (out["w"][~U] == 0).all() and (out["wF"].permute(0, 2, 1)[~U] == 0).all()
    assert torch.allclose(out["w"].abs().sum(-1), torch.ones(3), atol=1e-5)


def test_permutation_equivariance():
    model, X, U, R_hist, R_t = model_and_inputs()
    perm = torch.randperm(X.shape[1])
    a = model(X, U, R_hist, R_t, 0.015)["w"]
    b = model(X[:, perm], U[:, perm], R_hist[:, :, perm], R_t[:, perm], 0.015)["w"]
    assert torch.allclose(a[:, perm], b, atol=1e-5)


def test_cross_sectionally_constant_feature_has_no_effect():
    # PLAN.md D4: a feature equal for every asset adds the same logit to every asset → cancels.
    torch.manual_seed(0)
    factors = AttentionFactors(3, K=4)
    X = torch.rand(2, 25, 3)
    U = torch.ones(2, 25, dtype=torch.bool)
    shifted = X.clone()
    shifted[..., 2] = torch.tensor([[0.3], [-0.7]])  # constant across assets, different per day
    base = X.clone()
    base[..., 2] = 0.0
    assert torch.allclose(factors(base, U), factors(shifted, U), atol=1e-6)
