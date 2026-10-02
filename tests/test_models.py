import torch

from statarb.config import deep_merge, load_config
from statarb.models.arbitrage import ArbitrageModel
from statarb.models.longconv import LongConvSignal
from statarb.models.lstm_residual import LSTMSignal


def test_signal_shapes():
    x = torch.randn(17, 30, 2)
    assert LSTMSignal()(x).shape == (17,)
    assert LongConvSignal()(x).shape == (17,)


def test_longconv_is_causal():
    torch.manual_seed(0)
    conv = LongConvSignal(hidden=8, length=30).eval()
    x = torch.randn(4, 30, 2)
    y = conv.conv(x)
    for tau in (5, 17, 29):
        x2 = x.clone()
        x2[:, tau:] += torch.randn_like(x2[:, tau:])
        assert torch.allclose(conv.conv(x2)[..., :tau], y[..., :tau], atol=1e-5)
        assert not torch.allclose(conv.conv(x2)[..., tau], y[..., tau])


def test_squash_zeroes_small_kernel_entries():
    conv = LongConvSignal(hidden=4, length=10, squash=0.5)
    with torch.no_grad():
        conv.kernel.copy_(torch.linspace(-1, 1, 40).reshape(4, 10))
    k = conv.squashed_kernel()
    assert (k[conv.kernel.abs() <= 0.5] == 0).all()
    assert torch.allclose(k[conv.kernel > 0.5], conv.kernel[conv.kernel > 0.5] - 0.5)


def test_weights_do_not_depend_on_the_realized_return():
    for sequence in ("lstm", "longconv"):
        torch.manual_seed(0)
        model = ArbitrageModel(15, deep_merge(load_config()["model"], {"sequence": sequence})).eval()
        X, U = torch.rand(2, 20, 15) - 0.5, torch.ones(2, 20, dtype=torch.bool)
        R_hist = torch.randn(2, 30, 20) * 0.02
        w1 = model(X, U, R_hist, torch.zeros(2, 20), 0.015)["w"]
        w2 = model(X, U, R_hist, torch.randn(2, 20) * 0.5, 0.015)["w"]
        assert torch.equal(w1, w2)


def test_pca_factor_variant_uses_given_weights():
    torch.manual_seed(0)
    model = ArbitrageModel(15, deep_merge(load_config()["model"], {"factors": "pca", "K": 3})).eval()
    assert model.factor_parameters() == []
    X, U = torch.rand(2, 20, 15), torch.ones(2, 20, dtype=torch.bool)
    out = model(X, U, torch.randn(2, 30, 20) * 0.02, torch.zeros(2, 20), 0.015, pca_wF=torch.randn(2, 20, 20))
    assert out["wF"].shape == (2, 3, 20)
