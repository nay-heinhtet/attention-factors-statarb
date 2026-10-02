"""The full model: factors → residuals → sequence signal → stock weights (PLAN.md §10.2–10.6)."""

from __future__ import annotations

import math

import torch
from torch import nn

from statarb.models.attention_factor import AttentionFactors, loadings, map_to_assets, residuals
from statarb.models.longconv import LongConvSignal
from statarb.models.lstm_residual import LSTMSignal


class ArbitrageModel(nn.Module):
    def __init__(self, n_features: int, mcfg: dict):
        super().__init__()
        self.kind = mcfg["factors"]
        self.K = mcfg["K"]
        self.ridge_rel = mcfg["ridge_rel"]
        self.factors = AttentionFactors(n_features, self.K, mcfg["d"]) if self.kind == "attention" else None
        if mcfg["sequence"] == "lstm":
            self.signal = LSTMSignal(mcfg["hidden"], mcfg["layers"], mcfg["dropout"])
        else:
            self.signal = LongConvSignal(mcfg["hidden"], mcfg["lookback"], mcfg["dropout"], mcfg["squash"])

    def factor_parameters(self):
        return [] if self.factors is None else list(self.factors.parameters())

    def forward(self, X, U, R_hist, R_t, scale: float, pca_wF=None) -> dict[str, torch.Tensor]:
        """X (B, A, M), U (B, A) bool, R_hist (B, L, A), R_t (B, A) → weights and residuals.

        R_hist and R_t must already be zero outside U (and where returns are missing).
        """
        wF = self.factors(X, U) if self.factors is not None else pca_wF[:, : self.K]
        beta = loadings(wF, self.ridge_rel)
        eps_hist = residuals(R_hist, wF, beta)                       # (B, L, A)
        eps_t = residuals(R_t, wF, beta)                             # realized, for explained variance

        seq = eps_hist.permute(0, 2, 1)[U]                           # (n_active, L)
        L = seq.shape[-1]
        x = torch.stack([seq / scale, torch.cumsum(seq, dim=-1) / (scale * math.sqrt(L))], dim=-1)
        w_port = torch.zeros_like(U, dtype=R_t.dtype)
        w_port[U] = self.signal(x)

        w = map_to_assets(w_port, wF, beta) * U
        w = w / (w.abs().sum(-1, keepdim=True) + 1e-8)
        return {"w": w, "eps_t": eps_t, "wF": wF}
