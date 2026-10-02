"""Stage 1: factor portfolio weights ω^F and the residual projection (PLAN.md §10.2–10.4).

Attention factors: K learned query vectors attend over each asset's embedded characteristics;
the softmax over assets *is* each factor's portfolio weight (rows of ω^F sum to 1). Loadings and
residuals follow from ω^F alone:

    βᵀ = ω^Fᵀ (ω^F ω^Fᵀ + λ I)⁻¹,    ε = R − βᵀ ω^F R

so the residuals carry (almost) no exposure to any factor portfolio. The A×A matrix ω^ε is never
formed; everything goes through the K×K Gram matrix.
"""

from __future__ import annotations

import math

import torch
from torch import nn

MASKED = -1e9


class AttentionFactors(nn.Module):
    def __init__(self, n_features: int, K: int, d: int = 32):
        super().__init__()
        self.embed = nn.Linear(n_features, d, bias=False)  # a bias would cancel in the softmax
        self.queries = nn.Parameter(torch.randn(K, d))
        self.scale = 1.0 / math.sqrt(d)

    def forward(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """X (B, A, M), U (B, A) → ω^F (B, K, A); zero weight outside the universe."""
        logits = torch.einsum("kd,bad->bka", self.queries, self.embed(X)) * self.scale
        logits = logits.masked_fill(~U[:, None, :], MASKED)
        return torch.softmax(logits, dim=-1) * U[:, None, :]


def loadings(wF: torch.Tensor, ridge_rel: float) -> torch.Tensor:
    """β (B, K, A) = (G + λI)⁻¹ ω^F with G = ω^F ω^Fᵀ and λ = ridge_rel · mean(diag G), in float64."""
    w64 = wF.double()
    G = w64 @ w64.transpose(1, 2)
    lam = (ridge_rel * torch.diagonal(G, dim1=1, dim2=2).mean(-1)).detach()
    eye = torch.eye(G.shape[-1], dtype=G.dtype, device=G.device)
    return torch.linalg.solve(G + lam[:, None, None] * eye, w64).to(wF.dtype)


def residuals(R: torch.Tensor, wF: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """ε = R − βᵀ ω^F R for returns R (B, A) or a window R (B, L, A)."""
    if R.dim() == 2:
        return R - torch.einsum("bka,bk->ba", beta, torch.einsum("bka,ba->bk", wF, R))
    F = torch.einsum("bka,bla->blk", wF, R)
    return R - torch.einsum("bka,blk->bla", beta, F)


def map_to_assets(w_port: torch.Tensor, wF: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """(ω^ε)ᵀ ω^port = ω^port − ω^Fᵀ (β ω^port): residual-portfolio weights → stock weights."""
    return w_port - torch.einsum("bka,bk->ba", wF, torch.einsum("bka,ba->bk", beta, w_port))
