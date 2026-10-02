"""Stage 2 (ablation): the paper's LongConv sequence model (PLAN.md §10.5; Fu et al. 2023).

y = 𝒦 ∗ u + D ⊙ u with one learnable kernel per channel, computed as a causal FFT
convolution. The kernel is squashed in the forward pass (an ℓ1 proximal step) and initialized
with geometric decay so different channels see short and long horizons.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LongConvSignal(nn.Module):
    def __init__(self, hidden: int = 32, length: int = 30, dropout: float = 0.1, squash: float = 1e-3):
        super().__init__()
        self.inp = nn.Linear(2, hidden)
        t = torch.arange(1, length + 1, dtype=torch.float32) / length
        h = torch.arange(1, hidden + 1, dtype=torch.float32) / hidden
        decay = torch.exp(-t[None, :] * (hidden / 2) ** h[:, None])
        self.kernel = nn.Parameter(torch.randn(hidden, length) * decay)
        self.D = nn.Parameter(torch.randn(hidden))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        self.squash = squash

    def squashed_kernel(self) -> torch.Tensor:
        k = self.kernel
        return torch.sign(k) * F.relu(k.abs() - self.squash)

    def conv(self, x: torch.Tensor) -> torch.Tensor:
        """x (n, L, 2) → hidden sequence (n, hidden, L); output step τ only sees inputs ≤ τ."""
        u = self.inp(x).transpose(1, 2)
        L = u.shape[-1]
        y = torch.fft.irfft(torch.fft.rfft(u, n=2 * L) * torch.fft.rfft(self.squashed_kernel(), n=2 * L), n=2 * L)
        return y[..., :L] + self.D[:, None] * u

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(F.gelu(self.conv(x))[..., -1])).squeeze(-1)
