"""Stage 2 (primary): a shared LSTM over each asset's 30-day residual history (PLAN.md §10.5).

Input per asset and day: [ε_τ / s, cumsum(ε)_τ / (s·√L)] for the L days before the decision,
i.e. the scaled residual return and the scaled cumulative residual ("spread level").
Output: one score per asset, the raw residual-portfolio weight.
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMSignal(nn.Module):
    # Sequences per cuDNN call. One call over a whole block (~25k sequences) can request a
    # multi-GB workspace for larger hidden sizes; sequences are independent, so chunking is exact.
    CHUNK = 8192

    def __init__(self, hidden: int = 32, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden, layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (n, L, 2) → scores (n,)."""
        last = torch.cat([self.lstm(part)[1][0][-1] for part in torch.split(x, self.CHUNK)])
        return self.head(self.drop(last)).squeeze(-1)
