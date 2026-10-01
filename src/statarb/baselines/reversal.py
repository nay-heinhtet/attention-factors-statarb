"""Short-term reversal baseline (PLAN.md §9.3): buy recent losers, sell recent winners.

w_t ∝ −(r_k,i − mean over U_t of r_k), using the k-day return known at the close of t−1,
L1-normalized to gross exposure 1. Rebalanced daily.
"""

from __future__ import annotations

import pandas as pd

FEATURE = {1: "ret_1d", 5: "ret_5d", 21: "ret_21d"}


def reversal_weights(past_return: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """``past_return``: dates × assets k-day return dated at its close (calendar);
    ``universe``: holding days × assets boolean U_t. Returns target weights per holding day."""
    signal = past_return.shift(1).reindex(index=universe.index, columns=universe.columns).where(universe)
    centred = signal.sub(signal.mean(axis=1), axis=0)
    w = -centred.fillna(0.0)
    return w.div(w.abs().sum(axis=1).where(lambda s: s > 0), axis=0).fillna(0.0)
