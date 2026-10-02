"""Model-ready tensors (PLAN.md §10.1).

Everything is indexed by *holding day* t and built only from data dated ≤ t−1:

* ``U[t]``: the month's universe ∩ assets valid at the close of t−1;
* ``Xn[t]``: the 15 characteristics at t−1, rank-normalized within U_t (0 outside it);
* ``pca_wF[t]``: PCA eigenportfolio weights from the 252 returns t−252..t−1 of U_t;
* ``R``: daily returns on the full calendar, so a model can read the 30-day residual window
  (rows t−30..t−1) and the realized return R_t.

The ``noise`` variant replaces returns with iid noise of the same volatility per asset, keeping
the universe and validity masks; any strategy that "works" on it reveals leakage (PLAN.md §13).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from statarb.data.panel import load_panel, wide
from statarb.data.universe import daily_universe, load_universe
from statarb.data.yahoo import load_prices, master_calendar
from statarb.features import compute_features, feature_names, rank_normalize
from statarb.paths import Paths, project_paths

log = logging.getLogger(__name__)


@dataclass
class ArbData:
    holding_days: pd.DatetimeIndex
    hold_cal_idx: np.ndarray        # calendar row of each holding day
    asset_ids: list[str]
    feature_names: list[str]
    R: torch.Tensor                 # (T_cal, A) float32, NaN where not valid
    Xn: torch.Tensor                # (T_hold, A, M) float32
    U: torch.Tensor                 # (T_hold, A) bool
    pca_wF: torch.Tensor | None     # (T_hold, K_max, A) float32

    def to(self, device) -> "ArbData":
        move = lambda t: None if t is None else t.to(device)  # noqa: E731
        return ArbData(self.holding_days, self.hold_cal_idx, self.asset_ids, self.feature_names,
                       move(self.R), move(self.Xn), move(self.U), move(self.pca_wF))

    def day_range(self, start, end) -> np.ndarray:
        """Indices of holding days in [start, end]."""
        days = self.holding_days
        return np.flatnonzero((days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end)))


def pca_weights(window: np.ndarray, k_max: int) -> np.ndarray:
    """Avellaneda–Lee eigenportfolios: top-k eigenvectors of the correlation matrix, divided by σ."""
    x = np.nan_to_num(window)
    sd = x.std(axis=0, ddof=1)
    ok = sd > 1e-10
    z = (x[:, ok] - x[:, ok].mean(axis=0)) / sd[ok]
    corr = z.T @ z / (len(z) - 1)
    vals, vecs = np.linalg.eigh(corr)
    top = vecs[:, np.argsort(vals)[::-1][:k_max]]
    w = np.zeros((k_max, x.shape[1]))
    w[: top.shape[1], ok] = (top / sd[ok, None]).T
    return w


def build_tensors(
    calendar: pd.DatetimeIndex,
    ret: pd.DataFrame,
    raw_features: dict[str, pd.DataFrame],
    universe_daily: pd.DataFrame,
    k_max: int = 20,
    pca_window: int = 252,
) -> ArbData:
    """Assemble ArbData from calendar-indexed returns/features and the daily universe U_t."""
    assets = list(universe_daily.columns[universe_daily.any(axis=0)])
    days = universe_daily.index
    hold_cal_idx = calendar.get_indexer(days)
    names = list(raw_features)
    R = ret.reindex(index=calendar, columns=assets).to_numpy(dtype=np.float32, copy=True)
    F = np.stack([raw_features[n].reindex(index=calendar, columns=assets).to_numpy(dtype=np.float32) for n in names], -1)
    U = universe_daily[assets].to_numpy(dtype=bool, copy=True)

    Xn = np.zeros((len(days), len(assets), len(names)), dtype=np.float32)
    wF = np.zeros((len(days), k_max, len(assets)), dtype=np.float32)
    for k, c in enumerate(hold_cal_idx):
        cols = np.flatnonzero(U[k])
        if len(cols) == 0:
            continue
        today = pd.DataFrame(F[c - 1, cols, :], columns=names)  # characteristics at t−1
        Xn[k, cols] = rank_normalize(today).to_numpy(dtype=np.float32)
        if c >= pca_window:
            wF[k][:, cols] = pca_weights(R[c - pca_window: c, cols], k_max)
    return ArbData(days, hold_cal_idx, assets, names, torch.from_numpy(R), torch.from_numpy(Xn),
                   torch.from_numpy(U), torch.from_numpy(wF))


def noise_panel(ret: pd.DataFrame, close_adj: pd.DataFrame, valid: pd.DataFrame, seed: int):
    """iid normal returns with each asset's real volatility, on the real validity mask."""
    rng = np.random.default_rng(seed)
    sd = ret.std()
    fake = pd.DataFrame(rng.standard_normal(ret.shape), index=ret.index, columns=ret.columns) * sd
    fake = fake.where(valid)
    first = close_adj.where(valid).bfill().iloc[0].fillna(100.0)
    fake_close = (1 + fake.fillna(0.0)).cumprod() * first
    return fake, fake_close.where(valid)


def tensors_path(paths: Paths, universe: str, variant: str) -> object:
    return paths.processed / f"tensors_{universe}_{variant}.pt"


def save(data: ArbData, path) -> None:
    torch.save({"holding_days": data.holding_days.to_numpy(), "hold_cal_idx": data.hold_cal_idx,
                "asset_ids": data.asset_ids, "feature_names": data.feature_names, "R": data.R,
                "Xn": data.Xn, "U": data.U, "pca_wF": data.pca_wF}, path)


def load(path) -> ArbData:
    d = torch.load(path, weights_only=False)
    return ArbData(pd.DatetimeIndex(d["holding_days"]), d["hold_cal_idx"], d["asset_ids"], d["feature_names"],
                   d["R"], d["Xn"], d["U"], d["pca_wF"])


def run(cfg: dict, args) -> None:
    paths = project_paths(cfg)
    calendar = master_calendar(cfg, paths)
    fields = ["close_adj", "valid", "ret", "high", "low", "dollar_volume"]
    panel = load_panel(paths, fields)
    frames = {f: wide(panel, f).reindex(calendar) for f in fields}
    frames["valid"] = frames["valid"].eq(True)
    variant = args.variant
    if variant == "noise":
        frames["ret"], frames["close_adj"] = noise_panel(frames["ret"], frames["close_adj"], frames["valid"], args.noise_seed)
        variant = f"noise{args.noise_seed}"
    spy = load_prices(paths, "SPY")["Adj Close"].reindex(calendar)
    raw = compute_features(**frames, market_ret=spy / spy.shift(1) - 1, fcfg=cfg["features"])
    raw = {n: raw[n] for n in feature_names(cfg, apply_drops=False)}

    universe = load_universe(paths, args.universe)
    U = daily_universe(universe, frames["valid"], cfg["dates"]["first_holding_day"], cfg["dates"]["data_end"])
    data = build_tensors(calendar, frames["ret"], raw, U)
    path = tensors_path(paths, args.universe, variant)
    save(data, path)
    log.info("tensors %s: %d holding days × %d assets × %d features → %s",
             path.name, len(data.holding_days), len(data.asset_ids), len(data.feature_names), path.name)
