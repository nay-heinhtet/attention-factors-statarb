"""Joint training on net-of-cost Sharpe + explained variance, and prediction (PLAN.md §10.7–10.9).

A training step sees one block of 127 consecutive holding days. Day 0 only supplies the previous
weights, so turnover — and therefore cost — is charged on the 126 days that follow. Weights live
in the global asset space, so an asset leaving the universe counts as a sale.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from statarb.features import feature_names
from statarb.models.arbitrage import ArbitrageModel
from statarb.tensors import ArbData
from statarb.utils import get_device, set_seed

log = logging.getLogger(__name__)


def make_batch(data: ArbData, h: np.ndarray, lookback: int, feat_idx: list[int], device) -> dict:
    """Inputs for holding days ``h`` (consecutive indices); every tensor already on ``device``."""
    h_t = torch.as_tensor(h, device=device)
    c = torch.as_tensor(data.hold_cal_idx[h], device=device)
    U = data.U[h_t]
    window = c[:, None] + torch.arange(-lookback, 0, device=device)[None, :]
    R_hist = torch.nan_to_num(data.R[window]) * U[:, None, :]
    R_raw = data.R[c]
    ok = ~torch.isnan(R_raw) & U
    batch = {"X": data.Xn[h_t][..., feat_idx], "U": U, "R_hist": R_hist,
             "R_t": torch.nan_to_num(R_raw) * ok, "ok": ok}
    batch["pca_wF"] = data.pca_wF[h_t] if data.pca_wF is not None else None
    return batch


def explained_variance(eps: torch.Tensor, R: torch.Tensor, ok: torch.Tensor, min_days: int) -> torch.Tensor:
    """Mean over assets (with ≥ min_days observations) of 1 − Var(ε_i)/Var(R_i), masked and centred."""
    m = ok.to(R.dtype)
    n = m.sum(0)

    def var(x):
        mean = (x * m).sum(0) / n.clamp(min=1)
        return (((x - mean) * m) ** 2).sum(0) / n.clamp(min=1)

    keep = n >= min_days
    if not keep.any():
        return R.new_zeros(())
    return (1 - var(eps) / (var(R) + 1e-12))[keep].mean()


def portfolio_stats(w: torch.Tensor, R_t: torch.Tensor, tc_bps: float, short_bps: float) -> dict[str, torch.Tensor]:
    """Daily gross/net returns for weights w (B+1, A); day 0 only provides the previous weights."""
    w1, w0 = w[1:], w[:-1]
    gross = (w1 * R_t[1:]).sum(-1)
    turnover = (w1 - w0).abs().sum(-1)
    short = torch.relu(-w1).sum(-1)
    net = gross - tc_bps / 1e4 * turnover - short_bps / 1e4 * short
    return {"gross": gross, "net": net, "turnover": turnover, "short": short}


def block_loss(out: dict, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    stats = portfolio_stats(out["w"], batch["R_t"], cfg["costs"]["tc_bps"], cfg["costs"]["short_bps"])
    net = stats["net"]
    sr = net.mean() / (net.std(unbiased=False) + 1e-8)
    loss = -sr
    ev = explained_variance(out["eps_t"][1:], batch["R_t"][1:], batch["ok"][1:], cfg["train"]["ev_min_days"])
    if cfg["model"]["factors"] == "attention":
        loss = loss - cfg["train"]["lambda_var"] * ev
    return loss, {"sr": sr.item(), "ev": ev.item(), "turnover": stats["turnover"].mean().item()}


def input_scale(data: ArbData, h: np.ndarray) -> float:
    """Median over training days of the cross-sectional std of returns in U_t (raw returns only)."""
    R = data.R[data.hold_cal_idx[h]].cpu().numpy()
    U = data.U[h].cpu().numpy()
    sds = [np.nanstd(R[k, U[k]]) for k in range(len(h)) if U[k].sum() > 1]
    return float(np.median(sds))


def feature_index(data: ArbData, cfg: dict) -> list[int]:
    return [data.feature_names.index(n) for n in feature_names(cfg)]


def blocks_for_epoch(h: np.ndarray, block: int, min_block: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Consecutive (block+1)-day windows from a random offset; a short trailing block is dropped."""
    offset = int(rng.integers(0, block))
    out = []
    for start in range(offset, len(h) - 1, block):
        end = min(start + block, len(h) - 1)
        if end - start >= min_block:
            out.append(h[start: end + 1])
    rng.shuffle(out)
    return out


def fit(data: ArbData, cfg: dict, start, end, seed: int, device=None) -> tuple[ArbitrageModel, float, list[dict]]:
    """Train from scratch on holding days in [start, end]. Returns (model, input scale, history)."""
    device = device or get_device(cfg["train"]["device"])
    set_seed(seed)
    h = data.day_range(start, end)
    scale = input_scale(data, h)
    feat_idx = feature_index(data, cfg)
    tcfg = cfg["train"]
    model = ArbitrageModel(len(feat_idx), cfg["model"]).to(device)
    factor_ids = {id(p) for p in model.factor_parameters()}
    seq_params = [p for p in model.parameters() if id(p) not in factor_ids]
    opt = torch.optim.AdamW([
        {"params": seq_params, "weight_decay": tcfg["weight_decay_seq"]},
        {"params": model.factor_parameters(), "weight_decay": 0.0},
    ], lr=tcfg["lr"])
    rng = np.random.default_rng(seed)
    history = []
    model.train()
    for epoch in range(tcfg["epochs"]):
        logs = []
        for block in blocks_for_epoch(h, tcfg["block_days"], tcfg["min_block_days"], rng):
            batch = make_batch(data, block, cfg["model"]["lookback"], feat_idx, device)
            out = model(batch["X"], batch["U"], batch["R_hist"], batch["R_t"], scale, batch["pca_wF"])
            loss, info = block_loss(out, batch, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            logs.append(info)
        summary = {"epoch": epoch, **{k: float(np.mean([l[k] for l in logs])) for k in logs[0]}}
        history.append(summary)
        log.debug("epoch %d: block SR %.3f (×√252 = %.2f) EV %.3f turnover %.2f", epoch, summary["sr"],
                  summary["sr"] * np.sqrt(252), summary["ev"], summary["turnover"])
    return model, scale, history


@torch.no_grad()
def predict(model: ArbitrageModel, data: ArbData, cfg: dict, scale: float, start, end, device=None) -> pd.DataFrame:
    """Target weights for holding days in [start, end] (holding days × asset_ids)."""
    device = device or next(model.parameters()).device
    model.eval()
    h = data.day_range(start, end)
    batch = make_batch(data, h, cfg["model"]["lookback"], feature_index(data, cfg), device)
    out = model(batch["X"], batch["U"], batch["R_hist"], batch["R_t"], scale, batch["pca_wF"])
    return pd.DataFrame(out["w"].cpu().numpy(), index=data.holding_days[h], columns=data.asset_ids)
