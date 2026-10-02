"""Look-ahead, survivorship and robustness checks (PLAN.md §13) → results/reports/bias_checks.md.

Reads the frozen OOS runs (results/runs/*) and never retrains, except that the survivor-universe
comparison needs ``oos --strategy attention --universe survivor`` to have been run (make checks).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from statarb.backtest import bootstrap_sharpe_diff, load_market, run_backtest, sharpe
from statarb.config import deep_merge, read_yaml
from statarb.data.panel import load_panel, wide
from statarb.data.universe import daily_universe, load_universe
from statarb.data.yahoo import load_prices, master_calendar
from statarb.features import compute_features, feature_names
from statarb.models.arbitrage import ArbitrageModel
from statarb.paths import CONFIG_DIR, REPO_ROOT, project_paths
from statarb.tensors import build_tensors
from statarb.train import predict
from statarb.utils import markdown_table

log = logging.getLogger(__name__)

STRATEGIES = ["attention", "pca", "pairs", "reversal"]
CAUSALITY_TESTS = ["tests/test_features.py", "tests/test_universe.py", "tests/test_tensors.py", "tests/test_models.py",
                   "tests/test_backtest.py", "tests/test_projection.py", "tests/test_planted_signal.py"]


def causality_tests() -> pd.DataFrame:
    """Run the look-ahead-related unit tests (slow ones included) and collect pass/fail per test."""
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", "-m", "", *CAUSALITY_TESTS],
                         cwd=REPO_ROOT, capture_output=True, text=True).stdout
    rows = re.findall(r"^(PASSED|FAILED|ERROR) (\S+)", out, flags=re.M)
    return pd.DataFrame(rows, columns=["outcome", "test"])


def _perturb_after(frames: dict[str, pd.DataFrame], t_star: pd.Timestamp, seed: int) -> dict[str, pd.DataFrame]:
    """Replace every price/volume/return dated after t* with noise (validity mask unchanged)."""
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in frames.items()}
    after = out["ret"].index > t_star
    shape = out["ret"].loc[after].shape
    noise = pd.DataFrame(rng.normal(0, 0.05, shape), index=out["ret"].index[after], columns=out["ret"].columns)
    valid_after = out["valid"].loc[after]
    out["ret"].loc[after] = noise.where(valid_after)
    base = out["close_adj"].loc[~after].ffill().iloc[-1]
    out["close_adj"].loc[after] = ((1 + noise.fillna(0)).cumprod() * base).where(valid_after)
    for col, scale in (("high", 1.03), ("low", 0.97)):
        out[col].loc[after] = out["close_adj"].loc[after] * scale
    out["dollar_volume"].loc[after] = out["dollar_volume"].loc[after] * rng.uniform(0.2, 5.0, shape)
    return out


def future_perturbation(cfg: dict, paths, frozen: dict, n_dates: int = 5, seed: int = 0) -> pd.DataFrame:
    """End-to-end: scramble all raw data after t*, rebuild features and tensors, re-predict with the
    trained OOS model. Weights for holding days ≤ t*+1 must not move (CPU, tolerance 1e-7)."""
    c = deep_merge(cfg, frozen["strategies"]["attention"])
    calendar = master_calendar(c, paths)
    fields = ["close_adj", "valid", "ret", "high", "low", "dollar_volume"]
    panel = load_panel(paths, fields)
    frames = {f: wide(panel, f).reindex(calendar) for f in fields}
    frames["valid"] = frames["valid"].eq(True)
    spy = load_prices(paths, "SPY")["Adj Close"].reindex(calendar)
    market_ret = spy / spy.shift(1) - 1
    names = feature_names(c, apply_drops=False)
    base_features = compute_features(**frames, market_ret=market_ret, fcfg=c["features"])
    universe = load_universe(paths)

    rng = np.random.default_rng(seed)
    oos_days = calendar[(calendar >= "2019-02-01") & (calendar <= "2026-08-20")]
    rows = []
    for t_star in sorted(rng.choice(oos_days, n_dates, replace=False)):
        t_star = pd.Timestamp(t_star)
        pos = calendar.get_loc(t_star)
        days = calendar[pos - 5: pos + 3]  # holding days around t*: …, t*, t*+1, t*+2
        U = daily_universe(universe, frames["valid"], days[0], days[-1])
        ckpt = torch.load(paths.runs / "attention" / "checkpoints" / f"Y{t_star.year}_seed0.pt", weights_only=False)
        model = ArbitrageModel(len(feature_names(c)), c["model"])
        model.load_state_dict(ckpt["state_dict"])

        weights = []
        for variant in ("real", "perturbed"):
            f = frames if variant == "real" else _perturb_after(frames, t_star, seed)
            feats = base_features if variant == "real" else compute_features(**f, market_ret=market_ret, fcfg=c["features"])
            data = build_tensors(calendar, f["ret"], {n: feats[n] for n in names}, U)
            weights.append(predict(model, data, c, ckpt["scale"], days[0], days[-1], device="cpu"))
        diff = (weights[0] - weights[1]).abs().max(axis=1)
        safe = diff[diff.index <= days[-2]]           # holding days ≤ t*+1 only use data ≤ t*
        rows.append({"t_star": t_star.date(), "max_abs_diff_upto_t_plus_1": float(safe.max()),
                     "max_abs_diff_t_plus_2": float(diff.iloc[-1]), "pass": bool(safe.max() <= 1e-7 and diff.iloc[-1] > 0)})
        log.info("perturbation after %s: max |Δw| up to t*+1 = %.1e, at t*+2 = %.1e", t_star.date(),
                 rows[-1]["max_abs_diff_upto_t_plus_1"], rows[-1]["max_abs_diff_t_plus_2"])
    return pd.DataFrame(rows)


def load_weights(paths, run: str) -> pd.DataFrame:
    return pd.read_parquet(paths.runs / run / "weights_ensemble.parquet")


def robustness(cfg: dict, paths, market) -> pd.DataFrame:
    """Net Sharpe under a 1-day execution lag and under 0/2/5/10 bps costs (evaluation only)."""
    rows = []
    for s in STRATEGIES:
        w = load_weights(paths, s)
        for label, kw in [("base (lag 0, 5 bps)", {}), ("exec lag 1 day", {"exec_lag": 1}),
                          *[(f"{bps} bps", {"tc_bps": bps}) for bps in (0, 2, 10)]]:
            args = {"tc_bps": cfg["costs"]["tc_bps"], "short_bps": cfg["costs"]["short_bps"],
                    "gap_max_days": cfg["data"]["gap_max_days"], **kw}
            res = run_backtest(w, market, **args)
            rows.append({"strategy": s, "scenario": label, "sr_net": sharpe(res["net"]), "sr_gross": sharpe(res["gross"])})
    return pd.DataFrame(rows)


def survivorship(paths) -> dict:
    """Same frozen model on the point-in-time vs today's-constituents universe."""
    daily = {k: pd.read_csv(paths.reports / "oos" / f"{k}_daily.csv", index_col=0, parse_dates=True)["net"]
             for k in ("attention", "attention_survivor")}
    both = pd.concat(daily, axis=1).dropna()
    diff, lo, hi = bootstrap_sharpe_diff(both["attention_survivor"], both["attention"])
    coverage = pd.read_csv(paths.processed / "coverage_report.csv", index_col=0)
    oos_cov = coverage.loc[coverage.index >= 2019]
    return {"sr_net_pit": sharpe(both["attention"]), "sr_net_survivor": sharpe(both["attention_survivor"]),
            "delta_survivor_minus_pit": diff, "delta_ci": [lo, hi],
            "oos_member_days_with_data": float(1 - oos_cov["unrecoverable"].sum() / oos_cov[["yahoo", "tiingo", "alias", "unrecoverable"]].sum().sum()),
            "oos_tiingo_slot_share": float(oos_cov["tiingo_slots"].sum() / oos_cov["universe_slots"].sum())}


def halt_outcome(paths, market) -> list[str]:
    """What the attention strategy held into SVB's and First Republic's collapses."""
    lines = (paths.processed / "halt_check.txt").read_text().strip().splitlines()
    _, positions = run_backtest(load_weights(paths, "attention"), market, return_positions=True)
    for asset, day in (("tiingo:SIVBQ", "2023-03-28"), ("tiingo:FRCB", "2023-05-03")):
        if asset not in positions.columns:
            lines.append(f"{asset}: never in the strategy's universe")
            continue
        p = positions.loc[pd.Timestamp(day), asset]
        r = market.ret.loc[pd.Timestamp(day), asset]
        lines.append(f"{asset} on {day}: position {p:+.4f} (gross exposure 1), booked return {r:+.1%}, "
                     f"P&L contribution {p * r:+.3%} of capital")
    return lines


def run(cfg: dict, args) -> None:
    paths = project_paths(cfg)
    frozen = read_yaml(CONFIG_DIR / "frozen.yaml")
    market = load_market(cfg, paths)
    tests = causality_tests()
    perturb = future_perturbation(cfg, paths, frozen)
    robust = robustness(cfg, paths, market)
    surv = survivorship(paths)
    halts = halt_outcome(paths, market)
    null = pd.read_csv(paths.reports / "null_test.csv")

    out = paths.reports
    tests.to_csv(out / "bias_causality_tests.csv", index=False)
    perturb.to_csv(out / "bias_future_perturbation.csv", index=False)
    robust.to_csv(out / "bias_robustness.csv", index=False)
    (out / "bias_survivorship.json").write_text(json.dumps(surv, indent=1))

    table = robust.pivot(index="scenario", columns="strategy", values="sr_net")[STRATEGIES]
    md = ["# Bias and robustness checks", "", "Generated by `python -m statarb checks` (PLAN.md §13).", "",
          "## 1. Look-ahead: causality unit tests", "",
          f"{(tests['outcome'] == 'PASSED').sum()} passed, {(tests['outcome'] != 'PASSED').sum()} failed "
          f"(truncation invariance of features and universe, tensors built from t−1, weights independent of the "
          f"realized return, engine alignment, planted-signal controls).", "",
          "## 2. Look-ahead: end-to-end future perturbation on trained OOS models", "",
          "All raw data dated after t* is replaced with noise; features and tensors are rebuilt; the frozen "
          "attention model (seed 0, that year's checkpoint) re-predicts on CPU.", "",
          markdown_table(perturb, index=False, digits=10), "",
          "## 3. Look-ahead: pure-noise panels (tuning split)", "",
          markdown_table(null, index=False), "", f"Mean validation gross SR {null['valid_gross_sr'].mean():+.2f} "
          "(pass if |mean| < 1.0).", "",
          "## 4. Execution lag and cost sensitivity (OOS net Sharpe, 2019-01 → 2026-08)", "",
          markdown_table(table), "",
          "## 5. Survivorship", "",
          f"- Member-days 2019+ with price data: {surv['oos_member_days_with_data']:.2%}; universe slots filled by "
          f"delisted (Tiingo-only) names: {surv['oos_tiingo_slot_share']:.2%}.",
          f"- Frozen attention model, point-in-time universe: net SR {surv['sr_net_pit']:+.2f}; "
          f"today's-constituents universe: {surv['sr_net_survivor']:+.2f}.",
          f"- Survivor − point-in-time: {surv['delta_survivor_minus_pit']:+.2f} "
          f"[{surv['delta_ci'][0]:+.2f}, {surv['delta_ci'][1]:+.2f}] (95% stationary-bootstrap CI).", "",
          "## 6. Halts and failures (SVB, First Republic)", "", *[f"- {line}" for line in halts], ""]
    (out / "bias_checks.md").write_text("\n".join(md))
    log.info("\n".join(md))
