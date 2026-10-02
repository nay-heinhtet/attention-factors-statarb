"""Tuning, freezing and the out-of-sample walk-forward (PLAN.md §11–12).

``tune`` trains on 2011–2016 and scores every grid point by its 2017–2018 net Sharpe from the
backtest engine (mean over the tuning seeds). ``freeze`` writes the selected settings to
configs/frozen.yaml; from then on nothing may be tuned again. No function here reads data
dated 2019 or later until ``oos`` runs on a clean, frozen checkout.
"""

from __future__ import annotations

import itertools
import json
import logging
import time

import numpy as np
import pandas as pd
import yaml

from statarb.backtest import load_market, run_backtest, sharpe, summarize
from statarb.baselines.pairs import pairs_weights
from statarb.baselines.reversal import FEATURE, reversal_weights
from statarb.config import deep_merge, read_yaml
from statarb.data.panel import load_panel, wide
from statarb.data.universe import daily_universe, load_universe
from statarb.paths import CONFIG_DIR, project_paths
from statarb.tensors import load as load_tensors, tensors_path
from statarb.train import fit, predict

log = logging.getLogger(__name__)

MODEL_STRATEGIES = {"attention": {"model": {"factors": "attention"}}, "pca": {"model": {"factors": "pca"}}}
PARAM_PATH = {  # grid parameter → config location, per strategy
    "attention": {"K": ("model", "K"), "lambda_var": ("train", "lambda_var"), "lr": ("train", "lr"),
                  "lookback": ("model", "lookback"), "hidden": ("model", "hidden")},
    "pca": {"K": ("model", "K"), "lr": ("train", "lr")},
    "pairs": {"entry_z": ("pairs", "entry_z"), "max_pairs": ("pairs", "max_pairs")},
    "reversal": {"lookback": ("reversal", "lookback")},
}


def override(strategy: str, params: dict) -> dict:
    """Nested config override for one grid point (strategy defaults included)."""
    out = json.loads(json.dumps(MODEL_STRATEGIES.get(strategy, {})))
    for name, value in params.items():
        section, key = PARAM_PATH[strategy][name]
        out.setdefault(section, {})[key] = value
    return out


def grid(spec: dict) -> list[dict]:
    names = list(spec)
    return [dict(zip(names, values)) for values in itertools.product(*(spec[n] for n in names))]


def select(table: pd.DataFrame, tie_break: list, margin: float) -> dict:
    """Best mean validation net Sharpe; among configs within ``margin`` apply preferences in order."""
    scores = table.groupby("params")["sr_net"].mean()
    near = [json.loads(p) for p in scores.index[scores >= scores.max() - margin]]
    for name, preferred in tie_break:
        keep = [p for p in near if p.get(name) == preferred]
        near = keep or near
    return max(near, key=lambda p: scores[json.dumps(p, sort_keys=True)])


# ------------------------------------------------------------------------------- validation runs

def validate_model(cfg: dict, data, market, seed: int) -> dict:
    (t0, t1), (v0, v1) = cfg["dates"]["tune_train"], cfg["dates"]["tune_valid"]
    model, scale, history = fit(data, cfg, t0, t1, seed)
    weights = predict(model, data, cfg, scale, v0, v1)
    res = run_backtest(weights, market, **cfg["costs"], gap_max_days=cfg["data"]["gap_max_days"])
    return {"sr_net": sharpe(res["net"]), "sr_gross": sharpe(res["gross"]), "turnover": res["turnover"].mean(),
            "train_sr_last_epoch": history[-1]["sr"] * np.sqrt(252), "ev_last_epoch": history[-1]["ev"]}


def tune_models(cfg: dict, strategy: str, spec: dict, paths, market, data) -> pd.DataFrame:
    rows = []
    points = grid(spec)
    for n, params in enumerate(points, 1):
        c = deep_merge(cfg, override(strategy, params))
        for seed in cfg["train"]["seeds_tune"]:
            t = time.time()
            row = validate_model(c, data, market, seed)
            rows.append({"strategy": strategy, "params": json.dumps(params, sort_keys=True), "seed": seed, **row,
                         "seconds": round(time.time() - t, 1)})
            log.info("%s %d/%d %s seed %d: valid net SR %.2f (gross %.2f, turnover %.2f)", strategy, n, len(points),
                     params, seed, row["sr_net"], row["sr_gross"], row["turnover"])
    return pd.DataFrame(rows)


def tune_baseline(cfg: dict, strategy: str, spec: dict, paths, market) -> pd.DataFrame:
    v0, v1 = cfg["dates"]["tune_valid"]
    universe = load_universe(paths)
    U = daily_universe(universe, market.valid, v0, v1)
    rows, cache = [], {}
    if strategy == "pairs":
        close = wide(load_panel(paths, ["close_adj"]), "close_adj").reindex(market.calendar)
        sectors = pd.read_csv(paths.processed / "sectors.csv").set_index("asset_id")["sector"]
    else:
        features = pd.read_parquet(paths.processed / "features.parquet")
    for params in grid(spec):
        c = deep_merge(cfg, override(strategy, params))
        if strategy == "pairs":
            weights = pairs_weights(close, market.valid, universe, sectors, U.index[0], U.index[-1], c["pairs"], cache)
        else:
            col = FEATURE[c["reversal"]["lookback"]]
            past = features.pivot(index="date", columns="asset_id", values=col).reindex(market.calendar)
            weights = reversal_weights(past, U)
        res = run_backtest(weights, market, **cfg["costs"], gap_max_days=cfg["data"]["gap_max_days"])
        rows.append({"strategy": strategy, "params": json.dumps(params, sort_keys=True), "seed": -1,
                     "sr_net": sharpe(res["net"]), "sr_gross": sharpe(res["gross"]), "turnover": res["turnover"].mean()})
        log.info("%s %s: valid net SR %.2f (gross %.2f)", strategy, params, rows[-1]["sr_net"], rows[-1]["sr_gross"])
    return pd.DataFrame(rows)


def null_test(cfg: dict, params: dict, paths) -> pd.DataFrame:
    """Train/validate the chosen attention config on three pure-noise panels (PLAN.md §13.3)."""
    c = deep_merge(cfg, override("attention", params))
    (t0, t1), (v0, v1) = c["dates"]["tune_train"], c["dates"]["tune_valid"]
    rows = []
    for seed in (0, 1, 2):
        data = load_tensors(tensors_path(paths, "pit", f"noise{seed}")).to("cuda" if c["train"]["device"] != "cpu" else "cpu")
        model, scale, _ = fit(data, c, t0, t1, seed)
        w = predict(model, data, c, scale, v0, v1)
        R = data.R[data.hold_cal_idx[data.day_range(v0, v1)]].cpu().numpy()
        gross = (w.to_numpy() * np.nan_to_num(R)).sum(1)
        rows.append({"noise_seed": seed, "valid_gross_sr": sharpe(gross)})
        log.info("null test, noise draw %d: validation gross SR %.2f", seed, rows[-1]["valid_gross_sr"])
    return pd.DataFrame(rows)


def tuning_path(paths, strategy: str):
    return paths.reports / f"tuning_{strategy}.csv"


def tune(cfg: dict, args) -> None:
    paths = project_paths(cfg)
    paths.reports.mkdir(parents=True, exist_ok=True)
    grids = read_yaml(CONFIG_DIR / "grids.yaml")
    strategy, spec = args.strategy, grids[args.strategy]
    market = load_market(cfg, paths)
    if strategy in MODEL_STRATEGIES:
        device = "cuda" if cfg["train"]["device"] != "cpu" else "cpu"
        data = load_tensors(tensors_path(paths, "pit", "real")).to(device)
        table = tune_models(cfg, strategy, spec["grid"], paths, market, data)
        best = select(table, spec["tie_break"], grids["selection"]["tie_margin"])
        best_score = table[table["params"] == json.dumps(best, sort_keys=True)]["sr_net"].mean()
        if strategy == "attention" and best_score <= 0:
            log.warning("best attention validation net SR %.2f ≤ 0: running the pre-authorized extension", best_score)
            ext = {**{k: [v] for k, v in best.items()}, **spec["extension"]}
            table = pd.concat([table, tune_models(cfg, strategy, ext, paths, market, data)], ignore_index=True)
        table.to_csv(tuning_path(paths, strategy), index=False)
        if strategy == "attention":
            best = select(table, spec["tie_break"], grids["selection"]["tie_margin"])
            null = null_test(cfg, best, paths)
            null.to_csv(paths.reports / "null_test.csv", index=False)
            if abs(null["valid_gross_sr"].mean()) >= 1.0:
                raise RuntimeError(f"null test failed (mean gross SR {null['valid_gross_sr'].mean():.2f}): look for leakage")
    else:
        table = tune_baseline(cfg, strategy, spec["grid"], paths, market)
        table.to_csv(tuning_path(paths, strategy), index=False)
    summary = table.groupby("params")["sr_net"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    log.info("%s tuning (validation net SR by config):\n%s", strategy, summary.round(3).to_string())


def freeze(cfg: dict, args) -> None:
    """Select every strategy's settings from the tuning tables and write configs/frozen.yaml."""
    paths = project_paths(cfg)
    grids = read_yaml(CONFIG_DIR / "grids.yaml")
    frozen = {"frozen_on": pd.Timestamp.now().strftime("%Y-%m-%d"), "strategies": {}, "validation": {}}
    tables = []
    for strategy in ("attention", "pca", "pairs", "reversal"):
        table = pd.read_csv(tuning_path(paths, strategy))
        tables.append(table)
        best = select(table, grids[strategy]["tie_break"], grids["selection"]["tie_margin"])
        chosen = table[table["params"] == json.dumps(best, sort_keys=True)]
        frozen["strategies"][strategy] = override(strategy, best)
        frozen["validation"][strategy] = {"params": best, "valid_net_sr_mean": round(float(chosen["sr_net"].mean()), 4),
                                          "valid_net_sr_sd": round(float(chosen["sr_net"].std()), 4) if len(chosen) > 1 else None,
                                          "valid_gross_sr_mean": round(float(chosen["sr_gross"].mean()), 4)}
    pd.concat(tables, ignore_index=True).to_csv(paths.reports / "tuning.csv", index=False)
    header = "# Written by `python -m statarb freeze` from results/reports/tuning.csv. Do not edit.\n"
    (CONFIG_DIR / "frozen.yaml").write_text(header + yaml.safe_dump(frozen, sort_keys=False))
    log.info("frozen:\n%s", yaml.safe_dump(frozen, sort_keys=False))
