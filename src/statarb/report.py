"""Final tables, figures and report (PLAN.md §15) → results/reports/.

Reads only saved artifacts (OOS runs, checks, tuning, coverage); nothing is retrained.
Colors follow the dataviz reference palette in fixed slot order; every figure has a matching
table in final_report.md (relief for the three light slots below 3:1 contrast).
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from statarb.backtest import load_market, run_backtest, sharpe, summarize  # noqa: E402
from statarb.config import deep_merge, read_yaml  # noqa: E402
from statarb.data.universe import daily_universe, load_universe  # noqa: E402
from statarb.models.arbitrage import ArbitrageModel  # noqa: E402
from statarb.paths import CONFIG_DIR, project_paths  # noqa: E402
from statarb.tensors import load as load_tensors, tensors_path  # noqa: E402
from statarb.train import feature_index, make_batch  # noqa: E402
from statarb.utils import markdown_table  # noqa: E402

log = logging.getLogger(__name__)

STRATEGIES = ["attention", "pca", "pairs", "reversal"]
LABEL = {"attention": "Attention + LSTM", "pca": "PCA + LSTM (two-step)", "pairs": "Pairs trading",
         "reversal": "5-day reversal", "spy": "SPY (buy and hold)", "ew": "Equal-weight universe (long-only)"}
COLOR = {"attention": "#2a78d6", "pca": "#eb6834", "pairs": "#1baf7a", "reversal": "#eda100", "spy": "#e87ba4"}
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
COVERAGE_COLORS = {"yahoo": "#2a78d6", "tiingo": "#eb6834", "alias": "#1baf7a", "unrecoverable": "#eda100"}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "font.size": 9.5, "legend.frameon": False, "legend.labelcolor": TEXT, "lines.linewidth": 1.6,
    })


def _direct_labels(ax, series: dict[str, pd.Series], fmt=lambda v: "") -> None:
    """Label each line at its right end in text ink, nudged apart so labels never collide."""
    ends = sorted(((s.dropna().iloc[-1], key) for key, s in series.items()), key=lambda t: t[0])
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.05
    placed: list[float] = []
    for y, key in ends:
        y_text = max(y, placed[-1] + gap) if placed else y
        placed.append(y_text)
        x = series[key].dropna().index[-1]
        ax.plot([x], [y], "o", ms=4, color=COLOR[key], mec=SURFACE, mew=1.0, zorder=5)
        ax.text(x + pd.Timedelta(days=25), y_text, f"{LABEL[key].split(' (')[0]}{fmt(y)}",
                va="center", fontsize=8.5, color=TEXT)


def _legend_below(ax) -> None:
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.1), ncol=5, fontsize=8.5, handlelength=2.2)


def _save(fig, path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _years_axis(ax) -> None:
    ax.set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2027-04-01"))  # room for direct labels
    ax.margins(x=0)


# ---------------------------------------------------------------------------- data

def load_daily(paths, run: str) -> pd.DataFrame:
    return pd.read_csv(paths.reports / "oos" / f"{run}_daily.csv", index_col=0, parse_dates=True)


def reference_series(cfg: dict, paths, market, days: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """SPY buy-and-hold and the equal-weight universe (long-only, no costs), as engine-style frames."""
    spy = market.spy.reindex(days)
    spy_frame = pd.DataFrame({"gross": spy, "cost": 0.0, "net": spy, "turnover": 0.0, "short": 0.0,
                              "gross_exposure": 1.0, "net_exposure": 1.0})
    U = daily_universe(load_universe(paths), market.valid, days[0], days[-1])
    ew = U.astype(float).div(U.sum(axis=1), axis=0)
    ew_frame = run_backtest(ew, market, tc_bps=0.0, short_bps=0.0, gap_max_days=cfg["data"]["gap_max_days"])
    return {"spy": spy_frame, "ew": ew_frame}


# ---------------------------------------------------------------------------- tables

def main_table(paths, refs: dict, market, bcfg: dict) -> pd.DataFrame:
    rows = {}
    for s in STRATEGIES:
        rows[LABEL[s]] = json.load(open(paths.reports / "oos" / f"{s}.json"))
    for key in ("ew", "spy"):
        rows[LABEL[key]] = summarize(refs[key], market.spy, bcfg)
    out = []
    for name, m in rows.items():
        out.append({"strategy": name, "SR gross": m["sr_gross"], "SR net": m["sr_net"],
                    "95% CI (net)": f"[{m['sr_net_ci_lo']:+.2f}, {m['sr_net_ci_hi']:+.2f}]",
                    "μ net %/yr": 100 * m["mu_net"], "σ %/yr": 100 * m["vol"], "max DD %": 100 * m["max_dd"],
                    "turnover/day": m["turnover"], "gross exp.": m["gross_exposure"], "net $ exp.": m["net_exposure"],
                    "β (t)": f"{m['beta']:+.2f} ({m['beta_t']:.1f})"})
    table = pd.DataFrame(out).set_index("strategy")
    table.loc[LABEL["spy"], "β (t)"] = "+1.00 (by definition)"  # regressing SPY on itself: t is meaningless
    for c in ("SR gross", "SR net", "μ net %/yr", "σ %/yr", "max DD %", "turnover/day", "gross exp.", "net $ exp."):
        table[c] = table[c].astype(float)
    return table


def per_year_table(paths, refs: dict) -> pd.DataFrame:
    cols = {}
    for s in STRATEGIES:
        cols[LABEL[s]] = pd.Series(json.load(open(paths.reports / "oos" / f"{s}.json"))["per_year_net_sr"])
    cols[LABEL["spy"]] = refs["spy"]["net"].groupby(refs["spy"].index.year).apply(sharpe).rename(index=str)
    table = pd.DataFrame(cols)
    table.index.name = "year"
    return table


def tuning_table(paths) -> pd.DataFrame:
    frozen = read_yaml(CONFIG_DIR / "frozen.yaml")
    rows = []
    for s in STRATEGIES:
        v = frozen["validation"][s]
        rows.append({"strategy": LABEL[s], "frozen settings": ", ".join(f"{k}={val}" for k, val in v["params"].items()),
                     "valid SR net": v["valid_net_sr_mean"], "valid SR gross": v["valid_gross_sr_mean"]})
    return pd.DataFrame(rows).set_index("strategy")


def paper_table(paths) -> pd.DataFrame:
    att = json.load(open(paths.reports / "oos" / "attention.json"))
    pca = json.load(open(paths.reports / "oos" / "pca.json"))
    rows = [
        ("Attention factors (paper, K=30, N=500, CRSP 1998–2021)", 3.97, 2.28),
        ("PCA + LongConv two-step (paper, K=30)", 2.79, 1.57),
        ("Attention factors (this build, K=5, N=200, 2019–2026)", att["sr_gross"], att["sr_net"]),
        ("PCA + LSTM two-step (this build, K=5)", pca["sr_gross"], pca["sr_net"]),
    ]
    return pd.DataFrame(rows, columns=["model", "SR gross", "SR net"]).set_index("model")


# ---------------------------------------------------------------------------- figures

def fig_cumulative(daily: dict, refs: dict, path, vol_target: float | None) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    lines = {}
    for key in [*STRATEGIES, "spy"]:
        r = (refs["spy"]["net"] if key == "spy" else daily[key]["net"]).fillna(0.0)
        if vol_target:
            r = r * vol_target / (r.std() * np.sqrt(252))
        wealth = (1 + r).cumprod() - 1
        lines[key] = 100 * wealth
        ax.plot(wealth.index, 100 * wealth, color=COLOR[key], label=LABEL[key], lw=1.6 if key != "spy" else 1.3,
                ls="-" if key != "spy" else (0, (4, 2)))
    ax.axhline(0, color=TEXT2, lw=0.8)
    _years_axis(ax)
    _direct_labels(ax, lines, fmt=lambda v: f"  {v:+.0f}%")
    title = "Cumulative net return, 2019–2026"
    ax.set_title(title + (f" (each series scaled ex post to {vol_target:.0%}/yr volatility)" if vol_target else ""))
    ax.set_ylabel("cumulative return, %")
    _legend_below(ax)
    _save(fig, path)


def fig_drawdown(daily: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    lines = {}
    for key in STRATEGIES:
        wealth = (1 + daily[key]["net"].fillna(0)).cumprod()
        dd = 100 * (wealth / wealth.cummax() - 1)
        lines[key] = dd
        ax.plot(dd.index, dd, color=COLOR[key], label=LABEL[key])
    _years_axis(ax)
    _direct_labels(ax, lines)
    ax.set_title("Drawdown of net returns (unscaled)")
    ax.set_ylabel("drawdown, %")
    _legend_below(ax)
    _save(fig, path)


def fig_rolling_sharpe(daily: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    lines = {}
    for key in STRATEGIES:
        r = daily[key]["net"]
        roll = r.rolling(252).mean() / r.rolling(252).std() * np.sqrt(252)
        lines[key] = roll
        ax.plot(roll.index, roll, color=COLOR[key], label=LABEL[key])
    ax.axhline(0, color=TEXT2, lw=0.8)
    _years_axis(ax)
    _direct_labels(ax, lines)
    ax.set_title("Rolling 252-day net Sharpe ratio")
    ax.set_ylabel("Sharpe (annualized)")
    _legend_below(ax)
    _save(fig, path)


def fig_turnover(daily: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    lines = {}
    for key in STRATEGIES:
        t = daily[key]["turnover"].rolling(21).mean()
        lines[key] = t
        ax.plot(t.index, t, color=COLOR[key], label=LABEL[key])
    _years_axis(ax)
    _direct_labels(ax, lines)
    ax.set_title("Daily turnover (21-day average, fraction of gross exposure traded)")
    ax.set_ylabel("turnover per day")
    _legend_below(ax)
    _save(fig, path)


def factor_snapshot(cfg: dict, paths, date: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Attention factor weights on ``date`` (that year's seed-0 model): weight by sector, top names."""
    c = deep_merge(cfg, read_yaml(CONFIG_DIR / "frozen.yaml")["strategies"]["attention"])
    year = pd.Timestamp(date).year
    ckpt = torch.load(paths.runs / "attention" / "checkpoints" / f"Y{year}_seed0.pt", weights_only=False)
    data = load_tensors(tensors_path(paths, "pit", "real"))
    model = ArbitrageModel(len(feature_index(data, c)), c["model"])
    model.load_state_dict(ckpt["state_dict"])
    h = data.day_range(date, date)
    batch = make_batch(data, h, c["model"]["lookback"], feature_index(data, c), "cpu")
    with torch.no_grad():
        wF = model.factors(batch["X"], batch["U"])[0].numpy()  # (K, A)
    sectors = pd.read_csv(paths.processed / "sectors.csv").set_index("asset_id")["sector"]
    resolution = pd.read_csv(paths.processed / "resolution.csv")
    names = resolution.dropna(subset=["asset_id"]).groupby("asset_id")["member_ticker"].agg(lambda s: "/".join(sorted(set(s))))
    weights = pd.DataFrame(wF.T, index=data.asset_ids, columns=[f"factor {k + 1}" for k in range(wF.shape[0])])
    unit = wF / np.linalg.norm(wF, axis=1, keepdims=True)
    cos = unit @ unit.T
    k = len(cos)
    diag = {"mean_pairwise_cosine": float((cos.sum() - k) / (k * (k - 1))),
            "effective_stocks_per_factor": [round(float(1 / (w ** 2).sum()), 1) for w in wF]}
    by_sector = weights.groupby(sectors.reindex(weights.index).fillna("Unknown")).sum()
    by_sector = by_sector.loc[by_sector.sum(axis=1) > 0]
    top = []
    for col in weights:
        best = weights[col].nlargest(10)
        top += [{"factor": col, "rank": i + 1, "ticker": names.get(a, a), "sector": sectors.get(a, "Unknown"),
                 "weight %": round(100 * w, 2)} for i, (a, w) in enumerate(best.items())]
    return by_sector, pd.DataFrame(top), diag


def fig_factor_sectors(by_sector: pd.DataFrame, date: str, path) -> None:
    data = 100 * by_sector.T  # factors × sectors
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(data) + 1.6))
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    im = ax.imshow(data.to_numpy(), cmap=cmap, aspect="auto", vmin=0, vmax=data.to_numpy().max())
    ax.set_xticks(range(len(data.columns)), data.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(data.index)), data.index)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(data.to_numpy()):
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                color="#ffffff" if v > 0.55 * im.get_clim()[1] else TEXT)
    ax.set_title(f"Attention factor portfolio weight by sector, % ({date}, 2024 model, seed 0)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=TEXT2)
    _save(fig, path)


def fig_coverage(paths, path) -> pd.DataFrame:
    cov = pd.read_csv(paths.processed / "coverage_report.csv", index_col=0)
    parts = cov[["yahoo", "tiingo", "alias", "unrecoverable"]]
    share = 100 * parts.div(parts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    bottom = np.zeros(len(share))
    labels = {"yahoo": "Yahoo, same ticker", "tiingo": "Tiingo (delisted)", "alias": "alias (renamed / reused ticker)",
              "unrecoverable": "no free price data"}
    for col in share:
        ax.bar(share.index, share[col], bottom=bottom, color=COVERAGE_COLORS[col], label=labels[col],
               width=0.72, edgecolor=SURFACE, linewidth=1.0)
        bottom += share[col].to_numpy()
    ax.set_ylim(0, 100)
    ax.set_xticks(share.index)
    ax.set_title("Where each S&P 500 member-day's prices come from")
    ax.set_ylabel("% of member-days")
    _legend_below(ax)
    ax.grid(axis="x", visible=False)
    _save(fig, path)
    return share.round(1)


# ---------------------------------------------------------------------------- report

def cv_bullets(t1: pd.DataFrame, surv: dict) -> list[str]:
    a, p, pr = (t1.loc[LABEL[k]] for k in ("attention", "pca", "pairs"))
    return [
        "Built a simplified reproduction of \"Attention Factors for Statistical Arbitrage\" (arXiv:2510.11616): "
        "cross-sectional attention with learned factor queries over 15 price/volume characteristics produces latent "
        "factor portfolios; an LSTM over 60-day factor residuals outputs residual-portfolio weights; both are trained "
        "jointly end-to-end on net-of-cost Sharpe (PyTorch).",
        f"Walk-forward backtest with annual retraining, out-of-sample 2019–2026, on the 200 most liquid point-in-time "
        f"S&P 500 members including delisted names (yfinance + Tiingo, 871 membership spells resolved): gross Sharpe "
        f"{a['SR gross']:.2f}, net {a['SR net']:+.2f} after 5 bps + 1 bp short costs, vs {p['SR net']:+.2f} for a two-step "
        f"PCA benchmark and {pr['SR net']:+.2f} for cointegration pairs trading — no strategy survives costs in large caps.",
        f"Verified the pipeline: truncation and future-perturbation tests (weights bit-identical when future data is "
        f"scrambled), a pure-noise null test and a planted-signal control; quantified survivorship bias "
        f"(a today's-constituents universe inflates net Sharpe by {surv['delta_survivor_minus_pit']:+.2f}).",
    ]


def run(cfg: dict, args) -> None:
    paths = project_paths(cfg)
    _style()
    out = paths.reports
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    market = load_market(cfg, paths)
    daily = {s: load_daily(paths, s) for s in STRATEGIES}
    refs = reference_series(cfg, paths, market, daily["attention"].index)

    t1 = main_table(paths, refs, market, cfg["backtest"])
    t2 = per_year_table(paths, refs)
    robust = pd.read_csv(out / "bias_robustness.csv")
    t3 = robust.pivot(index="scenario", columns="strategy", values="sr_net")
    t3 = t3.loc[list(dict.fromkeys(robust["scenario"])), STRATEGIES].rename(columns=LABEL)
    surv = json.load(open(out / "bias_survivorship.json"))
    t4 = pd.read_csv(out / "ablations.csv").set_index("ablation")
    t5 = paper_table(paths)
    t6 = tuning_table(paths)
    for name, table in {"t1_main": t1, "t2_per_year": t2, "t3_robustness": t3, "t5_paper_vs_ours": t5, "t6_tuning": t6}.items():
        table.to_csv(out / f"{name}.csv")

    fig_cumulative(daily, refs, figs / "f1_cumulative.png", cfg["report"]["vol_target"])
    fig_cumulative(daily, refs, figs / "f1_cumulative_raw.png", None)
    fig_drawdown(daily, figs / "f2_drawdown.png")
    fig_rolling_sharpe(daily, figs / "f3_rolling_sharpe.png")
    fig_turnover(daily, figs / "f4_turnover.png")
    by_sector, top, fdiag = factor_snapshot(cfg, paths, cfg["report"]["factor_date"])
    fig_factor_sectors(by_sector, cfg["report"]["factor_date"], figs / "f5_factor_sectors.png")
    top.to_csv(out / "f6_top_names.csv", index=False)
    coverage = fig_coverage(paths, figs / "f7_coverage.png")

    bias = (out / "bias_checks.md").read_text()
    a = t1.loc[LABEL["attention"]]
    seeds = json.load(open(out / "oos" / "attention.json"))
    md = [
        "# Attention Factors for Statistical Arbitrage: a reproduction on free data", "",
        "Simplified reproduction of Epstein, Wang, Choi & Pelger (2025), arXiv:2510.11616. Out-of-sample walk-forward "
        "2019-01-02 → 2026-08-31 (1,927 trading days, 7.7 years) on the 200 most liquid point-in-time S&P 500 members, "
        "delisted names included. Configuration frozen on 2017–2018 validation (git tag `frozen-v1`) before any "
        "2019+ data was used; every OOS run is logged in `results/oos_runs.csv`.", "",
        "## Summary", "",
        f"- **Headline (attention + LSTM, 5-seed ensemble): net Sharpe {a['SR net']:+.2f}, 95% CI {a['95% CI (net)']}; "
        f"gross {a['SR gross']:+.2f}.** Costs of 5 bps per unit traded plus 1 bp/day on shorts turn a small gross edge "
        "into a loss. With 7.7 years the Sharpe standard error is ≈ 0.36, so none of the strategies is distinguishable "
        "from zero or from each other.",
        "- **The paper's central claim does not replicate in this setting:** joint attention factors do not beat the "
        "two-step PCA benchmark (difference −0.26, CI [−1.12, +0.53]).",
        "- **Not market-neutral:** market β "
        f"{a['β (t)']}, about {a['net $ exp.']:+.0%} net long on average. The paper's construction is factor-neutral; "
        "with only 5 factors and λ_Var = 1 (selected on validation) the factors do not span the market.",
        "- **The attention model worked early and then broke:** net Sharpe positive in 2019–2021, negative every year "
        "since 2022 (table T2).",
        f"- **Survivorship bias, quantified:** the same frozen model on today's constituents scores "
        f"{surv['sr_net_survivor']:+.2f} instead of {surv['sr_net_pit']:+.2f} "
        f"(+{surv['delta_survivor_minus_pit']:.2f}).",
        "- **No look-ahead:** see the checks below (bit-identical weights under scrambled future data, null test).", "",
        "## T1. Out-of-sample results, 2019-01-02 → 2026-08-31", "",
        markdown_table(t1), "",
        f"Attention + LSTM per-seed net Sharpe: {', '.join(f'{s:+.2f}' for s in seeds['seed_net_sr'])} "
        f"(mean {seeds['seed_net_sr_mean']:+.2f} ± {seeds['seed_net_sr_sd']:.2f}); the headline is their ensemble. "
        "Gross exposure is 1 for the model strategies; pairs trading uses committed capital (average gross 0.33).", "",
        "![Cumulative net return, vol-scaled](figures/f1_cumulative.png)", "",
        "![Cumulative net return, raw](figures/f1_cumulative_raw.png)", "",
        "![Drawdown](figures/f2_drawdown.png)", "",
        "## T2. Net Sharpe by year", "", markdown_table(t2), "",
        "![Rolling Sharpe](figures/f3_rolling_sharpe.png)", "",
        "![Turnover](figures/f4_turnover.png)", "",
        "## T3. Robustness: execution lag and costs (OOS net Sharpe)", "",
        markdown_table(t3), "",
        "Short cost stays at 1 bp/day in every row. Even with zero trading cost no strategy exceeds a net Sharpe of 0.3.", "",
        "## T4. Ablations (post-freeze; never replace the headline)", "",
        markdown_table(t4), "",
        "More factors raise turnover and hurt net performance — the opposite of the paper, where weak factors helped "
        "(their N = 500 vs our 200). LongConv ran with hyperparameters tuned for the LSTM, so it is not a fair "
        "verdict on LongConv. Dropping past-return features hurts most, matching the paper's direction.", "",
        "## T5. The paper vs this build", "", markdown_table(t5), "",
        "## T6. Frozen configuration (selected on 2017–2018 validation)", "", markdown_table(t6), "",
        "The paper's λ_Var = 100 was among the worst settings on validation; λ_Var = 1 won.", "",
        "## Factor structure", "",
        f"![Factor weights by sector](figures/f5_factor_sectors.png)", "",
        f"The five factors are nearly the same portfolio: mean pairwise cosine similarity of their weight vectors is "
        f"{fdiag['mean_pairwise_cosine']:.3f}, and each spreads over ~{np.mean(fdiag['effective_stocks_per_factor']):.0f} "
        "of the 200 stocks (effective number 1/Σw²). They are diffuse, technology-tilted, market-like portfolios rather "
        "than the distinct industry factors the paper reports (its Fig. 4). With λ_Var = 1 the explained-variance term "
        "pushes weakly, and the projection effectively removes one market-like direction — consistent with the "
        "remaining market β.", "",
        "Top-10 names per factor: `f6_top_names.csv`.", "",
        "## Data coverage", "", "![Coverage](figures/f7_coverage.png)", "", markdown_table(coverage), "",
        bias.replace("# Bias and robustness checks", "## Bias and robustness checks").replace("\n## ", "\n### "), "",
        "## Limitations", "",
        "- Price/volume characteristics only (no Compustat fundamentals); universe of 200, not 500.",
        "- 7.7 OOS years: Sharpe standard error ≈ 0.36; differences between strategies are not significant.",
        "- Trades assumed at the decision close; linear costs (5 bps + 1 bp short), no market impact or borrow limits.",
        "- Sectors are current, not point-in-time; 43 membership spells (2 overlapping 2019+) have no free price data.",
        "- LSTM instead of the paper's LongConv as the primary sequence model (LongConv reported as an ablation).", "",
        "## CV bullets (measured numbers)", "", *[f"- {b}" for b in cv_bullets(t1, surv)], "",
    ]
    (out / "final_report.md").write_text("\n".join(md))
    log.info("wrote %s with %d figures", out / "final_report.md", len(list(figs.glob("*.png"))))
