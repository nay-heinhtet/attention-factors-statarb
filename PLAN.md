# Attention-Based Statistical Arbitrage: Build Plan

**Status:** Phase 8 done 2026-10-02 (look-ahead, survivorship, lag and cost checks). Next: Phase 9 (ablations, optional) or Phase 10 (report).
**What this is:** the spec for building the project. Every design decision is already made (see §0).
**Paper:** Epstein, Wang, Choi, Pelger, "Attention Factors for Statistical Arbitrage", arXiv:2510.11616 (ICAIF '25). This plan is based on the full paper text plus live checks of every data source (Appendix B).

---

## 0. How to use this document

1. Build the phases in order (§4, then §6 to §15). Each phase ends with **Done when** criteria. Do not start a phase until the previous one meets them.
2. **Do not stop to ask questions.** If something here is ambiguous or missing:
   (a) do what the paper does; (b) if the paper says nothing, pick the simplest option that is **causal**, meaning a decision for holding day *t* uses only data dated ≤ *t−1*; (c) write the choice into `DECISIONS_LOG.md` as date, decision, reason.
3. Bugs can always be fixed. **Design changes after the freeze (§11) are not allowed.** The one exception is logged ablations, and those never replace the headline result.
4. Only one step needs the user: a free Tiingo API key (§4.3). Everything else is automated or covered by a stated rule.

---

## 1. Goal, deliverables, honesty rules

**Goal.** Build a simplified, honest reproduction of the paper on free data.
- **Stage 1:** cross-sectional attention factors, learned from price/volume characteristics, produce residual portfolios.
- **Stage 2:** an LSTM reads each asset's 30-day residual history and outputs residual-portfolio weights.
- Both stages are **trained jointly end to end** on net-of-cost Sharpe plus explained variance.
- **Walk-forward out-of-sample (OOS) backtest from 2019-01-02 to 2026-08-31.** The universe is the 200 most liquid point-in-time S&P 500 members, **including names that were later delisted**.
- **Baselines:**
  - Engle–Granger cointegration pairs trading (the CV requirement).
  - PCA + LSTM two-step. This tests the paper's central claim that joint training beats two-step.
  - Short-term reversal, as a sanity check.
- **Bias checks:**
  - Look-ahead: truncation tests, a null-data test, a 1-day execution-lag test.
  - Survivorship: a coverage report, plus a rerun on today's constituents only.

**Deliverables**
1. A reproducible repo. `make all` rebuilds everything from raw downloads.
2. `results/reports/final_report.md` with tables and figures (§15).
3. A `README.md` covering headline results, deviations from the paper, and limitations.
4. CV bullets filled in with the measured numbers (§15.4).

**Honesty rules (non-negotiable)**
- The "0.5 net Sharpe" in the draft CV bullet is a **hypothesis, not a target**. The CV reports whatever the frozen model produces.
- No data dated 2019-01-01 or later is used for any design or tuning choice. All tuning uses 2011–2018.
- Every OOS evaluation is appended automatically to `results/oos_runs.csv`. The headline is the **first** post-freeze run of the frozen main model.
- With about 7.7 OOS years, the standard error of an annualized Sharpe is ≈ 1/√7.7 ≈ **0.36**. The report must state this. For example, 0.5 is about 1.4 SE: suggestive, not significant.

---

## 2. What the paper does, and what we change

### 2.1 Paper method (verified from the full text)

```
N assets, returns R_t ∈ R^N, characteristics X_{t-1} ∈ R^{N×M}

Embedding        X̃ = X_{t-1} W_K                      W_K ∈ R^{M×d}, d = 32
Factor weights   ω^F = softmax_over_assets(Q X̃ᵀ / √d)  Q ∈ R^{K×d}  (row k = factor k, sums to 1)
Factors          F_t = ω^F R_t
Loadings         βᵀ = ω^Fᵀ (ω^F ω^Fᵀ + λ_ridge I_K)^{-1}
Residuals        ε_t = R_t − βᵀ F_t = ω^ε R_t,   ω^ε = I_N − βᵀ ω^F
Signal           ω^port_i = LongConv_θ(ε_{i, t−30..t−1})
Stock weights    ω = (ω^ε)ᵀ ω^port,  normalized so ‖ω‖₁ = 1
Costs            cost_t = 0.0005·‖ω_t − ω_{t−1}‖₁ + 0.0001·‖max(−ω_t, 0)‖₁
Objective        max  mean(R_net)/std(R_net)  +  λ_Var · (1/N) Σ_i (1 − Var(ε_i)/Var(R_i))
```

- **Training:** 8-year rolling windows, retrained every year, OOS Jan 1998 to Dec 2021. Hyperparameters were picked on the last 2 years of the *first* training window.
- **Selected settings:** Adam, lr 0.003, 30 epochs, weight decay 0.05 on the sequence model, dropout 0.1, λ_Var = 100, d = 32. LongConv has 1 layer, hidden size 32, squash 0.001, and geometric-decay kernel initialization.
- **Data:**
  - CRSP daily returns for the 500 largest stocks by prior-month market cap.
  - 39 Chen–Pelger–Zhu characteristics, rank-normalized.
  - Plus cross-sectional medians and the risk-free rate, for 79 dimensions in total.
  - Missing values are filled with the last observed value, otherwise the cross-sectional median.
- **Sequence model:** the paper says the choice of sequence model is flexible: "we expect alternative sequence models such as Transformers to perform similarly". This supports our LSTM substitution.
- **Sharpe convention:** the paper's reported Sharpe is μ/σ (e.g. 9.52/4.18 = 2.28), with no risk-free rate subtracted. We do the same, since long-short returns are already excess returns.

Paper results (OOS 1998–2021, N = 500), for later comparison:

| Paper model | K | SR gross | SR net |
|---|---|---|---|
| Attention factors | 8 | 3.35 | 1.94 |
| Attention factors | 15 | 3.81 | 2.25 |
| Attention factors | 30 | 3.97 | 2.28 |
| PCA + LongConv (two-step) | 30 | 2.79 | 1.57 |
| PCA + OU (Avellaneda–Lee) | 3 | 1.26 | −2.72 |
| Attention, past-return characteristics removed | 30 | 1.50 | 0.59 |
| Attention, trading-friction characteristics removed | 30 | 2.90 | 1.34 |

Removing any fundamentals group (value, investment, profitability, intangibles) changes almost nothing. This is why a price/volume-only build is meaningful.

### 2.2 Deviations (deliberate; list them in the README)

| # | Paper | This build | Why |
|---|---|---|---|
| D1 | CRSP 1990–2021 | yfinance + Tiingo, 2008-01-01 → 2026-08-31 | Free data |
| D2 | 500 largest by market cap | 200 most liquid (63-day median dollar volume) point-in-time S&P 500 members | Free sources have no point-in-time shares outstanding for delisted names |
| D3 | 39 characteristics + medians + risk-free rate (79 dims) | 15 price/volume characteristics (§8) | No Compustat. Paper Table 3 shows price-based features drive results |
| D4 | Cross-sectional medians and risk-free rate as inputs | Dropped | With a linear embedding and a softmax over assets, a feature that is constant across assets adds the same amount to every logit in a row. It cancels in the softmax, so it provably has no effect |
| D5 | LongConv | **LSTM** as primary, LongConv as an ablation (§14) | CV spec; paper says sequence models are interchangeable |
| D6 | K = 30 with N = 500 | K tuned over {5, 10, 20} with N = 200 | Smaller cross-section |
| D7 | λ_ridge not specified | λ_ridge = 1e-3 · mean(diag(ω^F ω^Fᵀ)), detached from the gradient | Scale-free numerical stability |
| D8 | Input is the residual sequence | 2 channels: scaled residual and scaled cumulative residual | Lets the LSTM see the spread level directly |
| D9 | How the residual history is composed is not specified | The t−1 composition (ω^F from X_{t−1}) is applied to the whole 30-day window | The history is then the history of exactly the portfolio being traded. It is causal and simple |
| D10 | 24 OOS years | 7.7 OOS years | Data span |
| D11 | Adam | AdamW (decoupled weight decay on sequence params only) + gradient clipping at 1.0 | The paper cites Loshchilov–Hutter; clipping is standard for LSTMs |
| D12 | No halt/delisting rules given | Explicit rules (§7.2) | Needed for SVB-type events |

**Kept exactly as in the paper:**
- Costs: 5 bps per unit of turnover + 1 bp per unit of short exposure per day.
- ‖ω‖₁ = 1, 30-day lookback, and the joint objective.
- 8-year rolling windows with yearly retraining; tuning on the last 2 years of the first window.
- 30 epochs, dropout 0.1, weight decay 0.05, hidden 32, 1 layer, d = 32.
- **No dollar-neutrality constraint.**

### 2.3 Corrections to the earlier draft

1. **A point-in-time constituent list alone does not fix survivorship bias.** yfinance returns *nothing* for delisted names. I verified this on 2026-09-25 with ATVI, TWTR, XLNX, CELG, PXD, CTXS, AGN, TIF, HES, DFS, SIVB and FRC. Tiingo's free tier does have them, so we backfill from Tiingo.
2. **Renamed tickers break naive pipelines.** Example: the membership file has `FB` through 2022-06-08 and `META` from 2022-06-09. Yahoo keeps Facebook's history only under `META`, and `FB` on Yahoo is now an **unrelated ETF** (listed 2025-06-26). An alias table plus coverage validation handles this (§6.4).
3. **The draft's fixed split conflicted with walk-forward and left only ~2.7 OOS years.** The draft had train 2019–23 / validate 2023–24 / test 2024–26. We replace it with the paper's protocol on data starting in 2008, so all of 2019-01 → 2026-08 is OOS. "Backtested over 2019–2026" then becomes literally true.
4. **Stage 2 must not rank or sort into deciles.** Sorting is non-differentiable and breaks joint Sharpe training. The LSTM outputs residual-portfolio weights directly. They are mapped to stocks through the factor projection and L1-normalized, as in the paper.
5. **The lookback is 30 days** (as in the paper), not 60.
6. **Dollar neutrality is not forced.**
   - The paper's portfolio is factor-neutral by construction, which is what makes it market-neutral. The paper reports β ≈ 0.05.
   - Forcing dollar neutrality by demeaning the weights would shift β by roughly *minus the net dollar exposure*.
   - Instead we report β and net dollar exposure, and "market-neutral" in the CV rests on the measured β.
7. **The "turnover penalty" is the cost term itself** (5 bps × turnover), inside the loss. There is no extra penalty.
8. **Market cap is not available point-in-time for free.** Size and liquidity are proxied by dollar volume.
9. **CV wording: this is not self-attention.** Assets do not attend to each other. It is cross-sectional attention: K learned factor queries attend over the assets' characteristic embeddings, and the attention weights *are* the factor portfolio weights.

---

## 3. Architecture at a glance

```
X_{t-1} (N×15 ranks) ─Linear(15→32)─► X̃ ─softmax_assets(Q·X̃ᵀ/√32)─► ω^F (K×N)      [Stage 1]
                                                      │  βᵀ = ω^Fᵀ(ω^Fω^Fᵀ+λI)⁻¹
R_{t-30..t-1} (30×N) ──────────────────────────────► ε_hist = R − βᵀ(ω^F R)  (30×N)
                                                      │
       per-asset shared LSTM on [ε/s, cumsum(ε)/(s√30)] ──► ω^port (N)          [Stage 2]
                                                      │
        ω = ω^port − ω^Fᵀ(β ω^port);  ω ← ω / ‖ω‖₁  ──► weights held on day t
                                                      │
   loss over a 127-day block = −Sharpe(net P&L) − λ_Var · explained variance     [joint]
```

---

## 4. Phase 0: environment and repository

### 4.1 Machine (checked 2026-09-25)
- Arch Linux, system Python 3.14.6, 32 cores, 30 GB RAM, 228 GB free on `/mnt/shared`.
- NVIDIA RTX 5060 Laptop GPU, 8 GB (Blackwell, sm_120), driver 610.43 (CUDA 13.3).
- No uv or conda installed; use `venv`.
- **`/mnt/shared` (the repo's drive) is exFAT:** no symlinks and no Unix permissions. `python -m venv` always creates a symlink, so the virtualenv lives on the ext4 home partition at **`~/.venvs/attention-based-arbitrage`** (50 GB free). Git copes with exFAT by itself (`core.symlinks=false`, `core.filemode=false`).

### 4.2 Setup
```bash
cd /mnt/shared/projects/.vscode/attention-based-arbitrage
git init -b main
make setup     # venv at ~/.venvs/attention-based-arbitrage, then pip install -e ".[dev]"
make lock      # writes requirements-lock.txt (exact installed versions)
~/.venvs/attention-based-arbitrage/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```
Expected output: `2.14.0 True NVIDIA GeForce RTX 5060 Laptop GPU (12, 0)`. The default PyPI torch 2.14.0 wheel is built against CUDA 13.0, which supports Blackwell.

If CUDA is not available:
1. Run `~/.venvs/attention-based-arbitrage/bin/pip install --force-reinstall torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130`.
2. If it still fails, set `train.device: cpu` and log that in `DECISIONS_LOG.md`. The model has about 6k parameters, so CPU training takes minutes per run instead of seconds.

**pyproject.toml.** Package name `statarb`, `src/` layout, setuptools build backend, `requires-python >= 3.12`. All pins below were checked on PyPI and have cp314 wheels:
```
torch==2.14.0  numpy==2.5.3  pandas==3.0.6  pyarrow==25.0.1  scipy==1.18.1
statsmodels==0.15.0  yfinance==1.7.0  matplotlib==3.11.2  pyyaml  requests
[dev] pytest==9.1.1        [notebook] jupyterlab
```
pytest config: `markers = ["slow: trains models"]`. `make test` runs `-m "not slow"`.

pandas 3 notes:
- Copy-on-write is always on, so never use chained assignment; use `.loc`.
- String columns default to the `str` dtype.

### 4.3 User prerequisite: Tiingo key (the only user action)
- Create a free account at tiingo.com and put the token in `.env` at the repo root: `TIINGO_API_KEY=...`.
- `.env` is gitignored, and `.env.example` shows the format.
- The code reads the key from the environment or `.env`. It sends the key **only** in the `Authorization: Token <key>` header, never in URLs or logs.
- Free-tier limits: 50 requests/hour, 1,000/day, 500 unique symbols/month, personal use only. So **never commit raw data**.
- We need about 150–200 symbols, which means one unattended download of about 4 hours.

**No key?** Run with `data.tiingo: false`.
- Delisted names are then excluded.
- The coverage report shows the gap.
- The report must label every result "survivorship-biased: delisted names missing".
- This is the plan's only branch.

### 4.4 Repository layout
```
attention-based-arbitrage/
  PLAN.md  README.md  DECISIONS_LOG.md  Makefile  pyproject.toml  requirements-lock.txt  .gitignore  .env.example
  configs/        base.yaml  grids.yaml  frozen.yaml (written by `freeze`)
  data/meta/      COMMITTED: ticker_aliases.csv  dual_class_drop.csv  sectors_manual.csv  price_fixes.csv
  data/raw/       gitignored: membership/  yahoo/  tiingo/  tiingo_supported_tickers.csv  sectors_yahoo.csv
  data/processed/ gitignored: spells, resolution, panel, member_daily, universe, features, tensors, coverage
  src/statarb/
    __main__.py            CLI (argparse subcommands, §4.5)
    config.py paths.py utils.py (seeding, logging, calendar helpers, config hash)
    data/membership.py data/yahoo.py data/tiingo.py data/resolve.py data/sectors.py
    data/panel.py data/universe.py data/coverage.py
    features.py
    tensors.py
    models/attention_factor.py   AttentionFactors, PCAFactors, projection helpers
    models/lstm_residual.py      LSTMSignal
    models/longconv.py           LongConvSignal
    models/arbitrage.py          ArbitrageModel = factors + projection + signal + mapping
    train.py                     loss, block sampler, fit, predict
    walkforward.py               tune, freeze, oos
    backtest.py                  engine, metrics, bootstrap
    baselines/reversal.py baselines/pairs.py
    bias_checks.py
    report.py
  tests/                         §16
  notebooks/final_comparison.ipynb   thin wrapper that calls report.py functions
  results/runs/ (gitignored)   results/reports/ (committed)   results/oos_runs.csv (committed)
```
Mapping to the draft layout: `features.py`, `models/attention_factor.py`, `models/lstm_residual.py` and `backtest.py` keep their names; `baseline_pairs.py` becomes `baselines/pairs.py`; `notebooks/` stays.

`.gitignore`: `.venv/ .env data/raw/ data/processed/ results/runs/ __pycache__/ *.pt .ipynb_checkpoints/`

**Git.** Local repo only. Commit at the end of every phase and tag `frozen-v1` at the freeze. Do not create or push a remote unless the user asks.

**Attribution (user rule).** Commits are authored by the user's git identity. Commit messages must carry **no Claude co-author trailer**, and PR descriptions must carry no Claude attribution line.

### 4.5 CLI and Makefile
```
python -m statarb membership | yahoo | resolve | tiingo | sectors | panel | universe [--kind pit|survivor]
                  coverage | features | tensors [--universe pit|survivor] [--variant real|noise]
                  tune --strategy attention|pca|pairs|reversal | freeze
                  oos --strategy attention|pca|pairs|reversal [--universe survivor] [--sequence longconv]
                      [--drop-group NAME] [--K n] [--exec-lag 1]
                  checks | report
```
Makefile targets:
- `setup`
- `data`: membership → yahoo → resolve → tiingo → resolve → sectors → panel → universe → coverage
- `features`, `tensors`, `test`, `test-all`, `tune`, `freeze`, `oos`, `checks`, `ablations`, `report`
- `all` runs everything in order.

The Tiingo step runs about 4 hours, so run it in the background. It is resumable.

**Done when:** the torch check passes (or the CPU fallback is logged), `pytest` runs, and the skeleton plus an empty `DECISIONS_LOG.md` are committed.

---

## 5. Global conventions

**Dates**
- `download_start` = 2008-01-01.
- **`data_end` = 2026-08-31.** This date is fixed and never changes.
- **Master calendar:** SPY's trading days on Yahoo within that range.

**OOS and tuning windows**
- OOS runs from 2019-01-02 to 2026-08-31.
- Test year Y ∈ {2019, …, 2026} trains on Jan 1 of Y−8 through Dec 31 of Y−1. For example: 2019 trains on 2011–2018; 2026 trains on 2018–2025 and is tested Jan 2 to Aug 31.
- Tuning trains on 2011-01-01 → 2016-12-31 and validates on 2017-01-01 → 2018-12-31.

**Holding-day convention (critical)**
- ω_t is the weight held from close *t−1* to close *t*.
- It is decided at close *t−1* using only data dated ≤ *t−1*.
- It earns R_t = P_t/P_{t−1} − 1.
- Every model input is stored indexed by holding day *t* and built only from data ≤ *t−1*. The tests enforce this.

**Execution.** Trades happen at the close of *t−1*, which is standard in this literature. A 1-day-lag robustness run is required (§13).

**Returns.** Simple returns computed from the adjusted close (a total-return index).
- Sharpe is annualized by √252 with no risk-free rate subtracted.
- Weights have L1 gross exposure = 1, and returns are per unit of gross exposure.

**Asset id.** `"{source}:{symbol}"`, e.g. `yahoo:META` or `tiingo:ATVI`. The source prefix prevents collisions between reused tickers.

**Seeds and determinism**
- Tuning uses seeds `[0,1,2]`; OOS uses `[0,1,2,3,4]`.
- Seed `torch`, `numpy` and `random`.
- Set `cudnn.deterministic = True` and `cudnn.benchmark = False`.

**Code conventions**
- float32 everywhere except the K×K Gram solve, which runs in float64.
- Library code uses `logging`, not `print`.
- All paths come from the config.
- Tests write to tmp dirs and never touch `results/`.

---

## 6. Phase 1: data acquisition

### 6.1 Point-in-time S&P 500 membership
- **Source:** `github.com/fja05680/sp500`, **pinned to commit `a2430f2af0c79ddf0748e91de11bdeb1616ab5a7`** (2026-09-07; last snapshot 2026-08-18, 503 names).
- **File:** `S&P 500 Historical Components & Changes (Updated).csv`, with columns `date,tickers` (tickers comma-separated). Download it via the raw URL at the pinned SHA into `data/raw/membership/sp500_membership.csv`.
- **Semantics (verified):** each row is a snapshot that takes effect on its date. Membership on day d is the row with the greatest date ≤ d. Example: `FB` is in the 2022-06-08 row and `META` replaces it in the 2022-06-09 row.
- **Spells:** for each ticker, find the maximal runs of consecutive snapshots that contain it. Each run gives `(member_ticker, spell_start, spell_end)`, where `spell_end` is the first snapshot date the ticker is missing (exclusive), or null. Keep spells that overlap [2010-01-01, data_end]; expect about 830 unique tickers.
- **Outputs:** `data/processed/spells.parquet`, plus the function `members_on(date) -> set[str]`.

### 6.2 Yahoo download
- **Symbols:**
  - every member ticker with `.` replaced by `-` (BRK.B → BRK-B, BF.B → BF-B);
  - every alias target whose source is yahoo;
  - `SPY`.
- **Call:** `yf.download(chunk, start="2008-01-01", end="2026-09-01", auto_adjust=False, actions=False, group_by="ticker", threads=True, progress=False)`. Note that `end` is exclusive.
- **Rate limiting:** chunks of 50 with a 3 s sleep between them. On an exception, sleep 60 s and retry (up to 5 times).
- **Storage:** save each non-empty symbol to `data/raw/yahoo/{symbol}.parquet` with columns Date, Open, High, Low, Close, Adj Close, Volume.
  - Skip symbols already cached, so the step is resumable.
  - Write empty results to `data/raw/yahoo_missing.txt`.
  - Never re-download silently; the cache is the source of truth.

### 6.3 Tiingo download
- **Ticker list.** Download `https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip` into `data/raw/tiingo_supported_tickers.csv`. No key is needed. Columns: ticker, exchange, assetType, priceCurrency, startDate, endDate. Do **not** filter by assetType, since some stocks are labeled ETF.
- **What to request.** Only symbols the resolution step needs (§6.4) that also appear in the supported list with dates overlapping the needed range.
- **Request.**
  - URL: `GET https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate=2008-01-01&endDate=2026-08-31&format=json`
  - Header: `Authorization: Token <key>`
  - Response fields: date, open, high, low, close, volume, adjOpen, adjHigh, adjLow, adjClose, adjVolume, divCash, splitFactor. Parse the date field to a plain date.
- **Pacing.** One request every 75 s (48/hour). On HTTP 429, or any non-200 whose body mentions a limit, sleep 3,600 s and retry (up to 3 times).
- **Caching.** Save to `data/raw/tiingo/{symbol}.parquet`. Resumable, and logs progress.
- **Cross-source check.**
  - Also fetch AAPL, JPM, XOM, JNJ and PG from Tiingo.
  - Over 2010–2026, their daily returns must correlate > 0.999 with Yahoo, and the median absolute difference must be < 1 bp.
  - Write `data/processed/source_consistency.csv`. If the check fails, it is a bug; fix it before going on.

### 6.4 Ticker resolution
**Goal.** Map every (member_ticker, date range) to exactly one price series, or explicitly declare it unrecoverable.

**`data/meta/ticker_aliases.csv`** (committed) has these columns:
- `member_ticker, valid_from, valid_to` (blank dates mean open-ended)
- `source` = yahoo, tiingo or none
- `symbol, confidence` (high or verify), `note`

Seed it from **Appendix A**.

**Algorithm.** For each spell, define the required range r = [max(spell_start, 2010-01-01), min(spell_end − 1 day, data_end)]. Then:
1. Split r at the alias `valid_from`/`valid_to` boundaries.
2. Build the candidates for each sub-range:
   - If an alias row matches, it is the only candidate. `source=none` means unrecoverable by decision.
   - Otherwise the candidates are `yahoo:{ticker with . → -}`, then `tiingo:{ticker}`, then `tiingo:{ticker}Q`. The Q suffix is the usual post-bankruptcy form: SIVB became SIVBQ.
3. Accept the first candidate that **passes validation** on the sub-range:
   - valid closes cover ≥ 95% of calendar trading days;
   - the first valid date is ≤ sub-range start + 5 trading days;
   - the last valid date is ≥ sub-range end − 5 trading days.
4. If nothing passes, mark the sub-range `unresolved`. Store hints: the tickers added to the index within ±3 trading days of `spell_end`.
5. Set `review=True` in any of these cases:
   - Tiingo's list has more than one listing for the symbol (a reused ticker);
   - the series has an internal gap of more than 10 trading days inside the sub-range;
   - `confidence=verify`.

**Manual pass.** Done by the builder, not the user. For each unresolved or review row, use public corporate-action information (web search is allowed) to decide which case applies:
- **(a) Rename:** add an alias row.
- **(b) Delisted under another symbol:** add an alias row with source tiingo.
- **(c) Unrecoverable:** add an alias row with `source=none` and a note.

**Reused tickers on Tiingo (found 2026-10-01).** For a reused ticker, Tiingo's price endpoint returns the *current* owner. The old listing can still be fetched by its **permaTicker**:
1. Find it with `GET https://api.tiingo.com/tiingo/utilities/search?query={ticker}&exactTickerMatch=true&includeDelisted=true`, then pick the inactive listing by company name. `TiingoClient.search` does this request.
2. Put the permaTicker in the alias `symbol` column, e.g. STI → `US000000000420` (SunTrust).
3. Limit the alias row's `valid_to` to the old spell, so later reuses of the ticker still resolve normally.

After editing aliases, re-run `yahoo`, then `tiingo`, then `resolve`. Cached symbols are skipped, so only new symbols download. Repeat until no rows are left `unresolved`.

**Expected magnitudes** (from the 2026-09-25 probe, Appendix B):
- about 618 resolve directly on Yahoo;
- about 148 resolve via Tiingo;
- about 65 resolve via aliases;
- about 15–20 are unrecoverable, almost all delisted before 2016 (APOL, BMC, CEPH, CVC, FII, JCP, MOLX, NOVL, QLGC, RSHCQ, SHLD, TLAB, …).

Spells overlapping 2019 or later should have **≤ 5** unrecoverable.

**Output:** `data/processed/resolution.csv` with columns member_ticker, spell_start, spell_end, sub_start, sub_end, status (yahoo, tiingo, alias, none), asset_id, coverage, review, note.

### 6.5 Sectors (pairs formation and factor interpretation)
- **Yahoo assets:** `yf.Ticker(symbol).info["sector"]`, with a 1 s sleep and a cache at `data/raw/sectors_yahoo.csv`. Verified: this works for listed tickers (AAPL → Technology, BKR → Energy) and returns None for delisted ones.
- **Tiingo assets:** fill `data/meta/sectors_manual.csv` (asset_id, sector) for every Tiingo asset that appears in any top-200 universe from 2018-01 on.
  - Use Yahoo's 11 sector names: Technology, Healthcare, Financial Services, Consumer Cyclical, Consumer Defensive, Industrials, Energy, Utilities, Real Estate, Basic Materials, Communication Services.
  - If a sector is unknown, write `Unknown`. That asset is then only excluded from pair formation.
- **Documented caveat:** these are current sector classifications, not point-in-time. The effect is minor.

**Done when:**
- spells are built;
- all Yahoo and Tiingo downloads are cached;
- the source-consistency check passes;
- `resolution.csv` has no `unresolved` rows, and every alias row has a note;
- sectors are complete;
- the phase is committed.

---

## 7. Phase 2: panel, delisting rules, universe, coverage report

### 7.1 Panel
For each asset_id, reindex to the master calendar and drop dates outside it. Columns:

| Column | Yahoo | Tiingo |
|---|---|---|
| `close_adj` (total-return index) | Adj Close | adjClose |
| `close_raw` (for dollar volume and the $5 filter) | Close | close |
| `high`, `low` (only same-day ratios are used) | High, Low | adjHigh, adjLow |
| `volume` | Volume | volume |

Derived fields:
- `valid` = close_adj is present and > 0.
- `dollar_volume` = close_raw × volume, or NaN if volume ≤ 0.
- `ret` = close_adj_t / close_adj at the previous valid day − 1 on valid days, NaN otherwise. This means **the whole return over a halt is booked on the first valid day after it.**

**Price fixes.** Apply `data/meta/price_fixes.csv` (asset_id, date, note), which sets `ret` to NaN. Use it only for confirmed data errors.

**Return review.**
- List every |ret| > 40% while the asset is eligible, and check each one against known events.
- Confirmed errors go into `price_fixes.csv`. Real events, such as SVB, stay.
- A price-level jump where two securities were spliced together (a reused ticker) shows up here too. Fix it with an alias.

**Membership panel.** `member[t, asset]` is true if any member ticker that maps to that asset (through the resolution sub-ranges) is in the snapshot for *t*.

**Outputs:** `data/processed/panel.parquet` (long format) and `member_daily.parquet`.

### 7.2 Halts and delistings (implemented in the engine, §9.1)
- **Tradability.** An asset is tradable at close *t* if it is valid at *t*. If it is not tradable at close *t*, its position is **frozen** through day *t+1*.
- **Halts.** A gap of ≤ 60 trading days followed by a valid close books the gap return on that close (this follows from the `ret` definition).
- **Permanent end.** This means no valid close after t_last up to data_end with t_last < data_end, or a gap longer than 60 days (treated as ending at t_last). In either case:
  - the position earns 0 after t_last;
  - it is **removed at no cost** from holding day t_last+2.

  This models a cash-out at the last close. It is right for mergers and optimistic for bankruptcies, so document it.
- **Training.** The training loss uses a simpler rule: non-tradable asset-days are masked out.
- **Check SVB and First Republic.** After downloading, check `tiingo:SIVBQ` and `tiingo:FRCB`. The report must say whether post-halt OTC prices exist, in which case the collapse is booked, or not, in which case the optimistic cash-out applies.

### 7.3 Universe (monthly, point-in-time)
The formation date f_m is the last trading day of month m−1. An asset is **eligible** at f_m if:
1. it is an S&P 500 member on f_m (snapshot ≤ f_m) and resolved to an asset_id;
2. its asset_id is not in `data/meta/dual_class_drop.csv`. The seed list is `yahoo:GOOG`, `yahoo:FOX`, `yahoo:NWS`, `tiingo:DISCK` and `yahoo:UA`; we keep GOOGL, FOXA, NWSA, DISCA and UAA. Pre-2014 member `GOOG` was class A, so it maps to `yahoo:GOOGL` via alias (Appendix A);
3. it is valid at f_m with close_raw ≥ $5;
4. it has ≥ 240 valid returns in the 252 trading days ending f_m;
5. its 63-day median dollar volume (with ≥ 50 valid days) is defined. This is the ranking key.

**U_m** is the top 200 eligible assets by that key, sorted by (dv63 descending, asset_id ascending). If fewer than 200 are eligible, take all of them and log it. For holding day *t* in month m, **U_t** = U_m ∩ {assets valid at *t−1*}.

**Duplicate check.** Log any pair in U_m whose 252-day return correlation is above 0.97. If both are the same company, add one to `dual_class_drop.csv`.

**Range.** Formations run from 2010-12-31 (for Jan 2011) to 2026-07-31 (for Aug 2026).

**Output:** `data/processed/universe_{kind}.parquet` with columns month, asset_id, rank, dv63.

**Variant `--kind survivor`** (used only in §13):
- Members are taken from the **2026-08-18 snapshot** only, i.e. today's index.
- Yahoo only, same ranking.
- This is the naive, survivor-biased yfinance universe.

### 7.4 Coverage report
`data/processed/coverage_report.csv` plus a figure (F7). For each year 2010–2026 it shows:
- point-in-time member-days;
- the share resolved via Yahoo, Tiingo and aliases, and the share unrecoverable;
- the number of U_m slots filled by Tiingo-only assets, i.e. names a yfinance-only pipeline would silently miss.

It also lists every unrecoverable name with its ticker, spell and reason.

**Done when:**
- the panel is built and the |ret| > 40% review is logged;
- the universe has 200 names for every month from 2011-01 to 2026-08;
- the coverage report is written;
- the panel and universe tests pass.

---

## 8. Phase 3: features

There are 15 characteristics, computed per asset on the master calendar using only data dated ≤ *t*. Rules:
- pandas rolling windows are trailing. **Never** use `center=True`, `bfill`, `shift(-k)` or full-sample statistics.
- Notation: P = `close_adj` forward-filled for up to 5 days; r = `ret`; m = SPY daily return; DV = `dollar_volume`.
- Windows are in trading days, and `min_periods` = ceil(0.8 × window).

| Group | Feature | Definition at date t | Paper analogue |
|---|---|---|---|
| past_returns | ret_1d | r_t | Ret_D1 |
| | ret_5d | P_t/P_{t−5} − 1 | Ret_W1 |
| | ret_21d | P_t/P_{t−21} − 1 | ST_Rev, r2_1 |
| | mom_12_2 | P_{t−21}/P_{t−252} − 1 | r12_2 |
| | mom_12_7 | P_{t−126}/P_{t−252} − 1 | r12_7 |
| | mom_36_13 | P_{t−252}/P_{t−756} − 1 | r36_13 |
| | rel_52w_high | P_t / max(P_{t−251..t}) | Rel2High |
| risk | vol_5d | std(r_{t−4..t}) | STD_W1 |
| | vol_63d | std(r_{t−62..t}) | Variance |
| | beta_252d | cov(r, m)/var(m) over t−251..t | Beta |
| | ivol_63d | std(r_τ − beta_252d_τ·m_τ), τ = t−62..t | Resid_Var |
| frictions | dvol_21d | mean(DV_{t−20..t}) | LME / Vol (size proxy) |
| | dvol_ratio | mean(DV_{t−4..t}) / mean(DV_{t−251..t}) | SUV / LTurnover |
| | hl_range_21d | mean(ln(high/low)) over t−20..t | Spread proxy |
| | amihud_21d | mean(\|r_τ\|/DV_τ) over t−20..t | illiquidity |

The rank transform makes monotone rescaling irrelevant, so no logs are applied.

**Normalization** is done when building tensors (§10.1), within each holding day's universe:
1. Forward-fill each raw feature for up to 5 days.
2. For holding day *t*, take the raw value at *t−1* for the assets in U_t.
3. Rank with average ties and map to (rank−1)/(n−1) − 0.5 ∈ [−0.5, 0.5].
4. Missing values become 0, which is the cross-sectional median. If n < 2, set everything to 0.

**Output:** `data/processed/features.parquet` with columns date, asset_id and the 15 raw features.

**Done when:** the known-value tests and the truncation test (§16) pass.

---

## 9. Phase 4: backtest engine and non-ML baselines

### 9.1 Engine (`backtest.py`)
**Input:** target weights W in long format (holding_day, asset_id, weight), plus the panel. For each holding day *t*:
1. `target_t = W_t` (0 where absent). With `exec_lag = 1`, use `target_t = W_{t−1}` instead.
2. For an asset that was not tradable at close *t−1* and has pos_{t−1} ≠ 0, keep the frozen position pos_t = pos_{t−1}. Otherwise pos_t = target_t. Then apply the permanent-end removals (§7.2).
3. gross_t = Σ pos_t · ret_t, counting NaN returns as 0.
4. turnover_t = Σ |pos_t − pos_{t−1}| over tradable assets. Cost-free removals are excluded.
5. short_t = Σ max(−pos_t, 0).
6. cost_t = tc_bps/1e4 · turnover_t + short_bps/1e4 · short_t, with defaults 5 and 1.
7. net_t = gross_t − cost_t.

The paper's turnover formula ignores drift between trades. It is used everywhere, in training and in the backtest. Positions start from zero on the first evaluated day.

### 9.2 Metrics (for both gross and net)
- **Returns and risk:**
  - annualized mean (×252), volatility (×√252), Sharpe;
  - Sharpe SE = √252·√((1 + SR_d²/2)/T), where SR_d is the daily Sharpe (Lo, 2002);
  - a **95% CI from a stationary block bootstrap** (mean block length 21, 2,000 resamples, seed 0).
- **Drawdown and hit rate:**
  - max drawdown of cumprod(1 + net);
  - Calmar ratio;
  - hit rate.
- **Exposure and turnover:**
  - average daily turnover;
  - average gross exposure;
  - average net dollar exposure (Σ pos);
  - average short exposure.
- **Market beta** vs SPY: OLS with intercept, Newey–West t-stat with 5 lags, via `statsmodels` `cov_type="HAC"`.
- **Per-year Sharpe.**
- **Sharpe differences** between two strategies: use the same bootstrap indices for both.

### 9.3 Short-term reversal baseline
w_t ∝ −(ret_kd at t−1 − its mean over U_t), then L1-normalized. k is tuned over {1, 5, 21} (§11).

### 9.4 Pairs trading baseline (`baselines/pairs.py`)
**Schedule**
- Formation dates are the first trading day of January and July.
- The formation window is the 252 trading days ending the day before.
- The trading period runs to the next formation date, or to data_end.

**Candidates.** Same-sector pairs (sector not `Unknown`) within U of the formation month, with ≥ 247 of 252 valid days. Forward-fill up to 5 days for the tests.

**Test each pair in both orientations**
- Run `statsmodels.tsa.stattools.coint(log P_y, log P_x, trend="c", autolag="aic")` and keep the orientation with the lower p-value.
- Estimate α and β with OLS: log P_y = α + β log P_x.
- The spread is s = log P_y − β log P_x − α. Its μ_s and σ_s come from the formation window.
- Half-life comes from regressing Δs_t = a + φ s_{t−1}: hl = −ln 2 / ln(1+φ). It is valid only if −1 < φ < 0.

**Select pairs**
1. Keep pairs with p < 0.05, 1 ≤ hl ≤ 63 and 0.2 ≤ β ≤ 5.
2. Sort by p ascending.
3. Pick greedily up to `max_pairs`, with each asset in at most 2 selected pairs.

**Trade.** z = (s − μ_s)/σ_s is computed from closes at *t−1* and sets the position for day *t*.
- Enter short-spread when z ≥ +entry, and long-spread when z ≤ −entry.
- Exit when |z| ≤ 0.5.
- Stop-loss when |z| ≥ 4. The pair is then disabled until the next formation.
- Close everything at the end of the trading period.
- Close the pair if either leg is not tradable.

**Weights of an open long-spread pair.** y gets +g/(1+β) and x gets −g·β/(1+β), with g = 1/max_pairs. A short spread is the reverse. Weights are summed across pairs, so gross exposure is ≤ 1. This is the committed-capital convention of Gatev et al.; report the average gross exposure.

**Evaluation.** The same engine, costs and windows as everything else.

**Done when:**
- engine tests pass (§16): a hand-computed example; the alignment test, where cheating with w_t = sign(R_t) must give an absurd Sharpe; freezing; lag;
- pairs tests pass on synthetic series;
- reversal and pairs run on 2011–2018.

---

## 10. Phase 5: the model

### 10.1 Tensors (`tensors.py`)
Built once per `universe ∈ {pit, survivor}` and `variant ∈ {real, noise}`, saved as `data/processed/tensors_{universe}_{variant}.pt`:
- `holding_days`: every holding day from 2011-01-03 to 2026-08-31.
- `asset_ids`: A = the union of all U_m. Expect about 350–450 assets.
- `R_cal`: (T_cal, A) float32 returns with NaN; `valid`: (T_cal, A) bool.
- `Xn`: (T_hold, A, 15) normalized features for holding day *t*, taken from *t−1* and ranked within U_t. Zero outside U_t.
- `U`: (T_hold, A) bool.
- `pca_wF`: (T_hold, 20, A) PCA factor weights (§10.3).

Everything fits on the 8 GB GPU (about 300 MB total).

**Input scale `s`.** For each training window: the median, over the window's holding days, of the cross-sectional std of R over U_t. It uses raw returns only and is stored in the checkpoint.

### 10.2 Attention factors (`models/attention_factor.py`)
```
x̃      = Linear(15 → 32, bias=False)(Xn)         # (B, A, 32); a bias would cancel in the softmax
logits = Q x̃ᵀ / √32,  Q = Parameter(randn(K, 32))  # (B, K, A)
logits[~U] = −inf;  ω^F = softmax(logits, dim=−1) # each factor row sums to 1 over active assets
```

### 10.3 PCA factors (two-step baseline, no learned factor parameters)
For each holding day *t*:
1. Take the returns of the U_t assets over *t−252..t−1*, with NaN set to 0.
2. Standardize each column and build the correlation matrix.
3. Take the top-K eigenvectors V.
4. Set ω^F = Vᵀ diag(1/σ). These are Avellaneda–Lee eigenportfolios.

Precompute for K_max = 20 and slice for smaller K.

### 10.4 Projection and residuals (shared by both factor types)
```
G  = ω^F ω^Fᵀ                     (float64)
λ  = 1e-3 · mean(diag(G))          (detached)
βᵀ = ω^Fᵀ (G + λI)⁻¹               via torch.linalg.solve; cast back to float32
History: R_h = R_cal rows t−30..t−1 (NaN→0, masked to U_t)
         F_h = ω^F R_hᵀ;   ε_h = R_h − (βᵀ F_h)ᵀ           # (B, 30, A)
Realized residual for EV: ε_t = R_t − βᵀ(ω^F R_t)
```
Never materialize the A×A matrix ω^ε.

### 10.5 Sequence models (per active (t, i), shared weights)
**Input:** x_τ = [ε_h,τ / s, cumsum(ε_h)_τ / (s·√30)] for τ = 1..30, giving shape (n_active, 30, 2).

**`LSTMSignal` (primary):** `LSTM(2→32, 1 layer, batch_first)` → last hidden state → `Dropout(0.1)` → `Linear(32→1)`.

**`LongConvSignal` (ablation):**
1. `Linear(2→32)`.
2. LongConv:
   - one depthwise kernel per channel, 𝒦 ∈ R^{32×30}, plus a skip term D ∈ R^{32}: y = 𝒦 ∗ u + D ⊙ u;
   - the causal FFT convolution zero-pads to length 60;
   - the squash 𝒦̄ = sign(𝒦)·max(|𝒦| − 0.001, 0) is applied in the forward pass;
   - initialization: 𝒦[h, t] = x·exp(−(t/30)·16^{h/32}) with x ~ N(0,1), for t = 1..30 and h = 1..32.
3. `GELU` → `Dropout(0.1)` → last step → `Linear(32→1)`.

The outputs are the scores ω^port. Scatter them to (B, A), with 0 outside U.

### 10.6 Mapping to stock weights
```
ω_raw = ω^port − ω^Fᵀ (β ω^port)        # = (ω^ε)ᵀ ω^port
ω     = ω_raw / (Σ|ω_raw| + 1e-8)       # gross 1; no dollar-neutral constraint (§2.3 item 6)
```

### 10.7 Loss (per block of B+1 = 127 consecutive holding days; day 0 only supplies ω_{t−1})
```
gross_t = Σ_i ω_t,i · R_t,i          (masked; NaN→0)                     t = 1..126
turn_t  = Σ_i |ω_t,i − ω_{t−1},i|    in global asset space → leaving the universe counts as a sale
short_t = Σ_i relu(−ω_t,i)
net_t   = gross_t − 5e-4·turn_t − 1e-4·short_t
SR      = mean(net) / (std(net, population) + 1e-8)                       # daily, not annualized
EV      = mean over assets active ≥ 20 block days of [1 − var(ε_i)/(var(R_i) + 1e-12)]  (masked, centered)
loss    = −SR − λ_Var·EV   (attention)          loss = −SR   (PCA: factors fixed)
```

### 10.8 Training (`train.py`)
**Block sampler.** Each epoch:
1. Draw a random offset in [0, 126).
2. Cut consecutive 127-day blocks from the window start + offset.
3. Drop a trailing block shorter than 63 days.
4. Shuffle the block order.

**Optimizer and schedule**
- One block per optimizer step.
- AdamW with the configured lr: weight decay 0.05 on sequence-model parameters and 0 on factor parameters.
- Gradient clipping at 1.0.
- 30 epochs, with no early stopping and no LR schedule.

**Logging and checkpoints**
- Log per epoch: train SR, EV and mean turnover.
- Checkpoint the state dict, `s` and the config.
- Retrain **from scratch** for every window and every seed.

**Scale.** About 6k parameters and about 480 steps per 8-year run, which takes seconds on the GPU.

### 10.9 Inference
- Run `model.eval()` with `torch.no_grad()` over the test period in one pass.
- Save per-seed weights to `results/runs/{run}/weights_seed{s}.parquet` in long format.
- The **ensemble** averages the per-seed weights, then L1-renormalizes each day.
- The engine computes turnover across year boundaries between consecutive models automatically.

**Done when:**
- model unit tests pass;
- the planted-signal test passes (§16);
- a smoke run (train 2011, predict Jan–Feb 2012) finishes without NaNs.

---

## 11. Phase 6: tuning and freeze

**Protocol.** Train on 2011–2016, validate on 2017–2018. The metric is **validation net Sharpe** from the engine (§9), averaged over seeds [0,1,2].

| Strategy | Grid | Fixed | Tie-break (any config within 0.2 of the best) |
|---|---|---|---|
| Attention + LSTM | K {5,10,20} × λ_Var {1,10,100} × lr {1e-3,3e-3} | d 32, hidden 32, 1 layer, dropout 0.1, L 30, 30 epochs, wd 0.05, block 126 | λ_Var 100 → lr 3e-3 → K 10 |
| PCA + LSTM | K {5,10,20} × lr {1e-3,3e-3} | same | lr 3e-3 → K 10 |
| Pairs | entry_z {1.5,2.0,2.5} × max_pairs {10,20,40} | exit 0.5, stop 4, p 0.05, hl [1,63] | (2.0, 20) |
| Reversal | k {1,5,21} | – | 5 |

**One pre-authorized extension.** If the best attention config has validation net Sharpe ≤ 0:
1. At the best (K, λ_Var, lr), also try lookback L ∈ {20, 60} and hidden ∈ {16, 64}. That is 4 configs × 3 seeds.
2. Choose using the same tie-break rule.
3. **Freeze regardless of the outcome.**

Also run the null-data test (§13.3) now; it uses the tuning window.

**Freeze**
1. `python -m statarb freeze` writes `configs/frozen.yaml`, containing the chosen parameters for all four strategies plus the validation table.
2. Save `results/reports/tuning.csv`.
3. Commit, then `git tag frozen-v1`.
4. From here on, `oos` **refuses to run** without `frozen.yaml` or with a dirty git tree. Every OOS run appends (timestamp, commit, config hash, run name, net Sharpe) to `results/oos_runs.csv`.

---

## 12. Phase 7: out-of-sample walk-forward

For each Y = 2019…2026:
- Train on [Y−8, Y−1] with the frozen hyperparameters and seeds [0..4].
- Predict year Y.

Strategies: attention + LSTM, PCA + LSTM, reversal, pairs. Concatenate the weights, run the engine, and compute metrics.

**Headline** = attention + LSTM, 5-seed ensemble, net of costs, exec_lag 0, 5 bps + 1 bp. Also report the per-seed mean ± std, which is the paper's convention.

---

## 13. Phase 8: bias and robustness checks

All checks live in `bias_checks.py` and write to `results/reports/bias_checks.md`.
1. **Truncation tests** (§16). Include the list of passes and failures.
2. **Future-perturbation test** on trained OOS models. For 5 random dates t*, replace all data dated after t* with noise. Run this check on CPU. Weights for holding days ≤ t*+1 must match to within 1e-7.
3. **Null-data test.**
   - Replace returns with iid N(0, σ_i²), keeping the validity masks.
   - Recompute the return-based features; keep the volume features as they are.
   - Make 3 independent noise draws (noise seeds 0, 1, 2) and train one model on each, on the tuning split.
   - **Pass** if |mean validation gross Sharpe| < 1.0, which is about 2.4 SE.
   - A failure counts as a leakage bug. Bisect with the truncation tests until it is found; it is not something to reinterpret.
4. **Execution lag.** Rerun the engine with exec_lag = 1 for every strategy.
5. **Cost sensitivity.** Evaluate at 0, 2, 5 and 10 bps, with the short cost fixed at 1 bp. This is evaluation only; no retraining.
6. **Survivorship.**
   - Include the coverage report (§7.4).
   - Retrain and evaluate the frozen main model on the `survivor` universe with the same OOS protocol.
   - Report ΔSharpe = survivor − point-in-time, with a bootstrap CI.
7. **SVB/FRC outcome.** Report which rule applied (§7.2).

---

## 14. Phase 9: ablations (optional)

These run after the freeze with the frozen hyperparameters and 5 seeds. They never replace the headline.
1. **LongConv** instead of the LSTM, which is the paper's sequence model.
2. **Drop feature groups**, one at a time: past_returns, risk, frictions. This is the analogue of paper Table 3.
3. **K sensitivity:** K ∈ {5, 10, 20}. This is the analogue of paper Table 2.

---

## 15. Phase 10: reporting

### 15.1 Tables (CSV + markdown in `results/reports/`)
- **T1, main OOS results.**
  - Rows: attention + LSTM (ensemble); attention + LSTM (seed mean ± sd); PCA + LSTM; pairs; reversal; equal-weight universe (long-only, gross); SPY.
  - Columns: SR gross; SR net with 95% CI; net μ; σ; max drawdown; turnover per day; gross exposure; net dollar exposure; β (t-stat).
- **T2:** net Sharpe per year.
- **T3, robustness:** cost sensitivity, execution lag, survivor vs point-in-time.
- **T4:** ablations.
- **T5:** the paper vs ours (§2.1 numbers next to our results).
- **T6:** tuning.

### 15.2 Figures (matplotlib PNG)

| File | Content |
|---|---|
| `f1_cumulative.png` | Cumulative net returns 2019–2026, all strategies, vol-scaled ex post to 10%/yr (labeled as such), plus a raw version |
| `f2_drawdown.png` | Drawdowns |
| `f3_rolling_sharpe.png` | Rolling 252-day net Sharpe |
| `f4_turnover.png` | Daily turnover |
| `f5_factor_sectors.png` | Heatmap of factor weights summed by sector, on 2024-01-02, for the 2024 model with seed 0 (analogue of paper Fig. 4) |
| `f6_top_names.csv` | Top-10 names per factor on 2024-01-02 |
| `f7_coverage.png` | Coverage by year, stacked by source |

### 15.3 README
- Summary and headline table.
- How to reproduce: make targets and the Tiingo key.
- Deviations (§2.2).
- Limitations:
  - price/volume-only features;
  - 7.7 OOS years and the Sharpe SE;
  - close-price execution;
  - flat linear costs;
  - current, not point-in-time, sectors;
  - any remaining unrecoverable names.
- Sources.

### 15.4 CV bullets (fill in the braces from T1 and T3)
- *Built a simplified reproduction of "Attention Factors for Statistical Arbitrage" (arXiv:2510.11616). Cross-sectional attention with learned factor queries over price/volume characteristics produces latent factor portfolios. An LSTM over 30-day factor residuals outputs residual-portfolio weights. Both are trained jointly end to end on net-of-cost Sharpe.*
- *Walk-forward backtest (annual retraining, out-of-sample 2019–2026) on the 200 most liquid point-in-time S&P 500 members, delisted names included. Net Sharpe {X} (gross {Y}; 5 bps + 1 bp short cost; market β {b}) vs {P} for cointegration pairs trading and {Q} for a two-step PCA benchmark. Verified no look-ahead with truncation and null-data tests, and quantified survivorship bias ({Δ} Sharpe vs a today's-constituents universe).*
- If net Sharpe ≤ 0, say it plainly: *"gross Sharpe {Y}; 5 bps costs reduce it to {X}"*. This is still a credible, rigorous result. Never claim a number that isn't in T1.
- Only name LongConv or ablations in the CV if Phase 9 was completed.

**Done when:** `make report` regenerates everything, the README and CV bullets are filled, and the final commit is made.

---

## 16. Test suite (pytest; the `slow` marker is for tests that train)

| File | Must assert |
|---|---|
| test_membership.py | `members_on("2020-06-01")` contains FB and not META; on 2022-07-01 the reverse. Every snapshot since 2010 has 495–510 names. FB's spell is [2013-12-23, 2022-06-09). |
| test_resolve.py (synthetic, no network) | A series starting after the range start is rejected (the FB/ETF case). Alias sub-range splitting works (IR → TT before 2020-03-02, IR after). Coverage < 95% is rejected. An internal gap > 10 days is flagged. |
| test_panel.py | Prices 100, NaN, NaN, 50 → ret on day 4 = −0.5. Dollar volume = close_raw × volume. |
| test_universe.py | 200 names per month when ≥ 200 are eligible. Dual-class names are excluded. Thresholds hold. **Truncation:** U_m is identical when all data after f_m is deleted. |
| test_features.py | Known values on a synthetic geometric path (e.g. ret_5d = (1+g)⁵ − 1). beta = 2 when r = 2m. rel_52w_high = 1 at a new high. **Truncation:** features at dates ≤ t* are unchanged when data after t* is deleted (5 random t*). Rank range is [−0.5, 0.5], median 0, missing → 0, ranked only within U_t. |
| test_tensors.py | Perturbing data at day t and later leaves Xn[t] unchanged (while Xn[t+1] changes). The residual window for holding day t is exactly rows t−30..t−1. |
| test_projection.py | ‖ω^F ε‖/‖F‖ < 1e-2. The mapped weights equal the explicit (I − βᵀω^F)ᵀω^port on a small N (float64, tol 1e-6). Masked assets get zero weight. Permuting assets permutes the outputs (equivariance). A feature that is constant across assets leaves ω^F unchanged (proves D4). |
| test_models.py | Output shapes. LongConv causality: changing input step τ leaves outputs < τ unchanged. Squash zeroes small kernel entries. Perturbing the realized R_t leaves ω_t unchanged. |
| test_loss.py | SR, turnover, short cost and EV match a NumPy reference. An asset leaving the universe counts \|w\| as turnover. |
| test_backtest.py | A hand-computed 3-asset × 4-day example. Weights = sign(R_t) give Sharpe > 10 (alignment check). Weights = sign(R_{t−1}) on iid noise give a small \|SR\|. A position stays frozen through a halt. A permanent end is removed at no cost. exec_lag shifts by one day. |
| test_pairs.py | Synthetic y = 1.5x + OU noise: coint p < 0.01, β within 1.5 ± 0.1, trades open at \|z\| ≥ 2 and close at \|z\| ≤ 0.5, positive gross P&L. Independent random walks are rejected in ≥ 90% of 50 seeded draws. |
| test_planted_signal.py (slow) | Synthetic data: N = 60, T = 1,500, 3 factors with loadings linear in 5 characteristics, AR(1) residuals with φ = −0.3, 1% daily idiosyncratic vol. Train 10 epochs on the first 1,000 days → validation gross SR > 2. The null version (φ = 0) gives \|SR\| < 2.5. |

---

## 17. Config reference: `configs/base.yaml`

```yaml
paths:                      # relative to the repo root (resolved by statarb.paths.project_paths)
  raw: data/raw
  processed: data/processed
  meta: data/meta
  runs: results/runs
  reports: results/reports
dates:
  download_start: "2008-01-01"
  data_end: "2026-08-31"
  first_holding_day: "2011-01-03"
  oos_start: "2019-01-01"
  train_years: 8
  tune_train: ["2011-01-01", "2016-12-31"]
  tune_valid: ["2017-01-01", "2018-12-31"]
data:
  membership_sha: a2430f2af0c79ddf0748e91de11bdeb1616ab5a7
  membership_file: "S&P 500 Historical Components & Changes (Updated).csv"
  spells_from: "2010-01-01"
  tiingo: true
  tiingo_seconds_between_requests: 75
  yahoo_chunk: 50
  coverage_min: 0.95
  edge_tolerance_days: 5
  review_gap_days: 10
  gap_max_days: 60
  return_review_threshold: 0.40
universe:
  kind: pit                 # pit | survivor
  survivor_snapshot: "2026-08-18"
  size: 200
  dv_window: 63
  dv_min_obs: 50
  history_days: 252
  history_min_valid: 240
  min_price: 5.0
  dup_corr: 0.97
features:
  groups:
    past_returns: [ret_1d, ret_5d, ret_21d, mom_12_2, mom_12_7, mom_36_13, rel_52w_high]
    risk: [vol_5d, vol_63d, beta_252d, ivol_63d]
    frictions: [dvol_21d, dvol_ratio, hl_range_21d, amihud_21d]
  drop_groups: []
  ffill_days: 5
  min_periods_frac: 0.8
costs: {tc_bps: 5.0, short_bps: 1.0}
model:
  factors: attention        # attention | pca
  K: 10
  d: 32
  ridge_rel: 1.0e-3
  sequence: lstm            # lstm | longconv
  hidden: 32
  layers: 1
  dropout: 0.1
  lookback: 30
  squash: 1.0e-3
train:
  epochs: 30
  lr: 3.0e-3
  weight_decay_seq: 0.05
  block_days: 126
  min_block_days: 63
  lambda_var: 100.0
  ev_min_days: 20
  grad_clip: 1.0
  device: auto              # auto | cuda | cpu
  seeds_tune: [0, 1, 2]
  seeds_oos: [0, 1, 2, 3, 4]
backtest: {exec_lag: 0, annualization: 252, bootstrap_n: 2000, bootstrap_block: 21, nw_lags: 5}
reversal: {lookback: 5}
pairs:
  formation_days: 252
  min_valid_days: 247
  pvalue_max: 0.05
  halflife: [1, 63]
  beta: [0.2, 5.0]
  max_pairs: 20
  max_pairs_per_asset: 2
  entry_z: 2.0
  exit_z: 0.5
  stop_z: 4.0
report: {vol_target: 0.10, factor_date: "2024-01-02"}
```
`configs/grids.yaml` holds the §11 grids. `configs/frozen.yaml` overrides `base.yaml` for OOS runs.

---

## 18. Timeline (part-time, about 3 working days per week)

| Week | Phases | Output |
|---|---|---|
| 1 | 0, 1 (the Tiingo download runs unattended) | environment, raw data, resolution |
| 2 | 2, 3 | panel, universe, coverage report, features + causality tests |
| 3 | 4, start of 5 | engine, reversal, pairs, tensors |
| 4 | 5, 6 | model + tests, tuning, **freeze** |
| 5 | 7, 8, 9, 10 | OOS, bias checks, ablations, report, README, CV |

If time runs short:
- Phases 0–8 and 10 are the minimum; the CV claims depend on them.
- Phase 9 is extra; do not mention it in the CV if it was skipped.

---

## 19. Risks and pre-decided responses

| Risk | Response |
|---|---|
| Yahoo rate limiting | Chunks of 50, 3 s sleep, 60 s backoff × 5, cache, resume |
| Tiingo limits | 75 s pacing, 1 h sleep on 429, resumable, pre-check the supported list |
| Reused ticker gives the wrong company | Coverage validation + review flags + \|ret\| > 40% review → fix with an alias |
| Name cannot be recovered | Alias `none` + listed in the coverage report |
| GPU unusable | CPU fallback, logged |
| Model learns nothing on validation | One grid extension (§11), then freeze anyway and report honestly |
| NaN during training | This is a bug. Check for all-masked softmax rows, the G solve, and division by zero in the L1 normalization. Fix it and re-run tuning |
| OOS result disappoints | Report it. No re-tuning on OOS data. Use the honest CV phrasing (§15.4) |
| Yahoo data changes between runs | Cached raw files are the source of truth; never re-download silently |

## 20. Out of scope (do not build unless the user asks)
- CRSP, Compustat or WRDS data; fundamentals; intraday data.
- A PCA + OU (Avellaneda–Lee) baseline; an attention two-step variant; Transformers.
- Hyperparameter search beyond §11.
- Market impact beyond linear bps; hard-to-borrow fees; risk-free modeling.
- Live or paper trading.
- A GitHub remote or any publishing.
- A t-SNE plot (the sector heatmap replaces it).

---

## Appendix A: alias seeds for `data/meta/ticker_aliases.csv`

The resolution step still validates every row. Rows marked `verify` describe events after mid-2026 or have uncertain lineage; confirm them during the manual pass (§6.4).

```csv
member_ticker,valid_from,valid_to,source,symbol,confidence,note
FB,,,yahoo,META,high,Facebook→Meta 2022-06-09; Yahoo FB is now an unrelated ETF (2025-06-26)
ANTM,,,yahoo,ELV,high,Anthem→Elevance 2022-06-28
ABC,,,yahoo,COR,high,AmerisourceBergen→Cencora 2023-08-30
BLL,,,yahoo,BALL,high,Ball ticker change 2022-05
COG,,,tiingo,CTRA,verify,Cabot→Coterra 2021-10; Yahoo CTRA empty as of 2026-09
CTL,,,yahoo,LUMN,high,CenturyLink→Lumen 2020-09
DISCA,,,yahoo,WBD,high,Discovery→Warner Bros. Discovery 2022-04
DWDP,,,yahoo,DD,high,DowDuPont→DuPont 2019-06
FBHS,,,yahoo,FBIN,high,Fortune Brands Home & Security→Fortune Brands Innovations 2022-12
FLT,,,yahoo,CPAY,high,Fleetcor→Corpay 2024-03
HRS,,,yahoo,LHX,high,Harris→L3Harris 2019-07
JEC,,,yahoo,J,high,Jacobs 2019-12
LB,,2021-08-03,yahoo,BBWI,high,L Brands→Bath & Body Works; LB reused by another company from 2024
SYMC,,,yahoo,GEN,high,Symantec→NortonLifeLock→Gen Digital
NLOK,,,yahoo,GEN,high,NortonLifeLock→Gen Digital 2022-11
HCP,,,yahoo,DOC,high,HCP→Healthpeak (PEAK)→DOC
PEAK,,,yahoo,DOC,high,Healthpeak→DOC 2024-03
PKI,,,yahoo,RVTY,high,PerkinElmer→Revvity 2023-05
RE,,,yahoo,EG,high,Everest Re→Everest Group 2023-07
TMK,,,yahoo,GL,high,Torchmark→Globe Life 2019-08
UTX,,,yahoo,RTX,high,United Technologies→RTX 2020-04
WLTW,,,yahoo,WTW,high,Willis Towers Watson 2022-01
KORS,,,yahoo,CPRI,high,Michael Kors→Capri
GPS,,,yahoo,GAP,high,Gap Inc. ticker change 2024
HFC,,,yahoo,DINO,high,HollyFrontier→HF Sinclair 2022-03
CDAY,,,tiingo,DAY,verify,Ceridian→Dayforce 2024-02; taken private (Tiingo DAY ends 2026-02-03)
ARNC,2016-11-01,2020-04-06,yahoo,HWM,verify,Arconic Inc.→Howmet 2020-04; ARNC later reused by Arconic Corp
ARNC,,2016-10-31,none,,high,pre-split Alcoa Inc. history not available on Yahoo
IR,,2020-02-28,yahoo,TT,verify,Ingersoll-Rand plc→Trane Technologies 2020-03-02
IR,2020-03-02,,yahoo,IR,verify,Ingersoll Rand Inc. (formerly Gardner Denver)
GOOG,,2014-04-02,yahoo,GOOGL,verify,pre-2014 GOOG was class A (today GOOGL); only if the file lists GOOG pre-2014
FI,,,yahoo,FISV,verify,Fiserv FISV→FI (2023-06)→FISV (2025-11)
BK,,,yahoo,BNY,verify,BNY Mellon ticker change (membership file 2026-05-21)
MMC,,,yahoo,MRSH,verify,Marsh McLennan ticker change (membership file 2026-01-14)
BHGE,,,yahoo,BKR,verify,Baker Hughes lineage BHI→BHGE→BKR; check the 2017 merger in the |ret| review
ADS,,,yahoo,BFH,verify,Alliance Data→Bread Financial 2022
CBS,,,yahoo,PSKY,verify,CBS→ViacomCBS→Paramount→Paramount Skydance; Yahoo PSKY history starts 2008
VIAC,,,yahoo,PSKY,verify,same lineage as CBS
SATS,,,yahoo,ECHO,verify,EchoStar ticker change (membership file 2026-06-24)
SIVB,,,tiingo,SIVBQ,high,SVB Financial; failed 2023-03 (Tiingo SIVBQ covers 1990→2026)
FRC,,,tiingo,FRCB,high,First Republic; failed 2023-05 (Tiingo FRCB covers 2010-12→2026)
```
`data/meta/dual_class_drop.csv` seed: `asset_id` = `yahoo:GOOG`, `yahoo:FOX`, `yahoo:NWS`, `tiingo:DISCK`, `yahoo:UA`.

## Appendix B: probe findings (2026-09-25)

**yfinance**
- yfinance 1.7.0 `download(auto_adjust=False)` returns Open, High, Low, Close, Adj Close and Volume with tz-naive dates.
- `.info["sector"]` works for listed tickers and returns None for delisted ones.

**Yahoo coverage of the 832 member tickers with spells from 2011 on**
- 638 return any data; 618 pass coverage validation.
- A further 148 are valid according to Tiingo's list.
- 67 are rename candidates, and 17 were unresolved before manual aliasing.

**Spells overlapping 2019 or later**
- 590 on Yahoo, 62 Tiingo-only, 34 renames, 7 unresolved before aliasing.
- Tiingo-only names in the OOS window include ATVI, CELG, TWTR, XLNX, PXD, CTXS, AGN, TIF, HES, DFS, JNPR, ANSS, WBA, K, MRO, CTLT, WRK, KSU, ALXN, MXIM, CERN, NLSN, ETFC, RHT, APC, RTN, STI, SBNY, DISH, VAR and FLIR.
- These are exactly the names a yfinance-only pipeline silently drops.

**Rename targets on Yahoo.** META, ELV, COR, BALL, LUMN, WBD, DD, FBIN, CPAY, LHX, J, BBWI, GEN, DOC, RVTY, EG, GL, RTX, WTW, GAP, DINO, FISV, BNY, MRSH, TT, BKR, BFH, CPRI, PSKY, ECHO and TFC all have full history (back to 2008, or to IPO). CTRA and DAY return nothing on Yahoo, so they come from Tiingo.

**Ticker reuse seen.** FB (now an ETF), APC, MON, CA, LB, LIFE, MMI, NE, Q, SUN, TE, SHLD, SNDK, CEG, DELL, IR, ARNC.

**Tiingo**
- The supported list is public: 108,834 rows with start and end dates.
- The free tier allows 50 requests/hour, 1,000/day and 500 symbols/month.
- The key is sent via the `Authorization: Token` header.

**Stooq.** The CSV endpoint now sits behind a JavaScript proof-of-work bot check. It is not usable, and we should not try to bypass it.

**Membership file.** 2,720 snapshots from 1996-01-02 to 2026-08-18. 848 unique tickers since 2010, 671 since 2019. Snapshot semantics were verified with FB → META.

**PyPI** (versions as of 2026-09-25, all with cp314 wheels): torch 2.14.0 (CUDA 13.0 build), pandas 3.0.6, numpy 2.5.3, statsmodels 0.15.0, scipy 1.18.1, pyarrow 25.0.1, matplotlib 3.11.2, yfinance 1.7.0, pytest 9.1.1.

## Appendix C: sources
- Paper: https://arxiv.org/abs/2510.11616 (full text: https://arxiv.org/html/2510.11616v1)
- Guijarro-Ordonez, Pelger, Zanotti, "Deep Learning Statistical Arbitrage", SSRN 3862004 (source of the cost model and the PCA benchmark)
- Avellaneda & Lee (2010), "Statistical arbitrage in the US equities market", Quantitative Finance 10(7)
- Fu et al. (2023), "Simple Hardware-Efficient Long Convolutions for Sequence Modeling", arXiv:2302.06646
- Gatev, Goetzmann & Rouwenhorst (2006), "Pairs Trading", Review of Financial Studies 19(3)
- Lo (2002), "The Statistics of Sharpe Ratios"; Politis & Romano (1994), stationary bootstrap
- Point-in-time S&P 500 membership: https://github.com/fja05680/sp500
- Tiingo: https://www.tiingo.com/documentation/end-of-day · https://www.tiingo.com/about/pricing · https://www.tiingo.com/documentation/general/connecting
