# Decisions log

Choices that PLAN.md doesn't spell out, recorded as they are made (PLAN.md §0).

| Date | Phase | Decision | Reason |
|---|---|---|---|
| 2026-09-25 | 0 | The virtualenv lives at `~/.venvs/attention-based-arbitrage` (ext4), not `.venv` in the repo. The Makefile's `VENV` variable points there and can be overridden. | `/mnt/shared` is exFAT: no symlinks, and `python -m venv` always creates a `lib64 -> lib` symlink. Git adapts on its own (`core.symlinks=false`, `core.filemode=false`). |
| 2026-09-25 | 0 | Added a `paths:` section to `configs/base.yaml`, resolved against the repo root by `statarb.paths.project_paths`. | PLAN.md §5 says all paths come from the config, but the §17 reference had no paths block. |
| 2026-09-25 | 0 | Exact installed versions are recorded in `requirements-lock.txt` via `make lock`. | Pins the transitive dependencies (pyyaml, requests, CUDA libraries) that pyproject.toml leaves unpinned. |
| 2026-10-01 | 1 | Old listings of reused tickers (STI, APC, WRK, MON, DOW, DELL, EMC, SNDK, S, …) are fetched from Tiingo by permaTicker, recorded in the alias table. | Tiingo's by-ticker endpoint returns the current owner of a reused ticker; coverage validation rejected those series, and the search API (`includeDelisted=true`) exposes each old listing's permaTicker. |
| 2026-10-01 | 1 | Names whose free history is incomplete are explicit `none` aliases rather than partial series: Tiingo history starting 2016 (BRCM, CVC, HAR, HOT, LLTC, PCP) or no old listing at all (CSRA, INFO, PLL, SII, SUN, ACS). | A spell is only used when one series covers ≥95% of it (§6.4); stitching partial coverage would need gap rules the plan doesn't define. |
| 2026-10-01 | 1 | Manual sectors (`sectors_manual.csv`) may also override Yahoo assets whose `info["sector"]` comes back blank (FISV, GENZ). | Yahoo returned no sector for two listed names. |
