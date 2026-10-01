# Decisions log

Choices that PLAN.md doesn't spell out, recorded as they are made (PLAN.md §0).

| Date | Phase | Decision | Reason |
|---|---|---|---|
| 2026-09-25 | 0 | The virtualenv lives at `~/.venvs/attention-based-arbitrage` (ext4), not `.venv` in the repo. The Makefile's `VENV` variable points there and can be overridden. | `/mnt/shared` is exFAT: no symlinks, and `python -m venv` always creates a `lib64 -> lib` symlink. Git adapts on its own (`core.symlinks=false`, `core.filemode=false`). |
| 2026-09-25 | 0 | Added a `paths:` section to `configs/base.yaml`, resolved against the repo root by `statarb.paths.project_paths`. | PLAN.md §5 says all paths come from the config, but the §17 reference had no paths block. |
| 2026-09-25 | 0 | Exact installed versions are recorded in `requirements-lock.txt` via `make lock`. | Pins the transitive dependencies (pyyaml, requests, CUDA libraries) that pyproject.toml leaves unpinned. |
