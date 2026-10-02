# The virtualenv lives on the home partition: /mnt/shared is exFAT, which has no symlinks,
# and `python -m venv` always creates one. Override with `make VENV=/path/to/venv ...`.
VENV ?= $(HOME)/.venvs/attention-based-arbitrage
PY := $(VENV)/bin/python
STATARB := $(PY) -m statarb

.PHONY: setup lock data features tensors test test-all tune freeze oos checks ablations report all

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

lock:
	$(PY) -m pip freeze --exclude-editable > requirements-lock.txt

# Phase 1-2. The Tiingo step takes ~4 hours (rate limit) and is resumable.
data:
	$(STATARB) membership
	$(STATARB) yahoo
	$(STATARB) resolve
	$(STATARB) tiingo
	$(STATARB) resolve
	$(STATARB) sectors
	$(STATARB) panel
	$(STATARB) universe --kind pit
	$(STATARB) universe --kind survivor
	$(STATARB) coverage

features:
	$(STATARB) features

tensors:
	$(STATARB) tensors --universe pit --variant real
	$(STATARB) tensors --universe pit --variant noise --noise-seed 0
	$(STATARB) tensors --universe pit --variant noise --noise-seed 1
	$(STATARB) tensors --universe pit --variant noise --noise-seed 2
	$(STATARB) tensors --universe survivor --variant real

test:
	$(PY) -m pytest -m "not slow"

test-all:
	$(PY) -m pytest

tune:
	$(STATARB) tune --strategy attention
	$(STATARB) tune --strategy pca
	$(STATARB) tune --strategy pairs
	$(STATARB) tune --strategy reversal

# One-time step after tuning: writes configs/frozen.yaml; then commit and `git tag frozen-v1`.
freeze:
	$(STATARB) freeze

oos:
	$(STATARB) oos --strategy attention
	$(STATARB) oos --strategy pca
	$(STATARB) oos --strategy pairs
	$(STATARB) oos --strategy reversal

checks:
	$(STATARB) oos --strategy attention --universe survivor
	$(STATARB) checks

ablations:
	$(STATARB) oos --strategy attention --sequence longconv
	$(STATARB) oos --strategy attention --drop-group past_returns
	$(STATARB) oos --strategy attention --drop-group risk
	$(STATARB) oos --strategy attention --drop-group frictions
	$(STATARB) oos --strategy attention --K 5
	$(STATARB) oos --strategy attention --K 10
	$(STATARB) oos --strategy attention --K 20

report:
	$(STATARB) report

# Full rebuild from raw downloads. `freeze` is deliberately not included: OOS runs use the
# committed configs/frozen.yaml and refuse to run without it.
all: data features tensors test tune oos checks ablations report
