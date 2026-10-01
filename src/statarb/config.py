"""Load ``configs/base.yaml`` and deep-merge optional override files (e.g. ``configs/frozen.yaml``)."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from statarb.paths import CONFIG_DIR


def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of ``base`` with ``override`` merged in: nested dicts merge, other values replace."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(*overrides: Path, config_dir: Path = CONFIG_DIR) -> dict:
    """Load ``base.yaml`` from ``config_dir``, then deep-merge each override file in order."""
    cfg = read_yaml(config_dir / "base.yaml")
    for path in overrides:
        cfg = deep_merge(cfg, read_yaml(path))
    return cfg
