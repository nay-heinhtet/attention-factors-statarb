"""Shared helpers: logging, seeding, device selection, config hashing."""

from __future__ import annotations

import hashlib
import json
import logging
import random

import numpy as np


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (all devices) and make cuDNN deterministic."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto"):
    """Map the config's ``train.device`` value to a torch device."""
    import torch

    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference in ("cuda", "cpu"):
        return torch.device(preference)
    raise ValueError(f"unknown device preference {preference!r}; expected auto, cuda or cpu")


def config_hash(cfg: dict) -> str:
    """Short hash of a config, independent of key order; recorded with every OOS run."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
