"""Repository locations. Data and result directories come from the config's ``paths`` section."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass(frozen=True)
class Paths:
    raw: Path
    processed: Path
    meta: Path
    runs: Path
    reports: Path


def project_paths(cfg: dict, root: Path = REPO_ROOT) -> Paths:
    """Resolve the config's ``paths`` section against ``root`` (the repo root by default)."""
    return Paths(**{name: root / rel for name, rel in cfg["paths"].items()})
