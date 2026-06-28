"""Configuration loading.

All tunable numbers live in ``config/*.yaml``; code never hard-codes a
threshold.  ``load_config`` returns a single merged, attribute-accessible
mapping so callers can write ``cfg.thresholds["excess"]["chi_w1_min"]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    """Locate the repository root (the directory containing ``config/``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config").is_dir():
            return parent
    # Fall back to two levels up from src/seti/.
    return here.parents[2]


class Config:
    """Lightweight holder for the three config files plus the repo root."""

    def __init__(self, root: Path, thresholds: dict, catalogs: dict, paths: dict):
        self.root = root
        self.thresholds = thresholds
        self.catalogs = catalogs
        self.paths = paths

    def path(self, key: str) -> Path:
        """Resolve a key from ``paths.yaml`` to an absolute path."""
        return (self.root / self.paths[key]).resolve()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config(root={self.root})"


def _read_yaml(p: Path) -> dict[str, Any]:
    with p.open() as fh:
        return yaml.safe_load(fh)


def load_config(root: Path | str | None = None) -> Config:
    """Load and merge ``config/{thresholds,catalogs,paths}.yaml``."""
    root = Path(root).resolve() if root is not None else _repo_root()
    cfg_dir = root / "config"
    return Config(
        root=root,
        thresholds=_read_yaml(cfg_dir / "thresholds.yaml"),
        catalogs=_read_yaml(cfg_dir / "catalogs.yaml"),
        paths=_read_yaml(cfg_dir / "paths.yaml"),
    )
