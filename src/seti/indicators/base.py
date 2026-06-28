"""Indicator protocol and result container.

An indicator scores each object on one independent anomaly axis.  It returns a
score in [0, 1] (higher = more anomalous), a boolean flag (score above the axis
threshold AND the measurement is available), and an ``available`` mask marking
objects for which the axis could be evaluated at all (so missing data is not
counted as either anomalous or normal).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass
class IndicatorResult:
    name: str
    score: pd.Series        # [0, 1] anomaly score, NaN where unavailable
    flag: pd.Series         # bool: flagged anomalous on this axis
    available: pd.Series    # bool: axis evaluable for this object
    detail: dict | None = None

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_available": int(self.available.sum()),
            "n_flagged": int(self.flag.sum()),
        }


class Indicator(Protocol):
    name: str

    def evaluate(self, df: pd.DataFrame, thresholds: dict) -> IndicatorResult: ...


def sigmoid_score(x: np.ndarray, x0: float, scale: float) -> np.ndarray:
    """Map a statistic to [0, 1], crossing 0.5 at ``x0`` over width ``scale``."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(x, dtype=float) - x0) / scale))
