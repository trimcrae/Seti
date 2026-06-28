"""Run the full multi-modal indicator suite and combine the axes."""

from __future__ import annotations

import pandas as pd

from .base import IndicatorResult
from .combine import combine_indicators
from .energy_balance import energy_balance, uv_deficit
from .other_axes import ir_excess, ir_variability, kinematic, optical_variability

# Independent anomaly axes, with combination weights (energy balance is the
# strongest single piece of evidence, kinematics the weakest).
INDICATORS = [
    ("ir_excess", ir_excess, 1.0),
    ("uv_deficit", uv_deficit, 1.5),
    ("energy_balance", energy_balance, 3.0),
    ("optical_variability", optical_variability, 2.0),
    ("ir_variability", ir_variability, 2.0),
    ("kinematic", kinematic, 0.3),
]


def run_multimodal(df: pd.DataFrame, thresholds: dict, min_axes: int = 2) -> pd.DataFrame:
    """Evaluate every indicator and return ``df`` with per-axis and combined
    multi-modal anomaly columns."""
    results: list[IndicatorResult] = []
    for _, fn, _w in INDICATORS:
        try:
            results.append(fn(df, thresholds))
        except Exception as exc:  # an unavailable axis must not abort the suite
            print(f"[indicators] {fn.__name__} skipped: {exc!r}")
    weights = {name: w for name, _fn, w in INDICATORS}
    return combine_indicators(df, results, weights=weights, min_axes=min_axes)


def indicator_summary(combined: pd.DataFrame) -> dict:
    out = {"n_objects": int(len(combined))}
    for name, _fn, _w in INDICATORS:
        col = f"flag_{name}"
        if col in combined:
            out[name] = int(combined[col].sum())
    for k in (2, 3):
        out[f"n_ge_{k}_axes"] = int((combined.get("n_axes", 0) >= k).sum())
    out["n_multimodal"] = int(combined.get("multimodal_candidate", False).sum())
    return out


__all__ = ["run_multimodal", "indicator_summary", "INDICATORS"]
