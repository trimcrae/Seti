"""Combine independent anomaly axes into a multi-modal candidate ranking.

The headline statistic is ``n_axes`` -- the number of independent axes on which an
object is flagged anomalous.  A single flagged axis (e.g. infrared excess) is
unremarkable; objects flagged on two or more *independent* axes are the exciting
candidates, because the natural confounders of any one axis (dust, brown-dwarf
companions, blends) do not generally reproduce the others, and least of all the
energy-balance coincidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import IndicatorResult


def combine_indicators(
    df: pd.DataFrame,
    results: list[IndicatorResult],
    weights: dict[str, float] | None = None,
    min_axes: int = 2,
) -> pd.DataFrame:
    """Attach per-axis scores/flags, an aggregate score, ``n_axes`` and the
    multi-axis candidate flag to a copy of ``df``."""
    # The indicator results are indexed by the *original* ``df`` index, which may
    # be non-contiguous (e.g. a boolean-filtered subset).  Align every result to
    # ``df.index`` (identity for results computed on ``df``) and read values
    # positionally, then relabel the output to a clean RangeIndex.  Reindexing to a
    # freshly reset 0..N-1 index here would silently misalign scores against rows.
    out = df.copy().reset_index(drop=True)
    weights = weights or {}

    n_axes = pd.Series(0, index=out.index, dtype=int)
    weighted = pd.Series(0.0, index=out.index)
    wsum = 0.0
    axes_flagged: list[list[str]] = [[] for _ in range(len(out))]

    for r in results:
        score = pd.Series(r.score.reindex(df.index).to_numpy(), index=out.index)
        flag = pd.Series(r.flag.reindex(df.index).fillna(False).astype(bool).to_numpy(),
                         index=out.index)
        out[f"score_{r.name}"] = score.to_numpy()
        out[f"flag_{r.name}"] = flag.to_numpy()
        n_axes = n_axes + flag.astype(int)
        w = float(weights.get(r.name, 1.0))
        weighted = weighted + w * score.fillna(0.0)
        wsum += w
        for i in np.where(flag.to_numpy())[0]:
            axes_flagged[i].append(r.name)
        # Surface useful per-axis diagnostics (e.g. nuv_deficit_frac, vtan_km_s)
        # as columns so downstream figures/tables can use them without re-running.
        for key, arr in (r.detail or {}).items():
            if key not in out.columns:
                out[key] = np.asarray(arr)[: len(out)] if np.ndim(arr) else arr

    out["n_axes"] = n_axes.to_numpy()
    out["axes_flagged"] = [",".join(a) for a in axes_flagged]
    out["multimodal_score"] = (weighted / wsum).to_numpy() if wsum else 0.0
    # Exciting candidate: anomalous on at least ``min_axes`` independent axes.
    out["multimodal_candidate"] = out["n_axes"] >= min_axes
    return out


__all__ = ["combine_indicators"]
