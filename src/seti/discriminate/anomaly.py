"""Rank infrared-excess white dwarfs by how poorly natural dust explains them.

The anomaly score combines:
  * being OUTSIDE the empirical debris-disk locus,
  * a swarm-like fractional luminosity (tau approaching unity),
  * NOT matching a known debris-disk catalogue.

High score == prioritised technosignature candidate for follow-up.  We are
explicit (here and in the manuscript) that a high score does NOT imply a
detection -- warm dust outside the nominal locus remains the leading natural
hypothesis until broken by variability / full-SED follow-up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dust_locus import in_dust_locus


def score_anomalies(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Add ``anomaly_score`` (0-1) and ``is_candidate`` columns."""
    out = df.copy()
    inside = in_dust_locus(out, thresholds)
    tau = out.get("tau", pd.Series(np.nan, index=out.index))
    known = out.get("known_disk", pd.Series(False, index=out.index)).fillna(False).astype(bool)

    tau_swarm = thresholds["discriminate"]["anomaly"]["tau_swarm_flag"]

    # Component scores.
    outside_locus = (~inside).astype(float)
    swarm_like = np.clip(tau.fillna(0.0) / max(tau_swarm, 1e-9), 0.0, 1.0)
    not_known = (~known).astype(float)

    score = (0.5 * outside_locus + 0.3 * swarm_like + 0.2 * not_known)
    # A known debris disk can never be a candidate.
    score = np.where(known, 0.0, score)

    out["anomaly_score"] = score
    out["outside_dust_locus"] = ~inside
    out["swarm_like"] = tau.fillna(0.0) >= tau_swarm
    out["is_candidate"] = (~known) & (~inside)
    return out
