"""Empirical white-dwarf debris-disk locus in (T_dust, tau) space.

White-dwarf debris disks are warm (sublimation-limited, ~1000-1700 K) and faint
(fractional luminosity tau ~ 1e-4 to 1e-1).  We honestly acknowledge that W1/W2
photometry alone cannot uniquely separate a partial Dyson swarm from warm dust;
the strategy is therefore *subtraction and outlier flagging*:

  * sources already in a known debris-disk catalogue -> labelled known dust;
  * sources whose (T_dust, tau) sit OUTSIDE the empirical locus (too cool, or
    swarm-like tau approaching unity) -> prioritised anomalies for follow-up.

The locus bounds are config-driven and can be recalibrated from the control
catalogues at build time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def in_dust_locus(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    """True where (T_dust, tau) fall inside the empirical WD debris-disk locus."""
    loc = thresholds["discriminate"]["dust_locus"]
    t = df.get("t_dust_k", pd.Series(np.nan, index=df.index))
    tau = df.get("tau", pd.Series(np.nan, index=df.index))
    inside = (
        (t >= loc["t_dust_min_k"])
        & (t <= loc["t_dust_max_k"])
        & (tau >= loc["tau_min"])
        & (tau <= loc["tau_max"])
    )
    return inside.fillna(False)


def calibrate_locus_from_controls(control_df: pd.DataFrame, q: float = 0.99) -> dict:
    """Derive locus bounds as central quantiles of the control debris-disk sample.

    Returns a dict matching the ``dust_locus`` config block, for optionally
    overriding the static thresholds during a science run.
    """
    t = control_df["t_dust_k"].dropna()
    tau = control_df["tau"].dropna()
    lo = (1 - q) / 2
    hi = 1 - lo
    return {
        "t_dust_min_k": float(t.quantile(lo)),
        "t_dust_max_k": float(t.quantile(hi)),
        "tau_min": float(tau.quantile(lo)),
        "tau_max": float(tau.quantile(hi)),
    }
