"""Gaia astrometric-quality cuts: keep well-measured, single, nearby sources."""

from __future__ import annotations

import numpy as np
import pandas as pd


def astrometry_pass(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    a = thresholds["contamination"]["astrometry"]
    ruwe = df.get("ruwe", pd.Series(np.nan, index=df.index))
    plx_snr = df.get("parallax_over_error", pd.Series(np.nan, index=df.index))
    excess_noise = df.get("astrometric_excess_noise", pd.Series(np.nan, index=df.index))

    # Degrade gracefully: a quality column that is absent/NaN does not reject the
    # source (the cut applies only where the measurement exists). This lets the
    # funnel run on real tables that may lack, e.g., RUWE, while still vetting
    # wherever the column is present.
    ruwe_ok = ruwe.isna() | (ruwe <= a["ruwe_max"])
    plx_ok = plx_snr.isna() | (plx_snr >= a["parallax_over_error_min"])
    noise_ok = excess_noise.isna() | (excess_noise <= a["astrometric_excess_noise_max_mas"])
    return (ruwe_ok & plx_ok & noise_ok).fillna(True)
