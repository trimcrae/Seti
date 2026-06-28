"""Gaia astrometric-quality cuts: keep well-measured, single, nearby sources."""

from __future__ import annotations

import numpy as np
import pandas as pd


def astrometry_pass(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    a = thresholds["contamination"]["astrometry"]
    ruwe = df.get("ruwe", pd.Series(np.nan, index=df.index))
    plx_snr = df.get("parallax_over_error", pd.Series(np.nan, index=df.index))
    excess_noise = df.get("astrometric_excess_noise", pd.Series(0.0, index=df.index))

    mask = (
        (ruwe <= a["ruwe_max"])
        & (plx_snr >= a["parallax_over_error_min"])
        & (excess_noise.fillna(0.0) <= a["astrometric_excess_noise_max_mas"])
    )
    return mask.fillna(False)
