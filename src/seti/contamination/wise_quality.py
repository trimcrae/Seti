"""WISE photometric-quality cuts: the excess must be real IR, not an artefact."""

from __future__ import annotations

import numpy as np
import pandas as pd


def wise_quality_pass(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    q = thresholds["contamination"]["wise_quality"]

    cc = df.get("cc_flags", pd.Series("0000", index=df.index)).astype(str)
    ph = df.get("ph_qual", pd.Series("AA", index=df.index)).astype(str)
    ext = df.get("ext_flg", pd.Series(0, index=df.index)).fillna(0)
    w1 = df.get("W1mag", pd.Series(np.nan, index=df.index))
    w2snr = df.get("w2snr", pd.Series(np.nan, index=df.index))

    # cc_flags "0000" => clean in all four bands (we require W1/W2 at least).
    cc_ok = cc.str[:2].isin(["00"])
    # ph_qual: first two chars (W1,W2) must be in the allowed set.
    allowed = set(q["allowed_ph_qual"])
    ph_ok = ph.str[0].isin(allowed) & ph.str[1].isin(allowed)
    ext_ok = ext <= q["ext_flg_max"]
    sat_ok = w1 >= q["w1_saturation_mag_min"]            # not saturated
    snr_ok = w2snr.fillna(99.0) >= q["w2_snr_min"]

    return (cc_ok & ph_ok & ext_ok & sat_ok & snr_ok).fillna(False)
