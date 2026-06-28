"""Additional independent anomaly axes: infrared excess, optical and mid-infrared
variability, and kinematics.

Each is deliberately independent of the others so that a multi-axis coincidence
is informative.  Variability axes use catalogue flags (Gaia) where available and
survey light-curve metrics (ZTF, NEOWISE) attached for the shortlist; objects
without a measurement are marked unavailable rather than normal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import IndicatorResult, sigmoid_score


def ir_excess(df: pd.DataFrame, thresholds: dict) -> IndicatorResult:
    """Infrared waste-heat axis: the pipeline's W1/W2 excess (re-expressed)."""
    chi2 = df.get("chi_W2", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    has = df.get("has_excess", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    available = pd.Series(np.isfinite(chi2), index=df.index)
    score = pd.Series(np.where(np.isfinite(chi2), sigmoid_score(chi2, 5.0, 2.0), np.nan),
                      index=df.index)
    return IndicatorResult("ir_excess", score, pd.Series(has.to_numpy(), index=df.index),
                           available)


def optical_variability(df: pd.DataFrame, thresholds: dict) -> IndicatorResult:
    """Optical variability axis (megastructure-transit / dimming signature).

    Uses the Gaia DR3 variability flag and amplitude where present, and a ZTF
    variability metric (fractional RMS or a structure score) attached for the
    shortlist. A white dwarf is intrinsically photometrically stable, so genuine
    optical variability is itself anomalous.
    """
    cfg = thresholds.get("indicators", {}).get("optical_variability", {})
    frac_rms_min = cfg.get("ztf_frac_rms_min", 0.03)

    n = len(df)
    stat = np.full(n, np.nan)
    # ZTF fractional RMS (shortlist), if attached.
    if "ztf_frac_rms" in df:
        stat = df["ztf_frac_rms"].to_numpy(dtype=float)
    # Gaia variability flag as a coarse fallback.
    gaia_var = df.get("phot_variable_flag", pd.Series("", index=df.index)).astype(str)
    gaia_flag = gaia_var.str.upper().str.startswith("VARIABLE")

    available = pd.Series(np.isfinite(stat) | gaia_flag.to_numpy(), index=df.index)
    with np.errstate(invalid="ignore"):
        s = np.where(np.isfinite(stat), sigmoid_score(stat, frac_rms_min, frac_rms_min),
                     np.where(gaia_flag.to_numpy(), 0.6, np.nan))
    score = pd.Series(s, index=df.index)
    flag = pd.Series((np.isfinite(stat) & (stat >= frac_rms_min)) | gaia_flag.to_numpy(),
                     index=df.index).fillna(False)
    return IndicatorResult("optical_variability", score, flag, available)


def ir_variability(df: pd.DataFrame, thresholds: dict) -> IndicatorResult:
    """Mid-infrared variability axis (changing waste heat) from NEOWISE epochs.

    A ``neowise_w1_frac_rms`` column (attached for the shortlist from the NEOWISE
    multi-epoch photometry) above a threshold flags a variable infrared source ---
    a structure under construction or an eclipsing/transiting configuration.
    """
    cfg = thresholds.get("indicators", {}).get("ir_variability", {})
    rms_min = cfg.get("neowise_frac_rms_min", 0.05)
    stat = df.get("neowise_w1_frac_rms", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    available = pd.Series(np.isfinite(stat), index=df.index)
    score = pd.Series(np.where(np.isfinite(stat), sigmoid_score(stat, rms_min, rms_min), np.nan),
                      index=df.index)
    flag = pd.Series((stat >= rms_min), index=df.index).fillna(False)
    return IndicatorResult("ir_variability", score, flag, available)


def kinematic(df: pd.DataFrame, thresholds: dict) -> IndicatorResult:
    """Kinematic-anomaly axis: unusually high tangential velocity.

    A weak, low-weight axis (an unusual space motion is an oddity, not a
    technosignature per se), included for completeness. Halo-velocity white dwarfs
    are themselves astrophysically interesting cross-checks.
    """
    cfg = thresholds.get("indicators", {}).get("kinematic", {})
    vtan_hi = cfg.get("vtan_km_s_hi", 200.0)
    pmra = df.get("pmra", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    pmdec = df.get("pmdec", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    plx = df.get("parallax", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    mu = np.hypot(pmra, pmdec)
    with np.errstate(invalid="ignore", divide="ignore"):
        vtan = 4.74 * mu / np.where(plx > 0, plx, np.nan)   # km/s
    available = pd.Series(np.isfinite(vtan), index=df.index)
    score = pd.Series(np.where(np.isfinite(vtan), sigmoid_score(vtan, vtan_hi, 60.0), np.nan),
                      index=df.index)
    flag = pd.Series((vtan >= vtan_hi), index=df.index).fillna(False)
    return IndicatorResult("kinematic", score, flag, available, detail={"vtan_km_s": vtan})


__all__ = ["ir_excess", "optical_variability", "ir_variability", "kinematic"]
