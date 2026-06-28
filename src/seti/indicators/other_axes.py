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


def periodicity(df: pd.DataFrame, thresholds: dict) -> IndicatorResult:
    """Optical periodicity axis: a statistically significant Lomb--Scargle period.

    A white dwarf is intrinsically stable on day-to-month timescales, so a
    significant period in its ZTF light curve --- as from a transiting or occulting
    structure on a closed orbit --- is strongly anomalous, and far more diagnostic
    than raw scatter. Flags objects whose Lomb--Scargle false-alarm probability is
    below a threshold and whose amplitude exceeds a floor (so trivially-significant
    micro-amplitude peaks are not flagged).
    """
    cfg = thresholds.get("indicators", {}).get("periodicity", {})
    fap_max = cfg.get("ls_fap_max", 1e-3)
    amp_min = cfg.get("ls_amp_mag_min", 0.02)

    fap = df.get("ztf_ls_fap", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    amp = df.get("ztf_ls_amp_mag", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    available = pd.Series(np.isfinite(fap), index=df.index)
    with np.errstate(invalid="ignore", divide="ignore"):
        # Score on -log10(FAP), crossing 0.5 at the threshold FAP.
        neglog = -np.log10(np.where(np.isfinite(fap) & (fap > 0), fap, np.nan))
    score = pd.Series(np.where(np.isfinite(neglog),
                               sigmoid_score(neglog, -np.log10(fap_max), 1.0), np.nan),
                      index=df.index)
    flag = pd.Series(np.isfinite(fap) & (fap <= fap_max)
                     & np.isfinite(amp) & (amp >= amp_min),
                     index=df.index).fillna(False)
    return IndicatorResult("periodicity", score, flag, available,
                           detail={"ztf_ls_period_d": df.get(
                               "ztf_ls_period_d",
                               pd.Series(np.nan, index=df.index)).to_numpy()})


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


__all__ = ["ir_excess", "optical_variability", "ir_variability", "periodicity",
           "kinematic"]
