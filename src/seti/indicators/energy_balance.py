"""UV-deficit and energy-balance indicators -- the centrepiece additional evidence.

Thermodynamics is the strongest argument for a Dyson-type structure: flux
intercepted at short wavelengths is re-radiated as infrared waste heat, so the
bolometric output is conserved but redistributed.  A white dwarf is an almost
ideal testbed because its photospheric spectral energy distribution is
predictable from $T_{\rm eff}$ and the Gaia parallax, so both a UV/optical
*deficit* and an infrared *excess* are cleanly measurable.

Two axes are provided:

* ``uv_deficit`` -- the GALEX NUV (and FUV) flux falls significantly *below* the
  predicted photosphere, as if short-wavelength light were being absorbed.  This
  is largely orthogonal to infrared excess: warm dust and brown-dwarf companions
  add infrared flux but do not remove ultraviolet flux.

* ``energy_balance`` -- the luminosity removed from the UV/optical (the deficit)
  matches, to order unity, the luminosity re-emitted in the infrared (the excess).
  A coincidence of *absorption* and *matched re-emission* is the specific
  signature an energy-collecting structure would imprint, and is far harder for
  any single natural confounder to reproduce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..photometry import mag_to_flux_jy
from .base import IndicatorResult, sigmoid_score

UV_BANDS = ("NUV", "FUV")


def _predicted_uv_jy(df: pd.DataFrame, band: str) -> np.ndarray:
    """Predicted photospheric UV flux (Jy) from the fitted blackbody scale.

    Uses the same solid-angle ``sed_scale`` fit to the optical/NIR anchor, so the
    UV prediction is an extrapolation of the photosphere shortward of the optical;
    we therefore treat only a *deficit* (observed well below predicted) as
    anomalous, never a marginal excess.
    """
    from ..photometry import band_freq_hz, planck_bnu

    scale = df.get("sed_scale", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    teff = df.get("teff", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        return scale * np.pi * planck_bnu(np.where(np.isfinite(teff), teff, 1.0),
                                          band_freq_hz(band)) * 1e26


def uv_deficit(df: pd.DataFrame, thresholds: dict) -> IndicatorResult:
    cfg = thresholds.get("indicators", {}).get("uv_deficit", {})
    sigma_min = cfg.get("sigma_min", 3.0)
    sys_frac = cfg.get("sys_frac", 0.30)   # large: UV is a long photosphere extrapolation

    n = len(df)
    best_sig = np.full(n, np.nan)
    for band in UV_BANDS:
        mcol, ecol = f"{band}mag", f"e_{band}mag"
        if mcol not in df:
            continue
        mag = df[mcol].to_numpy(dtype=float)
        finite = np.isfinite(mag)
        obs = np.where(finite, mag_to_flux_jy(np.where(finite, mag, 0.0), band), np.nan)
        merr = df[ecol].to_numpy(dtype=float) if ecol in df else np.full(n, 0.2)
        oerr = np.where(np.isfinite(merr), 0.4 * np.log(10.0) * obs * merr, 0.2 * obs)
        pred = _predicted_uv_jy(df, band)
        sigma = np.sqrt(oerr**2 + (sys_frac * pred) ** 2)
        with np.errstate(invalid="ignore"):
            # Deficit significance: predicted minus observed, in sigma (positive
            # => UV suppressed below the photosphere).
            sig = (pred - obs) / sigma
        best_sig = np.where(np.isfinite(sig) & (np.isnan(best_sig) | (sig > best_sig)),
                            sig, best_sig)

    available = pd.Series(np.isfinite(best_sig), index=df.index)
    score = pd.Series(np.where(np.isfinite(best_sig),
                               sigmoid_score(best_sig, sigma_min, 1.5), np.nan),
                      index=df.index)
    flag = pd.Series((best_sig >= sigma_min), index=df.index).fillna(False)
    return IndicatorResult("uv_deficit", score, flag, available,
                           detail={"uv_deficit_sigma": best_sig})


def energy_balance(df: pd.DataFrame, thresholds: dict) -> IndicatorResult:
    """Flag objects whose UV/optical deficit luminosity matches the IR excess.

    Both are expressed as a fractional luminosity (covering fraction).  The IR
    side reuses the fitted ``tau``; the UV side converts the NUV deficit into an
    equivalent absorbed fraction.  A ratio near unity (within a configurable band)
    with both sides significant is the energy-balance signature.
    """
    cfg = thresholds.get("indicators", {}).get("energy_balance", {})
    ratio_lo = cfg.get("ratio_lo", 0.3)
    ratio_hi = cfg.get("ratio_hi", 3.0)
    min_tau = cfg.get("min_tau", 5e-3)

    n = len(df)
    tau_ir = df.get("tau", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)

    # UV absorbed fraction: deficit flux / predicted photospheric flux at NUV,
    # weighted by the (large) UV/optical share of a hot WD's bolometric output is
    # beyond two-band reach, so we use the in-band NUV fractional deficit as a
    # proxy and compare its order of magnitude to tau_ir.
    nuv_def_frac = np.full(n, np.nan)
    if "NUVmag" in df:
        obs = mag_to_flux_jy(df["NUVmag"].to_numpy(dtype=float), "NUV")
        pred = _predicted_uv_jy(df, "NUV")
        with np.errstate(invalid="ignore", divide="ignore"):
            nuv_def_frac = np.clip((pred - obs) / pred, 0.0, 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = nuv_def_frac / tau_ir
    available = pd.Series(np.isfinite(ratio), index=df.index)
    balanced = (np.isfinite(ratio) & (ratio >= ratio_lo) & (ratio <= ratio_hi)
                & (tau_ir >= min_tau) & (nuv_def_frac >= min_tau))
    # Score peaks when log-ratio is near zero (perfect balance).
    with np.errstate(invalid="ignore", divide="ignore"):
        logr = np.abs(np.log10(np.where(ratio > 0, ratio, np.nan)))
    score = pd.Series(np.where(np.isfinite(logr), np.clip(1.0 - logr, 0.0, 1.0), np.nan),
                      index=df.index)
    flag = pd.Series(balanced, index=df.index).fillna(False)
    return IndicatorResult("energy_balance", score, flag, available,
                           detail={"uv_ir_ratio": ratio, "nuv_deficit_frac": nuv_def_frac})


__all__ = ["uv_deficit", "energy_balance"]
