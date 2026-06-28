"""Infrared-excess statistics and warm-dust (T_dust, tau) characterisation.

The excess significance follows the convention used by the white-dwarf
debris-disk literature (Dennihy et al. 2020; Madurga Favieres et al. 2024):

    chi_B = (F_obs,B - F_pred,B) / sigma_B

where sigma_B combines the observed photometric error with a systematic floor
that absorbs model + zero-point uncertainty.  We additionally compute a W1-W2
colour-excess significance so that single-band artefacts do not masquerade as
excess.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..photometry import (
    band_freq_hz,
    mag_err_to_flux_err_jy,
    mag_to_flux_jy,
    planck_bnu,
)


def _sys_floor_flux_jy(pred_jy: np.ndarray, sys_floor_mag: float) -> np.ndarray:
    """Systematic flux floor: a fixed fractional error on the predicted flux."""
    frac = 0.4 * np.log(10.0) * sys_floor_mag
    return frac * np.abs(pred_jy)


def compute_excess(df: pd.DataFrame, thresholds: dict, bands=("W1", "W2")) -> pd.DataFrame:
    """Add observed flux, excess flux, and per-band excess significance columns."""
    out = df.copy()
    sys_floor = thresholds["excess"]["sys_floor_mag"]

    for b in bands:
        mcol, ecol, predcol = f"{b}mag", f"e_{b}mag", f"{b}_pred_jy"
        obs_jy = mag_to_flux_jy(out[mcol].to_numpy(), b)
        merr = np.where(np.isfinite(out.get(ecol, np.nan)), out.get(ecol, 0.1), 0.1)
        obs_err = mag_err_to_flux_err_jy(out[mcol].to_numpy(), merr, b)
        pred_jy = out[predcol].to_numpy()
        sigma = np.sqrt(obs_err**2 + _sys_floor_flux_jy(pred_jy, sys_floor) ** 2)

        excess_jy = obs_jy - pred_jy
        out[f"{b}_obs_jy"] = obs_jy
        out[f"{b}_excess_jy"] = excess_jy
        out[f"chi_{b}"] = excess_jy / sigma

    # Colour-excess significance: observed (W1-W2) redward of photospheric colour.
    if {"W1", "W2"} <= set(bands):
        obs_color = out["W1mag"].to_numpy() - out["W2mag"].to_numpy()
        pred_color = out["W1_pred_mag"].to_numpy() - out["W2_pred_mag"].to_numpy()
        ce1 = np.where(np.isfinite(out.get("e_W1mag", np.nan)), out.get("e_W1mag", 0.1), 0.1)
        ce2 = np.where(np.isfinite(out.get("e_W2mag", np.nan)), out.get("e_W2mag", 0.1), 0.1)
        col_err = np.sqrt(ce1**2 + ce2**2 + (np.sqrt(2) * sys_floor) ** 2)
        # Excess colour is *redder* (W1-W2 larger) than photosphere.
        out["color_excess"] = obs_color - pred_color
        out["chi_color"] = out["color_excess"] / col_err
    return out


def select_excess(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    """Boolean mask of sources passing the infrared-excess selection.

    Following the white-dwarf debris-disk literature (Dennihy et al. 2020), we
    require a significant *red* W1-W2 colour excess plus a significant flux
    excess in at least one WISE band.  The "at least one band" (OR) logic is
    deliberate: warm dust shows in both W1 and W2, whereas a cool / swarm-like
    excess is W2-dominated and would be missed by an AND requirement -- biasing
    against precisely the cool technosignature regime we want to probe.
    """
    ex = thresholds["excess"]
    band_excess = (
        ((df["chi_W1"] >= ex["chi_w1_min"]) | (df["chi_W2"] >= ex["chi_w2_min"]))
        & (df["W2_excess_jy"] > 0)
    )
    mask = band_excess
    if "chi_color" in df:
        mask = mask & (df["chi_color"] >= ex["color_excess_sigma_min"])
    return mask.fillna(False)


def _band_flux_for_dust(temp_k: float, band: str, omega: float) -> float:
    """Warm-dust blackbody flux (Jy) in a band for solid angle ``omega``."""
    return float(omega * np.pi * planck_bnu(temp_k, band_freq_hz(band)) * 1e26)


def fit_dust(row: pd.Series, bands=("W1", "W2")) -> tuple[float, float]:
    """Estimate (T_dust, fractional luminosity tau) from the W1/W2 excess.

    With only two excess bands we solve the single-temperature blackbody that
    reproduces the W1/W2 excess colour, then scale to the excess flux.  Returns
    ``(t_dust_k, tau)``; ``tau`` is L_excess / L_photosphere approximated from
    the integrated blackbody luminosity ratio.
    """
    f1 = row.get("W1_excess_jy", np.nan)
    f2 = row.get("W2_excess_jy", np.nan)
    if not (np.isfinite(f1) and np.isfinite(f2)) or f1 <= 0 or f2 <= 0:
        return np.nan, np.nan

    # Colour (W1-W2) of the excess pins the dust temperature: invert the ratio
    # of Planck functions across a grid (robust, no derivatives).
    nu1, nu2 = band_freq_hz("W1"), band_freq_hz("W2")
    grid = np.linspace(80.0, 2500.0, 600)
    model_ratio = (
        planck_bnu(grid, nu1) / planck_bnu(grid, nu2)
    )
    target = f1 / f2
    t_dust = float(grid[np.argmin(np.abs(model_ratio - target))])

    # Scale solid angle to match the W1 excess flux, then a crude tau via the
    # ratio of the dust blackbody bolometric output to the WD photosphere.
    omega_dust = f1 / _band_flux_for_dust(t_dust, "W1", omega=1.0)
    scale = row.get("sed_scale", np.nan)
    teff = row.get("teff", np.nan)
    if not (np.isfinite(scale) and np.isfinite(teff)) or scale <= 0:
        return t_dust, np.nan
    # Bolometric ~ Omega * sigma T^4 (Stefan-Boltzmann); ratio cancels constants.
    tau = float((omega_dust * t_dust**4) / (scale * teff**4))
    return t_dust, tau


def characterise_dust(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``t_dust_k`` and ``tau`` columns for the excess sources."""
    out = df.copy()
    t_dust = np.full(len(out), np.nan)
    tau = np.full(len(out), np.nan)
    for i, (_, row) in enumerate(out.iterrows()):
        t_dust[i], tau[i] = fit_dust(row)
    out["t_dust_k"] = t_dust
    out["tau"] = tau
    return out


__all__ = ["compute_excess", "select_excess", "fit_dust", "characterise_dust"]
