"""Astrometric-acceleration technosignature analysis --- a new physical axis.

Every other channel in this project is photometric (dips, fades, glints) or
spectroscopic (laser / absorption lines).  This one is *astrometric*: Gaia DR3
publishes, in ``gaiadr3.nss_acceleration_astro``, stars whose sky path is
*accelerating* --- curved by the gravitational pull of an unseen companion.  Most
such companions are ordinary stars, but the *dark* ones are how Gaia found the
nearest dormant black holes (BH1/BH2/BH3).  Reframed as a technosignature search,
the question is: which stars are being pushed by something we cannot see, and is
that something impossible to explain naturally (a companion mass with no luminous
counterpart, or --- the exotic tail --- a non-gravitational / engineered push)?

This module is the pure analysis: from the Gaia acceleration solution and the
parallax it computes the physical acceleration, a fiducial implied companion
mass, and a "darkness" flag (the companion should be luminous at that mass but the
star's photometry is single-star).  Acquisition and ranking live alongside; the
maths here is unit-tested offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Physical constants (SI) and unit conversions.
_G = 6.674e-11                 # m^3 kg^-1 s^-2
_MSUN = 1.989e30               # kg
_AU = 1.496e11                 # m
_PC = 3.086e16                 # m
_YR = 3.156e7                  # s
_KMS_PER_MASYR_KPC = 4.740857  # v[km/s] = 4.74 * mu[mas/yr] * d[kpc]


def physical_acceleration(accel_mas_yr2: np.ndarray,
                          parallax_mas: np.ndarray) -> np.ndarray:
    """Tangential physical acceleration (m/s^2) from the Gaia photocentre
    acceleration (mas/yr^2) and parallax (mas)."""
    accel = np.asarray(accel_mas_yr2, float)
    plx = np.asarray(parallax_mas, float)
    d_kpc = np.where(plx > 0, 1.0 / plx, np.nan)          # 1/plx[mas] = d[kpc]
    # v[km/s] = 4.74 * mu[mas/yr] * d[kpc]; same relation for accel per year.
    a_kms_per_yr = _KMS_PER_MASYR_KPC * accel * d_kpc
    return a_kms_per_yr * 1.0e3 / _YR                      # (km/s)/yr -> m/s^2


def implied_companion_mass(a_ms2: np.ndarray, separation_au: float = 3.0) -> np.ndarray:
    """Companion mass (Msun) that produces acceleration ``a`` at a fiducial
    separation.  Astrometric acceleration solutions are most sensitive to
    companions at a few AU (orbital periods comparable to the ~3-yr Gaia baseline),
    so a fiducial ``separation_au`` gives a physically-scaled ranking mass:
    ``M = a r^2 / G``."""
    a = np.asarray(a_ms2, float)
    r = separation_au * _AU
    return a * r * r / _G / _MSUN


def expected_absolute_g_for_mass(mass_msun: np.ndarray) -> np.ndarray:
    """Very rough main-sequence M_G for a companion of a given mass, used only to
    decide whether a companion *should* have been seen.  A companion more massive
    than the primary but invisible is the interesting ('dark') case."""
    m = np.asarray(mass_msun, float)
    # Crude MS: M_G ~ 4.8 - 8*log10(M) (bright for massive, faint for low-mass).
    with np.errstate(divide="ignore", invalid="ignore"):
        return 4.8 - 8.0 * np.log10(np.where(m > 0, m, np.nan))


def analyze_accelerations(df: pd.DataFrame, separation_au: float = 3.0) -> pd.DataFrame:
    """Add physical-acceleration, implied-mass and darkness columns to a Gaia
    acceleration table.  Expects columns ``accel_ra``, ``accel_dec`` (mas/yr^2),
    their errors, ``parallax``, ``parallax_over_error``, ``phot_g_mean_mag``,
    ``bp_rp``; extra columns are passed through.
    """
    out = df.copy()
    ar = pd.to_numeric(out.get("accel_ra"), errors="coerce")
    ad = pd.to_numeric(out.get("accel_dec"), errors="coerce")
    are = pd.to_numeric(out.get("accel_ra_error"), errors="coerce")
    ade = pd.to_numeric(out.get("accel_dec_error"), errors="coerce")
    plx = pd.to_numeric(out.get("parallax"), errors="coerce")

    total = np.hypot(ar, ad)
    err = np.hypot(are.fillna(np.nan), ade.fillna(np.nan))
    out["accel_total_mas_yr2"] = total
    out["accel_significance"] = total / err.replace(0, np.nan)
    out["dist_pc"] = np.where(plx > 0, 1000.0 / plx, np.nan)
    a_ms2 = physical_acceleration(total, plx)
    out["accel_m_s2"] = a_ms2
    mass = implied_companion_mass(a_ms2, separation_au=separation_au)
    out["implied_companion_msun"] = mass

    # Primary absolute G (is the star itself main-sequence & nearby?).
    g = pd.to_numeric(out.get("phot_g_mean_mag"), errors="coerce")
    out["abs_g"] = g + 5.0 * np.log10(np.where(plx > 0, plx / 100.0, np.nan))
    # A companion of the implied mass would, on the main sequence, be this bright;
    # if that is *brighter* than the primary yet unseen, the companion is DARK.
    comp_absg = expected_absolute_g_for_mass(mass)
    out["companion_expected_abs_g"] = comp_absg
    out["dark_companion"] = (np.isfinite(comp_absg) & np.isfinite(out["abs_g"])
                             & (comp_absg < out["abs_g"] + 0.5))
    return out


def rank_candidates(df: pd.DataFrame, sig_min: float = 20.0,
                    max_dist_pc: float = 500.0) -> pd.DataFrame:
    """Rank the unexplained-acceleration shortlist by the *strength* of the unseen
    pull.  A companion mass cannot be fixed without the orbital period, so we do
    not hard-filter on the (fiducial) mass; instead we rank by the physically
    meaningful trio that flagged Gaia BH1: a high-significance acceleration, a
    large physical acceleration, and a nearby (characterisable) host.  The
    ``dark_companion`` flag and ``implied_companion_msun`` are carried as
    illustrative context for the follow-up RV that would settle the mass."""
    d = analyze_accelerations(df) if "accel_significance" not in df.columns else df
    sel = (d["accel_significance"] >= sig_min) & (d["dist_pc"] <= max_dist_pc)
    out = d[sel].copy()
    if not len(out):
        out["rank_score"] = []
        return out
    amax = float(out["accel_m_s2"].abs().max()) or 1.0
    out["rank_score"] = (out["accel_significance"].clip(upper=200) / 200.0
                         * np.clip(out["accel_m_s2"].abs() / amax, 0, 1)
                         * np.clip(1.0 - out["dist_pc"] / max_dist_pc, 0, 1))
    return out.sort_values("rank_score", ascending=False)


__all__ = ["physical_acceleration", "implied_companion_mass", "analyze_accelerations",
           "rank_candidates", "expected_absolute_g_for_mass"]
