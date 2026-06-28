"""Gaia<->WISE co-movement test -- the decisive, novel contamination cut.

White dwarfs are nearby and fast-moving (often >50 mas/yr).  If the infrared
source really is the white dwarf (waste heat, a dust disk, or the photosphere),
then propagating the precise Gaia position forward to the WISE mean epoch using
the Gaia proper motion must land on the WISE position, and -- where CatWISE2020
reports its own proper motion -- the two motions must agree.  A *static* or
differently-moving IR source at the white dwarf's catalogued position is a
chance-aligned background object injecting false excess.  This test is largely
absent from both Project Hephaistos and the debris-disk surveys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAS_PER_DEG = 3.6e6


def propagated_offset_arcsec(df: pd.DataFrame, gaia_epoch: float, wise_epoch: float) -> pd.Series:
    """Angular offset (arcsec) between the epoch-propagated Gaia position and WISE.

    ``pmra`` is the Gaia mu_alpha* (already including cos(dec)).
    """
    dt = wise_epoch - gaia_epoch  # years
    cosd = np.cos(np.radians(df["dec"].to_numpy()))

    # Gaia position propagated to the WISE epoch, in mas relative to Gaia epoch.
    gaia_dra_mas = df["pmra"].to_numpy() * dt          # already *cos(dec)
    gaia_ddec_mas = df["pmdec"].to_numpy() * dt

    # Observed WISE - Gaia offset, in mas (great-circle small-angle).
    obs_dra_mas = (df["ra_wise"].to_numpy() - df["ra"].to_numpy()) * MAS_PER_DEG * cosd
    obs_ddec_mas = (df["dec_wise"].to_numpy() - df["dec"].to_numpy()) * MAS_PER_DEG

    resid_ra = obs_dra_mas - gaia_dra_mas
    resid_dec = obs_ddec_mas - gaia_ddec_mas
    offset_mas = np.hypot(resid_ra, resid_dec)
    return pd.Series(offset_mas / 1000.0, index=df.index)


def pm_consistency_sigma(df: pd.DataFrame) -> pd.Series:
    """Chi-like consistency between Gaia and CatWISE proper motions."""
    if not {"pmra_wise", "pmdec_wise"} <= set(df.columns):
        return pd.Series(np.nan, index=df.index)
    e_pmra_w = df.get("e_pmra_wise", pd.Series(50.0, index=df.index)).fillna(50.0)
    e_pmdec_w = df.get("e_pmdec_wise", pd.Series(50.0, index=df.index)).fillna(50.0)
    e_pmra_g = df.get("pmra_error", pd.Series(1.0, index=df.index)).fillna(1.0)
    e_pmdec_g = df.get("pmdec_error", pd.Series(1.0, index=df.index)).fillna(1.0)

    d_ra = (df["pmra"] - df["pmra_wise"]) / np.hypot(e_pmra_w, e_pmra_g)
    d_dec = (df["pmdec"] - df["pmdec_wise"]) / np.hypot(e_pmdec_w, e_pmdec_g)
    return np.hypot(d_ra, d_dec)


def comovement_pass(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    cm = thresholds["contamination"]["comovement"]
    # Epochs are carried on the DataFrame attrs (set by the pipeline) or default.
    gaia_epoch = df.attrs.get("gaia_ref_epoch", 2016.0)
    wise_epoch = df.attrs.get("wise_mean_epoch", 2010.5)

    if not {"ra_wise", "dec_wise"} <= set(df.columns):
        # Without a WISE position we cannot test co-movement; do not reject.
        return pd.Series(True, index=df.index)

    offset = propagated_offset_arcsec(df, gaia_epoch, wise_epoch)
    offset_ok = offset <= cm["max_position_offset_arcsec"]

    pm_sig = pm_consistency_sigma(df)
    # Only apply the PM test where CatWISE proper motions exist.
    pm_ok = pd.Series(True, index=df.index)
    have_pm = pm_sig.notna()
    pm_ok[have_pm] = pm_sig[have_pm] <= cm["pm_consistency_sigma_max"]

    return (offset_ok & pm_ok).fillna(False)


__all__ = ["comovement_pass", "propagated_offset_arcsec", "pm_consistency_sigma"]
