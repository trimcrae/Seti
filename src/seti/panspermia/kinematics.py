"""Full 6D phase space (position + space velocity) from Gaia observables.

The clustering channel only needed *tangential* velocity (proper motion).  A
close-encounter search needs the **full space velocity**, because whether two
stars ever passed close to one another is set by their 3D relative position and
their 3D relative velocity -- and the line-of-sight (radial) component is half of
that.  So here we require Gaia's ``radial_velocity`` and build heliocentric
Galactic Cartesian position ``(X, Y, Z)`` in pc and velocity ``(U, V, W)`` in
km/s.

Two deliberate choices:

* Velocities are **heliocentric**, not corrected to the Local Standard of Rest.
  For an *encounter* we only ever use the velocity of one star *relative to
  another*; the solar-motion offset is common to both and cancels in the
  difference, so the LSR correction (and its uncertainty) never enters.  This is
  why no solar-motion constants appear in this module.
* The ICRS->Galactic rotation is the same matrix the clustering channel uses, so
  the two channels share one definition of the Galactic frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 1 AU/yr in km/s (IAU 2012). v_tan[km/s] = _K * mu[mas/yr] * d[pc] / 1000.
_K_AUYR_KMS = 4.740470446

# ICRS -> Galactic rotation (Hipparcos/Gaia convention); identical to
# cluster.phase_space so both channels live in the same frame.
_A_ICRS_TO_GAL = np.array([
    [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669],
])


def _col(df: pd.DataFrame, *names) -> np.ndarray:
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce").to_numpy(float)
    return np.full(len(df), np.nan)


def phase_space_6d(df: pd.DataFrame, parallax_floor_mas: float = 0.5) -> pd.DataFrame:
    """Add heliocentric Galactic ``X_pc,Y_pc,Z_pc`` and ``U_kms,V_kms,W_kms``.

    Requires columns ``ra``/``dec`` (deg), ``parallax`` (mas), ``pmra``/``pmdec``
    (mas/yr, Gaia convention: ``pmra`` already carries ``cos(dec)``) and
    ``radial_velocity`` (km/s).  Sources with parallax below
    ``parallax_floor_mas`` or a missing radial velocity get NaN velocity so they
    are dropped from the encounter search rather than silently mis-placed.
    """
    ra = np.radians(_col(df, "ra"))
    dec = np.radians(_col(df, "dec"))
    plx = _col(df, "parallax")
    pmra = _col(df, "pmra")            # mu_alpha* (includes cos dec)
    pmdec = _col(df, "pmdec")
    rv = _col(df, "radial_velocity", "rv", "dr2_radial_velocity")

    dist = np.where(plx >= parallax_floor_mas, 1000.0 / plx, np.nan)  # pc

    ca, sa = np.cos(ra), np.sin(ra)
    cd, sd = np.cos(dec), np.sin(dec)

    # Local ICRS orthonormal triad at each star.
    r_hat = np.stack([cd * ca, cd * sa, sd], axis=0)          # radial (away)
    a_hat = np.stack([-sa, ca, np.zeros_like(sa)], axis=0)    # +RA
    d_hat = np.stack([-sd * ca, -sd * sa, cd], axis=0)        # +Dec

    # Tangential speeds (km/s); pmra already cos-dec corrected -> use directly.
    v_a = _K_AUYR_KMS * pmra * dist / 1000.0
    v_d = _K_AUYR_KMS * pmdec * dist / 1000.0

    # Space velocity in ICRS Cartesian, then rotate to Galactic.
    v_icrs = rv * r_hat + v_a * a_hat + v_d * d_hat            # (3, N)
    r_icrs = dist * r_hat                                      # (3, N), pc
    v_gal = _A_ICRS_TO_GAL @ v_icrs
    r_gal = _A_ICRS_TO_GAL @ r_icrs

    out = df.copy()
    out["dist_pc"] = dist
    out["X_pc"], out["Y_pc"], out["Z_pc"] = r_gal[0], r_gal[1], r_gal[2]
    out["U_kms"], out["V_kms"], out["W_kms"] = v_gal[0], v_gal[1], v_gal[2]
    out["v_total_kms"] = np.sqrt(v_gal[0] ** 2 + v_gal[1] ** 2 + v_gal[2] ** 2)
    # Velocity is only defined where a radial velocity exists.
    no_rv = ~np.isfinite(rv)
    for c in ("U_kms", "V_kms", "W_kms", "v_total_kms"):
        out.loc[no_rv, c] = np.nan
    return out


__all__ = ["phase_space_6d"]
