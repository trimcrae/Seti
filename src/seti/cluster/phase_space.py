"""Convert Gaia observables to 3D position and velocity for clustering.

Clustering must be done in a physical frame, not on the sky: two sources that are
close in RA/Dec can be kiloparsecs apart in distance.  From (ra, dec, parallax)
we build heliocentric Galactic Cartesian coordinates (pc); from proper motion
(and radial velocity when available) we build the tangential velocity (km/s),
which lets a *co-moving* group be distinguished from a chance line-of-sight
alignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_KMS_PER_MASYR_KPC = 4.740857     # v[km/s] = 4.74 * mu[mas/yr] * d[kpc]


def _col(df: pd.DataFrame, *names) -> np.ndarray:
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce").to_numpy()
    return np.full(len(df), np.nan)


def galactic_xyz(df: pd.DataFrame, parallax_floor_mas: float = 0.2) -> pd.DataFrame:
    """Heliocentric Galactic Cartesian X, Y, Z (pc) from ra/dec/parallax.

    Distance is 1000/parallax (pc); sources with parallax below
    ``parallax_floor_mas`` (unreliable/negative) get NaN.  Uses the standard ICRS
    -> Galactic rotation so X points to the Galactic centre, Y to rotation, Z to
    the north Galactic pole.
    """
    ra = np.radians(_col(df, "ra"))
    dec = np.radians(_col(df, "dec"))
    plx = _col(df, "parallax")
    dist = np.where(plx >= parallax_floor_mas, 1000.0 / plx, np.nan)

    # ICRS unit vector.
    x_icrs = np.cos(dec) * np.cos(ra)
    y_icrs = np.cos(dec) * np.sin(ra)
    z_icrs = np.sin(dec)
    # ICRS -> Galactic rotation matrix (Hipparcos/Gaia convention).
    a = np.array([
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669],
    ])
    xg = a[0, 0] * x_icrs + a[0, 1] * y_icrs + a[0, 2] * z_icrs
    yg = a[1, 0] * x_icrs + a[1, 1] * y_icrs + a[1, 2] * z_icrs
    zg = a[2, 0] * x_icrs + a[2, 1] * y_icrs + a[2, 2] * z_icrs
    out = df.copy()
    out["X_pc"] = xg * dist
    out["Y_pc"] = yg * dist
    out["Z_pc"] = zg * dist
    out["dist_pc"] = dist
    return out


def tangential_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Tangential-velocity components (km/s) from proper motion and distance.

    ``v = 4.74 * mu[mas/yr] * d[kpc]``.  Returns ``vtan_ra_kms``, ``vtan_dec_kms``
    and their magnitude ``vtan_kms``; a co-moving group shares these.
    """
    pmra = _col(df, "pmra")
    pmdec = _col(df, "pmdec")
    plx = _col(df, "parallax")
    d_kpc = np.where(plx > 0, 1.0 / plx, np.nan)
    out = df.copy()
    out["vtan_ra_kms"] = _KMS_PER_MASYR_KPC * pmra * d_kpc
    out["vtan_dec_kms"] = _KMS_PER_MASYR_KPC * pmdec * d_kpc
    out["vtan_kms"] = np.hypot(out["vtan_ra_kms"], out["vtan_dec_kms"])
    return out


__all__ = ["galactic_xyz", "tangential_velocity"]
