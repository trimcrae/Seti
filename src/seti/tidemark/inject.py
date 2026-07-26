"""Synthetic parent samples and signal injection --- the calibration bench.

A spatial rate test is only worth running if it has been shown to (a) recover a
front that is really there and (b) *not* recover the Galaxy's own structure.
Both halves matter equally, and the second is the harder one: the stellar density
falls exponentially with Galactocentric radius and with height, the apparent
magnitude limit removes distant stars preferentially, extinction concentrates in
the plane, and metallicity has its own radial gradient.  Every one of those is a
route to a spurious "gradient in anomaly rate".

``synthetic_parent`` builds a parent sample with all of them switched on, so the
null tests are tests against the real confounders rather than against white
noise.  The injectors then add a known signal on top.

The separation the whole design turns on::

    p_flag(star) = p_exist(position, age)  x  p_detect(magnitude, extinction, ...)

``p_exist`` is the astrophysics we want; ``p_detect`` is the selection function.
The matched null is built from the ``p_detect`` covariates, so a test that works
recovers ``p_exist`` and is blind to ``p_detect``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..galactic.orbits import R0_KPC
from .ingest import add_galactic_frame, local_density, numeric

_A_GAL_TO_ICRS = np.array([
    [-0.0548755604162154, +0.4941094278755837, -0.8676661490190047],
    [-0.8734370902348850, -0.4448296299600112, -0.1980763734312015],
    [-0.4838350155487132, +0.7469822444972189, +0.4559837761750669],
])


def synthetic_parent(n: int = 40000, *, seed: int = 7, d_max_pc: float = 2500.0,
                     h_r_kpc: float = 2.6, h_z_pc: float = 350.0,
                     g_limit: float = 17.0, footprint: str = "all",
                     with_kinematics: bool = True) -> pd.DataFrame:
    """A Gaia-like parent sample carrying every confounder that matters.

    * exponential disk in ``R`` and ``z`` --- so the raw counts have a huge
      Galactocentric gradient that is *not* an anomaly-rate gradient;
    * a luminosity function plus a magnitude limit --- so detectability falls
      with distance, which in turn correlates with Galactocentric radius through
      the direction of the Galactic centre;
    * extinction that grows with path length in the plane --- so low ``|b|``
      sightlines are systematically harder;
    * a radial metallicity gradient and an age--velocity-dispersion relation ---
      the confounders for the age test.

    ``footprint`` may be ``"all"``, ``"high_lat"`` (|b| > 25, the classic
    extragalactic-survey shape), or ``"stripe"`` (a narrow declination band).
    """
    rng = np.random.default_rng(seed)
    want = int(n)
    keep = []
    ntot = 0
    # Rejection-sample heliocentric positions from a disk density profile.
    while ntot < want and len(keep) < 200:
        m = max(want * 4, 20000)
        x = rng.uniform(-d_max_pc, d_max_pc, m)
        y = rng.uniform(-d_max_pc, d_max_pc, m)
        z = rng.uniform(-3 * h_z_pc, 3 * h_z_pc, m)
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        inside = d <= d_max_pc
        r_gal = np.hypot(x / 1e3 - R0_KPC, y / 1e3)
        dens = np.exp(-(r_gal - R0_KPC) / h_r_kpc) / np.cosh(z / (2 * h_z_pc)) ** 2
        acc = inside & (rng.uniform(0, dens.max(), m) < dens)
        keep.append(np.stack([x[acc], y[acc], z[acc]], axis=1))
        ntot += int(acc.sum())
    pos = np.vstack(keep)[:want]
    if len(pos) < want:                      # pathological parameters: pad
        pos = np.vstack([pos] * (want // max(len(pos), 1) + 1))[:want]
    X, Y, Z = pos[:, 0], pos[:, 1], pos[:, 2]
    dist = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    # Back out sky coordinates (Galactic -> ICRS) so downstream code sees ra/dec.
    u_gal = pos.T / np.maximum(dist, 1e-9)
    u_icrs = _A_GAL_TO_ICRS @ u_gal
    ra = np.degrees(np.arctan2(u_icrs[1], u_icrs[0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(u_icrs[2], -1, 1)))
    b_deg = np.degrees(np.arcsin(np.clip(u_gal[2], -1, 1)))

    # Photometry: a broad absolute-magnitude distribution + distance modulus.
    abs_g = rng.normal(5.0, 2.2, want)
    ebv = 0.25 * (dist / 1000.0) * np.exp(-np.abs(Z) / 150.0) * rng.gamma(4.0, 0.25, want)
    g = abs_g + 5.0 * np.log10(np.maximum(dist, 1.0) / 10.0) + 3.1 * ebv
    bp_rp = np.clip(0.35 * (abs_g - 4.0) + rng.normal(0, 0.35, want) + 2.0 * ebv, -0.4, 4.5)

    r_gal_kpc = np.hypot(X / 1e3 - R0_KPC, Y / 1e3)
    feh = -0.06 * (r_gal_kpc - R0_KPC) - 0.25 * np.abs(Z) / 1000.0 + rng.normal(0, 0.18, want)
    df = pd.DataFrame({
        "source_id": np.arange(want, dtype=np.int64),
        "ra": ra, "dec": dec, "parallax": 1000.0 / np.maximum(dist, 1.0),
        "phot_g_mean_mag": g, "bp_rp": bp_rp, "ebv": ebv, "feh": feh,
        "n_obs": rng.integers(20, 200, want).astype(float),
    })
    if with_kinematics:
        # Age--velocity-dispersion: kinematically hot stars sit high and are old.
        age = np.clip(rng.gamma(2.0, 2.2, want) * (1.0 + 1.5 * np.abs(Z) / 1000.0), 0.1, 13.0)
        sigma = 18.0 * (age / 1.0) ** 0.35
        df["pmra"] = rng.normal(0, sigma, want) / (4.740857 * dist / 1000.0)
        df["pmdec"] = rng.normal(0, sigma, want) / (4.740857 * dist / 1000.0)
        df["true_age_gyr"] = age

    # Selection: magnitude limit + footprint.  Both are pure detectability.
    sel = df["phot_g_mean_mag"].to_numpy() < g_limit
    if footprint == "high_lat":
        sel &= np.abs(b_deg) > 25.0
    elif footprint == "stripe":
        sel &= np.abs(dec) < 12.0
    df = df[sel].reset_index(drop=True)
    df = add_galactic_frame(df)
    df["log_local_density"] = local_density(df)
    return df


def detectability(parent: pd.DataFrame, *, strength: float = 1.0) -> np.ndarray:
    """A hard, position-correlated *detectability* factor in [0, 1].

    Falls with apparent magnitude and extinction and rises with epoch count ---
    exactly the covariates the matched null is built from, and exactly the
    covariates that correlate with distance and therefore with position.
    """
    g = numeric(parent, "phot_g_mean_mag")
    ebv = numeric(parent, "ebv", 0.0)
    nobs = numeric(parent, "n_obs", 100.0)
    z = strength * (0.9 * (g - np.nanmedian(g)) + 3.0 * np.nan_to_num(ebv)
                    - 0.006 * (np.nan_to_num(nobs, nan=100.0) - 100.0))
    return 1.0 / (1.0 + np.exp(np.clip(z, -30, 30)))


def _draw(p: np.ndarray, rng) -> np.ndarray:
    return rng.uniform(0, 1, p.shape) < np.clip(p, 0, 1)


def inject_none(parent: pd.DataFrame, *, base_rate: float = 0.01, seed: int = 11,
                detect_strength: float = 1.0) -> np.ndarray:
    """Constant *intrinsic* rate; only detectability varies.  The pure-null case:
    the Galactic density gradient, the magnitude limit and the extinction map are
    all present, but there is no spatial structure in the anomaly rate."""
    rng = np.random.default_rng(seed)
    p = base_rate * detectability(parent, strength=detect_strength)
    return _draw(p / max(p.mean(), 1e-12) * base_rate, rng)


def inject_bubble(parent: pd.DataFrame, *, centre_pc=(600.0, -400.0, 0.0),
                  radius_pc: float = 900.0, contrast: float = 4.0,
                  base_rate: float = 0.01, seed: int = 12,
                  detect_strength: float = 1.0, edge_softness_pc: float = 0.0
                  ) -> np.ndarray:
    """A sharp-edged colonised volume: rate multiplied by ``contrast`` inside a
    sphere in heliocentric Galactic XYZ, on top of the detectability model."""
    rng = np.random.default_rng(seed)
    xyz = parent[["X_pc", "Y_pc", "Z_pc"]].to_numpy(float)
    r = np.sqrt(np.sum((xyz - np.asarray(centre_pc, float)) ** 2, axis=1))
    if edge_softness_pc > 0:
        inside = 1.0 / (1.0 + np.exp((r - radius_pc) / edge_softness_pc))
    else:
        inside = (r <= radius_pc).astype(float)
    p = base_rate * (1.0 + (contrast - 1.0) * inside) * detectability(
        parent, strength=detect_strength)
    return _draw(p, rng)


def inject_gradient(parent: pd.DataFrame, *, coord: str = "R_gal_kpc",
                    slope_ln_per_unit: float = 0.5, base_rate: float = 0.01,
                    seed: int = 13, detect_strength: float = 1.0) -> np.ndarray:
    """A genuine exponential gradient in *intrinsic* rate across ``coord``."""
    rng = np.random.default_rng(seed)
    x = pd.to_numeric(parent[coord], errors="coerce").to_numpy(float)
    x = np.where(np.isfinite(x), x, np.nanmedian(x))
    p = base_rate * np.exp(slope_ln_per_unit * (x - np.nanmedian(x))) * \
        detectability(parent, strength=detect_strength)
    return _draw(p, rng)


def inject_selection_artifact(parent: pd.DataFrame, *, base_rate: float = 0.01,
                              seed: int = 14, strength: float = 2.5) -> np.ndarray:
    """Rate driven *only* by detectability, hard.  Contains a strong apparent
    spatial gradient (because magnitude and extinction track position) with no
    intrinsic structure at all: the test must return a clean null."""
    rng = np.random.default_rng(seed)
    p = base_rate * detectability(parent, strength=strength)
    p = p / p.mean() * base_rate
    return _draw(p, rng)


def inject_age_dependence(parent: pd.DataFrame, *, base_rate: float = 0.01,
                          slope_per_gyr: float = 0.22, seed: int = 15,
                          saturate_gyr: float | None = None,
                          detect_strength: float = 1.0) -> np.ndarray:
    """Rate rising with true stellar age (optionally saturating above
    ``saturate_gyr``), on top of the detectability model."""
    rng = np.random.default_rng(seed)
    age = numeric(parent, "true_age_gyr")
    age = np.where(np.isfinite(age), age, np.nanmedian(age))
    eff = np.minimum(age, saturate_gyr) if saturate_gyr else age
    p = base_rate * np.exp(slope_per_gyr * (eff - np.nanmedian(eff))) * \
        detectability(parent, strength=detect_strength)
    return _draw(p, rng)


__all__ = ["synthetic_parent", "detectability", "inject_none", "inject_bubble",
           "inject_gradient", "inject_selection_artifact", "inject_age_dependence"]
