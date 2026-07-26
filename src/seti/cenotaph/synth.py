"""Synthetic CENOTAPH populations — the offline substrate for the test suite.

The sandbox has no archive egress, so every detector in this channel is
developed and validated against a population whose *truth* is known by
construction: a smooth main sequence in (Teff, log g, [M/H], [α/Fe]) with a
realistic intrinsic scatter, into which we can inject

* a grey deficit (an occulter),
* a reddening column (interstellar dust, which must NOT be flagged),
* a metal-poor subdwarf population (which must NOT be flagged),
* a hot companion (which biases Teff up and mimics underluminosity),

and check that the funnel keeps exactly the first one.

The absolute-magnitude surface is a linear approximation, which is fine and in
fact *desirable*: the twin estimator is explicitly a first-order differential
cancellation, so a surface with known gradients lets the tests assert the
cancellation quantitatively instead of hoping for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .extinction import BANDS, a_over_av

# dM_Ks/dTeff ≈ −1.6 mmag/K, dM_Ks/dlogg ≈ +0.5, dM_Ks/d[M/H] ≈ −0.5
# (metal-poor dwarfs are smaller and fainter at fixed Teff, hence the sign).
_M_KS_0 = 3.30
_DM_DTEFF = -1.6e-3
_DM_DLOGG = 0.5
_DM_DMH = -0.5
_DM_DALPHA = 0.2

# Intrinsic colours as smooth functions of Teff, anchored at 5500 K. Only the
# *gradients* matter to the tests; the twin construction removes the zero point.
_COLOUR_ANCHOR = {
    "fuv": 8.0, "nuv": 6.2, "bp": 1.85, "g": 1.55, "rp": 1.05,
    "j": 0.40, "h": 0.10, "ks": 0.0, "w1": -0.03, "w2": -0.05,
}
_COLOUR_SLOPE = {  # d(colour − Ks)/dTeff, per 1000 K
    "fuv": -2.2, "nuv": -1.6, "bp": -0.55, "g": -0.45, "rp": -0.30,
    "j": -0.12, "h": -0.03, "ks": 0.0, "w1": 0.0, "w2": 0.0,
}


def intrinsic_m_ks(teff, logg, mh, alphafe):
    """The noiseless absolute-Ks surface used to build the population."""
    return (_M_KS_0
            + _DM_DTEFF * (np.asarray(teff, float) - 5500.0)
            + _DM_DLOGG * (np.asarray(logg, float) - 4.40)
            + _DM_DMH * np.asarray(mh, float)
            + _DM_DALPHA * np.asarray(alphafe, float))


def intrinsic_colour(band: str, teff):
    """Intrinsic ``band − Ks`` colour at ``teff``."""
    t = (np.asarray(teff, float) - 5500.0) / 1000.0
    return _COLOUR_ANCHOR[band] + _COLOUR_SLOPE[band] * t


def make_population(n: int = 4000, seed: int = 7, intrinsic_scatter: float = 0.05,
                    bands: list[str] | None = None,
                    parallax_over_error: float = 50.0,
                    phot_err: float = 0.02,
                    mh_mean: float = -0.10, mh_sigma: float = 0.25) -> pd.DataFrame:
    """A clean synthetic dwarf population with no injected signal."""
    rng = np.random.default_rng(seed)
    bands = bands or [b.name for b in BANDS]

    teff = rng.uniform(4600.0, 6400.0, n)
    logg = rng.normal(4.40, 0.12, n)
    mh = rng.normal(mh_mean, mh_sigma, n)
    alphafe = np.clip(-0.25 * mh + rng.normal(0.0, 0.03, n), -0.1, 0.5)

    m_ks_true = intrinsic_m_ks(teff, logg, mh, alphafe)
    m_ks_true = m_ks_true + rng.normal(0.0, intrinsic_scatter, n)

    plx = rng.uniform(2.0, 12.0, n)                    # mas -> 83-500 pc
    dist_mod = 5.0 * np.log10(1000.0 / plx) - 5.0
    plx_err = plx / parallax_over_error

    df = pd.DataFrame({
        "source_id": np.arange(1, n + 1, dtype=np.int64),
        "teff": teff, "logg": logg, "mh": mh, "alphafe": alphafe,
        "parallax": plx, "parallax_error": plx_err,
        "parallax_over_error": np.full(n, parallax_over_error),
        "ruwe": rng.uniform(0.85, 1.05, n),
        "ipd_frac_multi_peak": np.zeros(n),
        "astrometric_excess_noise_sig": rng.uniform(0.0, 1.0, n),
        "non_single_star": np.zeros(n),
        "phot_g_mean_flux_over_error": np.full(n, 3000.0),
        "phot_g_n_obs": np.full(n, 200.0),
        "phot_bp_rp_excess_factor": np.full(n, np.nan),
        "pmra": rng.normal(0.0, 20.0, n), "pmdec": rng.normal(0.0, 20.0, n),
        "b_gal": rng.uniform(-80.0, 80.0, n),
        "ra": rng.uniform(0.0, 360.0, n), "dec": rng.uniform(-60.0, 60.0, n),
        "m_ks_true": m_ks_true,
        "grey_true": np.zeros(n), "av_true": np.zeros(n),
        "label": "clean",
    })

    # Apparent magnitudes: intrinsic absolute mag + distance modulus, per band.
    for b in bands:
        m_abs = m_ks_true + intrinsic_colour(b, teff)
        df[f"{b}_mag"] = m_abs + dist_mod + rng.normal(0.0, phot_err, n)
        df[f"{b}_mag_error"] = phot_err
    # Gaia-native aliases used by the vetting layer.
    df["g_mag"] = df["g_mag"]
    df["bp_rp"] = df["bp_mag"] - df["rp_mag"]
    df["phot_bp_rp_excess_factor"] = (
        1.162004 + 0.011464 * df["bp_rp"] + 0.049255 * df["bp_rp"] ** 2
        - 0.005879 * df["bp_rp"] ** 3 + rng.normal(0.0, 0.002, n)
    )
    return df


def inject_grey(df: pd.DataFrame, idx, grey_mag: float,
                bands: list[str] | None = None) -> pd.DataFrame:
    """Add an achromatic deficit ``grey_mag`` to every band for rows ``idx``."""
    out = df.copy()
    bands = bands or [b.name for b in BANDS]
    for b in bands:
        col = f"{b}_mag"
        if col in out.columns:
            out.loc[idx, col] = out.loc[idx, col] + grey_mag
    out.loc[idx, "grey_true"] = grey_mag
    out.loc[idx, "label"] = "grey"
    return out


def inject_reddening(df: pd.DataFrame, idx, av: float,
                     bands: list[str] | None = None) -> pd.DataFrame:
    """Add a chromatic column ``A_V`` following the R_V = 3.1 law."""
    out = df.copy()
    bands = bands or [b.name for b in BANDS]
    for b in bands:
        col = f"{b}_mag"
        if col in out.columns:
            out.loc[idx, col] = out.loc[idx, col] + av * a_over_av(b)
    out.loc[idx, "av_true"] = av
    out.loc[idx, "label"] = "reddened"
    return out


def add_subdwarfs(df: pd.DataFrame, n: int = 200, seed: int = 11,
                  mh_range: tuple[float, float] = (-1.6, -0.9),
                  bands: list[str] | None = None) -> pd.DataFrame:
    """Append a genuine metal-poor subdwarf population.

    These stars are *really* underluminous relative to solar-metallicity dwarfs
    of the same Teff — by ``|_DM_DMH| · Δ[M/H]`` ≈ 0.5–0.8 mag, i.e. far more
    than the injected occulter signal. They are the channel's primary
    astrophysical false positive, and the twin estimator must return them to
    zero because [M/H] is one of the matching axes.
    """
    rng = np.random.default_rng(seed)
    bands = bands or [b.name for b in BANDS]
    sub = make_population(n=n, seed=seed + 1, bands=bands,
                          mh_mean=float(np.mean(mh_range)),
                          mh_sigma=(mh_range[1] - mh_range[0]) / 4.0)
    sub["mh"] = rng.uniform(mh_range[0], mh_range[1], n)
    sub["alphafe"] = np.clip(-0.25 * sub["mh"] + rng.normal(0.0, 0.03, n), -0.1, 0.5)
    m_ks = intrinsic_m_ks(sub["teff"], sub["logg"], sub["mh"], sub["alphafe"])
    m_ks = m_ks + rng.normal(0.0, 0.05, n)
    dist_mod = 5.0 * np.log10(1000.0 / sub["parallax"]) - 5.0
    for b in bands:
        sub[f"{b}_mag"] = (m_ks + intrinsic_colour(b, sub["teff"]) + dist_mod
                           + rng.normal(0.0, 0.02, n))
    sub["m_ks_true"] = m_ks
    sub["label"] = "subdwarf"
    sub["source_id"] = np.arange(10_000_000, 10_000_000 + n, dtype=np.int64)
    sub["bp_rp"] = sub["bp_mag"] - sub["rp_mag"]
    return pd.concat([df, sub], ignore_index=True)


__all__ = ["add_subdwarfs", "inject_grey", "inject_reddening", "intrinsic_colour",
           "intrinsic_m_ks", "make_population"]
