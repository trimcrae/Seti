"""Contamination screens for CENOTAPH.

Ordered by how much damage each has historically done.

1. **Background-galaxy confusion in the beam.** This is the universal killer of
   Dysonian candidates: every Project Hephaistos candidate died of it, and
   JWST/MIRI resolved two of them into a Hot DOG at z≈0.9 and a dusty starburst
   at z≈0.4, both within ~1″. For a far-IR leg it is *worse* than for WISE work
   because IRAS and AKARI/FIS beams are 40–180″ rather than 6–12″, so the
   chance-coincidence area is 10²–10⁴ times larger. It is therefore a funnel
   stage here, not follow-up: high Galactic latitude, an explicit chance-match
   expectation computed per band, and a beam-crowding count.

2. **Unresolved binaries.** Zackrisson et al. (2018) executed a small
   underluminosity search on Gaia DR1 × RAVE DR5 and their one followed-up
   candidate, TYC 6111-1162-1, resolved to an unseen binary. That is a
   *known outcome* of this method, so it is screened explicitly rather than
   hoped away. Note the two directions:

   * an ordinary cool companion makes a star **over**luminous — the wrong sign,
     which is why the underluminous tail is the clean one;
   * a **hot** companion (white dwarf, sdB) biases the composite Teff *upward*,
     so the star is compared to hotter, brighter twins and appears
     underluminous. This is the dangerous one, and it is caught by its
     ultraviolet excess and by the SED goodness-of-fit.

3. **Edge-on large-grain disks.** They can grey-dim. They are separated by the
   energy-closure ratio in ``budget.py`` (a disk intercepts only its own solid
   angle) and, independently, by kinematics: a star with a large space velocity
   is old and has no business carrying a massive primordial disk.

4. **Blending and crowding.** These make a star *brighter*, so they are the safe
   direction for an underluminosity search — but a red neighbour blended into
   BP/RP fakes reddening, so the corrected BP/RP excess factor is still checked.

5. **Variability.** A star caught in a low state mimics a grey deficit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Hot dust-obscured galaxies: the class that ate Hephaistos candidates D and E.
HOT_DOG_DENSITY_PER_ARCSEC2: float = 9.0e-6

# All-sky far-IR catalogue source densities (sources per square degree), used to
# compute the *expected* number of chance associations before any candidate is
# believed. AKARI/FIS BSC is ~427k sources over the full sky.
FAR_IR_SOURCE_DENSITY_PER_SQDEG: dict[str, float] = {
    "akari": 427_000.0 / 41_252.96,
    "iras": 245_889.0 / 41_252.96,
}


# Measured on the runner from the full Gaia DR3 source table aggregated on a
# HEALPix nside=16 grid (1.81e9 sources, 3072 pixels) -- see
# results/farir_stats/gaia_density.json. Median sources per square degree in
# each |b| band. This turns the Galactic-latitude cut from a rule of thumb into
# a computed beam-crowding budget.
GAIA_DENSITY_PER_DEG2_BY_ABSB: tuple[tuple[float, float, float], ...] = (
    (0.0, 5.0, 101_852.8), (5.0, 10.0, 82_944.3), (10.0, 20.0, 30_298.7),
    (20.0, 30.0, 12_663.6), (30.0, 45.0, 6_573.8), (45.0, 60.0, 3_938.8),
    (60.0, 90.0, 3_118.8),
)

# SFD 100-um cirrus surface brightness, same source
# (results/farir_stats/cirrus_levels.json). Cirrus, not detector noise, sets the
# practical far-IR point-source limit, and it is a steep function of latitude:
# 64.5 MJy/sr in the plane versus 2.3 MJy/sr at the pole.
CIRRUS_I100_MJY_SR_BY_ABSB: tuple[tuple[float, float, float], ...] = (
    (0.0, 5.0, 64.508), (5.0, 10.0, 21.883), (10.0, 20.0, 8.598),
    (20.0, 30.0, 4.244), (30.0, 90.0, 2.5),
)


def _band_lookup(table, abs_b: float) -> float:
    for lo, hi, v in table:
        if lo <= abs_b < hi:
            return v
    return table[-1][2]


def gaia_density_at(abs_b_deg: float) -> float:
    """Measured Gaia DR3 source density (per deg²) at Galactic latitude ``|b|``."""
    return _band_lookup(GAIA_DENSITY_PER_DEG2_BY_ABSB, abs(abs_b_deg))


def cirrus_i100_at(abs_b_deg: float) -> float:
    """Measured SFD 100-µm cirrus brightness (MJy/sr) at ``|b|``."""
    return _band_lookup(CIRRUS_I100_MJY_SR_BY_ABSB, abs(abs_b_deg))


def beam_crowding_expectation(abs_b_deg: float, radius_arcsec: float,
                              bright_fraction: float = 1.0) -> float:
    """Expected number of Gaia sources inside a far-IR beam at latitude ``|b|``.

    At full Gaia depth a 25″ AKARI beam contains ~0.06 sources at the pole but
    ~4.3 in the plane, which is why the far-IR leg is restricted to high
    latitude and why the *measured* neighbour count (``count_beam_neighbours``,
    run per surviving candidate at G < 18) is a funnel stage rather than a
    footnote. ``bright_fraction`` scales to a magnitude-limited subset.
    """
    area_deg2 = math.pi * (radius_arcsec / 3600.0) ** 2
    return gaia_density_at(abs_b_deg) * area_deg2 * bright_fraction


def chance_match_probability(density_per_sqdeg: float, radius_arcsec: float) -> float:
    """Poisson probability of ≥1 unrelated catalogue source inside a radius."""
    area_sqdeg = math.pi * (radius_arcsec / 3600.0) ** 2
    return 1.0 - math.exp(-density_per_sqdeg * area_sqdeg)


def expected_false_matches(n_targets: int, density_per_sqdeg: float,
                           radius_arcsec: float) -> float:
    """How many of ``n_targets`` will match a far-IR source purely by chance.

    With ~10⁶ targets and a 40″ AKARI beam this is in the thousands, which is
    why a far-IR association is evidence only when the *closure ratio* is right
    and the beam is uncrowded — never on positional coincidence alone.
    """
    return n_targets * chance_match_probability(density_per_sqdeg, radius_arcsec)


def background_galaxy_probability(radius_arcsec: float,
                                  density_per_arcsec2: float
                                  = HOT_DOG_DENSITY_PER_ARCSEC2) -> float:
    """Probability of an unrelated obscured galaxy inside the beam."""
    return 1.0 - math.exp(-density_per_arcsec2 * math.pi * radius_arcsec**2)


# --- Gaia-native quality screens ---------------------------------------------
def bp_rp_excess_factor_corrected(bp_rp, excess_factor):
    """Riello et al. (2021) corrected BP/RP excess factor ``C*``.

    ``|C*|`` above a few σ means the BP/RP windows contain flux the G aperture
    does not: a blend, an extended source, or nebulosity. Blends brighten a
    star (harmless for an underluminosity hunt) but redden its BP−RP, which
    would be mistaken for a dust column, so the check stays in.
    """
    x = np.asarray(bp_rp, dtype=float)
    c = np.asarray(excess_factor, dtype=float)
    f = np.where(
        x < 0.5,
        1.154360 + 0.033772 * x + 0.032277 * x**2,
        np.where(
            x < 4.0,
            1.162004 + 0.011464 * x + 0.049255 * x**2 - 0.005879 * x**3,
            1.057572 + 0.140537 * x,
        ),
    )
    return c - f


def bp_rp_excess_sigma(g_mag):
    """1σ scatter of ``C*`` as a function of G (Riello et al. 2021, Eq. 18)."""
    g = np.asarray(g_mag, dtype=float)
    return 0.0059898 + 8.817481e-12 * np.power(g, 7.618399)


def gaia_variability_amplitude(phot_g_mean_flux_over_error, phot_g_n_obs):
    """Gaia photometric-scatter proxy (mag): ``√N · 2.5/ln10 / (F/σ_F)``.

    The standard trick (Belokurov et al. 2017): Gaia publishes the *error on the
    mean* flux, which for a variable star is inflated by the intrinsic scatter.
    A star caught in a low state — the classic false grey deficit — is variable,
    so this is the cheapest first-pass constancy test, later confirmed against
    ZTF/ATLAS.
    """
    foe = np.asarray(phot_g_mean_flux_over_error, dtype=float)
    n = np.asarray(phot_g_n_obs, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.maximum(n, 1.0)) * (2.5 / math.log(10.0)) / foe


def tangential_velocity_kms(pmra, pmdec, parallax_mas):
    """``v_tan`` (km/s) — a crude but robust kinematic age proxy.

    Thin-disk stars young enough to retain a massive primordial disk have small
    ``v_tan``; a star with ``v_tan ≳ 80 km/s`` belongs to the thick disk or halo
    and is several Gyr old, which strongly disfavours the edge-on-disk
    explanation for a grey deficit.
    """
    k = 4.740470446
    mu = np.hypot(np.asarray(pmra, dtype=float), np.asarray(pmdec, dtype=float))
    plx = np.asarray(parallax_mas, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return k * mu / plx


@dataclass
class VetThresholds:
    ruwe_max: float = 1.2
    parallax_over_error_min: float = 20.0
    ipd_frac_multi_peak_max: float = 2.0
    astrometric_excess_noise_sig_max: float = 2.0
    bp_rp_excess_nsigma_max: float = 3.0
    variability_amp_max_mag: float = 0.05
    galactic_latitude_min_deg: float = 20.0
    uv_excess_nsigma_max: float = 3.0
    """A *negative* UV residual this significant means a hot companion."""
    beam_neighbour_g_max: float = 18.0
    """Neighbours fainter than this cannot supply a far-IR source of interest."""


def vet_table(df: pd.DataFrame, thr: VetThresholds | None = None) -> pd.DataFrame:
    """Apply every catalogue-level screen; return one boolean column per test.

    Missing inputs never silently pass: a column that is absent produces
    ``NaN``-driven ``False`` for the *pass* flag and is recorded in
    ``vet_missing`` so that the summary can report degraded coverage rather
    than pretending the test was run.
    """
    thr = thr or VetThresholds()
    n = len(df)
    out = pd.DataFrame(index=df.index)
    missing: list[str] = []

    def col(name, default=np.nan):
        if name not in df.columns:
            missing.append(name)
            return pd.Series(np.full(n, default), index=df.index)
        return pd.to_numeric(df[name], errors="coerce")

    ruwe = col("ruwe")
    out["pass_ruwe"] = (ruwe < thr.ruwe_max).fillna(False)

    poe = col("parallax_over_error")
    out["pass_parallax"] = (poe > thr.parallax_over_error_min).fillna(False)

    ipd = col("ipd_frac_multi_peak")
    out["pass_ipd"] = (ipd <= thr.ipd_frac_multi_peak_max).fillna(False)

    aen = col("astrometric_excess_noise_sig")
    out["pass_excess_noise"] = (aen < thr.astrometric_excess_noise_sig_max).fillna(False)

    nss = col("non_single_star", 0.0)
    out["pass_non_single_star"] = (nss == 0).fillna(False)

    bp_rp = col("bp_rp")
    xs = col("phot_bp_rp_excess_factor")
    g = col("g_mag")
    cstar = bp_rp_excess_factor_corrected(bp_rp, xs)
    csig = bp_rp_excess_sigma(g)
    out["cstar"] = cstar
    with np.errstate(divide="ignore", invalid="ignore"):
        out["cstar_nsigma"] = np.abs(cstar) / csig
    out["pass_blend"] = (out["cstar_nsigma"] < thr.bp_rp_excess_nsigma_max).fillna(False)

    amp = gaia_variability_amplitude(col("phot_g_mean_flux_over_error"),
                                     col("phot_g_n_obs"))
    out["var_amp_mag"] = amp
    out["pass_constant"] = (pd.Series(amp, index=df.index)
                            < thr.variability_amp_max_mag).fillna(False)

    b = col("b_gal")
    out["pass_latitude"] = (np.abs(b) > thr.galactic_latitude_min_deg).fillna(False)

    # Hot-companion screen. dm_nuv/dm_fuv are the twin-differential residuals;
    # a hot companion makes the star *brighter* than its twins in the UV, i.e.
    # a significantly negative residual, which no occulter or dust column can do.
    uv_flags = []
    for band in ("fuv", "nuv"):
        dm = col(f"dm_{band}")
        err = col(f"dm_{band}_err", 0.1)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = dm / err
        uv_flags.append((z < -thr.uv_excess_nsigma_max).fillna(False))
    out["uv_excess_hot_companion"] = uv_flags[0] | uv_flags[1]
    out["pass_no_hot_companion"] = ~out["uv_excess_hot_companion"]

    out["v_tan_kms"] = tangential_velocity_kms(col("pmra"), col("pmdec"), col("parallax"))

    # Beam crowding for the far-IR association.
    nbr = col("n_beam_neighbours")
    out["pass_beam_uncrowded"] = (nbr <= 0).fillna(False)

    binary_tests = ["pass_ruwe", "pass_ipd", "pass_excess_noise",
                    "pass_non_single_star", "pass_no_hot_companion"]
    out["pass_binarity"] = out[binary_tests].all(axis=1)

    core = ["pass_ruwe", "pass_parallax", "pass_ipd", "pass_excess_noise",
            "pass_non_single_star", "pass_blend", "pass_constant",
            "pass_no_hot_companion"]
    out["pass_core"] = out[core].all(axis=1)
    out["pass_far_ir_context"] = out[["pass_latitude", "pass_beam_uncrowded"]].all(axis=1)
    out["pass_all"] = out["pass_core"] & out["pass_far_ir_context"]
    out.attrs["vet_missing"] = sorted(set(missing))
    return out


def vet_coverage(vetted: pd.DataFrame) -> dict:
    """Per-test pass counts plus which tests could not be run at all."""
    cols = [c for c in vetted.columns if c.startswith("pass_")]
    return {
        "n": int(len(vetted)),
        "passes": {c: int(vetted[c].sum()) for c in cols},
        "missing_columns": list(vetted.attrs.get("vet_missing", [])),
    }


__all__ = [
    "FAR_IR_SOURCE_DENSITY_PER_SQDEG",
    "HOT_DOG_DENSITY_PER_ARCSEC2",
    "VetThresholds",
    "background_galaxy_probability",
    "bp_rp_excess_factor_corrected",
    "bp_rp_excess_sigma",
    "chance_match_probability",
    "expected_false_matches",
    "gaia_variability_amplitude",
    "tangential_velocity_kms",
    "vet_coverage",
    "vet_table",
]
