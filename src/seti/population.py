"""A realistic white-dwarf population model for the projected-sensitivity forecast.

This is a *forecast* input, not measured data.  We draw a magnitude-limited
white-dwarf sample whose distance, temperature, mass/radius and resulting Gaia
and WISE apparent magnitudes reproduce the published statistics of the Gaia
EDR3 white-dwarf catalogue (Gentile Fusillo et al. 2021), then attach a WISE
photometric-error model derived from the catalogue 5-sigma depths.  Running the
*same* analysis pipeline on this population yields the injection-recovery
completeness and the occurrence-rate sensitivity the real all-sky search would
achieve.

Key modelled ingredients (all cited in ``config/population.yaml``):
  * constant local space density -> p(d) proportional to d^2 out to d_max;
  * a cooling pile-up effective-temperature distribution;
  * a mass distribution sharply peaked near 0.6 Msun -> radius via mass-radius;
  * a Gaia G detection limit that makes cool/distant white dwarfs drop out;
  * WISE sigma_mag(m) = 1.0857 / SNR(m), with SNR=5 at the 5-sigma depth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .photometry import (
    band_freq_hz,
    flux_jy_to_mag,
    planck_bnu,
)

R_SUN_M = 6.957e8
PC_M = 3.086e16
ANCHOR = ("J", "H", "Ks")
PRED = ("W1", "W2")
GAIA_EPOCH = 2016.0
WISE_EPOCH = 2010.5


def mass_radius_rsun(mass_msun: np.ndarray) -> np.ndarray:
    """Approximate white-dwarf mass-radius relation (Nauenberg 1972 form)."""
    mch = 1.44
    x = (mass_msun / mch) ** (2.0 / 3.0)
    return 0.0127 * np.sqrt(1.0 / x - x)


def _photosphere_apparent_mag(teff, radius_rsun, dist_pc, band):
    """Apparent Vega magnitude of a blackbody WD photosphere in a band."""
    omega = (radius_rsun * R_SUN_M / (dist_pc * PC_M)) ** 2   # (R/d)^2, steradian-like
    f_jy = omega * np.pi * planck_bnu(teff, band_freq_hz(band)) * 1e26
    return flux_jy_to_mag(f_jy, band)


def _draw_teff(rng, n, tmin, tmax, tpileup):
    """Draw effective temperatures with a cool-end cooling pile-up.

    Modelled as an exponential preference for cooler temperatures with scale
    ``tpileup`` (more cool white dwarfs), truncated to [tmin, tmax].
    """
    out = np.empty(n)
    filled = 0
    while filled < n:
        draw = rng.exponential(tpileup, size=2 * n) + tmin
        draw = draw[(draw >= tmin) & (draw <= tmax)]
        take = min(len(draw), n - filled)
        out[filled:filled + take] = draw[:take]
        filled += take
    return out


def _wise_sigma_mag(mag, m5, sys_floor, snr_at_limit):
    """WISE photometric error model: sigma_mag = 1.0857 / SNR(m), with a floor."""
    snr = snr_at_limit * 10.0 ** (0.4 * (m5 - mag))
    sigma = 1.0857 / np.maximum(snr, 1e-3)
    return np.sqrt(sigma**2 + sys_floor**2)


def generate_population(cfg, seed: int = 11, depth_set: str = "catwise2020") -> pd.DataFrame:
    """Generate a magnitude-limited WD population in the analysis-ready schema.

    The returned frame carries the same columns the pipeline expects (Gaia
    astrometry placeholders, Teff, anchor J/H/Ks, photospheric W1/W2 with
    realistic errors), plus ``detected`` (Gaia + WISE photosphere detected) and
    ``dist_pc`` / ``mass_msun`` / ``radius_rsun`` for the sensitivity analysis.
    """
    pop = cfg.population["population"]
    depth = cfg.population["wise_depth"]
    rng = np.random.default_rng(seed)
    n = int(pop["n_draw"])

    # Distance: constant space density -> p(d) ~ d^2 out to d_max.
    d_max = pop["d_max_pc"]
    u = rng.uniform(0, 1, n)
    dist_pc = d_max * u ** (1.0 / 3.0)

    teff = _draw_teff(rng, n, pop["teff_min_k"], pop["teff_max_k"], pop["teff_pileup_k"])
    mass = np.clip(rng.normal(pop["mass_mean_msun"], pop["mass_sigma_msun"], n),
                   pop["mass_min_msun"], pop["mass_max_msun"])
    radius = mass_radius_rsun(mass)

    # Gaia G apparent magnitude -> detection limit (cool/distant WDs drop out).
    g_app = _photosphere_apparent_mag(teff, radius, dist_pc, "G")
    gaia_detected = g_app <= pop["gaia_g_limit"]

    # WISE photospheric apparent magnitudes + depth-based errors.
    cw = depth[depth_set]
    rows = {"teff": teff, "logg": 8.0 + np.zeros(n), "mass_msun": mass,
            "radius_rsun": radius, "dist_pc": dist_pc, "parallax": 1000.0 / dist_pc,
            "Gmag": g_app}
    for b in ANCHOR:
        m = _photosphere_apparent_mag(teff, radius, dist_pc, b)
        rows[f"{b}mag"] = m + rng.normal(0, 0.03, n)
        rows[f"e_{b}mag"] = 0.03
    w_detected = np.ones(n, dtype=bool)
    for b, m5 in (("W1", cw["W1_5sigma"]), ("W2", cw["W2_5sigma"])):
        m = _photosphere_apparent_mag(teff, radius, dist_pc, b)
        sig = _wise_sigma_mag(m, m5, depth["sys_floor_mag"], depth["snr_at_limit"])
        rows[f"{b}mag"] = m + rng.normal(0, sig)
        rows[f"e_{b}mag"] = sig
        rows[f"{b}_5sigma"] = m5
        w_detected &= m <= m5    # photosphere itself detected at >=5 sigma
    rows["w2snr"] = 1.0 / (0.4 * np.log(10.0) * rows["e_W2mag"])

    df = pd.DataFrame(rows)
    # Quality / astrometry columns the funnel expects (clean by construction --
    # the forecast isolates *sensitivity*, not contamination, which is measured
    # separately on the labelled synthetic sample).
    df["source_id"] = rng.integers(1_000_000_000, 9_000_000_000, n)
    df["ra"] = rng.uniform(0, 360, n)
    df["dec"] = rng.uniform(-40, 80, n)
    # Kinematically-motivated proper motions: tangential speed from a 2D
    # Maxwellian (Rayleigh) with the WD-population velocity dispersion, converted
    # via mu = v_tan / (4.74 d). Nearby WDs therefore have large proper motions.
    sigma_v = pop.get("vtan_dispersion_km_s", 40.0)
    v_tan = rng.rayleigh(sigma_v, n)                       # km/s
    mu_total = v_tan / (4.74 * dist_pc) * 1000.0           # mas/yr
    theta = rng.uniform(0, 2 * np.pi, n)
    df["pmra"] = mu_total * np.cos(theta)
    df["pmdec"] = mu_total * np.sin(theta)
    df["pmra_error"] = 1.0
    df["pmdec_error"] = 1.0
    df["parallax_over_error"] = df["parallax"] / 0.05
    df["ruwe"] = rng.uniform(0.9, 1.2, n)
    df["astrometric_excess_noise"] = rng.uniform(0, 0.3, n)
    df["pwd"] = rng.uniform(0.95, 1.0, n)
    df["cc_flags"] = "0000"
    df["ph_qual"] = "AA"
    df["ext_flg"] = 0
    df["n_wise_neighbours"] = 1
    df["gaia_nn_arcsec"] = rng.uniform(8, 30, n)
    df["W1mag_unwise"] = np.nan
    df["known_disk"] = False
    df["detected"] = gaia_detected & w_detected
    return df


__all__ = ["generate_population", "mass_radius_rsun"]
