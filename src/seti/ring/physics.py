"""Ring physics for RING (S63): rings around the dead.

Everything here is a pure function of catalogue numbers, so the whole channel
is testable offline.  The three quantities the channel turns on:

* **The ring temperature.**  A thin absorbing ring at radius ``r`` around a
  host of luminosity ``L`` (a white dwarf's photosphere, a pulsar's spin-down
  power) sits at ``T^4 = L / (16 pi sigma r^2)``.  Osmanov (2016, 2018)
  computes habitable-temperature rings around pulsars at 300-700 K; that band
  is where a Wien peak falls in W2 (630 K) and W1 (852 K), so it is a **W1/W2
  excess**, which escapes the frozen W3/W4 ceiling that bounds every warm-dust
  channel in this repository.
* **The ring flux.**  A ring intercepting a fraction ``f`` of ``L`` re-emits
  ``f L`` as a blackbody at ``T``; at distance ``d`` its flux density is
  ``F_nu = (f L / 4 pi d^2) * pi B_nu(T) / (sigma T^4)``.  Inverting for ``f``
  at the survey limit gives the **per-host sensitivity** -- the covering
  fraction the survey could have seen -- which is how a null is reported as a
  number per host rather than as a silence.
* **The cooling ceiling.**  A free-floating object of planetary mass cannot be
  brighter than a 13 M_J object of its age; Burrows et al. (2001, RvMP 73,
  719) give the analytic brown-dwarf cooling law used here.  Its accuracy is a
  factor ~2-3 against full evolutionary tracks, so the ceiling carries an
  explicit systematic margin and the age/mass degeneracy is a named, reported
  killer rather than a hidden one.
"""

from __future__ import annotations

import numpy as np

from ..photometry import BANDS, band_freq_hz, planck_bnu

SIGMA_SB = 5.670374419e-8          # W m^-2 K^-4
L_SUN_W = 3.828e26
R_SUN_M = 6.957e8
AU_M = 1.495978707e11
PC_M = 3.0856775814913673e16
G_SI = 6.67430e-11
M_SUN_KG = 1.98847e30
M_JUP_MSUN = 9.5458e-4
I_NS_KG_M2 = 1.0e38                # canonical 10^45 g cm^2
PLANETARY_MASS_LIMIT_MJ = 13.0     # deuterium-burning limit


# --------------------------------------------------------------------------
# Blackbody colours and temperatures
# --------------------------------------------------------------------------

def blackbody_colour(t_k, band1: str = "W1", band2: str = "W2"):
    """Vega colour ``band1 - band2`` of a blackbody at ``t_k`` (vectorised)."""
    t = np.asarray(t_k, float)
    f1 = planck_bnu(t, band_freq_hz(band1)) / BANDS[band1]["zp_jy"]
    f2 = planck_bnu(t, band_freq_hz(band2)) / BANDS[band2]["zp_jy"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return -2.5 * np.log10(f1 / f2)


_T_GRID = np.geomspace(80.0, 6000.0, 1200)


def colour_to_temperature(colour, band1: str = "W1", band2: str = "W2"):
    """Blackbody temperature from a ``band1 - band2`` colour (NaN off the grid).

    The blackbody colour is monotonic in T for the WISE pairs, so the inversion
    is a table lookup.  A colour bluer than the Rayleigh-Jeans limit has no
    blackbody temperature and returns NaN rather than the grid edge.
    """
    c = np.asarray(colour, float)
    grid_c = blackbody_colour(_T_GRID, band1, band2)          # decreasing in T
    order = np.argsort(grid_c)
    out = np.interp(c, grid_c[order], _T_GRID[order], left=np.nan, right=np.nan)
    return out


# --------------------------------------------------------------------------
# Rings
# --------------------------------------------------------------------------

def ring_equilibrium_radius_au(l_host_w, t_k):
    """Radius at which a thin absorbing ring sits at ``t_k`` around ``l_host_w``."""
    lw = np.asarray(l_host_w, float)
    t = np.asarray(t_k, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.sqrt(lw / (16.0 * np.pi * SIGMA_SB * t ** 4))
    return r / AU_M


def ring_temperature_k(l_host_w, r_au):
    """Equilibrium temperature of a thin absorbing ring at ``r_au``."""
    lw = np.asarray(l_host_w, float)
    r = np.asarray(r_au, float) * AU_M
    with np.errstate(divide="ignore", invalid="ignore"):
        return (lw / (16.0 * np.pi * SIGMA_SB * r ** 2)) ** 0.25


def ring_flux_density_jy(l_host_w, f_intercept, t_k, d_pc, band: str):
    """Flux density (Jy) of a ring re-emitting ``f L`` as a blackbody at ``t_k``.

    ``pi B_nu(T) / (sigma T^4)`` integrates to one over frequency, so this is
    the bolometric flux ``f L / 4 pi d^2`` distributed with the Planck shape.
    """
    lw = np.asarray(l_host_w, float)
    t = np.asarray(t_k, float)
    d = np.asarray(d_pc, float) * PC_M
    f = np.asarray(f_intercept, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        bol = f * lw / (4.0 * np.pi * d ** 2)
        shape = np.pi * planck_bnu(t, band_freq_hz(band)) / (SIGMA_SB * t ** 4)
    return bol * shape * 1e26


def min_intercept_fraction(l_host_w, t_k, d_pc, band: str, limit_jy: float):
    """Covering fraction at which a ring at ``t_k`` reaches ``limit_jy``.

    Values above one mean the survey could not have seen even a complete ring
    at that temperature around that host; the host is then reported as
    ``unconstrained`` rather than as a null.
    """
    f1 = ring_flux_density_jy(l_host_w, 1.0, t_k, d_pc, band)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f1 > 0, float(limit_jy) / f1, np.inf)


# --------------------------------------------------------------------------
# Hosts
# --------------------------------------------------------------------------

def spin_down_luminosity_w(p0_s, p1, i_kg_m2: float = I_NS_KG_M2):
    """Pulsar spin-down power ``4 pi^2 I Pdot / P^3`` (W)."""
    p = np.asarray(p0_s, float)
    pd_ = np.asarray(p1, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 4.0 * np.pi ** 2 * i_kg_m2 * pd_ / p ** 3
    return np.where(np.isfinite(out) & (out > 0), out, np.nan)


def wd_radius_m(logg_cgs, mass_msun):
    """White-dwarf radius from surface gravity and mass: ``R = sqrt(G M / g)``."""
    g = 10.0 ** np.asarray(logg_cgs, float) * 1e-2       # cm s^-2 -> m s^-2
    m = np.asarray(mass_msun, float) * M_SUN_KG
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(G_SI * m / g)


def wd_luminosity_w(teff_k, logg_cgs, mass_msun):
    """White-dwarf photospheric luminosity ``4 pi R^2 sigma T^4`` (W)."""
    r = wd_radius_m(logg_cgs, mass_msun)
    t = np.asarray(teff_k, float)
    return 4.0 * np.pi * r ** 2 * SIGMA_SB * t ** 4


def luminosity_from_solid_angle_w(omega_sr, teff_k, d_pc):
    """Luminosity of a blackbody of solid angle ``omega`` at ``d_pc``.

    ``omega = (R/d)^2`` for a sphere, so ``L = 4 pi d^2 omega sigma T^4``.
    Used where the SED fit supplies the solid angle directly.
    """
    d = np.asarray(d_pc, float) * PC_M
    return 4.0 * np.pi * d ** 2 * np.asarray(omega_sr, float) * SIGMA_SB \
        * np.asarray(teff_k, float) ** 4


# --------------------------------------------------------------------------
# Cooling ceiling for free-floating planetary-mass objects
# --------------------------------------------------------------------------

def burrows_luminosity_lsun(mass_msun, age_gyr, kappa_r: float = 1e-2):
    """Burrows et al. (2001) analytic brown-dwarf cooling luminosity.

    ``L ~ 4e-5 L_sun (t / 1 Gyr)^-1.3 (M / 0.05 M_sun)^2.64 (kappa_R / 1e-2)^0.35``.
    Quoted by its authors as good to a factor of ~2-3 against full tracks for
    ages beyond a few tens of Myr; the ceiling below carries a margin for that.
    """
    m = np.asarray(mass_msun, float)
    t = np.asarray(age_gyr, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 4e-5 * t ** (-1.3) * (m / 0.05) ** 2.64 * (kappa_r / 1e-2) ** 0.35


def burrows_teff_k(mass_msun, age_gyr, kappa_r: float = 1e-2):
    """Companion effective-temperature law from the same source."""
    m = np.asarray(mass_msun, float)
    t = np.asarray(age_gyr, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1550.0 * t ** (-0.32) * (m / 0.05) ** 0.83 * (kappa_r / 1e-2) ** 0.088


def planetary_cooling_ceiling_log_lsun(age_gyr, margin_dex: float = 0.5,
                                       mass_limit_mj: float = PLANETARY_MASS_LIMIT_MJ):
    """log10 L/L_sun above which no object below ``mass_limit_mj`` can sit at ``age_gyr``.

    The margin (default 0.5 dex, a factor 3) absorbs the analytic law's own
    error.  An object brighter than this **and** catalogued as planetary-mass
    is either not planetary-mass, younger than its group, or a binary -- the
    three systematics the screen names for every flag.
    """
    l_ceiling = burrows_luminosity_lsun(mass_limit_mj * M_JUP_MSUN, age_gyr)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(l_ceiling) + float(margin_dex)


# --------------------------------------------------------------------------
# Shape classification of a fitted excess
# --------------------------------------------------------------------------

def classify_excess_shape(t_k, tau, cfg: dict) -> str:
    """Name the shape of a fitted (T, tau) excess against the config bands.

    ``companion`` above the grain-survival ceiling (inherited ledger: a fitted
    T > 1800 K is a photosphere); ``debris_disk`` inside the white-dwarf
    debris-disk locus (sublimation-limited, ~800-1700 K, faint); ``ring_band``
    inside the Osmanov band; ``cold_dust`` below it; ``warm_ambiguous`` in the
    gap between the ring band and the debris locus; ``unfit`` when no fit.
    """
    if not (np.isfinite(t_k)):
        return "unfit"
    r = cfg["ring"]
    d = cfg["debris_locus"]
    if t_k > float(cfg["companion_t_min_k"]):
        return "companion"
    in_locus = (float(d["t_min_k"]) <= t_k <= float(d["t_max_k"]))
    if np.isfinite(tau):
        in_locus = in_locus and (float(d["tau_min"]) <= tau <= float(d["tau_max"]))
    if in_locus:
        return "debris_disk"
    if float(r["t_min_k"]) <= t_k <= float(r["t_max_k"]):
        return "ring_band"
    if t_k < float(r["t_min_k"]):
        return "cold_dust"
    return "warm_ambiguous"


__all__ = ["blackbody_colour", "colour_to_temperature", "ring_equilibrium_radius_au",
           "ring_temperature_k", "ring_flux_density_jy", "min_intercept_fraction",
           "spin_down_luminosity_w", "wd_radius_m", "wd_luminosity_w",
           "luminosity_from_solid_angle_w", "burrows_luminosity_lsun", "burrows_teff_k",
           "planetary_cooling_ceiling_log_lsun", "classify_excess_shape",
           "L_SUN_W", "M_JUP_MSUN", "PLANETARY_MASS_LIMIT_MJ"]
