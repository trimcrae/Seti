"""Functional survival in interstellar space, scanned from kiloyears to infinity.

The earlier passive-branch treatment (passive.py, contact.py) counted arrivals
only after the carriers had phase-mixed around the annulus, which made a
survival time shorter than the mixing time (~1 Gyr) look fatal.  That is wrong
for a cloud expanding into a uniform field of stars: the landing rate summed
over every target inside the cloud is

    dN_land/dt = N_c(t) * eta * n_* * v_rel * sigma,

independent of the cloud's volume, because carrier density falls exactly as
the number of enclosed targets rises.  Survival therefore does not decide
whether landings happen; it decides WHERE (the reach R = v_inf * tau) and HOW
FAST a chain of seedings can move (a front at ~ v_inf once tau exceeds the hop
time to the nearest stars).

For a source releasing N_c carriers at speed v_inf with functional e-folding
time tau, over a window W:

  reach          R(tau) = v_inf * tau           (the e-folding distance)
  targets        planets within R: eta * n_* * (4/3) pi R^3, capped by the disc
                 thickness (2h) and then by the annulus
  R0(tau)        expected landings per source = N_c eta n_* v_rel sigma * tau (1 - e^{-W/tau})
  t_first        time to the first landing anywhere = max(d_nearest / v_inf, 1 / rate_0)
  front          if R0 > 1 and tau > t_first: a chain of seedings advances at
                 v_front ~ v_inf / (1 + (t_first + t_conv) v_inf / d_hop); else none
  t_galaxy       R_gal / v_front

Carrier classes (config `survival.carriers`): sub-micron devices blown out by
radiation pressure; micron dust-sized devices; devices riding inside
100-m-class ejected planetesimals; rocks impact-ejected from a planet.  The
survival physics is not modelled -- it is the scanned variable -- but the
context numbers (cosmic-ray hits per device per year; the ISM grain lifetime)
are computed so the reader can place a design on the axis.
"""
from __future__ import annotations

import math

import numpy as np

from .constants import KM, M_EARTH, PC, R_EARTH, YR, G


def sigma_strict_m2(v_ms: float) -> float:
    ve = math.sqrt(2.0 * G * M_EARTH / R_EARTH)
    return math.pi * R_EARTH**2 * (1.0 + (ve / v_ms) ** 2)


def landing_rate_per_yr(N_c: float, eta: float, n_star_pc3: float, v_ms: float) -> float:
    """Total strict landings per year from N_c functional carriers in a uniform field."""
    n = n_star_pc3 / PC**3
    return N_c * eta * n * v_ms * sigma_strict_m2(v_ms) * YR


def R0_of_tau(rate0_per_yr: float, tau_yr: float, W_yr: float) -> float:
    if math.isinf(tau_yr):
        return rate0_per_yr * W_yr
    return rate0_per_yr * tau_yr * (1.0 - math.exp(-W_yr / tau_yr))


def reach_pc(v_inf_ms: float, tau_yr: float) -> float:
    return v_inf_ms * tau_yr * YR / PC


def planets_within(R_pc: float, eta: float, n_star_pc3: float, h_pc: float, R_annulus_pc: float,
                   n_planets_annulus: float) -> float:
    """Planets inside the reach: a sphere until R > h, then a slab of thickness 2h, capped by the annulus."""
    if R_pc <= h_pc:
        vol = (4.0 / 3.0) * math.pi * R_pc**3
    else:
        vol = math.pi * R_pc**2 * 2.0 * h_pc
    return min(eta * n_star_pc3 * vol, n_planets_annulus)


def cosmic_ray_hits_per_yr(radius_um: float, flux_per_m2_s: float = 1.0e4) -> float:
    """Galactic-cosmic-ray primaries (E > ~100 MeV, all-sky) intercepted by a sphere of the given
    radius: ~1e4 m^-2 s^-1 is the order of magnitude of the integrated GCR nucleon flux outside
    the heliosphere."""
    return math.pi * (radius_um * 1e-6) ** 2 * flux_per_m2_s * YR


def scan(carrier: dict, cfg: dict, taus_yr: np.ndarray) -> dict:
    s = cfg["survival"]
    g = cfg["passive"]["galaxy"]
    c = cfg["contact"]
    n_star = s["n_star_pc3"]
    eta = c["n_earthlike_per_star"]
    W = cfg["inference"]["window_Gyr"] * 1e9
    v_inf = carrier["v_inf_kms"] * KM
    v_rel = carrier.get("v_rel_kms", cfg["passive"]["inflow_speed_kms"]) * KM
    N_c = carrier["carriers_per_source"]
    rate0 = landing_rate_per_yr(N_c, eta, n_star, v_rel)
    d_nearest_pc = (1.0 / n_star) ** (1.0 / 3.0)
    t_hop = d_nearest_pc * PC / v_inf / YR
    t_first = max(t_hop, 1.0 / rate0) if rate0 > 0 else float("inf")
    t_conv = carrier.get("t_conversion_yr", 100.0)
    R_gal_pc = cfg["active"]["R_galaxy_kpc"] * 1e3
    h_pc = g["scale_height_kpc"] * 1e3
    R_ann_pc = g["annulus_width_kpc"] * 1e3 / 2.0
    N_pl_ann = c["n_systems_annulus"] * eta
    rows = []
    for tau in taus_yr:
        R0 = R0_of_tau(rate0, tau, W)
        reach = reach_pc(v_inf, tau) if not math.isinf(tau) else float("inf")
        n_pl = planets_within(min(reach, 1e9), eta, n_star, h_pc, R_ann_pc, N_pl_ann)
        front = (R0 > 1.0) and (tau > t_first)
        v_front = v_inf / (1.0 + (t_first + t_conv) * YR * v_inf / (d_nearest_pc * PC)) if front else 0.0
        t_gal = (R_gal_pc * PC / v_front / YR) if v_front > 0 else float("inf")
        rows.append({"tau_yr": float(tau), "R0": R0, "reach_pc": reach, "planets_within_reach": n_pl,
                     "front": front, "v_front_kms": v_front / KM, "t_galaxy_yr": t_gal})
    return {"carrier": carrier["name"], "N_c": N_c, "v_inf_kms": carrier["v_inf_kms"],
            "rate0_per_yr": rate0, "d_nearest_pc": d_nearest_pc, "t_hop_yr": t_hop, "t_first_yr": t_first,
            "t_conversion_yr": t_conv,
            "cosmic_ray_hits_per_yr": cosmic_ray_hits_per_yr(carrier.get("radius_um", 0.5)),
            "tau_min_for_front_yr": t_first, "rows": rows}


def tau_grid(lo: float = 1e3, hi: float = 1e10, n: int = 71) -> np.ndarray:
    return np.concatenate([np.logspace(math.log10(lo), math.log10(hi), n), [float("inf")]])
