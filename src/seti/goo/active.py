"""Active branch: replicators that travel on purpose.

This is the classical von Neumann-probe front (Hart 1975; Tipler 1980; Newman &
Sagan 1981; Bjork 2007; Cotta & Morales 2009; Wiley 2011; Nicholson & Forgan
2013; Carroll-Nellenback et al. 2019).  A goo that inherits interstellar
capability from the probe it was, or from the civilisation whose shipyards it
ate, spreads at the front speed

    v_front = d / (d / v_ship + tau_build)

for nearest-neighbour hops of length d, and fills a galaxy of radius R in
R / v_front.  Everything published lands in 10^5-10^8 yr for the Milky Way; the
disagreement is only about v_ship and tau_build.

Beyond the galaxy the reach is bounded by the expansion of the universe: a
front launched today at speed v reaches a comoving distance
v * integral_{t0}^{inf} dt / a(t), which for v = c is the cosmic event horizon
(~ 16.5 Gly comoving in Planck-2018 LambdaCDM).  The number of galaxies that a
front can ever touch is that volume times the galaxy density.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

from .constants import KPC, MPC, PC, YR, C


def front_speed_ms(v_ship_ms: float, tau_build_s: float, d_hop_m: float) -> float:
    return d_hop_m / (d_hop_m / v_ship_ms + tau_build_s)


def galaxy_fill_time_s(R_kpc: float, v_ship_c: float, tau_build_yr: float, d_hop_pc: float) -> float:
    vf = front_speed_ms(v_ship_c * C, tau_build_yr * YR, d_hop_pc * PC)
    return R_kpc * KPC / vf


def fisher_kpp_speed_ms(D_m2_s: float, r_per_s: float) -> float:
    """Fisher-KPP front speed 2 sqrt(D r) for a diffusing, logistically growing population."""
    return 2.0 * math.sqrt(D_m2_s * r_per_s)


def hubble_time_s(H0_kms_Mpc: float) -> float:
    return MPC / (H0_kms_Mpc * 1e3)


def comoving_reach_Mpc(v_over_c: float, H0: float, Om: float, launch_a: float = 1.0) -> float:
    """Comoving distance reachable from now (scale factor launch_a) at constant peculiar speed v.

    d_c = v * int_{a0}^{inf} da / (a^2 H(a)),  H(a) = H0 sqrt(Om a^-3 + OL).
    """
    OL = 1.0 - Om
    tH = hubble_time_s(H0)

    def integrand(a: float) -> float:
        return 1.0 / (a**2 * math.sqrt(Om * a**-3 + OL))

    val, _ = quad(integrand, launch_a, np.inf, limit=200)
    return v_over_c * C * tH * val / MPC


def galaxies_within_reach(v_over_c: float, H0: float, Om: float, n_gal_Mpc3: float) -> float:
    d = comoving_reach_Mpc(v_over_c, H0, Om)
    return (4.0 / 3.0) * math.pi * d**3 * n_gal_Mpc3


def reach_table(v_list: list[float], H0: float, Om: float, n_gal_Mpc3: float) -> list[dict[str, float]]:
    rows = []
    for v in v_list:
        d = comoving_reach_Mpc(v, H0, Om)
        rows.append({"v_c": v, "reach_Mpc": d, "reach_Gly": d * 3.2616e-3,
                     "n_galaxies": (4.0 / 3.0) * math.pi * d**3 * n_gal_Mpc3})
    return rows
