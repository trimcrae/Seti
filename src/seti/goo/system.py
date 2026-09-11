"""Planetary-system scale: converting the small bodies and the planets, and the
swarm that is left behind.

Timescales.  Inside a belt the front is a branching process: each converted
body launches b offspring packets, a hop covers the mean inter-body spacing at
the hop speed, and a body saturates after ~100 local doublings.  With b ~ 100
and 10^6 bodies the belt is eaten in a handful of generations, so the belt
timescale is transport-limited (years to centuries), not replication-limited.

Planets are different: their mass can only leave at the rate the surface can
shed the heat of processing it (the radiative ceiling of planet.py) or, if the
swarm collects starlight to lift it, at the rate the captured luminosity
supplies the binding energy.

Residue.  Converted mass becomes a swarm.  Its covering fraction is
min(1, M / (sigma_areal * 4 pi r^2)); for micron-thick devices even the asteroid
belt covers the star completely at 2.7 AU, so a system-scale goo is a Dyson-sphere-
scale waste-heat source at the temperature of its feedstock's orbit.  When the
replication stops (feedstock gone, error catastrophe, competition) the swarm
grinds itself down on the collisional timescale ~ P_orb / f (Lacki 2025) and the
fragments leave by Poynting-Robertson drag and radiation pressure: the residue is
visible for 10^3-10^5 yr, not for the age of the star.
"""
from __future__ import annotations

import math

from .constants import AU, L_SUN, M_SUN, YR, binding_energy_uniform, t_eq_thin, v_circ


def belt_spacing_m(n_bodies: float, r_AU: float, width_AU: float = 1.0, height_AU: float = 0.5) -> float:
    """Mean spacing between n bodies in a torus of radius r, radial width and full height."""
    vol = 2.0 * math.pi * (r_AU * AU) * (width_AU * AU) * (height_AU * AU)
    return (vol / n_bodies) ** (1.0 / 3.0)


def belt_conversion_time_s(n_bodies: float, r_AU: float, hop_speed_ms: float,
                           branching: float, local_doublings: float, tau_s: float) -> float:
    """Generations x (local saturation + hop transit), plus one orbital phasing time per generation."""
    n_gen = max(1.0, math.log(n_bodies) / math.log(branching))
    t_local = local_doublings * tau_s
    t_hop = belt_spacing_m(n_bodies, r_AU) / hop_speed_ms
    # A hop that changes semi-major axis waits of order an orbital period for phasing.
    P_orb = 2.0 * math.pi * (r_AU * AU) / v_circ(r_AU * AU)
    return n_gen * (t_local + t_hop + P_orb)


def covering_fraction(M_kg: float, sigma_areal_kg_m2: float, r_AU: float) -> float:
    """Fraction of the star covered by mass M spread at areal density sigma on a shell at r."""
    return min(1.0, M_kg / (sigma_areal_kg_m2 * 4.0 * math.pi * (r_AU * AU) ** 2))


def swarm_temperature_K(r_AU: float, L: float = L_SUN) -> float:
    return t_eq_thin(r_AU * AU, L)


def captured_power_W(f_cov: float, L: float = L_SUN) -> float:
    return f_cov * L


def planet_disassembly_time_s(M: float, R: float, P_available: float) -> float:
    return binding_energy_uniform(M, R) / P_available


def collisional_time_s(r_AU: float, f_cov: float, M: float = M_SUN) -> float:
    """Lacki (2025): collisional time ~ orbital period / covering fraction."""
    P_orb = 2.0 * math.pi * (r_AU * AU) / v_circ(r_AU * AU, M)
    return P_orb / max(f_cov, 1e-12)


def beta_solar(radius_um: float, density_gcc: float, q_pr: float = 1.0) -> float:
    """Radiation-pressure to gravity ratio for a sphere around the Sun (Burns et al. 1979):
    beta = 0.57 Q_pr / (rho[g/cc] s[um])."""
    return 0.57 * q_pr / (density_gcc * radius_um)


def pr_drag_time_s(r_AU: float, beta: float) -> float:
    """Poynting-Robertson decay time from a circular orbit at r: 400 yr (r/AU)^2 / beta."""
    return 400.0 * YR * r_AU**2 / max(beta, 1e-12)


def covering_fraction_for_size(M_kg: float, radius_um: float, density_gcc: float, r_AU: float) -> float:
    """Covering fraction of mass M ground into bodies of radius s: each body presents
    pi s^2 for a mass (4/3) pi s^3 rho, so sigma_areal = (4/3) rho s."""
    sigma = (4.0 / 3.0) * density_gcc * 1e3 * radius_um * 1e-6
    return covering_fraction(M_kg, sigma, r_AU)


def residue_lifetime_s(r_AU: float, f_cov: float, radius_um: float, density_gcc: float) -> dict[str, float]:
    """How long a dead swarm stays a waste-heat source.

    Devices of radius s: if beta > 0.5 they are blown out on an orbital time; else
    they spiral in on the PR time.  A dense swarm (f_cov ~ 1) grinds itself to
    blowout sizes on the collisional time, after which the whole mass leaves on the
    orbital time.  The lifetime is min(PR time of the device, collisional time +
    orbital time), bounded below by one orbit.

    The corollary the paper leans on: a residue detectable as an infrared excess
    needs f_cov above some floor f_min, and its collisional time is then at most
    P_orb / f_min, so *a detectable dead residue is a short-lived one*.
    """
    b = beta_solar(radius_um, density_gcc)
    P_orb = 2.0 * math.pi * (r_AU * AU) / v_circ(r_AU * AU)
    t_pr = pr_drag_time_s(r_AU, b) if b < 0.5 else P_orb
    t_coll = collisional_time_s(r_AU, f_cov)
    return {
        "beta": b,
        "f_cov": f_cov,
        "t_orbit_s": P_orb,
        "t_pr_s": t_pr,
        "t_collisional_s": t_coll,
        "t_residue_s": max(P_orb, min(t_pr, t_coll + P_orb)),
    }


def max_visible_residue_s(r_AU: float, f_min_detectable: float) -> float:
    """Longest a dead swarm can stay above the detection floor f_min: P_orb / f_min."""
    return collisional_time_s(r_AU, f_min_detectable)


def rate_bound_per_star_per_yr(n_stars: float, t_residue_s: float, n_detected: int = 0, cl: float = 0.95) -> float:
    """Upper bound on the rate of goo events per star per year from a null search over
    n_stars, each visible for t_residue after an event: N_exp = R n_stars t_res.

    For n_detected = 0 the 95 % Poisson bound is 3.0 expected events.
    """
    from scipy.stats import chi2
    n_up = 0.5 * chi2.ppf(cl, 2 * (n_detected + 1))
    return n_up / (n_stars * (t_residue_s / YR))
