"""Passive interstellar dispersal: the only road out of a system for a replicator
that was never designed to travel.

The physics is that of interstellar dust, because that is what sub-micron
machines are once they are loose in space.

1. Blowout.  Around the Sun beta = 0.57 Q_pr / (rho s), so devices below
   ~0.6 um (rho = 2 g/cc) have beta > 0.5 and are unbound the moment they are
   released from an orbiting body; they leave on hyperbolic orbits with
   v_inf = v_circ sqrt(2 beta - 1).  A goo that eats an asteroid belt into
   micron-scale machines therefore ejects a fraction of its own mass at
   10-30 km/s, for free, and forever.

2. Arrival.  The same radiation pressure that ejected them repels them from the
   next star.  An incoming device of beta > 1 can only penetrate to
   r_min = 2 (beta - 1) G M / v_inf^2, so with the 26 km/s interstellar inflow
   speed only beta < 1 + (v_inf / v_esc(r))^2 ~ 1.4 reaches 1 AU.  The size
   window that both leaves and arrives is narrow: roughly 0.2-0.6 um at
   rho = 2 g/cc.  (Sub-0.1 um grains are additionally filtered by stellar
   magnetic fields at the astropause; Landgraf 2000.)

3. Survival.  Interstellar grains are destroyed by supernova shocks on
   ~5e8 yr (Jones, Tielens & Hollenbach 1996); a functioning machine is a much
   more fragile thing than a grain, so the functional e-folding time is scanned
   from 1e6 to 1e9 yr.

4. Spread.  Released at ~stellar speeds the population joins the disc's velocity
   field: epicyclic phase mixing spreads it over a ~1.6 kpc-wide annulus and
   differential rotation smears it around the annulus in ~ (R / Delta R) rotation
   periods ~ 1-2 Gyr.  Radial migration takes it across the disc on ~10 Gyr.

5. Seeding.  With N intact devices in the annulus volume V, an Earth-sized
   planet sweeps them up at n v sigma with sigma = pi R^2 (1 + v_esc^2 / v^2).
   This is the annulus-AVERAGED rate per planet.  It does not wait for phase
   mixing: the total landing rate over a uniform stellar field, N eta n_* v sigma,
   is independent of the volume the cloud currently occupies (survival.py), so
   the mixing time only decides when the rate is the same at every planet, not
   when landings begin.  Cumulative arrivals per planet are therefore
   rate0 * tau (1 - e^{-t/tau}) with no gate at the fill time.

The answer these steps give is that a released asteroid belt's worth of
micron machines delivers ~10^2 intact devices per year to every rocky planet in
the annulus, before survival losses; survival sets the reach, not the count.
"""
from __future__ import annotations

import math

from .constants import AU, KM, KPC, M_SUN, YR, G, v_circ, v_esc


def blowout_radius_um(density_gcc: float, q_pr: float = 1.0) -> float:
    """Device radius at which beta = 0.5 (larger devices stay bound)."""
    return 0.57 * q_pr / (density_gcc * 0.5)


def v_infinity_ms(beta: float, r_release_AU: float) -> float:
    """Hyperbolic excess speed of a device released from a circular orbit at r with beta > 0.5."""
    if beta <= 0.5:
        return 0.0
    return v_circ(r_release_AU * AU) * math.sqrt(2.0 * beta - 1.0)


def beta_max_to_reach(v_inf_ms: float, r_target_AU: float, M: float = M_SUN) -> float:
    """Largest beta that an incoming device with speed v_inf can have and still reach r_target."""
    return 1.0 + (v_inf_ms / v_esc(r_target_AU * AU, M)) ** 2


def deliverable_size_window_um(density_gcc: float, v_inf_ms: float, r_target_AU: float,
                               q_pr: float = 1.0) -> tuple[float, float]:
    """Radii [s_min, s_max] that (a) are blown out of the source system (beta > 0.5) and
    (b) can reach r_target of a Sun-like target (beta < beta_max)."""
    b_max = beta_max_to_reach(v_inf_ms, r_target_AU)
    s_min = 0.57 * q_pr / (density_gcc * b_max)
    s_max = blowout_radius_um(density_gcc, q_pr)
    return s_min, s_max


def capture_cross_section_m2(R_planet_m: float, v_ms: float, M_planet_kg: float) -> float:
    """Gravitationally focused geometric cross-section of a planet for particles at speed v."""
    ve = math.sqrt(2.0 * G * M_planet_kg / R_planet_m)
    return math.pi * R_planet_m**2 * (1.0 + (ve / v_ms) ** 2)


def annulus_volume_m3(R_kpc: float, width_kpc: float, scale_height_kpc: float) -> float:
    return 2.0 * math.pi * (R_kpc * KPC) * (width_kpc * KPC) * (2.0 * scale_height_kpc * KPC)


def annulus_fill_time_s(rotation_period_Myr: float, R_kpc: float, width_kpc: float) -> float:
    """Differential rotation smears a patch of radial extent Delta R around the annulus in
    ~ (R / Delta R) rotation periods (flat rotation curve, d Omega / d ln R = -Omega)."""
    return rotation_period_Myr * 1e6 * YR * (R_kpc / width_kpc)


def seeds_per_year(N_devices: float, V_m3: float, v_ms: float, R_planet_m: float,
                   M_planet_kg: float) -> float:
    n = N_devices / V_m3
    return n * v_ms * capture_cross_section_m2(R_planet_m, v_ms, M_planet_kg) * YR


def survival_fraction(t_s: float, t_survive_s: float) -> float:
    return math.exp(-t_s / t_survive_s)


def passive_seeding(M_release_kg: float, radius_um: float, density_gcc: float, f_blown: float,
                    galaxy: dict, R_planet_m: float, M_planet_kg: float, v_ms: float,
                    t_survive_s: float, t_elapsed_s: float) -> dict[str, float]:
    """Expected intact device arrivals per year on one Earth-like planet in the annulus,
    t_elapsed after release, and the cumulative number by then."""
    from .planet import seed_mass_kg
    m = seed_mass_kg(radius_um, density_gcc)
    N0 = f_blown * M_release_kg / m
    V = annulus_volume_m3(galaxy["R_sun_kpc"], galaxy["annulus_width_kpc"], galaxy["scale_height_kpc"])
    rate0 = seeds_per_year(N0, V, v_ms, R_planet_m, M_planet_kg)
    S = survival_fraction(t_elapsed_s, t_survive_s)
    # cumulative arrivals per planet, annulus-averaged: the integral of rate0 * exp(-t/ts)
    # from release to t_elapsed.  No gate at the fill time: the landing rate summed over
    # the planets inside the cloud is volume-independent (see module docstring and survival.py).
    t_fill = annulus_fill_time_s(galaxy["rotation_period_Myr"], galaxy["R_sun_kpc"], galaxy["annulus_width_kpc"])
    ts = t_survive_s
    cum = rate0 * (ts / YR) * (1.0 - math.exp(-t_elapsed_s / ts))
    return {
        "device_mass_kg": m,
        "N_released": N0,
        "n_per_m3_unsurvived": N0 / V,
        "arrivals_per_yr_unsurvived": rate0,
        "survival_fraction": S,
        "arrivals_per_yr": rate0 * S,
        "t_fill_annulus_s": t_fill,
        "cumulative_arrivals": cum,
    }


def blown_mass_fraction(size_window_um: tuple[float, float], s_lo_um: float, s_hi_um: float,
                        slope: float = -3.5) -> float:
    """Fraction of a collisional (power-law dn/ds ~ s^slope) mass distribution between
    s_lo and s_hi that lies inside the deliverable window.  With slope -3.5 the mass is
    dominated by the largest bodies, so the deliverable fraction is ~ sqrt(s_win / s_hi)."""
    a, b = size_window_um
    a = max(a, s_lo_um)
    b = min(b, s_hi_um)
    if b <= a:
        return 0.0
    p = slope + 4.0  # mass-weighted exponent: m dn ~ s^(slope+3) ds -> integral s^(slope+4)
    num = b**p - a**p
    den = s_hi_um**p - s_lo_um**p
    return num / den


def interstellar_hit_rate_per_planet(n_per_m3: float, v_ms: float, R_planet_m: float,
                                     M_planet_kg: float) -> float:
    return n_per_m3 * v_ms * capture_cross_section_m2(R_planet_m, v_ms, M_planet_kg) * YR


def galactic_diffusion_reach_kpc(sigma_v_kms: float, t_s: float, mean_free_time_s: float) -> float:
    """Random-walk reach sqrt(2 D t) with D = sigma^2 t_mfp for a population scattered on the
    disc's dynamical heating time (order of magnitude only)."""
    D = (sigma_v_kms * KM) ** 2 * mean_free_time_s
    return math.sqrt(2.0 * D * t_s) / KPC
