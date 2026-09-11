"""Planet scale: how fast can replicators convert a planet's matter in place?

Three independent ceilings, the slowest of which governs:

  exponential   N doublings from one seed: t = tau * log2(M / m0).  For any
                planetary reservoir this is ~100-130 doublings, i.e. days for
                tau = 1000 s.  This is the number everyone quotes and it is
                never the binding one.
  radiative     the conversion dissipates e_rep per kg (whether the energy is
                drawn from sunlight, feedstock chemistry or fission, it ends as
                heat) and a planetary surface can only shed 4 pi R^2 sigma
                (T^4 - T_eq^4) at the temperature the machines survive.
  energy source the energy has to come from somewhere: absorbed sunlight
                (~1.2e17 W for Earth), crustal U/Th, oceanic deuterium.

The paper's point is that the exponential estimate is irrelevant beyond the
biosphere; the crust is a 10^3-10^5 yr project and the bulk planet a
10^6-10^7 yr one, at any replication time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import LOG2, SIGMA_SB


@dataclass(frozen=True)
class Planet:
    mass_kg: float
    radius_m: float
    t_eq_K: float
    absorbed_solar_W: float
    biomass_kg: float
    crust_kg: float
    crust_U_Th_kg: float
    fission_J_per_kg: float
    ocean_kg: float
    deuterium_fraction: float
    dd_fusion_J_per_kg: float

    @property
    def area(self) -> float:
        return 4.0 * math.pi * self.radius_m**2


def seed_mass_kg(radius_um: float, density_gcc: float) -> float:
    """Mass of a spherical device of the given radius and density."""
    r = radius_um * 1e-6
    return (4.0 / 3.0) * math.pi * r**3 * density_gcc * 1e3


def doublings(M_target: float, m_seed: float) -> float:
    return math.log(M_target / m_seed) / LOG2


def t_exponential(M_target: float, m_seed: float, tau_s: float) -> float:
    """Time for one seed to become mass M_target at doubling time tau (s)."""
    return tau_s * doublings(M_target, m_seed)


def radiative_power_limit(planet: Planet, T_max_K: float) -> float:
    """Maximum steady heat a planet's surface can radiate at T_max (W), net of
    the equilibrium radiation it already emits."""
    return planet.area * SIGMA_SB * (T_max_K**4 - planet.t_eq_K**4)


def t_radiative(M_target: float, e_rep: float, planet: Planet, T_max_K: float) -> float:
    """Time to process M_target kg at e_rep J/kg if all heat leaves through the surface at T_max."""
    return M_target * e_rep / radiative_power_limit(planet, T_max_K)


def t_energy_source(M_target: float, e_rep: float, P_source: float) -> float:
    """Time to process M_target kg at e_rep J/kg from a power source P_source (W)."""
    return M_target * e_rep / P_source


def energy_reservoirs_J(planet: Planet) -> dict[str, float]:
    """Non-solar energy available on the planet itself."""
    return {
        "crustal_fission": planet.crust_U_Th_kg * planet.fission_J_per_kg,
        "oceanic_deuterium": planet.ocean_kg * planet.deuterium_fraction * planet.dd_fusion_J_per_kg,
    }


def conversion_table(planet: Planet, m_seed: float, e_rep: float, tau_s: float,
                     T_max_K: float) -> dict[str, dict[str, float]]:
    """All three ceilings for the biosphere, the crust and the bulk planet (seconds).

    The 'governing' entry is the maximum of the three: nothing can go faster than
    the slowest constraint.  Solar power is taken as the baseline energy source;
    the nuclear reservoirs are reported as the energy they could supply, so the
    reader can see that they shorten the crust's timescale but not the bulk's.
    """
    reservoirs = {
        "biosphere": planet.biomass_kg,
        "crust": planet.crust_kg,
        "bulk_planet": planet.mass_kg,
    }
    nuclear = sum(energy_reservoirs_J(planet).values())
    out: dict[str, dict[str, float]] = {}
    for name, M in reservoirs.items():
        t_exp = t_exponential(M, m_seed, tau_s)
        t_rad = t_radiative(M, e_rep, planet, T_max_K)
        t_sol = t_energy_source(M, e_rep, planet.absorbed_solar_W)
        E_needed = M * e_rep
        # If the on-planet nuclear reservoir can supply the whole job, the energy
        # source ceiling is the radiative one (nuclear can be burned as fast as
        # the heat can be shed); otherwise sunlight sets the pace.
        t_src = t_rad if nuclear >= E_needed else t_sol
        out[name] = {
            "mass_kg": M,
            "doublings": doublings(M, m_seed),
            "t_exponential_s": t_exp,
            "t_radiative_s": t_rad,
            "t_solar_s": t_sol,
            "nuclear_sufficient": float(nuclear >= E_needed),
            "t_governing_s": max(t_exp, t_rad, t_src),
        }
    return out


def t_disassembly_s(M: float, R: float, P: float) -> float:
    """Time to lift a planet's mass to infinity at power P (gravitational binding / P)."""
    from .constants import binding_energy_uniform
    return binding_energy_uniform(M, R) / P
