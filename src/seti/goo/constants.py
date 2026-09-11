"""Physical constants (SI) and unit conversions used across seti.goo."""
from __future__ import annotations

import math

G = 6.674e-11            # m^3 kg^-1 s^-2
C = 2.998e8              # m/s
SIGMA_SB = 5.670e-8      # W m^-2 K^-4
L_SUN = 3.828e26         # W
M_SUN = 1.989e30         # kg
AU = 1.496e11            # m
PC = 3.086e16            # m
KPC = 1e3 * PC
MPC = 1e6 * PC
YR = 3.156e7             # s
MYR = 1e6 * YR
GYR = 1e9 * YR
M_EARTH = 5.972e24
R_EARTH = 6.371e6
KM = 1e3

LOG2 = math.log(2.0)


def v_circ(r_m: float, M: float = M_SUN) -> float:
    """Circular orbital speed at radius r about mass M (m/s)."""
    return math.sqrt(G * M / r_m)


def v_esc(r_m: float, M: float = M_SUN) -> float:
    """Escape speed from radius r about mass M (m/s)."""
    return math.sqrt(2.0 * G * M / r_m)


def t_eq_thin(r_m: float, L: float = L_SUN) -> float:
    """Equilibrium temperature of a thin, isotropically radiating sheet at r (K).

    A flat absorber facing the star re-radiates from both faces:
    T = (L / (16 pi sigma r^2))^(1/4) = 278 K (r/AU)^(-1/2) for the Sun.
    """
    return (L / (16.0 * math.pi * SIGMA_SB * r_m**2)) ** 0.25


def binding_energy_uniform(M: float, R: float) -> float:
    """Gravitational binding energy of a uniform sphere (J): 3GM^2/5R."""
    return 3.0 * G * M**2 / (5.0 * R)
