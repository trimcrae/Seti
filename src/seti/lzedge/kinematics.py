"""Two-body kinematics of (in)elastic dark-matter–nucleus scattering, xenon.

Conventions: energies in keV, masses in GeV unless the name says otherwise,
speeds in km/s.  Endothermic (inelastic) scattering with mass splitting
``delta`` = m(chi*) - m(chi) >= 0; ``delta = 0`` is elastic.
"""
from __future__ import annotations

import math

import numpy as np

C_KMS = 299792.458
AMU_GEV = 0.93149410
HBARC_MEV_FM = 197.3269804

_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def trapz(y, x=None, axis=-1):
    """numpy 1.x / 2.x compatible trapezoidal integration."""
    return _TRAPZ(y, x, axis=axis) if x is not None else _TRAPZ(y, axis=axis)

# Natural xenon: (mass number, isotopic fraction).  Fractions sum to 0.9999.
XE_ISOTOPES: tuple[tuple[int, float], ...] = (
    (124, 0.00095), (126, 0.00089), (128, 0.01910), (129, 0.26401), (130, 0.04071),
    (131, 0.21232), (132, 0.26909), (134, 0.10436), (136, 0.08857),
)
XE_A_MEAN = sum(a * f for a, f in XE_ISOTOPES) / sum(f for _, f in XE_ISOTOPES)


def nucleus_mass_gev(A: float) -> float:
    return A * AMU_GEV


def reduced_mass_gev(m1: float, m2: float) -> float:
    return m1 * m2 / (m1 + m2)


def v_min_kms(E_keV, m_chi_gev: float, delta_keV: float = 0.0, A: float = XE_A_MEAN):
    """Minimum lab speed for a recoil of energy ``E_keV`` (vectorised in E)."""
    E = np.asarray(E_keV, dtype=float)
    mN = nucleus_mass_gev(A) * 1e6            # keV
    mu = reduced_mass_gev(m_chi_gev, nucleus_mass_gev(A)) * 1e6
    with np.errstate(divide="ignore", invalid="ignore"):
        v = C_KMS * (mN * E / mu + delta_keV) / np.sqrt(2.0 * mN * E)
    return np.where(E > 0, v, np.inf)


def v_threshold_kms(m_chi_gev: float, delta_keV: float, A: float = XE_A_MEAN) -> float:
    """Smallest speed at which endothermic scattering is possible: sqrt(2 delta / mu)."""
    if delta_keV <= 0:
        return 0.0
    mu = reduced_mass_gev(m_chi_gev, nucleus_mass_gev(A)) * 1e6
    return C_KMS * math.sqrt(2.0 * delta_keV / mu)


def e_star_keV(m_chi_gev: float, delta_keV: float, A: float = XE_A_MEAN) -> float:
    """Recoil energy at which v_min is smallest: mu * delta / m_N (= delta for a heavy WIMP)."""
    mu = reduced_mass_gev(m_chi_gev, nucleus_mass_gev(A))
    return mu * delta_keV / nucleus_mass_gev(A)


def recoil_energy_range_keV(v_kms: float, m_chi_gev: float, delta_keV: float = 0.0,
                            A: float = XE_A_MEAN) -> tuple[float, float] | None:
    """Kinematically allowed [E-, E+] for lab speed ``v_kms``; None below threshold."""
    mN = nucleus_mass_gev(A) * 1e6
    mu = reduced_mass_gev(m_chi_gev, nucleus_mass_gev(A)) * 1e6
    beta2 = (v_kms / C_KMS) ** 2
    disc = 1.0 - 2.0 * delta_keV / (mu * beta2)
    if disc < 0:
        return None
    pref = mu * mu * beta2 / mN
    root = math.sqrt(disc)
    lo = pref * (1.0 - delta_keV / (mu * beta2) - root)
    hi = pref * (1.0 - delta_keV / (mu * beta2) + root)
    return (max(lo, 0.0), hi)


def max_recoil_energy_keV(v_max_kms: float, m_chi_gev: float, delta_keV: float = 0.0,
                          A: float = XE_A_MEAN) -> float:
    r = recoil_energy_range_keV(v_max_kms, m_chi_gev, delta_keV, A)
    return 0.0 if r is None else r[1]


def max_splitting_keV(v_max_kms: float, m_chi_gev: float, A: float = XE_A_MEAN) -> float:
    """Largest delta reachable at any recoil energy for lab speed v_max: mu v^2 / 2."""
    mu = reduced_mass_gev(m_chi_gev, nucleus_mass_gev(A)) * 1e6
    return 0.5 * mu * (v_max_kms / C_KMS) ** 2


def max_splitting_at_energy_keV(v_max_kms: float, E_keV: float, m_chi_gev: float,
                                A: float = XE_A_MEAN) -> float:
    """Largest delta for which a recoil of exactly E_keV is reachable at v_max."""
    mN = nucleus_mass_gev(A) * 1e6
    mu = reduced_mass_gev(m_chi_gev, nucleus_mass_gev(A)) * 1e6
    return (v_max_kms / C_KMS) * math.sqrt(2.0 * mN * E_keV) - mN * E_keV / mu


def momentum_transfer_mev(E_keV, A: float = XE_A_MEAN):
    """q = sqrt(2 m_N E_R) in MeV/c."""
    E = np.asarray(E_keV, dtype=float)
    return np.sqrt(2.0 * nucleus_mass_gev(A) * 1e3 * E * 1e-3)


def helm_form_factor_sq(E_keV, A: float = XE_A_MEAN):
    """Helm nuclear form factor squared, Lewin & Smith (1996) parametrisation."""
    E = np.asarray(E_keV, dtype=float)
    q = momentum_transfer_mev(E, A) / HBARC_MEV_FM        # fm^-1
    c = 1.23 * A ** (1.0 / 3.0) - 0.60
    a, s = 0.52, 0.9
    rn = math.sqrt(c * c + (7.0 / 3.0) * math.pi ** 2 * a * a - 5.0 * s * s)
    x = q * rn
    with np.errstate(divide="ignore", invalid="ignore"):
        j1 = (np.sin(x) - x * np.cos(x)) / (x * x)
        f = 3.0 * j1 / x
    f = np.where(x < 1e-6, 1.0, f)
    return (f * f) * np.exp(-(q * s) ** 2)


def helm_zeros_keV(A: float = XE_A_MEAN, n: int = 3) -> list[float]:
    """Recoil energies of the first ``n`` zeros of the Helm form factor."""
    # zeros of j1(x)/x: x_k = 4.4934, 7.7253, 10.9041, ...
    xk = [4.493409, 7.725252, 10.904122, 14.066194][:n]
    c = 1.23 * A ** (1.0 / 3.0) - 0.60
    a, s = 0.52, 0.9
    rn = math.sqrt(c * c + (7.0 / 3.0) * math.pi ** 2 * a * a - 5.0 * s * s)
    mN_mev = nucleus_mass_gev(A) * 1e3
    out = []
    for x in xk:
        q_mev = x / rn * HBARC_MEV_FM
        out.append(q_mev * q_mev / (2.0 * mN_mev) * 1e3)
    return out


def resolution_sigma_keV(E_keV, sigma_at_ref_keV: float, E_ref_keV: float):
    """sigma_E(E) = sigma_ref * sqrt(E / E_ref): Poisson-like scaling anchored at the event."""
    E = np.asarray(E_keV, dtype=float)
    return sigma_at_ref_keV * np.sqrt(np.clip(E, 0.0, None) / E_ref_keV)
