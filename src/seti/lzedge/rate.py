"""Differential and window-integrated recoil rates in natural xenon.

Spin-independent-like coupling with the Helm form factor, summed over the
xenon isotopes, for elastic or endothermic (``delta > 0``) scattering:

    dR/dE = N_T (rho/m_chi) sigma_n A² F²(E) [m_A/(2 mu_n²)] c² g(v_min(E)),

with g the inverse-speed halo integral from ``halo.py``.  Output in
events / (tonne · year · keV).  Absolute normalisation matters only for the
cross-experiment counts; the timing and energy likelihoods are ratios.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import (
    AMU_GEV,
    XE_ISOTOPES,
    helm_form_factor_sq,
    nucleus_mass_gev,
    reduced_mass_gev,
    trapz,
    v_min_kms,
)

AMU_G = 1.66053907e-24
C_CMS = 2.99792458e10
SEC_PER_YEAR = 3.15576e7
M_NUCLEON_GEV = AMU_GEV


def n_targets_per_tonne(A_mean: float) -> float:
    return 1.0e6 / (A_mean * AMU_G)


@dataclass
class Efficiency:
    """Piecewise-linear efficiency table; zero outside the tabulated range."""
    E_keV: np.ndarray
    eps: np.ndarray

    @classmethod
    def flat(cls, E_lo: float, E_hi: float, value: float = 1.0) -> Efficiency:
        return cls(np.array([E_lo, E_hi]), np.array([value, value]))

    @classmethod
    def from_table(cls, rows) -> Efficiency:
        arr = np.asarray(rows, dtype=float)
        return cls(arr[:, 0], arr[:, 1])

    def __call__(self, E):
        E = np.asarray(E, dtype=float)
        return np.interp(E, self.E_keV, self.eps, left=0.0, right=0.0)

    @property
    def E_lo(self) -> float:
        return float(self.E_keV[0])

    @property
    def E_hi(self) -> float:
        return float(self.E_keV[-1])


def spectrum(E_keV, eta_fn, m_chi_gev: float, delta_keV: float = 0.0, sigma_n_cm2: float = 1e-45,
             rho_gev_cm3: float = 0.3, isotopes=XE_ISOTOPES, form_factor: bool = True) -> np.ndarray:
    """dR/dE in events/(tonne yr keV) for natural xenon."""
    E = np.asarray(E_keV, dtype=float)
    a_mean = sum(a * f for a, f in isotopes) / sum(f for _, f in isotopes)
    n_tot = n_targets_per_tonne(a_mean)
    mu_n = reduced_mass_gev(m_chi_gev, M_NUCLEON_GEV)
    total = np.zeros_like(E)
    for A, frac in isotopes:
        mA = nucleus_mass_gev(A)
        vmin = v_min_kms(E, m_chi_gev, delta_keV, A)
        g = np.asarray(eta_fn(vmin), dtype=float) * 1e-5          # s/km -> s/cm
        ff = helm_form_factor_sq(E, A) if form_factor else np.ones_like(E)
        pref = (n_tot * frac) * (rho_gev_cm3 / m_chi_gev) * sigma_n_cm2 * A * A \
            * (mA / (2.0 * mu_n * mu_n)) * 1e-6 * C_CMS * C_CMS  # 1/(keV s) per (s/cm of g)
        total += pref * ff * g
    return total * SEC_PER_YEAR


def window_rate(eta_fn, m_chi_gev: float, delta_keV: float, efficiency: Efficiency,
                sigma_n_cm2: float = 1e-45, rho_gev_cm3: float = 0.3, dE: float = 0.5,
                isotopes=XE_ISOTOPES) -> float:
    """Efficiency-weighted rate in the analysis window, events/(tonne yr)."""
    E = np.arange(max(efficiency.E_lo, dE), efficiency.E_hi + dE, dE)
    if E.size < 2:
        return 0.0
    dr = spectrum(E, eta_fn, m_chi_gev, delta_keV, sigma_n_cm2, rho_gev_cm3, isotopes)
    return float(trapz(efficiency(E) * dr, E))


def smeared_spectrum(E_obs, eta_fn, m_chi_gev: float, delta_keV: float, efficiency: Efficiency,
                     sigma_fn, sigma_n_cm2: float = 1e-45, rho_gev_cm3: float = 0.3, dE: float = 0.5,
                     isotopes=XE_ISOTOPES) -> np.ndarray:
    """Rate density in *observed* energy: ∫ dE eps(E) dR/dE N(E_obs; E, sigma(E))."""
    E = np.arange(max(efficiency.E_lo, dE), efficiency.E_hi + dE, dE)
    dr = efficiency(E) * spectrum(E, eta_fn, m_chi_gev, delta_keV, sigma_n_cm2, rho_gev_cm3, isotopes)
    sig = np.asarray(sigma_fn(E), dtype=float)
    Eo = np.atleast_1d(np.asarray(E_obs, dtype=float))
    kern = np.exp(-0.5 * ((Eo[:, None] - E[None, :]) / sig[None, :]) ** 2) / (np.sqrt(2 * np.pi) * sig[None, :])
    return trapz(kern * dr[None, :], E, axis=1)
