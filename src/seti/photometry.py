"""Photometric constants and magnitude<->flux conversions.

Band effective wavelengths and Vega zero-point flux densities used to turn
catalogue magnitudes into flux densities (Jy) for SED fitting.  WISE values are
the Vega-system zero points from Jarrett et al. 2011 / the WISE Explanatory
Supplement; 2MASS from Cohen et al. 2003; Gaia DR3 from the Gaia documentation.
"""

from __future__ import annotations

import numpy as np

# Effective wavelength (micron) and Vega zero-point flux density (Jy) per band.
BANDS: dict[str, dict[str, float]] = {
    # Gaia DR3 (AB-ish, but treated via its own zero points)
    "G":  {"lambda_um": 0.6730, "zp_jy": 3228.75},
    "BP": {"lambda_um": 0.5320, "zp_jy": 3552.01},
    "RP": {"lambda_um": 0.7970, "zp_jy": 2554.95},
    # 2MASS
    "J":  {"lambda_um": 1.235,  "zp_jy": 1594.0},
    "H":  {"lambda_um": 1.662,  "zp_jy": 1024.0},
    "Ks": {"lambda_um": 2.159,  "zp_jy": 666.7},
    # WISE / CatWISE2020
    "W1": {"lambda_um": 3.3526, "zp_jy": 309.540},
    "W2": {"lambda_um": 4.6028, "zp_jy": 171.787},
    "W3": {"lambda_um": 11.5608, "zp_jy": 31.674},
    "W4": {"lambda_um": 22.0883, "zp_jy": 8.363},
}

# Physical constants (SI)
H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
K_BOLTZ = 1.380649e-23


def band_freq_hz(band: str) -> float:
    """Effective frequency (Hz) of a band."""
    lam_m = BANDS[band]["lambda_um"] * 1e-6
    return C_LIGHT / lam_m


def mag_to_flux_jy(mag: np.ndarray | float, band: str) -> np.ndarray | float:
    """Vega magnitude -> flux density in Jy."""
    return BANDS[band]["zp_jy"] * 10.0 ** (-0.4 * np.asarray(mag, dtype=float))


def flux_jy_to_mag(flux_jy: np.ndarray | float, band: str) -> np.ndarray | float:
    """Flux density in Jy -> Vega magnitude."""
    return -2.5 * np.log10(np.asarray(flux_jy, dtype=float) / BANDS[band]["zp_jy"])


def mag_err_to_flux_err_jy(mag: np.ndarray | float, mag_err: np.ndarray | float,
                           band: str) -> np.ndarray | float:
    """Propagate a magnitude error to a flux-density error (Jy)."""
    flux = mag_to_flux_jy(mag, band)
    return 0.4 * np.log(10.0) * flux * np.asarray(mag_err, dtype=float)


def planck_bnu(temp_k: np.ndarray | float, freq_hz: float) -> np.ndarray | float:
    """Planck function B_nu(T) at frequency ``freq_hz`` (SI, W m^-2 Hz^-1 sr^-1)."""
    t = np.asarray(temp_k, dtype=float)
    x = H_PLANCK * freq_hz / (K_BOLTZ * t)
    # np.expm1 keeps precision in the Rayleigh-Jeans (small x) regime.
    return (2.0 * H_PLANCK * freq_hz**3 / C_LIGHT**2) / np.expm1(x)
