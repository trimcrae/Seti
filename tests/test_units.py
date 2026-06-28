"""Unit tests for the physics and statistics building blocks."""

import numpy as np

from seti.photometry import (
    flux_jy_to_mag,
    mag_to_flux_jy,
    planck_bnu,
)
from seti.stats.upper_limit import occurrence_upper_limit, poisson_upper_limit


def test_mag_flux_roundtrip():
    for band in ("J", "W1", "W2"):
        for mag in (8.0, 12.5, 16.3):
            assert np.isclose(flux_jy_to_mag(mag_to_flux_jy(mag, band), band), mag)


def test_planck_rayleigh_jeans_scaling():
    # Deep in the Rayleigh-Jeans regime (h*nu << k*T) B_nu -> linear in T, so
    # doubling T doubles B_nu.  Use a low frequency / high T to approach the limit.
    nu = 1.0e13  # ~30 micron, well into RJ for these temperatures
    t1, t2 = 10000.0, 20000.0
    ratio = planck_bnu(t2, nu) / planck_bnu(t1, nu)
    assert 1.95 < ratio < 2.05


def test_poisson_upper_limit_zero_counts():
    # Classic 95% Poisson upper limit for zero events is ~3.0.
    assert np.isclose(poisson_upper_limit(0, 0.95), 3.0, atol=0.05)


def test_occurrence_limit_null():
    lim = occurrence_upper_limit(k=0, n_eff=1000, confidence=0.95)
    assert lim.f_point == 0.0
    assert np.isclose(lim.f_upper, 3.0 / 1000, atol=2e-4)


def test_occurrence_limit_completeness_inflation():
    full = occurrence_upper_limit(k=0, n_eff=1000, completeness=1.0)
    half = occurrence_upper_limit(k=0, n_eff=1000, completeness=0.5)
    assert np.isclose(half.f_upper, 2 * full.f_upper, rtol=1e-6)
