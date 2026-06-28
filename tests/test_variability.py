"""Offline tests for the light-curve variability metric (no network)."""

from __future__ import annotations

import numpy as np

from seti.acquire.variability import _robust_frac_rms


def test_constant_lightcurve_is_not_variable():
    mag = np.full(30, 18.0)
    assert _robust_frac_rms(mag) == 0.0 or np.isclose(_robust_frac_rms(mag), 0.0)


def test_too_few_epochs_returns_nan():
    assert np.isnan(_robust_frac_rms(np.array([18.0, 18.1, 17.9])))


def test_noise_correction_removes_photometric_scatter():
    # A light curve whose scatter is entirely photometric noise should give a
    # near-zero intrinsic fractional RMS once the noise is subtracted.
    rng = np.random.default_rng(0)
    err = 0.05
    mag = 18.0 + rng.normal(0, err, 200)
    magerr = np.full(200, err)
    frac = _robust_frac_rms(mag, magerr)
    assert frac < 0.02  # de-noised scatter is small


def test_genuine_variability_detected():
    # A 0.3-mag sinusoid is clearly variable above a small photometric error.
    t = np.linspace(0, 10, 100)
    mag = 18.0 + 0.3 * np.sin(2 * np.pi * t)
    magerr = np.full(100, 0.02)
    frac = _robust_frac_rms(mag, magerr)
    # 0.3 mag amplitude -> ~0.21 mag RMS -> ~0.2 fractional flux RMS.
    assert frac > 0.05


def test_outliers_are_clipped():
    rng = np.random.default_rng(1)
    mag = 18.0 + rng.normal(0, 0.01, 100)
    mag[::20] = 25.0  # a few catastrophic outliers
    magerr = np.full(100, 0.01)
    frac = _robust_frac_rms(mag, magerr)
    assert frac < 0.05  # outliers clipped, not counted as variability
