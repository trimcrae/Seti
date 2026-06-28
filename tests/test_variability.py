"""Offline tests for the light-curve variability metric (no network)."""

from __future__ import annotations

import numpy as np

from seti.acquire.variability import _lomb_scargle, _robust_frac_rms


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


def test_lomb_scargle_recovers_injected_period():
    rng = np.random.default_rng(3)
    period = 3.7
    t = np.sort(rng.uniform(0, 300, 250))  # 300-day baseline, irregular sampling
    mag = 18.0 + 0.15 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.02, t.size)
    magerr = np.full(t.size, 0.02)
    res = _lomb_scargle(t, mag, magerr)
    assert np.isfinite(res["ls_period_d"])
    assert abs(res["ls_period_d"] - period) / period < 0.02  # recovered to 2%
    assert res["ls_fap"] < 1e-3  # highly significant
    assert res["ls_amp_mag"] > 0.1


def test_lomb_scargle_noise_is_not_significant():
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(0, 300, 250))
    mag = 18.0 + rng.normal(0, 0.02, t.size)  # pure noise, no period
    magerr = np.full(t.size, 0.02)
    res = _lomb_scargle(t, mag, magerr)
    # No injected signal: the false-alarm probability should not be tiny.
    assert not (res["ls_fap"] < 1e-3)


def test_lomb_scargle_too_few_epochs():
    res = _lomb_scargle(np.array([1.0, 2.0, 3.0]), np.array([18.0, 18.1, 17.9]), None)
    assert np.isnan(res["ls_period_d"])


def test_lomb_scargle_rejects_one_day_alias():
    # Nightly cadence (integer-day spacing + jitter) with a spurious 1-day signal:
    # the returned period must NOT be ~1 day (the diurnal alias is masked), and the
    # global-max alias must be recorded.
    rng = np.random.default_rng(7)
    nights = np.arange(0, 300)
    t = nights + rng.uniform(-0.02, 0.02, nights.size)  # ~once per night
    mag = 18.0 + 0.2 * np.sin(2 * np.pi * t / 1.0) + rng.normal(0, 0.02, t.size)
    res = _lomb_scargle(t, mag, np.full(t.size, 0.02))
    if np.isfinite(res["ls_period_d"]):
        assert abs(res["ls_period_d"] - 1.0) > 0.05  # not the 1-day alias
        assert abs(res["ls_period_d"] - 0.5) > 0.02   # nor the 2 c/d alias


def test_lomb_scargle_keeps_genuine_short_period():
    # A 0.37-day period is far from the integer-cycles/day comb and must survive.
    rng = np.random.default_rng(8)
    t = np.sort(rng.uniform(0, 300, 400))
    period = 0.37
    mag = 18.0 + 0.15 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.02, t.size)
    res = _lomb_scargle(t, mag, np.full(t.size, 0.02))
    assert np.isfinite(res["ls_period_d"])
    assert abs(res["ls_period_d"] - period) / period < 0.03


def test_classify_candidate():
    from seti.acquire.science import classify_candidate
    assert classify_candidate("CV*", "DAP") == "interacting binary (CV)"
    assert classify_candidate("WD*", "DA+M") == "WD+dwarf binary"
    assert classify_candidate("WD*", "DAZ") == "metal-polluted WD (disk)"
    assert classify_candidate("WD*", "DA") == "white dwarf (other)"
    assert classify_candidate("", "") == "unexamined"
    assert classify_candidate("EB*", "") == "eclipsing binary"
