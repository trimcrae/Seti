"""Offline tests for the anomalous-dimming (Boyajian-star) dip statistics."""

from __future__ import annotations

import numpy as np

from seti.dimming.dips import detect_dips


def _flat(n=200, base=12.0, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 1000, n))
    m = base + rng.normal(0, noise, n)
    return t, m, np.full(n, noise)


def test_flat_lightcurve_no_dips():
    t, m, e = _flat()
    s = detect_dips(t, m, e)
    assert s is not None
    assert s.n_dips == 0
    assert s.max_depth < 0.05
    assert s.score < 0.3


def test_boyajian_like_aperiodic_dimming_scores_high():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 1500, 400))
    m = 12.0 + rng.normal(0, 0.01, t.size)
    # A handful of deep, irregular (aperiodic) dips of 10-22% (0.11-0.27 mag).
    for tc, depth_mag in [(220, 0.16), (540, 0.27), (910, 0.12), (1290, 0.20)]:
        m += depth_mag * np.exp(-0.5 * ((t - tc) / 12.0) ** 2)
    e = np.full(t.size, 0.01)
    s = detect_dips(t, m, e)
    assert s is not None
    assert s.max_depth > 0.1
    assert s.n_dips >= 3
    assert s.asymmetry > 1.5           # dimming-only, not symmetric
    assert s.score > 0.5


def test_smooth_sinusoid_is_symmetric_low_score():
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 1000, 300))
    m = 12.0 + 0.05 * np.sin(2 * np.pi * t / 30.0) + rng.normal(0, 0.005, t.size)
    s = detect_dips(t, m, magerr=np.full(t.size, 0.005))
    # A symmetric oscillation is not dimming-dominated.
    assert s.asymmetry < 1.6
    assert s.score < 0.6


def test_eclipsing_binary_is_periodic():
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 1000, 600))
    period = 3.2
    phase = (t % period) / period
    m = 12.0 + rng.normal(0, 0.01, t.size)
    m += 0.3 * (np.abs(phase - 0.5) < 0.05)   # periodic box eclipses
    s = detect_dips(t, m, magerr=np.full(t.size, 0.01))
    assert s is not None
    # Strictly periodic eclipses produce strong Lomb-Scargle power.
    assert s.period_power > 0.1


def test_too_few_epochs_returns_none():
    t, m, e = _flat(n=10)
    assert detect_dips(t, m, e) is None
