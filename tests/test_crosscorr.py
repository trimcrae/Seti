"""Offline unit tests for the high-resolution cross-correlation engine.

No network: a synthetic planet O2 signal is injected along a known (Kp, Vsys)
track across simulated in-transit exposures and we verify the Kp-Vsys map recovers
the injection at high S/N, that pure noise yields no significant peak, and that the
air->vacuum conversion is explicit and correct.  Imports only
``seti.crosscorr.xcorr``.
"""

from __future__ import annotations

import numpy as np

from seti.crosscorr.xcorr import (
    C_KMS,
    air_to_vacuum,
    cross_correlation,
    kp_vsys_map,
    molecular_template,
    o2_a_band_template,
    planet_rv_track,
    vacuum_to_air,
)


# --- air <-> vacuum -------------------------------------------------------
def test_air_to_vacuum_offset_is_physical():
    # In the optical, vacuum wavelengths exceed air by ~1.8-2.3 A; the offset
    # must be positive and grow slowly with wavelength.
    air = np.array([5000.0, 6000.0, 7600.0, 9000.0])
    vac = air_to_vacuum(air)
    offset = vac - air
    assert np.all(offset > 1.0) and np.all(offset < 3.0)
    assert np.all(np.diff(offset) > 0)                 # monotonic in lambda
    # A known point: at ~7600 A the shift is ~2.1 A.
    assert abs(air_to_vacuum(np.array([7600.0]))[0] - 7600.0 - 2.1) < 0.3


def test_vacuum_to_air_roundtrip():
    air = np.array([5500.0, 7600.0, 8500.0])
    back = vacuum_to_air(air_to_vacuum(air))
    assert np.allclose(back, air, atol=1e-3)


def test_o2_template_is_in_a_band_and_vacuum():
    wl, depth = o2_a_band_template()
    assert wl.size > 20                                # a real line comb
    assert wl.min() > 7585.0 and wl.max() < 7702.0     # 0.76 um A-band window
    assert np.all(depth > 0) and abs(depth.max() - 1.0) < 1e-9
    # Vacuum scale: every line sits redward of its air value.
    assert np.all(wl > vacuum_to_air(wl))


# --- planet velocity track ------------------------------------------------
def test_planet_rv_track_zero_at_transit():
    assert abs(planet_rv_track(0.0, 40.0, -13.0) - (-13.0)) < 1e-9
    # Quarter phase reaches the full semi-amplitude above systemic.
    assert abs(planet_rv_track(0.25, 40.0, 0.0) - 40.0) < 1e-9


# --- injection recovery ---------------------------------------------------
def _simulate(kp_true, vsys_true, phases, wl, template_wl, template_depth,
              noise=0.02, line_sigma=0.03, seed=0):
    """Build in-transit residuals with an injected planet O2 forest + noise."""
    rng = np.random.default_rng(seed)
    resid = np.zeros((phases.size, wl.size))
    vtrack = planet_rv_track(phases, kp_true, vsys_true)
    for e, v in enumerate(vtrack):
        shifted = template_wl * (1.0 + v / C_KMS)
        prof = np.zeros_like(wl)
        for lam, d in zip(shifted, template_depth):
            prof += d * np.exp(-0.5 * ((wl - lam) / line_sigma) ** 2)
        # Absorption imprints a negative dip; scale so lines are ~2% deep.
        resid[e] = -0.02 * prof + rng.normal(0.0, noise, wl.size)
    return resid


def _grids():
    rv_grid = np.arange(-90.0, 90.01, 1.0)
    kp_grid = np.arange(0.0, 100.01, 2.0)
    vsys_grid = np.arange(-40.0, 40.01, 1.0)
    return rv_grid, kp_grid, vsys_grid


def test_kp_vsys_map_recovers_injected_signal():
    kp_true, vsys_true = 40.0, -13.0
    # Phases spread across (a hypothetical extended) transit so the planet's
    # velocity changes enough to constrain Kp as well as Vsys.
    phases = np.linspace(-0.06, 0.06, 15)
    wl = np.arange(7586.0, 7700.0, 0.02)
    twl, tdepth = o2_a_band_template()
    resid = _simulate(kp_true, vsys_true, phases, wl, twl, tdepth,
                      noise=0.02, seed=1)

    rv_grid, kp_grid, vsys_grid = _grids()
    ccfs = np.array([cross_correlation(wl, resid[e], twl, tdepth, rv_grid)
                     for e in range(phases.size)])
    out = kp_vsys_map(ccfs, phases, rv_grid, kp_grid, vsys_grid)

    assert abs(out["kp_peak"] - kp_true) <= 4.0
    assert abs(out["vsys_peak"] - vsys_true) <= 2.0
    assert out["significance"] > 8.0                   # clean detection


def test_pure_noise_yields_no_significant_peak():
    phases = np.linspace(-0.06, 0.06, 15)
    wl = np.arange(7586.0, 7700.0, 0.02)
    twl, tdepth = o2_a_band_template()
    rng = np.random.default_rng(7)
    resid = rng.normal(0.0, 0.02, (phases.size, wl.size))   # no injected planet

    rv_grid, kp_grid, vsys_grid = _grids()
    ccfs = np.array([cross_correlation(wl, resid[e], twl, tdepth, rv_grid)
                     for e in range(phases.size)])
    out = kp_vsys_map(ccfs, phases, rv_grid, kp_grid, vsys_grid)
    assert out["significance"] < 5.0                   # no false detection


def test_injection_beats_noise_control():
    # Same machinery, matched noise: the injected case must be far more
    # significant than the noise-only control.
    phases = np.linspace(-0.06, 0.06, 15)
    wl = np.arange(7586.0, 7700.0, 0.02)
    twl, tdepth = o2_a_band_template()
    rv_grid, kp_grid, vsys_grid = _grids()

    sig = _simulate(40.0, -13.0, phases, wl, twl, tdepth, noise=0.02, seed=3)
    ccf_sig = np.array([cross_correlation(wl, sig[e], twl, tdepth, rv_grid)
                        for e in range(phases.size)])
    s_sig = kp_vsys_map(ccf_sig, phases, rv_grid, kp_grid, vsys_grid)["significance"]

    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 0.02, (phases.size, wl.size))
    ccf_noi = np.array([cross_correlation(wl, noise[e], twl, tdepth, rv_grid)
                        for e in range(phases.size)])
    s_noi = kp_vsys_map(ccf_noi, phases, rv_grid, kp_grid, vsys_grid)["significance"]

    assert s_sig > 3.0 * s_noi
