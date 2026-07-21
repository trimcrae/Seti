"""Offline unit tests for the JWST biosignature spectrum scorers.

No network: every scorer is fed a synthetic spectrum built here.  Imports ONLY
:mod:`seti.jwst_bio.spectrum`.
"""

from __future__ import annotations

import numpy as np

from seti.jwst_bio.spectrum import (
    DEFAULT_BANDS,
    abiotic_false_positive,
    build_transmission_spectrum,
    disequilibrium_biosignature,
    eclipse_brightness_temperature,
    laser_line_scan,
    molecular_feature_detect,
    transit_mask_from_ephemeris,
)

_RNG = np.random.default_rng(20260721)


def _grid(lo=0.6, hi=12.0, n=2400):
    return np.linspace(lo, hi, n)


def _spectrum(wavelength, baseline_ppm=5000.0, bumps=None, noise_ppm=8.0):
    """Synthetic transmission spectrum: flat continuum + Gaussian absorption bumps.

    ``bumps`` is a list of ``(center_um, half_width_um, amplitude_ppm)``.  Returns
    (depth, depth_err) in fractional units.
    """
    depth = np.full(wavelength.shape, baseline_ppm, float)
    for center, hw, amp in (bumps or []):
        depth += amp * np.exp(-0.5 * ((wavelength - center) / (hw / 2.0)) ** 2)
    err = np.full(wavelength.shape, noise_ppm, float)
    depth = depth + _RNG.normal(0.0, noise_ppm, wavelength.shape)
    return depth * 1e-6, err * 1e-6


# --- build_transmission_spectrum -------------------------------------------
def test_build_transmission_spectrum_recovers_injected_depth():
    wl = np.linspace(1.0, 5.0, 200)
    true_depth = 0.005 + 0.001 * np.exp(-0.5 * ((wl - 3.3) / 0.1) ** 2)  # CH4 bump
    n_int = 400
    times = np.linspace(0.0, 0.2, n_int)          # days
    mask = (times > 0.07) & (times < 0.13)        # middle = in transit
    flux = np.empty((n_int, wl.size))
    for i in range(n_int):
        depth_i = true_depth if mask[i] else 0.0
        flux[i] = (1.0 - depth_i) + _RNG.normal(0.0, 2e-4, wl.size)
    spec = build_transmission_spectrum(wl, flux, mask)
    assert spec["n_in"] == int(mask.sum())
    assert spec["n_out"] == int((~mask).sum())
    # Recovered depth tracks the injected depth to well within a milli-fraction.
    assert np.nanmedian(np.abs(spec["depth"] - true_depth)) < 5e-4
    # And the CH4 bin is deeper than the continuum.
    ch4_bin = np.argmin(np.abs(wl - 3.3))
    cont_bin = np.argmin(np.abs(wl - 2.0))
    assert spec["depth"][ch4_bin] > spec["depth"][cont_bin]


def test_build_transmission_spectrum_needs_both_phases():
    wl = np.linspace(1.0, 2.0, 10)
    flux = np.ones((5, wl.size))
    spec = build_transmission_spectrum(wl, flux, np.zeros(5, bool))
    assert spec["n_in"] == 0
    assert np.all(np.isnan(spec["depth"]))


def test_transit_mask_from_ephemeris_selects_transit_window():
    period, t0, dur = 24.737, 0.0, 2.0 / 24.0
    times = np.linspace(-period / 2, period / 2, 5000)
    mask = transit_mask_from_ephemeris(times, t0, period, dur)
    assert mask.any() and (~mask).any()
    # In-transit points cluster tightly around mid-transit.
    assert np.abs(times[mask]).max() <= dur / 2 + 1e-6
    assert mask.mean() < 0.05          # transit is a small phase fraction


# --- molecular_feature_detect ----------------------------------------------
def test_flat_spectrum_has_no_features():
    wl = _grid()
    depth, err = _spectrum(wl, bumps=None, noise_ppm=6.0)
    det = molecular_feature_detect(wl, depth, err)
    for gas, d in det.items():
        assert not d["detected"], f"{gas} falsely detected in a flat spectrum"


def test_injected_bands_are_detected():
    wl = _grid()
    bumps = [(1.40, 0.15, 300.0),   # H2O
             (3.30, 0.15, 220.0),   # CH4
             (4.30, 0.15, 260.0)]   # CO2
    depth, err = _spectrum(wl, bumps=bumps, noise_ppm=6.0)
    det = molecular_feature_detect(wl, depth, err)
    for gas in ("H2O", "CH4", "CO2"):
        assert det[gas]["detected"], f"{gas} not detected"
        assert det[gas]["significance"] > 3.0
    # CO / O2 / O3 / N2O were not injected -> not detected.
    for gas in ("CO", "O2", "O3", "N2O"):
        assert not det[gas]["detected"]


# --- disequilibrium_biosignature -------------------------------------------
def _detections_from(wl, bumps, noise_ppm=6.0):
    depth, err = _spectrum(wl, bumps=bumps, noise_ppm=noise_ppm)
    return molecular_feature_detect(wl, depth, err)


def test_disequilibrium_flags_ch4_co2_pair():
    wl = _grid()
    det = _detections_from(wl, [(1.40, 0.15, 300.0),
                                (3.30, 0.15, 250.0),   # CH4
                                (4.30, 0.15, 280.0)])  # CO2
    bio = disequilibrium_biosignature(det)
    assert bio["is_biosignature"]
    assert bio["best_pair"] == "CH4+CO2"
    assert bio["best"]["joint_significance"] > bio["best"]["limiting_significance"]


def test_single_gas_is_not_a_biosignature():
    wl = _grid()
    det = _detections_from(wl, [(4.30, 0.15, 300.0)])   # CO2 only
    bio = disequilibrium_biosignature(det)
    assert not bio["is_biosignature"]
    assert bio["best_pair"] is None


# --- abiotic_false_positive ------------------------------------------------
def test_abiotic_flags_o2_o3_without_ch4():
    wl = _grid()
    det = _detections_from(wl, [(0.76, 0.02, 400.0),   # O2 A-band
                                (1.27, 0.04, 400.0),   # O2 CIA
                                (9.60, 0.40, 400.0)])  # O3, no CH4
    gate = abiotic_false_positive(det)
    assert gate["oxygen_present"]
    assert gate["abiotic_flag"]
    assert any("without CH4" in r for r in gate["reasons"])


def test_abiotic_does_not_flag_o2_with_ch4_and_no_co():
    wl = _grid()
    det = _detections_from(wl, [(0.76, 0.02, 400.0),
                                (1.27, 0.04, 400.0),   # O2
                                (3.30, 0.15, 300.0)])  # CH4 disequilibrium, no CO
    gate = abiotic_false_positive(det)
    assert gate["oxygen_present"]
    assert gate["ch4"]
    assert not gate["co"]
    assert not gate["abiotic_flag"]


def test_abiotic_flags_co_photolysis_tracer():
    wl = _grid()
    det = _detections_from(wl, [(1.27, 0.04, 400.0),   # O2
                                (3.30, 0.15, 300.0),   # CH4
                                (4.70, 0.12, 400.0)])  # CO -> CO2 photolysis
    gate = abiotic_false_positive(det)
    assert gate["co"]
    assert gate["abiotic_flag"]
    assert any("CO co-detected" in r for r in gate["reasons"])


# --- eclipse_brightness_temperature ----------------------------------------
_TEFF, _A_RS, _RP_RS = 3096.0, 94.4, 0.0730


def _eclipse_depth_ppm_for(t_day):
    """Invert the RJ relation to synthesise an eclipse depth for a day-side T."""
    return (_RP_RS ** 2) * (t_day / _TEFF) * 1e6


def test_eclipse_bare_rock_classified_as_rock():
    t_bare = _TEFF * np.sqrt(1.0 / _A_RS) * (2.0 / 3.0) ** 0.25
    depth = _eclipse_depth_ppm_for(t_bare)
    res = eclipse_brightness_temperature(depth, _RP_RS, _TEFF, _A_RS)
    assert res["classification"] == "bare_rock"
    assert res["t_day_brightness_k"] > res["t_full_redist_k"]


def test_eclipse_atmosphere_classified_as_atmosphere():
    t_full = _TEFF * np.sqrt(1.0 / (2.0 * _A_RS))
    depth = _eclipse_depth_ppm_for(t_full)
    res = eclipse_brightness_temperature(depth, _RP_RS, _TEFF, _A_RS)
    assert res["classification"] == "atmosphere"
    assert res["t_day_brightness_k"] < res["t_bare_rock_max_k"]


def test_eclipse_invalid_input():
    res = eclipse_brightness_temperature(100.0, 0.0, _TEFF, _A_RS)
    assert res["classification"] == "invalid_input"


# --- laser_line_scan -------------------------------------------------------
def test_laser_scan_flags_narrow_line_and_ignores_smooth():
    wl = np.linspace(1.0, 5.0, 800)
    smooth = 100.0 + 5.0 * (wl - 3.0) ** 2          # smooth continuum
    assert not laser_line_scan(wl, smooth)["laser_line_flag"]
    spike = smooth.copy()
    j = 400
    spike[j - 1:j + 2] += np.array([40.0, 80.0, 40.0])   # narrow, bounded, interior
    out = laser_line_scan(wl, spike)
    assert out["laser_line_flag"]
    assert out["peak"]["index"] == j


def test_default_bands_cover_expected_gases():
    assert {"H2O", "CH4", "CO2", "CO", "O2", "O3", "N2O"} <= set(DEFAULT_BANDS)
