"""Offline tests for the novel narrow-absorption-line technosignature search."""
from __future__ import annotations

import numpy as np

from seti.spectra.absorb import classify_absorption_line, find_absorption_lines
from seti.spectra.detect import EmissionLine


def _spectrum(n=2000, dlam=1.0, lam0=4000.0, snr_cont=200.0, seed=0):
    rng = np.random.default_rng(seed)
    wave = lam0 + dlam * np.arange(n)
    cont = 1.0 + 0.3 * np.sin(np.linspace(0, 3, n)) + 0.0002 * np.arange(n)
    err = cont / snr_cont
    flux = cont + rng.normal(0, err)
    return wave, flux, err, cont


def _absorb(wave, flux, lam_c, depth, sigma_pix, dlam=1.0):
    x = (wave - lam_c) / (sigma_pix * dlam)
    return flux - depth * np.exp(-0.5 * x**2)


def test_detects_unresolved_absorption_line():
    wave, flux, err, cont = _spectrum(seed=1)
    lsf = 1.5
    flux = _absorb(wave, flux, 5000.0, depth=0.5, sigma_pix=lsf)
    lines = find_absorption_lines(wave, flux, err, lsf_sigma_pix=lsf, snr_min=8.0)
    assert lines, "should detect the injected absorption line"
    best = min(lines, key=lambda ln: abs(ln.wavelength - 5000.0))
    assert abs(best.wavelength - 5000.0) <= 2.0
    assert best.significance > 8.0
    # An unresolved non-catalogued absorber at 5000 A survives the funnel.
    assert classify_absorption_line(best, redshift=0.0) is None


def test_rejects_stellar_and_ism_absorption():
    # H-alpha (stellar) and Na D (ISM) must be rejected as natural.
    ha = EmissionLine(0, 6562.8, significance=30.0, width_ratio=1.0, amplitude=1.0,
                      ew=1.0, n_pix=3, fwhm_pix=2.5)
    assert classify_absorption_line(ha, redshift=0.0) == "stellar_line"
    nad = EmissionLine(0, 5889.9, significance=30.0, width_ratio=1.0, amplitude=1.0,
                       ew=1.0, n_pix=3, fwhm_pix=2.5)
    assert classify_absorption_line(nad, redshift=0.0) in ("stellar_line", "ism_line", "sky_line")
    # A diffuse interstellar band (6283.8) is rejected.
    dib = EmissionLine(0, 6283.8, significance=20.0, width_ratio=1.0, amplitude=1.0,
                       ew=1.0, n_pix=3, fwhm_pix=2.5)
    assert classify_absorption_line(dib, redshift=0.0) in ("diffuse_band", "telluric")


def test_absorption_mode_end_to_end():
    from seti.spectra.vet import search_spectra
    lsf = 1.5
    res_match = 5000.0 / (lsf * 1.0 * 2.3548)
    wave, flux, err, _ = _spectrum(seed=5)
    flux = _absorb(wave, flux, 5000.0, depth=0.5, sigma_pix=lsf)
    out = search_spectra([{"spec_id": "A", "wave": wave, "flux": flux,
                           "ivar": 1.0 / err**2, "resolution": res_match}],
                         snr_min=8.0, mode="absorption")
    assert out["n_searched"] == 1
    assert any(abs(c["wavelength"] - 5000.0) <= 2.0 for c in out["candidates"])
