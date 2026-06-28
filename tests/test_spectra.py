"""Offline tests for the laser-line detection + rejection core.

We build synthetic survey-like spectra (smooth continuum + photon noise), inject
Gaussian emission features of controlled width, and verify that an unresolved
"laser" line is detected and survives the funnel while cosmic rays, resolved
astrophysical lines, sky lines and redshifted nebular lines are rejected.
"""

from __future__ import annotations

import numpy as np

from seti.spectra.detect import estimate_continuum, find_emission_lines
from seti.spectra.reject import classify_line, reject_lines


def _spectrum(n=2000, dlam=1.0, lam0=4000.0, lsf_sigma_pix=1.5, snr_cont=200.0,
              seed=0):
    rng = np.random.default_rng(seed)
    wave = lam0 + dlam * np.arange(n)
    # Gently sloped + curved continuum, ~unit level.
    cont = 1.0 + 0.3 * np.sin(np.linspace(0, 3, n)) + 0.0002 * np.arange(n)
    err = cont / snr_cont
    flux = cont + rng.normal(0, err)
    return wave, flux, err, cont, lsf_sigma_pix


def _inject(wave, flux, lam_c, amp, sigma_pix, dlam=1.0):
    x = (wave - lam_c) / (sigma_pix * dlam)
    return flux + amp * np.exp(-0.5 * x**2)


def test_continuum_ignores_narrow_line():
    wave, flux, err, cont, lsf = _spectrum()
    flux2 = _inject(wave, flux, 5000.0, amp=0.5, sigma_pix=lsf)
    est = estimate_continuum(flux2, window=101)
    # The median continuum should not chase the narrow spike.
    i = int(np.argmin(np.abs(wave - 5000.0)))
    assert est[i] < flux2[i] - 0.2


def test_detects_unresolved_laser_line():
    wave, flux, err, cont, lsf = _spectrum(seed=1)
    flux = _inject(wave, flux, 5000.0, amp=0.5, sigma_pix=lsf)  # ~50 sigma line
    lines = find_emission_lines(wave, flux, err, lsf_sigma_pix=lsf, snr_min=8.0)
    assert lines, "should detect the injected line"
    best = min(lines, key=lambda ln: abs(ln.wavelength - 5000.0))
    assert abs(best.wavelength - 5000.0) <= 2.0
    assert best.significance > 8.0
    assert 0.6 <= best.width_ratio <= 1.6  # unresolved
    assert classify_line(best, redshift=0.0) is None  # survives the funnel


def test_rejects_cosmic_ray_subpixel():
    wave, flux, err, cont, lsf = _spectrum(seed=2)
    flux = _inject(wave, flux, 5200.0, amp=0.8, sigma_pix=0.35 * lsf)  # too sharp
    lines = find_emission_lines(wave, flux, err, lsf_sigma_pix=lsf, snr_min=8.0)
    near = [ln for ln in lines if abs(ln.wavelength - 5200.0) <= 2.0]
    assert near, "CR spike is still detected by the matched filter"
    assert all(classify_line(ln) == "cosmic_ray" for ln in near)


def test_rejects_resolved_astrophysical_width():
    wave, flux, err, cont, lsf = _spectrum(seed=3)
    flux = _inject(wave, flux, 5300.0, amp=0.4, sigma_pix=3.5 * lsf)  # broad
    lines = find_emission_lines(wave, flux, err, lsf_sigma_pix=lsf, snr_min=8.0)
    near = [ln for ln in lines if abs(ln.wavelength - 5300.0) <= 4.0]
    assert near
    assert all(classify_line(ln) == "resolved_line" for ln in near)


def test_rejects_sky_line_wavelength():
    wave, flux, err, cont, lsf = _spectrum(seed=4)
    flux = _inject(wave, flux, 5577.34, amp=0.5, sigma_pix=lsf)  # [O I] sky
    lines = find_emission_lines(wave, flux, err, lsf_sigma_pix=lsf, snr_min=8.0)
    near = [ln for ln in lines if abs(ln.wavelength - 5577.34) <= 2.0]
    assert near
    assert all(classify_line(ln) == "sky_line" for ln in near)


def test_rejects_redshifted_halpha():
    # H-alpha 6564.6 rest, redshifted to z=0.2 -> 7877 A (clear of sky/telluric);
    # with the source redshift known, the funnel must recognise it as astrophysical.
    z = 0.2
    obs = 6564.61 * (1 + z)
    wave, flux, err, cont, lsf = _spectrum(n=4000, lam0=4000.0, seed=5)
    flux = _inject(wave, flux, obs, amp=0.5, sigma_pix=lsf)
    lines = find_emission_lines(wave, flux, err, lsf_sigma_pix=lsf, snr_min=8.0)
    near = [ln for ln in lines if abs(ln.wavelength - obs) <= 2.0]
    assert near
    assert all(classify_line(ln, redshift=z) == "astrophysical_line" for ln in near)
    # ...but at z=0 the same wavelength is NOT a known line -> survives.
    assert any(classify_line(ln, redshift=0.0) is None for ln in near)


def test_reject_lines_histogram():
    wave, flux, err, cont, lsf = _spectrum(seed=6)
    flux = _inject(wave, flux, 5000.0, amp=0.5, sigma_pix=lsf)        # laser
    flux = _inject(wave, flux, 5577.34, amp=0.5, sigma_pix=lsf)        # sky
    flux = _inject(wave, flux, 5200.0, amp=0.8, sigma_pix=0.35 * lsf)  # CR
    lines = find_emission_lines(wave, flux, err, lsf_sigma_pix=lsf, snr_min=8.0)
    survivors, counts = reject_lines(lines, redshift=0.0)
    assert len(survivors) >= 1
    assert counts.get("sky_line", 0) >= 1
    assert counts.get("cosmic_ray", 0) >= 1
    assert all(abs(s.wavelength - 5000.0) <= 2.0 for s in survivors)


def test_process_and_search_spectra():
    from seti.spectra.detect import EmissionLine
    from seti.spectra.vet import process_spectrum, score_line, search_spectra

    # Resolution chosen so the LSF (~1.5 px on this 1 A/pix grid) matches the
    # injected line width, i.e. an unresolved line: R = lam / (1.5 px * dlam * 2.355).
    lsf = 1.5
    res_match = 5000.0 / (lsf * 1.0 * 2.3548)
    # Spectrum A: a clean unresolved laser at 5000 A.
    wave, flux, err, cont, _ = _spectrum(seed=10)
    flux_a = _inject(wave, flux, 5000.0, amp=0.5, sigma_pix=lsf)
    ivar_a = 1.0 / err**2
    cands, counts = process_spectrum("A", wave, flux_a, ivar_a, redshift=0.0,
                                     resolution=res_match, snr_min=8.0)
    assert any(abs(c.wavelength - 5000.0) <= 2.0 for c in cands)

    # Spectrum B: only a sky line -> no surviving candidate.
    wave_b, flux_b, err_b, _, _ = _spectrum(seed=11)
    flux_b = _inject(wave_b, flux_b, 5577.34, amp=0.6, sigma_pix=lsf)
    ivar_b = 1.0 / err_b**2

    res = search_spectra([
        {"spec_id": "A", "wave": wave, "flux": flux_a, "ivar": ivar_a,
         "resolution": res_match},
        {"spec_id": "B", "wave": wave_b, "flux": flux_b, "ivar": ivar_b,
         "resolution": res_match},
    ], snr_min=8.0)
    assert res["n_searched"] == 2
    assert res["rejection_counts"].get("sky_line", 0) >= 1
    assert any(c["spec_id"] == "A" for c in res["candidates"])
    assert not any(c["spec_id"] == "B" for c in res["candidates"])

    # score_line: a clean strong unresolved isolated line scores high.
    strong = EmissionLine(0, 5000.0, significance=40.0, width_ratio=1.0,
                          amplitude=1.0, ew=1.0, n_pix=3)
    assert score_line(strong, n_survivors=1) > 0.8
    # a low-S/N, ill-matched, crowded line scores low.
    weak = EmissionLine(0, 5000.0, significance=8.0, width_ratio=2.0,
                        amplitude=0.1, ew=0.1, n_pix=2)
    assert score_line(weak, n_survivors=8) < 0.5


def test_lsf_sigma_pix_from_resolution():
    from seti.spectra.vet import _lsf_sigma_pix
    wave = 4000.0 + 0.8 * np.arange(3000)  # 0.8 A/pix, DESI-like
    # R=3000 at ~5200 A -> FWHM ~1.73 A -> sigma ~0.74 A -> ~0.9 pix (floored 0.6).
    s = _lsf_sigma_pix(wave, 3000.0)
    assert 0.6 <= s <= 2.0


def test_spectra_run_end_to_end(tmp_path):
    from seti.config import load_config
    from seti.spectra.run import spectra_run

    lsf = 1.5
    res_match = 5000.0 / (lsf * 1.0 * 2.3548)
    spectra = []
    # 5 clean spectra (no injected line).
    for s in range(5):
        wave, flux, err, _, _ = _spectrum(seed=100 + s)
        spectra.append({"spec_id": f"clean{s}", "wave": wave, "flux": flux,
                        "ivar": 1.0 / err**2, "resolution": res_match})
    # 1 spectrum with a clean unresolved laser line.
    wave, flux, err, _, _ = _spectrum(seed=200)
    flux = _inject(wave, flux, 5000.0, amp=0.5, sigma_pix=lsf)
    spectra.append({"spec_id": "laser", "wave": wave, "flux": flux,
                    "ivar": 1.0 / err**2, "resolution": res_match})

    cfg = load_config()
    cfg.root = tmp_path  # write results under the temp dir, not the repo
    summary = spectra_run(cfg, spectra=spectra, snr_min=8.0)

    assert summary["n_searched"] == 6
    assert summary["n_candidates"] >= 1
    assert "occurrence_limit" in summary
    assert summary["occurrence_limit"]["f_upper"] > summary["occurrence_limit"]["f_point"]
    assert (tmp_path / "results" / "spectra" / "summary.json").exists()


def test_fetch_spectra_parses_dict_records():
    from seti.spectra.acquire import fetch_spectra

    class _Res:
        def __init__(self, records, ids=None):
            self.records = records
            if ids is not None:
                self.ids = ids

    class _MockClient:
        all_datasets = ["DESI-DR1", "SDSS-DR17"]

        def find(self, outfields=None, constraints=None, sort=None, limit=None):
            # SPARCL find: when 'sparcl_id' is requested it appears in the records,
            # and the id list is also exposed on the result's .ids attribute.
            return _Res(
                [{"sparcl_id": "uuid-1", "ra": 10.0, "dec": 5.0, "redshift": 0.0,
                  "spectype": "STAR", "data_release": "DESI-DR1"},
                 {"sparcl_id": "uuid-2", "ra": 11.0, "dec": 6.0, "redshift": 0.3,
                  "spectype": "GALAXY", "data_release": "DESI-DR1"}],
                ids=["uuid-1", "uuid-2"])

        def retrieve(self, uuid_list=None, include=None):
            wave = np.linspace(4000, 9000, 500)
            recs = []
            for uid in uuid_list:
                recs.append({"sparcl_id": uid, "ra": 10.0, "dec": 5.0,
                             "redshift": 0.0, "spectype": "STAR",
                             "data_release": "DESI-DR1",
                             "wavelength": wave, "flux": np.ones_like(wave),
                             "ivar": np.full_like(wave, 100.0)})
            return _Res(recs)

    specs = fetch_spectra(n=2, dataset="DESI-DR1", client=_MockClient())
    assert len(specs) == 2
    s = specs[0]
    assert s["spec_id"] == "uuid-1"
    assert s["wave"].size == 500 and s["flux"].size == 500 and s["ivar"].size == 500
    assert s["resolution"] == 3000.0
    assert s["meta"]["spectype"] == "STAR"
