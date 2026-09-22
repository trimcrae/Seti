"""Offline tests for the per-exposure persistence test (seti.spectra.persist).

A synthetic SDSS *full* spec file (coadd HDU + per-exposure spCFrame HDUs) is
built in memory with astropy.  Injected signals: a line present in every
exposure (must come out ``persistent`` with the injected EW recovered), a line
in one exposure only (``transient``), a line that tracks a sky-model line
(``sky_residual``), the absorption analogue, and the degraded cases (no
exposures, wavelength not covered) which must be ``untestable`` rather than a
verdict.  The shard/reduce orchestration runs end-to-end on a temporary
results tree with the SAS fetch monkeypatched to the synthetic bytes.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from seti.spectra import persist
from seti.spectra.linelist import identify_rest_frame

LAM0 = 6765.5        # test line: r1 camera only, > 300 km/s from every known line
DEFAULT_SEED = 3     # every synthetic file seeds its OWN generator (see make_spec_file)
# A zero-signal null must never reach the significance at which a line is believed.
COMBINED_SIG_FLOOR = persist.COMBINED_SIG


def _grid(lo, hi):
    loglam = np.arange(np.log10(lo), np.log10(hi), 1e-4)
    return loglam, 10.0 ** loglam


def _gauss(wave, lam, amp, sigma_A):
    return amp * np.exp(-0.5 * ((wave - lam) / sigma_A) ** 2)


def _exposure_hdu(name, loglam, flux, ivar, sky, mask=None, wdisp=1.1, mjd=55000, exptime=900.0):
    n = loglam.size
    cols = [
        fits.Column(name="flux", format="E", array=flux.astype(np.float32)),
        fits.Column(name="loglam", format="E", array=loglam.astype(np.float32)),
        fits.Column(name="ivar", format="E", array=ivar.astype(np.float32)),
        fits.Column(name="mask", format="J", array=(mask if mask is not None else np.zeros(n, int))),
        fits.Column(name="wdisp", format="E", array=np.full(n, wdisp, np.float32)),
        fits.Column(name="sky", format="E", array=sky.astype(np.float32)),
        fits.Column(name="calib", format="E", array=np.ones(n, np.float32)),
        fits.Column(name="x", format="E", array=np.arange(n, dtype=np.float32)),
    ]
    h = fits.BinTableHDU.from_columns(cols, name=name)
    h.header["MJD"] = mjd
    h.header["EXPTIME"] = exptime
    return h


def make_spec_file(line_amp_per_exp, n_exp=4, sky_line_amp_per_exp=None, noise=0.05,
                   absorption=False, lam=LAM0, coadd_amp=None, seed=DEFAULT_SEED) -> bytes:
    """Synthetic full spec file.  ``line_amp_per_exp`` gives the injected peak
    amplitude (continuum = 10) in each exposure; ``sky_line_amp_per_exp`` puts a
    sky-model line at the same wavelength with that amplitude per exposure.

    The noise generator is seeded **per call** from ``seed``: a shared module-level
    generator would make every test's noise realisation depend on how many tests
    ran before it, so the same assertion would pass or fail with the test order
    (pytest-xdist, ``-k`` selection, a re-run of one test).  Each file is now a
    reproducible draw, and the tests that depend on a null being quiet average
    over an explicit ensemble of seeds instead of trusting one.
    """
    rng = np.random.default_rng(seed)
    # Per-exposure (native) grid for r1 and b1; coadd grid spans both.
    lb, wb = _grid(3800.0, 6200.0)
    lr, wr = _grid(5800.0, 9200.0)
    lc, wc = _grid(3800.0, 9200.0)
    sigma = lam / 2000.0 / 2.3548
    hdus = [fits.PrimaryHDU()]
    cont = 10.0
    # Coadd: continuum plus the mean injected line (the coadd shows the average).
    if coadd_amp is None:
        coadd_amp = float(np.mean(line_amp_per_exp))
    sgn = -1.0 if absorption else 1.0
    fc = cont + sgn * _gauss(wc, lam, coadd_amp, sigma) + rng.normal(0, noise / np.sqrt(n_exp), wc.size)
    coadd = _exposure_hdu("COADD", lc, fc, np.full(wc.size, n_exp / noise ** 2),
                          np.full(wc.size, 2.0))
    hdus.append(coadd)
    hdus.append(fits.BinTableHDU.from_columns([fits.Column(name="PLATE", format="J", array=[1000])],
                                              name="SPECOBJ"))
    hdus.append(fits.BinTableHDU.from_columns([fits.Column(name="LINEZ", format="E", array=[0.0])],
                                              name="SPZLINE"))
    for k in range(n_exp):
        amp = line_amp_per_exp[k]
        expid = 100 + k
        # The line (and any sky line) is in EVERY arm that covers it, as in a
        # real spCFrame set where the b/r overlap sees the same photons twice.
        sky_r, sky_b = np.full(wr.size, 2.0), np.full(wb.size, 2.0)
        if sky_line_amp_per_exp is not None:
            sky_r = sky_r + _gauss(wr, lam, sky_line_amp_per_exp[k], sigma)
            sky_b = sky_b + _gauss(wb, lam, sky_line_amp_per_exp[k], sigma)
        fr = cont + sgn * _gauss(wr, lam, amp, sigma) + rng.normal(0, noise, wr.size)
        fb = cont + sgn * _gauss(wb, lam, amp, sigma) + rng.normal(0, noise, wb.size)
        hdus.append(_exposure_hdu(f"B1-{expid:08d}", lb, fb, np.full(wb.size, 1 / noise ** 2),
                                  sky_b))
        hdus.append(_exposure_hdu(f"R1-{expid:08d}", lr, fr, np.full(wr.size, 1 / noise ** 2),
                                  sky_r))
    buf = io.BytesIO()
    fits.HDUList(hdus).writeto(buf)
    return buf.getvalue()


def _run(data: bytes, mode="emission", lam=LAM0):
    with fits.open(io.BytesIO(data)) as hd:
        parsed = persist.parse_sdss_spec(hd)
    fc, ex = persist.sdss_exposure_measurements(parsed, lam, mode)
    return parsed, fc, ex, persist.classify_persistence(fc, ex)


def test_persistent_line_recovered_with_ew():
    amp = 1.0                                    # EW = amp * sigma * sqrt(2pi) / cont
    parsed, fc, ex, cls = _run(make_spec_file([amp] * 4))
    assert len(parsed["exposures"]) == 8 and len(ex) == 4
    assert cls["persistence_class"] == "persistent", cls
    assert cls["n_present"] == 4 and cls["coadd_recovered"]
    ew_true = amp * (LAM0 / 2000.0 / 2.3548) * np.sqrt(2 * np.pi) / 10.0
    for e in ex:
        assert e["testable"] and abs(e["ew"] - ew_true) / ew_true < 0.25, e
    assert 0.7 < cls["ratio_to_coadd"] < 1.3
    assert not cls["on_sky_line"]


def test_power_at_the_weakest_survivor_strength():
    """The test must be able to SEE the survivors it is being used to reject.

    The 167 narrow-line survivors have coadd significances from 8.0 to 25.5.
    An "absent in the exposures" verdict on one of them is only meaningful if a
    real line of that strength would have shown up per exposure, so this pins
    the sensitivity: a line weaker than the weakest survivor must still come out
    persistent, present in every exposure, well above the 2.5 sigma
    per-exposure threshold.
    """
    _, fc, ex, cls = _run(make_spec_file([0.25] * 4))
    assert fc["sig"] < 8.0, fc["sig"]          # weaker in the coadd than any survivor
    assert cls["persistence_class"] == "persistent", cls
    assert cls["n_present"] == 4 and cls["n_tested"] == 4
    assert min(e["sig"] for e in ex) > persist.PRESENT_SIG + 1.0, [e["sig"] for e in ex]


def test_transient_single_exposure_spike():
    # One exposure carries a strong spike; the coadd (mean) still shows it.
    parsed, fc, ex, cls = _run(make_spec_file([4.0, 0.0, 0.0, 0.0]))
    assert cls["persistence_class"] == "transient", cls
    assert cls["dominant_frac"] > 0.8 and cls["n_present"] == 1
    assert persist.final_verdict({"persistence_class": "transient", "known_line_match": False}) \
        == "KILLED_transient"


def test_sky_residual_tracks_sky_model():
    sky_amps = [5.0, 20.0, 40.0, 60.0]
    line_amps = [0.1 * s for s in sky_amps]       # residual = 10 % of the sky line
    parsed, fc, ex, cls = _run(make_spec_file(line_amps, sky_line_amp_per_exp=sky_amps))
    assert cls["on_sky_line"]
    assert cls["sky_corr"] > 0.9
    assert cls["persistence_class"] == "sky_residual", cls


def test_absorption_mode_persistent():
    parsed, fc, ex, cls = _run(make_spec_file([1.0] * 4, absorption=True), mode="absorption")
    assert cls["persistence_class"] == "persistent", cls
    assert all(e["F"] > 0 for e in ex)            # deficit is positive in the searched sense
    # The same file measured in EMISSION mode must not find an emission line.
    _, _, _, cls_em = _run(make_spec_file([1.0] * 4, absorption=True), mode="emission")
    assert cls_em["persistence_class"] in ("absent_in_exposures", "ambiguous", "untestable"), cls_em


def test_absent_in_exposures_when_only_coadd_has_it():
    """A feature in the coadd that is in none of its inputs (coadd-stage artefact).

    This is the mission's central discriminator, so it is checked over an
    ensemble of noise realisations rather than one: with zero injected line the
    per-exposure significance is a standard normal draw, so a single seed can
    scatter up into ``partial`` by chance.  What must hold for EVERY realisation
    is the part that matters scientifically -- the coadd feature is recovered and
    the line is never promoted to a real detection -- and the nominal verdict
    must be ``absent_in_exposures`` in the large majority.
    """
    classes = []
    for seed in range(20):
        _, _, _, cls = _run(make_spec_file([0.0] * 4, coadd_amp=1.0, seed=seed))
        assert cls["coadd_recovered"], (seed, cls)
        assert cls["persistence_class"] not in ("persistent", "persistent_2exp", "transient",
                                                "sky_residual"), (seed, cls)
        assert cls["combined_sig"] < COMBINED_SIG_FLOOR, (seed, cls)
        classes.append(cls["persistence_class"])
    n_absent = classes.count("absent_in_exposures")
    assert n_absent >= 17, classes
    # ... and the default seed the other tests use is one of them.
    _, _, _, cls0 = _run(make_spec_file([0.0] * 4, coadd_amp=1.0))
    assert cls0["persistence_class"] == "absent_in_exposures", cls0
    # The same, down the PRODUCTION path: with the exposure stack and the
    # in-spectrum null that process_spectrum supplies, the verdict must not
    # change.  A calibration that rescued this case would be rescuing the very
    # artefact the channel exists to catch.
    parsed, fc, ex, _ = _run(make_spec_file([0.0] * 4, coadd_amp=1.0))
    st = persist.stack_exposures(parsed, LAM0)
    stack = persist.measure_line(st["wave"], st["flux"], st["ivar"], LAM0,
                                 persist.lsf_fwhm_A(LAM0, "SDSS-DR17"), "emission")
    null = persist.offset_null(persist.sdss_measure_at(parsed, "emission"), LAM0,
                               n=16, lo_A=12.0, hi_A=120.0)
    full = persist.classify_persistence(fc, ex, stack=stack, null=null)
    assert full["coadd_recovered"], full
    assert full["persistence_class"] == "absent_in_exposures", full
    assert full["combined_sig"] < COMBINED_SIG_FLOOR and full["stack_sig"] < 4.0, full
    assert persist.final_verdict({"persistence_class": "absent_in_exposures",
                                  "known_line_match": False}) == "KILLED_absent_in_exposures"


def test_untestable_degrades_honestly():
    # No exposures at all.
    cls = persist.classify_persistence(None, [])
    assert cls["persistence_class"] == "untestable" and cls["n_tested"] == 0
    # Wavelength outside every exposure's coverage.
    parsed, fc, ex, cls2 = _run(make_spec_file([1.0] * 4), lam=3500.0)
    assert cls2["persistence_class"] == "untestable"
    assert "not_covered" in cls2["basis"]
    assert persist.final_verdict({"persistence_class": "untestable", "known_line_match": False}) \
        .startswith("UNTESTED")
    # A masked line window is not a measurement either.
    wave = np.linspace(5000, 7000, 3000)
    flux = np.ones_like(wave)
    ivar = np.ones_like(wave)
    ivar[np.abs(wave - 6000) < 5] = 0.0
    m = persist.measure_line(wave, flux, ivar, 6000.0, 3.0)
    assert not m["testable"] and m["reason"] == "line_masked"


def test_sloped_continuum_does_not_manufacture_a_line():
    """A steep continuum slope plus a one-sided annulus must not fake a feature.

    This reproduces what run 35738206630 hit on real data: every SDSS survivor
    came back with a large NEGATIVE significance (-3 to -9 sigma) where genuine
    absence gives ~0.  These lines sit at 3900-5300 A, on the steep blue
    throughput slope of an SDSS exposure and close enough to the blue end that
    part of the continuum annulus is masked.  A median continuum has no slope
    term, so it returns the level of whichever side survived instead of the
    level under the line, and the difference reads as a line.
    """
    wave = np.linspace(4200.0, 4400.0, 800)
    lam = 4300.0
    fwhm = lam / 2000.0
    rng = np.random.default_rng(11)
    # 2 % per 100 A continuum slope, no line at all.
    for one_sided in (False, True):
        for slope_sign in (+1.0, -1.0):
            cont = 10.0 * (1.0 + slope_sign * 2e-4 * (wave - lam))
            flux = cont + rng.normal(0.0, 0.02, wave.size)
            ivar = np.full(wave.size, 1.0 / 0.02 ** 2)
            if one_sided:
                ivar[wave < lam - 2.0 * fwhm] = 0.0      # blue half of the annulus gone
            m = persist.measure_line(wave, flux, ivar, lam, fwhm, "emission")
            assert m["testable"], (one_sided, slope_sign, m)
            assert abs(m["sig"]) < 3.0, (one_sided, slope_sign, m)
            assert m["cont_one_sided"] is bool(one_sided), m
    # ... and a real line on the same sloped, one-sided continuum is still found
    # with the right equivalent width.
    amp, sigma = 1.0, fwhm / 2.3548
    cont = 10.0 * (1.0 + 2e-4 * (wave - lam))
    flux = cont + _gauss(wave, lam, amp, sigma) + rng.normal(0.0, 0.02, wave.size)
    ivar = np.full(wave.size, 1.0 / 0.02 ** 2)
    ivar[wave < lam - 2.0 * fwhm] = 0.0
    m = persist.measure_line(flux=flux, wave=wave, ivar=ivar, lam0=lam, fwhm_A=fwhm,
                             mode="emission")
    ew_true = amp * sigma * np.sqrt(2 * np.pi) / 10.0
    assert m["sig"] > 5.0, m
    assert abs(m["ew"] - ew_true) / ew_true < 0.3, (m["ew"], ew_true)


def test_two_exposure_and_partial_classes():
    _, _, _, cls = _run(make_spec_file([1.0, 1.0], n_exp=2))
    assert cls["persistence_class"] == "persistent_2exp"
    _, _, _, cls = _run(make_spec_file([1.0, 1.0, 0.0, 0.0, 0.0]))
    assert cls["persistence_class"] in ("partial", "inconsistent_strength"), cls


def test_combine_arms_weights_by_ivar():
    a = {"testable": True, "F": 1.0, "err": 0.1, "cont": 1.0, "sky_peak_sig": 1.0, "n_cosmic": 0}
    b = {"testable": True, "F": 3.0, "err": 1.0, "cont": 1.0, "sky_peak_sig": 8.0, "n_cosmic": 1}
    c = persist.combine_measurements([a, b, {"testable": False}])
    assert abs(c["F"] - (1.0 / 0.01 + 3.0) / (1 / 0.01 + 1)) < 1e-9
    assert c["n_arms"] == 2 and c["sky_peak_sig"] == 8.0 and c["n_cosmic"] == 1


def test_specobjid_decode_and_urls():
    s = (2345 << 50) | (123 << 38) | ((53710 - 50000) << 24) | (26 << 10)
    d = persist.decode_specobjid(s)
    assert d == {"plate": 2345, "fiberid": 123, "mjd": 53710, "run2d": "26"}
    s2 = (7000 << 50) | (5 << 38) | ((56000 - 50000) << 24) | (1302 << 10)
    assert persist.decode_specobjid(s2)["run2d"] == "v5_13_2"
    urls = persist.sdss_spec_urls(7000, 56000, 5, "v5_13_2")
    assert urls[0].endswith("eboss/spectro/redux/v5_13_2/spectra/full/7000/spec-7000-56000-0005.fits")
    urls = persist.sdss_spec_urls(2345, 53710, 123, "26")
    assert "sdss/spectro/redux/26/spectra/full/2345/spec-2345-53710-0123.fits" in urls[0]
    rec = {"specid": s, "plate": None}
    assert persist.sdss_ids_from_record(rec)["source"] == "specobjid_decode"
    assert persist.sdss_ids_from_record({"plate": 1, "mjd": 2, "fiberid": 3, "run2d": "26"})["plate"] == 1
    assert persist.sdss_ids_from_record({}) is None


def test_desi_helpers():
    assert persist.desi_bands_for(4000.0) == ["b"]
    assert persist.desi_bands_for(5790.0) == ["b", "r"]
    assert persist.desi_bands_for(9000.0) == ["z"]
    assert persist.desi_coadd_url("main", "dark", 10016).endswith(
        "healpix/main/dark/100/10016/coadd-main-dark-10016.fits")
    assert persist.desi_frame_urls("cframe", 20210517, 88888, "r", 3)[0].endswith(
        "exposures/20210517/00088888/cframe-r3-00088888.fits")
    # EXP_FIBERMAP parsing.
    t = fits.BinTableHDU.from_columns([
        fits.Column(name="TARGETID", format="K", array=np.array([11, 22, 11])),
        fits.Column(name="NIGHT", format="J", array=np.array([20210501, 20210501, 20210502])),
        fits.Column(name="EXPID", format="J", array=np.array([1, 1, 2])),
        fits.Column(name="TILEID", format="J", array=np.array([5, 5, 5])),
        fits.Column(name="PETAL_LOC", format="I", array=np.array([3, 3, 3])),
        fits.Column(name="FIBER", format="J", array=np.array([1501, 1502, 1501])),
        fits.Column(name="EXPTIME", format="E", array=np.array([900.0, 900.0, 1200.0])),
    ], name="EXP_FIBERMAP")
    hd = fits.HDUList([fits.PrimaryHDU(), t])
    rows = persist.desi_exposure_rows(hd, 11)
    assert [r["expid"] for r in rows] == [1, 2] and rows[1]["exptime"] == 1200.0
    # cframe-like row read via FIBERMAP lookup.
    wave = np.linspace(5800, 7600, 2000)
    flux = np.tile(np.arange(5, dtype=float)[:, None], (1, wave.size))
    cf = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(flux, name="FLUX"),
                       fits.ImageHDU(np.ones_like(flux), name="IVAR"),
                       fits.ImageHDU(np.zeros(flux.shape, np.int32), name="MASK"),
                       fits.ImageHDU(wave, name="WAVELENGTH"),
                       fits.BinTableHDU.from_columns([fits.Column(name="FIBER", format="J",
                                                                  array=np.arange(1500, 1505))],
                                                     name="FIBERMAP")])
    row = persist.desi_read_row(cf, ["FLUX", "IVAR", "MASK"], 1503)
    assert row is not None and float(row["flux"][0]) == 3.0


def test_rest_frame_identification_closes_balmer_leak():
    # The DESI absorption survivors at 3775.2 / 3755.2 A are H11 / H12 at z = +0.001.
    r = identify_rest_frame(3775.2, 0.001014, mode="absorption")
    assert r["known_line_match"] and r["known_line_label"] == "H Balmer"
    assert abs(r["known_line_dv_kms"]) < 50
    r = identify_rest_frame(3755.2, 0.001143, mode="absorption")
    assert r["known_line_match"] and r["known_line_label"] == "H Balmer"
    # Emission mode does not accept a photospheric Ni I line as the explanation.
    r = identify_rest_frame(6765.5, -0.000128, mode="emission")
    assert not r["known_line_match"] and r["nearest_any_label"] == "Ni I"
    # ... but a sky OH line in the observed frame kills either mode.
    r = identify_rest_frame(6300.30 * 1.000277, 0.0, mode="emission")
    assert r["known_line_match"] and r["known_line_frame"] in ("obs", "star")
    # A blank wavelength: no match, nearest reported.
    r = identify_rest_frame(5307.6, 0.00024, mode="emission")
    assert not r["known_line_match"] and r["known_line_label"]


def test_shard_and_reduce_end_to_end(tmp_path, monkeypatch):
    root = tmp_path
    tri = root / "results" / "spectra_triage"
    tri.mkdir(parents=True)
    ids = ["aaaa-1", "bbbb-2", "cccc-3"]
    rows = [
        {"spec_id": ids[0], "wavelength": LAM0, "significance": 12.0, "ra": 10.0, "dec": 1.0,
         "redshift": 0.0, "data_release": "SDSS-DR17", "search_mode": "emission",
         "n_lines_in_spectrum": 1, "simbad_otype": "", "simbad_id": "", "simbad_sptype": ""},
        {"spec_id": ids[1], "wavelength": LAM0, "significance": 9.0, "ra": 20.0, "dec": 2.0,
         "redshift": 0.0, "data_release": "SDSS-DR17", "search_mode": "emission",
         "n_lines_in_spectrum": 1, "simbad_otype": "", "simbad_id": "", "simbad_sptype": ""},
        {"spec_id": ids[2], "wavelength": 8713.6, "significance": 9.0, "ra": 30.0, "dec": 3.0,
         "redshift": -0.00004, "data_release": "DESI-DR1", "search_mode": "absorption",
         "n_lines_in_spectrum": 3, "simbad_otype": "", "simbad_id": "", "simbad_sptype": ""},
    ]
    pd.DataFrame(rows).to_csv(tri / "priority_targets.csv", index=False)
    files = {ids[0]: make_spec_file([1.0] * 4), ids[1]: make_spec_file([4.0, 0.0, 0.0, 0.0])}
    plates = {ids[0]: 1001, ids[1]: 1002}

    def fake_fetch(url, **kw):
        for sid, plate in plates.items():
            if f"spec-{plate:04d}-" in url:
                return files[sid]
        return None
    monkeypatch.setattr(persist, "fetch_bytes", fake_fetch)
    wave = np.linspace(3800, 9200, 4000)
    recs = [
        {"sparcl_id": ids[0], "plate": 1001, "mjd": 55000, "fiberid": 7, "run2d": "26",
         "wavelength": wave, "flux": np.full(wave.size, 10.0), "ivar": np.ones(wave.size)},
        {"sparcl_id": ids[1], "specid": (1002 << 50) | (8 << 38) | (5000 << 24) | (26 << 10),
         "wavelength": wave, "flux": np.full(wave.size, 10.0), "ivar": np.ones(wave.size)},
        # DESI record with no provenance -> untestable, never a verdict.
        {"sparcl_id": ids[2], "wavelength": wave, "flux": np.full(wave.size, 10.0),
         "ivar": np.ones(wave.size)},
    ]
    st = persist.run_shard(root, shard=0, n_shards=1, offline_records=recs, workdir=tmp_path / "w")
    assert st["n_processed"] == 3
    ck = root / "results" / "spectra_persist" / "ckpt"
    assert len(list(ck.glob("*.json"))) == 3
    # Re-running is a no-op thanks to checkpoints.
    st2 = persist.run_shard(root, shard=0, n_shards=1, offline_records=recs, workdir=tmp_path / "w")
    assert st2["n_processed"] == 0 and st2["n_done_before"] == 3
    # ... but a checkpoint written by a SUPERSEDED estimator is not evidence and
    # must be re-measured, never silently inherited.
    stale = ck / f"{ids[0]}.json"
    d = json.loads(stale.read_text())
    d["ckpt_version"] = persist.CKPT_VERSION - 1
    stale.write_text(json.dumps(d))
    st3 = persist.run_shard(root, shard=0, n_shards=1, offline_records=recs, workdir=tmp_path / "w")
    assert st3["n_processed"] == 1 and st3["n_done_before"] == 2
    assert json.loads(stale.read_text())["ckpt_version"] == persist.CKPT_VERSION

    summ = persist.reduce_results(root, do_simbad=False, do_nist=False)
    tab = pd.read_csv(root / "results" / "spectra_persist" / "persistence.csv")
    by = tab.set_index("spec_id")
    assert by.loc[ids[0], "persistence_class"] == "persistent"
    assert by.loc[ids[0], "verdict"] == "ALIVE_persistent_unidentified"
    assert by.loc[ids[0], "identifier"] == "1001-55000-0007"
    assert by.loc[ids[1], "persistence_class"] == "transient"
    assert by.loc[ids[1], "verdict"] == "KILLED_transient"
    assert by.loc[ids[1], "identifier"] == "1002-55000-0008"
    assert by.loc[ids[2], "persistence_class"] == "untestable"
    # The rest-frame check kills the N I 8711 line even though it was untestable.
    assert by.loc[ids[2], "verdict"] == "KILLED_known_line_rest_frame"
    assert summ["n_alive"] == 1 and summ["verdict"] == "PERSISTENT_UNIDENTIFIED_LINES_REMAIN"
    assert summ["persistence_class_counts"]["persistent"] == 1
    ex = json.loads((root / "results" / "spectra_persist" / "exposures.json").read_text())
    assert len(ex[f"{ids[0]}@{LAM0:.1f}"]) == 4


def test_verdict_when_nothing_ran(tmp_path):
    tri = tmp_path / "results" / "spectra_triage"
    tri.mkdir(parents=True)
    pd.DataFrame([{"spec_id": "x", "wavelength": 5000.0, "significance": 9.0, "ra": 1.0,
                   "dec": 1.0, "redshift": 0.0, "data_release": "SDSS-DR17",
                   "search_mode": "emission", "n_lines_in_spectrum": 1, "simbad_otype": "",
                   "simbad_id": "", "simbad_sptype": ""}]).to_csv(tri / "priority_targets.csv",
                                                                    index=False)
    (tmp_path / "results" / "spectra_persist" / "ckpt").mkdir(parents=True)
    s = persist.reduce_results(tmp_path, do_simbad=False, do_nist=False)
    assert s["verdict"] == "NO_DATA_REACHED" and s["n_alive"] == 0


@pytest.mark.parametrize("cls,expect", [
    ("persistent", "ALIVE_persistent_unidentified"),
    ("persistent_2exp", "OPEN_persistent_2exp"),
    ("sky_residual", "KILLED_sky_residual"),
    ("absent_in_exposures", "KILLED_absent_in_exposures"),
    ("partial", "OPEN_partial"),
])
def test_final_verdict_map(cls, expect):
    assert persist.final_verdict({"persistence_class": cls, "known_line_match": False}) == expect
    assert persist.final_verdict({"persistence_class": cls, "known_line_match": True}) \
        == "KILLED_known_line_rest_frame"


def test_second_epoch_kill_requires_sensitivity():
    r = {"persistence_class": "persistent", "known_line_match": False, "second_epoch": "not_seen",
         "mean_F": 10.0, "other_best_err_rel": 1.0}
    assert persist.final_verdict(r) == "KILLED_second_epoch_absent"
    r["other_best_err_rel"] = 5.0            # other epoch too noisy to have seen it
    assert persist.final_verdict(r) == "ALIVE_persistent_unidentified"


# ---------------------------------------------------------------------------
# Attributing a coadd/exposure disagreement: the stack of the exposure HDUs
# ---------------------------------------------------------------------------

def _measure_stack(parsed, lam=LAM0, mode="emission"):
    st = persist.stack_exposures(parsed, lam)
    assert st is not None, "no stack built from the exposure HDUs"
    m = persist.measure_line(st["wave"], st["flux"], st["ivar"], lam,
                             persist.lsf_fwhm_A(lam, "SDSS-DR17"), mode)
    return st, m


def test_stack_of_exposures_does_not_show_a_coadd_only_line():
    """The attribution test.  A feature in the coadd and in none of its inputs
    must be absent from the inverse-variance stack of those inputs too --
    otherwise a coadd/exposure disagreement could be blamed on the per-exposure
    measurement rather than on the data."""
    parsed, fc, _ex, _cls = _run(make_spec_file([0.0] * 4, coadd_amp=1.0))
    st, m = _measure_stack(parsed)
    assert st["n_used"] >= 4
    assert fc["testable"] and fc["sig"] > 5.0          # the coadd has the feature
    assert m["testable"] and abs(m["sig"]) < 3.0       # the stack of its inputs does not


def test_stack_of_exposures_reproduces_a_real_line():
    """The control: a line that IS in every exposure must come back out of the
    stack at the coadd's strength, so a null from the stack means something."""
    parsed, fc, _ex, _cls = _run(make_spec_file([1.0] * 4))
    _st, m = _measure_stack(parsed)
    assert m["testable"] and m["sig"] > 5.0
    assert abs(m["ew"] - fc["ew"]) < 0.3 * abs(fc["ew"])


def test_wave_lag_recovers_an_injected_wavelength_offset():
    """A coadd carries a heliocentric correction the native exposure frames do
    not; a lag of order the line window would move a real line into the
    continuum annulus and read out as a deficit in every exposure at once."""
    rng = np.random.default_rng(11)
    _lc, wc = _grid(6600.0, 6950.0)
    f = 10.0 + rng.normal(0, 0.3, wc.size)
    for lam in (6700.0, 6765.5, 6820.0, 6880.0):
        f = f - _gauss(wc, lam, 3.0, 1.4)
    shift = 1.5
    got = persist.wave_lag({"wave": wc, "flux": f},
                           {"wave": wc - shift, "flux": f}, LAM0)
    assert abs(got + shift) < 0.15
    same = persist.wave_lag({"wave": wc, "flux": f}, {"wave": wc, "flux": f}, LAM0)
    assert abs(same) < 0.1


def test_offset_null_is_centred_on_zero_on_a_clean_spectrum():
    """The control that makes an "absent in the exposures" verdict mean
    something: re-run at wavelengths with no line, the same estimator must read
    ~0 sigma.  A real run that comes back at -5 sigma here would be measuring
    its own bias."""
    parsed, _fc, _ex, _cls = _run(make_spec_file([1.0] * 4))
    # Offsets stay inside the r1 arm so the control is not just "not covered".
    null = persist.offset_null(persist.sdss_measure_at(parsed, "emission"), LAM0,
                               n=16, lo_A=12.0, hi_A=120.0)
    assert null["n_measured"] >= 12
    assert abs(null["combined_sig_median"]) < 2.0
    assert null["frac_below_minus2"] < 0.35


def test_offset_null_detects_an_estimator_that_is_biased_everywhere():
    """And it must actually fire when the bias is there: a spectrum whose
    exposures sit below their own coadd at EVERY wavelength reads negative at
    random offsets, which is how a bias is told apart from an absent line."""
    data = make_spec_file([0.0] * 4)
    with fits.open(io.BytesIO(data)) as hd:
        parsed = persist.parse_sdss_spec(hd)
    # Give the exposures a continuum that curves upward everywhere.  A straight
    # line fitted across the annulus then sits ABOVE the data at every window
    # centre, by -f''/2 times the annulus second moment -- a deficit at every
    # wavelength, which is what the SDSS runs looked like.
    for e in parsed["exposures"]:
        e["flux"] = e["flux"] + 1.2e-4 * (e["wave"] - LAM0) ** 2
    null = persist.offset_null(persist.sdss_measure_at(parsed, "emission"), LAM0,
                               n=16, lo_A=12.0, hi_A=120.0)
    assert null["combined_sig_median"] < -2.0
    assert null["frac_below_minus2"] > 0.5


def _bias_the_exposures(parsed, curvature=1.2e-4):
    """Upward-curving continuum: a straight line fitted across the annulus sits
    above the data at every window centre, so the estimator reads a deficit
    everywhere -- the shape the real SDSS blue frames produced."""
    for e in parsed["exposures"]:
        e["flux"] = e["flux"] + curvature * (e["wave"] - LAM0) ** 2
    return parsed


def test_null_calibration_stops_a_biased_estimator_calling_a_line_absent():
    """With the estimator reading several sigma negative everywhere, an absent
    line must still come out at ~0 sigma once the spectrum's own null is
    subtracted -- otherwise every survivor is 'killed' by the estimator."""
    data = make_spec_file([0.0] * 4)
    with fits.open(io.BytesIO(data)) as hd:
        parsed = _bias_the_exposures(persist.parse_sdss_spec(hd))
    fc, ex = persist.sdss_exposure_measurements(parsed, LAM0, "emission")
    null = persist.offset_null(persist.sdss_measure_at(parsed, "emission"), LAM0,
                               n=16, lo_A=12.0, hi_A=120.0)
    raw = persist.classify_persistence(fc, ex)
    cal = persist.classify_persistence(fc, ex, null=null)
    assert raw["combined_sig"] < -2.0                     # the bias, uncorrected
    assert abs(cal["combined_sig"]) < 2.0, cal            # and corrected away
    assert cal["null_calibrated"] and cal["null_exposure_bias_sig"] < -0.5
    assert abs(cal["combined_sig_raw"] - raw["combined_sig"]) < 1e-6


def test_null_calibration_keeps_a_real_line_under_the_same_bias():
    """And it must not throw the signal out with the bias: the same curved
    continuum with a real line in every exposure still reads persistent."""
    data = make_spec_file([1.0] * 4)
    with fits.open(io.BytesIO(data)) as hd:
        parsed = _bias_the_exposures(persist.parse_sdss_spec(hd))
    fc, ex = persist.sdss_exposure_measurements(parsed, LAM0, "emission")
    null = persist.offset_null(persist.sdss_measure_at(parsed, "emission"), LAM0,
                               n=16, lo_A=12.0, hi_A=120.0)
    cal = persist.classify_persistence(fc, ex, null=null)
    assert cal["persistence_class"] == "persistent", cal
    assert cal["n_present"] == cal["n_tested"] == 4
    assert cal["combined_sig"] > persist.COMBINED_SIG


def test_stack_only_is_not_reported_as_absent():
    """Exposures too noisy individually, but their own inverse-variance mean
    shows the line: that is a weak line, not an absent one, and calling it
    absent would throw away real data."""
    # Each exposure's error is rescaled to its own empirical scatter, which a
    # single noisy frame overestimates; the stack carries the coadd's ivar.  So
    # the stack legitimately outruns the combination of the exposures.
    ex = [{"testable": True, "F": 0.7, "err": 1.0, "sky_peak_sig": 0.0, "sky_level": 1.0}
          for _ in range(6)]
    coadd = {"testable": True, "F": 6.0, "err": 1.0, "sig": 6.0, "sky_peak_sig": 0.0}
    stack = {"testable": True, "F": 6.0, "err": 1.0, "sig": 6.0, "ew": 0.6}
    assert persist.classify_persistence(coadd, ex)["persistence_class"] == "absent_in_exposures"
    cls = persist.classify_persistence(coadd, ex, stack=stack)
    assert cls["persistence_class"] == "stack_only", cls
    assert persist.final_verdict({"persistence_class": "stack_only",
                                  "known_line_match": False}) == "OPEN_stack_only"


class _FakeSparcl:
    """A SPARCL stand-in serving synthetic control spectra."""

    def __init__(self, lam, amp_obs=0.0, amp_star=0.0, n=30, z=0.0005):
        self.lam, self.amp_obs, self.amp_star, self.n, self.z = lam, amp_obs, amp_star, n, z
        self._grid, self._wave = _grid(6000.0, 8000.0)

    def find(self, outfields=None, constraints=None, limit=None):
        return [{"sparcl_id": f"c{k:03d}", "data_release": "SDSS-DR17",
                 "redshift": self.z, "spectype": "STAR"} for k in range(self.n)]

    def retrieve(self, uuid_list=None, include=None, dataset_list=None):
        rng = np.random.default_rng(17)
        out = []
        for u in uuid_list:
            f = 10.0 + rng.normal(0, 0.05, self._wave.size)
            sig = self.lam / 2000.0 / 2.3548
            if self.amp_obs:
                f = f + _gauss(self._wave, self.lam, self.amp_obs, sig)
            if self.amp_star:
                # Same REST wavelength, placed at this star's own redshift.
                f = f + _gauss(self._wave, self.lam * (1.0 + self.z), self.amp_star, sig)
            out.append({"sparcl_id": u, "data_release": "SDSS-DR17", "wavelength": self._wave,
                        "flux": f, "ivar": np.full(self._wave.size, 1 / 0.05 ** 2)})
        return out

    def get_all_fields(self, *a, **k):
        return ["sparcl_id", "wavelength", "flux", "ivar", "redshift", "data_release", "spectype"]


def test_survey_subclass_normalises_a_simbad_spectral_type():
    """SIMBAD says M1V, the archive says M1.  Asking for M1V returns nothing,
    the query falls back to "any star", and the same-type control silently
    becomes a second copy of the all-stars control -- a clean-looking result
    that means nothing."""
    assert persist.survey_subclass("M1V") == "M1"
    assert persist.survey_subclass("K3III") == "K3"
    assert persist.survey_subclass("dM4e") == "M4"
    assert persist.survey_subclass("M") == "M"
    assert persist.survey_subclass("") == "" and persist.survey_subclass(None) == ""
    assert persist.survey_subclass("WD") == ""


def test_control_sample_separates_observed_frame_from_stellar_frame():
    """A feature at a fixed OBSERVED wavelength (sky, instrument) must light up
    the observed-frame control; one at a fixed REST wavelength (the spectral
    type's own structure) must light up the stellar-frame control.  A candidate
    peculiar to its object lights up neither."""
    lam = 6809.26
    sky_like = persist.control_sample(_FakeSparcl(lam, amp_obs=1.0), "SDSS-DR17", lam,
                                      "emission", z_cand=0.0, n=10)
    assert sky_like["obs_frame"]["frac_ge5"] > 0.8, sky_like
    # The two frames only separate when the velocities differ by more than a
    # resolution element, so the control stars here are put at 3000 km/s.
    star_like = persist.control_sample(_FakeSparcl(lam, amp_star=1.0, z=0.01), "SDSS-DR17",
                                       lam, "emission", z_cand=0.0, n=10)
    assert star_like["star_frame"]["frac_ge5"] > 0.8, star_like
    assert star_like["obs_frame"]["frac_ge3"] < 0.2, star_like
    clean = persist.control_sample(_FakeSparcl(lam), "SDSS-DR17", lam, "emission",
                                   z_cand=0.0, n=10)
    assert clean["obs_frame"]["frac_ge3"] < 0.2 and clean["star_frame"]["frac_ge3"] < 0.2


def test_fit_line_profile_tells_an_unresolved_line_from_a_resolved_one():
    """A monochromatic source is unresolved: its profile IS the LSF.  A feature
    measurably broader than the LSF cannot be a single narrow line, whatever
    else it does -- and the six lines left standing after the first run have
    triage width ratios of 1.09 to 1.47 against a NOMINAL R = 2000, which is
    why the fit has to be against the pipeline's own LSF column."""
    rng = np.random.default_rng(5)
    _lg, w = _grid(6600.0, 7000.0)
    lam, lsf = 6809.26, 6809.26 / 2000.0
    for factor in (1.0, 2.0):
        sigma = factor * lsf / 2.3548
        f = 10.0 + _gauss(w, lam, 1.0, sigma) + rng.normal(0, 0.02, w.size)
        fit = persist.fit_line_profile(w, f, np.full(w.size, 1 / 0.02 ** 2), lam, lsf)
        assert fit["fit_ok"], fit
        ratio = fit["fit_fwhm_A"] / lsf
        assert abs(ratio - factor) < 0.15, (factor, ratio, fit)
        assert abs(fit["fit_dv_kms"]) < 30.0
    # And the measured LSF is taken from the pipeline column when it is served.
    ws = np.full(w.size, 2.0)            # sigma = 2 A  ->  FWHM = 4.71 A
    assert abs(persist.lsf_fwhm_measured(w, ws, lam) - 2.3548 * 2.0) < 1e-6
    assert not np.isfinite(persist.lsf_fwhm_measured(w, None, lam))


class _EpochSparcl(_FakeSparcl):
    """Several epochs at one position, with the line at a chosen strength each."""

    def __init__(self, lam, amps):
        super().__init__(lam)
        self.amps = list(amps)

    def find(self, outfields=None, constraints=None, limit=None):
        return [{"sparcl_id": f"e{k}", "ra": 1.0, "dec": 1.0, "data_release": "SDSS-DR17",
                 "dateobs_center": f"200{k}-01-01"} for k in range(len(self.amps))]

    def retrieve(self, uuid_list=None, include=None, dataset_list=None):
        rng = np.random.default_rng(23)
        out = []
        for u in uuid_list:
            amp = self.amps[int(u[1:])]
            sig = self.lam / 2000.0 / 2.3548
            f = 10.0 + _gauss(self._wave, self.lam, amp, sig) + rng.normal(0, 0.03,
                                                                          self._wave.size)
            out.append({"sparcl_id": u, "data_release": "SDSS-DR17", "wavelength": self._wave,
                        "flux": f, "ivar": np.full(self._wave.size, 1 / 0.03 ** 2)})
        return out


def test_epoch_series_measures_every_epoch_not_just_the_best():
    """`second_epoch` keeps only the strongest detection, which answers "was it
    seen again" and nothing else.  A line of constant strength across years is
    a stable property of the star; one that varies is a different object."""
    lam = 6809.26
    steady = persist.epoch_series(_EpochSparcl(lam, [1.0] * 5), 1.0, 1.0, "e0", lam,
                                  "emission")
    assert steady["n_measured"] == 5 and steady["n_sig_ge4"] == 5
    assert steady["ew_spread_frac"] < 0.15, steady
    varying = persist.epoch_series(_EpochSparcl(lam, [0.2, 1.0, 2.0, 0.3, 1.5]),
                                   1.0, 1.0, "e0", lam, "emission")
    assert varying["n_measured"] == 5
    assert varying["ew_spread_frac"] > 0.3, varying
    assert any(e["is_self"] for e in steady["epochs"])


def test_reduce_refuses_to_overwrite_a_real_summary_with_stale_checkpoints(tmp_path):
    """A reduce arriving after an estimator change holds only superseded
    checkpoints.  Writing its empty summary over a real one would replace a
    measurement with a no-data verdict and commit that back."""
    tri = tmp_path / "results" / "spectra_triage"
    tri.mkdir(parents=True)
    pd.DataFrame([{"spec_id": "x", "wavelength": 5000.0, "significance": 9.0, "ra": 1.0,
                   "dec": 1.0, "redshift": 0.0, "data_release": "SDSS-DR17",
                   "search_mode": "emission", "n_lines_in_spectrum": 1, "simbad_otype": "",
                   "simbad_id": "", "simbad_sptype": ""}]).to_csv(tri / "priority_targets.csv",
                                                                    index=False)
    out = tmp_path / "results" / "spectra_persist"
    ck = out / "ckpt"
    ck.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(
        {"verdict": "PERSISTENT_UNIDENTIFIED_LINES_REMAIN", "n_alive": 6}))
    (ck / "x.json").write_text(json.dumps(
        {"spec_id": "x", "ckpt_version": persist.CKPT_VERSION - 1, "lines": []}))
    s = persist.reduce_results(tmp_path, do_simbad=False, do_nist=False)
    assert s["verdict"] == "PERSISTENT_UNIDENTIFIED_LINES_REMAIN" and s["n_alive"] == 6
    assert "reduce_skipped" in s
    # ... and the file on disk is untouched.
    assert json.loads((out / "summary.json").read_text())["n_alive"] == 6


def test_atmospheric_context_flags_a_telluric_band_and_an_oh_list_gap():
    """Five of the six lines left standing after the first run sit in the red,
    where the hand-kept OH list has gaps and the telluric bands are not listed
    at all.  Both facts have to be on the record next to the candidate."""
    from seti.spectra.linelist import atmospheric_context
    assert atmospheric_context(6967.87)["telluric_band"] == "H2O 7200"
    assert atmospheric_context(7620.0)["telluric_band"] == "O2 A"
    assert atmospheric_context(5000.0)["telluric_band"] == ""
    # 6809 A: no listed OH line within 50 A, but the forest is all around it.
    a = atmospheric_context(6809.26)
    assert a["oh_gap_A"] > 20.0 and a["oh_density_per_100A"] > 0.5
    # The blue is genuinely clear of the OH forest.
    assert atmospheric_context(4200.0)["oh_density_per_100A"] == 0.0


def test_json_safe_keeps_a_pixel_window_but_drops_a_whole_spectrum():
    """The diagnose stage dumps pixel windows under the same key names the bulk
    arrays use; stripping by name alone silently emptied exactly the evidence
    the stage exists to produce."""
    big = np.arange(4000.0)
    out = persist._json_safe({"window": {"wave": [1.0, 2.0], "flux": [3.0, 4.0]},
                              "spectrum": {"wave": big, "flux": big.tolist()}})
    assert out["window"] == {"wave": [1.0, 2.0], "flux": [3.0, 4.0]}
    assert out["spectrum"] == {}
