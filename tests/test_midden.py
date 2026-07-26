"""MIDDEN offline tests: synthetic spectra, RV, injection-recovery, veto, resume.

No network.  A synthetic HARPS-like spectrum generator (continuum + noise +
Gaussian absorption at specified air wavelengths + RV shift) drives:

(a) RV registration recovers injected shifts to < 2 km/s;
(b) a star with the Tc I triplet injected at 3-sigma-per-line is flagged as a
    candidate while 50 clean stars produce zero false positives (fixed seed);
(c) the dummy-control veto: a star with excess at the control wavelengths is
    NOT a candidate even with a strong Tc triplet;
(d) continuum normalization is robust to a sloped continuum;
(e) the batch checkpoint/resume logic skips completed batches (and the FITS
    reader handles both binary-table and image-HDU Phase-3 forms).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seti.midden import lines as L
from seti.midden import measure as M
from seti.midden.acquire import process_corpus

C = M.C_KMS
FE = L.rv_reference_wavelengths()
TC = [ln.wavelength for ln in L.tc_lines()]
DUMMY = [ln.wavelength for ln in L.control_lines()]

_WL = (3820.0, 4450.0)
_DPIX = 0.03
_SIG = 0.06          # Gaussian line sigma (A) ~ HARPS + slow rotator
_NOISE = 0.008


def make_spectrum(rng, rv_kms=0.0, extra_lines=(), noise=_NOISE, slope=0.0):
    """Continuum (optionally sloped) x template Fe I lines x extras + noise."""
    wave = np.arange(_WL[0], _WL[1], _DPIX)
    mid = 0.5 * (_WL[0] + _WL[1])
    flux = 1.0 + slope * (wave - mid) / (_WL[1] - _WL[0])
    lines = [(lam, 0.35 + 0.25 * ((i * 37) % 10) / 10.0, _SIG)
             for i, lam in enumerate(FE)]
    lines += list(extra_lines)
    for lam, depth, sig in lines:
        lam_s = lam * (1.0 + rv_kms / C)
        flux = flux * (1.0 - depth * np.exp(-0.5 * ((wave - lam_s) / sig) ** 2))
    flux = flux + rng.normal(0.0, noise, wave.size)
    return wave, flux


def _core_attenuation(core=0.15):
    """Central-depth attenuation of a Gaussian line averaged over the core."""
    x = np.arange(-core, core + 1e-9, _DPIX)
    return float(np.mean(np.exp(-0.5 * (x / _SIG) ** 2)))


def _depth_sigma(core=0.15):
    n_core = np.arange(-core, core + 1e-9, _DPIX).size
    return _NOISE / np.sqrt(n_core)


# ---------------------------------------------------------------------------
# Line-list invariants (offline part; NIST verification itself is runner-side)
# ---------------------------------------------------------------------------

def test_line_list_invariants():
    assert len(L.tc_lines()) == 3
    assert len(L.radionuclide_lines()) == 5
    assert len(L.control_lines()) == 3
    assert len(FE) >= 8
    assert L.check_control_spacing() == []


# ---------------------------------------------------------------------------
# (a) RV registration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rv", [-40.0, 0.0, 25.3, 90.0])
def test_rv_recovery(rv):
    rng = np.random.default_rng(42)
    wave, flux = make_spectrum(rng, rv_kms=rv)
    est = M.estimate_rv(wave, flux, FE)
    assert est["n_lines"] >= 5
    assert abs(est["rv_kms"] - rv) < 2.0


# ---------------------------------------------------------------------------
# (b) + (c): injection corpus (built once — scoring is cheap, spectra are not)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def injection_corpus():
    rng = np.random.default_rng(1234)
    a = _core_attenuation()
    d3 = 3.0 * _depth_sigma() / a          # central depth -> measured 3 sigma
    rows = []

    def add_star(star, extra, rv):
        wave, flux = make_spectrum(rng, rv_kms=rv, extra_lines=extra)
        meta = {"star": star, "tid": len(rows), "dp_id": f"{star}_ep0",
                "teff": 7500.0 + rng.uniform(-100, 100), "priority": 2,
                "source": "synthetic"}
        rows.extend(M.analyze_arrays(wave, flux, {}, meta))

    for i in range(50):
        add_star(f"clean_{i:02d}", (), rng.uniform(-30, 30))
    add_star("tc_star", [(lam, d3, _SIG) for lam in TC], 12.0)
    add_star("veto_star",
             [(lam, 0.05, _SIG) for lam in TC]
             + [(lam, 0.05, _SIG) for lam in DUMMY], -8.0)
    meas = pd.DataFrame(rows)
    stars, meas_z = M.score_corpus(meas)
    return stars, meas_z


def test_injected_triplet_is_candidate_no_false_positives(injection_corpus):
    stars, meas_z = injection_corpus
    tc_row = stars[stars["star"] == "tc_star"].iloc[0]
    assert bool(tc_row["candidate"]) is True
    assert bool(tc_row["tc_coherent_any"]) or tc_row["max_radio_flagged"] >= 2
    clean = stars[stars["star"].str.startswith("clean_")]
    assert int(clean["candidate"].sum()) == 0, \
        clean[clean["candidate"]]["star"].tolist()
    # the injected Tc lines really carry census excess
    tc_z = meas_z[(meas_z["star"] == "tc_star")
                  & (meas_z["species"] == "Tc I")]["z"].to_numpy(float)
    assert np.all(tc_z >= M.Z_TC_EACH)


def test_dummy_control_veto(injection_corpus):
    stars, meas_z = injection_corpus
    row = stars[stars["star"] == "veto_star"].iloc[0]
    # its Tc excess alone would pass ...
    tc_z = meas_z[(meas_z["star"] == "veto_star")
                  & (meas_z["species"] == "Tc I")]["z"].to_numpy(float)
    assert np.all(tc_z >= M.Z_LINE)
    # ... but the control excess vetoes it
    assert bool(row["any_control_veto"]) is True
    assert bool(row["candidate"]) is False


# ---------------------------------------------------------------------------
# (d) continuum robustness
# ---------------------------------------------------------------------------

def test_continuum_robust_to_slope():
    rng = np.random.default_rng(7)
    d_true = 0.30
    wave, flux = make_spectrum(rng, rv_kms=0.0, noise=0.004, slope=0.3,
                               extra_lines=[(4262.27, d_true, _SIG)])
    a = _core_attenuation()
    m = M.measure_line(wave, flux, 4262.27)
    assert abs(m["depth"] - a * d_true) < 0.02
    for lam in DUMMY:
        md = M.measure_line(wave, flux, lam)
        assert abs(md["depth"]) < 0.01


# ---------------------------------------------------------------------------
# (e) batch checkpoint / resume (+ both FITS forms)
# ---------------------------------------------------------------------------

def _write_fits(path, wave, flux, form):
    from astropy.io import fits

    if form == "table":
        cols = [fits.Column(name="WAVE", format=f"{wave.size}D",
                            array=np.asarray([wave])),
                fits.Column(name="FLUX", format=f"{flux.size}D",
                            array=np.asarray([flux]))]
        hdul = fits.HDUList([fits.PrimaryHDU(),
                             fits.BinTableHDU.from_columns(cols)])
    else:
        h = fits.PrimaryHDU(flux)
        w = wave / 10.0 if form == "image_nm" else wave   # nm-unit variant
        h.header["CRVAL1"] = w[0]
        h.header["CDELT1"] = float(w[1] - w[0])
        h.header["CRPIX1"] = 1.0
        hdul = fits.HDUList([h])
    hdul.writeto(path, overwrite=True)


def test_batch_checkpoint_resume(tmp_path):
    forms = ["table", "image", "image_nm", "table", "image", "table"]
    corpus = pd.DataFrame([
        {"star": f"s{i}", "tid": i, "dp_id": f"dp_{i}", "teff": 7400.0,
         "priority": 2, "source": "synthetic", "instrument_name": "HARPS",
         "t_min": 60000.0 + i, "snr": 100.0, "access_url": f"file://dp_{i}"}
        for i in range(6)])
    calls = {"n": 0}

    def fetch_fn(row, dest):
        calls["n"] += 1
        i = int(row["tid"])
        rng = np.random.default_rng(100 + i)
        wave, flux = make_spectrum(rng, rv_kms=5.0 * i)
        _write_fits(dest, wave, flux, forms[i])

    out = tmp_path / "meas"
    scratch = tmp_path / "scratch"
    meas1 = process_corpus(corpus, out, scratch, batch_size=2, fetch_fn=fetch_fn)
    assert calls["n"] == 6
    assert sorted(p.name for p in out.glob("meas_batch_*.parquet")) == \
        ["meas_batch_0000.parquet", "meas_batch_0001.parquet",
         "meas_batch_0002.parquet"]
    assert not list(scratch.glob("*.fits"))       # process-and-discard held

    meas2 = process_corpus(corpus, out, scratch, batch_size=2, fetch_fn=fetch_fn)
    assert calls["n"] == 6                        # nothing re-fetched
    pd.testing.assert_frame_equal(meas1, meas2)

    (out / "meas_batch_0002.parquet").unlink()    # kill the last batch
    meas3 = process_corpus(corpus, out, scratch, batch_size=2, fetch_fn=fetch_fn)
    assert calls["n"] == 8                        # only that batch re-ran
    assert len(meas3) == len(meas1)

    # both FITS forms (and the nm-unit variant) parsed into consistent frames:
    got = meas1[meas1["role"] != "error"]
    assert got["dp_id"].nunique() == 6
    for i in range(6):
        g = got[got["dp_id"] == f"dp_{i}"]
        assert len(g) == len(L.LINES)
        assert abs(float(g["rv_kms"].iloc[0]) - 5.0 * i) < 2.0


def test_line_flag_rates_empty_and_all_error():
    """An all-failure corpus must yield an empty honesty table, not a KeyError.

    Run 30200517861 crashed in line_flag_rates because every spectrum failed
    to download and the filtered measurement table had no columns at all.
    """
    empty = M.line_flag_rates(pd.DataFrame())
    assert len(empty) == 0 and "wavelength" in empty.columns

    errs = pd.DataFrame([{"star": "s", "role": "error", "error": "boom"}])
    out = M.line_flag_rates(errs)
    assert len(out) == 0 and "flag_rate" in out.columns


def test_process_corpus_reruns_all_error_batches(tmp_path):
    """A checkpoint of pure error rows is re-run, not fossilized.

    The poisoned run left meas_batch parquets containing only role='error'
    rows; artifact-seeded resumes must retry those batches with the fixed
    fetcher rather than silently reusing the failures.
    """
    corpus = pd.DataFrame([
        {"dp_id": f"dp_{i}", "star": f"s{i}", "tid": i, "teff": 6000.0,
         "priority": 1, "source": "test", "instrument_name": "HARPS",
         "t_min": 50000.0 + i, "snr": 10.0, "access_url": "http://x"}
        for i in range(2)])
    out = tmp_path / "meas"
    out.mkdir()
    poisoned = pd.DataFrame([{"star": "s0", "role": "error", "error": "old"}])
    poisoned.to_parquet(out / "meas_batch_0000.parquet", index=False)

    calls = {"n": 0}

    def fetch_fn(row, dest):
        calls["n"] += 1
        raise OSError("still failing")     # failure path is fine; must RE-TRY

    meas = process_corpus(corpus, out, tmp_path / "scratch", batch_size=2,
                          fetch_fn=fetch_fn)
    assert calls["n"] == 2                 # batch was re-run, not reused
    assert (meas["role"] == "error").all() and list(meas["error"]) != ["old", "old"]


# ---------------------------------------------------------------------------
# Deep-dive (HD 217522 epoch audit)
# ---------------------------------------------------------------------------

def _deep_meas(star, dp_id, inst, depths, depth_err=0.005, teff=7000.0):
    """Synthetic per-epoch measurement rows for the three Tc lines + controls."""
    from seti.midden.deepdive import TARGET  # noqa: F401 — import check
    rows = []
    lams = {"Tc I": [(4238.19, depths[0]), (4262.27, depths[1]),
                     (4297.06, depths[2])]}
    for sp, pairs in lams.items():
        for lam, d in pairs:
            rows.append({"star": star, "dp_id": dp_id, "instrument": inst,
                         "species": sp, "wavelength": lam,
                         "role": "radionuclide", "depth": d,
                         "depth_err": depth_err, "teff": teff,
                         "t_min": 50000.0 + hash(dp_id) % 1000})
    for lam in (4152.3, 4222.1):
        rows.append({"star": star, "dp_id": dp_id, "instrument": inst,
                     "species": "DUMMY", "wavelength": lam, "role": "control",
                     "depth": 0.01, "depth_err": depth_err, "teff": teff,
                     "t_min": 50000.0 + hash(dp_id) % 1000})
    return rows


def test_deepdive_stability_flags_variable_lines():
    """A pulsation-modulated blend must fail the constant-depth test."""
    from seti.midden.deepdive import _stability

    rows = []
    for i, d in enumerate([0.02, 0.02, 0.02, 0.02]):        # static line set
        rows += _deep_meas("HD 217522", f"s{i}", "HARPS", [d, 0.03, 0.01])
    static = pd.DataFrame(rows)
    st = _stability(static)
    assert st["4238.19"]["p_constant"] > 0.01                # constant passes

    rows = []
    for i, d in enumerate([0.01, 0.05, 0.01, 0.05]):        # modulating line
        rows += _deep_meas("HD 217522", f"v{i}", "HARPS", [d, 0.03, 0.01])
    varying = pd.DataFrame(rows)
    sv = _stability(varying)
    assert sv["4238.19"]["p_constant"] < 1e-6                # variability caught


def test_deepdive_scoring_end_to_end(tmp_path):
    """Panel percentile + epoch verdicts + report files from synthetic rows."""
    from seti.config import load_config
    from seti.midden.deepdive import score_deepdive

    rows = []
    # Comparison panel: 5 stars x 12 epochs of unremarkable depths.
    rng = np.random.default_rng(3)
    for s in range(5):
        for e in range(12):
            d = 0.010 + 0.002 * rng.standard_normal(3)
            rows += _deep_meas(f"HD {1000+s}", f"c{s}_{e}", "HARPS",
                               list(np.abs(d)))
    # Target: 6 epochs with a consistently deep, coherent triplet.
    for e in range(6):
        rows += _deep_meas("HD 217522", f"t{e}", "HARPS",
                           [0.030, 0.032, 0.031])
    meas = pd.DataFrame(rows)

    cfg = load_config()
    cfg.root = tmp_path
    summary = score_deepdive(cfg, meas)
    assert summary["n_target_epochs"] == 6
    assert summary["target_panel_percentile_tc_quad"] == 1.0
    assert summary["target_epochs_coherent"] >= 1
    # a static deep triplet must PASS stability
    for rec in summary["stability"].values():
        assert rec["p_constant"] > 0.01
    assert (tmp_path / "results" / "midden_deepdive" / "DEEPDIVE.md").exists()
    assert (tmp_path / "results" / "midden_deepdive"
            / "target_epochs.csv").exists()


def test_deepdive_resolution_gate_and_box():
    """R gate trusts em_res_power, whitelists known-R instruments on null."""
    from seti.midden.deepdive import resolution_ok, star_box

    df = pd.DataFrame([
        {"instrument_name": "XSHOOTER", "em_res_power": 9000.0},
        {"instrument_name": "UVES", "em_res_power": np.nan},
        {"instrument_name": "HARPS", "em_res_power": 115000.0},
        {"instrument_name": "MUSE", "em_res_power": np.nan},
    ])
    ok = resolution_ok(df)
    assert list(ok) == [False, True, True, False]

    box = star_box(345.629, -44.846, 20.0)
    assert box["dec_hi"] - box["dec_lo"] == pytest.approx(2 * 20.0 / 3600.0)
    # RA box must widen by 1/cos(dec)
    assert (box["ra_hi"] - box["ra_lo"]) > (box["dec_hi"] - box["dec_lo"])
