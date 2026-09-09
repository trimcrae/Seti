"""Offline tests for S41 -- the industrial line (``seti.roman.lines``).

Everything runs on synthesised G150/P127 spectra and dispersed-image cutouts
(``synthesise_spectrum`` / ``synthesise_dispersed_image``): an injected
unresolved line is recovered, flagged and survives a clean ledger; every
rejection rule is tripped by a case built for it; and a product missing a
field yields a PENDING feature, never a survivor (the honesty rule).  No
network (``conftest.py`` guards it).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from seti.roman import lines as L
from seti.roman.schema import Spectrum, json_safe, load_roman_config

CLEAN_NEIGHBOURS = [{"x": 10.0, "y": 1500.0, "mag": 18.0}]


@pytest.fixture(scope="module")
def conf():
    c = load_roman_config()
    assert c.get("config_status") == "loaded"
    return c


def _with_contam(spec: Spectrum) -> Spectrum:
    spec.contam = np.zeros(spec.n)
    return spec


def _screen_clean(spec: Spectrum, conf: dict) -> dict:
    """Screen with every geometric test runnable and clean."""
    return L.screen_spectrum(_with_contam(spec), conf, neighbours=CLEAN_NEIGHBOURS,
                             prior_bright_pixels=[])


def _g150_line(conf, lam=1.55, snr=15.0, seed=0, **kw) -> Spectrum:
    return L.synthesise_spectrum("G150", conf, 20.0, line_um=lam, line_snr=snr, rng=seed, **kw)


# ---------------------------------------------------------------- wavelengths and R

def test_air_to_vacuum_nd_yag():
    assert abs(L.air_to_vacuum_um(1.0641) - 1.0644) <= 1e-4
    arr = L.air_to_vacuum_um(np.array([0.5, 1.0641, 2.0]))
    assert arr.shape == (3,) and np.all(arr > np.array([0.5, 1.0641, 2.0]))


def test_industrial_lines_normalised(conf):
    rows = {r["name"]: r for r in L.industrial_lines(conf)}
    nd = rows["Nd:YAG 1064"]
    assert nd["kind"] == "line" and nd["um_lo"] == nd["um_hi"] == pytest.approx(1.0644)
    er = rows["Er fibre C-band"]
    assert er["kind"] == "band" and er["um_lo"] < 1.55 < er["um_hi"]
    assert all(set(r) == {"name", "um_lo", "um_hi", "kind"} for r in rows.values())


def test_resolving_power_g150_and_p127(conf):
    g = L.synthesise_spectrum("G150", conf, 20.0, rng=0)
    r_g = L.resolving_power(g, conf)
    assert np.interp(1.5, g.wavelength_um, r_g) == pytest.approx(691.5, abs=2.0)
    p = L.synthesise_spectrum("P127", conf, 20.0, rng=0)
    r_p = L.resolving_power(p, conf)
    assert r_p[0] == pytest.approx(80.0) and r_p[-1] == pytest.approx(180.0)
    assert np.all(np.diff(r_p) > 0)
    # The product's own R wins over the config, scalar or per-sample.
    g.resolving_power = 500.0
    assert np.all(L.resolving_power(g, conf) == 500.0)
    g.resolving_power = np.linspace(300.0, 900.0, g.n)
    assert L.resolving_power(g, conf)[0] == 300.0


def test_resolving_power_fallback_has_no_exception(conf):
    s = Spectrum("x", 0.0, 0.0, "UNKNOWN", np.linspace(1.0, 1.5, 200), np.ones(200), np.ones(200) * 0.05)
    r, note = L.resolving_power_with_note(s, conf)
    assert np.all(r == 100.0) and note.startswith("fallback")


def test_lsf_sigma_pix_g150_is_nyquist(conf):
    g = L.synthesise_spectrum("G150", conf, 20.0, rng=0)
    sig = L.lsf_sigma_pix(g, conf)
    # R = 461 lambda per 2 px at the nominal dispersion: FWHM 2 samples everywhere.
    assert np.median(sig) == pytest.approx(2.0 / L.FWHM_PER_SIGMA, abs=0.05)
    assert sig.max() / sig.min() < 1.05


# ---------------------------------------------------------------- injection / recovery

def test_injected_er_band_line_is_found_flagged_and_survives(conf):
    out = _screen_clean(_g150_line(conf, 1.55, 15.0, seed=0), conf)
    assert out["n_features"] == 1 and out["status"] == "survivor"
    f = out["features"][0]
    assert abs(f["lambda_um"] - 1.55) <= 0.002
    assert 0.7 <= f["width_resel"] <= 1.4
    assert f["snr"] >= 8.0
    assert f["industrial_flag"] == "Er fibre C-band"
    assert f["survives"] and f["tier"] == "UNRESOLVED_LINE_SURVIVOR"
    assert f["tests_not_run"] == [] and f["reasons"] == []
    assert all(v is False for v in f["vetoes"].values())
    assert f["continuum_snr"] > 10
    assert f["trace_pixel"] == pytest.approx(512.0 + f["sample_index"])


def test_injected_nd_yag_line_is_flagged(conf):
    out = _screen_clean(_g150_line(conf, 1.0644, 15.0, seed=3), conf)
    assert out["n_features"] == 1
    assert out["features"][0]["industrial_flag"] == "Nd:YAG 1064"
    assert out["features"][0]["survives"]


def test_injection_recovered_on_the_prism(conf):
    s = L.synthesise_spectrum("P127", conf, 20.0, line_um=1.064, line_snr=15.0, rng=1)
    out = _screen_clean(s, conf)
    assert out["n_features"] == 1
    f = out["features"][0]
    assert abs(f["lambda_um"] - 1.064) <= 0.01
    assert f["industrial_flag"] == "Nd:YAG 1064"


def test_injection_recovered_across_seeds(conf):
    found = sum(_screen_clean(_g150_line(conf, 1.55, 15.0, seed=k), conf)["n_survivors"]
                for k in range(6))
    assert found == 6


def test_screen_output_is_json_serialisable(conf):
    out = _screen_clean(_g150_line(conf, 1.55, 15.0, seed=0), conf)
    json.dumps(json_safe(out))


# ---------------------------------------------------------------- rejection rules

def test_resolved_line_is_not_kept(conf):
    s = _g150_line(conf, 1.55, 15.0, seed=3, resolved_width_resel=3.0)
    out = L.screen_spectrum(s, conf)
    assert out["n_features"] == 0
    assert out["funnel"]["rejections"].get("width_out_of_range", 0) >= 1


def test_single_pixel_spike_is_not_kept(conf):
    s = L.synthesise_spectrum("G150", conf, 20.0, rng=4)
    s.flux[400] += 18.0 * s.flux_err[400]
    out = L.screen_spectrum(s, conf)
    assert out["n_features"] == 0
    assert out["funnel"]["counts"]["features_raw"] >= 1
    assert out["funnel"]["rejections"].get("width_out_of_range", 0) >= 1


def test_pa_beta_is_a_known_stellar_line(conf):
    hit, note = L.is_stellar_line(1.28216, conf)
    assert hit and "Pa-beta" in note
    assert L.is_stellar_line(1.55, conf)[0] is False
    out = _screen_clean(_g150_line(conf, 1.28216, 15.0, seed=3), conf)
    assert out["n_features"] == 1
    f = out["features"][0]
    assert f["vetoes"]["known_stellar_line"] is True
    assert not f["survives"] and f["tier"] == "VETOED" and "known_stellar_line" in f["reasons"]
    assert out["status"] == "vetoed"
    assert out["funnel"]["rejections"]["known_stellar_line"] == 1


def test_not_a_point_source_is_a_single_line_emitter_suspect(conf):
    s = _g150_line(conf, 1.55, 15.0, seed=0, is_point_source=False)
    out = _screen_clean(s, conf)
    f = out["features"][0]
    assert f["vetoes"]["possible_high_z_emitter"] is True and not f["survives"]
    hit, note = L.single_line_emitter_suspect(f, s, conf)
    assert hit and "point source" in note


def test_weak_continuum_is_a_single_line_emitter_suspect(conf):
    s = L.synthesise_spectrum("G150", conf, 2.0, line_um=1.55, line_snr=15.0, rng=3)
    out = _screen_clean(s, conf)
    assert out["n_features"] == 1
    f = out["features"][0]
    assert f["continuum_snr"] < 5.0
    assert f["vetoes"]["possible_high_z_emitter"] is True and not f["survives"]


def test_unknown_morphology_is_pending_not_a_survivor(conf):
    s = _g150_line(conf, 1.55, 15.0, seed=0, is_point_source=None)
    out = _screen_clean(s, conf)
    f = out["features"][0]
    assert f["vetoes"]["possible_high_z_emitter"] is None
    assert f["veto_notes"]["possible_high_z_emitter"] == "morphology_unknown"
    assert f["tests_not_run"] == ["possible_high_z_emitter"]
    assert not f["survives"] and f["tier"] == "PENDING_possible_high_z_emitter"


def test_missing_contam_is_pending_not_a_survivor(conf):
    s = _g150_line(conf, 1.55, 15.0, seed=0)
    assert s.contam is None
    out = L.screen_spectrum(s, conf, neighbours=CLEAN_NEIGHBOURS, prior_bright_pixels=[])
    f = out["features"][0]
    assert f["vetoes"]["overlap_contaminated"] is None
    assert f["veto_notes"]["overlap_contaminated"] == "overlap_test_not_run"
    assert f["tests_not_run"] == ["overlap_contaminated"]
    assert not f["survives"] and f["tier"] == "PENDING_overlap_contaminated"
    assert out["status"] == "pending" and out["n_survivors"] == 0 and out["n_pending"] == 1
    assert out["funnel"]["counts"]["pending_overlap_contaminated"] == 1


def test_overlap_veto_fires_on_the_pipeline_contam_estimate(conf):
    s = _with_contam(_g150_line(conf, 1.55, 15.0, seed=0))
    f = L.find_unresolved_lines(s, conf)[0]
    assert L.overlap_contaminated(f, s, conf)[0] is False
    s.contam[f["sample_index"]] = 0.9 * s.flux[f["sample_index"]]
    hit, note = L.overlap_contaminated(f, s, conf)
    assert hit is True and "contam/flux" in note


def test_zeroth_order_on_the_feature_is_contamination(conf):
    s = _with_contam(_g150_line(conf, 1.55, 15.0, seed=0))
    f = L.find_unresolved_lines(s, conf)[0]
    offset = conf["lines"]["zeroth_order_offset_px"]["G150"]
    bad = [{"x": f["trace_pixel"] - offset + 2.0, "y": s.meta["trace_y"] + 1.0, "mag": 15.0}]
    assert L.zeroth_order_contaminated(f, s, bad, conf)[0] is True
    assert L.zeroth_order_contaminated(f, s, CLEAN_NEIGHBOURS, conf)[0] is False
    far = [{"x": f["trace_pixel"] - offset, "y": s.meta["trace_y"] + 40.0, "mag": 15.0}]
    assert L.zeroth_order_contaminated(f, s, far, conf)[0] is False
    hit, note = L.zeroth_order_contaminated(f, s, None, conf)
    assert hit is None and note.startswith("zeroth_order_test_not_run")
    out = L.screen_spectrum(s, conf, neighbours=bad, prior_bright_pixels=[])
    g = out["features"][0]
    assert g["vetoes"]["zeroth_order_contaminated"] is True and not g["survives"]


def test_zeroth_order_needs_trace_geometry(conf):
    s = _g150_line(conf, 1.55, 15.0, seed=0)
    f = L.find_unresolved_lines(s, conf)[0]
    s.meta.pop("trace_y")
    assert L.zeroth_order_contaminated(f, s, CLEAN_NEIGHBOURS, conf)[0] is None
    s.trace_pixel = None
    f = L.find_unresolved_lines(s, conf)[0]
    assert f["trace_pixel"] is None
    assert L.zeroth_order_contaminated(f, s, CLEAN_NEIGHBOURS, conf)[0] is None


def test_persistence_on_the_same_pixel(conf):
    s = _g150_line(conf, 1.55, 15.0, seed=0)
    f = L.find_unresolved_lines(s, conf)[0]
    same = [{"x": f["trace_pixel"] + 0.5, "y": s.meta["trace_y"], "detector": "SCA01"}]
    assert L.persistence_suspect(f, s, same)[0] is True
    other_detector = [{"x": f["trace_pixel"], "y": s.meta["trace_y"], "detector": "SCA07"}]
    assert L.persistence_suspect(f, s, other_detector)[0] is False
    assert L.persistence_suspect(f, s, [(3.0, 3.0)])[0] is False
    hit, note = L.persistence_suspect(f, s, None)
    assert hit is None and note.startswith("persistence_test_not_run")


def test_feature_near_the_edge_is_dropped(conf):
    ref = L.synthesise_spectrum("G150", conf, 20.0, rng=3)
    s = L.synthesise_spectrum("G150", conf, 20.0, line_um=float(ref.wavelength_um[5]),
                              line_snr=15.0, rng=3)
    out = L.screen_spectrum(s, conf)
    assert out["n_features"] == 0
    assert out["funnel"]["rejections"].get("edge", 0) >= 1
    assert all(f["sample_index"] >= 8 for f in out["features"])


def test_clean_spectrum_yields_no_features(conf):
    n_false = sum(L.screen_spectrum(L.synthesise_spectrum("G150", conf, 20.0, rng=k), conf)["n_features"]
                  for k in range(5))
    assert n_false == 0


def test_search_chunks_when_the_lsf_varies(conf):
    s = _g150_line(conf, 1.55, 15.0, seed=2)
    s.resolving_power = np.linspace(300.0, 1500.0, s.n)
    plan = L._chunk_plan(L.lsf_sigma_pix(s, conf), conf["lines"]["continuum_window"])
    assert len(plan) > 1 and plan[0][0] == 0 and plan[-1][1] == s.n
    out = L.screen_spectrum(s, conf)
    assert out["resolving_power_note"] == "spectrum:array"
    assert out["n_features"] == 1 and abs(out["features"][0]["lambda_um"] - 1.55) <= 0.002


# ---------------------------------------------------------------- population level

def _record(conf, seed, lam, origin=512.0):
    s = L.synthesise_spectrum("G150", conf, 20.0, line_um=lam, line_snr=15.0, rng=seed,
                              source_id=f"src{seed}", trace_origin_px=origin)
    return _screen_clean(s, conf)


def test_same_wavelength_on_three_sources_is_recurrent(conf):
    recs = [_record(conf, k, 1.55, origin=100.0 * k) for k in range(3)]
    assert all(r["n_survivors"] == 1 for r in recs)
    a = L.assess_features(recs, conf)
    assert a["n_sources"] == 3 and a["n_survivors_pre_recurrence"] == 3
    assert all(f["vetoes"]["recurrent_wavelength"] is True for f in a["features"])
    assert all(f["vetoes"]["recurrent_pixel"] is False for f in a["features"])
    assert a["n_survivors"] == 0 and a["tier"] == "NO_CANDIDATE"
    assert a["funnel"]["rejections"]["recurrent_wavelength"] == 3
    assert a["recurrent_wavelength_bins_um"]
    # The caller's records are left alone.
    assert "recurrent_wavelength" not in recs[0]["features"][0]["vetoes"]


def test_same_pixel_on_two_sources_is_recurrent(conf):
    r1 = _record(conf, 0, 1.55, origin=0.0)
    p1 = r1["features"][0]["trace_pixel"]
    p2 = _record(conf, 1, 1.30, origin=0.0)["features"][0]["trace_pixel"]
    r2 = _record(conf, 1, 1.30, origin=p1 - p2)
    assert r2["features"][0]["trace_pixel"] == pytest.approx(p1)
    a = L.assess_features([r1, r2], conf)
    assert all(f["vetoes"]["recurrent_pixel"] is True for f in a["features"])
    assert all(f["vetoes"]["recurrent_wavelength"] is False for f in a["features"])
    assert a["n_survivors"] == 0 and a["tier"] == "NO_CANDIDATE"
    assert a["recurrent_pixels"] == [["SCA01", int(round(p1))]]


def test_single_clean_survivor_is_a_candidate_pending_vet(conf):
    a = L.assess_features([_record(conf, 0, 1.55)], conf)
    assert a["tier"] == "INDUSTRIAL_LINE_CANDIDATES_PENDING_VET"
    assert a["n_survivors"] == 1 and a["n_fdr_pass"] == 1
    assert a["industrial_flag_counts"] == {"Er fibre C-band": 1}
    assert len(a["top_survivors"]) == 1 and a["top_survivors"][0]["fdr_pass"]
    assert a["fdr"]["alpha"] == conf["lines"]["fdr_alpha"]


def test_pending_features_are_never_promoted(conf):
    s = _g150_line(conf, 1.55, 15.0, seed=0)          # contam None: overlap test not run
    a = L.assess_features([L.screen_spectrum(s, conf)], conf)
    assert a["tier"] == "NO_CANDIDATE" and a["n_pending"] == 1 and a["n_survivors"] == 0


def test_assess_empty_list(conf):
    a = L.assess_features([], conf)
    assert a["tier"] == "NO_CANDIDATE" and a["n_sources"] == 0 and a["n_features"] == 0
    assert a["top_survivors"] == []
    json.dumps(json_safe(a))


# ---------------------------------------------------------------- 2-D mode

def _disp(lam):
    return (lam - 1.0) / 0.006, 0.0


def test_linear_dispersion_nominal(conf):
    g = L.linear_dispersion("G150", conf)
    assert g(1.0) == (0.0, 0.0)
    assert g(1.93)[0] == pytest.approx(0.93 / 10.9e-4, rel=1e-6)
    assert L.linear_dispersion("P127", conf)(1.80)[0] == pytest.approx(1.05 / 6e-3, rel=1e-6)


def test_dispersed_blob_found_at_the_right_wavelength(conf):
    lams = np.arange(1.0, 1.93, 0.003)
    rng = np.random.default_rng(0)
    img = L.synthesise_dispersed_image((60, 220), (20.0, 30.0), _disp, 1.1, 1.064, 300.0, 3.0, rng)
    hits = L.dispersed_blob_search(img, (20.0, 30.0), _disp, lams, 1.1, conf)
    assert len(hits) == 1
    h = hits[0]
    assert abs(h["lambda_um"] - 1.064) <= 0.004
    assert h["snr"] >= conf["lines"]["blob_snr_min"]
    assert h["compactness"] <= conf["lines"]["blob_compactness_max"]
    assert h["industrial_flag"] == "Nd:YAG 1064"
    assert h["x"] == pytest.approx(20.0 + _disp(h["lambda_um"])[0], abs=1e-9)


def test_dispersed_blob_survives_a_bright_continuum(conf):
    lams = np.arange(1.0, 1.93, 0.003)
    rng = np.random.default_rng(1)
    img = L.synthesise_dispersed_image((60, 220), (20.0, 30.0), _disp, 1.1, 1.55, 300.0, 40.0, rng)
    hits = L.dispersed_blob_search(img, (20.0, 30.0), _disp, lams, 1.1, conf)
    assert len(hits) == 1 and abs(hits[0]["lambda_um"] - 1.55) <= 0.004


def test_no_blob_on_a_star_without_a_line(conf):
    lams = np.arange(1.0, 1.93, 0.003)
    rng = np.random.default_rng(2)
    img = L.synthesise_dispersed_image((60, 220), (20.0, 30.0), _disp, 1.1, None, 0.0, 40.0, rng)
    assert L.dispersed_blob_search(img, (20.0, 30.0), _disp, lams, 1.1, conf) == []
    img = L.synthesise_dispersed_image((60, 220), (20.0, 30.0), _disp, 1.1, None, 0.0, 3.0, rng)
    assert L.dispersed_blob_search(img, (20.0, 30.0), _disp, lams, 1.1, conf, noise=1.0) == []


# ---------------------------------------------------------------- synthesis contract

def test_synthesised_spectrum_carries_the_geometry(conf):
    s = _g150_line(conf, 1.55, 15.0, seed=0)
    assert s.mode == "G150" and s.is_point_source is True and s.contam is None
    assert s.trace_pixel[0] == 512.0 and np.all(np.diff(s.trace_pixel) == 1.0)
    assert s.meta["trace_y"] == 1024.0 and s.detector == "SCA01"
    assert s.meta["injected_um"] == 1.55
    assert s.wavelength_um[0] == pytest.approx(1.00) and s.wavelength_um[-1] == pytest.approx(1.93)
    p = L.synthesise_spectrum("P127", conf, 20.0, rng=0, n=300)
    assert p.n == 300 and p.wavelength_um[0] == pytest.approx(0.75)
