"""Offline tests for LANTERN -- no network (tests/conftest.py enforces it).

Every case synthesises a JWST-like spectral time series (stellar continuum +
line forest + noise, with an eclipse or transit) and drives the same pure
functions the runner uses.  The battery covers: recovery of an injected
planet line and its eclipse-vanishing statistic; rejection of a constant
stellar line, a settling ramp, a cosmic ray, an artefact wavelength and a
recurrent wavelength; the transit-constancy test; honest degradation when the
archive returns nothing; and the ephemeris/phase machinery.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from seti.lantern import run as R
from seti.lantern.line import (
    assess_feature,
    bh_fdr,
    cosmic_ray_driven,
    eclipse_discriminant,
    feature_snr_in_mask,
    is_recurrent,
    known_artefact,
    line_flux_series,
    local_poly_continuum,
    narrow_feature_search,
    recurrent_wavelengths,
    time_average_spectrum,
    transit_consistency,
)
from seti.lantern.phase import (
    Ephemeris,
    eclipse_offset_fraction,
    ephemeris_from_archive_row,
    events_in_window,
    label_integrations,
    timing_uncertainty,
)
from seti.lantern.synth import synthesise_timeseries

LINE_WL = 4.05
_CONF = R.load_lantern_config()


def _labels(s):
    lab = label_integrations(s["times"], s["ephemeris"], _CONF["phase"])
    lab["cadence_days"] = float(np.median(np.diff(s["times"])))
    return lab


def _feature_near(s, lab, wl=LINE_WL, mask=None, clip=5.0):
    m = mask if mask is not None else (lab["out_eclipse"] if lab["phase_class"] in
                                       ("eclipse", "both") else np.ones(len(s["times"]), bool))
    avg = time_average_spectrum(s["flux"], m, s["flux_err"], clip)
    scan = narrow_feature_search(s["wavelength"], avg["spec"], avg["spec_err"], 2.0, _CONF["line"])
    near = [f for f in scan["features"] if abs(f["wavelength"] - wl) < 0.01]
    return (near[0] if near else None), scan, m


def _full(s, feature, lab, mask, artefact=None):
    ser = line_flux_series(s["flux"], feature["left"], feature["right"], s["flux_err"], _CONF["line"])
    disc = tr = None
    if lab["phase_class"] in ("eclipse", "both"):
        disc = eclipse_discriminant(ser["line"], ser["line_err"], ser["cont"], lab, s["times"],
                                    _CONF["discriminant"])
        disc["in_eclipse_spectrum_snr"] = feature_snr_in_mask(
            s["flux"], lab["in_eclipse"], feature["index"], s["flux_err"], 2.0, _CONF["line"])
    if lab["phase_class"] in ("transit", "both"):
        tr = transit_consistency(ser["line"], ser["line_err"], ser["cont"], lab)
    cr = cosmic_ray_driven(s["flux"], mask, feature["left"], feature["right"], 6.0,
                           flux_err=s["flux_err"], cfg=_CONF["line"])
    a = assess_feature(feature, disc, tr, lab["phase_class"], lab, artefact=artefact,
                       cosmic=cr, cfg=_CONF["discriminant"])
    return disc, tr, cr, a


# --- phase / ephemeris -----------------------------------------------------------------
def test_phase_labels_place_eclipse_with_baseline():
    s = synthesise_timeseries(line_amp=0.0)
    lab = _labels(s)
    assert lab["phase_class"] == "eclipse"
    cov = lab["coverage"]
    assert cov["n_in_eclipse"] > 0 and cov["n_out_eclipse"] > cov["n_in_eclipse"]
    assert cov["n_eclipse_contact"] > 0
    assert cov["n_baseline_before_eclipse_ingress"] >= _CONF["phase"]["min_baseline_before_ingress"]
    assert not lab["in_transit"].any()
    # In-eclipse integrations sit strictly inside the contacts.
    e = lab["eclipses"][0]
    t = s["times"]
    assert t[lab["in_eclipse"]].min() > e["t2"] and t[lab["in_eclipse"]].max() < e["t3"]


def test_phase_labels_transit_window():
    s = synthesise_timeseries(line_amp=0.0, centre="transit")
    lab = _labels(s)
    assert lab["phase_class"] == "transit"
    assert lab["in_transit"].sum() > 0 and not lab["in_eclipse"].any()


def test_eccentric_orbit_without_omega_is_phase_unresolved():
    s = synthesise_timeseries(line_amp=0.0)
    eph = s["ephemeris"]
    eph.ecc, eph.omega_deg = 0.3, None
    lab = label_integrations(s["times"], eph, _CONF["phase"])
    assert lab["phase_class"] == "phase_unresolved"
    assert "eccentric_omega_unknown" in lab["notes"]
    frac, reason = eclipse_offset_fraction(0.3, 45.0)
    assert reason == "eccentric_first_order" and abs(frac - 0.5) > 0.05


def test_stale_ephemeris_cannot_place_contacts():
    s = synthesise_timeseries(line_amp=0.0)
    eph = s["ephemeris"]
    eph.period_err = 0.05          # 10 epochs later that is half a day
    assert timing_uncertainty(eph, eph.t0 + 10 * eph.period) > 0.4
    lab = label_integrations(s["times"], eph, _CONF["phase"])
    assert lab["phase_class"] == "phase_unresolved"
    assert any("timing_uncertainty" in n for n in lab["notes"])


def test_observation_starting_inside_eclipse_has_no_baseline():
    # Shift so the window opens after ingress: no pre-ingress baseline -> not eclipse-class.
    s = synthesise_timeseries(line_amp=0.0, centre_shift_h=2.5, window_h=4.0)
    lab = _labels(s)
    assert lab["coverage"]["n_baseline_before_eclipse_ingress"] < _CONF["phase"]["min_baseline_before_ingress"]
    assert lab["phase_class"] != "eclipse"
    assert "eclipse_without_pre_ingress_baseline" in lab["notes"] or lab["coverage"]["n_in_eclipse"] == 0


def test_events_in_window_and_archive_row():
    row = {"pl_name": "X b", "pl_orbper": 2.0, "pl_tranmid": 2460000.0, "pl_trandur": 3.0,
           "pl_orbpererr1": 1e-5, "pl_tranmiderr1": 1e-3, "pl_orbeccen": None,
           "pl_orblper": None, "pl_ratror": 0.1}
    eph = ephemeris_from_archive_row(row)
    assert eph.valid() and abs(eph.duration - 0.125) < 1e-9
    ev = events_in_window(eph, 2460010.9, 2460011.1, "eclipse")
    assert len(ev) == 1 and abs(ev[0]["mid"] - 2460011.0) < 1e-9
    assert ev[0]["ingress_duration"] == pytest.approx(0.0125)
    bad = ephemeris_from_archive_row({"pl_name": "Y b", "pl_orbper": None})
    assert not bad.valid() and "missing_period_or_t0" in bad.notes


# --- detector: recovery ------------------------------------------------------------------
def test_recovers_vanishing_planet_line_as_candidate():
    s = synthesise_timeseries(line_amp=0.02)
    lab = _labels(s)
    f, scan, m = _feature_near(s, lab)
    assert f is not None, "injected line not recovered"
    assert f["snr"] > 6.0 and 0.5 <= f["width_resel"] <= 3.0
    disc, tr, cr, a = _full(s, f, lab, m)
    assert disc["eclipse_vanish_snr"] > 5.0
    assert disc["out_positive_snr"] > 5.0
    assert abs(disc["in_eclipse_sigma"]) < 3.0
    assert disc["in_eclipse_spectrum_snr"] < 2.0          # gone from the in-eclipse spectrum
    assert disc["line_fractional_drop"] > 0.8
    assert abs(disc["continuum_correlation"]) < 0.5
    assert abs(disc["free_step_offset_integrations"]) <= 3
    assert a["tier"] == "candidate" and a["vetoes"] == []
    assert scan["ew_5sigma_limit"] > 0


def test_faint_planet_line_still_recovered():
    s = synthesise_timeseries(line_amp=0.01, seed=3)
    lab = _labels(s)
    f, _, m = _feature_near(s, lab)
    assert f is not None
    disc, _, _, a = _full(s, f, lab, m)
    assert disc["eclipse_vanish_snr"] > 5.0 and a["tier"] in ("candidate", "interest")


def test_late_eclipse_is_timed_at_the_predicted_contact():
    s = synthesise_timeseries(line_amp=0.02, centre_shift_h=-1.8)
    lab = _labels(s)
    f, _, m = _feature_near(s, lab)
    disc, _, _, a = _full(s, f, lab, m)
    assert abs(disc["free_step_offset_integrations"]) <= 3
    assert "drop_not_at_eclipse" not in a["vetoes"] and "ramp_correlated" not in a["vetoes"]
    assert a["tier"] == "candidate"


def test_transit_constancy_for_planet_line():
    s = synthesise_timeseries(line_amp=0.02, centre="transit")
    lab = _labels(s)
    f, _, m = _feature_near(s, lab)
    assert f is not None
    disc, tr, _, a = _full(s, f, lab, m)
    assert disc is None
    assert tr["transit_constancy"] < 3.0
    assert tr["transit_excess_sigma"] < 3.0
    # No eclipse coverage: the best a constant line can be is 'watch', never a candidate.
    assert a["tier"] == "watch" and a["vetoes"] == ["insufficient_phase_coverage"]


# --- detector: rejection ------------------------------------------------------------------
def test_stellar_emission_line_constant_through_eclipse_is_rejected():
    s = synthesise_timeseries(line_amp=0.02, line_vanishes=False)
    lab = _labels(s)
    f, _, m = _feature_near(s, lab)
    assert f is not None                       # it IS a narrow feature...
    disc, _, _, a = _full(s, f, lab, m)
    assert disc["eclipse_vanish_snr"] < 3.0    # ...but it does not vanish
    assert abs(disc["in_eclipse_sigma"]) > 5.0
    assert disc["in_eclipse_spectrum_snr"] > 6.0   # still there in the in-eclipse spectrum
    assert a["tier"] == "none"
    assert "low_snr" in a["vetoes"] and "tracks_continuum" in a["vetoes"]


def test_detector_ramp_mimicking_a_drop_is_rejected():
    # Line decays like a persistence ramp; the eclipse sits late so the
    # out-of-eclipse mean is high and the in-eclipse mean low.
    s = synthesise_timeseries(line_amp=0.02, line_vanishes=False, line_ramp_amp=3.0,
                              ramp_tau=60.0, centre_shift_h=-1.8)
    lab = _labels(s)
    f, _, m = _feature_near(s, lab)
    assert f is not None
    disc, _, _, a = _full(s, f, lab, m)
    assert disc["eclipse_vanish_snr"] > 3.0    # the naive statistic is fooled...
    assert disc["ramp_correlation"] > 0.5
    assert disc["chi2_ramp"] < disc["chi2_step_predicted"]
    assert "ramp_correlated" in a["vetoes"]    # ...the veto is not
    assert a["tier"] == "none"


def test_whole_spectrum_ramp_does_not_manufacture_a_candidate():
    s = synthesise_timeseries(line_amp=0.02, line_vanishes=False, ramp_amp=0.05)
    lab = _labels(s)
    f, _, m = _feature_near(s, lab)
    _, _, _, a = _full(s, f, lab, m)
    assert a["tier"] == "none"


def test_single_integration_cosmic_ray_is_rejected():
    s = synthesise_timeseries(line_amp=0.0, cosmic_ray=(40, 3.0))
    lab = _labels(s)
    f, _, m = _feature_near(s, lab)
    assert f is None, "a one-integration spike must not survive the time average"
    # The explicit veto path: hand the CR pixel to the diagnostic directly.
    j = int(np.argmin(np.abs(s["wavelength"] - LINE_WL)))
    cr = cosmic_ray_driven(s["flux"], m, j, j + 1, 6.0, flux_err=s["flux_err"], cfg=_CONF["line"])
    assert cr["cosmic_ray_driven"] and cr["n_integrations_above_5mad"] >= 1
    fake = {"wavelength": LINE_WL, "snr": 6.5, "fwhm_samples": 2.0, "left": j, "right": j + 1}
    a = assess_feature(fake, {"eclipse_vanish_snr": 6.0}, None, "eclipse", lab, cosmic=cr)
    assert "cosmic_ray_single_integration" in a["vetoes"] and a["tier"] == "none"


def test_single_pixel_spike_is_not_a_line():
    s = synthesise_timeseries(line_amp=0.02, line_sigma_samples=0.2)
    lab = _labels(s)
    f, scan, _ = _feature_near(s, lab)
    assert f is None and scan["counters"]["single_pixel_spike"] >= 1


def test_wide_feature_is_not_a_line():
    s = synthesise_timeseries(line_amp=0.02, line_sigma_samples=6.0)
    lab = _labels(s)
    f, scan, _ = _feature_near(s, lab)
    assert f is None
    assert scan["counters"]["too_wide"] >= 1 or scan["counters"]["unbounded"] >= 1


def test_feature_at_artefact_wavelength_is_vetoed():
    s = synthesise_timeseries(line_amp=0.02)
    lab = _labels(s)
    f, _, m = _feature_near(s, lab)
    prof = R.instrument_profile(_CONF, "NIRSPEC", "G395H")
    assert prof["mode"] == "G395H" and prof["R"] == 2700
    # 4.05 um is not in the G395H table...
    assert known_artefact(f["wavelength"], prof["artefacts"], prof["edge_tolerance_um"]) is None
    # ...the detector gap is.
    gap = known_artefact(3.75, prof["artefacts"], prof["edge_tolerance_um"])
    assert gap is not None and "gap" in gap["note"]
    _, _, _, a = _full(s, f, lab, m, artefact=gap)
    assert "known_artefact_wavelength" in a["vetoes"] and a["tier"] == "none"


def test_feature_adjacent_to_nan_gap_is_rejected():
    s = synthesise_timeseries(line_amp=0.02)
    j = int(np.argmin(np.abs(s["wavelength"] - LINE_WL)))
    s["flux"][:, j + 3] = np.nan          # a dead column right beside the line
    lab = _labels(s)
    f, scan, _ = _feature_near(s, lab)
    assert f is None and scan["counters"]["adjacent_to_gap"] >= 1


def test_recurrent_wavelength_across_targets_is_instrumental():
    entries = [{"wavelength": 3.1000 + 0.001 * i, "target": f"star{i}"} for i in range(3)]
    entries.append({"wavelength": 4.05, "target": "star0"})
    bins = recurrent_wavelengths(entries, bin_um=0.004, min_targets=3)
    assert is_recurrent(3.1005, bins) and not is_recurrent(4.05, bins)
    a = assess_feature({"wavelength": 3.1, "fwhm_samples": 2.0},
                       {"eclipse_vanish_snr": 8.0, "out_positive_snr": 9.0, "in_eclipse_sigma": 0.1},
                       None, "eclipse", None, recurrent=True)
    assert "recurrent_across_targets" in a["vetoes"] and a["tier"] == "none"


def test_no_line_gives_no_feature_at_the_wavelength():
    s = synthesise_timeseries(line_amp=0.0)
    lab = _labels(s)
    f, _, _ = _feature_near(s, lab)
    assert f is None


def test_local_poly_continuum_follows_curvature_and_hole():
    x = np.arange(400, dtype=float)
    y = 1.0 + 1e-3 * ((x - 200) / 50) ** 2
    y2 = y.copy()
    y2[200:203] += 0.05
    # The hole keeps the feature's own samples out of their own continuum...
    cont = local_poly_continuum(y2, 31, 4)
    assert np.nanmax(np.abs(cont[200:203] - y[200:203])) < 1e-4
    # ...and with the feature masked (the search's second pass) the whole
    # neighbourhood follows the curvature.
    mask = np.zeros(400, bool)
    mask[198:205] = True
    cont2 = local_poly_continuum(y2, 31, 4, mask=mask)
    assert np.nanmax(np.abs(cont2[150:250] - y[150:250])) < 1e-4


def test_bh_fdr_uses_full_trial_count():
    p = np.array([1e-9, 1e-3, 0.02])
    rej, thr = bh_fdr(p, m_total=100000, alpha=0.05)
    assert rej.tolist() == [True, False, False]
    rej2, _ = bh_fdr(p, m_total=3, alpha=0.05)
    assert rej2.sum() >= 2


# --- orchestration: honest degradation ---------------------------------------------------------
def test_inventory_with_empty_archive_is_no_data_reached(tmp_path):
    inv = R.inventory(tmp_path, _CONF, fetch_planets_fn=lambda: pd.DataFrame(),
                      query_tso_fn=lambda i: pd.DataFrame(),
                      list_products_fn=lambda o: pd.DataFrame())
    assert inv["verdict"] == "NO_DATA_REACHED"
    assert json.loads((tmp_path / "shards.json").read_text()) == []
    planets = pd.DataFrame([{"pl_name": "X b", "hostname": "X", "ra": 10.0, "dec": -5.0,
                             "pl_orbper": 2.0, "pl_tranmid": 2460000.0, "pl_trandur": 3.0}])
    inv2 = R.inventory(tmp_path, _CONF, fetch_planets_fn=lambda: planets,
                       query_tso_fn=lambda i: pd.DataFrame(),
                       list_products_fn=lambda o: pd.DataFrame())
    assert inv2["verdict"] == "NO_DATA_REACHED"
    assert inv2["funnel"]["transiting_planets"] == 1


def test_inventory_matches_and_records_proprietary_products(tmp_path):
    planets = pd.DataFrame([{"pl_name": "X b", "hostname": "X", "ra": 10.0, "dec": -5.0,
                             "pl_orbper": 2.0, "pl_tranmid": 2460000.0, "pl_trandur": 3.0,
                             "pl_ratror": 0.1}])
    obs = pd.DataFrame([{"obsid": "1", "obs_id": "jw01_x", "s_ra": 10.0001, "s_dec": -5.0001,
                         "instrument_name": "NIRSPEC/SLIT", "filters": "G395H", "dataRights": "PUBLIC",
                         "target_name": "X", "t_min": 60000.0, "t_max": 60000.3, "t_exptime": 1e4,
                         "calib_level": 2, "proposal_id": "1"},
                        {"obsid": "2", "obs_id": "far", "s_ra": 50.0, "s_dec": 5.0,
                         "instrument_name": "MIRI/SLITLESS", "filters": "P750L", "dataRights": "PUBLIC",
                         "target_name": "Z", "t_min": 60000.0, "t_max": 60000.3, "t_exptime": 1e4,
                         "calib_level": 2, "proposal_id": "2"}])
    prods = pd.DataFrame([{"parent_obsid": "1", "productFilename": "jw01_x-seg001_nrs1_x1dints.fits",
                           "dataURI": "mast:JWST/a", "size": 1000, "calib_level": 2, "dataRights": "PUBLIC"},
                          {"parent_obsid": "1", "productFilename": "jw01_x-seg002_nrs1_x1dints.fits",
                           "dataURI": "mast:JWST/b", "size": 1000, "calib_level": 2, "dataRights": "EXCLUSIVE_ACCESS"}])
    prods["exposure_key"] = prods["productFilename"].map(
        __import__("seti.lantern.acquire", fromlist=["exposure_key"]).exposure_key)
    inv = R.inventory(tmp_path, _CONF, fetch_planets_fn=lambda: planets,
                      query_tso_fn=lambda i: obs, list_products_fn=lambda o: prods, n_shards=4)
    assert inv["verdict"] == "INVENTORIED"
    assert inv["funnel"]["observations_matched_to_hosts"] == 1
    assert inv["funnel"]["products_public"] == 1 and inv["funnel"]["products_proprietary_or_unknown"] == 1
    tgt = inv["targets"]["X"]
    assert len(tgt["observations"][0]["exposures"]) == 1          # two segments, one exposure
    assert inv["shards"] == [["X"]]


def test_screen_streams_and_checkpoints_and_assess_summarises(tmp_path):
    """The screen/assess stages with the archive stubbed by the synthesiser."""
    planets = pd.DataFrame([{"pl_name": "X b", "hostname": "X", "ra": 10.0, "dec": -5.0,
                             "pl_orbper": 1.5, "pl_tranmid": 2460000.0, "pl_trandur": 2.0,
                             "pl_ratror": 0.1, "pl_orbeccen": 0.0, "pl_orblper": 90.0}])
    obs = pd.DataFrame([{"obsid": "1", "obs_id": "jw01_x", "s_ra": 10.0, "s_dec": -5.0,
                         "instrument_name": "NIRSPEC/SLIT", "filters": "G395H", "dataRights": "PUBLIC",
                         "target_name": "X", "t_min": 0, "t_max": 0, "t_exptime": 0,
                         "calib_level": 2, "proposal_id": "1"}])
    prods = pd.DataFrame([{"parent_obsid": "1", "productFilename": "jw01_x_nrs1_x1dints.fits",
                           "dataURI": "mast:JWST/a", "size": 10, "calib_level": 2, "dataRights": "PUBLIC"},
                          {"parent_obsid": "1", "productFilename": "jw01_y_nrs1_x1dints.fits",
                           "dataURI": "mast:JWST/b", "size": 10, "calib_level": 2, "dataRights": "PUBLIC"},
                          {"parent_obsid": "1", "productFilename": "jw01_z_nrs1_x1dints.fits",
                           "dataURI": "mast:JWST/c", "size": 10, "calib_level": 2, "dataRights": "PUBLIC"}])
    prods["exposure_key"] = ["jw01_x_nrs1", "jw01_y_nrs1", "jw01_z_nrs1"]
    R.inventory(tmp_path, _CONF, fetch_planets_fn=lambda: planets, query_tso_fn=lambda i: obs,
                list_products_fn=lambda o: prods, n_shards=1)
    synth = {"mast:JWST/a": dict(line_amp=0.02),                       # vanishing line
             "mast:JWST/b": dict(line_amp=0.02, line_vanishes=False),  # stellar line
             "mast:JWST/c": None}                                       # download fails
    made = {}

    def dl(uri, local):
        if synth[uri] is None:
            return "FAILED: simulated"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(b"x")
        made[str(local)] = uri
        return "COMPLETE"

    def rd(path):
        s = synthesise_timeseries(**synth[made[str(path)]])
        s["meta"] = {"INSTRUME": "NIRSPEC", "GRATING": "G395H"}
        s["time_source"] = "int_times_bjd_tdb"
        return s

    log = R.screen(tmp_path, _CONF, shard=0, n_shards=1, download_fn=dl, read_fn=rd,
                   work_dir=tmp_path / "work")
    assert log["counts"]["analysed"] == 2 and log["counts"]["download_failed"] == 1
    cks = sorted((tmp_path / "obs" / "X").glob("*.json"))
    assert len(cks) == 3
    assert not list((tmp_path / "work").glob("*.fits"))     # products deleted after reading
    # Re-running skips everything already checkpointed (resume after a killed shard).
    log2 = R.screen(tmp_path, _CONF, shard=0, n_shards=1, download_fn=dl, read_fn=rd,
                    work_dir=tmp_path / "work")
    assert log2["counts"]["skipped_checkpoint"] == 3 and log2["counts"]["analysed"] == 0
    summary = R.assess(tmp_path, _CONF)
    assert summary["generated_utc"]
    assert summary["verdict"] == "VANISHING_LINE_CANDIDATES_PENDING_VET"
    f = summary["funnel"]
    assert f["exposures_analysed"] == 2 and f["exposure_statuses"]["download_failed"] == 1
    assert f["tiers"]["candidate"] + f["tiers"]["interest"] >= 1
    cands = [c for c in summary["candidates"] if abs(c["wavelength_um"] - LINE_WL) < 0.01]
    assert len(cands) == 1 and cands[0]["exposure_key"] == "jw01_x_nrs1"
    assert cands[0].get("fdr_pass") is True
    assert summary["rejections"]["tracks_continuum"] >= 1
    assert "X" in summary["targets"] and summary["targets"]["X"]["analysed"] == 2
    assert set(summary["verdict_vocabulary"]) == {"NO_DATA_REACHED", "NO_VANISHING_LINE",
                                                  "VANISHING_LINE_CANDIDATES_PENDING_VET",
                                                  "DEGRADED_SOURCE"}


def test_assess_with_no_checkpoints_is_no_data_reached(tmp_path):
    summary = R.assess(tmp_path, _CONF)
    assert summary["verdict"] == "NO_DATA_REACHED"
    assert summary["funnel"]["exposures_analysed"] == 0 and summary["candidates"] == []


def test_screen_without_inventory_is_no_data_reached(tmp_path):
    log = R.screen(tmp_path, _CONF, shard=0, n_shards=1)
    assert log["verdict"] == "NO_DATA_REACHED"


def test_assess_only_phase_unresolved_is_degraded(tmp_path):
    """Data reached but the discriminant could not run anywhere -> DEGRADED_SOURCE."""
    s = synthesise_timeseries(line_amp=0.0, centre="none")
    s["meta"] = {"INSTRUME": "NIRSPEC", "GRATING": "G395H"}
    s["time_source"] = "index_only"
    rec = R.analyse_stack(s, [s["ephemeris"]], _CONF, "X")
    assert rec["phase_class"] == "phase_unresolved"
    rec.update(status="analysed", exposure_key="e1", total_bytes=1)
    R._write_json(tmp_path / "obs" / "X" / "e1.json", rec)
    summary = R.assess(tmp_path, _CONF)
    assert summary["verdict"] == "DEGRADED_SOURCE"


def test_selftest_battery_and_cli(tmp_path):
    out = R.selftest(tmp_path, _CONF)
    assert out["all_as_expected"], out["cases"]
    assert R.main(["selftest", "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "selftest.json").exists()


def test_register_adds_lantern_parser():
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    R.register(sub)
    args = p.parse_args(["lantern", "assess", "--out-dir", "/tmp/x"])
    assert args.command == "lantern" and args.stage == "assess" and callable(args.func)


def test_read_x1dints_synthetic_fits(tmp_path):
    """The real FITS reader on an x1dints-shaped file: one EXTRACT1D per
    integration, INT_TIMES with MJD-based BJD_TDB mid-times, a descending
    wavelength grid (NIRISS-like), and segment concatenation."""
    from astropy.io import fits

    from seti.lantern.acquire import concatenate_segments, exposure_key, read_x1dints

    wl = np.linspace(2.8, 0.9, 50)                         # descending on purpose
    def make(path, n_int, t0):
        hdus = [fits.PrimaryHDU()]
        hdus[0].header.update({"INSTRUME": "NIRISS", "FILTER": "CLEAR", "PUPIL": "GR700XD",
                               "NINTS": n_int, "EXPSTART": t0, "EXPEND": t0 + 0.01 * n_int})
        for i in range(n_int):
            cols = [fits.Column(name="WAVELENGTH", format="D", array=wl),
                    fits.Column(name="FLUX", format="D", array=np.full(50, 100.0 + i)),
                    fits.Column(name="FLUX_ERROR", format="D", array=np.full(50, 1.0))]
            hdus.append(fits.BinTableHDU.from_columns(cols, name="EXTRACT1D"))
        it = fits.BinTableHDU.from_columns(
            [fits.Column(name="integration_number", format="J", array=np.arange(1, n_int + 1)),
             fits.Column(name="int_mid_BJD_TDB", format="D",
                         array=t0 + 0.01 * np.arange(n_int) + 0.005)], name="INT_TIMES")
        hdus.append(it)
        fits.HDUList(hdus).writeto(path, overwrite=True)

    p1, p2 = tmp_path / "jw01-seg001_nis_x1dints.fits", tmp_path / "jw01-seg002_nis_x1dints.fits"
    make(p1, 5, 60000.0)
    make(p2, 4, 60000.05)
    s1, s2 = read_x1dints(p1), read_x1dints(p2)
    assert s1["flux"].shape == (5, 50) and s1["flux_err"].shape == (5, 50)
    assert s1["time_source"] == "int_times_bjd_tdb"
    assert abs(s1["times"][0] - 2460000.505) < 1e-6              # MJD-based -> JD
    assert s1["wavelength"][0] < s1["wavelength"][-1]             # made ascending
    assert s1["flux"][0, 0] == 100.0 and s1["meta"]["INSTRUME"] == "NIRISS"
    assert exposure_key(p1.name) == exposure_key(p2.name) == "jw01_nis"
    cat = concatenate_segments([s2, s1])
    assert cat["flux"].shape == (9, 50) and cat["n_segments"] == 2
    assert np.all(np.diff(cat["times"]) > 0)
    prof = R.instrument_profile(_CONF, s1["meta"]["INSTRUME"], None, s1["meta"]["FILTER"],
                                s1["meta"]["PUPIL"])
    assert prof["mode"] == "SOSS" and prof["R"] == 700
    # A file without INT_TIMES falls back to the header, flagged.
    p3 = tmp_path / "noint_x1dints.fits"
    make(p3, 3, 60001.0)
    with fits.open(p3) as h:
        h2 = fits.HDUList([x for x in h if x.name != "INT_TIMES"])
        h2.writeto(p3, overwrite=True)
    s3 = read_x1dints(p3)
    assert s3["time_source"] == "header_linear" and s3["times"][0] > 2.4e6


def test_full_size_grid_recovers_line_and_rejects_stellar():
    """2048 samples x 800 integrations, the NIRSpec-like case, through analyse_stack."""
    for kw, expect in ((dict(line_amp=0.02), "candidate"),
                       (dict(line_amp=0.02, line_vanishes=False), "none"),
                       (dict(line_amp=0.0), None)):
        s = synthesise_timeseries(n_wl=2048, n_int=800, **kw)
        s["meta"] = {"INSTRUME": "NIRSPEC", "GRATING": "G395H"}
        s["time_source"] = "int_times_bjd_tdb"
        rec = R.analyse_stack(s, [s["ephemeris"]], _CONF, "X")
        assert rec["phase_class"] == "eclipse" and rec["n_scanned"] > 900
        near = [f for f in rec["features"] if abs(f["wavelength"] - LINE_WL) < 0.005]
        got = near[0]["tier_local"] if near else None
        assert got == expect, (kw, got, near[0]["vetoes_local"] if near else None)
        assert len(json.dumps(R._json_safe(rec))) < 200_000       # checkpoint stays small


def test_ephemeris_dataclass_roundtrip():
    e = Ephemeris(name="b", period=1.0, t0=0.0, duration=0.1)
    d = R._json_safe(e)
    assert d["name"] == "b" and d["ecc"] is None
