"""The re-vet of tiles run 35740159635, folded into the screen and the assess.

Each rung that retired a candidate by hand in docs/ignition.md 7.4 is now a
rule, and each rule is tested here offline:

* the stratified (magnitude x |beta|) ensemble correction in the screen;
* the W2/W1 rise-colour (dust) test in the pure ladder;
* the approaching-neighbour rung (Gaia proper motions into the 6" beam);
* the optical-trend and Gaia-scatter rungs, with unreachable archives
  recorded as a degradation and never as a pass.
"""

import numpy as np
import pandas as pd
import pytest

from seti.ignition.run import ignition_run, screen_epochs
from seti.ignition.vet import (
    gaia_scatter_verdict,
    neighbour_contamination,
    optical_rise_verdict,
    rise_colour_verdict,
    vet_star,
)
from seti.ignition.vet_online import online_vet
from test_ignition import (
    QueryResult,
    _asu_dead,
    _cone_factory,
    _config_for_tests,
    _FakeGaia,
    _irsa_dead,
)
from test_ignition_revet import _fake_epochs

# --------------------------------------------------------------------------
# 1. stratified ensemble in the screen
# --------------------------------------------------------------------------


def _screen_conf(mode):
    conf = _config_for_tests()
    conf["sensitivity"]["max_stars"] = 2
    conf.setdefault("screen", {})["ensemble_mode"] = mode
    conf["screen"]["stratum_min_stars"] = 20
    return conf


def _parent_for(ep):
    sids = ep["source_id"].unique()
    return pd.DataFrame({"source_id": sids, "ra": 266.0, "dec": 65.0})


def test_global_correction_makes_non_drifting_bright_stars_rise_stratified_does_not():
    """Faint stars drift fainter, bright ones do not: the global median fakes bright risers."""
    ep = _fake_epochs(n_faint=200, n_bright=40, seed=3)
    bright = {str(i) for i in range(200, 240)}
    df_g, _ = screen_epochs(ep, _screen_conf("global"), parent=_parent_for(ep))
    df_s, rep = screen_epochs(ep, _screen_conf("stratified"), parent=_parent_for(ep))
    rising_g = df_g[df_g["source_id"].isin(bright)]["w2_slope_sigma"] > 5
    rising_s = df_s[df_s["source_id"].isin(bright)]["w2_slope_sigma"] > 5
    assert rising_g.mean() > 0.9
    assert rising_s.sum() == 0
    assert rep["ensemble"]["mode"] == "stratified"
    assert rep["ensemble"]["stratified"]["frac_epochs_uncorrected"] == 0.0
    # the drift is still measured and reported
    assert rep["ensemble"]["drift"]["W2"]["n_applied"] > 0


def test_stratified_correction_falls_back_when_strata_are_thin():
    ep = _fake_epochs(n_faint=30, n_bright=5, seed=4)
    df, rep = screen_epochs(ep, _screen_conf("stratified"), parent=None)
    st = rep["ensemble"]["stratified"]
    # 30 faint stars fill their stratum (>= 20); the 5 bright ones do not fill
    # either their stratum or their magnitude bin, so they take the global median
    assert st["frac_epochs_fine"] == pytest.approx(30 / 35, abs=1e-3)
    assert st["frac_epochs_coarse"] == 0.0
    assert st["frac_epochs_global"] == pytest.approx(5 / 35, abs=1e-3)
    assert st["frac_epochs_uncorrected"] == 0.0


# --------------------------------------------------------------------------
# 2. rise colour
# --------------------------------------------------------------------------


def _row(**kw):
    row = {"source_id": "c", "ra": 266.0, "dec": 65.0, "b": 30.0, "parallax_over_error": 40.0,
           "pmra": 40.0, "pmdec": -20.0, "phot_variable_flag": "NOT_AVAILABLE",
           "w1_median": 10.0, "w2_median": 9.9, "frac_nb_gt1": 0.0, "frac_na_gt0": 0.0,
           "n_exp_raw": 300, "cut_ph_qual": 3, "w1w2_2010": 0.02, "teff_gspphot": 5000.0,
           "w1_slope_mag_yr": -0.006, "w1_slope_sigma": 12.0,
           "w2_slope_mag_yr": -0.012, "w2_slope_sigma": 12.0, "w1_rise_mag": 0.06}
    row.update(kw)
    return row


def test_grey_rise_is_stellar_coloured_and_rejected():
    """The fourteen stars of run 35740159635 had W2/W1 = 0.60-1.18."""
    r = _row(w2_slope_mag_yr=-0.0063, w2_slope_sigma=10.8, w1_slope_mag_yr=-0.0069,
             w1_slope_sigma=12.0)
    rc = rise_colour_verdict(r)
    assert rc["status"] == "stellar_coloured"
    assert rc["w2_over_w1_rise"] == pytest.approx(0.913, abs=0.01)
    assert rc["z_vs_dust_threshold"] >= 3.0
    assert vet_star(r).verdict == "rejected_rise_stellar_coloured"


def test_dust_coloured_rise_passes_and_a_noisy_grey_rise_is_not_killed():
    assert rise_colour_verdict(_row())["status"] == "dust_compatible"       # ratio 2.0
    v = vet_star(_row())
    assert v.verdict == "clean_optical_untested" and v.rise_colour == "dust_compatible"
    noisy = _row(w2_slope_mag_yr=-0.006, w2_slope_sigma=5.0, w1_slope_sigma=5.0)
    assert rise_colour_verdict(noisy)["status"] == "dust_compatible"        # 1.0 +- 0.28


def test_rise_colour_untested_without_slopes():
    r = _row()
    r.pop("w2_slope_mag_yr")
    v = vet_star(r)
    assert "rise_colour" in v.untested_checks and v.rise_colour == "untested"


# --------------------------------------------------------------------------
# 3. approaching neighbour
# --------------------------------------------------------------------------


def _pair(dg=1.95, nb_pm=0.0, d0=6.3):
    """Target moving 200 mas/yr west; a neighbour d0 arcsec west of it at 2016.0."""
    tgt = {"ra": 180.0, "dec": 0.0, "pmra": -200.0, "pmdec": 0.0, "phot_g_mean_mag": 13.4}
    nb = pd.DataFrame({"ra": [180.0 - d0 / 3600.0], "dec": [0.0], "pmra": [nb_pm],
                       "pmdec": [0.0], "phot_g_mean_mag": [13.4 + dg]})
    return tgt, nb


def test_lp387_28_geometry_is_an_approaching_neighbour():
    """dG 1.95 closing 6.7" -> 4.6" predicts ~0.05 mag, the observed W1 rise."""
    tgt, nb = _pair()
    rec = neighbour_contamination(tgt, nb, 0.053)
    assert rec["status"] == "approaching_neighbour"
    assert rec["worst"]["sep_t0"] == pytest.approx(6.7, abs=0.05)
    assert rec["worst"]["sep_t1"] == pytest.approx(4.6, abs=0.05)
    assert -0.07 < rec["pred_dmag"] < -0.03


def test_comoving_faint_or_distant_neighbours_are_clear():
    tgt, nb = _pair(nb_pm=-200.0)                      # common proper motion: static
    assert neighbour_contamination(tgt, nb, 0.053)["status"] == "clear"
    tgt, nb = _pair(dg=7.0)                            # 0.16 % of the flux
    assert neighbour_contamination(tgt, nb, 0.053)["status"] == "clear"
    tgt, nb = _pair(d0=25.0)                           # never near the beam
    assert neighbour_contamination(tgt, nb, 0.053)["status"] == "clear"
    assert neighbour_contamination(tgt, nb.iloc[0:0], 0.053)["status"] == "clear"


# --------------------------------------------------------------------------
# 4. optical trend and Gaia scatter
# --------------------------------------------------------------------------


def test_optical_brightening_kills_camera_drift_does_not():
    # 5802068055991339904: ASAS-SN g -37.6 +- 0.6 mmag/yr against W1 -6.9
    b = {"asassn:g": {"status": "OK", "slope_mmag_yr": -37.6, "slope_err_mmag_yr": 0.6,
                      "median_mag": 12.6, "n_used": 803}}
    assert optical_rise_verdict(b, -0.0069)["status"] == "brightening"
    small = {"asassn:g": {"status": "OK", "slope_mmag_yr": -1.5, "slope_err_mmag_yr": 0.2,
                          "median_mag": 12.6, "n_used": 800}}
    assert optical_rise_verdict(small, -0.0069)["status"] == "flat"        # 7.5 sigma, 0.2x
    fade = {"ztf:zr": {"status": "flat", "slope": 0.01, "slope_sigma": 12.0,
                       "median_mag": 14.0, "n": 500}}
    assert optical_rise_verdict(fade, -0.006)["status"] == "fading"


def test_saturated_or_missing_optical_is_untested_and_disagreeing_bands_inconsistent():
    sat = {"ztf:zr": {"status": "brightening", "slope": -0.02, "slope_sigma": -20.0,
                      "median_mag": 11.9, "n": 300}}
    assert optical_rise_verdict(sat, -0.006)["status"] == "untested"
    assert optical_rise_verdict({}, -0.006)["status"] == "untested"
    mixed = {"asassn:V": {"slope_mmag_yr": 13.6, "slope_err_mmag_yr": 1.4, "median_mag": 12.5,
                          "status": "OK"},
             "asassn:g": {"slope_mmag_yr": -25.0, "slope_err_mmag_yr": 1.6, "median_mag": 12.9,
                          "status": "OK"}}
    assert optical_rise_verdict(mixed, -0.0056)["status"] == "inconsistent"


def test_gaia_scatter_verdict():
    assert gaia_scatter_verdict({"a_g_percentile": 99.7, "a_bp_percentile": 99.0,
                                 "a_rp_percentile": 99.3})["status"] == "variable"
    assert gaia_scatter_verdict({"a_g_percentile": 99.3, "a_bp_percentile": 98.7,
                                 "a_rp_percentile": 98.0})["status"] == "quiet"
    assert gaia_scatter_verdict(None)["status"] == "untested"


# --------------------------------------------------------------------------
# 5. the rungs inside assess, with injected archives
# --------------------------------------------------------------------------


def _dead(*a, **k):
    raise RuntimeError("synthetic outage")


def _fetchers(ztf=None, cone=None, asassn=None, scatter=None):
    def _ztf_ok(row):
        return {"status": "OK", "zr": {"status": "flat", "slope": 0.0002, "slope_sigma": 0.5,
                                       "median_mag": 14.0, "n": 400}}

    def _cone_empty(ra, dec, r):
        return pd.DataFrame(columns=["source_id", "ra", "dec", "pmra", "pmdec",
                                     "phot_g_mean_mag"])

    def _scatter_quiet(targets, stars):
        return {"status": "OK", **{str(s): {"a_g_percentile": 50.0, "a_bp_percentile": 50.0,
                                            "a_rp_percentile": 50.0}
                                   for s in targets["source_id"]}}

    return {"ztf": ztf or _ztf_ok, "gaia_cone": cone or _cone_empty,
            "asassn": asassn or (lambda row: {"status": "NO_ROWS"}),
            "gaia_scatter": scatter or _scatter_quiet}


def _run(tmp_path, fetchers, kind="ramp"):
    return ignition_run("all", out_dir=tmp_path, asu_fetch_fn=_asu_dead, irsa_fetch_fn=_irsa_dead,
                        conf=_config_for_tests(), query_fn=_FakeGaia(n=6),
                        cone_fn=_cone_factory(kind), route="cone",
                        upload_fn=lambda *a, **k: QueryResult(label="u", service="irsa",
                                                              status="QUERY_FAILED",
                                                              error="synthetic"),
                        online_vet=True, vet_fetchers=fetchers)


def test_assess_with_every_archive_answering_flat_is_clean(tmp_path):
    rep = _run(tmp_path, _fetchers())
    assert rep["n_candidates"] == 6 and rep["n_clean"] == 6
    assert not any(d.startswith("vet_unreachable") for d in rep["degraded"])
    assert (tmp_path / "vet_online.json").exists()


def test_assess_optical_brightening_rejects_and_grey_rise_never_reaches_it(tmp_path):
    def _ztf_bright(row):
        return {"status": "OK", "zr": {"status": "brightening", "slope": -0.03,
                                       "slope_sigma": -30.0, "median_mag": 14.0, "n": 400}}
    rep = _run(tmp_path, _fetchers(ztf=_ztf_bright))
    assert rep["n_candidates"] == 0
    assert rep["veto_counters"]["vet"]["verdicts"] == {"rejected_optical_not_flat": 6}
    rep2 = _run(tmp_path / "grey", _fetchers(), kind="grey_ramp")
    assert rep2["n_candidates"] == 0
    assert rep2["veto_counters"]["vet"]["verdicts"] == {"rejected_rise_stellar_coloured": 6}


def test_unreachable_archives_are_degradations_not_passes(tmp_path):
    rep = _run(tmp_path, _fetchers(ztf=_dead, cone=_dead, asassn=_dead, scatter=_dead))
    deg = ";".join(rep["degraded"])
    assert "vet_unreachable:gaia_neighbours:6/6" in deg
    assert "vet_unreachable:optical:6/6" in deg
    assert "vet_unreachable:gaia_scatter:6/6" in deg
    assert rep["n_clean"] == 0                        # nothing unchecked is called clean
    assert rep["n_candidates_optical_untested"] == 6
    assert rep["verdict"].startswith("DEGRADED")


def test_online_vet_approaching_neighbour_rejects():
    cands = pd.DataFrame([{"source_id": "t", "ra": 180.0, "dec": 0.0, "pmra": -200.0,
                           "pmdec": 0.0, "phot_g_mean_mag": 13.4, "w1_rise_mag": 0.053,
                           "w1_slope_mag_yr": -0.005}])
    _, nb = _pair()
    cone = pd.concat([pd.DataFrame([{"source_id": "t", "ra": 180.0, "dec": 0.0, "pmra": -200.0,
                                     "pmdec": 0.0, "phot_g_mean_mag": 13.4}]),
                      nb.assign(source_id="n")], ignore_index=True)
    res, unreach = online_vet(cands, None, None,
                              fetchers=_fetchers(cone=lambda ra, dec, r: cone))
    assert "approaching_neighbour" in res["t"]["flags"]
    assert unreach == {"gaia_neighbours": 0, "optical": 0, "gaia_scatter": 0}
    assert np.isfinite(res["t"]["neighbour"]["pred_dmag"])
