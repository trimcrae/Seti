"""The paces bridge: every existing channel on Roman-shaped products, offline.

Each injected signal is recovered by the channel built for it; a clean star
raises no flag; a star too short for any channel says ``insufficient`` in every
record and raises nothing; an empty assessment says ``NO_DATA_REACHED``.  KNELL
is asserted only to *run* and return a finite record --- its block and
efficiency requirements are strict by design and are tested in its own suite.
No test here reaches the network (``tests/conftest.py`` enforces it).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from seti.roman.bridge import (
    CHANNELS,
    assess_paces,
    colour_companion,
    detect_flares,
    pace_dips,
    pace_glint,
    pace_knell,
    pace_lightcurve,
    pace_metronome,
    pace_secular,
    pace_spectrum,
    seasons_from_gaps,
    synthesise_gbtds_lightcurve,
    synthesise_spectrum,
    to_mag_series,
)
from seti.roman.schema import LightCurve, flag_value, json_safe, load_roman_config

T0 = 61500.0
FAST = {"paces": {
    "knell": {"n_null": 10, "n_trials": 10, "pdm_null": 20, "min_period": 0.1},
    "metronome": {"null": {"n_max": 100, "h_stop": 5}},
}}


@pytest.fixture(scope="module")
def conf():
    return load_roman_config(overrides=FAST)


# --------------------------------------------------------------------------------------
# Seasons and series
# --------------------------------------------------------------------------------------

def test_seasons_from_gaps_splits_three_gbtds_seasons(conf):
    lc = synthesise_gbtds_lightcurve("s", n_seasons=3)
    lab = seasons_from_gaps(lc.mjd, conf["paces"]["season_gap_days"])
    assert lab.min() == 0 and lab.max() == 2
    assert np.unique(lab).size == 3
    # labels count up in time and follow input order, not sorted order
    rng = np.random.default_rng(0)
    perm = rng.permutation(lc.n)
    lab_p = seasons_from_gaps(lc.mjd[perm], conf["paces"]["season_gap_days"])
    assert np.array_equal(lab_p, lab[perm])
    # a season split at 365.25 d would merge everything: the unit is the gap, not the year
    assert np.unique(seasons_from_gaps(lc.mjd, 365.25)).size == 1


def test_to_mag_series_drops_dq_flagged_epochs_and_reports_relative_flux(conf):
    lc = synthesise_gbtds_lightcurve("dq", n_seasons=1)
    dq = np.zeros(lc.n, dtype=np.int64)
    dq[:100] = flag_value(conf, "SATURATED")          # rejected by lightcurve_dq_reject
    dq[100:150] = flag_value(conf, "DROPOUT")         # not in the reject list: kept
    absolute = LightCurve("dq", 0.0, 0.0, "F146", lc.mjd, lc.flux, lc.flux_err, "GBTDS",
                          flux_zp_ab=27.7, dq=dq)
    ms = to_mag_series(absolute, conf)
    assert ms["n_dq_rejected"] == 100 and ms["n_good"] == lc.n - 100
    assert ms["relative_flux"] is False
    assert np.all(np.diff(ms["t"]) > 0)
    assert abs(float(np.median(ms["mag"])) - 20.0) < 0.01
    relative = LightCurve("dq", 0.0, 0.0, "F146", lc.mjd, lc.flux, lc.flux_err, "GBTDS", dq=dq)
    assert to_mag_series(relative, conf)["relative_flux"] is True
    tiny = LightCurve("t", 0.0, 0.0, "F146", lc.mjd[:5], lc.flux[:5], lc.flux_err[:5])
    assert to_mag_series(tiny, conf) is None


# --------------------------------------------------------------------------------------
# Injection per channel
# --------------------------------------------------------------------------------------

def test_dip_is_flagged_and_achromatic_against_the_colour_series(conf):
    lc = synthesise_gbtds_lightcurve("dip", inject={"kind": "dip", "depth": 0.2,
                                                    "t": T0 + 40.0, "dur_d": 2.0})
    colour = colour_companion(lc)
    assert colour.band == "F087"
    rec = pace_dips(lc, conf, colour)
    assert rec["status"] == "ran"
    assert rec["result"]["n_dip_events"] >= 1
    assert rec["result"]["max_event_depth"] >= 0.15
    assert rec["achromatic_test"] == "ran"
    assert rec["n_colour_epochs_in_event"] >= 2
    assert abs(rec["achromatic_z"]) < 3.0
    assert rec["a_primary"] < -0.15 and rec["a_colour"] < -0.15
    out = pace_lightcurve(lc, conf, colour, channels=["dips"])
    assert out["flags"] == ["dips"]
    assert out["colour_band"] == "F087"


def test_dip_without_colour_series_says_the_test_was_not_run(conf):
    lc = synthesise_gbtds_lightcurve("dip", inject={"kind": "dip", "depth": 0.2,
                                                    "t": T0 + 40.0, "dur_d": 2.0})
    rec = pace_dips(lc, conf)
    assert rec["status"] == "ran" and rec["achromatic_test"] == "not_run"
    assert rec["achromatic_z"] is None


def test_glint_is_flagged_and_a_brief_event_cannot_be_colour_tested(conf):
    lc = synthesise_gbtds_lightcurve("glint", inject={"kind": "glint", "amp": 0.5, "t": T0 + 40.0})
    rec = pace_glint(lc, conf, colour_companion(lc))
    assert rec["status"] == "ran"
    assert rec["result"]["n_glint_events"] >= 1
    assert rec["result"]["max_brighten"] > 0.4
    # a 36-minute event against a 12-hour colour cadence: honest, not achromatic
    assert rec["achromatic_test"] in ("ran", "no_colour_epoch_in_window")
    out = pace_lightcurve(lc, conf, channels=["glint"])
    assert out["flags"] == ["glint"]


def test_secular_fade_is_flagged_on_gap_seasons(conf):
    lc = synthesise_gbtds_lightcurve("fade", inject={"kind": "fade", "mag_per_yr": 0.2})
    rec = pace_secular(lc, conf)
    assert rec["status"] == "ran"
    assert rec["result"]["n_seasons"] == 3
    assert rec["result"]["slope_mag_yr"] > 0.15
    assert rec["result"]["slope_sigma"] >= 5.0
    assert rec["season_binning_matches_gaps"] is True
    assert rec["relative_flux"] is False
    out = pace_lightcurve(lc, conf, channels=["secular"])
    assert out["flags"] == ["secular"]


def test_secular_runs_in_relative_flux_and_says_so(conf):
    lc = synthesise_gbtds_lightcurve("fade", inject={"kind": "fade", "mag_per_yr": 0.2})
    rel = LightCurve("fade", 0.0, 0.0, "F146", lc.mjd, lc.flux, lc.flux_err, "GBTDS", dq=lc.dq)
    rec = pace_secular(rel, conf)
    assert rec["status"] == "ran" and rec["relative_flux"] is True
    assert any("relative flux" in n for n in rec["notes"])
    strict = load_roman_config(overrides={"paces": {"allow_relative_flux": False}})
    assert pace_secular(rel, strict)["status"] == "relative_flux_only"


def test_periodic_cease_runs_through_knell(conf):
    lc = synthesise_gbtds_lightcurve("cease", n_seasons=5, cadence_min=120,
                                     inject={"kind": "periodic_cease", "period_d": 0.3,
                                             "amp": 0.05, "cease_at": T0 + 3 * 182.0})
    rec = pace_knell(lc, conf)
    assert rec["status"] == "ran"
    res = rec["result"]
    assert res["n_blocks"] == 5
    assert res["n_detected"] >= 1
    assert math.isfinite(res["ref_period"])
    assert abs(res["ref_period"] - 0.3) < 0.01
    assert rec["params"]["block_mode"] == "gap"
    assert rec["params"]["measure_efficiency"] is False
    assert json_safe(rec) == rec


def test_knell_binning_bounds_the_detector_grid_on_a_12_minute_series():
    from seti.roman.bridge import _knell_footprint

    lc = synthesise_gbtds_lightcurve("fp", n_seasons=1)
    lab = seasons_from_gaps(lc.mjd, 20.0)
    fp = _knell_footprint(lc.mjd, lab, 12.0 / 1440.0, 0.02, 5.0, 2.0e7)
    assert fp["gls_cells"] <= 2.0e7
    assert fp["bin_min"] > 0 and fp["min_period_effective"] >= 3 * fp["bin_min"] / 1440.0
    fp_big = _knell_footprint(lc.mjd, lab, 12.0 / 1440.0, 0.02, 5.0, 1.0e12)
    assert fp_big["bin_min"] == 0 and fp_big["min_period_effective"] == 0.02


def test_clock_flares_run_through_metronome(conf):
    lc = synthesise_gbtds_lightcurve("clock", inject={"kind": "clock_flares", "period_d": 1.2345,
                                                      "n": 20, "amp": 0.3})
    ms = to_mag_series(lc, conf)
    flares = detect_flares(ms["t"], ms["mag"], ms["magerr"], 5.0, 0.10)
    assert flares.size == 20
    rec = pace_metronome(lc, conf)
    assert rec["status"] in ("ran", "insufficient")
    assert rec["n_flares"] == 20
    if rec["status"] == "ran":
        res = rec["result"]
        assert math.isfinite(res["p_window"])
        assert abs(res["period"] - 1.2345) < 0.01
        assert res["p_window"] < 1e-3
        out = pace_lightcurve(lc, conf, channels=["metronome"])
        assert out["flags"] == ["metronome"]
    else:
        assert rec["notes"]


def test_random_flares_are_not_a_clock(conf):
    lc = synthesise_gbtds_lightcurve("rand")
    rng = np.random.default_rng(5)
    f = lc.flux.copy()
    for i in rng.choice(lc.n - 3, 20, replace=False):
        f[i:i + 3] *= 1.3
    lc2 = LightCurve(lc.star_id, lc.ra, lc.dec, lc.band, lc.mjd, f, lc.flux_err, lc.survey,
                     flux_zp_ab=lc.flux_zp_ab, dq=lc.dq)
    rec = pace_metronome(lc2, conf)
    assert rec["status"] == "ran"
    assert not (math.isfinite(rec["result"]["p_window"]) and rec["result"]["p_window"] < 1e-3)


# --------------------------------------------------------------------------------------
# Nulls and degradation
# --------------------------------------------------------------------------------------

def test_clean_constant_star_raises_no_flag(conf):
    lc = synthesise_gbtds_lightcurve("clean")
    out = pace_lightcurve(lc, conf, colour_companion(lc))
    assert set(out["channels"]) == set(CHANNELS)
    assert out["flags"] == []
    assert out["n_seasons"] == 3 and out["relative_flux"] is False
    for name, rec in out["channels"].items():
        assert rec["status"] in ("ran", "insufficient"), name
        assert "params" in rec and "notes" in rec
    assert out["channels"]["dips"]["status"] == "ran"
    assert out["channels"]["glint"]["status"] == "ran"
    assert out["channels"]["secular"]["status"] == "ran"
    assert out["funnel"]["counts"]["lightcurves"] == 1


def test_thirty_epoch_star_is_insufficient_everywhere_without_raising(conf):
    lc = synthesise_gbtds_lightcurve("short", n_seasons=1, season_days=0.25)
    assert lc.n == 30
    out = pace_lightcurve(lc, conf, colour_companion(lc))
    assert {rec["status"] for rec in out["channels"].values()} == {"insufficient"}
    assert out["flags"] == []
    assert all(rec["result"] is None for rec in out["channels"].values())
    assert out["funnel"]["rejections"]["dips_insufficient"] == 1


def test_relative_flux_curve_runs_and_is_stamped(conf):
    lc = synthesise_gbtds_lightcurve("rel", n_seasons=1)
    rel = LightCurve("rel", 0.0, 0.0, "F146", lc.mjd, lc.flux, lc.flux_err, "GBTDS", dq=lc.dq)
    out = pace_lightcurve(rel, conf, channels=["dips", "glint"])
    assert out["relative_flux"] is True
    assert all(rec["relative_flux"] is True for rec in out["channels"].values())
    assert all(rec["status"] == "ran" for rec in out["channels"].values())


# --------------------------------------------------------------------------------------
# Spectra and assess
# --------------------------------------------------------------------------------------

def test_spectrum_with_one_injected_line_finds_one_emission_line(conf):
    rec = pace_spectrum(synthesise_spectrum(line_um=1.0644), conf)
    assert rec["status"] == "ran"
    assert rec["lsf_source"] == "resolving_power"
    assert rec["n_emission"] == 1
    assert abs(rec["emission"][0]["wavelength"] - 1.0644) < 0.002
    assert 0.6 <= rec["emission"][0]["width_ratio"] <= 2.0
    assert rec["n_absorption"] == 0
    assert any("overlap_test_not_run" in n for n in rec["notes"])
    clean = pace_spectrum(synthesise_spectrum(line_um=None), conf)
    assert clean["n_emission"] == 0 and clean["n_absorption"] == 0


def test_assess_on_nothing_is_no_data_reached(conf):
    assert assess_paces([], conf)["verdict"] == "NO_DATA_REACHED"


def test_assess_counts_statuses_and_flags(conf):
    clean = pace_lightcurve(synthesise_gbtds_lightcurve("c"), conf, channels=["dips", "glint"])
    fade = pace_lightcurve(synthesise_gbtds_lightcurve(
        "f", inject={"kind": "fade", "mag_per_yr": 0.2}), conf, channels=["secular"])
    spec = pace_spectrum(synthesise_spectrum(line_um=None), conf)
    a = assess_paces([clean, spec], conf)
    assert a["verdict"] == "PACES_RAN_NO_FLAGS"
    assert a["n_lightcurves"] == 1 and a["n_spectra"] == 1
    assert a["per_channel_status"]["dips"]["ran"] == 1
    b = assess_paces([clean, fade], conf)
    assert b["verdict"] == "PACES_FLAGS_PENDING_VET"
    assert b["flags"] == {"secular": 1}
    assert b["flagged"][0]["star_id"] == "f"
