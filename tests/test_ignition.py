"""Offline tests for IGNITION --- an infrared excess being born on an old star.

No network.  Every archive call is injected.  The decisive pair of tests is
``test_injected_linear_ramp_is_a_candidate`` against
``test_step_plus_decay_impact_is_not_a_candidate``: the channel's claim is that
a takeover's resource-acquisition phase rises for a decade while a natural born
excess rises in < 1 yr and decays, and the shape test is what separates them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seti.ignition.acquire import (
    EpochStore,
    acquire_stars,
    clean_frames,
    epochs_to_series,
    exposures_to_epochs,
    positions_at_neowise_epoch,
    reduce_star,
    star_quality,
    upload_query,
)
from seti.ignition.rise import (
    assess_band,
    assess_series,
    assess_star,
    inject_ramp,
    scan_period_grid,
    sensitivity_from_injections,
)
from seti.ignition.run import (
    ignition_run,
    load_ignition_config,
    parse_shard,
    screen_epochs,
    shard_rows,
    stage_assess,
    stage_sample,
)
from seti.ignition.sample import (
    GAIA_TRANSPORTS,
    SHAPES,
    GaiaQueryFailed,
    allwise_predicates,
    build_query,
    fetch_parent,
    gaia_predicates,
    inner_top,
    parallax_shells,
    run_gaia_query,
    select_parent,
)
from seti.ignition.vet import (
    in_star_forming_region,
    optical_flatness,
    summarise,
    vet_star,
)
from seti.vigil.acquire import QueryResult

# --------------------------------------------------------------------------
# Synthetic NEOWISE epoch series
# --------------------------------------------------------------------------
N_EPOCHS = 20          # 2014 -- 2023.5, two visits a year
SIGMA = 0.008          # per-epoch error at W1 ~ 10 (median of ~12 exposures)


def _epochs(n=N_EPOCHS, sigma=SIGMA, seed=0, w1=10.0, w2=9.95):
    r = np.random.default_rng(seed)
    t = 2014.0 + 0.5 * np.arange(n) + r.normal(0.0, 0.02, n)
    e = np.full(n, sigma)
    return {"W1": (t, w1 + r.normal(0.0, sigma, n), e),
            "W2": (t, w2 + r.normal(0.0, sigma, n), e)}


def _apply(series, fn):
    return {b: (t, m + fn(t), e) for b, (t, m, e) in series.items()}


def _impact(series, t0=2016.5, amp=0.3, tau=1.5):
    """A born excess: brightening step at t0 that decays back on tau years."""
    return _apply(series, lambda t: -amp * np.where(t >= t0, np.exp(-(t - t0) / tau), 0.0))


def _edd_like(series, seed=3):
    """Extreme-debris-disk phenomenology: secular decay plus stochastic bursts."""
    r = np.random.default_rng(seed)
    bursts = [(r.uniform(2014.5, 2023.0), r.uniform(0.03, 0.08)) for _ in range(3)]

    def f(t):
        v = 0.12 * (1.0 - np.exp(-(t - 2014.0) / 4.0))
        for tb, a in bursts:
            v -= a * np.exp(-((t - tb) / 0.4) ** 2)
        return v + r.normal(0.0, 0.02, t.size)
    return _apply(series, f)


def _mira_like(series, period_d=330.0, amp=0.6):
    return _apply(series, lambda t: amp * np.sin(2 * np.pi * t * 365.25 / period_d))


# --------------------------------------------------------------------------
# The detector
# --------------------------------------------------------------------------
def test_constant_star_is_not_a_candidate():
    _pb, v = assess_series(_epochs())
    assert v.is_candidate is False
    assert v.verdict in ("NOT_RISING", "IMPULSIVE_SHAPE", "NOT_MONOTONIC", "SCAN_SYSTEMATIC")


def test_injected_linear_ramp_is_a_candidate():
    """A 0.2 mag rise over 10 yr in both bands at the survey noise."""
    pb, v = assess_series(inject_ramp(_epochs(seed=1), 0.2, over_yr=10.0))
    assert v.verdict == "IGNITION_CANDIDATE", (v.reasons, pb["W1"].reasons, pb["W2"].reasons)
    assert v.is_candidate is True
    for b in ("W1", "W2"):
        r = pb[b]
        assert r.slope_mag_yr < 0 and r.slope_sigma > 5.0
        assert r.tau_rise > 0 and r.tau_p < 1e-3
        assert r.rise_mag == pytest.approx(0.19, abs=0.05)
        assert r.rise_sigma > 5.0
        assert r.mono_frac >= 0.6
        assert r.delta_bic > 6.0                      # ramp preferred over step + decay
        assert r.baseline_yr >= 5.0
        assert r.scan_amp_mag < r.rise_mag / 3.0


def test_ramp_with_scan_direction_sinusoid_is_still_a_candidate_and_reported():
    """The NEOWISE scan-direction systematic (P ~ 365 d) must be modelled out, not fatal."""
    s = inject_ramp(_epochs(seed=5), 0.3, over_yr=10.0)
    s = _apply(s, lambda t: 0.03 * np.sin(2 * np.pi * t * 365.25 / 365.0))
    pb, v = assess_series(s)
    assert v.verdict == "IGNITION_CANDIDATE", (pb["W1"].reasons, pb["W2"].reasons)
    for b in ("W1", "W2"):
        assert 345.0 <= pb[b].scan_period_d <= 385.0
        assert pb[b].scan_amp_mag == pytest.approx(0.03, abs=0.015)
        assert pb[b].scan_amp_mag < pb[b].rise_mag / 3.0


def test_scan_sinusoid_dominated_rise_is_rejected():
    """A small drift under a large scan-band sinusoid is the survey, not the star."""
    s = inject_ramp(_epochs(seed=6), 0.04, over_yr=10.0)
    s = _apply(s, lambda t: 0.08 * np.sin(2 * np.pi * t * 365.25 / 365.0))
    pb, v = assess_series(s)
    assert v.is_candidate is False
    assert "scan_systematic" in pb["W1"].reasons or "scan_systematic" in pb["W2"].reasons


def test_step_plus_decay_impact_is_not_a_candidate():
    """The decisive confounder: a born excess (rise < 1 yr, then decay)."""
    pb, v = assess_series(_impact(_epochs(seed=7)))
    assert v.is_candidate is False
    assert v.verdict == "IMPULSIVE_SHAPE"
    for b in ("W1", "W2"):
        assert pb[b].delta_bic < -6.0                 # step + decay decisively preferred
        assert "impulsive_shape" in pb[b].reasons
        # t0 lands on the epoch grid (~0.5 yr steps), so the amplitude is the
        # step's value at that epoch: 0.3 exp(-0.5 / 1.5) = 0.21 if it lands late.
        assert 0.15 < pb[b].step_amp_mag < 0.36
        assert pb[b].step_t0_yr == pytest.approx(2016.5, abs=0.75)


def test_slow_impact_decay_is_still_separated_from_a_ramp():
    """Even a 5-yr decay after an early step is not a sustained rise."""
    pb, v = assess_series(_impact(_epochs(seed=8), t0=2014.7, amp=0.25, tau=5.0))
    assert v.is_candidate is False
    assert v.verdict in ("IMPULSIVE_SHAPE", "FADING", "NOT_RISING")


def test_stochastic_edd_like_variability_is_not_a_candidate():
    _pb, v = assess_series(_edd_like(_epochs(seed=9)))
    assert v.is_candidate is False
    assert v.verdict != "IGNITION_CANDIDATE"


def test_mira_like_periodic_series_is_not_a_candidate():
    pb, v = assess_series(_mira_like(_epochs(seed=11)))
    assert v.is_candidate is False
    assert pb["W1"].tau_p > 1e-3 or "scan_systematic" in pb["W1"].reasons \
        or "not_monotonic" in pb["W1"].reasons


def test_one_band_rise_is_rejected_by_the_two_band_rule():
    base = _epochs(seed=13)
    ramp = inject_ramp(base, 0.2, over_yr=10.0)
    pb, v = assess_series({"W1": ramp["W1"], "W2": base["W2"]})
    assert pb["W1"].rising is True
    assert v.verdict == "ONE_BAND_ONLY"
    assert v.is_candidate is False
    assert any(r.startswith("w2:") for r in v.reasons)


def test_fading_star_is_labelled_fading_not_candidate():
    pb, v = assess_series(inject_ramp(_epochs(seed=14), -0.2, over_yr=10.0))
    assert v.verdict == "FADING"
    assert pb["W1"].slope_sigma < -5.0


def test_short_baseline_and_too_few_epochs_are_named():
    s = inject_ramp(_epochs(n=8, seed=15), 0.3, over_yr=10.0)      # 3.5 yr, 8 epochs
    pb, v = assess_series(s)
    assert v.is_candidate is False
    assert v.verdict in ("INSUFFICIENT_EPOCHS", "SHORT_BASELINE")
    assert "short_baseline" in pb["W1"].reasons or "insufficient_epochs" in pb["W1"].reasons
    assert assess_star(pb["W1"], None).verdict == "INSUFFICIENT_EPOCHS"


def test_accelerating_exponential_rise_is_recovered():
    """The claim is exponential growth; a linear ramp is only the fit, not the model."""
    s = _apply(_epochs(seed=16), lambda t: -0.02 * (np.exp((t - 2014.0) / 3.0) - 1.0))
    pb, v = assess_series(s)
    assert v.verdict == "IGNITION_CANDIDATE", (pb["W1"].reasons, pb["W2"].reasons)
    assert pb["W1"].ramp_order == 2 and pb["W1"].growth_tau_yr == pytest.approx(3.0, abs=1.6)


def test_scan_period_grid_is_confined_to_the_scan_band():
    g = scan_period_grid()
    assert g.min() == pytest.approx(345.0) and g.max() == pytest.approx(385.0)


def test_underestimated_errors_do_not_manufacture_a_slope():
    """Quoted errors 5x too small: the sqrt(chi2_red) inflation must absorb it."""
    t, m, e = _epochs(seed=17)["W1"]
    r = assess_band(t, m, e / 5.0, "W1")
    assert abs(r.slope_sigma) < 5.0
    assert r.rising is False


def test_sensitivity_injections_report_recovery_fractions():
    stars = [_epochs(seed=100 + i) for i in range(12)]
    sens = sensitivity_from_injections(stars, [0.1, 0.2, 0.4], over_yr=10.0)
    assert sens["n_stars"] == 12
    assert sens["amps"]["0.40"]["fraction"] == 1.0
    assert sens["amps"]["0.20"]["fraction"] >= 0.9
    assert sens["amps"]["0.10"]["n"] == 12


# --------------------------------------------------------------------------
# Frame cleaning and epoch binning
# --------------------------------------------------------------------------
def _frames(n_visits=20, n_exp=12, seed=0, ramp=0.0, w1=10.0, w2=9.95, sigma=0.03):
    """Synthetic single exposures: n_visits ~183 d apart, n_exp per visit."""
    r = np.random.default_rng(seed)
    t = np.concatenate([56700.0 + 183.0 * k + np.sort(r.uniform(0.0, 1.5, n_exp))
                        for k in range(n_visits)])
    yr = (t - t.min()) / 365.25
    d = pd.DataFrame({
        "ra": 266.0 + r.normal(0, 0.3 / 3600, t.size),
        "dec": 65.0 + r.normal(0, 0.3 / 3600, t.size),
        "mjd": t,
        "w1mpro": w1 - ramp * yr / 10.0 + r.normal(0, sigma, t.size),
        "w1sigmpro": sigma,
        "w2mpro": w2 - ramp * yr / 10.0 + r.normal(0, sigma, t.size),
        "w2sigmpro": sigma,
        "qual_frame": 10, "saa_sep": 30.0, "moon_masked": "00", "cc_flags": "0000",
        "ph_qual": "AA", "nb": 1, "na": 0})
    return d


def test_clean_frames_drops_bad_frames_and_keeps_a_ledger():
    d = _frames(n_visits=2, n_exp=5)
    d.loc[0, "qual_frame"] = 0
    d.loc[1, "cc_flags"] = "DH00"
    d.loc[2, "moon_masked"] = "10"
    d.loc[3, "ph_qual"] = "AC"
    d.loc[4, "saa_sep"] = 0.0
    out, led = clean_frames(d)
    assert led["n_raw"] == 10 and led["n_clean"] == 5
    assert led["cut_qual_frame"] == 1 and led["cut_cc_flags"] == 1
    assert led["cut_moon_masked"] == 1 and led["cut_ph_qual"] == 1 and led["cut_saa_sep"] == 1


def test_epoch_binning_groups_visits_and_measures_robust_errors():
    d = _frames()
    ep = exposures_to_epochs(d["mjd"], d["w1mpro"], d["w1sigmpro"], gap_days=90.0, min_exp=5)
    assert len(ep) == 20
    assert ep["n_exp"].median() == 12 and (ep["n_exp"] >= 10).all()
    # A 12-exposure visit at 0.03 mag scatter gives ~0.009 mag per epoch.
    assert ep["err"].median() == pytest.approx(0.03 / np.sqrt(12), rel=0.5)
    assert ep["t_yr"].iloc[0] == pytest.approx(2014.12, abs=0.05)   # MJD 56700 = 2014 Feb
    # A sparse visit is dropped rather than entering with an uncalibrated error.
    sparse = d.iloc[:12 * 19 + 3]
    ep2 = exposures_to_epochs(sparse["mjd"], sparse["w1mpro"], sparse["w1sigmpro"])
    assert len(ep2) == 19


def test_reduce_star_end_to_end_recovers_an_injected_ramp():
    ep, rec = reduce_star("s1", _frames(ramp=0.25, seed=2))
    assert rec["n_epochs_w1"] == 20 and rec["n_epochs_w2"] == 20
    assert rec["frac_nb_gt1"] == 0.0 and rec["frac_na_gt0"] == 0.0
    _pb, v = assess_series(epochs_to_series(ep))
    assert v.verdict == "IGNITION_CANDIDATE"


def test_star_quality_counts_deblended_frames():
    d = _frames(n_visits=1, n_exp=10)
    d.loc[:3, "nb"] = 2
    d.loc[:1, "na"] = 1
    q = star_quality(d)
    assert q["frac_nb_gt1"] == pytest.approx(0.4)
    assert q["frac_na_gt0"] == pytest.approx(0.2)


def test_positions_are_propagated_to_the_neowise_epoch():
    """The bug that cost a previous channel a whole run."""
    stars = pd.DataFrame({"source_id": ["a"], "ra": [100.0], "dec": [20.0],
                          "pmra": [500.0], "pmdec": [-500.0]})
    p = positions_at_neowise_epoch(stars)
    assert float(p["dec_mid"].iloc[0] - 20.0) * 3.6e6 == pytest.approx(-1500.0, rel=1e-6)
    assert p["sweep_arcsec"].iloc[0] > 7.0
    assert "TAP_UPLOAD.pos" in upload_query() and "CIRCLE('ICRS', p.ra, p.dec, p.rad)" in \
        upload_query()


# --------------------------------------------------------------------------
# The sample
# --------------------------------------------------------------------------
def test_gaia_query_carries_every_cut_and_the_subsampling():
    """Every science cut is in every shape; only the alias and the level move."""
    for shape, a in (("inner_cone", "gs"), ("inner_cone_postfilter", "gs"), ("flat", "g")):
        q = build_query(plx_lo=3.0, plx_hi=4.0, shard=2, n_shards=8, stride=3, cap=None,
                        shape=shape)
        for frag in (f"{a}.phot_g_mean_mag < 14.5", f"{a}.parallax > 3.0",
                     f"{a}.parallax_over_error > 10.0", f"ABS({a}.b) > 15.0",
                     f"{a}.ruwe < 1.4", f"{a}.phot_variable_flag != 'VARIABLE'",
                     f"{a}.bp_rp > 0.6", f"{a}.bp_rp < 2.5", f"5 * LOG10({a}.parallax) - 10",
                     f"MOD({a}.random_index, 24) = 2", f"{a}.parallax >= 3.0",
                     f"{a}.parallax < 4.0", "allwise_best_neighbour",
                     "allwise_original_valid"):
            assert frag in q, (shape, frag)
        assert "LOWER(" not in q
        assert "TOP" not in q                       # no cap: nothing is truncated anywhere
        # The AllWISE cuts are SQL in two shapes and a pandas post-filter in the third.
        wise = ("w.w1mpro - w.w2mpro < 0.15", "w.w1mpro - w.w2mpro > -0.1", "w.ext_flag = 0",
                "w.cc_flags = '0000'")
        assert all((f in q) is (shape != "inner_cone_postfilter") for f in wise), shape
        assert build_query(count_only=True, shape=shape).startswith("SELECT COUNT(*)")
    qf = build_query(field={"ra": 266.0, "dec": 65.0, "radius_deg": 1.0}, cap=20000)
    assert "CIRCLE('ICRS', 266.0, 65.0, 1.0)" in qf and "SELECT TOP 20000" in qf
    assert parallax_shells()[0] == (3.0, 4.0)


def test_the_inner_shapes_cut_the_cone_before_the_allwise_join():
    """The plan fix: the cone is applied to gaia_source ALONE, then AllWISE is joined."""
    field = {"ra": 266.0, "dec": 65.0, "radius_deg": 1.0}
    assert SHAPES[0] == "inner_cone" and SHAPES[-1] == "flat"
    q = build_query(field=field, cap=5)                      # the default shape
    inner = q[q.index("FROM (") + 6:q.index(") AS g")]
    assert "gaiadr3.gaia_source AS gs" in inner
    assert "CIRCLE('ICRS', 266.0, 65.0, 1.0)" in inner       # the cone is INSIDE
    assert "allwise" not in inner and "w." not in inner      # AllWISE is NOT
    inner_where = inner[inner.index("WHERE "):]
    assert inner_where.index("CIRCLE(") < inner_where.index("gs.phot_g_mean_mag")
    # The join and the w. predicates hang off the sub-select, not off the base table.
    outer = q[q.index(") AS g"):]
    assert "JOIN gaiadr3.allwise_best_neighbour" in outer
    assert outer.index("WHERE w.w1mpro") > outer.index("allwise_original_valid")
    # The flat shape (what timed out) has no sub-select at all.
    assert "FROM (" not in build_query(field=field, cap=5, shape="flat")
    # Identical science: same predicate set in every shape, only the alias differs.
    sets = [{p.replace("gs.", "g.") for p in gaia_predicates(field=field)} | set(
        allwise_predicates()) for _ in SHAPES]
    assert sets[0] == sets[1] == sets[2]
    # The inner TOP only ever bounds a capped query, and never a COUNT(*).
    assert inner_top(None, None) is None
    assert inner_top({}, 5) == 1000
    assert "TOP" not in build_query(field=field, count_only=True)


def _gaia_rows(n=6, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        "source_id": [str(10 + i) for i in range(n)],
        "ra": 266.0 + 0.01 * np.arange(n), "dec": 65.0 + 0.01 * np.arange(n), "b": 30.0,
        "parallax": 10.0, "parallax_over_error": 50.0, "pmra": 60.0, "pmdec": -30.0,
        "ruwe": 1.0, "phot_g_mean_mag": 11.0, "bp_rp": 1.0, "phot_variable_flag": "NOT_AVAILABLE",
        "non_single_star": 0, "teff_gspphot": 5000.0, "random_index": r.permutation(n),
        "w1mpro": 9.5, "w1mpro_error": 0.02, "w2mpro": 9.45, "w2mpro_error": 0.02,
        "w3mpro": 9.4, "w3mpro_error": 0.05, "cc_flags": "0000", "ph_qual": "AAAA",
        "ext_flag": 0, "allwise_sep_arcsec": 0.3, "allwise_n_neighbours": 1,
    })


def test_select_parent_applies_dwarf_photospheric_and_kinematic_rules():
    d = _gaia_rows(6)
    d.loc[0, "parallax"] = 0.5                   # a giant at 2 kpc pretending to be near
    d.loc[1, "w2mpro"] = 9.0                     # already an excess in 2010
    d.loc[2, "w2mpro"] = 9.7                     # negative W1-W2: a blend
    d.loc[3, "phot_variable_flag"] = "VARIABLE"
    d.loc[4, "pmra"] = 5.0
    d.loc[4, "pmdec"] = 0.0                      # slow: young-disc kinematics
    out, c = select_parent(d)
    assert c["n_in"] == 6 and c["n_out"] == 2
    assert c["cut_parallax"] == 1 and c["cut_w1w2_photospheric"] == 2
    assert c["cut_gaia_variable"] == 1
    assert set(out["source_id"]) == {"14", "15"}
    assert bool(out.loc[out["source_id"] == "15", "kinematically_old"].iloc[0]) is True
    assert bool(out.loc[out["source_id"] == "14", "kinematically_old"].iloc[0]) is False
    assert out["abs_g"].iloc[0] == pytest.approx(11.0 + 5 * np.log10(10.0) - 10.0)
    assert out["v_tan_kms"].iloc[1] == pytest.approx(4.74047 * np.hypot(60, 30) / 10.0, rel=1e-3)


class _FakeGaia:
    """Answers COUNT(*) and the sample query from synthetic rows, or fails."""

    def __init__(self, mode="ok", n=6):
        self.mode, self.n, self.queries = mode, n, []

    def __call__(self, adql):
        self.queries.append(adql)
        if self.mode == "fail":
            raise RuntimeError("synthetic archive outage")
        if adql.startswith("SELECT COUNT"):
            return pd.DataFrame({"n": [self.n * 10]})
        if self.mode == "zero":
            return _gaia_rows(0)
        return _gaia_rows(self.n)


def test_fetch_parent_reports_the_denominator_and_subsample_fraction():
    fake = _FakeGaia()
    stars, rep = fetch_parent({"fields": [{"ra": 266.0, "dec": 65.0, "radius_deg": 1.0}]},
                              mode="fields", n_shards=2, query_fn=fake)
    assert rep["status"] == "OK" and len(stars) == 6
    assert rep["parent_count"] == 60 and rep["subsample_fraction"] == pytest.approx(0.1)
    assert any(q.startswith("SELECT COUNT") for q in fake.queries)
    stars2, rep2 = fetch_parent({}, mode="allsky", n_shards=1, cap_per_shard=8, query_fn=fake)
    assert rep2["status"] == "OK"
    assert all(u["stride"] >= 1 for u in rep2["per_unit"])
    assert any("MOD(gs.random_index" in q for q in fake.queries)
    assert rep2["query_shape_used"] == "inner_cone"      # the first shape answered


def test_fetch_parent_separates_failure_from_zero_rows():
    _s, rep = fetch_parent({"fields": [{"ra": 1.0, "dec": 1.0, "radius_deg": 1.0}]},
                           mode="fields", query_fn=_FakeGaia("fail"))
    assert rep["status"] == "QUERY_FAILED" and rep["n_units_failed"] == 1
    _s, rep = fetch_parent({"fields": [{"ra": 1.0, "dec": 1.0, "radius_deg": 1.0}]},
                           mode="fields", query_fn=_FakeGaia("zero"))
    assert rep["status"] == "QUERY_RETURNED_ZERO_ROWS" and rep["n_units_failed"] == 0


# --------------------------------------------------------------------------
# The contaminant ladder
# --------------------------------------------------------------------------
def _cand_row(**kw):
    row = {"source_id": "c", "ra": 266.0, "dec": 65.0, "b": 30.0, "parallax_over_error": 40.0,
           "pmra": 40.0, "pmdec": -20.0, "phot_variable_flag": "NOT_AVAILABLE",
           "w1_median": 10.0, "w2_median": 9.9, "frac_nb_gt1": 0.0, "frac_na_gt0": 0.0,
           "n_exp_raw": 300, "cut_ph_qual": 3, "w1w2_2010": 0.02}
    row.update(kw)
    return row


def test_clean_candidate_without_optical_is_marked_optical_untested():
    v = vet_star(_cand_row())
    assert v.verdict == "clean_optical_untested"
    assert v.optical == "not_checked"
    assert "optical_flatness" in v.untested_checks


def test_extragalactic_kill_is_astrometric():
    """NGC 6447: +1.2 mag over 14 yr, optically flat --- and no parallax."""
    v = vet_star(_cand_row(parallax_over_error=0.8, pmra=0.2, pmdec=0.1, pm_error=0.3))
    assert v.verdict == "rejected_extragalactic"
    # The same object with astrometry is a star.
    assert vet_star(_cand_row(parallax_over_error=12.0)).verdict != "rejected_extragalactic"


def test_star_forming_region_and_plane_are_rejected():
    assert in_star_forming_region(67.0, 25.0, load_ignition_config()["vet"]["star_forming_boxes"]) \
        == "Taurus"
    assert vet_star(_cand_row(ra=67.0, dec=25.0, b=-15.5)).verdict == \
        "rejected_star_forming_region"
    assert vet_star(_cand_row(ra=100.0, dec=0.0, b=4.0)).verdict == "rejected_galactic_plane"


def test_gaia_variable_saturated_deblended_and_poor_photometry_rules():
    assert vet_star(_cand_row(phot_variable_flag="VARIABLE")).verdict == "rejected_gaia_variable"
    assert vet_star(_cand_row(w1_median=7.5)).verdict == "rejected_saturated"
    assert vet_star(_cand_row(frac_nb_gt1=0.6)).verdict == "rejected_deblended"
    assert vet_star(_cand_row(frac_na_gt0=0.5)).verdict == "rejected_deblended"
    assert vet_star(_cand_row(cut_ph_qual=200)).verdict == "rejected_poor_photometry"


def test_already_excess_and_agn_like_colour_flags():
    v = vet_star(_cand_row(w1w2_2010=0.9))
    assert v.verdict == "rejected_already_excess_2010"
    assert "nearest_agn_like" in v.flags


def _optical(kind, n=60, seed=0):
    r = np.random.default_rng(seed)
    t = np.linspace(2018.0, 2024.0, n)
    m = 13.0 + r.normal(0, 0.015, n)
    if kind == "fading":
        m = m + 0.12 * (t - 2018.0)            # R CrB-like decline while the IR rises
    elif kind == "brightening":
        m = m - 0.15 * (t - 2018.0)            # nova / AGB / YSO: the optical leads
    elif kind == "variable":
        m = m + 0.2 * np.sin(2 * np.pi * (t - 2018.0) / 0.9)
    return pd.DataFrame({"t_yr": t, "mag": m, "magerr": 0.015})


def test_optical_hook_flat_fading_brightening():
    assert optical_flatness(*(_optical("flat")[c] for c in ("t_yr", "mag", "magerr")))["status"] \
        == "flat"
    v = vet_star(_cand_row(), optical=_optical("flat"))
    assert v.verdict == "clean" and v.optical == "flat"
    assert "optical_flatness" not in v.untested_checks
    v = vet_star(_cand_row(), optical=_optical("fading"))
    assert v.verdict == "rejected_rcrb_like" and v.optical_slope_sigma > 3.0
    v = vet_star(_cand_row(), optical=_optical("brightening"))
    assert v.verdict == "rejected_optical_not_flat"
    v = vet_star(_cand_row(), optical=_optical("variable"))
    assert v.verdict == "rejected_optical_not_flat"
    v = vet_star(_cand_row(), optical=_optical("flat", n=5))
    assert v.verdict == "clean_optical_untested" and v.optical == "insufficient"


def test_every_vet_rule_has_a_counter():
    rows = [vet_star(_cand_row()), vet_star(_cand_row(w1_median=7.0)),
            vet_star(_cand_row(parallax_over_error=1.0))]
    s = summarise(rows)
    assert s["verdicts"]["clean_optical_untested"] == 1
    assert s["verdicts"]["rejected_saturated"] == 1
    assert s["flags"]["extragalactic"] == 1


# --------------------------------------------------------------------------
# Orchestration: injected archives, checkpointing, honest degradation
# --------------------------------------------------------------------------
def _cone_factory(kind="constant", status="OK"):
    """Per-star cone fetcher keyed on the star's RA so each star is distinct."""

    def f(ra, dec, pmra=0.0, pmdec=0.0, radius_arcsec=2.5):
        if status != "OK":
            return QueryResult(label="neowise_cone", service="irsa", status=status,
                               query="SELECT ...", error="synthetic")
        seed = int(round(ra * 1000)) % 997
        if kind == "ramp":
            d = _frames(seed=seed, ramp=0.3)
        elif kind == "impact":
            d = _frames(seed=seed)
            yr = 2000.0 + (d["mjd"] - 51544.5) / 365.25
            step = np.where(yr >= 2016.5, np.exp(-(yr - 2016.5) / 1.5), 0.0)
            d["w1mpro"] -= 0.3 * step
            d["w2mpro"] -= 0.3 * step
        elif kind == "empty":
            d = _frames(seed=seed).iloc[0:0]
        else:
            d = _frames(seed=seed)
        return QueryResult(label="neowise_cone", service="irsa",
                           status="OK" if len(d) else "QUERY_RETURNED_ZERO_ROWS",
                           n_rows=len(d), query="SELECT ...", data=d)
    return f


def _config_for_tests():
    conf = load_ignition_config()
    conf["sample"]["fields"] = [{"ra": 266.0, "dec": 65.0, "radius_deg": 1.0}]
    conf["sample"]["count_parent"] = True
    conf["sensitivity"]["max_stars"] = 5
    return conf


def test_parse_shard_and_shard_rows():
    assert parse_shard("2/8") == (2, 8) and parse_shard("") == (0, 1)
    with pytest.raises(SystemExit):
        parse_shard("8/8")
    df = pd.DataFrame({"source_id": [str(i) for i in range(10)]})
    a, b = shard_rows(df, 0, 3), shard_rows(df, 1, 3)
    assert len(a) == 4 and len(b) == 3
    assert not set(a["source_id"]) & set(b["source_id"])
    assert len(shard_rows(df, 0, 3, max_stars=2)) == 2


def test_empty_archive_is_no_data_reached(tmp_path):
    """A failed archive must never read as a science null."""
    rep = ignition_run("all", out_dir=tmp_path, conf=_config_for_tests(),
                       query_fn=_FakeGaia("fail"), cone_fn=_cone_factory(),
                       upload_fn=lambda *a, **k: QueryResult(label="u", service="irsa",
                                                             status="QUERY_FAILED",
                                                             error="synthetic"))
    assert rep["verdict"] == "NO_DATA_REACHED"
    assert rep["reason"] == "gaia_QUERY_FAILED"
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["stage_counts"]["screened"] == 0
    assert "NOT a null result" in s["note"]
    probe = json.loads((tmp_path / "probe.json").read_text())
    assert probe["verdict"] == "NO_DATA_REACHED"
    assert probe["neowise_route_recommended"] == "none"


def test_neowise_zero_rows_is_no_data_reached_with_the_right_reason(tmp_path):
    rep = ignition_run("all", out_dir=tmp_path, conf=_config_for_tests(),
                       query_fn=_FakeGaia(), cone_fn=_cone_factory("empty"), route="cone",
                       upload_fn=lambda *a, **k: QueryResult(label="u", service="irsa",
                                                             status="QUERY_FAILED",
                                                             error="synthetic"))
    assert rep["verdict"] == "NO_DATA_REACHED"
    assert rep["reason"] == "neowise_returned_no_usable_epochs"
    assert rep["denominators"]["n_stars_attempted_neowise"] == 6


def test_missing_shard_outputs_are_never_a_clean_null(tmp_path):
    """VIGIL's failure mode: an aggregator run on a tree without the shard files."""
    (tmp_path / "sample.json").write_text(json.dumps({"status": "OK", "n_after_local_cuts": 6,
                                                      "n_shards_planned": 4, "parent_count": 60,
                                                      "n_rows_pulled": 6}))
    s = stage_assess(_config_for_tests(), tmp_path)
    assert s["verdict"] == "NO_DATA_REACHED"
    assert s["reason"] == "no_shard_outputs_found"
    assert s["shards"]["expected"] == 4 and s["shards"]["found"] == 0


def test_end_to_end_synthetic_run_finds_the_injected_ignition(tmp_path):
    conf = _config_for_tests()
    rep = ignition_run("all", out_dir=tmp_path, conf=conf, query_fn=_FakeGaia(n=6),
                       cone_fn=_cone_factory("ramp"), route="cone",
                       upload_fn=lambda *a, **k: QueryResult(label="u", service="irsa",
                                                             status="QUERY_FAILED",
                                                             error="synthetic"))
    assert rep["verdict"] == "IGNITION_CANDIDATES"
    assert rep["n_candidates"] == 6
    assert rep["stage_counts"]["screened"] == 6
    assert rep["denominators"]["parent_count_archive"] == 60
    assert rep["shards"]["expected"] == 1 and rep["shards"]["found"] == 1
    cands = pd.read_csv(tmp_path / "candidates.csv", dtype={"source_id": str})
    assert len(cands) == 6
    assert (cands["vet_verdict"] == "clean_optical_untested").all()
    assert (cands["optical"] == "not_checked").all()
    assert (cands["w1_slope_sigma"] > 5).all() and (cands["w2_slope_sigma"] > 5).all()
    # The ledger of the acquisition survives to the assess stage.
    assert (tmp_path / "acquire_s0of1.json").exists()
    assert (tmp_path / "epochs_s0of1.csv").exists()


def test_end_to_end_field_of_impacts_yields_no_candidate_and_a_sensitivity(tmp_path):
    conf = _config_for_tests()
    rep = ignition_run("all", out_dir=tmp_path, conf=conf, query_fn=_FakeGaia(n=5),
                       cone_fn=_cone_factory("impact"), route="cone",
                       upload_fn=lambda *a, **k: QueryResult(label="u", service="irsa",
                                                             status="QUERY_FAILED",
                                                             error="synthetic"))
    assert rep["verdict"] == "NO_IGNITION_CANDIDATE"
    assert rep["stage_counts"]["screened"] == 5
    assert rep["veto_counters"]["screen"].get("IMPULSIVE_SHAPE", 0) == 5
    sens = rep["sensitivity"]["amps"]
    assert set(sens) == {"0.10", "0.20", "0.40"}
    assert sens["0.40"]["n"] == 5
    assert "not an occurrence limit" in rep["note"]


def test_optical_series_directory_is_used_by_assess(tmp_path):
    conf = _config_for_tests()
    ignition_run("sample,acquire,screen", out_dir=tmp_path, conf=conf, query_fn=_FakeGaia(n=3),
                 cone_fn=_cone_factory("ramp"), route="cone")
    odir = tmp_path / "optical"
    odir.mkdir()
    parent = pd.read_parquet(tmp_path / "parent.parquet")
    sids = parent["source_id"].astype(str).tolist()
    _optical("flat").to_csv(odir / f"{sids[0]}.csv", index=False)
    _optical("fading").to_csv(odir / f"{sids[1]}.csv", index=False)
    s = stage_assess(conf, tmp_path, n_shards_expected=1, optical_dir=odir)
    vet = pd.read_csv(tmp_path / "stars_vetted.csv", dtype={"source_id": str})
    by = dict(zip(vet["source_id"], vet["vet_verdict"], strict=False))
    assert by[sids[0]] == "clean"
    assert by[sids[1]] == "rejected_rcrb_like"
    assert by[sids[2]] == "clean_optical_untested"
    assert s["n_clean"] == 1 and s["n_candidates"] == 2


def test_acquire_checkpoints_and_resumes(tmp_path):
    stars = _gaia_rows(6)
    store = EpochStore.open(tmp_path, "s0of1")
    calls = {"n": 0}
    real = _cone_factory("constant")

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    roll = acquire_stars(stars, store, {"checkpoint_every": 2}, route="cone", cone_fn=counting)
    assert roll["n_ok"] == 6 and calls["n"] == 6
    assert (tmp_path / "epochs_s0of1.csv").exists()
    # A second pass over the same shard fetches nothing.
    store2 = EpochStore.open(tmp_path, "s0of1")
    assert len(store2.done) == 6
    roll2 = acquire_stars(stars, store2, {}, route="cone", cone_fn=counting)
    assert roll2["n_attempted"] == 0 and calls["n"] == 6
    ep = pd.read_csv(tmp_path / "epochs_s0of1.csv", dtype={"source_id": str})
    assert ep["source_id"].nunique() == 6


def test_field_route_groups_a_field_query_onto_the_stars(tmp_path):
    stars = _gaia_rows(4)

    def field_fn(ra, dec, radius_deg, w1_max=13.0, max_rows=0):
        frames = []
        for i, (_, s) in enumerate(stars.iterrows()):
            d = _frames(seed=i, ramp=0.3)
            d["ra"] = s["ra"] + np.random.default_rng(i).normal(0, 0.3 / 3600, len(d))
            d["dec"] = s["dec"] + np.random.default_rng(i + 7).normal(0, 0.3 / 3600, len(d))
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
        return QueryResult(label="neowise_field", service="irsa", status="OK", n_rows=len(df),
                           query="SELECT ...", data=df)

    # Stars have pmra = 60 mas/yr: the field-route grouping must propagate it (VIGIL's fix).
    store = EpochStore.open(tmp_path, "s0of1")
    roll = acquire_stars(stars, store, {}, route="field",
                         fields=[{"ra": 266.0, "dec": 65.0, "radius_deg": 1.0}],
                         field_fn=field_fn)
    assert roll["n_ok"] == 4
    ep = pd.read_csv(tmp_path / "epochs_s0of1.csv", dtype={"source_id": str})
    df, rep = screen_epochs(ep, _config_for_tests())
    assert rep["n_rise_candidates"] == 4


def test_upload_route_groups_by_uploaded_id(tmp_path):
    stars = _gaia_rows(3)

    def upload_fn(sub, radius_arcsec=2.5):
        frames = []
        for i, (_, s) in enumerate(sub.iterrows()):
            d = _frames(seed=i)
            d.insert(0, "sid", str(s["source_id"]))
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
        return QueryResult(label="neowise_upload", service="irsa", status="OK", n_rows=len(df),
                           query="SELECT ...", data=df)

    store = EpochStore.open(tmp_path, "s0of1")
    roll = acquire_stars(stars, store, {"upload_chunk": 2}, route="upload", upload_fn=upload_fn)
    assert roll["n_ok"] == 3 and len(store.done) == 3


# --------------------------------------------------------------------------
# The query plan: shapes, the transport ladder, and the probe's budget
# --------------------------------------------------------------------------
_TIMEOUT_500 = ("HTTPError('Error 500:\\nSQL exception: ERROR: canceling statement due to "
                "statement timeout')")


def _shape_of(adql: str) -> str:
    """Recover which of SHAPES an ADQL string is, the way a reader would."""
    if "FROM (" not in adql:
        return "flat"
    return "inner_cone" if "WHERE w.w1mpro" in adql else "inner_cone_postfilter"


class _ShapeGaia:
    """A Gaia stand-in that answers only the shapes it was told to answer.

    The archive's real behaviour on run 34787803862: the flat three-table join
    is killed by the statement timeout whatever the row cap, so the only way to
    get an answer is to send a different shape.
    """

    def __init__(self, answers=("inner_cone",), n=3, error=_TIMEOUT_500, delay_s=0.0,
                 delay_shapes=()):
        self.answers = tuple(answers)
        self.n, self.error = n, error
        self.delay_s, self.delay_shapes = float(delay_s), tuple(delay_shapes)
        self.queries: list[str] = []
        self.shapes: list[str] = []

    def __call__(self, adql):
        self.queries.append(adql)
        if adql.lstrip().startswith("SELECT TOP 1 *"):        # the AllWISE column peek
            return pd.DataFrame({c: [0] for c in ("designation", "ext_flag", "ph_qual",
                                                  "cc_flags", "w1mpro", "w2mpro")})
        shape = _shape_of(adql)
        self.shapes.append(shape)
        if shape in self.delay_shapes:
            time.sleep(self.delay_s)
        if shape not in self.answers:
            raise RuntimeError(self.error)
        if adql.startswith("SELECT COUNT"):
            return pd.DataFrame({"n": [self.n * 10]})
        return _gaia_rows(self.n)


def _upload_fails(*_a, **_k):
    return QueryResult(label="u", service="irsa", status="QUERY_FAILED", error="synthetic")


def test_probe_tries_the_inner_cone_shape_before_the_flat_one_and_records_it(tmp_path):
    fake = _ShapeGaia(answers=("inner_cone",))
    rep = ignition_run("probe", out_dir=tmp_path, conf=_config_for_tests(), query_fn=fake,
                       cone_fn=_cone_factory(), upload_fn=_upload_fails)
    assert fake.shapes[0] == "inner_cone"                  # first, before anything flat
    assert "flat" not in fake.shapes                       # a shape that answers ends the ladder
    assert rep["gaia_shape_working"] == "inner_cone"
    assert [s["shape"] for s in rep["gaia_shapes"]] == ["inner_cone"]
    assert rep["gaia_shapes"][0]["status"] == "OK"
    assert "FROM (" in rep["gaia_shapes"][0]["query"]
    assert rep["gaia_join"]["status"] == "OK"
    saved = json.loads((tmp_path / "probe.json").read_text())
    assert saved["gaia_shape_working"] == "inner_cone"     # on disk, for the next run
    assert saved["budget"]["per_shape_s"] and saved["budget"]["total_s"]


def test_the_sample_stage_reuses_the_shape_the_probe_recorded(tmp_path):
    """The probe already paid for finding a plan that returns; do not re-derive it."""
    (tmp_path / "probe.json").write_text(json.dumps(
        {"gaia_shape_working": "inner_cone_postfilter"}))
    fake = _ShapeGaia(answers=("inner_cone_postfilter",))
    rep = stage_sample(_config_for_tests(), tmp_path, n_shards=1, query_fn=fake)
    assert fake.shapes[0] == "inner_cone_postfilter"       # tried first, not third
    assert rep["query_shape_from_probe"] == "inner_cone_postfilter"
    assert rep["query_shape_used"] == "inner_cone_postfilter"
    assert rep["status"] == "OK" and rep["n_after_local_cuts"] == 3
    # The AllWISE cuts were not in the SQL, so the pandas post-filter carried them.
    assert all("w.w1mpro - w.w2mpro <" not in q for q in fake.queries)
    assert rep["local_cut_counters"]["cut_w1w2_photospheric"] == 0


def test_an_async_queue_failure_falls_back_to_the_next_shape(tmp_path):
    """The statement timeout that killed run 34787803862, verbatim, then a fallback."""
    fake = _ShapeGaia(answers=("inner_cone_postfilter",), error=_TIMEOUT_500)
    rep = ignition_run("probe", out_dir=tmp_path, conf=_config_for_tests(), query_fn=fake,
                       cone_fn=_cone_factory(), upload_fn=_upload_fails)
    assert fake.shapes[:2] == ["inner_cone", "inner_cone_postfilter"]
    assert "flat" not in fake.shapes
    assert rep["gaia_shape_working"] == "inner_cone_postfilter"
    recs = {s["shape"]: s for s in rep["gaia_shapes"]}
    assert recs["inner_cone"]["status"] == "QUERY_FAILED"
    assert "canceling statement due to statement timeout" in recs["inner_cone"]["error"]
    assert "CIRCLE('ICRS', 266.0, 65.0, 1.0)" in recs["inner_cone"]["query"]   # the ADQL sent
    assert recs["inner_cone_postfilter"]["status"] == "OK"
    assert rep["verdict"] == "GAIA_AND_NEOWISE_REACHABLE"


def test_the_probe_budget_expires_into_timed_out_rather_than_hanging(tmp_path):
    """A broken plan must cost minutes, not the 1,490 s of four blind retries."""
    conf = _config_for_tests()
    conf["probe"] = {**conf["probe"], "budget_s": 0.25, "total_budget_s": 0.8,
                     "columns_timeout_s": 0.25, "neowise_timeout_s": 0.25}
    fake = _ShapeGaia(answers=(), delay_s=30.0, delay_shapes=SHAPES)
    t0 = time.monotonic()
    rep = ignition_run("probe", out_dir=tmp_path, conf=conf, query_fn=fake,
                       cone_fn=_cone_factory(), upload_fn=_upload_fails)
    assert time.monotonic() - t0 < 20.0                    # it did not wait for the query
    stats = [s["status"] for s in rep["gaia_shapes"]]
    assert stats[0] == "TIMED_OUT"
    assert set(stats) <= {"TIMED_OUT", "NOT_ATTEMPTED"}
    assert rep["gaia_shapes"][0]["elapsed_s"] >= 0.25      # the elapsed seconds are recorded
    assert 0 < rep["gaia_shapes"][0]["timeout_s"] <= 0.25
    assert rep["gaia_shape_working"] is None
    assert rep["verdict"] == "NO_DATA_REACHED"
    assert rep["elapsed_s"] < 20.0


def test_the_probe_writes_probe_json_even_when_every_route_fails(tmp_path):
    """25 minutes of evidence must not survive only in a log someone reads by hand."""
    fake = _ShapeGaia(answers=())
    rep = ignition_run("probe", out_dir=tmp_path, conf=_config_for_tests(), query_fn=fake,
                       cone_fn=_cone_factory(status="QUERY_FAILED"), upload_fn=_upload_fails)
    p = tmp_path / "probe.json"
    assert p.exists()
    saved = json.loads(p.read_text())
    assert saved["verdict"] == "NO_DATA_REACHED" == rep["verdict"]
    assert saved["neowise_route_recommended"] == "none"
    assert [s["shape"] for s in saved["gaia_shapes"]] == list(SHAPES)     # all three tried
    for s in saved["gaia_shapes"]:
        assert s["status"] == "QUERY_FAILED"
        assert "canceling statement due to statement timeout" in s["error"]
        assert s["query"].startswith("SELECT")             # the ADQL sent, not a summary
    assert saved["gaia_shape_working"] is None
    assert saved["neowise_cone"]["status"] == "NOT_ATTEMPTED"
    assert "no query shape answered" in saved["neowise_cone"]["reason"]


def test_the_transport_ladder_is_async_first_and_records_the_queue():
    """The 503 was the sync endpoint's 150-job ceiling; bulk queries never go there."""
    assert [q for _n, q, _f in GAIA_TRANSPORTS][:2] == ["async", "async"]
    calls: list[str] = []

    def _mk(name, ok):
        def _f(_adql):
            calls.append(name)
            if not ok:
                raise RuntimeError(f"{name} refused")
            return pd.DataFrame({"n": [1]})
        return _f

    ladder = (("astroquery_async", "async", _mk("astroquery_async", False)),
              ("pyvo_async", "async", _mk("pyvo_async", True)),
              ("pyvo_sync", "sync", _mk("pyvo_sync", True)))
    _df, rec = run_gaia_query("SELECT 1", transports=ladder, retries_per_transport=1,
                              base_sleep=0.0)
    assert rec["status"] == "OK" and rec["transport"] == "pyvo_async" and rec["queue"] == "async"
    assert calls == ["astroquery_async", "pyvo_async"]     # the sync queue was never reached

    calls.clear()
    _df2, rec2 = run_gaia_query("SELECT 1", transports=ladder[:1] + ladder[2:],
                                retries_per_transport=1, base_sleep=0.0, allow_sync=False)
    assert rec2["status"] == "QUERY_FAILED" and "pyvo_sync" not in calls
    assert any("150-job ceiling" in (a.get("error") or "") for a in rec2["attempts"])

    # A deadline already in the past starts no attempt at all.
    calls.clear()
    _df3, rec3 = run_gaia_query("SELECT 1", transports=ladder, retries_per_transport=1,
                                deadline=time.monotonic() - 1.0)
    assert rec3["status"] == "TIMED_OUT" and calls == []
    exc = GaiaQueryFailed({"status": "TIMED_OUT", "error": "boom"})
    assert "TIMED_OUT" in str(exc) and exc.record["error"] == "boom"


def test_probe_recommends_the_upload_route_when_it_answers(tmp_path):
    conf = _config_for_tests()

    def upload_ok(sub, radius_arcsec=2.5):
        d = _frames(seed=1)
        d.insert(0, "sid", "10")
        return QueryResult(label="neowise_upload", service="irsa", status="OK", n_rows=len(d),
                           query="SELECT ...", data=d)

    rep = ignition_run("probe", out_dir=tmp_path, conf=conf, query_fn=_FakeGaia(n=3),
                       cone_fn=_cone_factory(), upload_fn=upload_ok)
    assert rep["verdict"] == "ALL_ROUTES_REACHABLE"
    assert rep["neowise_route_recommended"] == "upload"
    rep2 = ignition_run("probe", out_dir=tmp_path, conf=conf, query_fn=_FakeGaia(n=3),
                        cone_fn=_cone_factory(),
                        upload_fn=lambda *a, **k: QueryResult(label="u", service="irsa",
                                                              status="QUERY_FAILED",
                                                              error="synthetic"))
    assert rep2["neowise_route_recommended"] == "field"      # fields are configured


def test_config_thresholds_are_read_from_yaml_and_files_exist():
    conf = load_ignition_config()
    assert conf["rise"]["min_baseline_yr"] == 5.0
    assert conf["rise"]["slope_sigma_min"] == 5.0
    assert conf["rise"]["scan_period_days"] == [345.0, 385.0]
    assert conf["sample"]["w1w2_max"] == 0.15
    assert len(conf["sample"]["fields"]) >= 8
    assert conf["vet"]["w1_saturation_mag"] == 8.0
    assert conf["sample"]["query_shape"] == "auto"
    assert conf["probe"]["shapes"] == list(SHAPES)
    assert conf["probe"]["budget_s"] == 480.0        # ~8 min per candidate shape
    assert conf["probe"]["total_budget_s"] >= conf["probe"]["budget_s"]
    assert Path(".github/workflows/ignition.yml").exists()
    assert Path("docs/ignition.md").exists()


def test_the_workflow_commits_the_probe_and_not_only_an_artifact():
    """Run 34787803862 went green, took 25 min and committed nothing."""
    wf = Path(".github/workflows/ignition.yml").read_text()
    sample_job = wf[wf.index("\n  sample:"):wf.index("\n  acquire:")]
    assert "scripts/commit_results.sh" in sample_job          # the push-verifying script
    assert "results/ignition/probe.json" in sample_job.split("commit_results.sh")[1]
    assert "upload-artifact" in sample_job                    # the artifact is kept as well
    # The artifact upload runs BEFORE the commit, so evidence survives a failed push.
    assert sample_job.index("upload-artifact") < sample_job.index("commit_results.sh")
