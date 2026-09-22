"""Offline test suite for GROWTH stage ``direct`` --- every Kepler planet's TESS depth (S57).

No network anywhere (``conftest.py`` raises on any socket); every fetch in
:mod:`seti.growth.direct` takes an injectable callable and every test injects
one.  What this suite gates:

* **an injected depth change is recovered on synthetic SAP AND PDCSAP** and,
  with the duration scaled as a fixed impact parameter predicts, comes out a
  ``growth_candidate`` --- the load-bearing test;
* **a PDCSAP-only change is ``crowding_correction``** --- the Kepler-718 b
  lesson, as a rule: the raw aperture depth did not grow, PDC did;
* **a missing light curve is ``not_measured``** with the archive's own status,
  and ``QUERY_FAILED`` / ``QUERY_RETURNED_ZERO_ROWS`` / ``TIC_UNRESOLVED`` stay
  different facts;
* **a consistent planet states its sensitivity** (what change it could have
  seen) and a planet TESS cannot reach is ``not_measurable``, never
  ``consistent``;
* **the phase search recovers a shifted ephemeris** and a transit that is not
  there is ``transit_not_recovered``, not ``consistent``;
* **an eclipsing binary is vetoed by odd-even** and **a duration that tracks
  ``b`` is vetoed**;
* **sharding is by star and resume skips what is done**; the aggregate marks
  targets no shard reached ``NOT_REACHED``; the population scatter scales ``z``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seti.growth import direct as D
from seti.growth.drift import band_ratio, expected_t14_ratio
from seti.growth.stage2 import bkjd_to_btjd, synth_lightcurve

P = 2.0523499
T0_BKJD = 131.7513
T14_H = 2.078
T14_D = T14_H / 24.0
K_ROR = 0.10
B_IMP = 0.30
KOI_DEPTH = 10000.0


def _koi(kepoi="K00897.01", kepid=7849854, tic=268924036.0, **over) -> dict:
    row = {"kepoi_name": kepoi, "kepler_name": "Kepler-718 b", "kepid": kepid, "tic_id": tic,
           "tic_route": "name_planet", "koi_disposition": "CONFIRMED",
           "koi_pdisposition": "CANDIDATE", "koi_period": P, "koi_period_err1": 1e-7,
           "koi_period_err2": -1e-7, "koi_time0bk": T0_BKJD, "koi_time0bk_err1": 1e-4,
           "koi_time0bk_err2": -1e-4, "koi_duration": T14_H, "koi_duration_err1": 0.02,
           "koi_duration_err2": -0.02, "koi_depth": KOI_DEPTH, "koi_depth_err1": 30.0,
           "koi_depth_err2": -30.0, "koi_ror": K_ROR, "koi_impact": B_IMP, "koi_dor": 7.7,
           "koi_steff": 5800.0, "koi_slogg": 4.5, "koi_kepmag": 15.2, "koi_fpflag_nt": 0,
           "koi_fpflag_ss": 0, "koi_fpflag_co": 0, "koi_fpflag_ec": 0, "koi_count": 1,
           "ra": 285.0, "dec": 43.0}
    row.update(over)
    return row


def _ref_ppm() -> float:
    fr, _ = band_ratio(5800.0, 4.5, B_IMP, None)
    return KOI_DEPTH * fr


def _t0_tess() -> float:
    return bkjd_to_btjd(T0_BKJD) + 900 * P


def _products(depth_pdc_ppm, depth_sap_ppm, *, duration_days=T14_D, noise_ppm=300.0,
              n_transits=16, offset_days=0.0, odd_depth_factor=None, seed=3, span_durations=3.0,
              authors=(("SPOC", 120.0), ("TESS-SPOC", 600.0)), sectors=(41, 54)) -> list[dict]:
    """Synthetic products: each (author, sector) carries PDCSAP and SAP columns.

    ``span_durations`` is how much of the phase each epoch's window covers; the
    default (3 durations either side of mid-transit) is all the depth fit needs,
    and the control-phase tests widen it because an off-transit phase must land
    on data rather than in the fixture's own gap.
    """
    out = []
    i = 0
    for si, sec in enumerate(sectors):
        for author, exptime in authors:
            t0 = _t0_tess() + offset_days + si * n_transits * P
            cols = {}
            for col, dep in (("PDCSAP_FLUX", depth_pdc_ppm), ("SAP_FLUX", depth_sap_ppm)):
                i += 1
                lc = synth_lightcurve(period_days=P, t0_btjd=t0, duration_days=duration_days,
                                      depth=dep * 1e-6, exptime_s=exptime, n_transits=n_transits,
                                      noise_ppm=noise_ppm, sector=sec, author=author,
                                      seed=seed + i, span_durations=float(span_durations),
                                      odd_depth=(dep * odd_depth_factor * 1e-6
                                                 if odd_depth_factor else None))
                cols[col] = (lc["flux"], lc["flux_err"])
                time = lc["time"]
            out.append({"sector": sec, "author": author, "exptime_s": exptime, "time": time,
                        "fluxes": cols, "n_points": int(time.size)})
    return out


def _conf(**direct) -> dict:
    d = {"fetch": {"per_target_budget_s": 30.0, "shard_budget_s": 600.0, "retries": 1,
                   "retry_pause_s": 0.0, "clean_downloads": False, "tic_fallback": True},
         "classify": {"min_population_for_offset": 1000}}
    for k, v in direct.items():
        d.setdefault(k, {}).update(v)
    return {"direct": d, "stage2": {"fit": {"bootstrap_draws": 300}}, "limb_darkening": None,
            "archive": {}}


def _measure(products, entry=None, status="OK", conf=None):
    conf = conf or _conf()
    entry = entry or _koi()
    return D.measure_direct_target(
        entry, products, fetch_status=status, fetch_route="injected",
        fit=D.FitParams.from_config(conf), ensemble=D.EnsembleParams.from_config(conf),
        search=D.EpochSearchParams.from_config(conf), duration=D.DurationParams.from_config(conf),
        classify=D.ClassifyParams.from_config(conf))


# ---------------------------------------------------------------------------
# The load-bearing tests
# ---------------------------------------------------------------------------
def test_an_injected_depth_change_is_recovered_in_both_families_and_is_a_growth_candidate():
    ref = _ref_ppm()
    ratio = 2.0
    t14_fac = expected_t14_ratio(K_ROR, B_IMP, ratio)
    prods = _products(ref * ratio, ref * ratio, duration_days=T14_D * t14_fac)
    rec, members, sec, fold_df = _measure(prods)
    assert rec["lc_status"] == "OK" and rec["pdc_status"] == "OK" and rec["sap_status"] == "OK"
    for fam in ("pdc", "sap"):
        assert abs(rec[f"{fam}_depth_ppm"] - ref * ratio) < 4.0 * rec[f"{fam}_total_err_ppm"]
        assert rec[f"{fam}_z"] >= 5.0
        assert rec[f"{fam}_n_members_measured"] == 2      # SPOC and TESS-SPOC
        assert rec[f"{fam}_depth_reduction_spread_status"] == "OK"
        assert rec[f"{fam}_n_sectors_dropped_duplicate"] == 2   # one reduction per sector
    assert rec["ephemeris_recovered"] and abs(rec["epoch_offset_minutes"]) < 15.0
    assert rec["duration_verdict"] == "fixed_b"
    assert abs(rec["t14_factor"] - t14_fac) < 3.0 * rec["t14_factor_err"] + 0.05
    assert rec["class"] == D.CLASS_GROWTH and rec["vetoes"] == ""
    assert rec["would_be_candidate_without_vetoes"]
    assert len(members) == 4 and len(sec) and len(fold_df)


def test_a_pdc_only_change_is_a_crowding_correction_case_not_growth():
    """Kepler-718 b as a rule: PDC deeper than SAP, SAP no deeper than Kepler."""
    ref = _ref_ppm()
    prods = _products(ref * 2.0, ref * 0.8)
    rec, *_ = _measure(prods)
    assert rec["pdc_z"] >= 5.0
    assert rec["sap_z"] < 5.0
    assert rec["background_direction"] == "PDC_DEEPER_THAN_SAP"
    assert rec["sap_minus_pdcsap_z"] <= -3.0
    assert rec["class"] == D.CLASS_CROWDING
    assert "pdc_deeper_than_sap" in rec["flags_classify"]


def test_a_missing_light_curve_is_not_measured_and_the_statuses_stay_apart():
    rec, *_ = _measure([], status="QUERY_FAILED")
    assert rec["class"] == D.CLASS_NOT_MEASURED
    assert rec["not_measured_reason"] == "QUERY_FAILED"
    assert not np.isfinite(rec["pdc_depth_ppm"]) and not np.isfinite(rec["sap_depth_ppm"])
    rec0, *_ = _measure([], status="QUERY_RETURNED_ZERO_ROWS")
    assert rec0["class"] == D.CLASS_NOT_MEASURED
    assert rec0["not_measured_reason"] == "QUERY_RETURNED_ZERO_ROWS"
    rec1, *_ = _measure([], status="TIC_UNRESOLVED")
    assert rec1["not_measured_reason"] == "TIC_UNRESOLVED"
    rec2, *_ = _measure(_products(9000.0, 9000.0), entry=_koi(koi_time0bk=float("nan")))
    assert rec2["not_measured_reason"] == "EPHEMERIS_UNAVAILABLE"


def test_a_consistent_planet_states_what_change_it_could_have_seen():
    ref = _ref_ppm()
    rec, *_ = _measure(_products(ref, ref))
    assert rec["class"] == D.CLASS_CONSISTENT
    assert abs(rec["pdc_z"]) < 3.0 and abs(rec["sap_z"]) < 3.0
    for fam in ("pdc", "sap"):
        assert np.isfinite(rec[f"{fam}_detectable_ln_ratio"]) and rec[f"{fam}_detectable_ln_ratio"] > 0
        assert rec[f"{fam}_detectable_depth_change_ppm"] > 0
        assert rec[f"{fam}_expected_snr"] > 3.0
    assert rec["sap_vs_pdcsap_verdict"] == "BACKGROUND_TEST_AGREES"


def test_a_planet_tess_cannot_reach_is_not_measurable_never_consistent():
    ref = _ref_ppm()
    prods = _products(ref, ref, noise_ppm=60000.0, n_transits=4, sectors=(41,))
    rec, *_ = _measure(prods)
    assert rec["class"] in (D.CLASS_NOT_MEASURABLE, D.CLASS_NOT_RECOVERED)
    assert rec["class"] != D.CLASS_CONSISTENT
    # the sensitivity is still a number: what a 5-sigma change would have to be
    assert np.isfinite(rec["pdc_detectable_ln_ratio"])


def test_the_phase_search_recovers_a_shifted_ephemeris():
    ref = _ref_ppm()
    shift = 25.0 / 1440.0
    prods = _products(ref, ref, offset_days=shift)
    entry = _koi(koi_time0bk_err1=0.02, koi_time0bk_err2=-0.02)     # the window must cover it
    rec, *_ = _measure(prods, entry=entry)
    assert rec["ephemeris_recovered"]
    assert abs(rec["epoch_offset_minutes"] - 25.0) < 8.0
    assert rec["epoch_search_n_trials"] > 3
    assert abs(rec["pdc_depth_ppm"] - ref) < 4.0 * rec["pdc_total_err_ppm"]
    assert rec["class"] == D.CLASS_CONSISTENT
    # the depth at the NOMINAL ephemeris is shallower: that is what the search guards against
    assert rec["depth_at_nominal_ephemeris_ppm"] < rec["pdc_depth_ppm"]


def test_a_transit_that_is_not_there_is_not_recovered_rather_than_consistent():
    prods = _products(0.0, 0.0)
    rec, *_ = _measure(prods)
    assert not rec["ephemeris_recovered"]
    assert rec["class"] == D.CLASS_NOT_RECOVERED
    assert rec["pdc_expected_snr"] > 5.0        # TESS COULD have seen the KOI depth


def test_a_halved_depth_in_both_families_is_a_shrink_candidate_with_dilution_flagged():
    ref = _ref_ppm()
    ratio = 0.5
    t14_fac = expected_t14_ratio(K_ROR, B_IMP, ratio)
    rec, *_ = _measure(_products(ref * ratio, ref * ratio, duration_days=T14_D * t14_fac))
    assert rec["ephemeris_recovered"]
    assert rec["pdc_z"] <= -5.0 and rec["sap_z"] <= -5.0
    assert rec["class"] == D.CLASS_SHRINK
    assert "dilution_not_excluded" in rec["flags_classify"]


def test_an_eclipsing_binary_is_vetoed_by_its_odd_even_signature():
    ref = _ref_ppm()
    prods = _products(ref * 2.0, ref * 2.0, odd_depth_factor=0.5,
                      duration_days=T14_D * expected_t14_ratio(K_ROR, B_IMP, 1.5))
    rec, *_ = _measure(prods)
    assert rec["pdc_odd_even_sigma"] >= 3.0
    assert rec["class"] != D.CLASS_GROWTH
    assert D.VETO_ODD_EVEN in rec["vetoes"]


def test_a_depth_change_whose_duration_tracks_b_is_vetoed():
    ref = _ref_ppm()
    # twice the depth but a much SHORTER transit: a chord change, not a radius change
    prods = _products(ref * 2.0, ref * 2.0, duration_days=T14_D * 0.55)
    rec, *_ = _measure(prods)
    assert rec["t14_factor"] < 0.8
    assert rec["duration_verdict"] in ("tracks_b", "duration_inconsistent")
    assert rec["class"] != D.CLASS_GROWTH
    assert D.VETO_DURATION in rec["vetoes"]
    assert rec["would_be_candidate_without_vetoes"]


def test_a_koi_false_positive_flag_vetoes_a_would_be_candidate():
    ref = _ref_ppm()
    prods = _products(ref * 2.0, ref * 2.0,
                      duration_days=T14_D * expected_t14_ratio(K_ROR, B_IMP, 2.0))
    rec, *_ = _measure(prods, entry=_koi(koi_fpflag_ss=1))
    assert rec["would_be_candidate_without_vetoes"]
    assert D.VETO_FPFLAG in rec["vetoes"] and rec["class"] == D.CLASS_DEEPER


def test_the_z_falls_back_to_the_linear_form_when_the_depth_is_consistent_with_zero():
    ref = 10000.0
    c = D.compare_family(50.0, 400.0, 400.0, ref, 30.0)
    assert c["z_method"] == "linear" and c["z"] < -5.0
    assert np.isfinite(c["detectable_ln_ratio"])
    c2 = D.compare_family(20000.0, 400.0, 400.0, ref, 30.0)
    assert c2["z_method"] == "ln" and c2["z"] > 5.0
    c3 = D.compare_family(20000.0, 400.0, 400.0, float("nan"), float("nan"))
    assert c3["comparison_status"] == D.REASON_NO_REFERENCE


def test_the_duration_profile_recovers_an_injected_duration_change():
    ref = _ref_ppm()
    for fac in (0.7, 1.0, 1.4):
        prods = _products(ref, ref, duration_days=T14_D * fac, noise_ppm=200.0)
        segs, _ = D.dedupe_sectors(D.family_segments(prods, D.FAMILY_PDC))
        fr = D.detrended_fold(segs, period_days=P, t0_btjd=_t0_tess(), duration_days=T14_D,
                              params=D.FitParams(), window_durations=2.5, guard_durations=1.25)
        r = D.fit_duration(fr, duration_days=T14_D, ingress_frac=D.ingress_fraction(K_ROR, B_IMP))
        assert r["duration_fit_status"] == "OK"
        assert abs(r["t14_factor"] - fac) < 3.0 * r["t14_factor_err"] + 0.06


# ---------------------------------------------------------------------------
# Targets, sharding, resume, aggregate
# ---------------------------------------------------------------------------
def _koi_table() -> pd.DataFrame:
    rows = [
        _koi("K00001.01", 1000, float("nan"), kepler_name="Kepler-1 b", ra=280.0, dec=40.0),
        _koi("K00002.01", 2000, float("nan"), kepler_name="", ra=281.0, dec=41.0),
        _koi("K00002.02", 2000, float("nan"), kepler_name="Kepler-2 c", ra=281.0, dec=41.0),
        _koi("K00003.01", 3000, float("nan"), kepler_name="", ra=282.0, dec=42.0),
        _koi("K00004.01", 4000, float("nan"), kepler_name="", ra=283.0, dec=43.0,
             koi_disposition="FALSE POSITIVE"),
        _koi("K00005.01", 5001, float("nan"), kepler_name="", ra=284.0, dec=44.0,
             koi_period=45.0),
    ]
    df = pd.DataFrame(rows).drop(columns=["tic_id", "tic_route"])
    return df


def _ps_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"pl_name": "Kepler-1 b", "hostname": "Kepler-1", "tic_id": "TIC 111", "default_flag": 1,
         "disc_facility": "Kepler", "ra": 280.0, "dec": 40.0},
        {"pl_name": "Kepler-2 c", "hostname": "Kepler-2", "tic_id": "TIC 222", "default_flag": 1,
         "disc_facility": "Kepler", "ra": 281.0, "dec": 41.0},
        {"pl_name": "Kepler-9 b", "hostname": "Kepler-9", "tic_id": "TIC 333", "default_flag": 1,
         "disc_facility": "Kepler", "ra": 282.0, "dec": 42.0},
    ])


def test_build_targets_resolves_tics_by_every_ps_route_and_keeps_the_unresolved():
    t, rep = D.build_targets(_koi_table(), _ps_table())
    assert rep["n_koi"] == 6 and rep["n_confirmed_or_candidate"] == 5
    by = dict(zip(t["kepoi_name"], t["tic_route"], strict=True))
    assert by["K00001.01"] == "name_planet"
    assert by["K00002.02"] == "name_planet" and by["K00002.01"] == "name_host"   # sibling host
    assert by["K00003.01"] == "position_tic"
    assert by["K00005.01"] == "" and not np.isfinite(t.set_index("kepoi_name").loc["K00005.01",
                                                                                      "tic_id"])
    assert rep["n_with_tic"] == 4 and rep["n_without_tic"] == 1
    assert rep["n_long_period"] == 1
    assert "FALSE POSITIVE" not in set(t["koi_disposition"])
    assert rep["targets_statement"].startswith("5 of 6 KOIs are targets")


def test_direct_targets_writes_the_table_and_degrades_on_a_failed_archive(tmp_path):
    def qf(adql):
        if "from cumulative" in adql:
            return _koi_table()
        if "from ps" in adql:
            return _ps_table()
        return pd.DataFrame()
    rep = D.direct_targets(_conf(), tmp_path, query_fn=qf)
    assert rep["koi_status"] == "OK" and rep["n_targets"] == 5
    assert (tmp_path / "targets.csv").exists()

    def boom(_adql):
        raise RuntimeError("archive down")
    rep2 = D.direct_targets(_conf(), tmp_path / "b", query_fn=boom)
    assert rep2["koi_status"] == "QUERY_FAILED" and rep2["n_targets"] == 0


def test_shards_are_by_star_and_resume_skips_what_is_done(tmp_path):
    out = tmp_path / "direct"
    out.mkdir()
    targets, _ = D.build_targets(_koi_table(), _ps_table())
    D._write_csv(out / "targets.csv", targets)
    calls: list = []
    ref = _ref_ppm()

    def pf(tic, **_kw):
        calls.append(int(tic))
        return _products(ref, ref, n_transits=10, sectors=(41,))

    def tf(kepid, ra, dec, kepmag, **_kw):
        return 5555.0, "tic_region_kic"

    n_shards = 2
    conf = _conf()
    reps = [D.direct_measure(conf, out, shard=s, n_shards=n_shards, products_fn=pf, tic_fn=tf,
                             resume=True) for s in range(n_shards)]
    assert sum(r["n_measured_this_run"] for r in reps) == 5
    # the two planets of star 2000 share ONE download
    assert calls.count(222) == 1 and len(calls) == 4
    # the unresolved star went through the fallback and its route is on the record
    df = pd.concat([D._read_csv(D._shard_paths(out, s)["csv"]) for s in range(n_shards)])
    assert df.set_index("kepoi_name").loc["K00005.01", "tic_route"] == "tic_region_kic"
    assert set(df["shard"]) == {0, 1}
    assert all(D.shard_of(k, n_shards) == s for k, s in zip(df["kepid"], df["shard"], strict=True))
    # resume: nothing is fetched or measured again
    n_calls = len(calls)
    reps2 = [D.direct_measure(conf, out, shard=s, n_shards=n_shards, products_fn=pf, tic_fn=tf,
                              resume=True) for s in range(n_shards)]
    assert len(calls) == n_calls
    assert sum(r["n_measured_this_run"] for r in reps2) == 0
    assert sum(r["n_skipped_already_done"] for r in reps2) == 5


def test_an_unresolvable_tic_is_not_measured_with_its_own_reason(tmp_path):
    out = tmp_path / "direct"
    out.mkdir()
    targets, _ = D.build_targets(_koi_table(), _ps_table())
    D._write_csv(out / "targets.csv", targets)

    def tf(kepid, ra, dec, kepmag, **_kw):
        return float("nan"), ""

    def pf(tic, **_kw):
        return []
    D.direct_measure(_conf(), out, shard=0, n_shards=1, products_fn=pf, tic_fn=tf)
    df = D._read_csv(D._shard_paths(out, 0)["csv"]).set_index("kepoi_name")
    assert df.loc["K00005.01", "not_measured_reason"] == "TIC_UNRESOLVED"
    assert df.loc["K00001.01", "not_measured_reason"] == "QUERY_RETURNED_ZERO_ROWS"
    assert set(df["class"]) == {D.CLASS_NOT_MEASURED}


def test_a_tic_id_round_trips_through_csv_exactly(tmp_path):
    """Nine digits survive the write.  ``%.8g`` turned TIC 122785305 into
    1.227853e+08 --- TIC 122785300, a star that does not exist --- and every
    product query on it came back empty."""
    df = pd.DataFrame({"kepoi_name": ["K00889.01", "K00001.01"],
                       "kepid": [757450, 10666592],
                       "tic_id": [122785305.0, 351053728.0],
                       "koi_depth": [16053.4, 14000.123456789]})
    p = tmp_path / "t.csv"
    D._write_csv(p, df)
    text = p.read_text()
    assert "122785305" in text and "351053728" in text
    assert "e+08" not in text
    back = D._read_csv(p)
    assert list(back["tic_id"]) == [122785305, 351053728]
    assert list(back["kepid"]) == [757450, 10666592]
    # a missing id stays missing, and the float columns keep their short format
    D._write_csv(p, df.assign(tic_id=[122785305.0, float("nan")]))
    assert D._read_csv(p)["tic_id"].isna().iloc[1]


def test_a_truncatable_tic_is_verified_against_the_sky_before_it_is_believed():
    """The dangerous case is not the empty query --- it is the truncated id that
    lands on a REAL other star and gets measured silently."""
    assert D.tic_is_truncated(122785300.0)          # what %.8g leaves of 122785305
    assert D.tic_is_truncated(351053720.0)
    assert not D.tic_is_truncated(122785305.0)      # nine digits, not a multiple of ten
    assert not D.tic_is_truncated(26817004.0)       # eight digits: exact through %.8g
    assert not D.tic_is_truncated(float("nan")) and not D.tic_is_truncated(0.0)


def test_a_truncated_tic_is_never_queried_on_its_own_authority(tmp_path):
    """It is re-resolved from the sky first; unverifiable means not measured."""
    out = tmp_path / "direct"
    out.mkdir()
    targets, _ = D.build_targets(_koi_table(), _ps_table())
    good, wrong = 122785305.0, 122785300.0          # the real id and what %.8g left
    assert D.tic_is_truncated(wrong) and not D.tic_is_truncated(good)
    targets = targets.copy()
    targets["tic_id"] = targets["tic_id"].where(targets["tic_id"].isna(), wrong)
    D._write_csv(out / "targets.csv", targets)
    ref = _ref_ppm()
    asked: list[int] = []

    def pf(tic, **_kw):
        asked.append(int(tic))
        return _products(ref, ref, n_transits=10, sectors=(41,))   # ANY tic would serve
    rep = D.direct_measure(_conf(), out, shard=0, n_shards=1, products_fn=pf,
                           tic_fn=lambda *a, **k: (good, "tic_kic_crossid"))
    assert int(wrong) not in asked                  # the wrong star was never touched
    assert rep["n_tic_suspect_truncation"] >= 1 and rep["n_tic_repaired"] >= 1

    # and when the sky cannot name the star, it is a non-measurement, not a guess
    out2 = tmp_path / "direct2"
    out2.mkdir()
    D._write_csv(out2 / "targets.csv", targets)
    asked.clear()
    D.direct_measure(_conf(), out2, shard=0, n_shards=1, products_fn=pf,
                     tic_fn=lambda *a, **k: (float("nan"), ""))
    assert int(wrong) not in asked
    df2 = D._read_csv(D._shard_paths(out2, 0)["csv"])
    assert (df2["lc_status"] == D.REASON_TIC_UNRESOLVED).any()
    assert set(df2["class"]) == {D.CLASS_NOT_MEASURED}


def test_a_catalogue_tic_that_serves_nothing_is_rechecked_against_the_sky(tmp_path):
    """A stale or truncated catalogue TIC is re-resolved before it is called a
    non-detection; a star TESS really never observed comes back empty twice."""
    out = tmp_path / "direct"
    out.mkdir()
    targets, _ = D.build_targets(_koi_table(), _ps_table())
    good = float(pd.to_numeric(targets["tic_id"], errors="coerce").dropna().iloc[0])
    wrong = float(int(good) - int(good) % 10)          # what %.8g would have left
    targets = targets.copy()
    targets["tic_id"] = targets["tic_id"].where(targets["tic_id"].isna(), wrong)
    D._write_csv(out / "targets.csv", targets)
    ref = _ref_ppm()
    calls: list[int] = []

    def pf(tic, **_kw):
        calls.append(int(tic))
        if int(tic) == int(good):
            return _products(ref, ref, n_transits=10, sectors=(41,))
        return []                                       # the wrong star serves nothing

    rep = D.direct_measure(_conf(), out, shard=0, n_shards=1, products_fn=pf,
                           tic_fn=lambda *a, **k: (good, "tic_kic_crossid"))
    assert int(wrong) in calls and int(good) in calls   # tried the catalogue, then the sky
    assert rep["n_tic_rechecked"] >= 1 and rep["n_tic_repaired"] >= 1
    df = D._read_csv(D._shard_paths(out, 0)["csv"])
    ok = df[df["lc_status"] == "OK"]
    assert len(ok) and ok["tic_route"].str.contains("_after_").any()
    # the repaired rows carry the TIC that actually served the light curve
    assert (pd.to_numeric(ok.loc[ok["tic_route"].str.contains("_after_"), "tic_id"])
            == int(good)).all()

    # and the honest negative: nothing anywhere stays QUERY_RETURNED_ZERO_ROWS
    out2 = tmp_path / "direct2"
    out2.mkdir()
    D._write_csv(out2 / "targets.csv", targets)
    rep2 = D.direct_measure(_conf(), out2, shard=0, n_shards=1,
                            products_fn=lambda tic, **_kw: [],
                            tic_fn=lambda *a, **k: (good, "tic_kic_crossid"))
    assert rep2["n_tic_repaired"] == 0
    df2 = D._read_csv(D._shard_paths(out2, 0)["csv"])
    assert (df2["lc_status"] == D.REASON_ZERO_ROWS).any()
    assert set(df2["class"]) == {D.CLASS_NOT_MEASURED}


def test_a_hopeless_sensitivity_does_not_overflow_the_detectable_change():
    """The sensitivity of a star TESS cannot reach is +inf, never an exception.

    Run 35738702139 lost a whole shard to ``OverflowError`` here: a shallow
    reference depth against a huge TESS error drives ``detectable_ln_ratio``
    past 709, where ``exp`` has no finite double left.
    """
    assert D.detectable_change_ppm(300.0, 2000.0) == math.inf
    assert math.isnan(D.detectable_change_ppm(float("nan"), 1.0))
    assert D.detectable_change_ppm(1000.0, math.log(2.0)) == pytest.approx(1000.0)
    cmp = D.compare_family(float("nan"), float("nan"), 1.0e9, 3.0e-4, 1.0e-5)
    assert cmp["detectable_depth_change_ppm"] == math.inf
    assert np.isfinite(cmp["detectable_ln_ratio"])


def test_one_star_that_raises_costs_only_itself(tmp_path, monkeypatch):
    """A pathological target is a NON-measurement, not the death of the shard."""
    out = tmp_path / "direct"
    out.mkdir()
    targets, _ = D.build_targets(_koi_table(), _ps_table())
    D._write_csv(out / "targets.csv", targets)
    ref = _ref_ppm()
    real = D.measure_direct_target
    seen: list[str] = []

    def boom(entry, products, **kw):
        seen.append(str(entry.get("kepoi_name")))
        if len(seen) == 1:
            raise OverflowError("math range error")
        return real(entry, products, **kw)
    monkeypatch.setattr(D, "measure_direct_target", boom)

    def pf(tic, **_kw):
        return _products(ref, ref, n_transits=10, sectors=(41,))
    rep = D.direct_measure(_conf(), out, shard=0, n_shards=1, products_fn=pf,
                           tic_fn=lambda *a, **k: (5555.0, "tic_region_kic"))
    assert rep["n_measure_failed"] == 1
    assert rep["n_measured_this_run"] == len(seen) >= 2       # it kept going
    df = D._read_csv(D._shard_paths(out, 0)["csv"]).set_index("kepoi_name")
    bad = df.loc[seen[0]]
    assert bad["lc_status"] == D.REASON_MEASURE_FAILED
    assert "OverflowError" in str(bad["not_measured_reason"])
    assert bad["class"] == D.CLASS_NOT_MEASURED
    # and the rest of the shard is real
    assert (df["class"] != D.CLASS_NOT_MEASURED).any()


def test_an_exhausted_shard_budget_leaves_targets_unreached_not_measured(tmp_path):
    out = tmp_path / "direct"
    out.mkdir()
    targets, _ = D.build_targets(_koi_table(), _ps_table())
    D._write_csv(out / "targets.csv", targets)
    ref = _ref_ppm()

    def pf(tic, **_kw):
        return _products(ref, ref, n_transits=10, sectors=(41,))
    rep = D.direct_measure(_conf(), out, shard=0, n_shards=1, products_fn=pf,
                           tic_fn=lambda *a, **k: (5555.0, "tic_region_kic"), budget_s=0.0)
    assert rep["budget_exhausted"]
    assert rep["n_measured_this_run"] == 0 and rep["n_not_reached_budget"] >= 1
    s = D.direct_assess(_conf(), out)
    assert s["verdict"] == D.RUN_NO_DATA
    assert s["funnel"]["not_measured_reasons"].get("NOT_REACHED") == 5


def test_assess_aggregates_the_shards_and_the_funnel_is_honest(tmp_path):
    out = tmp_path / "direct"
    out.mkdir()
    targets, _ = D.build_targets(_koi_table(), _ps_table())
    D._write_csv(out / "targets.csv", targets)
    ref = _ref_ppm()
    grow = expected_t14_ratio(K_ROR, B_IMP, 2.0)

    def pf(tic, **_kw):
        if int(tic) == 111:                       # star 1000: grew in both families
            return _products(ref * 2.0, ref * 2.0, duration_days=T14_D * grow)
        if int(tic) == 222:                       # star 2000: PDC-only (crowding)
            return _products(ref * 2.0, ref * 0.8)
        if int(tic) == 333:                       # star 3000: unchanged
            return _products(ref, ref)
        raise RuntimeError("MAST down for this one")
    # only shard 0 of 2 runs: the other star(s) are NOT_REACHED
    D.direct_measure(_conf(), out, shard=0, n_shards=2, products_fn=pf,
                     tic_fn=lambda *a, **k: (5555.0, "tic_region_kic"))
    s = D.direct_assess(_conf(), out)
    f = s["funnel"]
    assert f["n_targets"] == 5
    assert f["n_not_reached"] >= 1
    assert f["n_in_shard_files"] + f["n_not_reached"] == 5
    assert sum(f["classes"].values()) == 5
    meas = pd.read_csv(out / "measurements.csv").set_index("kepoi_name")
    reached = meas[meas["not_measured_reason"].fillna("") != "NOT_REACHED"]
    for _k, r in reached.iterrows():
        tic = int(r["tic_id"]) if np.isfinite(r["tic_id"]) else -1
        if tic == 111:
            assert r["class"] == D.CLASS_GROWTH
        elif tic == 222:
            assert r["class"] == D.CLASS_CROWDING
        elif tic == 333:
            assert r["class"] == D.CLASS_CONSISTENT
        elif tic == 5555:
            assert r["class"] == D.CLASS_NOT_MEASURED and r["not_measured_reason"] == "QUERY_FAILED"
    if f["classes"][D.CLASS_GROWTH]:
        assert s["verdict"] == D.RUN_CANDIDATES
        c = pd.read_csv(out / "candidates.csv")
        assert set(c["class"]) <= set(D.CANDIDATE_CLASSES)
    assert (out / "long_period.csv").exists()
    assert "kepler_era_refit" in " ".join(s["checks_not_performed"])
    assert all(k in s["sensitivity"] for k in ("pdc", "sap"))
    assert isinstance(s["population"]["z_robust_scatter"]["pdc"], float)


def test_the_population_scatter_scales_z_before_the_gate():
    rng = np.random.default_rng(1)
    n = 200
    ln = rng.normal(0.1, 0.30, n)          # a population whose true scatter is 0.30 ...
    sig = np.full(n, 0.10)                 # ... against quoted errors of 0.10
    meas = pd.DataFrame({"pdc_ln_ratio": ln, "pdc_sigma_ln": sig, "pdc_measured_snr": 20.0,
                         "sap_ln_ratio": ln - 0.2, "sap_sigma_ln": sig, "sap_measured_snr": 20.0,
                         "ephemeris_recovered": True})
    off = D.population_offsets(meas, params=D.ClassifyParams(min_population_for_offset=20))
    assert abs(off["median_ln_ratio"]["pdc"] - 0.1) < 0.05
    assert 2.3 < off["z_robust_scatter"]["pdc"] < 3.8
    assert off["pdc_scale"] > 2.0 and abs(off["sap_offset"] - (-0.1)) < 0.05
    # a 5-sigma-on-the-quoted-error outlier is 5/scale in the population's own units
    rec = {"lc_status": "OK", "not_measured_reason": "", "pdc_comparison_status": "OK",
           "sap_comparison_status": "OK", "pdc_expected_snr": 20.0, "ephemeris_recovered": True,
           "pdc_ln_ratio": 0.1 + 0.5, "pdc_sigma_ln": 0.10, "sap_ln_ratio": 0.5,
           "sap_sigma_ln": 0.10, "sap_z": 5.0, "duration_verdict": "fixed_b",
           "koi_disposition": "CONFIRMED"}
    c = D.classify_direct(rec, params=D.ClassifyParams(), offsets=off)
    assert c["pdc_z_pop"] == pytest.approx((0.6 - off["pdc_offset"]) / (0.10 * off["pdc_scale"]),
                                           rel=1e-6)
    assert c["pdc_z_pop"] < 2.5
    assert c["class"] != D.CLASS_GROWTH
    c0 = D.classify_direct(rec, params=D.ClassifyParams(), offsets=None)
    assert c0["class"] == D.CLASS_GROWTH


def test_the_probe_records_the_routes_without_touching_the_network(tmp_path):
    rep = D.direct_probe(_conf(), tmp_path)
    assert "mast" in rep and "tic_fallback_importable" in rep
    assert (tmp_path / "probe.json").exists()


def test_the_vet_stage_hands_survivors_to_stage2_and_stage3(tmp_path, monkeypatch):
    out = tmp_path / "direct"
    out.mkdir()
    D._write_csv(out / "candidates.csv", pd.DataFrame([{
        "kepoi_name": "K00897.01", "kepler_name": "Kepler-718 b", "kepid": 7849854,
        "tic_id": 268924036.0, "class": D.CLASS_GROWTH, "pdc_z_pop": 7.0, "sap_z": 6.0}]))
    seen = {}

    def fake_stage2_measure(conf, s2_dir, *, shortlist=None, **kw):
        seen["s2"] = list(shortlist["kepoi_name"])
        Path(s2_dir).mkdir(parents=True, exist_ok=True)

    def fake_stage2_assess(conf, s2_dir):
        return {"primary_verdict": "MEASURED_DEPTH_CHANGE_CONFIRMED",
                "targets": [{"kepoi_name": "K00897.01",
                             "like_for_like_verdict": "MEASURED_DEPTH_CHANGED",
                             "z_measured_eras": 6.0, "sap_vs_pdcsap_verdict": "BACKGROUND_TEST_AGREES"}]}

    def fake_centroid_run(stage, *, out_dir=None, conf=None, shortlist=None, **kw):
        seen["c3"] = list(shortlist["kepoi_name"])
        return {"verdict": "TRANSIT_ON_TARGET",
                "targets": [{"kepoi_name": "K00897.01", "verdict": "TRANSIT_ON_TARGET",
                             "offset_arcsec": 0.3, "offset_sigma": 0.2}]}
    import seti.growth.centroid as C
    import seti.growth.stage2 as S2
    monkeypatch.setattr(S2, "stage2_measure", fake_stage2_measure)
    monkeypatch.setattr(S2, "stage2_assess", fake_stage2_assess)
    monkeypatch.setattr(C, "centroid_run", fake_centroid_run)
    rep = D.direct_vet(_conf(), out)
    assert seen["s2"] == ["K00897.01"] and seen["c3"] == ["K00897.01"]
    assert rep["n_vetted"] == 1 and rep["n_survive_vet"] == 1
    v = json.loads((out / "vet" / "summary.json").read_text())
    assert v["targets"][0]["survives_vet"] is True
    # and with nothing to vet, nothing is fabricated
    D._write_csv(out / "candidates.csv", pd.DataFrame())
    rep0 = D.direct_vet(_conf(), out)
    assert rep0["n_vetted"] == 0


def test_the_vet_stage_shards_round_robin_so_every_candidate_can_be_reached(tmp_path,
                                                                           monkeypatch):
    """With more candidates than one job's cap, the shards partition them and
    the gather says which were never vetted --- an open question, not a pass."""
    out = tmp_path / "direct"
    out.mkdir()
    names = [f"K0{900 + i}.01" for i in range(10)]
    D._write_csv(out / "candidates.csv", pd.DataFrame([
        {"kepoi_name": n, "kepler_name": "", "kepid": 7000000 + i, "tic_id": 1.0 * i,
         "class": D.CLASS_GROWTH, "pdc_z_pop": 20.0 - i, "sap_z": 6.0}
        for i, n in enumerate(names)]))
    per_shard: dict = {}

    def fake_stage2_measure(conf, s2_dir, *, shortlist=None, **kw):
        per_shard.setdefault("s2", []).append(list(shortlist["kepoi_name"]))
        Path(s2_dir).mkdir(parents=True, exist_ok=True)

    def fake_stage2_assess(conf, s2_dir):
        return {"primary_verdict": "X", "targets": [
            {"kepoi_name": n, "like_for_like_verdict": "MEASURED_DEPTH_CHANGED",
             "z_measured_eras": 6.0, "sap_vs_pdcsap_verdict": "BACKGROUND_TEST_AGREES"}
            for n in names]}

    def fake_centroid_run(stage, *, out_dir=None, conf=None, shortlist=None, **kw):
        return {"verdict": "TRANSIT_ON_TARGET", "targets": [
            {"kepoi_name": n, "verdict": "TRANSIT_ON_TARGET", "offset_arcsec": 0.3,
             "offset_sigma": 0.2} for n in names]}
    import seti.growth.centroid as C
    import seti.growth.stage2 as S2
    monkeypatch.setattr(S2, "stage2_measure", fake_stage2_measure)
    monkeypatch.setattr(S2, "stage2_assess", fake_stage2_assess)
    monkeypatch.setattr(C, "centroid_run", fake_centroid_run)
    conf = _conf(classify={"vet_max_targets": 3})
    for sh in range(4):
        r = D.direct_vet(conf, out, shard=sh, n_shards=4)
        assert r["shard"] == sh and r["n_candidates"] == 10
        assert (out / "vet" / f"shard_{sh:02d}" / "vetted.csv").exists()
    # round-robin BY RANK: no shard gets only the strongest candidates
    assert per_shard["s2"][0][0] == names[0] and per_shard["s2"][1][0] == names[1]
    g = D.direct_vet_gather(conf, out)
    # 4 shards x cap 3, but shards hold 3,3,2,2 candidates -> all 10 reachable
    assert g["n_vetted"] == 10 and g["n_candidates_not_vetted"] == 0
    assert g["n_survive_vet"] == 10 and g["n_shards_found"] == 4
    merged = pd.read_csv(out / "vet" / "vetted.csv")
    assert len(merged) == 10 and set(merged["kepoi_name"]) == set(names)
    # a candidate no shard vetted is REPORTED, never silently passed
    D._write_csv(out / "candidates.csv", pd.DataFrame([
        *[{"kepoi_name": n, "kepid": 1, "tic_id": 1.0, "class": D.CLASS_GROWTH,
           "pdc_z_pop": 1.0, "sap_z": 1.0} for n in names],
        {"kepoi_name": "K00999.99", "kepid": 2, "tic_id": 2.0, "class": D.CLASS_GROWTH,
         "pdc_z_pop": 1.0, "sap_z": 1.0}]))
    g2 = D.direct_vet_gather(conf, out)
    assert g2["n_candidates_not_vetted"] == 1
    assert g2["not_vetted"][0]["kepoi_name"] == "K00999.99"


def test_the_cli_exposes_growth_direct():
    from seti.cli import main as cli_main
    with pytest.raises(SystemExit) as e:
        cli_main(["growth-direct", "--help"])
    assert e.value.code == 0


def test_the_repository_config_carries_the_direct_block():
    from seti.growth.run import load_growth_config
    conf = load_growth_config()
    d = conf.get("direct") or {}
    assert d, "config/growth.yaml has no direct: block"
    assert float(d["fetch"]["shard_budget_s"]) <= 17400.0
    assert float(d["classify"]["n_candidate"]) == 5.0
    assert D.FetchParams.from_config(conf).max_products >= 20
    assert math.isclose(D.ClassifyParams.from_config(conf).sigma_sys_ln, 0.05)


# ---------------------------------------------------------------------------
# The FITS product, through BOTH readers (no network: a file written here)
# ---------------------------------------------------------------------------
def _fake_tess_lc_fits(path: Path, *, sector=41, n=3000, exptime_s=120.0) -> np.ndarray:
    from astropy.io import fits
    t = 2459400.0 + np.arange(n) * exptime_s / 86400.0        # BJD
    sap = 1000.0 + np.zeros(n)
    pdc = 900.0 + np.zeros(n)
    q = np.zeros(n, dtype=np.int32)
    q[::100] = 8                                                 # flagged cadences
    pdc[5] = np.nan
    cols = fits.ColDefs([
        fits.Column(name="TIME", format="D", unit="BJD - 2457000, days", array=t - 2457000.0),
        fits.Column(name="SAP_FLUX", format="E", array=sap),
        fits.Column(name="SAP_FLUX_ERR", format="E", array=np.full(n, 2.0)),
        fits.Column(name="PDCSAP_FLUX", format="E", array=pdc),
        fits.Column(name="PDCSAP_FLUX_ERR", format="E", array=np.full(n, 1.5)),
        fits.Column(name="QUALITY", format="J", array=q),
    ])
    hdu = fits.BinTableHDU.from_columns(cols, name="LIGHTCURVE")
    hdu.header["BJDREFI"] = 2457000
    hdu.header["BJDREFF"] = 0.0
    hdu.header["TIMEDEL"] = exptime_s / 86400.0
    hdu.header["TIMESYS"] = "TDB"
    hdu.header["TIMEUNIT"] = "d"
    pri = fits.PrimaryHDU()
    pri.header["TELESCOP"] = "TESS"
    pri.header["SECTOR"] = sector
    pri.header["TICID"] = 268924036
    pri.header["ORIGIN"] = "NASA/Ames"
    pri.header["PROCVER"] = "spoc-test"
    fits.HDUList([pri, hdu]).writeto(path, overwrite=True)
    return q


def test_a_fits_product_yields_both_families_through_both_readers(tmp_path):
    lk = pytest.importorskip("lightkurve")
    from lightkurve.io.tess import read_tess_lightcurve
    p = tmp_path / "tess-fake-s0041-lc.fits"
    q = _fake_tess_lc_fits(p)
    n_good = int((q == 0).sum())
    # the astroquery route's reader
    rec = D.read_tess_lc_fits_all(p)
    assert rec is not None and set(rec["fluxes"]) == {"SAP_FLUX", "PDCSAP_FLUX"}
    assert rec["sector"] == 41 and rec["n_points"] == n_good
    assert abs(rec["exptime_s"] - 120.0) < 1e-6
    # BTJD through BJDREFI/BJDREFF; cadence 0 is flagged, so the first kept point is 120 s on
    first = 2400.0 + 120.0 / 86400.0
    assert abs(rec["time"][0] - first) < 1e-6
    # the lightkurve route: quality-masked by lightkurve, every column kept
    lc = read_tess_lightcurve(str(p), quality_bitmask="default")
    prod = D.lightcurve_to_product(lc, fallback_sector=41, fallback_author="SPOC")
    assert prod is not None and set(prod["fluxes"]) == {"SAP_FLUX", "PDCSAP_FLUX"}
    assert prod["n_points"] == n_good and prod["author"] == "SPOC"
    assert abs(prod["exptime_s"] - 120.0) < 1e-3
    assert abs(prod["time"][0] - first) < 1e-6
    f, fe = prod["fluxes"]["SAP_FLUX"]
    assert np.nanmedian(f) == pytest.approx(1000.0) and np.nanmedian(fe) == pytest.approx(2.0)
    # a NaN stays a NaN (original cadence 5 is index 4 once flagged cadence 0 is dropped)
    assert np.isnan(prod["fluxes"]["PDCSAP_FLUX"][0][4])
    # and the family segments pick the right column per family
    segs = D.family_segments([prod], D.FAMILY_PDC)
    assert len(segs) == 1 and segs[0]["flux_column"] == "PDCSAP_FLUX"
    segs = D.family_segments([prod], D.FAMILY_SAP)
    assert len(segs) == 1 and segs[0]["flux_column"] == "SAP_FLUX"
    assert lk.__version__


# ---------------------------------------------------------------------------
# The control-phase null --- the look-elsewhere bias of the epoch search
# ---------------------------------------------------------------------------
def _ctrl_conf(**over) -> dict:
    c = _conf()
    c["direct"].setdefault("epoch_search", {}).update(
        {"control_phases": [0.15, 0.25, 0.75, 0.85]})
    c["direct"]["epoch_search"].update(over)
    return c


def test_a_real_transit_beats_its_own_control_phase_null():
    """The injected transit's search S/N is above anything the same search finds
    at phases where the planet is not."""
    ref = _ref_ppm()
    conf = _ctrl_conf()
    prods = _products(ref * 2.0, ref * 2.0, span_durations=8.0)
    rec, *_ = _measure(prods, conf=conf)
    assert rec["ephemeris_recovered"]
    for fam in ("pdc", "sap"):
        ctrl = D.control_phase_null(prods, fam, period_days=P, t0_btjd=rec["t0_btjd_used"],
                                    duration_days=T14_D,
                                    sigma_t0_days=rec["ephemeris_sigma_minutes"] / 1440.0,
                                    depth_ppm=rec[f"{fam}_depth_ppm"], ref_ppm=ref,
                                    fit=D.FitParams.from_config(conf),
                                    params=D.EpochSearchParams.from_config(conf))
        assert ctrl["control_n_phases_measured"] >= 3
        assert np.isfinite(ctrl["control_snr_max"])
        out = D.apply_control(rec, ctrl, family=fam)
        assert out["control_verdict"] == D.CTRL_ABOVE
        assert out["snr_excess"] > 0
        # the search cannot manufacture the injected excess out of this noise
        assert out["depth_excess_over_control_ppm"] > 0


def test_the_control_null_calls_a_search_that_did_not_beat_noise_within_search_noise():
    """With the transit's own S/N no better than the off-phase maxima, the
    verdict is WITHIN_SEARCH_NOISE --- the gate the max-over-trials needs."""
    ref = _ref_ppm()
    conf = _ctrl_conf()
    prods = _products(ref * 2.0, ref * 2.0, span_durations=8.0)
    rec, *_ = _measure(prods, conf=conf)
    ctrl = D.control_phase_null(prods, "pdc", period_days=P, t0_btjd=rec["t0_btjd_used"],
                                duration_days=T14_D, sigma_t0_days=0.0,
                                depth_ppm=rec["pdc_depth_ppm"], ref_ppm=ref,
                                fit=D.FitParams.from_config(conf),
                                params=D.EpochSearchParams.from_config(conf))
    # a record whose search S/N is only as good as the best off-phase maximum
    weak = dict(rec)
    weak["epoch_search_snr_best"] = float(ctrl["control_snr_max"]) - 0.5
    out = D.apply_control(weak, ctrl, family="pdc")
    assert out["control_verdict"] == D.CTRL_WITHIN
    assert out["snr_excess"] < 0


def test_an_empty_control_is_unavailable_and_never_a_pass():
    ctrl = D.control_phase_null([], "pdc", period_days=P, t0_btjd=0.0, duration_days=T14_D,
                                sigma_t0_days=0.0, depth_ppm=1e4, ref_ppm=1e4)
    assert ctrl["control_verdict"] == D.CTRL_UNAVAILABLE
    out = D.apply_control({"epoch_search_snr_best": 99.0}, ctrl, family="pdc")
    assert out["control_verdict"] == D.CTRL_UNAVAILABLE


def test_direct_control_reopens_every_changed_class_and_records_the_null(tmp_path):
    ref = _ref_ppm()
    conf = _ctrl_conf()
    prods = _products(ref * 2.0, ref * 2.0, span_durations=8.0)
    rec, *_ = _measure(prods, conf=conf)
    row = dict(rec)
    row["class"] = D.CLASS_GROWTH
    # a second planet on a star whose TIC never resolved: unavailable, not a pass
    row2 = dict(rec)
    row2.update({"kepoi_name": "K00999.01", "kepid": 1234567, "tic_id": float("nan"),
                 "class": D.CLASS_DEEPER})
    # a `consistent` planet is NOT re-opened: the bias can only push a depth up
    row3 = dict(rec)
    row3.update({"kepoi_name": "K00888.01", "kepid": 7654321, "class": D.CLASS_CONSISTENT})
    out = tmp_path / "direct"
    out.mkdir()
    pd.DataFrame([row, row2, row3]).to_csv(out / "measurements.csv", index=False)
    rep = D.direct_control(conf, out, products_fn=lambda tic, **kw: list(prods))
    assert rep["n_selected"] == 2
    df = pd.read_csv(out / "control" / "control.csv")
    assert set(df["kepoi_name"]) == {"K00897.01", "K00999.01"}
    got = df.set_index("kepoi_name")
    assert got.loc["K00897.01", "control_verdict"] == D.CTRL_ABOVE
    assert got.loc["K00897.01", "pdc_control_n_phases_measured"] >= 3
    assert got.loc["K00999.01", "control_status"] == "TIC_UNRESOLVED"
    assert got.loc["K00999.01", "control_verdict"] == D.CTRL_UNAVAILABLE
    assert rep["n_above_control"] == 1 and rep["n_control_unavailable"] == 1
    s = json.loads((out / "control" / "summary.json").read_text())
    assert s["control_phases"] and s["caveats"]


def test_resume_refuses_a_row_measured_against_an_unverified_tic(tmp_path):
    """The shards already committed carry non-detections that are artefacts of a
    truncated id.  Resume must redo them, not lock them in as facts."""
    assert D.record_tic_is_unverified({"tic_id": 122785300.0, "tic_route": "name_planet"})
    assert not D.record_tic_is_unverified({"tic_id": 122785305.0, "tic_route": "name_planet"})
    assert not D.record_tic_is_unverified({"tic_id": 122785300.0,
                                           "tic_route": "tic_kic_crossid_confirms_name_planet"})
    assert not D.record_tic_is_unverified({"tic_id": 122785300.0,
                                           "tic_route": "tic_kic_crossid"})

    out = tmp_path / "direct"
    out.mkdir()
    targets, _ = D.build_targets(_koi_table(), _ps_table())
    D._write_csv(out / "targets.csv", targets)
    names = list(targets["kepoi_name"].astype(str))
    # what the old code committed: a zero-rows non-detection on a truncated id,
    # beside a genuine one on a TIC the sky had named
    D._write_csv(D._shard_paths(out, 0)["csv"], pd.DataFrame([
        {"kepoi_name": names[0], "tic_id": 122785300.0, "tic_route": "name_planet",
         "lc_status": D.REASON_ZERO_ROWS, "not_measured_reason": D.REASON_ZERO_ROWS,
         "class": D.CLASS_NOT_MEASURED},
        {"kepoi_name": names[1], "tic_id": 122785300.0, "tic_route": "tic_kic_crossid",
         "lc_status": D.REASON_ZERO_ROWS, "not_measured_reason": D.REASON_ZERO_ROWS,
         "class": D.CLASS_NOT_MEASURED}]))

    def pf(tic, **_kw):
        return []

    def tf(kepid, *a, **k):
        return float("nan"), ""
    rep = D.direct_measure(_conf(), out, shard=0, n_shards=1, products_fn=pf, tic_fn=tf)
    assert rep["n_redone_unverified_tic"] == 1
    df = D._read_csv(D._shard_paths(out, 0)["csv"]).set_index("kepoi_name")
    # the unverified row came back from the target list, no longer carrying the
    # truncated id; the row the sky had already named was kept untouched
    assert int(pd.to_numeric(df.loc[names[0], "tic_id"])) != 122785300
    assert int(pd.to_numeric(df.loc[names[1], "tic_id"])) == 122785300
    assert df.loc[names[1], "lc_status"] == D.REASON_ZERO_ROWS
    assert rep["n_measured_this_run"] == 4      # everything but the kept row


def test_a_flaky_tic_query_is_retried_before_a_star_is_refused(tmp_path):
    """Refusing a suspect id makes the resolver load-bearing, so one bad call
    must not become a non-measurement that reads as TESS coverage."""
    out = tmp_path / "direct"
    out.mkdir()
    targets, _ = D.build_targets(_koi_table(), _ps_table())
    good = 122785305.0
    targets = targets.copy()
    targets["tic_id"] = targets["tic_id"].where(targets["tic_id"].isna(), 122785300.0)
    D._write_csv(out / "targets.csv", targets)
    ref = _ref_ppm()
    tries: list[int] = []

    def tf(kepid, *a, **k):
        tries.append(int(kepid))
        if tries.count(int(kepid)) == 1:
            raise TimeoutError("MAST timed out")
        return good, "tic_kic_crossid"

    def pf(tic, **_kw):
        assert int(tic) == int(good)               # the suspect id is never queried
        return _products(ref, ref, n_transits=10, sectors=(41,))
    conf = _conf(fetch={"retries": 3, "retry_pause_s": 0.0})
    rep = D.direct_measure(conf, out, shard=0, n_shards=1, products_fn=pf, tic_fn=tf)
    assert rep["n_tic_suspect_truncation"] >= 1
    df = D._read_csv(D._shard_paths(out, 0)["csv"])
    assert (df["lc_status"] == "OK").any()         # the retry rescued it
    assert not (df["lc_status"] == D.REASON_TIC_UNRESOLVED).all()


def test_assess_declares_the_rows_that_still_rest_on_an_unchecked_tic(tmp_path):
    """The first summary.json mixes pre-repair shards with repaired ones.  It
    must say so, or its funnel reads as coverage that was tried and failed."""
    out = tmp_path / "direct"
    out.mkdir()
    targets, _ = D.build_targets(_koi_table(), _ps_table())
    D._write_csv(out / "targets.csv", targets)
    names = list(targets["kepoi_name"].astype(str))
    D._write_csv(D._shard_paths(out, 0)["csv"], pd.DataFrame([
        # written before the repair: a truncated id on a bare catalogue route
        {"kepoi_name": names[0], "tic_id": 122785300.0, "tic_route": "name_planet",
         "lc_status": D.REASON_ZERO_ROWS, "not_measured_reason": D.REASON_ZERO_ROWS,
         "class": D.CLASS_NOT_MEASURED},
        # written after: the sky named this star
        {"kepoi_name": names[1], "tic_id": 122785305.0,
         "tic_route": "tic_kic_crossid_over_name_planet",
         "lc_status": D.REASON_ZERO_ROWS, "not_measured_reason": D.REASON_ZERO_ROWS,
         "class": D.CLASS_NOT_MEASURED}]))
    s = D.direct_assess(_conf(), out)
    assert s["funnel"]["n_rows_pending_tic_recheck"] == 1
    assert s["funnel"]["tic_routes"].get("tic_kic_crossid_over_name_planet") == 1
    assert any("never checked against the sky" in d for d in s["degraded"])
    assert s["verdict"] == D.RUN_NO_DATA          # and still not a statement about the sky
