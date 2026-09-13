"""Offline test suite for GROWTH --- the growing transit (S57).

No network anywhere (``conftest.py`` raises on any socket).  The suite is the
CI gate and covers, per ``docs/channel-brief.md`` §5:

* an injected 1.5x TESS depth with the duration consistent at fixed b is a
  ``GROWTH_CANDIDATE`` and the run's verdict is ``DEPTH_DRIFT_CANDIDATES``;
* the same depth change with the duration tracking a changed b (precession)
  is not;
* a uniform 12 % TESS deficit population (Han et al. 2025) yields only
  CONSISTENT / SHALLOWER_TESS and zero candidates;
* a Gaia neighbour able to supply the change vetoes the candidate;
* the limb-darkening band ratio against a hand computation at one Teff;
* an unreachable or empty archive is ``NO_DATA_REACHED`` and never a candidate;
* every veto rule has a case that trips it, and every counter is populated.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seti.growth import acquire as acq
from seti.growth.drift import (
    CLASS_CANDIDATE,
    CLASS_CONSISTENT,
    CLASS_DEEPER,
    CLASS_SHALLOWER,
    DUR_FIXED_B,
    DUR_TRACKS_B,
    DriftParams,
    aperture_weight,
    b_implied_by_duration,
    band_ratio,
    classify,
    contamination,
    corrected_log_ratio,
    duration_test,
    expected_t14_ratio,
    ld_coefficients,
    ld_depth_factor,
    planet_drift,
    t14_hours,
)
from seti.growth.run import (
    VERDICT_CANDIDATES,
    VERDICT_NO_DATA,
    VERDICT_NONE,
    growth_run,
    load_growth_config,
    long_period_list,
    main,
    screen_table,
    stage_assess,
)
from seti.growth.vet import (
    REPORT_FLAGS,
    VETO_ORDER,
    VetParams,
    grazing,
    host_names,
    neighbour_can_supply_change,
    rejection_counters,
    ttv_system,
    vet_planet,
    vet_table,
)

CONF = load_growth_config()
LD = CONF["limb_darkening"]

TEFF, LOGG, B, A_RS, K = 5800.0, 4.5, 0.3, 15.0, 0.0316


# ---------------------------------------------------------------------------
# synthetic archive tables
# ---------------------------------------------------------------------------
def _population(n: int = 20, *, growth: dict | None = None, deficit: float = 0.0,
                seed: int = 3, scatter: float = 0.01
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """KOI / TOI / ps tables for ``n`` ordinary planets plus an optional anomaly.

    Ordinary planets have TESS depth = Kepler depth x f_LD x (1 - deficit) with
    a per-planet ``scatter`` fractional scatter, and identical durations.
    ``growth`` (a dict
    with ``ratio`` and ``precession``) makes planet 0 anomalous: TESS depth
    ratio x f_LD, and a duration either at fixed b with k -> k sqrt(ratio)
    or at fixed k with b -> 0.8.
    """
    rng = np.random.default_rng(seed)
    koi_rows, toi_rows, ps_rows = [], [], []
    for i in range(n):
        period = 5.0 + 0.7 * i
        depth_k = 1000.0
        k = math.sqrt(depth_k * 1e-6)
        t14 = t14_hours(period, A_RS, k, B)
        f_ld, _ = band_ratio(TEFF, LOGG, B, LD)
        depth_t = depth_k * f_ld * (1.0 - deficit) * (1.0 + scatter * rng.standard_normal())
        dur_t = t14
        if growth is not None and i == 0:
            depth_t = depth_k * f_ld * float(growth["ratio"])
            if growth.get("precession"):
                dur_t = t14_hours(period, A_RS, k, 0.8)
            else:
                dur_t = t14_hours(period, A_RS, k * math.sqrt(float(growth["ratio"])), B)
        name = f"Kepler-{1000 + i} b"
        koi_rows.append({
            "kepid": 10000000 + i, "kepoi_name": f"K{100 + i:05d}.01", "kepler_name": name,
            "koi_disposition": "CONFIRMED", "koi_pdisposition": "CANDIDATE",
            "koi_period": period, "koi_period_err1": 1e-5, "koi_depth": depth_k,
            "koi_depth_err1": 30.0, "koi_depth_err2": -30.0, "koi_ror": k,
            "koi_ror_err1": 0.001, "koi_ror_err2": -0.001, "koi_impact": B,
            "koi_impact_err1": 0.05, "koi_impact_err2": -0.05, "koi_duration": t14,
            "koi_duration_err1": 0.05, "koi_duration_err2": -0.05, "koi_ingress": 0.1,
            "koi_dor": A_RS, "koi_steff": TEFF, "koi_slogg": LOGG, "koi_srad": 1.0,
            "koi_kepmag": 13.0, "ra": 290.0 + 0.01 * i, "dec": 44.0,
            "koi_fpflag_nt": 0, "koi_fpflag_ss": 0, "koi_fpflag_co": 0, "koi_fpflag_ec": 0,
            "koi_tce_delivname": "q1_q17_dr25_tce", "koi_count": 1})
        toi_rows.append({
            "toi": 5000.01 + i, "tid": 200000 + i, "tfopwg_disp": "KP", "pl_orbper": period,
            "pl_trandep": depth_t, "pl_trandeperr1": 40.0, "pl_trandeperr2": -40.0,
            "pl_trandurh": dur_t, "pl_trandurherr1": 0.05, "pl_trandurherr2": -0.05,
            "pl_rade": 3.4, "st_teff": TEFF, "st_logg": LOGG, "st_rad": 1.0, "st_tmag": 12.5,
            "ra": 290.0 + 0.01 * i + 1e-5, "dec": 44.0 + 1e-5})
        # ``tic_id`` is spelled the way the Exoplanet Archive spells it: a
        # STRING with a "TIC " prefix (the 2026-09-13 defect).
        ps_rows.append({"pl_name": name, "hostname": name[:-2], "tic_id": f"TIC {200000 + i}",
                        "default_flag": 1, "pl_ratror": k, "pl_trandep": depth_k / 1e4,
                        "pl_orbper": period, "disc_facility": "Kepler",
                        "pl_refname": "synthetic", "ra": 290.0 + 0.01 * i, "dec": 44.0})
    ps = pd.DataFrame(ps_rows)
    ps["pl_trandep_ppm"] = ps["pl_trandep"] * 1e4
    return pd.DataFrame(koi_rows), pd.DataFrame(toi_rows), ps


def _joined(koi, toi, ps):
    j, rep = acq.join_kepler_tess(koi, toi, ps)
    assert rep["n_joined"] == len(koi), rep
    return j


def _status(joined, ok=True):
    return pd.DataFrame({"planet_key": joined["planet_key"].astype(str),
                         "neighbours_status": "OK" if ok else "QUERY_FAILED",
                         "neighbours_route": "upload" if ok else ""})


def _neighbours(joined, extra: list[dict] | None = None) -> pd.DataFrame:
    """Every target sees itself at 0.1"; ``extra`` adds neighbours to planet 0."""
    rows = [{"key": i, "planet_key": k, "source_id": 1000 + i, "ra": ra, "dec": dec,
             "phot_g_mean_mag": 13.0, "sep_arcsec": 0.1}
            for i, (k, ra, dec) in enumerate(zip(joined["planet_key"], joined["ra"],
                                                 joined["dec"], strict=True))]
    for e in extra or []:
        rows.append({"key": 0, "planet_key": joined["planet_key"].iloc[0], "source_id": 999,
                     "ra": joined["ra"].iloc[0], "dec": joined["dec"].iloc[0], **e})
    return pd.DataFrame(rows)


def _screen_and_vet(koi, toi, ps, *, extra_neighbours=None, status_ok=True, conf=None):
    conf = conf or CONF
    j = _joined(koi, toi, ps)
    scr = screen_table(j, _neighbours(j, extra_neighbours), _status(j, status_ok), conf)
    vetted, pop = vet_table(scr, VetParams.from_config(conf))
    return vetted, pop


# ---------------------------------------------------------------------------
# (a) limb darkening --- hand computation
# ---------------------------------------------------------------------------
def test_ld_band_ratio_against_hand_computation_at_5800k():
    u1k, u2k, note = ld_coefficients(5800.0, 4.5, "kepler", LD)
    u1t, u2t, _ = ld_coefficients(5800.0, 4.5, "tess", LD)
    assert note == ""
    assert (u1k, u2k) == (0.40, 0.26) and (u1t, u2t) == (0.31, 0.25)
    # b = 0: I(1)/<I> = 1 / (1 - u1/3 - u2/6)
    fk = 1.0 / (1.0 - 0.40 / 3.0 - 0.26 / 6.0)
    ft = 1.0 / (1.0 - 0.31 / 3.0 - 0.25 / 6.0)
    assert math.isclose(ld_depth_factor(0.40, 0.26, 0.0), fk, rel_tol=1e-12)
    assert math.isclose(ld_depth_factor(0.31, 0.25, 0.0), ft, rel_tol=1e-12)
    r, info = band_ratio(5800.0, 4.5, 0.0, LD)
    assert math.isclose(r, ft / fk, rel_tol=1e-12)
    assert math.isclose(r, 0.96297, rel_tol=2e-4)
    # b = 0.9: mu = sqrt(1 - 0.81); the limb is darker in Kepler than TESS
    mu = math.sqrt(1.0 - 0.81)
    ik = (1.0 - 0.40 * (1 - mu) - 0.26 * (1 - mu) ** 2) * fk
    it = (1.0 - 0.31 * (1 - mu) - 0.25 * (1 - mu) ** 2) * ft
    r9, _ = band_ratio(5800.0, 4.5, 0.9, LD)
    assert math.isclose(r9, it / ik, rel_tol=1e-12) and r9 > 1.0 > r


def test_ld_interpolates_clamps_and_reports_missing_teff():
    u1a, _, _ = ld_coefficients(5500.0, 4.5, "kepler", LD)
    u1b, _, _ = ld_coefficients(5800.0, 4.5, "kepler", LD)
    u1m, _, _ = ld_coefficients(5650.0, 4.5, "kepler", LD)
    assert min(u1a, u1b) < u1m < max(u1a, u1b)
    _, _, note = ld_coefficients(9000.0, 4.5, "tess", LD)
    assert "clamped" in note
    _, _, note = ld_coefficients(float("nan"), float("nan"), "tess", LD)
    assert "teff_missing" in note and "logg_missing" in note


# ---------------------------------------------------------------------------
# (b) dilution
# ---------------------------------------------------------------------------
def test_aperture_weight_and_contamination():
    # 0.5 erfc(-21 / (sqrt2 * 10.5)) = 0.5 erfc(-sqrt2) = 0.977 at the centre
    assert math.isclose(aperture_weight(0.0, 21.0, 10.5), 0.9772, rel_tol=1e-3)
    assert math.isclose(aperture_weight(21.0, 21.0, 10.5), 0.5, rel_tol=1e-9)
    assert aperture_weight(60.0, 21.0, 10.5) < 1e-3
    cw, cmax = contamination([0.0, 2.5], [1.0, 60.0], r_ap_arcsec=21.0, psf_sigma_arcsec=10.5)
    assert math.isclose(cmax, 1.0 + 10 ** (-1.0), rel_tol=1e-9)
    # the near neighbour carries w(1") = 0.9715; the far one is weighted out
    assert math.isclose(cw, aperture_weight(1.0, 21.0, 10.5) + 0.1 * aperture_weight(60.0, 21.0, 10.5))
    assert 0.96 < cw < 0.98
    assert contamination([], [], r_ap_arcsec=21.0, psf_sigma_arcsec=10.5) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# (c) corrected ratio
# ---------------------------------------------------------------------------
def test_corrected_log_ratio_terms_and_floor():
    ln, sig, raw = corrected_log_ratio(1000.0, 30.0, 1500.0, 40.0, ld_ratio=0.963, contam=0.2,
                                       sigma_sys=0.05)
    assert math.isclose(raw, math.log(1.5))
    assert math.isclose(ln, math.log(1.5) - math.log(0.8) - math.log(0.963))
    assert math.isclose(sig, math.sqrt(0.03 ** 2 + (40 / 1500) ** 2 + 0.05 ** 2))
    assert all(math.isnan(v) for v in corrected_log_ratio(float("nan"), 1, 1, 1))
    _, sig2, _ = corrected_log_ratio(1000.0, float("nan"), 1500.0, 40.0)
    assert math.isnan(sig2)


# ---------------------------------------------------------------------------
# (d) duration test
# ---------------------------------------------------------------------------
def test_t14_formula_against_hand_value():
    # P = 10 d, a/R* = 20, k = 0.1, b = 0: T14 = (P/pi) asin(1.1 / 20)
    assert math.isclose(t14_hours(10.0, 20.0, 0.1, 0.0), 240.0 / math.pi * math.asin(1.1 / 20),
                        rel_tol=1e-12)
    assert math.isnan(t14_hours(10.0, 20.0, 0.1, 1.2))      # no chord
    assert math.isnan(t14_hours(10.0, 0.5, 0.1, 0.0))       # a/R* <= 1


def test_expected_duration_ratio_and_implied_b():
    r = expected_t14_ratio(K, B, 1.5)
    k2 = K * math.sqrt(1.5)
    assert math.isclose(r, math.sqrt(((1 + k2) ** 2 - B ** 2) / ((1 + K) ** 2 - B ** 2)))
    assert 1.0 < r < 1.02
    r_full = expected_t14_ratio(K, B, 1.5, a_rs=A_RS, period_days=10.0)
    assert math.isclose(r_full, r, rel_tol=1e-3)
    # the b that reproduces a given duration ratio at fixed k, round trip
    obs = t14_hours(10.0, A_RS, K, 0.8) / t14_hours(10.0, A_RS, K, B)
    assert math.isclose(b_implied_by_duration(K, B, obs), 0.8, rel_tol=2e-3)


def test_duration_test_verdicts():
    t0 = t14_hours(10.0, A_RS, K, B)
    t1 = t14_hours(10.0, A_RS, K * math.sqrt(1.5), B)
    d = duration_test(K, B, 1.5, t0, 0.05, t1, 0.05, a_rs=A_RS, period_days=10.0)
    assert d["duration_verdict"] == DUR_FIXED_B and abs(d["z_duration"]) < 0.5
    tb = t14_hours(10.0, A_RS, K, 0.8)
    d = duration_test(K, B, 1.5, t0, 0.05, tb, 0.05, a_rs=A_RS, period_days=10.0)
    assert d["duration_verdict"] == DUR_TRACKS_B and d["z_duration"] < -3
    assert math.isclose(d["b_implied_by_duration"], 0.8, rel_tol=2e-3)
    d = duration_test(K, B, 1.5, float("nan"), 0.05, t1, 0.05)
    assert d["duration_verdict"] == "inconclusive"


# ---------------------------------------------------------------------------
# (e) classification and the injected signal
# ---------------------------------------------------------------------------
def test_classify_boundaries():
    assert classify(0.0, 0.05, DUR_FIXED_B, [])[0] == CLASS_CONSISTENT
    assert classify(0.4, 0.05, DUR_FIXED_B, [])[0] == CLASS_CANDIDATE
    assert classify(0.4, 0.05, DUR_TRACKS_B, [])[0] == CLASS_DEEPER
    assert classify(0.4, 0.05, DUR_FIXED_B, ["grazing"])[0] == CLASS_DEEPER
    assert classify(0.2, 0.05, DUR_FIXED_B, [])[0] == CLASS_DEEPER      # 4 sigma: not 5
    assert classify(-0.4, 0.05, DUR_FIXED_B, [])[0] == CLASS_SHALLOWER
    assert classify(0.4, 0.05, DUR_FIXED_B, [], offset=0.4)[0] == CLASS_CONSISTENT
    assert classify(float("nan"), 0.05, DUR_FIXED_B, [])[0] == "UNMEASURED"


def test_injected_growth_at_fixed_b_is_a_candidate():
    koi, toi, ps = _population(20, growth={"ratio": 1.5})
    vetted, pop = _screen_and_vet(koi, toi, ps)
    top = vetted.iloc[0]
    assert top["class"] == CLASS_CANDIDATE and top["vetoes"] == "", top.to_dict()
    assert top["duration_verdict"] == DUR_FIXED_B
    assert top["z_ratio"] >= 5 and math.isclose(top["ln_ratio_corr"], math.log(1.5), abs_tol=0.02)
    assert abs(pop["population_offset"]) < 0.02
    assert (vetted["class"].iloc[1:] == CLASS_CONSISTENT).all()
    c = rejection_counters(vetted)
    assert c["n_candidates"] == 1 and c["classes"][CLASS_CANDIDATE] == 1


def test_growth_whose_duration_tracks_a_changed_b_is_not_a_candidate():
    koi, toi, ps = _population(20, growth={"ratio": 1.5, "precession": True})
    vetted, _ = _screen_and_vet(koi, toi, ps)
    top = vetted.iloc[0]
    assert top["duration_verdict"] == DUR_TRACKS_B
    assert math.isclose(top["b_implied_by_duration"], 0.8, rel_tol=2e-3)
    assert top["class"] == CLASS_DEEPER and top["z_ratio"] >= 5
    assert rejection_counters(vetted)["n_candidates"] == 0


@pytest.mark.parametrize("subtract", [True, False])
def test_uniform_han2025_deficit_population_yields_no_candidate(subtract):
    koi, toi, ps = _population(24, deficit=0.12)
    conf = json.loads(json.dumps(CONF))
    conf["drift"]["subtract_population_median"] = subtract
    vetted, pop = _screen_and_vet(koi, toi, ps, conf=conf)
    assert set(vetted["class"]) <= {CLASS_CONSISTENT, CLASS_SHALLOWER}
    assert rejection_counters(vetted)["n_candidates"] == 0
    assert math.isclose(pop["population_median_ln_ratio_raw"], math.log(0.88) + math.log(0.963),
                        abs_tol=0.02)
    if subtract:
        assert math.isclose(pop["population_offset"], math.log(0.88), abs_tol=0.02)
        assert (vetted["class"] == CLASS_CONSISTENT).all()
    else:
        assert pop["population_offset"] == 0.0
    assert (vetted["flags"].str.contains("within_han2025_deficit_band")).all()


def test_neighbour_that_can_supply_the_change_is_vetoed():
    koi, toi, ps = _population(20, growth={"ratio": 1.5})
    # a neighbour 0.5 mag fainter, 8" away: 63 % of the target flux in the pixel
    vetted, _ = _screen_and_vet(koi, toi, ps, extra_neighbours=[
        {"phot_g_mean_mag": 13.5, "sep_arcsec": 8.0}])
    top = vetted.iloc[0]
    assert top["n_gaia_neighbours"] == 1 and math.isclose(top["contam_max"], 10 ** -0.2, rel_tol=1e-6)
    assert top["contam_applied"] > 0.5
    assert top["first_veto"] == "neighbour_can_supply_change"
    assert top["class"] == CLASS_DEEPER and top["would_be_candidate_without_vetoes"]
    assert rejection_counters(vetted)["n_candidates"] == 0
    # a faint, distant neighbour cannot
    vetted2, _ = _screen_and_vet(koi, toi, ps, extra_neighbours=[
        {"phot_g_mean_mag": 19.0, "sep_arcsec": 18.0}])
    assert vetted2.iloc[0]["class"] == CLASS_CANDIDATE
    assert neighbour_can_supply_change(0.4, 0.05, 0.63, n_candidate=5.0)
    assert not neighbour_can_supply_change(0.4, 0.05, 0.0, n_candidate=5.0)


def test_unchecked_neighbours_veto_and_are_counted():
    koi, toi, ps = _population(20, growth={"ratio": 1.5})
    vetted, _ = _screen_and_vet(koi, toi, ps, status_ok=False)
    top = vetted.iloc[0]
    assert top["first_veto"] == "neighbours_not_checked" and top["class"] == CLASS_DEEPER
    assert rejection_counters(vetted)["vetoes_raised"]["neighbours_not_checked"] == 20


# ---------------------------------------------------------------------------
# every veto rule trips
# ---------------------------------------------------------------------------
def _growth_record() -> dict:
    koi, toi, ps = _population(3, growth={"ratio": 1.5})
    j = _joined(koi, toi, ps)
    scr = screen_table(j, _neighbours(j), _status(j), CONF)
    return scr.iloc[0].to_dict()


VP = VetParams.from_config(CONF)


@pytest.mark.parametrize("field,value,rule", [
    ("b_kepler", 0.99, "grazing"),
    ("kepler_name", "Kepler-9 b", "ttv_system"),
    ("koi_fpflag_nt", 1, "fpflag_nt"),
    ("koi_fpflag_ss", 1, "fpflag_ss"),
    ("koi_fpflag_co", 1, "fpflag_co"),
    ("koi_fpflag_ec", 1, "fpflag_ec"),
    ("koi_disposition", "FALSE POSITIVE", "koi_false_positive"),
    ("koi_pdisposition", "FALSE POSITIVE", "koi_false_positive"),
    ("tfopwg_disp", "FP", "toi_disposition_not_candidate"),
    ("tfopwg_disp", "APC", "toi_disposition_not_candidate"),
    ("period_alias", "2", "period_alias"),
    ("neighbours_status", "QUERY_FAILED", "neighbours_not_checked"),
    ("contam_max", 0.7, "neighbour_can_supply_change"),
])
def test_each_veto_rule_trips(field, value, rule):
    rec = _growth_record()
    assert vet_planet(rec, VP)["class"] == CLASS_CANDIDATE
    rec[field] = value
    v = vet_planet(rec, VP)
    assert v["first_veto"] == rule and v["class"] != CLASS_CANDIDATE, v
    assert v["would_be_candidate_without_vetoes"]


def test_grazing_and_ttv_helpers():
    assert grazing(0.99, 0.03) and not grazing(0.3, 0.03) and not grazing(float("nan"), 0.03)
    assert ttv_system({"kepoi_name": "K00142.01"}, ["KOI-142"])
    assert ttv_system({"kepoi_name": "K00142.01"}, ["K00142"])
    assert ttv_system({"kepid": 8410697}, ["KIC 8410697"])
    assert not ttv_system({"kepler_name": "Kepler-10 b"}, ["Kepler-100"])
    assert {"kepler-9 b", "kepler-9", "k00377", "koi-377"} <= host_names(
        {"kepler_name": "Kepler-9 b", "kepoi_name": "K00377.01"})


def test_every_counter_is_populated():
    recs = []
    for field, value in (("b_kepler", 0.99), ("kepler_name", "Kepler-9 b"),
                         ("koi_fpflag_nt", 1), ("koi_fpflag_ss", 1), ("koi_fpflag_co", 1),
                         ("koi_fpflag_ec", 1), ("koi_disposition", "FALSE POSITIVE"),
                         ("tfopwg_disp", "FP"), ("period_alias", "2"),
                         ("neighbours_status", "QUERY_FAILED"), ("contam_max", 0.7),
                         ("duration_verdict", "inconclusive"), ("ld_note", "teff_missing")):
        r = _growth_record()
        r[field] = value
        recs.append(r)
    r = _growth_record()
    r["ln_ratio_raw"] = math.log(0.88)
    recs.append(r)
    df = pd.DataFrame(recs)
    vetted, _ = vet_table(df, VetParams(n_candidate=5.0, subtract_population_median=False,
                                        ttv_systems=["Kepler-9"]))
    c = rejection_counters(vetted)
    for rule in VETO_ORDER:
        assert c["vetoes_raised"][rule] >= 1, (rule, c["vetoes_raised"])
        assert c["first_veto"][rule] >= 1 or rule in ("neighbour_can_supply_change",), rule
    for f in REPORT_FLAGS:
        assert c["flags_raised"][f] >= 1, (f, c["flags_raised"])


# ---------------------------------------------------------------------------
# the join
# ---------------------------------------------------------------------------
def test_period_match_and_aliases():
    assert acq.period_match(10.0, 10.005)[0] and acq.period_match(10.0, 10.005)[2] == "1"
    assert not acq.period_match(10.0, 10.05)[0]
    ok, r, alias = acq.period_match(10.0, 20.01)
    assert ok and alias == "2" and math.isclose(r, 2.001)
    ok, _, alias = acq.period_match(10.0, 3.3334)
    assert ok and alias == "1/3"
    assert not acq.period_match(10.0, 50.0, alias_max=4)[0]
    assert not acq.period_match(float("nan"), 5.0)[0]


ZERO_ROUTES = {"name_planet": 0, "name_host": 0, "position_tic": 0, "position_period": 0}


def test_join_routes_are_counted_separately():
    koi = pd.DataFrame([
        {"kepid": 1, "kepoi_name": "K00001.01", "kepler_name": "Kepler-A b", "koi_period": 10.0,
         "ra": 290.0, "dec": 44.0},
        {"kepid": 2, "kepoi_name": "K00002.01", "kepler_name": None, "koi_period": 3.0,
         "ra": 291.0, "dec": 44.0},
        {"kepid": 3, "kepoi_name": "K00003.01", "kepler_name": None, "koi_period": 7.0,
         "ra": 292.0, "dec": 44.0},
        {"kepid": 4, "kepoi_name": "K00004.01", "kepler_name": "Kepler-D b", "koi_period": 4.0,
         "ra": 293.0, "dec": 44.0}])
    toi = pd.DataFrame([
        {"toi": 1.01, "tid": 111, "pl_orbper": 10.002, "ra": 295.0, "dec": 40.0},   # by tic only
        {"toi": 2.01, "tid": 222, "pl_orbper": 6.0, "ra": 291.0 + 1e-4 / 3.6, "dec": 44.0},
        {"toi": 3.01, "tid": 333, "pl_orbper": 7.0, "ra": 292.0 + 0.01, "dec": 44.0},  # 36" off
        {"toi": 4.01, "tid": 444, "pl_orbper": 9.0, "ra": 293.0, "dec": 44.0}])       # P mismatch
    ps = pd.DataFrame([{"pl_name": "Kepler-A b", "hostname": "Kepler-A", "tic_id": "TIC 111",
                        "default_flag": 1, "pl_ratror": 0.03, "pl_trandep_ppm": 900.0,
                        "pl_refname": "x", "disc_facility": "Kepler"},
                       {"pl_name": "Kepler-D b", "hostname": "Kepler-D", "tic_id": "TIC 444",
                        "default_flag": 1, "pl_ratror": 0.03, "pl_trandep_ppm": 900.0,
                        "pl_refname": "x", "disc_facility": "Kepler"}])
    j, rep = acq.join_kepler_tess(koi, toi, ps, pos_radius_arcsec=2.0)
    assert rep["joined_by_route"] == {**ZERO_ROUTES, "name_planet": 1, "position_period": 1}
    assert rep["koi_with_tic_id"] == 2 and rep["tic_matched_no_period_match"] == 1
    assert rep["koi_with_tic_by_route"] == {"name_planet": 2, "name_host": 0, "position_tic": 0}
    assert rep["n_joined"] == 2 and rep["n_koi_with_tess_counterpart"] == 2
    assert math.isclose(rep["fraction_koi_with_tess_counterpart"], 0.5)
    assert "2 of 4 KOIs reached a TESS counterpart" in rep["join_statement"]
    a = j[j["kepoi_name"] == "K00001.01"].iloc[0]
    assert a["join_route"] == "name_planet" and a["tic_id"] == 111 and a["period_alias"] == "1"
    assert a["ps_pl_ratror"] == 0.03 and a["ps_pl_trandep_ppm"] == 900.0 and a["ps_n_refs"] == 1
    b = j[j["kepoi_name"] == "K00002.01"].iloc[0]
    assert b["join_route"] == "position_period" and b["period_alias"] == "2"
    assert b["join_sep_arcsec"] < 2.0 and b["tic_id"] == 222
    assert "K00003.01" not in set(j["kepoi_name"])
    e, rep0 = acq.join_kepler_tess(pd.DataFrame(), toi, ps)
    assert not len(e) and rep0["joined_by_route"] == ZERO_ROUTES


# ---------------------------------------------------------------------------
# DEFECT 1 --- the tic_id join produced nothing (run 34787801172)
# ---------------------------------------------------------------------------
def test_ps_tic_id_is_a_string_and_is_parsed():
    """``ps.tic_id`` is ``"TIC 122298563"``; a bare to_numeric on it is all-NaN."""
    assert acq.parse_tic_id("TIC 122298563") == 122298563.0
    assert acq.parse_tic_id("tic122298563") == 122298563.0
    assert acq.parse_tic_id(" TIC-122298563 ") == 122298563.0
    assert acq.parse_tic_id(122298563) == 122298563.0
    for bad in (None, float("nan"), "", "  ", "TIC", "not a tic", -1):
        assert math.isnan(acq.parse_tic_id(bad)), bad
    # the exact failure mode of the 2026-09-13 run
    assert pd.to_numeric(pd.Series(["TIC 111", "TIC 222"]), errors="coerce").isna().all()


def test_name_normalisation_is_punctuation_and_case_proof():
    for v in ("Kepler-22 b", " kepler-22   B ", "Kepler 22b", "KEPLER22B"):
        assert acq._norm_name(v) == "kepler22b", v
    assert acq._norm_name(None) == acq._norm_name(float("nan")) == acq._norm_name("") == ""
    assert acq.host_of_planet_name("Kepler-22 b") == "kepler22"
    assert acq.host_of_planet_name("Kepler-22b") == "kepler22"
    assert acq.host_of_planet_name("Kepler-22") == "kepler22"     # already a host
    assert acq.host_of_planet_name("KOI-142") == "koi142"
    assert acq.host_of_planet_name(None) == ""


def test_kois_with_an_empty_kepler_name_still_join_through_their_host():
    """Only confirmed KOIs have a ``kepler_name``; the rest reach TESS by host."""
    koi = pd.DataFrame([
        # .01 is confirmed and named; .02 on the SAME star is a bare candidate
        {"kepid": 7, "kepoi_name": "K00007.01", "kepler_name": "Kepler-22 b",
         "koi_period": 10.0, "ra": 290.0, "dec": 44.0},
        {"kepid": 7, "kepoi_name": "K00007.02", "kepler_name": "", "koi_period": 4.0,
         "ra": 290.0, "dec": 44.0},
        # a star with no named KOI at all: only the position can reach its TIC
        {"kepid": 8, "kepoi_name": "K00008.01", "kepler_name": None, "koi_period": 6.0,
         "ra": 291.0, "dec": 44.0}])
    toi = pd.DataFrame([
        {"toi": 1.01, "tid": 111, "pl_orbper": 10.0, "ra": 250.0, "dec": 10.0},
        {"toi": 1.02, "tid": 111, "pl_orbper": 4.0, "ra": 250.0, "dec": 10.0},
        {"toi": 2.01, "tid": 222, "pl_orbper": 6.0, "ra": 250.0, "dec": 10.0}])
    ps = pd.DataFrame([
        # NOTE: no pl_name row matches "Kepler-22 c"; the host row is all there is
        {"pl_name": "Kepler-22 b", "hostname": "Kepler-22", "tic_id": "TIC 111",
         "default_flag": 1, "pl_trandep_ppm": 900.0, "disc_facility": "Kepler",
         "ra": 290.0, "dec": 44.0},
        {"pl_name": "Kepler-99 b", "hostname": "Kepler-99", "tic_id": "TIC 222",
         "default_flag": 1, "pl_trandep_ppm": 800.0, "disc_facility": "Kepler",
         "ra": 291.0, "dec": 44.0}])
    j, rep = acq.join_kepler_tess(koi, toi, ps, pos_radius_arcsec=2.0)
    assert rep["joined_by_route"] == {"name_planet": 1, "name_host": 1, "position_tic": 1,
                                      "position_period": 0}
    assert rep["koi_with_tic_id"] == 3
    assert rep["n_ps_tic_id_parsed"] == 2 and rep["n_ps_tic_id_unparsed"] == 0
    routes = dict(zip(j["kepoi_name"], j["join_route"], strict=True))
    assert routes == {"K00007.01": "name_planet", "K00007.02": "name_host",
                      "K00008.01": "position_tic"}
    assert set(j["tic_id"]) == {111, 222}
    # every TOI is used at most once, and the period still picks the planet
    assert dict(zip(j["kepoi_name"], j["toi"], strict=True))["K00007.02"] == 1.02


def test_ps_rows_the_where_did_not_exclude_are_counted_and_dropped():
    ps = pd.DataFrame([
        {"pl_name": "Kepler-1 b", "hostname": "Kepler-1", "tic_id": "TIC 1", "default_flag": 1,
         "disc_facility": "Kepler", "pl_trandep_ppm": 900.0, "ra": 1.0, "dec": 1.0},
        {"pl_name": "TOI-9 b", "hostname": "TOI-9", "tic_id": "TIC 9", "default_flag": 1,
         "disc_facility": "Transiting Exoplanet Survey Satellite (TESS)",
         "pl_trandep_ppm": 900.0, "ra": 2.0, "dec": 2.0}])
    rep: dict = {}
    maps = acq.prepare_ps(ps, rep)
    assert rep["n_ps_rows"] == 2 and rep["n_ps_rows_kepler"] == 1
    assert rep["n_ps_rows_not_kepler_discovered"] == 1
    assert set(maps["name_to_tic"]) == {"kepler1b"} and set(maps["host_to_tic"]) == {"kepler1"}
    assert rep["ps_tic_id_samples"][0] == "TIC 1"
    # the column absent -> the question is unanswered, not answered "none"
    rep2: dict = {}
    acq.prepare_ps(ps.drop(columns=["disc_facility"]), rep2)
    assert rep2["n_ps_rows_not_kepler_discovered"] is None


def test_non_default_ps_rows_are_kept_as_depth_references():
    ps = pd.DataFrame([
        {"pl_name": "Kepler-1 b", "hostname": "Kepler-1", "tic_id": "TIC 1", "default_flag": 0,
         "pl_ratror": 0.02, "pl_trandep_ppm": 700.0, "pl_refname": "old",
         "disc_facility": "Kepler"},
        {"pl_name": "Kepler-1 b", "hostname": "Kepler-1", "tic_id": "TIC 1", "default_flag": 1,
         "pl_ratror": 0.03, "pl_trandep_ppm": 900.0, "pl_refname": "adopted",
         "disc_facility": "Kepler"},
        {"pl_name": "Kepler-1 b", "hostname": "Kepler-1", "tic_id": "TIC 1", "default_flag": 0,
         "pl_ratror": 0.04, "pl_trandep_ppm": 1100.0, "pl_refname": "newer",
         "disc_facility": "Kepler"}])
    rep: dict = {}
    ref = acq.prepare_ps(ps, rep)["ref_by_name"]["kepler1b"]
    assert rep["n_ps_default_rows"] == 1 and rep["n_ps_distinct_planet_names"] == 1
    assert ref["ps_refname"] == "adopted" and ref["ps_pl_trandep_ppm"] == 900.0
    assert ref["ps_n_refs"] == 3
    assert (ref["ps_trandep_ppm_min"], ref["ps_trandep_ppm_max"]) == (700.0, 1100.0)


def test_a_contested_toi_goes_to_the_more_reliable_route():
    """A TOI is used once, so `name_planet` must claim it before `position_tic`."""
    koi = pd.DataFrame([
        # the unnamed KOI comes FIRST in table order and would win a table-order scan
        {"kepid": 2, "kepoi_name": "K00002.01", "kepler_name": None, "koi_period": 10.0,
         "ra": 290.0, "dec": 44.0},
        {"kepid": 1, "kepoi_name": "K00001.01", "kepler_name": "Kepler-A b",
         "koi_period": 10.0, "ra": 290.0, "dec": 44.0}])
    toi = pd.DataFrame([{"toi": 1.01, "tid": 111, "pl_orbper": 10.0, "ra": 20.0, "dec": 4.0}])
    ps = pd.DataFrame([{"pl_name": "Kepler-A b", "hostname": "Kepler-A", "tic_id": "TIC 111",
                        "default_flag": 1, "disc_facility": "Kepler", "ra": 290.0, "dec": 44.0}])
    j, rep = acq.join_kepler_tess(koi, toi, ps)
    assert list(j["kepoi_name"]) == ["K00001.01"] and j["join_route"].iloc[0] == "name_planet"
    assert rep["tic_matched_no_period_match"] == 1 and rep["n_toi_used"] == 1


def test_routes_can_be_switched_off_in_config():
    koi = pd.DataFrame([{"kepid": 1, "kepoi_name": "K00001.01", "kepler_name": "Kepler-A b",
                         "koi_period": 10.0, "ra": 290.0, "dec": 44.0}])
    toi = pd.DataFrame([{"toi": 1.01, "tid": 111, "pl_orbper": 10.0, "ra": 20.0, "dec": 4.0}])
    ps = pd.DataFrame([{"pl_name": "Kepler-A b", "hostname": "Kepler-A", "tic_id": "TIC 111",
                        "default_flag": 1, "disc_facility": "Kepler", "ra": 290.0, "dec": 44.0}])
    _, on = acq.join_kepler_tess(koi, toi, ps)
    assert on["joined_by_route"]["name_planet"] == 1
    _, off = acq.join_kepler_tess(koi, toi, ps, routes=("position_period",))
    assert off["n_joined"] == 0 and off["routes_enabled"] == ["position_period"]


def test_ps_depth_percent_is_converted_to_ppm():
    fake = lambda adql: pd.DataFrame({"pl_name": ["Kepler-1 b"], "tic_id": [1],   # noqa: E731
                                      "pl_trandep": [0.5], "default_flag": [1]})
    df, status = acq.fetch_ps_kepler(query_fn=fake)
    assert status == "OK" and df["pl_trandep_ppm"].iloc[0] == 5000.0


def test_tap_csv_parsing_separates_error_bodies():
    assert len(acq.parse_tap_csv("a,b\n1,2\n")) == 1
    assert not len(acq.parse_tap_csv(""))
    with pytest.raises(RuntimeError):
        acq.parse_tap_csv("ERROR\nBad query")
    with pytest.raises(RuntimeError):
        acq.parse_tap_csv("<html><body>503</body></html>")
    url = acq.build_url("select top 5 koi_depth from cumulative")
    assert url.startswith(acq.EXOARCHIVE_TAP + "?query=select%20top") and "format=csv" in url


def test_gaia_neighbour_chunks_fail_per_chunk_not_per_run():
    stars = pd.DataFrame({"planet_key": [f"K{i}" for i in range(5)],
                          "ra": np.linspace(290, 291, 5), "dec": [44.0] * 5})
    calls = []

    def gaia(part, r):
        calls.append(len(part))
        if part["key"].iloc[0] == 2:
            raise RuntimeError("timeout")
        return pd.DataFrame({"key": part["key"], "source_id": 1, "ra": part["ra"],
                             "dec": part["dec"], "phot_g_mean_mag": 12.0})

    log = acq.AcquisitionLog()
    neigh, status = acq.fetch_gaia_neighbours(stars, chunk=2, gaia_fn=gaia, log=log)
    assert calls == [2, 2, 1]
    assert status["neighbours_status"].tolist() == ["OK", "OK", "QUERY_FAILED", "QUERY_FAILED", "OK"]
    assert len(neigh) == 3 and "sep_arcsec" in neigh and (neigh["sep_arcsec"] < 1e-6).all()
    assert log.as_dict()["n_query_failed"] == 1 and log.as_dict()["n_ok"] == 2
    # with no cone fallback the failure is still ATTRIBUTED, not a bare count
    bad = [s for s in log.stages if s["status"] == "QUERY_FAILED"][0]
    assert "timeout" in bad["error"] and bad["route"] == "upload"
    assert bad["attempts"] and "timeout" in bad["attempts"][0]["error"]


# ---------------------------------------------------------------------------
# DEFECT 2 --- Gaia neighbours failed for every target (run 34787801172)
# ---------------------------------------------------------------------------
def _stars(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({"planet_key": [f"K{i}" for i in range(n)],
                         "ra": np.linspace(290.0, 291.0, n), "dec": [44.0] * n})


def _cone_rows(part, radius, *, attempts=None, fail_keys=()):
    """A stand-in for the pyvo cone route: one neighbour per reachable target."""
    rows, failed = [], []
    for _, t in part.iterrows():
        if int(t["key"]) in set(fail_keys):
            failed.append(int(t["key"]))
            if attempts is not None:
                attempts.append({"key": int(t["key"]), "transport": "pyvo_sync", "ok": False,
                                 "error": "DALServiceError('cone 503')"})
            continue
        rows.append({"key": int(t["key"]), "source_id": 900 + int(t["key"]), "ra": t["ra"],
                     "dec": t["dec"], "phot_g_mean_mag": 15.0})
    return pd.DataFrame(rows, columns=["key", "source_id", "ra", "dec", "phot_g_mean_mag"]), failed


def test_gaia_upload_failure_falls_back_to_cones_and_records_the_error_text():
    """The upload route is refused; the cones carry the chunk, and the text lands."""
    def upload(part, r):
        raise acq.GaiaRouteFailed("upload refused", [
            {"attempt": 1, "transport": "astroquery_async", "ok": False,
             "error": "HTTPError('Error 500: SQL exception: ERROR: canceling statement "
                      "due to statement timeout')"},
            {"attempt": 2, "transport": "astroquery_sync", "ok": False,
             "error": "HTTPError('Error 500')"}])

    log = acq.AcquisitionLog()
    neigh, status = acq.fetch_gaia_neighbours(_stars(4), chunk=4, gaia_fn=upload,
                                              cone_fn=_cone_rows, log=log)
    assert len(neigh) == 4 and set(neigh["planet_key"]) == {"K0", "K1", "K2", "K3"}
    assert (status["neighbours_status"] == "OK").all()
    assert (status["neighbours_route"] == "cones").all()
    stage = log.stages[0]
    assert stage["status"] == "OK" and stage["route"] == "cones"
    # the thing that made run 34787801172 undiagnosable: the exception TEXT
    assert any("statement timeout" in a["error"] for a in stage["attempts"])
    from seti.growth.run import _gaia_errors
    errs = _gaia_errors(log)
    assert errs and errs[0]["n_failed_attempts"] == 2
    assert "statement timeout" in errs[0]["attempts"][0]["error"]


def test_targets_no_gaia_route_reaches_stay_not_checked():
    """Upload refused AND the cone refused for one star: that star is not isolated."""
    def upload(part, r):
        raise acq.GaiaRouteFailed("upload refused", [{"attempt": 1, "ok": False,
                                                      "error": "HTTPError('Error 500')"}])

    def cones(part, radius, *, attempts=None):
        return _cone_rows(part, radius, attempts=attempts, fail_keys=(1,))

    log = acq.AcquisitionLog()
    neigh, status = acq.fetch_gaia_neighbours(_stars(3), chunk=3, gaia_fn=upload,
                                              cone_fn=cones, log=log)
    assert status["neighbours_status"].tolist() == ["OK", "QUERY_FAILED", "OK"]
    assert status["neighbours_route"].tolist() == ["cones", "", "cones"]
    assert set(neigh["planet_key"]) == {"K0", "K2"}
    stage = log.stages[0]
    assert stage["status"] == "QUERY_FAILED" and stage["n_targets_failed"] == 1
    assert any("cone 503" in a["error"] for a in stage["attempts"])


def test_a_finished_gaia_chunk_is_checkpointed_and_not_requeried(tmp_path):
    calls = []

    def upload(part, r):
        calls.append(len(part))
        return pd.DataFrame({"key": part["key"], "source_id": 1, "ra": part["ra"],
                             "dec": part["dec"], "phot_g_mean_mag": 12.0})

    kw = {"chunk": 4, "gaia_fn": upload, "checkpoint_dir": tmp_path / "ck"}
    n1, s1 = acq.fetch_gaia_neighbours(_stars(4), log=acq.AcquisitionLog(), **kw)
    n2, s2 = acq.fetch_gaia_neighbours(_stars(4), log=(log2 := acq.AcquisitionLog()), **kw)
    assert calls == [4]                                   # the second run re-queried nothing
    assert len(n2) == len(n1) == 4 and (s2["neighbours_status"] == "OK").all()
    assert s2["neighbours_route"].tolist() == ["checkpoint"] * 4
    assert log2.stages[0]["route"] == "checkpoint"


def test_gaia_cone_adql_is_a_small_single_target_cone():
    q = acq.gaia_cone_adql(290.0, 44.0, 21.0)
    assert "gaiadr3.gaia_source" in q and "tap_upload" not in q
    assert "CONTAINS(POINT('ICRS', ra, dec)" in q and "CIRCLE('ICRS', 290.0000000, 44.0000000," in q
    assert f"{21.0 / 3600.0:.8f}" in q
    assert acq.GAIA_TAP == "https://gea.esac.esa.int/tap-server/tap"


def test_cone_route_retries_each_target_and_reports_the_ones_it_lost():
    seen = []

    def transport(adql):
        seen.append(adql)
        if "291.0000000" in adql:
            raise RuntimeError("DALServiceError('503')")
        return pd.DataFrame({"source_id": [7], "ra": [290.0], "dec": [44.0],
                             "phot_g_mean_mag": [14.0]})

    att: list = []
    targets = pd.DataFrame({"key": [0, 1], "ra": [290.0, 291.0], "dec": [44.0, 44.0]})
    neigh, failed = acq.gaia_neighbours_cones(targets, 21.0, retries=2, transport=transport,
                                              base_sleep=0.0, attempts=att)
    assert len(seen) == 3                                  # 1 for key 0, 2 tries for key 1
    assert list(neigh["key"]) == [0] and failed == [1]
    assert att and "503" in att[0]["error"] and att[0]["key"] == 1


# ---------------------------------------------------------------------------
# degraded archive and the end-to-end run
# ---------------------------------------------------------------------------
class _FakeTAP:
    """Scripted Exoplanet Archive: ``mode`` = 'fail' | 'zero' | 'ok'."""

    def __init__(self, mode: str, tables: tuple | None = None):
        self.mode, self.calls = mode, []
        self.koi, self.toi, self.ps = tables if tables else (None, None, None)

    def __call__(self, adql: str) -> pd.DataFrame:
        self.calls.append(adql)
        if self.mode == "fail":
            raise RuntimeError("CONNECT tunnel failed, response 403")
        if self.mode == "zero":
            return pd.DataFrame()
        q = adql.lower()
        table = self.koi if "from cumulative" in q else self.toi if "from toi" in q else self.ps
        if self.ps is not None and table is self.ps:
            table = table.drop(columns=["pl_trandep_ppm"])
        return table.head(5) if "top 5" in q else table.copy()


def _gaia_ok(part, r):
    return pd.DataFrame({"key": part["key"], "source_id": part["key"] + 1, "ra": part["ra"],
                         "dec": part["dec"], "phot_g_mean_mag": 13.0,
                         "sep_arcsec": 0.05})


def test_unreachable_archive_is_no_data_reached(tmp_path):
    out = tmp_path / "growth"
    rep = growth_run("all", out_dir=out, query_fn=_FakeTAP("fail"), gaia_fn=_gaia_ok)
    assert rep["verdict"] == VERDICT_NO_DATA and rep["reason"] == "archive_query_failed"
    s = json.loads((out / "summary.json").read_text())
    assert s["n_measured"] == 0 and s["classes"]["GROWTH_CANDIDATE"] == 0
    assert "NOT a null result" in s["note"]
    assert not len(pd.read_csv(out / "candidates.csv"))
    probe = json.loads((out / "probe.json").read_text())
    assert probe["n_reachable"] == 0
    assert all(v["status"] == "QUERY_FAILED" for v in probe["tables"].values())


def test_empty_archive_is_no_data_reached_with_its_own_reason(tmp_path):
    out = tmp_path / "growth"
    rep = growth_run("all", out_dir=out, query_fn=_FakeTAP("zero"), gaia_fn=_gaia_ok)
    assert rep["verdict"] == VERDICT_NO_DATA and rep["reason"] == "archive_returned_zero_rows"
    assert rep["join"]["n_joined"] == 0 if "n_joined" in rep["join"] else True


def test_end_to_end_synthetic_run_reaches_candidates(tmp_path):
    out = tmp_path / "growth"
    tables = _population(20, growth={"ratio": 1.5})
    fake = _FakeTAP("ok", tables)
    rep = growth_run("all", out_dir=out, query_fn=fake, gaia_fn=_gaia_ok)
    assert rep["verdict"] == VERDICT_CANDIDATES and rep["n_candidates"] == 1
    assert rep["join"]["joined_by_route"] == {**ZERO_ROUTES, "name_planet": 20}
    assert "20 of 20 KOIs reached a TESS counterpart" in rep["join_statement"]
    assert rep["gaia"]["n_targets_ok"] == 20 and rep["gaia"]["n_targets_failed"] == 0
    assert rep["gaia"]["errors"] == [] and rep["gaia"]["by_route"] == {"upload": 20}
    assert "20 of 20 joined stars have Gaia neighbours" in rep["gaia_statement"]
    assert rep["classes"]["GROWTH_CANDIDATE"] == 1 and rep["classes"]["CONSISTENT"] == 19
    assert rep["degraded"] == []
    assert rep["n_long_period"] == 0
    c = pd.read_csv(out / "candidates.csv")
    assert list(c["kepoi_name"]) == ["K00100.01"] and c["class"].iloc[0] == "GROWTH_CANDIDATE"
    j = pd.read_csv(out / "joined.csv")
    assert len(j) == 20 and set(j["neighbours_status"]) == {"OK"}
    assert (out / "acquisition_log.json").exists() and (out / "screened.csv").exists()
    # `main` re-assesses from disk, with no archive
    assert main(["--stage", "assess", "--out-dir", str(out)]) == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["verdict"] == VERDICT_CANDIDATES and s["rejection_counters"]["n_candidates"] == 1


def test_end_to_end_ordinary_population_is_no_candidate_not_a_null(tmp_path):
    out = tmp_path / "growth"
    tables = _population(12, deficit=0.12)
    rep = growth_run("all", out_dir=out, query_fn=_FakeTAP("ok", tables), gaia_fn=_gaia_ok)
    assert rep["verdict"] == VERDICT_NONE and rep["n_candidates"] == 0
    assert "count, not an occurrence limit" in rep["note"]
    assert math.isclose(rep["population"]["population_offset"], math.log(0.88), abs_tol=0.02)


def test_failed_gaia_cones_degrade_and_veto_rather_than_assume_isolation(tmp_path):
    out = tmp_path / "growth"
    tables = _population(6, growth={"ratio": 1.5})

    def gaia_fail(part, r):
        raise RuntimeError("Gaia 500")

    rep = growth_run("all", out_dir=out, query_fn=_FakeTAP("ok", tables), gaia_fn=gaia_fail)
    assert rep["verdict"] == VERDICT_NONE
    assert rep["degraded"] == ["gaia_neighbours_not_checked:6"]
    assert rep["rejection_counters"]["first_veto"]["neighbours_not_checked"] == 6
    assert rep["rejection_counters"]["n_would_be_candidate_without_vetoes"] == 1
    # honest degradation, AND the reason is on the record
    assert rep["gaia"]["errors"][0]["attempts"][0]["error"].count("Gaia 500")
    assert "may never be called isolated" in rep["gaia_statement"] or \
           "not_checked" in rep["gaia_statement"]
    acqrep = json.loads((out / "acquire.json").read_text())
    assert "Gaia 500" in json.dumps(acqrep["gaia"]["errors"])


def test_a_large_joined_sample_keeps_the_offset_and_the_thresholds_honest(tmp_path):
    """With the join fixed the sample is ~5x larger; re-check offset and classes.

    400 ordinary planets at the Han et al. 2025 deficit, with an 8 % depth
    scatter (comparable to the sigma_sys floor, so the tail is real), plus one
    injected 1.5x growth.  The population median must still land on ln(0.88),
    the injected planet must still be the only candidate, and the 3-sigma
    classes must populate the tail without the 5-sigma gate manufacturing
    candidates out of it.
    """
    out = tmp_path / "growth"
    tables = _population(400, growth={"ratio": 1.5}, deficit=0.12, seed=11, scatter=0.08)
    rep = growth_run("all", out_dir=out, query_fn=_FakeTAP("ok", tables), gaia_fn=_gaia_ok)
    assert rep["join"]["joined_by_route"] == {**ZERO_ROUTES, "name_planet": 400}
    assert rep["n_measured"] == 400
    # the median is robust to the tail: still the injected offset
    assert math.isclose(rep["population"]["population_offset"], math.log(0.88), abs_tol=0.02)
    assert rep["verdict"] == VERDICT_CANDIDATES and rep["n_candidates"] == 1
    assert rep["classes"]["GROWTH_CANDIDATE"] == 1
    pop, classes = rep["population"], rep["classes"]
    # a REAL tail exists at 3 sigma (so the thresholds are being exercised) ...
    assert classes["DEEPER_TESS"] + classes["SHALLOWER_TESS"] >= 2
    assert pop["n_z_above_3"] + pop["n_z_below_minus_3"] >= 2
    # ... and only the injected planet reaches the 5-sigma gate
    assert pop["n_z_above_5"] == 1
    assert classes["CONSISTENT"] + classes["DEEPER_TESS"] + classes["SHALLOWER_TESS"] == 399
    c = pd.read_csv(out / "candidates.csv")
    assert list(c["kepoi_name"]) == ["K00100.01"]


def test_run_wires_the_cone_fallback_when_the_upload_route_is_refused(tmp_path, monkeypatch):
    """End to end through ``growth_run``: upload refused -> cones -> OK, error kept."""
    seen: dict = {}

    def upload(part, r):
        raise acq.GaiaRouteFailed("upload refused", [
            {"attempt": 1, "transport": "astroquery_async", "ok": False,
             "error": "HTTPError('Error 500: canceling statement due to statement timeout')"}])

    def cones(part, radius, *, retries=2, tap_url=None, attempts=None, **kw):
        seen.update(retries=retries, tap_url=tap_url)
        return _cone_rows(part, radius, attempts=attempts)

    monkeypatch.setattr(acq, "gaia_neighbours_upload", upload)
    monkeypatch.setattr(acq, "gaia_neighbours_cones", cones)
    out = tmp_path / "growth"
    tables = _population(6, growth={"ratio": 1.5})
    rep = growth_run("all", out_dir=out, query_fn=_FakeTAP("ok", tables))   # gaia_fn=None
    assert seen["tap_url"] == acq.GAIA_TAP and seen["retries"] == CONF["gaia"]["cone_retries"]
    assert rep["gaia"]["n_targets_ok"] == 6 and rep["gaia"]["n_targets_failed"] == 0
    assert rep["gaia"]["by_route"] == {"cones": 6} and rep["degraded"] == []
    assert "statement timeout" in json.dumps(rep["gaia"]["errors"])
    assert rep["verdict"] == VERDICT_CANDIDATES         # the dilution term was reached
    assert (out / "data" / "gaia_checkpoint").exists()


def test_long_period_list_covers_joined_and_tess_only():
    koi, toi, ps = _population(3)
    koi.loc[0, "koi_period"] = 45.0
    toi.loc[0, "pl_orbper"] = 45.0
    extra = pd.DataFrame([{"toi": 9999.01, "tid": 5, "pl_orbper": 100.0}])
    toi2 = pd.concat([toi, extra], ignore_index=True)
    j = _joined(koi, toi, ps)
    lp = long_period_list(j, toi2, 30.0)
    assert sorted(lp["source"]) == ["joined", "tess_only"]
    assert lp[lp["source"] == "tess_only"]["key"].iloc[0] == "TOI-9999.01"
    assert not len(long_period_list(pd.DataFrame(), pd.DataFrame(), 30.0))


def test_planet_drift_handles_missing_ror_and_neighbours():
    row = {"koi_depth": 1000.0, "koi_depth_err1": 30.0, "koi_depth_err2": -30.0,
           "pl_trandep": 1000.0, "pl_trandeperr1": 40.0, "pl_trandeperr2": -40.0,
           "koi_impact": 0.3, "koi_steff": 5800.0, "koi_slogg": 4.5, "koi_kepmag": 13.0}
    d = planet_drift(row, None, DriftParams.from_config(CONF))
    assert math.isclose(d["k_kepler"], math.sqrt(1e-3))
    assert d["n_gaia_neighbours"] == 0 and d["contam_applied"] == 0.0
    assert d["duration_verdict"] == "inconclusive"
    assert math.isclose(d["ln_ratio_corr"], -math.log(d["ld_band_ratio"]))
    # a neighbour table with no self-match falls back to the Kepler magnitude
    neigh = pd.DataFrame({"phot_g_mean_mag": [14.0], "sep_arcsec": [5.0]})
    d2 = planet_drift(row, neigh, DriftParams.from_config(CONF))
    assert d2["target_g_source"] == "kepmag" and d2["n_gaia_neighbours"] == 1
    assert math.isclose(d2["contam_max"], 10 ** -0.4, rel_tol=1e-9)


def test_stage_assess_without_acquire_report_says_so(tmp_path):
    out = tmp_path / "growth"
    out.mkdir()
    s = stage_assess(CONF, out)
    assert s["verdict"] == VERDICT_NO_DATA and s["reason"] == "no_acquire_report"


def test_config_workflow_and_doc_exist():
    conf = load_growth_config()
    for k in ("archive", "join", "gaia", "aperture", "drift", "long_period", "limb_darkening",
              "vet"):
        assert k in conf, k
    assert conf["drift"]["n_candidate"] == 5.0 and conf["drift"]["sigma_sys"] == 0.05
    assert conf["gaia"]["neighbour_radius_arcsec"] == 21.0
    # every new assertion about a column or a service is in the config, with a
    # `verify` note beside it in the file
    assert conf["join"]["routes"] == list(acq.JOIN_ROUTES)
    assert conf["join"]["ps_tic_id_is_string"] is True
    assert conf["join"]["ps_tic_id_prefix"] == "TIC"
    assert conf["gaia"]["tap_url"] == acq.GAIA_TAP
    assert conf["gaia"]["fallback_cones"] is True and conf["gaia"]["cone_retries"] >= 1
    assert conf["archive"]["ps_require_kepler_in_disc_facility"] is True
    yml = Path("config/growth.yaml").read_text()
    for marker in ("ps_tic_id_is_string", "tap_url"):
        block = yml[max(0, yml.index(marker) - 1400):yml.index(marker)]
        assert "`verify`" in block, marker
    assert "Kepler-9" in conf["vet"]["ttv_systems"]
    grid = conf["limb_darkening"]["grids"][0]
    assert len(grid["teff"]) == len(grid["kepler"]) == len(grid["tess"])
    assert Path("config/growth.yaml").exists()
    assert Path(".github/workflows/growth.yml").exists()
    assert Path("docs/growth.md").exists()
