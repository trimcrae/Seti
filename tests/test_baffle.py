"""Offline test suite for BAFFLE --- reciprocal mid-IR absorbing screens.

No network anywhere (``conftest.py`` raises on any socket).  Per
``docs/channel-brief.md`` §5 the suite covers:

* a synthetic 20,000-star population with a realistic Ks−W1(J−Ks) locus: the
  fit recovers the injected relation to < 0.02 mag;
* an injected 0.5-mag TWO-band deficit is recovered as a candidate; an injected
  W1-only deficit is vetoed ``w1_only_methane_like``;
* every named veto has a case that trips it (saturated, LPV deferral, WISE
  artifact, extended, profile fit, WISE / Gaia variability, crowding, blend
  flux theft, multi-peak); report-only flags are reported, not vetoed;
* a clean null population yields 0 candidates, and the −tail is empty;
* the query builders emit the expected table names, the random_index locus
  clause and the LEFT JOIN … IS NULL of the missing track;
* the acquisition ledger keeps QUERY_FAILED / ZERO / TRUNCATED apart; a
  truncated band splits on random_index; a failed leaf never kills the band;
* an empty results directory screens to ``NO_DATA_REACHED``;
* the config loads, the CLI accepts ``baffle --stage screen``, the patch /
  radio guards degrade with a clear status.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seti.baffle import acquire as A
from seti.baffle import locus as L
from seti.baffle import run as R
from seti.baffle import screen as S

ACQ = dict(R.DEFAULTS["acquire"])
LOCUS_CFG = dict(R.DEFAULTS["locus"])
SCREEN_CFG = {k: (dict(v) if isinstance(v, dict) else v) for k, v in R.DEFAULTS["screen"].items()}


# ---------------------------------------------------------------------------
# synthetic population
# ---------------------------------------------------------------------------
def true_ksw1(jk):
    jk = np.asarray(jk, dtype=float)
    return 0.03 + 0.12 * (jk - 0.5) + 0.20 * (jk - 0.5) ** 2


def true_ksw2(jk):
    return true_ksw1(jk) - 0.02


def make_population(n: int = 20000, seed: int = 11, sigma: float = 0.03) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    jk = rng.uniform(0.12, 0.95, n)
    ks = rng.uniform(6.5, 12.0, n)
    plx = rng.uniform(2.0, 30.0, n)
    mg = 4.5 + 6.0 * jk                                   # dwarfs
    g = mg - 5.0 * np.log10(plx / 100.0)
    df = pd.DataFrame({
        "source_id": np.arange(1, n + 1, dtype=np.int64),
        "ra": rng.uniform(0, 360, n), "dec": rng.uniform(-30, 30, n),
        "l": rng.uniform(0, 360, n), "b": rng.uniform(-80, 80, n),
        "ecl_lat": rng.uniform(-60, 60, n),
        "parallax": plx, "parallax_error": 0.02, "parallax_over_error": plx / 0.02,
        "pmra": rng.normal(0, 15, n), "pmdec": rng.normal(0, 15, n), "ruwe": 1.0,
        "phot_g_mean_mag": g, "phot_bp_mean_mag": g + 0.3, "phot_rp_mean_mag": g - 0.4,
        "bp_rp": 0.6 + 2.0 * jk, "phot_variable_flag": "NOT_AVAILABLE",
        "non_single_star": 0, "ipd_frac_multi_peak": 0, "phot_bp_rp_excess_factor": 1.2,
        "random_index": np.arange(n), "is_locus_sample": True,
        "wise_angular_distance": 0.3, "wise_number_of_neighbours": 1, "wise_number_of_mates": 0,
        "tmass_angular_distance": 0.2, "tmass_number_of_neighbours": 1,
        "tmass_number_of_mates": 0,
        "j_m": ks + jk, "j_msigcom": 0.02, "h_m": ks + 0.7 * jk, "h_msigcom": 0.02,
        "ks_m": ks, "ks_msigcom": 0.02, "tmass_ph_qual": "AAA",
        "w1mpro": ks - true_ksw1(jk) + rng.normal(0, sigma, n), "w1mpro_error": 0.025,
        "w1snr": 60.0, "w1rchi2": 1.0,
        "w2mpro": ks - true_ksw2(jk) + rng.normal(0, sigma, n), "w2mpro_error": 0.025,
        "w2snr": 60.0, "w2rchi2": 1.0,
        "w3mpro": ks - true_ksw1(jk) - 0.03 + rng.normal(0, 0.08, n), "w3mpro_error": 0.06,
        "w3snr": 20.0, "w4mpro": np.nan, "w4mpro_error": np.nan, "w4snr": 1.0,
        "cc_flags": "0000", "ext_flag": 0, "var_flag": "0000", "ph_qual": "AAAB",
        "w1mjd_mean": 55300.0, "w2mjd_mean": 55300.0,
        "wise_designation": "J000000.00+000000.0", "tmass_designation": "00000000+0000000",
        "track": "deficit", "dec_band_index": 18, "dec_lo": 0.0, "dec_hi": 5.0,
    })
    return df


@pytest.fixture(scope="module")
def population() -> pd.DataFrame:
    return make_population()


@pytest.fixture(scope="module")
def locus(population) -> L.Locus:
    return L.fit_locus(population, LOCUS_CFG)


def _pick_clean_star(df: pd.DataFrame) -> int:
    """A locus-grade star near ks=10, jk=0.5: safely unsaturated, mid-locus."""
    d = (df["ks_m"] - 10.0).abs() + 4.0 * (df["j_m"] - df["ks_m"] - 0.5).abs()
    return int(d.idxmin())


def _inject(df: pd.DataFrame, idx: int, dw1: float, dw2: float, **overrides) -> pd.DataFrame:
    out = df.copy()
    out.loc[idx, "w1mpro"] += dw1
    out.loc[idx, "w2mpro"] += dw2
    for k, v in overrides.items():
        out.loc[idx, k] = v
    return out


def _screen(df: pd.DataFrame, locus: L.Locus, neighbours=None) -> dict:
    r = L.residuals(df, locus, LOCUS_CFG)
    return S.screen_deficit(r, SCREEN_CFG, neighbours)


# ---------------------------------------------------------------------------
# locus
# ---------------------------------------------------------------------------
def test_locus_recovers_the_injected_relation(population, locus):
    assert locus.has("dwarf", "w1") and locus.has("dwarf", "w2") and locus.has("all", "w1")
    for band, truth in (("w1", true_ksw1), ("w2", true_ksw2)):
        b = locus.bins["dwarf"][band]
        centers = np.asarray(b["centers"])
        med = np.asarray(b["median"])
        assert len(centers) >= 12
        assert np.max(np.abs(med - truth(centers))) < 0.02
        # robust scatter ~ the injected 0.03 photometric noise
        assert 0.02 < np.median(b["scatter"]) < 0.045
    assert 13000 < locus.meta["n_locus"] < 16000    # ks < 8 stars are saturated, excluded
    assert locus.meta["rchi2_thresholds"]["w1rchi2_max"] >= 1.0


def test_locus_predict_interpolates_and_clamps(locus):
    m, s = locus.predict([0.5, 0.52], "dwarf", "w1")
    assert np.all(np.isfinite(m)) and np.all(s > 0)
    lo, _ = locus.predict(-5.0, "dwarf", "w1")
    hi, _ = locus.predict(9.0, "dwarf", "w1")
    b = locus.bins["dwarf"]["w1"]
    assert lo[0] == pytest.approx(b["median"][0])
    assert hi[0] == pytest.approx(b["median"][-1])
    # an unfitted class falls back to the pooled locus
    m_g, _ = locus.predict(0.5, "giant", "w1")
    m_all, _ = locus.predict(0.5, "all", "w1")
    assert m_g[0] == pytest.approx(m_all[0])


def test_locus_round_trips_through_json(locus, tmp_path):
    p = tmp_path / "locus.json"
    locus.save(p)
    back = L.Locus.load(p)
    assert back.bins["dwarf"]["w1"]["median"] == locus.bins["dwarf"]["w1"]["median"]
    m1, _ = locus.predict(0.4, "dwarf", "w2")
    m2, _ = back.predict(0.4, "dwarf", "w2")
    assert m1[0] == m2[0]


def test_luminosity_class_is_a_straight_line_in_the_cmd():
    df = pd.DataFrame({"phot_g_mean_mag": [10.0, 10.0, 10.0, 12.0],
                       "parallax": [1.0, 100.0, 100.0, np.nan],
                       "bp_rp": [1.2, 1.2, 0.2, 1.0]})
    cls = L.luminosity_class(df, LOCUS_CFG).tolist()
    assert cls == ["giant", "dwarf", "blue", "dwarf"]
    mg = L.absolute_g(np.array([10.0, 10.0]), np.array([100.0, -1.0]))
    assert mg[0] == pytest.approx(10.0) and np.isnan(mg[1])


def test_residual_sign_convention_negative_is_deficit(population, locus):
    idx = _pick_clean_star(population)
    df = _inject(population, idx, 0.5, 0.5)
    r = L.residuals(df, locus, LOCUS_CFG)
    assert r.loc[idx, "resid_w1"] < -0.4 and r.loc[idx, "resid_w2"] < -0.4
    assert r.loc[idx, "sig_w1"] < -5 and r.loc[idx, "sig_w2"] < -5
    # W3 is only evaluated above the SNR floor
    df.loc[idx, "w3snr"] = 1.0
    r2 = L.residuals(df, locus, LOCUS_CFG)
    assert np.isnan(r2.loc[idx, "resid_w3"])


def test_tail_asymmetry_control_is_empty_on_a_clean_population(population, locus):
    r = L.residuals(population, locus, LOCUS_CFG)
    ok, _ = L.locus_quality_mask(population, LOCUS_CFG)
    t = L.tail_asymmetry(r[ok], LOCUS_CFG)
    assert t["w1"]["n"] > 13000
    assert t["w1"]["n_deficit_lt_-5sig"] == 0 and t["w2"]["n_deficit_lt_-5sig"] == 0
    assert t["w1"]["n_deficit_lt_-3sig"] < 0.005 * t["w1"]["n"]
    # sig is CONSERVATIVE by construction: the locus scatter already carries the
    # typical noise and the star's own errors are added again, so a clean
    # population sits below unit width (here ~0.7), never above it.
    assert 0.55 < t["w1"]["robust_sigma"] < 1.05


# ---------------------------------------------------------------------------
# screen: recovery and the named vetoes
# ---------------------------------------------------------------------------
def test_injected_two_band_deficit_is_recovered_as_a_candidate(population, locus):
    idx = _pick_clean_star(population)
    df = _inject(population, idx, 0.5, 0.5)
    res = _screen(df, locus)
    cand = res["candidates"]
    assert len(cand) == 1
    assert int(cand.iloc[0]["source_id"]) == int(df.loc[idx, "source_id"])
    assert cand.iloc[0]["first_veto"] == "" and cand.iloc[0]["vetoes"] == ""
    assert res["funnel"]["n_candidates"] == 1
    assert res["funnel"]["n_above_threshold_two_band"] == 1
    assert cand.iloc[0]["w3_status"] in ("normal", "excess", "deficit")
    assert {"etz", "nearby", "distance_pc"} <= set(cand.columns)
    assert cand.iloc[0]["distance_pc"] == pytest.approx(1000.0 / df.loc[idx, "parallax"])
    # blend_flux_theft could not be checked without a neighbour table
    assert res["neighbours_not_checked"] == 1


def test_injected_w1_only_deficit_is_vetoed_methane_like(population, locus):
    idx = _pick_clean_star(population)
    df = _inject(population, idx, 0.5, 0.0)
    res = _screen(df, locus)
    assert len(res["candidates"]) == 0
    vet = res["vetoed"]
    assert len(vet) == 1 and vet.iloc[0]["first_veto"] == "w1_only_methane_like"
    assert res["counters"]["w1_only_methane_like"] == 1
    assert res["counters_first_veto"]["w1_only_methane_like"] == 1


def test_w2_only_deficit_is_a_single_band_artefact(population, locus):
    idx = _pick_clean_star(population)
    res = _screen(_inject(population, idx, 0.0, 0.5), locus)
    assert len(res["candidates"]) == 0
    assert res["vetoed"].iloc[0]["first_veto"] == "w2_only_single_band"


def test_saturated_bright_star_with_a_deficit_is_vetoed(population, locus):
    idx = _pick_clean_star(population)
    df = population.copy()
    shift = 5.0 - df.loc[idx, "ks_m"]        # ks -> 5.0, W1 ~ 5 < 8
    for c in ("j_m", "h_m", "ks_m", "w1mpro", "w2mpro", "w3mpro"):
        df.loc[idx, c] += shift
    df = _inject(df, idx, 0.5, 0.5)
    res = _screen(df, locus)
    assert len(res["candidates"]) == 0
    assert res["vetoed"].iloc[0]["first_veto"] == "saturated"
    assert res["counters"]["saturated"] == 1


def test_lpv_colour_star_is_deferred_not_discarded(population, locus):
    idx = _pick_clean_star(population)
    df = _inject(population, idx, 0.5, 0.5, bp_rp=3.4)
    res = _screen(df, locus)
    assert len(res["candidates"]) == 0
    assert len(res["deferred_lpv"]) == 1
    assert res["deferred_lpv"].iloc[0]["first_veto"] == "lpv_colour"
    assert res["funnel"]["n_deferred_lpv"] == 1
    # j - ks > 1.1 trips the same rule
    df2 = _inject(population, idx, 0.5, 0.5)
    df2.loc[idx, "j_m"] = df2.loc[idx, "ks_m"] + 1.2
    assert _screen(df2, locus)["counters"]["lpv_colour"] == 1


def test_wise_artifact_flag_is_vetoed(population, locus):
    idx = _pick_clean_star(population)
    res = _screen(_inject(population, idx, 0.5, 0.5, cc_flags="D000"), locus)
    assert len(res["candidates"]) == 0
    assert res["vetoed"].iloc[0]["first_veto"] == "wise_artifact"


@pytest.mark.parametrize("overrides,veto", [
    ({"ph_qual": "CAAA"}, "poor_wise_phot_qual"),
    ({"ph_qual": "ACAA"}, "poor_wise_phot_qual"),
    ({"tmass_ph_qual": "AAB"}, "poor_tmass_phot_qual"),
    ({"ext_flag": 2}, "extended"),
    ({"w1rchi2": 5.0}, "bad_profile_fit"),
    ({"w2rchi2": 3.5}, "bad_profile_fit"),
    ({"var_flag": "7000"}, "wise_variable"),
    ({"var_flag": "0900"}, "wise_variable"),
    ({"phot_variable_flag": "VARIABLE"}, "gaia_variable"),
    ({"wise_number_of_mates": 1}, "crowded_match"),
    ({"tmass_number_of_neighbours": 2}, "crowded_match"),
    ({"wise_angular_distance": 2.0}, "crowded_match"),
    ({"ipd_frac_multi_peak": 25}, "multi_peak"),
    ({"non_single_star": 1}, "multi_peak"),
])
def test_each_named_veto_has_a_case_that_trips_it(population, locus, overrides, veto):
    idx = _pick_clean_star(population)
    res = _screen(_inject(population, idx, 0.5, 0.5, **overrides), locus)
    assert len(res["candidates"]) == 0
    assert res["vetoed"].iloc[0]["first_veto"] == veto
    assert res["counters"][veto] == 1
    assert veto in res["vetoed"].iloc[0]["vetoes"].split(";")


def test_first_veto_is_the_first_in_order_and_all_are_recorded(population, locus):
    idx = _pick_clean_star(population)
    res = _screen(_inject(population, idx, 0.5, 0.5, cc_flags="H000", ext_flag=3,
                          non_single_star=2), locus)
    row = res["vetoed"].iloc[0]
    assert row["first_veto"] == "wise_artifact"
    assert row["vetoes"].split(";") == ["wise_artifact", "extended", "multi_peak"]
    assert res["counters_first_veto"]["extended"] == 0 and res["counters"]["extended"] == 1


def test_blend_flux_theft_needs_a_neighbour_table(population, locus):
    idx = _pick_clean_star(population)
    df = _inject(population, idx, 0.5, 0.5)
    sid = int(df.loc[idx, "source_id"])
    g = float(df.loc[idx, "phot_g_mean_mag"])
    ra, dec = float(df.loc[idx, "ra"]), float(df.loc[idx, "dec"])
    # a brighter source 4" away: vetoed
    nb = pd.DataFrame({"target_source_id": [sid, sid],
                       "source_id": [sid, 999_999_999],
                       "ra": [ra, ra + 4.0 / 3600.0 / np.cos(np.radians(dec))],
                       "dec": [dec, dec], "phot_g_mean_mag": [g, g - 1.5]})
    res = _screen(df, locus, nb)
    assert len(res["candidates"]) == 0
    assert res["vetoed"].iloc[0]["first_veto"] == "blend_flux_theft"
    assert res["neighbours_not_checked"] == 0
    # a fainter one, or a brighter one outside 8": survives
    nb2 = nb.copy()
    nb2.loc[1, "phot_g_mean_mag"] = g + 2.0
    assert len(_screen(df, locus, nb2)["candidates"]) == 1
    nb3 = pd.DataFrame({"target_source_id": [sid], "source_id": [7], "sep_arcsec": [12.0],
                        "phot_g_mean_mag": [g - 3.0]})
    assert len(_screen(df, locus, nb3)["candidates"]) == 1


def test_report_only_flags_do_not_veto(population, locus):
    idx = _pick_clean_star(population)
    df = _inject(population, idx, 0.5, 0.5, ruwe=2.5, pmra=400.0, pmdec=100.0)
    res = _screen(df, locus)
    assert len(res["candidates"]) == 1
    row = res["candidates"].iloc[0]
    assert bool(row["bad_astrometry"]) and bool(row["high_pm_epoch_risk"])
    assert row["pm_total_mas_yr"] == pytest.approx(np.hypot(400.0, 100.0))
    assert res["report_flags"] == {"bad_astrometry": 1, "high_pm_epoch_risk": 1}


def test_etz_and_nearby_flags_and_denominators(population, locus):
    idx = _pick_clean_star(population)
    df = _inject(population, idx, 0.5, 0.5, ecl_lat=0.1, parallax=25.0)
    df.loc[idx, "parallax_over_error"] = 1000.0
    res = _screen(df, locus)
    row = res["candidates"].iloc[0]
    assert bool(row["etz"]) and bool(row["nearby"])
    assert res["funnel"]["n_candidates_etz"] == 1 and res["funnel"]["n_candidates_nearby"] == 1
    den = res["denominators"]
    assert den["n_screened"] == len(df)
    assert den["n_screened_etz"] == int((df["ecl_lat"].abs() < 0.264).sum())
    assert den["n_screened_nearby"] == int((df["parallax"] > 20).sum())


def test_clean_null_population_yields_no_candidates(population, locus):
    res = _screen(population, locus)
    assert res["funnel"]["n_candidates"] == 0
    assert res["funnel"]["n_above_threshold_two_band"] == 0
    assert len(res["candidates"]) == 0 and len(res["vetoed"]) == 0


def test_sensitivity_recovers_deep_deficits(population, locus):
    sens = S.sensitivity(population, locus, LOCUS_CFG, SCREEN_CFG, (0.2, 0.5, 1.0),
                         max_stars=3000, seed=3)
    rec = sens["recovered"]
    assert sens["n_injected_per_mag"] == 3000
    assert rec["1"]["fraction_survive"] > 0.9
    assert rec["0.5"]["fraction_survive"] > 0.9
    # 0.2 mag is below resid_min by construction: the honesty check says so
    assert rec["0.2"]["fraction_two_band"] == 0.0


def test_verdict_tokens():
    assert S.deficit_verdict("NO_DATA_REACHED", 0, 0) == "NO_DATA_REACHED"
    assert S.deficit_verdict("COMPLETE", 100, 0) == "NO_MIDIR_DEFICIT_SURVIVOR"
    assert S.deficit_verdict("COMPLETE", 100, 3) == "MIDIR_DEFICIT_CANDIDATES_PENDING_VET (n=3)"
    v = S.deficit_verdict("PARTIAL_SAMPLE", 100, 0)
    assert v.startswith("DEGRADED_SOURCE (") and "NO_MIDIR_DEFICIT_SURVIVOR" in v
    assert S.missing_verdict("COMPLETE", 50, 0) == "NO_MISSING_COUNTERPART_SURVIVOR"
    assert "MISSING_COUNTERPART_CANDIDATES_PENDING_VET (n=2)" == S.missing_verdict("COMPLETE", 50, 2)
    assert S.combine_verdicts("NO_DATA_REACHED", "NO_DATA_REACHED") == "NO_DATA_REACHED"
    assert S.combine_verdicts("NO_MIDIR_DEFICIT_SURVIVOR", "NO_DATA_REACHED") == \
        "NO_MIDIR_DEFICIT_SURVIVOR | NO_DATA_REACHED"


# ---------------------------------------------------------------------------
# missing track
# ---------------------------------------------------------------------------
def make_missing(n: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "source_id": np.arange(10_000, 10_000 + n, dtype=np.int64),
        "ra": rng.uniform(0, 360, n), "dec": rng.uniform(-30, 30, n),
        "l": rng.uniform(0, 360, n), "b": rng.uniform(-60, 60, n),
        "ecl_lat": rng.uniform(-30, 30, n), "parallax": rng.uniform(1, 40, n),
        "parallax_over_error": 50.0, "pmra": 0.0, "pmdec": 0.0, "ruwe": 1.0,
        "phot_g_mean_mag": rng.uniform(6, 13, n), "bp_rp": 1.0,
        "phot_variable_flag": "NOT_AVAILABLE", "non_single_star": 0,
        "j_m": 8.0, "h_m": 7.7, "ks_m": rng.uniform(4, 12, n), "tmass_ph_qual": "AAA",
        "track": "missing", "dec_band_index": 18,
    })
    return df


def make_denominators() -> pd.DataFrame:
    rows = []
    for bb in range(9):
        for gb in range(6, 13):
            rows.append({"babs_bin": bb, "g_bin": gb, "n": 1000 + 300 * (8 - bb),
                         "dec_band_index": 18, "dec_lo": 0.0, "dec_hi": 5.0,
                         "grouped": True, "babs_bin_deg": 10.0, "g_bin_mag": 1.0})
    return pd.DataFrame(rows)


def test_missing_track_screen_and_fractions():
    df = make_missing()
    den = make_denominators()
    res = S.screen_missing(df, SCREEN_CFG, den)
    cand = res["candidates"]
    assert len(cand) > 0
    assert (cand["b"].abs() > 10).all()
    assert cand["ks_m"].between(5, 11).all()
    assert res["counters"]["low_latitude"] == int((~(df["b"].abs() > 10)).sum())
    assert res["funnel"]["n_candidates"] == len(cand)
    fr = res["fractions"]
    assert fr["denominator_available"] and len(fr["by_babs"]) == 9 and len(fr["by_g"]) == 7
    assert sum(r["n_missing"] for r in fr["by_babs"]) == len(df)
    assert all(0 <= r["fraction"] < 1 for r in fr["by_babs"])
    # the rejection rules each have a case
    df2 = df.copy()
    df2.loc[df2.index[0], "phot_variable_flag"] = "VARIABLE"
    df2.loc[df2.index[1], "non_single_star"] = 1
    df2.loc[df2.index[2], "tmass_ph_qual"] = "AAE"
    r2 = S.screen_missing(df2, SCREEN_CFG, den)
    assert r2["counters"]["gaia_variable"] == 1 and r2["counters"]["non_single_star"] == 1
    assert r2["counters"]["poor_tmass_phot_qual"] == 1
    # no denominator: the fraction is honestly unavailable
    r3 = S.screen_missing(df, SCREEN_CFG, None)
    assert r3["fractions"]["denominator_available"] is False
    # plain-COUNT fallback: total only, no |b| structure
    plain = pd.DataFrame({"babs_bin": [np.nan], "g_bin": [np.nan], "n": [50000],
                          "dec_band_index": [18]})
    r4 = S.screen_missing(df, SCREEN_CFG, plain)
    assert r4["fractions"]["by_babs"] == [] and r4["fractions"]["missing_fraction_total"] > 0


# ---------------------------------------------------------------------------
# query builders (pure strings)
# ---------------------------------------------------------------------------
def test_deficit_query_names_every_table_and_the_locus_clause():
    q = A.build_deficit_query(-90.0, -85.0, ACQ, top=123)
    for t in ("gaiadr3.gaia_source", "gaiadr3.allwise_best_neighbour",
              "gaiadr1.allwise_original_valid", "gaiadr3.tmass_psc_xsc_best_neighbour",
              "gaiadr1.tmass_original_valid"):
        assert t in q
    assert q.startswith("SELECT TOP 123")
    assert f"g.random_index < {ACQ['locus_random_index_max']}" in q
    assert "(t.ks_m - w.w1mpro) < -0.15" in q and "(t.ks_m - w.w2mpro) < -0.15" in q
    assert "g.phot_g_mean_mag < 15" in q
    assert "g.dec >= -90 AND g.dec < -85" in q
    for col in ("g.ecl_lat", "g.random_index", "w.w1rchi2 AS w1rchi2", "w.cc_flags AS cc_flags",
                "t.ks_m AS ks_m", "t.ph_qual AS tmass_ph_qual",
                "xw.number_of_mates AS wise_number_of_mates",
                "xt.number_of_neighbours AS tmass_number_of_neighbours"):
        assert col in q
    assert "cc_flg" not in q            # speculative column is never assumed
    # the closed top band and the random_index slice predicate
    q2 = A.build_deficit_query(85.0, 90.0, ACQ, ridx_lo=10, ridx_hi=20)
    assert "g.dec >= 85 AND g.dec <= 90" in q2
    assert "g.random_index >= 10" in q2 and "g.random_index < 20" in q2
    # COUNT(*) shares the WHERE text verbatim
    c = A.build_deficit_count(85.0, 90.0, ACQ, ridx_lo=10, ridx_hi=20)
    assert c.startswith("SELECT COUNT(*)")
    assert c.split("WHERE", 1)[1] == q2.split("WHERE", 1)[1]


def test_missing_query_is_a_left_join_is_null():
    q = A.build_missing_query(0.0, 5.0, ACQ, top=10)
    assert "LEFT OUTER JOIN gaiadr3.allwise_neighbourhood AS xn ON xn.source_id = g.source_id" in q
    assert "xn.source_id IS NULL" in q
    assert "g.phot_g_mean_mag < 13" in q
    assert "allwise_original_valid" not in q and "allwise_best_neighbour" not in q
    assert "gaiadr1.tmass_original_valid" in q
    c = A.build_missing_count(0.0, 5.0, ACQ)
    assert c.split("WHERE", 1)[1] == q.split("WHERE", 1)[1]
    d = A.build_missing_denominator_query(0.0, 5.0, ACQ)
    assert "GROUP BY babs_bin, g_bin" in d and "COUNT(*) AS n" in d
    assert "allwise" not in d and "g.phot_g_mean_mag < 13" in d
    plain = A.build_missing_denominator_query(0.0, 5.0, ACQ, grouped=False)
    assert plain.startswith("SELECT COUNT(*) AS n") and "GROUP BY" not in plain


def test_probed_column_names_flow_into_the_query():
    cols = A.default_columns()
    cols.wise["w1mjd_mean"] = "w1mjdmean"
    cols.tmass["tmass_cc_flg"] = "cc_flg"
    cols.tmass_id = "original_psc_source_id"
    q = A.build_deficit_query(0.0, 5.0, ACQ, cols)
    assert "w.w1mjdmean AS w1mjd_mean" in q and "t.cc_flg AS tmass_cc_flg" in q
    assert "xt.original_psc_source_id" in q
    got, missing = A.resolve_from_columns(A.WISE_WANT, ["designation", "w1mpro", "W2MPRO",
                                                         "w1mjdmean"])
    assert got["w1mjd_mean"] == "w1mjdmean" and got["w2mpro"] == "w2mpro"
    assert "w3mpro" in missing
    # a probed-away column becomes NaN rather than a failure
    df = A.annotate_chunk(pd.DataFrame({"source_id": [1], "random_index": [5]}),
                          "deficit", 3, -75.0, -70.0, ACQ)
    assert bool(df.loc[0, "is_locus_sample"]) and np.isnan(df.loc[0, "w3mpro"])


def test_neighbours_query_and_band_plan():
    q = A.build_neighbours_query(10.5, -20.25, 3.0)
    assert "CIRCLE('ICRS', 10.5000000, -20.2500000, 0.0500000)" in q
    assert "gaiadr3.gaia_source" in q and "allwise" not in q
    bands = A.dec_bands(5.0)
    assert len(bands) == 36 and bands[0] == (-90.0, -85.0) and bands[-1] == (85.0, 90.0)
    shards = [A.bands_for_shard(bands, s, 6) for s in range(6)]
    assert sorted(sum(shards, [])) == list(range(36))
    assert all(len(s) == 6 for s in shards)


# ---------------------------------------------------------------------------
# acquisition: ledger, splitting, containment (fake runner, no network)
# ---------------------------------------------------------------------------
class FakeArchive:
    """Deterministic stand-in for run_gaia_query with a per-band behaviour table."""

    def __init__(self, n_per_band: int = 40, behaviour: dict | None = None, seed: int = 2):
        self.n = n_per_band
        self.behaviour = behaviour or {}
        self.calls: list[str] = []
        self.rng = np.random.default_rng(seed)

    def _rows(self, n: int, lo: int | None, hi: int | None) -> pd.DataFrame:
        ri = np.arange(0, A.GAIA_RANDOM_INDEX_MAX, A.GAIA_RANDOM_INDEX_MAX // n)[:n]
        if lo is not None:
            ri = ri[ri >= lo]
        if hi is not None:
            ri = ri[ri < hi]
        return pd.DataFrame({"source_id": 1000 + ri // 1000, "random_index": ri,
                             "ks_m": 10.0, "w1mpro": 9.9, "w2mpro": 9.9})

    def __call__(self, query, *, label="", expect_rows=None, timeout_s=None):
        self.calls.append(label)
        rec = {"label": label, "status": A.QUERY_OK, "n_rows": 0, "expected_rows": expect_rows,
               "transport": "fake", "attempts": [], "query": query, "error": None,
               "seconds": 0.1}
        band = next((b for b in self.behaviour if f"band{b:02d}" in label), None)
        mode = self.behaviour.get(band, "ok")
        if "COUNT(*)" in query and "GROUP BY" not in query:
            if mode == "count_fails":
                rec.update(status=A.QUERY_FAILED, error="HTTP 500")
                return pd.DataFrame(), rec
            n = 0 if mode == "zero" else self.n
            rec["n_rows"] = 1
            return pd.DataFrame({"n": [n]}), rec
        if "GROUP BY" in query:
            rec["n_rows"] = 2
            return pd.DataFrame({"babs_bin": [0, 3], "g_bin": [10, 11], "n": [500, 700]}), rec
        lo = hi = None
        for line in query.splitlines():
            if "g.random_index >=" in line:
                lo = int(line.split(">=")[1])
            if "g.random_index <" in line and "locus" not in line and "OR" not in line:
                hi = int(line.split("<")[1])
        depth_split = lo is not None or hi is not None
        if mode == "fails":
            rec.update(status=A.QUERY_FAILED, error="timeout")
            return pd.DataFrame(), rec
        if mode == "fails_at_top_only" and not depth_split:
            rec.update(status=A.QUERY_FAILED, error="Maximum execution time (60 s) reached")
            return pd.DataFrame(), rec
        if mode == "zero":
            rec["status"] = A.QUERY_ZERO
            return pd.DataFrame(), rec
        df = self._rows(self.n, lo, hi)
        if mode == "truncates_at_top" and not depth_split:
            df = df.head(self.n // 3)             # the 60 s cut
        if mode == "half_fails" and lo is not None and lo >= A.GAIA_RANDOM_INDEX_MAX // 2:
            rec.update(status=A.QUERY_FAILED, error="HTTP 408 Job timeout")
            return pd.DataFrame(), rec
        rec["n_rows"] = len(df)
        rec["status"] = A.QUERY_OK if len(df) else A.QUERY_ZERO
        return df, rec


def _ctx(tmp_path, runner, track="deficit", **kw):
    ledger = A.AcquisitionLedger()
    acqc = dict(ACQ, top=kw.pop("top", 1000), max_split_depth=kw.pop("max_depth", 4))
    return A.FetchContext(track=track, acq=acqc, cols=A.default_columns(),
                          chunk_dir=tmp_path / "chunks", ledger=ledger, runner=runner,
                          top=acqc["top"], max_depth=acqc["max_split_depth"]), ledger


def test_a_clean_band_is_one_count_one_leaf_one_checkpoint(tmp_path):
    fake = FakeArchive(n_per_band=40)
    ctx, ledger = _ctx(tmp_path, fake)
    df = A.fetch_band(ctx, 18, 0.0, 5.0)
    assert len(df) == 40
    kinds = [e["kind"] for e in ledger.entries]
    assert kinds == ["count", "leaf", "band_total"]
    assert ledger.entries[-1]["status"] == A.QUERY_OK
    assert ledger.entries[-1]["expected_rows"] == 40
    files = list((tmp_path / "chunks").glob("deficit_b18_*.parquet"))
    assert len(files) == 1
    back = pd.read_parquet(files[0])
    assert bool(back["is_locus_sample"].any())
    assert set(["track", "dec_band_index", "dec_lo", "dec_hi", "is_locus_sample"]) <= set(back.columns)
    # re-run reuses the checkpoint without querying
    n_calls = len(fake.calls)
    df2 = A.fetch_band(ctx, 18, 0.0, 5.0)
    assert len(df2) == 40
    assert len(fake.calls) == n_calls + 1          # only the COUNT(*) ruler ran
    assert any(e.get("reused") for e in ledger.entries)


def test_a_truncated_band_splits_on_random_index_and_completes(tmp_path):
    fake = FakeArchive(n_per_band=60, behaviour={18: "truncates_at_top"})
    ctx, ledger = _ctx(tmp_path, fake)
    df = A.fetch_band(ctx, 18, 0.0, 5.0)
    assert len(df) == 60
    assert any(e["kind"] == "split" and e["status"] == A.QUERY_TRUNCATED for e in ledger.entries)
    leaves = [e for e in ledger.entries if e["kind"] == "leaf"]
    assert len(leaves) == 2 and all(e["status"] == A.QUERY_OK for e in leaves)
    assert all("random_index" in e["query"] for e in leaves)
    total = [e for e in ledger.entries if e["kind"] == "band_total"][0]
    assert total["status"] == A.QUERY_OK and total["n_rows"] == 60
    assert len(list((tmp_path / "chunks").glob("deficit_b18_*.parquet"))) == 2


def test_a_failed_leaf_never_destroys_the_others(tmp_path):
    class Both(FakeArchive):
        """band18: top query truncated (forces a split), upper half fails; band19 fails."""

        def __call__(self, query, **kw):
            df, rec = FakeArchive.__call__(self, query, **kw)
            sliced = any("g.random_index >=" in ln for ln in query.splitlines())
            if "COUNT(*)" not in query and not sliced and "band18" in kw.get("label", "") \
                    and rec["status"] == A.QUERY_OK:
                df = df.head(10)                 # the 60 s cut on the top-level query
                rec["n_rows"] = 10
            return df, rec

    both = Both(n_per_band=60, behaviour={18: "half_fails", 19: "fails"})
    ctx, ledger = _ctx(tmp_path, both, max_depth=1)
    df = A.fetch_band(ctx, 18, 0.0, 5.0)
    assert 0 < len(df) < 60                      # the good half survived
    leaves = [e for e in ledger.entries if e["kind"] == "leaf"]
    assert {e["status"] for e in leaves} == {A.QUERY_OK, A.QUERY_FAILED}
    total = [e for e in ledger.entries if e["kind"] == "band_total"][0]
    assert total["status"] == A.QUERY_TRUNCATED and total["n_leaves_failed"] == 1
    assert len(list((tmp_path / "chunks").glob("deficit_b18_*.parquet"))) == 1
    # band 19 fails outright at every depth: contained, ledgered, no crash
    df19 = A.fetch_band(ctx, 19, 5.0, 10.0)
    assert len(df19) == 0
    t19 = [e for e in ledger.entries if e["kind"] == "band_total" and e["band_index"] == 19][0]
    assert t19["status"] == A.QUERY_FAILED
    summ = A.summarise_ledger(ledger.entries, "deficit")
    assert summ["acquisition_verdict"] == "PARTIAL_SAMPLE"
    assert summ["n_leaves_failed"] >= 2 and summ["n_bands_failed"] == 1
    assert summ["n_rows_returned"] == len(df)


def test_ledger_keeps_failed_zero_and_truncated_apart():
    assert A.summarise_ledger([])["acquisition_verdict"] == "NO_QUERY_ATTEMPTED"
    failed = [{"kind": "leaf", "track": "deficit", "status": A.QUERY_FAILED, "n_rows": 0},
              {"kind": "band_total", "track": "deficit", "status": A.QUERY_FAILED, "n_rows": 0}]
    assert A.summarise_ledger(failed)["acquisition_verdict"] == "NO_DATA_REACHED"
    zero = [{"kind": "leaf", "track": "deficit", "status": A.QUERY_ZERO, "n_rows": 0},
            {"kind": "band_total", "track": "deficit", "status": A.QUERY_ZERO, "n_rows": 0,
             "expected_rows": 0}]
    assert A.summarise_ledger(zero)["acquisition_verdict"] == "QUERY_RETURNED_ZERO_ROWS"
    ok = [{"kind": "leaf", "track": "missing", "status": A.QUERY_OK, "n_rows": 5},
          {"kind": "band_total", "track": "missing", "status": A.QUERY_OK, "n_rows": 5,
           "expected_rows": 5}]
    s = A.summarise_ledger(ok + failed)
    assert s["acquisition_verdict"] == "PARTIAL_SAMPLE"
    assert A.summarise_ledger(ok + failed, "missing")["acquisition_verdict"] == "COMPLETE"
    assert A.summarise_ledger(ok + failed, "deficit")["acquisition_verdict"] == "NO_DATA_REACHED"
    assert s["completeness"] == 1.0 and s["n_leaves_failed"] == 1


def test_fetch_track_missing_writes_denominators_and_survives_a_count_failure(tmp_path):
    fake = FakeArchive(n_per_band=20, behaviour={1: "count_fails"})
    ledger = A.AcquisitionLedger()
    df = A.fetch_track("missing", dict(ACQ, top=1000), tmp_path / "chunks", ledger,
                       band_indices=[0, 1], runner=fake)
    assert len(df) == 20                          # the fake repeats source_ids; dedup merges them
    den = A.load_denominators(tmp_path / "chunks")
    assert set(den["dec_band_index"]) == {0, 1} and den["grouped"].all()
    kinds = {e["kind"] for e in ledger.entries}
    assert {"count", "leaf", "band_total", "denominator"} <= kinds
    b1 = [e for e in ledger.entries if e["kind"] == "band_total" and e["band_index"] == 1][0]
    assert b1["expected_rows"] is None and b1["status"] == A.QUERY_OK
    ledger.save(tmp_path / "ledger.json")
    back = A.AcquisitionLedger.load(tmp_path / "ledger.json")
    assert len(back.entries) == len(ledger.entries)
    assert json.loads((tmp_path / "ledger.json").read_text())["summary"]["n_bands"] == 2


def test_run_gaia_query_reports_a_timeout_without_raising(monkeypatch):
    import time as _t

    def slow(query):
        _t.sleep(0.5)
        return pd.DataFrame({"n": [1]})

    def boom(query):
        raise RuntimeError("Filesystem quota exceeded for user anonymous")

    A.reset_transport_state()
    monkeypatch.setattr(A, "GAIA_TRANSPORTS", (("slow", slow), ("boom", boom)))
    df, rec = A.run_gaia_query("SELECT 1", label="t", timeout_s=0.05, base_sleep=0.0)
    assert rec["status"] == A.QUERY_FAILED and df.empty
    assert any("no answer within" in a["error"] for a in rec["attempts"] if not a["ok"])
    # the quota signature put 'boom' on cooldown for the next queries
    assert A._TRANSPORT_COOLDOWN.get("boom", 0) > 0
    df2, rec2 = A.run_gaia_query("SELECT 1", label="t", timeout_s=2.0, base_sleep=0.0)
    assert rec2["status"] == A.QUERY_OK and rec2["transport"] == "slow"
    A.reset_transport_state()


# ---------------------------------------------------------------------------
# stages end to end (offline)
# ---------------------------------------------------------------------------
def _seed_results(out: Path, population: pd.DataFrame, inject_idx: int | None) -> None:
    chunks = out / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    df = population.copy()
    if inject_idx is not None:
        df = _inject(df, inject_idx, 0.5, 0.5)
    half = len(df) // 2
    df.iloc[:half].to_parquet(chunks / "deficit_b18_rall_all.parquet", index=False)
    df.iloc[half:].to_parquet(chunks / "deficit_b19_rall_all.parquet", index=False)
    make_missing().to_parquet(chunks / "missing_b18_rall_all.parquet", index=False)
    make_denominators().to_parquet(chunks / "missing_denominator_b18.parquet", index=False)
    ledger = A.AcquisitionLedger()
    for i, track in ((18, "deficit"), (19, "deficit"), (18, "missing")):
        ledger.add(chunk=f"{track} band{i:02d}", track=track, kind="leaf", status=A.QUERY_OK,
                   n_rows=10, expected_rows=10)
        ledger.add(chunk=f"{track} band{i:02d}", track=track, kind="band_total",
                   status=A.QUERY_OK, n_rows=10, expected_rows=10, band_index=i)
    ledger.save(out / "acquisition_ledger_s0of2.json")
    A.AcquisitionLedger().save(out / "acquisition_ledger_s1of2.json")


def test_screen_stage_end_to_end_recovers_the_injected_star(tmp_path, population):
    idx = _pick_clean_star(population)
    _seed_results(tmp_path, population, idx)
    rep = R.baffle_run(None, stage="screen", out_root=tmp_path, run_sensitivity=False)
    assert rep["verdict_deficit"] == "MIDIR_DEFICIT_CANDIDATES_PENDING_VET (n=1)"
    assert rep["verdict_missing"].startswith("MISSING_COUNTERPART_CANDIDATES_PENDING_VET")
    assert rep["verdict"] == f"{rep['verdict_deficit']} | {rep['verdict_missing']}"
    for name in ("sample.parquet", "missing_sample.parquet", "locus.json", "candidates.csv",
                 "vetoed.csv", "deferred_lpv.csv", "missing_candidates.csv", "screen.json",
                 "summary.json"):
        assert (tmp_path / name).exists(), name
    cand = pd.read_csv(tmp_path / "candidates.csv")
    assert len(cand) == 1 and int(cand.loc[0, "source_id"]) == int(population.loc[idx, "source_id"])
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["verdict"] == rep["verdict"]
    assert s["acquisition"]["deficit"]["acquisition_verdict"] == "COMPLETE"
    assert s["deficit"]["funnel"]["n_candidates"] == 1
    assert s["deficit"]["tail_asymmetry"]["w1"]["n_deficit_lt_-5sig"] == 1   # the injected star
    assert s["deficit"]["sensitivity"] == {"skipped": True}
    assert s["missing"]["fractions"]["denominator_available"] is True
    assert s["missing"]["n_denominator_bands"] == 1
    assert "veto_counters" in s["deficit"] and "w1_only_methane_like" in s["deficit"]["veto_counters"]
    # assess rebuilds the same verdict from disk
    s2 = R.baffle_run(None, stage="assess", out_root=tmp_path)
    assert s2["verdict"] == rep["verdict"] and "patch" not in s2


def test_screen_stage_with_sensitivity_reports_recovery(tmp_path, population):
    _seed_results(tmp_path, population.head(6000).reset_index(drop=True), None)
    conf_path = tmp_path / "baffle.yaml"
    conf_path.write_text("sensitivity:\n  inject_mags: [0.5]\n  max_stars: 500\n")
    conf = R.load_baffle_config(None, path=conf_path)
    assert conf["sensitivity"]["inject_mags"] == [0.5] and conf["acquire"]["g_max"] == 15.0
    rep = R.stage_screen(conf, tmp_path)
    sens = rep["deficit"]["sensitivity"]
    assert sens["n_injected_per_mag"] == 500
    assert sens["recovered"]["0.5"]["fraction_survive"] > 0.9
    assert rep["verdict_deficit"] == "NO_MIDIR_DEFICIT_SURVIVOR"


def test_an_empty_results_directory_degrades_to_no_data_reached(tmp_path, capsys):
    rep = R.baffle_run(None, stage="screen,assess", out_root=tmp_path)
    assert rep["verdict"] == "NO_DATA_REACHED"
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["verdict"] == "NO_DATA_REACHED"
    assert s["acquisition"]["deficit"]["acquisition_verdict"] == "NO_QUERY_ATTEMPTED"
    assert "NO_DATA_REACHED" in capsys.readouterr().out
    # a ledger that says every band failed is NO_DATA_REACHED, not a null
    led = A.AcquisitionLedger()
    led.add(chunk="deficit band00", track="deficit", kind="leaf", status=A.QUERY_FAILED, n_rows=0)
    led.add(chunk="deficit band00", track="deficit", kind="band_total", status=A.QUERY_FAILED,
            n_rows=0, band_index=0)
    led.save(tmp_path / "acquisition_ledger.json")
    rep2 = R.baffle_run(None, stage="screen", out_root=tmp_path)
    assert rep2["verdict_deficit"] == "NO_DATA_REACHED"
    assert rep2["acquisition"]["deficit"]["acquisition_verdict"] == "NO_DATA_REACHED"


def test_partial_acquisition_marks_the_verdict_degraded(tmp_path, population):
    _seed_results(tmp_path, population, None)
    led = A.AcquisitionLedger.load(tmp_path / "acquisition_ledger_s0of2.json")
    led.add(chunk="deficit band20", track="deficit", kind="leaf", status=A.QUERY_FAILED, n_rows=0)
    led.add(chunk="deficit band20", track="deficit", kind="band_total", status=A.QUERY_FAILED,
            n_rows=0, band_index=20)
    led.save(tmp_path / "acquisition_ledger_s0of2.json")
    rep = R.baffle_run(None, stage="screen", out_root=tmp_path, run_sensitivity=False)
    assert rep["verdict_deficit"].startswith("DEGRADED_SOURCE (partial deficit sample)")
    assert "NO_MIDIR_DEFICIT_SURVIVOR" in rep["verdict_deficit"]


def test_probe_and_acquire_stages_run_on_a_fake_archive(tmp_path):
    fake = FakeArchive(n_per_band=10)

    class WithProbe(FakeArchive):
        def __call__(self, query, **kw):
            if query.startswith("SELECT TOP 1 *"):
                rec = {"status": A.QUERY_OK, "n_rows": 1, "transport": "fake", "attempts": [],
                       "query": query, "error": None, "seconds": 0.0, "label": kw.get("label")}
                if "allwise_original_valid" in query:
                    cols = ["designation", "ra", "dec", "w1mpro", "w1mpro_error", "w2mpro",
                            "w2mpro_error", "w1mjdmean", "cc_flags", "ph_qual"]
                elif "tmass_original_valid" in query:
                    cols = ["designation", "j_m", "ks_m", "ks_msigcom", "ph_qual"]
                elif "tmass_psc_xsc_best_neighbour" in query:
                    cols = ["source_id", "original_psc_source_id", "angular_distance"]
                else:
                    cols = ["source_id", "ra", "dec"]
                return pd.DataFrame({c: [0] for c in cols}), rec
            return FakeArchive.__call__(self, query, **kw)

    fake = WithProbe(n_per_band=10)
    rep = R.baffle_run(None, stage="probe", out_root=tmp_path, runner=fake)
    assert rep["status"] == "PROBE_OK" and rep["n_tables_ok"] == 6
    assert rep["columns"]["wise"]["w1mjd_mean"] == "w1mjdmean"
    assert rep["columns"]["tmass_id"] == "original_psc_source_id"
    assert "w3mpro" in rep["columns"]["missing_wise"]
    assert rep["timed_queries"]["deficit"]["status"] == A.QUERY_OK
    assert rep["timed_queries"]["missing_denominator"]["grouped_ok"] is True
    assert (tmp_path / "probe.json").exists()
    rep2 = R.baffle_run(None, stage="acquire", out_root=tmp_path, runner=fake,
                        shard=1, n_shards=6, tracks="deficit,missing")
    assert [b["index"] for b in rep2["bands"]] == [1, 7, 13, 19, 25, 31]
    assert rep2["tracks"]["deficit"]["summary"]["acquisition_verdict"] == "COMPLETE"
    assert (tmp_path / "acquisition_ledger_s1of6.json").exists()
    assert (tmp_path / "columns_s1of6.json").exists()
    assert len(list((tmp_path / "chunks").glob("deficit_b*.parquet"))) == 6
    assert len(list((tmp_path / "chunks").glob("missing_denominator_b*.parquet"))) == 6
    # an explicit band list overrides the shard plan
    rep3 = R.stage_acquire(R.load_baffle_config(None), tmp_path, tracks="deficit",
                           dec_band_index="2,3", runner=fake, assume_columns=True)
    assert [b["index"] for b in rep3["bands"]] == [2, 3]
    assert R.merged_ledger(tmp_path)


def test_patch_and_radio_guards(tmp_path, population, monkeypatch):
    idx = _pick_clean_star(population)
    _seed_results(tmp_path, population, idx)
    R.baffle_run(None, stage="screen", out_root=tmp_path, run_sensitivity=False)
    # modules missing: a clear MODULE_MISSING status, never a crash
    monkeypatch.setitem(sys.modules, "seti.baffle.patch", None)
    monkeypatch.setitem(sys.modules, "seti.baffle.radio", None)
    rep = R.baffle_run(None, stage="patch,radio,assess", out_root=tmp_path)
    assert rep["patch"]["status"] == "MODULE_MISSING" and rep["radio"]["status"] == "MODULE_MISSING"
    assert "run_patch_stage" in rep["patch"]["error"]
    # modules present: their dicts are merged under summary['patch'] / ['radio']
    seen = {}

    def run_patch_stage(cands, out_dir, cfg):
        seen["n"] = len(cands)
        seen["max"] = cfg["patch"]["max_objects"]
        return {"n_objects": len(cands), "note": "fake patch"}

    def run_radio_stage(cfg, out_dir):
        seen["radio_out"] = Path(out_dir).name
        seen["radio_cfg_has_key"] = "radio" in cfg
        return {"n_lines": 3}

    monkeypatch.setitem(sys.modules, "seti.baffle.patch",
                        types.SimpleNamespace(run_patch_stage=run_patch_stage))
    monkeypatch.setitem(sys.modules, "seti.baffle.radio",
                        types.SimpleNamespace(run_radio_stage=run_radio_stage))
    rep = R.baffle_run(None, stage="patch,radio,assess", out_root=tmp_path, max_patch_objects=5)
    assert seen == {"n": 1, "max": 5, "radio_out": "radio", "radio_cfg_has_key": False}
    assert rep["patch"]["status"] == "OK" and rep["patch"]["n_objects"] == 1
    assert rep["radio"]["n_lines"] == 3 and rep["radio"]["status"] == "OK"
    assert (tmp_path / "patch.json").exists() and (tmp_path / "radio.json").exists()
    assert rep["verdict"].startswith("MIDIR_DEFICIT_CANDIDATES_PENDING_VET (n=1)")


# ---------------------------------------------------------------------------
# config, CLI, workflow
# ---------------------------------------------------------------------------
def test_config_loads_from_the_repo_and_carries_every_threshold(cfg):
    conf = R.load_baffle_config(cfg)
    assert conf["acquire"]["g_max"] == 15.0 and conf["acquire"]["g_max_missing"] == 13.0
    assert conf["acquire"]["locus_random_index_max"] == 9058548
    assert conf["acquire"]["pre_cut_w1"] == -0.15
    assert conf["screen"]["sig_min"] == 5.0 and conf["screen"]["resid_min"] == 0.30
    assert conf["screen"]["etz_ecl_lat_deg"] == 0.264
    assert conf["locus"]["jk_bin_width"] == 0.05
    assert "radio" in conf and "patch" in conf
    assert (Path(cfg.root) / "config" / "baffle.yaml").exists()


def test_cli_accepts_baffle_stage_screen_and_rejects_a_bogus_stage(monkeypatch):
    from seti import cli

    seen = {}
    monkeypatch.setattr(cli, "load_config", lambda: None)
    monkeypatch.setattr(cli, "_cmd_baffle",
                        lambda args, cfg: seen.update(stage=args.stage, n=args.n_shards,
                                                      command=args.command))
    cli.main(["baffle", "--stage", "screen", "--n-shards", "6"])
    assert seen == {"stage": "screen", "n": 6, "command": "baffle"}
    with pytest.raises(SystemExit):
        cli.main(["baffle", "--stage", "bogus"])
    bp = R.build_parser()
    a = bp.parse_args(["--stage", "screen", "--n-shards", "6", "--shard", "2"])
    assert a.stage == "screen" and a.n_shards == 6 and a.shard == 2
    a = bp.parse_args(["--stage", "screen,patch,assess", "--dec-band-index", "3,4"])
    assert a.stage == "screen,patch,assess" and a.dec_band_index == "3,4"
    with pytest.raises(SystemExit):
        bp.parse_args(["--stage", "bogus"])
    with pytest.raises(ValueError):
        R.baffle_run(None, stage="bogus", out_root=Path("."))


def test_cli_main_runs_the_assess_stage_offline(tmp_path):
    rc = R.main(["--stage", "assess", "--out-root", str(tmp_path)])
    assert rc == 0
    assert json.loads((tmp_path / "summary.json").read_text())["verdict"] == "NO_DATA_REACHED"


def test_workflow_declares_the_interface():
    import yaml

    doc = yaml.safe_load(Path(".github/workflows/baffle.yml").read_text())
    inputs = doc[True]["workflow_dispatch"]["inputs"]
    assert {"stage", "tracks", "n_shards", "g_max", "reduce_only_run_id", "run_patch",
            "max_patch_objects"} <= set(inputs)
    assert inputs["n_shards"]["default"] == "6" and inputs["max_patch_objects"]["default"] == "200"
    assert doc["permissions"] == {"contents": "write"}
    jobs = doc["jobs"]
    assert {"plan", "probe", "acquire", "screen", "reduce-only"} <= set(jobs)
    assert jobs["acquire"]["strategy"]["fail-fast"] is False
    assert jobs["acquire"]["timeout-minutes"] == 330
    text = Path(".github/workflows/baffle.yml").read_text()
    assert "scripts/commit_results.sh" in text and "git push" not in text
    assert "ref: ${{ github.ref_name }}" in text
    assert "seti.baffle.run --stage acquire" in text and "--n-shards" in text
    uploads = [s for s in jobs["acquire"]["steps"] if "upload-artifact" in str(s.get("uses"))]
    assert uploads and uploads[0]["if"] == "always()"
    assert "results/baffle/chunks/" in uploads[0]["with"]["path"]
