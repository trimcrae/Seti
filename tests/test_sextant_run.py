"""Offline tests for SEXTANT's ephemeris layer, per-object fit, screen and assessment.

No network.  The propagator is scored against an analytic Kepler orbit; the
estimator against synthetic Gaia observations with a known injected ``A2``
(the generator in ``test_sextant_residuals``); the screen against records
built to trip each rule; the assessment against a table with known contents.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from test_sextant_residuals import JD0, make_observations

from seti.sextant import ephem as E
from seti.sextant import residuals as R
from seti.sextant import run as RUN
from seti.sextant.controls import score_control, summarise_controls

MU = E.GM_SUN_AU3_D2
CONF = RUN.load_config({"sextant": {}})
SYN_CONV = RUN.conventions_from({"epoch": "epoch:TDB:0.0", "apply_light_time": True,
                                 "apply_stellar_aberration": True,
                                 "apply_solar_deflection": True,
                                 "scan_pa_north_to_east": True})


def analytic(a, e, i, om, w, m0, dt):
    n = math.sqrt(MU / a ** 3)
    return E.elements_to_heliocentric_state([a], [e], [i], [om], [w],
                                            [m0 + math.degrees(n * dt)])[0]


# ---------------------------------------------------------------------------
# 1. The propagator and its interpolants
# ---------------------------------------------------------------------------
def test_elements_to_state_places_perihelion_at_a_one_minus_e():
    st = E.elements_to_heliocentric_state([2.5], [0.15], [8.0], [100.0], [40.0], [0.0])[0]
    assert abs(np.linalg.norm(st[:3]) - 2.5 * 0.85) < 1e-12
    # Specific angular momentum magnitude = sqrt(mu a (1-e^2)).
    h = np.linalg.norm(np.cross(st[:3], st[3:]))
    assert abs(h - math.sqrt(MU * 2.5 * (1 - 0.15 ** 2))) < 1e-12


def test_propagator_matches_kepler_to_centimetres_over_two_thousand_days():
    jd0 = 2457000.0
    els = (2.5, 0.15, 8.0, 100.0, 40.0, 0.0)
    st = E.elements_to_heliocentric_state(*[[x] for x in els])
    pert = E.synthetic_perturbers(jd0 - 100, jd0 + 2200)
    prop = E.NBodyPropagator(pert, relativity=False)
    ev = jd0 + np.array([-500.0, -1.03, 1.0, 10.77, 100.0, 500.31, 2000.0])
    r, v, a = prop.propagate(st, jd0, E.EvalRequest(np.zeros(ev.size, int), ev), h=0.05)
    for k, dt in enumerate(ev - jd0):
        truth = analytic(*els, dt)
        assert np.linalg.norm(r[k] - truth[:3]) * E.AU_KM * 1e3 < 0.05     # metres
        assert np.linalg.norm(v[k] - truth[3:]) < 1e-13
    # Acceleration returned at the epoch is the two-body one.
    an = -MU * r / np.linalg.norm(r, axis=1)[:, None] ** 3
    assert np.allclose(a, an, rtol=1e-6)


def test_propagator_batches_objects_with_different_epochs_per_object():
    jd0 = 2457000.0
    els = [(2.2, 0.1, 5.0, 20.0, 30.0, 45.0), (3.0, 0.25, 12.0, 200.0, 80.0, 300.0)]
    st = np.array([E.elements_to_heliocentric_state(*[[x] for x in e])[0] for e in els])
    pert = E.synthetic_perturbers(jd0 - 100, jd0 + 1200)
    prop = E.NBodyPropagator(pert, relativity=False)
    ev = jd0 + np.array([3.3, 700.1, 999.0, 12.0])
    obj = np.array([0, 1, 0, 1])
    r, _, _ = prop.propagate(st, jd0, E.EvalRequest(obj, ev), h=0.05)
    for k in range(4):
        truth = analytic(*els[obj[k]], ev[k] - jd0)
        assert np.linalg.norm(r[k] - truth[:3]) * E.AU_KM * 1e3 < 0.05


def test_relativity_term_is_present_and_of_the_expected_size():
    jd0 = 2457000.0
    st = E.elements_to_heliocentric_state([2.5], [0.15], [8.0], [100.0], [40.0], [0.0])
    pert = E.synthetic_perturbers(jd0 - 10, jd0 + 400)
    ev = jd0 + np.array([365.0])
    r0, _, _ = E.NBodyPropagator(pert, relativity=False).propagate(
        st, jd0, E.EvalRequest(np.zeros(1, int), ev), h=0.05)
    r1, _, _ = E.NBodyPropagator(pert, relativity=True).propagate(
        st, jd0, E.EvalRequest(np.zeros(1, int), ev), h=0.05)
    d_km = np.linalg.norm(r1 - r0) * E.AU_KM
    # Tens of km over a year at 2.5 au: several mas, so it must be in the model.
    assert 1.0 < d_km < 200.0


def test_step_buckets_shrink_for_eccentric_near_sun_orbits():
    h = E.step_for([2.5, 1.2, 0.9], [0.15, 0.6, 0.85])
    assert h[0] == 0.05
    assert h[1] < h[0]
    assert h[2] <= h[1]
    assert all(x in E.STEP_BUCKETS_DAYS for x in h)


def test_quintic_hermite_reproduces_a_kepler_step_to_micrometres():
    st0 = E.elements_to_heliocentric_state([2.5], [0.15], [8.0], [100.0], [40.0], [0.0])[0]
    h = 0.05

    def acc(x):
        return -MU * x[:3] / np.linalg.norm(x[:3]) ** 3

    st1 = R.propagate_two_body(st0, np.array([h]), mu=MU)[0]
    s = np.array([0.3, 0.7])
    rr, vv = E.hermite_quintic(h, s, np.tile(st0[:3], (2, 1)), np.tile(st0[3:], (2, 1)),
                               np.tile(acc(st0), (2, 1)), np.tile(st1[:3], (2, 1)),
                               np.tile(st1[3:], (2, 1)), np.tile(acc(st1), (2, 1)))
    truth = R.propagate_two_body(st0, s * h, mu=MU)
    assert np.all(np.linalg.norm(rr - truth[:, :3], axis=1) * E.AU_KM * 1e3 < 1e-3)
    assert np.all(np.linalg.norm(vv - truth[:, 3:], axis=1) < 1e-14)


def test_cubic_hermite_on_a_daily_grid_is_metre_accurate_for_a_planet_like_orbit():
    st0 = E.elements_to_heliocentric_state([1.0], [0.017], [0.0], [0.0], [100.0], [0.0])[0]
    tg = 2457000.0 + np.arange(0.0, 400.0, 1.0)
    hist = R.propagate_two_body(st0, tg - tg[0], mu=MU)
    t = np.array([tg[0] + 10.3, tg[0] + 200.77])
    p, v = E.hermite_cubic(tg, hist[:, :3], hist[:, 3:], t)
    truth = R.propagate_two_body(st0, t - tg[0], mu=MU)
    assert np.all(np.linalg.norm(p - truth[:, :3], axis=1) * E.AU_KM * 1e3 < 10.0)


# ---------------------------------------------------------------------------
# 2. Horizons parsing and the aligned ephemeris
# ---------------------------------------------------------------------------
HORIZONS_TEXT = """*******************************************************************************
Ephemeris / API_USER Tue Sep 22 00:00:00 2026 Pasadena, USA      / Horizons
*******************************************************************************
$$SOE
2457000.500000000, A.D. 2014-Dec-09 00:00:00.0000, -1.614753320000E+00,  1.179710940000E+00,  7.186646160000E-01, -8.041357910000E-03, -9.415739430000E-03, -2.611730930000E-03,
2457001.500000000, A.D. 2014-Dec-10 00:00:00.0000, -1.622790000000E+00,  1.170290000000E+00,  7.160510000000E-01, -8.030000000000E-03, -9.420000000000E-03, -2.615000000000E-03,
$$EOE
*******************************************************************************
"""


def test_horizons_vector_table_is_parsed_and_errors_are_named():
    jd, st = E.parse_horizons_vectors(HORIZONS_TEXT)
    assert jd.tolist() == [2457000.5, 2457001.5]
    assert st.shape == (2, 6)
    assert abs(st[0, 0] + 1.61475332) < 1e-12
    with pytest.raises(E.EphemerisError, match="No matches"):
        E.parse_horizons_vectors("No matches found.\n")


def test_aligned_ephemeris_expands_to_sub_metre_and_refuses_wrong_lengths():
    st0 = E.elements_to_heliocentric_state([2.5], [0.15], [8.0], [100.0], [40.0], [0.0])[0]
    jd = 2457000.0 + np.array([0.0, 100.0, 500.0])
    hist = R.propagate_two_body(st0, jd - jd[0], mu=MU)
    acc = -MU * hist[:, :3] / np.linalg.norm(hist[:, :3], axis=1)[:, None] ** 3
    eph = E.AlignedEphemeris(jd, hist[:, :3], hist[:, 3:], acc)
    dt = 0.012                                              # a 17-minute light time
    got = eph(jd - dt)
    truth = R.propagate_two_body(st0, jd - dt - jd[0], mu=MU)
    assert np.all(np.linalg.norm(got[:, :3] - truth[:, :3], axis=1) * E.AU_KM * 1e3 < 0.5)
    single = eph(np.array([jd[1] + 0.001]))
    assert single.shape == (1, 6)
    with pytest.raises(ValueError):
        eph(jd[:2])
    far = eph(jd + np.array([0.0, 5.0, 0.0]))
    assert np.isnan(far[1]).all() and np.isfinite(far[0]).all()


def test_transit_states_expand_to_every_ccd_epoch():
    st0 = E.elements_to_heliocentric_state([2.5], [0.15], [8.0], [100.0], [40.0], [0.0])[0]
    base = 2457000.0
    tid = np.repeat([7.0, 3.0, 11.0], 9)
    jd = base + np.repeat([0.0, 40.0, 200.0], 9) + np.tile(np.arange(9) * 4.85 / 86400.0, 3)
    uniq, first = E.transit_reference_epochs(jd, tid)
    assert uniq.tolist() == [3.0, 7.0, 11.0]
    st_tr = R.propagate_two_body(st0, first - base, mu=MU)
    eph = E.expand_transit_states(jd, tid, first, st_tr)
    truth = R.propagate_two_body(st0, jd - base, mu=MU)
    got = eph(jd)
    assert np.all(np.linalg.norm(got[:, :3] - truth[:, :3], axis=1) * E.AU_KM * 1e3 < 1e-3)


def test_sbdb_row_parsing_flags_nongrav_and_numbers():
    row = E.parse_sbdb_row({"pdes": "101955", "H": "20.6", "a": "1.126", "e": "0.2037",
                            "i": "6.03", "om": "2.06", "w": "66.2", "ma": "100.0",
                            "epoch": "2461000.5", "A2": "-4.6e-14", "A2_sigma": "1e-15"})
    assert row["number_mp"] == 101955 and row["nongrav_fitted"] is True
    assert abs(row["a2"] + 4.6e-14) < 1e-20
    plain = E.parse_sbdb_row({"pdes": "433", "H": "10.4", "a": "1.458", "e": "0.22",
                              "i": "10.8", "om": "304.3", "w": "178.9", "ma": "1.0",
                              "epoch": "2461000.5", "A2": None})
    assert plain["nongrav_fitted"] is False and math.isnan(plain["a2"])
    assert E.parse_sbdb_row({"pdes": "2020 SO"})["number_mp"] is None


# ---------------------------------------------------------------------------
# 3. The end-to-end fit on synthetic sky
# ---------------------------------------------------------------------------
def _synthetic_bundle(cols, s0, number=9001):
    """The pinned gravity-only route on the synthetic sky (Sun at the barycentre)."""
    pert = E.synthetic_perturbers(JD0 - 200, JD0 + 2300)
    bundles, skipped = RUN.integrator_bundles({number: cols}, {}, pert, SYN_CONV,
                                              relativity=False,
                                              start_states={number: (JD0, s0)})
    assert not skipped
    b = bundles[number]
    b.route = "horizons_pinned_gravity_only"
    return b, pert


@pytest.mark.parametrize("a2_true", [3e-13, -1.5e-13])
def test_injected_a2_is_recovered_through_the_stage_runner(a2_true):
    cols, _, s0 = make_observations(a2_true, n_transits=45, seed=11)
    b, pert = _synthetic_bundle(cols, s0)
    row = {"h": 17.0, "nongrav_fitted": True, "a2": a2_true, "a2_sigma": 0.1 * abs(a2_true),
           "a": 1.6, "e": 0.2, "i": 5.0, "node": 40.0, "argperi": 30.0}
    rec, series = RUN.fit_object(9001, cols, b, row, pert, SYN_CONV, CONF)
    assert rec["verdict"] == "FITTED", rec
    assert abs(rec["a2"] - a2_true) < 3.0 * rec["a2_err"] + 0.1 * abs(a2_true)
    assert rec["a2_snr"] > 5
    assert rec["model_verdict"] == "FORCE_LAW_PREFERRED"
    assert math.isfinite(rec["a1"]) and math.isfinite(rec["a3"])
    RUN.annotate_orbit(rec, row, CONF)
    assert rec["is_control"] is True
    sc = score_control(rec["a2"], rec["a2_err"], row["a2"], row["a2_sigma"])
    assert sc["verdict"] == "RECOVERED"


def test_pure_noise_is_ordinary_and_never_a_control_failure():
    cols, _, s0 = make_observations(0.0, n_transits=45, seed=3)
    b, pert = _synthetic_bundle(cols, s0)
    row = {"h": 17.0, "nongrav_fitted": False, "a2": float("nan"), "a2_sigma": float("nan"),
           "a": 1.6, "e": 0.2, "i": 5.0, "node": 40.0, "argperi": 30.0}
    rec, _ = RUN.fit_object(9002, cols, b, row, pert, SYN_CONV, CONF)
    assert rec["verdict"] == "FITTED"
    assert rec["a2_snr"] < 3.0
    RUN.annotate_orbit(rec, row, CONF)
    RUN.screen_record(rec, CONF)
    assert rec["tier"] == "ordinary"
    assert rec["is_control"] is False


def test_the_screen_promotes_a_ceiling_exceedance_only_with_a_force_law():
    # A 1e-13 transverse acceleration on a 5 km body (H ~ 14) is far above the
    # hard ceiling: a genuine exceedance that the force-law test must confirm.
    cols, _, s0 = make_observations(1e-13, n_transits=50, seed=5)
    b, pert = _synthetic_bundle(cols, s0)
    row = {"h": 14.0, "nongrav_fitted": False, "a2": float("nan"), "a2_sigma": float("nan"),
           "a": 1.6, "e": 0.2, "i": 5.0, "node": 40.0, "argperi": 30.0}
    rec, _ = RUN.fit_object(9003, cols, b, row, pert, SYN_CONV, CONF)
    RUN.annotate_orbit(rec, row, CONF)
    assert rec["ratio_hard"] > 1.0
    RUN.screen_record(rec, CONF)
    assert rec["tier"] in ("interest", "candidate"), (rec["tier"], rec["vetoes"])
    # The same numbers with a geometric explanation preferred are held at watch.
    rec2 = dict(rec)
    rec2["model_verdict"] = "GEOMETRIC_EXPLANATION_PREFERRED"
    rec2["best_geometric_model"] = "geometry:illumination"
    RUN.screen_record(rec2, CONF)
    assert rec2["tier"] == "watch" and any("geometric" in v for v in rec2["vetoes"])
    # A catalogued binary is vetoed however good the fit.
    from seti.sextant.screen import BinaryCatalogue

    rec3 = dict(rec)
    RUN.screen_record(rec3, CONF, BinaryCatalogue(rows=[{"number_mp": 9003}]))
    assert rec3["tier"] == "watch" and rec3["known_binary"] is True
    # An absorbed fraction at the ceiling is untestable, not ordinary.
    rec4 = dict(rec)
    rec4["a2_absorbed_fraction"] = 0.999
    RUN.screen_record(rec4, CONF)
    assert rec4["tier"] == "untestable"
    # Below the realistic envelope it is ordinary whatever the S/N.
    rec5 = dict(rec)
    rec5["ratio_realistic"] = 0.5
    RUN.screen_record(rec5, CONF)
    assert rec5["tier"] == "ordinary"


def test_a_short_arc_is_refused_not_fitted():
    cols, _, s0 = make_observations(1e-13, n_transits=12, seed=9, span=200.0)
    b, pert = _synthetic_bundle(cols, s0)
    rec, _ = RUN.fit_object(9004, cols, b, None, pert, SYN_CONV, CONF)
    assert rec["verdict"] == "FIT_REFUSED"
    assert "arc" in rec["reason"]
    RUN.annotate_orbit(rec, None, CONF)
    RUN.screen_record(rec, CONF)
    assert rec["tier"] == "untestable"


def test_a_horizons_style_source_with_a_fitted_a2_is_refused_but_gravity_only_is_not():
    with_ng = {"nongrav_fitted": True}
    with pytest.raises(R.CircularOrbitSourceError):
        R.require_independent_prediction(RUN.orbit_source_for("horizons", with_ng),
                                         allow_partial=True)
    assert R.require_independent_prediction(
        RUN.orbit_source_for("integrator", with_ng), allow_partial=True) == R.PARTIAL_SELF_FIT
    assert R.require_independent_prediction(
        RUN.orbit_source_for("horizons_pinned_gravity_only", with_ng),
        allow_partial=True) == R.PARTIAL_SELF_FIT


# ---------------------------------------------------------------------------
# 4. Selection, sharding, serialisation
# ---------------------------------------------------------------------------
def test_shards_partition_the_chosen_set_exactly_once_and_controls_come_first():
    numbers = list(range(1, 1001))
    sbdb = {n: {"nongrav_fitted": n % 97 == 0} for n in numbers}
    chosen = RUN.choose_objects(numbers, sbdb, max_objects=100, seed=1)
    assert len(chosen) == 100
    assert all(n in chosen for n in numbers if n % 97 == 0)
    parts = [RUN.shard_slice(chosen, k, 7) for k in range(7)]
    flat = sorted(x for p in parts for x in p)
    assert flat == sorted(chosen)
    assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 1
    assert RUN.choose_objects(numbers, sbdb, 0, 1) == numbers
    assert sum(len(c) for c in RUN.chunked(chosen, 30)) == 100


def test_shard_csv_round_trips_through_gzip(tmp_path):
    recs = [{"number_mp": 5, "denomination": "Astraea, x", "verdict": "FITTED", "tier": "ordinary",
             "a2": 1.5e-14, "a2_err": 2e-15, "reasons": ["a", "b"], "vetoes": [],
             "is_control": True, "h": 6.9},
            {"number_mp": 6, "verdict": "NO_OBSERVATIONS_RETURNED", "tier": "untestable"}]
    p = tmp_path / "fits" / "shard_0_of_1.csv.gz"
    RUN.write_shard_csv(p, recs)
    df = pd.read_csv(p)
    assert len(df) == 2 and set(df.columns) == set(RUN.CSV_COLUMNS)
    assert df.loc[0, "denomination"] == "Astraea, x"
    assert abs(df.loc[0, "a2"] - 1.5e-14) < 1e-25
    assert df.loc[0, "reasons"] == "a|b" and df.loc[0, "is_control"] == 1


# ---------------------------------------------------------------------------
# 5. The assessment
# ---------------------------------------------------------------------------
def _table(n=300, seed=2, n_controls=6, n_exceed=0, exceed_vetoed=True):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        h = rng.uniform(10, 18)
        ceiling = float(RUN.NG.momentum_ceiling_a2(h))
        err = 2e-15 * rng.uniform(0.5, 2)
        a2 = rng.normal(0, err)
        rec = {"number_mp": k + 1, "denomination": f"obj{k}", "route": "integrator",
               "verdict": "FITTED", "tier": "ordinary", "n_transits": int(rng.integers(10, 60)),
               "arc_days": rng.uniform(400, 1900), "median_sigma_al_mas": rng.uniform(0.3, 3),
               "a2": a2, "a2_err": err, "a2_err_pessimistic": err * 1.2, "a2_snr": abs(a2) / err,
               "a2_absorbed_fraction": 0.97, "excess_scatter": 1.0, "h": h,
               "a": rng.uniform(2.1, 3.2), "e": rng.uniform(0.02, 0.3), "i": rng.uniform(0, 20),
               "node": rng.uniform(0, 360), "mjd_min": 56900.0, "mjd_max": 58800.0,
               "ceiling_hard": ceiling, "ratio_hard": abs(a2) / ceiling,
               "ratio_realistic": abs(a2) / (0.3 * ceiling),
               "a2_expected_yarkovsky": 0.074 * ceiling * 0.5, "epsilon_eff": abs(a2) / ceiling,
               "model_verdict": "NO_MODEL_PREFERRED", "is_control": False,
               "jpl_a2": float("nan"), "jpl_a2_sigma": float("nan"),
               "published_control": None, "vetoes": "", "reasons": ""}
        if k < n_controls:
            true = -5e-14
            rec.update({"is_control": True, "jpl_a2": true, "jpl_a2_sigma": 3e-15,
                        "a2": true * 1.05, "a2_err": 5e-15, "a2_snr": 10.5})
        elif k < n_controls + n_exceed:
            big = 20 * ceiling
            rec.update({"a2": big, "a2_err": big / 12, "a2_snr": 12.0, "ratio_hard": 20.0,
                        "ratio_realistic": 66.0, "tier": "watch" if exceed_vetoed else "interest",
                        "model_verdict": ("GEOMETRIC_EXPLANATION_PREFERRED" if exceed_vetoed
                                          else "FORCE_LAW_PREFERRED"),
                        "vetoes": "geometric_explanation_preferred" if exceed_vetoed else ""})
        rows.append(rec)
    return pd.DataFrame(rows)


def test_assessment_reports_controls_and_a_gaussian_noise_distribution():
    out = RUN.assess_frame(_table(), CONF)
    assert out["verdict"] == "NO_CEILING_EXCEEDANCE"
    assert out["controls"]["verdict"] == "CONTROLS_RECOVERED"
    assert out["controls"]["n_controls"] == 6
    d = out["a2_distribution"]
    assert 0.7 < d["z_mad_sigma"] < 1.4
    assert out["population"]["verdict"] == "INSUFFICIENT_POPULATION"


def test_assessment_names_vetoed_exceedances_and_interest_exceedances_differently():
    vetoed = RUN.assess_frame(_table(n_exceed=4, exceed_vetoed=True), CONF)
    assert vetoed["verdict"].startswith("ALL_4_EXCEEDANCES_VETOED")
    assert vetoed["n_exceedances"] == 4 and vetoed["n_interest"] == 0
    live = RUN.assess_frame(_table(n_exceed=4, exceed_vetoed=False), CONF)
    assert live["verdict"].startswith("CEILING_EXCEEDANCES_4")
    assert live["n_interest"] == 4


def test_assessment_with_a_sign_flipped_estimator_flags_the_run():
    df = _table()
    df.loc[df["is_control"], "a2"] = -df.loc[df["is_control"], "a2"]
    out = RUN.assess_frame(df, CONF)
    assert out["controls"]["verdict"] == "CONTROLS_FAILED_SIGN"
    assert out["verdict"].startswith("ESTIMATOR_FAILS_CONTROLS")


def test_assessment_degrades_honestly_on_an_empty_table():
    out = RUN.assess_frame(pd.DataFrame(), CONF)
    assert out["verdict"] == "NO_DATA_REACHED"
    assert summarise_controls([])["verdict"] == "NO_CONTROLS_PRESENT"


def test_control_scoring_distinguishes_unmeasured_from_wrong():
    assert score_control(1e-14, 1e-14, 5e-14, 2e-15)["verdict"] == "CONSISTENT_BUT_NOT_DETECTED"
    assert score_control(5e-14, 1e-15, -5e-14, 2e-15)["verdict"] == "SIGN_WRONG"
    assert score_control(4e-14, 2e-15, 5e-14, 2e-15)["verdict"] == "RECOVERED"
    assert score_control(2e-13, 1e-15, 5e-14, 1e-15)["verdict"] == "MAGNITUDE_OFF"
    assert score_control(float("nan"), 1, 5e-14, 1e-15)["verdict"] == "NOT_MEASURED"
