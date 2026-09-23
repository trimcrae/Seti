"""Offline tests for SEXTANT's ephemeris layer, per-object fit, screen and assessment.

No network.  The propagator is scored against an analytic Kepler orbit; the
estimator against synthetic Gaia observations with a known injected ``A2``
(the generator in ``test_sextant_residuals``); the screen against records
built to trip each rule; the assessment against a table with known contents.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pandas as pd
import pytest

from seti.sextant import ephem as E
from seti.sextant import residuals as R
from seti.sextant import run as RUN
from seti.sextant.controls import a2_from_dadt, score_control, summarise_controls
from test_sextant_residuals import JD0, make_observations

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
    # The grid spans every requested epoch: the perturbers refuse to extrapolate.
    pert = E.synthetic_perturbers(jd0 - 600, jd0 + 2200)
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


def _hermite_cubic_error_m(step_days: float) -> float:
    """Max cubic-Hermite position error (metres) on an Earth-like Kepler arc."""
    st0 = E.elements_to_heliocentric_state([1.0], [0.017], [0.0], [0.0], [100.0], [0.0])[0]
    tg = 2457000.0 + np.arange(0.0, 400.0, step_days)
    hist = R.propagate_two_body(st0, tg - tg[0], mu=MU)
    t = tg[0] + np.array([10.3, 200.77, 55.13, 301.41])
    p, _ = E.hermite_cubic(tg, hist[:, :3], hist[:, 3:], t)
    truth = R.propagate_two_body(st0, t - tg[0], mu=MU)
    return float(np.max(np.linalg.norm(p - truth[:, :3], axis=1)) * E.AU_KM * 1e3)


def test_cubic_hermite_error_is_fourth_order_and_negligible_for_perturbers():
    """The interpolant is correct, and its error cannot reach the signal.

    Two separate claims.  (1) The error converges as ``h**4``: that is what
    proves the implementation is a cubic Hermite and not merely a curve that
    passes near the knots, and it is checked by halving the step twice.  (2) The
    size that actually matters is not the interpolation error itself --- this
    interpolant only ever carries the *perturbers*, since the target is
    integrated and densely output by :func:`E.hermite_quintic` --- but the
    target displacement that a perturber position error induces.  A 34 m error
    in the Earth's position perturbs a main-belt body by ``3 GM_p dr / d**3``,
    which even on the deliberately pessimistic assumption that the whole
    interpolation error is a *constant* offset accumulating quadratically for
    the entire 2000-day mission window comes to 2.3 cm --- 1.6e-5 mas of sky at
    2 au, five orders of magnitude below Gaia's ~1 mas per-CCD sigma.  The
    real error is far smaller than that, because it oscillates at the grid
    period and averages to zero rather than accumulating.  A loosened threshold
    would hide a broken interpolant; the convergence test would not.
    """
    e1, e_half, e_quarter = (_hermite_cubic_error_m(h) for h in (1.0, 0.5, 0.25))
    assert 20.0 < e1 < 60.0                       # measured 34 m; pinned, not assumed
    assert 12.0 < e1 / e_half < 20.0              # fourth order: expect 16
    assert 12.0 < e_half / e_quarter < 20.0
    assert e_quarter < 1.0                        # 0.25 d IS metre-accurate

    # (2) what that error does to a main-belt target, expressed where it has to
    #     be compared: milliarcseconds of sky, against a ~1 mas per-CCD sigma.
    gm_earth = 3.986004418e5 * E.GM_KM3S2_TO_AU3D2      # AU^3/day^2
    dr_au = e1 / 1e3 / E.AU_KM                          # perturber position error
    d_au = 2.0                                          # target-planet separation
    da = 3.0 * gm_earth * dr_au / d_au ** 3             # AU/day^2
    span = 2000.0                                       # mission window, days
    displacement_au = 0.5 * da * span ** 2              # pessimistic: error as DC
    sky_mas = displacement_au / d_au * R.MAS_PER_RAD
    assert sky_mas < 1e-4, sky_mas                      # measured 1.6e-5 mas


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


def test_work_order_puts_the_controls_first_and_shuffles_the_rest():
    """A shard that runs out of clock must still have paid for its controls.

    Two properties, and both matter for a *truncated* shard.  (1) Every object
    with a JPL non-gravitational solution is worked before any object without
    one, so the falsification test is complete even if the shard stops in its
    first chunk.  (2) What follows is a seeded shuffle, not ascending number:
    ascending number is descending size, so a truncation in that order would
    return a sample of large main-belt bodies whose element and pole statistics
    are not the catalogue's.  The shuffle makes any prefix an unbiased random
    subsample.  Deterministic in the seed, so a re-run does the same work.
    """
    numbers = list(range(1, 501))
    sbdb = {n: {"nongrav_fitted": n % 61 == 0} for n in numbers}
    ctrl = [n for n in numbers if n % 61 == 0]
    order = RUN.order_objects(numbers, sbdb, seed=7)
    assert sorted(order) == numbers                       # a permutation, nothing lost
    assert order[:len(ctrl)] == ctrl                      # controls first, in order
    rest = order[len(ctrl):]
    assert rest != sorted(rest)                           # and the rest is shuffled
    assert RUN.order_objects(numbers, sbdb, seed=7) == order        # deterministic
    assert RUN.order_objects(numbers, sbdb, seed=8) != order        # and seed-dependent
    # An early prefix of the non-controls is not a low-number prefix: the median
    # of the first 50 sits near the catalogue median, not near its bottom.
    head = rest[:50]
    assert 150 < float(np.median(head)) < 350, float(np.median(head))
    assert RUN.order_objects([], sbdb, seed=7) == []


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


def test_frame_to_rows_nulls_every_flavour_of_missing():
    """A missing value must arrive as ``None``, whatever dtype carried it.

    ``isinstance(v, float)`` sufficed only while a missing string sat in an
    object column as ``float('nan')``.  pandas 3 makes ``str`` the default
    string dtype, and a nullable or arrow-backed column yields ``pd.NA`` while
    a datetime column yields ``pd.NaT`` --- neither of which is a float, and
    ``float(pd.NA)`` raises rather than returning NaN.  Such a value would
    travel downstream as an object that is neither a number nor a null, which
    is the shape of bug that surfaces as a fit failure on one object in
    a thousand rather than as an error.
    """
    df = pd.DataFrame({"a": pd.array([1, None], dtype="Int64"),
                       "s": pd.array(["x", None], dtype="string"),
                       "t": pd.to_datetime(["2020-01-01", None]),
                       "f": [1.5, float("nan")],
                       "o": ["keep", None]})
    rows = RUN.frame_to_rows(df)
    assert rows[0]["a"] == 1 and rows[0]["s"] == "x" and rows[0]["o"] == "keep"
    assert rows[0]["f"] == 1.5
    for k in ("a", "s", "t", "f", "o"):
        assert rows[1][k] is None, (k, rows[1][k])


def test_is_rejected_is_read_by_content_whatever_dtype_it_arrives_in():
    """``bool('false')`` is True, and that would drop every FPR observation.

    The probe left this column's type explicitly UNVERIFIED, so the code may
    not depend on it.  Numpy is the trap: a column of Python strings becomes a
    ``'<U5'`` array rather than an object array, so a check on ``object`` alone
    would miss it and ``astype(float)`` would raise on ``'false'``.
    """
    def flags(vals):
        rows = [{"number_mp": 7, "is_rejected": v, "ra": 1.0, "dec": 2.0} for v in vals]
        return list(RUN.group_observations(rows)[7]["is_rejected"])

    assert flags(["false", "true"]) == [False, True]       # numpy '<U5', not object
    assert flags([b"false", b"true"]) == [False, True]     # bytes
    assert flags([True, False]) == [True, False]
    assert flags([0, 1]) == [False, True]
    assert flags([0.0, float("nan")]) == [False, False]    # unparsed is NOT rejected


def _stub_shard_io(monkeypatch, numbers, controls):
    """Replace every I/O-bound dependency of :func:`RUN.stage_shard`.

    What is left under test is the orchestration only --- the work order, the
    per-chunk checkpoint, the in-job clock and the shard verdict --- which is
    exactly the part that decides whether an overrunning runner produces a
    usable partial result or nothing at all.
    """
    sbdb = {n: {"nongrav_fitted": n in controls, "h": 15.0, "a": 2.4, "e": 0.1,
                "a2": float("nan"), "a2_sigma": float("nan")} for n in numbers}
    monkeypatch.setattr(RUN, "load_perturbers", lambda *a, **k: None)
    monkeypatch.setattr(RUN, "load_binaries", lambda *a, **k: None)
    monkeypatch.setattr(RUN, "load_sbdb", lambda *a, **k: {"rows": sbdb, "meta": {}})
    monkeypatch.setattr(RUN, "load_gaia_objects",
                        lambda *a, **k: [{"number_mp": n, "denomination": f"o{n}"}
                                         for n in numbers])
    worked: list[int] = []

    def fake_fetch(gaia, chunk, release, paths, tag, log=print):
        worked.extend(chunk)
        return {n: {"number_mp": np.array([n])} for n in chunk}, {"tag": tag}

    monkeypatch.setattr(RUN, "fetch_chunk", fake_fetch)
    monkeypatch.setattr(RUN, "integrator_bundles",
                        lambda groups, *a, **k: ({n: object() for n in groups}, {}))
    monkeypatch.setattr(RUN, "fit_object",
                        lambda n, cols, b, row, *a, **k: (
                            {"number_mp": n, "route": "integrator", "verdict": "FITTED",
                             "a2": 1e-16, "a2_err": 1e-16, "a2_snr": 1.0,
                             "a2_absorbed_fraction": 0.9, "excess_scatter": 1.0,
                             "n_transits": 30, "arc_days": 900.0}, None))
    monkeypatch.setattr(R, "fit_common_time_offset", lambda *a, **k: {})
    return worked


class _FakeGaia:
    calls = 0


def test_a_shard_that_runs_out_of_clock_keeps_its_chunks_and_says_so(tmp_path, monkeypatch):
    """The in-job clock, which is what makes an overrun survivable.

    A job killed by the runner's own ``timeout-minutes`` is *cancelled*, and a
    cancelled job does not reliably run its ``if: always()`` artifact upload ---
    so every chunk the shard had already fitted would be thrown away.  Stopping
    between chunks instead turns the overrun into a partial result that reaches
    ``assess``, labelled ``OK_PARTIAL_BUDGET`` with the count of chunks NOT
    attempted, so nothing downstream can read the unmeasured objects as a null.
    """
    numbers = list(range(1, 41))
    worked = _stub_shard_io(monkeypatch, numbers, controls={37, 38})
    monkeypatch.setitem(RUN.DEFAULT_CONFIG, "objects_per_chunk", 10)
    conf = RUN.load_config({"sextant": {"objects_per_chunk": 10}})
    paths = RUN.Paths.make(tmp_path / "results", tmp_path / "work")
    clock = iter([0.0] + [60.0 * k for k in range(0, 40)])
    rec = RUN.stage_shard(conf, paths, 0, 1, gaia=_FakeGaia(), client=_FakeGaia(),
                          budget_minutes=1.5, now=lambda: next(clock), log=lambda *a: None)

    assert rec["verdict"] == "OK_PARTIAL_BUDGET"
    assert rec["budget_stop"]["after_chunks"] < rec["budget_stop"]["of_chunks"]
    assert rec["n_records"] == 10 * rec["budget_stop"]["after_chunks"]
    # The controls were paid for first, so a truncated shard still has them.
    assert worked[:2] == [37, 38]
    assert rec["n_controls_assigned"] == 2
    # And what it did fit is on disk, readable by assess.
    df = pd.read_csv(paths.results / "fits" / "shard_0_of_1.csv.gz")
    assert len(df) == rec["n_records"] > 0


def test_a_shard_with_room_on_the_clock_finishes_every_chunk(tmp_path, monkeypatch):
    numbers = list(range(1, 41))
    _stub_shard_io(monkeypatch, numbers, controls={37})
    conf = RUN.load_config({"sextant": {"objects_per_chunk": 10}})
    paths = RUN.Paths.make(tmp_path / "results", tmp_path / "work")
    rec = RUN.stage_shard(conf, paths, 0, 1, gaia=_FakeGaia(), client=_FakeGaia(),
                          budget_minutes=600.0, log=lambda *a: None)
    assert rec["verdict"] == "OK" and "budget_stop" not in rec
    assert rec["n_records"] == 40


# ---------------------------------------------------------------------------
# 4b. Run 35746692260: four hours of fitting that assess read as NO_DATA_REACHED
# ---------------------------------------------------------------------------
def _legacy_csv_value(v):
    """The shard writer as it was when run 35746692260 wrote its 4250 rows."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (list, tuple)):
        return "|".join(str(x) for x in v)
    if isinstance(v, float):
        return "" if not math.isfinite(v) else repr(v)
    s = str(v)
    return '"' + s.replace('"', '""') + '"' if ("," in s or '"' in s) else s


def _legacy_write_shard_csv(path, records):
    import gzip

    lines = [",".join(RUN.CSV_COLUMNS)]
    for r in records:
        lines.append(",".join(_legacy_csv_value(r.get(c)) for c in RUN.CSV_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")


def _real_shaped_records(n_fitted=40, n_controls=6, seed=4):
    """Records with the SHAPES the real shard carried, screened by the real screen.

    The real shard's funnel: FITTED, RESIDUALS_FAILED, NO_EPHEMERIS, FIT_REFUSED,
    TOO_FEW_TRANSITS, NO_OBSERVATIONS_RETURNED.  The failure verdicts carry an
    exception string as ``reason``, and the screen copies it into ``reasons``
    --- the list cell the old writer left unquoted.  Fitted values arrive as
    numpy scalars, as they do from the fit.
    """
    rng = np.random.default_rng(seed)
    recs = []
    for k in range(n_fitted):
        h = float(rng.uniform(12, 17))
        err = 2e-15
        a2 = np.float64(rng.normal(0, err))
        r = {"number_mp": 1000 + k, "denomination": f"obj{k}", "route": "horizons",
             "verdict": "FITTED", "n_transits": np.int64(30), "arc_days": np.float64(1500.0),
             "a2": a2, "a2_err": np.float64(err), "a2_snr": np.float64(abs(a2) / err),
             "a2_err_pessimistic": np.float64(err * 1.3),
             "a2_snr_pessimistic": np.float64(abs(a2) / err / 1.3),
             "a2_absorbed_fraction": np.float64(0.5), "excess_scatter": np.float64(1.0),
             "h": h, "a": 2.5, "e": 0.1, "i": 5.0, "node": 30.0,
             "mjd_min": 56900.0, "mjd_max": 58800.0,
             "ceiling_hard": float(RUN.NG.momentum_ceiling_a2(h)),
             "model_verdict": "NO_MODEL_PREFERRED", "is_control": False,
             "jpl_a2": float("nan"), "jpl_a2_sigma": float("nan")}
        r["ratio_hard"] = float(abs(a2) / r["ceiling_hard"])
        if k < n_controls:
            r.update({"is_control": True, "jpl_a2": -5e-14, "jpl_a2_sigma": 3e-15,
                      "a2": np.float64(-5.2e-14), "a2_err": np.float64(5e-15),
                      "a2_snr": np.float64(10.4)})
        recs.append(RUN.screen_record(r, CONF))
    failures = [
        ("RESIDUALS_FAILED", "ValueError: shapes (3,4) and (5,) not aligned: 4 (dim 1) != 5 (dim 0)"),
        ("FIT_REFUSED", 'LinAlgError: Singular matrix, rank 5 < 6, "ill-posed"'),
        ("NO_EPHEMERIS", "horizons: No ephemeris for target, 2 matches"),
        ("TOO_FEW_TRANSITS", "7_transits_below_12"),
        ("RESIDUALS_FAILED", "RuntimeError: two-line\nmessage, with a comma"),
    ]
    for j, (verdict, reason) in enumerate(failures):
        r = {"number_mp": 5000 + j, "denomination": f"f{j}", "route": "horizons",
             "verdict": verdict, "reason": reason}
        recs.append(RUN.screen_record(r, CONF))
    recs.append({"number_mp": 6000, "denomination": "ghost", "route": "horizons",
                 "verdict": "NO_OBSERVATIONS_RETURNED", "tier": "untestable",
                 "reasons": ["no_rows_from_gaia"], "vetoes": []})
    return recs


def _shard_dir(tmp_path, records, writer, *, verdict="OK_PARTIAL_BUDGET"):
    paths = RUN.Paths.make(tmp_path / "results", tmp_path / "work")
    writer(paths.results / "fits" / "shard_0_of_1.csv.gz", records)
    E.save_json(paths.results / "fits" / "shard_0_of_1.json",
                {"verdict": verdict, "n_records": len(records), "n_assigned": 156793,
                 "budget_stop": {"after_chunks": 17, "of_chunks": 628}})
    # The Greenberg catalogue cached empty: the suite must not open a socket.
    E.save_json(paths.results / "controls_greenberg2020.json", {"rows": {}})
    return paths


def test_the_old_writer_really_did_produce_an_unreadable_shard(tmp_path):
    """Reproduce the failure first: the exact pandas error run 35746692260 logged."""
    paths = _shard_dir(tmp_path, _real_shaped_records(), _legacy_write_shard_csv)
    with pytest.raises(pd.errors.ParserError, match="Expected 83 fields"):
        pd.read_csv(paths.results / "fits" / "shard_0_of_1.csv.gz")


def test_a_legacy_shard_is_repaired_and_assessed_not_reported_as_no_data(tmp_path):
    """The reduce-only path: the surviving artifact must assess WITHOUT a refit."""
    import json

    recs = _real_shaped_records()
    paths = _shard_dir(tmp_path, recs, _legacy_write_shard_csv)
    out = RUN.stage_assess(CONF, paths, log=lambda *a: None)
    assert out["n_objects"] == len(recs)
    assert not out["verdict"].startswith("NO_DATA_REACHED")
    assert out["funnel"]["n_fitted"] == 40
    rep = out["shard_csv_repairs"]["shard_0_of_1.csv.gz"]
    assert rep["n_dropped"] == 0 and rep["n_repaired_overflow"] >= 2
    assert rep["n_rejoined_newline"] >= 1 and rep["n_numpy_repr_cells"] > 0
    # Values survive the repair exactly: the comma-bearing reason is whole again...
    df, _ = RUN.read_shard_csvs(paths, log=lambda *a: None)
    by = df.set_index("number_mp")
    for r in recs:
        if r.get("reason"):
            assert by.loc[r["number_mp"], "reasons"] == r["reasons"][0]
            assert by.loc[r["number_mp"], "reason"] == r["reason"]
    # ...and the numpy-repr cells are numbers again.
    assert by["a2"].dtype.kind == "f"
    assert float(by.loc[1010, "a2"]) == float(recs[10]["a2"])
    # The controls reach the scorer and are scored, and controls.json lands
    # BEFORE the summary that quotes it.
    c = out["controls"]
    assert c["n_controls"] == 6 and c["verdict"] == "CONTROLS_RECOVERED"
    cj = paths.results / "controls.json"
    sj = paths.results / "summary.json"
    assert cj.stat().st_mtime_ns <= sj.stat().st_mtime_ns
    summ = json.loads(sj.read_text())
    assert summ["coverage"]["n_objects"] == len(recs)
    assert summ["controls"]["n_controls"] == 6


def test_the_new_writer_round_trips_commas_quotes_and_newlines(tmp_path):
    recs = _real_shaped_records()
    paths = _shard_dir(tmp_path, recs, RUN.write_shard_csv)
    df = pd.read_csv(paths.results / "fits" / "shard_0_of_1.csv.gz")   # strict parse
    assert len(df) == len(recs)
    by = df.set_index("number_mp")
    for r in recs:
        if r.get("reason"):
            assert by.loc[r["number_mp"], "reason"] == r["reason"]
            assert by.loc[r["number_mp"], "reasons"] == "|".join(r["reasons"])
    assert by["a2"].dtype.kind == "f"            # no np.float64(...) text
    out = RUN.stage_assess(CONF, paths, log=lambda *a: None)
    assert out["n_objects"] == len(recs) and out["shard_csv_repairs"] == {}


def test_a_budget_stopped_shard_with_failed_objects_keeps_every_completed_record(
        tmp_path, monkeypatch):
    """The failure end to end: stop on the clock, with comma-bearing failures,
    then assess --- the completed objects and the controls must all arrive."""
    numbers = list(range(1, 41))
    worked = _stub_shard_io(monkeypatch, numbers, controls={37, 38})

    def fit_or_fail(n, cols, b, row, *a, **k):
        if n % 3 == 0:
            return ({"number_mp": n, "route": "integrator", "verdict": "RESIDUALS_FAILED",
                     "reason": f"ValueError: shapes ({n},4) and (5,) not aligned"}, None)
        ctrl = n in (37, 38)
        return ({"number_mp": n, "route": "integrator", "verdict": "FITTED",
                 "a2": np.float64(-5.1e-14 if ctrl else 1e-16),
                 "a2_err": np.float64(5e-15 if ctrl else 1e-16),
                 "a2_snr": np.float64(10.2 if ctrl else 1.0), "a2_absorbed_fraction": 0.9,
                 "excess_scatter": 1.0, "n_transits": 30, "arc_days": 900.0,
                 "h": 15.0, "is_control": ctrl,
                 "jpl_a2": -5e-14 if ctrl else float("nan"),
                 "jpl_a2_sigma": 3e-15 if ctrl else float("nan")}, None)

    monkeypatch.setattr(RUN, "fit_object", fit_or_fail)
    monkeypatch.setattr(RUN, "annotate_orbit", lambda r, row, conf: r)
    conf = RUN.load_config({"sextant": {"objects_per_chunk": 10}})
    paths = RUN.Paths.make(tmp_path / "results", tmp_path / "work")
    E.save_json(paths.results / "controls_greenberg2020.json", {"rows": {}})
    clock = iter([0.0] + [60.0 * k for k in range(0, 40)])
    rec = RUN.stage_shard(conf, paths, 0, 1, gaia=_FakeGaia(), client=_FakeGaia(),
                          budget_minutes=1.5, now=lambda: next(clock), log=lambda *a: None)
    assert rec["verdict"] == "OK_PARTIAL_BUDGET" and rec["n_records"] > 0
    assert rec["funnel"]["verdicts"].get("RESIDUALS_FAILED", 0) > 0
    assert worked[:2] == [37, 38]                      # controls were fitted first
    out = RUN.stage_assess(conf, paths, log=lambda *a: None)
    assert out["n_objects"] == rec["n_records"]
    assert out["shard_csv_repairs"] == {}              # the new writer needs no repair
    assert out["controls"]["n_controls"] == 2
    assert out["controls"]["verdict"] == "CONTROLS_RECOVERED"


def test_unreadable_shard_output_is_named_a_pipeline_defect_not_an_empty_sky(tmp_path):
    paths = _shard_dir(tmp_path, _real_shaped_records(), _legacy_write_shard_csv)
    (paths.results / "fits" / "shard_0_of_1.csv.gz").write_bytes(b"not gzip")
    out = RUN.stage_assess(CONF, paths, log=lambda *a: None)
    assert out["n_objects"] == 0
    assert out["verdict"] == "NO_DATA_REACHED__SHARD_OUTPUT_UNREADABLE"
    assert "error" in out["shard_csv_repairs"]["shard_0_of_1.csv.gz"]


def _moving_sun_grid(jd_lo, jd_hi):
    """A Sun on a Jupiter-period reflex circle: extrapolation errors are visible."""
    t = np.arange(jd_lo, jd_hi + 1.0, 1.0)
    w, r = 2 * np.pi / 4332.6, 0.005
    pos = np.stack([r * np.cos(w * t), r * np.sin(w * t), 0 * t], -1)[None]
    vel = np.stack([-r * w * np.sin(w * t), r * w * np.cos(w * t), 0 * t], -1)[None]
    return E.PerturberSet(t_grid=t, pos=pos, vel=vel, gm=np.array([MU]), labels=["sun"])


def test_the_perturbers_refuse_to_extrapolate_past_their_grid():
    """Run 35746692260's integrator failure, reproduced and refused.

    The grid ended at JD 2458930 and SBDB's osculation epoch is 2461200.5.
    The clipped cubic then puts the Sun ~0.02 au from where it is --- which is
    what the probe measured as a median integrator-vs-Horizons disagreement of
    1.4e7 mas.  It must raise instead of answering.
    """
    ps = _moving_sun_grid(2456820.0, 2458930.0)
    p, _v = E.hermite_cubic(ps.t_grid, ps.pos, ps.vel, np.array([2461200.5]))
    w, r = 2 * np.pi / 4332.6, 0.005
    truth = np.array([r * np.cos(w * 2461200.5), r * np.sin(w * 2461200.5), 0.0])
    assert np.linalg.norm(p[0, 0] - truth) > 0.01          # the silent failure
    with pytest.raises(E.EphemerisError, match="refusing to extrapolate"):
        ps.sun_state(np.array([2461200.5]))
    with pytest.raises(E.EphemerisError):
        ps.states(2461200.5)
    ps.sun_state(np.array([2457000.0, 2458930.5]))           # inside (+1 step) is fine


def test_the_perturber_window_reaches_the_sbdb_osculation_epoch():
    lo, hi = E.perturber_window([2461200.5, 2461000.5, None, float("nan"), 2450000.5])
    assert lo == E.WINDOW_JD[0] and hi == 2461200.5
    assert E.perturber_window([]) == E.WINDOW_JD
    # An absurd epoch cannot demand an unbounded grid.
    assert E.perturber_window([2499999.5])[1] == E.WINDOW_JD[1] + E.MAX_EPOCH_EXTENSION_DAYS


def test_the_integrator_skips_an_epoch_its_grid_does_not_reach_and_names_why():
    jd0 = 2457500.0
    pert = _moving_sun_grid(jd0 - 400, jd0 + 400)
    sbdb = {7: {"a": 2.5, "e": 0.1, "i": 5.0, "node": 30.0, "argperi": 40.0, "ma": 10.0,
                "epoch_jd": 2461200.5}}
    cols = {"epoch": np.array([jd0, jd0 + 10.0]), "ra": np.zeros(2)}
    bundles, skipped = RUN.integrator_bundles({7: cols}, sbdb, pert, SYN_CONV)
    assert 7 not in bundles
    assert "outside_perturber_grid" in skipped[7]


def test_load_perturbers_refetches_a_cached_grid_that_stops_short(tmp_path, monkeypatch):
    paths = RUN.Paths.make(tmp_path / "results", tmp_path / "work")
    short = _moving_sun_grid(E.WINDOW_JD[0] - 30, E.WINDOW_JD[1] + 30)
    short.labels = [lab for lab, _c, _g in E.PERTURBERS][:1]
    monkeypatch.setattr(E, "PERTURBERS", E.PERTURBERS[:1])
    short.save(paths.work / "perturbers.npz")
    asked = {}

    def fake_from_horizons(client, jd_lo, jd_hi, on_body=None, **k):
        asked["span"] = (jd_lo, jd_hi)
        return _moving_sun_grid(jd_lo - 30, jd_hi + 30)

    monkeypatch.setattr(E.PerturberSet, "from_horizons", staticmethod(fake_from_horizons))
    # Covers the Gaia window only: reused when nothing later is needed...
    assert RUN.load_perturbers(paths, None, log=lambda *a: None).t_grid[-1] == short.t_grid[-1]
    # ...and re-fetched out to the osculation epoch when it is.
    ps = RUN.load_perturbers(paths, object(), log=lambda *a: None, jd_hi=2461200.5)
    assert asked["span"] == (E.WINDOW_JD[0], 2461200.5)
    assert ps.covers(E.WINDOW_JD[0], 2461200.5)


def test_a_control_refused_on_the_horizons_route_is_scored_through_the_pinned_route(
        tmp_path, monkeypatch):
    """Run 35746692260: all 78 controls RESIDUALS_FAILED, controls={} in assess.

    On the Horizons route a control's bulk fit is refused as circular by
    design; the pinned gravity-only route is the one that can measure it, and
    it used to run only when the bulk route was the integrator.
    """
    numbers = list(range(1, 21))
    _stub_shard_io(monkeypatch, numbers, controls={3, 4})
    monkeypatch.setattr(RUN, "horizons_bundle",
                        lambda n, cols, client, pert, conv: types.SimpleNamespace(route="horizons"))
    monkeypatch.setattr(RUN, "pinned_bundle",
                        lambda n, cols, client, pert, conv, sbdb: types.SimpleNamespace(
                            route="horizons_pinned_gravity_only"))

    def fit(n, cols, b, row, *a, **k):
        if b.route == "horizons" and row.get("nongrav_fitted"):
            return ({"number_mp": n, "route": "horizons", "verdict": "RESIDUALS_FAILED",
                     "reason": "CircularOrbitSourceError: orbit source 'jpl_horizons' carried "
                               "fitted non-gravitational parameters, so ..."}, None)
        ctrl = bool(row.get("nongrav_fitted"))
        return ({"number_mp": n, "route": b.route, "verdict": "FITTED",
                 "a2": -5.1e-14 if ctrl else 1e-16, "a2_err": 5e-15 if ctrl else 1e-16,
                 "a2_snr": 10.2 if ctrl else 1.0, "a2_absorbed_fraction": 0.9,
                 "excess_scatter": 1.0, "n_transits": 30, "arc_days": 900.0, "h": 15.0,
                 "_fits": {"k": 1}}, None)

    def annotate(r, row, conf):
        ctrl = bool(row.get("nongrav_fitted"))
        r.update({"is_control": ctrl, "jpl_nongrav_fitted": ctrl,
                  "jpl_a2": -5e-14 if ctrl else float("nan"),
                  "jpl_a2_sigma": 3e-15 if ctrl else float("nan")})
        return r

    monkeypatch.setattr(RUN, "fit_object", fit)
    monkeypatch.setattr(RUN, "annotate_orbit", annotate)
    conf = RUN.load_config({"sextant": {"objects_per_chunk": 10}})
    paths = RUN.Paths.make(tmp_path / "results", tmp_path / "work")
    E.save_json(paths.results / "controls_greenberg2020.json", {"rows": {}})
    rec = RUN.stage_shard(conf, paths, 0, 1, gaia=_FakeGaia(), client=_FakeGaia(),
                          route="horizons", budget_minutes=600.0, log=lambda *a: None)
    assert rec["route"] == "horizons"
    assert rec["funnel"]["verdicts"].get("RESIDUALS_FAILED", 0) == 0
    df = pd.read_csv(paths.results / "fits" / "shard_0_of_1.csv.gz")
    ctrl = df[df["number_mp"].isin([3, 4])]
    assert set(ctrl["route"]) == {"horizons_pinned_gravity_only"}
    out = RUN.stage_assess(conf, paths, log=lambda *a: None)
    assert out["controls"]["n_controls"] == 2
    assert out["controls"]["verdict"] == "CONTROLS_RECOVERED"


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


def test_a2_distribution_is_quoted_against_the_yarkovsky_expectation():
    """The headline number is |A2| over what thermal recoil would give."""
    df = _table(n_exceed=3, exceed_vetoed=False)
    df["ratio_expected"] = np.abs(df["a2"]) / df["a2_expected_yarkovsky"]
    d = RUN.assess_frame(df, CONF)["a2_distribution"]
    r = d["ratio_to_yarkovsky_expectation_detected"]
    assert r["n"] >= 5 and r["p10"] <= r["median"] <= r["p90"]
    assert r["epsilon_median_measured"] == CONF["epsilon_median_measured"]
    # The injected exceedances are 20x the hard ceiling, so the tail must show.
    assert r["fraction_above_10x"] > 0


def test_assessment_survives_a_shard_written_by_an_earlier_build():
    """A column a previous build did not write degrades that statistic only.

    `assess` gathers shard CSVs that may come from a re-run of one failed shard
    or from another run's artifacts entirely.  A KeyError there would throw away
    every shard that did arrive, which is the opposite of what checkpointing is
    for.
    """
    df = _table().drop(columns=["ratio_expected", "a2_absorbed_fraction",
                                "epsilon_eff", "ratio_realistic",
                                "a2_expected_yarkovsky"], errors="ignore")
    out = RUN.assess_frame(df, CONF)
    assert out["verdict"] == "NO_CEILING_EXCEEDANCE"
    assert out["controls"]["verdict"] == "CONTROLS_RECOVERED"
    d = out["a2_distribution"]
    assert "ratio_to_yarkovsky_expectation_detected" not in d
    assert "absorbed_fraction_median" not in d
    assert d["n_with_a2"] == len(df)          # the statistics that CAN run, do


def test_dadt_to_a2_conversion_reproduces_bennu():
    """The published literature reports ``da/dt``; JPL reports ``A2``.

    Bennu is where the two are both known to better than a per cent: Chesley
    et al. 2014 measured ``da/dt = -19.0 +- 0.1e-4 au/Myr`` and JPL's solution
    carries ``A2 = -4.6e-14 au/day^2``.  If the conversion in
    :func:`controls.a2_from_dadt` were wrong by the orbit-averaging factor ---
    the easy error, since ``<r^-3>`` is not ``<r>^-3`` --- this test fails by
    more than an order of magnitude, and every Greenberg+2020 control would be
    scored against a wrong truth.
    """
    a2 = a2_from_dadt(-19.0, 1.1264, 0.2037)
    assert a2 < 0
    assert abs(a2 / -4.6e-14 - 1.0) < 0.05
    # Sign and linearity, and a refusal on nonsense rather than a silent number.
    assert a2_from_dadt(+19.0, 1.1264, 0.2037) == pytest.approx(-a2)
    assert a2_from_dadt(-38.0, 1.1264, 0.2037) == pytest.approx(2 * a2)
    assert not math.isfinite(a2_from_dadt(-19.0, float("nan"), 0.2))
    assert not math.isfinite(a2_from_dadt(-19.0, 1.1, 1.4))     # e >= 1


def test_greenberg_controls_are_scored_independently_of_jpl():
    """A published da/dt is a control even where SBDB carries no A2."""
    df = _table(n_controls=6)
    # Object 200 has no JPL solution at all; Greenberg published a drift for it.
    k = df.index[df["number_mp"] == 200][0]
    a_au, ecc = float(df.loc[k, "a"]), float(df.loc[k, "e"])
    truth = a2_from_dadt(-8.0, a_au, ecc)
    df.loc[k, "a2"] = truth
    df.loc[k, "a2_err"] = abs(truth) / 8.0
    df.loc[k, "a2_snr"] = 8.0
    yark = {200: {"dadt_1e4_au_per_myr": -8.0, "dadt_sigma_1e4_au_per_myr": -0.8,
                  "a_au": a_au, "e": ecc}}
    out = RUN.assess_frame(df, CONF, yarkovsky=yark)
    c = out["controls"]
    assert c["n_controls"] == 7                    # the six JPL ones plus this
    assert c["n_greenberg2020_rows"] == 1
    got = next(s for s in c["scored"] if s["number_mp"] == 200)
    assert got["published_source"].startswith("Greenberg+2020")
    assert got["vs_published"]["verdict"] == "RECOVERED"
    assert c["against_greenberg2020"]["verdict"] == "CONTROLS_RECOVERED"
    # An unreachable VizieR must not invent one.
    assert RUN.assess_frame(df, CONF)["controls"]["n_greenberg2020_rows"] == 0


def test_assessment_degrades_honestly_on_an_empty_table():
    out = RUN.assess_frame(pd.DataFrame(), CONF)
    assert out["verdict"] == "NO_DATA_REACHED"
    assert summarise_controls([])["verdict"] == "NO_CONTROLS_PRESENT"


def test_control_scoring_distinguishes_unmeasured_from_wrong():
    # below S/N and within 3 sigma of JPL: unexercised, not failed
    assert score_control(1e-14, 4e-14, 5e-14, 2e-15)["verdict"] == "CONSISTENT_BUT_NOT_DETECTED"
    # below S/N but far from JPL: NOT the same thing, and not a free pass
    assert score_control(1e-14, 1e-14, 5e-14, 2e-15)["verdict"] == "INCONSISTENT_BELOW_SNR"
    assert score_control(5e-14, 1e-15, -5e-14, 2e-15)["verdict"] == "SIGN_WRONG"
    assert score_control(4e-14, 2e-15, 5e-14, 2e-15)["verdict"] == "RECOVERED"
    assert score_control(2e-13, 1e-15, 5e-14, 1e-15)["verdict"] == "MAGNITUDE_OFF"
    assert score_control(float("nan"), 1, 5e-14, 1e-15)["verdict"] == "NOT_MEASURED"
    assert score_control(1e-14, 1e-15, float("nan"), 1e-15)["verdict"] == "NO_JPL_VALUE"


def test_controls_that_all_miss_jpl_below_snr_do_not_read_as_merely_insensitive():
    quiet = [score_control(1e-14, 4e-14, 5e-14, 2e-15) for _ in range(3)]
    assert summarise_controls(quiet)["verdict"] == "CONTROLS_BELOW_SENSITIVITY"
    displaced = [score_control(1e-14, 1e-14, 5e-14, 2e-15) for _ in range(3)]
    assert summarise_controls(displaced)["verdict"] == "CONTROLS_INCONSISTENT"
