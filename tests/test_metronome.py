"""Offline test suite for METRONOME --- clocks in stellar flare timing.

No network anywhere (``conftest.py`` raises on any socket).  The suite is the
CI gate and covers, per ``docs/channel-brief.md`` §5:

* an injected strict clock in Kepler-like windows (quarter gaps, 30-min
  cadence) and in TESS-like sectors is recovered at the right period with
  candidate-grade quality;
* a Poisson flare star is never a candidate; a batch of them yields none;
* a rotationally modulated flare star (rate ∝ 1 + cos) is rejected by
  ``rotation_alias`` when P_rot is known and by ``jitter_too_large`` when not;
* cross-star coincidences are removed before any scan;
* a catalogued RR Lyrae whose cycles were chopped into "flares" is rejected by
  ``periodic_variable``;
* an empty or failed VizieR response yields ``NO_DATA_REACHED`` /
  ``QUERY_RETURNED_ZERO_ROWS`` with zero counts and never a candidate;
* every rejection rule has a case that trips it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seti.metronome import acquire as acq
from seti.metronome.clock import (
    analyze_star,
    bh_fdr,
    cross_star_coincidence,
    decluster,
    frequency_grid,
    fundamental_period,
    gap_integer_fraction,
    gumbel_tail_p,
    h_statistic,
    phase_stats,
    shuffle_waiting_times,
)
from seti.metronome.run import (
    load_metronome_config,
    metronome_run,
    screen_catalogue,
    stage_assess,
)
from seti.metronome.vet import (
    HARD_VETO_ORDER,
    REPORT_FLAGS,
    assign_tiers,
    cadence_alias,
    calibrate_jitter,
    periodic_variable,
    rejection_counters,
    rotation_alias,
    vet_star,
)
from seti.metronome.windows import (
    KEPLER_LC_CADENCE_DAYS,
    Windows,
    guess_time_system,
    kepler_quarter_windows,
    star_windows,
    tess_sector_windows,
    windows_from_events,
)

# Narrow the period grid purely for runtime; science defaults live in
# config/metronome.yaml.
SCAN = dict(min_period_days=0.5, oversample=3.0, max_period_days=200.0)
NULL = dict(n_max=100, n_shuffle=40)
VET = dict()


# ---------------------------------------------------------------------------
# synthetic event lists
# ---------------------------------------------------------------------------
def kepler_windows_short() -> Windows:
    """Q2-Q8: seven quarters with real-shaped gaps, ~640 days, for speed."""
    full = kepler_quarter_windows()
    return Windows(full.starts[2:9], full.stops[2:9], cadence_days=KEPLER_LC_CADENCE_DAYS,
                   label="kepler_q2_q8")


def clock_events(mission: Windows, period: float, t0: float, *, duty: float = 0.5,
                 jitter_days: float = 0.002, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = np.arange(0, int(mission.span / period) + 3)
    t = t0 + k * period + rng.normal(0.0, jitter_days, len(k))
    t = t[mission.contains(t)]
    t = t[rng.random(len(t)) < duty]
    return mission.quantize(np.sort(t))


def poisson_events(mission: Windows, n: int, seed: int) -> np.ndarray:
    return mission.sample(n, np.random.default_rng(seed))


def rotation_events(mission: Windows, prot: float, n: int, seed: int) -> np.ndarray:
    """Flare rate ∝ 1 + cos(2π t / P_rot): visibility modulation, large jitter."""
    rng = np.random.default_rng(seed)
    cand = mission.sample(40 * n, rng)
    keep = rng.random(len(cand)) < 0.5 * (1.0 + np.cos(2.0 * np.pi * cand / prot))
    return np.sort(cand[keep][:n])


def bursty_events(mission: Windows, n_bursts: int, per_burst: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = mission.sample(n_bursts, rng)
    t = np.concatenate([c + np.sort(rng.uniform(0.0, 1.5, per_burst)) for c in centres])
    t = t[mission.contains(t)]
    return mission.quantize(np.sort(t))


def analyze(t, mission=None, seed=1, **kw):
    mission = mission or kepler_windows_short()
    w = star_windows(t, mission)
    return analyze_star(t, w, scan_conf=dict(SCAN, **kw.pop("scan", {})),
                        null_conf=dict(NULL, **kw.pop("null", {})),
                        rng=np.random.default_rng(seed), **kw)


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------
def test_observed_time_roundtrip_and_gap_handling():
    w = kepler_windows_short()
    t = np.array([w.starts[0] + 1.0, w.stops[0] + 0.5, w.starts[3] + 10.0])
    tau = w.observed_time(t)
    back = w.real_time(tau)
    assert np.isclose(back[0], t[0]) and np.isclose(back[2], t[2])
    # a time inside a gap maps to the window boundary (end of the preceding
    # window == start of the next one in observed-time coordinates)
    assert np.isclose(back[1], w.stops[0]) or np.isclose(back[1], w.starts[1])
    assert w.contains([t[0], t[2]]).all() and not w.contains([t[1]])[0]


def test_sample_is_cadence_quantised_and_inside_windows():
    w = kepler_windows_short()
    s = w.sample(500, np.random.default_rng(3))
    assert w.contains(s).all()
    frac = ((s - w.t_ref) / w.cadence_days) % 1.0
    assert np.all(np.minimum(frac, 1 - frac) < 1e-6)


def test_windows_from_events_finds_the_gaps():
    mission = kepler_windows_short()
    allt = mission.sample(20000, np.random.default_rng(5))
    w = windows_from_events(allt, cadence_days=KEPLER_LC_CADENCE_DAYS)
    assert w.n == mission.n, (w.as_dict(), mission.as_dict())
    assert np.all(np.abs(w.starts - mission.starts) <= 1.0)
    assert np.all(np.abs(w.stops - mission.stops) <= 1.0)


def test_star_windows_clips_to_span_and_drops_presumed_unobserved():
    mission = kepler_windows_short()
    rng = np.random.default_rng(9)
    # events only in windows 1, 2, 4, 5 at a high rate: window 3 has zero events
    parts = [Windows(mission.starts[[i]], mission.stops[[i]]).sample(40, rng)
             for i in (1, 2, 4, 5)]
    t = np.sort(np.concatenate(parts))
    w = star_windows(t, mission, drop_expected=5.0)
    assert w.n == 4, w.as_dict()
    assert w.starts[0] >= mission.starts[1] - 0.5
    assert "dropped1" in w.label


def test_guess_time_system():
    assert guess_time_system(np.array([500.0, 900.0, 1400.0]), "kepler") == "BKJD"
    assert guess_time_system(np.array([1400.0, 1900.0, 2300.0]), "tess") == "BTJD"
    assert guess_time_system(np.array([2458000.5, 2458100.5])) == "BJD"
    assert guess_time_system(np.array([58000.5, 58100.5])) == "MJD"
    assert guess_time_system(np.array([])) == "unknown"


# ---------------------------------------------------------------------------
# statistic
# ---------------------------------------------------------------------------
def test_frequency_grid_caps_the_longest_period_at_a_third_of_the_span():
    f = frequency_grid(600.0, min_period=0.5, max_period=1000.0, oversample=3.0)
    assert np.isclose(1.0 / f[0], 200.0)
    assert 1.0 / f[-1] >= 0.5 - 1e-9
    assert len(frequency_grid(1.0, min_period=0.5, max_period=100.0)) == 0


def test_h_statistic_reduces_to_rayleigh_for_one_harmonic():
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 100, 50))
    f = np.array([0.3, 1.7])
    h = h_statistic(t, f, m_max=1)
    for i, fi in enumerate(f):
        phi = 2 * np.pi * np.mod(fi * t, 1.0)
        z2 = (2.0 / len(t)) * (np.cos(phi).sum() ** 2 + np.sin(phi).sum() ** 2)
        assert np.isclose(h[i], z2)


def test_phase_stats_extremes():
    t = 100.0 + 2.5 * np.arange(40)
    ps = phase_stats(t, 2.5)
    assert ps["Q"] > 0.99 and ps["jitter"] < 1e-6 and ps["f_in_window"] == 1.0
    rng = np.random.default_rng(1)
    ps = phase_stats(rng.uniform(0, 1000, 400), 2.5)
    assert ps["Q"] < 0.3 and ps["jitter"] > 0.2


def test_fundamental_period_walks_up_from_a_subharmonic():
    t = 100.0 + 3.0 * np.arange(60)
    assert np.isclose(fundamental_period(t, 0.5, max_period=100.0), 3.0)
    assert np.isclose(fundamental_period(t, 1.0, max_period=100.0), 3.0)
    assert np.isclose(fundamental_period(t, 3.0, max_period=100.0), 3.0)


def test_gap_integer_fraction_uses_same_window_pairs_only():
    w = kepler_windows_short()
    t = clock_events(w, 2.0, w.starts[0] + 0.3, duty=1.0, jitter_days=0.0, seed=0)
    frac, n = gap_integer_fraction(t, 2.0, w, tol=0.05)
    assert frac == 1.0 and n == len(t) - w.n
    rng = np.random.default_rng(0)
    frac, _ = gap_integer_fraction(w.sample(200, rng), 2.0, w, tol=0.05)
    assert frac < 0.3


def test_decluster_merges_complex_flares():
    t = np.array([10.0, 10.02, 10.05, 12.0, 15.0, 15.01])
    tt, ee, n = decluster(t, np.arange(6.0), gap_days=0.1)
    assert n == 3 and list(tt) == [10.0, 12.0, 15.0] and list(ee) == [0.0, 3.0, 4.0]


def test_shuffle_preserves_the_waiting_time_multiset():
    w = kepler_windows_short()
    t = w.sample(60, np.random.default_rng(4))
    s = shuffle_waiting_times(t, w, np.random.default_rng(5))
    assert len(s) == len(t) and w.contains(s).all()
    g0 = np.sort(np.diff(w.observed_time(t)))
    g1 = np.sort(np.diff(w.observed_time(s)))
    # equal up to cadence re-quantisation
    assert np.all(np.abs(g0 - g1) <= 2 * w.cadence_days + 1e-9)


def test_gumbel_tail_extrapolation_is_monotone_and_finite():
    rng = np.random.default_rng(0)
    null = rng.gumbel(30.0, 3.0, 500)
    p1, mu, beta = gumbel_tail_p(45.0, null)
    p2, _, _ = gumbel_tail_p(60.0, null)
    assert 0 < p2 < p1 < 0.1 and abs(mu - 30) < 1 and abs(beta - 3) < 0.6
    assert np.isnan(gumbel_tail_p(45.0, null[:5])[0])


def test_bh_fdr():
    p = np.array([0.001, 0.01, 0.03, 0.2, 0.5, np.nan])
    rej = bh_fdr(p, 0.05)
    assert rej.tolist() == [True, True, True, False, False, False]
    assert not bh_fdr(np.array([0.2, 0.3]), 0.05).any()


# ---------------------------------------------------------------------------
# injection / recovery
# ---------------------------------------------------------------------------
def test_injected_clock_is_recovered_in_kepler_windows():
    w = kepler_windows_short()
    P = 3.137
    t = clock_events(w, P, w.starts[0] + 1.3, duty=0.5, jitter_days=0.002, seed=11)
    assert len(t) >= 60
    r = analyze(t, w)
    assert r["status"] == "scanned" and r["null_computed"]
    assert abs(r["period"] / P - 1.0) < 1e-3, r["period"]
    assert r["Q"] > 0.95 and r["jitter"] < 0.01 and r["gap_integer_frac"] > 0.9
    assert r["p_window"] < 1e-6
    assert r["wn_n_trials"] == NULL["n_max"] and r["wn_n_exceed"] == 0


def test_injected_clock_is_recovered_in_tess_sectors():
    w = tess_sector_windows(1325.3, 6)
    P = 0.913
    t = clock_events(w, P, 1326.0, duty=0.6, jitter_days=0.0005, seed=12)
    assert len(t) >= 80
    r = analyze(t, w, scan=dict(min_period_days=0.2))
    assert abs(r["period"] / P - 1.0) < 1e-3, r["period"]
    assert r["Q"] > 0.95 and r["jitter"] < 0.01 and r["p_window"] < 1e-6


def test_clock_with_missed_cycles_still_lands_on_the_fundamental():
    w = kepler_windows_short()
    P = 5.21
    t = clock_events(w, P, w.starts[0] + 2.0, duty=0.25, jitter_days=0.003, seed=13)
    r = analyze(t, w)
    assert abs(r["period"] / P - 1.0) < 1e-3
    assert 0.15 < r["cycle_occupancy"] < 0.4


def test_poisson_star_is_screened_out_or_ordinary():
    w = kepler_windows_short()
    r = analyze(poisson_events(w, 50, seed=21), w)
    assert r["status"] == "scanned"
    assert r["p_window"] > 0.01
    assert r["Q"] < 0.6 and r["jitter"] > 0.12


def test_batch_of_poisson_stars_yields_no_candidate():
    w = kepler_windows_short()
    recs = []
    for i in range(30):
        r = analyze(poisson_events(w, 12 + 3 * (i % 7), seed=100 + i), w)
        r.update({"star_key": f"kepler:{i}", "star_id": str(i), "mission": "kepler"})
        recs.append(r)
    vetted = assign_tiers(recs, {}, VET)
    tiers = rejection_counters(vetted)["tiers"]
    assert tiers["candidate"] == 0 and tiers["interest"] == 0
    assert tiers["none"] >= 28


def test_rotation_modulated_star_is_rejected_by_rotation_alias():
    w = kepler_windows_short()
    prot = 2.5
    t = rotation_events(w, prot, 160, seed=31)
    r = analyze(t, w)
    assert r["p_window"] < 1e-3            # the coherence is real ...
    assert r["jitter"] > 0.12 and r["Q"] < 0.6   # ... but it is not a clock
    r.update({"star_key": "kepler:rot", "star_id": "rot", "mission": "kepler"})
    v = assign_tiers([r], {"kepler:rot": {"mission": "kepler", "prot": prot,
                                          "variability_catalogues_reached": True}}, VET)[0]
    assert v["first_veto"] == "rotation_alias" and v["tier"] == "none"
    # unknown rotation: the jitter threshold still kills it
    v2 = assign_tiers([r], {"kepler:rot": {"mission": "kepler", "prot": float("nan"),
                                           "variability_catalogues_reached": True}}, VET)[0]
    assert v2["first_veto"] == "jitter_too_large" and v2["tier"] == "none"
    assert "rotation_unknown" in v2["flags"]


def test_rotation_alias_catches_harmonics():
    assert rotation_alias(2.5, 2.5)[0] and rotation_alias(1.25, 2.5)[1] == 0.5
    assert rotation_alias(5.0, 2.5)[0] and rotation_alias(2.5 / 3, 2.5)[0]
    assert not rotation_alias(1.7, 2.5)[0]
    assert not rotation_alias(1.7, float("nan"))[0]


def test_cross_star_coincidence_removes_shared_epochs_before_scanning():
    w = kepler_windows_short()
    rng = np.random.default_rng(41)
    sids, ts = [], []
    for s in range(40):
        t = w.sample(15, rng)
        sids += [str(s)] * len(t)
        ts += list(t)
    dump = w.quantize(np.array([w.starts[2] + 20.0]))[0]        # a momentum dump
    for s in range(40):
        sids.append(str(s))
        ts.append(dump)
    cc = cross_star_coincidence(np.array(sids), np.array(ts), bin_days=2 * w.cadence_days,
                                min_stars=5, tail_p=1e-6)
    assert cc["n_bad_bins"] == 1 and cc["n_removed_events"] == 40
    assert abs(cc["bad_bins"][0] - dump) < 2 * w.cadence_days
    # and the screen stage counts it per star
    ev = pd.DataFrame({"star_id": sids, "t_peak": ts})
    conf = load_metronome_config()
    conf["windows"]["min_events_for_data_driven"] = 10 ** 9
    conf["scan"].update(SCAN)
    conf["null"].update(NULL)
    recs, rep = screen_catalogue(ev, "synthetic", "kepler", conf, seed=1)
    assert rep["cross_star"]["n_removed_events"] == 40
    assert all(r["n_removed_cross_star"] == 1 for r in recs)
    assert rep["n_events_after_cross_star"] == len(ev) - 40


def test_catalogued_rr_lyrae_cycles_are_rejected_as_periodic_variable():
    w = kepler_windows_short()
    P = 0.5668
    t = clock_events(w, P, w.starts[0] + 0.1, duty=0.7, jitter_days=0.001, seed=51)
    r = analyze(t[:200], w)
    assert r["p_window"] < 1e-6 and abs(r["period"] / P - 1) < 1e-3
    r.update({"star_key": "kepler:rr", "star_id": "rr", "mission": "kepler"})
    ctx = {"kepler:rr": {"mission": "kepler", "prot": float("nan"),
                         "catalogued_periods": [("vsx", 0.56681, "RRAB")],
                         "variability_catalogues_reached": True}}
    v = assign_tiers([r], ctx, VET)[0]
    assert v["first_veto"] == "periodic_variable" and v["tier"] == "none"
    assert periodic_variable(1.1336, [("vsx", 0.5668, "RRAB")])[0]     # 2x harmonic
    assert not periodic_variable(0.7, [("vsx", 0.5668, "RRAB")])[0]


def test_cadence_alias_trips_on_instrumental_periods():
    inst = {"orbit": 13.7, "cadence_2min": 0.0013889}
    assert cadence_alias(13.7, inst)[1].startswith("orbit")
    assert cadence_alias(27.4, inst)[0] and cadence_alias(6.85, inst)[0]
    assert not cadence_alias(11.0, inst)[0]
    rec = _sig_record(period=13.72)
    v = vet_star(rec, {"mission": "tess", "prot": 5.0, "variability_catalogues_reached": True})
    assert v["first_veto"] == "cadence_alias" and v["tier"] == "none"


def _sig_record(**kw) -> dict:
    """A candidate-grade record; tests override one field at a time."""
    rec = {"status": "scanned", "fdr_significant": True, "fdr_watch": True, "period": 3.137,
           "Q": 0.97, "jitter": 0.005, "f_in_window": 1.0, "gap_integer_frac": 0.95,
           "n_gaps_used": 40, "p_window": 1e-9, "p_shuffle": 0.02, "p_window_source": "empirical",
           "energy_phase_p": 0.4, "wn_truncated_by_budget": False, "mission": "kepler"}
    rec.update(kw)
    return rec


FULL_CTX = {"mission": "kepler", "prot": 11.0, "prot_source": "mcquillan2014",
            "catalogued_periods": [], "variability_catalogues_reached": True}


def test_candidate_tier_when_every_veto_is_applied_and_passed():
    v = vet_star(_sig_record(), FULL_CTX)
    assert v["tier"] == "candidate" and v["first_veto"] is None and v["flags"] == []


def test_interest_tier_when_a_veto_could_not_be_applied():
    v = vet_star(_sig_record(), dict(FULL_CTX, prot=float("nan")))
    assert v["tier"] == "interest" and "rotation_unknown" in v["flags"]
    v = vet_star(_sig_record(), dict(FULL_CTX, variability_catalogues_reached=False))
    assert v["tier"] == "interest" and "variability_catalogue_unreached" in v["flags"]


def test_watch_tier_for_loose_quality_or_watch_only_significance():
    v = vet_star(_sig_record(Q=0.7, jitter=0.09), FULL_CTX)
    assert v["tier"] == "watch"
    v = vet_star(_sig_record(fdr_significant=False), FULL_CTX)
    assert v["tier"] == "watch"


def test_jitter_too_large_trips():
    v = vet_star(_sig_record(Q=0.4, jitter=0.2), FULL_CTX)
    assert v["first_veto"] == "jitter_too_large" and v["tier"] == "none"


def test_bursty_random_trips_only_without_clock_like_gaps():
    # shuffle null does not beat the data, gaps are not integer periods -> bursty
    v = vet_star(_sig_record(p_shuffle=0.6, gap_integer_frac=0.1), FULL_CTX)
    assert v["first_veto"] == "bursty_random"
    # a strict clock survives the shuffle too, but its gaps ARE integer periods
    v = vet_star(_sig_record(p_shuffle=1.0, gap_integer_frac=0.98), FULL_CTX)
    assert v["tier"] == "candidate"


def test_bursty_star_end_to_end_is_not_a_candidate():
    w = kepler_windows_short()
    t = bursty_events(w, 12, 6, seed=61)
    r = analyze(t, w, scan=dict(decluster_gap_days=0.02))
    r.update({"star_key": "kepler:b", "star_id": "b", "mission": "kepler"})
    v = assign_tiers([r], {"kepler:b": FULL_CTX}, VET)[0]
    assert v["tier"] in ("none", "watch")


def test_insufficient_and_not_significant_are_counted():
    v = vet_star({"status": "insufficient_events"}, FULL_CTX)
    assert v["first_veto"] == "insufficient_events"
    v = vet_star(_sig_record(fdr_watch=False, fdr_significant=False), FULL_CTX)
    assert v["first_veto"] == "not_significant"


def test_energy_incoherent_is_report_only():
    v = vet_star(_sig_record(energy_phase_p=1e-4), FULL_CTX)
    assert "energy_incoherent" in v["flags"] and v["tier"] == "candidate"
    v = vet_star(_sig_record(p_window_source="gumbel_extrapolated",
                             wn_truncated_by_budget=True), FULL_CTX)
    assert {"p_extrapolated", "null_truncated_by_budget"} <= set(v["flags"])
    assert v["tier"] == "candidate"


def test_every_rejection_rule_has_a_counter():
    recs = [dict(_sig_record(period=13.7), star_key="k:1", mission="tess"),
            dict(_sig_record(period=5.5), star_key="k:2"),
            dict(_sig_record(period=0.5668), star_key="k:3"),
            dict(_sig_record(p_shuffle=0.9, gap_integer_frac=0.05), star_key="k:4"),
            dict(_sig_record(Q=0.2, jitter=0.3), star_key="k:5"),
            dict({"status": "insufficient_events", "star_key": "k:6"}),
            dict(_sig_record(p_window=0.9), star_key="k:7")]
    ctx = {"k:1": {"mission": "tess", "prot": 100.0, "variability_catalogues_reached": True},
           "k:2": {"mission": "kepler", "prot": 11.0, "variability_catalogues_reached": True},
           "k:3": {"mission": "kepler", "prot": 30.0, "variability_catalogues_reached": True,
                   "catalogued_periods": [("vsx", 0.5668, "RRAB")]},
           "k:4": FULL_CTX, "k:5": FULL_CTX, "k:7": FULL_CTX}
    vetted = assign_tiers(recs, ctx, VET)
    first = rejection_counters(vetted)["first_veto"]
    for rule in HARD_VETO_ORDER + ("insufficient_events", "not_significant"):
        assert first[rule] >= 1, (rule, first)
    flags = rejection_counters(vetted)["flags_raised"]
    for f in REPORT_FLAGS:
        assert f in flags


def test_calibration_reports_where_thresholds_sit():
    vetted = [dict(_sig_record(), first_veto="rotation_alias", jitter=j, Q=0.3,
                   gap_integer_frac=0.1, status="scanned")
              for j in np.linspace(0.15, 0.3, 20)]
    c = calibrate_jitter(vetted)
    assert c["rotation_alias_population"]["jitter"]["n"] == 20
    assert c["rotation_alias_population"]["jitter"]["p5"] > c["thresholds"]["jitter_max"]
    assert c["fraction_of_rotation_population_below_jitter_max"] == 0.0


# ---------------------------------------------------------------------------
# schema discovery (pure) and degraded acquisition
# ---------------------------------------------------------------------------
def test_resolve_event_columns_on_realistic_names():
    yang_like = ["recno", "KIC", "Tstart", "Tpeak", "Tend", "Amp", "E", "Prot"]
    r = acq.resolve_event_columns(yang_like)
    assert r["star_id"] == "KIC" and r["t_peak"] == "Tpeak" and r["t_start"] == "Tstart"
    assert r["t_end"] == "Tend" and r["energy"] == "E" and r["prot"] == "Prot"
    pietras_like = ["TIC", "Sector", "BJDstart", "BJDpeak", "BJDend", "Ampl", "logE", "RAJ2000",
                    "DEJ2000"]
    r = acq.resolve_event_columns(pietras_like)
    assert r["star_id"] == "TIC" and r["t_peak"] == "BJDpeak" and r["sector"] == "Sector"
    assert r["energy"] == "logE" and r["ra"] == "RAJ2000" and r["dec"] == "DEJ2000"
    # substring traps: "Perr" must not resolve as a period
    assert "prot" not in acq.resolve_event_columns(["KIC", "Tpeak", "Perr"])


def test_score_event_table_rejects_a_per_star_table():
    score, roles, why = acq.score_event_table(["KIC", "Teff", "logg", "Nflares", "Prot"])
    assert score == 0 and "no t_peak/t_start" in why
    score, _, why = acq.score_event_table(["KIC", "Tpeak", "E"])
    assert score > 0 and why == "usable"


class _FakeTAP:
    """Scripted TAP: ``mode`` = 'fail' | 'zero' | 'ok'."""

    def __init__(self, mode: str, events: pd.DataFrame | None = None):
        self.mode, self.events, self.calls = mode, events, []

    def __call__(self, adql: str):
        self.calls.append(adql)
        if self.mode == "fail":
            raise RuntimeError("CONNECT tunnel failed, response 403")
        if "TAP_SCHEMA.tables" in adql:
            if self.mode == "zero":
                return pd.DataFrame(columns=["table_name", "description"])
            return pd.DataFrame({"table_name": ['"J/ApJS/241/29/table3"'],
                                 "description": ["Flares"]})
        if "TAP_SCHEMA.columns" in adql:
            if "V/133/kic" in adql:
                return pd.DataFrame({"column_name": ["KIC", "RAJ2000", "DEJ2000"]})
            return pd.DataFrame({"column_name": list(self.events.columns)})
        if "COUNT(*)" in adql:
            return pd.DataFrame({"n": [len(self.events)]})
        if "V/133/kic" in adql:
            ids = [s.strip() for s in adql.split("IN (")[1].rstrip(")").split(",")]
            return pd.DataFrame({"KIC": ids, "RAJ2000": [290.0] * len(ids),
                                 "DEJ2000": [44.0] * len(ids)})
        if "SELECT TOP 200" in adql:
            return self.events[["Tpeak"]].head(200)
        return self.events.copy()


def test_discovery_separates_failed_from_zero_rows():
    log = acq.AcquisitionLog()
    d = acq.discover_event_table("k", "J/ApJS/241/29", ("flare",), query_fn=_FakeTAP("fail"),
                                 log=log)
    assert d.table is None and d.status == "QUERY_FAILED"
    assert log.as_dict()["any_query_failed"]
    log = acq.AcquisitionLog()
    d = acq.discover_event_table("k", "J/ApJS/241/29", ("flare",), query_fn=_FakeTAP("zero"),
                                 log=log)
    assert d.table is None and d.status == "QUERY_RETURNED_ZERO_ROWS"
    assert not log.as_dict()["any_query_failed"]
    assert log.as_dict()["n_query_returned_zero_rows"] >= 1


def test_run_with_unreachable_archive_is_no_data_reached(tmp_path):
    out = tmp_path / "metronome"
    rep = metronome_run(None, stage="all", out_root=out, query_fn=_FakeTAP("fail"),
                        offline=True)
    assert rep["verdict"] == "NO_DATA_REACHED"
    s = json.loads((out / "summary.json").read_text())
    assert "generated_utc" in s and s["n_stars_scanned"] == 0
    assert s["tiers"]["candidate"] == 0 and s["tiers"]["interest"] == 0
    c = json.loads((out / "candidates.json").read_text())
    assert c["candidates"] == []
    probe = json.loads((out / "probe.json").read_text())
    assert probe["n_usable"] == 0
    assert all(v["status"] == "QUERY_FAILED" for v in probe["catalogues"].values())


def test_start_time_only_table_uses_start_as_event_time_and_says_so():
    ev = pd.DataFrame({"KIC": ["1", "1", "2"], "Tstart": [100.0, 103.0, 110.0]})
    fake = _FakeTAP("ok", ev)
    d = acq.discover_event_table("k", "J/ApJS/241/29", query_fn=fake)
    assert "t_peak" not in d.roles and d.roles["t_start"] == "Tstart"
    df = acq.fetch_events(d, query_fn=fake)
    assert list(df["t_peak"]) == [100.0, 103.0, 110.0]
    assert (df["t_peak_source"] == "t_start").all()
    assert "t_peak_from_start" not in df.columns


def test_bjd_and_mjd_catalogues_are_normalised_to_the_mission_system():
    from seti.metronome.run import normalise_time_system

    df = pd.DataFrame({"t_peak": [2455000.0], "t_start": [2454999.9]})
    out, native = normalise_time_system(df, "BJD", "kepler")
    assert native == "BKJD" and np.isclose(out["t_peak"][0], 167.0)
    assert np.isclose(out["t_start"][0], 166.9)
    df = pd.DataFrame({"t_peak": [58000.0]})
    out, native = normalise_time_system(df, "MJD", "tess")
    assert native == "BTJD" and np.isclose(out["t_peak"][0], 58000.0 + 2400000.5 - 2457000.0)
    df = pd.DataFrame({"t_peak": [900.0]})
    out, native = normalise_time_system(df, "BKJD", "kepler")
    assert native == "BKJD" and out["t_peak"][0] == 900.0


def test_rotation_discovery_with_empty_preferred_goes_straight_to_keywords():
    class TAP(_FakeTAP):
        def __call__(self, adql):
            self.calls.append(adql)
            if "TAP_SCHEMA.tables" in adql:
                assert "LIKE '%%'" not in adql
                return pd.DataFrame(columns=["table_name", "description"])
            raise AssertionError(adql)

    fake = TAP("ok", pd.DataFrame())
    df, rec = acq.discover_and_fetch_rotation("tess_rot", "", ("rotation", "TESS"), query_fn=fake)
    assert not len(df) and rec["status"] == "QUERY_RETURNED_ZERO_ROWS"
    assert len(fake.calls) == 1 and "description LIKE" in fake.calls[0]


def test_cone_failure_is_per_star_not_per_catalogue():
    pos = pd.DataFrame({"star_id": ["a", "b"], "ra": [10.0, 20.0], "dec": [1.0, 2.0]})

    def cone(table, ra, dec, r):
        if ra == 20.0:
            raise RuntimeError("timeout")
        return pd.DataFrame({"Period": [0.5], "Type": ["RRAB"]})

    v, reached = acq.fetch_variable_context(pos, {"vsx": {"table": "B/vsx/vsx"}}, cone_fn=cone)
    assert reached == {"a": {"vsx"}}
    assert v["a"] == [("vsx", 0.5, "RRAB")] and "b" not in v


def test_run_with_empty_archive_is_zero_rows_not_a_null(tmp_path):
    out = tmp_path / "metronome"
    rep = metronome_run(None, stage="all", out_root=out, query_fn=_FakeTAP("zero"),
                        offline=True)
    assert rep["verdict"] == "QUERY_RETURNED_ZERO_ROWS"
    assert rep["tiers"]["candidate"] == 0
    assert "NOT a null result" in rep["note"]


def _synthetic_catalogue() -> tuple[pd.DataFrame, dict]:
    w = kepler_windows_short()
    rows = []
    P_clock = 3.137
    for t in clock_events(w, P_clock, w.starts[0] + 1.3, duty=0.5, jitter_days=0.002, seed=71):
        rows.append(("1000001", t, 1e33))
    for t in rotation_events(w, 2.5, 150, seed=72):
        rows.append(("1000002", t, 1e32))
    for i in range(12):
        for t in poisson_events(w, 15 + i, seed=200 + i):
            rows.append((str(2000000 + i), t, 1e32))
    ev = pd.DataFrame(rows, columns=["KIC", "Tpeak", "E"])
    return ev, {"P_clock": P_clock, "prot_rot": 2.5}


def test_end_to_end_synthetic_run_reaches_pending_vet(tmp_path):
    ev, truth = _synthetic_catalogue()
    fake = _FakeTAP("ok", ev)
    out = tmp_path / "metronome"
    conf = load_metronome_config()
    conf["catalogues"] = {"kepler_synth": {"mission": "kepler", "preferred": "J/ApJS/241/29",
                                           "keywords": ["flare"], "enabled": True}}
    conf["rotation_catalogues"] = {}
    conf["windows"]["min_events_for_data_driven"] = 10 ** 9
    conf["scan"].update(SCAN)
    conf["null"].update(NULL)
    from seti.metronome.run import stage_acquire, stage_probe, stage_screen

    probe = stage_probe(conf, out, query_fn=fake)
    assert probe["n_usable"] == 1
    d = probe["catalogues"]["kepler_synth"]
    assert d["roles"]["t_peak"] == "Tpeak" and d["time_system_guess"] == "BKJD"
    acq_rep = stage_acquire(conf, out, query_fn=fake)
    assert acq_rep["catalogues"]["kepler_synth"]["status"] == "OK"
    assert (out / "data" / "kepler_synth_events.parquet").exists()
    stage_screen(conf, out)
    assert (out / "stars_kepler_synth.csv").exists()

    # offline assess: rotation unknown, variability catalogues unreached -> interest
    s = stage_assess(conf, out, offline=True)
    assert s["verdict"].endswith("CLOCK_CANDIDATES_PENDING_VET"), s["verdict"]
    assert s["verdict"].startswith("DEGRADED_SOURCE")       # rotation table absent
    assert s["n_interest"] == 1 and s["n_candidates"] == 0
    c = json.loads((out / "candidates.json").read_text())
    top = c["candidates"][0]
    assert top["star_id"] == "1000001" and abs(top["period"] / truth["P_clock"] - 1) < 1e-3
    assert top["tier"] == "interest" and "rotation_unknown" in top["flags"]
    assert s["funnel"]["stars_scanned"] == 14 and "generated_utc" in s
    assert s["rejection_counters"]["first_veto"]["jitter_too_large"] >= 1  # the rotator

    # full assess with a rotation table on disk and scripted cones -> candidate,
    # and the rotator is now named as rotation_alias
    rot = pd.DataFrame({"star_id": ["1000001", "1000002"], "prot": [11.0, truth["prot_rot"]],
                        "prot_source": ["fake_rot", "fake_rot"]})
    rot.to_parquet(out / "data" / "rotation_kepler.parquet", index=False)
    calls = []

    def cone(table, ra, dec, r):
        calls.append(table)
        return pd.DataFrame()

    s2 = stage_assess(conf, out, offline=False, query_fn=fake, cone_fn=cone)
    assert s2["n_candidates"] == 1 and s2["n_interest"] == 0
    assert s2["rejection_counters"]["first_veto"]["rotation_alias"] == 1
    assert calls and all(s2["variability_catalogues_reached"].values())
    assert not s2["degraded"]
    assert s2["verdict"] == "CLOCK_CANDIDATES_PENDING_VET"
    assert s2["jitter_calibration"]["rotation_alias_population"]["jitter"]["n"] == 1


def test_sharding_partitions_stars_without_overlap():
    ev, _ = _synthetic_catalogue()
    ev = ev.rename(columns={"KIC": "star_id", "Tpeak": "t_peak", "E": "energy"})
    conf = load_metronome_config()
    conf["windows"]["min_events_for_data_driven"] = 10 ** 9
    conf["scan"].update(SCAN)
    conf["null"].update(NULL)
    a, _ = screen_catalogue(ev, "s", "kepler", conf, shard=0, n_shards=3, seed=1)
    b, _ = screen_catalogue(ev, "s", "kepler", conf, shard=1, n_shards=3, seed=1)
    c, _ = screen_catalogue(ev, "s", "kepler", conf, shard=2, n_shards=3, seed=1)
    keys = [r["star_key"] for r in a + b + c]
    assert len(keys) == len(set(keys)) == 14


def test_config_file_parses_and_carries_every_threshold():
    conf = load_metronome_config()
    for k in ("catalogues", "rotation_catalogues", "variability_catalogues", "windows",
              "cross_star", "scan", "null", "vet", "acquire"):
        assert k in conf
    assert conf["scan"]["n_min"] == 8
    assert set(conf["vet"]["instrumental_periods"]) >= {"kepler", "tess"}
    assert Path("config/metronome.yaml").exists()
    for name, spec in conf["catalogues"].items():
        assert spec["mission"] in ("kepler", "tess") and spec["preferred"], name


def test_workflow_and_lit_script_exist():
    assert Path(".github/workflows/metronome.yml").exists()
    assert Path("scripts/metronomelit_fetch.py").exists()
    assert Path("docs/metronome.md").exists()


@pytest.mark.parametrize("n", [8, 9])
def test_n_min_boundary(n):
    w = kepler_windows_short()
    t = poisson_events(w, n, seed=n)
    r = analyze(t, w)
    assert r["status"] == "scanned"
    r = analyze(t[:7], w)
    assert r["status"] == "insufficient_events" and np.isnan(r["p_window"])
