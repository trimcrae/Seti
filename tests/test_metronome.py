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
    scan,
    shuffle_waiting_times,
)
from seti.metronome.run import (
    load_metronome_config,
    metronome_run,
    screen_catalogue,
    stage_assess,
)
from seti.metronome.vet import (
    DEFAULT_VET,
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
    infer_time_grid,
    kepler_quarter_windows,
    lattice_phase_limits,
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
           "energy_phase_p": 0.4, "wn_truncated_by_budget": False, "mission": "kepler",
           # null 3 ran and the star beat it; a record without these is a star
           # whose pool null could not be run, which is an incomplete vet
           "p_pool": 0.005, "pn_n_trials": 200, "pn_n_exceed": 0,
           "jitter_floor": 1e-6, "phase_spacing": 1e-5,
           # the period ticked 300 times inside the windows: a real recurrence
           "cycles_span": 300.0, "cycles_hit": 40,
           "grid_days": KEPLER_LC_CADENCE_DAYS, "grid_source": "measured"}
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


def _population(n: int, period_fn, **kw) -> list[dict]:
    """A scanned population of ``n`` kepler stars with given best periods."""
    return [dict(_sig_record(period=float(period_fn(i)), **kw), star_key=f"kepler:{i}",
                 mission="kepler") for i in range(n)]


def test_population_period_veto_kills_a_period_many_unrelated_stars_share():
    # a smooth background of 200 periods spread over a decade, plus 12 stars
    # all sitting within 1% of 243 d -- the shape the 2026-09-21 TESS run had
    rng = np.random.default_rng(7)
    bg = _population(200, lambda i: 10.0 ** rng.uniform(1.0, 2.6))
    spike = [dict(_sig_record(period=243.0 * (1.0 + 0.0005 * (j - 6))),
                  star_key=f"kepler:s{j}", mission="kepler") for j in range(12)]
    vetted = assign_tiers(bg + spike, {}, VET)
    by_key = {r["star_key"]: r for r in vetted}
    for j in range(12):
        r = by_key[f"kepler:s{j}"]
        assert r["first_veto"] == "population_period", (j, r["flags"])
        assert r["veto_detail"]["population_period"]["n_near"] >= 4
    # the background stars are not swept up with them
    swept = sum(1 for i in range(200)
                if by_key[f"kepler:{i}"]["first_veto"] == "population_period")
    assert swept <= 4, swept


def test_population_period_leaves_a_star_alone_at_its_period():
    rng = np.random.default_rng(11)
    bg = _population(200, lambda i: 10.0 ** rng.uniform(1.0, 2.6))
    lone = dict(_sig_record(period=4.7137), star_key="kepler:lone", mission="kepler")
    vetted = assign_tiers(bg + [lone], {}, VET)
    r = next(x for x in vetted if x["star_key"] == "kepler:lone")
    assert "population_period" not in r["flags"]
    assert r["tier"] in ("candidate", "interest", "watch")


def test_population_period_is_not_applied_to_a_population_too_small_to_measure():
    # five identical periods among five stars is not evidence of anything
    vetted = assign_tiers(_population(5, lambda i: 3.137), {}, VET)
    assert all("population_period" not in r["flags"] for r in vetted)


def test_few_cycles_veto_trips_when_the_period_barely_repeated():
    v = vet_star(_sig_record(period=243.0, cycles_span=3.0), FULL_CTX)
    assert v["first_veto"] == "few_cycles" and v["tier"] == "none"
    assert v["veto_detail"]["few_cycles"]["cycles_span"] == 3.0
    # exactly at the threshold is enough
    v = vet_star(_sig_record(cycles_span=float(DEFAULT_VET["cycles_min"])), FULL_CTX)
    assert v["tier"] == "candidate"


def test_cycle_bookkeeping_counts_opportunities_not_tick_instants():
    """cycles_hit can never exceed cycles_span, and occupancy is a fraction."""
    w = kepler_windows_short()
    t = clock_events(w, 2.5, w.starts[0] + 0.3, duty=0.5, jitter_days=0.002, seed=3)
    r = scan(t, star_windows(t, w), dict(SCAN))
    assert r.cycles_hit <= r.cycles_span
    assert 0.0 <= r.cycle_occupancy <= 1.0
    assert r.cycles_span > 20.0
    # forced onto a period comparable with the baseline, the same events have
    # only a handful of chances to repeat -- which is what few_cycles reads
    r2 = scan(t, star_windows(t, w), dict(SCAN, min_period_days=100.0,
                                          max_period_days=400.0))
    assert r2.cycles_span < 20.0


def test_window_overlap_is_not_window_containment():
    w = Windows(np.array([0.0, 10.0]), np.array([2.0, 12.0]))
    inside = w.overlaps(np.array([0.5]), np.array([1.5]))
    straddle = w.overlaps(np.array([1.5]), np.array([3.0]))
    gap = w.overlaps(np.array([4.0]), np.array([6.0]))
    assert bool(inside[0]) and bool(straddle[0]) and not bool(gap[0])
    assert not bool(w.contains(np.array([2.5]))[0])


def test_every_rejection_rule_has_a_counter():
    recs = [dict(_sig_record(period=13.7), star_key="k:1", mission="tess"),
            dict(_sig_record(period=5.5), star_key="k:2"),
            dict(_sig_record(period=0.5668), star_key="k:3"),
            dict(_sig_record(p_shuffle=0.9, gap_integer_frac=0.05), star_key="k:4"),
            dict(_sig_record(Q=0.2, jitter=0.3), star_key="k:5"),
            dict({"status": "insufficient_events", "star_key": "k:6"}),
            dict(_sig_record(p_window=0.9), star_key="k:7"),
            dict(_sig_record(p_pool=0.4), star_key="k:8"),
            dict(_sig_record(pn_n_trials=0, jitter=0.001, jitter_floor=0.001),
                 star_key="k:9"),
            dict(_sig_record(period=200.0, cycles_span=2.0), star_key="k:10")]
    # population_period needs a population: a TESS background wide of k:1's
    # 13.7 d, plus six stars piled up on one period
    rng = np.random.default_rng(3)
    recs += [dict(_sig_record(period=float(10.0 ** rng.uniform(1.6, 2.6))),
                  star_key=f"t:{i}", mission="tess") for i in range(60)]
    recs += [dict(_sig_record(period=243.0 * (1.0 + 0.0005 * (j - 3))),
                  star_key=f"t:s{j}", mission="tess") for j in range(6)]
    ctx = {"k:1": {"mission": "tess", "prot": 100.0, "variability_catalogues_reached": True},
           "k:2": {"mission": "kepler", "prot": 11.0, "variability_catalogues_reached": True},
           "k:3": {"mission": "kepler", "prot": 30.0, "variability_catalogues_reached": True,
                   "catalogued_periods": [("vsx", 0.5668, "RRAB")]},
           "k:4": FULL_CTX, "k:5": FULL_CTX, "k:7": FULL_CTX, "k:8": FULL_CTX, "k:9": FULL_CTX}
    vetted = assign_tiers(recs, ctx, VET)
    first = rejection_counters(vetted)["first_veto"]
    for rule in HARD_VETO_ORDER + ("insufficient_events", "not_significant"):
        assert first[rule] >= 1, (rule, first)
    flags = rejection_counters(vetted)["flags_raised"]
    for f in REPORT_FLAGS:
        assert f in flags


# ---------------------------------------------------------------------------
# the catalogue's own time lattice, and the null that carries it
# ---------------------------------------------------------------------------
def test_infer_time_grid_recovers_a_coarse_lattice_not_the_mission_cadence():
    """Tu+2022's TESS peak times are spaced by 0.0069 d, five 2-min cadences.

    The mission cadence in configuration says 0.0013889.  A null that snaps to
    the configured value cannot reproduce the data's lattice, so the lattice
    has to be MEASURED.
    """
    g = 0.0069
    rng = np.random.default_rng(3)
    t = 1325.0 + g * np.unique(rng.integers(0, 120000, size=4000))
    out = infer_time_grid(t)
    assert out["method"] == "lattice"
    assert abs(out["grid_days"] / g - 1.0) < 1e-6, out
    assert out["frac_on_grid"] > 0.999


def test_infer_time_grid_reports_no_lattice_for_continuous_times():
    rng = np.random.default_rng(4)
    out = infer_time_grid(np.sort(rng.uniform(0.0, 600.0, 3000)))
    assert not np.isfinite(out["grid_days"])
    assert out["method"] in ("no_lattice", "too_few_times")


def test_lattice_phase_limits_gives_the_jitter_floor_the_rounding_forces():
    """Rounding peak times to g forces rms phase jitter >= g / (P sqrt 12)."""
    g = KEPLER_LC_CADENCE_DAYS
    lim = lattice_phase_limits(0.4232741, g)
    assert abs(lim["jitter_floor"] - g / 0.4232741 / np.sqrt(12.0)) < 1e-12
    assert 0.013 < lim["jitter_floor"] < 0.015          # KIC 5879574's floor
    assert not lim["comb_coarser_than_window"]
    # a short period against a coarse TESS lattice: the comb IS coarser than
    # the +-0.05 cycle phase window, so f_in_window is quantised there
    assert lattice_phase_limits(0.1, 0.0069)["comb_coarser_than_window"]
    assert not np.isfinite(lattice_phase_limits(3.0, float("nan"))["jitter_floor"])


def test_quantisation_limited_is_report_only_and_names_the_floor():
    # jitter at the floor: flagged, but still a candidate -- the flag says the
    # tightness is the catalogue's rounding, the pool null says whether to care
    v = vet_star(_sig_record(jitter=0.001, jitter_floor=0.001), FULL_CTX)
    assert "quantisation_limited" in v["flags"] and v["tier"] == "candidate"
    assert v["veto_detail"]["quantisation_limited"]["jitter_floor"] == 0.001
    v = vet_star(_sig_record(jitter=0.005, jitter_floor=0.0001), FULL_CTX)
    assert "quantisation_limited" not in v["flags"]


def _lattice_catalogue(mission: Windows, grid: float, n_stars: int, n_each: int, seed: int):
    """Many stars whose events fall at random on ONE coarse lattice.

    Nothing here is periodic: each star's events are uniform over the lattice
    points inside the windows.  But every event of every star sits at an exact
    multiple of ``grid``, so at P = grid (and at every rational multiple of it)
    the phases are identical by construction.  This is the Tu+2022 failure
    mode with the lattice made coarse enough to see in a unit test.
    """
    rng = np.random.default_rng(seed)
    lat = np.arange(mission.starts[0], mission.stops[-1], grid)
    lat = lat[mission.contains(lat)]
    return {i: np.sort(rng.choice(lat, size=n_each, replace=False)) for i in range(n_stars)}


def test_pool_null_kills_a_lattice_artefact_that_the_window_null_calls_significant():
    mission = Windows(np.array([0.0, 220.0]), np.array([180.0, 400.0]),
                      cadence_days=KEPLER_LC_CADENCE_DAYS, label="two_blocks")
    grid = 0.7
    cat = _lattice_catalogue(mission, grid, n_stars=12, n_each=40, seed=7)
    t = cat[0]
    pool = np.concatenate([v for k, v in cat.items() if k != 0])
    w = star_windows(t, mission)
    rec = analyze_star(t, w, None, dict(SCAN), dict(NULL), np.random.default_rng(5),
                       pool_times=pool)
    # The window null snaps to the CONFIGURED cadence (30 min), so it cannot
    # place an event on the 0.7-d lattice: the star looks like a perfect clock.
    assert rec["status"] == "scanned"
    assert rec["jitter"] < 1e-6 and rec["Q"] > 0.999
    assert rec["p_window"] < 0.05, rec["p_window"]
    # The pool null draws from the same lattice and is not impressed.
    assert rec["pool_null_computed"] and rec["pn_n_trials"] > 0
    assert rec["p_pool"] > 0.05, rec["p_pool"]
    v = vet_star(dict(rec, fdr_significant=True, fdr_watch=True), FULL_CTX)
    assert v["first_veto"] == "pool_null_explains", (v["first_veto"], v["flags"])
    assert v["tier"] == "none"


def _clock_on_lattice(mission: Windows, grid: float, period: float, t0: float) -> np.ndarray:
    k = np.arange(0, int(mission.span / period) + 2)
    t = np.round((t0 + k * period) / grid) * grid
    return np.unique(t[mission.contains(t)])


def test_pool_null_does_not_kill_a_real_clock_on_the_same_lattice():
    """The pool is the control: a genuine clock beats it as it beats null 1."""
    mission = Windows(np.array([0.0, 220.0]), np.array([180.0, 400.0]),
                      cadence_days=KEPLER_LC_CADENCE_DAYS, label="two_blocks")
    grid = 0.05
    cat = _lattice_catalogue(mission, grid, n_stars=12, n_each=60, seed=8)
    pool = np.concatenate(list(cat.values()))
    P = 5.23                          # NOT a multiple of the lattice step
    t = _clock_on_lattice(mission, grid, P, 3.5)
    assert len(t) >= 40
    w = star_windows(t, mission)
    rec = analyze_star(t, w, None, dict(SCAN), dict(NULL), np.random.default_rng(6),
                       pool_times=pool)
    assert abs(rec["period"] / P - 1.0) < 1e-3, rec["period"]
    assert rec["pool_null_computed"] and rec["p_pool"] <= 0.02, rec["p_pool"]
    v = vet_star(dict(rec, fdr_significant=True, fdr_watch=True), FULL_CTX)
    assert "pool_null_explains" not in v["flags"]
    assert v["tier"] in ("candidate", "interest"), (v["tier"], v["flags"])


def test_pool_null_that_could_not_run_is_flagged_and_blocks_the_candidate_tier():
    v = vet_star(_sig_record(pn_n_trials=0, p_pool=float("nan")), FULL_CTX)
    assert "pool_null_unreached" in v["flags"]
    assert v["first_veto"] is None and v["tier"] == "interest"


def test_screen_measures_the_grid_and_runs_the_pool_null():
    """The screen stage measures the lattice, and the measurement propagates:
    into the window model, into the shortest period the scan will entertain
    (>= n_cadences_min lattice steps), and into the pool null."""
    mission = Windows(np.array([0.0, 220.0]), np.array([180.0, 400.0]),
                      cadence_days=KEPLER_LC_CADENCE_DAYS, label="two_blocks")
    grid = 0.05
    cat = _lattice_catalogue(mission, grid, n_stars=10, n_each=60, seed=9)
    rows = [(str(k), v) for k, v in cat.items()]
    rows.append(("clock", _clock_on_lattice(mission, grid, 5.23, 3.5)))
    ev = pd.DataFrame({"star_id": np.concatenate([[k] * len(v) for k, v in rows]),
                       "t_peak": np.concatenate([v for _, v in rows])})
    conf = load_metronome_config()
    conf = dict(conf, scan=dict(conf["scan"], **SCAN), null=dict(conf["null"], **NULL))
    recs, rep = screen_catalogue(ev, "synthetic", "kepler", conf)
    assert rep["time_grid"]["cadence_source"] == "measured"
    assert abs(rep["time_grid"]["cadence_used_days"] / grid - 1.0) < 1e-6
    assert rep["mission_windows"]["cadence_days"] == rep["time_grid"]["cadence_used_days"]
    assert rep["n_pool_null_computed"] >= 1
    assert all(np.isfinite(r.get("grid_days", np.nan)) for r in recs)
    assert all(np.isfinite(r.get("jitter_floor", np.nan)) for r in recs
               if r.get("status") == "scanned")
    clock = [r for r in recs if r["star_id"] == "clock"][0]
    assert abs(clock["period"] / 5.23 - 1.0) < 1e-3
    assert clock["pool_null_computed"] and clock["p_pool"] <= 0.02


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


def test_positions_fall_back_to_the_acquired_star_tables():
    """The TIC/KIC round trip reached 45% of the 2026-09-21 shortlist, and the
    stars it missed could not have the periodic-variable veto applied at all.
    The flare catalogues' own star tables carry positions and are already on
    disk, so they fill the gap; the round trip's answer still wins."""
    from seti.metronome.run import _fill_positions_from_rotation

    pos = pd.DataFrame({"star_id": ["1"], "ra": [10.0], "dec": [1.0]})
    rot = pd.DataFrame({"star_id": ["1", "2", "3", "4"], "prot": [1.0, 2.0, 3.0, 4.0],
                        "ra": [99.0, 20.0, 30.0, np.nan],
                        "dec": [99.0, 2.0, 3.0, np.nan]})
    out = _fill_positions_from_rotation(pos, ["1", "2", "3", "4"], rot)
    assert list(out.columns) == ["star_id", "ra", "dec"]
    assert set(out["star_id"]) == {"1", "2", "3"}          # "4" has no position
    assert float(out.loc[out["star_id"] == "1", "ra"].iloc[0]) == 10.0   # round trip wins
    assert float(out.loc[out["star_id"] == "2", "ra"].iloc[0]) == 20.0
    # a rotation table without positions changes nothing, and neither does an
    # empty round trip with no rotation table
    assert len(_fill_positions_from_rotation(pos, ["1", "2"], rot[["star_id", "prot"]])) == 1
    empty = pd.DataFrame(columns=["star_id", "ra", "dec"])
    assert len(_fill_positions_from_rotation(empty, ["2"], pd.DataFrame())) == 0
    assert len(_fill_positions_from_rotation(empty, ["2", "3"], rot)) == 2


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


# ---------------------------------------------------------------------------
# The VizieR route ladder: TAP -> a second TAP host -> ASU (non-TAP) -> astroquery
#
# 2026-09-13: TAPVizieR answered 503 to five attempts over each of https and
# http, and two channels reported NO_DATA_REACHED for an infrastructure reason.
# Everything below is offline: every route takes an injectable fetch/query
# callable, so a dead TAP and a live ASU can be scripted exactly.
# ---------------------------------------------------------------------------
ASU_TSV = "\n".join([
    "#RESOURCE=yCat_J/ApJS/241/29",
    "#Name: J/ApJS/241/29/table4",
    "#Title: Flares",
    "#INFO: -out.max=100000",
    "#",
    '#Column\t"KIC"\t()\tKepler Input Catalog identifier\t[ucd=meta.id]',
    "#Column\tTpeak\t(d)\tFlare peak time\t[ucd=time.epoch]",
    "#Column\tEbol\t(erg)\tBolometric energy",
    "#",
    '"KIC"\tTpeak\tEbol',
    "\td\terg",
    "--------\t--------\t--------",
    "1234567\t120.5\t1.0e34",
    "1234567\t123.5\t2.0e34",
    "7654321\t130.5\t",
])

ASU_META = "\n".join([
    "#RESOURCE=yCat_J/ApJS/241/29",
    "#Name: J/ApJS/241/29",
    "#Title: Kepler flare catalogue (Yang+, 2019)",
    "#Table\tJ_ApJS_241_29_table4:",
    "#Name: J/ApJS/241/29/table4",
    "#Title: Flare list",
    "#Column\tKIC\t()\tKepler Input Catalog identifier",
    "#Column\tTpeak\t(d)\tFlare peak time",
    "#Column\tEbol\t(erg)\tBolometric energy",
    "#Table\tJ_ApJS_241_29_refs:",
    "#Name: J/ApJS/241/29/refs",
    "#Title: References",
    "#Column\tRef\t()\tReference code",
])

README = "\n".join([
    "J/ApJS/241/29    Kepler flare catalogue (Yang+, 2019)",
    "================================================================================",
    "File Summary:",
    "--------------------------------------------------------------------------------",
    " FileName      Lrecl  Records   Explanations",
    "--------------------------------------------------------------------------------",
    "table4.dat       80   162262   Flare list",
    "--------------------------------------------------------------------------------",
    "Byte-by-byte Description of file: table4.dat",
    "--------------------------------------------------------------------------------",
    "   Bytes Format Units   Label     Explanations",
    "--------------------------------------------------------------------------------",
    "   1-  8  I8    ---     KIC       Kepler Input Catalog identifier",
    "  10- 20  F11.5 d       Tpeak     Flare peak time",
    "  22- 30  E9.2  erg     Ebol      Bolometric energy",
])


def _asu_web(meta: str = ASU_META, rows: str = ASU_TSV, readme: str = README):
    """A VizieR that answers over ASU but not over TAP; records every URL."""
    seen: list[str] = []

    def fetch(url: str) -> str:
        seen.append(url)
        if "-meta.all" in url:
            if meta is None:
                raise RuntimeError("503 Server Error: Service Unavailable")
            return meta
        if url.endswith("/ReadMe"):
            if readme is None:
                raise RuntimeError("404 Not Found")
            return readme
        if rows is None:
            raise RuntimeError("503 Server Error: Service Unavailable")
        return rows

    fetch.urls = seen
    return fetch


def _tap_503(tried: list[str]):
    def tap_fn(adql: str, endpoint: str):
        tried.append(endpoint)
        raise RuntimeError(f"DALServiceError: 503 Server Error: Service Unavailable "
                           f"for url: {endpoint}/sync")
    return tap_fn


def test_asu_tsv_parser_reads_metadata_units_and_a_quoted_column_header():
    df = acq.parse_asu_tsv(ASU_TSV)
    assert list(df.columns) == ["KIC", "Tpeak", "Ebol"]        # the quotes are stripped
    assert len(df) == 3 and df.attrs["asu_errors"] == []
    assert list(df["KIC"]) == [1234567, 1234567, 7654321]
    assert df["Tpeak"].tolist() == [120.5, 123.5, 130.5]
    assert np.isnan(df["Ebol"].iloc[2]) and df["Ebol"].iloc[0] == 1.0e34
    # a body with only metadata is zero rows, and an error page is not rows
    empty = acq.parse_asu_tsv("#Name: x\n#***** nothing found\n")
    assert not len(empty) and empty.attrs["asu_errors"]
    # no dash rule, no unit line: the first line is still the header
    plain = acq.parse_asu_tsv("#Name: x\nA\tB\n1\t2\n")
    assert list(plain.columns) == ["A", "B"] and len(plain) == 1


def test_asu_meta_and_readme_are_the_non_tap_table_existence_check():
    acq.reset_route_state()
    web = _asu_web()
    df, attempts = acq.asu_catalogue_tables("J/ApJS/241/29", fetch_fn=web)
    assert list(df["table_name"]) == ["J/ApJS/241/29/table4", "J/ApJS/241/29/refs"]
    assert list(df.iloc[0]["columns"]) == ["KIC", "Tpeak", "Ebol"]
    assert df.iloc[0]["units"]["Tpeak"] == "d"
    assert attempts[0]["route"] == "asu_tsv" and attempts[0]["status"] == "OK"
    assert "-meta.all" in web.urls[0] and "-source=J/ApJS/241/29" in web.urls[0]
    cols, _ = acq.asu_table_columns("J/ApJS/241/29/table4", fetch_fn=web)
    assert list(cols["column_name"]) == ["KIC", "Tpeak", "Ebol"]
    assert list(cols["unit"]) == ["", "d", "erg"]
    # the ReadMe is the backstop when even the metadata form is down
    df2, att2 = acq.asu_catalogue_tables("J/ApJS/241/29", fetch_fn=_asu_web(meta=None))
    assert list(df2["table_name"]) == ["J/ApJS/241/29/table4"]
    assert df2.iloc[0]["n_rows"] == 162262
    assert list(df2.iloc[0]["columns"]) == ["KIC", "Tpeak", "Ebol"]
    assert att2[-1]["route"] == "readme"


def test_route_ladder_falls_through_tap_503_to_asu_and_says_which_route_served():
    acq.reset_route_state()
    tried: list[str] = []
    web = _asu_web()
    res = acq.vizier_table("J/ApJS/241/29/table4", columns=["KIC", "Tpeak"], max_rows=500,
                           tap_fn=_tap_503(tried), tap_urls=acq.VIZIER_TAP_MIRRORS[:2],
                           fetch_fn=web)
    assert tried == list(acq.VIZIER_TAP_MIRRORS[:2])          # TAP 503, second host 503
    assert res.route == "asu_tsv" and res.status == "OK" and len(res.rows) == 3
    assert res.endpoint == acq.VIZIER_ASU
    assert [a["route"] for a in res.attempts] == ["tap", "tap", "asu_tsv"]
    assert all("503" in a["error"] for a in res.errors)
    assert "-out.max=500" in web.urls[-1] and "-out=KIC" in web.urls[-1]
    assert "-out.form=TSV" in web.urls[-1]
    assert acq.route_log_summary()["served_by"] == "asu_tsv"


def test_astroquery_is_the_last_route_when_tap_and_asu_are_both_down():
    acq.reset_route_state()
    called = {}

    def vizier_fn(cat, cols, limit):
        called.update({"cat": cat, "cols": cols, "limit": limit})
        return pd.DataFrame({"KIC": [1], "Tpeak": [2.0]})

    res = acq.vizier_table("J/ApJS/241/29/table4", columns=["KIC"], tap_fn=_tap_503([]),
                           tap_urls=acq.VIZIER_TAP_MIRRORS[:1],
                           fetch_fn=_asu_web(meta=None, rows=None, readme=None),
                           vizier_fn=vizier_fn)
    assert res.route == "astroquery" and len(res.rows) == 1
    assert called["cat"] == "J/ApJS/241/29/table4"
    assert {a["route"] for a in res.attempts} == {"tap", "asu_tsv", "astroquery"}


def test_every_route_failing_names_every_endpoint_and_never_invents_rows():
    acq.reset_route_state()
    tried: list[str] = []
    log = acq.AcquisitionLog()

    def dead_vizier(cat, cols, limit):
        raise RuntimeError("astroquery: connection refused")

    res = acq.vizier_table("J/ApJS/241/29/table4", tap_fn=_tap_503(tried), fetch_fn=_asu_web(
        meta=None, rows=None, readme=None), vizier_fn=dead_vizier, log=log)
    assert res.route == "none" and res.status == "QUERY_FAILED" and not len(res.rows)
    named = " ".join(f"{a['endpoint']} {a.get('error', '')}" for a in res.errors)
    for endpoint in acq.VIZIER_TAP_MIRRORS:
        assert endpoint in named, endpoint
    for endpoint in acq.VIZIER_ASU_MIRRORS:
        assert endpoint in named, endpoint
    assert "astroquery.vizier.Vizier" in named
    assert "503" in named and "connection refused" in named
    assert log.as_dict()["any_query_failed"] and log.stages[-1]["route"] == "none"


def test_adql_translation_is_narrow_and_says_what_it_cannot_do():
    t = acq.translate_adql('SELECT TOP 100 "Freq", "Species" FROM "J/ApJ/787/112/table2"')
    assert t == {"kind": "rows", "table": "J/ApJ/787/112/table2",
                 "columns": ["Freq", "Species"], "limit": 100, "constraints": {},
                 "row_slice": None}
    t = acq.translate_adql("SELECT TOP 60 table_name, description FROM TAP_SCHEMA.tables "
                           "WHERE table_name LIKE '%J/ApJS/241/29%'")
    assert t["kind"] == "tables" and t["pattern"] == "J/ApJS/241/29"
    t = acq.translate_adql("SELECT TOP 2000 column_name FROM TAP_SCHEMA.columns "
                           "WHERE table_name = 'J/ApJS/241/29/table4' OR table_name = '\"x\"'")
    assert t["kind"] == "columns" and t["table"] == "J/ApJS/241/29/table4"
    # recno chunking: ASU returns rows in recno order, so the window is exact
    t = acq.translate_adql('SELECT "KIC" FROM "t" WHERE recno BETWEEN 51 AND 100')
    assert t["limit"] == 100 and t["row_slice"] == (50, 100)
    for bad, why in [('SELECT COUNT(*) AS n FROM "t"', "COUNT"),
                     ("SELECT TOP 5 table_name FROM TAP_SCHEMA.tables WHERE description "
                      "LIKE '%flare%'", "description"),
                     ('SELECT "a" FROM "t" WHERE "x" > 3 AND "y" < 4', "WHERE")]:
        with pytest.raises(acq.AdqlNotTranslatable) as err:
            acq.translate_adql(bad)
        assert why in str(err.value)


def test_discovery_degrades_to_the_non_tap_existence_check_when_tap_is_down():
    acq.reset_route_state()
    web = _asu_web()
    dead = _FakeTAP("fail")
    tabs = acq.list_tables("J/ApJS/241/29", query_fn=dead, fetch_fn=web)
    assert list(tabs["table_name"]) == ["J/ApJS/241/29/table4", "J/ApJS/241/29/refs"]
    assert tabs.attrs["route"] == "asu_tsv"
    cols = acq.table_columns("J/ApJS/241/29/table4", query_fn=dead, fetch_fn=web)
    assert cols == ["KIC", "Tpeak", "Ebol"]
    # ... and the roles resolve off the non-TAP columns exactly as off TAP's
    assert acq.resolve_event_columns(cols)["star_id"] == "KIC"
    # a catalogue that does not exist is a DIFFERENT fact from an outage: the
    # non-TAP route says so by failing with every endpoint named
    with pytest.raises(acq.VizierRouteError) as err:
        acq.list_tables("J/Nope/1/1", query_fn=dead,
                        fetch_fn=_asu_web(meta=None, rows=None, readme=None))
    assert "non-TAP" in str(err.value) and acq.VIZIER_ASU in str(err.value)


# ---------------------------------------------------------------------------
# the circuit breaker: an intermittent endpoint must not become a total outage
# ---------------------------------------------------------------------------
def test_circuit_breaker_half_opens_after_a_short_cooldown_and_closes_on_success(monkeypatch):
    """ARC run 34792280736: TAPVizieR answered the FIRST discovery query and
    503'd the next two, and a five-minute cooldown then skipped all five
    remaining catalogues --- an intermittent service turned into a total
    outage.  The cooldown is now short, expiring it half-opens the endpoint,
    and any success closes it again."""
    acq.reset_route_state()
    monkeypatch.setattr(acq._time, "sleep", lambda *_a, **_k: None)
    ep = "https://tap.example/TAPVizieR/tap"
    assert acq.breaker_state(ep) == acq.STATE_CLOSED and acq._circuit_open(ep) == ""
    acq._circuit_fail(ep, "503")
    assert acq._circuit_open(ep) == ""                 # one failure is not an outage
    acq._circuit_fail(ep, "503")
    assert acq.breaker_state(ep) == acq.STATE_OPEN
    skip = acq._circuit_open(ep)
    assert skip.startswith("skipped:") and "retried after" in skip
    assert acq._ROUTE_FAIL_COOLDOWN_S <= 60.0          # seconds, not minutes
    # the cooldown expires -> ONE live attempt (half open), not a skip
    acq._ROUTE_FAILS[ep]["at"] -= acq._ROUTE_FAIL_COOLDOWN_S + 1.0
    assert acq._circuit_open(ep) == ""
    assert acq.breaker_state(ep) == acq.STATE_HALF_OPEN
    acq._circuit_ok(ep)                                # ... and a success closes it
    assert acq.breaker_state(ep) == acq.STATE_CLOSED and acq._circuit_open(ep) == ""
    moves = [t["transition"] for t in acq.BREAKER_LOG if t["endpoint"] == ep]
    assert moves == ["closed->open", "open->half_open", "half_open->closed"]
    # ... and the transitions ride along in the acquisition log, so a dispatch
    # that was starved by the breaker says so in its artefact
    blk = acq.AcquisitionLog().as_dict()["breaker"]
    assert blk["transitions"] == acq.BREAKER_LOG and blk["cooldown_s"] <= 60.0
    assert blk["endpoints"][ep]["state"] == acq.STATE_CLOSED
    acq.reset_route_state()


def test_an_endpoint_that_answered_once_is_never_skipped_without_a_fresh_attempt(monkeypatch):
    """The rule the starved run broke: a host that has SERVED in this process
    is intermittent, not down.  It is never skipped --- every later query gets
    a genuine attempt --- only degraded to one cheap try instead of the
    retry ladder."""
    acq.reset_route_state()
    monkeypatch.setattr(acq._time, "sleep", lambda *_a, **_k: None)
    ep = acq.VIZIER_TAP_MIRRORS[0]
    tried: list[str] = []
    state = {"up": True}

    def tap_fn(adql: str, endpoint: str):
        tried.append(endpoint)
        if not state["up"]:
            raise RuntimeError(f"DALServiceError: 503 Server Error for url: {endpoint}/sync")
        return pd.DataFrame({"table_name": ["J/ApJ/906/72/table2"], "description": ["x"]})

    df = acq.tap_query("SELECT 1", url=[ep], retries=1, tap_fn=tap_fn, allow_non_tap=False)
    assert len(df) == 1 and df.attrs["route"] == "tap" and df.attrs["endpoint"] == ep
    state["up"] = False
    for _ in range(2):
        with pytest.raises(acq.VizierRouteError):
            acq.tap_query("SELECT 2", url=[ep], retries=1, tap_fn=tap_fn, allow_non_tap=False)
    assert acq.breaker_state(ep) == acq.STATE_OPEN and len(tried) == 3
    # every later query STILL reaches the endpoint -- once each, cheaply, even
    # with retries=3: no catalogue is skipped because an earlier one 503'd
    for i in range(3):
        with pytest.raises(acq.VizierRouteError) as err:
            acq.tap_query(f"SELECT {i + 3}", url=[ep], retries=3, tap_fn=tap_fn,
                          allow_non_tap=False)
        assert "skipped:" not in str(err.value)
        assert len(tried) == 4 + i, "an endpoint that has served was skipped outright"
    # when the host comes back the very next query is served, and the success
    # closes the breaker
    state["up"] = True
    assert len(acq.tap_query("SELECT 6", url=[ep], retries=1, tap_fn=tap_fn,
                             allow_non_tap=False)) == 1
    assert acq.breaker_state(ep) == acq.STATE_CLOSED and len(tried) == 7
    # a host that has NEVER answered is still skipped after its cooldown opens
    # the breaker -- that is the cost the breaker exists to save
    dead = "https://never.example/TAPVizieR/tap"
    for _ in range(2):
        acq._circuit_fail(dead, "503")
    assert acq._circuit_open(dead).startswith("skipped:")
    acq.reset_route_state()


# ---------------------------------------------------------------------------
# the non-TAP existence check for an EXACT table id (ARC's six catalogues)
# ---------------------------------------------------------------------------
OKAMOTO_ASU_ROWS = "\n".join([
    "#RESOURCE=yCat_J/ApJ/906/72",
    "#Name: J/ApJ/906/72/table2",
    "#Title: Superflares on solar-type stars (Okamoto+, 2021)",
    "#Column\tKIC\t()\tKepler Input Catalog identifier",
    "#Column\tProt\t(d)\tRotation period",
    "#Column\tTeff\t(K)\tEffective temperature",
    "#Column\tRstar\t(Rsun)\tStellar radius",
    "#Column\tEflare\t(erg)\tBolometric flare energy",
    "#Column\tTpeak\t(d)\tFlare peak time",
    "KIC\tProt\tTeff\tRstar\tEflare\tTpeak",
    "\td\tK\tRsun\terg\td",
    "--------\t--------\t--------\t--------\t--------\t--------",
    "1234567\t12.0\t5800\t1.0\t1.0e35\t120.5",
])


def _asu_rows_only(body: str = OKAMOTO_ASU_ROWS):
    """A VizieR whose ``-meta.all`` form does NOT resolve --- the state ARC's
    2026-09-14 dispatch actually met --- but whose ``-source=`` rows do."""
    seen: list[str] = []

    def fetch(url: str, timeout: float = 180.0) -> str:
        seen.append(url)
        if "-meta.all" in url or url.endswith("/ReadMe"):
            raise RuntimeError("404 Client Error: Not Found")
        return body

    fetch.urls = seen
    return fetch


def test_a_table_id_resolves_over_asu_by_asking_for_one_row_when_tap_is_down():
    """``TAP_SCHEMA.tables ~ 'J/ApJ/906/72/table2'`` with no TAP: ASU has no
    TAP_SCHEMA, so the question becomes "give me one row of
    ``-source=J/ApJ/906/72/table2``" and the answer's header is the columns."""
    acq.reset_route_state()
    web = _asu_rows_only()
    dead = _FakeTAP("fail")
    tabs = acq.list_tables("J/ApJ/906/72/table2", query_fn=dead, fetch_fn=web)
    assert list(tabs["table_name"]) == ["J/ApJ/906/72/table2"]
    assert tabs.attrs["route"] == "asu_tsv"
    assert list(tabs.iloc[0]["columns"]) == ["KIC", "Prot", "Teff", "Rstar", "Eflare", "Tpeak"]
    assert "-source=J/ApJ/906/72/table2" in web.urls[0] and "-out.max=1" in web.urls[0]
    assert "-meta.all" not in web.urls[0]      # the row form is tried FIRST for a table id
    cols = acq.table_columns("J/ApJ/906/72/table2", query_fn=dead, fetch_fn=web)
    assert cols == ["KIC", "Prot", "Teff", "Rstar", "Eflare", "Tpeak"]
    rec, attempts = acq.asu_table_exists("J/ApJ/906/72/table2", fetch_fn=web)
    assert rec["units"]["Eflare"] == "erg" and rec["description"].startswith("Superflares")
    assert attempts[-1]["route"] == "asu_tsv" and attempts[-1]["status"] == "OK"
    acq.reset_route_state()


def test_a_bare_catalogue_id_also_falls_back_to_the_one_row_check():
    """Two of ARC's seeds are catalogue ids (``J/ApJS/209/5``,
    ``J/ApJS/255/17``).  When ``-meta.all`` and the ReadMe are both down, ASU
    still serves the catalogue's table, and the body names which table it is."""
    acq.reset_route_state()
    body = OKAMOTO_ASU_ROWS.replace("#Name: J/ApJ/906/72/table2",
                                    "#Name: J/ApJS/209/5/flares")
    web = _asu_rows_only(body)
    tabs = acq.list_tables("J/ApJS/209/5", query_fn=_FakeTAP("fail"), fetch_fn=web)
    assert list(tabs["table_name"]) == ["J/ApJS/209/5/flares"]
    assert tabs.attrs["route"] == "asu_tsv"
    # the metadata form is tried FIRST for a bare catalogue id (it lists every
    # table); the one-row request is the fallback, not the first move
    assert "-meta.all" in web.urls[0] and "-out.max=1" in web.urls[-1]
    acq.reset_route_state()


def test_an_asu_body_naming_another_catalogue_is_not_existence_evidence():
    """A mirror that answers with somebody else's table has not shown that
    ours exists: that is recorded as a failure, not read as rows."""
    acq.reset_route_state()
    wrong = _asu_rows_only(ASU_TSV)            # names J/ApJS/241/29/table4
    rec, attempts = acq.asu_table_exists("J/ApJ/906/72/table2", fetch_fn=wrong)
    assert rec is None
    assert all(a["status"] == "QUERY_FAILED" for a in attempts)
    assert "J/ApJS/241/29/table4" in attempts[0]["error"]
    with pytest.raises(acq.VizierRouteError):
        acq.list_tables("J/ApJ/906/72/table2", query_fn=_FakeTAP("fail"), fetch_fn=wrong)
    acq.reset_route_state()


def test_count_rows_is_unknown_rather_than_failed_without_tap():
    acq.reset_route_state()
    err = acq.VizierRouteError("every endpoint 503", asu_supported=False)

    def dead(adql: str):
        raise err

    assert acq.count_rows("J/ApJS/241/29/table4", query_fn=dead) is None
    with pytest.raises(acq.VizierRouteError):
        acq.count_rows("J/ApJS/241/29/table4",
                       query_fn=lambda a: (_ for _ in ()).throw(
                           acq.VizierRouteError("x", asu_supported=True)))


def test_config_route_ladder_matches_the_module_constants():
    conf = load_metronome_config()
    routes = conf["vizier_routes"]
    assert tuple(routes["tap_mirrors"]) == acq.VIZIER_TAP_MIRRORS
    assert tuple(routes["asu_tsv"]) == acq.VIZIER_ASU_MIRRORS
    assert routes["readme"] == acq.VIZIER_README
    assert acq.VIZIER_TAP == acq.VIZIER_TAP_MIRRORS[0]
    txt = Path("config/metronome.yaml").read_text()
    # every asserted-but-unreached host is marked, per the repository's rule
    assert txt.count("# verify:") >= 2 and "ASSERTED" in txt
    # the breaker the code runs is the breaker the config documents
    brk = routes["circuit_breaker"]
    assert brk["fail_limit"] == acq._ROUTE_FAIL_LIMIT
    assert brk["cooldown_s"] == acq._ROUTE_FAIL_COOLDOWN_S <= 60.0
    assert brk["never_skip_after_success"] is True
    # ... and the ASU spellings it marks `verify` are the ones asu_url emits
    p = routes["asu_params"]
    url = acq.asu_url("J/ApJ/906/72/table2", max_rows=int(p["existence_rows"]))
    assert f"{p['source']}=J/ApJ/906/72/table2" in url
    assert f"{p['max_rows']}={p['existence_rows']}" in url
    assert p["all_columns"] in url and p["form"] in url
    assert p["metadata"] in acq.asu_url("J/ApJ/906/72", meta=True)


def test_asu_tsv_survives_duplicate_column_names():
    """VizieR repeats a column name in some tables; a duplicate label made
    ``out[c]`` a DataFrame and to_numeric raised
    TypeError('arg must be a list, tuple, 1-d array, or Series').

    That is the error McQuillan+2014 and Guenther+2020 returned in ARC's run
    34792280736 while every other catalogue in the same run parsed cleanly.
    """
    from seti.metronome.acquire import parse_asu_tsv

    text = "\n".join([
        "#Column\tKIC\t(I8)\tKepler id",
        "#Column\tProt\t(F6.3)\tRotation period",
        "KIC\tProt\tProt\te_Prot",
        "--------\t------\t------\t------",
        "757076\t13.34\t13.34\t0.05",
        "757099\t8.12\t8.12\t0.03",
    ])
    df = parse_asu_tsv(text)
    assert len(df) == 2
    assert list(df.columns) == ["KIC", "Prot", "Prot__1", "e_Prot"]
    assert df.attrs["asu_renamed_columns"] == ["Prot -> Prot__1"]
    assert float(df["Prot"].iloc[0]) == 13.34
    assert float(df["e_Prot"].iloc[1]) == 0.03


def test_asu_tsv_prefers_the_column_metadata_over_a_misplaced_header_row():
    """``#Column`` names are the service's own and must beat the positional guess.

    ARC's re-run resolved ZERO roles on McQuillan+2014, Tu+2022 and
    Guenther+2020 -- tables that plainly carry a KIC column -- because the
    header row was picked by offset from the dashed rule and landed on the
    units line when the preamble had a different number of lines.
    """
    from seti.metronome.acquire import parse_asu_tsv

    text = "\n".join([
        "#Column\tKIC\t(I8)\tKepler identifier",
        "#Column\tProt\t(F7.3)\tRotation period",
        "#Column\tRper\t(F8.1)\tPhotometric range",
        "",
        "days\tppm\tmag",                 # a units line where the header is expected
        "----\t----\t----",
        "757076\t13.34\t2345.0",
        "757099\t8.12\t1180.0",
    ])
    df = parse_asu_tsv(text)
    assert list(df.columns) == ["KIC", "Prot", "Rper"], df.columns.tolist()
    assert len(df) == 2
    assert int(df["KIC"].iloc[0]) == 757076
    assert float(df["Rper"].iloc[1]) == 1180.0


def test_asu_tsv_keeps_a_good_header_row_when_metadata_disagrees_in_width():
    """A ``#Column`` block that does not match the table width must not win."""
    from seti.metronome.acquire import parse_asu_tsv

    text = "\n".join([
        "#Column\tKIC\t(I8)\tKepler identifier",       # only one meta column ...
        "KIC\tProt\te_Prot",                            # ... but three real ones
        "--------\t------\t------",
        "757076\t13.34\t0.05",
    ])
    df = parse_asu_tsv(text)
    assert list(df.columns) == ["KIC", "Prot", "e_Prot"]


def test_asu_tsv_never_takes_a_dashed_rule_as_the_header():
    """A rule is never a header.

    ARC's run 34796629576 came back with columns literally named
    "--------", "--------__1", ... for McQuillan+2014, Okamoto+2021, Tu+2022
    and Guenther+2020 — the amplitude sources the channel needs — because the
    header offset landed on a second dashed rule and every role then failed.
    """
    from seti.metronome.acquire import parse_asu_tsv

    text = "\n".join([
        "#Column\tKIC\t(I8)\tKepler identifier",
        "",
        "KIC\tProt\tRper",
        "--------\t------\t------",          # a first rule ...
        "--------\t------\t------",          # ... and a second, which the
        "757076\t13.34\t2345.0",             # offset used to land on
    ])
    df = parse_asu_tsv(text)
    assert not any(str(c).startswith("-") for c in df.columns), df.columns.tolist()
    assert list(df.columns) == ["KIC", "Prot", "Rper"]
    assert int(df["KIC"].iloc[0]) == 757076


def test_asu_tsv_rule_detection_accepts_a_one_character_rule():
    """VizieR writes a one-character rule under a one-character column.

    Requiring two dashes meant the rule line was not recognised at all, the
    parser fell back to body[0] — the rule itself — and McQuillan+2014,
    Tu+2022 and Guenther+2020 came back with columns named '-', '--', '----'.
    """
    from seti.metronome.acquire import parse_asu_tsv

    text = "\n".join([
        "#Column\tKIC\t(I8)\tKepler identifier",
        "KIC\tf\tProt",
        "--------\t-\t------",          # a one-character rule cell
        "757076\tA\t13.34",
    ])
    df = parse_asu_tsv(text)
    assert list(df.columns) == ["KIC", "f", "Prot"], df.columns.tolist()
    assert len(df) == 1 and int(df["KIC"].iloc[0]) == 757076


# ---------------------------------------------------------------------------
# The constraint ladder: which ASU parameter zeroed the query
# ---------------------------------------------------------------------------
_LADDER_ROWS = "\n".join([
    "#Column\tAllWISE\t\tdesignation",
    "AllWISE\tW1mag",
    "-------\t-----",
    "J000000.00+000000.0\t12.3",
])


def _ladder_fetch(zero_on: str):
    """An ASU server that serves rows until ``zero_on`` appears in the URL."""
    seen: list[str] = []

    def fetch(url: str, **_kw):
        seen.append(url)
        if zero_on and zero_on in url:
            return "#Name: II/328/allwise\n#Title: t\n"      # metadata only: zero rows
        return _LADDER_ROWS

    fetch.seen = seen                                        # type: ignore[attr-defined]
    return fetch


def test_constraint_ladder_names_the_parameter_that_zeroed_the_query():
    """IGNITION's probe reported zero rows for a bare one-degree AllWISE cone.

    A catalogue of 750 million sources has no empty degree, so a zero-row
    response is a request defect; the ladder is what says which parameter.
    """
    fetch = _ladder_fetch("-c.rd=")
    steps = acq.asu_constraint_ladder(
        "II/328/allwise", columns=["AllWISE", "W1mag"],
        constraints={"-c": "266+65", "-c.rd": "1", "-c.eq": "J2000"},
        max_rows=5, fetch_fn=fetch, bases=["https://example.invalid/viz-bin/asu-tsv"])
    labels = [s["label"] for s in steps]
    assert labels == ["bare", "columns", "+-c", "+-c.rd", "+-c.eq"], labels
    assert [s["status"] for s in steps[:3]] == [acq.STATUS_OK] * 3
    verdict = acq.ladder_verdict(steps)
    assert verdict["status"] == "CONSTRAINT_ZEROED_THE_QUERY"
    assert verdict["culprit"] == "+-c.rd" and verdict["last_ok"] == "+-c"
    # the server's own words are kept at the step that died
    assert "II/328/allwise" in (verdict.get("body_head") or "")


def test_constraint_ladder_calls_a_dead_bare_request_what_it_is():
    """Nothing from ``-source`` alone indicts the catalogue id, not any cut."""
    steps = acq.asu_constraint_ladder(
        "II/328/allwise_typo", columns=["AllWISE"], constraints={"-c": "266+65"},
        fetch_fn=_ladder_fetch("-source"),
        bases=["https://example.invalid/viz-bin/asu-tsv"])
    verdict = acq.ladder_verdict(steps)
    assert verdict["status"] == "BARE_REQUEST_EMPTY"
    assert "catalogue id" in verdict["note"]


def test_constraint_ladder_says_so_when_every_step_returns_rows():
    steps = acq.asu_constraint_ladder(
        "II/328/allwise", columns=["AllWISE"], constraints={"-c": "266+65"},
        fetch_fn=_ladder_fetch(""), bases=["https://example.invalid/viz-bin/asu-tsv"])
    assert acq.ladder_verdict(steps)["status"] == "ALL_STEPS_RETURNED_ROWS"
    assert acq.ladder_verdict([])["status"] == "NOT_RUN"


def test_zero_row_asu_response_records_the_body_head():
    """``QUERY_RETURNED_ZERO_ROWS`` with nothing else recorded is undiagnosable."""
    def fetch(url: str, **_kw):
        return "#Name: II/328/allwise\n#Title: nothing here\n"

    df, attempts = acq.asu_rows("II/328/allwise", fetch_fn=fetch,
                                bases=["https://example.invalid/viz-bin/asu-tsv"])
    assert not len(df)
    assert attempts[-1]["status"] == acq.STATUS_ZERO
    assert "nothing here" in attempts[-1]["body_head"]
    assert "\\n" in attempts[-1]["body_head"]        # newlines are made visible


# ---------------------------------------------------------------------------
# A catalogue has TABLES: the ReadMe is the inventory -meta.all is not
# ---------------------------------------------------------------------------
_SHIBAYAMA_README = """\
J/ApJS/209/5   Superflares on solar-type stars from Kepler  (Shibayama+, 2013)
================================================================================
File Summary:
--------------------------------------------------------------------------------
 FileName    Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe          80        .   This file
stars.dat      102      279   Properties of the superflare stars
flares.dat      66     1547   Superflares detected on solar-type stars
--------------------------------------------------------------------------------

Byte-by-byte Description of file: flares.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1-  8  I8    ---     KIC       Kepler identifier
  10- 20  F11.5 d       Tpeak     Flare peak time
  22- 32  E11.4 erg     Ebol      Bolometric flare energy
--------------------------------------------------------------------------------
"""

_META_ONE_TABLE = "\n".join([
    "#Name: J/ApJS/209/5",
    "#Title: Superflares on solar-type stars",
    "#Table  J_ApJS_209_5_stars:",
    "#Name: J/ApJS/209/5/stars",
    "#Title: Properties of the superflare stars",
    "#Column\tKIC\t(I8)\tKepler identifier",
    "#Column\tProt\t(d)\tRotation period",
])


def test_a_bare_catalogue_listing_is_completed_from_its_readme():
    """``-meta.all`` named ONE table for J/ApJS/209/5, losing the flares half.

    ARC then reported Shibayama+2013 as a catalogue exposing no usable flares
    table at all, when the catalogue plainly has one.  The ReadMe's File
    Summary is the catalogue's own inventory and settles it.
    """
    seen: list[str] = []

    def fetch(url: str, **_kw):
        seen.append(url)
        if "ReadMe" in url:
            return _SHIBAYAMA_README
        if "-meta.all" in url:
            return _META_ONE_TABLE
        raise RuntimeError(f"unexpected request: {url}")

    df, attempts = acq.asu_catalogue_tables("J/ApJS/209/5", fetch_fn=fetch)
    names = sorted(df["table_name"])
    assert names == ["J/ApJS/209/5/flares", "J/ApJS/209/5/stars"], names
    flares = df[df["table_name"] == "J/ApJS/209/5/flares"].iloc[0]
    # the ReadMe's byte-by-byte block gives the real labels, so the table can be
    # SCORED without another round trip
    assert "Ebol" in list(flares["columns"]) and "KIC" in list(flares["columns"])
    assert int(flares["n_rows"]) == 1547
    assert any(a["route"] == acq.ROUTE_README for a in attempts)


def test_an_exact_table_id_does_not_pay_for_the_readme():
    """A full table name resolves on the one-row check; no extra round trip."""
    def fetch(url: str, **_kw):
        assert "ReadMe" not in url, "the ReadMe is for bare catalogue ids only"
        return "\n".join(["#Name: J/ApJS/209/5/flares", "KIC\tEbol",
                          "---\t----", "757076\t1.0e34"])

    df, _att = acq.asu_catalogue_tables("J/ApJS/209/5/flares", fetch_fn=fetch)
    assert list(df["table_name"]) == ["J/ApJS/209/5/flares"]


def test_readme_tables_records_a_dead_readme_rather_than_raising():
    def fetch(url: str, **_kw):
        raise RuntimeError("404")

    tables, attempts = acq.asu_readme_tables("J/ApJS/209/5", fetch_fn=fetch)
    assert tables == {}
    assert attempts[0]["status"] == acq.STATUS_FAILED and "404" in attempts[0]["error"]


# ---------------------------------------------------------------------------
# The position ladder: which spelling of a cone VizieR will honour
# ---------------------------------------------------------------------------
def test_position_spellings_put_a_decimal_point_and_a_sign_on_the_coordinates():
    """"266 65" is not a sky position to VizieR; "266.000000 +65.000000" is."""
    spellings = dict(acq.asu_position_spellings(266.0, 65.0, 1.0))
    assert spellings["decimal_signed"]["-c"] == "266.000000 +65.000000"
    assert spellings["decimal_signed"]["-c.rd"] == "1.000000"
    assert spellings["split_ra_dec"]["-c.ra"] == "266.000000"
    assert spellings["radius_arcmin"]["-c.rm"] == "60.0000"
    south = dict(acq.asu_position_spellings(10.0, -24.5, 0.5))
    assert south["decimal_signed"]["-c"] == "10.000000 -24.500000"


def test_position_ladder_reports_the_first_spelling_that_returns_rows():
    """Which spelling works is measured, not asserted."""
    def fetch(url: str, **_kw):
        if "%2B65" in url:                      # a real plus sign survived the wire
            return "\n".join(["#Name: I/355/gaiadr3", "Source\tGmag",
                              "------\t----", "123\t12.0"])
        return "#Name:\n#Title:\n"              # VizieR's empty resource

    rep = acq.asu_position_ladder("I/355/gaiadr3", ra=266.0, dec=65.0, radius_deg=1.0,
                                  columns=["Source", "Gmag"], fetch_fn=fetch,
                                  bases=["https://example.invalid/viz-bin/asu-tsv"])
    assert rep["working"] == "decimal_signed"
    assert rep["verdict"]["status"] == "SPELLING_FOUND"
    assert len(rep["steps"]) == 1               # it stops at the first that works


def test_position_ladder_refuses_to_call_a_dead_cone_an_empty_sky():
    def fetch(url: str, **_kw):
        return "#Name:\n#Title:\n"

    rep = acq.asu_position_ladder("II/328/allwise", ra=266.0, dec=65.0, radius_deg=1.0,
                                  fetch_fn=fetch,
                                  bases=["https://example.invalid/viz-bin/asu-tsv"])
    assert rep["working"] is None
    assert rep["verdict"]["status"] == "NO_CONE_SPELLING_WORKS"
    assert "no empty sky may be inferred" in rep["verdict"]["note"]
    assert len(rep["verdict"]["tried"]) == 6


def test_a_constraint_value_never_puts_a_bare_plus_on_the_wire():
    """A literal '+' in a query string is a SPACE; a sign must be percent-encoded."""
    url = acq.asu_url("I/355/gaiadr3", constraints={"-c": "266.0 +65.0"}, max_rows=5)
    assert "%20%2B65.0" in url and "-c=266.0+65.0" not in url


# ---------------------------------------------------------------------------
# TAPVizieR quotes COLUMN names too (ARC run 35040024670)
# ---------------------------------------------------------------------------
def test_tap_schema_column_names_come_back_unquoted():
    """ARC reached ZERO rows on all five flare catalogues, and nothing about the
    sky had changed.

    TAPVizieR came back up, so discovery switched from the ASU route (bare
    header cells) to TAP, and TAP_SCHEMA.columns returns '"KIC"' with the
    double quotes inside the string. Every role pattern stopped matching and
    the scoreboard recorded 'rejected: no star_id' for a table whose second
    column is plainly KIC.
    """
    def query_fn(adql: str):
        assert "TAP_SCHEMA.columns" in adql
        return pd.DataFrame({"column_name": ['"recno"', '"KIC"', '"Teff"', '"BP-RP"', 'Prot']})

    cols = acq.table_columns("J/ApJ/906/72/table2", query_fn=query_fn)
    assert cols == ["recno", "KIC", "Teff", "BP-RP", "Prot"], cols
    # the roles now resolve, which is the whole point
    roles = acq.resolve_columns(cols, {"star_id": [r"^kic$"], "teff": [r"^teff$"]})
    assert roles == {"star_id": "KIC", "teff": "Teff"}
    # and the name that goes into the query is the bare one, which fetch_table
    # re-quotes itself — a name left quoted here becomes ""KIC"" in the SELECT
    assert not any(c.startswith('"') for c in cols)


def test_a_quoted_column_name_from_any_other_path_still_resolves():
    """Defence in depth: the canonicaliser strips quotes as well."""
    roles = acq.resolve_columns(['"KIC"', '"Prot"'], {"star_id": [r"^kic$"], "prot": [r"^prot$"]})
    assert roles == {"star_id": '"KIC"', "prot": '"Prot"'}


def test_known_columns_from_a_non_tap_listing_are_unquoted_too():
    def query_fn(adql: str):
        raise RuntimeError("503")

    cols = acq.table_columns("J/X/1/t", query_fn=query_fn, known=['"KIC"', "E"])
    assert cols == ["KIC", "E"]


# ---------------------------------------------------------------------------
# The MEASURED VizieR column names (results/arc/probe.json), which the first
# dispatch never saw: Yang & Liu's Begin/End, Okamoto's and Shibayama's Date,
# Tu+2022's PDate.  A pattern set that cannot name these cannot scan a star.
# ---------------------------------------------------------------------------
def test_measured_vizier_flare_tables_resolve_their_time_columns():
    yang = ["recno", "KIC", "Q", "Begin", "End", "logE"]
    score, r, why = acq.score_event_table(yang)
    assert score > 0 and why == "usable", (score, r, why)
    assert r["star_id"] == "KIC" and r["t_start"] == "Begin" and r["t_end"] == "End"
    assert r["energy"] == "logE" and r["sector"] == "Q" and "t_peak" not in r
    tu = ["recno", "ID", "Sector", "Source", "PeakNum", "Label", "PDate", "Lum", "Energy",
          "Duration", "File"]
    score, r, _ = acq.score_event_table(tu)
    assert score > 0 and r["star_id"] == "ID" and r["t_peak"] == "PDate"
    assert r["sector"] == "Sector" and r["energy"] == "Energy" and r["duration"] == "Duration"
    okamoto = ["recno", "KIC", "Teff", "e_Teff", "Rstar", "E_Rstar", "e_Rstar", "Prot", "e_Prot",
               "Amp", "Kemag", "Aspot", "Date", "Dur", "E", "Flag", "_RA", "_DE"]
    score, r, _ = acq.score_event_table(okamoto)
    assert score > 0 and r["t_peak"] == "Date" and r["prot"] == "Prot" and r["energy"] == "E"
    assert r["ra"] == "_RA" and r["dec"] == "_DE"
    shibayama = ["recno", "KIC", "BVAmp", "Date", "FAmp", "Dur", "E"]
    score, r, _ = acq.score_event_table(shibayama)
    assert score > 0 and r["t_peak"] == "Date"
    gunther = ["recno", "TIC", "Sec", "Outbst", "Flare", "tpeak", "e_tpeak", "E_tpeak", "Amp",
               "Ebol", "Tmag", "Teff", "Rad", "logg", "SpT", "Prot-t", "Prot-k", "Prot", "Dup"]
    score, r, _ = acq.score_event_table(gunther)
    assert score > 0 and r["t_peak"] == "tpeak" and r["star_id"] == "TIC"
    assert r["energy"] == "Ebol" and r["prot"] == "Prot" and r["sector"] == "Sec"
    # Davenport 2016 table1 is a per-star FFD summary, not an event list
    dav = ["recno", "KIC", "g-i", "Mass", "Prot", "Nfl", "Nfl68", "Lfl/Lkp", "e_Lfl/Lkp",
           "alpha", "beta", "M14", "C16", "_RA", "_DE"]
    score, _, why = acq.score_event_table(dav)
    assert score == 0 and "t_peak/t_start" in why
    # the rotation tables join on the same id
    r = acq.resolve_columns(["recno", "ID", "Tmag", "Teff", "logg", "Rad", "Per", "Rvar",
                             "Flag", "Nflares", "_RA", "_DE"])
    assert r["star_id"] == "ID" and r["prot"] == "Per"


def test_clean_star_id_gives_one_spelling_per_star():
    assert acq.clean_star_id("KIC 757099") == "757099"
    assert acq.clean_star_id(757099.0) == "757099"
    assert acq.clean_star_id(" 757099 ") == "757099"
    assert acq.clean_star_id("TIC 1234") == "1234"
    ev = pd.DataFrame({"KIC": ["KIC 1", "1.0", "2"], "Begin": [100.0, 103.0, 110.0],
                       "End": [100.1, 103.1, 110.2], "Q": [2, 2, 3]})
    d = acq.discover_event_table("k", "J/ApJS/241/29", query_fn=_FakeTAP("ok", ev))
    df = acq.fetch_events(d, query_fn=_FakeTAP("ok", ev))
    assert list(df["star_id"]) == ["1", "1", "2"]
    assert (df["t_peak_source"] == "t_start").all() and list(df["sector"]) == [2, 2, 3]


def test_probe_records_what_the_catalogue_says_its_time_column_is(tmp_path):
    ev = pd.DataFrame({"KIC": ["1"] * 10, "Tpeak": np.linspace(200.0, 300.0, 10)})

    class TAP(_FakeTAP):
        def __call__(self, adql):
            if "TAP_SCHEMA.columns" in adql and "description" in adql:
                return pd.DataFrame({"column_name": ['"KIC"', '"Tpeak"'], "unit": ["", "d"],
                                     "description": ["KIC number", "Flare peak (BJD-2454833)"]})
            return super().__call__(adql)

    from seti.metronome.run import stage_probe
    conf = load_metronome_config()
    conf["catalogues"] = {"k": {"mission": "kepler", "preferred": "J/ApJS/241/29",
                                "keywords": [], "enabled": True}}
    rep = stage_probe(conf, tmp_path, query_fn=TAP("ok", ev))
    d = rep["catalogues"]["k"]
    assert d["time_column"] == "Tpeak"
    assert "2454833" in d["time_column_meta"]["description"] and d["time_column_meta"]["unit"] == "d"
    assert d["time_system_guess"] == "BKJD"


# ---------------------------------------------------------------------------
# sector spans: the windows a catalogue's own sector column states
# ---------------------------------------------------------------------------
def test_sector_spans_cut_what_the_density_model_bridges():
    from seti.metronome.run import build_mission_windows
    from seti.metronome.windows import intersect_windows, windows_from_sectors

    rng = np.random.default_rng(3)
    # two TESS sectors 40 days apart, 30 sparse events each: too few for the
    # density model to see the boundary
    t1 = np.sort(rng.uniform(1325.0, 1352.0, 30))
    t2 = np.sort(rng.uniform(1392.0, 1419.0, 30))
    ev = pd.DataFrame({"t_peak": np.concatenate([t1, t2]),
                       "sector": [1] * 30 + [3] * 30, "star_id": ["a"] * 60})
    spans = windows_from_sectors(ev["t_peak"], ev["sector"], cadence_days=0.0014)
    assert spans.n == 2 and spans.starts[0] == t1.min() and spans.stops[1] == t2.max()
    conf = load_metronome_config()
    conf["windows"]["min_events_for_data_driven"] = 10 ** 9   # force the sparse route
    w = build_mission_windows(ev, "tess", conf)
    assert w.n == 2 and w.total < 60.0, w.as_dict()
    assert not w.contains(np.array([1370.0]))[0]
    # the sparse density model alone must not invent gaps INSIDE a sector: at
    # 60 events over 94 days the conservative rate demands a ~30-day empty run
    dens = windows_from_events(ev["t_peak"].to_numpy(), cadence_days=0.0014)
    assert dens.n == 2, dens.as_dict()
    # without the sector column the same events never claim LESS observed
    # time than the spans allow
    w0 = build_mission_windows(ev.drop(columns=["sector"]), "tess", conf)
    assert w0.total >= w.total
    # intersection geometry
    a = Windows(np.array([0.0, 10.0]), np.array([5.0, 20.0]))
    b = Windows(np.array([3.0, 12.0, 18.0]), np.array([4.0, 15.0, 30.0]))
    i = intersect_windows(a, b)
    assert np.allclose(i.starts, [3.0, 12.0, 18.0]) and np.allclose(i.stops, [4.0, 15.0, 20.0])


# ---------------------------------------------------------------------------
# flare re-detection on a light curve, and the clock test on what it finds
# ---------------------------------------------------------------------------
def _synthetic_kepler_lightcurve(flare_times, *, seed=5, amp=0.012, noise=1e-3,
                                 decay_days=0.05):
    """Four quarters of 30-min cadence with monthly 1-day downlinks, a slow
    sinusoidal variability and flares at ``flare_times`` with a one-cadence
    linear rise and an exponential decay (the classical template shape)."""
    rng = np.random.default_rng(seed)
    cad = KEPLER_LC_CADENCE_DAYS
    segs = []
    for q, (s, e) in enumerate([(170.0, 258.0), (261.0, 349.0), (353.0, 442.0), (444.0, 538.0)]):
        t = np.arange(s, e, cad)
        # monthly downlink gaps of a day
        for g in np.arange(s + 30.0, e, 30.0):
            t = t[(t < g) | (t > g + 1.0)]
        f = 1.0 + 0.004 * np.sin(2 * np.pi * t / 11.3) + rng.normal(0.0, noise, len(t))
        for tf in flare_times:
            dt = t - tf
            sel = (dt >= -cad) & (dt < 0.4)
            shape = np.where(dt[sel] < 0, 1.0 + dt[sel] / cad,
                             np.exp(-np.clip(dt[sel], 0, None) / decay_days))
            f[sel] += amp * shape
        segs.append({"sector": q + 2, "quarter": q + 2, "exptime_s": cad * 86400.0,
                     "cadence": "long", "time": t, "flux": f * 1e4,
                     "flux_err": np.full(len(t), noise * 1e4), "n_points": len(t)})
    return segs


def test_redetect_finds_injected_flares_and_the_clock_on_them():
    from seti.metronome.redetect import (
        find_flares,
        lightcurve_windows,
        redetect_star,
        stitch_segments,
    )

    rng = np.random.default_rng(9)
    P = 3.137
    ticks = 171.3 + P * np.arange(0, 118)
    ticks = ticks[rng.random(len(ticks)) < 0.6]
    ticks = ticks + rng.normal(0.0, 0.002, size=len(ticks))
    randoms = np.sort(rng.uniform(172.0, 536.0, 25))
    injected = np.sort(np.concatenate([ticks, randoms]))
    segs = _synthetic_kepler_lightcurve(injected)
    t, f, meta = stitch_segments(segs)
    assert meta["n_segments"] == 4 and abs(meta["cadence_days"] - KEPLER_LC_CADENCE_DAYS) < 1e-6
    fl = find_flares(t, f, cadence_days=meta["cadence_days"])
    # every injected flare that fell in observed time is recovered within a cadence
    w = lightcurve_windows(t, cadence_days=meta["cadence_days"])
    assert w.n >= 4                                   # quarters split at monthly gaps
    inj = injected[w.contains(injected)]
    near = np.array([np.min(np.abs(fl["t_peak"].to_numpy() - x)) for x in inj])
    assert np.mean(near <= 0.05) >= 0.9, np.mean(near <= 0.05)
    # and few spurious ones
    spurious = np.array([np.min(np.abs(inj - x)) > 0.1 for x in fl["t_peak"]])
    assert spurious.sum() <= 0.1 * len(inj), spurious.sum()
    rec = redetect_star(segs, {}, scan_conf=SCAN, null_conf=NULL, rng=np.random.default_rng(1),
                        catalogue_times=injected, period_catalogue=P)
    assert rec["status"] == "scanned" and rec["catalogue_recovery_frac"] >= 0.9
    assert abs(rec["rd_period"] / P - 1.0) < 2e-3, rec["rd_period"]
    assert rec["period_agrees_with_catalogue"] and rec["clock_in_lightcurve"]
    # 25 of 83 events are random flares: the rms Q is ~0.5 and it is the CORE
    # route (f_in_window ~0.75, core jitter ~0.002) that confirms the clock
    assert rec["confirms_catalogue_clock"] and rec["rd_f_in_window"] >= 0.6
    assert rec["rd_jitter_core"] <= 0.05 and rec["rd_n_core"] >= 8


def test_a_sinusoid_does_not_manufacture_flares_but_a_sharp_pulsator_does():
    """THE confounder of the re-detection stage, measured rather than assumed.

    A running median over ``detrend_window_days`` cannot flatten a photometric
    oscillation of comparable period, so the worry is that its maxima are
    re-detected as a train of "flares" at exactly the photometric period ---
    with the phase stability of the oscillation rather than of any flare
    mechanism.  KIC 5879574's re-detected period (0.4233 d) sits in that band.

    Measured here, on light curves that contain NO flares at all:

    * a pure SINUSOID does not do it.  The residual is a sinusoid too, and the
      MAD sigma is set by that same sinusoid, so its maxima sit at ~1.4 sigma,
      under the 2.5/3.5 sigma gates.  The detector is safe against this case
      and the assertion records it.
    * a SHARP-PEAKED periodic signal does.  A pulsator's sawtooth maximum, a
      heartbeat brightening, an ellipsoidal or contact-binary light curve with
      a narrow maximum: the bulk of the cycle sets the MAD, the narrow peak
      clears it every cycle, and out comes a perfect clock at P_phot.

    The second case is what the photometric veto is for.
    """
    from seti.metronome.redetect import (
        find_flares,
        photometric_period,
        redetect_star,
        stitch_segments,
    )

    P_phot = 0.4233
    # (a) sinusoid: no manufactured flares
    segs = _synthetic_kepler_lightcurve([], seed=17, noise=4e-4)
    for s in segs:
        s["flux"] = s["flux"] * (1.0 + 0.010 * np.sin(2 * np.pi * s["time"] / P_phot))
    t, f, meta = stitch_segments(segs)
    assert len(find_flares(t, f, cadence_days=meta["cadence_days"])) == 0

    # (b) sharp periodic maximum: a manufactured clock
    segs = _synthetic_kepler_lightcurve([], seed=17, noise=4e-4)
    for s in segs:
        ph = (s["time"] / P_phot) % 1.0
        s["flux"] = s["flux"] * (1.0 + 0.040 * np.exp(-0.5 * (np.minimum(ph, 1.0 - ph)
                                                              / 0.05) ** 2))
    t, f, meta = stitch_segments(segs)
    fl = find_flares(t, f, cadence_days=meta["cadence_days"])
    assert len(fl) >= 50, len(fl)
    ph = photometric_period(t, f, cadence_days=meta["cadence_days"])
    assert abs(ph["period"] / P_phot - 1.0) < 0.02, ph
    rec = redetect_star(segs, {}, scan_conf=dict(SCAN, min_period_days=0.3),
                        null_conf=NULL, rng=np.random.default_rng(3),
                        catalogue_times=np.array([]), period_catalogue=P_phot)
    assert rec["status"] == "scanned"
    assert abs(rec["rd_period"] / P_phot - 1.0) < 0.02, rec["rd_period"]
    assert rec["period_is_photometric"], (rec["rd_period"], rec["phot_period"])
    assert not rec["clock_in_lightcurve"] and not rec["confirms_catalogue_clock"]
    # the shape statistics agree with the verdict: a symmetric maximum
    assert rec["rd_rise_frac_median"] > 0.35, rec["rd_rise_frac_median"]


def test_photometric_veto_leaves_a_real_flare_clock_alone():
    """The control: real flares on a star whose photometric period is nowhere
    near the clock must still confirm."""
    from seti.metronome.redetect import redetect_star

    rng = np.random.default_rng(19)
    P = 3.137                             # the light curve's own sinusoid is 11.3 d
    ticks = 171.3 + P * np.arange(0, 118)
    ticks = ticks[rng.random(len(ticks)) < 0.6] + rng.normal(0.0, 0.002, size=None)
    segs = _synthetic_kepler_lightcurve(np.sort(ticks))
    rec = redetect_star(segs, {}, scan_conf=SCAN, null_conf=NULL, rng=np.random.default_rng(4),
                        catalogue_times=np.sort(ticks), period_catalogue=P)
    assert not rec["period_is_photometric"], (rec["rd_period"], rec["phot_period"])
    assert rec["clock_in_lightcurve"] and rec["confirms_catalogue_clock"]
    assert rec["rd_duty_cycle"] < 0.1, rec["rd_duty_cycle"]


def test_redetect_reports_no_clock_for_random_flares():
    from seti.metronome.redetect import redetect_star

    rng = np.random.default_rng(11)
    injected = np.sort(rng.uniform(172.0, 536.0, 60))
    segs = _synthetic_kepler_lightcurve(injected, seed=12)
    rec = redetect_star(segs, {}, scan_conf=SCAN, null_conf=NULL, rng=np.random.default_rng(2),
                        catalogue_times=injected, period_catalogue=3.137)
    assert rec["status"] == "scanned" and rec["n_flares"] >= 40
    assert not rec["clock_in_lightcurve"] and not rec["confirms_catalogue_clock"]


def test_redetect_stage_degrades_honestly_and_selects_the_shortlist(tmp_path):
    from seti.metronome.redetect import select_targets, stage_redetect

    vetted = pd.DataFrame({
        "star_key": ["kepler:1", "kepler:2", "kepler:3", "tess:9"],
        "star_id": ["1", "2", "3", "9"], "mission": ["kepler"] * 3 + ["tess"],
        "catalogue": ["k"] * 3 + ["t"], "tier": ["none", "watch", "interest", "none"],
        "n_events": [500, 20, 30, 9], "period": [1.0, 2.0, 3.0, 4.0],
        "p_window": [0.5, 0.01, 0.001, 0.9], "Q": [0.1, 0.7, 0.9, 0.2],
        "jitter": [0.3, 0.1, 0.02, 0.3], "status": ["scanned"] * 4})
    conf = load_metronome_config()
    conf["redetect"].update({"top_by_events": 1, "retries": 1, "budget_s": 60.0,
                             "per_target_budget_s": 10.0})
    tg = select_targets(vetted, conf["redetect"])
    assert [t["star_key"] for t in tg] == ["kepler:3", "kepler:2", "kepler:1", "tess:9"]
    assert tg[0]["why"] == "tier:interest" and tg[2]["why"] == "top_events:k"
    vetted.to_csv(tmp_path / "stars_vetted.csv", index=False)

    def dead(*a, **k):
        raise RuntimeError("MAST unreachable")

    rep = stage_redetect(conf, tmp_path, lc_fn=dead, kepler_lc_fn=dead)
    assert rep["verdict"] == "NO_DATA_REACHED" and rep["n_fetched"] == 0
    assert rep["n_fetch_failed"] == 4 and rep["n_confirms_catalogue_clock"] == 0
    assert (tmp_path / "redetect.json").exists()

    def empty(*a, **k):
        return []

    rep = stage_redetect(conf, tmp_path, lc_fn=empty, kepler_lc_fn=empty)
    assert rep["verdict"] == "NO_LIGHTCURVES_FOUND" and rep["n_no_lightcurve"] == 4
    assert all(t["status"] == "NO_LIGHTCURVE" for t in rep["targets"])

    # a real light curve for one star, nothing for the rest: the per-star
    # record says which, and the stage verdict is a count
    rng = np.random.default_rng(21)
    injected = np.sort(rng.uniform(172.0, 536.0, 40))

    def one(kepid, **k):
        return _synthetic_kepler_lightcurve(injected, seed=22) if str(kepid) == "2" else []

    conf["scan"].update(SCAN)
    conf["null"].update(NULL)
    rep = stage_redetect(conf, tmp_path, lc_fn=empty, kepler_lc_fn=one, max_stars=3)
    assert rep["n_targets"] == 3 and rep["n_fetched"] == 1 and rep["n_scanned"] == 1
    assert rep["verdict"] == "REDETECT_CONFIRMS_NONE"
    got = {t["star_key"]: t for t in rep["targets"]}
    assert got["kepler:2"]["status"] == "scanned" and got["kepler:3"]["status"] == "NO_LIGHTCURVE"
    assert (tmp_path / "stars_redetect.csv").exists()


def test_clock_with_a_natural_flare_background_reaches_strict_quality():
    """A beacon on a star that also flares: 30% of the events are random.

    MEASURED before the core route existed: Q = 0.49, jitter = 0.13 --- failing
    even the watch thresholds at p ~ 1e-64 --- and gap_integer_frac 0.56, which
    would have tripped bursty_random.  The core numbers say what the ticks do.
    """
    w = kepler_windows_short()
    P = 3.137
    ticks = clock_events(w, P, w.starts[0] + 1.3, duty=0.6, jitter_days=0.002, seed=31)
    noise = poisson_events(w, int(0.45 * len(ticks)), seed=32)
    t = np.sort(np.concatenate([ticks, noise]))
    pool = np.concatenate([poisson_events(w, 200, seed=400 + i) for i in range(4)])
    r = analyze(t, w, seed=33, pool_times=pool)
    assert r["status"] == "scanned" and abs(r["period"] / P - 1.0) < 2e-3, r["period"]
    # the ±0.15 core still admits ~30% x 0.3 of the background, so its jitter
    # is not the ticks' 0.002 but stays well under the strict 0.05
    assert r["f_in_window"] >= 0.6 and r["jitter_core"] <= 0.05 and r["n_core"] >= 8
    assert r["p_window"] < 1e-3
    assert r["gap_integer_frac_core"] >= 0.6
    r["fdr_significant"] = r["fdr_watch"] = True
    v = vet_star(r, {"mission": "kepler", "prot": 11.0, "catalogued_periods": [],
                     "variability_catalogues_reached": True})
    assert v["tier"] == "candidate", (v, r["Q"], r["jitter"], r["gap_integer_frac"])
    assert "bursty_random" not in v["flags"] and "jitter_too_large" not in v["flags"]
    # and the rotator is still nothing like a clock on the core route
    rot = rotation_events(w, 2.5, 200, seed=34)
    rr = analyze(rot, w, seed=35)
    assert rr["f_in_window"] < 0.4 or rr["jitter_core"] > 0.05, (rr["f_in_window"],
                                                                 rr["jitter_core"])


def test_redetect_is_never_part_of_stage_all():
    from seti.metronome.run import STAGE_REDETECT, STAGES

    assert STAGE_REDETECT not in STAGES


def test_reconcile_demotes_a_candidate_whose_clock_is_its_photometric_period(tmp_path):
    """The light curve has the last word on the headline verdict."""
    from seti.metronome.redetect import reconcile_summary

    (tmp_path / "summary.json").write_text(json.dumps({
        "verdict": "CLOCK_CANDIDATES_PENDING_VET", "n_candidates": 1, "n_interest": 1,
        "funnel": {"stars_candidate": 1, "stars_interest": 1}}))
    (tmp_path / "candidates.json").write_text(json.dumps({
        "candidates": [{"star_key": "kepler:1", "tier": "candidate", "flags": ""},
                       {"star_key": "kepler:2", "tier": "interest", "flags": "p_extrapolated"}],
        "watch": [{"star_key": "kepler:3", "tier": "watch", "flags": ""}]}))
    records = [
        {"star_key": "kepler:1", "status": "scanned", "period_is_photometric": True,
         "phot_period": 0.4233, "rd_period": 0.4233, "n_flares": 74},
        {"star_key": "kepler:2", "status": "scanned", "period_is_photometric": False,
         "confirms_catalogue_clock": True, "rd_period": 3.137, "n_flares": 40},
    ]
    res = reconcile_summary(tmp_path, records, verdict="REDETECT_CONFIRMS_1")
    assert res["status"] == "OK" and res["demoted"] == ["kepler:1"]

    cj = json.loads((tmp_path / "candidates.json").read_text())
    c1, c2 = cj["candidates"]
    assert c1["tier"] == "none" and c1["first_veto"] == "photometric_oscillation"
    assert "photometric_oscillation" in c1["flags"]
    assert c1["redetect"]["phot_period"] == 0.4233
    # the star the photometry did not explain keeps its tier and its evidence
    assert c2["tier"] == "interest" and c2["redetect"]["confirms_catalogue_clock"]
    # a star the budget never reached says so rather than being scored
    assert cj["watch"][0]["redetect"] == {"status": "not_attempted"}

    sj = json.loads((tmp_path / "summary.json").read_text())
    assert sj["n_candidates"] == 0 and sj["n_interest"] == 1
    assert sj["funnel"]["stars_demoted_photometric"] == 1
    assert "REDETECT_DEMOTED_1_PHOTOMETRIC" in sj["verdict"]
    assert sj["redetect"]["verdict"] == "REDETECT_CONFIRMS_1"


def test_reconcile_never_promotes_and_survives_a_missing_summary(tmp_path):
    from seti.metronome.redetect import reconcile_summary

    # no summary at all: a recorded fact, not a crash
    assert reconcile_summary(tmp_path, [])["status"] == "NO_SUMMARY"
    (tmp_path / "summary.json").write_text(json.dumps({"verdict": "NO_CLOCK_CANDIDATES",
                                                       "n_candidates": 0}))
    (tmp_path / "candidates.json").write_text(json.dumps({
        "candidates": [], "watch": [{"star_key": "k:9", "tier": "watch", "flags": ""}]}))
    res = reconcile_summary(tmp_path, [{"star_key": "k:9", "status": "scanned",
                                        "confirms_catalogue_clock": True,
                                        "clock_in_lightcurve": True}])
    assert res["demoted"] == []
    cj = json.loads((tmp_path / "candidates.json").read_text())
    assert cj["watch"][0]["tier"] == "watch"        # confirmation does not promote
    sj = json.loads((tmp_path / "summary.json").read_text())
    assert sj["verdict"] == "NO_CLOCK_CANDIDATES"   # and does not rewrite the verdict
