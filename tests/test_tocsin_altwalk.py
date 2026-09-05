"""Offline tests for the alternative-feed walk: the job deadline, the walk state,
the prior-backed reduction, the star-night denominator, and the concurrent ATLAS
scheduler (``seti.tocsin.altwalk``).

Each test pins one of the three failures the first four weeks of
``tocsin-altfeeds`` runs produced --- a job killed by the runner, the same six
stars walked every week, and a ledger whose denominator dropped every target
after the first --- so that none of them can come back quietly.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from seti.tocsin import altfeeds as A
from seti.tocsin import altwalk as W

MJD0 = 60000.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _targets(ids=("a", "b", "c"), mag=14.0, pm=0.0):
    n = len(ids)
    return pd.DataFrame({
        "source_id": list(ids),
        "ra": [10.0 + i for i in range(n)], "dec": [2.0] * n,
        "pmra": [pm] * n, "pmdec": [0.0] * n,
        "g_sdss_mag": [mag + 0.1 * i for i in range(n)],
        "r_sdss_mag": [mag - 0.2 + 0.1 * i for i in range(n)],
        "i_sdss_mag": [mag - 0.4 + 0.1 * i for i in range(n)],
        "z_sdss_mag": [mag - 0.5 + 0.1 * i for i in range(n)],
        "u_sdss_mag": [mag + 1.0 + 0.1 * i for i in range(n)],
    })


def _atlas_lc(tid="a", n=200, t0=MJD0, band="o", noise=500.0, event_idx=None,
              event_njy=0.0, seed=1):
    rng = np.random.default_rng(seed)
    flux = rng.normal(0.0, noise, n)
    if event_idx is not None:
        flux[event_idx] = event_njy
    return A.LightCurve(target_id=tid, ra=10.0, dec=2.0, survey="atlas",
                        mjd=np.arange(n, dtype=float) + t0,
                        flux_njy=flux, flux_err_njy=np.full(n, noise),
                        band=np.array([band] * n, dtype=object))


def _atlas_text(n=60, band="o", t0=MJD0, seed=2, event_idx=None, event_ujy=0.0):
    rng = np.random.default_rng(seed)
    cols = ["MJD", "m", "dm", "uJy", "duJy", "F", "err", "chi/N", "RA", "Dec",
            "x", "y", "maj", "min", "phi", "apfit", "mag5sig", "Sky", "Obs"]
    lines = ["###" + "  ".join(cols)]
    for k in range(n):
        f = 5.0 * rng.normal()
        if event_idx is not None and k == event_idx:
            f = event_ujy
        lines.append(" ".join([f"{t0 + k:.5f}", "18.0", "0.05", f"{f:.3f}", "5.000",
                               band, "0", "1.02", "10.0", "2.0", "100", "100", "2.0",
                               "1.9", "10", "-0.1", "19.0", "20.0", f"o{k:05d}o"]))
    return "\n".join(lines) + "\n"


Q = {"o": A.Quiescent(1.0e5, 1.0e3, "atlas_reduced_images")}


# ---------------------------------------------------------------------------
# 1. the job deadline
# ---------------------------------------------------------------------------
def test_a_job_deadline_clips_a_survey_budget():
    """The 2026-09-02 failure: every survey inside its budget, the job dead."""
    env = {W.JOB_DEADLINE_ENV: "2000.0"}
    budget, note = W.effective_budget_s(9000.0, reserve_s=900.0, env=env,
                                        wall_now=1000.0, mono_now=50.0)
    assert budget == pytest.approx(100.0)        # 1000 s left, 900 reserved
    assert note and "clipped" in note and "9000" in note


def test_a_budget_that_already_fits_is_not_touched():
    env = {W.JOB_DEADLINE_ENV: "100000.0"}
    budget, note = W.effective_budget_s(9000.0, env=env, wall_now=1000.0, mono_now=50.0)
    assert budget == 9000.0 and note is None


def test_no_job_deadline_means_the_survey_budget_stands():
    budget, note = W.effective_budget_s(1234.0, env={}, wall_now=1.0, mono_now=1.0)
    assert budget == 1234.0 and note is None


def test_a_garbled_deadline_is_ignored_not_fatal():
    budget, _ = W.effective_budget_s(10.0, env={W.JOB_DEADLINE_ENV: "soon"},
                                     wall_now=1.0, mono_now=1.0)
    assert budget == 10.0


def test_a_spent_job_deadline_gives_a_zero_budget_not_a_negative_one():
    env = {W.JOB_DEADLINE_ENV: "0.0"}
    budget, _ = W.effective_budget_s(9000.0, env=env, wall_now=5000.0, mono_now=0.0)
    assert budget == 0.0


# ---------------------------------------------------------------------------
# 2. the walk state: plan, record, refresh
# ---------------------------------------------------------------------------
def test_a_fresh_state_plans_full_walks_up_to_max_new():
    st = W.WalkState(survey="atlas")
    plan = st.plan(["a", "b", "c"], survey_start_mjd=57200.0, now=61000.0, max_new=2)
    assert [p["target_id"] for p in plan.fresh] == ["a", "b"]
    assert plan.refresh == []
    assert all(p["mode"] == "full" and p["mjd_lo"] == 57200.0 for p in plan.fresh)
    # The request frontier sits behind the wall clock by the ingest lag.
    assert plan.hi_request == pytest.approx(61000.0 - W.DEFAULT_INGEST_LAG_DAYS)


def test_walked_stars_are_refreshed_first_and_not_counted_against_max_new():
    st = W.WalkState(survey="atlas")
    st.record("a", mjd_lo=57200.0, mjd_hi=60980.0, mode="full", usable=True,
              n_epochs=1000, baseline={"o": {"level": 0.0, "scatter": 500.0, "n_used": 900}})
    plan = st.plan(["a", "b", "c"], survey_start_mjd=57200.0, now=61000.0, max_new=1)
    assert [p["target_id"] for p in plan.refresh] == ["a"]
    assert plan.refresh[0]["mode"] == "refresh"
    assert plan.refresh[0]["mjd_lo"] == 60980.0            # from the high-water mark
    assert [p["target_id"] for p in plan.fresh] == ["b"]   # max_new spent on NEW stars
    # Refreshes precede full walks in the request order.
    assert [p["target_id"] for p in plan.requests] == ["a", "b"]


def test_a_star_current_to_within_the_refresh_interval_is_left_alone():
    st = W.WalkState(survey="atlas")
    st.record("a", mjd_lo=57200.0, mjd_hi=60996.0, mode="full", usable=True, n_epochs=10,
              baseline={"o": {"level": 0.0, "scatter": 1.0, "n_used": 10}})
    plan = st.plan(["a"], survey_start_mjd=57200.0, now=61000.0, max_new=5)
    assert plan.requests == []
    assert plan.skipped_current == 1
    assert any("current" in n for n in plan.notes)


def test_windows_for_one_star_are_disjoint_across_runs():
    """The property the star-night denominator rests on."""
    st = W.WalkState(survey="atlas")
    p1 = st.plan(["a"], survey_start_mjd=57200.0, now=60900.0, max_new=1)
    w1 = p1.fresh[0]
    st.record("a", mjd_lo=w1["mjd_lo"], mjd_hi=w1["mjd_hi"], mode="full", usable=True,
              n_epochs=10, baseline={"o": {"level": 0.0, "scatter": 1.0, "n_used": 10}})
    p2 = st.plan(["a"], survey_start_mjd=57200.0, now=60930.0, max_new=1)
    w2 = p2.refresh[0]
    assert w2["mjd_lo"] == w1["mjd_hi"]
    assert w2["mjd_hi"] > w2["mjd_lo"]


def test_the_archive_frontier_caps_the_recorded_window():
    """ZTF via IRSA is a data release: never mark a star screened past it."""
    st = W.WalkState(survey="ztf")
    st.observe_archive_frontier(60968.0)
    plan = st.plan(["a"], survey_start_mjd=58194.0, now=61283.0, max_new=1)
    assert plan.hi_request == 60968.0
    st.observe_archive_frontier(60900.0)              # never lowered
    assert st.archive_frontier_mjd == 60968.0
    st.observe_archive_frontier(None)
    assert st.archive_frontier_mjd == 60968.0


def test_no_refresh_is_planned_until_the_archive_advances():
    st = W.WalkState(survey="ztf")
    st.observe_archive_frontier(60968.0)
    st.record("a", mjd_lo=58194.0, mjd_hi=60968.0, mode="full", usable=True, n_epochs=500,
              baseline={"zg": {"level": 1.0e5, "scatter": 1.0e3, "n_used": 500}})
    plan = st.plan(["a"], survey_start_mjd=58194.0, now=61300.0, max_new=1)
    assert plan.requests == []                         # archive has not moved
    st.observe_archive_frontier(61100.0)               # a new release lands
    plan = st.plan(["a"], survey_start_mjd=58194.0, now=61300.0, max_new=1)
    assert len(plan.refresh) == 1
    assert plan.refresh[0]["mjd_lo"] == 60968.0
    assert plan.refresh[0]["mjd_hi"] == 61100.0


def test_an_unusable_star_is_not_refreshed_but_is_rewalked_eventually():
    st = W.WalkState(survey="atlas")
    st.record("a", mjd_lo=57200.0, mjd_hi=60900.0, mode="full", usable=False, n_epochs=3)
    plan = st.plan(["a"], survey_start_mjd=57200.0, now=60950.0, max_new=1)
    assert plan.requests == [] and plan.skipped_unusable == 1
    # Long enough later, a full walk is due again.
    st.targets["a"]["walked_utc"] = "2020-01-01T00:00:00Z"
    plan = st.plan(["a"], survey_start_mjd=57200.0, now=60950.0, max_new=1)
    assert len(plan.fresh) == 1 and plan.fresh[0]["mode"] == "rewalk"


def test_a_refresh_keeps_the_full_history_priors_and_advances_the_mark():
    st = W.WalkState(survey="atlas")
    st.record("a", mjd_lo=57200.0, mjd_hi=60900.0, mode="full", usable=True, n_epochs=1000,
              baseline={"o": {"level": 0.0, "scatter": 500.0, "n_used": 900}},
              quiescent={"o": {"flux_njy": 1.0e5, "err_njy": 1.0e3,
                               "source": "atlas_reduced_images"}},
              chi_med=1.1, elo_med=1.05)
    st.record("a", mjd_lo=60900.0, mjd_hi=60920.0, mode="refresh", usable=False,
              n_epochs=12, baseline={"o": {"level": 9.0, "scatter": 9.0, "n_used": 12}})
    rec = st.get("a")
    assert rec.mjd_hi == 60920.0
    assert rec.n_epochs == 1012 and rec.n_walks == 2
    assert rec.usable is True, "a short window cannot demote a star"
    assert rec.baseline["o"]["scatter"] == 500.0, "priors are never overwritten by a refresh"
    prior = st.prior_for("a")
    assert prior.bands["o"]["n_used"] == 900
    assert prior.quiescent["o"]["source"] == "atlas_reduced_images"
    assert prior.chi_med == 1.1


def test_the_state_round_trips_through_json(tmp_path):
    st = W.WalkState(survey="atlas")
    st.record("a", mjd_lo=57200.0, mjd_hi=60900.0, mode="full", usable=True, n_epochs=10,
              baseline={"o": {"level": float("nan"), "scatter": 1.0, "n_used": 10}})
    st.observe_archive_frontier(60950.0)
    p = tmp_path / "walked.json"
    st.save(p)
    raw = json.loads(p.read_text())
    assert raw["targets"]["a"]["baseline"]["o"]["level"] is None    # NaN -> null
    again = W.WalkState.load(p)
    assert again.get("a").mjd_hi == 60900.0
    assert again.archive_frontier_mjd == 60950.0
    assert again.n_walked() == 1 and again.n_usable() == 1
    assert W.WalkState.load(tmp_path / "missing.json", survey="ztf").survey == "ztf"


def test_a_prior_needs_a_finite_positive_scatter():
    st = W.WalkState(survey="atlas")
    st.record("a", mjd_lo=0.0, mjd_hi=1.0, mode="full", usable=True, n_epochs=1,
              baseline={"o": {"level": 0.0, "scatter": 0.0, "n_used": 5}})
    prior = st.prior_for("a")
    lc = _atlas_lc(n=8)
    red = A.reduce_lightcurve(lc, A.ATLAS, quiescent=Q, prior=prior)
    assert red.bands["o"].usable is False           # falls back to the epoch rule
    assert "fewer_than" in red.bands["o"].reason


# ---------------------------------------------------------------------------
# 3. reduction against a prior
# ---------------------------------------------------------------------------
def test_a_short_window_is_unusable_alone_but_a_set_of_trials_with_a_prior():
    lc = _atlas_lc(n=8)                              # below min_good_epochs
    alone = A.reduce_lightcurve(lc, A.ATLAS, quiescent=Q)
    assert alone.usable is False
    prior = W.Prior(bands={"o": {"level": 0.0, "scatter": 500.0, "n_used": 900}},
                    quiescent={})
    with_prior = A.reduce_lightcurve(lc, A.ATLAS, quiescent=Q, prior=prior)
    assert with_prior.usable is True
    assert len(with_prior.visit_mjds) == 8
    assert with_prior.alerts == []
    assert with_prior.baseline_from_prior is True
    assert "baseline_from_full_history_prior" in with_prior.notes


def test_a_deviant_epoch_in_a_refresh_window_is_judged_against_the_long_baseline():
    # 8 epochs, one of them at -6000 nJy: against a 500 nJy full-history scatter
    # that is a 12-sigma dip; a scatter estimated from these 8 points alone
    # would be dragged up by the very epoch it is meant to judge.
    lc = _atlas_lc(n=8, event_idx=3, event_njy=-6000.0)
    prior = W.Prior(bands={"o": {"level": 0.0, "scatter": 500.0, "n_used": 900}},
                    quiescent={})
    red = A.reduce_lightcurve(lc, A.ATLAS, quiescent=Q, prior=prior)
    assert len(red.alerts) == 1
    assert red.alerts[0].dflux_njy == pytest.approx(-6000.0)
    assert red.alerts[0].snr == pytest.approx(-12.0)


def test_the_prior_supplies_the_quiescent_flux_when_no_reduced_pass_ran():
    lc = _atlas_lc(n=8)
    prior = W.Prior(bands={"o": {"level": 0.0, "scatter": 500.0, "n_used": 900}},
                    quiescent={"o": {"flux_njy": 2.0e5, "err_njy": 2.0e3,
                                     "source": "atlas_reduced_images"}})
    red = A.reduce_lightcurve(lc, A.ATLAS, quiescent=None, prior=prior)
    assert red.usable is True
    assert red.bands["o"].quiescent_njy == 2.0e5
    assert red.bands["o"].quiescent_source == "atlas_reduced_images"


def test_without_a_prior_the_reduction_is_unchanged():
    lc = _atlas_lc(n=200, event_idx=50, event_njy=3.0e4)
    a = A.reduce_lightcurve(lc, A.ATLAS, quiescent=Q)
    b = A.reduce_lightcurve(lc, A.ATLAS, quiescent=Q, prior=None)
    assert len(a.alerts) == len(b.alerts) == 1
    assert a.visit_mjds == b.visit_mjds
    assert b.baseline_from_prior is False


def test_prior_quality_medians_replace_a_short_windows_own():
    lc = _atlas_lc(n=8)
    lc.chi_n = np.array([1.0] * 7 + [4.0])         # 4.0 is fine against the window's
    prior = W.Prior(bands={"o": {"level": 0.0, "scatter": 500.0, "n_used": 900}},
                    quiescent={}, chi_med=0.5)     # ... but an outlier against history
    red = A.reduce_lightcurve(lc, A.ATLAS, quiescent=Q, prior=prior)
    assert red.chi_med == 0.5


# ---------------------------------------------------------------------------
# 4. the denominator: trials keyed to the star-night
# ---------------------------------------------------------------------------
def _verdict_for(tid: str, targets, n=200):
    lc = _atlas_lc(tid=tid, n=n)
    red = A.reduce_lightcurve(lc, A.ATLAS, quiescent=Q)
    return A.screen_lightcurves([red], targets, A.ATLAS, A.funnel_thresholds(A.ATLAS))


def test_two_stars_walked_on_different_runs_both_count_toward_the_denominator(tmp_path):
    """The ATLAS ledger of 2026-09-02: 1337 nights, 1337 trials, six stars."""
    t = _targets(("a", "b"))
    p = tmp_path / "ledger_atlas.json"
    A.fold(_verdict_for("a", t), p, targets_n=1)
    A.fold(_verdict_for("b", t), p, targets_n=2)        # the SAME 200 nights
    led = json.loads(p.read_text())
    assert led["n_target_visits"] == 400, "the second star's trials were dropped"
    assert len(led["nights"]) == 200
    assert led["n_targets_screened"] == 2


def test_the_rubin_ledger_still_dedupes_by_night(tmp_path):
    from seti.tocsin.ledger import Ledger
    led = Ledger()
    led.add_night("n1", [], target_visits=5, targets_in_footprint=5, alerts_seen=0)
    led.add_night("n1", [], target_visits=5, targets_in_footprint=5, alerts_seen=0)
    assert led.n_target_visits == 5
    led.add_night("n1", [], target_visits=5, targets_in_footprint=5, alerts_seen=0,
                  dedupe_night=False)
    assert led.n_target_visits == 10
    assert led.nights == ["n1"]


# ---------------------------------------------------------------------------
# 5. the concurrent ATLAS scheduler
# ---------------------------------------------------------------------------
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += float(s)


class FakeAtlas:
    """A queue that finishes a task ``ticks`` polls after submission."""

    poll_s = 10.0

    def __init__(self, clock, ticks=2, throttle_first_n=0, fail_targets=()):
        self.clock = clock
        self.ticks = ticks
        self.tasks: dict[str, dict] = {}
        self.n = 0
        self.inflight_peak = 0
        self.throttle_left = throttle_first_n
        self.fail_targets = set(fail_targets)
        self.submissions: list[dict] = []

    def submit(self, ra, dec, mjd_lo, mjd_hi, use_reduced=False):
        if self.throttle_left > 0:
            self.throttle_left -= 1
            raise A.AltFeedError("ATLAS queue is throttling (429); reduce the batch size")
        self.n += 1
        url = f"task/{self.n}"
        self.tasks[url] = {"polls": 0, "ra": ra, "reduced": use_reduced,
                           "lo": mjd_lo, "hi": mjd_hi}
        self.submissions.append({"url": url, "ra": ra, "reduced": use_reduced,
                                 "lo": mjd_lo, "hi": mjd_hi})
        live = sum(1 for t in self.tasks.values() if t["polls"] < self.ticks)
        self.inflight_peak = max(self.inflight_peak, live)
        return url

    def poll_once(self, url):
        t = self.tasks[url]
        t["polls"] += 1
        if t["polls"] < self.ticks:
            return None, None
        if any(abs(t["ra"] - (10.0 + i)) < 1e-9 for i, tid in enumerate("abcdefgh")
               if tid in self.fail_targets):
            raise A.AltFeedError("ATLAS task failed: No data returned")
        return _atlas_text(n=60, t0=t["lo"]), 300.0


def _jobs(ids, with_baseline=True, lo=60000.0, hi=60100.0):
    return [W.AtlasTargetJob(target_id=t, ra=10.0 + i, dec=2.0, pmra=0.0, pmdec=0.0,
                             mjd_lo=lo, mjd_hi=hi, with_baseline=with_baseline)
            for i, t in enumerate(ids)]


def test_the_scheduler_keeps_several_tasks_in_flight_and_assembles_every_target():
    clk = FakeClock()
    fake = FakeAtlas(clk)
    walk = W.AtlasWalk(fake, concurrency=4, clock=clk.now, sleep=clk.sleep)
    lcs, q, notes = walk.run(_jobs("abcdef"), deadline=10_000.0)
    assert set(lcs) == set("abcdef")
    assert all(len(lc) == 60 for lc in lcs.values())
    assert all("o" in q[t] for t in "abcdef"), "the reduced pass measured F*"
    assert fake.inflight_peak <= 4
    assert fake.inflight_peak >= 2, "nothing was overlapped"
    assert walk.n_tasks_submitted == 12 and walk.n_tasks_finished == 12
    assert any("achieved parallelism" in n for n in notes)


def test_a_refresh_job_skips_the_reduced_pass():
    clk = FakeClock()
    fake = FakeAtlas(clk)
    walk = W.AtlasWalk(fake, concurrency=2, clock=clk.now, sleep=clk.sleep)
    lcs, q, _ = walk.run(_jobs("a", with_baseline=False), deadline=10_000.0)
    assert set(lcs) == {"a"}
    assert q["a"] == {}
    assert all(not s["reduced"] for s in fake.submissions)


def test_the_scheduler_abandons_in_flight_targets_at_the_deadline():
    clk = FakeClock()
    fake = FakeAtlas(clk, ticks=50)                     # each task needs 50 polls
    walk = W.AtlasWalk(fake, concurrency=2, clock=clk.now, sleep=clk.sleep)
    lcs, _q, notes = walk.run(_jobs("abcd"), deadline=100.0)   # 10 polls' worth
    assert lcs == {}
    assert any("deadline reached" in n and "discarded" in n for n in notes)
    assert any("never started" in n for n in notes)
    assert clk.now() <= 100.0 + fake.poll_s


def test_the_scheduler_does_not_start_a_target_it_cannot_finish():
    clk = FakeClock()
    fake = FakeAtlas(clk, ticks=3)                      # a target takes ~30 s
    walk = W.AtlasWalk(fake, concurrency=1, clock=clk.now, sleep=clk.sleep)
    # Enough for two targets; the third would run past the deadline.
    lcs, _q, notes = walk.run(_jobs("abc"), deadline=100.0)
    assert set(lcs) == {"a", "b"}
    assert any("would not finish" in n for n in notes)
    assert fake.n == 4, "the third target's tasks were never queued"


def test_a_throttled_queue_pauses_and_retries_the_same_task():
    clk = FakeClock()
    fake = FakeAtlas(clk, throttle_first_n=1)
    walk = W.AtlasWalk(fake, concurrency=2, clock=clk.now, sleep=clk.sleep,
                       throttle_pause_s=30.0)
    lcs, _q, notes = walk.run(_jobs("a"), deadline=10_000.0)
    assert set(lcs) == {"a"}
    assert any("throttled" in n for n in notes)
    assert walk.n_tasks_submitted == 2                  # nothing duplicated


def test_a_failed_task_fails_only_its_own_target():
    clk = FakeClock()
    fake = FakeAtlas(clk, fail_targets={"b"})
    walk = W.AtlasWalk(fake, concurrency=3, clock=clk.now, sleep=clk.sleep)
    lcs, _q, notes = walk.run(_jobs("abc"), deadline=10_000.0)
    assert set(lcs) == {"a", "c"}
    assert any(n.startswith("b:") and "failed" in n for n in notes)


def test_segments_carry_their_own_positions_and_windows():
    clk = FakeClock()
    fake = FakeAtlas(clk)
    walk = W.AtlasWalk(fake, concurrency=8, clock=clk.now, sleep=clk.sleep)
    job = W.AtlasTargetJob(target_id="fast", ra=10.0, dec=2.0, pmra=2000.0, pmdec=0.0,
                           mjd_lo=57200.0, mjd_hi=61000.0, with_baseline=False)
    lcs, _q, _ = walk.run([job], deadline=10_000.0)
    assert "fast" in lcs
    assert len(job.segs) > 1
    los = sorted(s["lo"] for s in fake.submissions)
    assert los[0] == pytest.approx(57200.0) and len(los) == len(job.segs)
    assert any("CAPPED" in n for n in lcs["fast"].notes) == (len(job.segs) ==
                                                              A.AtlasForcedPhotometry.MAX_PM_SEGMENTS)


# ---------------------------------------------------------------------------
# 6. the planned fetch, end to end with a serial fake
# ---------------------------------------------------------------------------
class SerialAtlas:
    available = True

    def __init__(self, *a, **k):
        pass

    calls: list[dict] = []

    def lightcurve(self, tid, ra, dec, pmra, pmdec, lo, hi, th, with_baseline=True,
                   deadline=None):
        SerialAtlas.calls.append({"tid": tid, "lo": lo, "hi": hi, "baseline": with_baseline})
        return _atlas_lc(tid=tid, n=200, t0=lo), dict(Q)


def test_the_planned_fetch_records_windows_and_a_second_run_only_refreshes(monkeypatch):
    monkeypatch.setattr(A, "AtlasForcedPhotometry", SerialAtlas)
    monkeypatch.delenv(W.JOB_DEADLINE_ENV, raising=False)
    monkeypatch.setenv("ALTFEEDS_FETCH_BUDGET_S", "9000")
    SerialAtlas.calls.clear()
    t = _targets(("a", "b", "c"))
    st = W.WalkState(survey="atlas")
    lcs, _q, notes, windows = A._fetch_planned(A.ATLAS, t, A.LightCurveThresholds(),
                                               max_targets=2, mjd_lo=None, mjd_hi=None,
                                               state=st, now=61000.0)
    assert set(lcs) == {"a", "b"}
    assert windows["a"]["mode"] == "full"
    assert windows["a"]["mjd_lo"] == 57200.0
    # The default window ends at the wall clock minus the ingest lag --- not at
    # the fixed MJD 61300 the old code carried, which would have gone stale on
    # 2026-09-21 and silently excluded every night after it.
    assert windows["a"]["mjd_hi"] == pytest.approx(61000.0 - W.DEFAULT_INGEST_LAG_DAYS)
    assert any("walked 2 of 2" in n for n in notes)
    for tid, w in windows.items():
        st.record(tid, mjd_lo=w["mjd_lo"], mjd_hi=w["mjd_hi"], mode=w["mode"], usable=True,
                  n_epochs=200, baseline={"o": {"level": 0.0, "scatter": 500.0, "n_used": 200}})

    SerialAtlas.calls.clear()
    lcs2, _q2, notes2, windows2 = A._fetch_planned(A.ATLAS, t, A.LightCurveThresholds(),
                                                   max_targets=1, mjd_lo=None, mjd_hi=None,
                                                   state=st, now=61030.0)
    modes = {tid: w["mode"] for tid, w in windows2.items()}
    assert modes == {"a": "refresh", "b": "refresh", "c": "full"}
    by_tid = {c["tid"]: c for c in SerialAtlas.calls}
    assert by_tid["a"]["lo"] == pytest.approx(61000.0 - W.DEFAULT_INGEST_LAG_DAYS)
    assert by_tid["a"]["baseline"] is False, "a refresh does not repeat the reduced pass"
    assert by_tid["c"]["baseline"] is True
    assert any("2 refresh, 1 full" in n for n in notes2)


def test_a_spent_job_deadline_stops_the_walk_before_any_call(monkeypatch):
    monkeypatch.setattr(A, "AtlasForcedPhotometry", SerialAtlas)
    monkeypatch.setenv("ALTFEEDS_FETCH_BUDGET_S", "9000")
    monkeypatch.setenv(W.JOB_DEADLINE_ENV, "0")           # long past
    SerialAtlas.calls.clear()
    lcs, _q, notes = A._fetch(A.ATLAS, _targets(("a",)), A.LightCurveThresholds(),
                              max_targets=1, mjd_lo=58000.0, mjd_hi=59000.0)
    assert lcs == {} and SerialAtlas.calls == []
    assert any("clipped" in n for n in notes)
    assert any("budget" in n and "exhausted" in n for n in notes)


def test_run_survey_saves_the_walk_state_and_counts_walked_stars(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "AtlasForcedPhotometry", SerialAtlas)
    monkeypatch.delenv(W.JOB_DEADLINE_ENV, raising=False)
    monkeypatch.setenv("ALTFEEDS_FETCH_BUDGET_S", "9000")
    t = _targets(("a", "b"), mag=15.0)
    tp = tmp_path / "targets.parquet"
    t.to_parquet(tp)
    out = tmp_path / "out"
    rec = A.run_survey("atlas", targets_path=tp, out_dir=out, max_targets=2)
    assert rec["verdict"] == "OK"
    assert rec["walk"]["n_walked"] == 2 and rec["walk"]["n_usable"] == 2
    assert rec["walk"]["this_run"] == {"full": 2}
    st = W.WalkState.load(out / "walked.json")
    assert set(st.targets) == {"a", "b"}
    assert st.get("a").baseline["o"]["n_used"] > 0
    assert st.get("a").quiescent["o"]["source"] == "atlas_reduced_images"
    led = json.loads((out / "ledger_atlas.json").read_text())
    assert led["n_targets_screened"] == 2, "the population is the stars walked, not the list"
    assert led["n_target_visits"] == 400

    # A second run, a month on: both stars refreshed, judged against the priors,
    # and nothing double counted.
    monkeypatch.setattr(W, "now_mjd", lambda: W.MJD_UNIX_EPOCH + __import__("time").time() / 86400.0 + 30.0)
    rec2 = A.run_survey("atlas", targets_path=tp, out_dir=out, max_targets=2)
    assert rec2["walk"]["this_run"] == {"refresh": 2}
    led2 = json.loads((out / "ledger_atlas.json").read_text())
    assert led2["n_target_visits"] > 400
    assert math.isfinite(W.WalkState.load(out / "walked.json").get("a").mjd_hi)


def test_a_target_whose_atlas_tasks_fail_is_recorded_unusable_not_retried(monkeypatch):
    """Run 33940907907: two of five walked stars failed with 'No data returned' and,
    unrecorded, would have been walked again on every run."""
    class Failing:
        available = True

        def __init__(self, *a, **k):
            pass

        def lightcurve(self, tid, ra, dec, pmra, pmdec, lo, hi, th, with_baseline=True,
                       deadline=None):
            raise A.AltFeedError("ATLAS task failed: No data returned")

    monkeypatch.setattr(A, "AtlasForcedPhotometry", Failing)
    monkeypatch.delenv(W.JOB_DEADLINE_ENV, raising=False)
    monkeypatch.setenv("ALTFEEDS_FETCH_BUDGET_S", "9000")
    st = W.WalkState(survey="atlas")
    t = _targets(("a",))
    lcs, _q, _n, windows = A._fetch_planned(A.ATLAS, t, A.LightCurveThresholds(),
                                            max_targets=1, mjd_lo=None, mjd_hi=None,
                                            state=st, now=61000.0)
    assert lcs == {} and windows == {}
    rec = st.get("a")
    assert rec is not None and rec.usable is False
    assert any("atlas_failed" in n for n in rec.notes)
    plan = st.plan(["a"], survey_start_mjd=57200.0, now=61001.0, max_new=1)
    assert plan.requests == [] and plan.skipped_unusable == 1


def test_the_reduced_pass_is_off_unless_asked_for(monkeypatch):
    seen = {}

    class Client:
        available = True

        def __init__(self, *a, **k):
            pass

        def submit(self, ra, dec, lo, hi, use_reduced=False):
            seen.setdefault("reduced", []).append(use_reduced)
            return f"t{len(seen['reduced'])}"

        def poll_once(self, url):
            return _atlas_text(n=30), 100.0

    monkeypatch.setattr(A, "AtlasForcedPhotometry", Client)
    monkeypatch.delenv(W.JOB_DEADLINE_ENV, raising=False)
    monkeypatch.delenv("ALTFEEDS_ATLAS_REDUCED_PASS", raising=False)
    monkeypatch.setenv("ALTFEEDS_FETCH_BUDGET_S", "9000")
    A._fetch_planned(A.ATLAS, _targets(("a",)), A.LightCurveThresholds(), max_targets=1,
                     mjd_lo=None, mjd_hi=None, state=W.WalkState(survey="atlas"), now=61000.0)
    assert seen["reduced"] and not any(seen["reduced"])
    seen.clear()
    monkeypatch.setenv("ALTFEEDS_ATLAS_REDUCED_PASS", "1")
    A._fetch_planned(A.ATLAS, _targets(("a",)), A.LightCurveThresholds(), max_targets=1,
                     mjd_lo=None, mjd_hi=None, state=W.WalkState(survey="atlas"), now=61000.0)
    assert any(seen["reduced"])
