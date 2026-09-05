"""Tests for the unattended notifier.

The notifier's whole job is to be silent until it should not be, so the tests
that matter are the ones that pin BOTH directions: that ordinary output does not
notify, and that the specific conditions worth a human's attention do.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from seti.alerts import (
    DATA_LAG_LIMIT_DAYS,
    FRONTIER_STALL_DAYS,
    MJD_EPOCH,
    STALE_DAYS,
    check,
    current_frontiers,
    evaluate,
    frontier_recovery_alerts,
    health_alerts,
    issue_body,
    issue_labels,
    issue_title,
    loom_alerts,
    outage_context,
    record_frontier,
    tocsin_alerts,
)

NOW = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)


def _write(root: Path, rel: str, payload: dict) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))
    return p


def _stamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Silence on ordinary output
# ---------------------------------------------------------------------------
def test_empty_tree_is_silent(tmp_path):
    """No results at all is not an alert -- a channel that has never run is not
    a channel that has died, and confusing the two notifies on day one."""
    assert evaluate(tmp_path, NOW) == []


def test_ordinary_null_run_is_silent(tmp_path):
    """The state the repository is actually in: everything ran, nothing was
    promoted. This must produce no notification, or the alert is noise."""
    _write(tmp_path, "results/tocsin/summary.json",
           {"verdict": "NO_CANDIDATES", "run_at_utc": _stamp(NOW)})
    _write(tmp_path, "results/tocsin/assessment.json",
           {"tier_counts": {"candidate": 0, "interest": 3}})
    _write(tmp_path, "results/tocsin/population.json", {"verdict": "NO_STRUCTURE"})
    _write(tmp_path, "results/loom/screen.json",
           {"screened_at_utc": _stamp(NOW),
            "funnel_final": {"n_candidate": 0, "n_ordinary": 7},
            "coverage": {"law_discrimination_available": False}})
    _write(tmp_path, "results/loom/assessment.json",
           {"replication": {"verdict": "NO_STRUCTURE"},
            "controls": {"verdict": "SCREEN_VALIDATED"}})
    assert evaluate(tmp_path, NOW) == []


# ---------------------------------------------------------------------------
# The conditions that should notify
# ---------------------------------------------------------------------------
def test_tocsin_candidate_fires_and_keys_on_targets(tmp_path):
    _write(tmp_path, "results/tocsin/assessment.json",
           {"tier_counts": {"candidate": 2}, "candidates": ["A", "B"]})
    a = tocsin_alerts(tmp_path)
    assert [x.severity for x in a] == ["candidate"]

    # Re-running on the same two targets must produce the SAME key, or the
    # weekly re-screen notifies about the same pair forever.
    _write(tmp_path, "results/tocsin/assessment.json",
           {"tier_counts": {"candidate": 2}, "candidates": ["B", "A"]})
    assert tocsin_alerts(tmp_path)[0].key == a[0].key

    # A third target joining is a different key: new information notifies.
    _write(tmp_path, "results/tocsin/assessment.json",
           {"tier_counts": {"candidate": 3}, "candidates": ["A", "B", "C"]})
    assert tocsin_alerts(tmp_path)[0].key != a[0].key


def test_loom_candidate_and_replication_fire(tmp_path):
    _write(tmp_path, "results/loom/screen.json",
           {"funnel_final": {"n_candidate": 1}, "frontier_mjd": 61000.0})
    _write(tmp_path, "results/loom/assessment.json",
           {"replication": {"verdict": "REPLICATION_STRUCTURE_DETECTED"},
            "n_anomalous": 5})
    sev = sorted(x.severity for x in loom_alerts(tmp_path))
    assert sev == ["candidate", "candidate"]


def test_failed_positive_control_is_a_health_alert(tmp_path):
    """A screen shown a known artificial object that fails to flag it produces
    no error and a clean null. That is the worst outcome available here."""
    _write(tmp_path, "results/loom/assessment.json",
           {"controls": {"verdict": "SCREEN_INSENSITIVE"}})
    a = loom_alerts(tmp_path)
    assert [x.severity for x in a] == ["health"]


def test_law_discrimination_milestone(tmp_path):
    """LOOM's central discriminant needs two apparitions and the survey is a
    month old. The week that changes is the week the channel can answer its own
    question."""
    _write(tmp_path, "results/loom/screen.json",
           {"coverage": {"law_discrimination_available": True,
                         "n_multi_apparition": 12,
                         "residual_arc_days_median": 240.0}})
    a = [x for x in loom_alerts(tmp_path) if x.severity == "milestone"]
    assert len(a) == 1
    assert "law-discrimination" in a[0].title


def test_no_data_reached_is_not_a_null(tmp_path):
    _write(tmp_path, "results/loom/screen.json",
           {"verdict": "NO_DATA_REACHED", "screened_at_utc": _stamp(NOW)})
    a = [x for x in health_alerts(tmp_path, NOW) if "no_data_reached" in x.key]
    assert len(a) == 1
    assert a[0].severity == "health"


# ---------------------------------------------------------------------------
# Staleness -- the check that has to work on a machine where mtime is a lie
# ---------------------------------------------------------------------------
def test_staleness_is_read_from_the_file_not_the_mtime(tmp_path):
    """THE REGRESSION THIS EXISTS FOR.

    A CI runner clones the repository fresh, so every file's mtime is the
    checkout time. A staleness check keyed on mtime is not merely inaccurate on
    the runner -- it can NEVER fire there, which disables the only thing that
    can tell a dead cron from an empty sky.
    """
    old = NOW - timedelta(days=30)
    p = _write(tmp_path, "results/loom/screen.json",
               {"screened_at_utc": _stamp(old), "funnel_final": {}})
    # mtime is now (the file was just written), exactly as after a checkout.
    assert abs(p.stat().st_mtime - datetime.now(timezone.utc).timestamp()) < 60

    stale = [x for x in health_alerts(tmp_path, NOW) if ":stale:" in x.key]
    assert len(stale) == 1
    assert stale[0].detail["age_days"] == pytest.approx(30.0, abs=0.1)


def test_fresh_results_are_not_stale(tmp_path):
    _write(tmp_path, "results/tocsin/summary.json", {"run_at_utc": _stamp(NOW)})
    _write(tmp_path, "results/loom/screen.json", {"screened_at_utc": _stamp(NOW)})
    assert [x for x in health_alerts(tmp_path, NOW) if ":stale:" in x.key] == []


def test_staleness_escalates_by_doubling_not_by_every_period(tmp_path):
    """A dead channel should keep nagging, but nagging every four days for a
    year is how a notification becomes noise."""
    limit = STALE_DAYS["loom"]
    keys = []
    for days in (limit * 1.5, limit * 2.5, limit * 5.0, limit * 9.0):
        _write(tmp_path, "results/loom/screen.json",
               {"screened_at_utc": _stamp(NOW - timedelta(days=days))})
        keys.append([x.key for x in health_alerts(tmp_path, NOW)
                     if ":stale:" in x.key][0])
    assert keys == sorted(set(keys), key=keys.index)      # each one distinct
    assert len(keys) == 4

    # ...but two ages inside the same octave share a key, so they notify once.
    seen = []
    for days in (limit * 2.1, limit * 3.9):
        _write(tmp_path, "results/loom/screen.json",
               {"screened_at_utc": _stamp(NOW - timedelta(days=days))})
        seen.append([x.key for x in health_alerts(tmp_path, NOW)
                     if ":stale:" in x.key][0])
    assert seen[0] == seen[1]


# ---------------------------------------------------------------------------
# The data going dark, as distinct from the channel going dark
# ---------------------------------------------------------------------------
def _mjd_at(dt: datetime) -> float:
    return (dt - MJD_EPOCH).total_seconds() / 86400.0


def test_a_stalled_mirror_alerts_even_though_the_channel_is_healthy(tmp_path):
    """THE FAILURE EVERY OTHER CHECK MISSES.

    Both channels read Rubin through ALeRCE's public mirror. If that mirror
    stops ingesting LSST, the channels keep running on schedule, keep writing a
    fresh run stamp, keep committing, and keep reporting a clean null. Every
    liveness check stays green while the repository has silently stopped
    tracking Rubin at all.
    """
    # Freshly run -- so the staleness check is satisfied...
    _write(tmp_path, "results/tocsin/summary.json", {"run_at_utc": _stamp(NOW)})
    _write(tmp_path, "results/loom/screen.json", {"screened_at_utc": _stamp(NOW)})
    # ...but screening data from three months ago.
    _write(tmp_path, "results/tocsin/ledger.json",
           {"last_mjd_screened": _mjd_at(NOW - timedelta(days=90))})

    alerts = health_alerts(tmp_path, NOW)
    assert [x.key for x in alerts if ":stale:" in x.key] == []      # channel fine
    frontier = [x for x in alerts if "data_frontier" in x.key]
    assert len(frontier) == 1
    assert frontier[0].severity == "health"
    assert frontier[0].detail["lag_days"] == pytest.approx(90.0, abs=0.1)


def test_the_broker_s_ordinary_lag_does_not_alert(tmp_path):
    """The mirror lags ~16 days by design (15.6 d measured). If that fired, the
    alert would be permanently on and therefore worthless."""
    _write(tmp_path, "results/tocsin/ledger.json",
           {"last_mjd_screened": _mjd_at(NOW - timedelta(days=16))})
    _write(tmp_path, "results/loom/screen.json",
           {"frontier_mjd": _mjd_at(NOW - timedelta(days=16)),
            "screened_at_utc": _stamp(NOW)})
    assert [x for x in health_alerts(tmp_path, NOW) if "data_frontier" in x.key] == []
    assert DATA_LAG_LIMIT_DAYS > 16.0


def test_loom_frontier_is_checked_too(tmp_path):
    _write(tmp_path, "results/loom/screen.json",
           {"frontier_mjd": _mjd_at(NOW - timedelta(days=120)),
            "screened_at_utc": _stamp(NOW)})
    a = [x for x in health_alerts(tmp_path, NOW) if "data_frontier" in x.key]
    assert len(a) == 1 and a[0].channel == "loom"


def test_a_missing_or_zeroed_frontier_is_not_an_alert(tmp_path):
    """Zero is this repository's recurring 'missing' value, and an upper bound
    that admits it turns every absent field into a 61,000-day lag."""
    _write(tmp_path, "results/loom/screen.json",
           {"frontier_mjd": 0, "screened_at_utc": _stamp(NOW)})
    _write(tmp_path, "results/tocsin/ledger.json", {"last_mjd_screened": None})
    assert [x for x in health_alerts(tmp_path, NOW) if "data_frontier" in x.key] == []


# ---------------------------------------------------------------------------
# A frontier that has stopped MOVING, as distinct from one that is merely OLD
# ---------------------------------------------------------------------------
def _tocsin_at(root: Path, mjd: float, now: datetime) -> None:
    """A healthy tocsin run whose broker frontier sits at ``mjd``."""
    _write(root, "results/tocsin/summary.json",
           {"run_at_utc": _stamp(now), "broker_frontier_mjd": mjd})


def test_a_frozen_frontier_alerts_long_before_the_age_check_would(tmp_path):
    """THE POINT OF THE CHECK.

    A mirror that stops ingesting is ~16 days behind already, so the absolute
    age check cannot fire for another fortnight -- a fortnight of nulls that
    mean 'no new sky' being read as 'clean sky'. Measuring the frontier against
    itself sees the same failure in the first week.
    """
    mjd = _mjd_at(NOW - timedelta(days=17))          # ordinary lag: age check silent
    _tocsin_at(tmp_path, mjd, NOW - timedelta(days=10))
    record_frontier(tmp_path, NOW - timedelta(days=10))
    _tocsin_at(tmp_path, mjd, NOW)                   # same epoch, ten days later

    alerts = health_alerts(tmp_path, NOW)
    assert [x for x in alerts if "data_frontier" in x.key] == []      # not yet old
    stalled = [x for x in alerts if "frontier_stalled" in x.key]
    assert len(stalled) == 1
    assert stalled[0].severity == "health" and stalled[0].channel == "tocsin"
    assert stalled[0].detail["frozen_days"] == pytest.approx(10.0, abs=0.1)


def test_an_advancing_frontier_is_silent(tmp_path):
    """The mirror doing its job must never notify, however far behind it runs."""
    for day in range(12):
        when = NOW - timedelta(days=11 - day)
        _tocsin_at(tmp_path, _mjd_at(when - timedelta(days=16)), when)
        record_frontier(tmp_path, when)
        assert [x for x in health_alerts(tmp_path, when)
                if "frontier_stalled" in x.key] == []


def test_a_pause_inside_the_limit_is_silent(tmp_path):
    """Below the threshold this is ordinary batching, and an alert that fires on
    ordinary behaviour is trained into noise inside a month."""
    mjd = _mjd_at(NOW - timedelta(days=17))
    _tocsin_at(tmp_path, mjd, NOW - timedelta(days=6))
    record_frontier(tmp_path, NOW - timedelta(days=6))
    _tocsin_at(tmp_path, mjd, NOW)
    assert [x for x in health_alerts(tmp_path, NOW)
            if "frontier_stalled" in x.key] == []
    assert FRONTIER_STALL_DAYS < DATA_LAG_LIMIT_DAYS


def test_the_first_sighting_of_a_frontier_does_not_alert(tmp_path):
    """A value seen once has not been observed to sit still. Starting the clock
    at zero -- rather than at the file's date -- is what keeps a first run, or a
    channel that has just come online, from notifying on day one."""
    _tocsin_at(tmp_path, _mjd_at(NOW - timedelta(days=200)), NOW)
    assert [x for x in health_alerts(tmp_path, NOW)
            if "frontier_stalled" in x.key] == []


def test_the_stall_clock_survives_runs_that_see_no_change(tmp_path):
    """The regression this is most exposed to: if an unchanged frontier rewrote
    first_seen_utc, every run would reset the clock and the alert could never
    fire -- while looking perfectly well-implemented."""
    mjd = _mjd_at(NOW - timedelta(days=17))
    for day in range(10):                                  # ten runs, no advance
        when = NOW - timedelta(days=9 - day)
        _tocsin_at(tmp_path, mjd, when)
        record_frontier(tmp_path, when)
    rec = json.loads((tmp_path / "results/alerts/frontier.json").read_text())
    entry = rec["channels"]["tocsin"]
    assert entry["first_seen_utc"] == _stamp(NOW - timedelta(days=9))
    assert entry["last_seen_utc"] == _stamp(NOW)
    assert entry["n_sightings"] == 10
    assert [x for x in health_alerts(tmp_path, NOW)
            if "frontier_stalled" in x.key] != []


def test_the_broker_frontier_is_preferred_to_the_screened_one(tmp_path):
    """A channel that breaks while the mirror keeps advancing freezes the
    SCREENED frontier. That is a channel bug, and reporting it as a mirror
    outage sends the reader to the wrong system."""
    _write(tmp_path, "results/tocsin/summary.json",
           {"run_at_utc": _stamp(NOW),
            "broker_frontier_mjd": _mjd_at(NOW - timedelta(days=16))})
    _write(tmp_path, "results/tocsin/ledger.json",
           {"last_mjd_screened": _mjd_at(NOW - timedelta(days=120))})
    mjd, source = current_frontiers(tmp_path)["tocsin"]
    assert source == "summary.json:broker_frontier_mjd"
    assert mjd == pytest.approx(_mjd_at(NOW - timedelta(days=16)))


def test_the_screened_frontier_is_the_fallback(tmp_path):
    """loom has no broker field, and tocsin predates one."""
    _write(tmp_path, "results/tocsin/ledger.json",
           {"last_mjd_screened": _mjd_at(NOW - timedelta(days=16))})
    _write(tmp_path, "results/loom/screen.json",
           {"frontier_mjd": _mjd_at(NOW - timedelta(days=16))})
    got = current_frontiers(tmp_path)
    assert got["tocsin"][1] == "ledger.json:last_mjd_screened"
    assert got["loom"][1] == "screen.json:frontier_mjd"


def test_loom_stalls_too(tmp_path):
    _write(tmp_path, "results/loom/screen.json",
           {"frontier_mjd": _mjd_at(NOW - timedelta(days=17)),
            "screened_at_utc": _stamp(NOW)})
    record_frontier(tmp_path, NOW - timedelta(days=9))
    a = [x for x in health_alerts(tmp_path, NOW) if "frontier_stalled" in x.key]
    assert len(a) == 1 and a[0].channel == "loom"


def test_a_missing_frontier_is_never_recorded_or_stalled(tmp_path):
    """Zero is this repository's recurring 'missing' value; an unknown frontier
    is not a stopped one."""
    _write(tmp_path, "results/loom/screen.json",
           {"frontier_mjd": 0, "screened_at_utc": _stamp(NOW)})
    _write(tmp_path, "results/tocsin/ledger.json", {"last_mjd_screened": None})
    rec = record_frontier(tmp_path, NOW)
    assert rec["channels"] == {}
    assert [x for x in health_alerts(tmp_path, NOW)
            if "frontier_stalled" in x.key] == []


def test_the_stall_alert_escalates_by_doubling(tmp_path):
    """A mirror that stays dead should keep nagging, but not every week."""
    mjd = _mjd_at(NOW - timedelta(days=17))
    _tocsin_at(tmp_path, mjd, NOW)
    record_frontier(tmp_path, NOW)
    keys = set()
    for days in (8, 10, 13, 15, 20, 30, 60):
        _tocsin_at(tmp_path, mjd, NOW + timedelta(days=days))
        keys |= {x.key for x in health_alerts(tmp_path, NOW + timedelta(days=days))
                 if "frontier_stalled" in x.key}
    assert keys == {"tocsin:frontier_stalled:0", "tocsin:frontier_stalled:1",
                    "tocsin:frontier_stalled:2", "tocsin:frontier_stalled:3"}


def test_the_observed_cadence_accumulates_for_calibration(tmp_path):
    """FRONTIER_STALL_DAYS is a conservative guess pending a measurement. The
    history is that measurement, so it has to actually record advances."""
    for day in range(6):
        when = NOW - timedelta(days=5 - day)
        _tocsin_at(tmp_path, _mjd_at(when - timedelta(days=16)), when)
        record_frontier(tmp_path, when)
    cadence = (json.loads((tmp_path / "results/alerts/frontier.json").read_text())
               ["channels"]["tocsin"]["observed_advance_days"])
    assert cadence["n_advances"] == 5
    assert cadence["median_days"] == pytest.approx(1.0)


def test_a_dry_run_does_not_move_the_stall_clock(tmp_path):
    """The other way the check could be silently disabled: if the evaluation
    pass recorded the frontier, the clock would reset on every check."""
    mjd = _mjd_at(NOW - timedelta(days=17))
    _tocsin_at(tmp_path, mjd, NOW - timedelta(days=9))
    record_frontier(tmp_path, NOW - timedelta(days=9))
    before = (tmp_path / "results/alerts/frontier.json").read_text()
    _tocsin_at(tmp_path, mjd, NOW)
    rep = check(tmp_path, now=NOW, record=False)
    assert (tmp_path / "results/alerts/frontier.json").read_text() == before
    assert any("frontier_stalled" in a["key"] for a in rep["new"])


def test_the_report_carries_the_frontier_even_when_silent(tmp_path):
    """Below the threshold a stalling mirror is invisible in every other field,
    and it is the first thing to check before reading a null as a result."""
    _tocsin_at(tmp_path, _mjd_at(NOW - timedelta(days=16)), NOW - timedelta(days=2))
    record_frontier(tmp_path, NOW - timedelta(days=2))
    _tocsin_at(tmp_path, _mjd_at(NOW - timedelta(days=16)), NOW)
    rep = check(tmp_path, now=NOW)
    assert not rep["alert"]
    f = rep["frontier"]["tocsin"]
    assert f["frozen_days"] == pytest.approx(2.0, abs=0.1)
    assert f["lag_days"] == pytest.approx(16.0, abs=0.1)


# ---------------------------------------------------------------------------
# Literature suppression
# ---------------------------------------------------------------------------
def _calibration_with(name: str) -> dict:
    return {"epsilon": {"asteroid_rho_2000": {"survivors": [
        {"name": name, "epsilon_effective": 3.2, "A2_au_day2": 1e-13,
         "a2_snr": 8.0, "data_arc_days": 9000, "diameter_m": 1200.0,
         "diameter_measured": True}]}}}


def test_exceedance_already_in_the_literature_does_not_notify(tmp_path):
    """Most ceiling exceedances resolve to the dark-comet population, which is
    incomplete and growing. Alerting on objects already published trains the
    notification into noise using the literature's own progress."""
    _write(tmp_path, "results/loom/calibration.json",
           _calibration_with("469219 Kamooalewa"))
    assert len([x for x in loom_alerts(tmp_path) if "exceedance" in x.key]) == 1

    _write(tmp_path, "results/loom/litcheck.json",
           {"objects": {"469219 Kamooalewa": {"explained_in_literature": True}}})
    assert [x for x in loom_alerts(tmp_path) if "exceedance" in x.key] == []


def test_exceedance_present_but_unexplained_still_notifies(tmp_path):
    _write(tmp_path, "results/loom/calibration.json", _calibration_with("875163"))
    _write(tmp_path, "results/loom/litcheck.json",
           {"objects": {"875163": {"explained_in_literature": False}}})
    a = [x for x in loom_alerts(tmp_path) if "exceedance" in x.key]
    assert len(a) == 1
    assert a[0].key == "loom:exceedance:875163"


# ---------------------------------------------------------------------------
# Deduplication -- the property the whole design rests on
# ---------------------------------------------------------------------------
def test_a_finding_notifies_exactly_once(tmp_path):
    _write(tmp_path, "results/loom/calibration.json", _calibration_with("875163"))
    first = check(tmp_path, now=NOW)
    assert first["n_new"] == 1 and first["alert"] is True

    second = check(tmp_path, now=NOW)
    assert second["n_new"] == 0 and second["alert"] is False
    # ...but it is still ACTIVE. Deduplication must not make a live condition
    # look resolved.
    assert second["n_active"] == 1


def test_dry_run_does_not_consume(tmp_path):
    _write(tmp_path, "results/loom/calibration.json", _calibration_with("875163"))
    assert check(tmp_path, now=NOW, record=False)["n_new"] == 1
    assert check(tmp_path, now=NOW, record=False)["n_new"] == 1
    assert check(tmp_path, now=NOW, record=True)["n_new"] == 1
    assert check(tmp_path, now=NOW, record=True)["n_new"] == 0


def test_check_writes_latest_and_state(tmp_path):
    _write(tmp_path, "results/loom/calibration.json", _calibration_with("875163"))
    check(tmp_path, now=NOW)
    latest = json.loads((tmp_path / "results/alerts/latest.json").read_text())
    assert latest["n_new"] == 1
    state = json.loads((tmp_path / "results/alerts/state.json").read_text())
    assert "loom:exceedance:875163" in state["seen"]
    assert state["seen"]["loom:exceedance:875163"]["severity"] == "candidate"


def test_state_survives_a_corrupt_file(tmp_path):
    """A truncated commit must re-notify, not crash the workflow: an alerting
    system that dies on bad input is worse than one that repeats itself."""
    (tmp_path / "results/alerts").mkdir(parents=True)
    (tmp_path / "results/alerts/state.json").write_text("{ not json")
    _write(tmp_path, "results/loom/calibration.json", _calibration_with("875163"))
    assert check(tmp_path, now=NOW)["n_new"] == 1


# ---------------------------------------------------------------------------
# Issue rendering -- this text is passed through a shell
# ---------------------------------------------------------------------------
def test_issue_title_is_shell_safe_and_bounded():
    alerts = [{"severity": "candidate", "channel": "loom",
               "title": "LOOM: `875163`\nwith a newline " + "x" * 400,
               "body": ""}]
    t = issue_title(alerts)
    assert "\n" not in t and "`" not in t and len(t) <= 250


def test_issue_title_summarises_when_there_are_several():
    alerts = [{"severity": "candidate", "channel": "loom", "title": "a", "body": ""},
              {"severity": "health", "channel": "tocsin", "title": "b", "body": ""}]
    t = issue_title(alerts)
    assert "1 candidate" in t and "1 health" in t


def test_issue_labels_cover_severity_and_channel():
    alerts = [{"severity": "candidate", "channel": "loom", "title": "a", "body": ""},
              {"severity": "health", "channel": "tocsin", "title": "b", "body": ""}]
    assert issue_labels(alerts) == ["candidate", "health", "loom", "tocsin"]


def test_issue_body_states_a_measurement_not_a_finding(tmp_path):
    _write(tmp_path, "results/loom/calibration.json", _calibration_with("875163"))
    rep = check(tmp_path, now=NOW, record=False)
    body = issue_body(rep["new"], repo_url="https://example.test/r")
    assert "875163" in body
    assert "request to look, not a claim" in body
    assert "results/alerts/latest.json" in body
    # It must not assert a detection.
    lowered = body.lower()
    assert "technosignature" not in lowered
    assert "artificial object found" not in lowered


# ---------------------------------------------------------------------------
# A frontier that starts MOVING again
# ---------------------------------------------------------------------------
def test_a_recovery_after_a_stall_fires_a_milestone(tmp_path):
    """THE POINT OF THE CHECK.

    A stall is visible in every run that follows it, so it can be noticed late.
    A recovery is visible in exactly ONE run -- the first whose frontier differs
    from the record -- and afterwards the mirror simply looks healthy, as though
    it always had been. Nothing else in the repository would say the data came
    back.
    """
    old = _mjd_at(NOW - timedelta(days=40))
    _tocsin_at(tmp_path, old, NOW - timedelta(days=20))
    record_frontier(tmp_path, NOW - timedelta(days=20))
    _tocsin_at(tmp_path, _mjd_at(NOW - timedelta(days=16)), NOW)

    a = frontier_recovery_alerts(tmp_path, NOW)
    assert len(a) == 1
    assert a[0].severity == "milestone" and a[0].channel == "tocsin"
    assert a[0].detail["sky_days_gained"] == pytest.approx(24.0, abs=0.1)
    assert a[0].detail["advance_interval_days"] == pytest.approx(20.0, abs=0.1)
    assert a[0].detail["caught_up"] is True


def test_an_ordinary_advance_never_notifies(tmp_path):
    """A healthy mirror advances every night. Notifying on that is precisely the
    noise this module exists to avoid, so the recovery gate is the same
    threshold the stall check uses: an advance is only worth reporting if the
    stall it ended was itself worth reporting."""
    for day in range(12):
        when = NOW - timedelta(days=11 - day)
        _tocsin_at(tmp_path, _mjd_at(when - timedelta(days=16)), when)
        record_frontier(tmp_path, when)
        assert frontier_recovery_alerts(tmp_path, when) == []


def test_a_pause_inside_the_limit_recovers_silently(tmp_path):
    """The symmetry that keeps this honest: a pause too short to raise a stall
    must also be too short to raise a recovery, or the module announces the end
    of something it never reported the start of."""
    mjd = _mjd_at(NOW - timedelta(days=17))
    _tocsin_at(tmp_path, mjd, NOW - timedelta(days=6))
    record_frontier(tmp_path, NOW - timedelta(days=6))
    assert [x for x in health_alerts(tmp_path, NOW)
            if "frontier_stalled" in x.key] == []
    _tocsin_at(tmp_path, _mjd_at(NOW - timedelta(days=11)), NOW)
    assert frontier_recovery_alerts(tmp_path, NOW) == []


def test_a_frontier_going_backwards_is_not_a_recovery(tmp_path):
    """A broker re-indexing, or a channel that started reading a different
    table, moves the frontier the wrong way. Announcing that as recovery would
    promise data that is not there."""
    _tocsin_at(tmp_path, _mjd_at(NOW - timedelta(days=16)), NOW - timedelta(days=20))
    record_frontier(tmp_path, NOW - timedelta(days=20))
    _tocsin_at(tmp_path, _mjd_at(NOW - timedelta(days=60)), NOW)
    assert frontier_recovery_alerts(tmp_path, NOW) == []


def test_a_first_sighting_is_not_a_recovery(tmp_path):
    """A channel coming online has no recorded predecessor, so there is no
    advance to report -- only a first value."""
    _tocsin_at(tmp_path, _mjd_at(NOW - timedelta(days=16)), NOW)
    assert frontier_recovery_alerts(tmp_path, NOW) == []


def test_a_partial_backfill_says_it_is_not_caught_up(tmp_path):
    """One advance is not a recovery. If the mirror is still past the age limit
    the next null still means 'no new sky', and the alert has to say so rather
    than declare victory."""
    _tocsin_at(tmp_path, _mjd_at(NOW - timedelta(days=200)), NOW - timedelta(days=20))
    record_frontier(tmp_path, NOW - timedelta(days=20))
    _tocsin_at(tmp_path, _mjd_at(NOW - timedelta(days=150)), NOW)

    a = frontier_recovery_alerts(tmp_path, NOW)
    assert len(a) == 1 and a[0].detail["caught_up"] is False
    assert "partial backfill" in a[0].body


def test_the_recovery_re_arms_the_stall_key(tmp_path):
    """THE ONE-SHOT BUG.

    The stall key escalates by doubling, so once consumed it is consumed for the
    life of the repository and the NEXT outage is silent until it grows past the
    longest escalation already seen. Re-arming on the recovery that answers it is
    what makes the detector repeatable rather than single-use.
    """
    old = _mjd_at(NOW - timedelta(days=26))
    _tocsin_at(tmp_path, old, NOW - timedelta(days=10))
    check(tmp_path, now=NOW - timedelta(days=10))          # first sighting
    _tocsin_at(tmp_path, old, NOW - timedelta(days=1))
    rep = check(tmp_path, now=NOW - timedelta(days=1))     # stall fires
    assert any("frontier_stalled" in a["key"] for a in rep["new"])

    _tocsin_at(tmp_path, _mjd_at(NOW - timedelta(days=16)), NOW)
    rep = check(tmp_path, now=NOW)                         # recovery
    assert any("frontier_recovered" in a["key"] for a in rep["new"])

    seen = json.loads((tmp_path / "results/alerts/state.json").read_text())["seen"]
    assert not [k for k in seen if "frontier_stalled" in k]

    # And the detector genuinely works a second time.
    frozen = _mjd_at(NOW - timedelta(days=16))
    _tocsin_at(tmp_path, frozen, NOW + timedelta(days=9))
    rep = check(tmp_path, now=NOW + timedelta(days=9))
    assert any("frontier_stalled" in a["key"] for a in rep["new"])


def test_re_arming_never_clears_a_condition_that_is_still_true(tmp_path):
    """Dropping a key whose condition still holds re-raises it on the very next
    run, notifying a human about something they were already told."""
    _write(tmp_path, "results/tocsin/ledger.json",
           {"last_mjd_screened": _mjd_at(NOW - timedelta(days=200))})
    _tocsin_at(tmp_path, _mjd_at(NOW - timedelta(days=200)), NOW - timedelta(days=10))
    check(tmp_path, now=NOW - timedelta(days=10))
    _tocsin_at(tmp_path, _mjd_at(NOW - timedelta(days=200)), NOW - timedelta(days=1))
    check(tmp_path, now=NOW - timedelta(days=1))

    # The frontier advances, but the SCREENED data is still ancient, so the
    # age alert is still active and must survive the re-arm.
    _tocsin_at(tmp_path, _mjd_at(NOW - timedelta(days=150)), NOW)
    rep = check(tmp_path, now=NOW)
    assert any("frontier_recovered" in a["key"] for a in rep["new"])
    active = {a["key"] for a in rep["active"]}
    seen = json.loads((tmp_path / "results/alerts/state.json").read_text())["seen"]
    for k in active:
        if "data_frontier" in k:
            assert k in seen, "an active age alert was re-armed and will re-notify"


# --- the cross-broker verdict, carried into the frontier alerts --------------
#
# Both frontier alerts used to end by telling the reader to check whether ALeRCE
# was still ingesting.  It was, and the stream had stopped instead.  Once
# `rubin-outage` has settled that, repeating the question sends the reader to
# re-diagnose a solved problem and to suspect a service that is working.

def _outage(tmp_path, verdict, alerce_mjd=61235.41918, fink_mjd=61235.5):
    d = tmp_path / "results" / "rubin_outage"
    d.mkdir(parents=True, exist_ok=True)
    (d / "brokers.json").write_text(json.dumps({
        "checked_at_utc": "2026-08-25T20:40:00Z",
        "decision": {"verdict": verdict,
                     "alerce_frontier_mjd": alerce_mjd,
                     "alerce_frontier_utc": "2026-07-14T10:03:37Z",
                     "other_brokers": {"fink": {"frontier_mjd": fink_mjd,
                                                "frontier_utc": "2026-07-14T12:00:00Z"}}}}))


def _frontier_files(tmp_path, mjd=61235.41918):
    for channel, name, key in (("tocsin", "summary.json", "broker_frontier_mjd"),
                               ("loom", "screen.json", "frontier_mjd")):
        d = tmp_path / "results" / channel
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(json.dumps({key: mjd}))


def test_a_settled_sky_stopped_verdict_reaches_the_alert_body(tmp_path):
    _frontier_files(tmp_path)
    _outage(tmp_path, "SKY_STOPPED")
    text = outage_context(tmp_path)
    assert "SKY_STOPPED" in text
    assert "no change to the broker path" in text
    assert "docs/rubin-outage.md" in text


def test_a_mirror_stalled_verdict_is_marked_urgent(tmp_path):
    _frontier_files(tmp_path)
    _outage(tmp_path, "MIRROR_STALLED", fink_mjd=61270.5)
    text = outage_context(tmp_path)
    assert "MIRROR_STALLED" in text and "urgent" in text


def test_an_undecided_verdict_says_nothing(tmp_path):
    # UNDETERMINED_SINGLE_SOURCE is precisely the state in which the reader
    # SHOULD still go and check; asserting a cause there would be a fabrication.
    _frontier_files(tmp_path)
    _outage(tmp_path, "UNDETERMINED_SINGLE_SOURCE")
    assert outage_context(tmp_path) == ""


def test_a_verdict_is_dropped_once_the_frontier_moves_past_it(tmp_path):
    # The mirror advanced after the check ran, so the old verdict explains an
    # epoch nobody is looking at any more. A stale explanation on a live alert is
    # worse than none: it stops the reader looking.
    _frontier_files(tmp_path, mjd=61300.0)
    _outage(tmp_path, "SKY_STOPPED")
    assert outage_context(tmp_path) == ""


def test_no_outage_file_at_all_is_silent_not_an_error(tmp_path):
    _frontier_files(tmp_path)
    assert outage_context(tmp_path) == ""


def test_every_watched_channel_has_a_marker_file_configured():
    # A marker that does not exist is skipped SILENTLY, so a channel missing
    # from STALE_MARKER -- or named with a typo -- has no staleness check at all
    # while looking configured. That is the failure this module exists to catch,
    # applied to itself.
    from seti.alerts import STALE_DAYS, STALE_MARKER

    assert set(STALE_DAYS) == set(STALE_MARKER)
    for channel, name in STALE_MARKER.items():
        assert name.endswith(".json"), channel


def test_a_red_gate_raises_and_a_green_one_does_not(tmp_path):
    """CI red on main was invisible for 22 days in 2026. Not any more."""
    from seti.alerts import gate_alerts

    d = tmp_path / "results" / "cronwatch"
    d.mkdir(parents=True)

    def sweep(gate):
        (d / "status.json").write_text(json.dumps({"gate": gate, "workflows": []}))
        return gate_alerts(tmp_path)

    got = sweep({"status": "RED", "conclusion": "failure", "workflow": "ci.yml",
                 "branch": "main", "head_sha": "deadbeefcafe1234",
                 "run_url": "https://example.invalid/run/7"})
    assert len(got) == 1
    assert got[0].key == "gate:red:deadbeefcafe"
    assert "failure" in got[0].title

    assert sweep({"status": "GREEN", "conclusion": "success"}) == []
    # An API that did not answer is not a red gate, and not a green one either.
    assert sweep({"status": "UNKNOWN", "error": "502"}) == []
    assert sweep({"status": "CANCELLED", "conclusion": "cancelled"}) == []


def test_a_red_gate_notifies_once_per_broken_commit(tmp_path):
    from seti.alerts import gate_alerts

    d = tmp_path / "results" / "cronwatch"
    d.mkdir(parents=True)

    def key_for(sha):
        (d / "status.json").write_text(json.dumps({"gate": {
            "status": "RED", "conclusion": "failure", "head_sha": sha}}))
        return gate_alerts(tmp_path)[0].key

    first = key_for("aaaaaaaaaaaa1111")
    assert key_for("aaaaaaaaaaaa1111") == first        # same head, one alert
    assert key_for("bbbbbbbbbbbb2222") != first        # a new break notifies


def test_a_feed_that_answers_but_cannot_serve_a_light_curve_alerts(tmp_path):
    """ASAS-SN's actual state on 2026-08-26: host up, every cone request 500.

    A screen over it runs, commits and finds nothing -- which is what a quiet sky
    looks like.  The distinction has to be raised, not inferred.
    """
    from seti.alerts import feed_alerts

    d = tmp_path / "results" / "tocsin_altfeeds"
    d.mkdir(parents=True)
    (d / "probe.json").write_text(json.dumps({
        "surveys": {"asassn": {"reached": True, "usable": False,
                               "unusable_reason": "no cone request was accepted"},
                    "atlas": {"reached": True, "usable": True, "verdict": "OK"}}}))

    got = feed_alerts(tmp_path)
    assert [a.channel for a in got] == ["tocsin_altfeeds"]
    assert "ASASSN" in got[0].title
    assert "no cone request was accepted" in got[0].body


def test_a_feed_alert_is_keyed_by_the_reason_so_a_new_break_re_notifies(tmp_path):
    from seti.alerts import feed_alerts

    d = tmp_path / "results" / "tocsin_altfeeds"
    d.mkdir(parents=True)

    def probe(reason):
        (d / "probe.json").write_text(json.dumps({"surveys": {"atlas": {
            "reached": True, "usable": False, "unusable_reason": reason}}}))
        return feed_alerts(tmp_path)[0].key

    first = probe("the token was rejected")
    assert probe("the token was rejected") == first       # same break, same key
    assert probe("the queue endpoint moved") != first     # a new break notifies


def test_the_channel_that_explains_the_outage_is_watched_too(tmp_path):
    """`rubin-outage` going quiet has a second cost the others do not.

    Its verdict is quoted inside every frontier alert through
    `outage_context`, so a channel that has stopped keeps attributing a live
    alert to a cause nobody re-checked.
    """
    from seti.alerts import STALE_MARKER, health_alerts

    d = tmp_path / "results" / "rubin_outage"
    d.mkdir(parents=True)
    stamp = (NOW - timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%SZ")
    (d / STALE_MARKER["rubin_outage"]).write_text(
        json.dumps({"checked_at_utc": stamp, "decision": {"verdict": "SKY_STOPPED"}}))

    keys = [a.key for a in health_alerts(tmp_path, now=NOW)]
    assert any(k.startswith("rubin_outage:stale:") for k in keys)


def test_a_new_channel_going_quiet_raises_a_stale_alert(tmp_path):
    from seti.alerts import STALE_MARKER, health_alerts

    d = tmp_path / "results" / "tocsin_altfeeds"
    d.mkdir(parents=True)
    stamp = (NOW - timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%SZ")
    (d / STALE_MARKER["tocsin_altfeeds"]).write_text(
        json.dumps({"run_at_utc": stamp, "verdict": "OK"}))

    keys = [a.key for a in health_alerts(tmp_path, now=NOW)]
    assert any(k.startswith("tocsin_altfeeds:stale:") for k in keys)


def test_tocsin_ztf_candidate_fires_and_keys_on_targets(tmp_path):
    from seti.alerts import tocsin_ztf_alerts

    _write(tmp_path, "results/tocsin_ztf/assessment.json",
           {"tier_counts": {"candidate": 1, "interest": 3}, "candidates": ["Z1"]})
    a = tocsin_ztf_alerts(tmp_path)
    assert [x.severity for x in a] == ["candidate"] and a[0].channel == "tocsin_ztf"
    assert "proxy" in a[0].body
    _write(tmp_path, "results/tocsin_ztf/assessment.json",
           {"tier_counts": {"candidate": 1}, "candidates": ["Z1"]})
    assert tocsin_ztf_alerts(tmp_path)[0].key == a[0].key
    _write(tmp_path, "results/tocsin_ztf/assessment.json",
           {"tier_counts": {"interest": 5}})
    assert tocsin_ztf_alerts(tmp_path) == [], "interest is a watchlist entry, not an alert"


def test_tocsin_ztf_is_a_watched_channel():
    from seti.alerts import STALE_DAYS, STALE_MARKER, evaluate  # noqa: F401
    assert "tocsin_ztf" in STALE_DAYS and STALE_MARKER["tocsin_ztf"] == "run.json"
