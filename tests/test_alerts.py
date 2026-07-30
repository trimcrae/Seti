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
    STALE_DAYS,
    check,
    evaluate,
    health_alerts,
    issue_body,
    issue_labels,
    issue_title,
    loom_alerts,
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
