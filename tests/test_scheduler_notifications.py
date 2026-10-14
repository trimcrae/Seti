"""Self-healing scheduler gaps stay quiet; unresolved incidents notify once."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from seti.alerts import check, scheduler_alerts
from seti.cronwatch import RECOVERY_WAIT_HOURS, sweep

NOW = datetime(2026, 10, 14, 18, 36, tzinfo=timezone.utc)
LAST = datetime(2026, 10, 14, 8, 52, tzinfo=timezone.utc)


class Api:
    def __init__(self):
        self.runs = []
        self.dispatched = []
        self.error = None
        self.dispatch_error = None
        self.last = LAST

    def last_scheduled_run(self, file):
        return self.last

    def recent_runs(self):
        if self.error:
            raise self.error
        return self.runs

    def dispatch(self, file, ref):
        if self.dispatch_error:
            raise self.dispatch_error
        self.dispatched.append((file, ref))


def repo(root, dispatch=True):
    directory = root / ".github" / "workflows"
    directory.mkdir(parents=True)
    (directory / "tocsin-ztf.yml").write_text(
        "name: tocsin-ztf\non:\n  schedule:\n"
        "    - cron: '25 15 * * *'\n    - cron: '25 3 * * *'\n"
        + ("  workflow_dispatch:\n" if dispatch else ""))
    return root


def run(stamp=NOW + timedelta(seconds=16), conclusion="success", **extra):
    record = {"id": 7, "path": ".github/workflows/tocsin-ztf.yml",
              "head_branch": "main", "event": "workflow_dispatch",
              "actor": {"login": "github-actions[bot]"}, "status": "completed",
              "conclusion": conclusion, "run_started_at": stamp.isoformat().replace("+00:00", "Z"),
              "html_url": "https://github.com/trimcrae/Seti/actions/runs/7"}
    return record | extra


def workflow(report):
    return report["workflows"][0]


def test_dispatch_is_pending_not_success_and_survives_repeated_sweeps(tmp_path):
    root, api = repo(tmp_path), Api()
    first = workflow(sweep(root, api=api, now=NOW))
    assert first["recovery_status"] == "PENDING"
    assert scheduler_alerts(root) == []
    second = workflow(sweep(root, api=api, now=NOW + timedelta(hours=1)))
    assert second["catchup_dispatched_utc"] == first["catchup_dispatched_utc"]
    assert second["recovery_age_hours"] == 1
    assert api.dispatched == [("tocsin-ztf.yml", "main")]
    third = workflow(sweep(root, api=api,
                           now=NOW + timedelta(hours=RECOVERY_WAIT_HOURS)))
    assert third["recovery_status"] == "UNRESOLVED"
    assert len(scheduler_alerts(root)) == 1


def test_confirmed_success_is_quiet_and_survives_history_pagination(tmp_path):
    root, api = repo(tmp_path), Api()
    sweep(root, api=api, now=NOW)
    api.runs = [run()]
    recovered = workflow(sweep(root, api=api, now=NOW + timedelta(minutes=20)))
    assert recovered["recovery_status"] == "RECOVERED"
    assert scheduler_alerts(root) == []
    api.runs = []
    assert workflow(sweep(root, api=api, now=NOW + timedelta(hours=3)))["recovery_status"] == "RECOVERED"
    state = json.loads((root / "results/cronwatch/state.json").read_text())
    assert state["scheduler_incidents"] == {}


@pytest.mark.parametrize("conclusion", ["failure", "timed_out", "cancelled", "startup_failure"])
def test_failed_recovery_is_actionable_without_waiting(tmp_path, conclusion):
    root, api = repo(tmp_path), Api()
    sweep(root, api=api, now=NOW)
    api.runs = [run(conclusion=conclusion)]
    record = workflow(sweep(root, api=api, now=NOW + timedelta(minutes=20)))
    assert record["recovery_status"] == "UNRESOLVED"
    assert scheduler_alerts(root)[0].detail["latest_execution"]["conclusion"] == conclusion


def test_new_slots_and_new_dispatches_do_not_restart_one_outage(tmp_path):
    root, api = repo(tmp_path), Api()
    sweep(root, api=api, now=NOW)
    sweep(root, api=api, now=NOW + timedelta(hours=3))
    first = check(root, now=NOW + timedelta(hours=3))
    assert first["n_new"] == 1
    later = workflow(sweep(root, api=api, now=NOW + timedelta(days=1)))
    assert later["recovery_status"] == "UNRESOLVED"
    assert later["recovery_age_hours"] == 24
    assert len(api.dispatched) == 2
    assert scheduler_alerts(root)[0].key == first["new"][0]["key"]
    assert check(root, now=NOW + timedelta(days=1))["n_new"] == 0


def test_recovery_rearms_a_genuinely_new_outage(tmp_path):
    root, api = repo(tmp_path), Api()
    first = workflow(sweep(root, api=api, now=NOW))["scheduler_incident_key"]
    api.runs = [run()]
    sweep(root, api=api, now=NOW + timedelta(minutes=20))
    api.runs = []
    new = workflow(sweep(root, api=api, now=NOW + timedelta(days=1)))
    assert new["scheduler_incident_key"] != first
    assert new["recovery_status"] == "PENDING"


@pytest.mark.parametrize("extra", [
    {"actor": {"login": "trimcrae"}},  # manual dispatch may be probe-only
    {"head_branch": "research"},
    {"event": "push"},
    {"path": ".github/workflows/other.yml"},
])
def test_unrelated_or_partial_runs_cannot_establish_recovery(tmp_path, extra):
    root, api = repo(tmp_path), Api()
    sweep(root, api=api, now=NOW)
    api.runs = [run(**extra)]
    record = workflow(sweep(root, api=api, now=NOW + timedelta(hours=3)))
    assert record["recovery_status"] == "UNRESOLVED"
    assert "last_successful_execution_utc" not in record
    assert len(scheduler_alerts(root)) == 1


def test_no_dispatch_and_dispatch_error_remain_actionable(tmp_path):
    root, api = repo(tmp_path, dispatch=False), Api()
    assert workflow(sweep(root, api=api, now=NOW))["recovery_status"] == "UNRESOLVED"
    assert len(scheduler_alerts(root)) == 1
    api.dispatch_error = RuntimeError("403 resource not accessible")
    root = repo(tmp_path / "other")
    assert "403" in workflow(sweep(root, api=api, now=NOW))["catchup_error"]
    assert len(scheduler_alerts(root)) == 1


def test_outcome_api_failure_is_unknown_not_success_and_wait_is_bounded(tmp_path):
    root, api = repo(tmp_path), Api()
    api.error = RuntimeError("502 outcome query failed")
    pending = workflow(sweep(root, api=api, now=NOW))
    assert pending["recovery_status"] == "PENDING"
    assert "502" in pending["recovery_error"]
    expired = workflow(sweep(root, api=api, now=NOW + timedelta(hours=3)))
    assert expired["recovery_status"] == "UNRESOLVED"
    assert len(scheduler_alerts(root)) == 1


def test_existing_slot_key_does_not_generate_migration_email(tmp_path):
    root, api = repo(tmp_path), Api()
    path = root / "results/alerts/state.json"
    path.parent.mkdir(parents=True)
    old_key = "cron:tocsin-ztf.yml:2026-10-14T15:25:00Z"
    path.write_text(json.dumps({"seen": {old_key: {"first_seen_utc": "2026-10-14T18:39:00Z"}}}))
    sweep(root, api=api, now=NOW)
    sweep(root, api=api, now=NOW + timedelta(hours=3))
    assert scheduler_alerts(root)[0].key == old_key
    assert check(root, now=NOW + timedelta(hours=3))["n_new"] == 0


def test_a_failed_scheduled_start_cannot_rearm_the_same_outage(tmp_path):
    root, api = repo(tmp_path), Api()
    old = workflow(sweep(root, api=api, now=NOW))["scheduler_incident_key"]
    api.last = NOW + timedelta(minutes=5)
    api.runs = [run(stamp=api.last, conclusion="failure", event="schedule")]
    started = workflow(sweep(root, api=api, now=NOW + timedelta(minutes=20)))
    assert not started["overdue"]
    later = workflow(sweep(root, api=api, now=NOW + timedelta(days=1)))
    assert later["scheduler_incident_key"] == old
    assert later["recovery_status"] == "UNRESOLVED"


def test_success_on_a_non_overdue_sweep_is_recorded_and_rearms(tmp_path):
    root, api = repo(tmp_path), Api()
    old = workflow(sweep(root, api=api, now=NOW))["scheduler_incident_key"]
    api.last = NOW + timedelta(minutes=5)
    api.runs = [run(stamp=api.last, event="schedule")]
    healthy = workflow(sweep(root, api=api, now=NOW + timedelta(minutes=20)))
    assert not healthy["overdue"]
    assert healthy["last_successful_execution_utc"]
    api.runs = []
    new = workflow(sweep(root, api=api, now=NOW + timedelta(days=1)))
    assert new["scheduler_incident_key"] != old
    assert new["recovery_status"] == "PENDING"


def test_hourly_successful_catchup_rearms_without_a_new_scheduled_start(tmp_path):
    root, api = repo(tmp_path), Api()
    path = root / ".github/workflows/tocsin-ztf.yml"
    path.write_text("name: tocsin-ztf\non:\n  schedule:\n"
                    "    - cron: '17 * * * *'\n  workflow_dispatch:\n")
    api.last = NOW - timedelta(hours=15)
    sweep(root, api=api, now=NOW)
    sweep(root, api=api, now=NOW + timedelta(hours=3))
    first = check(root, now=NOW + timedelta(hours=3))["new"][0]["key"]
    api.runs = [run(stamp=NOW + timedelta(hours=3, minutes=5))]
    sweep(root, api=api, now=NOW + timedelta(hours=3, minutes=20))
    api.runs = []
    sweep(root, api=api, now=NOW + timedelta(days=1))
    sweep(root, api=api, now=NOW + timedelta(days=1, hours=3))
    second = check(root, now=NOW + timedelta(days=1, hours=3))["new"][0]["key"]
    assert second != first


def test_cached_success_on_another_branch_cannot_resolve_this_branch(tmp_path):
    root, api = repo(tmp_path), Api()
    api.runs = [run(head_branch="research")]
    sweep(root, api=api, now=NOW + timedelta(minutes=20), ref="research")
    api.runs = []
    current = workflow(sweep(root, api=api, now=NOW + timedelta(hours=3), ref="main"))
    assert current["recovery_status"] != "RECOVERED"


@pytest.mark.parametrize("timeout,wait", [(120, 3), (240, 5), (1440, 12)])
def test_pending_grace_allows_declared_runtime_but_remains_bounded(tmp_path, timeout, wait):
    root, api = repo(tmp_path), Api()
    path = root / ".github/workflows/tocsin-ztf.yml"
    with path.open("a") as stream:
        stream.write(f"jobs:\n  screen:\n    timeout-minutes: {timeout}\n")
    record = workflow(sweep(root, api=api, now=NOW))
    assert record["recovery_wait_hours"] == wait
    pending = workflow(sweep(root, api=api, now=NOW + timedelta(hours=wait - 0.1)))
    assert pending["recovery_status"] == "PENDING"
    expired = workflow(sweep(root, api=api, now=NOW + timedelta(hours=wait)))
    assert expired["recovery_status"] == "UNRESOLVED"
