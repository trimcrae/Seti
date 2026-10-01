"""Workflow schedule unions, including the live twice-daily ZTF screen."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from seti import cronwatch as cw


def utc(day, hour, minute=0):
    return datetime(2026, 10, day, hour, minute, tzinfo=timezone.utc)


def assess(crons, now, last, *, born=None, **kwargs):
    workflow = cw.ScheduledWorkflow(
        file="screen.yml", name="screen", crons=crons, has_dispatch=True)
    return cw.assess(
        [workflow], {"screen.yml": last}, now,
        changed_at={"screen.yml": born} if born else None, **kwargs)[0]


@pytest.mark.parametrize(("crons", "now", "hours", "grace"), [
    (["25 3 * * *", "25 15 * * *"], utc(1, 16), 12, 3),
    (["25 15 * * *", "25 3 * * *"], utc(2, 4), 12, 3),
    (["0 3 * * *", "0 4 * * *"], utc(1, 4, 1), 1, 2),
    (["0 3 * * *", "0 4 * * *"], utc(2, 3, 1), 23, 5.75),
    (["0 3 * * *", "0 3 * * *"], utc(1, 4), 24, 6),
    (["0 3,15 * * *", "0 15 * * *"], utc(1, 16), 12, 3),
    (["not a cron", "25 3 * * *", "25 15 * * *"], utc(1, 16), 12, 3),
])
def test_cadence_and_grace_use_distinct_union_slots(crons, now, hours, grace):
    rec = assess(crons, now, utc(1, 0))
    assert rec["cadence_hours"] == hours
    assert rec["grace_hours"] == grace


def test_the_real_ztf_workflow_has_a_twelve_hour_cadence():
    root = Path(__file__).resolve().parents[1]
    workflow = next(w for w in cw.read_schedules(root) if w.file == "tocsin-ztf.yml")
    rec = cw.assess([workflow], {"tocsin-ztf.yml": utc(1, 3, 30)}, utc(1, 16))[0]
    assert rec["cadence_hours"] == 12
    assert rec["grace_hours"] == 3
    assert rec["status"] == "WITHIN_GRACE"


@pytest.mark.parametrize(("now", "status"), [
    (utc(1, 18, 25), "WITHIN_GRACE"),
    (utc(1, 18, 26), "MISSED"),
])
def test_a_twice_daily_dropped_slot_uses_three_hours_of_grace(now, status):
    rec = assess(["25 3 * * *", "25 15 * * *"], now, utc(1, 3, 30))
    assert rec["status"] == status
    if status == "MISSED":
        assert rec["missed_fire_utc"] == "2026-10-01T15:25:00Z"
        assert rec["missed_grace_hours"] == 3


def test_a_fresh_slot_does_not_erase_the_previous_mature_miss():
    rec = assess(["25 3 * * *", "25 15 * * *"], utc(2, 15, 26), utc(1, 15, 30))
    assert rec["status"] == "MISSED"
    assert rec["expected_last_fire_utc"] == "2026-10-02T15:25:00Z"
    assert rec["missed_fire_utc"] == "2026-10-02T03:25:00Z"
    assert rec["missed_hours_late"] == pytest.approx(12.02, abs=0.01)
    assert rec["missed_cadence_hours"] == 12
    assert rec["missed_grace_hours"] == 3


def test_a_short_interval_does_not_make_an_irregular_schedule_hourly():
    # Both today's slots are within their own grace.  Applying the latest
    # incoming 1 h interval to 24 h of silence would call this missed too soon.
    rec = assess(["0 3 * * *", "0 4 * * *"], utc(2, 4, 1), utc(1, 4, 5))
    assert rec["status"] == "WITHIN_GRACE"
    assert rec["overdue"] is False


def test_each_irregular_slot_gets_its_own_grace():
    crons = ["0 3 * * *", "0 4 * * *"]
    rec = assess(crons, utc(2, 6), utc(1, 4, 5))
    assert rec["status"] == "WITHIN_GRACE"
    rec = assess(crons, utc(2, 6, 1), utc(1, 4, 5))
    assert rec["status"] == "MISSED"
    assert rec["missed_fire_utc"] == "2026-10-02T04:00:00Z"
    assert rec["missed_grace_hours"] == 2


def test_an_older_irregular_slot_can_mature_before_the_latest_one():
    rec = assess(["0 3 * * *", "0 8 * * *"], utc(2, 8, 1), utc(1, 8, 5))
    assert rec["status"] == "MISSED"
    assert rec["expected_last_fire_utc"] == "2026-10-02T08:00:00Z"
    assert rec["missed_fire_utc"] == "2026-10-02T03:00:00Z"
    assert rec["missed_cadence_hours"] == 19
    assert rec["missed_grace_hours"] == 4.75


def test_a_completed_union_slot_prevents_false_silence_after_it():
    rec = assess(["0 3 * * *", "0 4 * * *"], utc(2, 23), utc(2, 4, 5))
    assert rec["status"] == "OK"


def test_a_slot_before_the_schedule_appeared_is_never_recovered():
    rec = assess(
        ["25 3 * * *", "25 15 * * *"], utc(2, 15, 26), None,
        born=utc(2, 10))
    assert rec["status"] == "WITHIN_GRACE"
    assert "missed_fire_utc" not in rec


def test_no_history_and_no_birth_time_cannot_invent_silence():
    rec = assess(["25 3 * * *", "25 15 * * *"], utc(2, 15, 26), None)
    assert rec["status"] == "WITHIN_GRACE"


@pytest.mark.parametrize("kwargs", [
    {"unknown": {"screen.yml"}},
    {"running": {"screen.yml"}},
])
def test_existing_refusals_win_over_a_mature_union_miss(kwargs):
    rec = assess(
        ["25 3 * * *", "25 15 * * *"], utc(2, 15, 26), utc(1, 15, 30), **kwargs)
    assert rec["overdue"] is False
    assert "missed_fire_utc" not in rec


def test_catchup_dedup_uses_the_mature_slot_and_survives_a_fresh_slot():
    crons = ["25 3 * * *", "25 15 * * *"]
    before = assess(crons, utc(2, 14), utc(1, 15, 30))
    after = assess(crons, utc(2, 15, 26), utc(1, 15, 30))
    state = {"caught_up": {"screen.yml@2026-10-02T03:25:00Z": "2026-10-02T14:00:00Z"}}
    assert cw.plan_catchup([before], {}) == [before]
    assert cw.plan_catchup([before], state) == []
    assert cw.plan_catchup([after], state) == []


def test_union_cadence_retains_the_actual_run_drift_guard():
    now = utc(2, 6, 40)
    crons = ["17 * * * *", "17 * * * *"]
    assert assess(crons, now, now - timedelta(hours=2, minutes=58))["status"]         == "WITHIN_GRACE"
    assert assess(crons, now, now - timedelta(hours=3, minutes=2))["status"] == "MISSED"


@pytest.mark.parametrize(("expression", "expected"), [
    ("20/15 * * * *", {20, 35, 50}),
    ("5/20 * * * *", {5, 25, 45}),
    ("20-50/15 * * * *", {20, 35, 50}),
    ("20 * * * *", {20}),
])
def test_a_numeric_cron_step_continues_to_the_field_maximum(expression, expected):
    assert cw.parse_cron(expression).minutes == frozenset(expected)


def test_slot_alerts_identify_the_mature_miss_with_legacy_fallback(tmp_path):
    from seti.alerts import scheduler_alerts

    directory = tmp_path / "results" / "cronwatch"
    directory.mkdir(parents=True)
    record = {
        "workflow": "screen.yml", "name": "screen", "overdue": True,
        "has_dispatch": True, "missed_by": "grace",
        "expected_last_fire_utc": "2026-10-02T15:25:00Z",
        "cron_matched": "25 15 * * *", "cadence_hours": 24, "hours_late": 0.02,
        "missed_fire_utc": "2026-10-02T03:25:00Z",
        "missed_cron": "25 3 * * *", "missed_cadence_hours": 12,
        "missed_grace_hours": 3, "missed_hours_late": 12.02,
    }
    path = directory / "status.json"
    path.write_text(json.dumps({"workflows": [record]}))
    alert = scheduler_alerts(tmp_path)[0]
    assert alert.key == "cron:screen.yml:2026-10-02T03:25:00Z"
    assert "2026-10-02T03:25:00Z" in alert.title
    assert "cron `25 3 * * *`, cadence 12 h" in alert.body
    assert "12.02 h past that slot" in alert.body
    assert alert.detail["missed_grace_hours"] == 3

    # Old ledgers still identify their only slot as before.
    legacy = {key: value for key, value in record.items() if not key.startswith("missed_")}
    path.write_text(json.dumps({"workflows": [legacy]}))
    assert scheduler_alerts(tmp_path)[0].key == "cron:screen.yml:2026-10-02T15:25:00Z"


def test_slot_alert_identity_is_stable_across_a_fresh_union_slot(tmp_path):
    from seti.alerts import check

    directory = tmp_path / "results" / "cronwatch"
    directory.mkdir(parents=True)
    path = directory / "status.json"
    crons = ["25 3 * * *", "25 15 * * *"]
    last = utc(1, 3, 30)
    before = assess(crons, utc(2, 14), last)
    after = assess(crons, utc(2, 15, 26), last)
    assert before["missed_by"] == after["missed_by"] == "grace"
    assert before["missed_fire_utc"] == after["missed_fire_utc"]
    path.write_text(json.dumps({"workflows": [before]}))
    first = check(tmp_path, now=utc(2, 14))
    assert first["n_new"] == 1
    path.write_text(json.dumps({"workflows": [after]}))
    second = check(tmp_path, now=utc(2, 15, 26))
    assert second["n_new"] == 0
