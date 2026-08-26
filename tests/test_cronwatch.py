"""The scheduler watch: does a cron that never fired get noticed, and re-fired?

The failure this guards against leaves NO trace anywhere else in the repository:
no failed run for the hourly sweep to retry, no stale result until the staleness
window expires days later, and a workflow page that looks healthy because every
run it *did* have was green.  So every rule here is tested against a fake API and
a frozen clock -- the live service is exactly the part that cannot be exercised
in the sandbox, and a monitor nobody can test is a monitor nobody should trust.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from seti import cronwatch as cw

NOW = datetime(2026, 8, 26, 21, 30, tzinfo=timezone.utc)      # a Wednesday


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# 1. cron arithmetic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("expr,expected", [
    ("17 * * * *", utc(2026, 8, 26, 21, 17)),                 # hourly
    ("10 14 * * *", utc(2026, 8, 26, 14, 10)),                # daily, already past
    ("5 23 * * *", utc(2026, 8, 25, 23, 5)),                  # daily, not yet today
    ("40 18 * * 3", utc(2026, 8, 26, 18, 40)),                # weekly, today
    ("20 13 * * 1", utc(2026, 8, 24, 13, 20)),                # weekly, Monday
    ("10 17 2 * *", utc(2026, 8, 2, 17, 10)),                 # monthly, 2nd
    ("20 19 3 * *", utc(2026, 8, 3, 19, 20)),                 # monthly, 3rd
    ("0 */6 * * *", utc(2026, 8, 26, 18, 0)),                 # step
    ("0 0 1 1 *", utc(2026, 1, 1, 0, 0)),                     # yearly
])
def test_prev_fire_matches_the_schedules_this_repo_actually_uses(expr, expected):
    assert cw.prev_fire(expr, NOW) == expected


def test_the_minute_of_the_current_hour_is_respected():
    """A firing due in three minutes has not happened yet."""
    assert cw.prev_fire("33 21 * * *", utc(2026, 8, 26, 21, 30)) == utc(
        2026, 8, 25, 21, 33)
    assert cw.prev_fire("33 21 * * *", utc(2026, 8, 26, 21, 33)) == utc(
        2026, 8, 26, 21, 33)


def test_day_of_month_and_day_of_week_are_or_ed_not_and_ed():
    """Vixie semantics, which GitHub follows.

    AND-ing them would make `0 12 1 * 1` look like it fires only on a first of
    the month that is also a Monday -- so this module would report a live
    channel as dropped for months at a time.
    """
    c = cw.parse_cron("0 12 1 * 1")
    assert c.matches_day(utc(2026, 9, 1, 0, 0))               # a 1st (Tuesday)
    assert c.matches_day(utc(2026, 8, 24, 0, 0))              # a Monday
    assert not c.matches_day(utc(2026, 8, 25, 0, 0))          # neither


def test_sunday_is_both_zero_and_seven():
    assert cw.parse_cron("0 0 * * 7").dows == frozenset({0})
    assert cw.prev_fire("0 0 * * 0", NOW) == utc(2026, 8, 23, 0, 0)


@pytest.mark.parametrize("bad", ["", "1 2 3", "60 * * * *", "* 24 * * *",
                                 "0 0 * * 8", "*/0 * * * *", "5-1 * * * *"])
def test_a_malformed_cron_is_an_error_not_a_silent_never(bad):
    with pytest.raises(ValueError):
        cw.parse_cron(bad)


def test_cadence_and_grace_scale_with_the_channel():
    assert cw.cadence("17 * * * *", NOW) == timedelta(hours=1)
    assert cw.cadence("10 14 * * *", NOW) == timedelta(days=1)
    assert cw.cadence("40 18 * * 3", NOW) == timedelta(days=7)
    # The floor keeps ordinary GitHub drift from alerting; the cap keeps a weekly
    # channel from waiting two days to admit it never ran.
    assert cw.grace_for(timedelta(hours=1)) == cw.MIN_GRACE
    assert cw.grace_for(timedelta(days=1)) == timedelta(hours=6)
    assert cw.grace_for(timedelta(days=7)) == cw.MAX_GRACE


# ---------------------------------------------------------------------------
# 2. the registry, read from the workflow files
# ---------------------------------------------------------------------------
def write_workflow(root, name, body):
    d = root / ".github" / "workflows"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(body)


def test_the_registry_survives_yamls_boolean_on_key(tmp_path):
    """`on:` parses as the BOOLEAN True in YAML 1.1, which PyYAML implements.

    Reading `doc["on"]` finds nothing in every real workflow file, and a registry
    that finds no schedules reports a perfectly healthy repository forever.
    """
    write_workflow(tmp_path, "screen.yml", """
name: screen
on:
  schedule:
    - cron: "40 18 * * 3"
  workflow_dispatch:
jobs:
  go:
    runs-on: ubuntu-latest
    steps: [{run: "true"}]
""")
    write_workflow(tmp_path, "manual.yml", """
name: manual
on:
  workflow_dispatch:
jobs:
  go:
    runs-on: ubuntu-latest
    steps: [{run: "true"}]
""")
    got = cw.read_schedules(tmp_path)
    assert [w.file for w in got] == ["screen.yml"]            # unscheduled ignored
    assert got[0].crons == ["40 18 * * 3"]
    assert got[0].has_dispatch is True
    assert got[0].name == "screen"


def test_the_registry_reads_this_repository(tmp_path):
    """The real workflows, so a rename cannot quietly empty the watch list."""
    got = {w.file: w for w in cw.read_schedules(".")}
    assert "tocsin.yml" in got and "watchdog.yml" in got
    assert got["tocsin-altfeeds.yml"].crons == ["40 18 * * 3"]
    # Every scheduled workflow must be re-firable, or a dropped firing cannot be
    # recovered by anything short of a human.
    for wf in got.values():
        assert wf.has_dispatch, f"{wf.file} has a schedule but no workflow_dispatch"


# ---------------------------------------------------------------------------
# 3. the verdict
# ---------------------------------------------------------------------------
def wf(file="screen.yml", cron="40 18 * * 3", dispatch=True):
    return cw.ScheduledWorkflow(file=file, name=file.removesuffix(".yml"),
                                crons=[cron], has_dispatch=dispatch)


def test_a_firing_that_never_happened_is_reported():
    """tocsin-altfeeds on 2026-08-26, which is why this module exists."""
    got = cw.assess([wf()], {"screen.yml": utc(2026, 8, 19, 18, 40)},
                    utc(2026, 8, 27, 12, 0))
    assert got[0]["status"] == "MISSED"
    assert got[0]["overdue"] is True
    assert got[0]["expected_last_fire_utc"] == "2026-08-26T18:40:00Z"
    assert got[0]["hours_late"] == pytest.approx(17.33, abs=0.02)


def test_a_workflow_that_ran_on_time_is_not_reported():
    got = cw.assess([wf()], {"screen.yml": utc(2026, 8, 26, 18, 44)},
                    utc(2026, 8, 27, 12, 0))
    assert got[0]["status"] == "OK"
    assert got[0]["overdue"] is False


def test_ordinary_github_lateness_is_not_a_missed_firing():
    """`alerts` ran 2 h 09 m after its slot on 2026-08-26 and was fine.

    A monitor that fires on that is a monitor that gets muted.
    """
    got = cw.assess([wf(cron="5 17 * * *")], {"screen.yml": utc(2026, 8, 25, 17, 6)},
                    utc(2026, 8, 26, 19, 20))
    assert got[0]["status"] == "WITHIN_GRACE"
    assert got[0]["overdue"] is False


def test_a_workflow_that_has_never_run_on_a_schedule_is_overdue_once_due():
    got = cw.assess([wf()], {"screen.yml": None}, utc(2026, 8, 27, 12, 0))
    assert got[0]["status"] == "MISSED"
    assert "never" in got[0]["note"]


def test_an_unreadable_history_is_unknown_never_overdue():
    """Absence of an answer is not absence of a run."""
    got = cw.assess([wf()], {"screen.yml": None}, utc(2026, 8, 27, 12, 0),
                    unknown=frozenset({"screen.yml"}))
    assert got[0]["status"] == "UNKNOWN"
    assert got[0]["overdue"] is False


def test_several_crons_are_judged_by_whichever_fired_last():
    w = cw.ScheduledWorkflow(file="two.yml", name="two",
                             crons=["0 3 * * 1", "0 9 * * 3"], has_dispatch=True)
    got = cw.assess([w], {"two.yml": utc(2026, 8, 24, 3, 1)},
                    utc(2026, 8, 26, 21, 30))
    assert got[0]["cron_matched"] == "0 9 * * 3"
    assert got[0]["expected_last_fire_utc"] == "2026-08-26T09:00:00Z"
    assert got[0]["overdue"] is True


def test_a_schedule_is_not_blamed_for_a_slot_that_predates_it():
    """A cron added on Tuesday did not miss Monday.

    Every one of the four channels put on a schedule on 2026-08-25 would
    otherwise have been reported overdue -- and re-fired -- the moment the watch
    first ran, for slots that existed only in arithmetic.
    """
    got = cw.assess([wf(cron="20 13 * * 1")], {"screen.yml": None},
                    utc(2026, 8, 26, 21, 30),
                    changed_at={"screen.yml": utc(2026, 8, 25, 22, 38)})
    assert got[0]["status"] == "SCHEDULE_TOO_NEW"
    assert got[0]["overdue"] is False
    assert got[0]["schedule_changed_at_utc"] == "2026-08-25T22:38:00Z"


def test_a_workflow_that_fired_is_ok_even_if_the_file_changed_since():
    """The 2026-08-26 sweep mislabelled `tocsin-altfeeds` this way.

    It fired its 18:40 slot (late, at 21:12) and was then reported
    SCHEDULE_TOO_NEW because the file had been edited afterwards for an unrelated
    concurrency change.  A run that covers the firing settles the question.
    """
    got = cw.assess([wf()], {"screen.yml": utc(2026, 8, 26, 21, 12)},
                    utc(2026, 8, 27, 12, 0),
                    changed_at={"screen.yml": utc(2026, 8, 26, 21, 44)})
    assert got[0]["status"] == "OK"


def test_a_slot_after_the_last_edit_is_judged_normally():
    got = cw.assess([wf()], {"screen.yml": None}, utc(2026, 8, 27, 12, 0),
                    changed_at={"screen.yml": utc(2026, 8, 25, 22, 38)})
    assert got[0]["status"] == "MISSED"


# ---------------------------------------------------------------------------
# 4. catch-up
# ---------------------------------------------------------------------------
class FakeApi:
    def __init__(self, last: dict, fail: set[str] = frozenset()):
        self.last = last
        self.fail = set(fail)
        self.dispatched: list[tuple[str, str]] = []

    def last_scheduled_run(self, file):
        if file in self.fail:
            raise RuntimeError("502 from the API")
        return self.last.get(file)

    def dispatch(self, file, ref="main"):
        self.dispatched.append((file, ref))


def repo_with(tmp_path, crons: dict, dispatch=True):
    for name, cron in crons.items():
        write_workflow(tmp_path, name, f"""
name: {name.removesuffix('.yml')}
on:
  schedule:
    - cron: "{cron}"
{"  workflow_dispatch:" if dispatch else ""}
jobs:
  go:
    runs-on: ubuntu-latest
    steps: [{{run: "true"}}]
""")
    return tmp_path


def test_a_dropped_firing_is_re_fired_once_and_only_once(tmp_path):
    root = repo_with(tmp_path, {"screen.yml": "40 18 * * 3"})
    api = FakeApi({"screen.yml": utc(2026, 8, 19, 18, 40)})
    now = utc(2026, 8, 27, 12, 0)

    rep = cw.sweep(root, api=api, now=now, out_dir=root / "out")
    assert rep["n_overdue"] == 1 and rep["n_dispatched"] == 1
    assert api.dispatched == [("screen.yml", "main")]

    # Second sweep, same missed firing, nothing new: no second dispatch.  A
    # catch-up that fails is the failure sweep's business, not a loop here.
    rep2 = cw.sweep(root, api=api, now=now + timedelta(hours=1),
                    out_dir=root / "out")
    assert rep2["n_overdue"] == 1 and rep2["n_dispatched"] == 0
    assert len(api.dispatched) == 1

    state = json.loads((root / "out" / "state.json").read_text())
    assert list(state["caught_up"]) == ["screen.yml@2026-08-26T18:40:00Z"]


def test_the_next_missed_firing_is_a_new_catch_up(tmp_path):
    root = repo_with(tmp_path, {"screen.yml": "40 18 * * 3"})
    api = FakeApi({"screen.yml": utc(2026, 8, 19, 18, 40)})
    cw.sweep(root, api=api, now=utc(2026, 8, 27, 12, 0), out_dir=root / "out")
    cw.sweep(root, api=api, now=utc(2026, 9, 3, 12, 0), out_dir=root / "out")
    assert len(api.dispatched) == 2


def test_a_workflow_without_dispatch_is_reported_but_not_fired(tmp_path):
    root = repo_with(tmp_path, {"screen.yml": "40 18 * * 3"}, dispatch=False)
    api = FakeApi({"screen.yml": utc(2026, 8, 19, 18, 40)})
    rep = cw.sweep(root, api=api, now=utc(2026, 8, 27, 12, 0), out_dir=root / "out")
    assert rep["n_overdue"] == 1 and rep["n_dispatched"] == 0
    assert api.dispatched == []


def test_a_sweep_never_fires_more_than_its_cap(tmp_path):
    root = repo_with(tmp_path, {f"s{i}.yml": "40 18 * * 3" for i in range(6)})
    api = FakeApi({f"s{i}.yml": utc(2026, 8, 19, 18, 40) for i in range(6)})
    rep = cw.sweep(root, api=api, now=utc(2026, 8, 27, 12, 0), out_dir=root / "out")
    assert rep["n_overdue"] == 6
    assert rep["n_dispatched"] == cw.MAX_CATCHUPS_PER_SWEEP


def test_an_api_failure_does_not_become_a_missed_firing(tmp_path):
    root = repo_with(tmp_path, {"screen.yml": "40 18 * * 3"})
    api = FakeApi({}, fail={"screen.yml"})
    rep = cw.sweep(root, api=api, now=utc(2026, 8, 27, 12, 0), out_dir=root / "out")
    assert rep["n_unknown"] == 1 and rep["n_overdue"] == 0
    assert api.dispatched == []


def test_a_failed_dispatch_is_recorded_and_retried_next_sweep(tmp_path):
    class Refusing(FakeApi):
        def dispatch(self, file, ref="main"):
            raise RuntimeError("403 resource not accessible by integration")

    root = repo_with(tmp_path, {"screen.yml": "40 18 * * 3"})
    api = Refusing({"screen.yml": utc(2026, 8, 19, 18, 40)})
    rep = cw.sweep(root, api=api, now=utc(2026, 8, 27, 12, 0), out_dir=root / "out")
    assert rep["n_dispatched"] == 0
    assert "403" in rep["workflows"][0]["catchup_error"]
    state = json.loads((root / "out" / "state.json").read_text())
    assert state["caught_up"] == {}                    # not marked as caught up


def test_the_report_is_written_where_the_alert_layer_reads_it(tmp_path):
    root = repo_with(tmp_path, {"screen.yml": "40 18 * * 3"})
    api = FakeApi({"screen.yml": utc(2026, 8, 26, 18, 41)})
    cw.sweep(root, api=api, now=utc(2026, 8, 27, 12, 0), out_dir=root / "out")
    rep = json.loads((root / "out" / "status.json").read_text())
    assert rep["n_workflows"] == 1 and rep["n_overdue"] == 0
    assert rep["workflows"][0]["last_scheduled_run_utc"] == "2026-08-26T18:41:00Z"
