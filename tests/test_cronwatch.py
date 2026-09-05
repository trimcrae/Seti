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
    assert got["tocsin-altfeeds.yml"].crons == ["40 18 * * 3,6"]   # twice weekly since 2026-09-05
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
# 3b. the gate
# ---------------------------------------------------------------------------
class GateApi:
    """An API that answers only the gate query."""

    def __init__(self, run):
        self.run = run
        self.asked: list[tuple] = []

    def latest_run(self, workflow_file, branch=None, event=None):
        self.asked.append((workflow_file, branch, event))
        if isinstance(self.run, Exception):
            raise self.run
        return self.run


@pytest.mark.parametrize("conclusion,expected", [
    ("success", "GREEN"),
    ("failure", "RED"),
    ("timed_out", "RED"),
    ("startup_failure", "RED"),          # the alerts.yml break of 2026-08-26
    ("cancelled", "CANCELLED"),          # a human stopped it; not a verdict
    (None, "PENDING"),
])
def test_the_gate_reads_the_conclusion(conclusion, expected):
    api = GateApi({"conclusion": conclusion, "status": "completed",
                   "head_sha": "abc123", "id": 1})
    got = cw.gate_status(api, branch="main")
    assert got["status"] == expected
    assert api.asked == [("ci.yml", "main", None)]


def test_an_unread_gate_is_not_a_green_one():
    got = cw.gate_status(GateApi(RuntimeError("502")), branch="main")
    assert got["status"] == "UNKNOWN"
    assert "502" in got["error"]


def test_the_gate_is_recorded_in_the_sweep(tmp_path):
    root = repo_with(tmp_path, {"screen.yml": "40 18 * * 3"})

    class Api(FakeApi):
        def latest_run(self, workflow_file, branch=None, event=None):
            return {"conclusion": "failure", "status": "completed",
                    "head_sha": "deadbeefcafe", "id": 7,
                    "html_url": "https://example.invalid/run/7"}

    rep = cw.sweep(root, api=Api({"screen.yml": utc(2026, 8, 26, 18, 41)}),
                   now=utc(2026, 8, 27, 12, 0), out_dir=root / "out")
    assert rep["gate"]["status"] == "RED"
    assert json.loads(
        (root / "out" / "status.json").read_text())["gate"]["head_sha"] == "deadbeefcafe"


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


def test_the_watch_does_not_depend_on_a_single_cron(tmp_path):
    """The hole found on 2026-08-26: the watcher could not watch itself.

    `cronwatch` ran only inside `watchdog`, which is itself hourly and scheduled.
    So the exact failure it exists to catch, applied to `watchdog`, also switched
    it off -- and that night `watchdog`'s last scheduled run was 22:10 UTC with
    the next still missing 2 h 15 m later, past its own grace, with nothing
    saying so because saying so was its job.
    """
    import yaml

    lanes = []
    for name in ("watchdog.yml", "cronwatch.yml"):
        doc = yaml.safe_load(open(f".github/workflows/{name}"))
        triggers = doc.get(True, doc.get("on")) or {}
        crons = [e["cron"] for e in (triggers.get("schedule") or [])]
        assert crons, f"{name} must be scheduled"
        assert any("cron_watch.py" in str(s.get("run", ""))
                   for s in doc["jobs"][next(iter(doc["jobs"]))]["steps"]), \
            f"{name} must actually run the sweep"
        lanes.append(crons[0])

    assert len(set(lanes)) == 2, "two lanes on the same cron are one lane"
    # Different minute as well as different cadence: a scheduler dropping one
    # particular slot must not take both.
    minutes = {c.split()[0] for c in lanes}
    assert len(minutes) == 2, f"both lanes fire at the same minute: {lanes}"


def test_only_one_lane_dispatches_catch_ups_for_any_ordinary_channel():
    """Two schedules that both re-fire could double-fire the same missed slot.

    The failure being guarded against is the watch going SILENT, and that is
    fixed by a second REPORTER, not by a second actor.

    NARROWED ON 2026-08-27, and this test says why rather than being deleted.
    The rule held for every channel except the actor itself: `watchdog` cannot
    re-fire `watchdog`.  That morning its 04:17 AND 05:17 UTC firings were both
    dropped and it sat 2 h 23 m past a 2 h grace, seen only by the lane that was
    forbidden to act.  So the second lane now runs `--self-heal-only`, which
    dispatches `seti.cronwatch.SELF_HEAL_ONLY` and nothing else -- the
    single-actor rule intact for every ordinary channel, and no race introduced,
    since a `watchdog` that were running is a `watchdog` that is not overdue.
    """
    import yaml

    from seti.cronwatch import SELF_HEAL_ONLY

    doc = yaml.safe_load(open(".github/workflows/cronwatch.yml"))
    steps = doc["jobs"]["sweep"]["steps"]
    runs = " ".join(str(s.get("run", "")) for s in steps)
    assert "--self-heal-only" in runs
    # The exception must stay an exception.
    assert SELF_HEAL_ONLY == {"watchdog.yml"}

    doc2 = yaml.safe_load(open(".github/workflows/watchdog.yml"))
    runs2 = " ".join(str(s.get("run", "")) for s in doc2["jobs"]["sweep"]["steps"])
    assert "cron_watch.py" in runs2
    assert "--no-dispatch" not in runs2 and "--self-heal-only" not in runs2


# ---------------------------------------------------------------------------
# The one workflow the single-actor rule cannot cover (2026-08-27).
#
# Catch-ups belong to `watchdog` alone so two lanes cannot double-fire a missed
# slot.  That holds for every channel except `watchdog` itself, where it is
# circular: `watchdog` cannot re-fire `watchdog`, because the case is precisely
# that `watchdog` did not run.  Measured that morning -- its 04:17 and 05:17 UTC
# firings both dropped, 2 h 23 m past a 2 h grace, and the only lane that could
# see it was the one forbidden to act.
# ---------------------------------------------------------------------------

def _overdue(workflow):
    return {"workflow": workflow, "overdue": True, "has_dispatch": True,
            "expected_last_fire_utc": "2026-08-27T05:17:00Z"}


def test_the_second_lane_may_re_fire_the_watchdog_and_nothing_else():
    from seti.cronwatch import SELF_HEAL_ONLY, plan_catchup

    findings = [_overdue("watchdog.yml"), _overdue("tocsin.yml"),
                _overdue("loom.yml")]
    planned = plan_catchup(findings, {}, only=SELF_HEAL_ONLY)
    assert [r["workflow"] for r in planned] == ["watchdog.yml"]


def test_the_watchdog_lane_itself_still_dispatches_everything():
    """`only` must not become the default, or every catch-up stops."""
    from seti.cronwatch import plan_catchup

    findings = [_overdue("tocsin.yml"), _overdue("loom.yml")]
    assert len(plan_catchup(findings, {})) == 2


def test_a_watchdog_catch_up_is_still_issued_only_once():
    """The single-dispatch ledger rule is not weakened by the exception."""
    from seti.cronwatch import SELF_HEAL_ONLY, plan_catchup

    findings = [_overdue("watchdog.yml")]
    first = plan_catchup(findings, {}, only=SELF_HEAL_ONLY)
    assert len(first) == 1
    state = {"caught_up": {"watchdog.yml@2026-08-27T05:17:00Z": "2026-08-27T05:36:00Z"}}
    assert plan_catchup(findings, state, only=SELF_HEAL_ONLY) == []


def test_the_second_lane_is_wired_to_self_heal_not_to_silence():
    import yaml

    doc = yaml.safe_load(open(".github/workflows/cronwatch.yml"))
    body = "\n".join(s.get("run", "") for s in doc["jobs"]["sweep"]["steps"])
    assert "--self-heal-only" in body
    assert "--no-dispatch" not in body, (
        "the second lane can see a dropped watchdog and nothing else can; "
        "forbidding it to act leaves that case reported but never recovered")


# ---------------------------------------------------------------------------
# The check was blindest where it mattered most (2026-08-27).
#
# Lateness measured against the most recent EXPECTED slot cannot exceed one
# cadence, because a new slot keeps arriving and resetting it.  So any channel
# firing more often than its own grace window could never be reported missed --
# and the fastest channel here is `watchdog`, the actor for every other one.
#
# That morning its 04:17, 05:17 and 06:17 firings were all dropped.  At 06:40,
# 3 h 23 m with no run at all, the ledger said WITHIN_GRACE.
# ---------------------------------------------------------------------------

def _hourly(file="watchdog.yml"):
    from seti.cronwatch import ScheduledWorkflow

    return ScheduledWorkflow(file=file, name=file.removesuffix(".yml"),
                             crons=["17 * * * *"], has_dispatch=True)


def test_a_sustained_outage_of_an_hourly_channel_is_reported_missed():
    """THE REGRESSION.  Three dropped firings in a row, and the old test slept."""
    from seti.cronwatch import assess

    now = datetime(2026, 8, 27, 6, 40, tzinfo=timezone.utc)
    last = datetime(2026, 8, 27, 3, 16, 49, tzinfo=timezone.utc)
    rec = assess([_hourly()], {"watchdog.yml": last}, now)[0]

    assert rec["status"] == "MISSED", (
        "3 h 23 m with no run read as healthy because a fresh slot arrives "
        "every hour and resets the lateness clock")
    assert rec["overdue"] is True
    assert rec["missed_by"] == "silence"
    assert rec["hours_since_last_run"] == pytest.approx(3.39, abs=0.02)


def test_one_dropped_hourly_firing_is_still_not_an_incident():
    """The new clock must not turn ordinary scheduler drift into noise.

    Measured drift in this repository runs to 151 minutes; a single skipped
    hourly slot has to stay quiet.
    """
    from seti.cronwatch import assess

    now = datetime(2026, 8, 27, 6, 40, tzinfo=timezone.utc)
    last = datetime(2026, 8, 27, 5, 16, 49, tzinfo=timezone.utc)   # 1 h 23 m ago
    rec = assess([_hourly()], {"watchdog.yml": last}, now)[0]
    assert rec["status"] == "WITHIN_GRACE"
    assert rec["overdue"] is False


def test_the_silence_clock_trips_at_one_cadence_plus_grace():
    """Hourly: 1 h cadence + 2 h grace = 3 h, and not a minute sooner."""
    from seti.cronwatch import assess

    now = datetime(2026, 8, 27, 6, 40, tzinfo=timezone.utc)
    just_inside = now - timedelta(hours=2, minutes=58)
    just_outside = now - timedelta(hours=3, minutes=2)
    assert assess([_hourly()], {"watchdog.yml": just_inside}, now)[0]["status"] \
        == "WITHIN_GRACE"
    assert assess([_hourly()], {"watchdog.yml": just_outside}, now)[0]["status"] \
        == "MISSED"


def test_a_slow_channel_is_still_caught_by_the_slot_clock_not_the_silence_one():
    """A dropped WEEKLY firing must be called in 12 h, not in 8 days.

    The silence clock for a weekly channel is 168 h + 12 h; waiting for that
    would report a missed Wednesday the following Thursday, which is useless.
    The original test stays the sensitive one wherever it can speak.
    """
    from seti.cronwatch import ScheduledWorkflow, assess

    wf = ScheduledWorkflow(file="tocsin-altfeeds.yml", name="tocsin-altfeeds",
                           crons=["40 18 * * 3"], has_dispatch=True)
    now = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)     # Thu
    last = datetime(2026, 8, 19, 18, 45, tzinfo=timezone.utc)   # the WEEK BEFORE
    rec = assess([wf], {"tocsin-altfeeds.yml": last}, now)[0]
    assert rec["status"] == "MISSED"
    assert rec["missed_by"] == "grace", (
        "the slot clock should have called this ~17 h after the missed slot, "
        "long before 180 h of silence -- `missed_by` names the clock that "
        "could SEE it, and for slow cadences that is still the slot clock")


def test_a_new_hourly_cron_that_has_never_fired_once_is_caught_too():
    """Otherwise the same blindness returns by another door.

    A brand-new hourly schedule that never fires has no last run to measure
    silence from, and its slot clock resets every hour -- so it would sit at
    WITHIN_GRACE for ever.  Measured from when the schedule appeared instead.
    """
    from seti.cronwatch import assess

    now = datetime(2026, 8, 27, 6, 40, tzinfo=timezone.utc)
    born = datetime(2026, 8, 27, 1, 0, tzinfo=timezone.utc)      # 5 h 40 m ago
    rec = assess([_hourly()], {"watchdog.yml": None}, now,
                 changed_at={"watchdog.yml": born})[0]
    assert rec["status"] == "MISSED" and rec["missed_by"] == "silence"
    assert "hours_since_last_run" not in rec, "there is no last run to report"
    assert "when its schedule appeared" in rec["note"]


def test_a_schedule_too_new_to_have_missed_anything_still_wins():
    """The refusal is decided before either clock, and must stay that way."""
    from seti.cronwatch import assess

    now = datetime(2026, 8, 27, 6, 40, tzinfo=timezone.utc)
    born = datetime(2026, 8, 27, 6, 30, tzinfo=timezone.utc)     # after the slot
    rec = assess([_hourly()], {"watchdog.yml": None}, now,
                 changed_at={"watchdog.yml": born})[0]
    assert rec["status"] == "SCHEDULE_TOO_NEW" and rec["overdue"] is False


def test_silence_never_overrides_the_refusals():
    """UNKNOWN and SCHEDULE_TOO_NEW still win: they are decided earlier."""
    from seti.cronwatch import assess

    now = datetime(2026, 8, 27, 6, 40, tzinfo=timezone.utc)
    long_ago = datetime(2026, 8, 20, tzinfo=timezone.utc)
    unknown = assess([_hourly()], {"watchdog.yml": long_ago}, now,
                     unknown={"watchdog.yml"})[0]
    assert unknown["status"] == "UNKNOWN" and unknown["overdue"] is False

    born = datetime(2026, 8, 27, 6, 30, tzinfo=timezone.utc)   # after the slot
    fresh = assess([_hourly()], {"watchdog.yml": long_ago}, now,
                   changed_at={"watchdog.yml": born})[0]
    assert fresh["status"] == "SCHEDULE_TOO_NEW" and fresh["overdue"] is False
