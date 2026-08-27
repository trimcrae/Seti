"""The failure sweep's decisions, including the one that cost a runner lane.

The first test in this file is the incident of 2026-08-27 written down as a
regression: run 33022081059 failed at commit `1de2f34`, the bug was fixed and
merged, `main` moved six commits on, and the hourly sweep re-ran it anyway --
starting the removed code on a four-hour lane ahead of two runs carrying the
fix.  See the `seti.failsweep` module docstring.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from seti.failsweep import (
    NEVER_RETRY,
    RETRY_CEILING,
    plan,
    read_skiplist,
    resolve_branch_heads,
    sweep,
)

NOW = datetime(2026, 8, 27, 3, 17, tzinfo=timezone.utc)


def run(**kw) -> dict:
    base = {"id": 1, "name": "tocsin-altfeeds", "head_branch": "main",
            "head_sha": "a" * 40, "conclusion": "failure", "status": "completed",
            "run_attempt": 1, "html_url": "https://example/1",
            "created_at": (NOW - timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    base.update(kw)
    return base


def only(report, key="retry"):
    return [r["id"] for r in report[key]]


def reason_for(report, run_id):
    return next(r["reason"] for r in report["skipped"] if r["id"] == run_id)


# --------------------------------------------------------------------------
# The incident.
# --------------------------------------------------------------------------

def test_a_failed_run_whose_commit_is_no_longer_the_branch_head_is_not_retried():
    """THE REGRESSION.  A re-run checks out the run's own commit, not the head.

    So re-running a failure whose cause has since been fixed and merged can
    only reproduce the failure -- or half-succeed and commit the superseded
    code's results over the fresh ones.
    """
    old = run(id=33022081059, head_sha="1de2f34" + "0" * 33)
    rep = plan([old], branch_heads={"main": "703d465" + "0" * 33}, now=NOW)
    assert only(rep) == []
    assert reason_for(rep, 33022081059) == "superseded"
    # And it says which two commits, because a bare "skipped" teaches nobody.
    note = next(r["note"] for r in rep["skipped"] if r["id"] == 33022081059)
    assert "1de2f34" in note and "703d465" in note


def test_the_same_run_IS_retried_while_its_commit_is_still_the_head():
    """The rule must not simply switch the sweep off.

    A failure at the current head is exactly what auto-retry is for: a flaky
    network call, a runner that vanished, a transient 502.
    """
    cur = run(id=7, head_sha="c" * 40)
    rep = plan([cur], branch_heads={"main": "c" * 40}, now=NOW)
    assert only(rep) == [7]
    assert rep["skipped"] == []


def test_a_feature_branch_is_judged_against_its_own_head_not_main():
    """A run on a branch is superseded when THAT branch moves, not when main does."""
    on_branch = run(id=8, head_branch="claude/work", head_sha="d" * 40)
    heads = {"claude/work": "d" * 40, "main": "e" * 40}
    assert only(plan([on_branch], branch_heads=heads, now=NOW)) == [8]
    heads["claude/work"] = "f" * 40
    assert reason_for(plan([on_branch], branch_heads=heads, now=NOW), 8) == "superseded"


# --------------------------------------------------------------------------
# The refusals.
# --------------------------------------------------------------------------

def test_an_unreadable_branch_head_refuses_rather_than_guesses():
    """Unknown is not permission -- the same call `cronwatch` makes for a dead API."""
    orphan = run(id=9, head_branch="deleted-branch")
    rep = plan([orphan], branch_heads={"deleted-branch": None}, now=NOW)
    assert only(rep) == []
    assert reason_for(rep, 9) == "branch_head_unknown"


def test_a_branch_missing_from_the_map_is_also_unknown_not_retryable():
    """The distinction matters: `.get(b)` returning None must not read as a match."""
    rep = plan([run(id=10, head_branch="never-resolved")], branch_heads={}, now=NOW)
    assert reason_for(rep, 10) == "branch_head_unknown"


def test_a_run_still_in_flight_is_not_asked_for_another_attempt():
    """An in-progress attempt 2 is not a failure waiting for attempt 3."""
    live = run(id=11, status="in_progress", conclusion="failure", run_attempt=2)
    rep = plan([live], branch_heads={"main": "a" * 40}, now=NOW)
    assert only(rep) == []
    assert reason_for(rep, 11) == "still_running"


@pytest.mark.parametrize("conclusion,attempt,retried", [
    ("failure", 1, True), ("failure", 2, True), ("failure", 3, False),
    ("timed_out", 2, True), ("timed_out", 3, False),
    ("cancelled", 1, True), ("cancelled", 2, False),
])
def test_the_attempt_ceiling_depends_on_how_the_run_ended(conclusion, attempt, retried):
    """A cancel gets one retry, not two: it is often a human saying stop."""
    rep = plan([run(id=12, conclusion=conclusion, run_attempt=attempt)],
               branch_heads={"main": "a" * 40}, now=NOW)
    assert (only(rep) == [12]) is retried
    if not retried:
        assert reason_for(rep, 12) == "retries_exhausted"
    assert RETRY_CEILING[conclusion] in (2, 3)


def test_ci_and_watchdog_are_never_retried_and_never_even_listed():
    """Retrying a red test suite argues with the signal; retrying the sweep spins."""
    runs = [run(id=13, name=n) for n in NEVER_RETRY]
    rep = plan(runs, branch_heads={"main": "a" * 40}, now=NOW)
    assert rep["retry"] == [] and rep["skipped"] == []
    assert rep["n_considered"] == 0


def test_a_success_is_not_a_failure_and_a_stale_failure_is_out_of_window():
    old = run(id=14, created_at=(NOW - timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    good = run(id=15, conclusion="success")
    rep = plan([old, good], branch_heads={"main": "a" * 40}, now=NOW)
    assert rep["n_considered"] == 0 and rep["retry"] == []


def test_the_skiplist_still_holds_for_reasons_a_sha_cannot_express():
    rep = plan([run(id=16)], branch_heads={"main": "a" * 40}, now=NOW,
               skiplist=[16])
    assert reason_for(rep, 16) == "skiplisted"


def test_a_sweep_will_not_fire_an_unbounded_number_of_retries():
    """Twenty failures at once is a GitHub-wide problem, not twenty bugs."""
    runs = [run(id=100 + i) for i in range(9)]
    rep = plan(runs, branch_heads={"main": "a" * 40}, now=NOW, max_retries=5)
    assert len(rep["retry"]) == 5
    over = [r for r in rep["skipped"] if r["reason"] == "over_sweep_budget"]
    assert len(over) == 4


# --------------------------------------------------------------------------
# Every decision is recorded.  Silence is how the incident happened.
# --------------------------------------------------------------------------

def test_every_considered_run_appears_in_exactly_one_of_the_two_lists():
    runs = [run(id=20), run(id=21, head_sha="z" * 40), run(id=22, run_attempt=3),
            run(id=23, status="queued"), run(id=24, head_branch="gone")]
    rep = plan(runs, branch_heads={"main": "a" * 40, "gone": None}, now=NOW)
    seen = only(rep) + only(rep, "skipped")
    assert sorted(seen) == [20, 21, 22, 23, 24]
    assert rep["n_considered"] == 5
    assert all(r.get("reason") for r in rep["skipped"])


# --------------------------------------------------------------------------
# The driver, against a fake API.
# --------------------------------------------------------------------------

class FakeApi:
    def __init__(self, runs, heads):
        self.runs, self.heads = runs, heads
        self.reran: list[int] = []
        self.head_calls: list[str] = []

    def recent_runs(self, per_page=100):
        return list(self.runs)

    def branch_head(self, branch):
        self.head_calls.append(branch)
        if branch not in self.heads:
            raise RuntimeError("boom")
        return self.heads[branch]

    def rerun_failed_jobs(self, run_id):
        self.reran.append(int(run_id))


def test_the_sweep_retries_the_live_failure_and_leaves_the_superseded_one(tmp_path):
    api = FakeApi([run(id=30, head_sha="a" * 40), run(id=31, head_sha="b" * 40)],
                  {"main": "a" * 40})
    rep = sweep(tmp_path, api=api, now=NOW)
    assert api.reran == [30]
    assert reason_for(rep, 31) == "superseded"
    assert rep["retried"] == 1


def test_a_branch_head_call_that_raises_becomes_unknown_not_a_crash(tmp_path):
    api = FakeApi([run(id=32, head_branch="explodes")], {})
    rep = sweep(tmp_path, api=api, now=NOW)
    assert api.reran == []
    assert reason_for(rep, 32) == "branch_head_unknown"


def test_no_retry_plans_the_same_thing_and_dispatches_none(tmp_path):
    api = FakeApi([run(id=33, head_sha="a" * 40)], {"main": "a" * 40})
    rep = sweep(tmp_path, api=api, now=NOW, retry=False)
    assert api.reran == []
    assert only(rep) == [33] and rep["retried"] == 0


def test_the_ledger_is_written_even_when_nothing_failed(tmp_path):
    """A file that appears only on bad days cannot say 'the sweep ran and all
    was well' -- which is the exact ambiguity docs/cronwatch.md is about."""
    api = FakeApi([run(id=34, conclusion="success")], {"main": "a" * 40})
    sweep(tmp_path, api=api, now=NOW)
    written = json.loads((tmp_path / "results" / "watchdog" / "status.json").read_text())
    assert written["n_considered"] == 0
    assert written["checked_at_utc"] == "2026-08-27T03:17:00Z"


def test_a_sweep_with_no_api_says_so_instead_of_reporting_health(tmp_path):
    rep = sweep(tmp_path, api=None, now=NOW)
    assert rep["no_api"] is True and rep["n_considered"] == 0


def test_branch_heads_are_resolved_once_per_branch_not_once_per_run():
    api = FakeApi([], {"main": "a" * 40})
    resolve_branch_heads(api, ["main", "main", "main"])
    assert api.head_calls == ["main"]


def test_a_missing_or_corrupt_skiplist_is_empty_rather_than_fatal(tmp_path):
    assert read_skiplist(tmp_path) == []
    d = tmp_path / "results" / "watchdog"
    d.mkdir(parents=True)
    (d / "skiplist.json").write_text("{not json")
    assert read_skiplist(tmp_path) == []
    (d / "skiplist.json").write_text("[1, 2]")
    assert read_skiplist(tmp_path) == [1, 2]


# --------------------------------------------------------------------------
# The wiring.  A tested module the workflow does not call is not a fix.
# --------------------------------------------------------------------------

def test_the_watchdog_runs_the_tested_sweep_and_not_inline_jq():
    """The whole point of the rewrite.

    The retry rules lived in `gh api | jq` inside the workflow file, where no
    test could reach them, and the first time anyone found out what they
    actually did was when they resurrected a superseded run.  If they migrate
    back into shell, this fails.
    """
    import yaml

    doc = yaml.safe_load(open(".github/workflows/watchdog.yml"))
    steps = doc["jobs"]["sweep"]["steps"]
    body = "\n".join(s.get("run", "") for s in steps)
    assert "scripts/fail_sweep.py" in body
    assert "scripts/cron_watch.py" in body
    assert "rerun-failed-jobs" not in body, (
        "the retry decision is back in shell, where nothing tests it")


def test_the_watchdog_can_write_the_ledgers_it_is_asked_to_commit():
    """`contents: write` or the ledger silently never lands."""
    import yaml

    doc = yaml.safe_load(open(".github/workflows/watchdog.yml"))
    assert doc["permissions"]["contents"] == "write"
    # `actions: write` is what lets it re-run a job and dispatch a catch-up.
    assert doc["permissions"]["actions"] == "write"
