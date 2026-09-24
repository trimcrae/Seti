"""The workflow files themselves, checked the way GitHub checks them.

WHY THIS EXISTS.  On 2026-08-26 an edit to `alerts.yml` left a step with two
`env:` keys.  PyYAML accepts that silently -- last one wins -- so the local
check that parsed the file and printed its step names passed.  GitHub's parser
does not: it rejected the whole file, and every `alerts` run from 21:54 UTC
onward failed in zero seconds with the workflow named by its path rather than
its `name:`, which is what a startup failure looks like.  The alerting channel
was down and the only symptom was four instant red runs.

So these tests read the workflows the strict way, and assert the handful of
properties the rest of this repository depends on being true of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOWS = sorted(Path(".github/workflows").glob("*.yml"))


class StrictLoader(yaml.SafeLoader):
    """A loader that refuses duplicate mapping keys, as GitHub's does."""


def _no_duplicates(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping", node.start_mark,
                f"found duplicate key {key!r}", key_node.start_mark)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


def test_there_are_workflows_to_check():
    """Guard the guard: a bad glob would make every test below vacuous."""
    assert len(WORKFLOWS) > 20


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_a_workflow_has_no_duplicate_keys(path):
    """The exact failure of 2026-08-26: two `env:` blocks in one step."""
    yaml.load(path.read_text(), Loader=StrictLoader)


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_a_workflow_declares_a_name_a_trigger_and_a_job(path):
    doc = yaml.load(path.read_text(), Loader=StrictLoader)
    assert isinstance(doc, dict), path.name
    assert doc.get("name"), f"{path.name} has no name:"
    # `on:` is the BOOLEAN True after YAML 1.1 parsing -- the same trap
    # `seti.cronwatch.read_schedules` has to handle.
    triggers = doc.get(True, doc.get("on"))
    assert isinstance(triggers, dict) and triggers, f"{path.name} has no triggers"
    assert doc.get("jobs"), f"{path.name} has no jobs"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_a_scheduled_workflow_can_also_be_dispatched(path):
    """A dropped cron has to be recoverable.

    `seti.cronwatch` re-fires a firing GitHub dropped, and it can only do that
    through `workflow_dispatch`.  A scheduled workflow without one can lose a
    week of data with no way back short of a human editing the file.
    """
    doc = yaml.load(path.read_text(), Loader=StrictLoader)
    triggers = doc.get(True, doc.get("on")) or {}
    if "schedule" not in triggers:
        pytest.skip("not scheduled")
    assert "workflow_dispatch" in triggers, (
        f"{path.name} is scheduled but cannot be dispatched, so a dropped "
        f"firing cannot be recovered")


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_a_schedule_is_a_cron_this_repository_can_reason_about(path):
    """Every cron must parse under `seti.cronwatch`, which watches them.

    A cron GitHub accepts but the watch cannot read would be dropped from the
    watch list silently -- the channel would look supervised and not be.
    """
    from seti.cronwatch import parse_cron

    doc = yaml.load(path.read_text(), Loader=StrictLoader)
    triggers = doc.get(True, doc.get("on")) or {}
    entries = triggers.get("schedule") or []
    if not entries:
        pytest.skip("not scheduled")
    for entry in entries:
        parse_cron(entry["cron"])


def test_the_altfeeds_job_deadline_matches_its_timeout():
    """`seti.tocsin.altwalk` clips every survey's fetch budget to a job-wide
    deadline the workflow derives from its own timeout.  If the two numbers
    drift apart the clip is wrong in one of two directions: too short wastes
    runner time, too long recreates the 2026-09-02 cancellation."""
    import re

    path = Path(".github/workflows/tocsin-altfeeds.yml")
    doc = yaml.load(path.read_text(), Loader=StrictLoader)
    job = doc["jobs"]["altfeeds"]
    timeout_min = int(job["timeout-minutes"])
    runs = [s.get("run") or "" for s in job["steps"]]
    declared = [int(m.group(1)) for r in runs
                for m in re.finditer(r"^\s*TIMEOUT_MIN=(\d+)\s*$", r, re.M)]
    assert declared, "no step sets TIMEOUT_MIN for the job-wide deadline"
    assert declared == [timeout_min], (
        f"TIMEOUT_MIN {declared} must equal timeout-minutes {timeout_min}")
    assert any("ALTFEEDS_JOB_DEADLINE_UNIX" in r for r in runs)


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_a_lane_that_can_dispatch_is_allowed_to(path):
    """A sweep that may re-fire a dropped cron needs `actions: write`.

    THE FAILURE THIS PINS.  `cronwatch.yml` was written as a pure reporter, so
    `actions: read` was correct for it.  `--self-heal-only` was then added to
    that same job -- the one dispatch `watchdog` cannot make for itself, because
    the case is that `watchdog` did not run -- and the permission block was left
    behind.  The result reports normally and fails only at the single API call
    it exists to make: on 2026-09-15 `watchdog` missed its 16:17 UTC firing,
    this lane saw it, and the catch-up came back

        dispatch watchdog.yml -> 403 {"message":"Resource not accessible by
        integration"}

    so the self-heal that had been added three weeks earlier had never once
    worked.  `actions: read` on a dispatching lane is not a smaller permission,
    it is a broken one.
    """
    doc = yaml.load(path.read_text(), Loader=StrictLoader)
    runs = [step.get("run") or ""
            for job in (doc.get("jobs") or {}).values()
            for step in (job.get("steps") or [])]
    # The dispatching callers: the scheduler sweep unless it is told not to
    # dispatch, and the failure sweep, which re-runs jobs through the same
    # `actions` scope.
    dispatches = any(
        ("cron_watch.py" in r and "--no-dispatch" not in r)
        or "fail_sweep.py" in r
        for r in runs)
    if not dispatches:
        pytest.skip("this workflow neither dispatches nor re-runs")
    actions = ((doc.get("permissions") or {}).get("actions"))
    assert actions == "write", (
        f"{path.name} re-fires or re-runs workflows but grants "
        f"`actions: {actions}` -- the POST comes back 403 "
        f"'Resource not accessible by integration'")


# ---------------------------------------------------------------------------
# Shard hygiene (docs/channel-brief.md §0.7)
# ---------------------------------------------------------------------------
def _purged_paths(steps) -> list[str]:
    """Every path an `rm -f` / `rm -rf` in these steps' `run:` names."""
    import shlex

    out: list[str] = []
    for step in steps:
        run = (step.get("run") or "").replace("\\\n", " ")
        for line in run.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line.startswith(("rm ", "sudo rm ")):
                continue
            try:
                words = shlex.split(line, posix=True)
            except ValueError:
                continue
            out += [w.rstrip("/") for w in words
                    if w not in ("rm", "sudo") and not w.startswith("-")
                    and w not in ("||", "true", "2>/dev/null")]
    return out


def _covered(path: str, purged: list[str]) -> bool:
    import fnmatch

    p = path.rstrip("/")
    return any(p == t or p.startswith(t + "/") or fnmatch.fnmatch(p, t)
               for t in purged)


def _shard_upload_offenders(doc) -> list[str]:
    offenders = []
    for job_name, job in (doc.get("jobs") or {}).items():
        if "matrix" not in (job.get("strategy") or {}):
            continue
        steps = job.get("steps") or []
        for k, step in enumerate(steps):
            if "actions/upload-artifact" not in str(step.get("uses", "")):
                continue
            purged = _purged_paths(steps[:k])
            for raw in str((step.get("with") or {}).get("path", "")).splitlines():
                path = raw.strip()
                if not path or path.startswith("!"):
                    continue
                # Shard-scoped by construction: named from the matrix / a step
                # that derives the shard's own names, or staged outside the
                # checkout (SPECTRA-PERSIST's and LANTERN's `runner.temp`).
                if "${{" in path:
                    continue
                if not _covered(path, purged):
                    offenders.append(f"{job_name}: {path}")
    return offenders


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_a_shard_uploads_only_what_it_wrote(path):
    """A matrix job must not upload files its checkout put there.

    THE FAILURE THIS PINS, twice.  SPECTRA-PERSIST run 35758868818: each shard
    uploaded its whole checkpoint directory, including checkpoints checked out
    from the branch, and the reduce merged the artifacts concurrently into one
    folder -- 14 corrupted JSON, 35 stale copies.  GROWTH-direct run
    35859780689: each shard uploaded shard_*.csv, i.e. every OTHER shard's file
    as checked out at job start, and assess merged with merge-multiple, so an
    early shard's stale copies overwrote fresher results (4,358 rows reported,
    4,725 on the branch).

    THE HEURISTIC.  For every upload in a matrix job, each literal path (no
    `${{ ... }}` in it) must be covered by an `rm -f`/`rm -rf` in an earlier
    step of the same job -- the SEXTANT "Purge checkout-inherited shard
    outputs" step.  A path carrying an expression is taken to be shard-scoped
    (named by the matrix value, by a step that derives the shard's own file
    names, or staged under `runner.temp`); that is a convention, not a proof,
    and review is still what checks the expression really names only the
    shard's own files.  Paths are checked whether or not anything is tracked
    there today: a commit step can put files there tomorrow.
    """
    doc = yaml.load(path.read_text(), Loader=StrictLoader)
    offenders = _shard_upload_offenders(doc)
    assert not offenders, (
        f"{path.name}: a matrix job uploads a checkout path it never purged "
        f"-- stale copies of other shards' outputs ride along into the merge "
        f"(docs/channel-brief.md §0.7): {offenders}")


def test_the_shard_hygiene_check_catches_the_growth_direct_defect():
    """Guard the guard: the pre-fix GROWTH-direct shape must be flagged, and
    the SEXTANT reference fix must not be."""
    broken = yaml.safe_load("""
jobs:
  measure:
    strategy: {matrix: {shard: [0, 1]}}
    steps:
      - uses: actions/checkout@v4
      - run: python -m seti.growth.direct --stage measure
      - uses: actions/upload-artifact@v4
        with: {name: s, path: "results/growth/direct/shards/shard_*.csv"}
""")
    fixed = yaml.safe_load("""
jobs:
  fit:
    strategy: {matrix: {shard: [0, 1]}}
    steps:
      - uses: actions/checkout@v4
      - name: Purge checkout-inherited shard outputs
        run: rm -rf results/sextant/fits
      - run: python -m seti.cli sextant --stage fit
      - uses: actions/upload-artifact@v4
        with: {name: f, path: results/sextant/fits/}
""")
    assert _shard_upload_offenders(broken) == [
        "measure: results/growth/direct/shards/shard_*.csv"]
    assert _shard_upload_offenders(fixed) == []
