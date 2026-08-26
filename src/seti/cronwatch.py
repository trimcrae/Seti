"""Does the scheduler actually fire the screens?  Ask GitHub, and re-fire what it dropped.

WHY THIS MODULE EXISTS.  Every liveness check in this repository asks whether a
channel's *results* are fresh (``alerts.STALE_DAYS``) or whether a run *failed*
(``.github/workflows/watchdog.yml``).  Neither one sees the failure that actually
happens most often on GitHub's hosted scheduler: **the run never starts at all.**
A cron that does not fire produces no run to fail, no log to read and no result
to age -- until the staleness window expires days later, by which time the
missing night is unrecoverable for any feed with a rolling archive.

It is not hypothetical.  On 2026-08-26 ``tocsin-altfeeds`` did not fire at its
first scheduled slot (Wednesdays 18:40 UTC); at the same time ``watchdog``,
nominally hourly at :17, fired at 19:16, 16:33, 14:02, 13:13, 11:42, 10:50,
09:55 and 09:00 UTC -- a scheduler that delays firings by hours and drops some
outright.  GitHub documents this: scheduled workflows are best effort and are
dropped under load.

WHAT IT DOES.  Reads the crons out of the workflow files themselves (never a
hand-kept list, which drifts silently), asks the Actions API when each workflow
last ran *on a schedule event*, and compares that against the firing the cron
says should already have happened.  Anything older than its own grace window is
reported -- and, when the workflow accepts ``workflow_dispatch``, re-fired once,
because an alert nobody reads for a week does not screen the sky.

WHAT IT DELIBERATELY DOES NOT DO.
  * It does not re-fire a workflow twice for the same missed firing.  The state
    file is keyed by (workflow, missed firing), so a catch-up that itself fails
    is a failure for ``watchdog`` to retry, not a loop for this to spin.
  * It does not treat lateness as absence.  The grace window is a quarter of the
    channel's own cadence (two hours minimum, a day maximum), so the ordinary
    hour-scale drift above never fires it and a genuinely dropped firing does.
  * It does not claim a workflow is dead when the API could not be reached.  No
    answer is recorded as ``UNKNOWN`` and never as overdue: a monitor that cries
    wolf on its own transport failure gets muted, and then it is not a monitor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Grace: how long after a firing SHOULD have happened before its absence is a
# finding.  A quarter of the cadence, floored at two hours (GitHub's ordinary
# drift is hour-scale, and an hourly cron that fires 40 minutes late is not an
# incident) and capped at half a day.  The cap matters most for the weekly and
# monthly channels: a quarter of a week is 42 hours, and a weekly screen that
# waits two days to admit it never ran has lost a third of the interval before
# anyone can act.  Twelve hours sits well clear of the worst drift measured here
# (2 h 09 m, `alerts` on 2026-08-26) and well inside every cadence in the repo.
MIN_GRACE = timedelta(hours=2)
MAX_GRACE = timedelta(hours=12)
GRACE_FRACTION = 0.25

# How many catch-up dispatches one sweep may issue.  A cap, not a target: if the
# scheduler has dropped six channels at once the problem is GitHub-wide and
# firing six jobs into it is not the answer.
MAX_CATCHUPS_PER_SWEEP = 3

DEFAULT_RESULTS_DIR = "results/cronwatch"


# ---------------------------------------------------------------------------
# 1. cron arithmetic
#
# Only what GitHub accepts: five space-separated fields, UTC, with `*`, lists,
# ranges and steps.  No @weekly aliases (GitHub rejects them), no seconds field.
# ---------------------------------------------------------------------------
# Day-of-week runs 0-7 because cron accepts BOTH 0 and 7 for Sunday; a range
# of 0-6 would reject a legal `* * * * 7` schedule as malformed and drop that
# workflow out of the watch entirely.  `parse_cron` folds 7 back onto 0.
FIELD_RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))


def _parse_field(spec: str, lo: int, hi: int) -> set[int]:
    """Expand one cron field into the set of values it matches."""
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty cron field element in {spec!r}")
        step = 1
        if "/" in part:
            part, _, step_s = part.partition("/")
            step = int(step_s)
            if step <= 0:
                raise ValueError(f"non-positive step in {spec!r}")
        if part in ("*", "?"):
            start, end = lo, hi
        elif "-" in part.lstrip("-"):
            a, _, b = part.partition("-")
            start, end = int(a), int(b)
        else:
            start = end = int(part)
        if start < lo or end > hi or end < start:
            raise ValueError(f"cron field {part!r} out of range [{lo},{hi}]")
        out |= set(range(start, end + 1, step))
    if not out:
        raise ValueError(f"cron field {spec!r} matches nothing")
    return out


@dataclass(frozen=True)
class Cron:
    """A parsed five-field cron expression, evaluated in UTC."""

    expr: str
    minutes: frozenset[int]
    hours: frozenset[int]
    doms: frozenset[int]
    months: frozenset[int]
    dows: frozenset[int]
    dom_restricted: bool
    dow_restricted: bool

    def matches_day(self, day: datetime) -> bool:
        """Does this cron fire at all on ``day``?

        Vixie semantics, which GitHub follows: when BOTH day-of-month and
        day-of-week are restricted the two are OR-ed, not AND-ed.  Getting this
        backwards would silently make a `0 0 1 * 1` schedule look like it never
        fires, and this module would then report a live channel as dropped.
        """
        if day.month not in self.months:
            return False
        dow = (day.weekday() + 1) % 7          # python Mon=0 -> cron Sun=0
        if self.dom_restricted and self.dow_restricted:
            return day.day in self.doms or dow in self.dows
        if self.dom_restricted:
            return day.day in self.doms
        if self.dow_restricted:
            return dow in self.dows
        return True


def parse_cron(expr: str) -> Cron:
    fields = str(expr).split()
    if len(fields) != 5:
        raise ValueError(f"expected 5 cron fields, got {len(fields)}: {expr!r}")
    minute, hour, dom, month, dow = fields
    sets = [_parse_field(f, lo, hi)
            for f, (lo, hi) in zip(fields, FIELD_RANGES, strict=True)]
    dows = {d % 7 for d in sets[4]}            # cron allows 7 for Sunday
    return Cron(expr=str(expr).strip(), minutes=frozenset(sets[0]),
                hours=frozenset(sets[1]), doms=frozenset(sets[2]),
                months=frozenset(sets[3]), dows=frozenset(dows),
                dom_restricted=dom.strip() not in ("*", "?"),
                dow_restricted=dow.strip() not in ("*", "?"))


def prev_fire(cron: Cron | str, now: datetime, *, horizon_days: int = 400
              ) -> datetime | None:
    """The most recent firing at or before ``now``, or None within the horizon.

    Walked day by day rather than minute by minute: a monthly cron would need
    forty thousand minute steps to find its own last firing, and this runs on
    every hourly sweep over every scheduled workflow in the repository.
    """
    c = cron if isinstance(cron, Cron) else parse_cron(cron)
    now = now.astimezone(timezone.utc).replace(second=0, microsecond=0)
    day = now
    for offset in range(horizon_days + 1):
        day = (now - timedelta(days=offset)).replace(hour=0, minute=0)
        if not c.matches_day(day):
            continue
        limit_h, limit_m = (now.hour, now.minute) if offset == 0 else (23, 59)
        for h in sorted(c.hours, reverse=True):
            if h > limit_h:
                continue
            mins = [m for m in c.minutes if not (h == limit_h and m > limit_m)]
            if mins:
                return day.replace(hour=h, minute=max(mins))
    return None


def cadence(cron: Cron | str, at: datetime) -> timedelta | None:
    """How long between the firing at ``at`` and the one before it."""
    c = cron if isinstance(cron, Cron) else parse_cron(cron)
    this = prev_fire(c, at)
    if this is None:
        return None
    before = prev_fire(c, this - timedelta(minutes=1))
    return None if before is None else this - before


def grace_for(cadence_: timedelta | None) -> timedelta:
    if cadence_ is None:
        return MAX_GRACE
    return max(MIN_GRACE, min(MAX_GRACE, cadence_ * GRACE_FRACTION))


# ---------------------------------------------------------------------------
# 2. the schedule registry, read from the workflows themselves
# ---------------------------------------------------------------------------
@dataclass
class ScheduledWorkflow:
    """One workflow file's schedule, as the repository actually declares it."""

    file: str
    name: str
    crons: list[str] = field(default_factory=list)
    has_dispatch: bool = False


def _on_block(doc: dict) -> dict:
    """The workflow's trigger block.

    YAML 1.1 -- which PyYAML implements -- reads a bare ``on:`` key as the
    BOOLEAN True, so `doc["on"]` misses it on every real workflow file.  A
    registry that silently finds no schedules would leave this module reporting
    a perfectly healthy repository forever.
    """
    for key in (True, "on", "On", "ON"):
        block = doc.get(key)
        if isinstance(block, dict):
            return block
    return {}


def read_schedules(root: Path | str = ".") -> list[ScheduledWorkflow]:
    """Every scheduled workflow in ``.github/workflows``, with its crons.

    Read from the files rather than a list kept here on purpose: a hand-kept
    registry is one rename away from watching a workflow that no longer exists
    while ignoring the one that replaced it, and it would look correct.
    """
    import yaml

    out: list[ScheduledWorkflow] = []
    wf_dir = Path(root) / ".github" / "workflows"
    for path in sorted(wf_dir.glob("*.yml")) + sorted(wf_dir.glob("*.yaml")):
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except Exception:                                      # noqa: BLE001
            continue
        if not isinstance(doc, dict):
            continue
        on = _on_block(doc)
        sched = on.get("schedule") or []
        crons = [str(e.get("cron")).strip() for e in sched
                 if isinstance(e, dict) and e.get("cron")]
        if not crons:
            continue
        out.append(ScheduledWorkflow(
            file=path.name, name=str(doc.get("name") or path.stem),
            crons=crons, has_dispatch="workflow_dispatch" in on))
    return out


# ---------------------------------------------------------------------------
# 3. the finding
# ---------------------------------------------------------------------------
def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def assess(workflows: list[ScheduledWorkflow], last_runs: dict[str, datetime | None],
           now: datetime, *, unknown: frozenset[str] | set[str] = frozenset(),
           changed_at: dict[str, datetime | None] | None = None) -> list[dict]:
    """Compare each workflow's last scheduled run against what its cron promised.

    ``last_runs`` maps workflow file name to the start of its most recent run
    with ``event == schedule`` (None when it has never had one).  ``unknown``
    names the files whose history could NOT be read; those are reported with
    status UNKNOWN and never as overdue -- see the module docstring.

    ``changed_at`` maps each file to when it last changed on the default branch,
    and exists to stop this module blaming a schedule for slots it did not have.
    A cron added on a Tuesday has a "last firing" the previous Monday that could
    not possibly have happened -- and a monthly channel added on the 25th looks
    three weeks overdue the moment it is merged.  Reported as SCHEDULE_TOO_NEW,
    never as MISSED, and never re-fired: the first real test of a new schedule is
    its first real slot.
    """
    findings: list[dict] = []
    for wf in workflows:
        # A workflow may carry several crons.  The relevant one is whichever
        # fired most recently: the promise the repository has already made.
        expected: datetime | None = None
        expected_cron = None
        for expr in wf.crons:
            try:
                fire = prev_fire(expr, now)
            except ValueError:
                continue
            if fire is not None and (expected is None or fire > expected):
                expected, expected_cron = fire, expr
        cad = cadence(expected_cron, now) if expected_cron else None
        grace = grace_for(cad)
        last = last_runs.get(wf.file)
        rec = {"workflow": wf.file, "name": wf.name, "crons": list(wf.crons),
               "cron_matched": expected_cron,
               "cadence_hours": None if cad is None else round(
                   cad.total_seconds() / 3600.0, 3),
               "grace_hours": round(grace.total_seconds() / 3600.0, 3),
               "expected_last_fire_utc": _iso(expected),
               "last_scheduled_run_utc": _iso(last),
               "has_dispatch": wf.has_dispatch,
               "status": "OK", "overdue": False, "hours_late": None}
        if wf.file in unknown:
            rec["status"] = "UNKNOWN"
            rec["note"] = ("the Actions API did not answer for this workflow; "
                           "absence of an answer is not absence of a run")
            findings.append(rec)
            continue
        if expected is None:
            rec["status"] = "NO_FIRING_IN_HORIZON"
            findings.append(rec)
            continue
        # SATISFIED FIRST.  A run that already covers the expected firing settles
        # the question, and asking it before anything else keeps a workflow that
        # fired normally from being labelled by an unrelated edit to its file --
        # `tocsin-altfeeds` fired its 18:40 slot at 21:12 on 2026-08-26 and was
        # then reported SCHEDULE_TOO_NEW purely because the file had been touched
        # since, for a concurrency change that has nothing to do with its cron.
        if last is not None and last >= expected:
            findings.append(rec)
            continue
        born = (changed_at or {}).get(wf.file)
        if born is not None and expected < born:
            rec["status"] = "SCHEDULE_TOO_NEW"
            rec["schedule_changed_at_utc"] = _iso(born)
            rec["note"] = (
                f"the firing at {_iso(expected)} predates the last change to "
                f"{wf.file} ({_iso(born)}), so this schedule never had that slot")
            findings.append(rec)
            continue
        if now - expected <= grace:
            # Too soon to call: the firing is due but GitHub's scheduler runs
            # late as a matter of course.
            rec["status"] = "WITHIN_GRACE"
            findings.append(rec)
            continue
        rec["status"] = "MISSED"
        rec["overdue"] = True
        rec["hours_late"] = round((now - expected).total_seconds() / 3600.0, 2)
        rec["note"] = (
            f"{wf.name} should have fired at {_iso(expected)} "
            f"(cron {expected_cron!r}) and its last scheduled run was "
            f"{_iso(last) or 'never'}.  GitHub drops scheduled firings under "
            f"load; this is that, not a failure -- there is no failed run to "
            f"retry, which is why the hourly failure sweep cannot see it.")
        findings.append(rec)
    return findings


# ---------------------------------------------------------------------------
# 4. catch-up
# ---------------------------------------------------------------------------
def _catchup_key(rec: dict) -> str:
    return f"{rec['workflow']}@{rec['expected_last_fire_utc']}"


def plan_catchup(findings: list[dict], state: dict, *,
                 max_catchups: int = MAX_CATCHUPS_PER_SWEEP) -> list[dict]:
    """Which missed firings this sweep should re-fire.

    One dispatch per missed firing, ever.  A catch-up that fails leaves a failed
    run, and a failed run is the failure sweep's business; re-dispatching it here
    would be two monitors fighting over the same job.
    """
    done = set((state or {}).get("caught_up") or {})
    out = []
    for rec in findings:
        if not rec.get("overdue") or not rec.get("has_dispatch"):
            continue
        if _catchup_key(rec) in done:
            continue
        out.append(rec)
        if len(out) >= max_catchups:
            break
    return out


# ---------------------------------------------------------------------------
# The gate: is the repository's own test workflow green on the default branch?
#
# WHY THIS LIVES HERE.  `watchdog`'s failure sweep deliberately skips `ci` --
# `select(.name != "watchdog" and .name != "ci")` -- because auto-retrying a
# failing test suite is fighting the signal rather than reading it.  That is the
# right call about RETRYING and it left nobody watching: CI was red on `main`
# from 2026-07-31 until 2026-08-22, twenty-two days, over a single lint error,
# and it was found by accident.  Every screen in this repository is defended by
# tests that nothing was checking still passed.
#
# So: not retried, but read, and reported through the same alert path as
# everything else.
GATE_WORKFLOW = "ci.yml"


def gate_status(api, workflow_file: str = GATE_WORKFLOW,
                branch: str = "main") -> dict:
    """The newest completed run of the test workflow on the default branch."""
    rec: dict = {"workflow": workflow_file, "branch": branch,
                 "status": "UNKNOWN"}
    if api is None or not hasattr(api, "latest_run"):
        rec["note"] = "no API available; the gate was not read"
        return rec
    try:
        run = api.latest_run(workflow_file, branch=branch)
    except Exception as exc:                                       # noqa: BLE001
        rec["error"] = str(exc)[:200]
        rec["note"] = ("the Actions API did not answer; an unread gate is not a "
                       "green one, and it is not a red one either")
        return rec
    if not run:
        rec["status"] = "NO_RUN"
        return rec
    conclusion = str(run.get("conclusion") or "").lower()
    rec.update({"conclusion": conclusion or None,
                "run_status": run.get("status"),
                "head_sha": run.get("head_sha"),
                "run_url": run.get("html_url"),
                "run_started_at": run.get("run_started_at"),
                "run_id": run.get("id")})
    if conclusion == "success":
        rec["status"] = "GREEN"
    elif conclusion in ("failure", "timed_out", "startup_failure"):
        rec["status"] = "RED"
    elif conclusion == "cancelled":
        # A human (or a superseding push) stopped it.  Not a verdict on the code.
        rec["status"] = "CANCELLED"
    else:
        rec["status"] = "PENDING"
    return rec


def sweep(root: Path | str = ".", *, api=None, now: datetime | None = None,
          ref: str = "main", dispatch: bool = True,
          out_dir: Path | str | None = None,
          max_catchups: int = MAX_CATCHUPS_PER_SWEEP) -> dict:
    """One pass: read the schedules, ask the API, report, and re-fire the drops.

    ``api`` is anything with ``last_scheduled_run(file)`` and
    ``dispatch(file, ref)``; :class:`ActionsApi` is the live one and the tests
    pass a fake, because a monitor that can only be tested against the real
    service is a monitor that is never tested.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    root = Path(root)
    out = Path(out_dir) if out_dir else root / DEFAULT_RESULTS_DIR
    state_path = out / "state.json"
    state: dict = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except Exception:                                      # noqa: BLE001
            state = {}
    state.setdefault("caught_up", {})

    workflows = read_schedules(root)
    last_runs: dict[str, datetime | None] = {}
    unknown: set[str] = set()
    changed_at: dict[str, datetime | None] = {}
    for wf in workflows:
        try:
            last_runs[wf.file] = api.last_scheduled_run(wf.file) if api else None
        except Exception as exc:                               # noqa: BLE001
            unknown.add(wf.file)
            last_runs[wf.file] = None
            state.setdefault("errors", {})[wf.file] = str(exc)[:200]
        # Asked over the API rather than from git: `actions/checkout` clones at
        # depth 1, so `git log -- <file>` on the runner answers for exactly the
        # files touched by the head commit and nothing else.
        try:
            changed_at[wf.file] = (api.workflow_changed_at(wf.file) if api
                                   and hasattr(api, "workflow_changed_at") else None)
        except Exception:                                      # noqa: BLE001
            changed_at[wf.file] = None
    if api is None:
        unknown |= {wf.file for wf in workflows}

    findings = assess(workflows, last_runs, now, unknown=frozenset(unknown),
                      changed_at=changed_at)
    fired: list[dict] = []
    if dispatch and api is not None:
        for rec in plan_catchup(findings, state, max_catchups=max_catchups):
            try:
                api.dispatch(rec["workflow"], ref)
            except Exception as exc:                           # noqa: BLE001
                rec["catchup_error"] = str(exc)[:200]
                continue
            rec["catchup_dispatched_utc"] = _iso(now)
            state["caught_up"][_catchup_key(rec)] = _iso(now)
            fired.append(rec)

    # Keep the ledger from growing without bound: a missed firing older than a
    # season is history, not state.
    cutoff = now - timedelta(days=120)
    state["caught_up"] = {
        k: v for k, v in state["caught_up"].items()
        if not v or _parse_iso(v) is None or _parse_iso(v) >= cutoff}

    gate = gate_status(api, branch=ref)
    report = {"checked_at_utc": _iso(now),
              "gate": gate,
              "n_workflows": len(workflows),
              "n_overdue": sum(1 for f in findings if f.get("overdue")),
              "n_unknown": sum(1 for f in findings if f["status"] == "UNKNOWN"),
              "n_dispatched": len(fired),
              "ref": ref,
              "workflows": findings}
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    state_path.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    return report


def _parse_iso(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:                                          # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# 5. the live API client (runner-only; the sandbox has no GitHub egress)
# ---------------------------------------------------------------------------
class ActionsApi:
    """The GitHub Actions REST endpoints this needs, and nothing else.

    Runs inside a workflow with ``GITHUB_TOKEN``.  Dispatching with that token is
    deliberate and documented: ``workflow_dispatch`` and ``repository_dispatch``
    are the two events GitHub exempts from the no-recursive-triggering rule, so a
    catch-up here really does start the run -- unlike a push made with the same
    token, which would not.
    """

    def __init__(self, repo: str, token: str, api_root: str = "https://api.github.com",
                 timeout: float = 30.0):
        self.repo = repo
        self.token = token
        self.api_root = api_root.rstrip("/")
        self.timeout = float(timeout)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"}

    def last_scheduled_run(self, workflow_file: str) -> datetime | None:
        import requests

        url = (f"{self.api_root}/repos/{self.repo}/actions/workflows/"
               f"{workflow_file}/runs")
        r = requests.get(url, headers=self._headers(), timeout=self.timeout,
                         params={"event": "schedule", "per_page": 1})
        if r.status_code == 404:
            # A workflow file that exists on this branch but has never been seen
            # by Actions (a new file on a branch, say).  Not an error, and not a
            # missed firing either.
            return None
        r.raise_for_status()
        runs = (r.json() or {}).get("workflow_runs") or []
        if not runs:
            return None
        stamp = runs[0].get("run_started_at") or runs[0].get("created_at")
        return _parse_iso(str(stamp)) if stamp else None

    def latest_run(self, workflow_file: str, branch: str | None = None,
                   event: str | None = None) -> dict | None:
        """The newest run of a workflow, optionally on one branch."""
        import requests

        url = (f"{self.api_root}/repos/{self.repo}/actions/workflows/"
               f"{workflow_file}/runs")
        params: dict = {"per_page": 1}
        if branch:
            params["branch"] = branch
        if event:
            params["event"] = event
        r = requests.get(url, headers=self._headers(), timeout=self.timeout,
                         params=params)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        runs = (r.json() or {}).get("workflow_runs") or []
        return runs[0] if runs else None

    def workflow_changed_at(self, workflow_file: str) -> datetime | None:
        """When the file last changed on the default branch.

        Used only to refuse to blame a schedule for a slot that predates it.
        """
        import requests

        url = f"{self.api_root}/repos/{self.repo}/commits"
        r = requests.get(url, headers=self._headers(), timeout=self.timeout,
                         params={"path": f".github/workflows/{workflow_file}",
                                 "per_page": 1})
        if r.status_code >= 400:
            return None
        commits = r.json() or []
        if not commits:
            return None
        stamp = (((commits[0] or {}).get("commit") or {}).get("committer")
                 or {}).get("date")
        if not stamp:
            return None
        try:
            return datetime.fromisoformat(str(stamp).replace("Z", "+00:00")
                                          ).astimezone(timezone.utc)
        except ValueError:
            return None

    def dispatch(self, workflow_file: str, ref: str = "main") -> None:
        import requests

        url = (f"{self.api_root}/repos/{self.repo}/actions/workflows/"
               f"{workflow_file}/dispatches")
        r = requests.post(url, headers=self._headers(), timeout=self.timeout,
                          json={"ref": ref})
        if r.status_code >= 400:
            raise RuntimeError(f"dispatch {workflow_file} -> {r.status_code} "
                               f"{(r.text or '')[:200]}")
