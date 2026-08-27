"""The first silence, made testable: a run that FAILED.

WHAT THIS REPLACES, AND WHY.  The hourly `watchdog` found failed runs and
re-ran them from twenty lines of `gh api` piped through `jq`, inline in the
workflow file.  That logic was never executed by a test -- it could only be
exercised by pushing it to `main` and waiting an hour -- and on 2026-08-27 it
did something no reviewer had considered.

THE INCIDENT.  Run 33022081059 was a `tocsin-altfeeds` ATLAS pass at commit
`1de2f34`, killed by its job timeout after walking 133 minutes and writing
nothing.  The bug that caused it -- an unbounded ATLAS walk with no wall-clock
budget -- was found, fixed, merged, and by 03:17 UTC `main` was six commits
past it.  The failure sweep saw a run whose conclusion was `failure` and whose
`run_attempt` was 1, and re-ran it.  A re-run checks out **the run's own
head_sha**, not the current head: it started the very code the fix had removed,
on a four-hour lane, ahead of two queued runs carrying the fix, and would have
committed that superseded code's results over theirs.

So the rule that was missing is not a special case.  It is the general one:

    A RE-RUN RE-RUNS OLD CODE.  If the commit a failed run was built from is no
    longer the head of its own branch, then the reason it failed may already be
    fixed, and re-running it can only reproduce the failure or -- worse -- half
    succeed and commit stale results over fresh ones.

There WAS a guard for this: `results/watchdog/skiplist.json`, a hand-kept list
of run ids.  It is the wrong shape, and the incident is the proof.  A list of
ids can only contain a run someone thought of in advance; this one was created,
failed, and resurrected inside six hours, and nobody had listed it.  The
skiplist is still honoured -- it holds runs superseded for reasons a sha cannot
express -- but it is no longer the thing standing between a fix and its own
undoing.

WHAT IS DELIBERATELY NOT RETRIED, each for its own reason:

* `ci` -- auto-retrying a failing test suite argues with the signal instead of
  reading it.  `seti.cronwatch.gate_status` READS it instead.
* `watchdog` -- a sweep that retries itself can spin.
* a run that is not `completed` -- an in-flight attempt 2 is not a failure
  waiting for attempt 3, and asking for one races the attempt already running.
* a run past its attempt ceiling -- three for a failure or a timeout, two for a
  cancel, because a cancel is often a human saying stop and being overruled
  twice is enough.
* a branch whose head cannot be read -- the same refusal `cronwatch` makes for
  an API that will not answer.  Unknown is not permission.  This one is counted
  and printed rather than passed over, because a sweep that quietly retries
  nothing looks exactly like a sweep with nothing to retry.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Attempt ceilings, by how the run ended.  `run_attempt` is 1 on the first try,
# so `run_attempt < ceiling` allows (ceiling - 1) retries.
RETRY_CEILING = {"failure": 3, "timed_out": 3, "cancelled": 2}
RETRYABLE = tuple(RETRY_CEILING)

# Names never retried, whatever they did.  See the module docstring.
NEVER_RETRY = ("watchdog", "ci")

WINDOW_HOURS = 24.0

# A sweep that fires twenty re-runs into a GitHub-wide incident is not helping,
# and mirrors `cronwatch.MAX_CATCHUPS_PER_SWEEP` for the same reason.
MAX_RETRIES_PER_SWEEP = 5


def _parse_iso(s) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(
            timezone.utc)
    except ValueError:
        return None


def _iso(dt: datetime | None) -> str | None:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None


def _keep(run: dict, keys=("id", "name", "head_branch", "head_sha", "conclusion",
                           "status", "run_attempt", "html_url", "created_at")) -> dict:
    return {k: run.get(k) for k in keys}


def plan(runs: list[dict], *, branch_heads: dict[str, str | None],
         now: datetime, skiplist=(), window_hours: float = WINDOW_HOURS,
         never_retry=NEVER_RETRY,
         max_retries: int = MAX_RETRIES_PER_SWEEP) -> dict:
    """Decide, for each run, retry or not -- and say why not.

    Pure: no network, no clock, no filesystem.  `branch_heads` maps a branch
    name to its current head sha, or to None where that could not be read;
    resolving it is the caller's job precisely so this can be tested.

    Every run that is considered and not retried appears in `skipped` with a
    `reason`.  Silence about a decision is how the incident above happened.
    """
    cutoff = now - timedelta(hours=float(window_hours))
    skipset = {int(x) for x in (skiplist or [])}
    retry: list[dict] = []
    skipped: list[dict] = []

    def drop(run, reason, note=None):
        rec = _keep(run)
        rec["reason"] = reason
        if note:
            rec["note"] = note
        skipped.append(rec)

    considered = 0
    for run in runs:
        if run.get("name") in never_retry:
            continue                      # not "skipped": never in scope
        created = _parse_iso(run.get("created_at"))
        if created is None or created < cutoff:
            continue
        if run.get("conclusion") not in RETRYABLE:
            continue
        considered += 1

        if str(run.get("status")) != "completed":
            drop(run, "still_running",
                 "an attempt is in flight; asking for another races it")
            continue
        if int(run.get("id", 0)) in skipset:
            drop(run, "skiplisted", "listed in results/watchdog/skiplist.json")
            continue

        branch = run.get("head_branch")
        head = branch_heads.get(branch, "__missing__")
        if head == "__missing__" or head is None:
            drop(run, "branch_head_unknown",
                 f"could not read the current head of {branch!r}; a re-run "
                 f"checks out this run's own commit, so an unverifiable branch "
                 f"is not a retryable one")
            continue
        if head != run.get("head_sha"):
            drop(run, "superseded",
                 f"built from {str(run.get('head_sha'))[:7]}, but {branch} is "
                 f"now at {str(head)[:7]} -- a re-run would re-run the old "
                 f"code, and could commit its results over the new")
            continue

        ceiling = RETRY_CEILING[run["conclusion"]]
        attempt = int(run.get("run_attempt") or 1)
        if attempt >= ceiling:
            drop(run, "retries_exhausted",
                 f"attempt {attempt} of at most {ceiling} for a "
                 f"{run['conclusion']} run")
            continue
        retry.append(_keep(run))

    if len(retry) > max_retries:
        for run in retry[max_retries:]:
            drop(run, "over_sweep_budget",
                 f"more than {max_retries} retryable runs in one sweep is a "
                 f"repository-wide or GitHub-wide problem, not {run['name']}'s")
        retry = retry[:max_retries]

    return {"checked_at_utc": _iso(now), "window_hours": float(window_hours),
            "n_considered": considered, "n_to_retry": len(retry),
            "n_skipped": len(skipped), "retry": retry, "skipped": skipped}


def read_skiplist(root: Path | str = ".") -> list[int]:
    path = Path(root) / "results" / "watchdog" / "skiplist.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return []
    return [int(x) for x in data] if isinstance(data, list) else []


def resolve_branch_heads(api, branches) -> dict[str, str | None]:
    """Current head sha per branch, None where it could not be read."""
    heads: dict[str, str | None] = {}
    for branch in branches:
        if branch in heads:
            continue
        try:
            heads[branch] = api.branch_head(branch)
        except Exception:                       # noqa: BLE001 -- unknown, not fatal
            heads[branch] = None
    return heads


def sweep(root: Path | str = ".", *, api=None, now: datetime | None = None,
          retry: bool = True, window_hours: float = WINDOW_HOURS,
          out_dir: Path | str | None = None) -> dict:
    """Find failed runs, decide, act, and write the ledger."""
    now = now or datetime.now(timezone.utc)
    if api is None:
        report = {"checked_at_utc": _iso(now), "window_hours": float(window_hours),
                  "n_considered": 0, "n_to_retry": 0, "n_skipped": 0,
                  "retry": [], "skipped": [], "retried": 0,
                  "no_api": True}
    else:
        runs = api.recent_runs(per_page=100)
        # Resolve only the branches that could matter, one call each.
        wanted = {r.get("head_branch") for r in runs
                  if r.get("name") not in NEVER_RETRY
                  and r.get("conclusion") in RETRYABLE
                  and (_parse_iso(r.get("created_at")) or now)
                  >= now - timedelta(hours=float(window_hours))}
        heads = resolve_branch_heads(api, sorted(b for b in wanted if b))
        report = plan(runs, branch_heads=heads, now=now,
                      skiplist=read_skiplist(root), window_hours=window_hours)
        report["branch_heads"] = heads
        n = 0
        for rec in report["retry"]:
            if not retry:
                rec["retry_dispatched"] = False
                rec["retry_note"] = "--no-retry"
                continue
            try:
                api.rerun_failed_jobs(int(rec["id"]))
                rec["retry_dispatched"] = True
                n += 1
            except Exception as exc:            # noqa: BLE001
                rec["retry_dispatched"] = False
                rec["retry_error"] = str(exc)[:300]
        report["retried"] = n

    # The ledger is written on EVERY sweep, unlike the shell version, which
    # wrote it only when something failed.  A file that appears only on bad days
    # cannot distinguish "nothing failed" from "the sweep did not run" -- which
    # is the whole subject of docs/cronwatch.md.
    out = Path(out_dir) if out_dir else Path(root) / "results" / "watchdog"
    out.mkdir(parents=True, exist_ok=True)
    # Kept for continuity with the shell ledger that came before.
    report["failures"] = report["retry"] + report["skipped"]
    (out / "status.json").write_text(json.dumps(report, indent=1, sort_keys=True)
                                     + "\n")
    return report
