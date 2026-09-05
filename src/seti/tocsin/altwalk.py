"""Walk state, the job deadline, and a concurrent ATLAS scheduler for TOCSIN's alternative feeds.

WHY THIS MODULE EXISTS.  Three things went wrong with the first four weeks of
``tocsin-altfeeds`` runs, and each one is a property of *how the walk was
organised*, not of any feed:

1. **The job ran out of clock and was killed.**  The 2026-09-02 run spent its
   whole ATLAS budget, then started the ZTF step with no knowledge of how much
   job time was left, and the runner cancelled it 86 minutes later.  Every
   survey had a budget; the job had none.  :func:`job_deadline_monotonic` reads
   one deadline for the whole job (set by the workflow from its own
   ``timeout-minutes``) and every survey's budget is clipped to it.

2. **Every run re-walked the same six stars.**  The walk order is deterministic
   (brightest measurable first) and nothing recorded which targets had already
   been walked, so a weekly run bought the same six full-history light curves
   again and the ledger never grew.  :class:`WalkState` records, per target, the
   window screened so far and the baseline measured from its full history; the
   next run asks each walked star only for the epochs it has not yet seen, which
   for a live survey is a few nights and costs seconds on the server rather than
   twenty minutes.

3. **The ledger's denominator was undercounted, by a lot.**  The Rubin ledger
   de-duplicates trials by *night*, because there one run screens every target
   over one night.  Here one run screens a few targets over eleven years, and a
   later run adds more targets over the same eleven years; the night labels
   collide and the second target's trials were dropped.  The committed ATLAS
   ledger showed 1337 nights and exactly 1337 target-visits over six stars.  The
   fix is in :func:`seti.tocsin.altfeeds.fold` (trials keyed to the star-night,
   with novelty guaranteed by the disjoint per-target windows this module
   maintains); this module is what makes that guarantee hold.

Everything here is offline-testable; the only network call is inside
:class:`AtlasWalk`, through a client the tests replace.
"""

from __future__ import annotations

import datetime
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MJD_UNIX_EPOCH = 40587.0

#: Unix time (seconds) by which the WHOLE job must be done with network work.
#: Set once by the workflow from its own ``timeout-minutes``; every survey's
#: budget is clipped to it.  Absent (a local run), budgets are what the caller
#: says they are.
JOB_DEADLINE_ENV = "ALTFEEDS_JOB_DEADLINE_UNIX"

#: Seconds held back from the job deadline for the work that follows the fetch:
#: reduce, screen, fold the ledger, write the summary, and commit.  Measured at
#: well under five minutes on the runner; fifteen is the margin.
DEFAULT_RESERVE_S = 900.0

#: How far behind the wall clock a LIVE survey's newest epoch is assumed to sit,
#: so that a refresh window never claims to have screened nights the server has
#: not finished processing.  ATLAS images are reduced within a day or two;
#: three days is generous and the cost is only that the newest three nights are
#: screened on the NEXT run rather than this one.
DEFAULT_INGEST_LAG_DAYS = 3.0

#: A walked target is refreshed once this many days of unscreened sky have
#: accrued.  Below this the request would carry a handful of exposures and the
#: per-request overhead would dominate.
DEFAULT_REFRESH_AFTER_DAYS = 7.0

#: A target whose full-history walk found it unusable (too few epochs, a scatter
#: above the limit, no detected quiescent flux) is walked again in full after
#: this long; something may have changed at the survey, and a permanently
#: written-off star is a permanently missing trial.
DEFAULT_REWALK_UNUSABLE_AFTER_DAYS = 180.0


def now_mjd() -> float:
    return MJD_UNIX_EPOCH + time.time() / 86400.0


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def mjd_to_utc(mjd: float | None) -> str | None:
    if mjd is None or not math.isfinite(mjd):
        return None
    dt = datetime.datetime(1858, 11, 17, tzinfo=datetime.timezone.utc) + \
        datetime.timedelta(days=float(mjd))
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# The job deadline
# ---------------------------------------------------------------------------
def job_deadline_monotonic(reserve_s: float = DEFAULT_RESERVE_S,
                           env: dict | None = None,
                           wall_now: float | None = None,
                           mono_now: float | None = None) -> float | None:
    """The job's network deadline on the ``time.monotonic()`` clock, or None.

    Translated from the unix deadline in :data:`JOB_DEADLINE_ENV` by the offset
    between the two clocks *now*; the reserve is subtracted so the fetch stops
    with time in hand for the steps after it.  ``env``/``wall_now``/``mono_now``
    are injectable for the tests.
    """
    env = os.environ if env is None else env
    raw = str(env.get(JOB_DEADLINE_ENV, "")).strip()
    if not raw:
        return None
    try:
        deadline_unix = float(raw)
    except ValueError:
        return None
    wall = time.time() if wall_now is None else float(wall_now)
    mono = time.monotonic() if mono_now is None else float(mono_now)
    return mono + (deadline_unix - wall) - float(reserve_s)


def effective_budget_s(own_budget_s: float, reserve_s: float = DEFAULT_RESERVE_S,
                       env: dict | None = None, wall_now: float | None = None,
                       mono_now: float | None = None) -> tuple[float, str | None]:
    """Clip a survey's own budget to what the job has left.

    Returns ``(budget_s, note)``; the note says when and by how much the clip
    bit, so the committed summary records that the slice was short because the
    JOB was short, which is a different fact from the feed being slow.
    """
    own = max(0.0, float(own_budget_s))
    dl = job_deadline_monotonic(reserve_s, env=env, wall_now=wall_now, mono_now=mono_now)
    if dl is None:
        return own, None
    mono = time.monotonic() if mono_now is None else float(mono_now)
    remaining = max(0.0, dl - mono)
    if remaining < own:
        return remaining, (f"fetch budget clipped from {own:.0f}s to {remaining:.0f}s "
                           f"by the job deadline ({JOB_DEADLINE_ENV}), keeping "
                           f"{reserve_s:.0f}s in reserve for screen+ledger+commit")
    return own, None


# ---------------------------------------------------------------------------
# Walk state: what has been screened, per target, and the baseline it measured
# ---------------------------------------------------------------------------
@dataclass
class TargetWalk:
    """One target's screening record: the window covered and the priors measured."""

    target_id: str
    mjd_lo: float
    mjd_hi: float                        # screened up to here
    walked_utc: str
    usable: bool
    n_epochs: int = 0
    n_walks: int = 1
    # Per native band, from the FULL-HISTORY walk: {level, scatter, n_used}.
    # A refresh window carries too few epochs to measure its own scatter, so it
    # is judged against these.  See `altfeeds.reduce_lightcurve(prior=...)`.
    baseline: dict[str, dict] = field(default_factory=dict)
    # Per native band: {flux_njy, err_njy, source}.  ATLAS's F* comes from a
    # separate reduced-image pass, which a refresh does not repeat.
    quiescent: dict[str, dict] = field(default_factory=dict)
    chi_med: float | None = None
    elo_med: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class WalkPlan:
    """What this run will ask the survey for, and why."""

    refresh: list[dict] = field(default_factory=list)   # {target_id, mjd_lo, mjd_hi, mode}
    fresh: list[dict] = field(default_factory=list)
    skipped_current: int = 0
    skipped_unusable: int = 0
    hi_request: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def requests(self) -> list[dict]:
        return list(self.refresh) + list(self.fresh)


@dataclass
class WalkState:
    """The per-survey checkpoint: ``results/tocsin_altfeeds/<survey>/walked.json``."""

    version: int = 1
    survey: str = ""
    updated_utc: str = ""
    # For a data-release archive (ZTF via IRSA) the newest epoch the archive
    # holds AT ALL.  A target is never marked screened past it, because the
    # epochs after it do not exist yet and will arrive with the next release.
    archive_frontier_mjd: float | None = None
    targets: dict[str, dict] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, survey: str = "") -> WalkState:
        import json
        p = Path(path)
        if not p.exists():
            return cls(survey=survey)
        raw = json.loads(p.read_text())
        st = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        if survey and not st.survey:
            st.survey = survey
        return st

    def save(self, path: str | Path) -> None:
        import json
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.updated_utc = utc_now()
        p.write_text(json.dumps(_sanitize(asdict(self)), indent=1, sort_keys=True,
                                allow_nan=False) + "\n")

    # -- queries -----------------------------------------------------------
    def get(self, target_id: str) -> TargetWalk | None:
        raw = self.targets.get(str(target_id))
        if raw is None:
            return None
        return TargetWalk(**{k: v for k, v in raw.items()
                             if k in TargetWalk.__dataclass_fields__})

    def n_walked(self) -> int:
        return len(self.targets)

    def n_usable(self) -> int:
        return sum(1 for t in self.targets.values() if t.get("usable"))

    def most_observed(self) -> TargetWalk | None:
        best = None
        for raw in self.targets.values():
            if best is None or int(raw.get("n_epochs") or 0) > int(best.get("n_epochs") or 0):
                best = raw
        return None if best is None else self.get(best["target_id"])

    # -- planning ----------------------------------------------------------
    def plan(self, order: list[str], *, survey_start_mjd: float, now: float,
             max_new: int | None, mjd_lo: float | None = None,
             mjd_hi: float | None = None,
             ingest_lag_days: float = DEFAULT_INGEST_LAG_DAYS,
             refresh_after_days: float = DEFAULT_REFRESH_AFTER_DAYS,
             rewalk_unusable_after_days: float = DEFAULT_REWALK_UNUSABLE_AFTER_DAYS,
             archive_frontier: float | None = None) -> WalkPlan:
        """Decide, for each target in walk order, what window to request.

        ``order`` is the survey's measurable targets, brightest first.  Refreshes
        of already-walked stars come FIRST in the returned plan: they are cheap
        (a few nights each) and they are what keeps the ledger current; new
        full-history walks fill whatever budget is left, up to ``max_new``.

        The recorded high-water mark of every request is ``hi_request``, which is
        the wall clock minus the ingest lag, further capped at the archive's
        own frontier when one is known.  Windows for one target are therefore
        disjoint across runs by construction, which is what lets the ledger add
        trials without a night-level de-duplication.
        """
        plan = WalkPlan()
        hi = float(mjd_hi) if mjd_hi is not None else float(now) - float(ingest_lag_days)
        fr = archive_frontier if archive_frontier is not None else self.archive_frontier_mjd
        if fr is not None and math.isfinite(fr):
            hi = min(hi, float(fr))
        plan.hi_request = hi
        lo_default = float(mjd_lo) if mjd_lo is not None else float(survey_start_mjd)
        now_utc_days = float(now)

        n_new = 0
        for tid in order:
            tid = str(tid)
            rec = self.get(tid)
            if rec is None:
                if max_new is not None and n_new >= int(max_new):
                    continue
                plan.fresh.append({"target_id": tid, "mjd_lo": lo_default,
                                   "mjd_hi": hi, "mode": "full"})
                n_new += 1
                continue
            if not rec.usable:
                walked_mjd = _utc_to_mjd(rec.walked_utc)
                age = (now_utc_days - walked_mjd) if walked_mjd is not None else math.inf
                if age >= float(rewalk_unusable_after_days):
                    if max_new is not None and n_new >= int(max_new):
                        continue
                    plan.fresh.append({"target_id": tid, "mjd_lo": lo_default,
                                       "mjd_hi": hi, "mode": "rewalk"})
                    n_new += 1
                else:
                    plan.skipped_unusable += 1
                continue
            gap = hi - float(rec.mjd_hi)
            if gap >= float(refresh_after_days):
                plan.refresh.append({"target_id": tid, "mjd_lo": float(rec.mjd_hi),
                                     "mjd_hi": hi, "mode": "refresh"})
            else:
                plan.skipped_current += 1
        if plan.skipped_current:
            plan.notes.append(f"{plan.skipped_current} walked targets are current to "
                              f"within {refresh_after_days:g} d of the request "
                              f"frontier (MJD {hi:.2f}) and were not re-requested")
        if plan.skipped_unusable:
            plan.notes.append(f"{plan.skipped_unusable} targets were unusable on their "
                              f"last full walk and are not due a re-walk yet")
        return plan

    # -- recording ---------------------------------------------------------
    def record(self, target_id: str, *, mjd_lo: float, mjd_hi: float, mode: str,
               usable: bool, n_epochs: int, baseline: dict | None = None,
               quiescent: dict | None = None, chi_med: float | None = None,
               elo_med: float | None = None, notes: list[str] | None = None) -> TargetWalk:
        """Fold one walk into the state.

        A ``refresh`` keeps the priors measured by the full-history walk and only
        advances the high-water mark: a two-week window cannot re-measure a
        star's scatter and must not be allowed to overwrite the number that was
        measured from thousands of epochs.  A ``full`` or ``rewalk`` replaces
        everything.
        """
        prev = self.get(target_id)
        if mode == "refresh" and prev is not None:
            rec = prev
            rec.mjd_hi = max(float(rec.mjd_hi), float(mjd_hi))
            rec.n_epochs = int(rec.n_epochs) + int(n_epochs)
            rec.n_walks = int(rec.n_walks) + 1
            rec.walked_utc = utc_now()
            # A refresh that finds the star unusable does not demote it: the
            # verdict came from a short window judged against full-history
            # priors, so "unusable" here can only mean "nothing to say tonight".
            for n in (notes or []):
                if n not in rec.notes:
                    rec.notes.append(n)
        else:
            rec = TargetWalk(target_id=str(target_id), mjd_lo=float(mjd_lo),
                             mjd_hi=float(mjd_hi), walked_utc=utc_now(),
                             usable=bool(usable), n_epochs=int(n_epochs),
                             n_walks=1 if prev is None else int(prev.n_walks) + 1,
                             baseline=dict(baseline or {}),
                             quiescent=dict(quiescent or {}),
                             chi_med=_finite_or_none(chi_med),
                             elo_med=_finite_or_none(elo_med),
                             notes=list(notes or []))
        self.targets[str(target_id)] = asdict(rec)
        return rec

    def observe_archive_frontier(self, mjd: float | None) -> None:
        """Raise the archive frontier to the newest epoch seen; never lower it."""
        if mjd is None or not math.isfinite(mjd):
            return
        if self.archive_frontier_mjd is None or mjd > self.archive_frontier_mjd:
            self.archive_frontier_mjd = float(mjd)

    def prior_for(self, target_id: str):
        """The :class:`Prior` a refresh window is reduced against, or None."""
        rec = self.get(target_id)
        if rec is None or not rec.usable or not rec.baseline:
            return None
        return Prior(bands={b: dict(v) for b, v in rec.baseline.items()},
                     quiescent={b: dict(v) for b, v in rec.quiescent.items()},
                     chi_med=rec.chi_med, elo_med=rec.elo_med)


@dataclass(frozen=True)
class Prior:
    """Full-history statistics a short refresh window is reduced against."""

    bands: dict[str, dict]           # band -> {level, scatter, n_used}
    quiescent: dict[str, dict]       # band -> {flux_njy, err_njy, source}
    chi_med: float | None = None
    elo_med: float | None = None


def _utc_to_mjd(s: str | None) -> float | None:
    if not s:
        return None
    try:
        dt = datetime.datetime.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc)
    except ValueError:
        return None
    return MJD_UNIX_EPOCH + dt.timestamp() / 86400.0


def _finite_or_none(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


# ---------------------------------------------------------------------------
# The concurrent ATLAS scheduler
# ---------------------------------------------------------------------------
@dataclass
class AtlasTask:
    target_id: str
    seg_index: int
    kind: str                 # "diff" or "reduced"
    ra: float
    dec: float
    mjd_lo: float
    mjd_hi: float
    url: str | None = None
    text: str | None = None
    submitted_mono: float = float("nan")
    finished_mono: float = float("nan")


@dataclass
class AtlasTargetJob:
    target_id: str
    ra: float
    dec: float
    pmra: float
    pmdec: float
    mjd_lo: float
    mjd_hi: float
    with_baseline: bool = True
    segs: list[dict] = field(default_factory=list)
    capped: bool = False
    tasks: list[AtlasTask] = field(default_factory=list)
    started_mono: float = float("nan")
    failed: str | None = None

    @property
    def done(self) -> bool:
        return all(t.text is not None for t in self.tasks)


class AtlasWalk:
    """Walk many targets through the ATLAS job queue with several tasks in flight.

    WHY.  The serial walk submitted one task, polled it to completion (a
    full-history job runs ~20 minutes server-side), then submitted the next.
    ATLAS accepts many queued tasks per user, so keeping ``concurrency`` of them
    in flight overlaps their server time and removes the poll latency between
    them.  How much it helps is a property of ATLAS's own scheduler that this
    sandbox cannot measure; the walk therefore RECORDS the achieved rate (tasks
    finished per hour, and the server-reported runtimes where present) so the
    next tuning decision is made on a number rather than a guess.

    BUDGET DISCIPLINE.  One deadline bounds the whole walk.  A target is only
    STARTED when the running median time-to-complete of finished targets fits
    before the deadline, so the walk does not queue work it will abandon; and a
    target whose tasks are still in flight at the deadline is dropped with a
    note rather than folded as a half-walked light curve (a gap where the budget
    ran out reads as a dip, the exact artefact this channel searches for).

    Throttling.  A 429 from the queue pauses submission for ``throttle_pause_s``
    and retries the same task; nothing is lost and nothing is duplicated.
    """

    def __init__(self, client, concurrency: int | None = None,
                 poll_s: float | None = None, throttle_pause_s: float = 60.0,
                 clock=None, sleep=None):
        self.client = client
        env_c = os.environ.get("ALTFEEDS_ATLAS_CONCURRENCY", "").strip()
        self.concurrency = int(concurrency if concurrency is not None
                               else (env_c or 6))
        self.concurrency = max(1, self.concurrency)
        self.poll_s = float(poll_s if poll_s is not None else getattr(client, "poll_s", 20.0))
        self.throttle_pause_s = float(throttle_pause_s)
        self.now = clock or time.monotonic
        self.sleep = sleep or time.sleep
        self.notes: list[str] = []
        self.n_tasks_finished = 0
        self.n_tasks_submitted = 0
        self.server_runtimes: list[float] = []
        self.target_wall_s: list[float] = []

    # -- planning one target -----------------------------------------------
    def _expand(self, job: AtlasTargetJob, th) -> None:
        from .altfeeds import ATLAS, AtlasForcedPhotometry, pm_segments
        segs = pm_segments(job.ra, job.dec, job.pmra, job.pmdec, job.mjd_lo, job.mjd_hi,
                           ATLAS, th)
        cap = AtlasForcedPhotometry.MAX_PM_SEGMENTS
        job.capped = len(segs) > cap
        if job.capped:
            segs = pm_segments(job.ra, job.dec, job.pmra, job.pmdec, job.mjd_lo,
                               job.mjd_hi, ATLAS, th, max_segments=cap)
        job.segs = segs
        job.tasks = []
        for i, seg in enumerate(segs):
            job.tasks.append(AtlasTask(job.target_id, i, "diff", seg["ra"], seg["dec"],
                                       seg["mjd_lo"], seg["mjd_hi"]))
        if job.with_baseline:
            for i, seg in enumerate(segs):
                job.tasks.append(AtlasTask(job.target_id, i, "reduced", seg["ra"],
                                           seg["dec"], seg["mjd_lo"], seg["mjd_hi"]))

    # -- the loop -------------------------------------------------------------
    def run(self, jobs: list[AtlasTargetJob], deadline: float, th=None,
            on_target=None) -> tuple[dict, dict, list[str]]:
        """Walk ``jobs`` in order until done or ``deadline`` (monotonic seconds).

        Returns ``(lightcurves, quiescent, notes)`` keyed by target id, exactly
        as the serial path did.  ``on_target(tid, lc, q)`` is called as each
        target completes so a caller can checkpoint progressively.
        """
        from .altfeeds import AltFeedError, LightCurveThresholds
        th = th or LightCurveThresholds()
        lightcurves: dict = {}
        quiescent: dict = {}
        pending_jobs = deque(jobs)
        active: dict[str, AtlasTargetJob] = {}
        task_queue: deque[AtlasTask] = deque()
        inflight: list[AtlasTask] = []
        throttled_until = -math.inf
        started_mono = self.now()
        n_abandoned = 0

        def _time_left() -> float:
            return float(deadline) - self.now()

        def _est_target_s() -> float | None:
            if not self.target_wall_s:
                return None
            return float(np.median(self.target_wall_s))

        while (pending_jobs or task_queue or inflight) and _time_left() > 0:
            # 1. Start new targets while the queue has room for their tasks and
            #    the running estimate says they can finish.
            while (pending_jobs and not task_queue
                   and len(inflight) < self.concurrency):
                est = _est_target_s()
                if est is not None and est > _time_left():
                    self.notes.append(
                        f"stopped starting targets with {_time_left():.0f}s left: the "
                        f"median target has taken {est:.0f}s and would not finish")
                    pending_jobs.clear()
                    break
                job = pending_jobs.popleft()
                try:
                    self._expand(job, th)
                except Exception as exc:                          # noqa: BLE001
                    self.notes.append(f"{job.target_id}: could not plan segments: "
                                      f"{str(exc)[:160]}")
                    continue
                job.started_mono = self.now()
                active[job.target_id] = job
                task_queue.extend(job.tasks)

            # 2. Submit queued tasks up to the concurrency limit.
            while task_queue and len(inflight) < self.concurrency and self.now() >= throttled_until:
                t = task_queue[0]
                job = active.get(t.target_id)
                if job is None or job.failed:
                    task_queue.popleft()
                    continue
                try:
                    t.url = self.client.submit(t.ra, t.dec, t.mjd_lo, t.mjd_hi,
                                               use_reduced=(t.kind == "reduced"))
                except AltFeedError as exc:
                    msg = str(exc)
                    if "429" in msg or "throttl" in msg.lower():
                        throttled_until = self.now() + self.throttle_pause_s
                        self.notes.append(f"ATLAS queue throttled; pausing submission "
                                          f"{self.throttle_pause_s:.0f}s")
                        break
                    self._fail(job, f"submit failed: {msg[:160]}", task_queue, inflight)
                    continue
                except Exception as exc:                          # noqa: BLE001
                    self._fail(job, f"submit error: {str(exc)[:160]}", task_queue, inflight)
                    continue
                task_queue.popleft()
                t.submitted_mono = self.now()
                self.n_tasks_submitted += 1
                inflight.append(t)

            if not inflight and not task_queue and not pending_jobs:
                break
            if not inflight:
                # Only throttled submissions remain; wait the pause out.
                wait = max(0.0, min(throttled_until - self.now(), _time_left()))
                if wait > 0:
                    self.sleep(wait)
                continue

            # 3. Poll everything in flight once.
            for t in list(inflight):
                job = active.get(t.target_id)
                if job is None or job.failed:
                    # `_fail` may already have pulled this task out when a
                    # sibling task of the same target failed earlier in this pass.
                    if t in inflight:
                        inflight.remove(t)
                    continue
                try:
                    text, runtime = self.client.poll_once(t.url)
                except AltFeedError as exc:
                    self._fail(job, f"task {t.kind}[{t.seg_index}] failed: "
                                    f"{str(exc)[:160]}", task_queue, inflight)
                    continue
                except Exception as exc:                          # noqa: BLE001
                    # A transient read error is not a failed task; poll again.
                    self.notes.append(f"{t.target_id}: poll error {str(exc)[:120]}")
                    continue
                if text is None:
                    continue
                t.text = text
                t.finished_mono = self.now()
                inflight.remove(t)
                self.n_tasks_finished += 1
                if runtime is not None:
                    self.server_runtimes.append(float(runtime))
                if job.done:
                    self._finish(job, th, lightcurves, quiescent, on_target)
                    active.pop(job.target_id, None)

            if inflight and _time_left() > 0:
                self.sleep(min(self.poll_s, max(0.0, _time_left())))

        # 4. Anything still in flight at the deadline is abandoned, loudly.
        for job in list(active.values()):
            if not job.done and not job.failed:
                n_abandoned += 1
                self.notes.append(
                    f"{job.target_id}: walk deadline reached with "
                    f"{sum(1 for t in job.tasks if t.text is None)} of {len(job.tasks)} "
                    f"tasks unfinished; discarded rather than folded into a gappy "
                    f"light curve")
        if pending_jobs:
            self.notes.append(f"{len(pending_jobs)} planned targets were never started "
                              f"(walk deadline)")
        wall_h = max(1e-9, (self.now() - started_mono) / 3600.0)
        rate = self.n_tasks_finished / wall_h
        summary = (f"atlas scheduler: concurrency {self.concurrency}, "
                   f"{self.n_tasks_submitted} tasks submitted, "
                   f"{self.n_tasks_finished} finished, {rate:.1f} tasks/h, "
                   f"{len(lightcurves)} targets completed, {n_abandoned} abandoned")
        if self.server_runtimes:
            summary += (f"; server runtime median {np.median(self.server_runtimes):.0f}s, "
                        f"achieved parallelism "
                        f"{sum(self.server_runtimes) / max(1e-9, wall_h * 3600.0):.2f}")
        self.notes.append(summary)
        return lightcurves, quiescent, list(self.notes)

    def _fail(self, job: AtlasTargetJob, why: str, task_queue: deque,
              inflight: list) -> None:
        job.failed = why
        self.notes.append(f"{job.target_id}: {why}")
        for t in list(task_queue):
            if t.target_id == job.target_id:
                task_queue.remove(t)
        for t in list(inflight):
            if t.target_id == job.target_id:
                inflight.remove(t)

    def _finish(self, job: AtlasTargetJob, th, lightcurves: dict, quiescent: dict,
                on_target) -> None:
        from .altfeeds import AtlasForcedPhotometry
        diff = [t.text for t in sorted((t for t in job.tasks if t.kind == "diff"),
                                       key=lambda t: t.seg_index)]
        red = [t.text for t in sorted((t for t in job.tasks if t.kind == "reduced"),
                                      key=lambda t: t.seg_index)]
        try:
            lc, q = AtlasForcedPhotometry.assemble(job.target_id, job.ra, job.dec,
                                                   job.segs, diff, red, th,
                                                   capped=job.capped,
                                                   baseline_complete=True)
        except Exception as exc:                                  # noqa: BLE001
            self.notes.append(f"{job.target_id}: assembly failed: {str(exc)[:160]}")
            return
        lightcurves[job.target_id] = lc
        quiescent[job.target_id] = q
        if math.isfinite(job.started_mono):
            self.target_wall_s.append(self.now() - job.started_mono)
        if on_target is not None:
            try:
                on_target(job.target_id, lc, q)
            except Exception as exc:                              # noqa: BLE001
                self.notes.append(f"{job.target_id}: checkpoint callback failed: "
                                  f"{str(exc)[:120]}")
