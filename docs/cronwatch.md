# The third silence: a screen that never started

*Opened 2026-08-26. Code: `src/seti/cronwatch.py` + `scripts/cron_watch.py`
(never started), `src/seti/failsweep.py` + `scripts/fail_sweep.py` (failed),
both wired into `.github/workflows/watchdog.yml` and the first also into
`.github/workflows/cronwatch.yml`. Live state: `results/cronwatch/status.json`,
`results/cronwatch/state.json`, `results/watchdog/status.json`.*

## The gap this closes

Every channel here can fall silent in five different ways, and when this was
opened only two of them were watched. All four are watched now — and the first
row, the one that *was* covered, turned out to be covered wrongly; see **The
retry that made things worse** below.

| what stopped | what it looks like | who catches it |
|---|---|---|
| the run **failed** | a red run in the Actions tab | `seti.failsweep`, hourly inside `watchdog`, and it re-runs the failed jobs *a retry can fix* |
| the result **never landed** | a GREEN run, the work done, the commit stranded by a conflicted rebase | `scripts/commit_results.sh`, which verifies against the remote and fails loudly |
| the **data** stopped | runs stay green, results stay fresh, the frontier stops moving | `alerts.health_alerts` — the frontier lag and stall checks, and `rubin-outage` for the cause |
| the run **never started** | *nothing at all* | nobody, until now |
| the **tests** went red | CI red on `main`; every screen still runs and still commits | nobody, until now — `watchdog` skips `ci` on purpose |

The third is the nastiest, because it leaves no artefact anywhere. There is no
failed run to retry — there is no run. The results do not change, so nothing is
obviously wrong; they simply age, and the staleness windows in
`alerts.STALE_DAYS` are deliberately set at a little over twice each channel's
cadence so that one missed firing is *not* an incident. For `tocsin_altfeeds`
that window is 16 days. So a Wednesday that never fires is invisible for a
fortnight, by design, in the check that was supposed to notice.

## It is not hypothetical

GitHub's hosted scheduler is documented as best effort, and drops firings under
load. Measured in this repository on 2026-08-26:

* `watchdog`, cron `17 * * * *` — nominally hourly, on the hour — fired at
  09:00, 09:55, 10:50, 11:42, 13:13, 14:02, 16:33 and 19:16 UTC. Not one of
  those is `:17`, and the gaps run from 49 to 151 minutes.
* `tocsin-altfeeds` reached its 18:40 slot at **21:12** — 2 h 32 m late.
* `alerts` reached its 17:05 slot at 19:14 — 2 h 09 m late.
* `loom`, weekly, has fired every Monday since 2026-08-03, between 13 and 46
  minutes late each time.

So: *late* is normal here and must never alert. *Absent* is the failure, and the
two have to be told apart by something that knows each channel's cadence.

## How it decides

1. **The schedules are read out of the workflow files**, not from a list kept in
   the module. A hand-kept registry is one rename away from watching a workflow
   that no longer exists while ignoring the one that replaced it — and it would
   look correct the whole time. (PyYAML reads a bare `on:` key as the boolean
   `True`, YAML 1.1 style, so the parser looks for both. Getting that wrong
   yields an empty watch list and a permanently clean bill of health.)
2. **The Actions API is asked when each workflow last ran on a `schedule`
   event** — not on any event, because a manual dispatch is not evidence that
   the cron fired.
3. **The cron says when it should last have fired.** Vixie semantics, in UTC,
   including the rule that day-of-month and day-of-week are OR-ed when both are
   restricted.
4. **The grace window is a quarter of that channel's own cadence**, floored at
   2 hours and capped at 12. Hourly channels get 2 h, daily 6 h, weekly and
   monthly 12 h. Every delay measured above sits inside its floor; a dropped
   weekly firing is called within half a day, six days before the next one.
5. Past grace with no run, the firing is **MISSED**: reported to the alert layer
   and, once, **re-dispatched**. `workflow_dispatch` is one of the two events
   GitHub exempts from the no-recursive-triggering rule, so the workflow token
   can genuinely start the run — a push made with the same token could not.

## What it refuses to do

Each of these is a way this check could have become noise, and noise is how a
monitor gets ignored:

* **A firing that predates the workflow file's last change is
  `SCHEDULE_TOO_NEW`, never missed.** Four channels were put on schedules on
  2026-08-25; without this rule the first sweep would have reported all four
  overdue — and re-fired them — for slots that existed only in arithmetic.
* **An API that does not answer is `UNKNOWN`, never overdue.** A monitor that
  cries wolf over its own transport failure gets muted, and then it is not a
  monitor.
* **One catch-up per missed firing, ever.** A catch-up that fails leaves a
  failed run, which is the failure sweep's job. Two monitors fighting over one
  job is worse than either alone.
* **At most three catch-ups per sweep.** Six channels dropping at once is a
  GitHub-wide problem, and firing six jobs into it is not the answer.

### The clock that could not see the outage it was built for

Grace was measured against the most recent **expected slot**. That lateness can
never exceed one cadence, because a new slot keeps arriving and resetting it —
so any channel firing more often than its own grace window could never be
reported missed at all. The fastest channel here is `watchdog`, which is the
*actor* for every other one.

Measured on 2026-08-27: `watchdog`'s 04:17, 05:17 **and** 06:17 firings were all
dropped. At 06:40, with no run in 3 h 23 m, the ledger said `WITHIN_GRACE` — and
would have said it indefinitely. The sharper the cadence, the blinder the check,
which is exactly backwards.

So there are two clocks now, and a channel is overdue if **either** trips:

* **the slot clock** — `now - expected_slot > grace`. Unchanged, and still the
  sensitive one wherever it can speak: a dropped weekly firing is called in 12 h
  rather than after 180 h of silence.
* **the silence clock** — `now - last_actual_run > cadence + grace`. Hourly:
  3 h. This is the only one that can see a sustained outage of a fast channel.

`missed_by` in the ledger names which clock could see it, so `silence` marks the
case the original test was structurally blind to. A channel that has never run at
all is measured from when its schedule appeared, or the same blindness returns by
another door — a new hourly cron that never fires once would otherwise sit at
`WITHIN_GRACE` for ever. `SCHEDULE_TOO_NEW` and `UNKNOWN` are decided before
either clock and still win.

One dropped hourly firing is still not an incident: measured drift here runs to
151 minutes, and 1 h 23 m of silence stays quiet.

### One refusal that had to be narrowed

Catch-ups were the `watchdog` lane's job *alone*, so that two schedules could not
race between reading the catch-up ledger and writing it and double-fire the same
slot. The second lane therefore only reported.

That is right for every channel except the actor itself, and there it is
circular: **`watchdog` cannot re-fire `watchdog`**, because the case in question
is precisely that `watchdog` did not run. Measured on 2026-08-27 — its 04:17 and
05:17 UTC firings were *both* dropped, and at 05:36, 2 h 23 m past a 2 h grace,
the only lane that could see it was the one forbidden to act. Reported, never
recovered, until a human happened to look.

So the second lane runs `--self-heal-only`: it may dispatch
`cronwatch.SELF_HEAL_ONLY` — `{watchdog.yml}` — and nothing else. The
single-actor rule stands for every ordinary channel, and no race is introduced,
because a `watchdog` that were running is a `watchdog` that is not overdue.

## The retry that made things worse

A re-run is not a fresh run. **It checks out the run's own `head_sha`**, and
that one fact turns the obvious retry rule — *it failed, try it again* — into a
way to undo a fix.

On 2026-08-27 at 03:17 UTC the hourly sweep re-ran `tocsin-altfeeds` run
33022081059: an ATLAS pass built from `1de2f34`, killed by its job timeout after
walking 133 minutes and writing nothing. The bug behind it — an ATLAS walk with
no wall-clock budget — had been found, fixed and merged hours earlier, and
`main` was six commits past it. The retry started the deleted code, on a
four-hour lane, ahead of two queued runs that carried the fix; had it reached
its commit step it would have written that superseded code's ledger over theirs.

So the rule the sweep was missing is the general one:

> If the commit a failed run was built from is no longer the head of **its own
> branch**, a re-run can only reproduce the failure or half-succeed and commit
> stale results over fresh ones. So it is not retried — it is recorded as
> `superseded`, naming both commits.

Judged per branch against that branch's head, so a feature-branch run is
superseded when *that* branch moves, not when `main` does.

There had been a guard: `results/watchdog/skiplist.json`, ten hand-listed run
ids. It is the wrong shape, and this is the proof — a list can only hold a run
someone thought of in advance, and this one was created, failed and resurrected
inside six hours. It is still honoured, for runs superseded in ways a sha cannot
express. It is no longer what stands between a fix and its own undoing.

The other four refusals, each previously implicit or absent:

* **A run that is not `completed` is not retried.** An in-flight attempt 2 is
  not a failure waiting for attempt 3, and asking for one races the attempt
  already running.
* **A branch whose head cannot be read is skipped, not guessed** — the same
  refusal this file already makes for an unanswering API. The count is printed
  loudly, because a sweep that retries nothing looks exactly like a sweep with
  nothing to retry.
* **At most five retries per sweep**, for the reason there are at most three
  catch-ups.
* **Every considered run lands in `retry` or `skipped` with a reason.** Silence
  about a decision is how this happened.

And the reason none of it was caught before it ran: those rules were twenty
lines of `gh api | jq` inside `watchdog.yml`, which no test could reach and
which could only be exercised by merging and waiting an hour. They are now
`src/seti/failsweep.py` under `tests/test_failsweep.py`, with the incident as
the first test and a wiring test that fails if the logic migrates back into the
workflow file. The ledger is also written on *every* sweep now, not only when
something failed — a file that appears only on bad days cannot tell "nothing
failed" from "the sweep did not run", which is this document's whole subject.

## The fifth silence: a result that was computed and never landed

Found at 04:58 UTC on 2026-08-27, and it is the worst of the family so far,
because the run is *green* and the work was *done*.

`tocsin-altfeeds` run 33029076779 spent **169 minutes** walking the ATLAS queue,
screened it, wrote `atlas/summary.json`, `atlas/events.json` and
`atlas/ledger_atlas.json`, and committed them. `main` had moved eight commits in
those 169 minutes. Then:

```
git pull --rebase --autostash origin main || true     # CONFLICT in census.json
git push origin HEAD:main || { ... retry ... }        # "Everything up-to-date"
```

The rebase conflicted on a generated JSON file that the concurrent run had also
rewritten. `|| true` swallowed it, leaving **a rebase in progress with the commit
unapplied**. `git push` then had nothing to push, printed *Everything
up-to-date*, and exited 0. The retry could not help: a second `git pull --rebase`
fails while a rebase is in progress, and its `|| true` swallowed that too. The
step exited 0, the run went green, and 169 minutes of queue time reached nobody.

And it would not have been noticed. The file it should have refreshed simply did
not change, so `alerts.STALE_DAYS` reads the channel as a little old rather than
broken — the same blind spot the empty-catalogue commit exploited from the other
direction. **A check that measures whether a file changed cannot tell you whether
the thing that changed it did any work — and it cannot tell you that the work was
done and thrown away either.**

`scripts/commit_results.sh` replaces that pattern in all ten scheduled
workflows, on two rules:

1. **Never rebase.** These are generated artefacts, wholly rewritten by the run
   that produces them. There is no line-level history to preserve, and a
   three-way merge of two machine-written JSON files is meaningless even when it
   succeeds. So the tree is re-created at whatever the branch head is *now* and
   the files are laid over it. A conflict is then impossible, and
   last-writer-wins is stated rather than discovered.
2. **Verify the result, not the call.** A zero exit from `git push` is not
   evidence that anything was pushed — *Everything up-to-date* is also zero. So
   the commit must afterwards be an ancestor of the **remote** ref, read back
   with `git ls-remote`. That is the check the old pattern lacked, and the only
   one that could have caught this.

Anything unresolvable exits non-zero, which turns the run red so the failure
sweep sees it. A loud failure beats a silent drop: the run that drops silently is
indistinguishable from one that had nothing to say.

`tests/test_commit_results.py` runs both patterns against real repositories with
a real conflicting change on the remote — including one test that asserts the
*old* sequence really did exit 0 while dropping the result, so if the premise
ever stops being true, that shows up as a failing test rather than as a rewrite
of this page.

The rule is enforced going forward: a test walks every **scheduled** workflow and
fails if it swallows the outcome of a `git pull` or `git push`. The line is drawn
at *scheduled* deliberately — a dispatch-only workflow is started by someone who
watches it finish, while a scheduled one has only its exit code to speak with,
and `|| echo` takes even that away. The other 76 dispatch-only workflows still
carry the old pattern and should be converted as they are touched.

## And one check that is about the repository, not the sky

The same sweep reads **the gate**: the newest `ci.yml` run on the default
branch. `watchdog`'s failure sweep skips `ci` deliberately —
`select(.name != "watchdog" and .name != "ci")` — because auto-retrying a
failing test suite argues with the signal instead of reading it. That is the
right call about *retrying*, and it left nobody *reading*: CI was red on `main`
from 2026-07-31 to 2026-08-22, twenty-two days, over one lint error, and it was
found by accident.

So the gate is read and reported, never retried. `RED` (failure, timed out, or
a startup failure — the shape a workflow file that will not parse takes) raises
a health alert keyed by the commit, so one broken head notifies once. `CANCELLED`
does not raise: a human or a superseding push stopped it, which is not a verdict
on the code. Neither does an API that would not answer — an unread gate is not a
green one, and it is not a red one either.

Every screen in this repository rests on those tests. A suite nobody checks
still passes is not a safety net; it is a story about one.

## Reading the output

`results/cronwatch/status.json` carries one record per scheduled workflow with
its matched cron, cadence, grace, the firing that was expected, the run that
actually happened, and the status. `results/cronwatch/state.json` is the ledger
of catch-ups already issued, keyed by workflow and missed firing, pruned at 120
days.

Anything overdue also reaches `results/alerts/latest.json` through
`alerts.scheduler_alerts`, keyed by the missed firing — so one dropped Wednesday
notifies once however many times the sweep re-reads it.

## The sibling failure this was found next to

The same day, `loom-catalogue`'s first run from `main` finished in four seconds,
went green, and committed an empty `NO_DATA_REACHED` record over 15,796 lines of
screened catalogue — and because the file it wrote was *fresh*, the staleness
check would have read the channel as healthy for another 45 days. That was a
resume keyed on the chunk list rather than the rows (fixed in
`src/seti/loom/catalogue.py`; see that commit), and it is worth recording here
because it belongs to the same family: **a check that measures whether a file
changed cannot tell you whether the thing that changed it did any work.**
