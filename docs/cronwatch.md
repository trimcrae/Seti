# The third silence: a screen that never started

*Opened 2026-08-26. Code: `src/seti/cronwatch.py`, `scripts/cron_watch.py`,
wired into `.github/workflows/watchdog.yml`. Live state:
`results/cronwatch/status.json`.*

## The gap this closes

Every channel here can fall silent in three different ways, and until now only
two of them were watched.

| what stopped | what it looks like | who catches it |
|---|---|---|
| the run **failed** | a red run in the Actions tab | `watchdog`, hourly, and it re-runs the failed jobs |
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
