# Unattended tracking and notification

How the Rubin channels keep running, keep committing findings, and reach a human
inbox when — and only when — something needs a person.

Nothing in this document requires a model in the loop. Every stage below is a
GitHub Actions cron that commits its own results back to the repository.

---

## 1. What runs, and when

Times are given in UTC (what the cron file says) and US Eastern (what you
experience). Eastern is EDT = UTC−4 through early November, EST = UTC−5 after.

| Workflow | Cadence | UTC | Eastern | What it does |
|---|---|---|---|---|
| `tocsin` | nightly | 14:10 | 10:10 | Screens the previous Chilean night's Rubin alerts for S30 flash/dip events on catalogued nearby dwarfs; updates the cross-night ledger |
| `loom` | weekly, Mondays | 15:40 | 11:40 | Screens Rubin solar-system detections against the radiation momentum ceiling and residual structure; runs the population replication tests |
| `loom-calibrate` | monthly, 1st | 16:20 | 12:20 | Recomputes the ceiling's realised efficiency over JPL's whole published non-gravitational catalogue; re-measures the `yarkovsky` unit |
| `loom-litcheck` | **chained** off `loom-calibrate` | — | — | Reads the survivor list the calibration just wrote and searches the dark-comet literature and arXiv for each designation |
| `alerts` | **chained** off `tocsin`, `loom`, `loom-litcheck`, **plus** a daily heartbeat | 17:05 | 13:05 | Decides whether a human has to look, and opens an assigned GitHub issue if so |
| `watchdog` | hourly | :17 | :17 | Auto-retries any run that failed, cancelled or timed out in the last 24 h |

The staggering is deliberate: the three data channels never contend for the same
public TAP service, and each is offset from the hour to avoid GitHub's `:00`
scheduler pile-up.

### Why two of them are chained rather than scheduled

`loom-litcheck` reads the survivor list `loom-calibrate` produces, and is the
only thing that can change it. On a clock it would either run before the list
exists or re-read a stale one.

`alerts` deliberately does **not** wake on `loom-calibrate`. It wakes on
`loom-litcheck`, so the chain is `calibrate → litcheck → alert`. Waking on the
calibration would evaluate exceedances *before* the literature check that
explains most of them, and notify about objects published months ago.

---

## 2. What counts as worth a human's attention

`src/seti/alerts.py`. Three severities, and they are genuinely different things.

**`candidate`** — a channel promoted something. This is what the apparatus
exists for. If it fires monthly, the thresholds are wrong, not the sky.

- TOCSIN put a target at tier `candidate`
- TOCSIN's population-structure tests returned anything other than a null
- LOOM put an object at tier `candidate`
- LOOM's replication tests detected population structure — *this is the
  channel's actual decision criterion*
- A **new** ceiling exceedance appeared in the monthly calibration that the
  literature check does not already explain

**`health`** — the pipeline is broken in a way that produces *no error*. These
matter more than they look, because every one of them turns into "a clean null"
if nobody notices, and a clean null is the failure this repository is most
exposed to.

- LOOM failed its positive control: a known artificial object was measured by
  the screen and *not* flagged. No null from the screen is interpretable until
  that is fixed.
- A channel wrote `verdict: NO_DATA_REACHED` — the broker was unreachable or
  answered unusably. That is not a null result and must not be read as one.
- A channel has gone quiet (§3).
- **The data has gone quiet even though the channel has not** (§3).

**`milestone`** — a capability came online. There is one that matters now:
LOOM's central discriminant is *which heliocentric-distance law the acceleration
follows*, which needs objects observed at two apparitions. The LSST survey began
2026-06-30, so every object has one, and the test has returned
`INSUFFICIENT_R_SPAN` on every object since the channel was built. The week that
changes is the week the channel starts being able to answer its own question,
and it is worth knowing that week rather than a year later.

---

## 3. The two things that make an unattended alarm work

### It notifies once

Every alert carries a stable `key`. `results/alerts/state.json` remembers the
keys already raised, and only **new** keys notify. Without this, the first
promoted candidate would email every week forever and the notification would be
trained into noise inside a month — at which point the apparatus has a working
detector and a human who ignores it.

An alert that has been consumed is still reported as **active** in
`results/alerts/latest.json`. Deduplication must not make a live condition look
resolved.

To deliberately re-raise something: dispatch `alerts` with `rearm: true`, or
delete its key from `state.json`.

### It notices silence

A screen that has silently died and a sky with nothing in it produce the same
unchanging results directory. `watchdog` catches runs that **fail**; nothing
else catches runs that **stop happening**. The daily 17:05 UTC heartbeat runs
whether or not any channel did, and alerts on the **age of the results** rather
than their content — `tocsin` after 4 days, `loom` after 10.

Staleness escalates on a doubling (4 d, 8 d, 16 d, …), not on every period. A
dead channel should keep nagging; nagging every four days for a year is how a
notification becomes noise.

> **The age is read from inside the file, not from its mtime.** A runner clones
> the repository fresh, so every file's mtime is the checkout time — a channel
> dead for a year would look thirty seconds old. `seti.alerts` reads the
> `run_at_utc` / `screened_at_utc` / `calibrated_at_utc` stamp each channel
> writes into its own result file. This is the difference between a staleness
> check and a staleness check that can never fire on the machine it has to work
> on.

### It notices the *data* stopping, which is a different question

Every check above asks whether the screens **ran**. This one asks whether they
ran on anything **new**, and it is the only one that can catch the failure both
Rubin channels are most exposed to.

Both read Rubin through ALeRCE's public mirror, and that mirror lags — 15.6 days
measured, with the frontier at MJD 61235 (2026-07-14) on 2026-07-30. If ALeRCE
simply stops ingesting LSST, both channels keep running on schedule, keep
writing a fresh run stamp, keep committing, and keep reporting a clean null.
Every liveness check stays green while the repository has silently stopped
tracking Rubin at all — and a clean null from a screen that is no longer being
shown any data is the single most misleading thing this apparatus can produce.

So `alerts` compares the wall clock against the frontier each channel reports
(`results/tocsin/ledger.json` → `last_mjd_screened`, `results/loom/screen.json`
→ `frontier_mjd`) and raises `health` past **30 days** — roughly twice the
measured mirror lag, so ordinary latency never fires and a stalled mirror shows
up inside a fortnight. A missing or zero frontier is treated as *unknown*, not
as a 61,000-day lag; zero is this repository's recurring "missing" value and an
upper bound that admits it turns every absent field into an alert.

### It notices the frontier *not moving*, which is earlier than it being old

The check above measures the frontier against the wall clock, so it cannot fire
until a freeze has burned through the whole 30-day budget. But the mirror is
already ~16 days behind *when* it stops — so a mirror that dies today is
reported in a fortnight, and the fortnight in between is a run of clean nulls
that mean "no new sky" being filed as "clean sky".

So there is a second check, measuring the frontier against **itself**: the same
epoch, run after run, is a stall regardless of how recent that epoch happens to
be. It needs memory, and `results/alerts/frontier.json` is that memory — nothing
else in the repository has it, because every channel's result file describes the
run that wrote it, and from a single file a frozen mirror and an advancing one
are indistinguishable.

```
results/alerts/frontier.json
  channels.<name>.mjd                     the newest epoch now visible
  channels.<name>.first_seen_utc          when it last MOVED — the stall clock
  channels.<name>.last_seen_utc           the most recent sighting of that value
  channels.<name>.history                 previous values, with their spans
  channels.<name>.observed_advance_days   the measured ingest cadence
```

Two details carry the whole check. `first_seen_utc` is preserved by a run that
sees no change — if an unchanged frontier rewrote it, every run would reset the
clock and the alert could never fire while looking perfectly well-implemented.
And the frontier is only recorded on the **recording** pass, never the dry run,
for the same reason.

It reads the **broker's** frontier in preference to the **screened** one
(`results/tocsin/summary.json` → `broker_frontier_mjd`, falling back to
`ledger.json` → `last_mjd_screened`). The two are equal while a channel is
caught up, but they come apart exactly when it matters: if the channel breaks
while the mirror keeps advancing, the screened frontier freezes and the data has
not stopped at all — reporting that as a mirror outage sends you to the wrong
system.

**The threshold is 7 days, and it is a placeholder.** ALeRCE's ingest cadence
has not been measured here: the frontier sat at MJD 61235.41918 unchanged across
every run from 2026-07-30 to 2026-07-31, which is the only observation there is.
A threshold below the real batch interval would fire on ordinary behaviour and
be trained into noise inside a month — the failure this whole module is built to
avoid — so 7 days is deliberately conservative, above any plausible batching and
still four times faster than the age check. `observed_advance_days` accumulates
the real cadence run by run; tighten `FRONTIER_STALL_DAYS` from that record
rather than by guessing again.

The frontier is reported in `results/alerts/latest.json` and in the workflow log
**whether or not it alerts**, because below the threshold a stalling mirror is
invisible in every other field, and "how old is the newest sky we have seen" is
the first thing to check before reading any null as a statement about the sky.

### It notices the frontier *starting* again, which is visible only once

The stall check answers "has the data stopped". Nothing answered "has it
started", and the two are not symmetric. A stall is visible in every run that
follows it, so it can be noticed late and still be noticed. A recovery is
visible in **exactly one** run — the first whose frontier differs from the
record — and afterwards the mirror simply looks healthy, as though it always had
been. Miss that run and the only trace is a number in `frontier.json` that
quietly changed.

So `frontier_recovery_alerts` raises a `milestone` when a channel's frontier
advances past the recorded one. It is gated on the **same threshold as the
stall check**, so a recovery fires if and only if the stall it ended had itself
been worth reporting: this alert is always the answer to a question already
asked, never an unprompted one. An ordinary nightly advance notifies nothing —
a healthy mirror advances every night, and reporting that is the noise this
module exists to avoid. A frontier that moves *backwards* is not a recovery
either; that is a broker re-indexing or a channel reading a different table, and
announcing it would promise data that is not there.

The alert carries the **advance interval** — first sighting of the old value to
first sighting of the new one, measured exactly as `observed_advance_days`
measures it — which is the cadence datum `FRONTIER_STALL_DAYS` has never had. It
is an *upper bound* on the mirror's true ingest gap: the frontier is sampled
about once a day and a change finer than that cannot be seen from here.

It also says whether the mirror is merely *inside the age limit* or genuinely
current, because one advance after a long outage may be a partial backfill, and
a null read during the catch-up still means *no new sky*.

**A recovery re-arms the stall keys it answers.** Without this the stall
detector is one-shot for the life of the repository: the stall key escalates by
doubling (`:0`, `:1`, `:2`), so once reported those keys are consumed forever and
the *next* outage stays silent until it grows past the longest escalation already
seen. This repository consumed `:0` and `:1` during the July 2026 Rubin outage,
which would have bought the next stall 28 days of silence and the one after that
56 — a detector whose blind spot doubles every time it fires is worse than none,
because the silence still reads as health. The recovery is the honest moment to
clear them: the condition has demonstrably ended, so re-arming announces nothing
and loses nothing. Keys whose condition is **still true** are never cleared, or
they would re-raise on the next run and re-notify about something already sent.

### GitHub's own inactivity policy

The other thing that can stop a cron is GitHub's own policy: **scheduled
workflows are disabled after 60 days without repository activity**. The channels
commit their results back, which is activity, so this is self-sustaining as long
as at least one of them is working. If they all stop, nothing re-arms them
automatically — see §6.

---

## 4. How it reaches your inbox

`alerts.yml` opens a GitHub issue and **assigns it to the repository owner**.

Assignment — not the issue — is the mechanism that actually reaches an inbox.
Email on a push depends on the recipient's watch settings, which this repository
cannot set for you. GitHub emails an **assignee** regardless of watch settings.

The issue is labelled by severity (`candidate` / `health` / `milestone`) and by
channel (`tocsin` / `loom`), so an inbox filter can separate "something was
promoted" from "the pipeline is broken".

### Order of operations

The alerts are evaluated in `--dry-run` first, the issue is opened, and *only
then* is the state recorded and committed. A crash between those two steps
produces a duplicate notification; the reverse order produces a **lost** one.
Duplicated is recoverable, lost is not.

### What the issue says

An alert is a request for attention, not a claim. Each entry states what was
measured, what the channel's own verdict was, which file to read, and — where
the repository has already been burned — what to check before believing it. It
never says a thing was found.

---

## 5. Where findings land

Everything is committed back to `main` by the runner that produced it.

```
results/tocsin/ledger.json        cumulative target x night state; the trial denominator
results/tocsin/summary.json       the night's screen
results/tocsin/assessment.json    tiers over the whole ledger
results/tocsin/population.json    population-structure tests
results/loom/screen.json          the funnel, the frontier, coverage
results/loom/objects.csv          per-object tiers and reasons
results/loom/assessment.json      replication tests, positive controls
results/loom/calibration.json     the ceiling's realised efficiency; exceedances
results/loom/litcheck.json        which exceedances the literature already explains
results/alerts/latest.json        every active alert, and which are new
results/alerts/state.json         the deduplication memory
results/alerts/frontier.json      where the broker mirror was on previous runs
results/watchdog/status.json      unresolved breakage
```

---

## 6. What is NOT automated (i.e. what still needs you)

1. **Your GitHub notification settings.** This repository can assign you an
   issue; it cannot configure your email. Check
   <https://github.com/settings/notifications> — "Participating and @mentions"
   must be set to deliver to email. Assignment counts as participating.
2. **Re-enabling scheduled workflows** if GitHub disables them for inactivity
   (60 days with no repository activity). GitHub emails the repository admin
   before doing it, and re-enabling is a button in the Actions tab.
3. **Judging a candidate.** The alert tells you which file to read and which
   systematic burned this repository last time. It does not adjudicate.
4. **Deciding a threshold was wrong.** If a `candidate` alert fires often, that
   is a statement about the thresholds, and no automation should quietly widen
   them to make itself quiet.

---

## 7. Verifying it works without waiting for the sky

```bash
# What would fire right now, without consuming anything:
python -m seti.cli alert-check --dry-run

# Force a notification end-to-end (opens a real, assigned issue):
#   Actions -> alerts -> Run workflow -> rearm: true

# Evaluate and commit state without opening an issue:
#   Actions -> alerts -> Run workflow -> no_issue: true
```

`tests/test_alerts.py` pins both directions: that an ordinary null run is
silent, and that each condition above fires exactly once.
