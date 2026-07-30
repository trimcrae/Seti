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
