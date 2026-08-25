# The frozen Rubin frontier: mirror, or sky?

*Opened 2026-08-25. **Answered the same day: `SKY_STOPPED`.** Live state:
`results/rubin_outage/brokers.json`; the observatory's own words in
`results/rubin_outage/status_pages.json`.*

## The answer

**Rubin is not observing. Nothing is being missed, and nothing in the pipeline
is wrong.**

Two independent brokers stop on the same night, and their nightly histograms
agree count-for-count:

| night | ALeRCE detections | Fink alerts |
|---|---|---|
| 2026-07-11 | 125,278 | 125,442 |
| 2026-07-12 | 71,983 | 71,683 |
| 2026-07-13 | 744,559 | 744,559 |
| **2026-07-14** | **473,344** | **473,344** |
| 2026-07-15 → | *(nothing)* | *(nothing)* |

The stream ends on a **full-sized night**, not a taper — 473k alerts, then
silence. A mirror cut off mid-ingest would leave a short final night in one
broker and a complete one in the other; instead both stop together, on the same
night, at the same counts. Fink's LSST portal holds 132 nights in total and its
newest is 2026-07-14. `MIRROR_STALLED` is excluded.

**The observatory's own posts confirm it, and date it to the day**
(`results/rubin_outage/status_pages.json`, fetched from the Rubin forum's
Discourse API with per-post timestamps):

* **2026-07-15** — *"A major winter storm is expected in the next few days to
  impact observatory operations. All NOIRLab facilities on Cerro Tololo and
  Cerro Pachón are preparing for shutdown. Cerro Pachón will be evacuated later
  today."*
* **2026-07-20** — *"our access road is cut at the bottom near the main highway
  from La Serena … We will be off sky for days at a minimum."*
* **2026-08-02** — *"the going has been slowed by the large amount of snow high
  on the road to the summit. A new storm came in and stopped work on Friday."*
* **2026-08-21** (most recent) — *"getting a back up generator working reliably
  and then bringing up the lights for the first time since mid July inside the
  control room, level 3 and the dome … then look in the next period to the
  telescope, camera, and dome. Assessments continue, but the big three
  (telescope, camera, and dome) look good so far."*

Our last epoch, the night of 13/14 July, is **the last night before the
evacuation**. No restart date has been announced; as of 21 August the summit was
being brought back up on generator power, with water and fuel delivery named as
the prerequisites for sustained operation.

**What this means for the record.** Every null either Rubin channel files for
the interval from 2026-07-15 onward means *no new sky*, not *clean sky*. The
frontier alerts are correct and must not be relaxed: `FRONTIER_STALL_DAYS` is
measuring a real condition. `alerts.py::outage_context` now carries this verdict
into both frontier alert bodies, so a reader is not sent to re-diagnose it —
and drops it automatically once the frontier moves past the epoch it explains.
`frontier_recovery_alerts` will announce the restart.

### An incidental finding: ALeRCE's non-LSST feed stopped earlier

The survey-currency control (`alerce_tap.object` grouped by `sid`/`tid`) shows
`sid=0` — the non-LSST survey, 9.09M objects — with a newest epoch of
**2026-04-30**, two and a half months before the LSST feed stopped. So ALeRCE
offers no live control, and more practically: **routing a starved channel to
ZTF through the same broker is not an option.** Worth knowing before it is
proposed. Fink's ZTF portal (`api.fink-portal.org`) did not answer at all
(connect timeout); its LSST portal answered normally.

---


## The observation

Both Rubin channels have reported the **same newest epoch on every run since
2026-07-30**:

| channel | source field | frontier MJD | UTC |
|---|---|---|---|
| `tocsin` | `summary.json:broker_frontier_mjd` | 61235.41918 | 2026-07-14T10:03:37Z |
| `loom` | `screen.json:frontier_mjd` | 61235.41627 | 2026-07-14T09:59:26Z |

62 sightings, `n_advances = 0` (`results/alerts/frontier.json`). As of
2026-08-25 that is **26 days frozen** and **42 days behind the wall clock**,
against a normal ALeRCE mirror lag of ~16 days. Both frontier checks in
`src/seti/alerts.py` are firing and have been since 2026-08-07.

The channels themselves are healthy: they run on schedule, reach ALeRCE, get a
valid answer, and commit. `tocsin` reports `NO_NEW_DATA` — *"the watermark has
caught up with the broker's newest epoch"* — which is the correct verdict for a
screen that has nothing left to screen.

## Why the existing alerts cannot finish the diagnosis

Both alert bodies end with *"Check whether ALeRCE is still ingesting the LSST
alert stream."* That names one suspect out of two, and the two have opposite
consequences:

* **MIRROR_STALLED** — Rubin is observing; ALeRCE has stopped mirroring. We are
  blind to sky that exists. Every night of waiting is a night of real data going
  unscreened, and the fix is to add or switch brokers.
* **SKY_STOPPED** — Rubin is not observing. Nothing is being missed, no change to
  the broker path recovers anything, and the only available error is to read the
  run of nulls as a statement about the sky.

**A single broker cannot tell these apart.** ALeRCE answering "my newest LSST row
is 2026-07-14" is consistent with both. The discriminator has to be a second,
independent broker.

## The check

`scripts/rubin_outage_check.py` (workflow `rubin-outage.yml`, dispatch-only;
runner-only because broker egress is 403-blocked in the sandbox) asks each
reachable broker for its newest LSST epoch:

* **ALeRCE TAP** — `MAX(mjd)` over the `detection × lsst_detection` join (the
  `tocsin` frontier), over `lsst_ss_detection` (the `loom` frontier), and over
  bare `detection` (which carries ZTF too — an ALeRCE that is *current on ZTF and
  frozen on LSST* is a broken LSST ingest, not a dead service). Plus a **nightly
  detection histogram over the last 120 days**: a mirror cut off mid-ingest
  tapers into a short final night, whereas a stream that stopped at the source
  ends on a night of ordinary size.
* **Fink LSST** (`api.lsst.fink-portal.org`) — public, no auth. Endpoint names of
  the LSST portal are unverified from the sandbox, so several are tried and every
  raw response is recorded; the epoch is extracted by scanning the payload for
  time-like fields rather than by assuming a spelling, and JD is normalised to
  MJD.
* **Lasair-LSST** — only when `LASAIR_TOKEN` is configured; skipped cleanly
  otherwise.

Verdicts: `MIRROR_STALLED` (another broker is ahead by more than one night),
`SKY_STOPPED` (every broker reached stops on the same night),
`UNDETERMINED_SINGLE_SOURCE` (only ALeRCE answered — **not** to be read as
`SKY_STOPPED`), `NO_BROKER_REACHED` (our network failed; says nothing about
Rubin). The one-night tolerance exists because brokers ingest a night in batches
and their newest epochs drift by hours while both are current; treating that as
"ahead" would raise `MIRROR_STALLED` on ordinary behaviour. Pinned by
`tests/test_rubin_outage.py`.

## The hypothesis that got there first (confirmed 2026-08-25)

Public reporting places a **historic winter storm over Chile's Coquimbo region on
15–21 July 2026** — the worst since 1997, 13 dead, >100,000 residents cut off,
a national state of emergency declared 16 July. NOIRLab reports that all
facilities on Cerro Tololo and Cerro Pachón shut down, with snow accumulations
up to 3 m, the summit road unable to carry water and fuel trucks, further storms
halting recovery, and conditions finally clearing on **18 August** for road
inspection and snow clearing.

**Our last alert epoch is the night of 13/14 July — the last night before that
storm arrived.** The coincidence was close enough that a Rubin shutdown was the
obvious explanation, and it predicted `SKY_STOPPED`.

It was recorded as a hypothesis rather than a result because it rested on news
coverage rather than on data we hold, and both official sources
(`community.lsst.org`, `rubinobservatory.org`) are egress-blocked from the
sandbox. The cross-broker check settled it from the alert stream itself, and
`rubin_status_fetch.py` then read the observatory's own posts on the runner.
Both agree with it.

### Sources for the hypothesis

- <https://noirlab.edu/science/news/announcements/sci26037> — Storms impacting AURA/NOIRLab operations in Chile
- <https://noirlab.edu/public/images/20260803-update-04-CC/> — storm damage, Cerro Pachón
- <https://community.lsst.org/t/winter-storm-in-chile/12295> — Winter Storm in Chile (Rubin forum)
- <https://community.lsst.org/t/rubin-observatory-status/12397> — Rubin Observatory Status (Rubin forum)
- <https://www.upi.com/Top_News/World-News/2026/07/16/latam-chile-state-of-emergency-severe-weather/6351784217281/> — state of emergency, 16 July 2026

## What follows from each verdict

* `SKY_STOPPED` — nothing in the pipeline is wrong. Do **not** relax
  `FRONTIER_STALL_DAYS`: the threshold is doing exactly its job, and the
  condition it reports is real and ongoing. Every null filed while this holds
  means *no new sky*, and any occurrence statement covering this interval must
  say so. `alerts.py` already carries a frontier-**recovery** notification, so
  the resumption of observing will announce itself.
* `MIRROR_STALLED` — urgent. Route both channels through whichever broker is
  current (Fink for `tocsin`'s stellar sample; for `loom`, check first whether
  the substitute exposes `ssSource.ephOffset*`, which is the whole observable —
  no broker but ALeRCE is known to serve it, and if none does, the channel is
  blocked on ALeRCE regardless of what the others carry).
