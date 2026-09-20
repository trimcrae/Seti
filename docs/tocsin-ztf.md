# TOCSIN on the live ZTF alert stream

*Opened 2026-09-05. Code: `src/seti/tocsin/ztf_live.py`; tests:
`tests/test_tocsin_ztf_live.py`; workflow: `.github/workflows/tocsin-ztf.yml`;
config: the `ztf:` block of `config/tocsin.yaml`. Results:
`results/tocsin_ztf/`.*

## 0. Why, and why now

Rubin has been off sky since the night of 13/14 July 2026
(`docs/rubin-outage.md`, verdict `SKY_STOPPED`), and the substitute work in
`docs/tocsin-altfeeds.md` was built on light-curve services because the
repository believed the ZTF public stream had ended. That belief rested on two
facts about **brokers** — ALeRCE's TAP mirror stopped carrying non-LSST data on
2026-04-30, Fink's ZTF portal is unreachable — and on 2026-09-05 `rubin-outage`
measured the stream itself: ZTF's own nightly alert archive, ALeRCE's separate
ZTF REST API and ANTARES all held the night of 2026-09-04, 0.6 days behind the
wall clock. **ZTF is observing and its public alerts are being served.**

That matters because ZTF is the one public feed shaped like Rubin's: a nightly
difference-image *alert stream*, both polarities issued (a dip is an alert with
`isdiffpos = f`), 30-second visits, r ≲ 20.5, over the whole northern sky. It is
shallower than Rubin by ~4 magnitudes and blind south of δ ≈ −31°, but it is
live, and the S30 signature — an unclassified achromatic transient on a
catalogued nearby star, promoted only on **coherence across nights** — transfers
to it with the funnel, the ledger and the target model unchanged.

## 1. What is reused and what is new

Reused verbatim: `screen.screen_alerts` (the per-event funnel), `ledger.Ledger`
(cumulative trials, BH-FDR, duty cycle, cadence-matched timing null),
`targets` (Gaia DR3 nearby stars, proper-motion propagation, the matcher). This
module only *acquires* and *normalises*, against two public services:

| service | role | shape |
|---|---|---|
| **ALeRCE ZTF API** `api.alerce.online/ztf/v1` | the numerator | `/objects` by `lastmjd` window; `/objects/{oid}/detections`; `/objects/{oid}/non_detections` |
| **IRSA TAP** `ztf.ztf_current_meta_sci` | the denominator | every public science quadrant: `obsjd, fid, ra, dec, ra1..dec4, maglimit, programid` |

Neither needs a credential. Both are documentation-derived until the workflow's
probe (`tocsin-ztf-probe`, first step, always) records their live shapes into
`results/tocsin_ztf/probe.json`; every field is read by name through tolerant
lookups so a missing column degrades one discriminator to *untestable* rather
than crashing the night.

## 2. The sweep

A window `[lo, hi)` is one to three nights. The run lists every ALeRCE object
whose **newest** detection falls in the window (`lastmjd` range, paged by
`has_next`, ~300 pages of 1000 for a full night), cross-matches their mean
positions to the propagated northern target list (139k-scale, `cKDTree`), and
for each matched object fetches its full detection history and its upper limits.
Matched objects are few — nearby stars that alerted — so the per-object cost is
small; `max_matched_objects` guards the pathological night where a reference
image changed under thousands of stars.

**Why listing by `lastmjd` is correct for a backfill.** For the live nightly run
"newest detection in the window" is exactly "alerted last night". Backfilling,
an object is met once, in the window of its *last* alert, with its whole history
attached; an event on an earlier night is folded if that night's trials are
already in the ledger and is otherwise **deferred** for the sweep to reach
(`events_deferred_to_sweep` in the summary). Numerator and denominator
therefore always cover the same set of nights.

## 3. The denominator: quadrants, not proxies

Rubin's channel had to reconstruct "which stars were looked at" from a
1-degree binning of where detections happened, because the broker's forced
photometry covered 0 % of star-nights. Here the survey publishes the answer:
IRSA's exposure table holds every public science quadrant with its four corners
and its own 5σ limiting magnitude. `quadrant_footprint` propagates the targets
to the window's epoch and runs a gnomonic point-in-polygon test per quadrant
(candidates pre-selected with a 0.75° KD-tree ball), yielding:

* the trial set `{(target, night)}` — the ledger's denominator;
* the bands observed per star-night — for the funnel's one-sided
  non-detection test ("a grey event this large would have shown in the band
  that stayed silent");
* the quadrant's own limit per star-night-band — the *measured* threshold
  that test needs, rather than an estimate from the night's flux errors;
* every exposure epoch that covered the star — the per-visit history the
  timing null resamples.

The object's own upper limits (`non_detections`) are merged into the visit
history and limits but **never added as trials**: the footprint already counts
those star-nights, and counting them twice is the denominator bug the
alternative feeds had (`docs/tocsin-altfeeds.md` §12.4).

**The window never advances past the exposure table's frontier**
(`frontiers.irsa_exposures_mjd`). IRSA's metadata lags the stream by more than
the brokers do, and a night folded without its trials would count events
against nothing; the cap costs only that the newest night or two is screened on
the next run.

## 4. Normalisation choices that decide the physics

* **Sign.** `magpsf` is the magnitude of |ΔF|; `isdiffpos` (`t`/`f`, or ±1)
  is the sign. A dip is a first-class alert.
* **Flux.** AB: `F[nJy] = 10^((8.90 − m)/2.5) × 10^9`; `σ_F = F σ_m ln10/2.5`.
* **The quiescent flux F\*.** ALeRCE's `magpsf_corr` is the difference
  photometry combined with the reference-catalogue magnitude at the same
  position — the total apparent magnitude. Total minus the signed difference is
  the **reference flux in the same band and system**, the analogue of Rubin's
  `templateFlux`, so dF/F\* is a ratio of two ZTF measurements with no passband
  transformation. Where `corrected` is false (no reference source within 1.4″)
  the field is left `None` and the funnel falls back to Gaia GSPC with its
  passband error, as on the Rubin path.
* **Reliability.** `drb` (the deep real/bogus score), `rb` when absent;
  `min_reliability` stays at the configured 0 for the same reason as on Rubin
  (stellar subtractions score low; the incompleteness is reported, not
  deepened).
* **Quality.** ALeRCE's `dubious` flag stands in for pixel flags. ZTF alerts
  carry no per-detection astrometric error, no dipole, trail or extendedness
  fields and no solar-system association; the funnel records each as
  unavailable and the ledger's recurrence requirement carries the load, as it
  does for the alternative feeds.
* **Bands.** `fid` 1/2/3 → g/r/i, carried under the shared schema's labels; ZTF
  g/r/i are close to SDSS g/r/i, the same standing as the ATLAS mapping.

## 5. What this channel can and cannot say

Preserved: both polarities; the cross-night recurrence ledger (the channel's
actual instrument); an **exact, survey-published denominator**; the 30-second
visit, so a sub-second glint is diluted exactly as in a Rubin visit; the
intra-night g+r pair on many nights, so the achromaticity test runs routinely.

Lost: ~4 magnitudes of depth; the southern sky; Rubin's morphological flags
(dipole, trail, extendedness) and its solar-system association; independent
per-detection astrometric errors. Every event carries the corresponding
`*_unavailable` reasons.

## 6. Running it

Nightly at 11:25 ET (`tocsin-ztf.yml`), two windows per run until the backfill
from 2026-01-01 catches up with the frontier, then one night per night. Dispatch
inputs: `chunks`, `mjd_lo`/`mjd_hi` (an explicit window does not move the
watermark), `probe_only`, `rebuild_targets`. Outputs committed per run:
`probe.json`, `summary.json`, `ledger.json`, `assessment.json`,
`watchlist.csv`, `events_latest.csv`, `rejected_latest.csv`,
`excluded_targets.csv` and `vet/<source_id>.json` for every candidate-tier
target (§8d). `tocsin-ztf-vet.yml` vets any star by id on demand.

## 7. What the first live runs taught (2026-09-05, runs 1-4)

Every field name in §1-§4 was documentation-derived until the probe recorded
the services; `results/tocsin_ztf/probe.json` now holds the live shapes and
they match — every detection, non-detection and object key listed above is
present, `isdiffpos` is served as ±1, and IRSA's quadrant table has every
column named. The northern list is 166,899 stars. What did not match was on our
side, and each item is now pinned by a test:

| run | what happened | fix |
|---|---|---|
| 1 | The sweep took one page of 1000 and called a three-night window complete: with `count=false` the service answers `page: null, has_next: false` on every page. | Pagination no longer trusts `has_next`. |
| 1 | IRSA returned a VOTable error: the table has no `programid`; the public-survey column is `ipac_gid`. | `ipac_gid = 1`; the error text is extracted, not the document head. |
| 1 | All 18 detections of three catalogued stars were rejected as astrometric offsets at 3–28σ: ZTF alerts carry no position error, so Rubin's 50 mas floor applied to centroids sitting 0.16–1.4″ from the propagated Gaia position while ZTF's own `distnr` put each within 0.1″ of its reference source. | A measured 0.25″ per-axis floor (`ZTF_ASTROMETRIC_FLOOR_ARCSEC`); 1″ is the 3σ line. |
| 1 | Four nights were folded with zero trials when the exposure query failed, and the watermark advanced past them. | `NO_DENOMINATOR`: nothing folded, watermark kept. |
| 3 | One three-night `lastmjd` query paged 27 minutes into the offset and drew HTTP 500. | — |
| 4 | A quarter-night slice ordered by `oid` drew nginx 504 on page 1: ordering by `oid` walks the id index filtered by epoch, so a *narrow* window is *slower*. | **Keyset pagination on `lastmjd`**: order by the filtered column, advance the lower bound to the newest epoch seen, de-duplicate the boundary. |
| 4 | IRSA's `MAX(obsjd)` over the whole table read-timed out at 240 s. | Bounded lookback (3 d, widened only when empty). |
| 4 | IRSA's public exposure table is **60 days behind the stream** (frontier MJD 61228.19 = 2026-07-07 against a stream at 61288.17). | The stream caps the window; nights the table has reached to their end get the exact quadrant denominator, later nights the detection-footprint proxy (§3), recorded per night. |

The proxy is the Rubin path's own denominator and carries the same caveat: a
1° bin over-counts trials slightly relative to a 0.73° quadrant and carries no
per-visit limit, so events on proxy nights cannot use the one-sided
non-detection test. When IRSA's table catches up with a night that was folded
on the proxy, that night is not re-folded — the summary's
`denominator_by_night` is the record of which kind each night got.

## 8. The first complete window (run 5, 2026-09-05 05:03–06:33 UTC)

The keyset sweep worked. The first chunk, nights 60676–60679 (1–4 January
2026), went end to end and folded:

| | |
|---|---|
| objects that alerted in the window | ~114k per 1.2 nights (~95 pages of 1000 per night) |
| alerts on catalogued nearby stars | 902 |
| events kept by the funnel | 64, on 64 stars (all `watch`: single events) |
| trials (star-nights, exact quadrant footprint) | 111,889 over 95,171 stars |
| ensemble rate per star-night | 5.7 × 10⁻⁴ |
| rejections | 196 astrometric offset · 162 chromatic · 140 low significance · 76 `dubious` · 4 mixed polarity same night |

The colour test is running (162 chromatic rejections — flares — is the
discriminant the ZTF glint search could rarely apply, working here on the
intra-night g+r pairs), `visits_exact` is true for every target, and every
event carries the unavailability reasons of §5.

What the run also measured: **~18 s of service latency per page of 1000**, so a
night is half an hour serially, and the second chunk began with 43 minutes left,
swept for 39, and was truncated by the deadline — 127 pages that folded nothing.
Both are fixed: the window is now swept by four keyset walkers over equal
sub-ranges (`sweep_workers`), and a chunk is started only if the previous one
would fit in the remaining budget. `run.json` records every chunk of a run.

**Where the backfill starts.** The ledger was reset and the sweep restarted at
MJD 61235, the night of 2026-07-14 — Rubin's last night. The channel exists to
cover the Rubin-dark interval; at ~8 minutes a night that is reached in about a
week of nightly runs, after which each run screens the previous night. Earlier
nights can be folded later with explicit `mjd_lo`/`mjd_hi` windows, which fold
their nights but never move the watermark.

### 8a. The first parallel run (run 6, 06:38–07:58 UTC) and the night-boundary bug

Four sweep workers: six nights (2026-07-14 to 07-20) in 75 minutes, 462k
objects swept, 76 catalogued stars alerted, 1,755 detections, 31 events folded
(121 dips and 54 flashes among the 175 kept before the fold rule), 44 of them
colour-tested on same-night g+r pairs, one star already at `interest`. Every
night lay beyond IRSA's frontier and used the detection proxy: 140,180
star-nights.

It also exposed a bug: a night label runs 16:00 UTC to 16:00 UTC, so a window
cut at an integer MJD begins inside the night the previous chunk had just
folded with almost no trials, and the ledger's night-level de-duplication then
dropped the real ones — one night in three without a denominator. Windows now
sit on night boundaries (`night_start`), a night the frontier falls inside
waits for the next run, and the ledger was reset once more so those six nights
are refolded whole.

### 8b. Run 7 (08:02–09:26 UTC): the six nights refolded whole

| | |
|---|---|
| nights | 6 (2026-07-14 → 07-20), all on the detection proxy |
| objects swept | 461k in 515 pages (four workers) |
| catalogued stars that alerted | 79 |
| detections on them | 1,771 |
| events kept by the funnel / folded into these nights | 453 / 32 (the rest are on earlier nights the sweep has not reached and are deferred) |
| trials (star-nights) | 163,768 — against 140,180 for the same six nights with the boundary bug |
| tiers | 30 `watch`, 1 `interest` |

The `interest` star is Gaia DR3 4276040238425545344 (RA 273.07°, Dec +1.54°):
two grey-tested dips of 17 % and 18 % in g and r on consecutive nights, 59
visited nights, duty cycle 0.03. Two consecutive nights is what a single
multi-day event looks like as much as what a repeater does, and the tier says
exactly that — *interest*, not candidate. It is the first entry on the ZTF
watchlist and nothing more yet.

Throughput after run 7: six sweep workers and three chunks per run, with a
second daily firing at 23:25 ET while the backfill catches up (~47 nights to
go, ~9 nights per run).

### 8c. Run 17 (2026-09-10, 08:15–08:57 UTC): the first two candidates — as read then, and as corrected in 8d

Run 17 folded nights 61288–61290 and the ledger promoted two stars to
`candidate`; the alerts workflow opened issue #10 at 17:42 UTC. Both were
rejected on inspection as **saturated stars**, on this reading:

| star (Gaia DR3) | events | amplitude *as read on 2026-09-10* | real/bogus (`drb`) | baseline *as inferred then* |
|---|---|---|---|---|
| 2752213329586862976 (RA 2.32°, Dec +9.00°) | 6 flashes, g+r+i, all grey | +6 to +10 % | 0.21–0.38 | g 11.3, **r 10.6**, i 9.6 |
| 4497414466452138496 (RA 274.27°, Dec +13.47°) | 6 flashes, g+r, all grey, plus one i-band outlier at `drb` 0.22 | +1 to +4 % | 0.95–1.00 | g 12.4, **r 11.5** |

**That reading was wrong, and 8d corrects it.** The ledger's `a` for these
events was 1.03–1.11, not 0.03–0.11: the fractional amplitude was *one hundred
and three per cent*, measured against the stars' own Gaia synthetic baseline
(`baseline_source: gaia_gspc_synthetic` on every event, in the pre-reset
ledger at commit `30c947e`). Reading 1.03 as "+3 %" and dividing the difference
flux by 0.03 manufactured the "r 11.5" and "r 10.6" baselines; the stars are
G 15.6 and fainter than 13 in every band (both are in the list rebuilt with the
G > 13 cut, and the second is on the current ledger at `interest`). Neither is
saturated. What both had was a difference flux **equal to the star's own
flux** on every visit — the star absent from its own reference image — which
is the mechanism 8d establishes for the first with the runner's evidence.

The two rules the run added remain, on their own merits rather than on the
run-17 evidence: `min_reliability: 0.5` on `drb` (the first star's alerts
really were scored 0.21–0.38, below braai's own threshold); and
`saturation_mag: 13.0`, which is correct physics for a 30 s ZTF exposure
(Masci et al. 2019) even though neither run-17 star was an instance of it
(`n_removed_saturated: 25` at the rebuild). The timing signature noted then —
best period 1.00 d, Rayleigh concentration 0.93–0.96 — is real and is the
signature of *any* per-visit residual, saturated or not.

Two rules, both in the `ztf:` block of `config/tocsin.yaml`, both recorded in
each run's `summary.json` under `ztf_thresholds`:

* **`saturation_mag: 13.0`.** Applied twice. The northern list is built without
  stars brighter than this in Gaia G or in GSPC g or r (`n_removed_saturated`
  in the build record) — they are not trials, since no visit of them can yield
  a valid alert. And the funnel rejects any alert whose star is brighter than
  this in the *alert's own band* as `saturated_target`, using the same baseline
  the amplitude is measured against (ZTF's reference flux where `corrected`,
  else the GSPC synthetic magnitude), so an i-band alert on a star saturated
  only in i is rejected while its g and r visits stay trials. Every event now
  records `baseline_mag` per band.
* **`min_reliability: 0.5`.** Braai's own decision threshold (Duev et al.
  2019) on `drb`, and the same floor on the rare `rb`-only alert.

**The ledger was reset.** The two cuts change the population, not only the
verdict: with saturated stars in the list the denominator counted non-trials
and the numerator carried their residuals, so the ensemble rate every
per-target p-value is measured against was wrong in both directions at once.
Purging the events and leaving the trials would deflate the rate and make
every surviving p-value anti-conservative. The backfill restarts at MJD 61235
against the rebuilt list (the cache key follows the config), at ~18 nights a
day. The seventy `interest`-tier stars of the old ledger are re-derived by the
sweep, not carried over.

**The same question for the Rubin list.** Rubin saturates near r ≈ 16 and the
Rubin target list (`target:` block, `g_max` only) has no bright cut either; the
funnel's `saturation_mag` is `None` there. That is a change to make before
Rubin returns, with its own ledger reset, and is not made here.

### 8d. 2026-09-20: the candidate that had walked off its own reference image

The nightly run of 04:47 EDT (08:47 UTC) folded night 61301 and the assessment
at 05:13 EDT promoted **Gaia DR3 4497414466452138496** — the same star as
run 17's second candidate — to `candidate`: 11 flash events on 11 nights
between 61234 and 61296, all grey (9 colour-tested, |z| ≤ 1.04), fractional
amplitude 0.97–1.07 in g and r, "colour temperature" 4590–4900 K, duty cycle
0.13 over 83 visits, p = 4.2 × 10⁻⁵ against the local rate. The alerts
workflow opened issue #15 at 1:27 PM EDT.

**What the star is.** Parallax 66.11 mas (15.1 pc), proper motion
(−436, −1116) mas/yr = **1.20″/yr**, G 15.62, BP−RP 1.15, RUWE 1.02, no
non-single-star flag; M_G = 14.7 — a cool white dwarf at 15 pc. The Gaia
synthetic g − r of 0.74 corresponds to ~4700 K, i.e. the "colour temperature"
of the excess is the colour of the star.

**What the runner measured** (`tocsin-ztf-vet`, run 35527006716, 1:48 PM EDT;
`results/tocsin_ztf/vet/4497414466452138496.json`):

| | |
|---|---|
| ALeRCE objects within 5″ of the propagated position | two: ZTF25aabynax (185 detections, 2025-01-14 → 2026-08-06, 0.09″ off) and ZTF26abftlhy (192 detections, 2026-06-09 → 2026-09-19, 0.39″ off) — ZTF opens a new object as the star walks out of the old one's 1.5″ association |
| `distnr`, distance to the nearest reference-catalogue source | median **9.5″ and 9.7″** (min 0.5–0.7″); `corrected` on 1 % and 9 % of detections |
| difference flux / the star's own Gaia flux | g **1.03, 1.04**; r **1.02, 1.02** (MAD 0.01) over 96 + 96 and 87 + 78 detections |
| `drb` | median 0.99 and 0.98 |
| Gaia DR3 within 60″ | 58 sources; nearest 10.0″ (G 16.4, parallax 0.26 mas — background); **none brighter than 13** |
| detections on nights the ledger counts (61235–61302) | **51 of the 51 nights the star was visited** — 11 folded as events, the rest rejected as `bad_pixels` (ALeRCE's `dubious`) or `chromatic` |

The star has moved ~9″ since the ZTF reference epoch. At its present position
the reference image holds nothing (the nearest catalogued reference source is
the background star 10″ away), so the difference image holds **the whole
star**: a "flash" of exactly the star's flux, with exactly the star's colour,
on every visit, in every band. It is the *proper-motion dipole* the funnel's
docstring names as the systematic of a nearby-star sample, in the form ZTF
serves it — the positive lobe alone, at the propagated Gaia position, with no
dipole flag to reject it on. The nightly funnel kept the nights on which the
`dubious` flag happened to be clear and g and r happened to agree within 3σ;
the ledger saw eleven grey repeats and did what it is built to do.

**Why the ledger's own defence did not fire.** The duty-cycle veto exists for
exactly this — a residual repeats at every visit, an event does not — and it
reads 11/83 = 0.13 against a limit of 0.2. The 83 "visits" were wrong: the
object's own history (its upper limits and detections) was being merged into
the star's visit list *from the object's first alert in January 2025*, months
before the ledger's first night, so nights on which no event could ever have
been folded were counted as trials. Over the nights the ledger actually
counts the star was visited on 51 and alerted on 51, and the veto would have
rejected it at 11/51 = 0.22. The same inflation touched 195 of the 225
targets on the ledger (16,523 visit-nights dropped by the correction below).

**Run 17, re-read.** The pre-reset ledger (commit `30c947e`) shows the same
star with 7 events at a = 1.01–1.06 against `gaia_gspc_synthetic`, and run 17's
other candidate, 2752213329586862976, with 6 events at a = 1.03–1.11 against
the same baseline. Neither was "+1 to +4 %" nor "+6 to +10 %", neither had a
ZTF reference of r 11.5 or r 10.6, and neither is saturated (8c, corrected).
Both were this mechanism. The second is on the current ledger at `interest`
and is vetted with this run.

**Four changes, each pinned by a test** (`tests/test_tocsin_ztf_vet.py`):

1. **The funnel subtracts a persistent level** (`rebaseline_persistent_residuals`).
   Per object and band, a series of ≥ 10 detections that is a *level* and not
   a light curve — ≥ 90 % one sign, scatter about a 60-day running median
   below 20 % of the median — has that level subtracted from every alert,
   with the MAD added to the error. This is ALeRCE's own `magpsf_corr`
   correction applied where ALeRCE cannot apply it (no reference source
   within 1.4″): the level *is* the star's missing reference flux. An
   ordinary visit becomes ~0 ± MAD and fails `low_significance`; a real
   departure survives at its true size — a flash as a > level, a dip as
   a < level — so the star **stays a trial**. A variable crossing its
   reference mean fails the sign test; a flare star keeps its flares.
2. **Visits are the ledger's nights** (`Ledger.restrict_visits_to_ledger_nights`,
   and the same filter in `screen_window` before folding). Applied at every
   assessment, so the existing ledger is corrected in place.
3. **A star vetted as no valid trial is taken out with its trials**
   (`Ledger.remove_targets`; `tocsin-ztf-assess --remove ID --remove-reason …`).
   Under the old funnel every visit of this star was already an "event", so a
   real flash on it could not have been distinguished; its 51 visits were not
   trials. Its events and its visits go together — neither purging the events
   alone (which deflates the ensemble rate, 8c) nor leaving them in (which
   inflates it). The removal is recorded in the ledger's `removed`.
4. **Promoted targets are vetted before the alert fires** (`tocsin-ztf-vet`,
   stage 4 of the nightly workflow, and its own dispatchable workflow):
   Gaia DR3 within 60″, every ALeRCE object within 5″ with its full history
   and the `distnr`/`corrected`/reference-magnitude fields the funnel
   discards, SIMBAD and VSX (at J2000, the radius widened by the proper-motion
   drift — the first vet asked at the 2026 position and missed a star that
   has moved 32″ since 2000), and IRSA's DR light curve (at the reference
   epoch, likewise widened). The record carries named flags and one
   classification; this star's is `systematic:proper_motion_reference_artefact`
   on `self_flux`, `high_proper_motion_drift` (9.1″) and
   `no_reference_source_at_position` (`distnr` 9.5″).

And one rule written on the way that is kept because the mechanism it closes
is real, though it was not this star's: **a target inside a saturated
neighbour's exclusion radius is not a trial** (`neighbour_radius_arcsec: 5.0`
at the saturation magnitude, ×10 per 5 mag brighter, capped at 120″;
`targets.flag_bright_neighbours`). The saturation cut sees only the target's
own magnitude; a saturated star's residual lands wherever its PSF does,
including on a faint catalogued star beside it. The list is rebuilt under the
new cache key with a Gaia scan for stars brighter than 13 (36 RA stripes; a
stripe that fails is named in `targets.json` and its targets stay unflagged
rather than silently isolated); the dropped stars are committed to
`results/tocsin_ztf/excluded_targets.csv` and pruned from the ledger by rule 3.
Expected cost: of order 10⁻³ of the list.

**Disposition of the candidate:** rejected, `systematic:proper_motion_reference_artefact`;
removed from the ledger with its 51 visits; issue #15 closed with the vet
record. Under rule 1 the star remains in the list and is screened for
departures from its level from the next night on.

## 9. Status

Live since 2026-09-05, twice daily (11:25 and 23:25 ET); the backfill from
2026-07-14 caught up with the stream on 2026-09-20 and each run now screens
the previous night. Ledger reset on 2026-09-10 (§8c) with the saturation and
real/bogus floors. On 2026-09-20 (§8d) the funnel gained the persistent-level
subtraction, the ledger's visit denominators were corrected in place, promoted
targets are vetted nightly before the alert fires, and the list is rebuilt
under the bright-neighbour rule. Per the charter the objective is a detection;
a clean null over the season is a reason to change the question, not to write
it up.
