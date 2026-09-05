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
`watchlist.csv`, `events_latest.csv`, `rejected_latest.csv`.

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

## 9. Status

Live since 2026-09-05, twice daily (11:25 and 23:25 ET) until the backfill from
2026-07-14 reaches the stream, then screening the previous night each morning.
36 offline tests. Per the charter the objective is a detection; a clean null
over the season is a reason to change the question, not to write it up.
