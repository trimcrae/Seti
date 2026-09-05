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

## 7. Status

Built and offline-tested (22 tests), lint-clean. **Not yet run against either
live service** at the time of writing; the probe is the first thing the workflow
does and every field name above is to be read against `probe.json` before any
number from this channel is believed. Per the charter the objective is a
detection: if the northern nearby-star sample produces a clean null over a
season, that is a reason to change the question, not to write up the null.
