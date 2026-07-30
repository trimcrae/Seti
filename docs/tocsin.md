# TOCSIN — the nightly alarm bell on the Rubin/LSST alert stream

*A tocsin is the bell rung to raise an alarm.* KNELL looks for clocks that
already stopped; TOCSIN watches the sky **tonight**.

---

## 1. The claim

Screen the world-public Rubin/LSST alert stream, every night, for
**difference-image events at the positions of catalogued nearby stars that are
achromatic, unresolved, non-moving, and — decisively — that recur across
nights**, in both flux polarities:

* **flash** (positive difference flux) — signature **S30** of
  `docs/necrosignatures.md` ("an unclassified blackbody transient on a
  catalogued nearby dwarf, matching neither flare, nova, nor microlensing"),
  together with its specular reading: a flat reflector returns the *stellar*
  spectrum, so an artificial glint is grey, whereas a stellar flare is blue.
* **dip** (negative difference flux) — brief **grey** occultations, the
  short-timescale end that the ZTF `dimming` channel could not reach (it
  exhausted at a 1.6–7.4 % systematics floor).

The deliverable is a **detection**, not a rate. Per the charter, a clean null
changes the question rather than becoming a paper.

## 2. Novelty position — stated honestly, and narrowed where it must be

A dedicated prior-art sweep was run on the runner (65 arXiv + 31 OpenAlex
queries, then full text of 37 papers and the citing-sets of 11 proposal papers;
raw evidence committed under `results/alertlit/`, `results/alertlit2/`,
`results/alertlit3/`, every arXiv ID verified against the arXiv API before
citation). What it establishes:

**Unoccupied — the axis this channel is built on.**
*Cross-night recurrence of achromatic alert-stream events at a fixed position on
a quiescent star.* No prior art in any survey. Searches for "repeating" +
"technosignature" return radio work only; the sole recurrence-at-a-position SETI
logic in the literature is Villarroel et al.'s *spatial alignment within a single
photographic plate* (arXiv:2110.15217, and the PASP 2025 multiple-transient
paper), not cross-night repetition in a modern stream.

**Proposed but never executed — what the flash mode finally does.**
Lacki 2019 (arXiv:1903.05839, *A Shiny New Method for SETI: Specular Reflections
from Interplanetary Artifacts*, PASP) computes the per-exposure reach of
Pan-STARRS1, **LSST** and Evryscope for specular glints. In seven years it has
twelve citing works and **not one executes a glint search**. Jaiswal 2023
(arXiv:2306.07859) and Kopparapu et al. (arXiv:2405.04560) likewise propose
without executing. Rogers, Lintott, Croft, Schwamb & Davenport
(arXiv:2401.08763) explicitly flag Lacki's glints as the opportune extension of
LSST anomaly detection — and do not do it.

**Partly occupied — the dip mode is an *extension*, not a new channel.**
Gallay, Davenport & Croft, *Technosignature Searches with Real-time Alert
Brokers* (arXiv:2506.14744, AJ 2025, 170, 95) **already screen negative-flux
alerts**: they use ZTF's `isdiffpos` flag, require every alert-packet magnitude
to be fainter than the reference magnitude, and funnel ~10⁶ alerts/night to ~20
to ~5 dipper candidates, framed as artificial-occultation follow-up. This
channel must not claim negative-flux screening as new. What their screen does
*not* contain — verified by grepping the full text — is any use of **colour**
(`achromatic`: 0 hits, `colour`: 0, `glint`: 0) or of **recurrence** (`recurr`:
0, `repeat`: 0). Their discriminant is single-band amplitude with K–S/χ² tests.
TOCSIN's dip mode is therefore positioned as *the grey and recurrent variant of
an existing search*, on a new instrument.

**Adjacent, and to be differentiated from, not claimed.**
Kovačević et al. (arXiv:2606.00574, IAU S404, May 2026) simulate achromatic
coherent variability against chromatic natural variability using cross-band
correlation indices in LSST *ugrizy* colour space. It is **simulation only** (500
synthetic light curves, no data, no alerts) and targets *stable periodic*
signals rather than transient events — but it means "achromaticity as an LSST
discriminant" is no longer a blank page, and any write-up must cite it.
Similarly, *Anomaly Hunter for Alerts* (arXiv:2602.12955, Iskandarli, Lintott,
Croft, Stevance & Weston, Feb 2026) applies unsupervised autoencoders to the ZTF
alert stream via Lasair and explicitly frames it as a technosignature
opportunity — so **generic ML anomaly detection on alerts is now occupied**, and
this channel deliberately does not do it.

**What the community has already claimed as planned.** The Rubin LSST TVS
Roadmap (arXiv:2208.04499, Hambleton et al., PASP 2023) names four
technosignature families: unnatural orbit alterations of solar-system objects;
unnatural flux patterns of normal variability; spatial correlations of events;
spatial over-densities. **Glint, achromaticity and negative-flux screening
appear in none of them.** Headline scarcity: `abs:"technosignature" AND
abs:"LSST"` returns five papers in all of arXiv; adding `abs:"broker"` returns
one.

**The scarce asset.** LSST began its survey on 30 June 2026; the first
world-public alerts flowed on 24 February 2026. As of 30 July 2026 **no
published SETI screen of real Rubin alerts exists.**

**The standing risk.** Davenport, Croft and Lintott author arXiv:2506.14744,
arXiv:2602.12955, arXiv:2401.08763 and arXiv:2508.16825, all on Lasair/ZTF.
This channel sits in their path. Its defensibility rests on the *specific
discriminant* — grey + recurrent + trial-corrected — and on being on Rubin
first, not on privileged access to data anyone can read.

## 3. The governing methodology

**The observable is coherence, not brightness.** A single achromatic flash is
indistinguishable from a cosmic ray, a satellite glint or an unflagged
subtraction residual. What none of those do is come back to the same catalogued
star. So the per-event funnel exists only to feed a clean ensemble to a
**persistent cross-night ledger**, and the promotion decision is made there.

That has three consequences the code enforces:

1. **Cumulative trials.** Significance is quoted against the running total of
   `target × night` screenings since the ledger opened, never against tonight's.
   A nightly screen that forgets its own history manufactures a 3σ event every
   few weeks by construction. Promotion uses Benjamini–Hochberg across all
   targets ever screened (FDR, not Bonferroni: at 10⁵ targets Bonferroni would
   reject a genuine repeater along with the noise).
2. **A real denominator — and it is not forced photometry.** Alerts exist only
   where there was a *detection*, so the stream cannot say how many times a star
   was looked at and showed nothing. Forced photometry (`prvDiaForcedSources`)
   is the textbook answer, and it is what this channel was built on — until
   measurement contradicted it. The first real backfill returned
   **forced-photometry coverage of 0%** of screened star-nights. With no
   non-detection information the denominator collapses onto the numerator, the
   ensemble rate pins at exactly 1.0, and `P(X ≥ k | p = 1)` is 1 for every k:
   **no target could ever be promoted**, and the channel would have stayed
   inert while looking healthy — committing tidy summaries and accumulating a
   ledger forever.

   The denominator now comes from the **observed footprint**. Detections trace
   where the camera pointed, so a 1° sky bin holding any detection on night *N*
   was observed on night *N*, and a catalogued star in that bin was screened on
   night *N* whether or not it alerted. That is a real rate over *all* targets,
   not just those the broker happened to give a `diaObject` — which is closer to
   the trial space this channel wanted in the first place. It is aggregated
   server-side with `GROUP BY`, so a night's footprint costs one small result
   rather than millions of rows.

   Deliberately conservative: a target counts only if its **own** bin was
   observed, with no neighbour dilation. Stars at field edges in empty bins are
   missed, which under-counts trials, over-estimates the rate, and enlarges
   every p-value. For a search, erring toward fewer detections is the right
   direction. Forced photometry is still used where present and the three
   sources (forced, footprint, detection) are unioned; the summary reports the
   coverage fraction of each, so a window whose denominator is weak says so.
3. **A cadence-matched null.** The LSST cadence has strong structure (in-night
   pairs ~33 min apart, ~3–4 day revisits, seasonal gaps, the lunar cycle), so
   *every* event spacing is commensurate with it. The timing test therefore
   draws its null by resampling the star's **own visited nights**: a "period"
   that is really the survey's revisit cadence scores identically under the null
   and cancels. This is the same discipline KNELL applies with
   injection-measured efficiency, and it is unit-tested from both sides
   (`test_timing_null_is_cadence_matched`,
   `test_evenly_spaced_events_beat_the_cadence_matched_null`).

**The event unit is the star-night**, not the star-band and not the alert. A grey
flash seen in the night's *g* and *r* visits is one event measured twice — the
second band is the colour measurement, not a second occurrence. Numerator and
denominator both count star-nights.

### 3.1 The achromaticity test, and why Rubin makes it possible

For an event of difference flux `dF` on a star of quiescent flux `F*`, the
**fractional amplitude** `a_b = dF_b / F*_b` has a band-to-band equality that
separates the hypotheses:

| hypothesis | expectation |
|---|---|
| specular reflection | `a` equal in all bands; difference-flux colour temperature = the **star's** temperature |
| stellar flare | `a` larger in the blue; colour temperature ~9000–10⁴ K whatever the host |
| grey occulter | `a` equal in all bands, negative |
| line-of-sight dust | `a` negative and *reddened* — never equal |

One statistic (`greyness_z`) therefore carries flare rejection for flashes and
dust rejection for dips. For a 9000 K flare continuum on a 3200 K dwarf the
predicted contrast is `a_g/a_r ≈ 3.7` from the blackbody ratio alone (real flares
are bluer still, from line emission the model omits) — a firm discriminant at
Rubin's photometric precision, though not the order-of-magnitude effect it is
sometimes loosely described as.

Two Rubin properties make this work where ZTF failed. First, **the LSST baseline
takes intra-night visit pairs in *different* filters** (u+g, u+r, g+r, r+i, i+z,
z+y; only y+y repeats a filter), ~33 minutes apart — so a colour is the *default*
data product for any event lasting more than half an hour. The ZTF glint channel
died precisely because it could seldom test achromaticity; when it could, it
killed all 15 candidates as chromatic M-dwarf flares. Second,
`diaSource.templateFlux` gives Rubin's own forced PSF flux on the coadd template
— the quiescent flux, in the same band, same system, same pixels — so `dF/F*` is
a ratio of two Rubin measurements with **no cross-survey passband transformation
error at all**. Gaia DR3 synthetic photometry (GSPC, SDSS *ugriz* + PS1 *y*) is
the documented fallback and carries an explicit passband-mismatch systematic;
the channel never claims greyness tighter than it.

### 3.2 Single-band events, and the one-sided test

The two-band test turned out to fire almost never. In the first correct live
window **every event was single-band** (22 of 22), so the achromaticity
discriminant — the channel's headline argument — was dormant. Measured, not
assumed: it is why `bands_per_event` is now a first-class output.

A single-band detection is not colour-blind, though. It is **one-sided**. A grey
event has equal *fractional* amplitude in every band, so on a red star it is
brighter in *absolute* flux in the redder band. On an M dwarf with *g* = 18,
*r* = 17, a grey 10% event puts 22,909 nJy into *g* but **57,544 nJy** into *r*.
So if the event is seen in *g* while *r* was observed the same night and stayed
silent, the grey hypothesis predicted a redder-band signal that never appeared —
evidence against greyness, and exactly what a flare looks like.

Everything the test needs is measured rather than assumed: the footprint query
groups by **band** as well as bin and night (a band's silence is only evidence
if that band was observed), and each band's effective detection limit comes from
that window's own detections — the median flux error times the stream's ~5σ
threshold. The grey prediction must clear the limit by `nondetection_margin`
(3×) before silence counts; below that the event is recorded as *attempted but
untestable*, never as passed.

**It works on real data.** The verification window rejected **five** events as
chromatic by redder-band non-detection — a rejection that had fired zero times
in every earlier run. This is the discriminant the ZTF glint channel died on
(15 candidates, all chromatic flares), now operating on the Rubin stream.

Events with no other band observed remain untestable, and their only route to
promotion is recurrence — which is the design working as intended.

## 4. Contamination ledger

Inherited discipline from `docs/channel-brief.md` §4, plus what is specific here.

| confounder | why it matters | test |
|---|---|---|
| **Stellar flares** | The dominant astrophysical event on nearby M dwarfs by orders of magnitude | Achromaticity: flares are blue. Rejected as `chromatic` |
| **Satellite glints** | *An exact mimic*: a brief achromatic specular reflection. ~73,000 glint events pollute 3.6 % of ZTF science images (arXiv:2202.05719, arXiv:2310.17322); see also arXiv:2011.02495, arXiv:2011.03497, arXiv:2411.03258, arXiv:2403.04942 | Rubin's own `glint_trail` flag and `pixelFlags_streak` are fatal. Untrailed point glints survive that — and are killed by **recurrence**: a satellite does not return to the same catalogued star on many nights. This is the single strongest argument for the ledger |
| **Proper-motion subtraction dipoles** | *The* systematic for a nearby-star sample: high-PM stars are exactly the ones whose template position is wrong, so they subtract badly and alert spuriously | `isDipole` is fatal; unflagged dipoles are caught by the **duty-cycle** test — a subtraction failure repeats at *every* visit, an event does not. Mixed polarity in one night is rejected outright |
| **Solar-system objects** | ~400 alerts/visit, up to ~5,000 near the ecliptic | `sid = 1` excludes SSO-associated alerts server-side; `trailLength` and `extendedness` catch the unassociated residue (Rubin already deletes trails >10 °/day upstream) |
| **Cosmic rays** | Single-visit, single-band, unrepeatable | `pixelFlags_cr*` are fatal; recurrence finishes the job |
| **Un-propagated proper motion** | Produces a **clean null**, the most dangerous failure mode — it looks like a result | Positions are propagated to the alert epoch before matching, and the failure is a regression test (`test_match_fails_without_proper_motion_propagation`) |
| **Deep-drilling fields** | *Found the hard way.* A DDF is deeper, revisited far more often (34–48 nights against 7 elsewhere) and subtracts differently, so its true per-star-night alert rate is genuinely higher — measured at 0.006–0.027 against an all-sky 1.57×10⁻³. Testing a DDF star against the all-sky rate does not detect anything, it rediscovers the observing strategy | The binomial null is **stratified by 1° sky bin**: each target is tested against the more conservative of the all-sky and its own bin's rate |
| **Low-amplitude variable stars** | *Also found the hard way.* The dominant astrophysical population at \|a\| ≲ 3%, and the first three candidates were all of them | **Single-mechanism coherence**: a reflector flashes, a grey occulter dips. A star doing both across nights is varying intrinsically, and caps at `interest` |
| **Misassociation by the broker's cross-match** | ALeRCE matches to Gaia at the *catalogue* epoch, so the highest-PM nearby dwarfs — this channel's best targets — are the ones it can orphan | The server-side Gaia join is a *cheap* cut, never the authoritative association; the repository's own PM-propagated match decides. Periodic audit runs with the join disabled measure what it loses |

### 4.1 The completeness limitation that is not ours to fix

Rubin's production pipeline applies `minReliability: 0.5` **before an alert is
issued**. DMTN-337 measures that model's true-positive rate on **variable stars**
at **3.5 %** (v0.1, on DP1) — it scores stellar point-source-on-point-source
subtractions characteristically low, and v0.3 still does. This channel's signal
*is* a stellar point-source event, so the alert stream is systematically biased
against it, by an amount set by someone else's classifier and not recoverable
downstream (sub-threshold sources are never alerted at all).

Two consequences, both binding: the channel applies **no additional reliability
cut** (`min_reliability: 0.0` in `config/tocsin.yaml`) so as not to compound a
loss it did not choose; and **any result must quote this incompleteness**. It
also means a null here is weak evidence about the sky and strong evidence only
about the alert stream — one more reason the deliverable is a detection.

## 5. Data path

| stage | source | why |
|---|---|---|
| target list | Gaia DR3 via `astroquery.gaia`, in equal-volume parallax shells | *d* < 100 pc keeps trials at ~10⁵ (so a per-target p-value is interpretable) and Gaia astrometry excellent |
| nightly detections | **ALeRCE TAP** (`https://tap.alerce.online/tap`), public ADQL, **no credentials** | The only broker path that supports an unattended cron: one ADQL statement answers "every detection from night X", indexed on `mjd`/`ra`/`dec` |
| denominator | ALeRCE `forced_photometry`, same Gaia pre-cut | Bulk, so the visit history costs one query rather than one call per object |
| deep vetting (optional) | Lasair-LSST `/api/object/?lite=False` | Full per-epoch `diaSources`; needs a free token, 100 calls/hour at the registered tier |
| enrichment (optional) | Fink `/api/v1/objects` | Uniquely rich cross-match: SIMBAD, **VSX**, **GCVS**, Gaia DR3 variability flags — the cheapest way to ask "is this star already a catalogued variable?" |

Rubin *alerts* are world-public; Rubin *data releases* (coadd catalogues, images)
are data-rights restricted. That is why baseline photometry comes from
`templateFlux` inside the alert or from Gaia, never from a Rubin catalogue query.

### 5.1 What the live probe measured (2026-07-30, `results/tocsin/probe.json`)

Every ADQL column name in this channel was inferred from the brokers' published
source, because the sandbox it was written in has no egress. The probe exists to
replace inference with measurement, and it changed three things:

* **The broker's mirror is not live.** Newest LSST epoch MJD 61235.4 against a
  wall clock of 61251.1 — a **15.6-day lag**. A screen asking for "the last two
  nights" would therefore return nothing *every night, forever*, and an empty
  result is indistinguishable from a real null. The window is now anchored to
  the newest epoch the broker actually holds, and a **watermark** in the ledger
  advances through the data, which makes coverage gapless and non-overlapping
  whatever the lag does next.
* **There is a 262-night backlog.** LSST detections span MJD 60973 → 61235
  already. The first runs are a *backfill of real archival data*, not a wait for
  new sky — so the recurrence statistics that need many nights become available
  in days. `max_nights_per_run` caps one job's bite.
* **The encodings, measured rather than assumed.** `sid=0/tid=0` is ZTF;
  `sid=1/tid=1` is LSST diaObject (5.16M); `sid=2/tid=1` is LSST ssObject
  (131k). So `sid=1` is right and drops ~300k solar-system detections per 30
  days. `catid=1` is Gaia DR3, `catid=0` AllWISE. The Gaia join returns real
  nearby stars (parallax ~12 mas at ~1.4″).

Two schema traps the probe caught, both of which would have produced confident
nulls rather than errors: `gaiadr3_source` has **no `source_id`** (the join key
is `oid_catalog` on both sides), and `oid_catalog` **cannot be SELECTed** at all
— the service declares it integer while AllWISE ids are strings, so it fails
VOTable serialisation. It is fine in a JOIN condition, which is the only place
this channel needs it.

One observation to confirm rather than rely on: an unfiltered sample of
`lsst_detection` contained `reliability` values of 0.10 and 0.33, i.e. **below**
the 0.5 cut Rubin's production pipeline applies before issuing alerts. If that
holds for current data it materially softens the completeness limitation in
§4.1. But the sample was not time-filtered and the mirror reaches back into the
commissioning era, so this is not yet established and nothing here depends on it.

Fluxes are **nanojansky and signed**; times are **MJD TAI**; `mag = 31.4 −
2.5 log₁₀(F/nJy)`. ALeRCE encodes the band as an integer with **u = 6, not 0**,
and lower-cases every ADQL column name — both are unit-tested, because either
would silently corrupt every colour in the channel.

## 6. Promotion tiers

| tier | meaning |
|---|---|
| `watch` | one event; not yet testable |
| `interest` | a grey-confirmed single event, or ≥2 events, or a repeater rejected by one of the coherence rules below |
| `candidate` | ≥2 events, ≥1 grey-confirmed, FDR-significant against the **stratified** null, exact visit denominator, duty cycle below threshold, and **single polarity** |
| `alarm` | a candidate whose event epochs also beat the cadence-matched timing null |

A tier is a statement about *evidence*, not a ranking of excitement.

## 6.1 First full walk: 263 nights, and why zero candidates is the right answer

The complete backlog (MJD 60973 → 61235, the broker's whole LSST holding) gives:

```
263 nights · 55,424 star-night trials · 87 events · 42 targets with events
all-sky rate 1.57e-03 per star-night · 1,927 sky bins with trials
tiers: 25 watch · 13 interest · 0 candidate · 4 none
```

Three targets *did* reach candidate tier before the last two discriminators were
added, and both discriminators came from examining them rather than from theory:

1. The first two were both in **COSMOS** — a deep-drilling field where the local
   alert rate is 4–75× the all-sky value. Stratifying the null moved their
   p-values from 1.3×10⁻³ and 2.7×10⁻⁷ to 0.035 and 0.003.
2. All three showed **both polarities across nights** — flash on some, dip on
   others. That is intrinsic variability, not a reflector and not an occulter.

The four `none` targets are duty-cycle rejections: stars alerting on most of
their visits, which is a subtraction residual rather than an event. The
highest-multiplicity target in the whole walk (7 events in 7 visits, duty 1.0)
is one of them.

So the honest summary of the first walk is: **the funnel works, every
discriminator fires on real data, and nothing survives.** Per the charter that is
a reason to keep accumulating and to keep sharpening the question — not a result
to write up.

## 7. Status

Built and offline-tested (80 tests). The probe has run against the live service
and its record is committed at `results/tocsin/probe.json`; §5.1 lists what it
measured and the three bugs it caught. The schema dump is committed verbatim so
a later broker change appears as a diff in version control rather than as an
unexplained null.

The next thing that matters is **accumulation**. The ledger is worthless on
night one and gains power monotonically: a single grey flash can never exceed
`interest`, because only repetition at a fixed position separates the signal
from a cosmic ray, a satellite glint, or an unflagged subtraction residual. With
262 nights of backlog to walk through, that is a matter of days rather than
months.

Two known gaps, neither blocking:

* the Fink enrichment layer (SIMBAD/VSX/GCVS "is this already a catalogued
  variable?") is designed and documented but not yet wired in — it is a
  shortlist-time query, so it costs nothing until there is a shortlist;
* the Lasair deep-vetting path is implemented but untested against the live
  service, because it needs a token nobody has registered for yet. Nothing in
  the channel depends on it.
