# TOCSIN on alternative feeds — ASAS-SN Sky Patrol v2 and ATLAS forced photometry

*Written 2026-08-25, while Rubin is snowed in. Code: `src/seti/tocsin/altfeeds.py`.
Tests: `tests/test_tocsin_altfeeds.py`. Workflow: `.github/workflows/tocsin-altfeeds.yml`.*

---

## 0. The situation, and what is not an option

Rubin has been off sky since the night of 13/14 July 2026 (`docs/rubin-outage.md`,
verdict `SKY_STOPPED`). Both alert brokers stop on that night, count for count,
and the observatory's own posts date the shutdown to the day. TOCSIN reports
`NO_NEW_DATA` — the correct verdict for a screen with nothing left to screen.

The cheapest substitute is **dead by measurement**, not by assumption
(`docs/substitute-surveys.md`): ALeRCE's non-LSST feed (`sid=0`, 9,093,519
objects) has a newest epoch of **2026-04-30**, two and a half months *before* the
LSST feed stopped, and Fink's ZTF portal did not answer at all. Pointing TOCSIN
at ZTF through the broker it already uses returns nothing, and an empty result
is indistinguishable from a real null. That route is closed.

What is reachable, measured on the runner the same morning:

| feed | HTTP | auth | shape |
|---|---|---|---|
| **ASAS-SN Sky Patrol v2** | 200 | **none** | full-sky light curves, ~1 d, g ≲ 18 |
| **ATLAS forced photometry** | 200 | free account + token | full sky incl. south, ~1 d, 30 s exposures in quads |
| ZTF forced photometry (IPAC) | 401 | account | northern only |

---

## 1. The structural change: an alert stream is not a light curve

`brokers.py` consumes an **alert stream**. A broker decided, upstream, that a
difference-image detection was worth issuing, and the hard problem is that
non-detections are invisible — which is why TOCSIN's denominator had to be
rebuilt from the *observed footprint* after forced-photometry coverage measured
0 % (`docs/tocsin.md` §3).

Both feeds here are the opposite shape: a **per-target light curve**, delivered
whether or not anything happened.

* **ASAS-SN Sky Patrol v2** returns every epoch on which the field was observed,
  including the epochs the star did nothing. Its pipeline uses image
  subtraction and adds the reference flux back, so the served value is *total*
  flux and the quiescent level is the sigma-clipped median of the light curve
  itself.
* **ATLAS forced photometry** is literally forced photometry: PSF photometry on
  the *difference* images at coordinates you supply, on every exposure that
  covered them. The served `uJy` column is signed difference flux — the same
  observable as a Rubin alert's `psfFlux`.

So the two halves of the ledger swap difficulty. The **numerator must be
constructed** here (an epoch is an event only if it deviates from the star's own
baseline, at ≥ 6σ *of the light curve's own measured scatter*, not of the
survey's optimistic formal errors). The **denominator comes for free and is
exact**: every good epoch is a trial, per target, with no footprint proxy, no
union with the numerator, and no `visits_exact = False` cap on promotion.

That inversion is the single most important property of this adapter, and it is
the one respect in which an alternative feed is strictly **better** than the
Rubin path it stands in for.

---

## 2. Cadence honesty: what survives the switch and what does not

This section exists because *a screen that reports a null on a signature it
cannot detect is the failure mode this repository is most exposed to*. The same
statements are carried in `altfeeds.signature_transfer()` and written into every
committed summary, so a reader of an artefact never has to come here to find the
caveats.

### 2.1 Preserved, intact

| capability | why it survives |
|---|---|
| **The cross-night recurrence ledger** | This channel's actual instrument. Recurrence at a fixed catalogued position is a property of accumulated state, not of any survey's optics. Both feeds carry many years of epochs, so the ledger has power on day one instead of accumulating for months. |
| **The denominator — improved** | Forced photometry answers "how many times was this star looked at, and showed nothing" directly. |
| **The duty-cycle test** | Needs only visit epochs, which are now exact per visit rather than per night. |
| **The cadence-matched timing null** | Same: it resamples the star's own visited nights. |
| **Both polarities** | A dip and a flash are symmetric in a light curve exactly as they are in a difference image. |

### 2.2 The timescale — and where the received wisdom is wrong

Cadence (~1 day) sets *how many chances you get*. **Exposure** sets *how much a
short event is diluted*: a glint of duration τ ≪ t_exp is recorded at amplitude
τ/t_exp of its instantaneous value. These are different quantities and the
distinction decides whether the flash mode transfers at all.

* A **Rubin visit is 2 × 15 s = 30 s**. An **ATLAS exposure is 30 s**. The
  dilution of a sub-second specular glint is therefore **identical** in the two.
  ATLAS additionally takes a **quad** — four exposures spanning ~1 hour — which
  is the same intra-night structure Rubin's 33-minute visit pair provides, minus
  the filter change. *For ATLAS the flash mode's timescale transfers exactly;
  only its depth does not.*
* **ASAS-SN stacks 3 × 90 s** dithered images per epoch, so a short event is
  diluted by roughly an order of magnitude more than in a Rubin visit, on top of
  being shallower. For ASAS-SN the fast end genuinely is inaccessible, and what
  remains is the slow end: grey dips and slow brightenings lasting hours to days.

Because ATLAS's quad resolves an hour, an event is **classified, never cut**:

* `single_exposure_fast_or_cosmic_ray` — present in one of the night's exposures.
  Consistent with a fast glint, and equally consistent with a cosmic ray.
* `multi_exposure_night_slow_event` — present in two or more. Slow and robust,
  but no longer the sub-visit signature.

Cutting on this would quietly redefine the signal to fit the data, which is the
move this module refuses to make. Recording it lets the ledger's recurrence test
do the discriminating it was built for.

### 2.3 Lost, and not recoverable from these feeds

| capability | status |
|---|---|
| **Depth** | Rubin r ≈ 24.5 in a 30 s visit; ATLAS o ≈ 19.0, ASAS-SN g ≈ 18.0. A penalty of ~5.0–5.5 mag, i.e. a factor 100–160 in flux. Quantified against the real target list in §3. |
| **Achromaticity, in ASAS-SN** | **Absent, not degraded.** Sky Patrol v2 post-2018 is a single band (Sloan g′). The discriminant that killed all 15 candidates of the ZTF glint search *cannot run*. Every ASAS-SN event carries `greyness_unavailable_single_band_survey`. |
| **Achromaticity, in ATLAS** | Two filters (`c`, `o`) but scheduled by lunation — `c` near new moon, `o` otherwise — so same-night two-band coverage is rare. Rare is not never, and the fraction is **measured per run** as `two_band_night_fraction` rather than assumed. |
| **Independent astrometry** | Forced photometry is measured *at the position you asked for*, so the separation between "the event" and "the star" is zero by construction and `astrometric_offset` can never fire. Every event carries `astrometry_not_independent`. This removes the Rubin path's defence against an unrelated blended source — see §5. |
| **Rubin's flag suite** | `reliability`, `isDipole`, `glint_trail`, `extendedness`, `pixelFlags_*`, `ssObjectId` do not exist. Left as `None` they would make five rejection rules *silently inert* — the rules would run, find nothing to test, and pass everything. Where a per-epoch substitute exists it is constructed (ATLAS `chi/N` → `pixel_flag_bad`; ATLAS `maj`/`min` elongation → `raw["trail_flag"]`; ASAS-SN `quality` and seeing outliers → bad-epoch rejection). Where none exists, the absence is **named in every event's `reasons`**. |

---

## 3. Depth versus sample — the number that decides whether this is worth running

### 3.1 The headline depth is not the usable depth

"g ≲ 18" is the magnitude at which the **star** is detected at 5σ. Detecting a
*fractional* event of amplitude `a` on that star needs `a·F*` to clear `n` σ of
the per-epoch noise. In the background-limited regime the noise is roughly
independent of the star, so

```
a · F*(m)  ≥  (n/5) · F(m_5σ)
⇒   m  ≤  m_5σ − 2.5 log₁₀( n / (5a) )
```

For a **10 % event at 6σ** that is **2.70 mag brighter** than the nominal depth:

| feed | band | 5σ depth | saturates | usable window, a = 10 % | a = 30 % | a = 100 % |
|---|---|---|---|---|---|---|
| ASAS-SN | g | 18.0 | 10.5 | 10.5 – **15.30** | 10.5 – 16.49 | 10.5 – 17.80 |
| ATLAS | o | 19.0 | 12.5 | 12.5 – **16.30** | 12.5 – 17.49 | 12.5 – 18.80 |
| ATLAS | c | 19.5 | 12.5 | 12.5 – **16.80** | 12.5 – 17.99 | 12.5 – 19.30 |

Quoting the raw depth instead would over-state the usable sample by a large
factor. Note also that reach is a **window**: both feeds saturate, and a 100 pc
nearby-star catalogue is richest at the bright end, so counting saturated stars
as reachable would overstate the sample precisely where the error is largest.

The bright end is background-limited only approximately — near saturation the
noise becomes photon-dominated and scales as √F*, which makes the true faint cut
slightly *fainter* than the table says. The background-limited form is used
because a search should err toward claiming less reach, not more.

### 3.2 What the target list actually contains

The authoritative number comes from the offline **census** stage, which needs no
network at all:

```
python -m seti.cli tocsin-targets        # runner-only; builds .cache/tocsin/targets.parquet
python - <<'PY'
from seti.tocsin.altfeeds import census
census()                                  # writes results/tocsin_altfeeds/census.json
PY
```

`census()` reads the cached Gaia list, computes each target's magnitude in each
feed's **native** band (`g_sdss_mag` directly for ASAS-SN; an interpolation
between the bracketing GSPC SDSS bands for ATLAS `c` and `o`), and reports the
reachable count per amplitude, the saturated count, the un-assessable count, and
the proper-motion exclusions. It is stage 1 of the workflow and it runs before
anything is fetched, because if the reachable fraction at a plausible amplitude
is negligible then the right response is to **change the question**, not to run
and file a null.

**An indicative measurement, made offline from the committed Rubin ledger.** The
ledger stores, per event, both `dflux_njy` and `a = dF/F*` with
`baseline_source = rubin_template`, so the quiescent flux is recoverable as
`F* = dF/a`. Over the 42 TOCSIN targets that produced events in the full 263-night
walk:

| band | n | min | median | max |
|---|---|---|---|---|
| g | 21 | 16.53 | **19.08** | 20.19 |
| r | 21 | 16.90 | **17.83** | 18.73 |
| i | 10 | 15.75 | **16.18** | 17.66 |

Interpolating to the ATLAS bands gives median o ≈ 17.0 and c ≈ 18.6. Against the
table in §3.1: **none of these 42 targets is reachable at a = 10 % in either
feed**, and only the a ≈ 30 % end of ATLAS `o` touches them at all.

That number must be read with its bias stated, and the bias is large and runs in
a specific direction: **there is no target brighter than g = 16.5 in the sample,
and that is not a property of nearby stars — it is Rubin's saturation limit.**
A 30 s LSST visit saturates near r ≈ 16, so the alert stream structurally cannot
screen the bright half of a 100 pc catalogue. The 42-target sample is therefore
biased *faint* by construction, and the census over the full list will be more
favourable. How much more is exactly what the census measures.

### 3.3 The complementary population — why this is more than a stopgap

The same fact turned round is the strongest argument for running these feeds at
all, and it is not a substitution argument:

> **ASAS-SN and ATLAS work precisely where Rubin cannot.**
> Rubin saturates near r ≈ 16. ASAS-SN saturates near g ≈ 10.5 and ATLAS near
> 12.5. The window 10.5 ≲ m ≲ 16 is a population of catalogued nearby stars that
> the Rubin alert stream **cannot screen at any amplitude**, and both feeds
> screen it at ~1 day cadence over a decade of archive.

`reachable_fraction()` counts it directly (`n_reachable_and_rubin_saturated`),
and the census reports it per band and per amplitude. For a channel whose charter
puts novelty first, a search over stars an instrument is structurally blind to is
a better position than a shallower repeat of the same one.

---

## 4. The band labels, and the three places a label could silently lie

`schema.validate` accepts only the LSST band letters `ugrizy`, so a native band
name (`c`, `o`, `V`) is rejected as malformed and never reaches the funnel.
Rather than edit the shared schema, each native band is carried under its nearest
LSST **label**, with the native name preserved in `raw["native_band"]`:

| native | label | justification |
|---|---|---|
| ASAS-SN `g` | `g` | Sloan g′. The channel already treats SDSS g as the stand-in for LSST g (`targets.GSPC_MAG_COLUMN`), so this is the approximation already in use. |
| ATLAS `c` | `g` | 420–650 nm, λ_eff ≈ 533 nm. The bluer ATLAS band; the label is used only for ordering and pairing. |
| ATLAS `o` | `r` | 560–820 nm, λ_eff ≈ 679 nm. The redder band. |
| ASAS-SN `V` | — | Johnson V sits between g and r with no defensible proxy. Pre-2018 V epochs are dropped from the **numerator and the denominator together**. Dropping them from the numerator alone would inflate trials, deflate the ensemble rate and make every per-target p-value too small — the anti-conservative error that manufactures significance. |

**A label is not a passband.** There are exactly three places where the label,
taken literally, produces a wrong number. Each is closed:

1. **The baseline flux.** `screen._baseline_flux` falls back to the Gaia GSPC
   synthetic magnitude *of the label's band* when the alert carries no template
   flux. For an ATLAS `c` detection that would divide by an SDSS `g` flux, and
   on a red dwarf — this channel's whole sample — `c` collects far more flux than
   `g`, so every fractional amplitude would be inflated.
   **Closed by construction:** the adapter *always* sets `template_flux_njy` in
   the native band, and a band for which no native F* can be established emits
   no alerts at all (`no_quiescent_flux`). `_baseline_flux` prefers the template
   flux whenever it is present and positive, so the GSPC branch is never reached.
   F* provenance, best first: the feed's own total-flux light curve
   (`<survey>_lightcurve_median`); a second ATLAS pass with `use_reduced=True`
   (`atlas_reduced_images`); GSPC interpolated to the native effective wavelength
   (`gspc_interpolated_native`), carrying a 20 % systematic — an interpolation,
   not a published transformation, and labelled as one everywhere.
2. **The difference-flux colour temperature.**
   `photometry.blackbody_colour_temperature` looks the wavelength up by the LSST
   label, so an ATLAS c/o pair would be fitted at 0.483/0.622 µm when it was
   taken at 0.533/0.679 µm. The bias lands on the one number the S30 claim rests
   on — "the transient's temperature is the star's". **Closed by**
   `native_colour_temperature()`, which refits at the native wavelengths and
   overwrites the value on every event. The test suite pins that the two answers
   differ by >10 %, which is the whole reason the refit exists.
3. **The one-sided non-detection test.** See §7 — it is disabled in the funnel
   and re-run correctly here.

---

## 5. Contamination ledger, specific to these feeds

Inherited discipline from `docs/tocsin.md` §4, plus what is new when the data is
a light curve rather than an alert.

| confounder | why it matters here | test |
|---|---|---|
| **Aperture blending** | *The dominant new systematic.* ASAS-SN's PSF is 16″ FWHM (8″ pixels), so the aperture routinely contains other stars, and with no centroid there is nothing to reject a flare on one of them. This is the defence the Rubin path gets free from `astrometric_offset`. | `blend_neighbours()` counts catalogue neighbours inside the aperture by flux ratio. It must be run against a catalogue **complete to the survey's depth** (a Gaia cone search on the runner); against the nearby-star list alone it measures only a lower bound, and says so. |
| **Proper-motion aperture drift** | See §6. Manufactures dips on exactly the best targets. | Segmentation for ATLAS, exclusion for ASAS-SN. |
| **Survey systematics read as events** | Formal error bars omit flat-fielding, blending and subtraction systematics, so believing them calls systematics events. | The per-epoch uncertainty is `max(formal_err, measured scatter)`, where the scatter is 1.4826 × MAD of the star's own light curve. Every threshold is therefore relative to what *that star* actually does. |
| **Low-amplitude variables** | Still the dominant astrophysical population, and now with no colour to separate them (ASAS-SN) — this is where the loss of achromaticity hurts most. | The ledger's duty-cycle test is the designed defence; `frac_scatter` is reported per band as an early warning. |
| **Cosmic rays / artefacts** | No `pixelFlags_*` exists. | ATLAS `chi/N` outliers against the *star's own median* (self-calibrating, since χ²/N depends on magnitude, seeing and field) → `pixel_flag_bad`. ASAS-SN `quality` flag plus seeing outliers → epoch dropped. |
| **Trailed sources / satellites** | No `glint_trail` exists. | ATLAS `maj`/`min` elongation outliers → `raw["trail_flag"]`, which `screen._per_alert_flags` already treats as fatal. ASAS-SN has no substitute; recurrence is the only defence. |
| **Solar-system objects crossing the aperture** | *Worse here than on the Rubin path.* An 8″-pixel aperture on the ecliptic is crossed by asteroids often, and there is no `ssObjectId` to exclude them. | **Not yet implemented.** The correct check is JPL's SB-Identification API (`ssd-api.jpl.nasa.gov/sb_ident.api`, measured reachable) or the MPC, queried per *event* rather than per epoch — a few hundred calls, not millions. Until it exists, every event carries `ss_association_unavailable_in_<survey>` and no event may be promoted on the strength of a single epoch near the ecliptic. |
| **Multi-site "nights"** | Both surveys are global networks (ASAS-SN: Hawaii, Chile, Texas, South Africa, China; ATLAS: Hawaii ×2, Chile, South Africa), so the Cerro Pachón night boundary in `schema.night_id` can merge two genuinely independent visits 12 h apart into one star-night label. | Kept deliberately, because changing the unit would make the ledger's numerator and denominator incomparable with everything else in the channel. The effect is conservative in both directions — two detections merge into one event, and two trials merge into one trial — and `star_nights_observed` versus `epochs_total` measures it. |

---

## 6. Proper motion, which is not a detail here

TOCSIN's targets are nearby stars, which are the high-proper-motion stars. A star
moving 1″/yr moves **10″ in a decade** — larger than ATLAS's PSF and comparable
with ASAS-SN's 16″ FWHM. Forced photometry at a *single* fixed coordinate across a
multi-year baseline therefore walks off the star, and the symptom is not an error:
it is a slow decline that a robust baseline partially absorbs and that leaves the
ends of the light curve looking like **dips**. That is a manufactured signal, in
the polarity this channel searches, on precisely its best targets.

* **ATLAS** takes arbitrary coordinates, so requests are **segmented in time**
  (`pm_segments`), each segment short enough that the drift stays under
  `max_drift_frac × FWHM` (default 0.25), and each submitted at the position
  propagated to that segment's mid-epoch.
* **ASAS-SN** photometers *its own* catalogued position (Sky Patrol v2 light
  curves are keyed to a source list built from ATLAS RefCat2, ~2015), so the
  drift cannot be controlled from outside. Targets whose drift over the requested
  window exceeds the same allowance are **excluded from the numerator and the
  denominator together**, and counted. Screening them badly while keeping them in
  the trial count would be worse than not screening them.

---

## 7. A defect in an existing file, described and not fixed

`screen._baseline_flux(alert, row, band, rel_err)` returns
`alert.template_flux_njy` whenever it is present and positive — **without
consulting the `band` argument at all**. That is correct for its main caller,
which asks for the detected band, and wrong for the one-sided non-detection test
in `screen.screen_alerts`:

```python
base_o, _e, _src = _baseline_flux(r0["alert"], row, other, th.baseline_rel_err)
```

Here `other` is the *silent* band, but the function hands back the *detected*
band's template flux. The predicted grey signal is then `|a| × F*(detected)`
instead of `|a| × F*(other)`, and on a red star those differ by a factor of a
few — so the test is under-powered when the detection is in the bluer band and
over-powered when it is in the redder one. **This affects the Rubin path too**;
`docs/tocsin.md` §3.2 records five real events rejected by this test.

This module does not touch it. Instead it passes `observed_bands=None`, which
disables that path entirely, and runs `native_nondetection_test()` with the
correct other-band baseline — pinned by
`test_native_nondetection_test_uses_the_other_bands_own_baseline`, which shows
the two inputs give opposite verdicts on the same event.

The minimal fix, for whoever owns `screen.py`: have `_baseline_flux` consult
`band` before returning the template flux, e.g. return the template flux only
when `band == alert.band`, and fall through to GSPC otherwise.

---

## 8. Auth posture

* **ASAS-SN Sky Patrol v2 — no token.** Measured on the runner 2026-08-25: HTTP
  200, no credentials. This is the feed that can run unattended today, and it is
  why the workflow defaults to it.
* **ATLAS — `ATLAS_TOKEN`.** Read from the environment exactly as the existing
  workflows read `LASAIR_TOKEN`; `ATLAS_USERNAME` + `ATLAS_PASSWORD` will mint one
  instead. Absent, the stage prints a named skip and exits 0 — *a missing
  credential, not a null result* — and nothing in the run fails.

  **Action required outside any session:** register free at
  <https://fallingstar-data.com/forcedphot/> and add the token as the repository
  secret `ATLAS_TOKEN`. Until then the closest like-for-like to a Rubin visit is
  unavailable.

---

## 9. What must be verified on the runner before any claim

Every column name in `altfeeds.py` was read from documentation. On the Rubin
path the same posture was wrong in three places (`docs/tocsin.md` §5.1) — a join
key that did not exist, a column that could not be SELECTed, and a filter
encoding that would have returned a clean, plausible, empty night. None of those
would have raised; all three would have produced confident nulls.

So `probe()` runs first and always, records each service's live response
**verbatim** into `results/tocsin_altfeeds/probe.json`, and that file is
committed so a later change appears as a diff in version control rather than as
an unexplained quiet result months later. Specifically unverified until it runs:

* ASAS-SN light-curve column names and their spellings (`jd` vs `hjd` vs `mjd`;
  `flux_err` vs `fluxerr`), the flux unit (assumed mJy), and whether the
  service's own client or raw HTTP is the working path;
* ATLAS's queue endpoints, the result-file header, and the flux unit (assumed
  µJy).

Both units are additionally **checked at run time** against the survey's own
magnitude column (`implied_flux_unit_njy`) and a disagreement is recorded as a
note, because a uniform flux-scale error cancels exactly in every ratio the
channel computes — `dF/F*` is scale-free — and would therefore never announce
itself, while corrupting every absolute limit and the non-detection test.

---

## 10. Running it

```
# stage 0+1 only: probe the live services and compute the go/no-go census
gh workflow run tocsin-altfeeds.yml -f probe_only=true

# stage 2: ASAS-SN, no credentials needed
gh workflow run tocsin-altfeeds.yml -f survey=asassn -f max_targets=200

# stage 3: ATLAS, once ATLAS_TOKEN exists
gh workflow run tocsin-altfeeds.yml -f survey=atlas -f max_targets=100
```

Artefacts, all committed:

```
results/tocsin_altfeeds/probe.json            live response shapes, verbatim
results/tocsin_altfeeds/census.json           reachable fraction per feed / band / amplitude
results/tocsin_altfeeds/<survey>/summary.json verdict, counts, signature_transfer
results/tocsin_altfeeds/<survey>/events.json  events + rejections + per-band reductions
results/tocsin_altfeeds/<survey>/ledger_<survey>.json
```

**One ledger per survey, never the Rubin one.** The ledger's statistic is a rate
per target-visit with cumulative trials; pouring ASAS-SN star-nights into a
denominator built from Rubin star-nights would produce a rate of nothing —
different depth, different cadence, different systematics, different minimum
detectable amplitude. The surveys are combined, if ever, at the level of *which
targets recur in more than one of them*, which is a much stronger statement and
needs no shared denominator.

---

## 11. Status

Built and offline-tested (65 tests, `tests/test_tocsin_altfeeds.py`), `ruff`
clean, workflow YAML parses. **Not yet run against either live service** — the
sandbox has no egress, and the probe is the first thing the workflow does.

Two things are known to be missing and neither is blocking:

* the solar-system cross-check (§5) — a per-event JPL SB-Identification query,
  worth building before any candidate is taken seriously, especially near the
  ecliptic;
* the Gaia cone-search that makes `blend_neighbours()` authoritative rather than
  a lower bound.

Per the charter, the point of all this is a **detection**. If the census says the
reachable window is empty, that is a reason to change the question — not to write
up how empty it was.
