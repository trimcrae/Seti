# METRONOME — clocks in stellar flare timing

**Signature S28** (`docs/necrosignatures.md` §2.VI): *structure in timing
series.*  The one informational-residue signature in the taxonomy that had
never been built.

---

## 0. What has actually been measured

**Run 35652897914, 2026-09-21** (the channel's first run that scanned
anything; the 2026-09-06 run scanned zero stars and is not a result).

| | |
|---|---|
| Catalogued flare times read | **1,523,888** |
| Stars in the catalogues | 69,928; with N ≥ 8 events: 27,392 |
| Stars scanned (after declustering and cross-star removal) | **3,131** |
| Events removed as cross-star spacecraft epochs | 7,016 |
| Catalogues reached | Yang+2019 162,262 / Okamoto+2021 2,344 / Shibayama+2013 1,547 / Günther+2020 8,695 / Tu+2022 15,638.  Pietras+2022 `QUERY_RETURNED_ZERO_ROWS` under its bibcode and under an author keyword |
| Significant at the watch FDR / at α = 0.05 | 225 / 141 |
| Tiers as first reported | 53 `watch`, 14 `interest`, 1 `candidate` |

Then the contamination work, which is where the numbers move.

**The long-period tail was window structure.**  154 of the 2,548 Kepler stars
put their best period in 355–389 d and 73 of the 583 TESS stars in 234–257 d
— unrelated stars agreeing on a period to better than 1%, the TESS group at
period/span = 0.32 ± 0.02 against the span/3 grid edge and the Kepler group
at the 372.5 d spacecraft year.  Every one had had 1–5 chances to repeat.
`population_period` and `few_cycles` (§5) remove **877 of the 3,131** scanned
stars, and **20 of the 68 survivors** — all `watch`, all at 195–381 d with
1–5 cycles or in the 0.21–0.23 d Kepler grid-floor pile-up.  The 14 `interest`
stars and the `candidate` are untouched by both, which is the intended
asymmetry.

**The light curve then contradicts most of what is left.**  40 shortlist
stars were fetched from MAST (run 35652897914's `redetect` job):

* `kepler:5879574`, the only `candidate`: the independent detector found
  **74** flares of its own against 21 catalogued, at P = 0.4232819 d versus
  the catalogue's 0.4232741 d — agreement to 2 × 10⁻⁵, p ~ 10⁻⁵⁴, strict
  quality, `confirms_catalogue_clock`.  The clock was in the photometry and
  not only in the catalogue.  **It is now settled, and it is not a clock in
  the flares.**  See the next section.
* the 14 `interest` stars are **all** `tess_tu2022`, and 13 of them recover
  **0.000** of their catalogued flares while the detector finds 8–51 flares
  of its own on the same light curves.  The median recovery over all 40 stars
  is 0.064.  That is not yet a rejection — a detector missing 94% of a
  catalogue cannot reject anything — which is exactly why §4.7b's
  threshold-free epoch stack exists.

**The one candidate is settled: the signal is real, it is not flares, and it
is not this star's.**  Runs 35796061650, 35796768720 and 35797574316
(2026-09-22) ran the single-star vet (§4.7d) on `kepler:5879574` — every
variability catalogue by name, the fold at P and at 2P, the events' and the
catalogued epochs' place in phase, Gaia astrometry, the aperture census and
the spacecraft-roll test.  Verdict, with nothing unreached:

```
MUNDANE_EXPLANATION_FOUND(
  CONTAMINATING_VARIABLE_AT_P:gaia2053563953175635712@13.3arcsec,
                              KIC 5879583,type=RR,P=0.4232946,in=vsx+ztf_chen2020;
  COHERENT_OSCILLATION_AT_P:amp=0.00082,p_control=0.00498,z=68.7_with_events_masked;
  EVENTS_ON_THE_CREST:offset=-0.036cycles;
  AMPLITUDE_TRACKS_SPACECRAFT_ROLL:F=3.07,p=0.0015,ratio=2.37)  [P=0.423282 d]
```

| Question | Answer |
|---|---|
| Is it a **catalogued** eclipsing binary? | **No.**  Kepler Eclipsing Binary Catalog (Kirk+2016, all ten `J/AJ/151/68` tables, by KIC): not listed.  Gaia DR3 `vari_eclipsing_binary`, `vari_summary`, `vari_classifier_result`, `vari_rotation_modulation`, `vari_short_timescale`, `nss_two_body_orbit`: not listed.  Gaia DR3 `I/358/veb`, `I/358/vclassre`, ZTF (Chen+2020): not listed.  **VSX lists it**, as `ROT`, P = 11.107 d, amplitude 0.014 Kp — the rotation, not the clock. |
| Is it an **uncatalogued** eclipsing binary folded at half its period? | **No.**  At 2P the fold's two half-phase minima are equal to `delta = 0.14σ` (masked: 0.11σ), and the Fourier fit at 2P puts A₁ = 3.2 × 10⁻⁶ against A₂ = 3.78 × 10⁻⁴ — a factor 117. There is no signal at 2P at all; all of it is at P. |
| Is the extremum at P an eclipse? | **No.**  It is a *crest*: height 5.75 × 10⁻⁴ against depth 2.65 × 10⁻⁴, and `frac_below_half_depth` = 0.32 — the 1/3 of a sinusoid, not the few percent of an eclipse.  Fundamental fraction 0.90, A₂/A₁ = 0.32. |
| Is it a **spectroscopic** binary? | **No evidence.**  Gaia DR3 2053563953175632768 (0.049″ away once proper motion is propagated to the KIC epoch): RUWE 0.995, `non_single_star` 0, `astrometric_excess_noise` 0.0, `ipd_frac_multi_peak` 0, `duplicated_source` 0.  No Gaia RV at G = 14.79, so the RV route is silent rather than clean. |
| Then what **is** at 0.42328 d? | **A coherent photometric oscillation, at exactly the clock period, whose crests the flare detector counts as flares.**  Folded amplitude 8.40 × 10⁻⁴ peak-to-peak (0.084%), and **8.21 × 10⁻⁴ with all 74 detected events masked out** — the oscillation is not made by the events.  Against 200 control-period folds of the same light curve: z = 68.7, p at the 1/201 floor.  Once the 11.05 d rotation is detrended away the clock period is the **rank-1** peak of the periodogram, at 0.4232737 d, Baluev FAP 0. |
| Where do the flares sit? | **On the crest.**  Rayleigh on the 74 re-detected event phases at P: r̄ = 0.957, p = 3 × 10⁻²⁸, mean phase 0.936; the fitted photometric maximum is at 0.936–0.971.  Offset **−0.036 cycles = 22 minutes**.  At 2P the events do not cluster at all (r̄ = 0.107, p = 0.43), which is the same verdict from the other direction. |
| And the CATALOGUED epochs? | **The same crest.**  All 21 of Yang & Liu 2019's epochs for this star, fetched from VizieR by the vet itself (run 35797574316): r̄ = 0.974, p = 8 × 10⁻⁹, mean phase **0.943** against the re-detected events' 0.936.  At 2P, r̄ = 0.141, p = 0.66.  The published flare list and the independent detector are counting the same pulsation maxima. |

So `period_is_photometric: false` was right about what it measured and wrong
about what it was taken to mean.  It compares the clock against the *global*
Lomb–Scargle maximum of the raw light curve, and there an 0.6%-amplitude
11.05 d rotation buries an 0.084% signal completely.  **The veto as written
cannot see any oscillation that is not the largest thing in the light curve —
which is every oscillation that matters, because a large one would have been
catalogued.**  §4.7d is the replacement.

**And the signal is not this star's.**  The folded amplitude,
measured independently in each of the 17 quarters, is a function of
`quarter % 4` — the Kepler roll orientation:

| season (`quarter % 4`) | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| mean folded amplitude | 6.59 × 10⁻⁴ | 5.96 × 10⁻⁴ | 1.41 × 10⁻³ | 1.18 × 10⁻³ |
| quarters | 4 | 5 | 4 | 4 |

Every quarter respects the grouping; the phase of maximum does not move
(0.89–0.99 in all seventeen).  Between-season over within-season scatter
F = 3.07, p = 5.0 × 10⁻⁴ from 20,000 relabellings, ratio 2.37.  `quarter % 4`
is a fact about the spacecraft and not about the sky, and an intrinsic signal
diluted by crowding moves with the mask by tens of percent, not by factors.
Gaia names the source: **2053563953175635712, 13.3″ away (3.3 Kepler pixels),
G = 14.37 against the target's 14.79 — brighter — and flagged `VARIABLE`.**

**And the signal has an owner.**  The Gaia source the roll test pointed at is
a catalogued RR Lyrae, and three catalogues give its period:

| catalogue | name | type | period |
|---|---|---|---|
| VSX | **KIC 5879583** | `RR` | **0.4232946 d**, amplitude 0.575 mag (r) |
| Gaia DR3 `I/358/vclassre` | 2053563953175635712 | `RR`, score 0.982 | — |
| ZTF (Chen+2020) | ZTFJ193127.18+410759.8 | `RR` | **0.4232946 d** (g: 0.423297) |

against the clock re-detected in `kepler:5879574`'s photometry at
0.42328185 d: a fractional difference of **3.0 × 10⁻⁵**, which over Kepler's
1,340.8 observed days (3,168 cycles) accumulates **0.095 of a cycle** — 58
minutes of phase in 3.7 years, and the ZTF period is measured a decade after
the Kepler data in any case.  The two are the same period to the precision
either measurement has.

And ZTF publishes that star's Fourier shape, **R₂₁ = 0.326**, against
**A₂/A₁ = 0.3168** measured in the fold of the *target's* light curve.  (Those
are different passbands — ZTF *r* against Kepler's broad band — and RR Lyrae
Fourier parameters do move with wavelength, so this is corroboration and not a
second independent identification.  The period is the identification.)  The
signal in KIC 5879574's aperture has the period, and to within the band
difference the harmonic shape, of the RR Lyrae 13.3″ away.

**And the arithmetic closes.**  The RR Lyrae's catalogued amplitude is 0.575
mag peak-to-peak, which is **51.8%** in flux, and Gaia makes it 0.419 mag
brighter than the target (G = 14.373 against 14.792), so
F_neighbour / F_target = 1.471.  For that to appear as the **0.084%** folded
amplitude actually measured in the target's light curve, the fraction of the
RR Lyrae's flux falling inside the target's aperture must be

    f = A_obs / [ (F_nb/F_tgt) x (A_nb - A_obs) ] = 0.00110,

**0.11% of the neighbour's flux** — and per roll season, 0.083% (seasons 0
and 1) to 0.171% (seasons 2 and 3).  At 13.3″, 3.3 Kepler pixels, the PRF
wings are at exactly that level, and a factor of 2.1 between the two mask
orientations is ordinary.  So the contamination hypothesis is not merely
consistent in period and shape; it requires a leakage fraction that is the one
the geometry actually provides.  (Stated approximations: Gaia G stands in for
Kepler's band on both stars, the aperture's third-party flux is ignored, and
PDC's own crowding correction — which would raise the apparent fractional
amplitude and so lower f — is not undone.  Each is a tens-of-percent effect on
f, not an order of magnitude.)


A 0.575 mag pulsator diluted into a neighbouring Kepler aperture is an 0.084%
oscillation; its crests clear the flare detector's running-median σ on some
cycles; the result is a catalogue of "flares" on a perfect clock.  That is the
whole of METRONOME's candidate, and the verdict now leads with it:
`CONTAMINATING_VARIABLE_AT_P:gaia2053563953175635712@13.3arcsec,KIC 5879583,type=RR,P=0.4232946,in=vsx+ztf_chen2020`.

**Why the channel did not catch this before the vet.**  The `periodic_variable`
veto runs a VSX / Gaia-vari / ZTF cone at **3″** — a radius set by positional
uncertainty, which is the right radius for asking *is this star a variable*
and the wrong one for asking *is a variable putting flux in this star's
aperture*.  A Kepler pixel is 3.98″ and the optimal aperture is several of
them, so the contaminating radius is 10–20″, and at 13.3″ KIC 5879583 was
outside every cone the channel ran.  **The fix is a second cone at the
aperture scale whose hit is a contamination flag rather than an identity
flag** — the vet does this (`neighbour_context`), the assess stage does not,
and any future run of this channel should carry it before the shortlist is
believed.

**What is still not proved.**  Five independent lines — period to 0.095 of a
cycle, harmonic shape, roll-season amplitude, the required leakage fraction,
and the events sitting on the crest — all say the 0.4233 d signal in
`kepler:5879574` is KIC 5879583's.  None of them is pixel-level photometry.
Fitting the RR Lyrae's signal out of the target's target-pixel files, or
measuring the flux centroid's motion in phase with it, would convert this from
an overwhelming circumstantial case into a direct one.  It is not done here
because the candidate is already dead whichever star the oscillation belongs
to: either way it is a pulsation and not a flare clock.

Per `CLAUDE.md` this is a clean result and is not written up.  It changes the
question: see §8.

**And the channel's own calibration falsified one of its design assumptions.**
`fraction_of_rotation_population_below_jitter_max` came back **0.857** — 24 of
the 28 stars the run itself rejected as `rotation_alias` are inside *both*
strict quality gates.  The fitted jitter is a function of the event count before it
is a function of the star (median 0.030 at N = 8–11, 0.217 at N ≥ 80), because
the period is free on a ~10⁴-point grid.  The nulls are not fooled — they
maximise over the same grid, and the window null's own best fit is inside the
strict gate for under 1% of stars — so `p_window` and the FDR remain honest.
But the quality gate is a look-elsewhere floor rather than a clock criterion,
worst of all at small N, and stars below 35 events now carry
`quality_uninformative`.  13 of the 15 `interest`/`candidate` stars have
N ≤ 33.  §5 carries the measured table.

**Not yet answered, and named as such:** 13 of the 14 `interest` stars carry
`variability_catalogue_unreached` (the 2026-09-21 run reached 45.3% of its
shortlist with the KIC/TIC round trip, so the periodic-variable veto could
not be applied to them at all), and none of the 3,131 had the pool null,
the measured time lattice or the photometric veto — all of which postdate
that run.  `interest` means *the vet is incomplete*, not *it passed*.

---

## 1. The claim

Is there any star whose catalogued brief optical brightenings — the "flares"
of the public Kepler and TESS flare catalogues — occur on a **strict clock**?

Natural flares are stochastic.  Their waiting times are Poisson-like (or
power-law-tailed and clustered, which is worse for coherence, not better).
The only natural quasi-periodicity in flare *occurrence* is rotational
modulation of flare visibility — the active hemisphere turning toward us —
and that carries a phase jitter of order a tenth of a rotation cycle or more,
because a spot group is not a point and a flare is not a tick.

An artificial periodic energy release — a pulsed transmitter, a duty-cycled
engine, a beacon — is a clock whose timing jitter is far below anything
rotation produces.  The observable is therefore the **timing regularity of
catalogued events, not the events themselves**.  Everything a flare catalogue
already did (finding the events, measuring their energies) is inherited; the
channel adds one statistic and the machinery to keep it honest.

This is not a search for periodic *photometric variability* — Lomb–Scargle
over Kepler and TESS light curves has been run to exhaustion — it is a search
for periodicity in a **point process** built from the catalogued event times.
A clock with a 1% duty cycle contributes no detectable Fourier power to a light
curve and is invisible to every periodogram-based survey; in the event-time
domain it is a delta function in phase.

---

## 2. Novelty position — stated honestly, to be verified by `metronomelit`

**What is claimed, as far as this session knows:** no published work has taken
the public flare catalogues (Yang & Liu 2019; Davenport 2016; Pietras et al.
2022; Günther et al. 2020; Tu et al. 2020) and asked, per star, whether the
catalogued **event times** are phase-coherent at any period with a jitter far
below rotational, against a null that resamples the mission's own observing
windows and cadence.

**What is adjacent and must not be confused with it** (the decoys the
literature sweep tags explicitly):

| Adjacent work | Why it is not this |
|---|---|
| **Quasi-periodic pulsations (QPPs)** in individual flares | Oscillations *within* one flare, seconds to minutes, MHD physics of the flaring loop.  Different timescale, different observable. |
| **Rotational-phase dependence of flare occurrence** (Hawley et al. 2014 on GJ 1243; Doyle et al. 2018/2019 in K2/TESS; Roettenbacher & Vida 2018) | Asks whether flares cluster at a rotational phase.  They largely do not, and where they do the jitter is a sizeable fraction of a cycle.  This is the natural quasi-periodicity METRONOME's `rotation_alias` and `jitter_too_large` vetoes exist for — the *baseline*, not the target. |
| **Periodic activity in FRBs and magnetars** (CHIME 2020 and after) | Different objects; but the statistical machinery — H-test on event phases, window-resampled nulls — is the same, and it is the precedent that event-time periodicity can be established from catalogued bursts alone. |
| **Flare waiting-time distributions** (Wheatland 2000 for the Sun; Kepler/TESS follow-ups) | Characterise the *natural* null this channel resamples.  Nobody, as far as this session knows, has inverted the question. |
| **Optical-SETI pulse searches** (nanosecond laser pulse detection) | Search for the *pulse*, at ns timescales, on dedicated instruments.  METRONOME searches for the *schedule*, at hours-to-months timescales, in archival catalogues. |

**Verification status.**  The sandbox has no archive egress; nothing above was
read in full text here.  `scripts/metronomelit_fetch.py` runs on the runner,
saves **verbatim** arXiv abstracts under `results/metronomelit/`, checks every
asserted arXiv id against its expected title (`id_title_check.json`), and runs
a decoy-aware concept scan (`concept_scan.json`).  The novelty position above
is **"to be verified by metronomelit"** until that file exists and its
`decoy_free_hits` has been read.  If a decoy-free hit turns out to be exactly
this search, the channel's claim narrows to whatever that paper did not do —
and the doc changes before anything else does.

---

## 3. Data

All public, small, on VizieR (reachable from the GitHub runner; not from the
sandbox).  Table ids in `config/metronome.yaml` are **preferred seeds**; the
probe stage discovers the real table under each id at runtime (§4.6).

| Role | Catalogue | VizieR table (MEASURED columns) | Event time |
|---|---|---|---|
| Kepler flares | Yang & Liu 2019, ApJS 241, 29 — 162,262 flares / 3,420 stars | `J/ApJS/241/29/table2`: `KIC, Q, Begin, End, logE` | `Begin` (start; no peak column — `t_peak_source = t_start`) |
| Kepler superflares | Okamoto+2021, ApJ 906, 72 — 2,344 flares / 266 solar-type stars | `J/ApJ/906/72/table2`: `KIC, Date, Dur, E, Prot, Amp, …` | `Date` (peak) |
| Kepler superflares | Shibayama+2013, ApJS 209, 5 — 1,547 flares / 279 G dwarfs | `J/ApJS/209/5/table7`: `KIC, BVAmp, Date, FAmp, Dur, E` | `Date` (peak) |
| TESS flares | Günther+2020, AJ 159, 60 — 8,695 flares / 1,228 stars, sectors 1–2 | `J/AJ/159/60/table1`: `TIC, Sec, tpeak, Ebol, Prot, …` | `tpeak` |
| TESS superflares | Tu+2022, ApJ 935, 90 — 15,638 flares / 3,715 stars | `J/ApJ/935/90/table2`: `ID, Sector, PDate, Energy, Duration, …` | `PDate` (peak) |
| TESS flares | Pietras+2022, ApJ 935, 143 | `J/ApJ/935/143` returned **zero rows** from `TAP_SCHEMA` on 2026-09-06; re-tried by author keyword, result recorded in `probe.json` | — |
| *(not an event list)* | Davenport 2016, ApJ 829, 23 | `J/ApJ/829/23/table1` is a per-star FFD summary (`KIC, Prot, Nfl, alpha, beta`) — a rotation source, not a flare list | — |
| Kepler rotation | McQuillan+2014 `J/ApJS/211/24/table1`; Santos+2021 `J/ApJS/255/17/table1`; Reinhold+2013 `J/A+A/560/A4`; Yang+2019 `table1`; Davenport 2016 `table1` | `KIC, Prot` | — |
| TESS rotation | Tu+2022 `J/ApJ/935/90/table1` (71,732 stars, `Per`); Günther+2020 `J/AJ/159/60/table2` | `ID/TIC, Per/Prot` | — |
| Positions | KIC `V/133/kic`; TIC `IV/39/tic82` / `IV/38/tic` | shortlist only | |
| Periodic variables | VSX `B/vsx/vsx`; Gaia DR3 `I/358/vclassre`; ZTF Chen+2020 `J/ApJS/249/18` | 3″ cones, shortlist only | |

The column names above are the ones ARC's acquire measured (`results/arc/probe.json`)
and are the reason the first dispatch scanned nothing (§10): `Begin`, `Date`
and `PDate` matched no role pattern, and `TAP_SCHEMA` was handing every name
back double-quoted besides.  The probe now also records the catalogue's own
unit and description of the time column (`time_column_meta`) beside the
value-range guess, so a wrong time system is visible in the artefact.

**Observing windows** are built from the catalogue's own event density
(`windows_from_events`): every event in the catalogue is binned in 0.1-day
bins, and a run of empty bins is a gap when it is ≥ 0.5 d long *and* the
catalogue's mean rate — events per day over the whole span, a lower bound —
predicts ≥ 20 events in it.  With ~10² events/day in a mission-scale
catalogue, Kepler's 1–3 d inter-quarter gaps and ~1 d monthly downlinks
resolve; a sparse catalogue resolves only its long gaps, and the window label
records the effective minimum gap.  (Measured over *occupied* bins instead,
as it first was, 60 events over two TESS sectors rated themselves at ~10/day
and cut every two-day lull inside a sector into a "gap".)  When the catalogue
records the sector / quarter of every event, the density windows are
**intersected with the per-sector spans** (`windows_from_sectors`,
`intersect_windows`): the spans cut the unobserved stretches a sparse
catalogue bridges, the density model keeps resolving the gaps inside a
sector.  A published (approximate) Kepler quarter table is the fallback
below 2,000 events.  Per star, the
mission windows are clipped to the star's own event span and a window with no
events in which the star's own rate predicts ≥ 5 is dropped as presumed
unobserved (module failure; TESS target not on silicon that sector).  **This is
a stated approximation**: it slightly weakens the evidence against a clock that
fell silent for a window, in the conservative direction for a candidate claim
but the permissive one for the null.

---

## 4. Method (`src/seti/metronome/`)

### 4.1 Pre-processing, per catalogue
* **Cross-star coincidence removal** (`clock.cross_star_coincidence`).  Bins
  of two long cadences (Kepler) / five 2-min cadences (TESS); a bin in which
  the number of *distinct* stars with an event exceeds
  `max(5, Poisson_isf(10⁻⁶/n_bins, λ) + 1)` is spacecraft, not sky (momentum
  dumps, Argabrightenings, scattered-light excursions).  Every event in it is
  removed from every star **before any star is scanned**, and the epochs and
  counts are in `summary.json`.
* **Declustering** (`clock.decluster`).  Events within 0.1 d (2.4 h) are one
  event; complex and sympathetic flares are one energy release.

### 4.2 The statistic, per star with N ≥ 8 events
For each trial frequency on a grid from ~10 cadences (0.2 d floor) up to a
third of the star's span, oversampled ×5 in 1/span: the **H-test** (de Jager
et al. 1989) on the event phases, `H = max_m (Z²_m − 4m + 4)`, m ≤ 4.  A clock
is a delta in phase and fills every harmonic; rotational modulation
(`rate ∝ 1 + cos φ`) is a pure fundamental.  The grid peak is refined locally
and then walked up the harmonic ladder (`fundamental_period`): a zero-jitter
clock is exactly as coherent at P/7 as at P, so the grid maximum can land on a
sub-harmonic, and the walk takes the longest multiple that keeps the phase
concentration — stopping at 2P for any clock that ticks in consecutive cycles.

At the best period the **clock quality** is read off the phase distribution:

| Quantity | Definition | Clock | Rotation |
|---|---|---|---|
| `Q` | `1 − s/s₀`, s = angular deviation √(2(1−R̄)), s₀ its uniform expectation | → 1 | ~0.3 |
| `jitter` | rms residual from `t₀ + kP`, in units of P | ≲ 0.01 | 0.15–0.3 |
| `f_in_window` | fraction within ±0.05 cycle of the clock phase | → 1 | ~0.1 |
| `gap_integer_frac` | fraction of consecutive same-window waiting times that are integer periods (±0.05) | → 1 | ~0.05–0.1 |
| `jitter_core`, `n_core` | the same rms, over the events within ±0.15 cycle of the clock phase (the ticks, when a natural background is mixed in) | ≲ 0.01 (0.03 with a 30% background) | ~0.087 |
| `gap_integer_frac_core` | `gap_integer_frac` over the core events only | → 1 | ~0.1 |
| `cycle_occupancy` | ticks with an event / ticks in observed time | duty cycle | — |

`gap_integer_frac` is the property a clock has and nothing else does, and it
is what makes the second null interpretable (§4.3).

**Why the core numbers exist.**  A beacon on a star that also flares
naturally is a delta in phase plus a uniform background, and the rms
statistics over *all* events do not know which is which.  Measured offline
(`test_clock_with_a_natural_flare_background_reaches_strict_quality`): a
strict clock with 30% random flares mixed in has Q = 0.49 and jitter = 0.13
— failing even the *watch* thresholds — at p ≈ 10⁻⁶⁴, with 75% of its
events inside ±0.05 cycle and a core jitter of 0.002; and its
`gap_integer_frac` of 0.56 would have tripped `bursty_random`, because a
background flare between two ticks splits one integer gap into two
non-integer ones without the ticks having moved.  So the quality gates have
two routes, either suffices: the **rms route** (Q, jitter over everyone) and
the **core route** (`f_in_window` ≥ 0.6 *and* `jitter_core` ≤ 0.05 over
≥ 8 core events; watch: 0.4 / 0.12), and the integer-gap test accepts the
core events' gaps as well as everyone's.  Rotational modulation cannot take
the core route: `rate ∝ 1 + cos φ` puts 0.20 of its events inside ±0.05
cycle and has a core jitter of ≈ 0.087 over ±0.15, below both gates, and
the rotator test still lands on `jitter_too_large`.

### 4.2b The catalogue's own time lattice, MEASURED

A published peak time is not a real number: it is the time stamp of a cadence,
so every time in a catalogue sits on a lattice, and the spacing of that lattice
is **not reliably the mission cadence**.  `windows.infer_time_grid` measures it
from the catalogue's own distinct times (candidate spacings built from the
smallest observed gaps and their integer divisors; the largest spacing on which
≥ 97% of distinct times sit within 12% of an integer step wins, then a
least-squares refinement).  It is recorded per catalogue in
`summary.json["coverage"]["time_grid"]` with the fraction on the lattice, so a
reader can see whether one was found at all.

Why it matters, concretely: Tu+2022's TESS peak times are spaced by **0.0069 d**
(9.94 min) — every one of the 30 cross-star epochs in the 2026-09-21 run is an
exact multiple of it — while the configured TESS cadence is 0.0013889 d (2 min).
The window null snaps its draws to `Windows.cadence_days`; a null quantised five
times finer than the data is a null that cannot reproduce the data's own
sampling.  The measured grid now sets that cadence, and with it the shortest
period the scan will entertain (`n_cadences_min` *real* steps rather than
assumed ones).

`windows.lattice_phase_limits` turns the same number into the two limits it
implies at a given period P: a **jitter floor** `g / (P √12)` — the rms phase
jitter the rounding forces on *any* clock, however perfect — and a phase-comb
spacing `g/P`.  A star whose measured jitter sits at that floor is as tight as
its time stamps allow and no tighter, and is flagged `quantisation_limited`
(report-only).  What is *not* claimed: a low-denominator rational P/g is not by
itself a problem.  With P/g = a/b in lowest terms the comb has **a** teeth, not
b, so the only resonances that bite are short periods, and for the catalogues
in hand that means P ≲ 30 g — below or at the scan's own floor.

### 4.3 Three nulls, and what each is for
1. **Window-resampled** (`clock.window_null`): N times uniform in the star's
   own observed time, snapped to the mission cadence, scanned identically.
   Quarter gaps, sector gaps, orbit gaps and 30-min quantisation are inside
   the null by construction.  The p-value is the sequential Monte Carlo p of
   Besag & Clifford (1991): trials continue until 10 exceedances or 2,000
   trials (300 s budget).  An ordinary star stops after ~20 trials; a star
   nothing exceeds runs the full budget and its p is quoted from a Gumbel fit
   to the null's own max-H tail, **flagged `p_extrapolated`**.  Stars whose
   loose Bonferroni bound (`n_freq · e^{−0.3H}`, deliberately heavier-tailed
   than the true `e^{−0.4H}`) already exceeds 0.5 skip the null: they could not
   have been significant, and `null_computed = False` says so.  A star whose
   Q / jitter already fail the *watch* thresholds can never rank above `none`
   whatever its p, so its null is capped at 200 trials
   (`null_budget_mode = not_clock_reduced`) — an exact restatement of the tier
   condition, and what keeps a catalogue full of rotation-modulated stars from
   spending the whole budget on p-values that cannot change a tier.  Measured
   at the science grid (~35,000 frequencies): 0.18 s per scan at N = 50, 5.5 s
   at N = 1,500; a rotator costs ~3 s, a strict clock runs its full budget.
2. **Waiting-time shuffle** (`clock.shuffle_null`): the star's own waiting
   times, permuted in observed-time coordinates.  Burstiness is preserved;
   long-range phase order is destroyed.  **A strict clock survives this null**
   — its waiting times *are* the signal — so `p_shuffle` is not a candidate
   criterion by itself.  The joint condition *`p_shuffle ≥ 0.05` and
   `gap_integer_frac < 0.6`* is the `bursty_random` rejection: coherence
   explained by the waiting-time distribution *without* the waiting times being
   clock-like.

3. **Pool-resampled** (`clock.pool_null`), the empirical one: N times drawn,
   without replacement, from the **other stars' catalogued event times** that
   fall inside this star's own windows.  Nulls 1 and 2 model the sampling —
   the windows, and a cadence.  This one models nothing.  It resamples the
   catalogue's real time lattice, its sector duty cycle and whatever epochs
   its detector preferred, exactly as they are, without anyone having to know
   what they are.  The star's own times are excluded; cross-star coincidences
   are already removed, so a shared instrumental epoch cannot be laundered
   through the pool.  200 trials, so the smallest reachable p is ~0.005; it
   exists to kill sampling artefacts, not to re-rank clocks that null 1 has
   already put at p ~ 10⁻²⁰.  `pool_null_explains` (p_pool ≥ 0.05) is a hard
   veto, ordered **first** because it is the most mundane explanation there
   is.  A pool of fewer than 3N times inside the windows cannot support the
   draw; that is `pool_null_unreached`, which caps the tier at `interest` —
   never a silent pass.

   Measured offline (`test_pool_null_kills_a_lattice_artefact_...`): a
   catalogue of stars whose events are placed **at random on one coarse
   lattice** — nothing periodic anywhere — is called a perfect clock
   (jitter < 10⁻⁶, Q > 0.999, p_window < 0.05) by a window null snapped to a
   finer cadence, and is killed by the pool null (p_pool > 0.05).  A genuine
   clock on the same lattice beats both.

**Calibration of the extrapolation** (measured in the sandbox, 300 Poisson
stars in Q2–Q8 Kepler windows, N = 30, 200-trial nulls): fraction of stars
with Gumbel-extrapolated p < 0.01: **1.7%**; < 0.05: **7.3%**; < 0.5: 47%.
The empirical p on the same stars gave 1.7% / 6.7% / 48%.  Within binomial
noise of uniform; the extrapolation is not conservative and is therefore
flagged wherever it is used.

### 4.4 Tiers, with BH-FDR across every star scanned
`p_window` for every scanned star (empirical where available, the Bonferroni
screen bound where the null was skipped, the flagged Gumbel value where pinned
at the floor) goes through Benjamini–Hochberg at α = 0.05 (`fdr_significant`)
and α = 0.25 (`fdr_watch`).  The per-star max-statistic null already absorbs
the period-grid trials; BH absorbs the star count.

| Tier | Requires |
|---|---|
| `none` | not `fdr_watch`, or any hard veto (§5) |
| `watch` | `fdr_watch`, no hard veto, loose quality (Q ≥ 0.6, jitter ≤ 0.12 — or `f_in_window` ≥ 0.4 with `jitter_core` ≤ 0.12) |
| `interest` | `fdr_significant`, strict quality (Q ≥ 0.85, jitter ≤ 0.05 — or `f_in_window` ≥ 0.6 with `jitter_core` ≤ 0.05 over ≥ 8 core events; and `gap_integer_frac` or `gap_integer_frac_core` ≥ 0.6 over ≥ 4 gaps), but a veto **could not be applied** (no P_rot; a variability catalogue unreached) |
| `candidate` | as `interest` with every veto applied and passed.  **Always pending light-curve inspection** — which §4.7 now performs. |

### 4.5 Energy–phase coherence (report only)
Spearman ρ of log energy against distance from the clock phase.  A clock does
not care how bright the tick is; rotational modulation does (visibility).
`energy_incoherent` when p < 0.01.  Never rejects — a beacon *could* modulate
amplitude — but a reader sees it.

### 4.6 Runtime schema discovery (`acquire.py`)
VizieR catalogue numbers and column names are not stable facts (the sibling
`tailings` channel lost three dispatches to a renumbered table and a mangled
column).  The probe stage lists every table under the preferred id in
`TAP_SCHEMA.tables`, reads their real columns from `TAP_SCHEMA.columns`,
resolves the roles (star id; peak / start / end time; energy; sector; P_rot;
position) with exact canonicalised regexes (`resolve_columns` — substring
matching is what let `Per` match `Perr` elsewhere), scores each table as a
per-flare list, falls back to a keyword search, and writes the whole
scoreboard to `probe.json` with the time-system guess (`BKJD` / `BTJD` / `BJD`
/ `MJD`, from the median of a 200-row peek through a verified column) and the
catalogue's own unit / description of that column.  The acquisition log
separates `QUERY_FAILED` from `QUERY_RETURNED_ZERO_ROWS` at every stage.
Star ids get one spelling across every table (`clean_star_id`, shared with
ARC): `KIC 757099`, `757099.0` and `757099` are one star, and the rotation
join depends on it.

### 4.7 Flare re-detection on the shortlist's own light curves (`redetect.py`)
The catalogue stages inherit somebody else's flare finder.  For the stars
that matter — every star at `watch` or better after assess, plus the most
flare-rich stars per catalogue, capped at `redetect.max_stars` — the
`redetect` stage goes back to the light curve: the Kepler / TESS products are
fetched through GROWTH's bounded MAST loop (`seti.growth.stage2`,
`lightkurve` first, `astroquery.mast` + FITS without it; long cadence only,
as the Kepler catalogues used), the brief brightenings are found again with
one stated detector — a running-median baseline over 0.5 d inside each
contiguous run of cadences, a MAD sigma per run, ≥ 3 consecutive cadences
above 2.5σ with the peak above 3.5σ, the peak cadence as the tick — and the
identical clock statistic runs on the re-detected peaks with the observing
windows read off the light curve itself (every downlinked cadence is in the
file, so these are the true windows).  Per star the record carries the
fraction of the *catalogue's* flares the detector recovered
(`catalogue_recovery_frac`, its calibration on that star), the re-detected
period / Q / jitter / core numbers / p, whether the period agrees with the
catalogue's (or a low harmonic), and `confirms_catalogue_clock`: a catalogue
clock is **confirmed** only when the independent detector finds it at the
same period with the same strict quality gate the tiers use **and** the
period is not the star's own dominant photometric periodicity (below).

**The re-detection's own confounder, measured rather than assumed.**  A
running median over 0.5 d cannot flatten a photometric oscillation of
comparable period, so the worry is that its maxima become a train of "flares"
at exactly the photometric period — with the phase stability of the
oscillation rather than of any flare mechanism.  Two synthetic light curves
containing **no flares at all** settle what the detector actually does
(`test_a_sinusoid_does_not_manufacture_flares_but_a_sharp_pulsator_does`):

* a pure **sinusoid** does *not* do it.  The residual is a sinusoid too and
  the MAD sigma is set by that same sinusoid, so its maxima sit at ~1.4σ,
  under the 2.5σ / 3.5σ gates.  Zero flares are found.
* a **sharp-peaked** periodic signal does.  A pulsator's sawtooth maximum, a
  heartbeat brightening, a contact binary: the bulk of the cycle sets the MAD,
  the narrow peak clears it every cycle, and out comes a perfect clock at the
  photometric period.

So `redetect` measures the Lomb–Scargle peak of the star's own (unmasked)
flux — `phot_period`, `phot_amplitude_frac` — and `clock_in_lightcurve` is
False when the re-detected period *is* that dominant photometric periodicity
or a low harmonic of it.  The flux is used unmasked deliberately: masking the
detected events would punch a hole at exactly the period under test and
imprint it on the window function.  The price is that a genuine flare clock
also contributes some power at its own period, bounded by the events' duty
cycle — which is reported beside it, with the median event duration and the
rise fraction `(t_peak − t_start)/duration`.  A flare rises in about a cadence
and decays over several (rise fraction well under 0.5); a symmetric
photometric maximum has ~0.5.  Those two are **reported, not vetoed on**: at
30-min cadence a three-point event cannot resolve the asymmetry.  A light-curve
clock the catalogue did not show is reported separately
(`LIGHTCURVE_CLOCK_WITHOUT_CATALOGUE_AGREEMENT_n`) and is not a candidate
until vetted.  A dead MAST is `NO_DATA_REACHED`, an empty one
`NO_LIGHTCURVES_FOUND`; the stage checkpoints after every star.

### 4.7b The catalogue-epoch stack — the threshold-free question

Re-detection asks a harder question than the shortlist needs, and the
2026-09-21 run showed it is not sharp enough on its own.  Over 40 stars
fetched from MAST the median `catalogue_recovery_frac` was **0.064**, and
13 of the 14 `interest` stars (all `tess_tu2022`) recovered **0.000** of
their catalogued flares while the detector found 8–51 flares of its own on
those same light curves.  17 of the 40 never got past `TOO_FEW_FLARES`.  A
detector that misses 94% of a catalogue cannot be the reason a star is
rejected: a non-confirmation from it is a statement about its threshold.

So `catalogue_epoch_response` asks the simpler question, with no threshold
anywhere in it: **is there flux at the catalogue's own epochs?**  It reads
the detrended residual in units of the run's own robust σ at the times the
catalogue put its flares, and compares that to the same statistic at
`epoch_n_control` random times inside the same observing windows, with a
one-sided p bootstrapped from the control stack.  If the catalogued epochs
carry no more flux than random epochs, then whatever pattern they form is a
pattern in the *catalogue*, not in the star — and no amount of phase
coherence changes that.  It runs **before** the `n_min` gate, because the
stars whose light curve yields too few flares to re-detect are exactly the
ones with no other answer available.

The same stack is repeated at a list of time offsets — including
±2 400 000.5 and ±2 457 000, applied to *every* catalogued time rather than
only those already inside a window.  A catalogue in the wrong time system
then announces itself as a stack peak at a named shift
(`cat_best_offset_days`) instead of looking like an empty catalogue.

**What that 0.064 median means about the flare catalogues themselves.**  The
epoch stack turns the question round and makes it answerable, and the answer
is about the catalogues rather than about the stars.  Of the 40 shortlisted
stars, 39 were fetched and 39 got an epoch-stack verdict:

| | stars | epoch σ (median) | control σ (median) |
|---|---|---|---|
| catalogued epochs **are** brightenings | 22 | 3.91 | 0.82 |
| catalogued epochs are **not** | **17** | 1.19 | 0.95 |

Per catalogue: **14 of 28** tested stars from Tu+2022 (TESS), **2 of 10** from
Yang & Liu 2019 (Kepler), **1 of 1** from Shibayama+2013.  **18 of the 39
recover exactly 0.000** of their catalogued flares with an independent
detector on the same light curve, and the median recovery is 0.064.  Nine
stars were demoted on `catalogue_epochs_absent`; all nine are `tess_tu2022`.

Three things make this a statement about the lists rather than about the
pipeline that read them:

* **it is not a time-system error.**  The stack is repeated at ±2 400 000.5
  and ±2 457 000 applied to every epoch; not one of the 17 picked up a shifted
  peak.  `cat_best_offset_days` is 0.0 for 8 of them and |offset| ≤ 0.1 d for
  five more; the largest is −2.0 d, which is noise on this statistic;
* **it is not the detector's threshold.**  The stack has no threshold in it.
  It compares the detrended residual in run-σ at the catalogued times against
  the same statistic at thousands of random times inside the same windows;
* **the control arm behaves.**  Among the 17 the control σ (0.95) is
  indistinguishable from the epoch σ (1.19), while among the 22 the epochs
  stand 4.8× above their controls.  The instrument separates the two
  populations cleanly.

Three caveats are owed, and they bound the claim rather than dissolve it.
**(i)** These 40 stars are *not* a random sample of the catalogues — they are
the clock shortlist, selected because their catalogued times form an unusually
regular pattern, and a list of spurious epochs laid down by a pipeline's own
periodic systematics is exactly the kind of list that would be selected.  So
44% is **not** a catalogue-wide false-positive rate, and this channel cannot
produce one from this run.  **(ii)** Tu+2022 publishes *superflare
candidates*, and the paper says so; finding that half of a selected subset of
them have no measurable photometric response is consistent with the label.
**(iii)** The light-curve product this channel fetched need not be the one the
catalogue was built on, and for TESS that gap is real.

What the result does establish is narrower and still worth recording: **for
a substantial fraction of shortlisted stars, the published flare epochs carry
no more flux than random times in the same observing windows** — so any
analysis that treats a flare catalogue's timing as a measurement of the star,
without going back to the photometry, is resting on times that in these cases
are not events.  That is why `catalogue_epochs_absent` runs *before* the
`n_min` gate and why it is a hard veto.

### 4.7c Reconciliation — the light curve has the last word

Assess runs before any light curve is fetched, so its headline was written by
the catalogue statistics alone.  `reconcile_summary` folds the photometry
back into `summary.json` and `candidates.json`: every candidate / interest /
watch row gets a `redetect` block — including the honest
`{"status": "not_attempted"}` — and a candidate or interest star is demoted
to `none` on either of two named vetoes, most-mundane-first:

| | |
|---|---|
| `catalogue_epochs_absent` | the epochs are not brightenings in the photometry |
| `photometric_oscillation` | the re-detected period *is* the dominant photometric period |

Counts, the funnel and the verdict string are recomputed from the demoted
tiers.  **Demotion only ever removes a claim**: a light curve confirming a
clock is recorded (`confirms_catalogue_clock`) and left to the vet, because
agreeing with the catalogue is not the same as having passed the gauntlet.

`stage=redetect` with `reduce_only_run_id` runs this stage **alone** over a
prior run's acquire + assess artifacts — one runner, MAST only, no archive
and no screen matrix — so the decisive test can be re-run without waiting
behind a re-acquisition.

### 4.7d The single-star vet (`vetstar.py`) — the mundane explanations, asked by name

`period_is_photometric` asks whether the clock period is the light curve's
**dominant** periodicity.  Run 35796061650 measured what that misses: a
coherent 0.084% oscillation at the clock period, buried under a 0.6% rotation,
returned `false` — and it was the whole answer.  A grazing or diluted
eclipsing binary, a g-mode pulsator or a blended neighbour is *never* the
dominant periodicity, because if it were it would already be catalogued.

So the vet asks one star every mundane question explicitly, and reports each
as a number:

1. **What the catalogues say**, by name, with three distinct answers —
   `OK`, `NOT_LISTED`, `UNREACHED` — never merged, because an archive that
   could not be reached is not a statement about the star.  Gaia DR3
   `vari_summary`, `vari_classifier_result`, `vari_eclipsing_binary`,
   `vari_rotation_modulation`, `vari_short_timescale` and `nss_two_body_orbit`
   by `source_id`; VSX, ZTF (Chen+2020), `I/358/vclassre` and `I/358/veb` by
   cone; and for a Kepler star the one that settles it — the **Kepler
   Eclipsing Binary Catalog** (`J/AJ/151/68`), by KIC, every table under the
   seed enumerated through `TAP_SCHEMA` first.
2. **The fold, at P and at 2P.**  Detrended on a 2-day running median: long
   compared with the clock (which survives at ~95% of its amplitude) and short
   compared with an 11-day rotation (which does not).  The detector's own
   0.5 d window would eat the very signal under test.  Reported: peak-to-peak,
   crest-or-dip, `frac_below_half_depth` (⅓ for a sinusoid, a few percent for
   an eclipse), the first four Fourier amplitudes, and **at 2P the two
   half-phase minima separately** — unequal minima are an eclipsing binary
   folded at half its orbital period, which is the classic trap.
3. **The fold with the detected events masked out.**  This is the measurement
   that separates the two hypotheses, and nothing else does: if the folded
   amplitude survives the mask, the star carries an oscillation independent of
   the events; if it collapses, the fold *is* the events.
4. **Where the events sit.**  Rayleigh at P and at 2P, and — because a clock
   is by construction a phase-clustering statement at its own period, which
   makes the P test circular — the offset between the events' mean phase and
   the **fitted** photometric crest.  Fitted, not binned: masking the events
   punches a hole at the crest, and the argmax of a holed profile sits on the
   edge of the hole.
5. **The astrometric route.**  RUWE, `non_single_star`, astrometric excess
   noise, `ipd_frac_multi_peak`, and RV scatter where Gaia has an RV.
6. **The aperture.**  Every Gaia DR3 source within 20″ (a Kepler pixel is
   3.98″), proper motion propagated back to the KIC epoch before the
   separation is measured; the brighter and the Gaia-`VARIABLE` ones are put
   to the same variability tables as the target, and any of them catalogued at
   P, 2P or P/2 ends the question (`contaminating_variable_at_p`).  And
   `roll_season_test` (§5) on the per-quarter folded amplitudes.
7. **Its own catalogued epochs, without an artifact.**  One star's flare times
   are a VizieR query, not a download: `catalogue_epochs_for_star` resolves
   the catalogue's time columns from `TAP_SCHEMA` and takes `t_peak`, or the
   `Begin`/`End` midpoint where the catalogue has no peak column.

**The folded amplitude is judged against control periods, not against the
per-bin error.**  With ~600 cadences in a phase bin the standard error is
tiny, and `amplitude_sigma` clears five on a pure flare clock too — measured
at 5.5 on the offline synthetic where the correct answer is "nothing there".
So the same fold is repeated at 200 periods drawn uniformly in frequency
across ±35%, skipping the peak's own frequency width (the larger of 2% and
twenty resolution elements) and the low-order rational multiples that a
per-bin **median**, unlike a mean, leaks signal through.  Those folds carry
the star's own red noise, gaps and cadence.  `p_empirical` and `z_control`
are what the vetoes read.

`stage=vetstar` runs it on one star, one runner, in ~2 minutes.

---

## 5. Contamination ledger — every rejection is a named counter

Applied most-mundane-first; a star tripping several is reported under the
dullest.  `summary.json["rejection_counters"]` carries `first_veto` (one per
star), `flags_raised` (every flag) and `tiers`.

| Rule | Mechanism | Test |
|---|---|---|
| `cross_star_coincidence` | Spacecraft systematics shared by many stars (momentum dumps, Argabrightening, scattered light) | Removed at event level before scanning; counted per star (`n_removed_cross_star`) and per catalogue |
| `insufficient_events` | < 8 events after declustering and removal | — |
| `not_significant` | The window-resampled null explains the coherence | BH-FDR |
| `cadence_alias` | P at a named instrumental period or its 2×, 3×, ½, ⅓: Kepler long cadence, ~3 d momentum dumps, ~31 d downlinks, ~93 d quarters; TESS 2-min / 10-min / 200-s / 30-min cadences, ~3.5 d early-sector momentum dumps, 13.7 d orbit, 27.4 d sector | 2% tolerance.  The brief's "6.02 h" Kepler figure could not be verified and is **not** applied. |
| `rotation_alias` | Rotational modulation of flare visibility — the dominant natural quasi-periodicity | P within 3% of P_rot, P_rot/2, /3, /4, 2P_rot, 3P_rot, from McQuillan/Santos/Reinhold or the flare catalogue's own P_rot |
| `periodic_variable` | A pulsator's or eclipsing binary's cycles chopped into "flares" by the flare finder (RR Lyrae, δ Sct, EBs) | VSX / Gaia DR3 vari / ZTF cone at 3″; P within 3% of the catalogued period or its ½, ⅓, 2×, 3× |
| `pool_null_explains` | The catalogue's own sampling — its time lattice, its sector duty cycle, its preferred epochs — reproduces the coherence, with nothing modelled | `p_pool ≥ 0.05` against 200 draws from the other stars' event times inside this star's windows |
| `population_period` | **Unrelated stars of the same mission share this period.** A clock belongs to one star; a period many independent stars agree on to 1% is a property of the mission's sampling. Measured from the run's own scanned population, so it needs no list of instrumental periods and catches the ones nobody wrote down | ≥ 4 other scanned stars within 0.005 dex, Poisson-rarer than 10⁻³ against the local background density over ±0.25 dex. Not applied to a mission with < 50 scanned stars |
| `few_cycles` | The period repeated too few times inside the observing windows for "recurs" to mean anything — the long-period tail where P approaches the span/3 grid edge and three sector groups phase up | `cycles_span < 10`, where `cycles_span` counts the ticks whose ±0.05-cycle phase window had *any* observing coverage |
| `catalogue_epochs_absent` *(light curve)* | **The catalogued epochs are not brightenings in the star's own photometry.** Whatever pattern they form is a pattern in the catalogue, not in the star | Detrended residual in run-σ at the catalogued times vs. the same at random times inside the same windows; bootstrap p > 0.01 or median < 2σ over ≥ 8 epochs |
| `photometric_oscillation` *(light curve)* | The re-detected period **is** the star's dominant photometric period: a running median over 0.5 d cannot flatten an oscillation of comparable period with a narrow maximum, and the surviving maxima are detected as a flare train | Lomb–Scargle peak of the flux itself within 2% of P or its ½, 2×, ⅓, 3× |
| `contaminating_variable_at_p` *(vet)* | **A Gaia source inside the aperture is a catalogued variable at the clock period.**  Not "the amplitude behaves like a blend" but "that star, this far away, is listed at this period" — the least speculative statement available, so the verdict leads with it | Every Gaia DR3 source within 20″ that is brighter than the target or flagged `VARIABLE` is put to the Gaia vari tables and a VSX / ZTF / Gaia-vari cone; a catalogued period within 1% of P, 2P or P/2 |
| `coherent_oscillation_at_p` *(vet)* | **A coherent photometric oscillation sits at the clock period and survives masking every detected event.**  The events are its crests; the detector's running median cannot flatten a cycle of comparable length.  This is `photometric_oscillation` without the requirement that the oscillation be the *dominant* one | Folded amplitude with events masked, against 200 control-period folds of the same light curve: `p_empirical ≤ 0.01` **and** `z_control ≥ 5` |
| `events_on_the_crest` *(vet)* | The events' mean phase coincides with the fitted photometric maximum: they are that oscillation's peaks being counted as flares | `|offset| ≤ 0.15` cycles, with the oscillation significant |
| `narrow_dip_at_p` *(vet)* | The fold at P is a narrow **dip**, not a crest — an eclipse | `extremum_is_dip`, `frac_below_half_depth ≤ 0.25`, fold significant against controls |
| `unequal_minima_at_2p` *(vet)* | The fold at 2P has two half-phase minima of different depth: an eclipsing binary found at half its orbital period | `delta_sigma ≥ 3` on the **unmasked** fold (masking brightenings manufactures both a dip and an inequality), with the 2P fold significant against controls |
| `amplitude_tracks_spacecraft_roll` *(vet, Kepler)* | **The folded amplitude is a function of `quarter % 4`** — the roll orientation, a fact about the spacecraft and not about the sky.  An intrinsic signal is diluted by crowding, which moves with the mask by tens of percent; a neighbour's signal scales with how much of *that* star's flux the mask catches, which moves by factors.  So the signal belongs to another star | Between-season over within-season scatter of the per-quarter folded amplitude, p from 20,000 relabellings into the same season sizes: `p ≤ 0.01` **and** max/min season mean `≥ 1.5` |
| `catalogued_eclipsing_binary` *(vet)* | Some catalogue types the star as eclipsing, or gives it a period at P, 2P or P/2 | Type/class regex or period within 3% of P, 2P, P/2 |
| `bursty_random` | Clustered-but-random flaring whose coherence the waiting-time shuffle reproduces | `p_shuffle ≥ 0.05` **and** neither `gap_integer_frac` nor `gap_integer_frac_core` ≥ 0.6 |
| `jitter_too_large` | Not a clock: fails even the loose thresholds on both routes | (Q < 0.6 or jitter > 0.12) **and** (`f_in_window` < 0.4 or `jitter_core` > 0.12) |
| `energy_incoherent` *(report)* | Energy depends on clock phase — visibility, not a beacon | Spearman p < 0.01 |
| `rotation_unknown`, `variability_catalogue_unreached` *(report)* | A veto could not be applied | Caps the tier at `interest` |
| `quality_uninformative` *(report)* | Fewer than 35 events, so the strict Q / jitter gates are a look-elsewhere floor rather than a clock criterion on this star (see below) and its case rests on the null alone | `n_events < n_quality_informative` |
| `quantisation_limited` *(report)* | The measured jitter is at the floor `g/(P√12)` the catalogue's own time rounding forces: the tightness is a property of the time stamps, not of the star | jitter ≤ 1.5 × the floor from the **measured** lattice |
| `pool_null_unreached` *(report)* | Fewer than 3N other-star times inside the windows, so null 3 could not run | Caps the tier at `interest` |
| `p_extrapolated`, `null_truncated_by_budget` *(report)* | Statistical provenance | — |

**Where the clock thresholds sit — and the run's answer, which was not the
expected one.**  The assess stage measures the jitter and Q distributions of
the stars it *rejected* as `rotation_alias` — the natural quasi-periodic
population — and reports the percentiles beside the thresholds
(`summary.json["jitter_calibration"]`), with the fraction of that population
inside the strict gate.  The design intent was that the fraction be zero.

**Run 35652897914 measured 0.857.**  24 of the 28 stars it rejected as
`rotation_alias` are inside the strict gate on *both* Q and jitter.  The
absolute quality gates do not separate a clock from rotational modulation in
this data, and a large part of the reason is visible in one table — the fitted
jitter falls steeply toward small N, because the period is free on a
~10⁴-point grid and a handful of times phase up whatever they are:

| N events | jitter (obs) | jitter (this star's own window null) | Q (obs) | Q (null) |
|---|---|---|---|---|
| 8–11 | 0.030 | 0.135 | 0.845 | 0.409 |
| 12–15 | 0.039 | 0.163 | 0.801 | 0.320 |
| 16–23 | 0.059 | 0.184 | 0.713 | 0.252 |
| 24–39 | 0.079 | 0.208 | 0.626 | 0.179 |
| 40–79 | 0.147 | 0.229 | 0.347 | 0.132 |
| ≥ 80 | 0.217 | 0.247 | 0.157 | 0.081 |

(The N-trend is a population median, not a per-star law: three of the 24
rotation-alias stars inside the gate have N = 83, 83 and 115, and the four
outside it have N = 13, 16, 28 and 30.  The gate is weak across the whole
range; small N is where it is weakest.)

The window null is **not** fooled by this — it maximises over the same grid,
so it reproduces the same degeneracy: its own best fit reaches Q ≈ 0.25,
jitter ≈ 0.18, and it lands inside the strict gate for 0.8% of stars on
jitter and 0% on Q.  So `p_window` and the BH-FDR built on it remain honest,
and they are what the tiers actually rest on.  The consequence is stated
rather than papered over: **below `n_quality_informative` = 35 events the
strict quality pass carries no discriminating power**, the star gets the
report flag `quality_uninformative`, and its case rests on the null alone.
Of the 15 `interest`/`candidate` stars from that run, 13 have N ≤ 33 and 14
have N ≤ 55; only `tess:149573659` (N = 105) sits where the population's own
fitted jitter (median 0.217 at N ≥ 80) is nowhere near the gate, so its
jitter of 0.036 is the one quality number in the shortlist that is hard to
get by fitting alone.

---

## 6. Measured performance (`tests/test_metronome.py`, offline, no network)

| Case | Result |
|---|---|
| Strict clock (P = 3.137 d, duty 0.5, σ = 3 min) in Kepler Q2–Q8 windows with 30-min quantisation | recovered to < 0.1% in P; Q > 0.95, jitter < 0.01, `gap_integer_frac` > 0.9, p < 10⁻⁶ (0 exceedances in 100 trials) |
| Strict clock (P = 0.913 d) in six synthetic TESS sectors with orbit gaps, 2-min cadence | recovered to < 0.1%; same quality |
| Clock with duty 0.25 (grid peak lands on P/7) | harmonic walk returns P to < 0.1% |
| 30 Poisson stars, N = 12–30 | 0 candidate, 0 interest, ≥ 28 `none` |
| Rotationally modulated star (rate ∝ 1 + cos, P_rot = 2.5 d, N = 160) | coherence is real (p < 10⁻³) but jitter > 0.12, Q < 0.6; `rotation_alias` with P_rot known, `jitter_too_large` without |
| Momentum-dump epoch shared by 40 stars | 1 bad bin, 40 events removed before scanning, counted per star |
| Catalogued RR Lyrae (P = 0.5668 d) whose cycles are "flares" | recovered as a perfect clock, then `periodic_variable` |
| Bursty star (12 bursts × 6 events) | not `interest` or better |
| Unreachable archive (every query raises) | `NO_DATA_REACHED`, zero counts, empty candidates, every probe entry `QUERY_FAILED` |
| Archive answers with no tables | `QUERY_RETURNED_ZERO_ROWS`, "NOT a null result" in the note |
| End-to-end synthetic catalogue through probe → acquire → screen → assess | offline: `DEGRADED_SOURCE (rotation_kepler:none); CLOCK_CANDIDATES_PENDING_VET`, the clock at `interest` with `rotation_unknown`; with a rotation table and scripted cones: 1 `candidate`, the rotator now `rotation_alias`, verdict clean |
| Every hard veto, `insufficient_events`, `not_significant`, every report flag | each has a case that trips it and appears in the counters |
| Strict clock (P = 3.137 d, duty 0.6) with 30% Poisson flares mixed in | P recovered; rms Q ≈ 0.5, jitter ≈ 0.13; **core route**: `f_in_window` ≥ 0.6, `jitter_core` ≤ 0.05, `gap_integer_frac_core` ≥ 0.6 → `candidate`, no `bursty_random`; the rotator still `jitter_too_large` |
| The measured VizieR headers (Yang `Begin/End`, Okamoto & Shibayama `Date`, Tu `PDate`, Günther `tpeak`) | every table scores as an event list with the right time column; Davenport `table1` scores 0 |
| 60 sparse events over two TESS sectors 40 d apart | density model alone: 2 windows (no invented in-sector gaps); with the sector column: 40.5 d observed, the inter-sector stretch excluded |
| Injected flares (linear rise, 0.05 d decay, 12σ peak) in four synthetic Kepler quarters with monthly downlinks; 61 on a P = 3.137 d clock + 25 random | detector recovers ≥ 90% within a cadence, ≤ 10% spurious; light-curve windows split at the downlinks; the clock is recovered to < 0.2% in P and `confirms_catalogue_clock` on the core route; 60 random flares → no clock |
| Re-detection with MAST dead / empty / one star served | `NO_DATA_REACHED` / `NO_LIGHTCURVES_FOUND` / `REDETECT_CONFIRMS_NONE` with per-star status; `redetect` is never part of `--stage all` |
| 12 stars piled on one period inside a 200-star background | all 12 `population_period`; ≤ 4 of the 200 swept up; a lone star at its own period untouched; a five-star population is not tested at all |
| `few_cycles` at the boundary | 3 cycles → vetoed, exactly 10 → `candidate`; on a recovered injected clock `cycles_hit ≤ cycles_span` and occupancy is a fraction |
| 60 injected flares stacked at the catalogued epochs | epoch σ > 5× the control's, bootstrap p ≤ 0.01, > 80% of epochs above 3σ against a far lower control fraction |
| A perfectly clocked epoch list with **no** flares injected | epoch and control medians within 0.6σ, p > 0.01 → `catalogue_epochs_absent` |
| Catalogued epochs shifted by +2400000.5 (wrong time system) | none inside any window; the offset scan recovers −2400000.5 with a stack above 5σ, so a time system is named rather than read as an empty catalogue |
| Reconciliation | a candidate whose clock is its photometric period, and an interest star whose epochs are absent, are both demoted with the named `first_veto`; the summary counts, funnel and verdict follow; a confirmed star is *not* promoted; an unreached star reads `not_attempted`; a missing summary is `NO_SUMMARY` |

---

## 7. Coverage limits — what the search can and cannot see

* **The catalogue's flare-detection threshold is the channel's sensitivity
  floor.**  A clock whose ticks fall below the catalogue's amplitude / energy
  threshold, or whose ticks the catalogue's pipeline rejected, is invisible
  here by construction.  Each catalogue's threshold is inherited, not
  re-measured.
* **Cadence.**  A peak time cannot define a clock finer than ~10 cadences
  (0.2 d for Kepler long cadence; 0.014 d for TESS 2-min), and the configured
  floor `min_period_days: 0.2` binds for both missions — a deliberate choice
  that keeps the TESS grid the same size as Kepler's; lowering it for TESS is
  a one-line config change.  Shorter periods are not scanned.
* **Span — searched.**  Periods longer than a third of the star's event span
  are not scanned (fewer than three cycles is a trend).  Kepler: ≲ 500 d; a
  single TESS sector: ≲ 9 d.
* **Span — *believed*.**  Searching to span/3 and believing to span/3 are
  different things, and the 2026-09-21 run showed the difference.  154 of
  2,548 Kepler stars and 73 of 583 TESS stars put their best period in the
  long tail (355–389 d and 234–257 d respectively), unrelated stars agreeing
  on a period to better than 1%; every one of them had had 1–5 chances to
  repeat.  `few_cycles` therefore requires ten ticks inside the windows, and
  the real long-period reach is **span / 10**, reported per mission as
  `coverage.max_period_credible_days`.  For the median Kepler star that is
  ≈ 145 d and for the median TESS star ≈ 76 d.  A genuine beacon slower than
  that is outside what this data can establish, and the channel says so
  rather than reporting it.
* **The catalogue must be right about its own epochs.**  Everything upstream
  of `redetect` is a statement about somebody else's list of times.  The
  light-curve stage asks the photometry directly — is there flux at those
  epochs, and is the period the star's own photometric period — and demotes
  on either.  A star that was never reached by that stage keeps
  `redetect: {"status": "not_attempted"}` and is not credited with passing
  it.
* **N ≥ 8** after declustering: stars with fewer catalogued flares are not
  tested, and the TESS catalogues are dominated by such stars.
* **Duty cycle.**  A clock is detectable at any duty cycle high enough to leave
  ≥ 8 events; `cycle_occupancy` reports it.  A clock that ticks once per
  window is a trend.
* **Presumed-unobserved windows** (§3) are dropped, which removes the
  evidence a silent window would carry against a clock.
* **Vetting reach.**  P_rot exists for a minority of TESS flare stars; the
  variability cones are run only for the FDR shortlist.  A star at `interest`
  is one whose vet is incomplete, not one that passed.
* **The periodic-variable cone is at 3″ and the contaminating radius is
  10–20″.**  A 3″ cone answers *is this star a catalogued variable*; it cannot
  answer *is a catalogued variable putting flux into this star's aperture*,
  which on Kepler (3.98″ pixels, several-pixel masks) is the question that
  matters.  MEASURED: the RR Lyrae that produced the channel's only candidate
  sits at 13.3″ and was outside every cone the assess stage ran.  The vet runs
  the aperture-scale cone; the assess stage does not, and until it does no
  shortlist from this channel should be believed on the variability veto alone.
* **A low-amplitude oscillation at the clock period is the channel's hardest
  confounder, and it is not screened at scale.**  §4.7d catches it, but §4.7d
  runs on one star at a time and needs that star's whole light curve.  The
  screen and assess stages never see the photometry at all, so *every* star
  above `watch` should be assumed to carry an unasked 0.1%-level oscillation
  until the vet has been run on it.  `kepler:5879574` was the only star that
  had survived far enough to deserve the vet, and the vet killed it.
* **The roll-season test is Kepler-only.**  TESS sectors do not repeat a
  camera orientation the way Kepler quarters do, so the sharpest contamination
  diagnostic in the vet has no TESS counterpart here; a TESS star's blending
  would need the pixel data.

---

## 8. What a null would and would not mean

A `NO_CLOCK_CANDIDATES` verdict means: among stars with ≥ 8 catalogued flares
in these catalogues, none shows event-time phase coherence at a period between
~10 cadences and span/3 that (a) beats its own window-resampled null after
BH-FDR at 0.05, (b) has jitter ≤ 0.05 P, and (c) is not a rotation, cadence,
or catalogued-variable alias.  It says **nothing** about clocks below the
catalogue's detection threshold, clocks with fewer than 8 visible ticks,
clocks faster than ten cadences, clocks slower than a third of the span, or
clocks in the ~10⁵ Kepler/TESS stars that never entered a flare catalogue.

Per `CLAUDE.md`, a clean null is **not written up**.  It is a reason to change
the question, and the 2026-09-22 vet says which way.  The channel's one
candidate died not as a false clock but as a **real 0.084% periodicity that
nobody had catalogued**, sitting at a period no variability survey had
recorded for this star, and most probably belonging to a neighbour 13″ away.
Three of those facts are directions:

1. **Run the flare finder rather than inherit it.**  The catalogues' epochs
   were the weakest link twice over — 17 of 39 shortlisted stars' epochs are
   not brightenings at all (§4.7b), and the one star whose epochs *were* real
   turned out to be an oscillation's crests.  Detecting on the photometry
   directly removes both failure modes and recovers the sub-threshold regime.
2. **The confounder is a population, and it is a mapped one.**  The thing that
   killed the candidate was not exotic: it was an ordinary RR Lyrae 13″ away,
   catalogued in VSX, Gaia DR3 and ZTF, that no cone in the channel was wide
   enough to see.  Every Kepler/TESS flare star with a catalogued variable
   inside its aperture is a potential false clock, and the crossmatch that
   finds them is one pass over the flare stars against VSX/Gaia-vari/ZTF at an
   aperture-scale radius.  Doing that *first* is what would make a survivor
   mean something.  It is a prerequisite, not a paper.
3. **Move to an event class with no catalogue in the way**: the ZTF alert
   stream's brief brightenings, or the CHIME-style burst catalogues.  Or test
   weaker structure than a strict clock — arithmetic progressions with a
   drifting period, prime-number or Fibonacci gap patterns (S28's broader
   reading), which the H-test is not built for.  `NO_DATA_REACHED` and `QUERY_RETURNED_ZERO_ROWS`
are statements about archive access and the catalogues' contents, not about
the sky, and the workflow refuses to let either read as a science null.

---

## 9. Layout

```
src/seti/metronome/windows.py   observing-window model + the MEASURED time lattice (infer_time_grid)
src/seti/metronome/clock.py     H-test scan, clock quality, three nulls, cross-star removal, BH [pure]
src/seti/metronome/vet.py       the gauntlet and the tiers (rms and core quality routes)        [pure]
src/seti/metronome/redetect.py  flare detector on light curves, the photometric veto, the
                                catalogue-epoch stack, the redetect stage and reconcile_summary
src/seti/metronome/vetstar.py   the SINGLE-STAR vet: variability catalogues by name, the fold at
                                P and 2P, the events-masked fold against control periods, event
                                phase, Gaia astrometry, the aperture census, the roll-season test
src/seti/metronome/acquire.py   runner-only VizieR access, runtime schema discovery, AcquisitionLog
src/seti/metronome/run.py       stages probe / acquire / screen / assess (+ redetect, vetstar)
config/metronome.yaml           every threshold, every table seed
tests/test_metronome.py         offline suite (the CI gate)
tests/test_metronome_vetstar.py offline suite for the vet (injected EB, real clock, crest-counting
                                oscillation, roll contamination, every archive failure mode)
.github/workflows/metronome.yml probe+acquire -> screen matrix -> assess -> commit-back; lit job;
                                assess-only, redetect-only and vetstar over a prior run's artifacts
scripts/metronome_vetstar_report.py  prints the vet verbatim into the run log
scripts/metronomelit_fetch.py   prior-art sweep, verbatim abstracts -> results/metronomelit/
```

Entry point: `python -m seti.metronome.run --stage {probe|acquire|screen|assess|all|redetect|vetstar}
[--catalogues a,b] [--shard i --n-shards n] [--max-stars k] [--max-rows r] [--offline]
[--budget-s s] [--star-key kepler:5879574 --period 0.42328185409991]`
(`redetect` and `vetstar` run only when named — they open MAST);
programmatic `seti.metronome.run.metronome_run(cfg, stage=..., catalogues=..., shard=...,
n_shards=..., max_stars=..., max_rows=..., offline=..., seed=..., out_root=...)`.

Outputs: `probe.json`, `acquire.json`, `acquisition_log.json`, `screen_<cat>[_s<i>of<n>].json`,
`stars_<cat>[...].csv` (every star scanned), `stars_vetted.csv`, `summary.json`
(verdict, funnel, rejection counters, jitter calibration, coverage, `generated_utc`,
per-catalogue acquisition log), `candidates.json` (interest + candidate, and the watch list),
`redetect.json` + `stars_redetect.csv` (the light-curve re-detection of the shortlist,
including the catalogue-epoch stack), the `redetect` block reconciliation writes back
into `summary.json` / `candidates.json`, and `vetstar.json` + `vetstar_fold.csv`
(the single-star vet: every catalogue's verbatim answer, the four folded profiles bin by
bin, the control-period nulls, the event phases and the per-quarter roll table).
