# LANTERN — a narrow emission line that vanishes at secondary eclipse

Phase-resolved, eclipse-gated optical/IR laser SETI, executed on the JWST
exoplanet time-series archive at population scale.

---

## 1. The claim

Across **every public JWST time-series observation of a transiting exoplanet**
(transits and secondary eclipses; NIRSpec G395H/G235H/G140H/M/PRISM, NIRISS
SOSS, NIRCam grism, MIRI LRS), is there an **unresolved
(instrument-resolution-limited) emission feature** that is present outside
secondary eclipse and **absent while the planet is hidden behind its star**?

A monochromatic source *on the planet* — a laser, a beacon — is the only thing
that produces a narrow line whose flux tracks the planet's **visibility**:

* a **stellar** line does not care whether the planet is occulted; it follows
  the stellar continuum through eclipse (fractional change ≈ the eclipse depth,
  ~10⁻³, not 1);
* a **detector or instrument** feature has no orbital phase at all;
* the planet's **own thermal emission** does vanish at eclipse — but it is
  broad (molecular bands, day-side continuum). The *narrowness* requirement is
  what separates a beacon from astrophysics.

Transit is the second phase reference. During transit the planet is *in front
of* the star, so a planet-origin line is **constant** in-vs-out of transit,
while during eclipse it **disappears**. The two references together rule out
every non-planet origin: a feature that vanishes at eclipse *and* vanishes at
transit is stellar or systematic; one that vanishes at eclipse and holds
through transit is on the planet.

---

## 2. Novelty — the honest position

**To be verified by `lanternlit` on the runner** (`scripts/lanternlit_fetch.py`
→ `results/lanternlit/concept_scan.json`). The sandbox has no arXiv/OpenAlex
egress; nothing below has been re-read from a primary source in this session.

What the record is expected to show, and what the sweep is written to test:

| Layer | What exists | Why it is not this |
|---|---|---|
| **The proposal.** Kipping & Teachey 2016, *A cloaking device for transiting planets* (arXiv:1603.08928, MNRAS 459, 1233) | Lasers fired *during transit* to alter or cloak the transit light curve; a corollary that lasers make natural broadcast beacons | A transit-light-curve idea, and a proposal. It does not propose, and nobody has executed, a search of eclipse-phased **spectra** for a line that turns off when the planet is occulted. |
| **Executed optical-SETI line searches.** Tellis & Marcy 2015/2017 (Keck/HIRES, ~5600 stars), Marcy 2021–22 (LRIS), Zuckerman et al. | Single-epoch, phase-agnostic scans of stellar spectra for unresolved emission lines | No orbital phase, no planet, no occultation discriminant. The in-house precedent `jwst_bio` (one planet, out-of-transit only) is of this kind. |
| **JWST eclipse spectroscopy** (many programmes, 2022–) | Broad day-side emission spectra measured *as* the eclipse-depth spectrum | Measures the vanishing of *broad* thermal emission; never looks for an unresolved line, and the eclipse is the signal, not the gate. |
| **High-resolution day-side emission** (cross-correlation, e.g. CO/H₂O at R~10⁵) | Resolved molecular line *forests* from the planet, detected in phase-resolved ground-based data | Line forests of known species, detected by template cross-correlation; not a search for a single unidentified unresolved line, and not on JWST. |

The claim this channel is entitled to make, pending the sweep: **the
eclipse-vanishing discriminant has been proposed as physics but never executed
as a search, and never on the archive at population scale.** If the sweep
returns a decoy-free hit (an abstract tying a laser/artificial narrow line to
the planet's occultation as the test), the position is revised in this
section, with the citation, before anything else is written.

---

## 3. Method

### 3.1 Data path (runner-only)

1. **Planets.** NASA Exoplanet Archive `pscomppars`, `tran_flag=1`: period,
   T₀, T₁₄, eccentricity, ω, Rp/Rs and their errors. A row without T₁₄ gets
   it from the geometry, (P/π)·asin(√((1+k)²−b²)/(a/Rs)), flagged
   `duration_derived_from_geometry`; without that it is `phase_unresolved`.
2. **Observations.** MAST `Observations.query_criteria(obs_collection="JWST",
   dataproduct_type="timeseries", instrument_name=<inst>*)` — one query per
   instrument. TSO products are `timeseries`, not `spectrum` (the lesson
   `jwst_bio` learned). `dataRights` is kept, so proprietary products are
   **recorded as such**, never silently dropped.
3. **Cross-match.** KD-tree on the sky to the planet hosts within 30″
   (JWST `s_ra/s_dec` are epoch-of-observation; the hosts are nearby,
   high-proper-motion stars).
4. **Products.** `x1dints` only, segments grouped per exposure, one exposure
   at a time: `download_file(dataURI)` (not `download_products`, which trips a
   MAST server-side bug), read, **delete**, checkpoint the exposure's JSON
   immediately. A killed shard loses minutes.
5. **Times.** `INT_TIMES` `int_mid_BJD_TDB` (MJD-based → JD); if absent, a
   linear interpolation of `EXPSTART/EXPEND` with **no barycentric
   correction**, flagged and carried as an extra ±8-minute timing uncertainty.

### 3.2 Phase labels (`phase.py`)

Each integration is labelled `in_transit / transit_contact / out_transit` and
`in_eclipse / eclipse_contact / out_eclipse`. Eclipse time = T₀ + P/2 for
e ≲ 0.05 (or e unknown, flagged `assumed_circular`); for eccentric orbits with
known ω the first-order offset (2/π)·e·cos ω·P is applied **and** the timing
uncertainty is widened by the full possible offset, because the ω convention
differs between catalogues. Eccentric with no ω → `phase_unresolved`.

Timing uncertainty is propagated to the epoch, σ_t = √(σ_T₀² + (N σ_P)²), and
added to every contact exclusion; above 0.5·T₁₄ the contacts cannot be placed
and the observation is `phase_unresolved`. An observation that **starts inside
eclipse has no pre-ingress baseline** and is not eclipse-class: the drop must
be separable from the detector settling ramp.

### 3.3 Narrow-feature search (`line.py`)

On the time-averaged, continuum-normalised out-of-eclipse spectrum (each
integration divided by its own median before a 5σ-clipped mean; the per-pixel
error is the larger of the scatter across integrations and the propagated
error):

* continuum = **notched local quadratic** — window 31 samples, ±2 excluded
  around each sample, with a **fixed two-pass** >4σ excursion clip (the
  excursion and its ±2 wings are masked from the fit). Three things were
  learned building this on a synthetic M-dwarf line forest: a running median
  returns the centre sample exactly on a steep continuum and a line shifts
  its rank; a wide hole (as wide as the widest allowed line) turns the fit
  into an extrapolation from the window's outer lobes whose error at low
  S/N is several σ (seen as ±10σ artefacts in the in-eclipse spectrum);
  and iterating the clip to convergence with the unmasked noise is a
  positive feedback loop that ends with everything masked;
* noise = block-wise MAD of the **unmasked** residual, floored by the
  propagated error (the clip threshold itself uses the conservative
  all-sample MAD, so it cannot run away); on the synthetic forest the
  unmasked z has robust σ = 0.95;
* a feature is an interior (≥8 samples from either end), **bounded** local
  maximum at ≥6σ, not adjacent to a NaN gap, whose **matched-template width
  is consistent with 1–3 resolution elements**: Gaussian templates from a
  single-sample spike to 9 elements are fitted with a linear baseline
  (significant absorption, z < −3, excluded), and the guards act on the
  2σ-consistent (Δχ² ≤ 4) width range — a spike is one for which even 0.75
  elements is excluded, too wide is one for which every width up to 3
  elements is excluded. A point estimate of the width wanders with wing
  noise at 8σ; the range does not. These are the guards of
  `seti.panspermia.dossier.narrow_feature_scan` / `jwst_bio.laser_line_scan`,
  re-implemented for a 2048-sample grid with a line-rich stellar continuum.

Per-instrument resolving powers and sampling are a config table
(`config/lantern.yaml: instruments`), as is the artefact list.

### 3.4 The discriminant

For each feature, the **line flux per integration** is the summed excess over
a local side-window continuum; the **continuum light curve** is an independent
broad-band median (the star). Then:

| Statistic | Meaning | Candidate needs |
|---|---|---|
| `out_positive_snr` | line flux out of eclipse over its error | ≥ 5 |
| `in_eclipse_spectrum_snr` | the feature's significance on the **in-eclipse time-averaged spectrum** (same masked continuum as the search) — the bias-free "consistent with zero" test; `in_eclipse_sigma` (series mean over its error) is also reported | ≤ 2 |
| `eclipse_vanish_snr` | (out − in) / joint error | ≥ 5 (≥ 3 for `interest`) |
| `line_fractional_drop` vs `continuum_fractional_drop` | did the line drop like the star (~10⁻³) or like the planet (~1)? | line ≠ continuum, consistent with 1 |
| `continuum_correlation` | line vs continuum over **out-of-eclipse** integrations (a planet line is constant there; any correlation is systematics) | \|r\| < 0.5 |
| `ramp_correlation`, χ²_ramp vs χ²_step | best exponential settling template vs the eclipse-step model | step fits better |
| `free_step_offset` | where a free two-level model puts the drop, vs the predicted ingress | within 2 ingress durations + σ_t |
| `transit_constancy`, `transit_excess_sigma` | line change in transit vs the continuum's own change | line changes no more than the continuum |

Errors are the larger of the propagated and the empirical (scatter) error, so
a red-noise-dominated series cannot manufacture significance. The
per-integration noise of the line series is measured from the *temporal*
residual of the side pixels (each integration minus the time-median spectrum
there), never from their spread around their own median — that spread is
stellar structure, and using it suppressed every SNR ten-fold in
development.

Synthetic performance (a 2048-sample NIRSpec-like grid with a line forest,
800 integrations, 2×10⁻³ per-pixel noise): a planet line at 2% of the
continuum is recovered at 185σ with vanish SNR ≫ 5 and in-eclipse residual
1.7σ; at 0.8% it is 85σ; the same line held constant is rejected
(`low_snr`, `tracks_continuum`); a 1-sample spike, a 14-sample-wide band and
a null all yield no tier. Analysis is 2–3 s per exposure.

### 3.4a The difference search — why §3.3 alone is not enough

The first runner measurement (run 35737559234, 2026-09-22) verified the reader
and the phase labels on both known eclipses and still failed verification, on
the same thing in both cases: an injected line at **2% of the continuum
produced zero features**.

| Verify case | integrations | eclipse depth | free step vs predicted ingress | injected 2% line |
|---|---|---|---|---|
| WASP-43 b MIRI/LRS `jw01366-o011` (phase curve) | 9 216 (30 EXTRACT1D tables, row BJD_TDB) | 6 681 ppm at 15.4σ | 0.402 d (locked onto the **transit**) | 0 features |
| WASP-18 b NIRISS/SOSS `jw01366-o021` | 2 720 (6 tables, orders 1–3) | 1 451 ppm at 23.1σ | 0.007 d (tolerance 0.020 d) ✓ | 0 features |

The reason is physical. On a real `x1d` product the time-averaged spectrum's
residual around its local continuum sits at **~1% of the continuum however
long the exposure** — the static pixel pattern of the extraction (flat-field
residual, undersampled trace, wavelength-solution ripple), not photon noise.
Measured from the reported 5σ equivalent-width limits: a noise median of
**1.3% on SOSS over 2 720 integrations** and **1.1% on LRS over 9 216**. A 6σ
trigger on that spectrum therefore needs a line brighter than ~8% of the
continuum, and the screen would have been worthless at scale.

That pattern is **identical in the in-eclipse and out-of-eclipse averages**,
so it cancels exactly in their difference. What survives is what *changed*
when the planet was occulted: the planet's own (broad, smooth) emission
spectrum, which the local quadratic removes, plus any narrow line that
vanished. So the search runs on

&nbsp;&nbsp;&nbsp;&nbsp;`spec = 1 + (⟨out-of-eclipse⟩ − ⟨in-eclipse⟩)`

with the offset keeping the continuum near unity, so residuals, equivalent
widths and significances stay in fractions of the *stellar* continuum and the
whole of §3.3 applies unchanged. The out-of-eclipse search still runs (a very
bright line is found in both, `found_in = both`); both 5σ EW limits are
recorded per exposure and the difference's is the quoted sensitivity.

A **transit-class** exposure gets the analogous out-of-transit minus
in-transit difference. It reaches the same depth, but it is not this channel's
signature and it can never produce a candidate: with no eclipse there is no
vanishing test (`insufficient_phase_coverage`), and a line that changes across
transit more than the continuum does is not a steady source on the planet
(`transit_inconsistent`).

**The drift null.** A difference is only as good as its control. The
out-of-event integrations **before** the event, minus those **after** it,
span the same stretch of visit and the same detector drift with no occultation
in between — so a narrow feature present there is drift, and one present in
the event difference but not there is what the channel is looking for. It is
used twice: as the "consistent with zero" statistic for a difference-found
feature (the in-eclipse residual is not the matched null there, because the
static pattern it carries is present out of eclipse too and would veto a real
line sitting on a pattern bump), and as the veto `present_in_drift_control`.
`cosmic_ray_driven` likewise re-evaluates on whichever spectrum the feature
was found in.

Measured on synthetic stacks carrying the 1% pattern (600 integrations,
600 samples, 2×10⁻³ per-pixel noise):

| Case | out-of-eclipse search | difference search |
|---|---|---|
| 2% vanishing line | **0 features** | 109σ, tier `candidate`, no veto |
| 0.5% vanishing line | 0 features | 26σ, `candidate` |
| constant stellar line, same amplitude | 0 features | 0 features (it cancels) |
| persistence-decaying line | 0 features | found, then `present_in_drift_control` at 15σ |
| no line | 0 features | 0 features |
| 5σ EW limit | 2.8×10⁻⁴ µm | **8.0×10⁻⁶ µm** (35× deeper) |

The faint-line floor is set by the photon noise of the two averages, as it
should be: at 600 integrations the difference triggers down to ~1.5×10⁻³ of
the continuum.

### 3.5 Vetoes (every one has a counter in `summary.json`)

`known_artefact_wavelength` · `recurrent_across_targets` (same wavelength in
≥3 different targets, the `spectra-triage` rule) · `tracks_continuum` ·
`ramp_correlated` · `drop_not_at_eclipse` · `cosmic_ray_single_integration`
(and `single_pixel_spike`, `adjacent_to_gap` at the search stage) ·
`transit_inconsistent` · `insufficient_phase_coverage` · `low_snr` ·
`present_in_drift_control` (§3.4a) · `fdr_not_significant`.

**Tiers:** `none` → `watch` (a clean narrow feature whose phase coverage cannot
test vanishing; kept for the recurrence census) → `interest` → `candidate`.
BH-FDR at α = 0.05 is applied across the population with the **full trial
count** (every scanned resolution element of every exposure), not just the
features that reached the test.

### 3.6 Verdicts

`NO_DATA_REACHED` · `NO_VANISHING_LINE` · `VANISHING_LINE_CANDIDATES_PENDING_VET`
· `DEGRADED_SOURCE` (data reached but the discriminant could not run on any
exposure, or most downloads failed). The workflow refuses a verdict other than
`NO_DATA_REACHED` when zero exposures were analysed.

---

## 4. Contamination ledger

| Confounder | Why it is not the signal | Where it is caught |
|---|---|---|
| Stellar emission line (He I 1.083 µm, Paschen/Brackett, CO bandhead peaks, the "peaks between absorption lines" of an M-dwarf forest) | Follows the star through eclipse: drops by the eclipse depth, not to zero | `tracks_continuum`, `low_snr` on `eclipse_vanish_snr` |
| Flare in an emission line | Not phased to the ephemeris; rises rather than vanishes; chromatic | `drop_not_at_eclipse`, `continuum_correlation`, `transit_inconsistent` |
| Detector settling ramp / persistence decay | A monotonic decay looks like a drop when the eclipse sits late in the window | pre-ingress baseline required; `ramp_correlated` (ramp template must not beat the step model); `drop_not_at_eclipse` |
| Cosmic ray / single-integration event | Present in ≤2 integrations | 5σ clip in the time average; the scatter-based error self-suppresses it; `cosmic_ray_single_integration` |
| Hot / dead pixel, detector gap edge, order overlap, filter edge | Fixed wavelength, no phase | `single_pixel_spike`, `adjacent_to_gap`, `known_artefact_wavelength` (config table), `recurrent_across_targets` |
| Fixed-pattern residuals of the extraction | **Identical in and out of eclipse, so they cancel in the difference the search runs on** (§3.4a); and the same wavelength in unrelated targets | the difference search itself; `recurrent_across_targets` |
| Drift of that pattern across the visit (the difference's own confounder) | Shows the same step between two out-of-eclipse blocks with no occultation in between | the drift null: `present_in_drift_control` (§3.4a) |
| The planet's own dayside spectrum (it vanishes in eclipse too, by construction) | Molecular bands are broad; a resolved feature is not an unresolved line | width guard (1–3 resolution elements); the residual band structure is removed by the local quadratic continuum. **A genuinely unresolved planetary emission line would pass every veto here — the channel cannot separate "a narrow line on the planet" from "a beacon on the planet", and a survivor is a target for higher-resolution follow-up, not a detection** |
| Planet thermal / molecular emission | Vanishes at eclipse — but broad | width guard (1–3 resolution elements); a wide vanishing feature is astrophysics and is counted as `too_wide` |
| Wrong ephemeris | Contacts misplaced; a real step lands "not at eclipse" | propagated σ_t widens the exclusions; stale ephemerides → `phase_unresolved`, never a candidate |
| Eccentric-orbit eclipse timing | ω convention ambiguity | widened σ_t; e > 0.05 without ω → `phase_unresolved` |
| Multi-planet hosts | The wrong planet's ephemeris | every planet is labelled; the one with an event in the window is used and named |
| Red noise in the line series | Inflates a naive propagated significance | error = max(propagated, empirical scatter) |
| Look-elsewhere | 10⁵–10⁶ resolution elements across the archive | BH-FDR with the full trial count |

A `candidate` that survives all of the above is **`PENDING_VET`**: the next
step is an independent extraction (a second pipeline, e.g. the stage-3 vs
stage-2 product, or a re-reduction) and a second eclipse of the same planet.
Nothing in this channel is a detection on its own.

---

## 5. Coverage and sensitivity — what a null does not mean

* **Resolution.** JWST resolving powers are R ≈ 2700 (NIRSpec H gratings),
  ~1000 (M), ~1600 (NIRCam grism), ~700 (SOSS order 1), ~100 (PRISM, MIRI LRS).
  A "narrow" feature is therefore **≥ 110 km/s** wide at best, ~430 km/s at
  SOSS and ~3000 km/s at PRISM/LRS. A true laser (Δλ/λ ~ 10⁻⁸) is always
  unresolved; what is measured is its **equivalent width**. Every exposure
  records **three** numbers in `summary.json: sensitivity` —
  `ew_5sigma_limit_out_um` (the out-of-eclipse spectrum, held at ~1% of the
  continuum by the static pixel pattern, §3.4a), `ew_5sigma_limit_diff_um`
  (the difference spectrum, photon limited), and `ew_5sigma_limit_um`, which
  is the difference's wherever the difference ran. **Quote the difference
  limit**: the other one is the pattern, not the data.
* **What the limit means.** An equivalent width is a fraction of the *star's*
  continuum. Each eclipse-class exposure also measures its own broad-band
  event depth (`event_depth`, the planet's day-side flux over the star's,
  from the same detrended continuum light curve the verification uses), so
  `line_contrast_5sigma` (= 5 × the difference spectrum's noise, the faintest
  line peak the exposure could have shown, as a fraction of the stellar
  continuum) divided by that depth gives
  **`beacon_fraction_of_event_flux_5sigma`**: the faintest beacon detectable
  *as a fraction of the planet's own broad-band emission*, in one resolution
  element. That is the number to quote about a *planet*, and it needs neither
  a distance nor a stellar model. On the synthetic 600-integration stack it is
  ~0.10 for a 1% eclipse and ~0.48 for a 0.2% one.
* **Phase.** Only eclipse-class exposures (≥8 in-eclipse and ≥16 out-of-eclipse
  integrations *and* a pre-ingress baseline) test vanishing, and only they get
  the eclipse difference. A transit-class exposure gets the transit difference
  and reaches the same depth, but produces `watch` at most (§3.4a);
  `phase_unresolved` exposures get neither and are limited by the pattern —
  they contribute constant-line entries and recurrence statistics, nothing
  more.
* **Duty cycle.** A beacon that is off during the observation, or pointed
  elsewhere, is invisible. A beacon brighter than the star's local continuum by
  less than ~5 noise units per resolution element is invisible.
* **Targets.** Only hosts with public JWST time series; proprietary products
  are counted, not analysed.

`NO_VANISHING_LINE` is a **count** over the analysed exposures at the quoted
sensitivities. It is not an occurrence limit and is not written up
(`CLAUDE.md`: never write up a null result). A clean null changes the
question — first candidates for a changed question are phase *curves* (two
eclipses, longer baselines) and the stage-3 products.

---

## 6. Files

```
src/seti/lantern/{__init__,acquire,phase,line,synth,run}.py
config/lantern.yaml           thresholds, phase windows, per-instrument artefact table
tests/test_lantern.py         offline battery (CI gate)
tests/test_lantern_reader.py  the table-per-segment x1dints layout
tests/test_lantern_difference.py  the out-minus-in difference search and its drift null
.github/workflows/lantern.yml inventory -> sharded screen -> assess (+ lit)
scripts/lanternlit_fetch.py   prior-art sweep -> results/lanternlit/
results/lantern/              summary.json, candidates.json, exposures.json,
                              inventory.json, probe.json, selftest.json (committed);
                              obs/<host>/<exposure>.json checkpoints (run artifact
                              only -- re-assessable with reduce_only_run_id)
```

Entry point: `python -m seti.lantern.run {probe|inventory|screen|assess|selftest}`,
or `seti.lantern.run.register(sub)` for the main CLI.
