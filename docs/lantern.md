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
   T₀, T₁₄, eccentricity, ω, Rp/Rs and their errors.
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

On the time-averaged, continuum-normalised out-of-eclipse spectrum:

* continuum = **notched local quadratic** (window 31 samples, ±hole excluded,
  >4σ excursions masked on a second pass) — a running median fails on a
  steep continuum, a local line leaves band curvature in the residual;
* noise = block-wise MAD of the residual, floored by the propagated error;
* a feature is an interior (≥8 samples from either end), **bounded** local
  maximum at ≥6σ, not adjacent to a NaN gap, with a **matched-template FWHM
  of 1–3 resolution elements** (a single-sample spike is a hot pixel or
  cosmic ray, wider than 3 elements is a band). These are the guards of
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
| `in_eclipse_sigma` | in-eclipse line flux over its error | \|·\| ≤ 2 (consistent with zero) |
| `eclipse_vanish_snr` | (out − in) / joint error | ≥ 5 (≥ 3 for `interest`) |
| `line_fractional_drop` vs `continuum_fractional_drop` | did the line drop like the star (~10⁻³) or like the planet (~1)? | line ≠ continuum, consistent with 1 |
| `continuum_correlation` | line vs continuum over **out-of-eclipse** integrations (a planet line is constant there; any correlation is systematics) | \|r\| < 0.5 |
| `ramp_correlation`, χ²_ramp vs χ²_step | best exponential settling template vs the eclipse-step model | step fits better |
| `free_step_offset` | where a free two-level model puts the drop, vs the predicted ingress | within 2 ingress durations + σ_t |
| `transit_constancy`, `transit_excess_sigma` | line change in transit vs the continuum's own change | line changes no more than the continuum |

Errors are the larger of the propagated and the empirical (scatter) error, so
a red-noise-dominated series cannot manufacture significance.

### 3.5 Vetoes (every one has a counter in `summary.json`)

`known_artefact_wavelength` · `recurrent_across_targets` (same wavelength in
≥3 different targets, the `spectra-triage` rule) · `tracks_continuum` ·
`ramp_correlated` · `drop_not_at_eclipse` · `cosmic_ray_single_integration`
(and `single_pixel_spike`, `adjacent_to_gap` at the search stage) ·
`transit_inconsistent` · `insufficient_phase_coverage` · `low_snr` ·
`fdr_not_significant`.

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
| Fixed-pattern residuals of the extraction | Same wavelength in unrelated targets | `recurrent_across_targets` |
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
  unresolved; what is measured is its **equivalent width**. Every exposure's
  5σ EW limit (`ew_5sigma_limit_um`, one resolution element) and velocity
  width are recorded in `summary.json: sensitivity`.
* **Phase.** Only eclipse-class exposures (≥8 in-eclipse and ≥16 out-of-eclipse
  integrations *and* a pre-ingress baseline) test vanishing. Transit-only and
  `phase_unresolved` exposures contribute constant-line entries (`watch`) and
  recurrence statistics, nothing more.
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
.github/workflows/lantern.yml inventory -> sharded screen -> assess (+ lit)
scripts/lanternlit_fetch.py   prior-art sweep -> results/lanternlit/
results/lantern/              summary.json, candidates.json, inventory.json,
                              probe.json, selftest.json, obs/<host>/<exposure>.json
```

Entry point: `python -m seti.lantern.run {probe|inventory|screen|assess|selftest}`,
or `seti.lantern.run.register(sub)` for the main CLI.
