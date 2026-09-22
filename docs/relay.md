# RELAY — intercepting node-to-node beams by geometry (S60)

**Claim.** A persisting machine network talks with beams, not broadcasts. Earth
never receives a beam aimed at it; it receives the beam aimed at *someone
else* when it happens to sit inside the transmitter's cone. With a transmitter
T, a receiver R and a full beam width θ, that requires the angle at T between
T→R and T→Earth to be ≤ θ/2. Two geometries satisfy it:

* **spillover** — T behind R on nearly the same line of sight; the beam passes
  R and reaches Earth, weaker than R receives it by
  (|T−R|/|T|)² < 1;
* **between** — R near T's antipode; the beam passes Earth on its way to R,
  *stronger* than R receives it by (|T−R|/|T|)² > 1.

Gaia DR3 gives every pair within 100 pc, its exact cone angle, and its 6D
kinematics; the Breakthrough Listen open-data target list says which of those
transmitters a telescope has already pointed at; the published narrowband hit
catalogues are re-cut on that geometry against a per-pointing Doppler-drift
prior. The prior art is single-target only (Tusay+2022, the SGL antipode of
α Cen; Hort+2024; Tusay+2024, TRAPPIST-1 planet–planet occultations) and the
network papers are design studies (Hippke; arXiv:2204.08296; Gertz; Forgan
2019); `results/necrofrontier_lit/` group g10 found no star-pair spillover
selection at catalogue scale.

Status 2026-09-22: built and offline-tested; first runner dispatch pending (see
`STATUS.md`).

## 1. Layout

| Piece | Path |
|---|---|
| physics (pure, offline) | `src/seti/relay/geometry.py` |
| archives (injectable transports) | `src/seti/relay/acquire.py` |
| stages, verdicts, files | `src/seti/relay/run.py` |
| thresholds, beams, sites, seeds | `config/relay.yaml` |
| offline suite (CI gate) | `tests/test_relay.py` |
| workflow | `.github/workflows/relay.yml` |
| results | `results/relay/` |

CLI: `python -m seti.cli relay -- --stage {probe,targets,geometry,recut,assess,all}`
(or `python -m seti.relay.run`).

## 2. The geometry, exactly

Earth is the origin; a star is `d·u` with d = 1000/ϖ pc. For a directed pair
the **transmitter angle** is

    cos α = ((R − T) · (−T)) / (|R − T| |T|)

and the pair qualifies at beam θ when α ≤ θ/2. The brief's sky conditions,

    spillover:  δ  ≤ (θ/2)(d_T − d_R)/d_R
    between:    δ′ ≤ (θ/2)(d_T + d_R)/d_R        (δ′ from T's antipode)

are the small-angle limits of the exact bounds, with k = sin(θ/2):

    spillover:  δ  ≤ arcsin(k d_T/d_R) − θ/2
    between:    δ′ ≤ arcsin(k d_T/d_R) + θ/2

which **saturate to the whole sky** when d_R ≤ d_T sin(θ/2): a receiver within
a few parsecs qualifies at *any* separation, because α peaks at
arcsin(d_R/d_T). The offline brute-force cross-check found exactly that pair
the small-angle search missed, so the search radii are the exact bounds at the
widest allowed d_T and every candidate is decided on α. Pairs closer than
0.05 pc (~10⁴ AU) are dropped as bound binaries.

**Expected counts** for N stars uniformly filling the sphere (integrating the
cap fraction δ²/4 over both distances):

    spillover: θ² N² / 32        between: 0.3156 θ² N²

The between geometry yields ten times the spillover geometry, and the yield is
dominated by the *nearest receivers* (the integrand per unit d_R is flat while
the number of receivers per unit d_R grows as d_R²), so a realised count is far
from Poisson about its expectation — the 30,000-star synthetic sample in the
test suite sits at 0.8–1.0 of the coefficient. For the 100 pc Gaia sample
(N ≈ 2.7–3.3 × 10⁵ after quality cuts) this gives:

| beam | θ | spillover | between |
|---|---|---|---|
| 10 m optical at 1 µm | 0.025″ | 3 × 10⁻⁵ | 3 × 10⁻⁴ |
| 100 m at 1.42 GHz | 8.9′ | ~2 × 10⁴ | ~2 × 10⁵ |
| 10 m at 8 GHz | 0.26° | ~2 × 10⁵ | ~2 × 10⁶ |
| 10 m at 1.42 GHz | 1.47° | ~2 × 10⁶ | ~2 × 10⁷ |
| over-filled 5° | 5° | ~3 × 10⁷ | ~3 × 10⁸ |

A large-aperture optical network leaves Earth outside every beam; the
interceptable regime is radio with modest apertures or deliberately
over-filled beams, exactly as the brief said. Counts are computed exactly for
every beam; rows are stored only where they are needed (§4).

## 3. The Doppler prior — what Gaia can and cannot give

A relay transmitter that de-drifts its carrier so R receives a constant
frequency emits f₀(1 + v_TR/c). Earth sees f₀(1 + (v_TR − v_TE)/c) and a drift

    ḟ = (f₀/c)(a_TR − a_TE)

with a_TR the T–R range acceleration and a_TE the T–Earth one. **The
kinematic part is unmeasurably small.** For straight-line relative motion the
range acceleration is v_⊥²/r: 30 km/s across 30 pc is 10⁻⁹ m/s², a drift of
5 × 10⁻⁹ Hz/s at L band, four orders below the 0.0093 Hz/s resolution of a
2.79 Hz × 300 s search. The channel computes it for every kept pair (both
range rates, both range accelerations, the offset v_TR − v_TE that shifts any
"magic" rest frequency) and reports the p99 of |ḟ_kin| in `geometry.json` so
the statement is measured, not asserted. This corrects the brief: the pair's
*acceleration* is not a usable drift prior. What is:

1. **The Earth term.** a_TE contains the observatory's own acceleration
   projected on the line of sight — rotation (≤ 0.034 m/s², up to ±0.16 Hz/s
   at L band, sign set by the hour angle: at transit the site accelerates
   *away* from the source and the drift is negative) and orbit (≤ 0.006 m/s²).
   Every published pipeline searches ±4 Hz/s and none corrects the topocentric
   drift, so a de-drifted relay shows a *specific non-zero* drift at a known
   epoch, while terrestrial interference (co-rotating) shows zero. The channel
   computes it from the file's MJD (read from the BL file name
   `..._57650_67573_...` when the API gives no time), the site and the source.
2. **The small-angle leak.** T de-drifts its own platform (rotation, orbit) in
   the T→R direction; toward Earth the residual is a_local·(ê_TE − ê_TR)
   ≤ α a_local. With a_local ≤ 1 m/s² (a habitable-zone planet of an M dwarf
   is 0.7) and α ≤ 4.5′ this is ≤ 0.006 Hz/s at L band — the *tight* half-width,
   set per pair by Gaia's α.
3. **If T also pre-compensates R's platform** the whole of R's local
   acceleration reaches Earth and the *loose* half-width is ±4.7 Hz/s: the
   entire search range, no discrimination. Both widths are reported; the
   candidate test uses the tight one and says so.

So the prior per pointing is `centre = (f/c)(a_kin + a_Earth)`,
`half-width = √(σ² + [(f/c) α a_max]²)`. When the epoch is unknown the Earth
term's bound is added to the width and the centre carries the kinematic part
only (`earth_term_known = false` on the row).

## 4. Stages and files

**probe** — `api/list-targets`, `api/list-telescopes`, `api/list-file-types`
and a sample `api/query-files` on the BL open-data API (routes from the
archive's own README; response shapes and the first record's keys recorded
verbatim); a COUNT on `gaiadr3.gaia_source`; a SIMBAD `ident` resolution;
discovery of every VizieR table under the seed ids in `config/relay.yaml`
(Enriquez+2017, Price+2020, Sheikh+2020, Margot+2021/2023, Traas+2021,
Gajjar+2021, Franz+2022, Pinchuk+2022, Choza+2024, Isaacson+2017) plus a
keyword search. A table is a **hit** table only if it exposes a frequency *and*
a drift rate; one with names/positions but no frequency is a **target** table.
→ `probe.json`.

**targets** — the BL target list, resolved to Gaia DR3 ids through SIMBAD's
`ident` table (`Gaia DR3 …` aliases), positions as the fallback; if the API
is down, the VizieR target tables are the second route. → `targets.json`,
`targets.csv`.

**geometry** — the 100 pc sample (ESA TAP in parallax shells; VizieR `I/355`
and the GCNS `J/A+A/649/A6` as the non-ESA routes; RV zero-point per
Katz+2023; RVs worse than 5 km/s or from hot templates dropped), then for
every beam the exact directed-pair enumeration. Counts are exact for the whole
sample; rows are kept for every pair whose transmitter is a BL target or an
in-beam neighbour of one (within 8′), or for every pair of a beam narrow
enough to commit whole. Kinematics attached to every kept row. →
`geometry.json`, `gaia_sample.json`, `pairs_full_<beam>.csv.gz`,
`pairs_targets.csv.gz`; the parquet intermediates travel in the artifact.

**recut** — per resolved target: pair counts as transmitter per beam and
geometry, in-beam neighbour pairs; then the BL files per target (telescope,
band, epoch) grouped into pointings, and for each pointing × beam the pair
count and the drift window (centre, tight and loose half-widths, the smallest
α, the largest flux ratio). The physical requirement is that the
*transmitter* is in the telescope's primary beam — pointing at the near star
of a spillover pair catches the far one only if δ < HPBW/2 — so a pointing
counts pairs whose T is the target *or* a Gaia star inside its HPBW. →
`recut.json`, `recut_pointings.csv`, `recut_targets.csv`, `bl_files.csv.gz`.

**assess** — the hit tables fetched, standardised (MHz, Hz/s), matched to
sample stars by name (through the resolved target list, then SIMBAD), then
by position within the beam. Per hit per beam: on a pair line or not; the
window from the hit's own frequency, epoch and telescope; `drift_match` if
inside `match_window_sigma` tight half-widths; `candidate` if also not RFI.
RFI rules: |drift| below one resolution element (co-rotating source); a
frequency recurring on ≥ 3 unrelated sightlines within 1 kHz (instrumental).
The chance expectation per beam is the fraction of *off-pair-line* hits whose
drift falls inside a window of the same width at the same centre, times the
number of pair-line hits — the trials are the pair-line hits. →
`hits.json`, `hits_crossmatch.csv`, `summary.json`.

Verdicts: `GEOMETRY_COMPUTED` (`_ON_PARTIAL_SAMPLE` when a shell was lost),
`TARGETS_RESOLVED (n/m via route)`, `RECUT_COMPLETE` (or
`RECUT_COMPLETE_GEOMETRY_ONLY (NO_DATA_REACHED for BL files …)`),
`NO_HIT_CATALOGUE_REACHED`, `NO_PAIRLINE_HIT`, `PAIRLINE_HITS_OFF_PRIOR`,
`PAIRLINE_DRIFT_MATCH (n; see n_expected_by_chance)`. `NO_DATA_REACHED`
anywhere is about the run, never the sky.

## 5. Contamination model

* **Terrestrial interference** is the dominant confounder of every narrowband
  hit list; its signature is zero topocentric drift, which the prior places
  *outside* the window whenever the Earth term is resolved. Non-zero-drift
  RFI exists (moving transmitters, oscillator drift) and is not excluded by
  this test; a candidate stays a candidate pending the on/off cadence check
  the original pipelines applied, which is recorded as a systematic not
  excluded.
* **Trials.** 10⁴–10⁸ pairs per beam and every pair-line hit is a test; the
  chance expectation is computed from the off-pair-line drift distribution
  and printed beside every count. A single match at a 5° beam, where the tight
  window is already ±0.66 Hz/s, is not a detection.
* **Unknown epochs** widen the window by the Earth term's bound; such matches
  are flagged and weigh less.
* **Gaia RV completeness** (~40 % of the 100 pc sample) limits only the
  supplementary rest-frequency offset test; the geometry and the Earth term
  need no RV.
* **Bound binaries** are excluded by the 0.05 pc separation cut; a wider
  common-proper-motion pair is a legitimate field pair.
* **The θ² N² honesty.** The counts scale as θ²; the fitted slope of
  log N vs log θ across the grid is written to `geometry.json` with the
  analytic coefficient beside every measured count.

## 6. Offline tests

`tests/test_relay.py` (25 tests, no socket): an injected spillover pair is
found at the beam that admits it and not at a narrower one, with α at the
closed form and the flux ratio at (40/60)²; the between pair is counted
separately with flux ratio > 1; the tree search equals brute force; counts
scale as θ² and track the analytic coefficients (Monte Carlo of the
coefficients themselves); the range acceleration equals a finite difference;
the Earth term has the closed-form sign and magnitude at transit and vanishes
six hours later; the tight window is the small-angle leak and the loose one
spans the search range; BL file records are normalised and MJDs read from
file names; every archive empty → `NO_DATA_REACHED`; and the full pipeline
through fake transports finds the injected hit at the prior while a zero-drift
hit, a recurrent frequency, a hit on a star on no pair line and a hit off the
window are each rejected by the named rule.

## 7. What the runner has to settle (verify list)

* the BL API's JSON shapes and the file record's key names (recorded in
  `probe.json`; `normalise_bl_file` resolves them by regex);
* whether `query-files` needs a non-empty `target` for a MeerKAT/positional
  query;
* which seed ids exist on VizieR and which of their tables carry a drift
  column (Margot's UCLA tables are the most likely to);
* the ESA TAP's behaviour on the 90–100 pc shell (~90 k rows).
