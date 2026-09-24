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

Status 2026-09-22: **measured on the runner.** The geometry is computed for the
whole reachable 100 pc Gaia DR3 sample (264,973 stars, 5/5 beams, counts
exact); see §8 for the numbers and for the one thing the sky did not supply.

## 1. Layout

| Piece | Path |
|---|---|
| physics (pure, offline) | `src/seti/relay/geometry.py` |
| archives (injectable transports) | `src/seti/relay/acquire.py` |
| published event tables from e-prints | `src/seti/relay/papers.py` |
| stages, verdicts, files | `src/seti/relay/run.py` |
| thresholds, beams, sites, seeds | `config/relay.yaml` |
| offline suite (CI gate) | `tests/test_relay.py`, `tests/test_relay_papers.py` |
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
| 100 m at 1.42 GHz | 8.9′ | ~1.5 × 10⁴ | ~1.5 × 10⁵ |
| 10 m at 8 GHz | 0.26° | ~4.6 × 10⁴ | ~4.6 × 10⁵ |
| 10 m at 1.42 GHz | 1.47° | ~1.5 × 10⁶ | ~1.5 × 10⁷ |
| over-filled 5° | 5° | ~1.7 × 10⁷ | ~1.7 × 10⁸ |

(at the N = 264,973 the runner actually reached; §8.1 has the measured counts
beside these.)

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

**The supplementary rest-frequency test.** The pair's *velocity* difference,
unlike its acceleration, is large and measured: a relay de-drifted so R sees a
chosen rest line puts that line, for Earth, at `f_rest (1 + (v_TR − v_TE)/c)`.
`assess` tests every pair-line hit against HI 1420.4058, OH 1665.4018 and
OH 1667.3590 shifted by that pair's own `dv_offset_kms`, with a ±5 km/s RV
tolerance (≈ ±24 kHz at HI). It is a second, independent discriminant, and it
is priced as such: a pointing whose pairs lack either radial velocity is
recorded as **not tested** rather than as a non-match, so the trials count
(`n_magic_line_tests`) counts only real tests, and `n_magic_expected_by_chance`
= tests × window / searched band (~7 × 10⁻⁴ per 100 tests over an 800 MHz band)
is printed beside every match. Only ~43 % of the sample has the RVs this needs.

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

Beams run **narrowest first** under one wall clock (`geometry.budget_s`), so
the diffraction-limited and 9′ beams always finish and the 5° beam — ~3 × 10⁸
directed pairs on the full sample — can never eat the run and starve recut and
assess. A beam the clock cuts short is `PARTIAL_COUNTS` with
`n_receivers_done / n_receivers`; one never started is `NOT_COMPUTED` with the
reason. Neither count is compared with the analytic expectation, neither
enters the θ² slope fit, and the verdict carries `BEAMS_INCOMPLETE`. The recut
stage records each beam's `geometry_status`, so a zero pair count on an
unfinished beam reads as the absence of a *search*, not of a pair.

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
The chance expectation per beam (`src/seti/relay/chance.py`, since
2026-09-23) is Σᵢ pᵢ over the pair-line hits, with pᵢ the fraction of every
*other* valid hit whose drift falls inside hit i's own window; the tail
P(N ≥ observed) is the Poisson-binomial tail. (The earlier estimator used only
the off-pair-line hits on sample stars — two hits in run 35745111146 — and was
uninformative.) The verdict is `PAIRLINE_DRIFT_MATCH` only when some beam's
p < 0.01, otherwise `PAIRLINE_MATCHES_AT_CHANCE`. Hygiene before any count:
rows read from a non-frequency column ("Frequency rank"), injection-table rows
and duplicate rows are invalid; RFI now also includes the known L-band
allocations (`assess.known_rfi_bands_mhz`) and recurrence on ≥ 3 sightlines
within 500 kHz. → `hits.json`, `hits_crossmatch.csv`, `summary.json`.

**targetlist** (not in `all`; needs the geometry stage's parquet sample in the
same job) — see §9.2. → `targetlist.json`, `targetlist.csv.gz`, and a
`targetlist` pointer in `summary.json` carrying its own timestamp.

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

**Run the suite under both pandas majors before dispatching.** The sandbox venv
holds pandas 2.3.3 and the runner's `pip install -e ".[dev]"` fetches 3.0.6;
CRADLE lost a whole dispatch to an API pandas 3 removed, dying on its own
offline gate before one archive call (`docs/channel-brief.md` §0 item 5). All
50 RELAY tests were run under 3.0.6 as well as 2.3.3 on 2026-09-22, without
touching the shared venv:

    pip install --target <dir> --no-deps "pandas>=3"
    PYTHONPATH=<dir>:src pytest tests/test_relay*.py -q

RELAY's only `errors="ignore"` is on `DataFrame.drop`, where the argument
survives; the numeric conversions all use `errors="coerce"`.

`tests/test_relay_magic.py` (7 tests) pins the rest-frequency test: the window
is the offset plus the RV tolerance and moves with the pair's velocity; a
pointing without radial velocities is **not tested** rather than counted as a
miss; a hit on the shifted line matches while the *unshifted* line at 42 km/s
does not; the OH lines are told apart; and the chance rate is the window over
the searched band and is small but not negligible.

`tests/test_relay_papers.py` (13 tests, no socket) covers the e-print route on
fixture bytes: a deluxetable with `$-$` signs and `\tablenotemark` glued to a
number yields the right numbers; a target list with coordinates and no drift
column is **not** a hit table; a GHz header and a plain `tabular` are read; an
AAS machine-readable table parses by byte range; an `arxiv_id` that resolves to
another paper's title is refused with both strings and falls back to a search;
a failed e-print fetch is a recorded failure on both mirrors, not zero hits;
and the discovery sweep does not re-fetch a paper the seed list already named.

## 7. What the runner has to settle (verify list)

* the BL API's JSON shapes and the file record's key names (recorded in
  `probe.json`; `normalise_bl_file` resolves them by regex);
* whether `query-files` needs a non-empty `target` for a MeerKAT/positional
  query;
* which seed ids exist on VizieR and which of their tables carry a drift
  column (Margot's UCLA tables are the most likely to) — **settled, §8.2: none
  of them do**;
* the ESA TAP's behaviour on the 90–100 pc shell (~90 k rows) — **settled**:
  the widest shell returned 70,506 rows, no truncation, no error.

## 8. What the runner measured

### 8.1 The geometry (run 35740854882, branch `claude/goap-relay-geom`)

Sample: ESA Gaia DR3 TAP, seven parallax shells, ϖ ≥ 10 mas, ϖ/σ_ϖ > 10,
RUWE < 1.4 — **264,973 stars**, every shell `OK` and untruncated, 114,148 with
a usable RV (43 %). That is the *clean* subset of the 331,312-star GCNS volume;
the astrometric-quality cuts, not the archive, set the difference, and the
counts below scale as N² if one prefers the looser sample.

Directed pairs, counts **exact over the whole sample** (no row cap in the
counting path), against the uniform-density expectation of §2:

| beam | θ | spillover | analytic | between | analytic | meas/analytic (spill) |
|---|---|---|---|---|---|---|
| 10 m optical, 1 µm | 0.0252″ | **0** | 3.27 × 10⁻⁵ | **0** | 3.30 × 10⁻⁴ | — |
| 100 m, 1.42 GHz | 8.85′ | **18,381** | 1.456 × 10⁴ | **145,734** | 1.470 × 10⁵ | 1.26 |
| 10 m, 8 GHz | 15.7′ | **54,147** | 4.586 × 10⁴ | **461,277** | 4.632 × 10⁵ | 1.18 |
| 10 m, 1.42 GHz | 1.48° | **1,586,109** | 1.456 × 10⁶ | **14,613,310** | 1.470 × 10⁷ | 1.09 |
| over-filled 5° | 5° | **17,841,566** | 1.671 × 10⁷ | **167,627,412** | 1.688 × 10⁸ | 1.07 |

Fitted slopes of log N vs log θ over the four computed beams: **1.954** for
spillover and **2.000** for between. The θ² law is measured, not assumed, and
the real sky sits 7–26 % *above* the uniform-density coefficient — the excess is
largest at the narrowest beam, which is what local clustering does to a pair
count whose yield is dominated by the nearest receivers.

**The honest headline is the yield itself.** A diffraction-limited 10-m optical
link at 1 µm gives **zero** qualifying pairs in this sample and 3 × 10⁻⁵
expected: a large-aperture optical network leaves Earth outside every beam, and
no amount of searching changes that. The interceptable regime is radio with
modest apertures or deliberately over-filled beams.

There the interception is **not** leakage, and the measurement says so rather
than the argument. Over the kept pairs of every radio beam the *median* flux
ratio at Earth versus at the intended receiver — `(|T-R|/|T|)^2`, below 1 for
spillover and above 1 for the between geometry — is

| beam | spillover | between |
|---|---|---|
| 100 m, 1.42 GHz | **0.547** | **3.89** |
| 10 m, 8 GHz | 0.517 | 3.82 |
| 10 m, 1.42 GHz | 0.600 | 3.86 |
| over-filled 5° | 0.624 | 3.85 |

— Earth gets about **55 % of what the receiver gets** in the spillover geometry
(2.6 dB down, not a sidelobe) and about **4× more than the receiver** in the
near-antipodal one, because Earth is then the nearer point on the same beam.
The median transmitter angle sits at 0.67–0.71 of the cone half-width, i.e. the
qualifying pairs fill the cone rather than hugging its axis.

The price is stated in the same breath — 1.8 × 10⁴ to 1.8 × 10⁷ qualifying
pairs per beam is also 1.8 × 10⁴ to 1.8 × 10⁷ trials, which is why every count
in `hits.json` is printed beside its `n_expected_by_chance` and `n_trials`.

**And the kinematic drift prior is dead, measured.** Over the same pairs the
median |a_kin| is **5.3 × 10⁻¹⁰ m/s²** and the p99 of |ḟ_kin| at 1.42 GHz is
**5–7 × 10⁻⁸ Hz/s** — five to six orders below the 0.0093 Hz/s resolution of a
2.79 Hz × 300 s turboSETI product. The brief's "relative radial acceleration
gives a Doppler-drift prior" is therefore *false as stated*, and §3 already
replaced it with the two terms that do carry information: the unremoved
topocentric Earth term (±0.16 Hz/s at L band, sign set by hour angle) and the
small-angle leak of the transmitter's own platform acceleration (≤ 0.006 Hz/s),
plus the rest-frequency offset test, which uses the pair's *velocity* — a
quantity Gaia measures well — instead of its acceleration.

The near-antipodal "Earth between the nodes" geometry is counted separately
throughout and is the larger population by ~9–10× at every beam, exactly the
0.3156/(1/32) ratio of §2. It is also the geometry whose pairs are *brighter*
at Earth than at the intended receiver, and the one a single-target search
(Tusay+2022 at the α Cen SGL antipode) has only ever sampled one line of.

### 8.2 The hit catalogues: VizieR has the targets, not the events

Run 35738937745 probed all 15 seed catalogue ids plus six keyword sweeps over
VizieR's `TAP_SCHEMA`: **73 tables discovered, 0 of kind `hits`** — that is, not
one exposed both a frequency and a drift-rate column. Margot+2021
(`J/AJ/161/55`) and Choza+2024 (`J/AJ/167/103`) deposit *target* tables;
Enriquez+2017, Price+2020, Sheikh+2020, Traas+2021, Gajjar+2021, Franz+2022,
Margot+2023, Ma+2023, Sheikh+2021 and Tusay+2022/2024 returned zero rows under
the asserted ids. The community deposits the observed-star lists and keeps the
event lists in the papers.

So the events are read from the papers: `src/seti/relay/papers.py` fetches the
arXiv e-print source and parses `deluxetable`/`longtable`/`tabular`
environments and any AAS machine-readable table shipped in the same tarball.
Three rules keep that safe to run a search on:

* an `arxiv_id` in the config is a **hint**, never an identification — the Atom
  title is fetched and must contain every phrase the seed declares, and a
  mismatch is recorded with both strings and falls back to a title search;
* **every hit row carries its provenance**: arXiv id, resolved title, the file
  inside the tarball, the table's caption, and the verbatim header text of the
  frequency and drift columns it was read from, so any candidate is traceable
  to a line of LaTeX;
* a table is a **hit table only if it proves it** — a frequency column and a
  drift column both resolve and at least one row parses as numbers. Frequency
  units come from the header (`(MHz)`, `(GHz)`); a table with no unit is scaled
  by magnitude and flagged `freq_unit_assumed`.

A discovery sweep over the arXiv full-text index (`abs:"drift rate" AND
abs:technosignature` and two siblings) covers the papers no seed list
remembered, on identical rules. Every paper that fails to resolve, fails to
download or yields no hit table is written to `hits.json` with the reason, and
the verdict says so: an empty harvest is `NO_HIT_CATALOGUE_REACHED (…)`, which
is a statement about what was reachable and never about the sky.

## 9. The 22 "drift matches" of run 35745111146, priced — and the question that replaced them

### 9.1 What matched, and why it means nothing

Run 35745111146 (dispatched 2026-09-22T15:07Z, summary 16:58Z) reported
`PAIRLINE_DRIFT_MATCH (22 hit-beam matches)`. The committed files are
self-consistent — summary 3 s after `hits.json`, 135 hits in all three files,
per-beam counts equal to the CSV's flags, 22 = 1 + 2 + 8 + 11 — but the count
describes almost nothing. Recomputed offline from those committed files
(`python -m seti.relay.chance --source-run 35745111146`, written to
`results/relay/chance_run35745111146.json` and as `chance_offline` in that
run's `summary.json`):

* **22 matches are 11 hits**, the same hit counted at up to four beams.
* **3 of the 11 are not frequencies.** HIP 13402 at "989 MHz" and HIP 62207
  at "980/998 MHz" were read from the *Frequency rank* column of
  arXiv:2505.03927's candidate table. They account for 8 of the 22. The
  parser now refuses rank/index columns, refuses injection-recovery tables
  (arXiv:2011.05265's two injected signals had also been read as hits) and
  reads an e-print once even when two seeds resolve to it (42 duplicate rows
  from arXiv:1901.04057).
* **The other 8 are Enriquez+2017's own "most significant events"**, which that
  paper did not claim as signals. Six lie inside the GPS L3 main lobe
  (1379.28–1384.21 MHz; five of them — plus HIP 4436 and HIP 82860, which did
  not match the sample — within 1380.88–1381.21 MHz on seven unrelated stars:
  recurrence across sightlines, i.e. terrestrial) and two in the Inmarsat/MSS
  downlink (1522.18, 1528.46 MHz). The run's 1 kHz recurrence rule missed the
  family because a drifting emitter seen on different days does not repeat to
  1 kHz.
* **The count is the chance count.** The prior's width is the pipeline
  resolution plus the Earth term (the kinematic part is 10⁻⁸ Hz/s, §8.1), and
  published hits cluster at small |drift|, so the 3σ window at 1.48° contains
  ~64 % of all hits' drifts and at 5° ~94 %. Against the hits' own drift
  distribution:

  | beam | trials | matches | expected | p(≥) |
  |---|---|---|---|---|
  | 100 m L | 3 | 1 | 1.11 | 0.90 |
  | 10 m 8 GHz | 5 | 2 | 1.55 | 0.50 |
  | 10 m L (1.48°) | 12 | 8 | 7.73 | 0.57 |
  | 5° | 12 | 11 | 11.25 | 0.83 |
  | **total** | | **22** | **21.6** | |

  After hygiene: 14 observed vs 14.6 expected. After hygiene and the band /
  wide-recurrence RFI rules: **0 candidates**.
* **The geometry selects nothing at the wide beams either**: 99.7 % of the
  in-sample BL targets are the transmitter of some qualifying pair at 1.48°
  and 5° (29 % at 100 m, 62 % at 15.7′). "On a pair line" there is a property
  of every star.

Verdict: **chance-level, and every matched hit is traced to a systematic**
(a parse error or a named terrestrial allocation). RELAY's hit re-cut is empty
by construction — the Gaia kinematic prior is too weak to narrow the drift and
the wide beams put every star on a pair line — so no future hit list will make
it informative. That question is retired; the channel keeps the chance model
so any rerun says so automatically.

### 9.2 The question that is not empty: whose *short* links cross Earth?

A network links nearest neighbours (link cost ∝ distance²). For a link only a
few parsecs long, Earth — tens of parsecs away — is inside the beam only if R
sits almost exactly on T's sightline to Earth; a random link does so with
probability (1 − cos θ/2)/2 = 4 × 10⁻⁷ at 100 m L band. Over the ~1.3 × 10⁶
five-nearest-neighbour links of the 100 pc sample that is ~0.5 links expected
at the narrow beam and ~600 at 5°. So the list of stars whose ordinary
neighbour traffic would cross Earth is short, exact and specific — and set
against BL's pointings it is a list of targets nobody has chosen for this
reason.

`stage_targetlist` (`src/seti/relay/targetlist.py`): 5 nearest neighbours
within 10 pc (≥ 0.05 pc) per star; exact α; P(α ≤ θ/2) per beam from 400
parallax Monte-Carlo draws (the T/R ordering along the sightline is exactly
what a parallax error flips); pairs with |Δϖ| < 3σ (radial separation not
measured) and comoving pairs (Δv_tan ≤ 3 km/s at ≤ 1 pc projected: likely
bound, true depth far below the noise) are counted and never ranked; "observed"
= within half the GBT L-band beam of a resolved BL target (the unresolved BL
names are counted, not checked). Ranked by the narrowest beam at which
P ≥ 0.5, then P, then distance; the isotropic expectation is printed beside
every count. It is a target list, not evidence: a count at the isotropic
expectation is what geometry predicts. Dispatched on `claude/handoff-relay`
with `stage=probe,targets,geometry,recut,assess,targetlist`; the numbers land
in `targetlist.json`.

### 9.3 What the runner measured (runs 35992222801 → 35993…, re-reduced on run 35745111146's intermediates)

Run 35860719426 died at its 300-min clock: BL's open-data API returned 503 (the
targets stage fell back to 30,102 VizieR names and spent ~3 h in SIMBAD) and
every ESA Gaia shell timed out. The workflow now takes `reuse_run_id`, which
restores only a prior run's parquet intermediates, so `assess,targetlist` ran
on run 35745111146's sample and pair tables in ~5 min.

* **assess**: `PAIRLINE_MATCHES_AT_CHANCE (14 hit-beam drift matches vs 14.6
  expected by chance; all RFI-flagged, 0 candidates)` — the parser fixes left
  88 valid hits (no rank/injection/duplicate rows).
* **targetlist, first pass**: 13 links at the 100 m beam against 0.55
  isotropic. A 24× excess is a systematic until shown otherwise, and it was:
  all 13 were 1–2″ pairs (20–150 AU projected) with Δv_tan 3–5 km/s — orbital
  motion — and 3–6σ parallax differences, the known close-pair astrometric
  bias. Bound-consistent pairs (Δv_tan below the escape speed at the projected
  separation for 2 M☉, +1 km/s) and pairs closer than 4″ are now never ranked.
* **targetlist, final** (1,320,953 links, median length 2.1 pc):

  | beam | isotropic expectation | rankable links with P ≥ 0.5 | Σ P | transmitter unobserved by BL |
  |---|---|---|---|---|
  | 100 m L | 0.55 | **0** | 0.0 | 0 |
  | 10 m 8 GHz | 1.73 | **0** | 0.0 | 0 |
  | 10 m L (1.48°) | 54.8 | 28 | 28.5 | 27 |
  | 5° | 629 | 370 | 380 | 369 |

  Below isotropic everywhere (the exclusions remove ~40 % of links), so the
  list is geometric coincidence, as expected: no star in 100 pc has a
  measured nearest-neighbour link that puts Earth inside a 100-m or 8-GHz
  10-m beam. The 1.48° list (all spillover, links 1.4–2.8 pc, T at 37–65 pc,
  G = 10–17) is a target list for a wide-beam relay premise, nothing more.
