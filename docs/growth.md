# GROWTH — the growing transit: construction around a planet, 2009 → 2026

**Signature S57** (`docs/necrofrontier.md` §2, "The growing transit"; sweep
row g11 in §5).  Stage 1 — the catalogue pass — built 2026-09-13.  **Stage 2A
— the measured TESS depth (`src/seti/growth/stage2.py`, workflow
`growth_stage2.yml`) — built 2026-09-16** and described in §7; the rest of
stage 2 (per-epoch `k(t)`, achromaticity, pixel-level dilution, the long-period
asymmetry branch) is still design and is §8.

---

## 1. The claim

A shell or swarm being **built** around a transiting planet grows the
planet's transit depth over years, and grows it **achromatically** — a
structure at the planet's orbit occults the same fraction of the star in
every band, while the two natural ways a depth changes between epochs are
each chromatic or geometric: starspots and blends change the depth by
different amounts in different bands, and nodal precession changes the
impact parameter, which changes the *duration* in a way a radius change does
not.  A planet being dissolved or mined adds a chromatic, asymmetric tail
instead (the long-period branch; stage 2).

The observable is therefore a per-planet **radius-ratio time series** across
missions: the Kepler-era depth (2009–2013, the KOI `koi_depth`) against the
TESS-era depth (2018–2026, the TOI `pl_trandep`) for every Kepler planet or
candidate that TESS re-detected, after the two deterministic band
corrections, with the duration change as the geometric discriminant.

Haqq-Misra et al. 2022 name "transit depths that change over time due to
construction" as a technosignature; nobody ran it.

---

## 2. Novelty position (§5 row g11) — stated honestly

`results/necrofrontier_lit/concept_scan.json` (2,135 verbatim abstracts,
runs 3–4, 2026-09-13) returns 21 decoy-free hits for group g11 and **none is a
cross-mission secular-growth search**:

| Adjacent work | Why it is not this |
|---|---|
| **Wang & Espinoza 2024** | Transit-depth variations *within* TESS on 330 planets; none robust.  Baseline of years, one band, one pipeline. |
| **Zuckerman et al. 2023** | Single-transit anomalies in 218 Kepler systems; none unexplained.  Event-level, not secular. |
| **Kaye & Aigrain 2025** | Kepler-vs-TESS *ephemerides* (periods and epochs); depths not compared. |
| **Wright et al. 2016, Ĝ IV** | Lists changing depth as a signature class; no search. |
| **HIP 67522 and spot-induced depth variation** | Chromatic, tracks the activity cycle; the `within_han2025_deficit_band` and band-ratio machinery is the baseline for this, not the target. |
| **Han et al. 2025** | The population-level result that public TESS radii are ~6 % low from residual blending — the *expected offset* this channel measures and subtracts. |
| **J1407 "construction zones", exocomet tails** | Puns and the natural chromatic-tail case for the long-period branch. |

**What is claimed:** no published work has compared the Kepler-era and
TESS-era depths of the re-detected Kepler population at fixed impact
parameter, with the limb-darkening band ratio and the Gaia-neighbour
dilution term divided out, and asked which planets got *deeper*.

**What must not be over-claimed:** stage 1 compares two heterogeneous
catalogue numbers (§9).  A `GROWTH_CANDIDATE` here is a reason to run stage 2
on that planet, not a detection.

---

## 3. Data (all public; reached from the runner by the necrofrontier probe)

| Role | Table | Endpoint | Columns used |
|---|---|---|---|
| Kepler-era depth | KOI `cumulative` | Exoplanet Archive TAP sync, `format=csv` | `koi_depth[_err1/2]` (ppm), `koi_ror`, `koi_impact`, `koi_duration[_err1/2]` (h), `koi_dor` (a/R*), `koi_steff`, `koi_slogg`, `koi_kepmag`, `koi_fpflag_{nt,ss,co,ec}`, dispositions, `koi_tce_delivname` |
| TESS-era depth | `toi` | same | `pl_trandep[err1/2]` (ppm), `pl_trandurh[err1/2]` (h), `tid`, `tfopwg_disp`, `pl_orbper`, `st_*` |
| Join key | `ps` rows with `disc_facility LIKE '%Kepler%'` | same | `pl_name` ↔ `hostname` ↔ `tic_id`; `pl_ratror`, `pl_trandep` (**percent** → ppm on read), `default_flag`, `ra`/`dec` |
| Dilution | Gaia DR3 `gaia_source` | Gaia TAP: upload crossmatch first, **per-target sync cones through `pyvo` against `https://gea.esac.esa.int/tap-server/tap` as the fallback** | `phot_g_mean_mag`, separation, within 21″ (one TESS pixel) of the Kepler position |

**The join** (`acquire.join_kepler_tess`, pure) runs **four routes**, tried in
order and counted separately in `summary.json["join"]["joined_by_route"]`:

| Route | How it reaches TESS |
|---|---|
| `name_planet` | `cumulative.kepler_name` = `ps.pl_name` → `ps.tic_id` |
| `name_host` | the KOI's **host** — from its own `kepler_name`, or inherited from a sibling KOI on the same `kepid` — = `ps.hostname` → `ps.tic_id`.  This is the route for the KOIs that have no `kepler_name` at all (only *confirmed* planets get one) |
| `position_tic` | the KOI position → the nearest `ps` row with a TIC id, within 2″.  **No names at all**, and no column beyond the `ra`/`dec`/`tic_id` already verified in `probe.json` |
| `position_period` | the nearest TOI within 2″ whose period matches — no catalogue crossmatch |

The first three end at a TIC id; `toi.tid = tic_id` then gives the TOIs on that
star and the period picks the planet.  A period match is `|ΔP/P| < 10⁻³` or an
integer alias `n` / `1/n` (n ≤ 4), recorded as `period_alias` (a veto for
candidates: the folded events are not the same set of transits).  Each TOI is
used once.  `summary.json["join"]` carries the count by route, how many KOIs
resolved a TIC id by which route, the KOIs with a TIC id whose TOIs matched no
period, the positional matches that failed the period test, and the `ps`
diagnostics (`n_ps_tic_id_parsed` / `n_ps_tic_id_unparsed`, `n_ps_default_rows`,
`n_ps_rows_not_kepler_discovered`); `summary.json["join_statement"]` says all of
it in one sentence.  The **ceiling** is reported apart from the yield —
`n_distinct_koi_tics`, `n_koi_tics_with_a_toi`, `n_toi_on_koi_tics` — so a small
join can be attributed: few TICs with any TOI means TESS has no candidate on
those stars and there is nothing to fix, while many TICs with TOIs and a large
`tic_matched_no_period_match` means the *period test* is what is rejecting.

> **The 2026-09-13 defect (run 34787801172).**  That run reported
> `joined_by_route: {tic_id: 0, position_period: 108}` and `koi_with_tic_id: 0`
> while holding 28,592 `ps` rows.  The cause is that **`ps.tic_id` is a string**
> — `"TIC 122298563"`, not an integer — so the `pd.to_numeric` on it was
> all-NaN, the `dropna` that followed emptied the name → TIC map, and the whole
> join fell through to `position_period`, which only reaches the 108 KOIs whose
> TOI happens to sit within 2″ *and* share a period.  `acquire.parse_tic_id` is
> now the only reader of that column, names are normalised to bare
> alphanumerics (case-folded, punctuation and spacing removed) before any
> comparison, and the `ps` row count, the parsed/unparsed TIC counts and three
> verbatim samples of the column are recorded every run so the assertion is
> checked rather than trusted (`config/growth.yaml`, `join.ps_tic_id_is_string`,
> marked `verify`).  The recorded ADQL shows the `disc_facility` predicate *was*
> sent and honoured — 28,592 is the multi-**reference** row count over the
> Kepler-discovered planets, since `ps` carries one row per published solution
> and `default_flag` marks the adopted one — and the predicate is now re-applied
> in code so a dropped `WHERE` would show up as
> `n_ps_rows_not_kepler_discovered > 0` instead of passing unnoticed.  The
> `default_flag = 1` row supplies the name → TIC map and the reference
> ratio/depth; the other rows are kept as depth references (`ps_n_refs`,
> `ps_trandep_ppm_min/max` per planet).

**The Gaia neighbours, and how they are reached.**  The upload crossmatch
(`tap_upload.targets` joined to `gaiadr3.gaia_source`) is tried first, chunked
and retried.  **Run 34787801172 got `Error 500 … canceling statement due to
statement timeout` on all three attempts for a single 108-target chunk, and
every star ended `not_checked`** — the anonymous ESA queue will not finish an
upload JOIN over the whole `gaia_source`.  So: every attempt's exception text is
now recorded (`acquire.json["gaia"]["errors"]`, with the per-attempt transport
and message — the previous report carried only `n_targets_failed: 108`, which is
what made it undiagnosable), and a chunk whose upload route raises falls back to
**one small sync cone per target through `pyvo`** against
`https://gea.esac.esa.int/tap-server/tap` (the transport
`src/seti/baffle/acquire.py` uses), retried, with finished chunks checkpointed
under `results/growth/data/gaia_checkpoint/` so a re-run does not re-query them.
A target neither route reaches keeps `neighbours_status = QUERY_FAILED`, is
reported as `neighbours_not_checked`, and **may never be called isolated** —
the veto is unchanged.  `summary.json["gaia"]["by_route"]` and
`summary.json["gaia_statement"]` say which route supplied each star.

**Proper motion:** the Gaia epoch (2016.0) sits between the two missions and
the search radius is 21″; a 100 mas/yr star moves 1″ over the whole baseline.
No propagation is applied; the assumption is stated here.

---

## 4. Method (`src/seti/growth/drift.py`, pure functions)

For each joined planet, `R = depth_TESS / depth_Kepler`.

### 4.1 Limb-darkening band ratio (correction a)
In the small-planet limit the observed depth is `k² · I(μ)/⟨I⟩` with
`μ = √(1 − b²)`, `I(μ)/I(1) = 1 − u₁(1 − μ) − u₂(1 − μ)²`,
`⟨I⟩/I(1) = 1 − u₁/3 − u₂/6` (Mandel & Agol 2002; the small-planet form of
Csizmadia et al. 2013).  Kepler and TESS have different `(u₁, u₂)`, so the
*same* `k` gives a different depth in each band; the ratio
`f_LD = F_TESS(b)/F_Kepler(b)` is divided out at the catalogue `koi_impact`.
Coefficients are a Teff grid at logg 4.5 in `config/growth.yaml`
(`limb_darkening`), interpolated linearly in Teff, clamped at the grid ends,
nearest-logg grid; the source is Claret & Bloemen 2011 (Kepler) and Claret
2017 (TESS) and **every row is marked `verify`** — the values were transcribed
from memory in a sandbox with no archive egress.  The correction is small
(0.963 at b = 0, 1.04 at b = 0.9 for a solar-type star), so a 0.05 error in
`u₁` moves `ln R` by ~0.01, well inside `σ_sys`.  A missing Teff uses 5800 K
and is flagged `ld_teff_missing`.

### 4.2 Dilution (correction b)
`c = Σ_neighbours 10^(−0.4 ΔG) · w(d)` over Gaia DR3 sources within 21″,
`ΔG = G_neighbour − G_target` (the target is the nearest Gaia source within
1.5″; else Kp is used and `target_g_source = kepmag`), and
`w(d) = ½ erfc((d − r_ap)/(√2 σ_psf))` with `r_ap = 21″`, `σ_psf = 10.5″`
from config — an approximate aperture-capture fraction (the TESS PRF is
undersampled and position-dependent).  The TESS depth is divided by `(1 − c)`.
The **unweighted** sum `c_max` is carried alongside as the upper bound.

**The Kepler side is taken as deblended by the pipeline's own flux fraction**:
the DV depths in `cumulative` are fitted on PDCSAP flux, which carries the
CROWDSAP crowding correction from the Kepler Input Catalog.  The SPOC TOI
depths are fitted the same way (CROWDSAP from the TIC), and QLP depths carry a
TIC contamination-ratio correction — so the Gaia term applied here **can
double-correct**.  It is applied as specified (`apply_dilution_to_tess: true`)
because the residual Han et al. 2025 deficit shows the catalogue corrections
are incomplete, and the ambiguity is handled by the neighbour veto (§5):
a candidate must survive the *full* range `[0, c_max]`.

### 4.3 The corrected log ratio (correction c)
`ln R_corr = ln D_T − ln(1 − c) − ln D_K − ln f_LD`,
`σ² = (e_T/D_T)² + (e_K/D_K)² + σ_sys²`, catalogue errors symmetrised as
`max(|err1|, |err2|)`, `σ_sys = 0.05` (config) as the floor for
pipeline-to-pipeline depth heterogeneity.

### 4.4 The duration test (d)
A depth change by `R` at fixed `b` is `k → k√R`, and the total duration
changes as `T₁₄'/T₁₄ = √(((1 + k')² − b²)/((1 + k)² − b²))` (the full
`arcsin` form with `koi_dor` when present).  The observed
`pl_trandurh / koi_duration` is compared with this at
`σ² = (e_T/T_T)² + (e_K/T_K)² + σ_dur,sys²`; `|z| < 3` is **`fixed_b`**.  When
it disagrees, the `b'` that would give the observed ratio at fixed `k` is
solved: `b'² = (1 + k)² − ratio² ((1 + k)² − b²)`.  A real `b'` means the
change **`tracks_b`** — nodal precession (KOI-120, Kepler-13Ab, Kepler-47d) or
a grazing geometry — and is never a candidate; no real `b'` is
`duration_inconsistent`.  Missing durations are `inconclusive` (flagged, and
not `fixed_b`, so not a candidate).

### 4.5 The expected sign, and the population offset
Han et al. 2025 find public TESS planet radii ~6 % low from residual
blending: a **uniform ~12 % depth deficit is the ordinary outcome** for this
comparison, so `SHALLOWER_TESS` is the expected direction and `DEEPER_TESS`
the anomalous one.  The channel measures this on its own sample — the median
`ln R_corr` over every measured planet (`population.population_offset`,
reported beside `ln(1 − 0.12) = −0.128`) — and subtracts it before
classification (`subtract_population_median: true`).  Growth is growth
*relative to the population's own systematic offset*.  A planet whose raw
ratio sits within 3σ of the Han deficit is flagged
`within_han2025_deficit_band` (report only).

### 4.6 Classification (e)
| Class | Condition |
|---|---|
| `CONSISTENT` | `|z| < 3`, `z = (ln R_corr − offset)/σ` |
| `SHALLOWER_TESS` | `z ≤ −3` |
| `DEEPER_TESS` | `z ≥ 3` but not a candidate |
| `GROWTH_CANDIDATE` | `z ≥ 5` **and** `duration_verdict = fixed_b` **and** no veto (§5) — `koi_fpflag_*` all 0, both dispositions candidate/confirmed, no neighbour able to supply the change |
| `UNMEASURED` | a depth or its error missing |

`would_be_candidate_without_vetoes` is carried for every row so the veto
ledger has a denominator.

### 4.7 The long-period branch (list only)
Every joined or TESS-only planet with `P > 30 d` goes to
`results/growth/long_period.csv` (`source = joined | tess_only`) as the
stage-2 input for the asymmetry / tailed-transit statistic: no
photoevaporative engine exists beyond ~0.3 AU, so a tailed cold planet has no
natural model.  Nothing is computed on it here.

---

## 5. Contamination ledger (`src/seti/growth/vet.py`)

Every veto is a named mechanism with its own counter
(`summary.json["rejection_counters"]`: first veto, every veto raised, every
flag raised, the duration verdicts).

| Veto | Mechanism |
|---|---|
| `grazing` | `b > 1 − k`: the depth is set by the chord and any precession moves it |
| `ttv_system` | known large-TTV systems (config list from Holczer et al. 2016 and the named cases — **incomplete, `verify`**); a folded depth is smeared differently by each mission's baseline and cadence.  `koi_tce_delivname` is carried as provenance |
| `fpflag_nt` / `fpflag_ss` / `fpflag_co` / `fpflag_ec` | the KOI false-positive flags (not transit-like; stellar eclipse; centroid offset; ephemeris-match contamination) |
| `koi_false_positive` | `koi_disposition` / `koi_pdisposition` outside {CONFIRMED, CANDIDATE} |
| `toi_disposition_not_candidate` | `tfopwg_disp` outside {PC, CP, KP} (APC, FP, FA, EB all veto) |
| `period_alias` | the TESS period is an integer alias of the Kepler one |
| `neighbours_not_checked` | **every** Gaia route for this star failed (the chunk's upload crossmatch *and* its per-target cone): isolation is unknown, not established |
| `neighbour_can_supply_change` | `(ln R_corr + ln(1 − c_max) − offset)/σ < 5`: if every neighbour inside the pixel had already been removed by the pipeline, the growth would fall below the gate |

Report-only flags: `multi_sector_scatter:not_checked` (per-sector TESS depths
are not in the catalogue — stage 2), `within_han2025_deficit_band`,
`duration_inconclusive`, `ld_teff_missing`.

Inherited from `docs/channel-brief.md` §4: **a single-band anomaly is an
artefact until confirmed in a second band.**  Stage 1 has exactly one band
per epoch; the achromaticity claim is stage 2's to make (§7).

---

## 6. Outputs (`results/growth/`)

| File | Content |
|---|---|
| `probe.json` | reachability and real column names of the three tables |
| `acquire.json`, `acquisition_log.json` | table statuses (`OK` / `QUERY_FAILED` / `QUERY_RETURNED_ZERO_ROWS` kept apart), join counts by route with the `ps` diagnostics and `join_statement`, Gaia chunk outcomes **with every attempt's exception text** (`gaia.errors`) and the route that supplied each star (`gaia.by_route`) |
| `screened.csv` | every joined planet with every drift quantity |
| `joined.csv` | the same, vetted and classed |
| `candidates.csv` | the `GROWTH_CANDIDATE` rows (slim columns) |
| `long_period.csv` | the stage-2 list |
| `summary.json` | `verdict`, `join`, `join_statement`, `gaia`, `gaia_statement`, `classes`, `rejection_counters`, `population`, `n_long_period`, `degraded`, `checks_not_performed` |

Verdicts: **`NO_DATA_REACHED`** (nothing measured; `reason` says whether an
archive failed, answered empty, or the join was empty),
**`NO_DEPTH_DRIFT_CANDIDATE`** (planets measured, none at the gate — a count,
not a limit), **`DEPTH_DRIFT_CANDIDATES`** (≥ 1 candidate, pending stage 2).
`degraded` lists every partial failure separately; the verdict string is never
decorated.  None of these is written up as a result (CLAUDE.md).

Run: `python -m seti.growth.run --stage {probe,acquire,screen,assess,all}
[--out-dir results/growth] [--skip-gaia]`; workflow `growth.yml`
(`workflow_dispatch`, input `stage`).

---

## 7. Stage 2A — the measured TESS depth (`src/seti/growth/stage2.py`, BUILT)

### 7.1 Why, and what it settles

Run 35038510064 returned one `GROWTH_CANDIDATE` — **Kepler-718 b = K00897.01 =
KIC 7849854 = TIC 268924036 = TOI 4490.01**, `P = 2.0523499 d`, CONFIRMED /
TFOPWG `KP`, Kepler-era `koi_depth` 14,281 ± 18 ppm against TOI `pl_trandep`
34,476 ± 2,349 ppm, `ln R_corr = 0.961`, `z = 8.67`, duration ratio 1.022
observed against 1.073 expected at fixed `b` (`z = −0.45`, so `fixed_b`), 7 Gaia
neighbours inside 21″ and no veto.  Two things make it untrustworthy as it
stands, and stage 2A settles the first:

1. **The stage-1 error model is broken.**  The same run's population has
   **31 of 108 measured planets above 5σ** of the population median and **37
   above 3σ** (`summary.json["population"]`).  A sample in which a third of the
   objects are five-sigma outliers has formal errors that do not describe the
   pipeline-to-pipeline scatter, so `z = 8.67` is **not** an 8.67-sigma
   statement.  Stage 2A does not try to repair the error model: it removes the
   dependence on it by **measuring the TESS depth from the light curve**.
2. **A deeper TESS transit cannot be caused by dilution** — dilution makes a
   transit *shallower* — **but a signal that originates on a different star
   inside the ~21″ TESS pixel produces exactly this**, and the `contam` term
   does not test for it.  That is the centroid test, it is **not** done here,
   and it is named in `summary.json["checks_not_performed"]`.

### 7.2 The decisive measurement: three numbers, not two

| Number | Where it comes from |
|---|---|
| `depth_kepler_ppm` | KOI `cumulative.koi_depth` (**Kepler band**) |
| `depth_toi_ppm` | TOI `toi.pl_trandep` (**TESS band**) |
| `depth_measured_ppm` | **this module's fit to the TESS light curve** |

The Kepler depth is multiplied by `f_LD = band_ratio(Teff, logg, b)`
(§4.1, `F_TESS(b)/F_Kepler(b)`) before any comparison, so the bandpass
difference is not charged to "growth"; `depth_kepler_in_tess_band_ppm` is
reported.  Each comparison is `z = ln(D_meas/D_ref)/σ` with
`σ² = (e_meas/D_meas)² + (e_ref/D_ref)² + σ_sys,ln²`, `σ_sys,ln = 0.05`
(config, `verify`: a stated floor, not a measurement — a shortlist of one
cannot measure its own scatter).  `|z| < n_agree` (3.0) is "agrees with".

**The verdict vocabulary** (`stage2.TARGET_VERDICTS`):

| Verdict | Meaning |
|---|---|
| `MEASURED_DEPTH_MATCHES_KEPLER` | agrees with Kepler, disagrees with the TOI → **the TOI value was wrong, the candidate dies**, and the channel has learned its stage-1 error model is catalogue-driven |
| `MEASURED_DEPTH_MATCHES_TOI` | agrees with the deep TOI value, disagrees with Kepler → **the depth really did change**; the candidate survives **to the centroid test**, which is not a detection |
| `MEASURED_DEPTH_MATCHES_NEITHER` | outside both.  **Said, not resolved** — no pick is made |
| `MEASURED_DEPTH_MATCHES_BOTH` | the measurement does not separate the two references.  An admission, never a pick; `z_toi_vs_kepler` and `references_separated` say whether the failure is the measurement's precision or the references' own agreement |
| `UNMEASURED` | no depth was fitted.  `unmeasured_reason` ∈ {`QUERY_FAILED`, `QUERY_RETURNED_ZERO_ROWS`, `EPHEMERIS_UNAVAILABLE`, `NO_USABLE_TRANSIT`, `BUDGET_EXHAUSTED`, `NO_REFERENCE_DEPTH`} — the last of these is a depth that WAS measured with no usable catalogue value to compare it with, which is not the same as disagreeing with one |

**A target whose light curve could not be fetched is `UNMEASURED` and is never
reported as agreeing with anything**; `QUERY_FAILED` (the service errored) and
`QUERY_RETURNED_ZERO_ROWS` (it answered with nothing) stay different facts, and
the workflow fails the run if an `UNMEASURED` row carries an agreement flag or
no reason.  Run verdicts: `NO_DATA_REACHED`, `STAGE1_DEPTH_CHANGE_REFUTED`
(every measured target matched Kepler), `STAGE1_DEPTH_CHANGE_CONFIRMED` (≥ 1
matched the TOI), `STAGE1_DEPTH_CHANGE_UNRESOLVED`.

### 7.3 The ephemeris — the one arithmetic that must not be wrong

The period, epoch and duration are the KOI's own (`koi_period`,
`koi_time0bk`, `koi_duration`) and are **fixed**: stage 2A measures a *depth*,
it does not re-derive an ephemeris.  `koi_time0bk` is **not** in the stage-1
column list, so stage 2A pulls it from `cumulative` itself.  The two missions
count from different zero points —

* Kepler **BKJD** = BJD − 2454833.0
* TESS  **BTJD** = BJD − 2457000.0
* therefore `t_BTJD = t_BKJD − 2167.0` **exactly**

— and the hand-worked value the test suite checks is: BKJD 170.0 = BJD
2455003.0 = **BTJD −1997.0**.  The epoch is then propagated by an integer
number of periods to the TESS window and the accumulated uncertainty
`√(σ_T0² + (n σ_P)²)` is reported in minutes (`ephemeris_sigma_minutes`,
`n_epochs_propagated`); above 10 % of the duration it raises
`ephemeris_drift_over_10pct_of_duration`, because a smeared fold is a shallow
depth and that must be visible rather than absorbed.

### 7.4 The fit

1. **each sector is normalised separately** by its own robust median;
2. every predicted transit gets a local window `|Δt| ≤ w·T₁₄`; the baseline is
   a straight line in `Δt` fitted over the **out-of-transit part only** (the
   transit is masked, with a guard band at `0.75 T₁₄`), and both sides of the
   transit must be present;
3. that transit's depth is `1 − ⟨flux/baseline⟩` over a **core** window
   `|Δt| ≤ 0.35 T₁₄` **shrunk by half the exposure time**, so a sample whose
   integration straddles ingress is not counted as flat-bottom flux;
4. **the cadence sets the window, not the other way round**: the outer window
   is widened until it can hold `min_baseline_points` at the *actual* cadence,
   and the core requirement drops to one sample when the cadence cannot supply
   two.  Without this a 30-minute FFI transit would be discarded for want of a
   baseline rather than measured and flagged;
5. per-transit depths are combined by inverse variance; the quoted error is a
   **bootstrap over transits** (≥ 8 transits) or the **analytic** propagation
   below that, and `depth_measured_err_method` always says which.  The
   analytic, χ²-scaled and bootstrap errors are all reported.

**Reported beside the depth, per target:** the depth **per sector**
(`sectors.csv` — a depth that differs between sectors is a systematic, not
growth; `sector_scatter_chi2_per_dof` and the `sector_scatter` flag), the
out-of-transit scatter, the **odd–even** depth difference
(`odd_even_diff_ppm`, `odd_even_sigma`, flag `odd_even_significant` at 3σ) —
the classic eclipsing-binary signature, nearly free once the fold exists and
the only handle stage 2A has on the "different star in the pixel" mechanism —
and a binned fold per target under `folds/`.

**Smearing is flagged, never quietly corrected.**  `exptime_over_duration` and
`smeared` (exposure > 0.2 T₁₄) are columns; `depth_is_lower_bound` marks the
case where no flat core survives the integration at all.  `sector_list`,
`authors` and `exptimes_s` record **which product and which sectors** each
depth came from, so a 30-minute FFI depth on a 2-hour transit cannot be read as
if it were a 2-minute SPOC one.

### 7.5 Data access, budgets, outputs

Light curves come from MAST, **SPOC 2-minute where it exists, TESS-SPOC / QLP
FFI otherwise** (`stage2.mast.authors`).  Two routes, established **at
runtime** and recorded in `stage2/probe.json`: `lightkurve`
(`search_lightcurve("TIC n", mission="TESS")`) when it is installed — it is the
optional `tess` extra in `pyproject.toml`, which `growth_stage2.yml` installs —
and otherwise `astroquery.mast` `Observations.query_criteria` →
`get_product_list` → `download_file(dataURI)` with the FITS read directly
(`PDCSAP_FLUX`, else `KSPSAP_FLUX` / `DET_FLUX` / `SAP_FLUX`, recorded per
sector; `TIME` converted through the file's own `BJDREFI`/`BJDREFF` rather than
assumed to be BTJD; `QUALITY != 0` masked).

**Every network stage has a wall-clock budget** — `stage2.mast.budget_s` for
the whole measure stage, `per_target_budget_s` per target,
`archive_timeout_s`/`archive_retries` for the Exoplanet Archive pulls — copied
from the `gaia.cone_budget_s` pattern and for the reason stated there: run
34789826297 sat three hours in an unbounded fetch loop and would have been
killed with nothing committed.  What the budget does not reach is `UNMEASURED`
with reason `BUDGET_EXHAUSTED`, never a depth.

Outputs, `results/growth/stage2/`:

| File | Content |
|---|---|
| `probe.json` | which MAST route exists on this machine (imports only) |
| `acquire.json`, `acquisition_log.json` | the ephemeris/TOI pull statuses, the per-target light-curve status and route, elapsed time against the budget, every failed attempt's exception text |
| `measurements.csv` | one row per shortlisted target: the three depths, both `z`s, the verdict, the ephemeris record, the per-target flags |
| `sectors.csv` | per sector: author, exposure, points, transits, depth ± error, `smeared`, `exptime_over_duration` |
| `folds/<KOI>.csv` | the binned fold, for a human to look at |
| `summary.json` | `verdict`, `reason`, `target_verdicts`, `unmeasured_reasons`, `flags`, `targets`, `acquisition`, `config`, `checks_not_performed` |

The shortlist and every threshold are in `config/growth.yaml` under `stage2:`
(`shortlist`, `mast`, `fit`, `compare`), with a `verify:` note on everything
asserted rather than measured.  Run:
`python -m seti.growth.stage2 --stage {probe,measure,assess,all}`
(or `seti growth-stage2 ...`); workflow `growth_stage2.yml`.

### 7.6 What stage 2A cannot say

* **it does not test the centroid.**  A deeper TESS transit cannot come from
  dilution, but a nearby eclipsing binary at the same period inside the pixel
  gives exactly this.  `odd_even_significant` catches one flavour; the
  per-pixel centroid / difference-image test is stage 3.
* **the Kepler depth is still the catalogue's.**  Only the TESS side is
  re-measured here, so `MEASURED_DEPTH_MATCHES_TOI` establishes that the TESS
  depth is what the TOI says, not yet that it differs from a re-fitted Kepler
  depth.
* **one band per epoch**, exactly as at stage 1: achromaticity is untested.
* **`σ_sys,ln = 0.05` is asserted**, not measured (§7.2).

---

## 8. Stage 2B — design (not built)

Per candidate (and per long-period planet), from the light curves:

1. **Per-epoch `k(t)`.**  Kepler long-cadence PDCSAP (Q0–Q17) and every TESS
   sector (SPOC 2-min where it exists, else TESS-SPOC / QLP FFI), each fitted
   with `batman` at **fixed** `(P, T₀, a/R*, b)` from a joint fit and
   per-band quadratic LD from the *verified* Claret tables, with only `k` and
   the baseline free per transit (Kepler) or per sector (TESS).  Cadence
   integration (30 min Kepler, 2/10/30 min TESS) inside the model, not as a
   correction.
2. **Fit `k(t) = k₀ + k₁ t`**; require `|k₁|/σ > 5` with the residuals
   consistent between missions once the per-band `k` are on the same
   footing (the band ratio must match the LD prediction — this is the
   achromaticity test, and kills spots and blends).
3. **Duration and ingress at every epoch**: `T₁₄` and `τ` must change as `k`
   (not as `b`); a drifting `b` is precession and ends the case.
4. **Dilution from the pixels**, not the catalogue: re-derive the TESS
   crowding from Gaia DR3 sources rendered through the sector PRF, and check
   the Kepler CROWDSAP against the same sources.
5. **Multi-sector scatter**: the per-sector `k` variance against the
   photometric expectation (the `multi_sector_scatter` check stage 1 records
   as `not_checked`).
6. **Long-period branch**: for `P > 30 d`, an ingress/egress asymmetry
   statistic (the difference of the fitted ingress and egress durations, and
   the pre-/post-transit residual power) and the epoch-to-epoch depth
   variance; a chromatic, asymmetric tail on a cold planet has no natural
   model.

Scale: the joined sample, i.e. the KOIs of the 9,564-row `cumulative` table
that a TOI re-detects.  Run 34787801172 reached **108** of them — the defect
above, with only `position_period` working; with the three TIC routes restored
the expectation is several hundred, and `summary.json["join_statement"]` reports
the actual number and its breakdown by route, so the figure quoted here is
never the one a reader has to trust.  The light curves are on MAST
(`astroquery.mast`; `download_file(dataURI)`, per `channel-brief.md` §2).

---

## 9. Limits — what stage 1 can and cannot say

* **Catalogue depths are heterogeneous by pipeline and epoch.**  `koi_depth`
  is a DV fit on Q1–Q17 DR25 long-cadence PDCSAP; `pl_trandep` is whichever
  pipeline delivered the TOI (SPOC 2-min DV, TESS-SPOC FFI, or QLP), on a
  different cadence, with a different crowding model, over a different
  baseline — and is revised as sectors accumulate.  `σ_sys = 0.05` is the
  channel's stated floor for this, not a measurement of it; the population
  scatter in `ln R_corr` (`population.ln_ratio_corr_p16/p84`) is the
  measurement, and is what a reader should compare the candidates against.
* **The two epochs are each one band.**  Achromaticity — the property that
  makes a shell different from a spot — is untested at stage 1.
* **Dilution may be double-counted** (§4.2); the neighbour veto absorbs this
  in the conservative direction for candidates and the permissive one for the
  count of `SHALLOWER_TESS`.
* **The LD grid and the TTV list are unverified transcriptions** (marked in
  config); both are small corrections at stage 1 and both are replaced by the
  real tables at stage 2.
* **The join is incomplete by construction**: a KOI TESS never re-detected
  (too shallow at TESS precision, or with a period longer than the sector
  coverage folds) is absent, so the joined population is biased to deep,
  short-period planets — the population offset is measured on that
  population and applies to it.  The four routes bound how much of that
  incompleteness is *ours*: `join_statement` reports the yield of each, so a
  route that stops working is visible as a number rather than as a quietly
  smaller sample.  `position_tic` matches a KOI to a `ps` star within 2″ and
  inherits that star's TIC id — in a crowded field the nearest `ps` row is not
  guaranteed to be the KOI's own star, which is one more reason the Gaia
  neighbour veto is not optional.
* **`NO_DEPTH_DRIFT_CANDIDATE` is a count**, not an occurrence limit on
  construction around the Kepler planets, and is not written up.
