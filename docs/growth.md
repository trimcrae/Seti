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

## 7. Stage 2A — the measured depth in BOTH eras (`src/seti/growth/stage2.py`, BUILT)

### 7.0 The asymmetry, and its removal

The first form of this stage fitted the **TESS** era from the light curve and
then compared it against `cumulative.koi_depth` — **a catalogue number**,
produced by a different pipeline, in a different decade, through a different
aperture.  That is precisely the heterogeneity that broke stage 1's error model
(§7.1), so the comparison inherited the flaw it was built to remove.  Run
35041932130 therefore measured 30,614 ± ~1,690 ppm in the TESS era and put it
~10σ from a number nobody had re-derived.

Stage 2A now fits **both eras with the same fitter** — the same fold, the same
locally fitted transit-masked baseline, the same exposure-shrunk core window,
the same inverse-variance combination and bootstrap over transits, the same
`dedupe_sectors`.  Only the archive product and the time system differ.  The
**primary** result is the like-for-like verdict (§7.2a).  Both answers are
valuable and neither is favoured:

* if our Kepler-era fit also gives ~30,000 ppm, **nothing changed** — the KOI
  catalogue depth is simply wrong for this object, the candidate dies, and the
  channel has learned something load-bearing about its stage-1 *inputs*;
* if our Kepler-era fit reproduces ~14,000 ppm, the change survives a
  comparison in which nothing but the sky differs and becomes much harder to
  dismiss.

The three-number catalogue comparison (§7.2) is **kept**, and is now a
diagnostic of the **catalogues** rather than of the sky: it says which table is
right about the TESS era.  `summary.json` states which verdict is primary.

And the error the verdict is formed from is no longer the bootstrap alone.  The
**reduction ensemble** (§7.5a) measures the depth under *every* reduction the
archive serves for each era and folds the spread over those members into each
era's error as a systematic; the like-for-like `z` uses the **total**.  This
closes the second hole: the quoted TESS error did not describe the real
scatter, and this channel's own duplicated-sector run proves it (20–30 %
disagreement between two reductions of the same pixels, against a 1,521 ppm
quoted error).

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

### 7.2a THE PRIMARY VERDICT — our Kepler-era fit against our TESS-era fit

| Number | Where it comes from |
|---|---|
| `depth_kepler_measured_ppm` | **this module's fit to the KEPLER light curve** |
| `depth_measured_ppm` (= `depth_tess_measured_ppm`) | **this module's fit to the TESS light curve** |
| `depth_kepler_measured_in_tess_band_ppm` | the former × `band_ratio(Teff, logg, b)` |

`z_measured_eras = ln(D_TESS / D_Kepler,in TESS band) / σ`, positive meaning
deeper in the TESS era, with the same `σ² = (e₁/D₁)² + (e₂/D₂)² + σ_sys,ln²`
the catalogue comparison uses.  The band ratio (§4.1) is applied to our Kepler
**measurement** for the same reason it is applied to the Kepler **catalogue**
depth: each era is one bandpass, and the band difference must not be charged to
"growth".

| Verdict (`like_for_like_verdict`) | Meaning |
|---|---|
| `MEASURED_DEPTH_CHANGED` | `|z| ≥ n_agree`.  The change survives a comparison in which nothing but the sky differs.  **Still not a detection** — the centroid test is outstanding |
| `MEASURED_DEPTH_UNCHANGED` | `|z| < n_agree` **and** the comparison had the power to have seen the change under test.  Nothing changed: `koi_depth` is wrong for this object and the candidate dies |
| `MEASURED_DEPTH_UNRESOLVED` | `|z| < n_agree` but `detectable_ln_ratio = n_agree·σ` exceeds `ln_ratio_under_test`.  **An agreement without power is not a refutation** and is never reported as one |
| `MEASURED_DEPTH_ERA_UNMEASURED` | one or both eras produced no depth.  `like_for_like_unmeasured_reason` names the era and the reason (`KEPLER_ERA:QUERY_FAILED`, `TESS_ERA:QUERY_RETURNED_ZERO_ROWS`, `KEPLER_ERA:KEPLER_ERA_NOT_ATTEMPTED`, …).  **An unmeasured era never agrees with the other one** |

`ln_ratio_under_test` is the change stage 1 claimed —
`ln(TOI / KOI-in-TESS-band)`, ≈ 0.88 for K00897.01 — taken from the catalogues
whenever both are usable, and `compare.min_detectable_ln_ratio` (0.10) when they
are not.  Without this power floor a fit too noisy to tell 14,000 ppm from
30,000 ppm would "refute" the candidate by failing to measure it: stage 1's
error in the opposite direction.

Run verdicts (`primary_verdict`): `LIKE_FOR_LIKE_NO_DATA`,
`MEASURED_DEPTH_CHANGE_CONFIRMED`, `MEASURED_DEPTH_CHANGE_REFUTED` (every
comparison agreed **with** the power to disagree), `MEASURED_DEPTH_CHANGE_UNRESOLVED`.

**And what our Kepler-era fit says about the KOI table.**  The same fit is
compared with `cumulative.koi_depth` **in the Kepler band, with no band ratio**
— there is no band change to correct — giving `koi_depth_verdict` ∈
{`KOI_DEPTH_CONFIRMED`, `KOI_DEPTH_CONTRADICTED`, `KOI_DEPTH_UNCHECKED`} and
`z_kepler_measured_vs_koi`.  A contradiction is a **finding about the KOI
table**, i.e. about stage 1's own input, and is reported as one in
`summary.json["koi_catalogue_check"]` — with the offending target, both depths
and the ratio — not buried in the depth story.

### 7.2 The secondary comparison: three numbers, and what they now diagnose

| Number | Where it comes from |
|---|---|
| `depth_kepler_ppm` | KOI `cumulative.koi_depth` (**Kepler band**) |
| `depth_toi_ppm` | TOI `toi.pl_trandep` (**TESS band**) |
| `depth_measured_ppm` | **this module's fit to the TESS light curve** |

With both eras now fitted here, this comparison no longer carries the claim
about the sky.  What it says is **which catalogue is right about the TESS
era**, and that is worth saying: it is a measurement of the tables stage 1 is
built on.  `summary.json` marks it `verdict_is_primary: false`.

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
2455003.0 = **BTJD −1997.0**.

**Now that both eras are fitted, the conversion is a per-era decision and is
made once, by name** (`epoch_in_era(t0_bkjd, era)`):

* `ERA_TESS` — TESS light curves are BTJD, so subtract 2167 d;
* `ERA_KEPLER` — **Kepler light curves are already BKJD, and so is
  `koi_time0bk`: no conversion at all.**

Converting on the Kepler side too is the obvious bug in a two-era module.  It
would put the fold 2167 days from any transit and return no depth — which reads
as a Kepler era that "did not change" while in truth nothing was measured.
Both branches are tested, and a test shows the double conversion destroying the
Kepler fold.  The epoch is then propagated within each era by an integer
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

**Kepler light curves come from the same archive by the same two routes** —
`lightkurve` `search_lightcurve("KIC 7849854", mission="Kepler")`, else
`astroquery.mast` `query_criteria(obs_collection="Kepler",
dataproduct_type="timeseries", target_name=…)` → `get_product_list` →
`download_file(dataURI)` with the `LLC`/`SLC` products read by
`read_kepler_lc_fits` (`PDCSAP_FLUX` then `SAP_FLUX`; quality is `SAP_QUALITY`,
not `QUALITY`; `TIME` converted through the file's own `BJDREFI`/`BJDREFF` to
**BKJD**).  Which route answered, and which spelling of the target name, is a
**runtime** fact and is recorded per target (`kepler_lc_route`,
`kepler_lc_status`).

**Short cadence is preferred where it exists.**  Kepler long cadence is 270
co-added frames = 1765.5 s = **29.4 min**, which on K00897.01's 2.078-hour
transit is 0.236 T₁₄ — over `fit.smear_fraction`, so a long-cadence quarter
carries `smeared` exactly as a 30-minute TESS FFI sector does, and the
exposure-shrunk core window is what keeps its depth unbiased rather than
shallow.  Nothing is filtered by cadence at the archive: every product is
fetched, **short-cadence month-files are merged into quarters first**
(`merge_kepler_segments` — Kepler ships short cadence three files to a quarter,
and keying the dedupe on the quarter without merging would throw two thirds of
them away), and `dedupe_sectors` then keeps the shortest-cadence reduction of
each quarter, because two cadences of one quarter are **the same pixels**.
Each month-file is normalised by its own robust median before concatenation.
Whether short cadence exists for KIC 7849854 is recorded as
`kepler_has_short_cadence` / `kepler_cadences`.

**Every network stage has a wall-clock budget** — `stage2.mast.budget_s` for
the TESS side of the measure stage and **`kepler_budget_s` for the Kepler
side**, counted separately so a slow fetch on one side cannot starve the other
and leave the like-for-like comparison one-sided; `per_target_budget_s` /
`kepler_per_target_budget_s` per target;
`archive_timeout_s`/`archive_retries` for the Exoplanet Archive pulls — copied
from the `gaia.cone_budget_s` pattern and for the reason stated there: run
34789826297 sat three hours in an unbounded fetch loop and would have been
killed with nothing committed.  What the budget does not reach is `UNMEASURED`
with reason `BUDGET_EXHAUSTED`, never a depth.

`stage2.mast.kepler_enabled` is **true** in `config/growth.yaml` and **false**
in the code's own defaults, deliberately: a caller that has not thought about
the Kepler era must not silently open a socket (the test suite raises on any
socket, and an exception swallowed by the retry loop would read as
`QUERY_FAILED`).  `KEPLER_ERA_NOT_ATTEMPTED` and `QUERY_FAILED` are different
facts, and `growth_stage2.yml` **fails the run** if `summary.json` comes back
with the primary comparison switched off — so a dropped key is loud rather than
a quietly one-sided result.

### 7.5a The reduction ensemble — the error that did not describe the scatter

**The hole.**  The quoted TESS error does not describe the real scatter, and
the evidence is in this channel's own output.  The out-of-transit scatter is
**414 ppm in Kepler and 67,631 ppm in TESS** — a factor of 163; at G = 15.23
the TESS light curve is faint and systematics-limited.  Worse, and decisively:
before `dedupe_sectors` existed, run 35041932130 measured **the same seven
sectors** under two pipelines (`results/growth/stage2/sectors.csv` at commit
`050402a`).  The 2-minute SPOC reductions gave **25,425–35,498 ppm**; the FFI
TESS-SPOC reductions **of the same pixels** gave **33,078–39,798 ppm**.  Two
reductions of identical photons disagreeing by 20–30 % is a systematic **far
larger than the 1,521 ppm quoted error**.  Stage 1's entire failure was a
quoted error that did not describe the real scatter (31 of 108 planets above
5 σ of the population median); repeating that one level down, on a single
object, would be worse.  The direction is not benign either way: a faint star
in a crowded aperture with an **over-subtracted background** yields an
*inflated* transit depth, which is a specific, testable, mundane explanation
for the whole result.

**What is built** (`stage2.measure_reduction_ensemble`, config
`stage2.ensemble`).  For each era, for each segment, the depth is measured
under **every available reduction** — the full cross product of pipeline author
(`SPOC`, `TESS-SPOC`, `QLP`; `Kepler` for the Kepler era) and flux column
(`PDCSAP_FLUX`, `KSPSAP_FLUX`, `DET_FLUX`, `SAP_FLUX`).  Reductions are
**enumerated, not chosen**: most combinations do not exist at the archive and
come back `FLUX_COLUMN_NOT_PRESENT`, which is a *recorded absence*, distinct
from `QUERY_FAILED` (the archive errored), `QUERY_RETURNED_ZERO_ROWS` (it
answered with nothing), `AUTHOR_NOT_SERVED` (it answered with **another
pipeline's** products) and `BUDGET_EXHAUSTED`.  Every member, measured or not,
is a row in `reductions.csv`.

`AUTHOR_NOT_SERVED` exists because run `bcc670c` produced it on real data: QLP
serves nothing for TIC 268924036, and `lightkurve_lc_fn` drops its author
filter when nothing matches it, so the three QLP members came back carrying
**SPOC's** light curves and entered the ensemble as byte-identical copies of the
SPOC members (29,255 and 11,196 ppm twice over).  That double-weighted one
reduction in the percentile spread (9,303 instead of 9,661 ppm) and turned two
genuine SAP/PDCSAP pairs into three (error 977 instead of 1,226 ppm) — shrinking
the very error the ensemble exists to size.  A member is now measured only if
the products really carry **both** the author and the column that were asked
for.

**The spread, and how it binds.**  `depth_reduction_spread_ppm` is **half the
range between the 16th and 84th percentiles** of the measured members' depths —
the robust analogue of one σ, not moved by a single pathological reduction the
way `max − min` is — reported with `min`, `max`, the percentiles, the
per-member table (`reductions.csv`) and each member's own per-segment
breakdown (`sectors.csv`, `scope = "ensemble"`), which is what separates a
uniform pipeline offset from sector-specific noise.  It enters the error as a
systematic:

```
depth_*_total_err_ppm = sqrt(statistical² + depth_reduction_spread_ppm²)
```

**and the like-for-like verdict is formed from the TOTAL**, with
`z_measured_eras_stat_only` kept beside it so the effect of the systematic is
visible rather than merely applied.  A ratio that survives the ensemble is a
much stronger statement than one that survives only the bootstrap; a ratio that
does **not** survive it comes back `MEASURED_DEPTH_UNRESOLVED`, and that is the
correct answer.  Below `min_members_for_spread` members there is **no** spread:
NaN with `depth_reduction_spread_status = TOO_FEW_MEMBERS`, never 0 — a zero
spread would assert that every reduction agreed when only one was reached.

**What the ensemble does NOT do: it does not move the depth.**  The primary
measurement stays the **deduplicated** one — `dedupe_sectors` keeps one
reduction per sector and the reported depth is unchanged by adding ensemble
members (a test asserts the two are bit-identical).  Re-stacking the members
would count every transit as many times as the archive serves it, which is
exactly the error run 35041932130 made.  **The ensemble sets the error, and
only the error.**

**An inflated error must never buy a refutation.**  `MEASURED_DEPTH_UNCHANGED`
kills a candidate, so it now requires the claimed change to be *excluded* —
`claim_excluded_sigma = (ln_ratio_under_test − |ln ratio observed|) / σ ≥
n_agree` — not merely that `n_agree·σ` is smaller than the claim.  A comparison
that measures a ratio of 2.16 and can no longer call it significant has refuted
nothing; the honest answer is `MEASURED_DEPTH_UNRESOLVED`.  The distinction is
dormant while the errors are small and becomes load-bearing the moment a
reduction systematic is folded in.

**The background test, by number.**  PDC is where the crowding and background
corrections are applied, so `SAP_FLUX` is in the grid deliberately and is never
skipped.  `sap_minus_pdcsap_ppm` pairs the two **within a pipeline author** (so
the aperture is held fixed) and combines the per-author differences by inverse
variance; `sap_minus_pdcsap_z` gives the significance and `background_direction`
the sense.  `PDC_DEEPER_THAN_SAP` beyond `background_n_agree` σ is the case that
*supports* the mundane explanation, and it is reported as a named finding in
`summary.json["reduction_ensemble"]["background_test_finding"]`, not left as a
worry.

The ensemble carries **its own wall-clock budgets** (`stage2.ensemble.budget_s`
and `kepler_budget_s`, plus `per_member_budget_s` / `member_timeout_s`),
separate from the measurement's, because the grid multiplies the number of
requests and a systematic estimate must never be able to consume the budget of
the measurement it exists to qualify.  `stage2.ensemble.enabled` is **true** in
`config/growth.yaml` and **false** in the code's own defaults, for exactly the
reason `kepler_enabled` is; `growth_stage2.yml` **fails the run** if the summary
comes back with the ensemble off, or if a target reached `CHANGED`/`UNCHANGED`
with `reduction_systematic_applied: false`.

Outputs, `results/growth/stage2/`:

| File | Content |
|---|---|
| `probe.json` | which MAST route exists on this machine (imports only) |
| `acquire.json`, `acquisition_log.json` | the ephemeris/TOI pull statuses, the per-target light-curve status and route **for each era**, elapsed time against each budget, every failed attempt's exception text |
| `measurements.csv` | one row per shortlisted target: **both measured depths and the like-for-like verdict**, the KOI-table check, the two catalogue depths, every `z`, the ephemeris record per era, the per-era flags |
| `sectors.csv` | per sector/quarter, with an **`era`** column and a **`scope`** column: `primary` rows are the deduplicated reduction the reported depth came from; `ensemble` rows are each reduction's own per-segment breakdown (`reduction_author`, `reduction_flux_column`), i.e. the same table that made the problem visible at commit `050402a`. Author, exposure, points, transits, depth ± error, `smeared`, `exptime_over_duration` |
| `reductions.csv` | **the reduction ensemble**: one row per (`era`, author, flux column), measured or not, with its depth ± error, transit count, segment list and — when it produced no depth — its own status (`QUERY_FAILED` / `QUERY_RETURNED_ZERO_ROWS` / `FLUX_COLUMN_NOT_PRESENT` / `NO_USABLE_TRANSIT` / `BUDGET_EXHAUSTED`), plus `is_primary_reduction` marking the one the reported depth came from |
| `folds/<KOI>.csv` | the binned fold **per era** (`era` column), for a human to look at |
| `summary.json` | `primary_verdict` (+ `primary_verdict_field`, `primary_verdict_is`), `like_for_like_verdicts`, `koi_catalogue_check`, **`reduction_ensemble`** (the spread definition, how it enters the total error, the per-target members and the background-test finding), then the secondary `verdict` (marked `verdict_is_primary: false`), `target_verdicts`, `unmeasured_reasons`, `flags`, `targets`, `acquisition`, `config`, `checks_not_performed` |

The shortlist and every threshold are in `config/growth.yaml` under `stage2:`
(`shortlist`, `mast`, `ensemble`, `fit`, `compare`), with a `verify:` note on
everything asserted rather than measured.  Run:
`python -m seti.growth.stage2 --stage {probe,measure,assess,all}`
(or `seti growth-stage2 ...`); workflow `growth_stage2.yml`.

### 7.6 What stage 2A cannot say

* **it does not test the centroid.**  A deeper TESS transit cannot come from
  dilution, but a nearby eclipsing binary at the same period inside the pixel
  gives exactly this.  `odd_even_significant` catches one flavour; the
  per-pixel centroid / difference-image test is stage 3.
* **one band per epoch**, exactly as at stage 1: achromaticity is untested.
  Both eras are now *measured*, but each is still a single bandpass, and the
  band ratio that carries one into the other rests on the `verify` LD grid.
* **the ephemeris is not re-derived.**  Both eras are folded on the KOI
  ephemeris, which is itself a Kepler-era product.  Stage 2A measures a depth.
* **`σ_sys,ln = 0.05` is asserted**, not measured (§7.2), and it is applied to
  the like-for-like comparison too — where it is the most conservative term,
  since the two eras go through identical code and the pipeline-to-pipeline
  floor it was written for does not strictly apply.  It is kept rather than
  dropped: keeping it can only widen the agreement band, never manufacture a
  change.
* **it does not re-extract the pixels.**  The reduction ensemble (§7.5a)
  measures every reduction the *archive serves*; it does not build its own
  aperture from the target pixel files.  A systematic common to every delivered
  pipeline — a background model they all share, say — would not appear in the
  spread.  This is listed in `checks_not_performed` as
  `aperture_photometry_of_our_own`.
* **what it no longer cannot do**: the Kepler depth is *not* still the
  catalogue's (§7.0).  `kepler_era_lightcurve_refit` has been removed from
  `checks_not_performed`, because leaving it there would be a false disclaimer.
  And the quoted error is no longer the bootstrap alone: the reduction spread
  enters it as a systematic and the verdict uses the total (§7.5a).

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

---

## 10. Stage 3 — is the transit ON THE TARGET? (`src/seti/growth/centroid.py`, BUILT)

### 10.1 Why, and what it settles

Stage 2A measured Kepler-718 b's TESS-era depth **from the light curve**:
**30,614 ± 1,195 ppm** over sectors 41, 54, 55, 74, 75, 81 and 82, against a
Kepler-era 13,884 ppm carried into the TESS band — a factor 2.2 deeper, and
`MEASURED_DEPTH_MATCHES_TOI`.

**Dilution cannot produce that.**  Extra light in the aperture makes a transit
*shallower*, and TESS's ~21″ pixels admit strictly more light than Kepler's
~4″ ones, so on the same star TESS must read shallower, never deeper.  The one
reading that *does* produce it is that **the two apertures are not measuring the
same source**: if the dimming originates on a neighbour, an aperture centred
differently reports a different depth entirely, and "the depth grew" is a
statement about two different stars.

Three facts make that pressing for this target:

* stage 1 counted **7 Gaia sources within 21″** (`n_gaia_neighbours`);
* the out-of-transit scatter in the TESS photometry is **56,698 ppm — 5.7 %**,
  nearly twice the transit depth, so the aperture is dominated by something
  noisy;
* stage 2A lists `per_pixel_centroid_test` in `summary.json["checks_not_performed"]`.

The odd–even depth difference is **0.20 σ**, so an eclipsing binary at *twice*
the period is already disfavoured.  That does not exclude a source at the
**same** period on a different star.  Stage 3 answers with two measurements.

### 10.2 The neighbour census — the cheapest decisive test

Every Gaia DR3 source within a configurable radius (default one TESS pixel,
21″) with `G`, separation and position angle, and its **flux fraction in the
aperture**

```
frac_i = 10^(−0.4 (G_i − G_target)) · w(d_i) / Σ_j 10^(−0.4 (G_j − G_target)) · w(d_j)
```

with `w` the same erfc capture model stage 1 uses (`drift.aperture_weight`,
`aperture.radius_arcsec` / `psf_sigma_arcsec`).  Then the quantity that settles
most cases **without touching a pixel**: the depth that neighbour would need *in
its own light* to produce the observed aperture depth,

```
required_depth_i = observed_aperture_depth / frac_i .
```

A required depth above 100 % is **arithmetically impossible** — a star cannot
fade by more than all of its light — so that neighbour is `excluded_by_arithmetic`
and the census reports the number that excluded it, per neighbour, rather than a
verdict word.  For a 30,614 ppm aperture depth a source needs
`frac_i ≥ 0.0306`, i.e. roughly **ΔG ≤ 3.7 mag** of the target
(`min_delta_g_that_can_supply_depth`, reported per target); everything fainter
than that is excluded before a byte of pixel data is downloaded.

**What stage 1's committed numbers already imply for K00897.01, before the
census is run at all.**  `candidates.csv` (run 35038510064) carries
`contam_applied = 0.0504` and `contam_max = 0.0746` — the weighted and
unweighted sums of `10^(−0.4 ΔG)` over the seven neighbours.  With a 30,614 ppm
aperture depth:

| From the committed aggregate | Value |
|---|---|
| flux ratio a single neighbour must have | ≥ 0.0329 of the target (ΔG ≤ **3.71 mag**) |
| required depth if **all seven together** did it | **44 %** simultaneous eclipse (not impossible — the ensemble is *not* excluded) |
| how many of the seven can individually clear the bar | **at most 2** (0.0746 / 0.0329), and at most **1** on the weighted sum |

So the arithmetic alone already says the transit source, if it is a neighbour at
all, is **one of at most two of the seven** — every other neighbour is excluded
before any pixel is fetched.  *Which* ones requires the per-source `G`, which is
exactly what the census cone supplies; the sandbox has no Gaia egress, so that
step runs on the runner.

`census_statement` says in one sentence what the arithmetic did and did not
settle.  A Gaia cone that fails is `QUERY_FAILED` and **no neighbour is then
excluded**: an unchecked star is not an isolated one.  Sources below the Gaia
completeness limit are neither implicated nor excluded, and that is in
`checks_not_performed`.

### 10.3 The difference image

Per sector, from the TESS **target pixel files**: a mean in-transit image over
the flat-bottom core (`0.7 T₁₄`, shrunk by half the exposure), a mean
out-of-transit image over the **local** annulus `0.75 T₁₄ ≤ |Δt| ≤ 2.5 T₁₄`
(local, so a slowly varying background enters both the same way), and the
difference `oot − in`, in which the transit source is a **positive peak**.  A
thresholded flux-weighted centroid (pixels above `0.2 ×` the peak) is fitted to
the difference image and to the out-of-transit image, and the **offset between
them** is reported in pixels and arcseconds with an uncertainty — from a
**bootstrap over cadences** when there are enough of them, the per-pixel
propagation otherwise, and `centroid_err_method` always says which.

Three things make the offset honest rather than merely computed:

* **a systematic floor**, `centroid_sys_floor_arcsec = 1.0″` (config, `verify`):
  a thresholded first moment on 21″ pixels cannot be trusted below about an
  arcsecond, and the direct image's faint wings shift its centroid by a fraction
  of a pixel that the difference image does not share.  *Measured on the
  synthetic gate:* an on-target injection gives a **0.124″** offset with a
  **0.05″** cadence-bootstrap error — 2.6 σ of nothing.  The floor is added in
  quadrature per sector **and** to the combined error, because stacking sectors
  averages down what is random, not what every sector shares;
* **the length is debiased.**  `hypot(dx, dy)` is positive by construction, so
  averaging several sectors' magnitudes shrinks the error without shrinking the
  bias and manufactures a significant offset out of noise;
  `offset_debiased = √(max(r² − σx² − σy², 0))` is what the magnitude
  combination uses;
* **sectors are combined in the SKY frame when the file's WCS allows it**
  (`dRA cos δ, dDec`), because two sectors observe the same star at different
  roll angles and their pixel `(dx, dy)` are not the same quantity.  Without a
  WCS the scalar (debiased) magnitudes are combined and
  `combined_from_magnitudes` says so.

Reported **per sector as well as combined**: a consistent offset across seven
sectors is a very different fact from a scattered one, and
`sector_scatter_chi2_per_dof` is the number that tells them apart.  The
out-of-transit centroid is itself pulled toward a bright neighbour, so the
offset measured against it is a **lower bound** on the displacement from the
target; where the WCS gives the target's own pixel, `offset_from_target_*` is
reported beside it.

### 10.4 The verdict vocabulary (`centroid.TARGET_VERDICTS`)

| Verdict | Meaning |
|---|---|
| `TRANSIT_ON_TARGET` | the offset is consistent with zero (`\|offset\|/σ < n_offset`) **and** the measurement could have *seen* the offset that matters: `n_offset · σ ≤` the separation of the nearest neighbour the census could not exclude (or `on_target_max_offset_arcsec` when the census named none) |
| `TRANSIT_OFFSET_FROM_TARGET` | the offset is significant at `n_offset` σ.  The Gaia source it lands on is named, with the depth that source would need — which may itself be impossible, and then the offset points at something Gaia does not resolve |
| `OFFSET_UNRESOLVED` | an offset was measured, its error bar spans zero, and the precision was **not** enough to exclude the neighbour that matters — or no centroid could be fitted.  **Never reported as `TRANSIT_ON_TARGET`** |
| `NO_DATA_REACHED` | no pixels.  `reason` ∈ {`QUERY_FAILED`, `QUERY_RETURNED_ZERO_ROWS`, `EPHEMERIS_UNAVAILABLE`, `NO_TRANSIT_CADENCES`, `BUDGET_EXHAUSTED`, `NOT_ATTEMPTED`} |

`QUERY_FAILED` (the archive errored) and `QUERY_RETURNED_ZERO_ROWS` (it answered
with nothing) stay different facts, and **a target whose pixels could not be
fetched is never reported as on target** — the workflow fails the run if a
`TRANSIT_ON_TARGET` row carries a detectable offset larger than the resolution
it claims, or a `NO_DATA_REACHED` row carries no reason.  The run verdict is the
same vocabulary: any target off target ⇒ `TRANSIT_OFFSET_FROM_TARGET`; all
measured targets on target ⇒ `TRANSIT_ON_TARGET`; otherwise
`OFFSET_UNRESOLVED`, and `NO_DATA_REACHED` when nothing was measured.

### 10.5 The offline gate

`tests/test_growth_centroid.py`, network-free like the rest of the suite.  The
load-bearing test is a **synthetic 11×11 pixel stamp with the transit injected
on a KNOWN pixel** (`centroid.synth_tpf`, a Gaussian-PRF stand-in integrated
over the ephemeris):

| Injection | Measured |
|---|---|
| transit on the **target** at pixel (5, 5) | difference-image centroid **(5.0002, 5.0002)**; offset from the out-of-transit centroid **0.0059 px = 0.124″ ± 1.00″ = 0.12 σ** — consistent with zero |
| transit on a **neighbour 3 px east** at (8, 5) | difference-image centroid **x = 7.998 ± 0.0023 px** (the injected 8.000, within its own error); offset from the target's pixel **2.998 px = 62.96″**; offset from the out-of-transit centroid **2.74 px = 57.6″ ± 1.0″ = 57 σ** |

The suite also gates: the census arithmetic (a 5-mag-fainter neighbour is
excluded and the >100 % depth it would have needed is reported; a 1-mag-fainter
one is *not* excluded); a failed fetch is `QUERY_FAILED` and never an on-target
verdict, with `QUERY_RETURNED_ZERO_ROWS` kept apart; an error bar spanning zero
without the precision is `OFFSET_UNRESOLVED`; seven on-target sectors combined
do not manufacture an offset; the per-target sector cap and the wall-clock
budget are honoured (`BUDGET_EXHAUSTED`, never a depth).

### 10.6 Data access, budgets, outputs

Gaia comes through the **existing** route — `acquire.gaia_neighbours_cones` over
`acquire.pyvo_sync_transport` against `gaia.tap_url` — with
`centroid.census.cone_budget_s` / `cone_timeout_s` bounding it exactly as
`gaia.cone_budget_s` does.  Target pixel files come from MAST, `lightkurve`
(`search_targetpixelfile`) where it is installed and `astroquery.mast`
`query_criteria → get_product_list → download_file(dataURI)` with the FITS read
directly otherwise (`TIME` carried through the file's own `BJDREFI`/`BJDREFF`,
`QUALITY != 0` masked, the `APERTURE` extension's WCS supplying `sky_fn` and the
target's pixel).  Which route the machine has is established at runtime and
recorded in `centroid/probe.json`.

**Target pixel files are large** (tens of MB per sector), so the fetch carries
both a wall-clock budget (`centroid.tpf.budget_s`, `per_target_budget_s`) and a
**per-target sector cap** (`max_sectors`, default 8) — the `gaia.cone_budget_s`
discipline and the same reason: run 34789826297 sat three hours in an unbounded
fetch and would have been killed with nothing committed.

Outputs, `results/growth/centroid/`:

| File | Content |
|---|---|
| `probe.json` | which routes to MAST and Gaia exist here (imports only) |
| `census.csv` | one row per Gaia source: `G`, separation, position angle, flux fraction, **required depth**, `excluded_by_arithmetic` and the note that says why |
| `census.json`, `targets.csv` | the per-target census summary and statement; the ephemeris, position and observed depth used |
| `sectors.csv` | per sector: cadence counts, both centroids, the offset in px and arcsec with its error, `centroid_err_method`, `offset_from_target_*` |
| `offsets.csv` | per target: the combined offset, sector scatter, fetch status and route |
| `verdicts.csv`, `summary.json` | the verdict with the numbers that justify it, `checks_not_performed`, the config used |
| `acquire.json`, `acquisition_log.json` | every fetch's status and every failed attempt's exception text, elapsed against the budget |

Run: `python -m seti.growth.centroid --stage {probe,census,difference,assess,all}`
(or `seti growth-centroid ...`); workflow `growth_centroid.yml`.

### 10.7 What stage 3 cannot say

* **it is not a PRF fit.**  The centroid is a thresholded first moment, not a
  fitted TESS PRF; a PRF fit would tighten the offset error, not change its
  sign, and the systematic floor is there because the moment is approximate.
* **Gaia's completeness is the census's completeness.**  An unresolved source
  inside the pixel is in neither the census nor the exclusions.
* **only the TESS pixels are tested.**  The Kepler-era pixel centroid (and the
  `koi_fpflag_co` centroid flag, which is 0 for this target) is a separate
  statement.
* `centroid_sys_floor_arcsec` is **asserted**, not measured on real TESS data.

---

## 11. Stage `direct` — the TESS depth of EVERY Kepler planet, from the light curves (`src/seti/growth/direct.py`, BUILT)

### 11.1 Why this stage exists

Stage 1 measured **108 of 9,564 KOIs** — 1.1 %. The bottleneck is not TESS
coverage; it is the *join*. Stage 1 reached the TESS era through the **TOI
alert catalogue**, and 3,055 KOIs resolve a TIC id while only **94 of their
1,975 distinct stars ever had a TOI at all**. TOI is an alert list, not a
systematic re-measurement of Kepler's planets, so leaning on it cost 99 % of
the sample *and* imported whichever pipeline produced each alert — the same
heterogeneity that broke stage 1's error model (31 of 108 planets above 5σ of
the population median; a third of a sample cannot be five-sigma outliers).

This stage removes the catalogue from the measurement entirely. For every
confirmed or candidate KOI with a TIC id it fetches **every TESS light-curve
product MAST serves** and fits the depth itself, with the same code that fits
the Kepler era in stage 2A. Sample and error model are fixed by the same move.

### 11.2 What it does, per star

1. **Fetch, once per star.** `search_lightcurve("TIC …", mission="TESS")`
   through `lightkurve`, falling back to `astroquery.mast` + FITS. Authors
   `SPOC` (2-min), `TESS-SPOC` and `QLP` (FFI) — the author filter is
   **strict**: a request that matches nothing returns nothing, never another
   pipeline's products. (The `AUTHOR_NOT_SERVED` lesson: a dropped author
   filter once returned SPOC's products under a QLP request and put a phantom
   duplicate into the reduction ensemble, shrinking the very error the ensemble
   exists to size.) 20-second products are skipped; every flux column of a
   downloaded product is read, so the SAP and PDCSAP families cost **one**
   fetch, not two. A multi-planet system's planets share the download.
2. **Propagate the ephemeris and search phase.** `t0` is carried from
   `koi_time0bk` over the ~2,000 epochs to the TESS window; the accumulated
   `σ_T0 = n · σ_P ⊕ σ_T0` is often larger than the transit. A window of
   `3 σ_T0 + 0.25 T14` (capped) is scanned coarsely then finely, ≤ 161 trials;
   the offset, its significance and the trial count are all on the record. The
   transit is `ephemeris_recovered` only when the best-offset depth reaches
   S/N ≥ 3 — otherwise the planet is `transit_not_recovered`, **never**
   `consistent`.
3. **Fit both families with the stage-2 fitter.** Same masked baseline, same
   exposure-shrunk core, same bootstrap, same `dedupe_sectors`, run over the
   **raw** `SAP_FLUX` family and the **corrected** `PDCSAP_FLUX`/`KSPSAP`/`DET`
   family. The spread over pipeline *authors* within a family is the
   reduction-ensemble systematic and enters `total_err_ppm`.
4. **Profile the duration** on the detrended fold (wider baseline window than
   the depth fitter's, so a duration that *grew* is not masked into the
   baseline) and test `T14` against the fixed-impact-parameter prediction.
5. **Compare like for like** against `koi_depth` carried into the TESS band by
   the limb-darkening ratio, and state **per planet what change could have been
   seen**: `detectable_ln_ratio = n_candidate · σ_expected` and
   `detectable_depth_change_ppm`. A non-detection on a star is then an honest
   statement about that star's TESS precision, not silence.

### 11.3 The Kepler-718 b rule, as a gate

A bigger aperture admits more contaminating light, so on the same star the
**raw SAP depth must read shallower than Kepler, never deeper**. Kepler-718 b
read SAP 11,196 ppm against PDCSAP 29,255 ppm — a factor 2.61 between raw and
corrected photometry of the *same photons* — and the difference image put the
transit source 29.25 ± 4.18″ off target. The "growth" was PDC dividing the raw
depth by a crowding factor the TIC got wrong.

So `classify_direct` requires, for `growth_candidate`:

* **both** families up at `n_candidate` = 5σ on the **total** error, the SAP on
  its own raw `z` (a deeper-than-Kepler raw depth is never dilution) and the
  PDCSAP on its population-corrected, scatter-scaled `z`;
* `duration_verdict == fixed_b`; no odd-even signature at 3σ; the transit
  recovered at its ephemeris; no `koi_fpflag_*`; disposition still
  CONFIRMED/CANDIDATE; neither family's depth a lower bound.

A PDCSAP-only rise with `sap_minus_pdcsap_z ≤ −3` and direction
`PDC_DEEPER_THAN_SAP` is classed **`crowding_correction`** — Kepler-718 b's
class — and is not a candidate.

**What the SAP rule does *not* buy, stated plainly.** Dilution can only make a
depth shallower, so a raw SAP depth *deeper* than Kepler's is not dilution of
the target's own transit. It is **not** proof that the signal is on the target:
a blended eclipsing binary contributes a fractional depth of (its eclipse depth
× its flux) / (total aperture flux), and if the binary is deep and bright
enough that product can exceed the target's own Kepler depth in the raw
photometry too. The SAP rule kills the *crowding-correction* artefact — the
specific way Kepler-718 b was manufactured — and nothing more. The test that
settles where the signal is remains stage 3's difference image, which is why
`vet` runs it on every survivor. A TIC resolved only by the weakest route
(nearest source in the cone with a compatible Tmag) carries
`tic_identified_by_position_only` for the same reason. `deeper_tess` / `shallower_tess` hold the
one-family changes; `consistent` is only ever written where the sensitivity was
sufficient to see the change.

### 11.4 The error model, corrected in the population's own units

`population_offsets` measures, per family, the median `ln(D/D_ref)` (the band
and aperture offset that must be removed before anything is compared) and the
**robust scatter of z about it** (`1.4826 · MAD`). When that scatter exceeds
one, every `z` is divided by it before the gate: a candidate has to be 5σ in
the units the population's own scatter defines, which is the check stage 1
failed. Both numbers, and the counts above ±3 and ±5, are in `summary.json`.

### 11.5 Bookkeeping that is never a statement about the sky

**How much of the phase the search can actually cover.** Computed on the
committed `targets.csv` (4,604 of 4,725 targets carry the full ephemeris
arithmetic; TESS mid-epoch BTJD 2600): the median planet is propagated over
**353 epochs** (maximum 17,839) and arrives with **σ_T0 = 38 min**; p90 is
168 min and p99 is 692 min. Against that, the search half-width
`3σ_T0 + 0.25 T14` capped at `min(0.6 d, 0.3 P)` is 180 min at the median.
The cap bites for **269 targets (5.8 %)** — 204 of them long-period — where the
window is narrower than 3σ (median shortfall a factor 1.6, worst 73), and for
**6 short-period targets σ_T0 exceeds P/2**, so their phase is unconstrained
outright. A `transit_not_recovered` on any of those is a statement about the
*ephemeris*, not about the sky, and `epoch_search_halfwidth_minutes` next to
`ephemeris_sigma_minutes` says which case each planet is.

`not_measured` keeps `QUERY_FAILED`, `QUERY_RETURNED_ZERO_ROWS`,
`TIC_UNRESOLVED`, `EPHEMERIS_UNAVAILABLE`, `NO_USABLE_TRANSIT`,
`BUDGET_EXHAUSTED` and `NOT_REACHED` apart. A shard the clock killed leaves its
targets `NOT_REACHED` — absent from the shard CSV, marked by the aggregate —
and never `consistent`. `not_measurable` says TESS could not have seen the KOI
depth at all. None of these is a null result and none is written up.

### 11.6 The look-elsewhere null (`control` stage)

The epoch search of §11.2 step 2 takes the **maximum** fitted signal-to-noise
over up to 161 offsets, and the depth is then fitted *at the winning offset*. A
maximum over trials is biased upward, and the bias grows as the signal weakens
— which is exactly the regime that would manufacture a spuriously **deeper**
TESS depth, the direction a growth candidate lives in. Nothing in the depth's
bootstrap error, the reduction-ensemble spread or the population scaling knows
about it.

So the `control` stage re-opens every planet whose class was
`growth_candidate`, `shrink_candidate`, `deeper_tess` or `crowding_correction`
— `consistent` needs no null, since the bias can only push a depth up and
therefore cannot manufacture agreement — re-fetches the star, and repeats the
**identical** search (same half-width, same coarse and fine steps, same fitter,
same deduplicated segments) centred on `epoch_search.control_phases` of the
period, where the planet is not. What it finds there is what the search
produces from that star's own noise:

* `control_snr_max` — the best signal-to-noise the search reached off-transit;
* `control_depth_max_ppm` — the deepest depth it fitted there;
* `snr_excess = epoch_search_snr_best − control_snr_max`;
* `depth_excess_over_control_ppm = (D_TESS − D_ref) − max(control_depth_max, 0)`
  — a "growth" smaller than this is not a growth, it is the search.

The verdict per family is `ABOVE_CONTROL` or `WITHIN_SEARCH_NOISE`, and the
planet takes the **weaker** of its two families: a change the search can
manufacture in either reduction is not a change. A star the stage could not
reach is `CONTROL_UNAVAILABLE` — never a pass. `direct_vet` requires the
survivor not be `WITHIN_SEARCH_NOISE`, and a survivor whose null has not been
run carries `CONTROL_NOT_RUN` as an open systematic, counted in
`n_control_not_run`.

Two caveats are on the record rather than buried: in a multi-planet system a
control phase can land on a **sibling's** transit, which makes the null
conservative rather than permissive; and a control phase that lands in a data
gap returns nothing and is not counted (`control_n_phases_measured`).

### 11.7 Vetting every survivor, not the top of the list

Stage 2 (both eras refitted with one fitter, plus the reduction ensemble) and
stage 3 (the Gaia census and the difference image) each cost several fetch
budgets per target, so one job carries at most `classify.vet_max_targets` of
them. With a candidate list longer than that, taking "the strongest N" would
leave the rest with the difference-image question — the question that killed
Kepler-718 b — never asked.

So the vet stage **shards**. The candidates are ranked by |z| and split
**round-robin by rank** across `vet_shards` jobs, so shard 0 gets ranks
0, N, 2N…, shard 1 gets 1, N+1, … — every shard carries a mix of strong and
weak candidates, and no shard is the "leftovers" job. Each writes
`results/growth/direct/vet/shard_NN/`; `vet-gather` merges them into
`vet/vetted.csv` and `vet/summary.json` and reports
`n_candidates_not_vetted` **with identifiers**: a candidate nobody ran stage 3
on is an open question, never a pass.

### 11.8 Running it

`growth_direct.yml`: `targets` → `measure` (sharded **by star**, `kepid mod n`,
so a system's planets share one download; each shard checkpoints its CSV after
every star and commits it `if: always()`) → `assess` → `vet` (stage 2 both eras
+ stage 3 census and difference image on every survivor). A shard cut off by
its budget is resumed by re-dispatching with `resume=true`, which skips the
targets already committed. Planets with `koi_period > 30 d` are carried
separately in `long_period.csv`: TESS's 27-day sectors give them few or no
transits and their sensitivity is stated, not assumed.
