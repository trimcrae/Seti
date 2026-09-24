# CRADLE — the shattered cradle (S52) and slag, not glass (S53)

**Warm debris at the habitable-zone radius of a MATURE star, above the
collisional steady-state maximum.**

A destroyed or disassembled planet leaves warm dust at its orbit. Nature makes
extreme debris disks too — 17 in Moór et al. 2021, 21 with spectra in Su et al.
2026, all sitting 10⁴–10⁶ above the collisional maximum — but they are **young**
(a few tens of Myr, the terrestrial-planet-formation epoch) or, in the two
mature cases, **hot**: BD+20 307 (~1 Gyr, ~400–450 K) and TYC 4479-3-1
(5 ± 2 Gyr, ~400 K). **No extreme debris disk at 250–350 K around a star older
than 1 Gyr is known.** That cell — the blackbody temperature of the habitable
zone, on a star old enough that primordial planet formation is long over — is
what this channel searches. It is empty in the literature, and it is empty for
a reason: nothing natural is supposed to put a planet's worth of dust at 1 AU
around a 5 Gyr star.

S53 is the spectral stage: hypervelocity collision debris is silica glass
**with** crystalline forsterite/enstatite, FeS and SiO (HD 172555, HD 23514,
HD 15407A are the templates). Refined material — smelter slag, structural
ceramics — is CaO–Al₂O₃–SiO₂ glass with the Fe, Mg and S phases stripped. The
target is therefore a *featureless-to-silica-only* 8–13 µm band on a star that
has a strong warm excess.

## The statistic

For every star, from the W3/W4 excess over an **empirical** photosphere:

| quantity | definition |
|---|---|
| `t_bb_k` | single-temperature blackbody `F_ν,exc = Ω_d B_ν(T)` fitted to the W2/W3/W4 excess fluxes on a 400-point log grid, parabolically refined |
| `f_ir` | `f = L_IR/L_* = (Ω_d σ T⁴/π) / F_bol`, with `F_bol = L_*/4πd²` |
| `r_bb_au` | `r = (278.3 K / T_bb)² √(L_*/L_☉)` AU |
| `fmax_1gyr` | Wyatt et al. 2007 eq. 20 in Moór's catalogue form: `f_max = 0.16×10⁻³ r^{7/3} M_*^{−5/6} L_*^{−1/2} t^{−1}` (r in AU, t in Myr) |
| `log_f_fmax_1gyr` | `log₁₀(f / f_max)` evaluated at **t = 1000 Myr** |

`t = 1000 Myr` is the *youngest* age the cell admits, and `f_max ∝ 1/t`, so
`log_f_fmax_1gyr` is a **lower bound** on `log(f/f_max)` for every star that
really is older than 1 Gyr. A star that clears 3 dex at 1 Gyr clears it at
every age the cell allows. `log_f_fmax_adopted` uses the star's own adopted
age where two indicators give one.

The photosphere is not a synthetic atmosphere. It is the robust running median
of `K_s − W_i` against `BP − RP` fitted on the parent sample's own bright,
clean stars (`locus_ks_max = 7.5`, where a bare photosphere is still a ≥ 5σ W4
detection — at fainter K_s the W4-detected subsample is *selected for excess*
and its median would be biased). `K_s − W1 < 0.2` is required so the anchor
itself is not already contaminated. The locus refuses to extrapolate: a star
bluer or redder than any well-populated colour bin gets no photosphere and is
not assigned one.

**Two different significances, and they are not the same number.** The
archive cut is a *detection* significance — W3 and W4 each measured at ≥ 5σ,
spelled `wXmpro_error < 1.0857/5` because the ESA mirror carries magnitude
errors and no SNR column. The screen then applies an *excess* significance —
`chi_W3 ≥ 3` **and** `chi_W4 ≥ 3` over the empirical photosphere, with the
star-to-star scatter of the locus, the photometric error and a 0.03 mag
systematic floor all in the denominator. A star can be a 20σ W4 *detection*
and a 0σ W4 *excess*; only the second is a disk.

**Two windows.** `shortlist.csv` is deliberately wider than the cell —
180–450 K, so that the hot and cold shoulders are enriched and reported and
the cell's edges are visible rather than assumed. The strict 250–350 K cell is
applied at `assess`, and stars either side come out `ABOVE_FMAX_HOT` /
`ABOVE_FMAX_COLD`.

**The cell.** A candidate needs all three, simultaneously:

```
250 K ≤ T_bb ≤ 350 K        AND    log(f/f_max) > 3  (16th percentile, not the point estimate)
                            AND    age > 1 Gyr from ≥ 2 INDEPENDENT indicators
```

Colour corrections for a 250–350 K blackbody in W3/W4 (Wright et al. 2010, of
order 5–10 %) are **not** applied: they move `T_bb` by ~10 K and `f` by < 0.05
dex, against a threshold of 3 dex.

## The sample

Gaia DR3 × AllWISE, selected **in the archive**:

* `G < 13.5`, `parallax > 2 mas` (d < 500 pc), `parallax/σ > 10`, `|b| > 10°`,
  `RUWE < 1.4`, `0.45 < BP−RP < 1.85` (~F2–K7), and the absolute-magnitude
  dwarf cut `M_G > 2.0 + 3.3 (BP−RP − 0.6)`;
* AllWISE **W3 AND W4 each ≥ 5σ**, spelled `wXmpro_error < 1.0857/5 = 0.21715`
  because the ESA mirror carries magnitude errors and no SNR column;
* 2MASS `K_s` through `tmass_psc_xsc_best_neighbour` for the anchor;
* `astrophysical_parameters` for FLAME ages/masses/luminosities, GSP-Spec
  [α/Fe], and the ESP-CS activity index; `vari_rotation_modulation` for a
  rotation period.

Sky coverage is exact, not sampled. Gaia DR3 `source_id` carries the level-12
NESTED HEALPix index in its high bits (`source_id // 2³⁵`), so a level-3 pixel
`k` is the contiguous range `k·4⁹·2³⁵ ≤ source_id < (k+1)·4⁹·2³⁵`, served by
the primary key. **768 level-3 pixels are the whole sky**, with no gaps and no
double counting; pixel `k` goes to shard `k mod n`. A unit that times out is
split into its four NESTED children and retried, down to level 6.

The join is written **inner-first** — every Gaia-only cut in a sub-select on
`gaia_source` ALONE, with AllWISE joined to that small result. IGNITION run
34787803862 proved the alternative: a flat `WHERE` over the three-table join
lets the planner start from the 750-million-row AllWISE mirror and cannot
return even `TOP 5`.

### What the archive actually did — run 35741356662, shard 3, 11:46 a.m. EDT

Measured, not assumed:

```
[cradle] s3of8 hp3_763: OK rows=53 parent=2541 shape=full 24.9s
[cradle] acquire s3of8: units ok=86 zero=10 partial=0 failed=0 rows=4990 in 1906.3 s
```

The `full` shape answers — the inner-first join with 2MASS reached *directly*
through `tmass_psc_xsc_best_neighbour.original_ext_source_id` (the spelling the
config marked `verify`), plus `astrophysical_parameters` and
`vari_rotation_modulation` — at **~25 s per level-3 pixel**. IGNITION's
flat-join failure does not recur. 96 of 96 units returned; the 10 empty ones
are the |b| > 10° cut removing Galactic-plane pixels. **4,990 parent stars from
one eighth of the sky**, so the all-sky parent is of order 40,000, and a shard
costs 31.8 minutes against its 300-minute budget.

`screen` is not the bottleneck anyone would guess: timed on 40,000 synthetic
rows, `select_parent` + `harmonise` + `fit_loci` + `excess_table` + `fit_disk`
(the 400-point temperature grid with 300 Monte-Carlo draws per star) is **18
seconds** end to end, against a 120-minute job cap. The expensive stage is
`ages`, where a NEOWISE cone is ~90 s per star, and that is the one with the
wall-clock budget and the priority-ordered shortlist.

## Ages — two indicators or nothing

HD 15407A is the warning written into the mission: a **2.1 Gyr isochrone age
against an 80 Myr AB Doradus membership**. One indicator is not an age. A star
with a single old indicator is `AGE_UNDETERMINED` and is **never** a candidate.

| indicator | what it is | can say OLD? |
|---|---|---|
| `iso` | Gaia DR3 FLAME isochrone age, the **16th-percentile bound** `age_flame_lower > 1 Gyr`, not the point estimate | yes |
| `kin` | the age–velocity relation as a likelihood ratio: the star's measured velocity components (full U,V,W with an RV, the tangential 2-vector without) against the Aumer & Binney 2009 dispersion tensor with the asymmetric drift, marginalised over a flat SFH separately below and above 1 Gyr. `log₁₀ LR ≥ +1` is OLD | yes |
| `alpha` | GSP-Spec [α/Fe]: a thick-disc star (`α_lower > 0.15`, [M/H] < 0) is > 8 Gyr — chemistry, independent of both | yes |
| `gyro` | a rotation period through the Mamajek & Hillenbrand 2008 gyrochrone | mostly a YOUTH detector |
| `act` | ESP-CS Ca II IRT activity index: high is YOUNG | **no** — the inactive locus is not a calibrated age |

**What this costs, stated up front.** Requiring two indicators means most
in-cell stars will land in `IN_CELL_AGE_UNDETERMINED`, not in `CANDIDATE`. The
kinematic test certifies age only by **excess** velocity, so a genuinely old
star on a quiet orbit is `UNDETERMINED`, never young. GSP-Spec [α/Fe] exists
only for the brighter half of the parent. Gyrochronology's catalogue coverage
is two Kepler/K2 fields — a few hundred square degrees against an all-sky
parent — and Gaia's own `vari_rotation_modulation` is populated by active
stars, so it detects youth and rarely age. `IN_CELL_AGE_UNDETERMINED` is
therefore the expected *modal* outcome for an interesting star, and it is a
statement about the age data, not about the star: it names exactly what a
follow-up spectrum would have to supply. Two vetoes override the count entirely: membership
of a young moving group (21 groups in XYZ + UVW, 3-D with an RV and the
tangential projection without, Sco–Cen subgroups included) and a
star-forming-region sky box at the region's parallax range.

## The kill list

Every rule has a name, a counter and a reason string, and a rule whose column
is missing is recorded `untested`, never silently passed.

| contaminant | what kills it |
|---|---|
| **background dusty galaxies in the 6–12″ WISE PSF** — *every* Project Hephaistos candidate | `ext_flag`; AllWISE `n_mates` (another Gaia source shares the WISE source); Gaia beam neighbours from a per-star cone at the W3 and W4 FWHM; IRSA's profile-fit `w3rchi2`/`w4rchi2`; the W1−W2 AGN and W1−W2/W2−W3 galaxy colours; SIMBAD object type |
| **cirrus at W4** | W3 **and** W4 must be in excess at the *same* temperature (the SED fit does this); `cc_flags` clean in W3/W4; SFD E(B−V) gate |
| **mis-aged Sco–Cen / ρ Oph stars** | the region sky-boxes at the region parallax, plus the moving-group veto in position and kinematics |
| **wide companions** (comet delivery — 5/6 of Moór's hosts; BD+20 307's WD companion) | Gaia sources within 1000 AU that are parallax- and PM-consistent; `non_single_star`; RUWE |
| **mantle-only impact debris** | *cannot* be excluded photometrically. It is the S53 mineralogy's job, and it is listed by name among the systematics NOT excluded for every candidate without an IRS spectrum |

NEOWISE W1/W2 epoch-binned variability is carried as a **descriptor**, not a
cut: 14 of 17 natural extreme debris disks vary.

## S53 — the mineralogy score

Where a Spitzer/IRS low-resolution spectrum exists in IRSA's `irs_enhv211`
(the CASSIS replacement the probe located), a local linear continuum is fitted
across anchor windows either side of each feature and the contrast
`F/F_cont − 1` integrated over the band:

* **10 µm silicate**: anchors 7.4–7.9 and 12.7–13.4 µm, band 9.0–11.8 µm; plus
  the peak wavelength, the 11.3/9.8 µm ratio (crystalline forsterite) and the
  8.9–9.3 µm contrast (silica);
* **18 µm silicate**: anchors 15.0–16.0 and 21.0–22.5, band 16.5–20.0;
* **23 µm FeS**: anchors 21.0–22.5 and 25.5–27.0, band 23.0–24.5.

`SILICATE_FEATURED` (contrast > 0.10 at > 3σ) → `NATURAL_SILICATE_*`.
`FEATURELESS` (|contrast| < 0.05 with σ < 0.05) on a star with a significant
excess → `SLAG_CONSISTENT`. **Two caveats the verdict string carries and this
document states plainly:** grains larger than ~5 µm suppress the features too,
and **no laboratory optical constants for Ca–Al slag glass exist** to fit
against. The score is a discriminant between *silicate-featured* and
*featureless*, not a positive identification of slag.

## Controls

Three known mature/extreme debris disks are pulled through the same join with
the W3/W4 SNR cut switched off, and `controls.json` records where each one
lands. They are never candidates (`CONTROL:` prefixes their class).

| star | literature | expected landing |
|---|---|---|
| BD+20 307 | ~1 Gyr, T_dust ~400–450 K, f ~3×10⁻², WD companion | `CONTROL:ABOVE_FMAX_HOT` — far above f_max but **hotter than the cell** |
| TYC 4479-3-1 | 5 ± 2 Gyr, ~400 K (Moór+2021) | above f_max, hotter than the cell; no asserted position, resolved by name only |
| HD 15407A | 2.1 Gyr isochrone vs 80 Myr AB Dor membership, ~500–800 K | the **mis-age warning**: hot, and the age machinery must not call it mature on the isochrone alone |

A control landing somewhere else is a statement about this pipeline, and the
`controls.json` entry carries the literature expectation next to the measured
class so the disagreement is visible rather than buried.

## Stages and outputs

`seti cradle --stage {probe,acquire,screen,ages,assess,all} --shard i/n`
(equivalently `python -m seti.cradle.run`), driven by
`.github/workflows/cradle.yml`.

| stage | what it does | writes |
|---|---|---|
| `probe` | one minimal live call per route: the ESA mirrors' column names, each join **shape** on one HEALPix unit under a wall-clock budget, the three controls, IRSA's AllWISE and `irs_enhv211` columns, a NEOWISE cone, each VizieR table's existence | `probe.json` |
| `acquire` | shard `i/n` of the 768 HEALPix units, checkpointed per unit, timed-out units split; shard 0 also pulls the controls | `parent_s{i}of{n}.csv`, `acquire_s{i}of{n}.json` |
| `screen` | every shard merged → empirical loci → W3/W4 excess → T_bb, f, log(f/f_max) | `shortlist.csv`, `screen.json`, `parent_screened.csv` |
| `ages` | shard `i/m` of the shortlist: Gaia neighbours, IRSA profile fits, SIMBAD, E(B−V), NEOWISE, VizieR catalogues, Spitzer/IRS + S53, then the age indicators | `enriched_s{i}of{m}.csv`, `ages_s{i}of{m}.json` |
| `assess` | the kill list, the classes, the controls | `candidates.csv`, `shortlist_vetted.csv`, `controls.json`, `summary.json` |

### The offline gate is not the runner's environment

`pyproject.toml` asks for `pandas>=2.0`. The sandbox venv has **2.3.3**; the
runner's `pip install -e ".[dev]"` fetches **3.0.6**. A green local suite is
therefore not a green gate: run 35741356662's `probe` job died on its own
offline gate at 11:05 a.m. EDT, before one archive call, because
`parse_ipac_table` used `pd.to_numeric(errors="ignore")` — deprecated in
pandas 2, **removed in pandas 3, where it raises**. To reproduce the runner
without touching the shared venv:

```
pip install --target /tmp/pd3 --no-deps "pandas>=3"
PYTHONPATH=/tmp/pd3:src pytest tests/test_cradle.py -q
```

The suite is kept green under both.

### Shard economics — two ways a runner slot is lost, and what stops them

Neither of these changes a number; both decide whether the numbers are ever
measured at all.

**A wrong shape must cost one unit, not ninety-six.** The probe measures the
join shapes on *one* HEALPix pixel and `acquire` reads its answer from
`probe.json`. If that answer is wrong for some other part of the sky, or the
artifact never reaches the job (the download is `continue-on-error`), the
ladder would re-pay the failing shape's timeout on every unit of the shard.
The acquire loop therefore **learns**: the shape that actually answered is
promoted to the front for the units that follow, and the rollup records
`shapes_planned`, the running `shapes` order and every `shape_relearned`
event, so the log says which shape the sky really wanted.

**One pathological pixel must not cost the whole job.** A timed-out unit
splits into its four children, recursively to `healpix_split_max_level = 6`.
Unbounded that is `1 + 4 + 16 + 64 = 85` queries at `query_timeout_s = 1200 s`
— 28 hours inside a 350-minute job, which would lose every unit the shard had
not yet reached. `fetch_unit` now takes the shard's own deadline and carries
it through the recursion; past it the split is abandoned, the record carries
`deadline_exceeded`, and the count reaches `screen` as
`coverage.n_units_deadline_exceeded` and `assess` as a `DEGRADED` reason.
Units already done are checkpointed, so what is lost is bounded and named.

### A refused photosphere is not a quiet star

The locus refuses to extrapolate, so a star outside every well-populated colour
bin gets no predicted photosphere and a NaN `chi` — and then fails
`excess_significant` for exactly the reason a star with no excess does. On a
partial sky the bins are thin, so that silence would read as a clean null when
it is a coverage statement. The screen funnel therefore counts
`n_photosphere_assigned` and `n_no_photosphere_locus_refused` apart, and
`assess` raises `DEGRADED (locus_refused_photosphere:n/m)` when more than a
fifth of the K_s-bearing stars were never placed.

### Classes

`CANDIDATE` (in the cell, ≥ 2 old indicators, no kill) ·
`IN_CELL_AGE_UNDETERMINED` (in the cell photometrically, awaiting a second
independent age indicator) · `IN_CELL_KILLED` (with `kill_reasons`) ·
`ABOVE_FMAX_HOT` / `ABOVE_FMAX_COLD` · `BELOW_FMAX` · `NOT_SIGNIFICANT` ·
`KS_MISSING` · `CONTROL:` prefix.

### Verdicts

`NO_DATA_REACHED` — no parent rows came back; `reason` names the route. **This
is never a statement about the sky.**
`NO_CRADLE_CANDIDATE` — stars were screened; none is in the cell, mature and
unkilled. A **count over what was actually searched** (`stage_counts`), with
the in-cell / age-undetermined / killed breakdown. Never an occurrence limit.
`CRADLE_CANDIDATES` — ≥ 1 star in the cell with ≥ 2 old indicators and no kill.
A `DEGRADED (…)` prefix names missing shards, failed units and untested vetoes.

## Run 35741356662 and the deep vet (2026-09-23)

### What the run did

The run went red because of **one job**: `probe` failed its offline gate at
15:05Z (11:05 a.m. EDT) on `pd.to_numeric(errors="ignore")` under pandas 3
(already fixed in `mineralogy.py`). No other job needed the probe: all 8
`acquire` shards (768/768 HEALPix units, 0 failed, 0 past the deadline, 39,763
parent stars), `screen`, all 4 `ages` shards (523 shortlisted stars) and
`assess` succeeded, and their files are self-consistent. The class counts add
up to 523. 103 stars are in the cell (66 killed, 25 age-undetermined, 12
mature), and the T_bb histogram sums to 916 = `n_above_fmax_3dex`. The one
labelling quirk is that `ages_s*.json`'s `generated_utc` is the stage's
*start*, not its write time: shards 0 and 1 are stamped 22:41Z and finished
at 02:25Z. Shards 0 and 1 also have identical age-class counts (89/30/12),
but their neighbour, SIMBAD and NEOWISE counts differ, so they processed
different stars.

**The checkout-inherited checkpoint bug (fixed on main in d4b74539) did not
touch this run.** The run's checkout, 16956052, contained no
`results/cradle/` files at all, so no shard could resume from an inherited
`acquire_s*of8.json`. The shards' `n_rows_this_run` values (5010 + 4871 + 4975
+ 4990 + 5013 + 4952 + 4930 + 5022) sum to exactly the 39,763 parent rows. Each
shard's `done` list holds 96 `hp3_*` units, and shard 0's 97th entry is the
`controls` unit.

**Verdict: `DEGRADED (vetoes_untested_in_cell:simbad_type=24); CRADLE_CANDIDATES` — 12 candidates.**

The SIMBAD veto was "untested" for 165 stars because
`vigil.acquire.fetch_simbad_type` returns `""` both for "nothing within 5″"
and for "the query raised". `deepvet` asks SIMBAD by TAP, where an empty cone
is a *tested* answer and a failed call an untested one.

### The deep vet — `python -m seti.cradle.deepvet` (`stage=deepvet`)

This stage reads the committed `summary.json`/`shortlist.csv` and does not
re-screen anything. It runs seven checks per target: SIMBAD (types, otypes,
refs, neighbours within 15″); the full AllWISE row (nb, `w3nm/w3m`,
`w4nm/w4m`, aperture-vs-profile, neighbours within 30″); the local chance-blend
rate from every AllWISE W3/W4 source in 10′; Legacy Surveys Tractor deblended
forced W3/W4; Gaia neighbours of every magnitude within 15″; VizieR (the full
6″ footprint plus 12″ cones on VSX, WDS, Gaia variability, 2MASX and the
published WISE-excess catalogues); unWISE W3/W4-vs-W1 centroids; and a
DESI/SDSS spectrum (Li 6708, Hα) via SPARCL.

Deep-vet dispatches, all on `claude/handoff-cradle`:

- 35860601552 was the first full pass. Its centroid significances had no
  registration floor, which made them overstated.
- 35864699584 cut every unWISE fetch off at 600 s. It then overwrote the
  complete output with an 8-target partial.
- 35868341936 and 35877629887 never ran. They died at the offline gate:
  sparclclient's `pandas==2.1.1` is ABI-incompatible with numpy 2.3.
- 35884499464 (16:27Z) was the first complete pass: 38 targets, nothing
  skipped. Its DESI check failed on a missing `jwt`.
- 35891332109 (17:26Z) reproduced every verdict. Its DESI check failed on a
  missing `specutils`.
- 35896124736 (18:13Z) reached SPARCL. Both DESI spectra then raised on a
  numpy truth-test bug.
- **35901988646 is the pass of record** (19:08Z, 3:08 p.m. EDT): 38 targets,
  nothing skipped, and verdicts identical to the three passes before it.

The W3-vs-W1 centroid floor, measured from 24 field stars in the candidates'
own unWISE cutouts, is **0.39″ per axis** (median 2-D offset 0.46″). W4 had
too few field sources, so it uses a conservative 0.90″.

**The SIMBAD veto is closed.** All 12 candidates were tested. Three have no
SIMBAD object within 15″: 6225457312033584384, 3168144078565656832 and
5295632592220066688. None of the 12 carries a young, evolved or galaxy type.
One of the 25 age-undetermined stars (1722706885596251264) sits on UGC 10222,
a galaxy.

| Gaia DR3 | T_bb (K) | log f/f_max | fate | mechanism (numbers) |
|---|---|---|---|---|
| 1394698167320555648 | 268 | 4.28 | **killed: galaxy blend** | W3 centroid 1.84″ off (4.7σ), W4 5.14″ off (5.2σ). An SDSS galaxy-morphology entry at 4.2″ and a parallax-less G = 21.5 Gaia source at 3.6″. W4 SNR 5.5, detected in 1/42 frames |
| 1276273278883611264 | 276 | 4.11 | **killed: galaxy blend** | W3 2.53″ off (6.4σ), W4 4.15″ off (4.4σ). W3 aperture 0.20 mag brighter than the profile fit (5.1σ). Deblended LS forced photometry gives W3 0.62× AllWISE and **no W4 excess (χ = −0.7)**. Gaia/SDSS galaxy candidates and a DESI entry within 6″ |
| 2415134847267401472 | 282 | 4.00 | **killed: blend + youth** | W3 1.26″ off (3.2σ), W4 4.61″ off (5.0σ). LS forced: no W4 excess (χ = −0.4), W3 0.69×. Listed in Žerjal+2017 young active RAVE dwarfs, so the "mature" age is contradicted |
| 4659120294392894848 | 263 | 3.28 | **killed: blend** | AllWISE nb = 2, with a W3-bright neighbour (W3 = 9.90, SNR 24) at 4.0″. That neighbour's W4 = 8.47 vs the star's 8.09. LMC outskirts |
| 4926561345188246400 | 281 | 3.51 | **killed: W4 confusion** | AllWISE neighbour at 4.6″ with W4 = 8.88 vs the star's 8.20, inside the 12″ beam |
| 5054402421143189248 | 258 | 3.37 | **killed: close binary** | WDS J03184−3244AB (TDS 2518), with SIMBAD components at 0.24″ and 0.5″. The photosphere, Teff (GSP-Spec 2750 K for BP−RP 0.81) and isochrone age all come from a blended SED |
| 1289707176373257856 | 270 | 3.40 | survives, **already published** | Centroids 0.40″ (1.0σ) and 1.84″ (1.9σ). LS forced W3 χ = 5.9, W4 χ = 5.4. It is in J/AJ/167/275, the 2024 catalogue of 1047 warm debris disks from Gaia, WISE and Spitzer. W4 SNR 6.0 |
| 1757504263952843776 | 254 | 3.94 | survives, **T_bb unsupported, already published** | LS forced **W4 χ = 1.3**: the W4 excess that fixes T_bb is not confirmed after deblending. In Cotten & Song 2016, McDonald+2017 and J/AJ/167/275 |
| 2120651376692536960 | 264 | 3.80 | survives, **already published, age tension** | LS forced W3 χ = 5.1, W4 χ = 7.7. In Cotten & Song 2016, McDonald+2017 and J/AJ/167/275. SIMBAD V*: a SuperWASP single-sinusoid (rotational) variable, i.e. a spotted, active star |
| 3168144078565656832 | 261 | 3.69 | survives with untested | Outside LS. W4 centroid 4.02″ off (3.9σ, just under the 4σ kill). W4 SNR 5.7, detected in 2/14 frames. Not in SIMBAD |
| 5295632592220066688 | 298 | 4.41 | survives with untested | Outside LS. W4 SNR 6.0, detected in 2/47 frames. SFD E(B−V) = 0.199 at b = −15°, against a 0.20 cirrus kill. Not in SIMBAD |
| **6225457312033584384** | **262 (254–272)** | **4.86** | **SURVIVES_DEEP_VET (nothing untested)** | See below |

Ten of the 12 candidates (every one except 6225457312033584384 and
4659120294392894848) have W4 SNR 5.5–6.5 against a 5σ
archive cut, and 4 to 40 % single-frame detection. Their T_bb is fixed by a
W4 flux that was *selected* at threshold. That biases it high, so T_bb comes
out low. The cell membership of those rows is weak even where nothing kills
them.

**6225457312033584384** (WISEA J144944.62−272822.6; G = 12.84, BP−RP = 0.92,
ϖ = 2.67 mas, l = 332.8°, b = +28.4°):

- **Excess and temperature.** W1−W3 = 2.37, with W3 at SNR 45 and W4 at SNR
  15 (W4 detected in 17/24 frames). f = 1.5 × 10⁻², T_bb = 262 K,
  r_bb = 1.08 AU, and log f/f_max = 4.86 at 1 Gyr.
- **Blends ruled out.** There is no SIMBAD object within 15″ and no published
  IR-excess catalogue entry. There is no Gaia source inside 10.6″ and no Legacy
  Surveys DR10 source inside 10.6″ to r ≈ 24.7. The deblended forced
  photometry keeps the excess on the star (W3 χ = 17.7, W4 χ = 8.8, W3 1.06×
  AllWISE). The W3 and W4 centroids sit on the star (0.26″, 0.7σ; 0.75″,
  0.8σ). The local density of W3 sources at this flux gives an expected ~1
  chance blend across the whole 38,848-star parent, and the deblending above
  is what rules it out here.
- **Age and variability.** Age is OLD from iso + kin: FLAME 5.6 (4.4–6.8) Gyr,
  kinematic log LR = +2.3 with v_tan = 76 km s⁻¹ and RV = −36 km s⁻¹. NEOWISE
  W1/W2 are variable (χ²_red 4.1 and 14.2). So are 14 of 17 known extreme
  debris disks.
- **DESI DR1 spectrum.** Redrock classifies it `STAR` at z = −1.4 × 10⁻⁴
  (cz ≈ −43 km s⁻¹, against Gaia RV −36). Hα is 0.20 Å *deeper* than the
  model, with no chromospheric filling and no emission, so there is no
  accretion or YSO signature. The EW over 6707.6–6711.8 Å (vac) is 58 mÅ.
  That figure **includes the Fe I 6709.3 Å (vac) blend** and has no formal
  error. It is below the 100 mÅ youth threshold, but it is not old-diagnostic
  either: it allows ages from about the Hyades to a few Gyr. The Marton+2016
  "Class I/II YSO candidate" tag is a WISE-colour classification, so it is not
  independent of the excess.

It remains **the one star in the empty cell that has passed every test the
vet ran** (`SURVIVES_DEEP_VET`, nothing untested). **It is not a
technosignature.** Natural late instability, as for BD+20 307 and
TYC 4479-3-1, is the default reading. What is still open:

- a proper Li abundance, deblended from Fe I, with an error;
- a Ca II H&K activity age;
- mid-IR spectroscopy (JWST/MIRI), which is S53's test, since there is no IRS
  spectrum;
- the NEOWISE light-curve shape, where an EDD-like brightening/decay is
  expected;
- a literature search beyond what SIMBAD and VizieR index (2025–26 papers).

**Pipeline defects the vet exposed:**

- `assess` kills on a main SIMBAD type of `IR`. BD+20 307, the positive
  control, carries `IR` among its otypes (its main type is `SB*`), so any
  debris disk SIMBAD files primarily as an IR source is killed. The deep vet
  flags `IR` instead. The 17 `simbad_type` kills have not been re-examined.
- As a sanity check, the pipeline recovers a known extreme debris disk:
  TYC 8105-370-1 (Moór+2021) is among the 25 `IN_CELL_AGE_UNDETERMINED` stars.
- `ages`'s known-disk crossmatch missed Cotten & Song 2016 table 4, McDonald+2017
  and J/AJ/167/275 for 4 candidates.
- `harmonise` prefers GSP-Spec Teff even when it is unphysical (2750 K).

## Lead vet — trying to kill Gaia DR3 6225457312033584384 (2026-09-24)

`python -m seti.cradle.leadvet` (`stage=leadvet`, optional `blocks=`).
Dispatches, all on `claude/handoff-cradle`:

- 36003679119: the first pass.
- 36007839429 (15:17Z, 11:17 a.m. EDT): the full pass of record.
- 36021814255 (block re-run, 15:48Z, 11:48 a.m. EDT): TESS across every pipeline,
  NEOWISE with the AllWISE-epoch field comparison, and PSF.

Everything is in `results/cradle/leadvet.json`.

| test | measurement | reading |
|---|---|---|
| NEOWISE W1, 22 visits 2014–2024 | range 0.043 mag, χ²_red 1.85 (93rd percentile of 14 field stars of matched W1); slope +0.0020 mag/yr | flat |
| NEOWISE W2 | range 0.099 mag (64th pct), χ²_red 6.0 (highest of 14); fade +0.0052 mag/yr against a field median of +0.0032 ± 0.0009 | most of the fade is instrumental (the field shares it); the residual +0.002 mag/yr is 2.3 MAD. At most a marginal W2 excess scatter |
| AllWISE 2010, two visits | W3 fainter by 0.195 ± 0.033 mag, W4 by 0.09 ± 0.19. The same visit pair for 21 field sources: W3 −0.02 (MAD 0.11) | 1.9 MAD, **not significant** |
| SED | no AKARI IRC/FIS, IRAS PSC/FSC or Herschel detection (all far above the 17 mJy W4 flux). The excess is measured at 2 wavelengths only | T_bb = 262 K (250–275) from AllWISE, 235 K (224–248) from the LS deblended forced photometry, 256 K (248–264) joint. **The cell edge (250 K) is inside the systematics.** A two-temperature fit is not constrained. W2 excess 0.34 ± 0.26 mJy, against 0.10 mJy predicted by the 262 K body; an 800 K component is ≤ 32 % of f (3σ) |
| DESI DR1 (R ≈ 4500) | Li 6708: EW 54 mÅ blend − 10 mÅ Fe I (Soderblom+93) = **44 ± 27 mÅ**. Comparison star 1276273278883611264 (FLAME 10.9 Gyr): −23 ± 17 mÅ | below a Hyades-age G dwarf (~70–90 mÅ at 5700 K); not a youth signal, but it cannot exclude ~1 Gyr |
| DESI chromosphere, core/model ratio | lead: Ca II K 1.68, H 1.25, IRT 0.99/1.00/0.99, Hα 0.95. Old comparison: K 2.86, H 1.51, IRT 1.00/1.04/1.03, Hα 0.99 | the K ratio is template mismatch; the lead is **no more active than a 10.9 Gyr star** |
| TESS (SPOC, TESS-SPOC, QLP, TGLC, eleanor, TARS; sectors 11/38/65/91) | no period recurs across sectors or pipelines. QLP-SAP s91 gives 8.28 d, but it does not survive per-orbit detrending and TARS s91 gives 6.5 d. PDCSAP p5–p95 is 3.7–6.4 ppt | **no rotation detected**: consistent with an inactive star, but not a gyro age |
| ZTF | saturated (r ≈ 12.6) | untested |
| Kinematics | U, V, W = −61, −31, +12 km/s (LSR). Bensby+14: thin 0.94, thick 0.06. Orbit: R 6.0–9.0 kpc, e = 0.20, z_max = 0.29 kpc | a heated thin-disc orbit, typical of several Gyr, not of a young star. Not thick disc |
| Isochrone | FLAME 5.6 (4.4–6.8) Gyr, logg 4.39, M_G = 4.80, [M/H] −0.25 (GSP-Phot) / −0.51 (StarHorse). No GSP-Spec | a main-sequence G dwarf. The isochrone age is weak, as it always is on the main sequence |
| Companions | RUWE 0.98. Astrometric excess noise 0.071 mas (sig 5.2, common at G ≈ 13). IPD multi-peak 1 %. No NSS solution. Gaia RV −36.3 ± 2.3 (12 transits) vs DESI −42.6 km/s. **No co-moving source among 17 within 60″** (≈ 22,000 AU) | no companion found. The 6 km/s RV difference is within DESI's stellar-velocity systematics |
| Confusion | IRSA DUST: E(B−V) 0.133, I₁₀₀ = 6.67 ± 0.09 MJy/sr (smooth). unWISE FWHM: W1 5.67″ (field 5.70 ± 0.16), W3 6.97″ (field 7.17 ± 0.42, n = 4), W4 11.1″ | point-like, low cirrus |
| Literature | no SIMBAD object within 60″. None of 140 VizieR catalogues matching debris / infrared excess / warm dust / extreme debris / exozodi keywords has an entry within 10″. IDs: TIC 117092, 2MASS J14494466−2728225, UCAC4 313-078514 | not previously published as far as SIMBAD and VizieR index. Papers not yet ingested (2025–26) are unchecked |

**Occurrence.** The Gaia parent holds **1,282,731 FGK dwarfs** (G < 13.5,
d < 500 pc). Scaled from the lead itself, a 262 K disc with f = 10⁻² is a
≥ 5σ W4 detection for W1 < 11.7 (G ≲ 13.3), so most of the parent is
sensitive. Two known old EDDs come from Tycho-scale searches, BD+20 307 and
TYC 4479-3-1. In this parent: BD+20 307 (hot, out of the cell) plus this star.
That is ~2 per 10⁶ FGK dwarfs. In this same sample, young and age-undetermined
EDDs pile up at 250–350 K: 27 in-cell stars have f > 5 × 10⁻³, including the
Moór+2021 EDD TYC 8105-370-1 at 265 K. So **one old EDD in the cell is not
statistically surprising.** What would be new is only its temperature–age
combination.

**Physics of a natural reading.** At r_bb = 1.08 AU around 0.92 L_☉ the
orbital period is 1.1 yr. The collisional lifetime of the smallest grains is
~P/(4πf) ≈ 3 yr (dr/r = 0.5). The dust therefore has to be resupplied, which
means a recent event, as in every EDD. The cross-section 4πr²f ≈ 5 × 10²¹ m²
is ~2 × 10¹⁹ kg in 1 µm grains, or ~10²¹ kg (a few % of a lunar mass)
extended to cm sizes. One collision between asteroid-to-Moon-mass bodies
accounts for it.

**Natural late collision vs genuinely unexplained.** These tests would settle
it:

1. **Mid-IR spectrum (JWST/MIRI):**
   - silicate emission with crystalline forsterite or enstatite, or silica →
     impact debris (natural);
   - featureless 8–13 µm → S53's slag-compatible class, but large grains do
     that too.
2. **A robust age:**
   - a high-resolution Li abundance (A(Li)) and log R′_HK;
   - an asteroseismic or eclipsing constraint if one appears.

   An age below ~1 Gyr moves it into the ordinary young-EDD population.
3. **Mid-IR time-domain** (a new W3/W4-like epoch, e.g. SPHEREx or NEO
   Surveyor). EDD-like decay or brightening over years favours a collision.
   Constancy over decades would sit oddly with a ~3 yr collisional lifetime and
   would require a large, steady reservoir.
4. **A dynamical trigger:** none found (no companion within 22,000 AU, clean
   astrometry). A planet perturber is unconstrained.

Nothing measured so far distinguishes it from a natural late collision.
**It is an unusual natural-candidate object, not evidence of technology.**

**The two secondaries:**

- **3168144078565656832 is compromised:**
  - a coherent **0.45 d, ~5 ppt signal in 9 TESS sectors** (QLP and SPOC). Its
    source is unidentified (TESS pixels are 21″); if it is on-target, the star
    is a close binary;
  - W3 FWHM 7.69″ against the field's 6.94 ± 0.07″ (only 3 field stars);
  - W4 centroid 4.0″ off (3.9σ);
  - kinematically thin disc.
- **5295632592220066688 is old but weakly detected:**
  - genuinely old: thick disc (Bensby P_thick 0.99, z_max 1.56 kpc), GSP-Spec
    [M/H] −0.72 and [α/Fe] +0.43, FLAME 12.2 Gyr;
  - its excess is weak: W4 SNR 6.0, detected in 2/47 single frames;
  - W3 FWHM 7.64″ vs 7.03 ± 0.28″ (2.1 MAD);
  - IRSA SFD E(B−V) 0.207 at the reference pixel, over the pipeline's 0.20
    cirrus kill, with I₁₀₀ = 11.3 MJy/sr;
  - no TESS period, no DESI spectrum.

## The honest weakness

Natural late instability explains both known old extreme debris disks, and it
would explain a third. A star in the cell is not a technosignature; it is an
object that the collisional-cascade picture says should not exist at that
temperature and that age, and that nobody has looked for. The discriminant
that could move it further is S53, and S53's own limit — no lab constants for
Ca–Al glass — is stated above. Every surviving candidate carries its
`not_excluded` list, and `natural late instability` is always on it.
