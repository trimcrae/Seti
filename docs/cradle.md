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

## The honest weakness

Natural late instability explains both known old extreme debris disks, and it
would explain a third. A star in the cell is not a technosignature; it is an
object that the collisional-cascade picture says should not exist at that
temperature and that age, and that nobody has looked for. The discriminant
that could move it further is S53, and S53's own limit — no lab constants for
Ca–Al glass — is stated above. Every surviving candidate carries its
`not_excluded` list, and `natural late instability` is always on it.
