# FORGE — hot exozodis read as ~1500 K swarm candidates (S47)

Package `src/seti/forge/`; CLI `seti forge --stage {probe,acquire,screen,assess,all}`
(or `python -m seti.forge.run`); workflow `.github/workflows/forge.yml`;
config `config/forge.yaml`; results `results/forge/`. Signature S47 of
[`necrofrontier.md`](necrofrontier.md) ("The forge"). Built 2026-09-22.

## 1. Claim

The cheapest place to collect stellar energy per unit collector mass is as
close to the star as materials allow, ~1500 K. Near-infrared interferometry
finds a ~1 % H/K-band excess around 10–30 % of nearby main-sequence stars
(Absil et al. 2013; Ertel et al. 2014; the 2025 review), read as sub-micron
grains that should be blown out in hours; the leading model (magnetic
trapping of nano-grains, Rieke, Gáspár & Ballering 2016) has unresolved
problems. Hephaistos I excludes T > 1000 K by construction, and nobody has
framed this population as swarm candidates (the necrofrontier literature
sweep, group g2, found no hot-exozodi-as-swarm reading).

**But the physics cuts against the simplest version.** A grey 1500 K body
producing a 1 % K excess gives 6–8 % at 10 µm (6.7 % for a G star, 8.1 % for
an A star — `physics.planck_extrapolation`). The K-bright / N-faint pattern
that forces nano-grains (Kirchschlager et al. 2017: grains < 0.5 µm) is
exactly what a Planck swarm cannot do. **What survives is the outlier: a star
whose N-band excess IS consistent with the Planck extrapolation of its K
excess.**

## 2. Method

### 2.1 The per-star statistic

Every excess is a fraction f(λ) = F_excess / F_star; the photosphere is a
blackbody at T_eff (adequate from H to N at the percent level — the
calibration floor below is larger than the departure). Two families:

* **grey body** — f(λ) ∝ B_λ(T) / B_λ(T_eff), free (T, f_K); fitted twice:
  with T confined to the swarm range 700–1800 K (the hypothesis) and with T
  free to 7000 K (a companion photosphere shows up here);
* **nano-grain family** — the same with an emissivity Q(λ) = min(1, (2πa/λ)^β),
  a ≤ 0.5 µm, β ∈ {1, 2}: the small-grain limit that makes the excess
  K-bright and N-faint. Fitted with T confined to the temperatures the
  published fits occupy (1500–2000 K: grains at or just outside sublimation;
  the ceiling is carbon's sublimation temperature) and, separately, with T
  free.

χ² over every measurement with a Gaussian error (a "null excess" of
0.5 ± 0.3 % is a measurement, not a limit) plus a one-sided term for quoted
upper limits; a per-band cross-instrument calibration floor is added in
quadrature (0.1 % H/K, 0.3 % N — the FLUOR/PIONIER/JouFLU visibilities and
the LBTI/KIN nulls are not tied to each other). The statistic is
Δχ² = χ²_nano(restricted) − χ²_grey(swarm range).

### 2.1b The polarimetric null: read, carried, and deliberately not in χ²

The polarimetry table (Marshall+2016, HIPPI/AAT) is fetched on the runner
(`acquire.fetch_polarimetry_table`), keyed to the sample, and carried per star
as `polarimetry.limit_ppm` — in `star_table.csv` and on every candidate
record. It is **not** a term in the likelihood ratio, and the reason is
physical rather than convenience.

Both families the statistic compares emit **thermally** at H and K: a 1500 K
body peaks at ~1.9 µm, and sub-micron grains at their sublimation temperature
are hotter still. Thermal emission from an optically thin, randomly oriented
swarm is essentially unpolarised in *either* case. A polarisation limit
therefore constrains the **scattered-light fraction**, which is a different
axis from the emissivity law; folding it into χ² would let a constraint that
cannot tell the two families apart masquerade as evidence that can. Where it
is genuinely informative is on a survivor — a scattering constraint on a
candidate is a real follow-up discriminant — so that is where it is attached.

Two policy rules go with it: a star the polarimetry table lists but the
infrared sample does not is **not** added (a polarimetric null on a star with
no measured excess says nothing this channel can use, unlike an excess table's
rows, which do define the sample), and where a star is listed more than once
the **tightest** constraint is the one carried.

### 2.2 The degeneracy every candidate carries

A nano-grain population **cooler** than 1500 K reproduces the K/N ratio of a
1500 K grey body: lowering T raises N/K by the factor the emissivity lowers
it. From K + N alone the restricted and unrestricted nano fits therefore
differ, and the record carries both (`delta_chi2`, `delta_chi2_unrestricted`,
and `delta_chi2_by_nano_t_floor` at 1000 / 1200 / 1500 K) so the reader sees
how the verdict depends on the sublimation-temperature assumption. A
candidate is "a hot excess the standard small-grain model at its sublimation
temperature cannot produce"; breaking the residual degeneracy needs L/M-band
interferometry (MATISSE), which is the follow-up every candidate points to.

### 2.3 Tiers (`physics.assess_star`, in gate order)

| Tier | Meaning |
|---|---|
| `NO_NIR_EXCESS` | no H/K excess at ≥ 3σ to extrapolate |
| `N_UNTESTED` | NIR excess but no N-band measurement of any kind — **never a candidate**; the Planck prediction is recorded so the star is a target, not a result |
| `COMPANION_TEMPERATURE` | the free grey body's temperature is *measured* above 1800 K and beats the swarm-range grey body by Δχ² ≥ 9: a photosphere (ledger: "T > 1800 K is a companion"). See §2.6 for why "measured" is the load-bearing word |
| `KNOWN_COMPANION` | a catalogued companion inside the interferometric field at Δmag ≤ 6 (WDS, or the embedded companions asset); closure phases do not exclude faint companions at the 1 % level (Tsishchankava et al. 2025) |
| `UNVERIFIED_INPUT` | the driving values (the NIR anchor and the N measurement) are embedded transcriptions the run did not confirm against an archive |
| `candidate` | Δχ² ≥ 9, N-band detected at ≥ 3σ, within 2σ of the grey extrapolation, grey fit acceptable, driving values archive-read or archive-verified, and **no** competitive photospheric-temperature alternative (§2.6) |
| `interest` | N band consistent with the grey extrapolation but not significant |
| `nano_preferred` | Δχ² ≤ −9: K-bright / N-faint (the population's signature) |
| `inconclusive` / `NEITHER_MODEL_FITS` | N measured, neither family preferred / both rejected |

A **variability term** (`physics.variability`): repeat epochs in one band
(FLUOR 2013 vs JouFLU 2017, PIONIER 2014 vs 2021) give the largest pairwise
|f₁ − f₂|/σ; a swarm being built or decaying changes, grains in equilibrium
do not. Reported, never a kill: only κ Tuc varied in the record and it is now
a companion (Stuber et al. 2026).

### 2.4 Data

The published near-infrared / mid-infrared interferometric excess tables,
all read from VizieR on the runner with runtime schema discovery (TAP_SCHEMA
first, the ASU / ReadMe inventory when TAP is down, keyword search of the
descriptions as the fallback — the ladder ARC and METRONOME use):

| Key | Table | Band | Stars |
|---|---|---|---|
| `absil2013` | Absil+2013 A&A 555 A104, CHARA/FLUOR | K | 42 |
| `ertel2014` | Ertel+2014 A&A 570 A128, VLTI/PIONIER | H | 92 |
| `nunez2017` | Nuñez+2017 A&A 608 A113, CHARA/JouFLU | K | 44 |
| `absil2021` | Absil+2021 A&A 651 A45, PIONIER | H | 52 |
| `ertel2018`, `ertel2020` | Ertel+2018 AJ 155 194; Ertel+2020 AJ 159 177, HOSTS / LBTI nulling | N (11 µm) | 30, 38 |
| `mennesson2014`, `millangabet2011` | Keck Interferometer Nuller 8–13 µm | N | 47, 25 |
| `marshall2016` | Marshall+2016 ApJ 825 124, HIPPI polarimetry of exozodi hosts | pol | — |
| `kirchschlager2017` | Kirchschlager+2017 MNRAS 467 1614, modelled systems | model | 9 |

The VizieR identifiers are asserted from the journal references and marked
`verify` in the config; the probe stage lists what the archive actually holds
under each and records the real column names and the roles it resolved
(`hd` / `hip` / `name`, `excess`, `excess_err` by VizieR's `e_<col>`
convention, `significance`, `limit_flag`, `band`). A table that exposes only a
zodi level is rejected as needing the paper's model to convert.

**Embedded seed tables** (`src/seti/data_assets/forge_*.csv`): `forge_targets.csv`
(the union of the surveys' samples as recalled from the publications — HD,
HIP, names, spectral type, which surveys), `forge_excess.csv` (a handful of
headline values with per-row citations: Vega, τ Cet, ζ Aql, β Leo, Altair
from the FLUOR series; Fomalhaut's VINCI K and KIN 8–9 µm excesses; η Crv's
KIN excess; the eight PIONIER detections as membership rows), `forge_companions.csv`,
`forge_polarimetry.csv`. **These are seeds, not data.** They were transcribed
without the archive in reach, so every numeric row carries
`verify = unverified` and the runner's `acquire` stage marks each
`verified` / `discrepant` (the archive value is then used) /
`not_in_archive` / `table_not_reached` (`results/forge/forge_excess_verified.csv`).
Stars the archive lists that the seed does not are added to the sample from
the archive row, so the archive, not the transcription, defines the sample.

Per-star context: positions and spectral types from Simbad TAP by identifier
(T_eff from the spectral type on the Pecaut & Mamajek scale); WDS pairs inside
the field; Gaia DR3 RUWE / NSS / neighbours where Gaia has the star at all
(it is missing or saturated for the brightest); the embedded companions.

### 2.5 The scale leg: broadband colours

For every star in the interferometric samples (2MASS / AllWISE cones) and for
the whole < 30 pc main-sequence population (Gaia DR3 × 2MASS × AllWISE
through the ESA archive's own cross-match tables, in declination bands, the
`tmass_psc_xsc_join` bridge first and the direct chain as fallback), the
K−W1 vs W1−W2 diagram, with W2−W3 as the third colour. A hot grey component
reddens K−W1 and W1−W2 in a fixed ratio (~1 : 0.8 at 1500 K) and W2−W3 by a
larger fixed amount. Empirical loci (running median and MAD in BP−RP) give
residuals; `BROADBAND_HOT_EXCESS` requires both NIR-side residuals ≥ 3σ,
agreeing K-excess estimates, and a W3 residual within 3σ of the
extrapolation (too faint is nano-like, too bright is cooler dust).
Saturated photometry (K < 4.5, W1 < 8, W2 < 7 — which is most of the
interferometric sample) is `SATURATED`; a negative W1−W2 residual is a blend
(the ledger).

**Sensitivity, stated in the interferometric unit.** A 1 % K excess at
1500 K moves K−W1 by 0.015 mag and W1−W2 by 0.012 mag; the locus scatter is
0.03–0.05 mag, so the 3σ sensitivity is a K excess of roughly 6–10 % — an
order of magnitude worse than interferometry. The leg reports
`f_k_3sigma_pct` per colour bin and the population median; it can only find
the rare bright case.

### 2.6 Separating a companion from small grains — and why it is not obvious

The companion kill cannot be written as "a hot grey body beats the small
grains", which is what it looked like it should be. At the top of its
temperature range (~2000 K) with β = 1 and a ≪ λ the nano-grain emissivity
Q ∝ 1/λ cancels most of the Planck slope, so the small-grain family is itself
nearly flat from H to N and fits a companion photosphere about as well —
**however precise the data**. Numerically, for a 4600 K companion giving
H = 0.95, K = 1.00, N = 1.20 %, the best restricted nano fit sits within
Δχ² ≈ 2–6 of the best free grey fit and does not fall further as the errors
shrink.

What does separate them is the *shape of the χ² minimum*, not its depth:

* a **K-bright / N-faint** star (the population's signature) is bluer than
  any unit-emissivity body can be, so the free grey fit runs away to the hot
  edge of the grid and its "temperature" is an artefact of where the grid
  stops — Vega and Fomalhaut both peg at 7000 K;
* a **companion photosphere** gives a nearly flat H-to-N excess and the free
  grey fit converges on an interior minimum — a real colour temperature.

So `fit_family` reports `t_at_grid_edge`, and `COMPANION_TEMPERATURE`
requires an interior minimum above 1800 K. The nano degeneracy is not allowed
to veto the kill; it is reported per star as
`hot_grey.nano_fits_comparably` with its Δχ², so the reader sees that the
kill rests on the colour temperature rather than on rejecting grains.

The same logic protects the candidate tier from the other direction. Gate
`hot_photosphere_alternative`: if a free grey body *above* the sublimation
ceiling with a measured temperature fits at all better than the swarm-range
one, the star is `inconclusive`, not a candidate. A genuine 1500 K swarm with
a detected N band pins its own temperature — the N/K ratio measures it — so
the free fit lands inside the swarm range and the gate stays quiet on the
injected signal.

### 2.7 One published number is one datum

The embedded seed rows and the archive rows describe the *same* published
measurements. Merging them on the citation string ("Akeson+2009 ApJ 691 1896;
Absil+2013" against the survey key `absil2013`) kept both, so a star with a
seed *and* its archive table entered χ² twice with its error effectively
divided by √2. `Measurement` therefore carries a survey id and the merge key
is `(survey, band)`; the published tables are one row per star per band, and
genuine repeat epochs come from different surveys (FLUOR 2013 / PIONIER 2014
/ JouFLU 2017), which are kept apart. On the fake-archive end-to-end this
moved β Leo's Δχ² from −7.8 to −4.2 — from "nearly nano-preferred" to
"nothing preferred", which is the honest reading of 0.94 ± 0.26 % in K
against 1.70 ± 0.30 % in N.

## 3. Offline tests (`tests/test_forge.py`, the CI gate; no network)

**A green local suite is not a green gate** (`channel-brief.md` §0 item 5).
The sandbox venv holds pandas 2.3.3; the runner installs **pandas 3.0.6**, and
a sibling channel lost a whole dispatch to an API pandas 3 had removed, dying
two minutes in, before a single archive call. FORGE was therefore run against
the runner's major version in an isolated interpreter (pandas 3.0.6, numpy
2.4.6, pyarrow 25.0.1): **33/33 pass**, and 32/32 at the commit the first
dispatch is pinned to. `pyarrow` is a declared dependency, so the population
parquet checkpoint has an engine on the runner.

Two specific hazards were checked rather than assumed. No `errors="ignore"`,
`DataFrame.append`, `iteritems` or `applymap` appears anywhere in the package
(every `append` in it is `list.append`). The one expression the pandas-2
FutureWarning names —
`table["driving_verified"].fillna(False).astype(bool).sum()`, the count of
stars whose driving values were archive-verified — returns the same answer
under 2.3.3 and 3.0.6 for bool, object-with-`None`, float-with-`NaN` and
all-`None` columns, because the `.astype(bool)` is explicit and does not rely
on the intermediate downcast that changed. It is fed an in-memory frame built
from Python bools, never a re-read CSV, which matters: on string data
`"False"` is truthy and the count would be wrong in *both* versions.

* an injected Planck 1500 K excess with consistent H, K and N is recovered
  as `candidate` with T within 15 % (G and A star); Δχ² 25–39;
* a nano-grain system (K-bright / N-faint) is never a candidate; the typical
  real case (1 % K, N null 0.3 ± 0.4 %) is `nano_preferred` (Δχ² ≈ −13);
* a missing N band is `N_UNTESTED` even at a 30σ K excess;
* a 5000 K photosphere at 3 % is `COMPANION_TEMPERATURE` (at 1 % the same
  shape sits within 2σ of a 2000 K large grain, and the gate stays quiet);
  a catalogued companion is `KNOWN_COMPANION`; embedded, unverified driving
  values are `UNVERIFIED_INPUT` until both the anchor and the N value verify;
* the broadband confounder ladder: a body much cooler than 1500 K fails the
  K−W1 / W1−W2 consistency test first and reads `NORMAL` — only a hot
  component *with* a cold belt reaches `BROADBAND_W3_TOO_BRIGHT`;
* the end-to-end run on a fake VizieR (three tables, four synthetic stars,
  one discrepant transcription) verifies the asset, adds the archive's stars,
  recovers the swarm, separates the flat-spectrum companion from the
  K-bright/N-faint star (§2.6), counts the seeded and archived copies of one
  published value as one datum (§2.7), and flags the 15 % broadband
  injections while leaving the 1 % ones alone and reading a hot component
  plus a cold belt as `BROADBAND_W3_TOO_BRIGHT`;
* a dead archive is `NO_DATA_REACHED | DEGRADED (...)`, every embedded row
  `table_not_reached`, no candidate.

## 4. Verdict vocabulary (`results/forge/summary.json`)

`NO_DATA_REACHED` (no table reached and nothing verified),
`PLANCK_CONSISTENT_CANDIDATES`, `NO_PLANCK_CONSISTENT_OUTLIER` — the last is
**a count, not a limit**: the funnel's `n_N_UNTESTED` says how much of the
population the Planck test never touched. `DEGRADED (...)` lists every table
that failed and every veto that could not be applied.

## 5. Kills and systematics not excluded

1. **Faint companions at the 1 % level.** Closure phases do not exclude them
   (Tsishchankava et al. 2025); WDS / Gaia / the companions asset veto the
   catalogued ones and the temperature gate the hot ones, but an
   uncatalogued late-type companion at 1 % remains on every candidate's
   `systematics_not_excluded`.
2. **K-vs-N cross-instrument calibration.** The floors are asserted; a
   candidate whose Δχ² collapses when the N floor is doubled is not one.
3. **The cool nano-grain degeneracy** (§2.2), on the record per star.
4. **Small sample**: ~150 stars with any interferometric excess, fewer with
   an N-band measurement; most of the sample is `N_UNTESTED` and says nothing.
5. **Transcription**: no embedded value drives a candidate until the archive
   confirms it.

## 6. Runs

### 6.0 Offline dry run on the embedded seeds (not a measurement)

Before the first dispatch, `assess_star` was run on the seven embedded seed
rows that carry a numeric value. **This is not a result**: the seeds are
transcriptions marked `verify = unverified`, so nothing here may drive a
candidate. It is reported because it shows the detector behaving as designed
on real published numbers rather than on injected ones.

| Star | Bands (%) | Tier | Δχ² |
|---|---|---|---|
| HD 172167 (Vega) | K 1.26 ± 0.27 | `N_UNTESTED` | — |
| HD 10700 (τ Cet) | K 0.98 ± 0.19 | `N_UNTESTED` | — |
| HD 177724 (ζ Aql) | K 1.69 ± 0.27 | `N_UNTESTED` | — |
| HD 102647 (β Leo) | K 0.94 ± 0.26 | `N_UNTESTED` | — |
| HD 187642 (Altair) | K 3.07 ± 0.24 | `N_UNTESTED` | — |
| HD 216956 (Fomalhaut) | K 0.88 ± 0.12, N8 0.35 ± 0.10 | `nano_preferred` | **−21.8** |
| HD 109085 (η Crv) | N 4.90 ± 0.10 | `NO_NIR_EXCESS` | — |

Two things to read from it. First, **five of the seven best-known hot-exozodi
hosts are `N_UNTESTED`** — they have a K excess and no N-band measurement of
any kind, so the Planck test never touches them. Their predicted 10.5 µm
excesses (6.3 % for τ Cet, 10.3 % for Vega, 23.2 % for Altair) are what an
N-band observation would have to find and did not look for; that is a target
list, not a result, and it is the single largest limit on the channel.

Second, **Fomalhaut — the one seed system with both a K excess and an N-band
measurement — is `nano_preferred` at Δχ² = −21.8** (best fit a = 0.4 µm,
β = 2). Its K excess of 0.88 % implies 7 % at 10 µm for a grey 1500 K body;
Keck measures 0.35 ± 0.10 % at 8.5 µm. This is the K-bright / N-faint pattern
in its purest form, and it is the expected outcome: **the small-grain model is
expected to win wherever the data actually constrain it.** The channel's
deliverable is the ranked Planck-consistency list and whatever survives it,
not the count of stars that behaved as the standard model says.

### 6.1 Runner dispatches

| Run | Stage | Dispatched (EDT) | Verdict |
|---|---|---|---|
| [35744731075](https://github.com/trimcrae/Seti/actions/runs/35744731075) | `all`, `skip_population=true` | 2026-09-22 11:04 | in flight |

*(filled in from `results/forge/` after each runner dispatch — see STATUS.md)*
