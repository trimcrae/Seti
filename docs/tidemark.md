# TIDEMARK — the front that stopped

*A tidemark is what a receding tide leaves behind: a line on the sand marking
where the water stopped. If an expansion halted, the residue has an edge.*

## 1. The claim

Every executed technosignature search reports **individual candidates** and then
dies on **per-object contamination**. Any single infrared excess is a blend, a
background galaxy, or cirrus; any single dimming is dust; any single abundance
anomaly is a pipeline systematic. This repository has proved that six times over.

TIDEMARK asks a different question, of the same data:

> **Is the anomaly rate *per star* spatially structured?**

That question is immune to exactly the failure mode that kills the others. A
contaminant population traces the ordinary stellar density and the survey's own
selection function; it cannot manufacture a coherent **gradient** in rate per
star, and it certainly cannot manufacture a **sharp edge** in one. Contamination
degrades the sensitivity of a population test; it does not fake its signal.

Three statistics, one null:

| Statistic | Question | Module |
|---|---|---|
| **Gradient** | does the rate trend monotonically with Galactocentric radius, \|z\|, or Galactic longitude? | `gradient.py` |
| **Edge** | is there a sharp step in rate — a boundary — in 1D, in 3D shells, or in sky caps? | `edge.py` |
| **Age** | does the rate rise, saturate, or turn over with stellar age? | `agerate.py` |

## 2. Novelty: three published predictions that contradict each other, and not
one has ever been tested

This is the unusual part. The theory literature does not merely permit a spatial
technosignature test — it contains **mutually exclusive, falsifiable predictions
about the direction of the effect**, made by prominent authors, some of them
by the *same* authors in different papers. None has ever been confronted with
data.

**Prediction 1 — the OUTER RIM.** Ćirković & Bradbury 2006, *"Galactic
Gradients, Postbiological Evolution and the Apparent Failure of SETI"*, New
Astronomy 11(8), 628–639, doi `10.1016/j.newast.2006.04.003`, arXiv
`astro-ph/0506110`. Verbatim:

> "we suggest that the **outer regions of the Galactic disk are most likely
> locations for advanced SETI targets**, and that intelligent communities will
> tend to **migrate outward** through the Galaxy as their capacities of
> information-processing increase, for both thermodynamical and astrochemical
> reasons."

and in the body: *"the **maximum probability will be located in the ring on the
periphery of the Milky Way**"*. The physics is the cold reservoir: computational
efficiency scales as `(I/E)_max ∝ √R` from `σT_D⁴(R) ∝ L*R⁻²`.
→ **Signature: anomaly rate rising with `R_gal`.**

**Prediction 2 — the GALACTIC CENTRE.** Wright, Carroll-Nellenback, Frank &
Scharf 2021, RNAAS 5(6), 141, doi `10.3847/2515-5172/ac0910`. The title *is* the
prediction: *"The Dynamics of the Transition from Kardashev Type II to Type III
Galaxies **Favor Technosignature Searches in the Central Regions of Galaxies**"*.
The settlement-front speed scales as maximum ship range over mean stellar
separation, so the inward-moving front accelerates into rising stellar density
while the outward front stalls.
→ **Signature: anomaly rate falling with `R_gal`.** Exactly the opposite sign.

**Prediction 3 — NO STRUCTURE AT ALL.** Wright et al. 2014 (Ĝ I), ApJ 792, 26,
arXiv `1408.1133` — the same first author as prediction 2, seven years earlier:

> "The slow expansion of an ETI should thus be modeled not as an expanding circle
> or sphere… A better model is as the mixing of a gas"; "rotational shear and the
> thermal motions will **disperse and 'mix' any Fermi bubbles on a rotational
> timescale**"; "Until we have discovered 100 galaxy-spanning supercivilizations,
> we should not expect to find any Fermi bubbles."

→ **Signature: no gradient and no edge beyond the matched null.** Note that this
is a *positive* prediction that TIDEMARK can confirm, not a fallback — a clean
null in a well-calibrated test is evidence *for* the shear-mixing model.

**Prediction 4 — a BOUNDARY is the observable.** Carrigan 2010, *"Starry
Messages: Searching for Signatures of Interstellar Archaeology"*, JBIS 63, 90,
arXiv `1001.5455` (the "Fermi bubble": a colonised region within the Galaxy whose
*boundary* is what you look for); Landis 1998, JBIS 51, 163 (percolation
colonisation in clusters with **sharp boundaries**); Hanson, Martin, McCarter &
Paulson 2021, arXiv `2102.01522`, doi `10.3847/1538-4357/ac2369`, which
explicitly estimates *"how common in the sky the volume borders would be, **for
which astronomers might search**"* — an in-paper invitation with, as far as the
citation trees show, no published response.
→ **Signature: a sharp step in rate at a shell radius or cap edge.**

Purely modelled, never confronted with data: Newman & Sagan 1981 (Icarus 46,
293); Carroll-Nellenback, Frank, Wright & Scharf 2019 (AJ 158, 117, arXiv
`1902.04450`); Hair & Hedman 2013 (IJAsB 12, 45); Lingam 2016 (Astrobiology, doi
`10.1089/ast.2015.1411`); Vukotić & Ćirković 2012 (OLEB 42, 347 — predicts
*"strong clustering of advanced civilizations and their colonies, with large
portions of the Galactic Habitable Zone unoccupied"*); Olson's cosmological
series (CQG 32, 215025; JCAP 2016(04), 021; IJAsB 2017).

### 2.1 Novelty verdict, and where it is weaker than it sounds

**Verdict: a selection-corrected, statistically evaluated gradient or edge test
on a real technosignature anomaly population is unoccupied. Confidence high.**
Evidence, from the fetched corpora under `results/necrolit/`,
`results/litcheck/`, `results/litcheck_dyson/`, `results/dysonlit/`,
`results/hephlit/`:

1. **No executed search does it.** Across Carrigan 2009/2010, Ĝ/G-HAT I–IV,
   Garrett 2015, Griffith 2015, Zackrisson 2015/2018, Hephaistos I–IV, Ren et
   al. 2024–26, Contardo & Hogg 2024, Huang et al. 2026, Lacki 2016/2025 and
   VASCO, Galactic position appears in exactly four modes and none of them is
   this test: **as a cut** (`|b| ≥ 10`, bulge/LMC excision); **as qualitative
   Aitoff eyeballing** (Carrigan 2009 Fig. 8); **as a sky-averaged,
   position-independent surface density** for chance-alignment budgets; and **as
   rate vs heliocentric distance where the trend *is* the incompleteness**
   (Hephaistos I, Table 1, stated as such). Everybody imposes the selection
   function; nobody inverts it.
2. **The 2026 field-wide review is silent.** *The Search for Technosignatures: a
   Review of Possibilities* (arXiv `2605.21093`) contains zero occurrences of
   "galactocentric", "spatial distribution", "sky distribution", "percolation",
   "Fermi bubble", "expansion front" or "border". It cites Ćirković & Bradbury
   2006 and Wright et al. 2021 as theory only.
3. **The citation trees are empty of it** (642 distinct citing titles checked),
   and the RNAAS 2021 note has `cited_by_count = 3`.

**The honest qualifications.** The claim "never been tested for spatial
structure" is slightly too absolute, and the defensible form is *no
selection-corrected, statistically evaluated gradient or edge test has ever been
executed*. Two near-misses matter:

* **Blain 2024** (arXiv `2409.11447`) *proposes* the sky-distribution test —
  *"The distribution of DSM candidates should allow a statistical view of whether
  they shadow the distribution of Gaia stars or not"* — and implements nothing.
  His null is "traces the Gaia stellar density", i.e. exactly the selection
  function; TIDEMARK's signal is the residual *after* dividing that out.
* **Huang, Tao & Zhang 2026** (arXiv `2605.06072`) is the closest methodological
  precedent: a "crystallization index" regressed against Galactocentric distance
  across 79 globular clusters, with an empirical null that *"preserve[s] its
  radial selection function… but erase[s] any coherent higher-order structure"*.
  The units are clusters, not stars; the observable is a dynamical metric, not a
  technosignature anomaly; and the paper explicitly disclaims technosignature
  interpretation.

Confidence is **lower (~75%) for the age leg (S25)** — no dedicated literature
probe for "technosignature occurrence rate vs stellar age" was run, so that is an
absence of evidence rather than evidence of absence. Every age reference in the
corpus is per-candidate gyrochronology, never a population rate; Contardo & Hogg
2024 explicitly declines to compute an occurrence rate at all.

Also worth recording: **Ćirković 2008** ("Against the Empire", arXiv `0805.1821`,
JBIS) is Ćirković's own later move toward spatially compact, optimisation-driven
civilisations, which weakens his 2006 outward-migration prediction. Cite it
whenever prediction 1 is used as a directional anchor.

## 3. The method

### 3.1 The selection function, made explicit and auditable

This is the whole scientific difficulty and the whole contribution. `STATUS.md`
already noted the problem — *"position alone is washed out by the Galactic
density gradient"* — and that is a reason the test is **hard**, not a reason it
has been done.

Stratify the parent sample on the covariates that control **detectability**
(apparent magnitude, distance, colour, extinction, crowding, epoch count,
photometric error, metallicity) and **never** on the coordinate under test.
Within stratum `s` the parent has `N_s` rows of which `c_s` are anomalies. Every
parent star then carries the weight

```
w_i = c_{s(i)} / N_{s(i)}
```

— the empirical, non-parametric probability that a star with row `i`'s
detectability is flagged. For any region `W` of any coordinate,

```
E(W) = Σ_{i ∈ W} w_i     and     ρ(W) = n_obs(W) / E(W)
```

`Σ_i w_i = n_anom` **exactly**, so the correction redistributes anomalies but
never invents them. `ρ ≡ 1` is the null hypothesis. The weights are written out
per star to `results/tidemark/<channel>/selection_weights.csv` so a reader can
audit the correction rather than take it on trust.

`MatchedNull` **raises** if you try to stratify on a coordinate under test. A
silently self-cancelling test is worse than no test.

### 3.2 Two modes, both reported

* **strict** — heliocentric distance *is* a stratification covariate. A radial
  gradient must then appear as a **difference between directions at matched
  distance** (a star 1 kpc away toward `l=0` sits at `R ≈ 7.2` kpc; the same
  distance toward `l=180` is `R ≈ 9.2` kpc). Immune to every distance-dependent
  selection effect. **This is the primary claim.**
* **permissive** — distance free. More sensitive; distance-dependent
  detectability leaks straight into the radial gradient. Cross-check only.

### 3.3 The statistics

**Gradient.** A Poisson regression of the binned counts with `log E` as a fixed
offset, `log μ_b = log E_b + α + β x_b`, so `β` is the log rate-ratio gradient per
kpc and `exp(β)` the multiplicative change in rate per kpc — an **amplitude with
a confidence interval**, not just a p-value. Reported alongside a *binless*
mean-shift permutation test (no bin choices, no model) and a Spearman
monotonicity test. Longitude is periodic, so it is fitted as a harmonic dipole:
the amplitude answers "is there a preferred direction?", the phase answers "which
one?" — the form in which predictions 1 and 2 actually differ.

**Edge.** Equal-expected-count bins, then a Kulldorff-style Poisson
likelihood-ratio scan statistic across a two-window step, reported as
`S = sign · √Λ` (behaves like a Gaussian sigma). Two things make it honest:

1. **The null already contains the fitted smooth gradient.** A cubic log-rate
   trend is fitted to `ρ(x)` and the matched null is *tilted* by it, so the
   question asked is "is there a step **beyond** the smooth gradient?" Calibrated
   on injections: a pure 0.8/kpc gradient fires the edge detector 15% of the time
   with a quadratic trend model and **5% (nominal) with a cubic**, at no cost in
   bubble sensitivity. That is why `smooth_order: 3` is the default.
2. **The look-elsewhere effect is paid for.** The reported statistic is the
   **maximum** `|S|` over the entire scan — every position, every width, and for
   the 3D scan every centre — and its p-value is the distribution of *that same
   maximum* recomputed on each matched-null realisation.

Three geometries: 1D (any scalar coordinate), 3D spherical shells over a grid of
candidate centres plus the anomaly centroid (the literal bubble-boundary test),
and spherical caps on the sky (for a boundary nearer than the distance precision
resolves).

**Age.** Best available proxy, most direct first: a spectroscopic `age_gyr`;
`[α/Fe]`; kinematic heating (`|W|`, total space velocity, or tangential velocity)
via the age–velocity-dispersion relation; thick-disk/halo membership. Shape is
classified from the *fitted curve over the sampled range* — flat / rising /
saturating / turnover — by ΔAIC calibrated against the matched null. **Metallicity
is forced into the stratification** whenever it exists, because old stars are
metal-poor and metal-poor stars host fewer giant planets: a rate rising with age
could otherwise be the planet-occurrence–metallicity relation read backwards. The
test is run both with and without `|z|` matched out, and both are reported,
because scale height grows with age and the two are not fully separable.

### 3.4 Calibration on injection — into the *real* parent

Two calibration stages run against the actual dataset, not a toy, so the quoted
sensitivity is this footprint's:

* **Bubble recovery**: inject sharp-edged spheres of known radius and contrast
  and report which are recovered. On the synthetic bench with the full confounder
  stack: a radius-900 pc, contrast-5 bubble is recovered at p = 0.004 with the
  centre localised to **89 pc** and the radius to **902 pc vs 900 pc true**.
* **Gradient transfer**: inject known slopes and measure what fraction comes
  back. This is the correction's price tag, and it is substantial — see §4.

## 4. The honest limits of the selection-function correction

**This section is the one to read before believing any number this channel
produces.**

1. **The correction is conservative, and measurably so.** The detectability
   covariates are themselves correlated with Galactic position — apparent
   magnitude with distance, distance with `R_gal` through the direction of the
   Galactic centre, extinction with `|b|`. Matching on them therefore absorbs
   part of a *real* positional trend along with the selection effect. Measured on
   the synthetic bench: an injected 0.6/kpc radial gradient comes back as
   **≈0.27/kpc, a transfer coefficient of ≈0.45**. `gradient_transfer()` measures
   this per dataset and the run reports
   `R_gal_slope_deabsorbed_ln_per_kpc = measured / transfer`. **A measured slope
   is a lower bound on the intrinsic slope, never the intrinsic slope itself.**
   Quoting the raw fitted amplitude as the astrophysical amplitude would be an
   unstated systematic of order a factor of two.
2. **The smooth-trend removal in the edge test also absorbs part of a genuine
   edge.** A cubic can partially fit a step. The residual 5% false-positive rate
   under a strong gradient is nominal but not zero: an edge claim in the presence
   of a strong gradient needs the gradient reported next to it.
3. **The null is only as good as the covariate list.** Any detectability effect
   not represented by a supplied covariate is not corrected. Concretely: for a
   cross-epoch anomaly axis, high proper motion breaks positional matching and is
   also a kinematic age proxy — a purely instrumental route to an apparent age
   trend, corrected only if proper motion or epoch separation is in the covariate
   list. The channel cannot detect its own missing covariates.
4. **Strata are collapsed when thin.** With six covariates at five quantile bins
   the stratification is fine; strata below `min_pool` are progressively merged
   onto coarser stratifications. `null_diagnostics` reports `min_pool`,
   `median_pool`, `frac_anom_in_thin_strata`, `effective_parent_size` and a
   per-covariate standardised-mean-difference balance check. **If
   `frac_anom_in_thin_strata` is not ~0, the correction is not trustworthy.**
5. **The trials correction is measured, not assumed.** The statistics are
   strongly correlated — a bubble offset from the Sun produces a radial gradient,
   a longitude dipole and a shell edge simultaneously — so a Šidák correction
   over the raw count of tests would be wrong in both directions depending on the
   case. Each edge geometry therefore reports *which anomalies produced its
   step*, geometries overlapping by Jaccard ≥ 0.5 are counted once, and the
   correction runs over `n_effective_independent_tests`. This is still
   approximate: the grouping threshold is a choice, and gradient tests carry no
   firing set so they are always counted separately (conservative).
6. **Monte Carlo p-values have a resolution floor** of `1/(n_null + 1)`. A
   p-value at the floor is a **bound**, not a measurement, and is reported as
   `p < x` in `p_repr`. Floor-limited statistics are automatically re-run with 8×
   the draws up to `max_n_null`; if they survive the cap they keep the inequality
   and **cannot** satisfy the `not_floor_limited` gate. See §4A.
7. **Distance is `1/parallax`.** No Bayesian distance prior, no parallax
   zero-point correction beyond what the parent channel applied. Adequate for
   `parallax_over_error > 10`; not adequate beyond ~2 kpc.
8. **The anomaly axis inherits its own contamination.** TIDEMARK does not vet
   individual anomalies — that is the point. But if a contaminant class is itself
   spatially structured (background galaxies concentrate at high `|b|`; cirrus at
   low `|b|`; crowding in the plane), it will produce a real gradient in a real
   anomaly rate. `log_local_density`, `ebv` and the per-object photometric error
   are covariates for exactly this reason, and a detection must still be traced to
   a systematic before it is believed. **A gradient that survives is a starting
   point for contamination work, not the end of it.**
9. **The parent sample is a hard requirement.** A channel that publishes only its
   survivors has no denominator and gets `NO_PARENT_SAMPLE`. A synthetic
   denominator is never substituted — that would be fabricating data.

## 4A. Verdict semantics — what each result string commits to

The first committed TIDEMARK run emitted `verdict: DETECTION`. It should not
have, and the reasons are worth keeping in the open because every one of them is
a way a population statistic can look like a discovery while being an artefact.
What went wrong, and what now prevents it:

| Failure | Fix |
|---|---|
| p-value equal to `1/(n_null+1)` reported as a point estimate and fed to a trials correction | every p ships with `p_floor`, `floor_limited` and `p_repr`; a floor-limited statistic is **re-run with 8× the draws** (up to `max_n_null`) and, if it survives, is reported as `p < x` throughout. A bound is *not* discarded — the floor is the conservative end of it, so it enters the family at that value — but it is admitted only once escalation has actually run (`bound_was_escalated`), because an unescalated floor is an artefact of `n_null`, and the verdict carries `best_p_is_bound` |
| three "independent" edge geometries returning the *identical* p | each edge test reports which anomalies produced its step; geometries whose firing sets overlap by Jaccard ≥ 0.5 are **one feature** and the correction counts them once (`n_effective_independent_tests`) |
| 2555 anomalies in the catalogue, 30 with a parallax, and a 3D scan run on the 30 while guarded by the 2555 | every statistic counts the anomalies carrying **its own** coordinate and returns `INSUFFICIENT_ANOMALIES` with `p_value: null` and a reason the aggregator surfaces in `insufficient_tests`. Scans additionally require **5 anomalies per bin** (`min_anomalies_for_scan`): a 24-bin scan needs 120, because the maximum over hundreds of near-empty windows is not a statistic |
| `p = None` silently skipped by the aggregator | insufficient tests are a first-class entry; they never appear in `p_values` and never contribute to a verdict |
| the tested coordinate was the *worst*-balanced covariate (SMD 0.197) and this was buried in diagnostics | `coordinate_balance` travels next to every p-value, graded good/marginal/poor on the Rubin (2001) convention, and `|SMD| ≥ 0.10` blocks `DETECTION` |
| the channel declared `[g_mag, bp_rp, n_epochs]` but the code used the global list whose magnitude column is `phot_g_mean_mag`, so **apparent magnitude was never matched on at all** | `_resolve_covariates` unions the channel's declared columns with the global defaults, resolves them into physical *families* through an alias table, and **refuses `DETECTION` if no magnitude covariate exists** |
| the anomaly set was a bare top-1% score cut | `anomaly_definition` is recorded (`percentile_cut` / `score_threshold` / `vetted_candidate_list` / `explicit_mask`); only a vetted list sets `vetted=True`, and only a vetted population can earn `DETECTION` |

A percentile cut deserves its own note. It selects, by construction, exactly the
fraction of the parent you asked for, whatever the data looks like. Spatial
structure in the top percentile of a score is therefore at least as likely to
trace the survey's footprint, cadence and depth as anything on the sky — and for
`dimming_secular` specifically, `STATUS.md` records that the population sits **at
the ZTF systematics floor** (19 marginal faders assessed, 18 of 19 not confirmed
in a second band). That channel now carries `caveat_tag: AT_SYSTEMATICS_FLOOR`
into its result string, so the caveat cannot be lost between the doc and the JSON.

### The vocabulary

| Result | Means |
|---|---|
| `DETECTION` | resolved, family-corrected, covariate-balanced structure in a **vetted** population, with every gate passed |
| `STRUCTURE_UNRESOLVED` | the winning statistic is a bound that was never escalated — its p-value is an artefact of the draw count, not a measurement |
| `STRUCTURE_UNCORRECTED` | significant, but the parent lacks a covariate the null needs (typically magnitude): may be a depth map |
| `STRUCTURE_UNVETTED_POPULATION` | significant, but the anomaly set is a bare score percentile |
| `STRUCTURE_CONFOUNDED` | significant, but the tested coordinate is itself poorly balanced |
| `CLEAN_NULL` | no structure beyond the parent-matched null |
| `NOT_TESTABLE` | no statistic could be computed |

`verdict_gates` and `failed_gates` are always written, so a non-detection says
*which* gate stopped it rather than leaving the reader to guess.

## 5. What can be discriminated, and what cannot

| Prediction | Discriminated? | How |
|---|---|---|
| Ćirković & Bradbury 2006 (outer rim) | **Yes** | sign of `β(R_gal)`, strict mode |
| Wright et al. 2021 (Galactic centre) | **Yes** | opposite sign of the same statistic |
| Wright et al. 2014 (shear mixes everything) | **Yes** | a calibrated clean null in gradient *and* edge, with the injection sensitivity quoted |
| Carrigan 2010 / Landis 1998 / Hanson et al. 2021 (bounded region) | **Yes, if the boundary is inside the sampled volume** | 3D shell scan; sky-cap scan if it only projects |
| Olson's cosmological expansion series | **No** | extragalactic; wrong scale entirely |
| Carroll-Nellenback et al. 2019 settlement dynamics | **Partially** | it predicts front geometry, not an observable rate; only its spatial-coherence consequence is testable here |

Two structural limits on scope. **A boundary larger than the sampled volume
appears as a gradient, not an edge** — the shell scan cannot see a sphere whose
radius exceeds the sample. And **a front moving faster than the anomaly's
observable lifetime leaves no coherent structure at all**, which is precisely
prediction 3; TIDEMARK cannot distinguish "was never there" from "was mixed away".

## 6. Interface

Any anomaly catalogue with a defined parent sample can be tested. Adapters are
declarative in `config/tidemark.yaml`:

```yaml
channels:
  cenotaph_grey_deficit:
    parent: results/cenotaph/greyfit.parquet   # every star searched
    score_col: grey_sigma                      # continuous, on the parent
    score_min: 3.0
    covariates: [phot_g_mean_mag, dist_pc, bp_rp, ebv, log_local_density]
```

Onboarding a channel is adding a block, not writing code. In Python:

```python
cat = ingest.from_frames("my_axis", parent_df, mask=my_bool_mask)
res = analyse_catalogue(cat)
```

The union over channels is supported and matches on `n_channels_searched`,
because a star searched by three channels has three chances to be flagged and
the union would otherwise map which patch of sky each channel happened to cover.

**Verdicts** are first-class: `OK`, `NO_PARENT_SAMPLE`, `EMPTY_ANOMALY_SET`,
`INSUFFICIENT_ANOMALIES` (< 30 anomalies), `NO_POSITIONS`, `NO_DATA_REACHED`.
Untested channels are counted as **untested**, never as nulls.

Deliberately excluded, and why (recorded in `config/tidemark.yaml` so the
omission is a decision rather than an oversight): `derelict` (solar-system bodies,
no sky position — a Galactic rate is undefined), `midden` (named stars, no
`ra`/`dec` in the scored table), `herdsman` (rows are convergence *events*, not
stars), `shroud` (the parent's own selection function is that channel's object of
study).

## 7. Running it

```bash
python -m seti.cli tidemark-acquire --grid sparse   # runner only: parent sample
python -m seti.cli tidemark-search  --from-parent 'results/tidemark/parent/*.parquet'
python -m seti.cli tidemark-run                     # sweep configured channels
```

`.github/workflows/tidemark.yml` runs both legs: a `fail-fast: false` matrix over
interleaved cone shards for acquisition (a lost shard costs uniform sky coverage,
not a footprint change), then a reduce stage that **fits the W1−W2 excess locus
once across the whole sky**. Fitting it per cone would normalise every field to
its own median and delete exactly the field-to-field rate differences this
channel exists to measure — this is the single most important line in
`acquire.py`, and `tests/test_tidemark.py` asserts it against the per-cone
counterfactual.

## 8. Standing directive

The deliverable is a **detection, not a limit**. If every axis comes back null,
that is a reason to change the question, not to write an occurrence-limit paper.
But a *measured, selection-corrected* gradient or edge in a real anomaly
population would be the first such measurement in the field either way — and a
calibrated clean null is itself the first observational test of Wright et al.
2014's shear-mixing prediction. It is built to be decisive rather than
suggestive: an amplitude with a confidence interval, an injection-measured
transfer coefficient, a per-star selection weight anyone can audit, and a
sensitivity quoted from injections into the real footprint.
