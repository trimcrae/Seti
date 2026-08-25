# LOOM-CATALOGUE — the momentum ceiling over the whole small-body catalogue

*Two objects above a ceiling mean nothing without a denominator.*

**Channel status:** built, offline-tested (68 tests), **not yet run against live
data**. Runner-only (`.github/workflows/loom-catalogue.yml`, `workflow_dispatch`);
the development sandbox has no egress to JPL.

---

## 1. The question

LOOM carries two standing exceedances. From `results/loom/calibration.json`
(2026-07-30):

| object | eps_eff | S/N | arc | n_obs | U | T_J | literature |
|---|---|---|---|---|---|---|---|
| `875163 (1998 SH2)` | **1.578** | 14.3 | 9,900 d | 394 | 0 | 2.913 | not found |
| `428209 (2006 VC)` | **1.304** | 3.8 | 7,113 d | 175 | 0 | 3.720 | not found |

Both are above the ε = 1 hard radiation momentum ceiling — sunlight cannot drive
them — both have clean orbit solutions, and `loom-litcheck` searched the
dark-comet literature and the arXiv corpus and returned
`NOT_FOUND_IN_SEARCHED_LITERATURE` for each.

They came out of **one query**: `sb-cdata={"AND":["A2|DF"]}`, 939 rows. And a
tail with two entries and a tail with two hundred are the same list until you
know three things:

1. **how many objects were screened** — the denominator;
2. **what the bulk of the distribution looks like** — the shape they are in the
   tail of;
3. **how many the tail was expected to hold** — because a tail that is fuller
   than an ordinary thermal-recoil population predicts is a statement about the
   *tail*, not about any object in it.

This channel measures all three, over every small body in JPL's Small-Body
Database with a fitted non-gravitational parameter.

**It cannot promote anything.** That is stated first because it is the most
important constraint on how the output may be read. LOOM's tier ladder
(`docs/loom.md` §3.5) requires an *artificiality* channel — an anomalous
area-to-mass ratio, or an acceleration independent of heliocentric distance —
and SBDB's query fields carry neither. "Inactive small body with an acceleration
above Yarkovsky expectations" is Seligman et al. (2023)'s dark comets, that
population is **known to be incomplete** (Farnocchia & Seligman 2024 extended it
and found two distinct classes), and a magnitude cut will keep finding
unpublished members of it. The ceiling tops out at `interest`. The tier
assignment is not reimplemented here: `record_from_entry` builds a
`seti.loom.screen.ObjectRecord` and calls `seti.loom.screen.assign_tier`, so this
screen *cannot* promote on magnitude alone even by accident.

---

## 2. What is different from `loom-calibrate`

### 2.1 The denominator is counted, not assumed

JPL's SBDB query API will serve `spkid,full_name,pdes,kind,class,A1,A2,A3` for
every small body with no constraint at all. This is not a hope: this repository's
DERELICT channel measured it on the same API
(`results/derelict/completeness.json`, 2026-07-30) — **1,553,263 asteroids** and
**4,069 comets** in one request each.

So the number of objects with each non-gravitational parameter fitted is a
*counted* quantity, and — the part that matters — the constrained queries can be
**checked against it object by object**. A `sb-cdata` constraint that quietly
misses rows turns a population screen into a statement about the query, and the
failure is invisible from inside the constrained result: there is nothing to
compare it to. Comparing against the unconstrained census makes it a set
difference. `completeness_check` reports `CONSTRAINT_COMPLETE`,
`CONSTRAINT_INCOMPLETE` (with the missing keys) or `NO_CENSUS` — the last of
which says plainly that the denominator is *assumed*.

DERELICT ran exactly this check on its `A1` census and got
`CONSTRAINT_COMPLETE`, which is why its result could be believed.

### 2.2 All three components, not `A2` alone

The momentum ceiling bounds the **total** non-gravitational acceleration. JPL's
model is

```
a = ( A1 r̂ + A2 t̂ + A3 n̂ ) · g(r)
```

— three orthogonal components. Screening on `|A2|` alone understates every
object with a fitted radial or normal term, which is most comets and a handful
of asteroids. This screen uses `|A| = sqrt(A1² + A2² + A3²)` over whichever
components were fitted, and names them.

The uncertainty is propagated first-order,
`σ_|A|² = Σ (A_i/|A|)² σ_i²`, which **ignores the covariances**. SBDB's query API
does not serve the covariance matrix (the per-object `sbdb.api` endpoint does,
one object at a time — `fetch_covariance` records whether it is there). The
components of a non-gravitational solution are correlated, often strongly, so
this signal-to-noise is an approximation and is labelled as one. It is used as a
*gate*, never as a detection statistic, and where `A2` is the only fitted
component — the overwhelming majority of asteroids — it is exact and reduces to
`|A2|/σ_A2`, the quantity `calibrate.vet_exceedance` already gates on.
`test_vet_reduces_to_calibrate_for_a2_only_rows` pins that reduction across nine
spoiled-row cases.

### 2.3 Where the comparison is made, and when that matters

`A1/A2/A3` are the *coefficients* of `g(r)`, so "the acceleration at 1 au" equals
`|A|` only if `g(1 au) = 1`. It does, for both of JPL's laws:

| law | `g(1 au)` |
|---|---|
| radiation, `(1 au / r)^d`, `d = 2` | 1 (exactly) |
| radiation, `d = 2.25` | 1 (exactly) |
| water-ice sublimation, `α (r/r₀)^−m [1+(r/r₀)^n]^−k` | **1.0000000024** |

The first two are trivial and are the reason the `d = 2` versus `d = 2.25`
ambiguity in JPL's Yarkovsky fits never reaches the ceiling comparison. The third
holds only because `α = 0.1112620426` was chosen to make it so — a fact worth
*computing* rather than believing, because if it were false every comet in this
screen would be compared against the wrong ceiling by a constant factor.
`g_normalisation()` computes it from `seti.loom.residuals`'s own constants and
`test_g_laws_are_normalised_at_one_au` pins it.

**That law-independence holds only at 1 au.** Since the ceiling itself falls as
`r^−2`,

```
eps(r) = eps(1 au) · g(r) · r²
```

For the radiation law the `r²` cancels `g` exactly, so the ratio is the same
everywhere — which is the whole reason `A2` is the natural quantity for a
Yarkovsky screen, and the reason the 1 au number needs no defence for an
asteroid. For the sublimation law it does not cancel: the acceleration falls off
far faster than the sunlight does. At `r = 3 au` the ratio is **65× smaller** than
at 1 au; at `r = 0.5 au` it is 14% larger. A comet whose perihelion is 3 au and
which never comes near 1 au is being credited at 1 au with an efficiency it never
realises anywhere.

So every entry reports both `epsilon_1au` and `epsilon_at_perihelion`, with the
law flagged as **inferred from the object's kind**, because the query API does
not serve which `g` JPL actually fitted. For the two standing exceedances — both
asteroids on radiation-law fits — the distinction does not arise.

### 2.4 Every reliability cut reports what it removed

A cut that quietly eats 90% of the catalogue changes the meaning of every
fraction downstream. `cut_ledger` reports, for each gate:

* **`n_failing_alone`** — how many objects fail this gate ignoring every other
  one. Order-free, and the right number for "is this cut doing anything?".
* **`n_removed_in_sequence`** — how many it removes applied in pipeline order,
  after the earlier ones. The right number for "where did the population go?",
  and it is what sums to the total.

The two differ wherever cuts are correlated, and for a small-body catalogue they
heavily are: short arcs, few observations and a poor condition code are three
views of the same objects. A large marginal count with a small sequential one
means the gate is redundant against an earlier one; the reverse is impossible.

The gates themselves are **reused** from `seti.loom.calibrate`, not restated:

| gate | threshold | why |
|---|---|---|
| `has_nongrav_fit` | any of A1/A2/A3 fitted and non-zero | zero is *absence*, never a measured zero |
| `has_usable_size` | a measured diameter, or an `H` | no size, no ceiling |
| `nongrav_snr` | `\|A\|/σ ≥ 3` | Del Vigna et al. (2018) condition 1 |
| `orbit_rms` | `≤ 0.8″` | Catalina's astrometric RMS is ~0.69″ |
| `data_arc` | `≥ 3650 d` | Yarkovsky needs many apparitions |
| `n_obs_used` | `≥ 100` | |
| `condition_code` | `≤ 2` | MPC `U` parameter, 0 = best |

The reason these are not fussiness: a blind search for Yarkovsky signal in
minor-planet astrometry returns a **majority** of spurious detections at nominal
S/N > 3, and short arcs with sparse or isolated astrometry are the documented
cause. That is why Del Vigna et al. require two conditions rather than one.

### 2.5 Was the tail expected to be this full?

`tail_expectation` fits a log-normal to the *core* of the efficiency distribution
and extrapolates it to each threshold. Ordinary thermal recoil is positive,
multiplicative in several roughly independent factors (size, thermal inertia,
obliquity, spin rate) and spans a decade, so a log-normal is the natural
reference — and the population LOOM measured on 2026-07-30 (median 0.074, p90
0.143 at ρ = 2000) is consistent with one.

The fit uses **the lower half only**: the median, and the scale from the gap
between the 25th percentile and the median. Fitting a scale that includes the
tail would let the objects under test inflate their own expectation, which is the
classic way to make an excess disappear.

How robust that is, stated exactly rather than implied:

* completely immune to the tail's **values** — an object at ε = 50 and one at
  ε = 5×10⁴ move nothing (`test_..._immune_to_the_tails_values`);
* immune to the tail's **count** only to first order, because every added object
  shifts the quantile *positions*. A tail holding a fraction `f` of the sample
  moves the 25th percentile to the core's `0.25/(1−f)` and the median to its
  `0.5/(1−f)`, which at `f = 0.05` inflates σ by ~3.5%. That bias raises the
  expected tail count and so makes an excess **harder** to claim, not easier —
  and at `f = 0.05` the verdict is `TAIL_DENSELY_POPULATED` anyway, at which
  point a log-normal reference has stopped being the interesting question.

Three things are reported and all three are needed: expected versus observed at
each threshold with a Poisson probability; a **model check** at the sample's own
90th and 99th percentiles, so a reader can see whether the log-normal already
misses *before* trusting its extrapolation to ε = 1; and the fitted parameters,
so the extrapolation can be redone.

The Poisson p is **descriptive**. It has no trials correction and the log-normal
is a reference, not a theory. It is there to answer "is four exceedances a lot?",
not to detect anything.

---

## 3. Asteroids and comets are screened apart, everywhere

This is not a refinement, it is the point, and it is the same split
`calibrate.summarise_epsilon` makes. A comet accelerates by **shedding mass**, so
it is not subject to the radiation momentum budget at all and exceeds it by
orders of magnitude — all 81 comets in the 2026-07-30 calibration run were above
the hard ceiling, and reporting them together with the asteroids produced a
headline of "92 objects above the hard ceiling" that read like the gate failing
when it was the gate working exactly as designed.

The split therefore runs through the whole output:

* separate distributions, cut ledgers, tail expectations and tier counts;
* **separate tail rankings**. A merged ranking would be arithmetic rather than a
  comparison: a comet at ε = 10⁴ is not "more anomalous" than an asteroid at
  ε = 3, and a merged list capped at 400 would push every asteroid off the page
  with objects whose exceedance is expected of them — and would report the two
  standing objects as ranked several hundredth.
* **survivors are asteroids only.** Feeding hundreds of comets to a literature
  search would bury the handful of objects the search exists for.

**Dark comets are unaffected by all of this**, and that is the entire difficulty:
they are classified as *asteroids*. They are the contaminant this channel has to
reject by argument — Tisserand parameter, comet-like dynamics, the literature —
rather than one it can filter out by kind.

---

## 4. What the output says

`results/loom-catalogue/catalogue.json`, plus `catalogue_objects.csv` (one row per
screened object, so the distribution can be re-binned or re-plotted offline
without re-querying JPL).

| verdict | meaning |
|---|---|
| `NO_DATA_REACHED` | JPL did not answer. A **dead fetch**, not a null result: the standing exceedances are neither confirmed nor withdrawn by it |
| `NO_SCREENABLE_POPULATION` | rows came back but none has both a fitted parameter and a usable size |
| `NOTHING_ABOVE_CEILING` | no asteroid exceeds ε = 1 |
| `ALL_EXCEEDANCES_ALREADY_IDENTIFIED` | everything above the ceiling is unreliably fitted or already known to be anomalous |
| `TAIL_SPARSE_SURVIVORS_PRESENT` | some exceedances are reliable and unidentified — the case that goes to `litcheck` |
| `TAIL_DENSELY_POPULATED` | **more than 5% of screened asteroids exceed the ceiling.** Exceeding it is then a common property of this population, and no individual exceedance — including the two standing ones — is remarkable on magnitude alone |

`TAIL_DENSELY_POPULATED` is defined on the *fraction*, not the count: a tail of
40 in 600 is a population, a tail of 40 in 60,000 is a tail. It exists because
the expensive error here is not a false null, it is reporting a long list of
"exceedances" as though each were a lead, when what the list means is that the
dark-comet population is large and unpublished and the ceiling alone cannot
separate it. **If the tail is crowded, that is the finding**, and per `CLAUDE.md`
it is a reason to change the question rather than a paper.

The `standing_exceedances` block answers the commissioning question directly, for
each of `875163` and `428209`: its efficiency now that all three components are
used, its rank **among asteroids** above the ceiling, how many sit above it, its
percentile in the screened asteroid population, and — if it is no longer in the
screened set at all, which a re-fit since 2026-07-30 could cause —
`NOT_IN_SCREENED_POPULATION`, which is a finding with a name and not a blank.

Every tail entry also carries `sensitivity`, the assumption grid from
`calibrate.sensitivity_grid`: an object is above the ceiling *given a density and
a size*, and an exceedance that exists only at ρ = 2000 has not been established,
it has been assumed. `robust_above_ceiling` means it survives the whole grid.

---

## 5. The query, and what it should return

Two passes, eight requests, checkpointed after every one.

**Detail pass** — six requests, `{A1, A2, A3} × {asteroid, comet}`:

```
GET https://ssd-api.jpl.nasa.gov/sbdb_query.api
    ?sb-kind={a|c}
    &full-prec=1
    &sb-cdata={"AND":["{A1|A2|A3}|DF"]}
    &limit=200000
    &fields=spkid,full_name,pdes,name,kind,class,neo,pha,H,diameter,
            diameter_sigma,albedo,a,e,i,q,om,w,moid,A1,A2,A3,DT,rms,
            n_obs_used,data_arc,condition_code,epoch
            [,A1_sigma,A2_sigma,A3_sigma,DT_sigma,n_del_obs_used,n_opp,
             first_obs,last_obs,producer,two_body,rot_per,GM,spec_B,spec_T]
```

Expected: **~600 asteroids** and **~350 comets** for `A2|DF` (measured: 939 rows
for the unsplit query, of which 589 asteroids had a usable diameter and 349 were
comets); **22 asteroids** and **272 comets** for `A1|DF` (both measured by
DERELICT); a handful for `A3|DF`. **Union after de-duplication: roughly
1,000–1,400 objects.** Duplicates are expected and are merged on `spkid` — an
object with both `A1` and `A2` comes back from two queries, and counting it twice
would inflate the denominator and double its tail entry.

**Census pass** — two requests, issued **last**:

```
GET .../sbdb_query.api?sb-kind={a|c}&full-prec=1
    &fields=spkid,full_name,pdes,kind,class,A1,A2,A3
```

Expected: **1,553,263** and **4,069** rows. This is the only slow request in the
job and it is issued last on purpose — a job that dies here still has the screen
on disk, whereas a job that dies in the detail pull has nothing.

Three things the fetch layer refuses to guess at, each because a guess has cost a
run in this repository before:

* **Field names.** One invalid name returns 400 for the *whole* query, so a
  single wrong guess costs every field and the run comes back reading "no object
  has a fitted A2" — which is what happened to the second calibration run. The
  API names the field it objected to, so the request drops that name and retries,
  and every drop is recorded. Measured facts: `A2_sigma` is accepted;
  `sigma_A1`, `sigma_A2`, `sigma_A3`, `sigma_DT` are all rejected
  (`results/derelict/queries.json` has the 400s by name). `A1_sigma` and
  `A3_sigma` are therefore requested optimistically and will be *discovered*, not
  assumed. The documented `info=field` probe runs first and prunes the optional
  list before the first real request where the server answers it; where it does
  not, the repair loop handles it, so discovery is an optimisation and never a
  dependency.
* **Truncation.** A response of exactly `limit` rows is a row *cap*, not a row
  count. Read as a population it would silently understate the denominator — the
  one number this whole channel exists to establish — so it is flagged
  `truncated` with the note `TRUNCATED`.
* **Whether a chunk finished.** A cancelled GitHub Actions job never runs its
  commit step, so a stage that only writes at the end loses everything it
  learned; this cost TOCSIN a three-hour backfill. Every chunk checkpoints to
  disk the moment it returns, `completed_chunks` records what landed, and a
  re-run skips them. A run that did *not* finish every chunk carries its rows in
  `resume_rows` and says on the record that **every fraction it reports is over a
  partial pull**; a run that finished drops that payload, because
  `catalogue_objects.csv` is the re-derivable artefact and a megabyte of
  duplicated raw input has no business in git.

---

## 6. Feeding `litcheck`, and what "survivor" is allowed to mean

Survivors are: above the hard ceiling, **an asteroid**, **reliably fitted**, and
**not already annotated** as something known (`calibrate.annotate` recovers
'Oumuamua, 362P, Phaethon, Elst-Pizarro and the published dark comets). Each
carries its Tisserand parameter and `comet_like_dynamics` flag, which is the one
discriminator in the whole screen that assumes **no density and no albedo**:
`T_J < 3` is comet-like dynamics, the object is Jupiter-coupled, and hidden
outgassing is the natural reading.

The workflow chains `loom-litcheck` on those names, into the same output
directory, so a survivor list is never published without the literature check
that decides whether it is a lead at all. It writes to
`results/loom-catalogue/litcheck.json` and deliberately **not** to
`results/loom/litcheck.json`, which the scheduled `loom-litcheck` run owns.

`NOT_FOUND_IN_SEARCHED_LITERATURE` is a statement about the searches as much as
about the object, and the dark-comet population is *known* to be incomplete — so
an object absent from the literature is not thereby unexplained. That is why
`875163 (1998 SH2)`'s `T_J = 2.913` matters: it is comet-like dynamics, and the
honest reading of an unexplained acceleration on such an orbit is a dark comet
nobody has published yet. `428209 (2006 VC)` at `T_J = 3.720` is asteroidal,
which is the harder case to explain away and the more interesting one — and also
the one with `S/N = 3.8`, barely over the reliability gate.

---

## 7. Files

| file | what |
|---|---|
| `src/seti/loom/catalogue.py` | the screen: fetch, magnitude, cuts, distribution, tail expectation, verdict, orchestration |
| `tests/test_loom_catalogue.py` | 68 offline tests, including the fetch layer through a fake transport |
| `.github/workflows/loom-catalogue.yml` | `workflow_dispatch` + monthly (`10 17 2 * *` = **13:10 ET on the 2nd**) |
| `results/loom-catalogue/catalogue.json` | the record |
| `results/loom-catalogue/catalogue_objects.csv` | one row per screened object |

Reused from the existing channel rather than reimplemented:
`calibrate.epsilon_effective`, `calibrate.diameter_m`, `calibrate.vet_exceedance`,
`calibrate.sensitivity_grid`, `calibrate.tisserand_j`,
`calibrate.del_vigna_ratio`, `calibrate.is_comet`, `calibrate.annotate`,
`calibrate._INVALID_FIELD`, `residuals.g_comet`, `residuals.g_radiation`,
`screen.ObjectRecord`, `screen.Thresholds`, `screen.assign_tier`,
`nongrav.EPSILON_*`, and `run._write_json` / `run.load_loom_config` /
`run.thresholds_from_config`.

Four regression tests are load-bearing and should not be weakened:
`test_standing_exceedances_reproduce_the_calibration_run` (pins 1.578 and 1.304),
`test_vet_reduces_to_calibrate_for_a2_only_rows` (pins the reliability gate to
`calibrate`), `test_g_laws_are_normalised_at_one_au` (pins the fact that makes
the 1 au comparison law-independent), and
`test_screen_never_promotes_to_candidate` (pins LOOM's central rule).

---

## 8. Related

* `docs/loom.md` — the channel this extends; §3.1 is the ceiling, §3.5 the tier
  ladder, §2.3 the calibration run that produced the two standing exceedances.
* `docs/derelict.md` — the catalogue-scale `A1` census on this same API, whose
  completeness discipline and measured row counts this channel builds on.
