# DERELICT — thin-film debris, found by what radiation pressure does to it

Necrosignature **S19** (`docs/necrosignatures.md`): *dead lightsails — large
non-gravitational acceleration with no outgassing.*

---

## 1. The claim

A derelict lightsail or thin-film structure has an **enormous area-to-mass
ratio**. Solar radiation pressure therefore gives it a large **radial**
non-gravitational acceleration **with no outgassing**. A natural body cannot do
this: the gap is four to five orders of magnitude, and — this is the point — it
is a gap in a quantity JPL *already fits and publishes* for every small body
with enough astrometry.

### 1.1 The conversion chain

JPL fits the Marsden, Sekanina & Yeomans (1973) decomposition

```
a_ng = A1 g(r) r̂  +  A2 g(r) t̂  +  A3 g(r) n̂
```

with `A1` the **radial** term and, for asteroids, `g(r) = (1 au / r)²` — so `A1`
is literally the radial acceleration at 1 au in au/day².

Solar radiation pressure is radial and falls as `1/r²`, *the same functional
form as gravity*, so it is written as the dimensionless ratio

```
β ≡ F_rad / F_grav = [L_sun Q_pr / (4π c G M_sun)] · (A/m)
```

Equating `β GM_sun/r²` to `A1 g(r)` at `r = 1 au` gives the two conversions the
whole channel rests on:

```
β    = A1 / GM_sun[au³/day²]        =  3379.38  · A1[au/day²]
AMR  = β · 4π c GM_sun / (L_sun Q_pr) = 1306.08 · β / Q_pr    m²/kg
⇒ AMR = 4.4137e6 · A1[au/day²] / Q_pr                          m²/kg
```

with `GM_sun = k² = 2.9591220828559115e-4 au³/day²` (k = Gauss's constant),
`L_sun = 3.828e26 W`, `c = 2.99792458e8 m/s`, `GM_sun = 1.32712440018e20 m³/s²`.

A solid sphere of diameter `D` and bulk density `ρ` has

```
AMR_natural(D, ρ) = 3 / (2 D ρ)
```

These are implemented in `src/seti/derelict/radiation.py` and unit-tested
against the table below in `tests/test_derelict.py`.

### 1.2 The benchmark table

| Object | AMR (m²/kg) | β | Q_pr |
|---|---|---|---|
| Natural sphere D=100 m, ρ=2000 | 7.5×10⁻⁶ | 5.7×10⁻⁹ | 1 |
| Natural sphere D=10 m, ρ=2000 | 7.5×10⁻⁵ | 5.7×10⁻⁸ | 1 |
| **1I/'Oumuamua (if pure SRP)** | **1.08** | **8.3×10⁻⁴** | 1 |
| IKAROS sailcraft (196 m², 310 kg) | 0.63 | 9.7×10⁻⁴ | 2 |
| Bare 1 µm mylar | 714 | 1.09 | 2 |

**An IKAROS-class sailcraft sits at essentially the same β as 'Oumuamua, and
both are 4–5 orders of magnitude above any natural 10–100 m body.**

### 1.3 The cross-check that validates the chain

Micheli et al. 2018 (Nature 559, 223) give 1I/'Oumuamua a radial non-grav
acceleration of `4.92e-6 m/s²` at 1 au, i.e. `A1 = 2.4551e-7 au/day²`. Pushing
that through the chain:

```
β   = 8.297e-4
AMR = 1.0836 m²/kg
m/A = 0.9228 kg/m² = 0.0923 g/cm²
```

Bialy & Loeb 2018 (arXiv:1810.11490, ApJL 868, L1) quote **~0.1 g/cm²**. The
chain reproduces it to **7.7%**. That agreement is asserted as a test
(`test_oumuamua_reproduces_bialy_loeb_surface_density`), so any regression in
the constants fails CI.

---

## 2. Novelty verdict

**Every ingredient exists separately. Nobody has combined them.** The three
antecedents and exactly where each stops:

**Bialy & Loeb 2018** did precisely this `A1 → β → AMR → lightsail` inference —
**for 1I/'Oumuamua alone**. It was never generalised to a catalogue.

**JPL and the MPC** run the same area-to-mass test routinely, but *reactively
and per-object*, to unmask **human** hardware: 2020 SO (a Centaur upper stage
from Surveyor 2), J002E3 (an Apollo 12 S-IVB), WT1190F. Never systematic, never
framed as a technosignature search.

**The dark-comet papers selected the OPPOSITE population.** Seligman et al. 2023
(arXiv:2212.08115, PSJ 4, 35; 7 objects) and 2024 (PNAS 121, e2406424121; 14
total) built the only catalogue of coma-free non-gravitationally accelerating
objects — but selected on large **non-radial** acceleration, precisely the
signature that **excludes** radiation pressure. Radiation pressure is purely
radial; an object with a large A2 or A3 is being pushed by something that is not
sunlight.

> **The primary target is therefore the set the dark-comet literature threw
> away: significant A1, with A2 and A3 consistent with zero, and no coma.**

### 2.1 Status of the verdict — read this before citing it

The offline evidence base in this repository **cannot currently support the
dark-comet half of the claim**, for a specific and correctable reason:

`results/necrolit/{ar5iv,txt,arxiv_id}_dark_comets.*` are **the wrong paper**.
`scripts/necrolit_fetch.py` carried arXiv id `2306.16966`, which is
*"Self-interacting dark matter implied by nano-Hertz gravitational waves"*
(hep-ph) — a dark-matter/dark-comet mixup. The fetch *succeeded*, so nothing
flagged it. Verbatim check on that file: `comet` 0 hits, `Seligman` 0,
`Oumuamua` 0, `non-gravitational` 0.

That id is now corrected in `scripts/necrolit_fetch.py`, and
`scripts/derelict_lit.py` re-fetches all four decisive sources **with an
id-and-title verification step** that records a `id_mismatch` rather than
writing the wrong paper to disk.

**That verification immediately caught a second instance.** In run 30203392288
the id guessed for the Seligman PNAS 2024 companion, `2412.02384`, resolved to
*"Theory building for empirical software engineering in qualitative research:
Operationalization"*. Two wrong-paper incidents from two guessed ids, both of
which *fetched successfully*, is the pattern: **an arXiv id recalled from memory
is not evidence.** That paper is therefore now resolved **by title search** at
fetch time, with the id read off the match, and no id for it is stored anywhere.

Until the corrected fetch has been read, the components of the verdict stand as:

| Component | Status |
|---|---|
| Seligman et al. selected on non-radial A2/A3 | **UNVERIFIED** — must be read from the real source |
| They never compute AMR/β, never test an A1-only model | **UNVERIFIED** — same |
| No catalogue-scale AMR/β/lightsail search of solar system bodies exists | **SUPPORTED** — a sweep of ~35 literature directories finds none; the targeted arXiv query `lightsail AND search AND solar system` returns **0 results**; `high area-to-mass` returns 0 tree-wide |
| Bialy & Loeb applied the inference to 1I alone | **SUPPORTED (weakly)** — cited on disk only as a single-object claim |

### 2.2 The live novelty risk, stated plainly

**Loeb & Cloete 2025 (arXiv:2503.03552), *"Is the 'Dark Comet' 2005 VL1 the
Venera 2 Spacecraft?"*** argues that one dark comet is artificial, and a 2026
technosignature review already discusses it. So **"a dark comet might be
artificial" is already published**, and this channel must not claim that idea as
its own.

What remains unoccupied, and what this channel therefore claims, is the
**method**: a *systematic, catalogue-scale* selection of the A1-only complement,
converted to β and area-to-mass, and normalised by object size into a single
statistic. Loeb & Cloete identify one object by orbit-matching it to a known
human spacecraft — not by an `A1 → β → AMR` inference, and not over a
population. `scripts/derelict_lit.py` fetches that paper explicitly to check
whether they computed an AMR; if they did, this section gets narrowed again.

### 2.3 An honest coverage gap

The repository's literature sweeps are SETI-oriented. The **space-situational-
awareness corpus on HAMR (high area-to-mass ratio) debris** — where the
engineering version of this test lives — was never swept, and `HAMR` returns
zero hits locally. `scripts/derelict_lit.py` adds four queries targeting it.
That corpus is about Earth-orbiting debris rather than heliocentric small
bodies, so it is unlikely to contain the search; but "unlikely" is not
"checked", and it is recorded here as a gap rather than glossed.

---

## 3. The search

### Screen 1 — the A1-only complement

```
A1/σ_A1 > 3   AND   |A2|/σ_A2 < 1   AND   |A3|/σ_A3 < 1   AND   no coma
```
plus `condition_code ≤ 4`, `data_arc ≥ 30 d`, `n_obs ≥ 30` to kill short-arc
garbage. Missing orbit-quality metadata **fails the gate** — a spurious A1 is
overwhelmingly a badly-constrained orbit, so absence is not a pass.

**The honesty rule that shapes the implementation:** "A2 was fitted and came out
consistent with zero" and "A2 was never fitted" are *different statements*, and
only the first is evidence. JPL's asteroid non-grav solutions are overwhelmingly
A2-only (Yarkovsky), so the fully-constrained subset is small. Every row carries
`nonradial_constrained`, and the funnel reports `screen1_a1_only` and
`screen1_a1_only_strict` separately. An object with an unfitted A2 is never
quietly promoted to "A2 is zero".

### Screen 2 — the normalised outlier statistic (the real discriminant)

```
R = AMR_implied / AMR_natural(D, ρ)
```

`R ≈ 1` natural, `R >> 1` anomalous. This is what turns a meaningless 1-D `A1`
cut into a properly normalised test: a 3 m rock genuinely *does* show a
detectable radiation-pressure signal, while a 3 km one cannot, and only `R`
knows the difference.

`R` is **linear in D**, so a darker assumed albedo (a larger body, more mass
behind the same cross-section) makes the same measured acceleration *more*
anomalous. When `D` comes from `H` rather than a published diameter, `R_lo`/`R_hi`
bracket the albedo range 0.05–0.60 and `screen_r_flag_conservative` uses `R_lo`.

Two conservative choices are baked in, both in the direction that makes the
anomaly *harder* to claim: `Q_pr = 1` (a reflective sail needs only **half** the
area-to-mass for the same β) and `ρ = 1000 kg/m³` (a low density raises
`AMR_natural` and so lowers `R`).

### Screen 3 — the negative-A1 census

Radiation pressure **cannot push sunward**. Any significant `A1 < 0` is a
systematic or something very strange. Cheap, never published — and it doubles as
the channel's own **empirical false-positive floor**: the rate at which orbit
fits manufacture a spurious A1 of the wrong sign bounds the rate at which they
manufacture one of the right sign.

### Screen 4 — impossible albedo

Geometric albedo > 0.7 is impossible for natural regolith and trivial for
aluminised film. (Fresh ice and E-type enstatite reach ~0.6.) Where an albedo
uncertainty exists, the *excess* must be significant, not just the point
estimate.

### Control sample

Comets are pulled through the **identical** machinery. They *should* light up —
their radial acceleration is real and is outgassing. If the comet control does
not fire, the pipeline is broken, not the sky.

---

## 4. Contamination model

Rejection routes, in descending order of how much parameter space they own.
Implemented in `src/seti/derelict/vet.py`.

**1. Human hardware — the dominant contaminant, not a minor one.** JPL/MPC
already use exactly this test to unmask spent upper stages that received
asteroid designations. A high implied area-to-mass on a low-e, low-i, ~1 au
orbit is a rocket body until spectroscopy says otherwise (`ARTIFICIAL_SUSPECT`).
Known cases are on a documented watchlist and are used as **positive controls**:
a pipeline that fails to flag 2020 SO is broken.

**2. Outgassing.** A comet with an undetected coma produces a genuine radial
acceleration. Cometary designation, a fitted `DT`, a cometary `g(r)` (signalled
by ALN/NM/NN/NK/R0 in `orbit.model_pars`), or a reported coma all remove the
object. **A cometary `g(r)` also invalidates the `A1 → β` conversion itself** —
under that law `A1` is not a radiation-pressure coefficient — so `r_statistic`
refuses to convert rather than returning a wrong number.

**3. Short-arc fit artefacts.** Survivors face a tighter gate than the screen
(`condition_code ≤ 2`, `data_arc ≥ 180 d`).

**4. Yarkovsky leakage.** `A1` and `A2` are correlated in short-arc solutions, so
a real transverse Yarkovsky signal can bleed into the radial term. Tested from
the SBDB covariance matrix; `|corr(A1,A2)| ≥ 0.9` is a rejection. Where the
covariance could not be retrieved, `no_covariance` is recorded as an **untested**
flag rather than a pass.

**5. Genuinely small natural bodies.** A metre-scale rock really does have a
measurable radiation-pressure signal. `R` normalises this out by construction,
but sub-10 m survivors are labelled so nothing is over-read.

A survivor is marked `UNEXPLAINED` **only** when every route above has been
checked and none fires — and even then the untested flags travel with it.
`UNEXPLAINED` is not a detection claim.

---

## 5. Scope honesty

JPL's asteroid non-gravitational solutions are overwhelmingly **A2-only**
(Yarkovsky). The set with a fitted `A1` **and** a non-cometary classification may
be **tens of objects, not thousands**. That is not a weakness: it makes screens
1–4 a *complete, tractable census* rather than a scale play. Nothing in the code
assumes a population size, and `results/derelict/summary.json` reports the
**actual** row counts at every funnel stage.

The JPL SBDB Query API contract was **unverified at build time** (the sandbox is
egress-blocked). The acquisition therefore walks a ladder of constraint
syntaxes and falls back to an unconstrained pull filtered client-side;
`results/derelict/schema.json` records which strategy actually worked and which
fields the server accepted.

### 5.1 What the API actually does — measured, run 30203392288

The first run returned **zero rows**, and the cause was neither the sky nor the
`sb-cdata` grammar:

```
{"message":"invalid field specified: 'sigma_A1'","code":"400"}
```

`sigma_A1` is **not a valid SBDB Query field**. Because every strategy in the
ladder sent the same field list, a *single* bad column name 400'd all four, and
"our query has a typo" presented as "the entire database contains no object
with a fitted A1". Three fixes followed, and they are the real lesson of the
run:

1. **Self-healing field pruning.** The server names the offending field in its
   error body, so the fetch parses it out, drops that column and retries.
   Required fields (`A1`, `full_name`) are never dropped — if the server rejects
   `A1` there is genuinely nothing to search for.
2. **The sigmas move to the per-object records.** `A1`/`A2`/`A3`/`DT` *are*
   accepted in bulk; only their uncertainties are not. They are available from
   `sbdb.api`'s `orbit.model_pars[].sigma`, which is the authoritative source
   anyway — and the same record carries the parameter *names* that reveal a
   cometary `g(r)`, plus the covariance. So the bulk query's job is reduced to
   *finding* the A1 population; every precision number comes from the
   per-object record.
3. **Zero is never reported as a result.** The verdict now distinguishes
   `A1_FIELD_NOT_IN_SCHEMA` (the server has no such column — our query is
   wrong), `A1_COLUMN_PRESENT_BUT_ALL_NULL` (our constraint is wrong),
   `QUERY_RETURNED_NO_ROWS`, and `NO_DATA_REACHED`, and the funnel carries
   `rows_returned_by_server` alongside `input` so a reader can always tell a
   query failure from an empty screen. **SBDB certainly contains objects with a
   fitted A1 — every non-gravitational comet solution has one — so a zero here
   is a query defect until proven otherwise, never an occurrence limit.**

Also measured: the documented `?info=field` schema probe returns HTTP 400, so
field discovery cannot be relied on and the error-driven pruning above is the
mechanism that actually works.

---

## 6. Layout

```
src/seti/derelict/radiation.py   β/AMR conversions, R statistic (pure, tested)
src/seti/derelict/acquire.py     SBDB query + per-object detail (runner-only)
src/seti/derelict/screen.py      the four screens + funnel counts
src/seti/derelict/vet.py         contamination rejection
src/seti/derelict/run.py         orchestration -> results/derelict/
scripts/derelict_lit.py          id-verified novelty-check fetcher
tests/test_derelict.py           offline CI gate
.github/workflows/derelict.yml   workflow_dispatch
```

CLI: `python -m seti.cli derelict [--stage probe] [--limit N] [--max-vet N]
[--offline-input FILE] [--skip-control]`.

Outputs: `schema.json`, `nongrav.csv`, `negative_a1.csv`, `high_albedo.csv`,
`control_comets.csv`, `candidates.json`, `summary.json`, `REPORT.md`.

---

## 7. Not built, and why

**The ISO-v∞ anomaly search (S20).** Mamajek 2017 (RNAAS 1, 21,
arXiv:1710.11364) already found 1I's velocity anomalously close to the LSR *and
explained it*; gravitational focusing makes low-v∞ interstellar objects
**over**-represented, so the anomaly is the expected outcome rather than a
surprise; and N = 3 cannot support a population claim. Documented, not built.
