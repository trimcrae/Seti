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

> **Provenance caveat.** The Bialy & Loeb half of this cross-check is verified
> against the fetched text on disk. The Micheli et al. input value is **not** —
> the file that should hold it (`results/derelictlit/txt_micheli2018_*.txt`) is a
> quark-matter paper fetched under a guessed arXiv id. See §2.1. The
> *arithmetic* is unit-tested and the *output* matches an independently verified
> source, but the input number is currently pending re-fetch.

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

**Status as of 2026-07-26: the dark-comet half of the claim is now supported
verbatim from the primary sources (§2.1.1).** It was not for most of this
channel's life, and the reason is worth keeping on the record permanently,
because the same failure recurred four times and the fourth was caused by the
fix for the first.

It began here:

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

**And a third and fourth instance, found 2026-07-26 by reading the fetched text
rather than the summary.** `results/derelictlit/summary.json` reported
`n_id_mismatch: 0`, yet two of its files are the wrong paper:

| entry | requested id | what actually arrived |
|---|---|---|
| `micheli2018_oumuamua_nongrav` | 1811.05519 | *"Acoplanarity of Lepton Pair to Probe the Electromagnetic Property of Quark Matter"* (nucl-th) |
| `seligman_2021_darkcomet_precursor` | 2104.10184 | *"Finite-size evaporating droplets in weakly compressible homogeneous shear turbulence"* (physics.flu-dyn) |

Both were recorded `id_ok: true, title_ok: true`. The cause is a one-line
defect in the verifier itself: `ok_title = (fragment in title) if fragment else
True`, and both entries carried an **empty** `expect_title`. So the check that
existed to catch exactly this failure returned *pass* precisely when it had
checked nothing. **"The title was never checked" and "the title checked out"
were sharing a value.**

Fixed: an absent fragment now yields `title_ok: null` plus an explicit
`title_unverified` flag, counted separately in the summary as
`n_title_unverified` and printed by the workflow, so an unchecked paper can
never hide inside a zero-mismatch report. Micheli et al. 2018 moves to the
title-resolution path (no id guessed); the Seligman "precursor" entry is
**removed** rather than guessed a fifth time.

The practical consequence for this document: **§1.3's calibration anchor
(`a = 4.92e-6 m/s²` at 1 au, Micheli et al. 2018, Nature 559, 223) is not
currently supported by anything on disk** — the file that should hold it is a
quark-matter paper. The conversion chain it validates is independently
self-consistent and unit-tested, and it reproduces Bialy & Loeb's ~0.1 g/cm²
(which *is* on disk, verified), but the anchor value itself is
**pending re-fetch**.

**The pattern, stated once: four wrong papers, every one of them fetched
successfully, and the first three recorded as verified.** An identifier recalled
from memory is not evidence, and a verification step that can pass without
checking anything is worse than none — it converts an unknown into a false
positive. Both defences (title-resolution instead of id-guessing; an explicit
unverified state) exist because assertion failed here repeatedly.

With that history in view, the components of the verdict stand as:

| Component | Status |
|---|---|
| Seligman et al. selected on non-radial A2/A3 | **SUPPORTED — verbatim** (see §2.1.1) |
| They never compute AMR/β, never test an A1-only model | **SUPPORTED — verbatim** (see §2.1.1) |
| Micheli et al. 2018's published A1 (§1.3 anchor) | **NOT ON DISK** — the fetched file is the wrong paper (see above); pending re-fetch |
| No catalogue-scale AMR/β/lightsail search of solar system bodies exists | **SUPPORTED** — a sweep of ~35 literature directories finds none; the targeted arXiv query `lightsail AND search AND solar system` returns **0 results**; `high area-to-mass` returns 0 tree-wide |
| Bialy & Loeb applied the inference to 1I alone | **SUPPORTED (weakly)** — cited on disk only as a single-object claim |

### 2.1.1 The complement claim, now read from the source

The two decisive components above were `UNVERIFIED` until 2026-07-26, when the
id-verified fetches were finally *read* rather than counted. Both are now
supported by quotation from `results/derelictlit/txt_seligman2023_dark_comets.txt`
(arXiv:2212.08115v2, title check passed) and
`txt_seligman2024_two_populations.txt` (PNAS 121(51), resolved by title search).

**They selected on the non-radial terms, and said so:**

> "Solar radiation pressure is a radial acceleration and is therefore **less
> effective at producing significant orbital deviations**. As such, it has been
> measured only on a handful of small asteroids."

> "Based on this analysis, we conclude that there are **significant out-of-plane
> accelerations for all of these objects** … solar radiation pressure is mostly
> radial … **these accelerations are inconsistent with radiation effects**."

**And they explicitly discarded the radial term:**

> "…the magnitudes of the nongravitational accelerations are inconsistent with
> being caused by the Yarkovsky effect or radiation pressure. Therefore, for the
> remainder of this paper, we hypothesize that the non-radial nongravitational
> accelerations are caused by outgassing. We calculate the implied production
> rates of H₂O **using only the dominant, nonradial accelerations**."

The 2023 Table 1 caption is equally explicit: "Statistically robust results
(σ ≥ 3) are found for the **A3** component of all newly reported objects in this
paper. For the A2 component, only 2016 NJ33, 2003 RM and 2006 RH120 have robust
detections." A1 is the term that *fails* significance across their sample —
which is precisely why the A1-only complement was left on the table.

**Neither paper converts A1 into anything.** Verbatim term counts across both
full texts:

| term | 2023 | 2024 |
|---|---|---|
| `area-to-mass` / `area to mass` / `A/m` | 0 | 0 |
| `beta` | 0 | 0 |
| `lightsail` / `light sail` | 0 | 0 |
| `artificial` | 0 | 0 |
| `radiation pressure` | 7 | 5 |

Every occurrence of "radiation pressure" is an argument for why it *cannot*
explain their objects, or a citation to a per-object SRP detection (Micheli,
Tholen & Elliott 2012 on 2009 BD; 2014; Vokrouhlický & Milani 2000). **Nobody in
either paper computes β, an area-to-mass ratio, or a size-normalised statistic
from A1.** That is the gap this channel occupies, and it is now evidenced rather
than asserted.

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

### Screen 0 — is the A1 census actually the whole A1 population?

**This runs before anything is believed.** The constrained query
`sb-cdata={"AND":["A1|DF"]}` returned **22** asteroid rows in run 30204805880.
22 is small enough that two completely different statements are
indistinguishable from the result alone:

* the constraint is subtly wrong and is silently omitting objects, or
* JPL really has only fitted an `A1` to 22 asteroids.

Only the second makes §5's "complete, tractable census" claim true, so it is
proved rather than assumed. `completeness_probe()` pulls **every** small body of
each kind with **no constraint at all** and a minimal column set
(`spkid,full_name,pdes,kind,class,A1,A2,A3` — ~1.4M asteroid rows, a few hundred
MB), counts non-null `A1` client-side, and compares **designation sets** in both
directions.

Sets, not counts: two different sets can have the same size. And designations are
normalised before comparison (`(2005 VL1)` vs `2005 VL1` vs `523599 (2003 RM)`
are the same object), because a formatting difference read as a missing object
would manufacture a false `CONSTRAINT_INCOMPLETE`.

| verdict | meaning |
|---|---|
| `CONSTRAINT_COMPLETE` | the sets are identical: the census is provably complete |
| `CONSTRAINT_INCOMPLETE` | the unconstrained pull found an `A1` the constraint missed → **switch the primary path to the unconstrained pull** |
| `PROBE_INCONSISTENT` | nothing missing, but the probe did not see something the constraint returned → the *probe* is unreliable (truncation/paging); completeness stays unproven |
| `PROBE_FAILED` | the unconstrained pull never returned → completeness is **UNTESTED**, and this is never reported as agreement |

Two further guards. The server's own `count` is compared against the number of
rows actually parsed, and a mismatch raises "the pull may be truncated, so a
'complete' verdict would be unsafe". And if the single monolithic pull fails
(memory, timeout) the probe falls back to chunking by `sb-class` — a *fallback*
axis, not an assertion, so the class list is not claimed to be exhaustive and
the chunked union's shortfall against the server count is reported.

Output: `results/derelict/completeness.json`, asteroids and comets separately,
with the differing designations listed so a disagreement is actionable.

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

**It worked on the first real run.** Run 30204137011 returned exactly one
significant negative-A1 object, and it is about as informative as it could be:

| | 4179 Toutatis (1989 AC) |
|---|---|
| class | APO |
| A1 | −3.149×10⁻¹³ au/day² |
| σ(A1) | 9.279×10⁻¹⁴ |
| A1/σ | **−3.39 (sunward)** |
| data arc | 33,698 d (92 yr) |
| n_obs | 7,141 |
| condition code | 0 (best) |
| diameter | 5.4 km |

Toutatis is one of the best-observed asteroids in the sky, and it shows a 3.4σ
*sunward* radial acceleration — which radiation pressure cannot produce. So this
is a clean measurement of the systematic floor of JPL's non-gravitational fits,
and it converts directly into the units this channel cares about:

```
|β|  = 1.06×10⁻⁹
|AMR| = 1.39×10⁻⁶ m²/kg
AMR_natural(5.4 km, ρ=1000) = 2.78×10⁻⁷ m²/kg
⇒ |R| = 5.0
```

**The floor sits at |R| ≈ 5, and the flag threshold is R = 10.** The thresholds
are therefore set a factor of ~2 above the measured noise, not inside it — and
that statement now rests on data rather than on assertion. Any future candidate
with R of order 10 must be treated as marginal against this floor.

#### Reported as a rate, on both populations

One object is not a measurement of a rate. `negative_a1_census()` therefore
reports `n_negative / n_a1_fitted` **with its denominator stated**, per
population, and does so on the **comet control as well as the asteroids** —
because the control is an order of magnitude larger (272 vs 22 objects in run
30204805880, with 25 significant negative-A1 comets), so the floor is far better
measured there.

Every negative-A1 object carries its own `|R|`. `r_statistic()` deliberately
*refuses* a negative `A1` — radiation pressure cannot push sunward, so `R` is
meaningless as a physical quantity — but the **magnitude** is exactly what
screen 3 exists to measure: it is how large a spurious area-to-mass the orbit
fit manufactured, in the same units the flag threshold is written in.
`results/derelict/negative_a1.csv` carries both populations with a `population`
label, `abs_R`, and the diameter and diameter-source used to form it.

### Screen 4 — impossible albedo, run independently of A1

Geometric albedo > 0.7 is impossible for natural regolith and trivial for
aluminised film. (Fresh ice and E-type enstatite reach ~0.6.)

**This is a technosignature in its own right and has nothing to do with whether
an orbit solution happened to include a non-gravitational term.** Restricting it
to the 22-object A1 sample — as the first implementation did, returning 0 — was
answering a far narrower question than the one worth asking. It now runs as an
**independent, catalogue-wide** SBDB query over all asteroids (and comets) with
a defined albedo, walking a ladder of constraint syntaxes
(`albedo|GT|0.7` → `albedo|RG|0.7|1.5` → `albedo|DF` → unconstrained) because
the numeric-comparison grammar was never verified from the runner. **Every rung
is filtered client-side regardless**, so a server-side operator that does not
mean what we assumed cannot leak rows into the survivors.

Two honesty requirements:

* **A single-source albedo is a fit artefact until something independent
  agrees.** Survivors are cross-checked against IRSA (NEOWISE diameter+albedo).
  The table and its columns are *discovered* through `TAP_SCHEMA`, not
  hard-coded, because a renamed table looks exactly like an unreachable archive.
  Failure modes each get their own status (`IRSA_NOT_REACHED`,
  `NO_ALBEDO_TABLE_FOUND`, `NO_ALBEDO_COLUMN_FOUND`) and **none of them is ever
  reported as "checked and agreed"** — an unconfirmed row stays unconfirmed.
* Where an albedo uncertainty exists the *excess over 0.7* must be significant,
  not just the point estimate. SBDB appears not to expose one (the speculative
  `albedo_sigma` request is pruned by the self-healing field logic), so that
  test is recorded as **UNTESTED** rather than silently passed, and the IRSA
  value plus its error is what actually supplies it.

The **expected dominant population is albedo fit artefacts on short arcs**, so
`condition_code`, `data_arc`, `n_obs_used` and `rms` travel with every row in
`results/derelict/high_albedo.csv` — so that can be checked rather than assumed.

### Screen 5 — the dark-comet named-target census

The Seligman et al. sample is the one place the literature has already assembled
coma-free non-gravitationally accelerating objects, and it selected on
**non-radial** acceleration — the complement of what this channel wants. The
questions worth asking of those 14 objects are therefore: **which of them have a
JPL-fitted `A1` at all, which are A1-*only*, and what `R` do they sit at?**

Each designation is resolved individually through `sbdb.api`
(`cov=mat&full-prec=1&phys-par=1`), the fitted `A1`/`A2`/`A3` **and their
sigmas** are read from `orbit.model_pars` (the bulk query rejects `sigma_A1`
outright, so this is the only source), and the rows go through the *same*
`run_screens()` the main funnel uses. An **unresolvable designation is reported**
in `unresolved`, never silently dropped.

The designations live in `config/derelict.yaml` under `dark_comets.targets`,
**each tagged with the paper and table it was read from**, and they were read
verbatim out of the fetched full text committed in `results/derelictlit/` — not
recalled. §2.1 explains why that distinction is not pedantry. One consequence is
already visible: the Seligman 2023 *abstract* prints "2016 RH120", which its own
Table 1, figure captions and notes contradict with `2006 RH120`; the config
carries `2006 RH120` and a test asserts the abstract's typo has not propagated.

Eight of the 14 are already in `screened.csv` (1998 KY26, 2005 VL1, 2006 RH120,
2010 RF12, 2013 BA74, 2013 XY20, 2016 GW221, 2016 NJ33) and their JPL `A1`
values match Seligman et al. 2024 Table 2 to the printed digits — 2005 VL1 at
`−8.30×10⁻¹⁰ ± 7.59×10⁻¹⁰` in both. The other six (2003 RM, 2010 VL65, 2001 ME1,
2005 UY6, 1998 FR11, 2012 UR158) were **not** in the constrained pull, which
predicts they have no JPL-fitted `A1` at all; the census is what turns that
prediction into a measurement. Output: `results/derelict/dark_comets.csv`.

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
acceleration. A cometary designation/classification, a reported coma, or a
**fitted time-delay `DT`** all remove the object — a lagged response is
something radiation pressure cannot produce.

**What is *not* outgassing evidence — and why this nearly killed the channel.**
JPL fits `A1`/`A2`/`A3` using the Marsden, Sekanina & Yeomans (1973) `g(r)` as
its **default parameterisation for every object it fits non-gravs to, including
every dark comet**. An earlier version of this channel treated the mere presence
of the Marsden shape parameters (`ALN`/`NM`/`NN`/`NK`/`R0` in
`orbit.model_pars`) as "this object outgasses" and refused to convert. Run
30204137011 showed what that costs: **20 of 22 objects discarded** — 91% of the
sample, and precisely the population the channel was built to examine.

It is wrong, for a reason worth stating exactly:

```
Marsden g(1 au) = 1.0000000024
```

The standard parameters are **normalised so that `g(1 au) = 1`** to within
2×10⁻⁷ — exactly like the inverse-square law. So `A1` *is* the radial
acceleration at 1 au under **both** laws, and the `A1 → β → AMR` conversion is
valid either way. `g_at_1au()` applies that normalisation rather than assuming
it, so a genuinely non-standard fitted shape would still be handled correctly.

What the two laws *do* disagree about is the radial **dependence**:

| r (au) | Marsden `g(r)` | inverse-square `1/r²` |
|---|---|---|
| 0.5 | 4.54 | 4.00 |
| 1.0 | 1.000 | 1.000 |
| 2.0 | 0.109 | 0.250 |
| 5.0 | ~10⁻⁶ | 0.040 |

That divergence is large and it is the genuinely decisive test — but **no
catalogue column can settle it.** Separating the two requires refitting the
archival astrometry under each law and comparing the evidence (the "step 5"
below). It is therefore the mandatory **follow-up for a survivor**, not a
selection cut, and the channel records `nongrav_law` and `g_1au` per object as
descriptive metadata instead.

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

### 5.2 The fix worked — measured, run 30204805880 (2026-07-26 13:48 UTC, main @ 046089d)

**The constraint syntax was never the problem, and it is now confirmed
correct.** With the self-healing field pruning in place, the *first* strategy on
the ladder succeeded:

| | asteroids (`sb-kind=a`) | comets (`sb-kind=c`, the control) |
|---|---|---|
| status | `OK` | `OK` |
| strategy | `cdata_A1_defined` | `cdata_A1_defined` |
| rows | **22** | **272** |

`cdata_A1_defined` is `sb-cdata={"AND":["A1|DF"]}`, so **that grammar is
verified against the live API**, not assumed. The server rejected exactly
`sigma_A1`, `sigma_A2`, `sigma_A3`, `sigma_DT` — one at a time, four retries —
and accepted every other requested field including `A1`/`A2`/`A3`/`DT`. All 22
sigmas were then recovered per object from `orbit.model_pars`, and 267 of 272
comets enriched successfully.

**The control fires.** 272 comets with a fitted `A1`, 210 of them significant,
and every one correctly excluded from screen 1 by its cometary classification
(`gate_fail_coma = 272`). A zero on the asteroid side would therefore have been
a real measurement — but it is not zero, it is 22.

The funnel that run measured:

| stage | n |
|---|---|
| input | 22 |
| a1_fitted | 22 |
| a1_positive | 19 |
| a1_significant | 11 |
| nonradial_constrained | 10 |
| screen1_a1_only | 4 |
| screen1_a1_only_strict | 0 |
| screen3_negative_a1 | 1 (asteroids) / 25 (comets) |
| screen4_albedo | 0 (A1 sample only — see screen 4 above) |
| unexplained_after_vetting | 0 |

Verdict `ALL_SURVIVORS_EXPLAINED`: three of the four survivors are
`ARTIFICIAL_HUMAN_SUSPECT` (Earth-like orbits) and one is a
`SHORT_ARC_ARTEFACT`. `screened.csv` contains 'Oumuamua (R ≈ 1.08×10⁵, AMR 1.23
m²/kg — the positive control, recovered), 457175 (2008 GO98) at R ≈ 2.9×10⁵ but
failing the A2-zero gate at 11σ, 4179 Toutatis (the negative-A1 floor), and
(2005 VL1) (the Loeb & Cloete object).

**What run 30204805880 did *not* establish**, and what screens 0 and 4–5 above
exist to settle: whether 22 is the *whole* A1 population or an artefact of the
constraint; what the dark comets outside those 22 look like; and whether
anything in the catalogue has an impossible albedo irrespective of `A1`.

### 5.3 The acquisition guard — an empty result can never look like an unreached archive

The lesson of 30203392288 is now enforced structurally rather than by care.
Every query the channel issues is recorded in `results/derelict/queries.json`
(and inlined into `summary.json` under `queries`) as

```json
{"label": …, "url": <verbatim, unredacted>, "http_status": 400,
 "status": "QUERY_FAILED", "n_rows": 0, "attempt": 1, "error": …}
```

with two statuses that **must never merge**:

* `QUERY_FAILED` — the server never answered with a parseable table. `http_status`
  carries the real HTTP code when one was received and is `null` when the request
  never got that far (DNS/TLS/proxy/timeout), because those are different failures.
* `QUERY_RETURNED_ZERO_ROWS` — the server answered, and the answer was empty.
  That is a *measurement*.

Per-object `sbdb.api` fetches are logged the same way, so an unresolvable
designation ("no such object", a real answer) is distinguishable from an
unreachable archive. Nothing is redacted; a reader can paste any recorded URL
into a browser and reproduce the response. If the inline copy in `summary.json`
is capped, the cap is recorded explicitly in `queries_truncated` and the
complete log is still on disk.

---

## 6. Layout

```
src/seti/derelict/radiation.py   β/AMR conversions, R statistic (pure, tested)
src/seti/derelict/acquire.py     SBDB query + per-object detail + query log (runner-only)
src/seti/derelict/screen.py      the four screens + funnel counts
src/seti/derelict/census.py      completeness probe, dark-comet census,
                                 negative-A1 rate, catalogue-wide albedo + IRSA
src/seti/derelict/vet.py         contamination rejection
src/seti/derelict/run.py         orchestration -> results/derelict/
scripts/derelict_lit.py          id-verified novelty-check fetcher
tests/test_derelict.py           offline CI gate
.github/workflows/derelict.yml   workflow_dispatch
```

CLI: `python -m seti.cli derelict [--stage all|probe|search|completeness|dark_comets|high_albedo]
[--limit N] [--completeness-limit N] [--max-vet N] [--offline-input FILE]
[--skip-control] [--skip-completeness] [--skip-dark-comets] [--skip-high-albedo]`.

Stages are individually dispatchable so a heavy pull can be re-run without
re-running the funnel. `all` runs everything; `search` runs the funnel plus the
per-object census stages but skips the ~1.4M-row completeness pull. A named
stage writes `summary_<stage>.json` and **never overwrites `summary.json`** — a
partial run must not be able to replace a full one and make the channel look
like it regressed.

Outputs: `schema.json`, `screened.csv` (**the parent sample** — every object
pulled, with every screen column; without it a zero at any gate is unauditable
and there is no denominator for a rate), `nongrav.csv` (survivors),
`completeness.json` (screen 0), `dark_comets.csv` (screen 5),
`negative_a1.csv` (screen 3, both populations, with `abs_R`),
`high_albedo.csv` (screen 4, catalogue-wide) and `high_albedo_a1_sample.csv`
(the narrower A1-sample view, kept separate so the two are never conflated),
`control_comets.csv`, `candidates.json`, `queries.json` (every query verbatim),
`summary.json`, `REPORT.md`.

The funnel also carries per-gate failure counts
(`gate_fail_a1_not_significant`, `_a2_nonzero`, `_a3_nonzero`,
`_orbit_quality`, `_coma`, `_outgassing`) so that a zero at screen 1 explains
itself. They overlap — an object can fail several gates — so they do not sum to
the input.

### 6.1 The decisive follow-up (step 5), and why it is not a screen

The one test that genuinely separates radiation pressure from outgassing is a
**model-selection refit of the archival MPC astrometry** under a pure-SRP `1/r²`
law versus the Marsden `g(r)`, comparing the evidence per object. The table in
§4 shows why it has real power: the two laws differ by ~4 orders of magnitude in
predicted acceleration at 5 au, so an object observed over a range of
heliocentric distances discriminates them strongly.

It is **not built here.** MPC publishes no `A1`/`A2`/`A3` — MPCORB carries only
the six osculating elements, `H`, `G` and `U` — so it is the *astrometry* source
for such a refit, not a non-grav catalogue, and the refit needs an orbit
determination code this repository does not have. It is recorded as the correct
next action for any survivor rather than attempted badly.

---

## 7. Not built, and why

**The ISO-v∞ anomaly search (S20).** Mamajek 2017 (RNAAS 1, 21,
arXiv:1710.11364) already found 1I's velocity anomalously close to the LSR *and
explained it*; gravitational focusing makes low-v∞ interstellar objects
**over**-represented, so the anomaly is the expected outcome rather than a
surprise; and N = 3 cannot support a population claim. Documented, not built.
