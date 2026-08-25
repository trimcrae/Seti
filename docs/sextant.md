# SEXTANT — the discard pile, at milliarcsecond precision

*A sextant measures the angle between where a thing is and where it ought to be.*

**Channel status:** residual computation and screen built and offline-tested (45
tests); acquisition is a parallel workstream; nothing has been run on live data.
Rubin has been off sky since the night of 13/14 July 2026 (`docs/rubin-outage.md`,
verdict `SKY_STOPPED`), and this is not a stopgap for it — see §1.

---

## 1. The question, and how it changed

LOOM's observable is the ephemeris residual of a known minor planet, decomposed
so the along-track component is separated from everything with no directional
preference. Rubin serves that residual pre-computed, in arcsec. Gaia serves the
*ingredients* at milliarcsecond precision, for **46,264,083** observations of
**156,823** objects (`gaiafpr.sso_observation`) plus **23,336,467** of **158,152**
(`gaiadr3.sso_observation`).

The obvious thing to do with that is rank objects by anomalous post-fit residual.
**That is occupied, twice over, on this exact table.**

| work | what it did | why it closes the obvious door |
|---|---|---|
| Liberato, Tanga, Mary, Lallemand, Liu, Carry, Desmars, Hestroffer, Minker & Siakas, *Follow the wobble: Statistical methods to detect astrometric binary asteroids in Gaia FPR*, arXiv:2605.22702 (2026) | along-scan projection of post-fit residuals, FPR-consistent noise model, Monte-Carlo noise-only control, trend detection, period search, improved statistical selection → **343 binary candidates / 410 windows** | this *is* the population-scale anomalous-residual screen, with our null-control design, on `gaiafpr.sso_observation` |
| Dziadura, Bartczak & Oszkiewicz, *Assessing the detection of the Yarkovsky effect using Gaia DR3 and FPR catalogues*, A&A 693, A31 (arXiv:2411.09750) | OrbFit fits of six elements **plus a non-gravitational A2** to 446 NEAs and 54,094 inner-belt / Mars-crossers on Gaia FPR | the amplitude question was asked at our scale and came back **null for the main belt** |
| Liberato, Tanga et al., *Binary asteroid candidates in Gaia DR3 astrometry*, A&A 688, A50 (arXiv:2406.07195) — VizieR `J/A+A/688/A50` | the DR3 predecessor | the contamination catalogue this channel subtracts |
| Gaia-CRF3 vs planetary ephemerides, 1001 asteroids | the O−C construction itself, executed and published | ~96% of along-scan residuals within ±5 mas, ~52% sub-mas; across-scan strongly non-Gaussian |

So the residual machinery is not the novelty. **It is the substrate.** Two things
remain unoccupied, and the screen is aimed at the first with the second as its
second stage.

### The chosen question: the discard pile

Every search above operates **post-fit**. They screen what the astrometric
solution successfully modelled. An object whose astrometry is *systematically
un-fittable* — one the pipeline keeps throwing away — is invisible to all of
them, by construction.

So the observable is the **rejection pattern**: `is_rejected`,
`astrometric_outcome_ccd`, `astrometric_outcome_transit`, as a per-object rate
against the **attempts denominator**. Published context for the scale: the
outlier fraction is ~0.58% in DR3 and ~1% in DR2 — small, well measured, and
structured by observing conditions, which is what makes an excess meaningful
rather than merely large.

**Why this and not the alternative.** The other candidate was to make
law-discrimination the screen itself. It was rejected as the *primary* because it
conditions on an amplitude that Dziadura et al. have already measured to be
absent in the main belt: a model-selection test on a signal that is not there
selects between three ways of fitting noise. The discard pile does not inherit
that null — it is a different *sample*, not a different statistic on the same
sample. Law discrimination is kept, and it is where the second novelty lives, but
it runs on what the rejection screen surfaces (§4).

**This is a measurement before it is a search.** Nobody has published the
per-object distribution of Gaia SSO astrometric rejection rate. The first output
is that distribution and its dependence on observing conditions; the search is
what is left after the conditions are divided out.

### Coverage limit, stated once and meant

Gaia SSO astrometry is archival — the mission's 2014–2020 window. It tests a
**static** population, which is exactly the question LOOM asks, and it cannot
detect a new event or replace a nightly cadence. It replaces the population
question, not the alert.

---

## 2. The residual, and its error model

For observation *i* at barycentric dynamical time *t_i*, with Gaia at barycentric
position **R**_i and velocity **V**_i — both archive columns, so **no model of
Gaia's orbit appears anywhere in this channel**:

1. solve the light time τ_i for `|r_target(t_i − τ_i) − R_i| = c τ_i`;
2. form the geometric direction `u = (r_target(t_i − τ_i) − R_i)/|·|`;
3. deflect it in the Sun's field (finite-distance ERFA `ld` form);
4. aberrate it with **V**_i (special-relativistic; the second-order term is ~2 mas
   and therefore not optional);
5. subtract from the observed direction in the local tangent plane at the
   prediction, giving `(d_east, d_north)` in mas;
6. rotate into **(AL, AC)** with `position_angle_scan`.

Magnitudes, because they decide what may be dropped:

| term | size | dropped? |
|---|---|---|
| light time | ~10 arcmin of apparent motion | no |
| stellar aberration | 20.5 arcsec (1st order), ~2 mas (2nd) | no, and not the classical form |
| solar light deflection | 4.07 mas at 90° elongation, ~9.8 mas at Gaia's 45° solar aspect angle | no |
| TDB − TT periodic | ≤1.7 ms → 1.4×10⁻⁵ arcsec | computed anyway, and negligible |

### The along-scan axis *is* the error model

Gaia's precision is strongly anisotropic. AL is the good axis; AC is roughly an
order of magnitude worse and, on the published Gaia-CRF3 minor-planet
comparison, strongly non-Gaussian. So:

* the covariance is built from the archive's **separate random and systematic**
  terms and **their correlations** — `radec_covariance` assembles the full 2×2 in
  (East, North) and `project_covariance` gives the AL and AC variances;
* `ra_error` is already an error in `ra·cos(dec)`, so **no `cos(dec)` factor is
  applied**. Applying one would be a smooth `sec(dec)` inflation across an
  ecliptic-confined sample — i.e. it would look like sky-coherent structure;
* the default fit uses **AL only**, one scalar equation per observation, which is
  what Gaia's own astrometric solution does and for the same reason. AC is
  carried and tested separately (`scan_axis_partition`), never pooled;
* `epoch_err` enters the AL budget as `rate_AL × σ_t`, with the geometry applied
  — a time uncertainty *is* a position uncertainty along the track.

**The systematic term is not added in quadrature and forgotten.** A systematic
shared inside a transit (nine CCD crossings behind one attitude solution and one
calibration) does not average down across those nine. `_gls_normal_equations`
inverts a block covariance `V_b = diag(σ_rand²) + σ̄_sys² 11ᵀ` by Woodbury —
exact, and free. Every fit is run under **two** correlation models: the nominal
per-transit one and a pessimistic per-180-day one in which a slowly drifting
calibration projects onto the linear and quadratic terms. **The larger of the two
uncertainties is the one quoted.**

### The scan-angle convention is verified, not assumed

Which way `position_angle_scan` is measured is a schema fact the sandbox could
not read. `verify_scan_convention` settles it with **no ephemeris at all**: the
archive's own covariance is anisotropic with its *minor* axis along-scan, so the
angle between the covariance minor eigenvector and the PA-implied AL direction
decides the handedness. Both readings are scored and the margin is reported.
Getting this wrong does not flip a sign — it *mixes AL into AC* — so it is not a
guess. If neither reading matches, the verdict is
`NEITHER_CONVENTION_MATCHES` and nothing downstream is believed.

---

## 3. Independence, and immunity by construction

**Gaia's own SSO orbit solutions are fitted to precisely these observations.** A
residual against them is minimised by construction and measures the fit, not the
sky. That is `CIRCULAR`, and it is refused in every mode with no override.

Worse, and less obvious: **every current JPL solution for a numbered object
contains Gaia astrometry too.** Gaia DR2 (2018) and DR3 (2022) SSO astrometry
were delivered to the MPC and are used, at high weight, in modern fits. So the
"independent orbit" LOOM pulls is, here, `PARTIAL_SELF_FIT` — not independent.

Two mechanisms handle this, and the second is the load-bearing one.

**(a) The API cannot be used without stating provenance.** `OrbitSource` is a
required, frozen argument of every function that produces a residual. It has no
default and no name-guessing constructor. `gaia_sso_astrometry_in_fit=None`
raises `UnknownProvenanceError` rather than defaulting; `PARTIAL_SELF_FIT` raises
unless the caller passes `allow_partial=True` in the call, which is then stamped
into the output record; an orbit whose fit **itself carried A1/A2/A3** *and* saw
the Gaia data is refused outright, because the signal has been fitted out rather
than merely the elements; and a two-body dynamical model is refused, because
planetary perturbations reach arcseconds over a six-year arc.

**(b) The estimator is invariant to what any orbit fit could have done.** An
orbit fit has exactly six free parameters. Whatever data it used and whatever
weight it gave them, the only thing it could have removed from these residuals is
a vector in the span of the six partial derivatives of predicted sky position
with respect to the target's state. Those six columns are computed at ingest
(`state_partial_basis`, central differences of a two-body propagation) and
carried as **nuisance regressors** in every fit. Marginalising them removes
exactly the subspace a fit can move in.

The cost is sensitivity, and it is *reported*, not taken: `absorbed_fraction` is
the fraction of each model column that lies in that span. On the synthetic tests
it runs 0.95–0.99 — a quadratic over a six-year arc is largely mimicked by an
element error — and the estimator still recovers an injected A2 to better than
1%. Above `max_absorbed_fraction = 0.995` the object is vetoed: at that point an
orbit fit could have removed essentially all of it.

**The gold path remains chronological independence.** An MPCORB snapshot dated
before 2014-07 cannot contain Gaia data as a matter of arithmetic. It is
registered as `mpcorb_pre_gaia_snapshot` and classified `INDEPENDENT`; the price
is a larger ephemeris uncertainty, which shows up as a larger
`absorbed_fraction` rather than as a surprise.

---

## 4. Which law, not how much force

This is the second unoccupied axis, and it is the one a station-keeping object
cannot imitate.

Both published treatments **assume a functional form and fit its amplitude** —
a Yarkovsky A2, or a binary photocentre wobble. Neither asks which form the
residual prefers. `model_comparison` fits six models to the same series, all
marginalised over the same six-dimensional orbit-error subspace and all under the
same block-correlated error model, so the comparison is between *shapes*:

| family | model | along-scan shape |
|---|---|---|
| **force** (acts on the orbit) | `radiation` | double time-integral of `(1 au/r)²` |
| | `sublimation` | double time-integral of JPL's water-ice `g(r)` (knee at 2.8 au) |
| | `constant` | double time-integral of 1 — what nothing natural does |
| **geometry** (acts on the measurement) | `illumination` | sunward axis on the sky × `sin(phase)` |
| | `deflection` | sunward axis × `cot(elongation/2)` |
| | `timing` | the along-scan sky rate |

A force accumulates quadratically and is indifferent to where the Sun is; an
illumination or deflection residue tracks the sunward direction on the sky and
reverses with it; a timing error is proportional to sky rate. **The families are
separated by shape, not by size.**

Two questions are answered separately because they have different answers:

* **family** — force vs geometry. Usually decisive.
* **law** — which force. Frequently *not* answerable, and `law_separability`
  measures that directly: the correlation between the design columns after
  whitening and after the orbit-error subspace is projected out. Two columns
  correlating at 0.9999 are the same curve, and a χ² difference between them is
  noise however large. LOOM returned `INSUFFICIENT_R_SPAN` on every Rubin object
  because a one-month baseline samples no heliocentric range; Gaia's 2014–2020
  arc covers one to three revolutions, which is far better but still not always
  enough, and the honest outcome is `LAWS_NOT_SEPARABLE`.

### The factor of three, which LOOM does not carry

`seti.loom.residuals.drift_fit` converts a fitted quadratic to an acceleration as
`a = 2·c2` — the kinematic reading, `a t²/2`. That is not what a transverse force
does to an orbit. A transverse acceleration `a_T` raises the semimajor axis
(`da/dt = 2a_T/n`), which *lowers* the mean motion (`dn/dt = −3a_T/a`), so

```
d²S/dt² = −3 a_T(t)
```

— three times larger, and of the opposite sign: an object pushed forward **falls
behind**. Checked against the orbit-averaged secular formula LOOM itself uses for
`da/dt`: for a circular orbit the two agree exactly
(`test_variational_response_reproduces_the_factor_of_three`).

This is recorded here rather than patched into LOOM: that number feeds a
published calibration and changing it is a decision for the channel's owner.

### And the scalar formula is still not good enough

`d²S/dt² = −3a_T` is the *along-track* secular response, and it reproduces a
direct integration to within 2%. But the true displacement is a three-vector —
radial part, periodic along-track terms of order *e*, normal part — and the
residual is a projection of the whole vector onto the sky. Measured on synthetic
data with a known injected A2: the scalar basis recovers the amplitude to 2% with
no nuisance columns and is **49% low** once the six state partials are
marginalised, because the unmodelled part is absorbed asymmetrically.

So the signal basis is the **exact linearised response**, integrated per object:

```
d²(δr)/dt² = −μ/r³ δr + 3μ (r·δr) r/r⁵ + g(r) t̂(t)
```

with `δr = δṙ = 0` at the reference epoch, about the unperturbed two-body
trajectory, projected onto the sky and then onto the scan axis
(`variational_response`). `t̂` is JPL's **transverse** direction — in the orbit
plane, perpendicular to **r** — not the velocity direction, because that is how
A1/A2/A3 are defined.

The reference epoch is free: changing it adds a homogeneous solution of the same
equation, which is exactly a state perturbation, which is exactly what the six
nuisance partials span. `test_variational_basis_reference_epoch_invariance` pins
that the fitted amplitude does not move.

*Runner note.* Every object shares the same mission time window, so the
per-object RK4 loop vectorises across objects on one common grid. That is the
form to use at catalogue scale; the per-object function here is the reference
implementation and the thing the tests exercise.

---

## 5. Binaries

An unresolved binary's photocentre wobbles about the barycentre: an anomalous
astrometric residual with the right amplitude, the right timescale and the right
dataset. Three defences, in increasing order of strength.

1. **The catalogue.** `J/A+A/688/A50` (Liberato et al. 2024, arXiv:2406.07195),
   extended by arXiv:2605.22702's 343 FPR candidates. VizieR is unreachable from
   the sandbox, so `BinaryCatalogue` is an **input** — which is not a
   convenience, it is what lets the offline tests exercise the rejection path
   instead of leaving it a branch that has never run. Matched on `number_mp`
   first, then on `denomination`.
2. **The amplitude bound** (`photocentre_bound`). A photocentre offset cannot
   exceed the primary-to-barycentre distance — at most a few hundred km for a
   small-body binary; the constant is set at an indefensibly generous 1000 km.
   **It is reported, not a veto**, and the reason is honest: a genuine
   Yarkovsky-scale drift implies ~100 km of displacement over a six-year arc,
   comfortably inside what a wobble could supply, so vetoing on it would veto
   every real signal. What it says *positively* is that a large displacement
   cannot be a wobble at all.
3. **The timescale**, which is the one that works at ordinary amplitudes. A
   wobble runs on the *satellite's* period — hours to days — which no
   heliocentric basis function reaches, so it appears as excess scatter about the
   fitted model rather than as a drift. `excess_scatter` = √(reduced χ²) of the
   winning fit; above `max_excess_scatter = 2.0` the object is vetoed as a
   photocentre-wobble candidate.

---

## 6. The screen

### The measurement first

`sample_rejection_summary` reports the sample-wide rate, the per-object rate
quantiles, and the number of objects with zero rejections — and compares the rate
with the published DR2/DR3 outlier fractions. Landing far outside that band
returns `RATE_INCONSISTENT_WITH_PUBLISHED_OUTLIER_FRACTION` and **stops the
run**: if the flags are not being read as the mission means them, every number
after that is a misinterpretation with error bars.

### The null

Not parametric. Per-object expectation comes from the survey's own statistics
within strata of the covariates that drive the rate:

| covariate | the mundane reading it controls |
|---|---|
| `apparent_motion_mas_per_day` | a fast mover smears along scan — the reading that most resembles the signal |
| `abs_galactic_latitude_deg` | crowding: more neighbours, more confusion, more discards |
| `magnitude` | saturation at the bright end, low signal at the faint end |
| `median_phase_deg` | a systematically offset photocentre at extreme phase |
| `n_attempts` | an object with three attempts has no measurable rate |

Three properties of `_excess_z` are load-bearing:

* **Quasi-binomial dispersion, estimated per stratum, robustly.** Rejections
  cluster — inside a transit, inside a field, inside a scan — so the binomial
  variance is an underestimate and a naive *z* would flag a large fraction of the
  catalogue. The dispersion is estimated from the stratum's own MAD, so the
  objects being searched for do not inflate the yardstick meant to find them.
  This is the same error LOOM made once with `scatter/√n`, in a different
  costume.
* **The bin count adapts to the sample.** Five covariates at four bins each is
  1024 cells; on 500 objects that is half an object per cell, every stratum falls
  below the calibration floor, and the screen returns NaN for every object *while
  looking like it ran*. That is exactly what the first version did. The
  per-covariate bin count is now the largest that keeps average occupancy at
  `min_per_stratum`, capped at `n_bins`. On the real catalogue the cap binds.
* **The trials factor is already inside the p-value.** The maximum excess is
  calibrated against matched random subsets of the same screened sample
  (`matched_draws`), so `p_max_matched` carries the >100,000-object trials factor
  rather than acquiring it later.

The excess is then recomputed with **each covariate dropped in turn**. The
primary *z* already controls for all of them, so this is not the control — it is
the robustness check on the control: an excess present only under one particular
stratification is a property of that stratification. `covariate_survival` is the
worst-case fraction retained and `binding_covariate` names the culprit. Survival
above 1 is common and benign; it is collapse that disqualifies.

### The tier ladder

| tier | meaning |
|---|---|
| `untestable` | fewer than `min_attempts` attempts or `min_transits` transits, or a stratum too sparse to calibrate an excess. **Never folded into `ordinary`** |
| `ordinary` | discard rate consistent with the matched expectation |
| `watch` | an excess, with a named reason it may not be the object's: not robust to the null, low galactic latitude, a preferred geometric explanation, a catalogued binary, excess short-timescale scatter, or an absorbed fraction above the ceiling |
| `interest` | an excess that survives every covariate control, with no veto |
| `candidate` | `interest` **and** the surviving astrometry prefers a **force law** over every geometric artefact, at S/N ≥ 5 on the pessimistic error model, absorbed fraction below the ceiling, not a catalogued binary |

Promotion to `candidate` **requires the second axis to speak.** A discard-rate
excess is an anomaly detector, not an explanation, and promoting on it alone
would be the "large number, therefore interesting" move this repository refuses
everywhere else.

`reasons` and `vetoes` are separate fields. They were conflated once and the
consequence was silent: `assign_tier` runs twice — before and after
characterisation — and on the second pass it read its own informational note from
the first pass as a veto and demoted every object it had just promoted.

### The decision is on the SET

Unweakened, and reused rather than reimplemented: `seti.loom.replication` —
mis-linkage collapse first, then element clustering, orbital-pole coherence (an
orientation tensor, because poles are *axes*), inclination isotropy, resonance
concentration — each against matched random subsets of **this** screened sample.

The one substitution is the matching covariates: they are SEXTANT's, not LOOM's.
An object flagged by a discard-rate excess is preferentially fast, faint and
crowded, and objects like that are not uniformly distributed in the belt.
Matching on the covariates of the *rejection rate* is what stops the population
statistics from firing on the scanning law. Below 200 screened objects or 5
flagged ones the stage returns `INSUFFICIENT_POPULATION` and refuses to run.

`photometric_homogeneity` needs per-band absolute magnitudes, which
`sso_observation` does not carry; it degrades with a named reason rather than
silently passing.

---

## 7. Unsettled assumptions — the probe list

These could not be settled offline. Every one of them is **loud in the code**,
and every one is measurable on the runner. They are listed in the order a probe
should answer them.

1. **`epoch` vs `epoch_utc`, and the time scale of `epoch`.** Not settled.
   Candidates in `CANDIDATE_EPOCH_CONVENTIONS`: `epoch` in TCB / TDB / TT with
   zero point JD 2455197.5 (2010-01-01, Gaia's reference epoch) or a plain
   Julian date; `epoch_utc` in UTC or TDB.
   *Why it matters:* TCB − TDB is **18.8 s** by mid-2015, TT − UTC is 68.184 s,
   TCB − UTC ≈ **87 s**. A main-belt object moves ~30 arcsec/hour, so 87 s is
   ~0.7 arcsec of along-track offset **on every object, proportional to sky
   rate** — which is precisely what this channel would otherwise report as a
   population-wide detection.
   *How it is settled:* `resolve_conventions` runs the chain under every
   candidate and picks the one that minimises the robust along-scan residual.
   Every wrong answer is orders of magnitude worse, so the margin is reported and
   a margin below 3 returns `AMBIGUOUS` rather than a winner.
   `fit_common_time_offset` independently reports the residual timing error in
   seconds and flags it when it lands near 1 s (a leap second), 32.184 s
   (TT − TAI), 68.184 s, 18.8 s or 87 s.
2. **Whether the archive's reduced positions already contain light time, stellar
   aberration and solar deflection.** Not settled; all three are switches
   resolved by the same measurement. Omitting aberration costs 11–19 arcsec of
   along-scan residual in the synthetic tests — four orders of magnitude above
   the signal.
3. **The sense of `position_angle_scan`.** Not settled from a schema; settled
   from the data by `verify_scan_convention` (§2), which needs no ephemeris.
4. **The encoding of `astrometric_outcome_ccd` / `astrometric_outcome_transit`.**
   Not settled. `NOMINAL_OUTCOME_VALUES = (0.0,)` is a placeholder. **Histogram
   both columns before any rate computed from them is believed** — a wrong
   reading here does not fail, it changes the denominator.
5. **Whether `is_rejected` and the outcome flags are the same rejection.** If
   they disagree, that disagreement is the first result and both rates should be
   carried.
6. **`nongrav_parameters_fitted` per object.** JPL fits A1/A2/A3 for a small
   minority. The per-object value must be filled from the SBDB record: for those
   objects the signal was fitted out and the residual is not a blind measurement.
7. **Whether `x_gaia…vz_gaia` are barycentric or heliocentric in the archive's
   naming.** The channel *requires* barycentric and says so; a heliocentric
   observer against a barycentric target is an 0.008 au error ≈ 1 arcsec. The
   `*_geocentric` twins exist and must not be crossed with them.
8. **The Sun's barycentric state.** Optional (`sun_state`); without it the Sun is
   put at the barycentre, a 0.008 au offset that is a 0.3% error on `g(r)` and
   0.08% on the along-track basis direction. Recorded in `notes`, and a runner
   with the Sun's ephemeris should pass it.
9. **The systematic's true correlation length.** Modelled as shared within a
   transit (nominal) and within 180 days (pessimistic); the larger uncertainty is
   quoted. The real correlation structure is a mission-calibration fact worth
   measuring from the residuals themselves once they exist.

---

## 8. How the live parts are pulled

Nothing in `residuals.py` or `screen.py` touches the network. Three inputs come
from the runner.

**The ephemeris.** `target_state(jd_tdb) -> (N, 6)` **barycentric** au and au/day.
The practical form is JPL Horizons `VECTORS` with `CENTER=@0`, in two passes: ask
at the observation epochs, compute τ from the returned range, then ask again at
`t − τ`. That is exact and avoids interpolation error entirely. If a gridded pull
is used instead, cubic Hermite on a **0.25 day** grid gives ~0.02 mas; a 1-day
grid gives ~5 mas and would swamp the signal.

**The orbit source.** JPL SBDB per object, for `a`, `e`, `H`, the fit arc, and —
critically — whether the fit carried non-gravitational parameters. Register it as
an `OrbitSource` with the provenance filled in; `KNOWN_ORBIT_SOURCES` holds the
three classifications already established.

**The binary catalogue.** VizieR `J/A+A/688/A50`, all rows, keeping the number
and the designation; hand it in as `BinaryCatalogue(rows=..., retrieved_utc=...)`.
Add arXiv:2605.22702's FPR candidate list when it is machine-readable.

---

## 9. Files

| file | what |
|---|---|
| `src/seti/sextant/residuals.py` | provenance gate, time scales, the astrometric chain, the scan frame and its verification, two-body propagation and state partials, the variational signal basis, block-correlated GLS, the six-model comparison, the convention resolver |
| `src/seti/sextant/screen.py` | the rejection-pattern observable, the stratified quasi-binomial null, covariate robustness, binaries, the tier ladder, the population decision |
| `src/seti/sextant/acquire.py` | Gaia TAP acquisition (runner-only; separate workstream) |
| `tests/test_sextant_residuals.py` | 45 offline tests, all on synthetic observations with a known injected signal |

## 10. Related channels

- **LOOM** (`docs/loom.md`) — the same observable at arcsecond scale on Rubin,
  the population architecture this reuses, and the source of the `a = 2c2`
  conversion corrected in §4.
- **`docs/rubin-outage.md`** — why Rubin is dark, and why this is not a stopgap.
- **`docs/substitute-surveys.md`** — the measured column inventory that made this
  channel possible.
