# LOOM — a population search for self-replicating probes in the solar system

*A loom is the machine that makes copies of a pattern.*

**Channel status:** built, offline-tested (86 tests), run against live data on
2026-07-30. **The channel works and the data is not yet old enough to use it** —
see §2.2. The probe changed the architecture; read §2.1 first.

---

## 1. The claim

A von Neumann probe is **defined** by self-replication. That single fact
determines the whole design, and it is the reason this is not another
"weird asteroid" search.

A single object with an unexplained non-gravitational acceleration has a dozen
mundane readings — a badly determined orbit, an unmodelled perturber, an
undetected satellite dragging the photocentre, a comet nobody has caught
outgassing. Every one of those has been the answer before. Replication predicts
something no individual object can supply: a **population sharing an origin**.

So LOOM screens objects one at a time and **decides on the set**. The per-object
stage produces a ranked anomalous set; the decision is made by
`seti.loom.replication`, against a null of matched random subsets of the same
screened sample. This is the architecture TOCSIN arrived at for stars, for the
same reason: a per-object test is a contamination problem, whereas a
population-structure test on the *set* of flagged objects is immune to any
contaminant that does not itself cluster in orbital elements and orbital pole.

---

## 2. What the data gives

Rubin's alert packets (schema v11.1) carry, for **every detection matched to a
known minor planet**, the observed-minus-predicted ephemeris offset already
decomposed. Verified against the upstream `lsst.v11_1.ssSource` and
`lsst.v11_1.mpc_orbits` Avro schemas; ALeRCE exposes the packet fields verbatim,
lower-cased, in schema order.

`alerce_tap.lsst_ss_detection` — 40 columns, one row per solar-system detection:

| what | columns |
|---|---|
| **the observable** | `ephoffsetalongtrack`, `ephoffsetcrosstrack`, `ephoffset`, `ephoffsetra`, `ephoffsetdec` — arcsec, UCD `stat.fit.omc` |
| the prediction | `ephra`, `ephdec`, `ephvmag`, `ephrate`, `ephratera`, `ephratedec` |
| geometry | `phaseangle`, `elongation`, `toporange`, `heliorange`, and both rates |
| state vectors | `helio_x…helio_vtot`, `topo_x…topo_vtot` |
| association quality | `diadistancerank` (1 = nearest source to the prediction) |
| identity | `ssobjectid`, `designation` |

`alerce_tap.lsst_mpc_orbits` — 53 columns, one row per object: full elements each
with an uncertainty twin, `h`/`g`, the fit-quality block (`arc_length_total`,
`nobs_total`, `nopp`, `u_param`, `normalized_rms`, `epoch_mjd`), `earth_moid`,
and the non-gravitational block `yarkovsky`, `srp`, `a1`, `a2`, `a3`, `dt` with
uncertainties.

`alerce_tap.lsst_ss_object` — a **six-band phase-curve fit per object**: `H` and
`G12` per filter with covariances, plus `extendedness`, `moidearth`,
`tisserandj`. Column spellings here are *not* verified against a primary source,
so the channel asks `TAP_SCHEMA` for the real list rather than guessing (§7).

Nothing in this channel needs an orbit integrator, a proprietary catalogue, or a
credential. ALeRCE's TAP service is public.

### Two unit traps, both load-bearing

- `lsst_mpc_orbits.yarkovsky` is documented in units of **1e-10 au/day²**. Bennu's
  fitted `A2 = -4.62e-14` appears in the column as `-4.6e-4`. Reading the column
  raw overstates every acceleration by ten orders of magnitude and puts the entire
  catalogue above every ceiling. Guarded by
  `test_yarkovsky_column_unit_conversion`.
- `srp` is in **m²/ton** (1 m²/ton = 1e-3 m²/kg). `a1`/`a2`/`a3` are *also*
  documented m²/ton, which is dimensionally wrong for Marsden accelerations, so
  those three are treated as unit-unverified and are not used until calibrated
  against objects with published JPL solutions.

Both conversions live in one module (`nongrav.py`) so there is exactly one place
they can be wrong.

---

## 2.1 What the live mirror actually holds — measured 2026-07-30

`loom-probe` ran against ALeRCE on 2026-07-30 (`results/loom/probe.json`, 38
queries). It is the reason this section exists, and **four of its findings changed
the code**. Each one would otherwise have produced a confident wrong answer rather
than an error, which is the failure mode this repository fears most.

**1. There is no `diasourceid`.** The per-detection key of `lsst_ss_detection` is
`measurement_id`, so the join to `detection` — needed for the epoch *and* the sky
position, neither of which `ssSource` carries — is
`d.measurement_id = ss.measurement_id AND d.oid = ss.ssobjectid`. The
`diasourceid` form raised "No such field known", which is at least loud. The
tempting alternative, `d.oid = ss.ssobjectid` alone, does **not** raise: it
silently returns the cross product of every prediction for an object with every
detection of that object, so a query that looks like a residual time series is an
N×M cartesian join and its scatter is an artefact of the join. It is retained
as `join_on: object`, labelled diagnostic-only, and `_join_clause` raises on
anything else. Confirmed alongside: solar-system `detection` rows carry `sid = 2`.

**2. Only the *rotation* is missing, not the offset.** `ephoffsetalongtrack` and
`ephoffsetcrosstrack` are NULL for **all 961,558** solar-system detections, and
`ephrate`/`ephratera`/`ephratedec`/`ephvmag` are identically zero. But
`ephoffsetra` and `ephoffsetdec` **are** populated, and they reproduce `ephoffset`
in quadrature to the last digit — on object 20607267131895873,
`hypot(0.009776699, −0.028678588) = 0.030299261` against `ephoffset =
0.030299261`. So the survey's own offset *vector* is available in arcsec (with the
`cos(dec)` factor already applied, as documented), and the only thing that has to
be supplied locally is the direction of motion to project it onto.

`residuals.decompose_offset` does that rotation. The track direction comes from the
object's own neighbouring detections rather than from the (populated) state
vectors, deliberately: the alert schema does not state whether those vectors are
ecliptic or equatorial, and a 23.4° frame error would rotate along-track into
cross-track and destroy the exact quantity being measured. Two detections half an
hour apart define the track with no frame assumption at all. Differencing the raw
positions is kept as a fallback and was verified against the offset columns to
better than a milliarcsecond on live rows.

The chain is **validated for free**: the reconstructed magnitude must agree with
`ephoffset`, and `analyse_series` returns
`RECONSTRUCTION_DISAGREES_WITH_EPHOFFSET` and stops if it does not. An object
observed only once per night has no measurable track direction and is reported
untestable on that axis rather than assigned an arbitrary one.

**3. The 1″ association radius is confirmed, and it is a hard ceiling.** The
`ephoffset` histogram over all 961,558 rows:

| `ephoffset` (arcsec) | detections |
|---|---|
| 0.0 – 0.2 | 886,817 (92.2%) |
| 0.2 – 0.5 | 58,236 |
| 0.5 – 0.9 | 14,313 |
| 0.9 – 0.99 | 2,007 |
| 0.99 – 1.0 | 185 |
| **≥ 1.0** | **0** |

Nothing at all above 1.0″. That is the source-association radius, not a
coincidence, and it means **the channel is blind to residuals above 1″ by
construction**: a large enough anomaly fails to associate with its prediction and
never enters the table. The direction of the bias is the worst possible one — it
removes the *most* anomalous objects — so a null from this channel is weak
evidence about the solar system and strong evidence only about **sub-arcsecond
residual structure**. Any result must say so. (What partly rescues it: an
acceleration builds up, so an object crossing the 1″ boundary mid-arc shows a
*truncated* series, and the disappearance of a previously-tracked object is itself
a detectable signature — a lead worth following rather than a limitation to
accept.)

Two more zero-fill traps found the same way: `toporange` and `heliorange` go down
to ~1e-8 au on rows where the geometry was not computed. The arcsec-to-km
conversion is *proportional* to the range, so a zero-filled range would turn a
real angular residual into a zero physical displacement — a clean null on an object
that was never measured. `residuals.usable_range` floors both at 0.005 au (twice
the lunar distance, below anything physically possible) and propagates NaN.

**4. Zero is this mirror's "missing", and zero is not NULL.** `srp`, `a1`, `a2`,
`a3` and `dt` are non-NULL for 1812 of 130,909 orbit rows and **every one of those
values is exactly 0.0** — fill, not measurement. `yarkovsky` is non-NULL for 1822
rows but **non-zero for only 12**. The same pattern appears in `a`, `mean_motion`,
`period`, `not_normalized_rms`, `arc_length_total`, `u_param` (non-zero for 2,431
of 130,909) and `ephvmag` (non-NULL for 456,282, non-zero for **none**).

`COUNT(col)` counts a zero as present, so the first null-fraction query reported
these columns as populated. Every read now goes through `screen._fz`, which treats
exact zero as missing. This matters because a *measured* non-gravitational term of
zero is the strongest possible statement that an object is ordinary, whereas an
absent one means untestable — and the whole channel is built on not confusing those
two.

*A bug of mine, found the same way and worth recording:* the fix,
`SUM(CASE WHEN col <> 0 THEN 1 ELSE 0 END)`, made the service infer a 16-bit type
from the literal `1` and then fail VOTable serialisation for every count above
32767 — `Field 'n_nonzero', value '961558': 'h' format requires …`. So the non-zero
count was lost for exactly the columns that had one. It is now a separate
`COUNT(*) … WHERE col <> 0` query, which returns a proper integer.

**The consequences for the architecture, stated plainly.** **12** orbit rows carry a
genuine `yarkovsky`, and **7** reach |A2| > 3σ. The `srp` column — the
area-to-mass ratio, which §3.5 identifies as the single strongest artificiality
discriminant and the one thing outgassing does not reproduce — is **entirely
zero-filled**. So:

- **Path A is 7 objects.** It is a cross-check, not a search. This is the outcome
  the design anticipated and the reason the parent-population query does not
  default to `require_nongrav`.
- **Path B is the channel**: per-detection `ephoffset` structure over 961,558
  detections, on the rotated `ephoffsetra`/`ephoffsetdec` vector. It is also the
  unbiased path, since it does not inherit MPC's choice of which objects were worth
  fitting. All 130,909 orbit rows have detections, so `H` and the fit quality are
  available for every object in it, and there are many objects with ≥8 detections.
- **The timing veto is currently untestable.** `ephrate` is zero-filled, so
  `fit_common_timing` cannot run and a shutter or clock offset is *indistinguishable*
  from a real along-track acceleration in this mirror. The run records that
  explicitly rather than leaving it implicit in a NaN.
- **Designations differ in form between the tables.** `lsst_ss_detection` carries
  the packed form (`J97L01J`, `K16Cd3G`); `lsst_mpc_orbits` carries both. The
  control set is written unpacked (`2020 SO`), so matching the packed string
  against it would find nothing — silently — and report `NO_CONTROLS_PRESENT` for an
  object that was right there. The screen now takes
  `unpacked_primary_provisional_designation`.
- **`orbit_type_int` is −1 on every row sampled**, so the dynamical-class label is
  unusable and population membership has to be inferred from the elements.

Confirmed working, for the record: the corrected join is **exactly 1:1**
(`join_cardinality` returns no `measurement_id` matching more than one detection
row), and the broker frontier is MJD **61235.4**.
- **The photometric axis is present in the schema and empty in the data.**
  `lsst_ss_object` exists with 81 columns (real names measured: `g_h`, `g_g12`,
  `g_chi2`, `g_slope_fit_failed`, `extendednessmedian`, `tisserand_j`, keyed on
  `oid`) and **0 rows**. `ss_objects()` reports `EMPTY`; it is not a null result.
- **Promotion to `candidate` cannot currently happen through the AMR channel**, so
  in this mirror it requires the law-discrimination channel — which needs
  multi-apparition arcs, and the ALeRCE orbit epochs span MJD 59000–61200 with
  arcs up to 59,535 days, so those exist.

---

## 2.2 What the live runs found: the survey is a month old

Three screens ran on 2026-07-30. The funnel is now sound — 66,686 orbits pass the
quality cuts, 2,759 of them have ≥12 solar-system detections, the shortlist and the
parent are the same population (`shortlist_in_parent_fraction = 1.0`), the join is
1:1, and 2,287 objects were analysed in 20 batched queries in 967 s.

**The channel currently reports nothing, and the reason is the binding one.**

The second run reported `REPLICATION_STRUCTURE_DETECTED` on 150 anomalies, with
pole coherence and inclination isotropy both at the randomisation floor. It traced
to a bug in this code: `analyse_series` used `scatter / sqrt(n)` as the per-point
astrometric uncertainty. That is the uncertainty on the *mean*, so for 25
detections it understated the per-point error five-fold, inflating every
acceleration signal-to-noise by five and every Δχ² by twenty-five. The
corroborating evidence was already in the output — the anomaly score correlated
with detection count at ρ = −0.475, which is the "ranks objects by how well they
were observed" failure — and the warning did not fire because the threshold was
0.5. Fixed with a two-pass fit (rescale σ by √(reduced χ²) about the *fitted
model*, never below the instrumental floor); the quality-correlation threshold is
now 0.3 and configurable. **150 anomalies became 4.**

The remaining four were also artefacts, and their diagnosis is the channel's
current limit: their **residual series span 2 to 29 days**, against orbit arcs of
8,000 to 16,000 days, and their implied accelerations were 10⁴ to 10⁸ times the
momentum ceiling. A quadratic fitted over a two-week baseline has no leverage on
curvature; the fitted acceleration is an extrapolation and its formal error means
nothing.

The cause is simply that **the LSST survey proper began on 2026-06-30**. Every
solar-system object in the mirror has one apparition, so:

- `law_discrimination` returns `INSUFFICIENT_R_SPAN` for every object — the
  heliocentric distance barely changes over a month, and that test is the
  channel's central novelty claim;
- `apparition_trend` returns `TOO_FEW_APPARITIONS` for every object;
- `fit_common_timing` cannot run at all, because `ephrate` is zero-filled, so a
  shutter offset remains indistinguishable from a real along-track acceleration.

Two gates now enforce this rather than letting it leak into a result:
`min_residual_arc_days = 180` and `min_apparitions_for_promotion = 2`. Under them
all four objects become `untestable` with the reason named, and the assessment
returns `INSUFFICIENT_POPULATION`.

**This is a "not yet", not a null.** The channel is correct, the reconstruction is
validated on real data to 1.4×10⁻⁸ arcsec, and every discriminant it needs becomes
available as the arc lengthens. That is what a standing weekly screen is for. Per
`CLAUDE.md` a clean null would be a reason to change the question; a
coverage-limited non-result is a reason to keep the screen running and wait.

### A correction to the record

An earlier reading of the second run reported that `2020 SO` (the 1966 Surveyor 2
Centaur) and `2007 VN84` (the Rosetta spacecraft) were present in the screened
sample. **They were not.** `normalise_designation` stripped whitespace and then
matched a leading run of digits as the permanent number, dropping trailing letters
as a name — which turns every provisional designation into its discovery *year*.
`2020 SO` became `2020`, and so did every other object discovered in 2020. That
matched hundreds of ordinary asteroids, forced 287 of them into a shortlist as
"positive controls", and produced a `SCREEN_INSENSITIVE` verdict about objects that
were never there.

Fixed, with a regression test: the permanent-number branch now fires only when the
digits stand alone or are followed by a name of three or more letters that is not a
plausible discovery year. The corrected verdict is `NO_CONTROLS_PRESENT` — no known
artificial object is in this sample, the control is **unexercised, not passed**, and
sensitivity to the artificiality discriminant remains untested.

A matcher that is too permissive does not merely add noise in this channel. It
fabricates the one falsifiable check the channel has.

---

## 3. The governing methodology

### 3.1 The gate is a theorem, not a fit

Yarkovsky acceleration is recoil from re-radiating absorbed sunlight, so the
thermal photons cannot carry away more momentum than the intercepted beam
delivered. For a body of area-to-mass ratio `AMR` at heliocentric distance `r`:

```
|a_NG|  ≤  ε · (Φ_1au / c) · AMR · (1 au / r)²        Φ_1au/c = 4.5398e-6 N/m²
```

- `ε = 1` — every absorbed photon's momentum re-emitted in a single direction.
  Physically unreachable.
- `ε = 2` — perfect specular reflection. The absolute limit for **any**
  radiation-driven process whatsoever.

Calibrated against three objects with independently published `A2`, the *realised*
efficiency is 2–8%:

| object | D (m) | ρ | measured \|A2\| | ceiling (ε=1) | ε_eff |
|---|---|---|---|---|---|
| (101955) Bennu | 490 | 1190 | 4.62e-14 | 5.83e-13 | **0.079** |
| 2005 ES70 | 60 | 1500 | 1.285e-13 | 3.78e-12 | **0.034** |
| 2009 BD | 4 | 1500 | 1.143e-12 | 5.66e-11 | **0.020** |

So `ε = 0.1` is already a generous envelope for real thermal recoil. This table is
recomputed by `nongrav.calibration_table()` and pinned by
`test_momentum_ceiling_matches_measured_objects`: if a refactor moves the ceiling
by any factor, the test suite says so rather than the candidate list changing
silently.

Why a momentum budget instead of a thermophysical model: the textbook route needs
albedo, thermal inertia, spin period and obliquity, none of which are known for
almost any object in the sample. A gate resting on four unmeasured parameters
cannot be trusted in either direction. The momentum ceiling holds whatever those
parameters are.

The empirical envelope (`fit_envelope`, a high quantile of |A2| per H bin fitted
to the survey's own population) runs **alongside** as a cross-check, not as the
gate. A survey-wide calibration error in `H`, or a units error in the offset
columns, moves the empirical envelope and leaves the ceiling alone — so a
disagreement between the two is diagnostic and is reported rather than reconciled.

### 3.2 Amplitude is not the discriminant

This is the channel's tightest constraint and the easiest thing to get wrong.

`ephOffset` is *Rubin's* position (astrometric precision ~10 mas, with a 3–7 mas
systematic floor) minus a prediction from an *MPC* orbit fitted to decades of
heterogeneous historical astrometry whose star-catalogue biases reach **175 mas**.
A well-observed main-belt object shows `ephoffset` at the ~0.1 arcsec level
(Pan-STARRS1 astrometric RMS is 0.12 arcsec; Catalina 0.69); a short-arc object
can legitimately show several arcsec. Residuals of 0.1–1 arcsec are routine and
carry no information about the object.

What carries information is the residual's **geometry** and its **time
structure**:

| test | what it asks | why the confounders fail it |
|---|---|---|
| `along_cross_partition` | is the residual confined to the along-track direction? | a transverse force displaces an object along its track; star-catalogue bias is a property of the reference frame and mis-association picks a random neighbour — neither has a directional preference |
| `apparition_trend` | does the per-apparition mean offset grow monotonically, with consistent sign? | an acceleration accumulates; an orbit-fit error's sign depends on where in the fitted arc the epoch falls, so it wanders |
| `drift_fit` | is the *quadratic* term in time significant over the linear one? | a wrong mean motion is exactly linear; only an acceleration is quadratic |
| `law_discrimination` | which heliocentric-distance law does the drift follow? | see §3.3 — this is the novel one |
| `fit_common_timing` | is the sample's residual field just a shutter offset? | a clock error is `rate × dt`: linear in `ephrate`, identical for every object, identifiable across a population though perfectly degenerate within one |
| `sky_coherence` | how much of the residual field is explained by *where on the sky* it was measured? | star-catalogue bias is coherent with position, not with the object |
| `quality_independence` | does the anomaly score track arc length, opposition count or `normalized_rms`? | a blind Yarkovsky search returns a *majority* of spurious detections at nominal S/N > 3, and short arcs with inflated residuals are the usual cause |

### 3.3 The discriminant that is actually new: which law?

"An inactive small body with a non-gravitational acceleration larger than
Yarkovsky predicts" is **not novel and not unexplained**. Seligman et al. (2023,
PSJ 4, 35) found seven — 1998 KY26, 2005 VL1, 2016 NJ33, 2010 VL65, 2006 RH120,
2010 RF12, 2003 RM — the "dark comets", and hidden outgassing is the accepted
reading, extended by Farnocchia & Seligman (2024, PNAS) to two distinct
populations. A magnitude threshold rediscovers them.

What separates hidden outgassing from an engineered object is **what the
acceleration does with heliocentric distance**:

- **sublimation** — JPL's water-ice law `g(r) = α(r/r₀)^−m [1+(r/r₀)^n]^−k` with
  α = 0.1112620426, r₀ = 2.808 au, m = 2.15, n = 5.093, k = 4.6142: steeply
  falling with a knee near 2.8 au;
- **radiation** (Yarkovsky, SRP) — a clean `r^−2`;
- **constant** — distance-independent, which is what nothing natural does.

With a multi-apparition arc the drift rate is measured at several heliocentric
distances and the three are separable per object. The comparison is made on
physical along-track displacement (km, via `toporange`) rather than arcsec, so the
observing geometry is not left in the signal, and the Δχ² is rescaled by the
winning model's own reduced χ² — the standard remedy for a common error-scale
error, without which every object with a slightly wrong σ shows a decisive
preference. Where the arc does not sample enough heliocentric range the test
returns `INSUFFICIENT_R_SPAN`, which is the honest answer for a single apparition.

### 3.4 Mis-linkage: the contaminant that looks exactly like a family

This is not hypothetical and it is the most dangerous thing in the channel.
`|A_i| ≳ 1e-8 au/day²` breaks MPC's linking algorithms outright: tracklets fail to
link across apparitions and can be attached to *multiple* designations. So one
genuinely accelerating object can enter the catalogue several times, giving a set
with near-identical orbital elements, anomalous residuals, and non-overlapping
observation epochs — precisely the signature this channel calls replication.

The separator is epoch coverage: real distinct objects in a family are observed
*contemporaneously*; the same object under two designations is observed in
disjoint intervals. `replication.linkage_duplicates` runs **before any statistic
sees the set**, collapses each group to one representative, and reports the
collapse. A cluster that evaporates under it is a finding about MPC linking, not
about the solar system.

### 3.5 The tier ladder

| tier | meaning |
|---|---|
| `untestable` | no usable `H`, or no acceleration measurement, or a fitted value with no uncertainty. Recorded as its own outcome — "we could not look" is never folded in with "we looked and it was fine" |
| `ordinary` | below the realistic thermal-recoil envelope (ε = 0.1). Essentially every asteroid |
| `watch` | above ε = 0.1, below the hard ceiling. Real Yarkovsky can plausibly reach here for an extreme spin state |
| `interest` | above the ε = 1 hard ceiling. Sunlight cannot drive this: either mass loss (a dark comet) or not a rock |
| `candidate` | anomalous in **area-to-mass ratio**, or above the hard ceiling **with** a distance-independent acceleration — and with every systematic explanation excluded |

Promotion to `candidate` **requires an artificiality channel, never magnitude
alone.** That is deliberate and it is the entire novelty position. Area-to-mass
ratio is where outgassing and engineering part company: mass loss raises an
object's acceleration but does not turn a rock into a thin shell.

| object | AMR (m²/kg) | implied ρ·D (kg/m²) |
|---|---|---|
| 2009 BD (natural, H≈28.4) | (2.97 ± 0.33)e-4 | 5050 |
| 2011 MD (natural, D≈6 m) | ~2.3–3.9e-4 | 2100–3600 |
| **J002E3** (Apollo 12 S-IVB) | 7.9e-3 | **190** |
| **WT1190F** (lunar-origin body) | (1.18 ± 0.05)e-2 | **127** |
| 1I/'Oumuamua (if pure SRP) | 1.08 | 1.4 |

Identified artificial debris sits 20–40× above the natural small-body locus, and
ρ·D ≈ 130–190 kg/m² implies ρ ≲ 100 kg/m³ for a metre-scale body: diagnostic of a
hollow shell, impossible for rock. The gate is AMR against the object's **own
size** (`amr_ceiling_ratio`, with ρ_min = 500 kg/m³ — already the bottom of the
measured rubble-pile range, so a ratio above 1 is not an argument about porosity),
which is strictly stronger than a flat floor.

Each channel is gated on **its own** parameter's signal-to-noise. An object with a
clean orbit and a well-measured `srp` but no fitted `yarkovsky` is perfectly
testable on the radiation-pressure channel, and a combined gate would reject it
for lacking an `A2` it never needed.

---

## 4. The population tests — where the channel decides

All against a null of **matched random subsets of the same screened population**,
stratified on `H`, arc length, opposition count, `normalized_rms` and detection
count. Anomalous objects are preferentially small, faint and short-arc, and
objects like that are not uniformly distributed in the belt — so the matching is
what stops the statistics from firing on the selection function instead of on the
sky. TOCSIN shipped this machinery inert once and promoted four
deep-drilling-field artefacts; the lesson is in the tests here.

| test | statistic | gate? |
|---|---|---|
| `element_clustering` | median k-NN distance in standardised (a, e, sin i) — left tail | yes |
| `pole_coherence` | largest eigenvalue of the orientation tensor `Σ p pᵀ/n` — right tail | yes |
| `inclination_isotropy` | mean `sin i` — right tail; a captured population keeps no memory of this disc's plane | yes |
| `resonance_concentration` | median distance to the nearest mean-motion resonance — left tail (signature **S29**, "monuments at stable points") | yes |
| `photometric_homogeneity` | dispersion in six-band colour/`G12` space — left tail | yes |
| `size_distribution` | cumulative `H` slope against a Dohnanyi reference | **no — reported only** |

Two design notes that are not incidental:

- **Poles are axes, not vectors.** The orientation tensor is the correct
  concentration measure; a vector mean would partially cancel antipodal poles and
  understate real alignment. This is the same axial-statistics care COMPASS takes
  with Gaia NSS orbits, where treating projective-sphere data as full vectors is
  wrong by construction.
- **`size_distribution` cannot be a gate.** The anomalous set is *selected* by
  exceeding a size-dependent envelope, so its size distribution is already shaped
  by that selection, and a narrowness statistic on it would partly measure the
  selection function. It is informative only inside an already-identified cluster,
  and even there real families scatter about the reference slope. It returns
  evidence for a human to weigh, with `gate=False`.
- **`photometric_homogeneity` cannot separate manufactured from collisional.** A
  collisional family is homogeneous too — the fragments came off one parent. What
  it *can* separate is "a set with a common origin" from "a random subset of the
  belt", and it is an independent test because nothing photometric enters the
  dynamical selection. The manufactured-vs-collisional separation is what
  `size_distribution` is for, which is why that function exists despite not being
  a gate.

`n_null` must resolve p well below the Bonferroni threshold for the number of
gates, or the randomisation floor swamps the correction; `replication_tests`
refuses to report a detection when it does not
(`INSUFFICIENT_RESOLUTION`), and refuses to run at all below 200 screened objects
or 5 anomalies (`INSUFFICIENT_POPULATION`).

---

## 5. The positive control — the thing no other channel here has

Every other search in this repository has no confirmed positive. There is no
object known to be a technosignature, so a screen's sensitivity can only be argued
from injection tests against a *model* of the signal, and if the model is wrong the
argument is worthless.

A solar-system artefact search is the exception. Human beings have put artificial
objects into orbits that were catalogued as minor planets, and in each case what
gave them away is exactly what this channel measures:

- **`J002E3`** — the Apollo 12 S-IVB third stage, identified by AMR 7.9e-3 m²/kg.
- **`WT1190F`** — a lunar-origin rocket body, AMR (1.18 ± 0.05)e-2.
- **`2020 SO`** — the Centaur upper stage from the 1966 Surveyor 2 launch, found by
  a survey as an asteroid, flagged by anomalous area-to-mass ratio, and confirmed
  by near-infrared spectroscopy of 301 stainless steel. The cleanest end-to-end
  case there is.
- **`2007 VN84`** — the Rosetta spacecraft, designation retracted on
  identification. The canonical demonstration that a survey *will* designate a
  spacecraft as a minor planet.
- also `2010 KQ`, `6Q0B44E` (probable), and `1991 VG` (genuinely disputed, and
  therefore excluded from the pass/fail arithmetic while still being reported).

**If the screen does not recover objects like these, it does not work.** That is a
falsifiable statement about the pipeline, obtainable from real data.

Two further control sets do different jobs. `NONGRAV_DETECTED` (Bennu, 2009 BD,
2005 ES70, 1999 MN, Golevka, Apophis, 152563) validates sensitivity to a *real*
non-gravitational acceleration: they must be recovered as accelerating and then
correctly rejected as natural because they sit below the momentum ceiling.
`DARK_COMETS` — Seligman's seven — is the specificity test: they *must* be flagged
by a magnitude cut, because that is what makes a magnitude cut insufficient, and
they must be separated from an engineered object by `law_discrimination` preferring
sublimation, by ordinary area-to-mass ratios, and by resolved extent where a coma
is detectable.

Most of these objects are not currently observable — WT1190F impacted Earth in
November 2015 — so the honest expectation is that a Rubin-era sample contains none
of them. `control.validate` therefore reports `NO_CONTROLS_PRESENT` as a
first-class outcome: **unexercised, not passed.** The control set earns its place
the moment the survey catalogues one new artificial object, which it will.

Provenance discipline: each entry carries the source of its number and a
`confidence` field, and a value that could not be verified from a primary source
is recorded as `None` with the reason. `2020 SO`'s fitted AMR is not in any
reachable public source, so it is present as an identification without a number
rather than with a plausible figure.

---

## 6. Novelty — the honest accounting

A prior-art sweep ran 178 arXiv queries across eight angles on a GitHub runner
(the sandbox has no egress to arXiv/ADS), verified 108 IDs against their real
titles before citation, and machine-counted term occupancy over 34 full texts.
Evidence is in `results/vnprobelit*/`. What it found:

### Occupied — cite and differentiate, do not claim

- **Single-object anomalous acceleration as a technosignature is named in the
  literature and a competitor is in review.** Lazio, *Solar System
  Technosignatures* (arXiv:2606.13797, June 2026) §§2.1.1, 2.3.4 explicitly treats
  anomalous non-gravitational acceleration, discontinuous orbital change
  (manoeuvres), secular change from SRP or low-thrust propulsion, and the
  Yarkovsky/outgassing confound — and notes that dark-comet accelerations are
  "well within the achieved capabilities of spacecraft propulsion systems that our
  civilization has produced". It cites **Lazio & Mahabal 2026, *On Anomalous
  Asteroid Accelerations*, Acta Astronautica, submitted** — not on arXiv, by
  exactly the right two people, on exactly this observable. Treat the single-object
  non-grav channel as occupied in flight.
- **Probes/lurkers in the solar system: proposed repeatedly, executed on a
  different observable.** Benford, *Looking for Lurkers* (arXiv:1903.09582) is the
  canonical proposal — 23 citing works in seven years and not one an executed
  observational artificiality test of a co-orbital; its full text contains zero
  occurrences of "non-gravitational", "Yarkovsky", "residual" or "LSST". Gertz's
  four papers are electromagnetic. Villarroel et al. (arXiv:2510.17907) *is*
  executed on data but uses Earth's shadow as a filter on ZTF 30-second exposures.
  Valdes & Freitas (1983, *Icarus*; 1985, *Acta Astronautica*) surveyed the
  Earth–Moon Lagrange points photographically — null, and no arXiv record exists,
  so cite the journals.
- **Population photometric outlier detection on minor planets is done, at Rubin
  scale, without SETI framing.** SNAPS (arXiv:2302.01239, 2405.20176, 2604.27420)
  is the only Solar-System-dedicated Rubin broker; it runs unsupervised outlier
  detection over 31,693 ZTF asteroids and 32,752 feature-space permutations on 15
  features (`G_BR, H_BR, G_BG, H_BG, LCAMP, ROTPER, GRCOLOR, SIGGRCOLOR,
  PERIPOWER, NEOWISE_DIAM, NEOWISE_VALBEDO, MPC_A, MPC_E, MPC_I, HAVG`) — none of
  them a dynamical residual or a non-gravitational parameter. **Do not claim**
  nobody has looked for photometric outliers among minor planets. The defensible
  claim is only that nobody has read them as technosignatures or combined them
  with dynamical residuals.
- **The Lagrange-point and co-orbital niches are searched hard, astrophysically.**
  DECam L5 (2001.08229), track-before-detect L4 (2102.09059), WFST L4
  (2606.31751, the tightest limit to date). Six independent SETI-framed queries on
  these niches returned zero results, so the *framing* is unclaimed; the *sky* is
  not unsearched.

### Unoccupied — the actual novelty

- **`ephOffset` as a population observable.** `abs:"ephemeris" AND abs:"offset"
  AND abs:"alert"` → 0 results. `abs:"astrometric residuals" AND
  abs:"anomalous"` → 0. `abs:"non-gravitational" AND abs:"population" AND
  abs:"survey" AND abs:"asteroids"` → 0. The phrase "ephemeris residual" appears
  zero times in SNAPS, Fink-SSO, AHA, Lazio 2606.13797, Davenport 2508.16825 and
  Ellery 2510.00082. And it is the only Rubin SSO observable that *scales*:
  delivered per detection for every known object. The `ephOffset*` fields first
  appear in alert schema v10.0 (committed 2025-11-24) and went world-public on
  2026-02-24 — a five-month-old public data product.
- **Cross-object structure in residual space.** `abs:"self-replicating" AND
  abs:"solar system" AND abs:"technosignature"` → 0. `all:"self-replicating
  spacecraft"` → 0. `all:"Bracewell probe"` → 0. The only paper on
  self-replicating-probe technosignatures is Ellery (arXiv:2510.00082), which is
  theory: 69 mentions of "self-replicating", 139 of "lunar", **zero** of
  "non-gravitational", "Yarkovsky", "ephemeris residual", "LSST" or
  "asteroid family". **The replication axis has never been turned into an
  observational test of any kind, in any dataset.**
- **SFD shape and albedo homogeneity as an artificiality discriminant.**
  `abs:"manufactured" AND abs:"population" AND abs:"asteroid"` → 0.
  `abs:"asteroid family" AND abs:"artificial"` → three hits, all ML taxonomy. The
  natural-science machinery is mature (1705.10903, 2009.04489, 1802.01783) and
  nobody has proposed using it this way.
- **Rubin SSO alerts for technosignature work.** Zero occurrences of
  "technosignature" in every Rubin data-products document captured: DMTN-087, the
  DP0.3 DPDD, the SSP pipeline page, `sdm-schemas` APDB and DP0.3, the SSSC home
  page, RTN-011, DMTN-337. The SSSC roadmap (arXiv:1802.01783) does claim
  "detection and characterization of the non-gravitational forces … acting on
  NEOs" and "homogeneity of collisional families at small sizes" — with zero
  mentions of technosignatures.

### The case against, which is substantial and is why the design is what it is

1. **The `mpc_orbits` non-grav path cannot support a population.** Measured
   against JPL's SBDB on 2026-07-30: **589** asteroids have a fitted `A2` (544 at
   >3σ), **22** have `A1`, **11** have `A3`, out of 1,553,300. A "family sharing
   anomalous non-gravitational acceleration" would have to be found among 589
   objects, nearly all NEOs whose `A2` is Yarkovsky-consistent by construction.
   *Consequence for the design:* Path A is a cross-check, not the channel. Path B
   — per-detection `ephOffset` — is the channel, and `loom-probe` measures the
   mirror's null fractions before anything is built on either.
2. **This repository has already run the catalogue-scale single-object version and
   got a clean null.** `results/derelict/`: an `A1` census over 1,553,263 asteroids
   and 4,069 comets, verdict `CONSTRAINT_COMPLETE` / `ALL_SURVIVORS_EXPLAINED`,
   four survivors all traced (three `ARTIFICIAL_HUMAN_SUSPECT`, one
   `SHORT_ARC_ARTEFACT`). Per `CLAUDE.md` a clean null is a reason to change the
   *question*, so the question moves to `ephOffset` and to the population, and
   does not stay on `A1`/`A2`/`A3`.
3. **`ephOffset` is dominated by orbit-fit error, not physics.** Addressed in
   §3.2: the channel keys on geometry, time structure and quality-independence,
   never on amplitude.
4. **Ellery's own argument cuts against the asteroid version.** arXiv:2510.00082
   concludes that "evidence of asteroidal processing will be difficult to discern
   from natural processes given the constraints imposed by self-replication", and
   pivots to lunar Th-232/Nd-144 isotope ratios. He never considers the dynamical
   channel, so the objection is not fatal — but it is a published, on-point
   pessimistic prior and it is engaged here rather than ignored.
5. **Linkage pathology mimics the signal.** arXiv:2410.06874. Addressed in §3.4;
   note that the derelict channel's survivor base rate for this failure mode was
   100%.
6. **Trial counting.** Over 5 M objects, "a cluster of ≥N objects sharing anomalous
   residuals" arises by chance. The null resamples matched subsets of the same
   screened population and the resolution guard refuses to report below the
   randomisation floor.

**Bottom line.** The proposal as literally stated — a population sharing anomalous
non-gravitational *parameters* — is bounded to ~589 objects, already nulled
in-house, and occupied in flight. It is rescued, and becomes genuinely novel, by
moving the observable to `ssSource.ephOffset*` as a per-detection population
statistic and the question from "which object is anomalous" to "does the *set* of
anomalies have structure no natural population produces". Angles 3, 4 and 7 — the
replication axis, the artificial-family axis, and Rubin SSO alerts for
technosignature work — are unoccupied on this evidence. Angles 1, 2, 5 and 6 are
proposal- or execution-occupied and are cited and differentiated above.

---

## 7. Running it

```
python -m seti.cli loom-probe          # stage 0, runner-only: THE GO/NO-GO
python -m seti.cli loom-screen         # stage 1, runner-only
python -m seti.cli loom-assess         # stage 2, offline: the decision
```

Workflows: `.github/workflows/loom-probe.yml` (dispatch) and
`.github/workflows/loom.yml` (weekly, `40 15 * * 1` = **11:40 ET Mondays**;
deliberately offset from TOCSIN's 10:10 ET so the two do not hit the same TAP
service concurrently). Weekly rather than nightly because the parent population is
a catalogue and orbit solutions do not change from one night to the next — what
accumulates is arc length.

### What `loom-probe` must settle before anything else

1. **Are the non-gravitational columns populated?** `null_fractions` counts
   server-side. If `yarkovsky` and `srp` are all-NULL, Path A is dead in this
   mirror and `screen_orbits` writes `NONGRAV_COLUMNS_EMPTY` — a dead path, not a
   null result.
2. **What unit are the `ephoffset*` columns in?** The measured distribution
   decides: ~0.1 means arcsec, ~1e2 means mas, ~1e-6 means radians.
3. **Which key joins `lsst_ss_detection` to `detection`?** `ssSource` carries no
   epoch, so the join is required; both candidates
   (`detection.measurement_id = lsst_ss_detection.diasourceid`, and on the object
   id) are tried and the working one is applied through config, not code.
4. **What is `sid` for solar-system rows?** TOCSIN measured `sid=1` → LSST
   diaObject and `sid=0` → ZTF; the ssObject value is not measured, so the config
   default is `null`, which drops the clause. A wrong guess here returns an empty
   result rather than an error — the failure mode that cost TOCSIN a week.
5. **What are `lsst_ss_object`'s real column names?** Asked, not guessed.

Every query in the probe runs independently with its error captured, and the
record is written after each one, so a job timeout part-way through leaves every
answer already obtained on disk.

### Wall-clock discipline

A cancelled GitHub Actions job never runs its commit step, so a run that overshoots
loses everything it learned — this cost TOCSIN a three-hour backfill. `loom-screen`
takes a budget (default 5400 s inside a 180-minute job), yields voluntarily, and
records in `notes` exactly how many shortlisted objects were *not* screened so they
are never counted as trials.

---

## 8. Files

| file | what |
|---|---|
| `src/seti/loom/nongrav.py` | the anomaly boundary: momentum ceiling, AMR/β conversions, `A2 ↔ da/dt`, the empirical envelope, fit-quality gates |
| `src/seti/loom/residuals.py` | per-detection residual analysis: geometry, apparition trend, drift fit, law discrimination, breakpoint scan, sky coherence, quality independence, timing degeneracy |
| `src/seti/loom/replication.py` | the population tests, mis-linkage collapse, and the orchestrator |
| `src/seti/loom/control.py` | the three control sets and `validate` |
| `src/seti/loom/screen.py` | `Thresholds`, per-object screening, the tier ladder |
| `src/seti/loom/acquire.py` | ALeRCE TAP solar-system queries (runner-only) |
| `src/seti/loom/run.py` | the three stages |
| `config/loom.yaml` | every selection number, each traced to its source |
| `tests/test_loom.py` | 64 offline tests |
| `results/vnprobelit*/` | the prior-art evidence base |

---

## 9. Related channels

- **DERELICT** (`docs/derelict.md`) — ran the catalogue-scale `A1` census that
  nulled the single-object version of this question. LOOM is the change of
  question, not a refinement of that search.
- **TOCSIN** (`docs/tocsin.md`) — the Rubin *stellar* alert screen. Shares the
  ALeRCE TAP client, the wall-clock and checkpointing discipline, the
  matched-null population machinery, and the lesson about stratifying on
  observation quality.
- **COMPASS** (`docs/next-question.md`) — the axial-statistics treatment of
  orbital orientation that `pole_coherence` transposes from binary orbits to
  heliocentric ones.
- **Signature S29** (`docs/necrosignatures.md`), "monuments at stable points" —
  what `resonance_concentration` tests.
