# TAILINGS — the sparse chemical anomaly

**Claim being tested.** Natural stellar abundance space is *low-dimensional*.
Every star is a mixture of a small number of nucleosynthetic sources, and each
source moves an entire element **family** together — the alpha elements track
core-collapse supernovae, the Fe-peak tracks Type Ia, the s-process tracks AGB
pollution, the r-process tracks neutron-star mergers. Industrial refining does
the opposite: it is *defined* by separating **one element** from its chemical
neighbours.

So the artificial signature is a **sparse** anomaly — one or two elements
extreme, everything else normal — as against the **dense** anomaly that natural
chemical peculiarity and pipeline failure both produce. Sparsity, not
amplitude, is the discriminant.

Two readings of the same observable, and the search does not need to choose
between them:

* **tailings** — the residue of planetary disassembly dumped into the host
  star, because a star is the cheapest waste sink in a system; or
* **a tombstone** — a deliberate marker. A photospheric abundance flag is the
  cheapest galaxy-visible, Myr-durable thing a dying civilization could leave.
  (S26 in `docs/necrosignatures.md`; the channel realises S12, S15 and S22.)

---

## 1. Why cool dwarfs, and how strong the null actually is

Single-element peculiarity *does* occur in nature — in Ap, Am/Fm and HgMn
stars, where a radiative envelope lets atomic diffusion and radiative
levitation separate species faster than convection can re-mix them. That
mechanism has a sharp boundary: it needs a **thin surface convection zone**.
Once the envelope becomes massive enough, the mixing timescale beats the
diffusion timescale and the anomalies never appear at the surface. This is why
the Am/Fm phenomenon is confined to roughly A–early F stars and disappears on
the cool side.

Restricting to **G/K/M dwarfs (Teff < 6000 K, log g > 4.0)** therefore places
the search in the regime where the known production mechanism for a
single-element anomaly does not operate. The convective envelope mass runs from
~0.02 M⊙ at solar Teff to ~0.2–0.4 M⊙ in K/early-M dwarfs and to the whole star
below ~3500 K — three to four orders of magnitude more diluting material than
an A star has.

**The null is strong but it is not airtight, and the honest statement is that
it is a strong prior, not a theorem.** Known and suspected leaks, each of which
the funnel must handle rather than assume away:

| Leak | Is it sparse? | How it is handled |
|---|---|---|
| AGB-companion mass transfer (barium/CH dwarfs) | **No** — raises the whole s-process, Sr/Y/Zr *and* Ba/La/Ce, usually with C | family-coherence veto; the s-process is two families and both move |
| Planet engulfment / rocky accretion | **No** — raises every refractory along a condensation-temperature trend | family veto; and stage 4 tests it quantitatively |
| lambda Boo depletion | **No** — depletes all refractories at once; and it is a hot-star class | population cut plus family veto |
| Li depletion / Li enhancement | **Yes**, genuinely single-element | Li is **excluded by construction** from carrying a candidacy |
| C/N mixing, dredge-up | partially | C and N excluded; dwarfs only |
| NLTE / 3D modelling error in one species | **Yes**, and this is the serious one | per-element flag rates, per-field rates, cross-survey confirmation, and raw-spectrum re-measurement |
| Blends and line-list error at one line | **Yes** | element caveat table; re-measurement against Teff-matched peers |
| Unresolved binary contaminating one line region | sometimes | RUWE, RV scatter, vbroad |

The last three are why this channel's centre of gravity is the *vetting*, not
the statistic. A single-element catalogue outlier is, on the prior, a bad
measurement. The design earns its keep only if every one of those routes has a
discriminator that does not require trusting the catalogue.

---

## 2. Novelty adjudication

Runner-fetched evidence in `results/tailingslit/` (107/107 fetches successful,
run `30202628318`) plus `results/necrolit/`, `results/litcheck/` and
`results/przybylski_lit/`.

### 2.1 The real competitor: Huang, Tao & Zhang 2026 (arXiv:2605.29811)

*"A Calibrated Bayesian Search for Potential Chemical Technosignatures in
Polluted White Dwarfs"*, accepted ApJ, 21 pp. This is the only **executed**
chemical technosignature search on photospheric abundances anywhere in the
fetched corpus, and it is a serious piece of work. The distinction has to be
stated precisely, so here it is on five axes, from their full text:

| | Huang et al. 2026 | TAILINGS |
|---|---|---|
| **Population** | polluted white dwarfs only; acceptance window `7.7 ≤ log g ≤ 8.3`. "Main sequence" appears once in the paper, in the phrase "post–main-sequence evolution" | main-sequence G/K/M dwarfs, `log g > 4.0` — ~3.5 dex away in surface gravity |
| **Data** | PEWDD literature compilation: 2,223 Ca-referenced constraints over **697 records / ≥397 distinct objects**; "GALAH" and "APOGEE" appear zero times | GALAH DR4 + APOGEE DR17, ~10⁵–10⁶ objects, ~20–30 uniformly measured elements each |
| **Natural reference** | a 3-component Gaussian mixture over **3,493 laboratory meteorite whole-rock analyses** (chondrite / achondrite / other) | an **empirical stellar residual manifold**: each `[X/Fe]` regressed on ([Fe/H], Teff, log g, alpha proxy) over the survey itself. They never regress abundances on stellar parameters; stellar parameters enter their work only as inputs to a diffusion correction |
| **Alternative hypothesis** | a **fixed dense template** — the fiducial siderophile concentrate simultaneously sets Fe +2.36, Ni +1.17, Cr +0.79, Mn +0.47 up and Na −1.68, Ti −1.81 down. Every template in the paper moves many elements | **agnostic to which element** is anomalous, and requires that only one or two are |
| **Sparsity** | **the opposite sign.** "discrimination … typically requires ≳ 5 detected elements for decisive support"; power rises monotonically with the number of elements | one or two elements discrepant, ≥3 is an automatic rejection |

The two searches would rank the same object list in nearly opposite order.
Their headline diffusion-corrected candidate, GD 362, fires on a full
nine-element panel; under the TAILINGS rule a nine-element anomaly is a
rejection. Their highest-Bayes-factor records (G165−7, G157−35, WD 1202−232)
are one- and two-element records that they then explicitly disqualify.

**A terminological trap, and their strongest criticism.** In their paper the
word "sparse" always means *missing data* — few elements **measured** — never
"anomaly confined to few elements". And they show quantitatively that a record
with a sparse *panel* can produce a large Bayes factor while being
information-starved: *"records with sparse detected element panels can yield
large ln BF values while still being information-limited … sparse-panel
candidates are best interpreted as high-priority follow-up targets rather than
as robust classifications."*

That criticism is correct and it applies to any one-element claim. **The answer
is the reason this channel exists in surveys rather than in a literature
compilation.** Their information-starved records are cases where the other
elements were *never measured*. Here the other 20–30 elements are measured
**and quiet**, and it is exactly that information their archival corpus lacks.
The statistic `n_quiet` — the count of elements measured and inside 2σ — is
carried as a first-class quantity per candidate for precisely this reason, and
a candidate is only as strong as its `n_quiet`.

They also conclude that depth beats breadth: *"the most efficient path … is
deeper, broader multi-element measurements for a smaller number of well-chosen
targets, rather than simply increasing sample size with one–two-element
records."* That conclusion is derived for a fixed dense template under archival
censoring, where "breadth" means more one-element records. GALAH DR4 and
APOGEE DR17 break the premise: they deliver breadth *and* depth simultaneously,
uniformly measured. Their trade-off does not bind a survey that is not
censored.

**Verdict: not subsumed.** Different population, different data, different null
model, different alternative, and the sparsity term enters with the opposite
sign. Their own stated gaps — "expanded template families", "processed
materials that do not resemble the adopted template", "future datasets with
more uniform selection and broader element inventories" — are two-thirds of
what this channel occupies, and they never name main-sequence stars as a
direction. Their citation count is zero (queried 2026-07-26).

### 2.2 Whitmire & Wright 1980

*Icarus* **42**, 149–156, bibcode `1980Icar...42..149W`, DOI
`10.1016/0019-1035(80)90253-5` — "Nuclear waste spectrum as evidence of
technological extraterrestrial civilizations". Note the correct citation: it is
Whitmire **& Wright**, 1980, *Icarus*.

Their specific prediction: slow-neutron fission of ²³⁹Pu/²³³U makes **Pr and Nd**
the most overabundant products, and they restricted the host class to **A5–F2**
on convective-mixing grounds — a star convective enough to keep waste in the
photosphere but quiet enough to see weak lines.

The 46-year citation tree (56 citing works, OpenAlex, verified in
`results/przybylski_lit/openalex_ww1980_citedby.json` and re-fetched here)
contains **reviews and essays only** — Ćirković 2009, Carrigan 2010/2011,
Stevens, Forgan & O'Malley-James 2015, Haqq-Misra, Lacki's Exotica Catalog,
Wright's strategy papers, Perryman's handbook. Not one executed spectroscopic
or abundance survey. The 2026 flagship review (Vidal et al., arXiv:2605.21093,
118 pp) devotes exactly one paragraph to "Stellar Pollution", and it is a list
of proposals plus one contested Ap star.

**TAILINGS deliberately inverts their host-class argument.** They chose A5–F2
so the waste would *stay* in a thin photosphere. That is the same regime in
which diffusion and levitation manufacture single-element anomalies naturally —
which is why the one long-running claim in this space, Przybylski's star, is an
Ap star and has been contested for sixty years. Trading signal amplitude for a
clean null is the right trade when the null is the binding constraint, and here
it is: a 0.2 dex single-element excursion in a K dwarf is unexplainable in a
way that the same excursion in an Ap star simply is not.

The sibling channel `midden` searches for the **radionuclide lines themselves**
(Tc, U, Th, Pm) in high-resolution ESO spectra, which is the direct execution of
Whitmire & Wright. TAILINGS is disjoint: no line list, no isotope, no decay
clock — an abundance-space geometry test on survey catalogues. The two channels
share only the 1980 paper as an ancestor.

### 2.3 Abundance-space anomaly detection

The direct arXiv queries `all:"anomaly detection" AND all:APOGEE`,
`… AND all:GALAH`, `abs:"outlier detection" AND abs:"stellar abundances"`, and
`all:"abundance anomaly" AND all:"single element"` all return **zero results**.
Zero-result queries are narrow phrase searches and are evidence, not proof; but
they are consistent, and the positive evidence points the same way.

What *does* exist is the chemical-tagging literature, and every statistic in it
is a **global distance**: PCA and EMPCA (Ting et al. 2012; Price-Jones & Bovy
2018), functional PCA (Patil et al. 2022), t-SNE (Anders et al. 2018), k-means
in 15-D (Hogg et al. 2016), latent-factor models (Casey et al. 2019), graph
autoencoders (Quandt-Rodriguez et al. 2026), spectral similarity (de Mijolla &
Ness 2021). All of them are built to **cluster stars into birth groups**, not
to score an individual star's abundance vector for a single-element excursion —
and a reconstruction-error or full-vector-distance statistic is *maximised* by
dense anomalies and actively suppresses sparse ones. That is the structural gap
this channel occupies.

### 2.4 What the tagging literature supplies: the thresholds

The intrinsic star-to-star scatter within a birth cluster, which sets the floor
below which "one element differs" is not a meaningful statement:

* Bovy 2016 (M67, NGC 6819, NGC 2420; 15 elements): <0.01 dex for C and Fe,
  ≲0.015 for N, O, Mg, Si, Ni; ≲0.02 for Al, Ca, Mn; ≲0.03 for Na, S, K, Ti, V.
* Cheng et al. 2020 (17 tagged birth clusters): ≲0.02 dex for C; ≲0.03 for O,
  Mn, Fe; ≲0.04 for Si, Ni; ≲0.05 for N, Mg, Ca.
* Patil et al. 2022 (M67): Fe ≲0.02, C ≲0.03, O/Mg/Si/Ni ≲0.04, Ca ≲0.05.
* Casamiquela et al. 2021: internal coherence "typically 0.03 dex".
* Ness et al. 2018: at 0.03 dex precision, ~0.3% of *unrelated* field-star
  pairs are already indistinguishable (~1.0% at fixed solar [Fe/H]).

Dimensionality, which is the quantitative form of the "families" claim:
~8–9 independent dimensions (Ting et al. 2012), ≲10 principal components
(Price-Jones & Bovy 2018), ~10 functional PCs (Patil et al. 2022), 6 latent
factors at N=2,566 (Casey et al. 2019).

So a 6σ excursion on an empirical 0.03–0.05 dex width is a 0.2–0.3 dex
single-element event: an order of magnitude above the chemical individuality of
co-natal stars.

**A caution that cuts against APOGEE.** Manea et al. 2025 followed up 25 APOGEE
DR17 doppelgänger pairs at R~60,000 and found neutron-capture differences of
0.02–0.38 dex despite near-identical H-band abundances at SNR>300. The H band
carries one n-capture element (Ce). So a *non*-confirmation in APOGEE for an
n-capture element is weak evidence and is recorded as `not_covered`, never as a
refutation.

---

## 3. Method

### 3.1 Sample (`acquire.py`)

Cool dwarfs with abundances and clean flags: `Teff < 6000 K`, `Teff > 3000 K`,
`log g > 4.0`, `SNR > 40`. Sources are an **ordered registry** per survey with
recorded provenance — GALAH DR4 preferred, GALAH DR3 as the certain fallback;
APOGEE DR17 via VizieR with the SDSS SAS file behind it; LAMOST MRS third.
Whichever answered is stated in `provenance.json` and in the report, so the
numbers are never attributed to the release that was merely intended.

Pulls are **chunked in Teff** (async TAP first, sync fallback), because a
monolithic query above ~10⁵ rows times out or truncates silently. Two failure
modes are recorded as first-class degradation: a chunk that returns exactly its
row cap marks the sample `TRUNCATED`, and a missing chunk marks the temperature
coverage incomplete. Columns are resolved by pattern, so the GALAH
(`mg_fe`/`e_mg_fe`/`flag_mg_fe`), APOGEE (`MG_FE`/`MG_FE_ERR`/`MG_FE_FLAG`) and
VizieR (`__Mg_Fe_`/`e__Mg_Fe_`/`f__Mg_Fe_`) conventions all reduce to the same
canonical table and a schema change costs nothing. Per-element pipeline flags
are honoured **before** the manifold is fitted, so known-bad values stay out of
both the reference surface and the candidate list.

### 3.2 The natural manifold (`manifold.py`)

For each element X, fit

```
[X/Fe] = f([Fe/H], Teff, log g, alpha_proxy) + r_X
```

with `f` a quadratic polynomial fitted by iteratively reweighted least squares
with 4σ clipping over 4 iterations. Two design points matter:

* **The alpha proxy is leave-one-out.** When fitting Mg, the proxy is built
  from the other alpha elements. Otherwise an element partly predicts itself
  and its residual is artificially crushed. A useful consequence: a *coherent*
  shift of the whole alpha family is absorbed by its own predictor, so global
  alpha offsets — chemical evolution, not a refinery — cannot reach the
  candidate list by any route.
* **Clipping is not cosmetic.** The manifold is fitted on the same stars it
  then tests. Without clipping a genuine 6–8σ anomaly drags the surface toward
  itself and partially hides.

`Teff` and `log g` are in the regression because a spectroscopic `[X/Fe]` is a
*fitted* quantity whose systematic error is a smooth function of the
atmosphere. Regressing them out means a candidate cannot be manufactured by
sitting at an unusual Teff.

### 3.3 The denominator is measured, not assumed

Catalogue abundance uncertainties are formal fit errors and are routinely too
small — they do not know about line-list error, unresolved blends or continuum
systematics. Using them to define a 6σ outlier would manufacture candidates by
the thousand. Instead:

```
sigma_X(SNR, Teff)  =  robust (MAD) width of r_X in bins of (SNR, Teff)
sigma_used          =  max( sigma_X(SNR, Teff),  sigma_reported )
```

Thin cells fall back to the Teff-marginal then the global width — a
sparsely-populated cell must never supply an optimistically small denominator.
The reported error takes over when a particular star's fit was unusually bad.

A consequence worth stating: the threshold **in dex** is element-dependent by
construction. Ba/La/Ce carry a real astrophysical s-process spread that
([Fe/H], Teff, log g, alpha) cannot predict, so their empirical width is
several times the Fe-peak's. A fixed dex threshold would flag n-capture
elements preferentially, which is exactly the systematic that would manufacture
a fake candidate population.

### 3.4 The sparse statistic (`sparse.py`)

Per star, from the standardised residual vector:

* `z_max`, `element_max`, `z_second` — the amplitude and its carrier;
* `n_discrepant` = #{|z| ≥ 6}, `n_active` = #{|z| ≥ 2}, `n_quiet` = the rest;
* `z_rest_rms` — RMS over every element *except the largest*: the density
  diagnostic;
* `z_background_rms` — RMS outside the discrepant set;
* `contrast = z_max / max(z_background_rms, 1)`.

Classification, rules applied **in order** so the rejection reason stays
physically meaningful:

1. fewer than 12 measured elements → `INSUFFICIENT`;
2. nothing at 6σ → `NORMAL`;
3. ≥3 elements discrepant → `DENSE` ("a family/global event");
4. more than one background element above 2σ → `DENSE` ("the rest of the vector
   is not quiet");
5. the flagged element's nucleosynthetic siblings have mean |z| ≥ 2 → `DENSE`
   ("family co-moves");
6. contrast < 3 → `DENSE`;
7. otherwise `SPARSE`.

Li, Be, B, C and N are **excluded from carrying a candidacy**: they are known
natural single-element variables, and a known one cannot be evidence for an
unknown one. They are still measured and reported, because a Li excess is the
classic engulfment tracer and is diagnostic in the opposite direction.

**The headline output is the contrast table** — the sparse/dense split binned
in `z_max`, with the median `z_rest_rms` in each bin. That two-dimensional
distribution is what makes the claim falsifiable rather than a threshold
choice. If the SPARSE fraction is flat with `z_max`, the sample is noise plus
systematics. A real population appears as a SPARSE excess that *survives* to
high `z_max` while DENSE dominates at moderate `z_max`, where genuine chemical
peculiarity lives.

Six sigma is **not a p-value** and must never be quoted as one: the residual
distribution is not Gaussian in the tails. The false-positive control is
entirely empirical and lives in the next section.

### 3.5 Vetting (`vet.py`)

| systematic | why it mimics | discriminator |
|---|---|---|
| low SNR | heavy tails, not just a wider core | sigma measured *in SNR bins*, plus a hard SNR ≥ 40 floor |
| bad spectral fit | a locally bad fit moves one element | pipeline χ², global and per-element flags |
| fast rotation | blended lines, meaningless abundances | vbroad/vsini ≤ 15 km/s |
| unresolved binary | a second spectrum with different parameters — the failure mode that killed the nearest prior attenuation search's one candidate | RUWE ≤ 1.4, RV scatter ≤ 1 km/s |
| a systematic in one element | line list, not Galaxy | per-element flag rate; >2% of the sample is a veto |
| a systematic in one observation | shared calibration within a field/plate | per-field flag rate >5× global is a veto — the abundance-space form of "a feature recurring across unrelated sightlines is instrumental" |
| a known-difficult line | telluric overlap, hyperfine structure, severe NLTE, one weak line only | per-survey element caveat table: **demotes**, does not delete, and the caveat travels with the candidate |
| duplicate rows | inflated counts | dedupe on survey ID, keep highest SNR |
| single-survey anomaly | one line list, one wavelength region, one pipeline | cross-survey confirmation where covered; absence of coverage recorded as `not_covered`, never as refutation |

**The decisive step is re-measurement.** A catalogue-level survivor is a
*target*, not a detection. `measure_ew` re-measures the specific line from the
raw spectrum with a robust local continuum (deg-2 polynomial over ±3 Å, core
excluded, asymmetric clipping so absorption cannot drag the fit down and
manufacture a line), and `census_z` ranks it against Teff-matched peers
observed with the same instrument. That comparison is self-calibrating: blends,
telluric residuals and blaze structure common to the temperature slice cancel
identically, and no absolute spectral synthesis is required — which is what
makes it independent of the pipeline under test. This is the same standard
`midden` adopts from Andrievsky et al. 2023.

### 3.6 Stage 4 — co-natal wide binaries (`twins.py`)

Two stars in a Gaia wide binary formed from the same material at the same time,
so a *differential* abundance is not chemical evolution, it is what happened to
one of them afterwards — and the differential analysis cancels most pipeline
systematics. Pairs come from El-Badry, Rix & Heintz 2021 with the standard
`R_chance_align < 0.1` purity cut: a chance alignment of two unrelated stars
has no reason to share a composition and would manufacture exactly the
signature being sought.

Engulfment is the strongest natural competitor, and it is real: Liu et al. 2024
(Nature) found ≥7 new ingestion instances among 91 co-natal pairs, an ~8%
occurrence rate. So the stage tests it two independent ways.

**Test A — the mass budget.** Dissolving `M_rock` of rock into a convective
envelope of mass `M_cz` gives

```
d[X/H] = log10( 1 + M_rock · f_X^rock / (M_cz · Z_X,⊙ · 10^[Fe/H]) )
```

Calibration: one Earth mass into the solar convective zone (0.021 M⊙) gives
0.015 dex in [Fe/H], the canonical number. Inverting turns an observed
differential into an implied engulfed rocky mass. The budget ceiling is **100
Earth masses** — about 3.5× the largest engulfment ever inferred (Kronos/Krios,
HD 240430/240429, ~0.20–0.23 dex differential, ~28 M⊕, described as
exceptionally large) and above the ~10–30 M⊕ core at which runaway gas
accretion turns a rocky body into a gas giant. More rock than this is outside
every planet-formation model as well as every observation.

Every approximation runs one way. `M_cz` is scaled *down* by a safety factor of
0.5 by default, which makes the pollutant less diluted, the implied mass
smaller, and the "unexplainable" verdict harder to reach. Thermohaline mixing
after accretion dilutes a real signature further, so the true required mass is
larger than computed. A pair only clears the bar under the most
engulfment-friendly assumptions.

**Test B — composition, independent of mass.** Rock is not a single element.
Any engulfment moves Fe, Mg, Si, Ni, Ca, Al, Cr, Ti together in fixed
proportion along a condensation-temperature trend. A pair whose difference is
**one element with the rest identical** is not engulfment *at any mass*. This is
the same sparsity argument applied differentially, and it is the stronger test
because it does not depend on a convective-envelope model at all.

Verdicts: `NO_DIFFERENCE`, `ENGULFMENT_CONSISTENT`, `ENGULFMENT_EXCESSIVE`
(a Tcond trend needing more rock than the budget), `SPARSE_UNEXPLAINABLE`
(one element, no Tcond trend), `DENSE_NOT_ENGULFMENT` (many elements, no
positive Tcond trend — check co-natality and the pipeline before anything
else), `INSUFFICIENT`.

**Honest limits.** The `M_cz(Teff)` tabulation is coarse — good to a factor of
~2 in F–G, worse for M dwarfs; the safety factor exists because of that. Rock
composition is taken as bulk Earth; a differentiated core-rich fragment is more
siderophile-rich and would produce a *different* pattern — which is exactly the
hypothesis Huang et al. test in white dwarfs, and it is a dense ≥5-element
template, so it is caught by the family and Tcond tests here rather than being
confused with a sparse anomaly.

---

## 4. Interpretation ladder

1. **Instrumental / systematic** — element or field flag rate elevated,
   caveated line, re-measurement refutes: vetoed, feeds the contamination
   ledger.
2. **Known natural** — barium dwarf, engulfment, unresolved binary in disguise:
   reclassified; interesting only as a catalogue correction.
3. **New stellar astrophysics** — a genuine, re-measured, cross-confirmed
   single-element photospheric anomaly in a convective-envelope dwarf, with the
   whole rest of the abundance vector quiet. That is publishable astrophysics
   regardless of interpretation, because no known mechanism makes it. This is
   the "remarkable either way" rung.
4. **TAILINGS candidate** — rung 3 plus a co-natal companion showing the same
   sparse differential, or a refractory difference beyond the engulfed-planet
   budget. Follow-up: high-resolution differential spectroscopy of the pair,
   isotope-sensitive features for the flagged element, and a search for the
   system's missing mass (S22).

---

## 5. No-null rule (CLAUDE.md)

An empty candidate list at these thresholds is a statement about **this corpus,
these elements and these thresholds** — not a publishable null, and no
occurrence limit will be written up. The escalation path is:

* **more elements** — the optical n-capture lines the H band cannot reach
  (Manea et al. 2025 showed APOGEE doppelgängers hide 0.02–0.38 dex n-capture
  differences), i.e. GALAH and LAMOST MRS over APOGEE;
* **a second survey** for cross-confirmation of anything marginal;
* **the differential co-natal channel**, which reaches 0.01–0.02 dex where the
  field channel reaches 0.03–0.05 — a factor of ~3 in sensitivity for the price
  of a much smaller sample;
* **lower thresholds with a matched null**, if and only if the contrast table
  shows the SPARSE fraction rising with `z_max` rather than flat.

The question changes; the write-up waits for a detection.
