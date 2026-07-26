# NECROSIGNATURES — technosignatures of technological extinction

**The inversion.** Every executed SETI search to date looks for a civilization
that is *working*: a beacon transmitting, a swarm radiating waste heat, an
industry polluting. If the Great Filter lies ahead of us, those searches are
tuned to the wrong target. What the galaxy would then contain is not
civilizations but **residue** — and residue has different observables, several
of which are the *photometric negative* of what current searches require.

This document defines the signature taxonomy, maps each near-term ASI failure
mode onto its astronomical residue, and specifies the detection design for the
channels this repository will execute.

The organising claim is falsifiable and specific:

> A civilization that stops leaves signatures that are **cessations, deficits,
> and sparse anomalies** — not excesses. Every published waste-heat search
> requires a positive infrared excess. A structure that is cold, shut down, or
> thermodynamically efficient produces **attenuation with no excess**, and is
> therefore invisible to all of them by construction.

---

## 1. ASI failure modes → astronomical residue

The user's emphasis is on the extinction modes humanity may plausibly face
soon. Each has a distinct observational fingerprint.

| Failure mode | What ends | Residue | Channel |
|---|---|---|---|
| Misaligned optimizer (resource maximizer) | biology; machines expand | expansion front with an edge; disassembled planets; refractory pollution of the host star | TIDEMARK, TAILINGS |
| AI-enabled bio/nuclear catastrophe | biology only | orphaned industrial atmosphere — technosignature *without* biosignature; radionuclides | EXHAUST, (MIDDEN) |
| Gradual disempowerment / value lock-in / wireheading | growth, not matter | **waste heat that switched off**; artifacts persist but go quiet | EMBER, VIGIL |
| Efficient or reversible computing | nothing visible | **matter without energy**: attenuation with no re-radiation | CENOTAPH |
| Aestivation (deliberate dormancy until the universe cools) | activity, by choice | cold hoarded matter; grey dimming; far-IR-only re-emission | CENOTAPH |
| Autonomous-weapons mutual destruction | everything, correlated | spatially/temporally clustered extinctions | TIDEMARK |
| Upload and departure | presence | abandoned system; unmaintained structures grinding down | OSSUARY, DERELICT |

Two of these — "the heat turned off" and "matter that absorbs but does not
re-radiate where anyone has looked" — are structurally undetectable by the
entire existing Dysonian corpus.

---

## 2. Signature taxonomy

Organised by *what the residue physically is*, not by wavelength.

### I. Thermal residue — waste heat and its cessation
* **S1 Extinguished waste heat.** An infrared excess present in IRAS (1983) /
  AKARI (2006) and absent in WISE (2010) / NEOWISE (2025). → **EMBER**
* **S2 Grey dimming.** Achromatic flux deficit with no reddening and no
  infrared excess. → **CENOTAPH**
* **S3 Bolometric budget deficit.** Full FUV→22 µm SED integration: a star
  radiating measurably less than its radius and T_eff demand. → CENOTAPH stage 3
* **S4 Duty-cycled waste heat.** Mid-IR variable, optically constant — thermal
  load tracking computational load; and the cessation of that variability.
  → **VIGIL**
* **S5 Isothermal excess.** Engineered radiators are single-temperature; dust
  spans a radial temperature range.
* **S6 Matrioshka step structure.** Discrete, quantised temperature components
  rather than a continuous gradient.

### II. Structural residue — the ruins
* **S7 Kessler cascade.** Un-station-kept swarms grind to dust. Warm debris
  around stars that cannot make it: metal-poor, halo-kinematic, ≥10 Gyr.
  → **OSSUARY**
* **S8 Refined-material dust.** Metallic/graphitic/anomalously crystalline
  mineralogy rather than amorphous silicate.
* **S9 Unmaintained decay.** Photometric irregularity secularly *increasing*.
  → **RUST** — *built and dispatched 2026-07-26 (`docs/rust.md`).*
  *(Corrected 2026-07-26: this was wrongly marked as covered by the existing
  `dimming` channel. It is not — `dimming/secular.py` fits a weighted linear
  trend to season medians, i.e. a trend in **brightness**. A secular rise in
  aperiodic variability **amplitude** is a different statistic and has been run
  neither here nor, as far as the sweeps could establish, anywhere.)*
  The channel's whole difficulty is one systematic: robust scatter estimators
  are biased low at small N, epochs-per-season is set by survey cadence, and
  cadence trends with calendar time — so an uncorrected version of this search
  measures ZTF's operations calendar rather than astrophysics. See
  `docs/rust.md` §5 for the five-layer correction and its measured false-positive
  rate. Sensitivity is bounded honestly: RUST sees **many-element swarms whose
  collisional time `P/f` falls inside the survey decade**, and is blind to the
  passively stable architectures of McInnes 2026 (arXiv:2603.00203) — which need
  no upkeep and therefore never decay — and largely blind to Wright 2020's
  (arXiv:2006.16734) monolithic spheres, whose instability is catastrophic
  rather than a decade-long ramp.
* **S10 Tumbling ruin.** Deep, grey, non-repeating, non-Keplerian occultations.
  *(covered by `dimming`)*
* **S11 Orphaned occulters** around white dwarfs and neutron stars — hosts that
  outlive their makers. *(covered and resolved by the `science` WD channel)*

### III. Chemical tombstones
* **S12 Sparse abundance anomaly.** Natural abundance space is low-dimensional
  and moves in nucleosynthetic *families*; refining moves **one element**.
  Restricted to convective-envelope cool dwarfs, where diffusion and radiative
  levitation — the natural sources of single-element peculiarity — are
  suppressed. → **TAILINGS**
* **S13 Short-lived radionuclides** (Tc, Pm, actinides) in unevolved stars.
  *(covered by the existing `midden` channel)*
* **S14 Fission-product pattern** in a photosphere. *(covered by `midden`)*
* **S15 Twin-pair refractory excess** beyond any engulfed-planet mass budget.
  → TAILINGS stage 4
* **S16 Orphaned industrial gases + decay clock.** CF₄ (τ≈50 kyr) : SF₆ (≈3 kyr)
  : NF₃ (≈500 yr) : CFC-12 (≈100 yr) decay at known, different rates — their
  **ratios date the death**. The ratio set is a genuine chronometer with 10²–10⁴
  yr dynamic range, self-calibrating (ratios cancel the unknown emission scale),
  and it resolves an ambiguity the literature explicitly poses and leaves open:
  the 2026 review calls PFCs an "extinct technosignature… persisting long after a
  civilization might have **vanished or transitioned to cleaner technology**" —
  the clock separates those two cases.
  **NOT SEARCHED — killed on feasibility, honestly.** At Earth's actual
  industrial abundances every one of these gases fails to reach slant optical
  depth 1 on a TRAPPIST-1-class planet: the signal is not weak, it is absent.
  Detection needs a >15 m space-based mid-IR spectrograph (LIFE/HWO class), not
  JWST. Compounding it, NF₃ and CFC-12 both absorb near 10.8 µm and are not
  separable by band centre at all. The nearest non-hopeless variant is a
  transiting planet around a **white dwarf** (~1,170 ppm per scale height vs
  15.7 for TRAPPIST-1e, and a WD habitable zone is *by construction* a
  post-catastrophe environment) — but no such system with adequate archival
  spectra exists. Documented, not built.
* **S17 Nuclear-war atmospheric scars** — NOx-driven ozone destruction, soot.
* **S18 Artificial molecules in the ISM.**

### IV. Dynamical residue
* **S19 Dead lightsails.** Large non-gravitational acceleration with **no
  outgassing** — a high area-to-mass thin film. → **DERELICT**
* **S20 Anomalous interstellar objects** — v∞ anomalously low relative to the
  LSR (a decelerated object), extreme albedo, extreme axis ratio. → DERELICT
* **S21 Relativistic launch residue.**
* **S22 Disassembled planets** — missing system mass with matching stellar
  refractory tailings. → TAILINGS

### V. Spatial residue — the front that stopped
* **S23 A sharp anomaly edge** in the Galactic disk. → **TIDEMARK** *(built
  2026-07-26; `docs/tidemark.md`. Three geometries — 1D, 3D spherical shells over
  a centre grid, and sky caps — each scored against a null that already contains
  the fitted smooth gradient, so a gradient cannot be reported as an edge.)*
* **S24 Anomaly clustering** beyond a matched null. *(engine exists in
  `cluster`; needs new anomaly axes — this program supplies them. Note
  clustering is **not** a front: TIDEMARK's gradient/edge statistics are a
  different question, and `cluster_run` now also writes the parent sample
  without which no rate test of any kind is possible.)*
* **S25 Anomaly rate vs stellar age** — a filter clock. → **TIDEMARK**
  *(`agerate.py`: flat / rising / saturating / turnover, with metallicity forced
  into the stratification because old = metal-poor = fewer planets would
  otherwise read as a filter clock.)*
* **S23a A large-scale rate *gradient*** — the discriminating observable nobody
  has measured. Ćirković & Bradbury 2006 predicts the **outer rim**, Wright et
  al. 2021 (RNAAS 5, 141) the **Galactic centre**, and Wright et al. 2014 **no
  coherent structure at all** (shear mixes any bubble on a rotation timescale).
  Three mutually exclusive published predictions, none ever tested. → TIDEMARK

### VI. Informational residue — deliberate tombstones
* **S26 A photospheric flag** — the cheapest galaxy-visible, Myr-durable marker
  a dying civilization could leave. *(the deliberate reading of S12)*
* **S27 Mathematically patterned transits.** *(adjacent to `dimming`)*
* **S28 Structure in timing series.** **S29 Monuments at stable points.**

### VII. Event residue — the moment of the filter
* **S30 "Singularity flash"** — an unclassified blackbody transient on a
  catalogued nearby dwarf, matching neither flare, nova, nor microlensing.
* **S31 Vanished radio sources** with stellar counterparts (NVSS/FIRST→VLASS).
  → **SHROUD**
* **S32 Clocks that stopped** — periodicity that ceased. Eclipsing binaries do
  not stop. → **KNELL**
* **S33 Optically vanished sources *with* an infrared counterpart** = enshrouded,
  not destroyed. → SHROUD

### VIII. Exotic filter residues
* **S34 Anomalous compact objects** (physics accident). **S35 511 keV point
  sources** (antimatter leakage). **S36 Anomalous mass loss on cool dwarfs**
  (star-lifting residue). **S37 Vacuum-decay boundary** — listed for
  completeness; effectively unobservable until arrival.

---

## 3. CENOTAPH — the flagship design

*An empty tomb: a monument for a body that is elsewhere.*

### 3.1 The physical test is energy conservation

A swarm with geometric covering fraction `f` intercepts a fraction `f` of the
stellar luminosity. The observed flux is

```
F_λ = (1 − f) · F_λ,0 · 10^(−0.4 A_λ)
```

so a swarm imposes `Δm = −2.5 log₁₀(1 − f)` in **every band equally**
(f = 0.1 → 0.114 mag; f = 0.2 → 0.24 mag; f = 0.5 → 0.75 mag), whereas
interstellar dust imposes `R_b · E`, which varies by a factor ≈ 30 from the FUV
to W2. **Grey and chromatic attenuation are orthogonal vectors in colour
space** — so extinction need not be assumed zero, it is *fitted* and separated.

The intercepted energy must reappear. If it is not in WISE (3–22 µm), the
re-radiator is colder than ~80 K, hence further than ~10 AU, hence enormous —
and it must then appear in the far-IR (IRAS 60/100 µm, AKARI FIS, Herschel,
Planck). **If the energy appears nowhere from the FUV to 100 µm, the budget
does not close**, and that is the extraordinary case: energy beamed, stored, or
converted rather than thermalised. This is exactly the aestivating or
thermodynamically-efficient ASI, and it is the one case every excess-based
search discards by construction.

### 3.2 Estimator

The naive route (isochrone `M_pred` from spectroscopic T_eff, log g, [Fe/H]) is
limited by T_eff systematics: `dM_G/dT_eff ≈ −0.0022 mag/K` on the lower main
sequence, so a 100 K T_eff error alone injects 0.22 mag — comparable to the
signal at f = 0.2. Two better estimators, run in parallel:

1. **Spectral twins.** Two stars whose spectra match pixel-by-pixel have the
   same atmospheric parameters and therefore the same absolute magnitude to
   within the intrinsic spread. Pipeline systematics cancel differentially. The
   statistic is the target's `M_G` minus the median of its ≥50 nearest spectral
   twins.
2. **Data-driven spectrophotometric residual.** Train a map
   `spectrum → M_K` on the bulk of the survey and hunt the **underluminous
   tail** of the residuals. Unresolved binaries — the dominant confounder in
   luminosity work — scatter to the *overluminous* side, so the negative tail is
   intrinsically the cleaner one.

### 3.3 Staging

| Stage | Test | Kills |
|---|---|---|
| 1 | Underluminous vs spectral twins, ΔM > 3σ | pipeline scatter |
| 2 | Achromaticity: joint (grey `g`, reddening `E`) fit over GALEX FUV/NUV, Gaia G/BP/RP, 2MASS JHKs, WISE W1/W2 — require `g` significant with `E` consistent with the 3D dust map | interstellar dust, anomalous R_V |
| 3 | Energy budget: integrate the observed SED; locate the missing `f·L` | warm/cold dust re-radiation |
| 4 | Far-IR veto: IRAS/AKARI FIS/Herschel/Planck at the position | massive cold edge-on disks |
| 5 | Astrometric quality: RUWE < 1.2, parallax_over_error > 20, zero-point corrected | distance error, binaries |
| 6 | Photometric constancy in ZTF/ATLAS/Gaia | a star caught in a low state |

### 3.4 Sample and scale

Gaia DR3 × (LAMOST DR10 LRS ~11.8 M spectra, APOGEE DR17 733 k, GALAH DR4
~917 k), cut to dwarfs with `parallax_over_error > 20`. Expected working sample
**~3–5 × 10⁶ stars** — larger than any Dysonian search yet executed, and in a
signature regime disjoint from all of them.

### 3.5 The honest limitation

The floor is set by the empirical twin scatter, realistically 0.05–0.10 mag,
so the search is sensitive to `f ≳ 0.05–0.10`. Existing IR-excess searches
reach `f ~ 0.01` but **only for warm re-radiation (~100–1000 K)**. The two
regimes barely overlap: CENOTAPH owns cold and dark occulters, Hephaistos owns
warm ones. That complementarity, not a sensitivity claim, is the argument.

---

## 3A. The seven structural seams (literature sweep, 2026-07-26)

Eight parallel sweeps plus three runner-side fetch harnesses (884 arXiv records,
verbatim abstracts and full text under `results/necrolit/`, `results/seamlit/`,
`results/hephlit/`, `results/dysonlit/`, `results/disaplit*/`) established what
every executed waste-heat search shares. Each assumption leaves a class of
object undetectable:

1. **A positive infrared excess is required.** Hephaistos II demands a W3 *and*
   W4 detection at stage 2 — anything without both is gone before modelling.
   The sole published exception is Zackrisson et al. 2018 (see §3B).
2. **The temperature window is 100–1000 K**, set by W4/IRAS-LRS at 22–23 µm.
   Hephaistos I states it verbatim: *"Temperatures that are lower than 100 K
   would place this excess in the far-IR, whereas temperatures above 1000 K
   would shift it into the near-IR."* **Nothing stellar-scale exists below
   100 K.** The only sub-100 K constraint anywhere is Lacki 2016 via Planck
   100 GHz, at galaxy/cluster scale. **30–60 K at stellar scale is untouched,
   and no far-IR Dysonian search has ever been published** — Carrigan
   *deliberately discarded* IRAS 60/100 µm for cirrus and zodiacal light while
   noting an ideal instrument "would need to cover wavelengths from 3 to 100 µm."
3. **Single-epoch SEDs.** Every catalogue search treats the SED as static. There
   is no blind time-domain excess search at 12–25 µm, and NEOWISE structurally
   cannot supply one: **it carries only W1/W2; W3/W4 exist for the single 2010
   cryogenic epoch.** The only route to a decades-long baseline at the relevant
   wavelengths is IRAS 12/25 (1983) → AKARI IRC 9/18 (2006) → WISE W3/W4 (2010),
   and IRAS 12/25 vs WISE W3/W4 are near-identical bandpasses 27 years apart.
4. **Main-sequence, thin-disk, near-solar-metallicity hosts.** No search selects
   on kinematics, age, or metallicity. White dwarfs are explicitly pruned from
   both Hephaistos papers. *"No halo (Pop II) star has a confirmed warm IR
   excess."*
5. **Amplitude and colour, not shape.** Only Carrigan 2009 used SED shape
   (Planck fits to 92-point IRAS LRS spectra over 11,224 sources, 100 K floor),
   and his candidates all collapsed on a distance/luminosity degeneracy that
   Gaia now breaks. Hephaistos II's discriminant is an *unweighted* RMSE over
   the photometric vector — deliberately error-blind, and with **no shape
   feature that distinguishes a circumstellar blackbody from a redshifted dusty
   galaxy SED**, which is exactly why Hot DOGs sailed through.
6. **Background-galaxy confusion is the universal killer.** Of the seven
   Hephaistos candidates, **none survives cleanly**: D and E resolved by
   JWST/MIRI into a Hot DOG at z≈0.9 and a dusty starburst at z≈0.4, both
   within ~1″ (Zackrisson et al. 2026); A, B and G killed or badly damaged by
   radio counterparts; C by a near-IR companion; F marginal. WISE PSF FWHM is
   ≈6″ (W3) and ≈12″ (W4). **Any new infrared channel must build high-latitude
   selection and sub-arcsecond astrometric registration in as funnel stages,
   not as follow-up.** Note the series contains **no co-movement or
   astrometric-epoch test** — the gap `seti.contamination.comovement` already
   fills, now independently validated as the fix.
7. **Variability is rejected, not used.** Hephaistos II's `G_var > 2` cut
   explicitly "rejects potential Dyson swarms with very large absorbing
   elements."

### The two facts that make this program necessary rather than merely novel

**(i) Deeper infrared data is anti-correlated with colder sensitivity — a hard
instrumental ceiling.** Wien peaks: W1 3.4 µm → **852 K**; W2 4.6 µm → **630 K**;
W3 12 µm → **241 K**; W4 22 µm → **132 K**. NEOWISE-R, CatWISE2020 and the deep
unWISE coadds are **W1/W2 only**. **W3/W4 depth is frozen at the 2010 cryogenic
mission and cannot improve until a new mid-IR all-sky survey flies.** So the two
largest searches ever run — Contardo & Hogg's 4,898,812 stars (arXiv:2403.18941)
and anything built on CatWISE — are *structurally incapable* of detecting a
100–300 K Dyson sphere, and everyone who can reach cold temperatures is stuck at
AllWISE W4 depth, the shallowest and most confusion-limited band. **The infrared
route into the cold regime is closed by instrumentation, not by effort.** That is
the argument for attenuation (CENOTAPH) and far-IR recovery.

**(ii) The natural confounder sits exactly on top of the signal.** The
characteristic warm-debris-disk temperature locus is **~190 K** (Morales et al.
2011, ApJL 730, L29), directly inside the Dyson-sphere range every search
targets. Any positive-excess search at 130–250 K competes against an
astrophysical population that is *identical in two broadband colours*. And the
empirical false-positive rate confirms the consequence: **at most 7.9% ± 0.2% of
AllWISE-selected infrared excesses are good disk candidates** (Silverberg et al.
2018) — a ~92% false-positive rate, with the McDonald and Marton searches above
70% and *all* Theissen & West candidates at W4 S/N > 3 spurious. Amplitude and
colour cannot separate the populations because they overlap in exactly those
observables. **Shape can** (ISOTHERM), and **attenuation sidesteps the issue
entirely** (CENOTAPH).

Two population gaps follow, both unoccupied: **no Dyson search has targeted solar
analogues or K dwarfs as a defined population** (Hephaistos II was M dwarfs;
Contardo & Hogg is FGK but W1/W2-limited), and **no LAMOST stellar sample has
ever been used for a technosignature search** — despite Chinese groups having
already built the LAMOST+WISE IR-excess machinery.

## 3B. Honest novelty positions

* **CENOTAPH.** Zackrisson et al. 2018 (arXiv:1804.08351) is the one published
  attenuation-based search, and it **was executed** — on Gaia DR1 × RAVE DR5,
  by spectrophotometric-distance-vs-parallax discrepancy, explicitly with no IR
  excess required. Its one followed-up candidate, TYC 6111-1162-1, had "no
  detectable infrared excess" and resolved to an **unseen binary**. The method
  is theirs. This channel's contribution is (a) population scale, (b) the
  achromaticity veto they did not impose, (c) the far-IR recovery leg, and
  (d) sensitivity to partial covering fractions — their method needs f≈1. Their
  failure mode (unresolved binaries) is a mandatory funnel stage here.
  Ĝ II *rejected* the attenuation channel on the grounds that the starlight
  deficit is linear in α and "only observationally distinguishable for values of
  α near 1"; the achromaticity veto plus far-IR recovery is the answer to that.
  Blain 2024 (arXiv:2409.11447) names the blind spot outright but implements no
  search.
* **EMBER, ISOTHERM, OSSUARY, TIDEMARK, RUST, DERELICT.** Unoccupied as searches
  (see per-channel docs for the adjudication and citations).
* **Not built:** EXHAUST (S16) — killed on feasibility, see §2. S20 (ISO v∞
  anomaly) — Mamajek 2017 already found and explained 1I's low LSR-relative
  velocity, gravitational focusing makes low-v∞ objects *over*-represented so
  the anomaly is the expected outcome, and N=3 cannot support a claim.

## 4. Contamination discipline

Inherited from the repository ledger (`STATUS.md`), non-negotiable:

* A single-band anomaly is an artefact until confirmed in a second band.
* AllWISE W4 is unreliable for faint stars; W4-only excesses are cirrus.
* A negative W1−W2 is a blend, not a photosphere.
* Grey vs chromatic must be *measured*, not assumed — the existing `dimming`
  channel killed its best candidate on exactly this test (IR faded at 6% of the
  optical rate = ordinary dust; a grey occulter sits at ≳30%).
* Every candidate is traced to a systematic before it is believed.

## 5. Status

Channel specifications, novelty verdicts, and results are tracked per channel;
the live scoreboard is `STATUS.md`.
