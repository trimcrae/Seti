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
* **S9 Unmaintained decay.** Photometric irregularity secularly increasing.
  *(covered and exhausted by the existing `dimming` channel)*
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
  **ratios date the death**. Diagnostic pairing: industrial gases present,
  biosignature disequilibrium absent. → **EXHAUST**
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
* **S23 A sharp anomaly edge** in the Galactic disk. → **TIDEMARK**
* **S24 Anomaly clustering** beyond a matched null. *(engine exists in
  `cluster`; needs new anomaly axes — this program supplies them)*
* **S25 Anomaly rate vs stellar age** — a filter clock. → TIDEMARK

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
