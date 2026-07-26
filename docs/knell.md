# KNELL — the clock that stopped

**Signature S32** (`docs/necrosignatures.md`): *periodicity that ceased.*

---

## 1. The claim

A **periodic** signal that ceased. Eclipsing binaries do not stop; pulsators do
not stop. A clock that stops is the cleanest "the mechanism ended" observable
available in the optical time domain, and it has one property no other
necrosignature has:

> **A cessation cannot be produced by intervening dust.** Obscuration changes a
> signal's *amplitude*, not the *existence* of a period. Every other channel in
> this program — dimming, fading, IR excess appearing or vanishing — has to
> fight line-of-sight extinction as its first-order confounder. This one does
> not, because extinction is not a mechanism that can delete a frequency.

What replaces dust as the dominant confounder is not astrophysics at all. It is
**survey-dependent detectability**, and §3 is the whole channel.

---

## 2. Novelty

**Verdict: the search is unoccupied, but the claim has to be stated narrowly and
one substrate has to be abandoned.**

### 2.1 What is genuinely absent from the record

Enormous *discovery* catalogues exist — ASAS-SN, ZTF, ATLAS, Gaia DR3, GCVS,
VSX — all built to find variables. **None was ever run backwards.** No paper
takes a catalogue of confirmed periodic variables and asks, at fixed detection
sensitivity, *which of these stopped?*

The strongest evidence is negative space in a 2024/2025 publication. Kurtz et
al., **"HD 60435: The star that stopped pulsating"** (arXiv:2412.04840, MNRAS
536, 2103, 2025), claim a **first** — that no pulsating star had previously been
observed to cease pulsating entirely. A claim of *first ever*, made in a
refereed journal in 2025, is not sustainable if a systematic ex-variable search
existed anywhere.

### 2.2 The claim this channel is entitled to make

**Not** "the first search for stars that stopped varying" — that is falsifiable
inside eclipsing binaries (see §2.4). The defensible statement is:

> Cessation events have only ever been found **one at a time and
> serendipitously**. No survey has ever **measured the rate**, at fixed
> detection sensitivity, across a variable-star population.

### 2.3 The prior art, ranked by how close it comes

| Work | Scale | Why it is not this |
|---|---|---|
| **Juryšek et al. 2018** (arXiv:1709.08087, A&A 609, A46) | 32,259 OGLE-III EBs → 58 compact-triple candidates | Fits eclipse-depth **trends**, so it structurally cannot find systems that ceased mid-baseline or before it. **The closest study extant, and its blind spot is exactly this target.** |
| **Borkovits et al. 2025** (arXiv:2502.09480, Kepler→TESS) | ~200–250 Kepler triples | Reports ~28 systems unresolvable *because the eclipses had disappeared*. Cessation is an obstacle there, not the target — but it proves the measurement works at sample scale. |
| **Ansari, Eyer & Kerschbaum 2023** (MNRAS 522, 6087) | 58,200 GCVS × Gaia DR3 | **9,881 GCVS variables carry no Gaia variability flag**, attributed *entirely* to detection capability. This is KNELL's null hypothesis, quantified, and the number the efficiency machinery exists to beat. |
| **Graczyk et al. 2011** (arXiv:1108.0446) | 26,121 OGLE-III LMC EBs | Defines a **"Transient Eclipsing Binaries"** class and flags ~17 fast-precessing systems — as a catalogue byproduct. |
| **Soszyński et al. 2016** (arXiv:1601.02020) | OGLE Cepheid catalogue | Abstract explicitly lists *"objects ceasing pulsations"*. Found **by accident inside a catalogue paper** — the strongest evidence the search is both novel and non-empty. |
| **Hey & Aerts 2024** (arXiv:2405.01539) | ~60,000 OBAF pulsators, Gaia→TESS | Exactly this machinery (catalogue → new epoch → re-detect) applied to a *classification* question. The precedent for cross-mission re-detection systematics. |
| **Järvinen & Strassmeier 2025** (arXiv:2504.19670) | 78,111 RAVE dwarfs, Maunder-minimum candidates | A real search for *ceased cyclic activity* — chromospheric, not photometric. **13 candidates → 11 killed by Gaia parallaxes showing evolved stars; 2 survived.** The best vetting lesson in the sweep, and the reason an HR-diagram check runs before anything is believed. |
| **Petz & Kochanek 2025** (arXiv:2501.14058) | 9,361,613 ASAS-SN sources → 782 slow variables | First-moment secular trends. The template for the funnel's shape (10⁷ → 10³ → visual), not for its statistic. |
| **VASCO** (arXiv:1606.08992, 1911.05068) | 600M USNO-B1.0 vs Pan-STARRS | Searches for **vanishing flux**, not vanishing **periodicity**. Orthogonal statistic, orthogonal systematics (plate defects, asteroids, high proper motion). The obvious "hasn't this been done?" objection, with a crisp answer: no. |
| **Kipping & Teachey 2016** (arXiv:1603.08928, MNRAS 459, 1233) | theory | *A Cloaking Device for Transiting Planets* — **the only paper proposing deliberate erasure of a periodic signal as a technosignature**, and it proposes no search. This channel is the untested observational half of a decade-old hypothesis. |

### 2.4 The substrate decision, which is the sharpest result of the sweep

**Eclipsing binaries are the worst possible substrate for a technosignature
cessation claim, and are used here only as a calibration set.**

Nodal precession driven by a non-coplanar tertiary is a *named, reviewed,
modelled, actively hunted* channel with a rate high enough to swamp anything
exotic — ~28 lost systems in a single Kepler→TESS revisit, ~17 transient EBs per
26,000 LMC EBs — and it comes with **predicted return dates** (SS Lac → 23rd
century; HS Hya → ~2195; V907 Sco has already returned). The correct use of EBs
is therefore: *can KNELL recover the known disappearances?* Then exclude them
from the anomaly sample.

The novelty lives where no analogous geometric mechanism exists: **pulsators**,
and in particular the high-amplitude δ Scuti / SX Phe regime, where the sweep
found no amplitude-cessation literature at all.

### 2.5 Verification status of the citations — stated honestly

Sandbox egress is 403-blocked for `WebFetch` and `curl` against *every* host
including `example.com`; only `WebSearch` worked. So "verified" here means **the
arXiv ID and the title appeared together as a title+URL pair in a live search
index**, which reliably verifies ID→title mappings. **No full text was read, and
no quotation below has been confirmed character-for-character.** A control test
(searching a deliberately corrupted variant of the Kurtz sentence) returned the
correct paper, proving the backend's phrase matching is not strict — so
exact-phrase hits do **not** prove verbatim wording. Re-verification of every
quoted string is a runner-side job before any manuscript.

Two corrections to the citations this channel was briefed with, both material:

* **arXiv:astro-ph/9805019 is not Torres & Stefanik.** It is Tomasella &
  Munari 1998, *"Spectroscopic orbit of the ex-eclipsing binary SS Lac in the
  young open cluster NGC 7209"*. The real Torres & Stefanik SS Lac paper is
  *"The Cessation of Eclipses in SS Lacertae: The Mystery Solved"*, **AJ 119,
  1914 (2000)** — and **no arXiv ID for it was found**; none is asserted here.
* **arXiv:1807.03448 is the WISE catalogue of periodic variables, not ZTF's.**
  The ZTF periodic-variable catalogue is **Chen et al. 2020, arXiv:2005.08662,
  ApJS 249, 18** (781,602 periodic variables) — that is the correct handle for
  the discovery-catalogue argument.
* Also: **Torres 2001**, not 2000 (arXiv:astro-ph/0012542 posted Dec 2000, AJ
  121, 2227, 2001); Eggleton & Kiseleva-Eggleton's journal title differs from
  its arXiv title (ApJ 562, 1012, 2001); and arXiv:1605.02760's actual title is
  *"The Stability of F-star Brightness on Century Timescales"* — the Menzel-Gap
  result is inside it, but it is not the paper's title.

One nuance in the anchor paper that must not be dropped: **HD 60435's cessation
was not permanent within the dataset** — the paper reports a brief resurgence of
a single mode during one rotation cycle. Cessation is always a statement over a
stated baseline. §7 encodes that.

---

## 3. THE GOVERNING METHODOLOGY — cessation is established at FIXED SENSITIVITY

This section is the channel. Everything else is bookkeeping.

### 3.1 The trap

A "cessation" is a **non-detection**. A non-detection has two causes:

1. the signal stopped — the target;
2. the block could not have detected it anyway — the confounder.

Cause 2 is not rare, it is the **default**. A variable "ceases" whenever the
newer data have worse cadence, a shorter seasonal window, larger photometric
errors, a different alias comb, or a different passband. Between two seasons of
*the same survey* the epoch count changes, the window length changes, the alias
structure moves, and the errors change with depth and moon phase — and every one
of those changes the probability of recovering a **fixed** signal.

A statistic that compares peak significance early against peak significance late
therefore ranks stars by **how much the cadence degraded**. The sibling RUST
channel measured the analogous failure in the second moment: an uncorrected
version of its statistic flagged **46 of 60** injected confounders
(`docs/rust.md` §5.0).

### 3.2 The design consequence

**The primary search is intra-survey.** ZTF's own early seasons against its own
late seasons, in both g and r, where passband, pipeline and calibration are
constant by construction. Cross-survey (GCVS/VSX → ZTF) is a **secondary**
layer, and §6 states the extra burden it carries.

**Every claimed non-detection is normalised by the injection-measured detection
efficiency for that star's own period and amplitude in that block's own sampling
and noise.** Formally,

```
eta(P, A, block) = P( the blind block detector fires
                    | a signal of period P and amplitude A is present,
                      observed at THIS block's epochs,
                      with THIS block's noise,
                      scored against THIS block's own permutation threshold )
```

### 3.3 How η is measured (`src/seti/knell/efficiency.py`)

* **Injection into the block's own observed magnitudes.** By the cessation
  hypothesis the post-transition block contains no signal, so
  `y_obs + A sin(2πft + φ)` — random phase, `n_trials` draws — is a light curve
  with the block's real sampling, real error distribution, real correlated
  systematics and real outliers, plus a signal of known amplitude. **Nothing
  about the noise is modelled, so nothing about the noise can be modelled
  wrongly.** (Gaussian and residual-resampling modes exist and are labelled;
  a test confirms the Gaussian idealisation is *optimistic* relative to real
  fat-tailed noise, which is why it is not the default.)
* **The identical detector.** The injected curves are scored with the same
  batched generalised Lomb–Scargle over the same frequency grid against the same
  permutation threshold that the search applies to the data. An efficiency
  measured with a different criterion than the search uses would be worse than
  none. The GLS implementation is verified against `astropy.timeseries.LombScargle`
  to `1e-8` in a test — the detector is checked against an independent
  implementation, not merely against itself.
* **Injected at the conservative amplitude.** The injection uses
  `A_pre − 1σ`, so a marginal early detection cannot be laundered into a
  confident late non-detection by assuming the signal was stronger than measured.
* **Batched.** `gls_power_batch` computes hundreds of periodograms on one shared
  time sampling as matrix products. This is not a micro-optimisation: it is what
  makes an injection-measured efficiency affordable per star rather than a
  survivor-only luxury.

### 3.4 What η is used for

1. **A gate.** A post block with `eta < eta_min` (0.90) carries no information
   about cessation. Verdict `low_efficiency`, not "ceased". **This one rule is
   what makes a degrading cadence unable to produce a candidate.**
2. **A p-value.** Under the null "the signal persisted", the probability of the
   observed run of non-detections is `Π_i (1 − eta_i)` over post blocks. Each
   factor is a **Clopper–Pearson 95% upper bound** on that block's miss rate, so
   the product is an upper bound: the number can only overstate the chance the
   clock is still running.
3. **A sensitivity curve.** `efficiency_curve` / `amplitude_at_efficiency` give
   the amplitude at which each block reaches 50% and 90% — the honest statement
   of what that block could and could not have seen.

### 3.5 P-values are quoted as inequalities at the resolution floor

With `n_trials` injections, once every trial is recovered the miss rate is only
*bounded*, never measured: the smallest resolvable value is
`1 − 0.05^(1/n_trials)` — 1.5% at n = 200, 0.75% at n = 400. The module sets
`pinned_at_floor`, and `format_pvalue` renders the result as
`<= 4.97e-05 (injection-resolution limited)`. A pinned p-value is a signal to
**escalate the trial count**, and a test asserts that escalating tightens the
bound. No point estimate is ever emitted at the floor.

### 3.6 The threshold is also per-block, and also permutation-derived

A block is "periodic" when its maximum GLS power exceeds the `1 − fap` quantile
of the max-power distribution over **permutations of that same block's own
magnitudes**. Shuffling destroys coherence while preserving (a) the exact
observing window, hence the exact spectral window and alias structure, and (b)
the exact magnitude distribution, hence any non-Gaussian tail. A single analytic
FAP applied to every block would carry neither property, and the difference
between blocks *is* the systematic. Measured: a 16-epoch block demands ≳0.2 more
normalised power than an 80-epoch block for the same false-alarm rate.

**One documented approximation, stated because it is easy to hide.** The
magnitudes are permuted but the per-epoch error vector is held in its original
order, because a per-draw weight vector cannot be written as the matrix product
that makes this affordable per block per star. Since `magerr` tracks `mag` in
real photometry, the null's weighting is slightly mismatched to its magnitudes.
The direction is known — it under-weights whichever epochs carry the extreme
magnitudes, widening the null and *raising* the threshold — and a raised
threshold makes detection harder in the data **and** in the injections. Because
η is measured against this same threshold, the approximation **cancels between
the two** rather than biasing the cessation statistic.

The same logic is applied to the PDM cross-check: **Θ is calibrated by
permutation, not by an absolute cut**, because Θ's null distribution depends on
epoch count, bin occupancy and phase coverage — precisely the dependence that
must not leak in. The post-block PDM veto takes a Šidák correction over the
number of post blocks, so the veto's own false-rejection rate does not rise with
the length of the baseline.

---

## 4. Method

1. **Data.** ZTF DR g **and** r, 2018–2025, bulk-pulled per sky tile from the
   IRSA light-curve API and paired **positionally** at 1.5″ (ZTF assigns
   different `oid` per filter). Machinery inherited from `seti.rust.acquire`.
2. **Block.** Observing seasons, fixed origin, ≥15 epochs per block, ≥4 blocks.
   Thin blocks are **dropped, not merged** — merging would smear the transition
   the channel is trying to localise. Blocks are never merged across bands
   either: the two-band rule compares block *indices*.
2a. **Triage, and it is lossless by construction.** The pattern requires a
   *prefix* of detected blocks, so a star not periodic in its **first** block can
   never be a candidate. Triage tests that one block at a deliberately looser
   threshold than the search (`fap` 0.05, 60 permutations vs 0.01, 200) and skips
   the rest. It therefore removes nothing the full test would keep — a test
   asserts exactly that — while removing the overwhelming majority of a ZTF
   field, which is constant stars. Without it the sweep spends its whole IRSA
   time budget running permutation nulls on stars that were never clocks.
3. **Per-block periodogram**, each against its own permutation threshold (§3.6),
   plus a PDM cross-check at the reference period. **Never a global
   periodogram** — a global one averages the cessation away, which is plausibly
   how the effect has stayed uncatalogued.
4. **Reference frequency** from the detected blocks, refined on their union.
   Every later statistic is evaluated at this fixed frequency, so there is **no
   trials factor anywhere after this step**.
5. **Transition pattern.** Detection must be a run of `True` followed by a run of
   `False`, ≥2 of each, with the post-transition baseline exceeding 500 days.
6. **Efficiency gate + persistence p-value** over the post blocks (§3).
7. **Mean flux unchanged** (|Δ| ≤ 0.05 mag). A star that faded until its signal
   sank into the noise is a fade — a different and mundane phenomenon. Note the
   efficiency gate catches this too (fainter, noisier data ⇒ lower η), so the two
   guards are **independent**, which is why both are kept.
8. **Excess-variance drop.** The frequency-agnostic variability budget must fall
   (post/pre ≤ 0.35). See §5, mode switching.
9. **Two-band coincidence, at scoring time.** Both bands must show cessation, at
   the **same transition block** and the **same period** (or a low harmonic of
   it). A bad reference image, a filter-specific ghost, or a blend with a
   variable neighbour of one colour all stop a "period" in one band alone; a
   coincidence of two unrelated single-band artefacts fails the epoch-agreement
   test.

---

## 5. The astrophysical cessation mechanisms, and how each is closed

Each of these is a *named mechanism by which a real clock stops or appears to
stop* — not a generic quality cut. The vetting order is instrumental first, then
survey-detectability, then astrophysics, so a star tripping several is reported
under the most mundane one.

| Mechanism | Reference | How KNELL separates it |
|---|---|---|
| **Third-body nodal precession** (the SS Lac case) | Torres 2001 (astro-ph/0012542, AJ 121, 2227); Torres & Stefanik 2000 (AJ 119, 1914); Eggleton & Kiseleva-Eggleton 2001 (astro-ph/0104126, ApJ 562, 1012); Zasche et al. 2025 (2504.17298) | Amplitude **declines** for years before vanishing → `pre_decline_precession_like`. Plus the astrometric companion test: Gaia `ruwe`, `non_single_star`, `astrometric_excess_noise`. Both → `third_body_precession`. |
| **Catalogue error / spurious original entry** | GCVS class `CST` is literally *"nonvariable stars, formerly suspected to be variable and hastily designated"*; ASAS-SN Cat. II (1809.07329) | Structurally impossible in the intra-survey primary — the period is *measured here*, in the same data, before it is claimed to have stopped. This is a further argument for the intra-survey design. |
| **Detection-sensitivity mismatch between epochs** | Ansari et al. 2023 (9,881/58,200 GCVS) | §3, the entire efficiency machinery. |
| **Blazhko-like amplitude modulation** | V445 Lyr drops ~1 mag → ~0.07 mag p-p at Blazhko minimum (1205.1344); SuperWASP 983 Blazhko candidates (1707.02045) | The amplitude **comes back**, breaking the required run of non-detections. Plus ≥500 d post baseline, plus a pre-transition modulation index, plus SIMBAD RR Lyr class → `amplitude_modulated`. |
| **Pulsation mode switching** | V338 Boo (2411.09739); OGLE-BLG-RRLYR-12245 (1403.6476) | Two independent closures. (a) The blind per-block detector fires on *any* frequency, so a mode switch leaves late blocks detected and no transition exists. (b) **Total power is conserved in a mode switch** — the excess-variance test catches it even when the new mode falls somewhere the periodogram handles badly → `mode_switch`. |
| **δ Scuti amplitude modulation** | Bowman et al. 2016 (MNRAS 460, 1970): **61% of 983 Kepler δ Sct** show amplitude modulation in ≥1 mode; KIC 8712760 declined then **rose again** | The astrophysical noise floor of the recommended substrate. Handled as Blazhko is, and stated as a limitation in §7 rather than pretended away. |
| **Evolution out of the instability strip** | M33 V19, *"A Cepheid is No More"* (astro-ph/0102453, ApJ 550, L159), amplitude 1.1 mag → <0.1 mag; Polaris (predicted to stop, then reversed) | Accompanied by a **mean-magnitude and colour change** — V19 brightened ~0.5 mag. The constant-mean-flux requirement is the test. |
| **End-of-AGB pulsation loss** | Engels et al. (1811.06906); R Hya (2402.09819) | Long periods, large amplitudes, huge mean-flux changes; excluded by period range and the mean-flux test. |
| **Spot-cycle evolution in rotational variables** | Montet et al. 2017; Reinhold et al. 2019 | **The hardest one**, and it is not fully separable. Photometric amplitude can fall to zero while magnetic activity persists. Survivors whose period is 0.5–60 d at ≤60 mmag carry an explicit `spot_cycle_plausible` flag and the write-up names it as the **leading benign interpretation**, rather than folding it silently into the survivor count. |
| **CV disc states / VY Scl low states** | 2411.07744 (MNRAS 535, 3035); HS 0506+7725 (1902.04334) | SIMBAD class → `cataclysmic_disc_state`; and disc-state changes move the mean, which the mean-flux test rejects independently. |
| **Be-star disc loss** | ~22% of Be stars undergo a disc-loss/renewal episode within 1 yr | Mean-flux change plus SIMBAD class. |
| **AGN red noise** | — | Red-noise power is not a clock, but over one block it can produce a formally significant peak that does not repeat. SIMBAD class → `agn_red_noise`; statistically, the requirement of ≥2 consecutive pre-detections **at a consistent frequency** is the defence. |
| **Blending** | — | Gaia neighbour census inside the ZTF PSF → `blended`. |
| **Instrumental walls** | — | Both ZTF limits manufacture this signature: a saturating star's amplitude is compressed non-linearly, and a star near the faint limit loses its peak whenever survey depth dips — a cadence effect in a photometric costume. |

Two cross-cutting rules the literature hands over for free, both encoded:

* **Cessation is frequently reversible.** V907 Sco, Polaris, V338 Boo, RR Lyr's
  own Blazhko modulation, KIC 8712760 — and HD 60435 itself had a mode resurge.
  "Ceased" is therefore always defined **over a stated baseline** and never
  asserted absolutely; `post_span_days` is reported on every candidate.
* **Check the HR-diagram position first.** The Maunder-minimum search lost 11 of
  13 candidates to Gaia parallaxes showing they were evolved stars. Gaia
  astrometry is pulled for every candidate before anything is believed.

---

## 6. The cross-survey secondary layer, and its extra burden

`knell_cross` compares catalogued VSX/GCVS variables against ZTF. It is
**secondary and labelled as such**, because here the passband, cadence, aperture
and depth all change *at the same time as the epoch* — the naive comparison
measures two telescopes against each other, and that is precisely the confound
the intra-survey primary exists to avoid.

Every cross-survey candidate therefore carries `crossmatch_demonstration`: the
**catalogued** period and amplitude are injected into the **ZTF light curve's
own epochs and noise**, and the identical blind detector is applied. A candidate
whose `demonstrated` is False is not reported as a cessation at all — it is
reported as untestable. The cross layer is held to a **stricter** efficiency
floor than the primary (0.95 vs 0.90) for exactly this reason.

**On the plate archives.** The longest baselines (DASCH, APPLAUSE) are the
natural next reach for any survivor — HS Hya (arXiv:2107.10954) shows the
archival-plate route works over >125 yr. But the systematic that killed a famous
prior claim applies: **Lund, Pepper, Stassun & Hippke 2016** (arXiv:1605.02760,
*"The Stability of F-star Brightness on Century Timescales"*) showed apparent
century-long dimming trends in DASCH F stars are artifacts of the **Menzel
Gap** — the suspension of the Harvard photographic sky patrol under HCO director
Donald Menzel, which the paper treats as a **systematic offset between data
before 1953 and after 1969**. (Sources disagree on the exact bracket — 1953–1969
vs 1954–1965 with incomplete recovery to ~1970 — so it is cited as "the ~1953–1969
gap" with the before-1953/after-1969 offset as the operative definition, and not
stated more precisely than the record supports.) **Any DASCH work that does not
model the Menzel Gap will reproduce that error.** No DASCH stage is wired in this
channel until a survivor justifies it; wiring it earlier would only produce an
unvalidated code path.

---

## 7. Measured performance (`tests/test_knell.py`, offline, no network)

| Case | Realisations | Flagged |
|---|---|---|
| **Constant-amplitude periodic signal, DEGRADING CADENCE** (80,80,70 → 18,15,15 epochs/season) — the test that decides the channel | 24 | **naive statistic: 22. With the efficiency gate: 0.** |
| Constant amplitude, **degrading errors** at fixed cadence (×1 → ×4) | 10 | **0** |
| Signal stops mid-baseline at constant mean flux | 6 | recovered ≥5 |
| Star faded below the noise (+1.2 mag, ×6 errors) | 4 | **0**, all → `faded_not_ceased` |
| Mode-switching pulsator (0.63 d → 0.41 d) | 6 | **0** |
| Blazhko-like minimum with recovery | 6 | **0** |
| Constant star | 12 | **0** |
| Pure-noise block false detections at fap = 0.01 | 20 | ≤3 |
| Gradual pre-cessation decline (SS Lac-like) | 6 | ≥4 correctly flagged `pre_decline_precession_like`; with astrometric companion → `third_body_precession` |

The degrading-cadence line is the point of the channel, and it is worse than the
sibling channel's: **the uncorrected statistic flags 22 of 24** — a 92%
false-positive rate against a star that never changed. Both numbers matter. The
naive count shows the confounder *bites* (the test exercises a real systematic,
not a straw man); the corrected count shows the gate holds. Had the channel been
built without `efficiency.py`, essentially every candidate it produced would have
been a record of when ZTF re-planned its field roster.

Directly measured confounder magnitude, at fixed signal (30 mmag, 0.63 d,
20 mmag errors): **η = >0.9 in an 80-epoch block, <0.6 in a 16-epoch block.**
If efficiency did not depend on cadence there would be no confounder and no
reason for this channel's central correction; it does, and there is.

---

## 8. Layout

```
src/seti/knell/acquire.py     runner-only ZTF/VSX/GCVS/Gaia/SIMBAD + acquisition log
src/seti/knell/blocks.py      epoch blocking, batched GLS, PDM, per-block detection
src/seti/knell/efficiency.py  injection-measured detection efficiency   [LOAD-BEARING]
src/seti/knell/cease.py       the cessation statistic and its mechanism flags
src/seti/knell/vet.py         the contamination gauntlet (pure functions)
src/seti/knell/run.py         stage orchestration -> results/knell/
config/knell.yaml             every threshold
tests/test_knell.py           offline suite; the degrading-cadence test is the point
.github/workflows/knell.yml   sharded, checkpointed, commit-back
```

CLI: `seti knell-sweep` (per field, shardable), `seti knell-vet` (aggregate),
`seti knell-cross` (secondary layer).

---

## 9. Honest limitations

* **The spot-cycle degeneracy is not closed.** A decaying starspot pattern on a
  rotational variable produces a genuine, mundane cessation at exactly the
  periods and amplitudes this search is most sensitive to. It is flagged, named
  as the leading interpretation, and left standing — not solved.
* **δ Scuti amplitude modulation is a ~60% phenomenon** in the recommended
  substrate (Bowman et al. 2016). The ≥2-post-block and ≥500-day requirements
  bound it; they do not eliminate it.
* **Cessation is a statement over a baseline, not a permanent fact.** Half the
  named single-object precedents reversed. A ZTF decade constrains a ZTF decade.
* **Eclipsing binaries are excluded from the anomaly sample by design** (§2.4),
  which throws away the largest and best-characterised population of real
  cessations. That is the price of a defensible claim.
* **Two-band coincidence roughly squares the per-band completeness.** That is
  the price of the ledger's first rule and it is paid deliberately, at scoring
  time rather than at follow-up.
* **The frequency grid caps `max_period` at the block baseline.** A "period"
  longer than a block is a trend, not a clock; admitting one would let a slow
  fade masquerade as periodicity. The cost is blindness to genuine periods
  longer than an observing season, which excludes long-period variables outright.
* **η's error bar covers phase, not the noise realisation.** Injecting into the
  block's own observed magnitudes fixes the noise to its one real realisation and
  varies only the injected phase — the right question for "would *this* block have
  detected it", but the binomial bound on `1 − η` does not include the sampling
  variance of that realisation. `noise_mode="resample"` adds it when wanted; the
  default deliberately does not, and this is stated rather than buried.
* **The permutation null holds the error vector in epoch order** while permuting
  the magnitudes (§3.6). The bias direction is known and it cancels between the
  data and the injections, but it is an approximation, not an identity.
* **A clean null is not written up.** Per `CLAUDE.md`, a null here is a reason to
  move the search — to a longer baseline, to a different substrate, or to the
  plate archives with the Menzel Gap modelled — not to publish an occurrence
  limit.
