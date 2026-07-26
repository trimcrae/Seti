# ISOTHERM — the SHAPE of the waste heat in temperature space

**Signatures:** S5 (isothermal excess), S6 (Matrioshka step structure) from
`docs/necrosignatures.md`.
**Code:** `src/seti/isotherm/`. **Tests:** `tests/test_isotherm.py`.
**Workflow:** `.github/workflows/isotherm.yml`. **CLI:** `seti isotherm`.

---

## 1. Why this is necessary, not merely novel

The natural warm-debris-disk temperature locus sits at **~190 K** (Morales et
al. 2011, *"Common Warm Dust Temperatures Around Main-Sequence Stars"*, ApJL
730, L29: "nearly the same characteristic dust temperatures (∼190 K and ∼60 K
for the inner and outer dust components, respectively)", from Spitzer IRS/MIPS
across B8–A7 and F5–K0 hosts — **cite by journal reference; this paper has no
locatable arXiv preprint, so any eprint number attached to it is fabricated**) —
**directly on top of the Dyson-sphere temperature range every waste-heat search
targets.** Any positive-excess search at 130–250 K is competing against an
astrophysical population that looks identical in two broadband colours.

The false-positive rates confirm it. Silverberg et al. 2018 find **at most
7.9% ± 0.2%** of AllWISE-selected infrared excesses are good disk candidates —
a ~92% false-positive rate. McDonald et al. and Marton et al. exceed 70%. All
13 of Theissen & West's candidates with W4 S/N > 3 are false positives. Every
Project Hephaistos candidate died of background-galaxy confusion: JWST/MIRI
resolved candidates D and E as a **Hot DOG at z ≈ 0.9** and a **dusty starburst
at z ≈ 0.4**, both within ~1″ (arXiv:2607.09460); Hot DOGs at 9×10⁻⁶ arcsec⁻²
can account for all seven candidates out of 5 M stars (Ren et al. 2024).

**Amplitude and colour cannot separate two populations that overlap in exactly
those observables. Shape can.** That is this channel's entire justification.

And amplitude selection has no upgrade path in the cold regime. The Wien peaks
of the WISE bands are W1 3.4 µm → 852 K, W2 4.6 → 630 K, W3 12 → 241 K,
W4 22 → 132 K, and **W3/W4 depth is frozen at the 2010 cryogenic mission** —
NEOWISE-R, CatWISE2020 and the deep unWISE coadds are W1/W2 only. No future
broadband dataset will rescue amplitude-based selection below ~250 K.

---

## 2. Novelty status — audited, and NOT the naive claim

Verified against runner-fetched full texts in `results/dysonlit/`,
`results/hephlit/`, `results/litcheck/`, `results/seamlit/`,
`results/litcheck_dyson/` (full text for Carrigan 2009, Ĝ I–IV, Hephaistos I–II
and IV, Lacki 2016, Garrett 2015, the CatWISE extragalactic search, the 2026
review; abstracts for Wright 2023 and the 5 M-star search).

### What is cleanly novel

| Statistic | Status |
|---|---|
| **Emissivity index β fitted as a free parameter** | **CLEANLY NOVEL.** Zero occurrences of β-as-free-parameter in any technosignature paper in the corpus. `emissivity index`, `greybody`, `modified blackbody` (in the dust sense) return zero hits. β = 0 vs β ≈ 1–2 as an artificiality discriminant appears nowhere. |
| **≥ 3 discrete components in geometric progression** | **CLEANLY NOVEL.** Zero hits for `geometric progression`, `discrete temperature`, `three temperature`, `computronium`, `nested shell`. |
| **Width of the temperature distribution** | **NOVEL as a statistic.** No search has ever fitted a temperature *distribution* and measured its dispersion. |
| **CASSIS / Spitzer-IRS as the input corpus** | **NO PRECEDENT ANYWHERE.** The only grep hit for "CASSIS" in the whole literature cache is the author name "S. Cassisi". |

### What is NOT novel — stated plainly

* **"No search thresholds on shape" is FALSE.** **Carrigan 2009**
  (arXiv:0811.2376, ApJ 698, 2075) fitted Planck functions to 92-point IRAS LRS
  spectra over the 11,224-source Calgary catalogue, used a blackbody-vs-
  second-degree-polynomial model comparison, and eliminated ~80% of a
  1527-object sample by "direct scanning of the spectra for non-Planck shapes".
  **Hephaistos II**'s actual selection cut is an RMSE goodness-of-fit
  (< 0.2 mag) to a star+blackbody model, not an excess amplitude. The defensible
  claim is narrower: *no search has thresholded on the WIDTH of the dust
  temperature distribution, on a free emissivity index, or on component
  multiplicity in geometric progression.*
* **Absence of silicate features is NOT a new idea.** **Wright et al. 2014
  (Ĝ II, §3.3)** proposed it in print: "MIR spectroscopy will detect the
  characteristic PAH emission and silicate absorption features from dust, while
  one would not expect those features to be present if the origin of the bulk of
  the MIR radiation were thermal waste heat." Carrigan 2009 *executed* a 9.7 µm
  silicate rejection — the 3058-source "E" group was his single largest cut.
  Hephaistos IV uses 10 µm silicate strength via the Spoon diagram to classify
  contaminants. What survives: nobody has used feature strength as a
  **quantitative forward selection statistic over a spectral archive** rather
  than as proposed-but-unexecuted follow-up, a categorical class-letter veto, or
  a post-hoc contaminant diagnostic. The **18 µm** band appears nowhere.
* **A wide temperature grid is not novel.** Hephaistos II ran 6,216,900 models
  over 10–400 K and f = 10⁻⁴–0.4. Temperature as a *nuisance grid parameter* is
  standard; temperature-distribution *shape* as the selection statistic is not.

### The argument this channel must answer, and does

Ĝ II states flatly: *"a broadband search cannot distinguish the spectral
features of dust from that of other sources of MIR radiation."* Correct — for
four photometric bands. That is precisely why the corpus is **spectra**, not
WISE colours. WISE's four bands cannot measure β and cannot detect a silicate
feature; the statistic requires resolved continuum shape.

**Wright 2023** (arXiv:2309.06564) is the only substantive theory treatment and
it is an *efficiency* verdict, not a spectrum: "there is little to no advantage
to nesting shells (as in a 'Matrioshka Brain')", "the optimal use of mass is
generally to make very small and hot Dyson spheres", and for complete spheres
"we expect optical depths of several". **No paper computes a stepped
multi-blackbody SED, and none has ever been fitted to data.** The efficiency
objection is real and is why S6 is framed as a *partial-covering* cascade: an
ideal opaque nested shell shows only its outermost layer and is
observationally a single cold blackbody. Multiple components are visible only
if each stage leaks.

**Curtis et al. 2026** (arXiv:2604.21886, *"The Dyson Minds 2025 Workshop: SETI
around Black Holes"*, PASP 138, 046001, doi 10.1088/1538-3873/ae5a02) explicitly
recommends "apply[ing] anomaly-detection methods to archival datasets,
including those from WISE, JWST, and the Event Horizon Telescope, to identify
unusual sources potentially overlooked by standard reduction pipelines" — a
published invitation with no published response. **Its subject is Dyson Minds
around supermassive black holes, not around stars**, so it is cited here for the
archival-anomaly recommendation only, not as stellar-technosignature motivation.

**Carrigan 2009 is this channel's methodological ancestor**, and his failure
mode is fixable. He reached a colder floor (100 K) than any WISE search because
IRAS had 25/60/100 µm, but his three best candidates collapsed on the
distance/luminosity degeneracy: a nearby 1 L☉ sphere and a distant 10³ L☉ red
giant produce the same spectrum. **Gaia parallaxes break exactly that
degeneracy.** Every candidate here is Gaia-anchored.

---

## 3. The physics

### The models

All in `src/seti/isotherm/sed_model.py`, in `F_ν` against wavelength in µm.

**Modified blackbody.** `F_ν = A · (λ/λ₀)^(−β) · B_ν(λ, T)`. Real grains have
`Q_abs ∝ λ^(−β)` for `λ > 2πa`, so their SED falls off *faster* than Planck in
the Rayleigh–Jeans tail; astronomical silicate and carbon sit at β ≈ 1–2.
**β = 0 is a true Planck function**, requiring emitters large compared with
every observed wavelength.

**Continuous radial gradient — THE NATURAL NULL.** Optically thin dust with
`Σ ∝ r^(−p)` and the equilibrium profile `T ∝ r^(−1/2)` emits

```
F_ν(λ) ∝ λ^(−β) ∫ dT · T^(2p−5) · B_ν(λ, T)
```

over `T_out < T < T_in`. The change of variables `r = r_in (T/T_in)^(−2)` gives
`r^(1−p) dr ∝ T^(2p−5) dT`. A radial gradient is *necessarily* a broad
superposition of Planck functions, with width set by radial extent through
`dT/T = 0.5 · dr/r`.

**Discrete N-component.** `F_ν = Σ_k A_k (λ/λ₀)^(−β) B_ν(λ, T_k)`, amplitudes
non-negative.

### The discriminating comparison

**Discrete-N versus continuous-gradient — not "does a single blackbody fit
well".** Debris-disk practice already quotes a single `T_dust` for hundreds of
disks, and **two-temperature disks (warm belt + cold belt) are a well-populated
and entirely natural class** (Kennedy & Wyatt 2014, *"Do two-temperature debris
discs have multiple belts?"*, arXiv:1408.4116, MNRAS 444, 3164 — which compiles
such a sample and finds the warm/cool temperature ratio clustered at 2–4. Note
the title is a question: commonness is a property of the sample it assembles,
not the paper's headline result). Neither a good single-blackbody fit nor a
two-component decomposition is anomalous. What has no natural counterpart is:

1. a temperature width narrower than any physical radial extent allows, **at
   β = 0, with no silicate emission**; or
2. **≥ 3 resolved components in geometric progression** that beat the
   continuous-gradient null.

### The natural floor on temperature width

From `T ∝ r^(−1/2)`, `dT/T = 0.5 · dr/r`. Two contributions:

* **Radial extent.** The tightest resolved debris rings (HR 4796A, Fomalhaut)
  sit at `dr/r ≈ 0.06–0.18`. The **nominal floor** uses `dr/r = 0.20`
  → `dT/T = 0.10`; the **absolute floor** uses `dr/r = 0.05` → `dT/T = 0.025`,
  below every published value.
* **Grain-size spread.** At fixed orbital radius, `T_grain ∝ a^(−β/(4+β))`, so a
  realistic size distribution alone contributes `dT/T ≈ 0.1–0.2`. It is quoted,
  not imposed, because a β = 0 emitter has no such spread by construction —
  which is exactly why the width and β tests must be passed *together*.

The statistic is the **emission-weighted** width `sqrt(12·Var[ln T])`, not the
nominal `T_in/T_out` bounds. With a free surface-density index the weight
`T^(2p−5)` can concentrate nearly all the flux at one end, so a nominally
decade-wide gradient can be physically isothermal. Using nominal bounds made
this statistic seed-dependent: 4 of 5 noise realisations of a genuinely
isothermal source reported a decade-wide temperature distribution.

---

## 4. Method and funnel

| Stage | Test |
|---|---|
| probe | Archive reachability; explicit `NO_DATA_REACHED` rather than silent degradation |
| screen | ~1 s/spectrum: SNR, rest-frame dust features, dust-like β, module-stitching step. **May only reject, never promote.** |
| shape | ~20–50 s/spectrum on survivors: full statistics below |
| score | Funnel counts, candidates, sensitivity map, `REPORT.md` |

**Shape statistics** (`shape_stats.py`): free β with a Δχ²-profile uncertainty
and an explicit *leverage gate*; temperature-distribution width with a 95%
upper limit; equivalent widths for silicate 9.7/18, PAH 6.2/7.7/11.3/12.7,
H₂O/CO₂ ice, C₂H₂, crystalline forsterite 23.7/33.6; component count by BIC;
geometric-progression test; energy ladder; extragalactic interloper scan;
order-step statistic.

**Model selection.** Components are chosen by BIC with ΔBIC ≥ 10 ("very strong"
on Kass & Raftery). BIC uses the number of **resolution elements**, not pixels:
IRS low-res is ~2× oversampled, so raw-pixel BIC over-rewards complexity by
~ln 2 per parameter and would manufacture cascades. Spectra are rebinned to
R ≈ 100 before any fit, and a 5% fractional systematic is added in quadrature.

---

## 5. Contamination ledger

Inherited rules plus channel-specific ones:

* **Background galaxies are the universal killer.** A coherent PAH/silicate
  system at one redshift ≥ 0.03 vetoes. The scan is ~80 redshifts × 5 features,
  so the bar is 5σ **and** |EW| ≥ 0.15 µm. Under a significance-only 3σ cut this
  test flagged a pure blackbody at every redshift offered. Cuts *for* the
  channel: a Hot DOG is a redshifted, feature-rich, multi-component SED, so
  these statistics are among the few that could have rejected Hephaistos D and E
  from the data alone rather than needing JWST.
* **Continuum curvature manufactures features.** A featureless Planck function
  is curved in log–log, so a *linear* local continuum yields a false 9.7 µm
  EW of **+1.67 µm** — as large as a real silicate band. The continuum is a
  **cubic** in log–log (false EW −0.006 µm), and anchor windows must lie
  entirely inside the observed range: **no extrapolation, ever.** A truncated
  red anchor fabricated 0.25 µm "features". This bug vetoed every candidate
  until fixed.
* **A single-component β is biased to 0 for multi-temperature sources.** A
  β = 1 gradient disk reports β = 0.00 ± 0.001 from a one-component fit — a
  false "true Planck function". β is read off **the model the data select**.
* **β and T are degenerate outside the in-band-peak window.** On the R–J tail
  `F_ν ∝ T λ^(−(2+β))`. β is reported as unconstrained unless the Wien peak is
  in band with ≥ 5 elements each side.
* **A fitted component > 1800 K** is hotter than grains survive → unresolved
  companion photosphere.
* **IRS module stitching.** SL and LL have different slit widths; a step at
  ~14.2 µm above 25% means the two halves are not the same object's flux, which
  manufactures spurious multi-component structure.
* **Photospheric feedback.** Huston & Wright 2022 show a star enclosed by a
  Dyson sphere expands and cools, shifting the *stellar* SED component. The
  decomposition therefore fits the photosphere rather than fixing a template.
* **Gaia anchoring is mandatory** — Carrigan's distance/luminosity degeneracy.

---

## 6. Corpus

**Intended primary: CASSIS**, the Cornell Atlas of Spitzer/IRS Sources —
~13,000 public low-resolution 5–38 µm spectra, never used for a technosignature
search. Access routes are probed independently and the run reports which
worked: CASSIS direct HTTP; IRSA TAP Spitzer/IRS Enhanced Products (table names
are *discovered* from `TAP_SCHEMA`, not hard-coded); VizieR (the CASSIS
catalogue, and the IRAS LRS Calgary atlas that was Carrigan's corpus).

### CASSIS is NOT reachable from the runner — measured, not assumed

Probe run **30208087571** (2026-07-26, 11:24 EDT) settled this. Stated plainly,
because the channel brief forbids confusing an unreachable archive with an
empty one:

| Route | Result |
|---|---|
| `cassis.sirtf.com/atlas/` (home, `radec.py`, `ascii.py`) | **HTTP 402**, 4383 bytes of `<title>Making sure you're not a bot!</title>` with a `/.within.website/x/xess/` stylesheet — i.e. **Anubis**, a proof-of-work anti-scraper gate. All three endpoints, identically. |
| `cassis.astro.cornell.edu/atlas/` | `SSLCertVerificationError` — **hostname mismatch**; the certificate is not valid for that name. |
| VizieR `J/ApJS/196/8` (the CASSIS catalogue) | not present — `reachable: false, table: null`. |

So CASSIS is *up but gated*, not down. It cannot be reached non-interactively,
and no amount of retrying changes that. `cassis_reachable: false` is a
first-class field in `archive_probe.json`.

### What the corpus actually is

The probe's verdict was **`SPECTRAL_ARCHIVE_REACHED`**, not `NO_DATA_REACHED`,
because the fallback is *also spectra* rather than the photometric backstop:

* **IRSA TAP — reachable**, 154 Spitzer/IRS-matching tables, including
  **`irs_enhv211`, the Spitzer/IRS Enhanced Products v2.1.1 atlas**. This is the
  natural CASSIS stand-in: the same low-resolution IRS spectra, pipeline-reduced,
  served over a working TAP endpoint. It is the corpus this channel now runs on.
* **VizieR TAP — reachable**, and **`III/197/lrs`**, the IRAS LRS Calgary atlas
  (**Carrigan 2009's actual corpus**), resolves with its full column set. This
  gives a direct, like-for-like comparison against the methodological ancestor.
* **Gaia TAP — reachable**, which the mandatory parallax anchoring requires.

**Table selection is ranked, not first-past-the-post** (`rank_irs_tables`). The
discovery query matches on `irs`/`spitzer`/`cassis`, which also catches IRSA's
own TAP bookkeeping tables (`irsa_groups`, `irsa_directory`,
`irsa_serv_descriptors`), the unrelated **IRTS** near-IR spectrometer catalogue
(`irts_nirspsc`), and ~90 image mosaics. Every one of them answers a `SELECT`
successfully, so taking the first responder silently adopted a bookkeeping table
as "the Spitzer/IRS corpus" — a wrong-corpus failure that is indistinguishable
from a successful run. Non-spectral tables are now dropped outright and the
known low-resolution products ranked explicitly.

**What this costs.** The novelty claim in §2 was "CASSIS as the input corpus has
no precedent anywhere". That is unchanged in substance — the IRS Enhanced
Products are the same Spitzer/IRS spectra and have equally never been used for a
technosignature search — but the claim must be **written as IRS Enhanced
Products, not CASSIS**, and the source-count is the Enhanced Products' coverage,
not CASSIS's ~13,000. Do not quote a CASSIS sample size for a run that never
touched CASSIS.

**Backstop:** Gaia + 2MASS + WISE + AKARI IRC/FIS + IRAS SED assembly at
catalogue scale. Far-IR bands are what give the photometric path any β leverage
at all — WISE alone cannot do it.

---

## 7. Sensitivity and honest limitations

Measured by injection-recovery (`cascade_sensitivity`, committed to
`results/isotherm/sensitivity.csv`):

* **Cascade separability.** A 3-shell cascade beats the continuous-gradient
  null at temperature ratio ≳ 3 even at SNR 60; at ratio 2 it needs SNR ≳ 400;
  at ratio 1.5 it is not separable at any SNR tested.
* **Band-limited ratio.** Three Wien peaks inside 5–38 µm force a temperature
  ratio ≤ 2.8 (`λ_peak·T = 5099 µm·K`). Ratios large enough for easy separation
  push components out of band, where their temperatures are extrapolated from
  one flank. This is a hard property of the band, and it is why **per-component
  narrowness is a high-confidence tier, not a gate** — requiring it would reject
  every cascade the band can actually contain. Extending the baseline with
  AKARI FIS / IRAS 60/100 µm photometry re-opens that tier.
* **β leverage.** Constrained only for components whose Wien peak is in band,
  i.e. T ≈ 130–1000 K. Outside that, β is reported unconstrained.
* **Width test needs SNR ≳ 300.** Measured 95% upper limit on `dT/T` for a
  genuinely isothermal 250 K emitter, 5 noise realisations each:

  | median SNR | 95% upper limit on `dT/T` | below the 0.10 floor |
  |---|---|---|
  | 60 | 0.196 | 0/5 seeds |
  | 120 | 0.130 | 1/5 seeds |
  | 300 | 0.080 | 5/5 seeds |
  | 1000 | 0.043 | 5/5 seeds |

  The limit scales as roughly `SNR^(−1/2)`. Below SNR ≈ 300 the S5 test cannot
  reach the natural floor and correctly returns no flag — the channel is
  sensitive to isothermal emitters only in good spectra, and says so rather
  than lowering the floor to manufacture candidates.
* **Redshift reach.** PAH 6.2/7.7 leave the 5–38 µm band above z ≈ 1.5–3.9, so
  high-z interlopers are not excluded by the spectral test alone.

**No-null rule (CLAUDE.md).** An empty candidate list is a statement about this
corpus at these thresholds, never a publishable result. The sensitivity map
exists so a non-detection is interpretable — it states where a detection would
have been possible at all.
