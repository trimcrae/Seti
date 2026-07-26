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

### 1.1 The boundary, quantitatively

The diffusion/levitation zoo — Ap, Am/Fm, HgMn, roAp, He-weak, lambda Boo —
terminates at a *sharp* boundary. Xiang et al. (arXiv:2006.03329), on ~15,000
LAMOST A/F stars: the peculiar stars are *"almost exclusively main sequence and
subgiant stars with Teff ≳ 6300 K"*, tracing *"a sharp border at low
temperatures along a roughly fixed-mass trajectory (around 1.4 M☉) that
corresponds to an upper limit in convective envelope mass fraction of around
10⁻⁴"*. Above 1.5 M☉ these stars are **40% of everything** — the phenomenon is
ubiquitous right up to the boundary and absent below it. The coolest lambda Boo
object in the fetched corpus is a pre-main-sequence Herbig star at Teff 6500 K
and **log g 3.5 — not a dwarf**.

The mechanism side agrees. Sweigart (astro-ph/0103133): *"the onset of
radiative levitation … coincide[s] with the disappearance of surface
convection."* Michaud et al. (arXiv:1102.1969) find the large anomalies require
separation confined to a mixed mass of ~10⁻⁷–10⁻⁶ M☉. Church et al.
(arXiv:1908.06988) measure **3.45 × 10⁻³ M☉** already at a solar-metallicity
M67 turnoff — a three-to-four decade margin before one even reaches a G dwarf.

The amplitude ladder tracks envelope mass across ~7 decades:

| regime | anomaly amplitude |
|---|---|
| radiative-envelope CP stars | **1–4 dex** |
| metal-poor MS turnoff | 0.1–0.3 dex |
| solar-metallicity turnoff | 0.05–0.2 dex |
| **G/K/M dwarfs** | **≲0.02–0.04 dex** |

Korn et al. (arXiv:2111.00913) measure 0.3 dex at [Fe/H] = −2.3 falling to
0.1 dex at −1.1 — the trend runs the right way, and caps the natural effect at
~0.3 dex even in the most favourable case ever measured. In the target box
itself, Souto et al. (arXiv:2105.01667) find Coma Ber G/K/M dwarfs at Teff
3200–6500 K, log g 4.3–5.0 homogeneous to ⟨[Fe/H]⟩ = +0.04 ± 0.02 dex.

**Two things must be conceded, not hidden.**

*Sparse anomalies are physically possible* — Te at ~10⁶× solar, Zr/Pb/Ge/Y, Br,
Sb, Ni at +0.6 dex against solar Cr/Mn/Co. But **every documented instance is at
Teff ≳ 10,000–20,000 K in a star with essentially no convection zone.** The
physics that makes a sparse anomaly possible is exactly what this population
lacks; that is the boundary condition which *strengthens* the argument rather
than weakening it.

*Diffusion is detected in cool main-sequence stars* — at ~0.1 dex, in clusters,
at the turnoff. But it is never sparse. Michaud et al. (astro-ph/0402544) on
M67 and NGC 188: *"small **generalized** underabundances"*. Souto et al.
(arXiv:2607.14208) on NGC 752 and Ruprecht 147 measure Fe, C, N, Na, Mg, Al,
Si, S, K, Ca, Ti, V, Cr, Mn, Co and Ni and find the warmer stars depleted
*"at the ≥1σ level for **all elements available in the analysis**"* — sixteen
elements, one direction. The honest nuance is that the amplitudes differ
smoothly by element, ordered by radiative acceleration (Mg 0.2 / Fe 0.1 /
Ti 0.07 dex), so the correct claim is "many elements move together,
monotonically with Teff, with a smooth element ordering" — not that the shift
is uniform. Either way it forbids a lone-element outlier.

### 1.2 The metallicity bound — a leak that had to be closed

Convective protection is a function of **metallicity as well as Teff and
log g**. Matrozis et al. (arXiv:1605.02791): *"stars with typical CEMP-s star
masses (M ~ 0.85 M☉) have very shallow convective envelopes (Menv < 1e-7
M☉)."* Four orders of magnitude thinner than a solar-metallicity dwarf — and
such a star can pass a Teff < 6000 K / log g > 4.0 cut while keeping exactly
the thin envelope that makes diffusive peculiarity possible.

The sample is therefore bounded at **[Fe/H] ≥ −1.0**, enforced in the ADQL, in
`config/thresholds.yaml` and again in the vetting funnel. Korn's measured
amplitudes (0.3 dex at −2.3, 0.1 dex at −1.1) put the natural effect an order
of magnitude below the signal at that floor. There is a cross-channel irony
worth stating: OSSUARY *selects* the metal-poor halo stars this channel must
*exclude*, and for the mirror-image reason.

The sample is also bounded at **Teff ≥ 4000 K**. GALAH's own release notes
state that cool stars carry systematic trends *"that can reach values of 0.5
dex for some elements"* and that *"dwarf stars are most affected at Teff <
4600 K"*; a measured G-vs-K offset of 0.08 dex within a single cluster
(Praesepe), with 5 of 18 elements disagreeing by >0.1 dex, says the same. The
M-dwarf tail is the least trustworthy part of any abundance sample, so it is
bounded out rather than silently included, and survivors between 4000 and
4600 K carry a `cool_star_caveat`.

### 1.3 Where the null leaks

**The null is strong but it is not airtight, and the honest statement is that
it is a strong prior, not a theorem.** Known and suspected leaks, each of which
the funnel must handle rather than assume away:

| Leak | Is it sparse? | How it is handled |
|---|---|---|
| AGB-companion mass transfer (barium/CH dwarfs) | **No** — raises the whole s-process, Sr/Y/Zr *and* Ba/La/Ce, usually with C | family-coherence veto; the s-process is two families and both move. Rekhi et al. 2025 (arXiv:2509.13413) map this population — s-process-polluted **cool dwarfs**, i.e. squarely inside the target box — so it is a live contaminant, not a hypothetical one |
| Planet engulfment / rocky accretion | **No** — raises every refractory along a condensation-temperature trend | family veto; and stage 4 tests it quantitatively |
| lambda Boo depletion | **No** — depletes all refractories at once; and it is a hot-star class | population cut plus family veto |
| Li depletion / Li enhancement | **Yes**, genuinely single-element | Li is **excluded by construction** from carrying a candidacy |
| C/N mixing, dredge-up | partially | C and N excluded; dwarfs only |
| NLTE / 3D modelling error in one species | **Yes**, and this is the serious one | per-element flag rates, per-field rates, cross-survey confirmation, and raw-spectrum re-measurement |
| Blends and line-list error at one line | **Yes** | element caveat table; re-measurement against Teff-matched peers |
| Unresolved binary contaminating one line region | sometimes | RUWE, RV scatter, vbroad |

The last three are why this channel's centre of gravity is the *vetting*, not
the statistic. A single-element catalogue outlier is, on the prior, a bad
measurement — Griffith et al. found, from inspecting large-residual stars,
roughly **40% physical and 60% data problems** (unflagged binarity, poor
wavelength solutions, poor telluric subtraction). Design for a 60% junk rate.

**The mass-transfer family must be conceded outright**, because it is cool,
main-sequence and hugely anomalous: barium dwarfs, CH subgiants, dC and CEMP-s
stars are F/G main-sequence stars reaching log g 4.6 and Teff ~4300 K. The
defence is *not* "cool dwarfs are not peculiar". It is that **cool dwarfs are
peculiar only densely**. Liu et al. (arXiv:0811.2079) is the cleanest contrast:
*"Y, Zr, Ba, La, Eu show obvious overabundance… Other elements, including Na,
Mg, Al, Si, Ca, Sc, Ti, V, Cr, Mn, Ni, show comparable abundances to the Solar
ones."* Fifteen s-process elements up, eleven light and Fe-peak normal — a
family, caught by the family veto. These are ~100% binaries, so RV monitoring
and C/N are the additional discriminants.

**Lithium is conceded and pre-empted.** It is the one genuinely
quasi-single-element anomaly and it is nuclear, not diffusive. Sun et al.
(arXiv:2410.20632) measure an intrinsic scatter of **0.35 dex for G/F dwarfs
and up to 0.6 dex for older, cooler stars** at fixed parameters; Spina et al.
delete stars below 6000 K from their Li analysis entirely because it is
unreliable there, and document a co-natal pair differing by **ΔLi = 1.9 dex**
and identical in everything else — a textbook sparse anomaly with a mundane
cause. Li, Be and B therefore cannot carry a candidacy here, by construction.

**Three classes a referee will invoke, and why they do not apply.** *P-rich
stars* are the most-cited "exotic single element" class in modern spectroscopy,
but the discovery abstract itself names five co-enhanced elements — *"15
phosphorus-rich stars with unusual overabundances of O, Mg, Si, Al, and Ce"* —
the follow-up adds Sr, Y, Zr, Ba, La, Nd, Pb and Cu, and they are metal-poor
**giants**. *K-rich stars* are one node of an O–Na–Mg–Al–Si–K–Ca–Sc H-burning
network in giants, and the dwarf test was actually run: Carretta et al.
(arXiv:1303.4740) measured K in turn-off and subgiant stars of four globular
clusters and found *"the stars lie in the K-Mg abundance plane on the same
locus occupied by … field stars. This holds both for giants and less evolved
stars."* *Przybylski's star*, the most famous weird-abundance star in
astronomy, is the maximally **dense** counterexample — *"the abundances
determined for about 60 chemical elements"* — and it is a cool magnetic Ap, not
a G/K/M dwarf; its exotic-element claims are upper limits rather than
detections.

**The floor to beat.** Nature does not move one element alone even at the
0.02 dex level. Ting & Weinberg reach σ ≲ 0.02 dex after conditioning and note
*"Despite the small scatter, residual abundances display clear correlations
between elements"*; Weinberg et al. find the residual noise itself is
element-correlated, in *"a correlated element group comprised of Ca, Na, Al, K,
Cr, and Ce and a separate group comprised of Ni, V, Mn, and Co."* That is
simultaneously the strongest form of the argument and the detection threshold.

---

## 2. Novelty adjudication

Runner-fetched evidence in `results/tailingslit/` plus `results/necrolit/`,
`results/chemlit/`, `results/litcheck/` and `results/przybylski_lit/`.

**A corpus-integrity failure, recorded because it nearly poisoned this
document.** The first fetch (run `30202628318`) reported 107/107 URLs
successful — and **12 of its 24 hardcoded arXiv identifiers had resolved to
entirely unrelated papers**: a neutron-star precession paper standing in for
Richer's AmFm diffusion work, an LHC dark-matter paper for Vick, a
condensed-matter paper for the lambda Boo review, *Plenoxels* for APOGEE DR17.
Every one fetched cleanly. **A successful fetch is no evidence at all that the
paper is the right one.** The mismatches are enumerated in
`results/tailingslit/INTEGRITY.md`, the offending files have been deleted
rather than left where a later reader could quote them, and the harness now
resolves every decisive paper by **title search with a title-token
verification step**, writing `verification.json` and fetching nothing for a
slug it cannot verify. The numbers quoted below come from arXiv *search-result*
Atom files — which carry the real title and abstract of whatever matched and
cannot fail this way — and from the verified full texts, chief among them
arXiv:2605.29811.

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

**The field excludes this signature by construction, and says so in print.**
This is the sharpest novelty statement available and it is stronger than "nobody
thought of it". Weinberg et al. (arXiv:2108.08860) built exactly the residual
manifold this channel uses — 16 elements, 34,410 stars, two-process model, RMS
residuals 0.01–0.03 dex for the best-measured abundances — and then searched it
star by star. Their selection rule, verbatim:

> *"High χ² values can arise from single deviant measurements, which may have a
> variety of mundane observational causes. To preferentially select genuine
> physical outliers, we have used a modified χ² criterion in which … for each
> star, we omit the element that makes the single largest contribution to χ².
> **This criterion thus requires at least two anomalous abundances**."*

Li et al. 2015 hard-code the same convention (*"at least two elements … showing
deviations larger than 0.5 dex"*). So the one group that built the residual
manifold and searched it deliberately deleted the single-element outliers
*before looking*, for a defensible reason — a lone deviant element usually *is*
an artifact — and nobody has since asked whether real sparse anomalies are
hiding behind that choice. TAILINGS is the question that convention forecloses.

**And sparse anomalies are real.** Griffith et al. (arXiv:2110.06240), while
inspecting large residuals: *"we identify **15 stars that have 0.3–0.6 dex
enhancements of Na but normal abundances of other elements from O to Ni**."*
Amplitude ~10× the residual scatter, found incidentally, never by a dedicated
per-element scan. Those 15 stars are this channel's proof of concept and its
validation target: the pipeline must recover them.

What *does* exist is the chemical-tagging literature, and every statistic in it
is a **global distance**: PCA and EMPCA (Ting et al. 2012; Price-Jones & Bovy
2018), functional PCA (Patil et al. 2022), t-SNE (Anders et al. 2018), k-means
in 15-D (Hogg et al. 2016), latent-factor models (Casey et al. 2019), graph
autoencoders (Quandt-Rodriguez et al. 2026), spectral similarity (de Mijolla &
Ness 2021). All of them are built to **cluster stars into birth groups**, not
to score an individual star's abundance vector for a single-element excursion —
and a reconstruction-error or full-vector-distance statistic is *maximised* by
dense anomalies and actively suppresses sparse ones.

**That last claim is demonstrated, not cited** — no paper states it, so
asserting it with a citation would be dishonest. `sparse.global_statistics`
computes the reduced χ² and the Weinberg leave-one-out χ² alongside the sparse
statistic, and `tests/test_tailings.py` injects two stars with the *same*
per-element amplitude — one single-element, one four-element — and measures
what each statistic does. The global statistic ranks the dense star far above
the sparse one; this one inverts the ordering; and the leave-one-out criterion
scores the sparse star at essentially zero. That injection–recovery comparison
is the methodological core, and it is a result rather than an assumption.

The empirical support that *can* be cited is Manea et al. 2025
(arXiv:2508.16717): 25 APOGEE "doppelgänger" pairs declared near-identical by a
global 20-element statistic differ by up to **0.38 dex in single
neutron-capture elements** on follow-up at R~60,000. The global statistic is
demonstrably blind to exactly this.

Prior art that makes the *method* sound rather than unprecedented, and which
should be cited as such: Signor et al. 2025 (arXiv:2511.09733), per-element VAE
decoders — but on synthetic data with three labels; Kügler et al. 2015
(arXiv:1409.8417), per-feature rather than global model fitting; Rousseeuw et
al. 2017 (arXiv:1608.05012), pointwise directional outlyingness on spectra. The
statistical machinery of per-coordinate outlyingness exists; it has never been
pointed at survey abundances.

### 2.3a The residual machinery already exists — reuse it, do not claim it

The honest position is that **S12 is partially done**, and the channel is
stronger for saying so.

**Griffith, Weinberg, Buder et al. — arXiv:2110.06240, published as ApJ 931, 23
(2022)** — already built most of this method: *"Residual Abundances in GALAH
DR3: Implications for Nucleosynthesis and Identification of Unique Stellar
Populations"*, 82,910 Galactic disk stars, 16 elements, two-process residuals
with RMS Δ[X/H] ≲ 0.07 dex. It is also the source of the 15 Na-rich stars
above, and that detail is verified verbatim against the paper: **15 stars with
0.3–0.6 dex enhancements of Na but normal abundances of other elements from O
to Ni**, alongside positive average residuals in Cu, Zn, Y and Ba. Cite the
**2022** journal version; 2021 is the preprint year only.

**Sit et al. 2024 (arXiv:2403.08067)** did the same for 288,789 APOGEE DR17
stars × 17 elements — *"Chemical Cartography with APOGEE: Two-process
Parameters and Residual Abundances for 288,789 Stars from DR17"*. It remains a
methodological reference and a published proof that the machinery works, **not
an input catalogue for a dwarf search** — but the reason must be stated
precisely, because the obvious shorthand is wrong in a way a referee would
catch. Its sample *is* evolved (T_eff 3000–5500 K), yet the log g restriction
runs the other way from the usual gloss: the **training** set is cut at
0 ≤ log g ≤ 3.5, while the **application** set imposes only 0 ≤ log g with *no
upper bound*, and the paper's headline advance is precisely that it is 8×
larger than earlier analyses *because it relaxes* the restricted log g range
they used. So the correct statement is that Sit et al.'s two-process
calibration is **trained on giants**, not that the catalogue was gated to
evolved stars by a dwarf-excluding log g cut. It is also not a planetary-
engulfment paper; it is a residual-abundance catalogue.

So the residual manifold is not this channel's contribution and must not be
presented as one. What is genuinely open is exactly two things:

1. **The population cut.** Neither Griffith nor Sit restricted to
   convective-envelope dwarfs, which is where the natural single-element
   mechanism cannot operate and therefore where the statistic means something.
2. **Sparsity as a designed test statistic**, with a stated null and a
   sparse-versus-dense hypothesis, rather than ad-hoc inspection of the
   largest residuals. Griffith's 15 Na-rich stars were found *incidentally*;
   nobody has scanned for them on purpose.

Neither has been applied to LAMOST, and there is no technosignature framing
anywhere in that lineage.

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
`log g > 4.0`, `SNR > 40`.

Catalogue **table names are discovered at runtime**, not encoded. Two dispatches
failed exactly here, in two different ways, and both are worth recording because
the honest verdict made each look like an archive problem when neither was.

Run `30204487245` reached discovery, probed the right tables and still returned
`NO_DATA_REACHED` — because `TAP_SCHEMA.tables.table_name` comes back
**already double-quoted** (`"III/283/allstar"`, quote characters included).
Interpolating that into a quoted `FROM` clause produced `FROM ""III/283/allstar""`,
which every table rejects. A quoting bug wearing the costume of an
archive-access statement. `unquote_table` now strips it at every entry point and
a test asserts no double-quoted name can reach the service; the general lesson,
which belongs in the ledger, is that **a channel whose null verdict is
indistinguishable from its bug verdict will mislead its own author**.

The first dispatch (run `30203627605`) failed differently: every encoded VizieR locator
— `III/298/galahdr4`, `III/283/allstar`, `J/MNRAS/506/2269/table1` — came back
"table not found", and `III/286/catalog` resolved with no `[Fe/H]` and zero
elements. VizieR catalogue numbers drift between releases. VizieR catalogue *numbers* drift between releases, so hard-coding one
means the channel dies the day CDS renumbers. The fix asks the service what it
actually holds: `TAP_SCHEMA.tables` is queried by keyword, every candidate is
probed for one row, and each is **scored** by how much of what the channel needs
it has — stellar parameters, SNR, an identifier, and above all how many
elements. The highest scorer is used, not the first that answers, because
discovery returns per-field subsets, value-added catalogues and README stubs
alongside the one main abundance table. The full scoreboard travels in
`provenance.json`, so the choice is auditable rather than incidental, and the
report always states which release the numbers actually came from instead of
the one that was merely intended. This is the same reasoning that made the
*column* names dynamic, applied one level up.

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

**The Karinkuzhi test — resolution and saturation.** There is a published
precedent showing that low resolution *manufactures* sparse anomalies, and it
is the single most dangerous result for this channel. Karinkuzhi et al.
(arXiv:2107.08401) re-observed at R~86,000 the 15 brightest of 895 s-process
candidates an ML pipeline had selected from LAMOST at R~1,800 — **13 classified
"Sr-only" and 2 "Ba-only"**, i.e. exactly this morphology. Every one dissolved:
*"four have no s-process overabundances, eight are mild barium stars, and one
is a strong barium star. The two Ba-only stars turn out to be both strong
barium stars and are actually dwarf barium stars."* Their conclusion is adopted
here as a rule: *"blending effects and saturated lines have to be considered
very carefully when using machine-learning techniques, especially on
low-resolution spectra."*

So two gates sit in the funnel, not in the discussion:

* `resolution_verdict` — below R = 20,000 a sparse anomaly is presumed to be
  unresolved blending, and the candidate carries
  `needs_high_resolution_confirmation = True`. This is why GALAH (R=28,000) and
  APOGEE (R=22,500) are the primary corpus and LAMOST MRS (R=7,500) is third,
  contributing targets rather than candidates.
* `curve_of_growth_regime` — an abundance is only recoverable from a line on
  the **linear** part of the curve of growth. A saturated core is insensitive
  to abundance, so it can carry an arbitrary apparent abundance error in either
  direction and a pipeline that fits it will report one.

**Instrumental covariates.** The dominant population of real single-element
outliers is instrumental, and instruments leave footprints in *instrument*
coordinates. Weinberg et al. traced two high-Ca APOGEE stars to bad pixels hit
by one particular radial-velocity + fibre combination, and a whole *population*
of low-K stars to a heliocentric velocity near −70 km/s that slid the K lines
onto a telluric band; their own conclusion is that a rare outlier and a rare
reduction problem *"are not always easy to tell one from the other"*. A real
anomaly has no reason to correlate with the star's radial velocity or its fibre
number. `covariate_rate_veto` bins the flag rate in each such covariate and
vetoes the outlying bins — the abundance-space form of the ledger rule that a
feature recurring across unrelated sightlines is instrumental.

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

**The dilution ceiling — it sets the bar and the sensitivity at once.** The
convective-zone metal reservoir governs both halves of the trade:

| Host | M_cz (M☉) | CZ metal reservoir | Δ[M/H] per M⊕ | M⊕ for +1.0 dex |
|---|---|---|---|---|
| F5 / 1.4 M☉ | 5×10⁻⁴ | 2.2 M⊕ | **0.161 dex** | 20 |
| G2 / Sun | 0.020 | 89 M⊕ | 0.0048 dex | ~800 |
| K2 / 0.8 M☉ | 0.090 | 402 M⊕ | 0.0011 dex | ~3,600 |
| M3 / 0.3 M☉ | fully convective | 1,339 M⊕ | 0.0003 dex | ~12,000 |

*The favourable half.* The Solar System's total heavy-element inventory is
≈77 M⊕, so engulfing **the entire planetary system** into a solar-type star
gives **Δ[M/H] = 0.27 dex — the absolute ceiling**. The observed record, Kronos
(HD 240430) at ~0.23 dex, already sits at ~85% of it. So a coherent refractory
excess above **~0.3 dex in a G dwarf**, or **~0.05–0.10 dex in a K dwarf**, is
outside the engulfment hypothesis by the mass budget of a whole planetary
system rather than by a threshold choice. `engulfment_ceiling_dex` computes it
per star, because the bar is spectral-type dependent and quoting one number for
the survey would be wrong. And a **single-element** excess is unexplainable by
engulfment at *any* amplitude, because rock is a mixture and must move all
refractories together along a condensation-temperature trend. That sentence is
the cleanest statement of the discriminant.

*The unfavourable half, stated plainly.* The same deep convection that makes
the natural null airtight **also destroys the signal**. An M3 dwarf needs
~12,000 M⊕ of rock for +1 dex and a K2 dwarf ~3,600 — more material than any
planetary system contains. The cool end of the sample is therefore where the
null is strongest *and* where a real signature is most diluted, and the two
cannot be optimised together. `minimum_rock_mass_for` reports the required mass
per star so the run states which side of that trade each sub-sample sits on. A
detection in an M dwarf would require an implausible amount of material — which
is itself worth saying, and is a reason to weight G and early-K dwarfs, where
the envelope is thin enough for a plausible mass to register and still thick
enough to forbid diffusion.

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

## 6. The structural tension with MIDDEN — stated, not buried

**TAILINGS and MIDDEN cannot share a population, and each one's argument
undercuts the other's.** This must be confronted because a referee will find it
immediately.

Whitmire & Wright's proposal *requires* a shallow convective envelope to keep
injected waste visible in the photosphere — which is precisely why they specify
**A5–F2**, and MIDDEN correctly follows them there. TAILINGS argues the
opposite: restrict to **cool dwarfs precisely because deep convection suppresses
the natural single-element mechanism**. Both arguments are correct, and they are
incompatible.

The resolution is not to pretend the tension away but to name what each channel
buys and pays:

| | MIDDEN (A5–F2) | TAILINGS (G/K/M dwarfs) |
|---|---|---|
| **Buys** | signal survives: a thin envelope keeps injected material at the surface, so a given injected mass gives a large photospheric excursion | a clean null: the mechanism that manufactures natural single-element anomalies does not operate |
| **Pays** | a filthy null: A5–F2 is the Am/Fm domain, where diffusion and levitation make single-element anomalies *routinely*, which is why the one long-running claim in this space (Przybylski's star) has been contested for sixty years | signal is diluted: the ceiling table above — a K dwarf needs thousands of Earth masses for a large excursion, an M dwarf more than exists |
| **Discriminator that makes it work anyway** | multi-line laboratory-wavelength coincidence plus dummy-line controls (the Andrievsky standard) | sparsity itself: dense is a rejection, so the diffusion signature — which is always generalized — cannot mimic it |

TAILINGS' bet is that **a clean null is worth more than a large signal**, because
the null is what a detection has to survive. A 0.2 dex single-element excursion
in a K dwarf is unexplainable in a way the same excursion in an Ap star simply
is not. But the price is real and is quantified in §3.6: the amount of material
required is large, and for the coolest stars it is implausible. The two channels
are complementary bets on opposite sides of the same trade, and neither
subsumes the other.

## 7. The Griffith validation, and the recalibration it forced

**The requirement.** The pipeline must recover Griffith et al.'s 15 GALAH DR3
stars with **0.3–0.6 dex Na enhancement and normal O-through-Ni** (arXiv:
2110.06240; the detail is verified verbatim against the paper, not paraphrased).
This is the one published population with exactly this channel's target
morphology. If the statistic cannot find it, the statistic is wrong.

**It is a live test, not a claim.** `src/seti/tailings/validate.py` synthesises a
GALAH-DR3-like population in the channel's canonical schema, injects exactly 15
Na-only enhancements across 0.3–0.6 dex, and runs the *real* detection chain
(`to_xh → fit_manifold → zscores → sparse_statistics`). `seti tailings-validate`
writes `results/tailings/validation.json`; `tests/test_tailings_validation.py`
is the CI gate. The synthetic population is not stacked in the pipeline's
favour: its Na residual RMS is **0.065 dex**, inside Griffith's own stated
"RMS residuals ≲ 0.07 dex for well-measured elements", and a test asserts it.

### It failed first, and that is the useful part

At the thresholds this channel originally shipped, recovery was **8/15 at the
pinned seed and 8.5/15 averaged over 10 seeds** — against a 12/15 requirement.
Two rules were **mis-specified, not merely strict**:

1. **`z_flag = 6.0` was calibrated against the wrong population.** Its
   justification was "6σ on an empirical 0.03–0.05 dex width is a 0.2–0.3 dex
   excursion" — but that width came from the *co-natal intrinsic scatter*
   literature (Bovy 2016; Cheng 2020; Patil 2022; Casamiquela 2021). The scatter
   the z-score is actually divided by is the **survey residual**, ~0.058 dex for
   Na. At that width 6σ = **0.35 dex**, which sits *above the bottom third of
   Griffith's published 0.3–0.6 dex range*. The threshold excluded part of its
   own validation target by arithmetic. → **5.0**, i.e. 0.29 dex.
2. **`max_quiet_excess` was an absolute count.** With ~19 elements, ~0.9 exceed
   `z_quiet = 2` by chance, so ~1 in 5 genuine sparse anomalies was relabelled
   DENSE by noise in the elements that are *not* anomalous. It was the one rule
   that **penalised the channel for better data** — more elements measured meant
   a strictly harder test. → replaced by a Poisson rate,
   `quiet_excess_allowance = expected + 2√expected`, with the old constant as a
   floor. It yields 3 at 19 elements and scales properly (2 at 12, 4 at 30).

### And widening it broke the negative control — which is the real lesson

A wider quiet budget necessarily weakens the DENSE rule for a star whose
coherent enrichment is only marginally resolved. Measured: the dense-control
leak (all of O-through-Ni raised 0.40 dex together, which **must** be rejected)
went from 0.10/15 to **0.60/15**. Two changes pay that back:

* **Sign coherence.** The Poisson budget is derived for *noise*, and noise is
  sign-symmetric; a coherent enrichment is not. Background excesses that share
  the sign of the flagged element are now counted separately against a one-sided
  allowance (`coherent_excess_allowance`).
* **`family_max_mean_z` 2.0 → 1.5.** A nucleosynthetic family whose siblings
  average 1.5σ is already leaning coherently — this is the "dense caught early"
  veto and 2.0 was lenient.

### Where it lands

| Configuration | Griffith recovery | Dense-control leak |
|---|---|---|
| Superseded (`z_flag` 6.0, flat quiet budget 1, family 2.0) | **8.5/15** (fails) | 0.10/15 |
| Recalibrated, family veto left at 2.0 | 13.0/15 | 0.60/15 (fails) |
| **Shipped** (`z_flag` 5.0, rate budget, family 1.5) | **12.3/15** mean; **14/15** at the pinned seed | **0.20/15** mean; **0/15** at the pinned seed |

False-positive rate on the un-injected population: **6.7 × 10⁻⁴** (≈ 4 per 6,000
stars), up from 2.8 × 10⁻⁴. That is the price and it is paid knowingly.

**This is a recalibration, not threshold-shopping**, and the distinction has to
survive a referee. Both original values were defective on their own terms — one
calibrated against a population it was not measuring, the other an absolute
count where a rate was required — and both were fixed in the direction the
*physics* dictates, before the recovery number was consulted. The costs are
published rather than hidden: the tests pin the superseded numbers
(`test_the_superseded_thresholds_are_what_failed`), assert the dense control and
the false-positive bound, and re-derive the leak, so the change cannot be
quietly reverted or quietly widened further.

**The residual miss is honest.** The one star still missed at the pinned seed is
`amplitude_below_threshold` — injected near 0.30 dex, against a 5σ ≈ 0.29 dex
cut. That is a sensitivity limit, not a rule defect, and it is the correct
behaviour: the channel says where it can and cannot see.

## 8. The first real-data run, and what the rate actually says

The first run to reach data (GALAH DR4 **73,820** cool dwarfs + APOGEE DR17
**137,047** = **210,867** stars) returned **4,514 sparse and 2,100 vetted
survivors — 1.0% of the sample.**

**That number refutes itself as a discovery, and saying so is the result.**
Griffith et al. 2022 found **15** single-element Na stars in **82,910** GALAH
stars: a rate of **1.8 × 10⁻⁴**. TAILINGS is flagging **~55× more**. Griffith's
own contamination budget for large residuals is ~40% physical / ~60% data
problems. An artificial photospheric marker in one star in a hundred is not a
detection; it is the **systematics floor of the statistic**, and the channel's
honest primary output is therefore not a candidate list but the curve of
*surviving rate versus threshold* (`threshold_sweep`, written per survey as
`results/tailings/threshold_sweep_<survey>.csv`, carrying
`rate_over_griffith` explicitly so a count can never be quoted without its rate).

### The population is non-Gaussian tail, not chance

Under a **pure-noise null** — independent Gaussian z per element, 20 elements
per star — the shipped rules produce **exactly zero** sparse candidates. Not a
small number: zero. So the 1,341 GALAH sparse candidates are **entirely
non-Gaussian tail**. Nothing about them is explained by counting statistics;
they are the survey's error model.

### And the element identity says the same thing

Of GALAH's 1,341 sparse candidates, **K carries 528 and Y carries 249 — 58% in
two elements**, both few-line species with the largest intrinsic measurement
scatter (Bedell: 0.03–0.06 dex for few-line species against ~0.01 dex for
well-measured ones). A real refining signature has no reason to prefer
potassium and yttrium; a measurement floor has every reason to.

### The uncomfortable part: tightening alone does not fix it

Measured on the injection harness, recovery of the Griffith 15 against the
false-positive rate:

| `z_flag` | Griffith recovery | FP rate | × Griffith rate |
|---|---|---|---|
| 5.0 (shipped) | **12.3/15** | 6.7 × 10⁻⁴ | 3.7 |
| 6.0 | 10.7/15 | 3.9 × 10⁻⁴ | 2.2 |
| 7.0 | 7.8/15 | 2.5 × 10⁻⁴ | 1.4 |
| 8.0 | 5.0/15 | 1.4 × 10⁻⁴ | 0.8 |
| 10.0 | 1.5/15 | 0 | 0 |

**No amplitude threshold satisfies both constraints at once.** The threshold
that reaches the published rate (≈8) recovers only a third of the positive
control; the threshold that recovers the control sits ~4× above the published
rate — and that is on *synthetic Gaussian* data. On the real data the rate is
1.0 × 10⁻², another **15× worse**, because real residual tails are far heavier
than the harness models.

The conclusion follows and must not be softened: **the Griffith-scale population
lies inside the systematics population, not above it. Amplitude cannot separate
them.** This is the same structural problem the ISOTHERM channel has with the
190 K debris locus, in a different observable — and the answer is the same in
form: the discriminating power has to come from something orthogonal to
amplitude. Here that is **element identity** (few-line species carry the floor)
and the **instrumental covariate vetoes** (RV, fibre, field, detector position),
not a tighter cut. Any future version of this channel should be built on those
axes, with amplitude as a necessary but grossly insufficient condition.

### Coverage gap: the co-natal test got nothing

`twins: n_pairs: 0` — **no wide binary had both components in the spectroscopic
sample.** S15 was not tested at all. This is a *coverage* statement, not a null:
the differential co-natal test is the one measurement that would break the
degeneracy above, because a pair of co-natal stars cancels the manifold
systematics that dominate the single-star statistic. The engulfment-ceiling
thresholds it would apply are unchanged and still stated in §3.6: **>0.3 dex in
a G dwarf is unexplainable by engulfment at any mass, and a single-element
excess is unexplainable at any amplitude.** Getting pairs into the sample —
by cross-matching El-Badry's catalogue against GALAH/APOGEE *before* the
cool-dwarf cut rather than after — is the highest-value next step.

### Acquisition provenance

`DEGRADED_SOURCE (GALAH->GALAH_DR4_allstar_cloud)`: the pull fell back from the
canonical Data Central path to the `cloud.datacentral.org.au` mirror, which was
the only GALAH host that answered (the `_dc` and `_flat` variants all 404). The
file is the genuine `galah_dr4_allstar_240705.fits` (757,998,720 bytes), so
"degraded" here means *route*, not content. The 73,820 stars used against DR4's
~917,588 spectra is the intended cool-dwarf selection chain (Teff 4000–6000 K,
log g > 4.0, SNR > 40, [Fe/H] > −1), reported stage by stage in
`provenance.json` under `stage_counts` (`rows_in_file` → `rows_after_selection`
→ `rows_normalised`).

## 9. Stated limitations

* **An iron-only anomaly is invisible by construction.** The manifold
  conditions on [Fe/H], so a star whose only anomaly is iron is absorbed into
  its own predictor. This is the price of working in [X/H] rather than [X/Fe],
  which is itself non-negotiable: an error in the star's own [Fe/H] otherwise
  propagates into every element at once and smears a sparse anomaly into a weak
  dense one.
* **The convective-envelope mass for a K or M dwarf is modelled, not cited.**
  The quantitative chain runs Xiang (envelope mass fraction 10⁻⁴ at the
  boundary) → Church (3.45 × 10⁻³ M☉ at a solar-metallicity turnoff) → Michaud
  (10⁻⁷–10⁻⁶ M☉ required for large anomalies). No source on hand gives the K/M
  dwarf value directly, and the `M_cz(Teff)` tabulation in `twins.py` is a
  coarse standard-model interpolation good to a factor of ~2. It is validated
  externally at one point: applied to bulk metals it reproduces Church's
  published 0.128 dex for 5.2 M⊕ of rock at that envelope mass, exactly.
* **No closed-form diffusion-versus-mixing timescale criterion is quoted**,
  because none was found in the fetched corpus. The literature expresses the
  criterion as a *mass* (mixed mass, envelope mass fraction), and that is how it
  is used here.
* **Two elements at once is allowed, and Gao et al. (arXiv:1804.06394) is the
  nearest natural false positive**: Al and Si trends surviving NLTE in M67 while
  every other element flattens. Two elements, partly attributed to modelling.
  A two-element candidate is therefore weaker than a one-element one, not
  stronger, and the report ranks them accordingly.
* **NLTE and 3D line-formation error is the real adversary**, because it is
  single-element by construction. Non-LTE corrections for Al I 3944/3961 are
  *"significantly large (0.3 < Δ < 1.0 dex depending on Teff)"* and reach
  0.4 dex for Ti. No candidate can be believed without an NLTE audit of the
  specific line it rests on.

The question changes; the write-up waits for a detection.
