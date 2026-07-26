# CENOTAPH — the cold-Dyson search in the unprobed T < 100 K regime

*An empty tomb: a monument for a body that is elsewhere.*

## 1. The claim

Every Dyson/waste-heat search executed **at stellar scale and survey scale**
lives between roughly **100 K and 1000 K**, and is instrumentally capped by
WISE W4 at 22 µm:

| Search | Bands | Stated temperature range | Scale |
|---|---|---|---|
| Carrigan 2009 (arXiv:0811.2376) | IRAS LRS 8–22 µm | "blackbody temperature region **100 < T < 600 K**" | 1,527 sources |
| Ĝ / G-HAT II (arXiv:1408.1134) | WISE | fiducial **T_waste ≈ 285 K** | — |
| Hephaistos I (arXiv:2201.11123) | WISE | "emit thermal waste-heat in the **100 – 1000 K** range" | G ≤ 21 |
| Hephaistos II (arXiv:2405.02927) | Gaia+2MASS+WISE | "**100 to 700 K** to align with WISE's infrared detection capabilities" | 5 × 10⁶ → 7 candidates |
| Huang, Tao & Zhang 2026 (arXiv:2601.07297) | WISE W3/W4 | "**T = 150–600 K**" | 2MRS galaxies |
| Contardo & Hogg 2024 (arXiv:2403.18941) | W1/W2 only | (ML residual, no explicit T) | 4,898,812 → 53 |

**Two honest exceptions, both tiny, and one near-miss.** A flat claim that
nothing beyond 22 µm has ever been used would be wrong:

* **Slysh 1985** (in Papagiannis ed., Reidel, p. 315) fitted four IRAS filters
  *including 60/100 µm*, considering 50 K < T < 400 K, and reported six sources
  at 85–350 K. Carrigan notes "it is not clear how large a fraction of the IRAS
  data set was searched." **This is the one genuinely unresolved case** — the
  paper is not online and could not be verified directly; it warrants a
  library-grade check before any publication.
* **Timofeev, Kardashev & Promyslov 2000** (Acta Astronautica 46, 655) fitted
  all four IRAS filters, finding ~100 sources near **115 K** and 285 K — but
  drawn only from the **3,000 brightest** IRAS sources.
* **Lacki 2016** (arXiv:1604.07844) reached ~2.7–6 K using Planck PCCS2 at
  353/545/857 GHz — but for *extragalactic* "blackboxes", galaxy-filling
  artificial dust, explicitly not stellar Dyson spheres.

So the corrected claim is narrower and still decisive: **at stellar scale, the
temperature floor of the entire field is ~100 K, and no far-IR search has ever
been run at survey scale.** Six sources and the 3,000 brightest IRAS objects do
not constitute a census.

### 1.1 The gap was identified and then not filled

**Lacki 2016 named this exact resource, in the conditional, and nobody took
it up:** *"That leaves a rather large window around FIR-emitting temperatures of
tens of K… the best extant survey is that of AKARI's Far-Infrared Surveyor
(Kawada et al., 2007), which **could** rule out 30 K Chilly Ways… out to about
200 Mpc."* Proposed 2016; never executed. A purpose-built arXiv full-text query
for `AKARI + technosignature` returns **zero entries**
(`results/necrolit/arxiv_q_akari_fis_technosignature.atom`).

**Worse — the standard pipelines would actively *reject* a cold Dyson sphere.**
Carrigan argued explicitly *against* using the far-IR: *"relying on information
from the 60 to 100 micron region may obscure an evaluation of a source in the
Dyson Sphere regime covered by the F[12] and F[25] micron filters."* And
Wright's thermal-IR white paper (arXiv:1907.07829) states that technosignatures
are *"distinguished from dust by their **lack** of far-infrared emission."*
A cold, far-IR-bright, mid-IR-faint object is not merely missed by the existing
corpus; it is the thing the existing corpus is built to throw away.

### 1.2 The theoretical motivation, stated accurately

**Ćirković & Bradbury 2006** (New Astronomy 11, 628;
doi 10.1016/j.newast.2006.04.003; astro-ph/0506110) is the motivating argument,
and it must be quoted for what it actually says. It cites **Landauer (1961)**
and **Brillouin (1962)** explicitly, giving the Brillouin inequality
`I_max = ΔE/(k_B T ln 2) = 1.05 × 10¹⁶ · ΔE/T`, and states that
*"computation becomes more efficient as the temperature of the heat reservoir
in contact with the computer is lower."* From this it advances the **migration
hypothesis**: *"ATCs will tend to move their computing facilities toward the
colder regions of the Milky Way"*, anchored on molecular-cloud interiors at
**T ~ 10 K**.

*It does not contain the strings "50 K" or "300 K".* An earlier draft of this
document attributed a "50 K versus 300 K" contrast to it; that quotation is not
in the paper and has been removed. The physical argument — colder reservoir,
more computation per joule, therefore a preference for low temperatures —
stands on the Landauer/Brillouin material that *is* there, and the specific
temperatures searched here (30–100 K) are ours, chosen by what the far-IR
all-sky catalogues can actually reach, not by a quotation.

**The predicted regime and the searched regime do not overlap.** Wien peaks:
100 K → 29 µm, already past WISE's longest band; 50 K → 58 µm; 30 K → 97 µm.

### 1.3 The mid-infrared route to the cold regime is closed by instrumentation

This is the structural argument, and it is the reason the channel exists rather
than being a refinement of an existing one. The blackbody temperature whose
Wien peak falls in each WISE band is

| Band | λ | Wien-peak temperature |
|---|---|---|
| W1 | 3.35 µm | 865 K |
| W2 | 4.60 µm | 630 K |
| W3 | 11.56 µm | 251 K |
| W4 | 22.09 µm | 131 K |

A survey is efficiently sensitive only near its own Wien ceiling. Every WISE
catalogue that has got *deeper* since 2010 — NEOWISE-R, CatWISE2020
(1.89 × 10⁹ sources, 1.7 mag deeper, "at 3.4 and 4.6 µm"), the unWISE coadds —
is **W1/W2 only**. W3 and W4 depth is frozen at the 2010 cryogenic phase of
WISE and cannot improve until a new mid-infrared all-sky survey flies.

So the two largest waste-heat searches ever run — Contardo & Hogg's 4.9 × 10⁶
stars and anything built on CatWISE — are structurally incapable of reaching
100–300 K, and **making them deeper makes them warmer, not colder**. Below
~130 K the infrared route is closed by instrumentation, not by effort. That is
the argument for attenuation plus far-infrared recovery as the only way in.

## 2. The signature — three legs, all required

This is an **energy-conservation test**, not a photometric anomaly hunt.

**Leg 1 — grey attenuation.** An occulter of covering fraction `f` dims the
star by `Δm = −2.5 log₁₀(1−f)` **equally in every band**; interstellar dust does
not (`A_band/A_V` spans ×88 from GALEX NUV to WISE W2, ×38 over the bands
actually used). Grey and reddening are near-orthogonal vectors in magnitude
space, so extinction is **fitted jointly, never assumed zero**.

**Leg 2 — no mid-IR excess** in W1–W4. This is precisely what makes such an
object invisible to Hephaistos and to every other executed search. It is
measured here from the photometry, *not* inherited from a published excess
catalogue: Silverberg et al. 2018 find at most 7.9 % ± 0.2 % of AllWISE-selected
infrared excesses are good disk candidates, and all 13 of Theissen & West's
W4 S/N > 3 candidates are false positives. Inheriting an excess catalogue would
import a ~92 % error rate.

**Leg 3 — far-IR recovery** at 60–160 µm carrying the intercepted `f·L`. This
is the crux, and it converts "missing energy" into "found the energy where
nobody looked". Datasets: **AKARI/FIS Bright Source Catalogue** (all-sky,
65/90/140/160 µm, ~427 k sources), **IRAS PSC/FSC** 60/100 µm, Planck 353–857 GHz.

### 2.1 The closure ratio is the diagnostic

Energy conservation forces, for an **isotropic** occulter,

```
f_IR ≡ L_IR/L_star  =  f_dim ≡ 1 − 10^(−0.4 g)
```

independently of geometry or optical depth, because every photon the star emits
either escapes or is thermalised and re-emitted. Define `ρ = f_IR / f_dim`:

| ρ | Interpretation |
|---|---|
| ≈ 1 | isotropic occulter — the energy is found where nobody looked |
| ≪ 1 | an **edge-on disk**, which blocks the sightline while intercepting only its own small solid angle (the #1 astrophysical confounder) — **or** the extraordinary case where the energy is not thermalised |
| > 1 | the far-IR flux is not all from this star: cirrus, a blend, or a background galaxy in the beam |

This directly answers a caveat Zackrisson et al. state and cannot resolve: their
`f_cov` "does not need to represent the global covering fraction of the star,
but simply the obscured fraction of the face of the star facing the observer."
The far-IR leg converts a line-of-sight `f` into a global `f` and tests the two
against each other.

It also answers the **grey-dust** objection, which Zackrisson et al. §5.2
concede as irreducible ("could — at sufficiently high optical depth — reduce the
apparent brightness of a star in a way that would be very similar to the
expected signatures of a Dyson sphere"). A foreground interstellar grey screen
is heated by the whole interstellar radiation field, so the target star's
contribution to its emission is negligible and its far-IR signature is
*extended cirrus*, not a point source with `ρ ≈ 1`. Their proposed discriminant
(diffuse interstellar bands) was never performed; this one is quantitative and
runs on catalogue data.

### 2.2 Leg 3 is decisive, not decorative — computed

A Sun-like star at 100 pc with `f = 0.1` re-radiating at 50 K delivers **≈6.7 Jy
at 100 µm**, far above the AKARI/FIS 90 µm (~0.55 Jy) and IRAS 100 µm (~1 Jy)
practical limits. The detection horizon scales as `√f`:

| f | T | radius (AU) | Wien peak | AKARI 90 µm horizon |
|---|---|---|---|---|
| 0.10 | 50 K | 62 | 58 µm | ~370 pc |
| 0.05 | 50 K | 62 | 58 µm | ~260 pc |
| 0.20 | 30 K | 172 | 97 µm | (see `summary.json`, `far_ir_horizons`) |

`budget.coverage_table()` emits the full grid at run time. Outside the horizon a
non-detection carries no information and the channel says so
(`verdict: far_ir_undecidable`) rather than making a claim.

### 2.3 The honest mass cost of going cold

Area scales as `r²`, so a 50 K shell needs ~16× the material of a 200 K one at
the same `f`. Blain 2024 puts numbers on it: an Earth mass at 1 mm thickness
reaches R ≤ 4.5 AU (T ≥ 140 K); a **Jupiter mass reaches R ≤ 81.3 AU
(T ≥ 33 K)**. Our independent calculation reproduces this to within a factor 2.
So a 30–60 K structure implies a Jupiter-mass-scale build at millimetre
thickness — or ~0.2 Earth masses at 10 µm film thickness. The requirement spans
three orders of magnitude depending on assumed areal density, and that is the
honest statement; it is stated rather than hidden.

## 3. Novelty status — the mandatory Zackrisson engagement, answered

> **Verdict: the underlying idea is Zackrisson's, and he executed a small
> version of it. The novelty claimed here is population scale, the achromaticity
> veto, the far-IR closure leg, and sensitivity to partial covering fractions —
> not the idea.**

**Zackrisson et al. 2018**, "SETI with Gaia: The observational signatures of
nearly complete Dyson spheres" (arXiv:1804.08351; ApJ 862, 21;
doi 10.3847/1538-4357/aac386) states in its abstract: *"A star enshrouded in a
Dyson sphere with high covering fraction may manifest itself as an optically
subluminous object with a spectrophotometric distance estimate significantly in
excess of its parallax distance."* The three required answers, from the full
text (arXiv v2, the accepted version; cached in `results/zacklit/`):

**Q1. Did they EXECUTE a search or only forecast?** **They executed one.** It is
not a forecast paper and a claim of "first execution" would be false. Verbatim
(abstract): *"carry out a limited search for Dyson-sphere candidates in the
sample of stars common to Gaia Data Release 1 and RAVE Data Release 5. We find
that a small fraction of stars indeed display distance discrepancies of the type
expected for nearly complete Dyson spheres."* §3: 2.3 × 10⁵ cross-matched →
**8,441 stars** with adequate errors → 75 at `f_cov > 0.9` → **6 at
`f_cov > 0.7`** → 2 survive an SED check → 1 followed up. They took NOT/FIES,
Magellan/MIKE and Mercator/HERMES spectra of that one candidate,
**TYC 6111-1162-1**, which resolved to an **unseen ~1 M☉ white-dwarf companion**
— and Gaia DR2 then showed the DR1 parallax itself was flawed. Gaia DR1/TGAS
parallaxes were the measured quantity inside the statistic, not an illustration.

**Q2. Do they impose an achromaticity test?** **No.** Greyness is an
*assumption*, stated as such in §2: *"If this obscuring structure blocks the
emerging optical/near-IR light **like a grey absorber (i.e. all wavelengths
throughout this interval are equally affected)**…"*. Extinction is inherited
wholesale as a single borrowed scalar — for the one followed-up candidate,
*"Using A_V = 0.10 following RAVE"*. §5.2 concedes grey dust as an unresolved
confounder and proposes diffuse-interstellar-band spectroscopy as future work:
*"High signal-to-noise spectroscopy of the spectral regions featuring these
bands **may therefore be** a suitable way to further scrutinize potential
Dyson-sphere candidates."* There is no joint grey + reddening fit anywhere in
the paper. **This is the methodological gap CENOTAPH's leg 1 fills.**

**Q3. Do they treat a cold, non-reradiating sphere?** **No — 50–1000 K only,
with a 300 K working case.** §1: *"as long as the Dyson sphere operates at a
temperature in the **50–1000 K** range"*. Their escape route from the mid-IR is
thermodynamic and geometric (non-thermal ν-losses, *"where the energy is somehow
stored rather than radiated away"*, anisotropic emission), **not** a cold,
large-radius sphere whose blackbody peak has moved into the far-IR. They never
model T < 50 K and never compute a cold sphere's SED. The longest wavelength
used anywhere in the analysis is **WISE W3, 12 µm**; W4 (22 µm) is not used at
all. IRAS, AKARI, Spitzer and Herschel are named in exactly one sentence of §1
as an avenue and then never taken; Planck is never mentioned.

**What is therefore genuinely new here:**

1. **Population scale.** 8,441 stars (DR1 × RAVE, which they call "this pilot
   project") versus a target working sample of 3–5 × 10⁶ from Gaia DR3
   GSP-Spec, with LAMOST DR10 / APOGEE DR17 / GALAH DR4 as independent
   secondaries. They forecast ~10⁶ by DR3; **that forecast was never realised by
   anyone**, including them — Zackrisson is a co-author on Hephaistos I, II and
   IV, and his own group moved to infrared excess instead. An OpenAlex/Semantic
   Scholar audit of all ~29 citing works (`results/zacklit/`) finds only two
   that executed a Dyson search at all, Hephaistos I and II, both pure
   IR-excess; neither implements their eq. (5).
2. **The achromaticity veto** (§2 above), which they do not impose.
3. **The far-IR closure leg**, which no Dysonian search has ever run at survey
   scale (see §1 for the two small IRAS exceptions and the Lacki 2016
   near-miss). Carrigan *deliberately discarded* IRAS 60/100 µm for cirrus and
   zodiacal light; Ĝ II used IRAS 60/100 and Planck 857 GHz only as a dust-
   discrimination lever arm. The only published far-IR excess census of normal
   stars is **Liu, Wang & Jiang 2014** (arXiv:1308.5593, AJ 148, 3): Hipparcos
   main-sequence stars × AKARI/FIS → 136 detections → 83 debris-disk candidates.
   Tiny, non-SETI, and used here as the calibration/comparison set.
4. **Sensitivity to partial covering fractions.** Their estimator is
   `f_cov = 1 − (D_trig/D_spec)²`, and they state the limitation explicitly:
   *"`f_cov ≈ 0.1` would correspond to a distance discrepancy of just ≈5 %,
   which is undetectably small. In fact, it takes **`f_cov > 0.75`** to produce a
   factor of >2 discrepancy."* That floor is set by 20–30 % spectrophotometric
   distance errors, i.e. ≈0.4–0.6 mag. Replacing isochrone distances with a
   twin-differential absolute magnitude replaces that error budget with the twin
   scatter (~0.05 mag measured) plus the parallax term. Measured on the
   synthetic population: **`f ≳ 0.29` at `ϖ/σ_ϖ = 20`, `0.18` at 50, `0.15` at
   100** — a factor 2.6–5 improvement, opening the entire *partial*-covering
   regime that their square-root statistic cannot reach.
5. **Their failure mode is our funnel stage.** RUWE, `ipd_frac_multi_peak`,
   `astrometric_excess_noise_sig` and `non_single_star` did not exist in DR1.
   TYC 6111-1162-1 would be screened before follow-up here.

**Other prior art engaged.** *Zackrisson et al. 2015* (arXiv:1508.02406, ApJ
810, 23), the extragalactic ancestor: 1,359 disk galaxies underluminous at fixed
rotation velocity in the Tully-Fisher relation, giving a conservative ≲3 % and a
tentative **≲0.3 %** on Kardashev-III disk galaxies, "10–20 times stronger than
that of Annis 1999" — a population-scale *attenuation* argument, but for whole
galaxies and with no per-object energy budget. *Annis 1999* (JBIS 52, 33)
placed the original limit from the same underluminosity logic on 57 disk
galaxies and 106 ellipticals.
*Wright 2023* (arXiv:2309.06564) finds "for 'complete' Dyson spheres we expect
**optical depths of several**" — an optically thick, grey occulter — and, more
directly useful still, concludes that "we should not expect Dyson spheres to be
'complete', but to provide **a few magnitudes of gray extinction**". A few
magnitudes of grey extinction is precisely leg 1's observable. His paper also
cuts the other way and the tension should be stated: he argues "the optimal use
of mass is generally to make very small and hot Dyson spheres", which is an
argument *against* the cold regime on mass-efficiency grounds. CENOTAPH tests
the cold branch because it is unobserved, not because it is favoured. *Blain 2024* (arXiv:2409.11447) names this
blind spot outright — "a DSM might extinguish its star at optical/near-IR
wavelengths, and thus either not appear or appear anomalously faint in a stellar
catalogue" — but implements no search. *Torlakcik 2026* (arXiv:2607.07781,
"Stellar J-Harvesting") is explicit that its signature "need not produce a
detectable mid-infrared excess", and likewise does not search.
*Contardo & Hogg 2024* (arXiv:2403.18941, AJ 168), the largest search to date at
N = 4,898,812 with a 53-object yield (rate 1.1 × 10⁻⁵), is the structural
**mirror image** of this estimator: they predict expected W1/W2 from optical +
NIR and take the positive residual; we take the achromatic negative one. Their
selection is reused here for comparability (G < 16, Teff 4000–7000 K, RUWE < 1.4).

**Ĝ II explicitly rejected the attenuation channel** on the grounds that the
starlight deficit is linear in α and "only observationally distinguishable for
values of α near 1". That objection is correct *for a statistic with 0.4–0.6 mag
errors and no way to tell grey from dust*. It is answered here by two things
they did not have: an achromaticity discriminant that separates the two
components rather than assuming one away, and a 0.05-mag twin-differential error
budget that moves the "distinguishable" boundary from α ≈ 1 down to α ≈ 0.15–0.3.

**Population gaps this sample occupies.** No LAMOST stellar sample has ever been
used for a technosignature search, despite the LAMOST + WISE IR-excess machinery
existing. And no Dyson search has targeted solar analogues or K dwarfs as a
defined population — Hephaistos II was M dwarfs, Contardo & Hogg is FGK but
W1/W2-limited.

## 4. Estimator — and why the naive route fails

Isochrone `M_pred` from spectroscopic Teff is limited by Teff systematics, not
photon noise: `dM_G/dTeff ≈ −0.0022 mag/K`, so a 100 K scale error injects
0.22 mag — twice the whole signal at f = 0.10. Instead:

**Parameter twins.** For each target, the ≥50 nearest neighbours in
`(Teff, log g, [M/H], [α/Fe])`, measured by the *same pipeline*; the statistic is
`ΔM_Ks = M_Ks(target) − median{M_Ks(twins)}`. Every pipeline systematic that is a
smooth function of the parameters is shared and cancels differentially. Two
properties make the negative tail the clean one:

* **Metal-poor subdwarfs cannot leak in**, because [M/H] is a matching axis, so
  a subdwarf's twins are subdwarfs. Verified in the test suite against a
  synthetic subdwarf locus that is genuinely 0.35–0.8 mag underluminous.
* **Unresolved binaries scatter overluminous** — up to 0.75 mag for an
  equal-mass pair. The overluminous tail is a binary sequence; the underluminous
  tail is not. The channel reports both, and their *ratio* is the null control:
  an occulter population is an excess of one tail over the other, not merely a
  count of 3σ objects.

Two subtleties that would otherwise silently break the error model, both handled:

* **σ is measured about the local linear fit, not about the twin median.**
  Scatter about the median also contains the parameter *gradient across the twin
  box*, which is a deterministic trend, not noise. Using it inflates every error
  bar by (box width × dM/dTeff) — in practice 0.14 mag instead of 0.05 — and
  throws away most of the sensitivity.
* **A parallax error is exactly a grey offset**, and the twin scatter is
  common-mode across bands because every band's residual is built from the same
  reference-band deficit. Both therefore enter the fit as a **rank-1 fully
  correlated** covariance term, not on the diagonal. Treating them as
  independent per band would inflate every significance in the channel by
  ~√N_bands. With this handled, the empirical null on 5,605 synthetic stars has
  robust σ = 0.88, median grey = −0.003 mag, and symmetric tails (11 above +3σ,
  10 below −3σ).

`Ks` is the luminosity band: `A_Ks/A_V = 0.078`, so extinction contributes
< 0.025 mag for `A_V < 0.3` — nearly extinction-immune.

**The fit refuses degenerate band sets.** Ks + W1 + W2 alone span
`R_b = 0.078 → 0.026` and cannot separate grey from reddening at all; the fit
returns `verdict: degenerate_band_set` rather than a number. A blue band is
required.

## 5. Sample

Gaia DR3 `astrophysical_parameters` (GSP-Spec) is the primary source: it
supplies all four twin axes including **[α/Fe]** for ≈5.6 × 10⁶ stars from a
*single* pipeline with no crossmatch attrition. That is not a convenience — the
twin cancellation is exact only within one pipeline. Cuts: dwarfs
(`logg > 3.8`), Teff 4000–7000 K, `parallax_over_error > 20`, `RUWE < 1.4` at
query time and `< 1.2` at vetting so the funnel stays visible, GSP-Spec quality
flags (first 13 ≤ 1), Lindegren et al. 2021 parallax zero-point (with an
explicit `parallax_zp_method` field recording the fallback if the official
tables are unavailable). LAMOST DR10 / APOGEE DR17 / GALAH DR4 are supported as
independent secondaries via `fetch_vizier_spectro`.

## 6. Contamination model

| Confounder | Handling |
|---|---|
| **Background-galaxy confusion in the beam** | The universal killer: every Hephaistos candidate died of it, and JWST/MIRI (Hephaistos IV, arXiv:2607.09460) resolved candidates D and E into a Hot DOG at z≈0.9 and a dusty starburst at z≈0.4, both within ~1″. For a far-IR leg it is **worse**: 25–40″ beams give a 10²–10⁴× larger coincidence area. Handled as funnel stages — `\|b\| > 20°`, measured beam-neighbour counts at G < 18, and an explicit chance-match expectation computed per band and reported in `summary.json`. With 10⁶ targets and a 25″ AKARI beam the chance-match expectation is in the **thousands**, so a positional association is never evidence on its own; only the closure ratio is. |
| **Unresolved binaries** | Zackrisson's own failure mode. Screened on RUWE, `ipd_frac_multi_peak`, `astrometric_excess_noise_sig`, `non_single_star`. The dangerous direction is a **hot** companion (WD/sdB), which biases composite Teff *upward* so the star is compared to hotter, brighter twins and looks underluminous; caught by its GALEX UV excess and by the SED goodness-of-fit, since a two-temperature SED is not grey + R_V = 3.1 reddening. |
| **Metal-poor subdwarfs** | Cancelled by construction ([M/H] is a matching axis); `param_edge` flags stars at the edge of the parameter distribution whose twin cloud is lopsided, and a local linear fit removes the first-order gradient bias. |
| **Teff / [Fe/H] pipeline systematics** | Differential cancellation within one pipeline; secondary samples from independent surveys test the residual. |
| **Parallax zero-point** | Lindegren Z5/Z6 applied; method recorded, never silently assumed. |
| **Anomalous R_V** | The joint fit measures A_V rather than assuming it; `χ²/dof > 6` rejects SEDs not describable as grey + an R_V = 3.1 law. |
| **Large-grain edge-on circumstellar disks** | *The* astrophysical confounder — they **can** grey-dim. Separated by the closure ratio (a disk intercepts only its own solid angle, so ρ ≪ 1) and independently by kinematics: `v_tan ≳ 80 km/s` means a thick-disk/halo star several Gyr old with no business carrying a massive primordial disk. |
| **Variability (low state)** | Gaia scatter proxy `√N · 2.5/ln10 / (F/σ_F)` < 0.05 mag, to be confirmed against ZTF/ATLAS. |
| **Crowding / blending** | Makes stars *brighter*, so it is the safe direction — but a red neighbour blended into BP/RP fakes reddening, so the Riello et al. 2021 corrected excess factor `C*` is checked at 3σ. |
| **Cirrus** | Measured: SFD 100 µm surface brightness is 64.5 MJy/sr in the plane versus 4.2 at `\|b\| = 20–30°` (`results/farir_stats/cirrus_levels.json`), which is what sets the practical far-IR limit and motivates the latitude cut. |

## 7. Honest limitations

* Per-star sensitivity is `f ≳ 0.15–0.29` depending on parallax quality — worse
  than the `f ~ 0.01` that warm IR-excess searches reach, but those reach it
  **only for 100–1000 K re-radiation**. The two regimes barely overlap:
  CENOTAPH owns cold and dark occulters, Hephaistos owns warm ones.
  Complementarity, not a sensitivity claim, is the argument.
* A *complete* (f → 1) shell has no optical counterpart and is outside this
  sample by construction, since Gaia astrometry and a spectrum are required.
  The channel is sensitive to **partial** covering, f ≈ 0.15–0.6.
* The far-IR leg only rules on stars **inside** the horizon for their f, L and
  assumed T. Outside it, a non-detection is reported as
  `far_ir_undecidable` and no claim is made.
* Leg 3 assumes a blackbody re-radiator. A greybody with ε(λ) < 1 in the far-IR
  would be warmer and fainter there, so the quoted horizons are upper bounds.

## 8. Outputs

`results/cenotaph/summary.json` carries the verdict, funnel counts at every
stage, the empirical null (including the over/underluminous tail asymmetry), the
computed sensitivity floor, the chance-coincidence budget, per-test vetting
coverage with missing columns named, the WISE Wien ceilings, and the far-IR
detection-horizon grid. `candidates.csv` carries survivors with every
intermediate quantity that produced them.

Run: `python -m seti.cli cenotaph --synthetic` offline, or dispatch
`.github/workflows/cenotaph.yml` for the archive run.

## 9. Acquisition — what the Gaia archive actually does, measured

The first archive run (30203250183) reported `NO_DATA_REACHED, n_sample: 0,
"the parent-sample query returned nothing"`. **That was false.** The ADQL, the
TAP endpoint, the column names, the join and the cuts were all correct, and the
run returned 703,555 rows before dying. The failure was entirely in transport
and reporting. The probe (`--stage probe`,
`.github/workflows/cenotaph-probe.yml`, run 30209664654) measured what is
really going on; `results/cenotaph/probe.json` is the record.

**1. ESA's anonymous async result store is full.** astroquery reports only
`HTTP 500 Cannot find result 'result' for job <id>. Path does not exists:
/gaia_netapp/tap-server/storage/O/anonymous/…`, which reads like a server bug.
The same query through pyvo returns the actual message:

> `Filesystem quota exceeded for user anonymous (Currently using 200 GB,
> increasing it with 128 KB exceeds allowed quota)`

The job executes; the server then cannot write its result file, so retrieval
500s. Retrying cannot help — the old ladder spent ~75 of its 110 minutes doing
exactly that. But it is **not** permanent: the quota read 200 GB and 199 GB in
two attempts seconds apart as other users' jobs expired, and a small `COUNT(*)`
result *did* get through on async. Hence a cooldown rather than a ban, and a
preference for small results, which are the ones that can still fit.

**2. The "row cap" on the sync endpoint is a time cut.** The failing run got
*exactly 8193* rows for the [2, 2.5) mas shell; the probe re-ran the identical
query and got *16385*, against a `COUNT(*)` of **199,572**. Same query, same
columns, two different powers of two plus one — so this is not a fixed `MAXREC`
but the response being cut mid-stream, the parser recovering whole buffered
blocks. pyvo's sync endpoint states the limit outright: `Maximum execution time
(60 s) reached. Job aborted.`

The consequence for design is that the fix is **smaller queries, not a bigger
`TOP`**. Each parallax shell is counted once with `SELECT COUNT(*)` built from
the *same* `WHERE` text as the `SELECT`, then pre-split on the indexed,
uniformly distributed `random_index` into slices of ~15,000 rows (well inside
the server's window, and small enough that an async result may still fit the
free quota). `random_index` rather than `MOD(source_id, k)` because the former
is indexed and the latter would force a full scan. The last slice is always
left open-ended so a wrong upper bound cannot drop the catalogue tail. A slice
that still comes back short of its expectation is re-counted exactly and split
again — suspicion first, then proof, so a Poisson fluctuation is never called a
truncation nor a truncation a fluctuation.

**3. Failure is contained per slice, and every count is reported.** A shell or
slice that cannot be fetched is a named hole in the sample, not a reason to
discard the ones that worked. `summary.json` carries the full acquisition
ledger — query text, transport, rows returned and `COUNT(*)` expected at every
stage, plus a per-shell reconciliation — on success as well as failure.

**The four verdicts are never merged**, because they call for opposite actions:

| Verdict | Meaning | What it implies |
|---|---|---|
| `NO_DATA_REACHED` | no query executed anywhere | the archive is unreachable; nothing is known |
| `QUERY_RETURNED_ZERO_ROWS` | the archive answered, and the cuts matched nothing | a statement about the **selection** — check signs, units, bounds |
| `QUERY_TRUNCATED` | the server capped the result below its own `COUNT(*)` | sub-split and refetch; the sample is **not** whole |
| `PARTIAL_SAMPLE` | some chunks arrived, some did not | usable, with the hole and the completeness fraction stated |

Reserving `NO_DATA_REACHED` for its literal meaning is the point. Reporting a
valid-but-empty query as an unreached archive hides a selection bug; reporting
an unreached archive as an empty result invents a null. Neither is allowed, and
the offline suite has a test for each.
