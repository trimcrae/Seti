# CENOTAPH — the cold-Dyson search in the unprobed T < 100 K regime

*An empty tomb: a monument for a body that is elsewhere.*

## 1. The claim

Every executed Dyson/waste-heat search lives between roughly **100 K and
1000 K**, and is instrumentally capped by WISE W4 at 22 µm:

| Search | Stated temperature range |
|---|---|
| Carrigan 2009 (arXiv:0811.2376), IRAS LRS | ~100–600 K |
| Project Hephaistos I (arXiv:2201.11123) | verbatim: "emit thermal waste-heat in the **100 – 1000 K** range" |
| Huang, Tao & Zhang 2026 (arXiv:2601.07297) | "blackbody waste heat temperatures **T = 150–600 K**" |

Meanwhile **Ćirković & Bradbury 2006** (New Astronomy 11, 628;
doi 10.1016/j.newast.2006.04.003; astro-ph/0506110) argued on
Landauer/Brillouin grounds that postbiological computation prefers a **cold
reservoir**, and noted the "considerable difference in practical observational
terms whether one expects a Dyson shell to be close to a blackbody at **50 K**,
as contrasted to a blackbody at **300 K**."

**The predicted regime and the searched regime do not overlap.** Wien peaks:
100 K → 29 µm, already past WISE's longest band; 50 K → 58 µm; 30 K → 97 µm.

### 1.1 The mid-infrared route to the cold regime is closed by instrumentation

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
3. **The far-IR closure leg**, which no Dysonian search has ever run. Carrigan
   *deliberately discarded* IRAS 60/100 µm for cirrus and zodiacal light while
   noting an ideal instrument "would need to cover wavelengths from 3 to
   100 µm"; Ĝ II used IRAS 60/100 and Planck 857 GHz only as a dust-
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
810, 23), the extragalactic ancestor: galaxies underluminous at fixed rotation
velocity in the Tully-Fisher relation, ≤0.3 % Type III — a population-scale
*attenuation* argument, but for whole galaxies and with no per-object energy
budget. *Annis 1999* (JBIS 52, 33) placed the original limit on star-fed
Kardashev III civilisations from the same underluminosity logic.
*Wright 2023* (arXiv:2309.06564) finds "for 'complete' Dyson spheres we expect
**optical depths of several**" — an optically thick, grey occulter, which is the
physical justification for leg 1. *Blain 2024* (arXiv:2409.11447) names this
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
