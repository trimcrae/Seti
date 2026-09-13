# NECROFRONTIER — the residue of a non-biological successor

*Added 2026-09-13. The user's question: find exhaustive new ways to search for
necrosignatures — signs that a non-biological intelligence wiped out a
biological population — with new techniques, new data sources, and places
nobody has looked. This document is the answer: a sharpened physical
prediction (§1), eighteen new signatures S46–S63 with their data, statistic,
contaminants and prior-art position (§2), a ledger of public archives this
repository has never touched (§3), and the build order (§4). The prior-art
sweep `necrofrontier-lit.yml` and the data-source probe
`necrofrontier-probe.yml` are dispatched from `main`; nothing below is a
measurement of the sky until their results land in `results/`.*

*Provenance. Twelve parallel literature agents (2026-09-13) established the
positions quoted here from search-engine records and from full texts already
cached under `results/*lit*/`. The sandbox blocks every scholarly host, so
arXiv identifiers marked "verify" below are checked by the runner sweep's
`id_title_check.json` before any of them is cited in a manuscript; a wrong id
has cost this repository a citation before (`docs/baffle.md` §4). Three
agents (terrestrial record, in-situ dust, post-biological literature) were
stopped before reporting; their groups are covered by the sweep's g9, g14, g15
and the positions in §2 for S56, S58 and S62 rest on the other agents'
partial coverage and are marked accordingly.*

---

## 1. What "an AI killed them" predicts that "they died" does not

`docs/necrosignatures.md` maps *extinction* onto residue: cessations,
deficits, sparse anomalies. Most of those residues are agnostic about the
killer. The user's question is narrower, and the narrowing is what generates
new observables. If the extinction agent was a non-biological intelligence
that persists, then unlike every other Great-Filter outcome **the system is
not empty after the death**. Six consequences follow, each with a different
observable class, and each falsifiable:

| # | Prediction | Why it follows | Observable class | Signatures |
|---|---|---|---|---|
| P1 | **Purity.** A machine successor's substrate and tailings are *refined* — isotopically, elementally, molecularly — in ways no condensation sequence, nucleosynthetic source, fractionation or biochemistry produces. | Computing substrates are isotopically purified (²⁸Si, ¹²C); fusion consumes D and ⁶Li; metallurgy makes alloy vectors; fluorine chemistry makes molecules with several F atoms that life and astrochemistry both avoid. | Isotope ratios in solids; abundance vectors; molecular lines | S46, S51, S53, S54, S58 |
| P2 | **Relocation.** A successor that needs no habitable zone moves its activity to wherever energy per unit mass is cheapest: the sublimation radius, compact remnants, cold sinks. The cradle is abandoned. | The 100–1000 K window of every waste-heat search is the biological comfort band; a machine has no such preference. | Excess at ~1500 K; rings at 300–700 K around WD/NS; technosignatures at hosts where biology is impossible | S47, S63 |
| P3 | **Transition.** The takeover is fast — decades — so its resource-acquisition phase is a *rising* signature in decade-baseline archives, not a steady state. | Exponential growth of a replicating industry; the only comparable natural transients (impacts, novae) rise in < 1 yr then decay. | Monotonic mid-IR rise; growing transit depth | S57, S61 |
| P4 | **The weapon.** Some extinction modes leave physical scars: a disassembled or shattered planet, a sterilised star, a stripped atmosphere. | Planetary destruction produces warm debris at the planet's orbit; stellar manipulation produces flares above the spot-energy ceiling; atmosphere removal produces airless planets above the cosmic shoreline. | Debris at HZ temperature above f_max; superflare energy excess; bare rock where retention is predicted | S52, S59, S62 |
| P5 | **Persistence with machine habits.** A persisting successor communicates node-to-node with beams, not broadcasts; it may survive at century timescales where nothing biological does. | Point-to-point links are interceptable only by geometry; century plate archives are the longest baseline we hold. | Star-pair spillover geometry; century light curves | S50, S60 |
| P6 | **The archive at home.** The one place a Gyr-old artifact survives near us is a lunar cold trap; the one place a prior technological extinction on *this* planet is recorded is the sedimentary column. | Regolith turnover is metres per Gyr in permanent shadow; fission products and refined particulates survive diagenesis. | Anisothermal hot components in PSRs; fission vectors at extinction boundaries | S55, S56 |

Two things the framing does **not** predict, stated so they are not
searched: a persisting successor need not radiate in the 100–1000 K band at
all (P2), so the Dysonian corpus's null is uninformative about it; and a
successor that *left* (upload-and-depart) is indistinguishable from plain
extinction, which is the existing taxonomy's territory.

---

## 2. The signatures

Each entry: **claim** — what the residue physically is; **data** — the public
archive, at what scale; **statistic** — the test; **kills** — the contamination
ledger; **record** — the prior-art position established by the agents (to be
confirmed by `necrofrontier-lit`); **verdict** — build priority under the
charter (novelty first, then scale, never a null write-up).

### XII. Purity residue (P1)

**S46 The impossible grain — isotopic purity in the Presolar Grain Database.**
*Claim.* Natural solids carry isotope anomalies only as *correlated
nucleosynthetic packages* (a supernova X grain is ²⁸Si-rich **and** carries
⁴⁴Ti, ²⁶Al/²⁷Al ~ 0.1–0.6, ¹⁵N, ⁴⁹Ti) or as per-mil mass-dependent
fractionation. Technology makes mono-isotopic material: quantum-grade ²⁸Si is
99.99 %+ pure (δ²⁹Si ≈ δ³⁰Si ≈ −999 ‰), ¹²C diamond, separated ⁶Li/⁷Li,
enriched actinides. The supernova end-member reaches δ²⁹Si ≈ −650, δ³⁰Si ≈
−1000 (Lin et al. 2010, ApJ 709, 1157, the Si/S zone) but always with its
package. A grain at δ²⁹Si ≈ δ³⁰Si ≲ −950 ‰ with **solar** C, N, Al–Mg and
Ca–Ti is a refined product; so is a ¹³C-free carbon grain with solar N, or a
grain with a large *mass-independent* anomaly in one element and none in its
nucleosynthetic partners.
*Data.* The Presolar Grain Database, SiC release (Stephan et al. 2024, ApJS
270, 27): **20,230 grains** with δ²⁹Si, δ³⁰Si, ¹²C/¹³C, ¹⁴N/¹⁵N, ²⁶Al/²⁷Al,
⁴⁴Ti/⁴⁸Ti, Ca, Mo, Sr, Ba where measured; a DOI'd IEDA/EarthChem download,
not an API. Graphite release 2024 (MetSoc abstract); oxides/silicates pending.
Meteoritical Bulletin "ungrouped/anomalous" classifications as a second
frame.
*Statistic.* Per grain: the distance in (δ²⁹Si, δ³⁰Si) from the CCSN mixing
line, and a *partner test* — the probability under the X-grain package that
the measured ⁴⁴Ti/⁴⁸Ti, ²⁶Al/²⁷Al and ¹⁴N/¹⁵N are all solar given the Si
purity. Candidates: purity beyond every catalogued X grain **and** partner
signals absent at ≥ 3σ each. The purity floor is set empirically by the
database's own most extreme classified X grain, not by a model.
*Kills.* SIMS/NanoSIMS precision (10–50 ‰) and instrumental fractionation;
grains with only Si measured (no partner test possible → `INSUFFICIENT_PANEL`,
never a candidate); the new type "D" and the unclassified residue, which are
exactly where a hit would hide and where measurement problems concentrate;
terrestrial/spacecraft contamination (a ²⁸Si-pure grain is *also* what a
silicon wafer fragment looks like — the C/N/Al isotopes of a real SiC grain
decide).
*Record.* No proposal of stable-isotope purity in solids as a technosignature
was found. Nearest: Whitmire & Wright 1980 (fission waste in stellar
photospheres, element pattern); Catling, Krissansen-Totton & Robinson 2025
(ApJ 979, 137, arXiv:2411.18595 — D/H depletion by fusion in planetary water);
Ellery 2025 (arXiv:2510.00082 — Th/Nd, Th/Ba ratios from lunar reactors).
*Verdict.* **Build first.** Novel, executable offline once the table is
fetched, 2×10⁴ objects, a clean natural background with a *correlated*
signature the artificial case lacks by construction. Sub-note: the stellar
analogue (Mg, Li, Ti isotope ratios in dwarfs) has no survey-scale catalogue
(Yong et al. 2003, 61 stars; Lind et al. 2013 — no ⁶Li in 3D-NLTE) and is
listed, not built.

**S51 The refinery vector — polluted white dwarfs beyond the natural family.**
*Claim.* A WD photosphere is a mass spectrometer of what fell in. Huang, Tao
& Zhang 2026 (arXiv:2605.29811) tested one artificial template (a dense
siderophile concentrate) against a meteorite mixture on PEWDD. What remains
untested is the *process-orthogonal pair residual*: natural processes move
element ratios along four levers — condensation temperature, metal–silicate
partitioning, melt incompatibility, photospheric sinking — and element pairs
degenerate in all four (Ti/Al, Sc/Ca, Ca/Al, Sr/Ca, Mn/Cr, Ni/Co) stay within
~0.3 dex across chondrites, achondrites, mantles, crusts and cores. A refinery
does not respect them. Also: alloy vectors (Ti/Al/V without Fe), pure Si
without O/Mg, Cu/Zn/Sn/Pb enrichment, Be without the Li/B spallation partners,
and Fe/Mg *below* the most depleted natural mantle.
*Data.* PEWDD (Williams et al. 2024, A&A 691, A352; VizieR J/A+A/691/A352;
GitHub `jamietwilliams/PEWDD`): 1,739 WDs, 24 metals, provenance per row;
87/697 Ca-referenced records carry ≥ 5 of the nine common elements; GD 362
all nine plus Sc, Ti, V, Co, Cu, Sr. Hollands cool-DZ tables; Kaiser 2021/2024
Li/K/Na objects; Klein 2021 Be objects; Rogers 2024 gas-disc sample.
PyllutedWD (`andrewmbuchan4/PyllutedWD_Public`) as the natural forward model.
*Statistic.* Tier 1: posterior-predictive misfit p of the best natural model
(meteorite mixture ∪ PyllutedWD parameter space: core/mantle/crust fraction,
condensation temperature, differentiation pressure, fO₂, accretion phase) per
object — the calibrated "unexplainable" list nobody has published. Tier 2:
|z_pair| > 4 for ≥ 1 orthogonal pair while the rest of a ≥ 5-element panel is
natural, with σ_sys = 0.15 dex and both accretion-phase corrections.
Injection calibration from meteorite draws and PyllutedWD forward draws.
*Kills.* Cool-He line physics (Ca I / Na I / K I / Li I broadening — the
Li/K objects are exactly where models are weakest); asynchronous accretion
(Brouwers 2022: time-variable core/mantle and ice fractions within one event);
exotic natural mantles (Putirka & Xu 2021) and carbonate crusts (SDSS
J1043+0855); the 1–2-element records that dominate Huang's low-likelihood
tail (`INFORMATION_LIMITED`, never a candidate).
*Record.* Huang 2026 is executed prior art for the siderophile template; the
pair test, the trace-element panel (Sc, V, Co, Cu, Sr, Zn, Be, Li, K) and a
calibrated misfit list are not. Literature outliers to run first: PG 1225−079
(Ca/Sc/Ti/V ×2–3 at bulk-Earth Fe-peak, "no single meteorite"), LHS 2534
("defies all three hypotheses", Kaiser 2024), GALEX J2339−0424 / GD 378 (Be
+2 dex), NLTT 19868 (extreme Fe-poor), WD 0106−328 (Fe delivered as pure
metal, Farihi 2026).
*Verdict.* **Build second.** Depth, not scale (~50–80 objects with ≥ 5
elements), but the population is complete and every object is already
measured; the deliverable is the misfit list, which has never existed.

**S53 Slag, not glass — the mineralogy of warm debris.** *Claim.* Refined
debris (smelter slag, structural ceramics) is CaO–Al₂O₃–SiO₂ glass with the
Fe, Mg and S phases stripped; hypervelocity collision debris is silica glass
**with** SiO gas, crystalline forsterite/enstatite and FeS (HD 172555, HD
23514, HD 15407A templates). Fe-depletion alone is *not* clean — mantle-only
impact debris is Fe-poor — so the target is a Ca–Al glass with no olivine,
pyroxene, FeS or forsterite features and no SiO. *Data.* CASSIS (every
Spitzer/IRS low-resolution spectrum, > 11,000 sources; the ISOTHERM corpus),
JWST MIRI/MRS of the 16 EDDs in Su et al. 2026 (arXiv:2607.06684). *Statistic.*
A feature-presence score over the 8.9 µm silica EW, 10 µm olivine, 11.3/33 µm
forsterite, 23 µm FeS and SiO; a candidate is silica-present, everything-else
absent. *Kills.* Optically thin vs thick geometry, grain-size suppression of
features, and the absence of laboratory optical constants for Ca–Al slag
glasses (a gap the paper must state). *Record.* Stevens, Forgan &
O'Malley-James 2016 declare refined-material detection "unlikely"; nobody has
run the discriminant. *Verdict.* Build as the spectral stage of S52 on the
~20 objects with spectra.

**S54 The molecule life and chemistry both refuse — artificial species in
line surveys.** *Claim.* No interstellar molecule with more than one fluorine
atom has ever been detected; fluorine chemistry terminates at HF and CF⁺
(Neufeld & Wolfire 2009; Acharyya & Herbst 2017 find no route to CF₂/CF₃), and
terrestrial life makes no N–F, S–F or fully fluorinated bonds (Seager et al.
2023). A rotational line pattern of CHF₃, CH₂F₂, CF₃Cl, CF₂Cl₂, CFCl₃, CF₃CN,
COF₂, SO₂F₂, NF₃ or the CF₂ radical in a molecular cloud, a circumstellar
envelope or a debris-disc gas is therefore a residue of chemistry, not of
astrochemistry. The clean gases of the exoplanet literature (CF₄, SF₆, C₂F₆)
have no dipole and are radio-invisible; they need the IR track (SF₆ 10.55 µm,
CF₄ 7.8 µm, NF₃ 907 cm⁻¹ against MIRI/MRS and ISO-SWS).
*Data.* The public unidentified-line inventories: Orion KL HIFI (Crockett et
al. 2014, ApJ 787, 112 — ~1,730 U-lines, 12 % of the survey), QUIJOTE TMC-1
(1,591 features, 188 unidentified), GOTHAM (Xue et al. 2025), IRC+10216 (He
et al. 2008, 17 U-lines), ALCHEMI, PILS, ReMoCA, PRIMOS; the ALMA archive
cubes for stacking; CDMS/JPL for the species that have entries (CH₃F, COF₂;
the rest need SPFIT/SPCAT predictions from the published constants).
*Statistic.* ≥ 3 predicted transitions coincident within the source linewidth
after v_LSR correction, LTE-consistent relative intensities at one (T_ex, N),
no coincidence with any known species/isotopologue/vibrational state, and a
false-alarm rate from shuffled predicted frequencies; then a matched-filter
stack in the cubes. Falsifier: a fluorocarbon column exceeding the HF/CF⁺
fluorine budget is contamination, not detection.
*Kills.* Vibrationally excited states of abundant species (the dominant
U-line explanation), isotopologues, blends, instrumental features; CH₃Cl and
CH₃F (natural organohalogens: Fayolle et al. 2017) as the baseline.
*Record.* No artificial-molecule search of any ISM/CSE line survey exists;
the exoplanet CFC literature (Lin et al. 2014; Haqq-Misra et al. 2022;
Schwieterman et al. 2024; the 2026 review calling PFCs an "extinct
technosignature") is decoy, not prior art.
*Verdict.* **Build third.** Novel, executable on published tables, and the
one channel here where a single confirmed pattern is decisive.

**S58 The device in the dust — in-situ interstellar grain composition.**
*Claim.* `docs/goo.md` shows starlight-blown-out devices of 0.21–0.57 µm are
exactly the interstellar-grain size range detected in situ. Cassini CDA
measured 36 interstellar grains (Altobelli et al. 2016, Science 352, 312):
Mg-rich silicate, Fe-bearing, homogeneous, volatile-depleted. A refined
device is pure metal, pure Si, pure C or an alloy — a compositional outlier
in a population that is homogeneous. *Data.* Cassini CDA time-of-flight
spectra (PDS Small Bodies Node), Ulysses DUST (flux/mass only), Stardust ISPE
(7 candidates; the aluminium contamination story is the cautionary tale),
IMAP/IDEX (launched 2025; composition of interstellar grains — verify data
release), Europa Clipper SUDA in cruise, DESTINY+ (2028). *Statistic.* Per
grain, distance from the CDA population's Mg/Si/Fe/Ca locus; candidates are
outliers with no spacecraft-material signature (Al, Ti from the instrument;
Rh from the target). *Kills.* Instrument contamination (the ISPE lesson),
impact-ionisation fractionation, the interplanetary-dust background in the
same mass range. *Record.* Lacki 2026 (arXiv:2606.08373) predicts
"technograins" dispersed to the ISM; no composition-record test exists.
*Verdict.* List and probe; n = 36 today, but IDEX changes that and the probe
records when.

### XIII. Relocation residue (P2)

**S47 The forge — hot exozodis as ~1500 K swarms.** *Claim.* The cheapest
place to collect stellar energy per unit collector mass is as close to the
star as materials allow, ~1500 K. Interferometry finds a ~1 % H/K-band excess
around 10–30 % of nearby main-sequence stars (Absil et al. 2013; Ertel et al.
2014, 2025 review), attributed to sub-micron grains that should be blown out
in hours; the leading model is magnetic trapping of nano-grains (Rieke,
Gáspár & Ballering 2016) and it has unresolved problems. Hephaistos I
excludes T > 1000 K by construction. Nobody has framed the population as swarm
candidates. *But the physics cuts against the simplest version:* a grey
1500 K blackbody producing a 1 % K excess gives ~8 % at 10 µm, and the
K-bright/N-faint pattern that forces nano-grains (Kirchschlager et al. 2017:
grains < 0.5 µm) is exactly what a Planck swarm cannot do. What survives is the
outlier: a star whose N-band excess **is** consistent with the Planck
extrapolation of its K excess. *Data.* Absil 2013 (FLUOR, 42 stars), Ertel
2014 (PIONIER, 92), Nuñez 2017 (JouFLU, 44), Absil 2021 (52), HOSTS N-band
(Ertel 2020, 38 stars), Keck-nuller limits, Marshall 2016 polarimetry nulls;
JMMC OiDB for the OIFITS. *Statistic.* Per star, a likelihood ratio of a
single-temperature grey body (T free, f free) against the nano-grain
emissivity family over H/K/L/N with the polarimetric null, and a variability
term (a swarm being built or decaying changes; only κ Tuc varied, and it is now
a companion, Stuber et al. 2026). *Kills.* Faint companions at the 1 % level
(Tsishchankava et al. 2025 — closure phases do not exclude them), the
K-vs-N calibration across instruments, ~150 stars total. *Verdict.* Novel
framing, small sample; build as a re-read of published tables, ranked by the
Planck-consistency outlier statistic, with the honest expectation that
nano-grain physics wins for the well-constrained systems.

**S63 Rings around the dead — post-biological hosts by construction.**
*Claim.* A technosignature at a host where biology is impossible is a
post-biological signature with no biological alternative: pulsar planets (PSR
B1257+12, B1620−26), white dwarfs and neutron stars generally, brown/Y
dwarfs, free-floating planets. Osmanov (2016, 2018) computes Dyson rings
around pulsars at 300–700 K — a W1/W2 excess, which escapes the frozen W3/W4
ceiling of `docs/necrosignatures.md` §3A(i). OSSUARY and the WD channel own
the *debris* reading of WD excesses; this is the *ring-temperature* reading
over WD + NS in Gaia × WISE, plus a brown-dwarf VIGIL (Y-dwarf Wien peaks sit
in W2; NEOWISE series exist for every WISE Y dwarf) and the "FFP hotter than
cooling allows" test (age/mass degeneracy is the killer; tens of objects).
*Record.* Theory only (Osmanov; Ćirković & Bradbury 2006; Opatrný 2017; Hsiao
2021; Kayali 2025 light curves); no executed search on these hosts; no SETI
pointing on record at either pulsar-planet system. *Verdict.* Build the WD+NS
ring pass as an OSSUARY re-read (same acquisition, different temperature
prior); pulsar planets are a two-target vet.

### XIV. Transition residue (P3)

**S61 Ignition — an infrared excess being born on an old star.** *Claim.* A
takeover's resource-acquisition phase is exponential over decades. The
observable is a **monotonic W1/W2 rise ≥ 5 yr with a flat optical light curve
on a Gaia-astrometric, kinematically old field star, with no excess in
AllWISE 2010 / AKARI 2006 / IRAS 1983** — the reverse of EMBER. Natural "born"
excesses around mature stars (giant impacts, comet swarms) rise in < 1 yr and
then decay or vary stochastically (HD 15407A −14 % over 15 yr; ID8/P1121);
appearance rate ~10⁻⁷–10⁻⁶ per star per yr → 5–50 such events per decade over
5×10⁶ old stars, none sustained. The phenomenology's best published example
is a galaxy (NGC 6447: +1.2 mag W1/W2 over 14 yr, optical flat, SPHEREx warm
dust → obscured AGN turn-on), which fixes the dominant contaminant and its
kill (parallax and proper motion). *Data.* NEOWISE single-exposure table
(IRSA `neowiser_p1bs_psd`, 2014–2024), unWISE/unTimely epochal coadds (Kang et
al. 2025, arXiv:2511.22071 — 8.3 M W1 variables with a "smoothness" statistic,
tables pending), Gaia DR3, ZTF/ATLAS/ASAS-SN for the optical, SPHEREx QR2 for
the 2025–26 endpoint and its 3–5 µm colour temperature. *Statistic.* Per
star, the Spearman/Kendall monotonicity and the linear slope of the W1 and W2
series with the scan-direction period (345–385 d) modelled out, jointly with
an optical slope consistent with zero; rank by slope significance × baseline;
require the 2010 epoch to sit on the photosphere. *Kills.* Obscured AGN and
dust echoes (parallax), YSOs (plane, Hα, IR class), AGB/Mira dust formation
(optically variable), R CrB (optical declines — opposite sign), novae (optical
eruption precedes), unTimely deblending/latent artefacts, W1 ≲ 8 saturation.
*Record.* Unoccupied for field stars: the YSO and AGN communities each did
half inside their own populations. *Verdict.* **Build fourth.** Catalogue
scale (5×10⁶), a decade baseline that exists now, and a natural background
whose time signature differs from the signal.

**S57 The growing transit — construction around a planet, 2009→2026.**
*Claim.* A shell or swarm being built around a transiting planet grows its
depth achromatically over years; a planet being dissolved or mined adds a
chromatic asymmetric tail. No systematic Kepler-vs-TESS depth (R_p/R_*)
consistency catalogue exists (Wang & Espinoza 2024 searched within-TESS TDVs
on 330 planets — none robust; Zuckerman et al. 2023 searched single-transit
anomalies in 218 Kepler systems — none unexplained; Kaye & Aigrain 2025
compared ephemerides only). Haqq-Misra et al. 2022 name "transit depths that
change over time due to construction" as a technosignature; nobody ran it.
*Data.* Exoplanet Archive KOI (`koi_depth`, `koi_ror`, `koi_impact`,
`koi_duration`), TOI (`pl_trandep`), the Planetary Systems table's `tic_id`
join; TESS sectors over the Kepler field (~2 per year since 2019); Gaia DR3
for deblending. *Statistic.* Per planet, per-epoch k = R_p/R_* at fixed
(a/R_*, b, per-band limb darkening); fit k(t) = k₀ + k₁t; require |k₁|/σ > 5,
T₁₄ and ingress duration changing as R_p (not as b — kills precession), the
Kepler/TESS band ratio matching the limb-darkening prediction (kills spots and
blends), and no Gaia neighbour inside the TESS aperture able to supply the
change. Long-period branch: for every KOI/TOI with P > 30 d, an ingress/egress
asymmetry statistic and epoch-to-epoch depth variance — no photoevaporative
engine exists beyond ~0.3 AU, so a tailed cold planet has no natural model.
*Kills.* Dilution (Han et al. 2025: TESS radii low by 6 %; CROWDSAP errors),
nodal precession (KOI-120, Kepler-13Ab, Kepler-47d), oblateness (≲ 1 %),
activity cycles, cadence and TTV smearing, grazing degeneracy. *Verdict.*
Build; ~500–1,000 planets, every input public, a positive would be
extraordinary. The occurrence-versus-age "missing planets" idea was checked
and is **dropped**: Sayeed et al. 2025 and PAST IV find it flat; it would be a
null paper.

### XV. Weapon residue (P4)

**S52 The shattered cradle — warm debris at the habitable-zone radius of a
mature star.** *Claim.* A destroyed or disassembled planet leaves warm dust at
its orbit, above the collisional steady-state maximum for the star's age
(Wyatt et al. 2007: f_max = 0.16×10⁻³ r^{7/3} t^{−1}, r in AU, t in Myr; Moór's
catalogue form with M_* and L_*). Known extreme debris disks (17 in Moór et
al. 2021; 21 with spectra in Su et al. 2026) sit 10⁴–10⁶ above f_max and are
young — only BD+20 307 (~1 Gyr, 400 K) and TYC 4479-3-1 (5 ± 2 Gyr, 400 K) are
mature, and **no EDD at 250–350 K around a > 1 Gyr star is known**. That empty
cell is the target. *Data.* Gaia DR3 FGK dwarfs × AllWISE (≥ 5σ in W3 **and**
W4, K_s − W1 < 0.2, sub-arcsecond registration, |b| > 10°), ages from ≥ 2
independent indicators (HD 15407A's 2.1 Gyr isochrone vs 80 Myr AB Dor
membership is the warning), Cotten & Song 2016, Kennedy & Wyatt 2013, Moór
2021 (VizieR), CASSIS/JWST for S53. *Statistic.* log(f/f_max) > 3 **and**
T_bb in 250–350 K **and** age > 1 Gyr; NEOWISE variability as a descriptor
(14/17 natural EDDs vary), the S53 mineralogy score as the discriminant.
*Kills.* Background dusty galaxies in the 6–12″ WISE PSF (every Hephaistos
candidate), cirrus at W4, mis-aged young stars near Sco–Cen/ρ Oph,
wide-companion-driven comet delivery (5/6 of Moór's hosts; BD+20 307's WD
companion), mantle-only impact debris. Expected natural yield ~100 old EDDs to
500 pc, so the *population* is not the novelty — the HZ-temperature cell and
the mineralogy are. *Record.* Stevens et al. 2016 §2.5 name planetary
destruction → debris and say it is "unlikely to elucidate its origins";
Lacki 2025 predicts the dust veil is transient. No age-gated,
temperature-gated search exists. *Verdict.* Build as a VIGIL/OSSUARY sibling
on the same acquisition; the honest weakness (natural late instability
explains both known old EDDs) is stated up front.

**S59 The arc — superflares above the starspot energy ceiling.** *Claim.* A
flare's energy is bounded by the magnetic energy its spots store, E_mag ≈ f
B²/8π A_spot^{3/2} (B = 3 kG, f ≤ 1), and Okamoto et al. 2021 find the
Kepler upper envelope consistent with that bound. A flare that **exceeds** the
bound on a star whose rotational amplitude gives A_spot is either a
measurement problem or not a flare. Deliberate stellar manipulation (a weapon,
or engineering) would show as repeated target-localised superflares on a
spot-free slow rotator. *Data.* Notsu et al. 2019 (ApJ 876, 58), Okamoto et
al. 2021 (ApJ 906, 72), Tu et al. 2020/2021, Vasilyev et al. 2024 (Science;
56,450 Sun-like stars, 2,889 superflares — never cross-matched to spot
amplitudes), Yang & Liu 2019 (J/ApJS/241/29), Pietras et al. 2022
(J/ApJ/935/143), McQuillan 2014 / Santos 2021 rotation, Gaia RUWE/NSS,
eROSITA DR2 (1.9 M X-ray sources, July 2026) for activity. *Statistic.* ξ =
log E_flare,max − log[f_max B²/8π A_spot,max^{3/2}] with f_max = 1 and A_spot
from the largest amplitude in any quarter/sector, marginalised over spot
latitude (polar-spot floor); candidates have ξ > 0 on ≥ 2 independent flares
and pass the centroid test. *Kills.* Unresolved companion flarers (the reason
the slow-rotator superflare count fell in 2019 — Kepler TPF / TESS FFI
centroid shift during the flare is decisive), flux-normalisation bias for
blends, spot-area underestimate (amplitude is a lower bound), bolometric
conversion, pulsators, evolved stars, momentum dumps. *Record.* No
flare-as-technosignature paper exists (Lingam & Loeb 2017 is flare
*mitigation*); the ξ > 0 tail has never been mined and is, by construction,
Notsu's contamination set. *Verdict.* Build as a METRONOME sibling on the
same catalogues; the deliverable is the vetted ξ > 0 list.

**S62 The dead cradle — anti-biosignature with technosignature.** *Claim.* A
planet whose atmosphere has been removed where the cosmic shoreline (Zahnle &
Catling 2017: cumulative XUV ∝ v_esc⁴) predicts retention, in a system that
also carries a technosignature, is the direct observable of P4. Today the
conjunction is testable for **one** system: TRAPPIST-1 (b, c bare; d flat; e
ambiguous — JWST) with the deepest single-target radio limits (Tusay et al.
2024, ATA planet–planet occultation windows, EIRP 2–13 TW; FAST 2025). GJ 1132
b is bare with no SETI pointing; LHS 1140 b is a tentative *pro*-habitability
case. The JWST Rocky Worlds DDT (nine targets; GJ 3929 b bare at 118 ± 22 ppm)
and the "evolving shoreline and sandbar" population (arXiv:2608.23912: bare
M-dwarf terrestrials where atmospheres were expected) grow the sample to ~15.
*Statistic.* Shoreline margin m = log(I_XUV,cum/I_⊕) − 4 log(v_esc/v_esc,⊕) −
c with c marginalised over the Meni-Gallardo & Pallé 2026 zero-point posterior;
a candidate needs m < −2σ_m (should have retained) and a bare-rock eclipse.
*Kills.* The unconstrained zero point, M-dwarf XUV histories (×3–10), the
natural "airless valley", high-albedo surfaces, MIRI precision. *Record.* No
paper frames technology on a dead world; no techno + anti-bio conjunction test
exists. *Verdict.* Fails SCALE; kept as the single-target side test on
TRAPPIST-1 and the ~15 Rocky Worlds planets, and as the framing of every
S57/S61 candidate's atmosphere follow-up.

### XVI. Persistence residue (P5)

**S60 The relay — intercepting node-to-node beams by geometry.** *Claim.* A
persisting machine network communicates with beams. Earth intercepts an A→B
beam only if it lies inside the far-field spillover cone past B: with A at
d_A, B at d_B < d_A and full divergence θ, the pair must appear within
δ_max = (θ/2)(d_A − d_B)/d_B on the sky. Counts scale as θ²N²: for the 331,312
GCNS stars within 100 pc, a diffraction-limited 10-m optical link gives 3×10⁻⁵
qualifying pairs (un-interceptable — a large-aperture network leaves Earth
outside every beam), a 100-m dish at 1.4 GHz (9′) ~2×10⁴ pairs, a 10-m dish
(1.5°) ~2×10⁶. So the interceptable regime is **radio with modest apertures or
deliberately over-filled beams**, and the intercepted flux is only slightly
below what the receiver gets — a strong signal, not leakage. *Data.* Gaia DR3
for the pair geometry; Breakthrough Listen open data (GBT/Parkes, ~1,700
nearby stars; MeerKAT commensal ~10⁶ stars in primary beams) for the re-cut:
for each observed star B, the background stars A inside δ_max define the
pointings that also sit on a pair line, and Gaia gives the pair's relative
radial acceleration — a **Doppler-drift prior** no search uses. *Kills.*
Trials over 10⁴–10⁶ pairs; RFI; the near-antipodal "Earth between nodes"
geometry must be included with its own count. *Record.* Single-target
precedents only: Tusay et al. 2022 (SGL antipode of α Cen), Hort et al. 2024
(anti-solar point), Tusay et al. 2024 (TRAPPIST-1 planet–planet occultations —
the intra-system version of exactly this); the 2026 review calls a relay
search "a valid strategy" and none exists. *Verdict.* Build the geometry
(cheap, Gaia-only) and the archival re-cut; state the θ² yield honestly.

**S50 The century — cessation and decay on a 100-year baseline.** *Claim.*
KNELL (period ceased), RUST (scatter rising) and the secular-fade test
(`dimming/secular`) are all confined to ZTF's ~6 years. DASCH DR7 (29 Dec 2024;
252,458,490 sources, 23.6 billion measurements, 1885–1992) is a
century-baseline light curve for every star to B ≈ 15, and no blind fade /
cessation / rising-scatter search on DR7 has been published. A Kessler cascade
(decades) or an unmaintained structure's grind-down is best seen on this
baseline; a period that stopped between 1900 and 1950 and never returned is
visible nowhere else. *Data.* DASCH DR7 via the documented web API
(`querycat` cone → `POST /dasch/dr7/lightcurve`) and the `daschlab` client;
per-exposure limiting magnitudes; APPLAUSE (70,000 plates) as the independent
plate set. *Statistic.* Per star, the KNELL efficiency-normalised
non-detection of a catalogued period in early vs late plate blocks, the RUST
season-scatter slope, and the secular fade — each against the star's **own
plate-limit and blending history**, with the Menzel gap (no plates 1954–1970)
and the pre/post-gap photometric offset modelled as a step, never as a trend.
*Kills.* The Menzel-gap offset (Hippke et al. 2016; Lund et al. 2016 — it
manufactured KIC 8462852's "century dimming"), per-plate limiting magnitude,
time-clustered blending on low-resolution series, emulsion changes, high-PM
stars, long-period giants (Tang et al.), the contested VASCO-adjacent
transient claims of 2026. *Record.* Unoccupied; `docs/knell.md` §Plates
already records why no DASCH stage was wired (Menzel gap) — the fix is to
model it, which the DR7 per-exposure limits now allow. *Verdict.* Build as
KNELL/RUST/secular on a new backend, on the periodic-variable and
bright-star populations first.

### XVII. The archive at home (P6)

**S55 The crypt — artifacts in lunar permanent shadow.** *Claim.* Regolith
turnover in a permanently shadowed region is metres per Gyr and the floor
temperature 25–40 K; it is the one place near us where a Gyr-old object
survives, and a thermally active one is visible against nothing. In a 240 m
Diviner pixel at 35 K, a 1 m² source at 300 K raises channel 6 (13–23 µm)
brightness temperature to **57 K** while channel 9 (100–400 µm) moves by
0.01 K: the discriminant is *anisothermality* (ch6/ch7 ≫ ch9), the same physics
as Diviner rock-abundance retrievals, never applied inside PSRs. A metre-scale
metallic dihedral is a ~10× single-pixel radar bright spot with CPR > 1 and no
optical boulder. *Data.* Diviner Polar Cumulative Product (240 m/px, 80–90°,
channels 3–9, all nadir data 2009–2019, split by season and sub-solar
longitude; PDS Geosciences / UCLA mirror); Mini-RF S-band Stokes CDRs; ShadowCam
(1.7 m/px; PDS4 quarterly, 5.6 TiB in the Aug 2026 release, COGs allow range
reads); LROC NAC PSR long-exposure atlas (324 PSRs); LOLA 5 m DEMs. *Statistic.*
Per PSR-interior pixel (≥ 2 footprints from the illumination boundary), a
two-component (T_cold, f·T_hot) fit over ch6–9 ranked by ch6-excess
significance under count-weighted noise, required to be **season-independent**
(scattered-light heating is not); radar: point-like high-CPR/high-σ⁰ pixels
with no NAC/ShadowCam boulder; ground truth: the Apollo SNAP-27 RTGs (~1 kW,
500 K fins) in nighttime ch6 stacks. *Kills.* PSR-edge sunlit walls inside the
160×320 m footprint, secondary scattered light (Shackleton floor ~90 K),
geolocation on steep walls, ch6/7 calibration striping, cosmic rays, boulder
fields and fresh ejecta (warmer, never above the seasonal mean), pits
(outside PSRs), human hardware (LCROSS Centaur, Luna-25, IM-2, Chang'e-7).
*Record.* Davies & Wagner 2013 proposed optical NAC inspection and named
"nuclear waste heat"; every executed search since is optical-NAC machine
learning (Lesnikowski 2020; AAS 2025; arXiv:2608.09350, Aug 2026). No thermal
or radar PSR artifact search exists. *Verdict.* Build the Diviner
anisothermality pass (product sizes ~10² MB per channel/season — runner
scale); radar and ShadowCam as the vet.

**S56 The grave — a technological extinction in Earth's own record.** *Claim.*
Schmidt & Frank 2018 (the Silurian hypothesis) asked whether a prior
industrial civilisation on Earth is detectable and listed generic markers. The
sharper question is whether a mass-extinction boundary carries a
*technological* residue distinguishable from impact and volcanism: the
FALLOUT fission-yield vector (Zr/Mo/Ru light peak, Ba/La/Ce/Nd heavy peak,
Ag–Te valley, no Pb) or long-lived reactor nuclides (¹²⁹I, ²³⁶U, ¹³⁵Cs, ⁹³Zr,
¹⁰⁷Pd, ²³⁷Np, ²⁴⁴Pu) **without** the r-process partners (⁶⁰Fe, ²⁴⁴Pu pulses
of Wallner et al. 2021) — the same no-correlated-partner logic as S46. Oklo
(2 Ga) is the positive control for how fission products are found (Nd, Ru, Sm
isotope anomalies; depleted ²³⁵U). Contested boundaries: end-Ordovician, the
Devonian Kellwasser/Hangenberg events (a supernova has been proposed, Fields
et al. 2020), the Carnian, the PETM. *Data.* The Sedimentary Geochemistry and
Paleoenvironments Project (tens of thousands of shale analyses with ages, API
at sgp-search.io), EarthChem/PetDB, GEOROC compilation files; the FeMn-crust
¹²⁹I profiles (Pacific crusts, 7×10⁻¹⁴–1.3×10⁻¹², exponential decay, 55–70 Ma)
and Wallner's Crust-3 archive for ²³⁶U/¹²⁹I/¹³⁵Cs re-measurement. *Statistic.*
FALLOUT's likelihood ratio (fission vector vs best natural mixture) run on
boundary-layer samples binned by age against the same lithology away from
boundaries; for nuclides, a pulse in ¹²⁹I or ²³⁶U/²³⁸U ≫ 10⁻¹⁰ with no ⁶⁰Fe
co-pulse. Pre-industrial "industrial-looking" spherules are listed as a
frame, with the IM1/coal-ash episode (Gallardo 2023; Desch & Jackson 2023,
2024) as the reason the burden of proof is extreme. *Kills.* Element mobility
in sediments, diagenesis, PGE/Hg volcanic and impact signatures, natural
²³⁸U spontaneous fission and Oklo-type reactors, contamination at every
stage. *Record.* Schmidt & Frank 2018 and Wright 2018 frame the question;
neither proposes the fission vector or a database test; no ²³⁶U crust profile
older than 1 Myr is published. *Verdict.* Build the SGP boundary test (it is a
FALLOUT re-target on a public database); the nuclide test needs new AMS runs
and is listed as such.

### XVIII. All-sky spectral residue on the new surveys

**S48 The spark — SPHEREx single-channel stellar excess.** *Claim.* S41 (the
industrial NIR line: Nd:YAG 1.064, Yb 1.03–1.09, Er-fibre 1.53–1.57 µm) waits
for Roman. SPHEREx is already taking 102-channel 0.75–5 µm spectra of the
whole sky (QR2 weekly since Oct 2025; R = 35–130; 6.15″ pixels; 19.5–19.9 AB
depth after two years), and its Reference Catalog SPLICES lists ~9 million
isolated IR-bright point sources. At R ≈ 40 a laser line is unresolved and
adds flux to **one** spectral element; the observable is a stellar spectrum
with a single-channel excess at ≥ 5σ that repeats across the two independent
sky passes and across detector position. *Data.* IRSA TAP `spherex.plane` ×
`spherex.artifact` (cutout URIs), the cutout service, SPLICES as the seed
list; the per-source High Reliability Source Catalog is not yet released (the
probe records when). *Statistic.* Per source, forced aperture photometry
through every exposure covering it → per-channel residual against a
smooth-continuum fit; candidate if one channel exceeds 5σ in ≥ 2 independent
passes at ≥ 2 detector positions, with the industrial wavelengths as *flags*,
not filters. *Kills.* Detector persistence, cosmic rays, the ZODI model,
pixel-wavelength mapping errors at the LVF edges, stellar lines at low
resolution (Paβ 1.28, He I 1.083, CO bandheads at 2.3 µm — all catalogued),
emission-line galaxies mis-seeded as stars (Gaia parallax kills them),
blends at 6″. *Record.* No SPHEREx or Euclid technosignature search exists;
Vides et al. 2019 (WFIRST coronagraph laser) is the targeted precedent.
*Verdict.* **Build fifth** — the largest-scale item here (~10⁷ stars) on a
brand-new public archive.

**S49 The spark, resolved — Euclid NISP.** *Claim.* Euclid Q1 (March 2025)
serves 1D red-grism spectra (1.21–1.89 µm, R ≈ 450) for **4,313,551 sources**
with a pre-computed line-feature table (`euclid_q1_spe_lines_line_features`:
per-source line SNR and wavelength), and DR1-Foundation (Nov 2026) scales that
by ~30. The Er-fibre band lies inside the grism. The observable is an
unresolved emission feature on a *stellar* point source (Gaia parallax) in the
line table that no stellar or galactic identification matches. *Statistic.*
Join the line table to Gaia stars; veto known stellar lines and every
redshifted galaxy line; require the feature in ≥ 2 dither/roll angles
(slitless zeroth orders and trace overlap are the killers, as in
`docs/roman.md` §2.2). *Verdict.* Build with S48; it is the ROMAN/lines
detector on the substitute the channel index already names.

---

## 3. The data-source ledger

Archives this repository has never queried, what each serves, and how a
runner reaches it (every line is probed by `necrofrontier-probe.yml`; the state column is
filled from `results/necrofrontier/probe.json` — three runs on 2026-09-13,
the last reaching 57 of 67 endpoints with the expected product):

| Source | Serves | Access | Signatures | State (probe run 3, 2026-09-13 8:01 AM ET) |
|---|---|---|---|---|
| SPHEREx QR2 + SPLICES | 102-channel 0.75–5 µm all-sky spectral images; the seed list | IRSA TAP `spherex.plane`/`artifact`/`obscore`, `splices`; cutout URL; AWS `spherex-qr` | S48, S61 | **REACHED**: 1,367,432 public spectral-image products in `spherex.obscore`; `splices` holds **9,925,660** sources with 157 columns (2MASS/WISE photometry, designations); the plane→artifact join returns QR2 level-2 URIs |
| Euclid Q1 (DR1-F Nov 2026) | 4.3 M NISP 1D spectra + line-feature table | IRSA TAP: 26 `euclid_q1_*` tables incl. `euclid_q1_spe_lines_line_features` (`spe_line_snr_gf`, `spe_line_name`, …); SSA; S3 | S49 | **REACHED**, columns as asserted |
| DASCH DR7 | 252 M century light curves, per-exposure limits | `dasch.cfa.harvard.edu/dr7/web-apis` (querycat → POST lightcurve), column docs, `daschlab` on PyPI | S50 | **REACHED** (all four endpoints) |
| Presolar Grain Database | 20,230 SiC grains with isotope panels | `presolar.physics.wustl.edu` (DOI'd IEDA release) | S46 | **NOT REACHED from the runner**: connect timeout on both 443 and 80 in three runs; the guessed DOI was wrong; ADS is behind a WAF. The EarthChem Library (`ecl.earthchem.org`) answers and is where the release is DOI'd — the table must be located there by title search or requested from the authors. **This is S46's open item.** Meteoritical Bulletin reached |
| PEWDD + PyllutedWD | 1,739 polluted WDs, 24 metals; natural forward model | VizieR `J/A+A/691/A352/pewdd` (also Bonsor+2020 `J/MNRAS/492/2683/tablea1`); GitHub `jamietwilliams/PEWDD`, `andrewmbuchan4/PyllutedWD_Public` | S51 | **REACHED** (all) |
| CASSIS / IRS Enhanced Products | Spitzer/IRS low-res spectra | `cassis.sirtf.com` **times out from the runner (3 runs)**; IRSA TAP `irs_enhv211` "IRS Enhanced Products" answers: **16,986 spectra, 85 columns** (object, RA/Dec, IRAC/MIPS photometry, extraction metadata) | S52, S53 | **PARTIAL → build S53 on `irs_enhv211`**; also Meng+2015 `J/ApJ/805/77` and 15 debris/flare/rotation tables found by prefix; Moór+2021 is not at VizieR (use the erratum table from the paper) |
| CDMS / JPL / Splatalogue | rotational line lists | `spec.jpl.nasa.gov/ftp/pub/catalog/catdir.cat`; `cdms.astro.uni-koeln.de/classic/entries/` (a script-rendered page; species not in the HTML); Splatalogue | S54 | **REACHED**. JPL directory carries **NF₃, COF₂, CH₂F₂ and CH₃Cl** (by the probe's regexes) and **not** CHF₃, CH₃F, SO₂F₂, CF₃CN, CF₃Cl, CF₂Cl₂, CFCl₃, CHClF₂, CF₂ — those need SPFIT/SPCAT predictions from the published constants (S54 step 1) |
| Diviner PCP / Mini-RF / ShadowCam | polar thermal maps 240 m; S-band Stokes; 1.7 m PSR imagery | PDS Geosciences `lro-l-dlre-4-rdr-v1/` has **two volumes, `lrodlr_1001` (RDR by year) and `lrodlr_1002`** (the level-3/4 products); ODE PCP description page; Mini-RF page; ShadowCam archive page and SIS (1.6 MB PDF) | S55 | **REACHED** except the UCLA mirror (TLS certificate fails verification; not bypassed) — use the PDS `lrodlr_1002` volume |
| SGP / EarthChem / GEOROC | sediment geochemistry with ages | `sgp-search.io` (SPA; `env.js` names `archive.sgp-search.io` as the bulk archive — also an SPA, so the API must be read from the app bundle or the SGP paper); `www.earthchem.org`, `portal.earthchem.org`; `georoc.eu` | S56 | **PARTIAL**: sites answer, no documented API path found yet |
| Exoplanet Archive KOI / TOI / PS | depths per mission, `tic_id` join | TAP `cumulative` (`koi_depth`), `toi` (`pl_trandep`), `ps` (`tic_id`) | S57 | **REACHED** (all three) |
| Cassini CDA / Ulysses DUST | interstellar-grain composition and flux | SBN dust holdings page lists Ulysses, Cassini (`resource/cocda.html`), Galileo, New Horizons | S58 | **PARTIAL**: CDA resource reached; the Ulysses resource path still unresolved (two guesses 404) |
| Superflare + rotation tables | E_flare and spot amplitude per star | VizieR: Okamoto+2021 `J/ApJ/906/72/table2`, Tu+2022 `J/ApJ/935/90/table2`, Shibayama+2013 `J/ApJS/209/5`, Yang & Liu `J/ApJS/241/29`, McQuillan `J/ApJS/211/24/table1`, Davenport `J/ApJ/829/23`, Günther `J/AJ/159/60` | S59 | **REACHED**; eROSITA DR1 page reached (DR2 to add) |
| Breakthrough Listen open data | archival GBT/Parkes/MeerKAT products | `seti.berkeley.edu/opendata` reached; `bldata.berkeley.edu` denies directory listing (products must be addressed by path) | S60 | **PARTIAL**; Gaia TAP returned 266,536 stars within 100 pc with good astrometry in run 1 and HTTP 500 in runs 2–3 (transient) |
| NEOWISE single exposures | decade W1/W2 series | IRSA TAP `neowiser_p1bs_psd` | S61 | **REACHED** |
| Interferometric exozodi tables | H/K/L/N excesses, polarimetry | JMMC OiDB reached; VizieR prefix search for Absil 2013 / Ertel 2014 / Nuñez 2017 / Absil 2021 / HOSTS | S47 | **REACHED** (OiDB and the VizieR prefix query answer; the specific tables are read in the S47 build) |

Windows ahead, for the scheduler: Gaia DR4 **2 Dec 2026** (all epoch data —
S57's astrometric deblending and S63's WD/NS kinematics); Euclid DR1-Foundation
Nov 2026; UNIONS DR1 Oct 2026; SPHEREx HRSC (per-source spectra) after the
third sky pass; IMAP/IDEX composition data (S58); Rubin DR1 ~June 2028 (alerts
only until then).

---

## 4. Build order and the honest limits

Ranked by novelty × reach × natural-background cleanliness, as the charter
orders:

1. **S46 ISOTOPE** — one download, 2×10⁴ objects, a correlated natural
   signature the artificial case lacks by construction.
2. **S51 SLAG-WD** — a calibrated misfit list on a complete, measured
   population; nobody has produced it.
3. **S54 ULINE** — published U-line tables against predicted patterns; a
   single confirmed pattern is decisive.
4. **S61 IGNITION** — 5×10⁶ stars, a decade baseline that exists now, a
   background whose time signature differs.
5. **S48/S49 SPARK** — ~10⁷ stars on the newest all-sky archive.
6. **S57 GROWTH**, **S59 ARC**, **S50 CENTURY**, **S52+S53 CRADLE**, **S55
   CRYPT**, **S60 RELAY**, **S63 RING**, **S56 GRAVE**, **S47 FORGE** — in
   that order; S62 and S58 are listed with the dates on which their data reach
   catalogue scale.

Dropped after the check, so nobody rebuilds them: planet occurrence vs
stellar age (done, flat: Sayeed et al. 2025; PAST IV); a 511 keV point-source
stack at nearby stars (SPI sensitivity makes it a limit, not a detection
channel; COSI 2027); AMS-02 anti-helium (events unpublished); the hypervelocity
stellar-engine limit (Lingam & Loeb 2020, done); D/H depletion as a
technosignature (Catling et al. 2025, proposed — a JWST/HWO question, not a
catalogue one).

What none of this can do: a successor that computes cold and dark, needs no
planet and sends no beam is CENOTAPH's territory and stays there; a successor
that left is indistinguishable from plain extinction. The eighteen signatures
above are the residues of one that **stayed, refined, relocated, and kept
working** — and every one of them is a catalogue test with a natural
background whose signature is different in kind, not merely in amplitude.

---

## 5. What the record says (sweep runs 3 and 4, 2026-09-13)

`necrofrontier-lit` took three runs to get past the arXiv API's throttling
(run 1: 429 on 64 of 65 requests; run 2: 429 on 31 of 32, and the ADS
fallback found the repository's `ADS_TOKEN` secret unset; run 3: keyless
fallbacks — arXiv abstract pages for ids, OpenAlex for searches). Run 3
fetched twelve of the fifteen groups before its deadline; run 4 (groups
g13–g15 only, 40 of 67 fetches, half through OpenAlex) completed the set, and
the decoy-aware scan ran over **2,135 abstracts (1,928 unique)**. The record
(`results/necrofrontier_lit/concept_scan.json`, verbatim abstracts) reads:

| Group | Signature | Decoy-free hits | What they are | Verdict |
|---|---|---|---|---|
| g1 isotope purity | S46 | 10 | Catling+2025 (D/H by fusion), Ellery 2025 (Th/Nd from lunar reactors), Carrigan 2010 "Starry Messages" (interstellar archaeology), OpenAlex noise | **Unoccupied**: no paper reads stable-isotope purity in solids |
| g2 hot swarm | S47 | 4 | Hephaistos I, Lacki "Sunscreen", NO₂ pollution, BL transit anomalies | **Unoccupied**: no hot-exozodi-as-swarm reading |
| g3 SPHEREx/Euclid line | S48/S49 | 0 | — | **Unoccupied** |
| g4 century fade | S50 | 12 | the DASCH DR7 paper, plate-archive status papers, J1407, FO Aqr, an RCB star, LBV sleep | **Unoccupied**: no blind DR7 fade / cessation search |
| g5 WD residue | S51 | 3 | Huang+2026 (the siderophile template), cecilia pipeline | **Occupied only by Huang+2026**; the pair residual and misfit list are not in the record |
| g6 shattered cradle | S52/S53 | 6 | Lacki 2025 "Ground to Dust", Lacki 2026 "Dust to Dust", the lunar-regolith micron paper | **Unoccupied**: no age- and temperature-gated EDD search, no slag discriminant |
| g7 artificial molecule | S54 | 9 | generic technosignature reviews; every CFC paper is exoplanet-atmosphere (decoyed) | **Unoccupied**: no ISM line-survey search for industrial species |
| g8 lunar crypt | S55 | 18 | Davies & Wagner 2013, Benford 2019/2021, Lesnikowski 2020, the Aug 2026 ML lunar-anomaly search, "Is ET Lurking in Our Cosmic Backyard?", Chandrayaan-2 DFSAR (natural), "Unravelling the Mystery of Lunar Anomalous Craters" (radar + IR, natural, ice) | **Unoccupied for thermal/radar**: every artifact search is optical |
| g9 terrestrial record | S56 | 3 | Schmidt & Frank 2018, Wright 2018 (framing only) | **Unoccupied**: no fission-vector or database test |
| g10 relay geometry | S60 | 12 | Hippke's network I/II, "Engineering an Interstellar Communications Network by Deploying Relay Probes" (arXiv:2204.08296), Gertz's nodes and landbases, Forgan 2019 transit network, "Exoplanet Occultations as Technosignature Targets", the TRAPPIST-1 ATA search | **Unoccupied**: network *design* papers and single-target occultation searches; no star-pair spillover selection at catalogue scale |
| g11 growing transit | S57 | 21 | Zuckerman+2023 (single-transit anomalies), Wright+2016 Ĝ IV, spot-induced depth variations (HIP 67522), the J1407 ring "construction zones" pun, exocomet tails | **Unoccupied**: no cross-mission secular-growth search |
| g12 ignition | S61 | 6 | Metzger+2017 secular *dimming* of KIC 8462852, nearby-excess census, TWA disks | **Unoccupied** |
| g13 spot ceiling | S59 | 8 | Davenport 2016 Kepler flare catalogue, Okamoto+2021 statistics, an X-ray superflare on a fast rotator, "Empirical flare energy limits for the largest historical sunspots", "Starspot Activity and Superflares on Solar-type Stars" (the natural bound literature, decoyed by occurrence statistics) | **Unoccupied**: the spot-energy bound is established and never mined as an anomaly set; no flare-as-technosignature paper |
| g14 post-biological hosts | S62/S63 | 25 | Osmanov 2016/2018 and Kayali+2025 (pulsar Dyson rings, theory), Ćirković & Bradbury 2006, the LTT 3780 ATA narrowband search (an M-dwarf planet host, not a post-biological one), Vidal's "pulsar positioning system", "Lens Flare" X-ray-binary beacons, "Searching for Intelligent Life in Gravitational Wave Signals" (arXiv:2212.02065), multiresonant-system beacons (arXiv:2204.14259), **"The Dyson Minds 2025 Workshop: SETI Around Black Holes" (arXiv:2604.21886)** — the nearest neighbour for the compact-object branch and to be read before S63 is written up | **Unoccupied**: no executed search on pulsar planets, WD/NS ring temperatures, brown dwarfs or free-floating planets; no techno + anti-bio conjunction test |
| g15 in-situ grains | S58 | 7 | the Stardust ISPE series (VIII, IX, XI and the seven-particle paper), "Compositional Analysis of Interstellar Dust as seen by the Cassini CDA" and Altobelli+2016 (decoyed as flux papers), the IMAP/IDEX instrument paper, the FOSSIL and DuneXpress mission concepts | **Unoccupied**: the composition record exists and no paper reads it for artificial outliers |

**Id verification (`id_title_check.json`).** Seven asserted arXiv ids
resolved to unrelated papers and were removed from the sweep in favour of
title searches: the PGD SiC paper, Lin+2010 (Qingzhen SiC), Rieke+2016
(magnetic trapping), Vides+2019 (WFIRST laser), Hippke+2016 (plate accuracy),
Lisse+2009 (HD 172555), Wallner+2021 (⁶⁰Fe/²⁴⁴Pu). None of those ids appears
in this document. Six others carried the expected paper under a title my
fragment did not match (Absil 2013 = paper III of the FLUOR series at
1307.2488; Ertel 2014 = paper IV at 1409.6143; Lund 2016; Xu 2013; Kennedy &
Wyatt 2013; Forgan 2019; the NGC 6447 turn-on; and from run 4 Forgan 2013 at
1306.1672, "On the Possibility of Detecting Class A Stellar Engines Using
Exoplanet Transit Curves", and Opatrný+2017 at 1601.02897, "Life under a black
sun") and are confirmed. Run 4 found one more wrong id (Strub+2015 Ulysses,
asserted 1510.06181), removed. Pre-arXiv journal papers (Whitmire & Wright
1980; Kawka & Vennes 2016) returned no arXiv or OpenAlex entry and are cited
by journal reference only; the Kaye & Aigrain 2025 title was a guess and is
dropped.
