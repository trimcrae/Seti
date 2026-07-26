# OSSUARY — warm dust around stars that cannot make it

*An ossuary holds what is left when the flesh is gone.*

**Signature S7 (Kessler cascade) in `docs/necrosignatures.md`.**
Channel code: `src/seti/ossuary/` · workflow `.github/workflows/ossuary.yml` ·
results `results/ossuary/`.

---

## 1. The claim

A megastructure that is no longer maintained does not sit there being a
megastructure. It grinds itself to dust.

The mechanism was published in 2025 and its author proposed **no observable and
ran no search**:

> Brian C. Lacki, *"Ground to Dust: Collisional Cascades and the Fate of
> Kardashev II Megaswarms"*, **arXiv:2504.21151**, ApJ 985, 191 (2025).
>
> "Although long-lived megaswarms are extremely powerful technosignatures, **they
> are liable to be subject to collisional cascades once guidance systems start
> failing. The collisional time is roughly an orbital period divided by the
> covering fraction of the swarm.** … I further show that once the collisional
> cascade begins, **it can develop extremely rapidly for hypervelocity
> collisions.** … **Most megaswarms are thus likely to be short-lived on cosmic
> timescales without active upkeep.**"

`excess.cascade_timescale_yr` implements that scaling. At 1 AU with covering
fraction *f* = 10⁻³ the cascade time is ~10³ yr — instantaneous beside any
stellar age. So the intact swarm is not the relic. **The dust is the relic.**

Lacki's follow-up review, *"Dust to Dust: Prospects for Passive Technosignatures
as Relics of ETI"* (**arXiv:2606.08373**, 2026), frames passive relics as an
agenda rather than a result, and closes: *"In the end, what we may be left with
are the end products of collisional cascades: dust."*

### The search

Dust around a star is normally boring. So the search inverts the question: rather
than asking which stars have dust, it asks **which stars cannot have dust**, and
then looks for dust there anyway.

The natural-null argument is unusually strong for three overlapping populations:

* **Metal-poor stars** ([Fe/H] < −1). Planetesimal formation and giant-planet
  occurrence fall steeply with metallicity.
* **Halo-kinematic stars** (|**v** − **v**_LSR| ≳ 200 km/s). Old, and dynamically
  unrelated to the thin disk where every known debris disk lives.
* **Old stars generally** (≳10 Gyr). Debris-disk incidence decays steeply with
  age; stirring dies; Poynting–Robertson drag and radiation-pressure blowout
  clear micron grains in 10⁴–10⁶ yr, four to six orders of magnitude faster than
  the stellar age.

**A confirmed warm (≳200 K) excess around a metal-poor halo dwarf has no natural
explanation.** Whether one exists is the search.

### Lacki predicted this exact sample and nobody looked

The strongest single piece of evidence for this channel is that the theory paper
names the hunting ground and no observational search followed it:

> "**The collisional destruction of megaswarms also may have implications for
> where in a galaxy we might find them. First, stellar encounters should be
> minimized, implying low density environments. These are found on the outskirts
> of large galaxies and in their halos, as well as in dwarf galaxies. Second, low
> metallicity stars are more likely to be found in these same environments. This
> suggests that megaswarms are more likely to be found in regions that are
> sometimes considered disfavorable for habitability.**" (Lacki 2025, §7)

He also flags the honest tension: below [Fe/H] ≈ −1.5 to −2 a civilization may
not be able to *build* a swarm in the first place (thresholds of −2 in Zackrisson
et al. 2016, −1.5 in Johnson & Li 2012, up to −0.6 in Andama et al. 2024). Our
primary cut at **[Fe/H] < −1** sits deliberately at that boundary: metal-poor
enough that the natural reservoir is gone, not so metal-poor that construction is
implausible. The strict tier at −1.5 is reported separately.

---

## 2. Novelty verdict

**No infrared-excess or debris-disk search selected on metallicity or on halo
kinematics has ever been published.** Established from runner-fetched literature
evidence in `results/necrolit/`, `results/seamlit/`, `results/hephlit/`,
`results/litcheck_dyson/` (the sandbox has no arXiv/ADS/OpenAlex egress; all
fetches are verbatim API output committed to the repository).

The evidence, in descending order of strength:

1. **Two independent full-corpus arXiv queries for halo-star infrared excess
   return exactly zero results.** `all:"halo star" AND all:"infrared excess"` → 0
   (`results/necrolit/arxiv_q_halo_star_infrared_excess.atom`);
   `abs:"infrared excess" AND abs:"halo stars"` → 0
   (`results/seamlit/arxiv_q_excess_halo_stars.atom`). Different scripts,
   different field scopes, singular and plural. Verbatim from the sweep: *"No
   halo (Pop II) star has a confirmed warm IR excess."*
2. **`all:"metal-poor" AND all:"debris disk"` returns three papers in total.**
   Exactly one is a dust search around metal-poor stars — Venn et al. 2014
   (arXiv:1407.1449, ApJ 791, 98), *n* = 7 stars at [Fe/H] ≲ −5 — and its
   motivation is *inverted*: they were testing whether dust depletion **causes**
   the low measured metallicity. Six of seven show no mid-IR excess; they
   explicitly leave cooler and fainter disks unconstrained.
3. **Every executed Dyson / IR-excess search uses a near-solar, thin-disc, or
   unselected sample.** Project Hephaistos I (arXiv:2201.11123) states outright:
   *"We also take tracks with metallicities between Z = 0.012 to Z = 0.018. This
   range should be representative of the thin-disc stars within 100 pc."* All four
   Hephaistos papers and Huang et al. 2026 contain **zero** occurrences of
   `metal-poor`, `[Fe/H]`, `halo`, or `Population II`. The most thorough executed
   Dyson search in the literature excludes this channel's sample by construction.
4. **The prediction exists and is unexploited** — Lacki 2025 §7, above, published
   April 2025 and followed by no search.
5. **An unexploited observational hint.** Contardo & Hogg (2024, arXiv:2403.18941)
   note of their 53 extreme mid-IR-excess candidates versus a 4.9 M-star parent
   sample: *"We also observe an off-set trend in [M/H] towards lower values than
   the full sample, with a few outliers standing out"* — and defer *"confidently
   old stars with an unusually high IR excess"* to future work.

### Prior art this channel must position against

| work | what it did | why it is not this |
|---|---|---|
| **Theissen & West 2014/2017** (arXiv:1409.0016, 1702.08465) | WISE excess in proper-motion-verified field M dwarfs; Galactic height \|Z\| as an age proxy; 584 extreme excesses | **The closest prior art.** Proper-motion-verified, not halo-selected; \|Z\| ≲ 700 pc is thin/thick disk; no [Fe/H] cut anywhere; M dwarfs only. The two papers reach *opposite* conclusions on age dependence. |
| **Gáspár, Rieke & Ballering 2016** (arXiv:1604.07403, ApJ 826, 171) | 662 disks (222 detected, 440 upper limits), disk mass vs metallicity | A correlation study on an existing thin-disk sample, not a metallicity-selected search. Supplies our strongest quantitative null (§3). |
| **de la Reza et al. 2023** (arXiv:2302.01850, A&A 671, A136) | The only kinematically-framed debris work | Debris+planet hosts are overwhelmingly metal-enriched **thin disk**. Three old **thick-disk** exceptions (HD 10700, HD 20794, HD 40307; 8–11 Gyr) have 70 µm dust masses *"lower than that of the Kuiper belt … by several orders of magnitude"*, and **cold, not warm**. |
| **Luo & Liu 2026** (arXiv:2602.23004) | Warm debris disks around nearby FGK stars, LAMOST DR12 × Gaia × multi-band IR, <150 pc; 12 candidates, 10 new | Metallicity- and kinematics-blind, thin-disk, nearby. The machinery we want, pointed at the population we are *not* studying. Useful as a methodological template and as the normal-population comparison. |
| **Kenyon, Bromley & Najita 2026** (arXiv:2603.11994, AJ 171, 223) | Assembled a **3,675-star** Cold Debris Disk Survey catalogue with metallicities; *plan* an excess-versus-host-property analysis | **A competitor forming.** Not done yet. Also a useful input catalogue. |
| **Huang, Liu, Wyatt & Kennedy 2025** (arXiv:2505.07602) | 10 pc sample (339 stars), W3 excess at 3σ | 5 candidates, **all 5 spurious**; detection rate 0/339. Their recommendation of **5σ, not 3σ**, is adopted here. |

One framing point worth stating plainly: because the entire warm-Dyson candidate
list (Hephaistos A–J) has now collapsed under JWST/MIRI follow-up
(arXiv:2607.09460), the field's warm-excess channel is effectively exhausted for
thin-disk samples. A metallicity- and kinematics-selected sample is one of the few
places a *new* population could still hide.

### One honest correction

The query `all:"debris disk" AND all:"old stars" AND all:incidence` returns 0, but
this is a **phrase artefact**, not evidence: Montesinos et al. 2016 is literally
titled *"Incidence of debris discs around FGK stars in the solar neighbourhood"*
and uses British "discs". **We do not claim nobody has studied old-star debris-disk
incidence.** They have — see §3.

---

## 3. The natural null, quantified

This is what "stars that cannot make it" means numerically.

**Versus age.** Kennedy & Wyatt 2013 (MNRAS 433, arXiv:1305.6607) measured the
12 µm warm-dust luminosity function over Hipparcos main-sequence stars within
150 pc (*N* = 24,174): **old (>Gyr) dusty systems occur at 1 in 10⁴; young
(<120 Myr) systems at ~1%.** A factor ~100 decline, measured, on a thin-disk
near-solar-metallicity sample. Our halo sample is ≳10 Gyr — past the end of that
curve.

The decay is *steeper* than the canonical steady-state cascade *f* ∝ *t*⁻¹:

* Pawellek et al. 2021 (arXiv:2101.12049), observed: *"the fractional luminosities
  … drop by two orders of magnitude within the first 100 Myr … a decay equivalent
  to 1/age²."*
* Wyatt, Clarke & Booth 2011 (arXiv:1103.5499), theoretical: cross-sectional area
  ∝ 1/age² in the P–R-dominated regime; dust mass ∝ 1/age^2.8.

Other anchors: Hyades (625 Myr) **0/67 FGK**; Herschel DEBRIS field F–K
**17.1 (+2.6/−2.3)%** at ~10⁻⁵ fractional luminosity (Sibthorpe et al. 2018);
DUNES FGK **22% (23/105)** (Montesinos et al. 2016); *"fewer than 1:1000 stars
have 12 µm excesses larger than a factor of five"* (Kennedy & Wyatt 2012).
Note the honest tension: Montesinos et al. find *detection rate* is flatter with
age than disk *brightness*.

**Versus metallicity.** Gáspár, Rieke & Ballering 2016: *"**disk-bearing stars
seldom have metallicities less than [Fe/H] = −0.2**"*, and warm components *"lack
… large mass around stars of low metallicity ([Fe/H] < −0.085)"*. Our cut at
[Fe/H] < −1 is **0.8 dex below the lowest metallicity at which a debris disk has
ever been catalogued in a systematic sample.**

There is a genuine literature disagreement here and it must be reported: Beichman
et al. 2006, Moro-Martín et al. 2015 and Marshall et al. 2014 all found debris
incidence **uncorrelated** with metallicity, and Gáspár et al. attribute that to
methodology (dust masses rather than detections, upper limits included, age
evolution accounted for). **Both camps agree there is no evidence of debris disks
in the metal-poor regime** — which is all this channel needs.

Planet occurrence scales as *P*(*Z*) ∝ 10^(2[Fe/H]) (Wyatt, Clarke & Greaves
2007): a factor **100** suppression at [Fe/H] = −1.

**Direct measurement in a metal-poor population.** McDonald et al. 2011
(arXiv:1104.5155), ω Centauri: *"**Aside from the post-AGB star V1, we find no
star from the cluster's bulk, metal-poor ([Fe/H] < −1.5) population — including
the carbon stars — to be producing detectable amounts of dust.**"* That is the
empirical background this channel searches against.

A caution from the same work: metal-poor circumstellar dust is **featureless**
(metallic iron rather than silicate). So "featureless continuum" is **not** a
technosignature discriminant in this regime, and no mineralogy argument is made
here.

---

## 4. Sample

Three tracks, run as an independent workflow matrix (`src/seti/ossuary/acquire.py`):

| track | selection | metallicity | kinematics | why |
|---|---|---|---|---|
| `spec` | Gaia DR3 GSP-Spec [M/H] < −1 | RVS spectroscopy | full **UVW** (every RVS star has a radial velocity) | The gold sample. Bright (G_RVS < 12), so WISE is high-S/N and *not* confusion-limited. |
| `phot` | Gaia DR3 GSP-Phot [M/H] < −1 **and** the upper confidence bound still metal-poor | photometric | mixed | Breadth, at the cost of per-star metallicity reliability. |
| `halo` | high tangential **or** radial velocity, no metallicity required | none needed | full UVW | Tests the halo leg without depending on any spectroscopic pipeline. |

Quality: `parallax_over_error > 5`, `RUWE < 1.4`, `G < 18`, 0.35 < BP−RP < 3.0,
3800 K < T_eff < 6500 K. Expected working sample ~10⁵–10⁶, narrowing to ~10⁵ with
halo kinematics.

Photometry comes from the **Gaia archive's own AllWISE and 2MASS mirrors**
(`gaiadr1.allwise_original_valid`, `gaiadr1.tmass_original_valid`) joined through
the official pre-computed cross-match tables. This is deliberate: **the official
cross-match already propagates proper motion to the epoch of the external
catalogue** (Marrese et al. 2019). A naive positional match on a halo sample —
mean proper motion of order 100 mas/yr over a 5.5 yr Gaia→AllWISE baseline —
silently returns nothing for exactly the fastest, most interesting stars. *That
bug cost a previous channel in this repository a whole run.* The propagation is
then independently re-derived from raw positions in `vet.astrometry_gate` rather
than trusted.

Every catalogue column name is resolved from a live `TOP 1` probe
(`acquire.probe_columns`) rather than assumed, so a renamed mirror column degrades
one field instead of breaking the pull.

### Metallicity provenance is carried per star

The natural-null argument is only as strong as the metallicity behind it. A
GSP-Phot [M/H] is a far weaker claim than an APOGEE one, so `feh_provenance` and
`feh_is_spectroscopic` travel with every row and are reported. GSP-Spec [M/H]
carries a known log g-dependent bias; rather than hard-code a calibration
polynomial we (a) keep the cut conservative at −1.0, (b) report a strict −1.5
tier separately, and (c) require any headline candidate to be **either**
spectroscopically metal-poor **or** halo-kinematic — two independent routes to
"no reservoir".

---

## 5. The estimator: an empirical photosphere, not a model one

Every published IR-excess search predicts the photosphere from a synthetic stellar
atmosphere. On a sample *selected* to be metal-poor that model is extrapolated
exactly where it is least calibrated, and a colour offset of a few hundredths of a
magnitude — entirely plausible at [Fe/H] < −1 — would manufacture a false excess in
**every star at once**.

So this channel uses no synthetic photosphere. `excess.fit_colour_locus` builds the
locus **from the sample itself**: the robust running median of Ks−W1, Ks−W2, Ks−W3
against BP−RP, in colour bins, with two σ-clipping iterations. The median has a 50 %
breakdown point, so the rare genuine excesses cannot drag the locus they are measured
against. Any metallicity-dependent colour offset is *absorbed*, not assumed away, and
the significance is measured against the empirical star-to-star scatter rather than an
assumed model error.

Three consequences worth stating:

* **The anchor is Ks, and that is conservative.** Ks (2.16 µm) is the longest
  wavelength still overwhelmingly photospheric for a >4000 K dwarf, so the lever
  arm to W1 is short. And an unresolved cool companion — the classic false excess —
  contributes at Ks too, which *inflates* the anchor and therefore *suppresses* the
  inferred W1 excess.
* **The W1−W2 and W1−W3 colour excesses are anchor-independent.** Ks cancels out of a
  band difference, so those statistics do not inherit the anchor's error. The ledger
  requires a detection to appear there, and `excess.select_excess` enforces it.
* **The locus refuses to extrapolate.** A star bluer or redder than any well-populated
  bin gets NaN, not a guess, and leaves the funnel unflagged.

Dust is then characterised by a two-stage-grid single-temperature blackbody fit to
the excess fluxes, with **400 Monte-Carlo redraws** of every excess flux for honest
16/84 percentiles on (T_dust, τ) — the fit is strongly non-linear at the cold end, so
a formal curvature error would be misleading. With three or four bands the χ² is a
real goodness-of-fit, and a bad χ² is itself diagnostic: a blend is not a
single-temperature blackbody.

---

## 6. Contamination model

**Assume nine in ten raw flags are junk.** Silverberg et al. 2018 measured that at
most **7.9 % ± 0.2 %** of AllWISE-selected infrared excesses are good disk
candidates — a ~92 % false-positive rate. They further find the McDonald et al. and
Marton et al. searches have false-positive rates **>70 %**, and that **all 13 of
Theissen & West's candidates with W4 S/N > 3 are false positives**. So
`vet.funnel_counts` reports **per-stage removals**, not just a running total: a
funnel is only credible if it can say which stage removed what.

### Gates, in order (`src/seti/ossuary/vet.py`)

| # | gate | kills |
|---|---|---|
| 1 | **WISE quality** — per-band `ph_qual` ∈ {A,B}, `cc_flags` free of D/P/H/O, not saturated, `number_of_mates` = 0, `ext_flag` = 0 | artefacts, ambiguous cross-matches, resolved galaxies |
| 2 | **Inherited ledger** — excess must appear in W1/W2/W3 (**never W4 alone**); W1−W2 ≥ −0.05; G ≤ 18; 0.35 ≤ BP−RP ≤ 3.0; W1−W2 ≤ 0.8 | cirrus, blends, confusion-limited photometry, unanchored photospheres, AGN |
| 3 | **Unresolved companion** — fitted T_dust ≤ 1800 K **and** no significant J/H excess | companion photospheres (two independent handles) |
| 4 | **Astrometric registration** — WISE centroid within 1″ of the Gaia position *propagated to the AllWISE epoch*; Gaia↔CatWISE proper motions consistent | background galaxies, static interlopers |
| 5 | **Chance-superposition prior** — a-priori probability of an interloper bright enough to supply the excess, ≤ 10⁻³ | Hot DOGs, dusty starbursts |
| 6 | **Galactic cirrus** — SFD E(B−V) ≤ 0.10, \|b\| ≥ 15°, plus a **population-level** Spearman test of flag rate against E(B−V) | diffuse 12/22 µm foreground |
| 7 | **Globular-cluster veto** | crowded metal-poor fields (see below) |
| 8 | **Impostor gate** — λ Boo / blue straggler / SIMBAD-flagged | the impostors this sample selects for |
| 9 | **Luminosity class** — dwarfs only | metal-poor giants have winds and dusty envelopes |
| 10 | **Null-reservoir host** — metal-poor **or** halo | out-of-sample stars whose excess is real but unremarkable |

Beam blending is then run per candidate (`vet.beam_blend_verdict`): unlike the
white-dwarf version of this test, the denominator is the star's *measured excess
flux*, not its total W1 flux. A main-sequence star dominates its own beam easily,
so the question is never "does a neighbour outshine it" but "can a neighbour
account for the small extra flux we are calling a technosignature". A neighbour
4 magnitudes fainter is irrelevant to the total and decisive for a 3 % excess.

### The three impostors this sample selects for

These are not generic contaminants — they are objects that arrive *wearing this
channel's badge*, and each has a gate.

1. **λ Bootis stars.** A/early-F stars whose **surface** is metal-depleted by
   accreting gas-depleted ISM — not primordially metal-poor. Murphy et al. 2020
   (arXiv:2008.02392): **21 of 34 have infrared excesses**, and *"stars previously
   classified in the literature as blue horizontal branch stars … have a high
   probability of being λ Boo stars."* Photometrically metal-poor, blue, sometimes
   catalogued as BHB, and carrying a WISE excess. They live at 6500–9000 K, so the
   T_eff ceiling at 6500 K removes essentially all of them while keeping the
   metal-poor FGK dwarfs the claim is about. **This is the single largest
   false-positive risk for this channel.**
2. **sdA halo binaries.** Brown et al. 2017 (arXiv:1703.07799): *"the majority of
   sdA stars are metal-poor A–F type stars in the halo"*, several with infrared
   excess fully explained by a ~0.8 M☉ companion. Handled by the T_eff ceiling,
   the RUWE cut, and gate 3.
3. **Evolved blue stragglers.** Yong et al. 2016 (arXiv:1603.07034) found two to
   three α-rich metal-poor "young" giants with debris-disk-like excesses that are
   merger products. Flagged blueward and brighter than the metal-poor turnoff.

### Why globular clusters are vetoed, not vetted

Boyer et al. 2010 (arXiv:1002.1348) showed that a **published** red-giant-branch-wide
infrared excess across 47 Tuc — a metal-poor, old population, exactly this channel's
target class — was *entirely* stellar blending and imaging artefacts, using the same
archival Spitzer imagery as the original claim. Metal-poor stars concentrate toward
globular clusters, which are the most crowded fields on the sky. This channel does not
try to vet cluster sightlines. It removes them.

### Background galaxies: a funnel stage with a number attached

Every Project Hephaistos candidate died of extragalactic confusion. JWST/MIRI
(Hephaistos IV, arXiv:2607.09460) resolved candidates D and E into a **Hot DOG at
z ≈ 0.9** and a **dusty starburst at z ≈ 0.4**, both within ~1″, and the Hot DOG sky
density of ~9 × 10⁻⁶ arcsec⁻² *"can probably account for the contamination of all 7"*
(arXiv:2405.14921).

`vet.expected_chance_alignments` computes the budget for the actual sample, and the
number goes in `summary.json` and `REPORT.md`. For a 3 × 10⁵-star sample within a 1″
registration radius:

| quantity | expected |
|---|---|
| Hot DOGs (geometric, 9 × 10⁻⁶ arcsec⁻²) | **~8.5** |
| AllWISE sources bright enough to supply the excess (W1 ≈ 12) | **~4.7** |
| the same, in the full 6.5″ beam without the registration cut | ~198 |
| leverage bought by the registration cut | **~42×** |

Read this honestly. **The expectation is of order a few, not ≪ 1.** That has three
consequences, all of which are design decisions rather than caveats:

* The astrometric-registration stage is **load-bearing**, not decorative. Without it
  the contamination is ~200 objects and nothing is believable.
* Halo stars' large proper motions give that stage real leverage: over the 5.5 yr
  Gaia→AllWISE baseline a 200 mas/yr star moves 1.1″, so a static background source
  fails registration outright. This channel's sample is unusually well suited to the
  test that killed Hephaistos.
* **A single surviving candidate is not a detection.** With a handful of chance
  alignments expected, any survivor needs sub-arcsecond imaging or a spectrum before
  it means anything.

### A latitude claim that is easy to get wrong

The halo sample sits at high |b|, and that is a **large real advantage against
Galactic cirrus and against stellar blending** — both fall steeply away from the
plane, and both are quantified in gates 6 and 7. It is **not** an advantage against
extragalactic confusion: if anything, high |b| means slightly *more* visible galaxies
per square degree. Gates 4 and 5 handle the extragalactic case. Latitude does not.

---

## 7. Honest limits

**The sensitivity floor is instrumental, not a choice.** Wien peaks: W1 → 852 K,
W2 → 630 K, W3 → 241 K, W4 → 132 K. **W3/W4 depth has been frozen since the 2010
cryogenic mission ended** — NEOWISE-R, CatWISE2020 and the deep unWISE coadds are
W1/W2 only. Below ~200 K the only route is W4, the shallowest and most
confusion-limited band, whose flux for a 22 µm-faint star is cirrus. There is no
deeper 12/22 µm measurement available. So the claim leans on **W3 with W1−W2
support**, and the warm floor is set at **200 K**.

**The threshold is 5σ, not 3σ.** Huang et al. 2025 searched 339 stars for W3 excess
at 3σ, got 5 candidates, and found all five spurious. `chi_min = 5.0` in
`config/thresholds.yaml`, with an *additional* 3σ requirement on the
anchor-independent colour excess.

**What this channel cannot do.** It cannot distinguish ground-up megaswarm debris
from an exotic natural dust source by mineralogy — metal-poor dust is featureless
(§3). It cannot detect cold (<200 K) relics at all. It cannot confirm a candidate:
the terminal state of this pipeline is a shortlist for sub-arcsecond imaging and
spectroscopy, and it says so.

**What would count as a detection.** A dwarf star that is (a) spectroscopically
[Fe/H] < −1 **or** halo-kinematic with a measured UVW, (b) carrying a ≥5σ excess in
W3 *and* a ≥3σ red W1−W3 colour excess against its own population's locus, (c)
registered to <1″ at the propagated WISE epoch with a chance-superposition prior
<10⁻³, (d) fitted T_dust between 200 and 1800 K with no J/H excess, and (e) clear of
cirrus, clusters, beam neighbours and every impostor class in §6. Anything less is
reported as what it is.

---

## 8. Running it

```bash
# offline: the whole detector + gauntlet against synthetic data
make test                              # tests/test_ossuary.py

# runner (GitHub Actions, workflow_dispatch): .github/workflows/ossuary.yml
#   inputs: tracks, g_max, limit_per_band, max_followup, skip_followup,
#           reduce_only_run_id
seti ossuary --stage acquire           # Gaia x AllWISE x 2MASS, checkpointed
seti ossuary --stage analyze           # offline given results/ossuary/sample.parquet
```

Outputs in `results/ossuary/`: `summary.json` (verdict, funnel counts with
per-stage removals, population counts, chance-alignment budget, cirrus
correlation, the fitted locus), `REPORT.md`, `excess_flagged.csv`,
`candidates.csv`, `giants_excess.csv` (analysed separately, never counted as
clean), `followup.csv`.

Verdicts are first-class: `NO_DATA_REACHED` when no archive track returned rows,
`NO_LOCUS` when no colour bin reached minimum occupancy, `OK` otherwise. A
degraded run reports its degradation rather than emitting an empty candidate list
that would read like a result.
