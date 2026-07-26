# EMBER — mid-infrared waste heat that switched off

*Signature **S1** of `docs/necrosignatures.md`: an infrared excess present at an
early epoch (IRAS 1983, AKARI 2006–07) and absent at a later one (WISE 2010).*

---

## 1. The claim

Every waste-heat technosignature search ever executed measures the sky **once**.
That is not an accident of scheduling; it is a stated assumption. The field's
Astro2020 white paper on thermal-infrared technosignatures (Wright, Zackrisson &
Lisse 2019, arXiv:1907.07829) justifies the whole programme on the grounds that

> "the signature is as long-lived as the underlying technology"

and Carrigan (2010, arXiv:1001.5455) argues that a Dyson sphere's lifetime
"could be substantially longer than the lifetime of a civilization". If that is
right, a single epoch suffices. If it is wrong — if waste heat can stop — then
**the one observable that most cleanly distinguishes an artefact from a dust
disc has never been looked for.**

EMBER looks for it: a robust mid-infrared excess at an early epoch that is
significantly reduced or absent decades later.

The physical case that it *can* stop is now published. Lacki (2025,
arXiv:2504.21151) shows megaswarms are "liable to be subject to collisional
cascades once guidance systems start failing" and are therefore "short-lived on
cosmic timescales without active upkeep". Blanco, Haqq-Misra & Profitiliotis
(2026, arXiv:2604.13774) derive civilization duty cycles of 0.38–1.00 and an
"effective detectability duration". Neither proposes an observation. This
channel is the observation.

---

## 2. Novelty status — **NOVEL in framing, PARTIALLY ANTICIPATED in method**

Verified against 187 full texts and 782 bibliographic records already fetched
into `results/disaplit{,2,3}/`, `results/seamlit/`, `results/dysonlit/`,
`results/litcheck{,_dyson}/`, `results/hephlit/`, `results/vanishlit{,2,3}/`,
`results/offlit/` and `results/necrolit/`, and re-verified on the runner by
`scripts/ember_novelty.py` → `results/emberlit/`.

### 2.1 What is genuinely unoccupied

**The entire Dyson/waste-heat lineage is single-epoch, by word count.**
Occurrences of "epoch" in a photometric sense:

| Paper | arXiv | "epoch" | "disappear" | "vanish" | "turn/switch off" | "cease" |
|---|---|---|---|---|---|---|
| Carrigan 2009 | 0811.2376 | **0** | 0 | 0 | 0 | 0 |
| Ĝ I | 1408.1133 | **0** | 0 | 1\* | 0 | 1\* |
| Ĝ II | 1408.1134 | **0** | 0 | 0 | 0 | 0 |
| Ĝ III | 1504.03418 | **0** | 0 | 0 | 0 | 0 |
| Hephaistos I | 2201.11123 | **0** | 0 | 0 | 0 | 0 |
| Hephaistos II | 2405.02927 | 1† | 0 | 0 | 0 | 0 |

\* Non-observational prose ("humanity … will forever cease to increase it").
† "35 epochs", a Gaia astrometric-solution hyperparameter.

Carrigan 2009 predates WISE and never mentions it. The one paper in the lineage
with 42 "epoch" hits — Ren, Garrett, Zackrisson, Korn, Siemion & Wright 2026
(arXiv:2607.03619) — uses the word **exclusively for astrometric propagation**,
comparing Gaia positions propagated to the AllWISE epoch against WISE centroids
to find background contaminants. Its own summary describes its two methods as
"an *astrometric* examination of AllWISE four-band image centroids" and "a
search for potential extragalactic companions". There is no photometric epoch
comparison anywhere in the Hephaistos series.

**The 2026 flagship review** (Vidal et al., arXiv:2605.21093, ~502,000
characters, 22 authors, reviewing every proposed technosignature from the Planck
to the universal scale) contains, as technosignature terms:
`turn off` = 0, `switch off` = 0, `shut down` = 0, `cessation` = 0,
`ceased` = 0, `multi-epoch` = 0, `duty cycle` = 0, `AKARI` = 0, `NEOWISE` = 0,
`fading` = 0. Its waste-heat section describes only single-epoch cross-matches.

**Variability is actively discarded.** Suazo et al. (2024) state that their
`G_var` cut "rejects potential Dyson swarms with very large absorbing elements
since these in principle could generate detectable variations in the photometry
of the host star." Hephaistos I likewise excludes variables to remove Miras.
*A megastructure that changed is cut from every existing search by construction.*

**VASCO** (arXiv:1606.08992 and its successors) searches **optical**
disappearance on photographic plates; `infrared` = 0 hits across the founding
paper and all replications. No VASCO paper runs an infrared-excess variant.

### 2.2 What must be cited and distinguished — three real antecedents

Honesty requires stating these plainly; the claim is narrower than "nobody has
ever compared these catalogues".

1. **Kim et al. 2015 (arXiv:1501.05721) ran the identical three-catalogue
   comparison — upward.** "Using IRAS, AKARI, and WISE point source catalogs, we
   found that 4 sources … significantly **brightened** at MIR wavelengths over
   the 20–30 years of difference in observing times." Same method, opposite
   sign, framed as YSO/AGB astrophysics. **EMBER cannot claim methodological
   novelty.** Its all-sky yield of 4 is also a useful prior on how many real
   cross-epoch changers exist.
2. **Sedgwick & Serjeant 2022 (arXiv:2207.09985)** built the IRAS×AKARI all-sky
   cross-match over a 23.4-year baseline — to find outer-Solar-System planets by
   **proper motion**. The machinery exists and was validated; repointing it at
   *flux* change is the novel move, and the paper is the best available source of
   systematics lore for this exact cross-match.
3. **Melis et al. 2023 (arXiv:2306.11945)** ran the comparison *downward*, but
   targeted at R Coronae Borealis stars, where "IRAS, AKARI, and WISE data
   reveals similar fading trends, bursts that can show a factor of up to 10
   change in flux density between epochs". A known class, targeted, not blind.
4. **Liu 2020 (arXiv:2008.12611)** re-examined Rhee et al.'s 146 IRAS-detected
   debris hosts with WISE, but treats IRAS–WISE discrepancies as **calibration**,
   not time-domain disappearance. ~1/1000 the scale, different question.

### 2.3 The claim that survives, stated precisely

> No published work has searched, **blind and at catalogue scale**, for the
> monotonic disappearance of a mid-infrared excess between the IRAS/AKARI and
> WISE epochs, nor framed such a disappearance as a technosignature. The
> waste-heat literature is uniformly single-epoch because it explicitly assumes
> the signature outlives its builders.

---

## 3. Systematics verdict per epoch pair

Computed by `seti ember --stage audit` from the band model itself; committed to
`results/ember/pair_audit.json` on every run. `spread` is the factor by which
the early→late flux transfer moves as the excess temperature runs over
150–1500 K; `beam` is the ratio of early-to-late beam solid angle; the window is
bounded below by the early survey's sensitivity and above by the late band's
saturation, expressed in early-band flux.

The numbers below are computed from the **real SVO relative system response
curves**, fetched on the runner in run 30203763934 and committed to
`src/seti/data_assets/rsr/` (all eleven bands, `rsr_source = "svo"` throughout).
They supersede an earlier version of this table computed from the documented
trapezoid fallback; §3.1 records what changed and why it matters.

| pair | baseline | transfer@300K | spread | bandpass sys | beam ratio | usable window | verdict |
|---|---|---|---|---|---|---|---|
| **I25 → W4** | 26.9 yr | 0.925 | **1.03** | 0.9% | 84× | 0.40–13.1 Jy | **PRIMARY** — near-null transfer |
| **I12 → W3** | 26.9 yr | 0.910 | **1.71** | 6.2% | 286× | 0.40–1.05 Jy | **SECONDARY**, conditional on beam-summing |
| L18W → W4 | 3.7 yr | 0.984 | 1.98 | 0.6% | 0.23× | 0.09–12.3 Jy | **USABLE** — short baseline |
| I25 → L18W | 23.2 yr | 0.940 | 2.01 | 1.2% | 372× | 0.40–95.7 Jy | conditional |
| I12 → S9W | 23.2 yr | 0.632 | 4.81 | 2.5% | 400× | 0.40–285 Jy | conditional; AKARI arbitrates where W3 saturates |
| S9W → W3 | 3.7 yr | 1.44 | **8.25** | 6.2% | 0.72× | 0.05–0.67 Jy | **DEMOTED** — see below |

### 3.1 The finding that reorganised the channel

The original design brief proposed that "the cleanest pair is almost certainly
AKARI 9 µm (2006) → WISE W3 12 µm (2010)". **That is wrong on the spectral
axis, and the audit shows it quantitatively.** AKARI S9W and WISE W3 have a
transfer that moves by a factor of **8.25** across the plausible dust
temperature range: converting a 9 µm excess to a 12 µm excess without knowing
the excess temperature can manufacture — or conceal — nearly an order of
magnitude of change on its own. It is the *worst*-conditioned pair spectrally,
even though it is the best astrometrically, and its 3.7-year baseline is the
shortest available.

**The near-identical pair is I25 → W4**: spread **1.03**, i.e. the transformation
is null to within 3% no matter what the dust temperature is, with a bandpass
systematic under 1%, a beam ratio 3.4× smaller than I12→W3, and a usable flux
window **twelve times wider**, because W4 saturates at 12 Jy while W3 saturates
at 0.96 Jy. Together with a 27-year baseline that makes **I25 → W4 the discovery
pair**, I12 → W3 the corroborating pair, and AKARI the arbiter.

#### The real response curves cost I12 → W3 its primacy

This is a correction, and it is worth stating plainly because it moved the
channel's primary pair. With the **trapezoid** response fallback, I12 → W3 has a
temperature spread of 1.20 and I25 → W4 has 1.24, which made 12 µm look like the
cleanest available step and put it first. With the **real SVO curves** the two
numbers are 1.71 and 1.03: I12 → W3 is *worse* than the trapezoid implied and
I25 → W4 is *better*, and the ordering reverses. The reason is structural rather
than numerical — the true IRAS 12 µm band runs 7.5–15 µm and the true WISE W3
runs 7.5–17 µm, so they overlap heavily but weight the short-wavelength Wien
side very differently, whereas IRAS 25 µm and WISE W4 both sample the
Rayleigh-Jeans side of 150–1500 K dust where the ratio barely moves.

The lesson generalises: **a trapezoidal bandpass is not a conservative
approximation.** Here it understated the systematic on the pair the channel had
chosen as primary by 40%. `rsr_source` is recorded per band in every audit for
exactly this reason, and `tests/test_ember.py` now asserts the *ordering* of the
three spreads rather than any single number.

### 3.2 What was rejected, and why

* **NEOWISE as an epoch — rejected on physics.** NEOWISE carries W1/W2
  (3.4/4.6 µm) only; W3/W4 exist for the 2010 cryogenic phase alone. W1/W2 reach
  only T ≳ 500–700 K dust, so *a decade-baseline search at the wavelengths where
  100–300 K waste heat actually lives is impossible with NEOWISE*. It is used
  here exclusively as the post-drop flatness requirement (§4.9), which is a
  scientifically load-bearing role but not an epoch.
* **IRAS 60/100 µm → anything — rejected.** No late-epoch counterpart band
  exists, and these are the most cirrus-dominated IRAS channels. The 100 µm
  channel enters EMBER only as the cirrus *veto* (§4.2), which is the single most
  valuable use of it.
* **S9W → W3 demoted from primary to confirmation**, per §3.1.

### 3.3 The structural tension nobody has stated

To beat IRAS confusion you want bright sources. To avoid WISE saturation you
want faint ones. W3 saturates at ≈0.96 Jy while the IRAS PSC is complete only to
≈0.4 Jy, so **the clean I12→W3 window is barely half a decade wide in flux**.
Above it, W3 under-reports and manufactures exactly the signal being hunted —
which is why `late_saturated` is a hard rejection, not a warning. Three
mitigations, all implemented: use I25→W4 where the window is 8× wider; use AKARI
as the arbiter above the W3 ceiling (S9W saturates ~180 Jy); and refuse to claim
a fade from a saturated late band at all.

Sensitivity is set by the *early* epoch. At the IRAS 12 µm limit of 0.4 Jy a bare
photosphere implies Ks ≈ 4.3 — a naked-eye star — so IRAS-based pairs probe only
**very large** excesses, of order the ×30 drop seen in TYC 8241 2652 1. AKARI's
50 mJy limit reaches roughly Ks ≈ 6.5. This is a bright, shallow search by
construction, and the working sample is of order 10⁵ sources, not 10⁶.

---

## 4. Contamination model

Ordered by measured damage, worst first. Implemented in `src/seti/ember/vet.py`;
every rule has a test that trips it in `tests/test_ember.py`.

**4.1 IRAS beam blending — the dominant contaminant by orders of magnitude.**
The IRAS 12 µm beam is ~0.75′×4.5′, some 286× the solid angle of WISE W3. An
IRAS flux is the *sum* over everything in that footprint, so comparing it with
the single nearest WISE counterpart is guaranteed to fabricate fades in crowded
fields. The only defensible comparison sums **all** late-epoch sources inside the
early beam (`crossepoch.beam_sum_consistency`). Corroborated directly:
Kennedy & Wyatt (2012) and arXiv:1507.00708 ("in 24 cases … source confusion is
playing a role, in that either the source that is bright in the optical is not
responsible for the IR flux, or there is more than one source").

**4.2 Cirrus — and the single most valuable cut in the channel.**
Kennedy & Wyatt (2012, arXiv:1207.0521): "about 8,000 stars that have excess
emission, mostly at 12 µm. The positions of these stars correlate with the 100 µm
background level so **most of the flux measurements associated with these
excesses are spurious**." Cutting at IRAS 100 µm < 5 MJy/sr leaves 271 of
180,000 — the per-star false-excess rate drops from 4.4×10⁻² to 1.5×10⁻³, a
factor of ~30 for one number. This is mandatory for every IRAS-based pair.

**4.3 Eddington/Malmquist bias — the one systematic a two-sided null cannot
calibrate.** The early epoch is flux-limited, so near-threshold sources are
catalogued only when noise pushes them up; a deeper later survey measures the
truth and the pair reads as a fade with *nothing having changed*. Unlike every
other contaminant it is strictly one-directional. Handled twice: deboosted using
the source-count slope measured from the catalogue itself, and bounded by an
early-epoch S/N floor of 7.

**4.4 The two-sided statistic and its empirical null.** Every symmetric
systematic populates the fading and rising tails equally. The threshold is
therefore set from the sample's **own rising tail**, and the excess of faders
over risers above it is the only quantity that can contain signal. In offline
injection tests this is what caught a sign error in the photosphere prediction
that had been flagging 495 of 900 clean sources as faders.

**4.5 Background galaxies.** Hot DOGs and dusty starbursts destroyed *every*
Project Hephaistos candidate: JWST/MIRI (arXiv:2607.09460) resolved candidates D
and E as a Hot DOG at z≈0.9 and a dusty starburst at z≈0.4, both within ~1″, and
the Hot DOG sky density of ~9×10⁻⁶ arcsec⁻² "can probably account for the
contamination of all 7". The veto is astrometric and decisive: **a galaxy has no
parallax and no proper motion.** EMBER requires both at high significance.
Note that a cessation search is *intrinsically* more robust here than a
single-epoch excess search — a background galaxy does not switch off — so the
residual risk is blending, which §4.1 handles.

**4.6 WISE saturation** (§3.3) — a hard rejection on the late band.

**4.7 Solar-system objects.** IRAS, AKARI and WISE all catalogued moving
objects (Usui et al. 2014, arXiv:1403.7854); an asteroid seen once and never
again is a perfect false cessation.

**4.8 Genuine variables.** Three independent handles: the IRAS catalogue's own
VAR index (Carrigan used the same flag), Gaia DR3 variability, and position on
the colour–magnitude diagram. AGB/Mira mid-IR variable fractions are ~20–35%.
R CrB stars swing by factors of ten between exactly these epochs and are vetoed
on spectral class.

**4.9 The class discriminant: the late epoch must be FLAT.**
This is the most powerful rule in the funnel. Every natural class that varies in
the mid-infrared varies *persistently* — 14 of 17 extreme debris disks changed at
3–5 µm between 2010 and 2019 (Moór et al. 2021, arXiv:2103.00568), 5 of 6 on
sub-year timescales; ~85% of white-dwarf discs vary on all timescales. **None of
them produces a single monotonic step down followed by a decade of flatness.**
TYC 8241 2652 1 is the sole known object that does, and Moór et al. confirm "the
WISE fluxes of TYC 8241-2652-1 did not show significant changes between 2010 and
2019, before this period, its disk underwent a dramatic fading." This is where
NEOWISE earns its place.

**4.10 Youth — the astrophysical discriminant.** Every known natural analogue
sits around a young star with a collisional dust reservoir: TYC 8241 2652 1 is a
~10 Myr Sco-Cen member; the extreme debris disks are 10–200 Myr. An old,
kinematically heated main-sequence star has no such reservoir. Requiring youth
indicators to be absent removes the entire known analogue population while
keeping exactly the regime in which no explanation exists.

**4.11 Smooth disc dispersal cannot produce the signal.** Ribas et al. (2014,
arXiv:1312.0609) measure an inner-disc decay timescale of 2–3 Myr at 3.4–12 µm.
Over 27 years that is a ~10⁻⁵ fractional change. **Only discrete, stochastic
events can make a step of this size** — which sharpens what a detection would
mean and narrows the natural explanations available.

**Empirical floor.** HD 172555's mid-infrared flux "has been stable within 4 per
cent over the last 27 years, from IRAS (1983) to WISE (2010)" (arXiv:1210.6258)
— the only explicit IRAS→WISE stability measurement in the literature, and
therefore this comparison's floor. A ×30 drop sits 3.7 mag above it; a
statistically significant 10% fade does not, and is rejected on amplitude
regardless of significance.

---

## 5. The objection that must be answered

**Balbi & Ćirković (2021, arXiv:2103.02923)** show that technosignature
*duration* is "almost entirely separable from the age of the civilization that
produces it", and their Monte Carlo finds that **only the longest-duration
technosignatures are likely to be detected**. A signature defined by having
stopped is, by construction, one that was not long-lived. That is the sharpest
theoretical argument against this entire channel and it deserves a direct answer
rather than a footnote.

**The honest answer is that the rate argument is brutal and EMBER does not
defeat it.** Lacki's (2025) own decay physics gives megaswarm grind-down
timescales of 10⁵–10⁷ yr; the probability of catching a given system inside a
40-year window is then ~4×10⁻⁴ to 4×10⁻⁶ *per system that is decaying at all*.
Against a working sample of order 10⁵, that expects nothing.

Three things nonetheless make the search worth executing:

1. **The relevant timescale may not be the grind-down timescale.** Lacki's
   10⁵–10⁷ yr is the *collisional cascade* after upkeep stops. A *collapse* —
   the failure mode Blanco et al. model, with duty cycles of 0.38–1.00 — is not
   the cascade; it is the moment the swarm stops being fed energy, and nothing in
   the literature bounds how fast the thermal signature follows. TYC 8241 2652 1
   demonstrates that a natural mid-IR excess can drop by a factor of 30 in under
   two years, so the *observable* is capable of changing on a timescale ~10⁷
   times shorter than the theoretical decay estimate.
2. **The search costs one archive query per source and has never been done.**
   An untried region of parameter space with a defensible physical motivation is
   worth one pass even at unfavourable odds, particularly when the machinery
   (Sedgwick & Serjeant 2022) already exists and is validated.
3. **The by-product is a quantity the field lacks entirely.** Nobody has ever
   measured the *rate* of mid-infrared excess appearance and disappearance at
   12–25 µm. The two-sided statistic produces it as a matter of course.

Per the standing directive, that rate is an internal honesty check, not a
deliverable. **A null here changes the question; it is not written up.**

---

## 6. The existence proof, which cuts both ways

**TYC 8241 2652 1** (Melis et al. 2012, arXiv:1207.1162, *Nature* 487, 74) was
detected in the infrared by **IRAS in 1983**, stayed stable for ~25 years, and
then dropped by "a factor of about 30, over a period of less than two years"
between 2008 and 2010: "no currently available physical model satisfactorily
explains the observations." Günther et al. (2017, arXiv:1611.01371) is titled
"no smoking gun yet" and it is still unexplained fourteen years on.

This proves the phenomenon is real and that these very catalogues can capture
it — the object has precisely the IRAS-high / 2008-high / WISE-low morphology
EMBER's ladder classifies as `fade_2007_2010`, so it is the channel's natural
recovery test. It equally proves that **a single such object cannot be argued
into a technosignature**: TYC 8241 2652 1 is young, and EMBER's youth cut would
reject it. That is the correct behaviour, and it sets the standard a candidate
must beat — an old host with no dust reservoir, flat afterwards, coherent across
three instruments.

---

## 7. Method

1. **Audit** (offline). Compute the transfer, its temperature spread, the
   bandpass systematic, the beam ratio and the usable flux window for every
   pair; reject pairs that cannot work. Fetch real SVO response curves when the
   runner has egress; otherwise fall back to a documented trapezoid and *say so*
   in `rsr_source`.
2. **Acquire** (runner, sharded by RA). AKARI/IRC and IRAS PSC+FSC → Gaia DR3 →
   AllWISE, with Gaia astrometry propagated to each survey's epoch before the
   association is tested.
3. **Excess.** Fit the photospheric colour locus **empirically per band** from
   the sample itself, anchored on low quantiles so a majority-contaminated
   calibration sample cannot drag the ridge upward. This absorbs
   stellar-atmosphere error, each survey's calibration scale, and the response
   model's own error in one step.
4. **Cessation.** Transport the early excess into the late band as a blackbody
   at the fitted excess temperature — measured from two same-epoch bands where
   they exist, marginalised over 150–1500 K where they do not, with the penalty
   for not knowing it carried explicitly in the error.
5. **Adjudicate the ladder.** IRAS / AKARI / WISE separates a real fade from an
   IRAS artefact; a two-epoch detection with no usable middle epoch returns
   `no_mid_epoch` and is **not** a candidate.
6. **Calibrate** the threshold on the rising tail; **vet** through §4;
   **report** with funnel counts and coverage.

---

## 7.1 What the first run actually did, and the three bugs it exposed

Run **30203763934** (2026-07-26) reported `verdict: NO_DATA_REACHED`,
`acquisition: {"source": "cache", "n_rows": 0, "archive_reachable": false}` and
`counts.acquired: 0`. **Every one of those statements was false**, and the way
they became false is the most important operational lesson this channel has
produced. Reading the runner logs rather than the committed summary shows what
happened.

**What worked.** The `audit` job fetched all eleven SVO response curves and
produced the systematics verdict above. All six acquisition shards pulled the
**AKARI/IRC PSC in full** — 12 RA slices, 871,331 rows, checkpointed to parquet
and uploaded as artifacts:

```
[ember] akari RA[0,30)   ->  24,543 rows      [ember] akari RA[240,270) -> 177,739 rows
[ember] akari RA[30,60)  ->  24,979 rows      [ember] akari RA[270,300) -> 225,288 rows
…                                             [ember] akari RA[330,360) ->  33,296 rows
```

So `II/297/irc` is the right VizieR identifier and the RA-chunked TAP pull is
sound. **`archive_reachable: false` was simply wrong.**

**Bug 1 — IRAS was lost to an unresolvable RA column, silently.** No
`iras_psc_ra*.parquet` was written in any shard and no IRAS line appears in any
log — not even a retry failure. That combination has exactly one explanation:
the `SELECT TOP 1 *` probe *succeeded* and `resolve_columns` then failed to find
an RA column, raising a `RuntimeError` that `build_working_table`'s `_pull`
caught in a bare `except` and converted into an empty frame. The alias table
knew only J2000 spellings, and **IRAS PSC (II/125) and FSC (II/156A) are B1950
catalogues.** A schema mismatch presented as an empty catalogue.

Two defences, both now in place. Position columns are resolved from
`TAP_SCHEMA.columns` by **UCD** (`pos.eq.ra;meta.main`), which is frame- and
name-independent, with the alias list — now including the B1950 spellings — only
as a fallback; and a resolved B1950 column is **precessed to J2000** rather than
used as though it were ICRS. That second point is not pedantry: B1950 and J2000
differ by ~0.5°, some 300× the 6″ match radius, so a B1950 position used raw
would have produced zero cross-matches while every query looked healthy.

**Bug 2 — Gaia 500'd and took the whole shard with it.**

```
[gaia:0] attempt 1/4 failed: HTTPError('500')   …   attempt 4/4 failed: HTTPError('Error 500:\nnull')
```

The ESA archive refused a 20,000-row anonymous TAP upload on all four attempts.
The shard then returned an empty frame and **discarded 871,331 already-fetched
catalogue rows** because the archive that ran *second* was down. Acquisition now
(a) walks a chunk-size ladder — 5,000, then 2,000, then 500 — before concluding
the service is unusable, (b) falls back to **CDS X-Match** against
`vizier:I/355/gaiadr3`, which needs no upload quota, and (c) checkpoints the
infrared-only anchor table *before* the optical join, so a Gaia outage costs the
join and not the catalogue pull.

**Bug 3 — one IRSA cone query per source.** `_allwise_for_rows` issued a
positional query per row. At ~1 s each and a working sample of order 10⁵ that is
over a day of wall clock per shard; it never ran, so it never showed up as a
failure. AllWISE is now cross-matched in bulk via CDS X-Match against
`vizier:II/328/allwise`, with the per-object cone retained only below
`ALLWISE_CONE_MAX_ROWS = 2,000` and as the fallback.

**The meta-bug, and the guard against it.** Three unrelated causes — a schema
mismatch, a service outage, and an unrun stage — all surfaced as the single
number `acquired: 0`. Acquisition now returns a `FetchStatus` per archive
carrying `OK` / `QUERY_RETURNED_ZERO_ROWS` / `QUERY_FAILED`, **the literal query
text**, the row count and the exception; the per-shard status is written *inside*
the cache directory so it travels in the artifact; and `summary.json`
distinguishes three verdicts that used to be one:

| verdict | meaning |
|---|---|
| `ACQUISITION_QUERY_FAILED` | at least one archive query blew up — a defect in our query or in the service, **not** a statement about the sky |
| `ARCHIVES_RETURNED_ZERO_ROWS` | every query succeeded and returned nothing; the queries are printed so they can be re-issued and checked |
| `NO_DATA_REACHED` | nothing was even attempted |

`seti ember --stage probe` exists so this never has to be diagnosed from a
post-mortem again: it issues one query per primitive and reports the VizieR
schema and resolved frame for all four catalogues, the largest upload the
anonymous ESA archive will accept, whether CDS X-Match can substitute, and what
IRSA returns — with row counts and first rows. It is dispatchable on its own
(`probe_only: true`).

## 8. Honest limitations

* NEOWISE cannot see 100–300 K dust. There is no epoch after 2010 at 12–25 µm,
  and there will not be one until a new mid-infrared all-sky survey flies.
* Sensitivity is set by the shallow early epoch; this is a bright, large-excess
  search reaching of order 10⁵ sources.
* The trapezoidal response fallback carries a bandpass systematic of up to ~9%
  for W3. Real SVO curves remove most of it, and which was used is recorded.
* The Vega blackbody stand-in reproduces published zero points to 0.8% in W3 but
  6.6% in W4; the empirical locus absorbs this for the photosphere, but not for
  the dust-excess transfer.
* The duty-cycle argument of §5 is not answered, only contextualised.
* **A clean null is not a result and will not be written up.** It is a reason to
  change the question.
