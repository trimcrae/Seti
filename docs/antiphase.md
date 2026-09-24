# ANTIPHASE — an achromatic optical fade answered by a mid-infrared rise

Package `src/seti/antiphase/`, workflow `.github/workflows/antiphase.yml`,
config `config/antiphase.yaml`, tests `tests/test_antiphase.py`, results
`results/antiphase/`.

## 1. The claim

A structure that intercepts starlight and re-radiates it — a swarm of
collectors, a shell, anything large compared with a wavelength — does two
things at once, and both are measurable in public archives:

1. **The optical fades grey.** Collectors much larger than 0.5 µm take the
   same fraction out of ZTF g and r: Δ(g − r) = 0. Ordinary small grains
   redden (A_g/A_r ≈ 1.41 for R_V = 3.1 dust in the ZTF bands);
   scattering-dominated dust blues.
2. **The mid-infrared rises, simultaneously, by the energy that went
   missing.** If the occulter covers the star as seen from everywhere (a shell
   or a swarm), the optical flux lost at Earth, f·F_bol, comes back as
   thermal emission: L_IR,gained / L_opt,lost ≈ 1. An occulter that sits only
   on *our* line of sight (a companion's disk, a ring, a single clump) takes
   light out of our beam without re-radiating a comparable luminosity toward
   us, so the ratio falls far below 1.

The observable, on a kinematically old, AllWISE-photospheric field dwarf, is
therefore **a grey ZTF fade (g and r equal) coinciding with a NEOWISE W2
rise, with the IR luminosity gained equal to the optical luminosity lost
within a factor of 3**.

The budget is a sensitive probe. For a Sun-like star, an energy-balanced 5 %
grey fade re-emitted at 600 K brightens W2 by 1.06 mag and W1 by 0.31 mag; at
300 K (the habitable-zone blackbody) W2 still brightens by 0.15 mag while W1
moves by 0.004 mag (`null.ir_excess_dmag`). A 1 % fade at 300 K is a 0.03 mag
W2 rise, comparable to a NEOWISE epoch error. The IR is the more sensitive
half of the pair, which is why the detector keys on W2 (§4.1).

## 2. Novelty status

Checked by web search on 2026-09-24 (arXiv, ADS and publisher pages were
reachable only as search snippets from the sandbox; every statement below is
from those snippets and should be re-read against the papers before it is
quoted):

* **Coupled events are known only one at a time, from alerts.**
  ASASSN-21qj (Kenworthy et al. 2023, *Nature* 622, 251, arXiv:2310.08360):
  NEOWISE W1 +0.4 / W2 +0.8 mag about **900 d before** a 2021 optical
  dimming — coupled but **not simultaneous**, interpreted as a planetary
  collision afterglow whose debris later transited. Gaia-GIC-1 = Gaia20ehk
  (Tzanidakis & Davenport 2026, ApJL, arXiv:2603.10952): a young F star whose
  IR brightens while its optical dims from ~2021; a planetesimal-collision
  candidate. ZTF J2327+0019 (arXiv:2609.08923): a white dwarf obscured for
  over a year with a simultaneous WISE flare, described there as the first
  such pairing around a white dwarf. All three were found from their optical
  alerts; none from a search for the coupling.
* **The systematic dimming searches are optical only.** The ZTF main-sequence
  dipper search (Tzanidakis et al. 2025, arXiv:2508.03964; 63 M FGK stars, 81
  candidates) reports that its dippers show "no infrared excess"; it does not
  ask what the IR does *during* the dip. The ASAS-SN "big dippers" search
  (arXiv:2507.19594) is optical as well.
* **The mid-IR transient searches are IR-first and elsewhere.** NEOWISE
  image-subtraction transient programmes (De et al.; WNTR) look for
  large-amplitude IR outbursts, mostly in the Galactic plane.
* **This repository's own channels bracket it.** IGNITION requires a *flat*
  optical; the dimming channel killed faders whose IR also faded; CENOTAPH
  looks for grey dimming with *no* mid-IR (a cold shell).
* **Grey dimming alone is not new, and not a discriminant.** ASASSN-24fw
  (Forés-Toribio et al. 2025, OJAp 8, 114, arXiv:2507.03080) dimmed by
  4.1 mag almost achromatically (Δ(g − z) = 0.31 ± 0.15) — explained by an
  occulting substellar companion with a circumsecondary disk. It is exactly
  the natural grey case the **energy budget** separates: its occulter is on
  our line of sight only.

What has not been done, as far as the searches show: a **population search
for the simultaneous antiphase coupling**, with achromaticity *and* the
energy budget as the discriminants, over old field dwarfs.

## 3. Data

| Role | Source | Access | Notes |
|---|---|---|---|
| Parent + NEOWISE decade | IGNITION's tiles parent (Gaia DR3 G < 14.5 dwarfs, ϖ > 3 mas, \|b\| > 15°, RUWE < 1.4, not Gaia-variable, AllWISE −0.1 < W1−W2 < 0.15) and its NEOWISE-R epochs (`neowiser_p1bs_psd`, upload join) | IGNITION's own shard artifacts, `ignition-shard-<i>` of `ignition_run_id` (default 35923951760: 181,210 stars screened over 695/2,047 tiles) | nothing re-acquired; the shard count equals IGNITION's (12) |
| Drift correction | IGNITION's stratified ensemble (`ensemble.stratified_offsets`): per band, W1/W2 magnitude bin × \|ecliptic latitude\| band × 0.25-yr bin, ≥ 40 stars a cell, falling back to magnitude-only then global | computed per shard | removes the brightness-dependent NEOWISE fade measured by IGNITION (W1 0.24→1.45 mmag/yr, W2 1.7→3.5 mmag/yr from W1 8–9 to 11.5–12) |
| Optical | ZTF DR light curves, g and r | IRSA `nph_light_curves`, one PM-propagated cone per star (epoch 2021.0, 1.5″ + half the 2018–2025 sweep), `catflags == 0`, dominant object id per filter | dec ≥ −31°; a star whose median ZTF mag is brighter than 12.5 in either band is `ZTF_SATURATED` and not evaluated |
| Controls | NEOWISE cone (IGNITION `fetch_neowise_cone` + `reduce_star`), ZTF, Gaia alerts light curves (G), ASAS-SN Sky Patrol (if the client installs and the service answers), CDS Sesame, VizieR `I/355/gaiadr3` and `II/328/allwise` | runner | §5 |
| Blend model | Gaia DR3 neighbours within 20″ | VizieR `I/355/gaiadr3` ASU (the route that answered while ESA TAP was dark) | survivors only |

## 4. Method

### 4.1 The detector (`coupling.py`, pure)

ZTF points are binned at every NEOWISE visit (±60 d, ≥ 5 points, 5σ MAD
clip, error `max(1.4826 MAD/√n, 3 mmag)` ⊕ 7 mmag season-to-season floor).
At the ≥ 6 epochs where g, r, W1 and W2 all exist, the **unfaded level** is
the median of the optically brightest half; the IR reference is the IR at
those same epochs. **Faded epochs** F: g *and* r fainter by ≥ 4σ and by ≥
0.02 mag. Then:

* **IR answer**: the weighted-mean W2 brightening over F ≥ 3σ, and W1 not
  fainter by ≥ 2σ. W1 is not required to rise, because a ~300 K re-radiator
  adds almost nothing at 3.4 µm; the second band that confirms the event is
  the optical, measured in two filters by another instrument. A W1 rise as
  well is recorded (`ir_two_band`) and constrains the temperature.
* **Specificity** (`ir_contrast`): the W2 brightening at F against the star's
  own W2 scatter at the other epochs (1.4826 MAD, which contains the noise and
  any unrelated variability) must be ≥ 3σ. The regression slope of the IR
  flux ratio on the optical deficit is reported but is not a gate (its
  flux-space errors grow with the excess, so a large event down-weights
  itself).
* **Lag**: the correlation of deficit and IR excess with the IR shifted by
  −6…+6 epochs. An IR that leads (ASASSN-21qj) is labelled, not discarded.
* **Chromaticity**: `dg = k·dr` through the origin with errors in both
  (effective variance). `GREY` if |k − 1| ≤ 2σ_k **and** k is below 1.41 by
  ≥ 3σ_k; `REDDENING_ISM_LIKE` / `_STEEP` / `_SHALLOW`, `BLUEING` or
  `AMBIGUOUS` otherwise.

Labels: `NO_OPTICAL`, `NO_IR`, `INSUFFICIENT_MATCHED`, `NO_FADE`
(`IR_RISE_NO_FADE` when the IR rises anyway), `FADE_IR_FADES`, `FADE_IR_FLAT`,
`W1_ONLY`, `IR_BANDS_DISAGREE`, `NOT_PROPORTIONAL`, `COUPLED`.

### 4.2 The energy budget (`energy.py`, pure)

F_bol from G and the Andrae et al. (2018) BC_G(Teff) (Teff from
`teff_gspphot`, else a dwarf BP−RP table); the optical loss is `f·F_bol`.
The IR gain is a single-temperature blackbody through the W1/W2 excess flux
densities (zero points 309.540 / 171.787 Jy), fitted on a 100–3000 K grid;
**every temperature the two points allow** (χ² ≤ χ²_min + 2.3) is carried
into a range of ratios. Verdicts: `BALANCED` (the range meets [1/3, 3]),
`IR_DEFICIT`, `IR_SURPLUS`, `STELLAR_TEMPERATURE` (best and lowest allowed
T > 1800 K: a photosphere, not grains), `UNDETERMINED`. A W2-only excess
leaves low temperatures open, so its ratio range is wide and is flagged
`t_range_unconstrained`; that is stated, not hidden.

### 4.3 The ladder (`classify.py`, pure) — first match names the event

| label | natural mechanism | test |
|---|---|---|
| `YSO_DIPPER` | accretion-disk clumps; the disk's warm dust varies with them | star-forming-region box, AllWISE W1−W2 > 0.15 or W2−W3 > 0.5 |
| `RCRB_LIKE` | carbon cloud condensing on our line of sight | M_G < 2.5 and depth > 1 mag |
| `LPV` | pulsation, optical and IR with a phase offset | Lomb–Scargle P ≥ 60 d, FAP < 10⁻⁶, amplitude > 0.05 mag, **repeating in ≥ 3 cycles** (a single long fade also gives a long-period peak), or a red giant host |
| `EB_DUSTY_DISK` | a companion's disk eclipses the primary | significant P < 60 d, repeating |
| `NATURAL_CHROMATIC` | ordinary grains | chroma REDDENING_* or BLUEING |
| `STELLAR_IR` | companion photosphere or blend | energy `STELLAR_TEMPERATURE` |
| `GREY_IR_DEFICIT` | a line-of-sight-only occulter (ASASSN-24fw) | energy `IR_DEFICIT` |
| `GREY_IR_SURPLUS` | IR not from the absorbed light | energy `IR_SURPLUS` |
| `LAGGED` | collision afterglow then transit (ASASSN-21qj) | best lag ≥ 2 epochs from 0 and its correlation ≥ 0.2 above lag 0 |
| `BLEND_PREDICTED` | a Gaia neighbour moving through the 6″ WISE beam | the neighbour model (§4.4) predicts ≥ half the W1 or W2 gain at F |
| `COUPLED_INCOMPLETE` | — | coupled, no natural reason, but achromaticity or the budget could not be measured |
| `ANTIPHASE_CANDIDATE` | — | grey **and** balanced measured, nothing natural applies |

An untested rung is listed in `untested` and is never a pass.

### 4.4 Contamination model

* **(a)/(c) Raw photometry only**: ZTF DR PSF photometry, NEOWISE single
  exposures; no forced or difference photometry is mixed in.
* **(b) Blends.** A neighbour drifting *into* the 6″ WISE beam raises the IR,
  but it cannot fade the star in ZTF's ~2″ PSF photometry (a neighbour moving
  closer adds flux there too). The blend model propagates each Gaia neighbour
  (VizieR `I/355/gaiadr3`, epoch 2016.0) with both proper motions to every
  NEOWISE epoch, estimates its W1 from G and BP−RP with a dwarf relation made
  0.75 mag brighter (the conservative end), weights it by the Gaussian PSF at
  the separation of the day (FWHM 6.08″/6.84″), and asks what fraction of the
  observed gain at F it predicts.
* **(d) NEOWISE drift** is a fade (it runs opposite to the signal) and is
  removed by the magnitude- and ecliptic-latitude-matched ensemble anyway.
* **(e) Chance alignment.** The pair null (each star's IR against a random
  other star's optical, aligned by nearest time) and the shift null (a star's
  own optical moved by ±2, ±4 epochs) run on every shard's own stars, 30
  pair rounds. A candidate's FAP is the fraction of null trials whose coupling
  score reaches its own; it must satisfy FAP × N_tested < 0.1. The null's
  resolution floor, N_tested / (trials + 1), is reported, and a candidate list
  the null cannot resolve is a `DEGRADED (null_resolution…)` run.
* **Saturation.** ZTF saturates near 12.5–13 mag; a star brighter than 12.5 in
  either band's median is excluded, not evaluated. IGNITION's parent is
  G < 14.5, so this removes its bright end.

### 4.5 Efficiency

Grey fades of depth 3, 5, 10 and 20 % over two consecutive NEOWISE epochs,
with the energy-balanced IR excess at 300, 600 and 1000 K, are injected into
≤ 150 real stars per shard and scored by the full ladder
(`summary.json["injection"]`: coupled fraction and full-ladder fraction).

## 5. Positive controls (run first; they gate the shards)

| control | kind | what the data can and cannot test |
|---|---|---|
| Gaia-GIC-1 / Gaia20ehk | coupled event | dec −34.2°: **south of ZTF**. Optical from the Gaia alerts light curve (G only) and ASAS-SN if it answers, so achromaticity is untestable: the most it can be is `RECOVERED_COUPLED_INCOMPLETE` |
| ASASSN-21qj | coupled, IR leads by ~900 d | dec −39.0°: **south of ZTF**; ASAS-SN only if it answers. The IR rise predates the optical: expected `LAGGED` or not simultaneous |
| ASASSN-24fw | natural grey occultation | in ZTF, but the dimming (2024-09…2025-05) is **after NEOWISE ended** (2024-08): expected `UNTESTABLE_NO_FADE_IN_WINDOW` |
| ZTF J2327+0019 | coupled (white dwarf) | coordinates only via Sesame; a faint WD is probably below NEOWISE single-exposure depth |
| RW Aur, KH 15D, V718 Per, EE Cep, ES Aql, SV Sge | natural classes (YSO, disk-occulted binaries, R CrB) | whatever overlap the data hold; each must be `NATURAL` if it comes out coupled |

Gate: `FAIL` if any control comes out `ANTIPHASE_CANDIDATE` (the ladder
leaks) or a coupled-event control whose optical fade *is* inside the NEOWISE
window is not coupled (the detector is blind); `PASS` if at least one
coupled-event control is recovered and nothing failed; `UNTESTED` if no
control's data overlap, in which case the gate rests on the injections alone
and the summary verdict carries `CONTROLS_UNTESTED`. The shard matrix does not
run on `FAIL` unless `ignore_gate=true`, and then the verdict still carries
`CONTROLS_FAILED`.

The coverage problem is itself a finding: the three best-documented coupled
events in the literature sit where this channel's survey data cannot see them
(two south of ZTF, one after NEOWISE), which is also why no survey has
reported the coupling as a population.

## 6. Running

```
# register on main first (docs/channel-brief.md §0.6), then:
workflow_dispatch antiphase.yml ref=<branch>
  stage=all ignition_run_id=35923951760 shards=12 max_stars=0 budget_s=16200
# re-reduce a finished run
  reduce_only_run_id=<run>
```

Outputs: `controls.json`, `summary.json` (verdict, gate, funnel, null,
injection grid, self-consistency), `coupled.csv` (every COUPLED star with its
null FAP), `candidates.csv` (the vetted survivors), `shard_s*of12.json`.

## 7. Run log

(Filled from committed results only. Times UTC, with US Eastern in brackets.)

### 7.1 Pilot run 36006324981 (2026-09-24 13:32–15:10 UTC [09:32–11:10 EDT])

`stage=all shards=12 max_stars=300 budget_s=2400 ignore_gate=true`, parent
from IGNITION run 35923951760.

* **The pipeline runs end to end on real data.** 182,490 parent stars in the
  12 IGNITION shard artifacts; 139,530 in the ZTF footprint; 138,223 with ≥ 6
  NEOWISE epochs after 2018.2 in both bands. The stratified drift correction
  put 92–99 % of epochs in a fine (magnitude × |β|) cell.
* **ZTF throughput is the limit.** The IRSA light-curve API's positional
  query took **~56 s per star per worker** (1,452 OK + 490 NO_ROWS + 130
  saturated + 7 failed in 40 min on 4 workers × 12 shards). The batched
  alternative (one `TAP_UPLOAD` join against the ZTF objects table, then
  multi-ID light curves; run 36018334549) did not return its first join inside
  its 30-minute budget — pyvo's `run_sync` has no timeout of its own — so the
  full survey spends its ZTF budget down a W2 prescore list and reports the
  prescore it reached (§4.1, `coverage.complete_above_w2_prescore`).
* **Funnel of the 1,452 evaluated**: `NO_FADE` 826, `NO_OPTICAL` 207 (one of
  g/r missing), `INSUFFICIENT_MATCHED` 198, `FADE_IR_FLAT` 98, `FADE_IR_FADES`
  65, `IR_RISE_NO_FADE` 58, **`COUPLED` 0**. Sums to 1,452;
  self-consistency checks all true.
* **Null**: 43,110 pair trials, 0.3 coupled per round of 1,437 stars
  (2 × 10⁻⁴ per star), shift null 1 in 5,748. Real sample: 0 of 1,452.
* **Injections (1,170 real series per cell)**: coupled-recovery 0.20/0.44/0.43
  at 3 % depth (300/600/1000 K), 0.68/0.85/0.86 at 5 %, 0.78/0.89/0.89 at 10 %,
  0.81/0.89/0.89 at 20 %. Full-ladder recovery (coupled **and** measured grey
  **and** balanced) is 0.01–0.05 at 3 %, 0.18–0.23 at 5 %, 0.68–0.79 at 10 %,
  0.74–0.83 at 20 %: **achromaticity is only measurable for fades ≳ 10 %** —
  below that k = Δg/Δr cannot be told from 1.41 at 3σ, and such events are
  `COUPLED_INCOMPLETE`, never candidates.
* **Controls**: gate `UNTESTED`. Gaia-GIC-1 — no ZTF (dec −34°), and no NEOWISE
  epoch survives IGNITION's frame cuts (a faint star at b ≈ −4°); ASASSN-21qj —
  21 NEOWISE epochs, no ZTF, ASAS-SN V and g reached but V ended in 2018 as g
  began, so requiring both at once matched nothing (fixed: each band is now
  tried alone); ASASSN-24fw — 12 matched g/r/W1/W2 epochs, no fade in the
  NEOWISE window (its dimming began 2024-09, after NEOWISE), as expected;
  ZTF J2327+0019 — Sesame does not resolve the name; RW Aur, EE Cep, SV Sge —
  no ZTF rows (bright/saturated); KH 15D — no NEOWISE epoch survives the
  cuts (NGC 2264); V718 Per, ES Aql — 1–3 matched epochs. None of the natural
  controls reached the ladder; the VSX natural-class sample stage was added
  to measure the ladder on hundreds of known natural variables instead.

### 7.2 The positive control that reached the ladder: ASASSN-21qj

Run 36030068228 (controls 16:52 UTC [12:52 EDT]) — with each optical band now
tried alone — put ASASSN-21qj through the ladder for the first time: ASAS-SN
g, 12 epochs matched to its 21 NEOWISE epochs, **5 faded epochs 2020.9–2022.9,
depth 0.17 mag**. The simultaneous test said `FADE_IR_FADES` (W2 −103σ against
the reference), and the gate read **FAIL** (`MISSED`), which stopped the
survey shards — the gate doing its job. The reason is physical, not a
detector fault: 21qj's NEOWISE brightening of 2019–2020 sits in the
optically-unfaded epochs that define the reference, so the IR *at* the faded
epochs is below it. The lag scan already had it — best lag **−4 epochs (IR
leading by ~2 yr)**, r = 0.60 against −0.12 at lag 0 — which is Kenworthy et
al.'s afterglow-then-transit (they give ~900 d).

**What was changed, stated as an a-posteriori change:** a non-simultaneous
event is now labelled `LAGGED_COUPLING` when the best shift is ≥ 2 epochs,
correlates at ≥ 0.5 and ≥ 0.2 above lag 0, and W2 at the lag-shifted faded
epochs is ≥ 3σ above the star's all-epoch median; it is always `NATURAL`
(class `LAGGED`), because the claim here is the *simultaneous* answer. The
rule was written after seeing this control, so 21qj's recovery is a
consistency check on it, not an independent validation. Offline tests pin the
rule on a synthetic afterglow that has faded by the transit, and on the case
where the afterglow is still partly present (`COUPLED` → `LAGGED`).

Run 36032055387's controls (17:34 UTC [13:34 EDT]): **ASASSN-21qj
`RECOVERED_NATURAL`** (`LAGGED_COUPLING`, W2 +193σ at the shifted epochs);
RW Aur (ASAS-SN g, 6 faded epochs, depth 0.47 mag) `FADE_IR_FADES` with the IR
*lagging* (best lag +5, r = 0.80) but no W2 rise at the shifted epochs —
not coupled; EE Cep one faded epoch, `IR_BANDS_DISAGREE`; the rest
untestable as in §7.1. **Gate `PASS`** on one recovered coupled-event
control, no natural object passed as a candidate. Gaia-GIC-1 remains
untestable: no ZTF, and IRSA returns no NEOWISE single-exposure rows at 2.5″
or 4″.
