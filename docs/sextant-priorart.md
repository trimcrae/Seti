# SEXTANT — the prior-art position

*Adjudication of the claim: screen Gaia's per-observation asteroid astrometry
(`gaiafpr.sso_observation`, 46,264,083 observations of 156,823 objects;
`gaiadr3.sso_observation`, 23,336,467 of 158,152) for a **population** of objects
whose milliarcsecond ephemeris residuals — taken against independent JPL/MPC
orbits and projected on the Gaia scan direction — carry an anomalous,
non-gravitational, non-binary signature, read as a technosignature.*

**Written 2026-08-25. Evidence: `results/sextantlit/websearch_log.md` (34
WebSearch queries, all recorded with result counts) plus the runner battery in
`scripts/sextantlit_fetch.py` / `.github/workflows/sextantlit.yml` (89 arXiv
queries, 22 ID verifications, 12 full texts, term-occupancy matrix, OpenAlex
citing sets, VizieR ReadMes) — which has NOT yet been dispatched. Every verdict
below is marked with the standard of evidence it actually rests on.**

---

## 0. The evidential standard reached here, stated first

The vnprobelit sweeps hold this repository to three things: *record every query
with its result count*, *verify a paper's real title before citing it*, and
*count term occupancy over full texts*. This document reaches the first fully,
the second partially, and the third **not at all**, because the sandbox has no
egress:

`curl` and `WebFetch` were both refused for `export.arxiv.org`, `arxiv.org`,
`api.openalex.org`, `api.semanticscholar.org`, `ui.adsabs.harvard.edu`,
`www.aanda.org`, `vizier.cds.unistra.fr`, `ssd.jpl.nasa.gov` and
`en.wikipedia.org` — every one a `403` on CONNECT, confirmed against the proxy's
own `recentRelayFailures`. **No full text was retrievable in-sandbox, so no term
was counted here.** WebSearch works, and every claim about a paper's *content*
below comes from a search-index summary of that paper, not from the paper.

That matters for one verdict in particular (Q2, the circularity question) and it
is flagged where it bites. It does **not** soften the headline finding, which is
negative and rests on titles, abstracts and author lists that multiple
independent queries returned consistently.

**One title correction, immediately.** `docs/substitute-surveys.md` and
`docs/loom.md` name the nearest neighbour *"Gaia astrometric asteroid binary
candidates"*. That is not its title. The paper is **Liberato, Tanga, Mary,
Minker, Carry, Spoto, Bartczak, Sicardy, Oszkiewicz & Desmars, "Binary asteroid
candidates in Gaia DR3 astrometry", A&A 688, A50 (2024)**, doi
`10.1051/0004-6361/202349122`, arXiv:2406.07195. The VizieR id `J/A+A/688/A50`
is correct. Fix the name before it is cited anywhere.

---

## Q1 — Has anyone searched Gaia SSO astrometric residuals for anomalous / non-gravitational / unexplained acceleration signatures?

> ### VERDICT: **OCCUPIED — more heavily than the brief assumes.**
> Nobody has run an *unmodelled-anomaly* search over Gaia SSO residuals. But the
> non-gravitational acceleration has already been **fitted, using Gaia asteroid
> astrometry, over 54,540 objects including 54,094 main-belt bodies — and the
> main belt came back null.** The sentence "nobody has looked for
> non-gravitational acceleration in Gaia's asteroid astrometry" is false.

**The paper that occupies it.** Dziadura, Bartczak & Oszkiewicz, *Assessing the
detection of the Yarkovsky effect using Gaia DR3 and FPR catalogues*, A&A 693,
A31 (2025), arXiv:2411.09750 (submitted 14 Nov 2024). They ran OrbFit
least-squares solutions for **446 NEAs (93 PHAs) and 54,094 inner-main-belt and
Mars-crossing asteroids**, fitting six orbital elements *plus* the transverse
non-gravitational acceleration A2, on Gaia FPR astrometry (66-month arc)
complemented by MPC optical and JPL radar. Result: a robust Yarkovsky detection
in **43 NEAs**, nine of them new relative to DR3, several at S/N > 10 — and **no
Yarkovsky drift detected for any main-belt asteroid**.

That is a population-scale non-gravitational-acceleration screen, on SEXTANT's
own dataset, at SEXTANT's own scale, already published, already null in the belt.

**What is genuinely *not* done, and why it is a thin gap.** Their A2 is a
*parameter inside the orbit solution*, fitted with an assumed functional form
(transverse, Yarkovsky-shaped) on a *merged* Gaia + ground-based arc. It is not
a residual-space search, it does not look for accelerations of the wrong shape,
and it is not framed as anomaly detection. The distinction is real. It is also
narrow: an object with a genuine unmodelled along-track acceleration large
enough to matter would, in most geometries, show up as a fitted A2 or as a fit
that will not close.

**The adjacent occupancy that also counts.**

* **Residual-space mining of Gaia SSO data is an active programme** — for
  binarity, by the Gaia SSO team itself (Q2).
* **Anomalous-acceleration searches in asteroid astrometry exist under a
  new-physics prior**: arXiv:2107.04038 (*Novel constraints on fifth forces and
  ultralight dark sector with asteroidal data*, 9 NEOs) and arXiv:2309.13106
  (Bennu, from OSIRIS-REx tracking; Comms. Phys. 2024). Small samples,
  ground/radar data, not Gaia epoch astrometry — but the *question* "is there an
  unexplained acceleration in this asteroid's astrometry" is not new.
* **Population outlier detection on minor planets** exists (SNAPS,
  arXiv:2405.20176) but is **photometric**: no dynamical residual, no
  non-gravitational parameter. This matches what `docs/loom.md` §6 already
  recorded.
* **Anomaly detection on Gaia data** is a live ML field — but on stars and
  astrometric-binary/exoplanet solutions, not on `sso_observation`.

*Evidence: WebSearch #2, #8, #9, #15, #18, #25, #31, #33. Runner queries
`Q1_*` (18 queries) will supply totalResults and the zero-hit set.*

---

## Q2 — Liberato & Tanga: what exactly did it do, and how much of SEXTANT does it occupy?

> ### VERDICT: **HEAVILY OCCUPIED. This is not a neighbouring paper; it is the same experiment with a different hypothesis attached.**
> The DR3 paper occupies the observable, the projection, the population framing
> and the residual statistic. Its **2026 FPR successor occupies the dataset as
> well, adds the noise model, the Monte-Carlo null, false-discovery-rate control,
> and — decisively — an explicit *linear-trend* detection stage in the same
> residuals.** What remains to SEXTANT is the *reference orbit* and the
> *interpretation*, not the method.

### What the DR3 paper did

* **Sample.** Every Gaia DR3 asteroid with at least one "window" of ≥10
  consecutive observations — ~30,000 objects.
* **Observable.** The **along-scan (AL) projection of the post-fit orbital
  residual**. Gaia's astrometry is effectively one-dimensional per transit; the
  AL direction is known from the satellite attitude and rotates by a few degrees
  over hours. *This is exactly the projection `docs/substitute-surveys.md`
  presents as SEXTANT's design insight (`position_angle_scan`). It is their
  standard practice, not a new idea.*
* **Statistic.** A dedicated **period-detection** method on the AL post-fit
  residuals within each window, i.e. a search for a *periodic* photocentre
  wobble caused by an unresolved companion.
* **Vetting.** Statistical tests, then physical filters — the detected period is
  interpreted as a two-body wobble of uniform-albedo spheres, and systems with
  implausible densities or separations are cut.
* **Yield.** 343 candidates across 410 windows (≈352 in the ESA release, 8 Aug
  2024) — described as the first large list of binary candidates obtained from
  astrometry alone.

### What the FPR successor did — this is the one that hurts

**Liberato + 9 co-authors, *Follow the wobble: Statistical methods to detect
astrometric binary asteroids in Gaia FPR*, arXiv:2605.22702 (21 May 2026).** It
works on **Gaia FPR astrometric residuals — SEXTANT's exact table** — and its
stated method components are:

1. a dedicated noise model for post-fit residuals consistent with the Gaia FPR
   error model (i.e. the random/systematic split SEXTANT's brief highlights);
2. **identification and detrending of linear systematics in the residuals**
   before the period search;
3. **a trend-detection stage in its own right** — "linear trends in astrometric
   residuals may signal the presence of wide binaries, even when the wobble
   period cannot be directly measured";
4. a period-search algorithm and multi-window detection;
5. Monte-Carlo statistical modelling of the data, with **noise-only control
   simulations** yielding 88% fewer detections than the real data;
6. **explicit false-discovery-rate control** in the selection.

Yield reported: 9 known binaries recovered, 25 candidates overlapping
Pan-STARRS, 99 overlapping the DR3 search, and **45 objects with residual trends
suggestive of wide binaries**.

### Occupancy audit, line by line

| SEXTANT design element | occupied by | status |
|---|---|---|
| `gaiafpr.sso_observation` as the search table | Liberato+ 2026 | **occupied** |
| Residual as the observable | Liberato+ 2024, 2026 | **occupied** |
| Projection on `position_angle_scan` (AL) | Liberato+ 2024 | **occupied** |
| Random/systematic error model used, not assumed | Liberato+ 2026 | **occupied** |
| Population-scale screen (10⁴–10⁵ objects) | Liberato+ 2024 (30k), Dziadura+ 2025 (54k) | **occupied** |
| Monte-Carlo null of matched non-signal data | Liberato+ 2026 | **occupied** |
| Multiple-testing control over the screened set | Liberato+ 2026 (FDR) | **occupied** |
| **Linear trend** in residual as a detection statistic | Liberato+ 2026 | **occupied** |
| Binary contamination handled explicitly | Liberato+ 2024/2026 *is* the catalogue | **occupied** |
| Residual against an **independent** (JPL/MPC) orbit | — | **open, see caveat** |
| Secular/cross-epoch acceleration rather than within-window signal | — | **open** |
| Non-binary reading of a residual anomaly | — | **open** |
| Technosignature interpretation | — | **open (Q3)** |

### The caveat that decides how much is left — UNRESOLVED here

Every description reachable in-sandbox says the residuals are **post-fit**, from
an orbital fit made "using only the astrometric measurements" of the Gaia
asteroids. If that fit is **Gaia-only**, then it is minimised on the same data,
any *secular* along-track acceleration is absorbed into the fitted elements, and
SEXTANT's move to an external JPL/MPC orbit does recover a component their
residual cannot contain. If instead they fit a joint Gaia+MPC solution — as
Dziadura et al. do, and as the Gaia FPR orbit paper does when assessing against
independent orbits — the remaining gap narrows to almost nothing.

**This is the single most load-bearing unresolved fact in the whole
adjudication, and it needs the full text.** `FULLTEXT_TARGETS` in
`scripts/sextantlit_fetch.py` fetches both papers and the term matrix will
settle it (`OrbFit`, `JPL`, `Minor Planet Center`, `AstDyS` counts per document).
Nothing in SEXTANT should be built until it is answered.

*Evidence: WebSearch #1, #3, #4, #5, #6, #23, #26, #27, #28. Runner queries
`Q2_*` (13 queries) + full texts `liberato_dr3_binaries`,
`follow_the_wobble_fpr`, `astrometric_binary_asteroids`.*

---

## Q3 — Has any technosignature paper used asteroid astrometry at all?

> ### VERDICT: **UNOCCUPIED IN EXECUTION, OCCUPIED IN PROPOSAL — and the proposal end is more crowded than it was when LOOM was written.**
> No technosignature search has ever used minor-planet astrometry — Gaia's or
> anyone's. But "anomalous asteroid acceleration as a technosignature" is named
> in a 2026 review, has a submitted paper behind it by the right authors, and now
> has a published artificial-origin claim attached to a specific non-gravitational
> object.

**What exists:**

* **Lazio, *Solar System Technosignatures* / *Technosignatures in the Solar
  System*, arXiv:2606.13797 (June 2026)** — the review LOOM already positions
  against. Treats anomalous non-gravitational acceleration, manoeuvres, secular
  SRP/low-thrust change and the Yarkovsky/outgassing confound; cites **Lazio &
  Mahabal, *On Anomalous Asteroid Accelerations*, Acta Astronautica
  (submitted)**. I could not find that paper anywhere independent of Lazio's own
  citation of it — it remains a citation, not a retrievable record. Nothing
  found suggests either uses Gaia or per-observation astrometry.
* **Hibberd, Crowl, Gómez de Olea Ballester & Loeb, *Is the Dark Comet 1998
  KY₂₆ the Spacecraft Phobos 1?*, arXiv:2606.01288 (31 May 2026, v2 2 Jun).**
  A published artificial-origin claim about an object *selected by* its
  non-gravitational acceleration, motivated by Hayabusa2's 2031 rendezvous.
  Method: mission history and trajectory backtracking — **not** astrometric
  residual mining. Its existence means the "artificial object identified by
  non-gravitational behaviour" framing is now in print with Loeb's name on it.
* **Gaia in SETI is stellar only** — SETI Ellipsoid target prioritisation
  (arXiv:2206.04092, 2402.11037), stellar-engine limits (arXiv:2608.16060),
  VLBI localisation. **No SETI use of Gaia solar-system data was found in any
  query.**
* **Technosignature astrometry means interstellar objects**, not minor planets:
  3I/ATLAS follow-up protocols, Breakthrough Listen GBT observations,
  arXiv:2508.16825.

**What does not exist:** any paper, in any framing, that takes minor-planet
astrometric residuals as a technosignature observable.

**On LOOM's established unoccupied terms.** Nothing found here challenges
`"ephemeris residual"` (0 arXiv hits), `abs:"self-replicating" AND abs:"solar
system" AND abs:"technosignature"` (0), or the Ellery arXiv:2510.00082
characterisation. But those were *counted* by the vnprobelit sweep over 34 full
texts; **I could not recount them here**, so they are carried over as prior
findings, not re-established. The runner re-issues all three
(`Q1_ephemeris_residual_phrase`, `Q3_self_replicating_solar_system_technosignature`,
`Q3_ellery_selfreplicating`) and re-counts them over a 12-document corpus.

*Evidence: WebSearch #10, #11, #12, #17, #21, #22. Runner queries `Q3_*`
(20 queries) + full texts `lazio_solar_system_technosig`, `ky26_phobos1`,
`ellery_selfrep`.*

---

## Q4 — Who else computed O−C for Gaia asteroid observations against independent orbits, and what is in the tails?

> ### VERDICT: **OCCUPIED. The construction has been executed, the residual
> distribution has been published, and the error model that must be removed
> first has been published by JPL.** The tails are characterised. What has *not*
> been published is a per-object outlier list at catalogue scale — the outlier
> sets that exist are small and live in element space, not residual space.

**The construction, already executed.** *Comparison of the Gaia-CRF3 and
planetary ephemerides via asteroid observations*, A&A 2025 (`aa52534-24`): take
**1001 asteroids**, compute osculating orbits **from data independent of Gaia**
under DE440, propagate to the Gaia observation epochs, and take the positional
differences. That is SEXTANT's O−C construction, run — for reference-frame
orientation, at N = 1001. Orientation offsets ~10 mas were found and attributed
to systematic biases in historical ground-based astrometry; adding Gaia
observations pulls them sub-mas.

**The tails, already published.**

| quantity | value | source |
|---|---|---|
| AL residuals within ±5 mas | 96% | Gaia DR3/FPR SSO papers |
| AL residuals sub-mas | 52% | ″ |
| across-scan residuals | strongly non-Gaussian, σ ≈ 50× AL, non-zero mean, negative tail | ″ |
| DR3 outlier fraction flagged by the pipeline | 0.58% | ″ |
| DR2 observations discarded in outlier rejection | ~1% (27,981) | ″ |
| orbits with anomalous \|Δa\|/a > 1×10⁻⁷ vs independent solutions | **110** (~0.2% of the compared set) | Gaia FPR orbit assessment, A&A 680, A37 = arXiv:2310.14699 |

**The error model, published by the people who would referee SEXTANT.**
Fuentes-Muñoz, Farnocchia, Naidu & Park (JPL), *Asteroid Orbit Determination
Using Gaia FPR: Statistical Analysis*, AJ 167, 290 (2024),
doi `10.3847/1538-3881/ad4291`. Two findings that constrain SEXTANT directly:
**centre-of-light offsets due to phase variation must be modelled to fit the
data at all**, and **the reported Gaia uncertainties are optimistic** unless
inflated for centre-of-mass error. Any residual excess SEXTANT finds will be
attributed to these before it is attributed to anything else.

**Related occupancy of the same residuals:** dynamical masses from mutual
perturbations (AJ, doi `10.3847/1538-3881/ace52b`; and a 2026 close-encounter
mass paper), Gaia-DR2/INPOP ephemeris work (arXiv:2203.01586), re-weighting
schemes for Gaia asteroid OD (arXiv:2604.08820), and the ground-survey analogue
— Vereš, Farnocchia, Chesley & Chamberlin, Icarus 296, 139 (2017),
arXiv:1703.03479, which does population-scale residual statistics over the 13
most productive surveys and turns them into an **outlier-robust weighting
scheme**. That is the field's instinct in one line: when the residual tail is
heavy, downweight it. Nobody goes looking in it.

**The one honest opening here:** no catalogue-scale, per-object residual outlier
list from Gaia SSO data has been published. The 110-orbit set is element-space
and small; the pipeline's 0.58% rejected observations are flagged per
observation (`is_rejected`, `astrometric_outcome_ccd`) but nobody has asked
*which objects* they concentrate on.

*Evidence: WebSearch #7, #13, #14, #24, #32, #33, #34. Runner queries `Q4_*`
(16 queries) + full texts `gaia_fpr_orbits`, `gaia_dr2_inpop`.*

---

## Q5 — Non-gravitational acceleration in the main belt: dark comets, Yarkovsky catalogues, and sensitivity

> ### VERDICT: **OCCUPIED for near-Earth objects; NULL for the main belt — and
> the main-belt null was obtained on SEXTANT's own data.** Separately, the
> sensitivity argument in `docs/substitute-surveys.md` does not survive contact
> with the error budget.

**Yarkovsky catalogues.** Del Vigna et al., *Detecting the Yarkovsky effect
among near-Earth asteroids from astrometric data*, A&A 617, A61 (2018),
arXiv:1805.05947: **87 reliable detections, 24 marginally significant**, and —
directly relevant — an explicit list of detections judged **spurious** because
they are "unrealistic or not explicable with the Yarkovsky effect". That list is
simultaneously prior art (someone has already found the anomalous non-Yarkovsky
accelerations and dismissed them) and the contamination catalogue SEXTANT would
need. Successors: Dziadura et al. 2023 (Gaia DR3, NEA densities) and Dziadura
et al. 2025 (Q1 above).

**Dark comets.** Seligman/Farnocchia et al. (arXiv:2212.08115, PSJ 2023;
arXiv:2310.02733; arXiv:2412.07603 / PNAS 2024): a population of **near-Earth**
bodies, roughly 3–15 m in radius, with non-gravitational accelerations
inconsistent with radiative effects and no visible coma, split into an inner and
an outer population. Explained by sub-detection-threshold outgassing. **They are
not a main-belt phenomenon**, and the mechanism scales with heliocentric
distance in a way that suppresses it in the belt.

**Directly material to this repository:** *Non-gravitational acceleration
indicative of cometary activity of near-Earth object*, Nature Astronomy 2026,
doi `10.1038/s41550-026-02913-7` (Farnocchia et al.) reports that **875163
(1998 SH2)** — one of LOOM's two standing momentum-ceiling exceedances
(`docs/substitute-surveys.md`, `docs/loom.md` §5) — was found 19σ from its
gravity-only predicted position at its August 2025 close approach, and then
**confirmed with a faint coma and tail and continuous dust release over late
August to late September 2025**. It is a low-activity dark comet. It is
explained. *LOOM's candidate list should be updated to one.* (Reported here
because the sweep turned it up; `docs/loom.md` is not mine to edit.)

**Sensitivity — the claim that does not hold.** `docs/substitute-surveys.md`
argues that a Gaia sample "7× smaller than Rubin's but ~10³× more precise is a
*stronger* test of the same hypothesis". That treats the astrometric noise as
the error budget. It is not:

* The photocentre–barycentre offset from **irregular shape plus solar phase
  angle** is **3.3–5.4 mas for (21) Lutetia**, varying systematically with phase
  angle — i.e. **3–5× Gaia's per-transit AL precision**, and *phase-correlated*,
  so it neither averages down nor looks like white noise. The binary literature
  states plainly that shape-and-phase offsets are "comparable to or larger than
  the expected binary-induced signal" in some configurations, and JPL's FPR
  analysis says centre-of-light phase offsets **must be modelled** before the
  data can be fitted at all.
* Gaia's arc is 66 months (FPR). A constant along-track acceleration accumulates
  as ½at², so the *long* ground-based arcs that Del Vigna and Dziadura fold in
  do much of the work; Gaia's contribution is precision at a small number of
  epochs, not baseline.
* Empirically, the test has been run: 54,094 main-belt objects, Gaia FPR + MPC +
  radar, six elements plus A2 — **no detection**.

The correct statement is that Gaia's precision buys sensitivity to *short-period
structure within a transit window* (which is why it detects binaries), and buys
comparatively little on *secular* acceleration, where the systematic floor is
photocentre modelling and the leverage is arc length.

*Evidence: WebSearch #8, #16, #18, #19, #20, #25, #29, #30. Runner queries
`Q5_*` (15 queries) + full texts `delvigna_yarkovsky`, `dark_comets_seligman`,
`dziadura_yarkovsky_gaia`.*

---

## Bottom line — does SEXTANT's specific claim survive as novel?

> ### **NO. Not as briefed.**

Held against CLAUDE.md priority 1 ("a new signature or a new dataset/population,
not a refinement of an existing search"), SEXTANT as specified is a refinement:

* **the dataset is not new** — `gaiafpr.sso_observation` is the Gaia SSO team's
  own active hunting ground, being mined for residual anomalies *right now*
  (arXiv:2605.22702, May 2026);
* **the observable is not new** — the AL-projected post-fit residual is their
  standard observable, and the scan-direction projection is their standard
  practice;
* **the population framing is not new** — 30,000 objects (Liberato 2024),
  54,094 (Dziadura 2025);
* **the statistical architecture is not new** — Monte-Carlo null on matched
  non-signal data plus false-discovery-rate control over the screened set is
  exactly Liberato et al. 2026;
* **the linear-trend statistic is not new** — it is a named stage of the 2026
  method, both as a detection channel and as a systematic to detrend;
* **the physical question has been asked and answered on this data** — a
  non-gravitational acceleration fit over 54,094 main-belt bodies with Gaia
  astrometry returned null;
* **the O−C-against-independent-orbits construction has been executed** — 1001
  asteroids, orbits independent of Gaia, propagated to Gaia epochs (A&A 2025).

Two things do survive, and neither is enough on its own:

1. **The reference orbit** — residual against an *independent* JPL/MPC solution
   at catalogue scale, retaining the secular component a post-fit residual
   absorbs. **Conditional** on the runner confirming Liberato's fit is Gaia-only;
   if it is not, this shrinks to nothing.
2. **The interpretation** — the technosignature reading and LOOM's
   population-structure/replication axis, which remain unoccupied (Q3).

An unoccupied *interpretation* laid over an occupied *method* on an occupied
*dataset* is not novelty under this repository's rule. It is a SETI label on
Liberato + Dziadura. Say so plainly rather than shipping it.

### The strongest single piece of prior art

**Liberato, L. + 9 co-authors, *Follow the wobble: Statistical methods to detect
astrometric binary asteroids in Gaia FPR*, arXiv:2605.22702 (submitted 21 May
2026).** Same table, same residual, same projection, same scale, with an
FPR-consistent noise model, a Monte-Carlo noise-only control (88% fewer
detections than the data), FDR-controlled selection, and an explicit
linear-trend detection stage reporting 45 trend objects. *Title and author count
returned consistently by three independent queries; full-text verification
pending the runner.*

Runner-up, and the one that kills the physics rather than the method:
**Dziadura, K., Bartczak, P. & Oszkiewicz, D., *Assessing the detection of the
Yarkovsky effect using Gaia DR3 and FPR catalogues*, A&A 693, A31 (2025),
arXiv:2411.09750** — 54,094 main-belt objects, non-gravitational acceleration
fitted with Gaia astrometry, null.

---

## What WOULD be novel — two adjacent questions that are not occupied

Offered because a clean negative is a reason to change the question, not to
publish (CLAUDE.md priority 3). Neither has been prior-art-swept to the standard
above; both are candidates for the *next* sweep, not conclusions.

### A. Which force law, not how much force — the functional-form discriminant

Every paper found in this sweep **assumes a functional form and fits its
amplitude**: Del Vigna and Dziadura fit transverse A2 (Yarkovsky-shaped);
Marsden's g(r) is assumed for outgassing; Liberato assumes a two-body wobble;
the fifth-force papers assume a Yukawa potential. **Nobody tests which law the
residual actually prefers.**

A natural non-gravitational acceleration has a mandatory, checkable dependence
on observing geometry — heliocentric distance (r⁻² for SRP, a steeper and
hysteretic law for outgassing), spin-axis and insolation geometry for Yarkovsky,
a fixed transverse orientation. A *controlled* acceleration has no such
obligation. So the discriminant is: **per object, fit competing force laws
against the residual as a function of geometry, and rank by which law wins and
by how badly all natural laws lose** — not by amplitude, which is exactly what
`docs/loom.md` §3.2/§3.3 already argued and what the whole Gaia asteroid
literature does not do.

This is not defeated by the main-belt null: a null on *A2 amplitude* says
nothing about objects whose residual carries structure that a transverse-only
model cannot represent, which is precisely the class such a fit discards.

### B. The discard pile — screening on *rejection pattern* rather than on residual

Gaia's SSO pipeline rejects observations: ~1% in DR2 (27,981), 0.58% flagged in
DR3, exposed per observation in FPR as `is_rejected`,
`astrometric_outcome_ccd`, `astrometric_outcome_transit`. The binary literature
complains that this rejection "excludes the most informative observations". So:
**which objects do the rejections concentrate on, and in what geometry?** An
object whose astrometry is systematically un-fittable is invisible to every
search that works on the *surviving* residuals, including Liberato's and
Dziadura's — because those objects fail the fit rather than producing a
suspicious residual.

Nothing found in 34 queries touches this. It is cheap (the flags are columns in
the table SEXTANT already plans to pull), it is a genuinely different population
rather than a different statistic, and its first output is a measurement, not a
model. It should be measured *before* any residual machinery is built, in the
same spirit as `loom-probe`.

---

## Files

| path | what |
|---|---|
| `docs/sextant-priorart.md` | this adjudication |
| `results/sextantlit/websearch_log.md` | all 34 in-sandbox queries, counts, blocked hosts |
| `scripts/sextantlit_fetch.py` | the runner battery: 89 arXiv queries, 22 verifications, 12 full texts, term matrix, OpenAlex citing sets, VizieR ReadMes |
| `.github/workflows/sextantlit.yml` | `workflow_dispatch` — not yet dispatched |

**Status: pass 1 complete, pass 2 pending dispatch.** The verdicts on Q1, Q3, Q4
and Q5 and the bottom line are stable against what the runner can find — they
rest on the *existence* of papers whose titles and authors multiple queries
returned consistently. The one verdict the runner can still move is **Q2's
caveat**: whether Liberato's residual is taken against a Gaia-only fit. If it is
not, opening (1) closes and SEXTANT has nothing left but the label.
