# Research scoreboard

Live per-channel state of the search. Update this file whenever a run,
vet, or triage changes the candidate picture — it is the single place a
human (or a fresh agent session) looks to know what is hot and what to do
next. Last updated: 2026-09-22.

New sections are added at the top, so the newest state is first; older
sections below are dated but not strictly ordered. This file is a *log*; for
the one-line-per-channel map of what exists, where it lives, and its current
verdict, see **[docs/channels.md](docs/channels.md)**.

### GRAVE: a pandas-3 read-only array was about to eat the first screen, 2026-09-22

S56 asks whether any horizon in Earth's sedimentary record carries a
fission-product residue that no non-negative mixture of twelve natural
reservoirs can build. The screen is per-sample, but nothing is a claim until
the **age stack** says it recurs at one stratigraphic level across independent
sections, the way the K–Pg iridium does. That makes the stack the only load
path to a detection — and it was carrying an uncorrected p.

Thirteen boundary windows were tested at once and any window with
`p_hypergeom` < 0.01 was promoted. Family-wise that is 1 − 0.99¹³ = **0.122**:
a spurious "stratigraphic cluster" somewhere in the catalogue about one run in
eight. `p_hypergeom` is now **Holm-corrected** over the windows that were
testable at all (those holding ≥ 1 sampled section — a boundary the corpus
never sampled was never a test, and padding *m* with it only costs power), and
the promotion rule reads the corrected `p_family` at a family-wise
`cluster_p` = 0.05. Net of the change the channel is **stricter** than before:
FWER 0.05, not 0.122. Both p values are reported per window; `summary.json`
carries `multiple_testing`, `n_boundaries_tested`, `cluster_p_is_family_wise`.

Both positive controls survive with margin — a correction that killed the K–Pg
iridium would be the wrong correction:

| control | sections | p_raw | p_Holm | m |
|---|---|---|---|---|
| injected six-section K–Pg fission cluster | 6 of 9 | 3.97e-4 | 5.16e-3 | 13 |
| chondritic Ir-anchored impact layer | 5 of 10 | 7.76e-4 | 7.76e-3 | 10 |

and a window that clears 0.01 raw but not the correction (2 candidate sections
of 10 sampled, against 5 candidate sections in a 300-section corpus,
p_raw = 9.5e-3 → p_Holm = 0.067 over 7 tested windows) is now held at
`multi_section_at_background_rate`. The suite asserts that case explicitly.

**The ash kill was also half-written.** The brief names both of volcanic ash's
ratios, Zr/Hf *and* Nb/Ta; only Zr/Hf was on the fission path, and it fired
solely when Zr was the driver. `tephra_signature` now tests the coherence an
ash fall actually produces — all four of Zr, Hf, Nb, Ta up together by ≥ 2×
with Zr/Hf in 25–60 *and* Nb/Ta in 5–40 — and vetoes `volcanic_ash` when the
driver is an element the tephra itself carries. Written on coherence the kill
is blind to a real fission residue: fission gives Zr with no Hf and has no
path to Ta at all. The suite also *measured* something the doc had assumed — a
plain ash bed never reaches the vet, because `rhyolite` is already one of the
twelve reservoirs and the mixture absorbs it outright (LR = 0.0).

29 offline tests pass, ruff clean.

**The run that was in flight would have produced nothing.** Under pandas 3's
copy-on-write `DataFrame.to_numpy()` returns a **read-only** view, and the
screen stage masks non-positive concentrations to NaN on the very next line:
`full[full <= 0] = np.nan` raises `ValueError: assignment destination is
read-only`. Both the full element matrix (which every kill reads) and the
design matrix (which every fit reads) were built that way, so run 35742065160
was going to die at the top of the screen — *after* paying for the whole
acquisition. The sandbox could not see it: this venv holds pandas 2.3.3, the
runner installs 3.0.6. Reproduced in a scratch pandas-3.0.6 venv (three
end-to-end tests fail), fixed with `copy=True`, and **29 tests now pass on
both pandas 2.3.3 and 3.0.6**. `read_csv(low_memory=False)` and the
python-engine `on_bad_lines="skip"` reader were checked against 3.0.6 too and
are clear; there is no `to_numeric(errors="ignore")` in the channel.

**Data state.** The SGP schema is established on the runner (runs 35738860553,
35739776468): `POST sgp-search.io/api/frontend/post-paged`, 94 field codes
accepted, **114,688 samples** behind the [0, 4000] Ma filter, pages of 5,000
not capped. Ru and Rh are *not served*, so the light peak rests on Mo, Pd and
Te. EarthChem's REST service is gone (ten-rung endpoint ladder recorded);
GEOROC/DIGIS supplies the tephra reference. **No screen has run yet**: the
full-corpus run `35742065160` was dispatched 10:41 EDT, waited 32 min in the
queue, started 11:13 EDT and is expected to fail at the screen for the reason
above (it will still commit `acquisition.json`, which is a real measurement of
what SGP served). The replacement, `35747786123`, was dispatched 11:30 EDT on
the fixed head. `results/grave/` holds `probe.json` only — **there is no
verdict about the sedimentary record yet**, and nothing in the repo should be
read as one.

Next decisive action: land 35747786123 and read `summary.json` — the funnel,
which of the twelve named vetoes fired and how often, and whether any boundary
window reaches `STRATIGRAPHIC_CLUSTER` under the corrected p.

### SLAG-WD: the natural family was too small, and the objects were the wrong objects, 2026-09-22

SLAG-WD (S51) asks, per polluted white dwarf, how badly the best *natural*
parcel reproduces its photospheric abundance vector — the calibrated misfit
list that has never been published — and then whether a process-orthogonal
element pair sits outside the natural envelope while the rest of its panel is
natural. The acquisition has been green since 14:36 EDT minus 4 (3,547 PEWDD
rows over VizieR TAP in 14.2 s, run 35739746529); what did not exist was a
verdict. Three things had to be corrected first, each found by looking at the
served data rather than at the brief.

**The objects were the wrong objects.** PEWDD is one row per star per paper,
and each paper uses its own designation: `GD 378` and `WD 1822+410` are one
star, `PG 0843+516` / `PG 0843+517` / `WD0843+516` are one star. Grouping on
the (qualifier-stripped) name gave 2,441 "objects" for 3,547 rows and silently
disabled every check that compares an object's own sources — the
multi-reference disagreement kill above all. All 3,547 rows carry coordinates,
so objects are now built by single-linkage **on the sky** within 5″: **1,576
objects**, 633 of them merging more than one designation. The name deliberately
does not link two sky positions — PEWDD has rows called `WD1202-232` 40° apart
and rows called `L745-46A` carrying Ross 640's position, and joining on the
name chained those into blobs of up to 27 rows over ten unrelated
designations.

**The natural family was 18 compiled vectors; it is now 1,227 measured
bodies.** PEWDD's own repository ships the meteorite compilations it compares
against, and the acquire stage had already fetched them. Read as log₁₀ number
ratios and de-duplicated across their three reference elements, they say the
brief's premise is wrong: **Ti/Al spans 2.95 dex across 1,096 real stones
against 0.31 dex across the compiled end-members**; Ca/Al reaches +2.97 in
pallasites (compiled maximum +0.04) because Al is a trace element in an
olivine–metal rock; Mn/Cr 3.21, Ni/Co 3.53. A Tier 2 exceedance measured
against the compiled end-members alone would have been an artefact of the
compilation. The two pairs that stay narrow, Sc/Ca and Sr/Ca, are narrow only
in coverage — 8 and 0 measured bodies carry both elements — and the record now
says so instead of trading on it. The same bodies also provide a second,
harder misfit calibration: the draw is a real meteorite, so a small posterior p
beside a large meteorite p is a statement about the star, and two small p
values are a statement about the model.

**Two of the seven literature controls cannot be candidates here.**
WD 0106−328 (served as `HE 0106-3253`) and NLTT 19868 reach only **four**
measured elements in their best published PEWDD panel, so both are
`INFORMATION_LIMITED` by construction. Two alias corrections came with that:
NLTT 19868 is not WD/PG 0843+516 (a different DA 62° away) and LHS 2534 is not
WD 1214+032 (PEWDD serves it as `WD 1212-022`).

**In flight: run 35747793625** on `claude/goap-slag`, `slag-solo.yml`,
`stage=all`. The sharded `slag.yml` needs to win a runner slot six times in
sequence and twice failed to start at all (35739746529's screen matrix, then
35745205866, which sat 52 minutes without its first job); the solo workflow
does the same work in one job on one slot, which is affordable because only
168 of 3,547 panels reach the 5-element floor and carry the calibration cost.
No `results/slag/summary.json` exists yet — the verdict line stays empty until
that run commits one.

Offline: 40 tests, green under **both** pandas 2.3.3 (sandbox) and 3.0.6 (what
the runner installs), per `docs/channel-brief.md` §0 item 5.

### SEXTANT: dispatched uncapped over all 156,823 objects, on one runner, 2026-09-22

SEXTANT asks LOOM's question — is a minor planet accelerating in a way
sunlight cannot supply — on Gaia's SSO astrometry, where the residual is
milliarcseconds rather than arcseconds. Until today only the acquisition probe
had ever run. Now the whole pipeline exists and is on a runner:
`.github/workflows/sextant.yml`, `probe` → N `fit` shards → `assess`, wired to
`python -m seti.cli sextant`.

**In flight: run 35746692260** on `claude/goap-sextant` — `gaiafpr`,
`max_objects: 0` (every numbered object in the release), the new **solo** path:
probe, fit and assess as three steps of ONE job on ONE runner, with a
230-minute in-job clock inside a 350-minute cap.

It is the third dispatch and the first that can realistically start. Run
35739803943 (8 shards, capped at 800 objects) sat queued 45 minutes at a commit
predating the independent Greenberg+2020 control and was cancelled; its
replacement 35744966028 (4 shards, uncapped) sat queued another 15 without its
single probe job starting, against an account queue of **69 waiting runs**
across 18 channels. The sharded path needs six separate scheduling events — a
probe slot, four simultaneous fit slots, an assess slot — and a run that never
starts measures nothing. The solo path needs one. It reaches roughly a quarter
of the objects; because a shard fits its controls first and then works a seeded
shuffle, that quarter is a smaller *unbiased sample of the same catalogue* with
the same complete control set, not a different sample. `solo: false` still runs
the sharded path when there is capacity to spend.

Uncapped is now the *safer* choice, not the riskier one, because of two
changes made before the dispatch. A shard works its **positive controls
first** and then the rest in a seeded shuffle, so a shard that never finishes
still has a complete control set — ascending `number_mp` would have reached the
NEAs last and left the `A2` distribution with nothing to check itself against
— and any truncation is an unbiased random subsample rather than a sample of
large main-belt bodies. And the fit stage runs on a clock *inside* the job's
cap, stopping between chunks, because a job killed by `timeout-minutes` is
cancelled and a cancelled job does not reliably upload its artifacts. A
stopped shard reports `OK_PARTIAL_BUDGET` with the chunks it did not attempt,
and `summary.json`'s `coverage` block says how much of the assigned sample was
reached, so unmeasured objects can never be read as a null.

**The probe's decisive finding, written down.** `epoch` is **TCB**, and it is
derived rather than assumed. The probe measured `epoch_utc − epoch` as
−85.564 s at MJD 56864 and −90.250 s at MJD 58868; computing TCB − UTC from
`L_B = 1.550519768e-8` plus TT − TAI plus the leap seconds in force gives
85.564 s and 90.249 s. Both ends agree to under a millisecond, and the 4.686 s
drift across the mission decomposes as 2.685 s of secular `L_B` and exactly 2 s
of leap seconds. Nothing but TCB does that. This mattered more than it sounds:
87 s of time-tag error is ~0.7 arcsec of along-track offset on *every* object
in proportion to sky rate — a catalogue-wide fake detection shaped exactly like
the signal. Also settled: the observer state vectors are equatorial
(`max|z_gaia| = 0.4098 au`, which is `y_ecl·sin 23.44°` and not an ecliptic
slab); `is_rejected` runs at 0.6304% against a published 0.58%; and the
DR3/FPR union deduplicates under **both** candidate keys.

**The controls are the falsifiable part, and there are now two of them.** The
primary is JPL's fitted `A2`, pulled live from SBDB. It has a weakness — JPL's
solutions saw Gaia DR2/DR3 astrometry at high weight — so the channel now also
scores against Greenberg+2020 (AJ 159, 92; VizieR `J/AJ/159/92`), 247 `da/dt`
measurements from optical and radar. The conversion is exact for JPL's
`g(r) = (1 au/r)²`: `da/dt = 2 A2 / (n a² (1−e²))`, which reproduces Bennu's
published −19.0 ± 0.1e−4 au/Myr from JPL's `A2 = −4.6e−14 au/day²` to 1%. If
the Gaia-only fit does not return these in sign and magnitude, nothing else in
the output is believed, and `assess` stamps `ESTIMATOR_FAILS_CONTROLS` onto the
run verdict rather than reporting the exceedances.

**What to do next:** read run 35746692260's `results/sextant/controls.json`
before anything else in `summary.json` — `verdict`, `n_measured` and
`recovered_fraction`. If it reads `CONTROLS_FAILED_SIGN` or
`CONTROLS_INCONSISTENT`, the exceedance list is a property of the estimator and
`assess` will already have stamped `ESTIMATOR_FAILS_CONTROLS` on the run
verdict; fix the fit before reading anything else. Only if the controls recover
does `coverage` (how many of the assigned objects were actually reached),
`a2_distribution` and the population verdict mean anything.

### ARC stage 2 on the pixels: 5 of 21 catalogued flares were on a NEIGHBOUR, 2026-09-22

S59 (`docs/arc.md` §9). The centroid test that decides this channel has now
run on real target pixel files with the NaN-pixel fix in place (run
**35738785437**, 21 of 30 stars before it was superseded — the per-star
checkpoint in `results/arc/stage2/stars.json` holds every one):

| | run 35675112803 | run 35738785437 |
|---|---|---|
| `flare_on_target` | 6 | **8** |
| `flare_on_neighbour` | 1 | **5** |
| `centroid_ambiguous` | 1 | 5 |
| `centroid_untestable` | **22** | **3** |

58 flares examined: 21 on target, **5 on a neighbour**, 9 ambiguous, 7 with
too few pixels, 2 unattributed, 14 untestable. The five misattributions are
KIC 10288777 (**16.8σ** from the target), KIC 7009116 (14.6σ), KIC 9268205
(7.0σ), KIC 7174965 (4.8σ) and KIC 9139163 (3.9σ), each consistent with a
named Gaia DR3 source. That is a statement about the *flare catalogues*
— their per-star attribution comes from the pipeline aperture — not about
this channel's candidates.

**The channel's two named candidates are settled, and differently.**

| star | ξ stage 1 | ξ measured | Teff, R★, M★ (Berger+2020 table2) | verdict |
|---|---|---|---|---|
| KIC 8487271 | +0.104 | **−0.920** | 5998.6 K, 1.301 R☉, 1.209 M☉ | **`flare_on_target`** |
| KIC 11507705 | +0.440 | **−1.240** | 6365.3 K, 1.311 R☉, 1.187 M☉ | **`centroid_ambiguous`** |

8487271's flare (4.57e34 erg) put the difference-image centroid **0.171 px
(1.5σ) from the target**, every neighbour rejected — so it passes the pixel
test and dissolves on the parameters anyway. 11507705's one testable flare is
consistent with the target *and* with DR3 2129762445437102464; its own
quarter amplitude is 2.7× the catalogue value, which is most of the further
0.65 dex it fell. `stellar_params_assumed` is closed on both: measured Teff /
R★ / M★ from Berger+2020 `J/AJ/159/280/table2`, 20 of 21 stars measured.

**Stage 1 re-run (run 35741271294, assess, 288 s).** A shortlisted star now
takes measured parameters whether or not it was flagged assumed. Funnel
unchanged — 190,486 flares, 8,908 stars, **4,206 assessable**, 4,702 with no
amplitude and so no ceiling, **1** above the conservative ceiling, 0/0/13
candidate/interest/watch — but the one excess grew: ξ_conservative,max
**+0.462 → +0.715**, flares above 4 → 6 (nominal 11 → 12).

**The Santos+2021 lever is measured and it is not the lever STATUS predicted.**
It was already in the sample (245 of 2,507 Yang & Liu stars, 22 of 279
Shibayama) and moved the assessable count 4,204 → 4,206. The 4,702 unassessable
are not waiting on Santos: 1,117 are Günther TESS stars with no amplitude
anywhere and 913 are Yang & Liu stars in neither McQuillan nor Santos. What
the lever did instead was **raise the ceiling** — `Sph` → range (×2√2) lifts
`E_mag` by 0.68 dex — taking conservative positives 5 → 2 → 1.

**KIC 9418692 — and the number that decides it is its amplitude.** ξ = +0.715
on 6 flares (ξ_nominal +1.431 on 12) of 14 Yang & Liu events, on Santos's
`Sph` as a range, 2.008e-4. The *same star's* Shibayama record carries
6.0e-4, and on that amplitude the identical flares give **ξ = +0.002** —
exactly at the ceiling — because `E_mag ∝ A^{3/2}` turns a factor 3 into 0.71
dex. Run **35744902798** (queued 11:05 ET) is the first to put its Yang & Liu
record on the pixels *and* measure its amplitude from its own Kepler light
curve; `load_shortlist` ranks it first of 13 as `vetoed_excess`. Its only
remaining killer, Gaia RUWE 1.556, is an astrometric suspicion, not a
detection.

**Two defects fixed, each measured on real output.** An archive fetch had no
wall clock of its own — the stage budget is checked *between* stars, so one
hung MAST request in run 35738785437 held the process from the moment its
9,000 s budget expired; `_fetch_products` now abandons a fetch after
`stage2.product_timeout_s` and records `QUERY_FAILED`, never zero rows. And a
star that leaves every tier when better parameters arrive left every
shortlist with it (KIC 8487271, +0.104 → −0.920), so `--stars` /
the workflow's `stars` input reads named rows straight from `xi_table.csv`
whatever tier they ended in.

**Next decisive action:** read run 35744902798 for KIC 9418692's quarter
`Rvar` and its centroid verdict; then the Gaia DR3 non-single-star solutions
and any archival spectroscopy for the RUWE 1.556 companion.
### RELAY measured: Earth sits in 1.8e4-1.8e7 node-to-node beams, and none of them optical, 2026-09-22

S60 (`docs/relay.md`). The channel asks a question nobody has asked at catalogue
scale: for every DIRECTED star pair inside 100 pc, does Earth fall inside the
transmitter's beam on its way to the other node? Two geometries, counted apart:
**spillover** (T behind R on the same sightline, Earth gets (|T-R|/|T|)^2 of what
R gets) and **between** (R near T's antipode, the beam crosses Earth first and
Earth gets MORE than R). The prior art is single-target only -- Tusay+2022 at
the alpha Cen SGL antipode -- and the network papers are design studies.

**Run 35740854882 (targets+geometry, 264,973 Gaia DR3 stars, counts exact over
the whole sample, 5/5 beams):**

| beam | theta | spillover | between | meas/analytic |
|---|---|---|---|---|
| 10 m optical, 1 um | 0.0252" | **0** | **0** | (3.3e-5 expected) |
| 100 m, 1.42 GHz | 8.85' | 18,381 | 145,734 | 1.26 |
| 10 m, 8 GHz | 15.7' | 54,147 | 461,277 | 1.18 |
| 10 m, 1.42 GHz | 1.48 deg | 1,586,109 | 14,613,310 | 1.09 |
| over-filled 5 deg | 5 deg | 17,841,566 | 167,627,412 | 1.07 |

Fitted slopes of log N vs log theta: **1.954** (spillover), **2.000** (between) --
the theta^2 law measured, not assumed. The sample is the clean subset of the
GCNS volume (parallax >= 10 mas, parallax/error > 10, RUWE < 1.4); all seven
parallax shells returned OK and untruncated; 43 % carry a usable RV.

**The yield is the result.** A diffraction-limited 10-m optical link leaves
Earth outside every one of the 7e10 directed pairs -- that channel is closed,
and saying so is worth more than searching it. The interceptable regime is
radio with modest apertures or deliberately over-filled beams, and there the
intercepted flux is only slightly below (spillover) or above (between) what the
intended receiver gets: a strong-signal channel, not leakage. The same numbers
are the trials budget -- 1.8e4 to 1.8e7 qualifying pairs per beam -- so every
hit count is printed beside `n_expected_by_chance` and `n_trials`, and RFI at
non-zero drift stays an unexcluded systematic on any survivor.

**What the sky did not supply, and the fix.** Run 35738937745 probed 15 seed
VizieR ids plus six keyword sweeps: 73 tables, **0** carrying both a frequency
and a drift rate. These surveys deposit their observed-star lists and keep
their event lists in the papers. So `src/seti/relay/papers.py` now reads the
events out of the arXiv e-print source (deluxetable/longtable/tabular plus AAS
machine-readable tables), with the id verified against the Atom title before
use, provenance on every row down to the header text it was parsed from, and a
table counted as a hit table only when a frequency column and a drift column
both resolve and at least one row parses as numbers. 13 offline tests, no
socket.

**In flight:** run **35745111146**, `stage=all` on `claude/goap-relay`, the
first full-scale pass with the e-print route: probe -> targets -> geometry ->
recut -> assess -> `results/relay/summary.json`. The smoke pass
(24,878-star debug cap) already committed a summary; this one replaces it at
full scale.

**Next decisive action:** read 35745111146's `hits.json` -- specifically
`arxiv.papers[]` (which papers resolved, which e-prints downloaded, which
tables carried a drift column) and the per-beam `n_hits_on_pair_line` beside
`n_expected_by_chance`. If the e-print route also comes back empty, the honest
move is not a limit paper but a change of question: the 1,959 BL targets that
ARE in the 100 pc sample have 993+ public data files, and the pair-line
pointings with a predicted drift window can be searched in the raw filterbank
products directly rather than through anyone's published hit list.

### SHROUD: the SVO VASCO service is dead, so the sample is 127 — and the USNO-B1.0 rebuild is the only way back to scale, 2026-09-22

S33 (`docs/shroud.md`; §9 is the new route ledger). SHROUD looks for POSS-I
sources absent from the modern optical but **present and warm in the infrared**
— enshrouded, not destroyed.

**What answers and what does not** (measured on the runner, every endpoint and
error verbatim in `results/shroud/acquire_verdict.json`):

| route | answer |
|---|---|
| Solano+2022 SVO `vanish-neowise` / `vanish-possi` | **dead** — TCP timeout at 25 s, http and https, 4 path spellings each |
| `svo2.cab.inta-csic.es/vocats/` | **403**; every `vanish-*` path **404**, while the host root returns 200 |
| VizieR TAP_SCHEMA keyword search | **200, and the answer is "no"** — 11 hits, 9 of them the surnames *Vasco D.* / *Vasconcelos M.J.* |
| VizieR `J/AJ/159/8` (Villarroel+2020) | **200, 127 rows** |
| VizieR `I/284/out` (USNO-B1.0) | **200, 12/12 fields, 33,273 raw rows, 9.425 deg²** |
| CDS X-Match | **200**, 15–21 s per chunk |

So the intended ~172,000-source Solano sample is **unreachable by any route**,
the verdict ceiling is `VIZIER_FALLBACK`, and the committed sample is 127
objects — three orders of magnitude short. Population fractions from it are
indicative only and `summary.json` says so.

**The 5″ excess is −0.9 σ, and that is not a null — it is the wrong question.**
The committed offset-position null measured 7 matches in 127 real sightlines
against 39 in 508 displaced ones: 9.8 expected by chance. At 5″ against AllWISE
(~1.8×10⁴ deg⁻²) the matched subsample is chance-dominated *by construction*,
so its excess is consistent with zero however real the physics. The channel now
measures the excess **as a function of radius** (1–5″): chance grows with the
search area, a genuine counterpart is already counted at the smallest radius, so
a real population shows up concentrated at small separation with a significance
that peaks near the astrometric error. Tested both ways offline — half-associated
population recovered at `f_true = 0.5` with the peak at ≤ 2″; background alone
gives |σ| < 3 at every radius.

**Two VizieR failure modes, each of which cost a run and each of which looks
exactly like an empty sky**, are now closed and documented:

1. A literal `+` in a query string decodes to a **space**, so `-c=266+65`
   arrives as the unsigned pair `266 65` and VizieR returns an empty resource.
   (This is the bug that cost IGNITION a dispatch.) The sign is percent-encoded
   on every rung of the query ladder, with a test per rung and for negative dec.
2. `-meta.all` lists a catalogue's **default output columns, not its
   dictionary**. I/284/out's defaults are the eight astrometric ones, so the
   probe declared `B1mag`/`R1mag`/`R2mag`/`Imag`/`Ndet` absent from a catalogue
   that plainly has them — and run 35738062833 let that probe *edit* the
   request. All 12 fields came back as bare positions, and "POSS-I red present,
   everything else absent" cannot be expressed by a frame with no magnitudes.
   All 12 reported `n_poss1_only = 0`; **that zero was an artefact of the
   request, not a property of the sky.** The probe now reports only, a rung's
   answer is accepted only if it carries `RAJ2000`/`DEJ2000`/`R1mag`, and the
   ladder otherwise falls through to the `-out.all` rung, which names no columns.

**A cancelled run erased a measurement, once.** `analyze` runs `if: always()`,
so it also runs when `acquire` was cancelled and no artifact exists; it then
writes `NO_DATA_REACHED` / `n_sample 0` — a statement about archive *access* —
and at 14:58 EDT-4 (2026-09-22T14:58Z) commit `05df5117` pushed that over the
127-source summary. Guarded now: an empty summary may be committed only when
`HEAD` holds no sampled one; otherwise the results are checked back out and the
empty attempt is kept beside them as `summary_attempt.json`.

**The probe ladder has no ceiling — a bound, not a diagnosis.** The SVO probe
walks (configured roots + RegTAP roots + roots scraped from an index page) × 5
URL forms, each a 25 s timeout against a host dead at the TCP level in every
run this channel has ever made. The root count is contributed by the registry
and by a page scrape, not by this channel, so the cost has no upper bound: 200
roots is seven hours. And it lands on exactly the wrong route —
`reconstruct_from_usnob1` is handed `max(deadline - elapsed, 60)`, so the only
route that can restore the channel's scale gets 60 seconds in the limit.
Bounded now: `acquire.svo_probe_budget_s` = 600 s, also capped at 25% of the
remaining deadline, reporting `budget_exhausted` with the count of roots not
tried. *Correction:* an earlier version of this section said run 35741075121
had spent two hours in that step. That was wrong — I mistook my own elapsed
working time for the run's. At the time of writing it had been in the step for
21 minutes, against 11 minutes for the same step in run 35738062833. The bound
stands on the unbounded root count, not on an overrun that was observed.

**In flight.** Run **35741075121** (dispatched 10:32 EDT, started 11:07 EDT)
is the first to carry the column fix, so it is the first that *can* return a
non-zero `n_poss1_red_only`. The decisive number to read from it is
`sky_coverage.n_poss1_red_only` against `n_usnob1_raw_rows = 33,273` per
9.425 deg²: if the POSS-I-red-only fraction is of order 10⁻³–10⁻² the rebuilt
sample reaches 10⁴–10⁵ objects at full grid and the channel has its scale back
from a source that does not depend on SVO being alive. Runs 35738062833 and
35740203590 were cancelled as superseded.

**Not yet measured:** the current summary's zeros for IR presence come from a
photometry job that never ran, not from a search that found nothing. That
distinction is now carried explicitly — `summary.json:photometry_reached`
names which catalogues answered and `degraded_reason` spells out what each
zero does and does not mean, with `REPORT.md` leading on it rather than on the
funnel. No survivor stands as of this entry.

**Runner-version gate.** Per the repo-wide pandas warning, both shroud suites
(96 tests) were run against a throwaway venv holding **pandas 3.0.6 / numpy
2.4.6 / astropy 8.0.1** — what the runner installs, not the sandbox's pandas
2.3.3 — and pass unchanged. The channel uses no removed API; `np.trapz` was
already behind a `getattr(np, "trapezoid", ...)` fallback.

### LANTERN: the reader, the phase and the verification all pass on real data, 2026-09-22

The first LANTERN run (34036760527) analysed 21 of 832 exposure checkpoints —
`read_failed 656`, and **0** exposures classed as eclipse or transit. Run
**35737559234** settled both causes on the runner, against real data.

**The reader.** Every `x1dints` product in MAST now carries the
`TSOMultiSpecModel` layout — one `EXTRACT1D` table per segment and spectral
order, one *row* per integration, 2-D spectral columns, per-row `TDB-MID`
times. The old reader took each HDU as one integration. The rewritten one read
WASP-43 b MIRI/LRS as **9 216 integrations across 30 tables** and WASP-18 b
NIRISS/SOSS as **2 720 across 6 tables and 3 orders**, both with times sourced
`row_bjd_tdb`. Re-planning the committed inventory with it drops the
double-counted level-2 segments the level-3 products already hold: **467
scheduled exposures, 509 GB** (78 eclipse-class at 123 GB, 265 transit, 124
unresolved; 632 superseded, 182 proprietary) out of the 1.12 TB first counted.

**The phase.** The labeller now places eclipses correctly on real ephemerides:
WASP-43 b **6 681 ppm at 15.4σ**, WASP-18 b **1 451 ppm at 23.1σ** with the
free step 0.007 d from the predicted ingress against a 0.020 d tolerance. (The
WASP-43 timing check failed for a reason of its own: it is a *phase curve*, and
a free step fitted over the whole visit locks onto the transit, which is
deeper. The check now runs inside ±0.25 P of one eclipse with transit
integrations excluded.)

**The finding that matters.** Both verifications still failed, on the same
thing: an injected line at **2% of the continuum produced zero features**. The
reason is physical. On a real `x1d` product the time-averaged spectrum's
residual around its local continuum sits at **~1% of the continuum however long
the exposure** — the static pixel pattern of the extraction, not photon noise
(1.3% on SOSS over 2 720 integrations, 1.1% on LRS over 9 216). A 6σ trigger
there needs a line brighter than **~8% of the continuum**; the screen would have
been worthless across 509 GB.

That pattern is identical in and out of eclipse, so the search now runs on
`1 + (⟨out⟩ − ⟨in⟩)`, where it cancels exactly and what is left is what changed
when the planet was occulted. Its control is the **drift null** — out-of-event
integrations before the event minus those after it, the same visit and the same
drift with no occultation in between — used both as the "consistent with zero"
statistic and as the veto `present_in_drift_control`. Measured on synthetic
stacks carrying the 1% pattern: a 2% vanishing line is **invisible** to the old
search and returns at **109σ** as a clean candidate in the difference; a
constant stellar line of the same amplitude cancels to nothing; a
persistence-decaying line is caught by the drift null at 15σ. The 5σ EW limit
improves **2.8×10⁻⁴ → 8.0×10⁻⁶ µm, a factor of 35**, and the trigger is now
photon limited (~1.5×10⁻³ of the continuum at 600 integrations).

Transit-class exposures — 265 of the 467 — get the analogous out-of-transit
minus in-transit difference, reaching the same depth. They can never produce a
candidate (`insufficient_phase_coverage`, and `transit_inconsistent` for a line
that changes across transit more than the continuum does), which is correct:
this channel's signature needs an eclipse.

**The verification now passes on real data.** Run **35741401724** put both
known eclipses through the difference search:

| | WASP-18 b NIRISS/SOSS | WASP-43 b MIRI/LRS |
|---|---|---|
| integrations / tables | 2 720 / 6 | 9 216 / 30 |
| eclipse depth | 1 451 ppm at 23.1σ | 4 223 ppm at 7.0σ (in-window) |
| free step vs predicted ingress | 0.0003 d (tol 0.020 d) ✓ | −0.169 d, but better by only Δχ² = 9.9 |
| 5σ EW limit, out-of-eclipse | 8.87×10⁻⁵ µm | 1.378×10⁻³ µm |
| 5σ EW limit, **difference** | **1.73×10⁻⁶ µm** (51×) | **1.93×10⁻⁴ µm** (7.1×) |
| injected vanishing line | 62.6σ `candidate`, no veto | 10.6σ `candidate`, no veto |
| its drift null / in-eclipse residual | 0.42σ / −0.13σ | 0.39σ / −0.32σ |

That run's screen was nonetheless **skipped**, because the gate demanded both
cases and WASP-43 b failed one check — the free-step timing test. That failure
was not about the sky: on a thermal phase curve the arch a linear detrend
leaves pulls a free two-level step away from the eclipse, and the step it
found improved χ² by 9.9 over the step held at the predicted ingress, which
itself beat flat by 49. Three things changed as a result. The timing check now
asks whether the data *prefer* a differently placed eclipse (free step inside
the tolerance, **or** not beating the predicted step by Δχ² > 25); a t₀ shifted
by 0.55·T₁₄ on a synthetic eclipse still fails it. The gate is the **phase**
question only — one clean known eclipse settles whether in-eclipse integrations
can be identified — with the injected-line recovery reported as a separate
`injection_verdict`. And the injection amplitude is now measured from the
exposure's own noise rather than fixed at 2% of the continuum, which on both
these exposures is *below* their 5σ EW limit and so tested nothing.

Two other things that cost the earlier runs: nothing is skipped as `too_large`
any more (the largest public product is 10.33 GB against a 12 GB cap, so no
chunking is needed), and the shard deadline is now predictive — it refuses to
*start* an exposure whose estimated cost would run past it, because a
checkpoint is only safe once the shard's artifact uploads.

**Runner-version gate.** Per the repo-wide pandas warning, all 60 LANTERN
tests were re-run in a throwaway venv holding **pandas 3.0.6 / numpy 2.4.6 /
astropy 8.0.1 / scipy 1.17.1** — what the runner installs, not the sandbox's
pandas 2.3.3 — and pass unchanged. The channel uses no removed API (no
`errors="ignore"`, no `applymap`, no `fillna(method=)`, no `inplace=`; every
frame is `.copy()`/`.reset_index()`-ed before a column is assigned, so
copy-on-write is a no-op), and every `pd.concat` filters empty frames first.
Independently, run 35741401724's probe executed the whole acquisition path —
NASA Exoplanet Archive TAP (4,738 transiting planets) and four MAST
`query_criteria` calls (4,001 timeseries observations) — on the runner under
pandas 3.0.6 and succeeded.

Run **35745941769** (4 shards, eclipse-first, `require_verify=true`,
`deadline_minutes=270`) is queued with all of this. Checkpoint version is 3 and
deliberately unchanged, so a mid-flight assess cannot mark a running screen's
checkpoints stale — which is how run 35737559234 reported 159 stale checkpoints
and zero exposures. The honest limitation stands, in the contamination ledger:
an *unresolved* emission line from the planet's own atmosphere passes every
veto this channel has, so a survivor is a target for higher-resolution
follow-up, not a detection.
### ARC closed out: 4,206 stars on the ceiling, one left standing (KIC 9418692), 2026-09-22

S59 (`docs/arc.md`, §9 carries every number). The stage-1 assess stage had
never once finished — run 35675114711 sat in it for **4 h 54 m** on a `pyvo`
async job with no time limit and was killed by the workflow cap with no
`summary.json` written at all. With a per-query clock and a stage budget it
now runs in **277 s** (run **35738218021**, `NO_CEILING_EXCESS`):

| | |
|---|---|
| flares screened / stars | 190,486 / 8,908 |
| **assessable** (has a rotational amplitude) | **4,206** |
| no amplitude, so no ceiling, so untested | 4,702 |
| `ξ_conservative > 0` | **1** |
| candidate / interest / watch | 0 / 0 / 14 |

**The Santos+2021 lever is not what STATUS expected.** It was already in the
sample (245 of 2,507 Yang & Liu stars, 22 of 279 Shibayama); it moved the
assessable count 4,204 → 4,206, not "well beyond". What it actually did was
**raise the ceiling**: rescaling `Sph` from a standard deviation to a range
(×2√2) lifts `E_mag` by 4.75 (**0.68 dex**) and took the conservative
positives 5 → 2 and the nominal positives 24 → 9.

**Stage 2 ran** (run **35675112803**, 2 h 08 m, 30 stars, 69 flares, pixel
centroids + Gaia census + Berger+2020 parameters). Both stage-1 interest
stars dissolved on measured parameters — KIC 11507705 `ξ 0.440 → −0.590`
(0.677 dex of it the `Sph` rescaling, 0.353 dex the 1.311 R☉ radius) and KIC
8487271 `ξ 0.104 → −0.920`. Neither was ever tested on the pixels. The pixel
test does bite on real data: of 69 flares, **9 on target, 1 on a neighbour**
(KIC 7009116), 2 ambiguous.

**What is left is one object, not a null.** **KIC 9418692**:
`ξ_conservative = +0.462` on **4 flares** (ξ_nominal = +1.178 on 11) of 14
Yang & Liu events; `E_flare,max = 9.78e34` vs `E_mag,cons = 3.37e34 erg`;
amplitude `2.008e-4` from Santos **with** the ×2.828 scaling already applied.
On Berger+2020 (5677 K, 1.089 R☉) instead of the Shibayama star table (5378
K, 1.300 R☉) the same flares give **ξ = +0.715** — the excess *grows* on the
better parameters. It is `first_veto = companion_suspect` on Gaia **RUWE =
1.556** alone, which put it in no tier and so outside every stage-2
shortlist. Its Gaia census: target supplies **99.73 %** of the flux inside
one Kepler pixel; the two neighbours (G = 20.4 at 3.2″, G = 19.7 at 5.2″)
would need **39 %** and **41 %** brightenings, and neither is excluded by
arithmetic. Its centroid test has never been run on its Yang & Liu flares.

Three defects found and fixed this session, each measured on real output: the
assess-stage hang; a difference image that required **every** pixel of a
cadence to be finite, which cost 11 of 30 stars their centroid test while the
same flares had 5–7 in-flare and 68–82 baseline cadences in the aperture
centroid; and a `Sph` rescaling that would have been applied **twice** now
that stage 1 applies it (ceiling ×4.75 too high — the direction that hides a
candidate). Stage 2 now shortlists hard-vetoed ceiling-excess stars *first*.

**Next decisive action:** the stage-2 pixel test on KIC 9418692's 14 Yang &
Liu flares with the Berger radius, plus a Gaia DR3 non-single-star and
archival-spectroscopy look for the companion RUWE 1.556 only suspects.

### FORGE dispatched — the detector gate is green and the measurement is in flight, 2026-09-22

S47 reads the hot-exozodi population as ~1500 K swarm candidates. The channel
existed but had never been run, and its injected-signal test was red; both are
now resolved.

**The gate.** `test_end_to_end_recovers_the_injected_swarm_and_verifies_the_asset`
passes, and not only locally: CI run **35740042459** on `181dea2f` is green on
the runner, 32/32 forge tests, ruff clean. Nothing the channel ranks would have
meant anything until the detector could recover a signal it was handed, so this
was the precondition for reading any ranking.

**What blocked the measurement was not the science.** `forge.yml` lived only on
`claude/goap-forge`; GitHub 404s a `workflow_dispatch` for a workflow absent
from the default branch, so the channel could never be run at all. Merging the
branch to `main` (`6e2bb280`, purely additive — 15 files, all forge-only plus
one `docs/channels.md` row and the fenced FORGE block in `cli.py`) registered
workflow `364339715`. First dispatch: run **35744731075**, `stage=all`,
`skip_population=true`, queued 11:04 EDT.

**The statistic.** Per star, χ² of a free grey body (T, f) against the
nano-grain emissivity family Q(λ) = min(1, (2πa/λ)^β), a ≤ 0.5 µm, β ∈ {1, 2},
over the H/K/L/N excesses plus the polarimetric null, with a per-band
cross-instrument calibration floor in quadrature and a variability term from
repeat epochs. Δχ² = χ²_nano − χ²_grey. The Planck arithmetic that defines the
observable, recomputed and confirmed against the brief: a grey 1500 K body at
1 % in K gives **6.6 % at 10.5 µm for a G2V star, 7.1 % for F5V, 8.1 % for
A0V** — and 3.0–3.3 % at L. Sub-micron grains cannot do that, which is the
whole point of the test.

**Stated plainly, before any result: nano-grain physics is expected to win for
the well-constrained systems.** The K-bright / N-faint pattern that forces
small grains is exactly what a Planck swarm cannot produce, so `nano_preferred`
is the anticipated modal outcome. The deliverable is the ranked
Planck-consistency list and any survivor of it, not a count of how many stars
behaved as the standard model says they should.

Kills carried on every candidate: faint companions at the 1 % level (closure
phases do not exclude them — κ Tuc varied and turned out to be a companion),
K-vs-N cross-instrument calibration, the cool-nano-grain degeneracy, and the
sample size (~150 stars with any interferometric excess, far fewer with an
N-band measurement). A star with no N-band measurement is `N_UNTESTED` and is
never a candidate; `NO_PLANCK_CONSISTENT_OUTLIER` is a count, not a limit.

Next decisive action: read `results/forge/` from 35744731075 — `probe.json`
first, for which of the nine VizieR ids actually resolved — then dispatch the
broadband population leg (`skip_population=false`) separately.

### CRADLE built and dispatched — the empty cell at 250–350 K, 2026-09-22

S52/S53 went from a package that had never been run to a channel with a
workflow, a doc, CLI wiring and a green offline suite. The target is one cell
that is empty in the literature: **250 ≤ T_bb ≤ 350 K** (the habitable-zone
blackbody radius) **and** log(f/f_max) > 3 (three decades above the Wyatt 2007
collisional maximum) **and** age > 1 Gyr from **two independent** indicators.
Every known extreme debris disk is young, or — in the two mature cases,
BD+20 307 and TYC 4479-3-1 — hot (~400 K).

What the offline suite proves before any archive is touched: an injected 300 K
excess at log(f/f_max) = 4.0 on a 3 Gyr star is recovered into the cell; the
same excess on a Sco–Cen star is vetoed by position and parallax; a galaxy
blend is vetoed by `ext_flag` and a Gaia beam neighbour; a star with one old
indicator is `IN_CELL_AGE_UNDETERMINED`, never a candidate; an empty archive is
`NO_DATA_REACHED`; a missing ages shard is `DEGRADED`, never a clean null; and
every one of the seventeen kill rules trips on its own case and has a counter.

Two bugs the suite found in the inherited code, both silent killers:

* `excess.harmonise` **renamed** `ks_m` → `Ksmag`, so the K_s anchor vanished
  from the shortlist contract and every star downstream came out `KS_MISSING`.
  It now adds the OSSUARY spellings and keeps the archive ones.
* `assess` only honoured `--shards` when the stage was `all`, so a sharded
  production run would have reported a clean null over a partial set of ages
  shards instead of `DEGRADED (ages_shards_missing:…)`.

Sky coverage is exact rather than sampled: `source_id` carries the level-12
NESTED HEALPix index, so 768 level-3 pixels are the whole sky as contiguous
primary-key ranges; pixel *k* goes to shard *k* mod *n*, and a unit that times
out splits into its four children.

**Run 35741356662** (`stage=all`, 8 acquire shards over the 768 units, 4 ages
shards, branch `claude/goap-cradle`) was dispatched at 10:35 a.m. EDT and is
**queued**: the account's Actions concurrency is fully occupied. Nothing has
been measured on the sky yet, and `results/cradle/` is empty — the channel's
verdict is not `NO_CRADLE_CANDIDATE`, it is *not yet run*. The first thing to
read when it lands is `probe.json`: which of the three join shapes answers,
whether the three controls resolve and come back through the join, and whether
`irs_enhv211` and each VizieR table exist. `acquire` reads the working shape
out of that artifact.

### CRADLE: the runner installs pandas 3, the sandbox has 2.3 — the probe died on its own gate, 2026-09-22

Run 35741356662 started at 10:51 a.m. EDT after 16 minutes queued. Its `probe`
job reached the runner at 11:03 a.m. and **failed at 11:05 without making one
archive call**: the offline gate step raised

```
tests/test_cradle.py::test_ipac_table_parser_reads_a_spectrum
ValueError: invalid error value specified   (pandas/core/tools/numeric.py:183)
```

`seti.cradle.mineralogy.parse_ipac_table` called `pd.to_numeric(errors="ignore")`.
That spelling was deprecated in pandas 2 and **removed in pandas 3 — it now
raises**. The sandbox venv is **pandas 2.3.3**; `pyproject.toml` asks only for
`pandas>=2.0`, so the runner's `pip install -e ".[dev]"` fetched **3.0.6**. The
offline suite was therefore green locally and red on the runner, and it is red
for the same reason in any other channel that uses a pandas API removed in 3.
The suite is now run under both: `pip install --target <dir> "pandas>=3"` and
`PYTHONPATH=<dir>:src pytest` reproduces the runner exactly without touching
the shared venv.

The parser now coerces and keeps the original strings only when a column is
not numeric at all, so a missing value becomes NaN instead of poisoning a whole
column back to text.

Two shard-economics bugs were fixed in the same pass, neither of which changes
a number but both of which decide whether the numbers are measured at all:

* **A refused join shape cost one timeout per unit, not one per shard.** The
  probe measures the shapes on one pixel; `acquire` took its answer as a fixed
  order for all 96 units. When `probe.json` never arrives — which is exactly
  what happened here, the artifact upload warned "No files were found" — the
  ladder re-pays the failing shape on every unit. The acquire loop now promotes
  the shape that actually answered and records `shapes_planned`,
  `shapes` and every `shape_relearned` event in the rollup.
* **One pathological pixel could outlast its job.** A timed-out unit splits to
  `healpix_split_max_level = 6`: unbounded that is `1 + 4 + 16 + 64 = 85`
  queries at `query_timeout_s = 1200 s`, 28 hours inside a 350-minute job, and
  every unit the shard had not yet reached would be lost. `fetch_unit` now
  carries the shard's deadline through the recursion and reports
  `deadline_exceeded`, which reaches `screen` as
  `coverage.n_units_deadline_exceeded` and `assess` as a `DEGRADED` reason.

The eight `acquire` shards of 35741356662 were queued at 11:05 a.m. EDT and run
without a probe; they self-correct their shape now instead of paying for it 96
times. `results/cradle/summary.json` is still not written — the channel's state
remains *not yet measured*, not a null.

### IGNITION: four transports refused identically, so it was never the transport, 2026-09-22

Run 35653615329 produced no shard output at all, and its two failures were
different problems that had been read as one.

**The upload ladder.** The probe walked all four rungs — pyvo's synchronous
form, a raw `POST` with the parameters in the URL (sync, then async), and
IRSA's Gator multi-object search — and three of them came back with the *same*
sentence from IRSA's own TAP: `INTERNAL_SERVER_ERROR: Unimplemented data type:
unicodeChar`. Four transports cannot fail identically on a transport fault,
and the server only gets to make that complaint after it has parsed the
request and read the upload — so those rungs were working. The refusal is
about a **column type**: `Table.from_pandas` on `source_id.astype(str)` gives a
numpy `<U19` column, astropy serialises it `datatype="unicodeChar"`, and IRSA
does not implement that type. `sid` now goes up as `long` (a Gaia `source_id`
is an integer by construction), a non-numeric id as ASCII `char`, every
remaining unicode column is converted on the way out, and if a service refuses
`long` too the ladder downgrades **once** to a 32-bit row index. None of it
can touch the science: rows are assigned to stars locally, by exact
unit-vector separation with per-star radii, never by the service's join column.

The same probe showed the hand-rolled async rung getting `200` with **no
`Location` header**, so the job URL is now also read from the job document in
the body, and pyvo's own UWS client is a fifth rung.

**The parent sample was a wall clock, not an archive.** That run's `sample`
step ran **2 h 22 min** over the same 20 one-degree cones without finishing —
run 35039105536 had pulled the identical 846-star parent in **719 s** — and
the job was cancelled with the acquire matrix never started. `sample_from_run_id`
now takes `sample.json` and `parent.parquet` from a prior run's artifact: the
`sample` job then takes **1 m 48 s** (measured, run 35738088082). The reused
`probe.json` is dropped rather than committed — a dispatch must not overwrite
the branch's live probe record with evidence it did not gather.

For the all-sky sweep the same stall is paid in *tiles never reached*, so
`fetch_parent` takes `unit_budget_s`: attempts begun after a unit has spent it
are recorded `SKIPPED_ON_UNIT_BUDGET` with the route and shape named, the unit
is a recorded `QUERY_FAILED`, and `degraded` carries
`unit_budget_skips:<n>/<units>`. The ladder's **order is untouched** and no
science cut changes — it bounds only how long one tile may be chased.

**Scale.** The pilot's 20 cones are 62.8 deg². The `tiles` sweep at 4° is
**2,047 tiles over 32,451 deg²** of the `|b| > 15°` sky — **517× the area** —
which at the measured 13.5 stars/deg² is an all-sky parent of order **4 × 10⁵**
stars. Two ways to make that bigger were rejected on the science, not the
effort: `|b| > 10°` samples stars `vet.py`'s `galactic_plane` rule exists to
kill, and `G < 15` buys stars at W1 ≈ 13 whose per-epoch scatter is several
times that of the W1 ≈ 10 stars the measured 0.1 mag/decade sensitivity was
established on. Scale here comes from **area**, not depth.

### CRYPT built: the thermal and radar axes of the lunar-PSR artifact search, 2026-09-22

S55 (`docs/crypt.md`). Every executed search for artifacts in permanently
shadowed regions is optical machine learning on NAC/ShadowCam frames (latest
arXiv:2608.09350); nobody has read the Diviner polar products or the Mini-RF
mosaics inside PSRs for a *point source*. The channel does both:

- **Thermal.** Inside the Diviner cold-trap mask (all-time bolometric maximum
  < 110 K, eroded 2 px) a pixel whose channel-6/7 brightness temperature
  exceeds the channel-9/8 reference by ≥ 5σ — σ measured from the PSR
  interior itself in 2 K bins of T_ref — in **both** the summer and the
  winter cumulative product, with the same excess radiance (f·L(T_hot),
  independent of T_cold), compact, unstriped, ≥ 10 observations, and a
  two-component spectrum that beats one temperature. A lit rim or scattered
  light is summer-only and dies as `seasonal`. The band model reproduces the
  brief's numbers (1 m² at 300 K on a 35 K pixel: channel 6 → 56.9 K,
  channel 9 → +0.009 K). The floor is not asserted from the instrument
  paper: sources of 1–10⁴ m² are injected into the *real* maps and the
  smallest area recovered in ≥ 50 % of trials is quoted.
- **Radar.** Compact (≤ 4 px) Mini-RF CPR ≥ 1 pixels in a quiet
  neighbourhood; rock fields and ejecta are extended and elevated.
- **Optical.** ShadowCam / NAC coverage of every survivor listed through the
  ODE footprint service, named as the unexcluded step.

Offline: 38 tests, no network — the injected 300 m² source is recovered in
both seasons, every rule is tripped by its own case, a scripted PDS archive
runs probe → assess end to end, and an empty archive is `NO_DATA_REACHED`.
The product naming inside `lrodlr_1002` is unknown here, so the first runner
pass is an inventory: `probe.json` commits every product name, the label
fields and the classifier's selection, and the patterns in
`config/crypt.yaml` are corrected from that evidence. Nothing is a sky
statement until `results/crypt/summary.json` lands.

### IGNITION goes from blocked to a live parent sample, 2026-09-16

The cone-spelling fix landed and the channel came apart in the right direction.
Probe run 35038504272:

| | before (34799195807) | after |
|---|---|---|
| AllWISE, 1 deg cone | 0 rows | **65,745 rows** |
| ESA `inner_cone` | timed out at 480 s, three times | **answered in 18.5 s** |
| probe elapsed | 1,503 s | 129 s |
| verdict | `NO_DATA_REACHED` | `GAIA_AND_NEOWISE_REACHABLE` |

Two things changed at once and the record separates them. The VizieR zero rows
were **ours**: a literal `+` in a query string decodes to a space, so
`-c=266+65` reached the service as the unsigned dotless pair `266 65`, which it
does not read as a sky position — it answered with an empty `#RESOURCE`. The
ESA timeouts were **theirs**: the same three query shapes that failed at 480 s
now answer in eighteen seconds, with no change on our side, so that was the
archive and not the query plan. Only the first of those was a defect to fix.

The sweep then ran: **846 parent stars** over 20 field cones, all served by the
authoritative ESA route with its own `allwise_best_neighbour` cross-match, no
mixed routes, 719 s. NEOWISE is reachable through IRSA (4,110 epochs on the
probe star, field route recommended).

Because the ESA archive was down for most of a day and will be again, a third
parent route through IRSA's own AllWISE table is being built — the ESA failure
was in the join against its 750-million-row AllWISE mirror, which no
rearrangement of the cuts avoids.

### ULINE: the census found a second U-line list and the sample grew 4.7x, 2026-09-16

Run 35039822190 is the first to search a source this channel found for itself
rather than one asserted from a paper. The column census added
`J/A+AS/142/181/table3` — Cernicharo, Guelin & Kahane's IRAM 30m lambda-2mm
survey of IRC+10216 — and it delivered:

| source | lines | unidentified | vetoed as known | clean |
|---|---|---|---|---|
| Cernicharo+2000 | 380 | 63 | 53 | 10 |
| He+2008 | 377 | 17 | 14 | 3 |

The U-line sample went from 17 to **80**, and the pairs evaluated from 6 to 12.
Verdict is still `NO_PATTERN`: no industrial species has three coincident
transitions. That is a count, not an abundance limit, and is not written up.

**The two limits are now sharper than the result.** Neither table carries an
intensity column, so the LTE consistency test is `untestable` rather than
passed — a three-frequency coincidence cannot be checked for Boltzmann
consistency, which is the test that would separate a real species from an
accident. And five of the target species — CF2Cl2, CFCl3, SO2F2, CHClF2, CF2 —
have no rotational line list in JPL or CDMS at all, so they were not searched.
Those are the most diagnostic industrial molecules there are, being both
long-lived and purely artificial; their absence from the catalogues, not from
the sky, is what excluded them.

### TRACKED DOWN: Kepler-718 b's "growing transit" is a TESS blend, 2026-09-16

Two independent tests converge and the candidate is dead. Run 35047871387 (the
reduction ensemble) and run 35047873385 (the difference image) agree, and they
agree on a mechanism rather than merely on a null.

**1. The depth "change" is the crowding correction, not the sky.** Measuring the
same TESS data under every reduction the archive serves:

| flux column | SPOC | TESS-SPOC |
|---|---|---|
| PDCSAP (corrected) | 29,255 ppm | 31,984 ppm |
| SAP (raw) | **11,196 ppm** | **11,522 ppm** |

Both pipelines agree with each other and disagree with themselves by a factor
**2.61** between raw and corrected photometry; `sap_minus_pdcsap_z = −15.6`,
verdict `BACKGROUND_TEST_DISAGREES`, direction `PDC_DEEPER_THAN_SAP`. On the
Kepler side the same test gives 13,912 against 13,893 ppm — the two agree to
0.14 % — so this is a TESS-specific correction, not a property of the star.

*(An earlier version of this entry showed a third, QLP column repeating SPOC's
numbers exactly. That was not a QLP measurement: QLP serves nothing for this
target, and the light-curve fetcher dropped its author filter when nothing
matched, so SPOC's products came back under the QLP request. Correcting it took
the members from 6 to 4 and the spread UP from 9,303 to 9,661 ppm — a phantom
duplicate had been shrinking the very error the ensemble exists to size. The
verdict is unchanged.)*

And the **raw** TESS depth is 0.80 of the Kepler depth. That is the right side of
one: a bigger aperture admits more contaminating light and must read *shallower*.
The whole 2.16 "growth" is the crowding correction dividing it back out, by a
factor the Gaia census says is far too large for a target supplying 95 % of the
aperture flux.

The verdict reflects this: the ensemble spread is 9,661 ppm — **33.0 %** — which
swamps the 1,521 ppm bootstrap error. z falls from 10.69 on the statistical
error to **2.28** on the total, and the primary verdict is
`MEASURED_DEPTH_CHANGE_UNRESOLVED`. The threshold was 24.7 % and the measured
spread landed past it, exactly as it was warned it might.

**2. The transit is not on the target.** The difference image puts the transit
source **29.25 ± 4.18 arcsec** from the out-of-transit centroid — **7.0 sigma**,
consistent across all 8 sectors, against a detectable floor of 12.55 arcsec.

**This also corrects our own census.** The census excluded all seven Gaia
neighbours within 21 arcsec and was right to; the source is at 29 arcsec, so it
was never in the search radius at all. The census answered its question
correctly and the question was too narrow. Its attribution of the offset to a
G = 20.76 source at 19.9 arcsec is inconsistent with its own arithmetic — that
source would need an 899 % eclipse — which is itself the tell that the real
source lies outside the radius.

**The mechanism, stated plainly.** A diluted eclipse from a source ~29 arcsec
away leaks into the TESS aperture; the raw depth is accordingly small and
consistent with Kepler; PDC's crowding correction then multiplies it by 2.6 and
manufactures a depth that was never there. Nothing grew.

**What this cost, and what it bought.** Four mundane explanations were closed by
measurement before this one was found: the KOI catalogue is correct, no odd-even
signature exists in either era, no neighbour within one pixel can supply the
depth, and both eras agree when fitted by the same code. The one that survived
was the one nobody had measured — and it was found only because the ensemble
asked what the same photons say under a different reduction.

### Kepler-718 b: the change is real in the data, and it is on the target, 2026-09-16

Two runs settled the two open questions. Neither result depends on a catalogue.

**Both eras, fitted by the same code** (run 35045466505). The Kepler light curve
now goes through the identical fold, masked baseline and bootstrap as the TESS
one:

| era | our fit | transits | segments |
|---|---|---|---|
| Kepler 2009-2013 | **13,912 ± 30 ppm** | 643 | 17 quarters, long cadence |
| TESS 2021-2024 | **29,255 ± 1,521 ppm** | 87 | 7 sectors, 2-min |

Ratio **2.16**, z = 10.3, primary verdict `MEASURED_DEPTH_CHANGED`. The sector
deduplication worked: seven duplicate FFI reductions were dropped and the
transit count fell from 175 to 87, exactly as intended.

**The KOI catalogue is not wrong.** Our own Kepler-era fit reproduces
`koi_depth` to 2.6 % (ratio 0.974, z = −0.52), so `KOI_DEPTH_CONFIRMED`. The
catalogue-error escape route is closed by measurement, not by assumption.

**No odd-even signature in either era**: 0.56 sigma in Kepler over 643 transits,
0.25 sigma in TESS over 87. An eclipsing binary at twice the period is excluded
in both.

**The transit cannot be on any neighbour** (run 35045467971). The Gaia census
found the target at G = 15.23 supplying 95.1 % of the aperture flux, and all
seven neighbours within one TESS pixel at G = 19.6 to 20.8 — 4.4 to 5.5
magnitudes fainter. Each supplies 0.3 to 1.0 % of the flux, so each would have
to fade by **293 % to 900 %** of its own light to make a 2.93 % aperture dip.
Every one is excluded by arithmetic, with no pixel downloaded. A source needs
to be within 3.81 magnitudes of the target to supply the depth at all, and none
is.

**What is still open, and it is not small.** The out-of-transit scatter is
414 ppm in Kepler and **67,631 ppm in TESS** — a factor of 163. At G = 15.2 that
is a faint, systematics-limited light curve, and the seven FFI reductions that
deduplication discarded had given systematically DEEPER depths than the 2-minute
ones for the same sectors (33,078 to 39,798 against 25,425 to 35,498). Two
reductions of identical pixels disagreeing by 20 to 30 % is a systematic far
larger than the 1,521 ppm quoted error, and stage 1's whole failure was a quoted
error that did not describe the real scatter. Until the depth is measured under
every available reduction and the spread reported as a systematic, the
2.16 ratio has an error bar nobody has written down.

The other open item is an unresolved companion inside Gaia's resolution, which
no census and no difference image can see.

### Tracking down Kepler-718 b: no cheap exit, and four natural mechanisms that do this, 2026-09-16

The targeted sweep `g16_transit_depth_offset` scanned 2,235 abstracts. **No
paper names Kepler-718, KOI-897, KIC 7849854, TOI-4490 or TIC 268924036 at
all**, so the object is not on record as a blend, a false positive or a revised
radius. The cheap exit did not fire. That is not evidence the object is
interesting; it is only the absence of a way out for nothing.

What the sweep did return is more useful: **a time-varying transit depth is a
documented phenomenon with at least four natural causes**, and each comes with
its own discriminator. These, not "is it a planet", are what S57 has to beat.

| mechanism | example in the hits | what it also does |
|---|---|---|
| disintegrating rocky planet with a dust tail | KIC 12557548b, K2-22b | depth varies **epoch to epoch**, transit is asymmetric with a trailing tail, depth tracks the stellar rotation period |
| oblateness + spin precession | two papers, one titled for the effect | the **duration** changes with the depth |
| ring system | TESS ring search | duration and ingress shape change |
| super-puff / inflated envelope | TOI-216 b, titled "Transit Depth Variations Reveal…" | slow, monotonic |

**Two of the four are already disfavoured by data in hand.** Our own TESS fit is
stable across sectors 41 to 82 — 2021 to 2024 — at 30,614 ppm with
chi2/dof = 1.53 and an odd-even difference of 0.20 sigma, where a disintegrating
planet varies stochastically between epochs. And stage 1's duration test puts
the observed duration ratio at 1.022 against the 1.073 expected for a larger
object at fixed impact parameter, z = −0.45, where precession and rings change
the duration along with the depth.

The two that remain open are the ones the running work is aimed at: whether the
change is real at all (both eras fitted by the same code, rather than ours
against a catalogue's), and whether the transit is even on the target star.

Nothing here is a detection, and the honest summary of the object is that it has
survived the cheap tests and not yet met the expensive ones.

### The isomer veto bug cost six unidentified lines, measured, 2026-09-16

Run 35041128720 re-ran ULINE with the atom-count matching route removed. The
prediction was that the old veto counts were too high; they were.

| source | clean before | clean after | vetoed before | vetoed after |
|---|---|---|---|---|
| Cernicharo+2000 | 10 | **16** | 53 | 47 |
| He+2008 | 3 | 3 | 14 | 14 |

**Six unidentified lines in Cernicharo+2000 were being discarded on the
strength of a molecule that was not the one named.** The clean U-line sample
across both surveys goes from 13 to 19, a 46 % increase, with no new data.

The attribution changed as well as the count. He+2008's vetoes were
`HCOOCH3 13, C2H5CN 5, CH3OH 3, HC3N 2, C4H 1`; they are now
`HCOOCH3 12, C2H5CN 5, CH3OH 3, HC3N 1`. C4H's single veto is gone entirely —
it had been justified by c-C3H, a different molecule — and one of HC3N's two
came from HCCNC or HNCCC, its isomers.

The verdict is still `NO_PATTERN`: no industrial species has three coincident
transitions among the 19 clean lines. But a null computed with a veto that
over-rejects is not the same statement as a null computed with one that does
not, and only the second was worth having.

### IGNITION's third route verifies, and confirms which half of ESA was slow, 2026-09-16

Probe run 35040375857 tested the IRSA parent route on the runner. Every name
the module asserted without being able to check it offline came back right:

* table `allwise_p3as_psd` exists at `https://irsa.ipac.caltech.edu/TAP`;
* all twelve columns resolved as asserted — `designation`, `ra`, `dec`,
  `w1mpro`, `w1sigmpro`, `w2mpro`, `w2sigmpro`, `w3mpro`, `w3sigmpro`,
  `cc_flags`, `ph_qual`, `ext_flg`;
* `gaia_only_status: OK`, four parent rows through the route, `usable: true`.

`parent_routes_tried` now reads `esa_gaia, irsa_tap, vizier_asu`, and ESA stays
the recommended route while it is healthy.

**The `gaia_only` shape answering settles the diagnosis.** Selecting from
`gaiadr3.gaia_source` alone is fast; the three shapes that timed out at 420-480
s all join `gaiadr1.allwise_original_valid`, the ~750-million-row AllWISE
mirror inside the ESA archive. It was that table, not `gaia_source` and not the
query plan, and no rearrangement of the cuts could have avoided it. The third
route exists precisely to sidestep it when the archive is slow again.

### GROWTH's first valid run: one depth-drift candidate, and why the error model forbids believing it, 2026-09-16

Run 35038510064 is the first GROWTH pass to complete: the `ps.tic_id` string
parse and the Gaia cone budget both held, the ESA archive was healthy, and all
108 targets got their neighbours (108/108 by the upload route, 747 neighbour
rows, nothing `not_checked`). Verdict `DEPTH_DRIFT_CANDIDATES`, one candidate.

**The candidate is Kepler-718 b** (KOI-897.01, KIC 7849854, TIC 268924036,
TOI 4490.01, P = 2.05234 d, CONFIRMED / TFOPWG `KP`):

| quantity | Kepler 2009-2013 | TESS 2018-2026 |
|---|---|---|
| depth | 14,281 ± 18 ppm | 34,476 ± 2,349 ppm |

That is a factor 2.41 in depth, a 1.55× growth in effective radius, at
z = 8.7. It passes the channel's own geometry discriminator: the duration
grew by 1.022 against the 1.073 expected for a larger object at fixed impact
parameter (z = −0.45), so the depth change is not a change in b. It survives
the dilution ambiguity (7 Gaia neighbours, contamination 0.050 applied,
0.075 maximum) and carries no veto.

**It is nevertheless not believable yet, and the reason is in the same file.**
Of 108 measured planets, **31 sit above 5σ** from the population median and 37
above 3σ. A sample where a third of the objects are five-sigma outliers has an
error model that is wrong, not a population of anomalies; the quoted σ is the
catalogue's formal depth error and it plainly does not describe the
pipeline-to-pipeline scatter. z = 8.7 is therefore not a 8.7σ statement about
anything. The classes bear this out: 36 DEEPER_TESS against 17 SHALLOWER_TESS,
around a population offset of +0.228 in ln depth ratio that had to be measured
and subtracted before anything could be compared at all.

The specific benign reading to close first: with 7 Gaia neighbours inside one
TESS pixel, a **nearby eclipsing binary at the same period** would put a deep
eclipse into the TESS aperture that is not the target's. Dilution makes TESS
*shallower*, so it cannot explain a deeper TESS transit — but a signal that
originates on a different star can, and the contamination term does not test
for that. That is a centroid / difference-image test, which is exactly the
stage-2 test `docs/growth.md` describes and which is **not built**.

**The structural finding is bigger than the candidate.** 108 of 9,564 KOIs
reached a TESS counterpart: 1.1 %. The bottleneck is named in the join
statement — 3,055 KOIs resolve a TIC id, but only **94 of 1,975 distinct KOI
TICs have any TOI at all**. TOI is an *alert* catalogue, not a systematic
re-measurement of Kepler's planets, so leaning on it costs 99 % of the sample
*and* imports whichever pipeline produced each alert, which is the same
heterogeneity that broke the error model. Measuring TESS depths directly from
the TESS-SPOC / QLP light curves would fix both at once.

### ARC's first complete run: 5,785 flare stars, a measured ξ distribution, no ceiling excess, 2026-09-14

Run 34798862983 is the first time S59's statistic has been **measured** rather
than reported as unreachable. ξ = log E_flare − log E_mag, where
E_mag = f·(B²/8π)·A_spot^{3/2} is the magnetic energy the star's own spot
coverage can store; ξ > 0 says a flare released more energy than its spots
could hold. Four flare catalogues acquired (Okamoto+2021, Yang & Liu 2019,
Günther+2020, Tu+2022), 5,785 stars carrying flares, **1,865 assessable** —
a star without a rotational amplitude has no spot area and therefore no
ceiling, and 3,920 stars had none.

| quantile | ξ conservative (f = 1) | ξ nominal |
|---|---|---|
| p1 | −4.25 | −3.53 |
| p50 | −2.44 | −1.72 |
| p95 | −1.30 | −0.58 |
| p99 | −0.80 | −0.09 |
| max | +0.91 | +1.63 |

**Verdict `NO_CEILING_EXCESS`.** Two stars exceeded the conservative ceiling
and the gauntlet took both: one `companion_suspect`, one `evolved`. Thirteen
stars sit in the `watch` tier. The population median flare is ~275× *below*
its own spot-energy ceiling, which is the reassuring shape — the statistic is
calibrated, not saturated, and a real excess would stand out rather than
drown.

**What limits it, and what is being done.** Three catalogues were reported as
degraded and each had a different cause, all now fixed:

* **Shibayama+2013** (1,547 superflares) and **Santos+2021** (Sph amplitudes
  for tens of thousands of Kepler stars) were never listed at all. Asked for a
  bare catalogue id, VizieR's `-meta.all` named exactly **one** table each —
  the star table, not the flares; the per-quarter Teff table, not the rotation
  table — and the rest were never scored. `asu_catalogue_tables` now unions the
  catalogue's **ReadMe** inventory whenever the metadata route names fewer than
  two tables. Santos is the single biggest lever on the 3,920 stars with no
  ceiling.
* **Davenport 2016** was mis-declared. Its columns — KIC, g−i, Mass, Prot,
  Nfl, α, β — are a per-star flare-frequency-distribution summary with **no
  per-flare energy at all**, so it could never have fed the ceiling test. It is
  now a star catalogue, contributing Prot and Mass.
* **Gaia context reached 57 % of the shortlist**, so some vetoes are unapplied
  rather than passed; that is recorded per star, never assumed benign.

This is a null and is **not** being written up. It is a reason to enlarge the
assessable sample and re-measure: Santos alone should multiply it.

### IGNITION: the constraint ladder names the defect — a plus sign that was a space, 2026-09-14

Probe run 34799195807 ran the new ASU constraint ladder against both parent
catalogues and returned the same verdict for each:

```
[0] bare      OK    rows=5      -source + -out.max
[1] columns   OK    rows=5      -out= x15
[2] +-c       ZERO  rows=0      -c=266+65
```

`-source` alone served five rows of `I/355/gaiadr3` and of `II/328/allwise`;
adding the cone — no radius, no equinox, no column cut — took both to zero, and
VizieR's reply carried an **empty** `#RESOURCE=`/`#Name:`/`#Title:`, which is
what it returns when it could not resolve the target at all.

The cause is one character. A literal `+` in a query string decodes to a
**space**, so `-c=266+65` reached the service as the unsigned, dotless pair
`266 65`, which VizieR does not read as a sky position. Constraint values are
now percent-encoded (`-c=266.000000%20%2B65.000000`), so a declination sign
survives the wire, and a new position ladder sends every candidate cone
spelling — decimal signed, decimal unsigned, split `-c.ra`/`-c.dec`, radius in
arcmin, bounding box, the old integer form — and records which one the service
honours. No spelling working is `NO_CONE_SPELLING_WORKS`, never an empty sky.

The ESA Gaia archive remains blocked independently: all three query shapes
timed out at 480 s and the async queue answered HTTP 500.

### GROWTH: the cone fallback was killed by the clock, 2026-09-14

Run 34789826297 entered the per-target Gaia cone fallback at 23:30 UTC and was
still in it three hours later, against an archive answering 500 to every other
channel that night. The workflow's 180-minute cap ended it with only
`probe.json` committed, so the `ps.tic_id` join fix has **still** not produced
a result. pyvo's `run_sync` takes no timeout, so one hung request blocks the
loop indefinitely. Two ceilings now bound it: a 120 s per-request timeout
imposed through the session, and a 3,600 s wall clock for the whole fallback
across all chunks. A target the budget does not reach is `QUERY_FAILED` and is
reported `not_checked` — the ceilings change what is *attempted*, never what
is claimed.

### ULINE reaches IRC+10216: 17 unidentified lines, no industrial pattern, 2026-09-14

The second channel to reach real data, and it took three runs plus a
four-route VizieR ladder to get there. **TAPVizieR has been answering 503 to
the runner over both http and https all night**, which is why ARC and ULINE's
first attempts read `NO_DATA_REACHED` — an infrastructure state, recorded
with every endpoint and error, never a sky statement. The shared helper now
walks TAP primary → mirror TAP hosts → the non-TAP ASU TSV interface →
astroquery, and on ULINE's third run that ladder reached the He+2008
IRC+10216 line survey.

**The measurement (`results/uline/`, verdict `NO_PATTERN`).** 377 catalogued
lines over 131.4–267.4 GHz, of which **17 are unidentified**; 14 of those 17
fall on a predicted transition of a known contaminant and are vetoed
(HCOOCH₃ 13, C₂H₅CN 5, CH₃OH 3, HC₃N 2, C₄H 1 — a line can be vetoed by more
than one), leaving **three clean unidentified features**. No industrial
species produces a ≥ 3-transition pattern among them. JPL now yields real
line lists after the catdir parser fix (403/403 lines, CDMS 1,327 → 5,690
entries); NF₃, COF₂, CH₂F₂ and CH₃Cl come from laboratory catalogues, while
CHF₃, CF₃Cl and CF₃CN still ride on predicted constants, and CF₂Cl₂, CFCl₃,
SO₂F₂, CHClF₂ and CF₂ have no line list at all and are reported unsearchable
rather than searched.

**Two limits stated rather than buried.** The He+2008 table carries no
intensity column, so the LTE consistency test could not run on this source —
recorded as `untestable`, not passed. And Orion KL (Crockett+2014, ~1,730
U-lines, the source that actually has the statistics) is still unreached
while TAPVizieR is down; it is where this channel's real sensitivity lives.

### ISOTOPE's first run: 20,502 presolar grains, no purity without a package, 2026-09-14

**The first NECROFRONTIER channel to reach real data, and it reached it by a
route the probe said did not exist.** Three probe runs found the Presolar
Grain Database's own host unreachable from the runner on both ports and the
guessed DOI wrong. The channel's fifth acquisition route — a DataCite title
search — found the release on **Zenodo (record 20317007)**, and the run
screened **20,502 SiC grains** (`results/isotope/`, run 2026-09-13 23:35 UTC).
Column roles resolved against the real headers (`PGD ID`, `PGD Type`,
`d(29Si/28Si)`, `err[d(29Si/28Si)]`, `12C/13C`, `14N/15N`, `26Al/27Al`,
`44Ti/48Ti`, `d(44Ca/40Ca)`) out of 154 columns, with nothing assumed.

**The measurement.** Grain types as the literature has them: 17,066
mainstream, 1,332 AB, 824 Y, **800 X**, 292 Z, 122 unclassified, 27 N, 25 C,
14 of the new type D. The **empirical purity frontier**, which is the number
this channel exists to produce, is set by the database's own most extreme
supernova grains: δ²⁹Si ≥ **−746.5 ± 16.0 ‰** (SiC-1996-HOP-100001) and
δ³⁰Si ≥ **−789.5 ± 8.8 ‰** (SiC-2018-HOP-001202), with the joint end-member
at (−661.5, −770.0) (SiC-2000-HOP-000025). The X package, also measured
rather than assumed: median ¹²C/¹³C 159, ¹⁴N/¹⁵N 63.9, ²⁶Al/²⁷Al 0.26,
⁴⁴Ti/⁴⁸Ti 0.0575, δ⁴⁴Ca +110 ‰.

**Verdict `NO_PURITY_CANDIDATE`: zero grains beyond that envelope, so zero
with purity and no nucleosynthetic partner.** 12,823 ordinary, 6,041
mass-dependent fractionation, 1,638 without silicon, 0 contamination
suspects. Every X grain classes `ORDINARY` with reason `inside_envelope` —
by construction, since the classes past the envelope are unreachable when
nothing crosses it. Quantum-grade ²⁸Si sits at δ ≈ −999 ‰, so what the run
establishes is that **the 210 ‰ between the natural frontier and refined
silicon is empty in every grain humanity has measured**. Per the charter that
changes the question, not the venue: the next places to ask are the graphite
and oxide releases (SiC is the only phase released so far), and the carbon
analogue, whose natural maximum this run also measured (¹²C/¹³C ≤ 21,400,
SiC-2018-HOP-001803, a type C grain).

**Three infrastructure findings from the same night.** GROWTH's first run
joined only 108 planets because `ps.tic_id` is the string `"TIC 122298563"`
and `pd.to_numeric` silently emptied the map, and its Gaia upload join hit
the anonymous ESA statement timeout (both fixed; four counted join routes and
per-chunk cones). ULINE's species matcher never fired because JPL's version
column is `2*`, which the parser's regex rejected, dropping 171 of 403 lines
(fixed: 403/403, CDMS 1,327 → 5,690 entries, NF₃ now found in a real
catalogue). **TAPVizieR is returning 503 to the runner over both http and
https**, which is why ARC and ULINE's U-line sources read `NO_DATA_REACHED` —
an infrastructure state, recorded verbatim with every endpoint and error, and
explicitly not a sky result.

### Five NECROFRONTIER channels built, 2026-09-13 (user: "Do it")

Built in parallel by five builders on disjoint paths, each offline-tested and
wired into the CLI (`seti isotope|growth|arc|ignition|uline`) and the channel
index; every one is on `main` and dispatched from it (ISOTOPE, GROWTH, ARC,
ULINE at `stage=all`; IGNITION at `stage=probe` first, as its doc advises):

* **ISOTOPE (S46, 30 tests)** — ²⁸Si / ¹²C purity beyond the Presolar Grain
  Database's own most extreme classified X grain with every measured
  nucleosynthetic partner solar. Five acquisition routes recorded per run
  (human URL, local path, wustl crawl, EarthChem Library search, DataCite DOI
  lookup by title); a stdlib xlsx reader. Expected first verdict
  `NO_DATA_REACHED` until the table is located — the wustl host did not
  answer the runner in three probe runs.
* **GROWTH (S57, 43 tests)** — Kepler-era vs TESS-era depth of every
  re-detected Kepler planet after the limb-darkening band ratio and
  Gaia-neighbour dilution, at fixed impact parameter (duration must change as
  k, not as b); population median subtracted so "growth" is relative to the
  Han+2025 deficit; the >30 d list carried for stage 2.
* **ARC (S59, 28 tests)** — ξ = log E_flare − log[B²/8π A_spot^{3/2}] at f = 1,
  B = 3 kG, spot area from the rotational amplitude marginalised over
  latitude (×3 conservative); ≥ 2 independent flares above the ceiling on a
  companion-free, unblended dwarf; the centroid test is stage 2 and no
  candidate is claimable before it.
* **IGNITION (S61, 43 tests)** — a monotonic W1/W2 rise sustained ≥ 5 yr on a
  Gaia-astrometric, kinematically old, 2010-photospheric field star; the
  scan-band sinusoid fitted jointly; a sustained-ramp family against
  step+decay at ΔBIC ≥ 6; VIGIL's field-wide NEOWISE cone reused, and its
  failure mode (an aggregator reading missing shards as a null) closed by
  `shards.expected` vs `shards.found`.
* **ULINE (S54, 27 tests)** — JPL/CDMS line lists (and a symmetric-top
  predictor with `verify`-flagged constants for CHF₃, CF₃Cl, CF₃CN, NF₃)
  matched against the Crockett+2014 Orion KL and He+2008 IRC+10216 U-line
  tables; ≥ 3 coincident features, LTE consistency, a rigid-shift false-alarm
  probability, contaminant vetoes from the vibrationally excited states of
  CH₃OH / CH₃CN / HCOOCH₃ / C₂H₅CN and the CH₃Cl / CH₃F baseline.

Their result files are read back into this log as the runs land. Not yet
built, in the brief's order: S48/S49 SPARK, S50 CENTURY, S52/S53 CRADLE,
S55 CRYPT, S60 RELAY, S63 RING, S56 GRAVE, S47 FORGE.

### The residue of a non-biological successor, 2026-09-13: NECROFRONTIER (S46–S63)

A new question from the user: *find exhaustive new ways to search for
necrosignatures — signs of non-biological intelligence having wiped out a
biological population; new techniques, new data sources, places others have
not looked.* `docs/necrofrontier.md` is the answer. The sharpening that
generates new observables: if the killer persists, the system is **not empty
after the death**, and six things follow — the successor's substrate and
tailings are *refined* (isotopically, elementally, molecularly); its activity
*relocates* out of the habitable band; the takeover is a *fast transition*
visible as a rising signature in decade archives; the *weapon* leaves scars
(a shattered planet, a sterilised star, a stripped atmosphere); a persisting
machine communicates by *beam* and outlasts a century; and the nearest
archives of a Gyr-old artifact or a prior technological extinction are the
lunar cold traps and Earth's own sedimentary column. Eighteen signatures
S46–S63 follow, each with its physics, public data source, statistic,
contamination ledger and prior-art position (taxonomy §XII–XVIII in
`docs/necrosignatures.md`).

**How the positions were established.** Twelve parallel literature agents
(a thirteenth trio was stopped at the user's request when their per-article
fetches became a flood of permission prompts — future sweeps run on the
runner only). The sandbox blocks every scholarly host, so each position rests
on search-engine records plus the full texts already cached under
`results/*lit*/`, and every arXiv id is checked by the runner sweep's
`id_title_check.json` before it is cited. Headline positions, to be confirmed
by that sweep: **no proposal anywhere of stable-isotope purity in solids as a
technosignature** (S46; nearest are Whitmire & Wright 1980, Catling+2025 D/H,
Ellery 2025 Th/Nd); **no Kepler-vs-TESS transit-depth consistency catalogue
exists** (S57; Wang & Espinoza 2024 and Zuckerman+2023 are within-mission);
**no blind DASCH DR7 fade/cessation search** (S50); **no artificial-molecule
search of any ISM line survey** (S54 — the CFC literature is all exoplanet
atmospheres, and no molecule with more than one fluorine has ever been
detected); **no thermal or radar artifact search of lunar PSRs** (S55 — every
executed search is optical-NAC machine learning, the latest arXiv:2608.09350);
**no monotonic-mid-IR-rise selection on field stars** (S61 — the YSO and AGN
communities each did half); **no star-pair beam-spillover selection at
catalogue scale** (S60 — Tusay+2022/2024 and Hort+2024 are single-target);
**no executed technosignature search on pulsar planets, brown dwarfs or FFPs**
(S63); the hot-exozodi population has **never been read as a swarm** (S47),
though the K-bright/N-faint physics already excludes a grey Planck swarm for
the well-constrained systems; polluted-WD technosignatures have one executed
search (Huang+2026, a siderophile template) and **no calibrated misfit list**
(S51). Checked and **dropped**: planet occurrence vs age (flat; Sayeed+2025,
PAST IV), 511 keV stacks (a limit channel), AMS anti-helium (unpublished), the
hypervelocity stellar-engine limit (done), D/H (proposed, not a catalogue
test).

**New data sources, verified as existing (reach to be measured by the
probe):** SPHEREx QR2 weekly spectral images since Oct 2025 with the 9-million-
source SPLICES seed list (IRSA TAP `spherex.plane`/`artifact`, cutout URIs; no
per-source spectra yet); Euclid Q1's 4,313,551 NISP spectra with a
pre-computed line-feature table (DR1-Foundation Nov 2026); DASCH DR7
(252,458,490 century light curves, documented web API + `daschlab`); the
Presolar Grain Database SiC release (20,230 grains, DOI'd download); PEWDD
(1,739 WDs, VizieR J/A+A/691/A352); the Diviner Polar Cumulative Product;
ShadowCam PDS4 (5.6 TiB Aug 2026); eROSITA DR2 (31 Jul 2026, 1.9 M sources);
SDSS DR20 (3 M MWM spectra); Gaia DR4 dated **2 Dec 2026**.

**The probe ran three times (7:28, ~7:50 and 8:01 AM ET; `results/necrofrontier/`),
each run correcting the brief: 39/50 → 50/63 → 57/67 endpoints reached
with the expected product.** What it established: SPHEREx has **1,367,432
public spectral-image products** and the SPLICES seed table holds
**9,925,660 sources** (157 columns) — S48 is runnable today; all 26 Euclid Q1
tables including the line-feature table answer (S49); DASCH DR7's API and
client answer (S50); PEWDD is at VizieR as `J/A+A/691/A352/pewdd` (S51); the
three Exoplanet Archive depth tables answer (S57); NEOWISE answers (S61); the
superflare tables with per-star spot amplitude (Okamoto+2021, Tu+2022,
Shibayama+2013) and every flare/rotation catalogue answer (S59); JPL carries
line lists for NF₃, COF₂, CH₂F₂ and CH₃Cl but not CHF₃/CH₃F/SO₂F₂/CF₃CN/CFCs
(S54 needs SPFIT predictions for those); the Diviner level-3/4 products are
the second PDS volume `lrodlr_1002` (S55). Two real blockers, stated: the
**Presolar Grain Database host does not answer the runner on either port
in three runs** (S46's data must come through the EarthChem Library or the
authors), and **cassis.sirtf.com times out**, for which IRSA's `irs_enhv211`
(16,986 IRS Enhanced spectra, 85 columns) is the replacement for S53.
TAPVizieR lesson recorded in the script: table names carry literal double
quotes, so a prefix `LIKE` matches nothing — lead with `%`.

**Built and offline-tested (13 new tests, ruff clean):**
`scripts/necrofrontier_probe.py` + `necrofrontier-probe.yml` (50 endpoints,
REST and TAP, one verdict each and a per-signature readiness map, committed to
`results/necrofrontier/`) and `scripts/necrofrontier_fetch.py` +
`necrofrontier-lit.yml` (15 query groups, 94 asserted ids title-checked, 23
title searches, 93 keyword sweeps, decoy-aware concept scan to
`results/necrofrontier_lit/`). Both dispatched from `main`; their results are
read before any position above is believed. **The first sweep run (7:28 AM ET)
was throttled by the arXiv API — HTTP 429 on 64 of 65 requests at the 3 s pace
every earlier sweep here used — and delivered one file.** The fetcher now
paces at 6 s, backs off 30/90/270 s or Retry-After on 429/503, asks for each
group's asserted ids in one `id_list` call, and falls back to the ADS API with
the repository's `ADS_TOKEN` secret (same Atom shape, source recorded per
file); re-dispatched ~8:50 AM ET. **That run was throttled too (1 of 32) and
its ADS attempts recorded `no ADS_TOKEN` — the repository's secret is unset.**
Third version (~9:35 AM ET): after the first request the API refuses three
times, the run goes keyless — arXiv abstract pages for ids, OpenAlex for
keyword and title searches (the routes lzlit / necrolit / zacklit used) —
with a `--scan-only` step so a deadline-killed run still yields the concept
scan. **Run 3 (11:05 AM ET, keyless) landed at ~12:20 PM ET: 1,713 abstracts
scanned across twelve of the fifteen groups (the deadline cut g13–g15,
re-fetched as run 4 with the new `groups` input). No decoy-free hit occupies
any new signature; the nearest neighbours the record itself supplied are the
relay-network design papers (Hippke I/II; "Engineering an Interstellar
Communications Network by Deploying Relay Probes", arXiv:2204.08296) for S60,
Huang+2026 for S51, Lacki 2025/2026 for S52, and — new to us — "The Dyson
Minds 2025 Workshop: SETI Around Black Holes" (arXiv:2604.21886) for the
compact-object branch of S63. Seven asserted arXiv ids resolved to unrelated
papers and were replaced by title searches; none is cited in the brief.
`docs/necrofrontier.md` §5 carries the per-group table. Run 4 (groups 13–15,
~12:35 PM ET, 40/67 fetches, half via OpenAlex) completed the set at 2,135
abstracts: the starspot energy bound is established in the natural literature
and never mined as an anomaly set (S59); pulsar-ring theory exists (Osmanov;
Kayali+2025) with no executed search on any post-biological host (S63); the
in-situ composition record (Stardust ISPE, Cassini CDA, the IMAP/IDEX
instrument paper) has never been read for artificial outliers (S58). **All
fifteen groups: unoccupied**, with Huang+2026 (S51) and the relay-network
design papers (S60) as the nearest executed and proposed neighbours.** Build order (charter: novelty,
then scale, never a null): S46 ISOTOPE → S51 SLAG-WD → S54 ULINE → S61
IGNITION → S48/S49 SPARK → the rest as listed in `docs/necrofrontier.md` §4.

### Grey goo as a technosignature, 2026-09-11: GOO (S44, S45) and two manuscripts

A new question from the user: *are there papers on how long it would take grey
goo to propagate through the galaxy / universe and looking for signs of it? If
not, write one, including how the findings (or lack of findings) should change
our thoughts on the likelihood of us grey-gooing ourselves; a generalised
version on the other ways we could accidentally kill ourselves with
self-replicating technology; and a target journal.* `docs/goo.md` is the
design; `seti.goo` the model (44 offline tests, ruff clean); `paper/goo/` the
main manuscript (builds, 0 undefined citations, 7 figures, 6 tables, 26 pages) and
`paper/replicators/` the companion; `paper/goo/JOURNAL.md` the venue analysis.

**The survival scan (user request ~10:40 AM ET, 2026-09-11: "model a range of
how long they last in interstellar space, from infinitely to short scales").**
`seti.goo.survival`, 8 tests, paper §5.1 + Table 2 + Fig. 3, `docs/goo.md`
§2C. It forced a correction: the passive branch had gated arrivals on the
1.2 Gyr annulus mixing time, so survival below 10⁸ yr "delivered nothing".
Wrong — the total landing rate of a cloud expanding into a uniform stellar
field, N_c η n_* v σ, is independent of the cloud's volume, so landings never
wait for mixing. Survival sets the **reach** v_∞τ and whether a **chain of
seedings** forms, not the count. Sub-micron blown-out devices (18 km/s): at
τ = 10⁴ yr they land on 10¹⁵ planets' worth of cross-section but all within
0.18 pc (no star), at 10⁵ yr the reach is 1.8 pc (0.26 planets), at 10⁶ yr
18 pc and 261 planets and **a front at 9 km/s crosses the Galaxy in
1.6 Gyr — the same time as for an immortal device**. The threshold is one
decade wide at τ ~ 10⁵–10⁶ yr (the hop to the nearest star), 10³× below the
old ≳10⁸ yr and below the 5×10⁸ yr grain lifetime. Rock-borne devices never
form a chain (R0 ≤ 2 at τ→∞ for planetesimal-borne, 10⁻⁵ for impact ejecta).
`contact.py`'s functional factor exp(−t_mix/τ) replaced by (τ/W)(1−e^{−W/τ});
abstract, §5, §7, discussion and conclusions rewritten accordingly.

**The literature, as far as the record shows.** Two literatures have never
met: grey goo (Drexler 1986; Freitas 2000 ecophagy; Phoenix & Drexler 2004
retraction; the risk surveys) stops at the planet, and self-replicating probes
(Hart 1975 → Tipler 1980 → Freitas 1980 → Newman & Sagan 1981 → Landis 1998 →
Bjørk 2007 → Cotta & Morales 2009 → Wiley 2011 → Nicholson & Forgan 2013 →
Carroll-Nellenback+2019; intergalactic: Armstrong & Sandberg 2013, Olson 2015,
Hanson+2021) assume design and control. Nearest neighbours: Sagan & Newman
1983 (probes would be "an uncontrollable danger" — a goo argument used against
Tipler), Stevens, Forgan & O'Malley-James 2016 (grey goo listed among
self-destruction modes, planetary signature sketched), Ellery 2022a,b / 2025
(designed probes as solar-system technosignatures and their curbing), Lacki
2025 (collisional cascades — the residue physics). **No paper computes the
propagation of an uncontrolled replicator beyond its planet, compiles the
constraints, or draws the implication for the civilisation asking.** The
runner sweep `goolit.yml` (28 phrase queries with decoys; per-reference
verification of all 109 citations via arXiv id title-check / title search /
Crossref) is the evidence and was dispatched at 7:48 AM ET (run 34595836444);
its `results/goolit/REPORT.md` is to be read before the novelty sentence is
believed. The first four in-sandbox literature agents were killed by an
interruption and the sandbox's proxy blocks arXiv/Crossref/ADS outright, which
is why the sweep runs on the runner (CLAUDE.md acquisition pattern).

**What the model says (`results/goo/`).** Dispersal, not replication rate,
sets reach. *Planet, in place:* the surface can only shed
4πR²σ(T⁴−T_eq⁴), so the crust is a **5 kyr** project and the bulk **1.1 Myr**
at any doubling time from 100 s to a year (the biosphere is hours, and that is
the only place the folk "days" number applies); lifting the planet costs its
binding energy — 7 d at L_⊙ (Dyson's number), 15 kyr at the 2000 K surface
ceiling, 58 Myr on absorbed sunlight. *System:* the belt is a transport-limited
branching front, **14 yr**; the Kuiper belt 1 kyr; planets 8 kyr–143 kyr at the
radiative ceiling; the converted mass is a swarm at the feedstock's orbital
temperature (169 K belt, 44 K Kuiper) — a Dyson-scale waste-heat source while
alive. **Theorem (the paper's organising result): a dead residue detectable
above an excess floor f_min lives at most P_orb / f_min** (Lacki 2025's
collisional time) — 4 kyr at 2.7 AU for 10⁻³ — so the waste-heat surveys bound
*live* goo and are blind to dead goo. *Galaxy, passive:* devices below 0.57 µm
are blown out on starlight (v_∞ 7–18 km/s); repulsion at the target star lets
only β < 1.38, the **0.21–0.57 µm** window, reach 1 AU; the annulus phase-mixes
in **1.2 Gyr**; a belt's worth of 0.5 µm devices delivers ~200 intact
arrivals/yr to every Earth-like planet in the annulus, and landings do not
wait for mixing (see the survival scan above): above a ~10⁶ yr functional
lifetime the passive branch is a chain of seedings crossing the Galaxy in
1.6 Gyr. *Galaxy, active:* 0.6 Myr (0.1c) – 6.4 Myr (0.01c) – 500 Myr
(30 km/s); cosmic reach 7×10⁸ galaxies at 0.5c, event horizon 16.6 Gly.

**Inference (`seti.goo.bayes`, with and without the anthropic shadow).** Three
data tiers: our own belt and biosphere (shadowed, 4 Gyr memory), the
5×10⁶-star waste-heat null (unshadowed, memory ≤ P_orb/f_min for dead goo),
the 10⁵-galaxy Ĝ null (unshadowed, indefinite memory for live goo). Prior-free
95 % bounds on p = p_goo·p_esc per civilisation: own galaxy naive 3/Λ, shadowed
none; **external persistent 3×10⁻⁷ at Λ = 100** — five decades below the
own-galaxy bound and immune to the shadow; external dead residue 3×10⁻³.
**Branch map:** the data touch only the escaping branch, so an elicited
planet-scale accident probability moves by ×0.990 if 1 % of goo accidents are
interstellar, ×0.90 at 10 %, ×0.50 at 50 %. A molecular replicator loose on a
planet is planet-bound; it needs space industry to be system-scale, sub-micron
size in space to be passive-interstellar, a starship to be active-interstellar.
**The sky is informative about goo that travels and silent about goo that
stays home; ours would stay home.** Corollary: a planet-bound goo filter leaves
a galaxy exactly as quiet as the one we see.

**Charter note.** This is not a null-result write-up: it is a propagation
calculation, a compiled record, and an inference whose deliverable is the
branch map. It does report what the record does *not* constrain, because that
is the question asked. Two new taxonomy entries (S44 ecophagic planet —
unsearchable, listed with the numbers; S45 feedstock-temperature swarm —
searchable, and OSSUARY's sample is where the natural background is absent).
Target journal: **IJA** for the main paper (every nearest neighbour was
published there), Acta Astronautica as alternative; **Futures** for the
companion, Risk Analysis as alternative (`paper/goo/JOURNAL.md`).

**Sweep read (run 34595836444 landed 8:07 AM ET, 121/121 fetches OK).**
Novelty: **0 of 58 arXiv abstracts across 28 phrase queries name grey/gray goo
or ecophagy**; no abstract pairs uncontrolled replication with galactic
propagation; the "self-replicating AND technosignature" hits are Ellery's
designed-probe papers. Verification: 85 of 109 references confirmed against
the arXiv/Crossref record; two placeholder author lists filled (Kowald 2015,
JBIS 68, 383; Chen, Ni & Ong 2022, EPJ Plus); Hanson+2021's third author and
three DOIs corrected; Moór+2021's title corrected to the record's; the
JBIS/QJRAS/JET papers marked *asserted* because neither API carries them.
Limitation stated in `docs/goo.md`: the novelty sweep reaches arXiv abstracts
only. Both manuscripts rebuilt with 0 undefined citations.

**The contact graph (user reframing, 2026-09-11 ~8:15 AM ET): "how long it
takes everything in the galaxy to be touched by something that has been
touched by something else."** `seti.goo.contact`, ten offline tests, paper
§7, figure 7, table 3; every rate from the record fetched by `goolit-contact.yml`
(run 34598452720, 44 references, 40 verified, number-bearing sentences
verbatim in `results/goolit_contact/REPORT.md`). Channels: interstellar
objects (n = 0.2 AU⁻³, Do+2018; capture 2 per kyr, Dehnen+2022), impact ejecta,
Earth-grazing bodies captured by binaries (10⁷–10⁹, Siraj & Loeb 2020),
free-floating planets (21 per star, Sumi+2023; 1 % of stars capture one),
interstellar dust (~10⁻⁴ m⁻² s⁻¹), blown-out belt devices, stellar flybys
(19.7 per Myr within 1 pc, Bailer-Jones+2018), supernova ejecta (⁶⁰Fe), the
birth cluster (10¹⁴–3×10¹⁶ bodies per sibling pair, Belbruno+2012). **In the
mass sense the graph is complete already: Earth has been struck ~33 times by
interstellar objects, 5×10¹⁷ interstellar grains land per year, one system's
ejecta are captured by 8×10⁶ others; every system is flyby-linked in 124 Myr.
For one system's material landing on other planets it is at threshold
(R0 ≈ 2.8 by interstellar objects, 3×10⁻⁴ by impact ejecta, 10¹⁹ by dust) —
contact is never the bottleneck for a goo among small bodies (size sets what it
reaches, survival how fast) and never the route for a planet-bound one.** The birth-cluster channel
gives ~5×10⁴ strict touches per sibling pair (Earth has very likely been struck
by material that condensed around another star) but no civilisation exists
during it. Two order-of-magnitude scan variables remain (rocks ejected from a
system over the age of the Earth; the Ulysses dust flux), marked in
`config/goo.yaml`.

**Full text, not abstracts (user directive ~8:40 AM ET; run 34599794718
landed ~8:55).** `goolit-fulltext.yml` fetched the full text of 91 of the 145
references (arXiv e-print LaTeX de-TeXed, arXiv PDFs, Unpaywall OA PDFs, the
web-only sources) and extracted the sentences each paper contributes against
per-paper query terms (`results/goolit_fulltext/`). Corrections forced: the
FHI 2008 nanotech-accident extinction median is **0.5 %** (the paper had
0.05 % from memory); Chen, Ni & Ong 2022 find predator probes *cannot*
sufficiently reduce prey probes (the paper had cited them for the opposite);
Freitas 2000's 100 MJ/kg, 4 °C, 20 months and 100 s are now quoted verbatim;
the ejecta count is Siraj & Loeb's 6×10¹⁰ (R0 strict 2×10⁻⁵, was 3×10⁻⁴); the
dust flux is Krüger+2019's Cassini 1.5×10⁻⁴ m⁻² s⁻¹; the birth-cluster channel
is a 10⁷–3×10¹⁶ range by mechanism with Adams & Spergel's 10⁻⁴ landing
fraction (R0 strict ~10⁷); Suazo+2022's and Zackrisson+2015's actual limits
are in Table 2. Second pass (Europe PMC, explicit OA URLs, landing pages)
landed ~9:20 AM ET: **94 of 145 in full**; added Ćirković+2010,
Snyder-Beattie+2019, Bar-On+2018 (550 Gt C confirmed), Wallner+2016 (⁶⁰Fe
1.7–3.2 and 6.5–8.7 Myr confirmed), Millett & Snyder-Beattie 2017, Noble+2018,
Esvelt+2014, Kriegman+2021, Staniford+2002. The rest are paywalled with no
open copy or are books.

**Open items.** (1) Paywalled full texts without an OA copy (Melosh 2003,
Grün 1993, Jones 1996, Burns 1979, Wallner 2016, Levison 2010) are cited on
their abstracts/records only, and the paper says which numbers rest on them.
(2) A second adversarial read of both manuscripts before submission.

### Roman is coming: the intake and four Roman-only signatures, 2026-09-09: ROMAN (S40–S43)

A new question from the user: *can we prepare to take in Roman Space
Telescope data when it is ready and put it through all the paces, including
channels other telescopes could not give us?* Roman is pre-launch (readiness
committed for no later than May 2027; no proprietary period; archive at
IRSA with cloud copies). `docs/roman.md` is the design; `seti.roman` is the
implementation; nothing in it is a measurement of Roman data yet.

**Intake (`roman.archive`, `roman.products`, `roman.schema`).** A probe that
asks IRSA TAP/SIA, the public S3 buckets (OpenUniverse 2024 simulations; the
expected flight bucket, whose existence is the test), MAST, the IPAC
simulation pages and the Python packages, and reduces what it sees to one
`data_state` (`NO_ARCHIVE_REACHED` / `NOT_YET_PUBLIC` / `SIMULATIONS_ONLY` /
`MISSION_DATA_PRESENT`) with the evidence listed. Readers that emit four
structures — `LightCurve`, `Spectrum`, `DQCutout`, `Ramp` — so no detector
ever sees a Roman file: lazy ASDF reads of the `dq` plane alone, Level 1
ramp boxes with resultant mid-times from the read pattern, table adapters
with runtime column-role resolution. A field a product does not carry is
`None` and the corresponding test is recorded as *not run*, never passed.
Every summary carries `simulated_inputs`; a run on simulations can only say
`SIMULATION_PACES_OK` / `SIMULATION_PACES_FAILED`, never a candidate tier
(enforced in `assess` and asserted by the workflow).

**Four signatures no earlier facility could reach** (taxonomy §X):

* **S40 the opaque lens (`roman.lens`).** A microlens opaque over a fraction
  ρ_L of its Einstein radius removes the minor image in the wings
  (u > 1/ρ_L − ρ_L): a *symmetric pair of downward steps of depth exactly
  A₋(u_c)*, one parameter fixing both where and how deep; a central hole for
  ρ_L ≥ 1. With θ_E from finite-source effects (or the lower bound from the
  peak magnification) the implied density is computed over every lens
  distance, and the window where a natural body could match is confined to
  tens of parsecs, where the lens is a moving blend Roman sees directly. Needs
  the GBTDS 12-minute cadence.
* **S41 the industrial line (`roman.lines`).** An unresolved line on a stellar
  point source in the 1.00–1.93 µm grism / 0.75–1.80 µm prism — the band every
  optical laser search stops short of and the band our own high-power lasers
  occupy (Nd:YAG 1.064, Yb 1.03–1.09, Er-fibre 1.53–1.57 µm). Blind over the
  band; industrial matches are flags. The slitless ledger: zeroth orders,
  trace overlap, persistence, recurrent wavelength and pixel, stellar lines,
  the single-line high-z emitter.
* **S42 the sub-exposure flash (`roman.flash`).** A pulse shorter than one
  resultant is a step in the ramp, exactly like a cosmic ray, and the
  pipeline flags it `JUMP_DET`; the difference is a PSF-shaped cluster centred
  on a catalogued star, then on the Level 1 ramp the step at one resultant in
  every PSF pixel with no slope change (a flare keeps rising) and no residual
  in the next exposure. Optical SETI's pulsed-beacon question on ~10⁸ stars at
  once, for the first time.
* **S43 the statite (`roman.statite`).** A CGI reflected-light point source
  whose position is fixed rather than Keplerian; grey, specular. Tiny
  population, screened for what no planet-hunter looks for.

**The paces (`roman.bridge`).** Dips, glint, secular fade, RUST, KNELL,
METRONOME and the narrow-line finders on Roman light curves and spectra with
GBTDS-season parameters and the F146/F087 achromaticity test.

**Operations.** `roman.yml`: monthly `readiness` cron (probe + inventory +
diff → `results/roman/readiness.json`); `stage=full` on dispatch runs
ingest → sharded screen → assess with the no-disguised-null and
no-simulated-candidate assertions; `roman-lit.yml` is the prior-art sweep.
`alerts.py` raises a milestone the first time the state reads
`MISSION_DATA_PRESENT` and a candidate alert only on flight data.

**First readiness run (34302574303, 10:16 PM ET): `SIMULATIONS_ONLY`.** All
twelve endpoints answered. No Roman table at IRSA TAP/SIA; `data/Roman/`
404; the expected flight bucket does not exist; MAST CAOM 400 (reason now
captured); the two IPAC simulation URLs assumed from memory were 404 and are
replaced by a crawl of the site root. The OpenUniverse 2024 bucket is real
and richer than assumed: SNANA HEAD/PHOT light curves (7,471 SNe per pair,
noiseless model magnitudes in 14 bands, ~295 epochs per band, 5-day cadence),
galsim TDS/WAS images with SCI/ERR/DQ planes and SIP WCS, per-image truth
indices with star pixel positions, the 57,365-exposure pointing sequence and
the point-source catalogue (`docs/roman.md` §1.4). The first inventory
classified none of it (flat 5,000-key crawl of one directory; `+` in a key
un-encoded) — rebuilt as a tree listing with OpenUniverse patterns, plus
`seti.roman.openuniverse` adapters, so the paces and S42's `DQCutout` path
run on real simulated products before launch. Selftest on the runner:
PASS 12/12.

**First full run through the screens (34349717932, 8:12 AM ET):
`SIMULATION_PACES_OK`, 5,378 simulated objects** (2,688 SN Ia light curves
in the eight Roman bands, two images, two pointing sequences; an earlier
run had shipped the manifest without the object store and screened nothing).
Two lessons the simulations taught, both fixed and tested (`docs/roman.md`
§4A): a supernova's asymmetric rise/decline lets the *symmetric* occulting
model out-fit Paczyński — 1,262 of 2,688 came out in the occulting tier —
so the S40 screen now folds every curve about t₀ and rejects time-asymmetric
events as `NOT_LENSING` before any occultation gate is believed; and at a
5-day cadence every transient satisfies the glint definition (1,595 flagged),
so glint is `not_applicable` above a 0.5-day cadence (the SNANA curves
turned out to be sampled daily, from an idealised SIMLIB) and transient-like
curves are tagged and counted apart. Confirmed in run 4 (9:39 AM ET): the
occulting tier fell from 1,262 to 145 (all pending on an edge-of-coverage
symmetry test), 2,511 supernovae `NOT_LENSING`, 1,205 tagged transient. The simulated DQ planes carry no flags, so
S42's jump path waits for flight data; MAST now answers cleanly (no Roman
collection); the crawl found the Roman Microlensing Data Challenge 2026
page, and the next readiness run (10:09 AM ET) read it: simulated Galactic
Bulge light curves in W149 (F146) and Z087 (F087) with known parameters,
false positives, parallax and astrometric series — the volume and type of
the real GBTDS — served from Hugging Face (`RGES-PIT/Beginner`,
`RGES-PIT/Experienced`, one Parquet per tier). That is the dataset S40 is
calibrated on before launch, and run 7 (11:29 AM ET) ingested its Beginner
tier with no unreadable product: 200 curves, 133 F146 + 67 F087 colour
companions. On genuine simulated microlensing S40 behaves as designed — 75
`LENSING_NO_OCCULTATION`, 56 `NOT_LENSING`, 2 occulting-pending to read
against the challenge truth — and every pace ran at the GBTDS-like cadence.
The glint pace flagged 69 microlensing peaks (a lensing peak *is* a brief
achromatic brightening), so the glint flag now also requires the event
duration to be shorter than a glint: run 8 (12:17 PM ET) halved the flags
to 36, and the rest are the challenge's hours-long free-floating-planet
events — the irreducible S30/microlensing confusion in the bulge, which
`assess` now separates by joining each star's S40 lensing fit to its glint
flag (`docs/roman.md` §4A).

**Prior art read (34302576338, 645 abstracts).** The occulting-lens
light-curve morphology is natural-body literature — Agol 2002, a 2003
astrometric finite-opaque-lens paper, and arXiv:2608.24009 (Aug 2026),
which warns that survey anomaly cuts reject such curves, so S40 must run on
the full GBTDS light-curve product and not on the event catalogue; the
density inference and the search are ours. No NIR slitless laser search, no
jump-flag flash detector and no non-Keplerian-astrometry test appear; the
Roman SETI record is RoSETZ (a transit survey) and Vides+2019 (WFIRST
coronagraph as a 575 nm laser detector). `docs/roman.md` §4.

### The LZ 248 keV recoil at the kinematic edge, 2026-09-07: LZEDGE

A new question from the user, outside the technosignature channels: LUX-ZEPLIN
(arXiv:2609.02823, 1 Sep) reports one event consistent with a 248 ± 23 ± 23 keV
nuclear recoil in 2.84 t yr (WS2024, 27 Mar 2023 – 1 Apr 2024, 4.71 t; recorded
21:22:39 UTC 16 June 2023), 2.6σ global, in a window extended to 5.4–270 keV
for EFT and inelastic spectra. *Is there an analysis worth a paper, and is
there public data?* The record was fetched over three runner sweeps
(`results/lzlit/`: the paper's LaTeX source with supplement and tables, the
19 papers INSPIRE lists as citing it, the halo-kinematics literature). The
paper's "Data Release" is not yet public (HEPData 155182, linked from
lz.lbl.gov, is the 2025 4.2 t yr record; HEPData 403s the runner).

**Novelty map (`docs/lzedge.md` §4).** All 19 follow-ups read an endothermic
recoil with the splitting near the kinematic edge. The event *date* enters one
likelihood (McCabe 2609.04181, uniform live time, v_esc fixed at 544, "how
deviations from it would change these results… future work"); Di Mauro
2609.02608 declines to use it; the Higgsino-sideband paper 2609.04175 scans
v_esc upward (544/567/610) and adds an LMC boosted Gaussian for the Higgsino
only; nobody integrates over the *measured* escape speeds, which are mostly
below 544: 484.6 +17.8/−7.4 (Necib & Lin 2022), 497 ± 8 (Koppelman & Helmi
2021), 521 +46/−30 (Roche 2024), 528 +24/−25 (Deason 2019), 580 ± 63 (Monari
2018) — all verified from the fetched texts.

**Built (`seti.lzedge`, 27 offline tests, CI green) and run
(`results/lzedge/`).** The same three-factor likelihood for every halo model —
energy on the date of the event (resolution and the ±23 keV scale systematic
marginalised), the date against the live time (R(t_obs)/⟨R⟩_live), and LZ's
empty 350–600 keV sideband as a Poisson factor — with the cross section
marginalised under a 10⁻³⁷ cm² ceiling; halo models: sharp / soft /
power-law tails, SHM++, v_esc 500/544/600, the 2609.04175 LMC admixture
(0.26 %, 0.6 %), and the six published escape-speed measurements integrated
over their uncertainties. What came out:

* **The sideband, not the edge, sets the splitting.** Under LZ's own halo the
  1 TeV posterior peaks at δ = 350 keV (1σ 330–360), 30 keV below McCabe's
  370–400: for δ ≳ 330 the spectral peak μδ/m_N enters the empty sideband
  (0.23 sideband events per window event at 350, 2.4 at 370, 10 at 380; our
  6.1 at 377 vs 04175's 4.9). With the escape speed integrated over each
  measurement the *mode* is 330–340 keV for all six; the measurement governs the
  95 % upper bound: 339 keV (Koppelman-Helmi, Necib-Lin), 362 (Deason), 373
  (Roche), 380 (RAVE/LZ), 394 (Monari). A model needing δ ≳ 370 keV at 1 TeV
  (the thermal Higgsino via Z exchange) is outside the 95 % region for the
  two precise Gaia measurements.
* **The date factor** is 2.3 at the conventional halo's 1 TeV posterior
  maximum and 4.3 at its 500 GeV one (1.0 for elastic scattering), and 2.7–4.8 at δ = 370 keV across plausible
  live-time masks (uniform 3.4). The reachability calendar: at 1 TeV a 248 keV
  recoil is possible all year for δ ≤ 360, 30 Jan–1 Oct for 370, 22 Mar–12 Aug
  for 380, 21 Apr–12 Jul for 385, never for ≥ 390 keV (v_esc 544).
* **The event cannot measure the escape speed.** Its own likelihood of v_esc is
  flat to ±5 % over 440–640 km/s and the six measurement-conditioned
  evidences agree within 2 %; a larger v_esc is absorbed into a larger δ. The
  halo evidences differ by < 2× (LMC admixtures 1.0/0.97, SHM++ 0.68, SHM 0.65,
  v_esc 500/600 0.64); the LMC admixture opens a distinct region (200 GeV,
  δ ≈ 410 keV, nothing in the sideband, date factor 6) that needs
  4.5 × 10⁻³⁸ cm² and vanishes under a 10⁻³⁹ ceiling.
* **Predictions.** One event per 2.84 t yr → 4.5 window events in LZ's 1000
  live days with 1.8–2.7 in the sideband; 90 % of recoils on the peak date in
  158–376 keV; the window rate above a tenth of its peak from 26 Jan to 8 Oct
  (v_esc 544) or 2 Mar to 29 Aug (v_esc 500); LZ's Dec–May first run kept
  84–97 % of the annual-mean sensitivity per unit exposure at these splittings.

Manuscript draft in `paper/lzedge/` (numbers generated into `numbers.tex`,
figures in `results/lzedge/figures/`). Honest framing: nothing here is a
detection claim; every statement is conditional on the event being an
endothermic recoil, and the one astrophysical null — the event does not
measure v_esc — is reported as such, with the splitting's dependence on the
adopted measurement as the deliverable.

### The warden's fence, 2026-09-06: BAFFLE (S38)

A new question from the user: *if the zoo hypothesis is right, the wardens
may be preparing to shield our own future Dyson swarm and other
technosignatures from other stars — how would they do it, how would we see
it, and is it there?* `docs/baffle.md` works the first two through. The
enclosing shell is excluded by the CMB dipole; a grey directional screen at
< 3 000 AU is a tens-of-kJy Planck source on a parallax ellipse and is
excluded by the compact-source catalogues; active camouflage and source-side
beaming leave nothing here. What survives as *searchable* is the cheapest
design — a passive, band-selective screen on the Sun→observer line — and it
is searchable because of a theorem: Lorentz reciprocity. A screen that stops
our 10 µm reaching star X stops X's 10 µm reaching us. So the observable is a
star **fainter than its photosphere in W1 and W2** with normal J/H/K_s (no
dust can do this: extinction is transparent in the infrared, not opaque),
inside a sharp-edged patch of radius R/d that moves on an annual parallax
ellipse; or, for a radio screen, a hole in LoTSS source counts at a nearby
star. Signature S38 in the taxonomy; S39 (the grey screen's own waste heat)
listed as unsearchable at catalogue level with the numbers that say why.

Built in parallel and offline-tested: `seti.baffle` (`acquire` — Gaia DR3 ×
2MASS × AllWISE through the PM-propagated archive cross-match with in-archive
pre-selection and a 0.5 % locus subsample, under the CENOTAPH acquisition
discipline; `locus` — empirical K_s − W_b versus J − K_s per luminosity
class; `screen` — two-band deficit ≥ 0.3 mag at ≥ 5σ and thirteen named
vetoes with counters, the LPV epoch-mismatch residue *deferred* rather than
dropped; `patch` — top-hat coherence, the (d, R) scan with the parallax
phase predicted by the ecliptic geometry, the constant-deficit test on the
star's own NEOWISE series; `radio` — Poisson voids in LoTSS DR2 around every
Gaia star within 50 pc, with four control positions per star as the
empirical null), plus the `missing` track (bright 2MASS stars with no AllWISE
source at all) and the prior-art sweep `bafflelit`. ETZ (|β| < 0.264°) and
nearby (ϖ > 20 mas) stars are flagged with their own denominators. Every row
in `docs/channels.md` says **Pending** until the runs land; what they say is
recorded below when they do.

**Prior art, read (run 34047603440, 13:07 ET, 116 fetches, 0 failures,
223 abstracts scanned).** Group 1 (mid-IR *deficit* searches): ten regex
hits, none a stellar catalogue search — transition-disk cavities ("infrared
deficit" at 10 µm inside a disk), solar umbral physics, T-dwarf methane,
white-dwarf collision-induced absorption, and one black-hole-binary SED
paper. **No published search selects stars for flux below the photosphere.**
Group 2 (zoo-hypothesis tests): Forgan 2011 (1105.2497) and 2017
(1608.08770) verified by id and are sociological/geometric; "A direct
communication proposal to test the Zoo Hypothesis" (1509.03652) proposes
*messaging*, not an observation; "The Recursive Panopticon Hypothesis" and
"Some Thoughts on the Future of Technosignature Searches" (the phrase
"deliberately concealed" fires there) are the nearest neighbours and neither
names a catalogue observable. The arXiv id `2401.01532` assumed for Crawford
& Schulze-Makuch 2024 resolves to "Generating New Spacetimes through Zermelo
Navigation" — **wrong id, must not be cited**; the title search found no
arXiv entry, so that paper is cited by journal reference only. Group 3
(occulters): Kipping & Teachey 2016 (1603.08928) is the cloaking precedent
(family C, from the inside); the Planet Nine searches in WISE/NEOWISE
(Meisner 2017/2018), AKARI and IRAS+AKARI are the parallax/motion baseline
for S39 — a grey 1-AU screen at 1 000 AU is a 130 Jy 100-µm source moving
400″ a year, four orders of magnitude above what those searches could see,
which is the exclusion `docs/baffle.md` §1(B1) states. Group 4 (radio
voids): every hit is ARCADE-2 background or radio-shadow jargon from other
fields; **no search for source-count holes at stars.** Group 5
(concealment): Lacki 2019 "Sunscreen" (1807.00077, galaxies partially
cloaked in Dyson spheres) and Kipping & Teachey are the only concealment
observables in the record, both for the *concealer's own* light. The novelty
position of §5 in `docs/baffle.md` stands as written.

**BRIGHT TIER, two runs (34048837928 at 13:30 ET → `NO_DATA_REACHED`;
34050635320 at 14:05 ET, 24 min → `NO_BRIGHT_MIDIR_DEFICIT_SURVIVOR`).** The
first run reached Gaia and 2MASS and nothing else: TAPVizieR serves
TAP_SCHEMA column names already double-quoted, the SELECT re-quoted them
into an empty identifier and every AKARI, IRAS and Hipparcos slab died on
`Encountered ""`; and the CDS X-Match route names its flags `Qfl/Rfl/X`,
which the quality gate read as missing and rejected every 2MASS row. Both
fixed (quote once, absent flag = unknown), both pinned by tests. The second
run is the real one: 36,735 targets (36,663 Gaia G < 7.5 + 72 Hipparcos-only
supplements, so the brightest 300 *are* covered), 30,742 with a 2MASS
anchor (29,904 of them in the Read-1 regime), 34,721 with AKARI, 22,290 with
IRAS, **29,080 with both an anchor and mid-IR photometry** — 132 of the 169
ETZ targets and 1,996 of the 2,495 stars within 50 pc. The empirical
K_s − [9] locus has a scatter of 0.09 mag on 28,247 stars (0.17 at 18 µm,
0.14 at 12 µm, 0.20 at 25 µm). The tails behave as the physics says they
must: at +5σ the excess side holds 777 (9 µm), 687 (18), 606 (12), 698 (25)
stars — the debris-disk / AGB / Be population — and the deficit side 83, 11,
3 and 36 *before* quality cuts; after them only **three** stars sit below
−5σ at 9 µm, all single-band (no 18/12/25 µm measurement), all blue
(BP−RP ≈ −0.1), all with AKARI 9-µm fluxes of 0.04–0.19 Jy, i.e. within a
factor of a few of the 50 mJy limit, where the catalogue's own flux bias is
lopsided. Two stars cross the 0.7-mag primary threshold; both are
`single_band_only`. **Zero candidates**, in the ETZ, nearby, or anywhere.
Injected-deficit completeness (two bands, two instruments): 0.5 % at
0.5 mag, 19 % at 0.7, 54 % at 1.0, 93 % at 2.0 — the Read-1 K_s errors
(0.2–0.3 mag) set this, and a near-opaque screen is the ≥ 1 mag case. The
fully-opaque limit: 13 stars whose predicted 9-µm flux exceeds ten times
the AKARI limit have **no AKARI and no IRAS source**. One is Antares
(K_s = −4.1; saturated and handled by neither catalogue as a point source).
The other twelve are K_s 4.9–5.2 stars, eleven at δ < 0° and six within 5°
of the ecliptic — the geometry of the AKARI/IRC Moon-avoidance gaps and the
IRAS unobserved strips, not of a screened population; one (Gaia DR3
6243032008973309440, 38 pc, |β| = 0.2°, an ETZ star) is worth the ten-second
check that decides it: WISE W3/W4 are unsaturated at K_s ≈ 5, so an AllWISE
cone at those twelve positions either shows a normal 12/22 µm photosphere
(coverage gap, closed) or does not. That check is queued as the bright vet.

**Bright vet, run 34052513594 (14:41 ET, 87 s): `VET_COMPLETE`, nothing
escalated.** AllWISE (VizieR `II/328`, 100 cones, 0 failures) at the 12
no-detection stars, the 27 IRAS-upper-limit stars and 60 photospheric
controls (K_s 4.5–5.5, |resid₉| < 0.05) drawn in the same run; the control
locus is K_s − W3 = −0.045 + 0.118 (J − K_s) with 0.024 mag scatter and
K_s − W4 = 0.019 + 0.120 (J − K_s) with 0.021. Of the twelve K_s ≈ 5 stars
with no AKARI/IRAS source, **nine have a normal 12 and 22 µm photosphere**
(`PHOTOSPHERE_PRESENT_AT_12_22_UM`) — the absence was survey coverage, as
the ecliptic/southern geometry said — and three are `INCONCLUSIVE` by the
letter of a 0.024-mag locus: the ETZ star Gaia DR3 6243032008973309440 sits
at K_s − W3 = −0.16 (−5.1σ) with W4 normal (−0.05) under `cc_flags = HHH0`
(halo of a bright neighbour in W1–W3), a second at −0.11/−0.08 under `0h00`,
a third at +0.09/+0.18. A screen opaque at 12 µm would take ≥ 1 mag, and a
3–50 µm screen cannot leave W4 untouched, so these are halo-flagged
photometry at the tenth-of-a-magnitude level, not candidates. The 27
IRAS-limit stars are `PHOTOSPHERE_PRESENT` (11), `INCONCLUSIVE` because W3
is saturated brighter than 3.8 mag (14), and one `NO_WISE_SOURCE`: a
K_s = 2.3 star whose nearest AllWISE entry is 20″ away flagged `ddHd` — the
diffraction-spike fragments AllWISE makes of very bright stars. Antares was
not queried. **The bright tier is closed: 29,080 of the sky's brightest
stars carry a photosphere, not a shadow, at 9–25 µm, and the 13 that
carried no mid-IR catalogue entry carry one in WISE.**

**MAIN FUNNEL, first real run (34048834972 dispatched 13:30 ET; acquisition
complete at 13:59 ET; the screen job's patch stage was cancelled at 15:04 ET
after 64 min and the screen re-reduced from the uploaded shards as run
34053752510, 6 min).** Acquisition: **complete by construction** — 36
declination bands, 36 leaves, every leaf `QUERY_OK`, 417,589 deficit-track
rows returned against a COUNT(*) of 417,589 (completeness 1.000), 602,383
missing-track rows against 602,383; six shards of ~25 min each, no truncation,
no split needed. The archive-side pre-selection (K_s − W1 or K_s − W2 below
−0.15) plus the 0.5 % random_index subsample means the parent population
screened is the ~30 million Gaia G < 15 stars with a 2MASS and an AllWISE
counterpart, of which 152,042 are the locus subsample (86,475 locus-grade
after quality cuts: 63,149 dwarfs, 22,492 giants, 834 blue). Denominators:
2,051 ETZ stars and 276 stars within 50 pc in the screened rows.

The tails: W1 at +5σ holds 13 stars, at −5σ **none**; at ±3σ 199 vs 29. W2
is symmetric at 5σ (30 vs 29) and 310 vs 189 at 3σ — the W2 negative tail
is populated, which is the CO-bandhead/giant-locus width and the crowding of
the W2 PSF, and is why the two-band requirement matters. Two-band deficits
≥ 0.3 mag at ≥ 5σ: 2,055. Vetoes (first-veto counts): `w2_only_single_band`
39,493, `w1_only_methane_like` 1,324, `poor_tmass_phot_qual` 971,
`wise_artifact` 578, `extended` 177, `gaia_variable` 56, `saturated` 47,
`poor_wise_phot_qual` 41, `wise_variable` 21, `crowded_match` 17,
`multi_peak` 10, `lpv_colour` 2 (deferred). **135 survive**, 0 in the ETZ,
0 within 50 pc. Sensitivity (20,000 injections per depth, same vetoes): 35 %
at 0.3 mag, 91 % at 0.5, 92 % at 1.0.

Reading the 135 before believing any: 107 lie at |b| < 10° and 76 at
|b| < 5°; their distances run 0.3–50 kpc (median 2 kpc); **no two lie within
10′ of each other** (no patch coherence anywhere); and the typical survivor
has W1 ≈ W2 ≈ W3 — a flat, photospheric WISE SED — sitting ~1 mag below
K_s (e.g. Gaia DR3 512945405846907904: K_s 8.64, W1 9.77, W2 9.79, W3 9.73,
b = 1.8°). A screen dims the star; it does not hand its counterpart a
different, fainter, self-consistent photosphere. That is the signature of a
deblended fragment or a wrong counterpart in a crowded field, and the one
veto built for it, `blend_flux_theft`, never ran (`neighbours_not_checked =
2055`: no neighbour table was supplied on the runner). The W3 residual was
also not computed (the W3 locus fitted zero stars: `w3snr` came back empty
from the Gaia mirror). Both are the vet stage now being built: Gaia
neighbours within 10″, AllWISE `nb/na/w?sat` (the deblending record the
Gaia mirror lacks), independent CatWISE2020 and unWISE W1/W2 photometry
against the same locus, and the W3 consistency test; survivors of that go
to the patch geometry.

The `missing` track is a catalogue measurement, not a candidate list:
602,383 of 7,311,754 bright 2MASS stars (8.2 %) have no entry in Gaia's
AllWISE neighbourhood table — 14 % at |b| < 10° falling to 3 % at
|b| > 20°, and 39 % at G 4–5 — i.e. the cross-match's behaviour on
saturated and crowded sources. The 74,154 that pass the |b| > 10°,
K_s 5–11, AAA, non-variable cuts (353 ETZ, 993 within 50 pc) are being
re-tested by direct positional match against AllWISE, CatWISE and unWISE in
the same vet, which measures the *real* absence rate.

**VET, run 34054735689 (15:23 ET; 25 min, 8 upload queries, 0 failures):
`NO_MIDIR_DEFICIT_SURVIVOR | TRULY_MISSING_COUNTERPARTS_PENDING (n=2)`.**
The W3 locus now fits (34,988 stars; +5σ tail 400, −5σ tail 7). Of the 137
deficit rows (135 + 2 deferred): 19 `DEBLENDED_COMPONENT` (AllWISE `nb > 1`
or active deblend — the Gaia mirror never carried that record), 15
`ALLWISE_PHOTOMETRY_WRONG` (CatWISE2020 puts them on the photosphere), 1
`W3_INCONSISTENT` (W3 excess on a W1/W2-deficit star), **0 survive**, and
102 `INCONCLUSIVE` — for a reason the table makes plain: every unWISE
residual sits at +2.35 (W1) / +2.97 (W2) mag, exactly the AB→Vega offset
applied the wrong way, so unWISE voted "excess" against CatWISE's "deficit"
on every star and the independent class became `ambiguous`. A bug, being
fixed with a self-check (the median unWISE − CatWISE offset must be ~0 or
unWISE does not vote). What CatWISE itself says of the 102 is the real
content: median residuals −0.34 (W1) and −0.39 (W2), 57 with both below
−0.3 mag at high significance — the same frames as AllWISE, so the same
answer, and the survivors' SED is grey (W1 ≈ W2 ≈ W3 on a photosphere that
is ~0.4 mag fainter than K_s says). Physically that is either the screen or
**a 2MASS blend that WISE resolves**: 2MASS sums a ~3″ pair into one K_s,
AllWISE's PSF fit gives the star its own flux, and the star then looks
K_s-bright, W-faint, grey. 26 of the 102 already carry `crowded_field`; 76
of the 135 lie at |b| < 5°. The deciding tests are being added: the 2MASS
PSC blend and contamination flags (`Bflg`, `Cflg`, `prox`), Gaia
neighbours at 2MASS resolution (4″, any magnitude), an **independent
higher-resolution K_s** from UKIDSS GPS / VVV / VHS / LAS (if it is fainter
than 2MASS by the deficit, 2MASS was contaminated; if it agrees, the deficit
is in the star), and a no-network G − K_s versus BP − RP consistency
residual for every screened star (a contaminated K_s makes G − K_s too red
by the same amount as the W deficit; a screen leaves it photospheric).

The missing track, re-tested by direct position on 2,341 stars (988 within
50 pc, 353 ETZ, 1,000 random controls): 2,218 have an AllWISE source within
6″, 113 within 6–15″ (the astrometric offsets of saturated bright stars),
**10 have none within 15″ — and every one of the 10 has its nearest AllWISE
entry 20–60″ away carrying diffraction-spike / halo flags** (`DdDD`,
`HHHH`, …). Eight are present in CatWISE or unWISE; the two absent from all
three (Gaia DR3 2858629802997575936, K_s 8.0, an M dwarf at 32 pc with
155 mas/yr — itself at the W1 saturation limit; and 3209766056875171712,
K_s 9.0, 250 pc) sit beside `DdDD` / `DDdD` fragments, the signature of a
bright star's artefact region where all three catalogues suppress
detections, or of a saturated primary whose own entry was lost. The
control absence rate is 0.4 % before that check; the "missing" fraction of
8 % in the cross-match table was the cross-match, not the sky. A
bright-star-within-3′ check closes the two in the next run.

**VET v2 + PATCH, run 34057027633 (16:07 ET, 3 h 55 min — 18 min of vet
uploads, 3 h 20 min of NEOWISE fetches for the patch stage): `MIDIR_DEFICIT_
CANDIDATES_SURVIVE_VET (n=0, no_hires_ks=9) | NO_TRULY_MISSING_COUNTERPART`.**
The unWISE zero point was the AB→Vega offset applied to fluxes that are
already Vega (`unwise_e_w1 ≈ 0.0005` in the table proved the flux branch ran);
fixed, with a naming-proof self-check (median unWISE − CatWISE per band must
be ~0 or unWISE does not vote; it now is). The new no-network G − K_s test
did the most work: **61 of the 135 first-pass candidates fail
`ks_too_bright_for_g`** at the screen — their K_s is too bright for their G
by the same amount their W1/W2 are too faint, which is a contaminated 2MASS
K_s, not a shadow — leaving 74 for the vet: 5 `BLEND` (a Gaia source within
4″), 9 `DEBLENDED_COMPONENT` (AllWISE `nb > 1`), 18 `ALLWISE_PHOTOMETRY_
WRONG` (CatWISE photospheric), 2 `KS_CONTAMINATED` (UKIDSS GPS / VHS K_s
fainter than 2MASS by the deficit — the deciding test, where it exists), 31
`INCONCLUSIVE` (CatWISE neither confirms nor clears), 9 `SURVIVES_VET_NO_
HIRES_KS` (CatWISE confirms the deficit, no higher-resolution K_s covers
them: VVV had no X-Match route configured, LAS returned nothing, GPS and VHS
answered for 35). Missing track: of the 10 stars without an AllWISE source
within 15″, 8 are `ARTEFACT_REGION_OR_SATURATED` (6 beside a W1 < 6 star
within 3′, 3 at their own saturation limit) and 2 are present in CatWISE or
unWISE — **no truly missing counterpart**; that track is closed.

The 9 went to the patch geometry, and the geometry closed them. Read with
`patches.csv` beside `vetted_candidates.csv`: **five have NEOWISE light
curves with reduced χ² of 731, 2 758, 873, 7 589 and 27 675 over 21 visits**
(own offsets from the AllWISE epoch down to −0.64 mag) — strongly variable
in the mid-IR, which no passive screen is; two sit in the Magellanic Clouds
(l 303°, b −44.6°, 14 Gaia sources within 10″; l 279°, b −36°, 525
neighbours within 10′) where NEOWISE and 2MASS are confusion-limited, and the
one `MODULATED` verdict there (5.7σ, null p 5 × 10⁻⁴) is
`modulation_degenerate` with the geometry-agnostic odd/even control at
5.2σ — the scan-direction systematic, not the parallax phase; Gaia DR3
512945405846907904 carries resid_gks = +0.91 at 14.7σ against a W deficit of
−1.20, the contaminated-K_s signature by 0.04 mag outside the veto's window;
and the single `ISOLATED_DEFICIT` with a constant own series (Gaia DR3
228086063620072832, 7.4 kpc, b = −6.2°) is a threshold-edge object
(−0.300 / −0.315 against a floor of 0.30) whose G − K_s residual (+0.19,
3.2σ, summing with resid_w1 to −0.11) is contamination-consistent. These
rules — NEOWISE constancy, confusion, the G − K_s window, the threshold edge
— are being folded into a post-patch `final_verdict` in the assess stage so
the pipeline says this itself. **Zero survivors of the mid-IR funnel across
~30 million Gaia stars with 2MASS and AllWISE counterparts, complete to
0.5-mag two-band deficits at 91 %.**

**FINAL ASSESS, run 34068258637 (19:56 ET, 1 min): the pipeline now says it
itself — `NO_MIDIR_DEFICIT_SURVIVOR | NO_TRULY_MISSING_COUNTERPART`.** The
post-patch rules on the nine: 5 `NEOWISE_VARIABLE`, 1 `MODULATION_DEGENERATE`,
3 `CONFUSION_LIMITED` (first-fired); as flags, `CONFUSION_LIMITED` fires on
all nine — every one sits in a plane or Cloud field with 590–744 Gaia sources
within 10′, which is stated so the rule is not read as tuned to the two
Magellanic stars — and `KS_CONTAMINATION_CONSISTENT` on four. 0 `SURVIVES_
ALL`, 0 `SURVIVES_ALL_NO_HIRES_KS`. `results/baffle/final_candidates.csv`
carries every rule per star.

**RADIO, three runs.** Run 34048836401 (13:30 ET) ground 5.5 h to its ceiling
with every LoTSS tile failing — TAPVizieR serves column names already
double-quoted and the tile query quoted them again into an empty identifier,
then burned three retries per tile against connect timeouts; run 34066251264
(19:13 ET, 7 min) had the fix but matched the catalogue hint `659/A1` to 42
unrelated A&A 659 tables and picked a star list (`DEGRADED_SOURCE`, 10
rows); run **34066710402 (19:22 ET, 50 min)** read `J/A+A/659/A1/catalog`:
568 tiles of 4°, 250 empty (outside the DR2 sky), 1 connect timeout (masked,
5 targets and 22 controls with it), **4,305,976 LoTSS sources** — the whole
DR2 catalogue. Targets: 7,404 Gaia stars within 50 pc in the two region
boxes, **4,951 in the footprint** by the annulus-density test (none of the
8 ETZ stars is). Statistic: Poisson void probability in apertures 30″–600″
at the annual set of baffle centres for d = 500–10⁴ AU (123 trials per
star, Bonferroni-corrected), against the local density from an 8′–20′
annulus; four control positions 45′ away per star as the empirical null.
**7 stars pass p < 10⁻⁵; the control false-void rate is 0.00141
(28 of 19,829 control positions), which predicts 7.0.** All seven are
10′-radius under-densities at the 500 AU grid extreme (the aperture centre
displaced 166″–385″ from the star), in mosaic regions whose local density is
300–570 deg⁻², a third to two thirds of LoTSS's nominal depth; two are the
same void seen from both members of a 0.1′ binary. The bright-source veto
was disabled because the flux column (`SpeakTot`, served) was not in the
alias table, and `MaskFract` / `Mosaic` were not read. The next run carries
the flux alias, a mosaic-edge/mask veto, a pair dedupe, and an
excess-over-control test as the verdict (`RADIO_VOIDS_AT_CONTROL_RATE`
unless the excess is significant). The reading already stands: **the radio
sky at 144 MHz has holes at nearby stars exactly as often as it has holes
anywhere else.** Reach, honestly: at LoTSS depth a 1 AU screen is detectable
only inside ~730 AU after the trials factor; this is a near-zone test on
27 % of the northern sky.

**RADIO, fourth run (34069675444, 20:23 ET, 63 min) and fifth (34074142157,
21:47 ET, 4 min, resumed from the fourth's tile checkpoints): `RADIO_VOIDS_
AT_CONTROL_RATE (n_obs=6, n_exp=4.50, p_excess=0.297)`.** The fourth run
carried the flux column and the mosaic record and its two new vetoes
removed every target (49 % of in-footprint stars have a > 1 Jy LoTSS source
within 30′; two overlapping mosaics within 20′ is the norm; MaskFract's
median is 0.47) — thresholds set before seeing the data, recalibrated from
it: 5 Jy, three-plus mosaics, MaskFract off. The fifth run: 4,951 in
footprint, 247 masked by a > 5 Jy source, 404 at a mosaic edge, 3,873 unique
positions evaluated (427 binary duplicates merged), **6 unique voids at
p < 10⁻⁵ against 4.50 expected from 15 firings at 12,907 control
positions, p_excess = 0.30**; within 25 pc, 1 against 0.30; no ETZ star
within 50 pc lies in the DR2 footprint. **The radio funnel is closed at the
control rate.** BAFFLE is complete as built: mid-IR (all-sky, ~30 M stars),
bright tier (29,080 stars at 9–25 µm), missing-counterpart limit, and radio
(4,951 nearby stars at 144 MHz) — zero survivors in each, with the
sensitivities stated in `docs/baffle.md` §8. Per the charter this changes
the question, not the venue: the passive band-selective screen is not in
use toward anything reachable; what a warden could still be using is
active camouflage, source-side beaming, or a grey screen at ~10⁴ AU at the
CMB temperature — the last a time-domain sub-millimetre question, not a
catalogue one.

### Three new questions, 2026-09-06: METRONOME, LANTERN, FALLOUT

The charter ranks novelty first, and after 35 channels the taxonomy in
`docs/necrosignatures.md` still carried two signatures nobody here or, as far
as the record shows, anywhere had built: **S28, structure in timing series**,
and **S14, the fission-product pattern** (marked "covered by midden", which
was wrong — MIDDEN searches the *lines* of Tc/Pm/actinides, not the stable
multi-element residue). A third question fell out of asking what the JWST
archive can test that no laser search ever has. All three were built in
parallel, offline-tested (121 new tests, full suite green), wired into the
CLI (`seti metronome|lantern|fallout`), the channel index and the taxonomy,
and are dispatched on the runner from `main`. None has a committed result
yet; every row in `docs/channels.md` says **Pending** and means it.

**METRONOME — clocks in flare timing (`docs/metronome.md`).** The public
Kepler and TESS flare catalogues (Yang & Liu 2019: 162,262 flares on 3,420
stars; Pietras+2022, Günther+2020, Davenport 2016, Tu+2020) are event lists
with peak times. Natural flares are stochastic; the only natural
quasi-periodicity is rotational modulation of visibility, with jitter of a
tenth of a cycle or more. A pulsed transmitter, a duty-cycled engine or a
beacon is a clock with jitter far below that. The observable is the *timing
regularity* of catalogued events — a point-process search that a periodogram
cannot see (a 1% duty-cycle clock contributes no Fourier power to a light
curve). Per star: an H-test/Rayleigh scan over trial periods, clock quality
Q and jitter, a null that resamples the star's **own observing windows** (so
the 30-min cadence, quarter gaps and sector boundaries are in the null by
construction) and a second null that shuffles the empirical waiting times.
Named vetoes with counters: `rotation_alias` (P_rot and harmonics from
McQuillan+2014 / Santos+2021), `cadence_alias` (the spacecraft's own periods),
`cross_star_coincidence` (momentum dumps — epochs shared by many stars are
removed before the scan), `periodic_variable` (VSX / Gaia vari / ZTF: an RR
Lyrae whose cycles a flare finder chopped up is recovered as a perfect clock
and then rejected, by test), `jitter_too_large`. Measured offline: a strict
clock in Kepler Q2–Q8 windows recovered to < 0.1% in P with p < 10⁻⁶; 30
Poisson stars → 0 candidates; a rotationally modulated star is coherent but
fails on jitter. Floor: N ≥ 8 events, P ≥ 0.2 d, P ≤ span/3.

**LANTERN — the line that vanishes at eclipse (`docs/lantern.md`).** Across
every public JWST exoplanet time series (NIRSpec, NIRCam, NIRISS, MIRI
`x1dints` from MAST, ephemerides from the Exoplanet Archive): is there an
unresolved emission feature present out of secondary eclipse and **absent
while the planet is behind its star**? Stellar lines, detector features and
artefacts do not care whether the planet is occulted, so a line whose flux
tracks the planet's visibility is planet-side, and a monochromatic one is a
laser or a beacon. During transit the same line must be constant in-vs-out —
the second phase reference. The one-planet, out-of-transit-only scan in
`jwst_bio` is the in-house precedent; Kipping & Teachey 2016 proposed lasers
during transit; every executed optical-SETI line search is single-epoch and
phase-agnostic. Building the detector found five real defects offline, each
now pinned by a test (a running-median continuum that failed on steep
continua; a line-series error taken from side-window *spread* that was
stellar structure and suppressed every SNR tenfold; a continuum series that
shared noise with the subtraction window; a quadratic that extrapolated
across a wide continuum hole; a clip that ran away with unmasked noise).
Final synthetic performance: a 2% line → `candidate` at 185σ with the
in-eclipse residual at 1.7σ; a constant stellar line, a cosmic-ray spike, a
14-pixel band, a null and a ramp-with-late-eclipse are all rejected with
named vetoes. Data-path risks, stated: products are 0.1–1+ GB each (streamed
one at a time, deleted, `max_file_gb` skips and records `too_large`); a 30k-
integration SOSS stack may reach several GB in memory; proprietary products
are counted, not fetched.

**FALLOUT — the fission residue (`docs/fallout.md`).** After every decay has
run, fission leaves a two-humped element pattern: a light peak (Zr, Mo, Ru),
a heavy peak (Ba, La, Ce, Nd), a valley at Ag–Te, almost nothing at Eu, no
Pb. From ENDF/B-VII.1 chain yields at a 1 Myr horizon (Zr 36.7, Mo 24.6,
Nd 20.7, Ru 17.5, Ba 13.0, Ce 12.1, La 6.4, Eu 0.58 per 100 fissions) folded
against Asplund+2021 solar, the vector is [Nd/Ba] ≫ 0 with [Eu/Nd] < 0 and
[Mo/Zr] > 0 — the s-process gives [Nd/Ba] < 0 and the r-process [Eu/Nd] > 0,
so neither, nor their mixture, reproduces it. The screen fits each GALAH DR4
cool dwarf (log g > 4, 4500–6300 K, where diffusion cannot make heavy-element
peculiarity) as solar + fission against solar + best s/r/s+r mixture, in a
Teff/logg/[Fe/H]-matched peer frame (TAILINGS' machinery, reused), and it is
only a pattern if **removing any one element leaves it standing** — the
leave-one-out test that separates this from TAILINGS' one-element anomaly.
Vetoes: `s_process_star`, `r_process_star`, `young_ba_enhancement` (La/Ce
anchor the heavy peak; Ba is reported but not load-bearing),
`nlte_saturated_lines`, `low_snr_or_flagged`, `single_element_driver`.
Sensitivity, measured offline on a 240,000-star synthetic dwarf population
with GALAH's per-element errors: complete for ≳ 1 dex Nd enrichment with the
full shape, half-complete near +0.7 dex, blind below ~0.4 dex; the
shuffled-element null puts the 99.9th percentile of the likelihood ratio at
2.3 against a floor of 8, and exactly one of 240,000 clean dwarfs crossed it
before being removed by the leave-one-out veto. The honest scale of the
proposal is in the doc: doubling a G dwarf's Nd needs ~6 × 10²¹ kg of fission
product, ~10¹¹ times humanity's spent-fuel inventory — a civilization that ran
fission at planetary throughput for a geological age, or a star into which a
whole system's waste was concentrated. The data route is the one that
actually worked for TAILINGS (Data Central cloud FITS), with the same
runtime schema discovery and `DEGRADED_SOURCE` reporting.

**FIRST REAL RUN, FALLOUT, 2026-09-06 09:42 ET (run 34036759185, 7 min).**
GALAH DR4 answered on the first route (395,752 rows, all 30 elements): 101,928
cool dwarfs and 78,344 giants screened over 11–12 n-capture elements. Verdict
`DEGRADED_SOURCE (no rv/fiber columns); FISSION_PATTERN_CANDIDATES_PENDING_VET`:
**0 dwarf survivors**, **2 giant survivors** at lower weight. Reading the run
found four things the synthetic population could not, each now a fix in flight:

1. **GALAH's quoted errors understate the real scatter.** The shuffled-element
   null put its 99.9th percentile at ln LR 17.0 (dwarfs) and 9.7 (giants)
   against 2.3 offline, so the threshold self-raised as designed — but at the
   cost of sensitivity. Fix: per-element error floors at the measured peer
   scatter (Nd 0.16, Ce 0.28, La 0.22 dex in dwarfs), then recalibrate.
2. **Both giant survivors fit fission badly and win only because the natural
   templates fit worse**: χ² of the fission model 229.7 on 9 elements and 52.5
   on 8. Both are carried by La (+1.17 and +1.04 dex against a template
   prediction of +0.3). Fix: an absolute goodness-of-fit gate
   (`UNEXPLAINED_BY_ALL_TEMPLATES`, never a candidate), a heavy-peak coherence
   veto (two of La/Ce/Nd must each carry the sign), and a La-vs-C/N/Teff
   regression among giants, because the La II lines sit in CN-blended regions
   of cool-giant spectra.
3. **78% of the dwarfs are `INSUFFICIENT`** (n-capture lines too weak to be
   unflagged), so the injected-sensitivity table was capped at ~20% and read as
   blind at 1 dex. Fix: sensitivity conditioned on testable stars, with the
   testable fraction reported beside it.
4. Two thirds of the above-threshold dwarfs fell to `low_snr_or_flagged` or
   `single_element_driver` — the leave-one-out test is doing what it was built
   for.

Both giants are recorded as *pending vet*, not as candidates, and the honest
reading is a La measurement systematic until the vet says otherwise.

**SECOND RUN, FALLOUT v2, 10:23 ET (run 34038831684): calibrated, and the
verdict is `NO_FISSION_PATTERN` with a sensitivity that says where the
question can and cannot be asked.** Per-element error floors at the measured
peer scatter (inflation 0.7–2.7× the quoted GALAH errors; Ba 2.7×, Nd 2.4×,
Y 2.4× in giants) bring the shuffled-element null's 99.9th percentile to
ln LR 4.1 / 3.9, so the configured floor of 8 now governs. Above it: 10
dwarfs and 21 giants; **0 survivors** in either sample after the vet stage;
29 stars `UNEXPLAINED_BY_ALL_TEMPLATES` (reduced χ² > 3 under every
template, fission included) are listed for a human, not counted as anything.
La carries no C/N/Teff correlation in either sample (|ρ| < 0.1), so the
first run's two La-driven giants were bad fits, not CN blends; both now fall
to `unexplained` + `single_element_driver`.

The sensitivity, conditioned on testable stars (≥ 5 elements and ≥ 2 of
La/Ce/Nd unflagged):

| sample | testable | LR+LOO completeness at Nd +0.6 / +0.78 / +1.04 / +1.32 dex |
|---|---|---|
| GALAH dwarfs (101,928) | **20%** | 0.0% / 0.1% / 1.5% / 4.3% |
| GALAH giants (78,344) | 98% | 7.9% / 37% / 68% / 80% |

So the cool-dwarf sample — the one where diffusion cannot manufacture a
heavy-element peculiarity — **cannot test the hypothesis at any enrichment
GALAH can measure**; the giants can, above roughly +0.8 dex in Nd, and show
nothing. Per the charter this changes the question rather than closing it:
the elements that *decide* fission against the s- and r-process — Pb (the
s-process makes it, fission never), Ag and Pd (the fission valley), Eu — are
not in GALAH at all. They are in the high-resolution literature
compilations (JINAbase, ~1,900 metal-poor stars; Hypatia, thousands of FGK
stars), with upper limits that enter a censored likelihood as evidence. A
`hires` stage on those is being built now. APOGEE (Ce and Nd only) cannot
separate the three processes and is not dispatched.

**Prior-art sweeps, read.** `metronomelit`: 478 verbatim abstracts across 38
queries, one decoy-free hit — eRO-QPE2, quasi-periodic X-ray eruptions from a
galactic nucleus — which is not a flare-timing search on stars. `lanternlit`:
79 abstracts plus the Kipping & Teachey citation tree, **zero** hits tying an
artificial narrow line to the planet's eclipse phase. `falloutlit`: 74 fetches,
12 of 18 id-title checks verified, and the decoy-aware concept scan over 198
abstracts finds **zero** decoy-free hits (the five regex hits are r-process
fission cycling in neutron-star mergers); Whitmire & Wright 1980 (Icarus 42, 149) has
25 citing works on Crossref and no executed search among them. Each sweep is
evidence of absence from arXiv, not proof; the docs keep the "to be verified"
wording until the concept scans are complete.

**Novelty is stated as unverified in all three docs** until each channel's
prior-art sweep (`metronomelit`, `lanternlit`, `falloutlit` — verbatim arXiv
abstracts, id-vs-title checks, decoy-aware concept scans) has run on the
runner and been read. **Next actions:** (1) read the three lit sweeps and
narrow any claim they touch; (2) read the first `summary.json` of each; a
`NO_DATA_REACHED` or `DEGRADED_SOURCE` is a data-path fault to fix, not a sky
statement; (3) anything at `interest` or above goes to light-curve / spectrum
inspection before it is called anything.

### Rubin is snowed in: SKY_STOPPED, settled 2026-08-25

Both Rubin channels have reported the same newest epoch, MJD 61235.419 =
2026-07-14T10:03Z, on **every run since 2026-07-30** — 62 sightings,
`n_advances = 0`, now 26 days frozen and 42 days behind the wall clock. The
frontier alerts have been firing correctly since 2026-08-07. The channels are
healthy: they run, reach ALeRCE, get a valid answer and commit; `tocsin` reports
`NO_NEW_DATA` because its watermark has caught the broker's frontier.

What the alerts cannot say is **which** thing stopped, and the two possibilities
have opposite consequences — a stalled ALeRCE mirror means real sky is going
unscreened and the broker path must change; a stopped alert stream means the
nulls are honest and no code change recovers anything. One broker cannot tell
them apart: "my newest LSST row is 2026-07-14" is consistent with both.

`scripts/rubin_outage_check.py` + `rubin-outage.yml` ask a **second** broker
(Fink LSST, public; Lasair when a token is set) for its newest LSST epoch, and
ALeRCE for a 120-night detection histogram. Verdicts: `MIRROR_STALLED`,
`SKY_STOPPED`, `UNDETERMINED_SINGLE_SOURCE` (only ALeRCE answered — explicitly
*not* SKY_STOPPED, since a confident single-source verdict would re-create the
blind spot), `NO_BROKER_REACHED`. Pinned by `tests/test_rubin_outage.py`.

**Verdict: `SKY_STOPPED`.** ALeRCE and Fink stop on the same night and their
nightly histograms agree count-for-count (2026-07-13: 744,559 both; 2026-07-14:
473,344 both), and the stream ends on a **full-sized night** rather than a
taper — a mid-ingest cutoff would leave the two brokers disagreeing. Rubin's own
forum posts, fetched on the runner, date it to the day: evacuation announced
**2026-07-15**, access road cut **07-20** ("off sky for days at a minimum"),
and as of the most recent post (**08-21**) the summit is being brought back up
on a backup generator — "lights for the first time since mid July" — with water
and fuel delivery the stated prerequisites for sustained operation. No restart
date announced. **Our last epoch is the night of 13/14 July, the last night
before the evacuation.**

Consequences, all recorded in `docs/rubin-outage.md`: every null either channel
files from 2026-07-15 on means *no new sky*, not *clean sky*; the frontier
thresholds are measuring a real condition and must **not** be relaxed;
`alerts.py::outage_context` carries the verdict into both frontier alert bodies
and drops it automatically once the frontier moves past the epoch it explains.

**Incidental, and it kills the cheapest substitute:** ALeRCE's non-LSST feed
(`sid=0`, 9.09M objects) has a newest epoch of **2026-04-30**, two and a half
months before the LSST feed stopped. Routing a starved channel to ZTF through
the same broker is not an option.

### The Rubin channels now run and report without a session, 2026-07-30

`tocsin` (nightly, 10:10 ET), `loom` (weekly, Mon 11:40 ET) and `loom-calibrate`
(monthly, 12:20 ET on the 1st) were already unattended crons committing results
back to `main`. What was missing was the half that closes the loop: something
that decides a human has to look, and says so out loud.

`alerts.yml` + `src/seti/alerts.py` do that. Three severities — `candidate` (a
channel promoted something), `health` (the pipeline is broken in a way that
produces *no error*), `milestone` (a capability came online). It opens a GitHub
issue **assigned to the repository owner**, because assignment is what reaches
an inbox regardless of the recipient's watch settings; the issue itself would
not. `loom-litcheck` now chains off `loom-calibrate` so the alert never
notifies about an exceedance the literature already explains.

Two design points that are load-bearing rather than decorative:

* **Deduplication by stable key** (`results/alerts/state.json`). A finding
  notifies once. Without it the first promoted candidate emails every week
  forever, and inside a month the apparatus has a working detector and a human
  who ignores it. A consumed alert is still reported as *active* — dedup must
  not make a live condition look resolved.
* **A separate check for the DATA stopping, not just the channel.** Both Rubin
  channels read through ALeRCE's mirror, which lags ~16 d (frontier MJD 61235 =
  2026-07-14 on 2026-07-30). If ALeRCE stops ingesting LSST, both channels keep
  running on schedule, keep writing a fresh run stamp, keep committing, and keep
  reporting a clean null — every liveness check stays green while we have
  silently stopped tracking Rubin. `health_alerts` therefore compares the wall
  clock to each channel's reported frontier and fires past 30 d (≈2× the
  measured lag). A clean null from a screen no longer being shown data is the
  most misleading thing this apparatus can produce.
* **Staleness is read from the timestamp INSIDE each result file**, never from
  its mtime. A runner clones the repository fresh, so every mtime is the
  checkout time and a channel dead for a year looks thirty seconds old. An
  mtime-based check would not merely be inaccurate on the runner, it could
  never fire there — and the daily heartbeat is the only thing that can tell a
  dead cron from an empty sky. Pinned by
  `tests/test_alerts.py::test_staleness_is_read_from_the_file_not_the_mtime`.

State seeded 2026-07-30 with `875163 (1998 SH2)` and `428209 (2006 VC)` marked
seen, so those two do not re-notify. Anything new does. See `docs/alerts.md`.

**Added 2026-07-31: the frontier not MOVING is a second, earlier check.** The
30-day test above measures the mirror against the wall clock, so it cannot fire
until a freeze has burned through the whole budget — and the mirror is already
~16 d behind *when* it stops, so a mirror that dies today is reported in a
fortnight. That fortnight is a run of clean nulls meaning *no new sky* filed as
*clean sky*. `results/alerts/frontier.json` is the memory that fixes it: nothing
else in the repository has any, because every channel result file describes the
run that wrote it, and from a single file a frozen mirror and an advancing one
are indistinguishable.

* `first_seen_utc` of the current value **is** the stall clock — the last time
  the frontier was observed to move. It is preserved by a run that sees no
  change, and the frontier is folded in only on the *recording* pass, never the
  dry run. Either mistake resets the clock every run and silently disables the
  check while it still looks implemented. Both are pinned by tests.
* Reads the **broker** frontier over the **screened** one
  (`summary.json:broker_frontier_mjd` before `ledger.json:last_mjd_screened`).
  Equal while a channel is caught up; they come apart exactly when it matters,
  because a channel that breaks while the mirror advances freezes the *screened*
  frontier — a channel bug, not a mirror outage.
* **The 7-day threshold is a placeholder and says so in the code.** ALeRCE's
  ingest cadence has never been measured here: the frontier has sat at MJD
  61235.41918 unchanged across every committed run since 2026-07-30T10:26:40Z.
  A threshold below the real batch interval would fire on ordinary behaviour and
  be trained into noise — the failure this module exists to avoid — so 7 d is
  deliberately above any plausible batching and still 4× faster than the age
  check. `observed_advance_days` accumulates the real cadence run by run;
  tighten `FRONTIER_STALL_DAYS` from that record, not by guessing again.
* Seeded from the record git already held, so the clock starts at the earliest
  committed sighting rather than at the moment the check was written. Those
  stamps are **lower bounds** — the freeze may predate either channel's first
  run — which delays the alert rather than raising it early.
* The frontier is reported in `latest.json`, the workflow log and the CLI
  **whether or not it alerts**: below the threshold a stalling mirror is
  invisible in every other field, and "how old is the newest sky we have seen"
  is the first thing to check before reading any null as a claim about the sky.

Verified live on the runner 2026-07-31 16:58 ET: `frozen_days` 1.44 (tocsin) /
1.32 (loom), silent, `first_seen_utc` preserved across the run.

### New channel: LOOM — von Neumann probe *population* search in Rubin SSO alerts (`loom/`), 2026-07-30

**The question this repository has not asked: is there a *population* of
artificial objects already in the solar system?** A von Neumann probe is defined
by self-replication, so the observable is not one anomalous object — it is a
population sharing an origin. LOOM screens solar-system objects one at a time and
**decides on the set**, against a null of matched random subsets of the same
screened sample.

**The observable, and why it is new.** `ssSource.ephOffset*` — the
observed-minus-predicted ephemeris offset, decomposed into along-track and
cross-track (arcsec, UCD `stat.fit.omc`), delivered **per detection for every
known minor planet**. Verified against the upstream LSST `v11_1.ssSource` and
`v11_1.mpc_orbits` Avro schemas; ALeRCE exposes the fields verbatim, lower-cased,
in schema order. These fields first appear in alert schema v10.0 (2025-11-24) and
went world-public 2026-02-24 — a five-month-old public data product that no Rubin
roadmap claims and no broker ingests.

**Prior-art position** (sweep on the runner: 178 arXiv queries across 8 angles, 108
IDs verified against real titles before citation, term-occupancy counted over 34
full texts; evidence under `results/vnprobelit{,2,3,4,5}/`):
- **Unoccupied:** `ephOffset` as a population observable
  (`abs:"ephemeris" AND abs:"offset" AND abs:"alert"` → **0 results**;
  `abs:"astrometric residuals" AND abs:"anomalous"` → **0**). The phrase
  "ephemeris residual" appears **zero times** in SNAPS, Fink-SSO, AHA, Lazio's
  2026 review, Davenport et al. or Ellery.
- **Unoccupied:** the replication axis, in any dataset. `abs:"self-replicating"
  AND abs:"solar system" AND abs:"technosignature"` → **0**. The only paper on
  self-replicating-probe technosignatures (Ellery, arXiv:2510.00082) is theory —
  69 mentions of "self-replicating", 139 of "lunar", **zero** of
  "non-gravitational", "Yarkovsky", "ephemeris residual" or "LSST".
- **Unoccupied:** SFD/albedo homogeneity as an *artificiality* discriminant
  (`abs:"manufactured" AND abs:"population" AND abs:"asteroid"` → **0**).
- **OCCUPIED IN FLIGHT — do not lead with it:** single-object anomalous
  acceleration. Lazio's *Solar System Technosignatures* review (arXiv:2606.13797)
  names the signature explicitly and cites **Lazio & Mahabal 2026, *On Anomalous
  Asteroid Accelerations*, Acta Astronautica, submitted** — not on arXiv, right
  authors, same observable.
- **ALREADY NULLED IN-HOUSE:** the catalogue-scale single-object version is
  `results/derelict/` — an `A1` census over 1,553,263 asteroids + 4,069 comets,
  verdict `ALL_SURVIVORS_EXPLAINED`. Per `CLAUDE.md` a clean null moves the
  *question*, which is exactly what LOOM does.
- **DO NOT CLAIM:** population photometric outlier detection on minor planets.
  SNAPS (arXiv:2302.01239, 2405.20176, 2604.27420) already does it at Rubin scale
  over 15 features — none a dynamical residual. The defensible claim is only that
  nobody reads them as technosignatures or fuses them with residuals.

**The gate is a theorem, not a fit.** Yarkovsky is recoil from re-radiated
sunlight, so thermal photons cannot carry more momentum than the intercepted beam:
`|a| ≤ ε·(Φ_1au/c)·(A/m)·(1au/r)²`. This holds whatever an object's spin,
obliquity, albedo or thermal inertia — none of which are known for almost any
object in the sample, which is why a thermophysical model would not do. Calibrated
on three objects with published `A2`: **ε_eff = 0.079 (Bennu), 0.034 (2005 ES70),
0.020 (2009 BD)** — so ε=0.1 is already generous, ε=1 unreachable, ε=2 the
specular limit for any radiation-driven process. Pinned by
`test_momentum_ceiling_matches_measured_objects`.

**Amplitude is NOT the discriminant** — the channel's tightest constraint.
`ephOffset` is Rubin's ~10 mas position minus an MPC prediction fitted to decades
of heterogeneous astrometry with star-catalogue biases up to **175 mas**, so
0.1–1″ residuals are routine and carry no information. The channel keys on
**geometry** (a transverse force displaces along-track; catalogue bias and
mis-association are isotropic), **time structure** (monotone growth across
apparitions; which heliocentric-distance law the drift follows), and
**independence from orbit quality** (a blind Yarkovsky search returns a *majority*
of spurious detections at nominal S/N>3).

**Promotion requires an artificiality channel, never magnitude alone.** Large
acceleration in an inactive body is Seligman et al. 2023's dark comets, already
explained by outgassing. Area-to-mass ratio is where outgassing and engineering
part company — mass loss raises acceleration but does not turn a rock into a thin
shell. Natural small-NEA locus ~3e-4 m²/kg; **J002E3 7.9e-3, WT1190F 1.18e-2**,
implying ρ·D ≈ 130–190 kg/m², i.e. ρ ≲ 100 kg/m³ for a metre-scale body.

**The one thing no other channel here has: a positive control.** `J002E3` (Apollo
12 S-IVB), `WT1190F`, `2020 SO` (1966 Surveyor 2 Centaur, confirmed by 301
stainless-steel NIR spectroscopy) and `2007 VN84` (Rosetta) are **real artificial
objects that a survey catalogued as minor planets**, identified by exactly this
observable. If the screen does not recover them it does not work — a falsifiable
statement about the pipeline, from real data. `control.validate` reports
`NO_CONTROLS_PRESENT` as **unexercised, not passed**, which is the expected
outcome until the survey catalogues a new one.

**Mis-linkage is collapsed before any statistic runs.** `|A_i| ≳ 1e-8 au/day²`
breaks MPC linking outright and tracklets can attach to *multiple* designations —
so one accelerating object can enter the catalogue several times with
near-identical elements and disjoint epochs, which is precisely what this channel
would otherwise call a family. The separator is epoch overlap; real family members
are observed contemporaneously. DERELICT's survivor base rate for this failure
mode was 100%.

**Two unit traps, both guarded by regression tests.**
`lsst_mpc_orbits.yarkovsky` is in **1e-10 au/day²** (Bennu's `A2 = -4.62e-14`
appears as `-4.6e-4`; reading it raw overstates every acceleration by ten orders
of magnitude and flags the whole catalogue). `srp` is in **m²/ton**. `a1/a2/a3`
are *also* labelled m²/ton, which is dimensionally wrong for Marsden
accelerations, so those three are treated as unit-unverified and unused.

**State: built, offline-tested (86 tests), FOUR live runs on 2026-07-30, final
state clean and self-consistent — and the honest verdict is "not yet", not a
null.**  Final run: 66,686 orbits quality-gated, 2,759 eligible (>=12
solar-system detections), **all 2,759 screened** with no top-N selection,
`shortlist_in_parent_fraction` 1.0, 0 spurious controls forced, 1,875 s.
Funnel: **0 candidate, 0 interest, 0 watch, 7 ordinary, 66,679 untestable**;
controls `NO_CONTROLS_PRESENT` (0 matched); replication
`INSUFFICIENT_POPULATION`.  Every one of the 2,759 residual series is
untestable on the acceleration axis because the baseline is two weeks — the 7
`ordinary` are the Path A objects, which need no survey baseline. Full account in `docs/loom.md`
§2.1-2.2. The funnel is now sound: 66,686 orbits pass the quality cuts, 2,759 have
>=12 solar-system detections, shortlist and parent are the same population
(fraction 1.0), the join is 1:1, 2,287 objects analysed in 20 batched queries in
967 s. The offset reconstruction validates against the survey's own `ephoffset` to
**1.4e-08 arcsec**.

**Two systematics were found and traced before being believed, both mine.**

1. A run reported `REPLICATION_STRUCTURE_DETECTED` on 150 anomalies with two
   statistics at the randomisation floor. `analyse_series` was using
   `scatter / sqrt(n)` as the per-point astrometric error — that is the error on
   the *mean*, understating the per-point value five-fold at 25 detections and
   inflating every acceleration S/N by 5 and every delta-chi-squared by 25. The
   corroborating evidence was already in the output: the score correlated with
   detection count at rho = -0.475, the "ranks objects by how well they were
   observed" failure, and the warning missed it because the threshold was 0.5.
   Fixed with a two-pass fit rescaling sigma by sqrt(reduced chi-squared) about the
   *fitted model*, never below the instrumental floor. **150 anomalies became 4.**
2. `normalise_designation` collapsed every provisional designation to its discovery
   **year** — `2020 SO` -> `2020` — so the control index matched hundreds of
   ordinary asteroids, 287 were forced into a shortlist as "positive controls", and
   a run reported that 2020 SO and the Rosetta spacecraft were in the sample.
   **They were not.** Corrected verdict: `NO_CONTROLS_PRESENT` — the control is
   unexercised, not passed.

**The binding limit is survey age.** The four surviving anomalies had residual
series spanning **2 to 29 days** against orbit arcs of 8,000-16,000 days, with
implied accelerations 10^4 to 10^8 times the momentum ceiling — fit blow-ups from
extrapolating a quadratic off a two-week baseline. LSST survey proper began
2026-06-30, so every object has ONE apparition, and the channel's central novelty
claim (which heliocentric-distance law the drift follows) returns
`INSUFFICIENT_R_SPAN` for every object. `min_residual_arc_days = 180` and
`min_apparitions_for_promotion = 2` now enforce that; all four become `untestable`
with the reason named and the assessment returns `INSUFFICIENT_POPULATION`.

**THE SIX EXCEEDANCES ARE RESOLVED — NOTHING SURVIVES, and the resolution is
itself the channel's best validation.** Of 589 asteroids with a fitted A2, eleven
exceed the hard momentum ceiling. Four were already labelled ('Oumuamua, 362P,
and two Seligman-2023 dark comets). The other seven were vetted on orbit quality
and A2 signal-to-noise; six passed. `loom-litcheck` then asked the literature
directly (full text of eight dark-comet and Yarkovsky papers, plus arXiv search
per designation), and **four of the six are in arXiv:2412.07603**, the dark-comet
follow-up — including both of the strongest:

| object | eps (rho=2000) | eps (rho=1000) | A2 S/N | in literature? |
|---|---|---|---|---|
| (2012 UR158) | 26.7 | 13.3 | 107 | **YES** — 2412.07603 dust-limit + non-grav tables |
| 452639 (2005 UY6) | 7.1 | 3.6 | 11.3 | **YES** — 2412.07603 |
| 152667 (1998 FR11) | 2.8 | 1.4 | 4.5 | **YES** — 2412.07603 |
| 139359 (2001 ME1) | 2.5 | 1.2 | 12.6 | **YES** — 2412.07603 |
| 875163 (1998 SH2) | 1.6 | **0.79** | 14.3 | not found |
| 428209 (2006 VC) | 1.3 | **0.65** | 3.8 | not found |

**LITERATURE CHECK WIDENED TO 116 PAPERS — the answer did not move.** The first
check rested on eight hand-picked papers, which tests my reading list rather than
the field. `loom-litcheck` now BUILDS its corpus: eight topic queries across the
non-gravitational literature (Yarkovsky, nongravitational acceleration, dark
comets, active asteroids, main-belt comets, NEA orbit determination), every
returned paper full-texted. **116 papers, 8.3 million characters, 100% fetch
success** (90 PDF, 26 HTML) — no silent gaps in the corpus.

Result unchanged: **4 of 6 explained, all in arXiv:2412.07603** — (2012 UR158),
452639 (2005 UY6), 152667 (1998 FR11), 139359 (2001 ME1). Widening the search
14-fold found nothing new, which upgrades the remaining claim from "not in eight
papers I chose" to "not in 116 papers spanning the field".

**Still not found: 428209 (2006 VC) and 875163 (1998 SH2).**

**What that claim still does NOT cover, and it matters:** arXiv only — no ADS, no
journals without preprints, no MPECs or MPC circulars; the corpus is
relevance-ranked at 20 results per query, so it is a slice of the field rather
than the field; and JPL fitted an A2 for both objects, which is itself a
deliberate act by an orbit-determination pipeline. Neither object is *unexamined*.
"Not found in 116 searched papers" is the honest ceiling on the claim.

**TISSERAND REVERSES THE RANKING (2026-07-30).** The one discriminator that
assumes no density and no albedo — T_J from a, e, i alone — was computed for all
eleven exceedances, and it changes which object is interesting:

| object | T_J | e | albedo | in literature? |
|---|---|---|---|---|
| 139359 (2001 ME1) | 2.67 | 0.87 | — | yes |
| 883607 (2016 TA56) | 2.69 | 0.78 | — | no (failed vetting) |
| 152667 (1998 FR11) | 2.89 | 0.71 | — | yes |
| **875163 (1998 SH2)** | **2.91** | **0.71** | **0.058** | **no** |
| 457175 (2008 GO98) | 2.93 | 0.28 | — | 362P, active |
| 452639 (2005 UY6) | 2.94 | 0.87 | 0.018 | yes |
| 523599 (2003 RM) | 2.95 | 0.61 | — | yes (dark comet) |
| (2012 UR158) | 3.00 | 0.86 | 0.023 | yes |
| **428209 (2006 VC)** | **3.72** | **0.49** | — | **no** |
| (2006 RH120) | 5.93 | 0.02 | — | yes (dark comet) |

**875163 (1998 SH2) is now largely explained.** T_J = 2.91, e = 0.71 and a
*measured* albedo of 0.058 make it dynamically and photometrically
indistinguishable from the confirmed dark comets beside it (2003 RM at 2.95,
1998 FR11 at 2.89, 2001 ME1 at 2.67, all dark). The natural reading is an
uncatalogued member of that population. My earlier "one genuinely open object"
call does not survive this.

**428209 (2006 VC) is the standout instead**: T_J = 3.72, a = 1.94, e = 0.49 —
the only literature-unmatched exceedance with **asteroidal** dynamics, where a
volatile reservoir should long since have been depleted. Its Del Vigna R is 5.8
to 16.4 across every plausible albedo, and R depends only on diameter, so the
density assumption that made its epsilon ambiguous does not touch it.

**But the control set undercuts the discriminator, and that has to be said:**
2006 RH120 is a CONFIRMED Seligman dark comet with T_J = 5.93 — more asteroidal
than 2006 VC. So T_J > 3 demonstrably does not exclude dark-comet status, and
2006 VC's asteroidal orbit makes it unusual, not unexplained. Its remaining
weaknesses are real: A2 signal-to-noise 3.8 (the base rate says most nominal
S/N > 3 Yarkovsky detections are spurious) and no measured diameter.

**THE TWO NOT IN THE LITERATURE WERE NOT FAIRLY RULED OUT — correction.** The
first write-up said both "fall below the ceiling under a generous density
(rho = 1000)" and called them not robust. That inverted the burden. Measured
asteroid densities run ~1200-1900 for rubble piles, ~1300 for C-types, ~2700 for
S-types: **rho = 1000 is the extreme low end, not a neutral default.** It is
chosen to make an exceedance hard to claim, so an object that drops below the
ceiling *only* there has been ruled out under an assumption almost no asteroid
satisfies.

The sensitivity grid (`calibrate.sensitivity_grid`, now in the code and tested):

| object | D | eps at rho = 1000 / 1500 / 2000 / 2500 / 3000 | Del Vigna R |
|---|---|---|---|
| 875163 (1998 SH2) | **383 m measured** | 0.79 / **1.18 / 1.58 / 1.97 / 2.37** | **11.8** |
| 428209 (2006 VC) | from H, albedo 0.05-0.25 | 0.49-1.09 / ... / 0.98-2.18 / ... / 1.46-3.27 | 9.8 |

**875163 (1998 SH2) is above the hard ceiling at every density except the extreme
low end**, has a *measured* diameter so the size is not assumed, S/N 14.3, a
27-year arc, 394 observations, condition code 0, and a Del Vigna R of 11.8 —
nearly six times the R <= 2 that marks a reliable Yarkovsky detection. It is not
ruled out. It is the one genuinely open object in this channel.

**428209 (2006 VC)** has no measured diameter and S/N 3.8, and spans 0.49-3.27
across the plausible grid. Genuinely unresolved in both directions.

Neither is a technosignature claim: an inactive body above the radiation ceiling
is what a dark comet looks like, and the dark-comet population is demonstrably
not fully catalogued — four of these six were only published in Dec 2024.

**Why this is a good outcome.** The ceiling independently rediscovered four
members of the published dark-comet population from H and A2 alone, with no
tuning and no knowledge of the paper — on top of 'Oumuamua, 362P and the two 2023
dark comets it had already recovered. Eight of eleven exceedances trace to known
anomalous objects. The paper's own A2 for 2001 ME1 (-2.47e-13) also matches the
SBDB value the pipeline used (-2.4e-13), an independent check on the input.

**A false negative in my own matcher had to be fixed first, and it mattered.**
LaTeX renders the order number as a subscript, so "2001 ME1" reaches a PDF as
"2001 ME 1"; my pattern required the digits adjacent to the letters, so it could
only match NUMBERED objects. (2012 UR158) is unnumbered and is the highest-S/N of
the six — it would have been declared absent from a paper it is squarely in.
Fixing the separator took the count from 2 to 4 and removed the top candidate.
Tests now cover the spaced form and the boundary that must not loosen with it.

**CALIBRATION (2026-07-30, `loom-calibrate`, needs no survey baseline).** 939
objects with a fitted A2 from JPL's SBDB. **The ceiling separates comets from
asteroids cleanly, without being told which is which**: 81/81 comets exceed it
(median eps_eff 7,298 — correct, a comet accelerates by shedding mass), and only
11/589 asteroids (1.9%) do. **Those eleven are the known anomalous population** —
1I/'Oumuamua at eps = 1.1e4 (matching the literature's independent ~1e4, from H
and A2 alone), 362P/(2008 GO98), and 523599 (2003 RM) and 2006 RH120, two of
Seligman's seven dark comets; Phaethon and Toutatis just below at 0.22. Recovering
essentially the whole published population of anomalously accelerating inactive
bodies, with no tuning, is how the gate earns the right to flag anything else.

**It also caught an error in my own threshold**: ordinary thermal recoil realises
median 0.074 / p90 0.143 of the momentum budget, not the 0.02-0.08 the
three-object anchor implied — so `epsilon_realistic = 0.1` was the ~85th
percentile and the `watch` tier fired on 27% of all asteroids. Now 0.3. The
density sensitivity is real (4.4% at rho=1000 vs 27% at rho=2000) and is reported.

**The `yarkovsky` unit is settled**: 7 matched objects, median ratio 1.009e-10
au/day^2 per count against the documented 1e-10, scatter 5.4%.

**The one measurement the data DOES support today**, and it is worth more than a
null. Path A needs no survey baseline (MPC fits `yarkovsky` against decades of
astrometry), so the 12 objects with a genuine non-zero value are testable now.
Seven pass the S/N >= 3 gate and every one sits below even the *realistic*
thermal-recoil envelope, at **eps_eff = 0.017-0.027** (2016 NB1 S/N 40.5,
2004 DH2 15.1, 1937 UB 9.8, 2007 CT26 8.3, 1982 DB 6.7, 2004 MW2 5.3,
2011 AA37 3.9; drifts -102 to +20 x10^-4 au/Myr, squarely inside the published
Yarkovsky range). The momentum ceiling was calibrated on three objects at
eps_eff = 0.020-0.079; these seven are an independent set fitted by MPC rather
than by the calibration papers, and they land in the same band. **The gate is now
anchored on ten objects rather than three**, and eps = 0.1 is confirmed generous —
no reliably measured natural object comes within a factor of four of it. The one
object exceeding the realistic envelope (2002 AX51, 6.5x) has S/N = 1.16 and is
correctly recorded `untestable`: the most interesting-looking number in the table
is the one with no signal behind it.

Per CLAUDE.md a clean null is a reason to change the question — but this is a
coverage-limited non-result, not a null, and the discriminants become available as
the arc lengthens. The channel runs weekly and waits.

**Next actions.** (1) Let the weekly cron accumulate; the law and apparition tests
switch on at the second apparition (~6-12 months for main-belt objects). (2) The
timing veto is untestable while `ephrate` is zero-filled — find another handle or
quote the limitation. (3) The 1-arcsec association radius removes the most
anomalous objects by construction; the one lead it leaves is that a *truncated*
series is itself a signature, worth building. (4) Calibrate `a1/a2/a3` units
against published JPL solutions before ever using those columns. (5) Read the
Del Vigna / Greenberg per-object A2 tables on the runner to turn eps_eff from a
3-object argument into a ~250-object measured distribution.

Docs: `docs/loom.md`. Config: `config/loom.yaml`. Workflows: `loom-probe.yml`
(dispatch), `loom.yml` (weekly, 11:40 ET Mondays — offset from TOCSIN so the two
do not hit the same TAP service concurrently).

### New channel: TOCSIN — nightly Rubin/LSST alert screen (`tocsin/`), 2026-07-30

**The first standing, recurring search this repository has had.** Every other
channel is a one-shot sweep of an archive; TOCSIN watches tonight's sky and
accumulates. Rubin alerts went world-public on 2026-02-24 and the survey proper
began 2026-06-30, and **as of 2026-07-30 no published SETI screen of real Rubin
alerts exists** — that window is the scarcest asset here.

**Signature.** S30 of `docs/necrosignatures.md` ("an unclassified blackbody
transient on a catalogued nearby dwarf, matching neither flare, nova, nor
microlensing") — the only event-residue signature in the taxonomy that needs a
*live* stream rather than an archive, and the one that was never built. Screened
in **both** difference-image polarities: `flash` (positive — S30 plus the
specular-glint reading, since a flat reflector returns the stellar spectrum and
is therefore grey where a flare is blue) and `dip` (negative — brief *grey*
occultation, the short-timescale end the ZTF `dimming` channel could not reach).

**The novel axis, and the honest narrowing** (prior-art sweep run on the runner:
65 arXiv + 31 OpenAlex queries, 37 full texts, citing-sets of 11 proposal papers;
evidence under `results/alertlit{,2,3}/`, every ID verified before citation):
- **Unoccupied:** cross-night *recurrence* of achromatic alert-stream events at a
  fixed position on a quiescent star. No prior art in any survey.
- **Proposed, never executed:** specular glint. Lacki 2019 (arXiv:1903.05839)
  computes LSST's reach; **12 citing works in 7 years, zero executions**. Rogers
  et al. (arXiv:2401.08763) explicitly flag it as the opportune extension.
- **DO NOT CLAIM AS NEW:** negative-flux alert screening. Gallay, Davenport &
  Croft (arXiv:2506.14744, AJ 2025) already do it on ZTF. Full-text grep: zero
  occurrences of `achromatic`, `colour`, `recurr`, `repeat` — their discriminant
  is single-band amplitude. Our dip mode is *the grey and recurrent variant*, an
  extension, on a new instrument.
- **Cite and differentiate:** Kovačević et al. (arXiv:2606.00574) simulate
  achromatic coherent variability in LSST colour space (simulation only, no
  data, periodic signals not events); AHA (arXiv:2602.12955) now occupies generic
  ML anomaly detection on alert streams, which this channel deliberately avoids.
- The Rubin TVS Roadmap's four planned technosignature families
  (arXiv:2208.04499) include **none** of glint, achromaticity or negative flux.

**Why a recurring screen can be honest.** Three disciplines, all unit-tested:
(1) significance is quoted against the **cumulative** target×night trial count
(BH-FDR across all targets ever screened) — a screen that forgets its history
manufactures a 3σ event every few weeks by construction; (2) the denominator is
measured from **forced photometry**, which exists whether or not anything was
detected, so the trial space is the well-defined *tracked* sample rather than an
assumption; (3) the timing null resamples each star's **own visited nights**, so
the ~3–4 day revisit cadence cannot read as a beacon.

**Why Rubin makes the discriminant work where ZTF's glint channel died (0/15,
all chromatic flares):** the LSST baseline takes intra-night pairs in *different*
filters ~33 min apart, so a colour is the **default** data product; and
`diaSource.templateFlux` supplies the quiescent flux in the same band and system,
so `dF/F*` carries no cross-survey passband error.

**Data path — no credentials.** ALeRCE's public IVOA TAP service
(`tap.alerce.online/tap`) is the only broker path that supports an unattended
cron: full ADQL, indexed on `mjd`/`ra`/`dec`, whole-night queries, plus the bulk
forced-photometry table. Lasair needs a token and allows 100 calls/h; Fink has no
whole-night endpoint and its bulk path requires a human web form.

**Known, unfixable-by-us incompleteness (must be quoted in any result):** Rubin
applies `minReliability: 0.5` *before* issuing an alert, and DMTN-337 measures
that model's true-positive rate on **variable stars at 3.5%** (v0.1/DP1); v0.3
still scores Gaia variables low. Our signal is a stellar point-source event, so
the stream is biased against it by someone else's classifier. The channel
therefore applies **no additional reliability cut**, and a null here is weak
evidence about the sky and strong evidence only about the alert stream.

**State: built, offline-tested (80 tests), and probed against the live service.**
`results/tocsin/probe.json` holds the measurement record. The probe earned its
keep — it caught three bugs that would each have produced a *confident null*
rather than an error, which is the failure mode this repository fears most:

1. `gaiadr3_source` has **no `source_id`**; the join key is `oid_catalog` on both
   sides. This broke the nearby-star pre-cut *and* the forced-photometry
   denominator — numerator and denominator of the recurrence statistic.
2. `oid_catalog` **cannot be SELECTed** (declared integer, but AllWISE ids are
   strings → VOTable serialisation error). Fine in a JOIN, which is all we need.
3. **The empty window was not a query bug at all**: ALeRCE's TAP mirror lags
   **15.6 days** (newest LSST epoch MJD 61235.4 vs wall clock 61251.1). Asking
   for "the last two nights" would have returned nothing every night forever.
   The screen now anchors to the broker's own frontier and advances a
   **watermark**, so coverage is gapless and non-overlapping whatever the lag does.

**The finding that changes the schedule: there is a 262-night backlog.** LSST
detections in ALeRCE already span MJD 60973 → 61235, so the first runs are a
*backfill of real archival data* rather than a wait for new sky — the recurrence
statistics that need many nights are days away, not months.

**Measured, no longer guessed:** `sid=0/tid=0` = ZTF, `sid=1/tid=1` = LSST
diaObject (5.16M), `sid=2/tid=1` = LSST ssObject (131k) — so `sid=1` is correct
and drops ~300k solar-system detections per 30 days; `catid=1` = Gaia DR3,
`catid=0` = AllWISE. The Gaia join returns real nearby stars (parallax ~12 mas at
~1.4″). All now live in `config/tocsin.yaml`, not in code.

**Two statistics bugs fixed before any data was screened**, both found by tests
rather than by inspection: the duty-cycle cut was rejecting every *first-night*
detection (one visit, one event → duty cycle 1.0 by arithmetic), and overlapping
run windows were double-counting trials, which deflates the ensemble rate and
makes every p-value **too small** — anti-conservative, i.e. it manufactures
significance. The ledger now folds night by night.

**RUNNING ON REAL DATA (2026-07-30).** Verification window MJD 61228→61235.4,
254k-target Gaia list, ~62 s of network time (`night_detections` 42.8 s,
`forced_photometry` 13.8 s, `footprint` 5.2 s):

| quantity | value |
|---|---|
| funnel | 378 detections → 238 quality → 22 matched → 20 associated → **13 events** |
| rejections | 140 extended, 5 **chromatic**, 2 astrometric offset |
| denominator | `observed_footprint`, **16,816 star-nights** |
| ensemble rate | **9.8×10⁻⁴** per star-night (a real rate, not a tautology) |
| baselines | 13/13 from Rubin's own `templateFlux` |
| tiers | 16 watch, 3 interest, **0 candidate** |

Three things that matter in that table. The **denominator works** — forced
photometry covered 0%, so it comes from the observed footprint instead
(detections trace where the camera pointed). The **colour test fires**: five
events rejected as chromatic by redder-band non-detection, a rejection that had
never once fired before — this is precisely the discriminant the ZTF glint
channel died on (15 candidates, all chromatic flares). And **zero candidates**,
which is correct: promotion requires recurrence, and recurrence needs nights.

Two further bugs the live runs exposed, both fixed: every event was single-band
so the two-band colour test was dormant (hence the one-sided non-detection test,
`docs/tocsin.md` §3.2), and the reach metric counted only *surviving* events —
so a window in which the test had just killed five flares reported "the
discriminant did not run at all", concealing its own success.

**FULL BACKLOG WALKED (2026-07-30).** All 262 nights the broker holds
(MJD 60973 → 61235), watermark at the frontier, nightly cron now the steady state:

```
263 nights · 55,424 star-night trials · 87 events · 42 targets with events
all-sky rate 1.57e-03/star-night · 1,927 sky bins · FDR threshold 0.0136
tiers: 25 watch · 13 interest · 0 candidate · 4 none
```

**Three candidates appeared and all three were killed by contamination tracing —
and both killing discriminators came from examining them, not from theory:**

1. **Deep-drilling fields.** The first two sat in COSMOS, where the *local* alert
   rate is 4–75× the all-sky value (a DDF is deeper, revisited 34–48 nights
   against 7 elsewhere, and subtracts differently). Testing a DDF star against
   the all-sky rate does not detect anything, it rediscovers the observing
   strategy. The null is now **stratified by 1° sky bin**; their p-values moved
   from 1.3e-03 and 2.7e-07 to 0.035 and 0.003.
2. **Low-amplitude variable stars.** All three showed **both polarities across
   nights** — flash on some, dip on others. The first fix vetoed mixed polarity
   outright; that was wrong in the expensive direction, because a megastructure
   is the hypothesis this repo chases from *both* sides (`dimming` occults,
   `glint` reflects), so a real structure should show both. The discriminator is
   **colour, not sign**: a variable crosses its own template mean, so its
   excursions are one continuous *chromatic* variation, while an engineered
   occulter-plus-reflector is grey both ways. Mixed polarity is therefore
   admitted on **grey confirmation in each polarity independently**
   (`docs/tocsin.md` §4.2). All three real candidates had a grey-confirmed flash
   but a single-band untested dip → still rejected, zero candidates, and a
   genuine dual-mode object stays reachable.

The 4 `none` targets are duty-cycle rejections — stars alerting on most of their
visits, i.e. subtraction residuals. The highest-multiplicity target in the walk
(7 events in 7 visits, duty 1.0) is one of them.

**Honest summary: the funnel works, every discriminator fires on real data, and
nothing survives.** Per the charter that is a reason to keep accumulating and to
sharpen the question — not a result to write up.

**Operational lessons worth not relearning.** (a) `forced_photometry` on this
broker takes 3151 s and then times out for 0–0.08% coverage; the footprint query
answers the same question in 5 s, so forced photometry is off and the visit
history is footprint-derived. (b) Three separate bugs had the same shape — state
handed to one arbitrary night of a multi-night fold and silently dropped (visit
history, bin trials, and the alerts counter). Anything accumulated per-window
must be keyed by night. (c) The tests passed through all three, because they
exercised single-night folds where that path never runs.

*Next decisive actions:* wire the Fink cross-match (SIMBAD/VSX/GCVS) so known
variables are labelled at ingest rather than diagnosed by hand; calibrate
`max_trail_arcsec` against a real `trailLength` distribution.

### New: 5-channel fan-out searching for life originating on LHS 1140 b (2026-07-21)

Five parallel-subagent-built channels (77 offline tests total), all runner-dispatched.
Each degrades honestly and never fabricates data.

1. **`jwst_bio`** — real JWST transmission-spectrum biosignature analysis (build the
   spectrum from `x1dints` in/out-of-transit; disequilibrium-*pair* logic CH₄+CO₂/
   O₂+CH₄, never a single gas; M-dwarf abiotic-O₂ gate; MIRI eclipse
   atmosphere-vs-bare-rock; laser scan). Took **three runs and two real
   data-access fixes** to reach the archive: (1) the MAST obs filter required
   `dataproduct_type=="spectrum"` but JWST TSO is `"timeseries"` → 0 products
   (fixed → 70 x1dints found); (2) all 70 downloads hit the MAST
   `download_products` "varchar to bigint" server bug → switched to
   `download_file(dataURI)`. **Run 3 reached real data:** `data_reached=true`, 2
   x1dints stacks read (NIRISS + NIRSpec). **Verdict `no_biosignature_detected`
   — correct and robust:** NIRISS covers only 0.85–2.83 µm, so CH₄ (3.3)/CO₂
   (4.3)/O₃ (9.6 µm) are out of range → **no redox-disequilibrium pair is even
   possible**, and a single gas is never a biosignature. **Vetting note (do not
   overclaim):** the pipeline's apparent "H₂O 1.4 µm, 72 ppm, 7σ" is a **reduction
   artefact, not a real detection** — the reader saw only 3 integrations (it treats
   each EXTRACT1D HDU as one integration, but modern x1dints pack all integrations
   in one 2-D table), the in/out split ran with `ephemeris_used=false`, and the
   per-band σ is inflated by counting ~46k correlated native pixels as independent.
   A valid transmission spectrum here needs multi-integration format handling +
   ephemeris phasing + systematics detrending + de-correlated binning — the
   publication-grade retrieval the channel explicitly disclaims. **Bottom line
   unchanged:** the biosignature *detectability* answer governs — LHS 1140 b's
   compact high-μ atmosphere puts every biosignature ~25+ transits out of reach, so
   none is detectable with current data. The infrastructure now reaches the real
   spectra; a genuine retrieval is the honest next boundary, not a runner task.
2. **`lhs1140_origin`** — panspermia **donor** list (classical rocky-HZ prior,
   mirror of K2-18). **Run OK:** 10,974 Gaia 6D stars → **22 recipients** within
   2 pc over 10 Myr, **0 co-movers**, closest approach 0.26 pc — but **all fast
   flybys** (top v_rel 51 km/s, transfer scores ~1e-3 = non-capturable), exactly
   like K2-18: no slow/close passive bridge. The directed-travel (technological)
   destination list ranks **5,490 reachable known-planet hosts**, top rocky-HZ
   targets HD 216520, HD 210277, HD 215497… (all temperate-planet hosts, reachable
   in 400–1,200 yr at 0.1c). Net: passive panspermia closed; directed-travel gives
   a concrete ranked target list.
3. **`crosscorr`** — high-res Doppler cross-correlation (O₂ A-band + H₂O, Kp–Vsys).
   **Run: `NO_ARCHIVAL_IN_TRANSIT_HIRES_SPECTRA_AVAILABLE`** — ESO archive reachable
   (118 ESPRESSO records) but **0 in-transit transmission sequences** (all
   out-of-transit RV monitoring); one LHS 1140 b transit sweeps Kp by only ~0.9 km/s,
   so a real search needs many stacked dedicated transits. Engine validated on
   injection; honest data-gap verdict.
4. **`seti_archive`** — targeted radio/optical SETI coverage + EIRP limits.
   **Run: `NO_TARGETED_RADIO_SETI_ON_RECORD`** — a genuine observational gap on a
   landmark HZ world; representative limits show a modest MeerKAT/GBT/Parkes pointing
   would constrain narrowband beacons to ~2–12×10¹⁰ W (~10³× below the Arecibo
   planetary radar, well sub-Kardashev-I). Deliverable is the coverage+limit map.
5. **`iso`** — interstellar-object back-tracking ('Oumuamua/Borisov/3I) through the
   Galactic potential. **Run: clean null** — all three ISOs stay at LHS 1140's
   present ~15 pc under back-integration (`d_min_p50 ≈ 14.96 pc`, `t_enc ≈ 0`),
   `any_consistent_with_origin=False`. The necessary-not-sufficient caveat ships as a
   first-class field (degree-scale radiants smear parsecs; apex projection; disk
   prior). None traces back to LHS 1140.

**Net across the fan-out:** no bio or techno detection; the two live scientific
outputs are the **directed-travel destination list** (a ranked answer to "which
rocky-HZ worlds could an LHS 1140 biosphere reach") and two identified **real
observational gaps** (no dedicated in-transit high-res spectra; no targeted radio
SETI) on a landmark world. The one channel that would give a *positive* atmospheric
measurement — `jwst_bio` on the actual JWST spectra — is re-running after the TSO
filter fix.

### New channel: long-baseline Galactic-orbit encounter search (`galactic/`)

**Question (user-directed, 2026-07-21):** expand the bio/techno-signature search
to **both** nearby biosignature-anchor systems (LHS 1140 + K2-18) **and any star
that passed near them over the past few hundred Myr**. The panspermia encounter
code is linear-motion (valid only ~10 Myr); a hundreds-of-Myr baseline **requires
integrating orbits in the Galactic potential** — differential rotation and the
vertical tide bend every trajectory well inside that window.

**Method (dynamics unit-tested offline, `test_galactic.py`, 5 tests):** an
axisymmetric MW potential (Miyamoto-Nagai disk + logarithmic halo, flat rotation
curve `V_c(R0)=232 km/s`) with a vectorised velocity-Verlet integrator; resolve
each anchor's 6D phase space, pull the RV-complete Gaia sample in a present-day
sphere, integrate every orbit back `t_max` (300 Myr), and track each star's
closest approach with an **analytic per-step segment minimum** (so a ~30 km/s
flyby is not stepped over between samples). Monte-Carlo the closest encounters and
report a **timing-recoverability flag** — the honest horizon beyond which phase
mixing erases the encounter *time* even where `d_min` stays robust. Cross-match the
shortlist to the NASA Exoplanet Archive and run the signature battery: the
astrometric hidden-companion (techno) screen on every encounter star, and the
biosignature-detectability (bio) answer on the anchors + any planet-hosting
encounter systems.

**Bio contrast that validates the framework (computed offline, confirmed on the
runner):** biosignature detectability is set by the atmosphere's mean molecular
weight, and the two nearby biosignature worlds sit on opposite sides of the line —
**K2-18 b** (expected low-μ hycean H₂ envelope, scale height **H=79 km**) is
biosignature-**REACHABLE in <1 transit** (exactly why the contested DMS claim,
Madhusudhan+2023, was even possible there), while **LHS 1140 b** (high-μ rocky
secondary, **H=3.6 km**) needs **~25 transits** → not detectable. Same JWST, same
distance; the atmosphere decides.

**Result (run 29793496625, committed under `results/galactic/`): both nearby
systems and their few-hundred-Myr encounter neighbours are clean of any bio or
techno signature.** For each anchor **149,979** RV-complete Gaia stars were
integrated back 300 Myr:
- **LHS 1140:** 34 stars pass within 3 pc; **0 are known planet hosts** (no bio
  target), **0 carry an astrometric companion flag** (no techno). Closest pass:
  Gaia DR3 `1939760926285276544` (G=7.4, now 132 pc away) at **d_min=0.086 pc**,
  2.9 Myr ago, RUWE 0.88 = ordinary single star — an interesting closest-stellar-
  approach, not a signature.
- **K2-18:** 16 stars within 3 pc; **0 planet hosts, 0 companion flags.** Closest:
  Gaia DR3 `983333660069405824` at **d_min=0.25 pc**, 1.9 Myr ago.
- **Empirical recoverability horizon (the honest headline):** *every*
  reconstructable close pass is **recent** — all 34 of LHS 1140's are within the
  last 4 Myr; 15 of K2-18's 16 are within 20 Myr (one outlier at −155 Myr). The
  300 Myr integration surfaces **no datable 100–300 Myr-ago close encounter**: over
  that baseline phase mixing erases the timing, so the only stars still traceable to
  a <3 pc approach are those making *recent* passes. The Monte-Carlo confirms it —
  the closest encounters have `t_enc` spreads of ~0 Myr (tightly recoverable) and
  all sit in the last few Myr. A "few hundred Myr" encounter search therefore
  collapses, in practice, to a recoverable window of **~tens of Myr** — exactly the
  regime the linear panspermia search already covered, now put on a rigorous
  orbit-integrated footing with the horizon quantified rather than assumed.

Net: no encounter neighbour of either biosignature world is a planet host or shows
a hidden-companion technosignature; the anchors' own bio answers stand (LHS 1140 b
not detectable, K2-18 b reachable). 15 offline tests (10 galactic/bio + 5 dynamics).

### New channel: LHS 1140 system deep-dive (`lhs1140/`)

**Question (user-directed, 2026-07-21):** LHS 1140 b is a ~1.7 R⊕ temperate
habitable-zone rocky/water world (M4.5V host at 14.96 pc) with a *reported
atmosphere* (Cadieux+2024). Exhaustively search every observation ever done of
the planet, its sibling (LHS 1140 c), the star, and the stellar neighbours for
any bio or techno signature. Because both planets **transit**, every photometric/
spectroscopic observation of the star is also an observation of the planets, so
this is a full multi-archive sweep of the system plus a catalogue-scale battery
over the local volume. Channel: `src/seti/lhs1140/`, workflow `lhs1140.yml`,
8 offline tests.

**Method (reuses the panspermia per-target detectors + two new pieces):** resolve
LHS 1140's live Gaia DR3 row (nearest source in a PM-tolerant cone →
`2371032916186181760`), then run the full battery — Gaia astrometric
hidden-companion, WISE IR-colour excess, NEOWISE mid-IR variability, ZTF g+r +
**TESS/K2** photometry (which carry the b/c transits), Gaia XP narrow laser-line
scan — plus (1) a **neighbour sweep** applying the IR-excess and companion screens
to every Gaia source within a distance sphere (PM-propagated IRSA WISE cones,
NASA-Exoplanet-Archive cross-match), and (2) a **biosignature-observation
inventory** (MAST) recording what atmosphere-capable spectroscopy exists.

**Result (runs 2–3, committed under `results/lhs1140/`): the system is clean of
technosignatures in every channel that returned data.**
- **LHS 1140 (star + planets b, c):** clean in all 6 channels — no Gaia XP laser
  line; no WISE IR excess (W1–W2=0.22, W1–W3=0.37, both below threshold — ordinary
  M-dwarf colours); no NEOWISE mid-IR trend (296 epochs); no anomalous transit in
  **TESS (3,548 epochs)** or ZTF (the real b/c transits are periodic/shallow, not
  flagged). The lone flag is **RUWE=1.53** (just over the 1.4 line) — *not* a
  technosignature: astrometric excess noise is only 0.28 mas (below the 1 mas
  amplitude gate), no Gaia NSS solution; at most a faint unseen stellar companion,
  ordinary for a nearby high-PM M dwarf.
- **Neighbours (38 stars ≤15 pc):** the raw WISE screen flagged 15/38 IR-excess —
  a 40% rate that traces **entirely to AllWISE systematics**, not waste heat. The
  hardened screen (require the excess in a star-dominated band W1–W2/W1–W3 with a
  physical W1–W2≥−0.05; W4-only and negative-W1–W2 → `needs_vetting`) drops it to
  4 survivors, and those are all explained too: three are faint (G=18–20, WISE
  confusion-limited) and one is **blue (bp_rp=1.16, not an M dwarf** → photosphere
  model invalid → blend/background); the single reasonable one (G=14.4 mid-M,
  W1–W3=0.73) is a mild W3 excess with photospheric W1–W2 = ordinary debris/cirrus.
  **No neighbour shows the hot-band (W1–W2) waste-heat signature.** 6 neighbours
  carry elevated RUWE (ordinary binaries); 3 planet hosts in the volume
  (LHS 1140 + HIP 4845 + one more).
- **Biosignature inventory:** **533 MAST observations, 209 spectroscopic**,
  atmosphere-capable = **True** — HST (WFC3/STIS/COS 409) + JWST
  (MIRI/NIRISS/NIRSpec 71), 0.1–16.5 µm, ~1.53 Ms total exposure.
- **Biosignature ANSWER (`biosignature.py`, `results/lhs1140/biosignature.json`,
  6 offline tests):** the detectability calculation converts "the spectra exist"
  into the actual answer. A molecular biosignature lives in the transmission
  spectrum, whose feature amplitude is `2 Rp H n_H / Rs²` with scale height
  `H = kT/(μ g)`. For LHS 1140 b (g ≈ 18 m/s², a dense ~5.6 M⊕ super-Earth) the
  **physically expected high-μ secondary (N₂) atmosphere has H ≈ 3.6 km → ~3.6 ppm
  per scale height**, so against JWST's ~26 ppm per-bin per-transit noise **CH₄
  (3.3 µm) needs ~25 transits, O₃ (9.6 µm) ~67, N₂O/CH₃Cl/O₂-CIA more** — versus
  the handful (~2–4 epochs) actually observed. **Verdict:
  `BIOSIGNATURE_NOT_DETECTABLE_WITH_CURRENT_DATA`.** The required bands *are*
  covered (MIRI→O₃/N₂O, NIRSpec→CH₄/CO₂, NIRISS→H₂O/O₂-CIA), so this is a
  sensitivity limit, not a coverage gap. A biosignature would be reachable only
  under a cleared low-μ (H₂-rich) envelope (H ≈ 44 km, <1 transit) — which the
  planet's density and the existing atmosphere data **disfavour**. This matches
  the literature: LHS 1140 b shows a *tentative secondary atmosphere / water-world
  hint* (Cadieux+2024), **no biosignature gas**, and reaching one needs dozens–
  hundreds of transits.

**Read:** LHS 1140, its planets, and its ≤15 pc neighbours are **clean of any
technosignature** in every public archive reached (Gaia astrometry+XP,
WISE/NEOWISE, ZTF, TESS/K2), and the **biosignature question is now answered too**:
with current JWST data a biosignature is **not detectable** on LHS 1140 b, because
its expected high-μ atmosphere makes every biosignature feature a few ppm — dozens
of transits below reach, not a matter of looking harder at existing spectra. This
is a complete *characterisation* of an individual high-value system (both bio and
techno), not a population null. Known gaps: radio/SETI, high-res HARPS/ESPRESSO
RV, X-ray; a true spectral *retrieval* on the raw JWST products (vs this
signal-to-noise budget) remains a heavier, non-catalogue-scale follow-up.

## Current best candidates (cross-channel, ranked)

1. **167 triaged laser-line priority targets** —
   `results/spectra_triage/priority_targets.csv`. The former #1 (spec 068839f0,
   7518/7542 Å) is **DEAD** — see Resolved below; the two "beacon" lines are
   Hα + [N II] 6584 at z = 0.145, an emission-line galaxy SDSS misclassified as
   a STAR. A new galaxy-redshift-consistency stage now removes such objects
   (`triage_verdict = galaxy_zmatch`; 3 spectra cut, two of them
   SIMBAD-confirmed galaxies — KUG 1207+134, Z 521-35). None of the remaining
   167 is cross-confirmed.
   *Next decisive test:* the `spectra-confirm` repeat-visit path is exhausted
   (zero overlapping SPARCL spectra for 19 single-line targets). Real remaining
   route: **per-exposure persistence** — fetch the coadd-input exposures for each
   target from the SDSS SAS (new acquisition code, runner-side); a real line
   persists across exposures, a cosmic ray does not. Single-line targets cannot
   be galaxy-tested internally — the per-exposure check is what separates a true
   narrow emitter from a cosmic-ray hit for them.
2. **WD IR-excess multimodal candidates — RESOLVED, no technosignature.**
   The 23 multi-axis excesses were put through the empirical WISE-blend +
   co-movement test (`science-blend`, `results/science/blend_followup.csv`) and a
   dust-sublimation physical filter: **3 are background WISE blends** (a bright
   Gaia neighbour in the 6.5″ beam), **7 are unresolved stellar companions**
   (fitted T_dust > 1800 K = hotter than grains survive → a WD+dM/dL binary, a
   single Gaia source, which is why it looked "isolated" and periodic), and the
   **13 survivors all have τ < 0.081 — ordinary WD debris disks**, nothing
   swarm-like. The τ=0.607 standout (`235890564653455488`, T_dust 2282 K) is a
   too-hot-for-dust stellar companion, not a swarm. No candidate survives.

## Resolved (killed) candidates

* **Laser-line #1, spec 068839f0…** (SDSS-DR17, RA 25.6212, Dec −8.2417) —
  ranked first in the whole search: a 31.9σ unresolved line at 7517.96 Å plus a
  second surviving line at 7542.23 Å. **Killed 2026-07-01 by internal
  redshift-consistency**: the pair is Hα 6562.8 and [N II] 6583.5 redshifted to
  **z = 0.1452** (residual 22 km/s on [N II]) — a background emission-line galaxy
  the SDSS pipeline classified as `STAR` (catalogue z ≈ 0, so the observed-frame
  known-line triage placed Hα at 6563 Å and never saw it). New rejection
  `seti.spectra.galaxy_reject.galaxy_redshift_match` (verdict `galaxy_zmatch`).
  A locked diagnostic pair (Hα+[N II], the [O III]/[S II] doublets) or ≥3 lines
  at one z is required, so an emission-line variable star is not mis-killed
  (V345 Sge was correctly spared).
* **Astrometric dark-companion class-3 shortlist** — the 8 AMRF class-3 systems
  (BH1 + 7) were cross-matched against the published Gaia compact-companion
  catalogue Shahaf+2023 (VizieR J/MNRAS/518/2991, 101,380 source_ids loaded;
  `results/accel/literature_crossmatch.csv`). **7 of 8 are already in Shahaf+2023**
  (1 is Gaia BH1, the validation object) — the channel *reproduces* the published
  AMRF catalogue rather than extending it. One system, **Gaia DR3
  3027759739607108992** (852 pc, M₂≈4.4 M☉, RUWE 4.9, no SIMBAD), is absent from
  Shahaf+2023, but it is the *weakest* solution in the set (farthest, lowest
  RUWE, mass nearest the 3 M☉ floor) — most plausibly below Shahaf's quality
  threshold rather than a new object. Not a remarkable novel candidate; would
  need the Shahaf+2024/2019 lists to load and an independent orbit check before
  any claim. Per the novelty directive this channel is a reproduction — change
  the question, do not write it up.
* **Gaia DR3 1268299311319369984** (RA 225.0080, Dec +26.8728) — the
  ASAS-SN-confirmed secular fader (0.073 mag/yr at 8.8σ, ~0.94 mag total,
  RUWE 0.98, `non_single_star=0`). **Killed 2026-07-01 by the NEOWISE
  counterpart test**: W1 fades at 0.0045 mag/yr (8.4σ, 20 seasons, 345
  epochs), W2 at 0.0041 mag/yr (3.8σ) — an IR/optical slope ratio of 0.062,
  precisely the standard small-grain extinction-law prediction
  (A_W1/A_optical ≈ 0.06). This is ordinary dust progressively obscuring the
  star, not a gray occulter (which would fade the IR at ≳30% of the optical
  rate) and not warm circumstellar dust (which would *brighten* W1/W2).
  Verdict `ir_fades_reddening_law`;
  `results/dimming/characterization.json`. Still an interesting *astrophysics*
  object (a decade-long monotonic obscuration event), but not a
  technosignature.


### New question (post-null pivot)

**Population-level anomaly clustering** (`cluster/`, `results/cluster/`). Every
single-object channel above closed on contamination. New, degeneracy-immune
question: does the Gaia×AllWISE **IR-excess tail over-cluster in phase space**
(3D position + velocity) beyond a magnitude/colour/distance-matched random null,
as an expanding waste-heat-building population would? `matched_null_clustering`
is offline-validated (recovers an injected co-moving group at p<0.01, clean null
on confounded sets).

**First run (cone RA200/Dec0, r10°, ≤500 pc, G<16):** 21,675 Gaia×AllWISE stars →
195 IR-excess (z≥4) sources → **position clustering p=0.64, over_clustered=False**
— the excess tail traces the ordinary stellar density; the two friends-of-friends
groups (n=23 @33 pc, n=9 @69 pc) are fully accounted for by the matched null (a
nearby-star over-density), i.e. not anomalous. The method works and this cone is
clean. *Note:* position alone is washed out by the Galactic density gradient; the
run now also tests **velocity** and **full phase space** (a technological moving
group shares a space velocity).

**Three cones now, all clean nulls** in all three spaces (p_pos / p_vel / p_phase):
- RA200/Dec0:   21,675 stars, 195 excess — p_pos=0.64.
- RA150/Dec30:  22,074 stars, 186 excess — 0.37 / 0.31 / 0.08.
- RA30/Dec−20:  22,295 stars, 183 excess — 0.61 / 0.65 / 0.59.
Consistent across ~66k stars in three independent ≤500 pc volumes: the WISE
IR-excess tail does **not** over-cluster in position, velocity or phase space; the
FoF groups are all nearby-density and non-significant under the matched null. This
is a robust null for the IR-excess indicator. A genuinely new signal would be
p_vel/p_phase ≪ 0.05 with an FoF group the matched null cannot reproduce.
*Next options:* (a) keep sweeping cones -> occurrence-style volume limit;
(b) point the same clustering engine at a **different anomaly axis** (astrometric
companion excess, UV/optical energy imbalance) where a clustered technological
population might show even though IR excess does not.

### New channel: K2-18 panspermia close-encounter search (`panspermia/`)

**Question (user-directed, 2026-07-02):** K2-18 b is the hycean world with a JWST
biosignature hint (DMS/DMSO; Madhusudhan+2023/2025 — contested; treated as the
*premise*, not a result). *If* life arose there, which stars could have received
K2-18-origin material? The transfer vector is unbound ejecta / dormant spores /
free-flying 'Oumuamua-class bodies, so the filter is **encounter geometry (close +
slow)**, not a continuous bridge — and because the stellar neighbourhood
**reshuffles over time**, the search is over *closest approach in full 6D phase
space*, not present-day proximity. This is a novel anchor + novel question (nobody
has computed K2-18's stellar-encounter recipient list); it is not a refinement of
any existing SETI baseline.

**Method (offline-validated, `test_panspermia.py`, 7 tests):** resolve K2-18's 6D
vector from Gaia DR3 (radial velocity essential); pull every Gaia DR3 source with
an RV in a heliocentric distance shell bracketing the search sphere; build
heliocentric Galactic 6D `(X,Y,Z, U,V,W)`; compute each star's **linear
closest-approach** to K2-18 (`t_enc`, `d_min`, `v_rel`) — the standard
straight-line treatment used for the Sun's own encounter list (García-Sánchez
2001; Bailer-Jones 2015+), valid over the recent few-Myr window where the Galactic
tide is negligible. Rank *past* (`t_enc<0`) close/slow encounters by a
transfer-plausibility score `(d_ref/d_min)·(v_ref/v_rel)²` (velocity-squared
mirrors the gravitational-capture cross-section; ordinal, not a probability), and
separately tag **co-moving companions** (shared low velocity + present proximity),
the strongest bridge of all. Relative velocities are frame-independent of the
solar motion (it cancels in the difference), so no LSR constants enter.
*Caveat:* linear motion is honest only inside `t_max` (default 10 Myr); a longer
baseline would need epicyclic/Galactic-potential integration.

**Status:** funnel + workflow (`panspermia.yml`) built, unit-tested offline, and
**first runner dispatch complete** (run 28609098955, 2026-07-02).

**First run (K2-18 anchor, 40 pc sphere, 10 Myr window):** Gaia DR3 resolved
K2-18 at 38.02 pc with space velocity UVW ≈ (−8.2, −14.8, −8.2) km/s (a
thin-disk-normal motion). **9,980** Gaia 6D stars in the surrounding distance
shell → **4,984** had a past closest approach → **15** within `d_min ≤ 2 pc`.
Headline geometry:
- **Closest approach 0.90 pc** — Gaia DR3 `3913239815437281536` (M dwarf, G=13.7,
  35.8 pc), ≈136 kyr ago — **but at v_rel 32 km/s** (a fast flyby).
- **Top transfer score** — Gaia DR3 `4358031335898505472` (d_min 1.13 pc, v_rel
  27 km/s, ≈1.35 Myr ago), a bright G=5.6 star at 9.9 pc.
- **Zero co-moving companions** (nothing within 5 pc sharing K2-18's velocity).

**Fast-interaction / transfer-regime analysis** (`panspermia-regime`, offline
post-process of `encounters_all.csv`; `results/panspermia/transfer_regime.csv`).
Fast encounters cannot capture gravitationally, but could in principle transfer by
*geometric interception* (sweeping through the donor's reservoir), which is
speed-independent — so this was tested explicitly. Two necessary conditions, both
required: the pass must cross the reservoir (`d_min < r_reservoir`) **and** be slow
enough to bind material there (`v_rel < v_esc(d_min)`). Result across reservoir
radii from 0.5 pc (max Oort) down to 50 AU (Kuiper): **0 of 4,984 past encounters
permit any passive transfer.** The closest pass (0.90 pc) was **551× too fast** to
capture even at that distance, and **nothing** came within even a 0.5 pc reservoir
to intercept. For a 0.36 M☉ donor the escape speed at the Oort edge is ~0.12 km/s,
while the *slowest* encounter in the whole sample is 2.97 km/s — so the local
RV-complete neighbourhood is closed to K2-18 panspermia by 2–3 orders of magnitude
in *both* speed and distance. (The fast tail itself is ordinary field/halo
kinematics: median 36 km/s, one 590 km/s halo interloper — not panspermia-related.)
The physics pins the *only* viable regime to an extremely close (<0.1 pc), nearly
co-moving (<0.2 km/s) pass — i.e. effectively a bound companion, of which the run
found zero. RV completeness for close-passing faint M dwarfs remains the one gap
where such a pass could still be hiding.

**Directed-travel (technological) reframing** (`panspermia-targets`,
`reachability.py` + `exohosts.py`). The passive channels assume life is cargo to
be *caught*; a **technological** disperser instead *chooses* a target, aims, and
decelerates — so relative velocity is irrelevant and reachability is trivial
(0.90 pc ≈ 3 ly ≈ 30 yr at 0.1c; every neighbour is a short hop). The filter then
collapses to **destination quality**, and the optimal launch window is each star's
closest approach (`t_enc`, min crossing distance `d_min`). Crucially, "habitable"
is judged from the *traveller's* biology: a K2-18-evolved organism seeks other
**hycean worlds** (Madhusudhan+2021) — sub-Neptunes (1.5–2.6 R⊕) with H₂/ocean
envelopes around **cool K/M dwarfs**, over a far wider insolation range than the
rocky HZ — not Earth-analogs. The destination prior is therefore hycean-centric
(peaks on M-dwarf colours like K2-18 itself); `--target classical` gives the
Earth-analog comparison. **Offline run:** 4,984 past-close neighbours, 4,742
main-sequence; the top destinations are all cool M-dwarf hosts (the hycean-host
class), reachable in 300–1,200 yr at 0.1c. The sharp discriminator — which of them
*already* host a known planet, and specifically a **hycean-candidate** sub-Neptune
— needs the NASA Exoplanet Archive cross-match: **`panspermia-targets.yml`
dispatched** (runner-side TAP). Outputs: `results/panspermia/reachable_targets.csv`
+ `targets_summary.json`.

**Cross-match result (run 4, `targets_summary.json`):** 1,483 Exoplanet-Archive
planets within 90 pc → **109 of K2-18's past-close neighbours are known planet
hosts, and 16 host a hycean-candidate sub-Neptune** (the destination class a
K2-18 organism would seek). Top hycean-analog destinations, by closest-approach
distance (all reachable in <1,300 yr at 0.1c, optimal window within the last
~1 Myr):
| Host | d_now | d_min | t_enc | cross@0.1c | planets |
|---|---|---|---|---|---|
| **LTT 3780** (M, bp_rp 2.68) | 22 pc | **7.28 pc** | −0.67 Myr | 237 yr | 2 |
| **K2-3** (M) | 44 pc | **7.47 pc** | −0.10 Myr | 244 yr | 3 |
| GJ 667 C (M) | 7.2 pc | 14.1 pc | −0.80 Myr | 460 yr | 5 |
| GJ 357, GJ 251, Wolf 1061, L 98-59 (5 pl)… | | | | | |
LTT 3780 and K2-3 stand out: **M-dwarf hosts like K2-18 itself**, the two closest
hycean-candidate approaches, both with known sub-Neptunes (K2-3 d is a
literature hycean candidate). These are the concrete, ranked answer to "which
nearby worlds would a K2-18 civilisation choose" — the search's first positive,
specific target list. *Next decisive step:* fold in encounter-time uncertainty
(the `panspermia-mc` Monte-Carlo) for LTT 3780 / K2-3 and check whether their
sub-Neptunes truly sit in the hycean HZ (radius + insolation) vs just passing the
size cut. The bare kinematics are a necessary, not sufficient, condition.

**Per-target deep dossier** (`panspermia-dossier`, `dossier.py`). With the field
down to two objects, stop doing statistics and interrogate *every archive a runner
can reach* for each, running every signature detector this repo has:
- **Gaia DR3 astrometry** → hidden-companion diagnostics (RUWE, excess-noise sig,
  IPD multi-peak, NSS) — an unseen massive companion / anomalous acceleration;
- **WISE** → W1-anchored IR colour excess (warm dust / Dyson-like waste heat);
- **ZTF g+r** → the `seti.dimming` dip / secular-fade / glint detectors
  (megastructure transits, slow enshrouding, specular glints);
- **Gaia XP** → a narrow, interior, bounded emission spike no smooth continuum
  reproduces (a laser line), with the XP-resolution guards from the spectra channel.
Now covers **six channels** on **three targets** (K2-18 itself + LTT 3780 + K2-3),
adding **NEOWISE mid-IR variability** and **TESS/K2 photometry** (lightkurve).
Contamination discipline is built in and *earned its keep*: the first run flagged
both destinations, but both were traps — a tiny (0.1–0.2 mas) astrometric excess
flagged on σ alone (fixed: require ≥1 mas amplitude), and a 75% single-band ZTF
"dip" on K2-3 (fixed: two-band achromatic confirmation, else `needs_vetting`). A
proper-motion fix (propagate the Gaia position to each survey epoch) recovered
WISE and NEOWISE, which had silently returned no data for these high-PM stars.

**Final result (run 3, all committed):** every target is **clean in every channel
that returned data** — no IR-excess (WISE W1–W4 ≤ 0.46), no mid-IR variability
(NEOWISE 274–289 epochs), no transit-shaped anomaly (TESS ~3.2–3.5k epochs), no
unseen massive companion (RUWE ≤ 1.24), no XP laser line. Coverage is tracked
honestly (verdict reads `clean_in_N_of_6_observed_channels`): K2-18 3/6 (no Gaia
astrometry row via the cone, no ZTF, no XP), LTT 3780 & K2-3 5/6. **TESS clean on
K2-3 corroborates that its single-band ZTF g-dip was an artefact.** Known gaps:
ZTF is partial (LTT 3780 saturates ZTF at r≈11; ZTF/IRSA flaky for the rest) but
TESS supersedes it; not covered = radio/SETI, high-res HARPS/ESPRESSO RV spectra,
X-ray. 20 offline tests. Net: the two destinations and the origin world show **no
technosignature** in any public archive reached — the honest state of the deep dive.

**Read:** every *passive* encounter is *fast* (v_rel 23–54 km/s) — the signature of random
field stars passing a normal thin-disk star, not a shared-origin group. No slow,
close bridge exists in the RV-complete local sample, and the transfer scores are
all ~1e-4 (dominated by the 1/v_rel² term). This is **not a null to write up** —
the *limiting factor is Gaia RV completeness*: most nearby M dwarfs lack a Gaia
radial velocity and are excluded, so a genuinely slow/close encounter could be
hiding among them. *Next decisive moves:* (1) supplement RVs for the RV-less
nearby M dwarfs (LAMOST/APOGEE/SDSS) to close the completeness gap that a slow
encounter would live in; (2) tighten the shortlist to the only regime that would
matter — `d_min < 0.3 pc AND v_rel < 5 km/s` — and Exoplanet-Archive cross-match
any survivor; (3) if a slow/close survivor appears, replace the linear
approximation with a Galactic-potential orbit integration to confirm it.

**Monte-Carlo encounter uncertainty** (`panspermia/uncertainty.py`,
`panspermia-mc`, `results/panspermia/recipient_candidates_mc.csv`, 3 offline
tests). The base shortlist gives point estimates; the rigorous treatment
(Bailer-Jones 2015/2018) resamples both K2-18 and each candidate from their Gaia
(parallax, pmra, pmdec, RV) covariances and reports the *distribution* of
`d_min`/`t_enc`/`v_rel`. Result (5,000 draws each): the geometry is **robust** —
**13 of 15** candidates are a *past* encounter in 100% of draws and stay within
2 pc in the majority; the closest, Gaia DR3 `3913239815437281536`, is
`d_min = 0.91 pc` with a tight 16–84% band of **0.90–0.95 pc** (136 kyr ago). So
the close passes are real, not astrometric flukes — but every median `v_rel` is
**23–54 km/s**, confirming with error bars that none is capturable (the flag is
*geometric* robustness, not transfer viability; the regime analysis above owns the
capture physics). The MC therefore hardens the null: the recipient list is a set
of well-measured *fast* flybys, exactly what the transfer-regime cut rejects.

## Channel state

| Channel | Searched so far | Surviving | Blocking issue / next action |
|---|---|---|---|
| Dimming (dips + secular) | 250,862 ZTF stars, 116 fields | 0 — top fader killed by NEOWISE reddening test; **19 `marginal_fade` assessed and set aside** (all 1.6–7.4% total fades, 18/19 not band-confirmed → optical slope ~0.004–0.015 mag/yr, where even a gray occulter gives only ~2σ in NEOWISE; ZTF systematics floor) | channel exhausted at the systematics floor — do not re-chase the marginal faders; new volume only helps if it reaches ≳0.1 mag band-confirmed fades |
| Specular glint | ran on 4 fields → 15 candidates, **all vetted to 0** | 0 | every candidate is a `chromatic_flare` (M-dwarf flare, bluer in g than r → not achromatic) or dusty; `glint_confirmed=False` for all 15. Achromaticity vet kills the channel. Single huge-brightening events are asteroid/cosmic-ray artifacts; multi-event ones are red-dwarf flares |
| Laser emission (SDSS-DR17) | 10,500+ spectra (latest committed run) | 112 triaged (was 118; 3 galaxies cut, incl. former #1) | per-exposure persistence check (repeat-visit path exhausted) |
| Astrometric dark companion (Gaia orbits) | 105,066 NSS orbits, ≤1 kpc | 0 novel (8 class-3 = BH1 + 7, but 7/8 already in Shahaf+2023; 1 borderline-absent is the weakest solution) | reproduction of the published AMRF catalogue — change the question |
| Laser absorption (DESI-DR1) | 6,500+ spectra (latest committed run) | 55 triaged | same; hot-star continua only (line-forest stars skipped by design) |
| WD IR excess | 7,716 clean WDs → 23 multi-axis → blend+sublimation test | 0 technosignature (3 WISE blends, 7 unresolved stellar companions, 13 ordinary τ<0.08 debris disks) | channel resolved; τ=0.6 standout is a too-hot-for-dust stellar companion. Next volume only helps if it reaches a τ→1 excess with T_dust *below* sublimation |
| Panspermia (K2-18 close encounters) | first run: 9,980 Gaia 6D stars, 4,984 past approaches, 15 within d_min≤2 pc | 0 slow/close bridge (all v_rel 23–54 km/s; closest 0.90 pc but at 32 km/s; 0 co-movers) | **RV completeness is the gap** — supplement RVs for RV-less nearby M dwarfs, then re-cut to d_min<0.3 pc & v_rel<5 km/s; Exoplanet-Archive cross-match any survivor |
| Gaia XP anomalies | RA283/Dec−3 dense field: 8,863 sources, reliable; narrow-feature shortlist examined | 0 credible | **channel bounded — see ledger.** Broad "anomalies" = reddened-M-dwarf molecular bands (degenerate with a Dyson SED); "narrow" ones = band-edge reconstruction artifacts + sub-resolution wiggles (XP LSF ≈5+ samples can't resolve a laser line). Guards added (width/interior/bounded). A clean low-extinction field could still test the *broad*-SED Dyson signature, but it is degenerate with reddening |
| Galactic long-baseline encounters | LHS 1140 + K2-18; 149,979 RV-complete Gaia stars each integrated back 300 Myr in the MW potential | 0 bio + 0 techno among encounter neighbours | **clean.** LHS 1140: 34 passes <3 pc (0 planet hosts, 0 companion flags); K2-18: 16 (same). All datable passes are recent (<~20 Myr) — phase mixing erases 100–300 Myr timing, so the honest recoverability horizon is ~tens of Myr. Closest: a G=7.4 star 0.086 pc from LHS 1140 2.9 Myr ago (single star, not a signature) |
| LHS 1140 system deep-dive | star + b/c + 38 neighbours ≤15 pc; 6 archives (Gaia astrometry+XP, WISE/NEOWISE, ZTF, TESS 3.5k epochs); 533 MAST obs inventoried; **biosignature detectability budget** | 0 technosignature; **0 detectable biosignature** | **clean + bio answered.** Star: all 6 channels clean; lone RUWE=1.53 is marginal binarity, not techno. Neighbours: raw 15/38 IR-excess all = WISE W4/blend systematics → 4 survive → faint/blue/ordinary-debris. **Bio:** under b's expected high-μ (N₂) atmosphere (H≈3.6 km) every biosignature feature is ~few ppm → CH₄ ~25 / O₃ ~67 transits vs ~2–4 observed → `NOT_DETECTABLE_WITH_CURRENT_DATA`; reachable only for a disfavoured cleared H₂ envelope |
| **OSSUARY** (warm dust where none can form) | built; first run dispatched (run 30203264572). Gaia DR3 GSP-Spec/GSP-Phot [Fe/H] < −1 **plus** a pure halo-kinematic track, × the Gaia archive's AllWISE + 2MASS mirrors via the official PM-aware cross-match; expected ~10⁵–10⁶ stars | pending first run | **Novelty confirmed:** two independent full-corpus arXiv queries for halo-star IR excess return **0**; `"metal-poor" AND "debris disk"` returns 3 papers, one of them a 7-star study at [Fe/H] ≲ −5 with inverted motivation (Venn+2014). Hephaistos I hard-codes Z = 0.012–0.018 = thin disc. Lacki 2025 (arXiv:2504.21151) *predicts* halo + low-metallicity hosts and ran no search. Competitor forming: Kenyon, Bromley & Najita 2026 have the catalogue and *plan* the analysis |
| **CENOTAPH** (cold Dyson, T<100 K) | built + dispatched (run 30203250183); target Gaia DR3 GSP-Spec dwarfs, ~5.6e6 with Teff/logg/[M/H]/[α/Fe] from one pipeline. **First run committed** (`results/cenotaph/summary.json`, index update 2026-09-01): 1,060,732 rows acquired on a `PARTIAL_SAMPLE` (3 of 88 parallax shells timed out), 674,198 fitted, 3,642 leg-1 grey-significant, 3,620 with no mid-IR excess, 486 vet-core | verdict `closure_but_crowded_beam`: of 486 decidable, 481 have no far-IR data, 2 close but in a crowded beam, 1 over-closes, 2 anisotropic/non-thermal — **0 clean-beam closures** | **new channel.** Three-leg energy-conservation test: grey attenuation (A_V fitted jointly, not assumed) + NO mid-IR excess + far-IR recovery of the intercepted f·L in AKARI/FIS + IRAS. Closure ratio ρ=f_IR/f_dim separates an isotropic occulter (ρ≈1) from an edge-on disk (ρ≪1). Measured floor f≳0.15–0.29 vs Zackrisson+2018's f_cov>0.75. Next: read `results/cenotaph/summary.json`, check the ±3σ tail asymmetry before believing any count |
| **SHROUD** (enshrouded, not destroyed; S33) | built; first run dispatched. The **never-analysed catalogue by-product** of Solano, Villarroel & Rodrigo 2022 (MNRAS 515, 1380): `vanish-neowise` = **171 753** POSS-I sources absent in the modern optical but detected in the infrared, plus `vanish-possi` = **5 399** with no counterpart at all, as the control | pending first run | **New channel.** Endpoints verified: `http://svocats.cab.inta-csic.es/vanish-{neowise,possi}/` (quoted in Watters+2026 Table 1 and the jannefi/vasco README). Solano+2022 is **not** in VizieR (runner fetch returns "Table or Catalog not found") — SVO is the only machine-readable route. Measurement: eta = F_IR(now)/[F_bol(POSS-I) − F_bol(modern)], a **pure flux ratio so distance cancels** — no parallax needed. Forés-Toribio & Kochanek 2026's progenitor/remnant ratio *is* eta, applied at ≥10 (merger remnant) and ≤0.3 (genuine disappearance). Scoped **strictly to the archival crossmatch** — no VASCO transient/Earth-shadow/nuclear-test analysis (docs/shroud.md §1) |
| **EMBER** (waste heat that switched off; S1) | built; **acquisition repaired and re-run** (probe run 30209647320, search run 30210104587). The only three all-sky surveys carrying 12–25 µm photometry — IRAS (1983), AKARI/IRC (2006–07), WISE cryogenic (2010) — cross-matched via Gaia DR3 with PM propagated to each survey epoch. **105,694 rows acquired per RA shard** after the repair (run 30203763934 had reported `acquired: 0`) | first run reached no data for three unrelated reasons that all presented as one zero; all three fixed and the distinction made structural | **New channel.** *Novelty:* the entire Dyson/waste-heat lineage is single-epoch by word count — "epoch" in a photometric sense = **0** in Carrigan 2009, Ĝ I/II/III and Hephaistos I; the 2026 flagship review (2605.21093) has `turn off`/`switched off`/`cessation`/`multi-epoch`/`AKARI`/`NEOWISE` all = 0; Suazo+2024 *explicitly cut* variable stars, discarding a changed megastructure by construction. **Antecedents that must be cited, not ignored:** Kim+2015 (1501.05721) ran the identical IRAS+AKARI+WISE comparison **upward** (4 sources all-sky); Sedgwick & Serjeant 2022 (2207.09985) built the IRAS×AKARI 23.4-yr cross-match for *proper motion*; Melis+2023 ran it downward but targeted at R CrB. **Audit reversed the brief, and then the real SVO curves reversed the audit:** AKARI 9 µm→W3 has a transfer spread of **8.25×** over 150–1500 K (worst-conditioned pair, demoted); **I25→W4 is spread 1.03 — near-null, <1% bandpass systematic, 12× wider unsaturated window — and is now the PRIMARY pair**, with I12→W3 (1.71) corroborating. NEOWISE **rejected as an epoch** (W1/W2 only; cannot see 100–300 K dust) and reused as the post-drop flatness test. **Measured acquisition facts (probe 30209647320):** IRAS PSC/FSC positions are **B1950** (`RA1950`/`DE1950`) and must be precessed; the anonymous ESA Gaia TAP upload returns **HTTP 500 even at 200 rows** so CDS X-Match is the primary Gaia route; AllWISE has 2.3M rows per 1° RA slice so only X-Match is viable. Next: read `results/ember/summary.json` funnel counts and the rising-tail asymmetry in `null_calibration.json` before believing any count |
| **TAILINGS** (the sparse chemical anomaly; S12/S15/S22) | built; first dispatch hit stale VizieR catalogue numbers and returned `NO_DATA_REACHED` honestly, fixed by runtime TAP_SCHEMA table discovery + schema scoring; re-dispatched. Target GALAH DR4 + APOGEE DR17 cool dwarfs (Teff<6000 K, logg>4.0), ~20–30 elements each | pending first archive run | **New channel, and the discriminant is inverted.** Natural abundance space is low-dimensional (~8–10 independent axes: Ting+2012, Price-Jones & Bovy 2018, Patil+2022) and every natural process moves an element FAMILY; refining moves ONE. So a **dense** anomaly is a REJECTION here — the opposite polarity to every existing abundance-outlier statistic, all of which are global distances (PCA/EMPCA/t-SNE/autoencoder/k-means) built to *cluster* stars, and all of which are maximised by dense anomalies. Direct arXiv queries for `"anomaly detection" AND APOGEE`, `… AND GALAH`, `"outlier detection" AND "stellar abundances"`, `"abundance anomaly" AND "single element"` all return **0**. **The real competitor, read in full: Huang, Tao & Zhang 2026 (arXiv:2605.29811)** — executed, meteorite-calibrated Bayesian test for refined material, but in **polluted white dwarfs** (7.7≤logg≤8.3; "GALAH" and "APOGEE" appear 0 times), on 697 literature records/≥397 objects, against a **fixed dense siderophile template**, and its power *rises* with element count — "typically requires ≳5 detected elements for decisive support". The two searches rank the same objects in nearly opposite order. Their strongest criticism (a 1–2-element anomaly can carry a big Bayes factor while information-starved) is answered by `n_quiet`: their cases are records where the other elements were never MEASURED; here 20–30 are measured **and quiet within 2σ**. Whitmire & Wright 1980's 56-citation tree still contains **no executed survey**; note this channel deliberately inverts their A5–F2 host choice — that band is where diffusion/levitation manufacture single-element anomalies naturally (hence 60 years of contested Przybylski claims), so trading amplitude for a clean null is the right trade. **Hardening after four adversarial sweeps:** (1) **metallicity leak closed** — convective protection depends on [Fe/H] too; a ~0.85 M☉ metal-poor turnoff star has M_env < 1e-7 M☉, four decades thinner than a solar-metallicity dwarf, and passes a Teff/logg cut (Matrozis 1605.02791). Floor at **[Fe/H] ≥ −1.0** in the ADQL, config and funnel; also **Teff ≥ 4000 K** because GALAH's own notes put cool-dwarf systematics at up to **0.5 dex below 4600 K**. (Cross-channel: OSSUARY *selects* what this excludes.) (2) **Search in [X/H], never [X/Fe]** — an error in the star's own [Fe/H] otherwise smears a sparse anomaly across every element (Weinberg's measurement aberration); stated cost is that an **iron-only** anomaly is now invisible by construction. (3) **Karinkuzhi gate** — 13 'Sr-only' + 2 'Ba-only' LAMOST (R~1,800) candidates re-observed at R~86,000 **all dissolved into dense barium stars**, so below R=20,000 sparsity is presumed to be blending and the candidate carries `needs_high_resolution_confirmation`; plus a curve-of-growth check (a saturated core is not a measure of abundance). (4) **Instrumental covariate veto** — Weinberg traced high-Ca stars to one RV+fibre bad-pixel combination and a *population* of low-K stars to a −70 km/s telluric coincidence, so flag rates are binned against RV and fibre. (5) **Validation target**: Griffith 2110.06240 found, incidentally, **15 stars with 0.3–0.6 dex Na enhancement and normal O–Ni** — the pipeline must recover them. Contamination baseline ~60% junk. **CORPUS-INTEGRITY WARNING (repo-wide lesson):** the first lit fetch reported 107/107 successful and **12 of 24 hardcoded arXiv IDs had resolved to unrelated papers** — Plenoxels for APOGEE DR17, an LHC dark-matter paper for Vick, neutron-star precession for Richer — all fetching cleanly. **A successful fetch is no evidence the paper is the right one.** See `results/tailingslit/INTEGRITY.md`; offending files deleted, harness now resolves by title search with title-token verification. **Two dispatch failures, both recorded because the honest verdict disguised each one:** run 30203627605 hit stale VizieR catalogue numbers (`III/283/allstar` etc. all 'table not found'), fixed by runtime `TAP_SCHEMA` discovery + schema scoring; run 30204487245 then reached discovery, probed the right tables and STILL returned `NO_DATA_REACHED` — because `TAP_SCHEMA.tables.table_name` comes back **already double-quoted**, so `FROM ""III/283/allstar""` was rejected by everything. **LEDGER RULE (general): a channel whose null verdict is indistinguishable from its bug verdict will mislead its own author** — the distinguishing test belongs in the suite, not in a log. Third dispatch queued. Next: read `results/tailings/contrast_*.csv` — if the SPARSE fraction is flat with z_max the sample is systematics, not a population |
| **RUST** (unmaintained decay; S9) | built; **novelty run 30203976309** (complete, 47/47 fetches) + **search run 30204427240** (8 high-galactic-latitude ZTF fields, g+r paired). Statistic: bias-corrected **season scatter** regressed on calendar time — the **second moment**, not the first. *(Run 1's sweep was cancelled and re-dispatched: it was executing pre-fix code, before the per-season line detrend closed the accelerating-fade leak — see below. Its novelty job had already completed and its evidence is kept.)* | pending first archive run | **New channel, and it is NOT the `dimming` channel again.** `dimming/secular.py` fits a weighted line to season **medians** (brightness, first moment) and is exhausted at the ZTF systematics floor; RUST regresses per-season robust **scatter**. The distinction is structural, not rhetorical: a shared zeropoint/reference drift moves every star's median together — the false fade that killed `dimming` — and leaves the *within-season scatter* untouched. But the exchange is not free, and RUST inherits two systematics `dimming` never had. **(a) Cadence bias.** `1.4826×MAD` recovers only ~66% of σ at N=3 and ~90% at N=8; epochs-per-season is set by survey cadence and cadence *trends with calendar time* (ZTF public went 3-day → 2-day in 2020). Uncorrected, this channel measures ZTF's operations calendar. Handled in five layers: per-season null computed with **that season's own N and its own per-epoch error vector** (MC bias table b(N) + heteroscedastic mixture-MAD scale); excess variance subtracted not divided, **including the second-order `E[s²]=(bσ)²(1+u²)` term** — a 17% offset at N=8 vs 2% at N=70, i.e. a pure function of cadence, hence of time; a **distribution-free** exact permutation p-value over all n! season orderings as the primary gate (Lacki gives a cascade *timescale*, not a light-curve shape, so a linear model would assume the result); per-CCD ensemble common mode in the second moment; and survivor-only exact-MC + **equal-N subsampling** cross-checks that share no machinery with the first layer. **(b) `magerr` mis-calibration drifting with time** — killed by the ensemble κ_s = median(σ_obs²/σ_null²) over the field. Measured offline: **0/600** constant stars flag across five cadence histories (rising 8→70, falling 70→8, ZTF-style jump, erratic, doubling), 0/2000 over a wider sweep; ~57% recovery at 36 mmag terminal amplitude, ~80% at 60 mmag. **Two-band g/r coincidence is enforced at SCORING, not follow-up** — a one-band source is never scored — and the g/r amplitude-growth *ratio* is the physics discriminant (grey occulter ≈1.00 / extinction law ≈1.42 / flare-accretion >1.70). **NEOWISE logic is INVERTED relative to `dimming`:** a mid-IR brightening *corroborates* dust production from a cascade instead of killing the candidate; its absence is informative but not fatal (W1/W2 only probe >600–850 K). **Novelty independently re-verified on the runner:** `scripts/rustlit_fetch.py` ran 24 targeted arXiv queries with the four decoy classes encoded as explicit regexes; over **215 abstracts** it found **1 hit, decoy-tagged `with_timescale` (red noise), and 0 decoy-free hits** (`results/rustlit/concept_scan.json`). The three counterweight/antecedent abstracts are now quoted from fetched text, not memory: McInnes 2026 — a ring-supported stellar engine "can in principle be **passively stable**" and dense-cloud bubbles "**passively self-stabilizing**"; Wright 2020 — monolithic spheres "**dynamically unstable** under gravity and radiation pressure, and **mechanically unstable to buckling**"; Petz & Kochanek 2025 — selection is on "brightness changes larger than ~0.03 mag/year", i.e. a mean-flux slope. **One novelty leg failed and is recorded as failed:** the OpenAlex citation tree resolved to arXiv preprint stubs reporting `cited_by_count=0` for all four targets *including Wright 2020*, so it is uninformative and carries no weight; the script now also searches by title so a re-dispatch fixes it. **A sixth systematic layer was added after the first dispatch:** a fitted **line** is removed per season, not just the median — a star fading at an *accelerating* rate drifts further within each successive season, which reads as a rising second moment produced entirely by a first-moment phenomenon, i.e. the `dimming` channel's own population leaking in (**46/60** false positives at 0.72 mag total fade; **0/60** with the line removed). The null table switches to its line-detrended variant to match, since a line fit costs an N-dependent amount (b(8): 0.900→0.811; b(80): 0.990→0.984). Honest two-band completeness: 34% at a 36 mmag terminal amplitude, ~65% at 60–90 mmag; the ~80% per-band plateau is the chi2-inflated linear gate refusing an accelerating rise, not noise. *Novelty:* a regex concept scan over 3,578 fetched abstracts for "amplitude/scatter increasing with **calendar time**" returns **0**; every apparent hit is a different statistic — Polaris (a single named star), T Tauri in WASP (amplitude vs **timescale**, red noise), YSOs in W51 (vs **evolutionary stage**), NLSy1 (vs **wavelength**). Nearest machinery is Petz & Kochanek 2025 (2501.14058), 9,361,613 ASAS-SN sources at >0.03 mag/yr — a **mean-flux slope**; the second moment is untouched. Bonus seam: Hephaistos II's `G_var > 2` cut explicitly discards "potential Dyson swarms with very large absorbing elements", i.e. exactly what this channel selects on. Sensitivity is stated *against* the counterweight: **not** sensitive to McInnes 2026 (2603.00203) passively-stable ring engines / dense-cloud bubbles, which need no upkeep, nor to Wright 2020 (2006.16734) monolithic spheres, whose failure is catastrophic rather than a decade-long ramp. Next: read `results/rust/summary.json` and `results/rustlit/concept_scan.json`; for any survivor demand the ASAS-SN cross-survey result — **red noise, not white, is the least-controlled residual** and the quoted false-positive rates are against Gaussian nulls |
| **KNELL** (the clock that stopped; S32) | built 2026-07-26; offline suite green (**32 tests**, all passing); dispatched (see run id in the channel log). **The observable is a PERIOD that ceased** — a clock that stops is the cleanest "the mechanism ended" signature, and it is the one necrosignature in this program that **dust cannot fake**: obscuration changes a signal's *amplitude*, not the *existence* of a period, so the line-of-sight-extinction confounder that dominates `dimming`, EMBER and CENOTAPH is absent by construction. **What replaces it is survey-dependent detectability, and that is the whole channel.** A variable "ceases" whenever the later data have worse cadence, a shorter seasonal window, larger errors or a different alias comb — so the primary search is **intra-survey** (ZTF's own early seasons against its own late seasons, in g *and* r, where passband/pipeline/calibration are constant by construction) and **every claimed non-detection is normalised by the injection-measured detection efficiency for that star's own period and amplitude in that block's own sampling and noise** (`src/seti/knell/efficiency.py`, the load-bearing module). Efficiency is measured by injecting a random-phase sinusoid **into the post-transition block's own observed magnitudes** — by the cessation hypothesis that block holds no signal, so it supplies the real sampling, real error distribution, real correlated systematics and real outliers for free; nothing about the noise is modelled, so nothing about the noise can be modelled wrongly — and scoring it with the **identical** detector (batched GLS, verified against `astropy.timeseries.LombScargle` to 1e-8, against the block's own **permutation** threshold). **Measured: the uncorrected statistic flags 22 of 24 constant-signal stars whose only change is a degrading cadence (80,80,70 → 18,15,15 epochs/season); the efficiency gate flags 0 of 24.** A 92% false-positive rate turned off — worse than the sibling RUST channel's 46/60, and the direct justification for the module. Degrading *errors* at fixed cadence: 0/10. p-values are Clopper-Pearson **upper bounds** and are printed as inequalities when pinned at the injection-resolution floor (`<= 4.97e-05 (injection-resolution limited)`); a test asserts that escalating trials tightens the bound. Confounders closed with named mechanisms: **mode switching** (blind per-block detector fires on *any* frequency, **plus** an excess-variance test, because in a mode switch the total power is *conserved* and merely moves); **Blazhko** (>=2 post blocks, >=500 d post baseline, pre-modulation index); **third-body nodal precession / SS Lac** (pre-cessation amplitude decline + Gaia RUWE/`non_single_star`); **fade** (mean flux must be unchanged, an independent guard from the efficiency gate); CV disc states, AGN red noise, YSOs, blending, both ZTF photometric walls. **Novelty is narrowed honestly.** Not "the first search for stars that stopped varying" — falsifiable inside EBs by Jurysek 2018 (1709.08087) and Graczyk 2011. The defensible claim: **cessation events have only ever been found one at a time and serendipitously, and no survey has ever measured the rate at fixed detection sensitivity.** Kurtz et al. 2025 (2412.04840, MNRAS 536, 2103) claim a *first ever* for one star ceasing pulsation; OGLE found "objects ceasing pulsations" **by accident inside a catalogue paper** (1601.02020); Ansari+2023 report **9,881 of 58,200 GCVS variables with no Gaia variability flag**, attributed entirely to detection capability — that is this channel's null hypothesis, quantified. Theoretical anchor: **Kipping & Teachey 2016 (1603.08928)**, the only paper proposing *deliberate erasure* of a periodic signal as a technosignature, which runs no search. **Substrate decision (the sharpest result of the literature sweep): eclipsing binaries are EXCLUDED from the anomaly sample** and used only as a calibration set — third-body nodal precession is a named, modelled, actively hunted mechanism whose rate (~28 lost systems in one Kepler->TESS revisit) would swamp anything exotic, and it comes with predicted return dates. The novelty lives in **pulsators**. **Two citation errors in the brief were caught and corrected:** arXiv:astro-ph/9805019 is **Tomasella & Munari 1998**, not Torres & Stefanik (whose SS Lac paper is AJ 119, 1914, 2000, with **no arXiv ID found** — none is asserted); and arXiv:1807.03448 is the **WISE** periodic-variable catalogue, not ZTF's (that is **2005.08662**, ApJS 249, 18). All verification was WebSearch-only (WebFetch/curl 403-blocked on *every* host incl. example.com), a control test proved the backend's phrase matching is not strict, so **no quotation is certified verbatim** and re-verification is a runner-side job. **A provenance bug was found and fixed in the process:** the inherited bulk ZTF fetcher catches its own HTTP errors and returns an empty dict, making a proxy 403 **indistinguishable from an empty sky box** — a whole run of refused queries would have reported "zero rows" and read as a search. A separate one-shot service probe now restores the distinction, and the sweep emits three different verdicts (`NO_DATA_REACHED` / `ARCHIVE_RETURNED_ZERO_SOURCES` / `NO_TESTABLE_LIGHT_CURVES`) where before there was one. The vet job **fails the workflow** if a verdict with 0 testable stars is anything other than a NO_DATA class. Next: read `results/knell/summary.json` and the per-field `acquisition_log.json`; for any survivor, check the **HR-diagram position first** — the closest analogous search (Jarvinen & Strassmeier 2025, 2504.19670) lost **11 of its 13** candidates to Gaia parallaxes showing they were evolved stars — then confront the `spot_cycle_plausible` flag, which is the one benign interpretation this channel does **not** close | dispatched, awaiting first archive run | **new channel; see `docs/knell.md`** |
| **VIGIL** (waste heat with a duty cycle; S4) | built 2026-07-26; offline suite green (39 tests). **Novelty + probe run 30215516935** (novelty and probe jobs COMPLETE; its sweep shards were cancelled and superseded because they were executing the pre-fix per-star acquisition — see throughput below). **Search run 30216181263** (8 fields near the north ecliptic pole, where the NEOWISE scan pattern piles up so per-star visit counts are highest, all at \|b\| = 23-37 deg so cirrus and crowding are suppressed at selection). **Novelty verdict UNOCCUPIED, established on the runner**: 39/39 fetches OK, **303 abstracts scanned** over 17 targeted queries, **0 genuine prior-art hits** — nothing matched the conjunction (mid-IR/WISE/NEOWISE) AND (variability) AND (optically constant) AND (technosignature/Dyson/SETI/megastructure) free of the four decoy classes (YSO, extreme debris disk, AGN/blazar, optical-transit megastructure); the only two abstracts scoring 3-4 concept groups were both AGN-tagged. **All four decisive arXiv IDs verified BY TITLE**, not by number: 2511.22071 = 'A Catalogue of Mid-infrared Variable Sources from unTimely' (**8,256,042 W1 and 7,147,661 W2 variables** — the catalogue is real and the size claim holds), 2103.00568 = the ALLWISE warm-EDD sample, 2006.16734 = 'Dyson Spheres', 2403.18941 = the five-million-star mid-IR excess search. The Hephaistos quote is now **verbatim from fetched full text**: Hephaistos II's check "rejects potential Dyson swarms with very large absorbing elements since these in principle could generate detectable variations in the photometry of the host star", and it assumes swarms "with no pieces large enough to cause stellar variability". **Data reachability, stated exactly:** NEOWISE per-epoch W1/W2 is reachable (`status: OK`, 32 rows against a `COUNT(*)` of 32, 27 surviving frame-quality cleaning). The **unTimely variable catalogue is NOT reachable as a queryable table** — and the first probe could not say so honestly, because VizieR returned an ADQL *syntax error* (its parser rejects `LOWER(col)` in WHERE, which IRSA and NOIRLab both accept) while the other two ran and returned zero rows: one transport failure plus two absences is not an absence claim. Query rewritten to spell out case variants, verdict vocabulary now separates ALL_TAP_ROUTES_FAILED / NOT_FOUND_BUT_SEARCH_INCOMPLETE / CATALOGUE_NOT_FOUND_ON_ANY_TAP_ROUTE. Corroborating evidence that this is a real absence and not a bad query: **35,230 characters of the paper's text contain no data URL, no Zenodo deposit, no VizieR identifier and no Data Lab table** — the catalogue has no machine-readable release yet. The parent unTimely Catalog does (catalog.unwise.me, github.com/fkiwy/unTimely). The channel does not depend on it: per-epoch NEOWISE is required regardless, since a variability *catalogue* carries detection flags, not the epochs the modulation index / morphology / colour statistics need; `preselect_from_untimely` uses the table when found and records `applied: False` with the reason when not. **Throughput fix from run 1:** a single-star NEOWISE cone measured **~92 s**, which caps a 400-star field at a few dozen stars — so the sweep now issues ONE field-wide cone and assigns exposures to stars locally by KD-tree, propagating each star's PM to mission mid-epoch and widening its own match radius by half its mission sweep (tested: a high-PM star is recovered with propagation and lost without). Per-star cones remain the fallback and a field-query failure lands in the ledger. **Second run-1 lesson:** the probe's NEOWISE call at a bare coordinate binned to **zero usable visits** — 32 sporadic detections of a marginal source, never 3+ inside one visit, so the within-visit noise calibration had nothing to work with. Returning `None` was correct, but a probe measuring empty sky tests nothing; it now resolves a real Gaia star first and reports median exposures-per-visit, which IS the per-star sensitivity and varies across the sky with the scan pattern. | pending first sweep results | The observable is a star **variable in the mid-infrared while constant in the optical** — steady interception, unsteady re-emission — and, in the extinction reading, that variability *ceasing*. **The novelty seam is a cut somebody else made:** Hephaistos II's `G_var > 2` explicitly *rejects* variable stars because a swarm with very large absorbing elements "could generate detectable variations in the photometry of the host star" — that is a cut against **optical** variability from occultation, while this channel selects on **mid-IR** variability from modulated re-emission, which no published search has ever used. **The confounder is the entire channel.** Extreme debris disks have exactly this phenomenology: strongly mid-IR variable with flat optical light curves (Moor et al. 2021, arXiv:2103.00568, warm EDDs from AllWISE). So **optical constancy buys nothing against the dominant contaminant** — it kills YSOs, dippers and AGN and nothing else — and a channel that stops at "mid-IR variable, optically constant" has produced an EDD catalogue and called it a search. **Primary discriminator:** mid-IR variability at **low** fractional excess, because an EDD is "extreme" precisely because `τ ~ 1e-2`. The arithmetic that makes it work: `F_exc/F_phot = τ·R` with `R = [B_ν(T_d)/T_d⁴]/[B_ν(T_*)/T_*⁴]`, so at 600 K around a 5000 K star `R(W2) = 23` — τ=1.5e-3 is a **3.5% band excess** (a ~37 mmag switch, well inside NEOWISE visit-mean precision) while an EDD at τ=1e-2 is **23%**. A factor ~5 separation against a 2–3% photosphere systematic: that is why the cut works, and it is applied to the **2σ upper limit** because "low" has to be a bound, not a point estimate that scattered low. **Sharpened form (the one real improvement on the brief):** `R` swings by >13× across the plausible dust-temperature range, and it *cancels* if you write the maximum amplitude an excess can produce against the **band** excess `f` rather than against τ — `A_max = 2f/(1+f)`, so the **modulation index** `m = A_obs(1+f)/(2f)` is free of T_dust, T_*, distance and luminosity. It measures *what fraction of the excess is actually switching*: an EDD is a perturbation on a large steady excess (`m ≪ 1`), a load-following radiator switches all of it (`m → 1`), and `m > 1` is a **falsifier** — the variability cannot be the excess (blend, bad epoch, underestimated photosphere). `m` is tightly constrained exactly where it must be (large, well-measured excesses = the confounder) and honestly abstains when the excess is marginal. **Two shape backstops:** decay-vs-duty-cycle morphology (weighted trend R², Kendall τ, likelihood two-state BIC split, squareness, transition rate, Lomb–Scargle, burst skew) — a smooth secular decay is a collisional cascade and is rejected **even at low excess**, independently tested; and colour-temperature stability of the *varying component* (a cascade changes amount **and** temperature as grains spread and cool; a radiator changes amount at fixed temperature). **Estimator:** NEOWISE visit structure is a free noise calibrator — ~10–20 exposures inside a 1-day visit measure the true per-exposure noise per star, while visit-to-visit scatter measures the 6-month variability, so the optimistic `w?sigmpro` values are **rescaled per star** by the within-visit scatter (an optimistic error is how a variability search manufactures candidates). Primary statistic is the **unbiased normalised excess variance** with the Vaughan et al. 2003 uncertainty carrying the exact N dependence — necessary because exposures-per-visit and visits-per-star both trend with ecliptic latitude, so an uncorrected version maps the NEOWISE scan pattern; plus equal-N re-measurement and a per-field ensemble common mode. **Repository gap closed:** there was **no PM-propagated `neowiser_p1bs_psd` fetcher anywhere** — all three existing single-exposure callers query at the Gaia epoch, and a 200 mas/yr star drifts ~2.1″ across the 2014–2024 mission, comparable to the cone radius. `vigil.acquire.fetch_neowise_epochs` propagates to mid-epoch 2019.0 and widens by the sweep. **Hard instrumental bound, stated not assumed away:** NEOWISE/CatWISE/unWISE are **W1/W2 only** (Wien peaks 852 K, 630 K), so VIGIL probes **hot** material and is structurally blind to the 100–300 K regime where most Dyson models sit; W3/W4 depth has been frozen since the 2010 cryogenic mission and W3/W4 are used here **only to reject** (a W4-only signal is cirrus). AGN are adjudicated on **astrometry, not colour**, because a ~350 K shroud has W1−W2 = 3.2 and sits inside the Stern/Assef box. Next: read `results/vigil/probe.json` (is the unTimely variable catalogue reachable, and what is it called), then `results/vigillit/summary.json` — and note the novelty script verifies every decisive arXiv ID **by title**, so a failed verification of 2511.22071 must be read as "the catalogue may not exist as claimed", not ignored |
| **DERELICT** (dead lightsails; S19) | built; **census extended and re-run** (run 30209538685; prior data run 30204805880). JPL SBDB fits a radial non-gravitational coefficient A1 for every small body with enough astrometry; `A1 → β → area-to-mass` is a two-line conversion and a thin film sits 4–5 orders of magnitude above any natural body of the same size. **22 asteroids have a fitted A1** (`sb-cdata={"AND":["A1\|DF"]}`); the **comet control returns 272** | the target set is small but **not empty**, and the control proves the query fires; funnel: a1_significant 11, screen1_a1_only 4, strict 0, negative-A1 1, verdict `ALL_SURVIVORS_EXPLAINED` | **The first run's `NO_DATA_REACHED` was a typo, not a sky.** `sigma_A1` is not a valid SBDB field; one bad column name 400'd all four constraint strategies, so "our query has a typo" presented as "the database contains no such object". Fixed by self-healing field pruning; sigmas now come from `sbdb.api`'s `orbit.model_pars[].sigma`, which is authoritative anyway. *Novelty:* Bialy & Loeb 2018 ran this exact inference **for 1I/'Oumuamua alone**; JPL/MPC run it **reactively and per-object** to unmask human hardware (2020 SO, J002E3, WT1190F); the dark-comet catalogues (Seligman+2023/2024) selected the **opposite** population — large *non-radial* acceleration, which excludes radiation pressure. The claim is the *method*: a systematic catalogue-scale selection of the A1-only complement, normalised by size into R = AMR_implied/AMR_natural. **Live novelty risk:** Loeb & Cloete 2025 (2503.03552) already argue one dark comet is artificial, so "a dark comet might be artificial" is published and is not claimed here. **Empirical systematic floor:** 4179 Toutatis shows a 3.4σ *sunward* A1 — which radiation pressure cannot produce — giving \|R\| ≈ 5 against a flag threshold of R = 10. 'Oumuamua recovers R ≈ 1.1×10⁵ (AMR 1.23 m²/kg) as the positive control. Next: `results/derelict/completeness.json` — whether the A1\|DF constraint really returns the whole A1 population — then `dark_comets.csv` and `high_albedo.csv` |

## Known systematics ledger (do not re-derive)

* **AllWISE infrared excesses are ~92% false positives.** Silverberg et al. 2018:
  at most **7.9 % ± 0.2 %** of AllWISE-selected excesses are good disk candidates;
  the McDonald et al. and Marton et al. searches exceed **70 %** false positives;
  **all 13** Theissen & West candidates with W4 S/N > 3 are spurious. Any new
  excess funnel must report *per-stage removals*, not just a final count.
* **Use 5σ, not 3σ, for an infrared excess.** Huang, Liu, Wyatt & Kennedy 2025
  (arXiv:2505.07602) searched the 10 pc sample (339 stars) for W3 excess at 3σ,
  got 5 candidates, and found **all five spurious**; detection rate 0/339.
* **W3/W4 depth is frozen at the 2010 cryogenic mission.** NEOWISE-R, CatWISE2020
  and the deep unWISE coadds are **W1/W2 only**. Wien peaks: W1 → 852 K, W2 →
  630 K, W3 → 241 K, W4 → 132 K. So below ~200 K the *only* route is W4, the
  shallowest and most confusion-limited band — there is no deeper 12/22 µm
  measurement to be had, and a warm-dust claim must lean on **W3 + W1−W2**.
* **λ Bootis stars are the metal-poor IR-excess trap.** A/early-F stars whose
  *surface* is metal-depleted by accreting gas-depleted ISM; Murphy et al. 2020
  find **21 of 34 have infrared excesses**, and some were previously catalogued
  as blue horizontal branch stars. A T_eff ceiling (~6500 K) removes them.
* **Globular-cluster sightlines must be vetoed, not vetted.** Boyer et al. 2010
  (arXiv:1002.1348) showed a *published* RGB-wide infrared excess across 47 Tuc —
  a metal-poor, old population — was entirely stellar blending and imaging
  artefacts, from the same archival imagery as the original claim.
* **Metal-poor circumstellar dust is featureless** (metallic iron, not silicate;
  McDonald et al. 2011, ω Cen). No mineralogy argument can discriminate it.
* **The natural warm-dust background vs age is measured:** Kennedy & Wyatt 2013,
  12 µm over 24,174 Hipparcos MS stars within 150 pc — old (>Gyr) dusty systems
  occur at **1 in 10⁴**, young (<120 Myr) at **~1 %**. Fractional luminosity
  decays as ~1/age² (Pawellek+2021 observed; Wyatt+2011 theoretical).
* **The natural background vs metallicity:** Gáspár, Rieke & Ballering 2016 —
  *"disk-bearing stars seldom have metallicities less than [Fe/H] = −0.2"* over
  662 disks. Planet occurrence ∝ 10^(2[Fe/H]) (Wyatt, Clarke & Greaves 2007).
* **Background galaxies killed the entire warm-Dyson candidate list.** JWST/MIRI
  resolved Hephaistos D and E into a z≈0.9 Hot DOG and a z≈0.4 dusty starburst,
  both within ~1″ (arXiv:2607.09460); Hot DOGs at ~9×10⁻⁶ arcsec⁻² can account
  for all seven. High |b| helps against **cirrus and stellar blends only** — it
  does *not* reduce extragalactic confusion. Only sub-arcsecond astrometric
  registration at the *propagated* epoch plus a chance-superposition prior does.

* **Deeper WISE data is *anti-correlated* with colder sensitivity.** Wien-peak
  temperatures: W1 3.35 µm → 865 K, W2 4.60 µm → 630 K, W3 11.56 µm → 251 K,
  W4 22.09 µm → 131 K. Every WISE catalogue that got deeper after 2010
  (NEOWISE-R, CatWISE2020, unWISE) is **W1/W2 only**; W3/W4 depth is frozen at
  the 2010 cryogenic mission. So the largest waste-heat searches ever run are
  structurally incapable of reaching 100–300 K, and improving them makes them
  *warmer*. Below ~130 K the mid-IR route is closed by instrumentation, not by
  effort. Do not propose "go deeper in WISE" as a route to cold Dyson spheres.
* **A parallax error is exactly a grey offset**, and any twin/reference-star
  scatter is common-mode across bands. Both must enter a multi-band fit as a
  **rank-1 fully correlated** covariance term, never on the diagonal — treating
  them as independent per band inflates every significance by ~√N_bands.
* **Scatter about a reference-star median is not the error bar.** It also
  contains the parameter gradient across the matching box (measured: 0.14 mag
  instead of 0.05 for a Teff box of ±150 K). Take the scatter about a *local
  linear fit* in parameter space instead.
* **A published IR-excess catalogue is ~92% false positive** (Silverberg et al.
  2018: at most 7.9%±0.2% of AllWISE-selected excesses are good disk
  candidates; all 13 Theissen & West W4 S/N>3 candidates are false). Measure an
  excess from the photometry; never inherit one.
* **Far-IR beams make background-galaxy confusion far worse than in WISE.**
  IRAS/AKARI beams are 25–180″ vs WISE's 6–12″, so the coincidence area is
  10²–10⁴× larger; with 10⁶ targets the chance-match expectation runs to
  thousands. Measured Gaia source density (results/farir_stats): 101,853/deg² at
  |b|<5° vs 3,119/deg² at |b|>60°. A far-IR positional association is never
  evidence on its own.

* **Gaia XP is low-resolution** (R≈30–100; LSF ≈5+ of the 2-nm samples). Two
  consequences: (1) it *cannot resolve a narrow laser line* — a real localised
  feature must be interior (≥8 samples from either end), bounded (falls below
  half-peak on both sides), and 2–5 samples wide; a 1-sample spike is
  sub-resolution noise and a monotonic ramp pinned to 336 nm / 1020 nm is a
  basis-function reconstruction artifact (both now cut in `xp.anomaly`).
  (2) Broad-SED "anomalies" (a Dyson reprocessing deficit/excess) are
  **degenerate with interstellar reddening** and with cool-star molecular bands
  (TiO/VO/H₂O) — a low-|b| field is the worst case. Fit the colour locus with
  ≥40 sources/bin (`fit_locus`) or the per-bin MAD collapses and flags ~70%.
* SDSS/DESI wavelengths are **vacuum**; all literature line lists (air) are
  converted via `seti.spectra.reject.air_to_vacuum` at definition time. This
  was a real leak: pre-fix "candidates" sat on He I 5876 / Ca II 8542 / O I
  8446. Fixed 2026-07-01.
* Catalogue redshift/RV errors move known lines outside the in-funnel
  rejection window → the observed-frame ±300 km/s triage
  (`seti.spectra.triage`, costs 22.7% of the band, honestly accounted) is
  mandatory before believing any spectral candidate.
* **Misclassified emission-line galaxies** are the worst spectral leak: a
  background star-forming/active galaxy that SDSS/DESI labels `STAR` (or gives a
  wrong z) drops its rest-frame nebular family into the search as
  high-significance "unresolved" lines. The observed-frame known-line triage
  cannot catch it (it uses the wrong catalogue z). Decisive test = *internal
  redshift consistency*: if ≥2 surviving lines in one spectrum form a locked
  nebular pair (Hα+[N II], [O III] 4959/5007, [S II] 6716/6731) or ≥3 lines at a
  common z, it is a galaxy (`galaxy_reject`, verdict `galaxy_zmatch`). This killed
  the former #1 candidate. Single-line candidates cannot be tested this way —
  they need the per-exposure persistence check.
* Candidate wavelengths recurring across unrelated sightlines (≥3 spectra
  within ±3 Å, across runs *and* modes) are instrumental. 31 killed.
* Merged candidate CSVs can contain duplicate rows (runs overlap) — 89 killed.
* ZTF single-band events are artifacts until g/r-coincident
  (`multiband_coincidence`, `secular_achromatic`, `glint_achromatic`).
* Stellar flares are chromatic (g ≫ r); a glint must be achromatic.
* WD IR-excess contaminants, in the order they bite: (1) **WISE blend** — a
  comparably-bright red Gaia neighbour inside the ~6.5″ W1 beam (the WD is
  IR-faint); test with `discriminate.blend` (Gaia beam neighbours + expected W1).
  (2) **Unresolved stellar companion** — a WD+dM/dL binary is a *single* Gaia
  source (looks "isolated") whose fitted excess temperature is >1800 K, hotter
  than grains survive: an "excess" above the dust sublimation temperature is a
  companion photosphere, not dust or a swarm (kills the τ=0.6 standout). (3) **CV**
  (accretion). Only after all three does a τ<0.08, T_dust<1800 K excess read as
  an ordinary debris disk.
* WD IR excess: dusty debris disks are the one natural confounder — subtract
  the labelled catalogues before scoring.
* **AllWISE W4 (22 µm) is unreliable for faint stars** — it is the shallowest,
  most confusion-limited band, so for 22-µm-faint M dwarfs the catalogue W4 flux
  is background cirrus / a noise measurement, producing huge (up to ~6 mag),
  formally-high-σ "W1−W4 excesses" with a *photospheric* W1−W2. A real
  warm-dust/waste-heat SED is bounded and lights up the star-dominated bands
  first, so a **W4-only excess is an artefact**, not a detection. Likewise a
  **negative W1−W2 is a W1/W2 blend** (a bare photosphere has W1−W2 ≥ 0), not a
  star. The LHS 1140 neighbour screen requires the excess in W1−W2 or W1−W3 with
  W1−W2 ≥ −0.05; W4-only/negative-W1−W2 go to `needs_vetting` (killed 11 of 15
  raw neighbour flags). Faint sources (G≳18) and non-M-dwarf (blue bp_rp) matches
  are additionally WISE-confusion/photosphere-model-invalid, not excesses.
* A secular optical fade with a NEOWISE fade at ~6% of the optical rate is
  ordinary line-of-sight dust (extinction-law ratio) — check
  `w1_to_optical_slope_ratio` before getting excited. Gray occulters sit at
  ≳30%; warm dust *brightens* the IR.
* ASAS-SN (pyasassn) is flaky on runners — pass `--optical-slope` to
  `dimming-characterize` so the mid-IR verdict never returns
  `insufficient_ir` for want of a known number.

* **SHROUD / VASCO sample (2026-07-26).** (1) A **plate defect has no infrared
  counterpart** — requiring a real IR detection is itself a strong artifact
  filter, applied at selection time. The residual worry is the opposite: a
  defect landing by chance within 5" of an unrelated IR source. (2) At the
  published **5" radius the chance-match probability is ~0.9% at high galactic
  latitude, ~10% against AllWISE all-sky, ~24% against CatWISE2020, and →1 in
  the plane** — of order 10^4–10^5 of the 172,163 "counterparts" may be
  coincidences. Watters+2026 leave this "undetermined"; **measure it with an
  offset-position null**, never with a uniform-random background (assuming
  uniformity is the exact error that broke the VASCO Earth-shadow analysis).
  (3) **A naive Stern+2012 `W1−W2 ≥ 0.8` AGN cut deletes the shroud population**
  — a 350 K shroud has W1−W2 = 3.2. Colour cannot separate them; **SED shape**
  can (AGN = power law, shroud = curved blackbody), and with <3 IR bands the two
  are formally undecidable. (4) `vanish-neowise` was matched to **NeoWISE, which
  carries W1/W2 only** — a 2-band IR integral badly *under*-estimates a thermal
  SED and would manufacture a spurious "IR too faint" result. Join AllWISE
  W3/W4 + 2MASS before issuing any deficit verdict. (5) Sample "S" was built by
  removing everything within 5" of **Gaia DR3 / Pan-STARRS DR2**, so modern
  optical non-detection is guaranteed by construction, not measured.
  (6) Expect **171,753** rows from the live archive, not the abstract's 172,163.

## Rules of engagement (from CLAUDE.md)

Novelty first, scale second, never write up a null. Merge every commit to
`main` as you go (non-fast-forward merge if diverged; never force-push).
Data-touching runs go through `workflow_dispatch`; the sandbox has no archive
egress.

### EMBER (cross-epoch mid-IR, IRAS/AKARI → WISE) — established 2026-07-26

* **NEOWISE cannot see waste heat.** It flies W1/W2 (3.4/4.6 µm) only; W3/W4
  exist for the 2010 cryogenic phase alone. W1/W2 reach only T ≳ 500–700 K, so
  *any* decade-baseline mid-IR excess-change search at 100–300 K is impossible
  with it. Do not propose one. NEOWISE's real value is as a **flatness** test.
* **IRAS 25 µm → WISE W4 is the near-identical pair**: the early→late flux
  transfer moves by only **1.03×** across dust temperatures of 150–1500 K.
  I12 → W3 moves by **1.71×** and **AKARI S9W → W3 by 8.25×** — the 9-to-12 µm
  step is emphatically *not* a null transformation and must never be treated as
  one. These are the numbers from the **real SVO response curves** (committed
  in `src/seti/data_assets/rsr/`); the documented trapezoid fallback gives 1.24,
  1.20 and 5.18 respectively, which **reverses the ordering of the top two
  pairs** and understates the systematic on I12→W3 by 40%. A trapezoidal
  bandpass is not a conservative approximation. `rsr_source` is recorded per
  band in every audit for exactly this reason.
* **The IRAS 100 µm background cut is worth a factor of ~30.** Kennedy & Wyatt
  2012 (arXiv:1207.0521): ~8,000 of 180,000 stars show an apparent IRAS excess
  correlated with the 100 µm background; below 5 MJy/sr, 271 remain. Mandatory
  for any IRAS-based excess work in this repository.
* **IRAS beam vs WISE beam is 286× in solid angle.** An IRAS flux is the sum
  over its footprint, so the only defensible comparison sums *all* late-epoch
  sources in the early beam. Comparing against the nearest counterpart alone
  fabricates fades wherever the field is crowded.
* **W3 saturates at ≈0.96 Jy, barely above the IRAS PSC completeness limit of
  0.4 Jy.** Bright IRAS sources are exactly the ones WISE cannot measure, and a
  saturated late band under-reports flux and mimics a cessation. Use I25→W4
  (saturates at 12 Jy) or let AKARI arbitrate.
* **Eddington/Malmquist bias is one-directional and fades only.** A flux-limited
  early epoch plus a deeper late epoch manufactures cessations with no
  astrophysical change. A two-sided null cannot calibrate it; deboost explicitly
  and impose an early-epoch S/N floor.
* **A blackbody at T_eff is not a stellar atmosphere.** Extrapolating 2MASS Ks to
  12 µm with a Planck function over-predicts by ~0.3 mag at 5000 K. Use an
  *empirical* per-band colour locus — it also absorbs each survey's calibration
  scale, which is what Liu 2020 attributed IRAS–WISE discrepancies to.
* **Fit the photospheric locus on low quantiles, not the median.** In an
  IR-selected catalogue the excess population can exceed 50%, which is exactly
  the median's breakdown point.
* **The published cross-epoch stability floor is 4%** (HD 172555, IRAS 1983 →
  WISE 2010, arXiv:1210.6258). Nothing within a few times that is believable,
  however significant.
* **Every natural mid-IR variable class varies *persistently*** — 14 of 17
  extreme debris disks changed at 3–5 µm between 2010 and 2019. A single
  monotonic step followed by a flat decade is the discriminant. TYC 8241 2652 1
  is the sole known step-and-stay object and is still unexplained.
* **Smooth disc dispersal cannot make this signal**: τ = 2–3 Myr at 3.4–12 µm is
  a ~10⁻⁵ change over 27 years. Only discrete events can.

### TIDEMARK (population-level spatial structure) — established 2026-07-26

The first channel here that asks a question **about a population rather than
about objects**, and therefore the first one immune to the per-object
contamination that ended the previous six. Full design and honest limits in
`docs/tidemark.md`.

* **The question.** Is the anomaly rate *per star* structured across the Galaxy
  — a gradient in Galactocentric R / |z| / longitude, a sharp edge (a boundary),
  or a trend with stellar age? Not "is this object real", which is the question
  that always fails.
* **Novelty: three published predictions that contradict each other, and none has
  ever been tested.** Ćirković & Bradbury 2006 (New Astronomy 11, 628) predicts
  the **outer rim**; Wright, Carroll-Nellenback, Frank & Scharf 2021 (RNAAS 5,
  141) predicts the **Galactic centre**; Wright et al. 2014 (Ĝ I, ApJ 792, 26)
  predicts **no coherent structure at all** because rotational shear mixes any
  Fermi bubble on a rotation timescale. Carrigan 2010 / Landis 1998 / Hanson et
  al. 2021 predict a **boundary** as the observable — Hanson et al. explicitly
  estimate "how common in the sky the volume borders would be, for which
  astronomers might search", with no published response. Verified against 642
  citing titles plus the 2026 field-wide review (arXiv:2605.21093), which
  contains zero occurrences of "galactocentric", "spatial distribution",
  "percolation", "Fermi bubble" or "border".
* **In every executed search, Galactic position appears in exactly four modes and
  none is this test**: as a cut (|b| ≥ 10, bulge excision); as qualitative
  Aitoff eyeballing (Carrigan 2009 Fig. 8); as a sky-averaged, position-*independent*
  surface density for chance-alignment budgets; and as rate vs heliocentric
  distance where the trend *is* the incompleteness (Hephaistos I, Table 1,
  stated as such). Everybody imposes the selection function; nobody inverts it.
* **Nearest misses.** Blain 2024 (arXiv:2409.11447) *proposes* the sky-distribution
  test and implements nothing — and his null ("do candidates shadow the Gaia
  stellar density?") is exactly the selection function TIDEMARK divides out.
  Huang, Tao & Zhang 2026 (arXiv:2605.06072) is the closest methodological
  precedent: a dynamical index regressed against R_GC over 79 globular clusters
  with a selection-preserving null — but the units are clusters, the observable
  is not a technosignature, and the paper disclaims the interpretation.
* **The selection function is the whole difficulty and the whole contribution.**
  Every parent star carries `w_i = c_s / N_s` (anomalies in its detectability
  stratum / parent rows in it): the empirical, non-parametric probability that a
  star like it is flagged. `Σ w_i = n_anom` exactly — the correction
  redistributes anomalies, never invents them. `ρ = n_obs / Σw` is the
  selection-corrected rate ratio and `ρ ≡ 1` is the null. Weights are written out
  per star so the correction can be audited rather than trusted.
* **`MatchedNull` raises if you stratify on the coordinate under test.** A
  silently self-cancelling test is worse than no test.
* **THE CORRECTION IS CONSERVATIVE BY A FACTOR OF ~2, AND THIS MUST BE QUOTED.**
  Detectability covariates are themselves correlated with position (magnitude
  with distance, distance with R_gal through the GC direction, extinction with
  |b|), so matching absorbs part of a *real* trend. Measured: an injected
  0.6/kpc radial gradient returns as ≈0.27/kpc — transfer ≈ 0.45. Every run
  measures its own transfer coefficient and reports
  `R_gal_slope_deabsorbed_ln_per_kpc`. **A measured slope is a lower bound.**
* **A smooth gradient will fake an edge unless the null contains the gradient.**
  Calibrated: a pure 0.8/kpc gradient fires the edge detector 15% of the time
  with a quadratic trend model in the null and 5% (nominal) with a **cubic**, at
  no cost in bubble power. Hence `smooth_order: 3`. Do not lower it.
* **The scan's look-elsewhere effect is paid for**, not ignored: the statistic is
  the max |S| over every centre, radius and width, calibrated against the same
  max recomputed on each matched-null draw.
* **Injection performance** (synthetic parent with the full confounder stack —
  exponential disk, magnitude limit, plane extinction, radial metallicity
  gradient, AVR): a 900 pc / contrast-5 bubble recovered at p = 0.004 with the
  centre localised to 89 pc and radius to 902 pc vs 900 pc true. Clean nulls on a
  pure density gradient, a hard detectability artifact, and both footprint
  shapes.
* **Most channels here cannot be ingested, because they publish survivors only.**
  Only `cenotaph` (`greyfit.parquet`) and `ossuary` (`sample.parquet`) write a
  real parent sample. `cluster_run` held its full frame in memory and threw it
  away — now fixed, it writes `parent_sample.parquet` per cone. **Any new channel
  should write its parent, not just its candidates**: without a denominator there
  is no rate, and TIDEMARK reports `NO_PARENT_SAMPLE` rather than invent one.
* **The excess locus must be fitted globally, never per cone.** Fitting per field
  normalises every field to its own median and deletes exactly the
  field-to-field rate differences being measured. `tests/test_tidemark.py`
  asserts this against the per-cone counterfactual.
* **`cites:doi:<doi>` is a broken OpenAlex filter** — it is accepted and silently
  returns `meta.count = 0` for every DOI. All earlier citation-tree fetches in
  this repo are empty for that reason, not because the papers are uncited.
  Resolve the DOI to a work ID and use `cites:W...`. Fixed in
  `scripts/necrolit_fetch.py`.

### TIDEMARK reporting-logic failure and fix — 2026-07-26

The first committed TIDEMARK run emitted `verdict: DETECTION`. It was wrong, and
every reason generalises to any population-level statistic this repo builds.
Recorded here so it is not re-derived.

* **A Monte Carlo p-value equal to `1/(n_null+1)` is a BOUND, not a
  measurement.** It means "no null realisation was this extreme". Three
  "independent" edge geometries all returned exactly `0.0033222591 = 1/301` —
  that is the floor, not agreement. Report floor-limited p as `p < x`, escalate
  the draw count before believing it, and never feed a floor value to a trials
  correction as though it were measured.
* **Identical p-values across "independent" tests mean they are not
  independent.** Each edge geometry now reports which anomalies produced its
  step; sets overlapping by Jaccard ≥ 0.5 are one feature and count once.
* **Guard every statistic on the anomalies carrying ITS OWN coordinate, never on
  the catalogue total.** The dimming catalogue had 2555 anomalies of which **30
  had a parallax**. The 3D shell scan and the |z| edge scan guarded on 2555 and
  ran on 30. A scan on 30 objects across 24 bins will find a step. Threshold is
  now 30 *usable* anomalies per test, returning `INSUFFICIENT_ANOMALIES` with
  `p_value: null` — which the aggregator surfaces rather than silently skips.
* **`p = None` must be a verdict, not a gap.** The original aggregator filtered
  non-finite p-values out of the family and then declared a detection on what
  remained.
* **Channels spell the same covariate differently, and a covariate list matched
  by literal name silently drops the ones it does not recognise.** `dimming`
  calls its magnitude `g_mag`; the global list says `phot_g_mean_mag`. Result:
  a 255,469-star catalogue matched on three columns, **no magnitude at all**, in
  44 strata. Covariates are now resolved into physical *families* through an
  alias table, and a missing magnitude family blocks a detection outright.
* **The tested coordinate is the one covariate the null makes no promise about,
  so its residual imbalance bounds that test's credibility and belongs next to
  the p-value.** `R_gal_kpc` had SMD 0.197 — the worst of any covariate — while
  the p-value it produced was being read as a detection. Rubin (2001)
  convention: |SMD| < 0.10 good, < 0.25 marginal. ≥ 0.10 now blocks a detection.
* **A top-N% score cut is not a candidate population.** It returns exactly the
  fraction you asked for whatever the data looks like, so structure in it traces
  the survey at least as readily as the sky. Only a vetted candidate list sets
  `vetted=True`, and only a vetted population can earn `DETECTION`.
* **A channel at a known systematics floor must carry that caveat in the result
  string, not only in its docs.** `dimming_secular` now carries
  `caveat_tag: AT_SYSTEMATICS_FLOOR`.
* **General rule: a committed `DETECTION` is the artifact that gets mistaken for
  a result later.** Verdicts are now gated, every gate is written out, and a
  non-detection names the gate that stopped it.
