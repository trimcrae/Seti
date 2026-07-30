# Research scoreboard

Live per-channel state of the search. Update this file whenever a run,
vet, or triage changes the candidate picture — it is the single place a
human (or a fresh agent session) looks to know what is hot and what to do
next. Last updated: 2026-07-30.

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
| **CENOTAPH** (cold Dyson, T<100 K) | built + dispatched (run 30203250183); target Gaia DR3 GSP-Spec dwarfs, ~5.6e6 with Teff/logg/[M/H]/[α/Fe] from one pipeline | pending first archive run | **new channel.** Three-leg energy-conservation test: grey attenuation (A_V fitted jointly, not assumed) + NO mid-IR excess + far-IR recovery of the intercepted f·L in AKARI/FIS + IRAS. Closure ratio ρ=f_IR/f_dim separates an isotropic occulter (ρ≈1) from an edge-on disk (ρ≪1). Measured floor f≳0.15–0.29 vs Zackrisson+2018's f_cov>0.75. Next: read `results/cenotaph/summary.json`, check the ±3σ tail asymmetry before believing any count |
| **SHROUD** (enshrouded, not destroyed; S33) | built; first run dispatched. The **never-analysed catalogue by-product** of Solano, Villarroel & Rodrigo 2022 (MNRAS 515, 1380): `vanish-neowise` = **171 753** POSS-I sources absent in the modern optical but detected in the infrared, plus `vanish-possi` = **5 399** with no counterpart at all, as the control | pending first run | **New channel.** Endpoints verified: `http://svocats.cab.inta-csic.es/vanish-{neowise,possi}/` (quoted in Watters+2026 Table 1 and the jannefi/vasco README). Solano+2022 is **not** in VizieR (runner fetch returns "Table or Catalog not found") — SVO is the only machine-readable route. Measurement: eta = F_IR(now)/[F_bol(POSS-I) − F_bol(modern)], a **pure flux ratio so distance cancels** — no parallax needed. Forés-Toribio & Kochanek 2026's progenitor/remnant ratio *is* eta, applied at ≥10 (merger remnant) and ≤0.3 (genuine disappearance). Scoped **strictly to the archival crossmatch** — no VASCO transient/Earth-shadow/nuclear-test analysis (docs/shroud.md §1) |
| **EMBER** (waste heat that switched off; S1) | built; **acquisition repaired and re-run** (probe run 30209647320, search run 30210104587). The only three all-sky surveys carrying 12–25 µm photometry — IRAS (1983), AKARI/IRC (2006–07), WISE cryogenic (2010) — cross-matched via Gaia DR3 with PM propagated to each survey epoch. **105,694 rows acquired per RA shard** after the repair (run 30203763934 had reported `acquired: 0`) | first run reached no data for three unrelated reasons that all presented as one zero; all three fixed and the distinction made structural | **New channel.** *Novelty:* the entire Dyson/waste-heat lineage is single-epoch by word count — "epoch" in a photometric sense = **0** in Carrigan 2009, Ĝ I/II/III and Hephaistos I; the 2026 flagship review (2605.21093) has `turn off`/`switched off`/`cessation`/`multi-epoch`/`AKARI`/`NEOWISE` all = 0; Suazo+2024 *explicitly cut* variable stars, discarding a changed megastructure by construction. **Antecedents that must be cited, not ignored:** Kim+2015 (1501.05721) ran the identical IRAS+AKARI+WISE comparison **upward** (4 sources all-sky); Sedgwick & Serjeant 2022 (2207.09985) built the IRAS×AKARI 23.4-yr cross-match for *proper motion*; Melis+2023 ran it downward but targeted at R CrB. **Audit reversed the brief, and then the real SVO curves reversed the audit:** AKARI 9 µm→W3 has a transfer spread of **8.25×** over 150–1500 K (worst-conditioned pair, demoted); **I25→W4 is spread 1.03 — near-null, <1% bandpass systematic, 12× wider unsaturated window — and is now the PRIMARY pair**, with I12→W3 (1.71) corroborating. NEOWISE **rejected as an epoch** (W1/W2 only; cannot see 100–300 K dust) and reused as the post-drop flatness test. **Measured acquisition facts (probe 30209647320):** IRAS PSC/FSC positions are **B1950** (`RA1950`/`DE1950`) and must be precessed; the anonymous ESA Gaia TAP upload returns **HTTP 500 even at 200 rows** so CDS X-Match is the primary Gaia route; AllWISE has 2.3M rows per 1° RA slice so only X-Match is viable. Next: read `results/ember/summary.json` funnel counts and the rising-tail asymmetry in `null_calibration.json` before believing any count |
| **TAILINGS** (the sparse chemical anomaly; S12/S15/S22) | built; first dispatch hit stale VizieR catalogue numbers and returned `NO_DATA_REACHED` honestly, fixed by runtime TAP_SCHEMA table discovery + schema scoring; re-dispatched. Target GALAH DR4 + APOGEE DR17 cool dwarfs (Teff<6000 K, logg>4.0), ~20–30 elements each | pending first archive run | **New channel, and the discriminant is inverted.** Natural abundance space is low-dimensional (~8–10 independent axes: Ting+2012, Price-Jones & Bovy 2018, Patil+2022) and every natural process moves an element FAMILY; refining moves ONE. So a **dense** anomaly is a REJECTION here — the opposite polarity to every existing abundance-outlier statistic, all of which are global distances (PCA/EMPCA/t-SNE/autoencoder/k-means) built to *cluster* stars, and all of which are maximised by dense anomalies. Direct arXiv queries for `"anomaly detection" AND APOGEE`, `… AND GALAH`, `"outlier detection" AND "stellar abundances"`, `"abundance anomaly" AND "single element"` all return **0**. **The real competitor, read in full: Huang, Tao & Zhang 2026 (arXiv:2605.29811)** — executed, meteorite-calibrated Bayesian test for refined material, but in **polluted white dwarfs** (7.7≤logg≤8.3; "GALAH" and "APOGEE" appear 0 times), on 697 literature records/≥397 objects, against a **fixed dense siderophile template**, and its power *rises* with element count — "typically requires ≳5 detected elements for decisive support". The two searches rank the same objects in nearly opposite order. Their strongest criticism (a 1–2-element anomaly can carry a big Bayes factor while information-starved) is answered by `n_quiet`: their cases are records where the other elements were never MEASURED; here 20–30 are measured **and quiet within 2σ**. Whitmire & Wright 1980's 56-citation tree still contains **no executed survey**; note this channel deliberately inverts their A5–F2 host choice — that band is where diffusion/levitation manufacture single-element anomalies naturally (hence 60 years of contested Przybylski claims), so trading amplitude for a clean null is the right trade. **Hardening after four adversarial sweeps:** (1) **metallicity leak closed** — convective protection depends on [Fe/H] too; a ~0.85 M☉ metal-poor turnoff star has M_env < 1e-7 M☉, four decades thinner than a solar-metallicity dwarf, and passes a Teff/logg cut (Matrozis 1605.02791). Floor at **[Fe/H] ≥ −1.0** in the ADQL, config and funnel; also **Teff ≥ 4000 K** because GALAH's own notes put cool-dwarf systematics at up to **0.5 dex below 4600 K**. (Cross-channel: OSSUARY *selects* what this excludes.) (2) **Search in [X/H], never [X/Fe]** — an error in the star's own [Fe/H] otherwise smears a sparse anomaly across every element (Weinberg's measurement aberration); stated cost is that an **iron-only** anomaly is now invisible by construction. (3) **Karinkuzhi gate** — 13 'Sr-only' + 2 'Ba-only' LAMOST (R~1,800) candidates re-observed at R~86,000 **all dissolved into dense barium stars**, so below R=20,000 sparsity is presumed to be blending and the candidate carries `needs_high_resolution_confirmation`; plus a curve-of-growth check (a saturated core is not a measure of abundance). (4) **Instrumental covariate veto** — Weinberg traced high-Ca stars to one RV+fibre bad-pixel combination and a *population* of low-K stars to a −70 km/s telluric coincidence, so flag rates are binned against RV and fibre. (5) **Validation target**: Griffith 2110.06240 found, incidentally, **15 stars with 0.3–0.6 dex Na enhancement and normal O–Ni** — the pipeline must recover them. Contamination baseline ~60% junk. **CORPUS-INTEGRITY WARNING (repo-wide lesson):** the first lit fetch reported 107/107 successful and **12 of 24 hardcoded arXiv IDs had resolved to unrelated papers** — Plenoxels for APOGEE DR17, an LHC dark-matter paper for Vick, neutron-star precession for Richer — all fetching cleanly. **A successful fetch is no evidence the paper is the right one.** See `results/tailingslit/INTEGRITY.md`; offending files deleted, harness now resolves by title search with title-token verification. **Two dispatch failures, both recorded because the honest verdict disguised each one:** run 30203627605 hit stale VizieR catalogue numbers (`III/283/allstar` etc. all 'table not found'), fixed by runtime `TAP_SCHEMA` discovery + schema scoring; run 30204487245 then reached discovery, probed the right tables and STILL returned `NO_DATA_REACHED` — because `TAP_SCHEMA.tables.table_name` comes back **already double-quoted**, so `FROM ""III/283/allstar""` was rejected by everything. **LEDGER RULE (general): a channel whose null verdict is indistinguishable from its bug verdict will mislead its own author** — the distinguishing test belongs in the suite, not in a log. Third dispatch queued. Next: read `results/tailings/contrast_*.csv` — if the SPARSE fraction is flat with z_max the sample is systematics, not a population |
| **RUST** (unmaintained decay; S9) | built; **novelty run 30203976309** (complete, 47/47 fetches) + **search run 30204427240** (8 high-galactic-latitude ZTF fields, g+r paired). Statistic: bias-corrected **season scatter** regressed on calendar time — the **second moment**, not the first. *(Run 1's sweep was cancelled and re-dispatched: it was executing pre-fix code, before the per-season line detrend closed the accelerating-fade leak — see below. Its novelty job had already completed and its evidence is kept.)* | pending first archive run | **New channel, and it is NOT the `dimming` channel again.** `dimming/secular.py` fits a weighted line to season **medians** (brightness, first moment) and is exhausted at the ZTF systematics floor; RUST regresses per-season robust **scatter**. The distinction is structural, not rhetorical: a shared zeropoint/reference drift moves every star's median together — the false fade that killed `dimming` — and leaves the *within-season scatter* untouched. But the exchange is not free, and RUST inherits two systematics `dimming` never had. **(a) Cadence bias.** `1.4826×MAD` recovers only ~66% of σ at N=3 and ~90% at N=8; epochs-per-season is set by survey cadence and cadence *trends with calendar time* (ZTF public went 3-day → 2-day in 2020). Uncorrected, this channel measures ZTF's operations calendar. Handled in five layers: per-season null computed with **that season's own N and its own per-epoch error vector** (MC bias table b(N) + heteroscedastic mixture-MAD scale); excess variance subtracted not divided, **including the second-order `E[s²]=(bσ)²(1+u²)` term** — a 17% offset at N=8 vs 2% at N=70, i.e. a pure function of cadence, hence of time; a **distribution-free** exact permutation p-value over all n! season orderings as the primary gate (Lacki gives a cascade *timescale*, not a light-curve shape, so a linear model would assume the result); per-CCD ensemble common mode in the second moment; and survivor-only exact-MC + **equal-N subsampling** cross-checks that share no machinery with the first layer. **(b) `magerr` mis-calibration drifting with time** — killed by the ensemble κ_s = median(σ_obs²/σ_null²) over the field. Measured offline: **0/600** constant stars flag across five cadence histories (rising 8→70, falling 70→8, ZTF-style jump, erratic, doubling), 0/2000 over a wider sweep; ~57% recovery at 36 mmag terminal amplitude, ~80% at 60 mmag. **Two-band g/r coincidence is enforced at SCORING, not follow-up** — a one-band source is never scored — and the g/r amplitude-growth *ratio* is the physics discriminant (grey occulter ≈1.00 / extinction law ≈1.42 / flare-accretion >1.70). **NEOWISE logic is INVERTED relative to `dimming`:** a mid-IR brightening *corroborates* dust production from a cascade instead of killing the candidate; its absence is informative but not fatal (W1/W2 only probe >600–850 K). **Novelty independently re-verified on the runner:** `scripts/rustlit_fetch.py` ran 24 targeted arXiv queries with the four decoy classes encoded as explicit regexes; over **215 abstracts** it found **1 hit, decoy-tagged `with_timescale` (red noise), and 0 decoy-free hits** (`results/rustlit/concept_scan.json`). The three counterweight/antecedent abstracts are now quoted from fetched text, not memory: McInnes 2026 — a ring-supported stellar engine "can in principle be **passively stable**" and dense-cloud bubbles "**passively self-stabilizing**"; Wright 2020 — monolithic spheres "**dynamically unstable** under gravity and radiation pressure, and **mechanically unstable to buckling**"; Petz & Kochanek 2025 — selection is on "brightness changes larger than ~0.03 mag/year", i.e. a mean-flux slope. **One novelty leg failed and is recorded as failed:** the OpenAlex citation tree resolved to arXiv preprint stubs reporting `cited_by_count=0` for all four targets *including Wright 2020*, so it is uninformative and carries no weight; the script now also searches by title so a re-dispatch fixes it. **A sixth systematic layer was added after the first dispatch:** a fitted **line** is removed per season, not just the median — a star fading at an *accelerating* rate drifts further within each successive season, which reads as a rising second moment produced entirely by a first-moment phenomenon, i.e. the `dimming` channel's own population leaking in (**46/60** false positives at 0.72 mag total fade; **0/60** with the line removed). The null table switches to its line-detrended variant to match, since a line fit costs an N-dependent amount (b(8): 0.900→0.811; b(80): 0.990→0.984). Honest two-band completeness: 34% at a 36 mmag terminal amplitude, ~65% at 60–90 mmag; the ~80% per-band plateau is the chi2-inflated linear gate refusing an accelerating rise, not noise. *Novelty:* a regex concept scan over 3,578 fetched abstracts for "amplitude/scatter increasing with **calendar time**" returns **0**; every apparent hit is a different statistic — Polaris (a single named star), T Tauri in WASP (amplitude vs **timescale**, red noise), YSOs in W51 (vs **evolutionary stage**), NLSy1 (vs **wavelength**). Nearest machinery is Petz & Kochanek 2025 (2501.14058), 9,361,613 ASAS-SN sources at >0.03 mag/yr — a **mean-flux slope**; the second moment is untouched. Bonus seam: Hephaistos II's `G_var > 2` cut explicitly discards "potential Dyson swarms with very large absorbing elements", i.e. exactly what this channel selects on. Sensitivity is stated *against* the counterweight: **not** sensitive to McInnes 2026 (2603.00203) passively-stable ring engines / dense-cloud bubbles, which need no upkeep, nor to Wright 2020 (2006.16734) monolithic spheres, whose failure is catastrophic rather than a decade-long ramp. Next: read `results/rust/summary.json` and `results/rustlit/concept_scan.json`; for any survivor demand the ASAS-SN cross-survey result — **red noise, not white, is the least-controlled residual** and the quoted false-positive rates are against Gaussian nulls |
| **KNELL** (the clock that stopped; S32) | built 2026-07-26; offline suite green (**32 tests**, all passing); dispatched (see run id in the channel log). **The observable is a PERIOD that ceased** — a clock that stops is the cleanest "the mechanism ended" signature, and it is the one necrosignature in this program that **dust cannot fake**: obscuration changes a signal's *amplitude*, not the *existence* of a period, so the line-of-sight-extinction confounder that dominates `dimming`, EMBER and CENOTAPH is absent by construction. **What replaces it is survey-dependent detectability, and that is the whole channel.** A variable "ceases" whenever the later data have worse cadence, a shorter seasonal window, larger errors or a different alias comb — so the primary search is **intra-survey** (ZTF's own early seasons against its own late seasons, in g *and* r, where passband/pipeline/calibration are constant by construction) and **every claimed non-detection is normalised by the injection-measured detection efficiency for that star's own period and amplitude in that block's own sampling and noise** (`src/seti/knell/efficiency.py`, the load-bearing module). Efficiency is measured by injecting a random-phase sinusoid **into the post-transition block's own observed magnitudes** — by the cessation hypothesis that block holds no signal, so it supplies the real sampling, real error distribution, real correlated systematics and real outliers for free; nothing about the noise is modelled, so nothing about the noise can be modelled wrongly — and scoring it with the **identical** detector (batched GLS, verified against `astropy.timeseries.LombScargle` to 1e-8, against the block's own **permutation** threshold). **Measured: the uncorrected statistic flags 22 of 24 constant-signal stars whose only change is a degrading cadence (80,80,70 → 18,15,15 epochs/season); the efficiency gate flags 0 of 24.** A 92% false-positive rate turned off — worse than the sibling RUST channel's 46/60, and the direct justification for the module. Degrading *errors* at fixed cadence: 0/10. p-values are Clopper-Pearson **upper bounds** and are printed as inequalities when pinned at the injection-resolution floor (`<= 4.97e-05 (injection-resolution limited)`); a test asserts that escalating trials tightens the bound. Confounders closed with named mechanisms: **mode switching** (blind per-block detector fires on *any* frequency, **plus** an excess-variance test, because in a mode switch the total power is *conserved* and merely moves); **Blazhko** (>=2 post blocks, >=500 d post baseline, pre-modulation index); **third-body nodal precession / SS Lac** (pre-cessation amplitude decline + Gaia RUWE/`non_single_star`); **fade** (mean flux must be unchanged, an independent guard from the efficiency gate); CV disc states, AGN red noise, YSOs, blending, both ZTF photometric walls. **Novelty is narrowed honestly.** Not "the first search for stars that stopped varying" — falsifiable inside EBs by Jurysek 2018 (1709.08087) and Graczyk 2011. The defensible claim: **cessation events have only ever been found one at a time and serendipitously, and no survey has ever measured the rate at fixed detection sensitivity.** Kurtz et al. 2025 (2412.04840, MNRAS 536, 2103) claim a *first ever* for one star ceasing pulsation; OGLE found "objects ceasing pulsations" **by accident inside a catalogue paper** (1601.02020); Ansari+2023 report **9,881 of 58,200 GCVS variables with no Gaia variability flag**, attributed entirely to detection capability — that is this channel's null hypothesis, quantified. Theoretical anchor: **Kipping & Teachey 2016 (1603.08928)**, the only paper proposing *deliberate erasure* of a periodic signal as a technosignature, which runs no search. **Substrate decision (the sharpest result of the literature sweep): eclipsing binaries are EXCLUDED from the anomaly sample** and used only as a calibration set — third-body nodal precession is a named, modelled, actively hunted mechanism whose rate (~28 lost systems in one Kepler->TESS revisit) would swamp anything exotic, and it comes with predicted return dates. The novelty lives in **pulsators**. **Two citation errors in the brief were caught and corrected:** arXiv:astro-ph/9805019 is **Tomasella & Munari 1998**, not Torres & Stefanik (whose SS Lac paper is AJ 119, 1914, 2000, with **no arXiv ID found** — none is asserted); and arXiv:1807.03448 is the **WISE** periodic-variable catalogue, not ZTF's (that is **2005.08662**, ApJS 249, 18). All verification was WebSearch-only (WebFetch/curl 403-blocked on *every* host incl. example.com), a control test proved the backend's phrase matching is not strict, so **no quotation is certified verbatim** and re-verification is a runner-side job. **A provenance bug was found and fixed in the process:** the inherited bulk ZTF fetcher catches its own HTTP errors and returns an empty dict, making a proxy 403 **indistinguishable from an empty sky box** — a whole run of refused queries would have reported "zero rows" and read as a search. A separate one-shot service probe now restores the distinction, and the sweep emits three different verdicts (`NO_DATA_REACHED` / `ARCHIVE_RETURNED_ZERO_SOURCES` / `NO_TESTABLE_LIGHT_CURVES`) where before there was one. The vet job **fails the workflow** if a verdict with 0 testable stars is anything other than a NO_DATA class. Next: read `results/knell/summary.json` and the per-field `acquisition_log.json`; for any survivor, check the **HR-diagram position first** — the closest analogous search (Jarvinen & Strassmeier 2025, 2504.19670) lost **11 of its 13** candidates to Gaia parallaxes showing they were evolved stars — then confront the `spot_cycle_plausible` flag, which is the one benign interpretation this channel does **not** close | dispatched, awaiting first archive run | **new channel; see `docs/knell.md`** |
| **VIGIL** (waste heat with a duty cycle; S4) | built 2026-07-26; offline suite green (39 tests). **Novelty + probe run 30215516935** (novelty and probe jobs COMPLETE; its sweep shards were cancelled and superseded because they were executing the pre-fix per-star acquisition — see throughput below). **Search run 30216181263** (8 fields near the north ecliptic pole, where the NEOWISE scan pattern piles up so per-star visit counts are highest, all at |b| = 23-37 deg so cirrus and crowding are suppressed at selection). **Novelty verdict UNOCCUPIED, established on the runner**: 39/39 fetches OK, **303 abstracts scanned** over 17 targeted queries, **0 genuine prior-art hits** — nothing matched the conjunction (mid-IR/WISE/NEOWISE) AND (variability) AND (optically constant) AND (technosignature/Dyson/SETI/megastructure) free of the four decoy classes (YSO, extreme debris disk, AGN/blazar, optical-transit megastructure); the only two abstracts scoring 3-4 concept groups were both AGN-tagged. **All four decisive arXiv IDs verified BY TITLE**, not by number: 2511.22071 = 'A Catalogue of Mid-infrared Variable Sources from unTimely' (**8,256,042 W1 and 7,147,661 W2 variables** — the catalogue is real and the size claim holds), 2103.00568 = the ALLWISE warm-EDD sample, 2006.16734 = 'Dyson Spheres', 2403.18941 = the five-million-star mid-IR excess search. The Hephaistos quote is now **verbatim from fetched full text**: Hephaistos II's check "rejects potential Dyson swarms with very large absorbing elements since these in principle could generate detectable variations in the photometry of the host star", and it assumes swarms "with no pieces large enough to cause stellar variability". **Data reachability, stated exactly:** NEOWISE per-epoch W1/W2 is reachable (`status: OK`, 32 rows against a `COUNT(*)` of 32, 27 surviving frame-quality cleaning). The **unTimely variable catalogue is NOT reachable as a queryable table** — and the first probe could not say so honestly, because VizieR returned an ADQL *syntax error* (its parser rejects `LOWER(col)` in WHERE, which IRSA and NOIRLab both accept) while the other two ran and returned zero rows: one transport failure plus two absences is not an absence claim. Query rewritten to spell out case variants, verdict vocabulary now separates ALL_TAP_ROUTES_FAILED / NOT_FOUND_BUT_SEARCH_INCOMPLETE / CATALOGUE_NOT_FOUND_ON_ANY_TAP_ROUTE. Corroborating evidence that this is a real absence and not a bad query: **35,230 characters of the paper's text contain no data URL, no Zenodo deposit, no VizieR identifier and no Data Lab table** — the catalogue has no machine-readable release yet. The parent unTimely Catalog does (catalog.unwise.me, github.com/fkiwy/unTimely). The channel does not depend on it: per-epoch NEOWISE is required regardless, since a variability *catalogue* carries detection flags, not the epochs the modulation index / morphology / colour statistics need; `preselect_from_untimely` uses the table when found and records `applied: False` with the reason when not. **Throughput fix from run 1:** a single-star NEOWISE cone measured **~92 s**, which caps a 400-star field at a few dozen stars — so the sweep now issues ONE field-wide cone and assigns exposures to stars locally by KD-tree, propagating each star's PM to mission mid-epoch and widening its own match radius by half its mission sweep (tested: a high-PM star is recovered with propagation and lost without). Per-star cones remain the fallback and a field-query failure lands in the ledger. **Second run-1 lesson:** the probe's NEOWISE call at a bare coordinate binned to **zero usable visits** — 32 sporadic detections of a marginal source, never 3+ inside one visit, so the within-visit noise calibration had nothing to work with. Returning `None` was correct, but a probe measuring empty sky tests nothing; it now resolves a real Gaia star first and reports median exposures-per-visit, which IS the per-star sensitivity and varies across the sky with the scan pattern. | pending first sweep results | The observable is a star **variable in the mid-infrared while constant in the optical** — steady interception, unsteady re-emission — and, in the extinction reading, that variability *ceasing*. **The novelty seam is a cut somebody else made:** Hephaistos II's `G_var > 2` explicitly *rejects* variable stars because a swarm with very large absorbing elements "could generate detectable variations in the photometry of the host star" — that is a cut against **optical** variability from occultation, while this channel selects on **mid-IR** variability from modulated re-emission, which no published search has ever used. | pending first archive run | **The confounder is the entire channel.** Extreme debris disks have exactly this phenomenology: strongly mid-IR variable with flat optical light curves (Moor et al. 2021, arXiv:2103.00568, warm EDDs from AllWISE). So **optical constancy buys nothing against the dominant contaminant** — it kills YSOs, dippers and AGN and nothing else — and a channel that stops at "mid-IR variable, optically constant" has produced an EDD catalogue and called it a search. **Primary discriminator:** mid-IR variability at **low** fractional excess, because an EDD is "extreme" precisely because `τ ~ 1e-2`. The arithmetic that makes it work: `F_exc/F_phot = τ·R` with `R = [B_ν(T_d)/T_d⁴]/[B_ν(T_*)/T_*⁴]`, so at 600 K around a 5000 K star `R(W2) = 23` — τ=1.5e-3 is a **3.5% band excess** (a ~37 mmag switch, well inside NEOWISE visit-mean precision) while an EDD at τ=1e-2 is **23%**. A factor ~5 separation against a 2–3% photosphere systematic: that is why the cut works, and it is applied to the **2σ upper limit** because "low" has to be a bound, not a point estimate that scattered low. **Sharpened form (the one real improvement on the brief):** `R` swings by >13× across the plausible dust-temperature range, and it *cancels* if you write the maximum amplitude an excess can produce against the **band** excess `f` rather than against τ — `A_max = 2f/(1+f)`, so the **modulation index** `m = A_obs(1+f)/(2f)` is free of T_dust, T_*, distance and luminosity. It measures *what fraction of the excess is actually switching*: an EDD is a perturbation on a large steady excess (`m ≪ 1`), a load-following radiator switches all of it (`m → 1`), and `m > 1` is a **falsifier** — the variability cannot be the excess (blend, bad epoch, underestimated photosphere). `m` is tightly constrained exactly where it must be (large, well-measured excesses = the confounder) and honestly abstains when the excess is marginal. **Two shape backstops:** decay-vs-duty-cycle morphology (weighted trend R², Kendall τ, likelihood two-state BIC split, squareness, transition rate, Lomb–Scargle, burst skew) — a smooth secular decay is a collisional cascade and is rejected **even at low excess**, independently tested; and colour-temperature stability of the *varying component* (a cascade changes amount **and** temperature as grains spread and cool; a radiator changes amount at fixed temperature). **Estimator:** NEOWISE visit structure is a free noise calibrator — ~10–20 exposures inside a 1-day visit measure the true per-exposure noise per star, while visit-to-visit scatter measures the 6-month variability, so the optimistic `w?sigmpro` values are **rescaled per star** by the within-visit scatter (an optimistic error is how a variability search manufactures candidates). Primary statistic is the **unbiased normalised excess variance** with the Vaughan et al. 2003 uncertainty carrying the exact N dependence — necessary because exposures-per-visit and visits-per-star both trend with ecliptic latitude, so an uncorrected version maps the NEOWISE scan pattern; plus equal-N re-measurement and a per-field ensemble common mode. **Repository gap closed:** there was **no PM-propagated `neowiser_p1bs_psd` fetcher anywhere** — all three existing single-exposure callers query at the Gaia epoch, and a 200 mas/yr star drifts ~2.1″ across the 2014–2024 mission, comparable to the cone radius. `vigil.acquire.fetch_neowise_epochs` propagates to mid-epoch 2019.0 and widens by the sweep. **Hard instrumental bound, stated not assumed away:** NEOWISE/CatWISE/unWISE are **W1/W2 only** (Wien peaks 852 K, 630 K), so VIGIL probes **hot** material and is structurally blind to the 100–300 K regime where most Dyson models sit; W3/W4 depth has been frozen since the 2010 cryogenic mission and W3/W4 are used here **only to reject** (a W4-only signal is cirrus). AGN are adjudicated on **astrometry, not colour**, because a ~350 K shroud has W1−W2 = 3.2 and sits inside the Stern/Assef box. Next: read `results/vigil/probe.json` (is the unTimely variable catalogue reachable, and what is it called), then `results/vigillit/summary.json` — and note the novelty script verifies every decisive arXiv ID **by title**, so a failed verification of 2511.22071 must be read as "the catalogue may not exist as claimed", not ignored |
| **DERELICT** (dead lightsails; S19) | built; **census extended and re-run** (run 30209538685; prior data run 30204805880). JPL SBDB fits a radial non-gravitational coefficient A1 for every small body with enough astrometry; `A1 → β → area-to-mass` is a two-line conversion and a thin film sits 4–5 orders of magnitude above any natural body of the same size. **22 asteroids have a fitted A1** (`sb-cdata={"AND":["A1|DF"]}`); the **comet control returns 272** | the target set is small but **not empty**, and the control proves the query fires; funnel: a1_significant 11, screen1_a1_only 4, strict 0, negative-A1 1, verdict `ALL_SURVIVORS_EXPLAINED` | **The first run's `NO_DATA_REACHED` was a typo, not a sky.** `sigma_A1` is not a valid SBDB field; one bad column name 400'd all four constraint strategies, so "our query has a typo" presented as "the database contains no such object". Fixed by self-healing field pruning; sigmas now come from `sbdb.api`'s `orbit.model_pars[].sigma`, which is authoritative anyway. *Novelty:* Bialy & Loeb 2018 ran this exact inference **for 1I/'Oumuamua alone**; JPL/MPC run it **reactively and per-object** to unmask human hardware (2020 SO, J002E3, WT1190F); the dark-comet catalogues (Seligman+2023/2024) selected the **opposite** population — large *non-radial* acceleration, which excludes radiation pressure. The claim is the *method*: a systematic catalogue-scale selection of the A1-only complement, normalised by size into R = AMR_implied/AMR_natural. **Live novelty risk:** Loeb & Cloete 2025 (2503.03552) already argue one dark comet is artificial, so "a dark comet might be artificial" is published and is not claimed here. **Empirical systematic floor:** 4179 Toutatis shows a 3.4σ *sunward* A1 — which radiation pressure cannot produce — giving |R| ≈ 5 against a flag threshold of R = 10. 'Oumuamua recovers R ≈ 1.1×10⁵ (AMR 1.23 m²/kg) as the positive control. Next: `results/derelict/completeness.json` — whether the A1|DF constraint really returns the whole A1 population — then `dark_comets.csv` and `high_albedo.csv` |

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
