# METRONOME — clocks in stellar flare timing

**Signature S28** (`docs/necrosignatures.md` §2.VI): *structure in timing
series.*  The one informational-residue signature in the taxonomy that had
never been built.

---

## 1. The claim

Is there any star whose catalogued brief optical brightenings — the "flares"
of the public Kepler and TESS flare catalogues — occur on a **strict clock**?

Natural flares are stochastic.  Their waiting times are Poisson-like (or
power-law-tailed and clustered, which is worse for coherence, not better).
The only natural quasi-periodicity in flare *occurrence* is rotational
modulation of flare visibility — the active hemisphere turning toward us —
and that carries a phase jitter of order a tenth of a rotation cycle or more,
because a spot group is not a point and a flare is not a tick.

An artificial periodic energy release — a pulsed transmitter, a duty-cycled
engine, a beacon — is a clock whose timing jitter is far below anything
rotation produces.  The observable is therefore the **timing regularity of
catalogued events, not the events themselves**.  Everything a flare catalogue
already did (finding the events, measuring their energies) is inherited; the
channel adds one statistic and the machinery to keep it honest.

This is not a search for periodic *photometric variability* — Lomb–Scargle
over Kepler and TESS light curves has been run to exhaustion — it is a search
for periodicity in a **point process** built from the catalogued event times.
A clock with a 1% duty cycle contributes no detectable Fourier power to a light
curve and is invisible to every periodogram-based survey; in the event-time
domain it is a delta function in phase.

---

## 2. Novelty position — stated honestly, to be verified by `metronomelit`

**What is claimed, as far as this session knows:** no published work has taken
the public flare catalogues (Yang & Liu 2019; Davenport 2016; Pietras et al.
2022; Günther et al. 2020; Tu et al. 2020) and asked, per star, whether the
catalogued **event times** are phase-coherent at any period with a jitter far
below rotational, against a null that resamples the mission's own observing
windows and cadence.

**What is adjacent and must not be confused with it** (the decoys the
literature sweep tags explicitly):

| Adjacent work | Why it is not this |
|---|---|
| **Quasi-periodic pulsations (QPPs)** in individual flares | Oscillations *within* one flare, seconds to minutes, MHD physics of the flaring loop.  Different timescale, different observable. |
| **Rotational-phase dependence of flare occurrence** (Hawley et al. 2014 on GJ 1243; Doyle et al. 2018/2019 in K2/TESS; Roettenbacher & Vida 2018) | Asks whether flares cluster at a rotational phase.  They largely do not, and where they do the jitter is a sizeable fraction of a cycle.  This is the natural quasi-periodicity METRONOME's `rotation_alias` and `jitter_too_large` vetoes exist for — the *baseline*, not the target. |
| **Periodic activity in FRBs and magnetars** (CHIME 2020 and after) | Different objects; but the statistical machinery — H-test on event phases, window-resampled nulls — is the same, and it is the precedent that event-time periodicity can be established from catalogued bursts alone. |
| **Flare waiting-time distributions** (Wheatland 2000 for the Sun; Kepler/TESS follow-ups) | Characterise the *natural* null this channel resamples.  Nobody, as far as this session knows, has inverted the question. |
| **Optical-SETI pulse searches** (nanosecond laser pulse detection) | Search for the *pulse*, at ns timescales, on dedicated instruments.  METRONOME searches for the *schedule*, at hours-to-months timescales, in archival catalogues. |

**Verification status.**  The sandbox has no archive egress; nothing above was
read in full text here.  `scripts/metronomelit_fetch.py` runs on the runner,
saves **verbatim** arXiv abstracts under `results/metronomelit/`, checks every
asserted arXiv id against its expected title (`id_title_check.json`), and runs
a decoy-aware concept scan (`concept_scan.json`).  The novelty position above
is **"to be verified by metronomelit"** until that file exists and its
`decoy_free_hits` has been read.  If a decoy-free hit turns out to be exactly
this search, the channel's claim narrows to whatever that paper did not do —
and the doc changes before anything else does.

---

## 3. Data

All public, small, on VizieR (reachable from the GitHub runner; not from the
sandbox).  Table ids in `config/metronome.yaml` are **preferred seeds**; the
probe stage discovers the real table under each id at runtime (§4.6).

| Role | Catalogue | VizieR seed |
|---|---|---|
| Kepler flares | Yang & Liu 2019, ApJS 241, 29 (162,262 flares / 3,420 stars) | `J/ApJS/241/29` |
| Kepler flares (cross-check) | Davenport 2016, ApJ 829, 23 | `J/ApJ/829/23` |
| TESS flares | Pietras et al. 2022, ApJ 935, 143 (~140k flares / ~25k stars) | `J/ApJ/935/143` |
| TESS flares | Günther et al. 2020, AJ 159, 60 | `J/AJ/159/60` |
| TESS superflares | Tu et al. (the brief's id; discovered at runtime) | `J/ApJS/253/35` |
| Kepler rotation | McQuillan+2014, Santos+2021, Reinhold+2013 | `J/ApJS/211/24`, `J/ApJS/255/17`, `J/A+A/560/A4` |
| TESS rotation | keyword-discovered | — |
| Positions | KIC `V/133/kic`; TIC `IV/39/tic82` / `IV/38/tic` | shortlist only |
| Periodic variables | VSX `B/vsx/vsx`; Gaia DR3 `I/358/vclassre`; ZTF Chen+2020 `J/ApJS/249/18` | 3″ cones, shortlist only |

**Observing windows** are built from the catalogue's own event density
(`windows_from_events`): every event in the catalogue is binned in 0.1-day
bins, and a run of empty bins is a gap when it is ≥ 0.5 d long *and* the
catalogue's mean rate predicts ≥ 20 events in it.  With ~10² events/day in a
mission-scale catalogue, Kepler's 1–3 d inter-quarter gaps and ~1 d monthly
downlinks resolve; a sparse catalogue resolves only its long gaps, and the
window label records the effective minimum gap.  A published (approximate)
Kepler quarter table is the fallback below 2,000 events.  Per star, the
mission windows are clipped to the star's own event span and a window with no
events in which the star's own rate predicts ≥ 5 is dropped as presumed
unobserved (module failure; TESS target not on silicon that sector).  **This is
a stated approximation**: it slightly weakens the evidence against a clock that
fell silent for a window, in the conservative direction for a candidate claim
but the permissive one for the null.

---

## 4. Method (`src/seti/metronome/`)

### 4.1 Pre-processing, per catalogue
* **Cross-star coincidence removal** (`clock.cross_star_coincidence`).  Bins
  of two long cadences (Kepler) / five 2-min cadences (TESS); a bin in which
  the number of *distinct* stars with an event exceeds
  `max(5, Poisson_isf(10⁻⁶/n_bins, λ) + 1)` is spacecraft, not sky (momentum
  dumps, Argabrightenings, scattered-light excursions).  Every event in it is
  removed from every star **before any star is scanned**, and the epochs and
  counts are in `summary.json`.
* **Declustering** (`clock.decluster`).  Events within 0.1 d (2.4 h) are one
  event; complex and sympathetic flares are one energy release.

### 4.2 The statistic, per star with N ≥ 8 events
For each trial frequency on a grid from ~10 cadences (0.2 d floor) up to a
third of the star's span, oversampled ×5 in 1/span: the **H-test** (de Jager
et al. 1989) on the event phases, `H = max_m (Z²_m − 4m + 4)`, m ≤ 4.  A clock
is a delta in phase and fills every harmonic; rotational modulation
(`rate ∝ 1 + cos φ`) is a pure fundamental.  The grid peak is refined locally
and then walked up the harmonic ladder (`fundamental_period`): a zero-jitter
clock is exactly as coherent at P/7 as at P, so the grid maximum can land on a
sub-harmonic, and the walk takes the longest multiple that keeps the phase
concentration — stopping at 2P for any clock that ticks in consecutive cycles.

At the best period the **clock quality** is read off the phase distribution:

| Quantity | Definition | Clock | Rotation |
|---|---|---|---|
| `Q` | `1 − s/s₀`, s = angular deviation √(2(1−R̄)), s₀ its uniform expectation | → 1 | ~0.3 |
| `jitter` | rms residual from `t₀ + kP`, in units of P | ≲ 0.01 | 0.15–0.3 |
| `f_in_window` | fraction within ±0.05 cycle of the clock phase | → 1 | ~0.1 |
| `gap_integer_frac` | fraction of consecutive same-window waiting times that are integer periods (±0.05) | → 1 | ~0.05–0.1 |
| `cycle_occupancy` | ticks with an event / ticks in observed time | duty cycle | — |

`gap_integer_frac` is the property a clock has and nothing else does, and it
is what makes the second null interpretable (§4.3).

### 4.3 Two nulls, and what each is for
1. **Window-resampled** (`clock.window_null`): N times uniform in the star's
   own observed time, snapped to the mission cadence, scanned identically.
   Quarter gaps, sector gaps, orbit gaps and 30-min quantisation are inside
   the null by construction.  The p-value is the sequential Monte Carlo p of
   Besag & Clifford (1991): trials continue until 10 exceedances or 2,000
   trials (300 s budget).  An ordinary star stops after ~20 trials; a star
   nothing exceeds runs the full budget and its p is quoted from a Gumbel fit
   to the null's own max-H tail, **flagged `p_extrapolated`**.  Stars whose
   loose Bonferroni bound (`n_freq · e^{−0.3H}`, deliberately heavier-tailed
   than the true `e^{−0.4H}`) already exceeds 0.5 skip the null: they could not
   have been significant, and `null_computed = False` says so.
2. **Waiting-time shuffle** (`clock.shuffle_null`): the star's own waiting
   times, permuted in observed-time coordinates.  Burstiness is preserved;
   long-range phase order is destroyed.  **A strict clock survives this null**
   — its waiting times *are* the signal — so `p_shuffle` is not a candidate
   criterion by itself.  The joint condition *`p_shuffle ≥ 0.05` and
   `gap_integer_frac < 0.6`* is the `bursty_random` rejection: coherence
   explained by the waiting-time distribution *without* the waiting times being
   clock-like.

**Calibration of the extrapolation** (measured in the sandbox, 300 Poisson
stars in Q2–Q8 Kepler windows, N = 30, 200-trial nulls): fraction of stars
with Gumbel-extrapolated p < 0.01: **1.7%**; < 0.05: **7.3%**; < 0.5: 47%.
The empirical p on the same stars gave 1.7% / 6.7% / 48%.  Within binomial
noise of uniform; the extrapolation is not conservative and is therefore
flagged wherever it is used.

### 4.4 Tiers, with BH-FDR across every star scanned
`p_window` for every scanned star (empirical where available, the Bonferroni
screen bound where the null was skipped, the flagged Gumbel value where pinned
at the floor) goes through Benjamini–Hochberg at α = 0.05 (`fdr_significant`)
and α = 0.25 (`fdr_watch`).  The per-star max-statistic null already absorbs
the period-grid trials; BH absorbs the star count.

| Tier | Requires |
|---|---|
| `none` | not `fdr_watch`, or any hard veto (§5) |
| `watch` | `fdr_watch`, no hard veto, loose quality (Q ≥ 0.6, jitter ≤ 0.12) |
| `interest` | `fdr_significant`, strict quality (Q ≥ 0.85, jitter ≤ 0.05, `gap_integer_frac` ≥ 0.6 over ≥ 4 gaps), but a veto **could not be applied** (no P_rot; a variability catalogue unreached) |
| `candidate` | as `interest` with every veto applied and passed.  **Always pending light-curve inspection.** |

### 4.5 Energy–phase coherence (report only)
Spearman ρ of log energy against distance from the clock phase.  A clock does
not care how bright the tick is; rotational modulation does (visibility).
`energy_incoherent` when p < 0.01.  Never rejects — a beacon *could* modulate
amplitude — but a reader sees it.

### 4.6 Runtime schema discovery (`acquire.py`)
VizieR catalogue numbers and column names are not stable facts (the sibling
`tailings` channel lost three dispatches to a renumbered table and a mangled
column).  The probe stage lists every table under the preferred id in
`TAP_SCHEMA.tables`, reads their real columns from `TAP_SCHEMA.columns`,
resolves the roles (star id; peak / start / end time; energy; sector; P_rot;
position) with exact canonicalised regexes (`resolve_columns` — substring
matching is what let `Per` match `Perr` elsewhere), scores each table as a
per-flare list, falls back to a keyword search, and writes the whole
scoreboard to `probe.json` with the time-system guess (`BKJD` / `BTJD` / `BJD`
/ `MJD`, from the median of a 200-row peek through a verified column).  The
acquisition log separates `QUERY_FAILED` from `QUERY_RETURNED_ZERO_ROWS` at
every stage.

---

## 5. Contamination ledger — every rejection is a named counter

Applied most-mundane-first; a star tripping several is reported under the
dullest.  `summary.json["rejection_counters"]` carries `first_veto` (one per
star), `flags_raised` (every flag) and `tiers`.

| Rule | Mechanism | Test |
|---|---|---|
| `cross_star_coincidence` | Spacecraft systematics shared by many stars (momentum dumps, Argabrightening, scattered light) | Removed at event level before scanning; counted per star (`n_removed_cross_star`) and per catalogue |
| `insufficient_events` | < 8 events after declustering and removal | — |
| `not_significant` | The window-resampled null explains the coherence | BH-FDR |
| `cadence_alias` | P at a named instrumental period or its 2×, 3×, ½, ⅓: Kepler long cadence, ~3 d momentum dumps, ~31 d downlinks, ~93 d quarters; TESS 2-min / 10-min / 200-s / 30-min cadences, ~3.5 d early-sector momentum dumps, 13.7 d orbit, 27.4 d sector | 2% tolerance.  The brief's "6.02 h" Kepler figure could not be verified and is **not** applied. |
| `rotation_alias` | Rotational modulation of flare visibility — the dominant natural quasi-periodicity | P within 3% of P_rot, P_rot/2, /3, /4, 2P_rot, 3P_rot, from McQuillan/Santos/Reinhold or the flare catalogue's own P_rot |
| `periodic_variable` | A pulsator's or eclipsing binary's cycles chopped into "flares" by the flare finder (RR Lyrae, δ Sct, EBs) | VSX / Gaia DR3 vari / ZTF cone at 3″; P within 3% of the catalogued period or its ½, ⅓, 2×, 3× |
| `bursty_random` | Clustered-but-random flaring whose coherence the waiting-time shuffle reproduces | `p_shuffle ≥ 0.05` **and** `gap_integer_frac < 0.6` |
| `jitter_too_large` | Not a clock: fails even the loose thresholds | Q < 0.6 or jitter > 0.12 |
| `energy_incoherent` *(report)* | Energy depends on clock phase — visibility, not a beacon | Spearman p < 0.01 |
| `rotation_unknown`, `variability_catalogue_unreached` *(report)* | A veto could not be applied | Caps the tier at `interest` |
| `p_extrapolated`, `null_truncated_by_budget` *(report)* | Statistical provenance | — |

**Where the clock thresholds sit.**  The assess stage measures the jitter and
Q distributions of the stars it *rejected* as `rotation_alias` — the natural
quasi-periodic population — and reports the percentiles beside the thresholds
(`summary.json["jitter_calibration"]`), with the fraction of that population
below `jitter_max`.  The thresholds are chosen so that number is zero; the run
reports whether it is.

---

## 6. Measured performance (`tests/test_metronome.py`, offline, no network)

| Case | Result |
|---|---|
| Strict clock (P = 3.137 d, duty 0.5, σ = 3 min) in Kepler Q2–Q8 windows with 30-min quantisation | recovered to < 0.1% in P; Q > 0.95, jitter < 0.01, `gap_integer_frac` > 0.9, p < 10⁻⁶ (0 exceedances in 100 trials) |
| Strict clock (P = 0.913 d) in six synthetic TESS sectors with orbit gaps, 2-min cadence | recovered to < 0.1%; same quality |
| Clock with duty 0.25 (grid peak lands on P/7) | harmonic walk returns P to < 0.1% |
| 30 Poisson stars, N = 12–30 | 0 candidate, 0 interest, ≥ 28 `none` |
| Rotationally modulated star (rate ∝ 1 + cos, P_rot = 2.5 d, N = 160) | coherence is real (p < 10⁻³) but jitter > 0.12, Q < 0.6; `rotation_alias` with P_rot known, `jitter_too_large` without |
| Momentum-dump epoch shared by 40 stars | 1 bad bin, 40 events removed before scanning, counted per star |
| Catalogued RR Lyrae (P = 0.5668 d) whose cycles are "flares" | recovered as a perfect clock, then `periodic_variable` |
| Bursty star (12 bursts × 6 events) | not `interest` or better |
| Unreachable archive (every query raises) | `NO_DATA_REACHED`, zero counts, empty candidates, every probe entry `QUERY_FAILED` |
| Archive answers with no tables | `QUERY_RETURNED_ZERO_ROWS`, "NOT a null result" in the note |
| End-to-end synthetic catalogue through probe → acquire → screen → assess | offline: `DEGRADED_SOURCE (rotation_kepler:none); CLOCK_CANDIDATES_PENDING_VET`, the clock at `interest` with `rotation_unknown`; with a rotation table and scripted cones: 1 `candidate`, the rotator now `rotation_alias`, verdict clean |
| Every hard veto, `insufficient_events`, `not_significant`, every report flag | each has a case that trips it and appears in the counters |

---

## 7. Coverage limits — what the search can and cannot see

* **The catalogue's flare-detection threshold is the channel's sensitivity
  floor.**  A clock whose ticks fall below the catalogue's amplitude / energy
  threshold, or whose ticks the catalogue's pipeline rejected, is invisible
  here by construction.  Each catalogue's threshold is inherited, not
  re-measured.
* **Cadence.**  A peak time cannot define a clock finer than ~10 cadences:
  0.2 d for Kepler long cadence, 0.014 d for TESS 2-min.  Shorter periods are
  not scanned.
* **Span.**  Periods longer than a third of the star's event span are not
  scanned (fewer than three cycles is a trend).  Kepler: ≲ 500 d; a single
  TESS sector: ≲ 9 d.
* **N ≥ 8** after declustering: stars with fewer catalogued flares are not
  tested, and the TESS catalogues are dominated by such stars.
* **Duty cycle.**  A clock is detectable at any duty cycle high enough to leave
  ≥ 8 events; `cycle_occupancy` reports it.  A clock that ticks once per
  window is a trend.
* **Presumed-unobserved windows** (§3) are dropped, which removes the
  evidence a silent window would carry against a clock.
* **Vetting reach.**  P_rot exists for a minority of TESS flare stars; the
  variability cones are run only for the FDR shortlist.  A star at `interest`
  is one whose vet is incomplete, not one that passed.

---

## 8. What a null would and would not mean

A `NO_CLOCK_CANDIDATES` verdict means: among stars with ≥ 8 catalogued flares
in these catalogues, none shows event-time phase coherence at a period between
~10 cadences and span/3 that (a) beats its own window-resampled null after
BH-FDR at 0.05, (b) has jitter ≤ 0.05 P, and (c) is not a rotation, cadence,
or catalogued-variable alias.  It says **nothing** about clocks below the
catalogue's detection threshold, clocks with fewer than 8 visible ticks,
clocks faster than ten cadences, clocks slower than a third of the span, or
clocks in the ~10⁵ Kepler/TESS stars that never entered a flare catalogue.

Per `CLAUDE.md`, a clean null is **not written up**.  It is a reason to change
the question: to run the flare finder rather than inherit it (recovering the
sub-threshold regime), to move to a different event class (the ZTF alert
stream's brief brightenings; the CHIME-style burst catalogues), or to test
weaker structure than a strict clock — arithmetic progressions with a drifting
period, prime-number or Fibonacci gap patterns (S28's broader reading), which
the H-test is not built for.  `NO_DATA_REACHED` and `QUERY_RETURNED_ZERO_ROWS`
are statements about archive access and the catalogues' contents, not about
the sky, and the workflow refuses to let either read as a science null.

---

## 9. Layout

```
src/seti/metronome/windows.py   observing-window model (Kepler quarters, TESS sectors, data-driven)
src/seti/metronome/clock.py     H-test scan, clock quality, two nulls, cross-star removal, BH  [pure]
src/seti/metronome/vet.py       the gauntlet and the tiers                                     [pure]
src/seti/metronome/acquire.py   runner-only VizieR access, runtime schema discovery, AcquisitionLog
src/seti/metronome/run.py       stages probe / acquire / screen / assess -> results/metronome/
config/metronome.yaml           every threshold, every table seed
tests/test_metronome.py         offline suite (the CI gate)
.github/workflows/metronome.yml probe+acquire -> screen matrix -> assess -> commit-back; lit job
scripts/metronomelit_fetch.py   prior-art sweep, verbatim abstracts -> results/metronomelit/
```

Entry point: `python -m seti.metronome.run --stage {probe|acquire|screen|assess|all}
[--catalogues a,b] [--shard i --n-shards n] [--max-stars k] [--max-rows r] [--offline]`;
programmatic `seti.metronome.run.metronome_run(cfg, stage=..., catalogues=..., shard=...,
n_shards=..., max_stars=..., max_rows=..., offline=..., seed=..., out_root=...)`.

Outputs: `probe.json`, `acquire.json`, `acquisition_log.json`, `screen_<cat>[_s<i>of<n>].json`,
`stars_<cat>[...].csv` (every star scanned), `stars_vetted.csv`, `summary.json`
(verdict, funnel, rejection counters, jitter calibration, coverage, `generated_utc`,
per-catalogue acquisition log), `candidates.json` (interest + candidate, and the watch list).
