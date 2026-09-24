# CONFLUENCE — do independent channels' tails share stars?

## 1. Claim

Every channel in this repository asks one question of one observable and ends
in a clean null or a traced mundane survivor. A single artefact of engineering
need not be extreme in any one observable; it may be *moderately* odd in
several at once. CONFLUENCE asks the population-level question no single
channel can: **do the same Gaia sources sit in the top-q tails of two or more
channels whose anomalies are measured with independent instruments, more
often than a covariate-matched null predicts?** A star in the 5 % tail of a
ZTF dip score *and* the 5 % tail of an APOGEE/GALAH sparse-abundance score is
unremarkable to either channel; if such stars are over-represented after
matching on everything that makes instruments misbehave, and the excess is not
carried by known astrophysical classes, the residual members are the
candidates.

## 2. Novelty status

Anomaly searches that fuse several observables into one statistic exist
(Project Hephaistos' Gaia+2MASS+WISE SED fits, Suazo et al. 2024; the
Breakthrough Listen "exotica" catalogue, Lacki et al. 2021, which *lists*
extreme objects of many kinds; anomaly detection in single surveys, e.g.
Giles & Walkowicz 2019). What we have not found is the specific population
test here: the tail overlap of *separately built* channels on independent
instruments, against an exact conditional-independence null, with known
classes as the positive control and the residual as the product. This is a
statement of what we know, not the result of a systematic literature search
by this run.

## 3. Method

1. **Per-star scores.** Most channels commit only survivors, but their runs
   upload the whole scored parent as a workflow artifact, readable for 90 days
   by any later run. `harvest.py` downloads those artifacts on the runner and
   extracts `(source_id | position, score, flag)` per star with schema
   discovery (an unrecognised file is reported with its columns, never
   guessed). Channels harvested: CENOTAPH (grey-fit significance, 674k),
   IGNITION (two-band NEOWISE rise significance), CRADLE (log f/f_max), RING
   (WD leg, if a parent-level table exists), TAILINGS (its own reduce re-run on
   its checkpointed GALAH/APOGEE catalogues), OSSUARY; committed:
   DIMMING (ZTF dip and signed secular fade), BAFFLE-bright (AKARI/IRAS deficit,
   and excess as a control observable), SLAG, GROWTH, SPARK; acquired: the whole
   Gaia DR3 `nss_acceleration_astro` table (ACCEL's parent, by definition).
2. **Independence.** Each channel names the instruments whose measurement
   *is* its anomaly (`primary`) and those used only to normalise it
   (`support`). A pair is tested only if the two tag sets are disjoint
   (strict); the relaxed rule (primaries disjoint, neither primary the other's
   support) is reported separately and never in place of it. Two WISE-based
   channels, two ZTF channels, or two Kepler/TESS channels are never paired.
3. **Identity.** Gaia DR3 `source_id` where the channel carries it; otherwise
   the nearest Gaia-keyed position within 2″; otherwise position-to-position
   across channels.
4. **Tails** are computed over each channel's whole parent (top 1 %, top 5 %,
   and the channel's own flag), *before* restriction to the joint parent J
   (stars in both parents).
5. **Null (headline).** Conditional independence given covariates: each star's
   probability of sitting in channel X's tail given (G, BP−RP, |b|, RUWE,
   neighbour density, Gaia scan coverage, WISE depth) is estimated by the tail
   fraction among its 100 nearest neighbours in robust-standardised covariate
   space; under the null star i is in both tails with probability
   p_A(i)·p_B(i), and the overlap count is Poisson-binomial, computed exactly.
   A shared systematic raises both propensities for the same stars and is
   absorbed. **Cross-check:** exact stratified permutation within coarse
   covariate cells (convolution of per-cell hypergeometrics), reported as
   `p_cells`. Coarse cells leak a systematic that switches on inside a bin
   (the offline test demonstrates it), which is why they are not the headline.
   Triples use p_A·p_B·p_C. Benjamini–Hochberg across all tests.
6. **Positive controls, run first.** (a) Known classes (SIMBAD otype, VSX type,
   Gaia DR3 variability class, Gaia multiplicity) must be enriched among
   overlap members relative to a random sample of the joint parents.
   (b) Injection: k stars planted in both channels' score tables between the
   upper-5 % and upper-0.5 % quantiles — inside the confluence tail, below
   either channel's alert level — must be recovered. If either fails, the
   verdict is `CONTROLS_FAILED` and nothing downstream is believed.
7. **Residual.** The test is re-run with known-class stars removed from every
   joint parent; each remaining member is traced (every channel's score and
   percentile, covariates, and the systematics that fire: RUWE > 1.4,
   multi-peak IPD, G < 6, crowding, DSC galaxy/quasar probability).

## 4. Contamination model

| Route to a false confluence | Handling |
|---|---|
| Shared instrument (two WISE channels) | never paired (strict tag rule) |
| Shared *normalisation* (both use 2MASS Ks) | excluded under strict, flagged under relaxed |
| Blends / unresolved companions | RUWE, IPD multi-peak and density are null covariates; trace flags them |
| Brightness-dependent systematics (saturation, S/N) | G is a null covariate; G < 6 flagged |
| Coverage depth (WISE, Gaia scan law) | ecliptic latitude and visibility periods are null covariates |
| Galaxies/QSOs in stellar samples | DSC probabilities in the trace; SIMBAD/Gaia class tags |
| Astrophysical multi-channel classes (YSO, EB, CV, Be, LPV, active stars) | tagged; they are the positive control, then removed |
| Duplicate rows across overlapping runs | one row per (channel, star), most anomalous kept |

## 5. Status

See `results/confluence/summary.json` and the STATUS.md section. On committed
files alone (before any harvest) the largest strict-independent joint parent
is 118 stars (CENOTAPH × GROWTH), and no pair can be tested: the statistic
needs the harvested per-star parents.
