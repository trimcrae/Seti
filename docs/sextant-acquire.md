# SEXTANT — the acquisition layer

*What `src/seti/sextant/acquire.py` gets, from where, in what shape, and what it
still does not know. Written 2026-08-25; nothing in it has been run against the
live service yet — that is what `sextant-probe` is for.*

## The source

`https://gea.esac.esa.int/tap-server/tap` — IVOA TAP, public, no auth, `pyvo`
already a dependency. Measured on the runner 2026-08-25:

| table | rows | columns | objects |
|---|---|---|---|
| `gaiafpr.sso_observation` | 46,264,083 | 34 | 156,823 |
| `gaiadr3.sso_observation` | 23,336,467 | 35 | 158,152 |
| `gaiafpr.sso_source` | 156,823 | — | — |
| `gaiadr3.sso_source` | 158,152 | — | — |

The two observation tables share **32** columns; DR3 adds `g_mag`/`g_flux`/
`g_flux_error` and FPR adds `is_rejected`/`fov`. That arithmetic — 32 + 3 = 35
and 32 + 2 = 34 — reproduces the measured column counts, and is the only offline
evidence that the column tuple in `acquire.py` is *complete*.
`test_column_inventory_reproduces_the_measured_counts` pins it, so a column
trimmed to save bandwidth fails the suite instead of quietly disappearing from a
46-million-row pull.

## The change of question (2026-08-25)

Screening Gaia's *surviving* SSO residuals is **not novel**. The Gaia SSO team is
running it: Liberato, Tanga, Mary et al., *Follow the wobble* (arXiv:2605.22702)
projects post-fit residuals from `gaiafpr.sso_observation` onto the along-scan
axis and reports 343 candidates; Dziadura, Bartczak & Oszkiewicz (A&A 693, A31;
arXiv:2411.09750) fitted non-gravitational `A2` to 54,094 inner-main-belt bodies
on the same data.

Every one of those searches works **post-fit**. An object whose astrometry *fails*
to fit is removed before their first statistic is computed, so it is invisible to
all of them. SEXTANT's question is therefore whether some objects are
systematically **un-fittable**, and `is_rejected`, `astrometric_outcome_ccd` and
`astrometric_outcome_transit` are the measurement rather than a filter.

Three things follow, and they are the design of this module:

1. **No ADQL built here ever narrows on a rejection flag.** The rows come back
   labelled and the caller decides. `QualityCuts.drop_rejected` defaults to
   `False` — reversing the obvious default deliberately, because a helper that
   discarded those rows would make the channel impossible *and the loss would be
   invisible*. `test_no_data_pull_ever_filters_on_a_rejection_flag` pins it.
2. **The denominator travels with the numerator.** `rejection_census()` returns,
   per object, one row per `(number_mp, flag value)` — attempts, failures and the
   full code breakdown in one `GROUP BY`, not by downloading 46M rows to count
   them. A rate needs its attempts, exactly as TOCSIN's event rate needs its
   forced-photometry denominator. `rejection_ledger()` folds it client-side and
   deliberately computes **no rate**: which codes are failures is unverified, and
   inventing one would bake in the assumption the channel exists to test.
3. **The codes stay data.** Nothing is hard-coded as "good". The probe measures
   the distribution per table and per object, and `interpret_rejection_fraction()`
   compares the global `is_rejected` fraction against the published Gaia outlier
   fractions (~0.58% DR3, ~1% DR2) — the check that says whether the column marks
   what we think it marks. At that rate the rejected set is 10⁵–10⁶ rows, which is
   comfortably fetchable and is what makes the screen possible at all.

The residual path is still fully carried: `position_angle_scan`, the observer
state vectors, and the two-part error model are all pulled, and
`check_columns_for_residuals()` refuses a pull that is missing any of them.

## The chunking

**The chunk key is `number_mp`.** A residual is a time series and every fit the
next stage runs needs an object's whole arc, so chunking on the object hands back
complete, immediately usable objects and lets a run stop and resume at any chunk
boundary with everything already downloaded still valid; chunking on `epoch` hands
back fragments of every object alive at that time, so nothing can be fitted until
the entire 46-million-row download has completed and been regrouped — an
all-or-nothing job on a runner with a hard timeout.

Boundaries come from the **actual sorted object list** (`plan_object_chunks`), not
from a stride in the index: Gaia's sample concentrates in the low-numbered bright
asteroids while MPC numbers run past 600,000, so a fixed stride would put almost
everything in the first chunk. And because the plan still cannot be trusted to be
correctly *sized*, `iter_observation_chunks` checks every result for overflow and
**bisects and retries** any chunk that hits `maxrec` — a truncated chunk is short,
not empty, and nothing downstream could tell. `chunks_cover()` audits a plan for
gaps and overlaps offline: a gap loses observations and reads as a sparser
catalogue, an overlap double-counts them and reads as a smaller residual scatter,
and both survive every downstream test.

`plan_epoch_chunks` / `observations_in_epoch_range` exist for the questions that
really are about time (coverage, cross-release comparison, restarting a partial
download) and say what they cost.

## Async by default

The ESA Gaia TAP service is believed to cap **synchronous** anonymous queries at a
few thousand rows. If that is true and this module used the synchronous path,
every bulk pull would return silently truncated and the channel would screen
0.004% of the catalogue while reporting success. So `GaiaSSO.query`/`fetch` are
asynchronous, `query_sync` is opt-in and used only for tiny metadata reads, and
every result is compared against the requested `maxrec`. The probe measures the
real cap rather than trusting the number.

Transport handling is composed from `seti.tocsin.brokers.AlerceTAP` — the real
per-request socket timeout, the exponential-backoff retry, the fast-fail on ADQL
errors — rather than re-derived, exactly as `seti.loom.acquire.AlerceSSO` does it.

## DR3 vs FPR: unsettled, and not guessed

46,264,083 / 23,336,467 = 1.98, against FPR's ~66-month window and DR3's ~34 whose
ratio is 1.94. That is *suggestive* that FPR is a re-reduction over a longer
baseline and therefore a superset — it is **not a measurement**, and both possible
errors are severe: treat an increment as a superset and half of every arc is
thrown away; treat a superset as an increment and every shared observation is
counted twice, halving each object's apparent scatter and inflating every S/N by
√2.

So `reconcile_observations()` decides from evidence in the rows. If the dedup key
matches across releases, the union deduplicates correctly. If it matches nothing
*while the epoch ranges overlap*, it returns `KEY_DISJOINT_BUT_EPOCHS_OVERLAP` and
**refuses to produce a union** unless `allow_unverified_union=True`. Two key
strategies: `observation_id` (the obvious one, but it does not survive a
re-minting) and `transit_ccd` — `(number_mp, transit_id, epoch rounded to 1e-6 d)`
— which keys on the physical event and does.

**TODO(SEXTANT-Q7/Q8):** the probe asks it three ways (per-object row counts in
each release, a cross-schema join on `observation_id`, a cross-schema join on
`(transit_id, number_mp)`) and dumps one object's verbatim ids from each release so
a human can compare them by eye if the joins are refused. Once it has run, the
answer belongs in the `reconcile_observations` docstring as a measurement and the
default policy becomes a decision rather than a negotiation.

## What is still assumed

`acquire.OPEN_QUESTIONS` is the machine-readable version; every entry names the
probe key that settles it, and `test_every_open_question_is_measured_by_the_probe`
asserts the probe really produces that key, so a question cannot be recorded and
then quietly never asked. The ones that would silently produce a wrong result:

| question | assumed | how it fails silently |
|---|---|---|
| synchronous row cap | 2000 | every bulk pull truncates and looks successful |
| state-vector frame | ICRS equatorial | 23.44° rotates along-scan into across-scan |
| `epoch` zero point / time scale | days from 2010.0 TCB | TCB−TDB ≈ 10 s → 3.5 mas coherent along-track shift over the whole catalogue |
| outcome-code meanings | nothing | a cut on a guessed good-value removes 3% or 99%, both of which complete |
| `is_rejected` type and rate | bool, ~0.58% | `bool('false')` is `True` — a naive drop discards every FPR row |
| rows per transit | ~9 (one per AF CCD) | they share an attitude and a scan angle; σ/√N understates the transit error 3× |
| rejection denominator | rows written = transits attempted | if a failed transit writes no row, the denominator is censored *with* the signal |
| `number_mp` completeness | always populated | a NULL is skipped by every `number_mp` chunk |
| `ra_error_*` and cos(δ) | true angle | mis-scales RA error by 1/cos δ and tilts the error ellipse |
| DR3 ⊂ FPR | superset | see above |

The frame test is worth singling out because it is cheap and decisive: Gaia sits
at L2, on the Sun–Earth line, which lies in the ecliptic. In an **ecliptic** frame
its barycentric `z` never leaves a thin slab about zero; in an **equatorial** frame
`z = y_ecl·sin(23.44°)` sweeps ±0.4 AU once a year. `MIN`/`MAX` of `z_gaia`
separates the two by two orders of magnitude —
`interpret_state_vector_frame()` reads it.

## Running it

```
python -m seti.cli sextant-probe        # stage 0, runner-only: the go/no-go
```

Workflow: `.github/workflows/sextant-probe.yml` (dispatch-only, 90-minute
timeout). It writes `results/sextant/probe.json` **after every query**, so a
cancelled job — which never runs its commit step — still leaves everything it
learned on disk, and commits the record verbatim so a schema change appears as a
diff in version control rather than as an unexplained null months later.

## Files

| file | what |
|---|---|
| `src/seti/sextant/acquire.py` | the client, the chunk planner, the quality/rejection helpers, the cross-release reconciler, and `probe()` |
| `tests/test_sextant_acquire.py` | 58 offline tests; no network |
| `.github/workflows/sextant-probe.yml` | the runner-only probe |

The residual computation, the screen and the population stage are separate modules
and are not this one's business: `acquire.py` pulls rows, labels them, and decides
nothing.
