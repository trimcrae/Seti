# PARALLAX4 ZTF alias phase bound — 2026-10-02

AI-authored implementation; independent science and code review coordinated by
the usage sprint. Base: `ace1e6d631d4c49cfacf10d4e9cf126b446b138d`. Branch:
`codex/usage-sprint-2026-10-01-ztf-alias-bound`.

## Demonstrated defect and bounded repair

The original phase chance expression used the base-period acceptance width for
all three trials (P, 2P, P/2). The fixed period-half trial actually accepts a
wider phase fraction. At P=1 day, duration=0.1 day, tolerance=0.03 and six Gaia
episodes, the old expression is 0.003221225472, below the unchanged 0.01
classification threshold; P/2 alone has conditional chance 0.019770609664.

Actual baseline [CI 36951305026](https://github.com/trimcrae/Seti/actions/runs/36951305026),
job `110664635451`, on `275076c1f63e25caa3d323fff7fa5d23e9cf006d` executed the
production phasing path with a fixed synthetic BLS ephemeris. Its original
transcript reports both values and the resulting `ZTF_ECLIPSING_PHASED`
classification. This is an analytic synthetic witness, not an observed sky false
positive or an empirical false-positive rate.

For each trial period, two circular windows are centered at primary phase 0 and
secondary phase 0.5. Their half-width is duration/(2*trial_period)+tolerance.
Their union covers min(1,4*half_width) of the phase circle: clipping accounts for
touching/overlapping windows. The repaired bound is the sum of each trial's
coverage raised to the number of Gaia episodes, capped at 1. The witness becomes
0.020957731392 and `ZTF_ECLIPSING_PHASE_AMBIGUOUS`.

The acceptance test consumes the same per-trial window widths used by the bound.
Outputs retain period, width, coverage, episode count and conditional chance for
every trial, plus the named analytic method and null assumption. The three trials
can be mutually dependent; the union bound does not assume alias independence.

## Scientific limits

The within-trial calculation **does assume independent, uniform Gaia episode
phases conditional on a fixed ZTF-derived ephemeris**. Gaia's actual scanning
cadence, correlated/duplicated episodes and fitted ephemeris uncertainty need
separate calibration. This repair does not establish a sky false-alarm
probability, target association, candidate detection, or any measured failure
rate. Missing Gaia episodes and unmatched phases never promote a phased veto.
The previous exact ZTF object-identity guard is preserved.

No historical candidate output, scientific threshold, preregistration, shared
queue or bot ledger is changed. No catalogue acquisition/science job is launched.

## Validation / integration

Reviewed and tested source: `d1f5780de8f3ae541aba0c17b9733890b64e7705`; source tree
`69b7a9ff559625fcbf225e641bc909748d0a8a68`. [PR #25](https://github.com/trimcrae/Seti/pull/25)
merged normally via `ab804c8204a87b2c3db28da41901410b251bb79a`.

Actual [scoped CI 36951545479](https://github.com/trimcrae/Seti/actions/runs/36951545479),
job `110665368935`, passed lint and **173 cases** in 50.29 s
(141 inherited + 32 new), including the existing real BLS detached-EB positive
control. Runner: Python 3.11.16, numpy 2.4.6, pandas 3.0.6, astropy 8.0.1.

Actual [full PR CI 36951571064](https://github.com/trimcrae/Seti/actions/runs/36951571064)
passed offline `110665447332` (lint, complete network-guarded suite,
sample reproduction/forecast and upload) and paper `110665447159`
(build and upload). The declared source is `d1f5780...`; actual PR checkout
`6db2a1fcc5e573e398c9042c64e8609f92d11319` has the exact source tree above.
Existing full-suite skips are retained; the suppressed summary gives no full
count, so none is inferred. The duplicate [full push CI 36951545526](https://github.com/trimcrae/Seti/actions/runs/36951545526)
also passed offline `110665698816` and paper `110665698603` on the exact
source checkout. No additional test or science dispatch was needed.

Independent code/evidence review: `/root/apps_round3_review`; independent
scientific/code review: `/root`. Both independently observed exact-source
scoped/full success and cleared the implementation. The main branch's
`f9b50202839a4661afea1f76022f9bcb6b153d0f` bot advance changes only
`results/cronwatch/status.json` and `results/watchdog/status.json`.
The normal merge preserved both verbatim; compared with the tested source,
those two operational files are its only differences. No force push occurred.

The receipt checkpoint changes documentation/transcripts only; all tested
implementation, test and workflow blob hashes are retained in
[the machine-readable receipt](usage-sprint-2026-10-01-ztf-alias-bound-receipt.json).
Original baseline/scoped/full-offline transcripts are preserved under
`docs/usage-sprint-2026-10-01-ztf-alias-bound-logs/`.
Automatic CI on subsequent documentation/main heads is separate from these
completed exact-source results and is not claimed complete by this receipt.

Tests cover the documented production counterexample, every trial's circular
acceptance measure by deterministic quadrature, overlap/saturation, invalid
geometry, zero episodes, matched/unmatched controls and epoch-order invariance.

Stop condition met: one reviewed implementation, passing actual scoped/full CI,
normal main source integration and an exact receipt checkpoint. No catalogue
acquisition, sky experiment, paid job or local Python execution is claimed.

## Next bounded task

Establish a proper-motion-aware catalogue association requirement for the VSX
first-row-within-10-arcsec path before assigning a Gaia survivor an eclipsing
mechanism. Separately, calibrate the conditional analytic phase bound against
Gaia sampling using scientifically justified controls; no calibration claim is
made by this patch.
