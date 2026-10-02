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
classification. This is an analytic synthetic witness, not a observed sky false
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

## Validation and stop condition

Actual final scoped/full CI and independent review receipts will be committed
after they settle on the exact source. Tests cover the documented production
counterexample, every trial's circular acceptance measure by deterministic
quadrature, overlap/saturation, invalid geometry, zero episodes, matched/unmatched
controls and epoch-order invariance. The existing real BLS detached-EB positive
control remains in the scoped suite.

Stop after one reviewed implementation, passing actual scoped/full CI, normal
main integration and committed exact receipts. No indefinite monitoring.

## Next bounded task

Establish a proper-motion-aware catalogue association requirement for the VSX
first-row-within-10-arcsec path before assigning a Gaia survivor an eclipsing
mechanism. Separately, calibrate the conditional analytic phase bound against
Gaia sampling using scientifically justified controls; no calibration claim is
made by this patch.
