# PARALLAX4 ZTF catalogue identity — 2026-10-02

AI-authored implementation with independent review coordinated by the usage sprint.
Base: `ee57cf1891e82e615b2b4deae4677946870e977c`.
Branch: `codex/usage-sprint-2026-10-01-ztf-identity`.

## Bounded question

Can a multi-object ZTF cone be stitched into one light curve and incorrectly
used as an eclipse mechanism for a Gaia survivor? Previously the fitter could
pool them: acquisition returned all good-quality cone rows and selected a filter
without checking its catalogue object IDs. This is a code-path finding, not an
observed sky false positive.

The selected band now requires one exact positive integer `oid`. CSV parsing
keeps IDs as strings; rounded floats, missing/malformed IDs and mixed IDs stop
before the fit. The report retains input/band row counts and exact object IDs.
Ambiguity gets `ZTF_AMBIGUOUS_OBJECTS` or `ZTF_IDENTITY_UNRESOLVED`; absent
other independent explanations, fate is `EVIDENCE_INCOMPLETE` with a flag.
The deep-vet summary counts incomplete evidence separately.

Multiple IDs can be field-specific representations of the same physical source.
This check does not label them physical contamination. A single ID establishes
a coherent catalogue series; it does **not** prove Gaia counterpart association.
Nearest-row selection is intentionally absent. Historical science outputs,
thresholds, candidate fates and existing bot work are untouched.

## Validation / integration

Reviewed source: `5a6b07e2bdc5d05932bc2a0a88ca418a21f89929`.
[PR #24](https://github.com/trimcrae/Seti/pull/24) merged that reviewed source via
`28770826fcd782c69dad3ca0ea34ccb6055eeafb`, preserving the prior main head.
The merge tree equals the reviewed source tree.

Actual [scoped CI 36946634335](https://github.com/trimcrae/Seti/actions/runs/36946634335),
job `110650147306`, passed lint and **141 cases** (121 existing + 20 new).
Runner: Python 3.11.16, pandas 3.0.6, astropy 8.0.1.
Actual [full CI 36946634201](https://github.com/trimcrae/Seti/actions/runs/36946634201)
passed both offline `110650147213` (lint, complete network-guarded suite,
sample reproduction/forecast and upload) and paper `110650146902`.
Actions declares source head `5a6b07e2...`; its PR checkout
`c83796a8070dd177fd7c7d944ee9ce06b4ad7a4a` has the exact same tree
`5821230ef5261af2b7378213ed086c7a92cdbd8e`. Existing full-suite skips
are retained; no local Python execution or sky experiment is claimed.

Independent code review: `/root/sprint_apps_review`; independent scientific
review: `/root`. Both also observed real CI success. This checkpoint changes
handoff/receipt documentation only; all five implementation/test/workflow blob
hashes are preserved in [the machine-readable receipt](usage-sprint-2026-10-01-ztf-identity-receipt.json).
Final checkpoint CI/integration state is recorded in PR #24; these receipts
specifically describe the tested source above, not future revisions.

The scoped suite includes a two-object cone with constant magnitudes per ID
whose stitched sampling resembles an eclipse; an explicit assertion prevents
such a cone from reaching BoxLeastSquares. Existing detached-EB positive control
now declares one exact object ID. Other controls cover adjacent large IDs above
2**53, null/malformed/float identifiers and separate band-specific IDs.

Stop condition met: reviewed implementation, passing actual scoped/full CI and
main source integration. No acquisition or science-run dispatch occurred.

## Next bounded task

Check the BLS period/half/double-period phase-window chance calculation against
each actual alias window and Gaia sampling. Current analytic phase bound assumes
independent uniform epoch phases; this patch does not calibrate or revise it.
The reviewer supplied a deterministic alias counterexample, independently
recomputed in V8: P = 1 d, duration = 0.1 d, phase_tol = 0.03, six dips gives
the current three-trial expression 0.0032212; the P/2 accepted-window fraction
is 0.52, so its independent-uniform chance alone is 0.0197706. This is
analytical evidence of an understated bound, not a measured false-positive
rate or a new claim about any retained sky candidate.
Separately, catalogue association (especially VSX first row within 10 arcsec)
needs proper-motion-aware identity evidence before assigning a target mechanism.
