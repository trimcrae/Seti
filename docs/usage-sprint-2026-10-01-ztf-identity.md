# PARALLAX4 ZTF catalogue identity — 2026-10-02

AI-authored implementation with independent review coordinated by the usage sprint.
Base: `ee57cf1891e82e615b2b4deae4677946870e977c`.
Branch: `codex/usage-sprint-2026-10-01-ztf-identity`.

## Bounded question

Can a multi-object ZTF cone be stitched into one light curve and incorrectly
used as an eclipse mechanism for a Gaia survivor? Previously yes: acquisition
returned all good-quality cone rows and the fitter selected a filter without
checking its catalogue object IDs.

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

Pending actual scoped and full GitHub CI at the source revision, independent
review, and normal main merge. No local Python execution is claimed.
The scoped suite includes a two-object cone with constant magnitudes per ID
whose stitched sampling resembles an eclipse; an explicit assertion prevents
such a cone from reaching BoxLeastSquares. Existing detached-EB positive control
now declares one exact object ID. Other controls cover adjacent large IDs above
2**53, null/malformed/float identifiers and separate band-specific IDs.

Stop condition: reviewed implementation, passing actual scoped/full CI and main
integration; no acquisition or science-run dispatch.

## Next bounded task

Check the BLS period/half/double-period phase-window chance calculation against
each actual alias window and Gaia sampling. Current analytic phase bound assumes
independent uniform epoch phases; this patch does not calibrate or revise it.
Separately, catalogue association (especially VSX first row within 10 arcsec)
needs proper-motion-aware identity evidence before assigning a target mechanism.
