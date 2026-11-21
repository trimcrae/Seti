# ZTF acquisition: intermittent IRSA failures, 2026-10-22

Issue: https://github.com/trimcrae/Seti/issues/45

The October 21 diagnostic is an unmeasured request matrix, not an astronomical
null. This investigation reproduced timeouts and subsequently obtained valid
responses from the unchanged endpoint. The failure is intermittent; these
measurements do not locate its cause inside IRSA versus the intervening network.
Changing the endpoint or its query syntax is not established as the remedy.

## Measurements

Evidence is in `results/ztf_issue45/`, separate from the scheduled probe and ledgers.
The qualification probe started October 22 at 17:08:51 EDT (21:08:51 UTC).

| Request | Observed result |
| --- | --- |
| IRSA covered cone, CSV and VOTable, initial short matrix | Read timeouts |
| IRSA southern control | HTTP 200, valid CSV header, no rows |
| IRSA documented ID 686103400034440 | HTTP 200, 214,915 bytes of CSV photometry |
| IRSA supported CIRCLE parameter at covered position | HTTP 200, 72,198 bytes; does not establish that POS is broken |
| IRSA original POS query, qualification probe | HTTP 200, 282 CSV rows; VOTable also HTTP 200 |
| IRSA southern control, qualification probe | HTTP 200, expected columns, zero rows |
| Patched ZtfIrsa.lightcurves at covered position | 282 epochs, 263 good after catflags, zg/zr/zi |
| ALeRCE covered cone, radius 2 arcsec | HTTP 200, three broker object IDs |
| ALeRCE detections for documented example ZTF18abbuksn | HTTP 200, 1,044 rows |
| ALeRCE v2 forced photometry for the same example | HTTP 200, 286 rows, all carrying that oid |
| Fink POST /api/v1/objects for the same example | HTTP 503, upstream connection timeout |

The IRSA covered field's per-band matchfile IDs were 473101200004287 (117 zg),
473201200005489 (139 zr), and 473301200007195 (26 zi). Their median coordinates
were approximately (187.27793, 2.05242), consistent across bands. This is a
qualifying control retrieval, not a completeness or identity guarantee for
arbitrary crowded-field targets. The live patched walk succeeded on its first
attempt; deterministic regression tests establish the timeout-then-success path.

## Correction and limits

The target walk now retries the **identical** archive query once after a requests
timeout/connection failure or HTTP 502/503/504. It records the transient failure
and checks the walk deadline before starting a retry. A second failure still
discards the target's partial segments and emits no curve. A throttled 429 is not
immediately retried. No broker observations are spliced into a matchfile curve;
coordinates, time windows, collection, band selection and proper-motion segments
are unchanged. The diagnostic probe remains a direct measurement, without hidden
retries. Archive-frontier acquisition is unchanged.

This mitigates intermittent failures; it cannot repair a sustained upstream
outage. As before, a requests read timeout bounds socket inactivity rather than
the full wall time of a trickling response; this patch does not strengthen that
existing limitation. The retry may cost one additional request timeout.

ALeRCE **can** serve object-resolved light curves, including forced photometry.
It is nevertheless not a validated drop-in replacement here: the cone returns
three broker identities, not one guaranteed target association; its detections
carry difference-image magnitudes and corrections, and the forced response uses
another schema. For the example, the first forced row has mag ~17.29 versus
mag_corr ~15.40, so mapping `mag` directly to the adapter's total-flux baseline
would be wrong. Full-history availability, epoch selection, signed difference
flux and reference-flux handling, quality flags and a Gaia/PM association rule
must be validated before such a fallback can supply this screen's numerator or
denominator. No empty broker cone would establish empty matchfile coverage.

The successful direct probe establishes sampled availability on October 22,
not sustained service health. Issue #45 remains open pending runner confirmation.
PR #43 changes scheduler notifications, not acquisition, and is independent.

## Validation

205 tests passed: `test_ztf_acquisition_retry.py`, `test_tocsin_altfeeds.py`,
`test_tocsin_altwalk.py`, and `test_tocsin_ztf_live.py`. Ruff, compile checks and
`git diff --check` passed. The live qualification probe and adapter result are
committed alongside the initial mixed-success matrix. No science ledger or
watermark was updated by this investigation.

Primary interface references:

- https://irsa.ipac.caltech.edu/docs/program_interface/ztf_lightcurve_api.html
- https://alerce.readthedocs.io/en/stable/_modules/alerce/ztf_search.html
