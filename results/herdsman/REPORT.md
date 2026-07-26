# HERDSMAN run report

Sample: 1260759 stars (fetched 2614640, 6D 2614640, sigma_v <= 0.8 km/s: 1263667); median sigma_v = 0.57 km/s.

Direction summaries (the backward scan doubles as the time-reversal control for the forward scan — phase-mixed dynamics is statistically time-symmetric, deliberate future assembly is not):

| direction | horizon (Myr) | epochs | raw | candidates | best surprise | p_global |
|---|---|---|---|---|---|---|
| forward | 11.0 | 10 | 0 | 0 | 0.00 | 1.000 |
| backward | 12.0 | 11 | 1 | 0 | 0.00 | 1.000 |

## Candidates

None above threshold in either direction at these settings.

## Honest sensitivity statement

The scan is sensitive only inside the self-computed horizon above (where an error-matched meeting ball still holds < 0.5 field stars by chance). Herds converging beyond that horizon, herds of fewer than 4 members, and herds whose members fall outside the precision subset are NOT probed. A null here is a statement about this domain only and is not to be written up as a result; it is a reason to deepen the sample (better RVs, DR4) or change the question.

## Session verdict (v2 full-depth, run 30203142288)

This is the definitive HERDSMAN-A v2 result: the earlier same-subject commit from run
30199588771 was invalid (every scan job crashed out-of-memory on a 43,692-star percolation
component and stale v1 shard checkpoints leaked into its reduce; see the fix commit).

- Configuration: d_max = 2 kpc (1,260,759 precision-6D stars), G < 14.5,
  sigma_v <= 0.8 km/s, herd-physics cuts sigv_int <= 5 km/s (arrival coldness) and
  min_epochs >= 2 (meeting persistence), rec_every = 4 (1.0 Myr epoch spacing),
  24 velocity-shuffled mocks per direction.
- Result: zero candidates in both time directions within the 11-12 Myr chance-occupancy
  horizons; forward raw = 0, backward raw = 1 (single-epoch, failed persistence).
  All 48 mocks behave identically (raw 0-1, zero passing) — the real sky is
  indistinguishable from its phase-randomized controls.
- Positive control at these exact production parameters: a synthetic 8-star herd with
  1.5 km/s arrival dispersion injected into the test sky is recovered at its true
  meeting epoch with all 8 members (surprise 17.3, sig_int 4.9 km/s, 3 epochs) —
  the null is a statement about the sky, not about detector sensitivity.
- Interpretation: the v1 p ~ 0.02 excess is confirmed as time-symmetric Galactic
  stream structure, not directed assembly. No cold, persistent, dispersed-today
  N >= 4 rendezvous exists among the 1.26M best-measured stars within 2 kpc over
  the past/future ~12 Myr. Per project policy this closes the question at DR3
  precision rather than becoming a writeup; the search moves to the sister
  channels (HERDSMAN-B spectroscopic, MIDDEN) and to DR4 when available.
