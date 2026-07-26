# DERELICT — thin-film debris via radiation-pressure acceleration

**Verdict:** `ALL_SURVIVORS_EXPLAINED`

## Conversions used

- `beta = 3379.38 * A1[au/day^2]`
- `AMR  = 4.41375e+06 * A1[au/day^2]` m^2/kg at Q_pr = 1.0
- `AMR_natural = 3 / (2 D rho)`, rho = 1000.0 kg/m^3

## Funnel

| stage | n |
|---|---|
| input | 22 |
| a1_fitted | 22 |
| a1_positive | 19 |
| a1_significant | 11 |
| nonradial_constrained | 10 |
| screen1_a1_only | 4 |
| screen1_a1_only_strict | 0 |
| r_computable | 19 |
| screen2_r_flag | 0 |
| screen2_r_strong | 0 |
| screen2_r_extreme | 0 |
| screen3_negative_a1 | 1 |
| screen4_albedo | 0 |
| gate_fail_a1_not_significant | 11 |
| gate_fail_a2_nonzero | 13 |
| gate_fail_a3_nonzero | 9 |
| gate_fail_orbit_quality | 1 |
| gate_fail_coma | 0 |
| gate_fail_outgassing | 0 |
| outgassing_evidence_excluded | 0 |
| law_marsden_g | 20 |
| law_inverse_square | 2 |
| screen3_negative_a1_asteroid | 1 |
| screen3_denominator_asteroid | 22 |
| screen3_negative_a1_comet | 24 |
| screen3_denominator_comet | 264 |
| screen4_albedo_catalogue_wide | 214 |
| screen4_albedo_confirmed_two_sources | 0 |
| vetted | 4 |
| unexplained_after_vetting | 0 |

## Comet control sample (should light up)

| stage | n |
|---|---|
| input | 272 |
| a1_fitted | 272 |
| a1_positive | 240 |
| a1_significant | 210 |
| nonradial_constrained | 108 |
| screen1_a1_only | 0 |
| screen1_a1_only_strict | 0 |
| r_computable | 0 |
| screen2_r_flag | 0 |
| screen2_r_strong | 0 |
| screen2_r_extreme | 0 |
| screen3_negative_a1 | 24 |
| screen4_albedo | 0 |
| gate_fail_a1_not_significant | 62 |
| gate_fail_a2_nonzero | 236 |
| gate_fail_a3_nonzero | 105 |
| gate_fail_orbit_quality | 101 |
| gate_fail_coma | 272 |
| gate_fail_outgassing | 272 |
| outgassing_evidence_excluded | 272 |
| law_marsden_g | 20 |
| law_inverse_square | 252 |

## Vetting

| verdict | n |
|---|---|
| ARTIFICIAL_HUMAN_SUSPECT | 3 |
| SHORT_ARC_ARTEFACT | 1 |

## Is the A1 census complete?

**CONSTRAINT_COMPLETE**

| kind | constrained | unconstrained rows | non-null A1 | missing | extra | verdict |
|---|---|---|---|---|---|---|
| asteroid | 22 | 1553263 | 22 | 0 | 0 | CONSTRAINT_COMPLETE |
| comet | 272 | 4069 | 272 | 0 | 0 | CONSTRAINT_COMPLETE |

## Screen 3 — the sunward-acceleration floor

| population | negative | denominator (A1 fitted) | rate | max \|R\| | median \|R\| |
|---|---|---|---|---|---|
| asteroid | 1 | 22 | 0.0455 | 5 | 5 |
| comet | 24 | 264 | 0.0909 | 9.47e+05 | 9.03e+04 |

Flag threshold R = 10; measured floor max |R| = 9.47e+05. Radiation pressure cannot push sunward, so every row above is a systematic.

## Dark-comet named-target census

- targets: 14, resolved: 14, unresolved: 0
- with a JPL-fitted A1 (value + sigma): 8
- A1-only (screen 1 pass): 0
- no A1 fitted at all: 1998 FR11, 2001 ME1, 2003 RM, 2005 UY6, 2010 VL65, 2012 UR158

## Screen 4 — catalogue-wide albedo (independent of A1)

| population | status | strategy | rows returned | above cut | confirmed by IRSA |
|---|---|---|---|---|---|
| asteroid | OK | cdata_albedo_gt | 214 | 214 | 0 |
| comet | OK | cdata_albedo_defined | 19 | 0 | - |

## Queries issued

| status | n |
|---|---|
| QUERY_FAILED | 53 |
| OK | 309 |
| QUERY_RETURNED_ZERO_ROWS | 2 |

_Every query is recorded verbatim (URL, HTTP status, row count) in `queries.json`. `QUERY_FAILED` and `QUERY_RETURNED_ZERO_ROWS` are different statements and are never merged._

## Degradation

- completeness[asteroid]: the constrained and unconstrained designation sets are identical: the A1 census is complete for this kind.
- completeness[asteroid]: counts: constrained=22, unconstrained rows=1553263, of which non-null A1=22
- completeness[comet]: the constrained and unconstrained designation sets are identical: the A1 census is complete for this kind.
- completeness[comet]: counts: constrained=272, unconstrained rows=4069, of which non-null A1=272
- bulk pull lacks usable sigma_A1 (or minimal-field fallback used); enriching 22 of 22 objects per-object from sbdb.api (orbit.model_pars carries A1/A2/A3 AND their sigmas) before screening

## Survivors (top by R)

| object | A1/sigma | R | AMR (m^2/kg) | verdict |
|---|---|---|---|---|
|        (2012 LA) | 5.66 | 2.05 | 0.000296 | SHORT_ARC_ARTEFACT |
|        (2021 VH2) | 3.76 | 1.43 | 0.000359 | ARTIFICIAL_HUMAN_SUSPECT |
|        (2020 CD3) | 54.2 | 0.615 | 0.000599 | ARTIFICIAL_HUMAN_SUSPECT |
|        (2021 GM1) | 5.2 | 0.484 | 0.000254 | ARTIFICIAL_HUMAN_SUSPECT |

_No candidate here is a detection claim. Every survivor is a systematic until traced; see `docs/derelict.md`._
