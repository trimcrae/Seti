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
| vetted | 4 |
| unexplained_after_vetting | 0 |

## Comet control sample (should light up)

| stage | n |
|---|---|
| input | 272 |
| a1_fitted | 272 |
| a1_positive | 240 |
| a1_significant | 210 |
| nonradial_constrained | 110 |
| screen1_a1_only | 0 |
| screen1_a1_only_strict | 0 |
| r_computable | 0 |
| screen2_r_flag | 0 |
| screen2_r_strong | 0 |
| screen2_r_extreme | 0 |
| screen3_negative_a1 | 25 |
| screen4_albedo | 0 |
| gate_fail_a1_not_significant | 62 |
| gate_fail_a2_nonzero | 235 |
| gate_fail_a3_nonzero | 107 |
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

## Degradation

- bulk pull lacks usable sigma_A1 (or minimal-field fallback used); enriching 22 of 22 objects per-object from sbdb.api (orbit.model_pars carries A1/A2/A3 AND their sigmas) before screening

## Survivors (top by R)

| object | A1/sigma | R | AMR (m^2/kg) | verdict |
|---|---|---|---|---|
|        (2012 LA) | 5.66 | 2.05 | 0.000296 | SHORT_ARC_ARTEFACT |
|        (2021 VH2) | 3.76 | 1.43 | 0.000359 | ARTIFICIAL_HUMAN_SUSPECT |
|        (2020 CD3) | 54.2 | 0.615 | 0.000599 | ARTIFICIAL_HUMAN_SUSPECT |
|        (2021 GM1) | 5.2 | 0.484 | 0.000254 | ARTIFICIAL_HUMAN_SUSPECT |

_No candidate here is a detection claim. Every survivor is a systematic until traced; see `docs/derelict.md`._
