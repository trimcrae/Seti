# DERELICT — thin-film debris via radiation-pressure acceleration

**Verdict:** `NO_SURVIVORS`

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
| screen1_a1_only | 0 |
| screen1_a1_only_strict | 0 |
| r_computable | 1 |
| screen2_r_flag | 0 |
| screen2_r_strong | 0 |
| screen2_r_extreme | 0 |
| screen3_negative_a1 | 1 |
| screen4_albedo | 0 |
| cometary_law_excluded | 20 |
| vetted | 0 |
| unexplained_after_vetting | 0 |

## Comet control sample (should light up)

| stage | n |
|---|---|
| input | 272 |
| a1_fitted | 272 |
| a1_positive | 240 |
| a1_significant | 0 |
| nonradial_constrained | 0 |
| screen1_a1_only | 0 |
| screen1_a1_only_strict | 0 |
| r_computable | 39 |
| screen2_r_flag | 0 |
| screen2_r_strong | 0 |
| screen2_r_extreme | 0 |
| screen3_negative_a1 | 0 |
| screen4_albedo | 0 |
| cometary_law_excluded | 44 |

## Degradation

- bulk pull lacks usable sigma_A1 (or minimal-field fallback used); enriching 22 of 22 objects per-object from sbdb.api (orbit.model_pars carries A1/A2/A3 AND their sigmas) before screening

_No candidate here is a detection claim. Every survivor is a systematic until traced; see `docs/derelict.md`._
