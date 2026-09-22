# SHROUD run report

Signature S33: optically vanished sources **with** an infrared
counterpart = enshrouded, not destroyed (docs/shroud.md).

**Scoping.** This channel analyses the *catalogue by-product* of
Solano, Villarroel & Rodrigo 2022 — the optical-absent / IR-present
crossmatch. It does not use, replicate, or depend on the contested
VASCO transient, Earth-shadow or nuclear-test analyses.

**Verdict:** `VIZIER_FALLBACK`  (degraded: True)

> the Solano+2022 SVO archive did not answer a cone probe; only the Villarroel+2020 surviving-candidate list (127 rows) was retrieved; population fractions from it are indicative only

Sample: **127** sources.

| sample | rows |
|---|---:|
| vasco2020_surviving_candidates | 127 |

## Funnel

| stage | n |
|---|---:|
| 1_sample | 127 |
| 2_no_modern_optical_within_5arcsec | 123 |
| 3_with_any_ir_detection | 7 |
| 4_ir_present_and_optically_absent | 3 |
| 5_residual_after_population_cascade | 5 |
| 6_survive_every_veto | 1 |
| 7_energy_conserving_obscuration | 0 |
| 7_ir_too_faint | 0 |

## Population breakdown

The first population analysis of this sample.

| class | n | fraction |
|---|---:|---:|
| ASTEROID | 63 | 0.4961 |
| PLATE_DEFECT | 57 | 0.4488 |
| RESIDUAL_UNEXPLAINED | 5 | 0.0394 |
| VARIABLE_STAR | 1 | 0.0079 |
| MODERN_OPTICAL_MATCH | 1 | 0.0079 |

## Obscuration vs destruction

- optically vanished **with** an IR counterpart: 0
- optically vanished with **no** counterpart: 120
- raw ratio: -
- after subtracting the mundane classes: 0.0417

## Chance-match null (offset positions)

Measured, not assumed: the same sightlines displaced by 45.0".

- real match fraction: 0.0551
- chance match fraction: 0.0768
- genuinely associated fraction: -0.0217
- expected chance matches in the sample: 10
- significance: -0.9 sigma

## Energy-budget verdicts

| verdict | n |
|---|---:|
| INSUFFICIENT_IR | 95 |
| NO_HISTORICAL_PHOTOMETRY | 28 |
| IR_UNDERSAMPLED | 3 |
| ENERGY_CONSERVING_OBSCURATION | 1 |

Survivors of every kill-test: **1** (0 energy-conserving, 0 IR-too-faint).

3 object(s) have too few infrared bands for a budget verdict (`IR_UNDERSAMPLED`). The published `vanish-neowise` table carries W1/W2 only, so this count is the measure of how much the AllWISE W3/W4 + 2MASS join still owes.

| source | RA | Dec | class | eta_max | budget | FTK |
|---|---:|---:|---|---:|---|---|
| VASCO2020-table2-0041 | 157.61429 | 22.73810 | RESIDUAL_UNEXPLAINED | 0.0266 | IR_UNDERSAMPLED | DISAPPEARANCE_LIKE |

No-null rule (CLAUDE.md): an empty survivor list is a statement
about THIS sample and these thresholds, never a publishable result.
The population breakdown and the obscuration-to-destruction ratio
are the standing measurements regardless.

