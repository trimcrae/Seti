# SHROUD run report

Signature S33: optically vanished sources **with** an infrared
counterpart = enshrouded, not destroyed (docs/shroud.md).

**Scoping.** This channel analyses the *catalogue by-product* of
Solano, Villarroel & Rodrigo 2022 — the optical-absent / IR-present
crossmatch. It does not use, replicate, or depend on the contested
VASCO transient, Earth-shadow or nuclear-test analyses.

**Verdict:** `VO_ARCHIVE`  (degraded: False)

Sample: **421043** sources.

| sample | rows |
|---|---:|
| solano2022_ir_present | 10458 |
| solano2022_no_counterpart | 10458 |
| vizier:I/284/out | 400000 |
| vasco2020_surviving_candidates | 127 |

## Sky coverage (USNO-B1.0 reconstruction)

- fields fetched: 200 / 200 of radius 0.5 deg = **157.08 deg^2**
- USNO-B1.0 rows returned (Ndet = 1, R1 <= limit): 714840
- POSS-I-red-only objects: 0

## How deep the search that found nothing actually went

An absence is only as good as the search behind it: a catalogue
that was never successfully queried leaves every source with an
empty magnitude, and empty reads as *gone*.

- modern catalogues that answered: gaia, ps1
- sources no modern catalogue covered: 20916
- required depth margin: 2.0 mag below the plate detection

## Funnel

| stage | n |
|---|---:|
| 1_sample | 421043 |
| 2_no_modern_optical_within_5arcsec | 90953 |
| 2b_absence_established_deeper_than_the_plate | 56727 |
| 2c_no_modern_catalogue_covered_the_position | 20916 |
| 3_with_any_ir_detection | 279906 |
| 4_ir_present_and_optically_absent | 3784 |
| 5_residual_after_population_cascade | 47470 |
| 6_survive_every_veto | 77 |
| 7_energy_conserving_obscuration | 21 |
| 7_ir_too_faint | 23 |

## Population breakdown

The first population analysis of this sample.

| class | n | fraction |
|---|---:|---:|
| MODERN_OPTICAL_MATCH | 183278 | 0.4353 |
| VARIABLE_STAR | 92735 | 0.2203 |
| ASTEROID | 66196 | 0.1572 |
| RESIDUAL_UNEXPLAINED | 47470 | 0.1127 |
| PLATE_DEFECT | 20973 | 0.0498 |
| BLEND_CONFUSION | 6699 | 0.0159 |
| HIGH_PM_STAR | 1737 | 0.0041 |
| AGN_QSO_COLOUR_ONLY | 961 | 0.0023 |
| AGN_QSO | 507 | 0.0012 |
| GALAXY | 487 | 0.0012 |

## Obscuration vs destruction

- optically vanished **with** an IR counterpart: 0
- optically vanished with **no** counterpart: 87169
- raw ratio: -
- after subtracting the mundane classes: 0.545

## Chance-match null (offset positions)

Measured, not assumed: the same sightlines displaced by 45.0".

- real match fraction: 0.657
- chance match fraction: 0.0843
- genuinely associated fraction: 0.572
- expected chance matches in the sample: 35501
- significance: 751.0 sigma

A single radius cannot tell a counterpart population from the
background: unrelated matches accumulate with the search area,
a genuine counterpart is already counted at the smallest radius.

| r (") | real matched | chance fraction | genuine fraction | sigma |
|---:|---:|---:|---:|---:|
| 1.0 | 184197 | 0.00339 | 0.434 | 566.8 |
| 1.5 | 209317 | 0.00756 | 0.49 | 633.0 |
| 2.0 | 225013 | 0.0134 | 0.521 | 673.2 |
| 3.0 | 248287 | 0.0304 | 0.559 | 726.8 |
| 4.0 | 265005 | 0.0541 | 0.575 | 752.6 |
| 5.0 | 276548 | 0.0843 | 0.572 | 751.0 |

Most significant radius: 4.0" at 752.6 sigma. Evidence only --- the selection radius is unchanged.

## Energy-budget verdicts

| verdict | n |
|---|---:|
| INSUFFICIENT_IR | 109048 |
| NO_DEFICIT | 101137 |
| NO_HISTORICAL_PHOTOMETRY | 86750 |
| ENERGY_CONSERVING_OBSCURATION | 83130 |
| IR_TOO_FAINT | 22712 |
| IR_TOO_FAINT_MARGINAL | 15419 |
| IR_UNDERSAMPLED | 2798 |
| IR_EXCEEDS_MISSING | 49 |

Survivors of every kill-test: **77** (21 energy-conserving, 23 IR-too-faint).

2798 object(s) have too few infrared bands for a budget verdict (`IR_UNDERSAMPLED`). The published `vanish-neowise` table carries W1/W2 only, so this count is the measure of how much the AllWISE W3/W4 + 2MASS join still owes.

| source | RA | Dec | class | eta_max | budget | FTK |
|---|---:|---:|---|---:|---|---|
| VASCO-VO-0006750 | 45.71873 | 1.46674 | RESIDUAL_UNEXPLAINED | 0.0152 | IR_UNDERSAMPLED | DISAPPEARANCE_LIKE |
| VASCO-VO-0012588 | 45.00016 | 1.36405 | RESIDUAL_UNEXPLAINED | 0.518 | ENERGY_CONSERVING_OBSCURATION | OBSCURATION_LIKE |
| VASCO-VO-0014734 | 45.47219 | 1.95766 | RESIDUAL_UNEXPLAINED | 0.0964 | IR_UNDERSAMPLED | DISAPPEARANCE_LIKE |
| VASCO-VO-0024526 | 47.22515 | 2.20694 | RESIDUAL_UNEXPLAINED | 0.275 | IR_TOO_FAINT_MARGINAL | DISAPPEARANCE_LIKE |
| VASCO-VO-0024527 | 47.22542 | 2.20664 | RESIDUAL_UNEXPLAINED | 0.091 | IR_TOO_FAINT | DISAPPEARANCE_LIKE |
| VASCO-VO-0025886 | 47.29109 | 2.26867 | RESIDUAL_UNEXPLAINED | 1.06 | ENERGY_CONSERVING_OBSCURATION | OBSCURATION_LIKE |
| VASCO-VO-0027408 | 46.89799 | 2.30209 | RESIDUAL_UNEXPLAINED | 0.793 | ENERGY_CONSERVING_OBSCURATION | OBSCURATION_LIKE |
| VASCO-VO-0027578 | 47.02053 | 2.44104 | RESIDUAL_UNEXPLAINED | 0.0427 | IR_UNDERSAMPLED | DISAPPEARANCE_LIKE |
| VASCO-VO-0033678 | 45.46264 | 2.23393 | RESIDUAL_UNEXPLAINED | 1.01 | ENERGY_CONSERVING_OBSCURATION | OBSCURATION_LIKE |
| VASCO-VO-0035039 | 45.24635 | 2.56479 | RESIDUAL_UNEXPLAINED | 0.104 | IR_TOO_FAINT_MARGINAL | DISAPPEARANCE_LIKE |
| VASCO-VO-0039737 | 46.81130 | 3.11187 | RESIDUAL_UNEXPLAINED | 0.00676 | IR_UNDERSAMPLED | DISAPPEARANCE_LIKE |
| VASCO-VO-0047390 | 43.43813 | 1.82221 | RESIDUAL_UNEXPLAINED | 0.0305 | IR_UNDERSAMPLED | DISAPPEARANCE_LIKE |
| VASCO-VO-0047392 | 43.43678 | 1.82262 | RESIDUAL_UNEXPLAINED | 0.0168 | IR_UNDERSAMPLED | DISAPPEARANCE_LIKE |
| VASCO-VO-0048161 | 43.18960 | 1.93076 | RESIDUAL_UNEXPLAINED | 0.00936 | IR_TOO_FAINT | DISAPPEARANCE_LIKE |
| VASCO-VO-0053220 | 44.04043 | 2.29235 | RESIDUAL_UNEXPLAINED | 0.356 | ENERGY_CONSERVING_OBSCURATION | OBSCURATION_LIKE |
| VASCO-VO-0054849 | 44.32510 | 2.53808 | RESIDUAL_UNEXPLAINED | 1.61 | ENERGY_CONSERVING_OBSCURATION | OBSCURATION_LIKE |
| VASCO-VO-0058049 | 42.97962 | 2.28820 | RESIDUAL_UNEXPLAINED | 0.000809 | IR_UNDERSAMPLED | DISAPPEARANCE_LIKE |
| VASCO-VO-0058252 | 42.90268 | 2.26441 | RESIDUAL_UNEXPLAINED | 0.137 | IR_TOO_FAINT_MARGINAL | DISAPPEARANCE_LIKE |
| VASCO-VO-0063494 | 42.97076 | 2.69942 | RESIDUAL_UNEXPLAINED | 0.542 | ENERGY_CONSERVING_OBSCURATION | OBSCURATION_LIKE |
| VASCO-VO-0087825 | 44.24878 | 3.32490 | RESIDUAL_UNEXPLAINED | 0.762 | ENERGY_CONSERVING_OBSCURATION | OBSCURATION_LIKE |
| VASCO-VO-0088936 | 44.76827 | 3.51386 | RESIDUAL_UNEXPLAINED | 0.0674 | IR_TOO_FAINT | DISAPPEARANCE_LIKE |
| VASCO-VO-0098304 | 47.83042 | 2.56799 | RESIDUAL_UNEXPLAINED | 0.514 | ENERGY_CONSERVING_OBSCURATION | OBSCURATION_LIKE |
| VASCO-VO-0102514 | 48.06281 | 3.11704 | RESIDUAL_UNEXPLAINED | 0.0406 | IR_UNDERSAMPLED | DISAPPEARANCE_LIKE |
| VASCO-VO-0111204 | 48.86608 | 3.37571 | RESIDUAL_UNEXPLAINED | 7.46 | ENERGY_CONSERVING_OBSCURATION | OBSCURATION_LIKE |
| VASCO-VO-0113394 | 48.25704 | 3.44298 | RESIDUAL_UNEXPLAINED | 0.0329 | IR_UNDERSAMPLED | DISAPPEARANCE_LIKE |
| VASCO-VO-0113395 | 48.25711 | 3.44369 | RESIDUAL_UNEXPLAINED | 0.0333 | IR_UNDERSAMPLED | DISAPPEARANCE_LIKE |
| VASCO-VO-0113396 | 48.25783 | 3.44420 | RESIDUAL_UNEXPLAINED | 0.0676 | IR_UNDERSAMPLED | DISAPPEARANCE_LIKE |
| VASCO-VO-0113480 | 48.10425 | 3.45573 | RESIDUAL_UNEXPLAINED | 0.135 | IR_UNDERSAMPLED | DISAPPEARANCE_LIKE |
| VASCO-VO-0119786 | 47.53445 | 3.39393 | RESIDUAL_UNEXPLAINED | 0.224 | IR_TOO_FAINT_MARGINAL | DISAPPEARANCE_LIKE |
| VASCO-VO-0126486 | 47.94963 | 3.74633 | RESIDUAL_UNEXPLAINED | 0.0118 | IR_UNDERSAMPLED | DISAPPEARANCE_LIKE |

No-null rule (CLAUDE.md): an empty survivor list is a statement
about THIS sample and these thresholds, never a publishable result.
The population breakdown and the obscuration-to-destruction ratio
are the standing measurements regardless.

