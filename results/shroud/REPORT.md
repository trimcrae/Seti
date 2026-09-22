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

## Sky coverage (USNO-B1.0 reconstruction)

- fields fetched: 12 / 12 of radius 0.5 deg = **9.425 deg^2**
- USNO-B1.0 rows returned (Ndet = 1, R1 <= limit): 33273
- POSS-I-red-only objects: 0

## How deep the search that found nothing actually went

An absence is only as good as the search behind it: a catalogue
that was never successfully queried leaves every source with an
empty magnitude, and empty reads as *gone*.

- modern catalogues that answered: NONE
- sources no modern catalogue covered: None
- required depth margin: 2.0 mag below the plate detection

## Funnel

| stage | n |
|---|---:|
| 1_sample | 127 |
| 2_no_modern_optical_within_5arcsec | 127 |
| 2b_absence_established_deeper_than_the_plate | 0 |
| 2c_no_modern_catalogue_covered_the_position | 127 |
| 3_with_any_ir_detection | 0 |
| 4_ir_present_and_optically_absent | 0 |
| 5_residual_after_population_cascade | 0 |
| 6_survive_every_veto | 0 |
| 7_energy_conserving_obscuration | 0 |
| 7_ir_too_faint | 0 |

## Population breakdown

The first population analysis of this sample.

| class | n | fraction |
|---|---:|---:|
| ASTEROID | 66 | 0.5197 |
| PLATE_DEFECT | 61 | 0.4803 |

## Obscuration vs destruction

- optically vanished **with** an IR counterpart: 0
- optically vanished with **no** counterpart: 127
- raw ratio: -
- after subtracting the mundane classes: 0

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
| INSUFFICIENT_IR | 99 |
| NO_HISTORICAL_PHOTOMETRY | 28 |

Survivors of every kill-test: **0** (0 energy-conserving, 0 IR-too-faint).

(no survivors at the current thresholds)

No-null rule (CLAUDE.md): an empty survivor list is a statement
about THIS sample and these thresholds, never a publishable result.
The population breakdown and the obscuration-to-destruction ratio
are the standing measurements regardless.

