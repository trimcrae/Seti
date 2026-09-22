# GRAVE report — 2026-09-22T16:38:59Z

**Verdict:** `DEGRADED_SOURCE (earthchem:NO_DATA_REACHED); NO_FISSION_VECTOR`  
**Refined-particulate:** `REFINED_PARTICULATE_CANDIDATES_PENDING_VET`

## Funnel

| stage | n |
|---|---|
| samples | 155304 |
| with_age | 101618 |
| sufficient_panel | 116515 |
| fission_lr_positive | 54814 |
| ambiguous | 11218 |
| above_threshold | 63 |
| assessed | 63 |
| survivors | 0 |

Threshold ln LR = 555.00 (floor shuffled-null q0.999 = 509.122).

## Vetoes

| veto | n |
|---|---|
| insufficient_panel | 0 |
| detection_limit_driver | 0 |
| unexplained_by_all_reservoirs | 63 |
| single_element_driver | 63 |
| peak_incoherent | 23 |
| redox_conditioned | 0 |
| femn_shuttle | 0 |
| heavy_mineral_zr_hf | 0 |
| volcanic_ash | 0 |
| monazite_th | 0 |
| hydrothermal_ba | 0 |
| impact_pge | 0 |

## Age stack (fission candidates)

| boundary | age | samples | sections | cand. | cand. sections | expected | p_raw | p_Holm | status |
|---|---|---|---|---|---|---|---|---|---|
| Ediacaran-Cambrian | 538.8 | 869 | 45 | 0 | 0 | 0.0 | 1.0 | 1.0 | no_candidate |
| End-Ordovician (Hirnantian) | 444.5 | 2374 | 263 | 0 | 0 | 0.0 | 1.0 | 1.0 | no_candidate |
| Late Devonian Kellwasser (F-F) | 372.2 | 2165 | 260 | 0 | 0 | 0.0 | 1.0 | 1.0 | no_candidate |
| Hangenberg (D-C) | 358.9 | 2349 | 303 | 0 | 0 | 0.0 | 1.0 | 1.0 | no_candidate |
| End-Guadalupian (Capitanian) | 259.5 | 91 | 23 | 0 | 0 | 0.0 | 1.0 | 1.0 | no_candidate |
| End-Permian | 251.9 | 1902 | 174 | 0 | 0 | 0.0 | 1.0 | 1.0 | no_candidate |
| Carnian Pluvial Episode | 233.0 | 460 | 159 | 0 | 0 | 0.0 | 1.0 | 1.0 | no_candidate |
| End-Triassic | 201.4 | 325 | 47 | 0 | 0 | 0.0 | 1.0 | 1.0 | no_candidate |
| Toarcian OAE | 183.0 | 1199 | 134 | 0 | 0 | 0.0 | 1.0 | 1.0 | no_candidate |
| Cenomanian-Turonian OAE2 | 93.9 | 1508 | 227 | 0 | 0 | 0.0 | 1.0 | 1.0 | no_candidate |
| Cretaceous-Paleogene | 66.0 | 1270 | 312 | 0 | 0 | 0.0 | 1.0 | 1.0 | no_candidate |
| PETM | 56.0 | 2092 | 593 | 0 | 0 | 0.0 | 1.0 | 1.0 | no_candidate |
| Eocene-Oligocene | 33.9 | 1436 | 286 | 0 | 0 | 0.0 | 1.0 | 1.0 | no_candidate |

Promotion reads the Holm-corrected `p_family` over the 13 testable windows, at a family-wise threshold of 0.05.

## Impact class as positive control (chondritic PGE, Ir-anchored)

No sample carries a PGE panel that classes as impact — the Ir positive control could not run on this corpus (state which elements were present: ['Os', 'Ir', 'Ru', 'Rh', 'Pt', 'Pd']).

## Survivors

none

## Refined-particulate classes

{'pge': {'pge_insufficient': 154679, 'pge_anomalous_unclassified': 320, 'pge_background': 303, 'refined_pge': 2}, 'alloy': {'alloy_none': 151214, 'hydrothermal_w': 2225, 'natural_nb_ta': 717, 'refined_w': 550, 'alloy_insufficient': 345, 'refined_ta': 253}}

- `sgp:6608` age 15.0 Ma U1601: pge None None alloy hydrothermal_w
- `sgp:6610` age 15.0 Ma U1601: pge None None alloy hydrothermal_w
- `sgp:6611` age 15.0 Ma U1601: pge None None alloy hydrothermal_w
- `sgp:6613` age 15.0 Ma U1601: pge None None alloy hydrothermal_w
- `sgp:6614` age 15.0 Ma U1601: pge None None alloy hydrothermal_w
- `sgp:6615` age 15.0 Ma U1601: pge None None alloy hydrothermal_w
- `sgp:6616` age 15.0 Ma U1601: pge None None alloy hydrothermal_w
- `sgp:6617` age 15.0 Ma U1602: pge None None alloy hydrothermal_w
- `sgp:6618` age 15.0 Ma U1602: pge None None alloy hydrothermal_w
- `sgp:6619` age 15.0 Ma U1602: pge None None alloy hydrothermal_w
- `sgp:6620` age 15.0 Ma U1602: pge None None alloy hydrothermal_w
- `sgp:6621` age 15.0 Ma U1602: pge None None alloy hydrothermal_w
- `sgp:6622` age 15.0 Ma U1602: pge None None alloy hydrothermal_w
- `sgp:6623` age 15.0 Ma U1602: pge None None alloy hydrothermal_w
- `sgp:6624` age 15.0 Ma WT2016.1: pge None None alloy hydrothermal_w
- `sgp:6626` age 15.0 Ma WT2016.1: pge None None alloy hydrothermal_w
- `sgp:6627` age 15.0 Ma WT2016.1: pge None None alloy hydrothermal_w
- `sgp:6628` age 15.0 Ma WT2016.1: pge None None alloy hydrothermal_w
- `sgp:6631` age 15.0 Ma WT2016.1: pge None None alloy hydrothermal_w
- `sgp:6634` age 15.0 Ma WT2016.1: pge None None alloy hydrothermal_w
- `sgp:6635` age 15.0 Ma WT2016.1: pge None None alloy hydrothermal_w
- `sgp:6636` age 15.0 Ma WT2016.1: pge None None alloy hydrothermal_w
- `sgp:6637` age 15.0 Ma WT2016.1: pge None None alloy hydrothermal_w
- `sgp:6638` age 15.0 Ma WT2016.2: pge None None alloy hydrothermal_w
- `sgp:6641` age 15.0 Ma WT2016.2: pge None None alloy hydrothermal_w
- `sgp:52810` age 11.53 Ma USGS-6181: pge None None alloy refined_ta
- `sgp:118316` age 0.0 Ma ST3: pge None None alloy refined_w
- `sgp:118317` age 0.0 Ma ST3: pge None None alloy refined_w
- `sgp:118319` age 0.0 Ma ST3: pge None None alloy refined_w
- `sgp:118323` age 0.0 Ma ST1: pge None None alloy refined_w

## Degradation

['earthchem:NO_DATA_REACHED']

