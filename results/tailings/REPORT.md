# TAILINGS — sparse chemical anomaly search

**Verdict.** DEGRADED_SOURCE (GALAH->GALAH_DR4_allstar_cloud, APOGEE->APOGEE_DR17_allStar_rev1); SPARSE_CANDIDATES_PENDING_REMEASUREMENT: 1809 catalogue-level sparse survivors and 0 co-natal pairs beyond the engulfment budget. None is a detection until the specific line is re-measured from the raw spectrum against Teff-matched peers.

## The discriminant

Natural abundance space is low-dimensional: ~8-10 independent chemical dimensions in the solar neighbourhood, and every natural process moves an element *family*. Industrial refining moves one element. So the statistic is sparsity, not amplitude: one or two elements extreme with the rest inside 2 sigma. A star with many elements discrepant is a **rejection** here.

## GALAH

- stars after quality cuts: **73,820**
- elements on the manifold: **25**
- classification: NORMAL 69,799, DENSE 2,598, SPARSE 1,341, INSUFFICIENT 82
- sparse candidates: **1341**, surviving vetting: **891**

### Sparse/dense contrast (the headline diagnostic)

| z_max | n | sparse | dense | sparse frac | median z_rest_rms |
|---|---|---|---|---|---|
| 2-3 | 27,019 | 0 | 0 | 0.000 | 0.93 |
| 3-4 | 8,498 | 0 | 0 | 0.000 | 1.19 |
| 4-6 | 4,359 | 569 | 867 | 0.131 | 1.51 |
| 6-8 | 1,424 | 515 | 886 | 0.362 | 1.95 |
| 8-12 | 871 | 234 | 629 | 0.269 | 2.43 |
| 12-inf | 242 | 23 | 216 | 0.095 | 3.70 |

### Surviving sparse candidates

`n_quiet` is the evidence, not `z_max`: it counts the elements that were **measured and found ordinary**. A one-element anomaly with 25 quiet elements is a strong claim; the same anomaly with 3 quiet elements is an information-starved one, which is the failure mode Huang et al. 2026 documented in polluted white dwarfs.

| star | element | z | n_quiet | contrast | Teff | [Fe/H] | caveats |
|---|---|---|---|---|---|---|---|
| 160327004101197 | **Y** | +14.0 | 15 | 9.0 | 5963 | -0.24 | nan |
| 160420005801261 | **Y** | +13.5 | 11 | 9.1 | 5986 | -0.74 | nan |
| 140708005301102 | **Y** | +13.3 | 17 | 13.3 | 5846 | -0.15 | nan |
| 140822001101056 | **Y** | +13.0 | 15 | 8.8 | 5953 | -0.30 | nan |
| 170413002601149 | **K** | -12.8 | 17 | 12.8 | 5719 | -0.73 | K I 7699 sits on a telluric O2 band and has an interstellar  |
| 170602003201086 | **Y** | +12.6 | 15 | 8.5 | 5722 | -0.69 | nan |
| 161107002101149 | **Y** | +12.2 | 15 | 9.9 | 5956 | -0.39 | nan |
| 161213001601210 | **Y** | +12.2 | 15 | 10.2 | 5954 | -0.40 | nan |
| 170614004101233 | **Y** | +12.0 | 17 | 10.6 | 5843 | -0.20 | nan |
| 170506003401289 | **Y** | +11.6 | 14 | 8.4 | 5838 | -0.32 | nan |
| 160530002801051 | **Y** | +11.5 | 16 | 10.0 | 5958 | -0.41 | nan |
| 170117002101368 | **K** | +11.3 | 18 | 11.3 | 5999 | +0.29 | K I 7699 sits on a telluric O2 band and has an interstellar  |
| 210711003101202 | **Y** | +11.2 | 15 | 8.8 | 5937 | -0.31 | nan |
| 180101002101008 | **Cr** | +11.0 | 17 | 6.5 | 5433 | -0.42 | nan |
| 170515003101235 | **Y** | +11.0 | 18 | 7.8 | 5748 | -0.03 | nan |
| 180628003301328 | **Y** | +10.9 | 16 | 8.1 | 5908 | -0.25 | nan |
| 161008002501364 | **Y** | +10.9 | 15 | 6.9 | 5729 | -0.62 | nan |
| 200810001101225 | **Y** | +10.9 | 15 | 7.0 | 5894 | -0.49 | nan |
| 161120002401352 | **K** | -10.9 | 18 | 10.9 | 5665 | -0.24 | K I 7699 sits on a telluric O2 band and has an interstellar  |
| 160129004701239 | **Y** | +10.8 | 11 | 8.4 | 5992 | -0.59 | nan |
| 161006004901245 | **Y** | +10.8 | 17 | 8.4 | 5491 | -0.48 | nan |
| 151111002101201 | **Y** | +10.7 | 16 | 6.9 | 5847 | -0.78 | nan |
| 170105002601012 | **Y** | +10.6 | 16 | 9.8 | 5895 | -0.43 | nan |
| 160402005101129 | **Y** | +10.6 | 15 | 5.9 | 5685 | -0.34 | nan |
| 160529005401048 | **Y** | +10.6 | 15 | 6.3 | 5781 | -0.82 | nan |


### Highest per-element flag rates (systematics check)

- `K` (odd_z): 1.21% of 72,772 measurements
- `Y` (s_light): 0.93% of 72,694 measurements
- `Cr` (fe_peak): 0.90% of 73,702 measurements
- `Ca` (alpha): 0.89% of 73,772 measurements
- `Ba` (s_heavy): 0.83% of 73,744 measurements

## APOGEE

- stars after quality cuts: **137,047**
- elements on the manifold: **17**
- classification: NORMAL 77,025, INSUFFICIENT 54,413, SPARSE 3,173, DENSE 2,436
- sparse candidates: **3173**, surviving vetting: **918**

### Sparse/dense contrast (the headline diagnostic)

| z_max | n | sparse | dense | sparse frac | median z_rest_rms |
|---|---|---|---|---|---|
| 2-3 | 37,956 | 0 | 0 | 0.000 | 0.96 |
| 3-4 | 14,631 | 0 | 0 | 0.000 | 1.21 |
| 4-6 | 10,339 | 1,418 | 949 | 0.137 | 1.34 |
| 6-8 | 3,421 | 1,221 | 849 | 0.357 | 1.43 |
| 8-12 | 1,754 | 465 | 465 | 0.265 | 1.78 |
| 12-inf | 598 | 69 | 173 | 0.115 | 3.37 |

### Surviving sparse candidates

`n_quiet` is the evidence, not `z_max`: it counts the elements that were **measured and found ordinary**. A one-element anomaly with 25 quiet elements is a strong claim; the same anomaly with 3 quiet elements is an information-starved one, which is the failure mode Huang et al. 2026 documented in polluted white dwarfs.

| star | element | z | n_quiet | contrast | Teff | [Fe/H] | caveats |
|---|---|---|---|---|---|---|---|
| 2M21322339+1223283 | **Mn** | -28.8 | 11 | 24.2 | 5812 | -0.67 | nan |
| 2M18553411-2153444 | **K** | -24.1 | 11 | 24.1 | 4974 | +0.17 | two weak lines with significant blending |
| 2M18534075+6707107 | **Cr** | -22.9 | 9 | 12.5 | 5438 | -0.47 | nan |
| 2M17111221+5623577 | **K** | -19.2 | 10 | 13.7 | 5829 | -0.37 | two weak lines with significant blending |
| 2M14362553+5204153 | **Ni** | +17.7 | 11 | 17.7 | 5831 | -0.38 | nan |
| 2M07162452+2903580 | **Cr** | -17.7 | 11 | 17.3 | 4966 | +0.06 | nan |
| 2M06464143-6704372 | **K** | -17.2 | 8 | 11.7 | 5089 | +0.14 | two weak lines with significant blending |
| 2M23432248+2924547 | **Cr** | -17.0 | 10 | 11.0 | 5849 | -0.21 | nan |
| 2M15092672+3621198 | **Cr** | -15.5 | 9 | 9.7 | 5075 | -0.19 | nan |
| 2M10074949+0457070 | **K** | -14.9 | 10 | 11.3 | 5742 | -0.56 | two weak lines with significant blending |
| 2M08175682+5132510 | **Mn** | -14.8 | 11 | 14.8 | 5940 | -0.73 | nan |
| 2M00295900-1420258 | **Cr** | -14.8 | 9 | 10.6 | 5440 | -0.44 | nan |
| 2M08195707+5553424 | **K** | -14.4 | 10 | 11.8 | 5869 | -0.11 | two weak lines with significant blending |
| 2M13011084+2728012 | **K** | -13.8 | 11 | 13.8 | 5763 | -0.22 | two weak lines with significant blending |
| 2M15325168+2851229 | **Mn** | -13.1 | 9 | 7.0 | 5468 | -0.96 | nan |
| 2M09472510+3240429 | **K** | -13.0 | 10 | 11.3 | 5943 | -0.16 | two weak lines with significant blending |
| 2M11181900+5337323 | **Cr** | -12.9 | 11 | 12.9 | 5286 | -0.39 | nan |
| 2M17095026+3248572 | **K** | -12.6 | 9 | 9.9 | 5624 | -0.34 | two weak lines with significant blending |
| 2M01445741-1715306 | **Cr** | -12.4 | 10 | 10.7 | 5871 | -0.05 | nan |
| 2M22585311-2340255 | **Cr** | -12.4 | 11 | 12.4 | 5773 | -0.00 | nan |
| 2M10265040+3644245 | **Cr** | -12.2 | 9 | 7.7 | 5690 | -0.42 | nan |
| 2M14234688+5754555 | **Cr** | -12.0 | 11 | 10.7 | 5460 | -0.24 | nan |
| 2M16351877+2424566 | **Mn** | -11.8 | 10 | 9.3 | 5826 | -0.63 | nan |
| 2M10561866+0648201 | **Cr** | -11.7 | 9 | 7.5 | 5486 | -0.27 | nan |
| 2M19354196+4628595 | **Cr** | -11.6 | 10 | 9.8 | 5139 | -0.52 | nan |


### Highest per-element flag rates (systematics check)

- `Na` (odd_z): 2.87% of 113,029 measurements
- `Cr` (fe_peak): 1.63% of 115,534 measurements
- `O` (alpha): 0.99% of 136,663 measurements
- `K` (odd_z): 0.86% of 136,910 measurements
- `Mn` (fe_peak): 0.78% of 132,306 measurements

## Stage 4 — co-natal wide binaries

NO_PAIRS_WITH_TWO_SPECTRA: no wide binary had both components in the spectroscopic sample


## What a survivor still has to pass

Nothing here is a detection. A catalogue-level sparse survivor is a *target*, and the decisive test is to re-measure the specific line from the raw spectrum against Teff-matched peers observed with the same instrument, so that blends, telluric residuals and continuum structure common to the temperature slice cancel. Until that is done the correct description is 'an unexplained single-element catalogue outlier'.

## No-null rule (CLAUDE.md)

An empty candidate list at these thresholds is a statement about this corpus, these elements and these thresholds — not a publishable null. The escalation path is more elements (optical n-capture lines that the H band cannot reach), a second survey for cross-confirmation, and the differential co-natal channel, which reaches ~0.01-0.02 dex where the field channel reaches ~0.03-0.05.

