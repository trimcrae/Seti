# OSSUARY — warm dust around stars that cannot make it

**Verdict:** `OK`

## Sample

* input rows: 6,192,472
* dwarfs: 4,837,836 | giants (analysed separately): 1,288,830
* metal-poor ([Fe/H] < -1): 4,136,289
* halo kinematics: 837,985
* hosts with no natural reservoir (metal-poor OR halo): 4,693,593
* photosphere anchor: `G`

### Galactic population

* `unclassified`: 4,731,450
* `halo`: 837,985
* `thick_disk`: 535,166
* `thin_disk`: 63,994
* `disk_intermediate`: 23,877

## Contamination funnel (excess-flagged sources only)

Silverberg et al. 2018 measured a ~92% false-positive rate for AllWISE-selected infrared excesses, so the funnel reports what each stage removed, not only the running total.

| stage | surviving | removed here |
|---|---|---|
| input | 17,211 |  |
| after_wise_quality | 11,099 | 6112 |
| after_ledger | 1,993 | 9106 |
| after_unresolved_companion | 1,985 | 8 |
| after_astrometric_registration | 1,959 | 26 |
| after_background_source | 1,764 | 195 |
| after_galactic_cirrus | 732 | 1032 |
| after_globular_cluster_sightline | 732 | 0 |
| after_lambda_boo_or_blue_straggler | 634 | 98 |
| after_giant_or_unclassified | 629 | 5 |
| after_not_a_null_reservoir_host | 584 | 45 |
| surviving | 584 |  |

### Rejections by first failing gate

* `ledger`: 9,106
* `wise_quality`: 6,112
* `galactic_cirrus`: 1,032
* `background_source`: 195
* `lambda_boo_or_blue_straggler`: 98
* `not_a_null_reservoir_host`: 45
* `astrometric_registration`: 26
* `unresolved_companion`: 8
* `giant_or_unclassified`: 5

### Expected chance extragalactic alignments

Over 6,192,472 stars, within the 1.0" registration radius:
* Hot DOGs (9e-6 arcsec^-2): **175** expected
* any AllWISE source bright enough to supply the excess: **96.9** expected
* the same, in the full 6.5" beam without the registration cut: 4.09e+03 (the cut buys a factor 42)

## Candidates

* excess-flagged: 17,211
* surviving the full gauntlet: **584**
* surviving per-object follow-up: **251**

See `docs/ossuary.md` for the claim, the novelty verdict and the contamination model.
