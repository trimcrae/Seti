# RING — rings around the dead (S63)

**Verdict:** `DEGRADED (ffp not reached; bd partial: acquisition PARTIAL; 47/232 targets with epochs); NO_RING_SURVIVOR; SECONDARY_FLAGS_PENDING_VET`

Generated 2026-09-23T14:32:09Z by run `35860904385`. Leg provenance (the run whose files each leg's numbers come from): wd: `35860904385` (2026-09-23T13:12:53Z), pulsar: `35860904385` (2026-09-23T12:37:30Z), bd: `35860904385` (2026-09-23T14:32:09Z), ffp: `35860904385` (2026-09-23T12:37:51Z)

## White dwarfs

* input hosts with an AllWISE counterpart: 25,932 (route: `vizier_parent+cds_xmatch_propagated`)
* SED anchor: {'gaia': 18491, 'nir': 6137, 'none': 1304}
* W1 / W2 detections: 25,903 / 17,773
* excess-flagged: 4,694; shapes: {'companion': 3222, 'debris_disk': 705, 'warm_ambiguous': 587, 'ring_band': 179, 'unfit': 1}
* surviving the catalogue gates: 0; gate rejections: {'unresolved_companion': 1907, 'astrometric_registration': 1055, 'ledger': 926, 'wise_quality': 806}
* ring-band candidates before follow-up: 0
* after follow-up: **n/a** (None)
* sensitivity: {'n_hosts_with_500K_ring_detectable_at_f_lt_1': 24626, 'median_f_min_500K': 0.0855302469981484, 'n_hosts_with_500K_ring_detectable_at_f_lt_0p01': 3955}

## Pulsars

* hosts: 4,318 (route: `tarball`)
* with an AllWISE / CatWISE counterpart inside the per-pulsar radius: 709 / 1,324 (any: 1,445)
* median control hits per host over 16 offset positions: 1.0
* counterpart shapes: {'unfit': 1195, 'companion': 173, 'debris_disk': 69, 'ring_band': 8}
* vetoes: {'position_not_localised': 1120, 'chance_coincidence': 188, 'nebula_cluster_or_optical_association': 86, 'single_band_no_colour': 23, 'non_degenerate_companion': 18, 'catalogued_counterpart': 5, 'companion_colour': 4}
* surviving: 1; ring-band: **0**
* localised hosts (2 x position error <= the widest aperture): 2770
* chance census (observed vs the local-control expectation): {'allwise': {'observed': 709, 'expected_by_chance': 607.4, 'observed_localised': 160, 'expected_localised': 80.4, 'observed_unlocalised': 549, 'expected_unlocalised': 527.0, 'ring_colour_observed': 1, 'ring_colour_expected_by_chance': 1.38, 'ring_colour_observed_localised': 1, 'ring_colour_expected_localised': 0.06}, 'catwise': {'observed': 1324, 'expected_by_chance': 1271.9, 'observed_localised': 253, 'expected_localised': 212.6, 'observed_unlocalised': 1071, 'expected_unlocalised': 1059.4, 'ring_colour_observed': 8, 'ring_colour_expected_by_chance': 9.56, 'ring_colour_observed_localised': 2, 'ring_colour_expected_localised': 1.62}}
* sensitivity: {'n_hosts_with_edot_and_distance': 2771, 'n_hosts_500K_ring_detectable_at_f_lt_1': 1572, 'n_hosts_500K_ring_detectable_at_f_lt_0p1': 911, 'median_f_min_500K': 7.298767902304851}

### Every ring-band-coloured counterpart and its fate

| pulsar | localised | pos err (") | colour src | W1-W2 | T (K) | p_chance(ring) | veto |
|---|---|---|---|---|---|---|---|
| J0040-7337 | True | 3.23 | catwise | 1.37 +/- 0.23 | 723 | 0.0018 | nebula_cluster_or_optical_association |
| J1750-3703D | True | 0.0177 | catwise | 1.44 +/- 0.03 | 693 | 9.5e-05 | nebula_cluster_or_optical_association |
| J1823-3021D | True | 0.362 | allwise | 1.23 +/- 0.02 | 789 | 1.4e-05 | nebula_cluster_or_optical_association |
| J1847-0308_P | False | 60 | catwise | 1.64 +/- 0.04 | 624 | 0.065 | position_not_localised |
| J1854+40 | False | 900 | catwise | 1.82 +/- 0.26 | 573 | 0.0027 | position_not_localised |
| J1929+2355_P | False | 60 | catwise | 2.10 +/- 0.28 | 507 | 0.0027 | position_not_localised |
| J2016+4231 | False | 895 | catwise | 1.27 +/- 0.08 | 767 | 0.0027 | position_not_localised |
| J2201+33 | False | 600 | catwise | 1.53 +/- 0.30 | 660 | 0.0027 | position_not_localised |

### The two pulsar-planet systems

* **B1257+12** (J1300+1240): VETTED; AllWISE match False (controls 0, p 0.0060217107152946); CatWISE match False; shape `no_counterpart`; verdict `no_counterpart` ; f_min(500 K, W2) 0.000140058009761; three planets (Wolszczan & Frail 1992); Spitzer MIPS limits on asteroidal dust (Bryden+2006, verify)
* **B1620-26** (J1623-2631): VETTED; AllWISE match False (controls 0, p 0.0060217107152946); CatWISE match False; shape `no_counterpart`; verdict `no_counterpart` ; f_min(500 K, W2) 0.0022918263500619; circumbinary planet in M4 (Sigurdsson+2003); globular-cluster sightline, crowded

## Brown dwarfs (Y / late T): W2 duty cycle

* targets: 232; with NEOWISE epochs: 47; tested (>= min epochs): 19
* population W2 reduced-chi2 median: 4.411020177613055; threshold 13.233060532839165
* duty-cycle flags: **1**
* coverage: 0.203 -- DEGRADED: acquisition PARTIAL; 47/232 targets with epochs

  * `WISE J181329.40+283533.3` T8: n=9, chi2_red=17.4, amp=0.70 mag, high-state fraction 0.5555555555555556

## Free-floating planetary-mass objects: hotter than cooling allows

* with L_bol: 63; with a group: 69; age sources: {'none': 69}; unmatched group names: {'HLM': 29, 'AM': 21, 'BM': 10, 'NM': 9}
* objects: 69; testable (L_bol and an age): 0; catalogued planetary-mass: 0
* above the 13 M_J cooling ceiling (with the 0.5 dex margin): 0; of which catalogued planetary-mass: **0**
* every flag carries `age_misassignment|mass_underestimate|unresolved_binary` as systematics not excluded

See `docs/ring.md` for the claim, the prior art and the contamination model.
