# RING — rings around the dead (S63)

**Verdict:** `DEGRADED (bd partial: 95/232 targets with epochs); NO_RING_SURVIVOR; SECONDARY_FLAGS_PENDING_VET`

Generated 2026-09-24T16:59:27Z by run `36016037098`. Leg provenance (the run whose files each leg's numbers come from): wd: `36016037098` (2026-09-24T16:04:43Z), pulsar: `36016037098` (2026-09-24T14:58:57Z), bd: `36016037098` (2026-09-24T16:32:10Z), ffp: `36016037098` (2026-09-24T14:57:02Z)

## White dwarfs

* input hosts with an AllWISE counterpart: 25,932 (route: `vizier_parent+cds_xmatch_propagated`)
* SED anchor: {'gaia': 18491, 'nir': 6137, 'none': 1304}
* W1 / W2 detections: 25,903 / 17,773
* excess-flagged: 4,694; shapes: {'companion': 3222, 'debris_disk': 705, 'warm_ambiguous': 587, 'ring_band': 179, 'unfit': 1}
* surviving the catalogue gates: 27; gate rejections: {'unresolved_companion': 1907, 'ledger': 926, 'wise_quality': 806, 'astrometric_registration': 789, 'background_source': 239}
* ring-band candidates before follow-up: 0
* after follow-up: **0** ({'comovement_untested': 27})
* sensitivity: {'n_hosts_with_500K_ring_detectable_at_f_lt_1': 24626, 'median_f_min_500K': 0.0855302469981484, 'n_hosts_with_500K_ring_detectable_at_f_lt_0p01': 3955}
* registration test evaluated for 25932 hosts; ring-band gate reasons: {'astrometric_registration': 82, 'background_source': 41, 'ledger': 37, 'wise_quality': 19}
* chance census (offset-position controls): None
* mechanism per flagged excess: None; ring vetoes: None

## Pulsars

* hosts: 4,318 (route: `tarball`)
* with an AllWISE / CatWISE counterpart inside the per-pulsar radius: 709 / 1,324 (any: 1,445)
* median control hits per host over 16 offset positions: 1.0
* counterpart shapes: {'unfit': 1195, 'companion': 173, 'debris_disk': 69, 'ring_band': 8}
* vetoes: {'position_not_localised': 1120, 'chance_coincidence': 188, 'nebula_cluster_or_optical_association': 86, 'single_band_no_colour': 23, 'non_degenerate_companion': 18, 'catalogued_counterpart': 5, 'companion_colour': 4, 'not_significant_after_trials': 1}
* surviving: 0; ring-band: **0**
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

* targets: 232; with NEOWISE epochs: 95; tested (>= min epochs): 60
* population W2 reduced-chi2 median: 3.3818401417183743; threshold 10.145520425155123
* duty-cycle flags: **4**
* coverage: 0.409 -- DEGRADED: 95/232 targets with epochs

  * `WISEPA J074457.15+562821.8` T8: n=16, chi2_red=10.3, amp=0.61 mag, high-state fraction 0.5625
  * `WISEPC J121756.91+162640.2AB` T9: n=9, chi2_red=19.9, amp=0.18 mag, high-state fraction 0.4444444444444444
  * `WISEPA J154151.66-225025.2` Y1: n=16, chi2_red=22.6, amp=0.43 mag, high-state fraction 0.625
  * `WISE J210200.15-442919.5` T9: n=11, chi2_red=13.3, amp=0.38 mag, high-state fraction 0.6363636363636364
  * vetoed `WISE J163940.86-684744.6` Y0pec: chi2_red=13.091026692922402, amp=0.8284997940063477 mag, W1-W2=-0.04028476517530244, pm_known=True -> colour_not_the_target

## Free-floating planetary-mass objects: hotter than cooling allows

* with L_bol: 63; with a group: 69; age sources: {'group_fallback': 67, 'none': 2}; unmatched group names: {'FIELD': 1, 'NONE': 1}
* objects: 69; testable (L_bol and an age): 61; catalogued planetary-mass: 8
* above the 13 M_J cooling ceiling (with the 0.5 dex margin): 43; of which catalogued planetary-mass: **0**
* every flag carries `age_misassignment|mass_underestimate|unresolved_binary` as systematics not excluded

See `docs/ring.md` for the claim, the prior art and the contamination model.
