# RING — rings around the dead (S63)

**Verdict:** `DEGRADED (wd, ffp not reached); RING_CANDIDATES_PENDING_VET`

## White dwarfs

* input hosts with an AllWISE counterpart: 0 (route: `none`)
* SED anchor: None
* W1 / W2 detections: 0 / 0
* excess-flagged: 0; shapes: None
* surviving the catalogue gates: 0; gate rejections: None
* ring-band candidates before follow-up: 0
* after follow-up: **n/a** (None)
* sensitivity: None

## Pulsars

* hosts: 4,318 (route: `tarball`)
* with an AllWISE / CatWISE counterpart inside the per-pulsar radius: 778 / 1,448 (any: 1,572)
* median control hits per host over 16 offset positions: 1.0
* counterpart shapes: {'unfit': 1282, 'companion': 194, 'debris_disk': 86, 'ring_band': 10}
* vetoes: {'chance_coincidence': 1359, 'nebula_cluster_or_optical_association': 110, 'non_degenerate_companion': 92, 'catalogued_counterpart': 5}
* surviving: 6; ring-band: **6**
* sensitivity: {'n_hosts_with_edot_and_distance': 2771, 'n_hosts_500K_ring_detectable_at_f_lt_1': 1572, 'n_hosts_500K_ring_detectable_at_f_lt_0p1': 911, 'median_f_min_500K': 7.298767902304851}

### The two pulsar-planet systems

* **B1257+12** (J1300+1240): VETTED; AllWISE match False (controls 0, p 0.0780815771190365); CatWISE match False; shape `no_counterpart`; verdict `no_counterpart` ; f_min(500 K, W2) 0.000140058009761; three planets (Wolszczan & Frail 1992); Spitzer MIPS limits on asteroidal dust (Bryden+2006, verify)
* **B1620-26** (J1623-2631): VETTED; AllWISE match False (controls 0, p 0.0780815771190365); CatWISE match False; shape `no_counterpart`; verdict `no_counterpart` ; f_min(500 K, W2) 0.0022918263500619; circumbinary planet in M4 (Sigurdsson+2003); globular-cluster sightline, crowded

## Brown dwarfs (Y / late T): W2 duty cycle

* targets: 232; with NEOWISE epochs: 28; tested (>= min epochs): 11
* population W2 reduced-chi2 median: 3.50242083657619; threshold 10.50726250972857
* duty-cycle flags: **0**


## Free-floating planetary-mass objects: hotter than cooling allows

* objects: 0; testable (L_bol and an age): 0; catalogued planetary-mass: 0
* above the 13 M_J cooling ceiling (with the 0.5 dex margin): 0; of which catalogued planetary-mass: **0**
* every flag carries `age_misassignment|mass_underestimate|unresolved_binary` as systematics not excluded

See `docs/ring.md` for the claim, the prior art and the contamination model.
