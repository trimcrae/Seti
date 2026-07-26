# HERDSMAN-B run report

Clusters scored: 5106; assembly candidates: 1.

A candidate is a bound group whose Teff-detrended, interloper-trimmed
[M/H] spread is a >=4-sigma outlier against comparable-N clusters,
exceeds twice its error floor, is unimodal, and mirrors the local
field spread (docs/herdsman.md section 5). Two-population outliers
are reported separately (natural heterogeneous channel).

Top of census (by z):

- NGC_1798: n=80, x_trim=33.80, z=5.7, field_likeness=0.13, two_pop=False, candidate=False
- Hogg_4: n=104, x_trim=20.87, z=4.6, field_likeness=0.59, two_pop=False, candidate=True
- NGC_5024: n=194, x_trim=16.46, z=4.2, field_likeness=0.47, two_pop=False, candidate=False
- IC_4499: n=175, x_trim=14.51, z=3.9, field_likeness=0.24, two_pop=False, candidate=False
- Ruprecht_101: n=58, x_trim=19.25, z=3.8, field_likeness=0.59, two_pop=False, candidate=False
- NGC_6779: n=83, x_trim=16.47, z=3.8, field_likeness=0.36, two_pop=False, candidate=False
- NGC_6235: n=87, x_trim=15.91, z=3.7, field_likeness=0.44, two_pop=False, candidate=False
- Turner_6: n=48, x_trim=17.77, z=3.5, field_likeness=0.59, two_pop=False, candidate=False
- Pismis_18: n=18, x_trim=25.92, z=3.5, field_likeness=0.40, two_pop=False, candidate=False
- Berkeley_92: n=52, x_trim=17.09, z=3.4, field_likeness=0.15, two_pop=False, candidate=False

Two-population flags (stripped-nucleus/merger channel): 38.

No-null rule: an empty candidate list at these thresholds is a domain statement (this census, these quality cuts), not a result — next moves are deeper chemistry (GALAH/APOGEE crossmatch), co-moving groups beyond the cluster census, and per-candidate spectroscopic follow-up.