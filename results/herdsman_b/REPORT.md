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
## Post-run vetting verdict (2026-07-26, session analysis)

Hogg_4 — the single formal candidate — FAILS member-level vetting:
corr(mh, G) = -0.68 (fainter members read metal-poorer), corr(mh, Teff) =
+0.49 residual, s_trim = 0.83 dex with members spanning [-4.1, +0.75] —
a GSP-Phot extinction-metallicity systematic at 4.2 kpc in the plane, not
chemistry. It passed field-likeness only because the field baseline at that
distance is inflated by the same systematic. The top-10 is dominated by
globular clusters and distant reddened open clusters — the two populations
where XP metallicities degrade most.

v1 conclusion: no credible completed-assembly candidate in the census at
GSP-Phot sensitivity; the selection machinery (1/5106 formal, killed on
vetting for an identifiable instrumental cause; 0 false positives in the
synthetic control) behaves as designed. v2 upgrades queued: |corr(mh,G)|
kill switch in the scorer, distance/extinction gating, distance-matched
field baseline, globular-cluster flagging, and the decisive step —
GALAH/APOGEE spectroscopic chemistry crossmatch, where a real gathered
population would still stand out and systematics of this kind cannot hide.

## Spectroscopic crossmatch (v2)

Surveys: galah, apogee. Clusters with >= 6 flag-clean spectroscopic members at prob >= 0.7: 80; spectro candidates: 0.

Spectroscopic [Fe/H] (GALAH flag_sp = flag_fe_h = 0; APOGEE ASPCAPflag = 0)
is immune to the GSP-Phot extinction/magnitude systematic that killed the
v1 photometric candidate; a spectro candidate has census-z >= 4, spread
>= 2x its error floor, is unimodal, and mirrors the survey's own
non-member field spread. Surveys are scored separately (zero-point
offsets between GALAH and APOGEE would otherwise fake a spread).

Top by census z:

- NGC_6715 [apogee]: n=8, s=0.872, x=25.69, z=3.2, field_likeness=0.26, two_pop=False, candidate=False
- HSC_955 [apogee]: n=10, s=0.378, x=11.94, z=2.4, field_likeness=0.48, two_pop=False, candidate=False
- IC_4665 [galah]: n=6, s=0.170, x=2.78, z=2.4, field_likeness=0.68, two_pop=False, candidate=False
- NGC_5139 [galah]: n=14, s=0.269, x=2.65, z=2.2, field_likeness=0.92, two_pop=False, candidate=False
- NGC_1579 [apogee]: n=18, s=0.255, x=7.90, z=2.2, field_likeness=0.89, two_pop=False, candidate=False
- Theia_7 [apogee]: n=16, s=0.234, x=7.22, z=1.8, field_likeness=0.97, two_pop=False, candidate=False
- CWNU_1129 [apogee]: n=8, s=0.226, x=7.00, z=1.7, field_likeness=1.00, two_pop=False, candidate=False
- NGC_2068 [apogee]: n=9, s=0.186, x=5.72, z=1.5, field_likeness=0.82, two_pop=False, candidate=False
- HSC_1318 [apogee]: n=12, s=0.164, x=5.22, z=1.5, field_likeness=0.72, two_pop=False, candidate=False
- NGC_2244 [apogee]: n=7, s=0.182, x=5.47, z=1.4, field_likeness=0.73, two_pop=False, candidate=False
