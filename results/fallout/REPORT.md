# FALLOUT — fission-product abundance pattern search

Generated 2026-09-06T14:16:13Z (UTC).

**Verdict.** DEGRADED_SOURCE (GALAH->GALAH_DR4_allstar_cloud: no rv column: the instrumental-covariate veto on rv cannot run on this table; no fiber column: the instrumental-covariate veto on fiber cannot run on this table); NO_FISSION_PATTERN: no star in any sample kept the fission-only preference through the vetoes at the null-calibrated threshold (after the vet stage); 184 above-threshold stars are UNEXPLAINED_BY_ALL_TEMPLATES and are listed separately. Per CLAUDE.md this is a reason to change the question (a second survey, the APOGEE Ce/Nd panel, differential co-natal pairs), not a result to write up.

## The discriminant

Fission product, once its short-lived members have decayed, is a fixed *shape* in n-capture space: light peak Zr-Mo-Ru, heavy peak Ba-La-Ce-Pr-Nd, a ~1000x valley (Ag-Sb), almost nothing past Sm, and no Pb. Against solar that is `[Nd/Ba] >> 0, [Ce/Ba] > 0, [La/Ba] > 0, [Mo/Zr] > 0, [Ru/Zr] > 0, [Eu/Nd] < 0`. The s-process gives `[Nd/Ba] < 0` and `[Mo/Zr] < 0`; the r-process gives `[Eu/Nd] > 0`. Each star is fitted as solar + {s, r, s+r, fission}; the statistic is the fission-only log-likelihood ratio against the best natural mixture, and a star is only a candidate if its best model actually fits (reduced chi2 below the cap), if removing any one element leaves the preference standing, and if at least two heavy-peak elements are individually up.

### Template vectors at amplitude 1 (Nd doubled), dex

| element | fission | s | r |
|---|---|---|---|
| Rb | +0.005 | +0.176 | +0.176 |
| Sr | +0.003 | +0.267 | +0.057 |
| Y | +0.016 | +0.283 | +0.033 |
| Zr | +0.049 | +0.263 | +0.068 |
| Mo | +0.150 | +0.176 | +0.097 |
| Ru | +0.144 | +0.121 | +0.207 |
| Ba | +0.037 | +0.258 | +0.072 |
| La | +0.213 | +0.209 | +0.137 |
| Ce | +0.147 | +0.248 | +0.083 |
| Nd | +0.301 | +0.193 | +0.158 |
| Sm | +0.183 | +0.111 | +0.225 |
| Eu | +0.087 | +0.025 | +0.288 |

## Acquisition

- **GALAH**: OK via file from `GALAH_DR4_allstar_cloud` — 395,752 rows, 30 elements
  - degradation: no rv column: the instrumental-covariate veto on rv cannot run on this table; no fiber column: the instrumental-covariate veto on fiber cannot run on this table
  - elements found: Al, Ba, C, Ca, Ce, Co, Cr, Cu, Eu, K, La, Mg, Mn, Mo, N, Na, Nd, Ni, O, Rb, Ru, Sc, Si, Sm, Sr, Ti, V, Y, Zn, Zr
  - extras found: {'flag_sp': 'flag_sp', 'flag_fe_h': 'flag_fe_h', 'e_teff': 'e_teff', 'e_logg': 'e_logg', 'e_fe_h': 'e_fe_h', 'age': 'age', 'mass': 'mass', 'vsini': 'vsini', 'rv_err': 'e_rv_comp_1', 'ebv': 'ebv', 'chi2_sp': 'chi2_sp'}; absent: ['binary_flag', 'log_lum']
  - GALAH: acquisition verdict OK via file from GALAH_DR4_allstar_cloud (395752 rows, 30 elements)
  - GALAH: degradation: no rv column: the instrumental-covariate veto on rv cannot run on this table; no fiber column: the instrumental-covariate veto on fiber cannot run on this table
  - GALAH: extra columns attached: flag_sp<-flag_sp, flag_fe_h<-flag_fe_h, e_teff<-e_teff, e_logg<-e_logg, e_fe_h<-e_fe_h, age<-age, mass<-mass, vsini<-vsini, rv_err<-e_rv_comp_1, ebv<-ebv, chi2_sp<-chi2_sp
  - GALAH: extra columns absent from the catalogue: binary_flag, log_lum

## GALAH/dwarf

- stars in the box: **101,928**; pattern elements: **11** (Rb, Sr, Y, Zr, Ru, Ba, La, Ce, Nd, Sm, Eu)
- classification: INSUFFICIENT 79,690, NORMAL 19,553, AMBIGUOUS 846, R_PROCESS 664, S_PROCESS 582, FISSION 396, S_PLUS_R 197
- threshold: ln LR ≥ **17.01** (shuffled-null q0.999 = 17.01 exceeds config lr_min); enrichment ln LR ≥ 12.5; reduced chi2 ≤ None
- null quantiles (ln LR): shuffled q0.5 0.00, q0.9 0.11, q0.99 3.97, q0.999 17.01, q0.9999 38.41 | sample q0.5 0.00, q0.9 0.37, q0.99 7.16, q0.999 19.84, q0.9999 44.16
- above threshold: **163** (raw-space: 405); unexplained by all templates: **0**; survivors after vetoes: **0**; after the vet stage: **0**
- vetoes (independent counts): low_snr_or_flagged 152, s_process_star 2, r_process_star 2, young_ba_enhancement 6, nlte_saturated_lines 101, single_element_driver 156, teff_peer_residual 14, teff_peer_residual_raw_only 277

### Sensitivity (injected fission pattern into real vectors)

| a_f | Δ[Nd/H] dex | LR pass (testable) | LR + LOO pass (testable) | LR pass (all) | LR + LOO pass (all) |
|---|---|---|---|---|---|
| 0.5 | +0.18 | nan | nan | 0.00 | 0.00 |
| 1 | +0.30 | nan | nan | 0.01 | 0.00 |
| 2 | +0.48 | nan | nan | 0.02 | 0.00 |
| 3 | +0.60 | nan | nan | 0.03 | 0.00 |
| 5 | +0.78 | nan | nan | 0.05 | 0.01 |
| 10 | +1.04 | nan | nan | 0.10 | 0.01 |
| 20 | +1.32 | nan | nan | 0.17 | 0.03 |

### Vet stage

- candidates re-vetted: 163; survivors at screen: 0; **after vet: 0**
- sigma: rebuilt from recorded per-element floors (quoted errors not in CSV); refit: True; La suspect: True (no La diagnostic in this summary (screened before the diagnostic existed); La in cool giants is distrusted until shown clean)
- vetoes: low_snr_or_flagged 97, unexplained_by_all_templates 127, s_process_star 2, r_process_star 2, young_ba_enhancement 6, nlte_saturated_lines 163, single_element_driver 163, heavy_peak_incoherent 106, la_cn_blend 0, teff_peer_residual 14, below_threshold_after_refit 163

## GALAH/giant

- stars in the box: **78,344**; pattern elements: **12** (Rb, Sr, Y, Zr, Mo, Ru, Ba, La, Ce, Nd, Sm, Eu)
- classification: NORMAL 72,372, S_PROCESS 2,466, INSUFFICIENT 1,157, S_PLUS_R 1,141, R_PROCESS 631, AMBIGUOUS 488, FISSION 89
- threshold: ln LR ≥ **9.72** (shuffled-null q0.999 = 9.72 exceeds config lr_min); enrichment ln LR ≥ 12.5; reduced chi2 ≤ None
- null quantiles (ln LR): shuffled q0.5 0.00, q0.9 0.14, q0.99 2.79, q0.999 9.72, q0.9999 38.19 | sample q0.5 0.00, q0.9 0.14, q0.99 2.42, q0.999 9.16, q0.9999 55.54
- above threshold: **72** (raw-space: 68); unexplained by all templates: **0**; survivors after vetoes: **2**; after the vet stage: **0**
- vetoes (independent counts): low_snr_or_flagged 55, s_process_star 1, r_process_star 7, young_ba_enhancement 5, nlte_saturated_lines 25, single_element_driver 61, teff_peer_residual 9, teff_peer_residual_raw_only 17

### Sensitivity (injected fission pattern into real vectors)

| a_f | Δ[Nd/H] dex | LR pass (testable) | LR + LOO pass (testable) | LR pass (all) | LR + LOO pass (all) |
|---|---|---|---|---|---|
| 0.5 | +0.18 | nan | nan | 0.01 | 0.00 |
| 1 | +0.30 | nan | nan | 0.11 | 0.01 |
| 2 | +0.48 | nan | nan | 0.50 | 0.16 |
| 3 | +0.60 | nan | nan | 0.71 | 0.34 |
| 5 | +0.78 | nan | nan | 0.86 | 0.56 |
| 10 | +1.04 | nan | nan | 0.94 | 0.76 |
| 20 | +1.32 | nan | nan | 0.97 | 0.85 |

### Survivors (catalogue-level; pending re-measurement)

| star | Teff | log g | [Fe/H] | n_el | ln LR | red. chi2 | no-Ba | LOO min (driver) | heavy≥2σ | raw | a_f |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 170203001601307 | 4785 | 2.46 | -0.13 | 9 | 52.5 | nan | 48.4 | 10.9 (La) | - | 61.6 | 1.53 |
| 230511003401363 | 4451 | 1.67 | -0.94 | 8 | 26.9 | nan | 27.7 | 14.1 (La) | - | 24.8 | 2.76 |

### Vet stage

- candidates re-vetted: 72; survivors at screen: 2; **after vet: 0**
- sigma: rebuilt from recorded per-element floors (quoted errors not in CSV); refit: True; La suspect: True (no La diagnostic in this summary (screened before the diagnostic existed); La in cool giants is distrusted until shown clean)
- vetoes: low_snr_or_flagged 54, unexplained_by_all_templates 57, s_process_star 1, r_process_star 7, young_ba_enhancement 5, nlte_saturated_lines 59, single_element_driver 68, heavy_peak_incoherent 15, la_cn_blend 31, teff_peer_residual 9, below_threshold_after_refit 58

## What a survivor still has to pass

Nothing here is a detection. A survivor is a *target*: the Ba II, La II, Ce II, Nd II and Eu II lines must be re-measured from the raw HERMES spectrum against Teff-matched peers, the pattern must hold element by element, and the star must be checked for an unresolved companion. Until then the correct description is 'an n-capture vector the s+r mixture does not fit'.

## No-null rule (CLAUDE.md)

An empty survivor list at this threshold is a statement about GALAH DR4's element panel and precision, not a publishable null. The escalation path is the APOGEE Ce/Nd panel as a second survey, the co-natal differential channel, and high-resolution re-measurement of the strongest ambiguous and unexplained stars.

