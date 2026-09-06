# FALLOUT — fission-product abundance pattern search

Generated 2026-09-06T14:23:12Z (UTC).

**Verdict.** DEGRADED_SOURCE (GALAH->GALAH_DR4_allstar_cloud: no rv column: the instrumental-covariate veto on rv cannot run on this table; no fiber column: the instrumental-covariate veto on fiber cannot run on this table); NO_FISSION_PATTERN: no star in any sample kept the fission-only preference through the vetoes at the null-calibrated threshold (after the vet stage); 29 above-threshold stars are UNEXPLAINED_BY_ALL_TEMPLATES and are listed separately. Per CLAUDE.md this is a reason to change the question (a second survey, the APOGEE Ce/Nd panel, differential co-natal pairs), not a result to write up.

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
- testable (≥5 elements and ≥2 of La/Ce/Nd measured): **20,360** = 20.0% of the sample
- classification: INSUFFICIENT 79,690, NORMAL 21,842, S_PROCESS 149, UNEXPLAINED_BY_ALL_TEMPLATES 120, R_PROCESS 63, S_PLUS_R 37, AMBIGUOUS 27
- error model: quoted error floored at the measured peer scatter — Rb 0.40 (×1.84), Sr 0.34 (×2.1), Y 0.08 (×1.65), Zr 0.23 (×0.75), Ru 0.31 (×1.59), Ba 0.10 (×1.73), La 0.21 (×0.9), Ce 0.28 (×0.91), Nd 0.16 (×1.02), Sm 0.22 (×1.29), Eu 0.32 (×2.27)
- La diagnostic: suspect=False — no correlation above 0.2 with teff, logg, C, N, vsini; correlations {'teff': -0.0122, 'logg': 0.003, 'C': -0.0183, 'N': -0.0002, 'vsini': 0.0163}
- threshold: ln LR ≥ **8.00** (config lr_min (shuffled-null q0.999 = 4.13 is below it)); enrichment ln LR ≥ 12.5; reduced chi2 ≤ 3.0
- null quantiles (ln LR): shuffled q0.5 0.00, q0.9 0.06, q0.99 1.17, q0.999 4.13, q0.9999 10.19 | sample q0.5 0.00, q0.9 0.17, q0.99 1.54, q0.999 4.31, q0.9999 7.79
- above threshold: **10** (raw-space: 17); unexplained by all templates: **8**; survivors after vetoes: **0**; after the vet stage: **0**
- vetoes (independent counts): low_snr_or_flagged 10, unexplained_by_all_templates 8, s_process_star 0, r_process_star 0, young_ba_enhancement 0, nlte_saturated_lines 10, single_element_driver 10, heavy_peak_incoherent 7, la_cn_blend 0, teff_peer_residual 0, teff_peer_residual_raw_only 9

### Sensitivity (injected fission pattern into real vectors)

| a_f | Δ[Nd/H] dex | LR pass (testable) | LR + LOO pass (testable) | LR pass (all) | LR + LOO pass (all) |
|---|---|---|---|---|---|
| 0.5 | +0.18 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | +0.30 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | +0.48 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | +0.60 | 0.01 | 0.00 | 0.01 | 0.00 |
| 5 | +0.78 | 0.05 | 0.00 | 0.02 | 0.00 |
| 10 | +1.04 | 0.17 | 0.01 | 0.06 | 0.00 |
| 20 | +1.32 | 0.40 | 0.04 | 0.15 | 0.01 |

Testable fraction 20.0% (1,500 injected).

### Unexplained by all templates (listed, never candidates)

| star | Teff | log g | [Fe/H] | n_el | ln LR | red. chi2 | no-Ba | LOO min (driver) | heavy≥2σ | raw | a_f |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 170203001901003 | 5885 | 4.61 | +0.09 | 5 | 11.3 | 11.0 | 3.8 | 2.6 (Nd) | 2 | 11.8 | 7.26 |
| 161120002401203 | 5479 | 4.45 | -0.62 | 3 | 9.4 | 5.1 | 5.3 | -0.0 (Nd) | 1 | 11.7 | 7.39 |
| 150719002901208 | 6209 | 4.06 | -0.20 | 3 | 9.4 | 19.0 | 0.4 | 0.0 (Nd) | 1 | 8.5 | 4.42 |
| 170507011101335 | 6111 | 4.18 | -0.21 | 3 | 9.3 | 6.0 | 6.5 | 0.0 (Nd) | 1 | 9.3 | 5.85 |
| 170506006401384 | 5957 | 4.11 | -0.17 | 3 | 9.2 | 10.1 | 5.2 | 0.0 (Nd) | 1 | 9.8 | 4.54 |
| 161210002601079 | 5813 | 4.15 | -0.24 | 6 | 9.1 | 8.0 | 3.0 | 3.0 (Ba) | 3 | 10.9 | 5.06 |
| 170114003601067 | 5882 | 4.45 | -0.23 | 6 | 8.9 | 9.3 | 0.5 | 0.5 (Ba) | 2 | 11.1 | 3.79 |
| 200810000601357 | 6176 | 4.23 | -0.28 | 2 | 8.1 | 11.2 | -0.0 | -0.0 (Ba) | 1 | 7.5 | 5.75 |

### Vet stage

- candidates re-vetted: 10; survivors at screen: 0; **after vet: 0**
- sigma: sig_ columns; refit: True; La suspect: False (no correlation above 0.2 with teff, logg, C, N, vsini)
- vetoes: low_snr_or_flagged 9, unexplained_by_all_templates 9, s_process_star 0, r_process_star 0, young_ba_enhancement 0, nlte_saturated_lines 10, single_element_driver 10, heavy_peak_incoherent 7, la_cn_blend 0, teff_peer_residual 0, below_threshold_after_refit 2

## GALAH/giant

- stars in the box: **78,344**; pattern elements: **12** (Rb, Sr, Y, Zr, Mo, Ru, Ba, La, Ce, Nd, Sm, Eu)
- testable (≥5 elements and ≥2 of La/Ce/Nd measured): **76,593** = 97.8% of the sample
- classification: NORMAL 75,891, INSUFFICIENT 1,157, S_PROCESS 563, UNEXPLAINED_BY_ALL_TEMPLATES 423, S_PLUS_R 154, AMBIGUOUS 84, R_PROCESS 71, FISSION 1
- error model: quoted error floored at the measured peer scatter — Rb 0.17 (×1.01), Sr 0.14 (×0.99), Y 0.08 (×2.41), Zr 0.09 (×1.26), Mo 0.12 (×0.74), Ru 0.16 (×0.67), Ba 0.12 (×2.66), La 0.10 (×1.84), Ce 0.10 (×1.11), Nd 0.09 (×2.37), Sm 0.09 (×1.7), Eu 0.19 (×1.43)
- La diagnostic: suspect=False — no correlation above 0.2 with teff, logg, C, N, vsini; correlations {'teff': 0.0022, 'logg': 0.0233, 'C': -0.0295, 'N': 0.096, 'vsini': 0.0884}
- threshold: ln LR ≥ **8.00** (config lr_min (shuffled-null q0.999 = 3.86 is below it)); enrichment ln LR ≥ 12.5; reduced chi2 ≤ 3.0
- null quantiles (ln LR): shuffled q0.5 0.00, q0.9 0.07, q0.99 1.20, q0.999 3.86, q0.9999 12.03 | sample q0.5 0.00, q0.9 0.06, q0.99 1.03, q0.999 3.19, q0.9999 17.77
- above threshold: **21** (raw-space: 29); unexplained by all templates: **20**; survivors after vetoes: **0**; after the vet stage: **0**
- vetoes (independent counts): low_snr_or_flagged 18, unexplained_by_all_templates 20, s_process_star 1, r_process_star 0, young_ba_enhancement 0, nlte_saturated_lines 6, single_element_driver 18, heavy_peak_incoherent 3, la_cn_blend 0, teff_peer_residual 0, teff_peer_residual_raw_only 10

### Sensitivity (injected fission pattern into real vectors)

| a_f | Δ[Nd/H] dex | LR pass (testable) | LR + LOO pass (testable) | LR pass (all) | LR + LOO pass (all) |
|---|---|---|---|---|---|
| 0.5 | +0.18 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | +0.30 | 0.01 | 0.00 | 0.01 | 0.00 |
| 2 | +0.48 | 0.16 | 0.01 | 0.15 | 0.01 |
| 3 | +0.60 | 0.45 | 0.08 | 0.44 | 0.08 |
| 5 | +0.78 | 0.74 | 0.36 | 0.74 | 0.38 |
| 10 | +1.04 | 0.91 | 0.67 | 0.90 | 0.67 |
| 20 | +1.32 | 0.96 | 0.80 | 0.95 | 0.82 |

Testable fraction 97.8% (1,500 injected).

### Unexplained by all templates (listed, never candidates)

| star | Teff | log g | [Fe/H] | n_el | ln LR | red. chi2 | no-Ba | LOO min (driver) | heavy≥2σ | raw | a_f |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 140304001501093 | 4240 | 3.23 | +0.01 | 12 | 84.7 | 46.2 | 79.0 | 48.9 (Sr) | 3 | 22.4 | 19.73 |
| 140304001501009 | 4187 | 2.98 | +0.22 | 11 | 45.2 | 17.2 | 44.5 | 27.9 (Ce) | 3 | 11.2 | 5.20 |
| 140304001501345 | 4217 | 2.61 | +0.66 | 12 | 33.1 | 9.9 | 32.1 | 17.6 (Y) | 3 | 8.8 | 4.85 |
| 140304001501294 | 4221 | 2.36 | +0.07 | 12 | 31.1 | 29.0 | 22.6 | 4.2 (Sr) | 3 | 19.5 | 6.27 |
| 140304001501303 | 4215 | 2.86 | -0.08 | 9 | 27.0 | 56.6 | 27.0 | 7.0 (Ce) | 3 | 12.1 | 4.42 |
| 150827004601193 | 4995 | 2.44 | -0.53 | 8 | 24.6 | 11.6 | 13.2 | 6.2 (Y) | 3 | 25.4 | 5.79 |
| 191108001601205 | 4718 | 3.39 | -0.17 | 9 | 21.3 | 12.7 | 27.3 | 4.7 (Sr) | 2 | 10.0 | 8.87 |
| 210925001601385 | 5479 | 2.90 | -0.02 | 6 | 20.6 | 27.0 | 16.7 | -0.2 (Nd) | 1 | 18.9 | 3.22 |
| 210925001601174 | 4980 | 2.79 | -0.63 | 6 | 17.2 | 10.5 | 13.6 | 6.4 (Nd) | 2 | 20.1 | 2.79 |
| 140610005701116 | 5047 | 3.24 | -0.27 | 7 | 15.9 | 12.2 | 7.1 | 7.1 (Ba) | 3 | 18.7 | 5.09 |

### Vet stage

- candidates re-vetted: 21; survivors at screen: 0; **after vet: 0**
- sigma: sig_ columns; refit: True; La suspect: False (no correlation above 0.2 with teff, logg, C, N, vsini)
- vetoes: low_snr_or_flagged 18, unexplained_by_all_templates 20, s_process_star 1, r_process_star 0, young_ba_enhancement 0, nlte_saturated_lines 8, single_element_driver 17, heavy_peak_incoherent 2, la_cn_blend 0, teff_peer_residual 0, below_threshold_after_refit 3

## What a survivor still has to pass

Nothing here is a detection. A survivor is a *target*: the Ba II, La II, Ce II, Nd II and Eu II lines must be re-measured from the raw HERMES spectrum against Teff-matched peers, the pattern must hold element by element, and the star must be checked for an unresolved companion. Until then the correct description is 'an n-capture vector the s+r mixture does not fit'.

## No-null rule (CLAUDE.md)

An empty survivor list at this threshold is a statement about GALAH DR4's element panel and precision, not a publishable null. The escalation path is the APOGEE Ce/Nd panel as a second survey, the co-natal differential channel, and high-resolution re-measurement of the strongest ambiguous and unexplained stars.

