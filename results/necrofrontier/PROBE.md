# NECROFRONTIER data-source probe

Verdict: **PARTIAL_REACH** over 67 endpoints ({"NOT_REACHED": 8, "REACHED_WITH_PRODUCT": 57, "REACHED_NO_PRODUCT": 2}).

| endpoint | verdict | HTTP | bytes | signatures | hits |
|---|---|---|---|---|---|
| `pgd_landing` | NOT_REACHED | None | 0 | S46 |  |
| `pgd_wustl_root_http` | NOT_REACHED | None | 0 | S46 |  |
| `pgd_ecl_search` | REACHED_WITH_PRODUCT | 200 | 29271 | S46 | (?i)earthchem; (?i)library |
| `pgd_ads_record` | NOT_REACHED | 405 | 2203 | S46 |  |
| `jmmc_oidb` | REACHED_WITH_PRODUCT | 200 | 19668 | S47 | (?i)oidb; (?i)tap|search |
| `spherex_irsa_docs` | REACHED_WITH_PRODUCT | 200 | 2553 | S48, S49 | (?i)spherex; (?i)quick\s*release|QR|spectral\s+image |
| `spherex_ibe` | REACHED_WITH_PRODUCT | 200 | 3894 | S48, S49 | (?i)qr|quick|spherex |
| `dasch_home` | REACHED_WITH_PRODUCT | 200 | 5138 | S50 | (?i)dasch; (?i)dr\s*7|data\s+release; (?i)starglass|daschlab|api |
| `dasch_dr7_web_apis` | REACHED_WITH_PRODUCT | 200 | 75989 | S50 | (?i)querycat; (?i)lightcurve |
| `dasch_dr7_columns` | REACHED_WITH_PRODUCT | 200 | 62099 | S50 | (?i)lim_mag|limiting; (?i)magcal|mag |
| `daschlab_pypi` | REACHED_WITH_PRODUCT | 200 | 4809 | S50 | (?i)\"name\":\s*\"daschlab\"; (?i)version |
| `pewdd_github` | REACHED_WITH_PRODUCT | 200 | 5236 | S51 | (?i)pewdd|white\s+dwarf |
| `pyllutedwd_github` | REACHED_WITH_PRODUCT | 200 | 5746 | S51 | (?i)pylluted|white\s+dwarf|\"name\" |
| `cassis_atlas` | NOT_REACHED | None | 0 | S52, S53 |  |
| `cassis_root` | NOT_REACHED | None | 0 | S52, S53 |  |
| `irsa_seip_docs` | REACHED_WITH_PRODUCT | 200 | 6300 | S52, S53 | (?i)spitzer; (?i)enhanced |
| `cdms_entries` | REACHED_WITH_PRODUCT | 200 | 483120 | S54 | (?i)CDMS |
| `jpl_catdir` | REACHED_WITH_PRODUCT | 200 | 31685 | S54 | \bNF3\b; \bCOF2\b; \bCH2F2\b; \bCH3Cl\b |
| `splatalogue` | REACHED_WITH_PRODUCT | 200 | 1080 | S54 | (?i)splatalogue |
| `diviner_pds` | REACHED_WITH_PRODUCT | 200 | 39445 | S55 | (?i)diviner; (?i)polar|GDR|RDR|temperature |
| `diviner_pcp_ode` | REACHED_WITH_PRODUCT | 200 | 16165 | S55 | (?i)polar\s+cumulative; (?i)240 |
| `diviner_pcp_ucla` | NOT_REACHED | None | 0 | S55 |  |
| `diviner_pds_volumes` | REACHED_WITH_PRODUCT | 200 | 1791 | S55 | (?i)dlre |
| `diviner_rdr_volume_root` | REACHED_WITH_PRODUCT | 200 | 927 | S55 | (?i)lrodlr_100\d; (?i)lrodlr_1002|lrodlr_1003 |
| `diviner_pds_data_dir` | REACHED_WITH_PRODUCT | 200 | 2058 | S55 | (?i)gdr|pcp|prp|polar|href |
| `shadowcam_archive` | REACHED_WITH_PRODUCT | 200 | 7747 | S55 | (?i)shadowcam; (?i)release|pds |
| `shadowcam_pds_sis` | REACHED_WITH_PRODUCT | 200 | 1656696 | S55 | %PDF |
| `minirf_pds` | REACHED_WITH_PRODUCT | 200 | 38617 | S55 | (?i)mini-rf|mini rf |
| `sgp_search` | REACHED_WITH_PRODUCT | 200 | 623 | S56 | (?i)sedimentary\s+geochemistry|SGP |
| `sgp_api` | REACHED_WITH_PRODUCT | 200 | 623 | S56 | (?i)sample|age|interpreted_age|\"data\" |
| `earthchem_portal` | REACHED_WITH_PRODUCT | 200 | 10193 | S56 | (?i)earthchem |
| `earthchem_home` | REACHED_WITH_PRODUCT | 200 | 267464 | S56 | (?i)earthchem |
| `sgp_env_js` | REACHED_WITH_PRODUCT | 200 | 200 | S56 | (?i)api|url|http |
| `sgp_archive` | REACHED_WITH_PRODUCT | 200 | 689 | S56 | (?i)sgp|sedimentary|csv|zip|download|archive |
| `sgp_github` | REACHED_NO_PRODUCT | 200 | 55 | S56 |  |
| `georoc` | REACHED_WITH_PRODUCT | 200 | 7764 | S56 | (?i)georoc |
| `exoarchive_koi_depth` | REACHED_WITH_PRODUCT | 200 | 666 | S57 | (?i)koi_depth |
| `exoarchive_toi_depth` | REACHED_WITH_PRODUCT | 200 | 714 | S57 | (?i)pl_trandep |
| `exoarchive_tess_kepler_xmatch` | REACHED_WITH_PRODUCT | 200 | 781 | S57 | (?i)tic_id |
| `cda_pds_sbn` | REACHED_WITH_PRODUCT | 200 | 42646 | S58 | (?i)cosmic\s+dust\s+analy|CDA |
| `ulysses_dust_pds` | NOT_REACHED | 404 | 196 | S58 |  |
| `sbn_dust_holdings` | REACHED_WITH_PRODUCT | 200 | 22238 | S58 | (?i)ulysses; (?i)cassini; (?i)galileo; (?i)new horizons; (?i)href="[^"]*cda[^"]* |
| `vizier_tap_home` | REACHED_WITH_PRODUCT | 200 | 14029559 | S59, S47, S52, S51 | (?i)tableset|schema|table |
| `bl_opendata` | REACHED_WITH_PRODUCT | 200 | 21075 | S60 | (?i)breakthrough|open\s*data|listen |
| `bl_bldata` | REACHED_NO_PRODUCT | 200 | 29 | S60 |  |
| `irsa_neowise_tap` | REACHED_WITH_PRODUCT | 200 | 4805 | S61 | (?i)w1mpro |
| `erosita_dr1` | REACHED_WITH_PRODUCT | 200 | 11560 | S59 | (?i)erosita; (?i)DR1|catalog |
| `metbull` | REACHED_WITH_PRODUCT | 200 | 74813 | S46 | (?i)meteoritical\s+bulletin |
| `vizier_flare_rotation_tables` | REACHED_WITH_PRODUCT | 200 | 2592 | S59, S47, S52, S51 | J/ApJS/241/29; J/ApJS/211/24; J/ApJS/225/15 |
| `vizier_search_exozodi` | REACHED_WITH_PRODUCT | 200 | 494 | S47 | (?i)exozodi|pionier|fluor|hot dust |
| `vizier_search_polluted_wd` | REACHED_WITH_PRODUCT | 200 | 583 | S51 | J/A\+A/691/A352; (?i)white dwarf |
| `vizier_moor2021_edd` | REACHED_WITH_PRODUCT | 200 | 2087 | S52 | J/ApJ/910/27|J/MNRAS/433/2334|J/ApJS/225/15|J/ApJ/805/77 |
| `irsa_irs_enhanced_columns` | REACHED_WITH_PRODUCT | 200 | 8121 | S52, S53 | (?i)ra|wave|flux|aor|object |
| `irsa_irs_enhanced_count` | REACHED_WITH_PRODUCT | 200 | 16 | S52, S53 | \"n\" |
| `irsa_splices_rows` | REACHED_WITH_PRODUCT | 200 | 18 | S48 | \"n\" |
| `irsa_splices_columns` | REACHED_WITH_PRODUCT | 200 | 15662 | S48 | (?i)ra|dec|mag|flux |
| `irsa_spherex_obscore_count` | REACHED_WITH_PRODUCT | 200 | 18 | S48, S61 | \"n\" |
| `vizier_superflare_tables` | REACHED_WITH_PRODUCT | 200 | 808 | S59 | (?i)superflare|J/ApJ/906/72|J/ApJ/876/58 |
| `vizier_search_debris_extreme` | REACHED_WITH_PRODUCT | 200 | 275 | S52 | (?i)debris |
| `irsa_spherex_tables` | REACHED_WITH_PRODUCT | 200 | 536 | S48, S49 | (?i)spherex |
| `irsa_spherex_plane_columns` | REACHED_WITH_PRODUCT | 200 | 6347 | S48, S49 | (?i)energy_bandpassname|obs|time |
| `irsa_spherex_recent_exposures` | REACHED_WITH_PRODUCT | 200 | 1055 | S48, S49 | (?i)uri |
| `irsa_splices_count` | REACHED_WITH_PRODUCT | 200 | 27 | S48 | (?i)splices|ices |
| `irsa_euclid_spe_lines` | REACHED_WITH_PRODUCT | 200 | 2038 | S48 | (?i)spe_line_snr|spe_line_name |
| `irsa_euclid_tables` | REACHED_WITH_PRODUCT | 200 | 2878 | S48 | (?i)euclid |
| `irsa_cassis_tables` | REACHED_WITH_PRODUCT | 200 | 2734 | S52, S53 | (?i)cassis|irs |
| `gaia_pairs_geometry_feasibility` | NOT_REACHED | None | 0 | S60 |  |

| signature | readiness | endpoints |
|---|---|---|
| S46 | PARTIAL | pgd_landing, pgd_wustl_root_http, pgd_ecl_search, pgd_ads_record, metbull |
| S47 | READY | jmmc_oidb, vizier_tap_home, vizier_flare_rotation_tables, vizier_search_exozodi |
| S48 | READY | spherex_irsa_docs, spherex_ibe, irsa_splices_rows, irsa_splices_columns, irsa_spherex_obscore_count, irsa_spherex_tables, irsa_spherex_plane_columns, irsa_spherex_recent_exposures, irsa_splices_count, irsa_euclid_spe_lines, irsa_euclid_tables |
| S49 | READY | spherex_irsa_docs, spherex_ibe, irsa_spherex_tables, irsa_spherex_plane_columns, irsa_spherex_recent_exposures |
| S50 | READY | dasch_home, dasch_dr7_web_apis, dasch_dr7_columns, daschlab_pypi |
| S51 | READY | pewdd_github, pyllutedwd_github, vizier_tap_home, vizier_flare_rotation_tables, vizier_search_polluted_wd |
| S52 | PARTIAL | cassis_atlas, cassis_root, irsa_seip_docs, vizier_tap_home, vizier_flare_rotation_tables, vizier_moor2021_edd, irsa_irs_enhanced_columns, irsa_irs_enhanced_count, vizier_search_debris_extreme, irsa_cassis_tables |
| S53 | PARTIAL | cassis_atlas, cassis_root, irsa_seip_docs, irsa_irs_enhanced_columns, irsa_irs_enhanced_count, irsa_cassis_tables |
| S54 | READY | cdms_entries, jpl_catdir, splatalogue |
| S55 | PARTIAL | diviner_pds, diviner_pcp_ode, diviner_pcp_ucla, diviner_pds_volumes, diviner_rdr_volume_root, diviner_pds_data_dir, shadowcam_archive, shadowcam_pds_sis, minirf_pds |
| S56 | PARTIAL | sgp_search, sgp_api, earthchem_portal, earthchem_home, sgp_env_js, sgp_archive, sgp_github, georoc |
| S57 | READY | exoarchive_koi_depth, exoarchive_toi_depth, exoarchive_tess_kepler_xmatch |
| S58 | PARTIAL | cda_pds_sbn, ulysses_dust_pds, sbn_dust_holdings |
| S59 | READY | vizier_tap_home, erosita_dr1, vizier_flare_rotation_tables, vizier_superflare_tables |
| S60 | PARTIAL | bl_opendata, bl_bldata, gaia_pairs_geometry_feasibility |
| S61 | READY | irsa_neowise_tap, irsa_spherex_obscore_count |

REACHED_NO_PRODUCT is a finding about the *brief*, not the sky: the URL, product name or table name asserted in `docs/necrofrontier.md` must be corrected from what the endpoint actually served (its `head` is in `probe.json`).
