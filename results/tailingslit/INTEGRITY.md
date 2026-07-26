# TAILINGS literature corpus — integrity check

**12 of 24 hardcoded arXiv identifiers in the first version of `scripts/tailingslit_fetch.py` resolved to entirely unrelated papers, and every one of them fetched successfully.** A successful fetch is no evidence that the paper is the right one. Citing from those files would have put fabricated attributions into the channel documentation, so they have been deleted rather than left on disk where a later reader could quote them.

The harness now resolves each decisive paper by **title search with a title-token verification step** (`resolve_by_title`), writes `verification.json`, and fetches nothing for a slug it could not verify.

## Quarantined — the ID fetched vs the paper it actually was

| slug | arXiv id fetched | what the file actually contained |
|---|---|---|
| `apogee_dr17` | 2112.05131v1 | Plenoxels: Radiance Fields without Neural Networks |
| `behmard2023_engulfment_signature` | 2210.11330v2 | Axion Quality Problem and Non-Minimal Gravitational Coupling in the Palatini For |
| `behmard2025_engulfment` | 2501.03252v3 | Science Opportunities of Wet Extreme Mass-Ratio Inspirals |
| `deal2020_diffusion_solar_type` | 2007.02528v2 | Triaxially-deformed Freely-precessing Neutron Stars: Continuous electromagnetic  |
| `lambda_boo_review` | 1908.03976v1 | Surface states in defect-free polyatomic lattices described by a tight-binding m |
| `liu2024_nature_ingestion` | 2405.10339v1 | Noncommutative Number Systems for Quantum Information |
| `michaud2011_diffusion_popii` | 1011.4212v1 | 3D-MHD simulations of the evolution of magnetic fields in FR II radio sources |
| `pricejones2018_dimensionality` | 1710.08442v2 | Modular operads and Batalin-Vilkovisky geometry |
| `richer2000_amfm_diffusion` | 0004035v1 | Radiative Precession of an Isolated Neutron Star |
| `ting2012_pca_dimensionality` | 1207.5074v1 | Identifying the young low-mass stars within 25 pc. II. Distances, kinematics and |
| `vick2010_amfm_massloss` | 1002.1922v1 | MSSM dark matter measurements at the LHC without squarks and sleptons |
| `weinberg2019_two_process` | 1810.01470v1 | CELLO-3D: Estimating the Covariance of ICP in the Real World |

## Verified and safe to cite

| slug | arXiv id | title |
|---|---|---|
| `baron_poznanski_weirdest` | 1611.07526v1 | The weirdest SDSS galaxies: results from an outlier detection algorithm |
| `bedell2018_chemical_homogeneity` | 1802.02576v2 | The Chemical Homogeneity of Sun-like Stars in the Solar Neighborhood |
| `galah_dr4` | 2409.19858v2 | The GALAH Survey: Data Release 4 |
| `huang2026_refined_material` | 2605.29811v1 | A Calibrated Bayesian Search for Potential Chemical Technosignatures in Polluted |
| `melendez2009_solar_twins` | 0910.5845v2 | The solar, exoplanet and cosmological lithium problems |
| `ness2018_doppelgangers` | 1701.07829v1 | Galactic Doppelganger: The chemical similarity among field stars and among stars |
| `reis_apogee_outliers` | 1711.00022v2 | Detecting outliers and learning complex structures with large spectroscopic surv |
| `spina2021_engulfment` | 2108.12040v1 | Chemical evidence for planetary ingestion in a quarter of Sun-like stars |
| `technosig_review_2026` | 2601.07297v2 | WISE/CatWISE Constraints on Dysonian Waste-Heat Technosignatures in Nearby Galax |
| `weinberg2021_two_process_residuals` | 2108.08860v1 | Chemical Cartography with APOGEE: Mapping Disk Populations with a Two-Process Mo |
| `wright2019_technosig_search_landscape` | 1907.07830v1 | Technosignatures in Transit |

## Where the channel's citations actually came from

The numbers quoted in `docs/tailings.md` were drawn from arXiv **search-result** Atom files (`arxiv_q_*.atom` here and in `results/necrolit/`), which carry the real titles and abstracts of whatever matched and cannot suffer this failure mode, and from the verified full texts above — not from the quarantined files. The decisive read, arXiv:2605.29811, is in the verified set.


Files removed: 36.
