# Channel index

One row per search channel: what it asks, what it reads, how to run it, where
its outputs live, and what those outputs currently say. This is the *map*;
the narrative (why each channel exists, what was found and traced to a
systematic, what to do next) is the log in [`STATUS.md`](../STATUS.md), and
the conventions for adding a channel are
[`channel-brief.md`](channel-brief.md).

Verdict strings are quoted verbatim from the committed result file named in
the row. Where a channel writes no `verdict` field, the row says which file
carries the headline instead. "Pending" means the package, tests and workflow
exist but no run has committed results. Every CLI command is
`python -m seti.cli <command>`; every workflow is `workflow_dispatch` unless a
cron is noted. All channels are unit-tested offline under `tests/` (the suite
is network-guarded: see `tests/conftest.py`).

Index compiled 2026-09-01 from the files on `main` (rows for METRONOME, LANTERN and FALLOUT added 2026-09-06); re-verify a verdict
against the result file before quoting it elsewhere.

## Waste heat and energy budget

| Channel | Question | Data | CLI | Workflow | Doc | Results → current verdict |
|---|---|---|---|---|---|---|
| **WD IR excess** (`acquire`, `sed`, `contamination`, `discriminate`, `stats`) | Dyson waste heat around white dwarfs that survives a five-stage contamination funnel and a debris-disk discriminant | Gaia DR3 × AllWISE × 2MASS | `acquire-run`, `science-run`, `science-blend`, `contamination-budget` | `science.yml`, `science-blend.yml` | `paper/` manuscript | `results/science/blend_followup_summary.json`: 13 isolated, 7 stellar companions, 3 blends; all τ < 0.08 (ordinary debris disks). **0 technosignature** |
| **CENOTAPH** (`cenotaph`) | Cold Dyson shells (T < 100 K): grey dimming with no mid-IR excess whose energy reappears at 60–160 µm | Gaia DR3 GSP-Spec dwarfs, 2MASS, AllWISE, AKARI/FIS, IRAS | `cenotaph --stage {sample,twins,grey,midir,farir,reduce,probe}` | `cenotaph.yml`, `cenotaph-probe.yml`, `cenotaph-recon.yml`, `cenotaph-recon2.yml` | `cenotaph.md` | `results/cenotaph/summary.json`: `closure_but_crowded_beam`, `sample_verdict: PARTIAL_SAMPLE` (3 of 88 parallax shells timed out) |
| **OSSUARY** (`ossuary`) | Warm dust around stars that cannot make it: metal-poor ([Fe/H] < −1) and halo-kinematic stars (S7, terminal state of an unmaintained swarm) | Gaia DR3 GSP-Spec/GSP-Phot joined in-archive to AllWISE + 2MASS | `ossuary` | `ossuary.yml` | `ossuary.md` | **Pending**: `results/ossuary/` does not exist; first run dispatched, nothing committed |
| **EMBER** (`ember`) | Waste heat that switched off: mid-IR excess present in an early epoch and absent decades later (S1) | IRAS (1983), AKARI/IRC (2006–07), WISE (2010), Gaia DR3 for proper-motion propagation | `ember --stage {audit,probe,acquire,analyse}` | `ember.yml` | `ember.md` | `results/ember/summary.json`: `NO_SURVIVOR` — but `with_early_photometry: 0` of 412,914 acquired, so this is acquisition-limited, not a sky statement |
| **VIGIL** (`vigil`) | Mid-IR variable while optically constant: waste heat with a duty cycle (S4), or that duty cycle ceasing | NEOWISE per-epoch W1/W2 (IRSA), Gaia DR3 | `vigil-probe`, `vigil-sweep`, `vigil-vet` | `vigil.yml` | `vigil.md` | `results/vigil/summary.json`: `NO_DATA_REACHED`, all counts 0 |
| **ISOTHERM** (`isotherm`) | The *shape* of the waste heat: isothermal excess (S5) or discrete temperature steps in geometric progression (S6) | CASSIS Spitzer/IRS low-res spectra; IRSA/VizieR fallbacks | `isotherm --stage {probe,corpus,screen,shape,calibrate,score}` | `isotherm.yml` | `isotherm.md` | `results/isotherm/summary.json`: `no_shape_anomaly_in_corpus`, `archive_verdict: NOT_PROBED`, 5,480 spectra all `INSUFFICIENT_DATA` |
| **Gaia XP anomalies** (`xp`) | BP/RP spectral shapes no normal-stellar locus reconstructs | Gaia DR3 XP spectra | `xp-run` | `xp.yml` | none | `results/xp/<field>/summary.json` (per-field shards, no aggregate). STATUS: 0 credible, channel bounded — broad anomalies are reddened M-dwarf bands, narrow ones are band-edge artefacts |
| **Cluster** (`cluster`) | Does the IR-excess tail over-cluster in 6D against a matched null, as an expanding population would? | Gaia DR3 × AllWISE cones | `cluster-run`, `cluster-aggregate` (the aggregate step is wired to no workflow) | `cluster.yml` | none | `results/cluster/AGGREGATE.json`: `global_over_clustered: false`; single cone p_pos 0.76 |
| **TIDEMARK** (`tidemark`) | Is any channel's anomaly *rate* spatially structured (gradient, edge, age trend) rather than tracing stellar density? | Own Gaia DR3 × AllWISE parent grid plus sibling channels' anomaly catalogues | `tidemark-acquire`, `tidemark-run`, `tidemark-search` | `tidemark.yml` | `tidemark.md` | `results/tidemark/summary.json`: `STRUCTURE_UNVETTED_POPULATION` ("most likely traces the survey rather than the sky"); 7 of 8 input catalogues `NO_PARENT_SAMPLE` |

## Optical time domain

| Channel | Question | Data | CLI | Workflow | Doc | Results → current verdict |
|---|---|---|---|---|---|---|
| **Dimming** (`dimming`, with `secular` and `glint`) | Boyajian-analogue deep dips; slow secular fade (enshrouding); brief achromatic brightening (specular glint) | ZTF g+r, Gaia DR3, NEOWISE and ASAS-SN in characterisation | `dimming-run`, `dimming-vet`, `dimming-characterize` (glint is produced inside `dimming-run`, vetted by `dimming-vet`) | `dimming.yml`, `dimming-sweep.yml`, `dimming-vet.yml`, `dimming-characterize.yml` | none | `results/dimming/AGGREGATE_occurrence_limit.json`: 0 confirmed analogues in 250,862 stars / 116 fields; top fader `ir_fades_reddening_law`. Glint: 15 candidates, all `chromatic_flare`. Exhausted at the ZTF systematics floor |
| **RUST** (`rust`) | Unmaintained decay: per-season photometric *scatter* (second moment) rising secularly (S9) | ZTF g+r via IRSA; Gaia DR3, SIMBAD, NEOWISE for vetting | `rust-sweep`, `rust-vet` | `rust.yml` | `rust.md` | `results/rust/summary.json`: `NO_CANDIDATES` (7 fields, 5,043 stars; "a count, not a limit") |
| **KNELL** (`knell`) | A period that ceased (S32), every non-detection normalised by injection-measured efficiency in that block's own sampling | ZTF g+r via IRSA; Gaia DR3, SIMBAD, VSX, GCVS | `knell-sweep`, `knell-vet`, `knell-cross` | `knell.yml` | `knell.md` | `results/knell/summary.json`: `NO_CANDIDATES` (6 of 8 fields returned data; 1,163 testable stars) |
| **TOCSIN** (`tocsin`) | Cross-night recurrence of *achromatic* difference-image events (flash and dip) on catalogued nearby stars in the Rubin alert stream (S30) | Rubin/LSST alerts via ALeRCE TAP (Fink as second broker), Gaia DR3 targets | `tocsin-probe`, `tocsin-targets`, `tocsin-screen`, `tocsin-population`, `tocsin-assess` | `tocsin.yml` (**nightly cron**), `tocsin-probe.yml` | `tocsin.md`, `rubin-outage.md` | `results/tocsin/summary.json`: `NO_NEW_DATA` — the broker frontier has been frozen at MJD 61235.4 (2026-07-14) since Rubin's summit evacuation; verdict `SKY_STOPPED` in `results/rubin_outage/` |
| **TOCSIN-ZTF** (`tocsin.ztf_live`) | The same S30 signature on the LIVE ZTF public alert stream (measured live 2026-09-05) while Rubin is dark | ALeRCE ZTF API (numerator), IRSA `ztf_current_meta_sci` quadrant table (denominator) | `tocsin-ztf-probe`, `tocsin-ztf-targets`, `tocsin-ztf-screen`, `tocsin-ztf-assess` | `tocsin-ztf.yml` (nightly) | `tocsin-ztf.md` | Live since 2026-09-05, twice daily: six nights from 2026-07-14 folded (163,768 star-nights, 32 events, one star at `interest`); backfilling toward the stream |
| **TOCSIN alt-feeds** (`tocsin.altfeeds`) | The same signature on non-Rubin feeds while Rubin is off sky | ASAS-SN Sky Patrol v2, ATLAS forced photometry | `tocsin-altfeeds-probe`, `tocsin-altfeeds-census`, `tocsin-altfeeds` | `tocsin-altfeeds.yml` | `tocsin-altfeeds.md`, `substitute-surveys.md` | `results/tocsin_altfeeds/census.json`: `OK` (139,706 targets); `probe.json`: `PARTIAL` (ATLAS and ZTF usable, ASAS-SN down). Walk rebuilt 2026-09-05 with a job deadline, concurrent ATLAS tasks and a per-star walk state (`tocsin-altfeeds.md` §12); ledgers restart from that run |
| **SHROUD** (`shroud`) | Enshrouded, not destroyed (S33): POSS-I sources absent from the modern optical but present in the infrared | SVO `vanish-neowise` / `vanish-possi` (Solano+2022 by-product), CDS X-Match, Gaia DR3 | `shroud --stage {acquire,photometry,analyze}` | `shroud.yml` | `shroud.md` | **Pending**: `results/shroud/` holds only a fetched HTML page; no `summary.json` committed |
| **METRONOME** (`metronome`) | Strict clocks in catalogued flare *timing* (S28): a star whose brief brightenings recur with jitter far below rotational quasi-periodicity | Kepler and TESS flare catalogues (Yang & Liu 2019, Pietras+2022, Günther+2020) and rotation / periodic-variable catalogues, all VizieR | `metronome --stage {probe,acquire,screen,assess}` (or `python -m seti.metronome.run`) | `metronome.yml` | `metronome.md` | **Pending**: built and offline-tested 2026-09-06; first runner dispatch pending |

## Spectroscopy and chemistry

| Channel | Question | Data | CLI | Workflow | Doc | Results → current verdict |
|---|---|---|---|---|---|---|
| **Narrow lines** (`spectra`, with `absorb`) | A single unresolved narrow emission line (CW laser) and its absorption-mode analogue | SDSS-DR17 and DESI-DR1 via SPARCL; SIMBAD | `spectra-run --mode {emission,absorption}`, `spectra-triage`, `spectra-confirm` | `spectra.yml`, `spectra-confirm.yml` | none | `results/spectra_triage/summary.json`: 167 survive (112 emission + 55 absorption) after duplicate / known-line / recurrent-wavelength / galaxy cuts. Next: per-exposure persistence |
| **TAILINGS** (`tailings`) | The sparse chemical anomaly: one element extreme, the other 20–30 quiet, in cool dwarfs where the convective envelope forbids it naturally (S12/S15/S22) | GALAH DR4, APOGEE DR17 via TAP with runtime schema discovery | `tailings`, `tailings-validate` (offline Griffith+2021 Na-star injection) | `tailings.yml`, `tailingslit.yml` | `tailings.md` | `results/tailings/summary.json`: `DEGRADED_SOURCE (...)` + `SPARSE_CANDIDATES_PENDING_REMEASUREMENT`: 1,809 catalogue-level survivors, none a detection until re-measured from the raw spectrum |
| **MIDDEN** (`midden`) | Whitmire & Wright 1980 nuclear-waste signature: Tc / Pm / actinide lines in photospheres that are not AGB or S-type | ESO HARPS/FEROS via ObsCore TAP; VizieR; NIST ASD | `midden --stage {verify-lines,targets,acquire,score,all}`, `midden-deep` | `midden.yml`, `midden-deepdive.yml` | `midden.md` | `results/midden/REPORT.md`: no population-level Tc excess (178 spectra, 99 stars); the one flag, HD 217522, is roAp rare-earth blending and is not claimed |
| **LANTERN** (`lantern`) | A narrow emission line that is present out of secondary eclipse and *vanishes* while the planet is occulted (S28a): a planet-side monochromatic source, across every public JWST exoplanet time series | MAST JWST `x1dints` (NIRSpec, NIRCam, NIRISS, MIRI) × NASA Exoplanet Archive ephemerides | `lantern {probe,inventory,screen,assess,selftest}` (or `python -m seti.lantern.run`) | `lantern.yml` | `lantern.md` | **Pending**: built and offline-tested 2026-09-06; first runner dispatch pending |
| **FALLOUT** (`fallout`) | The stable *residue* of fission (S14): the two-humped fission-yield abundance vector ([Nd/Ba] ≫ 0, [Eu/Nd] < 0, [Mo/Zr] > 0) in cool dwarfs, against the best natural s + r mixture in a Teff-matched peer frame | GALAH DR4 allstar (Data Central cloud FITS, the route that worked for TAILINGS); APOGEE DR17 optional | `fallout --stage {probe,acquire,screen,assess,all}` (or `python -m seti.fallout.run`) | `fallout.yml` | `fallout.md` | **Pending**: built and offline-tested 2026-09-06; first runner dispatch pending |

## Kinematics and geometry

| Channel | Question | Data | CLI | Workflow | Doc | Results → current verdict |
|---|---|---|---|---|---|---|
| **Astrometric dark companion** (`accel`) | Astrometric acceleration implying a massive dark companion with no luminous counterpart | Gaia DR3 `nss_acceleration_astro`; published compact-companion catalogues | `accel-run`, `accel-xmatch` | `accel.yml`, `accel-xmatch.yml` | none | `results/accel/literature_crossmatch_summary.json`: 8 class-3, 7 already in Shahaf+2023, 1 known BH; the one "novel" is the weakest solution. Reproduces the published catalogue |
| **HERDSMAN** (`herdsman`) | ≥4 unrelated field stars whose integrated orbits converge on one small volume (herding, or a past rendezvous) | Gaia DR3 6D, RV zero-point corrected | `herdsman-fetch`, `herdsman-scan`, `herdsman-reduce` (`herdsman` is the monolithic form) | `herdsman.yml` | `herdsman.md` | `results/herdsman/REPORT.md`: zero candidates in both time directions inside the 11–12 Myr chance-occupancy horizon |
| **HERDSMAN-B** (`herdsman_b`) | Completed assemblies: catalogued clusters that are chemically field-sampled rather than co-natal | Hunt & Reffert membership × Gaia GSP-Phot [M/H]; GALAH DR3 / APOGEE DR17 for the spectroscopic tier | `herdsman-b --stage ...` (incl. `spectro`) | `herdsman-b.yml` | `herdsman.md` §5 | `results/herdsman_b/REPORT.md`: the one formal candidate (Hogg_4) fails member-level vetting as an extinction–metallicity systematic; spectroscopic tier 0/80 |
| **COMPASS** (`compass`) | Spatially coherent patches of aligned binary orbital poles among unrelated neighbours (shared engineered geometry) | Gaia DR3 `nss_two_body_orbit` joined to `gaia_source` | `compass` | `compass.yml` | `compass.md`, `next-question.md` | `results/compass/REPORT.md`: null at every radius 10–100 pc ("also null, decisively"); rests until Gaia DR4 |

## Solar-system bodies

| Channel | Question | Data | CLI | Workflow | Doc | Results → current verdict |
|---|---|---|---|---|---|---|
| **DERELICT** (`derelict`) | Dead lightsails: large *radial* non-gravitational acceleration (A1) with no outgassing, normalised by size into an area-to-mass ratio (S19) | JPL SBDB non-gravitational fits; comet control | `derelict --stage ...` (incl. `probe`) | `derelict.yml` | `derelict.md` | `results/derelict/summary.json`: `ALL_SURVIVORS_EXPLAINED`; `completeness.verdict: CONSTRAINT_COMPLETE`. 'Oumuamua recovered as positive control |
| **LOOM** (`loom`) | A *population* of self-replicating probes: cross-object structure in Rubin per-detection ephemeris residuals, gated by the radiation-momentum ceiling | Rubin SSO alerts via ALeRCE; MPC/JPL SBDB orbits | `loom-probe`, `loom-screen`, `loom-assess`, `loom-calibrate`, `loom-litcheck` | `loom.yml` (**weekly cron**), `loom-probe.yml`, `loom-calibrate.yml` (**monthly cron**), `loom-litcheck.yml` | `loom.md` | `results/loom/assessment.json`: `INSUFFICIENT_POPULATION` (66,686 orbits screened; every residual series is one apparition long). Waits for Rubin's second apparition |
| **LOOM catalogue** (`loom.catalogue`) | Over the whole small-body catalogue, how many objects exceed the momentum ceiling: the denominator for any single exceedance | JPL SBDB full catalogue | `loom-catalogue` | `loom-catalogue.yml` (chains `loom-litcheck`) | `loom-catalogue.md` | `results/loom-catalogue/catalogue.json`: `TAIL_DENSELY_POPULATED` (30 of 594 exceed; 7 reliably fitted and unexplained in 116 papers) |
| **SEXTANT** (`sextant`) | Non-gravitational acceleration in Gaia's milliarcsecond minor-planet astrometry: the same question as LOOM at 100× the precision | Gaia DR3 + FPR `sso_observation` / `sso_source` via ESA Gaia TAP | `sextant-probe` | `sextant-probe.yml`, `sextantlit.yml` | `sextant.md`, `sextant-priorart.md`, `sextant-acquire.md` | `results/sextant/probe.json`: `OK` (2026-08-26). Only the probe has run; the residual search has not been dispatched |

## Life around LHS 1140 and K2-18 (origin-of-life fan-out)

| Channel | Question | Data | CLI | Workflow | Doc | Results → current verdict |
|---|---|---|---|---|---|---|
| **Panspermia** (`panspermia`) | Which stars passed close and slow to K2-18 and could carry its material? | Gaia DR3 6D; NASA Exoplanet Archive | `panspermia-run`, `panspermia-mc`, `panspermia-targets`, `panspermia-dossier`, `panspermia-regime` | `panspermia.yml`, `panspermia-mc.yml`, `panspermia-targets.yml`, `panspermia-dossier.yml` | none | `results/panspermia/summary.json`: 15 within 2 pc, closest 0.90 pc at 32 km/s, 0 co-movers. No capturable bridge; RV completeness is the gap |
| **LHS 1140 origin** (`lhs1140_origin`) | Same engine with LHS 1140 b as donor: recipients and directed-travel destinations | Gaia DR3 6D; Exoplanet Archive | `lhs1140-origin` | `lhs1140-origin.yml` | none | `results/lhs1140_origin/summary.json`: 22 recipients, 5,255 destinations, closest 0.26 pc (no verdict field) |
| **Galactic encounters** (`galactic`) | Full-orbit integration back 300 Myr: who passed the biosignature anchors? | Gaia DR3 RV-complete 6D; Exoplanet Archive | `galactic-encounters` | `galactic-encounters.yml` | none | `results/galactic/summary.json`: 0 planet-host encounters, 0 companion flags for both anchors; per-anchor biosignature `answer` fields |
| **ISO back-track** (`iso`) | Do 1I, 2I, 3I back-track toward LHS 1140 or any nearby star? | Published ISO elements; Gaia DR3 | `iso-backtrack` | `iso.yml` | none | `results/iso/summary.json`: `any_consistent_with_origin: false` (all three at ~15 pc) |
| **LHS 1140 deep-dive** (`lhs1140`) | Every techno/bio signature on the star, planets b/c and 38 neighbours, plus a biosignature detectability budget | Gaia (astrometry, XP), WISE/NEOWISE, ZTF, TESS, MAST inventory | `lhs1140` | `lhs1140.yml` | none | `results/lhs1140/summary.json`: `ANOMALY_FLAGGED` on one `companion` flag (RUWE 1.53, marginal binarity); `BIOSIGNATURE_NOT_DETECTABLE_WITH_CURRENT_DATA` |
| **JWST bio** (`jwst_bio`) | Real JWST/HST transmission spectrum of LHS 1140 b: disequilibrium pair, M-dwarf abiotic gate, MIRI eclipse, laser scan | MAST JWST `x1dints` (NIRISS, NIRSpec, MIRI) | `jwst-bio` | `jwst-bio.yml` | none | `results/jwst_bio/summary.json`: `no_biosignature_detected` (2 stacks analysed; MIRI eclipse not measured) |
| **Cross-correlation** (`crosscorr`) | Doppler-resolved O₂ / H₂O in LHS 1140 b's transit | ESO archive (ESPRESSO/HARPS/NIRPS), DACE | `crosscorr` | `crosscorr.yml` | none | `results/crosscorr/summary.json`: `NO_ARCHIVAL_IN_TRANSIT_HIRES_SPECTRA_AVAILABLE` |
| **SETI archive** (`seti_archive`) | Has any targeted radio/optical SETI campaign pointed at LHS 1140, and to what EIRP? | Breakthrough Listen open data, CADC ObsCore | `seti-archive` | `seti-archive.yml` | none | `results/seti_archive/summary.json`: `NO_TARGETED_RADIO_SETI_ON_RECORD` — an observational gap, with representative facility limits |

## Infrastructure (not channels)

| Module | Role | CLI / entry | Workflow | Doc | Outputs |
|---|---|---|---|---|---|
| `alerts.py` | Decides when a human must look: `candidate` / `health` / `milestone` alerts, deduplicated by stable key; frontier-stall and staleness checks read from timestamps *inside* result files | `alert-check` | `alerts.yml` (daily cron; opens issues assigned to the owner) | `alerts.md` | `results/alerts/{latest,frontier,state}.json` |
| `cronwatch.py` | Did the scheduler fire each cron? Reads crons from the workflow files, asks the Actions API, re-dispatches dropped runs | `cron-watch`; `scripts/cron_watch.py` | `cronwatch.yml` (6-hourly cron), `watchdog.yml` | `cronwatch.md` | `results/cronwatch/{status,state}.json` |
| `failsweep.py` | Re-runs failures from the last 24 h, never one whose `head_sha` is no longer its branch head | `scripts/fail_sweep.py` (no CLI subcommand) | `watchdog.yml` | `cronwatch.md` | `results/watchdog/{status,skiplist}.json` |
| `scripts/rubin_outage_check.py` | Asks a second broker whether Rubin's stream or ALeRCE's mirror stopped | script | `rubin-outage.yml` | `rubin-outage.md` | `results/rubin_outage/` |
| `indicators/` | Shared library: multi-axis anomaly scoring consumed by the WD IR-excess run | none | none | none | folded into `results/science/summary.json` |
| `io.py`, `config.py`, `photometry.py`, `population.py`, `pipeline.py`, `report.py`, `figures.py` | Shared helpers: cached VO access, config loading, photometric conversions, the WD-channel pipeline and manuscript figures | — | — | — | — |

## Literature sweeps

Prior-art and full-text sweeps run on the runner via `scripts/*lit*.py` and
commit evidence under `results/<name>lit*/`. They belong to the channel they
were run for:

| Results dir(s) | Channel | Workflow |
|---|---|---|
| `alertlit`, `alertlit2`, `alertlit3` | TOCSIN | `alertlit*.yml` |
| `vnprobelit` … `vnprobelit5` | LOOM | `vnprobelit.yml` |
| `derelictlit` | DERELICT | `derelict.yml` |
| `sextantlit` | SEXTANT | `sextantlit.yml` |
| `emberlit` | EMBER | `ember.yml` |
| `rustlit` | RUST | `rust.yml` |
| `vigillit` | VIGIL | `vigil.yml` |
| `tailingslit` (see its `INTEGRITY.md`: a successful fetch is no evidence the paper is the right one) | TAILINGS | `tailingslit.yml` |
| `chemlit`, `chemlit2`, `chemlit_nist`, `przybylski_lit` | MIDDEN (chemical technosignatures, NIST line data, Przybylski's star) | `chemlit.yml`, `chemlit2.yml`, `chemlit-nist.yml`, `przybylski-lit.yml` |
| `litcheck` | HERDSMAN novelty check (`scripts/litcheck_fetch.py`); `compass.yml` also writes its COMPASS novelty check here (`scripts/compasslit_fetch.py`) | `litcheck.yml`, `compass.yml` |
| `dysonlit`, `litcheck_dyson`, `hephlit`, `seamlit` | Dysonian waste-heat lineage: primary sources, Project Hephaistos dossier, and the structural "seams" an existing search could be evaded through (WD IR excess, CENOTAPH, OSSUARY, VIGIL) | `dysonlit.yml`, `litcheck-dyson.yml`, `hephlit.yml`, `seamlit.yml` |
| `zacklit` | Zackrisson+2018 "SETI with Gaia" prior art for the underluminosity / grey-dimming axis (CENOTAPH leg 1) | `zacklit.yml` |
| `offlit` | Sources that disappear or turn off: prior art for EMBER | `offlit.yml` |
| `disaplit`, `disaplit2`, `disaplit3`, `vanishlit`, `vanishlit2`, `vanishlit3` | Disappearance-class searches, VASCO and vanishing stars: prior art for SHROUD | `disap-lit.yml`, `disap-lit2.yml`, `disap-lit3.yml`, `vanishing-lit.yml`, `vanishing-lit2.yml`, `vanishing-lit3.yml` |
| `necrolit` | Novelty check for the necrosignature taxonomy in `necrosignatures.md` (S-numbered signatures cited by RUST, KNELL, VIGIL, EMBER, SHROUD, CENOTAPH, TOCSIN) | `necrolit.yml` |
| `farir_docs`, `farir_stats`, `catrecon` | AKARI-FIS / IRAS far-IR instrument documentation, measured crossmatch systematics and catalogue metadata (CENOTAPH leg 3, EMBER) | `farir-params.yml`, `catrecon.yml` |
| `survey_recon` | Live probe of what can feed TOCSIN and LOOM while Rubin is dark (`substitute-surveys.md`) | `survey-recon.yml` |
| `metronomelit`, `lanternlit`, `falloutlit` | Prior-art sweeps for METRONOME, LANTERN and FALLOUT (`scripts/<name>_fetch.py`; run as the `lit` job of each channel's workflow) | `metronome.yml`, `lantern.yml`, `fallout.yml` |

## Gaps this index exposed

Recorded so the next session does not rediscover them:

- **Absent from `STATUS.md` entirely:** MIDDEN, SEXTANT, HERDSMAN, HERDSMAN-B,
  ISOTHERM, COMPASS and the cronwatch/failsweep infrastructure. Their verdicts
  live only in their `results/*/REPORT.md` or `summary.json` (quoted above).
- **Stale `STATUS.md` row, now corrected:** the channel-state table said
  CENOTAPH was "pending first archive run" while
  `results/cenotaph/summary.json` already carried `closure_but_crowded_beam`
  on a partial sample. The row was updated on 2026-09-01; the general lesson
  is that a committed result and its `STATUS.md` row drift unless the run's
  commit-back step is the thing that edits the row.
- **Built but never run to a result:** OSSUARY and SHROUD have no committed
  `summary.json`; SEXTANT has run only its probe; VIGIL reached no data.
- **Wired but unreachable from any workflow:** `cluster-aggregate`;
  `failsweep` has no CLI subcommand (script-only).
- **No `docs/` page:** `accel`, `cluster`, `crosscorr`, `dimming`, `galactic`,
  `iso`, `jwst_bio`, `lhs1140`, `lhs1140_origin`, `panspermia`, `seti_archive`,
  `spectra`, `xp`. Their design is documented in `STATUS.md` sections and
  module docstrings.
- **Verdict lives outside `summary.json`:** WD IR excess, accel, cluster,
  spectra, midden, herdsman, herdsman_b, compass, galactic, iso, panspermia,
  lhs1140_origin. `alerts.py` only reads the Rubin channels, so none of these
  is covered by the staleness heartbeat.
