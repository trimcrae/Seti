# ROMAN — intake for the Nancy Grace Roman Space Telescope, and the channels only Roman can open

*Built 2026-09-09, pre-launch. Everything under "assumed" below is re-read from
the archive by the probe and inventory stages on the runner the day products
exist; nothing here is a measurement of Roman data.*

---

## 0. Why prepare now

Roman is not on sky (launch readiness committed for no later than May 2027;
the observatory shipped to the launch site in 2026). Its data carry **no
proprietary period**: every Wide Field Instrument exposure becomes public at
IRSA as it is calibrated. The three Core Community Surveys produce, together,
the largest time-domain, spectroscopic and imaging datasets ever taken from
space:

| Survey | What it delivers (assumed; `config/roman.yaml` → `surveys`, `verify: true`) | What that opens |
|---|---|---|
| **GBTDS** Galactic Bulge Time Domain Survey | ~2 deg² in ~7 bulge fields, **F146 every 12 min**, F087 every ~12 h, six ~72-day seasons over five years, ~2×10⁸ stars, ~1 % per-epoch precision at H ≈ 21 | a photometric time series of a scale, cadence and stability that no ground survey approaches, in the near-infrared, from L2 with no day/night or weather gap |
| **HLTDS** High Latitude Time Domain Survey | ~19 deg², **5-day cadence** in 4–6 filters for two years, **prism spectroscopy** of the field on every visit | the first spectroscopic time domain over square degrees |
| **HLWAS** High Latitude Wide Area Survey | ~2 000 deg² in four NIR filters to 26.5 AB at 0.11″ px, plus the **G150 grism (1.00–1.93 µm)** slitless spectrum of every source | the first wide-field near-infrared slitless spectroscopic survey of stars |
| **CGI** Coronagraph Instrument | technology demonstration: ~10 nearby stars at 10⁻⁸ contrast, 575–825 nm, imaging, R ≈ 50 spectroscopy, polarimetry | the first reflected-light images of the 1–5 AU zone around nearby stars |

Every existing light-curve and spectrum channel in this repository should run on
those products the afternoon they appear (§3, "the paces"), and four signatures
that **no earlier facility could reach** should be waiting, built and tested
(§2). This document is the design; `seti.roman` is the implementation;
`results/roman/` is where the probe records what the archive actually serves.

---

## 1. Intake

### 1.1 What Roman publishes (assumed, verified by `probe`)

* **Level 1** `*_uncal.asdf`: up-the-ramp cubes, `(n_resultants, 4096, 4096)`
  per detector, with the multi-accumulation read pattern in the header. A
  resultant is the average of a group of non-destructive frames (~3.04 s per
  frame). **This is the product that carries time inside an exposure.**
* **Level 2** `*_cal.asdf`: one calibrated exposure per detector — `data`
  (slope image), `err`, `var_*`, and **`dq`**, a uint32 bit mask in which the
  ramp-fitting/jump step marks every pixel whose ramp had a jump
  (`JUMP_DET`), plus `SATURATED`, `PERSISTENCE`, `DEAD`, `HOT`, … ASDF blocks
  are indexed, so a client can read `dq` alone by HTTP range request without
  fetching the ~200 MB file.
* **Level 3** coadds / mosaics; **Level 4** catalogues, **light curves** (the
  GBTDS per-star time series) and **extracted spectra** (grism / prism 1-D
  spectra with contamination estimates). Table products in Parquet / FITS /
  ECSV / ASDF.
* **CGI** products: PSF-subtracted images, spectra, polarimetric Stokes maps.

The exact layouts are not fixed before launch and will change between pipeline
builds, so `seti.roman.products` is the *only* module allowed to know what a
Roman file looks like. It emits the four structures in `seti.roman.schema`
(`LightCurve`, `Spectrum`, `DQCutout`, `Ramp`) and every detector consumes only
those. A product that does not carry a field a discriminator needs yields
`None`, and the funnel records that test as *not run* rather than as passed
(the TOCSIN lesson).

### 1.2 Discovery (`probe`, `inventory`)

The probe asks, in order and without assuming any answer:

1. IRSA TAP `TAP_SCHEMA.tables` for any table whose name/description carries
   `roman`, `wfi`, `gbtds`, `hltds`, `hlwas`, `openuniverse`;
2. IRSA SIA2 and the `data/Roman/` root;
3. the public S3 buckets — `nasa-irsa-simulations/openuniverse2024/roman/`
   (OpenUniverse 2024: simulated HLTDS-like and HLWAS-like images with truth
   catalogues) and the *expected* flight bucket, whose existence is the test;
4. MAST CAOM for a `Roman` collection;
5. the IPAC simulation pages (the 2018 microlensing data challenge: ~10⁴
   simulated GBTDS light curves — the GBTDS-like stand-in);
6. whether `roman_datamodels` / `asdf` install on the runner.

It writes `results/roman/probe.json` with per-endpoint status, and a single
**`data_state`**:

| `data_state` | Meaning |
|---|---|
| `NO_ARCHIVE_REACHED` | nothing answered (network); the run stops with `NO_DATA_REACHED` |
| `NOT_YET_PUBLIC` | archives answer, no Roman products of any kind |
| `SIMULATIONS_ONLY` | only pre-launch simulations exist — the intake is exercised on them and every result is stamped `simulated_inputs: true` |
| `MISSION_DATA_PRESENT` | flight products exist — the milestone `alerts.py` turns into a GitHub issue |

**First probe, run 34302574303 on 2026-09-09 (10:16 PM ET), `results/roman/probe.json`:
`data_state = SIMULATIONS_ONLY`.** Twelve of twelve endpoints answered. IRSA
TAP and SIA2 are up and carry no Roman table; `irsa.ipac.caltech.edu/data/Roman/`
is 404; the expected flight bucket `nasa-irsa-roman` does not exist
(`NoSuchBucket`); MAST CAOM returned HTTP 400 to the `COUNT(*)` query (the
reason is captured from the second probe onward); the IPAC simulation pages
assumed in the first config were 404 and are now discovered by crawling the
site root instead. What *does* exist is the OpenUniverse 2024 bucket, and it
is richer than the design assumed (§1.4). `asdf`, `roman_datamodels`, `gwcs`,
`fsspec` and `s3fs` all install on the runner.

`inventory` enumerates what exists into `RomanProduct` rows (level, kind,
survey, size, simulated) and a shard plan. **A simulated product is never
reported as a sky result**: every summary carries `simulated_inputs`, and the
verdicts on simulations are `SIMULATION_PACES_OK` / `SIMULATION_PACES_FAILED`
(did the data flow through every stage), never a candidate tier.

### 1.3 Readers (`products`)

* `read_asdf_arrays(uri, keys=("dq",), lazy=True)` — ASDF via `fsspec`
  range reads; only the requested blocks travel. Degrades to a
  `ReaderUnavailable` record when `asdf` is not importable.
* `read_ramp(uri, box)` — the Level 1 cube for a pixel box, with resultant
  mid-times from the read pattern and frame time.
* `read_lightcurve_table(path, colmap)`, `read_spectrum_table(path, colmap)`
  — Parquet / CSV / FITS / ECSV / ASDF tables, column roles resolved at run
  time from a role map (`time`, `flux`, `flux_err`, `band`, `dq`, …) and the
  unit/zero point taken from the table metadata when present, else from
  `config/roman.yaml`, with the source of the zero point recorded.
* `dq_flags()` — `roman_datamodels.dqflags.pixel` when importable, else the
  config table; the summary says which.

### 1.4 The pre-launch stand-in: OpenUniverse 2024 (`seti.roman.openuniverse`)

`s3://nasa-irsa-simulations/openuniverse2024/roman/` (anonymous, reachable
from this sandbox as well as the runner) holds, under `full/` and `preview/`:

| Product | Layout (verified from the files on 2026-09-09) | What it exercises |
|---|---|---|
| **SNANA light curves** `ROMAN+LSST_LARGE_SNIa-normal/…_HEAD.FITS.gz` + `…_PHOT.FITS.gz` | 7,471 SNe per file pair; `PHOT` carries only `MJD`, `BAND`, `SIM_MAGOBS` — noiseless model magnitudes in 14 bands (LSST `ugrizy` + Roman `R Z Y J W H F K` = F062 F087 F106 F129 F146 F158 F184 F213), ~295 epochs per band over MJD 61444–63269 (the HLTDS-like 1500-day, 5-day-cadence SIMLIB) | the light-curve reader and every paces channel, at HLTDS cadence in Roman bands; errors are **assumed** from a depth model and every curve says so (`errors_assumed`) |
| **TDS / WAS images** `RomanTDS/images/simple_model/<BAND>/<pointing>/Roman_TDS_simple_model_<BAND>_<pointing>_<SCA>.fits.gz` | galsim `roman_imsim` output: `PRIMARY` header (EXPTIME 161/302/901 s, MJD-OBS, FILTER, ZPTMAG, SIP WCS), HDUs `SCI` (float64 4088²), `ERR`, `DQ` (uint32) | the `DQCutout` path of S42 end to end; whether the sim `DQ` plane carries any jump flags is recorded as `dq_flag_census` in `calibration.json`, not assumed |
| **Truth indices** `RomanTDS/truth/<BAND>/<pointing>/Roman_TDS_index_…txt` | per-image table: `object_id ra dec x y realized_flux flux mag obj_type` (star / galaxy / transient) | catalogued-star positions per image without a cross-match; the flight-data path (catalogue + WCS) is the fallback |
| **Pointing sequence** `Roman_TDS_obseq_11_6_23.fits` | 57,365 exposures, seven filters × 8,195, MJD 62000–63563 | the survey-cadence record the config's `verify` rows are checked against |
| **Point-source catalogue** `roman_rubin_cats_v1.1.2_faint/pointsource_<healpix>.parquet` | 38,970 stars per pixel with `magnorm`, SED, proper motion, parallax, `variability_model` | the star list for the WCS-based path |

No grism/prism spectra and no CGI products exist in the simulation, so S41 and
S43 remain exercised only by their synthetic selftests until flight data.
Everything read from this bucket is stamped `simulated_inputs: true` and can
only produce `SIMULATION_PACES_OK` / `SIMULATION_PACES_FAILED`.

---

## 2. The four Roman-only channels

### 2.1 S40 — the opaque lens (`seti.roman.lens`)

**Claim.** A microlensing event whose lens is *opaque over a non-negligible
fraction of its Einstein radius*. For a point lens the two images sit at
θ± = (u ± √(u²+4))/2 in units of θ_E, with magnifications
A± = (u²+2)/(2u√(u²+4)) ± ½. An opaque disc of radius ρ_L (θ_E units) centred
on the lens removes an image whenever |θ| < ρ_L (Agol 2002 is the reference
for lenses that also occult). Because |θ₋| = (√(u²+4) − u)/2 *falls* from 1 at
u = 0 toward 1/u in the wings, an occulter with ρ_L < 1 eats the **minor image
in the wings** — for u > u_c = 1/ρ_L − ρ_L — and leaves the peak Paczyński. The
signature is therefore **a symmetric pair of downward steps at ±u_c, each of
depth exactly A₋(u_c)**, on both sides of an otherwise standard event:

| ρ_L | u_c | step depth A₋(u_c) |
|---|---|---|
| 0.3 | 3.03 | 0.8 % |
| 0.5 | 1.50 | 6.7 % |
| 0.7 | 0.73 | 31 % |
| 0.9 | 0.21 | 190 % |
| ≥ 1 | — | minor image always gone; major image gone for u < ρ_L − 1/ρ_L: a **central hole flanked by lensing wings** |

One parameter (ρ_L) fixes both *where* the steps are and *how deep*, a
two-for-one prediction no natural feature makes. Finite source size (ρ_*)
rounds the steps over Δu ≈ ρ_*; the model integrates over the source disc.

**Why no natural body.** With θ_E measured (finite-source ρ_* against the
source's angular radius θ_*, or a lower bound θ_E ≥ θ_* √(A_max² − 1)/2 from
the peak magnification alone), the occulter's physical radius is
R = ρ_L θ_E D_L and the lens mass M = θ_E² / (κ π_rel), κ = 8.144 mas M_⊙⁻¹,
π_rel = AU (1/D_L − 1/D_S). The implied mean density ρ̄ = 3M / (4π R³) is a
function of the unknown D_L; the channel reports it over the whole range and
the **distance window inside which a natural body could match**
(ρ̄ ≥ 0.01 g cm⁻³, a super-puff planet with margin — `lens.density_floor_g_cc`).
For any detectable event (ρ_* ≲ 1, hence θ_E ≳ θ_* ≈ 0.3–5 µas) and ρ_L ≥ 0.05
that window is confined to lenses within tens of parsecs, where the lens is a
planet-to-brown-dwarf mass object that Roman sees directly as a **blend that
moves ~1″ yr⁻¹** — a test the vet stage runs on the blend flux and its motion.
A lens that is opaque at ρ_L ≳ 0.5 with R ~ AU is, physically, a shell or
swarm around a dark central mass: the Dyson shell seen not by its waste heat
but by its **shadow across a background star's images**.

**Confounders and vetoes.** Finite-source flattening (rounds the peak, never
steps at u > 1); blending (scales, no steps); binary-lens caustic anomalies
(asymmetric, spikes/U-troughs, not a matched symmetric pair with A₋(u_c)
depth — `kink_asymmetry_max`); parallax and xallarap (slow asymmetries, no
steps); source variability (present outside the event); systematics in the
12-minute series (the F087 colour series must show the *same fractional*
deficit — an opaque body is achromatic; `colour_deficit_sigma_max`). The
central-hole regime (ρ_L ≥ 1) is separately guarded against a *transiting
planet on the source* coincident with a lensing event: the hole must be
centred on t₀ and its width must match ρ_L − 1/ρ_L from the wings.

**Data.** GBTDS Level 4 light curves (F146, F087). Pre-launch: the microlensing
data-challenge light curves (simulated) exercise the full path.

### 2.2 S41 — the industrial line (`seti.roman.lines`)

**Claim.** An **unresolved emission line on a stellar point source** in the
G150 grism (1.00–1.93 µm, R ≈ 461 λ/µm) or the P127 prism (0.75–1.80 µm,
R ≈ 80–180). Every published optical laser search (Tellis & Marcy, Breakthrough
Listen, our own SDSS/DESI `spectra` channel) stops near 0.98 µm. The band
Roman opens is where **our own high-power lasers live**: Nd:YAG 1.064 µm,
Yb-fibre 1.03–1.09, Er-fibre 1.53–1.57 (the eye-safe telecom band chosen for
free-space optical links precisely because it is atmospherically clean and
inexpensive at megawatt scale). A civilisation solving the same engineering
problem with the same materials physics lands in the same band. Matches to
the list in `lines.industrial_lines_um` are **flags, not filters**: the blind
search runs over the whole band, and the flag is reported for the reader.

**Method.** The existing LSF-matched filter (`seti.spectra.detect`) with the
LSF set from R(λ); the feature must be interior (≥ 8 samples from either end),
0.6–2 resolution elements wide, on a source with **stellar continuum** (S/N ≥ 5
in the continuum) that is a **point source** in the direct image. A 2-D mode
searches the dispersed image for a compact, PSF-like blob at the position the
dispersion solution predicts for λ — how an unresolved line on a faint
continuum actually looks in slitless data.

**Confounders and vetoes** (the slitless-spectroscopy ledger):
zeroth-order images of neighbours (predicted from the dispersion geometry,
`zeroth_order_offset_px`); overlapping traces (`trace_cross_dispersion_px`,
and the pipeline's `contam` estimate when it exists — its absence is recorded
as `overlap_test_not_run`); persistence from a bright source on the same
detector pixel in the previous exposure; **the same wavelength on ≥ 3 unrelated
sources** (instrumental) and **the same detector pixel on ≥ 2** (defect);
stellar and circumstellar lines (Pa β/γ/δ, He I 1.083, Fe II, [Fe II], H₂ —
`stellar_lines_um`); and the classic single-line emitter — a high-redshift
Hα/[O III]/Lyα galaxy — excluded by requiring a point source with a stellar
continuum, and flagged `possible_high_z_emitter` otherwise.

**Data.** HLWAS grism Level 4 spectra (∼10⁷–10⁸ sources); HLTDS prism spectra
(every visit, so a line can also be tested for *persistence in time*). Euclid
NISP (1.2–1.85 µm, red grism; DR1 due late 2026) is the substitute survey for
this channel and the reader accepts its 1-D tables through the same role map.

### 2.3 S42 — the sub-exposure flash (`seti.roman.flash`)

**Claim.** A pulse of light shorter than one resultant (~3–10 s) on a
catalogued star. In an up-the-ramp detector a pulse deposits charge that
*stays*, so it appears exactly as a cosmic ray does — a step in the ramp — and
the pipeline's jump detector flags it `JUMP_DET`. The difference is
**morphology and coincidence**: a cosmic ray hits one pixel or a short track,
anywhere; a flash lights the **whole PSF footprint of a star at the same
resultant, with a step amplitude map that is the PSF**. Roman is the first
wide-field survey with non-destructive reads, so this is the first time the
classic pulsed-beacon question of optical SETI can be asked of ~10⁸ stars
simultaneously, in the near-infrared, over ~4×10⁴ exposures.

**Two stages.**
1. *Level 2, `dq` only.* Connected components of `JUMP_DET` (3–40 px); reject
   elongated clusters (tracks), large round clusters with a saturated core
   (snowballs), clusters not centred on a catalogued star (within 0.75 FWHM),
   clusters that flag less than 60 % of the star's PSF footprint, stars with a
   `SATURATED` pixel within 3 px, and any pixel already in the **hot-pixel
   ledger** (flagged in ≥ 3 exposures). Survivors: `PSF_JUMP_ON_STAR_PENDING_RAMP`.
2. *Level 1, candidates only.* Fetch the ramp for the pixel box. Fit each
   pixel with slope-before / step / slope-after at every resultant; require
   the step at **one** resultant in ≥ 80 % of PSF pixels, `slope_after =
   slope_before` within 3σ (a **flare keeps rising** — its ramp is a slope
   change, not a step, and it persists into the next exposure), the amplitude
   map correlated with the PSF at ≥ 0.8, and no residual brightening in the
   next exposure of the same star.

**Sensitivity.** The peak pixel must jump by n_σ × read noise
(5 × 12 e⁻ = 60 e⁻); with ~20 % of a point source's flux in the peak pixel that
is ~300 e⁻ at the aperture, i.e. **the star's own photon budget over
t_eq = 300 e⁻ / rate** (0.6 s for H_AB = 21 in F146). At 8 kpc that is an
isotropic-equivalent ~10²⁵ J per pulse, or ~10⁸–10¹⁰ J transmitted through
a 100–10 m aperture at 1 µm. Bright by our standards; the population is the
point, and every non-detection is a fluence limit per star per exposure that
the assess stage tabulates as an honesty check, never as the deliverable.

**Data.** GBTDS Level 2 `dq` (the primary), HLTDS/HLWAS Level 2 `dq`; Level 1
ramps for candidates. Pre-launch: synthetic `dq` images built from the PSF
model in the selftest, and the OpenUniverse images once their `dq` layers are
confirmed by the probe.

### 2.4 S43 — the statite (`seti.roman.statite`)

**Claim.** A reflecting point source in a CGI field whose **position does not
obey Kepler**. A planet at projected separation s around a star of mass M
moves on an arc set by P = √(a³/M); a *statite* — a reflector held against
gravity by radiation pressure (Forward 1993), or any station-kept structure —
sits at a fixed position relative to the star. At CGI astrometry (~3 mas per
epoch) and separations 0.15–0.45″, a Keplerian companion at 10 pc moves tens
of milliarcseconds per year; a fixed one does not. Two supporting tests: the
reflectance is **grey** (no methane band depressing 730 nm against 575 nm) and
the **phase function** is specular (narrow) rather than Lambertian.

**Confounders.** Background stars (fit relative proper motion + parallax:
`fixed_vs_linear`), disc clumps and speckle residuals (polarimetric and
epoch-to-epoch consistency), a planet on a nearly face-on wide orbit whose arc
over the baseline is below the astrometric error (`min_span_fraction_of_period`
— the verdict is `UNRESOLVED` until the arc is testable, never a candidate).

**Data.** CGI Level 2–4 (a handful of targets, released after the technology
demonstration). The population is tiny and the channel says so; it exists so
that the images are screened for something no planet-hunter looks for.

---

## 3. The paces: every existing channel on Roman products (`seti.roman.bridge`)

| Channel | Detector reused | Roman adaptation (`paces:` in `config/roman.yaml`) |
|---|---|---|
| Deep dips (Boyajian analogue, S10) | `dimming.dips.detect_dips` | depth ≥ 3 %, k = 4σ, ≥ 200 epochs; F146 + F087 achromaticity via `tocsin.photometry.greyness_z` for dips longer than the colour cadence |
| Specular glint (S30) | `dimming.glint.detect_glints` | brief brightening ≥ 20 % at 6σ; merge gap 0.05 d; the same colour test |
| Secular fade (enshrouding) | `dimming.secular.detect_secular_fade` | GBTDS seasons (72 d) are the season unit; needs the survey zero point to compare seasons |
| RUST (S9) | `rust.trend.detect_rust` | one 72-day season per block; ≥ 100 epochs per season |
| KNELL (S32) | `knell.cease.analyze_band` | `block_mode="gap"`, periods 0.02–30 d |
| METRONOME (S28) | `metronome.clock.analyze_star` | flares detected in the light curve itself (k = 5σ, ≥ 10 %), windows from the survey's own sampling |
| Narrow emission/absorption lines | `spectra.detect.find_emission_lines`, `spectra.absorb.find_absorption_lines` | LSF from R(λ); vetoes from §2.2 |

Season-based channels use the GBTDS season structure found from gaps
(`season_gap_days`), never 365.25 d. Channels that need a magnitude system
run only when the product carries a zero point; otherwise they run in relative
flux and the record says `relative_flux: true`.

---

## 4. Novelty — the record, read

`romanlit` ran on 2026-09-09 (run 34302576338, 164 fetches, 115 OK, 645
abstracts scanned, `results/romanlit/concept_scan.json`); abstracts quoted
below are verbatim from the fetched records, ids title-checked.

* **S40.** The *phenomenology* of a lens that occults its own images is
  published as natural-body physics, and recently: Agol 2002 (occultation and
  microlensing, title-verified); an astrometric treatment of "a spherical
  finite-size lens consisting of opaque material" whose image trajectories
  "can be classified into three types according to occultation of the plus
  and minus images" (2003, doi:10.1086/377151); and, one month before this
  channel, **arXiv:2608.24009 (Aug 2026)**, "Occultation of microlensed images
  by circum-lens disks: Implications for surveys" — "microlensing surveys
  typically apply automated criteria to exclude lightcurves that depart from
  time-symmetric and achromatic Paczyński curves … occultation of microlensed
  images by circum-lens material could lead to underdetection or
  misclassification of ring- or disk-bearing compact object populations".
  That paper is the natural-body neighbour and, read carefully, a warning:
  **the GBTDS pipeline's own anomaly cuts may reject exactly the S40 shape**,
  so the channel must run on the full light-curve product, not on the
  pipeline's event catalogue. Nothing in the record turns the occulting
  radius into a density and asks whether any bound body can supply it, and
  nothing searches a survey for the symmetric-step signature; those two
  steps are the claim.
* **S41.** The near-infrared pulsed/continuous laser lineage is targeted and
  ground-based: PANOSETI's "optical and near-infrared (350–1650 nm)
  instrument" for "transient pulsed signals occurring between nanosecond to
  second time scales" (arXiv:1808.05772), NIROSETI, and Hippke's optimum
  "λ₀ ≈ 1 µm, Δλ ≈ 1.5 nm" pulses (arXiv:1804.01251). Vides et al. 2019
  (arXiv:1909.04128) model **WFIRST's coronagraph** as a laser detector at
  575 nm around nearby stars. No wide-field NIR slitless survey search
  appears in the record; the RoSETZ white paper (arXiv:2306.10202) is the one
  Roman SETI proposal found and it is a transit survey of the Earth Transit
  Zone. The S41 position stands as written.
* **S42.** Pulsed optical SETI is the Harvard all-sky lineage ("coincident
  optical pulses of nanosecond timescale", 2000), PANOSETI and the 2026
  "Cosmic Lighthouses" argument that "our extensive, existing surveys are
  mostly blind to the µs/ms-second universe". The engineering literature on
  up-the-ramp jump detection and snowballs is the calibration baseline. No
  record uses jump flags as an astrophysical flash detector; the S42 position
  stands.
* **S43.** The neighbours are Jaiswal 2023 (arXiv:2306.07859), specular
  glints from artificial *surfaces on a rotating planet* in direct imaging
  ("the reflected signal is very strong … for surfaces covering only few
  ppm"), and the statite concept in our own mission design
  (arXiv:2012.12935). Non-Keplerian astrometry of a direct-imaging point
  source as the test is not in the record. Vides et al. 2019 is the prior
  Roman-coronagraph SETI proposal (laser, not reflector) and is cited.

## 4A. What the simulations taught (run 34349717932, 2026-09-09, 8:12 AM ET)

The first `stage=full` run that reached the screen stage (`results/roman/`,
`SIMULATION_PACES_OK`, 5,378 simulated objects: 2,688 SNANA light curves in
the eight Roman bands from two HEAD/PHOT pairs, two TDS/WAS images, two
pointing sequences) exposed two detector behaviours that flight data would
have exposed later and worse:

* **A supernova out-fits Paczyński with the *occulting* model.** 1,262 of the
  2,688 SN Ia curves came out `OCCULTING_PREFERRED_PENDING_VET` and none
  reached the candidate tier (every density and colour test was pending).
  The reason is not the occultation physics: a fast-rise / slow-decline
  transient is asymmetric about its peak, a Paczyński curve is symmetric, and
  the occulting model has the extra freedom to absorb the asymmetry in the
  wings. Every lens of any kind is time-symmetric about the closest approach,
  so the screen now folds each curve about the fitted t₀ and rejects a
  reduced χ² above 3 as `time_asymmetric` before any occultation gate is
  believed (`lens.time_asymmetry_chi2_red_max`). Supernovae, novae, dwarf
  novae and flares in the GBTDS fields all fall to this gate; a real
  occulting event passes it (tested).
* **At a 5-day cadence every transient is "a glint".** 1,595 of the 1,621
  curves the glint pace ran on were flagged, and 1,371 by dips: a supernova
  is a brightening confined to a few epochs on an otherwise flat baseline,
  which is the glint definition. A specular glint lasts minutes to hours, so
  the pace now reports `not_applicable` when the series' median cadence
  exceeds `paces.glint.max_event_duration_d` (1 day), and every light curve
  carries a `transient_like` tag (one dominant, interior, time-asymmetric
  brightening) whose flags `assess` counts apart from the flags on stars.
  The secular, RUST and KNELL paces were `insufficient` on every curve
  (~300 epochs per band, no 100-epoch seasons at this cadence), which is the
  honest answer for HLTDS-like sampling.

Also measured: the simulated `DQ` planes carry no flag of any kind
(33.4 M pixels, `dq_flag_census_sim`, read through `roman_datamodels.dqflags`
on the runner), so S42's `JUMP_DET` path is exercised only by the synthetic
selftest until flight Level 2 products exist; the implied F087 zero point of
the galsim images is 26.40 against the config's 26.5; MAST's ObsCore now
answers cleanly with no Roman collection (the earlier 400 was a column name);
and the IPAC crawl reached the **Roman Microlensing Data Challenge 2026
(RGES PIT)** page, the GBTDS-like stand-in, which the next probe records in
full. Nothing here is a sky result: `simulated_inputs: true` on every record.

---

## 5. Operations

* `python -m seti.roman.run probe` — endpoints, packages, `data_state`.
* `… inventory` — products and shard plan (`results/roman/inventory.json`).
* `… ingest --kind {lightcurve,dq,spectrum,cgi} --limit N` — pull a bounded
  set of products into the four structures (cached under `data/cache/roman/`).
* `… screen --channel {lens,lines,flash,statite,paces} --shard i --n-shards n`
  — checkpoint one JSON per product immediately.
* `… assess` — recurrence vetoes across shards, tiers, `summary.json` per channel
  and the top-level `results/roman/summary.json`.
* `… selftest` — the synthetic injection/rejection battery through the same
  analysis paths (offline gate; runs on the runner too).
* `… readiness` — `probe` + inventory diff + the milestone record
  (`results/roman/readiness.json`) that `alerts.py` reads. Workflow
  `roman.yml` runs it on a monthly cron until `MISSION_DATA_PRESENT`, then
  weekly; `roman-lit.yml` is the prior-art sweep.

Verdicts: `NO_DATA_REACHED`, `ROMAN_NOT_YET_PUBLIC`, `SIMULATION_PACES_OK`,
`SIMULATION_PACES_FAILED`, `NO_CANDIDATE`, `<CHANNEL>_CANDIDATES_PENDING_VET`,
`DEGRADED_SOURCE`. A verdict never reads as a sky null when nothing real was
tested, and never reads as a candidate on simulated inputs.

---

## 6. What is assumed and how it gets verified

Every row in `config/roman.yaml` marked `verify: true` (filter edges, zero
points, PSF core fraction, read noise, MA tables, survey cadences, the S3
bucket names, the zeroth-order geometry) is a pre-launch documentation value.
The probe records what the archive says; the inventory records the delivered
headers; the first `screen` on flight data re-derives the PSF core fraction,
the noise floor and the jump-flag statistics from the data and writes them next
to the assumed values in `results/roman/calibration.json`. Until that file
exists, every sensitivity number in this document is a design number.
