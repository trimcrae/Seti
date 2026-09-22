# SPECTRA-PERSIST — does the narrow line survive its own exposures?

`src/seti/spectra/persist.py` · `.github/workflows/spectra-persist.yml` ·
`results/spectra_persist/`

## The question

A survey coadd is a sum of individual exposures. A real monochromatic source is in
**every** one of them at the same strength. A cosmic ray, a read-out artefact or a
one-off sky-subtraction failure is in one and is merely diluted into the coadd. A
sky-line residual tracks the sky-line brightness from exposure to exposure.

The 167 narrow-line survivors in `results/spectra_triage/summary.json` (112 SDSS-DR17
emission + 55 DESI-DR1 absorption, after the duplicate, known-line, recurrent-wavelength
and galaxy cuts) have only ever been measured in coadds. This channel measures each of
them in every exposure that built the coadd, **independently**.

## The data

| route | file | what it gives |
|---|---|---|
| `sdss_full_spec` | `spec-PLATE-MJD-FIBER.fits` (the **full** version) on the SAS | HDU 1 = coadd, HDUs 4+ = the spCFrame exposures, one per camera per exposure, with `flux, loglam, ivar, mask, wdisp, sky` |
| `desi_cframe` | the healpix `coadd` file's `EXP_FIBERMAP` → `cframe-{band}{petal}-{expid}.fits` and `sky-…` | one row per exposure per arm, read by HTTP range where the server allows it |
| `desi_spectra_file` | the healpix `spectra-*` file | fallback when the cframes are unreachable |

Not every SDSS reduction carries exposures: the legacy `run2d=26` full files are
**coadd-only** (verified in run 35740662260 — `EXTNAME`s are `COADD, SPECOBJ, SPZLINE`
and nothing else). Those spectra are `untestable` by this route, not "absent".

## The measurement

Per exposure, independently: a sigma-clipped **linear** continuum fit across an annulus
2–30 Å either side, the integrated excess (deficit, in absorption mode) inside ±1.2 LSF
FWHM, and an error that is the pipeline inverse variance rescaled to the empirical
continuum scatter with the continuum's own uncertainty propagated in.

Three things had to be added before any of those numbers meant anything.

### 1. The continuum must be fitted, not medianed

The first run returned `absent_in_exposures` for every SDSS survivor it reached with
combined significances of −2.2 to −9.2. Genuine absence gives 0. A median annulus has no
slope term, and the annulus is sampled asymmetrically whenever part of it is masked or
off the end of the spectrum — the rule near the blue end of an SDSS exposure. Replaced
with a sigma-clipped linear fit (commit 2be2bc49).

### 2. The estimator's own null, measured in the same spectrum

That was not enough. With the linear fit in place the second run still returned
`absent_in_exposures` for 15 of the 17 SDSS survivors it reached, at −2.6, −3.5, −3.6,
−3.8, −3.9, −4.5, −4.8, −4.9, −5.1, −6.0, −7.1, −7.8, −8.6, −10.8 and −10.9 σ.

`offset_null()` repeats the entire measurement — exposures, coadd, combination — at 24
random offsets of 12–200 Å from the candidate **in the same spectrum**. It answers the
one question the summary numbers cannot: is the deficit *at* the candidate wavelength, or
everywhere? Its median is the bias this estimator has in this spectrum; its MAD is the
scatter it really has. Every per-exposure flux is corrected by `bias × err` and every
error widened by that scatter (floored at 1.0 — the nominal error already carries unit
variance, and a narrow null is not a licence to claim more significance than the photons
allow). `combined_sig` is therefore an excess over what this spectrum returns for
*nothing at all*; `combined_sig_raw` keeps the uncorrected number so the correction is
visible rather than hidden.

**The bias is subtracted, so it must not be treated as exact.** A null from a handful of
offsets can land several sigma from the truth, and subtracting such a number from a
deficit does not merely mis-state a significance — it *creates* a line. The calibration
exists to stop the estimator inventing absences; it must not be allowed to invent
presences. So below 12 per-exposure readings nothing is subtracted at all
(`null_calibrated` says so, next to `null_n_measurements`), and above it the bias's own
standard error, 1.2533 × MAD / √n, goes into the error bar:
`err = √((err·sd)² + (se_bias·err)²)`. A thin-but-usable null widens the bar instead of
sharpening a spurious signal.

### 3. The stack of the exposures, as a second reference

`stack_exposures()` builds the inverse-variance mean of the exposure HDUs on the coadd's
own grid and measures the line in it with the same estimator. The coadd is *supposed to
be* that stack, so the two disagreeing is itself the finding:

* stack reads like the coadd → the per-exposure *measurement* is what disagrees;
* stack reads like the exposures → the archive's coadd does not agree with its own
  inputs, which is the artefact this channel exists to catch.

A line each exposure is too noisy to show but whose own stack shows at ≥ 4 σ is
`stack_only` (**OPEN**), not `absent_in_exposures` (**KILLED**). Absence has to survive
the stack as well as the exposures.

## Classes

`persistent` · `persistent_2exp` · `transient` · `sky_residual` · `absent_in_exposures` ·
`stack_only` · `inconsistent_strength` · `partial` · `single_exposure_only` ·
`ambiguous` · `untestable`

`final_verdict()` maps those to `KILLED_…` / `OPEN_…` / `ALIVE_persistent_unidentified`,
with a rest-frame known-line match overriding everything and a second-epoch non-detection
killing only when that epoch was sensitive enough to have seen the line at ≥ 5 σ.

Two verdicts do not correspond to a class, because they are interpretations of one:

* `KILLED_shared_ccd_column` — another *fibre of the same plate* carries a candidate at
  the same wavelength. Different objects, same detector columns; both cannot be sources,
  and a sky or ISM feature there would already have been taken by the known-line cut.
* `KILLED_not_significant_in_coadd` — the class is `absent_in_exposures` but the
  **calibrated** coadd significance is below 5 σ. Then "the coadd feature is not in its
  inputs" is the wrong sentence: there was no coadd feature. Only a line the calibrated
  coadd *does* show, and the exposures and their own stack do not, is the coaddition
  artefact this channel exists to catch.

## The cheap discriminators, run on every survivor

* **rest-frame line identification** at the star's own catalogue redshift against 792
  labelled lines (`linelist.py`), star-frame and observed-frame separately. Survey
  wavelengths are vacuum, the literature values are air, and the conversion happens at
  definition time.
* **second epoch** — every other SPARCL spectrum within 2″, with the line measured the
  same way.
* **SIMBAD** object type and spectral type at 3″.
* **NIST ASD context** — the strongest catalogued atomic lines within a few Å. Context,
  not a kill: every optical wavelength is within a few Å of *some* weak NIST line.
* **atmospheric context** — which telluric absorption band the observed wavelength falls
  in, how far the nearest listed OH line is, and the local density of listed OH lines.
  The hand-kept sky list jumps from 6753.3 to 6863.9 Å and carries no telluric bands at
  all, so a "no OH line within 50 Å" has to be read next to "and a forest all around it".

## Is the line even unresolved?

A monochromatic source is by definition **unresolved**: its profile is the instrument's
LSF. A feature measurably broader than the LSF cannot be a single narrow line whatever
else it does — and this is the cheapest discriminator in the channel.

The triage recorded a `width_ratio` for every survivor and nothing downstream read it.
The six lines left standing after the first run carry 1.09, 1.14, 1.29, 1.34, 1.42 and
1.47; the whole 167 have a median of 1.44. Those numbers could not be used as they stood,
because they are against a **nominal** R = 2000 while SDSS's real resolution runs from
about 1500 to 2500 across the spectrum and between fibres — "40 % broader than the LSF"
against the wrong LSF is not an argument about anything.

`lsf_fwhm_measured()` therefore takes the instrumental FWHM from the pipeline's own
per-pixel LSF column (SPARCL `wave_sigma`, an SDSS spec file's `wdisp`) at the candidate's
wavelength, falling back to the nominal value only when the column is not served and
recording which was used. `fit_line_profile()` fits a Gaussian on a locally-fitted
continuum and returns the centre, its velocity offset, the FWHM **and its error**, and
the amplitude significance. Reported, never cut: a 1.4 ± 0.3 must not be read as a
1.4 ± 0.05.

## Every epoch, not just the best one

`second_epoch` records only the strongest other detection at the position: it answers
"was it seen again" and nothing else. `epoch_series()` measures the line in every SPARCL
spectrum within 2″ separately — significance, EW, continuum, the sky model at the
wavelength, the fitted FWHM against that spectrum's own LSF, the fitted velocity offset —
and reports the median EW, the fractional MAD spread across epochs, how many reach 4 σ,
and the range. A line of constant strength across years is a stable property of the star;
one that varies is a different object. The per-exposure test cannot separate them,
because both are in every exposure.

## A molecular band gap — an explanation that fitted, and then did not

Checked against each star's own redshift, **all six** lines left standing lie between two
molecular band heads:

| λ_obs (Å) | blue head | red head |
|---|---|---|
| 6809.3 | CaH 6750.7 (−58.5 Å) | CaH 6908.8 (+99.5 Å) |
| 8578.3 | VO 8519.7 (−58.6 Å) | VO 8622.7 (+44.4 Å) |
| 6856.5 | CaH 6751.8 (−104.7 Å) | CaH 6909.8 (+53.3 Å) |
| 7490.3 | CN 7438.4 (−51.9 Å) | TiO 7590.7 (+100.4 Å) |
| 6403.2 | CaH 6385.5 (−17.8 Å) | ZrO 6477.5 (+74.3 Å) |
| 6967.9 | CaH 6955.2 (−12.7 Å) | TiO 7059.4 (+91.6 Å) |

In a cool star the flux *between* two band heads is a relative **maximum**. A matched
filter run against a local linear continuum that is itself inside the band structure
reads that maximum as an unresolved emission line — and the feature is then in every
exposure of the star and in every epoch of it, so neither the per-exposure test nor a
second epoch can see it. Two of the six objects are SIMBAD M1V dwarfs.

`band_gap_context()` puts the head either side, with its label and distance, on the
record next to each candidate. It is a hypothesis with a prediction: **other stars of the
same spectral type should show the same feature at the same wavelength.** That is what
the same-type control sample measures.

**And the profiles already argue against it.** `results/spectra/top_candidate_spectra.json`
stores the 81-pixel coadd window around the top 40 triage candidates, so five of the six
can be fitted offline. Against the nominal LSF:

| λ_obs (Å) | triage width ratio | fitted FWHM / LSF | peak above continuum |
|---|---|---|---|
| 6809.3 | 1.42 | **0.91** | 19.8 vs ~7.7 |
| 6856.5 | 1.47 | 1.10 | 15.8 vs ~11.3 |
| 7490.3 | 1.09 | 1.13 | 13.8 vs ~10.3 |
| 6403.2 | 1.14 | 1.11 | 3.9 vs ~2.2 |
| 6967.9 | 1.29 | **0.52** | 12.3 vs ~9.2 |

These are **narrow features two to three pixels wide sitting on the local continuum**, not
the broad relative maxima a molecular band gap produces. The triage's 1.42 was a
matched-filter width, not a profile fit. So the band-gap flag stays on the record as
context, and the leading systematic for the strongest candidate is not the star.

A ratio of 0.52 (6967.9 Å) is *narrower than the instrument can make*, which is its own
verdict: a single-pixel defect or a cosmic ray, not a spectral feature.

## A leak in the triage's recurrence cut

The triage removed a wavelength as `recurrent_across_runs` when **three or more** spectra
had a candidate within 3 Å. Pairs came through — and across the 350 triaged candidates
there are **114 pairs at exactly the same wavelength**. On a survey's common log-λ grid
the same wavelength is the same **pixel**, and unrelated sightlines do not agree to three
decimal places by accident. That is the inherited ledger's "a wavelength recurring across
unrelated sightlines is instrumental", operating one spectrum below the cut.

Counted against every triaged candidate (the ones the triage already removed are evidence
about the wavelength too), **76 of the 167 survivors** have at least one other sightline
within 3 Å: 57 with one, 15 with two, 4 with three or more.

Of the six left standing, 6809.3, 8578.3, 6856.5 and 7490.3 have **none** within 3 Å;
6403.2 and 6967.9 each have one, 1.64 Å and 1.60 Å away — about one pixel. So the two
strongest survive the stricter cut and the two weakest do not cleanly.
`n_other_candidates_within_3A` and `nearest_other_candidate_dA` now travel with every
line. Reported, not enforced — but a survivor that shares a pixel with another sightline
should not be read as a detection.

### And it is a measurable excess, not an anecdote

A survey coadd lives on one common wavelength grid, so *the same wavelength* means *the
same pixel index* for every spectrum in the release. A detector or reduction feature that
makes narrow spikes makes them at a fixed pixel; a source does not care which pixel it
lands on.

`pixel_coincidence()` histograms the pixel separation of every pair of candidates from
**different sightlines** and calibrates itself against separations of 3–10 pixels, which
carry the same clustering of the search's sensitivity with wavelength and none of the
same-pixel effect. Pairs from two spectra of the same object within 2″ are excluded.

| release | candidates | 0 px | 1 px | baseline (3–10 px) | excess at 0 px |
|---|---|---|---|---|---|
| SDSS-DR17 (log grid 10⁻⁴ dex) | 166 | **21** | 18 | 9.75 | +11.2 |
| DESI-DR1 (linear grid 0.8 Å) | 95 | **11** | 4 | 4.62 | +6.4 |

A Poisson z on that excess would be 3.6 and 3.0, but it treats the pair counts as
independent draws and they are not — one candidate sits in many pairs. The null is
therefore measured: each **sightline's** own set of pixels is slid bodily by a random
offset far larger than the window counted, which destroys cross-sightline alignment while
keeping how many candidates each sightline has and roughly where they sit. Over 500 draws:

| release | observed excess | permutation null | z | p |
|---|---|---|---|---|
| SDSS-DR17 | +11.2 | −4.27 ± 2.35 | **6.6** | < 0.002 |
| DESI-DR1 | +6.4 | −0.85 ± 0.94 | **7.7** | < 0.002 |

(A cluster bootstrap is the obvious alternative and is wrong here: resampling sightlines
with replacement makes duplicate copies of one sightline, and two copies land on the same
pixel by construction — measured, that "excess" comes out at 93 against an observed 11.
The null mean is also *negative* rather than zero, so the 3–10 px baseline slightly
over-states what the 0-px bin should hold, which is what the Poisson assumption hid.)

So about 19 SDSS and 6 DESI candidate pairs sit on a shared pixel for an instrumental
reason — roughly one candidate in eight. **None of the six lines left standing is one of
them at 0 px**; two have a neighbour one pixel away. The statistic goes in the summary per
release.

The two releases fail differently, which is what one would hope. The DESI candidates are
spread over 72 healpix with up to 5 in one, and **not a single same-pixel pair shares a
healpix** — so the DESI excess is release-wide, not per-tile. Of its 11 exact
coincidences, six sit within 22 Å of a listed OH line (three within 3.5 Å) and one is
inside the H₂O 7200 band: for an *absorption* search, an over- or under-subtracted
airglow line is exactly a narrow deficit at a fixed observed wavelength. The SDSS excess
is the opposite — it concentrates on plates (below), which is a detector, not the sky.

## A background galaxy in the fibre — the leading explanation for the strongest line

`galaxy_reject` already tests for this, but it tests the **candidate list**: it needs two
surviving candidates in one spectrum to land on one redshift. A galaxy whose Hα clears
the 8 σ search threshold while its [N II] and [S II] do not therefore leaves exactly one
candidate and passes the cut. That is the common case, not the rare one — [N II] 6584 is
typically 0.3 × Hα in a star-forming galaxy, so a 16 σ Hα comes with a ~5 σ companion that
was never a candidate. **0412-51942-0465 at 6809.26 Å has `n_lines_in_spectrum` = 1 for
exactly this reason.**

`background_galaxy_scan()` asks the **spectrum** instead: each strong nebular line is
tried as the anchor and, at the redshift that implies, every *other* line of the family is
measured directly in the data however weak. A companion landing within 2 LSF of the
candidate is skipped — a close doublet partner re-measuring the same feature is not a
companion, and that alone made a lone injected line look like an [O II] doublet at
z = 0.83.

On the 125 Å windows the triage stored (all that is reachable offline), Hα is the best
anchor for three of the five and the companions are suggestive but below 3 σ:

| candidate | anchor | implied z | best companion |
|---|---|---|---|
| 6809.26 | Hα | 0.037268 | [N II] 6584 at 6830.69 Å, **2.0 σ**, EW 1.39 Å |
| 7490.31 | Hα | 0.141014 | [N II] 6584 at 7513.89 Å, 2.8 σ |
| 6856.46 | Hα | 0.044458 | [N II] 6584 at 6878.04 Å, 1.9 σ |

For 6809.26 the [N II]/Hα equivalent-width ratio is 1.39 / 6.10 = **0.23**, which is
textbook star-forming. **The decisive lines are outside the stored window**: at
z = 0.037268, Hβ falls at **5043.9 Å** and [O III] 5008 at **5194.9 Å**. The control stage
holds the whole spectrum and measures them. If either shows up at ≥ 4 σ, the candidate is
an unremarkable background galaxy and the persistence, the second epoch and the unresolved
width are all explained at once — a real astrophysical source in the fibre, just not the
star.

## The detector

If the feature is narrow, unresolved and in every exposure and every epoch, the next
thing it can be is the **detector**. A bad CCD column puts a narrow feature at one
wavelength in every fibre of an exposure set; it is in every exposure of that plate and in
every repeat observation of it, so neither the per-exposure test nor a second epoch can
see it. Two things test it:

* the **same-plate control** — the same wavelength measured in other fibres of the same
  plate (`control_sample` with an explicit `{"plate": N}` constraint);
* **which fibre each "epoch" is.** SDSS repeat spectra of one object are very often the
  *same plate and fibre on another night* — the same CCD column. Eight "independent
  epochs" that are all one fibre confirm a detector defect exactly as well as they
  confirm a source, and a best-of second-epoch number cannot say which.
  `epoch_series()` records plate / MJD / fibre per epoch and reports `n_distinct_fibres`.

## The control sample — the test the others cannot do

None of the above can reject a feature that the star's **spectral type** produces. A gap
between TiO band heads in an M dwarf is in every exposure of that star and in every epoch
of it; it is persistent, it is repeatable, and it is not a source.

`--stage control` measures the candidate's wavelength in unrelated SPARCL spectra (40 per
line by default), in two samples and two frames:

* **same-type vs any-star.** A feature the spectral type produces appears in the
  same-type sample and not in the all-stars sample; one the sky or the instrument
  produces appears in both, at the same observed wavelength.
* **observed frame vs stellar frame** (the candidate's rest wavelength placed at each
  control star's own redshift). Honest limit: at 6800 Å the line window is ±4 Å, so stars
  within ~180 km s⁻¹ of each other land in the same window and the two frames say the
  same thing. For Galactic stars that is the usual case, which is why the *type*
  dependence carries the argument.

## State

**Run 35738206630 (reduce committed 2026-09-22 11:12 EDT) — the first real measurement,
and it is incomplete.**

`results/spectra_persist/summary.json`, verdict `PERSISTENT_UNIDENTIFIED_LINES_REMAIN`:

| | |
|---|---|
| survivor lines in | 167 (141 spectra) |
| spectra with a usable checkpoint | 62 of 141 (36 of 98 checkpoints ignored as stale) |
| lines never measured | 96 (`not_run`), all DESI plus part of SDSS |
| `absent_in_exposures` | 53 |
| `persistent` | 13 |
| `ambiguous` / `partial` / `untestable` | 2 / 1 / 2 |
| `KILLED_known_line_rest_frame` | 106 |
| `ALIVE_persistent_unidentified` | 6 |

Two caveats, both structural, both now fixed in code but **not** in those numbers:

1. **The run split into two estimators.** Its jobs queued for up to 45 minutes and
   `actions/checkout` took the *branch head* at each job's start, so early shards measured
   with the median continuum and late ones with the linear fit. The `ckpt_version` guard
   caught it (36 checkpoints ignored) so nothing stale entered the table — but 96 lines
   went unmeasured as a result. Shards now check out `github.sha`, and every checkpoint
   records the commit that measured it.
2. **Those numbers are uncalibrated** (`ckpt_version` 2). The offset null and the stack
   arrived in `ckpt_version` 3. A negative bias only *suppresses* a positive detection, so
   the six ALIVE lines are conservative; the 53 `absent_in_exposures` are not, and must be
   re-measured before any of them is believed.

### Two things the committed table says on its own

**The triage's significance is not calibrated.** Persist measures the *same line in the
same coadd* against the local scatter and gets a median of **0.34×** the triage's number.
Of the 70 lines with both, **51 % fall below 3 σ** and **74 % below 5 σ**. So
`absent_in_exposures` for most of them is not "the coadd made a feature its inputs lack";
there was no feature at a calibrated significance to begin with. Both readings are true
and they must not be confused — the per-exposure deficit is real *and* most of these
lines were never significant.

**The candidates cluster on plates.** An SDSS plate is one exposure set on one pair of
CCDs.

| plate | lines | coadd σ (persist) | triage σ | same-wavelength fibre pairs |
|---|---|---|---|---|
| 2333 (SEGUE, MJD 53682) | **11** of 71 | 1.1–3.3 | 8.4–13.3 | **2** (3947.299 Å, 4357.125 Å) |
| 3241 | 6 | 1.9–5.8, one at 11.9 | 8.5–14.7 | 0 |
| 0412 | **1** | 10.5 | 16.8 | 0 |

Two *different fibres* of plate 2333 carrying a candidate at exactly the same wavelength
are different objects on the same detector columns: that is a bad column, not a sky.
Plate 2333 contributes 15 % of everything measured and not one of its lines survives a
calibrated coadd measurement.

Against that background the two strongest candidates look structurally different: 6809.3
is the **only** candidate on its plate, and 8578.3 has more than twice the coadd
significance of any of its five plate-mates. `plate_n_other_candidates` and
`plate_other_fibre_same_wavelength` now travel with every line.

### The six lines left standing (run 35738206630, uncalibrated)

| plate-mjd-fiber | RA, Dec | λ_obs (Å) | mode | coadd EW (Å) | combined σ | present | χ²p | 2nd epoch | SIMBAD | atmosphere |
|---|---|---|---|---|---|---|---|---|---|---|
| 3241-54884-0388 | 36.270252, +24.174448 | 8578.276 | emission | 0.800 | 14.1 | 13/14 | 0.76 | none | — | OH list gap, 27 Å |
| 0412-51942-0465 | 47.488109, +0.504935 | 6809.261 | emission | 6.300 | 13.5 | 5/5 | 0.16 | **confirmed, 19.5 σ, 8 epochs** | LM* | OH list gap, 54 Å |
| 2750-54242-0547 | 223.444840, +14.514496 | 6856.461 | emission | 1.625 | 7.1 | 2/3 | 0.33 | none | — | OH 9.3 Å away |
| 0571-52286-0247 | 147.323570, +2.814817 | 7490.314 | emission | 1.334 | 6.4 | 2/5 | 0.49 | not seen | LM* | — |
| 2076-53442-0329 | 120.052400, +7.637376 | 6403.243 | emission | 2.816 | 5.7 | 3/7 | 0.46 | none | — | OH 15 Å away |
| 3327-54951-0356 | 169.349540, +17.489308 | 6967.869 | emission | 1.038 | 5.5 | 3/5 | 0.78 | none | — | **inside H₂O 7200 band** |

No sky-model line at any of them (`on_sky_line` false throughout; per-exposure
`sky_peak_sig` ≤ 3.1). All six are in the red, 6403–8578 Å — the part of the spectrum
where the hand-kept OH list is least complete and the telluric bands live.

**What would kill each**

* **0412-51942-0465 @ 6809.3 Å** — the strongest, and the one to settle first. Present in
  5/5 exposures across two nights at 5.2–7.0 σ, confirmed in a second epoch at 19.5 σ
  with 8 further epochs available, no sky line, only one candidate line in the whole
  spectrum. Its EW varies 5.1 → 9.4 → 9.3 → 5.8 → 5.5 Å between exposures, a factor 1.8,
  which is formally consistent (χ²p = 0.16) but is not what a steady source looks like.
  The object is an M1V dwarf (SIMBAD `LM*`, 2MASS J03095713+0030176). A profile fit to
  the stored coadd window gives **FWHM 3.09 Å against a 3.40 Å LSF — unresolved**, a peak
  of 19.8 on a continuum of ~7.7, two to three pixels wide. So it is not a molecular band
  gap. It is also the **only** candidate on its plate and the nearest other candidate in
  the whole 350 is 43.8 Å away, so neither the plate statistic nor the pixel-coincidence
  statistic touches it. What can still kill it, in order of how likely it now looks:
  **a background star-forming galaxy in the fibre at z = 0.037268** — [N II] 6584 is
  already there at 2.0 σ with a textbook ratio of 0.23, and Hβ at 5043.9 Å and [O III] at
  5194.9 Å decide it; **how many distinct fibres its 8 "other epochs" actually are** (if
  they are all 0412-…-0465 on other nights they are one CCD column, not eight epochs);
  **the same-plate control**; and the same-type control. All four are in
  `--stage control`.
* **3241-54884-0388 @ 8578.3 Å** — 14 exposures over six nights (MJD 54879–54884), every
  one positive, χ²p = 0.76, EW 0.42–0.96 Å, ratio to coadd 0.94. The cleanest persistence
  in the set. Killed by: a same-type control detection; a complete OH atlas covering the
  8548–8621 Å gap; or a species the 792-line list omits near 8578 Å (it is 655 km s⁻¹
  from Paschen 13, so not that).
* **3327-54951-0356 @ 6967.9 Å** — already dead three ways. Its fitted FWHM is 0.52 × LSF,
  narrower than the instrument can make, i.e. a one- or two-pixel defect. It is inside the
  H₂O 7200 telluric band, with `sky_peak_sig` up to 3.1, cosmic-ray flags in 2 of 5
  exposures and per-exposure σ of only 1.6–2.9. And **2027-53433-0246 carries a candidate
  at 6969.474 Å — 1.605 Å away, which at this wavelength is exactly one SDSS pixel**
  (Δλ = λ ln10 × 10⁻⁴ = 1.604 Å), on an unrelated sightline, also inside the telluric band
  and also classified `persistent`. Two unrelated fibres spiking one pixel apart in a
  telluric band is the 1-pixel coincidence excess in the flesh.
* **2750-54242-0547 @ 6856.5 Å** — an OH line 9.3 Å away at a local density of 4.2 listed
  lines per 100 Å, and only 3 exposures. Killed by the any-star control.
* **0571-52286-0247 @ 7490.3 Å** and **2076-53442-0329 @ 6403.2 Å** — present in 2/5 and
  3/7 exposures, EW varying by factors of 2 and 7. These reached `persistent` through the
  low-S/N branch of the classifier and should not survive the null calibration.

### Two guards that came out of this run

* **A sharded run must not be able to split into two estimators.** The shard jobs now
  check out `github.sha`, the dispatched commit, instead of the branch head; they only
  upload artifacts, so nothing needs the branch. The step that picks up already-committed
  checkpoints fetches `results/spectra_persist/ckpt` alone rather than pulling the branch
  over the pinned code. Every checkpoint records the commit that measured it (`code_sha`)
  and the summary tallies them (`checkpoint_code_shas`), so a split run is visible in the
  output instead of having to be reconstructed from job timestamps.
* **A late reduce must not replace a measurement with a no-data verdict.** A reduce that
  arrives after a `ckpt_version` bump holds only superseded checkpoints; it would emit
  `NO_DATA_REACHED` over a real summary and commit it, and the workflow's own assert
  would pass, because a no-data verdict with nothing measured is exactly what that assert
  allows. `reduce_results` now leaves the file alone in that case and returns the existing
  summary with `reduce_skipped` saying why. With nothing to overwrite it reports
  `NO_DATA_REACHED` honestly, as before.

### What each output file holds

| file | what it is |
|---|---|
| `summary.json` | the verdict, the funnel, `checkpoint_code_shas` (which commit measured what), `pixel_coincidence` per release, `most_crowded_plates`, and the `alive` records in full |
| `persistence.csv` | one row per survivor line, every column below |
| `exposures.json` | the per-exposure measurements behind each row, keyed `spec_id@wavelength` |
| `ckpt/<spec_id>.json` | one checkpoint per spectrum, with `ckpt_version` and `code_sha` |
| `control.json` | the control stage: three comparison samples, the epoch series, the profile fit and the background-galaxy scan per surviving line |
| `diagnose.json` | the side-by-side that found the systematic (SPARCL coadd vs file coadd vs each exposure) |
| `probe.json` | SPARCL fields served per release and which archive URLs answer |

The columns that decide a verdict, in the order they are applied:

`known_line_match` → `plate_other_fibre_same_wavelength` → `persistence_class` with
`coadd_sig_cal` → `second_epoch` with `other_best_err_rel`. The columns that should be
read *beside* a surviving line, none of which is enforced: `combined_sig_raw` next to
`combined_sig`, `null_exposure_bias_sig` and `null_n_measurements` and `null_calibrated`,
`stack_sig`, `n_other_candidates_within_3A` and `nearest_other_candidate_dA`,
`plate_n_other_candidates`, `telluric_band` and `oh_gap_A` and `oh_density_per_100A`,
`between_band_heads` with the two `band_*` columns, and `n_lines_in_spectrum`.

### Known limitations, stated rather than hidden

* **The DESI coadd significance is not calibrated.** `desi_measure_at` re-measures the
  cframe rows at the null's offsets but has no coadd to re-measure, so
  `n_coadd_measurements` is 0 for the DESI route and `coadd_sig_cal` falls back to the
  uncalibrated `coadd_sig`. The per-exposure statistic — the one the classification turns
  on — *is* calibrated for both routes. On the SDSS side the calibrated coadd
  significance runs a median 0.34× the triage's, so the DESI coadd numbers should be
  assumed optimistic by a comparable factor until this is closed. The fix is small:
  pass the SPARCL coadd arrays into `desi_measure_at` so the offsets measure it too. It
  was deliberately not made while run 35747997902 was queued, because it changes
  measured numbers and would have risked splitting that run across two estimators.
* **`plate_other_fibre_same_wavelength` is a lower bound.** It can only see fibres whose
  lines were also measured, so it fires where there is evidence and stays silent where
  there is none — the right way round for a channel whose object is a detection.
* **The OH list has gaps and the telluric bands are intervals, not lines.** Both are
  reported (`oh_gap_A`, `oh_density_per_100A`, `telluric_band`) and neither is enforced;
  the empirical replacement is the any-star control sample.

### In flight

| run | stage | dispatched (EDT) | state |
|---|---|---|---|
| **35758868818** | `run`, `ckpt_version` 4, 2 shards | 13:09 | queued |
| **35751666444** | `control`, 40 comparison spectra per line | 12:04 | queued |

Runner concurrency is the binding constraint on this repository — 18 channels share one
account — and run 35747997902 (the same `run` stage at 4 shards) sat queued for 98 minutes
without a single job starting before it was cancelled and re-dispatched smaller. Under the
old fixed `[0..7]` matrix a 4-shard dispatch asked the scheduler for **eight** runners,
four of them only to evaluate their own skip condition and exit; the `plan` job now builds
the matrix from `n_shards`, so 35758868818 asks for three.

### Next decisive action

1. Land 35758868818 so all 167 lines are measured once, with the offset null and the
   stack, by one commit.
2. Read 35751666444. For the strongest candidate it decides four things at once: whether
   Hβ (5043.9 Å) and [O III] (5194.9 Å) are there at z = 0.037268; how many **distinct
   fibres** its eight "other epochs" are; whether other fibres of plate 412 spike at
   6809.3 Å; and whether other M1 dwarfs do.
3. Close the stated limitation: pass the SPARCL coadd arrays into `desi_measure_at` so
   the DESI coadd significance is calibrated like the SDSS one. Needs a `CKPT_VERSION`
   bump, so it waits for a moment when no run is in flight.
4. If 8578.3 Å survives all of it, the next step is outside SDSS: a complete airglow
   atlas for 8500–8700 Å, where the hand-kept list jumps from 8548.5 to 8620.8 Å.
