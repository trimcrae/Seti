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
  The object is an M dwarf (SIMBAD `LM*`), and 6809 Å sits among the TiO band structure,
  so **the control sample on other M dwarfs is what kills it** — a band-head gap is
  persistent in every exposure and every epoch and neither of those tests can see it.
* **3241-54884-0388 @ 8578.3 Å** — 14 exposures over six nights (MJD 54879–54884), every
  one positive, χ²p = 0.76, EW 0.42–0.96 Å, ratio to coadd 0.94. The cleanest persistence
  in the set. Killed by: a same-type control detection; a complete OH atlas covering the
  8548–8621 Å gap; or a species the 792-line list omits near 8578 Å (it is 655 km s⁻¹
  from Paschen 13, so not that).
* **3327-54951-0356 @ 6967.9 Å** — inside the H₂O 7200 telluric band, `sky_peak_sig` up
  to 3.1, cosmic-ray flags in 2 of 5 exposures, per-exposure σ only 1.6–2.9. Expected to
  fall to the calibrated re-measurement on its own.
* **2750-54242-0547 @ 6856.5 Å** — an OH line 9.3 Å away at a local density of 4.2 listed
  lines per 100 Å, and only 3 exposures. Killed by the any-star control.
* **0571-52286-0247 @ 7490.3 Å** and **2076-53442-0329 @ 6403.2 Å** — present in 2/5 and
  3/7 exposures, EW varying by factors of 2 and 7. These reached `persistent` through the
  low-S/N branch of the classifier and should not survive the null calibration.

### Next decisive action

1. Land the `ckpt_version` 3 run (run 35747997902) so all 167 lines are measured once,
   with the offset null and the stack, by one commit.
2. Run `--stage control` on whatever is still standing. That is the only test left that
   can reject an M-dwarf band-head gap, and it is what the strongest candidate turns on.
3. If 8578.3 Å survives both, the next step is outside SDSS: a complete airglow atlas for
   8500–8700 Å, and the other 8 epochs of 0412-51942-0465 measured individually rather
   than through the best-of summary.
