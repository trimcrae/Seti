# MIDDEN — short-lived radionuclides in stellar photospheres

**Claim being tested.** Whitmire & Wright (1980, Icarus 42, 149) proposed
that a civilization disposing of fission waste into its host star would lace
the photosphere with isotopes that natural nucleosynthesis cannot maintain
there: **technetium** (no stable isotope; longest-lived ~4 Myr, and the
s-process channel that makes it naturally operates only in TP-AGB
interiors), **promethium** (longest-lived isotope 17.7 yr — *any* detection
implies injection within decades), and **actinide overabundances (U, Th)
without the accompanying r-process rare-earth pattern**. The signature has a
built-in radioactive-decay clock and — unlike almost every other
technosignature — survives the death of its makers for ~10 half-lives: it is
a search for **middens**, the residue of civilizations that may be long gone.

## 1. Novelty status

Proposed 46 years ago, never executed as a survey by anyone. The
Whitmire-Wright citation tree contains reviews and essays only (verified in
the 2026-07-25/26 literature sweeps; evidence in `results/litcheck/`, sister
channel queued in `docs/herdsman.md` section 6). What exists instead:

* **Contested single-star claims**, all in chemically peculiar Ap stars:
  Przybylski's Star **HD 101065** (Tc/Pm claims from the 1960s onward;
  Cowley et al. 2004 claimed Pm II and short-lived actinides), **HD 965**
  (Cowley's Pm II claim), **HR 465** (= HD 9996; Bidelman's Pm suggestion),
  and **HD 25354** (promethium claims in the Soviet-era literature). None is
  a survey; all are disputed.
* **The refutation standard**: Andrievsky et al. 2023 re-analysed the
  Tc I 4297.06 A region of Przybylski's Star and showed the "Tc" feature is
  reproduced by known lines of other species — i.e. single-line claims in
  REE-forest spectra die under multi-line laboratory-wavelength coincidence
  tests against Teff-matched comparison stars. MIDDEN adopts exactly that
  standard as its *detection* criterion (section 4).
* **Archive precedent**: Fields & Goodman 2025 mined the public HARPS
  archive at scale for a different signal, establishing that ESO Phase-3
  spectra support survey-scale line work of this kind. The 2026 polluted-WD
  *chemical* search is the nearest technosignature neighbour; MIDDEN differs
  in both population (main-sequence archives, not WD accretion) and
  discriminant (the decay clock, not bulk composition).

MIDDEN is therefore a new signature x new corpus: the first *survey-scale*
execution of the 1980 proposal, with the prior single-star claims recycled
as sanity anchors that the pipeline must naturally re-observe.

## 2. Target lines (`src/seti/midden/lines.py`)

| line | air A | why |
|---|---|---|
| Tc I | 4238.19, 4262.27, 4297.06 | resonance triplet (Meggers & Scribner lab wavelengths; Merrill's S-star discovery lines; the standard AGB Tc diagnostic; 4297.06 is the Andrievsky et al. 2023 line) |
| U II | 3859.57 | uranium cosmochronometer line (Cayrel et al. 2001) |
| Th II | 4019.13 | thorium cosmochronometer line (Butcher 1987) |
| Fe I x11 | 4045.81 ... 4404.75 | RV registration (strong, ubiquitous in A5-F2) |
| DUMMY x3 | 4152.30, 4222.10, 4288.40 | deliberately-not-lines in quasi-clean continuum: false-positive controls |

The encoded wavelengths are **never silently trusted**: the first CI step
(`midden --stage verify-lines`) queries the NIST Atomic Spectra Database
lines API for every non-control species and hard-fails the run unless each
wavelength matches a NIST line within 0.05 A
(`results/midden/line_verification.json` records the deltas). Controls are
instead checked to sit > 3 A from every encoded real line.

## 3. Corpus (`src/seti/midden/acquire.py`)

v1 = ESO Phase-3 **HARPS + FEROS** 1D spectra (both cover 3859-4405 A;
R ~ 115k / 48k), discovered by uploading the target list to
`https://archive.eso.org/tap_obs` (table `ivoa.ObsCore`,
`dataproduct_type='spectrum'`, instrument HARPS/FEROS) in checkpointed
chunks and position-matching at 2". Targets, in priority order:

1. **Anchors** — HD 101065, HD 965, HR 465 (HD 9996), HD 25354 (Sesame-resolved
   on the runner; encoded fallback coordinates get a 30" match radius).
2. **Renson & Manfroid (2009) Ap/Bp/CP stars** — VizieR `III/260` via
   TAPVizieR with dynamic column resolution (a TOP-1 probe maps current
   column names; naming drift cannot break the pull). This is where the
   prior claims live and where line-forest contamination is worst — the
   control lines exist for them.
3. **Bright A5-F2 dwarfs** — the Whitmire-Wright predicted repository class
   (a star convective enough to keep waste in the photosphere, quiet enough
   to see 10-mA lines): Gaia DR3 `teff_gspphot` 6800-8300 K,
   `logg_gspphot > 3.8`, G < 9, parallax > 2 mas.

De-duplication keeps up to **3 epochs/star** (persistence + decay-clock
tests); the corpus is capped (workflow input, default 3000 spectra) with
breadth-before-depth ordering. Download/analysis is a strict
**process-and-discard** loop: ~50 FITS to scratch, measure, checkpoint the
per-star results to parquet, DELETE the FITS — the ~14 GB runner disk never
accumulates spectra, and a killed job resumes at the batch where it died
(artifact-seeded resume via `resume_run_id`).

## 4. Detection statistic (`src/seti/midden/measure.py`)

Per spectrum: read wavelength/flux (binary-table and image-HDU Phase-3 forms
both handled), adopt the pipeline CCF RV from the header **only when the
Fe I cross-correlation confirms it** (catalog-RV keywords are never
trusted); otherwise use the cross-correlation RV, which also absorbs any
global air/vacuum or zero-point offset (both velocity-like at our 3% band).
Every line window is shifted to the stellar frame; each line gets a robust
local continuum (deg-2 polynomial over +-3 A, central +-0.25 A excluded,
asymmetric clipping so absorption cannot drag the fit) and a central depth
+ EW proxy with an uncertainty from the local scatter.

**Self-calibrating census z** (no absolute synthesis, same philosophy as
HERDSMAN-B): each star's depth at each wavelength is ranked (median/MAD)
against the depths *at the same wavelength* across all corpus stars within
+-250 K Teff. Blends, unresolved line forests, telluric and blaze structure
common to the Teff slice cancel identically.

A **candidate** requires all of:

* >= 2 radionuclide lines at z >= 4, **or** the coherent-triplet path: all
  three Tc I components at z >= 2.5 with quadrature sum >= 4.5 (three
  independent 3-sigma lines at the laboratory spacings — the
  Andrievsky-standard multi-line coincidence);
* every flagged line individually significant (depth >= 2 x its own error);
* **no control/dummy line flagged** in the same epoch (line-forest veto);
* when the star has >= 2 epochs: the verdict repeats in >= 2 epochs.

Per-line flag rates for the controls are reported next to the target lines
in `REPORT.md` — if the controls fire at a comparable rate, the target-line
excesses are systematics by the pipeline's own admission.

## 5. Contamination ledger

| systematic | why it mimics | discriminator |
|---|---|---|
| line blends (Nd II/Cr II/etc. at ~4238/4262/4297) | any single "Tc" line is reproducible by other species (this killed the Przybylski claims) | require multi-line coincidence at the laboratory wavelengths in the stellar frame; census z against Teff-matched peers subtracts the mean blend for that Teff |
| Ap-star exotic line forests | REE forests blanket 4200-4300 A and can beat any single-line statistic | dummy-control veto: a forest that lifts the target windows lifts the controls too; forest stars self-identify and are excluded from candidacy, not from the report |
| s-process Tc in true AGB/S giants (misclassified "dwarfs") | Tc triplet is genuinely there — natural | `logg_gspphot > 3.8` selection; anchor on the decay logic (an AGB star also shows ZrO/s-process signatures and sits elsewhere in the HRD); such hits are reported as a known natural class, never as candidates |
| telluric residuals | spurious absorption features | negligible tellurics blueward of 4400 A; and a telluric sits at fixed *observer-frame* wavelength — it cannot track the stellar frame across stars/epochs with different RVs and barycentric corrections |
| cosmic rays / bad pixels | narrow spurious features | affect single epochs (persistence requirement); depth is a multi-pixel core average with a local-scatter error bar |
| wavelength calibration / air-vacuum mismatch | global shift of every window | velocity-like; absorbed by the Fe I registration, which is measured per spectrum |
| continuum/blaze residuals | broadband depth offsets | local deg-2 continuum + census self-calibration at the same wavelength |
| RV failure (fast rotators, SB2s) | windows land off-line | `rv_source`/`rv_ccf_nlines` recorded; unconfirmed-RV spectra can be cut in re-scoring without re-download |

## 6. Interpretation ladder

1. **Instrumental/systematic** — control lines fire, or single-epoch only:
   vetoed, feeds the contamination ledger.
2. **Known natural** — evolved s-process star in disguise: reclassified,
   interesting only as a catalog correction.
3. **New stellar astrophysics** — a genuine non-AGB photospheric Tc/actinide
   excess with an r-process pattern: publishable astrophysics regardless of
   interpretation (this is the "remarkable either way" rung).
4. **Whitmire-Wright candidate** — multi-line Tc (and/or actinides *without*
   the r-process pattern) in a main-sequence star, persistent across epochs,
   controls clean. Follow-up: Pm II lines, isotope-sensitive features,
   epoch-resolved decay curves (Tc-98/99 vs Pm half-lives date the
   injection), targeted deep spectroscopy.

## 7. No-null rule

Per `CLAUDE.md`: an empty candidate list at these thresholds is a statement
about this corpus and these windows, **not a publishable null**. The
escalation path is more corpus (UVES/ESPRESSO/X-shooter collections; APOGEE
is useless here — wrong band), more lines (Pm II blue lines, additional Tc I
subordinate lines for confirmation), and sharper population cuts — the
question changes, the write-up waits for a detection.
