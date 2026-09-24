# OCCULT — an opaque microlens occulting its own images (S40, ground surveys)

Code: `src/seti/occult/` · workflow: `.github/workflows/occult.yml` ·
config: `config/occult.yaml` · tests: `tests/test_occult.py` ·
results: `results/occult/`

## 1. Claim

A stellar-mass lens that is also an **opaque body a sizeable fraction of its
own Einstein radius across** removes whichever lensed image falls behind it.
For a bulge event R_E ≈ 1–4 AU, so ρ_L = R/R_E ≈ 0.3–1 is an opaque body of
AU scale and stellar mass: a Dyson sphere or swarm, seen **by its gravity and
its shadow** rather than by the waste heat every Dyson search so far has
looked for. A natural lens (a star) has ρ_L ≈ 10⁻³ and is invisible to this
test; a natural body with ρ_L ~ 0.5 needs a density that no bound object at
bulge-lens distances has (the density argument in `seti.roman.lens.density_bound`,
reused here for any survivor).

The observable (Agol 2002 for the physics; `seti.roman.lens` derives it):
with images at |θ±| = (√(u²+4) ± u)/2,

* **wing regime, ρ_L < 1** — the minor image is hidden at every
  u > u_c = 1/ρ_L − ρ_L. An otherwise standard light curve loses
  f_s·A₋(u) outside u_c: a **pair of downward steps symmetric about t₀**, at
  t₀ ± t_E√(u_c² − u₀²), each of depth f_s·A₋(u_c). Where the steps are and
  how deep they are is fixed jointly by the one extra parameter ρ_L. If
  u₀ > u_c the minor image is hidden all the time and A₊ = (A+1)/2 is
  *exactly* a blended Paczyński curve: **undetectable** (this bounds the
  sensitive part of the ρ_L–u₀ plane from first principles).
* **central-hole regime, ρ_L ≥ 1** — the minor image is always hidden and the
  major image is hidden for u < u_h = ρ_L − 1/ρ_L: the flux falls to the
  blend, **below the baseline**, around t₀, between two lensing horns.

Requirements for a detection: (1) symmetric step pair / symmetric hole,
(2) seen by ≥ 2 sites that agree, (3) achromatic depth between V and I,
(4) not carried by one night, one dataset, or one data gap.

## 2. Novelty status

* Physics of lenses that occult: natural-body literature. Agol 2002
  (astro-ph/0207228); the opaque finite lens astrometry paper
  (doi:10.1086/377151); and arXiv:2608.24009 (Aug 2026) on **circum-lens
  disks**, which warns that survey pipelines reject non-Paczyński,
  non-achromatic curves and so could under-detect ring/disk-bearing lenses.
* The in-repo prior-art sweep `romanlit` (run 34302576338, 645 abstracts,
  `results/romanlit/`) found no search of any survey for the symmetric-step
  signature, and no use of the occulting radius as a density test.
* The only prior run of the S40 model in this repository is on **simulated**
  Roman data (`results/roman/lens/`, RMDC26). OCCULT is the first application
  to real photometry, and to the full public OGLE-IV EWS and KMTNet samples.

So: **new signature, new use of existing data**. Nothing here refines an
existing SETI search.

## 3. Data (as the runner probe measured it, not as remembered)

Probe runs 36003872955 and its round-2 successor (`results/occult/probe.json`):

| survey | catalogue | photometry | status |
|---|---|---|---|
| OGLE-IV EWS 2011–2019, 2022–2025 | `www.astrouw.edu.pl/ogle/ogle4/ews/<yr>/lenses.par` | `.../<yr>/blg-NNNN/phot.dat` (HJD, I, σ, seeing, sky), I only | reachable, ~0.13 s per file |
| KMTNet 2016–2025 (public seasons) | `kmtnet.kasi.re.kr/~ulens/event/<yr>/listpage.dat` | `.../data/KB<yy><nnnn>/pysis/pysis.tar.gz` — every site × field × band pySIS file (HJD, Δflux, σ, mag, σ, fwhm, sky, secz), I and V | reachable, ~1.5 s per event |
| MOA | alert pages 404 (`www.massey.ac.nz/~iabond`) or do not resolve (`it019909.massey.ac.nz`) | — | **unreached**, recorded as such |

`ogle.astrouw.edu.pl` serves only the HTML pages (404 on data). KMTNet
directory listings are forbidden (403); the tarball needs none. KMTNet's
policy keeps a season proprietary until 1 July of the next year: seasons in
that window are excluded (`acquire.public_seasons`). A KMTNet event and its
OGLE counterpart (from KMTNet's related-event token, else a 2″ / 20 d
positional match) form **one unit** fitted jointly.

## 4. Method

`detect.assess_event`, pure numpy/scipy:

1. **Data preparation.** Events with > 4,000 epochs (KMTNet prime fields
   reach ~40,000) are clipped model-free and inverse-variance binned per
   dataset at 0.002 t_E (≤ 1 h, well inside the finite-source rounding of a
   step). The per-dataset error scale is the **model-free** point-to-point
   scatter of the flux in units of the quoted errors — controls run
   36015751048 showed that scaling errors on residuals about a wrong null
   model inflates them until the signal vanishes. Local clipping removes a
   point only if it is an outlier against both a running median of its
   model residuals and a running median of the flux (a median follows an
   edge; either test alone clips real structure somewhere).
2. **Null fit**: FSPL (t₀, t_E ≥ 0.05 d, u₀, ρ_*), every dataset with its own
   linear f_s, f_b (OGLE magnitudes and KMTNet difference fluxes never need a
   common zero point). Small-u₀ events get a grid of ρ_* and t_E starts
   before any clipping. **Annual parallax** (π_E,N, π_E,E; Gould 2004
   geometry, low-precision solar ephemeris) is fitted for t_E ≥ 10 d and kept
   when it buys Δχ² > 20; it is carried into every occultation and anti fit,
   and step times follow the parallax track.
3. **Threshold scan**: Δχ² of the occultation template for every u_c (wing)
   and u_h (hole), to first order in every nuisance parameter
   (Δχ² = −2 r·d − d·d + (Jᵀd)ᵀ(JᵀJ)⁻¹(Jᵀd), all cumulative sums over u) —
   without the last term a deep step is absorbed by a shorter t_E and the scan
   points at the wrong u_c. The same scan with the sign flipped is the
   **anti-occultation**: same freedom, same sensitivity to every systematic,
   produced by no physics.
4. **Refinement** from the scan maxima and fixed ρ_L seeds (0.5, 0.7, 0.85,
   1.3 — never filtered by the null fit's geometry, which may be the wrong
   one), each started from the null shape, from the survey's alert solution
   and from a u₀ < 1 restart (an occultation can drag the null fit into a
   degenerate wide solution); a hollow centre (two horns, a trough 5σ deep)
   adds a t_E/u₀ grid centred on the excess-flux centroid. All shape
   parameters free; optimiser on a soft-edged hybrid model; every reported
   χ² the exact finite-source occulted model; in the hole regime a warm-
   started profile over ρ_L. Refinement stops starting fits after 600 s and
   records that.
5. **Gates** (`detect.gate_rejections`, a pure function of the diagnostics):
   depth tied to ρ_L (α = 1 within 3σ + 0.1); each side of t₀ significant on
   its own, equally deep, with per-side edge intervals (Δχ² ≤ 9) overlapping
   within 2 % of u_c; ≥ 2 sites see it and all sites agree; bands agree
   (achromatic); dropping any night keeps ≥ 50 % of Δχ², dropping the best
   dataset keeps ≥ 20 % (and ≥ 25); each wing step (≥ 2 of them) bracketed by
   data within max(5 d, 0.1 t_E); the hole sampled and its data ≥ 5σ below
   the unmagnified level; no positive residual bump above
   max(6σ, 0.15 √Δχ²) (caustics brighten, shadows do not); no red residual
   above max(2, 1.5 × the event's own out-of-event value) carrying > 5 % of
   the occultation's Δχ²; the sign-flipped (anti) template must not fit as
   well or better. Depth comparisons (side vs side, site vs site, V vs I)
   add a 5 % systematic floor in quadrature.
6. **Threshold**: a gate-passer counts only above the largest anti-occultation
   Δχ² over all assessed events (a 1/N false-alarm level), never below 50.

## 5. Contamination model

| impostor | why it looks like S40 | what kills it |
|---|---|---|
| finite-source peak | a smooth departure from Paczyński near t₀ | it is fitted in the null (FSPL); tested on named FSPL events |
| binary-lens caustic crossing / cusp | sharp features, U-shaped crossings | brightenings, not dimmings: `BRIGHTENING_LEFT_CAUSTIC_LIKE`, residual structure; tested on named binary events |
| circum-lens disc (the natural twin) | occults images, but unequally with orientation | asymmetric: `ONE_SIDED_STEP`, `SIDES_DIFFER_IN_DEPTH`, `SIDES_DISAGREE_ON_U_C`, depth not tied to ρ_L |
| parallax / xallarap | asymmetric wing distortion | same symmetry gates; anti template |
| seasonal / site zero-point offsets | a level change between seasons or sites | per-dataset fluxes; `STEP_IN_DATA_GAP`; ≥ 2 sites agree; dataset jackknife |
| one bad night | a local dip | night jackknife, max-night share |
| seeing / sky systematics | chromatic or site-specific | quality cuts; site and band agreement |
| a pipeline that never alerted the event | a hole-regime event is not Paczyński-like | **not handled**: the search sees only alerted events (a stated limit) |

## 6. Positive controls (run first; gate everything)

`config/occult.yaml`: 10 published finite-source single-lens events and 10
published binary-lens events must all come out NOT an occultation. Opaque
lenses (ρ_L = 0.5, 0.7, 0.85, 1.3) are injected into the real baselines of
every control and of 30 ordinary joint KMT+OGLE events (a fixed hash-ordered
sample, `run.baseline_hosts`); among hosts with ≥ 2 sites (a one-site host
cannot pass the two-site rule by construction), ≥ 70 % of the injections
whose nuisance-projected expected Δχ² exceeds 500 must be recovered with the
right regime and ρ_L (5 %; 15 % in the hole regime, where ρ_L trades with
t_E and u₀). Anything else stops the screen.

The gate failed on its first real-data runs, each time on a defect the real
photometry exposed and the synthetic suite had not: a t_E floor above the
free-floating-planet controls; no parallax; errors scaled on a wrong null
model; KMTNet's 2023 switch of `listpage.dat` t₀ to HJD − 2,400,000 (every
2023–2025 event read as having no photometry in its window); a
below-baseline hole *trigger* that silenced every real hole injection; and
refinements trapped in a degenerate null solution. Each is recorded in the
commit that fixed it.

## 7. Sensitivity

Every screened event gets the nuisance-projected expected Δχ² at its own
solution on the ρ_L grid (0.3 … 2.0; zero by construction where
u_min ≥ u_c), and every 8th
event gets three full injection–recovery trials through the whole funnel.
`summary.json → sensitivity` carries the recovered fraction per (ρ_L, u₀)
cell and the fraction of the plane with ≥ 50 % efficiency.

## 8. Results

See `STATUS.md` (OCCULT section) and `results/occult/summary.json`.
