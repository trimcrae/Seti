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

1. **FSPL fit** (t₀, t_E, u₀, ρ_*), every dataset with its own linear f_s, f_b
   (so OGLE magnitudes and KMTNet difference fluxes never need a common zero
   point). Small-u₀ events get a grid of ρ_* and t_E starts *before* any
   clipping (a flattened peak fitted as a point source otherwise gets clipped
   as outliers). Local-outlier clipping against a running median of the
   residuals (a step survives, a lone bad point does not). Per-dataset error
   scale = max(point-to-point, MAD), so a real step can only lower its own
   significance.
2. **Threshold scan**: Δχ² of the occultation template for every u_c (wing)
   and u_h (hole), to first order in every nuisance parameter
   (Δχ² = −2 r·d − d·d + (Jᵀd)ᵀ(JᵀJ)⁻¹(Jᵀd), all cumulative sums over u) —
   without the last term a deep step is absorbed by a shorter t_E and the scan
   points at the wrong u_c. The same scan with the sign flipped is the
   **anti-occultation**: same freedom, same sensitivity to every systematic,
   produced by no physics.
3. **Refinement** from the best scan maxima plus fixed ρ_L seeds (and, when a
   hollow centre is seen, a t_E/u₀ grid centred on the excess-flux centroid),
   all shape parameters free, optimiser on a soft-edged hybrid model, every
   reported χ² the exact finite-source occulted model.
4. **Gates** (`detect.gate_rejections`, a pure function of the diagnostics):
   depth tied to ρ_L (α = 1 within 3σ); each side of t₀ significant on its
   own, equally deep, with overlapping per-side edge intervals; ≥ 2 sites see
   it and all sites agree; bands agree (achromatic); dropping any night or
   any dataset keeps ≥ 50 % of Δχ²; each step bracketed by data; hole sampled
   and below baseline; no positive residual bump (caustics brighten, shadows
   do not); no red residual; anti-template Δχ² < ½ of the occultation's.
5. **Threshold**: a gate-passer counts only above the largest anti-occultation
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
published binary-lens events must all come out NOT an occultation; opaque
lenses (ρ_L = 0.5, 0.7, 0.85) injected into the real baselines of the
finite-source events must be recovered (≥ 70 % of injections whose Fisher
expected Δχ² exceeds 500). Anything else stops the screen.

## 7. Sensitivity

Every screened event gets the analytic expected Δχ² at its own solution on
the ρ_L grid (0.3 … 2.0; zero by construction where u₀ ≥ u_c), and every 8th
event gets three full injection–recovery trials through the whole funnel.
`summary.json → sensitivity` carries the recovered fraction per (ρ_L, u₀)
cell and the fraction of the plane with ≥ 50 % efficiency.

## 8. Results

See `STATUS.md` (OCCULT section) and `results/occult/summary.json`.
