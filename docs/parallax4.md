# PARALLAX4 — Gaia DR4 intake: is the anomaly ON the star?

Package `src/seti/parallax4/`, workflow `.github/workflows/parallax4.yml`,
config `config/parallax4.yaml`, tests `tests/test_parallax4.py`, results
`results/parallax4/`.

## 1. Claim

A grey (achromatic across G, BP, RP) dip or brightening in Gaia's
per-transit photometry that comes with **no simultaneous photocentre shift**
at the sub-mas level is a brightness change *on the source that dominates the
photocentre*. That kills the dominant failure mode of every optical anomaly
search in this repository — a blend, a neighbour, a contaminating variable in
the aperture — at a scale set by Gaia's along-scan centroid error (0.1–1 mas
per CCD) instead of a TESS pixel (21″): a localisation of the varying light
to ≲ 1–20 mas of the photocentre, i.e. 10³–10⁵ × finer.

The inverse diagnostic runs on every other channel's shortlist: does the
photocentre move with the brightness (a blend: reject) or not (the anomaly is
on the target: keep)?

The data that make this possible arrive with **Gaia DR4, scheduled for
2 December 2026** (ESA Gaia DR4 content page,
<https://www.cosmos.esa.int/web/gaia/dr4>): 66 months of data (mid-2014 to
early 2020), ~2.8 billion sources, and — for the first time — *all* epoch data:
per-transit astrometry for every source, per-transit G/BP/RP photometry,
epoch radial velocities, and epoch BP/RP and RVS spectra (~400 TB, most of it
epoch products). On 26 June 2026 ESA prereleased the epoch astrometry of 12
sources with a draft DR4 data model
(<https://www.cosmos.esa.int/web/gaia/dr4-prerelease>; the file
`gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip`, sha256 `07f0e8d9…fa0b`,
declares itself `Gaia DR4_RC3`). That file is the reader's format validation.

## 2. Novelty status

* **Varstrometry** — photocentre jitter from variability in an unresolved
  pair — is established: Hwang, Shen, Zakamska & Liu (2020, ApJ 888, 73,
  arXiv:1908.02292) used Gaia DR2's *summary* astrometric excess noise against
  photometric variability to find dual AGN and blends. **Variability-Induced
  Movers** go back to Hipparcos (Wielen 1996; the Hipparcos VIM solutions),
  and Gaia DR3's non-single-star pipeline published a VIM table
  (`gaiadr3.nss_vim_fl`; Gaia Collaboration, Arenou et al. 2023). The *physics*
  of this test is therefore not new, and it is cited as such.
* **What is new** is (i) the per-transit form, regressing each transit's
  along-scan residual on the same transit's flux, with scan-angle-resolved
  rejection of calibration systematics — impossible before DR4 because no
  per-transit stellar astrometry was public (the June-2026 prerelease has 12
  sources); (ii) applying it as a *veto for technosignature candidates*, i.e.
  requiring that a grey photometric anomaly be proven on-target at mas scale;
  (iii) the release-day watchlist that turns every shortlist in this
  repository into a DR4 test. No published technosignature search uses
  Gaia per-transit astrometry (none could: the data do not exist yet).
* Grey per-transit dips in Gaia DR3 epoch photometry as a population are,
  to our knowledge, unsearched as such; Gaia's own variability pipeline
  classifies light curves, it does not ask for achromatic multi-transit
  episodes.

## 3. Method

### 3.1 The photometric half (runs today on Gaia DR3)

Per source (DR3 epoch photometry, the ~11.75 M sources with published light
curves, read from the ESA CDN bulk files), `greydip.detect_source`:

1. baseline and robust scatter per band; the star's own variability `s_int`
   is folded into every transit's sigma;
2. same-sign ≥ 3σ G deviants are chained into episodes (gap ≤ 1 d, no normal
   transit in between); an episode qualifies at ≥ 5σ and ≥ 1% in G;
3. BP and RP, combined over the episode, must deviate ≥ 3σ with G's sign;
4. **grey**: one common fractional depth (χ², 2 dof, p ≥ 0.01, 0.3% floor)
   AND BP/RP depth ratio within 0.25 of 1 measured to ≤ 0.20 (small-grain dust
   gives 1.7; an unmeasured ratio is `GREY_UNCONSTRAINED`, never `GREY`);
5. **coherence**: ≥ 2 transits = `MULTI`; one transit whose other-FoV partner
   (106.5 min away) is normal = `SINGLE_CONTRADICTED`;
6. **periodicity**: with ≥ 3 dip episodes, a period search that folds every
   episode into one or two arcs (primary + secondary) at false-alarm < 0.01,
   with the arcs ≥ 50% dips, marks an eclipsing binary `PERIODIC`.

Tier A = GREY ∧ MULTI ∧ ¬PERIODIC. Tier B = GREY single transit with no
partner to contradict it. Everything else is kept in the artifact for the
funnel and never promoted.

Cross-source: `reduce` pools every shard's per-0.25-d histogram of usable
transits and grey episodes; a time bin whose grey-episode rate per transit
is > 10× the global rate (Poisson p < 10⁻⁶) is instrumental, and its events
are vetoed (`epoch_cluster`).

Vet (`vet_rules`): Gaia's own eclipsing-binary table / ECL class, DR3 VIM
membership, `duplicated_source` (a known source of spurious transits), and a
**misassigned-transit test** — the episode's flux equals a neighbour's (dip)
or target + neighbour (brightening) within 10″ — are kills; RUWE > 1.4,
`ipd_frac_multi_peak` > 10 and a neighbour within 2″ at Δm < 3 are flags.

### 3.2 The astrometric half (DR4; validated today on the prerelease)

`photocentre.fit_photocentre`: two sources, target T and neighbour N at
ρ·u, β = F_N / F. With φ = F/F_med − 1 and ψ = φ/(1 + φ),

    along-scan offset = const + ψ · (D_α* sin θ + D_δ cos θ),

exactly, for any depth: D = ρ(1 − β)·u if N varies, −ρβ·u if T varies, 0 if
the variation is on the photocentre. Per source a weighted least-squares fit
of the transit-level AL centroid on [sin θ, cos θ, parallax factor,
t sin θ, t cos θ, ψ sin θ, ψ cos θ, ψ], the last column a **detector-frame**
coupling that does not flip with scan direction (CTI, PSF-vs-magnitude
calibration). A jitter makes χ²/ν = 1 (orbits, attitude noise).

Verdicts in precedence: `INSUFFICIENT`, `NO_FLUX_VARIATION`,
`SCAN_ANGLE_FLUX` (flux predictable from θ harmonics, or from a known
neighbour's window membership: a static neighbour entering the 0.7″ × 2.1″
AF window mimics a blend), `DETECTOR_FRAME`, `SECTOR_INCONSISTENT` (per-
scan-sector slopes disagree with one sky vector), `BLEND_NEIGHBOUR` (D
significant and along a catalogued neighbour's direction, |D| within
[−ρβ, ρ(1−β)]), `BLEND_UNRESOLVED`, `ON_TARGET` (D consistent with 0,
95% bound ≤ 20 mas), `AMBIGUOUS`.

Conventions verified on real data (`epochs.py`): the AL unit vector is
(sin θ, cos θ) with θ = `scan_pos_angle` — a 5-parameter fit with it recovers
all 12 prerelease parallaxes (HD 114762 25.63 vs 25.6 mas; Gaia-4 13.62 vs
13.6; LSPM J1324+6124 10.11 vs 10.1); `obs_time_tcb` is at Gaia and
`t_bary = obs_time_tcb + obs_time_bary_corr` (draft data model); and **the DR3
photometric transit_id is the DR4 transit_id** — 63 of Gaia-4's 65 DR3
photometric transits are in the prerelease, barycentric times agreeing to
~1–5 s.

The reader discovers columns at run time (`schema.py`: roles → ordered
spellings, snake_case and CU9 camelCase, prerelease, DR3 DataLink, DR3 CDN
bulk, SSO per-transit tables); a role that resolves to nothing is named
(`SchemaError.resolution.missing`), never guessed. The probe downloads the
draft DR4 data model on the runner and records which roles its identifiers
resolve.

## 4. Controls (run first; they gate)

Photometric (gate the sweep — a sweep shard refuses to run without
`gate_photometric = PASS`):
P1 grey two-transit dips injected into real DR3 light curves recovered as
tier A ≥ 80% at S/N ≥ 10; P2 dust-ratio injections called grey ≤ 5%;
P4 bulk reader alignment; P3 (reported) Gaia EBs caught by the period test.

Astrometric (gate the DR4 verdicts):
A1 simulated DR4-shaped scenes return their known verdicts (blended EB →
`BLEND_NEIGHBOUR` with D to < 5%; EB on target and grey dip on target →
`ON_TARGET`; window neighbour → `SCAN_ANGLE_FLUX`; CTI → `DETECTOR_FRAME`);
A2 the real prerelease (12 parallaxes; DR3 photometry joined; the joint fit
on the sources with DR3 light curves); A3 DR3 varstrometry at population
level — blended (`ipd_frac_multi_peak` ≥ 8) eclipsing binaries out-jitter
blended quiet stars at matched G, the excess rises with amplitude, and is
smaller among isolated stars; A4 the release-day control list
(`dr4_controls.csv`): every DR3 VIM (expected BLEND) and 300 bright isolated
clean EBs (expected not BLEND). On release day `stage_dr4` runs these first
and issues no verdict unless ≥ 70% of testable VIMs are blends and ≤ 10% of
testable single EBs are.

## 5. Contamination model

| Failure mode | Signature | Where it dies |
|---|---|---|
| Small-grain dust | BP/RP depth ratio ≈ 1.7 | grey test |
| Spots, pulsation, flares | chromatic | grey test |
| Twin-temperature eclipsing binary | grey but periodic | period test; Gaia ECL / EB table |
| One-transit instrument glitch | single transit, FoV partner normal | coherence |
| Scan-wide instrument event | same epoch across unrelated sources | epoch-cluster veto |
| Transit assigned to the wrong source | episode flux = a neighbour's | misassigned-transit test |
| Duplicated source | `duplicated_source` | vet kill |
| Blend / neighbour variable | photocentre moves with flux | **DR4**: BLEND_* |
| Static neighbour entering the window | flux tracks scan geometry | **DR4**: SCAN_ANGLE_FLUX |
| CTI / PSF-vs-magnitude calibration | coupling that does not flip with θ | **DR4**: DETECTOR_FRAME |
| Large-grain (grey) natural dust | grey, on target | NOT excluded: a surviving event is a candidate for this before anything else |

A tier-A survivor today is a grey, colour-constrained, multi-transit,
non-periodic episode on a source Gaia does not call an eclipsing binary or a
VIM, not duplicated, with no neighbour whose flux reproduces the episode.
**It is not a detection** until DR4's per-transit photocentre returns
`ON_TARGET`, and even then natural grey occulters (large-grain dust,
debris) are the first explanation to exclude.

## 6. Running

    python -m seti.parallax4.run --stage probe|controls|sweep|reduce|vet|watchlist|dr4|summary
    # runner: workflow_dispatch parallax4.yml (stage=all; stage=dr4 on release day)

## 7. Results

See `results/parallax4/summary.json` and the PARALLAX4 section of `STATUS.md`.
