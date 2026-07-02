# Research scoreboard

Live per-channel state of the search. Update this file whenever a run,
vet, or triage changes the candidate picture — it is the single place a
human (or a fresh agent session) looks to know what is hot and what to do
next. Last updated: 2026-07-01.

## Current best candidates (cross-channel, ranked)

1. **167 triaged laser-line priority targets** —
   `results/spectra_triage/priority_targets.csv`. The former #1 (spec 068839f0,
   7518/7542 Å) is **DEAD** — see Resolved below; the two "beacon" lines are
   Hα + [N II] 6584 at z = 0.145, an emission-line galaxy SDSS misclassified as
   a STAR. A new galaxy-redshift-consistency stage now removes such objects
   (`triage_verdict = galaxy_zmatch`; 3 spectra cut, two of them
   SIMBAD-confirmed galaxies — KUG 1207+134, Z 521-35). None of the remaining
   167 is cross-confirmed.
   *Next decisive test:* the `spectra-confirm` repeat-visit path is exhausted
   (zero overlapping SPARCL spectra for 19 single-line targets). Real remaining
   route: **per-exposure persistence** — fetch the coadd-input exposures for each
   target from the SDSS SAS (new acquisition code, runner-side); a real line
   persists across exposures, a cosmic ray does not. Single-line targets cannot
   be galaxy-tested internally — the per-exposure check is what separates a true
   narrow emitter from a cosmic-ray hit for them.
2. **WD IR-excess multimodal candidates** —
   `results/science/multimodal_candidates.csv` (170 anomaly-scored excesses
   from 7,716 clean 100-pc-scale white dwarfs). Not yet pushed through
   variability/full-SED discrimination.

## Resolved (killed) candidates

* **Laser-line #1, spec 068839f0…** (SDSS-DR17, RA 25.6212, Dec −8.2417) —
  ranked first in the whole search: a 31.9σ unresolved line at 7517.96 Å plus a
  second surviving line at 7542.23 Å. **Killed 2026-07-01 by internal
  redshift-consistency**: the pair is Hα 6562.8 and [N II] 6583.5 redshifted to
  **z = 0.1452** (residual 22 km/s on [N II]) — a background emission-line galaxy
  the SDSS pipeline classified as `STAR` (catalogue z ≈ 0, so the observed-frame
  known-line triage placed Hα at 6563 Å and never saw it). New rejection
  `seti.spectra.galaxy_reject.galaxy_redshift_match` (verdict `galaxy_zmatch`).
  A locked diagnostic pair (Hα+[N II], the [O III]/[S II] doublets) or ≥3 lines
  at one z is required, so an emission-line variable star is not mis-killed
  (V345 Sge was correctly spared).
* **Astrometric dark-companion class-3 shortlist** — the 8 AMRF class-3 systems
  (BH1 + 7) were cross-matched against the published Gaia compact-companion
  catalogue Shahaf+2023 (VizieR J/MNRAS/518/2991, 101,380 source_ids loaded;
  `results/accel/literature_crossmatch.csv`). **7 of 8 are already in Shahaf+2023**
  (1 is Gaia BH1, the validation object) — the channel *reproduces* the published
  AMRF catalogue rather than extending it. One system, **Gaia DR3
  3027759739607108992** (852 pc, M₂≈4.4 M☉, RUWE 4.9, no SIMBAD), is absent from
  Shahaf+2023, but it is the *weakest* solution in the set (farthest, lowest
  RUWE, mass nearest the 3 M☉ floor) — most plausibly below Shahaf's quality
  threshold rather than a new object. Not a remarkable novel candidate; would
  need the Shahaf+2024/2019 lists to load and an independent orbit check before
  any claim. Per the novelty directive this channel is a reproduction — change
  the question, do not write it up.
* **Gaia DR3 1268299311319369984** (RA 225.0080, Dec +26.8728) — the
  ASAS-SN-confirmed secular fader (0.073 mag/yr at 8.8σ, ~0.94 mag total,
  RUWE 0.98, `non_single_star=0`). **Killed 2026-07-01 by the NEOWISE
  counterpart test**: W1 fades at 0.0045 mag/yr (8.4σ, 20 seasons, 345
  epochs), W2 at 0.0041 mag/yr (3.8σ) — an IR/optical slope ratio of 0.062,
  precisely the standard small-grain extinction-law prediction
  (A_W1/A_optical ≈ 0.06). This is ordinary dust progressively obscuring the
  star, not a gray occulter (which would fade the IR at ≳30% of the optical
  rate) and not warm circumstellar dust (which would *brighten* W1/W2).
  Verdict `ir_fades_reddening_law`;
  `results/dimming/characterization.json`. Still an interesting *astrophysics*
  object (a decade-long monotonic obscuration event), but not a
  technosignature.

## Channel state

| Channel | Searched so far | Surviving | Blocking issue / next action |
|---|---|---|---|
| Dimming (dips + secular) | 250,862 ZTF stars, 116 fields | 0 — top fader killed by NEOWISE reddening test; **19 `marginal_fade` assessed and set aside** (all 1.6–7.4% total fades, 18/19 not band-confirmed → optical slope ~0.004–0.015 mag/yr, where even a gray occulter gives only ~2σ in NEOWISE; ZTF systematics floor) | channel exhausted at the systematics floor — do not re-chase the marginal faders; new volume only helps if it reaches ≳0.1 mag band-confirmed fades |
| Specular glint | code merged, **no run yet** | — | dispatch `dimming.yml` on completed fields — glint scan reuses the same ZTF pulls |
| Laser emission (SDSS-DR17) | 10,500+ spectra (latest committed run) | 112 triaged (was 118; 3 galaxies cut, incl. former #1) | per-exposure persistence check (repeat-visit path exhausted) |
| Astrometric dark companion (Gaia orbits) | 105,066 NSS orbits, ≤1 kpc | 0 novel (8 class-3 = BH1 + 7, but 7/8 already in Shahaf+2023; 1 borderline-absent is the weakest solution) | reproduction of the published AMRF catalogue — change the question |
| Laser absorption (DESI-DR1) | 6,500+ spectra (latest committed run) | 55 triaged | same; hot-star continua only (line-forest stars skipped by design) |
| WD IR excess | 7,716 clean WDs | 170 scored | variability (ZTF) + SED follow-up of top anomaly scores |
| Gaia XP anomalies | 1 cone (RA180/Dec30): 159 sources — **undersampled/unreliable** (70% flagged; colour locus needs ≫159) | — | reliability guard added (needs n≥1500, anomaly_frac≤0.15); dense-field run (RA283/Dec−3, r1.5°, ~80k) dispatched |

## Known systematics ledger (do not re-derive)

* SDSS/DESI wavelengths are **vacuum**; all literature line lists (air) are
  converted via `seti.spectra.reject.air_to_vacuum` at definition time. This
  was a real leak: pre-fix "candidates" sat on He I 5876 / Ca II 8542 / O I
  8446. Fixed 2026-07-01.
* Catalogue redshift/RV errors move known lines outside the in-funnel
  rejection window → the observed-frame ±300 km/s triage
  (`seti.spectra.triage`, costs 22.7% of the band, honestly accounted) is
  mandatory before believing any spectral candidate.
* **Misclassified emission-line galaxies** are the worst spectral leak: a
  background star-forming/active galaxy that SDSS/DESI labels `STAR` (or gives a
  wrong z) drops its rest-frame nebular family into the search as
  high-significance "unresolved" lines. The observed-frame known-line triage
  cannot catch it (it uses the wrong catalogue z). Decisive test = *internal
  redshift consistency*: if ≥2 surviving lines in one spectrum form a locked
  nebular pair (Hα+[N II], [O III] 4959/5007, [S II] 6716/6731) or ≥3 lines at a
  common z, it is a galaxy (`galaxy_reject`, verdict `galaxy_zmatch`). This killed
  the former #1 candidate. Single-line candidates cannot be tested this way —
  they need the per-exposure persistence check.
* Candidate wavelengths recurring across unrelated sightlines (≥3 spectra
  within ±3 Å, across runs *and* modes) are instrumental. 31 killed.
* Merged candidate CSVs can contain duplicate rows (runs overlap) — 89 killed.
* ZTF single-band events are artifacts until g/r-coincident
  (`multiband_coincidence`, `secular_achromatic`, `glint_achromatic`).
* Stellar flares are chromatic (g ≫ r); a glint must be achromatic.
* WD IR excess: dusty debris disks are the one natural confounder — subtract
  the labelled catalogues before scoring.
* A secular optical fade with a NEOWISE fade at ~6% of the optical rate is
  ordinary line-of-sight dust (extinction-law ratio) — check
  `w1_to_optical_slope_ratio` before getting excited. Gray occulters sit at
  ≳30%; warm dust *brightens* the IR.
* ASAS-SN (pyasassn) is flaky on runners — pass `--optical-slope` to
  `dimming-characterize` so the mid-IR verdict never returns
  `insufficient_ir` for want of a known number.

## Rules of engagement (from CLAUDE.md)

Novelty first, scale second, never write up a null. Merge every commit to
`main` as you go (non-fast-forward merge if diverged; never force-push).
Data-touching runs go through `workflow_dispatch`; the sandbox has no archive
egress.
