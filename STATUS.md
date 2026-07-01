# Research scoreboard

Live per-channel state of the search. Update this file whenever a run,
vet, or triage changes the candidate picture — it is the single place a
human (or a fresh agent session) looks to know what is hot and what to do
next. Last updated: 2026-07-01.

## Current best candidates (cross-channel, ranked)

1. **173 triaged laser-line priority targets** —
   `results/spectra_triage/priority_targets.csv`, led by an unexamined
   SDSS-DR17 star with a 31.9σ unresolved line at 7518 Å (2900 km/s from the
   nearest known line, outside every telluric band) *plus a second surviving
   line at 7542 Å in the same spectrum*. None cross-confirmed yet — the first
   `spectra-confirm` pass found zero overlapping repeat spectra for the
   pre-triage ranking; it now consumes this shortlist.
   *Next decisive test:* re-dispatch `spectra-confirm.yml` (top ≈ 60); for
   targets without repeat spectra, check the per-exposure (coadd-input) frames
   — a real line persists across exposures, a cosmic/artifact does not.
2. **WD IR-excess multimodal candidates** —
   `results/science/multimodal_candidates.csv` (170 anomaly-scored excesses
   from 7,716 clean 100-pc-scale white dwarfs). Not yet pushed through
   variability/full-SED discrimination.

## Resolved (killed) candidates

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
| Dimming (dips + secular) | 250,862 ZTF stars, 116 fields | 0 (top fader killed by NEOWISE reddening-law test); 19 `marginal_fade`, 12 `single_band_unconfirmed` in `results/dimming/vetting.csv` | run the NEOWISE counterpart test (now wired into characterize) on the 19 marginal faders — one `dimming-characterize.yml` dispatch each with `--optical-slope` |
| Specular glint | code merged, **no run yet** | — | dispatch `dimming.yml` on completed fields — glint scan reuses the same ZTF pulls |
| Laser emission (SDSS-DR17) | 10,500+ spectra (latest committed run) | 118 triaged | cross-confirmation (see above) |
| Laser absorption (DESI-DR1) | 6,500+ spectra (latest committed run) | 55 triaged | same; hot-star continua only (line-forest stars skipped by design) |
| WD IR excess | 7,716 clean WDs | 170 scored | variability (ZTF) + SED follow-up of top anomaly scores |
| Gaia XP anomalies | pipeline ready, **no run yet** | — | dispatch `xp.yml` on a first cone |

## Known systematics ledger (do not re-derive)

* SDSS/DESI wavelengths are **vacuum**; all literature line lists (air) are
  converted via `seti.spectra.reject.air_to_vacuum` at definition time. This
  was a real leak: pre-fix "candidates" sat on He I 5876 / Ca II 8542 / O I
  8446. Fixed 2026-07-01.
* Catalogue redshift/RV errors move known lines outside the in-funnel
  rejection window → the observed-frame ±300 km/s triage
  (`seti.spectra.triage`, costs 22.7% of the band, honestly accounted) is
  mandatory before believing any spectral candidate.
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
