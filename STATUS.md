# Research scoreboard

Live per-channel state of the search. Update this file whenever a run,
vet, or triage changes the candidate picture — it is the single place a
human (or a fresh agent session) looks to know what is hot and what to do
next. Last updated: 2026-07-01.

## Current best candidates (cross-channel, ranked)

1. **Gaia DR3 1268299311319369984** (RA 225.0080, Dec +26.8728, G=16.05,
   d≈535 pc) — **secular fader**, the strongest surviving object in any
   channel. ZTF secular fade confirmed independently by ASAS-SN
   (0.073 mag/yr at 8.8σ over 1292 epochs, ~0.94 mag total), Gaia flags it
   VARIABLE, `non_single_star=0`, RUWE 0.98 (no binary excuse).
   `results/dimming/characterization.json`.
   *Next decisive test:* mid-IR — a slow dust-enshrouding event must brighten
   in NEOWISE W1/W2 as the optical fades. **Implemented**
   (`fetch_neowise` + `ir_counterpart_verdict` in
   `seti.dimming.characterize`): dispatch `dimming-characterize.yml` with
   `ra=225.0080427 dec=26.8728398`. Verdicts: `ir_brightens_dusty` (mundane),
   `ir_flat_chromatic_fade` (reddening), `ir_fades_gray_occulter` (gray
   occulter, no thermal signature — the regime that resists every dust
   explanation).
2. **173 triaged laser-line priority targets** —
   `results/spectra_triage/priority_targets.csv`, led by an unexamined
   SDSS-DR17 star with a 31.9σ unresolved line at 7518 Å (2900 km/s from the
   nearest known line, outside every telluric band) *plus a second surviving
   line at 7542 Å in the same spectrum*. None cross-confirmed yet — the first
   `spectra-confirm` pass found zero overlapping repeat spectra for the
   pre-triage ranking; it now consumes this shortlist.
   *Next decisive test:* re-dispatch `spectra-confirm.yml` (top ≈ 60); for
   targets without repeat spectra, check the per-exposure (coadd-input) frames
   — a real line persists across exposures, a cosmic/artifact does not.
3. **WD IR-excess multimodal candidates** —
   `results/science/multimodal_candidates.csv` (170 anomaly-scored excesses
   from 7,716 clean 100-pc-scale white dwarfs). Not yet pushed through
   variability/full-SED discrimination.

## Channel state

| Channel | Searched so far | Surviving | Blocking issue / next action |
|---|---|---|---|
| Dimming (dips + secular) | 250,862 ZTF stars, 116 fields | 1 (the fader above); 19 `marginal_fade`, 12 `single_band_unconfirmed` in `results/dimming/vetting.csv` | dispatch `dimming-characterize.yml` on the fader (NEOWISE test is wired in); re-vet marginals with the g/r achromatic check |
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

## Rules of engagement (from CLAUDE.md)

Novelty first, scale second, never write up a null. Merge every commit to
`main` as you go (non-fast-forward merge if diverged; never force-push).
Data-touching runs go through `workflow_dispatch`; the sandbox has no archive
egress.
