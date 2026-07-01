# Seti: a multi-channel search for a novel technosignature

The objective of this repository is a **detection**: one genuinely remarkable,
novel technosignature candidate in public astronomical data, found with
catalog-scale compute (GitHub Actions runners; no GPU, no multi-TB downloads).

Three standing priorities, in order (see `CLAUDE.md`):

1. **Novelty above all** — search signatures or populations *nobody has looked
   at before*; published searches (Dyson IR excess on main-sequence stars,
   optical laser emission, Boyajian-dimming refinements) are baselines to move
   beyond, not templates to repeat.
2. **Scale second** — once a novel angle is fixed, hit as many objects as the
   runners allow.
3. **A clean null changes the question, not the venue** — no
   null-result/occurrence-limit write-ups. Upper limits are computed as
   internal honesty checks, never as the deliverable.

## Search channels

Each channel is a `src/seti/` subpackage with an offline-tested funnel, a
GitHub Actions workflow for the data-touching runs, and committed results
under `results/`. Current per-channel state, surviving candidates, and next
decisive actions live in **[STATUS.md](STATUS.md)**.

| Channel | Novel angle | Package | Workflow |
|---|---|---|---|
| WD infrared excess | Dyson waste-heat around **white dwarfs** — excluded by Project Hephaistos II, only ever treated theoretically | `seti.{acquire,sed,contamination,discriminate,stats}` | `science.yml` |
| Deep dimming + secular fade | Boyajian-analogue + slow-enshrouding search in **ZTF**, at 25× the Kepler sample | `seti.dimming` | `dimming.yml`, `dimming-sweep.yml`, `dimming-vet.yml`, `dimming-characterize.yml` |
| Specular glint | Brief **achromatic brightening** (flat-mirror reflection returns the stellar spectrum; flares are blue) — no published search | `seti.dimming.glint` | `dimming.yml` (glint channel) |
| Narrow emission lines | Blind laser-line search in **SDSS-DR17 / DESI-DR1** spectra via SPARCL | `seti.spectra` | `spectra.yml`, `spectra-confirm.yml` |
| Narrow absorption lines | **Absorption-mode** analogue (engineered monochromatic absorber) — no published blind search | `seti.spectra.absorb` | `spectra.yml` (`mode: absorption`) |
| Gaia XP anomalies | Spectral shapes no normal-stellar model reconstructs, in ~220M BP/RP spectra | `seti.xp` | `xp.yml` |

## How work happens

The sandbox where code is developed has **no archive egress**; every funnel is
therefore built and unit-tested offline (synthetic spectra/light curves,
committed samples), and the data-touching runs are dispatched as
`workflow_dispatch` GitHub Actions which commit small result files back to the
branch. Contamination discipline is non-negotiable: every candidate is traced
to a systematic (sky lines, air/vacuum offsets, catalogue-RV errors, blends,
chromatic flares, dusty disks, cross-run recurrence...) before it is believed —
but the objective is the survivor, not the limit.

```bash
make install          # .venv with the package + dev deps
make test             # the full offline pytest suite (CI gate)
make analyze          # WD IR-excess funnel on the committed synthetic sample
python -m seti.cli --help   # all channels: *-run / *-vet / *-triage / *-confirm
```

## Layout

```
src/seti/        channel packages (acquire, sed, contamination, discriminate,
                 stats, dimming, spectra, xp, indicators)
config/          thresholds, catalogue IDs, paths (no magic numbers in code)
data/sample/     small committed synthetic sample for tests + CI
tests/           pytest suite validating every funnel stage offline
results/         committed candidate tables + summaries from runner searches
paper/           manuscript scaffolding (auto-generated numbers)
.github/workflows/  the workflow_dispatch searches
STATUS.md        live scoreboard: per-channel state + next decisive actions
```

## Status

Active search. See [STATUS.md](STATUS.md). License: Apache-2.0.
