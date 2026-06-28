# Seti-WD: an infrared-excess technosignature search around white dwarfs

A contamination-robust, fully reproducible search for **infrared-excess
technosignatures** (waste-heat / partial Dyson-sphere or -swarm signatures)
around **white dwarfs** — a stellar population that the flagship optical/IR
Dyson-sphere survey ([Project Hephaistos II](https://arxiv.org/abs/2405.02927),
Suazo, Zackrisson et al. 2024) deliberately excluded, and which prior
white-dwarf work has only ever treated *theoretically*
([Zackrisson et al. 2022](https://academic.oup.com/mnras/article/514/1/227/6575558);
[arXiv:2602.23270](https://arxiv.org/abs/2602.23270)).

## Why white dwarfs?

* **Clean SEDs.** A white-dwarf photosphere is a near-blackbody fixed by its
  Gaia parallax and `Teff`/`logg`, so an infrared *excess* is sharply defined —
  unlike the main-sequence stars where Hephaistos's candidates were challenged
  as [background contamination](https://arxiv.org/abs/2405.14921).
* **One natural confounder.** Essentially the only natural source of white-dwarf
  IR excess is a **dusty debris disk**, and those are already catalogued
  (Debes 2011; Dennihy/Xu 2020–21; Madurga Favieres 2024; Murillo-Ojeda 2026).
  We use them as a **labelled control population to subtract**.

The contribution is therefore a **contamination-robust pipeline** plus an
**occurrence-rate upper limit** — a result that is publishable whether or not
any anomaly survives (cf. the null [SETI-Ellipsoid TESS](https://arxiv.org/abs/2402.11037)
result, AJ 2024).

## The funnel

```
acquire ─▶ sed ─▶ contamination ─▶ discriminate ─▶ stats
```

1. **acquire** — Gaia EDR3 WD catalogue (Gentile Fusillo 2021, VizieR
   `J/MNRAS/508/3877`) × CatWISE2020 (`II/365`) / unWISE / AllWISE × 2MASS, plus
   the control debris-disk catalogues. All catalogue-scale (TAP / VizieR / CDS
   X-Match), memoised to parquet.
2. **sed** — predict the photospheric W1/W2 from a blackbody (self-contained)
   or Bergeron model atmospheres, then compute per-band excess significance
   `χ_W1, χ_W2` and a W1−W2 colour excess.
3. **contamination** — the core contribution: an auditable cut funnel
   (astrometry → WISE quality → crowding → **Gaia↔WISE co-movement** →
   extragalactic). The co-movement test — does the IR source move with the
   high-proper-motion white dwarf? — is the novel, decisive cut.
4. **discriminate** — fit `(T_dust, τ)` to each excess, subtract known debris
   disks, and flag sources **outside** the empirical dust locus (too cool, or
   swarm-like τ → unity) as ranked anomaly candidates.
5. **stats** — injection–recovery completeness `C(T_dust, τ)` and a
   Poisson/binomial occurrence-rate upper limit on white-dwarf Dyson swarms.

> **Honesty note.** W1/W2 photometry alone cannot uniquely separate a Dyson
> swarm from warm dust. A high anomaly score is a *follow-up target*, not a
> detection; warm dust outside the nominal locus remains the leading natural
> hypothesis until broken by variability (ZTF) or full-SED follow-up.

## Quick start (offline, no network)

```bash
make install          # create .venv and install the package + dev deps
make asset            # build the (synthetic stand-in) model-atmosphere asset
make analyze          # run the full funnel on the committed synthetic sample
make completeness     # injection–recovery completeness map
make figures          # render the manuscript figures
make test             # pytest — the CI reproducibility gate
```

`make analyze` prints the source counts at each funnel stage, the candidate
list, and the occurrence-rate upper limit. Everything runs on the committed
`data/sample/` so it works with no internet access — this is exactly what CI
exercises.

## Science run (real catalogues)

The offline sample is a *labelled synthetic stand-in* that validates the
pipeline (`seti/sample.py`). For a real search, the `seti.acquire.*` modules
pull the genuine catalogues into `data/cache/` (gitignored) and produce the
analysis-ready table the pipeline consumes. The `BergeronModel` reads a frozen
Montreal/Bergeron photometry table — replace the synthetic asset under
`src/seti/data_assets/` with the real table for science.

## Layout

```
src/seti/        pipeline package (acquire, sed, contamination, discriminate, stats)
config/          all thresholds, catalogue IDs, and paths (no magic numbers in code)
data/sample/     small committed synthetic sample for tests + CI (rest of data/ is gitignored)
tests/           pytest suite validating every funnel stage offline
paper/           manuscript (MNRAS style)
results/         figures + candidate tables (rebuilt by make)
scripts/         the model-atmosphere asset generator
```

## Status

Research code under active development; results on the committed sample are
synthetic and for validation only. License: Apache-2.0.
