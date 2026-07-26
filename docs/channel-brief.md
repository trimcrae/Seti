# Building a new search channel — conventions

Read this before writing code. It encodes what previous channels learned the
hard way. Deviating costs runs.

## 0. Non-negotiables

1. **The sandbox has no archive egress.** Gaia/VizieR/IRSA/MAST/SPARCL/ESO all
   fail with `CONNECT tunnel failed, response 403` locally. Every data-touching
   step must run on a GitHub Actions runner via `workflow_dispatch`. Build and
   test the funnel **offline against synthetic data**, then dispatch.
2. **Never fabricate data.** If an archive returns nothing, the channel emits an
   explicit verdict (`NO_DATA_REACHED`, `insufficient_ir`, …) and says so. A
   degraded run reports its degradation as a first-class field.
3. **Never write up a null result.** A clean null changes the question. Compute
   limits as internal honesty checks only.
4. **Trace every candidate to a systematic before believing it.**

## 1. Layout

```
src/seti/<channel>/__init__.py      package
src/seti/<channel>/acquire.py       archive pulls (runner-only, chunked, retried)
src/seti/<channel>/<detector>.py    the physics — pure functions, offline-testable
src/seti/<channel>/run.py           stage orchestration, writes results/<channel>/
tests/test_<channel>.py             offline pytest suite (CI gate)
.github/workflows/<channel>.yml     workflow_dispatch, commits results back
docs/<channel>.md                   claim, novelty status, method, contamination model
results/<channel>/                  committed outputs (small files only)
```

Wire a subcommand into `src/seti/cli.py` (`_cmd_<channel>` + `sub.add_parser`).
Config thresholds go in `config/`, not as magic numbers in code.

## 2. Archive access patterns that work on the runner

* **Gaia**: `astroquery.gaia`, `launch_job_async`, **chunked into parallax
  shells** — a single monolithic query at >10⁶ rows times out. Retry with
  exponential backoff; fall back to `launch_job` (sync) on the last attempt.
  See `src/seti/herdsman/acquire.py` for the reference implementation.
* **VizieR / TAP**: `pyvo`; chunk by sky region or ID block.
* **IRSA (WISE/NEOWISE/2MASS)**: TAP sync for catalogue rows; for per-object
  light curves, batch positional queries — and **propagate proper motion to the
  survey epoch** or high-PM stars silently return nothing. This bug cost a
  previous channel a whole run.
* **MAST**: `astroquery.mast`; note JWST TSO products are
  `dataproduct_type="timeseries"`, not `"spectrum"`, and `download_products`
  has a server-side bug — use `download_file(dataURI)`.
* **ESO**: TAP ObsCore has **no upload support** — pull a dec band in bulk and
  crossmatch locally with a KD-tree.

## 3. Workflow design

Copy `.github/workflows/herdsman.yml`. It encodes:

* **Checkpointing** — every completed unit writes its own JSON immediately;
  uploads use `if: always()`. A killed shard loses minutes, not hours.
* **Sharding** — heavy stages run as a `fail-fast: false` matrix; wall-clock is
  one shard, not the sum. Lost shards re-run individually.
* **Artifact passthrough** — acquisition happens once; the table travels as an
  artifact to later jobs.
* **`reduce_only_run_id`** — re-reduce a prior run's artifacts without refetching.
* **Commit-back** — `git add -f -- results/<channel>`, pull-rebase-autostash,
  push with one retry. Exclude large intermediates (`*.parquet` samples).
* `permissions: contents: write`, `timeout-minutes` generous on fetch stages.

## 4. Contamination ledger — inherited, do not re-derive

* A **single-band** anomaly is an artefact until confirmed in a second band.
* **AllWISE W4** is the shallowest, most confusion-limited band: for 22-µm-faint
  stars the catalogue flux is cirrus. A **W4-only excess is an artefact**. A
  real warm-dust SED lights up the star-dominated bands first.
* **Negative W1−W2 is a blend**, not a photosphere (bare photospheres have
  W1−W2 ≥ 0).
* **Faint (G ≳ 18) or blue (bp_rp small) WISE matches** are confusion-limited or
  have an invalid photosphere model.
* **Stellar flares are chromatic** (g ≫ r); anything claimed achromatic must be
  *measured* achromatic in two bands.
* **Grey vs chromatic fading**: a secular optical fade with a NEOWISE fade at
  ~6% of the optical rate is ordinary line-of-sight dust (the extinction-law
  ratio). A grey occulter sits at ≳30%. Warm dust *brightens* the IR.
* **Unresolved companions** masquerade as excesses: a fitted excess temperature
  >1800 K is hotter than grains survive → it is a companion photosphere.
* **Misclassified emission-line galaxies** are the worst spectral leak. Test
  internal redshift consistency: ≥2 lines forming a locked nebular pair
  (Hα+[N II], [O III] 4959/5007, [S II] 6716/6731) or ≥3 lines at one z → galaxy.
* **Air vs vacuum wavelengths**: SDSS/DESI are vacuum; literature line lists are
  air. Convert at definition time.
* Candidate wavelengths **recurring across unrelated sightlines** are instrumental.
* **Gaia XP is R≈30–100** — it cannot resolve a narrow line. Real features must
  be interior (≥8 samples from either end), bounded, and 2–5 samples wide.
* **Duplicate rows** across overlapping runs inflate candidate counts. Dedupe.

## 5. Offline test requirements

The pytest suite is the CI gate and must pass with **no network**. Every channel
ships tests that:

* **Recover an injected signal** — synthesise the signature at known strength
  and confirm the detector finds it.
* **Return a clean null on a confounded set** — synthesise the dominant
  astrophysical confounder and confirm the detector rejects it.
* **Degrade honestly** — simulate an empty/failed archive response and confirm
  the verdict field says so rather than emitting a candidate.
* Cover every rejection rule with a case that trips it.

## 6. Reporting

`results/<channel>/summary.json` carries the verdict, the counts at every funnel
stage, and the coverage (`clean_in_N_of_M_observed_channels` style). Update
`STATUS.md` with what was searched, what survived, and the next decisive action.
No overclaiming: state what the data can and cannot support.

## 7. Git

Develop on the session's designated branch, commit, push, then **merge to `main`
with `--no-ff`** and push `main`. A parallel agent commits here too — if `main`
has diverged, reconcile with a non-fast-forward merge. Never force-push `main`.
