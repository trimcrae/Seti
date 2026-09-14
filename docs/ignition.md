# IGNITION — an infrared excess being born on an old star

**Signature S61** (`docs/necrofrontier.md` §2 XIV, *Transition residue*; the
build-order's fourth item, §4). The time-reverse of EMBER.

---

## 1. The claim

A machine takeover's resource-acquisition phase is **exponential over
decades**: a replicating industry doubles its collecting area on a fixed
timescale until it saturates. In a decade-baseline mid-infrared archive that
is not a steady state — it is a **rising** signature. The observable is a star
whose W1 and W2 fluxes **rise monotonically for ≥ 5 yr** (2014–2024,
NEOWISE), on a host that is

* a **field star with Gaia astrometry** (parallax, proper motion — the kill for
  the one published example of the phenomenology, which is a galaxy),
* **kinematically old** (a tangential-velocity flag: nothing young is being
  claimed as a mature host),
* **photospheric in AllWISE 2010** (W1 − W2 < 0.15: the excess did not exist
  before the baseline), and
* **flat in the optical** (nothing that announces itself at 0.5 µm — novae,
  Miras, R CrB declines, YSO outbursts — is a candidate).

The natural background differs in **time signature, not amplitude**. Born
excesses on mature stars — giant impacts, comet swarms — rise in < 1 yr and
then decay or vary stochastically (HD 15407A: −14 % over 15 yr; ID8 and P1121:
years-long stochastic decay after a sub-year onset). Nothing natural on an old
field star rises smoothly for a decade. The best-published example of a
decade-long, optically flat W1/W2 rise is **NGC 6447** (+1.2 mag over 14 yr,
optical flat, SPHEREx warm dust → an obscured AGN turning on), which fixes the
dominant contaminant and its kill in the same sentence: parallax.

---

## 2. Novelty position (§5 row g12 of the necrofrontier sweep)

`necrofrontier-lit` runs 3 and 4 (2026-09-13; 2,135 abstracts, 1,928 unique,
decoy-aware) returned **six decoy-free hits for group g12** and none of them is
this search: Metzger et al. 2017 (secular *dimming* of KIC 8462852 — the
opposite sign, one star), a nearby-excess census (static excesses), and TWA
disks (young stars). The verdict recorded there is **Unoccupied**. The two
communities that each did half of it stayed inside their own populations: the
YSO community searches NEOWISE for outbursts among young stars; the AGN
community searches it for turn-ons among galaxies. Nobody has asked the
question of Gaia-astrometric, kinematically old, 2010-photospheric field
stars, and nobody has used the *shape* of the rise (sustained ramp versus
impulsive step-and-decay) as the discriminant.

What is adjacent and must not be confused with it: the unTimely
mid-IR variable catalogue (Kang et al. 2025, arXiv:2511.22071 — 8.3 M W1
variables with a smoothness statistic; **no machine-readable release yet**,
established by VIGIL's probe and its full-text scan) is a variability
catalogue, not a rise search; Silverberg et al. 2018 and the Hephaistos
searches are *static* excess selections. The framing of a rising excess as a
transition residue is the necrofrontier document's own.

---

## 3. Data

| Role | Source | Access | State |
|---|---|---|---|
| Parent sample | Gaia DR3 `gaia_source` × `allwise_best_neighbour` × `gaiadr1.allwise_original_valid` | ESA Gaia TAP, ADQL, in-archive join | reachable (every Gaia channel here) |
| The decade series | NEOWISE-R single exposures, `neowiser_p1bs_psd` (2013.9–2024.6) | IRSA TAP | **REACHED** (`results/necrofrontier/probe.json` `irsa_neowise_tap`; VIGIL's probe: 3,873 rows for one star, 21 visits) |
| Optical flatness | ZTF / ASAS-SN per-star CSV | hook: `--optical-dir <dir>/<source_id>.csv` | not wired to a service; recorded as `optical: not_checked` when absent |
| 2010 photosphere | AllWISE W1, W2, W3, `ph_qual`, `cc_flags`, `ext_flag` | inside the Gaia join | as above |

---

## 4. Method (`src/seti/ignition/`)

### 4.1 Sample (`sample.py`)
`G < 14.5`, `parallax > 3 mas`, `parallax_over_error > 10`, `|b| > 15°`,
`ruwe < 1.4`, `phot_variable_flag != 'VARIABLE'`, `0.6 < bp_rp < 2.5` with an
absolute-magnitude dwarf cut `M_G > 2.0 + 3.3 (bp_rp − 0.6)`
(`M_G = G + 5 log10 ϖ − 10`; ~1.5–3 mag above the main sequence, removes
giants and subgiants, keeps equal-mass binaries), and through the archive's
own AllWISE neighbour table `−0.1 < W1 − W2 < 0.15` (negative = blend,
channel-brief §4), `ext_flag = 0`, `cc_flags = '0000'`, A/B photometry in both
bands. `v_tan > 25 km/s` is a **flag**, not a cut. Two modes: `fields` (one
cone per configured high-ecliptic-latitude, high-`|b|` field — the mode the
proven NEOWISE route needs) and `allsky` (parallax shells with
`MOD(random_index, n_shards × stride) = shard`, stride set from a `COUNT(*)`
of each shell so the subsample is *uniform on the sky* and its fraction of the
parent is exact; a bare `TOP` returns a HEALPix-contiguous block). The
archive `COUNT(*)` of the full selection is the denominator in `sample.json`;
every verdict is a count over what was actually pulled.

**The query plan (fixed after run 34787803862).** That probe asked for
`SELECT TOP 5` inside a one-degree cone and was killed four times by
`Error 500 … canceling statement due to statement timeout`, then by
`Error 503 … maximum number of synchronous queued jobs (150) reached`. Five
rows cannot be a volume problem, so it was a **plan** problem and a **queue**
problem: the ADQL was one flat `WHERE` over the three-table join, which leaves
the planner free to start from the ~750-million-row AllWISE mirror and scan it
before the Gaia spatial index cuts the cone (`TOP 5` does not help — the hash
join builds its side first), and the `503` says the request went to the
*synchronous* endpoint, where the shared 150-job ceiling lives. Both are fixed:

* the cut an index can serve — the cone, the parallax shell, the
  `random_index` slice — plus **every** other Gaia-only cut now sits in an
  inner sub-select on `gaiadr3.gaia_source` alone (aliased `gs`), and the two
  AllWISE tables are joined to that small intermediate result;
* three **shapes** are tried in order and the one that answered is written to
  `probe.json` as `gaia_shape_working`, which the `sample` stage then uses
  instead of re-deriving it:

  | shape | inner sub-select | AllWISE colour/quality cuts |
  |---|---|---|
  | `inner_cone` | yes | in the outer SQL `WHERE` |
  | `inner_cone_postfilter` | yes | **pandas post-filter** (`select_parent` re-applies them to every frame anyway) |
  | `flat` | no | one flat `WHERE` — what timed out, kept last and still executable |

  The science cuts are identical in all three: `gaia_predicates()` and
  `allwise_predicates()` are the single source of both, and a shape only
  decides *where* they are written (`tests/test_ignition.py` asserts the
  predicate sets match);
* transport is a ladder — `astroquery` async, `pyvo` async, then the sync
  endpoints, which bulk queries never touch (`allow_sync=False`). Every
  attempt is time-boxed and the queue that served the query is recorded.

### 4.2 Acquisition (`acquire.py`)
Proper motion propagated to the mission mid-epoch (2019.0) and every match
radius widened by half the mission-long sweep (VIGIL's `propagate_pm`,
`pm_sweep_arcsec`). Three routes, chosen from `probe.json`:

1. **`upload`** — a `TAP_UPLOAD` positional join, 200 stars a query. The only
   route that scales to an all-sky subsample; **probed on two stars first,
   never assumed.**
2. **`field`** — VIGIL's proven path: one cone per field (`W1 < 13` bounds
   the rows; 377 s for 2.8 M rows at the NEP), exposures assigned to stars by
   a PM-propagated KD-tree (`seti.vigil.run.group_neowise_by_star`).
3. **`cone`** — one cone per star, *without* the `COUNT(*)` that doubled
   VIGIL's 92 s per star. The fallback.

Frames: `qual_frame > 0`, `saa_sep > 0`, `moon_masked = '00'`,
`cc_flags = '0000'`, `ph_qual` A/B in both bands; `nb > 1` / `na > 0` are
counted per star for the vet. Exposures are grouped into **epochs** by gaps
> 90 d (visits are ~183 d apart); per epoch the median and
`max(1.4826 MAD/√n, 0.005)` after a 5σ MAD clip; epochs with < 5 exposures
dropped. ~20 epochs over 2014–2024. Every batch is checkpointed (CSV append +
progress JSON) and a shard resumes from its progress file.

### 4.3 The detector (`rise.py`, pure)
Per band on the epoch series:

* **(a)** Kendall τ and p on the scan-cleaned series (sign flipped: + = brightening);
* **(b)** a weighted linear slope (mag/yr) and its significance, fitted
  jointly with a sinusoid whose period is confined to the **NEOWISE
  scan-direction band, 345–385 d**; the error is inflated by
  `√max(1, χ²_red)` so an underestimated epoch error cannot manufacture a
  significance; the sinusoid amplitude is reported;
* **(c)** the total rise, first-minus-last epoch on the scan-cleaned series, in σ;
* **(d)** the monotonicity fraction — consecutive epoch pairs that brighten;
* **(e)** the **shape test**: the best *sustained-ramp* model (linear, or
  exponential growth `a + b (e^{(t−t₀)/τ_g} − 1)` with τ_g on a grid — the
  claim *is* exponential, and the exponential is monotone by construction)
  against **step + exponential decay** (a brightening jump at an epoch,
  relaxing back with τ on 0.3–5 yr — the impact model), each with the same
  sinusoid; `ΔBIC = BIC(step+decay) − BIC(ramp)`.

Candidate per band: ≥ 10 epochs, baseline ≥ 5 yr, slope ≥ 5σ brightening,
τ p < 10⁻³, end-to-end rise ≥ 3σ, monotonicity ≥ 0.6, ΔBIC ≥ 6 (ramp
preferred), and the scan sinusoid (judged on whichever model it was fitted
alongside — a step leaks into the ramp model's sinusoid, and that leak is not
a scan systematic) < ⅓ of the rise. **Two-band rule**: W1 *and* W2 must each
pass (`ONE_BAND_ONLY` otherwise). The star-level label is the first failing
reason in a fixed priority so the veto counters partition the sample:
`INSUFFICIENT_EPOCHS`, `SHORT_BASELINE`, `FADING`, `SCAN_SYSTEMATIC`,
`IMPULSIVE_SHAPE` (step + decay *decisively* better, ΔBIC < −6),
`NOT_RISING`, `NOT_MONOTONIC`. A W2-minus-W1 rise colour is reported and a
rise greyer than the photosphere by > 2σ is flagged `grey_rise_suspect`
(dust or a radiator adds more to W2) — a flag, not a kill.

**Sensitivity** is measured on the sample itself: linear ramps of 0.1, 0.2
and 0.4 mag over 10 yr are injected on both bands of up to 200 screened,
non-rising stars per shard and the recovery fraction by the rise test is
reported in `summary.json["sensitivity"]`. At the survey noise of a
W1 ≈ 10 star (~0.008 mag per epoch) the synthetic recovery is complete at
0.2 mag and near-complete at 0.1 mag.

### 4.4 The contaminant ladder (`vet.py`, pure; the order is the kill order)

| Rule | Kills | How |
|---|---|---|
| `extragalactic` | obscured-AGN turn-on (NGC 6447), dust echoes | no significant parallax and no significant proper motion |
| `already_excess_2010` | anything with an old excess | AllWISE W1 − W2 > 0.15 (and `nearest_agn_like` flagged at > 0.8) — excluded upstream, re-checked |
| `star_forming_region` | YSO accretion outbursts | ten configured boxes (Taurus, Orion, Perseus, Oph, Upper Sco, Lupus, Cha, Serpens, CrA, Cepheus) |
| `galactic_plane` | YSOs, crowding, cirrus | `|b| < 15°` |
| `gaia_variable` | Miras, novae, R CrB, YSOs | Gaia DR3 `phot_variable_flag` |
| `saturated` | NEOWISE bright-source bias (a candidate factory for any trend search) | W1 < 8 or W2 < 7 |
| `deblended` | latent images, deblending artefacts | > 20 % of frames with `nb > 1` or `na > 0` |
| `poor_photometry` | marginal detections | > 50 % of frames cut on `ph_qual` |
| `rcrb_like` | R CrB dust puffs | optical **fades** (> 3σ) while the IR rises — the opposite sign |
| `optical_not_flat` | novae, AGB dust formation, YSOs | optical brightens > 3σ or rms > 0.05 mag |

The optical hook accepts a per-star CSV (`mjd|t_yr, mag[, magerr]`, ZTF or
ASAS-SN). When absent the star is carried as `optical: not_checked` and its
verdict is `clean_optical_untested`, never `clean` — an unchecked optical is
not a flat optical. Untested checks are named per star.

### 4.5 Stages and outputs (`run.py`)
`probe` → `probe.json` (`gaia_shapes`, `gaia_shape_working`,
`neowise_route_recommended`); `sample` →
`parent.parquet`, `sample.json`; `acquire` (shard `i/n`) →
`epochs_s{i}of{n}.csv`, `neowise_stars_s{i}of{n}.csv`, `acquire_s{i}of{n}.json`;
`screen` → `stars_s{i}of{n}.csv`, `screen_s{i}of{n}.json`; `assess` →
`stars_vetted.csv`, `candidates.csv`, `summary.json` (verdict, denominators,
per-stage counts, veto counters for screen and vet, aggregated sensitivity,
`shards.expected` vs `shards.found`). Verdicts: **`NO_DATA_REACHED`** (with
`reason`: which archive, and whether it failed or answered empty),
**`NO_IGNITION_CANDIDATE`**, **`IGNITION_CANDIDATES`**; a `DEGRADED (...)`
prefix names missing shards, failed sample units or failed NEOWISE queries.

---

## 5. What VIGIL's failure taught, and what is different here

VIGIL's `results/vigil/summary.json` reads `NO_DATA_REACHED, n_fields_searched: 0`
while eight `field_summary.json` files in the same directory read `SEARCHED`
with 186 stars scored each. The sweep worked; the **aggregator ran on a tree
in which the per-field artefacts were absent** and reported that absence as a
science verdict. Three things are different here:

1. `assess` records `shards.expected` against `shards.found`; zero found is
   `NO_DATA_REACHED` with reason `no_shard_outputs_found` (never a clean
   null), a shortfall is a `DEGRADED` prefix.
2. The workflow's assess job **lists what it downloaded and fails** when the
   acquire matrix ran but no shard output reached it, instead of writing a
   verdict over nothing; the reduce-only path does the same.
3. Per-star cones at ~92 s each (VIGIL's measurement) cannot reach a
   catalogue-scale sample; the field route (VIGIL's own fix, 186/187 stars
   from one query) is the default and the upload join is probed first.

And what this channel's own run 34787803862 taught — a **green** 25-minute
`--stage probe` that reached nothing and **committed nothing**:

4. **The probe commits.** Its evidence used to live only in an artifact and in
   the job log, because the commit-back belongs to the `assess` job and that
   job is skipped for `stage=probe`. The `sample` job now runs
   `scripts/commit_results.sh` (which verifies the push against
   `git ls-remote`) on `results/ignition/probe.json` itself, after the artifact
   upload, so a failed push cannot lose the evidence either.
5. **The probe is time-boxed.** It spent 1,490 s on four retries of a query
   that could not succeed. Each candidate shape now gets `probe.budget_s`
   (~8 min) and the whole stage `probe.total_budget_s`; a shape that overruns
   is recorded `TIMED_OUT` with its elapsed seconds and the probe moves to the
   next one. A route that fails keeps its **verbatim** error and the exact
   ADQL that was sent; `NO_DATA_REACHED` remains the honest verdict when
   nothing answers.

---

## 6. Honest limits

* **W1/W2 only.** NEOWISE probes hot material (Wien peaks 852 K and 630 K).
  A resource-acquisition phase radiating at 100–300 K is invisible here; the
  channel is sensitive to the *hot* end of the swarm, or to the photosphere-
  scale reprocessing it implies at 3–5 µm.
* **Two visits a year.** Anything periodic near 6 months or its harmonics is
  aliased; the scan-band sinusoid absorbs a periodic variable near 1 yr and
  the star is then labelled `SCAN_SYSTEMATIC`, which is honest but not
  specific.
* **The 2010 anchor is one epoch.** A star that ignited before 2010 and is
  still rising is excluded by the photospheric cut — deliberately, since the
  claim needs the excess to be born inside the baseline.
* **The optical is a hook, not a service.** Until per-star ZTF/ASAS-SN series
  are supplied, survivors are `clean_optical_untested`; nothing is called
  clean on an unchecked optical.
* **Pilot denominators.** `cap_per_shard` and `max_stars` make the first runs
  small; `sample.json` carries the archive `COUNT(*)` so the searched fraction
  is stated, and no verdict is a statement about the parent.
* **A null is a count.** `NO_IGNITION_CANDIDATE` is a count over
  `n_stars_screened`; no occurrence limit is computed or written up
  (CLAUDE.md). A clean null changes the question.

---

## 7. Status

Built 2026-09-13. Offline suite green (`tests/test_ignition.py`): an injected
linear ramp in both bands at the survey noise is recovered; the same ramp
under a 365 d sinusoid is recovered with the sinusoid reported; a step + decay
impact is rejected by the shape test (`IMPULSIVE_SHAPE`); stochastic
EDD-like variability, a Mira-like periodic series, a one-band rise, a fading
star, a short baseline and an empty archive each produce the named
non-candidate verdict; an accelerating (exponential) rise is recovered
through the exponential ramp. Dispatch via `.github/workflows/ignition.yml`
(`stage=probe` first). Run IDs and survivors go to `STATUS.md`.
