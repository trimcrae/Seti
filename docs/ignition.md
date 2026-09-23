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
| Parent sample | Gaia DR3 `gaia_source` × `allwise_best_neighbour` × `gaiadr1.allwise_original_valid` | ESA Gaia TAP, ADQL, in-archive join | **DARK** as of `results/ignition/probe.json` 2026-09-14T00:22Z — all three query shapes `TIMED_OUT` (480/480/420 s), prior run `Error 500` statement timeout and `Error 503` 150-job ceiling |
| Parent sample, **second route** | Gaia DR3 `gaia_source` **alone** (the `gaia_only` shape) × AllWISE `allwise_p3as_psd` | ESA Gaia TAP for the Gaia half, **IRSA TAP** (`https://irsa.ipac.caltech.edu/TAP`) for the AllWISE half, matched positionally | the shape that touches none of the tables that timed out; the AllWISE table and column names are **asserted, unverified** — `verify_service` settles them against IRSA's own `TAP_SCHEMA` at runtime |
| Parent sample, **third route** | VizieR mirrors `I/355/gaiadr3` × `II/328/allwise` (`I/355/paramp` for parameters) | VizieR **non-TAP** ASU, `viz-bin/asu-tsv`, HTTP GET → TSV | the transport that worked the same night (it served ULINE's IRC+10216 table while TAPVizieR itself returned 503); catalogue ids **asserted, unverified** — the probe checks them |
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

### 4.1b A second route to the same parent sample (`irsa_route.py`)

The shape ladder above is a fix for a *plan* problem, and `probe.json` on
`main` (97c3f99) says the plan is no longer the problem: **all three joined
shapes timed out (480/480/420 s) and both queues were refused.** What those
three have in common is not their arrangement — that is the one thing that
differs between them — but the *table*: every one of them reaches
`gaiadr1.allwise_original_valid`, ESA's ~750-million-row AllWISE mirror,
through `gaiadr3.allwise_best_neighbour`. No rearrangement of the predicates
avoids it, so no fourth *arrangement* of the same join can help either. The
only shape that can help is one that does not touch it at all.

That is **`gaia_only`**, the fourth member of `SHAPES`: an indexed cone on
`gaiadr3.gaia_source` alone, carrying every cut `gaia_predicates()` gives
every other shape and **not one WISE column, join or table**. It is not a
parent sample by itself — it has no W1 and no W2 — so it is deliberately *not*
in `JOINED_SHAPES`, the ESA route never fills the parent from it, and
`stage_sample` will not adopt it from `probe.json` even if it is the shape that
answered. `irsa_route.py` supplies the other half, from the archive AllWISE
actually lives in: **IRSA's own TAP service**,
`https://irsa.ipac.caltech.edu/TAP` — the same service this channel already
uses for the NEOWISE decade series.

**Identical science.** The Gaia cuts stay in ESA's own SQL (this route does not
hand the Gaia side to a mirror at all). The AllWISE cuts are
`apply_allwise_predicates`, the VizieR route's pandas spelling of
`allwise_predicates`, reused unchanged. The cross-match is `match_allwise`,
also reused unchanged: Gaia 2016.0 PM-propagated to the AllWISE epoch 2010.5,
nearest neighbour inside `match_radius_arcsec = 3″`, `allwise_n_neighbours`
counting every source in that radius, an unmatched Gaia row dropped exactly as
the archive's inner join drops it. `to_parent_frame` emits exactly
`sample.parent_columns()` in order, so no stage downstream of `sample` can tell
the three routes apart — asserted column-for-column against the ESA frame in
`tests/test_ignition.py`.

**The AllWISE cone is fetched *unfiltered*.** The colour and flag predicates
are deliberately not sent to IRSA in the `WHERE`: cutting the AllWISE side
before the positional match would let a Gaia star whose true counterpart was
removed by colour match a *different* source inside 3″, and would make
`allwise_n_neighbours` — the blend indicator — count only the survivors of a
cut. The archive joins first and filters second; so does this. IRSA does
serve `COUNT(*)`, so every cone is counted as well as fetched and a short read
is flagged `truncated` against that count (the ASU route cannot do this: it has
a row cap and no count). A parallax shell is not a sky chunk, so `mode=allsky`
is `NOT_SUPPORTED_FOR_MODE` rather than given a fabricated match.

**Asserted and unverified, then settled at runtime.** The table
`allwise_p3as_psd` and every column spelling (`designation`, `ra`, `dec`,
`w1mpro`, `w1sigmpro`, `w2mpro`, `w2sigmpro`, `w3mpro`, `w3sigmpro`,
`cc_flags`, `ph_qual`, `ext_flg`) are assertions about a service the sandbox
cannot reach, and `config/ignition.yaml` → `sample.irsa` is marked exactly as
`sample.vizier` is. Nothing here assumes they are right: `verify_service()`
asks IRSA's own `TAP_SCHEMA.tables` and `TAP_SCHEMA.columns` (falling back to
`SELECT TOP 1 *`) **before a single science row is fetched**, once per run and
carried across cones. A table that is not in the service's own list is
`CATALOGUE_NOT_FOUND` *with the names it did return*; a column list that
cannot be resolved is `COLUMNS_UNRESOLVED` with the service's own text; a
column that carries a cut but is absent names the cut it could not apply. And
the distinction the record never loses: a `TAP_SCHEMA` that **did not answer**
is `QUERY_FAILED`, never `CATALOGUE_NOT_FOUND` — a service that did not answer
has told you nothing about the sky.

### 4.1c A third route to the same parent sample (`vizier_route.py`)

The shape ladder above is a fix for a *plan* problem. `probe.json`
(2026-09-14T00:22Z) says the problem is no longer a plan: **all three shapes
timed out and both queues were refused**, so the block is at the ESA archive
itself and nothing about the ADQL can get past it. A channel with no parent
sample has nothing downstream to run, so IGNITION has a **second source for
the same sample**: VizieR's **non-TAP ASU interface**
(`https://vizier.cds.unistra.fr/viz-bin/asu-tsv`), a plain HTTP GET returning
TSV — no TAP queue, no planner, no statement timeout. It is the route that
demonstrably worked that night, serving ULINE's IRC+10216 table while
TAPVizieR itself was returning 503.

**The ladder, per chunk of the sweep.** ESA first, always — it is
authoritative and owns the in-archive `allwise_best_neighbour` cross-match — and
every joined shape is tried before anything else happens. Then the IRSA route
of §4.1b, which keeps ESA's own Gaia cuts in SQL and gives up only the
cross-match. Only for a chunk neither of those answered does the VizieR route
run. `sample.json` records `routes`
(units and rows per source), `route_fractions`, `route_used` and a per-row
`parent_route` column; **a parent assembled from both routes is `DEGRADED`**
(e.g. `mixed_parent_routes:esa_gaia+irsa_tap`), which `assess` carries into
`summary.json`'s verdict. The `probe` stage tries and records **all three**
routes on every run — probing is not using — so the next dispatch reads
`parent_route_recommended` (`esa_gaia` > `irsa_tap` > `vizier_asu`) and
`parent_routes_tried` from `probe.json` instead of re-deriving them.

**The request.** `-source=<catalogue>`, `-out.max=<n>`, `-out=<column>` per
column, `-out.form=TSV`, `-out.add=` for computed columns, the chunk as
`-c=<ra>+<dec>&-c.rd=<deg>&-c.eq=J2000` (or `Plx=lo..hi` for a parallax
shell), and each expressible Gaia cut as a column constraint —
`Gmag=<14.5`, `Plx=>3`, `RPlx=>10`, `RUWE=<1.4`, `BP-RP=0.6..2.5`. The body is
parsed by the **shared** helper `seti.metronome.acquire.parse_asu_tsv` /
`asu_rows` (mirror walk, breaker, and the rule that an error page never reads
as zero rows), not by a second parser.

**Identical science, different transport.** Every server-side constraint is an
*optimisation*: `select_parent` re-applies all of them locally, exactly as it
does for the ESA route. `|b| > 15°` is computed from the mirror's own RA/Dec
(IAU 1958 pole) rather than requested, so it does not depend on an unverified
`-out.add=_Glat`. `apply_allwise_predicates` is the pandas spelling of
`allwise_predicates` — `−0.1 < W1−W2 < 0.15`, `ext_flag = 0`,
`cc_flags = '0000'` — and is applied to every chunk, including `cc_flags`,
which `select_parent` does *not* re-apply. (The TSV parser turns a clean `ccf`
column into the integer `0`, so the comparison is made on a zero-padded string;
a naive `== '0000'` would reject every clean source.) `to_parent_frame` emits
exactly `sample.parent_columns()` in order, so the two routes are
indistinguishable downstream — asserted on a synthetic mirror TSV in
`tests/test_ignition.py`.

**What is *not* identical, and is recorded as such.** There is no ASU
equivalent of `allwise_best_neighbour`, so this route matches positionally:
Gaia 2016.0 positions are PM-propagated to the AllWISE mean epoch (2010.5)
with the channel's own `propagate_pm`, nearest neighbour inside
`match_radius_arcsec = 3″` wins, `allwise_n_neighbours` counts everything
inside that radius, and an unmatched Gaia row is dropped as the archive's inner
join drops it. There is no ASU `COUNT(*)`, only `-out.max`: a chunk that comes
back at the cap is flagged `capped` and its row count is reported as **what was
obtained**, never as a complete selection. A parallax shell is not a sky chunk,
so `mode=allsky` is recorded `NOT_SUPPORTED_FOR_MODE` rather than given a
fabricated cross-match.

**Asserted and unverified** (the sandbox has no network; the probe checks them
live and writes the answer to `probe.json`): the catalogue ids
`I/355/gaiadr3`, `I/355/paramp`, `II/328/allwise`; every mirror column name
(`Source`, `RA_ICRS`, `Plx`, `RPlx`, `BP-RP`, `VarFlag`, `RandomI`; `AllWISE`,
`W1mag`, `ccf`, `qph`, `ex`); and the ASU parameter spellings above. A wrong id
is a recorded `CATALOGUE_NOT_FOUND` with the URL and VizieR's own error text, a
missing column a recorded `COLUMNS_UNRESOLVED` naming the cut that could not be
applied — a status, never a crash and never a fabricated row.

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
`probe` → `probe.json` (`gaia_shapes`, `gaia_shape_working`, `irsa_tap`,
`vizier_asu`, `parent_routes_tried`, `parent_route_recommended`,
`neowise_route_recommended`); `sample` →
`parent.parquet`, `sample.json` (with `routes`, `route_fractions`,
`route_endpoints` and `route_errors`); `acquire` (shard `i/n`) →
`epochs_s{i}of{n}.csv`, `neowise_stars_s{i}of{n}.csv`, `acquire_s{i}of{n}.json`;
`screen` → `stars_s{i}of{n}.csv`, `screen_s{i}of{n}.json`; `assess` →
`stars_vetted.csv`, `candidates.csv`, `summary.json` (verdict, denominators,
per-stage counts, veto counters for screen and vet, aggregated sensitivity,
`shards.expected` vs `shards.found`, and `endpoints` — every parent-route host
that was asked, the probe's VizieR record and the verbatim per-unit errors).
Verdicts: **`NO_DATA_REACHED`** (with `reason`: which archive, and whether it
failed or answered empty — and, with both parent routes dark, `endpoints` is
the whole of what the run learned), **`NO_IGNITION_CANDIDATE`**,
**`IGNITION_CANDIDATES`**; a `DEGRADED (...)` prefix names missing shards,
failed sample units, failed NEOWISE queries, a **mixed-route parent**
(`mixed_parent_routes:esa_gaia+vizier_asu`) and any chunk that hit the ASU
`-out.max` cap. The probe's own verdict gains `VIZIER_PARENT_AND_NEOWISE` /
`IRSA_PARENT_ONLY` / `VIZIER_PARENT_ONLY` for the case these routes exist to
cover: the ESA archive
dark, the parent reachable anyway.

---

## 4.6 The two things that stopped run 35653615329, and what each one was

That dispatch produced no shard output at all. Its two failures look like
different problems and are recorded separately, because each has its own fix.

**The upload ladder was never a transport problem.** The probe walked all four
rungs — pyvo's synchronous form, a raw `POST` with the parameters in the URL
(sync, then async), and IRSA's Gator multi-object search — and three of them
came back with the *same* sentence from IRSA's own TAP service:

```
INTERNAL_SERVER_ERROR: Unimplemented data type: unicodeChar
```

Four transports cannot fail identically on a transport fault. The complaint is
about a **column type in the uploaded table**, and the server only gets to make
it after it has parsed the request and read the upload — so those rungs were
working. `Table.from_pandas` on `source_id.astype(str)` produces a numpy `<U19`
column; astropy serialises that as VOTable `datatype="unicodeChar"`; IRSA does
not implement that type. A Gaia `source_id` is an integer by construction, so
`sid` now goes up as `long`, a non-numeric id falls back to ASCII `char`
(`datatype="char"`, which IRSA does implement), and `_ascii_string_columns`
converts any remaining unicode column on the way out so no future column can
reintroduce it. **The uploaded id cannot change the science**: rows are
assigned to stars locally, by exact unit-vector separation with per-star radii
(`group_by_star`), and never by the service's own join column — so the only
thing `sid` does is make the returned table readable.

The same probe showed the hand-rolled async rung getting `200` with **no
`Location` header**, so `_uws_job_url` now also reads the job document in the
body (`<uws:jobId>`, an `xlink:href`), and pyvo's own UWS client is a fifth
rung, `pyvo_async`. The two rungs IRSA demonstrably parsed lead the ladder.

**The parent sample was a wall clock, not an archive.** The same run's `sample`
step ran **2 h 22 min** over the same 20 one-degree cones without finishing —
ESA was slow that night; run 35039105536 had pulled the identical 846-star
parent in **719 s** — and the job was cancelled with the acquire matrix never
started. Two things follow:

* `sample_from_run_id` takes `sample.json` and `parent.parquet` from a prior
  run's `ignition-sample` artifact and goes straight to acquire. The reused
  artifact's `probe.json` is **dropped, not committed**: the branch's own probe
  record is the most recent live one, and a dispatch must not overwrite it with
  evidence it did not gather. (Measured: the `sample` job then takes 1 m 48 s.)
* In the all-sky sweep the same stall is paid for in *tiles never reached*, so
  `fetch_parent` takes `unit_budget_s` — what one unit may cost across the
  **whole** route ladder. Attempts begun after it is spent are recorded
  `SKIPPED_ON_UNIT_BUDGET` with the route and shape named, the unit is a
  recorded `QUERY_FAILED`, and `degraded` carries `unit_budget_skips:<n>/<u>`.
  The ladder's **order is untouched** — ESA is authoritative, is still asked
  first and still answers first — and every science cut is unchanged: this
  bounds only how long one tile may be chased, never what is selected.

## 4.7 The scale axis, and the one it is deliberately not

The pilot's 20 one-degree cones are **62.8 deg²**. The `tiles` sweep at
`dec_step_deg = 4` is **2,047 tiles covering 32,451 deg²** of the `|b| > 15°`
sky — a **517× increase in area**, at a measured parent density of ~13.5
stars/deg², so an all-sky parent of order **4 × 10⁵ stars** at the present
cuts. That is the number this channel can honestly reach, and it is smaller
than the build order's nominal 5 × 10⁶; the difference is stated rather than
closed, because the two ways to close it both cost the detection:

* **`|b| > 10°` instead of `> 15°`** would add sky, but `vet.py`'s
  `galactic_plane` rule kills `|b| < 15°` as a YSO/crowding/cirrus contaminant.
  Sampling stars the contaminant ladder is built to reject is work that cannot
  produce a survivor, so the sweep keeps `|b| > 15°`.
* **`G < 15` instead of `< 14.5`** would add ~35 % more stars, at W1 ≈ 13 where
  the NEOWISE per-epoch scatter is several times the ~0.008 mag of a W1 ≈ 10
  star. The measured sensitivity (`summary.json["sensitivity"]`) is complete at
  a 0.2 mag/decade ramp and near-complete at 0.1 mag *at that brightness*; the
  stars a fainter cut would add cannot carry the signal the channel is looking
  for. Scale here comes from **area**, not from depth.

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

### 7.1 In flight, 2026-09-22 — and the next decisive action

Both dispatches have since run. **35738088082 is complete and its verdict is
committed** — read §7.3, which supersedes items 1–3 of the read order below.
**35740159635 is in flight**: its probe landed at 15:42 UTC (11:42 EDT) and
its 12-shard sweep matrix began at 15:48 UTC (11:48 EDT) under a 150-minute
per-shard wall clock, so the shards commit their own `sweep_s*of12` /
`screen_s*of12` records from about 18:20 UTC (14:20 EDT), each shard as it
finishes. Item 4 of the read order is the live one.

| run | what it settles | inputs |
|---|---|---|
| **35738088082** ✅ | Step 1: do the 846 parents get full 10-year series now? **Yes — 846/846, see §7.3** | `stage=all mode=fields shards=8 max_parallel=8 route=upload sample_from_run_id=35039105536` |
| **35740159635** ✅ | Step 2: how much of the `\|b\| > 15°` sky one dispatch covers — **650/2,047 tiles (0.3175 of the area), 169,749 stars, 9 candidates; see §7.4** | `stage=all mode=tiles shards=12 max_parallel=12 budget_min=150 route=upload` |

`route=upload` with `upload_fallback_cone: true` is deliberate: it tests the
`unicodeChar` fix on the real service, and a chunk no rung answers still goes
to per-star cones, so the run cannot come back empty because of the route.

**What to read first, in order:**

1. The acquire ledger of any shard — `results/ignition/acquire_s*.json`,
   `ledger[].label`. `neowise_upload[pyvo_sync]_<n>` or
   `neowise_upload[pyvo_async]_<n>` means the fix landed and the channel scales;
   `neowise_upload[none]_<n>` with `Unimplemented data type` still in `error`
   means the service refuses `long` as well and the run will show the one
   recorded downgrade to a 32-bit row index (`upload[.../int32_index]` in the
   job log). Anything else there is a new failure and is quoted verbatim.
2. `summary_fields.json` → `denominators.n_stars_attempted_neowise` /
   `n_stars_with_neowise_rows` / `n_stars_screened` against the 846, and
   `degraded` for `neowise_queries_failed:<n>` (314 last time).
3. `veto_counters.screen`. The previous run's 172 screened stars were
   `FADING:97, NOT_RISING:37, IMPULSIVE_SHAPE:12, INSUFFICIENT_EPOCHS:26` —
   97 five-sigma faders in both bands is the survey's zero point, and
   `ensemble.per_shard_drift` in the new summary says how much of it the
   ensemble correction removed. A `FADING` count that stays near half the
   sample means the correction did not take and the *screen input* is still
   wrong, not the sky.
4. `summary_tiles.json` → `coverage.tiles_done / tiles_in_sky`,
   `sky_fraction_done`, `n_parent_done_tiles`, `shards_stopped_on_budget`, and
   `degraded` for `unit_budget_skips:<n>/<units>`. That is the honest coverage
   statement for the sweep; a later dispatch continues it with
   `resume_run_id=35740159635` **at the same shard count (12)**.

### 7.2 The offline gate is only a gate at the runner's library versions

`docs/channel-brief.md` §0 item 5: the sandbox venv and the runner do not hold
the same pandas. A sibling channel lost a whole dispatch to `pandas.to_numeric
(errors="ignore")`, removed in pandas 3, two minutes into a runner job and
before a single archive call. IGNITION was audited against that failure mode on
2026-09-22:

* the channel's only `to_numeric` calls (`run.py` 824, 873/874) pass
  `errors="coerce"`, which pandas 3 keeps; there is no `errors="ignore"`,
  `applymap`, `iteritems`, `fillna(method=)`, `delim_whitespace`, `.mad()`,
  `get_values` or `append`-on-a-DataFrame anywhere under `src/seti/ignition/`
  (every `.append(` there is a plain Python list);
* the whole offline suite (`tests/test_ignition.py`,
  `tests/test_ignition_scale.py`, 102 tests) was re-run in a throwaway venv
  pinned to the runner's exact stack — pandas 3.0.6, numpy 2.4.6, astropy
  8.0.1 — and is green. The sandbox default (pandas 2.3.3) is *not* the gate;
  this is.

Independently confirmed on the metal: run 35738088082's eight acquire+screen
shards each installed pandas 3.0.6 on the runner and completed, so the acquire,
ensemble-correction and rise-test paths are proven at that version against real
archive data, not only in tests.

### 7.3 Run 35738088082 — the upload fix on the real service, and 846/846

Dispatched `stage=all mode=fields shards=8 route=upload
sample_from_run_id=35039105536`. The parent was the 846-star Gaia DR3 x AllWISE
sample already pulled by run 35039105536; reusing it took 1 s in the `sample`
job instead of the 2 h 22 min that killed the previous dispatch.

**The upload question is settled.** Every acquire shard logged the rung it
used:

```
[ignition] upload[pyvo_sync/long] 83 stars -> 253114 rows in 100 s   (shard 0)
[ignition] upload[pyvo_sync/long] 80 stars -> 281058 rows in  93 s   (shard 2)
[ignition] acquire s0of8: OK ok=106 zero=0 failed=0 route=upload
[ignition] acquire s2of8: OK ok=106 zero=0 failed=0 route=upload
```

`pyvo_sync/long` is the top rung of the ladder with `sid` serialised as VOTable
`long`. IRSA's TAP accepted it. No shard fell through to the `char` rung, none
took the one recorded 32-bit downgrade, and `Unimplemented data type:
unicodeChar` does not appear anywhere in the run. The same conclusion comes
independently from run 35740159635's probe, which walked the ladder live:

```
neowise_upload.label   = neowise_upload[pyvo_sync]_2   status OK   7919 rows
neowise_upload_transport = pyvo_sync
neowise_route_recommended = upload        (it had never been recommended before)
verdict = ALL_ROUTES_REACHABLE
```

**And the 314 lost stars are back.** Denominators, against the 846:

| | run 35039105536 | run 35738088082 |
|---|---|---|
| parents attempted | 336 of 846 | **846 of 846** |
| stars with NEOWISE rows | 172 | **846** |
| stars screened | 172 | **846** |
| acquire failures | 314 | **0** |

Per-shard: 106,106,106,106,106,106,105,105 screened = 846, with `ok=N zero=0
failed=0` on every shard. A shard's whole star list now comes back in two to
three minutes of wall clock.

**The ensemble zero-point correction has now been run on real data — and the
answer is split.** This is the claim that most needed testing, so it is worth
separating what the run establishes from what it does not.

*What it establishes.* The correction measures a real instrumental term, and
the evidence is that the eight shards agree on it. The shards are eight
disjoint sets of ~106 stars; each computes its own median offset per 0.25-yr
bin independently, 44 bins over a 10.75-yr baseline, all 44 applied:

| band | mean pairwise correlation between the 8 shards' curves | shard-to-shard scatter per bin (median) | ensemble-mean first → last |
|---|---|---|---|
| W1 | **+0.970** (min +0.952, max +0.983) | 0.0004 mag | **+0.0190 mag** |
| W2 | **+0.989** (min +0.980, max +0.994) | 0.0009 mag | **+0.0563 mag** |

Eight independent star samples reproducing the same 44-point curve to within
0.4–0.9 mmag, against a 19–59 mmag excursion, is not a stellar signal. NEOWISE
drifts faint by ~0.019 mag in W1 and ~0.056 mag in W2 over the decade, the
correction sees it, and it subtracts it. That had never been shown on sky data
before this run.

*What it does not establish.* It is **not** what removed the 97 faders. The
summary carries the uncorrected control: `raw_two_band_fading_5sigma`, the
count of stars whose **uncorrected** bare linear slope is >= 5 sigma fading in
both bands, is **18** over the 846, against **17** after the correction. The
correction moved one star. So the fall from `FADING` 97 of 172 (56.4%) to 17
of 846 (2.0%) is a change of **sample**, not of method, and the earlier
diagnosis — that those 97 were the survey zero point — is *not* confirmed
here.

The likely reason is that the earlier 172 were not a random 172 of the 846:
they were the remnant the broken `field` route happened to deliver, which is
the ecliptic-pole end of the sample, where NEOWISE stacks far more epochs per
star. A fixed 0.056 mag decade-long drift is worth ~sqrt(N_epochs) in slope
significance, so the same instrumental term that is harmless at ~100 epochs
becomes a 5-sigma two-band "fade" at ~1000. That is a mechanism, not a
measurement, and this run does not test it: the decisive test is to re-screen
the high-epoch-count tail with and without `--no-ensemble` and compare
`raw_two_band_fading_5sigma` to `FADING` there, where the two should finally
diverge. Until then the correction is retained because it demonstrably removes
a real term at negligible cost, not because it has been shown to rescue any
star.

`raw_two_band_rising_5sigma` is **0** on every shard, uncorrected — so no
candidate was created *or* destroyed by the correction either.

**The funnel.** `results/ignition/summary.json`, committed by the run itself at
16:15:33 UTC, carries `verdict: NO_IGNITION_CANDIDATE`, `degraded: []` (nothing
was degraded) and `shards: {expected 8, found 8, missing []}`:

| verdict | run 35039105536 (of 172) | run 35738088082 (of 846) |
|---|---|---|
| `NOT_RISING` | 37 | 780 |
| `IMPULSIVE_SHAPE` | 12 | 25 |
| `INSUFFICIENT_EPOCHS` | 26 | 23 |
| `FADING` | 97 | 17 |
| `SCAN_SYSTEMATIC` | — | 1 |
| **rise candidates** | 0 | **0** |

The five veto counts sum to 846 exactly, so no star is unaccounted for. The
two columns are *not* like for like: 172 is the biased remnant of a broken
route, 846 is the whole sample.

Injection sensitivity, measured on this run's own 823 screened non-rising
stars by injecting linear ramps on both bands over 10 yr and asking how many
the rise test recovers: **83.8%** at 0.10 mag, **95.7%** at 0.20 mag, **99.4%**
at 0.40 mag. So a 0.2 mag decade-long born excess would have been found in 19
of every 20 stars carrying one, and none was.

**What the run did not find.** Zero rise candidates in 846 stars, with the
sensitivity above. That is a clean result on a channel that now demonstrably
works end to end, and per `CLAUDE.md` it is a reason to widen the question, not
a deliverable: 846 stars is a pilot, and the `|b| > 15` tiles sweep — 2,047
tiles, 32,451 deg2, 517x this area — is the scale axis that follows.

### 7.4 Tiles run 35740159635 — nine candidates, the re-vet, and what is left (2026-09-23)

**What the files on the branch describe.** `summary.json` (generated
2026-09-22T22:47:17Z, committed 22:47:18Z by the run itself) is the **tiles**
sweep: `sample_mode: tiles`, 12/12 shards found, verdict `DEGRADED
(neowise_queries_failed:3); IGNITION_CANDIDATES`. It is self-consistent: the
eight screen verdicts sum to 169,749 = `n_stars_screened`; the twelve
`screen_s*of12.json` sum to 169,749 screened and 9 rise candidates;
`candidates.csv` and `stars_vetted.csv` hold the same 9 rows. It **replaced**
the pilot's `summary.json` (run 35738088082, 846 stars, 0 candidates, 16:15Z);
that record now survives only in git history and in the `*_s*of8.json` files,
which describe the pilot, not the sweep. There is no `summary_<mode>.json`.
`sample.json` (2026-09-16, `mode: fields`) is a stale fields-mode record and
says nothing about the tiles sweep; `probe.json` is the sweep's own probe
(15:42Z) until the resume below replaced it.

**Coverage — incomplete.** 650 of 2,047 tiles done (10,303 of 32,451 deg²,
sky fraction 0.3175), 12 tiles partial, 0 failed, 1,385 never started; **every
shard stopped on its 150-min budget**. Parents 168,203 in done tiles; 171,483
attempted (partial tiles included), 169,788 with NEOWISE rows, 3 failed
queries (all shard 9). Shards 3 and 9 covered only 18 and 28 tiles.

**Resume — first attempt broke, fixed.** Run 35859572295 (`resume_run_id=
35740159635`, 12 shards) died in every shard two minutes in:
`ArrowTypeError: Expected bytes, got a 'int' object ... column source_id`. A
resumed `parent_s*.parquet` returns str ids, a new tile's are int64, and the
concatenation could not be written. Fixed in `stage_sweep` (commit 3f7d4396,
test `test_sweep_resume_that_adds_a_tile_writes_the_parent`, which fails
without the fix). Shards 0–8 of 35859572295 had already failed and only
re-screened their old data; shards 9–11 checked out the fixed code and are
sweeping. **Next action for the coordinator:** when 35859572295 completes,
dispatch `stage=all mode=tiles shards=12 max_parallel=12
resume_run_id=35859572295` (its shard artifacts carry all prior progress).

**The nine candidates.** All nine are grey, small (0.026–0.083 mag over
2014–2024), W1 9.4–11.2, at |β| 47–75°, with 2010 W1−W2 within ±0.03.

**Re-vet (`src/seti/ignition/revet.py`; runs 35860165284, 35861792980).**

1. *Brightness-dependent drift — real, but not the explanation.* Over all
   169,749 screened stars the NEOWISE drift depends on brightness: W1
   0.24 → 1.45 mmag/yr and W2 1.7 → 3.5 mmag/yr from W1 8–9 to 11.5–12. The
   per-shard median correction therefore over-corrects bright stars, but by
   only ~0.2–0.5 mmag/yr at W1 9.4–10.3, against candidate slopes of 2–11
   mmag/yr. The candidates' **raw** slopes are 6–29 MAD outliers of stars of
   their own W1 (±0.25) and |β| band (1,756–7,786 peers each). Re-screening
   with a (W1-bin × |β|-band) stratified ensemble pooled over all shards:
   5 of 9 survive, 5 new stars pass (10 total from 497 fully re-tested).
2. *Dust colour.* For small changes W2rise/W1rise equals the ratio of
   fractional excesses; warm dust on these photospheres predicts ~2.0 (1000 K)
   or ~1.5 (1500 K, sublimation). Every one of the 14 stars has a ratio of
   0.60–1.18, excluding 1000 K dust at 3.6–16σ and 1500 K at 1.4–9σ. **No
   rise has the colour of a born warm excess**; they have the colour of a star.

| source_id | stratified | W1 / W2 slope (mmag/yr, σ) | W2/W1 | fate and mechanism |
|---|---|---|---|---|
| 4572744241447044352 (LP 387-28) | pass | −5.0 (10.4) / −4.5 (5.9) | 0.90±0.18 | **Killed — proper-motion blend.** PM 212 mas/yr; Gaia neighbour G=15.34 (ΔG=1.95) closes from 7.5″ (2010) → 6.7″ (2014) → 4.4″ (2024) into the 6″ beam; AllWISE resolves it at 6.1″ (W1 13.15). ZTF r/i flat (+0.5/+0.3 mmag/yr) at 1.4″, as a blend predicts. |
| 2698668556422137728 (LP 578-16, new) | pass | −3.9 (6.3) / −3.3 (5.6) | 0.84±0.20 | **Killed — proper-motion blend.** PM 209 mas/yr; an equal-brightness neighbour (ΔG=−0.05) closes 10.4″ → 9.6″ → 7.3″. ZTF g/r flat. |
| 5747379623232417792 (new) | pass | −5.9 (6.9) / −7.0 (6.4) | 1.18±0.25 | **Killed — optical not flat.** ZTF g −27.8 (31σ), r −21.8 (32σ) mmag/yr: the star brightens 3–4× more in the optical than in the IR — stellar variability, not an excess. |
| 1421075180688797952 (new) | pass | −4.5 (12.2) / −3.4 (7.4) | 0.77±0.12 | **Killed — optical not flat.** ZTF r −19.0 (13σ), g −39.6 (9σ) mmag/yr. |
| 4593063967944867968 | fail (W2 scan) | −5.6 / −5.0 | 0.90 | **Killed** — fails the stratified screen (`ONE_BAND_ONLY`, W2 `scan_systematic`); ZTF g brightening but saturated (g 12.9). |
| 4619030382440742144 | fail (W1 not monotonic) | −5.9 / −4.9 | 0.82 | **Killed** — fails stratified; VSX ASASSN-V ROT, P = 1.13 d (spotted rapid rotator). |
| 4614613575512315648 | fail | −2.9 / −2.7 | 0.91 | **Killed** — fails stratified; VSX ASASSN-V ROT, P = 0.84 d. |
| 6383068554366896000 | fail | −2.9 (4.6) / −2.9 (4.3) | 1.01 | **Killed** — fails stratified (its W2 rise was 0.16σ raw: made by the global correction); also an unresolved common-PM pair (G=13.21 at 1.44″). |
| 4787733051398426240 | pass | −10.9 (10.2) / −7.1 (10.1) | 0.65±0.09 | **Out of scope — active young star.** VSX ASASSN-V ROT P = 3.68 d, 0.16 mag; FLAME age 1.4 (0.2–4.7) Gyr. A spot/activity-cycle brightening is the mundane reading; the host is not old. |
| 6371268909112212864 (new) | pass | −3.7 (8.1) / −3.5 (5.7) | 0.95±0.20 | **Out of scope — active star.** VSX ASASSN-V ROT P = 1.14 d. |
| 5498602919742073088 (new) | pass | −5.0 (8.4) / −3.0 (5.9) | 0.60±0.13 | **Out of scope — active star.** VSX ASASSN-V ROT P = 5.53 d (M0, Teff 4070 K). |
| **5802068055991339904** | pass | −6.9 (12.0) / −7.3 (10.8) | 1.06±0.13 | **Open.** No VSX/SIMBAD/Milliquas/Gaia-vari/QSO; static neighbour G=16.8 at 4.2″ (1.5 % of the optical flux; would have to brighten ×3.5 in W1 and W2 to do this). FLAME 13.5 Gyr, GSP-Phot [M/H] −1.19. Optical untested (δ = −75.7°: no ZTF; ASAS-SN unreachable). |
| **4867872438154507904** | pass | −5.9 (11.0) / −4.2 (7.6) | 0.72±0.12 | **Open.** No neighbour within 24″; no catalogue entry. GSP-Phot [M/H] −1.25; v_tan 10.6 km/s (not kinematically old). Optical untested. |
| **4646037549114081792** (TYC 9155-878-1) | pass | −3.3 (8.6) / −3.0 (6.6) | 0.91±0.18 | **Open.** No neighbour within 22″; SIMBAD PM* only. GSP-Phot [M/H] −1.11. Optical untested. |

(Slopes are from the stratified re-screen's ramp fit; negative = brightening.
None of the 14 is saturated: W1 ≥ 9.41, W2 ≥ 9.44. Milliquas and Gaia
`qso_candidates` returned nothing for any of them; Gaia `vari_summary`
returned nothing — none is a DR3-classified variable.)

**What is left, stated plainly.** Three stars — 5802068055991339904,
4867872438154507904, 4646037549114081792 — brighten by 3–7 % in W1 and W2
over 2014–2024, significantly, monotonically, against peers of their own
brightness and cadence, with no neighbour, catalogued variability or AGN to
blame. **They are not an infrared excess being born**: the rise is grey,
excluding ≤ 1500 K dust at 3.4–6.7σ each, which is what the star itself
brightening (or a source-specific NEOWISE systematic) looks like. The
decisive missing measurement is a decade optical light curve; ZTF cannot see
them, ASAS-SN Sky Patrol (`asassn-lb01.ifa.hawaii.edu:9006`) timed out on
connect from the runner for every target, and Gaia DR3 publishes epoch
photometry only for its variables. The Gaia DR3 per-observation scatter proxy
against G/BP-RP peers is the one optical constraint in reach; its result is
in `results/ignition/revet.json` → `gaia_variability_proxy`. Three of three
survivors being metal-poor by GSP-Phot is noted, not interpreted (GSP-Phot
[M/H] is biased low for many dwarfs).

Nothing here is a technosignature candidate. The screen as built selects a
grey stellar-brightening population at the 10⁻⁵ level plus proper-motion
blends; the next version must (a) apply the stratified ensemble, (b) require
the rise colour to be dust-like (the table's W2/W1 test, now in
`revet.dust_colour_test`), and (c) kill any Gaia neighbour closing inside 9″.
Under (b) alone, all nine original and all five new candidates fail.
