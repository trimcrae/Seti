# ARC — superflares above the starspot energy ceiling

**Signature S59** (`docs/necrofrontier.md` §2, "The arc"; §5 row g13):
*a flare that carries more energy than the star's spots can store.*

---

## 1. The claim

A flare is a release of magnetic free energy stored in the field above a
starspot group.  For a spot of area `A_spot` threaded by a field `B`, the
energy available is bounded by

    E_mag = f · (B² / 8π) · A_spot^{3/2}        (cgs; f ≤ 1)

(Shibayama et al. 2013, ApJS 209, 5, eq. 1; Okamoto et al. 2021, ApJ 906, 72,
§5).  With `B = 3 kG` — the strongest field seen in sunspot umbrae — Okamoto
et al. find the **upper envelope** of Kepler superflare energies as a function
of spot area consistent with `f ≈ 0.1`.  Nothing in the natural sample sits
above `f = 1`.

The spot area itself is not observed; it is inferred from the star's
rotational modulation amplitude `ΔF/F` (Notsu et al. 2013, 2019).  So the
bound is a *prediction* that every catalogued flare on a star with a measured
amplitude must satisfy.  A flare that **exceeds** the `f = 1` bound is one of
three things: a measurement problem (the energy is overestimated, or the
amplitude underestimates the spots), an unresolved second star in the
aperture (the flare is not this star's), or not a flare at all.  Deliberate
stellar manipulation — a weapon, or engineering that dumps energy into a
star — would show as **repeated, target-localised superflares above the bound
on a spot-free slow rotator**.

The channel's deliverable is the vetted `ξ > 0` list.  By construction it is
the contamination set of Notsu et al. (2019): the stars whose "superflares"
turned out to belong to a hidden companion are exactly the stars this test
flags.  That is why the stage-2 pixel-centroid test (§6) is decisive and why
nothing here is claimed before it.

## 2. Novelty position

`docs/necrofrontier.md` §5 row g13 (eight fetches, decoyed by occurrence
statistics): the natural-bound literature is Okamoto et al. 2021 (the Kepler
superflare statistics with the `f B² A^{3/2}` envelope), Shibayama et al.
2013 and Notsu et al. 2019 ("Starspot Activity and Superflares on Solar-type
Stars"), and "Empirical flare energy limits for the largest historical
sunspots" (the solar calibration of the same bound).  Every one of them draws
the bound and reports that the population sits *below* it.  **None mines the
tail above it as an anomaly set**, and there is no flare-as-technosignature
paper (Lingam & Loeb 2017 is flare *mitigation*).  Vasilyev et al. 2024
(Science; 56,450 Sun-like stars, 2,889 superflares) were never cross-matched
to spot amplitudes.  The statistic below is therefore new in its *question*,
not in its ingredients — which is the position this repository's standing
priorities ask for.

To be verified on the runner by the literature sweep; the position is stated
as "as far as this session knows".

## 3. Method

### 3.1 The bound (`src/seti/arc/ceiling.py`, pure)

* Spot temperature from Berdyugina (2005, LRSP 2, 8) — **`verify`**:

      T_star − T_spot = 3.58×10⁻⁵ T_star² + 0.249 T_star − 808 K

  (1,822 K for `T_star = 5772 K`, i.e. `T_spot ≈ 3,950 K`).
* Spot area from the amplitude:

      A_spot = (ΔF/F) · π R_*² / [1 − (T_spot / T_star)⁴]

  For a 1 % amplitude on the Sun this is `1.95×10²⁰ cm²`.
* The ceiling at `f = 1`, `B = 3 kG`:

      E_mag = (B² / 8π) · A_spot^{3/2}       → 9.7×10³⁵ erg for the case above

  (the Shibayama normalisation `7×10³² erg (f/0.1)(B/1 kG)²(A/3×10¹⁹ cm²)^{3/2}`
  is reproduced to 10 %, which is the test in `tests/test_arc.py`).

### 3.2 The latitude / visibility floor

The amplitude is a **lower bound** on the spot area.  A polar spot, or a
symmetric distribution of spots, modulates the light curve weakly or not at
all while storing the same energy; only a low-latitude group that rotates in
and out of view produces the full amplitude.  The channel marginalises over
spot latitude with a geometric factor from `config/arc.yaml`
(`physics.geometric_factor`, default 3: the true area may be up to three
times the amplitude-implied area) and reports the residual at **both** areas:

    ξ_nominal      = log₁₀ E_flare − log₁₀ E_mag(A_nominal)
    ξ_conservative = log₁₀ E_flare − log₁₀ E_mag(3 · A_nominal)   = ξ_nominal − 0.716

A star is above the ceiling only when `ξ_conservative > 0`.  `ξ_nominal > 0`
alone is a *watch*.

The amplitude used per star is the **maximum** over every source that has
one (the flare catalogue's own brightness-variation column, Shibayama's
star table, Yang & Liu's star table, McQuillan+2014 `Rper`, Santos+2021
`Sph`) — again the conservative choice, since a larger amplitude raises the
ceiling.  Units differ (fraction / ppm / per cent / mmag); the config
declares them per source and a heuristic (`auto`) decides otherwise, with
the decision recorded per star as `amplitude_unit` / `amplitude_unit_guessed`.

### 3.3 The flare energy

Okamoto, Tu, Shibayama and Yang & Liu give a bolometric energy (a
9,000–10,000 K blackbody integral); it is used as is.  Günther+2020 likewise.
A catalogue giving a single-band energy carries the bolometric factor the
paper states in `config/arc.yaml` (`energy.factor`).  Davenport 2016 gives
equivalent durations; the energy is `ED × L_band` with `L_band` a configured
fraction of `L_bol` (**`verify`**, 0.35).  Log columns are detected (a
median below 100 is a log).  Events closer than
`physics.independent_gap_days` (0.5 d) are one energy release; the count
that matters is the number of **independent** flares with `ξ_conservative > 0`.

### 3.4 Per-star record

`ξ_nominal_max`, `ξ_conservative_max`, `n_above_conservative`,
`n_independent`, `E_flare,max`, `E_mag` at both areas, `A_spot` at both
areas, `T_spot`, the amplitude used and its source / unit, `P_rot` and its
source (the flare catalogue first, then McQuillan / Santos), `T_eff`, `R_*`,
`log g` and their sources, `params_assumed` (solar values were used because
no catalogue gave any — the star cannot reach *candidate* until the assess
stage replaces them with Berger+2020 / TIC values), `has_peak_times`.

## 4. Data and stages (`src/seti/arc/run.py`)

All VizieR, with runtime column-role discovery (`acquire.py`, built on
`seti.metronome.acquire`; TAPVizieR table names carry literal double quotes,
so the LIKE has a leading `%` and every table name is quoted).  Nothing is
queried by a column name that was not seen in the service's own metadata.

**The VizieR route ladder.**  ARC's first dispatch (run 34787802564) lost all
six discovery queries to 503s, and on 2026-09-13 *every* TAPVizieR spelling
answered 503 to five attempts each (`results/arc/summary.json →
acquisition.stages`), leaving the channel at `NO_DATA_REACHED` for an
infrastructure reason.  `seti.metronome.acquire` now reaches VizieR over four
routes in order, each recorded with its endpoint and error text:

1. `https://tapvizier.cds.unistra.fr/TAPVizieR/tap` (primary);
2. a second TAP **host** — `https://tapvizier.u-strasbg.fr/TAPVizieR/tap` and
   `https://vizier.cfa.harvard.edu/TAPVizieR/tap`, both marked **verify**:
   asserted from the hostnames CDS and the CfA publish and *not* confirmed
   from the sandbox, which has no egress — plus the plain-http spelling;
3. the **non-TAP** ASU interface, `https://vizier.cds.unistra.fr/viz-bin/asu-tsv`
   (`-source=…&-out.max=…&-out=…&-out.form=TSV` for rows;
   `-source=<cat>&-meta.all`, ReadMe as backstop, for **table existence** —
   the replacement for `TAP_SCHEMA` while TAP is down);
4. `astroquery.vizier.Vizier`.

ARC gets this for free: `discover_table` / `fetch_table` call `list_tables`,
`table_columns`, `count_rows` and `tap_query` from the shared helper, and
those degrade to the non-TAP route instead of raising, so a TAP-down +
VizieR-up dispatch returns **real rows** rather than `QUERY_FAILED`
(`tests/test_arc.py::test_tap_down_but_asu_up_gives_arc_rows_not_query_failed`).
Two consequences are stated rather than hidden: `COUNT(*)` has no ASU
equivalent, so `n_rows` is *unknown* (`None`) on that route and tables are
ranked on their columns alone; and the keyword/description search stays
TAP-only.  The route that served a run is in the acquisition log stages and in
`seti.metronome.acquire.route_log_summary()` (`served_by: asu_tsv`).  When
every route fails, every endpoint and error is recorded and the status stays
`QUERY_FAILED` — no route invents a row.

| Role | Tables (preferred seeds; discovery re-resolves them) |
|---|---|
| Superflares, per flare | Okamoto+2021 `J/ApJ/906/72/table2`; Shibayama+2013 `J/ApJS/209/5`; Tu+2022 `J/ApJ/935/90/table2`; Yang & Liu 2019 `J/ApJS/241/29/table2`; Günther+2020 `J/AJ/159/60/table1`; Davenport 2016 `J/ApJ/829/23/table1` |
| Per-star amplitude / P_rot | `J/ApJS/209/5/stars`, `J/ApJS/241/29/table1`, McQuillan+2014 `J/ApJS/211/24/table1` (`Rper`, ppm), Santos+2021 `J/ApJS/255/17`; Günther `J/AJ/159/60/table2` |
| Parameters + positions (shortlist) | Berger+2020 `J/AJ/159/280` (Gaia–Kepler `T_eff`, `R_*`, `log g`, evolutionary state, binary flag), KIC `V/133/kic` fallback; TIC `IV/39/tic82` |
| Gaia context (shortlist) | `I/355/gaiadr3` 12″ cone: RUWE, NSS, parallax, G, every neighbour; `I/358/vclassre` 2″: variability class |

`probe` → `probe.json`; `acquire` → `data/*.parquet`, `acquire.json`,
`acquisition_log.json` (QUERY_FAILED and QUERY_RETURNED_ZERO_ROWS kept
apart); `screen` → `xi_<cat>.csv`, `screen_<cat>.json`; `assess` →
`summary.json` (verdict, per-catalogue counts, ξ percentiles, the funnel of
`ξ_conservative > 0` before and after each veto, candidates), `xi_table.csv`
(every star), `candidates.csv` / `candidates.json` (interest and candidate
tiers with the Gaia context and the stage-2 pull list).

Verdicts: `NO_DATA_REACHED` (with `data_status` = QUERY_FAILED |
QUERY_RETURNED_ZERO_ROWS), `NO_CEILING_EXCESS`, `CEILING_EXCESS_CANDIDATES`.
`degraded` lists every failed source and every veto that could not be
applied.  None is written up; a null changes the question.

## 5. Contamination ledger (`src/seti/arc/vet.py`)

Applied most-mundane-first; each has a counter and the funnel records the
`ξ_conservative > 0` count after it.

| Veto | Rule | Why it is first-order |
|---|---|---|
| `catalogue_doubtful` | the catalogue's own flag marks the flare or star doubtful / binary / contaminated | Yang & Liu 2019 note earlier catalogues are "seriously polluted"; the flag is the catalogue authors' own vet |
| `companion_suspect` | Gaia RUWE > 1.4, or a DR3 non-single-star solution, or a catalogue binary flag | **The** contaminant: an M-dwarf flarer hidden in a G-dwarf's aperture puts a flare of the M dwarf's energy on a star whose amplitude says it has no spots.  This is the reason the slow-rotator superflare count fell between Maehara+2012 and Notsu+2019 |
| `blend` | a Gaia neighbour within 12″ brighter than `G_target + 3` | the aperture flux is normalised to the blend: the flare may be the neighbour's, and the amplitude is diluted — both push ξ up |
| `evolved` | `log g < 4.0` or `R_* > 1.6 R☉` | a subgiant's ceiling is not the dwarf's, and its "amplitude" is often granulation or pulsation, not spots |
| `pulsator_or_eb` | Gaia DR3 variability class ECL / DSCT / GDOR / RR / CEP / … (RS CVn and SOLAR_LIKE do **not** trip: those *are* spots) | eclipses and pulsation cycles chopped into "flares" by the flare finder, and an "amplitude" that is not a spot |

Report-only (never reject, but hold a star at *interest*):
`single_flare_above` (one exceeding flare is a measurement problem until it
repeats), `gaia_unreached`, `stellar_params_assumed`; and purely
informational `amplitude_unit_guessed`, `no_peak_times`.

Inherited from `docs/channel-brief.md` §4 and applied by construction:
momentum dumps and other cross-star epochs are already absent from the
superflare catalogues (Okamoto / Shibayama inspect every event by eye); the
bolometric conversion is the catalogue's own; duplicates across catalogues
are kept as separate records (`record_key`) and reported together per
`star_key`.

Tiers: `none` (unassessable / below ceiling / hard veto), `watch`
(`ξ_nominal > 0 ≥ ξ_conservative`), `interest` (above the conservative bound,
no hard veto, but a report-only hold), `candidate` (≥ 2 independent flares
above the conservative bound, every veto applied and passed) — **pending the
stage-2 centroid test always**.

## 6. Stage 2 — the pixel-centroid test (designed, not built)

The catalogues cannot say *which pixel flared*.  The Kepler target pixel
file and a TESS FFI cutout can.  For every candidate, stage 1 writes
`centroid: not_checked` and a `stage2_pulls` list: one entry per independent
flare above the conservative bound with its peak time, energy, ξ, and the
Kepler quarter (from BKJD via the published quarter table) or TESS sector
(from the catalogue's sector column), and the product to pull (`kepler_tpf`
via `lightkurve.search_targetpixelfile(KIC, quarter=Q)`, `tess_ffi_cutout`
via TESScut at the TIC position, sector S).

The test, per flare: the flux-weighted centroid of the aperture in a window
of ±3 cadences around the peak, against the centroid in the ±2 h out-of-flare
baseline, in the row and column directions; the shift in pixels divided by
its baseline scatter, and — the decisive quantity — the *direction* of the
shift, compared with the direction to every Gaia neighbour inside the
aperture.  A flare that belongs to the target shows no centroid motion above
the baseline scatter on *every* exceeding flare; a companion flarer moves the
centroid toward the same neighbour on every one.  A flare with no centroid
motion but a Gaia neighbour inside 1″ (unresolved by Gaia) remains
ambiguous and is reported as such; the difference-image method (the flaring
pixels minus the baseline, as the Kepler DV pipeline does for transits)
gives the companion's position where the centroid alone cannot.

Stage 2 also pulls the light curve itself around every exceeding flare, to
confirm the impulsive rise / exponential decay shape, and to remeasure the
amplitude in that same quarter / sector (the amplitude used in stage 1 is
the catalogue's, which may be from a different epoch — spots evolve).

## 7. Offline tests (`tests/test_arc.py`)

The Berdyugina and spot-area formulas against hand computations and against
the Shibayama normalisation; a 1 % amplitude star with two flares at 10× the
`f = 1` bound → `ξ_conservative = 1.0` on two independent flares and a
*candidate* once the Gaia cone answers and every veto passes; the same
energies on a 5 % amplitude star → `ξ_conservative < 0`, a *watch*; RUWE = 2
→ `companion_suspect`; `log g = 3.5` → `evolved`; a 12″ neighbour at
`G + 1.5` → `blend`; Gaia `ECL` → `pulsator_or_eb`; a catalogue flag →
`catalogue_doubtful`; a single exceeding flare → *interest*; a failed or
empty VizieR → `NO_DATA_REACHED` and never a candidate; the funnel counts in
order; an end-to-end run through a scripted TAP and Gaia cone; and the route
ladder in both directions — a TAP-down + ASU-up world yields real rows with
`served_by: asu_tsv` (discovery off `-meta.all`, rows off `asu-tsv`), while a
world with *every* route down stays `QUERY_FAILED` with every endpoint and
error named.  No test opens a socket: every route takes an injectable
fetch/query callable.

## 8. Limits — stated so nobody overclaims

* **The amplitude is not the spot area.**  Even at 3× the residual is a
  model statement.  Notsu et al. (2019) show the amplitude–area relation has
  factor-of-few scatter against Doppler-imaged stars; the geometric factor
  absorbs the systematic, not the scatter.  A single star at
  `0 < ξ_conservative < 0.5` is therefore inside the natural model
  uncertainty and is listed, not claimed.
* **`B = 3 kG` is an assumption**, and `f = 1` is a limit, not a physical
  release.  Both make the ceiling *high*, i.e. the test conservative; a star
  above it is remarkable *because* of that.
* **The bolometric energies are model integrals** (a 9,000–10,000 K
  blackbody).  A flare with a different SED changes `E` by a factor of ~2
  (0.3 dex).
* **Companion flarers are the contamination set** and stage 1 can only
  *suspect* them (RUWE, NSS, neighbours).  A companion inside ~0.1″ is
  invisible to Gaia and to the centroid test alike; only spectroscopy or
  the flare colour would resolve it.  Every stage-1 candidate is therefore a
  *pending* object by definition.
* **Coverage** is the catalogues' own: superflare completeness falls with
  energy, the Kepler catalogues are solar-type dwarfs by selection, and a
  star with no measured amplitude (a non-rotator, or a very slow one) is
  *unassessable* — the very stars the claim names (spot-free slow rotators)
  are the hardest to bound.  `n_stars_no_amplitude` in `summary.json` says
  how many.
* `NO_CEILING_EXCESS` is a count, not an occurrence limit, and is not
  written up (CLAUDE.md).
