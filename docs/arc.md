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
   (`-source=…&-out.max=…&-out=…&-out.form=TSV` for rows; for **table
   existence** — the replacement for `TAP_SCHEMA` while TAP is down — one row
   of `-source=<table>&-out.max=1&-out.all`, then `-meta.all`, then the
   ReadMe);
4. `astroquery.vizier.Vizier`.

**Table existence without `TAP_SCHEMA`.**  ASU addresses a catalogue as
`-source=J/ApJ/906/72/table2` and has **no `TAP_SCHEMA` of its own**, so the
question "does `TAP_SCHEMA.tables` contain this id" has to be re-asked as
"give me one row of `-source=<id>`": a well-formed TSV body is the existence
proof and its header *is* the column list, which is what discovery then scores
(`asu_table_exists`).  The dispatch of 2026-09-14 (run 34792280736) failed
precisely here — `-meta.all` was the only non-TAP existence route and it did
not resolve `J/ApJ/935/90/table2`, `J/ApJS/241/29/table2` or
`J/AJ/159/60/table1`, so those three read `no route to TAP_SCHEMA.tables ~ …`
while TAP was down.  The one-row check is tried **first** for an exact table
id (which is how five of the six catalogues are asserted).  A body that comes
back as a VizieR error page, or that names a *different* catalogue than the
one asked for, is recorded as a failed attempt — never read as rows.

**The circuit breaker.**  Per endpoint: after two consecutive failures it
opens and skips that host for a **45 s** cooldown, which then *half-opens* it
for one cheap attempt (one try, not the retry ladder); any success closes it.
A host that has **already served a query in this process is never skipped** —
it is intermittent, not down, and only the next attempt can show it is back.
That rule is the fix for the other half of run 34792280736: TAPVizieR served
`discover_kepler_okamoto2021_preferred` and then 503'd twice, and a
five-minute cooldown skipped all five remaining catalogues
(`skipped: … failed 2 times in this process, 82 s ago`) — an intermittent
service turned into a total outage, which is worse than no breaker at all.
Every breaker transition is recorded in the acquisition log
(`acquisition.breaker.transitions`).

ARC gets this for free: `discover_table` / `fetch_table` call `list_tables`,
`table_columns`, `count_rows` and `tap_query` from the shared helper, and
those degrade to the non-TAP route instead of raising, so a TAP-down +
VizieR-up dispatch returns **real rows** rather than `QUERY_FAILED`
(`tests/test_arc.py::test_tap_down_but_asu_up_gives_arc_rows_not_query_failed`).
Two consequences are stated rather than hidden: `COUNT(*)` has no ASU
equivalent, so `n_rows` is *unknown* (`None`) on that route and tables are
ranked on their columns alone; and the keyword/description search stays
TAP-only.  The route that served a run is recorded per catalogue in
`acquire.json` and `summary.json` (`route: asu_tsv`, beside
`discovery_route`, which says whether the *seed* or the keyword search found
the table), on every acquisition-log stage, and in
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

## 6. Stage 2 — is the flare on the target? (`src/seti/arc/stage2.py`, built 2026-09-21)

The catalogues cannot say *which pixel flared*.  The Kepler target pixel
file (and a TESS SPOC pixel file or TESScut cutout) can.  Stage 2 runs on
the stage-1 shortlist — interest tier first, then watch, by ξ — through
`arc-stage2.yml` (`python -m seti.arc.stage2 --stage all`), and writes
`results/arc/stage2/{probe,summary,stars}.json`, `flares.csv`, `census.csv`,
`acquisition_log.json`.  A stage-1 re-assess (`arc.yml`, `stage=all`) then
carries each star's verdict into the `centroid` column.

**Per star, in order** (every step injectable, every gap a stated status):

1. **The catalogue's own flare rows** (`fetch_flare_rows`): the star's rows
   from the table stage 1 used, with the roles re-resolved.  Yang & Liu 2019
   `table2` is `recno, KIC, Q, Begin, End, logE` — the start time was there
   all along; stage 1's `t_start` patterns matched `End` and not `Begin`,
   which is the whole of `no_peak_times` and the null quarters in the
   stage-1 pull lists (now fixed in `acquire.ROLE_PATTERNS`).  Times are
   brought to the mission-native system (`BKJD` / `BTJD`) by the same
   `guess_time_system` / `normalise_time_system` pair as stage 1.
2. **Stellar parameters** (`stellar_parameters`): Berger+2020 `table2`
   (Teff, log g, R★, M★, evolutionary state — no positions), `table1`
   (Gaia id, RUWE, positions), the KIC as the last resort, merged per column
   in that order with the source named (`fetch_star_params_by_id`, which no
   longer demands a position of every table); then Gaia DR3 FLAME
   (`I/355/paramp`: `Rad-Flame`, `Mass-Flame`, `Lum-Flame`, `Age-Flame`,
   columns discovered at runtime) by source id.  This closes
   `stellar_params_assumed`.
3. **The Gaia census** (`gaia_sources` + `pixels.census`): every DR3 source
   within 12″, its separation and position angle, its flux ratio to the
   target and its **capture fraction** into the aperture — from the pixel
   file's own pipeline aperture through a Gaussian PRF (`verify`: σ = 0.7 px
   for Kepler), and analytically for a 4″ (one pixel) and a 12″ circle.  The
   fraction of the aperture flux each neighbour supplies, `f_j`, gives the
   relative brightening it would need to make the observed aperture excess
   `a`: `a / f_j`.  A neighbour needing more than 2000 % (`verify`) is
   *excluded by arithmetic* and stays excluded whatever the pixels say.
4. **Light curves** (`lightkurve` PDCSAP, SAP as the second column, the
   pixel aperture sum as the fallback) for every quarter / sector holding a
   flare above the ceiling: the flare is **re-detected**
   (`flares.detect_flares`: iterated running-median baseline, ≥ 2 consecutive
   points above 3σ with a 4σ peak, the window extended to 1σ), **matched** to
   the catalogue row by time overlap where the row has a window and by energy
   rank within the quarter where it does not (the method is recorded), and
   **re-measured**: equivalent duration, rise and decay times, the Shibayama
   9,000 K blackbody bolometric energy (`flare_energy_shibayama`, the same
   construction as the catalogue's, through a tabulated Kepler response —
   `verify`) and the band energy `ED × L_band`.  The quarter's own
   **rotational amplitude** is measured from the flare-free baseline as
   `Rvar` (5th–95th percentile range, McQuillan's definition) and `Sph`
   (standard deviation, Santos's).
5. **The centroid test** (`pixels`), per matched flare on the target pixel
   file of that quarter:
   * *cadences*: in-flare = the detected window; the difference image uses
     the cadences above 20 % of the peak; baseline = every good cadence
     within ±1 d outside the flare padded by 3 cadences;
   * *the flux-weighted shift*: the aperture centroid per cadence, a linear
     trend fitted on the baseline (pointing drift), the in-flare mean minus
     the trend, with the error from the baseline residual scatter.  Every
     source predicts a shift `a/(1+a) · (c_source − c_baseline)`; on the
     target it is a few 10⁻⁴ px, on a neighbour it points at the neighbour;
   * *the difference image*: every pixel detrended against its own baseline,
     the in-flare residual averaged, its per-pixel error from the baseline
     scatter floored at the photon error; its thresholded centroid **is the
     position of whatever brightened**, with the analytic error ⊕ a 0.1 px
     systematic floor (`verify`);
   * *positions*: the target's pixel position is the baseline image's own
     centroid corrected for the neighbours' known flux fractions, and every
     neighbour keeps its WCS offset *relative to the target* — a WCS
     zero-point error cancels to first order and is recorded
     (`wcs_anchor_dx/dy_px`);
   * *attribution*: `z_j = |c_D − c_j| / σ` per source.  `on_target` when the
     target is within 3σ and every not-excluded neighbour is beyond 3σ;
     `on_neighbour:<id>` when a neighbour is within 3σ and the target beyond;
     `ambiguous` when both are, or when a not-excluded neighbour sits within
     0.5 px of the target (the pixels cannot separate them);
     `unattributed` when nothing is; `undetected_in_pixels` when the
     difference image's peak is below 3σ.
6. **ξ on measured parameters** (`_xi_block`): the catalogue energies with
   the measured Teff / R★ and (a) the catalogue amplitude — a Santos `Sph`
   rescaled by 2√2 to a range (§3.2) — and (b) the quarter's own `Rvar`; the
   **larger** amplitude is the headline (the conservative choice), and the
   re-measured energies are run through the same ceiling beside it.

**Verdict per star** (`pixels.star_verdict`): `flare_on_target` (every
attributable flare on the target), `flare_on_neighbour` (any on a
neighbour), `centroid_ambiguous`, `centroid_untestable` (no attributable
flare, with the reason: no pixel file / no light curve / flare not
re-detected / low pixel SNR / budget exhausted).  **Overall**:
`CEILING_EXCESS_ON_TARGET_PENDING_SPECTROSCOPY` (≥ 1 interest star on target
*and* ξ > 0 on measured parameters), `CEILING_EXCESS_TRACED_TO_NEIGHBOUR`,
`CEILING_EXCESS_DISSOLVED_ON_MEASURED_PARAMETERS` (every interest star below
the ceiling once the measured radius and the rescaled amplitude are in),
`CEILING_EXCESS_CENTROID_UNTESTABLE`, `CEILING_EXCESS_CENTROID_AMBIGUOUS`,
`STAGE2_NO_DATA_REACHED`.

**What it cannot do.**  A companion inside ~0.1″ is unresolved by Gaia and by
the pixels alike; `flare_on_target` therefore means "on the Gaia-resolved
target", and only spectroscopy or the flare's colour could go further.  Its
offline gate (`tests/test_arc_stage2.py`, 25 tests): the injected flare on
the target → `on_target` with a shift consistent with zero and the target's
prediction; on a neighbour 2 px away → `on_neighbour` with the centroid on
the neighbour and the shift toward it; a 0.05 px/d pointing drift removed by
the baseline trend; an unresolved neighbour → `ambiguous`; a faint neighbour
→ excluded by arithmetic; a missing or failed pixel fetch, a missing light
curve (the aperture sum takes over), a dead archive and an exhausted budget
each → `centroid_untestable` with the reason and never `on_target`; and the
whole stage end to end through scripted VizieR / Gaia / MAST callables,
including the stage-1 re-assess picking the verdict up.

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
ladder in every direction — a TAP-down + ASU-up world runs the *whole*
acquisition to real rows with `route: asu_tsv` per catalogue and a verdict
that is not `NO_DATA_REACHED` (discovery off the one-row existence check,
rows off `asu-tsv`), an *intermittent* TAP never skips a later catalogue
(every catalogue gets a genuine attempt, and one asked after the breaker
opened is still discovered), the breaker half-opens after its cooldown and
closes on the first success, an ASU body that names another catalogue is not
existence evidence, and a world with *every* route down stays `QUERY_FAILED`
with every endpoint and error named and nothing written.  No test opens a
socket: every route takes an injectable fetch/query callable.

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

## 9. What the runs measured (2026-09-22)

### 9.1 The stage-1 funnel

Run **35738218021** (`arc.yml`, `stage=all`) is the first stage-1 run whose
*assess* stage finished: probe 2 min, acquire 3 min, screen 7 s, assess
**277 s** (its predecessor, run 35675114711, sat in assess for 4 h 54 m on a
`pyvo` async job with no time limit and was killed by the workflow cap
without writing a line — §4, `assess.budget_s`).

| | |
|---|---|
| flares screened | 190,486 across five catalogues |
| stars with flares | 8,908 |
| **assessable** (a rotational amplitude, so a ceiling) | **4,206** |
| no amplitude, so no ceiling and no test | 4,702 |
| `ξ_conservative > 0` | **1** |
| candidate / interest / watch | 0 / 0 / 14 |

The Santos+2021 `Sph` lever was *already* in the sample before this run: it
supplies 245 of 2,507 assessable Yang & Liu stars and 22 of 279 Shibayama
stars, and the assessable count moved 4,204 → 4,206, not "well beyond".  What
the lever actually did was **raise the ceiling**: rescaling `Sph` from a
standard deviation to a range (§3.2, ×2√2) lifts `E_mag` by 2.828^1.5 = 4.75,
i.e. **0.68 dex**, and that alone cut the conservative-positive count from 5
to 2 and the nominal-positive count from 24 to 9.

Run **35741271294** re-ran `assess` alone (288 s) with one rule changed: a
shortlisted star takes its Teff and radius from the measured table whenever
that table answers, not only when the record was flagged
`stellar_params_assumed`.  Nothing else moved — 8,908 stars, 4,206 assessable,
one star above the conservative ceiling, 13 in `watch` — but the *size* of
that one excess did, because it had been carried on a KIC-era star table:
`ξ_conservative,max` **+0.462 → +0.715**, `n_above_conservative` 4 → 6,
`n_above_nominal` 11 → 12 (§9.3).  The displaced values stay on the record as
`teff_k_before` / `radius_rsun_before` / `xi_conservative_max_before`.

### 9.1a The centroid test on 21 real stars

Run **35738785437** is the first stage-2 pass with the NaN-pixel fix (a single
permanently-NaN pixel had been vetoing every cadence of a stamp, §6).  It was
cancelled at 21 of its 30 stars in favour of a run on the current shortlist,
and the per-star checkpoint in `stars.json` holds what it measured:

| | run 35675112803 (30 stars) | run 35738785437 (21 stars) |
|---|---|---|
| `flare_on_target` | 6 | **8** |
| `flare_on_neighbour` | 1 | **5** |
| `centroid_ambiguous` | 1 | 5 |
| `centroid_untestable` | **22** | **3** |

Per flare: 58 examined, **21 on target**, **5 on a neighbour**, 9 ambiguous,
7 with too few pixels above the difference-image threshold, 2 unattributed,
14 untestable.  Parameters were measured for 20 of 21 stars.

**Five of 21 stars had their catalogued flare land on a different Gaia source**
— KIC 10288777 (16.8σ from the target), KIC 7009116 (14.6σ), KIC 9268205
(7.0σ), KIC 7174965 (4.8σ), KIC 9139163 (3.9σ), each consistent with a named
DR3 neighbour.  That is not a statement about this channel's candidates; it is
a statement about the flare catalogues, whose per-star attribution comes from
the pipeline aperture and has here been contradicted at up to 17σ by the
difference image on the same cadences.

### 9.2 The two stage-1 interest stars dissolved

Stage 2 (run **35675112803**, 2 h 08 m, 30 stars, 69 flares) closed the
`stellar_params_assumed` flag on 25 of 30 stars from Berger+2020 `table2` and
recomputed ξ:

| star | ξ stage 1 | ξ measured | Teff, R★, M★ (Berger+2020 `table2`) | amplitude used | centroid verdict |
|---|---|---|---|---|---|
| KIC 11507705 | +0.440 | **−1.240** | 6365.3 K, 1.311 R☉, 1.187 M☉ | 8.890e-4 (quarter `Rvar`) | **`centroid_ambiguous`** |
| KIC 8487271 | +0.104 | **−0.920** | 5998.6 K, 1.301 R☉, 1.209 M☉ | 1.0746e-3 (catalogue, as range) | **`flare_on_target`** |

Both are settled, and settled differently from each other:

* **KIC 8487271 — the flare is on the star, and the star is under its
  ceiling.**  Its one testable flare (t_peak 143.487, 4.57e34 erg) put the
  difference-image centroid **0.171 px (1.5σ) from the target**, with every
  Gaia neighbour rejected at > 3σ or excluded by arithmetic.  So the pixel
  test passes and the object still dissolves — on the parameters, not on a
  blend.  Its quarter `Rvar` (3.45e-4) is *smaller* than the catalogue
  amplitude, so the conservative larger value is the one used.
* **KIC 11507705 — ambiguous, and further below the ceiling than stage 1
  thought.**  Of 6 catalogue flares, 2 reached the pixels: one is consistent
  with the target *and* with Gaia DR3 2129762445437102464 (`ambiguous`), the
  other had only 2 pixels above 20 % of the difference-image peak where 3 are
  required; the remaining 4 were not re-detected in the light curve.  The
  star's own quarter amplitude (8.890e-4) is **2.7× the catalogue value**, so
  ξ falls further than §9.1's numbers: +0.440 → **−1.240**.

The stage-1 → measured move for KIC 11507705 decomposes: **0.677 dex** from
the `Sph` → range rescaling, **0.353 dex** from the radius (1.311 R☉ for an
assumed 1.000), the rest from its own measured rotational amplitude.

### 9.3 The one star still standing: KIC 9418692

Of 4,206 assessable stars, exactly one is above the conservative ceiling.  On
run 35741271294's parameters (Berger+2020 `table2`: Teff 5677.4 K,
R = 1.089 R☉, replacing the Shibayama star table's 5378 K / 1.300 R☉):

* **ξ_conservative = +0.715 on 6 flares** (ξ_nominal = +1.431 on 12), of 14
  Yang & Liu 2019 flares in 13 independent events;
  `E_flare,max = 9.78 × 10^34 erg`.  On the displaced KIC-era parameters the
  same flares gave +0.462 on 4: the excess *grows* by 0.25 dex on the better
  stellar parameters, which is why they are now preferred whether or not the
  record was flagged assumed.
* Amplitude `2.008 × 10^-4` from Santos+2021 **with** the ×2.828 range
  scaling already applied — the ceiling is not being under-counted.
* **The amplitude is the whole result, and the two catalogues disagree by 3×.**
  The same star carries a *second* stage-1 record from Shibayama+2013 whose
  amplitude is `6.0 × 10^-4` (`amplitude_source: own`, `amplitude_unit`
  guessed).  Put the Yang & Liu energies on Berger's radius with that
  amplitude instead and `ξ_conservative = +0.002` — exactly at the ceiling,
  not 0.7 dex above it, because `E_mag ∝ A^{3/2}` turns a factor 3 in
  amplitude into 0.71 dex of ceiling.  Nothing about this object can be
  claimed until its rotational amplitude is measured from its own light
  curve, which is what stage 2's `amplitude_quarter_rvar` does.
* Gaia DR3 **RUWE = 1.556** → `first_veto = companion_suspect`, which put the
  star in *no tier at all* and therefore outside every stage-2 shortlist
  (fixed: `stage2.include_vetoed_excess`).
* Gaia census: 3 sources within 12″.  Inside one Kepler pixel (4″) the target
  supplies **99.73 %** of the flux; the two neighbours (G = 20.41 at 3.20″,
  G = 19.69 at 5.18″) supply 0.14 % and 0.13 % and would have to brighten by
  **39 %** and **41 %** to produce the observed aperture excess.  Neither is
  excluded by arithmetic, so only the pixels can decide.
* Its only stage-2 pass so far ran on its *Shibayama* record — one flare row
  at the 6.0e-4 amplitude, `ξ = −0.386` — and returned `centroid_untestable`.
  Run **35744902798** is the first to put its Yang & Liu record on the pixels:
  `load_shortlist` now ranks it first of 13 as `tier: vetoed_excess`.

This is not a null and it is not a candidate.  It is **one object with three
open questions, none of them yet answered by a measurement of this star**:

1. *Which rotational amplitude is right?*  Santos+2021's `Sph` (as a range,
   2.008e-4) puts it 0.7 dex above the ceiling; Shibayama's 6.0e-4 puts it
   exactly at the ceiling.  Decided by `amplitude_quarter_rvar` — the star's
   own Kepler light curve — in run 35744902798.
2. *Is the flare on the target?*  Untested until that run.  §9.1a shows the
   test is not a formality: it moved 5 of 21 stars onto a neighbour.
3. *Is there a companion?*  RUWE 1.556 is an astrometric **suspicion**, not a
   detection, and it is the only thing that vetoed the star.  Needs the Gaia
   DR3 non-single-star solutions and any archival spectroscopy.

Its Gaia census already bounds the third: the target supplies 99.73 % of the
flux inside one Kepler pixel, and the two catalogued neighbours would have to
brighten by 39 % and 41 % — large, but not excluded by arithmetic, which is
precisely why the centroid and not the census is the instrument.
