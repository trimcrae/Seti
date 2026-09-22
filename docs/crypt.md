# CRYPT — artifacts in lunar permanent shadow (S55)

*Built 2026-09-21. Package `src/seti/crypt/`, CLI `crypt --stage
{probe,pcp,assess_diurnal,acquire,screen,assess,all} --shard i/n`, workflow
`crypt.yml`, config `config/crypt.yaml`, results `results/crypt/`. The claim
and the data ledger come from `docs/necrofrontier.md` §2.XVII (S55) and §3.
Nothing below is a measurement of the sky until a run has committed
`results/crypt/summary.json`; §7 records what each run taught.*

> **What the archive turned out to hold, and what the channel therefore
> screens.** The brief called for multi-channel anisothermality inside the
> PSRs. There is no per-channel Diviner *polar gridded* product: ODE's own
> index lists 912 Polar Cumulative Products and every one of them is
> `AVG TBOL` (run 35737908601, `results/crypt/probe.json`). So §3.2 cannot
> be computed from a level-3/4 map product and is retained against the day
> one exists. The axis that **does** exist — and that nobody has screened —
> is the one the PCP is binned on: average bolometric temperature per 240 m
> pixel in local-time bins, separately for lunar summer and winter. That is
> a *diurnal curve inside permanent shadow*, and it is what §3.7 screens.

## 1. Claim

Regolith turnover in a permanently shadowed region is metres per gigayear
and the floor sits at 25–110 K. It is the one place near us where a
gigayear-old object survives, and a *thermally active* one — a decaying
radioisotope source, a reactor, anything with an internal heat budget — is
visible against nothing. A 1 m² surface at 300 K inside a 240 m Diviner pixel
at 35 K raises the channel-6 (13–23 µm) brightness temperature to **57 K**
while channel 9 (100–400 µm) moves by **0.01 K** (`thermal.BandModel`
reproduces both numbers; `tests/test_crypt.py::test_brief_numbers…`). The
discriminant is *anisothermality* — the same physics as Diviner's
rock-abundance retrieval at low latitude (Bandfield et al. 2011), never
applied inside PSRs. A metre-scale metallic dihedral is a single-pixel
Mini-RF bright spot with CPR > 1 and no boulder.

## 2. Novelty position

Davies & Wagner 2013 proposed optical NAC inspection of PSR floors and named
"nuclear waste heat" as a signature; every executed search since is optical
machine learning on NAC or ShadowCam frames (Lesnikowski 2020; the 2025 AAS
work; arXiv:2608.09350, Aug 2026). The necrofrontier literature sweep
(`results/necrofrontier_lit/`, group g8, 18 decoy-free hits) found **no
thermal and no radar artifact search inside PSRs**. Diviner PSR papers
(Paige et al. 2010 Science; Williams et al. 2019 JGR "Seasonal polar
temperatures on the Moon"; Hayne et al. 2021 micro cold traps) map bulk
temperatures and ice stability; the Diviner rock-abundance product stops at
the terminator. Mini-RF PSR papers (Spudis, Cahill, Thomson) read CPR for
ice and roughness, never for compact isolated reflectors. The thermal and
radar axes are therefore unoccupied, and this channel occupies both.

## 3. Method

### 3.1 The mask

The cold-trap mask is the Diviner definition: the all-time (or summer)
bolometric maximum below `psr_tmax_K = 110 K` (Williams et al. 2019 use
110 K as the water-ice cold-trap ceiling; PSR floors are far colder), eroded
by `edge_px = 2` footprints so the illumination boundary and its 160 × 320 m
footprint bleed are outside. When a LOLA/PGDA permanent-shadow raster is
reached it is ANDed in and the route is recorded; when none is reached the
summary carries `psr_from_thermal_definition_only`. No external polygon
source is assumed to exist: the probe records what answers.

### 3.2 The thermal statistic

For a single-temperature surface every channel returns the same brightness
temperature up to a smooth emissivity offset. With T_ref the longest
channel that has data (9, else 8):

    Δ_k = T_k(avg) − T_ref(avg),   k ∈ {6, 7, 8}

is measured per pixel per season. Its noise is **measured from the PSR
interior itself** in 2 K bins of T_ref (median → the calibration offset,
1.4826·MAD → σ; a bin with fewer than 50 pixels falls back to the global
values), and z_k = (Δ_k − median)/σ. The primary channel is the shortest one
with ≥ 20 % finite coverage of the interior. A pixel is *flagged* at z ≥ 5
in either season; the flagged table carries every channel, both seasons, the
observation counts and a grid fit of (f, T_hot) at T_cold = T_ref over
channels 6–9 (f and T_hot are degenerate on the grid; the well-constrained
quantity is the hot component's **excess radiance** f·L_k(T_hot), which is
also what the seasonal stability test compares, because it is independent
of T_cold).

### 3.3 The floor

The floor is stated, per season and per T_ref bin, as the 300 K source area
that would raise the primary channel by 5σ (`floor_area`), and then
*measured* by injection: sources of 1–10⁴ m² at 300 K and 500 K are added to
random interior pixels of the **real** maps and the whole screen is re-run
(`sensitivity_<pole>.json`); the summary quotes the smallest area recovered
in ≥ 50 % of trials. Nothing about the Diviner instrument's NER is asserted
from memory — the scatter of the product is the floor.

### 3.4 The rules (each a named counter)

| Rule | Kills | Why |
|---|---|---|
| `seasonal` | z ≥ 5 in one season, < 2.5 in the other | a lit rim, secondary illumination or scattered light follows the Sun; an internal source does not |
| `unstable` | excess radiance differs between seasons by > 3σ and > ×3 | a source whose output follows the season |
| `single_season` | only one season has data at the pixel | cannot pass the seasonal test; recorded as interest |
| `extended` | the connected z ≥ 2.5 patch exceeds 6 px (≈ 1.4 km) | boulder fields, ejecta, warm slopes — never a point source |
| `stripe` | ≥ 4 flagged pixels share a row/column within 200 px | calibration striping, map seams |
| `low_count` | < 10 observations in either season | a few footprints cannot average out a cosmic ray |
| `inconsistent_spectrum` | Δχ²(2-T vs 1-T) < 9, or Δ8 > Δ7 (or Δ7 > Δ6) beyond noise | the excess is not a hot component; noise at the floor |
| `human_hardware` | within a listed spacecraft site (LCROSS, IM-1, IM-2, Chang'e-7 region) | real warm hardware; flagged, never silently dropped |

Human hardware is a *flag*: a warm spacecraft in a PSR would also be the
positive control this channel otherwise lacks (the Apollo RTGs sit at low
latitude, outside the polar products).

### 3.7 The diurnal/seasonal invariance screen (the one that runs)

`src/seti/crypt/diurnal.py`, stages `pcp` and `assess_diurnal`.

**The mask.** The LOLA mapped permanently-shadowed raster
(`LPSR_65{N,S}_240M`, binary, 6420 × 6420, 240 m polar stereographic true at
the pole on R = 1737.4 km) is resampled onto the PCP grid and ANDed with the
Diviner cold-trap definition (never warmer than `psr_tmax_K` in any loaded
bin), then eroded by `edge_px`. The registration comes off the label's
`LINE_PROJECTION_OFFSET = 3209.5`, **not** the array centre: the pole sits at
0-based pixel 3208.5, one pixel off centre, and cropping about the centre
moves the mapped shadow by 240 m. A fallback to the centre is reported in
`pcp_<pole>.json` as `registration: array_centre_fallback`, never assumed
silently. The PCP grid is pixel-registered on the pole, so the two
registrations still differ by half a pixel (120 m); that residual is
irreducible from the published products.

**The statistic.** For each pixel, the **floor** temperature

    T_floor = min over every loaded (season, local-time) bin of T_bol

against a local *square* annulus (`r_in_px`–`r_out_px`) background. The
annulus is square because a box filter is separable and O(N): the polar
grids are 2535 px on a side and tens of layers deep, where a 33 × 33 direct
convolution per layer is not tractable. Only the annulus's symmetry about
the pixel enters the statistic, and a square has it.

The scatter needs two passes. A PSR floor is a *bowl*: the annulus mean of a
curved field is not its centre value, so the raw residual carries a
near-constant curvature bias that inflates σ and hides real sources. Pass 1
forms `field − annulus mean`; pass 2 removes the annulus mean of that
residual before measuring the spread (1.4826 × mean absolute residual). What
is left is the detection field, and `z_floor = excess/σ`.

**Why the floor.** Inside a PSR there is no direct sun at any local time,
but every passive heating term still has a clock: scattered light off a
sunlit rim and re-radiated IR from the surrounding terrain both rise and
fall with the sun's position and both collapse in winter. Taking the
*minimum over the whole (season, local-time) grid* is immune by construction
to heating that switches off at any point in the year. An internal source is
not.

**Flagging is on the floor OR the all-bin mean.** Flagging on the floor
alone would make every seasonal and diurnal confounder vanish silently
instead of being counted; flagging the mean as well puts each one into the
table where a named rule rejects it, and `not_in_floor` is itself a named
rejection.

**Rules** (each a counter in `summary.json["counts"]`):

| Rule | Kills |
|---|---|
| `human_hardware` | within a listed spacecraft site — flagged, never dropped |
| `low_coverage` | fewer than `n_bins_min` bins carry data at the pixel |
| `extended` | the connected elevated patch exceeds `max_cluster_px` |
| `stripe` | `stripe_min` flagged pixels share a row/column within `stripe_window` |
| `edge` | touches the mask boundary (illumination edge, footprint bleed) |
| `few_bins` | the excess is present in fewer than `n_bins_excess` bins |
| `not_in_floor` | flagged on the all-bin mean but not on the floor |
| `seasonal` | \|summer − winter\| mean exceeds `dseason_max_K` |
| `diurnal` | the winter diurnal amplitude exceeds `amp_max_K` |

**The floor, stated.** Bolometric mixing is exactly Stefan–Boltzmann, so the
detectable area inverts in closed form with no band model and no fitting:
`A_min` is the 300 K area that lifts a median PSR-interior pixel by
`floor_nsigma` of its own local scatter. It is then *measured* by injection
on the real maps (`diurnal_sensitivity_<pole>.json`). Each trial runs in a
window that contains the widest neighbourhood any rule consults, which is
what makes the table affordable on the real grid and is legitimate only
because every rule in the screen is local.

**Bins.** Which local-time bins exist is read off ODE's own PCP file index
(`pcp_index.json`), never assumed; a bin is used only when it exists in
*every* requested season, so the seasonal comparison a candidate must
survive can actually be made. Each table is ~262 MB of ASCII (≈4.4 M rows;
empty bins are omitted) and is deleted after it is rasterised, so peak disk
is one product. Which bins were used is recorded with every result.

### 3.8 The radar axis, on the same mapped PSR (stage `radar`)

The Mini-RF **polar stereographic mosaics** — `lsz_xxxxx_3cp_pfu_90{n,s}000_v1`
(circular polarisation ratio) and `..._3s1_...` (first Stokes, total power) —
are 1294 × 1294 `PC_REAL` at `MAP_SCALE = 947.6 m/pix` covering 70–90°, i.e.
6.7 MB each: the whole radar axis is a few tens of megabytes, not a data
problem. The `xxxxx` token marks the *merged* mosaic; a numeric token is a
single-orbit strip, and the merged product is preferred. The mosaics
directory is listed rather than the names assumed, and what was listed is
recorded with the result.

The mask is the **same LOLA raster the thermal screen uses**, re-sampled from
the 240 m PCP grid onto the Mini-RF grid by longitude and latitude, so both
axes screen the same region and a hit on one can be read off the other. The
rules are §3.5's, and the kills they name — `rock_field`,
`elevated_background`, `weak_echo`, `mask_edge` — are the natural high-CPR
sources the brief demands be excluded.

The floor here is coarse and stated as such: at 948 m/pixel a metre-scale
dihedral is a small perturbation on a resolution cell 16 times the area of a
Diviner one, so this axis constrains *compact isolated reflectors at the
mosaic's own scale*, not metre-scale hardware.

### 3.5 Radar (rules)

Inside the mask re-sampled onto the Mini-RF grid: CPR ≥ 1 pixels whose
connected patch is ≤ 4 px (`rock_field` otherwise), whose annulus (3–12 px)
median CPR is < 0.6 (`elevated_background`: ejecta and roughness), whose
total power exceeds the annulus median by ×3 when S1 exists (`weak_echo`:
a ratio of two noisy small numbers), and not within 2 px of the mask edge.
Every thermal survivor also gets the CPR reading at its position.

### 3.6 Optical

For every survivor the ODE footprint service is asked which ShadowCam, LROC
NAC and Mini-RF products cover the position; counts and product ids are
recorded. Frames are listed, not inspected: inspection is the next decisive
action and is named per candidate under `unexcluded_systematics`.

## 4. Data

| Product | Source | State |
|---|---|---|
| **Diviner Polar Cumulative Products** — `PCP_AVG_TBOL_POL{N,S}_{SUM,WIN}_LTIM{nn}_240.TAB`, 240 m, 80–90°, average bolometric temperature per local-time bin per season. PDS3 detached label, `RECORD_BYTES = 59`, five ASCII columns `x y lon lat T_bol` (x, y are distances from the pole normalised by R = 1737.4 km), one header record, **empty bins omitted** → ~4.4 M rows, 262 MB each. Polar stereographic, `MAP_SCALE = 240 m/pix`, `MAP_RESOLUTION = 126.347 pix/deg`. `LRO:DLRE_CLOCTIME_MIN/MAX` give the bin (LTIM01 = 0.00–0.25 h). Williams et al. 2019 JGR 124, 2505. | PDS Geosciences `urn-nasa-pds-lro_diviner_derived1/data_derived_pcp/diurnal/ltim/pol{n,s}/`; URLs resolved through ODE (`ihid=LRO&iid=DLRE&pt=PCP&results=f`, 912 products / 5328 files) | **read 2026-09-22** (§7) |
| Diviner per-channel polar gridded maps (channels 3–9) | — | **do not exist**; all 912 PCPs are AVG TBOL (§7). §3.2 is retained, not runnable |
| LOLA mapped permanently-shadowed raster `LPSR_65{N,S}_240M` (6420², LSB_INTEGER 16-bit, DN ±20000 → 1/0, `LINE_PROJECTION_OFFSET = 3209.5`) | `lro-l-lola-3-rdr-v1/lrolol_1xxx/extras/illumination/release_2014/img/` and `.../illumination/img/lpsr_65{n,s}_240m_201608.*` (the 2016-08 re-release ODE indexes as LRO/LOLA/GDRPSR) | both routes answered 200 |
| Mini-RF S-band mosaics / CDRs (CPR, S1) | `lro-l-mrflro-5-global-mosaic-v1`, `lro-l-mrflro-4-cdr-v1` | host reached; polar product naming learned by the probe |
| ShadowCam (PDS4, 1.7 m) | `pds.shadowcam.im-ldi.com`; ODE footprint search | archive and SIS reached |
| LOLA / PGDA permanent-shadow rasters | `pgda.gsfc.nasa.gov/products/{54,78,90}`, `lrolol_1xxx/data/lola_gdr/polar/` | tried per run; thermal definition is the fallback |

The reader (`labels.py`) handles PDS3 detached and attached labels (MSB/LSB
integers with scaling, PC_REAL), PDS4 `Array_2D_Image` labels with the
cartography class, and both polar-stereographic and equirectangular
georeferencing; `origin_check_px` records how far the parsed projection
origin sits from the raster centre, because a mis-read pixel convention
shows up there first.

## 5. Verdict vocabulary

| Verdict | Meaning |
|---|---|
| `NO_DIURNALLY_INVARIANT_SURVIVOR` | PSR-interior pixels screened, none survives every rule of §3.7 — a count at the stated floor, **not an occurrence limit, never written up** |
| `DIURNALLY_INVARIANT_CANDIDATES (n)` | n pixels whose floor temperature exceeds their local annulus while their winter diurnal amplitude and summer−winter offset are both below threshold — pending the optical/radar vet |
| `NO_DATA_REACHED` | no Diviner polar product was read — an access statement with every route and the inventory attached; never a sky statement |
| `NO_ANISOTHERMAL_SURVIVOR` | pixels screened, none survives every rule — a count at the stated floor, **not an occurrence limit, never written up** |
| `ANISOTHERMAL_CANDIDATES (n)` | n pixels survive, pending the optical/radar vet named per candidate |
| `; DEGRADED (…)` | what was missing: `single_season_product`, `no_radar_layer`, `psr_from_thermal_definition_only`, `mask_from_average_not_maximum`, a pole not screened |

## 6. Offline tests (no network anywhere)

### 6.1 The diurnal screen (`tests/test_crypt_diurnal.py`)

The closed-form bolometric mixing and its exact inverse; a **clean null** on
the un-injected synthetic bowl with its floor stated; the curvature-bias
second pass demonstrably reducing the measured scatter; an injected internal
source recovered as `candidate` with its area within a factor of three; a
source below the floor not flagged; each confounder tripping *its own* named
rule — summer-only → `seasonal`, day-bins-only → `diurnal`, a 5 × 5 patch →
`extended`, a row of eight → `stripe`, a listed site → `human_hardware`, a
holed pixel → not a candidate; a single-season cube, a missing PSR raster
and a wrong-shaped PSR raster each reported as *degraded* rather than
ignored; an empty cube → `NO_PSR_INTERIOR`, never a null. The windowed
injection–recovery table reproduces the whole-map table exactly, and
`crop_cube` preserves the georeference. A **real-format** PCP ASCII table is
read and rasterised at either line-axis sign and either grid origin with
zero collisions, and the wrong registration is visible in the residual. The
PSR mask is registered off label offsets — the test builds a raster whose
pole is deliberately *not* at the array centre and checks the centre crop
misses it. The bin index is built from an ODE listing, a bin present in only
one season is not offered, and the verified URL template is the fallback
when ODE is unreachable. End to end through a scripted archive serving the
real formats: the injected source is a candidate and the verdict is
`DIURNALLY_INVARIANT_CANDIDATES`; the clean archive gives
`NO_DIURNALLY_INVARIANT_SURVIVOR`; the empty archive gives
`NO_DATA_REACHED` with zero screened pixels; the tables are deleted after
rasterisation.

### 6.2 The anisothermal screen (`tests/test_crypt.py`, 40 tests)

The brief's numbers; the band model's single-temperature identity; the
injected 300 m² / 300 K source recovered as `candidate` in both seasons with
season-independent excess radiance; the summer-only rim → `seasonal`; a 5×5
warm block → `extended`; a row of six → `stripe`; three observations →
`low_count`; a summer-only product → `single_season` + degraded; a listed
site → `human_hardware`; channel 6 and 8 raised without 7 →
`inconsistent_spectrum`; a ×20 seasonal source → `unstable`; the clean map →
zero candidates; the sensitivity table recovers 1000 m² and misses 10 m²;
PDS3 16-bit scaled round trip with NaN and the polar georeference to
10⁻⁶ px; a PDS4 label with cartography; IIS and Apache listings; name and
label-text classification; selection preferring the unsplit 240 m product;
the ODE helpers; probe → acquire → screen → assess through a scripted archive
serving synthetic PDS3 products (candidate recovered, `DEGRADED (no_radar_layer,
psr_from_thermal_definition_only)`), the clean archive → `NO_ANISOTHERMAL_SURVIVOR`,
the empty archive and the listing-without-pixels archive → `NO_DATA_REACHED`;
the radar screen and its three kills; mask re-sampling; sharding; the CLI.

## 7. Runner log

*(filled per run; every entry names what the runner taught and what changed.)*

- **Not yet dispatched.** The first run's job is the inventory: `probe.json`
  commits every product name in `lrodlr_1002`, the axis counts of the
  classifier, the labels it read, and the selection it would make. If the
  selection is empty the verdict is `NO_DATA_REACHED` and the patterns in
  `config/crypt.yaml` are corrected from the committed names.
