# CRYPT — artifacts in lunar permanent shadow (S55)

*Built 2026-09-21. Package `src/seti/crypt/`, CLI `crypt --stage
{probe,acquire,screen,assess,all} --shard i/n`, workflow `crypt.yml`, config
`config/crypt.yaml`, results `results/crypt/`. The claim and the data ledger
come from `docs/necrofrontier.md` §2.XVII (S55) and §3; the probe of
2026-09-13 (`results/necrofrontier/probe.json`) established which hosts
answer the runner. Nothing below is a measurement of the sky until a run has
committed `results/crypt/summary.json`; §7 records what each run taught.*

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

### 3.5 Radar

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
| Diviner level-3/4 polar products (240 m, 80–90°, channels 3–9 and T_bol; max/avg/min/count; summer/winter/all) | PDS Geosciences `lro-l-dlre-4-rdr-v1/lrodlr_1002/data/{pcp,prp,gdr_l3,gdr_l4}` (PDS3 `.lbl`+`.img`, or the PDS4 mirror `urn-nasa-pds-lro_diviner_derived*`) | host reached 2026-09-13; **naming learned by the probe** (§7) |
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
| `NO_DATA_REACHED` | no Diviner polar raster was read — an access statement with every route and the inventory attached; never a sky statement |
| `NO_ANISOTHERMAL_SURVIVOR` | pixels screened, none survives every rule — a count at the stated floor, **not an occurrence limit, never written up** |
| `ANISOTHERMAL_CANDIDATES (n)` | n pixels survive, pending the optical/radar vet named per candidate |
| `; DEGRADED (…)` | what was missing: `single_season_product`, `no_radar_layer`, `psr_from_thermal_definition_only`, `mask_from_average_not_maximum`, a pole not screened |

## 6. Offline tests (`tests/test_crypt.py`, 38 tests, no network)

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
