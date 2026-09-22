# GRAVE — a technological extinction in Earth's own sedimentary record

**Signature S56** (`docs/necrofrontier.md` §2). Code `src/seti/grave/`, config
`config/grave.yaml`, workflow `.github/workflows/grave.yml`, results
`results/grave/`, CLI `python -m seti.cli grave --stage ...`.

---

## 1. The claim being tested

Schmidt & Frank 2018 (*Int. J. Astrobiology* 18, 142 — the "Silurian
hypothesis") asked whether a prior industrial civilisation on Earth would be
detectable in the geological record, and answered with generic markers: carbon
isotope excursions, sedimentation-rate changes, synthetic organics, plastics,
a ¹⁴C-dead carbon pulse. Every one of those is degenerate with a natural
cause, which is why the paper reads as a cautionary essay about the PETM
rather than as a search. Wright 2018 (*Int. J. Astrobiology* 17, 96) made the
complementary argument that the only durable technosignature is a physical
artifact, not a chemical trace.

GRAVE asks the sharper version of the question, the one that is *not*
degenerate:

> Does any horizon in the sedimentary record carry an excess of the
> **fission-product mass vector** — the two-humped ²³⁵U yield pattern, Zr–Mo–Ru
> on the light peak and Cs–Ba–La–Ce–Pr–Nd–Sm on the heavy peak, with the
> Ag–Cd–Te valley between and a cliff at Gd — that **no mixture of natural
> sedimentary reservoirs can build**, and does it recur at one stratigraphic
> level across *independent sections*, the way the K–Pg iridium does?

and its companion question:

> Does any horizon carry **refined particulate** — platinum-group metal in
> catalyst proportions rather than chondritic ones, Ta without Nb, W without
> its hydrothermal partners — that no sedimentary concentration mechanism
> produces?

The fission vector is the strong form because it is *made of ordinary
elements in an impossible ratio*. Nothing in igneous petrology, hydrothermal
chemistry, redox drawdown or impact ejecta delivers Ru and Rh enriched by a
factor of 10⁴·⁸ relative to Nd while leaving Hf, Th, U and Pb untouched. The
pattern is a nuclear fact, not a geochemical one.

---

## 2. Physics: the fission vector in a *sedimentary matrix*

### 2.1 Reuse of FALLOUT's yield physics

`seti.fallout.yields` carries the ²³⁵U cumulative mass-chain yields
(ENDF/B-VII.1 / JEFF-3.1.1, England & Rider 1994 tabulation), per 100 fissions,
with the element each chain sits on at a chosen **decay horizon**. That module
is pure and is reused verbatim; GRAVE adds no nuclear data of its own. The
horizon is **1 Myr**, the same choice FALLOUT makes and for the same reason:
it is the pattern that outlives its makers. At 1 Myr, Cs-137 and Sr-90 (30 yr)
have long since become Ba and Zr, Tc-99 (2.1 × 10⁵ yr) has become Ru — so
there is no technetium in this search at all — while Cs-135, Zr-93, Pd-107 and
I-129 are still sitting on their intermediates. The chains sum to 200.0 per
100 fissions.

### 2.2 The matrix decides which elements carry the signal

FALLOUT folds the yields against *solar* abundances. GRAVE folds them against
*shale*, and the ranking is completely different, because the crust is
~10⁵ times poorer in the platinum-group and chalcogen fission products than in
the rare earths.

Multiplying each chain yield by the atomic mass gives a composition by weight,
`Φ`. Normalised so that amplitude `a = 1` adds one PAAS-Nd (33.9 ppm) — i.e.
doubles the shale's neodymium — the relative enrichment each element suffers
is `(Φ_X / PAAS_X) / (Φ_Nd / PAAS_Nd)`:

| element | Φ ppm at a=1 | PAAS ppm | rel. to Nd | dex | PAAS value from UCC |
|---|---|---|---|---|---|
| Rh | 3.54 | 6.0e-5 | 5.90e4 | **+4.77** | yes |
| Ru | 20.0 | 3.4e-4 | 5.89e4 | **+4.77** | yes |
| Pd | 1.92 | 5.2e-4 | 3.69e3 | **+3.57** | yes |
| Te | 3.25 | 3.0e-3 | 1.08e3 | **+3.03** | yes |
| Mo | 26.8 | 1.0 | 26.8 | **+1.43** | no |
| Se | 0.43 | 0.09 | 4.77 | +0.68 | yes |
| Cs | 20.0 | 15 | 1.33 | +0.12 | no |
| Sm | 6.27 | 5.55 | 1.13 | +0.05 | no |
| Pr | 9.35 | 8.83 | 1.06 | +0.03 | no |
| **Nd** | **33.9** | **33.9** | **1.00** | **0.00** | no |
| I | 1.31 | 1.4 | 0.94 | −0.03 | yes |
| Eu | 1.00 | 1.08 | 0.93 | −0.03 | no |
| Cd | 0.071 | 0.09 | 0.79 | −0.10 | yes |
| Ag | 0.038 | 0.053 | 0.72 | −0.15 | yes |
| La | 10.1 | 38.2 | 0.26 | −0.58 | no |
| Ce | 19.2 | 79.6 | 0.24 | −0.62 | no |
| Zr | 38.0 | 210 | 0.18 | −0.74 | no |
| Y | 4.77 | 27 | 0.18 | −0.75 | no |
| Sb | 0.040 | 0.4 | 0.10 | −1.00 | yes |
| Sn | 0.141 | 4.0 | 0.035 | −1.45 | no |
| Ba | 20.2 | 650 | 0.031 | −1.51 | no |
| Rb | 3.76 | 160 | 0.024 | −1.63 | no |
| Gd | 0.097 | 4.66 | 0.021 | −1.68 | no |
| Sr | 3.53 | 200 | 0.018 | −1.75 | no |
| Tb | 0.0018 | 0.774 | 0.0023 | −2.63 | no |
| Dy | 1.5e-4 | 4.68 | 3.0e-5 | −4.50 | no |

(`summary.json["fission_discriminants"]` carries this table as the code
computes it, so it is never quoted from here.)

Three consequences the brief's ratios express, all of them read off the table
rather than asserted:

* **[Nd/Ba] ≫ 0.** Fission makes Nd and Ba at comparable *atom* yields (20.7
  and 13.0 per 100 fissions) but shale already holds 650 ppm Ba against 34 ppm
  Nd, so the vector raises Nd 1.5 dex more than Ba. Every natural REE carrier
  — monazite, apatite, the detrital fraction — comes with its own Ba budget or
  none at all; barite comes with Ba and no Nd. A coupled `Nd` rise with a flat
  `Ba` is not a mineral.
* **[Eu/Nd] < 0, slightly.** Fission is nearly flat across Pr–Nd–Sm–Eu
  (+0.03, 0.00, +0.05, −0.03 dex) — a *mid-REE plateau with a cliff at Gd*
  (−1.68) and a second cliff into La–Ce (−0.6). Apatite's MREE bulge is
  smooth and carries Gd–Tb–Dy with it; a hydrothermal fluid gives a strong
  *positive* Eu anomaly; monazite gives La–Ce–Nd together with 5 wt% Th. None
  of them has a cliff at Gd.
* **[Mo/Zr] > 0.** Both are light-peak products, but shale's Zr (210 ppm,
  mostly zircon) swamps its Mo (1 ppm), so the vector shows +1.43 dex on Mo
  against −0.74 on Zr. And fission adds **no Hf** with its Zr: zircon has
  Zr/Hf ≈ 40 and always brings Hf, so a Zr excess at constant Hf is not
  detrital heavy-mineral concentration.

### 2.3 The absent partners are the discriminant

At the 1 Myr horizon no fission chain sits on **Hf, Th, U, Pb, Ta, W, Ir, Pt,
Os, Au, Ni, Co, V or Mn**. That is what separates the vector from every
process in the kill list, and it is derived from the yields, not asserted:

| a rise in | natural carrier brings with it | fission brings |
|---|---|---|
| Zr | Hf (Zr/Hf ≈ 40, zircon) | nothing |
| La, Ce, Nd | Th (monazite, 5 wt%) | nothing |
| Mo | U and V (anoxic drawdown), or Mn and Co (Fe–Mn shuttle) | nothing |
| Ba | Sr and S (barite) | nothing |
| Ru, Rh, Pd | Ir, Os and Pt (chondrite, ultramafic detritus) | nothing |
| Te | Mn, Co, Pt (Fe–Mn crust) | nothing |

`vectors.fission_discriminants()` tabulates this from the yield table at run
time, and `summary.json` carries it, so a change of decay horizon
automatically changes the discriminants rather than leaving a stale claim in
prose.

---

## 3. Novelty position

The `necrofrontier` occupancy audit (`docs/necrofrontier.md` row *g9
terrestrial record*) scores S56 **unoccupied**: three framing papers
(Schmidt & Frank 2018; Wright 2018; and the Silurian-hypothesis commentary
that followed), **no fission-vector test and no database search**. Nothing
prior tests a *multi-element nuclear yield pattern* against a sedimentary
geochemical compilation.

Adjacent work, and why it is not this:

* **Oklo** (Gabon, 2 Ga natural fission reactors; Kuroda 1956 predicted them,
  Bodu et al. 1972 found them). Oklo is the **positive control for how fission
  products are recognised in rock**: depleted ²³⁵U, and Nd/Ru/Sm isotope
  anomalies in the ore. It is also the standing natural explanation that any
  candidate must beat — a natural reactor is a fission source with no
  technology behind it. Oklo's signature is *isotopic*; GRAVE's is *elemental*,
  because elemental abundances are what the compilations carry at scale.
* **The K–Pg iridium anomaly** (Alvarez et al. 1980). The methodological
  parent: a single trace element, at one stratigraphic level, in dozens of
  independent sections. GRAVE copies its epistemics exactly — the cross-section
  recurrence, not the amplitude, is what makes a layer believable — and uses
  the impact class as a **built-in positive control** (§6.3).
* **Cosmogenic-nuclide searches** (⁶⁰Fe and ²⁴⁴Pu in Fe–Mn crusts, Wallner et
  al. 2016, 2021; Knie et al. 2004). These look for supernova debris, i.e. for
  the r-process partners; GRAVE's logic is the mirror image — fission products
  *without* the r-process partners.
* **Anthropocene marker work** (Waters et al. 2016; the Crawford Lake GSSP
  candidate): the same chemistry, but in the last 80 years, where the
  civilisation is not in question.
* **"Industrial-looking" spherules.** The IM1 episode (Loeb et al. 2024;
  Gallardo 2023; Desch & Jackson 2023, 2024, who traced the Fe–Be–La–U pattern
  to coal fly ash) is the reason the burden of proof here is extreme and the
  reason `provenance_flag` exists per sample.

---

## 4. The data, and what the runner actually found

### 4.1 What the brief asserted, and what was true

`results/necrofrontier/probe.json` established that `sgp-search.io` answers a
GitHub runner but that the REST path the brief asserted (`GET /api/v1/samples`)
**serves the single-page app, not data** — 623 bytes of HTML. `env.js` names
`https://archive.sgp-search.io` as the bulk side; it is another SPA. There is
no published API document.

The real interface, established from the SGP front end, the Macrostrat
integration and the call published with Stockey et al. 2024 (*Nature
Geoscience* 17, 667), is

```
POST https://sgp-search.io/api/frontend/post-paged
{"type": "samples", "count": N, "page": k,
 "filters": {"interpreted_age": [lo, hi]},
 "show": ["interpreted_age", "coord_lat", "mo", "alu", ...]}
```

whose response `rows` are keyed by **display names** ("interpreted age", "site
latitude", "Mo (ppm)"), not by the codes you send. The published `show` list
(Stockey et al.) is only the couple of dozen codes that paper used. The full
code list is not published anywhere.

### 4.2 So the probe *learns* the schema

`stage_probe` does not assume the schema. It:

1. fetches `env.js` and the index of every configured host, extracts the app
   bundle path, downloads the bundle and scans it for `/api/...` paths and for
   `{value: "<code>", label: "<display name>"}` pairs;
2. issues the published Stockey body as a base call and records exactly what
   came back (status, row count, row keys, the response's own keys);
3. tries each `type` variant (`samples`, `nhhxrf`);
4. takes an anchor request (`interpreted_age`, `coord_lat`) as the baseline row
   key set, then sends **one two-row request per candidate field code**
   (~220 of them: every element symbol, every metadata field, the codes found
   in the bundle). A code is *accepted* only when the response rows carry a new
   key; it is *rejected* when the request errors; it is *silently dropped* when
   the API answers 200 and ignores it.

The result is a ledger — `accepted_codes` (code → display name),
`rejected_codes`, `silently_dropped`, `api_paths`, `label_value_pairs` — that
the acquire stage then uses as its `show` list. Every request is recorded with
URL, status, byte count and elapsed time. Nothing is assumed; nothing is
fabricated; a source that returns no rows is `NO_DATA_REACHED` **for that
source**, which is an access statement and never a statement about the rock.

The probe filters on one configured age window (`sgp.probe_age_window`,
`[0, 4000]` Ma by default). This is deliberate and was a bug fix: a code is
accepted only if a row comes back carrying its column, so a narrow window that
happened to hold no samples would have silently rejected the entire schema.

### 4.2a What the first probe run actually learned (run 35738860553)

The first runner probe (2026-09-22, results committed to `results/grave/probe.json`)
settled the schema:

* The endpoint works. The anchor call returns rows keyed `sample identifier`,
  `sample original name`, `interpreted age`, `site latitude` — note that the
  sample identifier and its original name come back **whether or not you ask
  for them**.
* **93 field codes were individually accepted.** The trace panel is real:
  `se rb sr y zr nb mo pd ag cd in sn sb te cs ba la ce pr nd sm eu gd tb dy
  ho er tm yb lu hf ta w re os ir pt au hg tl pb bi th u`, with units in the
  display name (`Pd (ppb)`, `Pt (ppb)`, `Os (ppt)`, `Mn (ppm)`, `P (ppm)`,
  `Al (wt%)`). Plus `toc tic tot_c loi fe_hr_fe_t fe_py_fe_hr` and the full
  metadata set (`section_name site_type country state_province coord_long
  height_meters strat_name lithology_name/type/text/comp basin_type basin_name
  environment_bin meta_bin interpreted_age_notes max_age min_age data_source
  collector`).
* **`ru` and `rh` are refused (400).** The two strongest discriminants in the
  whole vector (+4.77 dex each) are not in the database. The light peak
  therefore rests on **Mo (+1.43), Pd (+3.57) and Te (+3.03)**, with Zr
  (−0.74) and Y (−0.75) as the *negative* half of the shape. `as` and `i` are
  also refused although `ge`, `se`, `in` and `te` are served, so the code for
  those two is probably not the bare symbol; alternative spellings are in the
  candidate list for the next pass.
* **There is no reference, DOI or analytical-method field.** `ref_short`,
  `ref_long`, `reference`, `ref_doi`, `doi`, `ana_method` all 400;
  `analytical_method` and `original_name` are silently dropped. Per-sample
  provenance therefore rests on `data_source`, `collector` and `site_type`,
  and the doc says so rather than promising a citation the API cannot give.
* **The published Stockey body itself 400s**: `filter error: fe_t_al is not a
  valid attribute in this search type` (and `strat_name_long` likewise). One
  invalid code fails the entire request, so acquire sends **only** the probe's
  accepted list — the union with a config body would have poisoned every page.
* The bundle names the service's own listings (`/api/v1/post/attr`,
  `/api/v1/get/info/samples`, `/api/defs/lithologies`), which the probe
  requests directly — and every one of them answers **200 with the SPA's own
  index** (623 bytes of HTML). They are client-side routes, not API endpoints;
  the probe rejects an HTML body as "not a listing". Notably the bundle does
  **not** contain `/api/frontend/post-paged` even though that path serves: the
  published endpoint and the one the current app uses are not the same.

The second probe (run 35739776468) added the two numbers that size the search:

* the response carries its own **`count`**, and for the `[0, 4000]` Ma filter
  it is **114,688** — that is the corpus this channel screens;
* a page asked for 5,000 rows returns **5,000**, so the service caps nothing
  below that: 36 age bins need of order sixty requests. Each page records the
  service's count and each bin reports whether it got all of it (`complete`),
  because a bin that ends short is a bug to find, not a sparse record.
* `ars` → `As (ppm)` was accepted on the retry, taking the panel to 94 codes.
  `ru`, `rh` and `i` are refused under every spelling tried (`ru_`,
  `ruthenium`, `rh_`, `rhodium`, `iod`, `iodine`, `i_`), so ruthenium and
  rhodium are genuinely absent from SGP rather than differently named.

### 4.3 The other two sources

* **EarthChem Portal.** The documented REST service
  (`portal.earthchem.org/restsearchservice`, GET with
  `searchtype=count|rowdata|distinctitems`, `outputtype=json`, `minage`/`maxage`
  in Ma, `level1..level4`/`material`/`keyword`, `startrow`/`endrow` pages of at
  most 50) **answers 404 from Apache for every parameter spelling** (run
  35738860553). The service moved or was retired; the landing pages answer,
  which is why the earlier `necrofrontier` probe scored EarthChem as reachable
  — reaching a web page is not reaching a search. The probe now walks an
  **endpoint ladder** over ten candidate bases (`ecp.iedadata.org` http and
  https, `search.earthchem.org`, `api.earthchem.org`, `ecl.earthchem.org`, the
  portal's own `/api/search`), records what each served, and uses the first
  that answers with data. A `DEGRADED_SOURCE (earthchem:...)` prefix on the
  verdict means this ladder found nothing — an access fact, recorded as such.
* **GEOROC** via the Göttingen Dataverse (`data.goettingen-research-online.de`,
  subtree `digis`): `api/search` lists datasets, `api/datasets/:persistentId`
  lists files, `api/access/datafile/<id>?format=original` serves a CSV with a
  citation preamble before the header row. GEOROC is igneous. It is the
  **volcanic-ash and tephra reference population**, never a candidate source;
  `is_reference_only` marks its rows and they are excluded from every candidate
  count and from the age stack.

### 4.4 Column resolution at run time

Three sources, three naming conventions, all resolved by
`acquire.resolve_columns` against regexes rather than hard-coded lists:
`"Mo (ppm)"`, `"mo"`, `"MO(PPM)"`, `"AL2O3(WT%)"`, `"FEOT(WT%)"` all land on
the right element with the right scale to ppm (oxides via the mass fraction of
the cation; ppb → ×10⁻³; wt% → ×10⁴). Where a bare code carries no unit, the
unit its class implies is assumed (wt% for majors and organic carbon, ppm
otherwise) and **the assumption is recorded** in
`acquisition.json["resolution"]["assumed_units"]` next to the observed median
of each element, so a wrong guess shows up as a median three orders of
magnitude off PAAS instead of quietly biasing the fit. GEOROC ages in years
are converted to Ma when the median exceeds 10⁵.

---

## 5. Method

### 5.1 The statistic: a mixture, not a baseline

The wrong way to do this is to divide by PAAS and look for outliers; every
sediment is a *mixture*, and the mixture is what a naive normalisation calls
anomalous. So each sample's concentration vector `c` (majors included, so that
Al pins the detrital fraction, Ca the carbonate, P the apatite, Mn and Fe the
oxide shuttle, Ti the mafic/ash component) is fitted as a **non-negative
combination of natural reservoirs**

```
c ≈ Σ_i f_i R_i ,  f_i ≥ 0
```

and then again with the fission vector as one extra non-negative column:

```
c ≈ Σ_i f_i R_i + a Φ ,  f_i ≥ 0, a ≥ 0
```

The statistic is `fission_lr = ½ (χ²_natural − χ²_natural+fission)`, a
log-likelihood ratio for one extra non-negative parameter. It is ≥ 0 by
construction (the model is nested). The fit is a weighted NNLS in linear space
(weights `1/(σ c ln10)`, which is the linearisation of a log-space χ²),
refined by an L-BFGS-B minimisation of the exact log-space χ² whenever the
linear-space LR already exceeds `refine_lr`.

The natural family is deliberately **the list of the kills**, each vector
carrying its citation in `references.RESERVOIR_SOURCES`:

| reservoir | what it is the kill for | source |
|---|---|---|
| `paas` | Post-Archaean Australian Shale — the detrital baseline | Taylor & McLennan 1985; McLennan 2001 |
| `ucc_felsic` | felsic provenance | Rudnick & Gao 2003; PGE Peucker-Ehrenbrink & Jahn 2001 |
| `morb_mafic` | mafic provenance, ultramafic detritus | Gale et al. 2013; PGE Bezos et al. 2005 |
| `zircon` | heavy-mineral (Zr–Hf–Y–HREE) concentration | Hoskin & Schaltegger 2003; Belousova et al. 2002 |
| `monazite` | detrital LREE + Th | Williams et al. 2007 |
| `apatite` | phosphorite / bioapatite MREE bulge | Altschuler 1980; Emsbo et al. 2015 |
| `carbonate` | dilution by limestone | Turekian & Wedepohl 1961 |
| `authigenic` | **redox drawdown** — max(black shale − PAAS, 0) on the Tribovillard suite | Ketris & Yudovich 2009; Tribovillard et al. 2006 |
| `femn_oxide` | the Fe–Mn particulate shuttle (Mo, Te, Pt, Ce *without* U) | Hein & Koschinsky 2014 |
| `hydrothermal` | SEDEX/VMS barite + sulfide, positive Eu anomaly | Large et al. 2005; Hannington 2014; Michard 1989 |
| `chondrite` | **impact ejecta** | Lodders 2003; McDonough & Sun 1995 |
| `rhyolite` | distal tephra / volcanic ash | Le Maitre 1976; GEOROC rhyolite median |

Because the natural family already contains every process the brief lists as a
kill, *a sample that needs the fission column is a sample none of those
processes can build*. That is the whole design.

### 5.2 The error model is measured, not quoted

The compilations carry no uncertainties at all. So the error model **is** the
observed scatter: for each element, 1.4826 × MAD of `log10(observed / natural
model)` over the control population (samples away from every boundary window),
floored at a per-class nominal (0.05 dex majors, 0.10 trace, 0.20 ultra-trace,
in quadrature with a 0.05 dex systematic). An element whose scatter could
**not** be measured — fewer than 30 control residuals, which is the case for
exactly the rare and decisive elements Ru, Rh, Pd, Te, Ir — is given
`unmeasured_floor_dex` (0.30) rather than its nominal class sigma, because an
unmeasured scatter is not a small scatter. This is the FALLOUT lesson
(`docs/fallout.md` §4a) applied where there are no quoted errors at all.

### 5.3 The threshold is set by two nulls, and the report says which bound it

`threshold = max(lr_min, q_0.999 shuffled, q_0.999 control population)`.

* The **shuffled null** takes control samples, shuffles each one's
  PAAS-normalised *trace* enrichments among the trace slots (majors stay in
  place so the reservoir fractions stay physical) and refits. That keeps the
  per-sample amplitude structure and the element count and destroys only the
  alignment with a real pattern.
* The **control-population null** is the empirical `fission_lr` distribution of
  the same lithology **away from every boundary window** — which is exactly the
  comparison the brief asks for. The shuffled null cannot carry the correlated
  deviations real sediments have (an REE set that rises together, a redox suite
  that rises together); the control population does, at the price of being
  contaminated if a residue exists off-boundary too. Taking the higher of the
  two is conservative in the direction that matters.

`screen.json` carries both (`shuffled_null`, `control_population_null`) and
`threshold_bound_by` names which one set the threshold. If the control
population lifts the threshold above every candidate, that is the answer and
the funnel shows it.

### 5.3a The three named ratios, reported explicitly

The likelihood ratio tests the whole vector at once, but the brief and the
sibling FALLOUT channel speak in three ratios, so every vetted sample carries
them in `brief_ratios`, three ways: `obs_dex` = log₁₀(X/Y) of the sample;
`vs_paas` = the same against PAAS's own ratio; and `vs_model` = against *that
sample's* best natural mixture, which is the discriminating one — it asks
"given everything natural that can be in this rock, is the ratio still off?".
Beside them sits `fission_dex`, what the vector alone imposes, computed from
the yields rather than quoted: **[Nd/Ba] +1.51, [Eu/Nd] −0.03, [Mo/Zr] +2.17**.

### 5.4 The kills, each a named veto and a counter

Every above-threshold sample goes through all of them; `summary.json["vetoes"]`
counts each. The kills read **every element column the sample carries**, not
only the ones that entered the mixture design — Ir, Os, Ru, Rh, Pt and Pd are
reported on a per-cent minority of analyses, and a design cut on measurement
count would delete the impact test and the refined-catalyst test together.

| veto | what it rejects |
|---|---|
| `insufficient_panel` | fewer than `min_elements` (8) measured elements |
| `detection_limit_driver` | the element driving the preference is at a repeated reporting floor (a value repeated ≥ 20 times within 50 % of the column minimum) — a limit, not a measurement |
| `unexplained_by_all_reservoirs` | reduced χ² still > 4 with fission included: the sample is odd, but not odd in *this* way. Classed `UNEXPLAINED_BY_ALL_RESERVOIRS`, never a candidate |
| `single_element_driver` | leave-one-out: dropping one element takes the LR below threshold. **A one-element anomaly is a rejection here**, the opposite of TAILINGS |
| `peak_incoherent` | fewer than 2 of {Cs, Pr, Nd, Sm, Eu} *and* 1 of {Zr, Mo, Ru, Rh, Pd, Te} sit ≥ 2σ above the natural model. The signature is two humps or it is nothing |
| `redox_conditioned` | the sample is anoxic (U or V enrichment factor ≥ 3, or TOC ≥ 2 %) and the driver is a redox-sensitive element (Mo, U, V, Cd, Ag, Re, Se, Tl) |
| `femn_shuttle` | Mn enrichment factor ≥ 5 and the driver is one of the shuttle's own (Mo, Te, Pt, Ce, Co, Ni) |
| `heavy_mineral_zr_hf` | Zr drives, and Zr/Hf is in the natural 25–60 range |
| `volcanic_ash` | **all four** of Zr, Hf, Nb, Ta enriched ≥ 2× together, with Zr/Hf in 25–60 *and* Nb/Ta in 5–40 — the HFSE arrived as glass, in crustal proportion — and the driver is an element the tephra itself carries (Zr, Hf, Nb, Ta, Th, Y, Rb, Cs, Ba, LREE–Eu, U). **This kill cannot fire on a real fission residue**, which is the whole reason it is written on coherence rather than on any one element: fission gives Zr with *no* Hf (Hf's mass sits in the yield valley) and has no path to Ta at all, so the four cannot rise together. A sample that brings Hf and Ta along at crustal ratio has brought a rock, not a reactor |
| `monazite_th` | an LREE drives, and Th enrichment factor ≥ 3 |
| `hydrothermal_ba` | Ba drives, and Ba enrichment factor ≥ 5 |
| `impact_pge` | the PGE panel classes as impact or ultramafic and a PGE drives |

Enrichment factors are always Al-normalised, `(X/Al)_sample / (X/Al)_PAAS`, so
that dilution by carbonate or silica cannot masquerade as depletion.

A note on the ash kill, because the suite measured it rather than assuming
it: a *plain* distal tephra never reaches the vet at all. `rhyolite` is one
of the twelve reservoirs, so the mixture absorbs an ash bed outright — LR = 0,
reduced χ² below 4. The `volcanic_ash` veto exists for the harder case, an
anomaly that survives the fit *on a horizon that also carries glass*, where
the heavy-peak elements may simply be glass-borne. The suite holds the same
sample against itself with and without the glass and asserts that
`volcanic_ash` is the only entry that changes.

Two kills are *not* automatic rejections but are reported per sample, because
they cannot be settled from a database row:

* **`provenance_flag`** — the site type matches `core|drill|well|borehole|mine|
  quarry|tailing`. Drilling mud carries barite (Ba), drill-bit tungsten carbide
  carries W and Co, casing carries Cr–Ni–Mo. Modern industrial contamination is
  the single most likely explanation of any positive, and it can only be
  settled by re-sampling, so the flag rides with every candidate and
  `docs/grave.md` says so here rather than pretending the flag is a veto.
* **the reference** — the original publication for each analysis, carried
  through to `candidates.json`, because "one laboratory, one campaign" is a
  contamination hypothesis in itself.

### 5.5 The refined-particulate test

Independent of the fission fit, every sample with a panel is classified:

* **PGE, anchored on Ir.** Impact ejecta carry the PGE in CI proportions with
  Ir; ultramafic detritus carries the IPGE (Os, Ir, Ru) with Pd/Ir well below
  chondritic, because chromite and olivine hold them; a refined catalyst
  carries Pt–Pd–Rh with Ir and Os at crustal background; fission carries
  Ru–Rh–Pd with neither Ir nor Pt; the Fe–Mn shuttle carries Pt alone with Mn.
  Classes: `impact`, `ultramafic`, `refined_pge`, `fission_like_pge`,
  `femn_pt`, `pge_background`, `pge_anomalous_unclassified`. **A Pt/Pd-only
  panel is always `pge_anomalous_unclassified`** — without Ir it cannot exclude
  impact, and saying so is the honest answer.
* **Alloy.** Ta enriched ≥ 10× with Nb/Ta below the natural 5–40 is
  `refined_ta` (nature never separates the twins); W enriched ≥ 20× with Sn,
  Mo and Bi all below 3× is `refined_w` (a hydrothermal W deposit brings them).

### 5.6 The age stack — the part that decides everything

A single anomalous sample is **never** a candidate. The unit is the *section*,
not the sample: a section with ≥ 1 candidate counts once however many aliquots
it contributed, because ten analyses of one core are one opportunity for
contamination, not ten. A section is the named section; failing that, the
0.1° site cell.

The test is conditional, so the window cannot contaminate its own null. Given
that the whole corpus produced `C` candidate sections out of `S` sections, and
that `S_w` of those sections carry a sample inside a boundary window, the count
in the window under "candidates fall where sections are, regardless of
stratigraphic level" is `Hypergeometric(S, C, S_w)` and `p = P(X ≥ k)`.

*(The first implementation estimated a background rate from the whole corpus —
including the window's own candidates — and tested against
`Poisson(rate × S_w)`. That both inflates the null and loses power exactly
when every candidate sits at one level, which is the case the channel exists
to detect. It was replaced.)*

**The catalogue is searched, so the p-value has to be paid for.** Thirteen
boundary windows are tested at once. A per-window threshold of 0.01 lets a
spurious cluster appear *somewhere* in the catalogue about 12 % of the time
(1 − 0.99¹³ = 0.122) — which is precisely the error this channel cannot afford,
because the one thing that promotes a per-sample anomaly to a boundary-level
claim is the stack. So `p_hypergeom` is **Holm-corrected** over the windows
that were testable at all (those holding ≥ 1 sampled section; a boundary the
corpus never sampled was never a test, and inflating m with it would only cost
power), and the promotion rule reads the corrected `p_family` at a
**family-wise** `cluster_p` = 0.05. Both numbers are reported per window, and
`summary.json` carries `multiple_testing: "holm"`, `n_boundaries_tested` and
`cluster_p_is_family_wise: true`. Net of the correction this is *stricter*
than the rule it replaces: FWER 0.05 instead of ≈ 0.12.

Both positive controls survive it with margin. The injected six-section K–Pg
cluster: p_raw = 3.97 × 10⁻⁴, p_Holm = 5.16 × 10⁻³ over 13 tested windows. The
chondritic-iridium impact control: p_raw = 7.76 × 10⁻⁴, p_Holm = 7.76 × 10⁻³
over 10. A window that clears 0.01 raw but not the correction — two candidate
sections out of ten sampled, against five candidate sections in a 300-section
corpus, p_raw = 9.5 × 10⁻³, p_Holm = 0.067 over 7 — is held at
`multi_section_at_background_rate`, and the suite asserts exactly that case.

Statuses: `no_candidate`; `single_section` (**always** — one section is the
contamination hypothesis, not a find); `STRATIGRAPHIC_CLUSTER` (≥ 2 candidate
sections and `p_family` < `cluster_p` = 0.05 family-wise);
`multi_section_at_background_rate`.

Boundary windows (GTS2020 ages; half-widths are the age-model uncertainty of
SGP interpreted ages near each boundary, not the duration of the event):
Ediacaran–Cambrian 538.8 ± 3, end-Ordovician 444.5 ± 2, Kellwasser 372.2 ± 2,
Hangenberg 358.9 ± 2, Capitanian 259.5 ± 2, end-Permian 251.9 ± 2, Carnian
233.0 ± 2, end-Triassic 201.4 ± 1.5, Toarcian OAE 183.0 ± 1.5, OAE2 93.9 ± 1,
**K–Pg 66.0 ± 1**, PETM 56.0 ± 1, Eocene–Oligocene 33.9 ± 1.

---

## 6. Stages, outputs, verdicts

### 6.1 Stages

| stage | what it does | writes |
|---|---|---|
| `probe` | learns each service's real shape (§4.2) | `probe.json` |
| `acquire` | pages every age bin / window; resolves columns; canonicalises to one ppm column per element | `acquisition.json`, `data/<source>_canonical.csv` (artifact) |
| `screen` | design, detection-limit mask, error floors, shuffled null, mixture fit on every sample, full vet above threshold, refined classes on every panel | `samples.csv`, `candidates.json`, `refined.json`, `screen.json` |
| `assess` | the age stack, the impact positive control, the refined stack, the funnel | `summary.json`, `REPORT.md` |

The workflow runs `probe` **before** the offline test gate — the probe asserts
nothing about the sky, so a schema the runner has learned is worth keeping even
from a run whose detector gate then fails. Everything downstream of the probe
is gated.

### 6.2 Verdict vocabulary (`summary.json["verdict"]`)

| verdict | meaning |
|---|---|
| `NO_DATA_REACHED` | no source produced a canonical row. An **access** statement; never a statement about the rock |
| `DEGRADED_SOURCE (...)` | prefix: an enabled source failed, or a metadata role is missing |
| `NO_FISSION_VECTOR` | samples were screened; none survives the vet. **A count, not an occurrence limit**, and not written up (`CLAUDE.md`) |
| `FISSION_VECTOR_SINGLE_SECTION` | survivors exist but none recurs across sections at a boundary — the contamination hypothesis stands |
| `FISSION_VECTOR_STRATIGRAPHIC_CLUSTER` | ≥ 2 independent sections at one boundary beyond the conditional rate. **Pending** provenance and re-analysis |

plus `refined_verdict` ∈ {`NO_REFINED_PARTICULATE`,
`REFINED_PARTICULATE_CANDIDATES_PENDING_VET`}.

### 6.3 The built-in positive control

The impact class is stacked by the same machinery as the fission class. If the
corpus contains K–Pg boundary clays with chondritic PGE, the channel must
rediscover the Alvarez layer as a stratigraphic cluster of `impact`-class
samples — using only the ratios, with no knowledge of which samples those are.
`summary.json["impact_positive_control"]` carries that stack. If it comes back
empty, the report says which PGE were present at all, because a positive
control that could not run is a statement about the corpus, not about the
method.

The offline suite runs the same control on synthetic data: five independent
sections with a chondritic Ir–Os–Ru–Rh–Pt–Pd panel at one level, five more
sections sampled at that level carrying nothing, on a panel present in 5 of 130
analyses. The channel must class the ejecta `impact` (never fission) and
recover the cluster at p ≈ 8 × 10⁻⁴.

---

## 7. Interpretation ladder

1. `NO_DATA_REACHED` → fix access; the probe ledger names every endpoint tried
   and what it served. Says nothing about the rock.
2. `NO_FISSION_VECTOR` → a count of samples screened. Not an occurrence limit
   (the corpus is a convenience sample of what geochemists chose to analyse,
   and the decisive elements Ru, Rh, Pd, Te are measured on almost none of it),
   and not a paper. It is a reason to change the question — most obviously to
   the isotopic route (§8).
3. `FISSION_VECTOR_SINGLE_SECTION` → contamination until proven otherwise.
   Report the sample, its provenance flag, its reference, and what would settle
   it.
4. `FISSION_VECTOR_STRATIGRAPHIC_CLUSTER` → still not a discovery. It is a
   request for three things no database can supply: (a) the provenance of every
   sample in the cluster (core vs outcrop, laboratory, campaign, year);
   (b) re-analysis of the same horizon from fresh material by a different
   laboratory; (c) **isotopes** — ¹⁴⁵Nd/¹⁴⁴Nd and ¹⁵⁰Nd/¹⁴⁴Nd, ¹⁰⁰Ru/¹⁰¹Ru,
   ²³⁵U/²³⁸U depletion. The elemental vector can only ever be a *finder*; the
   isotopes are the proof, and they are exactly how Oklo was settled.

---

## 8. What a null does not mean, and the next decisive question

The corpus is a convenience sample. SGP was assembled to study redox and
oxygenation, so its dense columns are Fe speciation, Al, Mo, U, TOC — and its
thin ones are precisely the light-peak fission products. EarthChem's
sedimentary holdings are heterogeneous. Neither was collected to answer this
question. So `NO_FISSION_VECTOR` over N samples means "no sample in this
compilation, measured on the elements these workers happened to measure, needs
a fission component"; it does not bound the occurrence of a buried
technological extinction, and per `CLAUDE.md` it will not be written up as if
it did.

The decisive escalation, if the elemental route runs out, is **isotopic and
targeted**, not broader:

* the Fe–Mn crust ¹²⁹I profiles (Pacific crusts, ¹²⁹I/¹²⁷I 7×10⁻¹⁴–1.3×10⁻¹²
  with an apparent exponential decay over 55–70 Myr) and Wallner's Crust-3
  archive, for ²³⁶U/²³⁸U and ¹³⁵Cs — a pulse with **no ⁶⁰Fe co-pulse** is the
  no-correlated-partner logic of S46 applied to nuclides;
* published Nd and Ru isotope data at the boundaries, where a fission
  contribution shifts ¹⁴³Nd/¹⁴⁴Nd-normalised ratios in the direction Oklo does;
* ²³⁵U/²³⁸U in boundary shales, where fission and only fission depletes ²³⁵U.

---

## 9. Files

```
src/seti/grave/references.py   reservoir vectors, atomic masses, oxide factors, citations
src/seti/grave/vectors.py      fission mass vector, mixture fit, vetoes, PGE/alloy classes
src/seti/grave/agestack.py     boundary table, section keys, the hypergeometric stack
src/seti/grave/acquire.py      SGP / EarthChem / GEOROC, runtime schema discovery
src/seti/grave/run.py          stages, verdicts, REPORT.md
config/grave.yaml              every threshold, every candidate field code
tests/test_grave.py            offline suite (network-guarded)
.github/workflows/grave.yml    workflow_dispatch: stage, sources, max_rows
results/grave/                 probe.json, acquisition.json, screen.json,
                               candidates.json, refined.json, summary.json, REPORT.md
```
