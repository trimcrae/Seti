# FALLOUT — the fission-product abundance pattern in a photosphere

**Claim being tested.** A civilization that disposed of fission waste into its
host star (Whitmire & Wright 1980, *Icarus* 42, 149) leaves two things in the
photosphere. The first is a set of **radionuclide lines** — Tc I, Pm II, U/Th —
with a built-in decay clock; that is signature S13 and the sibling channel
`midden` searches for it. The second is what remains after every short-lived
member has decayed: a **fixed multi-element shape** in the neutron-capture
elements, set by the fission mass-yield curve and by nothing else. That shape
is signature S14 (`docs/necrosignatures.md` §2.III, hitherto marked "covered by
midden" — it was not: midden searches lines, never the vector). It is the only
part of the signature that outlives its makers by megayears, and it has never
been searched for anywhere.

FALLOUT is the complement of MIDDEN, not a refinement of it: different
observable (a catalogue abundance vector, not a line), different corpus (GALAH
DR4 survey abundances, not high-resolution ESO spectra), different population
(cool dwarfs, not A5–F2), and a different time window (≳ 10⁶ yr, not < 10⁵).
It is also disjoint from TAILINGS, which searches for a *sparse* one-element
anomaly; here a one-element anomaly is a **rejection**.

---

## 1. Physics: why the fission vector is unnatural

### 1.1 The yield curve

Thermal-neutron fission of ²³⁵U (and, with small shifts, ²³⁹Pu/²³³U) produces
two fragments per fission on a two-humped mass distribution:

* a **light peak** at A ≈ 90–105 → after decay: Sr, Y, **Zr, Mo, Ru**, Rh, Pd;
* a **heavy peak** at A ≈ 133–145 → Xe, Cs, **Ba, La, Ce, Pr, Nd**, Sm;
* a **valley** at A ≈ 110–125 (Ag, Cd, In, Sn, Sb, Te) roughly a thousand
  times lower;
* a steep fall beyond A ≈ 155 (Eu, Gd small; no Pt, Au) and **no Pb** — A = 208
  is never a fission product.

`src/seti/fallout/yields.py` carries the ²³⁵U cumulative mass-chain yields
from ENDF/B-VII.1 / JEFF-3.1.1 (England & Rider 1994 tabulation), per 100
fissions, with the element each chain sits on at a chosen **decay horizon**.
The horizon matters: Cs-137/Sr-90 (30 yr) are Ba/Zr within centuries; Tc-99
(2.1 × 10⁵ yr) has become Ru by 1 Myr; Cs-135, Zr-93, Pd-107 and I-129 outlast
that. The channel's default horizon is **1 Myr** — the pattern that outlives
its makers — and at that horizon there is no Tc at all, which is why this
search and MIDDEN's cannot be the same search. The chains sum to 200.0 per 100
fissions (a test asserts it).

Element yields at the 1 Myr horizon (per 100 fissions): Zr 36.7, Mo 24.6,
Xe 21.4, **Nd 20.7**, Ru 17.5, Cs 13.2, Ba 13.0, Ce 12.1, La 6.4, Pr 5.9,
Y 4.7, Rb 3.9, Sm 3.7, Sr 3.6, Te 2.2, Pd 1.6, Eu 0.58, Sn 0.10, Cd 0.06,
Gd 0.05, Ag 0.03.

### 1.2 Folded against solar

Adding fission product at amplitude *a* multiplies the number abundance of X
by `1 + a·F_X` with `F_X = (Y_X / N_⊙,X) / (Y_Nd / N_⊙,Nd)` — normalised so
that `a = 1` doubles Nd. Solar abundances are Asplund, Amarsi & Grevesse 2021
(meteoritic for Te/I/Cs; Lodders 2009 for Kr/Xe). The s- and r-process
templates are the solar-system fractional decompositions of Arlandini et al.
1999 (stellar model), cross-checked against Bisterzo et al. 2014, with the
p-process share of Mo (25%) and Ru (7%) kept separate so it does not inflate
the r residual. The table the search actually used, at `a = 1`:

| element | fission yield % | A(X)⊙ | s | p | r | F/F_Nd | fission dex | s dex | r dex |
|---|---|---|---|---|---|---|---|---|---|
| Rb | 3.88 | 2.60 | 0.50 | 0 | 0.50 | 0.012 | +0.005 | +0.176 | +0.176 |
| Sr | 3.55 | 2.83 | 0.85 | 0.01 | 0.14 | 0.007 | +0.003 | +0.267 | +0.057 |
| Y | 4.73 | 2.21 | 0.92 | 0 | 0.08 | 0.037 | +0.016 | +0.283 | +0.033 |
| Zr | 36.67 | 2.59 | 0.83 | 0 | 0.17 | 0.120 | +0.049 | +0.262 | +0.068 |
| Mo | 24.57 | 1.88 | 0.50 | 0.25 | 0.25 | 0.411 | +0.150 | +0.176 | +0.097 |
| Ru | 17.46 | 1.75 | 0.32 | 0.07 | 0.61 | 0.394 | +0.144 | +0.121 | +0.207 |
| Ba | 12.96 | 2.27 | 0.81 | 0.01 | 0.18 | 0.088 | +0.037 | +0.258 | +0.072 |
| La | 6.41 | 1.11 | 0.62 | 0.01 | 0.37 | 0.632 | +0.213 | +0.210 | +0.137 |
| Ce | 12.07 | 1.58 | 0.77 | 0.02 | 0.21 | 0.403 | +0.147 | +0.248 | +0.083 |
| Pr | 5.85 | 0.75 | 0.49 | 0 | 0.51 | 1.321 | +0.366 | +0.173 | +0.179 |
| **Nd** | 20.71 | 1.42 | 0.56 | 0 | 0.44 | **1.000** | **+0.301** | +0.193 | +0.158 |
| Sm | 3.67 | 0.95 | 0.29 | 0.03 | 0.68 | 0.524 | +0.183 | +0.111 | +0.225 |
| Eu | 0.58 | 0.52 | 0.06 | 0 | 0.94 | 0.223 | +0.087 | +0.025 | +0.288 |

(Pr is in the table for completeness; GALAH DR4 does not deliver it, so the
GALAH element space is the other twelve. Whitmire & Wright's specific
prediction — Pr and Nd the most overabundant products — is exactly the top of
the F column.)

### 1.3 The discriminant is a vector

The named ratios at `a = 1`, in dex:

| | [Nd/Ba] | [Ce/Ba] | [La/Ba] | [Mo/Zr] | [Ru/Zr] | [Eu/Nd] | [Sr/Nd] |
|---|---|---|---|---|---|---|---|
| **fission** | **+0.264** | **+0.110** | **+0.176** | **+0.101** | **+0.095** | **−0.214** | **−0.298** |
| s-process | −0.065 | −0.010 | −0.048 | −0.086 | −0.142 | −0.168 | +0.074 |
| r-process | +0.086 | +0.011 | +0.065 | +0.029 | +0.139 | +0.129 | −0.101 |

Read column by column. The s-process makes Ba and Zr *well*, so it drives
[Nd/Ba] and [Mo/Zr] **negative** and moves Sr/Y strongly; fission barely
touches Sr/Y (their solar abundances are enormous) and drives both ratios
**positive**. The r-process makes Eu, so [Eu/Nd] goes **positive**; fission
makes almost no Eu, so it goes **negative** — and a mixture of s and r cannot
have [Nd/Ba] ≫ 0 without the r-component bringing Eu with it. The fission
vector — heavy-peak up with Nd ≫ Ba, light peak Mo/Ru up with Zr, Sr/Y flat,
Eu low — is not a point in the (s, r) plane. That is the whole claim, and it
is a statement about nuclear physics, not about stellar models.

Note that [Eu/Nd] < 0 alone does *not* separate fission from s (both are
negative); [Nd/Ba] does. And [Nd/Ba] > 0 alone does not separate fission from
r; [Eu/Nd] does. It takes the vector.

---

## 2. Novelty position — *to be verified by falloutlit on the runner*

What can be stated from the corpus already on the record
(`results/tailingslit/`, `results/necrolit/`, `results/przybylski_lit/`):

* **Whitmire & Wright 1980** proposed the observable and predicted Pr/Nd. Its
  46-year citation tree (56 works via OpenAlex, re-fetched by `falloutlit`)
  is reviews and essays; no executed search. The 2026 flagship review devotes
  one paragraph to stellar pollution.
* **The only executed photospheric chemical-technosignature search** — Huang,
  Tao & Zhang 2026 (arXiv:2605.29811), polluted white dwarfs — tests
  meteorite-derived *siderophile* templates. "Fission" does not appear as a
  template family; the paper's own stated gap is "expanded template families".
* **MIDDEN** (this repository) is the first survey-scale execution of the
  1980 proposal and searches **lines** (Tc I 4238/4262/4297, U II 3859, Th II
  4019) in A5–F2 stars. It never fits an abundance vector.
* **TAILINGS** (this repository) searches GALAH/APOGEE cool dwarfs for a
  *sparse* one-element anomaly and rejects multi-element ones.
* **The nucleosynthesis literature** decomposes n-capture patterns into s, r
  (and i, p) components routinely — that machinery is exactly what the natural
  alternative here is built from — but it has never been asked whether a
  *fission* template fits any star better than the natural ones, because there
  is no natural reason to ask.
* The Przybylski's-star claims (Cowley et al. 2004; refuted for Tc by
  Andrievsky et al. 2023) are single-star line claims in an Ap star.

The honest position is therefore: **the fission-yield pattern as a fitted
template against s/r/s+r mixtures over a survey abundance catalogue has, on
the evidence currently on the record, never been done.** The lit sweep
(`scripts/falloutlit_fetch.py`, job `falloutlit` in `.github/workflows/
fallout.yml`) exists to test that against primary sources — verbatim arXiv
abstracts for "fission products stellar photosphere", "nuclear waste
technosignature star", "anomalous neodymium barium ratio dwarf", "s-process
r-process decomposition GALAH", "artificial abundance pattern technosignature"
and twenty more, the Whitmire & Wright and Huang et al. citation trees, and
title-verified full texts of the decomposition papers the templates rest on.
Nothing in this section is to be cited as established until
`results/falloutlit/` is on the branch and has been read.

---

## 3. Method

### 3.1 Data (`acquire.py`)

GALAH DR4 allstar, pulled through `seti.tailings.acquire` (the memmapped FITS
reader, the route prober, the runtime schema discovery, the canonical
renaming) — not re-implemented. Two things are FALLOUT's own:

* **Route order.** The runner-proven URL is tried first:
  `https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR4/catalogs/galah_dr4_allstar_240705.fits`
  (HTTP 200, 758 MB, `image/fits` on the tailings dispatch; the bare
  `datacentral.org.au` host was 404). The config lists it first with fallbacks;
  anything TAILINGS' registry knows that the config does not is appended.
* **Extra columns.** `flag_sp` (the star-parameter flag), an age, a binary
  flag, `e_teff/e_logg/e_fe_h`, `vsini`, are discovered by regex at runtime and
  re-attached to the canonical table. Every one is reported found-or-absent in
  `columns_found`; a veto whose column is absent is disabled *and says so*.

The pull uses one broad box (4000 < Teff < 6500 K, log g > 0.5, SNR > 30,
[Fe/H] > −1.5) so dwarfs and giants come down in one 760 MB download; the two
samples are split at screen time:

| sample | cuts | weight | why |
|---|---|---|---|
| **dwarf** | log g > 4, 4500 < Teff < 6300 K, [Fe/H] ≥ −1.0, SNR ≥ 40 | primary | the convective-envelope regime where diffusion/levitation cannot make heavy-element peculiarity (TAILINGS §1) and where AGB/mass-transfer s-process contamination is least |
| **giant** | 0.5 < log g < 3.5, 4000 < Teff < 5500 K, [Fe/H] ≥ −1.0, SNR ≥ 40 | secondary | AGB self-enrichment and mass-transfer s-process live here; scored, reported separately, never counted as a primary candidate |

Elements available in GALAH DR4 for the pattern space (from the tailings
provenance record): Rb, Sr, Y, Zr, Mo, Ru, Ba, La, Ce, Nd, Sm, Eu. Column
names are discovered, not encoded; the per-element `flag_x_fe` and `e_x_fe`
are honoured. APOGEE DR17 (Ce, Nd only) is an optional second stage that will
report `INSUFFICIENT` unless its panel reaches `min_elements` — an honest
degrade, not a silent one.

### 3.2 Peer residuals (`pattern.peer_residuals`)

Abundances are never used raw. Each `[X/H] = [X/Fe] + [Fe/H]` is regressed on
([Fe/H], Teff/1000, log g, leave-one-out alpha proxy) with TAILINGS'
`fit_element` — a quadratic fitted by iteratively 4σ-clipped least squares, so
a genuine enrichment cannot drag its own reference surface — and the residual
is the star's vector. Working in [X/H] with [Fe/H] as a predictor removes the
normalisation aberration: an error in the star's own [Fe/H] shifts every
[X/Fe] together, which is a *coherent* vector and precisely what a pattern
fit must not be fed. A test injects a 0.35 dex/1000 K Teff trend into the
heavy-peak elements and confirms the peer residual removes it and the
raw-space "patterns" it manufactures.

The error model is `σ_X = sqrt(e_reported² + 0.05²)`, with a missing reported
error replaced by the element's empirical residual scatter. The population
nulls below calibrate whatever this choice does.

### 3.3 The fit (`pattern.fit_patterns`)

Per star, five hypotheses over the measured elements:

```
d_X = log10( 1 + a_f·F_X + a_s·S_X + a_r·R_X )     a ≥ 0
```

null (all zero), pure s, pure r, s + r, pure fission; each amplitude scanned
on a 40-point log grid (26 × 26 for s + r) and refined parabolically. Two
statistics:

* `fission_lr = ½(χ²_best-natural − χ²_fission)` — the fission-only fit against
  the best of s, r and s + r. The natural alternative has two free amplitudes
  against fission's one, so the comparison is biased *against* fission.
* `enrich_lr = ½(χ²_null − χ²_best)` — whether anything is added at all.

Classification: `NORMAL` if `enrich_lr < 12.5`; else `FISSION` if
`fission_lr ≥ threshold`, `AMBIGUOUS` if fission is best but not by the
threshold, else `S_PROCESS` / `R_PROCESS` / `S_PLUS_R`. Benchmarks in
`tests/test_fallout.py`: a synthetic Ba star (pure s at `a_s = 3`,
[Ba/Fe] ≈ +0.5, [Nd/Ba] < 0) classifies s; a synthetic r-II star (pure r at
`a_r = 6`, [Eu/Fe] ≈ +0.8, [Eu/Nd] > 0) classifies r; a synthetic
fission-polluted star (`a_f = 5`, Nd +0.78 dex) classifies fission; random
s + r mixtures never reach the fission threshold.

### 3.4 The threshold is set by two nulls

`lr_min = 8` (Δχ² = 16) is a **floor**. The working threshold is the larger of
that and the 99.9th percentile of the **shuffled-element null**: each star's
(value, error) pairs are permuted across its element slots and re-scored, so
the amplitude structure and the per-element noise are kept and only the
alignment with the fission shape is destroyed. The whole-sample distribution
of `fission_lr` is reported alongside it. A survivor is a star above the
threshold on *both* readings, and the fraction of shuffled stars above the
threshold is the expected accidental rate, stated in `summary.json`.

### 3.5 The contamination ledger — seven named vetoes, each a counter

| veto | what it catches | rule |
|---|---|---|
| `low_snr_or_flagged` | pipeline distrust | `flag_sp ≠ 0`, SNR < 40, fewer than 5 pattern elements, or a flagged **core** element (Y, Zr, Ba, La, Ce, Nd, Sm, Eu). A flagged Rb/Mo/Ru simply drops out of the vector |
| `s_process_star` | Ba/CH/CEMP-s dwarfs, mass-transfer binaries (Rekhi et al. 2025 map them inside the dwarf box) | [Ba/Fe] > 0.5 with [Nd/Ba] < 0, or [Y/Fe] and [Zr/Fe] both > 0.3 with Ba up; or the catalogue's binary flag if present |
| `r_process_star` | r-I/r-II stars | [Eu/Fe] > 0.3 with [Eu/Nd] > 0 |
| `young_ba_enhancement` | the young-cluster Ba over-estimate (D'Orazi et al. 2009; Baratella et al. 2020) | A(Li) = [Li/Fe] + [Fe/H] + 0.96 > 2.3, or age < 1 Gyr, with [Ba/Fe] > 0.2 |
| `nlte_saturated_lines` | Ba II 5853/6141/6496 saturating in metal-rich cool dwarfs | the preference must hold with **Ba excluded**; `lr_noba` is reported next to `fission_lr` for every candidate, with La/Ce as the heavy-peak anchor |
| `single_element_driver` | a one-element anomaly wearing a pattern's clothes — **the crucial test** | leave-one-out: `fission_lr` recomputed with each element dropped in turn; the minimum must still clear the threshold, and the element whose removal hurts most is named |
| `teff_peer_residual` | a pattern manufactured by the regression | the raw [X/Fe] vector must carry at least half the threshold too; the counter also records how many raw-space passers the peer residual removed |
| `unexplained_by_all_templates` | a star that "wins" against worse models without fitting (both first-run survivors) | reduced χ² of the **best** model > 3 → `UNEXPLAINED_BY_ALL_TEMPLATES`, listed separately, never a candidate |
| `heavy_peak_incoherent` | one hot element carrying the heavy peak | fewer than two of La/Ce/Nd individually ≥ 2σ in the fission direction |
| `la_cn_blend` | CN blends on the La II lines in cool giants | giant, Teff < 4800 K, preference carried by La, in a sample whose La residual tracks C/N or Teff (or could not be checked) |

The order is the order of application; `first_veto` says which one a star hit
first, and every veto is also counted independently so the funnel cannot hide
a veto behind an earlier one.

### 3.6 Stages and outputs

`probe` → `probe.json` (every route, HTTP status, length). `acquire` →
`stars_<survey>.parquet` (artifact only) + `acquisition.json` (verdict, route,
degradation, columns found, log). `screen` → `scores_*.parquet`,
`vectors_*.parquet`, `templates.json`, `screen.json`. `assess` →
`candidates_<survey>_<sample>.csv`, `summary.json`, `REPORT.md`.

`summary.json` carries `generated_utc`, `verdict_code` ∈ {`NO_DATA_REACHED`,
`DEGRADED_SOURCE`, `NO_FISSION_PATTERN`,
`FISSION_PATTERN_CANDIDATES_PENDING_VET`}, the funnel counts per sample, the
per-veto counters, the acquisition log, `columns_found`, the template vectors
used, the thresholds and both null distributions, and the sensitivity curve.

---

## 4. Sensitivity — at what enrichment does the vector become visible?

GALAH DR4 per-element errors for the heavy peak are ~0.1–0.2 dex (Nd, Ce, La),
~0.07 dex for Ba and Y, ~0.15–0.2 dex for Mo, Ru, Rb. Injecting the fission
pattern at amplitude `a_f` into 1,500 peer-residual vectors of a 240,000-star
synthetic dwarf population with those errors (SNR 40–250, a real s axis and a
real r axis) and requiring both the likelihood-ratio threshold (ln LR ≥ 8)
**and** the leave-one-out test gives, offline:

| a_f | Δ[Nd/H] | Δ[La/H] | Δ[Mo/H] | LR pass | LR + LOO pass |
|---|---|---|---|---|---|
| 0.5 | +0.18 | +0.12 | +0.08 | 0% | 0% |
| 1 | +0.30 | +0.21 | +0.15 | 5% | 0.1% |
| 2 | +0.48 | +0.35 | +0.26 | 46% | 10% |
| 3 | +0.60 | +0.45 | +0.34 | 70% | 37% |
| 5 | +0.78 | +0.61 | +0.49 | 91% | 70% |
| 10 | +1.04 | +0.86 | +0.72 | 99% | 93% |
| 20 | +1.32 | +1.13 | +0.97 | 100% | 99% |

So the channel is complete for a **≳ 1 dex Nd enrichment with the full
shape**, half-complete near +0.7 dex, and blind below ~0.4 dex. The
leave-one-out requirement costs a factor of ~2 in completeness at `a_f = 3`
— that is the price of refusing to be a one-element search, and it is paid
knowingly. On the same population the shuffled-element null put the 99.9th
percentile of `fission_lr` at 2.3 (so the configured floor of 8 governed),
no shuffled vector reached 8, and of 240,000 clean dwarfs exactly one crossed
the threshold and was removed by `single_element_driver` — the accidental
rate before vetoes is ~4 × 10⁻⁶ per star and zero after them. The curve is recomputed on the
*real* vectors in every run (`sensitivity` in `summary.json`), so the quoted
completeness is the survey's, not a synthetic one's.

The shuffled null on the same synthetic population puts the 99.9th percentile
of `fission_lr` at ~2–3 and the fraction above `lr_min = 8` below 10⁻³, so on
a 10⁵-star dwarf sample the expected number of accidental survivors before the
vetoes is of order tens, and after the leave-one-out test of order unity or
below. The run reports the actual numbers.

**Where the signal has to come from.** A G dwarf's convective zone holds
~90 M⊕ of metals (TAILINGS §3.6); doubling its Nd — 20.7% of fission product
by chain yield, N_Nd/N_H = 2.6 × 10⁻¹¹ — needs ~10⁻⁴ M⊕ of Nd, i.e. ~10⁻³ M⊕
≈ 6 × 10²¹ kg of total fission product, ~10¹¹ times humanity's cumulative
spent-fuel inventory. That is a large number and it is the honest scale of the
proposal: this is a search for a civilization that ran fission at planetary
throughput for a geological age, or a star into which the waste of a whole
system was concentrated. For a K dwarf the requirement is ~4× larger; for an
M dwarf ~15× — which is why the sample is bounded at Teff > 4500 K.

---

## 4a. What the first real run showed, and what it changed (2026-09-06)

The channel ran on GALAH DR4 from the runner (`results/fallout/`, run
commit `3fe0d23`): 395,752 rows from `GALAH_DR4_allstar_cloud`, 101,928
dwarfs and 78,344 giants in the boxes, 11/12 pattern elements. Four things
were wrong, and each is now a rule rather than a lesson.

**The error model was wrong.** The shuffled-element null put its 99.9th
percentile at ln LR **17.0** (dwarfs) / **9.7** (giants) against **2.3** on
the synthetic population, and the sample null's tail ran to 91. The cause is
in `peer_scatter_dex`: GALAH's quoted per-element errors understate the
measured peer-residual scatter by 2–4× (dwarfs: Nd 0.16, La 0.22, Ce 0.28,
Rb 0.40, Sr 0.34, Eu 0.32 dex). Every element's error is now **floored at
that sample's measured peer scatter of that element**
(`pattern.error_floors`) — the per-element form of the sqrt(reduced χ²)
rescaling LOOM applies — and the floors and inflation ratios are recorded in
`summary.json` under `error_model`. Under the floors the null threshold falls
back toward the config floor and the LR values are calibrated. A test injects
4×-under-quoted errors into pure noise and confirms the floored statistic
stops manufacturing patterns while a real injected fission star survives.

**Winning is not fitting.** Both giant "survivors" — 170203001601307
(χ²_f = 229.7 on 9 elements, reduced ≈ 29) and 230511003401363 (χ²_f = 52.5
on 8) — beat the natural models only because those were worse. A star whose
*best* model has reduced χ² above `max_reduced_chi2 = 3` is now
`UNEXPLAINED_BY_ALL_TEMPLATES`: counted, listed separately in the report,
never a fission candidate.

**One hot element must not carry a pattern.** Both survivors were La-driven
(peer La +0.89 and +0.77 dex against a template prediction of ~+0.3 at their
a_f; Sm negative; Sr −1.0 in the first). Two vetoes were added:
`heavy_peak_incoherent` (at least two of La/Ce/Nd individually ≥ 2σ in the
fission direction) and `la_cn_blend` (in giants the La residual is regressed
on Teff, log g, [C/Fe], [N/Fe], vsini — GALAH's C and N are now carried — and
if it tracks C/N or anti-correlates with Teff, a La-carried preference below
4800 K is vetoed; if the diagnostic cannot be computed La is distrusted, not
trusted). The correlations are reported in `summary.json` under
`la_diagnostics`.

**Sensitivity was mis-reported.** The dwarf curve read 1% completeness at
a_f = 10 because 79,690 of 101,928 dwarfs are `INSUFFICIENT` (Rb, Sr, Ru, Eu
are rarely measured in dwarfs) and an injection into them cannot succeed.
Completeness is now reported on **testable** stars (≥ 5 elements and ≥ 2 of
La/Ce/Nd measured) with the testable fraction beside it, and the all-star
number kept as a second column.

**The offline re-vet of the committed candidates** (`vet` stage, run on the
working tree from `candidates_galah_*.csv` with sigmas rebuilt from the
recorded peer-scatter floors — the acquired table and the vectors are not on
disk, so the shuffled null and its threshold could *not* be recomputed and
the stale 17.0 / 9.7 thresholds were used): under the floored errors the two
giants' ln LR fell from 52.5 → 9.0 and 26.9 → 11.7; both are
`unexplained_by_all_templates` (reduced χ² 14.1 and 3.2), both fail
leave-one-out on La (LOO minima 0.6 and 6.8), both trip `la_cn_blend`.
**Vetted survivors: 0 dwarfs, 0 giants**; verdict `NO_FISSION_PATTERN`
(behind the `DEGRADED_SOURCE` prefix for the missing rv/fibre columns). Of the
163 dwarf and 72 giant above-threshold stars, 127 and 57 are unexplained by
every template — the list a human should read, because a star nothing fits is
either bad data or new astrophysics, and the report separates it from the
candidates rather than burying it. The calibrated numbers — floored null,
recomputed threshold, testable-conditioned completeness — require the next
runner dispatch.

**The concept scan** (`results/falloutlit/concept_scan.json`, decoy-aware,
modelled on metronomelit's): 198 verbatim abstracts, 5 regex hits for
"fission yield/product/fragment pattern against stellar/photospheric
abundances", **0 decoy-free** — the four survivors are r-process fission
cycling in neutron-star mergers and the R-Process Alliance (nucleosynthesis
decoys, listed as `nucleosynthesis_adjacent_hits` for a human to read). Six of
the 18 title-verified full-text targets remain unresolved (Cowley 2004,
Korotin 2015, Rekhi 2025, Karinkuzhi 2021, Hansen 2018, Prantzos 2020) and are
recorded as such in `verification.json`, not dropped.

## 5. Interpretation ladder

1. **Instrumental / systematic** — a flagged core element, the pattern
   collapsing without Ba, one element carrying the preference, or a pattern
   absent in raw space: vetoed, feeds the ledger.
2. **Known natural** — a Ba dwarf, an r-II star, a young Ba-enhanced dwarf:
   reclassified; a catalogue correction at most.
3. **New stellar astrophysics** — a cool dwarf whose n-capture vector no s + r
   mixture fits, with the preference surviving every leave-one-out and the raw
   spectrum re-measured: publishable regardless of interpretation, because
   nucleosynthesis does not make a [Nd/Ba] ≫ 0, [Eu/Nd] < 0, [Sr/Nd] ≪ 0
   star. (The i-process is the one natural process worth naming here: it
   makes a heavy-peak-rich pattern in CEMP-i stars — but with Ba *and* the
   light peak up and at low metallicity in evolved or mass-transfer stars, not
   with Sr/Y flat and Mo/Ru up in a solar-metallicity dwarf. It will be tested
   as a fourth template if a survivor ever gets that far.)
4. **FALLOUT candidate** — rung 3 plus a light-peak confirmation (Mo, Ru up
   with Zr; Sr, Y flat), Pb *not* enhanced, and no Tc — at which point MIDDEN's
   line search on the same star dates the injection.

---

## 6. What a null does not mean

An empty survivor list says that **no star in this corpus carries a
≳ 0.8 dex fission-shaped enrichment with all twelve elements unflagged at
SNR ≥ 40**. It says nothing about smaller enrichments (the completeness curve
is in the summary), about stars whose Nd or La were flagged, about elements
GALAH does not measure (Pr — the top of the yield curve — Cs, Xe, Tc), or
about the ~10⁵ GALAH stars outside the two boxes. Per `CLAUDE.md` a null is a
reason to change the question: the APOGEE Ce/Nd panel as a second survey, the
co-natal differential channel (where a 0.1 dex differential is meaningful), a
Pr-capable high-resolution follow-up of the strongest `AMBIGUOUS` stars, and
the giant sample scored with an i-process template. No occurrence limit will
be written up.

## 7. Files

`src/seti/fallout/{__init__,yields,pattern,acquire,run}.py`,
`config/fallout.yaml`, `tests/test_fallout.py`,
`.github/workflows/fallout.yml` (jobs `fallout`, `falloutlit`),
`scripts/falloutlit_fetch.py`, `results/fallout/`, `results/falloutlit/`.
Entry: `python -m seti.fallout.run --stage {probe,acquire,screen,assess,all}`;
`seti.fallout.run.register(sub)` for the CLI.
