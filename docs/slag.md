# SLAG-WD (S51) — the refinery vector: polluted white dwarfs beyond the natural family

*Signature S51 of `docs/necrofrontier.md`. Code `src/seti/slag/`, config
`config/slag.yaml`, workflow `.github/workflows/slag.yml`, results
`results/slag/`.*

A white-dwarf photosphere is a mass spectrometer of whatever last fell into
it. PEWDD (Williams et al. 2024, A&A 691, A352; VizieR `J/A+A/691/A352/pewdd`)
is the complete public readout of that spectrometer: every published
photospheric metal abundance of every polluted white dwarf, with provenance
per row.

The question this channel asks has not been asked of it. Huang, Tao & Zhang
2026 tested **one** artificial template (a dense siderophile concentrate)
against a meteorite mixture. What has never been published is the inverse: a
**calibrated ranking of every object by how badly the best natural parcel
reproduces it** — the misfit list — plus a test on the element pairs that no
natural process can separate.

---

## 1. Why a pair can be process-orthogonal

Nature moves element ratios in a polluted photosphere along exactly four
levers:

| lever | what it does | where it lives in the code |
|---|---|---|
| parent-body composition | mixes chondrite / planetary end-members | `family.py`, `data_assets/slag_natural_family.csv` |
| condensation temperature | depletes (or enriches) elements below a cut | `family.fractionation_factor` |
| metal–silicate + melt partitioning | core / mantle / crust end-members | `family.py` end-members |
| photospheric sinking | the Koester 2009 early / steady / declining phases | `sinking.py` |

A pair of elements that all four levers move **together** — same volatility,
same siderophile/lithophile behaviour, same incompatibility, near-equal
diffusion timescales — has a narrow natural envelope. Six such pairs are
tested: Ti/Al, Sc/Ca, Ca/Al, Sr/Ca, Mn/Cr, Ni/Co.

The channel does **not** assume the brief's claim that all six stay inside
~0.3 dex. Each envelope is computed and reported
(`summary.json["pair_statistics"]`, `["measured_meteorites"]["pairs"]`), from
two sources:

* the **compiled end-members** (18 averaged vectors) moved by the
  condensation lever — the envelope the brief's claim describes;
* the **measured meteorites**: 1,227 individually analysed bodies from the
  compilations PEWDD itself ships (`jamietwilliams/PEWDD`,
  `meteorite_database_{Si,Fe,Mg}.csv`, the same analyses against three
  reference elements, merged), read as log₁₀ number ratios.

They do not agree, and the measured ones are what the envelope uses, because
they can only widen it:

| pair | end-members | measured (n bodies) | total | widest single class |
|---|---|---|---|---|
| Ti/Al | 0.31 dex | 2.95 (1096) | 2.95 | EUC 2.04 |
| Ca/Al | 0.79 | 3.63 (1200) | 3.72 | CH 0.91 |
| Mn/Cr | 2.91 | 3.21 (1038) | 3.86 | EUC 1.68 |
| Ni/Co | 1.43 | 3.53 (737) | 3.53 | CL 1.18 |
| Sc/Ca | 0.77 | 0.08 (8) | 0.77 | — |
| Sr/Ca | 1.23 | — (0) | 1.23 | — |

**The brief's process-orthogonality premise does not survive the measured
meteorites.** Ti/Al is 0.31 dex across the compiled end-members and 2.95 dex
across real stones; Ca/Al reaches +2.97 in pallasites, where Al is a trace
element in an olivine–metal rock, against a compiled maximum of +0.04. Even
inside one class Ti/Al spans ~1 dex. Part of that width is real chemistry in
metal-rich bodies and part is analytical scatter on trace elements in single
small samples — and both belong in the envelope, because both are what a
"natural meteorite" is allowed to look like when one falls onto a white
dwarf. Tier 2 exceedances computed against the compiled end-members alone
would have been artefacts of the compilation.

The two pairs that stay narrow, Sc/Ca and Sr/Ca, are narrow *in the
compilation only*: 8 and 0 measured bodies carry both elements. Their
narrowness is a statement about coverage, not about nature, and the record
says so rather than trading on it.

---

## 2. Tier 1 — the calibrated misfit list (the deliverable)

For one panel (one star, one literature source) the natural model is

```
log(Z/X)_phot = log parcel_Z(w, T_cut, d) + phase_Z(tau_rel, t_acc, t_dec) + c
```

with `w` on the end-member simplex, `(T_cut, d)` the condensation lever,
`(t_acc, t_dec)` the sinking phase and `c` a free normalisation (the reference
element and the total accreted mass cancel).

Two details are load-bearing:

* The objective is a true −2 ln L — `sum res² + sum ln σ²` — not a bare
  chi-square. With a bare chi-square the optimiser "explains" any anomaly by
  choosing a deep declining phase whose timescale sensitivity inflates the
  error bar; the `ln σ²` term charges it for that.
* The misfit **probability is calibrated, not read off a chi-square table**.
  The best natural model is perturbed by the panel's own effective errors
  N times, each draw is refitted with *exactly* the same freedom, and `p` is
  the fraction of draws whose minimum objective is at least the observed one.
  A low `p` therefore means: the natural family, with all its freedom,
  reproduces this star worse than it reproduces its own draws.

There is a **second, harder calibration** with the same machinery and a
different null (`misfit_meteorite`, `fit.n_cal_meteorite` draws): the draw is
a *real measured meteorite* from the compilation above, given a random
sinking phase and this panel's own errors, refitted with the compiled
end-member model. No condensation lever is applied to it — a stone already
carries its own volatile depletion. It separates the two readings of a small
`p`: small posterior `p` **and** large meteorite `p` says the star is unlike
the model *and* the model handles real rocks, so the star is odd; both small
says the model cannot fit a rock either, and the misfit is the model's. A
panel whose elements the compilation does not cover reports
`SUITE_LACKS_ELEMENTS` or `TOO_FEW_BODIES_COVER_THE_PANEL` and is not
calibrated this way — never silently given a p.

### The calibrated per-element residual

The same draws answer a sharper question at no extra cost. Each of the N
refits leaves a residual at every element, so the distribution of *one
element's* residual under the natural model is already in hand:
`misfit["per_element"][el]["p"]` is the fraction of natural draws left at
least as badly fitted at that element as the data are.

This is the complement Tier 2 needs. The pair envelope asks whether a ratio
lies outside everything nature has been *measured* to do, and §1 shows that
envelope is wide — 2.95 dex for Ti/Al across real stones. The per-element p
asks instead how unusual this element is *given the model's full freedom*, so
it does not depend on the envelope at all. With a dozen elements per panel
the smallest of a dozen p values is small by construction, so
`_worst.p_min_corrected` (the Bonferroni-corrected minimum) is reported
beside it and is what any claim must use. It is a diagnostic that points at
*which element* carries a panel's misfit — never on its own a candidate rule.

`p < 0.01` is `UNEXPLAINED`, `p < 0.05` is `WATCH`. **A low `p` is a
measurement about the natural family's reach, not a technosignature** — that
is what Tier 2 is for.

## 3. Tier 2 — the pair residual and the refinery flags

`z_pair` is the distance of the observed log ratio *outside* the natural
envelope, in units of `sqrt(σ_a² + σ_b² + σ_sys²)` with σ_sys = 0.15 dex; it
is zero inside. A candidate needs `|z_pair| > 4` on at least one pair **while
the rest of a ≥ 5-element panel is natural**.

The sinking phase is **one number per object, not one per element**. So the
phase shift a pair is allowed is not the extreme over every phase — it is the
extreme over the phases the *rest of that panel* accepts (the acceptable phase
set of the Tier 1 fit of the panel with the pair removed). This makes the
argument Doyle et al. 2021 made by hand against sinking for the beryllium
objects mechanical.

Flags (each a comparison against the same envelope machinery with the same
rest-of-panel phase set): `ALLOY_TI_AL_V_WITHOUT_FE`, `PURE_SI_WITHOUT_O_MG_FE`,
`CU_ZN_SN_PB_ENRICHED`, `BE_WITHOUT_LI_B`, `FE_MG_BELOW_MANTLE`.

## 4. The kill ledger

Every kill is recorded per candidate and never silently applied; a killed
candidate stays in `candidates.csv` with its kill named.

| kill | what it rejects |
|---|---|
| `INFORMATION_LIMITED` | fewer than 5 measured elements — never a candidate |
| `REST_OF_PANEL_NOT_NATURAL` | the anomaly is the whole panel, not the pair |
| `COOL_HE_LINE_PHYSICS` | Ca/Na/K/Li in a He atmosphere below 7000 K, where the line physics is weakest (this is where the Li/K objects live) |
| `ASYNCHRONOUS_ACCRETION_EXPLAINS` | Brouwers et al. 2022: two natural parcels with their own phases fit it |
| `REFRACTORY_FRACTIONATION_EXPLAINS` | the free-fractionation model (no cut on T_cut) fits it |
| `CARBONATE_CRUST_POSSIBLE` | C/Ca high enough for a carbonate crust (SDSS J1043+0855) |
| `MULTI_REFERENCE_DISAGREEMENT` | another source for the same star disagrees by > 3σ |
| `EXOTIC_MANTLE_POSSIBLE` (caveat) | Putirka & Xu 2021 reduced / Mercury-like mantles |

---

## 5. What the runner corrected

The offline build was wrong about PEWDD in three ways that only the served
table could reveal. Each would have *manufactured* the signal this channel
hunts, so they are recorded here and regression-tested in `tests/test_slag.py`.

**Boron was the magnetic field.** PEWDD has a column `B` — the magnitude of
the white dwarf's magnetic field, unit `I8`. Matching an element by its bare
symbol picked it up as boron. Column-role resolution is now two-tiered: every
name that *states* a ratio against the dominant element (`log(Ti/H(e))`) is
scanned across all columns first, and in a table where ≥ 3 elements resolve
that way the bare-symbol tier is never consulted. PEWDD resolves 26 elements,
none spurious. (Run 35737518217.)

**Upper limits are marked by a negative error.** PEWDD carries no `l_` flag
column for the metals. An upper limit is a row whose error column holds −1.
Read as a detection with an assumed 0.2 dex error, such a value is a fake
depletion — on 126 of the 221 panels with ≥ 5 elements. A negative error now
routes the value to the panel's one-sided limit list.

The convention is not assumed, but *where* it can be checked is not where the
channel first looked. PEWDD publishes `total_detections` and
`total_upper_limits` per row — **in the database's own CSV only**. The VizieR
service serves 200 columns and neither of those two, so the check ran against
nothing and reported zeros that read like agreement. It now runs against
whichever fetched copy carries the counts
(`summary.json["limit_convention_check"]`), and on the served data it says:

> **3,465 of 3,475 rows** reproduce PEWDD's own upper-limit count exactly from
> "error < 0 means an upper limit", over this channel's 26 elements.

The *upper-limit* count is the strict test. The detection count cannot agree
and is reported only for completeness: PEWDD counts detections over every
element it carries, including H, He and elements outside this channel's list,
so a row with a detection of one of those is undercounted here by
construction (2,381 of 3,475 agree exactly, the rest low by 1–4).

**The PyllutedWD grids were fetched and silently ignored.** All twelve
`data/timescales_*.csv` files downloaded with status OK, and
`acquire.json["timescales"]["parsed"]` was `{}` — the parser recognised none
of them and said nothing about why, so the run looked healthy while quietly
using a different timescale source. A file that does not parse now records
`parse_diagnosis` (its row keys, row widths, the temperature grid it found and
which of the three conditions failed) and keeps its raw text under
`results/slag/data/timescales_raw_*`, so the real layout is readable from the
committed artifacts without refetching. This costs the channel nothing
scientifically — the source actually used, PEWDD's own per-star `SinTime*`
columns, is measured on these very stars and is the better one (below) — but
an unexplained silent fallback is not acceptable in the record.

**The sinking timescales are in the catalogue.** PEWDD publishes τ_Z per star
per element (`SinTimeCa`, …) for that star's own structure. Where a row has
them for the whole panel they are used verbatim. They also calibrate the
rest: log10(τ_Z/τ_Ca) is nearly constant across the 95 rows that publish it
(interquartile widths 0.03–0.15 dex, no significant Teff trend, H and He
agreeing to ~0.01 dex), so those rows build a relative library used for every
other panel. The library shows the embedded mass-scaling law was ~60 % too
shallow (Mg +0.171 measured against +0.098 assumed), which biased every
steady-state and declining-phase correction. The source actually used is
recorded per panel. The library, measured on 95 rows:

| element | log10(τ/τ_Ca) | σ | n |
|---|---|---|---|
| Ni | −0.140 | 0.113 | 25 |
| Mn | −0.128 | 0.091 | 25 |
| Fe | −0.120 | 0.094 | 39 |
| V | −0.112 | 0.058 | 14 |
| Cr | −0.079 | 0.074 | 28 |
| Ti | −0.076 | 0.039 | 30 |
| Sc | −0.064 | 0.016 | 13 |
| S | +0.076 | 0.110 | 12 |
| P | +0.089 | 0.046 | 7 |
| Si | +0.118 | 0.047 | 36 |
| Al | +0.128 | 0.044 | 37 |
| Na | +0.163 | 0.035 | 28 |
| Mg | +0.171 | 0.023 | 38 |
| O | +0.275 | 0.093 | 33 |
| C | +0.334 | 0.114 | 13 |
| N | +0.334 | 0.111 | 8 |
| Li | +0.541 | 0.132 | 8 |

Monotonic in atomic mass, as diffusion requires. The mass-scaling exponent
refitted to it is **β = 0.713**, against the 0.45 the offline build assumed.
Elements the library cannot reach (Be, K, Co, Cu, Zn, Sr, Sn, Ba) use that
refitted exponent rather than the literature guess.

**One star, several rows.** PEWDD distinguishes alternative solutions for one
star by a name suffix — `PG1225-079 Model 2`, `GD 362 Updated`,
`WDJ0649-7624 (phot)`. Ungrouped, one star entered the misfit list five times
and never saw its own object's other panels, so the multi-reference kill could
not fire. Qualifiers are stripped before grouping.

**One star, several designations — the object is a position, not a name.**
Stripping qualifiers is not enough: PEWDD is one row per star per paper, and
each paper writes the star the way its own field does. `GD 378` and
`WD 1822+410` are one He-atmosphere DBZ; `PG 0843+516`, `PG 0843+517` and
`WD0843+516` are one DA. Name grouping gave 2441 objects for 3547 rows.

All 3547 rows carry `RAJ2000`/`DEJ2000`, so objects are built by
single-linkage **on the sky** within 5″: **1576 objects**, 633 of which merge
more than one designation. The radius is on a plateau (1610 at 1″, 1594 at
2″, 1588 at 3″, 1576 at 5″, 1566 at 8″, 1559 at 12″); 5″ rather than 3″
because PEWDD's positions are per-paper transcriptions at different epochs
and these stars have large proper motions — GD 362's two served positions are
3.5″ apart and were split at 3″. A false 5″ pair among ~1600 objects over the
whole sky is ~10⁻³.

The name deliberately does **not** link two sky positions. PEWDD carries rows
whose designation belongs to a different star from their coordinates — two
rows called `WD1202-232` sit 40° apart, and rows called `L745-46A` carry Ross
640's position. Joining on the name as well as the sky chained those into
single-linkage blobs, one of them 27 rows over ten unrelated designations,
which would have pooled unrelated stars' abundances into one object. Names
link only rows with no coordinate at all, and a designation PEWDD reuses for
two positions keeps two object keys, each tagged by its position. The count
of such reuses is in `summary.json["object_grouping"]`.

This matters beyond bookkeeping: the shard unit, the misfit list's
one-row-per-object choice, and above all the `MULTI_REFERENCE_DISAGREEMENT`
kill all compare an object's own sources, and under name grouping 633
objects' sources were never compared.

---

## 6. The population

From the served table: 3547 rows, 2778 distinct star strings, 2441 distinct
name keys once PEWDD's per-solution qualifiers are stripped, and **1576
objects** once those names are reconciled on the sky (§5).

Counting *detections only* — upper limits excluded, per §5:

| n measured elements | panels (rows) |
|---|---|
| 0–1 | 2723 |
| 2–4 | 656 |
| **≥ 5 (screened and calibrated)** | **168 panels / 123 objects** |

(137 objects under the old name grouping; the sky reconciliation of §5 merges
14 of those into another object's designations.) 187 panels carry at least one
upper limit. (Before the negative-error
convention was understood, 209 panels appeared to reach ≥ 5 elements — the
extra 41 were limits read as detections.)

Atmospheres: 3033 He, 514 H, resolved from PEWDD's own `atmosphere` column.
Abundances are `log(Z/H(e))` throughout — H in a hydrogen atmosphere, He in a
helium one; the science uses only element ratios, in which the reference
cancels, but the sinking timescales do not, so the reference is resolved and
recorded per row.

## 7. Controls

The literature outliers run as controls and their landing is reported in
`results/slag/controls.json` whatever it is. Every alias list in
`config/slag.yaml` is now the set of designations PEWDD *actually serves at
that object's position*, read off the acquired table rather than guessed from
the literature, and the matched position, Teff and atmosphere travel with each
control so a wrong match is visible:

| control | served position | rows | best panel | served designations |
|---|---|---|---|---|
| GD 362 | 262.8931 +37.0881 | 5 | **16** | GD 362, GD 362 Updated, GD362, J1731+3705 |
| GD 378 | 275.9042 +41.0679 | 8 | 13 | GD 378, WD 1822+410, WD1822+410 |
| PG 1225−079 | 186.9473 −8.2439 | 8 | 11 | PG 1225-079 (+ Updated, Model 1/2/3), K 789-37 |
| GALEX J2339−0424 | 354.8210 −4.4069 | 1 | 9 | GALEXJ2339 |
| LHS 2534 | 183.7349 −2.5675 | 4 | 7 | LHS 2534, WD 1212-022, SDSS J121456.39-023402.7, J1214-0234 |
| WD 0106−328 | 17.1501 −32.6287 | 5 | **4** | HE 0106-3253, HE0106-3253 |
| NLTT 19868 | 129.0070 −10.1021 | 1 | **4** | NLTT 19868 |

Two alias corrections came out of this, each of which would have pointed a
control at the wrong star or at nothing:

* **NLTT 19868 is not WD/PG 0843+516.** That is a different polluted DA at
  131.7595 +51.4815, 62° away. PEWDD serves NLTT 19868 at 129.0070 −10.1021.
* **LHS 2534 is not WD 1214+032.** PEWDD serves it as `WD 1212-022` /
  `SDSS J121456.39-023402.7` at 183.7349 −2.5675.

Two of the seven are below the information floor in PEWDD: **WD 0106−328 and
NLTT 19868 reach only 4 measured elements** in their best published panel, so
they are `INFORMATION_LIMITED` by construction and can never be candidates
here. That is a statement about what has been published for them, not about
the stars — the Fe-as-pure-metal claim for WD 0106−328 (Farihi 2026) and the
extreme Fe-depletion of NLTT 19868 (Kawka & Vennes 2016) rest on panels PEWDD
does not carry at ≥ 5 elements.

## 8. Verdicts

| verdict | meaning |
|---|---|
| `NO_DATA_REACHED` | no PEWDD rows acquired — a statement about the archive, never about the sky |
| `INFORMATION_LIMITED_ONLY` | rows, but no panel with ≥ 5 measured elements |
| `MISFIT_LIST_PRODUCED` | the calibrated list exists; no pair or flag survived its kills |
| `REFINERY_VECTOR_CANDIDATE` | ≥ 1 pair residual or flag survived every kill — pending vet |

Degradation (a failed route, an assumed error, an unresolved atmosphere, an
embedded rather than catalogued timescale law) is a separate first-class
field, never folded into the verdict string.

## 9. Running it

```
python -m seti.slag.run --stage probe      # what VizieR and GitHub hold, and the column roles
python -m seti.slag.run --stage acquire    # the rows, the timescale grids, the meteorite tables
python -m seti.slag.run --stage screen --shard 1/8
python -m seti.slag.run --stage assess     # misfit_list.csv, pairs.csv, flags.csv, controls.json
```

or `seti slag --stage all`. On the runner: dispatch `slag.yml` with
`stage=all`, `shards=4` — four is enough (168 panels carry the calibration
cost; the other 3379 are seconds) and eight takes eight runner slots from the
other channels for no wall-clock gain.
