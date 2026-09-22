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
~0.3 dex. Each envelope is computed from the family and reported
(`summary.json["pair_statistics"]`). Ti/Al, Ca/Al and Sc/Ca are genuinely
tight; Sr/Ca, Mn/Cr and Ni/Co open by 1–1.5 dex once continental crust and
core-formation residues are in the family. A wide envelope makes a pair a
weak test, not a wrong one, and the number is in the record.

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
routes the value to the panel's one-sided limit list. The convention is not
assumed: PEWDD publishes its own `total_detections`, and "error < 0 means not
a detection" reproduces it on all 3475 rows. That agreement is recomputed
every run (`summary.json["limit_bookkeeping"]`).

**The sinking timescales are in the catalogue.** PEWDD publishes τ_Z per star
per element (`SinTimeCa`, …) for that star's own structure. Where a row has
them for the whole panel they are used verbatim. They also calibrate the
rest: log10(τ_Z/τ_Ca) is nearly constant across the 95 rows that publish it
(interquartile widths 0.03–0.15 dex, no significant Teff trend, H and He
agreeing to ~0.01 dex), so those rows build a relative library used for every
other panel. The library shows the embedded mass-scaling law was ~60 % too
shallow (Mg +0.171 measured against +0.098 assumed), which biased every
steady-state and declining-phase correction. The source actually used is
recorded per panel.

**One star, several rows.** PEWDD distinguishes alternative solutions for one
star by a name suffix — `PG1225-079 Model 2`, `GD 362 Updated`,
`WDJ0649-7624 (phot)`. Ungrouped, one star entered the misfit list five times
and never saw its own object's other panels, so the multi-reference kill could
not fire. Qualifiers are stripped before grouping.

---

## 6. The population

From the served table (3547 rows, 2778 distinct stars, VizieR TAP):

| n measured elements | rows |
|---|---|
| 0–1 | 2723 |
| 2–4 | 615 |
| **≥ 5 (screened and calibrated)** | **209 rows / 178 stars** |

Atmospheres: 3033 He, 514 H, resolved from PEWDD's own `atmosphere` column.
Abundances are `log(Z/H(e))` throughout — H in a hydrogen atmosphere, He in a
helium one; the science uses only element ratios, in which the reference
cancels, but the sinking timescales do not, so the reference is resolved and
recorded per row.

## 7. Controls

The literature outliers run as controls and their landing is reported in
`results/slag/controls.json` whatever it is. Their PEWDD names, found on the
runner: `PG 1225-079` (+ `Updated`, + `Model 1/2/3`), `LHS 2534`,
`GALEXJ2339`, `GD 378`, `NLTT 19868`, `HE 0106-3253` (= WD 0106−328),
`GD 362` (+ `Updated`, + `GD362`).

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
`stage=all`, `shards=8`.
