# ULINE — industrial fluorine molecules in public unidentified-line lists

**Signature S54** (`docs/necrofrontier.md` §2): *the molecule life and
chemistry both refuse.*  The third channel in the necrofrontier build order,
and the one where a single confirmed pattern is decisive.

---

## 1. The claim

No interstellar molecule with more than one fluorine atom has ever been
detected.  Fluorine chemistry in the ISM terminates at HF and CF⁺ (Neufeld &
Wolfire 2009); the gas-phase and grain networks find no route to CF₂ or CF₃
(Acharyya & Herbst 2017).  Terrestrial biochemistry makes no N–F, S–F or fully
fluorinated C–F bonds (Seager et al. 2023).  Fluorinated small molecules are,
on Earth, *only* made by industry: refrigerants (CFC-11 CFCl₃, CFC-12 CF₂Cl₂,
CFC-13 CF₃Cl, HCFC-22 CHClF₂, HFC-32 CH₂F₂, HFC-23 CHF₃), semiconductor
etchants and cleaning gases (NF₃, COF₂, CF₃CN, the CF₂ radical), fumigants
(SO₂F₂).

A rotational line **pattern** — several transitions of one of these species
at one excitation temperature, at the source velocity, in a molecular cloud, a
circumstellar envelope or a debris-disc gas — is therefore a residue of
chemistry, not of astrochemistry.  The test is run on the **published
unidentified-line (U-line) tables** of the deep line surveys: the lines the
authors could not assign to any known species, isotopologue or vibrational
state.  A U-line list is exactly the place an unexpected species would have
been left.

**Not the target.**  CH₃Cl (detected toward IRAS 16293−2422 and in 67P:
Fayolle et al. 2017) and CH₃F are natural organohalogens: single-halogen,
reachable from HCl/HF chemistry.  They are the **baseline and a veto**: a
U-line that coincides with a CH₃Cl or CH₃F line is not evidence for anything
artificial.  The perfluorinated gases of the exoplanet literature (CF₄, SF₆,
C₂F₆) have no permanent dipole and no rotational spectrum; they are invisible
to this channel by construction (§6).

---

## 2. Novelty position

`docs/necrofrontier.md` §5 row **g7** (sweep runs 3–4, 2,135 abstracts,
decoy-aware): **unoccupied**.  The nine decoy-free hits are generic
technosignature reviews; every paper that mentions a CFC is an
*exoplanet-atmosphere* paper (Lin et al. 2014; Haqq-Misra et al. 2022;
Schwieterman et al. 2024; the 2026 review calling PFCs an "extinct
technosignature") — transmission/emission spectroscopy of planets, in the
infrared, of non-polar species.  No search of any ISM or circumstellar
**line survey** for an industrial species exists in the record, and no paper
reads a published U-line list against predicted rotational patterns of
fluorinated molecules.  The nearest neighbours are the surveys' own
identification passes (Crockett et al. 2014 §5; Cernicharo's IRC+10216 work),
which test the *natural* inventory of CDMS/JPL and stop.

---

## 3. Data

| Role | Source | What it gives | Reach |
|---|---|---|---|
| U-lines | Crockett et al. 2014, ApJ 787, 112 (Orion KL, Herschel/HIFI 480–1907 GHz) | ~13,000 lines, ~1,730 unidentified (12 %), with peak intensities | VizieR TAP `J/ApJ/787/112/` |
| U-lines | He et al. 2008, ApJS 177, 275 (IRC+10216, ARO 12 m + SMT, 2 mm and 1.3 mm) | 17 U-lines in a carbon-rich envelope | VizieR TAP `J/ApJS/177/275/` |
| further inventories | any VizieR table whose description mentions "unidentified" | recorded in `probe.json` for the next dispatch | discovered at runtime |
| line lists | JPL `catdir.cat` + `c<tag>.cat` | species directory, log Q on 7 temperatures, every line with LGINT/ELO/GUP | reached by the necrofrontier probe: NF₃, COF₂, CH₂F₂, CH₃Cl present; CHF₃, CH₃F, SO₂F₂, CF₃CN, CF₃Cl, CF₂Cl₂, CFCl₃, CHClF₂, CF₂ absent |
| line lists | CDMS classic roots (entries, predictions/catalog, partition_function) | same `.cat` format; extra Q temperatures | entries page is script-rendered; roots are mined by regex for whatever renders |
| predictor | `config/uline.yaml` `predicted:` | symmetric-top constants for CHF₃, CF₃Cl, CF₃CN, NF₃ | placeholders, every one `verify: true` |
| predictor | `src/seti/data_assets/rotor_constants.yaml` | **full asymmetric-top constants** (A, B, C, quartic where known, dipole components, spin-weight rule, both Cl isotopologues) for CF₂Cl₂, CFCl₃, SO₂F₂, CHClF₂, CF₂ — and for the three validation species SO₂, CH₂F₂, COF₂ | every block `verify: true`, each with a citation and an uncertainty class |

### 3.1 How a species is found in a catalogue

`catdir.cat` is **not** a fixed-width file.  The name may carry spaces, commas
and `+` (`H2O v2,2v2,v`, `N-atom-D-st`), the version column may read `1`, `2*`,
`1␣` or be absent, and extra trailing columns occur.  The parser therefore
reads each line **by field position from the right** — the trailing run of
decimal tokens is the log-Q grid, the integer to its left is the line count,
everything between the tag and that integer is the name — and any line it still
cannot read is counted and reproduced verbatim (first 20) in
`acquire.json → jpl.unparsed_sample`.  The rigid regex it replaced dropped
171 of 403 lines on run 34787988880 (16 of 32 on the head the necrofrontier
probe captured), which is why every target species came back empty.

Species are then matched by **normalised formula**, not by anchored regex:
both the catalogue name and the target formula are case-folded and stripped of
everything that is not a letter or a digit, and a match is declared when the
two are equal (`normalised`), equal after a leading isotope mass
(`isotopologue`: `13CH3OH` → `CH3OH`), or have the same atom counts
(`atom_counts`: `HCCCN` = `HC3N`, `SiCC` = `SiC2`, `F2CO` = `CF2O` = `COF2`,
`CH3-35Cl, v=0` = `CH3Cl`).  Charge is kept, so `CF+` never matches `CF`, and
`CF2Cl2` never matches `CF2`.  The `patterns:` regexes in `config/uline.yaml`
remain an **additional** route, never the only one.  Every entry whose
normalised name merely *contains* the formula is recorded per species as
`near_miss_names`, so a species that still matches nothing tells the next
dispatch the archive's actual spelling instead of leaving it blind.

### 3.2 VizieR TAP discovery

TAPVizieR stores `table_name` **with its literal double quotes**, so every
`LIKE` opens with a leading `%` (`LIKE '%J/ApJ/787/112/%'`), `TAP_SCHEMA.columns`
is queried for both the bare and the quoted spelling, and the table is quoted
in `FROM`.  Every ADQL sent is recorded with its row count, and every failure
with its **full error text**, in `acquire.json → sources.<name>.queries` /
`.errors`; run 34787988880 recorded a bare `QUERY_FAILED` and was
undiagnosable for it (the cause was a TAPVizieR 503).

When an asserted catalogue id returns no table, or none with a usable
frequency + identification pair, a **description search** runs
(`description LIKE '%unidentified%'` AND one of `'%Orion%'` / `'%IRC+10216%'` /
`'%line survey%'`, case variants OR-ed) and **every** table it returns is
recorded in `sources.<name>.fallback`.  That is diagnostic only: no table is
ever selected from it automatically — a candidate found this way has to be
asserted in `config/uline.yaml` as a `vizier_like` with its own `verify` note
before it is used.

### 3.3 The VizieR route ladder (TAP is not the only door)

On 2026-09-13 TAPVizieR answered **503 to five attempts over `https://` and
five over `http://`** (`results/uline/summary.json → degraded`), and this
channel and ARC both reported `NO_DATA_REACHED` — for an infrastructure
reason, which is exactly the failure this repository must never publish as a
result about the sky.  VizieR is therefore now reached over four routes, tried
in order, **each recorded with its endpoint and its error text** so a failure
is always diagnosable (`seti.metronome.acquire`, shared with ARC):

| # | Route | Endpoint | Status |
|---|---|---|---|
| 1 | TAP | `https://tapvizier.cds.unistra.fr/TAPVizieR/tap` | current primary |
| 2 | TAP, second host | `https://tapvizier.u-strasbg.fr/TAPVizieR/tap` | **verify** — historical CDS hostname, asserted, unconfirmed |
| 2 | TAP, second host | `https://vizier.cfa.harvard.edu/TAPVizieR/tap` | **verify** — CfA mirror, asserted, unconfirmed |
| 2 | TAP, plain http | `http://tapvizier.cds.unistra.fr/TAPVizieR/tap` | same host, other spelling |
| 3 | **ASU (not TAP)** | `https://vizier.cds.unistra.fr/viz-bin/asu-tsv` (+ two **verify** mirrors) | the route most likely to work while TAP is down |
| 4 | astroquery | `astroquery.vizier.Vizier` | declared dependency; last resort |

The endpoints marked **verify** could not be tested from the sandbox (no
egress); they are asserted from the hostnames CDS and the CfA publish, and the
first runner dispatch that reaches — or fails to reach — them is what settles
it.  They are listed with the same note in `config/uline.yaml`, which
`tests/test_uline.py` pins to the module constants so the two cannot drift.

**Route 3 is the one that matters during an outage.**  `asu-tsv` is served by
the VizieR web application, not by the TAP service:

* rows: `?-source=<catalogue>/<table>&-out.max=<n>&-out=<column>…&-out.form=TSV`
  returns tab-separated text whose `#`-prefixed block carries one `#Column`
  line per column, followed by a header line (column names, sometimes
  double-quoted), a unit line, a rule of dashes and the data;
* **table existence**: `?-source=<catalogue>&-meta.all` lists the catalogue's
  tables and their columns — the non-TAP replacement for `TAP_SCHEMA`, with
  the catalogue's ReadMe (`cdsarc.cds.unistra.fr/ftp/<cat>/ReadMe`, whose File
  Summary and Byte-by-byte blocks give tables, row counts and column labels)
  as the backstop.

The narrow ADQL this repository emits (`SELECT [TOP n] cols FROM "table"
[WHERE recno BETWEEN a AND b | col IN (…)]`, `TAP_SCHEMA.tables … LIKE`,
`TAP_SCHEMA.columns … WHERE table_name =`) is translated to those forms
exactly; a `recno` window becomes "the first `hi` rows, keep the last
`hi−lo+1`", which is the same rows because ASU returns them in `recno` order.
Two forms have **no** ASU equivalent and are reported rather than guessed at:
`COUNT(*)` (the row count becomes *unknown*, and discovery ranks the table on
its columns alone) and the description search of §3.2 (which stays TAP-only).

A run served by route 3 reports `route: asu_tsv` — in
`probe.json → vizier.<source>.route`, in the winning entry of
`acquire.json → sources.<source>.queries`, in the acquisition log stages, and
in `df.attrs["route"]` — so an `OK` always says which door it came through.
When **every** route fails, every endpoint and every error text is in
`sources.<source>.routes` and the status stays an honest `QUERY_FAILED`: no
route ever fabricates a row.

Not yet reachable as tables (listed in S54, deferred): QUIJOTE TMC-1 (1,591
features, 188 unidentified), GOTHAM, ALCHEMI, PILS, ReMoCA, PRIMOS.  The
description-word discovery in the probe stage is how they enter: a table that
appears there gets a `sources:` entry with its v_LSR and linewidth and is
screened on the next run.

---

## 4. Method

### 4.1 Line lists at T_ex

For each species the run takes **one** list for matching (preference
`cdms → jpl → predicted`, configurable) and reports which was used.  Every
catalogue line is rescaled from 300 K to each T_ex on the grid
(10, 30, 50, 100, 150, 200 K) with the catalogue's own partition function:

$$
I(T) = I(300)\,\frac{Q(300)}{Q(T)}\,
\frac{e^{-c_2 E_l/T} - e^{-c_2 E_u/T}}{e^{-c_2 E_l/300} - e^{-c_2 E_u/300}},
\qquad E_u = E_l + \nu/c,\quad c_2 = hc/k = 1.438777\ \mathrm{cm\,K},
$$

with $\log Q$ interpolated linearly in $\log T$ on the tabulated grid
(Pickett et al. 1998).  The Boltzmann difference is evaluated in log space so
high-lying lines underflow to $-\infty$ rather than NaN.

For a species with no catalogue entry but placeholder constants, a symmetric
top R-branch is predicted:

$$
\nu(J\to J+1, K) = 2B(J+1) - 4D_J(J+1)^3 - 2D_{JK}(J+1)K^2,
$$

with intensity $\propto \nu\,S\,\mu^2\,g_K\,g_{ns}\,
[e^{-c_2E_l/T}-e^{-c_2E_u/T}]/Q(T)$, line strength
$S=((J+1)^2-K^2)/(J+1)$, $g_K = 1$ (K = 0) or 2, a K = 3n nuclear-spin weight
of 2 for C₃ᵥ tops with three equivalent I = ½ nuclei,
$E_l = BJ(J+1) + (X-B)K^2$ with $X$ the axial constant (C for oblate CHF₃; when
unknown the K ladder is Boltzmann-flat and the run says so), and $Q(T)$ summed
explicitly over the same levels.  Absolute scale is approximate; only relative
intensities are used.  Predicted frequencies carry an error
$\sigma = 0.5\ \mathrm{MHz} + 10^{-5}\nu$ (both distortion constants known) or
$10^{-4}\nu$ (unknown: the $4D_J(J+1)^3$ term is unmodelled).

### 4.1a The asymmetric-top predictor (the five uncatalogued species)

Five of the most diagnostic species — CF₂Cl₂, CFCl₃, SO₂F₂, CHClF₂ and the
CF₂ radical — have **no entry in JPL or CDMS** and were reported
`targets_unsearchable` by every earlier dispatch.  They are the most
diagnostic precisely because they are long-lived and purely artificial; their
absence from the *catalogues*, not from the sky, is what excluded them.  A
rotational spectrum is a computable object, so `src/seti/uline/rotor.py`
computes it: the **Watson A-reduced Hamiltonian** (Watson 1977; Gordy & Cook
1984 §8.5) with quartic *and* sextic distortion, set up in the |J,K⟩ basis and
block-diagonalised in the Wang basis, in either the `Ir` (z = a) or `IIIr`
(z = c) representation; levels labelled J_{Ka Kc} by the block's fixed K parity
and fixed Kₐ+K_c sum (robust where the τ-ordering theorem alone fails on a
near-degenerate K-doublet); line strengths from the eigenvectors and the
Clebsch–Gordan coefficients of the molecule-fixed dipole components, which
reduce exactly to S = ((J+1)²−K²)/(J+1) in the symmetric-top limit and obey
Σ S_g = 2J+1; intensities in the JPL convention with the partition function
summed explicitly over the same levels, so the output plugs into the same
`rescale_lgint` as a catalogue entry.  Nuclear-spin weights are a rule on the
parities of (Kₐ, K_c); quadrupole hyperfine structure is **not** modelled — its
blend width enters the frequency uncertainty instead.

**Validation is measured, not asserted.**  Offline
(`tests/test_uline_rotor.py`): closed-form J = 1 and 2 energies, the exact
symmetric-top limit including ΔJ/ΔJK/ΔK, the Wang-block labels against the
τ-ordering theorem, the three dipole selection rules, the strength sum rule,
representation invariance, and laboratory SO₂ lines.  On the runner
(`--stage validate`): SO₂, CH₂F₂ and COF₂ are predicted and compared **line by
line** with their JPL/CDMS entries, and then A, B, C and the quartic constants
are **refitted to the catalogue's own frequencies**.  The two residuals answer
different questions and `results/uline/rotor_validation.json` reports both:
the residual *before* the refit measures the embedded **constants** (recalled
from the literature, expected to be off); `rms_after_mhz` measures the
**Hamiltonian** and is the only number behind any claim that this predictor
reproduces a catalogue.  A third, `distortion_truncation`, re-predicts with the
quartic terms zeroed — the size of the error a species with unknown quartic
constants carries, at the same J.

**The distortion truncation is the binding limit, and the run says so.**  For
a heavy rotor an R-branch line at J ≈ 30 is displaced by ~4ΔJ(J+1)³, which is
orders of magnitude wider than any survey's matching tolerance.  Measured on
the embedded constants in the IRC+10216 2 mm band (129–172 GHz), the median
predicted-frequency error is **CF₂Cl₂ 138, CFCl₃ 60, SO₂F₂ 146, CHClF₂ 150,
CF₂ 92 MHz**, against a 30 km s⁻¹ linewidth that is ~15 MHz there and an Orion
KL linewidth of ~2 MHz.  Those figures are *after* an audit of the embedded
A, B, C uncertainties: a `recalled` constant is known only as well as the
independent check available for it, and where the r0 structure in the same
block disagreed by more than the stated uncertainty the uncertainty was raised
to the disagreement (CFCl₃ 0.5 → 21 MHz, CHClF₂ 0.5 → 13 MHz, and the
isotopologues scaled from them with it).  That moves CFCl₃ to 1.43 GHz and
CHClF₂ to 677 MHz — honest widths for numbers that were recalled rather than
read.  It matters in the dangerous direction: the tolerance is
max(linewidth, σ_pred, σ_U), so an error bound that is too small manufactures
coincidences.  The current statuses are CF₂Cl₂ ×9 **DEGRADED**, SO₂F₂ ×10
DEGRADED, CF₂ ×6 DEGRADED, CHClF₂ ×45 **FREQUENCY_LIMITED**, CFCl₃ ×95
FREQUENCY_LIMITED, against the IRC+10216 tolerance.

When σ_pred exceeds the linewidth the "tolerance" the
pattern test uses is the prediction's own ignorance, chance alignments rise
with it, and the rigid-shift FAP can never reach its gate.  That is reported
per species × source as `searchability`
(`SEARCHABLE` / `DEGRADED` / `FREQUENCY_LIMITED`) and rolled up as
`summary.json → targets_frequency_limited`, with the remedy named: **the
published quartic constants, not more data.**  `--stage litfetch` is the route
to them — CCCBDB, the NIST Triatomic Spectral Database, PubMed, Zenodo, the
JPL documentation files — and nothing it fetches ever silently overwrites an
embedded value: the two are written side by side in `literature.json` and
promoting one is a commit to `rotor_constants.yaml`.

**One species is model-limited, not only constant-limited.**  SO₂F₂ is an
*accidentally near-spherical* top (A ≈ B ≈ C).  Sarka, Demaison, Margulès et
al. found that Watson's **A-reduction fails** for it, that the S-reduction does
better, and that an *unreduced* Hamiltonian was required — in which six rather
than five quartic constants are determinable, the first asymmetric top for
which all six were measured.  This predictor is A-reduced, so for SO₂F₂ alone,
obtaining the published quartic set would not by itself make the prediction
right: the Hamiltonian would have to be extended.  That is recorded in the
species' `hamiltonian_caveat` rather than left to surface later as an
unexplained residual.

**A `verify` line alone never makes a candidate.**  Every block in
`rotor_constants.yaml` is `verify: true`.  A pattern found only on those
frequencies gives the verdict `PATTERN_CANDIDATE_VERIFY_CONSTANTS`, with the
pairs named in `pattern_candidates_verify_constants` — a reason to obtain the
laboratory line list, never a detection.  `PATTERN_CANDIDATE` is reserved for a
pattern on a catalogued list.

### 4.2 Features, geometry, tolerance

K-components of one J→J+1 transition are closer than a source linewidth for
most J (CHF₃ at J = 12: 0.47·K² MHz vs a 3.6 MHz linewidth at 269 GHz) and are
**one line** in a survey.  Predicted lines are therefore **merged into
features** within the tolerance before anything is counted — otherwise one
U-line "coincides" with six K-components and the count is inflated six-fold.

Rest frequencies are Doppler-shifted (radio convention) only when the survey
tabulates *sky* frequencies (`frequency_frame: sky`).  Both configured
surveys are expected to list rest frequencies at their assumed v_LSR (the
probe records the frequency column's description for the words *rest / LSR /
observed*), so their `v_lsr_km_s` (Orion KL 9, IRC+10216 −26) is recorded and
the spread between Orion KL's velocity components (hot core ~5.5, compact
ridge ~7–8, plateau ~8–9, extended ridge ~9 km s⁻¹) enters as a tolerance
term.  Pair tolerance:

$$
\Delta\nu = \max\!\big(\nu\,\mathrm{FWHM}/c,\ \sigma_{\rm pred},\ \sigma_{\rm U}\big) + \nu\,\Delta v_{\rm unc}/c .
$$

Linewidths: Orion KL 4 km s⁻¹, IRC+10216 30 km s⁻¹ (expansion 14.5 km s⁻¹).

### 4.3 The pattern test (per species × source × T_ex)

1. Features in the survey's **coverage**: some catalogued line (identified or
   U) within 1.5 GHz — a data-driven band mask that respects HIFI's band gaps.
2. The **top-N (40)** features by intensity at T_ex.
3. **Vetoes first.** A U-line within tolerance of a CH₃Cl / CH₃F line, or of a
   CH₃OH / CH₃CN / HCOOCH₃ / C₂H₅CN line (the usual U-line explanation: a
   vibrationally excited or torsional state, an isotopologue) within 4 dex of
   that species' strongest in-range line at the source's contaminant T_ex, is
   removed from the clean list and counted per species.  IRC+10216 adds HC₃N,
   SiC₂, C₄H.  Contaminant lists take the *union* of JPL and CDMS.
4. **Coincidences**: one-to-one nearest-neighbour matches of features to clean
   U-lines.  A PATTERN needs **≥ 3**.
5. **LTE consistency**: Spearman ρ between predicted intensity and the
   observed U-line peak/area over the coincident set, ρ ≥ 0.3 (untestable when
   the survey has no intensity column — reported, and the gate is then the
   next test alone); and **no strong line missing**: ≥ 60 % of the five
   strongest predicted features in coverage must coincide with *some*
   catalogued line (a U-line, or a blend with an identified line — reported
   separately).  A strong predicted line that is neither is missing, and no
   excitation model repairs that.
6. **False-alarm probability**: ≥ 1000 trials shift the whole feature set
   rigidly by ±U(50, 500) MHz — the pattern's internal spacing survives, its
   alignment with the list does not — and recount.  `p_false = (k+1)/(N+1)`
   with k the trials reaching the observed count; also `p_false_full` for
   trials that pass the count and top-5 tests.  Gate: `p_false ≤ 0.01`.

A species × source pair that passes all four at any T_ex is a
`PATTERN_CANDIDATE`; the best T_ex is quoted.  The candidate then owes the
S54 follow-up that this channel does *not* do: a line-by-line vet against the
full CDMS/JPL inventory including isotopologues and vibrational states, a
check that the implied column does not exceed the HF/CF⁺ fluorine budget, and
a matched-filter stack in the archive cubes.

### 4.4 Verdicts

| Verdict | Meaning |
|---|---|
| `NO_DATA_REACHED` | no U-line list acquired, or no species line list at all — says nothing about the sky |
| `NO_PATTERN` | lists compared; nothing met every test — a count, not an abundance limit, not written up |
| `PATTERN_CANDIDATE` | ≥ 1 species × source met every test **on a catalogued (laboratory) line list** — pending vet |
| `PATTERN_CANDIDATE_VERIFY_CONSTANTS` | the only passing pairs rest on `verify` constants reconstructed from the literature — a reason to obtain the laboratory line list, never a detection |

Degradation (a failed archive, an unsearchable species, predicted rather than
laboratory frequencies, a source without intensities) is the `degraded` list
in `summary.json`, never folded into the verdict string.

---

## 5. Contamination ledger

| Mechanism | Handling |
|---|---|
| Vibrationally excited / torsional states of abundant complex organics — the dominant U-line explanation | veto against CH₃OH (incl. vt), CH₃CN, HCOOCH₃, C₂H₅CN catalogue lines within tolerance (4 dex dynamic range); counted per species |
| Natural organohalogens | CH₃Cl (both Cl isotopologues), CH₃F veto |
| Isotopologues of the contaminants | regexes admit ¹³CH₃OH, CH₃¹³CN etc. where the archive names them; a residual leak is stated |
| K-component blends inflating the count | features merged within tolerance before counting |
| Chance alignment in a dense list (Orion: ~9 lines/GHz) | rigid-shift FAP with the list's own density; one-to-one matching |
| Excitation-inconsistent "patterns" | Spearman on intensities + missing-strong-line test |
| Wrong velocity frame | `frequency_frame` per source; probe records the column description; a sky-frame mistake is a 26 km s⁻¹ error for IRC+10216 and would show as `NO_PATTERN` |
| Hyperfine (Cl, N) splitting in predicted species | unmodelled; inside the tolerance at these linewidths for the Orion width, stated |
| Placeholder rotational constants | every `predicted:` block is `verify: true`; `summary.json["targets_with_predicted_frequencies"]` names them; the predicted error model is 10× wider without distortion constants |
| Band gaps counted as "missing" lines | coverage mask from the survey's own catalogued lines |
| Instrumental features recurring across sources | a coincident U-line frequency appearing in both surveys at different v_LSR is instrumental — visible in `coincidences.csv` |

---

## 6. Limits — stated, not hidden

* **No-dipole species are radio-invisible.**  CF₄, SF₆, C₂F₆, and the
  symmetric CF₂Cl₂/CFCl₃ isotopologue mixes with small dipoles are weak or
  absent here; that is the IR track (SF₆ 10.55 µm, CF₄ 7.8 µm, NF₃ 907 cm⁻¹
  against MIRI/MRS and ISO-SWS), not this channel.
* **The catalogue gap is closed; the constants gap is not.**  CF₂Cl₂, CFCl₃,
  SO₂F₂, CHClF₂ and CF₂ are no longer `targets_unsearchable`: §4.1a predicts
  their spectra from embedded constants.  What still binds is that their
  **quartic centrifugal-distortion constants are published but were not
  reachable** from the build sandbox, so the predicted frequencies carry
  60–150 MHz of truncation error and the species come back
  `FREQUENCY_LIMITED`.  That is a statement about the constants, not the sky,
  and the fix is the published quartic set (`--stage litfetch`), not more
  data.
* **Predicted constants are `verify`** — recalled or read off a search-engine
  excerpt, with the uncertainty class (`lab` / `recalled` / `snippet` /
  `structure` / `estimate`) stated per isotopologue.  A pattern found on
  predicted frequencies gives `PATTERN_CANDIDATE_VERIFY_CONSTANTS`, which is a
  reason to obtain the laboratory list, not a detection.
* **A U-line list is the survey's residue**, not the sky: lines the authors
  assigned (rightly or wrongly) are not U-lines.  A species whose strongest
  transitions were mis-assigned to something else is invisible here — which
  is why the top-5 test allows blends with *identified* lines and reports the
  U-only fraction alongside.
* **The U-line sample.**  Orion KL is oxygen-rich and hot; IRC+10216 is
  carbon-rich; Sgr B2 is a hot core in a third chemistry; comet Lovejoy is a
  coma, with a tolerance an order of magnitude tighter than any of them.  The
  Crockett+2014 Orion KL list (~1,730 U-lines *with intensities* — the only
  one with real statistics) is **measured absent from VizieR** over three
  independent routes (§6a), and is now sought over the IOP CDN, the article's
  `suppdata` directory, the CDS ftp mirror and IRSA's HEXOS delivery, each
  recorded with what it answered.  The U-line **census** (`probe.json →
  uline_column_census`) is how the sample grows: one TAP_SCHEMA.columns query
  on phrases only a spectral U-line table carries.  Nothing found there is
  screened until it is asserted under `sources:` with its own v_LSR,
  linewidth and `verify` note — guessing a velocity puts a wrong Doppler
  shift on every frequency.
* **The LTE test needs an intensity column.**  Runs 35039822190 and
  35041128720 reported `lte_testable: false` for every pair, which was a
  *column-matching* failure, not a property of the surveys: Cernicharo+2000
  carries `T(MB)dv` and He+2008 `Iint`, and neither matched the old regex
  list.  The integrated forms now come first.  A source with genuinely no
  intensity column is reported, and the gate falls back to the
  missing-strong-line test alone.

---

## 7. Running

```
python -m seti.uline.run --stage probe            # archive inventory; cheap
python -m seti.uline.run --stage all              # probe, acquire, screen, assess
python -m seti.uline.run --stage screen --species CHF3,NF3 --n-trials 5000
python -m seti.uline.run --stage validate         # the predictor vs SO2/CH2F2/COF2
python -m seti.uline.run --stage litfetch         # the microwave literature ladder
python -m seti.uline.run --stage propose          # offline: adjudicate that ledger
```

The workflow input `stage: full` runs everything in one dispatch, in the order
**probe → acquire → screen → assess → validate → litfetch → propose**: the
search writes `summary.json` *before* the literature ladder starts, so a slow
or hanging service can cost only its own output, never the verdict.  The
ladder also carries its own wall clock (`archives.litfetch_budget_s`), and a
source it does not reach is recorded `NOT_ATTEMPTED` rather than as a route
that answered with nothing.

Workflow `.github/workflows/uline.yml` (`workflow_dispatch`: `stage`,
`sources`, `species`, `n_trials`, `reduce_only_run_id`); results commit back
through `scripts/commit_results.sh`.  Outputs in `results/uline/`:
`probe.json`, `acquire.json`, `acquisition_log.json`, `screen.json`,
`coincidences.csv`, `summary.json`, `candidates.csv`,
`rotor_validation.json`, `literature.json`, `constants_proposal.json`.  Offline suite:
`tests/test_uline.py` — which, for §3.3, scripts a TAP that 503s every query
against a VizieR that answers over ASU (discovery off `-meta.all`, rows off
`asu-tsv`, reported as `route: asu_tsv`) and the world where every route is
down (`QUERY_FAILED`, every endpoint named, no rows invented).  No test opens
a socket: every route takes an injectable fetch/query callable.
