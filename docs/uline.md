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
| `PATTERN_CANDIDATE` | ≥ 1 species × source met every test — pending vet |

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
* **Only species with a line list are searched.**  The necrofrontier probe
  found NF₃, COF₂, CH₂F₂, CH₃Cl in JPL; CDMS may add CHF₃, CH₃F.  CF₂Cl₂,
  CFCl₃, SO₂F₂, CHClF₂ and CF₂ are asymmetric tops (or radicals) for which no
  predictor is offered; without a catalogue entry they are reported
  `targets_unsearchable`, which is a statement about the archives.
* **Predicted constants are placeholders** (B, D_J, D_JK for CHF₃ from the
  channel brief; B only for CF₃Cl, CF₃CN, NF₃).  A pattern found on predicted
  frequencies is a reason to obtain the laboratory list, not a detection.
* **A U-line list is the survey's residue**, not the sky: lines the authors
  assigned (rightly or wrongly) are not U-lines.  A species whose strongest
  transitions were mis-assigned to something else is invisible here — which
  is why the top-5 test allows blends with *identified* lines and reports the
  U-only fraction alongside.
* **Two sources.**  Orion KL is oxygen-rich and hot; IRC+10216 is carbon-rich
  and the list has 17 entries.  Reach is one dispatch; the description-word
  discovery is how the list grows.

---

## 7. Running

```
python -m seti.uline.run --stage probe            # archive inventory; cheap
python -m seti.uline.run --stage all              # probe, acquire, screen, assess
python -m seti.uline.run --stage screen --species CHF3,NF3 --n-trials 5000
```

Workflow `.github/workflows/uline.yml` (`workflow_dispatch`: `stage`,
`sources`, `species`, `n_trials`, `reduce_only_run_id`); results commit back
through `scripts/commit_results.sh`.  Outputs in `results/uline/`:
`probe.json`, `acquire.json`, `acquisition_log.json`, `screen.json`,
`coincidences.csv`, `summary.json`, `candidates.csv`.  Offline suite:
`tests/test_uline.py`.
