# SHROUD — enshrouded, not destroyed

**Necrosignature S33.** A star that vanishes from the optical sky but is *still
there in the infrared* has not been destroyed. It has been **enshrouded** — and
an enshrouded star is the completed-Dyson / dead-swarm-dust endpoint. A star
that vanishes with **no** counterpart at any wavelength is a different and much
more extreme claim. The two populations together give the first measurement of
the **ratio of obscuration to destruction**, and the obscured population is
where a completed structure would live.

---

## 1. Scoping statement — read this first

This channel uses **only the catalogue by-product** of Solano, Villarroel &
Rodrigo 2022 (MNRAS 515, 1380; arXiv:2206.00907): the optical-absent /
infrared-present crossmatch, distributed as two SVO Virtual Observatory
catalogues. That crossmatch is an ordinary archival position match and is
**independent of the interpretation** placed on the parent sample.

**This channel does not build on, cite approvingly, or attempt to replicate the
VASCO transient claims** — the POSS-I multiple-transient events, the
Earth-shadow analysis, or the nuclear-test correlation. Those are under severe
and, in this project's judgement, largely successful attack:

* **Hambly & Blair 2024** (arXiv:2402.00497): the transients are "likely to be
  spurious artefacts of the photographic emulsion," plausibly introduced by the
  plate-copying procedure.
* **Watters et al. 2026** (arXiv:2601.21946): the spatially-uniform-random
  background assumption underlying the Earth-shadow analysis is **false**; with
  the correct null there is **no significant shadow deficit**; the nuclear-test
  correlation becomes insignificant after normalising by observation days and is
  "almost completely determined by the observation schedule of the Palomar
  telescope." They further document unvalidated datasets containing catalogue
  stars, scan artefacts and plate defects, and a number-density gradient toward
  plate corners and edges.
* The 2026 "independent replications" are single-author, unrefereed, and
  **contradict each other**: Hayes (arXiv:2604.04810) reproduces the catalogue
  well but finds the nuclear correlation *not* significant, while Doherty and
  Cann claim support.

**Contamination is characterised here, not inherited.** If the underlying source
list is polluted by plate defects — and it certainly is — that is a systematic
this channel measures. The key structural point:

> **A plate defect has no infrared counterpart.** An emulsion flaw, a scan
> artefact, or a dust speck on a glass copy plate does not emit at 3–22 µm.
> Requiring a *real infrared detection* is therefore itself a strong artifact
> filter, and it is applied at selection time rather than as follow-up.

The residual worry is the opposite one — a defect that lands *by chance* within
5″ of an unrelated infrared source. That is quantified in §5, and it is the
single largest systematic in the sample.

---

## 2. The opportunity

Solano+2022 reports, verbatim:

> "We found **298 165 sources visible only in POSS I plates**, out of which
> 288 770 had a crossmatch within 5 arcsec in other archives (mainly in the
> infrared), 189 were classified as asteroids, 35 as variable objects, 3 592 as
> artefacts from the comparison to a second digitization (Supercosmos), and 180
> as high proper motion objects… The remaining unidentified transients
> (**5 399**) as well as **the 172 163 sources not detected in the optical but
> identified in the infrared regime are available from a Virtual Observatory
> compliant archive** and can be of interest in searches for strong M-dwarf
> flares, high-redshift supernovae, asteroids, or other categories of
> unidentified red transients."

That is where the analysis stopped. **No population analysis. No SED modelling.
No obscuration-vs-destruction test.** A 172 163-source sample of objects that
are absent in the modern optical and present in the infrared was published as a
by-product and set aside, and a 5 399-object sample of vanished sources *without*
any counterpart was published alongside it as the natural control.

### Where the data actually is

| Product | Endpoint | Rows |
|---|---|---|
| IR-present sample ("W") | `http://svocats.cab.inta-csic.es/vanish-neowise/` | 171 753 |
| No-counterpart control ("R") | `http://svocats.cab.inta-csic.es/vanish-possi/` | 5 399 |
| Villarroel+2020 candidates | VizieR `J/AJ/159/8` table2 (99) + table3 (28) | 127 |

Both SVO URLs are quoted verbatim in the Table 1 footnotes of Watters et al.
2026 ("Retrieved on 2025-12-08 from …") and independently in the README of the
`jannefi/vasco` reimplementation.

Two facts worth recording so they are not re-derived:

1. **Solano+2022 has no VizieR catalogue.** A runner-side fetch of
   `J/MNRAS/515/1380` on 2026-07-26 returned `Table or Catalog not found`
   (the response is committed at `results/disaplit2/vizier_solano_2022.xml`).
   The SVO endpoints are the only machine-readable route.
2. **Expect 171 753 rows, not 172 163.** Watters et al. downloaded the live
   archive on 2025-12-08 and state the smaller number four separate times. The
   channel validates against 171 753 and warns rather than failing.

### The selection function, inherited and stated

Sample "S" (298 165) was built by removing everything within 5″ of a **Gaia DR3
or Pan-STARRS DR2** source. So the sample is *by construction* free of modern
optical counterparts at that radius — which means the modern optical flux is a
genuine upper limit, and the energy budget's denominator is essentially the full
plate flux. It also means the `MODERN_OPTICAL_MATCH` and `VARIABLE_STAR` classes
can only be populated from catalogues Solano+2022 did not use.

Watters et al. state that dataset W "comprises an undetermined number of NeoWISE
objects along with an undetermined number of incidentally object-proximate
[plate features]". **Determining that number is a deliverable of this channel.**

---

## 3. The measurement: energy conservation, with distance cancelled

If the object was enshrouded rather than destroyed, the optical light it lost
was absorbed by circumstellar material and re-radiated in the thermal infrared:

```
eta  =  F_IR(now)  /  [ F_bol(POSS-I)  -  F_bol(modern optical) ]
```

**`eta` is a pure ratio of fluxes, so the distance cancels exactly.** Nothing in
this test needs a parallax — which is what makes it usable on a catalogue of
anonymous 1953 plate detections.

A vanished source normally has exactly **one** historical magnitude, so
`F_bol(then)` is not uniquely determined: it depends on the unknown progenitor
temperature. The channel therefore reports a *bounded* quantity rather than a
point estimate. For a blackbody, `F_bol / F_nu(lambda)` is minimised at
`h*nu/kT ≈ 3.92` (T ≈ 5 700 K for the POSS-I E band) — that temperature demands
the least missing energy and so gives the **largest possible** eta.

| Regime | Meaning | Robustness |
|---|---|---|
| `eta_max < 0.1` | **`IR_TOO_FAINT`** — the infrared cannot be the missing optical light | holds for *every* allowed progenitor temperature |
| `eta_max ≥ 0.3`, `eta_lo ≤ 3` | **`ENERGY_CONSERVING_OBSCURATION`** — enshrouded, energy conserved | holds for *some* progenitor temperature |
| `eta_lo > 3` | **`IR_EXCEEDS_MISSING`** — intrinsically IR-luminous, or an unrelated IR match | holds for *every* progenitor temperature |
| `< 3` IR bands, no fitted `T_dust` | **`IR_UNDERSAMPLED`** — no verdict issued | see below |

The two verdicts that matter are the ones that hold for *all* progenitor
temperatures. Interstellar extinction pushes in a helpful direction: it makes
the observed `F_bol(then)` an underestimate, hence `eta` an overestimate — so an
`IR_TOO_FAINT` verdict is conservative and would only strengthen on dereddening.

### Why `IR_UNDERSAMPLED` exists

`vanish-neowise` was matched against **NeoWISE, which carries only W1 and W2**.
With two bands the model-free infrared integral spans a sliver of a thermal SED
and badly *under*-estimates the total — which would manufacture the channel's
headline `IR_TOO_FAINT` result out of missing photometry. The code refuses to
issue a deficit verdict in that regime and says why. This is the entire reason
the pipeline joins AllWISE W3/W4 and 2MASS before believing anything.

### The Forés-Toribio & Kochanek kill-test is the same measurement

Forés-Toribio & Kochanek 2026 (arXiv:2604.05019) show that **merger remnants are
10–100× MORE luminous than their progenitors at late phases, whereas genuine
disappearance remnants are ~10× DIMMER**, and that asymmetric dust cannot
manufacture a factor ~100. Because an obscured object's present-day bolometric
output is dominated by its reprocessed infrared, that progenitor-to-remnant ratio
*is* eta. It is applied as a standing discriminant at `≥ 10` (merger remnant,
rejected) and `≤ 0.3` (genuine disappearance).

### Two competing SED models

Both are fitted to the **present-day** bands only — the 1953 plate point is
60+ years older and never enters a fit of the current SED, only the budget.

* **(a) reddened photosphere** — `(T_eff, scale, A_V)`
* **(b) obscured star + warm dust** — `(T_eff, scale_star, A_V, T_dust, scale_dust)`,
  where the star suffers the total column and the dust emission only the
  foreground one.

Non-detections are used, not discarded: a model brighter than a published upper
limit is penalised quadratically. Fitted dust hotter than 1 800 K is a companion
photosphere, not grains (repository ledger).

---

## 4. Population decomposition

The overwhelming majority of this sample is mundane, and the breakdown is itself
the first-ever analysis of it. Classes are assigned by a first-match cascade:

| class | test |
|---|---|
| `PLATE_DEFECT` | no counterpart at any wavelength |
| `ASTEROID` | as above, near the ecliptic |
| `MODERN_OPTICAL_MATCH` | a modern optical source of comparable brightness — it never vanished |
| `HIGH_PM_STAR` | a modern source propagates *back* onto the plate position |
| `VARIABLE_STAR` | modern optical counterpart ≥ 1 mag fainter — caught bright on the plate |
| `BLEND_CONFUSION` | > 1 infrared source inside the WISE PSF |
| `AGN_QSO` | red W1−W2 **and** a power-law mid-IR SED |
| `AGN_QSO_COLOUR_ONLY` | red W1−W2 with < 3 IR bands — formally undecidable |
| `GALAXY` | flagged extended in the infrared catalogue |
| `DUSTY_AGB` | bright, very red, rising through W3/W4 |
| `YSO` | low galactic latitude with a rising mid-IR SED |
| `RESIDUAL_UNEXPLAINED` | everything the cascade could not name — the only class that reaches the budget |

### The AGN cut had to be rebuilt

A naive Stern et al. 2012 mid-IR AGN cut (`W1−W2 ≥ 0.8`) **would delete exactly
the population this channel exists to find**: a 350 K shroud has `W1−W2 = 3.2`,
far redder than anything that wedge was calibrated on. Colour alone cannot
separate an AGN from an enshrouded star.

The separable axis is **SED shape**: an AGN is a power law across 3–22 µm, a
shroud is a *curved* single-temperature blackbody. On the channel's synthetic
pair the discriminant is unambiguous — for a `beta = −1` power law,
`chi2_PL ≈ 0` against `chi2_BB = 439`; for a 350 K shroud, `chi2_PL = 978`
against `chi2_BB ≈ 0`. With fewer than three infrared bands the two hypotheses
are not distinguishable even in principle, and the class says so rather than
guessing. (This is a *shape* test, which the repository's literature sweep
records as the axis every published catalogue search omitted.)

### Proper motion, done in the right direction

A star with µ = 200 mas/yr travels **12.6″** between the POSS-I epoch and Gaia
DR3 — far outside a 5″ match radius, so it is simply *absent* from its 1953
position. Solano+2022 removed 180 such objects from 298 165, which is
implausibly few. The channel searches a radius set by the largest proper motion
considered and propagates every modern neighbour **back** to the plate epoch,
rather than propagating the plate position forward with a proper motion the
vanished source by construction does not have. A source that never left the
match radius explains nothing and is not counted.

---

## 5. The chance-match systematic

At the published 5″ radius, the probability that an unrelated infrared source
falls inside the aperture is `P = 1 − exp(−n·pi·r²)`, with `pi·(5/3600)² =
6.06 × 10⁻⁶ deg²`:

| infrared reference | density | `P` at 5″ |
|---|---|---|
| high galactic latitude | ~1 500 deg⁻² | ~0.9 % |
| AllWISE all-sky mean | ~18 100 deg⁻² | ~10 % |
| CatWISE2020 all-sky mean | ~45 800 deg⁻² | ~24 % |
| Galactic plane | ≫ 10⁵ deg⁻² | → 1 |

So of order 10⁴–10⁵ of the 172 163 "infrared counterparts" could be coincidences
before any astrophysics is invoked. This is not a footnote; it is the dominant
systematic, and Watters et al. explicitly leave it "undetermined".

**It is measured, not modelled.** The same sightlines are displaced by a fixed
angle in a random direction and pushed through the *identical* crossmatch. An
offset-position null samples the real local source density, the real plate
coverage and the real Galactic structure — and assuming a spatially uniform
random background instead is precisely the error Watters et al. identified in
the Earth-shadow analysis, so this channel does not repeat it. The genuinely
associated fraction is `f_true = f_match − f_chance`, with binomial errors.

Note the sanity check this enables: the published match fraction is
`172 163 / 298 165 = 57.7 %`, which is well above any plausible all-sky chance
rate. Most of the infrared matches are therefore real associations — but *which*
ones is exactly what the per-object chance probability decides.

### Inherited ledger vetoes

Single-band anomalies, W4-only excesses (cirrus), negative W1−W2 (a blend),
> 1 IR source in the PSF, plate photometry within 0.7 mag of the plate limit
(emulsion-noise dominated — the Hambly & Blair regime), and saturated plate
photometry.

---

## 6. Prior art

* **Solano, Villarroel & Rodrigo 2022** built the sample and did not analyse it.
  Their own second unexploited result: "No point sources were detected by both
  POSS-I and POSS-II before vanishing, setting the rate of failed supernovae in
  the Milky Way during 70 years to less than one in one billion."
* **Driessen et al. 2023** (arXiv:2306.08059) found 52 FIRST-detected /
  RACS-mid-undetected stellar radio sources as a by-product of a proper-motion
  method — the radio analogue of this test, at 1/3000 the scale.
* **The failed-supernova literature** is the methodological gold standard for
  "vanished vs enshrouded", and shows how hard it is: N6946-BH1 is still
  contested 17 years and one JWST dataset later (Adams+2017 arXiv:1609.01283;
  Kochanek+2023 arXiv:2310.01514 vs Beasor+2024 arXiv:2309.16121, which shows
  the photometry is a blend of ≥ 3 sources). Neustadt et al. 2021
  (arXiv:2104.03318) put the failed-SN fraction at f = 0.16 (+0.23/−0.12).
* **Forés-Toribio & Kochanek 2026** (arXiv:2604.05019) supply the standing
  luminosity-ratio discriminant used in §3.

**Novelty position.** The archival crossmatch exists and is public; what does
not exist anywhere is a population decomposition of it, an SED energy budget on
it, a measured chance-match rate for it, or the obscuration-to-destruction ratio
between the two published samples. That is what this channel computes.

---

## 7. Pipeline and outputs

```
seti shroud --stage acquire      # both SVO catalogues (+ VizieR fallback)
seti shroud --stage photometry   # AllWISE W3/W4 + 2MASS + wide Gaia; offset null
seti shroud --stage analyze      # classify -> fit -> budget -> vet -> report
```

`.github/workflows/shroud.yml` runs the three stages as separate jobs with
artifact passthrough, checkpointing, and `reduce_only_run_id` re-reduction.
Acquisition is runner-only; the sandbox is 403-blocked on every archive host.

Outputs in `results/shroud/`:

| file | contents |
|---|---|
| `summary.json` | verdict, funnel counts, population, ratio, null stats |
| `REPORT.md` | the human-readable version |
| `population.csv` | the class census |
| `survivors.csv` | objects that beat every contamination kill-test |
| `null_stats.json` | measured chance-match rate |
| `acquire_verdict.json` | every URL tried, with status and row count |

**Verdicts are first-class.** `VO_ARCHIVE` (both catalogues),
`VO_ARCHIVE_PARTIAL` (one), `VIZIER_FALLBACK` (127 rows — three orders of
magnitude smaller, and population fractions from it are indicative only),
`LOCAL_CACHE`, `NO_DATA_REACHED` (nothing retrieved, nothing analysed, nothing
invented).

## 8. Honest limitations

* **`eta` bounds a range, not a point.** With one historical magnitude the
  progenitor temperature is unknown; only the all-temperature verdicts
  (`IR_TOO_FAINT`, `IR_EXCEEDS_MISSING`) are robust.
* **Photographic photometry is good to ~0.3 mag at best**, is non-linear near
  the plate limit, and carries no colour information for most sources. The
  systematic floor is set accordingly.
* **A single-temperature blackbody is a crude dust model.** It cannot
  distinguish a shell from a swarm, and it ignores emissivity slope.
* **`ENERGY_CONSERVING_OBSCURATION` is a consistency statement, not a detection
  of anything artificial.** Every ordinary dusty evolved star satisfies it too;
  that is why `DUSTY_AGB` and `YSO` are subtracted first and why the residual
  class is the only one that reaches the budget.
* Per CLAUDE.md, an empty survivor list is **not** a publishable result. The
  population breakdown and the obscuration-to-destruction ratio are the standing
  measurements regardless of whether anything survives.
