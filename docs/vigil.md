# VIGIL — waste heat with a duty cycle

*Necrosignature **S4**: duty-cycled waste heat, and the cessation of that duty cycle.*

## 1. The claim

A computing megastructure's waste heat tracks its computational load. The
covering fraction does not change on human timescales — the absorber is where it
is — but the **re-radiated** thermal output does, because the load does. The
observable is therefore a star that is

> **variable in the mid-infrared while constant in the optical.**

The intercepted starlight is steady; the re-emission is not. And the
extinction-relevant reading of the same channel is that variability **ceasing** —
a machine that stops computing goes thermally quiet while its structure remains.

This is the photometric negative of every executed waste-heat search, all of
which require a *static positive* infrared excess.

## 2. Novelty — verdict `UNOCCUPIED`, established on the runner

**No published search selects on mid-infrared variability with optical
constancy as a technosignature.** The `novelty` job of run **30215516935**
(`scripts/vigillit_fetch.py`, `results/vigillit/`) established this on a machine
with egress: **39/39 fetches succeeded, 303 abstracts scanned across 17 targeted
arXiv queries, and 0 genuine prior-art hits** — no abstract matched the
conjunction *(mid-IR OR WISE OR NEOWISE) AND (variability) AND (optically
constant) AND (technosignature OR Dyson OR SETI OR megastructure)* free of the
four decoy classes. The two abstracts scoring 3–4 concept groups were both
decoy-tagged `agn_blazar`. The decoys are excluded deliberately and named:
YSO/protostar mid-IR variability (ordinary accretion physics), extreme debris
disks (the physical **confounder**, not a technosignature claim), AGN/blazar
mid-IR variability, and optical-transit megastructure searches (Boyajian's star
and successors — a different observable). The verdict vocabulary distinguishes
`UNOCCUPIED` from `UNDETERMINED_FETCH_FAILED`, because "we could not look" and
"we found nothing" are different statements.

**Every decisive citation was verified by title, not by ID** — earlier agents in
this repository recorded hallucinated arXiv IDs that resolved to unrelated
physics. All four resolved correctly:

| ID | Fetched title | Verified |
|---|---|---|
| 2511.22071 | *A Catalogue of Mid-infrared Variable Sources from unTimely* | yes |
| 2103.00568 | *A new sample of warm extreme debris disks from the ALLWISE catalog* | yes |
| 2006.16734 | *Dyson Spheres* | yes |
| 2403.18941 | *A Data-Driven Search For Mid-Infrared Excesses Among Five Million Main-Sequence FGK Stars* | yes |

The unTimely variable catalogue's own abstract confirms the scale the channel
was designed around: it identifies **8,256,042 variable sources in W1 and
7,147,661 in W2** from unTimely coadded photometry, and states plainly that
"the WISE and NEOWISE missions have provided the only mid-infrared all-sky
time-domain data" and that "a comprehensive and systematic catalog of mid-infrared
variable sources has remained unavailable" until it. Its listed science
applications are stellar evolution, accretion and dust-enshrouded environments —
no technosignature use.

**The Hephaistos quote is now verbatim from fetched full text, not memory.**
Project Hephaistos II (arXiv:2405.02927) says of its variability check:

> "It is important to note that this check rejects potential Dyson swarms with
> very large absorbing elements since these in principle could generate
> detectable variations in the photometry of the host star."

and, in its modelling assumptions:

> "We also assume that Dyson spheres are built up slowly and uniformly
> everywhere, with equal covering factor (γ) in every direction, **with no pieces
> large enough to cause stellar variability**."

Hephaistos I (arXiv:2201.11123) is more equivocal — "while variability could
provide an interesting auxiliary diagnostic, we cannot easily dismiss DS
candidates based on whether they display variability" — so the *series* is aware
of the issue and still ends up excluding the population. Either way the point
stands and is now sourced: the nearest prior art discards variable stars, and in
any case its cut is on **optical** variability from occultation, not on
**mid-infrared** variability from modulated re-emission.

Three independent structural facts support the same conclusion:

1. **Variability is what the flagship Dyson search throws away.** Hephaistos II
   imposes a `G_var > 2` cut that explicitly *rejects* variable stars, on the
   stated grounds that a swarm with very large absorbing elements "could generate
   detectable variations in the photometry of the host star." That is a cut
   against **optical** variability caused by *occultation*. VIGIL selects on
   **mid-infrared** variability caused by *modulated re-emission*. The two are
   different observables of different physics, and the second one is discarded by
   construction in the largest existing search.
2. **Every catalogue search treats the SED as static.** The programme's earlier
   literature sweeps (`results/seamlit/`, `results/hephlit/`, `results/dysonlit/`)
   found no blind time-domain excess search at any wavelength: seam 3 of the
   seven structural seams is "single-epoch SEDs". Contardo & Hogg's 4.9 M stars
   and everything built on CatWISE are single-epoch by construction.
3. **The one adjacent mid-IR time-domain literature is astrophysical, not
   SETI.** Mid-IR variability *has* been studied — for YSOs, for AGN, and for
   extreme debris disks — but never with a technosignature selection, and never
   with the low-excess requirement that is this channel's entire discriminator.

The honest boundary: the *data* is not novel (NEOWISE has been mined hard), the
*statistic* is not exotic, and mid-IR variability catalogues exist. What is
unoccupied is the **selection**: mid-IR variability at a fractional excess too
small to be a debris disk, with the optical flat.

**One negative finding worth recording.** The unTimely variable catalogue paper
is real and verified, but 35,230 characters of its abstract and full text contain
**no data URL, no Zenodo deposit, no VizieR catalogue identifier and no Astro
Data Lab table name** — `results/vigillit/data_access_routes.json` is empty on
every access field. That is consistent with what the TAP probes found, and it
means the catalogue has no discoverable machine-readable release yet. The parent
unTimely Catalog does: `https://catalog.unwise.me/` and
`https://github.com/fkiwy/unTimely`. So the pre-selector is currently
unavailable *as a table*, and the channel runs on the NEOWISE field-sweep
architecture — which was always the fallback, and which the probe selects
automatically.

## 3. The confounder is the whole problem

**Extreme debris disks (EDDs) have exactly this phenomenology.** They vary
strongly in the mid-infrared, and in monitored cases their optical light curves
are flat — Moor et al. 2021 (arXiv:2103.00568), on warm extreme debris disks
selected from AllWISE, report that all monitored stars were stable with flat
light curves. This is a mature, actively studied class with a substantial
population, and **it will dominate any naive selection.**

So optical constancy buys essentially nothing against the dominant confounder.
It kills YSOs, dippers and AGN, and that is all it does. A channel that stops at
"mid-IR variable, optically constant" has produced an EDD catalogue and called it
a search.

### 3.1 The discriminator, and the arithmetic behind it

An EDD is "extreme" precisely because its fractional infrared excess is large:
`f ≡ L_IR/L_* ~ 10⁻²`. A duty-cycled radiator on a modest covering fraction would
vary **without** a large excess. So the primary axis is **mid-IR variability at
LOW fractional excess** — exactly as the brief specifies.

The physics that makes this measurable is one conversion factor. For a blackbody
radiator at `T_d` around a star at `T_*`, the ratio of in-band excess flux to
in-band photospheric flux is

```
F_exc(b)/F_phot(b) = τ · R(b, T_d, T_*),    R = [B_ν(T_d)/T_d⁴] / [B_ν(T_*)/T_*⁴]
```

`R` is large because the dust radiates near its Wien peak while the photosphere
is on its Rayleigh–Jeans tail. Computed at `T_* = 5000 K`
(`seti.vigil.excess.band_ratio_factor`):

| `T_d` (K) | `R`(W1) | `R`(W2) | W2 band excess at τ=1.5×10⁻³ | at τ=1×10⁻² |
|---|---|---|---|---|
| 300 | 0.06 | 2.0 | 0.3% | 2.0% |
| 400 | 0.73 | 8.6 | 1.3% | 8.6% |
| 500 | 2.6 | 16.8 | 2.5% | 16.8% |
| 600 | 5.1 | 23.0 | 3.5% | 23.0% |
| 850 | 10.5 | 27.0 | 4.1% | 27.0% |
| 1200 | 11.8 | 20.9 | 3.1% | 20.9% |

Two consequences, both load-bearing:

* **A low fractional excess is nevertheless detectable as variability.** τ = 1.5×10⁻³
  at 600 K is a 3.5% W2 band excess; switching it off is a ~37 mmag event, which
  NEOWISE visit means (10–20 exposures per visit) resolve comfortably. The channel
  is *not* asking for impossible photometry.
* **An EDD is a factor of ~5 brighter in band excess than the signal**, at the
  same dust temperature. That separation — ~20% versus ~4% — is resolvable against
  a realistic 2–3% photosphere-prediction systematic. This is what makes the cut
  work, and it is why the cut is applied to the **2σ upper limit** on the excess
  rather than to a point estimate: "low" has to be a bound.

### 3.2 The sharpened form: the modulation index

The `τ` cut inherits the full temperature sensitivity of `R`, which swings by a
factor >13 across the plausible range at fixed `τ`. That is the single largest
uncertainty in the channel, and it is removable.

Write the maximum peak-to-peak band amplitude an excess can produce. In the
extreme case — the radiator switches fully off half the time — the band flux
alternates between `F_phot` and `F_phot(1+2f)` where `f` is the measured **band**
excess, so `A_max = 2f/(1+f)`. Define

```
m ≡ A_obs / A_max  =  A_obs (1 + f) / (2 f)
```

**`R` cancels.** `m` is free of the dust temperature, the stellar temperature,
the distance and the luminosity: it is a ratio of two directly measured
photometric quantities, and it says *what fraction of the inferred excess is
actually switching*.

* An EDD's variability is a **perturbation on a large steady excess** — the
  collisional dust column drains and refills by tens of percent, not by 100%. So
  `m ≪ 1`.
* A load-following radiator has no steady floor to speak of: the excess **is**
  the modulation. So `m → 1`.

`m` also carries a falsifier. `m > 1` is physically impossible for circumstellar
re-emission, so a star whose amplitude exceeds what its excess allows is telling
you the variability is *not* the excess — a blend, a bad epoch, or an
underestimated photosphere. That is a rejection rule, not a candidate.

Both cuts are implemented and both are reported (`config/vigil.yaml`
→ `detect.tau_max` and `detect.modulation_min`). The `τ` cut selects; the `m`
cut rejects, and it is decisive exactly where it needs to be: for a large,
well-measured excess — the confounder population — `m` is tightly constrained,
while for a marginal excess it is honestly unconstrained and abstains rather than
guessing.

### 3.3 Two secondary shape discriminators

Both are built and both are tested (`src/seti/vigil/discriminate.py`).

* **Duty-cycle morphology.** A collisional cascade is a smooth secular decay (the
  dust column drains) punctuated by stochastic collisional brightening. A compute
  load has no reason to do that: it can be square, repeating, or two-state. The
  statistics are a weighted linear-trend `R²` and Kendall τ (scoring *against* a
  monotone decay), a likelihood two-state split with a BIC comparison against one
  state, a "squareness" fraction (points within 2σ of a state mean rather than in
  between), a transition rate, a Lomb–Scargle peak, and a burst skew. A
  decay-dominated light curve is rejected **even at low excess** — that gate is
  independently tested.
* **Colour-temperature stability during variation.** A cascade changes the amount
  *and* the temperature of the dust (fresh small grains are hot, then spread and
  cool). A load-following radiator changes the amount at roughly fixed
  temperature. The test is on the *varying component's* W2/W1 colour, epoch by
  epoch, with the photosphere subtracted — the total colour would be diluted by
  the star. Epochs where the excess is not detected at 3σ in both bands are
  dropped, because a ratio of two noise values is noise and would fabricate a
  drift.

## 4. The instrumental bound — stated, not assumed away

**NEOWISE, CatWISE2020 and the deep unWISE coadds are W1/W2 only.** Wien peaks:

```
W1  3.4 µm  →  852 K
W2  4.6 µm  →  630 K
```

So **this channel probes hot material and is structurally blind to the 100–300 K
regime where most Dyson models sit.** There is no way around it from within the
data: W3/W4 exist only for the single 2010 cryogenic epoch, and **W3/W4 depth has
been frozen since that mission ended and cannot improve until a new mid-IR
all-sky survey flies.** This is the same instrumental ceiling that shapes the
whole programme (`docs/necrosignatures.md` §3A(i)): deeper infrared data is
anti-correlated with colder sensitivity.

Consequences to state plainly:

* A megastructure radiating at 150 K produces **no** W1/W2 variability signature
  regardless of how hard its duty cycle swings. VIGIL cannot see it.
* The material VIGIL *can* see — 400–1200 K — is close-in: for a solar-luminosity
  host, equilibrium temperature 600 K corresponds to ~0.1 AU. This channel is
  therefore a search for **hot, close-in** computing hardware, which is where a
  thermodynamically aggressive design would actually put it (higher radiator
  temperature means a smaller radiator for the same waste power), but that is an
  argument, not a guarantee.
* A cessation reading is bounded the same way: only a *hot* radiator switching off
  is visible.

W3/W4 are pulled in the vetting stage, from AllWISE, **only to reject** — never
to detect. The ledger is explicit that AllWISE W4 is unreliable for faint stars
and that a W4-only signal is cirrus.

## 5. Method

```
probe    → is the unTimely mid-IR variable catalogue reachable, and what is it called?
           does NEOWISE per-epoch photometry come back?          [results/vigil/probe.json]
sweep    → per field:  Gaia DR3 sample (parallax_over_error > 10, G < 15)
           → ONE field-wide PM-propagated NEOWISE query, rows assigned to stars
             by KD-tree (per-star cones only for what it misses)
           → visit binning + empirical per-star noise calibration
           → field-wide ensemble common mode removed
           → normalised excess variance and its Vaughan et al. 2003 uncertainty
           → photosphere from 2MASS JHKs + T_eff → band excess → τ
           → modulation index, morphology, colour temperature
           → the cut                              [results/vigil/<field>/field_summary.json]
vet      → optical constancy (ZTF g+r), SIMBAD, AllWISE W3/W4, the gauntlet
                                                            [results/vigil/summary.json]
```

### 5.1 The estimator, and the two biases it corrects

**The visit structure is a free noise calibrator.** NEOWISE observes a field in
~1-day visits roughly twice a year, with order 10–20 exposures inside each visit.
Nothing circumstellar varies inside a day, so the scatter *within* a visit
measures the true per-exposure noise, empirically, per star — while the scatter
*between* visit means measures variability on the ~6-month timescale the channel
cares about. NEOWISE quoted `w?sigmpro` values are known to be optimistic for
bright sources, and an optimistic error is precisely how a variability search
manufactures candidates. So the per-epoch error is the quoted one rescaled by a
per-star, per-band factor fitted from the within-visit scatter, and the fitted
factor is reported rather than hidden.

The one signal this suppresses is variability faster than a day. NEOWISE cannot
separate sub-day variability from noise without an external noise model, and this
channel does not claim to.

**Cadence bias.** Exposures per visit and visits per star both vary strongly with
ecliptic latitude, because the NEOWISE scan pattern piles up at the poles. Any
scatter-to-noise ratio is biased at small N, and if N trends with position an
uncorrected search maps the scan pattern rather than astrophysics. Three
corrections:

1. The **primary statistic is the normalised excess variance**
   `nxs = (S² − ⟨σ²⟩)/⟨f⟩²`, which is unbiased in expectation at any N. The
   *biased* quantity is its square root `f_var`, which is reported for
   interpretability only; the significance quoted is that of `nxs`.
2. `nxs`'s uncertainty is the Vaughan et al. 2003 expression, which carries the
   exact N dependence the cadence imposes.
3. `equalize_visits` truncates every visit to a common exposure count, so a
   candidate can be re-measured with the finite-N bias made identical everywhere.

**The ensemble common mode** — per-visit zero-point wander shared by every star
in a field (moon, scan angle, thermal state) — is fitted and removed, and whether
it *could* be fitted is reported at the top level of the field summary. A field
too thin to measure a common mode produces uncorrected statistics, and that has
to be visible, not assumed away.

### 5.2 Proper motion

NEOWISE spans 2014–2024; Gaia positions are at epoch 2016.0. A 200 mas/yr star
drifts ~2.1″ across the mission, comparable to the cone radii these queries use.
**There was no PM-propagated `neowiser_p1bs_psd` fetcher anywhere in this
repository** — all three existing single-exposure callers query at the Gaia-epoch
position. `seti.vigil.acquire.fetch_neowise_epochs` propagates to the mission
mid-epoch (2019.0) and widens the cone by half the mission-long sweep, and
reports the correction it applied so a null on a high-PM star is distinguishable
from the bug this guards against. That bug cost a previous channel an entire run.

### 5.3 Data reached

* **Primary pre-selector:** the unTimely mid-IR variable catalogue
  (arXiv:2511.22071, >8M W1 and >7M W2 variables from WISE/NEOWISE unTimely
  coadds). Its access route is **discovered, not assumed** —
  `seti.vigil.acquire.probe_untimely` searches `TAP_SCHEMA` on VizieR, IRSA and
  NOIRLab Astro Data Lab (which already serves the parent unTimely Catalog), and
  `scripts/vigillit_fetch.py` mines the paper for Zenodo/VizieR/DOI routes.
  When a table *is* found, `preselect_from_untimely` restricts each field's Gaia
  sample to catalogued mid-IR variables; when it is not, the full sample is
  swept and `untimely_preselect.applied = False` is written into the field
  summary, because silently searching fewer stars and calling it a
  pre-selection would misreport the channel's own coverage.

  **Run 1 (30215516935) result, stated exactly:** VizieR returned an ADQL syntax
  error (its parser rejects `LOWER(col)` in a `WHERE` clause, which IRSA and
  NOIRLab both accept), while IRSA and NOIRLab each ran the query and returned
  **zero rows**. That is *one transport failure and two genuine absences*, which
  is **not** grounds for "the catalogue is unreachable" — so the verdict was
  recorded as `CATALOGUE_NOT_FOUND_ON_ANY_TAP_ROUTE` and the query was rewritten
  to spell out the case variants instead of calling `LOWER`. The verdict
  vocabulary now separates `ALL_TAP_ROUTES_FAILED`,
  `NOT_FOUND_BUT_SEARCH_INCOMPLETE` (some route errored — a partial search cannot
  support an absence claim) and `CATALOGUE_NOT_FOUND_ON_ANY_TAP_ROUTE`.
  The channel does not depend on the outcome: NEOWISE per-epoch photometry was
  reached in the same probe (`status: OK`, 32 rows against a `COUNT(*)` of 32,
  27 surviving frame-quality cleaning), so run 1 proceeded on the
  `neowise_field_sweep` architecture.

  **A second run-1 lesson, kept because it is a real sensitivity statement:** the
  probe's NEOWISE call was made at a bare coordinate and binned to **zero
  usable visits** — 32 sporadic single-exposure detections of a marginal source,
  never 3 or more inside any one visit, so the within-visit noise calibration had
  nothing to work with. The code was right to return `None` rather than
  manufacture a light curve, but a probe that measures an empty patch of sky is
  testing nothing. The probe now resolves a *real* star from Gaia in the field
  first and fetches NEOWISE at its PM-propagated position, so a zero there means
  the transport is broken, which is what a probe is for. It also reports the
  median exposures-per-visit, since that number is the channel's per-star
  sensitivity and it varies across the sky with the NEOWISE scan pattern.
* **Characteriser:** NEOWISE `neowiser_p1bs_psd` per-epoch W1/W2. This is
  required regardless: a variability *catalogue* contains detection flags, not the
  per-epoch photometry the modulation index, morphology and colour statistics need.
  If the catalogue is unreachable the channel degrades to a field-by-field NEOWISE
  sweep and the summary says so.
* **Optical constancy:** ZTF g+r. Gaia DR3 `phot_variable_flag` is carried as a
  flag. ATLAS is deliberately not wired — its forced-photometry service needs a
  per-user token this repository does not hold, so wiring it would produce a code
  path that can only fail on the runner.
* **Ancillary:** AllWISE W1–W4 + 2MASS JHKs, PM-propagated to 2010.5.

## 6. Contamination ledger

Inherited, not re-derived. Every rule below has a test that trips it
(`tests/test_vigil.py`).

| Confounder | Rule |
|---|---|
| **Extreme debris disk** | the discriminator of §3 — high excess and/or low modulation index and/or decay morphology and/or drifting colour temperature |
| **Cirrus** | a W4-only signal is cirrus; W4 can never contribute detection evidence. Low Galactic latitude is flagged |
| **Blends** | a negative W1−W2 is a blend, not a photosphere; a Gaia neighbour within the 6.1″ W1 beam and 2 mag |
| **AGN** | **colour alone cannot separate them** — a ~350 K shroud has W1−W2 = 3.2 and sits *inside* the Stern/Assef box. So an AGN colour hit is fatal only when the source is **not astrometrically stellar** (no significant parallax, no significant proper motion, or flagged extended). Astrometry, not colour, is the discriminant, and an in-box source *with* Gaia astrometry is flagged as interesting rather than rejected |
| **YSOs and dippers** | optically variable, so the optical-constancy cut does most of it; SIMBAD type and Galactic-plane position finish it |
| **AGB / Mira / LPV** | SIMBAD type; and phenomenologically, large amplitude + red + luminous |
| **Bright-source bias** | NEOWISE W1 saturates near 8 mag; a saturated profile fit produces spurious epoch-to-epoch scatter, which for a variability channel is a candidate factory, not a nuisance |
| **Optimistic errors** | the per-star within-visit noise calibration of §5.1 |
| **Cadence / scan pattern** | the three corrections of §5.1 |
| **High-PM nulls** | §5.2 — the PM sweep is reported so a null is attributable |
| **Untested checks** | every rule that could not run is named in `untested_checks`, so a verdict never quietly rests on checks that never happened |

Silverberg et al. 2018 puts the false-positive rate of AllWISE-selected infrared
excesses at ~92%. This channel is designed around that number: **the excess is
never the detection — the modulation is.**

## 7. Honest limits

* **Blind below ~200 K** (§4). This is instrumental and permanent until a new
  mid-IR all-sky survey flies.
* **Blind above ~1 day⁻¹.** The within-visit noise calibration by construction
  absorbs sub-day variability.
* **Aliasing.** Two visits per year means a duty cycle with a period near 6
  months or its harmonics is aliased. The median visit gap is reported per star so
  a candidate's accessible period range is explicit.
* **The excess precision floor.** With a 3% systematic on the predicted
  photosphere, `τ` can be bounded at roughly the 3×10⁻³ level — which is a factor
  of a few below the EDD regime, so the discrimination works, but it is not
  better than that. A star with a *marginal* excess has an unconstrained
  modulation index, and the code says so instead of guessing.
* **An unmeasured excess is not a low excess.** The funnel gates on measurability
  first: a star whose excess could not be bounded below the threshold is reported
  as `NOT_MEASURED`, never as a low-excess candidate. Conflating those two is how
  a channel turns its failures into detections.
* **This is a count, not a limit.** If nothing survives, the summary says
  `NO_CANDIDATES` over what was actually searched. No occurrence-rate limit is
  computed or claimed, and a transport failure emits `NO_DATA_REACHED`, which is a
  different word for a different thing.

## 8. Status

Built 2026-07-26. Offline suite green (`tests/test_vigil.py`, 30 tests),
including the decisive one: an injected extreme debris disk with the published
phenomenology — large fractional excess, decaying variability, drifting dust
temperature, flat optical light curve — is **genuinely mid-IR variable and
optically constant**, so a search that stopped at the headline phenomenology
would have flagged it, and it is rejected on the excess/modulation axis with the
reason recorded. Dispatched via `.github/workflows/vigil.yml`; run ID and
survivors are recorded in `STATUS.md`.
