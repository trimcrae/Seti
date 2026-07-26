# RUST — structures that stopped being maintained

**Signature S9** (`docs/necrosignatures.md`): *unmaintained decay — photometric
irregularity secularly **increasing***.

---

## 1. The claim

A megaswarm survives only while it is actively flown. Lacki 2025, *"Ground to
Dust: Collisional Cascades and the Fate of Kardashev II Megaswarms"*
(arXiv:2504.21151, ApJ 985, 191) states the mechanism in the abstract, verbatim:

> "Although long-lived megaswarms are extremely powerful technosignatures, they
> are liable to be subject to collisional cascades **once guidance systems start
> failing**. The collisional time is roughly an orbital period divided by the
> covering fraction of the swarm. … once the collisional cascade begins, it can
> develop **extremely rapidly** for hypervelocity collisions. … Most megaswarms
> are thus likely to be short-lived on cosmic timescales without active upkeep."

Lacki proposes **no photometric observable and runs no search**. The paper ends
at "implications for megastructure searches"; this channel is the observable.

**The observable.** A swarm that has lost station-keeping does not simply get
fainter — it gets *messier*. Elements de-phase, collide, and multiply into
fragments, so the star's **aperiodic variability amplitude grows secularly**
across the survey decade. The statistic is therefore the **second moment of the
light curve as a function of calendar time**.

**The timescale is right for a decade-baseline survey.** Lacki's collisional
time is `t_coll ≈ P / f`. For a swarm at 1 AU (`P = 1 yr`) with covering
fraction `f = 0.1`, `t_coll ≈ 10 yr`; at 0.2 AU (`P ≈ 0.1 yr`) with `f = 0.01`,
`t_coll ≈ 10 yr` again. ZTF's 2018–2025 baseline samples exactly the regime
where a cascade would be visibly *underway* rather than finished — and OSSUARY,
the sibling channel, looks for the terminal dust of one that finished.

---

## 2. What this channel is and is not sensitive to

The honest counterweight is that **not every architecture decays when control is
lost**, and the sensitivity statement has to say so.

| Architecture | Fate without upkeep | RUST sensitive? |
|---|---|---|
| Many-element swarm, randomised orbits, `f ≳ 10⁻³` | collisional cascade on `P/f` (Lacki 2025) | **Yes** — this is the target |
| Swarm with `t_coll ≫ decade` (low `f`, wide orbits) | cascade eventually, not now | No — outside the window |
| Ring-supported stellar engine / dense-cloud Dyson bubble | **passively stable**, needs no active control (McInnes 2026, arXiv:2603.00203) | **No** — nothing decays, so nothing to see |
| Rigid monolithic Dyson *sphere* | dynamically and mechanically unstable (Wright 2020, arXiv:2006.16734, SerAJ 200, 1–18) | Marginal — the failure is catastrophic, not a decade-long ramp |
| Cascade already complete | warm debris dust, no residual structure | No — that is **OSSUARY** |

The two counterweight abstracts, fetched on the runner and quoted rather than
paraphrased:

> **McInnes 2026** — "ultra-large reflectors in static equilibrium levitating
> above a central star (so-called stellar engines) are *always unstable* if the
> reflector comprises a uniform disc. However, if the reflector has a
> non-uniform mass distribution, specifically a ring supporting a reflector, a
> stellar engine can in principle be **passively stable**. Moreover, while …
> Dyson bubbles are unstable, in principle they can become **passively
> self-stabilizing** if arranged about the star as a dense cloud."

> **Wright 2020** — "I explicate the ways in which the popular imagining of them
> as monolithic objects would make them **dynamically unstable** under gravity
> and radiation pressure, and **mechanically unstable to buckling**."

A null here therefore constrains *decaying many-element swarms in the
decade-cascade window*, and nothing else. It says nothing about passively stable
architectures, which is the whole point of stating it.

---

## 3. Novelty

**Verdict: the statistic is unoccupied.** Three separate things have to hold,
and they are different claims. Runner-fetched evidence for all of them is under
`results/rustlit/` (run 30203976309, **47/47 fetches succeeded**); one leg of the
check failed and is marked as failed below rather than quietly dropped.

1. **The mechanism has no published observable.** Lacki 2025 is a dynamics
   paper. Full text at `results/necrolit/txt_lacki_ground_to_dust.txt`; the
   phrase "once guidance systems start failing" is at line 14 of the abstract as
   fetched, and there is no photometric search anywhere in the paper.

2. **Nobody has catalogued stars becoming more variable over time.** A regex
   concept scan over **3,578 fetched abstracts** for "amplitude/scatter
   increasing with calendar time" returned **zero** matching studies. Every hit
   was a *different statistic*, and the four decoy classes are worth naming
   because they are what a careless search would have counted:

   | Apparent hit | What it actually was |
   |---|---|
   | Polaris amplitude growth (arXiv:0805.1165, 0804.2793, 1703.02421) | a **single named star**, not a catalogue |
   | T Tauri stars in WASP (arXiv:1611.03013) | amplitude increasing **with timescale** — a red-noise property, not calendar time |
   | YSOs in W51 (arXiv:2510.12212) | increasing with **evolutionary stage** — a population statement |
   | NLSy1 galaxies (arXiv:2602.09171) | increasing toward **shorter wavelength** — a colour statement |

   `scripts/rustlit_fetch.py` re-ran this independently on the runner with the
   decoy classes encoded as explicit regexes. Over **215 abstracts** returned by
   24 targeted arXiv queries — written to catch the target concept *and* each
   decoy class — the scan found **1 regex hit, which was decoy-tagged
   `with_timescale` (red noise), and 0 decoy-free hits**
   (`results/rustlit/concept_scan.json`). The null is auditable rather than
   asserted, and it reproduces the earlier sweep on an independent query set.

3. **The nearest existing machinery is a first-moment search.** Petz & Kochanek
   2025, *"Life in the Slow Lane"* (arXiv:2501.14058), ran **9,361,613** isolated
   ASAS-SN sources at 13 < g < 14.5, selecting brightness changes > 0.03 mag/yr
   over 10 yr, and found 782 slow variables (433 new). That is a **mean-flux
   slope** — the abstract is explicit that the selection is on "brightness
   changes larger than ~0.03 mag/year". The second moment is untouched.

**The leg that failed, stated as failed.** The plan included a citation-tree
test — *do any works citing Lacki 2025 or Petz & Kochanek 2025 run a
second-moment search?* — and it returned nothing usable. The OpenAlex lookup by
arXiv DOI (`10.48550/arXiv.<id>`) resolves to the **preprint stub**, which
reported `cited_by_count = 0` for all four target papers *including Wright 2020*,
a review that certainly has many citations. An empty citation tree fetched that
way is not evidence of anything, so **no novelty weight is placed on it**.
`scripts/rustlit_fetch.py` now also searches OpenAlex by title and keeps
whichever record has the larger citation count, recording both routes in
`oa_pick_<name>.json`; a re-dispatch will make this leg informative. Until then
the novelty claim rests on legs 1–3 above, which do not depend on it.

**Bonus seam.** Hephaistos II's `G_var > 2` cut explicitly *"rejects potential
Dyson swarms with very large absorbing elements since these in principle could
generate detectable variations in the photometry of the host star."* The
variability this channel selects on is precisely what the flagship Dyson search
throws away.

---

## 4. Why this is not the `dimming` channel again

This matters enough to be explicit, because the repository already ran a secular
ZTF search and exhausted it.

`seti.dimming.secular` bins a light curve into seasons, takes the **median
magnitude** of each, and fits a weighted line: a trend in **brightness**, the
**first moment**. That channel ran 250,862 ZTF stars over 116 fields and is
recorded in `STATUS.md` as *exhausted at the ZTF systematics floor* — its best
candidate died on the NEOWISE reddening test and 19 marginal faders were set
aside at 1.6–7.4% total fade.

RUST regresses a bias-corrected **season scatter** against time: the **second
moment**. The two have *different systematics*, and the difference is
structural, not rhetorical:

| Systematic | Effect on the first moment (`dimming`) | Effect on the second moment (RUST) |
|---|---|---|
| Zeropoint / reference-image drift | shifts every star's season median together — **the dominant false fade** | shifts every epoch in the season equally, so the **within-season scatter is unchanged** |
| Slow calibration ramp | manufactures a linear fade | none to first order |
| Cadence change (N per season) | negligible — a median is nearly unbiased at any N | **manufactures a trend** — this is RUST's dominant systematic, see §5 |
| `magerr` mis-calibration drifting with time | none | **manufactures a trend** — see §5.4 |
| Seeing/depth trend | small | enters through the noise floor |

The floor that stopped `dimming` is a *first-moment* floor, so it does not
automatically apply — and the exchange is not free: RUST inherits two systematics
`dimming` never had to think about. Section 5 is how they are handled, and it is
the majority of the channel's code.

---

## 5. The cadence-bias systematic — the thing that decides whether this works

**The trap.** Robust scale estimators are biased **low** at small N. For
Gaussian data `1.4826 × MAD` recovers only ~66% of σ at N = 3, ~90% at N = 8,
~95% at N = 20. The number of epochs per season is set by the survey's cadence,
and **survey cadence trends with calendar time**: ZTF's public survey moved from
a 3-day to a 2-day cadence in 2020, and ZTF-I/II/III have different field rosters
and different seasonal depths. A perfectly constant star observed with a rising
cadence therefore shows a **rising measured scatter**. A search that skips this
does not measure astrophysics; it measures ZTF's operations calendar.

Six layers, none optional. Layers 0–3 are in `src/seti/rust/scatter.py`,
layer 4 in `trend.py`, layer 5 in `run.py`.

### 5.0 Remove a *line* per season, not just the season mean

Before any of the machinery below, each season's residuals are taken about a
**fitted line**, not about the season median. This is what makes the statistic
genuinely *aperiodic*, and it closes a confounder that would otherwise pipe the
sibling channel's entire population into this one:

> A star whose brightness is fading at an **accelerating** rate drifts further
> within each successive season than the last. Season-median subtraction leaves
> that drift in, so it reads as a rising second moment produced entirely by a
> *first*-moment phenomenon.

Measured (`tests/test_rust.py::test_an_accelerating_secular_fade_does_not_flag`):
a 0.72 mag accelerating fade flags **46 of 60** realisations without the line
removal and **0 of 60** with it. A megaswarm cascade is short-timescale and
irregular; a fade is smooth. Subtracting the line separates them.

The null table is switched to its **line-detrended variant** to match, because a
line fit removes two degrees of freedom by an N-dependent amount — `b(N)` falls
from 0.900 to 0.811 at N = 8 but only 0.990 → 0.984 at N = 80. Scoring detrended
residuals against the un-detrended table would reintroduce exactly the
cadence-tracking offset the module exists to remove.

### 5.1 Exact null expectation per season, at that season's own N

Every season's statistic is compared against what the estimator *would* read if
the star were perfectly constant — computed with **that season's own epoch count
and that season's own per-epoch error vector**. Two ingredients:

* `mad_null_table()` — a Monte-Carlo table of the finite-N bias `b(N)` and the
  relative sampling scatter `u(N)` of `1.4826 × MAD` on Gaussian data. `b(N)` is
  **not monotonic** at small N (the sample median alternates between even and odd
  N), so no smooth analytic correction would do; above N = 30 the alternation has
  died and the tabulated tail is replaced by its own least-squares fit, because
  Monte-Carlo jitter indexed by N is jitter indexed by *calendar time*.
* `mixture_mad_sigma(errs)` — the asymptotic MAD of a **heteroscedastic**
  zero-mean Gaussian mixture, solving `(1/N) Σ erf(c / (e_j √2)) = ½`. Within a
  season, seeing, airmass and moon phase all vary, so the epochs do not share one
  error; the population MAD is **not** `0.6745 × rms(e)`, and using the naive
  quadrature mean biases the noise floor in a magnitude-dependent way.

Because the null carries the same N as the data, the N-dependence cancels **by
construction rather than by hope**.

### 5.2 Excess variance, subtracted not divided — including the second-order term

`v_exc = σ̂² − σ_null²` in mag², and the excess is **allowed to go negative**.
Clipping at zero would rectify noise into a spurious positive trend for faint
stars.

The second-order term matters and is easy to miss: the estimator's *square* is
not the square of its expectation,

```
E[σ̂²] = (E[σ̂])² + Var(σ̂) = (b(N)·σ)² · (1 + u(N)²)
```

Since `u(N)` falls as N grows and N tracks cadence, omitting the `(1 + u²)`
factor leaves a residual N-dependent — hence calendar-time-dependent — offset.
At N = 8, `u = 0.41` and the term is a **17%** offset in variance; at N = 70 it
is 2%. It is exactly the class of error this channel exists to avoid.
`tests/test_rust.py::test_the_correction_is_load_bearing_at_the_estimator_level`
measures the raw bias directly (a resolved ~7% step in `E[σ̂²]/σ²` between an
8-epoch and a 70-epoch season) and confirms the corrected statistic is consistent
with zero at both.

### 5.3 Distribution-free detection

The primary gate is an **exact one-sided permutation p-value** for a positive
Spearman correlation of season excess-variance against season index, computed by
enumerating all `n!` orderings for `n ≤ 8` — which is the normal case, since a
ZTF decade gives 7–8 seasons. It is primary precisely because *Lacki gives a
cascade timescale, not a light-curve shape*: assuming the second moment rises
linearly would be assuming a result. Monotonicity is the physical claim, and a
rank test tests exactly that and nothing more.

The weighted linear fit of excess variance against time is retained for the
**amplitude scale** and as a second, model-dependent significance, with its slope
error inflated by `√(χ²/dof)`. For an accelerating rise a straight line fits
badly and the inflation is large, so that number is deliberately conservative and
is *not* the detector.

### 5.4 Per-CCD ensemble common mode in the second moment

The direct analogue of what `dimming.run._ensemble_detrend_secular` does for the
first moment, and the layer that handles the systematic §5.1–5.2 cannot:
**ZTF's reported `magerr` is a model, and the model drifts.** Seeing, background,
the ZTF-I → ZTF-II transition and reference-image rebuilds all shift the
true-to-reported error ratio as a function of calendar time, and an uncorrected
drift in that ratio *is* a spurious secular trend in excess variance.

The field ensemble measures it directly. For every star on a readout channel,
`σ̂²/σ_null²` should be ≈ 1 in every season if the errors are right and the star
is constant. The **median** of that ratio over many stars is dominated by the
constant majority, so it estimates the season's error-scale factor `κ_s`
regardless of a few real variables. Two passes:

1. multiplicative: recompute `v_exc = σ̂² − κ_s · σ_null²`;
2. additive: subtract the residual per-season common mode, each star contributing
   `v_exc(season) − median_over_seasons(v_exc)` so its own level cancels.

`κ_min`, `κ_median`, `κ_max` and the additive amplitude are reported in every
field summary. If the corrections are large, the write-up says so.

### 5.5 Two independent cross-checks on survivors

Too slow for a sweep, mandatory for a claim (`run.confirm_survivor`):

* **Exact per-season Monte Carlo** — drops every asymptotic approximation in
  §5.1 and simulates the season's actual error vector. If it disagrees with the
  fast path, the fast path is wrong.
* **Equal-N subsampling** — truncates every season to the smallest season's
  epoch count, so the estimator bias is *identical* in every season and cannot
  possibly produce a trend. It throws data away; that is the point. It shares no
  machinery with §5.1, so agreement between them is real evidence.

Both numbers are reported for every survivor, pass or fail, alongside the
per-season epoch counts so a reviewer can audit the cadence history by hand.

**Measured performance** (`tests/test_rust.py`, 39 offline tests):

| Null | Realisations | Flagged |
|---|---|---|
| Constant star, 5 cadence histories (8→70, 70→8, ZTF-style jump, erratic, doubling) | 600 | **0** |
| Constant star, wider 4-pattern sweep | 2,000 | **0** |
| Constant star, rising per-epoch **errors** (10 → 46 mmag) | — | rejected |
| Field-wide `magerr` drift (κ: 1.0 → 4.0), 60 stars | 60 | ≥20 before detrend, **0** after |
| Accelerating secular fade, 0.72 mag total | 60 | 46 without §5.0, **0** with |

Completeness for an injected linear amplitude rise at realistic ZTF sampling
(80 epochs/season, 7 seasons, 20 mmag photometric error):

| Terminal amplitude | Per band | **Two-band AND** |
|---|---|---|
| 36 mmag | 58% | 34% |
| 60 mmag | 81% | 66% |
| 90 mmag | 81% | 65% |
| ≥120 mmag | ~78% | ~60% |

The two-band requirement roughly squares the per-band completeness — that is the
price of the ledger's first rule and it is paid deliberately. The **plateau near
80% per band is a real ceiling, not noise**: it is the χ²-inflated linear-slope
gate refusing to certify an accelerating rise that a straight line fits badly.
The rank test almost never fails (2 of 200 at 120 mmag); the linear gate fails
~38%. Sensitivity is therefore set by a deliberately conservative secondary
statistic, and could be raised by fitting a growth law — at the cost of assuming
one, which §5.3 declines to do.

---

## 6. Method

1. **Data.** ZTF DR g **and** r, 2018–2025, pulled in bulk per sky tile from the
   IRSA light-curve API and paired **positionally** (ZTF assigns different `oid`
   values per filter; tolerance 1.5″, inside the ~2″ PSF and outside the
   per-band astrometric jitter). Intra-survey season-to-season comparison only,
   so calibration changes cannot masquerade as astrophysics. ASAS-SN is wired for
   survivors as the cross-survey confirmation. **ATLAS is deliberately not
   wired**: its forced-photometry service needs a per-user account token this
   repository does not hold, so wiring it would only produce a code path that
   always fails on the runner.

2. **Statistic.** Per-season robust scatter (MAD), corrected for the per-epoch
   photometric error and for the number of epochs in that season, per §5.

3. **Trend.** Exact rank p-value plus a weighted linear fit, with
   leave-one-season-out robustness.

4. **Two-band coincidence is mandatory — and is the physics.** A source
   measurable in only one band is never scored at all. Debris crossing a star
   blocks light *geometrically*, so the induced magnitude excursion is the same
   in g and r and the amplitude-growth ratio is ≈ 1. Every mundane mechanism is
   chromatic in a known direction:

   | g/r amplitude-growth ratio | Verdict | Interpretation |
   |---|---|---|
   | < 0.80 | `chromatic_red` | blend with a red variable |
   | 0.80–1.20 | **`achromatic_gray`** | grey occulter — **the RUST signature** |
   | 1.25–1.70 | `reddening_law` | growing line-of-sight dust column (`A_g/A_r ≈ 1.42` for `R_V = 3.1`) |
   | > 1.70 | `chromatic_blue` | flare / accretion / spots — not geometric |

   This mirrors what `dimming/characterize.py` does for fades, in the second
   moment.

5. **Vetting gauntlet** (`vet.py`, every verdict a pure function):
   blending (Gaia neighbour census inside the PSF), saturation and faint-limit
   proximity (**both** ZTF walls manufacture exactly this trend), known variable
   classes via SIMBAD (YSOs, dippers, cataclysmics — and **AGN**, whose red-noise
   amplitude grows with the *timescale* sampled, making a lengthening window the
   single most seductive false positive here), Lomb-Scargle periodicity (a
   growing-amplitude pulsator is coherent; this channel is aperiodic), Gaia RUWE
   and `non_single_star`, and NEOWISE.

6. **NEOWISE logic is inverted relative to `dimming`.** There, a mid-IR
   brightening *killed* a candidate — absorbed starlight reappearing as thermal
   dust emission, an ordinary enshrouding event. Here, a cascade grinding a swarm
   to fragments **should** produce dust and **should** brighten W1/W2, so a
   brightening is **corroboration**. Its absence is informative but not fatal:
   W1/W2 (3.4/4.6 µm) only probe material hotter than ~600–850 K, so a cascade
   beyond ~1 AU can be real and mid-IR-silent. The verdict says which case we are
   in; it does not pretend a non-detection settles anything.

---

## 7. Layout

```
src/seti/rust/acquire.py   paired ZTF g+r pulls (runner-only), Gaia crowding census
src/seti/rust/scatter.py   the bias-corrected season-scatter statistic  [§5.1-5.2]
src/seti/rust/trend.py     regression + exact rank test + ensemble common mode [§5.3-5.4]
src/seti/rust/vet.py       the contamination gauntlet (pure functions)
src/seti/rust/run.py       stage orchestration -> results/rust/
config/rust.yaml           every threshold
tests/test_rust.py         offline suite; the cadence-bias tests are the point
.github/workflows/rust.yml sharded, checkpointed, commit-back
scripts/rustlit_fetch.py   runner-side novelty evidence + decoy-aware scan
```

CLI: `seti rust-sweep` (per field, shardable) and `seti rust-vet` (aggregate).

---

## 8. Honest limitations

* **The sensitivity is to `f` through the induced photometric amplitude, not to
  `f` directly.** A cascade only shows up if the de-phasing fragments produce a
  ≳15 mmag aperiodic amplitude by the end of the baseline. Small-`f` swarms are
  invisible here regardless of whether they are decaying.
* **Seven or eight seasons is a short lever arm.** The exact rank test is the
  right tool for it, but with `n = 7` the smallest attainable p-value is
  `1/5040 ≈ 2 × 10⁻⁴`, so a blind search over ≳10⁵ stars *will* produce
  rank-significant stars by chance. The two-band requirement is what makes the
  search viable: it is applied at scoring, not follow-up, and an independent
  per-band false-positive rate squares.
* **Red-noise astrophysics is not white noise.** The false-positive rates quoted
  in §5.5 are measured against *Gaussian* nulls. Real stellar and AGN variability
  is correlated, and correlated noise over a short season count can mimic
  monotone growth. This is why AGN are a named rejection class and why the
  ensemble common mode is computed per readout channel rather than globally — but
  it remains the least-controlled residual, and any survivor's write-up must
  state its ASAS-SN cross-survey result rather than lean on ZTF alone.
* **A null changes the question, it is not written up.** Per `CLAUDE.md`, a clean
  null here is a reason to move the search — to shorter-period (closer-in) swarms
  where `t_coll = P/f` is decades even at small `f`, or to a survey with a longer
  baseline — not to publish an occurrence limit.
