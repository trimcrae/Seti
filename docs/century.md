# CENTURY — the hundred-year baseline

*Signature S50 of `docs/necrofrontier.md`. Channel package `src/seti/century/`,
config `config/century.yaml`, workflow `.github/workflows/century.yml`,
results `results/century/`.*

CENTURY asks the three questions KNELL, RUST and `dimming/secular` already ask
of ZTF — *did the clock stop, is the scatter rising, is the star fading* — of a
light curve that is **a hundred and seven years long instead of six**. The data
are the Digital Access to a Sky Century @ Harvard archive, DR7 (released
29 December 2024): 252,458,490 sources, 23.6 billion photometric measurements,
from plates taken between 1885 and 1992.

---

## 1. The claim

A technosignature that *changes state* is only detectable if the baseline is
longer than the change. Every cessation and decay channel in this repository is
confined to the CCD era, and the CCD era is short:

| Channel | Signature | Baseline |
|---|---|---|
| KNELL | a catalogued period that stopped | ZTF, ~6 yr |
| RUST | season-to-season scatter rising | ZTF, ~6 yr |
| `dimming/secular` | a monotonic secular fade | ZTF, ~6 yr |
| **CENTURY** | **all three** | **DASCH, 107 yr** |

Six years is shorter than the timescale of every process the necrofrontier
programme cares about. A Kessler cascade in a dense swarm runs for decades. An
unmaintained structure grinds down over a century. A period that stopped in
1931 and never came back is invisible to every survey that began after it and
is written plainly on the Harvard plates. That is the whole argument for this
channel: **the decisive observation may already have been made, on glass,
before anyone could ask the question.**

## 2. Novelty

`docs/necrofrontier.md` records the gap as **unoccupied** (row `g4 century
fade`, 12 papers reviewed). The published DASCH literature is:

* the survey and pipeline papers (Grindlay, Tang, Los, Servillat et al.);
* **targeted** century light curves — KIC 8462852, J1407, FO Aqr, an R CrB
  star, LBV "sleep", a handful of K giants (Tang et al.);
* variability *statistics* of known classes.

There is **no blind DR7 search for cessation, for a secular fade against the
star's own plate-limit history, or for rising scatter.** The reason is stated
in `docs/knell.md` §Plates and it is a good one: the archive contains a
systematic that manufactures exactly the signal a naive search would find. That
systematic is §3 below. The contribution of this channel is not "look at
DASCH"; it is **a DASCH search with the Menzel gap modelled**.

## 3. THE GOVERNING METHODOLOGY

Three rules. They are not tunable and they are not optional.

### 3.1 The Menzel gap is a STEP, never a trend

The Harvard plate patrol stopped in 1953/54 under HCO director Donald Menzel
and resumed around 1969/70 with different telescopes, different emulsions and a
different observing programme. There are essentially **no plates between 1954
and 1970**, and the photometry on either side has a systematic offset of order
0.1–0.4 mag that varies with field, magnitude and series.

Lund, Pepper, Stassun & Hippke (2016, arXiv:1605.02760) and Hippke et al.
(2016) showed that the famous "century-long dimming" of KIC 8462852 in DASCH is
what that offset looks like **when it is fitted with a straight line**. The
same paper showed the "dimming" is shared by comparison stars on the same
plates. This is the single most expensive known error in the use of this
archive, and any statistic that compares early plates to late plates inherits
it.

So every "does this quantity change with calendar time" question in this
channel is fitted as

```
q(t) = q0 + s·(t − t_ref) + D·H(t − t_gap)
```

(`src/seti/century/step.py`), and the answer is the slope `s` **with the step
`D` free**. Two further numbers are always reported next to it:

* `s_naive` — the slope with `D` frozen at zero. The difference `s_naive − s`
  **is** the Hippke/Lund number: how much of the apparent trend was the gap.
  The run summary reports how many stars had a "trend" that was the gap
  (`n_fade_naive_trend_was_gap`).
* `s_pre` — the slope fitted inside the pre-gap segment alone. Sixty years of
  leverage live there. **A trend the pre-gap segment does not show on its own
  is not a trend**, and the fade verdict requires `s_pre` to be significant and
  of the same sign (`fade.pre_sigma_min`).

The step is also carried by the *cessation* statistic. If the transition from
"period detected" to "period absent" falls across the gap, the mean-flux test
cannot be evaluated on the star alone — the plates moved too. The screen defers
it, flags `mean_flux_gap_uncorrected`, and the assess stage re-evaluates the
shift against the **field's own step** measured from every other star of the
same brightness on the same plates.

#### 3.1.1 A false alarm must not move the transition across the gap

Found and fixed during this build, and worth stating because it is the Hippke
failure mode wearing a periodogram: the per-block detector fires at rate `fap`
by construction, and the *first block after the gap* is the densest post-gap
block in DASCH, hence the likeliest place for a false alarm. A split rule that
ends the pre-segment at the last detected block therefore moves the transition
past the gap on a single noise detection — and the gap's 0.3 mag step stops
being deferred and is charged to the star, which is scored
`faded_or_brightened`.

The split is now chosen by a false-alarm test rather than by the last
detection: scan upward over detected blocks and take the **earliest** split
whose later detections are consistent with noise (at most
`max(1, ⌈3·fap·n_post⌉)` of them, never two adjacent — two adjacent late
detections are a clock that came back, not a false alarm). Detections after the
accepted split are excluded from the post blocks and recorded as
`post_isolated_detection`, not believed. `s = s_last` always satisfies the test,
so the scan always terminates; taking the earliest acceptable split is what
makes an isolated post-gap false alarm cost nothing.

Relatedly, `mean_shift_across_gap` is now a statement about **where the
photometry sits in time** (median year of the plates that made the pre-mean
against the earliest year that made the post-mean), not about a block index,
and the pre-transition mean is restricted to one side of the gap when the
pre-blocks straddle it. Otherwise the plates' own offset is averaged into the
star's "mean magnitude before", which is the very quantity the test then
compares across the gap.

### 3.2 Every non-detection is efficiency-normalised, and the injection is CENSORED

KNELL's rule (`docs/knell.md` §3): a non-detection of a period is evidence of
cessation only when normalised by the **injection-measured detection
efficiency of that block, in that block's own sampling and noise**. On plates
two more things vary block to block and both delete detections with no change
in the star:

1. **Per-plate limiting magnitude.** A variable whose faint phase sinks below a
   plate's limit is simply not measured on that plate. Plate depth is a
   property of the *series*, and series are a property of *calendar time* — so
   plate depth is a confounder that looks exactly like a secular change. The
   injection is therefore **censored**: the injected light curve is evaluated
   on every plate of the block, detections *and* non-detections alike, and a
   point survives only if its injected magnitude is brighter than **that
   plate's own limit**. A plate that did not detect the star *can* detect it
   when the injected phase is bright, and that gain is modelled too.
2. **Exposure smearing.** A patrol exposure of tens of minutes attenuates a
   sinusoid of frequency `f` by `|sinc(f·t_exp)|`. The injected amplitude is
   the intrinsic amplitude — the pre-block measurement deconvolved by the
   pre-block mean smear — times **each plate's own** smear factor. Without
   this, every short-period variable "ceases" whenever the exposure time
   changes.

Because a fixed-window periodogram is O(N) per frequency, **every injected
trial gets its own permutation threshold on its own censored point set**, so
the censoring changes the null exactly the way it changes the data. Both the
censored and the uncensored efficiency are reported; when they differ by more
than 0.2 the block is flagged `plate_limit_limited`, which is the honest
statement that the plates, not the star, set the answer.

### 3.3 Every statistic is re-done at a deeper margin and inside one series

A result that depends on shallow plates, or on a change of plate series, is a
property of the archive. So the fade and the scatter slope are each refitted

* at a **deeper plate-limit margin** (`margin + deep_extra`), keeping only
  plates at least that much deeper than the star (for catalogued variables the
  margin is raised to `amplitude + 0.5`, so the faint phase is never censored
  on the plates that are kept), and
* inside the **dominant plate series** alone — emulsion changes are series
  changes,

and the cessation is re-run inside a series common to the pre and post blocks
(`same_series_check`). A candidate that does not survive all three is not a
candidate.

## 4. The three statistics

Per star, from one DASCH light curve:

**(1) Cessation** (`cease.py`, KNELL on plates). Calendar blocks of 2 years.
Reference frequency refined on the pre-gap union over the catalogue frequency
*and its first harmonic* (an eclipsing binary catalogued at its orbital period
peaks at `2/P`). Each block is "detected" when its maximum GLS power in a
narrow window around the reference exceeds the `1 − fap` quantile of the same
statistic on permutations of that block's own magnitudes. Split by §3.1.1;
efficiency by §3.2 in every non-detected block; the persistence p-value from
`seti.knell.efficiency` combines the informative post blocks. A cessation must
pass fourteen checks simultaneously, including: the post-block amplitude
consistent with zero (on the **debiased squared** amplitude — a fitted sine
amplitude is Rayleigh-distributed, so averaging amplitudes over many blocks is
biased positive by ~1.25σ per block and its "significance" grows as √n), a
significant amplitude drop, PDM periodic before and not after, excess variance
dropped, stable blend fraction, overlapping series, and a **blind** periodogram
in the post blocks finding nothing (a mode switch is not a cessation).

**(2) Rising season scatter** (`scatter.py`, RUST on plates). The second moment
is inherited whole from `seti.rust.scatter` (bias-corrected excess variance per
season at that season's own epoch count and error vector) and
`seti.rust.trend` (exact rank test, χ²-inflated slope, leave-one-out). What
changes: the trend carries the Menzel step, the rank test is run **inside the
pre-gap segment** where the leverage is, and periodic stars are residualised
at the reference frequency (two harmonics per season) before the scatter is
taken.

**(3) Secular fade** (`fade.py`). The slope of the annual median magnitude
against calendar year, step free, with the three refits of §3.3, and the field
common mode (every star on the same plates shares the same emulsion, depth and
calibration history) removed at assess time per field and per magnitude bin.

## 5. Kills

The gauntlet (`vet.py`) is ordered cheapest-and-most-lethal first. Each kill is
named after the systematic it removes:

| Kill | What it removes |
|---|---|
| `high_pm` | a star that moved arcseconds across the century is measured against different sky, different blends, possibly a different aperture |
| `long_period_giant` | Miras / SR / L variables and red giants have huge, slow, irregular amplitude and mean changes (Tang et al.'s DASCH K giants) — excluded by class *and* by colour |
| `too_bright` | photographic saturation compresses amplitude non-linearly, and series-dependently |
| `vanished_not_ceased` | a star not recovered on plates deep enough to show it is a **vanishing-source** claim — the contested VASCO-adjacent class — which this channel does not make |
| `blend_transition` | the blend fraction jumps at the transition |
| `series_disjoint` | pre and post blocks share no plate series |
| `same_series_fails` | the cessation does not hold inside a common series |
| `mean_flux_changed` | after the field-ensemble step correction, the mean moved across the transition (a fade is a fade, not a cessation) |
| `single_plate_evidence` | any statistic resting on fewer plates than required |

`menzel_gap_transition` is a **note**, not a kill: a transition that falls in
the 1954–1970 hole is a lower-confidence class, because the archive cannot say
when inside sixteen years it happened.

## 6. Data and the API

DASCH DR7 is served as JSON over REST at
`https://api.starglass.cfa.harvard.edu/public/dasch/dr7/`, wrapped by the
`daschlab` Python client. Three endpoints are used:

| Endpoint | Gives |
|---|---|
| `querycat` | reference-catalogue (`apass` / `atlas`) sources around a position, with the DASCH identifiers `gsc_bin_index` and `ref_number` that a light curve needs |
| `queryexps` | every plate exposure covering a position — a field's plate density and its pre/post-gap split |
| `lightcurve` | one source's photometry: per-plate calibrated magnitude, local RMS, **per-plate limiting magnitude at the source position**, plate series, exposure time, AFLAGS/BFLAGS |

`results/necrofrontier/` recorded all four DASCH endpoints as `REACHED` from a
runner, but **reachability is not a schema**: nothing in this repository had
ever posted to them. So `api.py` treats every payload as a hypothesis — an
ordered list of key spellings, the first HTTP 200 wins, a 400/422 body (a
FastAPI validation error names the fields it wanted) advances to the next
variant, and the variant that worked is written into the acquisition log. The
`probe` stage exists to settle all of it on a runner in one cheap job before a
sweep is committed to: documentation pages as text, the OpenAPI schema if one
is served, the `daschlab` source (which is where the AFLAGS/BFLAGS bit meanings
live — `flags.py` parses them out and falls back to `config/century.yaml`,
reporting "no flag cuts applied" as a **degradation** if neither is available),
the plate density of every configured field, and one live
`queryexps → querycat → lightcurve` chain on four bright catalogued variables.

## 7. Targets

Two populations per field, in this order:

1. **Catalogued periodic variables** (AAVSO VSX via VizieR `B/vsx/vsx`, GCVS as
   fallback) with `mag_max ≤ 13`, `0.2 d ≤ P ≤ 100 d`, catalogued amplitude
   ≥ 0.3 mag, and a periodic (non-LPV) type. These carry the cessation test:
   it needs a period to look for.
2. **Bright reference-catalogue stars** (`8 ≤ stdmag ≤ 13`) from `querycat`
   tiles across the field. These carry the fade and rising-scatter tests, which
   need no catalogue prior.

`B < 13` is not a brightness preference, it is what keeps **every plate series
usable**: the plates reach B ≈ 14–17 depending on series, so a star at 13 is
above the limit of even the shallow series and its non-detections mean
something.

## 8. Layout

```
src/seti/century/
  api.py         DASCH DR7 REST client: payload variants, schema-tolerant
                 frame coercion, every exchange logged with its body head
  lightcurve.py  the DASCH answer -> CenturyLC (detections, non-detections,
                 per-plate limits, exposure times, series, blend, flags)
  flags.py       AFLAGS/BFLAGS bit definitions from the daschlab source
  step.py        the Menzel-gap step model (§3.1)
  cease.py       censored-injection cessation (§3.2, §4.1)
  fade.py        secular fade with the step and the three refits
  scatter.py     rising season scatter, RUST's moments with the step
  targets.py     VSX/GCVS variables + bright refcat stars, plate density
  vet.py         the gauntlet (§5)
  run.py         stages probe / targets / acquire / screen / assess
tests/test_century.py     offline, synthetic plates with a real Menzel gap
config/century.yaml       every threshold
.github/workflows/century.yml
results/century/          probe/, targets.csv, shards/, summary.json
```

## 9. Running it

```
# learn the API on a runner first — one cheap job
gh workflow run century.yml --ref <branch> -f stages=probe

# then the sweep
gh workflow run century.yml --ref <branch> -f stages=full -f n_shards=6
```

Locally (offline, synthetic only):

```
python -m seti.cli century --stage screen --shard 0/1
```

The acquire stage checkpoints per star (`acquire.jsonl` plus a rewritten
`lightcurves.npz` every `checkpoint_every` stars) and **resumes from both**, so
a killed shard loses minutes rather than a run.

## 10. Verdicts

`results/century/summary.json` carries `verdict`, `verdict_code`, `degraded`,
the funnel and the coverage. The codes are ordered by how much the run was able
to say:

| Code | Meaning |
|---|---|
| `NO_TARGETS` | target selection returned nothing |
| `NO_DATA_REACHED` | the DASCH service did not answer — **not a statement about the sky** |
| `NO_LIGHTCURVES_RETURNED` | the service answered and returned no photometry |
| `NO_TESTABLE_LIGHT_CURVES` | photometry returned, none long or dense enough to test |
| `NO_CANDIDATES` | stars tested, nothing flagged — **a count, not an occurrence limit** |
| `CANDIDATES_ALL_TRACED` | candidates found, every one killed by a named systematic |
| `SURVIVORS_FOR_FOLLOWUP` | something survived the gauntlet |

Any degradation (flag bits not applied, field ensemble too thin to apply, Gaia
context unreachable, a shard truncated by its time budget) is appended to the
verdict string as `DEGRADED (...)` and is never silently absorbed. The workflow
asserts that a verdict with zero usable light curves must read as one of the
no-data codes — a run that reached nothing must never print as a science null.

## 11. Honest limitations

* **The archive's floor.** DASCH photometry is good to ~0.1–0.15 mag per plate
  at best. Amplitudes below ~0.3 mag are not reachable; this channel cannot see
  a small effect, only a large one.
* **Sixteen blind years.** A transition inside 1954–1970 can be located no more
  precisely than "somewhere in the gap".
* **The catalogue prior.** The cessation test needs a catalogued period, so it
  sees only stars someone already called variable. A period that started *and*
  stopped before the variable-star catalogues is invisible to this test — the
  fade and scatter tests, which need no prior, are the coverage for that.
* **B < 13.** The deeper the star, the fewer series can see it and the more
  the plate-limit confounder dominates. The bright cut is a decision to trade
  sample size for interpretability.
* **A null here changes the question, it is not a result.** Per `CLAUDE.md`,
  this channel does not produce an occurrence-limit paper.
