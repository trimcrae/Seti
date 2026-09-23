# RING (S63) — rings around the dead

**The claim.** A technosignature at a host where biology is impossible has no
biological alternative. That is the whole novelty here: not a new statistic, a
new *host class*. Every warm-dust technosignature search on record has run over
main-sequence stars, where an infrared excess always has a natural reading
(debris, a disk, a companion, cirrus). This channel asks the same photometric
question of hosts that cannot have made the dust in place and cannot be
inhabited: white dwarfs, neutron stars, Y/T dwarfs, and free-floating planets.

**The temperature prior is what makes it a different search.** Osmanov (2016,
2018) computes Dyson rings around pulsars and white dwarfs at **300–700 K**.
That is a **W1/W2** excess. It sits *above* the frozen W3/W4 ceiling that bounds
every warm-dust channel in this repository (`docs/necrosignatures.md` §3A(i)),
and it is the opposite end of the temperature axis from CENOTAPH's cold-Dyson
regime. OSSUARY and the WD IR-excess channel own the **debris** reading of a
white-dwarf excess; RING is the **ring-temperature** reading of the same
photometry. The acquisition is deliberately reused (`src/seti/acquire`,
`sed`, `contamination`, `discriminate`, and OSSUARY's `excess`/`vet`); only the
prior on the fitted temperature is new.

**Prior art.** Theory only: Osmanov (2016, 2018); Ćirković & Bradbury (2006);
Opatrný (2017); Hsiao (2021); Kayali (2025) light curves. No executed search
over these hosts. No SETI pointing on record at either pulsar-planet system.

## The four legs

| leg | hosts | the measurement |
|---|---|---|
| `wd` | Gaia EDR3 white dwarfs (Gentile Fusillo+2021) × AllWISE (+2MASS) | photosphere predicted from J/H/Ks (Gaia G/BP/RP fallback); W1/W2 excess; blackbody fit for (T_ring, τ) |
| `pulsar` | ATNF pulsar catalogue × AllWISE / CatWISE2020 at proper-motion-propagated positions | a pulsar has **no infrared photosphere**, so *any* counterpart is an excess; the measurement is its W1−W2 colour temperature and its chance probability from offset-position controls |
| `bd` | Y and late-T dwarfs (Kirkpatrick+2021 20 pc census) | NEOWISE per-epoch W1/W2; the Wien peak of a Y dwarf sits in W2, so a W2 series that **switches** is a duty cycle |
| `ffp` | young free-floating planetary-mass objects (Faherty+2016) | is any object brighter than the cooling ceiling for 13 M_J at its group's age? |

Every leg is its own workflow job (`fail-fast: false`), so a lost leg costs that
leg. Acquisition is multi-route by design and the route that served each leg is
recorded: the white-dwarf leg tries (A) a Gaia-archive join, (B) a VizieR parent
catalogue plus an archive upload-join, (C) a CDS X-Match at hand-propagated
positions; the pulsar leg tries `psrqpy`, the psrcat tarball, then VizieR
`B/psr/psr`. A leg that reaches nothing returns `NO_DATA_REACHED` and the
channel verdict names it — a failed acquisition is never a statement about the
sky.

## Why proper motion is the whole game for the dead hosts

AllWISE's mean epoch is 2010.5 and Gaia's is 2016.0; CatWISE2020 is 2015.4.
White dwarfs are nearby and fast, and pulsar timing positions are milliarcsecond
but quoted at their own `POSEPOCH`. Every cross-match is made at the propagated
position, and the registration residual at that epoch is a gate, not a
diagnostic. For a fast host this is also the *discriminant*: a background source
does not move with the host, so a match that only works at the unpropagated
position is a background source.

For pulsars specifically, colour cannot do the extragalactic rejection, because
the ring band **is** the AGN wedge — a 500 K blackbody has W1−W2 = 2.15 and a
300 K one is redder still. The extragalactic discriminant here is therefore
astrometric and statistical: registration at the propagated epoch, CatWISE
co-movement, and an explicit chance-superposition prior measured from rings of
offset control positions (8 positions at each of 60″ and 90″ per pulsar). The
control rate is pooled with the global rate as an empirical-Bayes estimate, and
the rate that matters for the verdict is the rate of control matches that are
*also* in the ring colour band.

## The contamination ledger (inherited, not re-derived)

From `docs/channel-brief.md` §4, with the rule this channel adds:

* **A W4-only excess is an artefact.** A real warm SED lights the
  star-dominated bands first.
* **A negative W1−W2 is a blend**, not a photosphere.
* **A fitted T > 1800 K is a companion photosphere**, not a ring — hotter than
  grains survive. `companion_t_min_k` is exactly this line.
* **800–1800 K at sublimation-limited τ is the white-dwarf debris-disk locus** —
  the known, natural population. Named `debris_disk` and excluded from
  candidacy; it is not a competitor to the claim, it is the control sample.
* Below 250 K is `cold_dust`, outside the Osmanov band.

The shape classes are `ring_band`, `debris_disk`, `companion`, `cold_dust`,
`warm_ambiguous`, `unfit`, and only `ring_band` can be a candidate.

### Two admission routes, and why the second cannot manufacture a candidate

The excess selection requires a significant **colour** excess as well as
significant band excesses. That colour requirement is the guard against an
error in the SED anchor: mis-measure Ks and the whole predicted photosphere
scales, lifting W1 and W2 together with no colour signal.

A cool companion at 2500–3000 K does the same thing — its excess colour is
small even when its excess flux is several times the photosphere — so the
companion population was failing the colour test and was never flagged, never
fitted, and never *named*. An invisible contaminant is worse than a rejected
one, because it cannot be counted.

So there is a second admission route on **amplitude**: both bands at or above
the χ threshold *and* excess/photosphere ≥ 0.5 (0.44 mag) in both. An anchor
error of 0.02–0.1 mag cannot produce that. Rows admitted this way carry
`excess_route = "achromatic"`; rows admitted by colour carry `"colour"`, and the
counts of each are reported. **A ring candidate additionally requires
`excess_route == "colour"`**, which is free of cost: a 250–800 K ring has
W1−W2 > 1.3 mag by construction and therefore always enters by the colour
route. The achromatic route can only ever add named contaminants to the census.

## Sensitivity is reported per host, so a non-detection is a number

For every host the covering fraction a 300 / 500 / 700 K ring would need in
order to reach the AllWISE W2 depth is computed from the host luminosity
(spin-down power for a pulsar, the blackbody radius from log g and mass for a
white dwarf) and the distance. This is what makes the census interpretable:
"no ring" over a host where only f > 1 would have been detectable is not a
constraint on that host, and the summary separates the two.

## The named two-target vet

* **PSR B1257+12** (J1300+1240) — three planets (Wolszczan & Frail 1992);
  Spitzer MIPS limits on asteroidal dust (Bryden+2006, to verify).
* **PSR B1620−26** (J1623−2631) — circumbinary planet in M4 (Sigurdsson+2003);
  a globular-cluster sightline, and therefore crowded.

These are vetted by name and reported individually whatever the population
result, because they are the two systems where a ring is least *a priori*
absurd and where no infrared search is on record.

## Brown dwarfs: the duty-cycle test

The excess-variance test runs against the quoted per-epoch errors **and**
against the population's own median reduced χ², which absorbs a NEOWISE
error-underestimate that no single object can reveal. That population floor is
only an estimate when the tested sample is large enough that a genuine variable
cannot *be* the median, so below `pop_floor_min_n` tested objects the floor is
not applied and the threshold falls back to the per-object χ² requirement; the
summary records which applied. Anything that passes gets its amplitude and a
two-state duty cycle (the split that minimises within-state variance).

## Free-floating planets: the honest caveat

The flag is `log L_bol` above the analytic cooling ceiling for 13 M_J at the
group's age, with a margin. The systematics that produce this flag naturally are
named on every flagged row rather than argued away:
`age_misassignment | mass_underestimate | unresolved_binary`. The population is
tens of objects, so this leg is a vet, not a census.

## Verdicts

* `RING_CANDIDATES_PENDING_VET` — a white-dwarf or pulsar host survived every
  gate with a fitted temperature in the Osmanov band.
* `NO_RING_SURVIVOR; SECONDARY_FLAGS_PENDING_VET` — no ring, but a brown-dwarf
  duty cycle or an over-luminous planetary-mass object is outstanding.
* `NO_RING_SURVIVOR` — the ring question is answered in the negative for the
  hosts reached, at the sensitivity reported per host.
* `NO_DATA_REACHED` / `DEGRADED (<legs> not reached)` — an acquisition result,
  never a statement about the sky.

## Running it

```
python -m seti.ring.run --stage {probe,acquire,screen,assess,all} --leg {wd,pulsar,bd,ffp}
```

or `seti ring` with the same flags. On a runner:
`.github/workflows/ring.yml` (`workflow_dispatch`), which commits
`results/ring/summary.json`, `REPORT.md` and the per-leg screens back to the
branch it ran on.

Two things about *getting* it to run are worth writing down, because neither is
about the sky and both cost a dispatch:

* **A `workflow_dispatch` API call 404s unless the workflow file exists on the
  default branch.** `ring.yml` lived only on the channel branch, which is why
  this channel had never been run at all. The file on `main` is inert (a
  dispatch-only trigger); a dispatched run still executes the copy on its own
  ref, so the branch stays the place work happens.
* **The sandbox and the runner do not run the same library stack.** The
  sandbox venv holds pandas 2.3.3; a runner that installs the package fresh
  gets pandas 3, where `future.infer_string` is the default and catalogue text
  is Arrow-backed. An absent ATNF `assoc` field then survives `.astype(str)`
  as `NA` and a token test against it raises rather than returning `False`.
  Catalogue text is therefore read only through `screen.text_column`, which
  maps element by element and sends every missing value to `""` — the same
  reader also removes a silent failure mode on the older stack, where a
  float-`NaN` column came back as the *string* `"nan"` and the veto tokens
  were being matched against that.

## Run 35752692549 (2026-09-22, the first solo run): what it said and why it was wrong

Its summary read `DEGRADED (wd, ffp not reached); RING_CANDIDATES_PENDING_VET`
with six pulsar "ring candidates", and had no generation timestamp. Every one
of those statements traced to the pipeline, not the sky.

**Why the legs were not reached** (job 106830605498 log):

* *wd* — route A's Gaia-archive tables (`external.gaiaedr3_wd_main[_v2]`) do
  not exist (HTTP 500/400 on the probe). Route B's VizieR SELECT hard-coded
  `"chi2H"`/`"chi2He"`, which `J/MNRAS/508/3877/maincat` does not have, and
  TAPVizieR rejects a query naming any unknown column (`Unknown column
  ""chi2H""`), on every mirror. Had it got past that, `harmonise_wd` would have
  crashed on the absent chi-square columns, and the upload-join merge would
  have suffixed the Gaia photometry to `_x`/`_y`.
* *ffp* — Faherty+2016 names objects in a column literally called `2MASS`; no
  `name` pattern matched it, so no table carried the required (name, L_bol)
  pair (`table14` scored 1 of 2). The luminosity table also has no group or
  age column, and the screen would have crashed on the absent `age`.
* *bd*, reported `OK`, was not: `SpTO` (optical type, which late-T/Y dwarfs do
  not have) was picked over `SpTIR` on the 20 pc census, giving 0 of 682 late
  targets and a fallback to Kirkpatrick+2019 without proper motions; only 28
  of 232 targets returned NEOWISE epochs and 11 were tested. The loop's
  15 000 s budget exceeded the step's `timeout 5400`, so it was killed without
  writing an acquisition record — hence `acquisition.bd = NOT_RUN` beside a
  screen that said `OK`, the internal inconsistency in that summary.

**Every ring-band pulsar counterpart and its fate** (numbers from that run's
`pulsars_screened.csv`; the chance rate is the local control count):

| pulsar | pos. err | colour (source) | fate |
|---|---|---|---|
| J1453+1902 | 49.5″ (artefact) | 1.55 (CatWISE, W1 18.8) | an MSP with an ecliptic timing position; the parser gave every ecliptic position a fixed 0.01° unit (244 hosts), inflating its radius to 8″. The CatWISE source at 5.7″ is not at the pulsar. |
| J1633−2009 | 5.1″ | 1.38 (AllWISE, `ph_qual` UUBU) | the "colour" was two upper limits; CatWISE measures the same source at W1−W2 = 0.31 (stellar); 14/16 CatWISE controls hit. |
| J1847−0308_P | 60″ | 1.64 (CatWISE, W1 14.0) | position known to an arcminute, Galactic plane, 13/16 controls hit. |
| J1854+40 | 900″ | 1.82 (CatWISE, W1 18.9) | discovery position only; 11/16 controls hit. |
| J2016+4231 | 895″ | 1.27 (CatWISE) | discovery position only, Cygnus; 11/16 controls hit. |
| J2201+33 | 600″ | 1.53 (CatWISE, W1 18.8) | discovery position only; DM distance 50 kpc (saturated). |
| J0040−7337 | 3.2″ | 1.37 (CatWISE) | vetoed then and now: SMC, SNR DEM S5 / PWN. |
| J1750−3703D | 0.02″ | 1.44 (CatWISE) | vetoed: NGC 6441 globular cluster. |
| J1823−3021D | 0.36″ | 1.23 (AllWISE, W1 6.4) | vetoed: NGC 6624 globular cluster (a bright cluster giant). |
| J1929+2355_P | 60″ | 2.10 (CatWISE) | vetoed: He-WD companion; also unlocalised. |

The six "survivors" passed because the ring chance probability used the
*global* AllWISE ring-colour rate (p = 3.0×10⁻⁴ identically for all ten)
whatever catalogue supplied the colour, unscaled for aperture, with the 8″ cap
standing in for positions uncertain by up to 15′. On the sky the census is
pure chance where it should be: over the 1,792 hosts whose 2σ error exceeds
the 8″ cap, AllWISE/CatWISE counterparts number 626/1,226 against 604/1,211
expected from their own offset positions; only the 2,526 localised hosts show
an excess (152 vs 70 AllWISE, 222 vs 184 CatWISE), which is the known
companions, clusters and nebulae. The two pulsar-planet systems have no
counterpart in either catalogue (radius 1.5″, 0 control hits each).

**Fixes** (branch `claude/handoff-ring`, offline tests for each): runtime
VizieR column resolution with declination-banded pulls; `2MASS` name role,
membership-table join and BANYAN-style group aliases; ecliptic positions
keep their own precision; AllWISE colours only from `ph_qual` A/B/C;
aperture-area-scaled chance prior, ring rate from the colour catalogue; new
vetoes `position_not_localised` (2σ error > 8″), `colour_not_secure` (the 2σ
blue edge leaves the band), `aperture_confusion` (≥2 sources in the 6.5″
beam); IR spectral type and CatWISE positions for the brown dwarfs, a NEOWISE
budget under the step timeout, an `IN_PROGRESS` marker, and a leg below 50%
epoch coverage reported DEGRADED; the psrqpy `get_version` property; strict
JSON; provenance (`generated_at`, `run_id`, `git_sha`) on every JSON and per
leg in the summary; and a self-consistency block in `summary.json`.

**Re-run with the fixes (run 35860901093, legs pulsar + ffp, 2026-09-23
12:30–12:35 UTC).** Pulsars: 4,318 hosts, 2,770 localised (the 244 ecliptic
positions recovered); 1,445 with a counterpart, 1,120 of them vetoed
`position_not_localised`. The chance census is now calibrated: unlocalised
hosts 549/1,071 AllWISE/CatWISE counterparts vs 527/1,059 expected; localised
160/253 vs 80/213, the excess being catalogued associations. Ring-band
colours: 8 observed vs 9.6 expected by chance in CatWISE (2 vs 1.6 among
localised hosts) — no population excess. All 8 ring-band counterparts are
vetoed (three globular-cluster/SMC-PWN associations, five unlocalised
positions), so **0 ring candidates**. One counterpart survived the per-host
screen: J0418−4154, an AllWISE source at 0.47″ with W1 = 18.1 and
W1−W2 = 0.48 ± 0.52 (not a ring colour), p_chance = 0.006 — which across
2,770 localised hosts is p_trials ≈ 1; the look-elsewhere veto added after
this run removes it. The free-floating leg reached data for the first time
(69 objects, 63 with L_bol) but tested none: the membership column taken,
`Mm`, holds the membership *class* (HLM/AM/BM/NM), not a group, so no object
had an age. Fixed by choosing the group column by content.

**Verification run with both follow-up fixes (run 35864331050, legs pulsar +
ffp, on the side branch `claude/handoff-ring-verify` so it could not race the
full run, 13:02–13:08 UTC).** Pulsars: identical census; J0418−4154 now
vetoed `not_significant_after_trials`; **0 survivors, 0 ring candidates**.
Free-floating objects: group column chosen = BANYAN II `GBII` (148 resolvable
values vs 0 for `Mm`), 67/69 with a group age, **61 testable**, 8 catalogued
planetary-mass (≤ 13 M_J); **0 flags** — every planetary-mass object sits at
or below the 13 M_J cooling ceiling plus its 0.5 dex margin (closest:
2MASS J00470038+6803543, AB Dor, 11.8 M_J, 0.004 dex below; the TW Hya and
β Pic members are 1.0–1.5 dex below). 43 objects exceed the ceiling, all
catalogued above 13 M_J, i.e. ordinary brown dwarfs. Self-consistency block:
11 checks, 0 failures.

A note on the brown-dwarf leg's scope: it tests *W2 variability* (a duty
cycle), not a static W1/W2 excess. The static test would be wrong here as
built — CH₄ absorption in W1 makes late-T/Y dwarfs intrinsically very red in
W1−W2 (≳2–3 mag), so any excess must be measured against an empirical
spectral-type–colour relation, never a blackbody — which is why it is not run.

## Why the hotter question, in one number

OSSUARY asked the warm-dust question of 6,192,472 stars that cannot have made
the dust (metal-poor or halo-kinematic) and produced a census
(`results/ossuary/`) that is entirely **cold**: of its 251 follow-up
survivors, **none** has a W1 or W2 excess above 3σ, **251/251** are
significant in W3, the fitted dust temperature has a median of **182 K**, and
only **19** reach the 250–800 K band at all. The Osmanov ring prior lives two
bands hotter than anything that census contains. That is the sense in which
RING is a different question rather than a refinement: same photometry, a
temperature prior where the existing warm-dust channels have nothing, over
hosts where the biological alternative does not exist.
