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
