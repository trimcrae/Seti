# BAFFLE — the warden's fence: reciprocal shadows of a technosignature screen

**Claim being tested.** If the zoo hypothesis (Ball 1973) is the resolution of
the Fermi paradox, then at least one civilisation has decided that the solar
system is to be left alone — and a quarantine that is worth keeping is worth
keeping from *third parties* too. Our present radio leakage and our eventual
Dyson-swarm waste heat would otherwise advertise the zoo to every other
observer in the Galaxy. A warden that wants the quarantine to hold therefore
has a reason to **shield the solar system's technosignatures from other
stars**, and — because the swarm does not exist yet — a reason to have the
shielding *in place before it is needed*. This document asks three things in
order: how such shielding could physically be done; which of those methods
leave a trace we can see from inside; and what a catalogue-scale search for
that trace looks like. The search is then run.

The channel is named for the optical element: a *baffle* blocks light in a
chosen direction and passes it in every other.

---

## 1. How a warden could do it

Set the problem as the warden sees it. The source to be hidden is the whole
inner solar system: the Sun (whose spectrum a swarm would alter), a swarm of
radius ~1 AU radiating at ~300 K (the equilibrium temperature of a 1-AU
absorber; peak ≈ 10 µm), and radio transmitters on a body 1 AU from the Sun.
The observers to be blinded are other stars — all of them, in the worst case,
or the subset the warden knows to be inhabited or listening. Five families of
solution exist; the physics of each fixes whether it can be seen from here.

### (A) An enclosing shell

A spherical shell around the Sun at radius *d* that absorbs everything. In
radiative equilibrium with the Sun (two-sided radiator):

| *d* (AU) | T_shell (K) | angular radius of a 1-AU body at *d* |
|---|---|---|
| 100 | 33.1 | 34′ |
| 1 000 | 10.5 | 3.4′ |
| 3 000 | 6.0 | 69″ |
| 10 000 | 3.3 | 21″ |
| 30 000 | 1.9 | 7″ |

From inside, a full shell is an isotropic thermal background at T_shell. At
*d* ≈ 10⁴ AU that is, by an unhelpful coincidence, the temperature of the
cosmic microwave background — but such a shell would be at rest with respect
to the Sun, so it would carry no 370 km s⁻¹ dipole, and the measured CMB
dipole (3.36 mK, aligned with the Local Group motion) and its acoustic
anisotropy spectrum are the spectrum of a source far outside any shell. A
fully opaque shell is therefore excluded at every radius; a partly transparent
one is a foreground the CMB experiments would have to have mis-modelled, and
no catalogue-level test of that exists. **Family A is set aside as excluded or
untestable, not searched.**

### (B) Directional baffles

The economical solution: instead of enclosing us, put an absorbing screen on
each Sun→observer line. Geometry fixes the size. To shadow a source of radius
*r*_s toward a distant observer, the screen's radius must be ≥ *r*_s wherever
it sits (rays to a distant observer are parallel). A swarm of 1 AU needs a
screen of ≥ 1 AU radius; only its distance *d* is free, and *d* sets its
angular size ρ = *R*/*d* (table above) and its temperature.

Two sub-cases:

* **(B1) Grey (fully absorbing).** Then the target star X is hidden from us
  *entirely* — a passive absorber is reciprocal — and we cannot know X exists.
  What remains observable is the screen's own waste heat: the flux density of
  a 1-AU-radius blackbody disk at *d* is

  | *d* (AU) | T (K) | 100 µm (Jy) | 857 GHz (Jy) | 545 GHz (Jy) | 353 GHz (Jy) |
  |---|---|---|---|---|---|
  | 100 | 33 | 1.6 × 10⁸ | 1.2 × 10⁸ | 6 × 10⁷ | 3 × 10⁷ |
  | 1 000 | 10.5 | 130 | 5.9 × 10⁴ | 6.7 × 10⁴ | 5.0 × 10⁴ |
  | 3 000 | 6.0 | 6 × 10⁻⁴ | 360 | 1 100 | 1 500 |
  | 10 000 | 3.3 | — | 0.12 | 2.8 | 12 |

  and it moves on an annual parallax ellipse of semi-major axis 206 265″/*d*
  (34′ at 100 AU, 3.4′ at 1 000 AU, 21″ at 10⁴ AU). IRAS and AKARI would have
  catalogued the 100-AU case as the brightest compact source in the far-IR
  sky; Planck's 857/545 GHz catalogues would have caught anything inside
  ~3 000 AU as a tens-of-kJy source that shifts by arcminutes between the two
  half-yearly surveys. At 10⁴ AU it is a few-Jy sub-mm source at the CMB
  temperature, at the PCCS2 detection limit and *not* separable from the CMB
  by colour — the one place a grey baffle could hide, and a place catalogue
  products cannot test (Planck's catalogues are built from co-added full-
  mission maps, in which a source moving 40″ between surveys is neither a
  point source nor a catalogued object). **Signature S39, listed, not built.**

* **(B2) Band-selective (the engineered case).** The warden does not need to
  hide the *star*; the star is normal and visible from everywhere already. It
  needs to hide the *technosignature bands*: 3–50 µm for a swarm, metres to
  centimetres for radio leakage. A screen that absorbs only there passes the
  photosphere's optical light and is, in the mid-IR, effectively a cold
  notch filter: it absorbs 2.1 % of L_⊙ (the fraction of a 5 772 K blackbody
  beyond 3 µm), so its equilibrium temperature is 12.6 K at 100 AU, 4.0 K at
  1 000 AU, below the CMB at 10⁴ AU, and by Kirchhoff it emits only in the
  band it absorbs — at those temperatures, nothing detectable at 3–22 µm. A
  radio-opaque screen can be a mesh (wire spacing ≪ λ): a 1-AU-radius mesh of
  1-cm pitch and 10-µm aluminium wire is ~3 × 10¹⁸ kg, five parts in 10⁷ of
  an Earth mass, and is thermally invisible. **A band-selective baffle has no
  emission signature at all.** Its *only* observable is reciprocity.

### (C) Active camouflage

A screen that absorbs on one side and re-emits a synthetic "normal G2V"
spectrum on the other (a cloaking device in the sense of Kipping & Teachey
2016, generalised from transits to the whole SED). By construction it is
undetectable from the far side, and from our side an ideal two-sided cloak
shows us X's true light. Energy conservation forces a waste-heat dump
somewhere, which is family B1's thermal signature again, possibly beamed away
from both us and X. **Not independently searchable.**

### (D) Blinding the observer instead of cloaking the source

A screen near star X, on X's line of sight to us, hiding *our* direction from
*their* instruments. The size argument is unchanged (it must cover the
observer's own ~1-AU system), and from Earth it covers X's disk (0.01 AU)
completely. Grey: X is invisible — untestable. Band-selective: X is a
**single** star with a mid-IR (or radio) deficit and **no** surrounding patch
and **no** annual modulation (a screen at X's end does not move with Earth's
parallax). This is the `ISOLATED_DEFICIT` verdict in the patch stage — kept
distinct from a rejection, because it is exactly what family D predicts.

### (E) Intervening at the source

Reshaping or redirecting our own emission (beaming leakage away, or making
the swarm's radiators directional). Leaves no trace in any catalogue we can
reach. **Not searchable.**

### The reciprocity argument, stated once

For any passive, linear, time-invariant medium the transmission between two
points is symmetric (Lorentz reciprocity). A screen that stops our 10-µm
photons from reaching X stops X's 10-µm photons from reaching us, in the same
band, with the same optical depth. The only escapes are non-reciprocal media
(a magnetised plasma with Faraday rotation plus polarisers, or an active
device — family C) and those pay in energy, which is family B1's signature.
So: **a warden that shields us passively must, by the same act, put a shadow
of the same colour on the sky in the direction of every star it shields.**
That shadow is what BAFFLE searches for.

---

## 2. The observable: signature S38

A star (and, for a baffle at our end, everything within ρ of it) whose flux
in a technosignature band is **significantly below its photospheric
prediction** while its optical and near-infrared photometry is normal.

### 2.1 Why nothing natural does this

The search is for an *inverse extinction law*: opaque at 3–5 µm (or in the
radio), transparent at 0.5–2 µm.

* **Interstellar and circumstellar dust** extinguishes *more* in the optical
  than in the infrared (A_W1/A_V ≈ 0.04); any dusty foreground makes a star
  *redder*, never IR-faint. Grains large enough to be grey are grey, not
  inverse.
* **Free–free absorption** (τ ∝ ν⁻²) is inverse in sign but needs an
  emission measure of order 10¹⁴ cm⁻⁶ pc to reach τ ~ 1 at 10 µm — a compact
  H II region's worth of ionised gas in front of an ordinary star, itself a
  bright radio and recombination-line source.
* **Stellar atmospheres** depress W1 or W2 by known, bounded amounts: CO at
  4.6 µm makes K_s − W2 ≈ −0.1 to −0.2 in cool giants; CH₄ at 3.3 µm makes W1
  faint in T dwarfs (W1 − W2 ≫ 0, *W2 normal or bright*); C₂H₂ + HCN at
  3.1 µm depresses W1 in N-type carbon stars. None produces a *two-band*
  deficit of ≥ 0.3 mag with a normal K_s, and each is either colour-locus
  behaviour (absorbed into the empirical locus) or a named veto.
* **Time-domain mismatch** is the real contaminant: 2MASS (1997–2001) and
  AllWISE (2010) sample a long-period variable at different phases. This
  produces deficits *and* excesses of up to a magnitude in red stars. It is
  vetoed by colour and variability flags, and its residue is deferred, not
  discarded, with the fix named (a same-epoch comparison).
* **Catalogue mechanics**: saturation (W1 < 8, W2 < 7, K_s < 4.5), profile-fit
  failure (w?rchi2), blends whose flux the pipeline assigned to a brighter
  neighbour (the fainter member then shows a deficit), cross-match to the
  wrong source, artefacts (cc_flags). Each is a named veto with a counter.

Everything else — the whole natural IR-excess population of debris disks,
YSOs, Be stars, AGB shells — lives in the *positive* tail. The negative tail
should be empty. The channel reports both tails: the excess tail is the
control that shows the estimator works on the population that is known to
exist; the deficit tail is where the claim lives.

### 2.2 The geometric fingerprints

Photometry finds the star; geometry decides what it is.

1. **Patch coherence.** A baffle at our end of the line subtends ρ = *R*/*d*
   and shadows every background source inside it. The deficit fraction versus
   angular offset from X is a **top hat** of radius ρ (3.4′ for 1 AU at
   1 000 AU). A bright-star halo, a scattered-light artefact or a bad local
   calibration decays smoothly. Nothing natural makes a sharp-edged,
   arcminute-scale, colour-selective shadow.
2. **Annual parallax modulation.** The baffle must stay on the Sun→X line,
   not the Earth→X line, so from Earth its apparent centre traces an ellipse
   of semi-major axis 206 265″/*d* around X every year — the same ellipse as
   the parallax of a body at *d*, with the phase fixed by the ecliptic
   geometry. X itself stays covered (a screen of radius ≥ 1 AU always
   contains Earth's offset), so X's own NEOWISE light curve (2014–2024, two
   visits a year) must be a **flat, constant deficit** at the AllWISE level.
   Neighbours at offsets between ρ − π_b and ρ + π_b are covered for part of
   each year and uncovered for the rest, **with a phase the geometry
   predicts** — a prediction with no free parameters once (*d*, *R*) are
   chosen, tested against a phase-scrambled null.
3. **Isolated, constant deficit** with no patch and no modulation is family D.

### 2.3 Where to look first

The primary screen is all-sky: a warden hiding a swarm hides it from every
direction. Two priors are carried as *flags with their own denominators*, so
that an over-density of survivors in either is itself a test:

* **The Earth Transit Zone.** Stars within ±0.264° of the ecliptic
  (R_⊙/1 AU) see Earth transit the Sun and can already read our atmosphere
  in transmission (Heller & Pudritz 2016; Kaltenegger & Faherty 2021). These
  are the observers who may *already* have found us; they are the first the
  warden must blind.
* **Nearby stars** (parallax > 20 mas, *d* < 50 pc): the ones whose leakage
  detection thresholds we are closest to crossing.

---

## 3. The funnel

### 3.1 Mid-infrared (the `deficit` and `missing` tracks)

Data: Gaia DR3 × 2MASS × AllWISE through the Gaia archive's official
proper-motion-propagated cross-match tables (Marrese et al. 2019) — the same
route as OSSUARY, with the CENOTAPH acquisition discipline (COUNT(*)
reconciliation, random_index slicing, per-chunk ledger, transport cooldown)
because the ESA archive cuts synchronous queries at 60 s and its anonymous
asynchronous result store fills up.

The pre-selection is done *in the archive*: only rows with K_s − W1 < −0.15 or
K_s − W2 < −0.15 are returned, plus a uniform 0.5 % calibration subsample
(random_index < 9 058 548). The photospheric locus is empirical — the running
median and robust scatter of K_s − W_b versus J − K_s, per luminosity class,
fitted on the calibration subsample after quality cuts — so no stellar model
is trusted for the prediction. A candidate must show a deficit of ≥ 0.30 mag
at ≥ 5σ in **both** W1 and W2; W3/W4 are reported where measured.

Vetoes, in order, each counted: `saturated`, `poor_wise_phot_qual`,
`poor_tmass_phot_qual`, `wise_artifact`, `extended`, `bad_profile_fit`,
`wise_variable`, `gaia_variable`, `lpv_colour` (deferred, not dropped),
`crowded_match`, `blend_flux_theft`, `multi_peak`, `w1_only_methane_like`;
`bad_astrometry` and `high_pm_epoch_risk` are reported, not applied (a baffle
does not perturb astrometry, and the archive cross-match propagates proper
motion — the vet re-checks it).

The `missing` track is the fully-opaque limit of the same thing: a bright
(G < 13, K_s 5–11) 2MASS-AAA star with **no AllWISE source at all** within the
cross-match radius, at |b| > 10°. WISE is complete to W1 ≈ 17 there, so a
K_s = 10 star with no WISE counterpart is either a catalogue mechanic or a
shadow. The missing fraction is reported against |b| and G so that Galactic-
plane confusion is visible as the smooth function it is, not mistaken for a
population.

### 3.2 Patch and modulation (the `patch` stage)

For every survivor: Gaia + 2MASS + AllWISE neighbours within 10′ and their
residuals against the same locus; the deficit-fraction profile in radius; a
scan over (*d*, *R*) with the coverage each neighbour should show at each
NEOWISE epoch; the constant-deficit test on X's own NEOWISE series; a
phase-scrambled null. Verdicts `COHERENT_PATCH`, `MODULATED`,
`ISOLATED_DEFICIT`, `NOT_COHERENT`, `INSUFFICIENT_NEIGHBOURS`.

### 3.3 Radio (the `radio` stage, `baffle-radio.yml`)

The radio baffle is the one most likely to exist *now*. It is thermally
invisible, so the only observable is a **hole in deep radio source counts**
centred within one parallax amplitude of a nearby star. Only LoTSS DR2
(144 MHz; ~900 sources deg⁻²) is deep enough: a 1-AU screen at 1 000 AU
(ρ = 204″) empties an aperture that should hold ~9 LoTSS sources
(P ≈ 10⁻⁴ for one star; NVSS would hold 0.5 and cannot see a void at all).
For every Gaia star within 50 pc in the DR2 footprint: Poisson void
probability in apertures from 30″ to 10′ against the local density from an
8′–20′ annulus, evaluated at the annual set of possible baffle centres for
*d* from 500 to 10⁴ AU (the look-elsewhere cost is counted), with the same
statistic at four control positions 45′ away as the empirical null. Vetoes:
`outside_footprint_or_masked`, `bright_radio_source_mask`,
`low_expected_count`, `not_significant`.

---

## 4. Sensitivity, and where the search is blind

* The deficit track is complete for two-band deficits ≥ 0.3 mag on stars with
  K_s > 4.5, W1 > 8 and W2 > 7, i.e. **the brightest ~10⁴ stars in the sky —
  including most of the nearest ones — are outside it**, because 2MASS and
  WISE saturate there. For those the reciprocal shadow is read from the
  unsaturated far side: the `bright` tier (`bright.py`, `baffle-bright.yml`)
  takes every Gaia star with G < 7.5, anchors it on 2MASS Read-1 photometry
  (0.2–0.3 mag errors, carried), and tests AKARI/IRC 9 and 18 µm and IRAS 12
  and 25 µm against an empirical K_s − [9] locus, demanding a ≥ 0.7 mag
  deficit in two bands from two instruments. A warden's screen is presumably
  near-opaque in band, so the coarser precision costs little there. Gaia is
  itself incomplete below G ≈ 3, which the tier records.
* A baffle that passes 3–5 µm and blocks only 8–50 µm (a screen tuned
  narrowly to a 300 K swarm) is seen only in W3/W4, which reach W3 ≈ 11 and
  W4 ≈ 8: a few million stars, not fifty million. The W3 residual is
  computed and reported wherever W3 is measured.
* A baffle at 10⁴ AU has ρ = 21″: below the WISE beam. The patch test cannot
  resolve it; the star's own deficit and its constancy remain.
* The radio search covers the LoTSS DR2 footprint (27 % of the northern
  sky), one epoch per field.
* A warden using family C or E leaves nothing here to find, and a warden
  using family B1 at 10⁴ AU leaves a sub-mm source no catalogue can separate
  from the CMB. The honest statement is that BAFFLE tests **passive,
  band-selective, our-end or their-end screens** — the cheapest design —
  and nothing else.

---

## 5. Novelty

No published search selects stars for a mid-infrared *deficit* relative to
the photosphere; the excess is the object of every Dyson and debris-disk
search, and the negative tail has been treated as noise or cross-match error
when it was looked at at all. No search has asked whether deep radio source
counts have holes at stars. No zoo-hypothesis paper has proposed a catalogue
observable; the literature (Ball 1973; Baxter 2001; Forgan 2011, 2017;
Crawford & Schulze-Makuch 2024) argues the sociology and the geometry of a
"Galactic club", not a test. Kipping & Teachey (2016) proposed that *we*
could cloak our transit with a laser, which is family C from the inside. The
runner-side sweep `scripts/bafflelit_fetch.py` (`bafflelit.yml`) fetches the
record verbatim and scans it with decoys so that this paragraph can be
checked rather than believed; its result is recorded in `STATUS.md` (read 2026-09-06: no catalogue search for stellar mid-IR deficits, no source-count-hole search at stars, no zoo-hypothesis observable in the record; the nearest neighbours are Lacki 2019 "Sunscreen" and Kipping & Teachey 2016, both about the concealer's own light).

---

## 6. Honesty about the premise

The chain from "the zoo hypothesis is true" to "a screen at 10³ AU shadows
star X at 4.6 µm" has several links, each a choice the warden might not make.
What keeps the search worth running is that the **observable stands on its
own**: a star fainter than its photosphere in two mid-infrared bands, with a
sharp-edged arcminute shadow around it or an annual modulation with the
parallax phase, is an object with no natural explanation, whoever put it
there. The zoo hypothesis is the reason to look; the physics is the reason a
find would matter. And per the charter, a clean null here changes the
question, not the venue: it says the cheapest passive design is not in use,
which narrows the warden to families C, E, or a B1 shell at the CMB
temperature — and those are different questions.

---

## 7. Files

| | |
|---|---|
| Package | `src/seti/baffle/` — `acquire.py`, `locus.py`, `screen.py`, `patch.py`, `radio.py`, `bright.py`, `run.py` |
| Config | `config/baffle.yaml`, `config/baffle_radio.yaml`, `config/baffle_bright.yaml` |
| CLI | `python -m seti.cli baffle --stage {probe,acquire,screen,patch,radio,assess,all}`; `python -m seti.baffle.radio --stage {probe,run}`; `python -m seti.baffle.bright --stage {probe,run}` |
| Workflows | `baffle.yml` (mid-IR: probe → sharded acquire → screen + patch → commit), `baffle-radio.yml`, `baffle-bright.yml`, `bafflelit.yml` |
| Results | `results/baffle/` (`summary.json`, `candidates.csv`, `vetoed.csv`, `missing_candidates.csv`, `locus.json`, `patches.csv`, `acquisition_ledger.json`), `results/baffle_radio/`, `results/baffle_bright/`, `results/bafflelit/` |
| Tests | `tests/test_baffle.py`, `tests/test_baffle_patch.py`, `tests/test_baffle_radio.py`, `tests/test_baffle_bright.py` |
| Signatures | S38 (reciprocal shadow of a band-selective baffle), S39 (the grey baffle's own waste heat — listed, not searchable at catalogue level) in `docs/necrosignatures.md` |
