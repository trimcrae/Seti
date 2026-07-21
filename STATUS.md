# Research scoreboard

Live per-channel state of the search. Update this file whenever a run,
vet, or triage changes the candidate picture — it is the single place a
human (or a fresh agent session) looks to know what is hot and what to do
next. Last updated: 2026-07-21.

### New channel: long-baseline Galactic-orbit encounter search (`galactic/`)

**Question (user-directed, 2026-07-21):** expand the bio/techno-signature search
to **both** nearby biosignature-anchor systems (LHS 1140 + K2-18) **and any star
that passed near them over the past few hundred Myr**. The panspermia encounter
code is linear-motion (valid only ~10 Myr); a hundreds-of-Myr baseline **requires
integrating orbits in the Galactic potential** — differential rotation and the
vertical tide bend every trajectory well inside that window.

**Method (dynamics unit-tested offline, `test_galactic.py`, 5 tests):** an
axisymmetric MW potential (Miyamoto-Nagai disk + logarithmic halo, flat rotation
curve `V_c(R0)=232 km/s`) with a vectorised velocity-Verlet integrator; resolve
each anchor's 6D phase space, pull the RV-complete Gaia sample in a present-day
sphere, integrate every orbit back `t_max` (300 Myr), and track each star's
closest approach with an **analytic per-step segment minimum** (so a ~30 km/s
flyby is not stepped over between samples). Monte-Carlo the closest encounters and
report a **timing-recoverability flag** — the honest horizon beyond which phase
mixing erases the encounter *time* even where `d_min` stays robust. Cross-match the
shortlist to the NASA Exoplanet Archive and run the signature battery: the
astrometric hidden-companion (techno) screen on every encounter star, and the
biosignature-detectability (bio) answer on the anchors + any planet-hosting
encounter systems.

**Bio contrast that validates the framework (computed offline, confirmed on the
runner):** biosignature detectability is set by the atmosphere's mean molecular
weight, and the two nearby biosignature worlds sit on opposite sides of the line —
**K2-18 b** (expected low-μ hycean H₂ envelope, scale height **H=79 km**) is
biosignature-**REACHABLE in <1 transit** (exactly why the contested DMS claim,
Madhusudhan+2023, was even possible there), while **LHS 1140 b** (high-μ rocky
secondary, **H=3.6 km**) needs **~25 transits** → not detectable. Same JWST, same
distance; the atmosphere decides.

**Result (run 29793496625, committed under `results/galactic/`): both nearby
systems and their few-hundred-Myr encounter neighbours are clean of any bio or
techno signature.** For each anchor **149,979** RV-complete Gaia stars were
integrated back 300 Myr:
- **LHS 1140:** 34 stars pass within 3 pc; **0 are known planet hosts** (no bio
  target), **0 carry an astrometric companion flag** (no techno). Closest pass:
  Gaia DR3 `1939760926285276544` (G=7.4, now 132 pc away) at **d_min=0.086 pc**,
  2.9 Myr ago, RUWE 0.88 = ordinary single star — an interesting closest-stellar-
  approach, not a signature.
- **K2-18:** 16 stars within 3 pc; **0 planet hosts, 0 companion flags.** Closest:
  Gaia DR3 `983333660069405824` at **d_min=0.25 pc**, 1.9 Myr ago.
- **Empirical recoverability horizon (the honest headline):** *every*
  reconstructable close pass is **recent** — all 34 of LHS 1140's are within the
  last 4 Myr; 15 of K2-18's 16 are within 20 Myr (one outlier at −155 Myr). The
  300 Myr integration surfaces **no datable 100–300 Myr-ago close encounter**: over
  that baseline phase mixing erases the timing, so the only stars still traceable to
  a <3 pc approach are those making *recent* passes. The Monte-Carlo confirms it —
  the closest encounters have `t_enc` spreads of ~0 Myr (tightly recoverable) and
  all sit in the last few Myr. A "few hundred Myr" encounter search therefore
  collapses, in practice, to a recoverable window of **~tens of Myr** — exactly the
  regime the linear panspermia search already covered, now put on a rigorous
  orbit-integrated footing with the horizon quantified rather than assumed.

Net: no encounter neighbour of either biosignature world is a planet host or shows
a hidden-companion technosignature; the anchors' own bio answers stand (LHS 1140 b
not detectable, K2-18 b reachable). 15 offline tests (10 galactic/bio + 5 dynamics).

### New channel: LHS 1140 system deep-dive (`lhs1140/`)

**Question (user-directed, 2026-07-21):** LHS 1140 b is a ~1.7 R⊕ temperate
habitable-zone rocky/water world (M4.5V host at 14.96 pc) with a *reported
atmosphere* (Cadieux+2024). Exhaustively search every observation ever done of
the planet, its sibling (LHS 1140 c), the star, and the stellar neighbours for
any bio or techno signature. Because both planets **transit**, every photometric/
spectroscopic observation of the star is also an observation of the planets, so
this is a full multi-archive sweep of the system plus a catalogue-scale battery
over the local volume. Channel: `src/seti/lhs1140/`, workflow `lhs1140.yml`,
8 offline tests.

**Method (reuses the panspermia per-target detectors + two new pieces):** resolve
LHS 1140's live Gaia DR3 row (nearest source in a PM-tolerant cone →
`2371032916186181760`), then run the full battery — Gaia astrometric
hidden-companion, WISE IR-colour excess, NEOWISE mid-IR variability, ZTF g+r +
**TESS/K2** photometry (which carry the b/c transits), Gaia XP narrow laser-line
scan — plus (1) a **neighbour sweep** applying the IR-excess and companion screens
to every Gaia source within a distance sphere (PM-propagated IRSA WISE cones,
NASA-Exoplanet-Archive cross-match), and (2) a **biosignature-observation
inventory** (MAST) recording what atmosphere-capable spectroscopy exists.

**Result (runs 2–3, committed under `results/lhs1140/`): the system is clean of
technosignatures in every channel that returned data.**
- **LHS 1140 (star + planets b, c):** clean in all 6 channels — no Gaia XP laser
  line; no WISE IR excess (W1–W2=0.22, W1–W3=0.37, both below threshold — ordinary
  M-dwarf colours); no NEOWISE mid-IR trend (296 epochs); no anomalous transit in
  **TESS (3,548 epochs)** or ZTF (the real b/c transits are periodic/shallow, not
  flagged). The lone flag is **RUWE=1.53** (just over the 1.4 line) — *not* a
  technosignature: astrometric excess noise is only 0.28 mas (below the 1 mas
  amplitude gate), no Gaia NSS solution; at most a faint unseen stellar companion,
  ordinary for a nearby high-PM M dwarf.
- **Neighbours (38 stars ≤15 pc):** the raw WISE screen flagged 15/38 IR-excess —
  a 40% rate that traces **entirely to AllWISE systematics**, not waste heat. The
  hardened screen (require the excess in a star-dominated band W1–W2/W1–W3 with a
  physical W1–W2≥−0.05; W4-only and negative-W1–W2 → `needs_vetting`) drops it to
  4 survivors, and those are all explained too: three are faint (G=18–20, WISE
  confusion-limited) and one is **blue (bp_rp=1.16, not an M dwarf** → photosphere
  model invalid → blend/background); the single reasonable one (G=14.4 mid-M,
  W1–W3=0.73) is a mild W3 excess with photospheric W1–W2 = ordinary debris/cirrus.
  **No neighbour shows the hot-band (W1–W2) waste-heat signature.** 6 neighbours
  carry elevated RUWE (ordinary binaries); 3 planet hosts in the volume
  (LHS 1140 + HIP 4845 + one more).
- **Biosignature inventory:** **533 MAST observations, 209 spectroscopic**,
  atmosphere-capable = **True** — HST (WFC3/STIS/COS 409) + JWST
  (MIRI/NIRISS/NIRSpec 71), 0.1–16.5 µm, ~1.53 Ms total exposure.
- **Biosignature ANSWER (`biosignature.py`, `results/lhs1140/biosignature.json`,
  6 offline tests):** the detectability calculation converts "the spectra exist"
  into the actual answer. A molecular biosignature lives in the transmission
  spectrum, whose feature amplitude is `2 Rp H n_H / Rs²` with scale height
  `H = kT/(μ g)`. For LHS 1140 b (g ≈ 18 m/s², a dense ~5.6 M⊕ super-Earth) the
  **physically expected high-μ secondary (N₂) atmosphere has H ≈ 3.6 km → ~3.6 ppm
  per scale height**, so against JWST's ~26 ppm per-bin per-transit noise **CH₄
  (3.3 µm) needs ~25 transits, O₃ (9.6 µm) ~67, N₂O/CH₃Cl/O₂-CIA more** — versus
  the handful (~2–4 epochs) actually observed. **Verdict:
  `BIOSIGNATURE_NOT_DETECTABLE_WITH_CURRENT_DATA`.** The required bands *are*
  covered (MIRI→O₃/N₂O, NIRSpec→CH₄/CO₂, NIRISS→H₂O/O₂-CIA), so this is a
  sensitivity limit, not a coverage gap. A biosignature would be reachable only
  under a cleared low-μ (H₂-rich) envelope (H ≈ 44 km, <1 transit) — which the
  planet's density and the existing atmosphere data **disfavour**. This matches
  the literature: LHS 1140 b shows a *tentative secondary atmosphere / water-world
  hint* (Cadieux+2024), **no biosignature gas**, and reaching one needs dozens–
  hundreds of transits.

**Read:** LHS 1140, its planets, and its ≤15 pc neighbours are **clean of any
technosignature** in every public archive reached (Gaia astrometry+XP,
WISE/NEOWISE, ZTF, TESS/K2), and the **biosignature question is now answered too**:
with current JWST data a biosignature is **not detectable** on LHS 1140 b, because
its expected high-μ atmosphere makes every biosignature feature a few ppm — dozens
of transits below reach, not a matter of looking harder at existing spectra. This
is a complete *characterisation* of an individual high-value system (both bio and
techno), not a population null. Known gaps: radio/SETI, high-res HARPS/ESPRESSO
RV, X-ray; a true spectral *retrieval* on the raw JWST products (vs this
signal-to-noise budget) remains a heavier, non-catalogue-scale follow-up.

## Current best candidates (cross-channel, ranked)

1. **167 triaged laser-line priority targets** —
   `results/spectra_triage/priority_targets.csv`. The former #1 (spec 068839f0,
   7518/7542 Å) is **DEAD** — see Resolved below; the two "beacon" lines are
   Hα + [N II] 6584 at z = 0.145, an emission-line galaxy SDSS misclassified as
   a STAR. A new galaxy-redshift-consistency stage now removes such objects
   (`triage_verdict = galaxy_zmatch`; 3 spectra cut, two of them
   SIMBAD-confirmed galaxies — KUG 1207+134, Z 521-35). None of the remaining
   167 is cross-confirmed.
   *Next decisive test:* the `spectra-confirm` repeat-visit path is exhausted
   (zero overlapping SPARCL spectra for 19 single-line targets). Real remaining
   route: **per-exposure persistence** — fetch the coadd-input exposures for each
   target from the SDSS SAS (new acquisition code, runner-side); a real line
   persists across exposures, a cosmic ray does not. Single-line targets cannot
   be galaxy-tested internally — the per-exposure check is what separates a true
   narrow emitter from a cosmic-ray hit for them.
2. **WD IR-excess multimodal candidates — RESOLVED, no technosignature.**
   The 23 multi-axis excesses were put through the empirical WISE-blend +
   co-movement test (`science-blend`, `results/science/blend_followup.csv`) and a
   dust-sublimation physical filter: **3 are background WISE blends** (a bright
   Gaia neighbour in the 6.5″ beam), **7 are unresolved stellar companions**
   (fitted T_dust > 1800 K = hotter than grains survive → a WD+dM/dL binary, a
   single Gaia source, which is why it looked "isolated" and periodic), and the
   **13 survivors all have τ < 0.081 — ordinary WD debris disks**, nothing
   swarm-like. The τ=0.607 standout (`235890564653455488`, T_dust 2282 K) is a
   too-hot-for-dust stellar companion, not a swarm. No candidate survives.

## Resolved (killed) candidates

* **Laser-line #1, spec 068839f0…** (SDSS-DR17, RA 25.6212, Dec −8.2417) —
  ranked first in the whole search: a 31.9σ unresolved line at 7517.96 Å plus a
  second surviving line at 7542.23 Å. **Killed 2026-07-01 by internal
  redshift-consistency**: the pair is Hα 6562.8 and [N II] 6583.5 redshifted to
  **z = 0.1452** (residual 22 km/s on [N II]) — a background emission-line galaxy
  the SDSS pipeline classified as `STAR` (catalogue z ≈ 0, so the observed-frame
  known-line triage placed Hα at 6563 Å and never saw it). New rejection
  `seti.spectra.galaxy_reject.galaxy_redshift_match` (verdict `galaxy_zmatch`).
  A locked diagnostic pair (Hα+[N II], the [O III]/[S II] doublets) or ≥3 lines
  at one z is required, so an emission-line variable star is not mis-killed
  (V345 Sge was correctly spared).
* **Astrometric dark-companion class-3 shortlist** — the 8 AMRF class-3 systems
  (BH1 + 7) were cross-matched against the published Gaia compact-companion
  catalogue Shahaf+2023 (VizieR J/MNRAS/518/2991, 101,380 source_ids loaded;
  `results/accel/literature_crossmatch.csv`). **7 of 8 are already in Shahaf+2023**
  (1 is Gaia BH1, the validation object) — the channel *reproduces* the published
  AMRF catalogue rather than extending it. One system, **Gaia DR3
  3027759739607108992** (852 pc, M₂≈4.4 M☉, RUWE 4.9, no SIMBAD), is absent from
  Shahaf+2023, but it is the *weakest* solution in the set (farthest, lowest
  RUWE, mass nearest the 3 M☉ floor) — most plausibly below Shahaf's quality
  threshold rather than a new object. Not a remarkable novel candidate; would
  need the Shahaf+2024/2019 lists to load and an independent orbit check before
  any claim. Per the novelty directive this channel is a reproduction — change
  the question, do not write it up.
* **Gaia DR3 1268299311319369984** (RA 225.0080, Dec +26.8728) — the
  ASAS-SN-confirmed secular fader (0.073 mag/yr at 8.8σ, ~0.94 mag total,
  RUWE 0.98, `non_single_star=0`). **Killed 2026-07-01 by the NEOWISE
  counterpart test**: W1 fades at 0.0045 mag/yr (8.4σ, 20 seasons, 345
  epochs), W2 at 0.0041 mag/yr (3.8σ) — an IR/optical slope ratio of 0.062,
  precisely the standard small-grain extinction-law prediction
  (A_W1/A_optical ≈ 0.06). This is ordinary dust progressively obscuring the
  star, not a gray occulter (which would fade the IR at ≳30% of the optical
  rate) and not warm circumstellar dust (which would *brighten* W1/W2).
  Verdict `ir_fades_reddening_law`;
  `results/dimming/characterization.json`. Still an interesting *astrophysics*
  object (a decade-long monotonic obscuration event), but not a
  technosignature.


### New question (post-null pivot)

**Population-level anomaly clustering** (`cluster/`, `results/cluster/`). Every
single-object channel above closed on contamination. New, degeneracy-immune
question: does the Gaia×AllWISE **IR-excess tail over-cluster in phase space**
(3D position + velocity) beyond a magnitude/colour/distance-matched random null,
as an expanding waste-heat-building population would? `matched_null_clustering`
is offline-validated (recovers an injected co-moving group at p<0.01, clean null
on confounded sets).

**First run (cone RA200/Dec0, r10°, ≤500 pc, G<16):** 21,675 Gaia×AllWISE stars →
195 IR-excess (z≥4) sources → **position clustering p=0.64, over_clustered=False**
— the excess tail traces the ordinary stellar density; the two friends-of-friends
groups (n=23 @33 pc, n=9 @69 pc) are fully accounted for by the matched null (a
nearby-star over-density), i.e. not anomalous. The method works and this cone is
clean. *Note:* position alone is washed out by the Galactic density gradient; the
run now also tests **velocity** and **full phase space** (a technological moving
group shares a space velocity).

**Three cones now, all clean nulls** in all three spaces (p_pos / p_vel / p_phase):
- RA200/Dec0:   21,675 stars, 195 excess — p_pos=0.64.
- RA150/Dec30:  22,074 stars, 186 excess — 0.37 / 0.31 / 0.08.
- RA30/Dec−20:  22,295 stars, 183 excess — 0.61 / 0.65 / 0.59.
Consistent across ~66k stars in three independent ≤500 pc volumes: the WISE
IR-excess tail does **not** over-cluster in position, velocity or phase space; the
FoF groups are all nearby-density and non-significant under the matched null. This
is a robust null for the IR-excess indicator. A genuinely new signal would be
p_vel/p_phase ≪ 0.05 with an FoF group the matched null cannot reproduce.
*Next options:* (a) keep sweeping cones -> occurrence-style volume limit;
(b) point the same clustering engine at a **different anomaly axis** (astrometric
companion excess, UV/optical energy imbalance) where a clustered technological
population might show even though IR excess does not.

### New channel: K2-18 panspermia close-encounter search (`panspermia/`)

**Question (user-directed, 2026-07-02):** K2-18 b is the hycean world with a JWST
biosignature hint (DMS/DMSO; Madhusudhan+2023/2025 — contested; treated as the
*premise*, not a result). *If* life arose there, which stars could have received
K2-18-origin material? The transfer vector is unbound ejecta / dormant spores /
free-flying 'Oumuamua-class bodies, so the filter is **encounter geometry (close +
slow)**, not a continuous bridge — and because the stellar neighbourhood
**reshuffles over time**, the search is over *closest approach in full 6D phase
space*, not present-day proximity. This is a novel anchor + novel question (nobody
has computed K2-18's stellar-encounter recipient list); it is not a refinement of
any existing SETI baseline.

**Method (offline-validated, `test_panspermia.py`, 7 tests):** resolve K2-18's 6D
vector from Gaia DR3 (radial velocity essential); pull every Gaia DR3 source with
an RV in a heliocentric distance shell bracketing the search sphere; build
heliocentric Galactic 6D `(X,Y,Z, U,V,W)`; compute each star's **linear
closest-approach** to K2-18 (`t_enc`, `d_min`, `v_rel`) — the standard
straight-line treatment used for the Sun's own encounter list (García-Sánchez
2001; Bailer-Jones 2015+), valid over the recent few-Myr window where the Galactic
tide is negligible. Rank *past* (`t_enc<0`) close/slow encounters by a
transfer-plausibility score `(d_ref/d_min)·(v_ref/v_rel)²` (velocity-squared
mirrors the gravitational-capture cross-section; ordinal, not a probability), and
separately tag **co-moving companions** (shared low velocity + present proximity),
the strongest bridge of all. Relative velocities are frame-independent of the
solar motion (it cancels in the difference), so no LSR constants enter.
*Caveat:* linear motion is honest only inside `t_max` (default 10 Myr); a longer
baseline would need epicyclic/Galactic-potential integration.

**Status:** funnel + workflow (`panspermia.yml`) built, unit-tested offline, and
**first runner dispatch complete** (run 28609098955, 2026-07-02).

**First run (K2-18 anchor, 40 pc sphere, 10 Myr window):** Gaia DR3 resolved
K2-18 at 38.02 pc with space velocity UVW ≈ (−8.2, −14.8, −8.2) km/s (a
thin-disk-normal motion). **9,980** Gaia 6D stars in the surrounding distance
shell → **4,984** had a past closest approach → **15** within `d_min ≤ 2 pc`.
Headline geometry:
- **Closest approach 0.90 pc** — Gaia DR3 `3913239815437281536` (M dwarf, G=13.7,
  35.8 pc), ≈136 kyr ago — **but at v_rel 32 km/s** (a fast flyby).
- **Top transfer score** — Gaia DR3 `4358031335898505472` (d_min 1.13 pc, v_rel
  27 km/s, ≈1.35 Myr ago), a bright G=5.6 star at 9.9 pc.
- **Zero co-moving companions** (nothing within 5 pc sharing K2-18's velocity).

**Fast-interaction / transfer-regime analysis** (`panspermia-regime`, offline
post-process of `encounters_all.csv`; `results/panspermia/transfer_regime.csv`).
Fast encounters cannot capture gravitationally, but could in principle transfer by
*geometric interception* (sweeping through the donor's reservoir), which is
speed-independent — so this was tested explicitly. Two necessary conditions, both
required: the pass must cross the reservoir (`d_min < r_reservoir`) **and** be slow
enough to bind material there (`v_rel < v_esc(d_min)`). Result across reservoir
radii from 0.5 pc (max Oort) down to 50 AU (Kuiper): **0 of 4,984 past encounters
permit any passive transfer.** The closest pass (0.90 pc) was **551× too fast** to
capture even at that distance, and **nothing** came within even a 0.5 pc reservoir
to intercept. For a 0.36 M☉ donor the escape speed at the Oort edge is ~0.12 km/s,
while the *slowest* encounter in the whole sample is 2.97 km/s — so the local
RV-complete neighbourhood is closed to K2-18 panspermia by 2–3 orders of magnitude
in *both* speed and distance. (The fast tail itself is ordinary field/halo
kinematics: median 36 km/s, one 590 km/s halo interloper — not panspermia-related.)
The physics pins the *only* viable regime to an extremely close (<0.1 pc), nearly
co-moving (<0.2 km/s) pass — i.e. effectively a bound companion, of which the run
found zero. RV completeness for close-passing faint M dwarfs remains the one gap
where such a pass could still be hiding.

**Directed-travel (technological) reframing** (`panspermia-targets`,
`reachability.py` + `exohosts.py`). The passive channels assume life is cargo to
be *caught*; a **technological** disperser instead *chooses* a target, aims, and
decelerates — so relative velocity is irrelevant and reachability is trivial
(0.90 pc ≈ 3 ly ≈ 30 yr at 0.1c; every neighbour is a short hop). The filter then
collapses to **destination quality**, and the optimal launch window is each star's
closest approach (`t_enc`, min crossing distance `d_min`). Crucially, "habitable"
is judged from the *traveller's* biology: a K2-18-evolved organism seeks other
**hycean worlds** (Madhusudhan+2021) — sub-Neptunes (1.5–2.6 R⊕) with H₂/ocean
envelopes around **cool K/M dwarfs**, over a far wider insolation range than the
rocky HZ — not Earth-analogs. The destination prior is therefore hycean-centric
(peaks on M-dwarf colours like K2-18 itself); `--target classical` gives the
Earth-analog comparison. **Offline run:** 4,984 past-close neighbours, 4,742
main-sequence; the top destinations are all cool M-dwarf hosts (the hycean-host
class), reachable in 300–1,200 yr at 0.1c. The sharp discriminator — which of them
*already* host a known planet, and specifically a **hycean-candidate** sub-Neptune
— needs the NASA Exoplanet Archive cross-match: **`panspermia-targets.yml`
dispatched** (runner-side TAP). Outputs: `results/panspermia/reachable_targets.csv`
+ `targets_summary.json`.

**Cross-match result (run 4, `targets_summary.json`):** 1,483 Exoplanet-Archive
planets within 90 pc → **109 of K2-18's past-close neighbours are known planet
hosts, and 16 host a hycean-candidate sub-Neptune** (the destination class a
K2-18 organism would seek). Top hycean-analog destinations, by closest-approach
distance (all reachable in <1,300 yr at 0.1c, optimal window within the last
~1 Myr):
| Host | d_now | d_min | t_enc | cross@0.1c | planets |
|---|---|---|---|---|---|
| **LTT 3780** (M, bp_rp 2.68) | 22 pc | **7.28 pc** | −0.67 Myr | 237 yr | 2 |
| **K2-3** (M) | 44 pc | **7.47 pc** | −0.10 Myr | 244 yr | 3 |
| GJ 667 C (M) | 7.2 pc | 14.1 pc | −0.80 Myr | 460 yr | 5 |
| GJ 357, GJ 251, Wolf 1061, L 98-59 (5 pl)… | | | | | |
LTT 3780 and K2-3 stand out: **M-dwarf hosts like K2-18 itself**, the two closest
hycean-candidate approaches, both with known sub-Neptunes (K2-3 d is a
literature hycean candidate). These are the concrete, ranked answer to "which
nearby worlds would a K2-18 civilisation choose" — the search's first positive,
specific target list. *Next decisive step:* fold in encounter-time uncertainty
(the `panspermia-mc` Monte-Carlo) for LTT 3780 / K2-3 and check whether their
sub-Neptunes truly sit in the hycean HZ (radius + insolation) vs just passing the
size cut. The bare kinematics are a necessary, not sufficient, condition.

**Per-target deep dossier** (`panspermia-dossier`, `dossier.py`). With the field
down to two objects, stop doing statistics and interrogate *every archive a runner
can reach* for each, running every signature detector this repo has:
- **Gaia DR3 astrometry** → hidden-companion diagnostics (RUWE, excess-noise sig,
  IPD multi-peak, NSS) — an unseen massive companion / anomalous acceleration;
- **WISE** → W1-anchored IR colour excess (warm dust / Dyson-like waste heat);
- **ZTF g+r** → the `seti.dimming` dip / secular-fade / glint detectors
  (megastructure transits, slow enshrouding, specular glints);
- **Gaia XP** → a narrow, interior, bounded emission spike no smooth continuum
  reproduces (a laser line), with the XP-resolution guards from the spectra channel.
Now covers **six channels** on **three targets** (K2-18 itself + LTT 3780 + K2-3),
adding **NEOWISE mid-IR variability** and **TESS/K2 photometry** (lightkurve).
Contamination discipline is built in and *earned its keep*: the first run flagged
both destinations, but both were traps — a tiny (0.1–0.2 mas) astrometric excess
flagged on σ alone (fixed: require ≥1 mas amplitude), and a 75% single-band ZTF
"dip" on K2-3 (fixed: two-band achromatic confirmation, else `needs_vetting`). A
proper-motion fix (propagate the Gaia position to each survey epoch) recovered
WISE and NEOWISE, which had silently returned no data for these high-PM stars.

**Final result (run 3, all committed):** every target is **clean in every channel
that returned data** — no IR-excess (WISE W1–W4 ≤ 0.46), no mid-IR variability
(NEOWISE 274–289 epochs), no transit-shaped anomaly (TESS ~3.2–3.5k epochs), no
unseen massive companion (RUWE ≤ 1.24), no XP laser line. Coverage is tracked
honestly (verdict reads `clean_in_N_of_6_observed_channels`): K2-18 3/6 (no Gaia
astrometry row via the cone, no ZTF, no XP), LTT 3780 & K2-3 5/6. **TESS clean on
K2-3 corroborates that its single-band ZTF g-dip was an artefact.** Known gaps:
ZTF is partial (LTT 3780 saturates ZTF at r≈11; ZTF/IRSA flaky for the rest) but
TESS supersedes it; not covered = radio/SETI, high-res HARPS/ESPRESSO RV spectra,
X-ray. 20 offline tests. Net: the two destinations and the origin world show **no
technosignature** in any public archive reached — the honest state of the deep dive.

**Read:** every *passive* encounter is *fast* (v_rel 23–54 km/s) — the signature of random
field stars passing a normal thin-disk star, not a shared-origin group. No slow,
close bridge exists in the RV-complete local sample, and the transfer scores are
all ~1e-4 (dominated by the 1/v_rel² term). This is **not a null to write up** —
the *limiting factor is Gaia RV completeness*: most nearby M dwarfs lack a Gaia
radial velocity and are excluded, so a genuinely slow/close encounter could be
hiding among them. *Next decisive moves:* (1) supplement RVs for the RV-less
nearby M dwarfs (LAMOST/APOGEE/SDSS) to close the completeness gap that a slow
encounter would live in; (2) tighten the shortlist to the only regime that would
matter — `d_min < 0.3 pc AND v_rel < 5 km/s` — and Exoplanet-Archive cross-match
any survivor; (3) if a slow/close survivor appears, replace the linear
approximation with a Galactic-potential orbit integration to confirm it.

**Monte-Carlo encounter uncertainty** (`panspermia/uncertainty.py`,
`panspermia-mc`, `results/panspermia/recipient_candidates_mc.csv`, 3 offline
tests). The base shortlist gives point estimates; the rigorous treatment
(Bailer-Jones 2015/2018) resamples both K2-18 and each candidate from their Gaia
(parallax, pmra, pmdec, RV) covariances and reports the *distribution* of
`d_min`/`t_enc`/`v_rel`. Result (5,000 draws each): the geometry is **robust** —
**13 of 15** candidates are a *past* encounter in 100% of draws and stay within
2 pc in the majority; the closest, Gaia DR3 `3913239815437281536`, is
`d_min = 0.91 pc` with a tight 16–84% band of **0.90–0.95 pc** (136 kyr ago). So
the close passes are real, not astrometric flukes — but every median `v_rel` is
**23–54 km/s**, confirming with error bars that none is capturable (the flag is
*geometric* robustness, not transfer viability; the regime analysis above owns the
capture physics). The MC therefore hardens the null: the recipient list is a set
of well-measured *fast* flybys, exactly what the transfer-regime cut rejects.

## Channel state

| Channel | Searched so far | Surviving | Blocking issue / next action |
|---|---|---|---|
| Dimming (dips + secular) | 250,862 ZTF stars, 116 fields | 0 — top fader killed by NEOWISE reddening test; **19 `marginal_fade` assessed and set aside** (all 1.6–7.4% total fades, 18/19 not band-confirmed → optical slope ~0.004–0.015 mag/yr, where even a gray occulter gives only ~2σ in NEOWISE; ZTF systematics floor) | channel exhausted at the systematics floor — do not re-chase the marginal faders; new volume only helps if it reaches ≳0.1 mag band-confirmed fades |
| Specular glint | ran on 4 fields → 15 candidates, **all vetted to 0** | 0 | every candidate is a `chromatic_flare` (M-dwarf flare, bluer in g than r → not achromatic) or dusty; `glint_confirmed=False` for all 15. Achromaticity vet kills the channel. Single huge-brightening events are asteroid/cosmic-ray artifacts; multi-event ones are red-dwarf flares |
| Laser emission (SDSS-DR17) | 10,500+ spectra (latest committed run) | 112 triaged (was 118; 3 galaxies cut, incl. former #1) | per-exposure persistence check (repeat-visit path exhausted) |
| Astrometric dark companion (Gaia orbits) | 105,066 NSS orbits, ≤1 kpc | 0 novel (8 class-3 = BH1 + 7, but 7/8 already in Shahaf+2023; 1 borderline-absent is the weakest solution) | reproduction of the published AMRF catalogue — change the question |
| Laser absorption (DESI-DR1) | 6,500+ spectra (latest committed run) | 55 triaged | same; hot-star continua only (line-forest stars skipped by design) |
| WD IR excess | 7,716 clean WDs → 23 multi-axis → blend+sublimation test | 0 technosignature (3 WISE blends, 7 unresolved stellar companions, 13 ordinary τ<0.08 debris disks) | channel resolved; τ=0.6 standout is a too-hot-for-dust stellar companion. Next volume only helps if it reaches a τ→1 excess with T_dust *below* sublimation |
| Panspermia (K2-18 close encounters) | first run: 9,980 Gaia 6D stars, 4,984 past approaches, 15 within d_min≤2 pc | 0 slow/close bridge (all v_rel 23–54 km/s; closest 0.90 pc but at 32 km/s; 0 co-movers) | **RV completeness is the gap** — supplement RVs for RV-less nearby M dwarfs, then re-cut to d_min<0.3 pc & v_rel<5 km/s; Exoplanet-Archive cross-match any survivor |
| Gaia XP anomalies | RA283/Dec−3 dense field: 8,863 sources, reliable; narrow-feature shortlist examined | 0 credible | **channel bounded — see ledger.** Broad "anomalies" = reddened-M-dwarf molecular bands (degenerate with a Dyson SED); "narrow" ones = band-edge reconstruction artifacts + sub-resolution wiggles (XP LSF ≈5+ samples can't resolve a laser line). Guards added (width/interior/bounded). A clean low-extinction field could still test the *broad*-SED Dyson signature, but it is degenerate with reddening |
| Galactic long-baseline encounters | LHS 1140 + K2-18; 149,979 RV-complete Gaia stars each integrated back 300 Myr in the MW potential | 0 bio + 0 techno among encounter neighbours | **clean.** LHS 1140: 34 passes <3 pc (0 planet hosts, 0 companion flags); K2-18: 16 (same). All datable passes are recent (<~20 Myr) — phase mixing erases 100–300 Myr timing, so the honest recoverability horizon is ~tens of Myr. Closest: a G=7.4 star 0.086 pc from LHS 1140 2.9 Myr ago (single star, not a signature) |
| LHS 1140 system deep-dive | star + b/c + 38 neighbours ≤15 pc; 6 archives (Gaia astrometry+XP, WISE/NEOWISE, ZTF, TESS 3.5k epochs); 533 MAST obs inventoried; **biosignature detectability budget** | 0 technosignature; **0 detectable biosignature** | **clean + bio answered.** Star: all 6 channels clean; lone RUWE=1.53 is marginal binarity, not techno. Neighbours: raw 15/38 IR-excess all = WISE W4/blend systematics → 4 survive → faint/blue/ordinary-debris. **Bio:** under b's expected high-μ (N₂) atmosphere (H≈3.6 km) every biosignature feature is ~few ppm → CH₄ ~25 / O₃ ~67 transits vs ~2–4 observed → `NOT_DETECTABLE_WITH_CURRENT_DATA`; reachable only for a disfavoured cleared H₂ envelope |

## Known systematics ledger (do not re-derive)

* **Gaia XP is low-resolution** (R≈30–100; LSF ≈5+ of the 2-nm samples). Two
  consequences: (1) it *cannot resolve a narrow laser line* — a real localised
  feature must be interior (≥8 samples from either end), bounded (falls below
  half-peak on both sides), and 2–5 samples wide; a 1-sample spike is
  sub-resolution noise and a monotonic ramp pinned to 336 nm / 1020 nm is a
  basis-function reconstruction artifact (both now cut in `xp.anomaly`).
  (2) Broad-SED "anomalies" (a Dyson reprocessing deficit/excess) are
  **degenerate with interstellar reddening** and with cool-star molecular bands
  (TiO/VO/H₂O) — a low-|b| field is the worst case. Fit the colour locus with
  ≥40 sources/bin (`fit_locus`) or the per-bin MAD collapses and flags ~70%.
* SDSS/DESI wavelengths are **vacuum**; all literature line lists (air) are
  converted via `seti.spectra.reject.air_to_vacuum` at definition time. This
  was a real leak: pre-fix "candidates" sat on He I 5876 / Ca II 8542 / O I
  8446. Fixed 2026-07-01.
* Catalogue redshift/RV errors move known lines outside the in-funnel
  rejection window → the observed-frame ±300 km/s triage
  (`seti.spectra.triage`, costs 22.7% of the band, honestly accounted) is
  mandatory before believing any spectral candidate.
* **Misclassified emission-line galaxies** are the worst spectral leak: a
  background star-forming/active galaxy that SDSS/DESI labels `STAR` (or gives a
  wrong z) drops its rest-frame nebular family into the search as
  high-significance "unresolved" lines. The observed-frame known-line triage
  cannot catch it (it uses the wrong catalogue z). Decisive test = *internal
  redshift consistency*: if ≥2 surviving lines in one spectrum form a locked
  nebular pair (Hα+[N II], [O III] 4959/5007, [S II] 6716/6731) or ≥3 lines at a
  common z, it is a galaxy (`galaxy_reject`, verdict `galaxy_zmatch`). This killed
  the former #1 candidate. Single-line candidates cannot be tested this way —
  they need the per-exposure persistence check.
* Candidate wavelengths recurring across unrelated sightlines (≥3 spectra
  within ±3 Å, across runs *and* modes) are instrumental. 31 killed.
* Merged candidate CSVs can contain duplicate rows (runs overlap) — 89 killed.
* ZTF single-band events are artifacts until g/r-coincident
  (`multiband_coincidence`, `secular_achromatic`, `glint_achromatic`).
* Stellar flares are chromatic (g ≫ r); a glint must be achromatic.
* WD IR-excess contaminants, in the order they bite: (1) **WISE blend** — a
  comparably-bright red Gaia neighbour inside the ~6.5″ W1 beam (the WD is
  IR-faint); test with `discriminate.blend` (Gaia beam neighbours + expected W1).
  (2) **Unresolved stellar companion** — a WD+dM/dL binary is a *single* Gaia
  source (looks "isolated") whose fitted excess temperature is >1800 K, hotter
  than grains survive: an "excess" above the dust sublimation temperature is a
  companion photosphere, not dust or a swarm (kills the τ=0.6 standout). (3) **CV**
  (accretion). Only after all three does a τ<0.08, T_dust<1800 K excess read as
  an ordinary debris disk.
* WD IR excess: dusty debris disks are the one natural confounder — subtract
  the labelled catalogues before scoring.
* **AllWISE W4 (22 µm) is unreliable for faint stars** — it is the shallowest,
  most confusion-limited band, so for 22-µm-faint M dwarfs the catalogue W4 flux
  is background cirrus / a noise measurement, producing huge (up to ~6 mag),
  formally-high-σ "W1−W4 excesses" with a *photospheric* W1−W2. A real
  warm-dust/waste-heat SED is bounded and lights up the star-dominated bands
  first, so a **W4-only excess is an artefact**, not a detection. Likewise a
  **negative W1−W2 is a W1/W2 blend** (a bare photosphere has W1−W2 ≥ 0), not a
  star. The LHS 1140 neighbour screen requires the excess in W1−W2 or W1−W3 with
  W1−W2 ≥ −0.05; W4-only/negative-W1−W2 go to `needs_vetting` (killed 11 of 15
  raw neighbour flags). Faint sources (G≳18) and non-M-dwarf (blue bp_rp) matches
  are additionally WISE-confusion/photosphere-model-invalid, not excesses.
* A secular optical fade with a NEOWISE fade at ~6% of the optical rate is
  ordinary line-of-sight dust (extinction-law ratio) — check
  `w1_to_optical_slope_ratio` before getting excited. Gray occulters sit at
  ≳30%; warm dust *brightens* the IR.
* ASAS-SN (pyasassn) is flaky on runners — pass `--optical-slope` to
  `dimming-characterize` so the mid-IR verdict never returns
  `insufficient_ir` for want of a known number.

## Rules of engagement (from CLAUDE.md)

Novelty first, scale second, never write up a null. Merge every commit to
`main` as you go (non-fast-forward merge if diverged; never force-push).
Data-touching runs go through `workflow_dispatch`; the sandbox has no archive
egress.
