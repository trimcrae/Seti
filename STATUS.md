# Research scoreboard

Live per-channel state of the search. Update this file whenever a run,
vet, or triage changes the candidate picture — it is the single place a
human (or a fresh agent session) looks to know what is hot and what to do
next. Last updated: 2026-07-21.

### New: 5-channel fan-out searching for life originating on LHS 1140 b (2026-07-21)

Five parallel-subagent-built channels (77 offline tests total), all runner-dispatched.
Each degrades honestly and never fabricates data.

1. **`jwst_bio`** — real JWST transmission-spectrum biosignature analysis (build the
   spectrum from `x1dints` in/out-of-transit; disequilibrium-*pair* logic CH₄+CO₂/
   O₂+CH₄, never a single gas; M-dwarf abiotic-O₂ gate; MIRI eclipse
   atmosphere-vs-bare-rock; laser scan). Took **three runs and two real
   data-access fixes** to reach the archive: (1) the MAST obs filter required
   `dataproduct_type=="spectrum"` but JWST TSO is `"timeseries"` → 0 products
   (fixed → 70 x1dints found); (2) all 70 downloads hit the MAST
   `download_products` "varchar to bigint" server bug → switched to
   `download_file(dataURI)`. **Run 3 reached real data:** `data_reached=true`, 2
   x1dints stacks read (NIRISS + NIRSpec). **Verdict `no_biosignature_detected`
   — correct and robust:** NIRISS covers only 0.85–2.83 µm, so CH₄ (3.3)/CO₂
   (4.3)/O₃ (9.6 µm) are out of range → **no redox-disequilibrium pair is even
   possible**, and a single gas is never a biosignature. **Vetting note (do not
   overclaim):** the pipeline's apparent "H₂O 1.4 µm, 72 ppm, 7σ" is a **reduction
   artefact, not a real detection** — the reader saw only 3 integrations (it treats
   each EXTRACT1D HDU as one integration, but modern x1dints pack all integrations
   in one 2-D table), the in/out split ran with `ephemeris_used=false`, and the
   per-band σ is inflated by counting ~46k correlated native pixels as independent.
   A valid transmission spectrum here needs multi-integration format handling +
   ephemeris phasing + systematics detrending + de-correlated binning — the
   publication-grade retrieval the channel explicitly disclaims. **Bottom line
   unchanged:** the biosignature *detectability* answer governs — LHS 1140 b's
   compact high-μ atmosphere puts every biosignature ~25+ transits out of reach, so
   none is detectable with current data. The infrastructure now reaches the real
   spectra; a genuine retrieval is the honest next boundary, not a runner task.
2. **`lhs1140_origin`** — panspermia **donor** list (classical rocky-HZ prior,
   mirror of K2-18). **Run OK:** 10,974 Gaia 6D stars → **22 recipients** within
   2 pc over 10 Myr, **0 co-movers**, closest approach 0.26 pc — but **all fast
   flybys** (top v_rel 51 km/s, transfer scores ~1e-3 = non-capturable), exactly
   like K2-18: no slow/close passive bridge. The directed-travel (technological)
   destination list ranks **5,490 reachable known-planet hosts**, top rocky-HZ
   targets HD 216520, HD 210277, HD 215497… (all temperate-planet hosts, reachable
   in 400–1,200 yr at 0.1c). Net: passive panspermia closed; directed-travel gives
   a concrete ranked target list.
3. **`crosscorr`** — high-res Doppler cross-correlation (O₂ A-band + H₂O, Kp–Vsys).
   **Run: `NO_ARCHIVAL_IN_TRANSIT_HIRES_SPECTRA_AVAILABLE`** — ESO archive reachable
   (118 ESPRESSO records) but **0 in-transit transmission sequences** (all
   out-of-transit RV monitoring); one LHS 1140 b transit sweeps Kp by only ~0.9 km/s,
   so a real search needs many stacked dedicated transits. Engine validated on
   injection; honest data-gap verdict.
4. **`seti_archive`** — targeted radio/optical SETI coverage + EIRP limits.
   **Run: `NO_TARGETED_RADIO_SETI_ON_RECORD`** — a genuine observational gap on a
   landmark HZ world; representative limits show a modest MeerKAT/GBT/Parkes pointing
   would constrain narrowband beacons to ~2–12×10¹⁰ W (~10³× below the Arecibo
   planetary radar, well sub-Kardashev-I). Deliverable is the coverage+limit map.
5. **`iso`** — interstellar-object back-tracking ('Oumuamua/Borisov/3I) through the
   Galactic potential. **Run: clean null** — all three ISOs stay at LHS 1140's
   present ~15 pc under back-integration (`d_min_p50 ≈ 14.96 pc`, `t_enc ≈ 0`),
   `any_consistent_with_origin=False`. The necessary-not-sufficient caveat ships as a
   first-class field (degree-scale radiants smear parsecs; apex projection; disk
   prior). None traces back to LHS 1140.

**Net across the fan-out:** no bio or techno detection; the two live scientific
outputs are the **directed-travel destination list** (a ranked answer to "which
rocky-HZ worlds could an LHS 1140 biosphere reach") and two identified **real
observational gaps** (no dedicated in-transit high-res spectra; no targeted radio
SETI) on a landmark world. The one channel that would give a *positive* atmospheric
measurement — `jwst_bio` on the actual JWST spectra — is re-running after the TSO
filter fix.

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
| **OSSUARY** (warm dust where none can form) | built; first run dispatched (run 30203264572). Gaia DR3 GSP-Spec/GSP-Phot [Fe/H] < −1 **plus** a pure halo-kinematic track, × the Gaia archive's AllWISE + 2MASS mirrors via the official PM-aware cross-match; expected ~10⁵–10⁶ stars | pending first run | **Novelty confirmed:** two independent full-corpus arXiv queries for halo-star IR excess return **0**; `"metal-poor" AND "debris disk"` returns 3 papers, one of them a 7-star study at [Fe/H] ≲ −5 with inverted motivation (Venn+2014). Hephaistos I hard-codes Z = 0.012–0.018 = thin disc. Lacki 2025 (arXiv:2504.21151) *predicts* halo + low-metallicity hosts and ran no search. Competitor forming: Kenyon, Bromley & Najita 2026 have the catalogue and *plan* the analysis |
| **CENOTAPH** (cold Dyson, T<100 K) | built + dispatched (run 30203250183); target Gaia DR3 GSP-Spec dwarfs, ~5.6e6 with Teff/logg/[M/H]/[α/Fe] from one pipeline | pending first archive run | **new channel.** Three-leg energy-conservation test: grey attenuation (A_V fitted jointly, not assumed) + NO mid-IR excess + far-IR recovery of the intercepted f·L in AKARI/FIS + IRAS. Closure ratio ρ=f_IR/f_dim separates an isotropic occulter (ρ≈1) from an edge-on disk (ρ≪1). Measured floor f≳0.15–0.29 vs Zackrisson+2018's f_cov>0.75. Next: read `results/cenotaph/summary.json`, check the ±3σ tail asymmetry before believing any count |
| **SHROUD** (enshrouded, not destroyed; S33) | built; first run dispatched. The **never-analysed catalogue by-product** of Solano, Villarroel & Rodrigo 2022 (MNRAS 515, 1380): `vanish-neowise` = **171 753** POSS-I sources absent in the modern optical but detected in the infrared, plus `vanish-possi` = **5 399** with no counterpart at all, as the control | pending first run | **New channel.** Endpoints verified: `http://svocats.cab.inta-csic.es/vanish-{neowise,possi}/` (quoted in Watters+2026 Table 1 and the jannefi/vasco README). Solano+2022 is **not** in VizieR (runner fetch returns "Table or Catalog not found") — SVO is the only machine-readable route. Measurement: eta = F_IR(now)/[F_bol(POSS-I) − F_bol(modern)], a **pure flux ratio so distance cancels** — no parallax needed. Forés-Toribio & Kochanek 2026's progenitor/remnant ratio *is* eta, applied at ≥10 (merger remnant) and ≤0.3 (genuine disappearance). Scoped **strictly to the archival crossmatch** — no VASCO transient/Earth-shadow/nuclear-test analysis (docs/shroud.md §1) |
| **EMBER** (waste heat that switched off; S1) | built; first run dispatched. The only three all-sky surveys carrying 12–25 µm photometry — IRAS (1983), AKARI/IRC (2006–07), WISE cryogenic (2010) — cross-matched via Gaia DR3 with PM propagated to each survey epoch. Working sample of order 10⁵ (bright, large-excess regime set by the shallow early epoch) | pending first run | **New channel.** *Novelty:* the entire Dyson/waste-heat lineage is single-epoch by word count — "epoch" in a photometric sense = **0** in Carrigan 2009, Ĝ I/II/III and Hephaistos I; the 2026 flagship review (2605.21093) has `turn off`/`switched off`/`cessation`/`multi-epoch`/`AKARI`/`NEOWISE` all = 0; Suazo+2024 *explicitly cut* variable stars, discarding a changed megastructure by construction. **Antecedents that must be cited, not ignored:** Kim+2015 (1501.05721) ran the identical IRAS+AKARI+WISE comparison **upward** (4 sources all-sky); Sedgwick & Serjeant 2022 (2207.09985) built the IRAS×AKARI 23.4-yr cross-match for *proper motion*; Melis+2023 ran it downward but targeted at R CrB. **Audit reversed the brief:** AKARI 9 µm→W3 has a transfer spread of **5.18×** over 150–1500 K (worst-conditioned pair, demoted); **IRAS 12 µm→W3 is spread 1.20 — near-null — and I25→W4 is 1.24 with an 8× wider unsaturated window**. NEOWISE **rejected as an epoch** (W1/W2 only; cannot see 100–300 K dust) and reused as the post-drop flatness test. Next: read `results/ember/pair_audit.json` and the rising-tail asymmetry in `null_calibration.json` before believing any count |
| **TAILINGS** (the sparse chemical anomaly; S12/S15/S22) | built; first dispatch hit stale VizieR catalogue numbers and returned `NO_DATA_REACHED` honestly, fixed by runtime TAP_SCHEMA table discovery + schema scoring; re-dispatched. Target GALAH DR4 + APOGEE DR17 cool dwarfs (Teff<6000 K, logg>4.0), ~20–30 elements each | pending first archive run | **New channel, and the discriminant is inverted.** Natural abundance space is low-dimensional (~8–10 independent axes: Ting+2012, Price-Jones & Bovy 2018, Patil+2022) and every natural process moves an element FAMILY; refining moves ONE. So a **dense** anomaly is a REJECTION here — the opposite polarity to every existing abundance-outlier statistic, all of which are global distances (PCA/EMPCA/t-SNE/autoencoder/k-means) built to *cluster* stars, and all of which are maximised by dense anomalies. Direct arXiv queries for `"anomaly detection" AND APOGEE`, `… AND GALAH`, `"outlier detection" AND "stellar abundances"`, `"abundance anomaly" AND "single element"` all return **0**. **The real competitor, read in full: Huang, Tao & Zhang 2026 (arXiv:2605.29811)** — executed, meteorite-calibrated Bayesian test for refined material, but in **polluted white dwarfs** (7.7≤logg≤8.3; "GALAH" and "APOGEE" appear 0 times), on 697 literature records/≥397 objects, against a **fixed dense siderophile template**, and its power *rises* with element count — "typically requires ≳5 detected elements for decisive support". The two searches rank the same objects in nearly opposite order. Their strongest criticism (a 1–2-element anomaly can carry a big Bayes factor while information-starved) is answered by `n_quiet`: their cases are records where the other elements were never MEASURED; here 20–30 are measured **and quiet within 2σ**. Whitmire & Wright 1980's 56-citation tree still contains **no executed survey**; note this channel deliberately inverts their A5–F2 host choice — that band is where diffusion/levitation manufacture single-element anomalies naturally (hence 60 years of contested Przybylski claims), so trading amplitude for a clean null is the right trade. Next: read `results/tailings/contrast_*.csv` — if the SPARSE fraction is flat with z_max the sample is systematics, not a population |
| **RUST** (unmaintained decay; S9) | built; first run dispatched (**run 30203976309**, 8 high-galactic-latitude ZTF fields, g+r paired). Statistic: bias-corrected **season scatter** regressed on calendar time — the **second moment**, not the first | pending first archive run | **New channel, and it is NOT the `dimming` channel again.** `dimming/secular.py` fits a weighted line to season **medians** (brightness, first moment) and is exhausted at the ZTF systematics floor; RUST regresses per-season robust **scatter**. The distinction is structural, not rhetorical: a shared zeropoint/reference drift moves every star's median together — the false fade that killed `dimming` — and leaves the *within-season scatter* untouched. But the exchange is not free, and RUST inherits two systematics `dimming` never had. **(a) Cadence bias.** `1.4826×MAD` recovers only ~66% of σ at N=3 and ~90% at N=8; epochs-per-season is set by survey cadence and cadence *trends with calendar time* (ZTF public went 3-day → 2-day in 2020). Uncorrected, this channel measures ZTF's operations calendar. Handled in five layers: per-season null computed with **that season's own N and its own per-epoch error vector** (MC bias table b(N) + heteroscedastic mixture-MAD scale); excess variance subtracted not divided, **including the second-order `E[s²]=(bσ)²(1+u²)` term** — a 17% offset at N=8 vs 2% at N=70, i.e. a pure function of cadence, hence of time; a **distribution-free** exact permutation p-value over all n! season orderings as the primary gate (Lacki gives a cascade *timescale*, not a light-curve shape, so a linear model would assume the result); per-CCD ensemble common mode in the second moment; and survivor-only exact-MC + **equal-N subsampling** cross-checks that share no machinery with the first layer. **(b) `magerr` mis-calibration drifting with time** — killed by the ensemble κ_s = median(σ_obs²/σ_null²) over the field. Measured offline: **0/600** constant stars flag across five cadence histories (rising 8→70, falling 70→8, ZTF-style jump, erratic, doubling), 0/2000 over a wider sweep; ~57% recovery at 36 mmag terminal amplitude, ~80% at 60 mmag. **Two-band g/r coincidence is enforced at SCORING, not follow-up** — a one-band source is never scored — and the g/r amplitude-growth *ratio* is the physics discriminant (grey occulter ≈1.00 / extinction law ≈1.42 / flare-accretion >1.70). **NEOWISE logic is INVERTED relative to `dimming`:** a mid-IR brightening *corroborates* dust production from a cascade instead of killing the candidate; its absence is informative but not fatal (W1/W2 only probe >600–850 K). *Novelty:* a regex concept scan over 3,578 fetched abstracts for "amplitude/scatter increasing with **calendar time**" returns **0**; every apparent hit is a different statistic — Polaris (a single named star), T Tauri in WASP (amplitude vs **timescale**, red noise), YSOs in W51 (vs **evolutionary stage**), NLSy1 (vs **wavelength**). Nearest machinery is Petz & Kochanek 2025 (2501.14058), 9,361,613 ASAS-SN sources at >0.03 mag/yr — a **mean-flux slope**; the second moment is untouched. Bonus seam: Hephaistos II's `G_var > 2` cut explicitly discards "potential Dyson swarms with very large absorbing elements", i.e. exactly what this channel selects on. Sensitivity is stated *against* the counterweight: **not** sensitive to McInnes 2026 (2603.00203) passively-stable ring engines / dense-cloud bubbles, which need no upkeep, nor to Wright 2020 (2006.16734) monolithic spheres, whose failure is catastrophic rather than a decade-long ramp. Next: read `results/rust/summary.json` and `results/rustlit/concept_scan.json`; for any survivor demand the ASAS-SN cross-survey result — **red noise, not white, is the least-controlled residual** and the quoted false-positive rates are against Gaussian nulls |

## Known systematics ledger (do not re-derive)

* **AllWISE infrared excesses are ~92% false positives.** Silverberg et al. 2018:
  at most **7.9 % ± 0.2 %** of AllWISE-selected excesses are good disk candidates;
  the McDonald et al. and Marton et al. searches exceed **70 %** false positives;
  **all 13** Theissen & West candidates with W4 S/N > 3 are spurious. Any new
  excess funnel must report *per-stage removals*, not just a final count.
* **Use 5σ, not 3σ, for an infrared excess.** Huang, Liu, Wyatt & Kennedy 2025
  (arXiv:2505.07602) searched the 10 pc sample (339 stars) for W3 excess at 3σ,
  got 5 candidates, and found **all five spurious**; detection rate 0/339.
* **W3/W4 depth is frozen at the 2010 cryogenic mission.** NEOWISE-R, CatWISE2020
  and the deep unWISE coadds are **W1/W2 only**. Wien peaks: W1 → 852 K, W2 →
  630 K, W3 → 241 K, W4 → 132 K. So below ~200 K the *only* route is W4, the
  shallowest and most confusion-limited band — there is no deeper 12/22 µm
  measurement to be had, and a warm-dust claim must lean on **W3 + W1−W2**.
* **λ Bootis stars are the metal-poor IR-excess trap.** A/early-F stars whose
  *surface* is metal-depleted by accreting gas-depleted ISM; Murphy et al. 2020
  find **21 of 34 have infrared excesses**, and some were previously catalogued
  as blue horizontal branch stars. A T_eff ceiling (~6500 K) removes them.
* **Globular-cluster sightlines must be vetoed, not vetted.** Boyer et al. 2010
  (arXiv:1002.1348) showed a *published* RGB-wide infrared excess across 47 Tuc —
  a metal-poor, old population — was entirely stellar blending and imaging
  artefacts, from the same archival imagery as the original claim.
* **Metal-poor circumstellar dust is featureless** (metallic iron, not silicate;
  McDonald et al. 2011, ω Cen). No mineralogy argument can discriminate it.
* **The natural warm-dust background vs age is measured:** Kennedy & Wyatt 2013,
  12 µm over 24,174 Hipparcos MS stars within 150 pc — old (>Gyr) dusty systems
  occur at **1 in 10⁴**, young (<120 Myr) at **~1 %**. Fractional luminosity
  decays as ~1/age² (Pawellek+2021 observed; Wyatt+2011 theoretical).
* **The natural background vs metallicity:** Gáspár, Rieke & Ballering 2016 —
  *"disk-bearing stars seldom have metallicities less than [Fe/H] = −0.2"* over
  662 disks. Planet occurrence ∝ 10^(2[Fe/H]) (Wyatt, Clarke & Greaves 2007).
* **Background galaxies killed the entire warm-Dyson candidate list.** JWST/MIRI
  resolved Hephaistos D and E into a z≈0.9 Hot DOG and a z≈0.4 dusty starburst,
  both within ~1″ (arXiv:2607.09460); Hot DOGs at ~9×10⁻⁶ arcsec⁻² can account
  for all seven. High |b| helps against **cirrus and stellar blends only** — it
  does *not* reduce extragalactic confusion. Only sub-arcsecond astrometric
  registration at the *propagated* epoch plus a chance-superposition prior does.

* **Deeper WISE data is *anti-correlated* with colder sensitivity.** Wien-peak
  temperatures: W1 3.35 µm → 865 K, W2 4.60 µm → 630 K, W3 11.56 µm → 251 K,
  W4 22.09 µm → 131 K. Every WISE catalogue that got deeper after 2010
  (NEOWISE-R, CatWISE2020, unWISE) is **W1/W2 only**; W3/W4 depth is frozen at
  the 2010 cryogenic mission. So the largest waste-heat searches ever run are
  structurally incapable of reaching 100–300 K, and improving them makes them
  *warmer*. Below ~130 K the mid-IR route is closed by instrumentation, not by
  effort. Do not propose "go deeper in WISE" as a route to cold Dyson spheres.
* **A parallax error is exactly a grey offset**, and any twin/reference-star
  scatter is common-mode across bands. Both must enter a multi-band fit as a
  **rank-1 fully correlated** covariance term, never on the diagonal — treating
  them as independent per band inflates every significance by ~√N_bands.
* **Scatter about a reference-star median is not the error bar.** It also
  contains the parameter gradient across the matching box (measured: 0.14 mag
  instead of 0.05 for a Teff box of ±150 K). Take the scatter about a *local
  linear fit* in parameter space instead.
* **A published IR-excess catalogue is ~92% false positive** (Silverberg et al.
  2018: at most 7.9%±0.2% of AllWISE-selected excesses are good disk
  candidates; all 13 Theissen & West W4 S/N>3 candidates are false). Measure an
  excess from the photometry; never inherit one.
* **Far-IR beams make background-galaxy confusion far worse than in WISE.**
  IRAS/AKARI beams are 25–180″ vs WISE's 6–12″, so the coincidence area is
  10²–10⁴× larger; with 10⁶ targets the chance-match expectation runs to
  thousands. Measured Gaia source density (results/farir_stats): 101,853/deg² at
  |b|<5° vs 3,119/deg² at |b|>60°. A far-IR positional association is never
  evidence on its own.

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

* **SHROUD / VASCO sample (2026-07-26).** (1) A **plate defect has no infrared
  counterpart** — requiring a real IR detection is itself a strong artifact
  filter, applied at selection time. The residual worry is the opposite: a
  defect landing by chance within 5" of an unrelated IR source. (2) At the
  published **5" radius the chance-match probability is ~0.9% at high galactic
  latitude, ~10% against AllWISE all-sky, ~24% against CatWISE2020, and →1 in
  the plane** — of order 10^4–10^5 of the 172,163 "counterparts" may be
  coincidences. Watters+2026 leave this "undetermined"; **measure it with an
  offset-position null**, never with a uniform-random background (assuming
  uniformity is the exact error that broke the VASCO Earth-shadow analysis).
  (3) **A naive Stern+2012 `W1−W2 ≥ 0.8` AGN cut deletes the shroud population**
  — a 350 K shroud has W1−W2 = 3.2. Colour cannot separate them; **SED shape**
  can (AGN = power law, shroud = curved blackbody), and with <3 IR bands the two
  are formally undecidable. (4) `vanish-neowise` was matched to **NeoWISE, which
  carries W1/W2 only** — a 2-band IR integral badly *under*-estimates a thermal
  SED and would manufacture a spurious "IR too faint" result. Join AllWISE
  W3/W4 + 2MASS before issuing any deficit verdict. (5) Sample "S" was built by
  removing everything within 5" of **Gaia DR3 / Pan-STARRS DR2**, so modern
  optical non-detection is guaranteed by construction, not measured.
  (6) Expect **171,753** rows from the live archive, not the abstract's 172,163.

## Rules of engagement (from CLAUDE.md)

Novelty first, scale second, never write up a null. Merge every commit to
`main` as you go (non-fast-forward merge if diverged; never force-push).
Data-touching runs go through `workflow_dispatch`; the sandbox has no archive
egress.

### EMBER (cross-epoch mid-IR, IRAS/AKARI → WISE) — established 2026-07-26

* **NEOWISE cannot see waste heat.** It flies W1/W2 (3.4/4.6 µm) only; W3/W4
  exist for the 2010 cryogenic phase alone. W1/W2 reach only T ≳ 500–700 K, so
  *any* decade-baseline mid-IR excess-change search at 100–300 K is impossible
  with it. Do not propose one. NEOWISE's real value is as a **flatness** test.
* **IRAS 12 µm and WISE W3 are near-identical bandpasses**: the early→late flux
  transfer moves by only 1.20× across dust temperatures of 150–1500 K. **AKARI
  S9W → W3 moves by 5.18×** — the 9-to-12 µm step is emphatically *not* a null
  transformation and must never be treated as one.
* **The IRAS 100 µm background cut is worth a factor of ~30.** Kennedy & Wyatt
  2012 (arXiv:1207.0521): ~8,000 of 180,000 stars show an apparent IRAS excess
  correlated with the 100 µm background; below 5 MJy/sr, 271 remain. Mandatory
  for any IRAS-based excess work in this repository.
* **IRAS beam vs WISE beam is 286× in solid angle.** An IRAS flux is the sum
  over its footprint, so the only defensible comparison sums *all* late-epoch
  sources in the early beam. Comparing against the nearest counterpart alone
  fabricates fades wherever the field is crowded.
* **W3 saturates at ≈0.96 Jy, barely above the IRAS PSC completeness limit of
  0.4 Jy.** Bright IRAS sources are exactly the ones WISE cannot measure, and a
  saturated late band under-reports flux and mimics a cessation. Use I25→W4
  (saturates at 12 Jy) or let AKARI arbitrate.
* **Eddington/Malmquist bias is one-directional and fades only.** A flux-limited
  early epoch plus a deeper late epoch manufactures cessations with no
  astrophysical change. A two-sided null cannot calibrate it; deboost explicitly
  and impose an early-epoch S/N floor.
* **A blackbody at T_eff is not a stellar atmosphere.** Extrapolating 2MASS Ks to
  12 µm with a Planck function over-predicts by ~0.3 mag at 5000 K. Use an
  *empirical* per-band colour locus — it also absorbs each survey's calibration
  scale, which is what Liu 2020 attributed IRAS–WISE discrepancies to.
* **Fit the photospheric locus on low quantiles, not the median.** In an
  IR-selected catalogue the excess population can exceed 50%, which is exactly
  the median's breakdown point.
* **The published cross-epoch stability floor is 4%** (HD 172555, IRAS 1983 →
  WISE 2010, arXiv:1210.6258). Nothing within a few times that is believable,
  however significant.
* **Every natural mid-IR variable class varies *persistently*** — 14 of 17
  extreme debris disks changed at 3–5 µm between 2010 and 2019. A single
  monotonic step followed by a flat decade is the discriminant. TYC 8241 2652 1
  is the sole known step-and-stay object and is still unexplained.
* **Smooth disc dispersal cannot make this signal**: τ = 2–3 Myr at 3.4–12 µm is
  a ~10⁻⁵ change over 27 years. Only discrete events can.
