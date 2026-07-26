# HERDSMAN — kinematic technosignatures: N-star orbital convergences in Gaia DR3

**Claim being tested.** If any civilization manages stellar resources on Myr–Gyr
timescales (Shkadov/Caplan-class stellar engines; Hooper 2018's
gather-stars-against-dark-energy argument; Ćirković-style stellar husbandry),
then somewhere there are groups of stars **in mid-flight toward an assembly
point**. Collisionless galactic dynamics only phase-mixes and expands; every
natural channel that focuses orbits acts on **co-natal** stars (cluster birth,
traceback of moving groups, tidal-tail epicycles) or on exactly two discrete
populations (cluster–cluster collisions). A set of *unrelated* field stars
whose orbits converge to a common small volume — in the future (herding) or the
recent past (rendezvous) — is dynamically anomalous whatever its cause. Momentum
cannot be cloaked; this searches the momentum channel, which no prior
technosignature search has used.

## 1. Novelty status (literature check, 2026-07-25)

Five parallel deep literature sweeps (~150 distinct queries, ~200 abstracts),
then **verbatim verification on the Actions runner** (2026-07-26, evidence in
`results/litcheck/`: full texts of the 2026 near-neighbours, 23 arXiv-API
sweeps, Semantic Scholar / OpenAlex citation trees). Verified findings:

* Hooper 2018's complete citation tree contains no observational follow-up —
  only constraints theory (Lingam & Loeb 2020), Dyson-cloak photometry, and
  essays.
* The Vidal et al. 2026 review's stellar-engine entry (full text) lists
  detection via "high-precision astrometry (Gaia and successors)" as a
  *requirement/prospect*, names "as a pioneering observational search of
  stellar engines, Lingam and Loeb (2020) searched the Gaia catalog for
  hypervelocity stars, but without finding good candidates" (hypervelocity
  regime only), and calls Hansen 2022 "a pioneering study [that] looked for
  close stellar encounters in the solar neighborhood" (pairwise,
  target-selection framing). No convergence search exists in its ~118 pages.
* The phase-space-crystallization paper (2605.06072, full text) targets GC
  internal order only; its future-work section gestures at "forward models
  that connect hypothetical engineered phase-space structures to observable
  kinematic signatures" — i.e., the field is pointing at this gap, not
  filling it.
* Targeted arXiv API sweeps ("orbit convergence", "kinematic technosignature",
  "moved star", "phase space technosignature", "astrometric SETI", ...)
  surface no N-star convergence work under any framing.

Original sweep summary (agent-level, pre-verification):

* **Unclaimed (no proposal, no search found):** N≥3 forward-time orbital
  convergence of field stars; heterogeneous past-rendezvous searches;
  convergence statistics on Gaia 6D in any framing. The Vidal et al. 2026
  review (arXiv:2605.21093) explicitly lists Gaia astrometry of goal-directed
  stellar motion as an avenue and states no anomalous
  accelerations/speeds/clustering have been reported.
* **Prior art to cite and extend (pairwise / adjacent):**
  - Hansen & Zuckerman 2021 (AJ 161, 145, arXiv:2102.05703): civilizations
    migrate during close stellar passages. Hansen 2022 (AJ 163, 44,
    arXiv:2112.00852): catalog of 132 unbound *pairwise* close encounters
    within 100 pc as SETI targets. HERDSMAN is the N-star, full-6D-volume,
    chemistry-tagged, forward-time generalization.
  - Bailer-Jones encounter machinery (2015→2022, arXiv:2207.06258): Sun-only
    convergences of the same 33M-star sample (61 formal <1 pc approaches;
    ~1/3 of formal encounters spurious/binaries — the false-positive physics
    we inherit).
  - Grinenko & Kovaleva 2025 (arXiv:2509.00471): forward cluster–cluster
    encounter rates (35–40/Myr locally; 15/Myr age-discordant) — natural
    baseline for two-population meetings.
  - Lingam & Loeb 2020 (ApJ 905, arXiv:2009.08874): ≥0.01c stellar-engine
    abundance constraints (hypervelocity regime only).
  - 2026 genre neighbors: "Phase-Space Crystallization in Globular Clusters"
    (arXiv:2605.06072, GC internal order, null) and "Stellar J-Harvesting"
    (arXiv:2607.07781, spin technosignature, Kepler) — kinematic-technosignature
    searches began publishing in 2026; field stars/orbits remain open.
  - Kamdar et al. 2019 (ApJL 884, L42): "stars that move together were born
    together" — the co-natality null our anomaly must violate. Piatti & Malhan
    2021: one real heterogeneous rendezvous (two unrelated open clusters
    colliding) — nature's rate is nonzero and must be beaten statistically.
* **Killed design branch:** velocity-divergence *field* statistics. The local
  divergence is already measured (Oort K = −3.3 ± 0.6 km/s/kpc; Bovy 2017) and
  mapped in 3D (Nelson & Widrow 2022), and coherent ~0.3 km/s RV systematics
  fake ∇·v of 3–30 km/s/kpc on 10–100 pc scales. Convergence must therefore be
  established star-by-star in integrated orbits, never as a smoothed field.

## 2. Detector

Sample: Gaia DR3 6D within `d_max` (300 pc), `parallax_over_error > 10`,
`RUWE < 1.4`, cool RV templates only, RV zero-point corrected (Katz et al.
2023 quadratic, held constant past G_RVS = 14). Per-star scalar velocity error
`sigma_v` = RV error ⊕ 0.3 km/s astrophysical floor (gravitational redshift +
convective blueshift scatter) ⊕ tangential terms. Precision cut
`sigma_v ≤ 0.8 km/s` — sensitivity lives entirely in this subset. Resolved
co-moving pairs (<0.5 pc, Δv < 3 km/s) collapse to their brighter member.

Orbits: symplectic leapfrog in the repo's axisymmetric MW potential
(V_c(R0) ≈ 233 km/s), dt = 0.25 Myr, both time directions, detection every
0.5 Myr.

Statistic at epoch t: meeting ball `R(t) = r0 + kappa·sigma_v,ref·|t|` (a true
point-rendezvous appears exactly this large because measured velocities err by
sigma_v). Sets of ≥ `n_min` = 4 stars inside a common ball (compact: rms ≤
1.5 R), **dispersed today** (median pairwise > 20 pc; sub-pc subgroups count
once), **focusing** (rms contracts ≥ 3×), scored by Poisson surprise
`S = −log10 P(≥m | λ)` with λ from the local propagated density. Epoch
duplicates merge; persistence (number of epochs seen) is recorded — a true
co-moving assembly dwells, a chance crossing does not.

**Self-computing horizon.** Chance occupancy λ grows ∝ R(t)³; the scan stops
when typical λ hits `lambda_cap` = 0.5. The reached horizon is *measured* from
the sample's density and precision and reported (`t_horizon_myr`), never
assumed. With sigma_v ≈ 0.5 km/s and precision-subset densities this is
~10–20 Myr — comfortably inside the ~20–30 Myr orbit-coherence ceiling set by
Galactic-shear scrambling (Makarov, Olling & Teuben 2004), which is why
`t_max` = 20 Myr.

**Chance calibration.** (i) Velocity-shuffled mocks: velocities permuted among
stars within 40 pc cells — preserves spatial density and the local velocity
ellipsoid, destroys position–velocity phasing; the identical detector on each
mock absorbs the full look-elsewhere effect. (ii) Time-reversal control: the
backward scan of the *real* data is a matched astrophysical null for the
forward scan (phase-mixed dynamics is statistically T-symmetric; deliberate
future assembly is not). Global p = fraction of mocks whose best surprise beats
the observed best (add-one).

**Vetting per candidate.** (a) Ancestry: GSP-Phot metallicity spread — co-natal
(MAD < 0.05 dex) downgrades toward known-natural (dissolving group / tail
epicycle); heterogeneous (MAD > 0.12 dex or range > 0.35) is the signature.
Flag, never silently cut. (b) Rendezvous Monte Carlo: members re-propagated
under full astrometric + RV error draws; reports how point-like the meeting can
be (P(rms < 2, 5 pc)) and how well its epoch is determined. (c) Binarity
residue: RUWE and RV quality ride along in the sample cuts; DR3's
spurious-encounter experience says unresolved binaries fake single-star
velocities — any survivor gets a per-member companion screen before being
believed (second stage, existing repo machinery).

## 3. Contamination ledger (channel → discriminator → where handled)

| Natural channel | Discriminator | Handled |
|---|---|---|
| Cluster birth / moving-group traceback (backward only) | co-natal chemistry, known membership | chemistry vet; forward scan unaffected |
| Tidal-tail epicyclic clumps (Küpper; Hyades tails) | co-natal; tail geometry; velocities parallel to tail | chemistry vet + dispersed-today + focusing cuts |
| Cluster–cluster collisions (Piatti & Malhan; 15/Myr age-discordant locally) | exactly two discrete populations, both cataloged clusters | bimodality check on any survivor; mocks set base rate |
| Resonant moving groups / arches (Hercules etc.) | velocity-space sheets, not 3D points; potential-dependent | ball compactness + focusing; potential-sensitivity re-run on survivors |
| Phase spiral / breathing modes | kpc-scale coherence, vertical signature | 10–50 pc ball scale is far below; mocks inherit any residual |
| RV zero-point vs magnitude | fake brightness-correlated radial flow | Katz correction; survivor test: signal must persist in bright-only split |
| Gravitational redshift / convective blueshift | type-correlated LOS bias | 0.3 km/s floor inflates R(t); survivor test: dwarf/giant split |
| Unresolved binaries (RV wobble → spurious convergence) | RUWE, RV scatter, companion screens | sample cuts + per-survivor screen (Bailer-Jones: ~1/3 of formal encounters) |
| Chance N-star coincidence | — | Poisson score + shuffled mocks + T-reversal control |

## 4. Run protocol

Dispatch `.github/workflows/herdsman.yml` (defaults: 300 pc, G < 14.5,
sigma_v ≤ 0.8, ±20 Myr, n_min = 4, 24 mocks/direction). Results land in
`results/herdsman/` (summary.json, candidates_{forward,backward}.{json,csv},
mocks.json, REPORT.md). Offline correctness is enforced by
`tests/test_herdsman.py`, including end-to-end recovery of a synthetic 8-star
herd injected at t = +8 Myr and non-detection in the pure background.

**Interpretation ladder.** A candidate is interesting only if it survives, in
order: mock-global p < 0.05 → heterogeneous chemistry → rendezvous-MC
point-consistency → binarity screens → potential-model and bright-split
robustness. Anything surviving all five is, at minimum, a new dynamical
phenomenon (Liouville-defying coherence of unrelated stars); the artificial
hypothesis is entertained only after the natural ledger above is exhausted,
per the project's contamination-first discipline.

**No-null rule.** If the scan is empty at these settings, the result is a
*domain statement* (horizon, n_min, precision subset), not a publication: the
next moves are deeper RVs (DR4), n_min = 3 with harder vetting, and the
backward/rendezvous channel — change the question, don't write up the null.

## 5. Sister channels from the same derivation (queued)

* **MIDDEN** — survey-scale short-lived-radionuclide search (Tc/Pm/actinides
  without s-/r-process patterns) in public spectra archives; Whitmire & Wright
  1980 proposed, never executed by anyone (46-year citation tree: reviews and
  blogs only); differentiator vs the 2026 polluted-WD chemical search is the
  radioactive-decay clock and main-sequence archives.
* **MERIDIAN** — first implementation of Corbet 2003's opposition-scheduled
  SETI (concept owned by Corbet; never tested); demoted third because satellite
  glints + exposure bias mimic the signal by construction.
