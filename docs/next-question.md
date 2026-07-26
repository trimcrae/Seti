# Next question — proposal after three adjudicated nulls (2026-07-26)

Closed with verdicts: HERDSMAN-A (no cold, persistent N>=4 rendezvous among
1.26M precision-6D stars to 2 kpc), HERDSMAN-B (no chemically "impossible"
bound clusters; spectroscopic tier 0/80), MIDDEN (no photospheric
radionuclides; the one flag adjudicated as roAp blending). Per the charter a
null changes the question. The residue principle that generated those
channels — whatever the Fermi solution, technology conserves **mass**,
**momentum**, **history (isotopes)**, and **shared geometry** — has one
residue class still untouched by us or (to current knowledge) anyone:
**shared geometry at population scale**.

## Lead candidate: COMPASS — orbital-pole coherence patches in Gaia DR3 NSS

**Premise.** Nature orients binary orbital poles essentially isotropically
(small-sample studies, ~dozens of visual binaries, are consistent with
isotropy; a weak Galactic-plane alignment has been debated and never
established). Engineering does not: standardized construction, torquing for
energy harvesting, or communication-geometry optimization would leave
*spatially coherent patches* of aligned orbital poles among physically
unrelated neighbouring systems. Alignment is cheap to maintain and survives
every Fermi solution that leaves artifacts in place — it is a geometry
residue, not a luminosity signature.

**Data, at scale.** Gaia DR3 non-single-star (NSS) astrometric +
astrometric-spectroscopic orbits: ~1.7e5 systems with inclination i and
node angle Omega — the first catalogue in history large enough to test
pole-field *coherence* rather than global isotropy. Nobody has run a
technosignature-framed coherence scan on it (to be verified by litcheck
before any compute: search terms "binary orbital pole alignment", "spin-orbit
alignment wide binaries", Agati 2015, Hwang 2022 eccentricities, DR3 NSS
papers).

**Detector sketch.** Unit pole vectors from (i, Omega) with the intrinsic
astrometric ambiguities handled honestly (axial statistics: poles live on the
projective sphere — use orientation tensors / Bingham statistics, not vector
means). Scan spheres of 25-100 pc for excess pole concentration versus the
axial-isotropic null; significance from scanning-law-matched mocks.

**Contamination ledger (the channel lives or dies here).**
1. The Gaia scanning law imprints sky-position-dependent biases on NSS
   inclinations — the dominant systematic. Null mocks must shuffle poles
   only among systems with similar ecliptic latitude / scan coverage.
2. Star-forming relics: co-natal wide binaries in a dissolving group share
   angular momentum from their parent cloud — a *real* astrophysical
   alignment. Discriminator: co-natal systems share metallicity and velocity;
   an engineered patch spans field stars that share neither (the same
   heterogeneous-chemistry discriminator HERDSMAN-B used).
3. i vs 180-i and Omega mod 180 degeneracies collapse the statistics onto
   axes; any method that pretends full vectors are known is wrong by
   construction.
4. Orbit-quality cuts (significance of i, period coverage) correlate with
   brightness and separation; mocks must preserve the cut structure.

**Why it fits the charter.** Novel question (coherence, not isotropy; NSS
scale is new), catalog-scale on runners (one TAP pull, KD-tree + axial
statistics), and it inherits the project's proven discipline: self-limiting
nulls, shuffle mocks, chemistry discriminator, no-null-writeup.

## Alternates considered

- **KEEL** — station-keeping anomaly: chemically old ([alpha/Fe]-high) field
  stars on dynamically pristine (zero-eccentricity, zero-vertical-action)
  orbits, i.e. stars that "should" be kinematically heated but are not —
  orbit maintenance as a mass/momentum residue. Strong idea, but adjacent to
  the phase-space-crystallization prior art (arXiv:2605.06072) flagged in the
  HERDSMAN lit check; novelty risk is real.
- **CACHE** — exo-Trojan census at planetary L4/L5 in TESS phase curves
  ("lurkers" park at Lagrange points). Prior art exists (exo-Trojan searches
  by Hippke, Janson; Benford's lurker programme for the Earth-Moon system) —
  a refinement of an existing search, which the charter ranks below a new
  question.

## Proposed sequence (pending user steer)

1. litcheck workflow run on COMPASS terms (verbatim-abstract verification, as
   for HERDSMAN).
2. If clean: NSS pull + axial-statistics detector + scanning-law mocks,
   staged/checkpointed like herdsman.yml, chemistry vet for survivors.
