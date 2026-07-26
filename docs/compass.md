# COMPASS — orbital-pole coherence patches in Gaia DR3 non-single-star orbits

## 1. Question

Whatever the Fermi solution, artifacts and engineered systems that persist
share **geometry**: standardized construction, torquing for energy capture, or
communication-optimized orientation would leave the orbital angular momenta of
*physically unrelated neighbouring systems* pointing the same way. Nature does
not do this: binary orbital poles in the field are consistent with isotropy in
every study to date (small samples of visual binaries), and the only natural
alignment mechanisms — inheritance from a shared parent cloud — operate solely
among *co-natal* stars, which betray themselves by shared chemistry and
kinematics.

The searchable residue: **spatially coherent patches of aligned orbital-pole
axes among field binaries that share neither chemistry nor kinematics.**

This is a new question, not a refinement: prior work tests *global isotropy*
of pole catalogues of tens of systems; nobody (litcheck pending verification)
has scanned for *local coherence* at catalogue scale, and Gaia DR3's
non-single-star (NSS) tables — the first with ~1.7e5 orbital solutions
carrying (i, Omega) — are the first data that could support it.

## 2. Data

Gaia DR3 `nss_two_body_orbit` (astrometric `Orbital*` and
`AstroSpectroSB1` solution types carry inclination and node) joined to
`gaia_source` for (ra, dec, parallax, pm, radial_velocity, mh_gspphot,
teff_gspphot, ruwe). Quality gates: parallax_over_error, significance of the
orbital solution (e.g. `a0/a0_error` or the catalogue's `significance`),
period coverage < mission span, eccentricity validity.

## 3. Detector (implemented in `src/seti/compass/axial.py`)

1. **Axes, not vectors.** Astrometric orbits leave the pole known only up to
   sign (i vs 180-i, Omega mod 180). Poles are unit AXES on the projective
   sphere; all statistics use the orientation tensor `T = mean(p p^T)`.
2. **Frame.** Pole in the plane-of-sky tangent basis from (i, Omega), rotated
   ICRS -> Galactic so patches compare across the sky.
3. **Coherence scan.** Every NSS star centres a KD-tree neighbourhood
   (R in {25, 50, 100} pc, N >= 8); per-neighbourhood Bingham statistic
   `S = 15N/2 * sum (lambda_i - 1/3)^2` (asymptotically chi^2_5 under
   isotropy). Jaccard dedup of overlapping member sets as in HERDSMAN.
4. **Null.** The asymptotic chi^2 is a sanity scale only. Significance comes
   from **ecliptic-latitude-banded shuffles**: permute pole axes among stars
   in the same scanning-coverage band (destroys real spatial coherence,
   preserves the Gaia scanning-law imprint on the (i, Omega) field), rescan,
   and compare the real maximum statistic to the shuffle-maximum
   distribution. Global p = add-one rank, exactly the HERDSMAN mock pattern.

## 4. Contamination ledger

| # | Channel | Kill |
|---|---------|------|
| 1 | Gaia scanning law imprints ecliptic-latitude structure on NSS (i, Omega) | banded shuffles ARE the null; bands narrow (5 deg) |
| 2 | Co-natal relics (dissolving groups share cloud angular momentum) | survivors must be chemically heterogeneous (mh_gspphot MAD above co-natal 0.05 dex) and velocity-incoherent (>10 km/s spread) — the HERDSMAN-B discriminator |
| 3 | Solution-type mixing (Orbital vs SB1-derived inclinations have different bias structure) | scan within solution type; cross-type patch required to agree |
| 4 | Brightness/separation selection correlating with sky position | shuffles preserve per-band magnitude structure (permutation within band leaves the band's G distribution untouched) |
| 5 | Period aliasing near 1 yr / scanning periods biasing i | veto period windows (0.95-1.05 yr and harmonics) in a robustness pass |
| 6 | A few high-weight axes dominating a neighbourhood tensor | require lambda1 dominance to survive jackknife (drop-one member, statistic persists) |
| 7 | Software/self-inflicted: stale checkpoints, truncated pulls | versioned checkpoints from day one; band pulls request explicit maxrec and split on cap (MIDDEN lesson, applied preemptively) |

## 5. Verdict ladder

1. Real max S consistent with shuffle nulls -> **null**; per policy the
   question changes (widen radii, DR4 when available), no writeup.
2. Excess patch, chemically/kinematically coherent -> co-natal relic
   (interesting astrophysics, not a technosignature; note and move on).
3. Excess patch, heterogeneous chemistry, incoherent velocities, survives
   jackknife and solution-type split -> candidate; deep-dive per-system orbit
   re-fits and literature cross-ID before any claim.

## 6. Status

- 2026-07-26: axial core implemented + tested (8 offline tests); litcheck
  novelty sweep dispatched (pass=compass). Acquisition and staged workflow
  gated on the litcheck verdict.
