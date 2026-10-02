# Fixed-cadence synthetic phasing control

AI-assisted implementation by OpenAI Codex; independent review and actual test receipts will be recorded in the sprint handoff.

This read-only diagnostic conditions on the pinned Gaia DR3 fixture for source `1457486023639239296` at main `700be390a9cdb9d9af68e0679cb128b3a212167b`. The fixture is the existing 2026-09-22 public redistribution described in its README, not a new direct ESA acquisition. It retains BARYCENTER/TCB days from JD 2455197.5 and uses the production G quality mask; variability-rejection flags remain diagnostic.

The predeclared synthetic ephemeris is P=1 day, duration=0.1 day, t0=0 Gaia days, phase tolerance=0.03, k=10. Unique usable observing epochs are grouped by adjacent gap <= the existing one-day gap. Each representative is the first observed epoch. These are sampling proxies, not recovered dip peaks or independent physical episodes.

One null assigns ten labels uniformly without replacement among those fixed representatives. Exact three-alias inclusion-exclusion counts subsets whose ten labels all match a single P, 2P or P/2 window. The other null shifts the first ten chronological representatives by one common delta uniform over [0,2P), preserving relative cadence; exact interval intersections and their alias union integrate that model. The chronological selection does not depend on phase.

The positive deliberately selects the first ten P/2-matching representatives. It tests phase plumbing with a fixed synthetic BLS result, not photometric recovery, BLS fitting or independent validation. Its separate common-shift sensitivity and a one-epoch repeated-slot toy illustrate conditioning/correlation; neither replaces the main null. Insufficient usable/matching opportunities produce an explicit refusal rather than parameter tuning.

The comparison with the unchanged independent-uniform-phase bound is conditional mathematics. It does not estimate a sky false-positive rate, establish calibration or a detection, or alter any runtime classification, scientific threshold, preregistration or historical result. Small finite-set enumeration, independent time-distance quadrature, wrap/intersection/overlap controls and real-fixture positive/negative tests validate the diagnostic.

Run `python -m seti.parallax4.cadence_control` to print the canonical report and its SHA256 without writing science outputs. Actual execution is performed by the existing GitHub Actions environment; local execution has not been claimed.
