"""Targeted radio + optical technosignature archive search and sensitivity dossier.

This channel does for *radio/optical SETI* what :mod:`seti.lhs1140` does for the
bio/techno signature battery: it asks, for a single high-value target
(**LHS 1140**, a temperate habitable-zone world with a claimed atmosphere and a
prime Breakthrough Listen / GBT / MeerKAT / Parkes / ATA target), the two
questions nobody has assembled the answer to in one place:

1. **Coverage** -- has any radio/optical SETI campaign ever actually pointed at
   LHS 1140, in which band, and with what sensitivity?  The runner
   (:func:`seti.seti_archive.run.seti_archive_run`) queries the reachable public
   archives (Breakthrough Listen open data, NRAO/VLA, CADC ObsCore) and *degrades
   honestly* to a cited, clearly-labelled representative-facility inventory when an
   archive is down -- never fabricating an observation or a detection.
2. **Limit** -- for each facility that observed it (or, absent a real record, each
   representative GBT/MeerKAT/Parkes/ATA configuration), what minimum Equivalent
   Isotropic Radiated Power (EIRP) of a narrowband beacon would have been
   detectable?  The scorers in :mod:`seti.seti_archive.limits` turn a telescope
   System-Equivalent-Flux-Density, channel bandwidth, integration time and the
   14.96 pc distance into an EIRP limit and express it against the standard
   yardsticks (the ~2e13 W Arecibo planetary radar; a Kardashev-fraction beacon).

The deliverable is a *coverage-and-limit map*.  For this target a finding of
"no targeted radio SETI on record" is a legitimate, useful result -- it identifies
a real observational gap on a landmark habitable-zone planet -- so the honest
verdict is reported as such, not dressed up as a null-result paper.  The limit
scorers are pure-numpy and unit-tested offline; only the runner needs archive
egress.
"""
