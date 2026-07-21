"""Real JWST/HST transmission- and emission-spectrum biosignature analysis of
LHS 1140 b.

The companion :mod:`seti.lhs1140.biosignature` channel answered a *feasibility*
question -- given the system parameters and JWST's noise, how many transits does
each biosignature gas need?  This channel turns that budget into a *measurement*:
it acquires the actual archival JWST time-series extracted spectra (``x1dints``)
of LHS 1140 b, builds the transmission spectrum directly from the in- vs
out-of-transit integrations, and runs a battery of pure-logic detectors --
molecular feature significances, a redox-*disequilibrium* biosignature test, an
M-dwarf *abiotic false-positive* gate for any O2/O3 claim, a MIRI
secondary-eclipse brightness-temperature atmosphere-vs-bare-rock discriminant,
and a laser-line scan at the (higher-than-Gaia-XP) JWST resolution.

The scorers in :mod:`seti.jwst_bio.spectrum` are pure NumPy and unit-tested
offline; :mod:`seti.jwst_bio.run` does the runner-side MAST acquisition and
degrades honestly (records coverage, never fabricates a spectrum) when a download
is not feasible.  This is a *detection-level* pipeline -- a robust screen for a
disequilibrium signal -- not a publication-grade atmospheric retrieval.
"""

from __future__ import annotations

__all__: list[str] = []
