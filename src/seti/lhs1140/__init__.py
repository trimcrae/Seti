"""Exhaustive signature sweep of the LHS 1140 system, its planets, and neighbours.

LHS 1140 (GJ 3053) is a nearby (14.99 pc) M4.5V dwarf hosting two transiting
planets: **LHS 1140 b**, a ~1.7 R_earth temperate rocky/water world in the
habitable zone on which an atmosphere was reported (Cadieux et al. 2024), and
**LHS 1140 c**, a hotter ~1.3 R_earth planet.  Because the planets transit, every
photometric or spectroscopic observation of the *star* is also an observation of
the *planets* -- so an exhaustive search of "any observation ever done" of the
system is, in practice, a sweep of every public archive that has looked at the
star, plus the same battery on its stellar neighbours.

This channel reuses the per-target signature detectors built for the panspermia
dossier and points them at LHS 1140, adds a **neighbour sweep** (the same
technosignature battery at catalogue scale over every Gaia source in the local
volume), and an honest **biosignature-observation inventory** (which JWST/HST
spectroscopy actually exists, since a molecular biosignature search is
transmission-spectroscopy-scale, not catalogue-scale).
"""
