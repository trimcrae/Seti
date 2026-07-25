"""HERDSMAN — a kinematic technosignature search in Gaia DR3 6D phase space.

The channel searches for *artificial stellar dynamics*: sets of N >= 4 field
stars whose orbits, integrated forward in the Galactic potential, converge into
a common small volume ("herded" stars en route to an assembly point), and the
time-reverse — chemically heterogeneous sets whose orbits emerged from a common
small volume in the recent past (a rendezvous that cannot be a birth site).

Collisionless dynamics only phase-mixes and expands; the natural channels that
focus stars (cluster birth, tidal-tail epicycles) involve co-natal stars.  A
convergent set of *unrelated* stars is dynamically anomalous whatever its cause.

Design doc with the full derivation, prior art, and contamination model:
``docs/herdsman.md``.
"""

from .convergence import ConvergenceParams, detect_convergences, propagate
from .run import herdsman_run

__all__ = ["ConvergenceParams", "detect_convergences", "propagate", "herdsman_run"]
