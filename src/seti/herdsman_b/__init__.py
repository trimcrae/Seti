"""HERDSMAN-B — completed stellar assemblies: the "impossible cluster" audit.

Steady-state corollary of the herding hypothesis (docs/herdsman.md section 5):
finished collections persist and should outnumber mid-flight herds, and they
are hiding in plain sight — cataloged as ordinary clusters. Every naturally
born cluster is co-natal (chemically homogeneous at ~0.02-0.05 dex); a
gathered cluster is a bound group whose members are a random draw from the
surrounding *field* (field-like [M/H] spread on an ordinary disk orbit).
Nature's only heterogeneous compact systems are stripped dwarf nuclei, whose
stars are a *foreign coherent* population — locally-sampled-yet-bound has no
natural channel.

Census-wide chemical-coherence audit: Hunt & Reffert cluster membership x
Gaia DR3 GSP-Phot metallicities, scored per cluster as co-natal vs
field-sampled, self-calibrated against the census's own spread distribution.
"""

from .run import herdsman_b_run
from .score import score_census

__all__ = ["herdsman_b_run", "score_census"]
