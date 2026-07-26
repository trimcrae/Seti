"""SHROUD — enshrouded, not destroyed.

Signature **S33** of ``docs/necrosignatures.md``: an object present in the
POSS-I red plates (1949-1958) and absent from every modern optical survey, but
**detected in the infrared**, has not been destroyed — it has been *enshrouded*.
That is the completed-Dyson / dead-swarm-dust endpoint.  The companion sample of
optically-vanished sources with **no** counterpart at any wavelength is the
control, and the ratio of the two populations is the obscuration-vs-destruction
measurement.

The channel analyses the *catalogue by-product* of Solano, Villarroel & Rodrigo
2022 (MNRAS 515, 1380): 172 163 sources "not detected in the optical but
identified in the infrared regime", published to a VO archive and never
analysed.  It deliberately does **not** touch the contested VASCO transient /
Earth-shadow / nuclear-test analyses — see the scoping statement in
``docs/shroud.md``.

Modules
-------
``acquire``   runner-only archive access (SVO vocats, VizieR, CDS X-Match)
``sed``       photometric system, two competing SED models, the energy budget
``classify``  the mundane-population decomposition that must be subtracted
``vet``       the kill-tests (epoch propagation, chance match, ledger rules)
``run``       stage orchestration; writes ``results/shroud/``
"""

from __future__ import annotations

__all__ = ["acquire", "classify", "run", "sed", "vet"]
