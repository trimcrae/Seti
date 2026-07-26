"""OSSUARY — warm dust around stars that cannot make it.

An unmaintained megastructure grinds itself to dust: once station-keeping fails,
a swarm is subject to a collisional cascade whose timescale is roughly the
orbital period divided by the covering fraction (Lacki 2025, "Ground to Dust",
arXiv:2504.21151, ApJ 985, 191).  The mechanism was published with **no proposed
observable and no search**.  The terminal state of a dead swarm is warm dust.

The search therefore inverts the usual debris-disk question: instead of asking
which stars have dust, it asks which stars *cannot* have dust, and then looks
for dust there anyway.  The natural null is strongest for

  * metal-poor stars ([Fe/H] < -1) -- planetesimal formation and planet
    occurrence fall steeply with metallicity;
  * halo-kinematic stars (|v - v_LSR| > 200 km/s) -- old, dynamically unrelated
    to the thin disk;
  * old stars generally -- debris-disk incidence decays steeply with age, and
    Poynting-Robertson drag plus radiation-pressure blowout clear micron grains
    in 1e4-1e6 yr, orders of magnitude shorter than the stellar age.

Modules
-------
``acquire``     runner-only archive pulls (Gaia DR3 x AllWISE x 2MASS, chunked).
``kinematics``  UVW, LSR correction, halo membership, honest missing-RV handling.
``excess``      photosphere-anchored infrared excess with propagated errors.
``vet``         the contamination gauntlet (this channel lives or dies here).
``run``         stage orchestration; writes ``results/ossuary/``.
"""

from __future__ import annotations

__all__ = ["acquire", "excess", "kinematics", "run", "vet"]
