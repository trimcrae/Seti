"""VIGIL --- waste heat with a duty cycle.

The claim (necrosignature **S4**): a computing megastructure's waste heat tracks
its computational load, so the observable is a star that is **variable in the
mid-infrared while constant in the optical**.  The intercepted starlight is
steady --- the covering fraction does not change --- but the *re-radiated*
thermal output does.  The extinction-relevant reading of the same channel is
that variability **ceasing**.

Why this is not a Dyson search with extra steps
-----------------------------------------------
Hephaistos II's ``G_var > 2`` cut explicitly *rejects* variable stars, on the
stated grounds that a swarm with very large absorbing elements "could generate
detectable variations in the photometry of the host star".  The variability this
channel selects on is what the flagship Dyson search throws away --- and it is a
different variability: theirs is *optical* (occultation by the absorber), ours is
*mid-infrared only* (modulation of the re-emission).

The confounder is the whole problem
-----------------------------------
Extreme debris disks (EDDs) have exactly this phenomenology: strong mid-IR
variability with flat optical light curves (Moor et al. 2021, arXiv:2103.00568).
They are a mature, actively studied class and they dominate any naive selection.
The discriminator is in :mod:`seti.vigil.discriminate` and it is quantitative:
an EDD's mid-IR variability rides on a **large** fractional excess (f ~ 1e-2 ---
that is what makes it "extreme"), and only a small fraction of that excess is
modulated.  A duty-cycled radiator varies by essentially *all* of its excess.
So the primary axis is not amplitude and it is not excess: it is the
**modulation index** ``m = A_observed / A_max(tau)``, the fraction of the
inferred excess that is actually switching.

The instrumental bound
----------------------
NEOWISE, CatWISE2020 and the deep unWISE coadds are **W1/W2 only**.  Wien peaks:
W1 3.4 um -> 852 K, W2 4.6 um -> 630 K.  This channel therefore probes **hot**
material and is structurally blind to the 100-300 K regime where most Dyson
models sit.  W3/W4 depth has been frozen since the 2010 cryogenic mission.  See
``docs/vigil.md``.
"""

from __future__ import annotations

__all__ = [
    "acquire",
    "discriminate",
    "excess",
    "run",
    "variability",
    "vet",
]
