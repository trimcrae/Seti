"""KNELL --- the clock that stopped.

Signature **S32** (:doc:`docs/necrosignatures.md`): *periodicity that ceased*.

A periodic signal that stops is the cleanest "the mechanism ended" observable
available in the optical time domain.  Eclipsing binaries do not stop; pulsators
do not stop; and --- unlike a *fade* --- a cessation cannot be manufactured by
intervening dust, because obscuration changes a signal's **amplitude relative to
the continuum**, not the **existence of a period**.

The whole channel is organised around one adversarial fact:

    The dominant false positive is not astrophysics, it is
    **survey-dependent detectability**.  A variable "ceases" whenever the later
    data have worse cadence, a shorter window, larger errors, a different
    aliasing structure, or a different passband.

So the primary search is **intra-survey** (ZTF's own early seasons against its
own late seasons, in both g and r), where passband, pipeline and calibration are
constant by construction, and every claimed non-detection is scored against the
**injection-measured detection efficiency for that star's own period and
amplitude in that block's own sampling and noise** (:mod:`seti.knell.efficiency`).
Without that normalisation the statistic measures the survey's operations
calendar, exactly as the sibling RUST channel found for the second moment.

Modules
-------
``blocks``      epoch blocking, batched Lomb-Scargle, PDM cross-check
``efficiency``  injection-measured per-block detection efficiency  [load-bearing]
``cease``       the cessation statistic and its vetting flags
``vet``         the contamination gauntlet (pure functions)
``acquire``     runner-only ZTF / VSX / Gaia / SIMBAD access
``run``         stage orchestration -> ``results/knell/``
"""

from __future__ import annotations

__all__ = ["blocks", "cease", "efficiency", "vet"]
