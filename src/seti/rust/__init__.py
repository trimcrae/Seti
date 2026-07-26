"""RUST --- structures that stopped being maintained.

A megaswarm survives only while it is actively flown.  Lacki (2025, "Ground to
Dust", arXiv:2504.21151, ApJ 985, 191) shows that megaswarms "are liable to be
subject to collisional cascades **once guidance systems start failing**", with a
collisional time of roughly *an orbital period divided by the covering fraction*
and a cascade that "can develop extremely rapidly for hypervelocity collisions".
He proposes **no photometric observable and runs no search**.

This channel supplies the observable.  A swarm losing station-keeping does not
simply get fainter --- it gets *messier*: elements de-phase, fragment, and
multiply, so the star's **aperiodic variability amplitude grows secularly** over
the survey decade.  The statistic is therefore the **second moment of the light
curve as a function of calendar time**.

Why this is not the ``dimming`` channel again
---------------------------------------------
``seti.dimming.secular`` fits a weighted linear trend to *season medians* --- a
trend in **brightness** (the first moment).  That channel is exhausted at the
ZTF systematics floor.  RUST regresses a bias-corrected **season scatter** (the
second moment) against time.  A shared zeropoint drift moves every star's median
together and is what killed ``dimming``; it does *not* change anyone's scatter.
The two statistics have different systematics, so the floor that stopped
``dimming`` does not automatically apply --- see ``docs/rust.md``.

The systematic that *does* apply
--------------------------------
Robust scatter estimators are biased low at small N, and N-per-season tracks
observing cadence, which trends with calendar time.  Uncorrected, this channel
would measure nothing but ZTF survey history.  ``scatter.py`` therefore computes
each season's **exact null expectation using that season's own epoch count and
its own per-epoch error vector**, and ``trend.py`` removes a per-CCD ensemble
common mode in the *second* moment before any star is scored.

Modules
-------
``acquire``  runner-only ZTF g+r paired light-curve pulls (sandbox egress is 403).
``scatter``  the bias-corrected season-scatter statistic (pure, offline-testable).
``trend``    regression of season scatter on time + ensemble common-mode removal.
``vet``      two-band coincidence, achromaticity, blending, known classes, NEOWISE.
``run``      stage orchestration; writes ``results/rust/``.
"""

from __future__ import annotations

__all__ = ["acquire", "run", "scatter", "trend", "vet"]
