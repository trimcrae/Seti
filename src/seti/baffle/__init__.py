"""BAFFLE — reciprocal mid-infrared absorbing screens on Sun–star lines.

A warden that hides the solar system's technosignatures from other stars with
band-selective absorbers at 10²–10⁴ AU would, by reciprocity, dim those stars'
mid-IR *toward us* while passing their optical / near-IR light.  The observable
is a star whose W1 **and** W2 fluxes sit significantly below the photospheric
locus set by its own (J−Ks) colour, with normal Ks — a signature no natural
foreground produces (interstellar extinction makes the IR relatively brighter).

Modules
-------
``acquire``  runner-only Gaia-archive pulls (deficit + missing tracks), chunked,
             checkpointed, ledgered; every query builder is a pure string
             function so it is unit-testable offline.
``locus``    the empirical photospheric locus Ks−W_b vs (J−Ks) and residuals.
``screen``   selection, the named vetoes, ETZ / nearby flags, verdict tokens.
``run``      stage orchestration -> ``results/baffle/``.
``patch``, ``radio`` (owned by other agents) are imported lazily by ``run``.
"""

from __future__ import annotations

__all__ = ["acquire", "locus", "run", "screen"]
