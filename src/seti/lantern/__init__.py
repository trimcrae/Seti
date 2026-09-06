"""LANTERN -- a narrow emission line that vanishes at secondary eclipse.

Across every public JWST time-series observation of a transiting exoplanet, is
there an unresolved (instrument-resolution-limited) emission feature that is
present outside secondary eclipse and ABSENT while the planet is hidden behind
its star?  A monochromatic source on the planet is the only thing that produces
a narrow line whose flux tracks the planet's visibility: stellar lines,
instrumental artefacts and detector features do not care whether the planet is
occulted.  Transit is the second phase reference -- a planet-origin line is
constant in-vs-out of transit -- and the two together rule out every non-planet
origin.

Modules
-------
``phase``    ephemeris propagation, contact times, per-integration phase labels
``line``     narrow-feature search, per-integration line flux, the eclipse
             discriminant, vetoes, tiers, BH-FDR (pure NumPy, offline-tested)
``synth``    synthetic spectral time series for the tests and the runner gate
``acquire``  NASA Exoplanet Archive + MAST (runner-only; reuses jwst_bio's path)
``run``      stages ``probe`` / ``inventory`` / ``screen`` / ``assess``

The in-house precedent is the one-planet, out-of-transit-only laser scan in
:mod:`seti.jwst_bio`; this channel generalises it to every planet with a public
JWST time series and adds the eclipse-vanishing discriminant.
"""

from __future__ import annotations

__all__: list[str] = []
