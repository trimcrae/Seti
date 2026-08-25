"""SEXTANT --- the ephemeris residual of a known minor planet, at Gaia precision.

LOOM asks whether a *population* of artificial objects is already in the solar
system, and screens the observed-minus-predicted position of known minor planets
to find it.  Rubin serves that residual pre-computed at **arcsecond** scale.
Gaia serves the ingredients at **milliarcsecond** scale, for 46,264,083
observations of 156,823 objects (``gaiafpr.sso_observation``) plus 23,336,467 of
158,152 (``gaiadr3.sso_observation``).

The sensitivity of a non-gravitational-acceleration test scales with astrometric
precision, so a sample seven times smaller and a thousand times sharper is a
*stronger* test of the same hypothesis, not a weaker one.  And the question is
about a static population, so an archival dataset answers it as well as a live
one --- which is what makes this more than a stopgap for a telescope that is
currently snowed in (``docs/rubin-outage.md``).

See ``docs/substitute-surveys.md`` for the measured column inventory, and for the
two traps this channel is designed around: residuals taken against Gaia's own
orbit solutions are minimised by construction and mean nothing, and an unresolved
binary's photocentre wobble is exactly the anomaly being searched for.
"""
