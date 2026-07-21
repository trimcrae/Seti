"""Long-baseline (hundreds of Myr) stellar-encounter search via orbit integration.

The panspermia channel finds close stellar encounters with a *linear* (constant-
velocity) treatment, which is honest only over the recent ~10 Myr where the
Galactic tide is negligible.  Extending the search to the past few hundred Myr --
"which stars ever passed near LHS 1140 or K2-18" -- **requires integrating each
star's orbit in the Galactic potential**, because differential rotation and the
vertical tide bend every trajectory well within that baseline.

This package provides a compact, vectorised axisymmetric Milky Way potential and
a leapfrog integrator (``orbits``), a closest-approach finder over the integrated
trajectories with Monte-Carlo uncertainty (``encounters``), and honest reporting
of the lookback time beyond which phase mixing makes the encounter timing
unrecoverable.  The pure-dynamics pieces are unit-tested offline (circular-orbit
stability, energy conservation, a recovered synthetic encounter).
"""
