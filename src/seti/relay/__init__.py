"""RELAY (S60): intercepting node-to-node beams by geometry.

A persisting machine network talks with beams.  Earth intercepts an A -> B
beam only when it lies inside the transmitter's cone: with the transmitter T,
the receiver R and full beam width theta, the angle at T between T->R and
T->Earth must be <= theta/2.  Two geometries satisfy it --- the far-side
*spillover* (T behind R on the same line of sight, Earth beyond R) and the
*between* case (Earth near the T--R line, R close to T's antipode).  Gaia DR3
gives every pair within 100 pc, its exact cone angle, the flux Earth receives
relative to R, and the pair's 6D kinematics; the Breakthrough Listen open-data
target list says which of those transmitters a telescope has already pointed
at; the published narrowband hit catalogues are re-cut on that geometry with
a per-pointing Doppler-drift prior.

Modules: :mod:`geometry` (pure physics, offline), :mod:`acquire` (runner-only
archive access with injectable transports), :mod:`run` (stages and verdicts).
"""
