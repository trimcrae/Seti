"""Interstellar-object (ISO) back-tracking channel.

Every interstellar object -- 1I/'Oumuamua, 2I/Borisov, 3I/ATLAS -- arrives on a
hyperbolic orbit whose asymptotic *incoming* velocity (its ``v_infinity`` and
radiant, the direction it came from) encodes where in the Galaxy it was moving
before the Sun's gravity bent its path.  If a technological (or biological)
civilisation at LHS 1140 -- or any nearby star -- ever ejected material, some of
that debris would drift on Galactic orbits, and an ISO passing our Solar System
*could* dynamically trace back toward its parent star.

This channel converts each ISO's radiant + speed into a heliocentric Galactic
velocity vector, integrates it **backward through the same axisymmetric Galactic
potential** used by the ``galactic`` channel (we do NOT reimplement the
integrator), and asks how close its past trajectory came to LHS 1140 -- and, as
context, to a modest sample of the nearest Gaia stars.

CRITICAL HONESTY.  Back-tracking an ISO to a specific star is fundamentally
uncertainty-limited.  Even 'Oumuamua's origin cannot be pinned: its radiant sits
near the solar apex, so *many* stars lie along its track, and the velocity/radiant
error bars smear the reconstructed past position over tens of parsecs within a few
Myr.  A close pass is therefore **necessary but not sufficient** for a common
origin, and the astrophysical priors overwhelmingly favour a generic Galactic-disk
origin over any one named star.  Nothing in this channel may be read as a claim
that an ISO came from LHS 1140; it reports a *distribution* of closest-approach
distances with that caveat attached as a first-class result field.
"""
