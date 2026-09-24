"""OCCULT --- an opaque microlens occulting its own images (S40) in ground surveys.

A point lens that is also an opaque disc of radius rho_L (Einstein units)
removes the minor image whenever it falls inside rho_L.  For rho_L < 1 that
is every u > u_c = 1/rho_L - rho_L: an otherwise standard light curve with a
symmetric pair of DOWNWARD steps at t0 +- tE sqrt(u_c^2 - u0^2), each of depth
A_-(u_c) times the source flux, fixed jointly by the one extra parameter.
A natural lens has rho_L ~ 1e-3 (a star against an AU-scale Einstein ring);
rho_L ~ 0.3-1 is an AU-scale opaque body of stellar mass --- a Dyson sphere
seen by its gravity and its shadow rather than its heat.

Modules
-------
``probe``    what OGLE / KMTNet / MOA photometry is reachable from a runner
``acquire``  bulk per-event photometry pulls (runner) + pure parsers
``detect``   the detector: FSPL vs FSPL+opaque-lens vs free-step fits --- pure
``inject``   injection of opaque-lens signals into real event baselines
``run``      stage orchestration, ``results/occult/``
"""
