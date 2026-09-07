"""LZEDGE — the LZ 248 keV nuclear-recoil candidate at the kinematic edge.

Pure-function physics for reading a single high-energy recoil against the
local dark-matter velocity distribution: inelastic (endothermic) kinematics,
the Earth's velocity through the halo on a given date, halo models with
arbitrary high-speed components (SHM, SHM++, streams, an LMC-boosted tail),
annual modulation at the kinematic edge, and the likelihood of the event's
date and energy under each model.  Everything here runs offline; the record
it is built against is fetched by ``scripts/lzlit_fetch.py``.
"""
