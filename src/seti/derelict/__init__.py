"""DERELICT — thin-film debris, found by what radiation pressure does to it.

Signature S19 of ``docs/necrosignatures.md``.

The claim
---------
A derelict lightsail or thin-film structure has an **enormous area-to-mass
ratio**, so solar radiation pressure gives it a large **radial**
non-gravitational acceleration with **no outgassing**.  A natural body cannot:
the gap is four to five orders of magnitude, and it is a gap in a quantity JPL
already fits routinely and publishes for every small body it has enough
astrometry for.

The novelty
-----------
Every ingredient exists separately and nobody has combined them.  Bialy & Loeb
2018 ran the A1 -> beta -> area-to-mass -> lightsail inference **for 1I/'Oumuamua
alone**.  JPL and the MPC use the same test reactively to unmask *human*
hardware.  And the dark-comet catalogues (Seligman et al. 2023, 2024) — the only
systematic census of coma-free non-gravitationally-accelerating objects — were
built by selecting on **large non-radial acceleration**, which is precisely the
signature that *excludes* radiation pressure.

So the population this channel targets is the set the dark-comet literature
threw away: **significant radial A1, with A2 and A3 consistent with zero, and no
coma**.  That complement is unexamined.

Scope honesty
-------------
JPL's asteroid non-grav solutions are overwhelmingly A2-only (Yarkovsky).  The
set with a fitted A1 and a non-cometary classification is plausibly tens of
objects, not thousands — which makes this a *complete, tractable* search rather
than a scale play.  The run reports the true row counts; nothing in this package
assumes a population size.
"""

from .acquire import fetch_nongrav_table, fetch_object_detail
from .radiation import (
    AMR_PER_A1,
    AMR_PER_BETA,
    BETA_PER_A1,
    amr_from_a1,
    amr_from_beta,
    amr_natural,
    beta_from_a1,
    beta_from_amr,
    diameter_from_h,
    r_statistic,
)
from .run import derelict_run
from .screen import ScreenParams, run_screens
from .vet import VetParams, vet_object, vet_table

__all__ = [
    "AMR_PER_A1", "AMR_PER_BETA", "BETA_PER_A1",
    "amr_from_a1", "amr_from_beta", "amr_natural", "beta_from_a1",
    "beta_from_amr", "diameter_from_h", "r_statistic",
    "fetch_nongrav_table", "fetch_object_detail",
    "ScreenParams", "run_screens", "VetParams", "vet_object", "vet_table",
    "derelict_run",
]
