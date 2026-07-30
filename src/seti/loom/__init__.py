"""LOOM --- a population search for self-replicating probes in the solar system.

*A loom is the machine that makes copies of a pattern.*

The premise, and why it is a population search
----------------------------------------------
A von Neumann probe is **defined** by self-replication.  That single fact settles
the design: the signature is not one anomalous object, because a single object
with an unexplained acceleration has a dozen mundane readings — a bad orbit, an
unmodelled perturber, an undetected satellite, a comet nobody has caught
outgassing.  Replication predicts something no individual object can supply: a
**population sharing an origin**, over-clustered in orbital elements, coherent in
orbital pole, concentrated at dynamically privileged locations, and built to a
specification rather than broken off a parent body.

So LOOM screens objects one at a time and *decides* on the population.  The
per-object stage produces a ranked anomalous set; the decision is made by
:mod:`seti.loom.replication`, against a null of matched random subsets of the same
screened sample.  This is the same architecture TOCSIN arrived at for stars, for
the same reason: a per-object test is a contamination problem, and a
population-structure test on the *set* of flagged objects is immune to any
contaminant that does not itself cluster.

What the data actually gives, and what is genuinely new
-------------------------------------------------------
Rubin's alert packets carry, for every detection matched to a known minor planet,
the observed-minus-predicted ephemeris offset already decomposed into along-track
and cross-track components, plus the heliocentric distance at that epoch; the
MPC orbit table alongside carries the fitted Yarkovsky and radiation-pressure
terms where they exist.  Nothing in this channel needs an orbit integrator or a
proprietary catalogue.

Three claims have to be separated, because only two of them are new:

* **Not new.**  "An inactive small body with a non-gravitational acceleration
  larger than Yarkovsky predicts."  Seligman et al. (2023) found seven — the dark
  comets — and hidden outgassing is the accepted reading.  A magnitude threshold
  rediscovers them, and this channel treats them as its hardest confuser class
  and its sensitivity control, not as its output.
* **New.**  Asking *which law* the acceleration follows.  Sublimation has a
  characteristic dependence on heliocentric distance with a knee near 2.8 au;
  radiation pressure is a clean inverse square; something holding a trajectory is
  neither.  With a multi-apparition arc these are separable per object, and no
  published search does it.
* **New.**  Asking whether the *set* of anomalous objects has the structure
  replication implies.  Nobody has run a matched-null population test — element
  clustering, orbital-pole coherence, resonance concentration, inclination
  isotropy — on an anomaly-selected solar-system sample.

The one thing this channel has that nothing else here does
---------------------------------------------------------
A **positive control.**  Human beings have put artificial objects into orbits that
were catalogued as minor planets — the Apollo 12 S-IVB stage as ``J002E3``,
``WT1190F``, the 1966 Surveyor 2 Centaur as ``2020 SO`` — and in each case what
gave them away is exactly what this channel measures: an area-to-mass ratio no
rock can have, 20 to 40 times the natural small-body locus.  If the screen does
not recover objects like those, it does not work.  No other channel in this
repository can be validated against a real positive, because no confirmed
technosignature exists.  See :mod:`seti.loom.control`.

The gate is a theorem, not a fit
--------------------------------
Yarkovsky acceleration is recoil from re-radiating absorbed sunlight, so the
thermal photons cannot carry more momentum than the intercepted beam.  That gives
a hard ceiling on any radiation-driven acceleration as a function of area-to-mass
ratio, calibrated on objects with independently measured ``A2`` (the realised
efficiency is 2-8%, so a factor-ten margin is already generous).  A body above the
ceiling is not being pushed by sunlight, whatever else is true of it.  Using a
momentum budget rather than a thermophysical model means the gate does not rest on
an object's albedo, spin, obliquity or thermal inertia — none of which are known
for almost any object in the sample.  See :mod:`seti.loom.nongrav`.
"""

from .control import ARTIFICIAL, DARK_COMETS, NONGRAV_DETECTED, validate
from .nongrav import (
    EPSILON_HARD,
    EPSILON_INVIOLABLE,
    EPSILON_REALISTIC,
    a2_from_yarkovsky_column,
    amr_ceiling_ratio,
    amr_from_srp_column,
    amr_sphere,
    beta_from_amr,
    calibration_table,
    ceiling_ratio,
    dadt_au_per_myr,
    diameter_m_from_h,
    fit_envelope,
    fit_quality,
    momentum_ceiling_a2,
)
from .replication import (
    element_clustering,
    inclination_isotropy,
    orbital_poles,
    pole_coherence,
    replication_tests,
    resonance_concentration,
    resonance_locations,
)
from .residuals import (
    arcsec_to_km,
    breakpoint_scan,
    drift_fit,
    fit_common_timing,
    g_comet,
    g_radiation,
    law_discrimination,
    residual_significance,
    sky_coherence,
)
from .screen import Thresholds, assign_tier, screen_orbit_row, screen_orbits

__all__ = [
    "ARTIFICIAL",
    "DARK_COMETS",
    "EPSILON_HARD",
    "EPSILON_INVIOLABLE",
    "EPSILON_REALISTIC",
    "NONGRAV_DETECTED",
    "Thresholds",
    "a2_from_yarkovsky_column",
    "amr_ceiling_ratio",
    "amr_from_srp_column",
    "amr_sphere",
    "arcsec_to_km",
    "assign_tier",
    "beta_from_amr",
    "breakpoint_scan",
    "calibration_table",
    "ceiling_ratio",
    "dadt_au_per_myr",
    "diameter_m_from_h",
    "drift_fit",
    "element_clustering",
    "fit_common_timing",
    "fit_envelope",
    "fit_quality",
    "g_comet",
    "g_radiation",
    "inclination_isotropy",
    "law_discrimination",
    "momentum_ceiling_a2",
    "orbital_poles",
    "pole_coherence",
    "replication_tests",
    "residual_significance",
    "resonance_concentration",
    "resonance_locations",
    "screen_orbit_row",
    "screen_orbits",
    "sky_coherence",
    "validate",
]
