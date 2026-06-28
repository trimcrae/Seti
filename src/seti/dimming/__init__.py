"""Search for anomalous optical dimming --- Boyajian's-star ("Tabby's star")
analogues --- in public time-domain photometry.

A megastructure (a Dyson swarm under construction, or large artefacts on orbit)
passing in front of a star would produce deep, irregular, *aperiodic* dimming with
no infrared excess and no accompanying spectral change --- the defining,
still-unexplained signature of KIC 8462852.  This subpackage scores light curves
for that signature: deep flux dips that are neither the smooth sinusoid of a
pulsator nor the strictly periodic, symmetric eclipses of a binary.
"""

from .dips import DipStats, detect_dips
from .run import dimming_run

__all__ = ["DipStats", "detect_dips", "dimming_run"]
