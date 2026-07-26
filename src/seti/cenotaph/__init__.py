"""CENOTAPH — the cold-Dyson search in the unprobed T < 100 K regime.

*An empty tomb: a monument for a body that is elsewhere.*

Every executed Dyson/waste-heat search is instrumentally capped by WISE W4 at
22 µm and quotes 100–1000 K. Ćirković & Bradbury (2006) argued on
Landauer/Brillouin grounds that postbiological computation prefers a *cold*
reservoir, contrasting "a Dyson shell ... close to a blackbody at 50 K, as
contrasted to a blackbody at 300 K". The predicted regime and the searched
regime do not overlap.

This channel tests three legs jointly, as one energy-conservation argument:

1. **Grey attenuation** — a covering fraction ``f`` dims the star equally in
   every band; interstellar dust does not. Fitted jointly, never assumed.
2. **No mid-IR excess** — which is exactly what makes such an object invisible
   to every executed search.
3. **Far-IR recovery** — the intercepted ``f·L`` reappearing at 60–160 µm,
   turning "missing energy" into "found the energy where nobody looked".

See ``docs/cenotaph.md`` for the claim, the honest novelty status relative to
Zackrisson et al. (2018), and the contamination model.
"""

from .extinction import BANDS, EXCESS_BANDS, FIT_BANDS, covering_fraction_from_grey
from .greyfit import GreyFit, fit_grey_reddening, minimum_detectable_f
from .twins import TwinConfig, twin_statistics

__all__ = [
    "BANDS",
    "EXCESS_BANDS",
    "FIT_BANDS",
    "GreyFit",
    "TwinConfig",
    "covering_fraction_from_grey",
    "fit_grey_reddening",
    "minimum_detectable_f",
    "twin_statistics",
]
