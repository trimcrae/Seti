"""TIDEMARK --- the front that stopped.

Every executed technosignature search reports *individual* candidates and then
dies on per-object contamination: any one infrared excess is a blend, any one
dimming is dust, any one abundance anomaly is a pipeline systematic.  The
population-level question --- **is the anomaly rate spatially structured?** ---
is immune to exactly that failure mode, because a contaminant population traces
the ordinary stellar density and cannot manufacture a coherent gradient or a
sharp edge in *rate per star*.

It has also never been asked.  The literature contains published, falsifiable,
and mutually **contradictory** predictions about where technosignatures should
concentrate --- Cirkovic & Bradbury 2006 says the outer rim, Wright et al. 2021
(RNAAS 5, 141) says the Galactic centre, Wright et al. 2014 says rotational
shear erases any structure at all, and Carrigan 2010 / Landis 1998 / Hanson et
al. 2021 say the observable is a *boundary* --- and not one of them has ever been
confronted with data.  See ``docs/tidemark.md``.

The engine is the sibling of ``seti.cluster``, generalised from "is this subset
clumped?" to "does this subset's rate vary across a coordinate?".  Three
statistics, one null:

* ``gradient`` --- monotone trend in anomaly rate vs Galactocentric R, |z|, or
  Galactic longitude (as a dipole), reported as an amplitude with a CI;
* ``edge``     --- a matched-filter scan for a sharp step in rate, in 1D, in 3D
  spherical shells, and in sky caps, against a null that already contains the
  fitted smooth gradient;
* ``agerate``  --- rate vs stellar age: rising, saturating, or turning over.

All three are calibrated against ``nulls.MatchedNull``: random subsets of the
*parent catalogue itself*, matched on the covariates that control detectability
and never on the coordinate under test.  That parent-matched resampling is the
whole scientific difficulty of the channel --- position alone is washed out by
the Galactic density gradient --- and solving it honestly is the contribution.
"""

from .agerate import age_proxies, age_rate_test
from .edge import edge_scan_1d, edge_scan_cap, edge_scan_shell3d
from .gradient import gradient_test, rate_profile
from .ingest import (
                     AnomalyCatalogue,
                     add_galactic_frame,
                     from_frames,
                     load_all,
                     load_channel,
                     union_catalogue,
)
from .nulls import MatchedNull

__all__ = [
    "AnomalyCatalogue", "add_galactic_frame", "from_frames", "load_channel",
    "load_all", "union_catalogue", "MatchedNull", "gradient_test",
    "rate_profile", "edge_scan_1d", "edge_scan_shell3d", "edge_scan_cap",
    "age_proxies", "age_rate_test",
]
