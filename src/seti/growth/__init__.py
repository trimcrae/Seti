"""GROWTH --- the growing transit: construction around a planet, 2009 -> 2026.

Signature **S57** (:doc:`docs/necrofrontier.md` §2, "The growing transit"):
a shell or swarm being *built* around a transiting planet grows the planet's
transit depth achromatically over years.  No systematic Kepler-versus-TESS
transit-depth consistency catalogue exists (Wang & Espinoza 2024 searched
within-TESS TDVs; Zuckerman et al. 2023 searched single-transit anomalies in
Kepler; Kaye & Aigrain 2025 compared ephemerides only), and the sweep record
(§5, row g11) finds the question unoccupied.

Stage 1 --- this package --- is the **catalogue pass**: for every Kepler
planet or candidate that TESS re-detected, compare the Kepler-era depth
(KOI ``koi_depth``, 2009--2013) with the TESS-era depth (TOI ``pl_trandep``,
2018--2026) after the deterministic corrections (limb-darkening band ratio at
the catalogue impact parameter; Gaia-neighbour dilution; a systematic floor),
and flag significant growth or shrinkage at fixed impact parameter.

**Stage 2A** --- :mod:`seti.growth.stage2` --- measures the TESS depth **from
the light curve** for a shortlist of stage-1 candidates and compares THREE
numbers instead of two: the Kepler catalogue depth (carried into the TESS band
by the limb-darkening ratio), the TOI catalogue depth, and its own fit.  It
exists because stage 1's formal errors are demonstrably not a sigma scale ---
run 35038510064 measured 108 planets and put 31 of them above 5 sigma of the
population median --- so the only way past the catalogue is to stop reading it.
The rest of stage 2 (per-epoch ``k(t)``, achromaticity, pixel-level dilution,
the long-period asymmetry branch) is designed in ``docs/growth.md`` §8 and not
built here.

Three disciplines make the pass honest:

* **the expected sign is a deficit** --- Han et al. 2025 find public TESS
  radii ~6 % low from residual blending, i.e. a ~12 % depth deficit, so
  ``SHALLOWER_TESS`` is the ordinary direction and ``DEEPER_TESS`` the
  anomalous one; the population median ln-ratio is measured and reported;
* **a depth change at fixed impact parameter changes the duration by a
  computable factor** --- a change that instead tracks a changed ``b``
  (nodal precession) is named and never a candidate;
* **a null is a reason to change the question** --- the verdict vocabulary
  separates ``NO_DATA_REACHED`` from ``NO_DEPTH_DRIFT_CANDIDATE`` and neither
  is written up.

Modules
-------
``acquire``  Exoplanet Archive TAP (KOI ``cumulative``, ``toi``, ``ps``) and
             the Gaia DR3 neighbour cones; the Kepler <-> TESS join (pure)
``drift``    limb-darkening band ratio, dilution, corrected log ratio, the
             duration-consistency test, the classification (pure)
``vet``      grazing / TTV / EB / disposition / neighbour vetoes (pure)
``run``      stage orchestration -> ``results/growth/``
``stage2``   the MEASURED TESS depth: MAST light curves, a transit-masked
             depth fit at the fixed KOI ephemeris, per-sector and odd-even
             statistics, and the three-way verdict -> ``results/growth/stage2/``
"""

from __future__ import annotations

__all__ = ["acquire", "drift", "stage2", "vet"]
