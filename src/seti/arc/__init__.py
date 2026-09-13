"""ARC --- superflares above the starspot energy ceiling.

Signature **S59** (:doc:`docs/necrofrontier.md` §2, "The arc"): a flare's
energy is bounded by the magnetic energy its starspots store,

    E_mag = f (B^2 / 8 pi) A_spot^(3/2)          (cgs; B = 3 kG, f <= 1),

and Okamoto et al. (2021) find the Kepler superflare upper envelope consistent
with f ~ 0.1.  The spot area follows from the star's rotational modulation
amplitude.  A flare that EXCEEDS the bound at f = 1 on a star whose amplitude
gives A_spot is either a measurement problem or not a flare; deliberate
stellar manipulation would show as repeated, target-localised superflares
above the bound on a spot-free slow rotator.  The bound is established in the
natural literature and has never been mined as an anomaly set.

The statistic per star is

    xi = log10 E_flare - log10 E_mag(f = 1, B = 3 kG, A_conservative),

with A_conservative marginalised over spot latitude (the amplitude is a LOWER
bound on the spot area: polar or symmetric spots modulate weakly).  The
deliverable is the vetted ``xi > 0`` list, which is by construction the
contamination set of Notsu et al. (2019) --- unresolved companion flarers ---
until the stage-2 pixel-centroid test says otherwise.

Modules
-------
``ceiling``   the physics (Berdyugina spot temperature, spot area, the bound, xi)
``vet``       the contamination gauntlet and the tier assignment (pure)
``acquire``   runner-only VizieR / Gaia access with runtime schema discovery
``run``       stage orchestration -> ``results/arc/``
"""

from __future__ import annotations

__all__ = ["ceiling", "vet"]
