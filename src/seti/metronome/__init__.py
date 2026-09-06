"""METRONOME --- clocks in stellar flare timing.

Signature **S28** (:doc:`docs/necrosignatures.md` §2.VI): *structure in timing
series* --- the one informational-residue signature in the taxonomy that had
never been built.

The question is not "does this star flare?" but "do this star's catalogued
brief brightenings occur on a **strict clock**?"  Natural flares are stochastic:
their waiting times are Poisson-like, and the only natural quasi-periodicity
--- rotational modulation of flare visibility --- carries a phase jitter of
order a tenth of a cycle or more.  An artificial periodic energy release (a
pulsed transmitter, a duty-cycled engine, a beacon) is a clock whose timing
jitter is far below anything rotation produces.  The observable is therefore
the **timing regularity of catalogued events**, not the events themselves, and
the public flare catalogues (Kepler: Yang & Liu 2019, Davenport 2016; TESS:
Pietras et al. 2022, Günther et al. 2020, ...) are the substrate.

Three disciplines make the search honest:

* **the null carries the cadence and the gaps by construction** --- event
  times are resampled from each star's *own* observing windows and snapped to
  the mission cadence, so quarter gaps, sector gaps and 30-min quantisation are
  inside the null rather than inside the signal;
* **every rejection is a named mechanism with its own counter** ---
  ``cross_star_coincidence`` (spacecraft systematics shared by many stars),
  ``rotation_alias``, ``cadence_alias``, ``periodic_variable`` (a pulsator's
  cycles chopped into "flares"), ``bursty_random``, ``jitter_too_large``;
* **a null is a reason to change the question, not a result** --- the verdict
  vocabulary separates ``NO_DATA_REACHED`` from ``NO_CLOCK_CANDIDATES`` and
  neither is written up.

Modules
-------
``windows``   observing-window model (Kepler quarters, TESS sectors, data-driven)
``clock``     the phase-coherence detector, its two nulls, cross-star removal
``vet``       the contamination gauntlet and the tier assignment (pure functions)
``acquire``   runner-only VizieR access with runtime schema discovery
``run``       stage orchestration -> ``results/metronome/``
"""

from __future__ import annotations

__all__ = ["clock", "vet", "windows"]
