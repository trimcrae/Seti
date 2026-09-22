"""SLAG-WD (S51): the refinery vector --- polluted white dwarfs beyond the natural family.

A white-dwarf photosphere is a mass spectrometer of what fell in.  Every
natural process that shapes what falls in moves element ratios along four
levers --- condensation temperature, metal--silicate partitioning, melt
incompatibility and photospheric sinking --- and this package asks, per
object, whether the measured abundance vector is reachable by any
combination of them (Tier 1, the calibrated misfit list) and whether any
process-orthogonal element pair sits outside the whole natural envelope
while the rest of the panel is natural (Tier 2, the pair residual).

Modules
-------
``family``   the embedded natural end-member table, mixtures, condensation
             fractionation and per-pair natural envelopes
``sinking``  photospheric diffusion: relative timescales per element and the
             three accretion phases of Koester 2009
``misfit``   Tier 1: the natural-model fit of one panel and its calibrated
             misfit probability from injected natural draws
``pairs``    Tier 2: pair residuals, the refinery flags and the kills
``acquire``  PEWDD over the VizieR route ladder (GitHub as the second route),
             column-role resolution, atmosphere-type resolution
``run``      stage orchestration; writes ``results/slag/``
"""

from .family import NaturalFamily, load_family

__all__ = ["NaturalFamily", "load_family"]
