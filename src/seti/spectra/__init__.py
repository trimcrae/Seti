"""Blind, catalogue-scale search for laser-emission technosignatures in public
survey spectra (DESI / SDSS / BOSS via SPARCL).

A continuous-wave optical laser would appear in a stellar/galaxy spectrum as a
single narrow emission line that is (a) *unresolved* --- its width matches the
instrumental line-spread function (LSF), neither broader, as a real astrophysical
emission line, nor sharper, as a cosmic-ray hit --- and (b) inconsistent with any
known astrophysical, sky or telluric feature.  This subpackage implements that
search with the same contamination-robust, multi-axis, injection-calibrated
methodology developed for the white-dwarf infrared-excess search, but applied to
tens of millions of spectra rather than thousands of photometric sources.
"""

from .detect import EmissionLine, estimate_continuum, find_emission_lines

__all__ = ["EmissionLine", "estimate_continuum", "find_emission_lines"]
