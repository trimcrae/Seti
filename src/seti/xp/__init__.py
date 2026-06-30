"""Blind technosignature search of Gaia DR3 XP low-resolution spectra.

A novel application of a recent, vast dataset (~220M BP/RP spectrophotometric
spectra): find sources whose spectral *shape* no normal-stellar population model
reconstructs --- a partial-Dyson reprocessing deficit or an artificial spectral
feature --- then run them through a contamination funnel (QSO/galaxy, white dwarf,
emission-line star, low-quality XP) to isolate any that resist natural explanation.
"""

from .anomaly import XPLocus, anomaly_score, fit_locus, normalize_spectrum

__all__ = ["XPLocus", "anomaly_score", "fit_locus", "normalize_spectrum"]
