"""High-resolution cross-correlation biosignature search for LHS 1140 b.

A Doppler-resolved transmission-spectroscopy channel: during transit the planet's
atmosphere imprints a Doppler-shifted forest of molecular lines (O2 0.76 um
A-band, H2O) on ground-based high-resolution spectra (ESPRESSO/HARPS/NIRPS/
IGRINS).  Cross-correlating each in-transit residual against a molecular template
and stacking along the planet's ``Kp``-``Vsys`` velocity track pulls the signal out
of the noise -- complementary to JWST transmission spectroscopy and, for the O2
A-band, only reachable from the ground where the planetary lines Doppler-separate
from the stationary telluric O2.

:mod:`seti.crosscorr.xcorr` holds the pure, offline-tested engine (templates,
CCF, Kp-Vsys map); :mod:`seti.crosscorr.run` does the runner-side ephemeris
resolution, archive lookup, detrending and reporting.
"""

from .xcorr import (
    cross_correlation,
    h2o_template,
    kp_vsys_map,
    molecular_template,
    o2_a_band_template,
    planet_rv_track,
)

__all__ = [
    "planet_rv_track", "cross_correlation", "kp_vsys_map",
    "molecular_template", "o2_a_band_template", "h2o_template",
]
