"""Band definitions and the interstellar extinction law for CENOTAPH.

The whole channel rests on one fact: **interstellar extinction is steeply
chromatic and an occulter is not**. Over the bands used here, ``A_band/A_V``
spans a factor of ~88 (GALEX NUV 2.278 → WISE W2 0.026), so a grey flux deficit
and a reddening column are near-orthogonal vectors in magnitude space and can be
fitted simultaneously rather than assumed apart.

Coefficients are ``A_band / A_V`` for a Fitzpatrick R_V = 3.1 law:

* optical/near-IR/mid-IR from Wang & Chen (2019), ApJ 877, 116 — the standard
  Gaia-era determination, self-consistent across Gaia/2MASS/WISE;
* the two GALEX bands from Yuan, Liu & Xiang (2013), MNRAS 430, 2188, quoted
  there as ``A_band/E(B-V)`` and divided by R_V = 3.1 here. NUV exceeds FUV
  because the 2175 Å bump sits inside the NUV response.

``lambda_eff_um`` is used only for SED integration and reporting, never in the
fit itself.
"""

from __future__ import annotations

from dataclasses import dataclass

R_V = 3.1
"""Total-to-selective extinction ratio assumed by the coefficients below."""


@dataclass(frozen=True)
class Band:
    """A photometric band: extinction coefficient and effective wavelength."""

    name: str
    a_over_av: float
    lambda_eff_um: float
    zp_jy: float
    """Zero-point flux density (Jy) for a 0-mag source, for SED integration."""


# Ordered blue -> red. The lever arm between the extremes is what separates a
# grey deficit from a reddening column.
BANDS: tuple[Band, ...] = (
    Band("fuv", 1.410, 0.1528, 3631.0),   # Yuan+2013 A/E(B-V)=4.37
    Band("nuv", 2.278, 0.2271, 3631.0),   # Yuan+2013 A/E(B-V)=7.06 (2175 A bump)
    Band("bp", 1.002, 0.5050, 3552.0),    # Wang & Chen 2019
    Band("g", 0.789, 0.6230, 3229.0),     # Gaia G
    Band("rp", 0.589, 0.7727, 2555.0),
    Band("j", 0.243, 1.235, 1594.0),      # 2MASS
    Band("h", 0.131, 1.662, 1024.0),
    Band("ks", 0.078, 2.159, 666.7),
    Band("w1", 0.039, 3.353, 309.5),      # WISE
    Band("w2", 0.026, 4.603, 171.8),
)

# W3/W4 exist for the *excess veto* (leg 2) but are deliberately kept out of
# BANDS, i.e. out of the grey/reddening fit, for two reasons:
#   1. they are where an occulter's re-radiation would first appear, so
#      including them would let a genuine excess bleed into the fitted grey
#      term and cancel the very signal we are measuring;
#   2. their extinction coefficients are the least secure in the table — the
#      9.7 um silicate feature sits inside W3, so A_W3/A_Ks is larger than the
#      smooth continuum trend and varies with sightline.
EXCESS_BANDS: tuple[Band, ...] = (
    Band("w3", 0.030, 11.561, 31.67),     # A_W3/A_Ks ~ 0.38 (Wang & Chen 2019)
    Band("w4", 0.010, 22.088, 8.363),     # poorly constrained; ~0.13 A_Ks
)

BAND_INDEX: dict[str, Band] = {b.name: b for b in BANDS + EXCESS_BANDS}

FIT_BANDS: tuple[str, ...] = tuple(b.name for b in BANDS)
"""The bands the grey/reddening fit may use. Excludes W3/W4 by construction."""

# GALEX magnitudes are AB; Gaia are Vega-ish (G) but the fit is differential
# against twins measured in the same system, so system offsets cancel exactly
# and only the *relative* extinction coefficients matter.

EXTINCTION_FREE_BANDS: tuple[str, ...] = ("ks", "w1", "w2")
"""Bands where A_band <= 0.078 A_V, i.e. <0.025 mag for A_V < 0.3.

The primary luminosity statistic is built here: an absolute-magnitude deficit
measured in Ks or W1 is essentially extinction-immune, so it cannot be
manufactured by a dust column the 3D maps missed.
"""


def a_over_av(band: str) -> float:
    """Return ``A_band / A_V`` for ``band``."""
    try:
        return BAND_INDEX[band].a_over_av
    except KeyError as exc:  # noqa: TRY003
        raise KeyError(f"unknown band {band!r}; known: {sorted(BAND_INDEX)}") from exc


def reddening_vector(bands: list[str]) -> list[float]:
    """The chromatic response ``A_band/A_V`` for ``bands``, in order."""
    return [a_over_av(b) for b in bands]


def grey_vector(bands: list[str]) -> list[float]:
    """The achromatic response — unity in every band, by definition."""
    return [1.0] * len(bands)


def covering_fraction_from_grey(grey_mag: float) -> float:
    """Convert a grey magnitude deficit to an intercepted flux fraction ``f``.

    ``m_obs - m_int = -2.5 log10(1 - f)``  =>  ``f = 1 - 10**(-0.4 * grey)``.
    Negative ``grey`` (an over-luminous star) returns a negative ``f``, which is
    unphysical for an occulter and is how the caller detects the wrong sign.
    """
    return 1.0 - 10.0 ** (-0.4 * grey_mag)


def grey_from_covering_fraction(f: float) -> float:
    """Inverse of :func:`covering_fraction_from_grey`."""
    if f >= 1.0:
        raise ValueError("covering fraction must be < 1")
    return -2.5 * _log10(1.0 - f)


def _log10(x: float) -> float:
    from math import log10

    return log10(x)
