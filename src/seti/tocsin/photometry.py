"""Difference-image photometry primitives --- units, greyness, colour temperature.

Everything here is a pure function of numbers, so it is fully offline-testable
and carries no archive dependency.

The physics the channel turns on
-------------------------------
Rubin alerts carry *difference-image* PSF flux in **nanojanskys**, signed: a
source brighter than the template gives ``psfFlux > 0``, fainter gives
``psfFlux < 0``.  Working in flux rather than magnitudes is not a stylistic
choice --- a negative flux has no magnitude, and the dip half of this channel
lives entirely at negative flux.

For an event on a star of baseline flux ``F*``, the **fractional amplitude**
``a_b = dF_b / F*_b`` is the quantity whose *band-to-band equality* separates
the hypotheses:

======================  =====================================================
hypothesis              expectation
======================  =====================================================
specular reflection     ``a`` equal in all bands (the reflector returns the
                        stellar spectrum), and the difference-flux colour
                        temperature equals the **stellar** temperature
stellar flare           ``a`` strongly larger in blue bands; difference-flux
                        colour temperature ~9000-10^4 K regardless of host
grey occulter           ``a`` equal in all bands, negative
line-of-sight dust      ``a`` negative and *reddened*: ``a_g/a_r`` set by the
                        extinction law (~1.4 for Rubin g,r), never 1
======================  =====================================================

So one statistic --- band-to-band equality of ``a`` --- carries the flare
rejection for flashes and the dust rejection for dips.  This is the same
grey-vs-chromatic discipline the contamination ledger already imposes on the
ZTF channels (``docs/channel-brief.md`` §4), applied per-event instead of
per-light-curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
# AB zero point in nJy: 3631 Jy == 0 mag, so m = ZP - 2.5*log10(F/nJy) with
# ZP = 2.5*log10(3631e9) = 31.4003...  Rubin publishes alert fluxes in nJy
# (SDM: `psfFlux`), which is why this constant exists rather than a mag column.
AB_ZP_NJY = 31.4

# LSST effective wavelengths (um).  Used only for blackbody *ratios*, so the
# band-averaged effective wavelength is adequate; a full throughput integral
# would shift derived temperatures by less than the errors this channel can
# ever achieve on a two-band difference-flux colour.
LSST_BAND_WL_UM = {
    "u": 0.3671,
    "g": 0.4827,
    "r": 0.6223,
    "i": 0.7546,
    "z": 0.8691,
    "y": 0.9712,
}

_H = 6.62607015e-34   # J s
_C = 2.99792458e8     # m / s
_KB = 1.380649e-23    # J / K


def njy_to_ab(flux_njy: float | np.ndarray) -> float | np.ndarray:
    """AB magnitude from a *positive* flux in nJy; NaN where flux <= 0.

    Non-positive fluxes are returned as NaN rather than raising: a negative
    difference flux is a physically meaningful measurement (the dip mode), it
    simply has no magnitude, and callers must not silently treat it as bright.
    """
    f = np.asarray(flux_njy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = AB_ZP_NJY - 2.5 * np.log10(np.where(f > 0, f, np.nan))
    return float(m) if np.isscalar(flux_njy) or np.ndim(flux_njy) == 0 else m


def ab_to_njy(mag_ab: float | np.ndarray) -> float | np.ndarray:
    """Flux in nJy from an AB magnitude."""
    m = np.asarray(mag_ab, dtype=float)
    f = 10.0 ** (-0.4 * (m - AB_ZP_NJY))
    return float(f) if np.isscalar(mag_ab) or np.ndim(mag_ab) == 0 else f


def planck_nu(wl_um: float, temp_k: float) -> float:
    """Planck ``B_nu`` in arbitrary consistent units (only ratios are used)."""
    if temp_k <= 0:
        return 0.0
    wl_m = wl_um * 1e-6
    nu = _C / wl_m
    x = _H * nu / (_KB * temp_k)
    # Guard the Wien tail: exp(x) overflows for cool T at blue wavelengths, and
    # the physically correct limit there is zero flux, not an exception.
    if x > 700.0:
        return 0.0
    return (2.0 * _H * nu**3 / _C**2) / math.expm1(x)


# ---------------------------------------------------------------------------
# Fractional amplitude
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Amplitude:
    """Fractional difference-image amplitude ``a = dF / F*`` and its error.

    ``testable`` is False when the baseline flux is missing or non-positive, in
    which case ``a`` is NaN.  A missing baseline must propagate as *untestable*,
    never as a pass --- the ledger promotes on evidence, and an absent
    denominator is not evidence.
    """

    a: float
    a_err: float
    testable: bool
    reason: str = ""


def fractional_amplitude(dflux_njy: float, dflux_err_njy: float,
                         base_flux_njy: float | None,
                         base_flux_err_njy: float | None = None) -> Amplitude:
    """``a = dF/F*`` with errors propagated through the ratio.

    The baseline flux ``F*`` is the *template* (quiescent) flux of the star in
    the same band.  Its own error is included when known because for the faint
    end of a nearby-star sample it is not negligible relative to ``dF``.
    """
    if base_flux_njy is None or not np.isfinite(base_flux_njy) or base_flux_njy <= 0:
        return Amplitude(float("nan"), float("nan"), False, "no_baseline_flux")
    if not np.isfinite(dflux_njy) or not np.isfinite(dflux_err_njy) or dflux_err_njy <= 0:
        return Amplitude(float("nan"), float("nan"), False, "bad_difference_flux")
    a = dflux_njy / base_flux_njy
    rel_d = dflux_err_njy / abs(dflux_njy) if dflux_njy != 0 else float("inf")
    rel_b = 0.0
    if base_flux_err_njy is not None and np.isfinite(base_flux_err_njy):
        rel_b = base_flux_err_njy / base_flux_njy
    if not np.isfinite(rel_d):
        return Amplitude(float("nan"), float("nan"), False, "zero_difference_flux")
    a_err = abs(a) * math.hypot(rel_d, rel_b)
    return Amplitude(a, a_err, True)


def greyness_z(a1: float, a1_err: float, a2: float, a2_err: float) -> float:
    """Significance of the band-to-band *difference* in fractional amplitude.

    Zero for a grey event (reflection, or a grey occulter); large and positive
    when the bluer band has the larger amplitude (a flare, or reddening dust,
    depending on sign).  The caller supplies ``(a1, a1_err)`` for the **bluer**
    band by convention, so the sign of the returned ``z`` is interpretable.
    """
    den = math.hypot(a1_err, a2_err)
    if den <= 0 or not np.isfinite(den):
        return float("nan")
    return (a1 - a2) / den


def predicted_amplitude_ratio(band_blue: str, band_red: str,
                              temp_event_k: float, temp_star_k: float) -> float:
    """``a_blue / a_red`` expected when an event of temperature ``temp_event_k``
    is superposed on a star of temperature ``temp_star_k``.

    Because ``a_b = dF_b / F*_b``, the ratio is a ratio of blackbody ratios: the
    event's colour divided by the star's colour.  Returns NaN when either
    Planck function underflows to zero at these wavelengths (a cool star has no
    measurable u-band baseline, so the ratio is genuinely undefined).

    For the reflection hypothesis ``temp_event_k == temp_star_k`` and this
    returns exactly 1 --- which is the whole point of the statistic.
    """
    wb, wr = LSST_BAND_WL_UM[band_blue], LSST_BAND_WL_UM[band_red]
    ev = planck_nu(wb, temp_event_k), planck_nu(wr, temp_event_k)
    st = planck_nu(wb, temp_star_k), planck_nu(wr, temp_star_k)
    if ev[1] <= 0 or st[0] <= 0 or st[1] <= 0:
        return float("nan")
    return (ev[0] / ev[1]) / (st[0] / st[1])


# ---------------------------------------------------------------------------
# Difference-flux colour temperature
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ColourTemperature:
    """Blackbody temperature fitted to the *difference* fluxes themselves.

    This is the S30 quantity: an "unclassified blackbody transient".  A flare
    lands at ~9000-10^4 K whatever the host; a specular reflection lands on the
    host's own temperature.  ``t_lo``/``t_hi`` bracket the 1-sigma range from
    the chi^2 curve; ``bands`` records what actually constrained the fit.
    """

    temp_k: float
    t_lo: float
    t_hi: float
    chi2: float
    n_bands: int
    bands: tuple[str, ...]
    ok: bool
    reason: str = ""


def blackbody_colour_temperature(bands: list[str], dflux_njy: list[float],
                                 dflux_err_njy: list[float],
                                 t_grid: np.ndarray | None = None) -> ColourTemperature:
    """Fit a single blackbody to signed difference fluxes across >=2 bands.

    Only *positive* difference fluxes can constrain an emission temperature, so
    dip-mode events are refused here (their colour information is carried by
    :func:`greyness_z` against the extinction law instead).  The amplitude is
    profiled out analytically at each trial temperature, so the fit is a 1-D
    chi^2 scan and needs no optimiser.
    """
    if len(bands) < 2:
        return ColourTemperature(float("nan"), float("nan"), float("nan"),
                                 float("nan"), len(bands), tuple(bands), False,
                                 "need_two_bands")
    f = np.asarray(dflux_njy, dtype=float)
    e = np.asarray(dflux_err_njy, dtype=float)
    keep = np.isfinite(f) & np.isfinite(e) & (e > 0)
    if keep.sum() < 2:
        return ColourTemperature(float("nan"), float("nan"), float("nan"),
                                 float("nan"), int(keep.sum()), tuple(bands), False,
                                 "insufficient_finite_bands")
    bands_k = [b for b, k in zip(bands, keep, strict=True) if k]
    if any(b not in LSST_BAND_WL_UM for b in bands_k):
        return ColourTemperature(float("nan"), float("nan"), float("nan"),
                                 float("nan"), len(bands_k), tuple(bands_k), False,
                                 "unknown_band")
    f, e = f[keep], e[keep]
    if np.any(f <= 0):
        return ColourTemperature(float("nan"), float("nan"), float("nan"),
                                 float("nan"), len(bands_k), tuple(bands_k), False,
                                 "negative_flux_no_emission_temperature")
    if t_grid is None:
        # Log grid: the observable is a colour, whose sensitivity to T is
        # logarithmic once the Rayleigh-Jeans regime is reached.
        t_grid = np.geomspace(1500.0, 60000.0, 400)
    wl = np.array([LSST_BAND_WL_UM[b] for b in bands_k])
    chi2 = np.empty_like(t_grid)
    for j, t in enumerate(t_grid):
        model = np.array([planck_nu(w, float(t)) for w in wl])
        if not np.any(model > 0):
            chi2[j] = np.inf
            continue
        # Analytic best-fit scale for a linear model with Gaussian errors.
        w = 1.0 / e**2
        denom = float(np.sum(w * model * model))
        scale = float(np.sum(w * f * model)) / denom if denom > 0 else 0.0
        chi2[j] = float(np.sum(w * (f - scale * model) ** 2))
    jbest = int(np.argmin(chi2))
    best = float(t_grid[jbest])
    c_min = float(chi2[jbest])
    within = np.where(chi2 <= c_min + 1.0)[0]
    t_lo = float(t_grid[within[0]]) if within.size else float("nan")
    t_hi = float(t_grid[within[-1]]) if within.size else float("nan")
    # A two-band fit has zero degrees of freedom (one colour, one temperature):
    # chi^2 is ~0 by construction and carries no goodness-of-fit information.
    # Say so rather than letting a caller read chi2=0 as a good fit.
    reason = "two_band_zero_dof" if len(bands_k) == 2 else ""
    return ColourTemperature(best, t_lo, t_hi, c_min, len(bands_k),
                             tuple(bands_k), True, reason)


# ---------------------------------------------------------------------------
# One-sided colour test from a non-detection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GreyExclusion:
    """Whether a *non-detection* in another band rules out a grey event.

    Almost every event in the live stream is single-band, so the two-band
    achromaticity test seldom fires (0 of 22 events in the first correct run).
    That does not make the colour information unavailable --- it makes it
    one-sided.

    The physics: a grey event has the same *fractional* amplitude in every band,
    so on a red star it is BRIGHTER in absolute flux in the redder band.  If the
    event is seen in *g* while *r* was observed the same night and stayed silent,
    the grey hypothesis predicted a redder-band signal that did not appear ---
    which is evidence against greyness, and consistent with a flare.

    ``excluded`` is True only when the predicted flux exceeds the band's
    detection limit by ``margin``, so a marginal prediction is reported as
    untestable rather than counted either way.
    """

    excluded: bool
    tested: bool
    predicted_flux_njy: float
    limit_flux_njy: float
    other_band: str
    reason: str = ""


def grey_excluded_by_nondetection(a_obs: float, base_flux_other_njy: float | None,
                                  limit_flux_other_njy: float | None,
                                  other_band: str, margin: float = 3.0
                                  ) -> GreyExclusion:
    """Test the grey hypothesis against a silent band observed the same night.

    ``a_obs`` is the fractional amplitude measured in the detected band;
    ``base_flux_other_njy`` the star's quiescent flux in the silent band;
    ``limit_flux_other_njy`` that band's effective alert threshold.
    """
    nan = float("nan")
    if not np.isfinite(a_obs):
        return GreyExclusion(False, False, nan, nan, other_band, "amplitude_untestable")
    if base_flux_other_njy is None or not np.isfinite(base_flux_other_njy) \
            or base_flux_other_njy <= 0:
        return GreyExclusion(False, False, nan, nan, other_band, "no_baseline_other_band")
    if limit_flux_other_njy is None or not np.isfinite(limit_flux_other_njy) \
            or limit_flux_other_njy <= 0:
        return GreyExclusion(False, False, nan, nan, other_band, "no_limit_other_band")
    predicted = abs(float(a_obs)) * float(base_flux_other_njy)
    limit = float(limit_flux_other_njy)
    if predicted <= margin * limit:
        # The grey hypothesis does not predict a detectable signal there, so the
        # silence says nothing.  Untestable, not "passed".
        return GreyExclusion(False, False, predicted, limit, other_band,
                             "prediction_below_detection_limit")
    return GreyExclusion(True, True, predicted, limit, other_band, "")
