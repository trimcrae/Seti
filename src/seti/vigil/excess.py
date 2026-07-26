"""Fractional infrared excess ``f = L_IR/L_star`` for VIGIL, W1/W2 only.

The channel's discriminator is a statement about *how much* excess accompanies
the variability, so the excess has to be measured with an honest error bar and,
crucially, with an honest statement of what W1/W2 alone can and cannot pin down.

What two bands can do
---------------------
With only W1 and W2 the dust temperature and the dust solid angle are strongly
degenerate: a small amount of hot dust and a larger amount of cooler dust give
nearly the same pair of fluxes.  Rather than pretend to fit both, this module
reports ``tau`` on a **grid of assumed dust temperatures** and carries the
resulting range as the uncertainty.  The band-level excess ``F_exc/F_phot`` is
model-free and is what the primary cut actually uses; ``tau`` is the physical
translation and is reported with its temperature dependence explicit.

The key conversion factor
-------------------------
For a blackbody radiator of temperature ``T_d`` around a star of temperature
``T_*``, the ratio of *in-band* excess flux to *in-band* photospheric flux is

``F_exc(b)/F_phot(b) = tau * R(b, T_d, T_*)``,   ``R = [B_nu(T_d)/T_d^4] / [B_nu(T_*)/T_*^4]``

:func:`band_ratio_factor` computes ``R``.  It is large --- at W2, a 600 K
radiator around a 5000 K star has ``R ~ 23`` --- because the dust radiates near
its Wien peak while the photosphere is on its Rayleigh-Jeans tail.  That factor
is the whole reason a *low* fractional excess is nevertheless detectable as
mid-IR variability: ``tau = 1e-3`` produces a ~2% band excess, and switching it
off is a ~25 mmag event, which NEOWISE visit means resolve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..photometry import BANDS, band_freq_hz, mag_to_flux_jy, planck_bnu

# Dust temperatures over which tau is evaluated.  The lower edge is set by what
# W1/W2 can constrain at all, the upper by grain sublimation.
T_DUST_GRID_K = (300.0, 450.0, 600.0, 850.0, 1200.0)
T_DUST_REF_K = 600.0            # near the W2 Wien peak: the band's best case

# A band excess must be measured to at least this precision before "low excess"
# means anything.  An unmeasured excess is NOT a low excess --- that confusion is
# how a search reports the objects it failed to characterise as candidates.
EXCESS_PRECISION_MAX = 0.06     # sigma on F_exc/F_phot; the decisive gate is the tau one


def band_ratio_factor(band: str, t_dust_k: float, t_star_k: float) -> float:
    """``R`` --- in-band excess-to-photosphere flux ratio per unit ``L_IR/L_star``."""
    nu = band_freq_hz(band)
    bd = float(planck_bnu(t_dust_k, nu))
    bs = float(planck_bnu(t_star_k, nu))
    if bs <= 0:
        return float("nan")
    return float((bd / t_dust_k**4) / (bs / t_star_k**4))


def max_amplitude_for_frac_excess(frac_excess: float) -> float:
    """Largest peak-to-peak fractional band amplitude a band excess ``f`` can produce.

    Take the extreme case: the radiator switches fully off half the time.  Then
    the band flux alternates between ``F_phot`` and ``F_phot(1 + 2f)``, whose mean
    is ``F_phot(1 + f)``, so

    ``A_max = 2f / (1 + f)``.

    **This is the form that matters, and its virtue is what is missing from it.**
    Written against the *band* excess rather than against ``tau``, the conversion
    factor ``R`` cancels --- and ``R`` runs from 2 to 27 over the plausible dust
    temperature range at fixed ``tau`` (see :func:`band_ratio_factor`), which is
    the single largest uncertainty in this channel.  So the modulation index built
    on this denominator is free of the dust temperature, of the stellar
    temperature, of the distance, and of the luminosity.  What remains is a ratio
    of two directly measured photometric quantities.

    It is also a falsifier: a star whose measured amplitude exceeds ``A_max`` is
    saying the variability is not the excess (a blend, a bad epoch, or an
    underestimated photosphere).
    """
    f = float(frac_excess)
    if not np.isfinite(f) or f <= 0:
        return 0.0
    return float(2.0 * f / (1.0 + f))


def max_amplitude_for_tau(tau: float, band: str, t_dust_k: float,
                          t_star_k: float) -> float:
    """:func:`max_amplitude_for_frac_excess` expressed against ``tau`` instead.

    Retained because the brief states the cut in ``L_IR/L_star``, so both forms
    are computed and compared on the runner.  It inherits the full temperature
    sensitivity of ``R``, which is exactly why it is the secondary form.
    """
    r = band_ratio_factor(band, t_dust_k, t_star_k)
    if not np.isfinite(r) or tau <= 0:
        return 0.0
    return max_amplitude_for_frac_excess(tau * r)


def tau_for_amplitude(amp: float, band: str, t_dust_k: float,
                      t_star_k: float) -> float:
    """Inverse of :func:`max_amplitude_for_tau`: the minimum excess an amplitude needs."""
    r = band_ratio_factor(band, t_dust_k, t_star_k)
    if not np.isfinite(r) or r <= 0 or amp <= 0:
        return float("nan")
    if amp >= 2.0:
        return float("inf")
    return float(amp / (r * (2.0 - amp)))


@dataclass
class ExcessMeasurement:
    """Band-level excess plus its translation into ``L_IR/L_star``."""

    band: str
    f_obs_jy: float
    f_phot_jy: float
    f_exc_jy: float
    f_exc_err_jy: float
    frac_excess: float          # F_exc / F_phot  (model-free, temperature-free)
    frac_excess_err: float
    frac_excess_upper: float    # +2 sigma: the quantity the "low excess" cut uses
    chi: float                  # significance of the band excess
    tau_ref: float              # L_IR/L_star at T_DUST_REF_K
    tau_upper: float            # +2 sigma at T_DUST_REF_K -- the brief's cut
    tau_lo: float               # min over the temperature grid
    tau_hi: float               # max over the temperature grid
    tau_err: float              # from photometric error alone, at T_DUST_REF_K
    t_star_k: float
    measured: bool

    def as_dict(self) -> dict:
        return asdict(self)


def measure_excess(band: str, obs_mag: float, obs_magerr: float,
                   phot_pred_jy: float, phot_pred_err_jy: float,
                   t_star_k: float, sys_floor: float = 0.03,
                   t_grid=T_DUST_GRID_K) -> ExcessMeasurement:
    """One band's excess against a predicted photosphere.

    ``sys_floor`` is a fractional systematic floor on the *predicted* photosphere
    (colour-locus scatter, model mismatch, extinction).  Without it the excess
    error is the photometric error alone, which is far too small: the dominant
    uncertainty on a 2% band excess is the photosphere prediction, not the
    NEOWISE magnitude.
    """
    bad = ExcessMeasurement(band=band, f_obs_jy=float("nan"), f_phot_jy=float("nan"),
                            f_exc_jy=float("nan"), f_exc_err_jy=float("nan"),
                            frac_excess=float("nan"), frac_excess_err=float("nan"),
                            frac_excess_upper=float("nan"),
                            chi=float("nan"), tau_ref=float("nan"),
                            tau_upper=float("nan"),
                            tau_lo=float("nan"), tau_hi=float("nan"),
                            tau_err=float("nan"), t_star_k=float(t_star_k),
                            measured=False)
    if not (np.isfinite(obs_mag) and np.isfinite(phot_pred_jy) and phot_pred_jy > 0
            and np.isfinite(t_star_k) and t_star_k > 0):
        return bad

    f_obs = float(mag_to_flux_jy(obs_mag, band))
    e_obs = float(0.4 * np.log(10.0) * f_obs * (obs_magerr if np.isfinite(obs_magerr)
                                                else 0.0))
    e_pred = float(np.hypot(phot_pred_err_jy if np.isfinite(phot_pred_err_jy) else 0.0,
                            sys_floor * phot_pred_jy))
    f_exc = f_obs - float(phot_pred_jy)
    e_exc = float(np.hypot(e_obs, e_pred))
    frac = f_exc / float(phot_pred_jy)
    frac_err = e_exc / float(phot_pred_jy)
    chi = f_exc / e_exc if e_exc > 0 else float("nan")

    taus = []
    for td in t_grid:
        r = band_ratio_factor(band, td, t_star_k)
        taus.append(frac / r if np.isfinite(r) and r > 0 else np.nan)
    taus = np.array(taus, dtype=float)
    r_ref = band_ratio_factor(band, T_DUST_REF_K, t_star_k)
    tau_ref = frac / r_ref if np.isfinite(r_ref) and r_ref > 0 else float("nan")
    tau_err = frac_err / r_ref if np.isfinite(r_ref) and r_ref > 0 else float("nan")

    frac_upper = float(frac + 2.0 * frac_err)
    return ExcessMeasurement(
        band=band, f_obs_jy=f_obs, f_phot_jy=float(phot_pred_jy), f_exc_jy=f_exc,
        f_exc_err_jy=e_exc, frac_excess=float(frac), frac_excess_err=float(frac_err),
        frac_excess_upper=frac_upper,
        chi=float(chi), tau_ref=float(tau_ref),
        tau_upper=float(frac_upper / r_ref) if np.isfinite(r_ref) and r_ref > 0
        else float("nan"),
        tau_lo=float(np.nanmin(taus)) if np.isfinite(taus).any() else float("nan"),
        tau_hi=float(np.nanmax(taus)) if np.isfinite(taus).any() else float("nan"),
        tau_err=float(tau_err), t_star_k=float(t_star_k), measured=True)


def photosphere_from_nir(j_mag: float, h_mag: float, ks_mag: float,
                         t_star_k: float, bands=("W1", "W2")) -> dict[str, float]:
    """Predict W1/W2 photospheric flux by anchoring a blackbody on 2MASS JHKs.

    A blackbody is a crude stellar atmosphere, but at 3-5 um for a star anchored
    at 1-2 um it is on the Rayleigh-Jeans tail in both places, where the model
    dependence is weak.  The residual model error is what ``sys_floor`` in
    :func:`measure_excess` absorbs.  Returns ``{}`` when the anchor photometry is
    unusable, so the caller must handle "no photosphere" rather than silently
    receive a zero.
    """
    nu = {}
    obs = {}
    for b, m in (("J", j_mag), ("H", h_mag), ("Ks", ks_mag)):
        if m is None or not np.isfinite(m):
            continue
        nu[b] = band_freq_hz(b)
        obs[b] = float(mag_to_flux_jy(m, b))
    if not obs or not np.isfinite(t_star_k) or t_star_k <= 0:
        return {}
    model = {b: float(planck_bnu(t_star_k, nu[b])) for b in obs}
    num = sum(obs[b] * model[b] for b in obs)
    den = sum(model[b] ** 2 for b in obs)
    if den <= 0:
        return {}
    omega = num / den                       # Jy per (SI B_nu) -- absorbs all constants
    return {b: float(omega * planck_bnu(t_star_k, band_freq_hz(b))) for b in bands}


def teff_from_colour(bp_rp: float) -> float:
    """Crude but monotone T_eff(BP-RP) for main-sequence colours.

    Used only where a spectroscopic T_eff is missing.  It feeds ``R``, which
    depends on T_* through the Rayleigh-Jeans tail and is therefore forgiving:
    a 300 K T_eff error moves ``R`` by a few percent, well inside the systematic
    floor already carried on the photosphere prediction.
    """
    c = float(bp_rp)
    if not np.isfinite(c):
        return float("nan")
    c = float(np.clip(c, -0.2, 4.0))
    # Ballpark Gaia DR3 main-sequence relation (Pecaut & Mamajek-like), smooth.
    return float(4600.0 / (0.92 + 0.35 * c) + 1200.0 * np.exp(-1.5 * max(c, 0.0)))


# Nominal band centres, as quoted in the WISE mission literature and in the
# programme's temperature-ceiling arithmetic.  The *effective* wavelengths in
# ``seti.photometry.BANDS`` differ slightly (3.3526 um for W1), which moves the
# W1 Wien temperature by ~1.4%; the nominal values are used here so the stated
# bound is the same number the rest of the programme quotes.
NOMINAL_LAMBDA_UM = {"W1": 3.4, "W2": 4.6, "W3": 12.0, "W4": 22.0}


def wien_peak_k(band: str, nominal: bool = True) -> float:
    """Dust temperature whose Wien peak falls in ``band`` --- the channel's ceiling.

    W1 -> 852 K, W2 -> 630 K.  NEOWISE, CatWISE2020 and the deep unWISE coadds
    carry no other band, so VIGIL is structurally blind below ~200 K, and W3/W4
    depth has been frozen since the 2010 cryogenic mission ended.  That is an
    instrumental bound, not a design choice, and it bounds the claim.
    """
    lam = (NOMINAL_LAMBDA_UM.get(band) if nominal else None) or \
        BANDS[band]["lambda_um"]
    return float(2.897771955e3 / lam)


__all__ = ["EXCESS_PRECISION_MAX", "T_DUST_GRID_K", "T_DUST_REF_K",
           "ExcessMeasurement", "band_ratio_factor",
           "max_amplitude_for_frac_excess", "max_amplitude_for_tau",
           "measure_excess", "photosphere_from_nir", "tau_for_amplitude",
           "teff_from_colour", "wien_peak_k"]
