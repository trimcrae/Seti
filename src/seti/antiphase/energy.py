"""The ANTIPHASE energy budget.  Pure.

If a structure absorbs a fraction ``f`` of a star's light and re-radiates it,
the optical flux lost at Earth (``f * F_bol`` for a grey occulter that covers
the star as seen from everywhere, which is what a shell or a swarm does) comes
back in the mid-infrared.  ``L_IR,gained / L_opt,lost`` is then ~1.  An
occulter that sits only on *our* line of sight (a companion's disk, a ring, a
single clump) takes light out of our beam without re-radiating a comparable
luminosity toward us, and the ratio falls far below 1 --- that is the natural
case the budget separates.

Units: fluxes at Earth in W m^-2 (bolometric) and Jy (flux densities).

* ``F_bol`` from Gaia G and the Andrae et al. (2018, A&A 616, A8) bolometric
  correction ``BC_G(Teff)`` (their Table 8 polynomial, valid 3300--8000 K),
  with ``m_bol = 0  <->  2.518021002e-8 W m^-2`` (IAU 2015 Resolution B2).
* The W1/W2 baseline flux densities from the star's own magnitudes and the
  Wright et al. (2010) / Jarrett et al. (2011) zero points (309.540 Jy,
  171.787 Jy at 3.3526 and 4.6028 um).
* The IR gain is a single-temperature blackbody through the two excess flux
  densities; the temperature is fitted on a grid and **every temperature the
  two points allow** (chi2 <= chi2_min + ``dchi2``) is carried into a range of
  ratios, because two bands cannot pin a cold component: a W2-only excess can
  be 300 K dust carrying a lot of luminosity or 800 K dust carrying little.
"""

from __future__ import annotations

import numpy as np

H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23
SIGMA_SB = 5.670374419e-8
F_BOL_ZERO = 2.518021002e-8          # W m^-2 at m_bol = 0
JY = 1e-26                           # W m^-2 Hz^-1

WISE_BANDS = {"W1": {"lam_um": 3.3526, "f0_jy": 309.540},
              "W2": {"lam_um": 4.6028, "f0_jy": 171.787}}

#: Andrae et al. 2018 BC_G coefficients a0..a4 in powers of (Teff - 5772 K).
_BC_HOT = (6.000e-02, 6.731e-05, -6.647e-08, 2.859e-11, -7.197e-15)     # 4000-8000 K
_BC_COOL = (1.749e+00, 1.977e-03, 3.737e-07, -8.966e-11, -4.183e-14)    # 3300-4000 K

#: Dwarf Teff against Gaia BP-RP (Pecaut & Mamajek 2013, as tabulated in the
#: Mamajek "modern mean dwarf colours" table).  Used only when the parent row
#: has no teff_gspphot; the BC it feeds varies by < 0.1 mag over +-300 K.
_BPRP_TEFF = ((0.40, 7200.0), (0.60, 6450.0), (0.75, 5950.0), (0.82, 5780.0),
              (0.95, 5400.0), (1.10, 5100.0), (1.30, 4700.0), (1.50, 4450.0),
              (1.70, 4150.0), (1.85, 3900.0), (2.10, 3650.0), (2.40, 3500.0),
              (2.80, 3250.0), (3.20, 3050.0), (3.80, 2850.0))


def teff_from_bprp(bp_rp: float) -> float:
    x = np.array([p[0] for p in _BPRP_TEFF])
    y = np.array([p[1] for p in _BPRP_TEFF])
    if not np.isfinite(bp_rp):
        return float("nan")
    return float(np.interp(float(bp_rp), x, y))


def star_teff(teff_gspphot, bp_rp) -> tuple[float, str]:
    """Teff and where it came from (``gspphot`` or ``bp_rp``)."""
    try:
        t = float(teff_gspphot)
    except (TypeError, ValueError):
        t = float("nan")
    if np.isfinite(t) and 2500.0 < t < 12000.0:
        return t, "gspphot"
    return teff_from_bprp(float(bp_rp) if bp_rp is not None else float("nan")), "bp_rp"


def bc_g(teff: float) -> float:
    """Andrae et al. 2018 BC_G, clipped to its 3300--8000 K validity range."""
    if not np.isfinite(teff):
        return float("nan")
    t = float(np.clip(teff, 3300.0, 8000.0))
    a = _BC_HOT if t >= 4000.0 else _BC_COOL
    d = t - 5772.0
    return float(sum(c * d ** i for i, c in enumerate(a)))


def f_bol(g_mag: float, teff: float) -> float:
    """Bolometric flux at Earth (W m^-2) from G and BC_G(Teff)."""
    bc = bc_g(teff)
    if not (np.isfinite(g_mag) and np.isfinite(bc)):
        return float("nan")
    return float(F_BOL_ZERO * 10.0 ** (-0.4 * (float(g_mag) + bc)))


def wise_fnu_jy(mag: float, band: str) -> float:
    return float(WISE_BANDS[band]["f0_jy"] * 10.0 ** (-0.4 * float(mag)))


def planck_nu(nu, t):
    """B_nu(T) in W m^-2 Hz^-1 sr^-1."""
    nu = np.asarray(nu, float)
    x = np.clip(H * nu / (KB * float(t)), 1e-12, 700.0)
    return 2.0 * H * nu ** 3 / C ** 2 / np.expm1(x)


def excess_fnu(base_mag: float, dmag: float, dmag_err: float, band: str) -> tuple[float, float]:
    """Excess flux density (Jy) of a change ``dmag`` (negative = brighter) on ``base_mag``."""
    f0 = wise_fnu_jy(base_mag, band)
    ratio = 10.0 ** (-0.4 * float(dmag))
    ex = f0 * (ratio - 1.0)
    err = f0 * ratio * 0.4 * np.log(10.0) * float(dmag_err)
    return float(ex), float(abs(err))


def fit_blackbody(ex_jy: dict, err_jy: dict, t_grid=None, dchi2: float = 2.3) -> dict:
    """Single-temperature blackbody through the W1/W2 excess flux densities.

    Returns the best T, the allowed T range, the chi2 at the best T, and per
    allowed T the bolometric excess flux ``F_IR = Omega * sigma T^4 / pi``
    (W m^-2) --- as ``f_ir_best`` / ``f_ir_lo`` / ``f_ir_hi``.  Needs a
    positive excess in at least one band; the solid angle is constrained >= 0.
    """
    if t_grid is None:
        t_grid = np.geomspace(100.0, 3000.0, 240)
    bands = [b for b in ("W1", "W2") if b in ex_jy and np.isfinite(ex_jy[b])
             and np.isfinite(err_jy.get(b, np.nan)) and err_jy[b] > 0]
    out = {"status": "OK", "t_best_k": float("nan"), "t_lo_k": float("nan"),
           "t_hi_k": float("nan"), "chi2_min": float("nan"), "f_ir_best": float("nan"),
           "f_ir_lo": float("nan"), "f_ir_hi": float("nan"), "n_bands": len(bands)}
    if not bands:
        out["status"] = "NO_IR_EXCESS_DATA"
        return out
    y = np.array([ex_jy[b] * JY for b in bands])
    s = np.array([err_jy[b] * JY for b in bands])
    if not np.any(y > 0):
        out["status"] = "NO_POSITIVE_EXCESS"
        return out
    nu = np.array([C / (WISE_BANDS[b]["lam_um"] * 1e-6) for b in bands])
    chi2 = np.empty(len(t_grid))
    fir = np.empty(len(t_grid))
    for i, t in enumerate(t_grid):
        m = planck_nu(nu, t)
        w = 1.0 / s ** 2
        omega = max(float(np.sum(w * m * y) / np.sum(w * m * m)), 0.0)
        chi2[i] = float(np.sum(((y - omega * m) / s) ** 2))
        fir[i] = omega * SIGMA_SB * float(t) ** 4 / np.pi
    k = int(np.argmin(chi2))
    ok = chi2 <= chi2[k] + float(dchi2)
    ok &= fir > 0
    out.update({"t_best_k": float(t_grid[k]), "chi2_min": float(chi2[k]),
                "f_ir_best": float(fir[k])})
    if ok.any():
        out.update({"t_lo_k": float(np.min(np.asarray(t_grid)[ok])),
                    "t_hi_k": float(np.max(np.asarray(t_grid)[ok])),
                    "f_ir_lo": float(np.min(fir[ok])), "f_ir_hi": float(np.max(fir[ok]))})
    if len(bands) == 2 and ex_jy["W1"] > 0 and ex_jy["W2"] > 0:
        # Rayleigh-Jeans limit: F_nu(W1)/F_nu(W2) <= (4.6028/3.3526)^2 = 1.885
        out["ratio_w1_w2_fnu"] = float(ex_jy["W1"] / ex_jy["W2"])
    return out


def energy_budget(g_mag: float, teff: float, frac_lost: float, frac_lost_err: float,
                  base_mag: dict, dmag: dict, dmag_err: dict, conf: dict | None = None) -> dict:
    """``L_IR,gained / L_opt,lost`` with its temperature-allowed range.

    ``frac_lost`` is the fractional optical flux deficit (grey: the same in
    every band, so it is also the bolometric fraction).  ``base_mag`` /
    ``dmag`` / ``dmag_err`` are per WISE band.  Verdicts:

    * ``BALANCED`` --- the allowed ratio range meets [1/``factor``, ``factor``];
    * ``IR_DEFICIT`` --- even the largest allowed IR luminosity is < 1/factor of
      what was lost (an occulter on our line of sight only);
    * ``IR_SURPLUS`` --- even the smallest is > factor x the loss;
    * ``STELLAR_TEMPERATURE`` --- the best-fit T exceeds grain survival
      (``t_max_grain_k``): a companion photosphere or a blend, not dust;
    * ``UNDETERMINED`` --- no usable IR excess or no F_bol.
    """
    c = {"factor": 3.0, "t_max_grain_k": 1800.0, "dchi2": 2.3, **(conf or {})}
    fb = f_bol(g_mag, teff)
    ex, er = {}, {}
    for b in ("W1", "W2"):
        if b in dmag and np.isfinite(dmag.get(b, np.nan)) and np.isfinite(base_mag.get(b, np.nan)):
            ex[b], er[b] = excess_fnu(base_mag[b], dmag[b], dmag_err.get(b, np.nan), b)
    bb = fit_blackbody(ex, er, dchi2=float(c["dchi2"]))
    lost = float(frac_lost) * fb if np.isfinite(fb) and np.isfinite(frac_lost) else float("nan")
    rec = {"f_bol_w_m2": fb, "frac_lost": float(frac_lost), "frac_lost_err": float(frac_lost_err),
           "f_opt_lost_w_m2": lost, "excess_jy": ex, "excess_err_jy": er, "blackbody": bb,
           "ratio_best": float("nan"), "ratio_lo": float("nan"), "ratio_hi": float("nan")}
    if not (np.isfinite(lost) and lost > 0) or bb["status"] != "OK":
        rec["verdict"] = "UNDETERMINED"
        return rec
    rec["ratio_best"] = bb["f_ir_best"] / lost
    rec["ratio_lo"] = bb["f_ir_lo"] / lost if np.isfinite(bb["f_ir_lo"]) else float("nan")
    rec["ratio_hi"] = bb["f_ir_hi"] / lost if np.isfinite(bb["f_ir_hi"]) else float("nan")
    fac = float(c["factor"])
    lo = rec["ratio_lo"] if np.isfinite(rec["ratio_lo"]) else rec["ratio_best"]
    hi = rec["ratio_hi"] if np.isfinite(rec["ratio_hi"]) else rec["ratio_best"]
    if bb["t_best_k"] > float(c["t_max_grain_k"]) and \
            (not np.isfinite(bb["t_lo_k"]) or bb["t_lo_k"] > float(c["t_max_grain_k"])):
        rec["verdict"] = "STELLAR_TEMPERATURE"
    elif hi < 1.0 / fac:
        rec["verdict"] = "IR_DEFICIT"
    elif lo > fac:
        rec["verdict"] = "IR_SURPLUS"
    else:
        rec["verdict"] = "BALANCED"
    rec["t_range_unconstrained"] = bool(np.isfinite(hi) and np.isfinite(lo) and lo > 0
                                        and hi / lo > 10.0)
    return rec


__all__ = ["WISE_BANDS", "bc_g", "energy_budget", "excess_fnu", "f_bol", "fit_blackbody",
           "planck_nu", "star_teff", "teff_from_bprp", "wise_fnu_jy"]
