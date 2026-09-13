"""The starspot energy ceiling --- pure functions, offline-testable.

The physics (docs/arc.md §3).  A flare releases magnetic energy stored in the
starspot field above the photosphere.  For a spot of area ``A`` threaded by a
field ``B``, the free energy available is bounded by

    E_mag = f * (B^2 / 8 pi) * A^(3/2)          (cgs; f <= 1)

(Shibayama et al. 2013 eq. 1; Okamoto et al. 2021 §5, who find the upper
envelope of Kepler superflare energies consistent with f ~ 0.1 and B = 3 kG).
The spot area itself follows from the star's rotational modulation amplitude
``dF/F`` (Notsu et al. 2013, 2019):

    A_spot = (dF/F) * pi R_*^2 / [1 - (T_spot / T_star)^4]

with the spot temperature from the Berdyugina (2005) relation

    T_star - T_spot = 3.58e-5 T_star^2 + 0.249 T_star - 808   [K]   (``verify``)

which gives ~1800 K for a solar-type photosphere.

Why the amplitude is a LOWER bound on the spot area.  A polar spot, or a
symmetric distribution of spots, modulates the light curve weakly or not at
all while storing the same magnetic energy; only a low-latitude spot group
that rotates in and out of view produces the full amplitude.  The channel
therefore marginalises over spot latitude with a geometric factor from config
(default: the true area may be up to 3x the amplitude-implied area) and
reports the residual at BOTH the nominal and the conservative area:

    xi = log10 E_flare - log10 E_mag(f = 1, B = 3 kG, A_conservative)

A star exceeds the ceiling only when ``xi_conservative > 0``: the flare
carried more energy than the largest spot the amplitude can hide, released
with perfect efficiency, in the strongest field seen in sunspot umbrae.
"""

from __future__ import annotations

import math

import numpy as np

R_SUN_CM = 6.957e10            # IAU 2015 nominal solar radius
T_SUN_K = 5772.0               # IAU 2015 nominal effective temperature
SIGMA_SB = 5.670374e-5         # erg cm^-2 s^-1 K^-4
L_SUN_ERG_S = 3.828e33
B_DEFAULT_G = 3000.0           # sunspot-umbra field, the value Okamoto+2021 adopt
DEFAULT_GEOMETRIC_FACTOR = 3.0

BERDYUGINA_COEFFS = (3.58e-5, 0.249, -808.0)     # verify: Berdyugina 2005, LRSP 2, 8


# ---------------------------------------------------------------------------
# the bound
# ---------------------------------------------------------------------------
def spot_temperature(t_star_k):
    """Berdyugina (2005) spot temperature for a photosphere of ``t_star_k`` (``verify``)."""
    t = np.asarray(t_star_k, dtype=float)
    a, b, c = BERDYUGINA_COEFFS
    dt = a * t ** 2 + b * t + c
    return np.clip(t - dt, 1.0, None)


def spot_contrast(t_star_k, t_spot_k=None):
    """``1 - (T_spot / T_star)^4``: the flux deficit per unit projected spot area."""
    t = np.asarray(t_star_k, dtype=float)
    ts = spot_temperature(t) if t_spot_k is None else np.asarray(t_spot_k, dtype=float)
    return 1.0 - (ts / t) ** 4


def spot_area(amplitude, radius_rsun, t_star_k, *, geometric_factor: float = 1.0):
    """Spot area [cm^2] implied by a rotational amplitude ``dF/F`` (fraction).

    ``geometric_factor`` marginalises over spot latitude: 1 is the nominal,
    face-on low-latitude spot; the config default of 3 is the conservative
    case (polar / symmetric spots hide up to three times the area).
    """
    amp = np.asarray(amplitude, dtype=float)
    r = np.asarray(radius_rsun, dtype=float) * R_SUN_CM
    return float(geometric_factor) * amp * math.pi * r ** 2 / spot_contrast(t_star_k)


def magnetic_energy(a_spot_cm2, *, b_gauss: float = B_DEFAULT_G, f: float = 1.0):
    """``f B^2/8pi A^{3/2}`` [erg]; the ceiling at ``f = 1``."""
    a = np.asarray(a_spot_cm2, dtype=float)
    return float(f) * float(b_gauss) ** 2 / (8.0 * math.pi) * np.power(a, 1.5)


def xi(e_flare_erg, e_mag_erg):
    """``log10 E_flare - log10 E_mag``; positive means above the ceiling."""
    e = np.asarray(e_flare_erg, dtype=float)
    m = np.asarray(e_mag_erg, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log10(e) - np.log10(m)
    return np.where(np.isfinite(out), out, np.nan)


# ---------------------------------------------------------------------------
# unit normalisation --- the catalogues do not agree on anything
# ---------------------------------------------------------------------------
def normalise_amplitude(values, unit: str = "auto") -> tuple[np.ndarray, str]:
    """Rotational amplitude to a flux FRACTION ``dF/F``.

    McQuillan+2014 ``Rper`` is in ppm; Okamoto+2021 / Notsu+2019 brightness
    variation amplitudes are fractions; Santos+2021 ``Sph`` is in ppm; some
    tables give per cent or mmag.  ``unit`` is the config's declaration; with
    ``auto`` the median decides (ppm above 100, per cent between 1 and 100,
    fraction below 1) and the choice is returned so the record can say so.
    """
    v = np.asarray(values, dtype=float)
    u = str(unit or "auto").lower()
    if u == "auto":
        fin = v[np.isfinite(v) & (v > 0)]
        if not len(fin):
            return v, "fraction"
        med = float(np.median(fin))
        u = "ppm" if med > 100.0 else ("percent" if med > 1.0 else "fraction")
    if u == "ppm":
        return v * 1e-6, "ppm"
    if u in ("percent", "%"):
        return v * 1e-2, "percent"
    if u == "mmag":
        return 1.0 - np.power(10.0, -0.4 * v * 1e-3), "mmag"
    if u == "mag":
        return 1.0 - np.power(10.0, -0.4 * v), "mag"
    return v, "fraction"


def bolometric_luminosity(radius_rsun, t_star_k):
    """``4 pi R^2 sigma T^4`` [erg/s]."""
    r = np.asarray(radius_rsun, dtype=float) * R_SUN_CM
    return 4.0 * math.pi * r ** 2 * SIGMA_SB * np.asarray(t_star_k, dtype=float) ** 4


def energy_to_bolometric(values, *, kind: str = "bolometric", factor: float = 1.0,
                         log10: str | bool = "auto", radius_rsun=None, t_star_k=None,
                         band_fraction: float = 0.35) -> tuple[np.ndarray, dict]:
    """Catalogue energy column to a bolometric flare energy in erg.

    ``kind``: ``bolometric`` (Okamoto, Tu, Shibayama, Yang & Liu --- their
    9000-10000 K blackbody integrals ARE the bolometric estimate), ``band``
    (a single-band energy; ``factor`` is the bolometric factor the paper
    states), or ``equivalent_duration`` (Davenport 2016: ED in seconds times
    the star's band luminosity, ``band_fraction`` of L_bol --- ``verify``).
    ``log10``: ``auto`` treats a column whose median is below 100 as log10.
    """
    v = np.asarray(values, dtype=float)
    info = {"kind": kind, "factor": float(factor)}
    fin = v[np.isfinite(v)]
    is_log = (bool(log10) if isinstance(log10, bool)
              else (len(fin) > 0 and float(np.median(fin)) < 100.0))
    info["log10_input"] = bool(is_log)
    e = np.power(10.0, v) if is_log else v.copy()
    if kind == "equivalent_duration":
        if radius_rsun is None or t_star_k is None:
            raise ValueError("equivalent_duration needs radius_rsun and t_star_k")
        lbol = bolometric_luminosity(radius_rsun, t_star_k)
        e = e * lbol * float(band_fraction)
        info["band_fraction"] = float(band_fraction)
    e = e * float(factor)
    return e, info


# ---------------------------------------------------------------------------
# per-star assessment
# ---------------------------------------------------------------------------
def independent_flares(t_peak, energies, gap_days: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Cluster events closer than ``gap_days`` into one flare (complex / sympathetic
    flares and late-phase re-brightenings are one energy release).  Returns the
    peak energy of each cluster and its first time; with no times every row is
    its own flare."""
    e = np.asarray(energies, dtype=float)
    if t_peak is None:
        return e, np.full(len(e), np.nan)
    t = np.asarray(t_peak, dtype=float)
    ok = np.isfinite(t)
    if not ok.any():
        return e, t
    order = np.argsort(t[ok])
    tt, ee = t[ok][order], e[ok][order]
    out_e, out_t = [], []
    start = 0
    for i in range(1, len(tt) + 1):
        if i == len(tt) or tt[i] - tt[i - 1] > gap_days:
            seg = ee[start:i]
            out_e.append(float(np.nanmax(seg)) if np.isfinite(seg).any() else np.nan)
            out_t.append(float(tt[start]))
            start = i
    # rows without a time are kept as independent events (they cannot be clustered)
    for x in e[~ok]:
        out_e.append(float(x))
        out_t.append(np.nan)
    return np.asarray(out_e), np.asarray(out_t)


def star_ceiling(energies_erg, t_peak, amplitude_frac: float, radius_rsun: float,
                 t_star_k: float, *, b_gauss: float = B_DEFAULT_G,
                 geometric_factor: float = DEFAULT_GEOMETRIC_FACTOR,
                 independent_gap_days: float = 0.5) -> dict:
    """The per-star record: ceiling at both areas, xi_max, and the count of
    independent flares above each.  NaN amplitude / radius / Teff gives a
    record with NaN xi and ``assessable = False`` --- never a silent zero."""
    e_ind, t_ind = independent_flares(t_peak, energies_erg, independent_gap_days)
    e_ind = np.asarray(e_ind, dtype=float)
    fin = np.isfinite(e_ind) & (e_ind > 0)
    rec = {
        "n_flares": int(len(np.asarray(energies_erg))),
        "n_independent": int(fin.sum()),
        "e_flare_max_erg": float(np.nanmax(e_ind[fin])) if fin.any() else float("nan"),
        "amplitude_frac": float(amplitude_frac), "radius_rsun": float(radius_rsun),
        "teff_k": float(t_star_k), "b_gauss": float(b_gauss),
        "geometric_factor": float(geometric_factor),
    }
    ok = (np.isfinite(amplitude_frac) and amplitude_frac > 0 and np.isfinite(radius_rsun)
          and radius_rsun > 0 and np.isfinite(t_star_k) and t_star_k > 0 and fin.any())
    rec["assessable"] = bool(ok)
    if not ok:
        rec.update({"t_spot_k": float("nan"), "a_spot_nominal_cm2": float("nan"),
                    "a_spot_conservative_cm2": float("nan"), "e_mag_nominal_erg": float("nan"),
                    "e_mag_conservative_erg": float("nan"), "xi_nominal_max": float("nan"),
                    "xi_conservative_max": float("nan"), "n_above_nominal": 0,
                    "n_above_conservative": 0, "flares_above": []})
        return rec
    a_nom = float(spot_area(amplitude_frac, radius_rsun, t_star_k))
    a_con = float(spot_area(amplitude_frac, radius_rsun, t_star_k,
                            geometric_factor=geometric_factor))
    m_nom = float(magnetic_energy(a_nom, b_gauss=b_gauss))
    m_con = float(magnetic_energy(a_con, b_gauss=b_gauss))
    x_nom = xi(e_ind, m_nom)
    x_con = xi(e_ind, m_con)
    above = [{"t_peak": (float(t_ind[i]) if np.isfinite(t_ind[i]) else None),
              "e_flare_erg": float(e_ind[i]), "xi_nominal": float(x_nom[i]),
              "xi_conservative": float(x_con[i])}
             for i in range(len(e_ind)) if fin[i] and x_con[i] > 0]
    rec.update({
        "t_spot_k": float(spot_temperature(t_star_k)),
        "a_spot_nominal_cm2": a_nom, "a_spot_conservative_cm2": a_con,
        "e_mag_nominal_erg": m_nom, "e_mag_conservative_erg": m_con,
        "xi_nominal_max": float(np.nanmax(x_nom[fin])),
        "xi_conservative_max": float(np.nanmax(x_con[fin])),
        "n_above_nominal": int(np.sum(x_nom[fin] > 0)),
        "n_above_conservative": int(np.sum(x_con[fin] > 0)),
        "flares_above": above,
    })
    return rec


def percentiles(values, qs=(1, 5, 16, 50, 84, 95, 99)) -> dict:
    v = np.asarray([x for x in np.asarray(values, dtype=float).ravel() if np.isfinite(x)])
    if not len(v):
        return {"n": 0}
    d = {"n": int(len(v)), "min": float(v.min()), "max": float(v.max())}
    d.update({f"p{q}": float(np.percentile(v, q)) for q in qs})
    d["n_positive"] = int((v > 0).sum())
    return d


__all__ = ["B_DEFAULT_G", "BERDYUGINA_COEFFS", "DEFAULT_GEOMETRIC_FACTOR", "L_SUN_ERG_S",
           "R_SUN_CM", "SIGMA_SB", "T_SUN_K", "bolometric_luminosity", "energy_to_bolometric",
           "independent_flares", "magnetic_energy", "normalise_amplitude", "percentiles",
           "spot_area", "spot_contrast", "spot_temperature", "star_ceiling", "xi"]
