"""Flare physics and the flare finder for ARC stage 2 --- pure, offline-testable.

Stage 1 works from catalogued flare energies.  Stage 2 has to find the same
flare *in the light curve* (the Yang & Liu 2019 table carries a quarter and a
start / end time but no peak time; Tu+2022 carries a sector and nothing else),
measure it again with one method, and re-measure the rotational amplitude in
the same quarter --- spots evolve, and the amplitude stage 1 used may be from
another epoch.

What is here
------------
* the Kepler response function (a coarse tabulation, ``verify``) and the
  Planck integrals that turn a relative flare flux into a flare area and a
  bolometric energy the way Shibayama et al. (2013, ApJS 209, 5, §3.3) and
  Yang & Liu (2019) do: ``A_flare(t) = C'(t) pi R^2 int R B(T_eff) / int R B(T_fl)``,
  ``L_flare = sigma T_fl^4 A_flare``, ``E = int L_flare dt``, with
  ``T_fl = 9000 K``;
* the band-luminosity form ``E_band = ED x L_band`` as a second, simpler estimate;
* an iterative running-median detrender and a run-of-outliers flare finder
  (Chang et al. 2015 style: consecutive points above n sigma, then the window
  extended down to 1 sigma on both sides);
* matching of catalogued flares to detected ones --- by time overlap when the
  catalogue gives a window, by energy rank within the quarter / sector when
  it does not, with the method recorded;
* the quarter's own rotational amplitude (5th-95th percentile range of the
  flare-free baseline, the McQuillan ``Rper`` definition, and the standard
  deviation, the Santos ``Sph`` definition) so stage 1's catalogue amplitude
  can be checked against the epoch of the flare.
"""

from __future__ import annotations

import math

import numpy as np

from .ceiling import R_SUN_CM, SIGMA_SB, bolometric_luminosity

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # noqa: B009  numpy 1.x / 2.x

H_PLANCK = 6.62607015e-27       # erg s
C_LIGHT = 2.99792458e10         # cm/s
K_BOLTZ = 1.380649e-16          # erg/K
T_FLARE_K_DEFAULT = 9000.0      # Shibayama+2013 / Yang & Liu 2019 white-light flare temperature

#: The Kepler photometric response (dimensionless throughput vs wavelength in
#: nm), a coarse tabulation of the Kepler Instrument Handbook curve
#: (``verify``: asserted from memory of the published figure, not read from the
#: calibration file).  The energy estimate depends on it only through the
#: ratio ``int R B(T_eff) / int R B(T_fl)``, which moves by a few per cent for
#: any plausible reshaping of the curve; the absolute band luminosity used by
#: the ``E_band`` estimate is more sensitive and is reported beside the
#: Shibayama value rather than instead of it.
KEPLER_RESPONSE_NM: tuple[tuple[float, float], ...] = (
    (400.0, 0.00), (420.0, 0.03), (430.0, 0.20), (440.0, 0.45), (450.0, 0.60),
    (470.0, 0.72), (500.0, 0.78), (550.0, 0.82), (600.0, 0.85), (650.0, 0.84),
    (700.0, 0.80), (750.0, 0.72), (800.0, 0.60), (830.0, 0.50), (850.0, 0.40),
    (870.0, 0.25), (890.0, 0.10), (900.0, 0.03), (920.0, 0.00),
)
#: The TESS response, likewise coarse (600-1000 nm, red-optical; ``verify``).
TESS_RESPONSE_NM: tuple[tuple[float, float], ...] = (
    (580.0, 0.00), (600.0, 0.40), (620.0, 0.75), (650.0, 0.85), (700.0, 0.88),
    (750.0, 0.88), (800.0, 0.85), (850.0, 0.78), (900.0, 0.65), (950.0, 0.45),
    (1000.0, 0.20), (1050.0, 0.05), (1100.0, 0.00),
)


# ---------------------------------------------------------------------------
# blackbody integrals in a band
# ---------------------------------------------------------------------------
def planck_lambda(wavelength_cm, t_k: float):
    """``B_lambda(T)`` in erg s^-1 cm^-2 cm^-1 sr^-1."""
    lam = np.asarray(wavelength_cm, dtype=float)
    x = H_PLANCK * C_LIGHT / (lam * K_BOLTZ * float(t_k))
    with np.errstate(over="ignore"):
        return 2.0 * H_PLANCK * C_LIGHT ** 2 / lam ** 5 / np.expm1(x)


def response_curve(mission: str = "kepler") -> tuple[np.ndarray, np.ndarray]:
    tab = TESS_RESPONSE_NM if str(mission).lower().startswith("tess") else KEPLER_RESPONSE_NM
    lam = np.array([p[0] for p in tab], dtype=float)
    r = np.array([p[1] for p in tab], dtype=float)
    grid = np.arange(lam.min(), lam.max() + 0.5, 1.0)
    return grid, np.interp(grid, lam, r)


def band_integral(t_k: float, mission: str = "kepler") -> float:
    """``int R_lambda B_lambda(T) dlambda`` [erg s^-1 cm^-2 sr^-1] over the band."""
    lam_nm, r = response_curve(mission)
    lam_cm = lam_nm * 1e-7
    b = planck_lambda(lam_cm, float(t_k))
    return float(_trapz(r * b, lam_cm))


def band_luminosity(radius_rsun: float, t_k: float, mission: str = "kepler") -> float:
    """The star's luminosity through the band: ``4 pi R^2 * pi * int R B dlambda`` [erg/s]."""
    r_cm = float(radius_rsun) * R_SUN_CM
    return 4.0 * math.pi * r_cm ** 2 * math.pi * band_integral(t_k, mission)


def band_fraction(radius_rsun: float, t_k: float, mission: str = "kepler") -> float:
    """``L_band / L_bol`` --- the sanity number beside ``config physics.band_fraction_kepler``."""
    lb = float(bolometric_luminosity(radius_rsun, t_k))
    return band_luminosity(radius_rsun, t_k, mission) / lb if lb > 0 else float("nan")


def flare_energy_shibayama(time_days, rel_excess, *, teff_k: float, radius_rsun: float,
                           t_flare_k: float = T_FLARE_K_DEFAULT, mission: str = "kepler",
                           cadence_days: float | None = None) -> dict:
    """Bolometric flare energy from the relative flare flux ``C'(t) = F_fl / F_star``.

    Shibayama et al. 2013 eq. (2)-(4): the flare is a ``T_fl`` blackbody whose
    area ``A_fl(t)`` reproduces the observed band excess against the star's
    own blackbody through the same response; its bolometric luminosity is
    ``sigma T_fl^4 A_fl`` and the energy is the time integral.  ``rel_excess``
    is ``F/F_0 - 1`` per cadence over the flare; the cadence is the exposure.
    """
    t = np.asarray(time_days, dtype=float)
    c = np.asarray(rel_excess, dtype=float)
    ok = np.isfinite(t) & np.isfinite(c)
    t, c = t[ok], np.clip(c[ok], 0.0, None)
    if not t.size:
        return {"e_bol_erg": float("nan"), "ed_s": float("nan"), "a_flare_max_cm2": float("nan"),
                "n_cadences": 0}
    if cadence_days is None:
        cadence_days = float(np.median(np.diff(np.sort(t)))) if t.size > 1 else float("nan")
    dt_s = float(cadence_days) * 86400.0
    ratio = band_integral(teff_k, mission) / band_integral(t_flare_k, mission)
    r_cm = float(radius_rsun) * R_SUN_CM
    a_fl = c * math.pi * r_cm ** 2 * ratio
    l_fl = SIGMA_SB * float(t_flare_k) ** 4 * a_fl
    return {"e_bol_erg": float(np.sum(l_fl) * dt_s), "ed_s": float(np.sum(c) * dt_s),
            "a_flare_max_cm2": float(a_fl.max()), "n_cadences": int(t.size),
            "band_ratio": float(ratio), "t_flare_k": float(t_flare_k)}


def flare_energy_band(ed_s: float, *, teff_k: float, radius_rsun: float,
                      mission: str = "kepler") -> float:
    """``ED x L_band``: the energy radiated in the band (Davenport 2016 style)."""
    if not (np.isfinite(ed_s) and ed_s >= 0):
        return float("nan")
    return float(ed_s) * band_luminosity(radius_rsun, teff_k, mission)


# ---------------------------------------------------------------------------
# detrending and the finder
# ---------------------------------------------------------------------------
def robust_sigma(x) -> float:
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    med = float(np.median(v))
    return float(1.4826 * np.median(np.abs(v - med)))


def running_median(time, flux, window_days: float, mask=None) -> np.ndarray:
    """Running median of ``flux`` over ``+-window/2`` in time, from the unmasked
    points only; a window with no usable point falls back to the global median."""
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    use = np.isfinite(t) & np.isfinite(f)
    if mask is not None:
        use &= ~np.asarray(mask, dtype=bool)
    order = np.argsort(t)
    ts, fs, us = t[order], f[order], use[order]
    out = np.full(t.shape, np.nan)
    half = 0.5 * float(window_days)
    glob = float(np.median(fs[us])) if us.any() else float("nan")
    lo_idx = np.searchsorted(ts, ts - half, side="left")
    hi_idx = np.searchsorted(ts, ts + half, side="right")
    res = np.empty(ts.shape)
    for i in range(ts.size):
        seg = fs[lo_idx[i]:hi_idx[i]][us[lo_idx[i]:hi_idx[i]]]
        res[i] = float(np.median(seg)) if seg.size else glob
    out[order] = res
    return out


def detrend(time, flux, *, window_days: float = 0.5, n_iter: int = 3, clip_sigma: float = 3.0
            ) -> tuple[np.ndarray, np.ndarray, float]:
    """Iterated running-median baseline: ``(baseline, residual, sigma)`` with
    ``residual = flux / baseline - 1`` and ``sigma`` the MAD scatter of the
    residual outside the clipped (flaring) points.  Upward outliers are masked
    between iterations so a flare does not lift its own baseline."""
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    mask = np.zeros(t.shape, dtype=bool)
    base = running_median(t, f, window_days)
    sig = float("nan")
    for _ in range(max(int(n_iter), 1)):
        with np.errstate(divide="ignore", invalid="ignore"):
            resid = f / base - 1.0
        sig = robust_sigma(resid[~mask])
        if not np.isfinite(sig) or sig <= 0:
            break
        new_mask = np.isfinite(resid) & (resid > float(clip_sigma) * sig)
        # a masked point's neighbours are masked too: the decay tail of a
        # flare is below the clip and would otherwise pull the median up
        grown = new_mask.copy()
        grown[1:] |= new_mask[:-1]
        grown[:-1] |= new_mask[1:]
        if np.array_equal(grown, mask):
            break
        mask = grown
        base = running_median(t, f, window_days, mask=mask)
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = f / base - 1.0
    sig = robust_sigma(resid[~mask]) if np.isfinite(sig) else robust_sigma(resid)
    return base, resid, float(sig)


def _runs(idx: np.ndarray, time: np.ndarray, max_gap_days: float) -> list[tuple[int, int]]:
    """Maximal runs of consecutive indices whose time gaps stay below ``max_gap``."""
    if not idx.size:
        return []
    runs, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:], strict=True):
        if b != a + 1 or (time[b] - time[a]) > max_gap_days:
            runs.append((int(start), int(a)))
            start = b
    runs.append((int(start), int(idx[-1])))
    return runs


def detect_flares(time, flux, *, window_days: float = 0.5, n_sigma: float = 3.0,
                  min_consecutive: int = 2, peak_sigma: float = 4.0, extend_sigma: float = 1.0,
                  cadence_days: float | None = None, quality=None) -> tuple[list[dict], dict]:
    """Flares as runs of ``>= min_consecutive`` consecutive points above
    ``n_sigma`` with a peak above ``peak_sigma``, the window extended down to
    ``extend_sigma`` on both sides.  Returns ``(flares, info)``; each flare
    carries its indices, times, peak amplitude, equivalent duration (s), the
    rise and decay times and the point count.  A cadence with a non-zero
    ``quality`` flag is excluded from the finder entirely.
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    good = np.isfinite(t) & np.isfinite(f)
    if quality is not None:
        q = np.asarray(quality)
        good &= (q == 0)
    order = np.argsort(t)
    t, f, good = t[order], f[order], good[order]
    tt, ff = t[good], f[good]
    info = {"n_points": int(tt.size), "sigma": float("nan"), "cadence_days": float("nan"),
            "n_flares": 0}
    if tt.size < 20:
        return [], info
    if cadence_days is None:
        cadence_days = float(np.median(np.diff(tt)))
    info["cadence_days"] = float(cadence_days)
    base, resid, sig = detrend(tt, ff, window_days=window_days, clip_sigma=n_sigma)
    info["sigma"] = float(sig)
    if not (np.isfinite(sig) and sig > 0):
        return [], info
    hi = np.where(np.isfinite(resid) & (resid > float(n_sigma) * sig))[0]
    flares = []
    used = np.zeros(tt.shape, dtype=bool)
    for a, b in _runs(hi, tt, 2.5 * cadence_days):
        if b - a + 1 < int(min_consecutive):
            continue
        seg = resid[a:b + 1]
        if float(np.nanmax(seg)) < float(peak_sigma) * sig:
            continue
        ip = a + int(np.nanargmax(seg))
        s, e = a, b
        while s > 0 and np.isfinite(resid[s - 1]) and resid[s - 1] > float(extend_sigma) * sig \
                and (tt[s] - tt[s - 1]) <= 2.5 * cadence_days:
            s -= 1
        while e < tt.size - 1 and np.isfinite(resid[e + 1]) \
                and resid[e + 1] > float(extend_sigma) * sig \
                and (tt[e + 1] - tt[e]) <= 2.5 * cadence_days:
            e += 1
        if used[s:e + 1].any():
            continue
        used[s:e + 1] = True
        exc = np.clip(resid[s:e + 1], 0.0, None)
        ed_s = float(np.sum(exc) * cadence_days * 86400.0)
        flares.append({
            "i_start": int(s), "i_peak": int(ip), "i_end": int(e),
            "t_start": float(tt[s]), "t_peak": float(tt[ip]), "t_end": float(tt[e]),
            "amplitude": float(resid[ip]), "peak_sigma": float(resid[ip] / sig),
            "ed_s": ed_s, "n_points": int(e - s + 1),
            "n_above_n_sigma": int(b - a + 1),
            "t_rise_days": float(tt[ip] - tt[s]), "t_decay_days": float(tt[e] - tt[ip]),
            "rel_excess": [float(x) for x in exc],
            "times": [float(x) for x in tt[s:e + 1]],
        })
    info["n_flares"] = len(flares)
    info["baseline_median"] = float(np.nanmedian(base))
    return flares, info


# ---------------------------------------------------------------------------
# matching catalogued flares to detected ones
# ---------------------------------------------------------------------------
MATCH_TIME = "time_overlap"
MATCH_RANK = "energy_rank"
MATCH_NONE = "unmatched"


def match_flares(catalogue_rows: list[dict], detected: list[dict], *,
                 cadence_days: float, pad_cadences: float = 1.5) -> list[dict]:
    """Attach a detected flare to each catalogue row.

    A row with a finite ``t_start`` / ``t_end`` (or ``t_peak``) is matched by
    time overlap, padded by ``pad_cadences``; rows without times are matched
    to the remaining detected flares by **energy rank** (largest catalogued
    energy to largest detected equivalent duration), which is the only
    information the catalogue leaves.  Each returned row carries ``match``
    (the detected flare or ``None``), ``match_method`` and, for a rank match,
    the ``rank`` used.
    """
    pad = float(pad_cadences) * float(cadence_days)
    out = [dict(r) for r in catalogue_rows]
    taken: set[int] = set()

    def _t(r, k):
        try:
            v = float(r.get(k))
        except (TypeError, ValueError):
            return float("nan")
        return v

    # 1. by time
    for r in out:
        ts, te, tp = _t(r, "t_start"), _t(r, "t_end"), _t(r, "t_peak")
        if not np.isfinite(ts) and np.isfinite(tp):
            ts = tp
        if not np.isfinite(te) and np.isfinite(ts):
            te = ts + max(float(cadence_days), 0.0)
        r["match"], r["match_method"], r["rank"] = None, MATCH_NONE, None
        if not (np.isfinite(ts) and np.isfinite(te)):
            continue
        best, best_ov = None, 0.0
        for j, d in enumerate(detected):
            if j in taken:
                continue
            lo, hi = max(ts - pad, d["t_start"] - pad), min(te + pad, d["t_end"] + pad)
            ov = hi - lo
            if ov > 0 and ov > best_ov:
                best, best_ov = j, ov
        if best is not None:
            taken.add(best)
            r["match"], r["match_method"] = detected[best], MATCH_TIME
    # 2. by energy rank among the rest
    rest = [r for r in out if r["match"] is None and not np.isfinite(_t(r, "t_start"))
            and not np.isfinite(_t(r, "t_peak"))]
    rest.sort(key=lambda r: -(_t(r, "energy_erg") if np.isfinite(_t(r, "energy_erg")) else -1))
    free = sorted((j for j in range(len(detected)) if j not in taken),
                  key=lambda j: -float(detected[j].get("ed_s", 0.0)))
    for k, r in enumerate(rest):
        if k >= len(free):
            break
        r["match"], r["match_method"], r["rank"] = detected[free[k]], MATCH_RANK, k
        taken.add(free[k])
    return out


# ---------------------------------------------------------------------------
# the quarter's own rotational amplitude
# ---------------------------------------------------------------------------
def rotational_amplitude(time, flux, *, flare_mask=None, smooth_days: float = 0.5,
                         quality=None) -> dict:
    """``Rvar`` (5th-95th percentile range, McQuillan+2014 ``Rper``) and ``Sph``
    (standard deviation, Santos+2021) of the flare-free, lightly smoothed,
    normalised light curve --- the amplitude the spot model needs, measured in
    the quarter of the flare rather than taken from a catalogue epoch.  For a
    sinusoid ``Rvar = 2 sqrt(2) Sph`` to within the percentile clipping."""
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    good = np.isfinite(t) & np.isfinite(f)
    if quality is not None:
        good &= (np.asarray(quality) == 0)
    if flare_mask is not None:
        good &= ~np.asarray(flare_mask, dtype=bool)
    out = {"rvar": float("nan"), "sph": float("nan"), "n_points": int(good.sum()),
           "median_flux": float("nan")}
    if good.sum() < 50:
        return out
    tt, ff = t[good], f[good]
    med = float(np.median(ff))
    if not (np.isfinite(med) and med > 0):
        return out
    sm = running_median(tt, ff, smooth_days) / med
    out.update({"rvar": float(np.percentile(sm, 95) - np.percentile(sm, 5)),
                "sph": float(np.std(sm)), "median_flux": med})
    return out


__all__ = ["KEPLER_RESPONSE_NM", "MATCH_NONE", "MATCH_RANK", "MATCH_TIME", "T_FLARE_K_DEFAULT",
           "TESS_RESPONSE_NM", "band_fraction", "band_integral", "band_luminosity",
           "detect_flares", "detrend", "flare_energy_band", "flare_energy_shibayama",
           "match_flares", "planck_lambda", "response_curve", "robust_sigma",
           "rotational_amplitude", "running_median"]
