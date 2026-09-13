"""The depth-drift statistic for GROWTH: pure functions, offline-testable.

For one planet joined across the two epochs the observable is the depth ratio

    R = depth_TESS / depth_Kepler

and the question is whether ``ln R``, after the deterministic corrections, is
different from zero at fixed impact parameter.  The corrections:

(a) **limb-darkening band ratio.**  In the small-planet limit the observed
    depth is ``k^2 * I(mu) / <I>`` with ``mu = sqrt(1 - b^2)`` and the
    quadratic law ``I(mu)/I(1) = 1 - u1 (1 - mu) - u2 (1 - mu)^2``,
    ``<I>/I(1) = 1 - u1/3 - u2/6`` (Mandel & Agol 2002; Csizmadia et al.
    2013 for the small-planet form).  The Kepler and TESS bands have different
    ``(u1, u2)`` so the *same* ``k`` gives a different depth in each: the band
    ratio ``f_LD = F_TESS(b) / F_Kepler(b)`` is divided out.  Coefficients
    come from a small Teff grid in ``config/growth.yaml`` (Claret-style;
    marked ``verify`` there) and are interpolated in Teff at the nearest
    ``logg`` grid.

(b) **dilution.**  The TESS depth is divided by ``(1 - c)`` with
    ``c = sum_neighbours 10^(-0.4 dG) * w(sep)`` over Gaia DR3 neighbours
    inside the search radius, ``w`` an aperture-capture weight from config.
    The Kepler side is taken as deblended by the pipeline's own flux fraction
    (the DV depths are fitted on PDCSAP flux, which carries the CROWDSAP
    correction); the SPOC TOI depths carry an analogous correction, so the
    Gaia term may double-count --- which is why the *neighbour* veto asks the
    growth to survive the full range of the dilution ambiguity (§ ``vet``).

(c) **corrected log ratio** ``ln R_corr = ln D_T - ln(1 - c) - ln D_K - ln f_LD``
    with ``sigma^2 = (e_T/D_T)^2 + (e_K/D_K)^2 + sigma_sys^2``; catalogue
    errors are symmetrised as ``max(|err1|, |err2|)``.

(d) **duration consistency.**  A depth change by ``R`` at fixed ``b`` is a
    radius-ratio change ``k -> k sqrt(R)``, and the total duration changes as

        T14' / T14 = sqrt(((1 + k')^2 - b^2) / ((1 + k)^2 - b^2))

    (small-angle; the full ``arcsin`` form is used when ``a/R*`` is known).
    The observed TESS/Kepler duration ratio is compared with that prediction;
    a duration ratio that disagrees is instead solved for the ``b`` that
    would produce it at fixed ``k`` --- if one exists, the change *tracks a
    changed b* (nodal precession, or a grazing geometry) and is not growth.

(e) **classification.**  ``CONSISTENT`` (|ln R| < n_consistent sigma),
    ``DEEPER_TESS`` / ``SHALLOWER_TESS`` (beyond it), and ``GROWTH_CANDIDATE``
    when ln R > 0 at >= n_candidate sigma AND the duration verdict is
    ``fixed_b`` AND no veto in :mod:`seti.growth.vet` fired.  Because the
    expected sign of the whole population is a ~12 % deficit (Han et al.
    2025), the population median ln-ratio can be subtracted before
    classification (``subtract_population_median``); it is always reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import erfc

CLASS_CONSISTENT = "CONSISTENT"
CLASS_DEEPER = "DEEPER_TESS"
CLASS_SHALLOWER = "SHALLOWER_TESS"
CLASS_CANDIDATE = "GROWTH_CANDIDATE"
CLASS_UNMEASURED = "UNMEASURED"
CLASSES = (CLASS_CONSISTENT, CLASS_DEEPER, CLASS_SHALLOWER, CLASS_CANDIDATE, CLASS_UNMEASURED)

DUR_FIXED_B = "fixed_b"
DUR_TRACKS_B = "tracks_b"
DUR_INCONSISTENT = "duration_inconsistent"
DUR_INCONCLUSIVE = "inconclusive"

#: Fallback quadratic coefficients (solar-type star, logg 4.5) if the config
#: grid is absent.  Values are the same order as the config table and are
#: only a guard against a missing file; the config is the source of record.
_FALLBACK_LD = {
    "grids": [{"logg": 4.5,
               "teff": [3500, 4000, 4500, 5000, 5500, 5800, 6000, 6500, 7000],
               "kepler": [[0.50, 0.25], [0.60, 0.16], [0.62, 0.13], [0.53, 0.20],
                          [0.45, 0.24], [0.40, 0.26], [0.37, 0.27], [0.31, 0.30], [0.26, 0.32]],
               "tess": [[0.25, 0.35], [0.42, 0.22], [0.45, 0.20], [0.40, 0.22],
                        [0.34, 0.24], [0.31, 0.25], [0.28, 0.26], [0.23, 0.29], [0.19, 0.31]]}],
    "default_teff": 5800.0,
}


# ---------------------------------------------------------------------------
# (a) limb darkening
# ---------------------------------------------------------------------------
def ld_coefficients(teff: float, logg: float, band: str, table: dict | None = None
                    ) -> tuple[float, float, str]:
    """Quadratic ``(u1, u2)`` for ``band`` in {"kepler", "tess"} at (Teff, logg).

    Linear interpolation in Teff, clamped at the grid ends, on the grid whose
    ``logg`` is nearest.  Returns a note naming any fallback taken.
    """
    table = table or _FALLBACK_LD
    grids = table.get("grids") or _FALLBACK_LD["grids"]
    note = ""
    if not (teff is not None and np.isfinite(teff)):
        teff = float(table.get("default_teff", 5800.0))
        note = "teff_missing_default_used"
    if logg is None or not np.isfinite(logg):
        logg = 4.5
        note = (note + ";" if note else "") + "logg_missing_4.5_used"
    grid = min(grids, key=lambda g: abs(float(g.get("logg", 4.5)) - float(logg)))
    t = np.asarray(grid["teff"], dtype=float)
    uu = np.asarray(grid[band.lower()], dtype=float)
    order = np.argsort(t)
    t, uu = t[order], uu[order]
    tt = float(np.clip(teff, t[0], t[-1]))
    if tt != float(teff):
        note = (note + ";" if note else "") + "teff_clamped_to_grid"
    u1 = float(np.interp(tt, t, uu[:, 0]))
    u2 = float(np.interp(tt, t, uu[:, 1]))
    return u1, u2, note


def ld_depth_factor(u1: float, u2: float, b: float) -> float:
    """``I(mu) / <I>`` for the quadratic law at impact parameter ``b`` (small planet)."""
    bb = float(b) if b is not None and np.isfinite(b) else 0.0
    mu = math.sqrt(max(1.0 - bb * bb, 0.0))
    num = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    den = 1.0 - u1 / 3.0 - u2 / 6.0
    return float(num / den)


def band_ratio(teff: float, logg: float, b: float, table: dict | None = None
               ) -> tuple[float, dict]:
    """``f_LD = F_TESS(b) / F_Kepler(b)``: the depth ratio the same ``k`` produces."""
    u1k, u2k, nk = ld_coefficients(teff, logg, "kepler", table)
    u1t, u2t, nt = ld_coefficients(teff, logg, "tess", table)
    fk = ld_depth_factor(u1k, u2k, b)
    ft = ld_depth_factor(u1t, u2t, b)
    return float(ft / fk), {"u1_kepler": u1k, "u2_kepler": u2k, "u1_tess": u1t, "u2_tess": u2t,
                            "ld_factor_kepler": fk, "ld_factor_tess": ft,
                            "ld_note": nk or nt}


# ---------------------------------------------------------------------------
# (b) dilution
# ---------------------------------------------------------------------------
def aperture_weight(sep_arcsec, r_ap_arcsec: float, psf_sigma_arcsec: float):
    """Fraction of a neighbour's light captured, ``0.5 erfc((d - r_ap) / (sqrt2 sigma))``.

    One-dimensional stand-in for the integral of a Gaussian PSF of width
    ``sigma`` centred ``d`` from the aperture edge: ~1 well inside the
    aperture, 0.5 at the edge, ~0 beyond a few sigma.  Approximate by
    construction (the real TESS PRF is undersampled and position-dependent);
    the unweighted sum is carried alongside as the upper bound.
    """
    d = np.asarray(sep_arcsec, dtype=float)
    return 0.5 * erfc((d - float(r_ap_arcsec)) / (math.sqrt(2.0) * float(psf_sigma_arcsec)))


def contamination(delta_g, sep_arcsec, *, r_ap_arcsec: float, psf_sigma_arcsec: float
                  ) -> tuple[float, float]:
    """``(c_weighted, c_max)``: ``sum 10^(-0.4 dG) w(sep)`` and the unweighted sum.

    ``delta_g = G_neighbour - G_target``; a brighter neighbour has negative
    ``dG`` and a flux ratio above one.
    """
    dg = np.asarray(delta_g, dtype=float)
    sep = np.asarray(sep_arcsec, dtype=float)
    ok = np.isfinite(dg) & np.isfinite(sep)
    if not ok.any():
        return 0.0, 0.0
    flux = 10.0 ** (-0.4 * dg[ok])
    w = aperture_weight(sep[ok], r_ap_arcsec, psf_sigma_arcsec)
    return float(np.sum(flux * w)), float(np.sum(flux))


# ---------------------------------------------------------------------------
# (c) corrected log ratio
# ---------------------------------------------------------------------------
def sym_err(err1, err2) -> float:
    """``max(|err1|, |err2|)`` with NaN-tolerance; NaN if both are missing."""
    vals = [abs(float(e)) for e in (err1, err2) if e is not None and np.isfinite(e)]
    return max(vals) if vals else float("nan")


def corrected_log_ratio(depth_kepler: float, err_kepler: float, depth_tess: float,
                        err_tess: float, *, ld_ratio: float = 1.0, contam: float = 0.0,
                        sigma_sys: float = 0.05) -> tuple[float, float, float]:
    """``(ln R_corr, sigma, ln R_raw)``; NaN when either depth is unusable."""
    if not (np.isfinite(depth_kepler) and np.isfinite(depth_tess)) or depth_kepler <= 0 \
            or depth_tess <= 0:
        return float("nan"), float("nan"), float("nan")
    c = float(contam) if np.isfinite(contam) else 0.0
    c = min(max(c, 0.0), 0.999)
    ln_raw = math.log(depth_tess / depth_kepler)
    ln_corr = ln_raw - math.log(1.0 - c) - math.log(float(ld_ratio))
    rk = (err_kepler / depth_kepler) ** 2 if np.isfinite(err_kepler) else float("nan")
    rt = (err_tess / depth_tess) ** 2 if np.isfinite(err_tess) else float("nan")
    if not (np.isfinite(rk) and np.isfinite(rt)):
        return ln_corr, float("nan"), ln_raw
    return ln_corr, math.sqrt(rk + rt + float(sigma_sys) ** 2), ln_raw


# ---------------------------------------------------------------------------
# (d) duration consistency
# ---------------------------------------------------------------------------
def t14_hours(period_days: float, a_rs: float, k: float, b: float) -> float:
    """Total transit duration for a circular orbit (Seager & Mallén-Ornelas 2003)."""
    if not all(np.isfinite(x) for x in (period_days, a_rs, k, b)) or a_rs <= 1.0:
        return float("nan")
    chord = (1.0 + k) ** 2 - b * b
    if chord <= 0:
        return float("nan")
    sin_i = math.sqrt(max(1.0 - (b / a_rs) ** 2, 0.0))
    arg = math.sqrt(chord) / (a_rs * sin_i)
    if arg >= 1.0:
        return float("nan")
    return float(period_days * 24.0 / math.pi * math.asin(arg))


def expected_t14_ratio(k: float, b: float, depth_ratio: float, *, a_rs: float | None = None,
                       period_days: float | None = None) -> float:
    """``T14' / T14`` for ``k -> k sqrt(R)`` at fixed ``b`` (and fixed ``a/R*``)."""
    if not (np.isfinite(k) and np.isfinite(b) and np.isfinite(depth_ratio)) or depth_ratio <= 0:
        return float("nan")
    k2 = k * math.sqrt(depth_ratio)
    if a_rs is not None and period_days is not None and np.isfinite(a_rs) and a_rs > 1.0:
        t0 = t14_hours(period_days, a_rs, k, b)
        t1 = t14_hours(period_days, a_rs, k2, b)
        if np.isfinite(t0) and np.isfinite(t1) and t0 > 0:
            return float(t1 / t0)
    c0 = (1.0 + k) ** 2 - b * b
    c1 = (1.0 + k2) ** 2 - b * b
    if c0 <= 0 or c1 <= 0:
        return float("nan")
    return float(math.sqrt(c1 / c0))


def b_implied_by_duration(k: float, b: float, t14_ratio: float) -> float:
    """The ``b'`` that gives the observed ``T14'/T14`` at fixed ``k`` (NaN if none)."""
    if not (np.isfinite(k) and np.isfinite(b) and np.isfinite(t14_ratio)) or t14_ratio <= 0:
        return float("nan")
    c0 = (1.0 + k) ** 2 - b * b
    if c0 <= 0:
        return float("nan")
    b2 = (1.0 + k) ** 2 - t14_ratio ** 2 * c0
    if b2 < 0:
        return float("nan")
    return float(math.sqrt(b2))


def duration_test(k: float, b: float, depth_ratio: float, dur_kepler: float, err_kepler: float,
                  dur_tess: float, err_tess: float, *, a_rs: float | None = None,
                  period_days: float | None = None, sigma_dur_sys: float = 0.05,
                  nsigma: float = 3.0) -> dict:
    """Compare the observed duration ratio with the fixed-``b`` prediction.

    ``verdict``: ``fixed_b`` (|z| < nsigma), ``tracks_b`` (disagrees and a
    ``b'`` at fixed ``k`` reproduces the observed ratio), ``duration_inconsistent``
    (disagrees and no ``b'`` does either), ``inconclusive`` (inputs missing).
    """
    out = {"t14_ratio_observed": float("nan"), "t14_ratio_expected_fixed_b": float("nan"),
           "z_duration": float("nan"), "b_implied_by_duration": float("nan"),
           "duration_verdict": DUR_INCONCLUSIVE}
    if not all(np.isfinite(x) for x in (dur_kepler, dur_tess)) or dur_kepler <= 0 or dur_tess <= 0:
        return out
    obs = float(dur_tess / dur_kepler)
    out["t14_ratio_observed"] = obs
    exp = expected_t14_ratio(k, b, depth_ratio, a_rs=a_rs, period_days=period_days)
    out["t14_ratio_expected_fixed_b"] = exp
    out["b_implied_by_duration"] = b_implied_by_duration(k, b, obs)
    if not np.isfinite(exp):
        return out
    rk = (err_kepler / dur_kepler) ** 2 if np.isfinite(err_kepler) else 0.0
    rt = (err_tess / dur_tess) ** 2 if np.isfinite(err_tess) else 0.0
    sig = math.sqrt(rk + rt + float(sigma_dur_sys) ** 2)
    z = (math.log(obs) - math.log(exp)) / sig
    out["z_duration"] = float(z)
    if abs(z) < nsigma:
        out["duration_verdict"] = DUR_FIXED_B
    elif np.isfinite(out["b_implied_by_duration"]):
        out["duration_verdict"] = DUR_TRACKS_B
    else:
        out["duration_verdict"] = DUR_INCONSISTENT
    return out


# ---------------------------------------------------------------------------
# per-planet statistic
# ---------------------------------------------------------------------------
@dataclass
class DriftParams:
    sigma_sys: float = 0.05
    sigma_dur_sys: float = 0.05
    n_consistent: float = 3.0
    n_candidate: float = 5.0
    n_duration: float = 3.0
    r_ap_arcsec: float = 21.0
    psf_sigma_arcsec: float = 10.5
    apply_dilution_to_tess: bool = True
    subtract_population_median: bool = True
    han2025_expected_deficit: float = 0.12
    ld_table: dict | None = None

    @classmethod
    def from_config(cls, conf: dict) -> DriftParams:
        d = conf.get("drift") or {}
        ap = conf.get("aperture") or {}
        return cls(sigma_sys=float(d.get("sigma_sys", 0.05)),
                   sigma_dur_sys=float(d.get("sigma_dur_sys", 0.05)),
                   n_consistent=float(d.get("n_consistent", 3.0)),
                   n_candidate=float(d.get("n_candidate", 5.0)),
                   n_duration=float(d.get("n_duration", 3.0)),
                   r_ap_arcsec=float(ap.get("radius_arcsec", 21.0)),
                   psf_sigma_arcsec=float(ap.get("psf_sigma_arcsec", 10.5)),
                   apply_dilution_to_tess=bool(d.get("apply_dilution_to_tess", True)),
                   subtract_population_median=bool(d.get("subtract_population_median", True)),
                   han2025_expected_deficit=float(d.get("han2025_expected_deficit", 0.12)),
                   ld_table=conf.get("limb_darkening"))


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def planet_drift(row: dict, neighbours: pd.DataFrame | None, params: DriftParams) -> dict:
    """Every drift quantity for one joined planet (a dict of the joined row)."""
    depth_k = _f(row.get("koi_depth"))
    depth_t = _f(row.get("pl_trandep"))
    err_k = sym_err(_f(row.get("koi_depth_err1")), _f(row.get("koi_depth_err2")))
    err_t = sym_err(_f(row.get("pl_trandeperr1")), _f(row.get("pl_trandeperr2")))
    k = _f(row.get("koi_ror"))
    if not np.isfinite(k) and np.isfinite(depth_k) and depth_k > 0:
        k = math.sqrt(depth_k * 1e-6)
    b = _f(row.get("koi_impact"))
    teff = _f(row.get("koi_steff"))
    logg = _f(row.get("koi_slogg"))
    a_rs = _f(row.get("koi_dor"))
    period = _f(row.get("koi_period"))

    f_ld, ld = band_ratio(teff, logg, b if np.isfinite(b) else 0.0, params.ld_table)
    c_w, c_max, n_neigh, target_g, g_src = 0.0, 0.0, 0, float("nan"), "none"
    if neighbours is not None and len(neighbours):
        target_g, g_src, dg, sep = _neighbour_terms(neighbours, _f(row.get("koi_kepmag")))
        c_w, c_max = contamination(dg, sep, r_ap_arcsec=params.r_ap_arcsec,
                                   psf_sigma_arcsec=params.psf_sigma_arcsec)
        n_neigh = int(len(dg))
    contam = c_w if params.apply_dilution_to_tess else 0.0
    ln_corr, sig, ln_raw = corrected_log_ratio(depth_k, err_k, depth_t, err_t, ld_ratio=f_ld,
                                               contam=contam, sigma_sys=params.sigma_sys)
    ratio_corr = math.exp(ln_corr) if np.isfinite(ln_corr) else float("nan")
    dur = duration_test(k, b if np.isfinite(b) else 0.0, ratio_corr,
                        _f(row.get("koi_duration")),
                        sym_err(_f(row.get("koi_duration_err1")), _f(row.get("koi_duration_err2"))),
                        _f(row.get("pl_trandurh")),
                        sym_err(_f(row.get("pl_trandurherr1")), _f(row.get("pl_trandurherr2"))),
                        a_rs=a_rs, period_days=period, sigma_dur_sys=params.sigma_dur_sys,
                        nsigma=params.n_duration)
    out = {"depth_kepler_ppm": depth_k, "depth_kepler_err_ppm": err_k,
           "depth_tess_ppm": depth_t, "depth_tess_err_ppm": err_t,
           "k_kepler": k, "b_kepler": b, "a_rs_kepler": a_rs,
           "ld_band_ratio": f_ld, **ld,
           "n_gaia_neighbours": n_neigh, "target_g": target_g, "target_g_source": g_src,
           "contam_weighted": c_w, "contam_max": c_max, "contam_applied": contam,
           "ln_ratio_raw": ln_raw, "ln_ratio_corr": ln_corr, "sigma_ln_ratio": sig,
           "depth_ratio_corr": ratio_corr, **dur}
    return out


def _neighbour_terms(neigh: pd.DataFrame, kepmag: float, *, self_radius_arcsec: float = 1.5
                     ) -> tuple[float, str, np.ndarray, np.ndarray]:
    """Target G (nearest Gaia source within 1.5", else Kepler mag) and the neighbour lists."""
    n = neigh.copy()
    n["sep_arcsec"] = pd.to_numeric(n["sep_arcsec"], errors="coerce")
    n["phot_g_mean_mag"] = pd.to_numeric(n["phot_g_mean_mag"], errors="coerce")
    n = n.dropna(subset=["sep_arcsec"]).sort_values("sep_arcsec")
    if len(n) and n["sep_arcsec"].iloc[0] <= self_radius_arcsec \
            and np.isfinite(n["phot_g_mean_mag"].iloc[0]):
        target_g = float(n["phot_g_mean_mag"].iloc[0])
        src = "gaia_nearest"
        rest = n.iloc[1:]
    else:
        # Kp ~ G to ~0.1 mag for the Kepler field's FGK dwarfs; flagged.
        target_g = float(kepmag) if np.isfinite(kepmag) else float("nan")
        src = "kepmag" if np.isfinite(target_g) else "unknown"
        rest = n
    dg = rest["phot_g_mean_mag"].to_numpy(dtype=float) - target_g
    return target_g, src, dg, rest["sep_arcsec"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# (e) classification
# ---------------------------------------------------------------------------
def classify(ln_ratio: float, sigma: float, duration_verdict: str, vetoes: list[str], *,
             n_consistent: float = 3.0, n_candidate: float = 5.0, offset: float = 0.0
             ) -> tuple[str, float]:
    """``(class, z)`` with ``z = (ln_ratio - offset) / sigma``.

    ``offset`` is the population median (the Han et al. 2025 deficit measured
    on this sample) when ``subtract_population_median`` is on.  A candidate
    needs ``z >= n_candidate``, a ``fixed_b`` duration verdict and no veto.
    """
    if not (np.isfinite(ln_ratio) and np.isfinite(sigma)) or sigma <= 0:
        return CLASS_UNMEASURED, float("nan")
    z = (ln_ratio - float(offset)) / sigma
    if abs(z) < n_consistent:
        return CLASS_CONSISTENT, float(z)
    if z < 0:
        return CLASS_SHALLOWER, float(z)
    if z >= n_candidate and duration_verdict == DUR_FIXED_B and not vetoes:
        return CLASS_CANDIDATE, float(z)
    return CLASS_DEEPER, float(z)


def population_offset(ln_ratios, *, enabled: bool = True) -> tuple[float, int]:
    """Median ln-ratio over the measured population (0 when disabled or empty)."""
    v = np.asarray([x for x in ln_ratios if x is not None and np.isfinite(x)], dtype=float)
    if not enabled or not len(v):
        return 0.0, int(len(v))
    return float(np.median(v)), int(len(v))


__all__ = [
    "CLASSES", "CLASS_CANDIDATE", "CLASS_CONSISTENT", "CLASS_DEEPER", "CLASS_SHALLOWER",
    "CLASS_UNMEASURED", "DUR_FIXED_B", "DUR_INCONCLUSIVE", "DUR_INCONSISTENT", "DUR_TRACKS_B",
    "DriftParams", "aperture_weight", "b_implied_by_duration", "band_ratio", "classify",
    "contamination", "corrected_log_ratio", "duration_test", "expected_t14_ratio",
    "ld_coefficients", "ld_depth_factor", "planet_drift", "population_offset", "sym_err",
    "t14_hours",
]
