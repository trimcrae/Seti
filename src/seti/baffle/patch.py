"""BAFFLE patch stage — the *geometry* that separates a screen from a systematic.

The primary funnel finds stars X whose W1 and W2 sit significantly below the
photospheric locus.  A band-selective absorbing screen ("baffle") placed on the
Sun→X line at heliocentric distance ``d`` (10²–10⁴ AU) with radius ``R`` ≥ 1 AU
has two geometric consequences that no photometric systematic shares:

1. **Patch coherence.**  The screen subtends ``rho = R / d`` (206265″·R/d):
   3.4′ for R = 1 AU at 1000 AU, 34′ at 100 AU, 21″ at 10⁴ AU.  *Every* Gaia
   source within ``rho`` of X — foreground, background, any spectral type —
   shares the deficit, as a **top-hat**: full deficit inside, none outside.  A
   bright-star halo or a PSF-fit systematic centred on X decays *smoothly* with
   angular offset.  :func:`coherence_test` is a hypergeometric test of the
   deficit fraction inside radius r against the outer annulus, scanned over r;
   :func:`profile_shape` fits the residual-vs-offset profile with a top-hat and
   a power-law halo and reports which wins.

2. **Annual parallax modulation.**  Seen from Earth, the screen's centre is
   displaced from X by minus Earth's heliocentric position projected on the
   tangent plane, divided by ``d`` — the same geometry as stellar parallax, with
   semi-major axis ``pi_b = 206265″ / d_AU``.  X itself stays covered whenever
   R > 1 AU, so its own NEOWISE W1/W2 light curve must be **flat at the
   AllWISE deficit level**; a neighbour at offset ``theta`` with
   ``rho − pi_b < |theta| < rho + pi_b`` is covered only while
   ``|theta − p(t)| < rho`` and therefore switches **annually with a phase
   fixed by the geometry**.  :func:`predict_coverage` computes that boolean
   schedule with astropy's Earth ephemeris and :func:`modulation_test`
   compares the magnitudes in predicted-covered vs predicted-uncovered visits,
   with a label-permutation ("phase-scrambled") null.

Everything here is offline-testable: the two fetchers (Gaia+AllWISE+2MASS
neighbour cone; NEOWISE single-exposure epochs) are injected, and the network
defaults are used on the runner only.  Every network call is wrapped so one
failing object records ``FETCH_FAILED`` and the stage continues.

Known degeneracy, stated rather than hidden: NEOWISE visits a field twice a
year at fixed solar elongation, so an annual switch is sampled as an
*alternating-visit* pattern, which a scan-direction photometric systematic can
mimic.  The screen predicts **antiphase** switching for neighbours on opposite
sides of X, a common-mode systematic does not; ``alternation_control_sig`` and
``modulation_degenerate`` in the output report how far the geometric pattern
beats the best common-mode alternation for the neighbours actually fetched.

Config (``baffle.patch``; every key has the default given here)::

    search_radius_arcmin: 10       # Gaia/AllWISE/2MASS cone around X
    rho_grid_arcsec: null          # scan radii; null -> 40 log steps 20" .. search radius
    d_grid_au: [100, 200, 500, 1000, 2000, 5000, 10000]
    R_grid_au: [1, 2, 5]
    min_neighbours: 8              # below this: INSUFFICIENT_NEIGHBOURS
    deficit_sig: 3.0               # a neighbour is "deficit" if resid/sigma < -deficit_sig ...
    deficit_mag: 0.2               # ... and resid < -deficit_mag (combined W1+W2), both bands negative
    max_objects: 200
    coherence_p_max: 0.01          # trial-corrected hypergeometric p for COHERENT_PATCH
    min_deficit_inside: 2          # at least this many deficit neighbours inside best rho
    profile_dchi2: 4.0             # top-hat must beat the smooth halo by this much in chi^2
    profile_detect_dchi2: 9.0      # either model must beat "no deficit" by this much
    max_modulation_neighbours: 12  # NEOWISE pulls per object (runner cost)
    n_permutations: 2000           # phase-scrambled null
    modulation_sig_min: 5.0
    modulation_null_p_max: 1.0e-3
    min_own_visits: 6
    own_chi2_max: 3.0              # reduced chi^2 of X's NEOWISE series about a constant
    own_offset_max: 0.15           # |NEOWISE mean - AllWISE| tolerance, mag (zero-point + saturation slop)
    w1_saturation_mag: 8.0         # neighbours brighter than this in W1 are dropped
    fallback_locus: {w1: 0.05, w2: 0.03, scatter: 0.06}
    seed: 42
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seti.vigil.variability import bin_visits

ARCSEC_PER_RAD = 206264.806247
MJD_2014 = 56658.0            # 2014-01-01: start of the NEOWISE reactivation mission
MJD_2024_END = 60310.0        # 2024-01-01 (used only to label the span)
DAYS_PER_YEAR = 365.25

VERDICTS = ("MODULATED", "COHERENT_PATCH", "ISOLATED_DEFICIT", "NOT_COHERENT",
            "INSUFFICIENT_NEIGHBOURS", "FETCH_FAILED")

DEFAULTS: dict[str, Any] = {
    "search_radius_arcmin": 10.0,
    "rho_grid_arcsec": None,
    "d_grid_au": [100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0],
    "R_grid_au": [1.0, 2.0, 5.0],
    "min_neighbours": 8,
    "deficit_sig": 3.0,
    "deficit_mag": 0.2,
    "max_objects": 200,
    "coherence_p_max": 0.01,
    "min_deficit_inside": 2,
    "profile_dchi2": 4.0,
    "profile_detect_dchi2": 9.0,
    "max_modulation_neighbours": 12,
    "n_permutations": 2000,
    "modulation_sig_min": 5.0,
    "modulation_null_p_max": 1.0e-3,
    "min_own_visits": 6,
    "own_chi2_max": 3.0,
    "own_offset_max": 0.15,
    "w1_saturation_mag": 8.0,
    "fallback_locus": {"w1": 0.05, "w2": 0.03, "scatter": 0.06},
    "seed": 42,
}

PATCH_COLUMNS = [
    "source_id", "ra", "dec", "status", "n_neighbours", "n_deficit_total",
    "n_deficit_inside", "n_deficit_outside", "best_rho_arcsec", "coherence_p",
    "coherence_p_raw", "n_rho_tested", "profile_shape", "profile_dchi2_tophat",
    "profile_dchi2_smooth", "profile_amp_mag", "own_status", "own_neowise_n_visits",
    "own_flat_chi2", "own_flat_chi2_w2", "own_offset_from_allwise",
    "own_offset_from_allwise_w2", "own_constant", "n_modulation_neighbours",
    "best_d_au", "best_R_au", "modulation_sig", "modulation_null_p",
    "alternation_control_sig", "modulation_degenerate", "patch_verdict",
]


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def patch_config(cfg: dict | None) -> dict:
    """The ``patch`` subsection of the ``baffle`` mapping with every default filled."""
    cfg = cfg or {}
    sub = cfg.get("patch", cfg) if isinstance(cfg, dict) else {}
    sub = sub if isinstance(sub, dict) else {}
    out = dict(DEFAULTS)
    out["fallback_locus"] = dict(DEFAULTS["fallback_locus"])
    for k, v in sub.items():
        if k == "fallback_locus" and isinstance(v, dict):
            out["fallback_locus"].update(v)
        elif k in out:
            out[k] = v
        else:
            out[k] = v
    return out


def rho_grid_from_config(pc: dict) -> np.ndarray:
    """Radii (arcsec) at which the inside/outside deficit fractions are compared."""
    grid = pc.get("rho_grid_arcsec")
    r_max = float(pc["search_radius_arcmin"]) * 60.0
    if grid is None:
        return np.geomspace(20.0, r_max, 40)
    g = np.asarray(grid, dtype=float)
    g = g[(g > 0) & (g <= r_max)]
    return np.unique(g)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def rho_arcsec(d_au: float, R_au: float) -> float:
    """Angular radius of a screen of radius R at heliocentric distance d."""
    return ARCSEC_PER_RAD * float(R_au) / float(d_au)


def parallax_amplitude_arcsec(d_au: float) -> float:
    """Semi-major axis of the screen's annual apparent ellipse: 1 AU / d."""
    return ARCSEC_PER_RAD / float(d_au)


def _tangent_basis(ra_deg: float, dec_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra = math.radians(float(ra_deg))
    dec = math.radians(float(dec_deg))
    n = np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])
    east = np.array([-math.sin(ra), math.cos(ra), 0.0])
    north = np.array([-math.sin(dec) * math.cos(ra), -math.sin(dec) * math.sin(ra), math.cos(dec)])
    return n, east, north


def earth_heliocentric_au(times_mjd) -> np.ndarray:
    """Earth's heliocentric position (AU, ICRS axes) at each MJD, shape (n, 3).

    Uses astropy's built-in (erfa) ephemeris: no download, good to ~1e-5 AU,
    which is 1e-5 of the offsets this module cares about.  MJD is taken as UTC;
    the UTC-TDB difference (~69 s) moves Earth by 2 km, i.e. 1e-8 AU.
    """
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time

    t = Time(np.atleast_1d(np.asarray(times_mjd, dtype=float)), format="mjd", scale="utc")
    e = get_body_barycentric("earth", t)
    s = get_body_barycentric("sun", t)
    xyz = (e - s).xyz.to_value("AU")
    return np.asarray(xyz, dtype=float).T.reshape(-1, 3)


def baffle_offset_arcsec(times_mjd, ra: float, dec: float, d_au: float) -> tuple[np.ndarray, np.ndarray]:
    """Apparent (east, north) offset of the screen centre from X, in arcsec.

    The screen sits at heliocentric position ``d * n_hat`` on the Sun→X line;
    seen from Earth at heliocentric position ``r_E`` its direction is
    ``d n_hat − r_E``.  X is treated as at infinity (its own parallax, ≤ 0.1″
    for the funnel's stars, is ≤ 0.5 % of the smallest amplitude on the grid).
    The exact gnomonic projection is used, but to first order this is simply
    ``−(r_E · ê)/d, −(r_E · n̂orth)/d`` — minus the parallactic displacement of a
    body at distance d, as it must be.
    """
    n, east, north = _tangent_basis(ra, dec)
    r_e = earth_heliocentric_au(times_mjd)
    v = float(d_au) * n[None, :] - r_e
    depth = v @ n
    x = (v @ east) / depth
    y = (v @ north) / depth
    return x * ARCSEC_PER_RAD, y * ARCSEC_PER_RAD


def predict_coverage(theta_ra: float, theta_dec: float, times_mjd, ra: float, dec: float,
                     d_au: float, R_au: float) -> np.ndarray:
    """True at each epoch when a source at tangent-plane offset (theta_ra, theta_dec)
    arcsec from X (east, north; ``theta_ra`` already includes cos dec) is behind
    the screen of radius R at distance d."""
    px, py = baffle_offset_arcsec(times_mjd, ra, dec, d_au)
    rho = rho_arcsec(d_au, R_au)
    dist = np.hypot(float(theta_ra) - px, float(theta_dec) - py)
    return dist < rho


def tangent_offsets_arcsec(ra_x: float, dec_x: float, ra, dec) -> tuple[np.ndarray, np.ndarray]:
    """Gnomonic (east, north) offsets of sources from X, arcsec."""
    n, east, north = _tangent_basis(ra_x, dec_x)
    r = np.radians(np.asarray(ra, dtype=float))
    d = np.radians(np.asarray(dec, dtype=float))
    v = np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], axis=-1)
    depth = v @ n
    return (v @ east) / depth * ARCSEC_PER_RAD, (v @ north) / depth * ARCSEC_PER_RAD


# --------------------------------------------------------------------------
# Neighbour residuals against the photospheric locus
# --------------------------------------------------------------------------

class FlatLocus:
    """Stated fallback: a flat photospheric locus Ks−W1 = 0.05, Ks−W2 = 0.03, scatter 0.06.

    Used when the caller supplies no locus.  It is a *fallback*, not the
    calibrated locus: with it, late-type neighbours (whose true Ks−W1 rises
    toward 0.1–0.2) read as mildly *excess*, never as spurious deficits, so it
    errs against a patch rather than for one.
    """

    def __init__(self, w1: float = 0.05, w2: float = 0.03, scatter: float = 0.06):
        self.w1, self.w2, self.scatter = float(w1), float(w2), float(scatter)

    def predict(self, jk, lum_class, band):
        b = str(band).lower().replace("ks-", "").replace("ks_", "")
        med = self.w1 if b == "w1" else self.w2
        return med, self.scatter


def _lum_class(g, parallax, jk, bp_rp=None) -> str:
    """Giant / dwarf / blue from Gaia G + parallax, matching the locus module's split."""
    try:
        if parallax is not None and np.isfinite(parallax) and parallax > 0 and np.isfinite(g):
            m_g = float(g) + 5.0 * math.log10(float(parallax)) - 10.0
        else:
            m_g = float("nan")
    except (TypeError, ValueError):
        m_g = float("nan")
    colour = bp_rp if (bp_rp is not None and np.isfinite(bp_rp)) else None
    if colour is not None:
        if colour < 0.4:
            return "blue"
        if np.isfinite(m_g) and m_g < 2.5 and colour > 0.9:
            return "giant"
        return "dwarf"
    if np.isfinite(jk) and jk < 0.05:
        return "blue"
    if np.isfinite(m_g) and m_g < 2.5 and np.isfinite(jk) and jk > 0.45:
        return "giant"
    return "dwarf"


def _locus_predict(locus, jk: float, lum_class: str, band: str, fallback: FlatLocus) -> tuple[float, float]:
    if locus is not None:
        for name in (band, band.lower(), f"Ks-{band}", f"ks_{band.lower()}"):
            try:
                med, sc = locus.predict(jk, lum_class, name)
                med, sc = float(med), float(sc)
                if np.isfinite(med) and np.isfinite(sc) and sc > 0:
                    return med, sc
            except Exception:  # noqa: BLE001
                continue
    return fallback.predict(jk, lum_class, band)


def _col(df: pd.DataFrame, *names, default=np.nan) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([default] * len(df), index=df.index)


def _quality_mask(nb: pd.DataFrame, w1_sat: float) -> np.ndarray:
    ok = np.ones(len(nb), dtype=bool)
    for c in ("ks_m", "w1mpro", "w2mpro", "ks_msigcom", "w1mpro_error", "w2mpro_error"):
        if c in nb.columns:
            v = pd.to_numeric(nb[c], errors="coerce").to_numpy(float)
            ok &= np.isfinite(v)
            if c.endswith("error") or c.endswith("sigcom"):
                ok &= v > 0
        else:
            ok &= False
    if "w1mpro" in nb.columns:
        ok &= pd.to_numeric(nb["w1mpro"], errors="coerce").to_numpy(float) > w1_sat
    if "cc_flags" in nb.columns:
        cc = nb["cc_flags"].astype(str).str.strip()
        ok &= (cc.str[:2] == "00") | (cc == "") | (cc == "nan")
    if "ext_flag" in nb.columns:
        ext = pd.to_numeric(nb["ext_flag"], errors="coerce").fillna(0).to_numpy(float)
        ok &= ext <= 0
    if "ph_qual" in nb.columns:
        pq = nb["ph_qual"].astype(str).str.strip()
        four = pq.str.len() >= 4
        three = pq.str.len() == 3
        w_ok = pq.str[0].isin(list("AB")) & pq.str[1].isin(list("AB"))
        k_ok = pq.str[2].isin(list("AB"))
        ok &= (~four & ~three) | (four & w_ok) | (three & k_ok)
    return ok


def neighbour_residuals(nb: pd.DataFrame, ra_x: float, dec_x: float, pc: dict,
                        locus=None, exclude_source_id=None) -> pd.DataFrame:
    """Per-neighbour Ks−W1 / Ks−W2 residuals, deficit flag and tangent-plane offset.

    ``resid_w1 = (Ks − W1)_obs − locus``: negative means W1 fainter than the
    photosphere predicts, i.e. a mid-IR deficit.  ``resid`` is the inverse-
    variance combination of the two bands; a neighbour is ``deficit`` when the
    combination is below −``deficit_mag`` at more than ``deficit_sig`` sigma
    *and* both bands are individually negative (a screen dims both).
    """
    fb = pc["fallback_locus"]
    fallback = FlatLocus(fb.get("w1", 0.05), fb.get("w2", 0.03), fb.get("scatter", 0.06))
    nb = nb.reset_index(drop=True).copy()
    nb.columns = [str(c).lower() for c in nb.columns]
    ok = _quality_mask(nb, float(pc["w1_saturation_mag"]))
    if exclude_source_id is not None and "source_id" in nb.columns:
        sid = pd.to_numeric(nb["source_id"], errors="coerce")
        ok &= ~(sid == exclude_source_id).to_numpy()
    ex, ey = tangent_offsets_arcsec(ra_x, dec_x, _col(nb, "ra"), _col(nb, "dec"))
    theta = np.hypot(ex, ey)
    # X itself, if the fetcher returned it under a different id, is by
    # construction a deficit; keep the patch test blind to it.
    ok &= theta > 1.0
    nb = nb[ok].reset_index(drop=True)
    ex, ey, theta = ex[ok], ey[ok], theta[ok]
    if len(nb) == 0:
        return pd.DataFrame(columns=["source_id", "theta_arcsec", "theta_ra", "theta_dec",
                                     "resid_w1", "sig_w1", "resid_w2", "sig_w2", "resid",
                                     "resid_err", "sig", "deficit", "lum_class"])
    ks = pd.to_numeric(nb["ks_m"], errors="coerce").to_numpy(float)
    kse = pd.to_numeric(nb["ks_msigcom"], errors="coerce").to_numpy(float)
    w1 = pd.to_numeric(nb["w1mpro"], errors="coerce").to_numpy(float)
    w1e = pd.to_numeric(nb["w1mpro_error"], errors="coerce").to_numpy(float)
    w2 = pd.to_numeric(nb["w2mpro"], errors="coerce").to_numpy(float)
    w2e = pd.to_numeric(nb["w2mpro_error"], errors="coerce").to_numpy(float)
    j = pd.to_numeric(_col(nb, "j_m"), errors="coerce").to_numpy(float)
    g = pd.to_numeric(_col(nb, "phot_g_mean_mag"), errors="coerce").to_numpy(float)
    plx = pd.to_numeric(_col(nb, "parallax"), errors="coerce").to_numpy(float)
    bprp = pd.to_numeric(_col(nb, "bp_rp"), errors="coerce").to_numpy(float)
    jk = j - ks

    n = len(nb)
    r1 = np.full(n, np.nan)
    e1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    e2 = np.full(n, np.nan)
    classes = []
    for i in range(n):
        lc = _lum_class(g[i], plx[i], jk[i], bprp[i] if np.isfinite(bprp[i]) else None)
        classes.append(lc)
        m1, s1 = _locus_predict(locus, jk[i], lc, "W1", fallback)
        m2, s2 = _locus_predict(locus, jk[i], lc, "W2", fallback)
        r1[i] = (ks[i] - w1[i]) - m1
        e1[i] = math.sqrt(s1 ** 2 + kse[i] ** 2 + w1e[i] ** 2)
        r2[i] = (ks[i] - w2[i]) - m2
        e2[i] = math.sqrt(s2 ** 2 + kse[i] ** 2 + w2e[i] ** 2)
    w_1 = 1.0 / e1 ** 2
    w_2 = 1.0 / e2 ** 2
    resid = (w_1 * r1 + w_2 * r2) / (w_1 + w_2)
    # Ks error is common to both bands; the combination therefore cannot beat it.
    resid_err = np.sqrt(np.maximum(1.0 / (w_1 + w_2), kse ** 2))
    sig = resid / resid_err
    deficit = ((resid < -float(pc["deficit_mag"])) & (sig < -float(pc["deficit_sig"]))
               & (r1 < 0) & (r2 < 0))
    out = pd.DataFrame({
        "source_id": _col(nb, "source_id", default=-1).to_numpy(),
        "ra": _col(nb, "ra").to_numpy(float), "dec": _col(nb, "dec").to_numpy(float),
        "pmra": pd.to_numeric(_col(nb, "pmra", default=0.0), errors="coerce").fillna(0.0).to_numpy(float),
        "pmdec": pd.to_numeric(_col(nb, "pmdec", default=0.0), errors="coerce").fillna(0.0).to_numpy(float),
        "phot_g_mean_mag": g,
        "theta_arcsec": theta, "theta_ra": ex, "theta_dec": ey,
        "resid_w1": r1, "sig_w1": r1 / e1, "resid_w2": r2, "sig_w2": r2 / e2,
        "resid": resid, "resid_err": resid_err, "sig": sig,
        "deficit": deficit, "lum_class": classes,
    })
    return out.sort_values("theta_arcsec").reset_index(drop=True)


# --------------------------------------------------------------------------
# Patch coherence
# --------------------------------------------------------------------------

def coherence_test(theta, deficit, rho_grid) -> dict:
    """Is the deficit fraction inside r significantly above the outer annulus?

    For each r in ``rho_grid`` with at least two sources on each side, the
    number of deficit sources inside r is tested against the hypergeometric
    distribution of ``K`` deficits among ``N`` sources drawn ``n_in`` at a time
    (one-sided: excess inside).  The minimum p over the scan is reported raw
    and Bonferroni-corrected for the number of radii tested; the r at the
    minimum is ``best_rho_arcsec``.
    """
    from scipy.stats import hypergeom

    th = np.asarray(theta, dtype=float)
    df = np.asarray(deficit, dtype=bool)
    N = int(th.size)
    K = int(df.sum())
    best = {"best_rho_arcsec": float("nan"), "coherence_p_raw": 1.0, "coherence_p": 1.0,
            "n_deficit_inside": 0, "n_deficit_outside": K, "n_inside": 0, "n_outside": N,
            "n_rho_tested": 0, "n_deficit_total": K, "n_neighbours": N,
            "frac_inside": float("nan"), "frac_outside": float("nan"), "scan": []}
    if N < 4 or K == 0:
        return best
    scan = []
    for r in np.asarray(rho_grid, dtype=float):
        inside = th < r
        n_in = int(inside.sum())
        n_out = N - n_in
        if n_in < 2 or n_out < 2:
            continue
        k_in = int(df[inside].sum())
        k_out = K - k_in
        f_in, f_out = k_in / n_in, k_out / n_out
        p = float(hypergeom.sf(k_in - 1, N, K, n_in)) if f_in > f_out else 1.0
        scan.append((float(r), n_in, k_in, n_out, k_out, p))
    best["n_rho_tested"] = len(scan)
    if not scan:
        return best
    best["scan"] = [{"r": s[0], "n_in": s[1], "k_in": s[2], "n_out": s[3], "k_out": s[4], "p": s[5]}
                    for s in scan]
    i = int(np.argmin([s[5] for s in scan]))
    r, n_in, k_in, n_out, k_out, p = scan[i]
    best.update({"best_rho_arcsec": r, "coherence_p_raw": p,
                 "coherence_p": float(min(1.0, p * len(scan))),
                 "n_deficit_inside": k_in, "n_deficit_outside": k_out,
                 "n_inside": n_in, "n_outside": n_out,
                 "frac_inside": k_in / n_in, "frac_outside": k_out / n_out})
    return best


def profile_shape(theta, deficit_mag, err, rho_grid, dchi2: float = 4.0,
                  detect_dchi2: float = 9.0) -> dict:
    """Top-hat vs smooth-halo fit of the deficit-vs-offset profile.

    ``deficit_mag`` is the *positive* deficit (−resid).  Models, each with the
    amplitude solved analytically by weighted least squares and the shape
    parameter scanned:

    * top-hat: ``A · [theta < rho]``, rho over ``rho_grid`` (a screen);
    * smooth:  ``A · (20″/theta)^alpha``, alpha in {0.5, 1, 1.5, 2, 3} (a halo
      or PSF-wing systematic centred on X).

    ``shape`` is ``'flat'`` when neither model improves on "no deficit" by
    ``detect_dchi2``; ``'tophat'``/``'smooth'`` when one beats the other by
    ``dchi2``; ``'ambiguous'`` otherwise.  Also returns the binned profile.
    """
    th = np.asarray(theta, dtype=float)
    D = np.asarray(deficit_mag, dtype=float)
    e = np.asarray(err, dtype=float)
    ok = np.isfinite(th) & np.isfinite(D) & np.isfinite(e) & (e > 0) & (th > 0)
    th, D, e = th[ok], D[ok], e[ok]
    out = {"shape": "flat", "dchi2_tophat": 0.0, "dchi2_smooth": 0.0,
           "tophat_rho_arcsec": float("nan"), "tophat_amp_mag": float("nan"),
           "smooth_alpha": float("nan"), "smooth_amp_mag": float("nan"),
           "profile_bins": [], "n_used": int(th.size)}
    if th.size < 4:
        out["shape"] = "insufficient"
        return out
    w = 1.0 / e ** 2

    best_th = (0.0, float("nan"), float("nan"))
    for r in np.asarray(rho_grid, dtype=float):
        inside = th < r
        if inside.sum() < 2 or (~inside).sum() < 2:
            continue
        s_wd = float(np.sum(w[inside] * D[inside]))
        s_w = float(np.sum(w[inside]))
        if s_wd <= 0:
            continue
        gain = s_wd ** 2 / s_w
        if gain > best_th[0]:
            best_th = (gain, float(r), s_wd / s_w)
    best_sm = (0.0, float("nan"), float("nan"))
    for alpha in (0.5, 1.0, 1.5, 2.0, 3.0):
        f = (20.0 / th) ** alpha
        s_wdf = float(np.sum(w * D * f))
        s_wff = float(np.sum(w * f * f))
        if s_wdf <= 0 or s_wff <= 0:
            continue
        gain = s_wdf ** 2 / s_wff
        if gain > best_sm[0]:
            best_sm = (gain, float(alpha), s_wdf / s_wff)
    out.update({"dchi2_tophat": best_th[0], "tophat_rho_arcsec": best_th[1],
                "tophat_amp_mag": best_th[2], "dchi2_smooth": best_sm[0],
                "smooth_alpha": best_sm[1], "smooth_amp_mag": best_sm[2]})
    if max(best_th[0], best_sm[0]) < detect_dchi2:
        out["shape"] = "flat"
    elif best_th[0] - best_sm[0] > dchi2:
        out["shape"] = "tophat"
    elif best_sm[0] - best_th[0] > dchi2:
        out["shape"] = "smooth"
    else:
        out["shape"] = "ambiguous"

    # Binned profile: equal-count bins in log offset (compact, for the JSON).
    n_bins = int(np.clip(th.size // 6, 3, 10))
    order = np.argsort(th)
    for chunk in np.array_split(order, n_bins):
        if chunk.size == 0:
            continue
        ww = w[chunk]
        out["profile_bins"].append({
            "theta_lo": float(th[chunk].min()), "theta_hi": float(th[chunk].max()),
            "theta_med": float(np.median(th[chunk])), "n": int(chunk.size),
            "deficit_mean": float(np.sum(ww * D[chunk]) / np.sum(ww)),
            "deficit_err": float(np.sqrt(1.0 / np.sum(ww)))})
    return out


# --------------------------------------------------------------------------
# NEOWISE series: X's own constancy and the neighbours' modulation
# --------------------------------------------------------------------------

def _epochs_frame(res) -> pd.DataFrame | None:
    """Accept a QueryResult (``.data``) or a bare DataFrame from a NEOWISE fetcher."""
    if res is None:
        return None
    df = getattr(res, "data", res)
    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return None
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    return df


def visits_from_epochs(df: pd.DataFrame | None) -> pd.DataFrame:
    """Collapse single-exposure NEOWISE rows into visits with :func:`bin_visits`.

    Returns one row per visit with ``t_mjd, w1, w1_err, w2, w2_err`` (an error
    is the larger of the quoted-error and within-visit-scatter errors on the
    visit mean, the conservative choice for a bright star whose quoted errors
    are known to be optimistic).
    """
    cols = ["t_mjd", "w1", "w1_err", "w1_n", "w2", "w2_err", "w2_n"]
    if df is None or len(df) == 0 or "mjd" not in df.columns:
        return pd.DataFrame(columns=cols)
    frames = {}
    for band in ("w1", "w2"):
        mcol, ecol = f"{band}mpro", f"{band}sigmpro"
        if mcol not in df.columns:
            continue
        err = df[ecol] if ecol in df.columns else pd.Series(0.02, index=df.index)
        vis = bin_visits(pd.to_numeric(df["mjd"], errors="coerce").to_numpy(float),
                         pd.to_numeric(df[mcol], errors="coerce").to_numpy(float),
                         pd.to_numeric(err, errors="coerce").to_numpy(float))
        frames[band] = pd.DataFrame({
            "t_mjd": [v.t_mjd for v in vis], band: [v.mag for v in vis],
            f"{band}_err": [max(v.err_quoted, v.err_within) for v in vis],
            f"{band}_n": [v.n_exp for v in vis]})
    if not frames:
        return pd.DataFrame(columns=cols)
    if len(frames) == 1:
        out = next(iter(frames.values()))
        for c in cols:
            if c not in out.columns:
                out[c] = np.nan
        return out[cols]
    # Match W1 and W2 visits by time (same exposures, so same visit centres).
    a, b = frames["w1"], frames["w2"]
    rows = []
    for _, r in a.iterrows():
        k = np.argmin(np.abs(b["t_mjd"].to_numpy(float) - r.t_mjd)) if len(b) else None
        if k is not None and abs(b["t_mjd"].iloc[k] - r.t_mjd) < 30.0:
            rows.append([r.t_mjd, r.w1, r.w1_err, r.w1_n,
                         b["w2"].iloc[k], b["w2_err"].iloc[k], b["w2_n"].iloc[k]])
        else:
            rows.append([r.t_mjd, r.w1, r.w1_err, r.w1_n, np.nan, np.nan, 0])
    return pd.DataFrame(rows, columns=cols)


def own_constancy(visits: pd.DataFrame, allwise_w1: float = np.nan, allwise_w2: float = np.nan,
                  chi2_max: float = 3.0, offset_max: float = 0.15, min_visits: int = 6) -> dict:
    """X sits inside the screen at all times (R > 1 AU): its light curve must be flat
    and at the AllWISE (2010) level.  Reduced chi^2 about the weighted mean, and
    the NEOWISE-mean minus AllWISE offset, per band."""
    out = {"n_visits": int(len(visits)), "flat_chi2": float("nan"), "flat_chi2_w2": float("nan"),
           "mean_w1": float("nan"), "mean_w2": float("nan"),
           "offset_from_allwise": float("nan"), "offset_from_allwise_w2": float("nan"),
           "span_yr": float("nan"), "constant": False}
    if len(visits) == 0:
        return out
    t = visits["t_mjd"].to_numpy(float)
    out["span_yr"] = float((t.max() - t.min()) / DAYS_PER_YEAR) if len(t) > 1 else 0.0
    chi = {}
    for band, key in (("w1", "flat_chi2"), ("w2", "flat_chi2_w2")):
        m = visits[band].to_numpy(float)
        e = visits[f"{band}_err"].to_numpy(float)
        ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
        if ok.sum() < 2:
            continue
        w = 1.0 / e[ok] ** 2
        mean = float(np.sum(w * m[ok]) / np.sum(w))
        chi2 = float(np.sum(w * (m[ok] - mean) ** 2) / (ok.sum() - 1))
        out[key] = chi2
        out[f"mean_{band}"] = mean
        chi[band] = chi2
        ref = allwise_w1 if band == "w1" else allwise_w2
        if ref is not None and np.isfinite(ref):
            out["offset_from_allwise" if band == "w1" else "offset_from_allwise_w2"] = mean - float(ref)
    flat = all(c < chi2_max for c in chi.values()) if chi else False
    offs = [v for v in (out["offset_from_allwise"], out["offset_from_allwise_w2"]) if np.isfinite(v)]
    level_ok = all(abs(o) < offset_max for o in offs) if offs else True
    out["constant"] = bool(flat and level_ok and out["n_visits"] >= min_visits)
    return out


def _band_stat(m: np.ndarray, e: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed (covered − uncovered) difference and its error for one band, for a
    (n_perm, n_visits) matrix of boolean labelings.  Vectorised so the
    permutation null costs one matrix product."""
    w = 1.0 / e ** 2
    L = labels.astype(float)
    wc = L @ w
    wu = (1.0 - L) @ w
    mc = (L @ (w * m)) / np.where(wc > 0, wc, np.nan)
    mu = ((1.0 - L) @ (w * m)) / np.where(wu > 0, wu, np.nan)
    diff = mc - mu
    err = np.sqrt(1.0 / np.where(wc > 0, wc, np.nan) + 1.0 / np.where(wu > 0, wu, np.nan))
    return diff, err


def modulation_test(epochs_df: pd.DataFrame, predicted_bool, n_perm: int = 2000,
                    rng: np.random.Generator | int | None = None) -> dict:
    """Compare W1/W2 in predicted-covered vs predicted-uncovered visits.

    ``epochs_df`` is a visit-level frame (``t_mjd, w1, w1_err[, w2, w2_err]``,
    as from :func:`visits_from_epochs`; a raw single-exposure frame with
    ``mjd, w1mpro, ...`` is binned first).  ``predicted_bool`` is the coverage
    schedule at those visits.

    The detection statistic is the signed difference *covered − uncovered* in
    magnitudes (positive = fainter when covered, the sign a screen predicts),
    inverse-variance weighted, per band and combined; ``sig`` is that
    difference in sigma.  The null is phase-scrambled: the labels are permuted
    over visits ``n_perm`` times and ``null_p`` is the one-sided fraction of
    permutations whose combined statistic is at least the observed one,
    ``(k + 1) / (n_perm + 1)``.
    """
    if epochs_df is not None and "mjd" in getattr(epochs_df, "columns", []) and "w1" not in epochs_df.columns:
        epochs_df = visits_from_epochs(epochs_df)
    out = {"n_visits": 0, "n_covered": 0, "n_uncovered": 0, "diff_w1": float("nan"),
           "diff_w1_err": float("nan"), "sig_w1": float("nan"), "diff_w2": float("nan"),
           "diff_w2_err": float("nan"), "sig_w2": float("nan"), "diff": float("nan"),
           "diff_err": float("nan"), "sig": 0.0, "null_p": 1.0, "n_perm": 0,
           "status": "NO_DATA"}
    if epochs_df is None or len(epochs_df) == 0:
        return out
    cov = np.asarray(predicted_bool, dtype=bool).reshape(-1)
    if cov.size != len(epochs_df):
        raise ValueError(f"predicted_bool has {cov.size} entries for {len(epochs_df)} visits")
    out["n_visits"] = int(cov.size)
    out["n_covered"] = int(cov.sum())
    out["n_uncovered"] = int((~cov).sum())
    if out["n_covered"] < 2 or out["n_uncovered"] < 2:
        out["status"] = "NO_CONTRAST"
        return out
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    n_perm = int(n_perm)
    perms = np.empty((n_perm + 1, cov.size), dtype=bool)
    perms[0] = cov
    for i in range(1, n_perm + 1):
        perms[i] = rng.permutation(cov)

    diffs, errs = [], []
    for band in ("w1", "w2"):
        if band not in epochs_df.columns:
            continue
        m = epochs_df[band].to_numpy(float)
        e = epochs_df[f"{band}_err"].to_numpy(float)
        ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
        if ok.sum() < 4 or cov[ok].sum() < 2 or (~cov[ok]).sum() < 2:
            continue
        d, s = _band_stat(m[ok], e[ok], perms[:, ok])
        out[f"diff_{band}"] = float(d[0])
        out[f"diff_{band}_err"] = float(s[0])
        out[f"sig_{band}"] = float(d[0] / s[0])
        diffs.append(d)
        errs.append(s)
    if not diffs:
        out["status"] = "NO_CONTRAST"
        return out
    W = np.stack([1.0 / s ** 2 for s in errs])
    Dm = np.stack(diffs)
    comb = np.nansum(W * Dm, axis=0) / np.nansum(W, axis=0)
    comb_err = np.sqrt(1.0 / np.nansum(W, axis=0))
    z = comb / comb_err
    z = np.where(np.isfinite(z), z, 0.0)
    out["diff"] = float(comb[0])
    out["diff_err"] = float(comb_err[0])
    out["sig"] = float(z[0])
    k = int(np.sum(z[1:] >= z[0]))
    out["null_p"] = float((k + 1) / (n_perm + 1))
    out["n_perm"] = n_perm
    out["status"] = "OK"
    return out


def scan_modulation(series: list[dict], ra: float, dec: float, d_grid, R_grid,
                    n_perm: int = 2000, seed: int = 42) -> dict:
    """Joint (d, R) scan over several neighbours' NEOWISE visit series.

    ``series`` is a list of ``{"source_id", "theta_ra", "theta_dec", "visits"}``
    (visits as from :func:`visits_from_epochs`).  For each (d, R) every
    neighbour's predicted coverage is evaluated at its own visit times; the
    per-neighbour signed z (covered − uncovered, deficit-positive) are combined
    Stouffer-style over the neighbours with contrast.  The null permutes the
    labels of every neighbour independently, *once per permutation for the
    whole grid*, and the p-value is the fraction of permutations whose
    grid-maximum combined z reaches the observed grid-maximum — so the look-
    elsewhere effect over (d, R) is inside the null, not corrected after.
    """
    rng = np.random.default_rng(seed)
    out = {"best_d_au": float("nan"), "best_R_au": float("nan"), "modulation_sig": 0.0,
           "modulation_null_p": 1.0, "n_neighbours_used": 0, "grid": [],
           "alternation_control_sig": 0.0, "per_neighbour": [], "degenerate_grid": []}
    usable = [s for s in series if s.get("visits") is not None and len(s["visits"]) >= 4]
    if not usable:
        return out
    n_perm = int(n_perm)
    # Pre-draw one permutation table per neighbour (shared across the grid).
    prepared = []
    for s in usable:
        v = s["visits"]
        t = v["t_mjd"].to_numpy(float)
        perm_idx = np.stack([np.arange(t.size)] + [rng.permutation(t.size) for _ in range(n_perm)])
        prepared.append((s, v, t, perm_idx))

    def combined_z(labels_per_nb: list[np.ndarray | None]) -> np.ndarray:
        """labels_per_nb[i]: (n_perm+1, n_visits_i) bool or None -> combined z, (n_perm+1,)."""
        zs = []
        for (_s, v, _t, _perm_idx), lab in zip(prepared, labels_per_nb, strict=True):
            if lab is None:
                continue
            diffs, errs = [], []
            for band in ("w1", "w2"):
                if band not in v.columns:
                    continue
                m = v[band].to_numpy(float)
                e = v[f"{band}_err"].to_numpy(float)
                ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
                if ok.sum() < 4:
                    continue
                d, se = _band_stat(m[ok], e[ok], lab[:, ok])
                diffs.append(d)
                errs.append(se)
            if not diffs:
                continue
            W = np.stack([1.0 / se ** 2 for se in errs])
            Dm = np.stack(diffs)
            with np.errstate(invalid="ignore", divide="ignore"):
                z = (np.nansum(W * Dm, axis=0) / np.nansum(W, axis=0)) / np.sqrt(1.0 / np.nansum(W, axis=0))
            zs.append(np.where(np.isfinite(z), z, 0.0))
        if not zs:
            return np.zeros(n_perm + 1)
        return np.sum(zs, axis=0) / math.sqrt(len(zs))

    grid_obs = []
    grid_null = []
    for d in np.asarray(d_grid, dtype=float):
        for R in np.asarray(R_grid, dtype=float):
            labels = []
            n_used = 0
            for s, _v, t, perm_idx in prepared:
                cov = predict_coverage(s["theta_ra"], s["theta_dec"], t, ra, dec, d, R)
                if cov.sum() < 2 or (~cov).sum() < 2:
                    labels.append(None)
                    continue
                labels.append(cov[perm_idx])
                n_used += 1
            if n_used == 0:
                grid_obs.append({"d_au": float(d), "R_au": float(R), "sig": 0.0, "n_used": 0})
                grid_null.append(np.zeros(n_perm + 1))
                continue
            z = combined_z(labels)
            grid_obs.append({"d_au": float(d), "R_au": float(R), "sig": float(z[0]), "n_used": n_used})
            grid_null.append(z)
    out["grid"] = grid_obs
    if not grid_obs:
        return out
    Z = np.stack(grid_null)                       # (n_grid, n_perm+1)
    zmax = Z.max(axis=0)
    i = int(np.argmax(Z[:, 0]))
    out.update({"best_d_au": grid_obs[i]["d_au"], "best_R_au": grid_obs[i]["R_au"],
                "modulation_sig": float(Z[i, 0]), "n_neighbours_used": grid_obs[i]["n_used"],
                "modulation_null_p": float((int(np.sum(zmax[1:] >= zmax[0])) + 1) / (n_perm + 1))})
    # NEOWISE samples an annual switch at two phases, so every (d, R) that
    # predicts the same on/off schedule for the fetched neighbours is
    # indistinguishable: report the whole set within 1 sigma of the best
    # rather than pretend the argmax singled out a distance.
    out["degenerate_grid"] = [(g["d_au"], g["R_au"]) for g, z in zip(grid_obs, Z[:, 0], strict=True)
                              if z >= Z[i, 0] - 1.0 and g["n_used"] > 0]

    # Common-mode control: the same alternating-visit labelling for every
    # neighbour (the pattern a scan-direction systematic would produce).
    best_ctrl = 0.0
    for phase in (0, 1):
        labels = []
        for _s, _v, t, perm_idx in prepared:
            order = np.argsort(t)
            alt = np.zeros(t.size, dtype=bool)
            alt[order[phase::2]] = True
            labels.append(alt[perm_idx] if (alt.sum() >= 2 and (~alt).sum() >= 2) else None)
        z = combined_z(labels)
        best_ctrl = max(best_ctrl, abs(float(z[0])))
    out["alternation_control_sig"] = best_ctrl

    # Per-neighbour detail at the best grid point.
    d, R = out["best_d_au"], out["best_R_au"]
    for s, v, t, _perm_idx in prepared:
        cov = predict_coverage(s["theta_ra"], s["theta_dec"], t, ra, dec, d, R)
        mt = modulation_test(v, cov, n_perm=min(n_perm, 500), rng=rng)
        out["per_neighbour"].append({"source_id": s.get("source_id"), "theta_arcsec":
                                     float(math.hypot(s["theta_ra"], s["theta_dec"])),
                                     "n_visits": int(len(v)), "n_covered": int(cov.sum()),
                                     "sig": mt["sig"], "diff": mt["diff"], "status": mt["status"]})
    return out


# --------------------------------------------------------------------------
# Default (runner-only) fetchers
# --------------------------------------------------------------------------

_NEIGHBOUR_ADQL = """
SELECT g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.parallax, g.phot_g_mean_mag, g.bp_rp,
       t.j_m, t.ks_m, t.ks_msigcom, t.ph_qual AS tmass_ph_qual,
       w.w1mpro, w.w1mpro_error, w.w2mpro, w.w2mpro_error, w.ph_qual, w.cc_flags, w.ext_flag
FROM gaiadr3.gaia_source AS g
JOIN gaiadr3.allwise_best_neighbour AS wb ON wb.source_id = g.source_id
JOIN gaiadr1.allwise_original_valid AS w ON w.allwise_oid = wb.allwise_oid
JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS tb ON tb.source_id = g.source_id
JOIN gaiadr1.tmass_original_valid AS t ON t.designation = tb.original_ext_source_id
WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {r_deg:.7f}))
"""


def neighbour_query(ra: float, dec: float, radius_arcmin: float) -> str:
    """The ADQL for the neighbour cone (pure string; testable offline)."""
    return _NEIGHBOUR_ADQL.format(ra=float(ra), dec=float(dec), r_deg=float(radius_arcmin) / 60.0).strip()


def _fetch_gaia_neighbours(ra: float, dec: float, radius_arcmin: float = 10.0,
                           retries: int = 3) -> pd.DataFrame:
    """Gaia DR3 x AllWISE x 2MASS cone through the Gaia archive (runner only)."""
    from astroquery.gaia import Gaia

    q = neighbour_query(ra, dec, radius_arcmin)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            job = Gaia.launch_job_async(q)
            df = job.get_results().to_pandas()
            df.columns = [str(c).lower() for c in df.columns]
            return df
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"neighbour cone failed after {retries} attempts: {last!r}")


def _default_neowise_fetcher(ra, dec, pmra=0.0, pmdec=0.0, **kw):
    from seti.vigil.acquire import fetch_neowise_epochs
    return fetch_neowise_epochs(ra, dec, pmra, pmdec, **kw)


# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------

def _num(v, default=np.nan) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _cand_value(row: pd.Series, *names, default=np.nan):
    for n in names:
        if n in row.index:
            return _num(row[n], default)
    return default


def _pick_modulation_neighbours(res: pd.DataFrame, best_rho: float, d_grid, R_grid,
                                k: int) -> pd.DataFrame:
    """Which neighbours are worth a NEOWISE pull: those that any (d, R) on the grid
    predicts to switch (rho − pi_b < theta < rho + pi_b), ordered by deficit
    flag first, then nearness to the best rho edge, then brightness."""
    if len(res) == 0 or k <= 0:
        return res.iloc[0:0]
    th = res["theta_arcsec"].to_numpy(float)
    switchable = np.zeros(len(res), dtype=bool)
    for d in d_grid:
        pi_b = parallax_amplitude_arcsec(d)
        for R in R_grid:
            rho = rho_arcsec(d, R)
            switchable |= (th > rho - pi_b) & (th < rho + pi_b)
    sub = res[switchable].copy()
    if len(sub) == 0:
        return sub
    edge = abs(sub["theta_arcsec"] - best_rho) if np.isfinite(best_rho) else sub["theta_arcsec"]
    sub = sub.assign(_edge=edge.to_numpy(float),
                     _g=sub["phot_g_mean_mag"].fillna(99.0).to_numpy(float))
    sub = sub.sort_values(["deficit", "_edge", "_g"], ascending=[False, True, True])
    return sub.head(k).drop(columns=["_edge", "_g"])


_VERDICT_COUNTER = {
    "MODULATED": "n_modulated", "COHERENT_PATCH": "n_coherent_patch",
    "ISOLATED_DEFICIT": "n_isolated_deficit", "NOT_COHERENT": "n_not_coherent",
    "INSUFFICIENT_NEIGHBOURS": "n_insufficient_neighbours",
    "FETCH_FAILED": "n_fetch_failed_verdict",
}


def _verdict(row: dict, pc: dict) -> str:
    if row["status"] == "FETCH_FAILED":
        return "FETCH_FAILED"
    if row["n_neighbours"] < int(pc["min_neighbours"]):
        return "INSUFFICIENT_NEIGHBOURS"
    modulated = (row["modulation_sig"] >= float(pc["modulation_sig_min"])
                 and row["modulation_null_p"] <= float(pc["modulation_null_p_max"]))
    coherent = (row["coherence_p"] < float(pc["coherence_p_max"])
                and row["n_deficit_inside"] >= int(pc["min_deficit_inside"])
                and row["profile_shape"] == "tophat")
    if modulated:
        return "MODULATED"
    if coherent:
        return "COHERENT_PATCH"
    no_patch = (row["n_deficit_total"] <= max(1, int(0.05 * row["n_neighbours"]))
                and row["profile_shape"] in ("flat", "insufficient"))
    if no_patch and row["own_constant"]:
        return "ISOLATED_DEFICIT"
    return "NOT_COHERENT"


def run_patch_stage(candidates: pd.DataFrame, out_dir, cfg: dict, *, neighbour_fetcher=None,
                    neowise_fetcher=None, locus=None, max_objects=None) -> dict:
    """Assess every candidate's patch geometry; write ``patches.csv`` and
    ``patch_profiles.json``; return the counters and the interesting objects.

    ``candidates`` needs ``source_id, ra, dec`` (``pmra, pmdec, w1mpro, w2mpro``
    used when present: proper motion for the NEOWISE cone, AllWISE magnitudes
    for the own-constancy offset).  ``cfg`` is the ``baffle`` mapping (its
    ``patch`` subsection is read; every key defaults).  ``neighbour_fetcher(ra,
    dec, radius_arcmin) -> DataFrame`` and ``neowise_fetcher(ra, dec, pmra,
    pmdec) -> QueryResult | DataFrame`` default to the Gaia-archive cone and
    :func:`seti.vigil.acquire.fetch_neowise_epochs` (network; runner only).
    """
    pc = patch_config(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fetch_nb = neighbour_fetcher or _fetch_gaia_neighbours
    fetch_nw = neowise_fetcher or _default_neowise_fetcher
    rho_grid = rho_grid_from_config(pc)
    d_grid = [float(x) for x in pc["d_grid_au"]]
    R_grid = [float(x) for x in pc["R_grid_au"]]
    limit = int(max_objects if max_objects is not None else pc["max_objects"])
    radius = float(pc["search_radius_arcmin"])
    seed = int(pc["seed"])

    rows: list[dict] = []
    profiles: dict[str, Any] = {}
    counters = {"n_assessed": 0, "n_fetch_failed": 0, "n_coherent_patch": 0, "n_modulated": 0,
                "n_own_constant": 0, "n_isolated_deficit": 0, "n_not_coherent": 0,
                "n_insufficient_neighbours": 0}
    cands = candidates.reset_index(drop=True)
    if limit is not None:
        cands = cands.head(limit)

    for i, cand in cands.iterrows():
        sid = cand["source_id"] if "source_id" in cand.index else i
        if isinstance(sid, (float, np.floating)) and np.isfinite(sid) and float(sid).is_integer():
            sid = int(sid)
        elif hasattr(sid, "item"):
            sid = sid.item()
        ra, dec = float(cand["ra"]), float(cand["dec"])
        pmra = _cand_value(cand, "pmra", default=0.0)
        pmdec = _cand_value(cand, "pmdec", default=0.0)
        row: dict[str, Any] = {c: np.nan for c in PATCH_COLUMNS}
        row.update({"source_id": sid, "ra": ra, "dec": dec, "status": "OK", "n_neighbours": 0,
                    "n_deficit_total": 0, "n_deficit_inside": 0, "n_deficit_outside": 0,
                    "coherence_p": 1.0, "coherence_p_raw": 1.0, "n_rho_tested": 0,
                    "profile_shape": "insufficient", "own_status": "NOT_FETCHED",
                    "own_neowise_n_visits": 0, "own_constant": False,
                    "n_modulation_neighbours": 0, "modulation_sig": 0.0,
                    "modulation_null_p": 1.0, "alternation_control_sig": 0.0,
                    "modulation_degenerate": False})
        prof: dict[str, Any] = {"source_id": str(sid), "ra": ra, "dec": dec}
        counters["n_assessed"] += 1
        failed = False

        # -- 1. neighbours and patch coherence ----------------------------
        try:
            nb = fetch_nb(ra, dec, radius)
            if nb is None:
                raise RuntimeError("neighbour fetcher returned None")
            nb = getattr(nb, "data", nb)
            if not isinstance(nb, pd.DataFrame):
                raise RuntimeError(f"neighbour fetcher returned {type(nb).__name__}")
        except Exception as exc:  # noqa: BLE001
            failed = True
            row["status"] = "FETCH_FAILED"
            row["error"] = repr(exc)[:300]
            nb = None

        res = pd.DataFrame()
        if nb is not None:
            res = neighbour_residuals(nb, ra, dec, pc, locus=locus, exclude_source_id=sid)
            row["n_neighbours"] = int(len(res))
            coh = coherence_test(res["theta_arcsec"], res["deficit"], rho_grid)
            row.update({k: coh[k] for k in ("best_rho_arcsec", "coherence_p", "coherence_p_raw",
                                             "n_deficit_inside", "n_deficit_outside",
                                             "n_rho_tested", "n_deficit_total")})
            shape = profile_shape(res["theta_arcsec"], -res["resid"], res["resid_err"], rho_grid,
                                  dchi2=float(pc["profile_dchi2"]),
                                  detect_dchi2=float(pc["profile_detect_dchi2"]))
            row.update({"profile_shape": shape["shape"], "profile_dchi2_tophat": shape["dchi2_tophat"],
                        "profile_dchi2_smooth": shape["dchi2_smooth"],
                        "profile_amp_mag": shape["tophat_amp_mag"]})
            prof["coherence_scan"] = [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in s.items()}
                                      for s in coh["scan"]]
            prof["profile_bins"] = shape["profile_bins"]
            prof["profile_fit"] = {k: shape[k] for k in ("shape", "dchi2_tophat", "dchi2_smooth",
                                                         "tophat_rho_arcsec", "tophat_amp_mag",
                                                         "smooth_alpha", "smooth_amp_mag")}
            prof["neighbours"] = [
                {"source_id": str(r.source_id), "theta": round(float(r.theta_arcsec), 2),
                 "pa_ra": round(float(r.theta_ra), 2), "pa_dec": round(float(r.theta_dec), 2),
                 "resid": round(float(r.resid), 4), "err": round(float(r.resid_err), 4),
                 "deficit": bool(r.deficit)}
                for r in res.itertuples()]

        # -- 2a. X's own NEOWISE light curve ------------------------------
        own_visits = pd.DataFrame()
        try:
            own = visits_from_epochs(_epochs_frame(fetch_nw(ra, dec, pmra, pmdec)))
            own_visits = own
            row["own_status"] = "OK" if len(own) else "NO_EPOCHS"
        except Exception as exc:  # noqa: BLE001
            failed = True
            row["own_status"] = "FETCH_FAILED"
            row["own_error"] = repr(exc)[:300]
        oc = own_constancy(own_visits, _cand_value(cand, "w1mpro", "w1"),
                           _cand_value(cand, "w2mpro", "w2"), chi2_max=float(pc["own_chi2_max"]),
                           offset_max=float(pc["own_offset_max"]), min_visits=int(pc["min_own_visits"]))
        row.update({"own_neowise_n_visits": oc["n_visits"], "own_flat_chi2": oc["flat_chi2"],
                    "own_flat_chi2_w2": oc["flat_chi2_w2"],
                    "own_offset_from_allwise": oc["offset_from_allwise"],
                    "own_offset_from_allwise_w2": oc["offset_from_allwise_w2"],
                    "own_constant": bool(oc["constant"])})
        if len(own_visits):
            prof["own_visits"] = [[round(float(t), 2), round(float(a), 4), round(float(b), 4)]
                                  for t, a, b in zip(own_visits["t_mjd"], own_visits["w1"],
                                                     own_visits["w2"], strict=True)]

        # -- 2b. neighbours' annual modulation ----------------------------
        series = []
        if len(res):
            picks = _pick_modulation_neighbours(res, row["best_rho_arcsec"], d_grid, R_grid,
                                                int(pc["max_modulation_neighbours"]))
            for r in picks.itertuples():
                try:
                    ep = _epochs_frame(fetch_nw(float(r.ra), float(r.dec), float(r.pmra), float(r.pmdec)))
                    v = visits_from_epochs(ep)
                except Exception as exc:  # noqa: BLE001
                    failed = True
                    row["neighbour_fetch_errors"] = row.get("neighbour_fetch_errors", 0) + 1
                    row["neighbour_error"] = repr(exc)[:300]
                    continue
                if len(v) >= 4:
                    series.append({"source_id": r.source_id, "theta_ra": float(r.theta_ra),
                                   "theta_dec": float(r.theta_dec), "visits": v})
        row["n_modulation_neighbours"] = len(series)
        if series:
            mod = scan_modulation(series, ra, dec, d_grid, R_grid, n_perm=int(pc["n_permutations"]),
                                  seed=seed + int(i))
            row.update({"best_d_au": mod["best_d_au"], "best_R_au": mod["best_R_au"],
                        "modulation_sig": mod["modulation_sig"],
                        "modulation_null_p": mod["modulation_null_p"],
                        "alternation_control_sig": mod["alternation_control_sig"]})
            row["modulation_degenerate"] = bool(mod["alternation_control_sig"] >= 0.8 * mod["modulation_sig"]
                                                and mod["modulation_sig"] > 0)
            prof["modulation_grid"] = mod["grid"]
            prof["modulation_degenerate_grid"] = mod["degenerate_grid"]
            prof["modulation_neighbours"] = mod["per_neighbour"]
            prof["neighbour_visits"] = {
                str(s["source_id"]): [[round(float(t), 2), round(float(a), 4), round(float(b), 4)]
                                      for t, a, b in zip(s["visits"]["t_mjd"], s["visits"]["w1"],
                                                         s["visits"]["w2"], strict=True)]
                for s in series}

        if failed:
            counters["n_fetch_failed"] += 1
        row["patch_verdict"] = _verdict(row, pc)
        if row["own_constant"]:
            counters["n_own_constant"] += 1
        key = _VERDICT_COUNTER[row["patch_verdict"]]
        counters[key] = counters.get(key, 0) + 1
        prof["verdict"] = row["patch_verdict"]
        rows.append(row)
        profiles[str(sid)] = prof

    cols = PATCH_COLUMNS + sorted({k for r in rows for k in r} - set(PATCH_COLUMNS))
    table = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    table.to_csv(out_dir / "patches.csv", index=False)
    with open(out_dir / "patch_profiles.json", "w") as fh:
        json.dump(_json_clean(profiles), fh, default=_json_default, separators=(",", ":"),
                  allow_nan=False)

    interesting = ("COHERENT_PATCH", "MODULATED", "ISOLATED_DEFICIT")
    objects = [
        {k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in
         ("source_id", "ra", "dec", "patch_verdict", "n_neighbours", "n_deficit_inside",
          "best_rho_arcsec", "coherence_p", "profile_shape", "own_flat_chi2",
          "own_offset_from_allwise", "best_d_au", "best_R_au", "modulation_sig",
          "modulation_null_p", "modulation_degenerate")}
        for r in rows if r["patch_verdict"] in interesting]
    summary = dict(counters)
    summary.update({"n_candidates_in": int(len(candidates)), "objects": objects,
                    "patches_csv": str(out_dir / "patches.csv"),
                    "patch_profiles_json": str(out_dir / "patch_profiles.json"),
                    "config": {k: (list(v) if isinstance(v, (list, tuple, np.ndarray)) else v)
                               for k, v in pc.items()}})
    return summary


def _json_clean(o):
    """NaN/inf -> null recursively, so the file is valid JSON for any reader."""
    if isinstance(o, dict):
        return {str(k): _json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_json_clean(v) for v in o.tolist()]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)
