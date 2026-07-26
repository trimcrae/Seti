"""Statistic 2 --- the **edge**: is there a sharp step in the anomaly rate?

A gradient is what a slow, still-running process looks like.  A *front that
stopped* looks different: a spatially coherent region with a **boundary**, and
the boundary is the observable (Carrigan's "Fermi bubble"; Landis's percolation
clusters with sharp edges; Hanson et al.'s "volume borders ... for which
astronomers might search").  This module is the matched filter for that.

The filter
----------
Bin a radial-like coordinate into **equal-expected-count** bins (using the
matched-null weights, so bin statistical weight is uniform by construction).  For
an edge at bin ``j`` and half-width ``k`` bins, compare the inner window
``[j-k, j)`` with the outer window ``[j, j+k)`` by the Poisson
likelihood-ratio for "one rate ratio" versus "two":

    Lambda = 2 [ n_in log(n_in / mu_in) + n_out log(n_out / mu_out) ],
    mu_W   = E_W * (n_in + n_out) / (E_in + E_out)

signed by which side is over-dense, and reported as ``S = sign * sqrt(Lambda)``,
which behaves like a Gaussian sigma.  This is a Kulldorff-style scan statistic:
correct for small counts, and blind to the overall rate (which the ``rho0``
pooling divides out).

The two things that make it honest
----------------------------------
1. **The null contains the fitted smooth gradient.**  A strong monotone trend
   *will* produce a large step score against a flat null --- so before scanning,
   a low-order Poisson trend is fitted to ``rho(x)`` and the matched null is
   *tilted* by it (``MatchedNull.set_tilt``).  The question then asked is the
   right one: "is there a step **beyond** the smooth gradient?"  This is
   deliberately conservative: the smooth fit absorbs part of a genuine edge.
2. **The look-elsewhere effect is paid for.**  The reported statistic is the
   **maximum** ``|S|`` over the whole scan (all positions, all widths, and for
   the 3D scan all centres), and its p-value comes from the distribution of the
   *same maximum* recomputed on every matched-null realisation.  A scan over
   thousands of cells is therefore not a licence to find one at 3 sigma.

Three geometries
----------------
* ``edge_scan_1d``      --- a step in any scalar coordinate (R_gal, |z|, ...).
* ``edge_scan_shell3d`` --- a spherical shell in heliocentric XYZ, scanned over
  a grid of centres and radii: the literal bubble-boundary test.
* ``edge_scan_cap``     --- a spherical cap on the sky, for the case where the
  boundary is nearer than the distance precision and only projects.
"""

from __future__ import annotations

import numpy as np

from .gradient import bin_index, expected_quantile_edges, poisson_glm
from .nulls import (MIN_ANOMALIES_PER_TEST, MatchedNull, empirical_p,
                    insufficient, p_report)


def _xlogy(n: np.ndarray, r: np.ndarray) -> np.ndarray:
    out = np.zeros_like(n, dtype=float)
    ok = (n > 0) & (r > 0)
    out[ok] = n[ok] * np.log(r[ok])
    return out


def step_score(n_in, e_in, n_out, e_out, *, min_expected: float = 1.0) -> np.ndarray:
    """Signed sqrt of the Poisson likelihood-ratio for a rate step (~sigma)."""
    n_in = np.asarray(n_in, float)
    e_in = np.asarray(e_in, float)
    n_out = np.asarray(n_out, float)
    e_out = np.asarray(e_out, float)
    tot_n, tot_e = n_in + n_out, e_in + e_out
    with np.errstate(divide="ignore", invalid="ignore"):
        rho0 = np.where(tot_e > 0, tot_n / tot_e, np.nan)
        mu_in, mu_out = rho0 * e_in, rho0 * e_out
        lam = 2.0 * (_xlogy(n_in, np.divide(n_in, mu_in, out=np.zeros_like(n_in),
                                            where=mu_in > 0))
                     + _xlogy(n_out, np.divide(n_out, mu_out, out=np.zeros_like(n_out),
                                               where=mu_out > 0)))
        r_in = np.where(e_in > 0, n_in / e_in, np.nan)
        r_out = np.where(e_out > 0, n_out / e_out, np.nan)
    sign = np.sign(r_in - r_out)
    s = sign * np.sqrt(np.maximum(lam, 0.0))
    # Windows with almost no expectation carry no information and would
    # otherwise contribute spurious extremes to the max-statistic.
    bad = (e_in < min_expected) | (e_out < min_expected) | ~np.isfinite(s)
    return np.where(bad, 0.0, s)


def smooth_tilt(coords: dict, null: MatchedNull, *, order: int = 1,
                n_bins: int = 12) -> tuple[np.ndarray, dict]:
    """Fit a low-order smooth rate trend in each coordinate and return the
    per-parent-row tilt that reproduces it (product across coordinates).

    Tilting the null by this makes "smooth gradient" part of the null hypothesis
    for the edge test, so a gradient cannot be reported as an edge.
    """
    tilt = np.ones(len(null.parent))
    info = {}
    for name, x in coords.items():
        x = np.asarray(x, float)
        if not np.isfinite(x).any():
            continue
        edges = expected_quantile_edges(x, null.weights, n_bins)
        if edges.size < 4:
            continue
        b = bin_index(x, edges)
        nb = edges.size - 1
        n_obs = null.observed_binned(b, nb)
        e_exp = null.expected_binned(b, nb)
        centres = 0.5 * (edges[:-1] + edges[1:])
        # Centre and scale so a high-order polynomial stays conditioned.
        s = np.std(centres) or 1.0
        u = (centres - np.mean(centres)) / s
        Z = np.stack([u ** p for p in range(int(order) + 1)], axis=1)
        theta, _, conv = poisson_glm(n_obs, e_exp, Z)
        if not np.all(np.isfinite(theta)):
            continue
        ux = (x - np.mean(centres)) / s
        model = np.exp(np.clip(sum(theta[p] * ux ** p for p in range(int(order) + 1)),
                               -20, 20))
        model = np.where(np.isfinite(model) & (model > 0), model, 1.0)
        tilt = tilt * model
        info[name] = {"order": int(order), "converged": bool(conv),
                      "coefficients": [float(t) for t in theta]}
    tilt = np.where(np.isfinite(tilt) & (tilt > 0), tilt, 1e-6)
    return tilt, info


def _scan_counts(bidx: np.ndarray, weights: np.ndarray, mask: np.ndarray,
                 draws: np.ndarray, n_bins: int):
    """Cumulative observed / expected / per-draw counts over the binned axis."""
    ok = bidx >= 0
    n_obs = np.bincount(bidx[ok & mask], minlength=n_bins).astype(float)
    e_exp = np.bincount(bidx[ok], weights=weights[ok], minlength=n_bins).astype(float)
    n_draw, n_a = draws.shape
    if n_draw:
        b = bidx[draws.ravel()]
        offs = np.repeat(np.arange(n_draw) * n_bins, n_a)
        valid = b >= 0
        flat = np.bincount((b + offs)[valid], minlength=n_draw * n_bins)
        null_counts = flat.reshape(n_draw, n_bins).astype(float)
    else:
        null_counts = np.zeros((0, n_bins))
    return n_obs, e_exp, null_counts


def _best_step(n_cum: np.ndarray, e_cum: np.ndarray, widths, min_expected: float):
    """Max |step score| over positions and widths.  ``n_cum``/``e_cum`` are
    cumulative arrays with a leading zero; ``n_cum`` may be 2D (draws x bins+1)."""
    two_d = n_cum.ndim == 2
    nb = (n_cum.shape[-1] - 1)
    best = np.zeros(n_cum.shape[0]) if two_d else 0.0
    best_at = None
    for k in widths:
        k = int(k)
        if k < 1 or 2 * k > nb:
            continue
        j = np.arange(k, nb - k + 1)
        if j.size == 0:
            continue
        if two_d:
            n_in = n_cum[:, j] - n_cum[:, j - k]
            n_out = n_cum[:, j + k] - n_cum[:, j]
        else:
            n_in = n_cum[j] - n_cum[j - k]
            n_out = n_cum[j + k] - n_cum[j]
        e_in = e_cum[j] - e_cum[j - k]
        e_out = e_cum[j + k] - e_cum[j]
        s = np.abs(step_score(n_in, e_in, n_out, e_out, min_expected=min_expected))
        if two_d:
            best = np.maximum(best, s.max(axis=1))
        else:
            i = int(np.argmax(s))
            if s[i] > best:
                best = float(s[i])
                signed = step_score(n_in[i], e_in[i], n_out[i], e_out[i],
                                    min_expected=min_expected)
                best_at = {"bin": int(j[i]), "half_width_bins": k,
                           "score": float(signed),
                           "n_inside": float(n_in[i]), "n_outside": float(n_out[i]),
                           "expected_inside": float(e_in[i]),
                           "expected_outside": float(e_out[i]),
                           "rho_inside": float(n_in[i] / e_in[i]) if e_in[i] > 0 else None,
                           "rho_outside": float(n_out[i] / e_out[i]) if e_out[i] > 0 else None}
    return best, best_at


def _materialise_draws(null: MatchedNull, n_null: int, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(null.seed + 17 if seed is None else seed)
    rows = [null.draw(rng) for _ in range(int(n_null))]
    if not rows:
        return np.zeros((0, 0), dtype=np.int64)
    n = min(len(r) for r in rows)
    return np.stack([r[:n] for r in rows]).astype(np.int64)


# --- 1D ---------------------------------------------------------------------
def edge_scan_1d(coord: np.ndarray, null: MatchedNull, *, name: str = "coord",
                 n_bins: int = 24, widths=(1, 2, 3, 4, 6), n_null: int = 500,
                 smooth_order: int = 3, min_expected: float = 3.0,
                 seed: int | None = None) -> dict:
    """Scan a scalar coordinate for a step in rate beyond the smooth trend."""
    x = np.asarray(coord, float)
    n_usable = int((null.mask & np.isfinite(x)).sum())
    if n_usable < MIN_ANOMALIES_PER_TEST:
        return insufficient(
            f"only {n_usable} of {int(null.mask.sum())} anomalies have a finite "
            f"{name}; need {MIN_ANOMALIES_PER_TEST}",
            coordinate=name, n_anom=n_usable, n_anom_total=int(null.mask.sum()))
    tilt, tinfo = smooth_tilt({name: x}, null, order=smooth_order)
    tilted = null.copy_with_tilt(tilt)

    edges = expected_quantile_edges(x, tilted.weights, n_bins)
    if edges.size < 6:
        return {"coordinate": name, "insufficient": True,
                "reason": "coordinate does not support enough bins"}
    b = bin_index(x, edges)
    nb = edges.size - 1
    draws = _materialise_draws(tilted, n_null, seed)
    n_obs, e_exp, null_counts = _scan_counts(b, tilted.weights, tilted.mask, draws, nb)

    zc = np.zeros(1)
    n_cum = np.concatenate([zc, np.cumsum(n_obs)])
    e_cum = np.concatenate([zc, np.cumsum(e_exp)])
    obs, best = _best_step(n_cum, e_cum, widths, min_expected)
    null_cum = np.concatenate([np.zeros((null_counts.shape[0], 1)),
                               np.cumsum(null_counts, axis=1)], axis=1)
    null_max, _ = _best_step(null_cum, e_cum, widths, min_expected)

    out = {"coordinate": name, "insufficient": False, "n_bins": int(nb),
           "smooth_model": tinfo, "max_abs_score": float(obs),
           "null_max_mean": float(np.mean(null_max)) if null_max.size else float("nan"),
           "null_max_p95": float(np.percentile(null_max, 95)) if null_max.size else float("nan"),
           "p_value": empirical_p(obs, null_max, tail="greater")}
    if best is not None:
        lo, hi = float(edges[best["bin"]]), float(edges[best["bin"]])
        out["best_edge"] = dict(best, position=lo,
                                inner_range=[float(edges[max(best["bin"] - best["half_width_bins"], 0)]), hi],
                                outer_range=[lo, float(edges[min(best["bin"] + best["half_width_bins"], nb)])])
    out["significant"] = bool(np.isfinite(out["p_value"]) and out["p_value"] < 0.05)
    return out


# --- 3D shell ---------------------------------------------------------------
def centre_grid(xyz: np.ndarray, n_per_axis: int = 4, pad: float = 0.15) -> np.ndarray:
    """A coarse grid of candidate bubble centres spanning the sampled volume."""
    ok = np.all(np.isfinite(xyz), axis=1)
    if ok.sum() < 10:
        return np.zeros((0, 3))
    p = xyz[ok]
    lo, hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    axes = [np.linspace(lo[i], hi[i], int(n_per_axis)) for i in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.vstack([np.zeros((1, 3)), g])          # include the Sun as a centre


def edge_scan_shell3d(xyz: np.ndarray, null: MatchedNull, *, centres=None,
                      n_per_axis: int = 4, n_bins: int = 20, widths=(1, 2, 3, 4),
                      n_null: int = 300, min_expected: float = 3.0,
                      smooth_coords: dict | None = None, smooth_order: int = 3,
                      seed: int | None = None) -> dict:
    """Scan spherical shells about a grid of centres for a step in anomaly rate.

    The literal test of a colonised volume with a boundary.  The maximum |score|
    is taken over centres x radii x widths, and calibrated against the same
    maximum recomputed on every matched-null draw --- so the enormous trials
    factor of the scan is paid for, not ignored.
    """
    xyz = np.asarray(xyz, float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N,3)")
    # Count the anomalies with a finite 3D position, NOT the catalogue total.
    # A catalogue can carry thousands of anomalies of which a handful have a
    # parallax; scanning the handful while guarded by the thousands is how a
    # meaningless statistic acquires a small p-value.
    finite_xyz = np.all(np.isfinite(xyz), axis=1)
    n_usable = int((null.mask & finite_xyz).sum())
    if n_usable < MIN_ANOMALIES_PER_TEST:
        return insufficient(
            f"only {n_usable} of {int(null.mask.sum())} anomalies have finite 3D "
            f"positions; need {MIN_ANOMALIES_PER_TEST}",
            n_anom=n_usable, n_anom_total=int(null.mask.sum()))

    tinfo = {}
    tilted = null
    if smooth_coords:
        tilt, tinfo = smooth_tilt(smooth_coords, null, order=smooth_order)
        tilted = null.copy_with_tilt(tilt)

    if centres is None:
        centres = centre_grid(xyz, n_per_axis=n_per_axis)
        # An over-dense region drags the anomaly centroid toward itself, so the
        # centroid (and the excess-weighted centroid) are cheap, well-motivated
        # extra candidate centres that sharpen localisation far more than a
        # finer grid would.
        good = np.all(np.isfinite(xyz), axis=1) & null.mask
        if good.sum() >= 10:
            cen = xyz[good].mean(axis=0)
            base = xyz[np.all(np.isfinite(xyz), axis=1)].mean(axis=0)
            centres = np.vstack([centres, cen[None, :],
                                 (cen + 2.0 * (cen - base))[None, :]])
    centres = np.atleast_2d(np.asarray(centres, float))
    if centres.size == 0:
        return {"insufficient": True, "reason": "no usable centres"}

    draws = _materialise_draws(tilted, n_null, seed)
    w = tilted.weights
    best_overall, best_info = 0.0, None
    null_max = np.zeros(draws.shape[0])
    for c in centres:
        r = np.sqrt(np.sum((xyz - c) ** 2, axis=1))
        edges = expected_quantile_edges(r, w, n_bins)
        if edges.size < 6:
            continue
        b = bin_index(r, edges)
        nb = edges.size - 1
        n_obs, e_exp, null_counts = _scan_counts(b, w, tilted.mask, draws, nb)
        zc = np.zeros(1)
        n_cum = np.concatenate([zc, np.cumsum(n_obs)])
        e_cum = np.concatenate([zc, np.cumsum(e_exp)])
        obs, best = _best_step(n_cum, e_cum, widths, min_expected)
        if obs > best_overall and best is not None:
            best_overall = float(obs)
            best_info = dict(best, centre=[float(v) for v in c],
                             radius=float(edges[best["bin"]]))
        if null_counts.shape[0]:
            null_cum = np.concatenate([np.zeros((null_counts.shape[0], 1)),
                                       np.cumsum(null_counts, axis=1)], axis=1)
            nm, _ = _best_step(null_cum, e_cum, widths, min_expected)
            null_max = np.maximum(null_max, nm)

    out = {"insufficient": False, "n_centres": int(len(centres)),
           "n_radial_bins": int(n_bins), "smooth_model": tinfo,
           "max_abs_score": best_overall,
           "null_max_mean": float(np.mean(null_max)) if null_max.size else float("nan"),
           "null_max_p95": float(np.percentile(null_max, 95)) if null_max.size else float("nan"),
           "p_value": empirical_p(best_overall, null_max, tail="greater"),
           "best_shell": best_info}
    out["significant"] = bool(np.isfinite(out["p_value"]) and out["p_value"] < 0.05)
    return out


# --- sky cap ----------------------------------------------------------------
def _unit_vectors(l_deg, b_deg) -> np.ndarray:
    lon = np.radians(np.asarray(l_deg, float))
    lat = np.radians(np.asarray(b_deg, float))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=1)


def _fibonacci_directions(n: int) -> np.ndarray:
    i = np.arange(int(n)) + 0.5
    z = 1 - 2 * i / n
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(np.maximum(1 - z ** 2, 0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def edge_scan_cap(l_deg, b_deg, null: MatchedNull, *, n_directions: int = 96,
                  n_bins: int = 18, widths=(1, 2, 3, 4), n_null: int = 300,
                  min_expected: float = 3.0, seed: int | None = None) -> dict:
    """Scan spherical caps on the sky for a step in rate across their boundary.

    The projected version of the bubble test: if the boundary is closer than the
    distance precision resolves, it survives only as a great-circle-like edge in
    projection.
    """
    u = _unit_vectors(l_deg, b_deg)
    finite_dir = np.all(np.isfinite(u), axis=1)
    n_usable = int((null.mask & finite_dir).sum())
    if n_usable < MIN_ANOMALIES_PER_TEST:
        return insufficient(
            f"only {n_usable} of {int(null.mask.sum())} anomalies have finite sky "
            f"directions; need {MIN_ANOMALIES_PER_TEST}",
            n_anom=n_usable, n_anom_total=int(null.mask.sum()))
    dirs = _fibonacci_directions(n_directions)
    draws = _materialise_draws(null, n_null, seed)
    w = null.weights
    best_overall, best_info = 0.0, None
    null_max = np.zeros(draws.shape[0])
    for d in dirs:
        ang = np.degrees(np.arccos(np.clip(u @ d, -1, 1)))
        edges = expected_quantile_edges(ang, w, n_bins)
        if edges.size < 6:
            continue
        b = bin_index(ang, edges)
        nb = edges.size - 1
        n_obs, e_exp, null_counts = _scan_counts(b, w, null.mask, draws, nb)
        zc = np.zeros(1)
        n_cum = np.concatenate([zc, np.cumsum(n_obs)])
        e_cum = np.concatenate([zc, np.cumsum(e_exp)])
        obs, best = _best_step(n_cum, e_cum, widths, min_expected)
        if obs > best_overall and best is not None:
            gl = float(np.degrees(np.arctan2(d[1], d[0])) % 360.0)
            gb = float(np.degrees(np.arcsin(np.clip(d[2], -1, 1))))
            best_overall = float(obs)
            best_info = dict(best, cap_centre_l_deg=gl, cap_centre_b_deg=gb,
                             cap_radius_deg=float(edges[best["bin"]]))
        if null_counts.shape[0]:
            null_cum = np.concatenate([np.zeros((null_counts.shape[0], 1)),
                                       np.cumsum(null_counts, axis=1)], axis=1)
            nm, _ = _best_step(null_cum, e_cum, widths, min_expected)
            null_max = np.maximum(null_max, nm)

    out = {"insufficient": False, "n_directions": int(len(dirs)),
           "max_abs_score": best_overall,
           "null_max_mean": float(np.mean(null_max)) if null_max.size else float("nan"),
           "null_max_p95": float(np.percentile(null_max, 95)) if null_max.size else float("nan"),
           "p_value": empirical_p(best_overall, null_max, tail="greater"),
           "best_cap": best_info}
    out["significant"] = bool(np.isfinite(out["p_value"]) and out["p_value"] < 0.05)
    return out


__all__ = ["edge_scan_1d", "edge_scan_shell3d", "edge_scan_cap", "step_score",
           "smooth_tilt", "centre_grid"]
