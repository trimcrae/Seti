"""The pixel-centroid test for ARC stage 2 --- pure, offline-testable.

A flare catalogue says which *aperture* brightened; the target pixel file says
which *pixel*.  For every flare above the ceiling this module asks whether the
extra light came from the target or from a Gaia neighbour inside the aperture,
two ways:

1. **The difference image** (the Kepler DV method for transits, sign flipped):
   every pixel is detrended against its own out-of-flare baseline, the
   in-flare residual is averaged, and the centroid of that residual image IS
   the position of whatever flared.  Its distance from the target's position
   and from every neighbour's, in units of the propagated error plus a
   systematic floor, attributes the flare or declares it ambiguous.
2. **The flux-weighted centroid shift**: the aperture's centroid moves during
   the flare by ``a / (1 + a) * (c_source - c_baseline)`` where ``a`` is the
   aperture's relative excess.  On the target the predicted shift is tiny (the
   target dominates the baseline centroid); on a neighbour it points toward
   the neighbour.  The measured shift against the baseline scatter is the
   cross-check.

The **census arithmetic** is the number that often settles it without a
pixel: a neighbour supplying fraction ``f_j`` of the aperture flux must itself
brighten by ``a / f_j`` to make the observed aperture amplitude ``a``; a
neighbour that would need more than the largest white-light flare ever seen
is excluded by arithmetic and stated so.

Positions.  The target's pixel position is taken from the baseline image
itself (its own flux-weighted centroid, corrected for the neighbours' known
contributions), and neighbours are placed by their WCS offsets *relative to
the target*, so a WCS zero-point error cancels to first order.  The
difference-image centroid and the baseline centroid are the same estimator
on the same pixels, so PRF asymmetries largely cancel too; what remains is
the configured systematic floor (``verify``: 0.1 px on Kepler's 4" pixels).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erf

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # noqa: B009  numpy 1.x / 2.x

KEPLER_PIXEL_ARCSEC = 3.98
TESS_PIXEL_ARCSEC = 21.0
KEPLER_PRF_SIGMA_PX = 0.7      # verify: Kepler PRF FWHM ~ 1.5-1.7 px -> sigma ~ 0.65-0.7
TESS_PRF_SIGMA_PX = 0.9

OUTCOME_ON_TARGET = "on_target"
OUTCOME_ON_NEIGHBOUR = "on_neighbour"
OUTCOME_AMBIGUOUS = "ambiguous"
OUTCOME_UNATTRIBUTED = "unattributed"
OUTCOME_UNDETECTED = "undetected_in_pixels"
OUTCOME_UNTESTABLE = "untestable"

VERDICT_ON_TARGET = "flare_on_target"
VERDICT_ON_NEIGHBOUR = "flare_on_neighbour"
VERDICT_AMBIGUOUS = "centroid_ambiguous"
VERDICT_UNTESTABLE = "centroid_untestable"


# ---------------------------------------------------------------------------
# PRF and aperture capture
# ---------------------------------------------------------------------------
def pixel_gaussian(shape, x0: float, y0: float, sigma_px: float, flux: float = 1.0) -> np.ndarray:
    """A Gaussian PRF integrated over pixels: pixel ``(i, j)`` spans
    ``[j - 0.5, j + 0.5] x [i - 0.5, i + 0.5]`` around integer centres, so the
    stamp's fraction of the total flux is exact for the model (not renormalised)."""
    ny, nx = int(shape[0]), int(shape[1])
    s = max(float(sigma_px), 1e-6) * math.sqrt(2.0)
    xe = np.arange(nx + 1) - 0.5 - float(x0)
    ye = np.arange(ny + 1) - 0.5 - float(y0)
    fx = 0.5 * (erf(xe[1:] / s) - erf(xe[:-1] / s))
    fy = 0.5 * (erf(ye[1:] / s) - erf(ye[:-1] / s))
    return float(flux) * np.outer(fy, fx)


def aperture_capture(x: float, y: float, aperture, sigma_px: float) -> float:
    """Fraction of a source's PRF flux that falls in the aperture pixels."""
    ap = np.asarray(aperture, dtype=bool)
    if ap.ndim != 2 or not ap.any() or not (np.isfinite(x) and np.isfinite(y)):
        return float("nan")
    return float(pixel_gaussian(ap.shape, x, y, sigma_px)[ap].sum())


def circular_capture(sep_px: float, radius_px: float, sigma_px: float, n: int = 181) -> float:
    """Fraction of a 2-D Gaussian centred ``sep`` from the centre of a circular
    aperture of ``radius`` that lands inside it (numerical, polar grid)."""
    if not (np.isfinite(sep_px) and np.isfinite(radius_px) and radius_px > 0):
        return float("nan")
    s2 = 2.0 * max(float(sigma_px), 1e-6) ** 2
    r = np.linspace(0.0, float(radius_px), n)
    th = np.linspace(0.0, 2.0 * math.pi, n)
    rr, tt = np.meshgrid(r, th, indexing="ij")
    d2 = rr ** 2 + float(sep_px) ** 2 - 2.0 * rr * float(sep_px) * np.cos(tt)
    dens = np.exp(-d2 / s2) / (math.pi * s2)
    integrand = dens * rr
    return float(_trapz(_trapz(integrand, th, axis=1), r))


# ---------------------------------------------------------------------------
# the neighbour census: who could have supplied the flare
# ---------------------------------------------------------------------------
def census(sources: list[dict], *, target_index: int, aperture=None, prf_sigma_px: float,
           aperture_amplitude: float, max_neighbour_amplitude: float = 20.0,
           radius_px: float | None = None) -> tuple[list[dict], dict]:
    """Per source: its flux ratio to the target, its capture fraction into the
    aperture (from the aperture mask, or a circle of ``radius_px``), the
    fraction of the aperture flux it supplies, and the relative amplitude it
    would need to produce ``aperture_amplitude``.  A neighbour needing more
    than ``max_neighbour_amplitude`` (a 2000 % brightening by default,
    ``verify``: beyond the largest white-light M-dwarf flares on record) is
    ``excluded_by_arithmetic``.  Sources carry ``x``, ``y`` (pixels), ``gmag``
    and an ``id``."""
    out = []
    if not sources or target_index < 0 or target_index >= len(sources):
        return [], {"n_sources": 0, "n_neighbours": 0, "target_flux_fraction": float("nan"),
                    "contamination": float("nan"), "n_neighbours_excluded": 0,
                    "n_neighbours_not_excluded": 0, "statement": "no census: no sources"}
    gt = float(sources[target_index].get("gmag", np.nan))
    ratios, caps = [], []
    for j, s in enumerate(sources):
        g = float(s.get("gmag", np.nan))
        ratio = 10.0 ** (-0.4 * (g - gt)) if np.isfinite(g) and np.isfinite(gt) else float("nan")
        if j == target_index and not np.isfinite(ratio):
            ratio = 1.0
        x, y = float(s.get("x", np.nan)), float(s.get("y", np.nan))
        if aperture is not None:
            cap = aperture_capture(x, y, aperture, prf_sigma_px)
        elif radius_px is not None:
            xt, yt = float(sources[target_index]["x"]), float(sources[target_index]["y"])
            cap = circular_capture(math.hypot(x - xt, y - yt), float(radius_px), prf_sigma_px)
        else:
            cap = 1.0
        ratios.append(ratio)
        caps.append(cap)
    captured = np.array([r * c if np.isfinite(r) and np.isfinite(c) else np.nan
                         for r, c in zip(ratios, caps, strict=True)])
    total = float(np.nansum(captured))
    a = float(aperture_amplitude)
    n_ex = n_ok = 0
    for j, s in enumerate(sources):
        frac = captured[j] / total if total > 0 and np.isfinite(captured[j]) else float("nan")
        need = a / frac if np.isfinite(frac) and frac > 0 and np.isfinite(a) else float("nan")
        is_t = j == target_index
        excluded = (not is_t) and np.isfinite(need) and need > float(max_neighbour_amplitude)
        rec = dict(s)
        rec.update({"is_target": bool(is_t), "flux_ratio_to_target": float(ratios[j]),
                    "capture_fraction": float(caps[j]), "flux_fraction": float(frac),
                    "required_amplitude": float(need), "excluded_by_arithmetic": bool(excluded),
                    "note": ("the target itself" if is_t else
                             ("no G magnitude: arithmetic cannot be run" if not np.isfinite(need)
                              else (f"would need a {need * 100:.0f} % brightening of its own "
                                    f"light to make the aperture's {a * 100:.3f} % --- excluded"
                                    if excluded else
                                    f"could supply the aperture's {a * 100:.3f} % with a "
                                    f"{need * 100:.1f} % brightening of its own light --- "
                                    "NOT excluded")))})
        if not is_t:
            n_ex += int(excluded)
            n_ok += int(not excluded)
        out.append(rec)
    tf = out[target_index]["flux_fraction"]
    summary = {"n_sources": len(out), "n_neighbours": len(out) - 1,
               "target_flux_fraction": float(tf),
               "contamination": float(1.0 - tf) if np.isfinite(tf) else float("nan"),
               "n_neighbours_excluded": int(n_ex), "n_neighbours_not_excluded": int(n_ok),
               "aperture_amplitude": a, "max_neighbour_amplitude": float(max_neighbour_amplitude)}
    if len(out) == 1:
        summary["statement"] = "no Gaia neighbour in the census radius: nothing else to flare on"
    elif n_ok == 0:
        summary["statement"] = (f"all {n_ex} neighbours excluded by arithmetic: each would need "
                                f"more than a {max_neighbour_amplitude * 100:.0f} % brightening")
    else:
        summary["statement"] = (f"{n_ok} of {len(out) - 1} neighbours could supply the flare; "
                                f"{n_ex} excluded by arithmetic")
    return out, summary


# ---------------------------------------------------------------------------
# cadence selection
# ---------------------------------------------------------------------------
def flare_cadence_masks(time, t_start: float, t_end: float, *, cadence_days: float,
                        baseline_days: float = 1.0, exclude_cadences: int = 3,
                        quality=None, core_times=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(in_flare, baseline, core)`` masks on the pixel-file time axis.

    In-flare is ``[t_start - cad/2, t_end + cad/2]``; the baseline is every good
    cadence within ``baseline_days`` of the flare's middle but outside the
    flare padded by ``exclude_cadences``; ``core`` is the subset of in-flare
    cadences whose times match ``core_times`` (the light curve's
    signal-dominated cadences) to half a cadence, or all of in-flare when none
    is given.  Quality-flagged cadences are in no mask."""
    t = np.asarray(time, dtype=float)
    good = np.isfinite(t)
    if quality is not None:
        good &= (np.asarray(quality) == 0)
    cad = float(cadence_days)
    in_m = good & (t >= float(t_start) - 0.5 * cad) & (t <= float(t_end) + 0.5 * cad)
    mid = 0.5 * (float(t_start) + float(t_end))
    pad = float(exclude_cadences) * cad
    near = good & (np.abs(t - mid) <= float(baseline_days))
    base = near & ~((t >= float(t_start) - pad) & (t <= float(t_end) + pad))
    if core_times is not None and len(core_times):
        ct = np.asarray(core_times, dtype=float)
        core = in_m & np.array([np.min(np.abs(ct - v)) <= 0.5 * cad if np.isfinite(v) else False
                                for v in t])
        if not core.any():
            core = in_m.copy()
    else:
        core = in_m.copy()
    return in_m, base, core


# ---------------------------------------------------------------------------
# centroids
# ---------------------------------------------------------------------------
def image_centroid(image, *, err_image=None, threshold: float = 0.2, min_pixels: int = 3,
                   restrict=None) -> dict:
    """Thresholded flux-weighted centroid ``(x, y)`` of an image with the
    analytic error from ``err_image``; ``restrict`` limits the moment to a
    pixel mask.  ``0`` is the first pixel."""
    img = np.asarray(image, dtype=float)
    out = {"x": float("nan"), "y": float("nan"), "x_err": float("nan"), "y_err": float("nan"),
           "n_pixels": 0, "peak": float("nan"), "flux": float("nan"), "ok": False}
    if img.ndim != 2 or not np.isfinite(img).any():
        return out
    w = np.where(np.isfinite(img), img, 0.0)
    if restrict is not None:
        w = np.where(np.asarray(restrict, dtype=bool), w, 0.0)
    peak = float(w.max())
    out["peak"] = peak
    if not (np.isfinite(peak) and peak > 0):
        return out
    keep = (w >= float(threshold) * peak) & (w > 0)
    out["n_pixels"] = int(keep.sum())
    if int(keep.sum()) < int(min_pixels):
        return out
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ww = np.where(keep, w, 0.0)
    sw = float(ww.sum())
    x = float((ww * xx).sum() / sw)
    y = float((ww * yy).sum() / sw)
    out.update({"x": x, "y": y, "flux": sw, "ok": True})
    if err_image is not None:
        e = np.asarray(err_image, dtype=float)
        e = np.where(np.isfinite(e) & keep, e, 0.0)
        out["x_err"] = float(math.sqrt(float((((xx - x) / sw) ** 2 * e ** 2).sum())))
        out["y_err"] = float(math.sqrt(float((((yy - y) / sw) ** 2 * e ** 2).sum())))
    return out


def aperture_centroid_series(cube, aperture) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cadence flux-weighted centroid over the aperture pixels: ``(x, y, flux)``."""
    c = np.asarray(cube, dtype=float)
    ap = np.asarray(aperture, dtype=bool)
    ny, nx = ap.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    w = np.where(ap[None, :, :] & np.isfinite(c), c, 0.0)
    f = w.sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (w * xx[None]).sum(axis=(1, 2)) / f
        y = (w * yy[None]).sum(axis=(1, 2)) / f
    return x, y, f


def _fit_trend(t_base, v_base, t_eval, order: int = 1):
    """Least-squares polynomial trend on the baseline evaluated at ``t_eval``;
    falls back to the mean when the baseline is too short."""
    tb, vb = np.asarray(t_base, dtype=float), np.asarray(v_base, dtype=float)
    ok = np.isfinite(tb) & np.isfinite(vb)
    tb, vb = tb[ok], vb[ok]
    te = np.asarray(t_eval, dtype=float)
    if tb.size == 0:
        return np.full(te.shape, np.nan), np.full(tb.shape, np.nan)
    if tb.size < 2 * (order + 1) + 2:
        m = float(np.mean(vb))
        return np.full(te.shape, m), vb - m
    t0 = float(np.mean(tb))
    coef = np.polyfit(tb - t0, vb, order)
    return np.polyval(coef, te - t0), vb - np.polyval(coef, tb - t0)


def centroid_shift(time, cube, aperture, in_mask, base_mask, *, trend_order: int = 1) -> dict:
    """The flux-weighted aperture centroid in flare minus its baseline trend.

    Returns ``dx``, ``dy`` (px), their errors from the baseline residual
    scatter (``s * sqrt(1/n_in + 1/n_base)``), the relative aperture excess
    ``a`` (``F_in / F_trend - 1``) and the baseline centroid ``(x0, y0)``.
    """
    t = np.asarray(time, dtype=float)
    x, y, f = aperture_centroid_series(cube, aperture)
    inm, bm = np.asarray(in_mask, dtype=bool), np.asarray(base_mask, dtype=bool)
    inm &= np.isfinite(x) & np.isfinite(y) & np.isfinite(f)
    bm &= np.isfinite(x) & np.isfinite(y) & np.isfinite(f)
    out = {"dx": float("nan"), "dy": float("nan"), "dx_err": float("nan"), "dy_err": float("nan"),
           "n_in": int(inm.sum()), "n_base": int(bm.sum()), "a": float("nan"),
           "x0": float("nan"), "y0": float("nan"), "ok": False}
    if inm.sum() < 1 or bm.sum() < 4:
        return out
    px, rx = _fit_trend(t[bm], x[bm], t[inm], trend_order)
    py, ry = _fit_trend(t[bm], y[bm], t[inm], trend_order)
    pf, _ = _fit_trend(t[bm], f[bm], t[inm], trend_order)
    sx, sy = float(np.std(rx, ddof=1)), float(np.std(ry, ddof=1))
    n_in, n_b = int(inm.sum()), int(bm.sum())
    scale = math.sqrt(1.0 / n_in + 1.0 / n_b)
    out.update({"dx": float(np.mean(x[inm] - px)), "dy": float(np.mean(y[inm] - py)),
                "dx_err": sx * scale, "dy_err": sy * scale,
                "a": float(np.mean(f[inm] / pf - 1.0)),
                "x0": float(np.mean(x[bm])), "y0": float(np.mean(y[bm])),
                "baseline_scatter_px": float(math.hypot(sx, sy)), "ok": True})
    return out


def predicted_shift(a: float, source_xy, baseline_xy) -> tuple[float, float]:
    """``a / (1 + a) * (c_source - c_baseline)``: where the aperture centroid
    goes if source ``c_source`` supplies the whole relative excess ``a``."""
    if not np.isfinite(a) or a <= -1:
        return float("nan"), float("nan")
    k = a / (1.0 + a)
    return (k * (float(source_xy[0]) - float(baseline_xy[0])),
            k * (float(source_xy[1]) - float(baseline_xy[1])))


def difference_image(time, cube, in_mask, base_mask, *, err_cube=None, trend_order: int = 1
                     ) -> dict:
    """Per-pixel detrended in-flare residual image and its per-pixel error.

    Every pixel gets a polynomial trend in time from the baseline cadences;
    the in-flare residual is averaged.  The error is the baseline residual
    scatter over ``sqrt(n_in)``, floored at the propagated photon error when
    ``err_cube`` is supplied.  Returns ``image``, ``err``, ``n_in``, ``n_base``,
    ``snr_peak`` and the baseline mean image.
    """
    t = np.asarray(time, dtype=float)
    c = np.asarray(cube, dtype=float)
    inm, bm = np.asarray(in_mask, dtype=bool), np.asarray(base_mask, dtype=bool)
    fin = np.isfinite(c).all(axis=(1, 2)) & np.isfinite(t)
    inm, bm = inm & fin, bm & fin
    n_in, n_b = int(inm.sum()), int(bm.sum())
    out = {"image": None, "err": None, "n_in": n_in, "n_base": n_b, "snr_peak": float("nan"),
           "baseline_image": None, "ok": False}
    if n_in < 1 or n_b < 4:
        return out
    tb, t0 = t[bm], float(np.mean(t[bm]))
    order = int(trend_order) if n_b >= 2 * (trend_order + 1) + 2 else 0
    a_mat = np.vander(tb - t0, order + 1)
    fb = c[bm].reshape(n_b, -1)
    coef, *_ = np.linalg.lstsq(a_mat, fb, rcond=None)
    resid_b = fb - a_mat @ coef
    a_in = np.vander(t[inm] - t0, order + 1)
    resid_in = c[inm].reshape(n_in, -1) - a_in @ coef
    shape = c.shape[1:]
    d = resid_in.mean(axis=0).reshape(shape)
    sd = resid_b.std(axis=0, ddof=1).reshape(shape) / math.sqrt(n_in)
    if err_cube is not None:
        e = np.asarray(err_cube, dtype=float)
        if e.shape == c.shape:
            with np.errstate(invalid="ignore"):
                phot = np.sqrt(np.nanmean(e[inm] ** 2, axis=0) / n_in)
            sd = np.where(np.isfinite(phot), np.maximum(sd, phot), sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = d / sd
    out.update({"image": d, "err": sd, "baseline_image": c[bm].mean(axis=0),
                "snr_peak": float(np.nanmax(snr)) if np.isfinite(snr).any() else float("nan"),
                "snr_image": snr, "trend_order": order, "ok": True})
    return out


# ---------------------------------------------------------------------------
# positions: anchor the WCS to the baseline image
# ---------------------------------------------------------------------------
def anchor_sources(sources: list[dict], *, target_index: int, baseline_xy) -> list[dict]:
    """Re-anchor WCS pixel positions to the measured baseline centroid.

    ``c_target = (c_base - sum_j f_j c_j) / (1 - sum_j f_j)`` over the
    neighbours' aperture flux fractions, then every neighbour keeps its WCS
    offset *relative to the target*.  Each source gains ``x_est`` / ``y_est``;
    the WCS-to-image offset is returned on every source as ``anchor_dx`` /
    ``anchor_dy`` for the record."""
    out = [dict(s) for s in sources]
    if not out or not (np.isfinite(baseline_xy[0]) and np.isfinite(baseline_xy[1])):
        for s in out:
            s["x_est"], s["y_est"] = float(s.get("x", np.nan)), float(s.get("y", np.nan))
            s["anchor_dx"] = s["anchor_dy"] = float("nan")
        return out
    tx, ty = float(out[target_index]["x"]), float(out[target_index]["y"])
    sx = sy = sf = 0.0
    for j, s in enumerate(out):
        if j == target_index:
            continue
        f = float(s.get("flux_fraction", np.nan))
        if not (np.isfinite(f) and np.isfinite(s.get("x", np.nan))):
            continue
        sx += f * float(s["x"])
        sy += f * float(s["y"])
        sf += f
    if sf >= 1.0 or not np.isfinite(sf):
        sx = sy = sf = 0.0
    cx = (float(baseline_xy[0]) - sx) / (1.0 - sf)
    cy = (float(baseline_xy[1]) - sy) / (1.0 - sf)
    for s in out:
        s["anchor_dx"], s["anchor_dy"] = cx - tx, cy - ty
        s["x_est"] = cx + (float(s["x"]) - tx)
        s["y_est"] = cy + (float(s["y"]) - ty)
    return out


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------
def attribute_flare(diff: dict, sources: list[dict], *, target_index: int,
                    sys_floor_px: float = 0.1, n_sigma: float = 3.0, min_snr: float = 3.0,
                    threshold: float = 0.2, min_pixels: int = 3, shift: dict | None = None,
                    unresolved_px: float = 0.5) -> dict:
    """Which source the difference image points at.

    ``sources`` carry ``x_est`` / ``y_est`` (anchored positions), ``id``,
    ``excluded_by_arithmetic`` and ``flux_fraction``.  The difference-image
    centroid ``c_D`` with error ``sigma`` (analytic (+) ``sys_floor_px``) gives
    ``z_j = |c_D - c_j| / sigma`` per source; the target is *consistent* when
    ``z < n_sigma`` and a neighbour is *consistent* when ``z < n_sigma`` and
    it is not excluded by arithmetic.  A neighbour within ``unresolved_px`` of
    the target that the arithmetic does not exclude makes the flare
    ``ambiguous`` whatever the centroid says: the pixels cannot separate them.
    """
    out = {"outcome": OUTCOME_UNTESTABLE, "reason": "", "x_diff": float("nan"),
           "y_diff": float("nan"), "sigma_px": float("nan"), "snr_peak": float("nan"),
           "z_target": float("nan"), "d_target_px": float("nan"),
           "consistent_neighbours": [], "excluded_neighbours": [], "nearest_source": None,
           "n_pixels": 0, "sources": []}
    if not diff or not diff.get("ok"):
        out["reason"] = "no difference image (too few in-flare or baseline cadences)"
        return out
    out["snr_peak"] = float(diff.get("snr_peak", np.nan))
    if not (np.isfinite(out["snr_peak"]) and out["snr_peak"] >= float(min_snr)):
        out.update({"outcome": OUTCOME_UNDETECTED,
                    "reason": f"difference-image peak SNR {out['snr_peak']:.1f} < {min_snr}"})
        return out
    cen = image_centroid(diff["image"], err_image=diff["err"], threshold=threshold,
                         min_pixels=min_pixels)
    out["n_pixels"] = int(cen["n_pixels"])
    if not cen["ok"]:
        out.update({"outcome": OUTCOME_UNDETECTED,
                    "reason": f"difference-image centroid needs >= {min_pixels} pixels above "
                              f"{threshold:.0%} of the peak; got {cen['n_pixels']}"})
        return out
    err = math.hypot(cen["x_err"] if np.isfinite(cen["x_err"]) else 0.0,
                     cen["y_err"] if np.isfinite(cen["y_err"]) else 0.0) / math.sqrt(2.0)
    sigma = math.hypot(err, float(sys_floor_px))
    out.update({"x_diff": cen["x"], "y_diff": cen["y"], "sigma_px": sigma,
                "centroid_stat_err_px": err})
    if not sources or target_index < 0 or target_index >= len(sources):
        out.update({"outcome": OUTCOME_UNTESTABLE, "reason": "no source positions"})
        return out
    tx, ty = float(sources[target_index].get("x_est", np.nan)), \
        float(sources[target_index].get("y_est", np.nan))
    rows, consistent, excluded, unresolved = [], [], [], []
    nearest, nearest_d = None, float("inf")
    for j, s in enumerate(sources):
        x, y = float(s.get("x_est", np.nan)), float(s.get("y_est", np.nan))
        d = math.hypot(cen["x"] - x, cen["y"] - y) if np.isfinite(x) and np.isfinite(y) \
            else float("nan")
        z = d / sigma if np.isfinite(d) else float("nan")
        is_t = j == target_index
        ex = bool(s.get("excluded_by_arithmetic", False)) and not is_t
        cons = bool(np.isfinite(z) and z < float(n_sigma))
        sep_t = math.hypot(x - tx, y - ty) if np.isfinite(x) and np.isfinite(tx) else float("nan")
        row = {"id": str(s.get("id", j)), "is_target": is_t, "x_est": x, "y_est": y,
               "d_px": d, "z": z, "consistent": cons, "excluded_by_arithmetic": ex,
               "sep_from_target_px": sep_t,
               "flux_fraction": float(s.get("flux_fraction", np.nan)),
               "required_amplitude": float(s.get("required_amplitude", np.nan))}
        if shift and shift.get("ok"):
            px, py = predicted_shift(shift["a"], (x, y), (shift["x0"], shift["y0"]))
            row["pred_dx"], row["pred_dy"] = px, py
            if np.isfinite(px) and shift["dx_err"] > 0 and shift["dy_err"] > 0:
                row["shift_chi2"] = float(((shift["dx"] - px) / shift["dx_err"]) ** 2
                                          + ((shift["dy"] - py) / shift["dy_err"]) ** 2)
        rows.append(row)
        if is_t:
            out["z_target"], out["d_target_px"] = z, d
        elif ex:
            excluded.append(row["id"])
        else:
            if cons:
                consistent.append(row["id"])
            if np.isfinite(sep_t) and sep_t <= float(unresolved_px):
                unresolved.append(row["id"])
        if np.isfinite(d) and d < nearest_d and not ex:
            nearest, nearest_d = row["id"], d
    out.update({"sources": rows, "consistent_neighbours": consistent,
                "excluded_neighbours": excluded, "nearest_source": nearest,
                "unresolved_neighbours": unresolved})
    t_ok = bool(np.isfinite(out["z_target"]) and out["z_target"] < float(n_sigma))
    if unresolved:
        out.update({"outcome": OUTCOME_AMBIGUOUS,
                    "reason": f"neighbour(s) {unresolved} within {unresolved_px} px of the target "
                              "are not excluded by arithmetic: the pixels cannot separate them"})
    elif t_ok and not consistent:
        out.update({"outcome": OUTCOME_ON_TARGET,
                    "reason": f"difference-image centroid {out['d_target_px']:.3f} px "
                              f"({out['z_target']:.1f} sigma) from the target; every neighbour "
                              f"rejected at > {n_sigma} sigma or excluded by arithmetic"})
    elif consistent and not t_ok:
        out.update({"outcome": OUTCOME_ON_NEIGHBOUR, "neighbour": consistent[0],
                    "reason": f"difference-image centroid {out['z_target']:.1f} sigma from the "
                              f"target and consistent with neighbour {consistent[0]}"})
    elif t_ok and consistent:
        out.update({"outcome": OUTCOME_AMBIGUOUS,
                    "reason": f"centroid consistent with the target AND with {consistent}"})
    else:
        out.update({"outcome": OUTCOME_UNATTRIBUTED,
                    "reason": f"centroid {out['d_target_px']:.2f} px ({out['z_target']:.1f} sigma)"
                              " from the target and consistent with no catalogued source"})
    return out


def star_verdict(outcomes: list[str]) -> tuple[str, str]:
    """The star's stage-2 verdict from its per-flare outcomes."""
    tested = [o for o in outcomes if o in (OUTCOME_ON_TARGET, OUTCOME_ON_NEIGHBOUR,
                                             OUTCOME_AMBIGUOUS, OUTCOME_UNATTRIBUTED)]
    if not tested:
        why = ", ".join(sorted(set(outcomes))) if outcomes else "no flare reached the pixel test"
        return VERDICT_UNTESTABLE, f"no attributable flare ({why})"
    n_t = sum(1 for o in tested if o == OUTCOME_ON_TARGET)
    n_n = sum(1 for o in tested if o == OUTCOME_ON_NEIGHBOUR)
    if n_n:
        return VERDICT_ON_NEIGHBOUR, f"{n_n} of {len(tested)} attributable flares on a neighbour"
    if n_t == len(tested):
        return VERDICT_ON_TARGET, f"all {n_t} attributable flares on the target"
    return VERDICT_AMBIGUOUS, (f"{n_t} of {len(tested)} on the target, the rest ambiguous or "
                               "unattributed")


# ---------------------------------------------------------------------------
# synthetic pixel files (the test gate)
# ---------------------------------------------------------------------------
def synth_flare_tpf(*, sources, flare_source: int, t_peak: float, amplitude: float,
                    decay_days: float = 0.05, shape=(9, 9), prf_sigma_px: float = KEPLER_PRF_SIGMA_PX,
                    cadence_days: float = 29.4244 / 1440.0, span_days: float = 3.0,
                    noise: float = 0.0, seed: int = 11, aperture_radius_px: float = 2.0,
                    target_index: int = 0, drift_px_per_day: float = 0.0,
                    pixel_scale_arcsec: float = KEPLER_PIXEL_ARCSEC) -> dict:
    """A synthetic target pixel file with a flare injected on a **known** source.

    ``sources`` is a list of ``{"x", "y", "flux"}``; the flare is an
    impulsive rise over one cadence and an exponential decay of ``decay_days``
    on source ``flare_source`` with peak relative amplitude ``amplitude`` (of
    that source's own flux).  Slow pointing drift and white noise are
    optional.  The record has the shape the stage-2 test consumes: ``time``,
    ``flux`` (n, ny, nx), ``flux_err``, ``quality``, ``aperture`` (a disc of
    ``aperture_radius_px`` around the target), ``pixel_scale_arcsec`` and a
    ``pix_fn`` that maps the sources' own ``(x, y)`` "sky" onto pixels.
    """
    rng = np.random.default_rng(int(seed))
    n = int(round(2 * float(span_days) / float(cadence_days))) + 1
    t = float(t_peak) - float(span_days) + np.arange(n) * float(cadence_days)
    dt = t - float(t_peak)
    prof = np.where(dt >= 0, np.exp(-dt / max(float(decay_days), 1e-9)), 0.0)
    rise = (dt < 0) & (dt >= -float(cadence_days))
    prof[rise] = (dt[rise] + float(cadence_days)) / float(cadence_days)
    cube = np.zeros((n, int(shape[0]), int(shape[1])))
    for j, s in enumerate(sources):
        for i in range(n):
            dx = float(drift_px_per_day) * (t[i] - t[0])
            gain = 1.0 + (float(amplitude) * prof[i] if j == int(flare_source) else 0.0)
            cube[i] += pixel_gaussian(shape, s["x"] + dx, s["y"] + 0.5 * dx, prf_sigma_px,
                                      float(s.get("flux", 1.0)) * gain)
    if float(noise) > 0:
        cube = cube + rng.normal(0.0, float(noise), size=cube.shape)
    err = np.full(cube.shape, float(noise) if noise > 0 else np.nan)
    tgt = sources[int(target_index)]
    yy, xx = np.mgrid[0:int(shape[0]), 0:int(shape[1])]
    ap = np.hypot(xx - float(tgt["x"]), yy - float(tgt["y"])) <= float(aperture_radius_px)
    return {"mission": "kepler", "segment": 0, "time": t, "flux": cube, "flux_err": err,
            "quality": np.zeros(n, dtype=int), "aperture": ap,
            "pixel_scale_arcsec": float(pixel_scale_arcsec), "exptime_s": float(cadence_days) * 86400.0,
            "pix_fn": lambda ra, dec: (float(ra), float(dec)), "column": 0, "row": 0,
            "author": "synthetic", "n_cadences": int(n), "target_pixel": (float(tgt["x"]), float(tgt["y"]))}


__all__ = ["KEPLER_PIXEL_ARCSEC", "KEPLER_PRF_SIGMA_PX", "OUTCOME_AMBIGUOUS", "OUTCOME_ON_NEIGHBOUR",
           "OUTCOME_ON_TARGET", "OUTCOME_UNATTRIBUTED", "OUTCOME_UNDETECTED", "OUTCOME_UNTESTABLE",
           "TESS_PIXEL_ARCSEC", "TESS_PRF_SIGMA_PX", "VERDICT_AMBIGUOUS", "VERDICT_ON_NEIGHBOUR",
           "VERDICT_ON_TARGET", "VERDICT_UNTESTABLE", "anchor_sources", "aperture_capture",
           "aperture_centroid_series", "attribute_flare", "census", "centroid_shift",
           "circular_capture", "difference_image", "flare_cadence_masks", "image_centroid",
           "pixel_gaussian", "predicted_shift", "star_verdict", "synth_flare_tpf"]
