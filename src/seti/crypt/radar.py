"""The radar axis of CRYPT: compact circular-polarisation-ratio anomalies.

A metre-scale metallic dihedral (a corner reflector, a hull edge, a strut
against a plate) returns S-band with CPR > 1 in a single Mini-RF pixel and
nothing around it.  The natural high-CPR source is a rock field — a fresh
crater's floor, walls and ejecta — which is *extended* and sits in an
elevated-CPR neighbourhood.  So the statistic is:

* CPR ≥ ``cpr_min`` inside the PSR interior (the Diviner cold-trap mask
  re-sampled onto the radar grid);
* the connected patch of CPR ≥ ``cpr_min`` containing the pixel has at most
  ``max_cluster_px`` pixels (``rock_field`` otherwise);
* the median CPR in an annulus ``bg_r_in``–``bg_r_out`` around it is below
  ``bg_max`` (``elevated_background`` otherwise: ejecta, roughness);
* total power S1 (when the layer exists) above the annulus median by
  ``s1_contrast_min`` (``weak_echo`` otherwise: a CPR ratio of two noisy
  small numbers is not a target);
* not within ``edge_px`` of the mask edge (``mask_edge``).

Pure functions; the rasters come from ``acquire``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .labels import Georef

DEFAULT_RADAR: dict = {
    "cpr_min": 1.0,
    "max_cluster_px": 4,
    "bg_r_in": 3,
    "bg_r_out": 12,
    "bg_max": 0.6,
    "s1_contrast_min": 3.0,
    "edge_px": 2,
    "max_rows": 5000,
}

RADAR_CLASSES = ("radar_candidate", "rock_field", "elevated_background", "weak_echo", "mask_edge")


def resample_mask(mask: np.ndarray, src: Georef, dst: Georef) -> np.ndarray:
    """Nearest-neighbour re-sampling of a boolean mask from grid ``src`` onto grid ``dst``."""
    ii, jj = np.mgrid[0:dst.lines, 0:dst.samples]
    lon, lat = dst.pix_to_lonlat(ii, jj)
    si, sj = src.lonlat_to_pix(lon, lat)
    si = np.rint(si).astype(int)
    sj = np.rint(sj).astype(int)
    ok = (si >= 0) & (si < mask.shape[0]) & (sj >= 0) & (sj < mask.shape[1])
    out = np.zeros((dst.lines, dst.samples), dtype=bool)
    out[ok] = mask[si[ok], sj[ok]]
    return out


def _annulus(r_in: int, r_out: int) -> np.ndarray:
    y, x = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
    r = np.hypot(y, x)
    return (r >= r_in) & (r <= r_out)


def screen_radar(cpr: np.ndarray, mask: np.ndarray, georef: Georef, thr: dict | None = None,
                 s1: np.ndarray | None = None) -> dict:
    """The compact-CPR screen on one radar raster inside a PSR mask on the same grid."""
    thr = {**DEFAULT_RADAR, **(thr or {})}
    if cpr.shape != mask.shape:
        raise ValueError(f"cpr {cpr.shape} and mask {mask.shape} differ")
    edge = int(thr["edge_px"])
    interior = ndimage.binary_erosion(mask, iterations=edge, border_value=0) if edge else mask
    valid = np.isfinite(cpr) & mask
    high = valid & (cpr >= float(thr["cpr_min"]))
    lab, n_lab = ndimage.label(high, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(lab.ravel()) if n_lab else np.array([0])
    ann = _annulus(int(thr["bg_r_in"]), int(thr["bg_r_out"]))
    cpr_f = np.where(np.isfinite(cpr), cpr, np.nan)
    ii, jj = np.nonzero(high)
    rows = []
    n_total = int(high.sum())
    if n_total > int(thr["max_rows"]):
        order = np.argsort(-cpr[ii, jj])[: int(thr["max_rows"])]
        ii, jj = ii[order], jj[order]
    lon, lat = georef.pix_to_lonlat(ii, jj)
    ro = int(thr["bg_r_out"])
    for n_, (i, j) in enumerate(zip(ii.tolist(), jj.tolist(), strict=True)):
        reasons = []
        size = int(sizes[lab[i, j]])
        if size > int(thr["max_cluster_px"]):
            reasons.append("rock_field")
        if not interior[i, j]:
            reasons.append("mask_edge")
        i0, i1 = max(0, i - ro), min(cpr.shape[0], i + ro + 1)
        j0, j1 = max(0, j - ro), min(cpr.shape[1], j + ro + 1)
        win = cpr_f[i0:i1, j0:j1]
        a = ann[(i0 - (i - ro)):(i0 - (i - ro)) + (i1 - i0), (j0 - (j - ro)):(j0 - (j - ro)) + (j1 - j0)]
        bg = win[a]
        bg = bg[np.isfinite(bg)]
        bg_med = float(np.median(bg)) if bg.size else np.nan
        if np.isfinite(bg_med) and bg_med > float(thr["bg_max"]):
            reasons.append("elevated_background")
        s1_contrast = np.nan
        if s1 is not None:
            w1 = s1[i0:i1, j0:j1]
            b1 = w1[a]
            b1 = b1[np.isfinite(b1)]
            if b1.size and np.isfinite(s1[i, j]):
                m1 = float(np.median(b1))
                s1_contrast = float(s1[i, j] / m1) if m1 > 0 else np.inf
                if s1_contrast < float(thr["s1_contrast_min"]):
                    reasons.append("weak_echo")
        rows.append({"line": i, "sample": j, "lon": float(lon[n_]), "lat": float(lat[n_]),
                     "cpr": float(cpr[i, j]), "cluster_px": size, "bg_cpr_median": bg_med,
                     "s1_contrast": s1_contrast, "reasons": ";".join(reasons),
                     "class": _classify(reasons)})
    flags = pd.DataFrame(rows)
    counts = {c: int((flags["class"] == c).sum()) if len(flags) else 0 for c in RADAR_CLASSES}
    return {"status": "OK", "n_mask_px": int(mask.sum()), "n_valid_px": int(valid.sum()),
            "n_high_cpr": n_total, "counts": counts, "flags": flags, "thresholds": thr,
            "s1_available": s1 is not None}


def _classify(reasons: list[str]) -> str:
    for o in ("mask_edge", "rock_field", "elevated_background", "weak_echo"):
        if o in reasons:
            return o
    return "radar_candidate"


def sample_at(raster: np.ndarray, georef: Georef, lon, lat, *, radius_px: int = 2) -> dict:
    """The raster value at (lon, lat) and the median in a (2r+1)² window: the
    reading the thermal vet takes from a radar mosaic."""
    i, j = georef.lonlat_to_pix(lon, lat)
    i, j = int(np.rint(float(i))), int(np.rint(float(j)))
    if not (0 <= i < raster.shape[0] and 0 <= j < raster.shape[1]):
        return {"in_raster": False}
    i0, i1 = max(0, i - radius_px), min(raster.shape[0], i + radius_px + 1)
    j0, j1 = max(0, j - radius_px), min(raster.shape[1], j + radius_px + 1)
    win = raster[i0:i1, j0:j1]
    win = win[np.isfinite(win)]
    v = float(raster[i, j])
    return {"in_raster": True, "line": i, "sample": j, "value": v if np.isfinite(v) else None,
            "window_median": float(np.median(win)) if win.size else None,
            "window_max": float(np.max(win)) if win.size else None, "n_valid": int(win.size)}


def synthetic_radar(n_px: int = 160, *, seed: int = 5, mask_radius_px: int = 45,
                    mean_cpr: float = 0.3, sigma: float = 0.12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cpr, s1, mask) on a square grid: a quiet floor with Gaussian CPR scatter."""
    rng = np.random.default_rng(seed)
    cpr = np.clip(rng.normal(mean_cpr, sigma, (n_px, n_px)), 0.0, None).astype(np.float32)
    s1 = np.clip(rng.normal(1.0, 0.2, (n_px, n_px)), 0.05, None).astype(np.float32)
    yy, xx = np.mgrid[0:n_px, 0:n_px]
    mask = np.hypot(yy - (n_px - 1) / 2.0, xx - (n_px - 1) / 2.0) < mask_radius_px
    return cpr, s1, mask


__all__ = ["DEFAULT_RADAR", "RADAR_CLASSES", "resample_mask", "sample_at", "screen_radar",
           "synthetic_radar"]
