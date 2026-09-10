"""S42 -- the sub-exposure flash (``docs/roman.md`` section 2.3).

A pulse of light shorter than one up-the-ramp resultant deposits charge that
*stays*, so in a non-destructive-read detector it looks exactly like a cosmic
ray: a step in the ramp, and a ``JUMP_DET`` flag from the pipeline's jump
detector.  What separates it from a cosmic ray is morphology and coincidence.
A cosmic ray hits one pixel or a short track, anywhere; a flash lights the
whole PSF footprint of a catalogued star at the same resultant, with a step
amplitude map that *is* the PSF.  Roman is the first wide-field survey with
non-destructive reads, so this is the first time the pulsed-beacon question of
optical SETI can be asked of ~10^8 stars at once.

Two stages, both pure and offline-testable:

**Stage 1 -- Level 2, ``dq`` only.**  :func:`screen_dq_cutout` labels every
8-connected component of the jump bit.  The labels are the rejection rules of
the spec, applied in a fixed order (:func:`classify_cluster`): size window,
snowball, elongation (a track), the hot-pixel ledger, the star match, the
saturation exclusion, the footprint coverage.  Survivors are
``psf_on_star`` and become ``PSF_JUMP_ON_STAR_PENDING_RAMP`` records.

**Stage 2 -- Level 1, candidates only.**  :func:`ramp_flash_test` fits every
pixel of the PSF footprint with slope-before / step / slope-after at every
resultant and asks three things of the footprint: the step is at *one*
resultant in >= 80 % of the pixels; the slope after equals the slope before
(a flare keeps rising -- its ramp is a slope change, not a step); and the step
amplitude map is correlated with the PSF at >= 0.8.  :func:`assess_candidates`
then applies the per-pixel recurrence veto retroactively, reports (never
vetoes) per-star recurrence, and assigns the tier strings.

Every number comes from ``config/roman.yaml`` ``flash:``; nothing here is a
measurement of Roman.  The synthesis helpers at the bottom build ``dq`` images
and ramps from the same PSF model the detectors use, for the tests and the
selftest, and are the only place anything is made up -- and they say so in the
``image_id`` they stamp.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.special import erf

from .schema import DQ_FLAGS_FALLBACK, DQCutout, Funnel, Ramp

_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))

LABELS = ("cosmic_ray_track", "snowball", "psf_off_star", "psf_on_star", "hot_pixel",
          "near_saturated", "too_small", "too_large", "footprint_incomplete")
RAMP_VERDICTS = ("FLASH_CONSISTENT", "SLOPE_CHANGE_FLARE_LIKE", "NOT_COINCIDENT",
                 "NOT_PSF_SHAPED", "INSUFFICIENT_RESULTANTS")
_ELECTRON_UNITS = {"e-", "e", "electron", "electrons", "e-/s"}
# Reduced chi-square above which a pixel's residual scatter overrides the pooled sigma.
_RESIDUAL_INFLATE_CHI2_DOF = 3.0

_FLASH_DEFAULTS: dict[str, Any] = {
    "jump_flag": "JUMP_DET", "saturated_flag": "SATURATED", "persistence_flag": "PERSISTENCE",
    "min_cluster_px": 3, "max_cluster_px": 40, "centroid_tolerance_fwhm": 0.75,
    "elongation_max": 1.6, "footprint_frac": 0.10, "footprint_flagged_min": 0.6,
    "snowball_min_px": 30, "saturation_exclusion_px": 3, "hot_pixel_min_exposures": 3,
    "step_snr_min": 5.0, "slope_change_sigma_max": 3.0, "psf_amplitude_corr_min": 0.8,
    "resultant_coincidence_min_frac": 0.8, "next_exposure_residual_sigma_max": 3.0,
    "n_sigma_trigger": 5.0,
    "tiers": {"interest": "PSF_JUMP_ON_STAR_PENDING_RAMP",
              "candidate": "SUB_EXPOSURE_FLASH_CANDIDATE"},
}


def flash_conf(conf: dict | None) -> dict:
    """The ``flash:`` block, whether handed the whole Roman config or the block.

    Missing keys fall back to the values in ``config/roman.yaml`` as of writing,
    so a detector never crashes on a trimmed config -- but the fallback is
    recorded nowhere, so callers that care pass the full config.
    """
    conf = conf or {}
    block = conf.get("flash") if isinstance(conf.get("flash"), dict) else conf
    out = dict(_FLASH_DEFAULTS)
    out["tiers"] = dict(_FLASH_DEFAULTS["tiers"])
    for k, v in block.items():
        if k == "tiers" and isinstance(v, dict):
            out["tiers"].update({str(a): str(b) for a, b in v.items()})
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------------------
# PSF model, footprints, geometry
# --------------------------------------------------------------------------------------

def gaussian_psf(fwhm_px: float, size: int, center: tuple[float, float] | None = None) -> np.ndarray:
    """A normalised 2-D Gaussian stamp of odd ``size`` (even sizes are widened by one).

    ``center`` is ``(y, x)`` in stamp coordinates; default the middle pixel.  The
    stamp is *sampled* at pixel centres, not integrated over pixels -- the same
    convention the footprint and the amplitude-map correlation use, so a
    footprint and the model it is compared with never disagree about where the
    peak is.  The synthesis helpers integrate over pixels instead, on purpose:
    the detector must recover a signal built with a slightly different PSF.
    """
    size = int(size)
    if size % 2 == 0:
        size += 1
    size = max(size, 1)
    sigma = max(float(fwhm_px), 1e-3) * _FWHM_TO_SIGMA
    cy, cx = ((size - 1) / 2.0, (size - 1) / 2.0) if center is None else (float(center[0]), float(center[1]))
    yy, xx = np.mgrid[0:size, 0:size]
    g = np.exp(-0.5 * (((yy - cy) ** 2) + ((xx - cx) ** 2)) / sigma ** 2)
    s = g.sum()
    return g / s if s > 0 else g


def footprint_radius_px(fwhm_px: float, frac: float) -> float:
    """Radius at which the Gaussian falls to ``frac`` of its peak."""
    frac = min(max(float(frac), 1e-6), 0.999999)
    return max(float(fwhm_px), 1e-3) * _FWHM_TO_SIGMA * math.sqrt(-2.0 * math.log(frac))


def psf_footprint(fwhm_px: float, frac: float) -> np.ndarray:
    """Boolean stamp of the pixels above ``frac`` of the peak, odd size >= 5, star at the centre."""
    r = footprint_radius_px(fwhm_px, frac)
    size = max(5, 2 * int(math.ceil(r)) + 1)
    psf = gaussian_psf(fwhm_px, size)
    return psf >= float(frac) * psf.max()


def footprint_pixels(star_xy: tuple[float, float], fwhm_px: float, frac: float,
                     shape: tuple[int, int] | None = None) -> np.ndarray:
    """The ``(y, x)`` pixels of a star's PSF footprint at a sub-pixel position.

    The Gaussian is evaluated at pixel centres around ``star_xy = (x, y)`` and
    the footprint is every pixel at or above ``frac`` of the *brightest pixel on
    the grid* -- not of the analytic peak, which for a star centred between
    pixels no pixel reaches.  That is what a per-pixel jump detector sees.
    ``shape`` clips to the image.  Returns an ``(n, 2)`` int array.
    """
    x, y = float(star_xy[0]), float(star_xy[1])
    r = footprint_radius_px(fwhm_px, frac)
    half = int(math.ceil(r)) + 1
    cy, cx = int(round(y)), int(round(x))
    yy, xx = np.mgrid[cy - half:cy + half + 1, cx - half:cx + half + 1]
    sigma = max(float(fwhm_px), 1e-3) * _FWHM_TO_SIGMA
    g = np.exp(-0.5 * (((yy - y) ** 2) + ((xx - x) ** 2)) / sigma ** 2)
    keep = g >= float(frac) * g.max()
    if shape is not None:
        keep &= (yy >= 0) & (yy < int(shape[0])) & (xx >= 0) & (xx < int(shape[1]))
    return np.stack([yy[keep], xx[keep]], axis=1).astype(int)


def _elongation(pixels: np.ndarray) -> float:
    """sqrt of the ratio of principal second moments, with the 1/12 pixel variance added.

    The pixel term keeps a single pixel at 1.0 and a straight 3-pixel line at
    exactly 3.0; without it every line would be infinitely elongated and every
    single pixel undefined.
    """
    p = np.asarray(pixels, dtype=float)
    if p.shape[0] < 2:
        return 1.0
    c = p - p.mean(axis=0)
    cov = c.T @ c / p.shape[0] + np.eye(2) / 12.0
    ev = np.linalg.eigvalsh(cov)
    return float(math.sqrt(max(ev[1], 1e-12) / max(ev[0], 1e-12)))


# --------------------------------------------------------------------------------------
# Stage 1: clusters of the jump bit
# --------------------------------------------------------------------------------------

def find_jump_clusters(dq: np.ndarray, jump_bit: int, min_px: int, max_px: int,
                       saturated_bit: int = 0, snowball_min_px: int = 30,
                       elongation_max: float = 1.6) -> list[dict]:
    """8-connected components of ``(dq & jump_bit) != 0``, with their measured shape.

    Every component is returned, including those outside ``[min_px, max_px]``,
    so the funnel can count what it rejected; ``size_class`` says where each
    sits: ``small`` (below the window), ``psf_like`` (inside it), ``snowball``
    (at least ``snowball_min_px``, roughly round, and with a SATURATED pixel
    inside -- the large charge blobs the H4RG/H2RG literature calls snowballs),
    or ``large`` (above the window and not a snowball).  Each dict carries
    ``pixels`` ((n, 2) int array of (y, x)), ``n``, ``centroid`` (y, x),
    ``elongation``, ``bbox`` ((y0, y1, x0, x1), exclusive upper bounds) and
    ``has_saturated_core``.
    """
    dq = np.asarray(dq)
    if dq.ndim != 2 or dq.size == 0 or not jump_bit:
        return []
    d = dq.astype(np.int64, copy=False)
    jump = (d & int(jump_bit)) != 0
    if not jump.any():
        return []
    sat = (d & int(saturated_bit)) != 0 if saturated_bit else np.zeros_like(jump)
    labels, n_lab = ndimage.label(jump, structure=np.ones((3, 3), dtype=int))
    out: list[dict] = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        sub = labels[sl] == i
        yx = np.argwhere(sub)
        yx[:, 0] += sl[0].start
        yx[:, 1] += sl[1].start
        n = int(yx.shape[0])
        elong = _elongation(yx)
        has_sat = bool(sat[sl][sub].any())
        if n >= int(snowball_min_px) and elong <= float(elongation_max) and has_sat:
            size_class = "snowball"
        elif n < int(min_px):
            size_class = "small"
        elif n <= int(max_px):
            size_class = "psf_like"
        else:
            size_class = "large"
        out.append({
            "pixels": yx, "n": n,
            "centroid": (float(yx[:, 0].mean()), float(yx[:, 1].mean())),
            "elongation": elong,
            "bbox": (int(sl[0].start), int(sl[0].stop), int(sl[1].start), int(sl[1].stop)),
            "has_saturated_core": has_sat, "size_class": size_class,
        })
    return out


def match_cluster_to_stars(cluster: dict, stars: list[dict], fwhm_px: float,
                           tol_fwhm: float) -> dict | None:
    """The nearest catalogued star within ``tol_fwhm`` FWHM of the cluster centroid, or ``None``.

    Returns ``{"star": <the star dict>, "index", "distance_px", "distance_fwhm"}``.
    """
    if not stars:
        return None
    cy, cx = cluster["centroid"]
    xs = np.array([float(s.get("x", np.nan)) for s in stars])
    ys = np.array([float(s.get("y", np.nan)) for s in stars])
    d = np.hypot(xs - cx, ys - cy)
    d[~np.isfinite(d)] = np.inf
    j = int(np.argmin(d))
    tol = float(tol_fwhm) * max(float(fwhm_px), 1e-3)
    if not np.isfinite(d[j]) or d[j] > tol:
        return None
    return {"star": stars[j], "index": j, "distance_px": float(d[j]),
            "distance_fwhm": float(d[j] / max(float(fwhm_px), 1e-3))}


def footprint_coverage(cluster_pixels: np.ndarray, star_xy: tuple[float, float], fwhm_px: float,
                       frac: float, shape: tuple[int, int] | None = None) -> float:
    """Fraction of the star's PSF-footprint pixels that are in the cluster."""
    fp = footprint_pixels(star_xy, fwhm_px, frac, shape)
    if fp.shape[0] == 0:
        return 0.0
    have = {(int(y), int(x)) for y, x in np.asarray(cluster_pixels).reshape(-1, 2)}
    hit = sum(1 for y, x in fp if (int(y), int(x)) in have)
    return float(hit) / float(fp.shape[0])


def _nearest_saturated_px(dq_sat: np.ndarray, star_xy: tuple[float, float], radius: float) -> float | None:
    """Distance to the nearest SATURATED pixel within ``radius`` of the star, else ``None``."""
    x, y = float(star_xy[0]), float(star_xy[1])
    r = int(math.ceil(radius)) + 1
    ny, nx = dq_sat.shape
    y0, y1 = max(0, int(round(y)) - r), min(ny, int(round(y)) + r + 1)
    x0, x1 = max(0, int(round(x)) - r), min(nx, int(round(x)) + r + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    win = dq_sat[y0:y1, x0:x1]
    if not win.any():
        return None
    yy, xx = np.nonzero(win)
    d = np.hypot(yy + y0 - y, xx + x0 - x)
    dmin = float(d.min())
    return dmin if dmin <= float(radius) else None


class HotPixelLedger:
    """How many exposures have flagged each absolute detector pixel with the jump bit.

    A pixel that jumps in >= ``hot_pixel_min_exposures`` exposures is a defect
    (a hot, telegraph or persistence-prone pixel), whatever star sits under it.
    Keyed on ``(detector, y, x)`` in *detector* coordinates so cutouts of the
    same detector at different offsets accumulate into one count.  ``add`` is
    idempotent per ``image_id``: screening the same exposure twice does not
    count twice.  Cosmic-ray pixels are random and rarely recur, so a run over
    many exposures should :meth:`prune` singletons periodically; the ledger is
    JSON-serialisable through ``as_dict``/``from_dict`` so it survives between
    workflow runs.
    """

    def __init__(self) -> None:
        self._counts: dict[str, dict[tuple[int, int], int]] = defaultdict(dict)
        self._images: dict[str, set[str]] = defaultdict(set)

    def add(self, image_id: str, detector: str, pixels) -> int:
        """Count ``pixels`` ((y, x) pairs, detector frame) for ``image_id``; returns pixels added."""
        det = str(detector)
        if str(image_id) in self._images[det]:
            return 0
        self._images[det].add(str(image_id))
        table = self._counts[det]
        arr = np.asarray(pixels).reshape(-1, 2) if pixels is not None else np.zeros((0, 2), int)
        for y, x in np.unique(arr, axis=0) if arr.shape[0] else arr:
            key = (int(y), int(x))
            table[key] = table.get(key, 0) + 1
        return int(arr.shape[0])

    def count(self, detector: str, y: int, x: int) -> int:
        return int(self._counts.get(str(detector), {}).get((int(y), int(x)), 0))

    def hot(self, detector: str, y: int, x: int, min_exposures: int) -> bool:
        return self.count(detector, y, x) >= int(min_exposures)

    def n_images(self, detector: str) -> int:
        return len(self._images.get(str(detector), ()))

    def hot_pixels(self, detector: str, min_exposures: int) -> list[tuple[int, int, int]]:
        """``(y, x, count)`` for every pixel at or above ``min_exposures`` on ``detector``."""
        return sorted((y, x, c) for (y, x), c in self._counts.get(str(detector), {}).items()
                      if c >= int(min_exposures))

    def prune(self, min_count: int = 2) -> int:
        """Drop pixels seen fewer than ``min_count`` times; returns how many were dropped."""
        dropped = 0
        for det in list(self._counts):
            table = self._counts[det]
            for key in [k for k, c in table.items() if c < int(min_count)]:
                del table[key]
                dropped += 1
        return dropped

    def as_dict(self) -> dict:
        return {"detectors": {det: {"images": sorted(self._images.get(det, ())),
                                    "pixels": [[int(y), int(x), int(c)] for (y, x), c in sorted(table.items())]}
                              for det, table in self._counts.items()}}

    @classmethod
    def from_dict(cls, d: dict | None) -> HotPixelLedger:
        led = cls()
        for det, row in ((d or {}).get("detectors") or {}).items():
            led._images[str(det)] = {str(i) for i in row.get("images", [])}
            led._counts[str(det)] = {(int(y), int(x)): int(c) for y, x, c in row.get("pixels", [])}
        return led


def classify_cluster(cluster: dict, stars: list[dict], dq: np.ndarray, conf: dict,
                     flags: dict[str, int], hot_ledger: HotPixelLedger | None = None,
                     detector: str = "", x0: int = 0, y0: int = 0,
                     fwhm_px: float | None = None) -> dict:
    """Apply the stage-1 rejection rules to one cluster and return the label with its numbers.

    Order: ``too_small``; ``snowball``; ``too_large``; ``cosmic_ray_track``
    (elongation); ``hot_pixel`` (any pixel at or above the ledger threshold);
    the star match (none within tolerance: ``psf_off_star``);
    ``near_saturated`` (a SATURATED pixel within the exclusion radius of the
    star, or the catalogue saying the star saturates); ``footprint_incomplete``
    (coverage below the minimum); else ``psf_on_star``.  The snowball test is
    asked *before* the upper size bound because it is the only rule that can
    explain a cluster above the window -- otherwise every snowball would be
    an anonymous ``too_large`` and the funnel would not know it saw them.
    ``fwhm_px`` defaults to the ``psf_fwhm_px`` the caller puts on the cutout.
    """
    fc = flash_conf(conf)
    fwhm = float(fwhm_px if fwhm_px is not None else 1.2)
    sat_bit = int(flags.get(fc["saturated_flag"], 0) or 0)
    n = int(cluster["n"])
    elong = float(cluster["elongation"])
    out: dict[str, Any] = {
        "n_px": n, "elongation": elong, "centroid_yx": [float(c) for c in cluster["centroid"]],
        "size_class": cluster.get("size_class"), "has_saturated_core": bool(cluster.get("has_saturated_core")),
        "star_id": None, "distance_px": None, "distance_fwhm": None, "coverage": None,
        "saturated_distance_px": None, "hot_pixel_max_count": 0,
    }
    if n < int(fc["min_cluster_px"]):
        out["label"] = "too_small"
        return out
    if cluster.get("size_class") == "snowball":
        out["label"] = "snowball"
        return out
    if n > int(fc["max_cluster_px"]):
        out["label"] = "too_large"
        return out
    if elong > float(fc["elongation_max"]):
        out["label"] = "cosmic_ray_track"
        return out
    if hot_ledger is not None:
        counts = [hot_ledger.count(detector, int(y) + int(y0), int(x) + int(x0)) for y, x in cluster["pixels"]]
        out["hot_pixel_max_count"] = int(max(counts)) if counts else 0
        if out["hot_pixel_max_count"] >= int(fc["hot_pixel_min_exposures"]):
            out["label"] = "hot_pixel"
            return out
    m = match_cluster_to_stars(cluster, stars, fwhm, float(fc["centroid_tolerance_fwhm"]))
    if m is None:
        out["label"] = "psf_off_star"
        return out
    star = m["star"]
    out["star_id"] = str(star.get("star_id", m["index"]))
    out["star_xy"] = [float(star["x"]), float(star["y"])]
    out["distance_px"] = m["distance_px"]
    out["distance_fwhm"] = m["distance_fwhm"]
    star_xy = (float(star["x"]), float(star["y"]))
    if sat_bit:
        dsat = _nearest_saturated_px((np.asarray(dq).astype(np.int64) & sat_bit) != 0, star_xy,
                                     float(fc["saturation_exclusion_px"]))
        out["saturated_distance_px"] = dsat
        if dsat is not None:
            out["label"] = "near_saturated"
            return out
    if bool(star.get("saturated", False)):
        out["label"] = "near_saturated"
        return out
    cov = footprint_coverage(cluster["pixels"], star_xy, fwhm, float(fc["footprint_frac"]),
                             shape=np.asarray(dq).shape)
    out["coverage"] = cov
    if cov < float(fc["footprint_flagged_min"]):
        out["label"] = "footprint_incomplete"
        return out
    out["label"] = "psf_on_star"
    return out


def screen_dq_cutout(cut: DQCutout, conf: dict, flags: dict[str, int],
                     hot_ledger: HotPixelLedger | None = None) -> dict:
    """Stage 1 on one cutout: label every jump cluster, return the ``psf_on_star`` records.

    ``flags`` is the bit table the *product* carries (``products.dq_flags``),
    not the fallback: a product with no jump flag cannot be screened, and the
    result then says ``status: "jump_flag_unavailable"`` with no candidates,
    which the funnel records as a test not run rather than passed.  A missing
    SATURATED flag only disables the saturation rules and is noted.  When a
    ledger is given the cutout's jump pixels are added to it *before*
    classification, so the third exposure to flag a pixel is the one that
    labels it ``hot_pixel``.  Candidate centroids are in detector coordinates
    (``x0``/``y0`` added) and each record carries its detector pixels so the
    assess stage can apply per-pixel recurrence across exposures.
    """
    fc = flash_conf(conf)
    funnel = Funnel()
    base = {"image_id": cut.image_id, "detector": cut.detector, "band": cut.band, "mjd": cut.mjd,
            "n_clusters": 0, "labels": {}, "candidates": []}
    jump_name, sat_name = str(fc["jump_flag"]), str(fc["saturated_flag"])
    flags = {str(k): int(v) for k, v in (flags or {}).items()}
    if jump_name not in flags or not flags[jump_name]:
        funnel.note(f"{cut.image_id}: dq flag table carries no {jump_name}; screen not run")
        funnel.bump("cutouts_without_jump_flag")
        return {**base, "status": "jump_flag_unavailable", "funnel": funnel.as_dict()}
    jump_bit = flags[jump_name]
    sat_bit = flags.get(sat_name, 0)
    if not sat_bit:
        funnel.note(f"{cut.image_id}: no {sat_name} flag; saturation and snowball rules not run")
        funnel.bump("cutouts_without_saturated_flag")
    dq = np.asarray(cut.dq).astype(np.int64)
    funnel.bump("cutouts")
    funnel.bump("jump_pixels", int(((dq & jump_bit) != 0).sum()))
    clusters = find_jump_clusters(dq, jump_bit, int(fc["min_cluster_px"]), int(fc["max_cluster_px"]),
                                  saturated_bit=sat_bit, snowball_min_px=int(fc["snowball_min_px"]),
                                  elongation_max=float(fc["elongation_max"]))
    funnel.bump("clusters", len(clusters))
    if hot_ledger is not None and clusters:
        allpix = np.concatenate([c["pixels"] for c in clusters], axis=0)
        hot_ledger.add(cut.image_id, cut.detector, allpix + np.array([int(cut.y0), int(cut.x0)]))
    labels: dict[str, int] = {}
    cands: list[dict] = []
    for cl in clusters:
        res = classify_cluster(cl, cut.stars, dq, fc, flags, hot_ledger, detector=cut.detector,
                               x0=cut.x0, y0=cut.y0, fwhm_px=cut.psf_fwhm_px)
        lab = res["label"]
        labels[lab] = labels.get(lab, 0) + 1
        if lab != "psf_on_star":
            funnel.reject(lab)
            continue
        cy, cx = cl["centroid"]
        cands.append({
            "star_id": res["star_id"], "image_id": cut.image_id, "detector": cut.detector,
            "band": cut.band, "mjd": cut.mjd, "exposure_s": cut.exposure_s,
            "n_resultants": cut.n_resultants,
            "centroid_x": float(cx + cut.x0), "centroid_y": float(cy + cut.y0),
            "centroid_x_cutout": float(cx), "centroid_y_cutout": float(cy),
            "star_x": float(res["star_xy"][0] + cut.x0), "star_y": float(res["star_xy"][1] + cut.y0),
            "n_px": int(cl["n"]), "coverage": float(res["coverage"]), "elongation": float(cl["elongation"]),
            "distance_fwhm": float(res["distance_fwhm"]),
            "pixels": [[int(y + cut.y0), int(x + cut.x0)] for y, x in cl["pixels"]],
            "hot_pixel_max_count": int(res["hot_pixel_max_count"]),
            "tier": fc["tiers"]["interest"],
        })
    funnel.bump("psf_on_star", len(cands))
    return {**base, "status": "ok", "n_clusters": len(clusters), "labels": labels,
            "candidates": cands, "funnel": funnel.as_dict()}


# --------------------------------------------------------------------------------------
# Stage 2: the ramp
# --------------------------------------------------------------------------------------

def _lstsq(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least squares with the unscaled parameter covariance and the residual sum of squares."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    try:
        cov = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(X.T @ X)
    return beta, cov, float(resid @ resid)


def _nan_fit(status: str, n: int) -> dict:
    return {"status": status, "n": int(n), "k": None, "step": math.nan, "step_err": math.nan,
            "step_snr": math.nan, "s_before": math.nan, "s_after": math.nan,
            "slope_change_sigma": math.nan, "chi2": math.nan, "rss": math.nan, "dof": 0,
            "sigma_used": math.nan, "sigma_source": "none", "slope_after_constrained": False,
            "n_before": 0, "n_after": 0}


def fit_step_ramp(y, t, k: int, sigma: float | None = None, single_slope: bool = False) -> dict:
    """Slope-before / step / slope-after at resultant ``k`` for one pixel's ramp.

    Model: ``y = a + s_before * t`` for ``i < k``, and
    ``y = a + s_before * t_k + step + s_after * (t - t_k)`` for ``i >= k`` -- a
    step at ``t_k`` with an independent slope after it.  With fewer than two
    points on either side the two slopes cannot both be measured (one point
    before makes step and slope collinear), so the slope after is fixed equal to
    the slope before and ``slope_after_constrained`` is True with
    ``slope_change_sigma = 0``; that is the price of the minimal GBTDS ramp of 6
    resultants, where ``k`` in {1, 5} is only ever a step test.
    ``single_slope=True`` imposes that constraint at every ``k``: it is the
    *flash hypothesis* itself (charge lands, the rate does not change) and the
    step it measures is the more precise one; the free model at the same ``k``
    then asks whether the slope changed.

    ``step_err`` is the parameter error scaled by the pooled ``sigma`` (the
    caller's read-plus-shot noise for the pixel) when one is given, replaced by
    the residual scatter only when that scatter contradicts it (reduced
    chi-square above 3, with at least 3 degrees of freedom and more than 4
    resultants: the data then say the pixel is noisier than the noise model
    claims, and the data win; ``sigma_source`` says ``residual_inflated``).
    Without ``sigma`` the residual scatter is used, and with too few degrees
    of freedom for that to mean anything ``sigma_source`` says
    ``residual_low_dof``.  On a 6-resultant ramp the step error is ~1.7 sigma
    even under the single-slope model: three points a side barely pin the
    slope, and the slope error leveraged across the gap dominates the step.  ``k`` must be in ``[1, n - 1]``; ``n < 4`` returns
    ``status: "insufficient_resultants"`` with NaNs.
    """
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    n = int(y.size)
    if n < 4 or t.size != n:
        return _nan_fit("insufficient_resultants", n)
    k = int(k)
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be in [1, n-1]; got {k} for n={n}")
    tk = t[k]
    after = np.arange(n) >= k
    n_before, n_after = int(k), int(n - k)
    constrained = bool(single_slope) or n_before < 2 or n_after < 2
    if constrained:
        X = np.column_stack([np.ones(n), t, after.astype(float)])
    else:
        tb = np.where(after, tk, t)
        ta = np.where(after, t - tk, 0.0)
        X = np.column_stack([np.ones(n), tb, after.astype(float), ta])
    beta, cov, rss = _lstsq(X, y)
    p = X.shape[1]
    dof = n - p
    pooled = float(sigma) if (sigma is not None and math.isfinite(float(sigma)) and float(sigma) > 0) else None
    resid = math.sqrt(rss / dof) if (dof >= 3 and n > 4 and rss > 0) else None
    if pooled is not None and resid is not None and (resid / pooled) ** 2 > _RESIDUAL_INFLATE_CHI2_DOF:
        sig, src = resid, "residual_inflated"
    elif pooled is not None:
        sig, src = pooled, "pooled"
    elif resid is not None:
        sig, src = resid, "residual"
    elif dof >= 1 and rss > 0:
        sig, src = math.sqrt(rss / dof), "residual_low_dof"
    else:
        sig, src = math.nan, "undetermined"
    step = float(beta[2])
    step_err = float(math.sqrt(max(cov[2, 2], 0.0)) * sig) if math.isfinite(sig) else math.nan
    s_before = float(beta[1])
    if constrained:
        s_after, dslope_sig = s_before, 0.0
    else:
        s_after = float(beta[3])
        var = cov[3, 3] + cov[1, 1] - 2.0 * cov[1, 3]
        err = math.sqrt(max(var, 0.0)) * sig if math.isfinite(sig) else math.nan
        dslope_sig = (s_after - s_before) / err if (err and math.isfinite(err) and err > 0) else math.nan
    return {"status": "ok", "n": n, "k": k, "step": step, "step_err": step_err,
            "step_snr": (step / step_err) if (step_err and math.isfinite(step_err) and step_err > 0) else math.nan,
            "s_before": s_before, "s_after": s_after, "slope_change_sigma": float(dslope_sig),
            "chi2": (rss / sig ** 2) if (math.isfinite(sig) and sig > 0) else math.nan, "rss": rss,
            "dof": int(dof), "sigma_used": sig, "sigma_source": src,
            "slope_after_constrained": bool(constrained), "n_before": n_before, "n_after": n_after}


def fit_broken_ramp(y, t, k: int) -> dict:
    """The flare alternative: a continuous slope change at ``t_k`` and *no* step (3 parameters).

    Returned for the model comparison in :func:`ramp_flash_test`: a flash is a
    step the broken line cannot follow, a flare is a slope change the step
    model cannot follow, and the residual sums of squares say which.
    """
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    n = int(y.size)
    if n < 4:
        return {"status": "insufficient_resultants", "rss": math.nan, "k": None}
    k = int(k)
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be in [1, n-1]; got {k} for n={n}")
    ta = np.where(np.arange(n) >= k, t - t[k], 0.0)
    X = np.column_stack([np.ones(n), t, ta])
    beta, _cov, rss = _lstsq(X, y)
    return {"status": "ok", "k": k, "rss": rss, "s_before": float(beta[1]),
            "s_after": float(beta[1] + beta[2])}


def best_step_index(y, t, sigma: float | None = None, single_slope: bool = False) -> dict:
    """Scan ``k = 1 .. n-1`` and return the :func:`fit_step_ramp` result with the largest ``step / step_err``.

    Signed, not absolute: a flash deposits charge, so only a positive step is
    the signal; a negative jump is a cosmic-ray flag from the reference-pixel
    correction or a saturated reset, and not this channel's business.  The
    dict also carries ``scan`` -- every k's step SNR -- so a caller can see
    whether the choice was clear-cut.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = int(y.size)
    if n < 4:
        out = _nan_fit("insufficient_resultants", n)
        out["scan"] = []
        return out
    best: dict | None = None
    scan = []
    for k in range(1, n):
        f = fit_step_ramp(y, t, k, sigma, single_slope=single_slope)
        snr = f["step_snr"] if math.isfinite(f["step_snr"]) else -math.inf
        scan.append(float(snr) if math.isfinite(snr) else None)
        if best is None or snr > (best["step_snr"] if math.isfinite(best["step_snr"]) else -math.inf):
            best = f
    assert best is not None
    best = dict(best)
    best["scan"] = scan
    return best


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size < 3 or b.size != a.size:
        return math.nan
    a = a - a.mean()
    b = b - b.mean()
    da, db = math.sqrt(float(a @ a)), math.sqrt(float(b @ b))
    if da <= 0 or db <= 0:
        return 0.0
    return float(a @ b / (da * db))


def _psf_values_at(pixels: np.ndarray, star_xy: tuple[float, float], fwhm_px: float) -> np.ndarray:
    sigma = max(float(fwhm_px), 1e-3) * _FWHM_TO_SIGMA
    dy = pixels[:, 0] - float(star_xy[1])
    dx = pixels[:, 1] - float(star_xy[0])
    return np.exp(-0.5 * (dx ** 2 + dy ** 2) / sigma ** 2)


def psf_amplitude_correlation(pixels: np.ndarray, amplitudes: np.ndarray, star_xy: tuple[float, float],
                              fwhm_px: float, grid_step: float = 0.5, grid_half: float = 1.0) -> dict:
    """Pearson correlation of a per-pixel amplitude map with the PSF, centroid fitted on a coarse grid.

    The catalogue position of the star is good to a fraction of a pixel, not
    better, and the amplitude map has a handful of pixels; a full centroid fit
    would chase the noise.  So the PSF is evaluated on a ``grid_step``-px grid
    within ``grid_half`` px of the catalogue position and the best correlation
    is kept, with the offset reported.
    """
    pixels = np.asarray(pixels).reshape(-1, 2)
    amps = np.asarray(amplitudes, float).ravel()
    if pixels.shape[0] < 3:
        return {"corr": math.nan, "dx": 0.0, "dy": 0.0, "n_px": int(pixels.shape[0]),
                "note": "fewer than 3 footprint pixels: no shape test possible"}
    offs = np.arange(-grid_half, grid_half + 1e-9, grid_step)
    best = (-math.inf, 0.0, 0.0)
    for dy in offs:
        for dx in offs:
            model = _psf_values_at(pixels, (star_xy[0] + dx, star_xy[1] + dy), fwhm_px)
            c = _pearson(amps, model)
            if math.isfinite(c) and c > best[0]:
                best = (c, float(dx), float(dy))
    corr = best[0] if math.isfinite(best[0]) else math.nan
    return {"corr": corr, "dx": best[1], "dy": best[2], "n_px": int(pixels.shape[0])}


def _frames_per_resultant(ramp: Ramp) -> tuple[np.ndarray, str]:
    n = ramp.n_resultants
    rp = ramp.read_pattern
    if rp and len(rp) == n:
        m = np.array([max(1, len(g)) for g in rp], dtype=float)
        return m, "read_pattern"
    return np.ones(n), "assumed_1_frame_per_resultant"


def ramp_flash_test(ramp: Ramp, star_xy_in_box: tuple[float, float], fwhm_px: float, conf: dict,
                    read_noise_e: float, gain: float | None = None) -> dict:
    """Stage 2 on one candidate: is the Level 1 ramp a step across the PSF footprint at one resultant?

    Every pixel of the PSF footprint (above ``footprint_frac`` of the peak,
    around ``star_xy_in_box = (x, y)`` in box coordinates) gets
    :func:`best_step_index` under the flash hypothesis (one slope, one step)
    with a pooled per-pixel sigma -- read noise per resultant plus the shot
    noise of the pixel's own accumulated charge, or the empirical scatter of
    the off-footprint pixels about straight lines, whichever is larger -- and
    the free slope-before / step / slope-after fit at that index.  Then:

    ``coincidence_frac``  fraction of footprint pixels whose best step is at the
        modal resultant with ``step_snr >= step_snr_min``;
    ``slope_change_sigma_max_over_pixels``  the largest |slope_after -
        slope_before| significance among the pixels with a significant step
        (a flare keeps rising: its ramp is a slope change);
    ``delta_chi2_step_minus_flare``  summed over the footprint, the single-slope
        step model's residual chi-square minus the broken-line (flare) model's;
        above ``slope_change_sigma_max**2`` the footprint prefers a flare even
        where no step reached significance;
    ``psf_amplitude_corr``  Pearson correlation of the per-pixel step at the
        modal resultant with the PSF, centroid refined on a 0.5-px grid;
    ``total_step_e``  sum of the modal-resultant steps over the footprint, in
        electrons (``gain`` argument, else the ramp's, else 1 with a note).

    Verdict order: ``INSUFFICIENT_RESULTANTS`` (< 4); ``SLOPE_CHANGE_FLARE_LIKE``;
    ``NOT_COINCIDENT``; ``NOT_PSF_SHAPED``; ``FLASH_CONSISTENT``.
    """
    fc = flash_conf(conf)
    notes: list[str] = []
    n = ramp.n_resultants
    base = {"image_id": ramp.image_id, "n_resultants": n, "verdict": None,
            "coincidence_frac": math.nan, "modal_index": None, "n_footprint_px": 0,
            "n_significant_px": 0, "slope_change_sigma_max_over_pixels": math.nan,
            "delta_chi2_step_minus_flare": math.nan, "psf_amplitude_corr": math.nan,
            "total_step_e": math.nan, "per_pixel": [], "notes": notes}
    if n < 4:
        notes.append(f"{n} resultants: the step model needs at least 4")
        return {**base, "verdict": "INSUFFICIENT_RESULTANTS"}
    if gain is None:
        if str(ramp.unit).lower() in _ELECTRON_UNITS:
            gain = 1.0
        elif ramp.gain_e_per_dn:
            gain = float(ramp.gain_e_per_dn)
        else:
            gain = 1.0
            notes.append("no gain on the ramp and none given: electrons assumed equal to DN")
    gain = float(gain)
    cube = np.asarray(ramp.resultants, dtype=float) * gain
    t = np.asarray(ramp.times_s, dtype=float)
    ny, nx = cube.shape[1], cube.shape[2]
    frames, frames_src = _frames_per_resultant(ramp)
    if frames_src != "read_pattern":
        notes.append("no read pattern on the ramp: read noise per resultant taken as the per-frame value")
    rn2 = float(read_noise_e) ** 2 / frames        # per-resultant read variance, electrons^2

    fp = footprint_pixels(star_xy_in_box, fwhm_px, float(fc["footprint_frac"]), shape=(ny, nx))
    base["n_footprint_px"] = int(fp.shape[0])
    if fp.shape[0] == 0:
        notes.append("the star's footprint falls outside the box")
        return {**base, "verdict": "NOT_PSF_SHAPED"}

    # Empirical noise from the off-footprint pixels: straight-line residual scatter, pooled.
    inside = np.zeros((ny, nx), bool)
    inside[fp[:, 0], fp[:, 1]] = True
    off = np.argwhere(~inside)
    sig_emp = math.nan
    if off.shape[0] >= 8 and n >= 4:
        X = np.column_stack([np.ones(n), t])
        Y = cube[:, off[:, 0], off[:, 1]]
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        R = Y - X @ beta
        sig_emp = math.sqrt(float((R ** 2).sum()) / max(1, R.size - 2 * off.shape[0]))

    per_pixel: list[dict] = []
    dt = np.diff(t)
    for yy, xx in fp:
        ys = cube[:, yy, xx]
        rate = float(np.median(np.diff(ys) / np.where(dt > 0, dt, np.nan))) if n >= 2 else 0.0
        rate = rate if math.isfinite(rate) else 0.0
        shot = max(rate, 0.0) * np.maximum(t, 0.0)          # accumulated Poisson variance
        sig_model = math.sqrt(float(np.mean(rn2 + shot)))
        sig = max(sig_model, sig_emp) if math.isfinite(sig_emp) else sig_model
        sig = max(sig, 1e-6)
        best = best_step_index(ys, t, sig, single_slope=True)      # the flash hypothesis
        free = fit_step_ramp(ys, t, best["k"], sig)                  # did the slope change there?
        rss_step = min(fit_step_ramp(ys, t, k, sig, single_slope=True)["rss"] for k in range(1, n))
        rss_flare = min(fit_broken_ramp(ys, t, k)["rss"] for k in range(1, n))
        per_pixel.append({"y": int(yy), "x": int(xx), "k": best["k"], "step_e": best["step"],
                          "step_err_e": best["step_err"], "step_snr": best["step_snr"],
                          "s_before_e_s": free["s_before"], "s_after_e_s": free["s_after"],
                          "slope_change_sigma": free["slope_change_sigma"],
                          "slope_after_constrained": free["slope_after_constrained"],
                          "sigma_e": sig, "rate_e_s": rate,
                          "delta_chi2_step_minus_flare": (rss_step - rss_flare) / sig ** 2})
    snr_min = float(fc["step_snr_min"])
    sig_px = [p for p in per_pixel if math.isfinite(p["step_snr"]) and p["step_snr"] >= snr_min]
    base["n_significant_px"] = len(sig_px)
    ks = [p["k"] for p in sig_px]
    modal = int(max(set(ks), key=ks.count)) if ks else None
    if modal is None:
        counts_all = [p["k"] for p in per_pixel if p["k"] is not None]
        modal = int(max(set(counts_all), key=counts_all.count)) if counts_all else None
        notes.append("no footprint pixel shows a significant step")
    n_fp = len(per_pixel)
    coinc = sum(1 for p in sig_px if p["k"] == modal) / n_fp if n_fp else 0.0
    dslope = [abs(p["slope_change_sigma"]) for p in sig_px if p["k"] == modal
              and math.isfinite(p["slope_change_sigma"])]
    dslope_max = max(dslope) if dslope else 0.0
    dchi2 = float(sum(p["delta_chi2_step_minus_flare"] for p in per_pixel))

    # Amplitude map at the modal resultant, every footprint pixel refitted there.
    amps = np.zeros(n_fp)
    if modal is not None:
        for i, p in enumerate(per_pixel):
            f = fit_step_ramp(cube[:, p["y"], p["x"]], t, modal, p["sigma_e"], single_slope=True)
            amps[i] = f["step"]
            p["step_at_modal_e"] = f["step"]
    shape = psf_amplitude_correlation(fp, amps, star_xy_in_box, fwhm_px)
    corr = shape["corr"]
    total = float(amps.sum()) if modal is not None else math.nan

    sc_max = float(fc["slope_change_sigma_max"])
    if (dslope_max > sc_max) or (dchi2 > sc_max ** 2):
        verdict = "SLOPE_CHANGE_FLARE_LIKE"
    elif coinc < float(fc["resultant_coincidence_min_frac"]):
        verdict = "NOT_COINCIDENT"
    elif not math.isfinite(corr) or corr < float(fc["psf_amplitude_corr_min"]):
        verdict = "NOT_PSF_SHAPED"
    else:
        verdict = "FLASH_CONSISTENT"
    if "note" in shape:
        notes.append(shape["note"])
    return {**base, "verdict": verdict, "coincidence_frac": float(coinc), "modal_index": modal,
            "slope_change_sigma_max_over_pixels": float(dslope_max),
            "delta_chi2_step_minus_flare": dchi2, "psf_amplitude_corr": corr,
            "psf_centroid_offset_px": [shape["dx"], shape["dy"]], "total_step_e": total,
            "gain_e_per_dn": gain, "sigma_empirical_e": sig_emp, "frames_source": frames_src,
            "per_pixel": per_pixel}


def next_exposure_residual(flux_next: float, flux_err_next: float, baseline_flux: float,
                           baseline_err: float) -> float:
    """z-score of the next exposure's flux against the star's baseline; NaN if the errors are unusable.

    A flare persists into the next exposure; a sub-resultant flash leaves
    nothing behind.  Positive means brighter than baseline.
    """
    try:
        var = float(flux_err_next) ** 2 + float(baseline_err) ** 2
        if not math.isfinite(var) or var <= 0:
            return math.nan
        z = (float(flux_next) - float(baseline_flux)) / math.sqrt(var)
        return z if math.isfinite(z) else math.nan
    except (TypeError, ValueError):
        return math.nan


def fluence_threshold(read_noise_e: float, n_sigma: float, core_fraction: float,
                      rate_e_s: float | None = None) -> dict:
    """The section 2.3 sensitivity numbers: what the jump detector needs, in the star's own photons.

    The peak pixel must jump by ``n_sigma`` read noises (5 x 12 e- = 60 e-);
    with ``core_fraction`` of a point source in that pixel (0.20 in F146) the
    pulse must deliver ``60 / 0.20 = 300 e-`` at the aperture, which is the
    star's own photon budget over ``t_equivalent_s = 300 / rate`` (0.64 s at
    470 e-/s, H_AB = 21 in F146).  Every non-detection is such a limit per
    star per exposure -- an honesty check the assess stage tabulates, never a
    deliverable.
    """
    peak = float(n_sigma) * float(read_noise_e)
    cf = float(core_fraction)
    total = peak / cf if cf > 0 else math.nan
    out = {"n_e_peak_trigger": peak, "n_e_total_trigger": total, "core_fraction": cf,
           "n_sigma": float(n_sigma), "read_noise_e": float(read_noise_e), "t_equivalent_s": None}
    if rate_e_s is not None and float(rate_e_s) > 0 and math.isfinite(total):
        out["t_equivalent_s"] = total / float(rate_e_s)
        out["rate_e_s"] = float(rate_e_s)
    return out


# --------------------------------------------------------------------------------------
# Assess: recurrence and tiers
# --------------------------------------------------------------------------------------

_RAMP_REJECTS = {"SLOPE_CHANGE_FLARE_LIKE", "NOT_COINCIDENT", "NOT_PSF_SHAPED"}


def assess_candidates(records: list[dict], conf: dict, hot_ledger: HotPixelLedger | None = None) -> dict:
    """Recurrence and tiers over every ``psf_on_star`` record of a run.

    Two recurrences, treated oppositely.  The same *detector pixel* in
    ``hot_pixel_min_exposures`` or more distinct exposures (across the records,
    plus the ledger if given) is a defect: every record on it is vetoed
    ``hot_pixel``, retroactively.  The same *star* in two or more exposures is
    reported as ``recurrent_star`` and never vetoed -- a beacon that repeats is
    the point, and vetoing it would be vetoing the claim.

    A record reaches ``tiers.candidate`` only with a ramp verdict of
    ``FLASH_CONSISTENT`` (``record["ramp"]``, as :func:`ramp_flash_test`
    returns it) and a next-exposure residual (``record["next_exposure_residual_sigma"]``)
    at or below the threshold; a ramp verdict that rejects is a veto; anything
    still missing leaves it at ``tiers.interest`` with ``pending`` naming what
    is missing.  ``INSUFFICIENT_RESULTANTS`` cannot reject: it stays pending.
    """
    fc = flash_conf(conf)
    tiers = fc["tiers"]
    hot_min = int(fc["hot_pixel_min_exposures"])
    z_max = float(fc["next_exposure_residual_sigma_max"])
    funnel = Funnel()
    funnel.bump("records", len(records))

    pix_images: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for r in records:
        for y, x in r.get("pixels") or []:
            pix_images[(str(r.get("detector", "")), int(y), int(x))].add(str(r.get("image_id")))

    out_records: list[dict] = []
    hot_seen: set[tuple[str, int, int]] = set()
    for r in records:
        rec = dict(r)
        det = str(r.get("detector", ""))
        rec["veto"] = None
        rec["pending"] = []
        hot = []
        for y, x in r.get("pixels") or []:
            key = (det, int(y), int(x))
            c = len(pix_images[key])
            if hot_ledger is not None:
                c = max(c, hot_ledger.count(det, int(y), int(x)))
            if c >= hot_min:
                hot.append([int(y), int(x), int(c)])
                hot_seen.add(key)
        rec["hot_pixels"] = hot
        if hot:
            rec["veto"] = "hot_pixel"
        ramp = r.get("ramp") if isinstance(r.get("ramp"), dict) else None
        rv = ramp.get("verdict") if ramp else None
        rec["ramp_verdict"] = rv
        if rec["veto"] is None and rv in _RAMP_REJECTS:
            rec["veto"] = f"ramp_{rv.lower()}"
        z = r.get("next_exposure_residual_sigma")
        z = float(z) if isinstance(z, (int, float, np.floating)) and math.isfinite(float(z)) else None
        rec["next_exposure_residual_sigma"] = z
        if rec["veto"] is None and z is not None and z > z_max:
            rec["veto"] = "next_exposure_brightening"
        if rec["veto"] is not None:
            rec["tier"] = "REJECTED"
            funnel.reject(rec["veto"])
        elif rv == "FLASH_CONSISTENT" and z is not None:
            rec["tier"] = tiers["candidate"]
        else:
            if rv != "FLASH_CONSISTENT":
                rec["pending"].append("ramp" if rv is None else f"ramp_{rv.lower()}")
            if z is None:
                rec["pending"].append("next_exposure")
            rec["tier"] = tiers["interest"]
        out_records.append(rec)

    star_images: dict[str, set[str]] = defaultdict(set)
    for rec in out_records:
        if rec["veto"] is None:
            star_images[str(rec.get("star_id"))].add(str(rec.get("image_id")))
    recurrent = {s: sorted(imgs) for s, imgs in star_images.items() if len(imgs) >= 2}
    for rec in out_records:
        sid = str(rec.get("star_id"))
        rec["n_star_events"] = len(star_images.get(sid, ()))
        rec["recurrent_star"] = sid in recurrent

    n_cand = sum(1 for r in out_records if r["tier"] == tiers["candidate"])
    n_int = sum(1 for r in out_records if r["tier"] == tiers["interest"])
    funnel.bump("hot_pixels_vetoed", len(hot_seen))
    funnel.bump("interest", n_int)
    funnel.bump("candidates", n_cand)
    if n_cand:
        verdict = tiers["candidate"]
    elif n_int:
        verdict = tiers["interest"]
    else:
        verdict = "NO_SURVIVORS"
    return {"verdict": verdict, "n_records": len(records), "n_vetoed": len(records) - n_cand - n_int,
            "n_interest": n_int, "n_candidates": n_cand, "records": out_records,
            "recurrent_stars": recurrent, "hot_pixels": sorted(list(k) for k in hot_seen),
            "funnel": funnel.as_dict()}


# --------------------------------------------------------------------------------------
# Synthesis (tests and selftest only)
# --------------------------------------------------------------------------------------

def _pixel_integrated_psf(shape: tuple[int, int], star_xy: tuple[float, float], fwhm_px: float) -> np.ndarray:
    """Fraction of a point source's flux in each pixel: the Gaussian *integrated* over pixels."""
    ny, nx = int(shape[0]), int(shape[1])
    sigma = max(float(fwhm_px), 1e-3) * _FWHM_TO_SIGMA
    s = sigma * math.sqrt(2.0)
    ex = np.arange(nx + 1) - 0.5 - float(star_xy[0])
    ey = np.arange(ny + 1) - 0.5 - float(star_xy[1])
    fx = 0.5 * np.diff(erf(ex / s))
    fy = 0.5 * np.diff(erf(ey / s))
    return np.outer(fy, fx)


def _line_pixels(start: tuple[float, float], angle: float, length: int) -> np.ndarray:
    s = np.linspace(0.0, max(length - 1, 0), 2 * max(length, 1))
    yy = np.round(start[0] + s * math.sin(angle)).astype(int)
    xx = np.round(start[1] + s * math.cos(angle)).astype(int)
    return np.unique(np.stack([yy, xx], axis=1), axis=0)


def synthesise_dq_image(shape: tuple[int, int], stars: list[dict], fwhm_px: float,
                        flash_stars, n_cosmic_rays: int, n_snowballs: int, rng,
                        flags: dict[str, int], saturated_stars=()) -> np.ndarray:
    """A ``dq`` image with PSF-shaped jump footprints on ``flash_stars``, cosmic rays and snowballs.

    ``flash_stars`` / ``saturated_stars`` are star ids (or indices) from
    ``stars``.  Flash footprints are the pixels above ``footprint_frac`` (0.10)
    of the peak, as the detector defines them; saturated stars get SATURATED
    on their central pixel.  Cosmic rays are 1-3-pixel hits, one in ten of them
    a straight 4-8-pixel track; snowballs are discs of radius 4-6 px with a
    SATURATED core.  Cosmic rays and snowballs are kept clear of every star so
    the test can assert exact label counts -- a merger with a footprint is a
    real possibility on sky and is what the coverage and elongation rules
    absorb, but it is not what this image tests.  Returns ``uint32``.
    """
    ny, nx = int(shape[0]), int(shape[1])
    dq = np.zeros((ny, nx), dtype=np.uint32)
    jump = int(flags.get("JUMP_DET", DQ_FLAGS_FALLBACK["JUMP_DET"]))
    satb = int(flags.get("SATURATED", DQ_FLAGS_FALLBACK["SATURATED"]))
    rng = rng if rng is not None else np.random.default_rng(0)
    by_id = {str(s.get("star_id", i)): s for i, s in enumerate(stars)}
    flash_ids = {str(s) for s in flash_stars}
    sat_ids = {str(s) for s in saturated_stars}
    for sid in flash_ids:
        s = by_id[sid]
        fp = footprint_pixels((float(s["x"]), float(s["y"])), fwhm_px, 0.10, shape=(ny, nx))
        dq[fp[:, 0], fp[:, 1]] |= jump
    for sid in sat_ids:
        s = by_id[sid]
        cy, cx = int(round(float(s["y"]))), int(round(float(s["x"])))
        if 0 <= cy < ny and 0 <= cx < nx:
            dq[cy, cx] |= satb
    sx = np.array([float(s["x"]) for s in stars]) if stars else np.zeros(0)
    sy = np.array([float(s["y"]) for s in stars]) if stars else np.zeros(0)
    clear = 3.0 * float(fwhm_px) + 4.0

    def far_from_stars(y: float, x: float, margin: float) -> bool:
        return sx.size == 0 or bool(np.all(np.hypot(sx - x, sy - y) > margin))

    def place(margin: float):
        for _ in range(200):
            y = float(rng.uniform(margin, ny - 1 - margin))
            x = float(rng.uniform(margin, nx - 1 - margin))
            if far_from_stars(y, x, clear + margin):
                return y, x
        return None

    for _ in range(int(n_snowballs)):
        r = float(rng.uniform(4.0, 6.0))
        pos = place(r + 2)
        if pos is None:
            break
        yy, xx = np.mgrid[0:ny, 0:nx]
        d = np.hypot(yy - pos[0], xx - pos[1])
        dq[d <= r] |= jump
        dq[d <= 1.5] |= satb
    for i in range(int(n_cosmic_rays)):
        track = (i % 10 == 9)
        length = int(rng.integers(4, 9)) if track else int(rng.integers(1, 4))
        pos = place(length + 2)
        if pos is None:
            break
        if track or length > 1:
            pix = _line_pixels(pos, float(rng.uniform(0, math.pi)), length)
        else:
            pix = np.array([[int(round(pos[0])), int(round(pos[1]))]])
        ok = (pix[:, 0] >= 0) & (pix[:, 0] < ny) & (pix[:, 1] >= 0) & (pix[:, 1] < nx)
        dq[pix[ok, 0], pix[ok, 1]] |= jump
    return dq


def synthesise_ramp(box: tuple[int, int], times_s, star_xy: tuple[float, float], fwhm_px: float,
                    rate_e_s: float, read_noise_e: float, flash_e: float = 0.0,
                    flash_at_index: int | None = None, flare_rate_after: float = 0.0,
                    rng=None, frames_per_resultant: int = 1, sky_e_s: float = 0.0,
                    flash_pixels=None) -> Ramp:
    """A Level 1 ramp in electrons for one star: accumulation, a flash step or a flare.

    Per pixel the star's rate (``rate_e_s`` times the pixel-integrated PSF)
    plus ``sky_e_s`` accumulates as Poisson increments between resultants;
    read noise ``read_noise_e / sqrt(frames_per_resultant)`` is added per
    resultant.  ``flash_e`` electrons, PSF-distributed (or confined to
    ``flash_pixels``, a list of (y, x), for a cosmic-ray stand-in), land at
    resultant ``flash_at_index`` and stay.  ``flare_rate_after`` adds that many
    e-/s (PSF-distributed) from ``flash_at_index`` on -- a slope change, no
    step.  ``unit`` is ``e-``; the read pattern records ``frames_per_resultant``.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    ny, nx = int(box[0]), int(box[1])
    t = np.asarray(times_s, dtype=float)
    n = t.size
    psf = _pixel_integrated_psf((ny, nx), star_xy, fwhm_px)
    rate = float(rate_e_s) * psf + float(sky_e_s)
    cube = np.zeros((n, ny, nx))
    acc = np.zeros((ny, nx))
    prev_t = 0.0
    for i in range(n):
        r = rate.copy()
        if flash_at_index is not None and float(flare_rate_after) > 0 and i >= int(flash_at_index):
            # slope change from the flash index: the extra rate applies over the interval ending at i
            r = r + float(flare_rate_after) * psf
        dt = max(t[i] - prev_t, 0.0)
        acc += rng.poisson(np.maximum(r * dt, 0.0))
        if flash_at_index is not None and float(flash_e) > 0 and i == int(flash_at_index):
            if flash_pixels is not None:
                for y, x in flash_pixels:
                    acc[int(y), int(x)] += float(flash_e)
            else:
                acc += rng.poisson(np.maximum(float(flash_e) * psf, 0.0))
        cube[i] = acc + rng.normal(0.0, float(read_noise_e) / math.sqrt(max(int(frames_per_resultant), 1)),
                                   size=(ny, nx))
        prev_t = t[i]
    m = max(int(frames_per_resultant), 1)
    pattern = [list(range(i * m + 1, (i + 1) * m + 1)) for i in range(n)]
    return Ramp(image_id="synthetic_ramp", resultants=cube, times_s=t, unit="e-", gain_e_per_dn=1.0,
                read_pattern=pattern, meta={"synthetic": True, "flash_e": float(flash_e),
                                            "flash_at_index": flash_at_index,
                                            "flare_rate_after": float(flare_rate_after)})


def selftest(conf: dict | None = None, seed: int = 0) -> dict:
    """Build a synthetic dq image and three ramps, run both stages, report what was recovered.

    Every number returned is from synthetic input and says so.  ``passed`` is
    True when the flash stars are the only ``psf_on_star`` clusters, the
    saturated one is ``near_saturated``, the snowballs are labelled, the flash
    ramp is ``FLASH_CONSISTENT``, the flare ramp ``SLOPE_CHANGE_FLARE_LIKE``
    and the sensitivity numbers match section 2.3.
    """
    fc = flash_conf(conf)
    rng = np.random.default_rng(seed)
    flags = dict(DQ_FLAGS_FALLBACK)
    fwhm = 1.2
    stars = [{"star_id": f"S{i}", "x": 20.0 + 45.0 * (i % 5) + rng.uniform(-0.4, 0.4),
              "y": 20.0 + 45.0 * (i // 5) + rng.uniform(-0.4, 0.4)} for i in range(10)]
    dq = synthesise_dq_image((120, 240), stars, fwhm, ["S1", "S4", "S7", "S2"], 200, 2, rng, flags,
                             saturated_stars=["S2"])
    cut = DQCutout(image_id="synthetic_dq", dq=dq, stars=stars, detector="WFI01", band="F146",
                   mjd=60000.0, psf_fwhm_px=fwhm)
    screen = screen_dq_cutout(cut, fc, flags)
    found = sorted(c["star_id"] for c in screen["candidates"])
    times = np.linspace(3.0, 45.0, 6)
    # A 500 e- pulse on a faint star in a deep read pattern (the per-pixel step error on a
    # 6-resultant ramp is ~1.7 sigma, so the wing pixels need quiet reads to reach 5 sigma).
    fwhm_ramp = 1.5
    flash = synthesise_ramp((9, 9), times, (4.0, 4.0), fwhm_ramp, 2.0, 12.0, flash_e=500.0,
                            flash_at_index=3, rng=rng, frames_per_resultant=16)
    flare = synthesise_ramp((9, 9), times, (4.0, 4.0), fwhm_ramp, 100.0, 12.0, flash_at_index=3,
                            flare_rate_after=200.0, rng=rng, frames_per_resultant=2)
    r_flash = ramp_flash_test(flash, (4.0, 4.0), fwhm_ramp, fc, 12.0)
    r_flare = ramp_flash_test(flare, (4.0, 4.0), fwhm_ramp, fc, 12.0)
    fl = fluence_threshold(12.0, float(fc["n_sigma_trigger"]), 0.20, rate_e_s=470.0)
    passed = (found == ["S1", "S4", "S7"] and screen["labels"].get("near_saturated", 0) == 1
              and screen["labels"].get("snowball", 0) == 2
              and r_flash["verdict"] == "FLASH_CONSISTENT"
              and r_flare["verdict"] == "SLOPE_CHANGE_FLARE_LIKE"
              and abs(fl["n_e_total_trigger"] - 300.0) < 1e-9)
    return {"passed": bool(passed), "synthetic": True, "screen_labels": screen["labels"],
            "psf_on_star_ids": found, "flash_ramp": {k: r_flash[k] for k in
                                                     ("verdict", "coincidence_frac", "psf_amplitude_corr",
                                                      "total_step_e", "slope_change_sigma_max_over_pixels")},
            "flare_ramp": {k: r_flare[k] for k in ("verdict", "coincidence_frac",
                                                   "slope_change_sigma_max_over_pixels",
                                                   "delta_chi2_step_minus_flare")},
            "fluence": fl}


__all__ = [
    "HotPixelLedger", "LABELS", "RAMP_VERDICTS", "assess_candidates", "best_step_index",
    "classify_cluster", "find_jump_clusters", "fit_broken_ramp", "fit_step_ramp", "flash_conf",
    "fluence_threshold", "footprint_coverage", "footprint_pixels", "footprint_radius_px",
    "gaussian_psf", "match_cluster_to_stars", "next_exposure_residual", "psf_amplitude_correlation",
    "psf_footprint", "ramp_flash_test", "screen_dq_cutout", "selftest", "synthesise_dq_image",
    "synthesise_ramp",
]
