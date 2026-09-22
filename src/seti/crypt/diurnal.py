"""The thermal screen CRYPT can actually run: diurnal/seasonal invariance of
bolometric temperature inside permanent shadow.

Why this and not the multi-channel statistic
--------------------------------------------
``thermal.py`` screens for anisothermality across Diviner's channels 6/7
against 8/9.  The PDS gridded *polar* holdings do not contain per-channel
maps: every one of the 912 Polar Cumulative Products is
``PCP_AVG_TBOL_POL{N,S}_{SUM,WIN}_LTIM{01..24}_240`` — bolometric temperature,
average, on a 240 m polar stereographic grid, in twenty-four LOCAL-TIME bins,
separately for lunar summer and winter (probe of 2026-09-22).  The
multi-channel axis is therefore unavailable at map-product level, and the
axis that *is* available has never been screened for this.

The statistic
-------------
Inside a permanently shadowed region there is no direct sun at any local time
or season.  A PSR floor pixel is nevertheless heated, and every passive
heating term varies:

* scattered visible light off a sunlit rim rises and falls with the SUN's
  position, i.e. with local time, and is far weaker in winter;
* thermal infrared from the surrounding warm terrain follows the same clock
  with a lag;
* so a passively heated pixel has a diurnal amplitude and a summer−winter
  offset, both of order several kelvin at the warmer PSR margins.

An object generating its own heat does not.  Its contribution is the same at
every local-time bin and in both seasons.  So the signature is a pixel whose

* ``T_floor`` = min over all 48 (season, local-time) bins is high relative to
  its own neighbourhood (the excess), and whose
* winter diurnal amplitude ``A_win`` = max_h − min_h over winter bins, and
* seasonal offset ``D`` = mean(summer) − mean(winter)

are both consistent with zero while its neighbours' are not.  The background
is taken in a local annulus so the large-scale gradient of PSR floor
temperature (depth, latitude, surrounding topography) divides out.

The rules (each a named rejection counter)
------------------------------------------
* ``diurnal``: ``A_win`` above ``amp_max_K`` — the excess follows the sun
  around the sky, so it is secondary illumination or rim IR.
* ``seasonal``: ``|D|`` above ``dseason_max_K`` — a lit rim or scattered
  light, strong in summer and weak in winter.
* ``few_bins``: the excess appears in fewer than ``n_bins_excess`` of the
  bins — one bad bin, a map seam or a single-product artefact.
* ``low_coverage``: fewer than ``n_bins_min`` bins carry data at the pixel.
* ``extended``: the connected patch of elevated pixels exceeds
  ``max_cluster_px`` — a rock field, ejecta blanket or warm slope.
* ``stripe``: ``stripe_min`` flagged pixels share a row or column — striping
  or a seam between contributing maps.
* ``edge``: within ``edge_px`` of the PSR boundary.
* ``not_in_floor``: warm in the all-bin mean but not in the floor statistic —
  the heating switches off at some point in the year, so it is not internal.
* ``human_hardware``: within a listed impact or landing site.

The floor
---------
Bolometric temperature is exactly Stefan–Boltzmann-additive, so a source of
area ``A`` at ``T_hot`` on a pixel of area ``A_px`` at ``T_cold`` gives

    T_obs⁴ = T_cold⁴ + (A/A_px)(T_hot⁴ − T_cold⁴)

which inverts in closed form: no band model and no fitting is needed, and
``floor_area_bolometric`` states the detectable area directly.
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

from .labels import Georef


@contextlib.contextmanager
def _quiet_nan():
    """An all-NaN column is a pixel with no data, not an error."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN.*", RuntimeWarning)
        warnings.filterwarnings("ignore", r"Mean of empty slice.*", RuntimeWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            yield

DEFAULT_THRESHOLDS: dict = {
    "psr_tmax_K": 110.0,      # cold trap: never warmer than this in any bin
    "edge_px": 2,
    "n_bins_min": 24,         # (season, local-time) bins that must carry data
    "n_bins_excess": 12,      # bins in which the excess must be present
    "z_min": 5.0,             # T_floor excess over the local annulus, in sigma
    "z_bin_min": 3.0,         # per-bin excess counted towards n_bins_excess
    "amp_max_K": 2.0,         # winter diurnal amplitude allowed of a candidate
    "dseason_max_K": 2.0,     # |summer - winter| allowed of a candidate
    "elevated_frac": 0.5,     # z >= elevated_frac * z_min joins a cluster
    "max_cluster_px": 6,
    "stripe_min": 4,
    "stripe_window": 200,
    "r_in_px": 4,             # background annulus, in pixels
    "r_out_px": 16,
    "bg_min_px": 40,          # annulus pixels needed for a background
    "sigma_floor_K": 0.05,
    "floor_t_hot_K": 300.0,
    "floor_nsigma": 5.0,
}

CLASSES = ["candidate", "human_hardware", "low_coverage", "extended", "stripe", "edge",
           "few_bins", "seasonal", "diurnal", "not_in_floor"]


# ---------------------------------------------------------------------------
# the cube
# ---------------------------------------------------------------------------
@dataclass
class DiurnalCube:
    """Average bolometric temperature on one polar grid, by (season, bin)."""

    pole: str
    georef: Georef
    arrays: dict = field(default_factory=dict)      # (season, bin) -> 2-D float32
    sources: dict = field(default_factory=dict)

    def put(self, season: str, bin_id: str, arr, source=None) -> None:
        arr = np.asarray(arr, dtype=np.float32)
        if self.arrays:
            shape = next(iter(self.arrays.values())).shape
            if arr.shape != shape:
                raise ValueError(f"layer ({season},{bin_id}) shape {arr.shape} != grid {shape}")
        self.arrays[(season, str(bin_id))] = arr
        if source is not None:
            self.sources[(season, str(bin_id))] = source

    def get(self, season: str, bin_id: str):
        return self.arrays.get((season, str(bin_id)))

    @property
    def seasons(self) -> list[str]:
        return sorted({s for (s, _b) in self.arrays})

    def bins(self, season: str) -> list[str]:
        return sorted(b for (s, b) in self.arrays if s == season)

    @property
    def shape(self):
        return next(iter(self.arrays.values())).shape if self.arrays else (0, 0)

    @property
    def n_layers(self) -> int:
        return len(self.arrays)

    def stack(self, season: str | None = None) -> np.ndarray:
        """(n_bins, lines, samples); NaN where a bin has no data."""
        keys = sorted(k for k in self.arrays if season is None or k[0] == season)
        if not keys:
            return np.zeros((0, *self.shape), dtype=np.float32)
        return np.stack([self.arrays[k] for k in keys], axis=0)

    def copy(self) -> DiurnalCube:
        return DiurnalCube(self.pole, self.georef, {k: v.copy() for k, v in self.arrays.items()},
                           dict(self.sources))


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------
def mix_bolometric(T_cold, f: float, T_hot: float):
    """Bolometric temperature of a pixel that is a fraction ``f`` covered by
    ``T_hot``.  Stefan-Boltzmann is linear in radiance, so this is exact."""
    Tc = np.asarray(T_cold, dtype=float)
    return (Tc**4 + float(f) * (float(T_hot) ** 4 - Tc**4)) ** 0.25


def floor_area_bolometric(T_cold: float, dT: float, T_hot: float, pixel_area_m2: float) -> float:
    """Source area (m²) at ``T_hot`` that raises a ``T_cold`` pixel by ``dT``."""
    if not (np.isfinite(dT) and dT > 0 and np.isfinite(T_cold) and T_cold > 0):
        return float("nan")
    if T_hot <= T_cold + dT:
        return float("nan")
    f = ((T_cold + dT) ** 4 - T_cold**4) / (T_hot**4 - T_cold**4)
    return float(f * pixel_area_m2)


# ---------------------------------------------------------------------------
# mask
# ---------------------------------------------------------------------------
def cold_trap_mask(cube: DiurnalCube, thr: dict, external: np.ndarray | None = None) -> dict:
    """Pixels that are never warmer than ``psr_tmax_K`` in any bin, ANDed with
    an external PSR raster when one was served, then eroded by ``edge_px``."""
    rep: dict = {"degraded": [], "external_psr": external is not None}
    if not cube.arrays:
        return {**rep, "mask": None, "interior": None, "n_mask": 0, "n_interior": 0,
                "degraded": [*rep["degraded"], "no_layers"]}
    cube_all = cube.stack()
    with _quiet_nan():
        tmax = np.nanmax(cube_all, axis=0)
    n_valid = np.sum(np.isfinite(cube_all), axis=0)
    mask = np.isfinite(tmax) & (tmax <= float(thr["psr_tmax_K"]))
    rep["n_cold_trap"] = int(mask.sum())
    if external is not None:
        ext = np.asarray(external)
        if ext.shape != mask.shape:
            rep["degraded"].append(f"external_psr_shape_{ext.shape}_vs_{mask.shape}")
        else:
            mask &= ext.astype(bool)
            rep["n_after_external"] = int(mask.sum())
    else:
        rep["degraded"].append("no_external_psr_raster")
    mask &= n_valid >= int(thr["n_bins_min"])
    rep["n_after_coverage"] = int(mask.sum())
    e = int(thr["edge_px"])
    interior = ndimage.binary_erosion(mask, np.ones((2 * e + 1, 2 * e + 1), dtype=bool)) if e > 0 \
        else mask.copy()
    rep.update({"mask": mask, "interior": interior, "n_mask": int(mask.sum()),
                "n_interior": int(interior.sum()),
                "tmax_K": float(np.nanmin(tmax[mask])) if mask.any() else float("nan")})
    return rep


# ---------------------------------------------------------------------------
# the statistic
# ---------------------------------------------------------------------------
def _box_sum(x: np.ndarray, r: int) -> np.ndarray:
    """Sum over the (2r+1)² square, separable and O(N) in the map size."""
    if r < 0:
        return np.zeros_like(x)
    k = 2 * r + 1
    return ndimage.uniform_filter(x, size=k, mode="constant", cval=0.0) * float(k * k)


def _annulus_mean(x: np.ndarray, valid: np.ndarray, r_in: int, r_out: int):
    """Mean of ``x`` over a SQUARE annulus: the (2·r_out+1)² box minus the
    (2·r_in−1)² box.  A square annulus rather than a circular one because a
    box filter is separable — the polar maps are ~9000 px on a side and 48
    layers deep, where a 33×33 direct convolution per layer is not tractable.
    The shape of the annulus does not enter the statistic, only its symmetry
    about the pixel, which a square has.
    """
    v = valid.astype(np.float64)
    xx = np.where(valid, np.nan_to_num(x, nan=0.0), 0.0).astype(np.float64)
    r_hole = int(r_in) - 1
    n = _box_sum(v, int(r_out)) - _box_sum(v, r_hole)
    s = _box_sum(xx, int(r_out)) - _box_sum(xx, r_hole)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / np.where(n > 0, n, 1.0), np.nan)
    return mean, n

def local_background(field_: np.ndarray, valid: np.ndarray, thr: dict) -> dict:
    """Local background level and scatter of ``field_``, in two passes.

    The background is the mean over an ANNULUS (not a disc), so a compact
    source never enters its own background, and a locally linear gradient
    averages to the centre value exactly.

    The scatter needs a second pass.  PSR floor temperature has a strong
    large-scale gradient — tens of kelvin across a crater — so the spread of
    the raw field inside a 16-pixel annulus is dominated by that gradient and
    not by the pixel-to-pixel scatter a detection has to beat.  Pass 1 forms
    the residual ``field - local mean``, which cancels the linear term; pass 2
    takes the mean absolute residual over the same annulus and scales it by
    1.4826.  That is the scatter a real point source is measured against.
    """
    r_in, r_out = int(thr["r_in_px"]), int(thr["r_out_px"])
    mean, n = _annulus_mean(field_, valid, r_in, r_out)
    res_ok = valid & np.isfinite(mean) & np.isfinite(field_)
    res = np.where(res_ok, field_ - mean, np.nan)
    # a smoothly CURVED field leaves a near-constant residual (the annulus mean
    # of a paraboloid is not its centre value), which is a bias, not scatter:
    # subtract the local mean of the residual before measuring the spread, or
    # every PSR floor bowl inflates its own sigma and hides real sources.
    res_mean, _ = _annulus_mean(res, res_ok, r_in, r_out)
    res_c = np.where(res_ok & np.isfinite(res_mean), res - res_mean, np.nan)
    mad, _ = _annulus_mean(np.abs(res_c), res_ok & np.isfinite(res_c), r_in, r_out)
    sigma = np.maximum(1.4826 * mad, float(thr["sigma_floor_K"]))
    enough = n >= float(thr["bg_min_px"])
    # ``excess`` is the detection field: the pixel above its own annulus with
    # the local level AND the local curvature bias taken out.
    return {"mean": np.where(enough, mean, np.nan), "sigma": np.where(enough, sigma, np.nan),
            "n": n, "enough": enough, "residual": res,
            "excess": np.where(enough, res_c, np.nan)}


def statistics(cube: DiurnalCube, interior: np.ndarray, thr: dict) -> dict:
    """Per-pixel floor temperature, diurnal amplitude, seasonal offset, and
    the excess of each over the local annulus background."""
    out: dict = {"seasons": cube.seasons, "n_layers": cube.n_layers}
    allst = cube.stack()
    with _quiet_nan():
        t_floor = np.nanmin(allst, axis=0)
        t_mean = np.nanmean(allst, axis=0)
    n_valid = np.sum(np.isfinite(allst), axis=0)
    per_season = {}
    for s in cube.seasons:
        st = cube.stack(s)
        with _quiet_nan():
            per_season[s] = {"min": np.nanmin(st, axis=0), "max": np.nanmax(st, axis=0),
                             "mean": np.nanmean(st, axis=0),
                             "n": np.sum(np.isfinite(st), axis=0)}
    # diurnal amplitude: winter if present (passive heating is weakest then),
    # otherwise the season that exists
    s_amp = "winter" if "winter" in per_season else (cube.seasons[0] if cube.seasons else None)
    amp = (per_season[s_amp]["max"] - per_season[s_amp]["min"]) if s_amp else np.full(cube.shape, np.nan)
    if "summer" in per_season and "winter" in per_season:
        dseason = per_season["summer"]["mean"] - per_season["winter"]["mean"]
        out["seasonal_pair"] = True
    else:
        dseason = np.full(cube.shape, np.nan)
        out["seasonal_pair"] = False
    bg = local_background(t_floor, interior & np.isfinite(t_floor), thr)
    bgm = local_background(t_mean, interior & np.isfinite(t_mean), thr)
    with _quiet_nan():
        z_floor = bg["excess"] / bg["sigma"]
        z_mean = bgm["excess"] / bgm["sigma"]
    # per-bin excess: how many of the 48 bins put the pixel above its annulus
    n_bins_hot = np.zeros(cube.shape, dtype=np.int32)
    for key in sorted(cube.arrays):
        layer = cube.arrays[key]
        b = local_background(layer, interior & np.isfinite(layer), thr)
        with _quiet_nan():
            zb = b["excess"] / b["sigma"]
        n_bins_hot += (np.isfinite(zb) & (zb >= float(thr["z_bin_min"]))).astype(np.int32)
    out.update({"t_floor": t_floor, "t_mean": t_mean, "n_valid": n_valid, "amp": amp,
                "amp_season": s_amp, "dseason": dseason, "z_floor": z_floor, "z_mean": z_mean,
                "mean_bg_mean": bgm["mean"], "mean_bg_sigma": bgm["sigma"],
                "bg_mean": bg["mean"], "bg_sigma": bg["sigma"], "bg_n": bg["n"],
                "n_bins_hot": n_bins_hot, "per_season": per_season})
    return out


def _hardware_hits(lon: float, lat: float, sites: list, radius_m: float = 1737400.0) -> list[str]:
    hits = []
    for s in sites or []:
        dlat = np.radians(float(lat) - float(s["lat"]))
        mlat = np.radians(0.5 * (float(lat) + float(s["lat"])))
        dlon = np.radians(((float(lon) - float(s["lon"]) + 180.0) % 360.0) - 180.0) * np.cos(mlat)
        d = radius_m * float(np.hypot(dlat, dlon))
        if d <= float(s.get("radius_m", 3000.0)):
            hits.append(f"{s['name']} ({d / 1000.0:.2f} km)")
    return hits


def _classify(reasons: str) -> str:
    if not reasons:
        return "candidate"
    order = ["human_hardware", "low_coverage", "extended", "stripe", "edge", "few_bins",
             "seasonal", "diurnal", "not_in_floor"]
    toks = reasons.split(";")
    for o in order:
        if any(t.startswith(o) for t in toks):
            return o
    return "candidate"


def screen_pole(cube: DiurnalCube, thr: dict | None = None, *, hardware: list | None = None,
                external_psr: np.ndarray | None = None) -> dict:
    """The whole diurnal/seasonal screen on one pole."""
    thr = {**DEFAULT_THRESHOLDS, **(thr or {})}
    rep: dict = {"pole": cube.pole, "seasons": cube.seasons, "n_layers": cube.n_layers,
                 "degraded": [], "counts": {c: 0 for c in CLASSES}}
    m = cold_trap_mask(cube, thr, external=external_psr)
    rep["mask"] = {k: v for k, v in m.items() if k not in ("mask", "interior")}
    rep["degraded"] += m["degraded"]
    if m["mask"] is None or m["n_interior"] == 0:
        rep["status"] = "NO_PSR_INTERIOR"
        rep["flags"] = pd.DataFrame()
        return rep
    interior = m["interior"]
    st = statistics(cube, interior, thr)
    if not st["seasonal_pair"]:
        rep["degraded"].append("single_season_only")
    with _quiet_nan():
        rep["stats"] = {"amp_season": st["amp_season"], "n_layers": st["n_layers"],
                    "median_bg_sigma_K": float(np.nanmedian(st["bg_sigma"][interior])),
                    "median_t_floor_K": float(np.nanmedian(st["t_floor"][interior])),
                    "median_amp_K": float(np.nanmedian(st["amp"][interior])),
                    "median_dseason_K": float(np.nanmedian(st["dseason"][interior]))
                        if st["seasonal_pair"] else None}
    z = st["z_floor"]
    zm = st["z_mean"]
    z_min = float(thr["z_min"])
    # Flag on EITHER statistic.  The floor (minimum over all 48 bins) is immune
    # to purely seasonal or purely diurnal passive heating by construction, so
    # on its own it would make those confounders vanish silently instead of
    # being counted; flagging the all-bin mean as well puts every warm pixel
    # into the table, where the named rules reject it and the funnel records it.
    hot_floor = interior & np.isfinite(z) & (z >= z_min)
    hot_mean = interior & np.isfinite(zm) & (zm >= z_min)
    flagged = hot_floor | hot_mean
    rep["n_flagged"] = int(flagged.sum())
    rep["n_flagged_floor"] = int(hot_floor.sum())
    rep["n_flagged_mean"] = int(hot_mean.sum())
    elevated = interior & ((np.isfinite(z) & (z >= float(thr["elevated_frac"]) * z_min))
                           | (np.isfinite(zm) & (zm >= float(thr["elevated_frac"]) * z_min)))
    lab, _n = ndimage.label(elevated, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(lab.ravel())
    ii, jj = np.nonzero(flagged)
    lon, lat = cube.georef.pix_to_lonlat(ii, jj)
    # stripes: rows / columns carrying many flagged pixels
    row_n = np.bincount(ii, minlength=cube.shape[0])
    col_n = np.bincount(jj, minlength=cube.shape[1])
    edge_mask = m["mask"] & ~interior
    rows = []
    for n_, (i, j) in enumerate(zip(ii.tolist(), jj.tolist(), strict=True)):
        reasons = []
        cluster = int(sizes[lab[i, j]]) if lab[i, j] else 1
        amp = float(st["amp"][i, j])
        dse = float(st["dseason"][i, j])
        nbh = int(st["n_bins_hot"][i, j])
        nv = int(st["n_valid"][i, j])
        hw = _hardware_hits(float(lon[n_]), float(lat[n_]), hardware or [])
        if hw:
            reasons.append("human_hardware:" + "|".join(hw))
        if nv < int(thr["n_bins_min"]):
            reasons.append(f"low_coverage:{nv}")
        if cluster > int(thr["max_cluster_px"]):
            reasons.append(f"extended:{cluster}")
        if row_n[i] >= int(thr["stripe_min"]) or col_n[j] >= int(thr["stripe_min"]):
            reasons.append(f"stripe:row{int(row_n[i])}col{int(col_n[j])}")
        if edge_mask[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].any():
            reasons.append("edge:boundary")
        if nbh < int(thr["n_bins_excess"]):
            reasons.append(f"few_bins:{nbh}")
        if not (np.isfinite(z[i, j]) and z[i, j] >= z_min):
            reasons.append(f"not_in_floor:{float(z[i, j]):.2f}")
        if np.isfinite(dse) and abs(dse) > float(thr["dseason_max_K"]):
            reasons.append(f"seasonal:{dse:.2f}K")
        if np.isfinite(amp) and amp > float(thr["amp_max_K"]):
            reasons.append(f"diurnal:{amp:.2f}K")
        rs = ";".join(reasons)
        cls = _classify(rs)
        t_floor = float(st["t_floor"][i, j])
        bgm = float(st["bg_mean"][i, j])
        sig = float(st["bg_sigma"][i, j])
        rows.append({
            "pole": cube.pole, "line": int(i), "sample": int(j),
            "lon": float(lon[n_]), "lat": float(lat[n_]),
            "z_floor": float(z[i, j]), "z_mean": float(zm[i, j]), "T_floor_K": t_floor, "T_bg_K": bgm, "sigma_bg_K": sig,
            "dT_K": t_floor - bgm, "amp_K": amp, "dseason_K": dse,
            "n_bins_hot": nbh, "n_bins_valid": nv, "cluster_px": cluster,
            "area_m2_300K": floor_area_bolometric(bgm, t_floor - bgm, float(thr["floor_t_hot_K"]),
                                                  cube.georef.pixel_area_m2),
            "class": cls, "reasons": rs,
        })
    flags = pd.DataFrame(rows)
    rep["flags"] = flags
    for c in CLASSES:
        rep["counts"][c] = int((flags["class"] == c).sum()) if len(flags) else 0
    rep["status"] = "OK"
    # the detection floor, stated: the 300 K area giving z_min on a median pixel
    med_bg = float(np.nanmedian(st["bg_mean"][interior]))
    med_sig = float(np.nanmedian(st["bg_sigma"][interior]))
    rep["floor"] = {
        "T_bg_K": med_bg, "sigma_K": med_sig, "nsigma": float(thr["floor_nsigma"]),
        "T_hot_K": float(thr["floor_t_hot_K"]),
        "pixel_area_m2": float(cube.georef.pixel_area_m2),
        "A_min_m2": floor_area_bolometric(med_bg, float(thr["floor_nsigma"]) * med_sig,
                                          float(thr["floor_t_hot_K"]),
                                          cube.georef.pixel_area_m2),
        "note": "area of a T_hot source that lifts the bolometric temperature of a median "
                "PSR-interior pixel by nsigma of its own local scatter; a COUNT at this "
                "floor is not an occurrence limit",
    }
    return rep


# ---------------------------------------------------------------------------
# injection
# ---------------------------------------------------------------------------
def inject(cube: DiurnalCube, pixels, f: float, T_hot: float, *, seasons=None, bins=None) -> DiurnalCube:
    """A copy of ``cube`` with a fraction ``f`` of each named pixel covered by
    ``T_hot``.  Restricting ``seasons`` or ``bins`` makes a *passive* source:
    that is what the seasonal and diurnal rejections must catch."""
    out = cube.copy()
    for (s, b), arr in out.arrays.items():
        if seasons is not None and s not in seasons:
            continue
        if bins is not None and b not in bins:
            continue
        for (i, j) in pixels:
            Tc = float(arr[i, j])
            if np.isfinite(Tc):
                arr[i, j] = float(mix_bolometric(Tc, f, T_hot))
    return out


def crop_cube(cube: DiurnalCube, i0: int, j0: int, n: int) -> DiurnalCube:
    """The ``n x n`` sub-cube whose top-left pixel is ``(i0, j0)``, with the
    georef carried across so longitudes and latitudes stay correct."""
    g = cube.georef
    sub = Georef(projection=g.projection, center_lat=g.center_lat, center_lon=g.center_lon,
                 map_scale_m=g.map_scale_m, line_offset=g.line_offset - i0,
                 sample_offset=g.sample_offset - j0, lines=n, samples=n, radius_m=g.radius_m,
                 map_resolution_ppd=g.map_resolution_ppd, lon_direction=g.lon_direction)
    out = DiurnalCube(cube.pole, sub, {}, dict(cube.sources))
    for k, arr in cube.arrays.items():
        out.arrays[k] = np.ascontiguousarray(arr[i0:i0 + n, j0:j0 + n])
    return out


def sensitivity(cube: DiurnalCube, thr: dict | None, areas_m2, T_hot: float = 300.0, *,
                n_per_area: int = 20, seed: int = 11, hardware: list | None = None,
                external_psr: np.ndarray | None = None, window_px: int | None = None) -> dict:
    """Injection–recovery on the real maps: the fraction of injected sources
    of each area that come back classed ``candidate``.

    Every statistic the screen uses is LOCAL — the annulus background has
    radius ``r_out_px`` and the largest neighbourhood rule looks
    ``stripe_window`` pixels along a row — so a source injected at the centre
    of a window several times that size is screened with exactly the numbers
    the full map would give it.  ``window_px`` sets that window; without it
    the whole map is re-screened per injection, which on the 2535 x 2535
    polar grid is minutes per trial and not a table anyone would wait for.
    """
    thr = {**DEFAULT_THRESHOLDS, **(thr or {})}
    m = cold_trap_mask(cube, thr, external=external_psr)
    if m["mask"] is None or m["n_interior"] == 0:
        return {"status": "NO_PSR_INTERIOR", "rows": []}
    ii, jj = np.nonzero(m["interior"])
    if len(ii) == 0:
        return {"status": "NO_PSR_INTERIOR", "rows": []}
    lines, samples = cube.shape
    w = int(window_px) if window_px else 0
    # the window must hold the widest neighbourhood any rule consults
    w_min = 2 * (int(thr["r_out_px"]) + int(thr["stripe_window"]) // 2 + int(thr["edge_px"])) + 3
    if w and w < w_min:
        w = w_min
    use_window = bool(w) and w < min(lines, samples)
    rng = np.random.default_rng(seed)
    px = float(cube.georef.pixel_area_m2)
    out = []
    for a in areas_m2:
        f = float(a) / px
        n_ok = 0
        n_try = 0
        for _ in range(int(n_per_area)):
            k = int(rng.integers(0, len(ii)))
            i, j = int(ii[k]), int(jj[k])
            if use_window:
                i0 = int(np.clip(i - w // 2, 0, lines - w))
                j0 = int(np.clip(j - w // 2, 0, samples - w))
                sub = crop_cube(cube, i0, j0, w)
                psr_sub = (None if external_psr is None
                           else np.asarray(external_psr)[i0:i0 + w, j0:j0 + w])
                ti, tj = i - i0, j - j0
            else:
                sub, psr_sub, ti, tj = cube, external_psr, i, j
            rep = screen_pole(inject(sub, [(ti, tj)], f, T_hot), thr, hardware=hardware,
                              external_psr=psr_sub)
            fl = rep.get("flags")
            n_try += 1
            if fl is not None and len(fl):
                hit = fl[(fl["line"] == ti) & (fl["sample"] == tj) & (fl["class"] == "candidate")]
                n_ok += int(len(hit) > 0)
        out.append({"area_m2": float(a), "f": f, "n": n_try, "n_recovered": n_ok,
                    "recovered_frac": (n_ok / n_try) if n_try else float("nan")})
    return {"status": "OK", "T_hot_K": float(T_hot), "pixel_area_m2": px,
            "window_px": (w if use_window else None), "rows": out}


# ---------------------------------------------------------------------------
# synthetic cube for the offline tests
# ---------------------------------------------------------------------------
def synthetic_cube(n_px: int = 120, *, pole: str = "south", seed: int = 5,
                   scale_m: float = 240.0, n_bins: int = 24) -> tuple[DiurnalCube, np.ndarray]:
    """A PSR floor with a realistic passive diurnal and seasonal signal.

    The floor temperature falls towards the centre of the region; the diurnal
    term is a cosine in local time whose amplitude grows towards the rim (that
    is where scattered light and rim IR reach), and summer is warmer than
    winter by a term with the same spatial shape.  Returns the cube and the
    external PSR raster.
    """
    rng = np.random.default_rng(seed)
    g = Georef(projection="polar_stereographic", center_lat=-90.0 if pole == "south" else 90.0,
               map_scale_m=scale_m, line_offset=n_px / 2.0, sample_offset=n_px / 2.0,
               lines=n_px, samples=n_px)
    y, x = np.mgrid[0:n_px, 0:n_px]
    r = np.hypot(y - n_px / 2.0, x - n_px / 2.0) / (n_px / 2.0)
    psr = r < 0.82
    base = 38.0 + 22.0 * np.clip(r, 0, 1) ** 2          # colder at the centre
    amp = 0.15 + 3.0 * np.clip(r - 0.35, 0, 1) ** 2      # rim-driven diurnal term
    cube = DiurnalCube(pole, g)
    for season, warm in (("summer", 1.0), ("winter", 0.0)):
        for h in range(1, n_bins + 1):
            phase = 2.0 * np.pi * (h - 1) / n_bins
            t = (base + warm * (0.2 + 2.2 * np.clip(r - 0.35, 0, 1) ** 2)
                 + amp * (1.0 + np.cos(phase)) * (0.35 + 0.65 * warm)
                 + rng.normal(0.0, 0.12, size=(n_px, n_px)))
            t = np.where(psr, t, np.nan).astype(np.float32)
            cube.put(season, f"ltim{h:02d}", t, source=f"synthetic/{season}/{h:02d}")
    return cube, psr


__all__ = ["CLASSES", "DEFAULT_THRESHOLDS", "DiurnalCube", "cold_trap_mask", "crop_cube",
           "floor_area_bolometric", "inject", "local_background", "mix_bolometric",
           "screen_pole", "sensitivity", "statistics", "synthetic_cube"]
