"""The thermal physics of CRYPT: anisothermal hot components inside PSRs.

Pure functions over in-memory rasters.  The layer store is a mapping
``(season, channel, statistic) -> 2-D array`` on one polar grid; ``season``
is ``summer`` / ``winter`` / ``all``; ``channel`` is ``tbol`` or ``"3"`` …
``"9"``; ``statistic`` is ``avg`` / ``max`` / ``min`` / ``count``.

The statistic
-------------
For a single-temperature surface every Diviner channel returns the same
brightness temperature (to within emissivity, which is a smooth, calibratable
offset).  A sub-pixel component at T_hot ≫ T_cold covering a fraction f of
the pixel raises the *short* channels (6: 13–23 µm, 7: 25–41 µm) far more
than the long ones (8: 50–100 µm, 9: 100–400 µm), because at 40–100 K the
short-channel radiance is on the exponential tail of the Planck function and
the hot component's contribution dominates it.  So

    Δ_k = T_k − T_ref            (ref = the longest channel with data)

is the observable, its noise is *measured* from the PSR interior itself in
bins of T_ref (median → calibration offset, 1.4826·MAD → σ), and
z_k = (Δ_k − median)/σ is the detection statistic.  The floor is stated as
the 300 K source area that would give 5σ in each bin (``floor_table``).

The rules (each a named rejection counter)
------------------------------------------
* ``seasonal``: z ≥ z_min in one season and < z_other_max in the other — a
  lit rim, secondary illumination or scattered light; an internal source is
  season-independent.
* ``single_season``: only one season has data at the pixel — cannot be a
  candidate, recorded as interest.
* ``extended``: the connected patch of elevated z around it exceeds
  ``max_cluster_px`` — a boulder field, ejecta or a warm slope, never a
  point source.
* ``stripe``: ≥ ``stripe_min`` flagged pixels share the row or column within
  ``stripe_window`` — calibration striping or a map seam.
* ``low_count``: fewer than ``n_min`` observations in either season.
* ``inconsistent_spectrum``: the two-component fit does not beat the
  single-temperature fit by ``dchi2_min``, or the channel ordering is wrong
  (Δ8 > Δ7 beyond noise) — noise at the floor, not a hot component.
* ``unstable``: the fitted covering fraction differs between seasons by more
  than ``f_ratio_max`` — a source whose heat output is seasonal.
* ``human_hardware``: within a listed spacecraft site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

from .labels import Georef

# Diviner channel bandpasses (µm), Paige et al. 2010, Space Sci. Rev. 150, 125.
CHANNELS: dict[str, tuple[float, float]] = {
    "3": (7.55, 8.05), "4": (8.10, 8.40), "5": (8.38, 8.68),
    "6": (13.0, 23.0), "7": (25.0, 41.0), "8": (50.0, 100.0), "9": (100.0, 400.0),
}
SHORT_ORDER = ("6", "7", "8")        # shortest first
REF_ORDER = ("9", "8")               # longest first

C1 = 1.191042972e8     # W µm^4 m^-2 sr^-1  (2 h c^2)
C2 = 14387.7688        # µm K               (h c / k)

CLASSES = ("candidate", "seasonal", "single_season", "extended", "stripe", "low_count",
           "inconsistent_spectrum", "unstable", "human_hardware")

DEFAULT_THRESHOLDS: dict = {
    "psr_tmax_K": 110.0,        # cold-trap ceiling on the summer/all-time bolometric maximum
    "edge_px": 2,               # erosion of the PSR mask (footprints from the boundary)
    "n_min": 10,                # minimum observation count per pixel per season
    "z_min": 5.0,               # detection threshold on the primary short channel
    "z_other_max": 2.5,         # below this in the other season → seasonal
    "max_cluster_px": 6,        # connected z >= z_min/2 patch larger than this → extended
    "stripe_min": 4,            # flagged pixels sharing a row/column → stripe
    "stripe_window": 200,       # along the row/column, pixels
    "dchi2_min": 9.0,           # two-component must beat one-component by this
    "f_ratio_max": 3.0,         # seasonal covering-fraction ratio allowed for a candidate
    "bin_width_K": 2.0,         # T_ref bins for the empirical noise
    "bin_min_n": 50,            # pixels per bin before its own MAD is used
    "sigma_floor_K": 0.05,      # never quote a σ below this
    "min_coverage_frac": 0.2,   # a short channel needs this fraction of the interior finite
    "t_hot_grid_K": [150, 200, 250, 300, 350, 400, 500, 600, 800],
    "f_grid_log10": [-7.0, 0.0, 71],
    "floor_t_hot_K": 300.0,
    "floor_nsigma": 5.0,
}


# ---------------------------------------------------------------------------
# Planck, band radiance, brightness temperature
# ---------------------------------------------------------------------------
def planck(lam_um, T):
    """Spectral radiance B_λ in W m⁻² sr⁻¹ µm⁻¹ (λ in µm, T in K)."""
    lam = np.asarray(lam_um, dtype=float)
    T = np.asarray(T, dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        x = C2 / (lam * T)
        out = C1 / lam**5 / np.expm1(x)
    return np.where(np.isfinite(out), out, 0.0)


class BandModel:
    """Band-averaged radiance L_k(T) and its inverse for every Diviner channel,
    tabulated once on a log temperature grid (5–2000 K) and interpolated."""

    def __init__(self, channels: dict | None = None, n_lam: int = 96, n_T: int = 1400):
        self.channels = dict(channels or CHANNELS)
        self.T_grid = np.logspace(np.log10(5.0), np.log10(2000.0), n_T)
        self.logL: dict[str, np.ndarray] = {}
        for k, (lo, hi) in self.channels.items():
            lam = np.linspace(lo, hi, n_lam)
            B = planck(lam[None, :], self.T_grid[:, None])
            L = np.trapezoid(B, lam, axis=1) / (hi - lo)
            self.logL[k] = np.log(np.maximum(L, 1e-300))
        self.logT = np.log(self.T_grid)

    def radiance(self, k: str, T):
        """L_k(T), vectorised; NaN in → NaN out."""
        T = np.asarray(T, dtype=float)
        out = np.full(T.shape, np.nan)
        ok = np.isfinite(T) & (T > 0)
        if ok.any():
            out[ok] = np.exp(np.interp(np.log(T[ok]), self.logT, self.logL[k]))
        return out

    def bt(self, k: str, L):
        """Brightness temperature of band radiance L in channel k."""
        L = np.asarray(L, dtype=float)
        out = np.full(L.shape, np.nan)
        ok = np.isfinite(L) & (L > 0)
        if ok.any():
            out[ok] = np.exp(np.interp(np.log(L[ok]), self.logL[k], self.logT))
        return out

    def two_component_bt(self, k: str, T_cold, f, T_hot):
        """BT_k of a pixel that is (1−f) at T_cold and f at T_hot."""
        Lc = self.radiance(k, T_cold)
        Lh = self.radiance(k, T_hot)
        f = np.asarray(f, dtype=float)
        return self.bt(k, (1.0 - f) * Lc + f * Lh)


_MODEL: BandModel | None = None


def band_model() -> BandModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = BandModel()
    return _MODEL


# ---------------------------------------------------------------------------
# layer store
# ---------------------------------------------------------------------------
@dataclass
class PoleLayers:
    """Rasters of one pole on one grid."""

    pole: str
    georef: Georef
    arrays: dict = field(default_factory=dict)      # (season, channel, stat) -> ndarray
    sources: dict = field(default_factory=dict)     # (season, channel, stat) -> provenance

    def put(self, season: str, channel: str, stat: str, arr: np.ndarray, source=None) -> None:
        arr = np.asarray(arr, dtype=np.float32)
        if self.arrays:
            shape = next(iter(self.arrays.values())).shape
            if arr.shape != shape:
                raise ValueError(f"layer ({season},{channel},{stat}) shape {arr.shape} != grid {shape}")
        self.arrays[(season, str(channel), stat)] = arr
        if source is not None:
            self.sources[(season, str(channel), stat)] = source

    def get(self, season: str, channel: str, stat: str):
        return self.arrays.get((season, str(channel), stat))

    @property
    def seasons(self) -> list[str]:
        return sorted({s for (s, _c, _t) in self.arrays})

    @property
    def shape(self):
        return next(iter(self.arrays.values())).shape if self.arrays else (0, 0)

    def channels(self, season: str, stat: str = "avg") -> list[str]:
        return sorted({c for (s, c, t) in self.arrays if s == season and t == stat and c != "tbol"},
                      key=lambda c: float(c))

    def copy(self) -> PoleLayers:
        return PoleLayers(self.pole, self.georef, {k: v.copy() for k, v in self.arrays.items()},
                          dict(self.sources))


# ---------------------------------------------------------------------------
# the PSR (cold-trap) mask
# ---------------------------------------------------------------------------
def psr_mask(layers: PoleLayers, thr: dict, external: np.ndarray | None = None) -> dict:
    """The cold-trap mask and its interior.

    ``all``/``summer`` bolometric maximum below ``psr_tmax_K`` (the Diviner
    definition of a cold trap: never warmer than the ceiling at any time of
    year) — or, when no maximum product exists, the average of the reference
    channel (degraded).  An external PSR raster (LOLA illumination), when
    given, is ANDed in and its route recorded.
    """
    src, arr = None, None
    for season in ("all", "summer", "winter"):
        for ch in ("tbol", "9", "8", "7"):
            a = layers.get(season, ch, "max")
            if a is not None:
                src, arr = f"{season}/{ch}/max", a
                break
        if arr is not None:
            break
    degraded = []
    if arr is None:
        for season in ("all", "summer", "winter"):
            for ch in ("tbol", "9", "8"):
                a = layers.get(season, ch, "avg")
                if a is not None:
                    src, arr = f"{season}/{ch}/avg", a
                    degraded.append("mask_from_average_not_maximum")
                    break
            if arr is not None:
                break
    if arr is None:
        return {"mask": None, "interior": None, "source": None, "degraded": ["no_layer_for_mask"],
                "n_mask": 0, "n_interior": 0}
    mask = np.isfinite(arr) & (arr < float(thr["psr_tmax_K"]))
    if external is not None:
        if external.shape != mask.shape:
            degraded.append("external_psr_shape_mismatch_ignored")
        else:
            mask &= external > 0
            src = f"{src} AND external_psr"
    edge = int(thr.get("edge_px", 2))
    interior = ndimage.binary_erosion(mask, iterations=edge, border_value=0) if edge > 0 else mask.copy()
    return {"mask": mask, "interior": interior, "source": src, "degraded": degraded,
            "n_mask": int(mask.sum()), "n_interior": int(interior.sum())}


# ---------------------------------------------------------------------------
# the anisothermality statistic
# ---------------------------------------------------------------------------
def _robust(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, 1.4826 * mad


def pick_reference(layers: PoleLayers, season: str, interior: np.ndarray, thr: dict) -> str | None:
    for k in REF_ORDER:
        a = layers.get(season, k, "avg")
        if a is not None and np.isfinite(a[interior]).mean() >= float(thr["min_coverage_frac"]):
            return k
    return None


def anisothermality(layers: PoleLayers, season: str, interior: np.ndarray, thr: dict,
                    model: BandModel | None = None) -> dict:
    """Per-channel Δ_k, z_k, the binned noise table and the floor for one season."""
    model = model or band_model()
    ref = pick_reference(layers, season, interior, thr)
    if ref is None:
        return {"season": season, "status": "NO_REFERENCE_CHANNEL", "ref": None, "z": {},
                "delta": {}, "noise": {}, "floor": {}, "primary": None, "coverage": {}}
    T_ref = layers.get(season, ref, "avg")
    bw = float(thr["bin_width_K"])
    lo = math.floor(np.nanmin(T_ref[interior]) / bw) * bw if interior.any() else 0.0
    hi = math.ceil(np.nanmax(T_ref[interior]) / bw) * bw + bw if interior.any() else bw
    edges = np.arange(lo, hi + bw, bw)
    out = {"season": season, "status": "OK", "ref": ref, "z": {}, "delta": {}, "noise": {},
           "floor": {}, "primary": None, "coverage": {}, "T_ref_bins": edges.tolist()}
    sig_floor = float(thr["sigma_floor_K"])
    for k in SHORT_ORDER:
        if k == ref or float(k) >= float(ref):
            continue
        T_k = layers.get(season, k, "avg")
        if T_k is None:
            continue
        both = interior & np.isfinite(T_k) & np.isfinite(T_ref)
        cov = float(both.sum() / max(1, interior.sum()))
        out["coverage"][k] = cov
        if cov < float(thr["min_coverage_frac"]):
            continue
        delta = np.where(both, T_k - T_ref, np.nan).astype(np.float32)
        gmed, gsig = _robust(delta[both])
        gsig = max(gsig, sig_floor)
        z = np.full(delta.shape, np.nan, dtype=np.float32)
        table = []
        idx = np.digitize(T_ref, edges) - 1
        for b in range(len(edges) - 1):
            sel = both & (idx == b)
            n = int(sel.sum())
            if n == 0:
                continue
            if n >= int(thr["bin_min_n"]):
                med, sig = _robust(delta[sel])
                sig = max(sig, sig_floor)
                own = True
            else:
                med, sig, own = gmed, gsig, False
            z[sel] = (delta[sel] - med) / sig
            Tc = 0.5 * (edges[b] + edges[b + 1])
            table.append({"T_lo": float(edges[b]), "T_hi": float(edges[b + 1]), "n": n,
                          "median_K": float(med), "sigma_K": float(sig), "own_noise": own,
                          "A_min_m2": floor_area(model, k, Tc, sig * float(thr["floor_nsigma"]),
                                                 float(thr["floor_t_hot_K"]),
                                                 layers.georef.pixel_area_m2)})
        out["delta"][k] = delta
        out["z"][k] = z
        out["noise"][k] = {"global_median_K": gmed, "global_sigma_K": gsig, "bins": table}
        if out["primary"] is None:
            out["primary"] = k
    if out["primary"] is None:
        out["status"] = "NO_SHORT_CHANNEL"
    return out


def floor_area(model: BandModel, k: str, T_cold: float, dT: float, T_hot: float,
               pixel_area_m2: float) -> float:
    """The source area (m²) at T_hot that raises channel k by dT on a T_cold pixel."""
    if not (np.isfinite(dT) and dT > 0 and np.isfinite(T_cold) and T_cold > 0):
        return float("nan")
    lo, hi = -9.0, 0.0
    target = T_cold + dT
    if model.two_component_bt(k, T_cold, 1.0, T_hot) < target:
        return float("nan")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if model.two_component_bt(k, T_cold, 10.0**mid, T_hot) < target:
            lo = mid
        else:
            hi = mid
    return float(10.0**hi * pixel_area_m2)


def fit_two_component(model: BandModel, bts: dict, sigmas: dict, T_cold: float, thr: dict) -> dict:
    """Grid fit of (f, T_hot) to the channel brightness temperatures of one pixel.

    ``bts``/``sigmas`` map channel → observed BT / σ (the empirical per-channel
    σ at that T_ref).  The single-temperature model is T_k = T_cold + median
    offset (already removed in z, so here it is T_k = T_cold).  Returns the
    best (f, T_hot), the χ² of both models and Δχ².
    """
    ks = [k for k in bts if np.isfinite(bts[k]) and np.isfinite(sigmas.get(k, np.nan))]
    if not ks or not np.isfinite(T_cold):
        return {"f": np.nan, "T_hot": np.nan, "chi2_1": np.nan, "chi2_2": np.nan, "dchi2": np.nan,
                "n_channels": 0}
    obs = np.array([bts[k] for k in ks])
    sig = np.array([max(sigmas[k], float(thr["sigma_floor_K"])) for k in ks])
    chi2_1 = float(np.sum(((obs - T_cold) / sig) ** 2))
    a, b, n = thr["f_grid_log10"]
    f_grid = np.logspace(float(a), float(b), int(n))
    best = (np.inf, np.nan, np.nan)
    for Th in thr["t_hot_grid_K"]:
        pred = np.stack([model.two_component_bt(k, T_cold, f_grid, float(Th)) for k in ks], axis=1)
        chi2 = np.sum(((pred - obs[None, :]) / sig[None, :]) ** 2, axis=1)
        i = int(np.nanargmin(chi2))
        if chi2[i] < best[0]:
            best = (float(chi2[i]), float(f_grid[i]), float(Th))
    return {"f": best[1], "T_hot": best[2], "chi2_1": chi2_1, "chi2_2": best[0],
            "dchi2": chi2_1 - best[0], "n_channels": len(ks)}


# ---------------------------------------------------------------------------
# the screen
# ---------------------------------------------------------------------------
def _sigma_at(noise: dict, T: float, thr: dict) -> float:
    for b in noise.get("bins", []):
        if b["T_lo"] <= T < b["T_hi"]:
            return float(b["sigma_K"])
    return float(noise.get("global_sigma_K", thr["sigma_floor_K"]))


def _median_at(noise: dict, T: float) -> float:
    for b in noise.get("bins", []):
        if b["T_lo"] <= T < b["T_hi"]:
            return float(b["median_K"])
    return float(noise.get("global_median_K", 0.0))


def _hardware_hits(lon: float, lat: float, sites: list, radius_m: float) -> list[str]:
    hits = []
    for s in sites or []:
        try:
            dlat = math.radians(lat - float(s["lat"]))
            dlon = math.radians(((lon - float(s["lon"]) + 180.0) % 360.0) - 180.0)
            d = 1_737_400.0 * math.hypot(dlat, dlon * math.cos(math.radians(lat)))
            if d <= float(s.get("radius_m", radius_m)):
                hits.append(str(s.get("name", "?")))
        except (KeyError, TypeError, ValueError):
            continue
    return hits


def screen_pole(layers: PoleLayers, thr: dict | None = None, *, hardware: list | None = None,
                external_psr: np.ndarray | None = None, model: BandModel | None = None) -> dict:
    """Run the whole thermal screen on one pole.  Returns a report dict with
    the flagged-pixel table (``flags``, a DataFrame), the class counts, the
    mask numbers, the per-season noise/floor tables and the degradations."""
    thr = {**DEFAULT_THRESHOLDS, **(thr or {})}
    model = model or band_model()
    rep: dict = {"pole": layers.pole, "seasons": layers.seasons, "degraded": [], "counts": {}}
    m = psr_mask(layers, thr, external=external_psr)
    rep["mask"] = {k: v for k, v in m.items() if k not in ("mask", "interior")}
    rep["degraded"] += m["degraded"]
    if m["mask"] is None or m["n_interior"] == 0:
        rep["status"] = "NO_PSR_INTERIOR"
        rep["flags"] = pd.DataFrame()
        rep["counts"] = {c: 0 for c in CLASSES}
        return rep
    interior = m["interior"]
    seasons = [s for s in ("summer", "winter") if any(k[0] == s for k in layers.arrays)]
    if not seasons:
        seasons = [s for s in layers.seasons]
    per_season = {s: anisothermality(layers, s, interior, thr, model) for s in seasons}
    rep["per_season"] = {s: {k: v for k, v in r.items() if k not in ("z", "delta")}
                         for s, r in per_season.items()}
    usable = [s for s in seasons if per_season[s]["status"] == "OK"]
    if not usable:
        rep["status"] = "NO_USABLE_SEASON"
        rep["flags"] = pd.DataFrame()
        rep["counts"] = {c: 0 for c in CLASSES}
        return rep
    if len(usable) < 2:
        rep["degraded"].append("single_season_product")
    # primary z per season and the union of flagged pixels
    z_min = float(thr["z_min"])
    flagged = np.zeros(layers.shape, dtype=bool)
    zprim: dict[str, np.ndarray] = {}
    for s in usable:
        z = per_season[s]["z"][per_season[s]["primary"]]
        zprim[s] = z
        flagged |= np.isfinite(z) & (z >= z_min)
    ii, jj = np.nonzero(flagged)
    rep["n_flagged_any_season"] = int(len(ii))
    # extended: connected components of z >= z_min/2 in any season
    elevated = np.zeros(layers.shape, dtype=bool)
    for s in usable:
        elevated |= np.isfinite(zprim[s]) & (zprim[s] >= 0.5 * z_min)
    lab, _n = ndimage.label(elevated, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(lab.ravel())
    rows = []
    lon, lat = layers.georef.pix_to_lonlat(ii, jj)
    for n_, (i, j) in enumerate(zip(ii.tolist(), jj.tolist(), strict=True)):
        row = {"pole": layers.pole, "line": i, "sample": j, "lon": float(lon[n_]),
               "lat": float(lat[n_]), "cluster_px": int(sizes[lab[i, j]]) if lab[i, j] else 1}
        reasons = []
        zs, fs = {}, {}
        excess: dict[str, tuple[float, float]] = {}
        for s in usable:
            r = per_season[s]
            ref = r["ref"]
            prim = r["primary"]
            T_ref = float(layers.get(s, ref, "avg")[i, j])
            row[f"T_ref_{s}"] = T_ref
            cnt = layers.get(s, ref, "count")
            row[f"count_{s}"] = float(cnt[i, j]) if cnt is not None else np.nan
            bts, sigs = {}, {}
            for k, z in r["z"].items():
                zz = float(z[i, j])
                row[f"z{k}_{s}"] = zz
                Tk = float(layers.get(s, k, "avg")[i, j])
                row[f"T{k}_{s}"] = Tk
                if np.isfinite(Tk) and np.isfinite(T_ref):
                    bts[k] = Tk - _median_at(r["noise"][k], T_ref)
                    sigs[k] = _sigma_at(r["noise"][k], T_ref, thr)
            zs[s] = float(zprim[s][i, j])
            # the hot component's excess radiance in the primary channel:
            # f·L(T_hot), the same in every season for a constant source
            if prim in bts and np.isfinite(T_ref):
                Lobs = float(model.radiance(prim, bts[prim]))
                Lcold = float(model.radiance(prim, T_ref))
                dLdT = float(model.radiance(prim, bts[prim] + 0.5) - model.radiance(prim, bts[prim] - 0.5))
                excess[s] = (Lobs - Lcold, abs(dLdT) * sigs[prim])
                row[f"excess_{s}"] = excess[s][0]
                row[f"excess_sigma_{s}"] = excess[s][1]
            fit = fit_two_component(model, bts, sigs, T_ref, thr)
            fs[s] = fit["f"]
            row[f"f_{s}"] = fit["f"]
            row[f"T_hot_{s}"] = fit["T_hot"]
            row[f"area_m2_{s}"] = fit["f"] * layers.georef.pixel_area_m2 if np.isfinite(fit["f"]) else np.nan
            row[f"dchi2_{s}"] = fit["dchi2"]
            # ordering: Δ8 must not exceed Δ7 (or Δ7 exceed Δ6) beyond noise
            ks = sorted(bts, key=float)
            for a_, b_ in zip(ks[:-1], ks[1:], strict=False):
                if (bts[b_] - T_ref) > (bts[a_] - T_ref) + 2.0 * math.hypot(sigs[a_], sigs[b_]):
                    reasons.append(f"ordering_{s}")
                    break
        zs_fin = {s: v for s, v in zs.items() if np.isfinite(v)}
        # classification, first match wins but every reason is kept
        hw = _hardware_hits(row["lon"], row["lat"], hardware or [], 2000.0)
        if hw:
            reasons.append("human_hardware:" + "|".join(hw))
        for s in usable:
            c = row.get(f"count_{s}")
            if c is not None and np.isfinite(c) and c < float(thr["n_min"]):
                reasons.append(f"low_count_{s}")
        if row["cluster_px"] > int(thr["max_cluster_px"]):
            reasons.append("extended")
        if len(zs_fin) < 2:
            reasons.append("single_season")
        else:
            hi_s = max(zs_fin, key=zs_fin.get)
            lo_s = min(zs_fin, key=zs_fin.get)
            if zs_fin[hi_s] >= z_min and zs_fin[lo_s] < float(thr["z_other_max"]):
                reasons.append("seasonal")
            elif zs_fin[lo_s] < z_min:
                reasons.append("below_threshold_other_season")
            # stability: the excess radiance must agree between seasons — a
            # ratio beyond f_ratio_max that is also significant (> 3σ) is a
            # source whose output follows the season, not an internal one
            ex = [excess[s] for s in usable if s in excess and np.isfinite(excess[s][0])]
            if len(ex) == 2:
                (e1, s1), (e2, s2) = ex
                sig = math.hypot(s1, s2)
                lo_e, hi_e = min(e1, e2), max(e1, e2)
                if (abs(e1 - e2) > 3.0 * sig) and (lo_e <= 0 or hi_e / lo_e > float(thr["f_ratio_max"])):
                    reasons.append("unstable")
        for s in usable:
            d = row.get(f"dchi2_{s}")
            if d is not None and np.isfinite(d) and d < float(thr["dchi2_min"]) and zs.get(s, 0) >= z_min:
                reasons.append(f"inconsistent_spectrum_{s}")
        row["reasons"] = ";".join(reasons)
        row["z_primary_max"] = max(zs_fin.values()) if zs_fin else np.nan
        row["z_primary_min"] = min(zs_fin.values()) if zs_fin else np.nan
        rows.append(row)
    flags = pd.DataFrame(rows)
    # stripes need the whole table
    if len(flags):
        w = int(thr["stripe_window"])
        smin = int(thr["stripe_min"])
        stripe = np.zeros(len(flags), dtype=bool)
        li = flags["line"].to_numpy()
        sa = flags["sample"].to_numpy()
        for n_ in range(len(flags)):
            same_row = (li == li[n_]) & (np.abs(sa - sa[n_]) <= w)
            same_col = (sa == sa[n_]) & (np.abs(li - li[n_]) <= w)
            if same_row.sum() >= smin or same_col.sum() >= smin:
                stripe[n_] = True
        flags["reasons"] = [(r + ";stripe" if st and r else ("stripe" if st else r))
                            for r, st in zip(flags["reasons"], stripe, strict=True)]
        flags["class"] = [_classify(r) for r in flags["reasons"]]
    counts = {c: int((flags["class"] == c).sum()) if len(flags) else 0 for c in CLASSES}
    counts["below_threshold_other_season"] = int((flags["class"] == "below_threshold_other_season").sum()) if len(flags) else 0
    rep["counts"] = counts
    rep["flags"] = flags
    rep["status"] = "OK"
    rep["n_candidates"] = counts["candidate"]
    rep["thresholds"] = thr
    return rep


def _classify(reasons: str) -> str:
    if not reasons:
        return "candidate"
    # a warm BLOCK is extended even though its rows also look like stripes;
    # a stripe is a line of otherwise isolated pixels
    order = ["human_hardware", "low_count", "extended", "stripe", "single_season", "seasonal",
             "below_threshold_other_season", "unstable", "ordering", "inconsistent_spectrum"]
    toks = reasons.split(";")
    for o in order:
        if any(t.startswith(o) for t in toks):
            return "inconsistent_spectrum" if o == "ordering" else o
    return "candidate"


# ---------------------------------------------------------------------------
# injection
# ---------------------------------------------------------------------------
def inject(layers: PoleLayers, pixels, f: float, T_hot: float, *, seasons=None,
           model: BandModel | None = None) -> PoleLayers:
    """A copy of ``layers`` with a hot component (fraction f at T_hot) added
    to every channel's average BT at the given (line, sample) pixels.  Each
    channel's own observed BT is used as its cold radiance, so calibration
    offsets between channels are preserved."""
    model = model or band_model()
    out = layers.copy()
    seasons = seasons or out.seasons
    for (s, ch, stat), arr in out.arrays.items():
        if stat != "avg" or s not in seasons or ch == "tbol" or ch not in model.channels:
            continue
        for (i, j) in pixels:
            Tc = float(arr[i, j])
            if not np.isfinite(Tc):
                continue
            arr[i, j] = model.two_component_bt(ch, Tc, f, T_hot)
    return out


def sensitivity(layers: PoleLayers, thr: dict | None, areas_m2, T_hot: float = 300.0, *,
                n_per: int = 25, seed: int = 11, hardware=None, external_psr=None,
                model: BandModel | None = None) -> dict:
    """Inject sources of the given areas at random interior pixels (that are
    not already flagged) and report how many come back as ``candidate``."""
    thr = {**DEFAULT_THRESHOLDS, **(thr or {})}
    model = model or band_model()
    base = screen_pole(layers, thr, hardware=hardware, external_psr=external_psr, model=model)
    if base.get("status") != "OK":
        return {"status": base.get("status"), "rows": []}
    m = psr_mask(layers, thr, external=external_psr)
    interior = m["interior"].copy()
    fl = base["flags"]
    if len(fl):
        interior[fl["line"].to_numpy(), fl["sample"].to_numpy()] = False
    # only pixels with finite primary z in every usable season
    for s, r in base.get("per_season", {}).items():
        if r.get("status") == "OK":
            for k in ("6", "7", "8"):
                a = layers.get(s, k, "avg")
                if a is not None and k == r["primary"]:
                    interior &= np.isfinite(a)
            interior &= np.isfinite(layers.get(s, r["ref"], "avg"))
    ii, jj = np.nonzero(interior)
    rng = np.random.default_rng(seed)
    rows = []
    pa = layers.georef.pixel_area_m2
    for A in areas_m2:
        f = float(A) / pa
        if len(ii) == 0:
            rows.append({"area_m2": float(A), "f": f, "n_injected": 0, "n_recovered": 0})
            continue
        pick = rng.choice(len(ii), size=min(n_per, len(ii)), replace=False)
        # inject sites spaced apart so they do not form clusters or stripes
        pix = [(int(ii[p]), int(jj[p])) for p in pick]
        inj = inject(layers, pix, f, T_hot, model=model)
        rep = screen_pole(inj, thr, hardware=hardware, external_psr=external_psr, model=model)
        got = rep["flags"]
        rec, zmed = 0, []
        classes: dict[str, int] = {}
        if len(got):
            key = set(zip(got["line"].tolist(), got["sample"].tolist(), strict=True))
            for (i, j) in pix:
                if (i, j) in key:
                    r = got[(got["line"] == i) & (got["sample"] == j)].iloc[0]
                    classes[r["class"]] = classes.get(r["class"], 0) + 1
                    if r["class"] == "candidate":
                        rec += 1
                    zmed.append(float(r["z_primary_min"]))
        rows.append({"area_m2": float(A), "f": f, "T_hot": T_hot, "n_injected": len(pix),
                     "n_recovered": rec, "recovered_frac": rec / len(pix),
                     "n_flagged": int(sum(classes.values())), "classes": classes,
                     "z_primary_min_median": float(np.median(zmed)) if zmed else np.nan})
    return {"status": "OK", "T_hot": T_hot, "n_per_area": n_per, "rows": rows,
            "pixel_area_m2": pa}


# ---------------------------------------------------------------------------
# synthetic data (tests and the offline smoke run)
# ---------------------------------------------------------------------------
def synthetic_pole(n_px: int = 160, *, pole: str = "south", seed: int = 3, scale_m: float = 240.0,
                   psr_radius_px: int = 45, T_floor_K: float = 40.0, T_summer_K: float = 70.0,
                   T_winter_K: float = 50.0, noise_K: dict | None = None, n_obs: int = 200,
                   model: BandModel | None = None, channels=("6", "7", "8", "9")) -> PoleLayers:
    """A polar map with one round cold trap (summer/winter average BTs that
    agree between channels up to noise) surrounded by warm, lit terrain."""
    model = model or band_model()
    rng = np.random.default_rng(seed)
    noise = {"6": 3.0, "7": 1.2, "8": 0.4, "9": 0.25, **(noise_K or {})}
    from .labels import polar_georef
    g = polar_georef(pole, n_px, scale_m)
    yy, xx = np.mgrid[0:n_px, 0:n_px]
    r = np.hypot(yy - (n_px - 1) / 2.0, xx - (n_px - 1) / 2.0)
    inside = r < psr_radius_px
    L = PoleLayers(pole, g)
    for season, T_in, T_out in (("summer", T_summer_K, 250.0), ("winter", T_winter_K, 120.0)):
        # a gentle radial gradient inside; warm outside
        T_true = np.where(inside, T_in + 0.15 * r, T_out + 50.0 * rng.random((n_px, n_px)))
        T_true = T_true.astype(np.float32)
        for k in channels:
            a = T_true + rng.normal(0.0, noise[k] / math.sqrt(max(1, n_obs / 200)), T_true.shape).astype(np.float32)
            L.put(season, k, "avg", a, source="synthetic")
        L.put(season, "tbol", "avg", T_true, source="synthetic")
        L.put(season, "9", "count", np.full(T_true.shape, float(n_obs), dtype=np.float32), source="synthetic")
    # all-time maximum: the summer average plus a margin inside; hot outside
    tmax = np.where(inside, T_summer_K + 0.15 * r + 8.0, 380.0).astype(np.float32)
    L.put("all", "tbol", "max", tmax, source="synthetic")
    return L


__all__ = ["BandModel", "CHANNELS", "CLASSES", "DEFAULT_THRESHOLDS", "PoleLayers",
           "anisothermality", "band_model", "fit_two_component", "floor_area", "inject",
           "planck", "psr_mask", "screen_pole", "sensitivity", "synthetic_pole"]
