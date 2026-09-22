"""Reading the Diviner Polar Cumulative Products onto a raster grid.

What the products are (read off PCP_AVG_TBOL_POLN_SUM_LTIM01_240.LBL on the
runner, 2026-09-22):

* PDS3, ``RECORD_TYPE = FIXED_LENGTH``, ``ROW_BYTES = 59``, five ASCII
  columns: ``x``, ``y`` (distances from the pole normalised by the mean
  radius 1737.4 km), ``longitude``, ``latitude``, bolometric temperature.
* Record 1 is a text header; the table starts at record 2.
* ``MAP_PROJECTION_TYPE = "POLAR STEREOGRAPHIC"``, ``MAP_SCALE = 240 m/pix``,
  latitude 80 deg to the pole, ``MAP_RESOLUTION = 126.347 pix/degree``.
* **Empty bins are omitted**, so a product is a sparse point list, not a
  raster: ~4.44 million rows, 262 MB each.
* ``LRO:DLRE_CLOCTIME_MIN/MAX`` give the local-time bin: LTIM01 is 0.00-0.25 h,
  so the local-time axis is 96 bins of a quarter hour, and "summer"/"winter"
  are subsolar latitude above/below zero (Williams et al. 2019, JGR 124,
  2505-2521).

Ninety-six bins x two seasons x two poles at 262 MB each is ~100 GB, which is
not a runner-scale download.  The screen does not need every bin: it needs the
diurnal curve sampled well enough to measure its amplitude and its floor.  So
a subsample of local-time bins spread evenly around the clock is loaded, and
which bins were used is recorded with the result.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .labels import Georef

MOON_RADIUS_M = 1737400.0
PCP_SCALE_M = 240.0
#: the PDS4 bundle path the products are actually served from
PCP_URL_TEMPLATE = ("https://pds-geosciences.wustl.edu/lro/"
                    "urn-nasa-pds-lro_diviner_derived1/data_derived_pcp/diurnal/ltim/"
                    "{pole_tag}/pcp_avg_tbol_{pole_tag}_{season_tag}_ltim{bin:02d}_240.{ext}")
POLE_TAG = {"north": "poln", "south": "pols"}
SEASON_TAG = {"summer": "sum", "winter": "win"}


def pcp_product_name(pole: str, season: str, bin_: int) -> str:
    return (f"PCP_AVG_TBOL_{POLE_TAG[pole].upper()}_{SEASON_TAG[season].upper()}"
            f"_LTIM{int(bin_):02d}_240")


def pcp_url(pole: str, season: str, bin_: int, ext: str = "tab") -> str:
    return PCP_URL_TEMPLATE.format(pole_tag=POLE_TAG[pole], season_tag=SEASON_TAG[season],
                                   bin=int(bin_), ext=ext)


def local_time_hours(bin_: int, n_bins: int = 96) -> float:
    """Centre of local-time bin ``bin_`` (1-based) in hours."""
    return (int(bin_) - 0.5) * 24.0 / float(n_bins)


def pcp_georef(pole: str, half_px: int) -> Georef:
    """The 240 m polar stereographic grid the PCP points fall on.

    Pixel-registered, origin at the pole, ``half_px`` pixels from the pole to
    the edge in each direction.
    """
    n = 2 * int(half_px) + 1
    return Georef(projection="polar_stereographic",
                  center_lat=90.0 if pole == "north" else -90.0, center_lon=0.0,
                  map_scale_m=PCP_SCALE_M, line_offset=float(half_px), sample_offset=float(half_px),
                  lines=n, samples=n, radius_m=MOON_RADIUS_M,
                  map_resolution_ppd=126.347)


def half_px_for(min_lat_deg: float = 80.0) -> int:
    """Pixels from the pole to ``min_lat_deg`` on the 240 m stereographic grid."""
    c = np.radians(90.0 - abs(float(min_lat_deg)))
    rho = 2.0 * MOON_RADIUS_M * np.tan(c / 2.0)
    return int(np.ceil(rho / PCP_SCALE_M))


def read_pcp_tab(path, *, chunk_rows: int = 2_000_000) -> dict:
    """Read one PCP table into (x, y, lon, lat, T) float arrays.

    Read in chunks so a 262 MB table never doubles in memory.  The five
    columns are whitespace-separated inside a fixed-length record, so the C
    parser reads them without needing the column widths from DLRE_PCP.FMT.
    """
    xs, ys, lons, lats, ts = [], [], [], [], []
    reader = pd.read_csv(path, sep=r"\s+", header=None, skiprows=1, engine="c",
                         names=["x", "y", "lon", "lat", "t"],
                         dtype={"x": np.float64, "y": np.float64, "lon": np.float32,
                                "lat": np.float32, "t": np.float32},
                         chunksize=int(chunk_rows), on_bad_lines="skip")
    for ch in reader:
        xs.append(ch["x"].to_numpy())
        ys.append(ch["y"].to_numpy())
        lons.append(ch["lon"].to_numpy())
        lats.append(ch["lat"].to_numpy())
        ts.append(ch["t"].to_numpy())
    if not xs:
        return {"n": 0}
    return {"n": int(sum(len(a) for a in xs)),
            "x": np.concatenate(xs), "y": np.concatenate(ys),
            "lon": np.concatenate(lons), "lat": np.concatenate(lats),
            "t": np.concatenate(ts)}


def _indices(x: np.ndarray, y: np.ndarray, half_px: int, flip_y: bool) -> tuple[np.ndarray, np.ndarray]:
    X = x * MOON_RADIUS_M / PCP_SCALE_M
    Y = y * MOON_RADIUS_M / PCP_SCALE_M
    j = np.rint(X).astype(np.int64) + half_px
    i = (half_px + np.rint(Y).astype(np.int64)) if flip_y else (half_px - np.rint(Y).astype(np.int64))
    return i, j


def rasterise(tab: dict, pole: str, half_px: int, *, flip_y: bool | None = None) -> dict:
    """Scatter one PCP table onto the grid, choosing the y sign empirically.

    The label says ``x``/``y`` are distances from the pole but not which way
    the line axis runs, so both are tried and the one whose reconstructed
    longitude/latitude match the table's own lon/lat columns is used.  The
    residual is returned, so a wrong grid can never pass silently.
    """
    if not tab.get("n"):
        return {"status": "EMPTY", "n": 0}
    g = pcp_georef(pole, half_px)
    best = None
    for fy in ([bool(flip_y)] if flip_y is not None else [False, True]):
        i, j = _indices(tab["x"], tab["y"], half_px, fy)
        keep = (i >= 0) & (i < g.lines) & (j >= 0) & (j < g.samples)
        if not keep.any():
            continue
        s = slice(0, min(20000, int(keep.sum())))
        ii, jj = i[keep][s], j[keep][s]
        lon_g, lat_g = g.pix_to_lonlat(ii, jj)
        dlon = np.abs(((lon_g - tab["lon"][keep][s].astype(float) + 180.0) % 360.0) - 180.0)
        dlat = np.abs(lat_g - tab["lat"][keep][s].astype(float))
        # near the pole longitude is degenerate; weight it by cos(lat)
        resid = float(np.nanmedian(dlat + dlon * np.cos(np.radians(lat_g))))
        if best is None or resid < best["resid_deg"]:
            best = {"flip_y": fy, "resid_deg": resid, "i": i, "j": j, "keep": keep}
    if best is None:
        return {"status": "OFF_GRID", "n": int(tab["n"])}
    i, j, keep = best["i"], best["j"], best["keep"]
    arr = np.full((g.lines, g.samples), np.nan, dtype=np.float32)
    t = tab["t"][keep]
    good = np.isfinite(t) & (t > 0)
    arr[i[keep][good], j[keep][good]] = t[good]
    return {"status": "OK", "array": arr, "georef": g, "n": int(tab["n"]),
            "n_on_grid": int(good.sum()), "n_off_grid": int((~keep).sum()),
            "flip_y": best["flip_y"], "georef_resid_deg": best["resid_deg"],
            "t_min": float(np.nanmin(t[good])) if good.any() else float("nan"),
            "t_max": float(np.nanmax(t[good])) if good.any() else float("nan")}


# ---------------------------------------------------------------------------
# the LOLA permanently-shadowed raster
# ---------------------------------------------------------------------------
LPSR_URL = ("https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/"
            "extras/illumination/release_2014/img/lpsr_{tag}_240m.{ext}")


def lpsr_url(pole: str, ext: str = "img") -> str:
    return LPSR_URL.format(tag="65n" if pole == "north" else "65s", ext=ext)


def crop_lpsr(psr: np.ndarray, half_px: int) -> np.ndarray:
    """Crop the 65 deg LPSR raster to the PCP grid.

    Both are pixel-registered 240 m polar stereographic true at the pole, so
    the only transform is a crop about the centre pixel.
    """
    a = np.asarray(psr)
    ci, cj = (a.shape[0] - 1) / 2.0, (a.shape[1] - 1) / 2.0
    n = 2 * int(half_px) + 1
    i0, j0 = int(round(ci)) - int(half_px), int(round(cj)) - int(half_px)
    out = np.zeros((n, n), dtype=bool)
    si0, sj0 = max(i0, 0), max(j0, 0)
    si1, sj1 = min(i0 + n, a.shape[0]), min(j0 + n, a.shape[1])
    if si1 > si0 and sj1 > sj0:
        out[si0 - i0:si1 - i0, sj0 - j0:sj1 - j0] = a[si0:si1, sj0:sj1] > 0
    return out


def parse_bin_spec(spec, n_bins: int = 96) -> list[int]:
    """``[1, 17, 33]`` or ``"every:12"`` or ``"n:8"`` → a list of 1-based bins."""
    if isinstance(spec, (list, tuple)):
        return [int(b) for b in spec]
    s = str(spec).strip()
    m = re.match(r"(?i)^every:(\d+)$", s)
    if m:
        return list(range(1, n_bins + 1, int(m.group(1))))
    m = re.match(r"(?i)^n:(\d+)$", s)
    if m:
        k = max(1, int(m.group(1)))
        return [int(round(1 + t * n_bins / k)) for t in range(k)]
    return [int(x) for x in re.split(r"[,\s]+", s) if x]


__all__ = ["MOON_RADIUS_M", "PCP_SCALE_M", "PCP_URL_TEMPLATE", "crop_lpsr", "half_px_for",
           "local_time_hours", "lpsr_url", "parse_bin_spec", "pcp_georef", "pcp_product_name",
           "pcp_url", "rasterise", "read_pcp_tab"]
