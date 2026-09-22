"""An all-sky tiling for IGNITION's ``tiles`` mode.  Pure; no network.

The parent sample is drawn tile by tile: each tile is a box in (RA, Dec) of
about ``dec_step_deg`` on a side, queried at the archive as the smallest cone
that covers it (:func:`cover_radius_deg`) and then cut to the box
(:func:`owns`), so every star is owned by exactly one tile and a partially
completed sweep is an exact statement about the tiles that finished.  Tiles
lying wholly inside the Galactic plane (``|b| < abs_b_min_deg`` everywhere
on their boundary) are never queried.

Why boxes and not HEALPix: ownership must be decidable in one line with no
optional dependency (``healpy`` is not installed on the runner, and a tiling
the assess stage cannot reproduce is a tiling it cannot audit).  The bands are
equal in declination; the number of boxes per band scales as ``cos(dec)`` so
the boxes stay roughly square and equal in area.

The tile order is a fixed pseudo-random permutation, so the first ``k`` tiles
of any shard --- and the union of what ``n`` shards finished in a time-limited
dispatch --- sample the sky uniformly rather than sweeping it from one pole.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: IAU 1958 Galactic pole (J2000), for ``|b|`` without astropy.
_RA_GP = np.radians(192.85948)
_DEC_GP = np.radians(27.12825)

DEFAULT_TILES: dict = {
    "dec_step_deg": 4.0,     # box side; ~2,600 boxes over the sky, ~1,900 at |b| > 15
    "abs_b_min_deg": 15.0,   # the sample's own plane cut; tiles wholly inside are skipped
    "cover_pad_deg": 0.02,   # added to the covering cone radius
    "seed": 20260921,        # the fixed permutation of the tile order
}


def galactic_latitude_deg(ra_deg, dec_deg) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    sb = (np.sin(dec) * np.sin(_DEC_GP)
          + np.cos(dec) * np.cos(_DEC_GP) * np.cos(ra - _RA_GP))
    return np.degrees(np.arcsin(np.clip(sb, -1.0, 1.0)))


def angular_sep_deg(ra1, dec1, ra2, dec2) -> np.ndarray:
    r1, d1, r2, d2 = (np.radians(np.asarray(x, float)) for x in (ra1, dec1, ra2, dec2))
    # Vincenty form: stable at every separation.
    dr = r2 - r1
    num = np.hypot(np.cos(d2) * np.sin(dr),
                   np.cos(d1) * np.sin(d2) - np.sin(d1) * np.cos(d2) * np.cos(dr))
    den = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(dr)
    return np.degrees(np.arctan2(num, den))


def box_area_deg2(ra0: float, ra1: float, dec0: float, dec1: float) -> float:
    dra = np.radians(float(ra1) - float(ra0))
    return float(np.degrees(dra * (np.sin(np.radians(dec1)) - np.sin(np.radians(dec0))))
                 * 180.0 / np.pi)


def _boundary_points(ra0, ra1, dec0, dec1, n: int = 24) -> tuple[np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 1.0, n)
    ras = np.concatenate([ra0 + (ra1 - ra0) * u, ra0 + (ra1 - ra0) * u,
                          np.full(n, ra0), np.full(n, ra1), [(ra0 + ra1) / 2]])
    decs = np.concatenate([np.full(n, dec0), np.full(n, dec1),
                           dec0 + (dec1 - dec0) * u, dec0 + (dec1 - dec0) * u,
                           [(dec0 + dec1) / 2]])
    return ras, decs


def cover_radius_deg(ra0, ra1, dec0, dec1, pad: float = 0.02) -> tuple[float, float, float]:
    """Centre and radius of the smallest simple cone covering the box.

    The centre is the box's midpoint; the radius is the largest separation from
    it to any boundary point (corners, edge midpoints and a dense sampling of
    the edges, which is where the maximum lies).
    """
    rc, dc = (ra0 + ra1) / 2.0, (dec0 + dec1) / 2.0
    ras, decs = _boundary_points(ra0, ra1, dec0, dec1)
    return float(rc), float(dc), float(angular_sep_deg(rc, dc, ras, decs).max() + pad)


def tile_grid(conf: dict | None = None) -> pd.DataFrame:
    """Every tile of the sky, in the fixed sweep order.

    Columns: ``tile`` (a stable id ``d<band>_r<k>``), ``order`` (position in
    the sweep), ``ra0, ra1, dec0, dec1``, ``ra_c, dec_c, radius_deg`` (the
    covering cone), ``area_deg2``, ``max_abs_b`` (over the boundary) and
    ``in_plane`` (True when the whole tile is below ``abs_b_min_deg`` and is
    therefore never queried).
    """
    c = {**DEFAULT_TILES, **(conf or {})}
    step = float(c["dec_step_deg"])
    n_bands = int(np.ceil(180.0 / step))
    step = 180.0 / n_bands
    rows = []
    for j in range(n_bands):
        dec0 = -90.0 + j * step
        dec1 = min(dec0 + step, 90.0)
        dmid = (dec0 + dec1) / 2.0
        n_ra = max(int(np.ceil(360.0 * np.cos(np.radians(dmid)) / step)), 1)
        for k in range(n_ra):
            ra0 = 360.0 * k / n_ra
            ra1 = 360.0 * (k + 1) / n_ra
            rc, dc, rad = cover_radius_deg(ra0, ra1, dec0, dec1, pad=float(c["cover_pad_deg"]))
            bra, bdec = _boundary_points(ra0, ra1, dec0, dec1)
            max_b = float(np.abs(galactic_latitude_deg(bra, bdec)).max())
            rows.append({"tile": f"d{j:02d}_r{k:03d}", "ra0": ra0, "ra1": ra1,
                         "dec0": dec0, "dec1": dec1, "ra_c": rc, "dec_c": dc,
                         "radius_deg": rad, "area_deg2": box_area_deg2(ra0, ra1, dec0, dec1),
                         "max_abs_b": max_b,
                         "in_plane": bool(max_b < float(c["abs_b_min_deg"]))})
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(int(c["seed"]))
    df["order"] = rng.permutation(len(df))
    return df.sort_values("order").reset_index(drop=True)


def owns(tile: dict | pd.Series, ra, dec) -> np.ndarray:
    """Which of ``(ra, dec)`` fall inside the tile's box (half-open on the upper edges)."""
    ra = np.asarray(ra, float) % 360.0
    dec = np.asarray(dec, float)
    ra0, ra1 = float(tile["ra0"]), float(tile["ra1"])
    dec0, dec1 = float(tile["dec0"]), float(tile["dec1"])
    top = (dec <= dec1) if dec1 >= 90.0 else (dec < dec1)
    return (ra >= ra0) & (ra < ra1) & (dec >= dec0) & top


def sky_tiles(conf: dict | None = None) -> pd.DataFrame:
    """The tiles that are actually swept: the grid without the in-plane tiles."""
    g = tile_grid(conf)
    return g[~g["in_plane"]].reset_index(drop=True)


def tiles_for_shard(tiles: pd.DataFrame, shard: int, n_shards: int) -> pd.DataFrame:
    """Round-robin partition in sweep order: tile ``k`` belongs to shard ``k mod n``."""
    n = max(int(n_shards), 1)
    idx = np.arange(len(tiles))
    return tiles[idx % n == int(shard)].reset_index(drop=True)


def tile_field(tile: dict | pd.Series) -> dict:
    """The tile's covering cone in the ``fields`` form the sample stage takes."""
    return {"ra": float(tile["ra_c"]), "dec": float(tile["dec_c"]),
            "radius_deg": float(tile["radius_deg"]), "tile": str(tile["tile"])}


__all__ = ["DEFAULT_TILES", "angular_sep_deg", "box_area_deg2", "cover_radius_deg",
           "galactic_latitude_deg", "owns", "sky_tiles", "tile_field", "tile_grid",
           "tiles_for_shard"]
