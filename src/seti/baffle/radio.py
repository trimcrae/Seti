"""BAFFLE / radio: a hole in the LoTSS DR2 source counts around a nearby star.

The hypothesis (docs of the BAFFLE channel): a warden civilisation hides the
solar system's technosignatures from other stars with band-selective absorbing
screens ("baffles") placed on the Sun -> observer-star line at heliocentric
distance d (10^2 - 10^4 AU) with radius R >= 1 AU --- a screen must be at least
as large as the source it shadows, and our radio leakage comes from a <= 2 AU
region.  The most urgent thing to hide is our PRESENT radio leakage, so a
radio-opaque, optically transparent baffle (a mesh) is the one most likely to
be already in place.  A thin mesh is thermally invisible, so the only
observable is **reciprocity**: the same screen blocks every background radio
source within angular radius

    rho = R / d  ->  206265" / d_AU for R = 1 AU
                     (34' at 100 AU, 3.4' at 1000 AU, 21" at 10^4 AU)

of the target star X as seen from Earth.  Its apparent centre is displaced
from X by Earth's projected heliocentric position divided by d --- an annual
ellipse of semi-major axis 206265"/d_AU --- so at the single epoch of a radio
survey pointing the void is centred within that displacement of X.  Nobody has
searched a deep radio catalogue for voids centred on nearby stars.

Why LoTSS DR2 and nothing else
------------------------------
A void is only meaningful where enough sources are *expected* inside it.  For
the fiducial 1 AU screen at 1000 AU the aperture is 3.4' in radius, i.e. an
area of pi (3.4/60)^2 = 0.0101 deg^2.  Expected counts in that aperture:

    LoTSS DR2  (LOFAR 144 MHz; Shimwell et al. 2022, A&A 659, A1)
               4.4 million sources over 5,634 deg^2, ~700-1000 / deg^2 -> ~9
    VLASS      ~100 / deg^2  -> ~1.0
    NVSS       ~50  / deg^2  -> ~0.5
    RACS       ~30  / deg^2  -> ~0.3

P(N = 0 | lambda) = exp(-lambda): a zero count in a 3.4' aperture is a
1e-4 event in LoTSS and a ~0.4-0.7 event in every other public survey.  The
shallower catalogues cannot say anything about any void smaller than ~15',
which is a screen at < 230 AU --- inside the range but only its innermost
edge.  They are therefore not implemented; this module is LoTSS DR2 only, and
the floor on what it can see is stated in :func:`void_statistics`.

Sensitivity floor (be honest about it).  With the default configuration the
look-elsewhere factor is n_trials = (1 + 5 distances x 8 phases) centres x 6
apertures = 246, so a survivor at p_min = 1e-5 needs a raw Poisson probability
below 4e-8, i.e. lambda >= 17 for an empty aperture.  At 900 / deg^2 that is
an aperture radius of ~280", so a *1 AU* screen is only detectable at
d <= ~730 AU here; at 1000 AU the screen must be >= 1.4 AU.  A 1 AU screen at
1000 AU comes out at p_raw ~ 1e-4 and is reported as ``not_significant`` ---
the number is in ``voids.csv`` for anyone who wants to look at the tail.

Access route
------------
VizieR TAP (``J/A+A/659/A1``).  **No table or column name is hard-coded**: the
table is discovered from ``TAP_SCHEMA.tables`` (``LIKE '%659/A1%'``) and the
RA / Dec / total-flux columns from ``TAP_SCHEMA.columns`` at run time, by UCD
first and by name second, and every name used is written into the ledger.  If
VizieR discovery fails the ASTRON TAP (``https://vo.astron.org/tap``, which
should hold ``lotss_dr2.main_sources``) is tried the same way, and the run is
recorded as ``DEGRADED_SOURCE`` if it has to use it.

The catalogue is pulled ONCE per 2 deg x 2 deg tile that contains a target or
a target's control neighbourhood (a cone query per star would be ~40k queries;
tiles are ~1,400), each tile checkpointed to ``tiles/*.parquet`` with a ledger
entry ``QUERY_OK`` / ``QUERY_RETURNED_ZERO_ROWS`` / ``QUERY_FAILED``.  One
failed tile never stops the run.

Statistic
---------
For every target X (Gaia DR3, parallax > 20 mas, parallax_over_error > 10,
inside the coarse LoTSS DR2 sky) and every control position (four points 45'
from X in +-RA / +-Dec, which sample the same mosaics):

* local density from the 8'-20' annulus *in the same catalogue*, so mosaic
  depth variations are absorbed;
* for apertures r in {30", 60", 120", 204", 300", 600"} and for the centre
  grid {X} + {annual ellipse at d in 500..10000 AU, 8 phases}: n_obs inside r,
  lambda = density x area, P = P(N <= n_obs | lambda) (regularised upper
  incomplete gamma, exact);
* the best (smallest) P over the grid, its trial count, and the Bonferroni
  p_trials = min(1, P x n_trials).

Vetoes, each a counter: ``outside_footprint_or_masked`` (annulus density below
``min_annulus_density_per_deg2``), ``bright_radio_source_mask`` (a source
brighter than 1 Jy within 30' --- LoTSS DR2 completeness drops near bright
sources), ``low_expected_count`` (no aperture reaches lambda >= 8),
``not_significant``.  The control false-void rate is the empirical null
reported next to the Poisson one.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import re
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaincc

from ..vigil.acquire import QueryResult, run_tap

CHANNEL = "baffle_radio"
ARCSEC_PER_RAD = 206264.80624709636
AU_ARCSEC = 206265.0           # 1 AU at d AU subtends AU_ARCSEC / d arcsec

VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
ASTRON_TAP = "https://vo.astron.org/tap"

VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_DEGRADED = "DEGRADED_SOURCE"
VERDICT_NO_SURVIVOR = "NO_RADIO_VOID_SURVIVOR"
VERDICT_CANDIDATES = "RADIO_VOID_CANDIDATES_PENDING_VET"

VETO_FOOTPRINT = "outside_footprint_or_masked"
VETO_BRIGHT = "bright_radio_source_mask"
VETO_LOW_EXPECTED = "low_expected_count"
VETO_NOT_SIGNIFICANT = "not_significant"
VETO_NONE = "none"
VETO_ORDER = (VETO_FOOTPRINT, VETO_BRIGHT, VETO_LOW_EXPECTED, VETO_NOT_SIGNIFICANT)

GAIA_EPOCH = 2016.0
LOTSS_DR2_MID_EPOCH = 2017.5   # DR2 observations span 2014 - 2020

#: Every threshold, with its default.  ``config/baffle_radio.yaml`` overrides.
DEFAULTS: dict = {
    "lotss": {
        "vizier_tap": VIZIER_TAP,
        "vizier_catalogue_hint": "659/A1",
        "astron_tap": ASTRON_TAP,
        "astron_table_hint": "lotss",
        "max_rows_per_tile": 40000,
        "tap_retries": 3,
    },
    "targets": {
        "parallax_min_mas": 20.0,
        "parallax_over_error_min": 10.0,
        "max_rows": 200000,
        # Coarse LoTSS DR2 sky (Shimwell+2022 Fig. 1), a little generous on
        # every edge: the annulus-density test decides footprint membership.
        "regions": [
            {"name": "13h", "ra_min": 100.0, "ra_max": 280.0, "dec_min": 22.0, "dec_max": 72.0},
            {"name": "0h", "ra_min": 325.0, "ra_max": 30.0, "dec_min": 12.0, "dec_max": 42.0},
        ],
        "etz_ecliptic_lat_deg": 0.264,
        "propagate_pm_to_epoch": LOTSS_DR2_MID_EPOCH,
    },
    "tile_deg": 2.0,
    "apertures_arcsec": [30.0, 60.0, 120.0, 204.0, 300.0, 600.0],
    "annulus_in_arcsec": 480.0,          # 8'
    "annulus_out_arcsec": 1200.0,        # 20'
    "min_annulus_density_per_deg2": 300.0,
    "bright_source_jy": 1.0,
    "bright_source_radius_arcsec": 1800.0,   # 30'
    "min_expected_count": 8.0,
    "p_min": 1.0e-5,
    "baffle_distances_au": [500.0, 1000.0, 2000.0, 5000.0, 10000.0],
    "n_phase": 8,
    "control_offset_arcsec": 2700.0,     # 45'
    "probe": {"ra": 200.0, "dec": 50.0, "half_width_deg": 0.05},
}


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
def _deep_update(base: dict, over: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_radio_config(path: Path | str | None = None) -> dict:
    """``config/baffle_radio.yaml`` (its ``radio:`` mapping) over :data:`DEFAULTS`.

    A missing or unreadable file degrades to the defaults and says so.  The
    returned mapping is what every function here calls ``cfg``; a caller that
    already merged the file into a larger mapping passes ``cfg["radio"]``.
    """
    try:
        import yaml
        if path is None:
            path = Path(__file__).resolve().parents[3] / "config" / "baffle_radio.yaml"
        path = Path(path)
        if not path.exists():
            return _deep_update(DEFAULTS, {})
        doc = yaml.safe_load(path.read_text()) or {}
        return _deep_update(DEFAULTS, doc.get("radio", doc) or {})
    except Exception as exc:                                  # noqa: BLE001
        print(f"[{CHANNEL}] config not loaded ({exc!r}); using defaults")
        return _deep_update(DEFAULTS, {})


def _radio_cfg(cfg: dict | None) -> dict:
    """The ``radio`` mapping from a full merged config, else the file."""
    if cfg and isinstance(cfg.get("radio"), dict):
        return _deep_update(DEFAULTS, cfg["radio"])
    return load_radio_config()


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def angular_separation_arcsec(ra1, dec1, ra2, dec2):
    """Great-circle separation (Vincenty form; stable at all separations)."""
    ra1 = np.radians(np.asarray(ra1, dtype=float))
    dec1 = np.radians(np.asarray(dec1, dtype=float))
    ra2 = np.radians(np.asarray(ra2, dtype=float))
    dec2 = np.radians(np.asarray(dec2, dtype=float))
    dra = ra2 - ra1
    s1, c1 = np.sin(dec1), np.cos(dec1)
    s2, c2 = np.sin(dec2), np.cos(dec2)
    num = np.hypot(c2 * np.sin(dra), c1 * s2 - s1 * c2 * np.cos(dra))
    den = s1 * s2 + c1 * c2 * np.cos(dra)
    return np.degrees(np.arctan2(num, den)) * 3600.0


def offset_position(ra, dec, dx_arcsec, dy_arcsec):
    """Move (ra, dec) by a tangent-plane offset (east, north) in arcsec."""
    dec_new = float(dec) + dy_arcsec / 3600.0
    dec_new = float(np.clip(dec_new, -90.0, 90.0))
    cosd = math.cos(math.radians(0.5 * (float(dec) + dec_new)))
    cosd = max(cosd, 1e-6)
    ra_new = (float(ra) + dx_arcsec / 3600.0 / cosd) % 360.0
    return ra_new, dec_new


def aperture_area_deg2(r_arcsec: float) -> float:
    return math.pi * (float(r_arcsec) / 3600.0) ** 2


def annulus_area_deg2(r_in_arcsec: float, r_out_arcsec: float) -> float:
    return math.pi * ((float(r_out_arcsec) / 3600.0) ** 2 - (float(r_in_arcsec) / 3600.0) ** 2)


def _arrays(sources_df) -> tuple[np.ndarray, np.ndarray]:
    if sources_df is None or len(sources_df) == 0:
        return np.empty(0), np.empty(0)
    return (np.asarray(sources_df["ra"], dtype=float),
            np.asarray(sources_df["dec"], dtype=float))


def count_in_aperture(sources_df, ra: float, dec: float, r_arcsec: float) -> int:
    """Number of catalogue sources within ``r_arcsec`` of (ra, dec)."""
    sra, sdec = _arrays(sources_df)
    if sra.size == 0:
        return 0
    sep = angular_separation_arcsec(ra, dec, sra, sdec)
    return int(np.count_nonzero(sep < float(r_arcsec)))


def annulus_density(sources_df, ra: float, dec: float, r_in_arcsec: float,
                    r_out_arcsec: float) -> float:
    """Local source density (per deg^2) from the annulus r_in <= sep < r_out."""
    sra, sdec = _arrays(sources_df)
    if sra.size == 0:
        return 0.0
    sep = angular_separation_arcsec(ra, dec, sra, sdec)
    n = int(np.count_nonzero((sep >= float(r_in_arcsec)) & (sep < float(r_out_arcsec))))
    return n / annulus_area_deg2(r_in_arcsec, r_out_arcsec)


def poisson_void_p(n_obs, lam):
    """P(N <= n_obs | lambda) = Q(n_obs + 1, lambda), the regularised upper
    incomplete gamma function --- exact, and vectorised over both arguments.
    lambda = 0 gives 1 (nothing was expected, so nothing missing is not a void).
    """
    n = np.asarray(n_obs, dtype=float)
    lam_a = np.asarray(lam, dtype=float)
    with np.errstate(invalid="ignore"):
        p = np.where(lam_a <= 0, 1.0, gammaincc(np.floor(n) + 1.0, np.maximum(lam_a, 0.0)))
    return float(p) if p.ndim == 0 else p


def ecliptic_latitude_deg(ra, dec):
    """Barycentric mean ecliptic latitude of ICRS (ra, dec), in degrees."""
    from astropy.coordinates import SkyCoord
    c = SkyCoord(ra=np.asarray(ra, dtype=float), dec=np.asarray(dec, dtype=float),
                 unit="deg", frame="icrs")
    return np.atleast_1d(c.barycentricmeanecliptic.lat.deg)


def earth_heliocentric_au(times):
    """Earth's heliocentric ICRS-oriented Cartesian position (3, n) in AU.

    Built-in ephemeris only (ERFA epv00): no download, no IERS table.
    """
    from astropy import units as u
    from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
    with solar_system_ephemeris.set("builtin"):
        e = get_body_barycentric("earth", times)
        s = get_body_barycentric("sun", times)
    return (e - s).xyz.to_value(u.AU)


@functools.lru_cache(maxsize=16)
def _earth_at_phases(t0: str, n_phase: int) -> np.ndarray:
    """Earth's heliocentric position at n_phase equal steps of a year from t0.

    The same for every star, so it is computed once per (t0, n_phase): the
    ephemeris call was ~90 % of the screen's run time before this cache.
    """
    from astropy import units as u
    from astropy.time import Time
    phases = np.arange(int(n_phase)) / float(n_phase)
    return earth_heliocentric_au(Time(t0, scale="tdb") + phases * u.yr)


def baffle_centre_grid(ra: float, dec: float, d_au: float, n_phase: int = 8,
                       t0: str = "2017-01-01") -> pd.DataFrame:
    """Where a screen at heliocentric distance ``d_au`` on the Sun -> X line
    appears from Earth, at ``n_phase`` phases of the year.

    A screen at S + d n_hat is seen from Earth (at S + r_E) in the direction
    d n_hat - r_E, so its apparent centre is displaced from X by
    -(r_E projected on the tangent plane) / d.  Over a year that traces an
    ellipse of semi-major axis 206265"/d_au (the projection of Earth's orbit)
    and semi-minor axis 206265"/d_au x |sin(ecliptic latitude)|.

    Returns one row per phase: ``phase`` (fraction of the year from ``t0``),
    ``dx_arcsec`` (east), ``dy_arcsec`` (north), ``ra``, ``dec``.
    """
    n_phase = int(n_phase)
    phases = np.arange(n_phase) / float(n_phase)
    r = _earth_at_phases(str(t0), n_phase)               # (3, n)
    ra_r, dec_r = math.radians(float(ra)), math.radians(float(dec))
    east = np.array([-math.sin(ra_r), math.cos(ra_r), 0.0])
    north = np.array([-math.sin(dec_r) * math.cos(ra_r),
                      -math.sin(dec_r) * math.sin(ra_r), math.cos(dec_r)])
    dx = -(east @ r) / float(d_au) * ARCSEC_PER_RAD
    dy = -(north @ r) / float(d_au) * ARCSEC_PER_RAD
    rows = []
    for k in range(n_phase):
        ra_k, dec_k = offset_position(ra, dec, dx[k], dy[k])
        rows.append({"phase": float(phases[k]), "d_au": float(d_au),
                     "dx_arcsec": float(dx[k]), "dy_arcsec": float(dy[k]),
                     "ra": ra_k, "dec": dec_k})
    return pd.DataFrame(rows)


def centre_grid(ra: float, dec: float, cfg: dict) -> pd.DataFrame:
    """X itself first, then every (distance, phase) baffle centre."""
    rows = [pd.DataFrame([{"phase": float("nan"), "d_au": float("inf"),
                           "dx_arcsec": 0.0, "dy_arcsec": 0.0,
                           "ra": float(ra), "dec": float(dec)}])]
    for d in cfg.get("baffle_distances_au", DEFAULTS["baffle_distances_au"]):
        rows.append(baffle_centre_grid(ra, dec, float(d), int(cfg.get("n_phase", 8))))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# Tiles
# --------------------------------------------------------------------------
def tile_index(ra, dec, tile_deg: float) -> tuple[np.ndarray, np.ndarray]:
    n_ra = int(round(360.0 / tile_deg))
    ix = (np.floor(np.mod(np.asarray(ra, dtype=float), 360.0) / tile_deg).astype(int)) % n_ra
    iy = np.floor((np.asarray(dec, dtype=float) + 90.0) / tile_deg).astype(int)
    iy = np.clip(iy, 0, int(math.ceil(180.0 / tile_deg)) - 1)
    return ix, iy


def tile_key(ix: int, iy: int) -> str:
    return f"t{int(ix):03d}_{int(iy):03d}"


def tile_bounds(ix: int, iy: int, tile_deg: float) -> dict:
    return {"key": tile_key(ix, iy), "ix": int(ix), "iy": int(iy),
            "ra_min": float(ix * tile_deg), "ra_max": float((ix + 1) * tile_deg),
            "dec_min": float(iy * tile_deg - 90.0),
            "dec_max": float(min((iy + 1) * tile_deg - 90.0, 90.0))}


def tiles_covering(ra: float, dec: float, pad_deg: float, tile_deg: float) -> list[tuple[int, int]]:
    """Indices of every tile overlapping the box (ra +- pad/cos dec, dec +- pad)."""
    n_ra = int(round(360.0 / tile_deg))
    n_dec = int(math.ceil(180.0 / tile_deg))
    dec_lo = max(float(dec) - pad_deg, -90.0)
    dec_hi = min(float(dec) + pad_deg, 90.0)
    iy_lo = int(math.floor((dec_lo + 90.0) / tile_deg))
    iy_hi = int(math.floor((dec_hi + 90.0) / tile_deg))
    iy_lo, iy_hi = max(iy_lo, 0), min(iy_hi, n_dec - 1)
    cosd = max(math.cos(math.radians(max(abs(dec_lo), abs(dec_hi)))), 1e-6)
    dra = pad_deg / cosd
    if dra >= 180.0:
        ixs = list(range(n_ra))
    else:
        ix_lo = int(math.floor((float(ra) - dra) / tile_deg))
        ix_hi = int(math.floor((float(ra) + dra) / tile_deg))
        ixs = sorted({i % n_ra for i in range(ix_lo, ix_hi + 1)})
    return [(ix, iy) for ix in ixs for iy in range(iy_lo, iy_hi + 1)]


def neighbourhood_radius_deg(cfg: dict) -> float:
    """How far from a target the screen needs catalogue rows: the farthest
    control plus the larger of its annulus and bright-source radii."""
    reach = max(float(cfg.get("annulus_out_arcsec", 1200.0)),
                float(cfg.get("bright_source_radius_arcsec", 1800.0)))
    return (float(cfg.get("control_offset_arcsec", 2700.0)) + reach) / 3600.0 + 0.02


def plan_tiles(targets_df: pd.DataFrame, tile_deg: float, pad_deg: float = 0.0) -> list[dict]:
    """Every tile a target lies in, plus (with ``pad_deg``) every tile its
    control neighbourhood reaches into.  Sorted by key; ``n_targets`` counts
    the targets whose own position is inside the tile.
    """
    tile_deg = float(tile_deg)
    wanted: dict[tuple[int, int], int] = {}
    if targets_df is not None and len(targets_df):
        ra = np.asarray(targets_df["ra"], dtype=float)
        dec = np.asarray(targets_df["dec"], dtype=float)
        ix, iy = tile_index(ra, dec, tile_deg)
        for a, b in zip(ix, iy, strict=True):
            wanted[(int(a), int(b))] = wanted.get((int(a), int(b)), 0) + 1
        if pad_deg > 0:
            for r, d in zip(ra, dec, strict=True):
                for k in tiles_covering(r, d, pad_deg, tile_deg):
                    wanted.setdefault(k, 0)
    out = []
    for (a, b) in sorted(wanted):
        t = tile_bounds(a, b, tile_deg)
        t["n_targets"] = wanted[(a, b)]
        out.append(t)
    return out


def bin_sources_into_tiles(sources_df: pd.DataFrame, tile_deg: float) -> dict[str, pd.DataFrame]:
    """Split one source table into the ``{tile_key: DataFrame}`` mapping that
    :func:`screen_targets` consumes (used by tests and by anyone with a local
    copy of the catalogue)."""
    if sources_df is None or len(sources_df) == 0:
        return {}
    ix, iy = tile_index(sources_df["ra"], sources_df["dec"], float(tile_deg))
    keys = np.array([tile_key(a, b) for a, b in zip(ix, iy, strict=True)])
    out = {}
    for k in np.unique(keys):
        out[str(k)] = sources_df.loc[keys == k].reset_index(drop=True)
    return out


def _gather_local(sources_by_tile: dict, ra: float, dec: float, radius_deg: float,
                  tile_deg: float) -> pd.DataFrame:
    parts = []
    for (a, b) in tiles_covering(ra, dec, radius_deg, tile_deg):
        df = sources_by_tile.get(tile_key(a, b))
        if df is not None and len(df):
            parts.append(df)
    if not parts:
        return pd.DataFrame({"ra": [], "dec": [], "flux_jy": []})
    df = pd.concat(parts, ignore_index=True)
    sep = angular_separation_arcsec(ra, dec, df["ra"].to_numpy(float), df["dec"].to_numpy(float))
    return df.loc[sep <= radius_deg * 3600.0].reset_index(drop=True)


# --------------------------------------------------------------------------
# The statistic
# --------------------------------------------------------------------------
def void_statistics(sources_df: pd.DataFrame, ra: float, dec: float, cfg: dict) -> dict:
    """The void statistic at one position, with its veto.

    ``sources_df`` must hold every catalogue source out to at least the
    larger of the annulus outer radius and the bright-source radius around
    (ra, dec); it may hold more.  Columns ``ra``, ``dec`` (deg) and optionally
    ``flux_jy``.
    """
    apertures = [float(r) for r in cfg.get("apertures_arcsec", DEFAULTS["apertures_arcsec"])]
    r_in = float(cfg.get("annulus_in_arcsec", 480.0))
    r_out = float(cfg.get("annulus_out_arcsec", 1200.0))
    min_density = float(cfg.get("min_annulus_density_per_deg2", 300.0))
    bright_jy = float(cfg.get("bright_source_jy", 1.0))
    bright_r = float(cfg.get("bright_source_radius_arcsec", 1800.0))
    min_lam = float(cfg.get("min_expected_count", 8.0))
    p_min = float(cfg.get("p_min", 1e-5))

    sra, sdec = _arrays(sources_df)
    sep = angular_separation_arcsec(ra, dec, sra, sdec) if sra.size else np.empty(0)
    n_annulus = int(np.count_nonzero((sep >= r_in) & (sep < r_out)))
    density = n_annulus / annulus_area_deg2(r_in, r_out)

    out: dict = {"ra": float(ra), "dec": float(dec), "n_annulus": n_annulus,
                 "annulus_density_per_deg2": float(density),
                 "bright_source_max_jy": float("nan"), "n_trials": 0,
                 "best_aperture_arcsec": float("nan"), "best_d_au": float("nan"),
                 "best_phase": float("nan"), "best_dx_arcsec": float("nan"),
                 "best_dy_arcsec": float("nan"), "best_n_obs": -1,
                 "best_lambda": float("nan"), "p_raw": float("nan"),
                 "p_trials": float("nan"), "n_sources_in_reach": int(sra.size)}
    for r in apertures:
        out[f"n_x_{int(r)}"] = int(np.count_nonzero(sep < r)) if sra.size else 0
        out[f"lambda_{int(r)}"] = float(density * aperture_area_deg2(r))

    if density < min_density:
        out["veto"] = VETO_FOOTPRINT
        return out

    if sources_df is not None and "flux_jy" in sources_df.columns and sra.size:
        flux = np.asarray(sources_df["flux_jy"], dtype=float)
        near = sep < bright_r
        if np.any(near):
            fmax = np.nanmax(np.where(near, flux, np.nan))
            out["bright_source_max_jy"] = float(fmax) if np.isfinite(fmax) else float("nan")
            if np.isfinite(fmax) and fmax > bright_jy:
                out["veto"] = VETO_BRIGHT
                return out

    usable = [r for r in apertures if density * aperture_area_deg2(r) >= min_lam]
    if not usable:
        out["veto"] = VETO_LOW_EXPECTED
        return out

    grid = centre_grid(ra, dec, cfg)
    n_trials = len(grid) * len(usable)
    out["n_trials"] = int(n_trials)
    # Only sources that can fall in any aperture around any centre matter.
    reach = max(usable) + float(np.max(np.hypot(grid["dx_arcsec"], grid["dy_arcsec"])))
    keep = sep <= reach
    kra, kdec = sra[keep], sdec[keep]
    best = None
    for g in grid.itertuples(index=False):
        if kra.size:
            gsep = np.sort(angular_separation_arcsec(g.ra, g.dec, kra, kdec))
        else:
            gsep = np.empty(0)
        for r in usable:
            n_obs = int(np.searchsorted(gsep, r, side="left"))
            lam = density * aperture_area_deg2(r)
            p = poisson_void_p(n_obs, lam)
            if best is None or p < best[0]:
                best = (p, r, g, n_obs, lam)
    p, r, g, n_obs, lam = best
    out.update({"best_aperture_arcsec": float(r), "best_d_au": float(g.d_au),
                "best_phase": float(g.phase), "best_dx_arcsec": float(g.dx_arcsec),
                "best_dy_arcsec": float(g.dy_arcsec), "best_n_obs": int(n_obs),
                "best_lambda": float(lam), "p_raw": float(p),
                "p_trials": float(min(1.0, p * n_trials))})
    out["veto"] = VETO_NONE if out["p_trials"] < p_min else VETO_NOT_SIGNIFICANT
    return out


def _control_positions(ra: float, dec: float, cfg: dict) -> list[tuple[str, float, float]]:
    off = float(cfg.get("control_offset_arcsec", 2700.0))
    out = []
    for name, dx, dy in (("+ra", off, 0.0), ("-ra", -off, 0.0),
                         ("+dec", 0.0, off), ("-dec", 0.0, -off)):
        out.append((name, *offset_position(ra, dec, dx, dy)))
    return out


def in_regions(ra, dec, regions) -> np.ndarray:
    """Membership in the union of the configured (RA-wrapping) boxes."""
    ra = np.mod(np.asarray(ra, dtype=float), 360.0)
    dec = np.asarray(dec, dtype=float)
    ok = np.zeros(ra.shape, dtype=bool)
    for reg in regions or []:
        lo, hi = float(reg["ra_min"]) % 360.0, float(reg["ra_max"]) % 360.0
        in_ra = (ra >= lo) & (ra <= hi) if lo <= hi else (ra >= lo) | (ra <= hi)
        ok |= in_ra & (dec >= float(reg["dec_min"])) & (dec <= float(reg["dec_max"]))
    return ok


def _touches_bad_tile(ra: float, dec: float, reach_deg: float, tile_deg: float,
                      bad_tiles) -> bool:
    if not bad_tiles:
        return False
    return any(tile_key(a, b) in bad_tiles
               for (a, b) in tiles_covering(ra, dec, reach_deg, tile_deg))


def _masked_record(ra: float, dec: float, cfg: dict) -> dict:
    """The record of a position whose catalogue reach includes a tile that
    failed or was truncated: the counts there are not the sky's, so the
    position is masked before any statistic is computed.  A failed tile next
    to a target would otherwise manufacture a void."""
    st = void_statistics(pd.DataFrame({"ra": [], "dec": [], "flux_jy": []}), ra, dec, cfg)
    st["veto"] = VETO_FOOTPRINT
    st["masked_by_bad_tile"] = True
    return st


def screen_targets(targets_df: pd.DataFrame, sources_by_tile: dict, cfg: dict,
                   bad_tiles=None) -> tuple[pd.DataFrame, dict]:
    """The statistic at every target and its four controls.

    ``targets_df`` needs ``ra``, ``dec`` and, if present, ``source_id``,
    ``parallax``, ``is_etz``.  ``bad_tiles`` is the set of tile keys whose
    query failed or hit the row cap: any position whose catalogue reach
    touches one is ``outside_footprint_or_masked``, never a void.
    Returns (voids_df, counters).
    """
    tile_deg = float(cfg.get("tile_deg", 2.0))
    reach_deg = max(float(cfg.get("annulus_out_arcsec", 1200.0)),
                    float(cfg.get("bright_source_radius_arcsec", 1800.0))) / 3600.0
    local_deg = neighbourhood_radius_deg(cfg)
    p_min = float(cfg.get("p_min", 1e-5))
    apertures = [float(r) for r in cfg.get("apertures_arcsec", DEFAULTS["apertures_arcsec"])]
    bad_tiles = set(bad_tiles or ())

    counters = {"n_targets": int(len(targets_df)), "n_etz": 0,
                "n_targets_in_footprint": 0, "n_etz_in_footprint": 0,
                **{v: 0 for v in VETO_ORDER}, "n_masked_by_bad_tile": 0,
                "n_candidates": 0, "n_etz_candidates": 0,
                "n_control_positions": 0, "n_control_evaluated": 0, "n_control_fired": 0,
                "n_controls_masked_by_bad_tile": 0,
                "control_false_void_rate": float("nan"),
                "n_trials_per_position": 0, "apertures_arcsec": apertures}
    rows = []
    t0 = _time.monotonic()
    for i, t in enumerate(targets_df.itertuples(index=False)):
        ra, dec = float(t.ra), float(t.dec)
        local = _gather_local(sources_by_tile, ra, dec, local_deg, tile_deg)
        lra, ldec = _arrays(local)
        if _touches_bad_tile(ra, dec, reach_deg, tile_deg, bad_tiles):
            st = _masked_record(ra, dec, cfg)
            counters["n_masked_by_bad_tile"] += 1
        else:
            sep = angular_separation_arcsec(ra, dec, lra, ldec) if lra.size else np.empty(0)
            st = void_statistics(local.loc[sep <= reach_deg * 3600.0 + 1.0], ra, dec, cfg)
            st["masked_by_bad_tile"] = False
        is_etz = bool(getattr(t, "is_etz", False))
        rec = {"source_id": getattr(t, "source_id", i), "ra": ra, "dec": dec,
               "parallax_mas": float(getattr(t, "parallax", float("nan"))),
               "is_etz": is_etz}
        rec.update({k: v for k, v in st.items() if k not in ("ra", "dec")})
        counters["n_trials_per_position"] = max(counters["n_trials_per_position"], st["n_trials"])
        if is_etz:
            counters["n_etz"] += 1
        veto = st["veto"]
        if veto != VETO_FOOTPRINT:
            counters["n_targets_in_footprint"] += 1
            if is_etz:
                counters["n_etz_in_footprint"] += 1
        if veto in counters:
            counters[veto] += 1
        rec["is_candidate"] = bool(veto == VETO_NONE)
        if rec["is_candidate"]:
            counters["n_candidates"] += 1
            if is_etz:
                counters["n_etz_candidates"] += 1

        # Controls: the same statistic 45' away in four directions.
        n_eval, n_fired, p_ctrl = 0, 0, []
        for name, cra, cdec in _control_positions(ra, dec, cfg):
            counters["n_control_positions"] += 1
            if _touches_bad_tile(cra, cdec, reach_deg, tile_deg, bad_tiles):
                cs = _masked_record(cra, cdec, cfg)
                counters["n_controls_masked_by_bad_tile"] += 1
            else:
                csep = (angular_separation_arcsec(cra, cdec, lra, ldec) if lra.size
                        else np.empty(0))
                cs = void_statistics(local.loc[csep <= reach_deg * 3600.0 + 1.0], cra, cdec, cfg)
            rec[f"control_{name}_veto"] = cs["veto"]
            rec[f"control_{name}_p_trials"] = cs["p_trials"]
            if cs["veto"] in (VETO_NONE, VETO_NOT_SIGNIFICANT):
                n_eval += 1
                p_ctrl.append(cs["p_trials"])
                if cs["veto"] == VETO_NONE:
                    n_fired += 1
        counters["n_control_evaluated"] += n_eval
        counters["n_control_fired"] += n_fired
        rec["n_control_evaluated"] = n_eval
        rec["n_control_fired"] = n_fired
        rec["control_min_p_trials"] = float(min(p_ctrl)) if p_ctrl else float("nan")
        rows.append(rec)
        if (i + 1) % 2000 == 0:
            print(f"[{CHANNEL}] screened {i + 1}/{len(targets_df)} targets "
                  f"({_time.monotonic() - t0:.0f} s)")

    voids = pd.DataFrame(rows)
    if counters["n_control_evaluated"]:
        counters["control_false_void_rate"] = (
            counters["n_control_fired"] / counters["n_control_evaluated"])
    counters["p_min"] = p_min
    # The empirical null next to the Poisson one: how often a control reaches
    # each target's p_trials (rank of the target among evaluated controls).
    if len(voids) and counters["n_control_evaluated"]:
        ctrl_cols = [c for c in voids.columns if c.endswith("_p_trials") and c.startswith("control_")]
        allc = voids[ctrl_cols].to_numpy(float).ravel()
        allc = np.sort(allc[np.isfinite(allc)])
        pt = voids["p_trials"].to_numpy(float)
        rank = np.searchsorted(allc, pt, side="right")
        emp = np.where(np.isfinite(pt), (rank + 1) / (allc.size + 1), np.nan)
        voids["p_empirical_control"] = emp
    return voids, counters


# --------------------------------------------------------------------------
# Runner-side acquisition: discovery, tiles, targets
# --------------------------------------------------------------------------
def _quote(name: str) -> str:
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else f'"{name}"'


def _quote_table(name: str) -> str:
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name) else f'"{name}"'


_RA_NAMES = ("raj2000", "ra_icrs", "ra", "_raj2000", "radeg", "ra_deg")
_DEC_NAMES = ("dej2000", "de_icrs", "dec", "de", "_dej2000", "dedeg", "dec_deg")
_FLUX_NAMES = ("stotal", "total_flux", "ftotal", "totalflux", "sint", "s_total", "flux_total",
               "fint", "flux")


def pick_columns(columns_df: pd.DataFrame) -> dict:
    """Choose RA / Dec / total-flux columns from a TAP_SCHEMA.columns result.

    UCD first (``pos.eq.ra;meta.main`` etc.), then a ranked name list; every
    choice is returned with the reason so the ledger shows what was used.
    Nothing is chosen that was not in ``columns_df``.
    """
    if columns_df is None or len(columns_df) == 0:
        return {"ra_col": None, "dec_col": None, "flux_col": None, "flux_unit": None,
                "reason": "no columns seen"}
    df = columns_df.copy()
    df.columns = [c.lower() for c in df.columns]
    names = [str(x) for x in df["column_name"]]
    ucds = [str(x).lower() if x is not None else "" for x in df.get("ucd", [""] * len(df))]
    units = [str(x) if x is not None else "" for x in df.get("unit", [""] * len(df))]
    lower = [n.lower() for n in names]

    def by_ucd(prefix, main=True):
        hits = [i for i, u in enumerate(ucds) if u.startswith(prefix)]
        if main:
            m = [i for i in hits if "meta.main" in ucds[i]]
            if m:
                return m[0]
        return hits[0] if hits else None

    def by_name(cands):
        for c in cands:
            if c in lower:
                return lower.index(c)
        return None

    reasons = []
    i_ra = by_ucd("pos.eq.ra")
    if i_ra is not None:
        reasons.append(f"ra by ucd {ucds[i_ra]}")
    else:
        i_ra = by_name(_RA_NAMES)
        reasons.append("ra by name" if i_ra is not None else "ra NOT FOUND")
    i_dec = by_ucd("pos.eq.dec")
    if i_dec is not None:
        reasons.append(f"dec by ucd {ucds[i_dec]}")
    else:
        i_dec = by_name(_DEC_NAMES)
        reasons.append("dec by name" if i_dec is not None else "dec NOT FOUND")
    # Flux: integrated / total flux density, not the peak.
    fl = [i for i, u in enumerate(ucds) if "phot.flux" in u and "stat.error" not in u
          and "peak" not in lower[i] and not lower[i].startswith("e_")]
    tot = [i for i in fl if "tot" in lower[i] or "int" in lower[i]]
    if tot:
        i_flux = tot[0]
        reasons.append(f"flux by ucd {ucds[i_flux]} ({names[i_flux]})")
    elif fl:
        i_flux = fl[0]
        reasons.append(f"flux by ucd {ucds[i_flux]} ({names[i_flux]})")
    else:
        i_flux = by_name(_FLUX_NAMES)
        reasons.append("flux by name" if i_flux is not None else "flux NOT FOUND")
    return {"ra_col": names[i_ra] if i_ra is not None else None,
            "dec_col": names[i_dec] if i_dec is not None else None,
            "flux_col": names[i_flux] if i_flux is not None else None,
            "flux_unit": units[i_flux] if i_flux is not None else None,
            "reason": "; ".join(reasons)}


def flux_to_jy_factor(unit: str | None) -> tuple[float, str]:
    """Multiplier that brings the catalogue flux column to Jy, and how it was decided."""
    u = (unit or "").strip()
    ul = u.lower()
    if ul in ("mjy", "millijy", "milli-jansky"):
        return 1e-3, f"unit '{u}' -> 1e-3"
    if ul in ("jy", "jansky"):
        return 1.0, f"unit '{u}' -> 1"
    if ul in ("ujy", "microjy", "µjy", "μjy"):
        return 1e-6, f"unit '{u}' -> 1e-6"
    return 1e-3, f"unit '{u}' not recognised; ASSUMED mJy (LoTSS native)"


def discover_lotss(cfg: dict) -> dict:
    """Find the LoTSS DR2 source table and its columns at run time.

    VizieR first (catalogue ``J/A+A/659/A1``), then ASTRON.  Returns a dict
    with ``status`` (``DISCOVERED`` / ``NOT_DISCOVERED``), the chosen
    ``service`` / ``table`` / columns, ``degraded`` (True when the fallback
    route had to be used or the flux column is missing) and the full ledger.
    """
    lc = cfg.get("lotss", DEFAULTS["lotss"])
    retries = int(lc.get("tap_retries", 3))
    out: dict = {"status": "NOT_DISCOVERED", "service": None, "table": None,
                 "ra_col": None, "dec_col": None, "flux_col": None, "flux_unit": None,
                 "flux_to_jy": None, "degraded": False, "degraded_reasons": [],
                 "ledger": [], "tables_seen": []}
    routes = [("vizier", lc.get("vizier_tap", VIZIER_TAP),
               f"table_name LIKE '%{lc.get('vizier_catalogue_hint', '659/A1')}%'"),
              ("astron", lc.get("astron_tap", ASTRON_TAP),
               f"table_name LIKE '%{lc.get('astron_table_hint', 'lotss')}%'")]
    for route, url, like in routes:
        q = f"SELECT table_name, description FROM TAP_SCHEMA.tables WHERE {like}"
        r = run_tap(url, q, label=f"lotss_tables@{route}", retries=retries, async_first=False)
        out["ledger"].append(r.to_ledger())
        if r.status != "OK" or r.data is None or not len(r.data):
            continue
        tdf = r.data.copy()
        tdf.columns = [c.lower() for c in tdf.columns]
        names = [str(x) for x in tdf["table_name"]]
        out["tables_seen"].extend({"route": route, "table": n} for n in names)
        # Prefer a main / source table over anything looking like a mosaic or
        # a Gaussian-component list.
        def score(n: str) -> tuple:
            nl = n.lower()
            return (("gaus" in nl) or ("mosaic" in nl) or ("component" in nl),
                    -(("main" in nl) + ("source" in nl) + ("dr2" in nl)), len(nl))
        table = sorted(names, key=score)[0]
        cq = ("SELECT column_name, ucd, unit, description FROM TAP_SCHEMA.columns "
              f"WHERE table_name = '{table}'")
        c = run_tap(url, cq, label=f"lotss_columns@{route}", retries=retries, async_first=False)
        out["ledger"].append(c.to_ledger())
        if c.status != "OK" or c.data is None:
            continue
        cols = pick_columns(c.data)
        out["columns_seen"] = [str(x) for x in c.data[c.data.columns[0]]][:400]
        out["column_choice_reason"] = cols["reason"]
        if not cols["ra_col"] or not cols["dec_col"]:
            out["degraded_reasons"].append(f"{route}: RA/Dec columns not identified in {table}")
            continue
        out.update({"status": "DISCOVERED", "service": url, "route": route, "table": table,
                    "ra_col": cols["ra_col"], "dec_col": cols["dec_col"],
                    "flux_col": cols["flux_col"], "flux_unit": cols["flux_unit"]})
        if cols["flux_col"]:
            fac, why = flux_to_jy_factor(cols["flux_unit"])
            out["flux_to_jy"], out["flux_unit_decision"] = fac, why
            if "ASSUMED" in why:
                out["degraded_reasons"].append(f"flux unit assumed mJy for {cols['flux_col']}")
        else:
            out["degraded_reasons"].append("no flux column: bright-source veto disabled")
        if route != "vizier":
            out["degraded_reasons"].append(f"LoTSS reached through fallback route {route}")
        break
    out["degraded"] = bool(out["degraded_reasons"]) and out["status"] == "DISCOVERED"
    return out


def build_tile_query(disc: dict, tile: dict, max_rows: int) -> str:
    cols = [_quote(disc["ra_col"]), _quote(disc["dec_col"])]
    if disc.get("flux_col"):
        cols.append(_quote(disc["flux_col"]))
    ra, dec = _quote(disc["ra_col"]), _quote(disc["dec_col"])
    return (f"SELECT TOP {int(max_rows)} {', '.join(cols)} FROM {_quote_table(disc['table'])} "
            f"WHERE {ra} >= {tile['ra_min']} AND {ra} < {tile['ra_max']} "
            f"AND {dec} >= {tile['dec_min']} AND {dec} < {tile['dec_max']}")


def normalise_sources(df: pd.DataFrame, disc: dict) -> pd.DataFrame:
    """Rename the discovered columns to ``ra``, ``dec``, ``flux_jy``."""
    out = pd.DataFrame({"ra": pd.to_numeric(df[disc["ra_col"]], errors="coerce"),
                        "dec": pd.to_numeric(df[disc["dec_col"]], errors="coerce")})
    if disc.get("flux_col") and disc["flux_col"] in df.columns:
        fac = float(disc.get("flux_to_jy") or 1e-3)
        out["flux_jy"] = pd.to_numeric(df[disc["flux_col"]], errors="coerce") * fac
    else:
        out["flux_jy"] = np.nan
    return out.dropna(subset=["ra", "dec"]).reset_index(drop=True)


def make_lotss_fetcher(disc: dict, cfg: dict):
    """A ``fetcher(tile) -> DataFrame(ra, dec, flux_jy)`` over the discovered table.

    Raises on QUERY_FAILED so the tile loop records it; returns an empty frame
    for QUERY_RETURNED_ZERO_ROWS.  ``TOP max_rows`` makes the row cap visible
    in the query text, and a result that hits it is flagged as truncated by the
    tile loop rather than accepted as complete.
    """
    lc = cfg.get("lotss", DEFAULTS["lotss"])
    max_rows = int(lc.get("max_rows_per_tile", 40000))
    retries = int(lc.get("tap_retries", 3))

    def fetch(tile: dict) -> pd.DataFrame:
        q = build_tile_query(disc, tile, max_rows)
        r: QueryResult = run_tap(disc["service"], q, label=f"lotss_tile@{tile['key']}",
                                 retries=retries, async_first=False)
        if r.status == "QUERY_FAILED":
            raise RuntimeError(r.error or "QUERY_FAILED")
        if r.data is None or not len(r.data):
            return pd.DataFrame({"ra": [], "dec": [], "flux_jy": []})
        return normalise_sources(r.data, disc)

    fetch.max_rows = max_rows                          # type: ignore[attr-defined]
    return fetch


def fetch_gaia_targets(cfg: dict) -> pd.DataFrame:
    """One Gaia DR3 query: nearby, well-measured stars inside the coarse LoTSS sky."""
    tc = cfg.get("targets", DEFAULTS["targets"])
    clauses = []
    for reg in tc.get("regions", []):
        lo, hi = float(reg["ra_min"]) % 360.0, float(reg["ra_max"]) % 360.0
        ra_c = (f"(ra >= {lo} AND ra <= {hi})" if lo <= hi
                else f"(ra >= {lo} OR ra <= {hi})")
        clauses.append(f"({ra_c} AND dec >= {float(reg['dec_min'])} AND dec <= {float(reg['dec_max'])})")
    region = " OR ".join(clauses) if clauses else "1=1"
    q = (f"SELECT TOP {int(tc.get('max_rows', 200000))} source_id, ra, dec, parallax, "
         f"parallax_over_error, pmra, pmdec, ruwe, phot_g_mean_mag, bp_rp "
         f"FROM gaiadr3.gaia_source WHERE parallax > {float(tc.get('parallax_min_mas', 20.0))} "
         f"AND parallax_over_error > {float(tc.get('parallax_over_error_min', 10.0))} "
         f"AND ({region})")
    from astroquery.gaia import Gaia
    job = Gaia.launch_job_async(q)
    df = job.get_results().to_pandas()
    df.columns = [c.lower() for c in df.columns]
    df.attrs["query"] = q
    return df


def prepare_targets(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Region pre-cut, proper-motion propagation to the LoTSS epoch, ETZ flag."""
    tc = cfg.get("targets", DEFAULTS["targets"])
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "ra" not in df or "dec" not in df:
        raise ValueError("targets need ra and dec columns")
    df["ra_gaia"], df["dec_gaia"] = df["ra"].astype(float), df["dec"].astype(float)
    if "pmra" in df and "pmdec" in df:
        from ..vigil.acquire import propagate_pm
        ra_p, dec_p = propagate_pm(df["ra_gaia"], df["dec_gaia"], df["pmra"], df["pmdec"],
                                   GAIA_EPOCH, float(tc.get("propagate_pm_to_epoch",
                                                            LOTSS_DR2_MID_EPOCH)))
        df["ra"], df["dec"] = np.mod(ra_p, 360.0), dec_p
    keep = in_regions(df["ra"], df["dec"], tc.get("regions", []))
    df = df.loc[keep].reset_index(drop=True)
    if len(df):
        df["ecl_lat_deg"] = ecliptic_latitude_deg(df["ra"], df["dec"])
        df["is_etz"] = np.abs(df["ecl_lat_deg"]) < float(tc.get("etz_ecliptic_lat_deg", 0.264))
    else:
        df["ecl_lat_deg"] = pd.Series(dtype=float)
        df["is_etz"] = pd.Series(dtype=bool)
    if "parallax" in df:
        df = df.sort_values("parallax", ascending=False).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Stage orchestration
# --------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
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
    path.write_text(json.dumps(obj, indent=2, default=default))


def acquire_tiles(tiles: list[dict], fetcher, tiles_dir: Path, max_rows: int | None = None,
                  ) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Fetch (or reload from checkpoint) every planned tile; never stop on one."""
    tiles_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, pd.DataFrame] = {}
    ledger: list[dict] = []
    t_start = _time.monotonic()
    for k, tile in enumerate(tiles):
        path = tiles_dir / f"{tile['key']}.parquet"
        entry = {"key": tile["key"], "ra_min": tile["ra_min"], "ra_max": tile["ra_max"],
                 "dec_min": tile["dec_min"], "dec_max": tile["dec_max"],
                 "n_targets": tile.get("n_targets", 0), "from_checkpoint": False,
                 "n_rows": 0, "status": None, "error": "", "elapsed_s": 0.0,
                 "truncated": False}
        if path.exists():
            try:
                df = pd.read_parquet(path)
                entry.update({"from_checkpoint": True, "n_rows": int(len(df)),
                              "status": "QUERY_OK" if len(df) else "QUERY_RETURNED_ZERO_ROWS"})
                sources[tile["key"]] = df
                ledger.append(entry)
                continue
            except Exception as exc:                          # noqa: BLE001
                entry["error"] = f"checkpoint unreadable, refetching: {exc!r}"
        t0 = _time.monotonic()
        try:
            df = fetcher(tile)
            df = pd.DataFrame(df)
            for c in ("ra", "dec"):
                if c not in df.columns:
                    raise ValueError(f"fetcher returned no '{c}' column")
            if "flux_jy" not in df.columns:
                df["flux_jy"] = np.nan
            entry["n_rows"] = int(len(df))
            entry["status"] = "QUERY_OK" if len(df) else "QUERY_RETURNED_ZERO_ROWS"
            cap = max_rows if max_rows is not None else getattr(fetcher, "max_rows", None)
            if cap is not None and len(df) >= int(cap):
                entry["truncated"] = True
                entry["status"] = "QUERY_TRUNCATED"
            df.to_parquet(path, index=False)
            sources[tile["key"]] = df
        except Exception as exc:                              # noqa: BLE001
            entry["status"] = "QUERY_FAILED"
            entry["error"] = repr(exc)[:500]
            print(f"[{CHANNEL}] tile {tile['key']} failed: {entry['error']}")
        entry["elapsed_s"] = round(_time.monotonic() - t0, 2)
        ledger.append(entry)
        if (k + 1) % 50 == 0:
            n_ok = sum(1 for e in ledger if e["status"] in ("QUERY_OK", "QUERY_TRUNCATED"))
            print(f"[{CHANNEL}] tiles {k + 1}/{len(tiles)}: {n_ok} with rows "
                  f"({_time.monotonic() - t_start:.0f} s)")
    return sources, ledger


def _ledger_counts(ledger: list[dict]) -> dict:
    out = {"n_tiles": len(ledger)}
    for s in ("QUERY_OK", "QUERY_RETURNED_ZERO_ROWS", "QUERY_FAILED", "QUERY_TRUNCATED"):
        out[s] = sum(1 for e in ledger if e.get("status") == s)
    out["n_from_checkpoint"] = sum(1 for e in ledger if e.get("from_checkpoint"))
    out["n_rows_total"] = int(sum(int(e.get("n_rows", 0)) for e in ledger))
    return out


def run_radio_stage(cfg: dict | None, out_dir, *, lotss_fetcher=None, target_fetcher=None,
                    max_targets: int | None = None) -> dict:
    """The whole radio stage: targets -> tiles -> statistic -> files -> summary.

    ``cfg`` is the full merged mapping (``cfg["radio"]`` is used when present,
    else ``config/baffle_radio.yaml``).  ``lotss_fetcher(tile) -> DataFrame``
    and ``target_fetcher(rcfg) -> DataFrame`` replace the archive calls (tests
    must inject both).  Writes ``voids.csv``, ``candidates.csv``,
    ``tiles_ledger.json``, ``summary.json`` under ``out_dir`` and returns the
    summary.
    """
    rcfg = _radio_cfg(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = _now()
    summary: dict = {"channel": CHANNEL, "stage": "run", "started_utc": started,
                     "config": {k: v for k, v in rcfg.items() if k != "probe"},
                     "degraded_reasons": [], "targets": {}, "discovery": None, "tiles": {},
                     "funnel": {}, "verdict": None, "verdict_code": None}

    # Targets ----------------------------------------------------------------
    try:
        raw = (target_fetcher or fetch_gaia_targets)(rcfg)
        targets = prepare_targets(pd.DataFrame(raw), rcfg)
        summary["targets"] = {"status": "OK" if len(targets) else "QUERY_RETURNED_ZERO_ROWS",
                              "n_raw": int(len(raw)), "n_after_region_cut": int(len(targets)),
                              "query": str(getattr(raw, "attrs", {}).get("query", ""))[:2000]}
    except Exception as exc:                                  # noqa: BLE001
        targets = pd.DataFrame({"ra": [], "dec": []})
        summary["targets"] = {"status": "QUERY_FAILED", "error": repr(exc)[:500],
                              "n_raw": 0, "n_after_region_cut": 0}
        print(f"[{CHANNEL}] target fetch failed: {exc!r}")
    if max_targets is not None and int(max_targets) > 0:
        targets = targets.head(int(max_targets)).reset_index(drop=True)
        summary["targets"]["max_targets"] = int(max_targets)
    summary["targets"]["n_used"] = int(len(targets))
    summary["targets"]["n_etz"] = int(targets["is_etz"].sum()) if "is_etz" in targets else 0

    # Catalogue ------------------------------------------------------------------
    fetcher = lotss_fetcher
    if fetcher is None:
        disc = discover_lotss(rcfg)
        summary["discovery"] = disc
        summary["degraded_reasons"].extend(disc.get("degraded_reasons", []))
        if disc["status"] == "DISCOVERED":
            fetcher = make_lotss_fetcher(disc, rcfg)
    else:
        summary["discovery"] = {"status": "INJECTED_FETCHER"}

    tile_deg = float(rcfg.get("tile_deg", 2.0))
    tiles = plan_tiles(targets, tile_deg, pad_deg=neighbourhood_radius_deg(rcfg)) if len(targets) else []
    sources: dict[str, pd.DataFrame] = {}
    ledger: list[dict] = []
    if fetcher is not None and tiles:
        sources, ledger = acquire_tiles(tiles, fetcher, out / "tiles")
    summary["tiles"] = {"tile_deg": tile_deg, "n_planned": len(tiles), **_ledger_counts(ledger)}
    _write_json(out / "tiles_ledger.json", {"written_utc": _now(), "tile_deg": tile_deg,
                                            "counts": summary["tiles"], "tiles": ledger})
    n_failed = summary["tiles"].get("QUERY_FAILED", 0)
    if ledger and n_failed:
        summary["degraded_reasons"].append(f"{n_failed}/{len(ledger)} tiles QUERY_FAILED")
    if summary["tiles"].get("QUERY_TRUNCATED", 0):
        summary["degraded_reasons"].append(
            f"{summary['tiles']['QUERY_TRUNCATED']} tiles hit the row cap (TOP)")

    # Statistic --------------------------------------------------------------------
    n_rows = summary["tiles"].get("n_rows_total", 0)
    bad_tiles = {e["key"] for e in ledger if e.get("status") in ("QUERY_FAILED", "QUERY_TRUNCATED")}
    if len(targets) and n_rows:
        voids, counters = screen_targets(targets, sources, rcfg, bad_tiles=bad_tiles)
    else:
        voids, counters = pd.DataFrame(), {"n_targets": int(len(targets)), "n_candidates": 0,
                                           "n_targets_in_footprint": 0, "n_etz": 0,
                                           **{v: 0 for v in VETO_ORDER},
                                           "control_false_void_rate": float("nan")}
    voids.to_csv(out / "voids.csv", index=False)
    cands = voids.loc[voids["is_candidate"]] if len(voids) else pd.DataFrame()
    cands.to_csv(out / "candidates.csv", index=False)
    summary["funnel"] = counters
    summary["n_targets_in_footprint"] = counters.get("n_targets_in_footprint", 0)
    summary["n_etz"] = counters.get("n_etz", summary["targets"]["n_etz"])
    summary["n_etz_in_footprint"] = counters.get("n_etz_in_footprint", 0)
    summary["control_false_void_rate"] = counters.get("control_false_void_rate")
    summary["n_trials_per_position"] = counters.get("n_trials_per_position", 0)
    summary["n_candidates"] = int(counters.get("n_candidates", 0))
    if len(cands):
        summary["candidates"] = cands.head(50).to_dict(orient="records")

    # Verdict --------------------------------------------------------------------
    if n_rows == 0 or not len(targets):
        code = VERDICT_NO_DATA
        why = ("no LoTSS rows reached" if len(targets) else
               f"no targets ({summary['targets'].get('status')})")
        text = f"{code}: {why}"
    elif summary["n_candidates"]:
        code = VERDICT_CANDIDATES
        text = (f"{code}: {summary['n_candidates']} of {summary['n_targets_in_footprint']} "
                f"in-footprint targets; control false-void rate "
                f"{summary['control_false_void_rate']}")
    elif summary["degraded_reasons"]:
        code = VERDICT_DEGRADED
        text = f"{code} ({'; '.join(summary['degraded_reasons'])})"
    else:
        code = VERDICT_NO_SURVIVOR
        text = (f"{code}: {summary['n_targets_in_footprint']} in-footprint targets, "
                f"none below p_trials < {rcfg.get('p_min')} with lambda >= "
                f"{rcfg.get('min_expected_count')}")
    summary["verdict_code"] = code
    summary["verdict"] = text
    summary["finished_utc"] = _now()
    _write_json(out / "summary.json", summary)
    print(f"[{CHANNEL}] " + json.dumps({"verdict": text, "targets": summary["targets"].get("n_used"),
                                        "tiles": summary["tiles"], "funnel": {
                                            k: v for k, v in counters.items()
                                            if isinstance(v, (int, float))}},
                                       default=str))
    return summary


def run_probe(cfg: dict | None, out_dir) -> dict:
    """TAP_SCHEMA discovery and one tiny box query; writes ``probe.json``."""
    rcfg = _radio_cfg(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec: dict = {"channel": CHANNEL, "stage": "probe", "started_utc": _now()}
    disc = discover_lotss(rcfg)
    rec["discovery"] = disc
    if disc["status"] == "DISCOVERED":
        pc = rcfg.get("probe", DEFAULTS["probe"])
        hw = float(pc.get("half_width_deg", 0.05))
        tile = {"key": "probe", "ra_min": float(pc["ra"]) - hw, "ra_max": float(pc["ra"]) + hw,
                "dec_min": float(pc["dec"]) - hw, "dec_max": float(pc["dec"]) + hw}
        q = build_tile_query(disc, tile, 5000)
        r = run_tap(disc["service"], q, label="lotss_probe_box",
                    retries=int(rcfg.get("lotss", {}).get("tap_retries", 2)), async_first=False)
        rec["probe_query"] = r.to_ledger()
        if r.status == "OK" and r.data is not None:
            df = normalise_sources(r.data, disc)
            area = (2 * hw) ** 2 * math.cos(math.radians(float(pc["dec"])))
            rec["probe_box"] = {"n_rows": int(len(df)), "area_deg2": area,
                                "density_per_deg2": float(len(df) / area) if area else None,
                                "flux_jy_max": float(np.nanmax(df["flux_jy"])) if len(df) and
                                df["flux_jy"].notna().any() else None,
                                "head": df.head(5).to_dict(orient="records")}
        rec["verdict"] = ("PROBE_OK" if r.status == "OK" else
                          f"PROBE_QUERY_{r.status}")
    else:
        rec["verdict"] = "NO_DATA_REACHED: LoTSS table not discovered on any route"
    rec["finished_utc"] = _now()
    _write_json(out / "probe.json", rec)
    print(f"[{CHANNEL}] probe: {rec['verdict']} table={disc.get('table')} "
          f"cols={disc.get('ra_col')},{disc.get('dec_col')},{disc.get('flux_col')}")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m seti.baffle.radio",
                                 description="BAFFLE radio: LoTSS DR2 voids around nearby stars")
    ap.add_argument("--stage", choices=("probe", "run"), default="run")
    ap.add_argument("--max-targets", type=int, default=0, help="0 = all")
    ap.add_argument("--tile-deg", type=float, default=None)
    ap.add_argument("--out", default="results/baffle_radio")
    a = ap.parse_args(argv)
    rcfg = load_radio_config()
    if a.tile_deg:
        rcfg["tile_deg"] = float(a.tile_deg)
    cfg = {"radio": rcfg}
    if a.stage == "probe":
        rec = run_probe(cfg, a.out)
        return 0 if not str(rec["verdict"]).startswith("NO_DATA") else 1
    summary = run_radio_stage(cfg, a.out, max_targets=a.max_targets or None)
    return 0 if summary["verdict_code"] != VERDICT_NO_DATA else 1


if __name__ == "__main__":
    sys.exit(main())
