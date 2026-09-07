"""The laboratory's velocity through the Galactic rest frame on a given date.

Galactic Cartesian frame: x toward the Galactic centre, y along Galactic
rotation, z toward the north Galactic pole.  v_lab = v_LSR + v_pec + v_Earth.
The Earth's orbital velocity follows McCabe (2014, arXiv:1312.1355) with the
epsilon vectors of the ecliptic in Galactic coordinates and phase zero at the
March equinox; the ~0.5 km/s eccentricity correction is ignored (far below the
halo-parameter uncertainties this module feeds).
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np

V_PEC_KMS = np.array([11.10, 12.24, 7.25])          # Schoenrich, Binney & Dearborn 2010
U_EARTH_KMS = 29.79
EPS1 = np.array([0.9940, 0.1095, 0.0031])
EPS2 = np.array([-0.0517, 0.4945, -0.8677])
TROPICAL_YEAR_DAYS = 365.24219
# J2000 March equinox, 2000-03-20 07:35 UTC; later equinoxes recur every tropical year.
EQUINOX_2000 = dt.datetime(2000, 3, 20, 7, 35, tzinfo=dt.timezone.utc)


def to_datetime(t) -> dt.datetime:
    if isinstance(t, dt.datetime):
        return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
    if isinstance(t, dt.date):
        return dt.datetime(t.year, t.month, t.day, 12, tzinfo=dt.timezone.utc)
    if isinstance(t, str):
        if len(t) == 10:                       # date only: noon UTC, as for dt.date
            return to_datetime(dt.date.fromisoformat(t))
        d = dt.datetime.fromisoformat(t)
        return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
    raise TypeError(f"unsupported time {t!r}")


def orbital_phase(t) -> float:
    """Ecliptic longitude-like phase lambda(t) in radians, zero at the March equinox."""
    d = (to_datetime(t) - EQUINOX_2000).total_seconds() / 86400.0
    return 2.0 * math.pi * (d / TROPICAL_YEAR_DAYS % 1.0)


def earth_velocity_kms(t) -> np.ndarray:
    lam = orbital_phase(t)
    return U_EARTH_KMS * (EPS1 * math.cos(lam) + EPS2 * math.sin(lam))


def sun_velocity_kms(v0_kms: float = 238.0, v_pec: np.ndarray | None = None) -> np.ndarray:
    vp = V_PEC_KMS if v_pec is None else np.asarray(v_pec, dtype=float)
    return np.array([0.0, v0_kms, 0.0]) + vp


def v_lab_kms(t, v0_kms: float = 238.0, v_pec: np.ndarray | None = None) -> np.ndarray:
    """Lab velocity vector in the Galactic rest frame on date ``t``."""
    return sun_velocity_kms(v0_kms, v_pec) + earth_velocity_kms(t)


def v_lab_speed_kms(t, v0_kms: float = 238.0, v_pec: np.ndarray | None = None) -> float:
    return float(np.linalg.norm(v_lab_kms(t, v0_kms, v_pec)))


def year_grid(year: int, n: int = 366) -> list[dt.datetime]:
    """``n`` instants spanning calendar year ``year``."""
    start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
    return [start + dt.timedelta(days=365.0 * i / n) for i in range(n)]


def date_grid(start, end, step_days: float = 1.0) -> list[dt.datetime]:
    a, b = to_datetime(start), to_datetime(end)
    n = int((b - a).total_seconds() / 86400.0 / step_days) + 1
    return [a + dt.timedelta(days=step_days * i) for i in range(n)]


def extremal_dates(year: int, v0_kms: float = 238.0):
    """Dates of maximum and minimum lab speed in ``year`` (1-hour resolution)."""
    grid = year_grid(year, 366 * 24)
    speeds = np.array([v_lab_speed_kms(t, v0_kms) for t in grid])
    return grid[int(np.argmax(speeds))], grid[int(np.argmin(speeds))]


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def galactic_lb_deg(v: np.ndarray) -> tuple[float, float]:
    """Galactic longitude/latitude (deg) of the direction of vector ``v``."""
    x, y, z = unit(np.asarray(v, dtype=float))
    lon = math.degrees(math.atan2(y, x)) % 360.0
    lat = math.degrees(math.asin(max(-1.0, min(1.0, z))))
    return lon, lat


def stream_peak_date(mean_xyz, year: int, v0_kms: float = 238.0, n: int = 366 * 24):
    """Date on which a component with Galactic-frame bulk velocity ``mean_xyz``
    is fastest in the lab frame (its lab-frame speed |u - v_lab(t)| peaks).

    Returns (peak_datetime, trough_datetime, v_peak, v_trough)."""
    u = np.asarray(mean_xyz, dtype=float)
    grid = year_grid(year, n)
    sp = np.array([np.linalg.norm(u - v_lab_kms(t, v0_kms)) for t in grid])
    i, j = int(np.argmax(sp)), int(np.argmin(sp))
    return grid[i], grid[j], float(sp[i]), float(sp[j])
