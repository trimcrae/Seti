"""SEXTANT ephemerides: JPL Horizons, JPL SBDB, and a full-force propagator.

The residual chain in :mod:`seti.sextant.residuals` needs, for every Gaia
observation, the **barycentric ICRF state of the target at the epoch** (au,
au/day) --- it does the light time, deflection and aberration itself, and it
refuses a two-body prediction because planetary perturbations reach arcseconds
over a six-year arc.  This module supplies that state by three routes, and says
which one produced each number:

``horizons``
    JPL Horizons ``VECTORS`` with ``CENTER='500@0'`` (the solar-system
    barycentre), ``REF_PLANE='FRAME'`` (ICRF), asked at **one epoch per Gaia
    transit** and expanded to each CCD crossing with a Taylor series.  The nine
    CCD epochs of a transit span ~40 s; the expansion's neglected term over that
    interval is below a metre, and it is the same expansion the light-time
    iteration uses (``tau`` ~ 10 min), so a single Horizons call per object is
    exact to well under a microarcsecond.  Horizons integrates JPL's current
    small-body solution with DE441 and the sixteen most massive asteroids ---
    the reference force model --- but one call per object is 156,823 calls at
    catalogue scale, which is why it is not the bulk route.

``integrator``
    JPL SBDB osculating elements (``full-prec``) propagated here with
    :class:`NBodyPropagator`: Sun, eight planets, the Moon, Pluto and the
    SB441-N16 asteroids as point masses whose barycentric states come from
    Horizons on a daily grid (26 calls per run, cached), plus the Schwarzschild
    term for the Sun.  Fixed-step RK4 vectorised across every object in a
    chunk, with the observation epochs evaluated by quintic Hermite
    interpolation inside the step that contains them.  One bulk SBDB query
    replaces 156,823 Horizons calls.  **It is used for the bulk only after the
    probe stage has measured its disagreement with Horizons on a sample**
    (``results/sextant/probe_ephemeris.json``); the number is a measurement, not
    a claim.

``horizons_pinned_gravity_only``
    The positive-control route.  For an object whose JPL solution carries a
    fitted ``A2``, Horizons *includes* that acceleration, so a residual against
    Horizons contains no Yarkovsky signal by construction.  Instead the Horizons
    state at the middle of the Gaia arc is taken as the initial condition and
    propagated **gravity-only** to every epoch.  The difference between the true
    (accelerated) trajectory and this prediction is exactly the variational
    response with zero initial conditions at the pinning epoch --- which is
    precisely the signal basis the fit uses --- so the fitted ``A2`` must
    reproduce JPL's, sign and magnitude, or the estimator is wrong.

Everything network-facing is in :class:`HorizonsClient` and :func:`sbdb_bulk`
and runs only on the GitHub runner; the propagator, the interpolants, the
element conversion and the parsing are pure and tested offline.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .residuals import C_AU_PER_DAY

AU_KM = 149597870.7
DAY_S = 86400.0
#: km^3/s^2 -> au^3/day^2
GM_KM3S2_TO_AU3D2 = DAY_S ** 2 / AU_KM ** 3
#: J2000 mean obliquity used by JPL for ecliptic <-> ICRF equatorial (arcsec).
OBLIQUITY_J2000_ARCSEC = 84381.448
OBLIQUITY_J2000_RAD = math.radians(OBLIQUITY_J2000_ARCSEC / 3600.0)

# ---------------------------------------------------------------------------
# The force model: DE440 planetary GMs and the SB441-N16 asteroid perturbers
# ---------------------------------------------------------------------------
#: (label, Horizons COMMAND, GM km^3/s^2).  Planetary values are DE440 (Park et
#: al. 2021, AJ 161, 105); the asteroid values are the SB441-N16 set as
#: transcribed.  Only Ceres, Pallas and Vesta are load-bearing at the
#: milliarcsecond level over a six-year arc --- a 1 km^3/s^2 body perturbs a
#: main-belt orbit by tens of metres over three years --- and the probe's
#: integrator-vs-Horizons comparison is what measures the whole table at once.
PERTURBERS: tuple[tuple[str, str, float], ...] = (
    ("sun", "10", 132712440041.279419),
    ("mercury", "199", 22031.868551),
    ("venus", "299", 324858.592000),
    ("earth", "399", 398600.435507),
    ("moon", "301", 4902.800118),
    ("mars_bc", "4", 42828.375816),
    ("jupiter_bc", "5", 126712764.100000),
    ("saturn_bc", "6", 37940584.841800),
    ("uranus_bc", "7", 5794556.400000),
    ("neptune_bc", "8", 6836527.100580),
    ("pluto_bc", "9", 975.500000),
    ("ceres", "1;", 62.6284),
    ("pallas", "2;", 13.6659),
    ("juno", "3;", 1.8221),
    ("vesta", "4;", 17.2883),
    ("iris", "7;", 0.8626),
    ("hygiea", "10;", 5.6296),
    ("eunomia", "15;", 2.0674),
    ("psyche", "16;", 1.5303),
    ("euphrosyne", "31;", 1.1301),
    ("europa", "52;", 1.6018),
    ("cybele", "65;", 0.9323),
    ("sylvia", "87;", 0.9898),
    ("thisbe", "88;", 0.8071),
    ("camilla", "107;", 0.7451),
    ("davida", "511;", 2.5311),
    ("interamnia", "704;", 2.3227),
)
PERTURBER_NUMBERS: frozenset[int] = frozenset(
    int(cmd.rstrip(";")) for _, cmd, _ in PERTURBERS if cmd.endswith(";"))
GM_SUN_AU3_D2 = PERTURBERS[0][2] * GM_KM3S2_TO_AU3D2

#: The Gaia SSO window (FPR: 2014-07-25 .. 2020-01-20) with a margin, as JD.
WINDOW_JD = (2456850.0, 2458900.0)

HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
SBDB_QUERY_API = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
SBDB_API = "https://ssd-api.jpl.nasa.gov/sbdb.api"


class EphemerisError(RuntimeError):
    """A network or parsing failure whose message names the object and route."""


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


# ---------------------------------------------------------------------------
# 1. Parsing Horizons' text output (pure)
# ---------------------------------------------------------------------------
_SOE = "$$SOE"
_EOE = "$$EOE"


def parse_horizons_vectors(text: str) -> tuple[np.ndarray, np.ndarray]:
    """``(jd_tdb (N,), state (N, 6))`` from a CSV ``VECTORS`` table.

    With ``CSV_FORMAT=YES`` and ``VEC_TABLE=2`` every data line between
    ``$$SOE`` and ``$$EOE`` reads ``JDTDB, calendar, X, Y, Z, VX, VY, VZ,``.
    Anything else --- ``No matches found``, ``Multiple major-bodies match``,
    a truncated response --- has no ``$$SOE`` and raises with the head of the
    text, so the failure names itself instead of returning an empty table.
    """
    if _SOE not in text or _EOE not in text:
        head = text.strip().replace("\n", " | ")[:400]
        raise EphemerisError(f"Horizons returned no vector table: {head}")
    body = text.split(_SOE, 1)[1].split(_EOE, 1)[0]
    jds, states = [], []
    for line in body.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        try:
            jd = float(parts[0])
            st = [float(p) for p in parts[2:8]]
        except ValueError:
            continue
        jds.append(jd)
        states.append(st)
    if not jds:
        raise EphemerisError("Horizons vector table parsed to zero rows")
    return np.asarray(jds, dtype=float), np.asarray(states, dtype=float)


# ---------------------------------------------------------------------------
# 2. The Horizons client (runner only)
# ---------------------------------------------------------------------------
class HorizonsClient:
    """Thin ``requests`` client for the Horizons API with a rate limit and retries.

    Written against the API directly rather than through ``astroquery`` so the
    request is exactly what the docstring says it is: ``CENTER='500@0'``
    (barycentric), ``REF_PLANE='FRAME'`` (ICRF equatorial), ``TIME_TYPE='TDB'``,
    ``VEC_CORR='NONE'`` (geometric states; the residual chain applies light time
    and aberration itself), ``OUT_UNITS='AU-D'``.  A discrete ``TLIST`` is sent
    in pieces of ``tlist_chunk`` epochs so the GET URL stays short.

    ``min_interval`` throttles the client (default one request per second per
    process); 429/5xx and transport errors back off exponentially.
    """

    def __init__(self, url: str = HORIZONS_API, timeout: float = 120.0,
                 min_interval: float = 1.0, retries: int = 5,
                 tlist_chunk: int = 150):
        self.url = url
        self.timeout = float(timeout)
        self.min_interval = float(min_interval)
        self.retries = int(retries)
        self.tlist_chunk = int(tlist_chunk)
        self.calls = 0
        self.failures = 0
        self._last = 0.0
        self._session = None

    def _get(self, params: dict) -> str:
        import requests

        if self._session is None:
            self._session = requests.Session()
        last = ""
        for attempt in range(self.retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.calls += 1
            try:
                resp = self._session.get(self.url, params=params, timeout=self.timeout)
            except Exception as exc:                          # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"[:300]
                self.failures += 1
                time.sleep(min(60.0, 2.0 ** attempt))
                continue
            if resp.status_code == 200:
                return resp.text
            last = f"HTTP {resp.status_code}: {resp.text[:200]}"
            self.failures += 1
            if resp.status_code in (400, 404):
                break
            time.sleep(min(90.0, 3.0 * 2.0 ** attempt))
        raise EphemerisError(f"Horizons request failed: {last}")

    @staticmethod
    def _base(command: str, center: str) -> dict:
        return {
            "format": "text", "COMMAND": f"'{command}'", "OBJ_DATA": "NO",
            "MAKE_EPHEM": "YES", "EPHEM_TYPE": "VECTORS", "CENTER": f"'{center}'",
            "REF_PLANE": "FRAME", "REF_SYSTEM": "ICRF", "VEC_TABLE": "2",
            "VEC_CORR": "NONE", "OUT_UNITS": "AU-D", "CSV_FORMAT": "YES",
            "VEC_LABELS": "NO", "TIME_TYPE": "TDB",
        }

    def vectors_at(self, command: str, jd_tdb, center: str = "500@0"
                   ) -> tuple[np.ndarray, np.ndarray]:
        """Barycentric ICRF states at discrete TDB epochs, in the order given."""
        jd = np.atleast_1d(np.asarray(jd_tdb, dtype=float))
        order = np.argsort(jd)
        out = np.full((jd.size, 6), np.nan)
        got = np.full(jd.size, np.nan)
        for k in range(0, jd.size, self.tlist_chunk):
            idx = order[k:k + self.tlist_chunk]
            tl = " ".join(f"{x:.9f}" for x in jd[idx])
            params = self._base(command, center)
            params.update({"TLIST_TYPE": "JD", "TLIST": f"'{tl}'"})
            t_ret, st = parse_horizons_vectors(self._get(params))
            if t_ret.size != idx.size:
                raise EphemerisError(
                    f"Horizons returned {t_ret.size} epochs for {idx.size} requested "
                    f"({command})")
            out[idx] = st
            got[idx] = t_ret
        bad = np.abs(got - jd) > 1e-6
        if np.any(bad):
            raise EphemerisError(
                f"Horizons epochs differ from the request by up to "
                f"{float(np.nanmax(np.abs(got - jd))) * DAY_S:.3f} s ({command})")
        return got, out

    def vectors_grid(self, command: str, jd_start: float, jd_stop: float,
                     step: str = "1 d", center: str = "500@0"
                     ) -> tuple[np.ndarray, np.ndarray]:
        """Barycentric ICRF states on a regular grid (for the perturbers)."""
        params = self._base(command, center)
        params.update({"START_TIME": f"'JD{jd_start:.6f}'",
                       "STOP_TIME": f"'JD{jd_stop:.6f}'",
                       "STEP_SIZE": f"'{step}'"})
        return parse_horizons_vectors(self._get(params))


# ---------------------------------------------------------------------------
# 3. Interpolants (pure)
# ---------------------------------------------------------------------------
def hermite_cubic(t_grid: np.ndarray, pos: np.ndarray, vel: np.ndarray, t
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Cubic Hermite interpolation of ``(pos, vel)`` sampled on a uniform grid.

    ``pos``/``vel`` are ``(..., M, 3)`` and ``t`` is a scalar or ``(K,)``; the
    result is ``(..., K, 3)`` (or ``(..., 3)`` for a scalar ``t``).

    The truncation error is fourth order in the step.  **Measured** on a Kepler
    arc, not estimated: on a 1-day grid, 34 m for an Earth-like orbit and
    ~0.7 km for the Moon's geocentric motion, falling by the expected factor of
    16 per halving of the step (2.2 m at 0.5 d, 0.14 m at 0.25 d).  The
    dimensional estimate ``(h/P)^4 x amplitude`` gives 8 m for the Earth and is
    low by the ~4x coefficient of the cubic Hermite error term; the measurement
    is what this docstring quotes and what
    ``test_cubic_hermite_error_is_fourth_order_and_negligible_for_perturbers``
    pins.

    This interpolant is used **only for the perturbers** --- the target is
    integrated directly and densely output through :func:`hermite_quintic`, so
    this error never enters the target's own trajectory.  What it does enter is
    the perturbing acceleration, where a 34 m error in a planet's position
    displaces a main-belt target by well under a millimetre over the whole
    mission window: 3 GM_p dr / d^3 integrated twice over 2000 days.  The same
    test asserts that number rather than leaving it as an argument.
    """
    tg = np.asarray(t_grid, dtype=float)
    h = float(tg[1] - tg[0])
    tt = np.atleast_1d(np.asarray(t, dtype=float))
    i = np.clip(np.floor((tt - tg[0]) / h).astype(int), 0, tg.size - 2)
    s = (tt - tg[i]) / h
    s2, s3 = s * s, s * s * s
    h00, h10 = 2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s
    h01, h11 = -2 * s3 + 3 * s2, s3 - s2
    d00, d10 = 6 * s2 - 6 * s, 3 * s2 - 4 * s + 1
    d01, d11 = -6 * s2 + 6 * s, 3 * s2 - 2 * s
    p0, p1 = pos[..., i, :], pos[..., i + 1, :]
    v0, v1 = vel[..., i, :] * h, vel[..., i + 1, :] * h
    sh = (...,) + (None,)
    p = h00[sh] * p0 + h10[sh] * v0 + h01[sh] * p1 + h11[sh] * v1
    v = (d00[sh] * p0 + d10[sh] * v0 + d01[sh] * p1 + d11[sh] * v1) / h
    if np.ndim(t) == 0:
        return p[..., 0, :], v[..., 0, :]
    return p, v


def hermite_quintic(h: float, s, r0, v0, a0, r1, v1, a1
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Quintic Hermite on one step from ``(r, v, a)`` at both ends.

    ``s`` is the fractional position in the step, ``(K,)``; the state arrays
    are ``(K, 3)``.  The truncation error is ``O(h^6)``: with a 0.05-day RK4
    step it is below a millimetre for any bound heliocentric orbit.
    """
    s = np.asarray(s, dtype=float)[:, None]
    s2, s3, s4, s5 = s * s, s ** 3, s ** 4, s ** 5
    H0 = 1 - 10 * s3 + 15 * s4 - 6 * s5
    H1 = s - 6 * s3 + 8 * s4 - 3 * s5
    H2 = 0.5 * (s2 - 3 * s3 + 3 * s4 - s5)
    H3 = 10 * s3 - 15 * s4 + 6 * s5
    H4 = -4 * s3 + 7 * s4 - 3 * s5
    H5 = 0.5 * (s3 - 2 * s4 + s5)
    dH0 = -30 * s2 + 60 * s3 - 30 * s4
    dH1 = 1 - 18 * s2 + 32 * s3 - 15 * s4
    dH2 = 0.5 * (2 * s - 9 * s2 + 12 * s3 - 5 * s4)
    dH3 = 30 * s2 - 60 * s3 + 30 * s4
    dH4 = -12 * s2 + 28 * s3 - 15 * s4
    dH5 = 0.5 * (3 * s2 - 8 * s3 + 5 * s4)
    r = (H0 * r0 + H1 * h * v0 + H2 * h * h * a0
         + H3 * r1 + H4 * h * v1 + H5 * h * h * a1)
    v = (dH0 * r0 + dH1 * h * v0 + dH2 * h * h * a0
         + dH3 * r1 + dH4 * h * v1 + dH5 * h * h * a1) / h
    return r, v


# ---------------------------------------------------------------------------
# 4. The perturbers
# ---------------------------------------------------------------------------
@dataclass
class PerturberSet:
    """Barycentric states of every perturbing body on a daily grid."""

    t_grid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3)))   # (B, M, 3)
    vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3)))
    gm: np.ndarray = field(default_factory=lambda: np.zeros(0))            # au^3/d^2
    labels: list[str] = field(default_factory=list)
    source: str = ""
    retrieved_utc: str | None = None

    @property
    def sun_index(self) -> int:
        return self.labels.index("sun")

    @property
    def n_bodies(self) -> int:
        return len(self.labels)

    def states(self, t) -> tuple[np.ndarray, np.ndarray]:
        """``(pos (B, K, 3), vel (B, K, 3))`` at ``t (K,)`` (or ``(B, 3)`` for scalar)."""
        return hermite_cubic(self.t_grid, self.pos, self.vel, t)

    def sun_state(self, jd) -> np.ndarray:
        """The Sun's barycentric state ``(N, 6)`` --- the ``sun_state`` callable."""
        jd = np.atleast_1d(np.asarray(jd, dtype=float))
        p, v = hermite_cubic(self.t_grid, self.pos[self.sun_index],
                             self.vel[self.sun_index], jd)
        return np.column_stack([p, v])

    def covers(self, jd_lo: float, jd_hi: float) -> bool:
        return (self.t_grid.size > 1 and self.t_grid[0] <= jd_lo
                and self.t_grid[-1] >= jd_hi)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, t_grid=self.t_grid, pos=self.pos, vel=self.vel,
                            gm=self.gm, labels=np.array(self.labels),
                            source=np.array(self.source),
                            retrieved_utc=np.array(self.retrieved_utc or ""))

    @classmethod
    def load(cls, path: Path) -> PerturberSet:
        z = np.load(Path(path), allow_pickle=False)
        return cls(t_grid=z["t_grid"], pos=z["pos"], vel=z["vel"], gm=z["gm"],
                   labels=[str(x) for x in z["labels"]], source=str(z["source"]),
                   retrieved_utc=str(z["retrieved_utc"]) or None)

    @classmethod
    def from_horizons(cls, client: HorizonsClient, jd_lo: float = WINDOW_JD[0],
                      jd_hi: float = WINDOW_JD[1], margin_days: float = 30.0,
                      step: str = "1 d", bodies=PERTURBERS, on_body=None
                      ) -> PerturberSet:
        """Pull every perturber's daily state; 26 calls, cached by the caller."""
        from datetime import datetime, timezone

        lo, hi = float(jd_lo) - margin_days, float(jd_hi) + margin_days
        t_ref = None
        pos, vel, gm, labels = [], [], [], []
        for label, cmd, gm_km in bodies:
            t, st = client.vectors_grid(cmd, lo, hi, step=step)
            if t_ref is None:
                t_ref = t
            elif t.size != t_ref.size or np.max(np.abs(t - t_ref)) > 1e-6:
                raise EphemerisError(f"perturber grid for {label} does not match the Sun's")
            pos.append(st[:, :3])
            vel.append(st[:, 3:])
            gm.append(gm_km * GM_KM3S2_TO_AU3D2)
            labels.append(label)
            if on_body is not None:
                on_body(label, int(t.size))
        return cls(t_grid=t_ref, pos=np.array(pos), vel=np.array(vel),
                   gm=np.array(gm), labels=labels,
                   source="JPL Horizons VECTORS, CENTER=500@0, REF_PLANE=FRAME, DE441",
                   retrieved_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))


def synthetic_perturbers(jd_lo: float, jd_hi: float, step_days: float = 1.0,
                         sun_only: bool = True) -> PerturberSet:
    """A Sun-at-rest perturber set for offline tests (two-body truth)."""
    t = np.arange(jd_lo, jd_hi + step_days, step_days)
    pos = np.zeros((1, t.size, 3))
    vel = np.zeros((1, t.size, 3))
    return PerturberSet(t_grid=t, pos=pos, vel=vel, gm=np.array([GM_SUN_AU3_D2]),
                        labels=["sun"], source="synthetic_sun_at_barycentre")


# ---------------------------------------------------------------------------
# 5. Elements -> state (pure)
# ---------------------------------------------------------------------------
def ecliptic_to_equatorial(v: np.ndarray) -> np.ndarray:
    """Rotate ``(..., 3)`` vectors from the J2000 ecliptic to the ICRF equator."""
    c, s = math.cos(OBLIQUITY_J2000_RAD), math.sin(OBLIQUITY_J2000_RAD)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return np.asarray(v, dtype=float) @ rot.T


def elements_to_heliocentric_state(a_au, e, i_deg, node_deg, argp_deg, ma_deg,
                                   mu: float = GM_SUN_AU3_D2) -> np.ndarray:
    """Heliocentric ICRF equatorial state ``(N, 6)`` from JPL ecliptic elements.

    Vectorised Kepler solve (Newton, 60 iterations; converges to machine
    precision for ``e < 0.99``).  The elements are heliocentric and osculating
    with the Sun's GM alone, which is how JPL defines them.
    """
    a = np.atleast_1d(np.asarray(a_au, dtype=float))
    ecc = np.atleast_1d(np.asarray(e, dtype=float))
    inc = np.radians(np.atleast_1d(np.asarray(i_deg, dtype=float)))
    om = np.radians(np.atleast_1d(np.asarray(node_deg, dtype=float)))
    w = np.radians(np.atleast_1d(np.asarray(argp_deg, dtype=float)))
    m = np.radians(np.atleast_1d(np.asarray(ma_deg, dtype=float))) % (2 * math.pi)
    n = np.sqrt(mu / a ** 3)
    E = np.where(ecc < 0.8, m, np.pi)
    for _ in range(60):
        f = E - ecc * np.sin(E) - m
        E = E - f / (1.0 - ecc * np.cos(E))
    xv = a * (np.cos(E) - ecc)
    yv = a * np.sqrt(1.0 - ecc ** 2) * np.sin(E)
    r = a * (1.0 - ecc * np.cos(E))
    vx = -a * a * n * np.sin(E) / r
    vy = a * a * n * np.sqrt(1.0 - ecc ** 2) * np.cos(E) / r
    cw, sw, co, so, ci, si = np.cos(w), np.sin(w), np.cos(om), np.sin(om), np.cos(inc), np.sin(inc)

    def orient(x, y):
        x1 = x * cw - y * sw
        y1 = x * sw + y * cw
        return np.column_stack([x1 * co - y1 * ci * so,
                                x1 * so + y1 * ci * co,
                                y1 * si])

    pos = ecliptic_to_equatorial(orient(xv, yv))
    vel = ecliptic_to_equatorial(orient(vx, vy))
    return np.column_stack([pos, vel])


def perihelion_angular_rate(a_au, e, mu: float = GM_SUN_AU3_D2) -> np.ndarray:
    """``d(true anomaly)/dt`` at perihelion, rad/day --- what sets the RK4 step."""
    a = np.asarray(a_au, dtype=float)
    ecc = np.clip(np.asarray(e, dtype=float), 0.0, 0.999)
    n = np.sqrt(mu / a ** 3)
    return n * (1.0 + ecc) ** 2 / (1.0 - ecc ** 2) ** 1.5


#: RK4 step, in days, chosen so the perihelion angular step never exceeds what a
#: 0.05-day step is for an e = 0.4 main-belt orbit (8e-4 rad).  Bucketed so that
#: objects sharing a step can be integrated as one array.
STEP_BUCKETS_DAYS: tuple[float, ...] = (0.05, 0.025, 0.0125, 0.00625, 0.003125)
_REF_ANGULAR_STEP = 0.05 * 0.0155


def step_for(a_au, e) -> np.ndarray:
    """The RK4 step bucket for each orbit."""
    w = perihelion_angular_rate(a_au, e)
    want = _REF_ANGULAR_STEP / np.maximum(w, 1e-9)
    # The largest bucket that is still <= the wanted step; the smallest bucket
    # for anything more extreme than the buckets cover.
    out = np.full(np.shape(want), STEP_BUCKETS_DAYS[-1])
    assigned = np.zeros(np.shape(want), dtype=bool)
    for b in STEP_BUCKETS_DAYS:            # descending
        take = (want >= b) & ~assigned
        out = np.where(take, b, out)
        assigned |= take
    return out


# ---------------------------------------------------------------------------
# 6. The propagator (pure numpy, vectorised across objects)
# ---------------------------------------------------------------------------
@dataclass
class EvalRequest:
    """Which epochs each object must be evaluated at."""

    obj: np.ndarray        # (K,) object index within the batch
    jd: np.ndarray         # (K,) TDB epochs


class NBodyPropagator:
    """Massless test particles in the field of the perturbers, with Sun GR.

    ``propagate(state0 (n, 6) barycentric ICRF at jd0, requests)`` integrates
    every object from the common epoch ``jd0`` through the span of the requested
    epochs and returns ``(r, v, a)`` at each request, ``(K, 3)`` each, evaluated
    by quintic Hermite inside the step that contains the epoch.  Fixed-step RK4;
    the step is the caller's choice (see :func:`step_for`).
    """

    def __init__(self, perturbers: PerturberSet, relativity: bool = True):
        self.p = perturbers
        self.relativity = bool(relativity)
        self.rhs_calls = 0

    def acceleration(self, t: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        """``(n, 3)`` barycentric acceleration at time ``t`` for positions ``r``."""
        self.rhs_calls += 1
        bp, bv = self.p.states(t)                 # (B, 3)
        d = bp[:, None, :] - r[None, :, :]        # (B, n, 3)
        d2 = np.sum(d * d, axis=-1)
        acc = np.sum(self.p.gm[:, None, None] * d / (d2 * np.sqrt(d2))[:, :, None],
                     axis=0)
        if self.relativity:
            k = self.p.sun_index
            rh = r - bp[k][None, :]
            vh = v - bv[k][None, :]
            rn2 = np.sum(rh * rh, axis=1)
            rn = np.sqrt(rn2)
            mu = float(self.p.gm[k])
            c2 = C_AU_PER_DAY ** 2
            v2 = np.sum(vh * vh, axis=1)
            rv = np.sum(rh * vh, axis=1)
            coef = mu / (c2 * rn2 * rn)
            acc = acc + coef[:, None] * ((4.0 * mu / rn - v2)[:, None] * rh
                                         + 4.0 * rv[:, None] * vh)
        return acc

    def _run(self, r, v, a, jd0: float, t_end: float, h: float,
             ev_jd: np.ndarray, ev_obj: np.ndarray, out_r, out_v, out_a, ev_slots):
        """Step from ``jd0`` to ``t_end`` (either direction), filling evaluations.

        Time is carried as ``tau = t - jd0`` with every node at an exact integer
        multiple of the step.  Accumulating ``t += h`` at ``t ~ 2.457e6`` costs
        ~5e-10 day of roundoff per addition --- 20 additions a day at a 0.05-day
        step is 1e-8 day, which at 0.01 au/day is fifteen metres a day and
        grows without bound.  Node times are therefore *computed*, never
        accumulated, and the perturbers are asked at ``jd0 + tau`` with a
        single rounding each.
        """
        direction = 1.0 if t_end >= jd0 else -1.0
        span = abs(t_end - jd0)
        n_steps = max(int(math.ceil(span / h - 1e-9)), 1)
        hs = direction * h
        ev_tau = ev_jd - jd0
        order = np.argsort(direction * ev_tau)
        ev_tau, ev_obj, ev_slots = ev_tau[order], ev_obj[order], ev_slots[order]
        cursor = 0
        for k in range(n_steps):
            tau = k * hs
            tau_new = (k + 1) * hs
            k1v = a
            k1r = v
            r2 = r + 0.5 * hs * k1r
            v2 = v + 0.5 * hs * k1v
            k2v = self.acceleration(jd0 + (tau + 0.5 * hs), r2, v2)
            k2r = v2
            r3 = r + 0.5 * hs * k2r
            v3 = v + 0.5 * hs * k2v
            k3v = self.acceleration(jd0 + (tau + 0.5 * hs), r3, v3)
            k3r = v3
            r4 = r + hs * k3r
            v4 = v + hs * k3v
            k4v = self.acceleration(jd0 + tau_new, r4, v4)
            k4r = v4
            r_new = r + hs / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
            v_new = v + hs / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
            a_new = self.acceleration(jd0 + tau_new, r_new, v_new)
            # Evaluations inside this step.  The final step takes everything
            # that remains, so an epoch at the very end of the span is never
            # lost to roundoff in the comparison.
            if k == n_steps - 1:
                end = ev_tau.size
            else:
                end = cursor
                while end < ev_tau.size and direction * (ev_tau[end] - tau_new) < 0.0:
                    end += 1
            if end > cursor:
                sl = slice(cursor, end)
                s = (ev_tau[sl] - tau) / hs
                o = ev_obj[sl]
                rr, vv = hermite_quintic(hs, s, r[o], v[o], a[o],
                                         r_new[o], v_new[o], a_new[o])
                out_r[ev_slots[sl]] = rr
                out_v[ev_slots[sl]] = vv
                # Acceleration at the evaluation epoch, for the light-time
                # expansion: linear in the step is ample (its own error is
                # second order in a ten-minute offset).
                out_a[ev_slots[sl]] = (a[o] + (a_new[o] - a[o]) * s[:, None])
                cursor = end
            r, v, a = r_new, v_new, a_new
        return r, v, a

    def propagate(self, state0: np.ndarray, jd0: float, req: EvalRequest,
                  h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(r, v, a)`` at every requested epoch, each ``(K, 3)``, barycentric."""
        st = np.atleast_2d(np.asarray(state0, dtype=float))
        n = st.shape[0]
        r, v = st[:, :3].copy(), st[:, 3:].copy()
        jd_e = np.asarray(req.jd, dtype=float)
        obj = np.asarray(req.obj, dtype=int)
        K = jd_e.size
        out_r = np.full((K, 3), np.nan)
        out_v = np.full((K, 3), np.nan)
        out_a = np.full((K, 3), np.nan)
        if K == 0 or n == 0:
            return out_r, out_v, out_a
        if obj.max() >= n:
            raise ValueError("evaluation request refers to an object outside the batch")
        slots = np.arange(K)
        a0 = self.acceleration(float(jd0), r, v)
        # Requests exactly at jd0 are the state itself.
        at0 = np.abs(jd_e - jd0) <= 1e-12
        if at0.any():
            out_r[at0], out_v[at0], out_a[at0] = r[obj[at0]], v[obj[at0]], a0[obj[at0]]
        fwd = jd_e > jd0 + 1e-12
        bwd = jd_e < jd0 - 1e-12
        if fwd.any():
            self._run(r, v, a0, float(jd0), float(jd_e[fwd].max()), h,
                      jd_e[fwd], obj[fwd], out_r, out_v, out_a, slots[fwd])
        if bwd.any():
            self._run(r, v, a0, float(jd0), float(jd_e[bwd].min()), h,
                      jd_e[bwd], obj[bwd], out_r, out_v, out_a, slots[bwd])
        return out_r, out_v, out_a


# ---------------------------------------------------------------------------
# 7. The aligned Taylor ephemeris the residual chain consumes
# ---------------------------------------------------------------------------
class AlignedEphemeris:
    """``target_state(jd)`` for :func:`seti.sextant.residuals.compute_residuals`.

    The chain calls ``target_state`` with an array aligned one-to-one with the
    observations (the light-time iteration evaluates at ``t_i - tau_i``), so this
    holds the state, acceleration and two-body jerk at each observation's
    reference epoch and expands about it: ``r + v dt + a dt^2/2 + j dt^3/6``.
    With ``dt`` of at most a few minutes the neglected term is far below a
    metre.  A call with a single epoch (the reference-state request) expands
    from the nearest reference; any other length, or an offset beyond
    ``max_offset_days``, raises rather than interpolating across a gap it was
    never meant to bridge.
    """

    def __init__(self, jd_ref: np.ndarray, r: np.ndarray, v: np.ndarray,
                 a: np.ndarray, sun_r: np.ndarray | None = None,
                 max_offset_days: float = 0.1):
        self.jd_ref = np.asarray(jd_ref, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.a = np.asarray(a, dtype=float)
        self.max_offset_days = float(max_offset_days)
        rh = self.r - (np.asarray(sun_r, dtype=float) if sun_r is not None else 0.0)
        rn = np.linalg.norm(rh, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.j = -GM_SUN_AU3_D2 * (self.v / rn[:, None] ** 3
                                       - 3.0 * np.sum(rh * self.v, axis=1)[:, None]
                                       * rh / rn[:, None] ** 5)

    def _expand(self, idx: np.ndarray, dt: np.ndarray) -> np.ndarray:
        dt = dt[:, None]
        r = (self.r[idx] + self.v[idx] * dt + 0.5 * self.a[idx] * dt ** 2
             + self.j[idx] * dt ** 3 / 6.0)
        v = self.v[idx] + self.a[idx] * dt + 0.5 * self.j[idx] * dt ** 2
        return np.column_stack([r, v])

    def __call__(self, jd) -> np.ndarray:
        t = np.atleast_1d(np.asarray(jd, dtype=float))
        if t.size == self.jd_ref.size:
            idx = np.arange(t.size)
        elif t.size == 1:
            idx = np.array([int(np.nanargmin(np.abs(self.jd_ref - t[0])))])
        else:
            raise ValueError(
                f"AlignedEphemeris expects {self.jd_ref.size} epochs (or 1), got {t.size}")
        dt = t - self.jd_ref[idx]
        bad = np.abs(dt) > self.max_offset_days
        out = self._expand(idx, np.where(bad, 0.0, dt))
        if np.any(bad):
            out[bad] = np.nan
        return out


def expand_transit_states(jd_obs: np.ndarray, transit_id: np.ndarray,
                          jd_tr: np.ndarray, state_tr: np.ndarray,
                          sun_state_fn=None) -> AlignedEphemeris:
    """Per-observation ephemeris from one Horizons state per transit.

    ``jd_tr``/``state_tr`` are the states at the first epoch of each unique
    transit (in ``np.unique`` order of ``transit_id``); the acceleration for the
    expansion is the two-body one about the Sun, whose neglected planetary part
    over 40 s is nanometres.
    """
    jd_obs = np.asarray(jd_obs, dtype=float)
    tid = np.asarray(transit_id)
    uniq, inv = np.unique(tid, return_inverse=True)
    if uniq.size != jd_tr.size:
        raise ValueError("one state per unique transit is required")
    st = state_tr[inv]
    sun = (np.atleast_2d(sun_state_fn(jd_tr))[inv, :3] if sun_state_fn is not None
           else np.zeros((jd_obs.size, 3)))
    rh = st[:, :3] - sun
    rn = np.linalg.norm(rh, axis=1)
    acc = -GM_SUN_AU3_D2 * rh / rn[:, None] ** 3
    base = AlignedEphemeris(jd_tr[inv], st[:, :3], st[:, 3:], acc, sun_r=sun)
    dt = jd_obs - jd_tr[inv]
    full = base._expand(np.arange(jd_obs.size), dt)
    acc_obs = acc + base.j * dt[:, None]
    return AlignedEphemeris(jd_obs, full[:, :3], full[:, 3:], acc_obs, sun_r=sun)


def transit_reference_epochs(jd_obs: np.ndarray, transit_id: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    """``(unique transit ids, earliest epoch of each)`` in ``np.unique`` order."""
    jd_obs = np.asarray(jd_obs, dtype=float)
    tid = np.asarray(transit_id)
    uniq, inv = np.unique(tid, return_inverse=True)
    first = np.full(uniq.size, np.inf)
    np.minimum.at(first, inv, np.where(np.isfinite(jd_obs), jd_obs, np.inf))
    return uniq, first


# ---------------------------------------------------------------------------
# 8. JPL SBDB (runner only) and its row parsing (pure)
# ---------------------------------------------------------------------------
SBDB_FIELDS: tuple[str, ...] = (
    "pdes", "full_name", "H", "G", "albedo", "diameter", "a", "e", "i", "om", "w",
    "ma", "q", "epoch", "A1", "A2", "A3", "A2_sigma", "n_obs_used", "data_arc",
    "first_obs", "last_obs", "rms", "condition_code", "n_opp", "producer",
    "two_body", "neo", "pha", "class",
)
#: Optional spellings tried and dropped on rejection, as loom.calibrate does.
SBDB_OPTIONAL: tuple[str, ...] = ("A1_sigma", "A3_sigma", "sigma_A1", "sigma_A3")
_INVALID_FIELD = re.compile(r"invalid field specified:\s*'([^']+)'")


def parse_sbdb_row(row: dict) -> dict:
    """Numeric-cleaned SBDB row keyed by the names the pipeline uses."""
    out = {
        "number_mp": int(_f(row.get("pdes"))) if _f(row.get("pdes")) == _f(row.get("pdes"))
        and float(_f(row.get("pdes"))).is_integer() else None,
        "full_name": (str(row.get("full_name") or "")).strip(),
        "h": _f(row.get("H")), "g": _f(row.get("G")), "albedo": _f(row.get("albedo")),
        "diameter_km": _f(row.get("diameter")),
        "a": _f(row.get("a")), "e": _f(row.get("e")), "i": _f(row.get("i")),
        "node": _f(row.get("om")), "argperi": _f(row.get("w")), "ma": _f(row.get("ma")),
        "q": _f(row.get("q")), "epoch_jd": _f(row.get("epoch")),
        "a1": _f(row.get("A1")), "a2": _f(row.get("A2")), "a3": _f(row.get("A3")),
        "a2_sigma": _f(row.get("A2_sigma")),
        "a1_sigma": _f(row.get("A1_sigma", row.get("sigma_A1"))),
        "a3_sigma": _f(row.get("A3_sigma", row.get("sigma_A3"))),
        "n_obs_used": _f(row.get("n_obs_used")), "data_arc": _f(row.get("data_arc")),
        "first_obs": row.get("first_obs"), "last_obs": row.get("last_obs"),
        "rms": _f(row.get("rms")), "condition_code": _f(row.get("condition_code")),
        "n_opp": _f(row.get("n_opp")), "producer": row.get("producer"),
        "two_body": row.get("two_body"), "neo": row.get("neo"), "pha": row.get("pha"),
        "orbit_class": row.get("class"),
    }
    out["nongrav_fitted"] = bool(math.isfinite(out["a2"]) or math.isfinite(out["a1"])
                                 or math.isfinite(out["a3"]))
    return out


def sbdb_bulk(numbered_only: bool = True, timeout: float = 900.0,
              fields=SBDB_FIELDS, optional=SBDB_OPTIONAL, retries: int = 3,
              url: str = SBDB_QUERY_API) -> dict:
    """Every numbered asteroid's JPL orbit, in full precision, in one query.

    ``full-prec=true`` is not optional: the default output rounds the elements
    to a precision that is kilometres at 2.5 au, i.e. milliarcseconds --- the
    signal.  A field the API rejects is dropped by name and the request retried,
    so one wrong spelling does not cost the run.
    """
    import requests

    current = list(fields) + list(optional)
    base = {"sb-kind": "a", "full-prec": "true"}
    if numbered_only:
        base["sb-ns"] = "n"
    out: dict = {"url": url, "params": dict(base), "dropped_fields": [],
                 "verdict": "NO_DATA_REACHED", "rows": {}, "n_rows": 0}
    last = ""
    for attempt in range(retries):
        for _ in range(len(optional) + 4):
            params = {"fields": ",".join(current), **base}
            try:
                resp = requests.get(url, params=params, timeout=timeout)
            except Exception as exc:                          # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"[:300]
                resp = None
                break
            if resp.status_code == 400:
                m = _INVALID_FIELD.search(resp.text or "")
                if m and m.group(1) in current:
                    current.remove(m.group(1))
                    out["dropped_fields"].append(m.group(1))
                    continue
            break
        if resp is None:
            time.sleep(10.0 * (attempt + 1))
            continue
        out["status"] = resp.status_code
        if resp.status_code != 200:
            last = resp.text[:300]
            time.sleep(10.0 * (attempt + 1))
            continue
        payload = resp.json()
        cols = payload.get("fields") or []
        data = payload.get("data") or []
        rows = {}
        for raw in data:
            rec = parse_sbdb_row(dict(zip(cols, raw, strict=False)))
            if rec["number_mp"] is not None:
                rows[rec["number_mp"]] = rec
        out["rows"] = rows
        out["n_rows"] = len(rows)
        out["fields_used"] = list(current)
        out["verdict"] = "OK" if rows else "EMPTY"
        return out
    out["error"] = last
    return out


def sbdb_single(number: int, timeout: float = 60.0, url: str = SBDB_API) -> dict:
    """One object's JPL orbit from ``sbdb.api`` (the per-object fallback)."""
    import requests

    resp = requests.get(url, params={"sstr": str(int(number)), "full-prec": "true"},
                        timeout=timeout)
    if resp.status_code != 200:
        raise EphemerisError(f"sbdb.api {number}: HTTP {resp.status_code}")
    p = resp.json()
    orb = p.get("orbit") or {}
    el = {e.get("name"): e.get("value") for e in (orb.get("elements") or [])}
    phys = {e.get("name"): e.get("value") for e in (p.get("phys_par") or [])}
    row = {
        "pdes": (p.get("object") or {}).get("des"),
        "full_name": (p.get("object") or {}).get("fullname"),
        "H": phys.get("H"), "G": phys.get("G"), "albedo": phys.get("albedo"),
        "diameter": phys.get("diameter"),
        "a": el.get("a"), "e": el.get("e"), "i": el.get("i"), "om": el.get("om"),
        "w": el.get("w"), "ma": el.get("ma"), "q": el.get("q"),
        "epoch": orb.get("epoch"), "n_obs_used": orb.get("n_obs_used"),
        "data_arc": orb.get("data_arc"), "first_obs": orb.get("first_obs"),
        "last_obs": orb.get("last_obs"), "rms": orb.get("rms"),
        "condition_code": orb.get("condition_code"), "n_opp": orb.get("n_opp"),
        "producer": orb.get("producer"), "two_body": orb.get("two_body"),
    }
    for m in orb.get("model_pars") or []:
        nm = m.get("name")
        if nm in ("A1", "A2", "A3"):
            row[nm] = m.get("value")
            row[f"{nm}_sigma"] = m.get("sigma")
    return parse_sbdb_row(row)


def save_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)
