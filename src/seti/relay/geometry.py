"""The relay geometry and its Doppler prior --- pure functions, offline-testable.

Conventions
-----------
* Earth (the solar-system barycentre; the difference is irrelevant at parsec
  scale) is the origin.  A star at (ra, dec, parallax) is the vector
  ``d * u`` with ``d = 1000 / parallax`` pc and ``u`` its unit vector.
* A *directed pair* is (T, R): T transmits, R receives.  ``theta`` is the
  full beam width, taken as ``1.22 lambda / D`` for a diffraction-limited
  aperture (the FWHM-class number the brief quotes: 9' for 100 m at 1.4 GHz,
  1.5 deg for 10 m).  Earth is inside the beam when the *transmitter angle*
  ``alpha`` --- the angle at T between T->R and T->Earth --- is ``<= theta/2``.
* ``alpha`` is computed EXACTLY from the 3-D positions.  The brief's
  small-angle sky conditions, ``delta <= (theta/2)(d_T - d_R)/d_R``
  (spillover) and ``delta' <= (theta/2)(d_T + d_R)/d_R`` (Earth between the
  nodes, ``delta'`` measured from T's antipode), are the limits of the exact
  bounds in :func:`delta_max_exact`; the exact bounds at the widest allowed
  d_T are the SEARCH radii, and every candidate is then tested on ``alpha``.
* Geometry class: ``spillover`` when the sky separation of T and R is below
  90 deg (Earth lies beyond R along the beam), ``between`` when it is above
  (the beam passes Earth before reaching R).  The two are counted separately
  because their yields differ by an order of magnitude and their flux ratios
  sit on opposite sides of one.
* ``flux_ratio`` is the flux Earth receives over the flux R receives:
  ``(|R - T| / |T|)^2`` --- below one for spillover, above one for the between
  geometry, where the intercept is *stronger* than the intended link.

Kinematics and the drift prior
------------------------------
A relay transmitter that de-drifts its carrier so R receives a constant
frequency emits ``f_e(t) = f0 (1 + v_TR/c)``, with ``v_TR`` the T--R range
rate.  Earth then sees ``f0 (1 + (v_TR - v_TE)/c)`` and a drift
``(f0/c)(a_TR - a_TE)``, where ``a_TR`` is the T--R range acceleration and
``a_TE`` the T--Earth one.  For free-flying field stars the range
acceleration of straight-line relative motion is ``v_perp^2 / r`` --- of order
1e-9 m/s^2, a drift of 1e-8 Hz/s at L band, below any pipeline's resolution.
What is NOT negligible in ``a_TE`` is Earth's own acceleration projected onto
the line of sight (rotation up to 0.034 m/s^2, orbit 0.006 m/s^2): the drift
a de-drifted relay shows at a telescope is the Earth term, with the pair's
kinematic term as a tiny correction, plus the transmitter's own local
acceleration leaking through the small angle ``alpha`` between the T->R and
T->Earth directions, ``<= alpha * a_local``.  So the prior is a NUMBER per
pointing --- not the +/- 4 Hz/s window every published search uses --- and
terrestrial interference, which sits at zero topocentric drift, lies outside
it whenever the Earth term is resolved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

C_M_S = 299_792_458.0
AU_M = 1.495978707e11
PC_M = 3.0856775814913673e16
GM_SUN = 1.32712440018e20            # m^3 s^-2
R_EARTH_M = 6.378137e6
OMEGA_EARTH = 7.2921150e-5           # rad/s, sidereal
K_TAN = 4.740470446                  # km/s per (mas/yr * kpc)
DEG = math.pi / 180.0
ARCMIN = DEG / 60.0
ARCSEC = DEG / 3600.0

GEOM_SPILLOVER = "spillover"
GEOM_BETWEEN = "between"


# ---------------------------------------------------------------------------
# beams
# ---------------------------------------------------------------------------
def theta_diffraction(diameter_m: float, wavelength_m: float) -> float:
    """Full beam width ``1.22 lambda / D`` in radians."""
    return 1.22 * float(wavelength_m) / float(diameter_m)


def beam_grid(conf_beams: list[dict]) -> list[dict]:
    """Resolve the configured beam grid into ``{name, theta_rad, theta_label, ...}``.

    Each entry gives either ``theta_deg`` / ``theta_arcmin`` / ``theta_arcsec``
    directly or an ``aperture_m`` + ``frequency_ghz`` | ``wavelength_um`` pair.
    """
    out = []
    for b in conf_beams:
        b = dict(b)
        if "theta_deg" in b:
            th = float(b["theta_deg"]) * DEG
        elif "theta_arcmin" in b:
            th = float(b["theta_arcmin"]) * ARCMIN
        elif "theta_arcsec" in b:
            th = float(b["theta_arcsec"]) * ARCSEC
        else:
            if "frequency_ghz" in b:
                lam = C_M_S / (float(b["frequency_ghz"]) * 1e9)
            else:
                lam = float(b["wavelength_um"]) * 1e-6
            th = theta_diffraction(float(b["aperture_m"]), lam)
        b["theta_rad"] = th
        b["theta_deg"] = th / DEG
        b["theta_label"] = angle_label(th)
        out.append(b)
    return out


def angle_label(theta_rad: float) -> str:
    if theta_rad >= 1 * DEG:
        return f"{theta_rad / DEG:.3g} deg"
    if theta_rad >= 1 * ARCMIN:
        return f"{theta_rad / ARCMIN:.3g} arcmin"
    if theta_rad >= 1e-3 * ARCSEC:
        return f"{theta_rad / ARCSEC:.3g} arcsec"
    return f"{theta_rad:.3g} rad"


# ---------------------------------------------------------------------------
# positions and velocities
# ---------------------------------------------------------------------------
def unit_vectors(ra_deg, dec_deg) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def positions_pc(ra_deg, dec_deg, parallax_mas) -> np.ndarray:
    d = 1000.0 / np.asarray(parallax_mas, float)
    return unit_vectors(ra_deg, dec_deg) * d[..., None]


def tangential_basis(ra_deg, dec_deg) -> tuple[np.ndarray, np.ndarray]:
    """East and north unit vectors at each position."""
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    east = np.stack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)], axis=-1)
    north = np.stack([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)],
                     axis=-1)
    return east, north


def space_velocities(ra_deg, dec_deg, parallax_mas, pmra_masyr, pmdec_masyr, rv_kms
                     ) -> np.ndarray:
    """Barycentric velocity vectors (km/s); NaN rows where any input is missing."""
    plx = np.asarray(parallax_mas, float)
    d_kpc = 1.0 / plx
    vt_e = K_TAN * np.asarray(pmra_masyr, float) * d_kpc
    vt_n = K_TAN * np.asarray(pmdec_masyr, float) * d_kpc
    rv = np.asarray(rv_kms, float)
    e, n = tangential_basis(ra_deg, dec_deg)
    u = unit_vectors(ra_deg, dec_deg)
    return vt_e[..., None] * e + vt_n[..., None] * n + rv[..., None] * u


def sky_separation_deg(u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    dot = np.clip(np.sum(u1 * u2, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def chord(angle_rad) -> np.ndarray:
    """Chord length on the unit sphere for an angular radius (capped at 180 deg)."""
    a = np.minimum(np.asarray(angle_rad, float), math.pi)
    return 2.0 * np.sin(a / 2.0)


# ---------------------------------------------------------------------------
# the cone condition
# ---------------------------------------------------------------------------
def transmitter_angle(xyz_t: np.ndarray, xyz_r: np.ndarray) -> np.ndarray:
    """Exact angle at T between the directions T->R and T->Earth (radians)."""
    tr = xyz_r - xyz_t
    te = -xyz_t
    num = np.sum(tr * te, axis=-1)
    den = np.linalg.norm(tr, axis=-1) * np.linalg.norm(te, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosa = np.clip(num / den, -1.0, 1.0)
    return np.arccos(cosa)


def delta_max_spillover(theta_rad: float, d_t, d_r) -> np.ndarray:
    """Small-angle sky-separation bound for the spillover geometry (radians)."""
    d_t = np.asarray(d_t, float)
    d_r = np.asarray(d_r, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.maximum(0.5 * theta_rad * (d_t - d_r) / d_r, 0.0)


def delta_max_between(theta_rad: float, d_t, d_r) -> np.ndarray:
    """Small-angle bound on the separation of R from T's antipode (radians)."""
    d_t = np.asarray(d_t, float)
    d_r = np.asarray(d_r, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * theta_rad * (d_t + d_r) / d_r


def delta_max_exact(theta_rad: float, d_t, d_r) -> tuple[np.ndarray, np.ndarray]:
    """EXACT sky-separation bounds for both geometries (radians).

    With ``k = sin(theta/2)`` the cone condition ``tan(alpha) = d_R sin(delta) /
    (d_T -/+ d_R cos(delta))`` solves to

        spillover:  delta  <= arcsin(k d_T / d_R) - theta/2
        between:    delta' <= arcsin(k d_T / d_R) + theta/2   (delta' from T's antipode)

    and when ``k d_T / d_R >= 1`` --- a receiver within ``d_T sin(theta/2)`` of
    Earth --- the transmitter angle never exceeds ``theta/2`` at ANY separation
    (its maximum is ``arcsin(d_R/d_T)``), so both bounds saturate to pi.  The
    small-angle forms are what these reduce to for ``arcsin(x) ~ x``; they are
    strictly smaller, so a search that used them would miss the nearest
    receivers (found: the brute-force cross-check in the test suite).
    """
    d_t = np.asarray(d_t, float)
    d_r = np.asarray(d_r, float)
    half = 0.5 * float(theta_rad)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = math.sin(half) * d_t / d_r
    sat = ~(x < 1.0)
    core = np.arcsin(np.clip(x, 0.0, 1.0))
    spill = np.where(sat, math.pi, np.maximum(core - half, 0.0))
    between = np.where(sat, math.pi, core + half)
    return spill, between


def analytic_expectation(theta_rad: float, n_stars: int) -> dict:
    """Uniform-density expectations for the directed-pair counts.

    Spillover: ``theta^2 N^2 / 32``; between: ``(9 * 0.5611 / 16) theta^2 N^2
    = 0.3156 theta^2 N^2`` --- both from integrating the small-angle cap
    fraction over a uniform sphere of N stars (docs/relay.md section 2).  They
    are what the brief's "theta^2 N^2 scaling" means numerically; the
    measured counts depart from them where the small-angle approximation and
    the uniform density do not hold (wide beams, the Galactic-plane gradient,
    the real distance distribution).
    """
    t2n2 = float(theta_rad) ** 2 * float(n_stars) ** 2
    return {"spillover": t2n2 / 32.0, "between": 0.315625 * t2n2, "theta2_n2": t2n2}


@dataclass
class PairSearchResult:
    pairs: pd.DataFrame
    n_spillover: int
    n_between: int
    n_candidates_tested: int
    n_kept_rows: int
    truncated: bool
    per_transmitter_spill: np.ndarray
    per_transmitter_between: np.ndarray

    def as_dict(self) -> dict:
        return {"n_spillover": int(self.n_spillover), "n_between": int(self.n_between),
                "n_directed_pairs": int(self.n_spillover + self.n_between),
                "n_candidates_tested": int(self.n_candidates_tested),
                "n_kept_rows": int(self.n_kept_rows), "rows_truncated": bool(self.truncated)}


def find_pairs(xyz: np.ndarray, theta_rad: float, *, d_max_pc: float | None = None,
               keep_transmitters=None, row_cap: int = 5_000_000,
               chunk_candidates: int = 4_000_000, min_separation_pc: float = 0.0,
               progress=None) -> PairSearchResult:
    """Every directed pair (T, R) whose transmitter angle is ``<= theta/2``.

    ``xyz`` is (N, 3) in parsecs.  Counts are exact for the whole set; ROWS are
    kept for every pair whose transmitter index is in ``keep_transmitters``
    (``None`` = keep all) up to ``row_cap``, after which counting continues
    without storing (``truncated`` says so).  ``min_separation_pc`` drops pairs
    closer than a bound-binary scale (the brief's field-star network, not a
    binary talking to itself).
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, float)
    n = len(xyz)
    d = np.linalg.norm(xyz, axis=1)
    u = xyz / np.maximum(d, 1e-30)[:, None]
    d_max = float(d_max_pc) if d_max_pc else float(np.nanmax(d)) if n else 0.0
    tree = cKDTree(u) if n else None
    keep_mask = None
    if keep_transmitters is not None:
        keep_mask = np.zeros(n, bool)
        idx = np.asarray(list(keep_transmitters), int)
        if len(idx):
            keep_mask[idx[(idx >= 0) & (idx < n)]] = True

    half = 0.5 * float(theta_rad)
    per_t_spill = np.zeros(n, np.int64)
    per_t_between = np.zeros(n, np.int64)
    n_spill = n_between = n_tested = 0
    kept: list[pd.DataFrame] = []
    n_kept = 0
    truncated = False

    if n == 0:
        return PairSearchResult(_empty_pairs(), 0, 0, 0, 0, False, per_t_spill, per_t_between)

    # Search radii around each RECEIVER: the exact bounds at the widest allowed
    # d_T (a superset of every qualifying separation; the exact angle decides).
    r_spill, r_between = delta_max_exact(theta_rad, d_max, d)
    # expected candidates per receiver ~ N * cap fraction, used to size chunks
    frac = 0.5 * (1 - np.cos(r_spill)) + 0.5 * (1 - np.cos(r_between))
    exp_per_r = np.maximum(frac * n, 1.0)
    order = np.argsort(d)               # nearby receivers have the widest caps
    starts = _chunk_starts(exp_per_r[order], chunk_candidates)

    for c0, c1 in zip(starts[:-1], starts[1:], strict=False):
        rows_r = order[c0:c1]
        for geom, centres, radii in (
            (GEOM_SPILLOVER, u[rows_r], r_spill[rows_r]),
            (GEOM_BETWEEN, -u[rows_r], r_between[rows_r]),
        ):
            live = radii > 0
            if not live.any():
                continue
            lists = tree.query_ball_point(centres[live], chord(radii[live]),
                                          return_sorted=False)
            lens = np.fromiter((len(x) for x in lists), int, count=len(lists))
            if lens.sum() == 0:
                continue
            t_idx = np.concatenate([np.asarray(x, int) for x in lists if len(x)])
            r_idx = np.repeat(rows_r[live], lens)
            # a star is not its own receiver; the between search can also return
            # T on the near side of R, which the exact angle rejects anyway
            m = t_idx != r_idx
            t_idx, r_idx = t_idx[m], r_idx[m]
            n_tested += len(t_idx)
            if not len(t_idx):
                continue
            alpha = transmitter_angle(xyz[t_idx], xyz[r_idx])
            ok = alpha <= half
            if min_separation_pc > 0:
                ok &= np.linalg.norm(xyz[t_idx] - xyz[r_idx], axis=1) >= min_separation_pc
            sep = sky_separation_deg(u[t_idx], u[r_idx])
            if geom == GEOM_SPILLOVER:
                ok &= sep < 90.0
            else:
                ok &= sep >= 90.0
            t_idx, r_idx, alpha, sep = t_idx[ok], r_idx[ok], alpha[ok], sep[ok]
            if not len(t_idx):
                continue
            counts = np.bincount(t_idx, minlength=n)
            if geom == GEOM_SPILLOVER:
                n_spill += len(t_idx)
                per_t_spill += counts
            else:
                n_between += len(t_idx)
                per_t_between += counts
            if truncated:
                continue
            sel = np.ones(len(t_idx), bool) if keep_mask is None else keep_mask[t_idx]
            if not sel.any():
                continue
            room = row_cap - n_kept
            if sel.sum() > room:
                truncated = True
                pick = np.flatnonzero(sel)[:max(room, 0)]
                sel = np.zeros(len(t_idx), bool)
                sel[pick] = True
            if sel.any():
                kept.append(_pair_frame(t_idx[sel], r_idx[sel], alpha[sel], sep[sel],
                                        d, xyz, geom, theta_rad))
                n_kept += int(sel.sum())
        if progress is not None:
            progress(int(c1), n, n_spill, n_between)

    pairs = pd.concat(kept, ignore_index=True) if kept else _empty_pairs()
    return PairSearchResult(pairs, n_spill, n_between, n_tested, n_kept, truncated,
                            per_t_spill, per_t_between)


def _chunk_starts(expected: np.ndarray, budget: int) -> list[int]:
    starts = [0]
    acc = 0.0
    for i, e in enumerate(expected):
        acc += float(e)
        if acc >= budget:
            starts.append(i + 1)
            acc = 0.0
    if starts[-1] != len(expected):
        starts.append(len(expected))
    return starts


_PAIR_COLUMNS = ["t_idx", "r_idx", "geometry", "theta_rad", "alpha_rad", "alpha_over_half_theta",
                 "sep_deg", "d_t_pc", "d_r_pc", "r_tr_pc", "flux_ratio_earth_over_receiver"]


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float) for c in _PAIR_COLUMNS})


def _pair_frame(t_idx, r_idx, alpha, sep, d, xyz, geom, theta_rad) -> pd.DataFrame:
    r_tr = np.linalg.norm(xyz[r_idx] - xyz[t_idx], axis=1)
    return pd.DataFrame({
        "t_idx": t_idx, "r_idx": r_idx, "geometry": geom, "theta_rad": theta_rad,
        "alpha_rad": alpha, "alpha_over_half_theta": alpha / (0.5 * theta_rad),
        "sep_deg": sep, "d_t_pc": d[t_idx], "d_r_pc": d[r_idx], "r_tr_pc": r_tr,
        "flux_ratio_earth_over_receiver": (r_tr / d[t_idx]) ** 2,
    })


# ---------------------------------------------------------------------------
# kinematics of a pair
# ---------------------------------------------------------------------------
def range_rate_and_acceleration(x_a: np.ndarray, v_a: np.ndarray, x_b: np.ndarray,
                                v_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Range rate (km/s) and range acceleration (m/s^2) of B relative to A.

    Straight-line relative motion: ``r_dot = dv . e``, ``r_ddot = |dv_perp|^2 / r``.
    Positions in pc, velocities in km/s.
    """
    dx = np.asarray(x_b, float) - np.asarray(x_a, float)
    dv = np.asarray(v_b, float) - np.asarray(v_a, float)
    r = np.linalg.norm(dx, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = dx / r[..., None]
        v_r = np.sum(dv * e, axis=-1)
        v_perp2 = np.sum(dv * dv, axis=-1) - v_r ** 2
        a = np.maximum(v_perp2, 0.0) * 1e6 / (r * PC_M)   # (km/s)^2 -> m^2/s^2
    return v_r, a


def pair_kinematics(xyz: np.ndarray, vel: np.ndarray, t_idx, r_idx) -> pd.DataFrame:
    """Per directed pair: T--R range rate/acceleration and T--Earth ones.

    ``vel`` rows may be NaN (no RV); the corresponding outputs are NaN and
    ``kinematics_complete`` is False.  The drift prior at frequency f is
    ``(f/c)(a_TR - a_TE_kin)`` with the Earth term added per observation by
    :func:`earth_acceleration_los`.
    """
    t_idx = np.asarray(t_idx, int)
    r_idx = np.asarray(r_idx, int)
    xt, xr = xyz[t_idx], xyz[r_idx]
    vt, vr = vel[t_idx], vel[r_idx]
    v_tr, a_tr = range_rate_and_acceleration(xt, vt, xr, vr)
    zero = np.zeros_like(xt)
    v_te, a_te = range_rate_and_acceleration(xt, vt, zero, zero)   # Earth at rest at origin
    # v_te = (0 - v_t) . (0 - x_t)/|x_t| = v_t . x_t/|x_t| -- the recession speed of T
    complete = np.isfinite(v_tr) & np.isfinite(v_te)
    return pd.DataFrame({
        "v_tr_kms": v_tr, "a_tr_m_s2": a_tr, "v_te_kms": v_te, "a_te_kin_m_s2": a_te,
        "dv_offset_kms": v_tr - v_te,            # frequency offset Earth sees vs R, in v/c
        "a_kin_m_s2": a_tr - a_te,               # the kinematic part of the drift prior
        "kinematics_complete": complete,
    })


def drift_hz_s(freq_hz, accel_m_s2):
    """Doppler drift for an acceleration along the line of sight."""
    return np.asarray(freq_hz, float) / C_M_S * np.asarray(accel_m_s2, float)


# ---------------------------------------------------------------------------
# Earth's acceleration toward a source (the term the pipelines do not remove)
# ---------------------------------------------------------------------------
def sun_direction(mjd) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apparent solar RA/Dec (deg) and Earth--Sun distance (AU), low precision.

    USNO "Approximate Solar Coordinates" (accuracy ~ 1 arcmin over 1950-2050),
    which is all a projected acceleration needs.
    """
    n = np.asarray(mjd, float) - 51544.5
    L = (280.460 + 0.9856474 * n) % 360.0
    g = np.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = np.radians(L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.radians(23.439 - 0.0000004 * n)
    ra = np.degrees(np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))) % 360.0
    dec = np.degrees(np.arcsin(np.sin(eps) * np.sin(lam)))
    dist = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)
    return ra, dec, dist


def gmst_deg(mjd) -> np.ndarray:
    """Greenwich mean sidereal time in degrees (UT1 ~ UTC; 1 s is 15 arcsec)."""
    d = np.asarray(mjd, float) - 51544.5
    hours = (18.697374558 + 24.06570982441908 * d) % 24.0
    return hours * 15.0


def earth_acceleration_los(mjd, site_lat_deg: float, site_lon_deg: float, ra_deg, dec_deg
                           ) -> dict:
    """Earth's acceleration projected onto the source direction (m/s^2, +toward source).

    Rotation: the site accelerates toward the spin axis, so at transit it
    accelerates AWAY from a source in the meridian (negative projection).
    Orbit: Earth accelerates toward the Sun.  The drift a de-drifted relay
    shows is ``(f/c) * a_los``; terrestrial interference shows none.
    """
    mjd = np.asarray(mjd, float)
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    lat = math.radians(site_lat_deg)
    lst = np.radians((gmst_deg(mjd) + site_lon_deg) % 360.0)
    hour_angle = lst - ra
    a_rot = -(OMEGA_EARTH ** 2) * R_EARTH_M * math.cos(lat) * np.cos(dec) * np.cos(hour_angle)
    ra_s, dec_s, dist = sun_direction(mjd)
    u_src = unit_vectors(np.degrees(ra), np.degrees(dec))
    u_sun = unit_vectors(ra_s, dec_s)
    cos_elong = np.sum(u_src * u_sun, axis=-1)
    a_orb = GM_SUN / (dist * AU_M) ** 2 * cos_elong
    return {"a_rot_m_s2": a_rot, "a_orb_m_s2": a_orb, "a_los_m_s2": a_rot + a_orb,
            "hour_angle_deg": np.degrees(hour_angle) % 360.0}


def earth_term_bounds(site_lat_deg: float, dec_deg) -> np.ndarray:
    """|a_los| upper bound when the observation time is unknown (m/s^2)."""
    dec = np.radians(np.asarray(dec_deg, float))
    return (OMEGA_EARTH ** 2 * R_EARTH_M * abs(math.cos(math.radians(site_lat_deg)))
            * np.abs(np.cos(dec)) + GM_SUN / AU_M ** 2 * 1.035)


# ---------------------------------------------------------------------------
# the drift window for a pointing
# ---------------------------------------------------------------------------
def drift_window(freq_hz, a_kin_m_s2, a_earth_m_s2, alpha_rad, *, a_local_max_m_s2: float,
                 sigma_drift_hz_s: float, earth_term_known=True) -> dict:
    """Predicted topocentric drift of a de-drifted relay, and its half-width.

    centre  = (f/c)(a_kin + a_earth)
    tight   = sqrt(sigma^2 + [(f/c) alpha a_local_max]^2)  -- T de-drifts for its own
              platform toward R; the leak toward Earth is the small-angle residual
    loose   = tight (+) (f/c) a_local_max                  -- T also pre-compensates
              R's platform; then R's whole local acceleration reaches Earth
    When the Earth term is unknown (no MJD) its bound is added to both widths and
    the centre carries the kinematic part only.
    """
    f = np.asarray(freq_hz, float)
    a_kin = np.nan_to_num(np.asarray(a_kin_m_s2, float), nan=0.0)
    a_e = np.asarray(a_earth_m_s2, float)
    known = np.asarray(earth_term_known, bool)
    centre = drift_hz_s(f, a_kin + np.where(known, a_e, 0.0))
    leak = drift_hz_s(f, np.asarray(alpha_rad, float) * a_local_max_m_s2)
    tight = np.sqrt(sigma_drift_hz_s ** 2 + leak ** 2)
    loose = np.sqrt(tight ** 2 + drift_hz_s(f, a_local_max_m_s2) ** 2)
    unknown_e = np.where(known, 0.0, drift_hz_s(f, np.abs(a_e)))
    return {"centre_hz_s": centre, "halfwidth_tight_hz_s": tight + unknown_e,
            "halfwidth_loose_hz_s": loose + unknown_e, "leak_hz_s": leak}


# ---------------------------------------------------------------------------
# the "magic frequency" offset prior (supplementary)
# ---------------------------------------------------------------------------
def rest_frequency_at_earth(f_rest_hz, dv_offset_kms, v_bary_kms=0.0):
    """Frequency Earth sees of a line R receives at rest: f_rest (1 + dv/c).

    ``dv_offset_kms = v_TR - v_TE`` from :func:`pair_kinematics`; ``v_bary_kms``
    adds the observatory's barycentric velocity toward the source at the time
    of observation when it is known (topocentric frequency).
    """
    dv = (np.asarray(dv_offset_kms, float) - np.asarray(v_bary_kms, float)) * 1e3
    return np.asarray(f_rest_hz, float) * (1.0 + dv / C_M_S)


__all__ = [
    "ARCMIN", "ARCSEC", "C_M_S", "DEG", "GEOM_BETWEEN", "GEOM_SPILLOVER", "PairSearchResult",
    "analytic_expectation", "angle_label", "beam_grid", "chord", "delta_max_between",
    "delta_max_exact", "delta_max_spillover", "drift_hz_s", "drift_window", "earth_acceleration_los",
    "earth_term_bounds", "find_pairs", "gmst_deg", "pair_kinematics", "positions_pc",
    "range_rate_and_acceleration", "rest_frequency_at_earth", "sky_separation_deg",
    "space_velocities", "sun_direction", "tangential_basis", "theta_diffraction",
    "transmitter_angle", "unit_vectors",
]
