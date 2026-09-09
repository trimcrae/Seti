"""S43 --- the statite: a reflecting point source whose position does not obey Kepler.

Every planet-hunter that looks at a Coronagraph Instrument (CGI) image asks
whether a point source *orbits* the star.  This module asks whether it *fails*
to.  A body at projected separation ``s`` around a star of mass ``M`` is not
free to sit still: at 3 AU from a solar-mass star the free-fall time is under a
year, and the arc it must trace over a season is set by Kepler's third law,
``P = sqrt(a^3 / M)`` years, not by any fitted parameter.  A statite --- a
reflector held against gravity by radiation pressure (Forward 1993), or any
station-kept structure --- sits at a fixed position relative to the star.  At
CGI astrometry (~3 mas per epoch, ``instruments.CGI.astrometric_precision_mas``)
and separations 0.15--0.45 arcsec, a Keplerian companion of a star at 10 pc moves
tens to hundreds of milliarcseconds per year; a fixed one does not.

Three astrometric hypotheses are fitted to the same (dx, dy) time series:

``fixed``    a constant offset --- 2 parameters;
``linear``   a background star: relative proper motion, plus the *imposed*
             parallax ellipse of the foreground star (amplitude 1000/d mas,
             not free) whenever the baseline and the geometry allow it --- 4
             (or 5) parameters.  Without the parallax term the background
             model nests the fixed one and cannot exclude it; the verdict then
             says so instead of pretending;
``kepler``   a bound orbit with the period *locked* to the semi-major axis by
             Kepler's third law --- 6 parameters (a, e, i, Omega, omega, T_peri),
             with ``e <= E_MAX`` (0.8, the conventional planet population) and
             ``a`` bounded to a few times the projected separation.  One
             physical degeneracy is real and is reported rather than hidden: an
             edge-on orbit with e ~ 0.95 seen at its projected turning point
             sits ~10 AU from the star for a 3 AU projected separation and moves
             only a few mas in a season.  Whenever the primary orbit is
             excluded, a second fit with ``e <= E_MAX_EXTREME`` looks for that
             mimic; if it fits, the verdict carries the caution and the
             baseline that would exclude it.

Two supporting tests: the reflectance is **grey** (no methane band depressing
730 nm against 575 nm, as a giant planet shows), and the phase function is
**specular** (narrow) rather than Lambertian.  Polarisation is reported but does
not discriminate on its own (Rayleigh-scattering planets and dust reach the
same degree at quadrature).

Contamination model (``docs/roman.md`` section 2.4): background stars (the
parallax + proper-motion model), disc clumps and speckle residuals (epoch and
polarimetric consistency, left to the vetting stage), and a planet on a wide
nearly face-on orbit whose arc over the baseline is below the astrometric
error --- ``arc_testability`` refuses to judge such a source and the verdict is
``UNRESOLVED``, never a candidate.

Everything here is pure numpy/scipy, offline, and driven by the ``statite:``
block of ``config/roman.yaml``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from .schema import Funnel, json_safe

__all__ = [
    "CGISource",
    "E_MAX",
    "E_MAX_EXTREME",
    "PLANET_TEMPLATE",
    "arc_testability",
    "assess_sources",
    "astrometric_verdict",
    "fit_fixed",
    "fit_kepler",
    "fit_linear",
    "kepler_period_yr",
    "kepler_xy",
    "lambert_phase",
    "parallax_factors",
    "phase_function_test",
    "reflectance_test",
    "screen_source",
    "solve_kepler",
    "synthesise_source",
]

DAYS_PER_YEAR = 365.25
#: MJD of the March equinox 2000 (2000-03-20 07:35 UTC); the zero of the Sun's
#: ecliptic longitude in the simple parallax model.
MJD_EQUINOX_2000 = 51623.32
#: Eccentricity bound of the *primary* Keplerian fit: the conventional planet /
#: wide-companion population (imaged companions cluster below e ~ 0.5; e > 0.8 is
#: exceptional).  ``statite.kepler_e_max`` in config overrides it.
E_MAX = 0.8
#: Eccentricity bound of the *extreme* fit run whenever the primary orbit is
#: excluded.  An edge-on orbit with e ~ 0.95 seen at its projected turning point
#: (r ~ 10 AU for a 3 AU projected separation) moves only a few mas in a season
#: and can mimic a fixed source; the extreme fit finds it and the verdict
#: reports it, with the baseline needed to exclude it, as a caution.
#: ``statite.kepler_e_max_extreme`` overrides it.
E_MAX_EXTREME = 0.95
#: Gravitational parameter of the Sun in AU^3 / yr^2 (4 pi^2).
GM_SUN_AU3_YR2 = 4.0 * math.pi ** 2
#: The Keplerian fit explores ``a`` within these multiples of the projected
#: semi-major axis (median separation converted to AU).
A_BOUNDS_REL = (0.2, 10.0)
#: Specular-lobe half-width bounds, degrees (a flat mirror is narrow).
SPECULAR_W_BOUNDS_DEG = (0.5, 10.0)

#: Relative reflectance of a methane-bearing giant planet by CGI band, normalised
#: to B1.  NOMINAL --- a Jupiter-like geometric-albedo pattern (bright at 575 nm,
#: the 727/790 nm CH4 bands depressing B3, partial recovery in B4); verify against
#: the delivered CGI throughputs and a proper albedo spectrum before use on data.
PLANET_TEMPLATE: dict[str, float] = {"B1": 1.00, "B2": 0.85, "B3": 0.45, "B4": 0.55}


# --------------------------------------------------------------------------------------
# Input structure
# --------------------------------------------------------------------------------------

def _arr(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=float))


@dataclass
class CGISource:
    """One point source in a CGI field: its offsets from the star, epoch by epoch.

    ``dx_mas`` is east (Delta RA cos Dec) and ``dy_mas`` north, both relative to the
    occulted star; ``err_mas`` is the per-epoch (isotropic) astrometric error, a
    scalar or one value per epoch.  ``contrast_by_band`` maps a CGI band name
    (``B1``..``B4``) to ``(contrast, error)`` for the reflectance test;
    ``phase_angle_deg`` with ``contrast_per_epoch`` (and ``contrast_err_per_epoch``)
    feed the phase-function test; ``polarization`` is ``(p, err)``.

    ``meta`` may carry what the astrometric background model needs:
    ``ecl_lat_deg`` (and ``ecl_lon_deg``) for the simple parallax-ellipse
    approximation, or exact per-epoch ``parallax_factor_x`` / ``parallax_factor_y``
    arrays (mas of star displacement per mas of parallax, east/north) which take
    precedence; ``pm_x_masyr`` / ``pm_y_masyr`` (the star's own proper motion) let
    the fitted relative motion be checked against what a background star must
    show; ``simulated: True`` marks a synthetic source.
    """

    source_id: str
    star_id: str
    star_mass_msun: float
    distance_pc: float
    epochs_mjd: np.ndarray
    dx_mas: np.ndarray
    dy_mas: np.ndarray
    err_mas: np.ndarray | float
    contrast_by_band: dict[str, tuple[float, float]] | None = None
    phase_angle_deg: np.ndarray | None = None
    contrast_per_epoch: np.ndarray | None = None
    contrast_err_per_epoch: np.ndarray | None = None
    polarization: tuple[float, float] | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.epochs_mjd = _arr(self.epochs_mjd)
        self.dx_mas = _arr(self.dx_mas)
        self.dy_mas = _arr(self.dy_mas)
        n = self.epochs_mjd.size
        if self.dx_mas.size != n or self.dy_mas.size != n:
            raise ValueError("CGISource epochs, dx and dy must have one length")
        e = _arr(self.err_mas)
        if e.size == 1:
            e = np.full(n, float(e[0]))
        if e.size != n:
            raise ValueError("err_mas must be a scalar or one value per epoch")
        self.err_mas = e
        if self.phase_angle_deg is not None:
            self.phase_angle_deg = _arr(self.phase_angle_deg)
        if self.contrast_per_epoch is not None:
            self.contrast_per_epoch = _arr(self.contrast_per_epoch)
        if self.contrast_err_per_epoch is not None:
            self.contrast_err_per_epoch = _arr(self.contrast_err_per_epoch)

    @property
    def n(self) -> int:
        return int(self.epochs_mjd.size)

    @property
    def span_yr(self) -> float:
        if self.n < 2:
            return 0.0
        return float((np.nanmax(self.epochs_mjd) - np.nanmin(self.epochs_mjd)) / DAYS_PER_YEAR)

    @property
    def separation_mas(self) -> float:
        return float(np.nanmedian(np.hypot(self.dx_mas, self.dy_mas)))

    @property
    def t_yr(self) -> np.ndarray:
        """Years since the first epoch (the reference epoch of every fit)."""
        return (self.epochs_mjd - float(np.nanmin(self.epochs_mjd))) / DAYS_PER_YEAR

    def as_dict(self) -> dict:
        return {"source_id": self.source_id, "star_id": self.star_id,
                "star_mass_msun": self.star_mass_msun, "distance_pc": self.distance_pc,
                "n_epochs": self.n, "span_yr": self.span_yr,
                "separation_mas": self.separation_mas,
                "median_err_mas": float(np.nanmedian(self.err_mas)),
                "bands": sorted(self.contrast_by_band) if self.contrast_by_band else [],
                "has_phase": self.phase_angle_deg is not None,
                "has_polarization": self.polarization is not None,
                "meta": json_safe(dict(self.meta))}


def _conf_block(conf: dict | None) -> dict:
    """The ``statite:`` block with the documented defaults filled in."""
    base = {"min_epochs": 3, "min_span_fraction_of_period": 0.15,
            "fixed_vs_kepler_delta_chi2_min": 25.0, "fixed_vs_linear_delta_chi2_min": 25.0,
            "grey_vs_planet_delta_chi2_min": 16.0, "specular_vs_lambert_delta_chi2_min": 16.0,
            "kepler_e_max": E_MAX, "kepler_e_max_extreme": E_MAX_EXTREME,
            "tiers": {"interest": "NON_KEPLERIAN_PENDING_VET", "candidate": "STATITE_CANDIDATE"}}
    blk = (conf or {}).get("statite") if isinstance(conf, dict) else None
    if isinstance(blk, dict):
        base.update({k: v for k, v in blk.items() if v is not None})
    return base


# --------------------------------------------------------------------------------------
# Kepler
# --------------------------------------------------------------------------------------

def kepler_period_yr(a_au: float, star_mass_msun: float) -> float:
    """Kepler's third law: ``P = sqrt(a^3 / M)`` years for ``a`` in AU, ``M`` in Msun."""
    return float(math.sqrt(max(float(a_au), 0.0) ** 3 / max(float(star_mass_msun), 1e-6)))


def solve_kepler(mean_anom: np.ndarray, e: float, n_iter: int = 40, tol: float = 1e-12) -> np.ndarray:
    """Eccentric anomaly from mean anomaly by Newton iteration (vectorised).

    Starts from ``E = M + e sin M`` (adequate to ``e ~ 0.95``); iterates until the
    largest correction is below ``tol`` or ``n_iter`` steps.
    """
    m = np.mod(np.asarray(mean_anom, dtype=float), 2.0 * np.pi)
    e = float(np.clip(e, 0.0, 0.999))
    ecc = m + e * np.sin(m) if e < 0.8 else np.full_like(m, np.pi)
    for _ in range(n_iter):
        f = ecc - e * np.sin(ecc) - m
        fp = 1.0 - e * np.cos(ecc)
        step = f / fp
        ecc = ecc - step
        if np.max(np.abs(step)) < tol:
            break
    return ecc


def _unpack_orbit(params) -> tuple[float, float, float, float, float, float]:
    """``params`` as a 6-tuple or a dict with the documented keys (angles in radians)."""
    if isinstance(params, dict):
        return (float(params["a_au"]), float(params["e"]), float(params["inc_rad"]),
                float(params["Omega_rad"]), float(params["omega_rad"]), float(params["t_peri_yr"]))
    a, e, inc, big_omega, omega, t_peri = (float(v) for v in params)
    return a, e, inc, big_omega, omega, t_peri


def kepler_xy(t_yr: np.ndarray, params, star_mass_msun: float,
              distance_pc: float) -> tuple[np.ndarray, np.ndarray]:
    """Sky-plane offsets (dx east, dy north, mas) of a Keplerian orbit at times ``t_yr``.

    ``params`` is ``(a_au, e, inc_rad, Omega_rad, omega_rad, t_peri_yr)`` --- or a dict
    with those keys.  The period is not a parameter: ``P = sqrt(a^3/M)``.  The
    projection is Thiele-Innes (dy = A X + F Y, dx = B X + G Y) with ``Omega`` the
    position angle of the ascending node east of north.
    """
    a, e, inc, big_omega, omega, t_peri = _unpack_orbit(params)
    t = np.asarray(t_yr, dtype=float)
    period = kepler_period_yr(a, star_mass_msun)
    mean_anom = 2.0 * np.pi * (t - t_peri) / max(period, 1e-9)
    ecc = solve_kepler(mean_anom, e)
    x = np.cos(ecc) - e                       # along the major axis, units of a
    y = math.sqrt(max(1.0 - e * e, 0.0)) * np.sin(ecc)
    cw, sw = math.cos(omega), math.sin(omega)
    co, so = math.cos(big_omega), math.sin(big_omega)
    ci = math.cos(inc)
    big_a = cw * co - sw * so * ci
    big_b = cw * so + sw * co * ci
    big_f = -sw * co - cw * so * ci
    big_g = -sw * so + cw * co * ci
    scale = a * 1000.0 / max(float(distance_pc), 1e-6)   # AU -> mas
    dy = scale * (big_a * x + big_f * y)
    dx = scale * (big_b * x + big_g * y)
    return dx, dy


# --------------------------------------------------------------------------------------
# Parallax factors for the background model
# --------------------------------------------------------------------------------------

def parallax_factors(src: CGISource) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Per-epoch (fx, fy, how) --- offset of a *background* source from the star, per mas
    of the star's parallax, east and north.

    Exact factors in ``meta['parallax_factor_x'/'_y']`` take precedence.  Otherwise a
    simple ecliptic approximation from ``meta['ecl_lat_deg']`` (and ``ecl_lon_deg``):
    the star's parallactic ellipse has semi-axes ``pi`` along ecliptic longitude and
    ``pi sin(beta)`` along latitude, and the x/y axes are taken as the ecliptic
    axes (the frame rotation at the source position is ignored --- ``how`` says so).
    With no longitude the ellipse phase is left free to the fitter (``how`` ends in
    ``free_phase``).  ``None`` when neither is available.
    """
    meta = src.meta or {}
    fx, fy = meta.get("parallax_factor_x"), meta.get("parallax_factor_y")
    if fx is not None and fy is not None:
        fx, fy = _arr(fx), _arr(fy)
        if fx.size == src.n and fy.size == src.n:
            return fx, fy, "exact_factors_from_meta"
    beta = meta.get("ecl_lat_deg")
    if beta is None:
        return None
    lam_sun = 2.0 * np.pi * (src.epochs_mjd - MJD_EQUINOX_2000) / DAYS_PER_YEAR
    if meta.get("ecl_lon_deg") is not None:
        lam = math.radians(float(meta["ecl_lon_deg"]))
        # The star is displaced by (+sin(lam_sun - lam), -sin(beta) cos(lam_sun - lam));
        # a background source's offset from it is the negative of that.
        fx = -np.sin(lam_sun - lam)
        fy = math.sin(math.radians(float(beta))) * np.cos(lam_sun - lam)
        return fx, fy, "ecliptic_approx_axes_unrotated"
    # Phase unknown: return the two quadrature bases; the fitter chooses the phase.
    fx = np.stack([-np.sin(lam_sun), -np.cos(lam_sun)])
    fy = math.sin(math.radians(float(beta))) * np.stack([np.cos(lam_sun), -np.sin(lam_sun)])
    return fx, fy, "ecliptic_approx_axes_unrotated_free_phase"


# --------------------------------------------------------------------------------------
# The three astrometric fits
# --------------------------------------------------------------------------------------

def _chi2(dx_model, dy_model, src: CGISource) -> float:
    r = np.concatenate([(src.dx_mas - dx_model) / src.err_mas, (src.dy_mas - dy_model) / src.err_mas])
    return float(np.sum(r * r))


def fit_fixed(src: CGISource) -> dict:
    """Constant offset: the error-weighted mean position.  2 parameters."""
    w = 1.0 / np.maximum(src.err_mas, 1e-9) ** 2
    x0 = float(np.sum(w * src.dx_mas) / np.sum(w))
    y0 = float(np.sum(w * src.dy_mas) / np.sum(w))
    chi2 = _chi2(x0, y0, src)
    return {"model": "fixed", "chi2": chi2, "n_params": 2, "n_data": 2 * src.n,
            "params": {"dx_mas": x0, "dy_mas": y0,
                       "dx_err_mas": float(1.0 / math.sqrt(np.sum(w))),
                       "dy_err_mas": float(1.0 / math.sqrt(np.sum(w)))},
            "notes": []}


def _weighted_lstsq(design: np.ndarray, target: np.ndarray, sigma: np.ndarray):
    aw = design / sigma[:, None]
    bw = target / sigma
    coef, *_ = np.linalg.lstsq(aw, bw, rcond=None)
    resid = bw - aw @ coef
    try:
        cov = np.linalg.inv(aw.T @ aw)
    except np.linalg.LinAlgError:
        cov = np.full((design.shape[1], design.shape[1]), np.nan)
    return coef, float(np.sum(resid * resid)), cov


def fit_linear(src: CGISource, min_parallax_span_yr: float = 0.5) -> dict:
    """A background star: ``dx = a + b t``, ``dy = c + d t`` (relative proper motion) plus,
    when the span is at least ``min_parallax_span_yr`` and the geometry is known, the
    parallax ellipse of the foreground star at its *fixed* amplitude 1000/d mas.

    4 parameters, or 5 when the ellipse phase has to be fitted (no ecliptic
    longitude).  ``params['parallax_imposed']`` records whether the term was applied;
    without it the model nests the fixed one and the verdict logic treats the
    background hypothesis as *not excluded* rather than as beaten.
    """
    t = src.t_yr
    n = src.n
    notes: list[str] = []
    plx = 1000.0 / max(float(src.distance_pc), 1e-6)
    factors = parallax_factors(src) if src.span_yr >= min_parallax_span_yr else None
    if src.span_yr < min_parallax_span_yr:
        notes.append(f"parallax_term_skipped: span {src.span_yr:.2f} yr < {min_parallax_span_yr} yr")
    elif factors is None:
        notes.append("parallax_term_skipped: no ecl_lat_deg or parallax factors in meta")

    design = np.zeros((2 * n, 4))
    design[:n, 0] = 1.0
    design[:n, 1] = t
    design[n:, 2] = 1.0
    design[n:, 3] = t
    sigma = np.concatenate([src.err_mas, src.err_mas])
    data = np.concatenate([src.dx_mas, src.dy_mas])

    def solve(offset: np.ndarray):
        return _weighted_lstsq(design, data - offset, sigma)

    n_params = 4
    phase_deg = None
    how = None
    if factors is None:
        coef, chi2, cov = solve(np.zeros(2 * n))
        imposed = False
    else:
        fx, fy, how = factors
        imposed = True
        if fx.ndim == 1:
            coef, chi2, cov = solve(plx * np.concatenate([fx, fy]))
        else:
            n_params = 5

            def chi2_at(phi: float) -> float:
                off = plx * np.concatenate([math.cos(phi) * fx[0] + math.sin(phi) * fx[1],
                                            math.cos(phi) * fy[0] + math.sin(phi) * fy[1]])
                return solve(off)[1]

            grid = np.linspace(0.0, 2.0 * np.pi, 73)[:-1]
            vals = np.array([chi2_at(p) for p in grid])
            k = int(np.argmin(vals))
            lo, hi = grid[k] - 2.0 * np.pi / 72, grid[k] + 2.0 * np.pi / 72
            res = minimize_scalar(chi2_at, bounds=(lo, hi), method="bounded")
            phi = float(res.x) if res.success and res.fun <= vals[k] else float(grid[k])
            off = plx * np.concatenate([math.cos(phi) * fx[0] + math.sin(phi) * fx[1],
                                        math.cos(phi) * fy[0] + math.sin(phi) * fy[1]])
            coef, chi2, cov = solve(off)
            phase_deg = math.degrees(phi) % 360.0
            notes.append("parallax_phase_fitted: ecliptic longitude unknown")
        notes.append(f"parallax_imposed: amplitude {plx:.2f} mas, {how}")
    err = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    params = {"dx0_mas": float(coef[0]), "pm_x_masyr": float(coef[1]),
              "dy0_mas": float(coef[2]), "pm_y_masyr": float(coef[3]),
              "pm_x_err_masyr": float(err[1]), "pm_y_err_masyr": float(err[3]),
              "parallax_mas": plx, "parallax_imposed": imposed, "parallax_model": how,
              "parallax_phase_deg": phase_deg}
    return {"model": "linear", "chi2": float(chi2), "n_params": n_params, "n_data": 2 * n,
            "params": params, "nests_fixed": not imposed, "notes": notes}


def _circular_faceon_fit(src: CGISource) -> dict:
    """Fallback for very short series: a circular face-on orbit, (a, phase), both senses."""
    t = src.t_yr
    d = float(src.distance_pc)
    m = float(src.star_mass_msun)
    a_proj = src.separation_mas * d / 1000.0
    best = None
    for sense in (+1.0, -1.0):
        def resid(p, sense=sense):
            a, phi = p
            per = kepler_period_yr(a, m)
            ang = phi + sense * 2.0 * np.pi * t / max(per, 1e-9)
            s = a * 1000.0 / d
            return np.concatenate([(src.dx_mas - s * np.sin(ang)) / src.err_mas,
                                   (src.dy_mas - s * np.cos(ang)) / src.err_mas])

        phi0 = math.atan2(float(src.dx_mas[0]), float(src.dy_mas[0]))
        for dphi in (0.0, 0.5, -0.5):
            try:
                r = least_squares(resid, [max(a_proj, 1e-3), phi0 + dphi],
                                  bounds=([A_BOUNDS_REL[0] * a_proj, -4 * np.pi],
                                          [A_BOUNDS_REL[1] * a_proj, 4 * np.pi]),
                                  max_nfev=200)
            except ValueError:
                continue
            c2 = float(np.sum(r.fun ** 2))
            if best is None or c2 < best[0]:
                best = (c2, float(r.x[0]), float(r.x[1]), sense)
    if best is None:
        return {"model": "kepler", "kepler_model": "circular_faceon", "chi2": float("nan"),
                "n_params": 2, "n_data": 2 * src.n, "params": {}, "notes": ["fit_failed"]}
    c2, a, phi, sense = best
    return {"model": "kepler", "kepler_model": "circular_faceon", "chi2": c2, "n_params": 2,
            "n_data": 2 * src.n,
            "params": {"a_au": a, "period_yr": kepler_period_yr(a, m), "phase_deg": math.degrees(phi) % 360.0,
                       "sense": "prograde_ccw" if sense > 0 else "retrograde_cw", "e": 0.0, "inc_deg": 0.0},
            "notes": ["n_epochs < 4: face-on circular family only (a, phase); "
                      "inclination and eccentricity not constrained"]}


def _orbit_radius_au(p: dict, t_yr: np.ndarray) -> np.ndarray:
    """Three-dimensional star-companion distance of a fitted orbit at ``t_yr``."""
    per = max(float(p["period_yr"]), 1e-9)
    mean_anom = 2.0 * np.pi * (np.asarray(t_yr, dtype=float) - float(p["t_peri_yr"])) / per
    ecc = solve_kepler(mean_anom, float(p["e"]))
    return float(p["a_au"]) * (1.0 - float(p["e"]) * np.cos(ecc))


def fit_kepler(src: CGISource, e_max: float = E_MAX, a_bounds_rel: tuple[float, float] = A_BOUNDS_REL,
               n_starts: int | None = None, coarse_nfev: int = 20, polish_nfev: int = 150,
               n_polish: int = 3) -> dict:
    """A Keplerian orbit around a star of mass ``M`` at distance ``d``: (a_AU, e, i, Omega,
    omega, T_peri) with the period locked by Kepler's third law.  6 parameters.

    Multi-start ``scipy.optimize.least_squares``: a coarse pass (``coarse_nfev``
    evaluations) from a grid of inclinations (0..180 deg, both senses), two
    eccentricities and four orbital phases with the node at the first-epoch
    position angle, then a polish of the ``n_polish`` best starts.  Bounds:
    ``e in [0, e_max]``, ``a`` within ``a_bounds_rel`` of the projected semi-major
    axis.  Reports the best fit, the starts that converged, and whether the
    solution is pinned at a bound (``params['at_bound']``).  With fewer than 4
    epochs (8 data values are the floor at which six parameters say anything) the
    face-on circular family is fitted instead and ``kepler_model`` says
    ``circular_faceon``.
    """
    if src.n < 4:
        return _circular_faceon_fit(src)
    t = src.t_yr
    d = float(src.distance_pc)
    m = float(src.star_mass_msun)
    e_max = float(np.clip(e_max, 0.0, 0.999))
    a_proj = max(src.separation_mas * d / 1000.0, 1e-4)
    lo = np.array([a_bounds_rel[0] * a_proj, 0.0, 0.0, -2 * np.pi, -2 * np.pi, -2 * np.pi])
    hi = np.array([a_bounds_rel[1] * a_proj, e_max, np.pi, 4 * np.pi, 4 * np.pi, 4 * np.pi])

    def model(p):
        a, e, inc, big_omega, omega, m0 = p
        per = kepler_period_yr(a, m)
        t_peri = -m0 * per / (2.0 * np.pi)
        return kepler_xy(t, (a, e, inc, big_omega, omega, t_peri), m, d)

    def resid(p):
        mx, my = model(p)
        return np.concatenate([(src.dx_mas - mx) / src.err_mas, (src.dy_mas - my) / src.err_mas])

    def run(x0, nfev):
        x0 = np.clip(np.asarray(x0, dtype=float), lo + 1e-9, hi - 1e-9)
        try:
            r = least_squares(resid, x0, bounds=(lo, hi), max_nfev=nfev,
                              x_scale=[a_proj, 0.3, 1.0, 1.0, 1.0, 1.0])
        except ValueError:
            return None
        return float(np.sum(r.fun ** 2)), r.x

    pa0 = math.atan2(float(src.dx_mas[0]), float(src.dy_mas[0]))
    incs = np.radians([0.0, 45.0, 90.0, 135.0, 180.0])
    e_starts = (0.0, min(0.75 * e_max, e_max - 0.01))
    starts = [(a_proj, e0, i0, pa0, 0.0, m0)
              for i0 in incs for e0 in e_starts for m0 in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)]
    if n_starts is not None:
        starts = starts[: max(1, int(n_starts))]
    coarse = [r for r in (run(x0, coarse_nfev) for x0 in starts) if r is not None]
    n_ok = len(coarse)
    if not coarse:
        return {"model": "kepler", "kepler_model": "full", "chi2": float("nan"), "n_params": 6,
                "n_data": 2 * src.n, "params": {}, "notes": ["fit_failed"], "e_max": e_max}
    coarse.sort(key=lambda r: r[0])
    best = coarse[0]
    for _c2, x in coarse[: max(1, int(n_polish))]:
        r = run(x, polish_nfev)
        if r is not None and r[0] < best[0]:
            best = r
    c2, p = best
    a, e, inc, big_omega, omega, m0 = (float(v) for v in p)
    per = kepler_period_yr(a, m)
    at_bound = []
    if e >= e_max - 1e-3:
        at_bound.append("e_max")
    if a <= lo[0] * (1 + 1e-3):
        at_bound.append("a_min")
    if a >= hi[0] * (1 - 1e-3):
        at_bound.append("a_max")
    notes = [f"multi_start: {n_ok}/{len(starts)} converged; bounds e<={e_max}, "
             f"a in [{lo[0]:.3g}, {hi[0]:.3g}] AU"]
    if at_bound:
        notes.append("solution_at_bound: " + ",".join(at_bound))
    params = {"a_au": a, "e": e, "inc_deg": math.degrees(inc), "Omega_deg": math.degrees(big_omega) % 360.0,
              "omega_deg": math.degrees(omega) % 360.0, "t_peri_yr": -m0 * per / (2.0 * np.pi),
              "mean_anomaly_t0_deg": math.degrees(m0) % 360.0, "period_yr": per,
              "inc_rad": inc, "Omega_rad": big_omega % (2 * np.pi), "omega_rad": omega % (2 * np.pi),
              "at_bound": at_bound}
    params["r_mid_au"] = float(_orbit_radius_au(params, np.array([0.5 * (t.min() + t.max())]))[0])
    return {"model": "kepler", "kepler_model": "full", "chi2": c2, "n_params": 6, "n_data": 2 * src.n,
            "params": params, "e_max": e_max, "notes": notes}


# --------------------------------------------------------------------------------------
# Testability and the astrometric verdict
# --------------------------------------------------------------------------------------

def arc_testability(src: CGISource, conf: dict | None = None) -> dict:
    """Is the observed arc long enough for Kepler to be testable at all?

    For a circular face-on orbit at the observed separation the expected motion
    over the span is ``2 pi (span / P) x separation``; the source is ``testable``
    when ``span / P >= min_span_fraction_of_period``, the expected motion is at
    least three times the median astrometric error, and there are at least
    ``min_epochs`` epochs.  An untestable source is never a candidate.
    """
    c = _conf_block(conf)
    sep = src.separation_mas
    a_proj = sep * float(src.distance_pc) / 1000.0
    per = kepler_period_yr(a_proj, float(src.star_mass_msun))
    span = src.span_yr
    frac = span / per if per > 0 else float("inf")
    expected = 2.0 * np.pi * frac * sep
    med_err = float(np.nanmedian(src.err_mas))
    reasons = []
    if src.n < int(c["min_epochs"]):
        reasons.append(f"too_few_epochs: {src.n} < {int(c['min_epochs'])}")
    if frac < float(c["min_span_fraction_of_period"]):
        reasons.append(f"arc_too_short: span/P = {frac:.3f} < {float(c['min_span_fraction_of_period'])}")
    if expected < 3.0 * med_err:
        reasons.append(f"motion_below_error: expected {expected:.2f} mas < 3 x {med_err:.2f} mas")
    return {"testable": not reasons, "reasons": reasons, "separation_mas": sep,
            "a_projected_au": a_proj, "period_faceon_circular_yr": per, "span_yr": span,
            "span_fraction_of_period": frac, "expected_motion_mas": float(expected),
            "median_err_mas": med_err, "n_epochs": src.n}


def _background_pm_check(src: CGISource, lin: dict) -> dict | None:
    """A background star's relative proper motion must be about minus the star's own.

    Uses ``meta['pm_x_masyr'/'pm_y_masyr']``; ``None`` when the star's motion is not
    known.  ``excluded`` when the fitted relative motion differs from ``-pm_star`` by
    more than ``own_pm_allowance`` plus 5 sigma (a background star has some proper
    motion of its own; 30 mas/yr is generous for anything distant).
    """
    meta = src.meta or {}
    if meta.get("pm_x_masyr") is None or meta.get("pm_y_masyr") is None:
        return None
    p = lin["params"]
    exp_x, exp_y = -float(meta["pm_x_masyr"]), -float(meta["pm_y_masyr"])
    dxp, dyp = p["pm_x_masyr"] - exp_x, p["pm_y_masyr"] - exp_y
    err = math.hypot(max(p["pm_x_err_masyr"], 1e-9), max(p["pm_y_err_masyr"], 1e-9))
    allowance = 30.0
    diff = math.hypot(dxp, dyp)
    return {"expected_relative_pm_masyr": [exp_x, exp_y],
            "fitted_relative_pm_masyr": [p["pm_x_masyr"], p["pm_y_masyr"]],
            "difference_masyr": diff, "own_pm_allowance_masyr": allowance,
            "excluded": bool(diff > allowance + 5.0 * err)}


def _extreme_orbit_caution(src: CGISource, fixed: dict, thr: float, e_max_extreme: float) -> dict:
    """The one Keplerian family that can mimic a fixed source, looked for explicitly.

    Refits with ``e <= e_max_extreme``.  ``excluded`` when even that orbit is worse
    than the fixed model by ``thr``.  Otherwise the sky-plane acceleration of the
    mimic at mid-epoch, ``GM s / r^3`` (``s`` the projected, ``r`` the true
    separation), gives the baseline over which its turning-point excursion would
    exceed the fixed model by ``thr`` --- an estimate (parabolic arc, uniform
    sampling) of what a longer CGI campaign needs to settle the source.
    """
    kx = fit_kepler(src, e_max=e_max_extreme)
    d = float(kx["chi2"] - fixed["chi2"])
    out: dict[str, Any] = {"fit": kx, "e_max_extreme": e_max_extreme, "delta_chi2": d,
                           "excluded": bool(np.isfinite(d) and d >= thr), "summary": ""}
    p = kx.get("params") or {}
    if out["excluded"] or not p:
        out["summary"] = f"extreme orbit (e<={e_max_extreme}) worse than fixed by {d:.1f}"
        return out
    r_mid = float(p.get("r_mid_au", np.nan))
    s_au = src.separation_mas * float(src.distance_pc) / 1000.0
    gm = GM_SUN_AU3_YR2 * float(src.star_mass_msun)
    acc_mas_yr2 = gm * s_au / max(r_mid, 1e-6) ** 3 * 1000.0 / float(src.distance_pc)
    sigma = float(np.nanmedian(src.err_mas))
    # chi2 of a parabola of end-excursion E against a constant, ~0.1 n (E/sigma)^2.
    need_excursion = sigma * math.sqrt(thr / (0.1 * max(src.n, 1)))
    span_needed = 2.0 * math.sqrt(2.0 * need_excursion / max(acc_mas_yr2, 1e-12))
    out.update({"r_mid_au": r_mid, "sky_acceleration_mas_yr2": float(acc_mas_yr2),
                "span_needed_yr_estimate": float(span_needed)})
    out["summary"] = (f"e={p.get('e', float('nan')):.2f}, i={p.get('inc_deg', float('nan')):.0f} deg, "
                      f"a={p.get('a_au', float('nan')):.1f} AU, r_mid={r_mid:.1f} AU fits within "
                      f"dchi2={d:.1f}; a baseline of ~{span_needed:.1f} yr would exclude it")
    return out


def astrometric_verdict(src: CGISource, conf: dict | None = None) -> dict:
    """Fixed vs background vs Kepler on one source.

    ``UNRESOLVED`` when the arc is not testable (with the numbers).  Otherwise the
    three fits are compared:

    * ``FIXED_PREFERRED`` --- the Keplerian fit is worse than the fixed one by
      ``fixed_vs_kepler_delta_chi2_min`` *and* the background hypothesis is
      excluded, either by the imposed parallax term (linear worse than fixed by
      ``fixed_vs_linear_delta_chi2_min``) or, when no parallax could be imposed, by
      the fitted relative proper motion being incompatible with the star's own;
    * ``KEPLERIAN`` --- the orbit is the best model and beats fixed by the threshold;
    * ``BACKGROUND`` --- the linear model is the best and beats fixed by the threshold;
    * ``UNRESOLVED`` otherwise, with the reason (``background_not_excluded`` when the
      linear model nests the fixed one and nothing else can separate them).

    ``FIXED_PREFERRED`` is a statement about orbits with ``e <= kepler_e_max``.  When
    the primary orbit is excluded, ``extreme_orbit`` records whether an orbit up to
    ``kepler_e_max_extreme`` still fits, and the baseline that would exclude it; the
    screen carries that as a caution for vetting, not as a demotion.
    """
    c = _conf_block(conf)
    test = arc_testability(src, conf)
    out: dict[str, Any] = {"source_id": src.source_id, "testability": test, "fits": {},
                           "delta_chi2": {}, "thresholds": {
                               "kepler": float(c["fixed_vs_kepler_delta_chi2_min"]),
                               "linear": float(c["fixed_vs_linear_delta_chi2_min"])},
                           "notes": [], "background_excluded_by": None}
    if not test["testable"]:
        out["verdict"] = "UNRESOLVED"
        out["notes"].append("arc_untestable: " + "; ".join(test["reasons"]))
        return out
    fx = fit_fixed(src)
    ln = fit_linear(src)
    kp = fit_kepler(src, e_max=float(c["kepler_e_max"]))
    out["fits"] = {"fixed": fx, "linear": ln, "kepler": kp}
    d_kep = kp["chi2"] - fx["chi2"]
    d_lin = ln["chi2"] - fx["chi2"]
    out["delta_chi2"] = {"kepler_minus_fixed": float(d_kep), "linear_minus_fixed": float(d_lin),
                         "kepler_minus_linear": float(kp["chi2"] - ln["chi2"])}
    thr_k, thr_l = out["thresholds"]["kepler"], out["thresholds"]["linear"]
    kepler_excluded = bool(np.isfinite(d_kep) and d_kep >= thr_k)
    out["extreme_orbit"] = None
    if kepler_excluded and kp.get("kepler_model") == "full":
        out["extreme_orbit"] = _extreme_orbit_caution(src, fx, thr_k, float(c["kepler_e_max_extreme"]))
        out["fits"]["kepler_extreme"] = out["extreme_orbit"].pop("fit")
        out["delta_chi2"]["kepler_extreme_minus_fixed"] = out["extreme_orbit"]["delta_chi2"]
        if not out["extreme_orbit"]["excluded"]:
            out["notes"].append("extreme_orbit_mimic_not_excluded: " + out["extreme_orbit"]["summary"])
    if ln.get("nests_fixed"):
        out["notes"].append("linear_model_nests_fixed: no parallax term could be imposed, so a "
                            "background star is not excluded by chi2")
        pm = _background_pm_check(src, ln)
        out["background_pm_check"] = pm
        if pm and pm["excluded"]:
            background_excluded = True
            out["background_excluded_by"] = "proper_motion_vs_star"
        else:
            background_excluded = False
    else:
        background_excluded = bool(d_lin >= thr_l)
        if background_excluded:
            out["background_excluded_by"] = "imposed_parallax"
        out["background_pm_check"] = _background_pm_check(src, ln)
    chi2s = {"fixed": fx["chi2"], "linear": ln["chi2"], "kepler": kp["chi2"]}
    best = min(chi2s, key=lambda k: chi2s[k] if np.isfinite(chi2s[k]) else np.inf)
    if kepler_excluded and background_excluded:
        verdict = "FIXED_PREFERRED"
    elif best == "kepler" and (fx["chi2"] - kp["chi2"]) >= thr_k:
        verdict = "KEPLERIAN"
    elif best == "linear" and (fx["chi2"] - ln["chi2"]) >= thr_l:
        verdict = "BACKGROUND"
    else:
        verdict = "UNRESOLVED"
        if kepler_excluded and not background_excluded:
            out["notes"].append("background_not_excluded")
        elif not kepler_excluded:
            out["notes"].append("kepler_not_excluded")
    if kp["params"].get("at_bound"):
        out["notes"].append("kepler_best_fit_at_bound: " + ",".join(kp["params"]["at_bound"]))
    out["best_model"] = best
    out["verdict"] = verdict
    return out


# --------------------------------------------------------------------------------------
# Reflectance and phase function
# --------------------------------------------------------------------------------------

def reflectance_test(contrast_by_band: dict[str, tuple[float, float]] | None,
                     conf: dict | None = None,
                     template: dict[str, float] | None = None) -> dict:
    """Grey reflector (one free contrast) versus a methane-bearing giant-planet template
    (``PLANET_TEMPLATE`` with one free scale).

    ``delta_chi2 = chi2_planet - chi2_grey``; ``GREY_PREFERRED`` at or above
    ``grey_vs_planet_delta_chi2_min``, ``PLANET_LIKE`` when grey is worse by the same
    margin, else ``UNRESOLVED``.  Bands the template does not know are skipped and
    named; fewer than two usable bands gives ``UNRESOLVED`` with
    ``insufficient_bands`` (a single band says nothing about colour --- the
    single-band rule of the contamination ledger).
    """
    c = _conf_block(conf)
    tpl = dict(template or PLANET_TEMPLATE)
    thr = float(c["grey_vs_planet_delta_chi2_min"])
    out: dict[str, Any] = {"test": "reflectance", "verdict": "UNRESOLVED", "chi2_grey": None,
                           "chi2_planet": None, "delta_chi2": None, "threshold": thr,
                           "bands_used": [], "notes": [], "template": tpl}
    if not contrast_by_band:
        out["notes"].append("insufficient_bands: none")
        return out
    bands, vals, errs, refl = [], [], [], []
    for b, pair in contrast_by_band.items():
        key = str(b).upper()
        if key not in tpl:
            out["notes"].append(f"band_not_in_template: {b}")
            continue
        try:
            v, e = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError):
            out["notes"].append(f"bad_contrast_entry: {b}")
            continue
        if not (np.isfinite(v) and np.isfinite(e) and e > 0):
            out["notes"].append(f"bad_contrast_entry: {b}")
            continue
        bands.append(key)
        vals.append(v)
        errs.append(e)
        refl.append(float(tpl[key]))
    if len(bands) < 2:
        out["notes"].append(f"insufficient_bands: {len(bands)} usable")
        out["bands_used"] = bands
        return out
    v, e, r = np.array(vals), np.array(errs), np.array(refl)
    by_band = dict(zip(bands, vals, strict=True))
    w = 1.0 / e ** 2
    g = float(np.sum(w * v) / np.sum(w))
    chi2_grey = float(np.sum(((v - g) / e) ** 2))
    s = float(np.sum(w * v * r) / np.sum(w * r * r))
    chi2_planet = float(np.sum(((v - s * r) / e) ** 2))
    d = chi2_planet - chi2_grey
    out.update({"chi2_grey": chi2_grey, "chi2_planet": chi2_planet, "delta_chi2": d,
                "n_params": {"grey": 1, "planet": 1}, "bands_used": bands,
                "params": {"grey_contrast": g, "planet_scale": s,
                           "ratio_B3_B1": (float(by_band["B3"] / by_band["B1"])
                                           if "B1" in by_band and "B3" in by_band else None)}})
    if d >= thr:
        out["verdict"] = "GREY_PREFERRED"
    elif d <= -thr:
        out["verdict"] = "PLANET_LIKE"
    return out


def lambert_phase(alpha_rad: np.ndarray) -> np.ndarray:
    """Lambertian-sphere phase function ``(sin a + (pi - a) cos a) / pi``, 1 at a = 0."""
    a = np.asarray(alpha_rad, dtype=float)
    return (np.sin(a) + (np.pi - a) * np.cos(a)) / np.pi


def phase_function_test(phase_angle_deg, contrast, err, conf: dict | None = None) -> dict:
    """Lambertian sphere (free scale) versus a specular flat mirror --- a Gaussian lobe
    ``S exp(-(a - a0)^2 / 2 w^2)`` with ``a0`` and ``w`` free, ``w <= 10 deg``.

    ``delta_chi2 = chi2_lambert - chi2_specular``; ``SPECULAR_PREFERRED`` at or above
    ``specular_vs_lambert_delta_chi2_min``, ``LAMBERTIAN`` when the mirror is worse by
    that margin, else ``UNRESOLVED``.  The specular model has two more parameters;
    the threshold is meant to absorb that.  Needs at least four epochs spanning at
    least 30 deg of phase, else ``insufficient_phase_coverage``.
    """
    c = _conf_block(conf)
    thr = float(c["specular_vs_lambert_delta_chi2_min"])
    out: dict[str, Any] = {"test": "phase_function", "verdict": "UNRESOLVED", "chi2_lambert": None,
                           "chi2_specular": None, "delta_chi2": None, "threshold": thr, "notes": []}
    if phase_angle_deg is None or contrast is None:
        out["notes"].append("insufficient_phase_coverage: no phase-resolved contrast")
        return out
    al = _arr(phase_angle_deg)
    cv = _arr(contrast)
    ev = _arr(err) if err is not None else np.full(al.size, np.nan)
    if ev.size == 1:
        ev = np.full(al.size, float(ev[0]))
    good = np.isfinite(al) & np.isfinite(cv) & np.isfinite(ev) & (ev > 0)
    al, cv, ev = al[good], cv[good], ev[good]
    span = float(np.ptp(al)) if al.size else 0.0
    if al.size < 4 or span < 30.0:
        out["notes"].append(f"insufficient_phase_coverage: {al.size} epochs over {span:.1f} deg")
        return out
    a_rad = np.radians(al)
    w = 1.0 / ev ** 2
    phi = lambert_phase(a_rad)
    s_l = float(np.sum(w * cv * phi) / max(np.sum(w * phi * phi), 1e-30))
    chi2_l = float(np.sum(((cv - s_l * phi) / ev) ** 2))

    def spec_chi2(a0: float, wd: float):
        f = np.exp(-0.5 * ((al - a0) / wd) ** 2)
        den = np.sum(w * f * f)
        s = float(np.sum(w * cv * f) / den) if den > 0 else 0.0
        return float(np.sum(((cv - s * f) / ev) ** 2)), s

    best = None
    for a0 in np.arange(0.0, 180.5, 1.0):
        for wd in (0.5, 1.0, 2.0, 4.0, 7.0, 10.0):
            c2, s = spec_chi2(a0, wd)
            if best is None or c2 < best[0]:
                best = (c2, a0, wd, s)

    def resid(p):
        s, a0, wd = p
        return (cv - s * np.exp(-0.5 * ((al - a0) / wd) ** 2)) / ev

    lo, hi = SPECULAR_W_BOUNDS_DEG
    try:
        r = least_squares(resid, [best[3], best[1], best[2]],
                          bounds=([-np.inf, 0.0, lo], [np.inf, 180.0, hi]), max_nfev=200)
        c2 = float(np.sum(r.fun ** 2))
        if c2 < best[0]:
            best = (c2, float(r.x[1]), float(r.x[2]), float(r.x[0]))
    except ValueError:
        pass
    chi2_s, a0, wd, s_s = best
    d = chi2_l - chi2_s
    out.update({"chi2_lambert": chi2_l, "chi2_specular": chi2_s, "delta_chi2": float(d),
                "n_params": {"lambert": 1, "specular": 3}, "n_epochs": int(al.size),
                "phase_span_deg": span,
                "params": {"lambert_scale": s_l, "specular_scale": s_s, "specular_alpha0_deg": a0,
                           "specular_width_deg": wd}})
    if d >= thr:
        out["verdict"] = "SPECULAR_PREFERRED"
    elif d <= -thr:
        out["verdict"] = "LAMBERTIAN"
    return out


# --------------------------------------------------------------------------------------
# Screening and assessment
# --------------------------------------------------------------------------------------

def screen_source(src: CGISource, conf: dict | None = None) -> dict:
    """Every test on one source, and its tier.

    ``candidate`` (``tiers.candidate``) only when the astrometry is ``FIXED_PREFERRED``
    *and* at least one of the physical tests supports a reflector (``GREY_PREFERRED``
    or ``SPECULAR_PREFERRED``); ``interest`` (``tiers.interest``) for
    ``FIXED_PREFERRED`` alone; ``UNRESOLVED`` when the arc could not be tested or the
    models were not separated; ``NO_ANOMALY`` when a Keplerian orbit or a background
    star fits.  ``rejections`` names what demoted the source; ``cautions`` names
    supporting tests that went the planet's way without demoting a fixed source
    (a body that does not move is anomalous whatever its colour).
    """
    c = _conf_block(conf)
    tiers = dict(c.get("tiers") or {})
    astro = astrometric_verdict(src, conf)
    refl = (reflectance_test(src.contrast_by_band, conf) if src.contrast_by_band
            else {"test": "reflectance", "verdict": "NOT_RUN", "notes": ["no_multiband_contrast"]})
    if src.phase_angle_deg is not None and src.contrast_per_epoch is not None:
        phase = phase_function_test(src.phase_angle_deg, src.contrast_per_epoch,
                                    src.contrast_err_per_epoch, conf)
    else:
        phase = {"test": "phase_function", "verdict": "NOT_RUN", "notes": ["no_phase_resolved_contrast"]}
    pol: dict[str, Any] = {"test": "polarization", "measured": src.polarization is not None,
                           "notes": ["polarization alone does not discriminate: Rayleigh-scattering "
                                     "planets and dust reach p ~ 0.3-0.5 near quadrature"]}
    if src.polarization is not None:
        p, pe = float(src.polarization[0]), float(src.polarization[1])
        pol.update({"p": p, "p_err": pe, "high_polarization": bool(p > 0.3)})
    rejections: list[str] = []
    cautions: list[str] = []
    av = astro["verdict"]
    if av == "KEPLERIAN":
        rejections.append("keplerian_orbit_fits")
    elif av == "BACKGROUND":
        rejections.append("background_star_fits")
    elif av == "UNRESOLVED":
        rejections.append("arc_untestable" if not astro["testability"]["testable"] else "models_not_separated")
    ext = astro.get("extreme_orbit")
    if ext and not ext.get("excluded"):
        cautions.append("extreme_orbit_mimic_not_excluded")
    if refl.get("verdict") == "PLANET_LIKE":
        cautions.append("planet_like_reflectance")
    if phase.get("verdict") == "LAMBERTIAN":
        cautions.append("lambertian_phase_function")
    supporting = [t for t, v in (("grey_reflectance", refl.get("verdict")),
                                 ("specular_phase", phase.get("verdict")))
                  if v in ("GREY_PREFERRED", "SPECULAR_PREFERRED")]
    if av == "FIXED_PREFERRED" and supporting:
        tier = str(tiers.get("candidate", "STATITE_CANDIDATE"))
    elif av == "FIXED_PREFERRED":
        tier = str(tiers.get("interest", "NON_KEPLERIAN_PENDING_VET"))
    elif av == "UNRESOLVED":
        tier = "UNRESOLVED"
    else:
        tier = "NO_ANOMALY"
    return json_safe({"source_id": src.source_id, "star_id": src.star_id, "source": src.as_dict(),
                      "tier": tier, "astrometry": astro, "reflectance": refl, "phase_function": phase,
                      "polarization": pol, "supporting_tests": supporting, "rejections": rejections,
                      "cautions": cautions, "simulated": bool((src.meta or {}).get("simulated", False))})


def assess_sources(records: list[dict], conf: dict | None = None) -> dict:
    """Tally screened sources into the channel summary.

    ``NO_DATA_REACHED`` for an empty list; ``STATITE_CANDIDATES_PENDING_VET`` when any
    source sits in the interest or candidate tier; ``NO_CANDIDATE`` otherwise.
    Simulated inputs never read as a sky result: when every record is synthetic
    the verdict is ``NO_CANDIDATE`` with the tiers reported and a note saying so.
    """
    c = _conf_block(conf)
    tiers = dict(c.get("tiers") or {})
    fun = Funnel()
    by_tier: dict[str, int] = {}
    by_astro: dict[str, int] = {}
    ids: dict[str, list[str]] = {}
    for r in records or []:
        fun.bump("sources_screened")
        t = str(r.get("tier", "UNKNOWN"))
        by_tier[t] = by_tier.get(t, 0) + 1
        av = str((r.get("astrometry") or {}).get("verdict", "UNKNOWN"))
        by_astro[av] = by_astro.get(av, 0) + 1
        for rej in r.get("rejections") or []:
            fun.reject(str(rej))
        if t in (tiers.get("candidate"), tiers.get("interest")):
            ids.setdefault(t, []).append(str(r.get("source_id")))
    n = len(records or [])
    n_sim = sum(1 for r in (records or []) if r.get("simulated"))
    n_pending = sum(by_tier.get(str(tiers.get(k)), 0) for k in ("candidate", "interest"))
    fun.bump("pending_vet", n_pending)
    if n == 0:
        verdict = "NO_DATA_REACHED"
        fun.note("no CGI sources reached the screen")
    elif n_pending > 0 and n_sim == n:
        verdict = "NO_CANDIDATE"
        fun.note(f"simulated_inputs_only: {n_pending} synthetic source(s) reached a tier; not a sky result")
    elif n_pending > 0:
        verdict = "STATITE_CANDIDATES_PENDING_VET"
    else:
        verdict = "NO_CANDIDATE"
    return json_safe({"channel": "statite", "verdict": verdict, "n_sources": n,
                      "simulated_inputs": (None if n == 0 else (True if n_sim == n else (False if n_sim == 0 else "mixed"))),
                      "counts_by_tier": by_tier, "counts_by_astrometric_verdict": by_astro,
                      "pending_vet_ids": ids, "tiers": tiers, "funnel": fun.as_dict()})


# --------------------------------------------------------------------------------------
# Synthesis (tests and the selftest)
# --------------------------------------------------------------------------------------

def synthesise_source(kind: str, n_epochs: int, span_yr: float, sep_mas: float, star_mass: float,
                      distance_pc: float, err_mas: float, rng: np.random.Generator | None = None,
                      **kw) -> CGISource:
    """A synthetic CGI source of ``kind`` in {"fixed", "kepler", "background"}.

    Epochs are spread evenly over ``span_yr`` from ``mjd0`` (default 61000); the
    position has Gaussian noise ``err_mas``.  ``kepler`` takes ``inc_deg``, ``e``,
    ``Omega_deg``, ``omega_deg``, ``mean_anomaly_deg`` (semi-major axis from
    ``sep_mas`` at ``distance_pc``); ``background`` takes ``pm_masyr=(px, py)`` and
    ``with_parallax`` (default True) and uses the same parallax factors the fitter
    imposes.  ``ecl_lat_deg`` (default 30) and ``ecl_lon_deg`` (default 120) go into
    ``meta`` with ``simulated: True`` and the truth.
    """
    rng = rng or np.random.default_rng(0)
    kind = str(kind)
    n = int(n_epochs)
    mjd0 = float(kw.get("mjd0", 61000.0))
    mjd = mjd0 + np.linspace(0.0, float(span_yr) * DAYS_PER_YEAR, n) if n > 1 else np.array([mjd0])
    t = (mjd - mjd0) / DAYS_PER_YEAR
    pa = math.radians(float(kw.get("pa_deg", rng.uniform(0.0, 360.0))))
    meta: dict[str, Any] = {"simulated": True, "kind": kind,
                            "ecl_lat_deg": float(kw.get("ecl_lat_deg", 30.0)),
                            "ecl_lon_deg": float(kw.get("ecl_lon_deg", 120.0))}
    for key in ("pm_x_masyr", "pm_y_masyr"):
        if key in kw:
            meta[key] = float(kw[key])
    truth: dict[str, Any] = {"sep_mas": float(sep_mas)}
    if kind == "fixed":
        dx = np.full(n, sep_mas * math.sin(pa))
        dy = np.full(n, sep_mas * math.cos(pa))
        truth.update({"pa_deg": math.degrees(pa)})
    elif kind == "kepler":
        a = float(sep_mas) * float(distance_pc) / 1000.0
        e = float(kw.get("e", 0.0))
        inc = math.radians(float(kw.get("inc_deg", 0.0)))
        big_omega = math.radians(float(kw.get("Omega_deg", math.degrees(pa))))
        omega = math.radians(float(kw.get("omega_deg", 0.0)))
        per = kepler_period_yr(a, star_mass)
        m0 = math.radians(float(kw.get("mean_anomaly_deg", 0.0)))
        t_peri = -m0 * per / (2.0 * np.pi)
        dx, dy = kepler_xy(t, (a, e, inc, big_omega, omega, t_peri), star_mass, distance_pc)
        truth.update({"a_au": a, "e": e, "inc_deg": math.degrees(inc), "Omega_deg": math.degrees(big_omega),
                      "omega_deg": math.degrees(omega), "t_peri_yr": t_peri, "period_yr": per})
    elif kind == "background":
        pmx, pmy = (float(v) for v in kw.get("pm_masyr", (50.0, 0.0)))
        dx = sep_mas * math.sin(pa) + pmx * t
        dy = sep_mas * math.cos(pa) + pmy * t
        truth.update({"pm_masyr": [pmx, pmy], "with_parallax": bool(kw.get("with_parallax", True))})
        if kw.get("with_parallax", True):
            probe = CGISource("probe", "probe", star_mass, distance_pc, mjd, dx, dy, err_mas, meta=meta)
            fx, fy, _ = parallax_factors(probe)
            plx = 1000.0 / float(distance_pc)
            dx = dx + plx * fx
            dy = dy + plx * fy
            truth["parallax_mas"] = plx
    else:
        raise ValueError(f"unknown kind {kind!r}")
    dx = dx + rng.normal(0.0, err_mas, n)
    dy = dy + rng.normal(0.0, err_mas, n)
    meta["truth"] = truth
    return CGISource(source_id=str(kw.get("source_id", f"synthetic_{kind}")),
                     star_id=str(kw.get("star_id", "synthetic_star")), star_mass_msun=float(star_mass),
                     distance_pc=float(distance_pc), epochs_mjd=mjd, dx_mas=dx, dy_mas=dy,
                     err_mas=float(err_mas), contrast_by_band=kw.get("contrast_by_band"),
                     phase_angle_deg=kw.get("phase_angle_deg"), contrast_per_epoch=kw.get("contrast_per_epoch"),
                     contrast_err_per_epoch=kw.get("contrast_err_per_epoch"),
                     polarization=kw.get("polarization"), meta=meta)
