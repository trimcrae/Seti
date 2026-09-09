"""S40 --- the opaque lens: a microlens that occults its own images.

A point-mass lens at angular separation ``u`` (in Einstein radii) from a
background source makes two images at

    theta_+- = (u +- sqrt(u^2 + 4)) / 2            [theta_E units]

with magnifications

    A_+- = (u^2 + 2) / (2 u sqrt(u^2 + 4)) +- 1/2.

The major image sits outside the Einstein ring (theta_+ > 1); the minor image
sits inside it, at |theta_-| = (sqrt(u^2 + 4) - u) / 2, which *falls* from 1
at u = 0 toward 1/u in the wings.  If the lens is an opaque disc of radius
rho_L (theta_E units), every image with |theta| < rho_L is removed (Agol 2002
treats the natural version, a lens that also occults).  For rho_L < 1 that
removes the **minor image in the wings**, at every u > u_c = 1/rho_L - rho_L,
and leaves the peak Paczynski: the light curve shows a *symmetric pair of
downward steps at +-u_c, each of depth exactly A_-(u_c)*.  One parameter fixes
both where the steps are and how deep, which no natural feature does.  For
rho_L >= 1 the minor image is always gone and the major image is gone for
u < rho_L - 1/rho_L: a central hole flanked by lensing wings.

A finite source of radius rho_* rounds each step over Delta u ~ rho_*; the
model integrates over the source disc.  With theta_E known (from rho_* against
the source's angular radius theta_*, or the lower bound
theta_E >= theta_* sqrt(A_max^2 - 1)/2 from the peak alone) the occulter's
physical radius is R = rho_L theta_E D_L and the lens mass follows from
theta_E^2 = kappa M pi_rel, so the implied mean density is a function of the
one unknown, D_L.  The channel reports it over the whole range and the
distance window inside which a *natural* body could match
(``lens.density_floor_g_cc``); for any detectable event that window is a few
parsecs to tens of parsecs, where the lens would be a blend that moves about
an arcsecond a year --- a test the vet stage runs.

Every function here is pure numpy/scipy and offline-testable.  The screen
never fabricates: a curve that cannot be fitted says so in ``rejections``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq, least_squares

from .schema import LightCurve, flag_value

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

KAPPA_MAS_PER_MSUN = 8.144          # theta_E^2 = kappa * M * pi_rel   (mas, M_sun, mas)
AU_KM = 1.495978707e8
MSUN_G = 1.98892e33
U_FLOOR = 1e-6                      # numerical floor for u -> 0 in the magnifications
_SOFT_DEFAULT = 0.0

TIER_NOT_LENSING = "NOT_LENSING"
TIER_NO_OCCULTATION = "LENSING_NO_OCCULTATION"
MODEL_KINDS = ("paczynski", "finite", "occult", "occult_finite")


# --------------------------------------------------------------------------------------
# Point-lens geometry
# --------------------------------------------------------------------------------------

def image_positions(u):
    """Image positions ``(theta_plus, theta_minus)`` in theta_E units.

    theta_plus = (u + sqrt(u^2+4))/2 > 1 lies on the source's side of the lens;
    theta_minus = -(sqrt(u^2+4) - u)/2 lies on the opposite side, inside the
    Einstein ring.  Occultation tests use ``abs(theta_minus)``.
    """
    u = np.asarray(u, dtype=float)
    s = np.sqrt(u * u + 4.0)
    return (u + s) / 2.0, -(s - u) / 2.0


def magnifications(u):
    """``(A_plus, A_minus)`` of the two images; ``u`` is floored at ``U_FLOOR``.

    A_+- = (u^2+2) / (2 u sqrt(u^2+4)) +- 1/2.  Their sum is the Paczynski
    magnification; their difference is exactly 1 at every u.
    """
    u = np.maximum(np.abs(np.asarray(u, dtype=float)), U_FLOOR)
    core = (u * u + 2.0) / (2.0 * u * np.sqrt(u * u + 4.0))
    return core + 0.5, core - 0.5


def paczynski_magnification(u):
    """A(u) = (u^2+2) / (u sqrt(u^2+4)), the point-source point-lens total."""
    a_plus, a_minus = magnifications(u)
    return a_plus + a_minus


def u_from_magnification(a):
    """Invert the Paczynski magnification: u = sqrt(2 (A/sqrt(A^2-1) - 1))."""
    a = np.maximum(np.asarray(a, dtype=float), 1.0 + 1e-9)
    return np.sqrt(np.maximum(2.0 * (a / np.sqrt(a * a - 1.0) - 1.0), 0.0))


def u_crit(rho_l: float) -> float:
    """The radius at which an occulter of radius ``rho_l`` switches an image off.

    For rho_l < 1 it is u_c = 1/rho_l - rho_l, beyond which the *minor* image is
    hidden (wing occultation).  For rho_l >= 1 it is u_h = rho_l - 1/rho_l,
    inside which the *major* image is hidden (central hole).  Both are the one
    number |1/rho_l - rho_l|; ``regime`` says which it means.
    """
    rho_l = float(rho_l)
    return abs(1.0 / rho_l - rho_l)


def regime(rho_l: float) -> str:
    """``"wing_occultation"`` for rho_l < 1, ``"central_hole"`` for rho_l >= 1."""
    return "central_hole" if float(rho_l) >= 1.0 else "wing_occultation"


def step_depth(rho_l: float) -> float:
    """The magnification removed at the step, in units of the source flux.

    Wing regime: A_-(u_c), the minor image's magnification where it vanishes.
    Central-hole regime: A_+(u_h), the major image's magnification at the hole's
    edge (the minor image is gone everywhere and produces no step).
    """
    uc = u_crit(rho_l)
    a_plus, a_minus = magnifications(uc)
    return float(a_minus if float(rho_l) < 1.0 else a_plus)


def _geometry(u):
    """(theta_plus, |theta_minus|, A_plus, A_minus) from one square root."""
    u = np.maximum(np.abs(np.asarray(u, dtype=float)), U_FLOOR)
    s = np.sqrt(u * u + 4.0)
    core = (u * u + 2.0) / (2.0 * u * s)
    return (u + s) / 2.0, (s - u) / 2.0, core + 0.5, core - 0.5


def _keep(theta_abs, rho_l: float, soft: float):
    """Fraction of an image at |theta| that an occulter of radius rho_l lets through.

    A hard edge (``soft == 0``) is the physics.  ``soft > 0`` replaces the edge
    by a logistic of width ``soft`` (theta_E units): a numerical regulariser
    used *only inside the optimiser*, because with a hard edge the chi^2 is
    piecewise constant between epochs and a finite-difference Jacobian sees no
    slope.  Reported chi^2 values are always recomputed with ``soft = 0``.
    """
    if soft and soft > 0:
        x = np.clip((theta_abs - rho_l) / float(soft), -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))
    return (theta_abs >= rho_l).astype(float)


def occulted_magnification_point(u, rho_l: float, soft: float = _SOFT_DEFAULT):
    """Total magnification of a point source behind an opaque lens of radius ``rho_l``.

    The sum of the images with |theta| >= rho_l.  For rho_l < 1 the minor image
    drops out for u > u_c = 1/rho_l - rho_l; for rho_l >= 1 it is always out and
    the major image is out for u < rho_l - 1/rho_l (zero source flux: the
    central hole).
    """
    th_plus, th_minus_abs, a_plus, a_minus = _geometry(u)
    return a_plus * _keep(th_plus, rho_l, soft) + a_minus * _keep(th_minus_abs, rho_l, soft)


# --------------------------------------------------------------------------------------
# Finite source
# --------------------------------------------------------------------------------------

def _gl_nodes(n: int):
    x, w = np.polynomial.legendre.leggauss(max(int(n), 1))
    return (x + 1.0) / 2.0, w / 2.0            # on [0, 1]


def _disc_average(u, rho_star: float, rho_l: float | None, n_samples: int, soft: float):
    """Average magnification over a uniform source disc of radius ``rho_star``.

    The disc is integrated in polar coordinates *centred on the lens* --- radius
    s (= the point-source u) and azimuth psi --- with Gauss-Legendre nodes
    weighted by area (s ds dpsi).  Two features of the integrand sit at fixed
    lens-centred radii and are handled exactly this way: the 1/s singularity of
    A(s) at s -> 0 (cancelled by the area element, the integrand A(s) s is
    smooth) and the occulter's edge at s_b = |1/rho_l - rho_l|, which splits
    each radial chord into an un-occulted and an occulted piece.  The rounding
    of the steps over Delta u ~ rho_* therefore comes out of the geometry, not
    out of a smoothing kernel.
    """
    u = np.atleast_1d(np.asarray(u, dtype=float)).ravel()
    rho = max(float(rho_star), 1e-9)
    n_psi = max(2, int(round(math.sqrt(max(int(n_samples), 4)))))
    n_s_total = max(2, int(n_samples) // n_psi)
    split = rho_l is not None
    n_s = max(2, n_s_total // 2) if split else n_s_total
    xs, ws = _gl_nodes(n_s)
    xp, wp = _gl_nodes(n_psi)

    psi_max = np.where(u < rho, np.pi, np.arcsin(np.clip(rho / np.maximum(u, 1e-300), 0.0, 1.0)))
    psi = psi_max[:, None] * xp[None, :]                      # (N, n_psi)
    wpsi = psi_max[:, None] * wp[None, :]
    q = np.sqrt(np.maximum(rho * rho - (u[:, None] * np.sin(psi)) ** 2, 0.0))
    uc = u[:, None] * np.cos(psi)
    s_lo = np.maximum(uc - q, 0.0)
    s_hi = uc + q

    if split:
        s_b = abs(1.0 / float(rho_l) - float(rho_l))
        mid = np.clip(s_b, s_lo, s_hi)
        pieces = ((s_lo, mid), (mid, s_hi))
    else:
        pieces = ((s_lo, s_hi),)

    total = np.zeros(u.shape)
    for a, b in pieces:
        length = (b - a)[:, :, None]                          # (N, n_psi, 1)
        s = a[:, :, None] + length * xs[None, None, :]
        w = length * ws[None, None, :]
        if split:
            mag = occulted_magnification_point(s, float(rho_l), soft)
        else:
            mag = paczynski_magnification(s)
        total += np.sum(wpsi * np.sum(mag * s * w, axis=2), axis=1)
    return 2.0 * total / (np.pi * rho * rho)


def paczynski_finite(u, rho_star: float, n_samples: int = 64):
    """Uniform-disc finite-source Paczynski magnification (no occulter).

    Reduces to the point-source form as rho_star -> 0 and to sqrt(1 + 4/rho_*^2)
    at u = 0, the classical flattened peak.
    """
    u = np.asarray(u, dtype=float)
    out = paczynski_magnification(u)
    near = np.abs(u) <= 10.0 * float(rho_star)
    if np.any(near):
        out = np.array(out, dtype=float, copy=True)
        out[near] = _disc_average(np.abs(u[near]), rho_star, None, n_samples, 0.0)
    return out


def occulted_magnification_finite(u, rho_l: float, rho_star: float, n_samples: int = 64,
                                  soft: float = _SOFT_DEFAULT):
    """Finite-source magnification behind an opaque lens of radius ``rho_l``.

    Far from both the lens and the occulter's edge (|u| > 10 rho_* and more
    than 3 rho_* from s_b) the point-source value is exact to O(rho_*^2/u^2)
    and is used directly; elsewhere the source disc is integrated.
    """
    u = np.abs(np.asarray(u, dtype=float))
    out = occulted_magnification_point(u, rho_l, soft)
    rho = float(rho_star)
    s_b = u_crit(rho_l)
    near = (u <= 10.0 * rho) | (np.abs(u - s_b) <= 3.0 * rho + 3.0 * float(soft or 0.0))
    if np.any(near):
        out = np.array(out, dtype=float, copy=True)
        out[near] = _disc_average(u[near], rho, rho_l, n_samples, soft)
    return out


# --------------------------------------------------------------------------------------
# Trajectory and flux models
# --------------------------------------------------------------------------------------

def trajectory(t, t0: float, tE: float, u0: float):
    """u(t) = sqrt(u0^2 + ((t - t0)/tE)^2) for rectilinear relative motion."""
    t = np.asarray(t, dtype=float)
    return np.sqrt(float(u0) ** 2 + ((t - float(t0)) / float(tE)) ** 2)


def magnification(u, kind: str, rho_l: float | None = None, rho_star: float | None = None,
                  n_samples: int = 64, soft: float = _SOFT_DEFAULT):
    """Dispatch on model kind: paczynski / finite / occult / occult_finite."""
    if kind == "paczynski":
        return paczynski_magnification(u)
    if kind == "finite":
        return paczynski_finite(u, float(rho_star), n_samples)
    if kind == "occult":
        if rho_star is not None and rho_star > 0:
            return occulted_magnification_finite(u, float(rho_l), float(rho_star), n_samples, soft)
        return occulted_magnification_point(u, float(rho_l), soft)
    if kind == "occult_finite":
        return occulted_magnification_finite(u, float(rho_l), float(rho_star), n_samples, soft)
    raise ValueError(f"unknown model kind {kind!r}")


def model_flux(t, params: dict, kind: str, n_samples: int = 64, soft: float = _SOFT_DEFAULT):
    """Flux = fs * A(u(t)) + fb in the light curve's units.

    ``params`` carries t0, tE, u0, fs, fb and, per kind, rho_star (``finite``,
    ``occult_finite``, optional for ``occult``) and rho_l (``occult``,
    ``occult_finite``).
    """
    u = trajectory(t, params["t0"], params["tE"], params["u0"])
    a = magnification(u, kind, rho_l=params.get("rho_l"), rho_star=params.get("rho_star"),
                      n_samples=n_samples, soft=soft)
    return float(params["fs"]) * a + float(params["fb"])


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------

_NL_NAMES = {
    "paczynski": ("t0", "tE", "u0"),
    "finite": ("t0", "tE", "u0", "rho_star"),
    "occult": ("t0", "tE", "u0", "rho_l"),
    "occult_finite": ("t0", "tE", "u0", "rho_l", "rho_star"),
}


def _profile_linear(a, f, w, fb_min: float = 0.0):
    """Weighted linear solve of f ~ fs * a + fb with fb >= fb_min; returns (fs, fb).

    The two flux parameters enter linearly, so they are profiled out of every
    non-linear fit: the optimiser only sees the shape parameters.  The blend
    floor closes the degenerate corner fs -> +inf, fb -> -inf at large u0 in
    which a broad, low-amplitude shape is scaled up to mimic any peak; with it
    the source flux cannot exceed the baseline.  ``fb_min`` is normally 0 (a
    negative blend is a photometry systematic, reported, not fitted through).
    """
    sw = np.sum(w)
    swa = np.sum(w * a)
    swaa = np.sum(w * a * a)
    swf = np.sum(w * f)
    swaf = np.sum(w * a * f)
    det = swaa * sw - swa * swa
    if not np.isfinite(det) or det <= 1e-300 * max(swaa * sw, 1.0):
        return 0.0, float(swf / sw) if sw > 0 else 0.0
    fs = (swaf * sw - swa * swf) / det
    fb = (swaa * swf - swa * swaf) / det
    if fb < fb_min:
        fb = fb_min
        fs = (swaf - fb * swa) / swaa if swaa > 0 else 0.0
    if fs < 0.0:                      # an inverted shape is not a lens: flat at the mean
        fs, fb = 0.0, float(swf / sw) if sw > 0 else 0.0
    return float(fs), float(fb)


def _wls(design, f, e):
    """Weighted least squares with covariance; returns (coef, chi2, cov)."""
    dw = design / e[:, None]
    fw = f / e
    coef, *_ = np.linalg.lstsq(dw, fw, rcond=None)
    resid = fw - dw @ coef
    chi2 = float(np.sum(resid * resid))
    try:
        cov = np.linalg.inv(dw.T @ dw)
    except np.linalg.LinAlgError:
        cov = np.full((design.shape[1], design.shape[1]), np.nan)
    return coef, chi2, cov


def _shape(x, kind, t, n_samples, soft):
    t0, tE, u0 = x[0], x[1], x[2]
    rho_l = rho_star = None
    names = _NL_NAMES[kind]
    if "rho_l" in names:
        rho_l = x[names.index("rho_l")]
    if "rho_star" in names:
        rho_star = x[names.index("rho_star")]
    u = trajectory(t, t0, tE, u0)
    return magnification(u, kind, rho_l=rho_l, rho_star=rho_star, n_samples=n_samples, soft=soft)


def _evaluate(x, kind, t, f, e, n_samples, soft, prior=None):
    """chi^2 and profiled (fs, fb) of the shape parameters ``x``."""
    a = _shape(x, kind, t, n_samples, soft)
    w = 1.0 / (e * e)
    fs, fb = _profile_linear(a, f, w)
    r = (f - fs * a - fb) / e
    chi2 = float(np.sum(r * r))
    if prior is not None:
        chi2 += float(_prior_residual(x, kind, prior) ** 2)
    return chi2, fs, fb


def _prior_residual(x, kind, prior):
    names = _NL_NAMES[kind]
    if prior is None or "rho_star" not in names:
        return 0.0
    mu, sig = prior
    return (x[names.index("rho_star")] - mu) / sig


def _fit_kind(kind, t, f, e, starts, bounds, n_samples, soft, max_nfev, prior=None):
    """least_squares over the shape parameters from each start; best exact chi^2 wins."""
    lo, hi = bounds
    w = 1.0 / (e * e)

    def resid(x):
        a = _shape(x, kind, t, n_samples, soft)
        fs, fb = _profile_linear(a, f, w)
        r = (f - fs * a - fb) / e
        if prior is not None and "rho_star" in _NL_NAMES[kind]:
            r = np.append(r, _prior_residual(x, kind, prior))
        return r

    best = None
    errors = []
    for x0 in starts:
        x0 = np.clip(np.asarray(x0, dtype=float), lo + 1e-12, hi - 1e-12)
        span = max(float(x0[1]), 1e-3)
        x_scale = np.array([0.05 * span, 0.1 * span, 0.05] + [0.05] * (len(x0) - 3))
        try:
            sol = least_squares(resid, x0, bounds=(lo, hi), x_scale=x_scale, max_nfev=max_nfev,
                                ftol=1e-7, xtol=1e-7, gtol=1e-8)
            x, ok, nfev = sol.x, bool(sol.success), int(sol.nfev)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{kind} start {np.round(x0, 4).tolist()}: {exc!r}")
            x, ok, nfev = x0, False, 0
        chi2, fs, fb = _evaluate(x, kind, t, f, e, n_samples, 0.0, prior)
        if best is None or chi2 < best["chi2"]:
            best = {"x": x, "chi2": chi2, "fs": fs, "fb": fb, "converged": ok, "nfev": nfev}
    best["errors"] = errors
    return best


def _refine_rho_l(best, kind, t, f, e, n_samples, prior, frac=0.05, n=None, n_max=1500):
    """Exact-model 1-D scan of rho_l around the soft-edge optimum.

    With a hard edge the chi^2 is piecewise constant in rho_l, changing only
    when the step crosses an epoch, so the scan is spaced at the epoch
    resolution in u (median cadence / tE, mapped through du_c/drho_l) unless
    ``n`` is given.
    """
    names = _NL_NAMES[kind]
    i = names.index("rho_l")
    x = np.array(best["x"], dtype=float)
    if n is None:
        du_epoch = float(np.median(np.diff(t))) / max(float(x[1]), 1e-6) if t.size > 1 else 1e-3
        drho = du_epoch / (1.0 / x[i] ** 2 + 1.0)
        n = int(np.clip(2.0 * frac * x[i] / max(drho, 1e-9), 41, n_max)) | 1
    grid = x[i] * np.linspace(1.0 - frac, 1.0 + frac, n)
    out = dict(best)
    for r in grid:
        xx = x.copy()
        xx[i] = r
        chi2, fs, fb = _evaluate(xx, kind, t, f, e, n_samples, 0.0, prior)
        if chi2 < out["chi2"]:
            out = dict(out, x=xx, chi2=chi2, fs=fs, fb=fb)
    return out


def _pack(kind, best, n):
    names = _NL_NAMES[kind]
    params = {k: float(v) for k, v in zip(names, best["x"], strict=True)}
    params["fs"] = float(best["fs"])
    params["fb"] = float(best["fb"])
    n_free = len(names) + 2
    return {"kind": kind, "params": params, "chi2": float(best["chi2"]), "n_free": n_free,
            "bic": float(best["chi2"] + n_free * math.log(max(n, 2))),
            "converged": bool(best["converged"]), "nfev": int(best["nfev"])}


def _running_median(x, w: int):
    w = max(int(w) | 1, 1)
    if x.size < w:
        return np.full(x.shape, np.median(x))
    pad = np.pad(x, (w // 2, w // 2), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, w), axis=1)


def initial_guess(t, f, e, smooth: int = 5) -> dict:
    """t0 at the (median-smoothed) peak, u0 from the peak amplitude, tE from the FWHM.

    Baseline = median flux (the event occupies a minority of epochs); the peak
    amplitude with fb = 0 gives u0 through the inverse Paczynski relation; the
    half-maximum crossings on both sides give the FWHM, and tE follows from
    u(t_half) = u_half.  An event truncated by the window falls back to a
    quarter of the span.
    """
    fs_ = _running_median(f, smooth)
    base = float(np.median(f))
    i_pk = int(np.argmax(fs_))
    peak = float(fs_[i_pk])
    t0 = float(t[i_pk])
    a_max = max(peak / base, 1.0 + 1e-6) if base > 0 else 1.0 + 1e-6
    u0 = float(np.clip(u_from_magnification(a_max), 1e-3, 3.0))
    half = base + 0.5 * (peak - base)
    above = fs_ > half
    left = i_pk
    while left > 0 and above[left - 1]:
        left -= 1
    right = i_pk
    while right < t.size - 1 and above[right + 1]:
        right += 1
    span = float(t[-1] - t[0]) if t.size > 1 else 1.0
    fwhm = float(t[right] - t[left])
    truncated = left == 0 or right == t.size - 1
    a_half = 1.0 + 0.5 * (a_max - 1.0)
    u_half = float(u_from_magnification(a_half))
    if fwhm > 0 and not truncated and u_half > u0:
        tE = 0.5 * fwhm / math.sqrt(u_half * u_half - u0 * u0)
    else:
        tE = 0.25 * span
    tE = float(np.clip(tE, 1e-3, 10.0 * max(span, 1e-3)))
    out = {"t0": t0, "tE": tE, "u0": u0, "a_max": a_max, "fs": max(base, 0.0), "fb": 0.0,
           "fwhm_d": fwhm, "truncated": bool(truncated), "peak_index": i_pk, "hole": None}
    # A central hole is a *dip* to the blend level flanked by lensing wings: locate the
    # contiguous region below half the dip depth and take its centre and half-width.
    sig = float(np.median(e)) / math.sqrt(max(int(smooth), 1))
    i_min = int(np.argmin(fs_))
    depth = base - float(fs_[i_min])
    if sig > 0 and depth / sig >= 5.0:
        below = fs_ < base - 0.5 * depth
        a = i_min
        while a > 0 and below[a - 1]:
            a -= 1
        b = i_min
        while b < t.size - 1 and below[b + 1]:
            b += 1
        out["hole"] = {"t0": float(0.5 * (t[a] + t[b])), "half_width_d": float(0.5 * (t[b] - t[a])),
                       "depth_snr": float(depth / sig),
                       "truncated": bool(a == 0 or b == t.size - 1)}
    return out


def peak_snr(f, e, smooth: int = 5) -> float:
    """S/N of the smoothed peak above the median baseline, in the smoothed noise."""
    fs_ = _running_median(f, smooth)
    base = float(np.median(f))
    sig = float(np.median(e)) / math.sqrt(max(int(smooth), 1))
    if not np.isfinite(sig) or sig <= 0:
        return 0.0
    return float((np.max(fs_) - base) / sig)


def dip_snr(f, e, smooth: int = 5) -> float:
    """S/N of the smoothed minimum below the median baseline (the central-hole regime)."""
    fs_ = _running_median(f, smooth)
    base = float(np.median(f))
    sig = float(np.median(e)) / math.sqrt(max(int(smooth), 1))
    if not np.isfinite(sig) or sig <= 0:
        return 0.0
    return float((base - np.min(fs_)) / sig)


def box_dip_chi2(t, f, e, t_in: float, t_out: float) -> dict:
    """A flat curve with a flat-bottomed box between ``t_in`` and ``t_out``.

    The non-lensing alternative for a central hole: a total eclipse or transit
    has the hole but not the lensing wings that must rise into it.  Two levels
    are solved linearly; with the two edges that is four free parameters.
    """
    inside = (t >= t_in) & (t <= t_out)
    w = 1.0 / (e * e)
    chi2 = 0.0
    levels = {}
    for name, m in (("outside", ~inside), ("inside", inside)):
        if np.any(m):
            lvl = float(np.sum(w[m] * f[m]) / np.sum(w[m]))
            chi2 += float(np.sum(w[m] * (f[m] - lvl) ** 2))
            levels[name] = lvl
    n_free = 4 if np.any(inside) and np.any(~inside) else 1
    return {"chi2": chi2, "n_free": n_free, "bic": chi2 + n_free * math.log(max(t.size, 2)),
            "levels": levels, "t_in": float(t_in), "t_out": float(t_out)}


def _good_arrays(lc: LightCurve, conf: dict):
    mask = 0
    for name in (conf.get("lightcurve_dq_reject") or []):
        try:
            mask |= flag_value(conf, str(name))
        except KeyError:
            continue
    g = lc.good(mask or None)
    o = np.argsort(lc.mjd[g])
    return lc.mjd[g][o], lc.flux[g][o], lc.flux_err[g][o]


def kink_test(t, f, e, params: dict, conf: dict, n_grid: int = 150) -> dict:
    """Locate the step on the rising and falling sides independently.

    With (t0, tE, u0) held at the occulting fit, the template of a hidden
    image is known up to where it switches off: in the wing regime the flux is
    fs (A_+ + A_-) + fb inside u_c and loses fs A_- outside; in the central-hole
    regime it loses fs A_+ inside u_h.  On each side of t0 the filter scans
    u_c over the data's own u-values and, for each, fits the three linear
    amplitudes (source, hidden-image fraction d, blend) --- a matched step
    filter with the physically predicted profile.  The u_c with the largest
    chi^2 gain and d > 0 (a *downward* step going outward) is that side's
    estimate; ``d`` is the measured step depth in units of the predicted one
    (an opaque body gives d = 1 on both sides).  A caustic crossing, a bump, or
    any one-sided feature yields different u_c on the two sides, or a step on
    one side only, and fails ``kink_asymmetry_max``.
    """
    lens_conf = conf.get("lens") or {}
    thr = float(lens_conf.get("kink_side_dchi2_min",
                              0.5 * float(lens_conf.get("occult_delta_chi2_min", 25.0))))
    min_side = int(lens_conf.get("kink_min_side_epochs", 20))
    t0, tE, u0 = float(params["t0"]), float(params["tE"]), float(params["u0"])
    rho_l = float(params["rho_l"])
    reg = regime(rho_l)
    u = trajectory(t, t0, tE, u0)
    a_plus, a_minus = magnifications(u)
    sides = {"rise": t < t0, "fall": t >= t0}
    out = {"regime": reg, "threshold_dchi2": thr, "sides": {}}
    for name, m in sides.items():
        rec = {"n": int(np.sum(m)), "detected": False, "u_c": None, "d": None, "d_err": None,
               "dchi2": None}
        if rec["n"] < min_side:
            rec["note"] = "too_few_epochs_on_side"
            out["sides"][name] = rec
            continue
        us, fsd, esd = u[m], f[m], e[m]
        ap, am = a_plus[m], a_minus[m]
        if reg == "wing_occultation":
            base_shape, hidden, outward = ap + am, am, True
        else:
            base_shape, hidden, outward = ap, ap, False
        design0 = np.column_stack([base_shape, np.ones_like(us)])
        _, chi2_0, _ = _wls(design0, fsd, esd)
        u_sorted = np.sort(us)
        lo_i, hi_i = 5, max(u_sorted.size - 5, 6)
        cand = np.unique(u_sorted[np.linspace(lo_i, hi_i - 1, min(n_grid, hi_i - lo_i)).astype(int)])
        best = None
        for uc in cand:
            sel = (us > uc) if outward else (us < uc)
            design = np.column_stack([base_shape, -hidden * sel, np.ones_like(us)])
            coef, chi2, cov = _wls(design, fsd, esd)
            if coef[0] <= 0:
                continue
            d = coef[1] / coef[0]
            if d <= 0:
                continue
            if best is None or chi2 < best[0]:
                var_d = cov[1, 1] / coef[0] ** 2 if np.isfinite(cov[1, 1]) else np.nan
                best = (chi2, float(uc), float(d), float(math.sqrt(max(var_d, 0.0))))
        if best is not None:
            dchi2 = chi2_0 - best[0]
            rec.update(u_c=best[1], d=best[2], d_err=best[3], dchi2=float(dchi2),
                       detected=bool(dchi2 >= thr))
        out["sides"][name] = rec
    rise, fall = out["sides"]["rise"], out["sides"]["fall"]
    det = [s for s in (rise, fall) if s["detected"]]
    out["u_c_rise"] = rise["u_c"] if rise["detected"] else None
    out["u_c_fall"] = fall["u_c"] if fall["detected"] else None
    if len(det) == 2:
        mean = 0.5 * (rise["u_c"] + fall["u_c"])
        out["kink_asymmetry"] = float(abs(rise["u_c"] - fall["u_c"]) / mean) if mean > 0 else None
        out["d_measured"] = float(0.5 * (rise["d"] + fall["d"]))
        out["d_err"] = float(0.5 * math.hypot(rise["d_err"], fall["d_err"]))
        out["u_c_measured"] = float(mean)
        out["status"] = "both_sides"
    elif len(det) == 1:
        out["kink_asymmetry"] = 1.0
        out["d_measured"] = float(det[0]["d"])
        out["d_err"] = float(det[0]["d_err"])
        out["u_c_measured"] = float(det[0]["u_c"])
        out["status"] = "one_side_only"
    else:
        out["kink_asymmetry"] = None
        out["d_measured"] = None
        out["d_err"] = None
        out["u_c_measured"] = None
        out["status"] = "not_detected"
    return out


def fit_event(lc: LightCurve, conf: dict, rho_star_prior=None) -> dict:
    """Fit the four models to one light curve and run the kink test.

    Models: Paczynski (t0, tE, u0, fs, fb), finite source (+rho_star), opaque
    lens (+rho_l), and opaque lens with finite source (+rho_l, rho_star).  The
    flux parameters are profiled; the shape parameters go through bounded
    ``least_squares`` from several starts over u0 (Paczynski), rho_l and, when
    the curve carries a significant dip, the central-hole geometry (occulting)
    so that a local minimum does not decide the comparison.  The point-source
    occulting fit uses a soft edge inside the optimiser and is then refined and
    reported with the exact hard-edged model at the epoch resolution; the
    finite-source occulting fit runs on the exact disc integrator with rho_star
    floored at twice the epoch spacing in u, and ``rho_star_resolved`` says
    whether its rho_star sits clear of that floor.  ``rho_star_prior`` may be a
    value or ``(value, sigma)``; the latter is a Gaussian penalty.

    Returns the per-model fits (params, chi2, n_free, BIC), the deltas the
    screen gates on (``dchi2_best_vs_flat`` is the lensing-at-all test, so a
    central hole whose Paczynski fit is flat still counts), the box-dip
    comparison in the central-hole regime, the kink test, and the regime.
    """
    lens_conf = conf.get("lens") or {}
    n_samples = int(lens_conf.get("n_source_samples", 64))
    soft = float(lens_conf.get("fit_soft_edge", 0.005))
    max_nfev = int(lens_conf.get("fit_max_nfev", 120))
    max_nfev_finite = int(lens_conf.get("fit_max_nfev_finite", 30))
    t, f, e = _good_arrays(lc, conf)
    n = int(t.size)
    out = {"n": n, "models": {}, "flat": None}
    if n < 8:
        out["error"] = "too_few_epochs"
        return out
    span = float(t[-1] - t[0]) if n > 1 else 1.0
    w = 1.0 / (e * e)
    mean = float(np.sum(w * f) / np.sum(w))
    chi2_flat = float(np.sum(w * (f - mean) ** 2))
    out["flat"] = {"params": {"flux": mean}, "chi2": chi2_flat, "n_free": 1,
                   "bic": chi2_flat + math.log(n)}

    prior = None
    if rho_star_prior is not None:
        if isinstance(rho_star_prior, (tuple, list)) and len(rho_star_prior) == 2:
            prior = (float(rho_star_prior[0]), max(float(rho_star_prior[1]), 1e-6))
        else:
            prior = (float(rho_star_prior), 0.3 * max(float(rho_star_prior), 1e-3))

    g = initial_guess(t, f, e)
    out["initial_guess"] = g
    lo_t, hi_t = t[0] - span, t[-1] + span
    b_t0, b_te, b_u0 = (lo_t, hi_t), (1e-3, 10.0 * max(span, 1e-3)), (1e-4, 5.0)
    b_rl, b_rs = (0.01, 5.0), (1e-3, 5.0)

    def bounds(kind):
        table = {"t0": b_t0, "tE": b_te, "u0": b_u0, "rho_l": b_rl, "rho_star": b_rs}
        lo = np.array([table[k][0] for k in _NL_NAMES[kind]])
        hi = np.array([table[k][1] for k in _NL_NAMES[kind]])
        return lo, hi

    # Paczynski: restarts over u0.
    u0_starts = [g["u0"] * s for s in (0.33, 1.0, 3.0)]
    starts = [[g["t0"], g["tE"], u] for u in u0_starts]
    pz = _fit_kind("paczynski", t, f, e, starts, bounds("paczynski"), n_samples, 0.0, max_nfev)
    out["models"]["paczynski"] = _pack("paczynski", pz, n)
    p0 = list(pz["x"])

    # Finite source: from the Paczynski optimum, restarts over rho_star.
    rs_starts = list(lens_conf.get("rho_star_starts", [0.1, 0.5]))
    if prior is not None:
        rs_starts = [prior[0]] + rs_starts
    starts = [p0 + [r] for r in rs_starts]
    fin = _fit_kind("finite", t, f, e, starts, bounds("finite"), n_samples, 0.0, max_nfev_finite,
                    prior)
    out["models"]["finite"] = _pack("finite", fin, n)

    # Opaque lens: restarts over rho_l, soft edge in the optimiser, exact refinement.  A
    # significant dip adds central-hole starts (rho_l > 1) whose tE makes the hole's
    # half-width match u_h = rho_l - 1/rho_l.
    rl_starts = list(lens_conf.get("rho_l_starts", [0.3, 0.6, 0.9, 1.5]))
    starts = [p0 + [r] for r in rl_starts]
    hole = g.get("hole")
    if hole and not hole["truncated"] and hole["half_width_d"] > 0:
        u0_h = 0.1
        for r in lens_conf.get("rho_l_hole_starts", [1.2, 1.5, 2.5]):
            u_h = u_crit(r)
            te_h = hole["half_width_d"] / math.sqrt(max(u_h * u_h - u0_h * u0_h, 1e-6))
            starts.append([hole["t0"], te_h, u0_h, r])
    occ = _fit_kind("occult", t, f, e, starts, bounds("occult"), n_samples, soft, max_nfev)
    occ = _refine_rho_l(occ, "occult", t, f, e, n_samples, None)
    out["models"]["occult"] = _pack("occult", occ, n)

    # Opaque lens with finite source.  The disc integrator is smooth in every parameter
    # once rho_star exceeds the epoch spacing in u, so the optimiser runs on the exact
    # model with rho_star floored there; a source smaller than that is the point-source
    # occulting model, which the BIC then prefers.  Starts: the occulting optimum with
    # several rho_star, and the finite-source optimum with the occulter beyond the data.
    du_epoch = float(np.median(np.diff(t))) / max(float(occ["x"][1]), 1e-6) if n > 1 else 1e-3
    lo_of, hi_of = bounds("occult_finite")
    lo_of = lo_of.copy()
    lo_of[-1] = max(lo_of[-1], 2.0 * du_epoch)
    rs2 = list(lens_conf.get("rho_star_starts_occult", [0.02, 0.2]))
    if prior is not None:
        rs2 = [prior[0]] + rs2
    starts = [list(occ["x"]) + [r] for r in rs2]
    u_max = float(np.max(trajectory(t, *fin["x"][:3])))
    uc_out = 1.5 * u_max
    starts.append(list(fin["x"][:3]) + [(math.sqrt(uc_out * uc_out + 4.0) - uc_out) / 2.0, fin["x"][3]])
    of = _fit_kind("occult_finite", t, f, e, starts, (lo_of, hi_of), n_samples, 0.0,
                   max_nfev_finite, prior)
    of = _refine_rho_l(of, "occult_finite", t, f, e, n_samples, prior, frac=0.03, n=21)
    out["models"]["occult_finite"] = _pack("occult_finite", of, n)
    out["rho_star_floor"] = float(lo_of[-1])

    m = out["models"]
    best_occ = "occult" if m["occult"]["bic"] <= m["occult_finite"]["bic"] else "occult_finite"
    out["best_occult_kind"] = best_occ
    bo = m[best_occ]
    out["dchi2_best_vs_flat"] = chi2_flat - min(v["chi2"] for v in m.values())
    out["dchi2_paczynski_vs_flat"] = chi2_flat - m["paczynski"]["chi2"]
    out["dchi2_occult_vs_paczynski"] = m["paczynski"]["chi2"] - bo["chi2"]
    out["dbic_occult_vs_finite"] = m["finite"]["bic"] - bo["bic"]
    out["dbic_occult_vs_paczynski"] = m["paczynski"]["bic"] - bo["bic"]
    out["dbic_finite_vs_paczynski"] = m["paczynski"]["bic"] - m["finite"]["bic"]
    out["dbic_occult_finite_vs_occult"] = m["occult"]["bic"] - m["occult_finite"]["bic"]
    # rho_star is *measured* only when the finite-source occulting model wins and its
    # rho_star sits clear of the epoch-resolution floor (a hard-edged truth pins it there).
    out["rho_star_resolved"] = bool(
        m["occult_finite"]["bic"] < m["occult"]["bic"]
        and m["occult_finite"]["params"]["rho_star"] > 2.0 * out["rho_star_floor"])

    p = bo["params"]
    out["rho_l"] = p["rho_l"]
    out["regime"] = regime(p["rho_l"])
    out["u_c"] = u_crit(p["rho_l"])
    out["step_depth_predicted"] = step_depth(p["rho_l"])
    out["a_max_unocculted"] = float(paczynski_magnification(p["u0"]))
    # Central hole: the non-lensing alternative is a box dip without wings.
    if out["regime"] == "central_hole":
        half = p["tE"] * math.sqrt(max(out["u_c"] ** 2 - p["u0"] ** 2, 0.0))
        box = box_dip_chi2(t, f, e, p["t0"] - half, p["t0"] + half)
        out["box"] = box
        out["dbic_occult_vs_box"] = box["bic"] - bo["bic"]
    else:
        out["box"] = None
        out["dbic_occult_vs_box"] = None
    kk = kink_test(t, f, e, p, conf)
    out["kink"] = kk
    out["u_c_rise"] = kk["u_c_rise"]
    out["u_c_fall"] = kk["u_c_fall"]
    out["kink_asymmetry"] = kk["kink_asymmetry"]
    out["kink_status"] = kk["status"]
    if kk["d_measured"] is not None:
        out["step_depth_measured"] = float(kk["d_measured"] * step_depth(p["rho_l"]))
        out["step_depth_measured_err"] = float(kk["d_err"] * step_depth(p["rho_l"]))
        out["step_depth_ratio"] = float(kk["d_measured"])
    else:
        out["step_depth_measured"] = None
        out["step_depth_measured_err"] = None
        out["step_depth_ratio"] = None
    return out


# --------------------------------------------------------------------------------------
# Physical scale
# --------------------------------------------------------------------------------------

def theta_e_lower_bound(a_max: float, theta_star_uas: float) -> float:
    """theta_E >= theta_* sqrt(A_max^2 - 1) / 2, in mas (theta_* given in uas).

    A uniform source of radius rho_* peaks at A = sqrt(1 + 4/rho_*^2), so a
    peak of A_max needs rho_* <= 2/sqrt(A_max^2 - 1), i.e. theta_E = theta_*/rho_*
    at least this large.
    """
    a = max(float(a_max), 1.0)
    return float(theta_star_uas) * 1e-3 * math.sqrt(max(a * a - 1.0, 0.0)) / 2.0


def lens_scale(rho_l: float, theta_e_mas: float, d_l_kpc, d_s_kpc: float) -> dict:
    """R, M and mean density of an opaque lens at distance ``d_l_kpc``.

    R = rho_L theta_E D_L (1 mas x 1 kpc = 1 AU by the definition of the
    parsec); M = theta_E^2 / (kappa pi_rel) with kappa = 8.144 mas/M_sun and
    pi_rel = 1/D_L - 1/D_S in mas for D in kpc; rho_bar = 3M / (4 pi R^3).
    """
    d_l = np.asarray(d_l_kpc, dtype=float)
    pi_rel = 1.0 / d_l - 1.0 / float(d_s_kpc)
    with np.errstate(divide="ignore", invalid="ignore"):
        mass = theta_e_mas ** 2 / (KAPPA_MAS_PER_MSUN * pi_rel)
        r_au = float(rho_l) * float(theta_e_mas) * d_l
        r_km = r_au * AU_KM
        vol_cc = 4.0 / 3.0 * np.pi * (r_km * 1e5) ** 3
        dens = mass * MSUN_G / vol_cc
    return {"d_l_kpc": d_l, "pi_rel_mas": pi_rel, "mass_msun": mass, "r_au": r_au, "r_km": r_km,
            "density_g_cc": dens}


def density_bound(rho_l: float, theta_e_mas: float, d_s_kpc: float, pi_e: float | None = None,
                  grid: int = 200, density_floor_g_cc: float = 0.01,
                  nearby_kpc: float = 0.2) -> dict:
    """The implied mean density of the occulter over every lens distance.

    rho_bar(D_L) is proportional to 1 / ((1/D_L - 1/D_S) D_L^3): it diverges as
    D_L -> 0 (a small, light, dense nearby body), falls to a minimum at
    D_L = 2 D_S / 3, and diverges again as D_L -> D_S (a lens of absurd mass
    just in front of the source).  The natural-body window is the *near*
    branch where rho_bar >= the floor: its outer edge
    ``natural_distance_max_kpc`` is solved exactly, and
    ``natural_distance_window_kpc`` is the same window read off the linear
    grid (empty when no grid point qualifies).  ``blend_moving_test_needed`` is
    True when every natural explanation needs a lens within ``nearby_kpc`` ---
    an object Roman would see as a blend moving ~1"/yr.  With ``pi_e`` the
    microlens parallax pins D_L through pi_rel = pi_E theta_E and the single
    value is reported next to the curve.
    """
    rho_l, theta_e, d_s = float(rho_l), float(theta_e_mas), float(d_s_kpc)
    k = np.arange(1, int(grid) + 1, dtype=float)
    d_grid = d_s * k / (float(grid) + 1.0)
    curve = lens_scale(rho_l, theta_e, d_grid, d_s)
    dens = curve["density_g_cc"]
    near_branch = d_grid <= 2.0 * d_s / 3.0
    nat = near_branch & (dens >= density_floor_g_cc)
    far = (~near_branch) & (dens >= density_floor_g_cc)
    window = [float(d_grid[nat].min()), float(d_grid[nat].max())] if np.any(nat) else None
    far_window = [float(d_grid[far].min()), float(d_grid[far].max())] if np.any(far) else None

    def log_excess(d):
        return math.log(float(lens_scale(rho_l, theta_e, d, d_s)["density_g_cc"])) - math.log(
            density_floor_g_cc)

    d_turn = 2.0 * d_s / 3.0
    d_tiny = 1e-9 * d_s
    if log_excess(d_turn) >= 0:
        natural_max = d_turn          # the whole near branch is natural-body territory
        whole = True
    else:
        whole = False
        try:
            natural_max = float(brentq(log_excess, d_tiny, d_turn, xtol=1e-12 * d_s))
        except ValueError:
            natural_max = 0.0
    out = {
        "rho_l": rho_l, "theta_e_mas": theta_e, "d_s_kpc": d_s,
        "density_floor_g_cc": density_floor_g_cc,
        "curve": {"d_l_kpc": d_grid, "mass_msun": curve["mass_msun"], "r_au": curve["r_au"],
                  "r_km": curve["r_km"], "density_g_cc": dens},
        "density_max_g_cc": float(np.nanmax(dens)),
        "density_min_g_cc": float(np.nanmin(dens)),
        "d_l_at_density_min_kpc": float(d_grid[int(np.nanargmin(dens))]),
        "natural_distance_window_kpc": window,
        "natural_distance_max_kpc": float(natural_max),
        "natural_everywhere_on_near_branch": bool(whole),
        "far_branch_window_kpc": far_window,
        "blend_moving_test_needed": bool(natural_max <= nearby_kpc),
        "nearby_kpc": nearby_kpc,
    }
    if pi_e is not None and pi_e > 0:
        pi_rel = float(pi_e) * theta_e
        d_l = 1.0 / (pi_rel + 1.0 / d_s)
        pin = lens_scale(rho_l, theta_e, d_l, d_s)
        out["pinned"] = {"pi_e": float(pi_e), "d_l_kpc": float(d_l),
                         "mass_msun": float(pin["mass_msun"]), "r_au": float(pin["r_au"]),
                         "r_km": float(pin["r_km"]), "density_g_cc": float(pin["density_g_cc"]),
                         "natural": bool(pin["density_g_cc"] >= density_floor_g_cc)}
    return out


# --------------------------------------------------------------------------------------
# Achromaticity
# --------------------------------------------------------------------------------------

def colour_consistency(lc_primary_deficit_frac: float, err: float,
                       lc_colour_deficit_frac: float, err2: float) -> float:
    """z-score of the two bands' fractional deficits; an opaque body gives z ~ 0."""
    denom = math.hypot(float(err), float(err2))
    if not np.isfinite(denom) or denom <= 0:
        return float("inf")
    return float((float(lc_primary_deficit_frac) - float(lc_colour_deficit_frac)) / denom)


def hidden_image_fraction(t, f, e, params: dict, min_epochs: int = 10) -> dict | None:
    """The fraction ``d`` of the predicted hidden image actually missing in one band.

    With the geometry (t0, tE, u0, rho_l) from the primary fit, the flux is
    fs A_shape - (fs d) A_hidden [occulted] + fb: three linear amplitudes.  An
    opaque body hides the same fraction (d = 1) of the image in every band;
    dust, a chromatic source or a systematic does not.  ``None`` when the band
    lacks epochs on either side of the edge.
    """
    t0, tE, u0, rho_l = (float(params[k]) for k in ("t0", "tE", "u0", "rho_l"))
    u = trajectory(t, t0, tE, u0)
    a_plus, a_minus = magnifications(u)
    uc = u_crit(rho_l)
    if regime(rho_l) == "wing_occultation":
        shape, hidden, sel = a_plus + a_minus, a_minus, u > uc
    else:
        shape, hidden, sel = a_plus, a_plus, u < uc
    if np.sum(sel) < min_epochs or np.sum(~sel) < min_epochs:
        return None
    design = np.column_stack([shape, -hidden * sel, np.ones_like(u)])
    coef, chi2, cov = _wls(design, f, e)
    if coef[0] <= 0 or not np.isfinite(cov[1, 1]):
        return None
    d = coef[1] / coef[0]
    d_err = math.sqrt(max(cov[1, 1], 0.0)) / coef[0]
    return {"d": float(d), "d_err": float(d_err), "n_occulted": int(np.sum(sel)),
            "n_unocculted": int(np.sum(~sel)), "fs": float(coef[0]), "fb": float(coef[2]),
            "chi2": chi2}


# --------------------------------------------------------------------------------------
# Screening
# --------------------------------------------------------------------------------------

def time_symmetry_test(t, f, e, t0: float, min_pairs: int = 20) -> dict:
    """Is the light curve symmetric about ``t0``?  Every lens is; no transient is.

    For each epoch at ``t0 + dt`` (dt > 0) the flux at ``t0 - dt`` is
    interpolated from the epochs before the peak, where the mirrored time is
    covered; the reduced chi-square of the differences over those pairs is
    the statistic.  Gravitational lensing (occulting or not, finite source or
    not) is time-symmetric about the closest approach, so a large value says
    the event is not a lens at all -- a supernova (fast rise, slow decline),
    a nova, a flare.  ``None`` when fewer than ``min_pairs`` mirrored pairs
    exist (the peak sits at an edge of the coverage).
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    e = np.asarray(e, float)
    before = t < t0
    after = t > t0
    if before.sum() < 3 or after.sum() < 3:
        return {"status": "insufficient_coverage", "chi2_red": None, "n_pairs": 0}
    tb, fb, eb = t[before], f[before], e[before]
    order = np.argsort(tb)
    tb, fb, eb = tb[order], fb[order], eb[order]
    dt = t[after] - t0
    mirror = t0 - dt
    ok = (mirror >= tb.min()) & (mirror <= tb.max())
    if ok.sum() < int(min_pairs):
        return {"status": "insufficient_coverage", "chi2_red": None, "n_pairs": int(ok.sum())}
    fm = np.interp(mirror[ok], tb, fb)
    em = np.interp(mirror[ok], tb, eb)
    diff = f[after][ok] - fm
    var = e[after][ok] ** 2 + em ** 2
    chi2 = float(np.sum(diff ** 2 / np.where(var > 0, var, np.inf)))
    n = int(ok.sum())
    return {"status": "ok", "chi2_red": chi2 / max(n, 1), "n_pairs": n,
            "mean_signed_diff": float(np.mean(diff)),
            "rise_minus_decline_flux": float(np.mean(diff))}


def screen_lightcurve(lc: LightCurve, conf: dict, colour_lc: LightCurve | None = None,
                      theta_star_uas: float | None = None, d_s_kpc: float | None = None,
                      rho_star_prior=None, pi_e: float | None = None) -> dict:
    """The S40 funnel for one star: gates, fits, kink, density, colour, tier.

    Tiers: ``NOT_LENSING`` (too few epochs, a weak peak, or no Paczynski
    improvement over a constant), ``LENSING_NO_OCCULTATION`` (a lensing event
    that the occulting model does not win, or wins with rho_L outside range),
    the ``interest`` tier from ``conf["lens"]["tiers"]`` (the occulting model
    preferred but a vet test failed or has not been run), and the
    ``candidate`` tier (every gate passed, the kink symmetric with the
    predicted depth, the deficit achromatic, the scale computed).
    ``rejections`` names every gate that failed; ``pending`` names every test
    that could not be run on what was supplied.
    """
    lens_conf = conf.get("lens") or {}
    tiers = lens_conf.get("tiers") or {}
    tier_interest = str(tiers.get("interest", "OCCULTING_PREFERRED_PENDING_VET"))
    tier_candidate = str(tiers.get("candidate", "OPAQUE_LENS_CANDIDATE"))
    rejections: list[str] = []
    pending: list[str] = []
    notes: list[str] = []
    out = {"star_id": lc.star_id, "band": lc.band, "tier": TIER_NOT_LENSING,
           "rejections": rejections, "pending": pending, "notes": notes, "fit": None,
           "kink": None, "density": None, "colour": None, "regime": None, "rho_l": None}

    t, f, e = _good_arrays(lc, conf)
    out["n_used"] = int(t.size)
    min_epochs = int(lens_conf.get("min_epochs", 200))
    if t.size < min_epochs:
        rejections.append("too_few_epochs")
        notes.append(f"{t.size} good epochs < {min_epochs}; no fit attempted")
        return out
    snr = peak_snr(f, e)
    dsnr = dip_snr(f, e)
    out["peak_snr"] = snr
    out["dip_snr"] = dsnr
    if max(snr, dsnr) < float(lens_conf.get("min_peak_snr", 20.0)):
        rejections.append("peak_snr_low")
        notes.append(f"peak S/N {snr:.1f} (dip {dsnr:.1f}) below the gate; no fit attempted")
        return out

    fit = fit_event(lc, conf, rho_star_prior=rho_star_prior)
    out["fit"] = fit
    if fit.get("error"):
        rejections.append(fit["error"])
        return out
    if fit["dchi2_best_vs_flat"] < float(lens_conf.get("paczynski_delta_chi2_min", 50.0)):
        rejections.append("not_lensing")
        return out

    out["regime"] = fit["regime"]
    out["rho_l"] = fit["rho_l"]
    out["kink"] = fit["kink"]
    occ_gates = False
    if fit["dchi2_occult_vs_paczynski"] < float(lens_conf.get("occult_delta_chi2_min", 25.0)):
        rejections.append("occult_not_preferred_vs_paczynski")
        occ_gates = True
    dbic_min = float(lens_conf.get("occult_delta_bic_min", 10.0))
    if fit["dbic_occult_vs_finite"] < dbic_min:
        rejections.append("occult_not_preferred_vs_finite")
        occ_gates = True
    if fit["dbic_occult_vs_paczynski"] < dbic_min:
        rejections.append("occult_bic_vs_paczynski_low")
        occ_gates = True
    if not (float(lens_conf.get("rho_l_min", 0.05)) <= fit["rho_l"]
            <= float(lens_conf.get("rho_l_max", 3.0))):
        rejections.append("rho_l_out_of_range")
        occ_gates = True
    if fit["regime"] == "central_hole" and fit["dbic_occult_vs_box"] < dbic_min:
        rejections.append("hole_without_wings")       # a total eclipse, not a lens
        occ_gates = True

    # Time symmetry about t0: a lens of any kind is symmetric; a supernova is not.
    # Fitted BEFORE the occultation gates are believed: the occulting model is
    # symmetric too, and on a SN Ia curve it out-fits Paczynski for the wrong
    # reason (run 34349717932: 1,262 of 2,688 simulated supernovae).
    t0_best = float(fit["models"][fit["best_occult_kind"]]["params"].get("t0",
                    fit["models"]["paczynski"]["params"].get("t0", np.median(t))))
    sym = time_symmetry_test(t, f, e, t0_best,
                             min_pairs=int(lens_conf.get("time_asymmetry_min_pairs", 20)))
    out["time_symmetry"] = sym
    if sym["status"] == "ok" and sym["chi2_red"] > float(lens_conf.get("time_asymmetry_chi2_red_max", 3.0)):
        rejections.append("time_asymmetric")
        notes.append("time-asymmetric about t0: a transient, not a lens")
        out["tier"] = TIER_NOT_LENSING
        return out
    if sym["status"] != "ok":
        pending.append("time_symmetry_not_run")

    # Kink symmetry and the two-for-one depth prediction (run whenever there is an event).
    asym_max = float(lens_conf.get("kink_asymmetry_max", 0.15))
    kk = fit["kink"]
    if kk["status"] == "not_detected":
        rejections.append("kink_not_detected")
    elif kk["status"] == "one_side_only" or kk["kink_asymmetry"] > asym_max:
        rejections.append("kink_asymmetric")
    if kk["d_measured"] is not None:
        tol = float(lens_conf.get("step_depth_ratio_tol", 0.5))
        dev = abs(kk["d_measured"] - 1.0)
        if dev > tol and dev > 3.0 * (kk["d_err"] if np.isfinite(kk["d_err"]) else 0.0):
            rejections.append("step_depth_mismatch")

    # Physical scale.
    d_s = float(d_s_kpc if d_s_kpc is not None else lens_conf.get("source_distance_kpc", 8.0))
    if theta_star_uas is not None and theta_star_uas > 0:
        m = fit["models"]
        best_occ = fit["best_occult_kind"]
        rho_star_fitted = (best_occ == "occult_finite" and fit["rho_star_resolved"]
                           and fit["dbic_occult_finite_vs_occult"] >= dbic_min)
        if rho_star_fitted:
            theta_e = float(theta_star_uas) * 1e-3 / m["occult_finite"]["params"]["rho_star"]
            theta_e_kind = "from_finite_source"
        else:
            theta_e = theta_e_lower_bound(fit["a_max_unocculted"], theta_star_uas)
            theta_e_kind = "lower_bound_from_a_max"
        dens = density_bound(fit["rho_l"], theta_e, d_s, pi_e=pi_e,
                             grid=int(lens_conf.get("distance_grid", 200)),
                             density_floor_g_cc=float(lens_conf.get("density_floor_g_cc", 0.01)))
        dens["theta_e_kind"] = theta_e_kind
        dens["theta_star_uas"] = float(theta_star_uas)
        out["density"] = dens
        if dens["blend_moving_test_needed"]:
            pending.append("blend_moving_test")
        elif dens["natural_everywhere_on_near_branch"] or dens["natural_distance_max_kpc"] > 0:
            notes.append("a natural body at D_L <= "
                         f"{dens['natural_distance_max_kpc']:.3g} kpc matches the density floor")
    else:
        pending.append("density_bound_not_run")
        notes.append("no theta_star supplied; theta_E and the density bound not computed")

    # Achromaticity.
    p = fit["models"][fit["best_occult_kind"]]["params"]
    prim = hidden_image_fraction(t, f, e, p)
    if colour_lc is None:
        pending.append("colour_test_not_run")
    else:
        tc, fc, ec = _good_arrays(colour_lc, conf)
        col = hidden_image_fraction(tc, fc, ec, p) if tc.size else None
        if prim is None or col is None:
            pending.append("colour_test_not_run")
            notes.append("colour band lacks epochs across the occultation edge")
        else:
            z = colour_consistency(prim["d"], prim["d_err"], col["d"], col["d_err"])
            out["colour"] = {"band": colour_lc.band, "primary": prim, "colour": col, "z": z}
            if abs(z) > float(lens_conf.get("colour_deficit_sigma_max", 3.0)):
                rejections.append("colour_chromatic")
    out["hidden_image_fraction_primary"] = prim

    if occ_gates:
        out["tier"] = TIER_NO_OCCULTATION
    elif rejections or pending:
        out["tier"] = tier_interest
    else:
        out["tier"] = tier_candidate
    return out


# --------------------------------------------------------------------------------------
# Synthetic events (tests and the selftest)
# --------------------------------------------------------------------------------------

def synthesise_event(t0: float = 60000.0, tE: float = 20.0, u0: float = 0.2, fs: float = 1.0,
                     fb: float = 0.1, rho_l: float | None = None, rho_star: float | None = None,
                     cadence_min: float = 12.0, noise_frac: float = 0.01, seasons=None,
                     seed: int = 0, star_id: str = "synthetic", band: str = "F146",
                     n_samples: int = 64, anomaly=None, flux_zp_ab: float | None = None,
                     meta: dict | None = None) -> LightCurve:
    """A GBTDS-like light curve of a (possibly occulting, possibly finite) event.

    ``seasons`` is a list of (start, end) MJD windows; the default is one
    72-day season centred on t0.  Noise is ``noise_frac`` of the baseline flux
    with Poisson-like scaling in the peak; ``anomaly`` is an optional callable
    t -> additive flux (a one-sided bump for the asymmetry tests).
    """
    if seasons is None:
        seasons = [(t0 - 36.0, t0 + 36.0)]
    step = cadence_min / 1440.0
    t = np.concatenate([np.arange(a, b, step) for a, b in seasons]) if seasons else np.array([])
    params = {"t0": t0, "tE": tE, "u0": u0, "fs": fs, "fb": fb}
    if rho_l is not None and rho_star is not None:
        kind, params = "occult_finite", dict(params, rho_l=rho_l, rho_star=rho_star)
    elif rho_l is not None:
        kind, params = "occult", dict(params, rho_l=rho_l)
    elif rho_star is not None:
        kind, params = "finite", dict(params, rho_star=rho_star)
    else:
        kind = "paczynski"
    model = model_flux(t, params, kind, n_samples=n_samples)
    if anomaly is not None:
        model = model + np.asarray(anomaly(t), dtype=float)
    base = fs + fb
    err = noise_frac * np.sqrt(np.maximum(model, 1e-12) * base)
    rng = np.random.default_rng(seed)
    flux = model + rng.normal(0.0, 1.0, size=t.size) * err
    m = {"synthetic": True, "kind": kind, "truth": dict(params), "noise_frac": noise_frac}
    if meta:
        m.update(meta)
    return LightCurve(star_id, 268.0, -29.0, band, t, flux, err, survey="synthetic",
                      flux_unit="relative", flux_zp_ab=flux_zp_ab, time_system="synthetic",
                      dq=np.zeros(t.size, dtype=np.int64), exposure_s=None, meta=m)
