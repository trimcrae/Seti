"""The OCCULT detector: an opaque lens in multi-site ground microlensing photometry.

Physics (reused from ``seti.roman.lens``, which derives it): a point lens that
is an opaque disc of radius ``rho_L`` (theta_E units) removes every image with
|theta| < rho_L.

* Wing regime, rho_L < 1: the minor image (|theta_-| < 1, falling toward 1/u)
  is hidden at every u > u_c = 1/rho_L - rho_L.  The light curve is the
  finite-source point-lens (FSPL) curve near the peak and drops by
  fs * A_-(u) outside u_c: a pair of DOWNWARD steps at
  t0 +- tE sqrt(u_c^2 - u0^2), each of depth fs * A_-(u_c).  Where the step
  is and how deep it is are fixed by the one number rho_L.  If u0 > u_c the
  minor image is hidden all the time and A_+ = (A + 1)/2 is *exactly* a
  blended Paczynski curve: undetectable.  Sensitivity therefore needs
  u0 < u_c.
* Central-hole regime, rho_L >= 1: the minor image is always hidden (a blend
  re-labelling, as above) and the major image is hidden for u < u_h =
  rho_L - 1/rho_L: the flux falls to the blend, BELOW the baseline, around t0.

Everything here is pure numpy/scipy.  Data travel as a :class:`Event` (all
datasets concatenated with an integer dataset index); each dataset has its
own linear source and blend flux (profiled, never fitted non-linearly) so
survey zero points (OGLE magnitudes, KMTNet difference fluxes) never matter.

The funnel (``assess_event``):

1. FSPL fit (t0, tE, u0, rho_*) with per-dataset fluxes, local-outlier
   clipping and per-dataset error renormalisation robust to steps.
2. A fast exact point-source scan of chi^2 improvement over every possible
   u_c (wing) and u_h (hole), each side of t0 separately, and of the
   ANTI-occultation (the same template with the sign flipped, which no
   physics produces --- its tail across the sample is the empirical null).
3. Full non-linear refinement of the best occultation and best
   anti-occultation (all shape parameters re-fitted, finite source).
4. The gates: symmetric step pair, depth tied to rho_L (alpha = 1),
   per-site and per-band agreement, night / dataset jack-knife, step
   bracketed by data, residuals clean after the fit (no brightening:
   caustics), and for the hole regime a dip genuinely below baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from ..roman.lens import (
    magnifications,
    occulted_magnification_finite,
    occulted_magnification_point,
    paczynski_finite,
    paczynski_magnification,
    u_crit,
)

# --------------------------------------------------------------------------------------
# Configuration (defaults; config/occult.json overrides)
# --------------------------------------------------------------------------------------

DEFAULT_CONF: dict = {
    "min_points": 60,            # an event needs this many good epochs in total
    "min_peak_snr": 10.0,        # FSPL amplitude detection, else NOT_LENSING
    "clip_sigma": 5.0,           # local-outlier clip (vs a running median of residuals)
    "clip_window": 11,           # points in the running median
    "err_floor_frac": 0.003,     # fractional error floor added in quadrature (x flux scale)
    "rho_star_max": 0.05,        # physical cap: a giant source at the bulge
    "rho_star_min": 1e-4,
    "dchi2_floor": 50.0,         # absolute floor on the refined Delta chi^2
    "side_dchi2_min": 16.0,      # each side of t0 must carry at least this much
    "edge_dchi2": 9.0,           # per-side edge-radius interval (3 sigma) that must overlap
    "alpha_nsig": 3.0,          # depth ratio alpha consistent with 1 at this many sigma
    "alpha_range": [0.5, 1.6],   # and inside this range
    "group_nsig": 3.0,           # sites / bands agree at this many sigma
    "group_min_detect_sigma": 2.0,   # a site "sees" the step if alpha/sigma exceeds this
    "jackknife_frac": 0.5,       # dropping any one night or dataset keeps >= this of Delta chi^2
    "max_night_share": 0.5,      # no single night supplies more than this share
    "bracket_days": 5.0,         # data within this of a step on both sides ...
    "bracket_frac_te": 0.1,      # ... or within this fraction of tE, whichever is larger
    "resid_redchi2_max": 2.0,    # after the occult fit, per-night binned reduced chi^2
    "bump_sigma": 6.0,           # a positive residual bump above this is a caustic, not a shadow
    "n_samples": 64,             # finite-source quadrature nodes
    "refine_top": 2,             # scan maxima refined per regime
    "seed_rho_l": [0.5, 0.7, 0.85, 1.3],   # always refined too (deep steps mislead a scan)
    "max_nfev": 300,
}


def conf_with(overrides: dict | None) -> dict:
    c = dict(DEFAULT_CONF)
    if overrides:
        c.update({k: v for k, v in overrides.items() if not k.startswith("_")})
    return c


# --------------------------------------------------------------------------------------
# Data container
# --------------------------------------------------------------------------------------

@dataclass
class Event:
    """All photometry of one microlensing event, datasets concatenated.

    ``datasets[k]`` is a dict with at least ``name``, ``site`` (OGLE, KMTA,
    KMTC, KMTS, MOA) and ``band`` (I, V, R); ``ds[i]`` is the dataset index of
    point i.  Fluxes are linear in the source flux in each dataset's own
    units (a magnitude series is converted with :func:`mag_to_flux`).
    """

    name: str
    t: np.ndarray
    f: np.ndarray
    e: np.ndarray
    ds: np.ndarray
    datasets: list
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.f = np.asarray(self.f, dtype=float)
        self.e = np.asarray(self.e, dtype=float)
        self.ds = np.asarray(self.ds, dtype=int)

    @property
    def n_ds(self) -> int:
        return len(self.datasets)

    def subset(self, mask) -> Event:
        mask = np.asarray(mask, dtype=bool)
        return Event(self.name, self.t[mask], self.f[mask], self.e[mask], self.ds[mask],
                     self.datasets, dict(self.meta))


def mag_to_flux(mag, err, zp: float = 18.0):
    """Magnitudes to linear flux relative to magnitude ``zp`` (flux 1 at zp)."""
    mag = np.asarray(mag, dtype=float)
    err = np.asarray(err, dtype=float)
    f = 10.0 ** (-0.4 * (mag - zp))
    return f, f * err * (0.4 * math.log(10.0))


def build_event(name: str, series: list, meta: dict | None = None) -> Event:
    """Concatenate ``series`` = [(dataset_dict, t, f, e), ...] into an Event."""
    ts, fs, es, ds, dsets = [], [], [], [], []
    for k, (d, t, f, e) in enumerate(series):
        t = np.asarray(t, dtype=float)
        ts.append(t)
        fs.append(np.asarray(f, dtype=float))
        es.append(np.asarray(e, dtype=float))
        ds.append(np.full(t.size, k, dtype=int))
        dsets.append(dict(d))
    if not ts:
        return Event(name, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, int), [], meta or {})
    ev = Event(name, np.concatenate(ts), np.concatenate(fs), np.concatenate(es),
               np.concatenate(ds), dsets, meta or {})
    o = np.argsort(ev.t, kind="stable")
    return Event(name, ev.t[o], ev.f[o], ev.e[o], ev.ds[o], dsets, ev.meta)


def clean_event(ev: Event) -> Event:
    """Drop non-finite or non-positive-error points; drop empty datasets' points."""
    ok = np.isfinite(ev.t) & np.isfinite(ev.f) & np.isfinite(ev.e) & (ev.e > 0)
    return ev.subset(ok)


# --------------------------------------------------------------------------------------
# Magnification models
# --------------------------------------------------------------------------------------

def traj(t, t0, te, u0):
    return np.sqrt(u0 * u0 + ((np.asarray(t, dtype=float) - t0) / te) ** 2)


def mag_fspl(u, rho, n_samples=64):
    if rho is None or rho < 1e-5:
        return paczynski_magnification(u)
    return paczynski_finite(u, float(rho), n_samples)


def mag_occult(u, rho_l, rho, n_samples=64, soft=0.0):
    if rho is None or rho < 1e-5:
        return occulted_magnification_point(u, float(rho_l), soft)
    return occulted_magnification_finite(u, float(rho_l), float(rho), n_samples, soft)


def mag_hybrid(u, kind: str, rho, rho_l, soft: float, n_samples=64):
    """Fast optimiser model: finite-source FSPL plus a point-source, soft-edged occultation term.

    The occultation term is paczynski(u) - occulted_point(u, rho_l, soft);
    the logistic edge of width ``soft`` (~ rho_*) stands in for the finite
    source's rounding of the step.  Used only inside the optimiser; every
    reported chi^2 is recomputed with the exact finite-source occulted model.
    """
    a_f = mag_fspl(u, rho, n_samples)
    corr = paczynski_magnification(u) - occulted_magnification_point(u, float(rho_l), soft)
    if kind == "occult":
        return a_f - corr
    if kind == "anti":
        return a_f + corr
    raise ValueError(kind)


def mag_model(u, kind: str, rho, rho_l=None, n_samples=64, soft=0.0):
    """kind: fspl | occult | anti (FSPL + the occulted light, the sign-flipped template)."""
    if kind == "fspl":
        return mag_fspl(u, rho, n_samples)
    occ = mag_occult(u, rho_l, rho, n_samples, soft)
    if kind == "occult":
        return occ
    if kind == "anti":
        a = mag_fspl(u, rho, n_samples)
        return 2.0 * a - occ
    raise ValueError(kind)


# --------------------------------------------------------------------------------------
# Linear flux profiling, per dataset
# --------------------------------------------------------------------------------------

def profile_fluxes(a, f, w, ds, n_ds):
    """Weighted per-dataset solve of f = fs_k a + fb_k; returns (fs[n_ds], fb[n_ds]).

    Unconstrained in fb (difference-flux zero points are arbitrary); a dataset
    whose design is degenerate (all a equal) gets fs = 0 and fb = its mean.
    """
    sw = np.bincount(ds, w, n_ds)
    swa = np.bincount(ds, w * a, n_ds)
    swaa = np.bincount(ds, w * a * a, n_ds)
    swf = np.bincount(ds, w * f, n_ds)
    swaf = np.bincount(ds, w * a * f, n_ds)
    det = swaa * sw - swa * swa
    good = np.isfinite(det) & (det > 1e-12 * np.maximum(swaa * sw, 1e-300))
    with np.errstate(divide="ignore", invalid="ignore"):
        fs = np.where(good, (swaf * sw - swa * swf) / det, 0.0)
        fb = np.where(good, (swaa * swf - swa * swaf) / det,
                      np.where(sw > 0, swf / np.maximum(sw, 1e-300), 0.0))
    return fs, fb


def _chi2_of(a, ev: Event, e=None):
    e = ev.e if e is None else e
    w = 1.0 / (e * e)
    fs, fb = profile_fluxes(a, ev.f, w, ev.ds, ev.n_ds)
    model = fs[ev.ds] * a + fb[ev.ds]
    r = ev.f - model
    return float(np.sum(r * r * w)), fs, fb, model


# --------------------------------------------------------------------------------------
# Non-linear fits
# --------------------------------------------------------------------------------------

@dataclass
class Fit:
    kind: str
    t0: float
    te: float
    u0: float
    rho: float
    rho_l: float | None
    chi2: float
    fs: np.ndarray
    fb: np.ndarray
    model: np.ndarray
    n_par: int
    converged: bool = True

    def params(self) -> dict:
        d = {"kind": self.kind, "t0": self.t0, "tE": self.te, "u0": self.u0, "rho_star": self.rho,
             "chi2": self.chi2, "n_par": self.n_par}
        if self.rho_l is not None:
            d["rho_l"] = self.rho_l
        return d


def evaluate(ev: Event, kind, t0, te, u0, rho, rho_l=None, n_samples=64, soft=0.0, e=None) -> Fit:
    u = traj(ev.t, t0, te, u0)
    a = mag_model(u, kind, rho, rho_l, n_samples, soft)
    chi2, fs, fb, model = _chi2_of(a, ev, e)
    npar = 4 + (1 if rho_l is not None else 0) + 2 * ev.n_ds
    return Fit(kind, float(t0), float(te), float(u0), float(rho), None if rho_l is None else
               float(rho_l), chi2, fs, fb, model, npar)


def _unpack(x, kind):
    t0, lte, u0, lrho = x[:4]
    rho_l = float(x[4]) if kind != "fspl" else None
    return float(t0), float(10 ** lte), float(abs(u0)), float(10 ** lrho), rho_l


def fit_model(ev: Event, kind: str, starts: list, conf: dict, rho_l_bounds=None) -> Fit:
    """least_squares from each start ``(t0, tE, u0, rho[, rho_l])``; best exact chi^2 wins.

    tE and rho are fitted in log10; the flux parameters are profiled.  For the
    occultation kinds a logistic edge of width ~ max(rho_*, 1e-3) regularises
    the hard edge inside the optimiser; the returned chi^2 is always the exact
    (hard-edge) value, followed by a 1-D exact scan of rho_l.
    """
    ns = int(conf.get("n_samples", 64))
    w = 1.0 / (ev.e * ev.e)
    lo = [ev.t.min() - 50.0, math.log10(0.3), 0.0, math.log10(conf["rho_star_min"])]
    hi = [ev.t.max() + 50.0, math.log10(1500.0), 3.0, math.log10(conf["rho_star_max"])]
    if kind != "fspl":
        rl_lo, rl_hi = rho_l_bounds or (0.05, 5.0)
        lo.append(rl_lo)
        hi.append(rl_hi)
    lo, hi = np.array(lo), np.array(hi)

    def resid(x):
        t0, te, u0, rho, rho_l = _unpack(x, kind)
        u = traj(ev.t, t0, te, u0)
        if kind == "fspl":
            a = mag_fspl(u, rho, ns)
        else:
            a = mag_hybrid(u, kind, rho, rho_l, max(rho, 2e-3), ns)
        fs, fb = profile_fluxes(a, ev.f, w, ev.ds, ev.n_ds)
        return (ev.f - fs[ev.ds] * a - fb[ev.ds]) / ev.e

    best = None
    for s in starts:
        t0, te, u0, rho = s[:4]
        x0 = [t0, math.log10(max(te, 0.31)), abs(u0), math.log10(max(rho, conf["rho_star_min"]))]
        if kind != "fspl":
            x0.append(s[4])
        x0 = np.clip(np.array(x0, dtype=float), lo + 1e-9, hi - 1e-9)
        try:
            sol = least_squares(resid, x0, bounds=(lo, hi), max_nfev=int(conf["max_nfev"]),
                                x_scale=np.array([max(te, 1.0) * 0.05, 0.05, 0.02, 0.2]
                                                 + ([0.02] if kind != "fspl" else [])),
                                ftol=1e-8, xtol=1e-8)
            x, ok = sol.x, bool(sol.success)
        except Exception:  # noqa: BLE001
            x, ok = x0, False
        t0, te, u0, rho, rho_l = _unpack(x, kind)
        fit = evaluate(ev, kind, t0, te, u0, rho, rho_l, ns)
        fit.converged = ok
        if best is None or fit.chi2 < best.chi2:
            best = fit
    if kind != "fspl" and best is not None:
        best = _exact_rho_l_scan(ev, best, ns)
    return best


def _exact_rho_l_scan(ev: Event, fit: Fit, ns: int, frac: float = 0.04, n: int = 81) -> Fit:
    """With a hard edge chi^2 is piecewise constant in rho_l: scan it exactly."""
    out = fit
    for r in fit.rho_l * np.linspace(1.0 - frac, 1.0 + frac, n):
        f2 = evaluate(ev, fit.kind, fit.t0, fit.te, fit.u0, fit.rho, r, ns)
        if f2.chi2 < out.chi2:
            out = f2
    out.converged = fit.converged
    return out


# --------------------------------------------------------------------------------------
# Initial guess, outlier clipping, error renormalisation
# --------------------------------------------------------------------------------------

def _running_median(x, w):
    w = max(int(w) | 1, 3)
    if x.size < w:
        return np.full(x.shape, np.median(x) if x.size else 0.0)
    pad = np.pad(x, (w // 2, w // 2), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, w), axis=1)


def initial_guesses(ev: Event, hint: dict | None = None) -> list:
    """Starts for the FSPL fit: the catalogue's (t0, tE, u0) if given, plus data-driven ones."""
    starts = []
    rho0 = 1e-3
    if hint and all(hint.get(k) is not None for k in ("t0", "tE", "u0")):
        h = (float(hint["t0"]), float(hint["tE"]), float(hint["u0"]))
        if np.isfinite(h).all() and h[1] > 0:
            starts += [(h[0], h[1], h[2], rho0), (h[0], h[1], max(h[2] * 0.5, 1e-3), rho0)]
    # data-driven: the brightest running-median point of the densest dataset
    k = int(np.bincount(ev.ds, minlength=ev.n_ds).argmax())
    m = ev.ds == k
    t, f = ev.t[m], ev.f[m]
    sm = _running_median(f, 7)
    i = int(np.argmax(sm))
    base = float(np.median(f))
    amp = sm[i] - base
    t0 = float(t[i])
    above = t[sm > base + 0.5 * amp] if amp > 0 else t[:0]
    fwhm = float(above.max() - above.min()) if above.size > 1 else 10.0
    for u0 in (0.05, 0.3, 0.8):
        # FWHM of a Paczynski curve ~ tE * 2 sqrt(...) ~ order u0-dependent; use a crude map
        te = max(fwhm / max(2.0 * math.sqrt(max(3.0 * u0 * u0, 0.05)), 0.3), 1.0)
        starts.append((t0, te, u0, rho0))
    return starts


def clip_local_outliers(ev: Event, model, conf) -> np.ndarray:
    """Mask of points kept: |r - running median of r| < clip * robust sigma, per dataset.

    Measured against a *running median of the residuals*, a step (which moves
    many consecutive points together) survives the clip; an isolated bad point
    does not.
    """
    keep = np.ones(ev.t.size, dtype=bool)
    r = (ev.f - model) / ev.e
    for k in range(ev.n_ds):
        m = np.where(ev.ds == k)[0]
        if m.size < 5:
            continue
        rr = r[m]
        d = rr - _running_median(rr, conf["clip_window"])
        s = 1.4826 * np.median(np.abs(d - np.median(d)))
        if not np.isfinite(s) or s <= 0:
            continue
        keep[m] = np.abs(d) < conf["clip_sigma"] * s
    return keep


def renorm_errors(ev: Event, model, conf) -> tuple[np.ndarray, list]:
    """Per-dataset error scale robust to steps, with a fractional floor.

    k_p2p from point-to-point differences (insensitive to steps and slow
    trends), k_mad from the median absolute residual; the LARGER is used, so
    a real step can only lower its own significance.
    """
    e = ev.e.copy()
    r = (ev.f - model) / ev.e
    scales = []
    for k in range(ev.n_ds):
        m = np.where(ev.ds == k)[0]
        if m.size < 5:
            scales.append(1.0)
            continue
        rr = r[m]
        k_p2p = 1.4826 * np.median(np.abs(np.diff(rr))) / math.sqrt(2.0)
        k_mad = 1.4826 * np.median(np.abs(rr - np.median(rr)))
        kk = float(max(k_p2p, k_mad, 0.3))
        scales.append(kk)
        fscale = float(np.median(np.abs(model[m]))) if m.size else 0.0
        e[m] = np.sqrt((ev.e[m] * kk) ** 2 + (conf["err_floor_frac"] * fscale) ** 2)
    return e, scales


# --------------------------------------------------------------------------------------
# Fast exact scans (point source, shape and fluxes held at the FSPL solution)
# --------------------------------------------------------------------------------------

def nuisance_jacobian(ev: Event, fit: Fit, e) -> np.ndarray:
    """Whitened Jacobian of the FSPL model in its nuisance parameters.

    Columns: d model / d(t0, tE, u0) (numerical, point source) and, per
    dataset, the source-flux (A) and blend-flux (1) columns.  Each row is
    divided by sigma_i.
    """
    t, ds = ev.t, ev.ds
    base = paczynski_magnification(traj(t, fit.t0, fit.te, fit.u0))
    cols = []
    for dp, j in ((max(1e-4 * fit.te, 1e-4), 0), (max(1e-4 * fit.te, 1e-4), 1), (1e-4, 2)):
        p = [fit.t0, fit.te, fit.u0]
        p[j] += dp
        a2 = paczynski_magnification(traj(t, p[0], p[1], abs(p[2])))
        cols.append(fit.fs[ds] * (a2 - base) / dp)
    for k in range(ev.n_ds):
        m = (ds == k).astype(float)
        cols.append(m * base)
        cols.append(m)
    return np.column_stack(cols) / e[:, None]


def _linear_scan(ev: Event, fit: Fit, d, e, above: bool, sign: float) -> dict:
    """Delta chi^2 of adding the template ``-sign * d * 1[u beyond threshold]`` to the FSPL fit.

    To first order in the nuisance parameters (t0, tE, u0 and every
    dataset's fluxes re-adjusting to absorb the template):

        Delta chi^2 = -2 s r'd - d'd + (J'd)' (J'J)^-1 (J'd)

    with r the whitened FSPL residual (orthogonal to J at the optimum), d the
    whitened template restricted to the thresholded points and J the
    whitened nuisance Jacobian.  Every term is a cumulative sum over points
    sorted by u, so every threshold costs O(p^2).  Without the last term a
    deep step is absorbed by the FSPL fit (tE shrinks) and the scan points at
    the wrong u_c; with it the scan sees what the refit will see.
    """
    u = traj(ev.t, fit.t0, fit.te, fit.u0)
    r = (ev.f - fit.model) / e
    dw = d / e
    jac = nuisance_jacobian(ev, fit, e)
    try:
        jtj_inv = np.linalg.pinv(jac.T @ jac, rcond=1e-10)
    except np.linalg.LinAlgError:
        jtj_inv = np.zeros((jac.shape[1], jac.shape[1]))
    base = -2.0 * sign * r * dw - dw * dw
    jd = jac * dw[:, None]
    out = {}
    for name, m in (("all", np.ones(u.size, bool)), ("rise", ev.t < fit.t0), ("fall", ev.t >= fit.t0)):
        if not m.any():
            out[name] = (np.zeros(0), np.zeros(0))
            continue
        uu = u[m]
        o = np.argsort(uu)
        us = uu[o]
        b = base[m][o]
        jj = jd[m][o]
        if above:
            cb = np.append(np.cumsum(b[::-1])[::-1][1:], 0.0)
            cj = np.vstack([np.cumsum(jj[::-1], axis=0)[::-1][1:], np.zeros((1, jj.shape[1]))])
        else:
            cb = np.cumsum(b)
            cj = np.cumsum(jj, axis=0)
        quad = np.einsum("ij,jk,ik->i", cj, jtj_inv, cj)
        out[name] = (us, cb + quad)
    return out


def scan_wing(ev: Event, fit: Fit, e=None, sign: float = 1.0) -> dict:
    """Delta chi^2 (improvement over FSPL) for every u_c, total and per side of t0.

    The occultation removes d_i = fs_k A_-(u_i) for u_i > u_c; ``sign=-1``
    ADDS it instead (the anti-occultation null, which no physics produces).
    """
    e = ev.e if e is None else e
    u = traj(ev.t, fit.t0, fit.te, fit.u0)
    _, am = magnifications(u)
    d = fit.fs[ev.ds] * am
    return _linear_scan(ev, fit, d, e, above=True, sign=sign)


def scan_hole(ev: Event, fit: Fit, e=None, sign: float = 1.0) -> dict:
    """Central-hole regime: flux falls to fb_true = fb - fs for u < u_h.

    Outside the hole the minor image is always gone and the curve is the FSPL
    fit relabelled (A_+ = (A+1)/2); inside it the deficit is fs (A + 1) in the
    fit's own flux units (fs' = fs_true/2).
    """
    e = ev.e if e is None else e
    u = traj(ev.t, fit.t0, fit.te, fit.u0)
    a = mag_fspl(u, fit.rho)
    d = fit.fs[ev.ds] * (a + 1.0)
    return _linear_scan(ev, fit, d, e, above=False, sign=sign)


def _scan_maxima(us, vals, u_lo, top: int, min_sep: float = 0.02):
    """Top local maxima of vals over thresholds us > u_lo, separated in u by min_sep (relative)."""
    ok = us > u_lo
    if not ok.any():
        return []
    us, vals = us[ok], vals[ok]
    order = np.argsort(vals)[::-1]
    picks = []
    for j in order:
        if vals[j] <= 0:
            break
        if all(abs(us[j] - p[0]) > min_sep * max(us[j], 1e-3) for p in picks):
            picks.append((float(us[j]), float(vals[j])))
        if len(picks) >= top:
            break
    return picks


def rho_l_from_uc(uc: float) -> float:
    """Inverse of u_c = 1/rho_l - rho_l on the wing branch (rho_l < 1)."""
    return float((-uc + math.sqrt(uc * uc + 4.0)) / 2.0)


def rho_l_from_uh(uh: float) -> float:
    """Inverse of u_h = rho_l - 1/rho_l on the hole branch (rho_l >= 1)."""
    return float((uh + math.sqrt(uh * uh + 4.0)) / 2.0)


def side_tests(ev: Event, fit: Fit, r0, dterm, e, ns: int = 64, span: float = 0.35,
               n_grid: int = 141, edge_dchi2: float = 9.0) -> dict:
    """Is the occultation there on BOTH sides of t0, equally deep, at the same u_c?

    Depth: alpha fitted on t < t0 and t >= t0 separately.  Edge: holding the
    refined shape and fluxes, the edge radius is scanned on each side alone
    over u_c (1 +- span); each side's interval (Delta chi^2 <= ``edge_dchi2``
    from its own minimum --- 3 sigma for the default; flat where the edge sits
    in a data gap) must overlap the other's.  A circum-lens disc, a one-sided systematic or an
    asymmetric (parallax / xallarap) residual fails one of the three.
    """
    out: dict = {}
    for side, m in (("rise", ev.t < fit.t0), ("fall", ev.t >= fit.t0)):
        out[side] = alpha_fit(r0, dterm, e, m)
        if out[side]["alpha"] is None:
            out[side]["snr"] = 0.0
    ar, af = out["rise"], out["fall"]
    if ar["alpha"] is not None and af["alpha"] is not None:
        out["alpha_z"] = float(abs(ar["alpha"] - af["alpha"]) / math.hypot(ar["sigma"], af["sigma"]))
    else:
        out["alpha_z"] = None
    uc = u_crit(fit.rho_l)
    wing = fit.rho_l < 1.0
    grid = uc * np.linspace(1.0 - span, 1.0 + span, n_grid)
    grid = grid[grid > (fit.u0 if wing else 0.0) + 1e-6]
    w = 1.0 / (e * e)
    u = traj(ev.t, fit.t0, fit.te, fit.u0)
    base = fit.fs[ev.ds]
    intervals = {}
    for side, m in (("rise", ev.t < fit.t0), ("fall", ev.t >= fit.t0)):
        if not m.any() or grid.size == 0:
            intervals[side] = None
            continue
        chis = []
        for g in grid:
            rl = rho_l_from_uc(g) if wing else rho_l_from_uh(g)
            a = mag_occult(u[m], rl, fit.rho, ns)
            mod = base[m] * a + fit.fb[ev.ds[m]]
            chis.append(float(np.sum((ev.f[m] - mod) ** 2 * w[m])))
        chis = np.array(chis)
        ok = chis <= chis.min() + edge_dchi2
        intervals[side] = [float(grid[ok].min()), float(grid[ok].max()),
                           float(grid[int(np.argmin(chis))])]
    out["edge_interval_rise"] = intervals["rise"]
    out["edge_interval_fall"] = intervals["fall"]
    ir, jf = intervals["rise"], intervals["fall"]
    out["edges_agree"] = bool(ir is not None and jf is not None
                              and ir[0] <= jf[1] and jf[0] <= ir[1])
    return out


# --------------------------------------------------------------------------------------
# Diagnostics of a refined occultation fit
# --------------------------------------------------------------------------------------

def occultation_term(ev: Event, fit: Fit, ns: int = 64) -> np.ndarray:
    """D_i = fs_k (A_unocculted - A_model) at the occultation solution (>= 0 for a shadow)."""
    u = traj(ev.t, fit.t0, fit.te, fit.u0)
    a0 = mag_fspl(u, fit.rho, ns)
    a1 = mag_model(u, fit.kind, fit.rho, fit.rho_l, ns)
    return fit.fs[ev.ds] * (a0 - a1)


def alpha_fit(r0, dterm, e, mask=None) -> dict:
    """Least-squares depth scale alpha of the occultation term: data = unocculted - alpha D."""
    m = np.ones(r0.size, bool) if mask is None else mask
    w = 1.0 / (e[m] ** 2)
    sdd = float(np.sum(dterm[m] ** 2 * w))
    if sdd <= 0:
        return {"alpha": None, "sigma": None, "snr": 0.0, "n": int(m.sum())}
    a = float(np.sum(-r0[m] * dterm[m] * w) / sdd)
    s = 1.0 / math.sqrt(sdd)
    return {"alpha": a, "sigma": s, "snr": a / s, "n": int(m.sum())}


def group_alphas(ev: Event, r0, dterm, e, key: str) -> dict:
    groups: dict = {}
    for k, d in enumerate(ev.datasets):
        groups.setdefault(str(d.get(key, "?")), []).append(k)
    out = {}
    for g, ks in groups.items():
        m = np.isin(ev.ds, ks)
        res = alpha_fit(r0, dterm, e, m)
        if res["alpha"] is not None and res["sigma"] < 5.0:
            out[g] = res
    return out


def groups_consistent(alphas: dict, nsig: float) -> dict:
    """Chi^2 of the group alphas about their weighted mean; max pairwise z."""
    vals = [(v["alpha"], v["sigma"]) for v in alphas.values() if v["alpha"] is not None]
    if len(vals) < 2:
        return {"n": len(vals), "consistent": None, "max_z": None}
    a = np.array([v[0] for v in vals])
    s = np.array([v[1] for v in vals])
    wm = float(np.sum(a / s**2) / np.sum(1 / s**2))
    z = np.abs(a - wm) / s
    return {"n": len(vals), "weighted_mean": wm, "max_z": float(z.max()),
            "consistent": bool(z.max() <= nsig)}


def jackknife(ev: Event, r0, r1, e) -> dict:
    """Delta chi^2 after dropping each night (integer day) and each dataset."""
    w = 1.0 / (e * e)
    per = (r0 * r0 - r1 * r1) * w
    tot = float(per.sum())
    night = np.floor(ev.t - 0.3).astype(np.int64)      # day boundary near local noon for Chile/SA/AU
    uniq, inv = np.unique(night, return_inverse=True)
    by_night = np.bincount(inv, per)
    by_ds = np.bincount(ev.ds, per, ev.n_ds)
    return {"dchi2": tot,
            "min_drop_night": float(tot - by_night.max()) if by_night.size else tot,
            "max_night_share": float(by_night.max() / tot) if tot > 0 and by_night.size else None,
            "worst_night_hjd": float(uniq[int(by_night.argmax())]) if by_night.size else None,
            "min_drop_dataset": float(tot - by_ds.max()) if by_ds.size else tot,
            "worst_dataset": ev.datasets[int(by_ds.argmax())]["name"] if by_ds.size else None}


def step_times(fit: Fit) -> list:
    uc = u_crit(fit.rho_l)
    if uc <= fit.u0:
        return []
    dt = fit.te * math.sqrt(uc * uc - fit.u0 * fit.u0)
    return [fit.t0 - dt, fit.t0 + dt]


def step_bracketing(ev: Event, fit: Fit, conf) -> list:
    """For each step: nearest epoch before and after, and whether both are close."""
    lim = max(conf["bracket_days"], conf["bracket_frac_te"] * fit.te)
    out = []
    for ts in step_times(fit):
        before = ev.t[ev.t < ts]
        after = ev.t[ev.t >= ts]
        gb = float(ts - before.max()) if before.size else float("inf")
        ga = float(after.min() - ts) if after.size else float("inf")
        out.append({"t_step": float(ts), "gap_before_d": gb, "gap_after_d": ga,
                    "bracketed": bool(gb <= lim and ga <= lim)})
    return out


def positive_bump(ev: Event, resid, e, t0: float, te: float, width_days: float | None = None) -> dict:
    """Largest positive excess of residuals binned by night inside |t - t0| < 3 tE.

    A binary-lens caustic crossing or cusp approach is a BRIGHTENING; after an
    occultation (or FSPL) fit it leaves a positive residual bump.  An opaque
    lens only removes light.
    """
    m = np.abs(ev.t - t0) < 3.0 * te
    if m.sum() < 5:
        return {"max_sigma": 0.0}
    t, r, w = ev.t[m], resid[m], 1.0 / e[m] ** 2
    wd = width_days or max(0.25, min(2.0, 0.05 * te))
    b = np.floor((t - t.min()) / wd).astype(int)
    sw = np.bincount(b, w)
    swr = np.bincount(b, w * r)
    ok = sw > 0
    mean = np.where(ok, swr / np.where(ok, sw, 1), 0.0)
    sig = mean * np.sqrt(sw)
    j = int(np.argmax(sig))
    return {"max_sigma": float(sig[j]), "t": float(t.min() + (j + 0.5) * wd),
            "min_sigma": float(sig[ok].min()) if ok.any() else 0.0}


def binned_redchi2(ev: Event, resid, e, t0, te) -> float:
    """Reduced chi^2 of night-binned residuals inside |t - t0| < 3 tE (catches red structure)."""
    m = np.abs(ev.t - t0) < 3.0 * te
    if m.sum() < 5:
        return 0.0
    night = np.floor(ev.t[m] - 0.3).astype(np.int64)
    key = night * 64 + ev.ds[m]
    uniq, inv = np.unique(key, return_inverse=True)
    w = 1.0 / e[m] ** 2
    sw = np.bincount(inv, w)
    swr = np.bincount(inv, w * resid[m])
    chi = (swr / sw) ** 2 * sw
    return float(np.mean(chi))


# --------------------------------------------------------------------------------------
# The funnel for one event
# --------------------------------------------------------------------------------------

TIER_NO_DATA = "NO_DATA"
TIER_NOT_LENSING = "NOT_LENSING"
TIER_NO_OCC = "NO_OCCULTATION"
TIER_REJECTED = "OCCULTATION_REJECTED"
TIER_CANDIDATE = "OCCULTATION_CANDIDATE"


def fit_fspl_clean(ev: Event, conf: dict, hint: dict | None = None):
    """FSPL fit with one round of local clipping and error renormalisation.

    Returns (clean_event, renormalised_errors, fit, info) or (None, None, None, info).
    """
    info: dict = {}
    ev = clean_event(ev)
    info["n_points_raw"] = int(ev.t.size)
    if ev.t.size < conf["min_points"]:
        info["reason"] = "too_few_points"
        return None, None, None, info
    starts = initial_guesses(ev, hint)
    fit = fit_model(ev, "fspl", starts, conf)
    if fit.u0 < 0.1:
        # a rho_* start far below the truth has no finite-source points in its
        # window and hence no gradient: seed the optimiser at larger sources too
        # (and at longer tE: a flattened peak fitted as a point source trades
        # tE for u0 at fixed tE*u0, and the optimiser cannot walk back out).
        # This happens BEFORE clipping, so a flattened peak is never clipped
        # away as "outliers" of a wrong point-source fit.
        fs_starts = [(fit.t0, fit.te, fit.u0, fit.rho)]
        fs_starts += [(fit.t0, fit.te * m, min(fit.u0 / m, 0.3 * r), r)
                      for r in np.geomspace(2e-3, 0.05, 8) for m in (1.0, 2.0, 4.0)]
        fit = fit_model(ev, "fspl", fs_starts, conf)
    keep = clip_local_outliers(ev, fit.model, conf)
    info["n_clipped"] = int((~keep).sum())
    ev = ev.subset(keep)
    fit = fit_model(ev, "fspl", [(fit.t0, fit.te, fit.u0, fit.rho)] + starts[:1], conf)
    e, scales = renorm_errors(ev, fit.model, conf)
    ev2 = Event(ev.name, ev.t, ev.f, e, ev.ds, ev.datasets, ev.meta)
    fit = fit_model(ev2, "fspl", [(fit.t0, fit.te, fit.u0, fit.rho)], conf)
    info["err_scales"] = {ev.datasets[k]["name"]: round(s, 3) for k, s in enumerate(scales)}
    info["n_points"] = int(ev.t.size)
    # detection of the lensing amplitude itself: flat vs FSPL
    flat_chi2, *_ = _chi2_of(np.ones(ev.t.size), ev2)
    info["lensing_dchi2"] = float(flat_chi2 - fit.chi2)
    info["peak_snr"] = float(math.sqrt(max(info["lensing_dchi2"], 0.0)))
    return ev2, e, fit, info


def _refine(ev, fit0, regime, uth, conf, kind="occult", shape_starts=()):
    rho_l0 = rho_l_from_uc(uth) if regime == "wing" else rho_l_from_uh(uth)
    bounds = (0.05, 0.9999) if regime == "wing" else (1.0001, 20.0)
    starts = [(fit0.t0, fit0.te, fit0.u0, fit0.rho, rho_l0)]
    starts += [(s[0], s[1], s[2], fit0.rho, rho_l0) for s in shape_starts]
    return fit_model(ev, kind, starts, conf, rho_l_bounds=bounds)


def excess_centroid(ev: Event) -> float | None:
    """Time centroid of the above-baseline flux in the densest dataset.

    For any time-symmetric light curve --- including a double-peaked one
    whose centre has been hollowed out --- this is t0, even when an FSPL fit
    has locked onto one of the two horns.
    """
    if ev.t.size == 0:
        return None
    k = int(np.bincount(ev.ds, minlength=ev.n_ds).argmax())
    m = ev.ds == k
    t, f = ev.t[m], _running_median(ev.f[m], 5)
    base = float(np.median(f))
    x = np.clip(f - base, 0.0, None)
    x = np.where(x > 3.0 * np.median(ev.e[m]), x, 0.0)
    return float(np.sum(t * x) / np.sum(x)) if np.sum(x) > 0 else None


def hollow_centre(ev: Event, tc: float | None, min_depth: float = 0.3) -> dict | None:
    """Does the densest dataset show two horns around ``tc`` with a trough between?

    The trigger for the (expensive) central-hole start grid: the running
    median must peak on BOTH sides of tc, and the level at tc must sit at
    least ``min_depth`` of the horn height below the lower horn.  Returns the
    horn separation (days) or None.
    """
    if tc is None:
        return None
    k = int(np.bincount(ev.ds, minlength=ev.n_ds).argmax())
    m = ev.ds == k
    t, f = ev.t[m], _running_median(ev.f[m], 7)
    base = float(np.median(f))
    left, right = t < tc, t > tc
    if left.sum() < 5 or right.sum() < 5:
        return None
    il = int(np.argmax(np.where(left, f, -np.inf)))
    ir = int(np.argmax(np.where(right, f, -np.inf)))
    horn = min(f[il], f[ir]) - base
    mid = (t > t[il]) & (t < t[ir])
    if horn <= 0 or mid.sum() < 3:
        return None
    trough = float(np.min(f[mid])) - base
    if trough > (1.0 - min_depth) * horn:
        return None
    return {"sep_days": float(t[ir] - t[il]), "horn": float(horn), "trough": trough}


def assess_event(ev: Event, conf: dict | None = None, hint: dict | None = None) -> dict:
    """Run the whole funnel on one event; returns a JSON-able verdict record."""
    conf = conf_with(conf)
    rec: dict = {"event": ev.name, "n_datasets": ev.n_ds,
                 "sites": sorted({d.get("site", "?") for d in ev.datasets}),
                 "bands": sorted({d.get("band", "?") for d in ev.datasets}),
                 "rejections": []}
    if ev.t.size == 0:
        rec["tier"] = TIER_NO_DATA
        return rec
    ev2, e, f0, info = fit_fspl_clean(ev, conf, hint)
    rec.update(info)
    if ev2 is None:
        rec["tier"] = TIER_NO_DATA
        return rec
    rec["fspl"] = f0.params()
    if info["peak_snr"] < conf["min_peak_snr"] or f0.te >= 1400:
        rec["tier"] = TIER_NOT_LENSING
        return rec
    n = ev2.t.size
    rec["redchi2_fspl"] = float(f0.chi2 / max(n - f0.n_par, 1))

    # ---- scans --------------------------------------------------------------
    sw = scan_wing(ev2, f0)
    sh = scan_hole(ev2, f0)
    sw_anti = scan_wing(ev2, f0, sign=-1.0)
    sh_anti = scan_hole(ev2, f0, sign=-1.0)
    u_max = float(traj(ev2.t, f0.t0, f0.te, f0.u0).max())
    rec["u_max_observed"] = u_max
    top = int(conf["refine_top"])
    cand = [("wing", u, v) for u, v in _scan_maxima(*sw["all"], f0.u0, top)]
    cand += [("hole", u, v) for u, v in _scan_maxima(*sh["all"], f0.u0, top)]
    anti = [("wing", u, v) for u, v in _scan_maxima(*sw_anti["all"], f0.u0, top)]
    anti += [("hole", u, v) for u, v in _scan_maxima(*sh_anti["all"], f0.u0, top)]
    rec["scan_best_dchi2"] = max([c[2] for c in cand], default=0.0)
    rec["scan_best_anti_dchi2"] = max([c[2] for c in anti], default=0.0)
    rec["scan_best"] = [list(c) for c in cand]
    # fixed seeds, so a scan misled by a very deep step cannot hide the answer
    for rl in conf.get("seed_rho_l", ()):
        uth = u_crit(rl)
        if f0.u0 < uth < u_max:
            cand.append(("wing" if rl < 1 else "hole", uth, -1.0))
            anti.append(("wing" if rl < 1 else "hole", uth, -1.0))

    # ---- refinement ---------------------------------------------------------
    # A central hole wrecks the FSPL fit (it locks onto a horn): give the hole
    # seeds extra shape starts centred on the excess-flux centroid.
    hole_starts = []
    if rec["redchi2_fspl"] > conf.get("poor_fit_redchi2", 3.0):
        tc = excess_centroid(ev2)
        hc = hollow_centre(ev2, tc)
        rec["hollow_centre"] = hc
        if hc is not None:
            sep = max(hc["sep_days"], 0.5)
            hole_starts = [(tc, sep * fac, u0) for u0 in (0.05, 0.2, 0.5) for fac in (0.3, 1.0, 3.0)]
            for rl in conf.get("hole_seed_rho_l", (1.1, 1.3, 1.8, 2.5)):
                cand.append(("hole", u_crit(rl), -2.0))
    rec["hole_starts_used"] = bool(hole_starts)

    def best_refined(cands, kind):
        best = None
        for regime, uth, v in cands:
            try:
                ft = _refine(ev2, f0, regime, uth, conf, kind,
                             hole_starts if (regime == "hole" and v == -2.0) else ())
            except Exception:  # noqa: BLE001
                continue
            if ft is None:
                continue
            d = f0.chi2 - ft.chi2
            if best is None or d > best[1]:
                best = (ft, d, regime)
        return best

    b_occ = best_refined(cand, "occult")
    b_anti = best_refined(anti, "anti")
    rec["dchi2_anti"] = float(b_anti[1]) if b_anti else 0.0
    if b_occ is None or b_occ[1] <= 0:
        rec["dchi2"] = 0.0
        rec["tier"] = TIER_NO_OCC
        return rec
    fo, dchi2, regime = b_occ
    rec["dchi2"] = float(dchi2)
    rec["regime"] = regime
    rec["occult"] = fo.params()
    rec["dbic"] = float(dchi2 - math.log(max(n, 2)))
    rec["u_c"] = float(u_crit(fo.rho_l))
    if dchi2 < conf["dchi2_floor"]:
        rec["tier"] = TIER_NO_OCC
        return rec

    # ---- diagnostics, then the gates (a pure function of them) --------------
    ns = int(conf["n_samples"])
    dterm = occultation_term(ev2, fo, ns)
    r0 = ev2.f - (fo.model + dterm)          # residual about the UN-occulted model
    r1 = ev2.f - fo.model                    # residual about the occultation model
    rec["alpha"] = alpha_fit(r0, dterm, e)
    rec["symmetry"] = side_tests(ev2, fo, r0, dterm, e, ns, edge_dchi2=conf["edge_dchi2"])
    sites = group_alphas(ev2, r0, dterm, e, "site")
    bands = group_alphas(ev2, r0, dterm, e, "band")
    rec["alpha_by_site"] = sites
    rec["alpha_by_band"] = bands
    rec["sites_consistency"] = groups_consistent(sites, conf["group_nsig"])
    rec["bands_consistency"] = groups_consistent(bands, conf["group_nsig"])
    rec["sites_seeing_step"] = [g for g, v in sites.items()
                                if v["snr"] >= conf["group_min_detect_sigma"]]
    rec["colour_tested"] = bool(rec["bands_consistency"]["n"] >= 2)
    rec["jackknife"] = jackknife(ev2, r0, r1, e)
    if regime == "wing":
        rec["steps"] = step_bracketing(ev2, fo, conf)
    else:
        inside = traj(ev2.t, fo.t0, fo.te, fo.u0) < rec["u_c"]
        rec["n_in_hole"] = int(inside.sum())
        # true-flux model: baseline fs + fb, hole level fb -> below baseline iff fs > 0
        rec["hole_below_baseline"] = bool(inside.any()
                                          and np.all(fo.fs[np.unique(ev2.ds[inside])] > 0))
    rec["positive_bump"] = positive_bump(ev2, r1, e, fo.t0, fo.te)
    rec["binned_redchi2_after"] = binned_redchi2(ev2, r1, e, fo.t0, fo.te)
    rec["rejections"] = gate_rejections(rec, conf)
    rec["tier"] = TIER_REJECTED if rec["rejections"] else TIER_CANDIDATE
    return rec


def gate_rejections(d: dict, conf: dict) -> list:
    """Every gate, as a pure function of the diagnostics in an assess record.

    (a) depth tied to rho_L (alpha = 1); (b) both sides of t0, equally deep,
    at one u_c --- the circum-lens disc and any one-sided systematic fail
    here; (c) >= 2 sites see it and agree; (d) bands agree (achromatic);
    (e) no single night or dataset carries it; (f) each wing step bracketed by
    data / the hole sampled and below baseline; (g) no brightening left
    (caustics brighten, a shadow only removes light) and no red residual;
    (h) the sign-flipped template does not do comparably well.
    """
    conf = conf_with(conf)
    rej = []
    al = d.get("alpha") or {}
    a, s = al.get("alpha"), al.get("sigma")
    lo, hi = conf["alpha_range"]
    if a is None or abs(a - 1.0) > conf["alpha_nsig"] * s + 0.1 or not (lo <= a <= hi):
        rej.append("DEPTH_NOT_TIED_TO_RHO_L")
    sym = d.get("symmetry") or {}
    side_snr = math.sqrt(conf["side_dchi2_min"])
    rs = (sym.get("rise") or {}).get("snr") or 0.0
    fs_ = (sym.get("fall") or {}).get("snr") or 0.0
    if not (rs >= side_snr and fs_ >= side_snr):
        rej.append("ONE_SIDED_STEP")
    elif sym.get("alpha_z") is not None and sym["alpha_z"] > conf["group_nsig"]:
        rej.append("SIDES_DIFFER_IN_DEPTH")
    if d.get("regime") == "wing" and not sym.get("edges_agree", False):
        rej.append("SIDES_DISAGREE_ON_U_C")
    if len(d.get("sites_seeing_step") or []) < 2:
        rej.append("FEWER_THAN_TWO_SITES_SEE_IT")
    if (d.get("sites_consistency") or {}).get("consistent") is False:
        rej.append("SITES_DISAGREE")
    if (d.get("bands_consistency") or {}).get("consistent") is False:
        rej.append("CHROMATIC")
    jk = d.get("jackknife") or {}
    tot = jk.get("dchi2") or 0.0
    if jk and (jk["min_drop_night"] < conf["jackknife_frac"] * tot
               or (jk.get("max_night_share") or 0.0) > conf["max_night_share"]):
        rej.append("ONE_NIGHT_DOMINATES")
    if jk and jk["min_drop_dataset"] < conf["jackknife_frac"] * tot:
        rej.append("ONE_DATASET_DOMINATES")
    if d.get("regime") == "wing":
        br = d.get("steps") or []
        if not br or not all(b["bracketed"] for b in br):
            rej.append("STEP_IN_DATA_GAP")
    elif d.get("regime") == "hole":
        if (d.get("n_in_hole") or 0) < 3:
            rej.append("HOLE_UNSAMPLED")
        if not d.get("hole_below_baseline"):
            rej.append("HOLE_NOT_BELOW_BASELINE")
    if (d.get("positive_bump") or {}).get("max_sigma", 0.0) >= conf["bump_sigma"]:
        rej.append("BRIGHTENING_LEFT_CAUSTIC_LIKE")
    if (d.get("binned_redchi2_after") or 0.0) > conf["resid_redchi2_max"]:
        rej.append("RESIDUAL_STRUCTURE")
    if (d.get("dchi2_anti") or 0.0) >= 0.5 * (d.get("dchi2") or 0.0):
        rej.append("ANTI_TEMPLATE_COMPARABLE")
    return rej


# --------------------------------------------------------------------------------------
# Synthesis (tests, positive controls) and analytic sensitivity
# --------------------------------------------------------------------------------------

def synth_event(t0=8000.0, te=25.0, u0=0.2, rho=2e-3, rho_l=None, sites=("KMTA", "KMTC", "KMTS"),
                bands=("I",), fs=1.0, fb=0.5, sigma=0.01, span=120.0, cadence_d=0.25,
                seed=0, kind="occult", extra=None, band_depth=None, site_depth=None) -> Event:
    """A multi-site synthetic event; ``rho_l=None`` gives the plain FSPL curve.

    ``extra(t) -> dA`` adds any magnification perturbation (a binary bump, a
    one-sided step) to the curve before noise.  ``band_depth`` / ``site_depth``
    scale the occultation term per band / site (a chromatic or one-site
    impostor).  Sites are phased by 8 h so the coverage is continuous, as
    KMTNet's is.
    """
    rng = np.random.default_rng(seed)
    series = []
    for j, site in enumerate(sites):
        for band in bands:
            step = cadence_d if band == "I" else cadence_d * 8
            t = np.arange(t0 - span, t0 + span, 1.0)[:, None] + (j / 3.0) + np.arange(0, 0.33, step)[None, :]
            t = t.ravel()
            u = traj(t, t0, te, u0)
            a = mag_model(u, kind, rho, rho_l) if rho_l is not None else mag_fspl(u, rho)
            scale = (band_depth or {}).get(band, 1.0) * (site_depth or {}).get(site, 1.0)
            if rho_l is not None and scale != 1.0:
                a0 = mag_fspl(u, rho)
                a = a0 + scale * (a - a0)
            if extra is not None:
                a = a + extra(t)
            fl = fs * a + fb
            err = np.full(t.size, sigma * (fs + fb))
            fl = fl + rng.normal(0.0, 1.0, t.size) * err
            series.append(({"name": f"{site}_{band}", "site": site, "band": band}, t, fl, err))
    return build_event("synthetic", series)


def expected_dchi2(ev: Event, fit: Fit, rho_l: float, e=None) -> float:
    """Fisher-style expected Delta chi^2 of an opaque lens of radius rho_l on this event.

    The occultation term evaluated at the event's own FSPL solution and errors,
    sum (D/sigma)^2: the chi^2 the template would buy if the event carried it
    (point source; shape re-fitting only lowers it, which injection measures).
    """
    e = ev.e if e is None else e
    u = traj(ev.t, fit.t0, fit.te, fit.u0)
    a0 = paczynski_magnification(u)
    a1 = occulted_magnification_point(u, rho_l)
    d = fit.fs[ev.ds] * (a0 - a1)
    return float(np.sum((d / e) ** 2))
