"""Per-detection ephemeris residuals: the observable LOOM is actually built on.

The LSST alert packet carries, for every detection associated with a known minor
planet, the *observed-minus-predicted* offset from the MPC orbit:
``ephoffsetalongtrack`` and ``ephoffsetcrosstrack`` (plus the total
``ephoffset``), alongside the predicted rate of motion ``ephrate`` and the
heliocentric and topocentric distances at that epoch.  A non-gravitational
acceleration is precisely a *secular, along-track* growth of that offset — so the
residual time series is a direct measurement of the acceleration, independent of
whether MPC ever fitted an ``A2`` for the object.  That independence is the whole
reason this module exists: the ``mpc_orbits`` non-gravitational block is fitted
for a small minority of objects, so a channel that reads only ``A2`` sees only
the objects somebody already thought were interesting.

Three things are done here that a naive residual search does not do, and each
one kills a specific class of false positive.

1. Work in physical displacement, not arcsec
--------------------------------------------
The same along-track displacement subtends a different angle at different
topocentric distance, so an arcsec residual mixes the signal with the observing
geometry.  ``toporange`` is in the same row, so the conversion is free:
``s_km = theta_arcsec * D_topo_au * 1 au / 206265``.  Everything downstream is in
kilometres.

2. The timing degeneracy, which is separable and usually is the answer
---------------------------------------------------------------------
A clock or shutter-timing error ``dt`` produces an along-track offset of exactly
``rate * dt``: linear in ``ephrate``, identical for every object, and completely
degenerate with a real along-track acceleration in any single object's data.
Both quantities are columns in the same table, so the test is free and internal:
regress along-track offset on ``ephrate``.  A timing error appears as a *common
slope across objects* with no per-object structure; an acceleration is
uncorrelated with ``ephrate`` and grows with time since the orbit's epoch.  This
test runs before anything else and its slope is subtracted, not merely reported.

3. Ask which law the acceleration follows, not how big it is
-----------------------------------------------------------
Large non-gravitational acceleration in an inactive small body is not novel and
is not unexplained: Seligman et al. (2023) found seven such objects — the "dark
comets" — and the accepted reading is hidden outgassing.  What distinguishes
hidden outgassing from an engineered object is *what the acceleration does with
heliocentric distance*.  Water-ice sublimation follows JPL's standard
``g(r)`` — steeply falling, with a knee at ~2.8 au — whereas radiation pressure
and thermal recoil follow ``r^-2``, and something that holds a trajectory
follows neither.  With a multi-apparition arc the residual drift rate is measured
at several heliocentric distances, and the three laws are distinguishable.  That
is the discriminant, and it is the reason the channel is not a dark-comet search.

The fourth test, :func:`breakpoint_scan`, is the necrosignature transposed to
dynamics.  A Yarkovsky drift is constant; sublimation switches on and off but
brightens the object when it does.  An acceleration that *stops*, at a discrete
epoch, with no photometric change, is what a derelict looks like — the
``docs/derelict.md`` argument applied to an orbit instead of a light curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

AU_KM = 1.495978707e8
ARCSEC_PER_RAD = 206264.806247

# JPL's standard water-ice sublimation law, g(r) = alpha (r/r0)^-m [1+(r/r0)^n]^-k,
# normalised so g(1 au) = 1.  These five constants are the published JPL comet
# non-gravitational model and are not free parameters here.
G_COMET_ALPHA = 0.1112620426
G_COMET_R0 = 2.808
G_COMET_M = 2.15
G_COMET_N = 5.093
G_COMET_K = 4.6142


def g_comet(r_au) -> np.ndarray:
    """JPL water-ice sublimation scaling ``g(r)``, normalised to 1 at 1 au."""
    r = np.asarray(r_au, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = r / G_COMET_R0
        return G_COMET_ALPHA * x ** (-G_COMET_M) * (1.0 + x ** G_COMET_N) ** (-G_COMET_K)


def g_radiation(r_au, d: float = 2.0) -> np.ndarray:
    """Radiation-driven scaling ``(1 au / r)^d``; ``d = 2`` for Yarkovsky/SRP."""
    r = np.asarray(r_au, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 0, r ** (-float(d)), np.nan)


def g_constant(r_au) -> np.ndarray:
    """A distance-independent acceleration: what nothing natural does."""
    r = np.asarray(r_au, dtype=float)
    return np.where(np.isfinite(r), 1.0, np.nan)


# ---------------------------------------------------------------------------
# 0. Reconstructing the decomposition the mirror does not carry
# ---------------------------------------------------------------------------
# MEASURED 2026-07-30: `ephoffsetalongtrack` and `ephoffsetcrosstrack` are NULL for
# all 961,558 solar-system detections in the ALeRCE mirror, and `ephrate`,
# `ephratera`, `ephratedec` are identically zero.  Only the scalar `ephoffset` is
# populated.  Since the along-track/cross-track split is the channel's central
# discriminant (§3.2 of docs/loom.md), it is reconstructed here from what IS there:
# the observed position from `detection`, the predicted position from `ephra`/
# `ephdec`, and the direction of motion from the object's own track.
#
# The track direction is taken from neighbouring detections of the same object
# rather than from the state vectors, deliberately.  The state vectors are
# populated, but the alert schema does not state whether they are ecliptic or
# equatorial, and a 23.4-degree frame error would rotate along-track into
# cross-track and destroy exactly the quantity being measured.  Two detections of
# the same object half an hour apart define the track direction with no frame
# assumption at all.
def offset_from_positions(ra_obs, dec_obs, eph_ra, eph_dec):
    """Observed-minus-predicted offset in arcsec, as (delta_ra*cos(dec), delta_dec).

    Returned as a pair rather than a magnitude so it can be projected.  The
    ``cos(dec)`` term is applied here, matching the convention the alert's own
    ``ephoffsetra`` documents, and the RA difference is wrapped so an object near
    0h does not acquire a 360-degree residual.
    """
    ra_o = np.asarray(ra_obs, dtype=float)
    dec_o = np.asarray(dec_obs, dtype=float)
    ra_p = np.asarray(eph_ra, dtype=float)
    dec_p = np.asarray(eph_dec, dtype=float)
    dra = (ra_o - ra_p + 180.0) % 360.0 - 180.0
    dec_mid = np.radians(0.5 * (dec_o + dec_p))
    return (dra * np.cos(dec_mid) * 3600.0, (dec_o - dec_p) * 3600.0)


def track_direction(mjd, ra_deg, dec_deg, max_gap_days: float = 0.5):
    """Unit sky-plane direction of motion at each epoch, from neighbouring epochs.

    For each detection the direction is taken from the nearest detection within
    ``max_gap_days`` — within a night an object moves along its track, so the
    displacement between two detections half an hour apart *is* the track
    direction, to far better precision than the residual being measured.  Epochs
    with no neighbour inside the window return NaN, which propagates: an object
    observed once a night for a month has no measurable track direction and must
    be reported as untestable on this axis rather than assigned an arbitrary one.
    """
    t = np.asarray(mjd, dtype=float)
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    n = t.size
    ex = np.full(n, np.nan)
    ey = np.full(n, np.nan)
    good = np.isfinite(t) & np.isfinite(ra) & np.isfinite(dec)
    idx = np.flatnonzero(good)
    if idx.size < 2:
        return ex, ey
    order = idx[np.argsort(t[idx])]
    ts = t[order]
    for k, i in enumerate(order):
        best_j, best_dt = -1, np.inf
        for k2 in (k - 1, k + 1):
            if 0 <= k2 < order.size:
                dt = abs(ts[k2] - ts[k])
                if 0 < dt <= max_gap_days and dt < best_dt:
                    best_dt, best_j = dt, order[k2]
        if best_j < 0:
            continue
        dra = (ra[best_j] - ra[i] + 180.0) % 360.0 - 180.0
        dx = dra * math.cos(math.radians(0.5 * (dec[i] + dec[best_j])))
        dy = dec[best_j] - dec[i]
        # Orient along INCREASING TIME, always.  If the neighbour used is the
        # earlier one the displacement points backwards, and leaving it that way
        # would flip the along-track sign for half the epochs, so a real lag and a
        # real lead would cancel and a secular drift would average to nothing.
        if t[best_j] < t[i]:
            dx, dy = -dx, -dy
        norm = math.hypot(dx, dy)
        if norm <= 0:
            continue
        ex[i], ey[i] = dx / norm, dy / norm
    return ex, ey


def rotate_to_track(dx, dy, ex, ey):
    """Rotate a sky-plane offset into (along-track, cross-track).

    Cross-track is the perpendicular in the sky plane, ``(ex, ey) -> (-ey, ex)``.
    """
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    along = dx * ex + dy * ey
    cross = dx * (-ey) + dy * ex
    return along, cross


def decompose_offset(mjd, ra_obs, dec_obs, eph_ra, eph_dec,
                     off_ra=None, off_dec=None, max_gap_days: float = 0.5):
    """Along-track and cross-track components of the O-C offset, in arcsec.

    Only the *rotation* is missing from the mirror, not the offset.  Measured
    2026-07-30: ``ephoffsetalongtrack``/``ephoffsetcrosstrack`` are NULL for all
    961,558 rows, but ``ephoffsetra`` and ``ephoffsetdec`` **are** populated and
    reproduce ``ephoffset`` in quadrature to the last digit — so the survey's own
    offset vector is available and all that has to be supplied locally is the
    direction of motion to project it onto.

    ``off_ra``/``off_dec`` are those columns and are used when given.  They are
    preferred over differencing the positions because they are the survey's own
    numbers, computed at full precision before any rounding into the table, and
    they already carry the ``cos(dec)`` factor.  When absent the offset is
    recomputed from the positions, which was verified against the columns to
    better than a milliarcsecond on live rows.

    Returns ``(along, cross, total)``.  ``total`` exists so a caller can check it
    against ``ephoffset``: agreement validates the whole chain, and a disagreement
    means the sign or frame convention is wrong and nothing downstream should be
    believed.
    """
    if off_ra is not None and off_dec is not None:
        dx = np.asarray(off_ra, dtype=float)
        dy = np.asarray(off_dec, dtype=float)
        if not np.any(np.isfinite(dx)) or not np.any(np.isfinite(dy)):
            dx, dy = offset_from_positions(ra_obs, dec_obs, eph_ra, eph_dec)
    else:
        dx, dy = offset_from_positions(ra_obs, dec_obs, eph_ra, eph_dec)
    ex, ey = track_direction(mjd, ra_obs, dec_obs, max_gap_days=max_gap_days)
    along, cross = rotate_to_track(dx, dy, ex, ey)
    return along, cross, np.hypot(dx, dy)


# A real solar-system object cannot be this close.  0.005 au is 750,000 km, twice
# the lunar distance, so anything below it is the mirror's zero-fill rather than a
# measurement -- `toporange` and `heliorange` both go down to ~1e-8 au on rows where
# the geometry was not computed.  This floor matters because the arcsec-to-km
# conversion is PROPORTIONAL to the range: a zero-filled range silently converts a
# real angular residual into a zero physical displacement, which reads as a clean
# null on an object that was never measured.
MIN_PHYSICAL_RANGE_AU = 0.005


def usable_range(range_au, min_au: float = MIN_PHYSICAL_RANGE_AU) -> np.ndarray:
    """Range in au with the mirror's zero-fill replaced by NaN."""
    d = np.asarray(range_au, dtype=float)
    return np.where(np.isfinite(d) & (d >= float(min_au)), d, np.nan)


def arcsec_to_km(offset_arcsec, toporange_au) -> np.ndarray:
    """Angular offset to physical along-track displacement (km).

    The range is passed through :func:`usable_range` first, so a zero-filled
    geometry propagates as NaN rather than collapsing the signal to zero.
    """
    th = np.asarray(offset_arcsec, dtype=float)
    d = usable_range(toporange_au)
    with np.errstate(divide="ignore", invalid="ignore"):
        return th / ARCSEC_PER_RAD * d * AU_KM


@dataclass
class Fit:
    """One residual-model fit, with its degradation stated explicitly."""

    name: str
    coeffs: dict = field(default_factory=dict)
    chi2: float = float("nan")
    dof: int = 0
    n: int = 0
    ok: bool = False
    reason: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def chi2_red(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else float("nan")

    def as_dict(self) -> dict:
        d = {"name": self.name, "chi2": self.chi2, "dof": self.dof,
             "chi2_reduced": self.chi2_red, "n": self.n, "ok": self.ok,
             "reason": self.reason}
        d.update({f"coeff_{k}": v for k, v in self.coeffs.items()})
        d.update(self.detail)
        return d


def _wls(A: np.ndarray, y: np.ndarray, sigma: np.ndarray):
    """Weighted least squares; returns (params, covariance, chi2)."""
    w = 1.0 / np.asarray(sigma, dtype=float)
    Aw = A * w[:, None]
    yw = y * w
    # lstsq rather than a normal-equation inverse: the design matrix for a
    # quadratic in time over a short arc is badly conditioned, and the normal
    # equations square that condition number.
    p, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    resid = yw - Aw @ p
    chi2 = float(resid @ resid)
    try:
        cov = np.linalg.inv(Aw.T @ Aw)
    except np.linalg.LinAlgError:
        cov = np.full((A.shape[1], A.shape[1]), np.nan)
    return p, cov, chi2


# ---------------------------------------------------------------------------
# 1. The timing degeneracy
# ---------------------------------------------------------------------------
@dataclass
class TimingSolution:
    """A common timing offset fitted across the whole sample."""

    dt_seconds: float = float("nan")
    dt_seconds_err: float = float("nan")
    variance_explained: float = float("nan")
    n: int = 0
    ok: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        return {"dt_seconds": self.dt_seconds, "dt_seconds_err": self.dt_seconds_err,
                "variance_explained": self.variance_explained, "n": self.n,
                "ok": self.ok, "reason": self.reason}


def fit_common_timing(along_arcsec, ephrate_arcsec_per_min) -> TimingSolution:
    """Fit the single timing offset that best explains along-track residuals.

    A shutter or clock error ``dt`` puts every object off by ``rate * dt`` along
    its own direction of motion.  Because ``dt`` is a property of the *telescope*
    and ``rate`` is a property of the *object*, the systematic is identifiable
    from a population even though it is perfectly degenerate with acceleration in
    any single object.  The slope of the through-origin regression of offset on
    rate *is* ``dt``, and ``variance_explained`` says how much of the sample's
    along-track scatter it accounts for: a large value means the sample's
    residuals are dominated by timing and no per-object acceleration should be
    believed until it is removed.
    """
    y = np.asarray(along_arcsec, dtype=float)
    x = np.asarray(ephrate_arcsec_per_min, dtype=float)
    good = np.isfinite(y) & np.isfinite(x)
    sol = TimingSolution(n=int(good.sum()))
    if sol.n < 20:
        sol.reason = "fewer_than_20_usable_detections"
        return sol
    xs, ys = x[good], y[good]
    denom = float(xs @ xs)
    if denom <= 0:
        sol.reason = "ephrate_identically_zero"
        return sol
    slope = float(xs @ ys) / denom                    # arcsec per (arcsec/min)
    resid = ys - slope * xs
    ss_tot = float(ys @ ys)
    sol.dt_seconds = slope * 60.0                     # (arcsec/min) -> minutes -> s
    n = xs.size
    if n > 1:
        s2 = float(resid @ resid) / (n - 1)
        sol.dt_seconds_err = math.sqrt(s2 / denom) * 60.0
    sol.variance_explained = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    sol.ok = True
    return sol


def subtract_timing(along_arcsec, ephrate_arcsec_per_min,
                    dt_seconds: float) -> np.ndarray:
    """Remove a fitted common timing offset from along-track residuals."""
    y = np.asarray(along_arcsec, dtype=float)
    x = np.asarray(ephrate_arcsec_per_min, dtype=float)
    return y - x * (float(dt_seconds) / 60.0)


def per_object_rate_correlation(along_arcsec, ephrate) -> float:
    """Correlation of one object's residuals with its own predicted rate.

    High correlation *within* an object means its residual tracks how fast it was
    moving at each epoch, which is the timing signature and not an acceleration.
    A real secular drift correlates with *time*, and ``ephrate`` varies with
    observing geometry rather than monotonically, so the two are separable even
    per object when the arc samples a range of rates.
    """
    y = np.asarray(along_arcsec, dtype=float)
    x = np.asarray(ephrate, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 5 or np.std(x[good]) <= 0 or np.std(y[good]) <= 0:
        return float("nan")
    return float(np.corrcoef(x[good], y[good])[0, 1])


# ---------------------------------------------------------------------------
# 2. Secular drift in physical displacement
# ---------------------------------------------------------------------------
def drift_fit(mjd, along_km, sigma_km, epoch_mjd: float | None = None) -> dict:
    """Fit constant / linear / quadratic along-track displacement in time.

    A pure orbit-element error is *linear* in time (a wrong mean motion); a
    constant acceleration is *quadratic*.  So the quantity that carries the
    non-gravitational signal is the quadratic coefficient, and the evidence for
    it is the improvement in chi-squared over the linear model — not the size of
    the residual, which any badly determined orbit supplies for free.

    Returns the three fits plus ``delta_chi2_quadratic``, which is the statistic
    a caller should threshold; with Gaussian errors it is chi-squared with one
    degree of freedom.
    """
    t = np.asarray(mjd, dtype=float)
    y = np.asarray(along_km, dtype=float)
    s = np.asarray(sigma_km, dtype=float)
    good = np.isfinite(t) & np.isfinite(y) & np.isfinite(s) & (s > 0)
    out: dict = {"n": int(good.sum()), "fits": []}
    if good.sum() < 6:
        out["verdict"] = "TOO_FEW_EPOCHS"
        return out
    t, y, s = t[good], y[good], s[good]
    t0 = float(epoch_mjd) if epoch_mjd is not None and math.isfinite(
        _f(epoch_mjd)) else float(np.median(t))
    x = t - t0
    out["epoch_mjd"] = t0
    out["arc_days"] = float(t.max() - t.min())

    fits = {}
    for name, order in (("constant", 0), ("linear", 1), ("quadratic", 2)):
        A = np.column_stack([x ** k for k in range(order + 1)])
        p, cov, chi2 = _wls(A, y, s)
        f = Fit(name, n=int(x.size), chi2=chi2, dof=max(int(x.size) - (order + 1), 0))
        f.coeffs = {f"x{k}": float(p[k]) for k in range(order + 1)}
        f.detail = {f"err_x{k}": float(math.sqrt(cov[k, k]))
                    if np.isfinite(cov[k, k]) and cov[k, k] >= 0 else float("nan")
                    for k in range(order + 1)}
        f.ok = f.dof > 0
        fits[name] = f
        out["fits"].append(f.as_dict())

    q = fits["quadratic"]
    lin = fits["linear"]
    out["delta_chi2_quadratic"] = lin.chi2 - q.chi2
    a_km_day2 = 2.0 * q.coeffs.get("x2", float("nan"))   # y = ... + c2 x^2
    err_c2 = q.detail.get("err_x2", float("nan"))
    out["accel_km_per_day2"] = a_km_day2
    out["accel_km_per_day2_err"] = 2.0 * err_c2 if math.isfinite(err_c2) else float("nan")
    out["accel_snr"] = (abs(a_km_day2) / out["accel_km_per_day2_err"]
                        if math.isfinite(out["accel_km_per_day2_err"])
                        and out["accel_km_per_day2_err"] > 0 else float("nan"))
    # An acceleration in the along-track direction of magnitude A produces
    # displacement A t^2 / 2; convert to au/day^2 so it is comparable with A2.
    out["accel_au_per_day2"] = a_km_day2 / AU_KM
    out["chi2_reduced_quadratic"] = q.chi2_red
    out["verdict"] = "OK"
    return out


# ---------------------------------------------------------------------------
# 3. Which law does the acceleration follow?
# ---------------------------------------------------------------------------
LAWS = {"sublimation": g_comet, "radiation": g_radiation, "constant": g_constant}


def law_discrimination(mjd, along_km, sigma_km, heliorange_au,
                       min_r_span: float = 0.25) -> dict:
    """Which heliocentric-distance law best explains the residual drift?

    The drift *rate* between consecutive epochs is measured at a known
    heliocentric distance, so with an arc that spans a range of ``r`` the three
    candidate laws separate.  The discriminant that matters is sublimation
    (a knee near 2.8 au, steep inside it) against radiation (a clean ``r^-2``)
    against distance-independent — because that is exactly where the dark-comet
    explanation and an engineered one differ, and it is not something a magnitude
    threshold can ask.

    ``min_r_span`` refuses the test when the arc does not sample enough
    heliocentric range to distinguish the laws.  This is the honest outcome for
    most main-belt objects observed over a single apparition, and reporting
    ``INSUFFICIENT_R_SPAN`` is the point: the alternative is a fitted preference
    between three curves that are indistinguishable over the sampled interval.
    """
    t = np.asarray(mjd, dtype=float)
    y = np.asarray(along_km, dtype=float)
    s = np.asarray(sigma_km, dtype=float)
    r = np.asarray(heliorange_au, dtype=float)
    good = (np.isfinite(t) & np.isfinite(y) & np.isfinite(s) & (s > 0)
            & np.isfinite(r) & (r > 0))
    out: dict = {"n": int(good.sum())}
    if good.sum() < 8:
        out["verdict"] = "TOO_FEW_EPOCHS"
        return out
    t, y, s, r = t[good], y[good], s[good], r[good]
    order = np.argsort(t)
    t, y, s, r = t[order], y[order], s[order], r[order]
    dt = np.diff(t)
    keep = dt > 0
    if keep.sum() < 6:
        out["verdict"] = "TOO_FEW_DISTINCT_EPOCHS"
        return out
    rate = np.diff(y)[keep] / dt[keep]
    rate_err = np.sqrt(s[:-1][keep] ** 2 + s[1:][keep] ** 2) / dt[keep]
    r_mid = 0.5 * (r[:-1][keep] + r[1:][keep])
    span = float(r_mid.max() / r_mid.min() - 1.0)
    out["r_span_fraction"] = span
    out["r_min_au"] = float(r_mid.min())
    out["r_max_au"] = float(r_mid.max())
    if span < float(min_r_span):
        out["verdict"] = "INSUFFICIENT_R_SPAN"
        out["note"] = (f"heliocentric distance spans only {span:.2f} of itself "
                       f"over the arc; the three laws are not separable below "
                       f"{min_r_span:.2f} and no preference is reported")
        return out
    results = {}
    for name, law in LAWS.items():
        g = np.asarray(law(r_mid), dtype=float)
        ok = np.isfinite(g) & np.isfinite(rate) & np.isfinite(rate_err) & (rate_err > 0)
        if ok.sum() < 5:
            results[name] = {"chi2": float("nan"), "reason": "law_undefined_on_arc"}
            continue
        A = g[ok][:, None]
        p, cov, chi2 = _wls(A, rate[ok], rate_err[ok])
        results[name] = {"chi2": chi2, "dof": int(ok.sum() - 1),
                         "scale": float(p[0]),
                         "scale_err": float(math.sqrt(cov[0, 0]))
                         if np.isfinite(cov[0, 0]) and cov[0, 0] >= 0 else float("nan")}
    out["laws"] = results
    usable = {k: v for k, v in results.items() if math.isfinite(v.get("chi2", float("nan")))}
    if len(usable) < 2:
        out["verdict"] = "LAWS_NOT_EVALUABLE"
        return out
    best = min(usable, key=lambda k: usable[k]["chi2"])
    second = min((k for k in usable if k != best), key=lambda k: usable[k]["chi2"])
    out["best_law"] = best
    out["next_law"] = second
    out["delta_chi2_vs_next"] = usable[second]["chi2"] - usable[best]["chi2"]
    dof = max(usable[best].get("dof", 0), 1)
    out["chi2_reduced_best"] = usable[best]["chi2"] / dof

    # The per-detection astrometric uncertainty is not in the alert, so the errors
    # used here are a floor and are certainly underestimated for some objects --
    # unmodelled perturbers, an undetected satellite's photocentre wobble, and
    # star-catalogue bias all add scatter that no formal error carries.  A raw
    # delta-chi-squared is therefore inflated by whatever common factor the errors
    # are wrong by, so the statistic is rescaled by the winning model's own reduced
    # chi-squared.  That is the standard remedy for a common error-scale error, it
    # is conservative, and without it every object with a slightly wrong sigma
    # would show a decisive law preference.
    scale = max(out["chi2_reduced_best"], 1.0)
    out["delta_chi2_scaled"] = out["delta_chi2_vs_next"] / scale
    out["error_scale_applied"] = scale
    out["verdict"] = ("LAW_PREFERRED" if out["delta_chi2_scaled"] >= 9.0
                      else "NO_LAW_PREFERRED")
    return out


# ---------------------------------------------------------------------------
# 3b. Residual STRUCTURE, which is the discriminant --- not residual amplitude
# ---------------------------------------------------------------------------
# Amplitude cannot be the discriminant, and this is the most important single
# constraint on the channel.  `ephOffset` is Rubin's position (astrometric
# precision ~10 mas) minus a prediction from an MPC orbit fitted to decades of
# heterogeneous historical astrometry whose star-catalogue biases reach 175 mas.
# Residuals of 0.1-1 arcsec are therefore routine and carry no information.  What
# does carry information is the residual's *geometry* and its *independence from
# the orbit's own quality*, both of which the alert ships alongside it.
def along_cross_partition(along_arcsec, cross_arcsec) -> dict:
    """Is the residual confined to the along-track direction, as physics requires?

    A transverse non-gravitational force changes the mean motion and therefore
    displaces the object *along its track*; the cross-track component is left
    essentially untouched.  The two dominant false-positive sources have no such
    preference: star-catalogue bias is a property of the reference frame and is
    isotropic on the sky, and a mis-association picks up a neighbouring source in
    a random direction.  So the ratio of along-track to cross-track power is a
    near-free discriminant, and it is one an amplitude cut cannot express.

    ``power_ratio`` near 1 means isotropic — reject.  Large means along-track
    dominated — the geometry an acceleration produces.
    """
    a = np.asarray(along_arcsec, dtype=float)
    c = np.asarray(cross_arcsec, dtype=float)
    good = np.isfinite(a) & np.isfinite(c)
    out: dict = {"n": int(good.sum())}
    if good.sum() < 5:
        out["verdict"] = "TOO_FEW_DETECTIONS"
        return out
    a, c = a[good], c[good]
    pa, pc = float(np.mean(a * a)), float(np.mean(c * c))
    out["rms_along_arcsec"] = math.sqrt(pa)
    out["rms_cross_arcsec"] = math.sqrt(pc)
    out["power_ratio"] = pa / pc if pc > 0 else float("inf")
    out["mean_along_arcsec"] = float(np.mean(a))
    out["mean_cross_arcsec"] = float(np.mean(c))
    # A coherent offset in one direction is stronger evidence than scatter in it,
    # so the mean-to-scatter ratio is reported separately per component.
    out["along_coherence"] = (abs(float(np.mean(a))) / float(np.std(a))
                              if np.std(a) > 0 else float("nan"))
    out["verdict"] = "OK"
    return out


def apparition_trend(mjd, along_arcsec, gap_days: float = 120.0) -> dict:
    """Does the along-track offset grow monotonically from apparition to apparition?

    An acceleration accumulates: each apparition's mean offset is larger than the
    last, with a consistent sign.  An orbit-fit error does not — it produces an
    offset whose sign and size depend on where in the fitted arc the epoch falls,
    and which therefore wanders rather than marches.  Testing per *apparition*
    rather than per detection is what makes this robust: within one apparition the
    offset is nearly constant and the many detections are not independent
    measurements of the trend.

    The statistic is Spearman's rank correlation of apparition mean against
    apparition index, which needs no assumption about the growth's functional form.
    """
    t = np.asarray(mjd, dtype=float)
    y = np.asarray(along_arcsec, dtype=float)
    good = np.isfinite(t) & np.isfinite(y)
    out: dict = {"n": int(good.sum())}
    if good.sum() < 6:
        out["verdict"] = "TOO_FEW_DETECTIONS"
        return out
    t, y = t[good], y[good]
    order = np.argsort(t)
    t, y = t[order], y[order]
    breaks = np.where(np.diff(t) > float(gap_days))[0] + 1
    groups = np.split(np.arange(t.size), breaks)
    means = np.array([float(np.mean(y[g])) for g in groups])
    epochs = np.array([float(np.mean(t[g])) for g in groups])
    out["n_apparitions"] = int(means.size)
    out["apparition_means_arcsec"] = means.tolist()
    out["apparition_epochs_mjd"] = epochs.tolist()
    if means.size < 3:
        out["verdict"] = "TOO_FEW_APPARITIONS"
        out["note"] = ("a trend across apparitions needs at least three; a single "
                       "apparition's offset is one measurement, however many "
                       "detections it contains")
        return out
    rank_x = np.argsort(np.argsort(epochs)).astype(float)
    rank_y = np.argsort(np.argsort(means)).astype(float)
    if np.std(rank_x) <= 0 or np.std(rank_y) <= 0:
        out["verdict"] = "DEGENERATE"
        return out
    out["spearman"] = float(np.corrcoef(rank_x, rank_y)[0, 1])
    out["sign_consistent"] = bool(np.all(means > 0) or np.all(means < 0))
    out["verdict"] = "OK"
    return out


def quality_independence(statistic, quality: dict, gate_keys=None) -> dict:
    """Is the anomaly statistic independent of how well each orbit is determined?

    The failure this prevents is the one KNELL documents and the one every blind
    non-gravitational search runs into: rank objects by residual and you rank them
    by how badly they were observed.  A majority of nominal S/N > 3 Yarkovsky
    detections in blind searches are spurious, and short arcs with few oppositions
    are the usual cause.  So the statistic is regressed, by rank, against every
    quality covariate the alert ships — ``arc_length_total``, ``nobs_total``,
    ``nopp``, ``u_param``, ``normalized_rms`` — and a strong correlation with any
    of them means the channel is measuring the survey, not the sky.

    Rank correlation rather than Pearson: the quality covariates are heavily
    skewed, and a single 40-year-arc object would otherwise set the slope.

    ``gate_keys`` selects which covariates may *invalidate* the result, as opposed
    to merely being reported.  Absolute magnitude belongs in the second group and
    the distinction is not cosmetic: the momentum ceiling is a function of ``H`` by
    construction, so an anomalous set being systematically faint is the signature's
    own shape and not a confounder — whereas the same correlation with arc length
    or observation count means the channel is ranking objects by how badly they
    were observed.  Both are reported; only the second decides anything.
    """
    s = np.asarray(statistic, dtype=float)
    gate = set(gate_keys) if gate_keys is not None else set(quality)
    out: dict = {"n": int(np.isfinite(s).sum()), "correlations": {},
                 "gated_on": sorted(gate),
                 "reported_only": sorted(set(quality) - gate)}
    if np.isfinite(s).sum() < 20:
        out["verdict"] = "TOO_FEW_OBJECTS"
        return out
    worst, worst_name = 0.0, None
    for name, col in quality.items():
        v = np.asarray(col, dtype=float)
        m = np.isfinite(s) & np.isfinite(v)
        if m.sum() < 20:
            out["correlations"][name] = None
            continue
        rx = np.argsort(np.argsort(s[m])).astype(float)
        ry = np.argsort(np.argsort(v[m])).astype(float)
        if np.std(rx) <= 0 or np.std(ry) <= 0:
            out["correlations"][name] = None
            continue
        r = float(np.corrcoef(rx, ry)[0, 1])
        out["correlations"][name] = r
        if name in gate and abs(r) > abs(worst):
            worst, worst_name = r, name
    out["max_abs_correlation"] = abs(worst)
    out["max_correlated_with"] = worst_name
    out["verdict"] = "OK"
    return out


# ---------------------------------------------------------------------------
# 4. The derelict signature: an acceleration that stops
# ---------------------------------------------------------------------------
def breakpoint_scan(mjd, along_km, sigma_km, n_null: int = 500,
                    min_per_segment: int = 4,
                    rng: np.random.Generator | None = None) -> dict:
    """Is a change in drift rate at one epoch better than a smooth quadratic?

    Yarkovsky drift is constant.  Sublimation switches on and off, but when it
    does the object brightens — that is what makes it a comet.  An acceleration
    that changes *discretely*, at one epoch, with no photometric change, is the
    dynamical form of the necrosignature this repository calls a derelict: a
    process that was running and stopped.

    The statistic is the chi-squared improvement of the best two-segment linear
    model over the single quadratic, and the null is generated by resampling the
    quadratic model's own residuals — which is essential, because scanning for
    the best breakpoint over many candidates finds an improvement in pure noise
    every time, and an uncalibrated delta-chi-squared would flag most objects.
    """
    t = np.asarray(mjd, dtype=float)
    y = np.asarray(along_km, dtype=float)
    s = np.asarray(sigma_km, dtype=float)
    good = np.isfinite(t) & np.isfinite(y) & np.isfinite(s) & (s > 0)
    out: dict = {"n": int(good.sum())}
    if good.sum() < 2 * min_per_segment + 3:
        out["verdict"] = "TOO_FEW_EPOCHS"
        return out
    t, y, s = t[good], y[good], s[good]
    order = np.argsort(t)
    t, y, s = t[order], y[order], s[order]
    x = t - float(np.median(t))

    A_q = np.column_stack([np.ones_like(x), x, x * x])
    p_q, _, chi2_q = _wls(A_q, y, s)
    model_q = A_q @ p_q
    resid_q = y - model_q

    def best_break(yy: np.ndarray) -> tuple[float, float]:
        best_d, best_t = -np.inf, float("nan")
        for i in range(min_per_segment, x.size - min_per_segment):
            xb = x[i]
            # Continuous piecewise-linear: intercept, slope, slope change.
            A = np.column_stack([np.ones_like(x), x, np.maximum(x - xb, 0.0)])
            _, _, chi2 = _wls(A, yy, s)
            _, _, c2q = _wls(A_q, yy, s)
            d = c2q - chi2
            if d > best_d:
                best_d, best_t = d, float(t[i])
        return best_d, best_t

    d_obs, t_break = best_break(y)
    rng = rng or np.random.default_rng(9393)
    null = np.empty(int(n_null))
    for i in range(int(n_null)):
        # Resample the quadratic fit's residuals with replacement and re-scan,
        # so the null carries the same scan-over-breakpoints advantage.
        yy = model_q + rng.choice(resid_q, size=resid_q.size, replace=True)
        null[i], _ = best_break(yy)
    out["delta_chi2"] = float(d_obs)
    out["break_mjd"] = t_break
    out["p_value"] = float((int((null >= d_obs).sum()) + 1) / (null.size + 1))
    out["null_median"] = float(np.median(null))
    out["p_resolution_floor"] = 1.0 / (int(n_null) + 1)
    out["verdict"] = "OK"
    return out


# ---------------------------------------------------------------------------
# 5. Sky-coherent systematics: star-catalogue bias
# ---------------------------------------------------------------------------
def sky_coherence(ra_deg, dec_deg, value, bin_deg: float = 5.0,
                  min_per_bin: int = 20) -> dict:
    """How much of a residual field is explained by *where on the sky* it was measured.

    Star-catalogue bias — the historically dominant systematic in minor-planet
    astrometry, with inter-catalogue position differences of tens of mas and up to
    175 mas — is coherent with sky position rather than with the object.  It is
    therefore separable exactly the way TOCSIN separates a deep-drilling field
    from a real transient rate: stack on the sphere and see how much of the
    variance the position accounts for.  A large value means the residual field
    is an astrometric reference problem and no object in it is a candidate.
    """
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    v = np.asarray(value, dtype=float)
    good = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(v)
    out: dict = {"n": int(good.sum()), "bin_deg": float(bin_deg)}
    if good.sum() < 4 * min_per_bin:
        out["verdict"] = "TOO_FEW_MEASUREMENTS"
        return out
    ra, dec, v = ra[good], dec[good], v[good]
    key = (np.floor(ra / bin_deg).astype(int) * 10_000
           + np.floor(dec / bin_deg).astype(int))
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    keep_bin = counts >= min_per_bin
    if keep_bin.sum() < 3:
        out["verdict"] = "TOO_FEW_POPULATED_BINS"
        return out
    sel = keep_bin[inv]
    v_s, inv_s = v[sel], inv[sel]
    means = np.zeros(uniq.size)
    for b in np.unique(inv_s):
        means[b] = float(np.mean(v_s[inv_s == b]))
    ss_tot = float(np.sum((v_s - v_s.mean()) ** 2))
    ss_within = float(np.sum((v_s - means[inv_s]) ** 2))
    out["n_bins"] = int(keep_bin.sum())
    out["variance_explained_by_sky_bin"] = (1.0 - ss_within / ss_tot
                                            if ss_tot > 0 else float("nan"))
    out["max_abs_bin_mean"] = float(np.max(np.abs(means[np.unique(inv_s)])))
    out["verdict"] = "OK"
    return out


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def residual_significance(offset_arcsec, normalized_rms, arc_length_days,
                          floor_arcsec: float = 0.05) -> np.ndarray:
    """Ephemeris residual in units of what that orbit's quality already implies.

    A large observed-minus-predicted offset is the *expected* outcome for a
    poorly-determined orbit, so a raw residual is not evidence of anything.  It is
    normalised here by the orbit's own fit quality (``normalized_rms``) and by a
    short-arc penalty, because residuals shrink as the arc lengthens.  Without
    this the channel would rank objects by how badly observed they are —
    precisely the mistake KNELL documents for cadence-degraded variables, where
    the uncorrected statistic flagged 22 of 24 unchanged stars.

    Reference scale: a well-observed main-belt object shows ``ephoffset`` at the
    ~0.1 arcsec level (Pan-STARRS1 astrometric RMS 0.12 arcsec; LSST per-epoch
    precision 10 mas with a 3-7 mas systematic floor), while a short-arc object
    can legitimately show several arcsec.
    """
    off = np.abs(np.asarray(offset_arcsec, dtype=float))
    rms = np.asarray(normalized_rms, dtype=float)
    arc = np.asarray(arc_length_days, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        arc_penalty = np.where(np.isfinite(arc) & (arc > 0),
                               np.sqrt(np.maximum(30.0 / arc, 1.0)), np.nan)
        scale = np.where(np.isfinite(rms) & (rms > 0), rms, 1.0) * arc_penalty
        scale = np.maximum(scale * floor_arcsec, floor_arcsec)
        return off / scale


def _f(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")
