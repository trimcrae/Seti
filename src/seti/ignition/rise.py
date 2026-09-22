"""The IGNITION detector.  Pure functions; no network, no I/O.

Per star, per band, the epoch series (one point per NEOWISE visit, ~2 per year)
is reduced to:

(a) **Kendall tau** and its p-value on the scan-cleaned series;
(b) a **weighted linear slope** (mag/yr) and its significance, fitted jointly
    with a sinusoid whose period is confined to the NEOWISE scan-direction band
    (345--385 d); the sinusoid's amplitude is reported, because a rise that is
    mostly that sinusoid is the survey, not the star;
(c) the **total rise** --- first-epoch minus last-epoch magnitude on the
    scan-cleaned series --- in sigma;
(d) a **monotonicity fraction**: the fraction of consecutive epoch pairs that
    brighten, after the sinusoid is removed;
(e) a **shape test** separating a sustained rise from an impulsive born
    excess: a linear ramp against a step + exponential decay, reported as
    ``delta_bic = BIC(step+decay) - BIC(ramp)`` (positive prefers the ramp).

Sign convention: magnitudes.  A *rise* in flux is a *decrease* in magnitude, so
``slope_mag_yr < 0`` is brightening.  ``rise_mag`` and ``tau_rise`` are reported
with the sign flipped so that positive means "getting brighter".

The two-band rule (docs/channel-brief.md §4): a single-band anomaly is an
artefact until confirmed in the second band, so :func:`assess_star` requires
the same sign and significance in W1 and W2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import stats

DEFAULT_RISE: dict = {
    "min_epochs": 10,
    "min_baseline_yr": 5.0,
    "slope_sigma_min": 5.0,
    "rise_sigma_min": 3.0,
    "tau_p_max": 1.0e-3,
    "mono_frac_min": 0.6,
    "dbic_min": 6.0,
    "scan_amp_frac_max": 1.0 / 3.0,
    "scan_period_days": [345.0, 385.0],
    "scan_period_step_days": 5.0,
    "decay_tau_grid_yr": [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "growth_tau_grid_yr": [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0],
    "err_floor_mag": 0.005,
}

#: Reasons in priority order.  A star that fails several is labelled by the
#: first one in this list, so the veto counters partition the sample.
REASON_PRIORITY = (
    "insufficient_epochs", "short_baseline", "fading", "scan_systematic",
    "impulsive_strong", "slope_not_significant", "tau_not_significant",
    "endpoint_rise_not_significant", "impulsive_shape", "not_monotonic",
)

VERDICT_BY_REASON = {
    "insufficient_epochs": "INSUFFICIENT_EPOCHS",
    "short_baseline": "SHORT_BASELINE",
    "fading": "FADING",
    "impulsive_strong": "IMPULSIVE_SHAPE",
    "slope_not_significant": "NOT_RISING",
    "tau_not_significant": "NOT_RISING",
    "endpoint_rise_not_significant": "NOT_RISING",
    "scan_systematic": "SCAN_SYSTEMATIC",
    "impulsive_shape": "IMPULSIVE_SHAPE",
    "not_monotonic": "NOT_MONOTONIC",
}


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class BandRise:
    band: str
    n_epochs: int = 0
    baseline_yr: float = float("nan")
    t_first_yr: float = float("nan")
    t_last_yr: float = float("nan")
    mag_median: float = float("nan")
    # (a) Kendall tau on the scan-cleaned series, sign flipped: + = brightening
    tau_rise: float = float("nan")
    tau_p: float = float("nan")
    tau_raw: float = float("nan")
    # (b) joint ramp + scan-band sinusoid fit
    slope_mag_yr: float = float("nan")
    slope_err: float = float("nan")
    slope_sigma: float = float("nan")
    scan_period_d: float = float("nan")
    scan_amp_mag: float = float("nan")          # sinusoid amplitude in the ramp model
    scan_amp_step_mag: float = float("nan")     # ... and in the step + decay model
    chi2_red_ramp: float = float("nan")
    ramp_order: int = 0                         # 1 linear, 2 exponential growth
    growth_tau_yr: float = float("nan")         # e-folding time of the chosen exponential
    # (c) end-to-end rise on the cleaned series, + = brightening
    rise_mag: float = float("nan")
    rise_sigma: float = float("nan")
    # (d) monotonicity
    mono_frac: float = float("nan")
    # (e) shape: ramp vs step + exponential decay
    delta_bic: float = float("nan")
    step_amp_mag: float = float("nan")
    step_t0_yr: float = float("nan")
    step_tau_yr: float = float("nan")
    chi2_red_step: float = float("nan")
    rising: bool = False
    reasons: list[str] = field(default_factory=list)

    def as_dict(self, prefix: str = "") -> dict:
        d = asdict(self)
        d["reasons"] = ";".join(self.reasons)
        return {f"{prefix}{k}": v for k, v in d.items()}


@dataclass
class StarVerdict:
    verdict: str
    is_candidate: bool
    reasons: list[str]
    n_bands_rising: int
    rise_w2_minus_w1_mag: float = float("nan")
    grey_rise_suspect: bool = False

    def as_dict(self) -> dict:
        d = asdict(self)
        d["reasons"] = ";".join(self.reasons)
        return d


# --------------------------------------------------------------------------
# Least-squares machinery
# --------------------------------------------------------------------------
def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted least squares: (beta, cov, chi2)."""
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    chi2 = float(resid @ resid)
    try:
        cov = np.linalg.inv(Xw.T @ Xw)
    except np.linalg.LinAlgError:
        cov = np.full((X.shape[1], X.shape[1]), np.nan)
    return beta, cov, chi2


def scan_period_grid(conf: dict | None = None) -> np.ndarray:
    c = {**DEFAULT_RISE, **(conf or {})}
    lo, hi = (float(x) for x in c["scan_period_days"])
    step = float(c["scan_period_step_days"])
    return np.arange(lo, hi + 0.5 * step, step)


def _sin_cos(t_yr: np.ndarray, period_d: float):
    ph = 2.0 * np.pi * t_yr * 365.25 / period_d
    return np.sin(ph), np.cos(ph)


def fit_ramp_scan(t_yr, mag, err, conf: dict | None = None) -> dict:
    """Sustained ramp + a sinusoid confined to the scan-direction period band.

    Two ramp shapes are tried: linear (4 parameters with the sinusoid) and
    **exponential growth**, ``a + b (exp((t - t_first) / tau_g) - 1)`` with the
    e-folding time ``tau_g`` on a grid (5 parameters) --- the claim is
    exponential over decades, so a rise that accelerates must not be penalised
    for not being a straight line, and the exponential is monotone by
    construction (a bump can never pass as a ramp).  The reported ``slope`` is
    always the linear fit's (the quantity the brief asks for), with its error
    inflated by ``sqrt(max(1, chi2_red))`` so an underestimated epoch error
    cannot manufacture a significance; the shape test and the scan amplitude
    use whichever ramp has the lower BIC.  Returns the *scan-cleaned*
    magnitudes (data minus the fitted sinusoid of the chosen ramp).
    """
    c_ = {**DEFAULT_RISE, **(conf or {})}
    t = np.asarray(t_yr, float)
    m = np.asarray(mag, float)
    e = np.asarray(err, float)
    n = t.size
    w = 1.0 / e**2
    t0 = float(np.mean(t))
    tc = t - t0
    ln_n = np.log(max(n, 2))
    best_lin = best_exp = None
    for P in scan_period_grid(conf):
        s, c = _sin_cos(tc, P)
        X = np.column_stack([np.ones(n), tc, s, c])
        beta, cov, chi2 = _wls(X, m, w)
        if best_lin is None or chi2 < best_lin["chi2"]:
            best_lin = {"chi2": chi2, "beta": beta, "cov": cov, "period_d": float(P),
                        "sin": s, "cos": c, "k": 4, "tau_g": float("nan")}
        if n >= 8:
            for tau_g in c_["growth_tau_grid_yr"]:
                g = np.exp((t - t.min()) / float(tau_g)) - 1.0
                Xg = np.column_stack([np.ones(n), g, s, c])
                bg, cg, chi2g = _wls(Xg, m, w)
                if best_exp is None or chi2g < best_exp["chi2"]:
                    best_exp = {"chi2": chi2g, "beta": bg, "cov": cg, "period_d": float(P),
                                "sin": s, "cos": c, "k": 5, "tau_g": float(tau_g), "g": g}
    lin = best_lin
    chosen = lin
    if best_exp is not None and (best_exp["chi2"] + 5 * ln_n) < (lin["chi2"] + 4 * ln_n):
        chosen = best_exp
    dof = max(n - lin["k"], 1)
    chi2_red_lin = lin["chi2"] / dof
    infl = float(np.sqrt(max(1.0, chi2_red_lin)))
    cov = lin["cov"]
    slope_err = float(np.sqrt(cov[1, 1])) * infl if np.isfinite(cov[1, 1]) else float("nan")
    b = chosen["beta"]
    amp = float(np.hypot(b[2], b[3]))
    sinus = b[2] * chosen["sin"] + b[3] * chosen["cos"]
    ramp = b[0] + b[1] * (chosen["g"] if chosen["k"] == 5 else tc)
    return {"intercept": float(lin["beta"][0]), "slope": float(lin["beta"][1]),
            "slope_err": slope_err, "period_d": chosen["period_d"], "amp": amp,
            "chi2": float(chosen["chi2"]), "chi2_red": float(chosen["chi2"] / max(n - chosen["k"], 1)),
            "chi2_red_linear": float(chi2_red_lin), "k": int(chosen["k"]),
            "ramp_order": 1 if chosen["k"] == 4 else 2, "growth_tau_yr": chosen["tau_g"],
            "cleaned": m - sinus, "model": ramp + sinus, "t0": t0}


def fit_linear_slope(t_yr, mag, err, err_floor: float = 0.005) -> tuple[float, float]:
    """A bare weighted linear slope (mag/yr) and its error, inflated by sqrt(chi2_red).

    Not the detector --- the screen records it on the *uncorrected* series so
    the ensemble zero-point correction's effect is on the record star by star.
    """
    t = np.asarray(t_yr, float)
    m = np.asarray(mag, float)
    e = np.maximum(np.asarray(err, float), err_floor)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e)
    t, m, e = t[ok], m[ok], e[ok]
    if t.size < 3:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones(t.size), t - t.mean()])
    beta, cov, chi2 = _wls(X, m, 1.0 / e**2)
    infl = float(np.sqrt(max(1.0, chi2 / max(t.size - 2, 1))))
    err_b = float(np.sqrt(cov[1, 1])) * infl if np.isfinite(cov[1, 1]) else float("nan")
    return float(beta[1]), err_b


def _step_decay_basis(t: np.ndarray, t0: float, tau: float) -> np.ndarray:
    """Brightening step at ``t0`` relaxing back with e-folding ``tau`` (years)."""
    dt = t - t0
    return np.where(dt >= 0.0, np.exp(-np.clip(dt, 0.0, None) / tau), 0.0)


def fit_step_decay(t_yr, mag, err, period_d: float, conf: dict | None = None) -> dict:
    """Impulsive born excess: step + exponential decay, plus the same sinusoid.

    ``m(t) = a - A * H(t - t0) * exp(-(t - t0) / tau) + sinusoid`` with A >= 0
    (a brightening jump that fades back), t0 on the epoch grid and tau on a
    configurable grid.  Linear in (a, A, sin, cos) at fixed (t0, tau); a fit
    that wants A < 0 is refitted with A = 0.  Parameter count 6.
    """
    c = {**DEFAULT_RISE, **(conf or {})}
    t = np.asarray(t_yr, float)
    m = np.asarray(mag, float)
    e = np.asarray(err, float)
    n = t.size
    w = 1.0 / e**2
    tm = float(np.mean(t))
    s, cc = _sin_cos(t - tm, period_d)
    best = None
    # A step *before* the first epoch (pure decay) is a legitimate impact model too.
    t0_grid = np.concatenate([[t[0] - 0.25], t[:-1]])
    for t0 in t0_grid:
        for tau in c["decay_tau_grid_yr"]:
            basis = _step_decay_basis(t, float(t0), float(tau))
            X = np.column_stack([np.ones(n), -basis, s, cc])
            beta, _cov, chi2 = _wls(X, m, w)
            if beta[1] < 0.0:
                X0 = np.column_stack([np.ones(n), s, cc])
                b0, _c0, chi2 = _wls(X0, m, w)
                beta = np.array([b0[0], 0.0, b0[1], b0[2]])
            if best is None or chi2 < best["chi2"]:
                best = {"chi2": float(chi2), "amp": float(beta[1]), "t0": float(t0),
                        "tau": float(tau), "amp_scan": float(np.hypot(beta[2], beta[3]))}
    dof = max(n - 6, 1)
    best["chi2_red"] = best["chi2"] / dof
    best["k"] = 6
    return best


def delta_bic(ramp: dict, step: dict, n: int) -> float:
    """BIC(step + decay) - BIC(ramp).  Positive prefers the sustained ramp."""
    ln_n = np.log(max(n, 2))
    bic_ramp = ramp["chi2"] + ramp["k"] * ln_n
    bic_step = step["chi2"] + step["k"] * ln_n
    return float(bic_step - bic_ramp)


# --------------------------------------------------------------------------
# Per band
# --------------------------------------------------------------------------
def assess_band(t_yr, mag, err, band: str = "W1", conf: dict | None = None) -> BandRise:
    """All of (a)--(e) for one band's epoch series."""
    c = {**DEFAULT_RISE, **(conf or {})}
    t = np.asarray(t_yr, float)
    m = np.asarray(mag, float)
    e = np.asarray(err, float)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e)
    t, m, e = t[ok], m[ok], e[ok]
    order = np.argsort(t)
    t, m, e = t[order], m[order], e[order]
    e = np.maximum(e, float(c["err_floor_mag"]))
    out = BandRise(band=band, n_epochs=int(t.size))
    if t.size < 3:
        out.reasons = ["insufficient_epochs"]
        return out
    out.baseline_yr = float(t[-1] - t[0])
    out.t_first_yr, out.t_last_yr = float(t[0]), float(t[-1])
    out.mag_median = float(np.median(m))
    if t.size < max(5, int(c["min_epochs"]) // 2):
        # Too few points for a four-parameter fit to mean anything.
        out.reasons = ["insufficient_epochs"]
        return out

    ramp = fit_ramp_scan(t, m, e, c)
    cleaned = ramp["cleaned"]
    out.slope_mag_yr = ramp["slope"]
    out.slope_err = ramp["slope_err"]
    out.slope_sigma = (float(-ramp["slope"] / ramp["slope_err"])
                       if ramp["slope_err"] > 0 else float("nan"))
    out.scan_period_d = ramp["period_d"]
    out.scan_amp_mag = ramp["amp"]
    out.chi2_red_ramp = ramp["chi2_red"]
    out.ramp_order = int(ramp["ramp_order"])
    out.growth_tau_yr = float(ramp["growth_tau_yr"])

    tau, p = stats.kendalltau(t, cleaned)
    out.tau_raw = float(stats.kendalltau(t, m)[0])
    out.tau_rise = float(-tau) if np.isfinite(tau) else float("nan")
    out.tau_p = float(p) if np.isfinite(p) else float("nan")

    out.rise_mag = float(cleaned[0] - cleaned[-1])
    out.rise_sigma = float(out.rise_mag / np.hypot(e[0], e[-1]))
    out.mono_frac = float(np.mean(np.diff(cleaned) < 0.0))

    step = fit_step_decay(t, m, e, ramp["period_d"], c)
    out.delta_bic = delta_bic(ramp, step, t.size)
    out.step_amp_mag = step["amp"]
    out.step_t0_yr = step["t0"]
    out.step_tau_yr = step["tau"]
    out.chi2_red_step = step["chi2_red"]
    out.scan_amp_step_mag = step["amp_scan"]

    reasons: list[str] = []
    if t.size < int(c["min_epochs"]):
        reasons.append("insufficient_epochs")
    if out.baseline_yr < float(c["min_baseline_yr"]):
        reasons.append("short_baseline")
    rise_significant = True
    if not np.isfinite(out.slope_sigma) or out.slope_sigma < float(c["slope_sigma_min"]):
        rise_significant = False
        if np.isfinite(out.slope_sigma) and out.slope_sigma <= -float(c["slope_sigma_min"]):
            reasons.append("fading")
        else:
            reasons.append("slope_not_significant")
    if (not np.isfinite(out.tau_p) or out.tau_p > float(c["tau_p_max"])
            or not (out.tau_rise > 0.0)):
        rise_significant = False
        reasons.append("tau_not_significant")
    if out.rise_sigma < float(c["rise_sigma_min"]):
        rise_significant = False
        reasons.append("endpoint_rise_not_significant")
    # The scan-sinusoid rule qualifies a rise; it is not a stand-alone reason
    # (any sinusoid exceeds a third of a zero rise).  The amplitude is judged on
    # whichever model it was fitted alongside: a step fitted by the ramp model
    # leaks into the sinusoid, and that leak is not a scan systematic.
    scan_amp = float(min(out.scan_amp_mag, out.scan_amp_step_mag))
    if rise_significant and scan_amp > float(c["scan_amp_frac_max"]) * out.rise_mag:
        reasons.append("scan_systematic")
    if out.delta_bic < float(c["dbic_min"]):
        reasons.append("impulsive_shape")
        if out.delta_bic < -float(c["dbic_min"]):
            # The step + decay is not merely "not disfavoured": it is decisively
            # the better description.  That is what a born excess looks like.
            reasons.append("impulsive_strong")
    if out.mono_frac < float(c["mono_frac_min"]):
        reasons.append("not_monotonic")
    out.reasons = reasons
    out.rising = not reasons
    return out


# --------------------------------------------------------------------------
# Per star: the two-band rule
# --------------------------------------------------------------------------
def assess_star(w1: BandRise | None, w2: BandRise | None,
                conf: dict | None = None) -> StarVerdict:
    """Combine both bands.  Candidate only when W1 *and* W2 each pass."""
    bands = [b for b in (w1, w2) if b is not None]
    if len(bands) < 2:
        return StarVerdict("INSUFFICIENT_EPOCHS", False, ["band_missing"], 0)
    n_rising = int(w1.rising) + int(w2.rising)
    colour = float("nan")
    grey = False
    if np.isfinite(w1.rise_mag) and np.isfinite(w2.rise_mag):
        colour = float(w2.rise_mag - w1.rise_mag)
        # Dust or a radiator adds more to W2 than W1; a rise that is *greyer*
        # than the photosphere by a wide margin is a blend or an artefact.  A
        # flag, not a kill, because the epoch errors are not tiny.
        sig = float(np.hypot(w1.rise_mag / max(w1.rise_sigma, 1e-9),
                             w2.rise_mag / max(w2.rise_sigma, 1e-9)))
        grey = bool(colour < -2.0 * sig) if np.isfinite(sig) and sig > 0 else False
    if n_rising == 2:
        return StarVerdict("IGNITION_CANDIDATE", True, [], 2, colour, grey)
    if n_rising == 1:
        other = w2 if w1.rising else w1
        return StarVerdict("ONE_BAND_ONLY", False,
                           [f"{other.band.lower()}:{r}" for r in other.reasons], 1,
                           colour, grey)
    union = set(w1.reasons) | set(w2.reasons)
    for r in REASON_PRIORITY:
        if r in union:
            return StarVerdict(VERDICT_BY_REASON[r], False, sorted(union), 0, colour, grey)
    return StarVerdict("NOT_RISING", False, sorted(union), 0, colour, grey)


def assess_series(series: dict, conf: dict | None = None) -> tuple[dict, StarVerdict]:
    """Convenience: ``{"W1": (t, m, e), "W2": (t, m, e)}`` -> per-band + star verdicts."""
    per_band = {}
    for b in ("W1", "W2"):
        if b in series and series[b] is not None:
            t, m, e = series[b]
            per_band[b] = assess_band(t, m, e, band=b, conf=conf)
    return per_band, assess_star(per_band.get("W1"), per_band.get("W2"), conf)


# --------------------------------------------------------------------------
# Injection and sensitivity
# --------------------------------------------------------------------------
def inject_ramp(series: dict, amp_mag: float, over_yr: float = 10.0,
                colour_ratio_w2_w1: float = 1.0) -> dict:
    """Superpose a linear brightening of ``amp_mag`` per ``over_yr`` on both bands.

    ``colour_ratio_w2_w1`` scales the W2 amplitude relative to W1 (a warm excess
    is redder; 1.0 is the conservative grey case used for the sensitivity
    figure, since a grey ramp is the hardest to tell from a systematic).
    """
    out = {}
    for b, (t, m, e) in series.items():
        t = np.asarray(t, float)
        scale = colour_ratio_w2_w1 if b == "W2" else 1.0
        dm = -amp_mag * scale * (t - t.min()) / over_yr
        out[b] = (t, np.asarray(m, float) + dm, np.asarray(e, float))
    return out


def sensitivity_from_injections(star_series: list[dict], amps_mag, conf: dict | None = None,
                                over_yr: float = 10.0, max_stars: int = 200,
                                seed: int = 20260913) -> dict:
    """Recovery fraction of injected ramps on real (or synthetic) epoch series.

    Injections go on top of whatever the star already does, so the result is a
    statement about *this* sample's noise and cadence, not an idealised one.
    """
    rng = np.random.default_rng(seed)
    if len(star_series) > max_stars:
        idx = rng.choice(len(star_series), size=max_stars, replace=False)
        star_series = [star_series[int(i)] for i in idx]
    out: dict = {"n_stars": int(len(star_series)), "over_yr": over_yr, "amps": {}}
    for a in amps_mag:
        n_rec = 0
        for s in star_series:
            _pb, v = assess_series(inject_ramp(s, float(a), over_yr), conf)
            n_rec += int(v.is_candidate)
        n = len(star_series)
        out["amps"][f"{float(a):.2f}"] = {"n": int(n), "n_recovered": int(n_rec),
                                          "fraction": (n_rec / n) if n else float("nan")}
    return out


__all__ = ["DEFAULT_RISE", "REASON_PRIORITY", "VERDICT_BY_REASON", "BandRise", "StarVerdict",
           "assess_band", "assess_series", "assess_star", "delta_bic", "fit_linear_slope",
           "fit_ramp_scan", "fit_step_decay", "inject_ramp", "scan_period_grid",
           "sensitivity_from_injections"]
