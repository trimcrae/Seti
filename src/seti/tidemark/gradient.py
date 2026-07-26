"""Statistic 1 --- the **gradient**: does the anomaly rate per star vary
monotonically across a Galactic coordinate, after the selection function is
divided out?

The measurement, not just the p-value
-------------------------------------
The observable is the selection-corrected rate ratio

    rho(x) = n_obs(x) / E(x),      E(x) = sum of matched-null weights in bin x

and the claim is an **amplitude with a confidence interval**, obtained from a
Poisson regression of the binned counts with ``log E`` as a fixed offset:

    log mu_b = log E_b + alpha + beta * x_b

so ``beta`` is the log rate-ratio gradient per unit coordinate --- e.g. dex per
kpc of Galactocentric radius --- and ``exp(beta)`` is the multiplicative change
in anomaly rate per kpc.  Because the offset is the matched expectation, a
constant ``rho`` (no spatial structure) gives ``beta = 0`` **whatever** the
underlying stellar density profile or survey footprint looks like.

Three statistics are reported, deliberately of different character:

* ``mean_shift`` --- binless.  The mean coordinate of the anomalies versus its
  matched-null distribution.  No bin choices, no model; the assumption-light
  primary p-value.
* ``slope`` --- the Poisson-GLM amplitude above, with a parametric standard
  error *and* a Monte-Carlo-calibrated p-value (the MC one is authoritative: it
  carries the extra variance the matched-null resampling injects).
* ``monotonicity`` --- Spearman rank correlation of ``rho`` against ``x``, which
  catches a monotone-but-curved trend that a linear slope would understate.

Longitude is periodic, so for it the linear model is replaced by a harmonic one
(dipole, optionally + quadrupole).  The dipole *amplitude* answers "is there a
preferred direction?" and its *phase* answers "which one?" --- the form in which
the Cirkovic--Bradbury (outer rim) and settlement-front (inner Galaxy)
predictions actually differ.
"""

from __future__ import annotations

import numpy as np

from .nulls import MIN_ANOMALIES_PER_TEST, MatchedNull, empirical_p, insufficient, p_report


# --- binning ----------------------------------------------------------------
def expected_quantile_edges(x: np.ndarray, w: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin edges with equal *expected* anomaly count per bin.

    Equal-expectation bins keep the Poisson fit well conditioned and stop a
    handful of distant stars in a sparsely-populated outer bin from dominating.
    """
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() < 2:
        return np.array([np.nanmin(x), np.nanmax(x)])
    xs, ws = x[ok], w[ok]
    order = np.argsort(xs)
    xs, ws = xs[order], ws[order]
    cw = np.cumsum(ws)
    tot = cw[-1]
    if tot <= 0:
        return np.array([xs[0], xs[-1]])
    targets = np.linspace(0.0, tot, int(n_bins) + 1)[1:-1]
    inner = np.interp(targets, cw, xs)
    edges = np.unique(np.concatenate([[xs[0]], inner, [np.nextafter(xs[-1], np.inf)]]))
    return edges


def bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, edges.size - 2)
    return np.where(np.isfinite(x), b, -1).astype(np.int64)


# --- Poisson regression with a fixed log-exposure offset --------------------
def poisson_glm(counts: np.ndarray, expected: np.ndarray, design: np.ndarray,
                *, max_iter: int = 50, tol: float = 1e-8):
    """IRLS fit of ``log mu = log(expected) + design @ theta``.

    Returns ``(theta, cov, converged)``.  Bins with zero expectation carry no
    information and are dropped.
    """
    n = np.asarray(counts, float)
    e = np.asarray(expected, float)
    Z = np.atleast_2d(np.asarray(design, float))
    ok = np.isfinite(n) & np.isfinite(e) & (e > 0) & np.all(np.isfinite(Z), axis=1)
    n, e, Z = n[ok], e[ok], Z[ok]
    p = Z.shape[1]
    if n.size <= p:
        return np.full(p, np.nan), np.full((p, p), np.nan), False
    theta = np.zeros(p)
    theta[0] = np.log(max(n.sum(), 0.5) / e.sum())
    cov = np.full((p, p), np.nan)
    for _ in range(max_iter):
        eta = np.log(e) + Z @ theta
        mu = np.clip(np.exp(eta), 1e-12, 1e12)
        # Working response and weights (canonical log link).
        z = (Z @ theta) + (n - mu) / mu
        W = mu
        A = Z.T @ (Z * W[:, None])
        bvec = Z.T @ (W * z)
        try:
            new = np.linalg.solve(A, bvec)
            cov = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return theta, cov, False
        if not np.all(np.isfinite(new)):
            return theta, cov, False
        step = np.max(np.abs(new - theta))
        theta = new
        if step < tol:
            return theta, cov, True
    return theta, cov, False


def _design(xc: np.ndarray, periodic: bool, n_harm: int) -> tuple[np.ndarray, list]:
    if periodic:
        cols, names = [np.ones_like(xc)], ["const"]
        for k in range(1, int(n_harm) + 1):
            cols += [np.cos(k * np.radians(xc)), np.sin(k * np.radians(xc))]
            names += [f"cos{k}", f"sin{k}"]
        return np.stack(cols, axis=1), names
    return np.stack([np.ones_like(xc), xc], axis=1), ["const", "slope"]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


# --- the profile ------------------------------------------------------------
def rate_profile(coord: np.ndarray, null: MatchedNull, *, n_bins: int = 10,
                 edges: np.ndarray | None = None) -> dict:
    """Selection-corrected anomaly rate ratio ``rho`` per coordinate bin."""
    x = np.asarray(coord, float)
    w = null.weights
    if edges is None:
        edges = expected_quantile_edges(x, w, n_bins)
    b = bin_index(x, edges)
    nb = edges.size - 1
    n_obs = null.observed_binned(b, nb)
    e_exp = null.expected_binned(b, nb)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(e_exp > 0, n_obs / e_exp, np.nan)
        # Poisson (counting) error on rho, ignoring the null's own variance,
        # which the Monte Carlo adds back at the test stage.
        rho_err = np.where(e_exp > 0, np.sqrt(np.maximum(n_obs, 1.0)) / e_exp, np.nan)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return {"edges": edges.tolist(), "centres": centres.tolist(),
            "n_obs": n_obs.tolist(), "n_expected": e_exp.tolist(),
            "rho": rho.tolist(), "rho_err": rho_err.tolist(),
            "n_parent_per_bin": np.bincount(b[b >= 0], minlength=nb).tolist()}


# --- the test ---------------------------------------------------------------
def gradient_test(coord: np.ndarray, null: MatchedNull, *, name: str = "coord",
                  n_bins: int = 10, n_null: int = 500, periodic: bool = False,
                  n_harm: int = 1, seed: int | None = None) -> dict:
    """Full gradient test on one coordinate against a parent-matched null."""
    x = np.asarray(coord, float)
    mask = null.mask
    w = null.weights
    good = np.isfinite(x)
    n_anom = int((mask & good).sum())
    # Guard on the anomalies that have *this* coordinate, not on the catalogue's
    # total.  A dimming catalogue can carry 2555 anomalies of which 30 have a
    # parallax; a radial gradient fitted to the 30 while guarded by the 2555 is
    # a meaningless number wearing a small p-value.
    if n_anom < MIN_ANOMALIES_PER_TEST:
        return insufficient(
            f"only {n_anom} of {int(mask.sum())} anomalies have a finite "
            f"{name}; need {MIN_ANOMALIES_PER_TEST}",
            coordinate=name, n_anom=n_anom, n_anom_total=int(mask.sum()),
            headline_p=None)

    edges = expected_quantile_edges(x, w, n_bins)
    prof = rate_profile(x, null, edges=edges)
    b = bin_index(x, edges)
    nb = edges.size - 1
    e_exp = np.asarray(prof["n_expected"], float)
    n_obs = np.asarray(prof["n_obs"], float)
    centres = np.asarray(prof["centres"], float)

    Z, pnames = _design(centres, periodic, n_harm)
    theta, cov, conv = poisson_glm(n_obs, e_exp, Z)

    # Binless mean shift.
    t_obs = float(np.mean(x[mask & good]))
    wg = np.where(good, w, 0.0)
    t_exp = float(np.sum(wg * np.nan_to_num(x)) / wg.sum()) if wg.sum() > 0 else float("nan")
    rho_obs = np.asarray(prof["rho"], float)
    sp_obs = _spearman(centres, rho_obs)

    rng = np.random.default_rng(null.seed + 7 if seed is None else seed)
    t_null = np.empty(n_null)
    th_null = np.full((n_null, Z.shape[1]), np.nan)
    sp_null = np.empty(n_null)
    for j, idx in enumerate(null.draws(n_null, rng)):
        xi = x[idx]
        ok = np.isfinite(xi)
        t_null[j] = float(np.mean(xi[ok])) if ok.any() else np.nan
        bn = b[idx]
        cn = np.bincount(bn[bn >= 0], minlength=nb).astype(float)
        th, _, _ = poisson_glm(cn, e_exp, Z)
        th_null[j] = th
        with np.errstate(divide="ignore", invalid="ignore"):
            sp_null[j] = _spearman(centres, np.where(e_exp > 0, cn / e_exp, np.nan))

    def _pack(i: int, label: str) -> dict:
        v = float(theta[i]) if np.isfinite(theta[i]) else float("nan")
        nullv = th_null[:, i]
        sd_mc = float(np.nanstd(nullv))
        se = float(np.sqrt(cov[i, i])) if np.isfinite(cov[i, i]) and cov[i, i] > 0 else float("nan")
        return {"name": label, "value": v,
                "se_parametric": se, "se_monte_carlo": sd_mc,
                "z_monte_carlo": (v - float(np.nanmean(nullv))) / sd_mc if sd_mc > 0 else float("nan"),
                "p_monte_carlo": empirical_p(v, nullv, tail="two"),
                "ci95": [v - 1.96 * sd_mc, v + 1.96 * sd_mc] if sd_mc > 0 else [None, None]}

    sd_t = float(np.nanstd(t_null))
    out = {
        "coordinate": name, "insufficient": False, "n_anom": n_anom,
        "n_parent": int(len(x)), "n_null": int(n_null), "periodic": bool(periodic),
        "profile": prof,
        "mean_shift": {
            "observed": t_obs, "expected": t_exp,
            "null_mean": float(np.nanmean(t_null)), "null_std": sd_t,
            "z": (t_obs - float(np.nanmean(t_null))) / sd_t if sd_t > 0 else float("nan"),
            "p_value": empirical_p(t_obs, t_null, tail="two"),
        },
        "monotonicity": {
            "spearman_rho_vs_x": sp_obs,
            "p_value": empirical_p(sp_obs, sp_null, tail="two"),
        },
        "glm_converged": bool(conv),
        "terms": {pnames[i]: _pack(i, pnames[i]) for i in range(1, Z.shape[1])},
    }
    if periodic:
        c, s = theta[1], theta[2]
        amp = float(np.hypot(c, s))
        amp_null = np.hypot(th_null[:, 1], th_null[:, 2])
        out["dipole"] = {
            "amplitude_ln": amp,
            "amplitude_rate_ratio": float(np.exp(amp)),
            "phase_deg": float(np.degrees(np.arctan2(s, c)) % 360.0),
            "null_mean": float(np.nanmean(amp_null)), "null_std": float(np.nanstd(amp_null)),
            "p_value": empirical_p(amp, amp_null, tail="greater"),
        }
        out["headline_p"] = out["dipole"]["p_value"]
        out["amplitude"] = out["dipole"]["amplitude_rate_ratio"]
    else:
        sl = out["terms"]["slope"]
        out["headline_p"] = min(p for p in (sl["p_monte_carlo"],
                                            out["mean_shift"]["p_value"])
                                if np.isfinite(p)) if np.isfinite(sl["p_monte_carlo"]) \
            else out["mean_shift"]["p_value"]
        out["slope_ln_per_unit"] = sl["value"]
        out["slope_dex_per_unit"] = sl["value"] / np.log(10.0)
        out["rate_ratio_per_unit"] = float(np.exp(sl["value"])) if np.isfinite(sl["value"]) else None
        out["direction"] = ("increasing_outward" if sl["value"] > 0 else
                            "decreasing_outward") if np.isfinite(sl["value"]) else None
    # The tested coordinate is the one covariate the null makes no promise
    # about, so its residual imbalance travels with the p-value.
    out["coordinate_balance"] = null.coordinate_balance(x, name)
    hp = out.get("headline_p")
    out["p"] = p_report(hp if hp is not None else float("nan"), n_null)
    out["floor_limited"] = out["p"]["floor_limited"]
    out["p_repr"] = out["p"]["p_repr"]
    # "Significant" requires a *resolved* p-value. A p sitting on the Monte
    # Carlo floor is a bound; escalate n_null and ask again.
    out["significant"] = bool(np.isfinite(hp if hp is not None else np.nan)
                              and hp < 0.05 and not out["floor_limited"])
    out["verdict"] = ("FLOOR_LIMITED" if out["floor_limited"] else "OK")
    return out


__all__ = ["gradient_test", "rate_profile", "poisson_glm", "expected_quantile_edges",
           "bin_index"]
