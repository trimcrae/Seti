"""Statistic 3 --- the **filter clock**: anomaly rate versus stellar age.

If an anomaly population is the residue of civilisations that arose and stopped,
its rate per star is a function of how long the host has existed.  Three shapes
mean three different things, and they are distinguishable:

* **rising**      --- the residue accumulates; nothing removes it.  The rate is
  a monotone integral of the origination history.
* **saturating**  --- every system that was going to acquire the signature has
  acquired it; the reservoir is exhausted.
* **turning over** --- the residue *decays* (structures grind down, waste heat
  cools, chemical tags diffuse), so the oldest hosts have lost it again.  This
  is the shape that carries a lifetime.

Age proxies, in decreasing order of directness
----------------------------------------------
1. a spectroscopic/isochrone ``age_gyr`` column, if the parent carries one;
2. ``alpha_fe`` --- the best chemical age indicator available at survey scale
   (alpha-enhanced = formed before Type Ia enrichment = old);
3. **kinematic heating**: the age--velocity-dispersion relation makes an old
   population kinematically hot, so |W| (or the total space velocity, or the
   tangential velocity when no radial velocity exists) is a noisy but unbiased
   monotone-in-expectation age proxy;
4. thick-disk / halo membership as a coarse three-level ordinal.

The confounders that must be controlled, and are
------------------------------------------------
* **Metallicity.**  Old stars are metal-poor and metal-poor stars host fewer
  giant planets.  A rate that rises with age could therefore be a planet
  occurrence--metallicity relation read backwards.  ``[Fe/H]`` is forced into
  the matched null's stratification covariates whenever it exists, so the age
  trend measured here is *at fixed metallicity*.
* **Scale height.**  Old populations sit at larger |z|, so the age test and the
  |z| gradient test are partially the same measurement.  Both the |z|-controlled
  and |z|-free versions are run and reported side by side; if they disagree, the
  honest reading is that the two are not separable in this sample.
* **Detectability.**  A kinematically hot star is a high-proper-motion star, and
  proper motion breaks cross-epoch positional matching --- a purely instrumental
  route to an age trend.  The parent-matched null absorbs this only insofar as
  the covariates capture it; the limitation is stated in ``docs/tidemark.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .gradient import bin_index, expected_quantile_edges, gradient_test, poisson_glm, rate_profile
from .ingest import numeric
from .nulls import MatchedNull, empirical_p

#: LSR solar motion (Schoenrich, Binney & Dehnen 2010), km/s.
_SOLAR_UVW = (11.1, 12.24, 7.25)


def age_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Attach every age proxy the parent sample can support.

    Adds (where the inputs exist): ``v_tan_kms``, ``U/V/W_kms``, ``v_tot_lsr_kms``,
    ``abs_w_kms``, ``pop_class`` (0 thin / 1 thick / 2 halo), ``age_proxy`` (the
    best available), and ``age_proxy_kind`` naming which one was used.
    """
    out = df.copy()
    plx = numeric(out, "parallax")
    dist = (numeric(out, "dist_pc") if "dist_pc" in out.columns
            else np.where(plx > 0, 1000.0 / plx, np.nan))
    pmra = numeric(out, "pmra")
    pmdec = numeric(out, "pmdec")
    out["v_tan_kms"] = 4.740857 * np.hypot(pmra, pmdec) * (dist / 1000.0)

    have_rv = "radial_velocity" in out.columns and \
        np.isfinite(pd.to_numeric(out["radial_velocity"], errors="coerce")).any()
    if have_rv:
        try:
            from ..panspermia.kinematics import phase_space_6d
            ps = phase_space_6d(out)
            for c in ("U_kms", "V_kms", "W_kms", "v_total_kms"):
                if c in ps.columns:
                    out[c] = ps[c].to_numpy()
            u = out["U_kms"].to_numpy(float) + _SOLAR_UVW[0]
            v = out["V_kms"].to_numpy(float) + _SOLAR_UVW[1]
            w = out["W_kms"].to_numpy(float) + _SOLAR_UVW[2]
            out["v_tot_lsr_kms"] = np.sqrt(u ** 2 + v ** 2 + w ** 2)
            out["abs_w_kms"] = np.abs(w)
            # Coarse population ordinal from the Toomre radius (km/s).
            vt = out["v_tot_lsr_kms"].to_numpy(float)
            out["pop_class"] = np.where(vt > 200.0, 2, np.where(vt > 70.0, 1, 0)).astype(float)
            out.loc[~np.isfinite(vt), "pop_class"] = np.nan
        except Exception as exc:                                # noqa: BLE001
            print(f"[tidemark] 6D kinematics unavailable: {exc!r}")
            have_rv = False

    # Best available proxy, most direct first.
    kind, proxy = None, None
    for col, label in (("age_gyr", "spectroscopic_age"),
                       ("alpha_fe", "alpha_enhancement"),
                       ("abs_w_kms", "kinematic_vertical"),
                       ("v_tot_lsr_kms", "kinematic_total"),
                       ("v_tan_kms", "kinematic_tangential")):
        if col in out.columns:
            v = pd.to_numeric(out[col], errors="coerce").to_numpy(float)
            if np.isfinite(v).sum() >= 100:
                kind, proxy = label, v
                break
    if proxy is not None:
        out["age_proxy"] = proxy
        out["age_proxy_kind"] = kind
    return out


def _shape_models(x: np.ndarray, n_obs: np.ndarray, e_exp: np.ndarray) -> dict:
    """Fit constant / linear / quadratic log-rate models and classify the shape."""
    s = np.std(x) or 1.0
    u = (x - np.mean(x)) / s
    fits = {}
    for order, label in ((0, "flat"), (1, "monotone"), (2, "curved")):
        Z = np.stack([u ** p for p in range(order + 1)], axis=1)
        theta, cov, conv = poisson_glm(n_obs, e_exp, Z)
        mu = e_exp * np.exp(np.clip(Z @ theta, -30, 30))
        ok = (e_exp > 0) & np.isfinite(mu) & (mu > 0)
        ll = float(np.sum(n_obs[ok] * np.log(mu[ok]) - mu[ok]))
        fits[label] = {"order": order, "loglike": ll, "aic": 2 * (order + 1) - 2 * ll,
                       "theta": [float(t) for t in theta], "converged": bool(conv)}
    lin, quad = fits["monotone"], fits["curved"]
    d_aic_linear = fits["flat"]["aic"] - lin["aic"]
    d_aic_curved = lin["aic"] - quad["aic"]

    shape, note = "flat", "no significant trend in rate with the age proxy"
    b1 = lin["theta"][1] if len(lin["theta"]) > 1 else float("nan")
    c1 = quad["theta"][1] if len(quad["theta"]) > 2 else float("nan")
    c2 = quad["theta"][2] if len(quad["theta"]) > 2 else float("nan")
    vertex = None
    if d_aic_linear > 2:
        shape = "rising" if b1 > 0 else "falling"
        note = "monotone trend preferred over constant"
        if d_aic_curved > 2 and np.isfinite(c2) and c2 != 0:
            vertex = float(-c1 / (2 * c2))
            # Classify from the *fitted curve over the sampled range*, not from
            # the sign of a coefficient: a quadratic can be formally concave with
            # its vertex just past the last bin while the rate is still climbing
            # there, and calling that a turnover would invent a lifetime.
            f = c1 * u + c2 * u ** 2
            i_max = int(np.argmax(f))
            decline = float(f[i_max] - f[-1])
            slope_lo = float(c1 + 2 * c2 * u[0])
            slope_hi = float(c1 + 2 * c2 * u[-1])
            if c2 < 0 and i_max < len(f) - 1 and decline > 0.15:
                shape = "turnover"
                note = ("rate peaks inside the sampled range and declines by "
                        f"{100 * (1 - np.exp(-decline)):.0f}% to the oldest bin")
            elif c2 < 0 and slope_hi > 0 and abs(slope_hi) < 0.4 * abs(slope_lo):
                shape = "saturating"
                note = "still rising at the oldest bin but decelerating strongly"
    return {"fits": fits, "delta_aic_linear_vs_flat": float(d_aic_linear),
            "delta_aic_curved_vs_linear": float(d_aic_curved),
            "shape": shape, "interpretation": note,
            "vertex_standardised": vertex}


def age_rate_test(parent: pd.DataFrame, anomaly_mask, *, covariates,
                  proxy_col: str = "age_proxy", n_bins: int = 8, n_null: int = 400,
                  control_z: bool = True, seed: int = 20260726) -> dict:
    """Anomaly rate versus an age proxy, with metallicity (and optionally |z|)
    matched out.  Returns the profile, the trend, and the shape classification."""
    if proxy_col not in parent.columns:
        return {"verdict": "NO_AGE_PROXY", "insufficient": True,
                "note": "the parent sample carries no usable age proxy "
                        "(no age, no [alpha/Fe], no proper motion, no radial velocity)"}
    x = pd.to_numeric(parent[proxy_col], errors="coerce").to_numpy(float)
    mask = np.asarray(anomaly_mask, bool)
    if int((mask & np.isfinite(x)).sum()) < 20:
        return {"verdict": "INSUFFICIENT_ANOMALIES", "insufficient": True,
                "n_anom_with_proxy": int((mask & np.isfinite(x)).sum())}

    # Metallicity is the confounder that must be inside the null, not outside it.
    covs = list(covariates)
    for c in ("feh", "mh", "m_h", "metallicity"):
        if c in parent.columns and c not in covs:
            covs.append(c)
            break
    results = {}
    variants = {"metallicity_matched": covs}
    if control_z and "abs_z_gal_kpc" in parent.columns:
        # Scale height grows with age; matching it isolates the part of the age
        # trend that is not simply the vertical structure of the disk.
        variants["metallicity_and_scaleheight_matched"] = covs + ["abs_z_gal_kpc"]

    for label, cov in variants.items():
        cov = [c for c in cov if c in parent.columns]
        try:
            null = MatchedNull(parent, mask, cov, forbid_cols=(proxy_col,),
                               seed=seed)
        except (ValueError, KeyError) as exc:
            results[label] = {"error": str(exc)}
            continue
        gt = gradient_test(x, null, name=proxy_col, n_bins=n_bins, n_null=n_null)
        edges = expected_quantile_edges(x, null.weights, n_bins)
        prof = rate_profile(x, null, edges=edges)
        b = bin_index(x, edges)
        nb = edges.size - 1
        n_obs = null.observed_binned(b, nb)
        e_exp = null.expected_binned(b, nb)
        centres = np.asarray(prof["centres"], float)
        shape = _shape_models(centres, n_obs, e_exp)

        # Calibrate the shape-selection statistic: how often does a matched null
        # with no age dependence prefer a trend this strongly?
        rng = np.random.default_rng(seed + 3)
        d_null = np.empty(n_null)
        for j, idx in enumerate(null.draws(n_null, rng)):
            bn = b[idx]
            cn = np.bincount(bn[bn >= 0], minlength=nb).astype(float)
            d_null[j] = _shape_models(centres, cn, e_exp)["delta_aic_linear_vs_flat"]
        results[label] = {
            "covariates": cov, "n_anom": gt.get("n_anom"),
            "profile": prof, "trend": {k: gt.get(k) for k in
                                       ("slope_ln_per_unit", "rate_ratio_per_unit",
                                        "headline_p", "mean_shift", "monotonicity",
                                        "terms")},
            "shape": shape,
            "shape_p_value": empirical_p(shape["delta_aic_linear_vs_flat"], d_null,
                                         tail="greater"),
            "null_diagnostics": null.diagnostics().as_dict(),
        }

    primary = results.get("metallicity_matched", {})
    consistent = None
    if len(results) > 1 and all("shape" in r for r in results.values()):
        shapes = {r["shape"]["shape"] for r in results.values()}
        consistent = len(shapes) == 1
    return {
        "verdict": "OK", "insufficient": False,
        "proxy": proxy_col,
        "proxy_kind": (str(parent["age_proxy_kind"].dropna().iloc[0])
                       if "age_proxy_kind" in parent.columns
                       and parent["age_proxy_kind"].notna().any() else None),
        "variants": results,
        "shape": (primary.get("shape") or {}).get("shape"),
        "shape_p_value": primary.get("shape_p_value"),
        "scaleheight_consistent": consistent,
        "significant": bool(np.isfinite(primary.get("shape_p_value") or np.nan)
                            and (primary.get("shape_p_value") or 1.0) < 0.05),
    }


__all__ = ["age_proxies", "age_rate_test"]
