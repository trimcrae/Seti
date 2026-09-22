"""Stellar ages for CRADLE: >= 2 INDEPENDENT indicators, or ``AGE_UNDETERMINED``.

HD 15407A is the warning written into the mission: a 2.1 Gyr isochrone age
against an 80 Myr AB Doradus membership.  One indicator is not an age.  The
cell therefore requires two independent indicators to say ``> 1 Gyr`` and
none to say ``< 1 Gyr``; a star with a single old indicator is
``AGE_UNDETERMINED`` and is never a candidate.

The indicators, each with its own verdict ``OLD`` / ``YOUNG`` / ``UNDETERMINED``:

``iso``   Gaia DR3 FLAME isochrone age (``age_flame_lower > 1 Gyr`` is OLD; the
          16th-percentile bound, not the point estimate).  Only where FLAME
          gives a finite interval.
``kin``   The age-velocity relation as a likelihood ratio.  The star's
          heliocentric velocity --- full ``U,V,W`` with a radial velocity, the
          tangential 2-vector without one --- is compared, in its own measured
          components, against the Aumer & Binney (2009) dispersion tensor
          ``sigma_i(tau)`` with the asymmetric drift, marginalised over a flat
          star-formation history separately over ``tau < 1 Gyr`` and
          ``tau > 1 Gyr``.  ``log10 LR >= +1`` is OLD (the velocity itself
          argues for it at 10:1), ``<= -1`` YOUNG.  A star with a small
          velocity is UNDETERMINED, never young: the kinematic test can certify
          age only by excess velocity.
``alpha`` GSP-Spec ``[alpha/Fe]``: a thick-disc star (``alphafe_lower > 0.15``,
          ``[M/H] < 0``) is > 8 Gyr old --- chemistry, independent of both.
``gyro``  A rotation period (Gaia ``vari_rotation_modulation`` or a VizieR
          rotation catalogue supplied by the enrichment stage) through the
          Mamajek & Hillenbrand (2008) gyrochrone; mostly a YOUTH detector
          because the Gaia table is populated by active stars.
``act``   Gaia ESP-CS Ca II IRT activity index: high is YOUNG.  Never counted
          as OLD (the inactive locus is not a calibrated age).

Vetoes (either is a kill regardless of the count):

* a young moving group / association within ``young_groups`` in XYZ + UVW
  (3D with a radial velocity, the tangential projection without), the
  Sco-Cen subgroups included;
* a star-forming region / Sco-Cen sky box at the region's parallax range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ossuary.acquire import gspspec_quality_ok
from ..ossuary.kinematics import _A_ICRS_TO_GAL, _K_AUYR_KMS, _triad, space_velocity

OLD, YOUNG, UNDET = "OLD", "YOUNG", "UNDETERMINED"

_trapz = getattr(np, "trapezoid", None) or np.trapz
CLASS_MATURE = "MATURE_2PLUS"
CLASS_UNDET = "AGE_UNDETERMINED"
CLASS_YOUNG = "YOUNG_VETO"

DEFAULT_AGES: dict = {
    "age_threshold_gyr": 1.0,
    "solar_motion_kms": [11.1, 12.24, 7.25],          # Schoenrich, Binney & Dehnen 2010
    # Aumer & Binney 2009: sigma_i(tau) = v10 * ((tau + tau1) / (10 + tau1))**beta, tau in Gyr
    "avr": {"U": {"v10": 41.899, "tau1": 0.001, "beta": 0.307},
            "V": {"v10": 28.000, "tau1": 0.715, "beta": 0.430},
            "W": {"v10": 23.831, "tau1": 0.001, "beta": 0.445}},
    "asymmetric_drift_k_kms": 80.0,                   # V_a = sigma_U^2 / k
    "age_grid_gyr": [0.02, 13.0, 260],
    "kin_log10_lr_old": 1.0,
    "kin_log10_lr_young": -1.0,
    "kin_min_components": 2,
    "alpha_fe_thick_min": 0.15,
    "alpha_mh_max": 0.0,
    "gyro": {"a": 0.407, "b": 0.325, "c": 0.495, "n": 0.566,   # Mamajek & Hillenbrand 2008
             "young_max_gyr": 0.7, "old_min_gyr": 1.5, "bv_min": 0.55},
    "activity_young_min_nm": 0.3,
    "young_group_vel_tol_kms": 6.0,
    "young_group_pos_scale": 1.0,
    "young_groups": [],           # filled from config/cradle.yaml
    "young_regions": [],          # sky boxes with a parallax range
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _num(d: pd.DataFrame, col: str) -> np.ndarray:
    if col not in d.columns:
        return np.full(len(d), np.nan, dtype=float)
    return pd.to_numeric(d[col], errors="coerce").to_numpy(float)


def sigma_avr(tau_gyr: np.ndarray, comp: dict) -> np.ndarray:
    tau = np.asarray(tau_gyr, float)
    return float(comp["v10"]) * ((tau + float(comp["tau1"])) / (10.0 + float(comp["tau1"]))) \
        ** float(comp["beta"])


def galactic_xyz(ra_deg, dec_deg, dist_pc) -> np.ndarray:
    """Heliocentric Galactic Cartesian ``(X, Y, Z)`` in pc, shape ``(3, n)``."""
    r_hat, _, _ = _triad(np.asarray(ra_deg, float), np.asarray(dec_deg, float))
    return (_A_ICRS_TO_GAL @ r_hat) * np.asarray(dist_pc, float)


def teff_to_bv(teff: np.ndarray) -> np.ndarray:
    """Invert Ballesteros (2012) ``Teff(B-V)`` on a grid (dwarfs, 0.2 < B-V < 1.8)."""
    bv = np.linspace(-0.2, 2.0, 2201)
    t = 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))
    order = np.argsort(t)
    tt = np.asarray(teff, float)
    out = np.interp(tt, t[order], bv[order], left=np.nan, right=np.nan)
    return np.where(np.isfinite(tt), out, np.nan)


# ---------------------------------------------------------------------------
# Kinematic likelihood ratio
# ---------------------------------------------------------------------------
def kinematic_log_lr(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """``log10 LR(old : young)`` from the star's measured velocity components.

    Adds ``kin_log10_lr``, ``kin_p_old`` (flat 0-13 Gyr prior),
    ``kin_n_components`` (3 with RV, 2 without), ``kin_tau_median_gyr`` (the
    posterior median age under the flat prior; descriptive only).
    """
    c = {**DEFAULT_AGES, **(cfg or {})}
    n = len(df)
    out = df.copy()
    lo, hi, m = c["age_grid_gyr"]
    tau = np.geomspace(float(lo), float(hi), int(m))
    thr = float(c["age_threshold_gyr"])
    sig = {k: sigma_avr(tau, c["avr"][k]) for k in ("U", "V", "W")}
    va = sig["U"] ** 2 / float(c["asymmetric_drift_k_kms"])
    vsun = np.asarray(c["solar_motion_kms"], float)
    mu_t = np.stack([np.zeros_like(tau), -va, np.zeros_like(tau)], axis=1) - vsun[None, :]  # helio
    ra, dec = _num(df, "ra"), _num(df, "dec")
    plx = _num(df, "parallax")
    poe = _num(df, "parallax_over_error")
    pmra, pmdec = _num(df, "pmra"), _num(df, "pmdec")
    pmra_e = np.nan_to_num(_num(df, "pmra_error"), nan=0.05)
    pmdec_e = np.nan_to_num(_num(df, "pmdec_error"), nan=0.05)
    rv = _num(df, "radial_velocity")
    rv_e = np.nan_to_num(_num(df, "radial_velocity_error"), nan=2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.where(plx > 0, 1000.0 / plx, np.nan)
        frac_d = np.where(np.isfinite(poe) & (poe > 0), 1.0 / poe, 0.1)
    r_hat, a_hat, d_hat = _triad(ra, dec)
    e_r = _A_ICRS_TO_GAL @ r_hat            # (3, n) unit vectors in the Galactic frame
    e_a = _A_ICRS_TO_GAL @ a_hat
    e_d = _A_ICRS_TO_GAL @ d_hat
    v_a = _K_AUYR_KMS * pmra * dist / 1000.0
    v_d = _K_AUYR_KMS * pmdec * dist / 1000.0
    s_a = np.hypot(_K_AUYR_KMS * pmra_e * dist / 1000.0, v_a * frac_d)
    s_d = np.hypot(_K_AUYR_KMS * pmdec_e * dist / 1000.0, v_d * frac_d)
    has_rv = np.isfinite(rv)

    log_lr = np.full(n, np.nan)
    p_old = np.full(n, np.nan)
    ncomp = np.zeros(n, int)
    tmed = np.full(n, np.nan)
    old = tau >= thr
    for i in range(n):
        if not (np.isfinite(v_a[i]) and np.isfinite(v_d[i]) and np.isfinite(dist[i])):
            continue
        rows = [e_a[:, i], e_d[:, i]]
        obs = [v_a[i], v_d[i]]
        err = [s_a[i], s_d[i]]
        if has_rv[i]:
            rows.append(e_r[:, i])
            obs.append(rv[i])
            err.append(rv_e[i])
        P = np.asarray(rows)                       # (k, 3)
        w = np.asarray(obs)
        cm = np.diag(np.asarray(err) ** 2)
        ncomp[i] = len(obs)
        ll = np.empty(len(tau))
        for j in range(len(tau)):
            S = P @ np.diag([sig["U"][j] ** 2, sig["V"][j] ** 2, sig["W"][j] ** 2]) @ P.T + cm
            mu = P @ mu_t[j]
            r = w - mu
            try:
                sign, logdet = np.linalg.slogdet(S)
                sol = np.linalg.solve(S, r)
            except np.linalg.LinAlgError:
                ll[j] = -np.inf
                continue
            ll[j] = -0.5 * (r @ sol) - 0.5 * logdet
        ll -= np.nanmax(ll)
        like = np.exp(ll)
        # flat SFH over each hypothesis' own age range (trapezoid in tau)
        l_old = _trapz(like[old], tau[old]) / (tau[old][-1] - tau[old][0])
        l_young = _trapz(like[~old], tau[~old]) / (tau[~old][-1] - tau[~old][0])
        if l_old > 0 and l_young > 0:
            log_lr[i] = np.log10(l_old / l_young)
            # flat prior over 0.02-13 Gyr: prior odds = (13-1)/(1-0.02)
            prior_odds = (float(hi) - thr) / (thr - float(lo))
            lr = l_old / l_young
            p_old[i] = prior_odds * lr / (prior_odds * lr + 1.0)
        cum = np.cumsum(like * np.gradient(tau))
        if cum[-1] > 0:
            tmed[i] = float(np.interp(0.5 * cum[-1], cum, tau))
    out["kin_log10_lr"] = log_lr
    out["kin_p_old"] = p_old
    out["kin_n_components"] = ncomp
    out["kin_tau_median_gyr"] = tmed
    return out


# ---------------------------------------------------------------------------
# Young-group and region vetoes
# ---------------------------------------------------------------------------
def young_group_membership(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Name of the first young group the star is consistent with, else ``""``.

    Position: inside ``radius_pc`` of the group's ``xyz_pc`` (scaled by
    ``young_group_pos_scale``).  Velocity: with a radial velocity, the 3-D
    distance from ``uvw_kms`` within ``young_group_vel_tol_kms``; without one,
    the group's UVW is projected onto the star's tangent plane and compared
    with the measured tangential velocity (a conservative veto: consistency in
    two components is enough to veto, never enough to confirm).
    """
    c = {**DEFAULT_AGES, **(cfg or {})}
    out = df.copy()
    groups = list(c.get("young_groups") or [])
    n = len(df)
    name = np.full(n, "", dtype=object)
    dv = np.full(n, np.nan)
    if not groups or n == 0:
        out["young_group"] = name
        out["young_group_dv_kms"] = dv
        return out
    sv = space_velocity(df)
    ra, dec = _num(df, "ra"), _num(df, "dec")
    dist = sv["dist_pc"].to_numpy(float)
    xyz = galactic_xyz(ra, dec, dist)                      # (3, n)
    r_hat, a_hat, d_hat = _triad(ra, dec)
    e_a = _A_ICRS_TO_GAL @ a_hat
    e_d = _A_ICRS_TO_GAL @ d_hat
    v_hel = np.stack([sv["U_kms"].to_numpy(float), sv["V_kms"].to_numpy(float),
                      sv["W_kms"].to_numpy(float)])       # NaN without RV
    v_tan = np.stack([sv["Ut_kms"].to_numpy(float), sv["Vt_kms"].to_numpy(float),
                      sv["Wt_kms"].to_numpy(float)])
    has_rv = sv["has_rv"].to_numpy(bool)
    tol = float(c["young_group_vel_tol_kms"])
    scale = float(c.get("young_group_pos_scale", 1.0))
    for g in groups:
        gx = np.asarray(g["xyz_pc"], float)[:, None]
        gv = np.asarray(g["uvw_kms"], float)
        rad = float(g.get("radius_pc", 30.0)) * scale
        gtol = float(g.get("vel_tol_kms", tol))
        near = np.sqrt(((xyz - gx) ** 2).sum(axis=0)) <= rad
        if not near.any():
            continue
        # 3-D where RV exists
        d3 = np.sqrt(((v_hel - gv[:, None]) ** 2).sum(axis=0))
        # 2-D: tangential components of both
        ga = (gv[:, None] * e_a).sum(axis=0)
        gd = (gv[:, None] * e_d).sum(axis=0)
        sa = (v_tan * e_a).sum(axis=0)
        sd = (v_tan * e_d).sum(axis=0)
        d2 = np.hypot(sa - ga, sd - gd)
        dist_v = np.where(has_rv, d3, d2)
        hit = near & np.isfinite(dist_v) & (dist_v <= gtol) & (name == "")
        name[hit] = str(g["name"])
        dv[hit] = dist_v[hit]
    out["young_group"] = name
    out["young_group_dv_kms"] = dv
    return out


def young_region_membership(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Sky-box + parallax-range veto (Sco-Cen, rho Oph, Taurus, ...)."""
    c = {**DEFAULT_AGES, **(cfg or {})}
    n = len(df)
    name = np.full(n, "", dtype=object)
    ra, dec, plx = _num(df, "ra"), _num(df, "dec"), _num(df, "parallax")
    ll, bb = _num(df, "l"), _num(df, "b")
    for box in list(c.get("young_regions") or []):
        if "l" in box:
            lo, hi = box["l"]
            if lo <= hi:
                in_l = (ll >= lo) & (ll <= hi)
            else:                       # wraps through 360
                in_l = (ll >= lo) | (ll <= hi)
            blo, bhi = box["b"]
            m = in_l & (bb >= blo) & (bb <= bhi)
        else:
            rlo, rhi = box["ra"]
            in_r = (ra >= rlo) & (ra <= rhi) if rlo <= rhi else ((ra >= rlo) | (ra <= rhi))
            dlo, dhi = box["dec"]
            m = in_r & (dec >= dlo) & (dec <= dhi)
        if "parallax_mas" in box:
            plo, phi = box["parallax_mas"]
            m &= (plx >= plo) & (plx <= phi)
        m &= name == ""
        name[m] = str(box["name"])
    return pd.Series(name, index=df.index)


# ---------------------------------------------------------------------------
# The indicators
# ---------------------------------------------------------------------------
def gyro_age_gyr(prot_d: np.ndarray, bv: np.ndarray, g: dict) -> np.ndarray:
    """Mamajek & Hillenbrand 2008: ``P = a [(B-V) - c]^b t^n`` (t in Myr)."""
    p = np.asarray(prot_d, float)
    x = np.asarray(bv, float) - float(g["c"])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_myr = (p / (float(g["a"]) * np.where(x > 0, x, np.nan) ** float(g["b"]))) \
            ** (1.0 / float(g["n"]))
    t_myr = np.where((np.asarray(bv, float) >= float(g["bv_min"])) & (p > 0), t_myr, np.nan)
    return t_myr / 1000.0


def assess_ages(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Every indicator, the vetoes, the count, and the class.

    Adds ``age_iso_verdict``, ``age_kin_verdict``, ``age_alpha_verdict``,
    ``age_gyro_verdict``, ``age_gyro_gyr``, ``age_act_verdict``,
    ``young_group``, ``young_region``, ``n_old_indicators``,
    ``n_young_indicators``, ``old_indicators``, ``age_class``,
    ``age_adopted_gyr`` and the kinematic columns of :func:`kinematic_log_lr`.
    """
    c = {**DEFAULT_AGES, **(cfg or {})}
    thr = float(c["age_threshold_gyr"])
    out = kinematic_log_lr(df, c)
    out = young_group_membership(out, c)
    out["young_region"] = young_region_membership(out, c)
    n = len(out)

    # iso (FLAME)
    a_lo, a_hi, a = _num(out, "age_flame_lower"), _num(out, "age_flame_upper"), _num(out, "age_flame")
    iso = np.full(n, UNDET, dtype=object)
    ok = np.isfinite(a) & np.isfinite(a_lo) & np.isfinite(a_hi)
    iso[ok & (a_lo > thr)] = OLD
    iso[ok & (a_hi < thr)] = YOUNG
    # kin
    lr = out["kin_log10_lr"].to_numpy(float)
    kin = np.full(n, UNDET, dtype=object)
    enough = out["kin_n_components"].to_numpy(int) >= int(c["kin_min_components"])
    kin[enough & (lr >= float(c["kin_log10_lr_old"]))] = OLD
    kin[enough & (lr <= float(c["kin_log10_lr_young"]))] = YOUNG
    # alpha (GSP-Spec)
    al_lo = _num(out, "alphafe_gspspec_lower")
    al = _num(out, "alphafe_gspspec")
    mh = _num(out, "mh_gspspec")
    alpha = np.full(n, UNDET, dtype=object)
    qual = gspspec_quality_ok(out["flags_gspspec"].astype(str)).to_numpy(bool) \
        if "flags_gspspec" in out else np.ones(n, bool)
    alpha[qual & np.isfinite(al) & np.isfinite(al_lo)
          & (al_lo > float(c["alpha_fe_thick_min"])) & (mh < float(c["alpha_mh_max"]))] = OLD
    # gyro
    prot = _num(out, "prot_d")
    prot = np.where(np.isfinite(prot), prot, _num(out, "gaia_rot_period_d"))
    teff = _num(out, "teff") if "teff" in out else _num(out, "teff_gspphot")
    bv = teff_to_bv(teff)
    g = {**DEFAULT_AGES["gyro"], **(c.get("gyro") or {})}
    t_gyro = gyro_age_gyr(prot, bv, g)
    gyro = np.full(n, UNDET, dtype=object)
    gyro[np.isfinite(t_gyro) & (t_gyro < float(g["young_max_gyr"]))] = YOUNG
    gyro[np.isfinite(t_gyro) & (t_gyro > float(g["old_min_gyr"]))] = OLD
    # activity (veto only)
    act_idx = _num(out, "activityindex_espcs")
    act = np.full(n, UNDET, dtype=object)
    act[np.isfinite(act_idx) & (act_idx > float(c["activity_young_min_nm"]))] = YOUNG

    old_names = []
    n_old = np.zeros(n, int)
    n_young = np.zeros(n, int)
    for _name, arr in (("iso", iso), ("kin", kin), ("alpha", alpha), ("gyro", gyro), ("act", act)):
        n_old += (arr == OLD)
        n_young += (arr == YOUNG)
    for i in range(n):
        old_names.append("+".join(nm for nm, arr in (("iso", iso), ("kin", kin), ("alpha", alpha),
                                                      ("gyro", gyro)) if arr[i] == OLD))
    grp = out["young_group"].astype(str).to_numpy()
    reg = out["young_region"].astype(str).to_numpy()
    veto = (n_young > 0) | (grp != "") | (reg != "")
    cls = np.where(veto, CLASS_YOUNG, np.where(n_old >= 2, CLASS_MATURE, CLASS_UNDET)).astype(object)
    adopted = np.where(np.isfinite(a) & (iso == OLD), a,
                       np.where(kin == OLD, out["kin_tau_median_gyr"].to_numpy(float), np.nan))

    out["age_iso_verdict"] = iso
    out["age_kin_verdict"] = kin
    out["age_alpha_verdict"] = alpha
    out["age_gyro_verdict"] = gyro
    out["age_gyro_gyr"] = t_gyro
    out["age_act_verdict"] = act
    out["prot_used_d"] = prot
    out["n_old_indicators"] = n_old
    out["n_young_indicators"] = n_young
    out["old_indicators"] = old_names
    out["age_class"] = cls
    out["age_adopted_gyr"] = adopted
    return out


__all__ = ["CLASS_MATURE", "CLASS_UNDET", "CLASS_YOUNG", "DEFAULT_AGES", "OLD", "UNDET", "YOUNG",
           "assess_ages", "galactic_xyz", "gyro_age_gyr", "kinematic_log_lr", "sigma_avr",
           "teff_to_bv", "young_group_membership", "young_region_membership"]
