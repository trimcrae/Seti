"""DR4-shaped synthetic data for the offline suite (no network, no archive).

The generator writes tables in the SAME column spellings and cell shapes as
the real products it stands in for, so the readers are exercised exactly as
they will be on release day:

* epoch astrometry: one row per FoV transit, per-CCD array cells of length 10
  (SM + AF1..AF9) with the prerelease's 37 names (``centroid_pos_al`` etc.),
  ``obs_time_tcb`` int64 ns at Gaia plus ``obs_time_bary_corr``;
* epoch photometry: the DR3 long (DataLink) or wide (CDN) shape.

Scenes (each a known answer):

``single``          constant star; D = 0
``grey_dip``        a grey dip ON the target; D = 0 -> ON_TARGET
``eb_on_target``    a periodic eclipsing binary that IS the target; D = 0
``blended_eb``      constant target + eclipsing neighbour at rho, PA -> BLEND_NEIGHBOUR
``hidden_blend``    as blended_eb but the neighbour is not supplied -> BLEND_UNRESOLVED
``window_neighbour`` constant target + constant neighbour at ~0.6" that enters the
                    AF window on some scan angles -> SCAN_ANGLE_FLUX
``cti``             intrinsic variable + a detector-frame flux-centroid coupling
                    (does not flip with scan direction) -> DETECTOR_FRAME

Scanning law.  Not Gaia's: transits come in FoV pairs 106.5 min apart,
visits every 6 h for 1-3 spins, visits every ~30 d, scan angle drifting with
a 63-d precession term plus a random phase per visit.  What matters for the
tests is that theta covers the circle and that the parallax factor is
computed from a real Earth orbit, which it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .epochs import J2017P5_DAYS, NS_PER_DAY
from .photocentre import window_membership

T0_DAYS = 1666.0          # 2014-07-25 in days from JD 2455197.5
T1_DAYS = 3669.0          # 2020-01-20
OBLIQ = np.deg2rad(23.4393)


def earth_xyz(t_days: np.ndarray) -> np.ndarray:
    """Barycentric Earth (~Gaia at L2) position in AU, equatorial; circular orbit."""
    jd = t_days + 2455197.5
    L = np.deg2rad((280.46 + 0.9856474 * (jd - 2451545.0)) % 360.0)
    # the Sun's geometric longitude L -> the Earth is at L + 180 deg
    x_ecl = -np.cos(L)
    y_ecl = -np.sin(L)
    x = x_ecl
    y = y_ecl * np.cos(OBLIQ)
    z = y_ecl * np.sin(OBLIQ)
    return np.stack([x, y, z], axis=-1)


def parallax_factors(ra_deg: float, dec_deg: float, t_days: np.ndarray):
    a, d = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    X, Y, Z = earth_xyz(t_days).T
    fa = X * np.sin(a) - Y * np.cos(a)
    fd = X * np.cos(a) * np.sin(d) + Y * np.sin(a) * np.sin(d) - Z * np.cos(d)
    return fa, fd


def scan_times(rng: np.random.Generator, n_visits: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """(t_days, theta_rad) for every FoV transit."""
    vt = np.sort(rng.uniform(T0_DAYS, T1_DAYS, n_visits))
    ts, th = [], []
    for v in vt:
        base = rng.uniform(0, 2 * np.pi) + 2 * np.pi * (v / 63.0)
        for spin in range(int(rng.integers(1, 4))):
            t = v + spin * 0.25
            for fov in (0.0, 106.5 / 1440.0):
                ts.append(t + fov)
                th.append(base + 0.02 * spin)
    th = (np.array(th) + np.pi) % (2 * np.pi) - np.pi
    return np.array(ts), th


def sigma_al_ccd(g: float) -> float:
    """Per-CCD AL centroid error (mas) vs G, a smooth fit to the prerelease
    (0.10-0.15 mas bright) and Gaia's faint-end scaling."""
    return float(0.10 + 0.9 * (10 ** (0.4 * (g - 17.0))) ** 0.9)


def eclipse_profile(t: np.ndarray, period: float, t0: float, depth: float, width: float,
                    depth2: float = 0.0) -> np.ndarray:
    """Fractional flux loss of a box-ish (cosine-edged) EB."""
    ph = ((t - t0) / period) % 1.0
    loss = np.zeros_like(t)
    for centre, dep in ((0.0, depth), (0.5, depth2)):
        dph = np.abs(((ph - centre + 0.5) % 1.0) - 0.5)
        inside = dph < width / 2
        loss[inside] = dep * np.cos(np.pi * dph[inside] / width) ** 0.5
    return loss


def simulate_source(scene: str = "single", *, seed: int = 0, g_mag: float = 14.0,
                    ra: float = 120.0, dec: float = -30.0, parallax: float = 2.0,
                    pmra: float = -5.0, pmdec: float = 3.0, rho_mas: float = 150.0,
                    pa_deg: float = 60.0, flux_ratio: float = 0.3, depth: float = 0.3,
                    period: float = 2.7, b_detector: float = 3.0, n_visits: int = 45,
                    source_id: int = 4_000_000_000_000_000_001) -> dict:
    """One synthetic source: per-transit truth, the per-CCD astrometry table
    (prerelease spellings) and the photometry table (DR3 long spellings)."""
    rng = np.random.default_rng(seed)
    t, th = scan_times(rng, n_visits)
    n = len(t)
    fa, fd = parallax_factors(ra, dec, t)
    s, c = np.sin(th), np.cos(th)
    pf_al = fa * s + fd * c
    t_yr = (t - J2017P5_DAYS) / 365.25
    ft0 = 10 ** (-0.4 * (g_mag - 25.6874))
    f_t = np.full(n, ft0)
    f_n = np.zeros(n)
    off_al = np.zeros(n)          # photocentre shift, mas, along scan
    u = np.array([np.sin(np.deg2rad(pa_deg)), np.cos(np.deg2rad(pa_deg))])
    nb_dra, nb_ddec = rho_mas * u[0], rho_mas * u[1]
    rho_al = nb_dra * s + nb_ddec * c
    if scene in ("single",):
        pass
    elif scene == "grey_dip":
        k = int(n * 0.4)
        f_t[k:k + 3] *= (1 - depth)
    elif scene == "eb_on_target":
        f_t *= 1 - eclipse_profile(t, period, T0_DAYS + 0.3, depth, 0.08, 0.6 * depth)
    elif scene in ("blended_eb", "hidden_blend"):
        fn0 = flux_ratio * ft0
        f_n = fn0 * (1 - eclipse_profile(t, period, T0_DAYS + 0.3, 0.8, 0.08, 0.5))
    elif scene == "window_neighbour":
        fn0 = flux_ratio * ft0
        inwin = window_membership(th, nb_dra, nb_ddec)
        f_n = np.where(inwin, fn0, 0.0)
    elif scene == "cti":
        f_t *= 1 + 0.15 * np.sin(2 * np.pi * t / 37.0)
    else:
        raise ValueError(scene)
    f_tot = f_t + f_n
    off_al = rho_al * f_n / f_tot
    phi_true = f_tot / np.median(f_tot) - 1.0
    if scene == "cti":
        off_al = off_al + b_detector * phi_true
    x_true = (0.0 * s + 0.0 * c + parallax * pf_al + pmra * t_yr * s + pmdec * t_yr * c + off_al)
    # the mean blend offset is absorbed by the 5 parameters, as in AGIS
    sccd = sigma_al_ccd(g_mag)
    n_ccd = 10
    rows = []
    phot = []
    g_err = ft0 * (0.0015 + 0.001 * 10 ** (0.4 * (g_mag - 15)))
    for i in range(n):
        cen = x_true[i] + rng.normal(0, sccd, n_ccd)
        used = np.ones(n_ccd, bool)
        used[0] = False                       # the SM CCD is never used by AGIS
        cen[0] += 7.0
        tns = np.int64(round((t[i] - 0.0035) * NS_PER_DAY)) + np.arange(n_ccd, dtype=np.int64) * 4_900_000_000
        rows.append({
            "solution_id": 1, "source_id": source_id, "transit_id": 10_000_000_000_000_000 + i * 1000,
            "ra0": ra, "dec0": dec, "agis_source_excess_noise": 0.0,
            "obs_time_tcb": tns, "obs_time_bary_corr": np.float32(0.0035 * NS_PER_DAY),
            "scan_pos_angle": np.full(n_ccd, np.degrees(th[i])), "zeta": 0.0,
            "parallax_factor_al": pf_al[i], "parallax_factor_ac": 0.0,
            "colour_factor_al": np.zeros(n_ccd), "colour_factor_ac": np.zeros(n_ccd),
            "nu_eff_used_in_astrometry": 1.5, "nu_eff_error": 0.0,
            "centroid_pos_al": cen, "centroid_pos_ac": np.zeros(0),
            "calculated_pos_ac": np.zeros(n_ccd), "centroid_pos_error_al": np.full(n_ccd, sccd),
            "centroid_pos_error_ac": np.zeros(0), "used_by_agis_al": used,
            "used_by_agis_ac": np.zeros(n_ccd, bool), "transit_acq_flags": 0,
            "transit_proc_flags": -32208, "ccd_proc_flags": np.full(n_ccd, 528),
            "multipeak": False, "blended": False, "ipd_error_al": np.full(n_ccd, 0.05),
            "ipd_error_ac": np.full(n_ccd, 0.1), "g_mag": g_mag, "g_class": 0,
            "gates": np.zeros(n_ccd, int), "source_dist_to_last_ci": np.full(n_ccd, 500.0),
            "ac_rate": 0.0, "sub_pixel_coord": np.full(n_ccd, 0.5), "mu": np.arange(n_ccd) + 500.0,
        })
        fg = f_tot[i] + rng.normal(0, g_err)
        # BP/RP: the same fractional change (grey scenes); +-1% colour noise
        fbp = 0.6 * f_tot[i] * (1 + rng.normal(0, 0.004))
        frp = 0.9 * f_tot[i] * (1 + rng.normal(0, 0.004))
        phot.append({"transit_id": 10_000_000_000_000_000 + i * 1000, "g_transit_time": t[i],
                     "g_transit_flux": fg, "g_transit_flux_error": g_err,
                     "bp_obs_time": t[i] + 1e-5, "bp_flux": fbp, "bp_flux_error": 0.004 * 0.6 * ft0,
                     "rp_obs_time": t[i] + 1e-5, "rp_flux": frp, "rp_flux_error": 0.004 * 0.9 * ft0,
                     "variability_flag_g_reject": False, "variability_flag_bp_reject": False,
                     "variability_flag_rp_reject": False, "rejected_by_photometry": False,
                     "g_other_flags": 0, "bp_other_flags": 0, "rp_other_flags": 0})
    nb = pd.DataFrame()
    if scene in ("blended_eb", "window_neighbour"):
        nb = pd.DataFrame({"source_id": [source_id + 1], "dra_mas": [nb_dra], "ddec_mas": [nb_ddec],
                           "rho_mas": [rho_mas], "flux_g": [flux_ratio * ft0]})
    return {"astro": pd.DataFrame(rows), "phot": pd.DataFrame(phot), "neighbours": nb,
            "target_flux_g": ft0, "truth": {"scene": scene, "t": t, "theta": th, "phi": phi_true,
                                            "D_true_mas": (rho_mas * (1 - flux_ratio / (1 + flux_ratio))
                                                           if scene in ("blended_eb", "hidden_blend")
                                                           else 0.0)}}


def simulate_grey_lightcurve(kind: str = "quiet", *, seed: int = 0, n_visits: int = 30,
                             depth: float = 0.2, ratio: float = 1.0, g_mag: float = 15.0,
                             source_id: int = 1) -> pd.DataFrame:
    """A DR3-long-shaped light curve for the photometric detector tests.

    kind: quiet | grey_multi | grey_single | dust_multi | flare | eb | g_only |
          partner_contradicted | variable
    """
    rng = np.random.default_rng(seed)
    t, _ = scan_times(rng, n_visits)
    n = len(t)
    f = np.ones(n)
    fb = np.ones(n)
    fr = np.ones(n)
    # a transit whose FoV partner follows 106.5 min later
    pair = [i for i in range(n - 1) if abs((t[i + 1] - t[i]) - 106.5 / 1440) < 1e-3]
    if kind in ("grey_multi", "dust_multi"):
        i0 = pair[len(pair) // 2]
        sel = [i0, i0 + 1]
        dbp = depth * (ratio if kind == "grey_multi" else 1.7)
        fr[sel] *= 1 - depth
        fb[sel] *= 1 - dbp
        f[sel] *= 1 - 0.5 * (depth + dbp)
    elif kind in ("grey_single", "partner_contradicted"):
        if kind == "partner_contradicted":
            sel = [pair[len(pair) // 2]]
        else:
            # a transit with no other transit within 0.35 d
            iso = [i for i in range(n) if np.sum(np.abs(t - t[i]) < 0.35) == 1]
            if not iso:
                ts = np.sort(t)
                g = int(np.argmax(np.diff(ts)))
                t = np.append(t, 0.5 * (ts[g] + ts[g + 1]))     # the middle of the widest gap
                f, fb, fr = np.append(f, 1), np.append(fb, 1), np.append(fr, 1)
                n = len(t)
                iso = [n - 1]
            sel = [iso[0]]
        for a in (f, fb, fr):
            a[sel] *= 1 - depth
    elif kind == "flare":
        i0 = pair[len(pair) // 2]
        f[[i0, i0 + 1]] *= 1 + depth
        fb[[i0, i0 + 1]] *= 1 + 3 * depth
        fr[[i0, i0 + 1]] *= 1 + 0.4 * depth
    elif kind == "eb":
        loss = eclipse_profile(t, 1.7371, T0_DAYS + 0.2, depth, 0.12, 0.9 * depth)
        f *= 1 - loss
        fb *= 1 - loss
        fr *= 1 - loss
    elif kind == "g_only":
        i0 = pair[len(pair) // 2]
        f[[i0, i0 + 1]] *= 1 - depth
    elif kind == "variable":
        amp = 0.3
        f *= 1 + amp * np.sin(2 * np.pi * t / 0.61)
        fb *= 1 + 1.4 * amp * np.sin(2 * np.pi * t / 0.61)
        fr *= 1 + 0.8 * amp * np.sin(2 * np.pi * t / 0.61)
    elif kind != "quiet":
        raise ValueError(kind)
    f0 = 10 ** (-0.4 * (g_mag - 25.6874))
    eg = f0 * 0.002
    eb = 0.6 * f0 * 0.006
    er = 0.9 * f0 * 0.005
    order = np.argsort(t)
    df = pd.DataFrame({
        "source_id": source_id, "transit_id": np.arange(n) + 1_000_000,
        "g_transit_time": t, "g_transit_flux": f0 * f + rng.normal(0, eg, n),
        "g_transit_flux_error": np.full(n, eg),
        "bp_obs_time": t, "bp_flux": 0.6 * f0 * fb + rng.normal(0, eb, n),
        "bp_flux_error": np.full(n, eb),
        "rp_obs_time": t, "rp_flux": 0.9 * f0 * fr + rng.normal(0, er, n),
        "rp_flux_error": np.full(n, er),
        "variability_flag_g_reject": False, "variability_flag_bp_reject": False,
        "variability_flag_rp_reject": False, "rejected_by_photometry": False,
    }).iloc[order].reset_index(drop=True)
    return df


def to_wide_csv_text(long_df: pd.DataFrame, style: str = "paren") -> str:
    """Serialise a long light-curve frame as the CDN bulk wide CSV (with an
    ECSV-like '#' header), to test the text-array parser end to end."""
    lb, rb = {"paren": ("(", ")"), "bracket": ("[", "]"), "brace": ("{", "}")}[style]
    cols = ["transit_id", "g_transit_time", "g_transit_flux", "g_transit_flux_error",
            "bp_obs_time", "bp_flux", "bp_flux_error", "rp_obs_time", "rp_flux", "rp_flux_error",
            "variability_flag_g_reject", "photometry_flag_noisy_data"]
    lines = ["# %ECSV 1.0", "# ---", "# datatype: [...]",
             "solution_id,source_id,n_transits," + ",".join(cols)]
    for sid, g in long_df.groupby("source_id", sort=False):
        cells = []
        for c in cols:
            if c == "photometry_flag_noisy_data":
                vals = ["false"] * len(g)
            elif c == "variability_flag_g_reject":
                vals = ["true" if v else "false" for v in g[c]]
            else:
                vals = ["null" if not np.isfinite(float(v)) else repr(float(v)) if "flux" in c or "time" in c
                        else str(int(v)) for v in g[c]]
            cells.append('"' + lb + ",".join(vals) + rb + '"')
        lines.append(f"375316653866487564,{sid},{len(g)}," + ",".join(cells))
    return "\n".join(lines) + "\n"
