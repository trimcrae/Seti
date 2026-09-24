"""The IGNITION contaminant ladder.  Pure, with an optical-series hook.

Every rule below is a named veto with its own counter (``summarise``), so the
funnel partitions the sample and the summary can say *which* contaminant took
what.  Order matters: the first rule that fires names the verdict.

| Rule | Kills | Why |
|---|---|---|
| ``extragalactic`` | obscured-AGN turn-on (NGC 6447) | no significant parallax / PM |
| ``star_forming_region`` | YSO accretion outbursts | position in a configured box |
| ``galactic_plane`` | YSOs, crowding | ``|b|`` below the floor |
| ``gaia_variable`` | Miras, novae, R CrB, YSOs | Gaia DR3 says VARIABLE |
| ``saturated`` | NEOWISE bright-source bias | W1 < 8 |
| ``deblended`` | latent / deblending artefacts | ``nb > 1`` or ``na > 0`` on most frames |
| ``poor_photometry`` | marginal detections | too many frames cut on ``ph_qual`` |
| ``already_excess_2010`` | anything with an old excess | AllWISE W1-W2 not photospheric |
| ``rcrb_like`` | R CrB dust puffs | optical *fades* while the IR rises |
| ``optical_not_flat`` | novae, AGB dust formation, YSOs | optical slope or scatter |

A star whose optical light curve could not be checked is **not** an optically
flat star; it is carried with ``optical: not_checked`` and its verdict says
``optical_untested`` rather than ``clean``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..vigil.vet import galactic_latitude

DEFAULT_VET: dict = {
    "parallax_over_error_min": 10.0,
    "pm_sig_min": 5.0,
    "galactic_latitude_min_deg": 15.0,
    "w1_saturation_mag": 8.0,
    "w2_saturation_mag": 7.0,
    "frac_nb_gt1_max": 0.2,
    "frac_na_gt0_max": 0.2,
    "ph_qual_cut_frac_max": 0.5,
    "agn_w1w2_min": 0.8,
    "w1w2_2010_max": 0.15,
    "optical_slope_sigma_max": 3.0,
    "optical_rms_max": 0.05,
    "optical_min_points": 20,
    # --- folded in from the re-vet of tiles run 35740159635 (docs/ignition.md 7.4)
    # Rise colour: W2rise/W1rise must not sit below warm dust at this
    # temperature by >= rise_colour_sigma (a stellar-coloured rise is the star
    # or a blend brightening, not a born excess).
    "rise_colour_dust_temp_k": 1500.0,
    "rise_colour_sigma": 3.0,
    # Approaching neighbour: a Gaia neighbour whose proper-motion drift over
    # 2014-2024 changes the flux a PSF fit at the target picks up by at least
    # this fraction of the observed W1 rise.
    "neighbour_radius_arcsec": 30.0,
    "neighbour_fwhm_arcsec": 6.1,
    "neighbour_rise_fraction": 0.3,
    # Optical: a band brightening/fading at >= this significance AND at >= this
    # fraction of the W1 rise rate is not a flat optical.
    "optical_rise_sigma": 5.0,
    "optical_rise_fraction": 0.3,
    "optical_saturation_mag": {"zg": 12.5, "zr": 12.5, "zi": 12.0, "V": 10.5, "g": 10.5},
    # Gaia DR3 per-observation scatter above this percentile of G/BP-RP peers in
    # G, BP and RP alike is an optical variable.
    "gaia_scatter_percentile": 99.0,
    # Known star-forming regions and young associations, generous boxes
    # (ra_min, ra_max, dec_min, dec_max) in degrees.
    "star_forming_boxes": [
        {"name": "Taurus", "ra": [60.0, 75.0], "dec": [15.0, 32.0]},
        {"name": "Orion", "ra": [78.0, 92.0], "dec": [-12.0, 12.0]},
        {"name": "Perseus", "ra": [50.0, 58.0], "dec": [29.0, 34.0]},
        {"name": "Ophiuchus", "ra": [243.0, 250.0], "dec": [-27.0, -21.0]},
        {"name": "UpperSco", "ra": [235.0, 250.0], "dec": [-30.0, -15.0]},
        {"name": "Lupus", "ra": [232.0, 246.0], "dec": [-43.0, -33.0]},
        {"name": "Chamaeleon", "ra": [160.0, 172.0], "dec": [-80.0, -74.0]},
        {"name": "Serpens", "ra": [275.0, 280.0], "dec": [-3.0, 3.0]},
        {"name": "CoronaAustralis", "ra": [283.0, 288.0], "dec": [-39.0, -35.0]},
        {"name": "Cepheus", "ra": [310.0, 340.0], "dec": [60.0, 72.0]},
    ],
}


@dataclass
class VetResult:
    verdict: str
    flags: list[str] = field(default_factory=list)
    optical: str = "not_checked"
    optical_slope_mag_yr: float = float("nan")
    optical_slope_sigma: float = float("nan")
    optical_rms_mag: float = float("nan")
    optical_n: int = 0
    region: str = ""
    untested_checks: list[str] = field(default_factory=list)
    rise_colour: str = "untested"
    rise_colour_ratio: float = float("nan")
    rise_colour_err: float = float("nan")
    rise_colour_z: float = float("nan")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["flags"] = ";".join(self.flags)
        d["untested_checks"] = ";".join(self.untested_checks)
        return d


def _f(row: dict, key: str, default=float("nan")) -> float:
    try:
        v = row.get(key, default)
        v = float(v)
        return v if np.isfinite(v) else float(default)
    except (TypeError, ValueError):
        return float(default)


def in_star_forming_region(ra: float, dec: float, boxes) -> str:
    """Name of the first configured box containing (ra, dec), else ''."""
    for b in boxes or []:
        r0, r1 = (float(x) for x in b["ra"])
        d0, d1 = (float(x) for x in b["dec"])
        if d0 <= dec <= d1 and r0 <= ra <= r1:
            return str(b.get("name", "box"))
    return ""


def optical_flatness(t_yr, mag, err=None, conf: dict | None = None) -> dict:
    """Weighted slope, its significance, and the rms of an optical series.

    Returns ``{"status": flat | brightening | fading | insufficient, ...}``.
    A significantly *positive* slope (fading in magnitude) while the IR rises
    is the R CrB sign; a significantly negative one is a nova / AGB / YSO
    brightening the IR is merely following.
    """
    c = {**DEFAULT_VET, **(conf or {})}
    t = np.asarray(t_yr, float)
    m = np.asarray(mag, float)
    e = (np.asarray(err, float) if err is not None else np.full(t.size, np.nan))
    ok = np.isfinite(t) & np.isfinite(m)
    t, m, e = t[ok], m[ok], e[ok]
    if t.size < int(c["optical_min_points"]):
        return {"status": "insufficient", "n": int(t.size), "slope": float("nan"),
                "slope_sigma": float("nan"), "rms": float("nan")}
    e = np.where(np.isfinite(e) & (e > 0), e, np.nanmedian(e[np.isfinite(e) & (e > 0)])
                 if np.any(np.isfinite(e) & (e > 0)) else 0.02)
    w = 1.0 / e**2
    tc = t - np.mean(t)
    X = np.column_stack([np.ones(t.size), tc])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], m * sw, rcond=None)
    resid = m - X @ beta
    chi2_red = float(np.sum(w * resid**2) / max(t.size - 2, 1))
    cov = np.linalg.inv((X * sw[:, None]).T @ (X * sw[:, None]))
    slope_err = float(np.sqrt(cov[1, 1]) * np.sqrt(max(1.0, chi2_red)))
    slope = float(beta[1])
    sig = slope / slope_err if slope_err > 0 else float("nan")
    rms = float(np.std(resid, ddof=2)) if t.size > 2 else float("nan")
    if np.isfinite(sig) and abs(sig) >= float(c["optical_slope_sigma_max"]):
        status = "fading" if slope > 0 else "brightening"
    elif rms > float(c["optical_rms_max"]):
        status = "variable"
    else:
        status = "flat"
    return {"status": status, "n": int(t.size), "slope": slope, "slope_sigma": float(sig),
            "rms": rms}


def load_optical_series(optical_dir: Path | str | None, source_id: str) -> pd.DataFrame | None:
    """``<optical_dir>/<source_id>.csv`` with columns ``mjd|t_yr, mag[, magerr]``."""
    if not optical_dir:
        return None
    p = Path(optical_dir) / f"{source_id}.csv"
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p)
    except Exception:                                  # noqa: BLE001
        return None
    d.columns = [str(x).lower() for x in d.columns]
    if "t_yr" not in d and "mjd" in d:
        d["t_yr"] = 2000.0 + (pd.to_numeric(d["mjd"], errors="coerce") - 51544.5) / 365.25
    if "t_yr" not in d or "mag" not in d:
        return None
    return d


def _bnu_ratio(temp_k: float, lam1_um: float = 3.35, lam2_um: float = 4.60) -> float:
    """B_nu(lam2) / B_nu(lam1) for a blackbody."""
    c2 = 14387.77
    x1, x2 = c2 / (lam1_um * temp_k), c2 / (lam2_um * temp_k)
    return float((lam1_um / lam2_um) ** 3 * np.expm1(x1) / np.expm1(x2))


def dust_colour_test(w1_slope, w1_sigma, w2_slope, w2_sigma, teff: float = 5000.0) -> dict:
    """Is the rise the colour of warm dust, or of a star?

    For small changes the magnitude rise in a band is the fractional flux
    excess, so W2rise/W1rise = [B(Td,4.6)/B(Td,3.35)] / [B(T*,4.6)/B(T*,3.35)].
    Dust at 1000 K on a 5000 K photosphere gives ~2.1, at 1500 K ~1.5
    (sublimation); a stellar-coloured contributor (the star itself, a blend, a
    multiplicative calibration term) gives ~1.  ``z_vs_dust_<T>K`` is how many
    sigma the measured ratio sits BELOW the dust prediction.
    """
    try:
        s1, s2 = float(w1_slope), float(w2_slope)
        e1, e2 = abs(s1 / float(w1_sigma)), abs(s2 / float(w2_sigma))
    except (TypeError, ValueError, ZeroDivisionError):
        return {}
    if not (np.isfinite(s1) and np.isfinite(s2) and np.isfinite(e1) and np.isfinite(e2)
            and s1 != 0 and s2 != 0):
        return {}
    ratio = s2 / s1
    err = abs(ratio) * float(np.hypot(e1 / s1, e2 / s2))
    teff = float(teff) if teff is not None and np.isfinite(teff) and teff > 2500 else 5000.0
    phot = _bnu_ratio(teff)
    out = {"w2_over_w1_rise": round(ratio, 3), "err": round(err, 3), "teff_used": teff}
    for td in (800.0, 1000.0, 1500.0):
        pred = _bnu_ratio(td) / phot
        out[f"pred_dust_{int(td)}K"] = round(pred, 3)
        out[f"z_vs_dust_{int(td)}K"] = round((pred - ratio) / err, 2) if err > 0 else None
    return out


def rise_colour_verdict(row: dict, conf: dict | None = None) -> dict:
    """``dust_colour_test`` on a screened row, with the kill decision."""
    c = {**DEFAULT_VET, **(conf or {})}
    dc = dust_colour_test(row.get("w1_slope_mag_yr"), row.get("w1_slope_sigma"),
                          row.get("w2_slope_mag_yr"), row.get("w2_slope_sigma"),
                          _f(row, "teff_gspphot", 5000.0))
    if not dc:
        return {"status": "untested"}
    td = float(c["rise_colour_dust_temp_k"])
    pred = _bnu_ratio(td) / _bnu_ratio(dc["teff_used"])
    z = (pred - dc["w2_over_w1_rise"]) / dc["err"] if dc["err"] > 0 else float("nan")
    dc["z_vs_dust_threshold"] = round(float(z), 2) if np.isfinite(z) else None
    dc["dust_temp_threshold_k"] = td
    dc["status"] = ("stellar_coloured" if np.isfinite(z) and z >= float(c["rise_colour_sigma"])
                    else "dust_compatible")
    return dc


def psf_weight(sep_arcsec, fwhm_arcsec: float = 6.1):
    """Fraction of an offset point source's flux a PSF fit at the origin absorbs.

    For a Gaussian PSF of width sigma the PSF-weighted flux of a source at
    offset r is exp(-r^2 / (4 sigma^2)) of its flux.
    """
    sig = float(fwhm_arcsec) / 2.3548
    r = np.asarray(sep_arcsec, float)
    return np.exp(-(r ** 2) / (4.0 * sig ** 2))


def _propagate(ra, dec, pmra, pmdec, t: float, gaia_epoch: float = 2016.0):
    dt = t - gaia_epoch
    dec = np.asarray(dec, float)
    dec2 = dec + np.nan_to_num(np.asarray(pmdec, float)) * dt / 3.6e6
    ra2 = np.asarray(ra, float) + (np.nan_to_num(np.asarray(pmra, float)) * dt / 3.6e6
                                   / np.cos(np.radians(dec)))
    return ra2, dec2


def _sep_arcsec(ra1, de1, ra2, de2):
    ra1, de1, ra2, de2 = (np.radians(np.asarray(x, float)) for x in (ra1, de1, ra2, de2))
    h = (np.sin((de2 - de1) / 2) ** 2
         + np.cos(de1) * np.cos(de2) * np.sin((ra2 - ra1) / 2) ** 2)
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))) * 3600.0


def neighbour_contamination(target: dict, neighbours: pd.DataFrame | None, rise_mag: float,
                            conf: dict | None = None, t0: float = 2014.0,
                            t1: float = 2024.5) -> dict:
    """Predicted W1 change from Gaia neighbours drifting relative to the target.

    ``target`` / ``neighbours`` carry Gaia ``ra, dec, pmra, pmdec,
    phot_g_mean_mag`` (epoch 2016.0).  Each neighbour's flux ratio is
    10^(-0.4 dG) (the optical ratio; a redder neighbour is relatively brighter
    in W1, hence the loose ``neighbour_rise_fraction``; a neighbour with no G
    is taken as bright as the target).  The magnitude change of a PSF fit at
    the (co-moving) target position between ``t0`` and ``t1`` is compared with
    the observed rise.  Check: LP 387-28 (dG 1.95, 6.7" -> 4.4") predicts
    -0.049 mag against an observed W1 rise of 0.053.
    """
    c = {**DEFAULT_VET, **(conf or {})}
    if neighbours is None or not len(neighbours):
        return {"status": "clear", "n_neighbours": 0, "pred_dmag": 0.0}
    g_t = _f(target, "phot_g_mean_mag")
    nb = neighbours
    g_n = pd.to_numeric(nb["phot_g_mean_mag"], errors="coerce").to_numpy(float)
    ratio = np.where(np.isfinite(g_n) & np.isfinite(g_t), 10 ** (-0.4 * (g_n - g_t)), 1.0)
    pmra_n = nb["pmra"] if "pmra" in nb else 0.0
    pmdec_n = nb["pmdec"] if "pmdec" in nb else 0.0
    seps = {}
    for t in (2010.5, t0, t1):
        rt, dt_ = _propagate(_f(target, "ra"), _f(target, "dec"), _f(target, "pmra", 0.0),
                             _f(target, "pmdec", 0.0), t)
        rn, dn = _propagate(nb["ra"], nb["dec"], pmra_n, pmdec_n, t)
        seps[t] = _sep_arcsec(rt, dt_, rn, dn)
    fwhm = float(c["neighbour_fwhm_arcsec"])
    w0, w1 = psf_weight(seps[t0], fwhm), psf_weight(seps[t1], fwhm)
    f0, f1 = float(np.sum(ratio * w0)), float(np.sum(ratio * w1))
    pred = float(-2.5 * np.log10((1.0 + f1) / (1.0 + f0)))
    rise = abs(float(rise_mag)) if rise_mag is not None and np.isfinite(rise_mag) else np.nan
    thr = float(c["neighbour_rise_fraction"]) * rise if np.isfinite(rise) else float("inf")
    worst = int(np.argmax(np.abs(ratio * (w1 - w0))))
    rec = {"n_neighbours": int(len(nb)), "pred_dmag": round(pred, 4),
           "observed_rise_mag": None if not np.isfinite(rise) else round(rise, 4),
           "threshold_mag": round(thr, 4) if np.isfinite(thr) else None,
           "static_contamination_frac": round(f0, 4),
           "worst": {"dg": (round(float(g_n[worst] - g_t), 2)
                            if np.isfinite(g_n[worst]) and np.isfinite(g_t) else None),
                     "sep_2010": round(float(seps[2010.5][worst]), 2),
                     "sep_t0": round(float(seps[t0][worst]), 2),
                     "sep_t1": round(float(seps[t1][worst]), 2)}}
    # A neighbour drifting IN brightens the fit (pred < 0) and can fake the rise.
    rec["status"] = "approaching_neighbour" if (pred < 0 and abs(pred) >= thr) else "clear"
    return rec


def _norm_band(rec) -> dict | None:
    """ZTF (optical_flatness) or ASAS-SN (camera-offset fit) band -> slope, sigma, median."""
    if not isinstance(rec, dict) or rec.get("status") in (None, "insufficient"):
        return None
    try:
        if "slope_mmag_yr" in rec:
            slope = float(rec["slope_mmag_yr"]) / 1e3
            err = float(rec.get("slope_err_mmag_yr")) / 1e3
        elif "slope" in rec:
            slope = float(rec["slope"])
            sg = float(rec.get("slope_sigma"))
            err = abs(slope / sg) if sg != 0 else float("nan")
        else:
            return None
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(slope) and np.isfinite(err) and err > 0):
        return None
    return {"slope_mag_yr": slope, "sigma": slope / err, "median_mag": rec.get("median_mag"),
            "n": rec.get("n_used", rec.get("n"))}


def optical_rise_verdict(bands: dict, w1_slope_mag_yr: float, conf: dict | None = None) -> dict:
    """Is the optical flat?  From per-band trends of ZTF / ASAS-SN.

    ``bands`` maps ``survey:filter`` (``ztf:zr``, ``asassn:V``) to a band dict.
    A band counts only if not saturated.  ``brightening`` / ``fading`` if a
    usable band moves at >= ``optical_rise_sigma`` AND at >= ``optical_rise_
    fraction`` of the W1 rate (a camera zero-point drift of a mmag/yr is not a
    trend); ``inconsistent`` if moving bands disagree in sign; ``flat`` if at
    least one band is usable and none moves; ``untested`` if none is usable.
    """
    c = {**DEFAULT_VET, **(conf or {})}
    sat = c["optical_saturation_mag"]
    try:
        w1r = abs(float(w1_slope_mag_yr))
    except (TypeError, ValueError):
        w1r = 0.0
    w1r = w1r if np.isfinite(w1r) else 0.0
    used, moving = {}, []
    for label, rec in (bands or {}).items():
        b = _norm_band(rec)
        if b is None:
            continue
        filt = str(label).split(":")[-1]
        med = b.get("median_mag")
        if med is not None and filt in sat and float(med) < float(sat[filt]):
            continue
        used[label] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in b.items()}
        if abs(b["sigma"]) >= float(c["optical_rise_sigma"]) and \
                abs(b["slope_mag_yr"]) >= float(c["optical_rise_fraction"]) * w1r:
            moving.append((label, b))
    if not used:
        return {"status": "untested", "bands_used": {}}
    if moving:
        bright = [m for m in moving if m[1]["slope_mag_yr"] < 0]
        fade = [m for m in moving if m[1]["slope_mag_yr"] > 0]
        if bright and fade:
            return {"status": "inconsistent", "bands_used": used,
                    "moving": [m[0] for m in moving]}
        return {"status": "brightening" if bright else "fading", "bands_used": used,
                "moving": [m[0] for m in moving]}
    return {"status": "flat", "bands_used": used}


def gaia_scatter_verdict(rec, conf: dict | None = None) -> dict:
    """Gaia DR3 per-observation scatter percentiles (G, BP, RP) against peers."""
    c = {**DEFAULT_VET, **(conf or {})}
    if not isinstance(rec, dict) or rec.get("a_g_percentile") is None:
        return {"status": "untested"}
    thr = float(c["gaia_scatter_percentile"])
    pct = [rec.get(f"a_{b}_percentile") for b in ("g", "bp", "rp")]
    pct = [float(p) for p in pct if p is not None]
    status = "variable" if len(pct) == 3 and min(pct) >= thr else "quiet"
    return {"status": status, "percentiles": pct, "threshold": thr}


#: The order in which flags name the verdict: the pure ladder, then the rungs
#: that need an archive (``vet_online``).
KILL_ORDER = ("extragalactic", "already_excess_2010", "star_forming_region",
              "galactic_plane", "gaia_variable", "saturated", "deblended",
              "poor_photometry", "rise_stellar_coloured", "approaching_neighbour",
              "rcrb_like", "optical_not_flat", "optical_variable_gaia")


def vet_star(row: dict, conf: dict | None = None, optical: pd.DataFrame | None = None
             ) -> VetResult:
    """Run the ladder on one screened star.  ``row`` carries the merged columns."""
    c = {**DEFAULT_VET, **(conf or {})}
    flags: list[str] = []
    untested: list[str] = []
    res = VetResult(verdict="clean")

    ra, dec = _f(row, "ra"), _f(row, "dec")
    poe = _f(row, "parallax_over_error")
    pmra, pmdec = _f(row, "pmra", 0.0), _f(row, "pmdec", 0.0)
    pm_err = _f(row, "pm_error", float("nan"))
    pm = float(np.hypot(pmra, pmdec))
    pm_sig = pm / pm_err if np.isfinite(pm_err) and pm_err > 0 else float("nan")
    astrometric = (np.isfinite(poe) and poe >= float(c["parallax_over_error_min"]))
    if np.isfinite(pm_sig):
        astrometric = astrometric or pm_sig >= float(c["pm_sig_min"])
    if not astrometric:
        flags.append("extragalactic")

    if np.isfinite(ra) and np.isfinite(dec):
        reg = in_star_forming_region(ra, dec, c["star_forming_boxes"])
        res.region = reg
        if reg:
            flags.append("star_forming_region")
        b = _f(row, "b", galactic_latitude(ra, dec))
        if abs(b) < float(c["galactic_latitude_min_deg"]):
            flags.append("galactic_plane")
    else:
        untested.append("position")

    if str(row.get("phot_variable_flag", "")).upper() == "VARIABLE":
        flags.append("gaia_variable")

    w1 = _f(row, "w1_median", _f(row, "w1mpro"))
    w2 = _f(row, "w2_median", _f(row, "w2mpro"))
    if np.isfinite(w1) and w1 < float(c["w1_saturation_mag"]):
        flags.append("saturated")
    elif np.isfinite(w2) and w2 < float(c["w2_saturation_mag"]):
        flags.append("saturated")

    nb, na = _f(row, "frac_nb_gt1"), _f(row, "frac_na_gt0")
    if (np.isfinite(nb) and nb > float(c["frac_nb_gt1_max"])) or \
       (np.isfinite(na) and na > float(c["frac_na_gt0_max"])):
        flags.append("deblended")
    elif not np.isfinite(nb) and not np.isfinite(na):
        untested.append("deblending")

    n_raw = _f(row, "n_exp_raw", float("nan"))
    cut_pq = _f(row, "cut_ph_qual", float("nan"))
    if np.isfinite(n_raw) and n_raw > 0 and np.isfinite(cut_pq):
        if cut_pq / n_raw > float(c["ph_qual_cut_frac_max"]):
            flags.append("poor_photometry")
    else:
        untested.append("ph_qual")

    w1w2 = _f(row, "w1w2_2010", _f(row, "w1mpro") - _f(row, "w2mpro"))
    if np.isfinite(w1w2):
        if w1w2 > float(c["agn_w1w2_min"]):
            flags.append("nearest_agn_like")
        if w1w2 > float(c["w1w2_2010_max"]):
            flags.append("already_excess_2010")
    else:
        untested.append("allwise_2010_colour")

    if bool(row.get("grey_rise_suspect", False)):
        flags.append("grey_rise_suspect")

    rcv = rise_colour_verdict(row, c)
    res.rise_colour = rcv.get("status", "untested")
    res.rise_colour_ratio = float(rcv.get("w2_over_w1_rise", float("nan")))
    res.rise_colour_err = float(rcv.get("err", float("nan")))
    z = rcv.get("z_vs_dust_threshold")
    res.rise_colour_z = float(z) if z is not None else float("nan")
    if rcv.get("status") == "stellar_coloured":
        flags.append("rise_stellar_coloured")
    elif rcv.get("status") == "untested":
        untested.append("rise_colour")

    # --- optical hook ------------------------------------------------------
    if optical is not None and len(optical):
        opt = optical_flatness(optical["t_yr"], optical["mag"],
                               optical["magerr"] if "magerr" in optical else None, c)
        res.optical = opt["status"]
        res.optical_slope_mag_yr = opt["slope"]
        res.optical_slope_sigma = opt["slope_sigma"]
        res.optical_rms_mag = opt["rms"]
        res.optical_n = opt["n"]
        if opt["status"] == "fading":
            flags.append("rcrb_like")
        elif opt["status"] in ("brightening", "variable"):
            flags.append("optical_not_flat")
        elif opt["status"] == "insufficient":
            untested.append("optical_flatness")
    else:
        res.optical = "not_checked"
        untested.append("optical_flatness")

    res.flags = flags
    res.untested_checks = untested
    for k in KILL_ORDER:
        if k in flags:
            res.verdict = f"rejected_{k}"
            return res
    res.verdict = "clean" if "optical_flatness" not in untested else "clean_optical_untested"
    return res


def summarise(results: list[VetResult] | list[dict]) -> dict:
    """Counts per verdict and per flag."""
    verd: dict = {}
    flg: dict = {}
    for r in results:
        d = r.as_dict() if isinstance(r, VetResult) else dict(r)
        v = str(d.get("verdict", ""))
        verd[v] = verd.get(v, 0) + 1
        for f in str(d.get("flags", "")).split(";"):
            if f:
                flg[f] = flg.get(f, 0) + 1
    return {"verdicts": verd, "flags": flg}


__all__ = ["DEFAULT_VET", "KILL_ORDER", "VetResult", "dust_colour_test", "gaia_scatter_verdict",
           "in_star_forming_region", "load_optical_series", "neighbour_contamination",
           "optical_flatness", "optical_rise_verdict", "psf_weight", "rise_colour_verdict",
           "summarise", "vet_star"]
