"""CRADLE lead vet: everything reachable from a runner about the deep-vet survivors.

Gaia DR3 6225457312033584384 is the one star of run 35741356662 that passed
every deep-vet test (``results/cradle/deepvet.json``).  This stage tries to
kill it, and does the same for the two survivors whose deep vet had untested
routes (3168144078565656832, 5295632592220066688).  Every block records its
route status; a failed route is ``UNTESTED``, never a pass.

Blocks
------
1. ``neowise``   NEOWISE-R single exposures 2014-2024 + AllWISE multi-epoch
                 (2010), binned per visit; amplitude, chi2, trend, W1-W2
                 colour behaviour; the same statistics for every AllWISE source
                 of similar W1 within 4' as the null distribution.
2. ``sed``       2MASS / WISE / unWISE / CatWISE / AKARI IRC+FIS / IRAS PSC+FSC
                 / Spitzer SEIP / Herschel point-source catalogues; one- vs
                 two-temperature fits to the excess; the hot-dust limit from W2.
3. ``age``       Gaia astrophysical parameters (FLAME photometric and
                 spectroscopic ages, GSP-Spec [M/H] [alpha/Fe]), StarHorse;
                 DESI Li 6708 with the Fe I 6707.4 blend removed and an error,
                 Ca II H&K and IRT cores and H-alpha against the Redrock model;
                 TESS (SPOC / TESS-SPOC / QLP) and ZTF rotation; UVW, a Galactic
                 orbit and Bensby+2014 thin/thick/halo odds.
4. ``companions`` Gaia RUWE, astrometric excess noise, IPD, RV scatter
                 statistics, NSS orbit tables, co-moving sources within 60".
5. ``confusion`` IRSA DUST (SFD E(B-V), IRAS 100 um), unWISE W1/W3/W4 PSF
                 widths of the target against the field's point sources.
6. ``literature`` SIMBAD 60" cone, and every VizieR catalogue whose description
                 matches debris / infrared excess / warm dust keywords, coned.
7. ``occurrence`` (offline) the size of the effectively searched parent and
                 what the known old extreme debris disks imply for it.
"""

from __future__ import annotations

import io
import json
import math
import re
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .deepvet import (
    ALLWISE_EPOCH,
    GAIA_EPOCH,
    GAIA_TAP,
    IRSA_TAP,
    SIMBAD_TAP,
    _cone,
    _json_default,
    call_with_timeout,
    fetch_unwise,
    find_peaks,
    http_get,
    mag_to_jy,
    propagate,
    sep_arcsec,
    tap_query,
)

LEADS = ("6225457312033584384", "3168144078565656832", "5295632592220066688")

H_PLANCK = 6.62607015e-34
K_B = 1.380649e-23
C_LIGHT = 2.99792458e8
WISE_LAM_UM = {"W1": 3.3526, "W2": 4.6028, "W3": 11.5608, "W4": 22.0883}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def bnu(lam_um, t):
    """Planck B_nu (arbitrary normalisation is fine for ratios)."""
    nu = C_LIGHT / (np.asarray(lam_um, float) * 1e-6)
    x = H_PLANCK * nu / (K_B * t)
    return nu ** 3 / np.expm1(np.clip(x, 1e-8, 700))


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# 1. NEOWISE light curve
# ---------------------------------------------------------------------------
def bin_visits(df: pd.DataFrame, bands=("w1", "w2"), gap_d: float = 30.0) -> pd.DataFrame:
    """Per-visit weighted means (a visit = exposures separated by < gap_d days)."""
    d = df.sort_values("mjd").reset_index(drop=True)
    if not len(d):
        return pd.DataFrame()
    vid = np.concatenate([[0], np.cumsum(np.diff(d["mjd"].to_numpy(float)) > gap_d)])
    d = d.assign(visit=vid)
    rows = []
    for v, g in d.groupby("visit"):
        r = {"visit": int(v), "mjd": float(g["mjd"].median()), "n": int(len(g))}
        for b in bands:
            m = pd.to_numeric(g.get(f"{b}mpro"), errors="coerce").to_numpy(float)
            e = pd.to_numeric(g.get(f"{b}sigmpro"), errors="coerce").to_numpy(float)
            ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
            if ok.sum() < 2:
                r[b], r[f"{b}_err"], r[f"{b}_n"] = float("nan"), float("nan"), int(ok.sum())
                continue
            w = 1.0 / e[ok] ** 2
            mu = float(np.sum(w * m[ok]) / np.sum(w))
            scat = float(np.std(m[ok], ddof=1) / math.sqrt(ok.sum()))
            r[b] = mu
            r[f"{b}_err"] = max(scat, float(1.0 / math.sqrt(np.sum(w))))
            r[f"{b}_n"] = int(ok.sum())
        rows.append(r)
    return pd.DataFrame(rows)


def lc_stats(v: pd.DataFrame, bands=("w1", "w2")) -> dict:
    out = {"n_visits": int(len(v))}
    if len(v) < 3:
        return out
    t_yr = (v["mjd"].to_numpy(float) - float(v["mjd"].min())) / 365.25
    for b in bands:
        m = v[b].to_numpy(float)
        e = v[f"{b}_err"].to_numpy(float)
        ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
        if ok.sum() < 3:
            continue
        w = 1.0 / e[ok] ** 2
        mu = float(np.sum(w * m[ok]) / np.sum(w))
        chi2r = float(np.sum(((m[ok] - mu) / e[ok]) ** 2) / (ok.sum() - 1))
        A = np.vstack([np.ones(ok.sum()), t_yr[ok]]).T
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        coef = cov @ (A.T @ (w * m[ok]))
        out[b] = {"mean": mu, "range": float(np.nanmax(m[ok]) - np.nanmin(m[ok])),
                  "rms": float(np.std(m[ok])), "median_err": float(np.median(e[ok])),
                  "chi2_red": chi2r, "slope_mag_per_yr": float(coef[1]),
                  "slope_sigma": float(coef[1] / math.sqrt(cov[1, 1])),
                  "max_adjacent_jump": float(np.nanmax(np.abs(np.diff(m[ok])))),
                  "brightest_mjd": float(v["mjd"].to_numpy(float)[ok][np.argmin(m[ok])]),
                  "faintest_mjd": float(v["mjd"].to_numpy(float)[ok][np.argmax(m[ok])])}
    if "w1" in out and "w2" in out:
        m1, m2 = v["w1"].to_numpy(float), v["w2"].to_numpy(float)
        e1, e2 = v["w1_err"].to_numpy(float), v["w2_err"].to_numpy(float)
        ok = np.isfinite(m1) & np.isfinite(m2)
        col, ce = m1[ok] - m2[ok], np.hypot(e1[ok], e2[ok])
        w = 1.0 / ce ** 2
        cmu = float(np.sum(w * col) / np.sum(w))
        out["w1w2"] = {"mean": cmu, "range": float(col.max() - col.min()),
                       "chi2_red": float(np.sum(((col - cmu) / ce) ** 2) / max(ok.sum() - 1, 1))}
        out["w1_w2_pearson_r"] = float(np.corrcoef(m1[ok], m2[ok])[0, 1]) if ok.sum() > 2 else None
        # redder-when-brighter (dust) vs grey: slope of colour against W2
        if ok.sum() > 2 and np.std(m2[ok]) > 0:
            out["dcolour_dW2"] = float(np.polyfit(m2[ok], col, 1)[0])
    return out


def _clean_neowise(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    keep = np.ones(len(d), bool)
    if "qual_frame" in d:
        keep &= pd.to_numeric(d["qual_frame"], errors="coerce").fillna(0).to_numpy() > 0
    if "qi_fact" in d:
        keep &= pd.to_numeric(d["qi_fact"], errors="coerce").fillna(0).to_numpy() >= 0.9
    if "saa_sep" in d:
        keep &= pd.to_numeric(d["saa_sep"], errors="coerce").fillna(0).to_numpy() > 0
    if "moon_masked" in d:
        keep &= d["moon_masked"].astype(str).str[:2].isin(["00", "0"]).to_numpy()
    if "cc_flags" in d:
        keep &= d["cc_flags"].astype(str).str[:2].isin(["00", "0", "nan", "None"]).to_numpy()
    return d[keep]


def mep_field_change(t: dict, ra0: float, dec0: float, mv: pd.DataFrame, tap=tap_query) -> dict:
    """The same visit-to-visit W3/W4 change for bright field sources in the AllWISE
    multi-epoch table: a shift common to the field (the 2010 warm-up) is instrumental."""
    out = {"status": "UNTESTED"}
    if len(mv) < 2:
        return out
    w3 = float(t.get("w3mpro"))
    r = tap(IRSA_TAP, "SELECT source_id_mf, mjd, w3mpro_ep, w3sigmpro_ep, w4mpro_ep, w4sigmpro_ep, "
            "qual_frame, qi_fact, saa_sep, moon_masked FROM allwise_p3as_mep WHERE "
            f"{_cone('ra', 'dec', ra0, dec0, 15.0 / 60)} AND w3mpro_ep BETWEEN {w3 - 1.5:.2f} AND "
            f"{w3 + 1.0:.2f}", f"allwise_mep_field:{t['source_id']}")
    out["route"] = r.ledger()
    if r.status != "OK":
        return out
    d = r.data.copy()
    d.columns = [str(c).lower() for c in d.columns]
    d = d.rename(columns={"w3mpro_ep": "w3mpro", "w3sigmpro_ep": "w3sigmpro",
                          "w4mpro_ep": "w4mpro", "w4sigmpro_ep": "w4sigmpro"})
    d = _clean_neowise(d)
    split = float(np.mean(mv["mjd"].iloc[:2]))
    deltas3, deltas4 = [], []
    for _, g in d.groupby("source_id_mf"):
        a, b = g[g["mjd"] < split], g[g["mjd"] >= split]
        if len(a) < 3 or len(b) < 3:
            continue
        deltas3.append(float(np.nanmedian(b["w3mpro"]) - np.nanmedian(a["w3mpro"])))
        v4a, v4b = np.nanmedian(a["w4mpro"]), np.nanmedian(b["w4mpro"])
        if np.isfinite(v4a) and np.isfinite(v4b):
            deltas4.append(float(v4b - v4a))
    out["n_field"] = len(deltas3)
    if len(deltas3) >= 3:
        d3 = np.array(deltas3)
        out["w3_field_median_delta"] = float(np.median(d3))
        out["w3_field_mad_delta"] = float(1.4826 * np.median(np.abs(d3 - np.median(d3))))
    if len(deltas4) >= 3:
        d4 = np.array(deltas4)
        out["w4_field_median_delta"] = float(np.median(d4))
        out["w4_field_mad_delta"] = float(1.4826 * np.median(np.abs(d4 - np.median(d4))))
    out["status"] = "TESTED"
    return out


def neowise_block(t: dict, tap=tap_query) -> dict:
    out = {"status": "UNTESTED"}
    ra, dec = propagate(t["ra"], t["dec"], t["pmra"], t["pmdec"], 2019.0 - GAIA_EPOCH)
    cols = ("ra, dec, mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro, qual_frame, qi_fact, saa_sep, "
            "moon_masked, cc_flags, nb, na, w1rchi2, w2rchi2, allwise_cntr")
    r = tap(IRSA_TAP, f"SELECT {cols} FROM neowiser_p1bs_psd WHERE "
            f"{_cone('ra', 'dec', ra, dec, 3.0 / 3600)}", f"neowise:{t['source_id']}")
    out["route"] = r.ledger()
    if r.status != "OK":
        return out
    raw = r.data
    d = _clean_neowise(raw)
    out["n_exposures_raw"] = int(len(raw))
    out["n_exposures_clean"] = int(len(d))
    v = bin_visits(d)
    out["visits"] = v.round(4).to_dict(orient="records")
    out["stats"] = lc_stats(v)
    # AllWISE multi-epoch (2010, cryogenic W1-W4)
    ra0, dec0 = propagate(t["ra"], t["dec"], t["pmra"], t["pmdec"], 2010.3 - GAIA_EPOCH)
    r2 = tap(IRSA_TAP, "SELECT * FROM allwise_p3as_mep WHERE "
             f"{_cone('ra', 'dec', ra0, dec0, 3.0 / 3600)}", f"allwise_mep:{t['source_id']}")
    out["allwise_mep_route"] = r2.ledger()
    if r2.status == "OK":
        m = r2.data.copy()
        m.columns = [str(c).lower() for c in m.columns]
        for b in ("w1", "w2", "w3", "w4"):
            if f"{b}mpro_ep" in m:
                m[f"{b}mpro"] = m[f"{b}mpro_ep"]
                m[f"{b}sigmpro"] = m.get(f"{b}sigmpro_ep")
        mjdcol = "mjd" if "mjd" in m else None
        if mjdcol:
            mv = bin_visits(_clean_neowise(m), bands=("w1", "w2", "w3", "w4"))
            out["allwise_visits"] = mv.round(4).to_dict(orient="records")
            if len(mv) >= 2:
                out["allwise_w3w4_change"] = {
                    b: {"first": _f(mv[b].iloc[0]), "last": _f(mv[b].iloc[-1]),
                        "delta": _f(mv[b].iloc[-1] - mv[b].iloc[0]),
                        "sigma": _f(math.hypot(_f(mv[f"{b}_err"].iloc[0]), _f(mv[f"{b}_err"].iloc[-1])))}
                    for b in ("w3", "w4") if b in mv}
                out["allwise_w3w4_change_field"] = mep_field_change(t, ra0, dec0, mv, tap)
    # comparison stars: every AllWISE-matched source of similar W1 within 10'
    w1 = float(t.get("w1mpro"))
    r3 = tap(IRSA_TAP, f"SELECT {cols} FROM neowiser_p1bs_psd WHERE "
             f"{_cone('ra', 'dec', ra, dec, 10.0 / 60)} AND w1mpro BETWEEN {w1 - 0.75:.2f} AND "
             f"{w1 + 0.75:.2f} AND allwise_cntr > 0", f"neowise_field:{t['source_id']}")
    out["field_route"] = r3.ledger()
    if r3.status == "OK":
        f = _clean_neowise(r3.data)
        comp = []
        tgt_cntr = None
        if "allwise_cntr" in d and len(d):
            tgt_cntr = pd.to_numeric(d["allwise_cntr"], errors="coerce").mode()
            tgt_cntr = int(tgt_cntr.iloc[0]) if len(tgt_cntr) else None
        for cntr, g in f.groupby("allwise_cntr"):
            if tgt_cntr is not None and int(cntr) == tgt_cntr:
                continue
            if len(g) < 60:
                continue
            s = lc_stats(bin_visits(g))
            if "w1" in s and "w2" in s:
                comp.append({"cntr": int(cntr), "w1": s["w1"]["mean"],
                             "chi2_w1": s["w1"]["chi2_red"], "chi2_w2": s["w2"]["chi2_red"],
                             "range_w1": s["w1"]["range"], "range_w2": s["w2"]["range"],
                             "slope_w1": s["w1"]["slope_mag_per_yr"],
                             "slope_w2": s["w2"]["slope_mag_per_yr"]})
        out["n_comparison"] = len(comp)
        if comp and "w1" in out["stats"] and "w2" in out["stats"]:
            cdf = pd.DataFrame(comp)
            st = out["stats"]
            out["comparison"] = {
                "median_chi2_w1": float(cdf["chi2_w1"].median()),
                "median_chi2_w2": float(cdf["chi2_w2"].median()),
                "median_range_w1": float(cdf["range_w1"].median()),
                "median_range_w2": float(cdf["range_w2"].median()),
                "pct_target_chi2_w1": float((cdf["chi2_w1"] < st["w1"]["chi2_red"]).mean() * 100),
                "pct_target_chi2_w2": float((cdf["chi2_w2"] < st["w2"]["chi2_red"]).mean() * 100),
                "pct_target_range_w2": float((cdf["range_w2"] < st["w2"]["range"]).mean() * 100),
                # a W2 fade common to the field is instrumental (NEOWISE drift)
                "median_slope_w1": float(cdf["slope_w1"].median()),
                "median_slope_w2": float(cdf["slope_w2"].median()),
                "mad_slope_w2": float(1.4826 * (cdf["slope_w2"] - cdf["slope_w2"].median()).abs().median()),
                "target_slope_w2_minus_field_in_mad":
                    float((st["w2"]["slope_mag_per_yr"] - cdf["slope_w2"].median())
                          / max(1.4826 * (cdf["slope_w2"] - cdf["slope_w2"].median()).abs().median(), 1e-4)),
            }
    out["status"] = "TESTED"
    return out


# ---------------------------------------------------------------------------
# generic VizieR rows
# ---------------------------------------------------------------------------
VIZIER_CATS = {
    # photometry
    "II/246/out": 3, "II/328/allwise": 3, "II/363/unwise": 3, "II/365/catwise": 3,
    "II/297/irc": 10, "II/298/fis": 30, "II/125/main": 60, "II/156A/main": 60,
    "II/335/galex_ais": 5, "II/379/smssdr4": 3, "II/367/vhs_dr5": 3,
    "VIII/106/hppsc070": 10, "VIII/106/hppsc100": 10, "VIII/106/hppsc160": 15,
    # parameters / RVs / ages
    "I/355/gaiadr3": 2, "I/355/paramp": 2, "I/355/paramsup": 2,
    "I/354/starhorse2021": 2, "J/A+A/700/A195/catalog": 3, "J/A+A/659/A95/sos": 3,
    "J/MNRAS/529/1802/targets": 3, "J/MNRAS/522/29/table2": 3, "J/ApJS/271/26/catalog": 3,
    "J/A+A/684/A29/main": 3, "J/A+A/695/A75/catalog": 3, "J/ApJ/984/58/table1": 3,
    "J/A+A/692/A115/pristine_ca": 3, "IV/39/tic82": 3, "V/161/zcatdr1": 3,
    "J/A+A/674/A1/table1": 3,
}


def vizier_rows(t: dict, cats: dict = VIZIER_CATS) -> dict:
    out = {}
    try:
        from astropy import units as u  # noqa: PLC0415
        from astropy.coordinates import SkyCoord  # noqa: PLC0415
        from astroquery.vizier import Vizier  # noqa: PLC0415
    except Exception as exc:                              # noqa: BLE001
        return {"_error": repr(exc)}
    ra, dec = propagate(t["ra"], t["dec"], t["pmra"], t["pmdec"], 2000.0 - GAIA_EPOCH)
    pos = SkyCoord(ra, dec, unit="deg")
    for cat, rad in cats.items():
        try:
            v = Vizier(columns=["**", "+_r"], row_limit=20, timeout=120)
            tl = v.query_region(pos, radius=rad * u.arcsec, catalog=cat)
            rows = []
            for tab in tl:
                for rr in tab:
                    d = {}
                    for k in tab.colnames:
                        val = rr[k]
                        try:
                            val = val.item()
                        except Exception:                 # noqa: BLE001
                            pass
                        if isinstance(val, bytes):
                            val = val.decode(errors="replace")
                        if isinstance(val, float) and not np.isfinite(val):
                            val = None
                        d[k] = val if isinstance(val, (int, float, str, type(None))) else str(val)
                    rows.append(d)
            out[cat] = {"status": "OK" if rows else "ZERO_ROWS", "radius_arcsec": rad, "rows": rows[:5]}
        except Exception as exc:                          # noqa: BLE001
            out[cat] = {"status": "FAILED", "error": repr(exc)[:200]}
    return out


def _row(vz: dict, cat: str) -> dict:
    r = vz.get(cat, {}).get("rows") or []
    return r[0] if r else {}


# ---------------------------------------------------------------------------
# 2. SED
# ---------------------------------------------------------------------------
_TGRID = np.geomspace(60, 3000, 500)


def fit_bb(lam_um, fx, ex, tgrid=_TGRID):
    """Best single blackbody F = Omega B_nu(T): chi2 over T with Omega analytic."""
    lam, fx, ex = (np.asarray(a, float) for a in (lam_um, fx, ex))
    best = (np.inf, np.nan, np.nan)
    chis = []
    for tt in tgrid:
        b = bnu(lam, tt)
        w = 1.0 / ex ** 2
        om = np.sum(w * fx * b) / np.sum(w * b * b)
        chi = float(np.sum(((fx - om * b) / ex) ** 2))
        chis.append(chi)
        if chi < best[0]:
            best = (chi, tt, om)
    chis = np.array(chis)
    ok = chis <= best[0] + 1.0
    return {"chi2": best[0], "t_k": float(best[1]), "omega": float(best[2]),
            "t_lo": float(tgrid[ok].min()), "t_hi": float(tgrid[ok].max()), "n": int(len(lam))}


def sed_block(t: dict, vz: dict, deep: dict) -> dict:
    out = {"status": "TESTED", "points": []}
    p = out["points"]
    phot3 = float(mag_to_jy(t["w3mpro"], "W3")) - float(t["W3_excess_jy"])
    phot4 = float(mag_to_jy(t["w4mpro"], "W4")) - float(t["W4_excess_jy"])
    # AllWISE
    for b, ph, ex in (("W3", phot3, t["W3_excess_jy"]), ("W4", phot4, t["W4_excess_jy"])):
        e = float(mag_to_jy(t[f"{b.lower()}mpro"], b)) * float(t.get(f"{b.lower()}mpro_error") or 0.03) / 1.0857
        p.append({"band": f"AllWISE {b}", "lam_um": WISE_LAM_UM[b], "excess_jy": float(ex),
                  "err_jy": math.hypot(e, 0.05 * float(ex)), "photosphere_jy": ph})
    # Legacy Surveys deblended forced photometry (from deepvet)
    ls = (deep or {}).get("checks", {}).get("ls", {})
    if ls.get("target_found"):
        for b, ph in (("w3", phot3), ("w4", phot4)):
            f, e = ls.get(f"{b}_forced_jy"), ls.get(f"{b}_forced_err_jy")
            if f is not None and e is not None and np.isfinite(f) and np.isfinite(e):
                p.append({"band": f"LS-forced {b.upper()}", "lam_um": WISE_LAM_UM[b.upper()],
                          "excess_jy": f - ph, "err_jy": math.hypot(e, 0.05 * f), "photosphere_jy": ph})
    # AKARI IRC: S09 / S18 in Jy (non-detections are simply absent from the PSC)
    irc = _row(vz, "II/297/irc")
    for col, lam in (("S09", 9.0), ("S18", 18.0)):
        if irc.get(col) is not None:
            p.append({"band": f"AKARI {col}", "lam_um": lam, "total_jy": _f(irc[col]),
                      "err_jy": _f(irc.get(f"e_{col}"))})
    # Spitzer / IRAS / Herschel noted as detected-or-not
    out["catalog_detections"] = {k: vz.get(k, {}).get("status") for k in
                                 ("II/297/irc", "II/298/fis", "II/125/main", "II/156A/main",
                                  "VIII/106/hppsc070", "VIII/106/hppsc100", "VIII/106/hppsc160",
                                  "II/363/unwise", "II/365/catwise")}
    ex = [q for q in p if "excess_jy" in q and np.isfinite(q["err_jy"]) and q["err_jy"] > 0]
    by_src = {}
    for q in ex:
        by_src.setdefault(q["band"].split()[0], []).append(q)
    fits = {}
    for src, qs in by_src.items():
        if len(qs) >= 2:
            fits[src] = fit_bb([q["lam_um"] for q in qs], [q["excess_jy"] for q in qs],
                               [q["err_jy"] for q in qs])
    if len(ex) >= 2:
        fits["all_excess_points"] = fit_bb([q["lam_um"] for q in ex], [q["excess_jy"] for q in ex],
                                           [q["err_jy"] for q in ex])
    out["single_bb_fits"] = fits
    n_indep_lam = len({round(q["lam_um"], 1) for q in ex})
    out["n_independent_wavelengths_in_excess"] = n_indep_lam
    out["two_temperature_fit"] = ("NOT_CONSTRAINED: excess measured at only "
                                  f"{n_indep_lam} wavelengths (W3, W4); a two-temperature model "
                                  "has 4 parameters") if n_indep_lam < 4 else "see fits"
    # hot-dust limit from W2 and the 262 K prediction at W2
    w2x, chi2 = _f(t.get("W2_excess_jy")), _f(t.get("chi_W2"))
    w2err = abs(w2x / chi2) if np.isfinite(chi2) and chi2 != 0 else float("nan")
    fa = fits.get("AllWISE") or fits.get("all_excess_points")
    if fa:
        pred_w2 = fa["omega"] * float(bnu(WISE_LAM_UM["W2"], fa["t_k"]))
        pred_w1 = fa["omega"] * float(bnu(WISE_LAM_UM["W1"], fa["t_k"]))
        out["w2_excess_measured_jy"] = w2x
        out["w2_excess_err_jy"] = w2err
        out["w2_excess_predicted_by_bb_jy"] = pred_w2
        out["w1_excess_predicted_by_bb_jy"] = pred_w1
        # the largest hot (800 K) component W2 allows at 3 sigma, as a fraction of f
        if np.isfinite(w2err):
            lim = max(w2x, 0.0) + 3 * w2err
            b800 = float(bnu(WISE_LAM_UM["W2"], 800.0))
            om_hot = lim / b800
            f_hot_over_f = (om_hot * 800.0 ** 4) / (fa["omega"] * fa["t_k"] ** 4)
            out["hot_800K_component_max_fraction_of_f_3sigma"] = float(f_hot_over_f)
    return out


# ---------------------------------------------------------------------------
# 3. Age
# ---------------------------------------------------------------------------
#: Mamajek dwarf sequence, Teff -> B-V (approximate, for the Fe I blend and gyrochronology)
_TEFF_BV = [(4830, 0.98), (4990, 0.94), (5100, 0.88), (5170, 0.86), (5280, 0.82), (5320, 0.80),
            (5380, 0.78), (5480, 0.74), (5550, 0.72), (5600, 0.70), (5660, 0.69), (5680, 0.68),
            (5720, 0.66), (5770, 0.65), (5880, 0.62), (5920, 0.60), (6000, 0.58), (6100, 0.54),
            (6250, 0.50)]


def teff_to_bv(teff: float) -> float:
    xs, ys = zip(*_TEFF_BV, strict=True)
    return float(np.interp(teff, xs, ys))


def gyro_age_myr(period_d: float, bv: float) -> float:
    """Mamajek & Hillenbrand 2008: P = a (B-V - c)^b t^n."""
    a, b, c, n = 0.407, 0.325, 0.495, 0.566
    if not (np.isfinite(period_d) and bv > c):
        return float("nan")
    return float((period_d / (a * (bv - c) ** b)) ** (1.0 / n))


def li_measure(w, f, ivar, teff: float, model=None) -> dict:
    """Li 6708 EW with the Fe I 6707.44 (air) blend removed (Soderblom+1993) and an error.

    The continuum windows hold weak lines (Ca I 6719.5 vac among them), so the
    linear continuum is fitted with iterative clipping of low pixels.  The
    empirical noise is the scatter of flux/model where a model exists (the model
    carries those lines), else the clipped continuum scatter.
    """
    w, f, iv = (np.asarray(a, float) for a in (w, f, ivar))
    cw = ((w >= 6695.0) & (w <= 6705.0)) | ((w >= 6714.0) & (w <= 6725.0))
    ok = cw & np.isfinite(f) & (iv > 0)
    if ok.sum() < 6:
        return {"status": "UNTESTED", "why": "no continuum pixels"}
    use = ok.copy()
    for _ in range(5):
        a, b0 = np.polyfit(w[use], f[use], 1)
        r = f - (a * w + b0)
        s = float(np.std(r[use])) or 1e-9
        new = ok & (r > -1.5 * s)
        if new.sum() < 6 or (new == use).all():
            break
        use = new
    sel = (w >= 6707.6) & (w <= 6711.8) & np.isfinite(f) & (iv > 0)
    cont = a * w[sel] + b0
    dl = np.gradient(w)[sel]
    ew = float(np.sum((1.0 - f[sel] / cont) * dl))
    sig = float(np.sqrt(np.sum((dl / cont) ** 2 / iv[sel])))
    if model is not None and np.isfinite(np.asarray(model, float)).any():
        md = np.asarray(model, float)
        nz = (w >= 6680) & (w <= 6740) & ~((w >= 6705) & (w <= 6714)) & np.isfinite(md) & (md > 0) \
            & np.isfinite(f)
        rel = f[nz] / md[nz]
        cont_snr = float(1.0 / (1.4826 * np.median(np.abs(rel - np.median(rel))))) if nz.sum() > 10 \
            else float("nan")
        noise_src = "flux/model MAD"
    else:
        resid = f[use] - (a * w[use] + b0)
        cont_snr = float(np.median(a * w[use] + b0) / np.std(resid)) if np.std(resid) > 0 else float("nan")
        noise_src = "clipped continuum scatter"
    sig_emp = float(np.sqrt(sel.sum()) * np.median(dl) / cont_snr) if np.isfinite(cont_snr) else sig
    bv = teff_to_bv(teff)
    ew_fe = (20.0 * bv - 3.0) / 1000.0
    return {"status": "TESTED", "ew_blend_A": ew, "ew_err_ivar_A": sig, "ew_err_empirical_A": sig_emp,
            "continuum_snr": cont_snr, "noise_source": noise_src, "bv_adopted": bv,
            "ew_fe_correction_A": ew_fe,
            "ew_li_A": ew - ew_fe, "ew_li_err_A": max(sig, sig_emp)}


def core_ratio(w, f, model, centre: float, half: float, cont_windows) -> float:
    """(core flux / pseudo-continuum) of the data over the same for the model."""
    w, f, m = (np.asarray(a, float) for a in (w, f, model))
    core = (w >= centre - half) & (w <= centre + half)
    cm = np.zeros_like(w, bool)
    for lo, hi in cont_windows:
        cm |= (w >= lo) & (w <= hi)
    if core.sum() < 2 or cm.sum() < 4:
        return float("nan")
    rd = np.nanmean(f[core]) / np.nanmean(f[cm])
    rm = np.nanmean(m[core]) / np.nanmean(m[cm])
    return float(rd / rm) if rm > 0 else float("nan")


def desi_block(t: dict) -> dict:
    out = {"status": "UNTESTED"}
    try:
        from sparcl.client import SparclClient  # noqa: PLC0415
    except Exception as exc:                              # noqa: BLE001
        out["error"] = repr(exc)[:200]
        return out
    try:
        client = SparclClient(connect_timeout=30.0)
    except Exception:                                     # noqa: BLE001
        client = SparclClient()
    d = 2.5 / 3600.0
    cosd = max(math.cos(math.radians(t["dec"])), 1e-3)
    cons = {"ra": [t["ra"] - d / cosd, t["ra"] + d / cosd], "dec": [t["dec"] - d, t["dec"] + d]}
    found = client.find(outfields=["sparcl_id", "redshift", "redshift_err", "spectype",
                                   "data_release"], constraints=cons, limit=20)
    recs = list(getattr(found, "records", found) or [])

    def g(r, k):
        return r.get(k) if isinstance(r, dict) else getattr(r, k, None)
    out["spectra"] = [{k: (g(r, k) if not isinstance(g(r, k), (np.floating,)) else float(g(r, k)))
                       for k in ("sparcl_id", "redshift", "redshift_err", "spectype", "data_release")}
                      for r in recs]
    if not recs:
        out["status"] = "NO_SPECTRUM"
        return out
    got = client.retrieve(uuid_list=[str(g(r, "sparcl_id")) for r in recs][:3],
                          include=["sparcl_id", "wavelength", "flux", "ivar", "model", "data_release"])
    grec = list(getattr(got, "records", got) or [])
    meas = []
    teff = float(t.get("teff_gspphot") or 5700.0)
    for r in grec:
        w = np.asarray(g(r, "wavelength"), float)
        f = np.asarray(g(r, "flux"), float)
        iv = np.asarray(g(r, "ivar"), float)
        md = g(r, "model")
        md = np.asarray(md, float) if md is not None else np.full_like(f, np.nan)
        m = {"sparcl_id": str(g(r, "sparcl_id")), "li": li_measure(w, f, iv, teff, md)}
        # Ca II H&K (vac 3934.78, 3969.59), IRT 8500.35/8544.44/8664.52, H-alpha 6564.61
        m["cahk_core_over_model"] = {
            "K": core_ratio(w, f, md, 3934.78, 1.0, [(3891, 3911), (3991, 4011)]),
            "H": core_ratio(w, f, md, 3969.59, 1.0, [(3891, 3911), (3991, 4011)])}
        m["irt_core_over_model"] = {
            str(c): core_ratio(w, f, md, c, 0.8, [(c - 12, c - 6), (c + 6, c + 12)])
            for c in (8500.35, 8544.44, 8664.52)}
        m["halpha_core_over_model"] = core_ratio(w, f, md, 6564.61, 1.0, [(6545, 6555), (6575, 6585)])
        blue = (w > 3900) & (w < 4000) & (iv > 0)
        m["snr_3950"] = float(np.median(f[blue] * np.sqrt(iv[blue]))) if blue.any() else None
        red = (w > 6690) & (w < 6730) & (iv > 0)
        m["snr_6700"] = float(np.median(f[red] * np.sqrt(iv[red]))) if red.any() else None
        meas.append(m)
    out["measurements"] = meas
    out["status"] = "TESTED"
    return out


def tess_block(t: dict, tic: str | None) -> dict:
    out = {"status": "UNTESTED", "tic": tic}
    try:
        from astropy.io import fits  # noqa: PLC0415
        from astropy.timeseries import LombScargle  # noqa: PLC0415
        from astroquery.mast import Observations  # noqa: PLC0415
    except Exception as exc:                              # noqa: BLE001
        out["error"] = repr(exc)[:200]
        return out
    if not tic:
        try:
            from astroquery.mast import Catalogs  # noqa: PLC0415
            cat = Catalogs.query_region(f"{t['ra']} {t['dec']}", radius=0.001, catalog="TIC")
            if len(cat):
                tic = str(cat[0]["ID"])
                out["tic"] = tic
        except Exception as exc:                          # noqa: BLE001
            out["tic_error"] = repr(exc)[:200]
    if not tic:
        return out
    obs = Observations.query_criteria(target_name=[tic, f"TIC {tic}", str(int(tic))],
                                      obs_collection=["TESS", "HLSP"])
    out["n_observations"] = int(len(obs))
    if not len(obs):
        out["status"] = "NO_TESS_LIGHTCURVE"
        return out
    prods = Observations.get_product_list(obs)
    names = [str(x) for x in prods["productFilename"]]
    keep = [i for i, n in enumerate(names) if n.endswith("lc.fits") or n.endswith("_llc.fits")]
    out["n_lc_products"] = len(keep)
    sectors = []
    for i in keep[:12]:
        uri = str(prods["dataURI"][i])
        try:
            # download_file returns (status, message, url) and writes local_path;
            # the first dispatch opened loc[1] (the message, None) for every sector
            path = f"/tmp/{names[i]}"
            st = Observations.download_file(uri, local_path=path)
            if isinstance(st, tuple) and str(st[0]).upper() not in ("COMPLETE", "SKIPPED"):
                raise RuntimeError(f"download {st}")
            h = fits.open(path)
            dat = h[1].data
            cols = dat.columns.names
            fcol = next((c for c in ("PDCSAP_FLUX", "KSPSAP_FLUX", "SAP_FLUX") if c in cols), None)
            tt = np.asarray(dat["TIME"], float)
            ff = np.asarray(dat[fcol], float)
            q = np.asarray(dat["QUALITY"], int) if "QUALITY" in cols else np.zeros_like(tt, int)
            ok = np.isfinite(tt) & np.isfinite(ff) & (q == 0)
            tt, ff = tt[ok], ff[ok] / np.nanmedian(ff[ok])
            if len(tt) < 200:
                continue
            freq, pw = LombScargle(tt, ff).autopower(minimum_frequency=1 / 13.0,
                                                     maximum_frequency=1 / 0.2, samples_per_peak=10)
            k = int(np.argmax(pw))
            sectors.append({"file": names[i], "flux_col": fcol, "n": int(len(tt)),
                            "period_d": float(1 / freq[k]), "power": float(pw[k]),
                            "fap": float(LombScargle(tt, ff).false_alarm_probability(pw[k])),
                            "p5_p95_ppt": float((np.percentile(ff, 95) - np.percentile(ff, 5)) * 1e3)})
        except Exception as exc:                          # noqa: BLE001
            sectors.append({"file": names[i], "error": repr(exc)[:200]})
    out["sectors"] = sectors
    good = [s for s in sectors if s.get("fap") is not None and s["fap"] < 1e-3 and s["power"] > 0.1]
    out["significant_periods_d"] = [round(s["period_d"], 3) for s in good]
    out["status"] = "TESTED"
    return out


def ztf_block(t: dict) -> dict:
    out = {"status": "UNTESTED"}
    if t["dec"] < -31:
        out["status"] = "OUT_OF_FOOTPRINT"
        return out
    url = ("https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE%20"
           f"{t['ra']:.6f}%20{t['dec']:.6f}%200.0005&BANDNAME=g,r&FORMAT=csv")
    code, content, err = http_get(url, timeout=120)
    out["http"] = code
    if code != 200:
        out["error"] = err[:200]
        return out
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:                              # noqa: BLE001
        out["error"] = repr(exc)[:200]
        return out
    out["n_rows"] = int(len(df))
    if not len(df):
        out["status"] = "NO_DATA"
        return out
    from astropy.timeseries import LombScargle  # noqa: PLC0415
    res = {}
    for band, g in df.groupby("filtercode"):
        # catflags 0 removed every point of the lead (it saturates ZTF at r~12.6);
        # keep everything except the hard-bad bit 32768 and flag saturation below
        g = g[(g["catflags"] < 32768)] if "catflags" in g else g
        if len(g) < 30:
            continue
        tt, mm, ee = g["mjd"].to_numpy(float), g["mag"].to_numpy(float), g["magerr"].to_numpy(float)
        freq, pw = LombScargle(tt, mm, ee).autopower(minimum_frequency=1 / 60.0,
                                                    maximum_frequency=1 / 0.2, samples_per_peak=10)
        k = int(np.argmax(pw))
        res[str(band)] = {"n": int(len(g)), "median_mag": float(np.median(mm)),
                          "rms_mag": float(np.std(mm)), "median_err": float(np.median(ee)),
                          "period_d": float(1 / freq[k]), "power": float(pw[k]),
                          "fap": float(LombScargle(tt, mm, ee).false_alarm_probability(pw[k])),
                          "saturation_warning": bool(np.median(mm) < 13.0)}
    out["bands"] = res
    out["status"] = "TESTED"
    return out


# --- kinematics ---------------------------------------------------------------
G_KPC = 4.30091e-6      # kpc (km/s)^2 / Msun


def _acc(x, y, z):
    """Bovy-like MW: Hernquist bulge + Miyamoto-Nagai disc + NFW halo."""
    r2 = x * x + y * y
    r = math.sqrt(r2 + z * z)
    # bulge
    mb, ab = 0.5e10, 0.6
    fb = -G_KPC * mb / (r * (r + ab) ** 2) if r > 0 else 0.0
    ax, ay, az = fb * x, fb * y, fb * z
    # disc
    md, a, b = 6.8e10, 3.0, 0.28
    zb = math.sqrt(z * z + b * b)
    den = (r2 + (a + zb) ** 2) ** 1.5
    ax += -G_KPC * md * x / den
    ay += -G_KPC * md * y / den
    az += -G_KPC * md * z * (a + zb) / (zb * den)
    # halo (NFW, M(<r) = 4 pi rho0 rs^3 [ln(1+x) - x/(1+x)])
    rs, m0 = 16.0, 8.0e11 / (math.log(1 + 15.3) - 15.3 / 16.3)
    xx = r / rs
    mr = m0 * (math.log(1 + xx) - xx / (1 + xx))
    fh = -G_KPC * mr / r ** 3 if r > 0 else 0.0
    return ax + fh * x, ay + fh * y, az + fh * z


def orbit(pos, vel, t_gyr: float = 3.0, dt_myr: float = 0.25) -> dict:
    kms_to_kpc_per_myr = 1.0227e-3
    x, y, z = pos
    vx, vy, vz = (v * kms_to_kpc_per_myr for v in vel)
    n = int(t_gyr * 1000 / dt_myr)
    R, Z = [], []
    ax, ay, az = (a * kms_to_kpc_per_myr ** 2 for a in _acc(x, y, z))
    for _ in range(n):
        vx += 0.5 * dt_myr * ax
        vy += 0.5 * dt_myr * ay
        vz += 0.5 * dt_myr * az
        x += dt_myr * vx
        y += dt_myr * vy
        z += dt_myr * vz
        ax, ay, az = (a * kms_to_kpc_per_myr ** 2 for a in _acc(x, y, z))
        vx += 0.5 * dt_myr * ax
        vy += 0.5 * dt_myr * ay
        vz += 0.5 * dt_myr * az
        R.append(math.hypot(x, y))
        Z.append(abs(z))
    rp, ra = min(R), max(R)
    return {"r_peri_kpc": rp, "r_apo_kpc": ra, "z_max_kpc": max(Z), "ecc": (ra - rp) / (ra + rp)}


def bensby_odds(U, V, W) -> dict:
    """Bensby, Feltzing & Oey 2014 (A&A 562, A71) Table A.1, LSR frame."""
    comp = {"thin": (0.85, 35, 20, 16, -15), "thick": (0.09, 67, 38, 35, -46),
            "halo": (0.0015, 160, 90, 90, -220)}
    p = {}
    for k, (X, su, sv, sw, va) in comp.items():
        norm = 1.0 / ((2 * math.pi) ** 1.5 * su * sv * sw)
        p[k] = X * norm * math.exp(-U * U / (2 * su * su) - (V - va) ** 2 / (2 * sv * sv)
                                   - W * W / (2 * sw * sw))
    return {"TD_over_D": p["thick"] / p["thin"], "TD_over_H": p["thick"] / p["halo"],
            "P": {k: v / sum(p.values()) for k, v in p.items()}}


def kinematics_block(t: dict) -> dict:
    out = {"status": "UNTESTED"}
    try:
        from astropy import units as u  # noqa: PLC0415
        from astropy.coordinates import Galactic, Galactocentric, SkyCoord  # noqa: PLC0415
    except Exception as exc:                              # noqa: BLE001
        out["error"] = repr(exc)
        return out
    rv = _f(t.get("radial_velocity"))
    if not np.isfinite(rv):
        out["error"] = "no RV"
        return out
    c = SkyCoord(ra=t["ra"] * u.deg, dec=t["dec"] * u.deg, distance=(1000.0 / t["parallax"]) * u.pc,
                 pm_ra_cosdec=t["pmra"] * u.mas / u.yr, pm_dec=t["pmdec"] * u.mas / u.yr,
                 radial_velocity=rv * u.km / u.s)
    gal = c.transform_to(Galactic())
    vel = gal.velocity.d_xyz.to(u.km / u.s).value        # heliocentric U,V,W (U toward GC)
    lsr = (11.1, 12.24, 7.25)                            # Schoenrich+2010
    U, V, W = (float(vel[i] + lsr[i]) for i in range(3))
    out.update({"U_lsr": U, "V_lsr": V, "W_lsr": W, "v_tot_lsr": math.sqrt(U * U + V * V + W * W),
                "bensby2014": bensby_odds(U, V, W)})
    gc = c.transform_to(Galactocentric())
    pos = (float(gc.x.to(u.kpc).value), float(gc.y.to(u.kpc).value), float(gc.z.to(u.kpc).value))
    vv = (float(gc.v_x.to(u.km / u.s).value), float(gc.v_y.to(u.km / u.s).value),
          float(gc.v_z.to(u.km / u.s).value))
    out["galactocentric_xyz_kpc"] = pos
    out["orbit"] = orbit(pos, vv)
    # RV error Monte Carlo on the orbit's z_max
    zs = []
    rng = np.random.default_rng(1)
    rve = _f(t.get("radial_velocity_error")) or 1.0
    for _ in range(20):
        c2 = SkyCoord(ra=t["ra"] * u.deg, dec=t["dec"] * u.deg,
                      distance=(1000.0 / t["parallax"]) * u.pc,
                      pm_ra_cosdec=t["pmra"] * u.mas / u.yr, pm_dec=t["pmdec"] * u.mas / u.yr,
                      radial_velocity=(rv + rng.normal(0, rve)) * u.km / u.s).transform_to(Galactocentric())
        o = orbit((float(c2.x.to(u.kpc).value), float(c2.y.to(u.kpc).value), float(c2.z.to(u.kpc).value)),
                  (float(c2.v_x.to(u.km / u.s).value), float(c2.v_y.to(u.km / u.s).value),
                   float(c2.v_z.to(u.km / u.s).value)), t_gyr=1.0, dt_myr=0.5)
        zs.append(o["z_max_kpc"])
    out["z_max_kpc_16_84"] = [float(np.percentile(zs, 16)), float(np.percentile(zs, 84))]
    out["status"] = "TESTED"
    return out


# ---------------------------------------------------------------------------
# 4. Companions (Gaia)
# ---------------------------------------------------------------------------
def gaia_block(t: dict, tap=tap_query) -> dict:
    sid = t["source_id"]
    out = {"status": "UNTESTED"}
    q = ("SELECT g.source_id, g.ruwe, g.astrometric_excess_noise, g.astrometric_excess_noise_sig, "
         "g.astrometric_chi2_al, g.astrometric_n_good_obs_al, g.ipd_gof_harmonic_amplitude, "
         "g.ipd_frac_multi_peak, g.ipd_frac_odd_win, g.duplicated_source, g.non_single_star, "
         "g.radial_velocity, g.radial_velocity_error, g.rv_nb_transits, g.rv_chisq_pvalue, "
         "g.rv_renormalised_gof, g.rv_amplitude_robust, g.rv_template_teff, g.phot_variable_flag, "
         "g.phot_bp_rp_excess_factor, g.vbroad, g.vbroad_error, g.grvs_mag, "
         "a.age_flame, a.age_flame_lower, a.age_flame_upper, a.age_flame_spec, "
         "a.age_flame_spec_lower, a.age_flame_spec_upper, a.mass_flame, a.mass_flame_spec, "
         "a.lum_flame, a.evolstage_flame, a.evolstage_flame_spec, a.teff_gspphot, a.logg_gspphot, "
         "a.mh_gspphot, a.teff_gspspec, a.logg_gspspec, a.mh_gspspec, a.alphafe_gspspec, "
         "a.flags_gspspec, a.activityindex_espcs, a.activityindex_espcs_uncertainty, "
         "a.classprob_dsc_combmod_star, a.teff_esphs, a.spectraltype_esphs "
         "FROM gaiadr3.gaia_source AS g LEFT JOIN gaiadr3.astrophysical_parameters AS a "
         f"ON a.source_id = g.source_id WHERE g.source_id = {sid}")
    del q  # the joined spelling failed to parse on the ESA server (dispatch 36003679119)
    row = {}
    routes = []
    for tab in ("gaia_source", "astrophysical_parameters"):
        r = tap(GAIA_TAP, f"SELECT * FROM gaiadr3.{tab} WHERE source_id = {sid}", f"{tab}:{sid}")
        routes.append(r.ledger())
        if r.status == "OK":
            for k, v in r.data.iloc[0].to_dict().items():
                v = v.item() if hasattr(v, "item") else v
                if isinstance(v, bytes):
                    v = v.decode(errors="replace")
                if isinstance(v, (int, float, str, bool, type(None))):
                    row.setdefault(str(k), v)
    out["routes"] = routes
    if row:
        out["row"] = row
        out["status"] = "TESTED"
    for tab in ("nss_two_body_orbit", "nss_acceleration_astro", "nss_non_linear_spectro",
                "nss_vim_fl"):
        r2 = tap(GAIA_TAP, f"SELECT * FROM gaiadr3.{tab} WHERE source_id = {sid}", f"{tab}:{sid}",
                 retries=2)
        out[tab] = r2.status
    # co-moving companions within 60"
    q3 = ("SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmdec, phot_g_mean_mag, "
          f"bp_rp, ruwe FROM gaiadr3.gaia_source WHERE {_cone('ra', 'dec', t['ra'], t['dec'], 60 / 3600)}")
    r3 = tap(GAIA_TAP, q3, f"gaia_60:{sid}")
    out["cone60_route"] = r3.ledger()
    if r3.status == "OK":
        d = r3.data.copy()
        d.columns = [str(c).lower() for c in d.columns]
        d = d[d["source_id"].astype(str) != sid]
        d["sep_arcsec"] = sep_arcsec(t["ra"], t["dec"], d["ra"], d["dec"])
        plx, pmra, pmdec = t["parallax"], t["pmra"], t["pmdec"]
        dplx = np.abs(d["parallax"] - plx)
        dpm = np.hypot(d["pmra"] - pmra, d["pmdec"] - pmdec)
        pm_tot = math.hypot(pmra, pmdec)
        cm = d[(dplx < 3 * np.hypot(d["parallax_error"].fillna(1), 0.05) + 0.2) & (dpm < 0.2 * pm_tot + 2)]
        out["n_sources_60"] = int(len(d))
        out["comoving"] = [{"source_id": str(o["source_id"]), "sep_arcsec": round(float(o["sep_arcsec"]), 2),
                            "sep_au": round(float(o["sep_arcsec"]) * 1000.0 / plx, 0),
                            "g": _f(o["phot_g_mean_mag"]), "parallax": _f(o["parallax"]),
                            "dpm_masyr": _f(math.hypot(o["pmra"] - pmra, o["pmdec"] - pmdec))}
                           for _, o in cm.iterrows()]
    return out


# ---------------------------------------------------------------------------
# 5. Confusion: cirrus + PSF widths
# ---------------------------------------------------------------------------
def dust_block(t: dict) -> dict:
    out = {"status": "UNTESTED"}
    url = f"https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr={t['ra']:.6f}+{t['dec']:.6f}+equ+j2000"
    code, content, err = http_get(url, timeout=120)
    out["http"] = code
    if code != 200:
        out["error"] = err[:200]
        return out
    txt = content.decode(errors="replace")
    def grab(section, tag):
        m = re.search(rf"<{section}>.*?<{tag}>\s*([-\d.eE+]+)", txt, re.S)
        return float(m.group(1)) if m else None
    out["ebv_sfd_mean"] = grab("result", "meanValueSFD")
    out["ebv_sfd_ref"] = grab("result", "refPixelValueSFD")
    # the 100 um block is the one whose description mentions 100 Micron
    blocks = re.findall(r"<result>(.*?)</result>", txt, re.S)
    for bl in blocks:
        if "100 Micron" in bl or "100 micron" in bl:
            m = re.search(r"<refPixelValue>\s*([-\d.eE+]+)", bl)
            s = re.search(r"<meanValue>\s*([-\d.eE+]+)", bl)
            sd = re.search(r"<std>\s*([-\d.eE+]+)", bl)
            out["i100_mjy_sr_ref"] = float(m.group(1)) if m else None
            out["i100_mjy_sr_mean"] = float(s.group(1)) if s else None
            out["i100_mjy_sr_std"] = float(sd.group(1)) if sd else None
    out["status"] = "TESTED"
    return out


def _gauss_fit(img, x0, y0, half=7):
    from scipy.optimize import least_squares  # noqa: PLC0415
    ny, nx = img.shape
    xi, yi = int(round(x0)), int(round(y0))
    if xi - half < 0 or yi - half < 0 or xi + half >= nx or yi + half >= ny:
        return None
    sub = img[yi - half:yi + half + 1, xi - half:xi + half + 1]
    yy, xx = np.mgrid[yi - half:yi + half + 1, xi - half:xi + half + 1]
    ok = np.isfinite(sub)
    if ok.sum() < 30:
        return None
    bg0 = float(np.nanmedian(sub))
    p0 = [float(np.nanmax(sub) - bg0), x0, y0, 1.2, bg0]

    def res(p):
        a, xc, yc, s, bg = p
        return (a * np.exp(-((xx - xc) ** 2 + (yy - yc) ** 2) / (2 * s * s)) + bg - sub)[ok]
    try:
        r = least_squares(res, p0, bounds=([0, x0 - 3, y0 - 3, 0.3, -np.inf],
                                           [np.inf, x0 + 3, y0 + 3, 8.0, np.inf]))
    except Exception:                                     # noqa: BLE001
        return None
    a, xc, yc, s, bg = r.x
    return {"amp": float(a), "x": float(xc), "y": float(yc), "fwhm_px": float(2.3548 * s)}


def psf_block(t: dict) -> dict:
    out = {"status": "UNTESTED"}
    from astropy.wcs import WCS  # noqa: PLC0415
    ra, dec = propagate(t["ra"], t["dec"], t["pmra"], t["pmdec"], ALLWISE_EPOCH - GAIA_EPOCH)
    imgs, info = fetch_unwise(ra, dec, 251)
    out["info"] = {k: v for k, v in info.items() if k != "members"}
    if 1 not in imgs:
        return out
    res = {}
    for band in (1, 2, 3, 4):
        if band not in imgs:
            continue
        img, hdr = imgs[band]
        w = WCS(hdr)
        x0, y0 = (float(v) for v in w.all_world2pix(ra, dec, 0))
        half = 9 if band == 4 else 6
        tg = _gauss_fit(img, x0, y0, half=half)
        field = []
        for _, px, py in find_peaks(img, nsig=25.0 if band < 3 else 12.0, border=half + 2):
            if (px - x0) ** 2 + (py - y0) ** 2 < 15 ** 2:
                continue
            f = _gauss_fit(img, px, py, half=half)
            if f and 0.5 < f["fwhm_px"] < 7.5:
                field.append(f["fwhm_px"])
        fw = np.array(field)
        res[f"w{band}"] = {
            "target_fwhm_arcsec": tg["fwhm_px"] * 2.75 if tg else None,
            "field_median_fwhm_arcsec": float(np.median(fw) * 2.75) if fw.size else None,
            "field_mad_fwhm_arcsec": float(1.4826 * np.median(np.abs(fw - np.median(fw))) * 2.75)
            if fw.size else None,
            "n_field": int(fw.size)}
        if tg and fw.size >= 3:
            mad = 1.4826 * np.median(np.abs(fw - np.median(fw))) or 0.05
            res[f"w{band}"]["target_minus_field_in_mad"] = float((tg["fwhm_px"] - np.median(fw)) / mad)
    out["result"] = res
    out["status"] = "TESTED" if "w3" in res else "UNTESTED"
    return out


# ---------------------------------------------------------------------------
# 6. Literature
# ---------------------------------------------------------------------------
LIT_KEYWORDS = ("debris disk", "debris disc", "infrared excess", "warm dust",
                "extreme debris", "exozodi", "mid-infrared excess", "circumstellar dust")


def literature_block(t: dict, tap=tap_query) -> dict:
    out = {"status": "UNTESTED"}
    ra, dec = propagate(t["ra"], t["dec"], t["pmra"], t["pmdec"], 2000.0 - GAIA_EPOCH)
    r = tap(SIMBAD_TAP, "SELECT b.main_id, b.otype, b.ra, b.dec FROM basic AS b WHERE "
            f"{_cone('b.ra', 'b.dec', ra, dec, 60 / 3600)}", f"simbad60:{t['source_id']}")
    out["simbad_60_route"] = r.ledger()
    if r.data is not None and len(r.data):
        d = r.data.copy()
        d["sep"] = sep_arcsec(ra, dec, d["ra"], d["dec"])
        out["simbad_60"] = [{"main_id": str(o["main_id"]), "otype": str(o["otype"]),
                             "sep_arcsec": round(float(o["sep"]), 1)} for _, o in d.sort_values("sep").iterrows()]
    try:
        from astropy import units as u  # noqa: PLC0415
        from astropy.coordinates import SkyCoord  # noqa: PLC0415
        from astroquery.vizier import Vizier  # noqa: PLC0415
        cats = {}
        for kw in LIT_KEYWORDS:
            try:
                found = Vizier.find_catalogs(kw)
                for k, v in found.items():
                    cats[k] = str(getattr(v, "description", ""))[:160]
            except Exception as exc:                      # noqa: BLE001
                out.setdefault("find_errors", []).append(f"{kw}: {exc!r}"[:160])
        out["n_keyword_catalogues"] = len(cats)
        hits = []
        keys = sorted(cats)
        pos = SkyCoord(ra, dec, unit="deg")
        for i in range(0, len(keys), 40):
            chunk = keys[i:i + 40]
            try:
                tl = Vizier(columns=["+_r"], row_limit=5, timeout=240).query_region(
                    pos, radius=10 * u.arcsec, catalog=chunk)
                for k in tl.keys():
                    base = k
                    hits.append({"table": k, "description": cats.get(base.rsplit("/", 1)[0], cats.get(base, "")),
                                 "n": int(len(tl[k]))})
            except Exception as exc:                      # noqa: BLE001
                out.setdefault("query_errors", []).append(repr(exc)[:160])
        out["keyword_catalogue_hits_10arcsec"] = hits
        out["status"] = "TESTED"
    except Exception as exc:                              # noqa: BLE001
        out["error"] = repr(exc)[:200]
    return out


# ---------------------------------------------------------------------------
# 7. Occurrence (offline)
# ---------------------------------------------------------------------------
def occurrence_block(out_dir: Path, lead: dict) -> dict:
    summ = json.loads((out_dir / "summary.json").read_text())
    scr = summ.get("screen", {})
    cov = scr.get("coverage", {})
    sl = pd.read_csv(out_dir / "shortlist.csv", low_memory=False)
    cand = pd.read_csv(out_dir / "candidates.csv")
    n_gaia = int(cov.get("n_parent_gaia_only") or 0)
    # detectability of a 262 K disc with f = 1e-2 scaled from the lead itself
    f_lead, w4x_lead, w1_lead = float(lead["f_ir"]), float(lead["W4_excess_jy"]), float(lead["w1mpro"])
    w4_5sig_jy = float(mag_to_jy(8.0, "W4"))          # typical AllWISE W4 5-sigma
    w1_lim = w1_lead + 2.5 * math.log10((w4x_lead * (1e-2 / f_lead)) / w4_5sig_jy)
    in_cell = cand[(cand["t_bb_k"] >= 250) & (cand["t_bb_k"] <= 350)]
    big = in_cell[in_cell["f_ir"] > 5e-3]
    return {
        "n_parent_gaia_only_fgk_dwarfs": n_gaia,
        "n_parent_with_w3w4_5sigma": int(summ.get("funnel", {}).get("n_parent") or 0),
        "w1_limit_for_f1e-2_262K_at_W4_5sigma": w1_lim,
        "note_w1_limit": f"a G~5700 K dwarf has G - W1 ~ 1.6, so W1 < {w1_lim:.1f} is G < "
                         f"~{w1_lim + 1.6:.1f}: compare with the G < 13.5 parent",
        "n_in_cell_f_gt_5e-3": int(len(big)),
        "in_cell_f_gt_5e-3_by_class": big["cradle_class"].value_counts().to_dict(),
        "in_cell_f_gt_5e-3_by_age_class": big["age_class"].value_counts().to_dict(),
        "n_shortlist_f_gt_1e-2_any_T": int((sl["f_ir"] > 1e-2).sum()),
        "known_old_edds_in_literature": ["BD+20 307 (~1 Gyr, 400-450 K)", "TYC 4479-3-1 (5+/-2 Gyr, ~400 K)"],
        "bd20307_in_this_parent": True,
    }


# ---------------------------------------------------------------------------
def _runner(res: dict, rep: dict, out_dir: Path):
    """A block runner bound to one target's result dict; checkpoints after every block."""
    def run(name, fn, *a, timeout=900.0):
        t1 = _time.monotonic()
        try:
            res[name] = call_with_timeout(fn, timeout, *a)
        except Exception as exc:                          # noqa: BLE001
            res[name] = {"status": "UNTESTED", "error": repr(exc)[:300]}
        if not isinstance(res[name], dict):
            res[name] = {"status": "UNTESTED", "error": "block returned no dict"}
        res[name]["elapsed_s"] = round(_time.monotonic() - t1, 1)
        print(f"[leadvet]   {name}: {res[name].get('status')} ({res[name]['elapsed_s']} s)", flush=True)
        (out_dir / "leadvet_partial.json").write_text(json.dumps(rep, indent=1, default=_json_default))
    return run


def run_leadvet(out_dir: str | Path = "results/cradle", leads=LEADS) -> dict:
    out_dir = Path(out_dir)
    sl = pd.read_csv(out_dir / "shortlist.csv", low_memory=False)
    sl["source_id"] = sl["source_id"].astype(str)
    deep = {}
    dpath = out_dir / "deepvet.json"
    if dpath.exists():
        deep = {x["source_id"]: x for x in json.loads(dpath.read_text()).get("targets", [])}
    rep = {"stage": "leadvet", "generated_utc": _now(), "deepvet_generated_utc":
           json.loads(dpath.read_text()).get("generated_utc") if dpath.exists() else None, "targets": {}}
    t0 = _time.monotonic()
    for sid in leads:
        row = sl[sl["source_id"] == sid]
        if not len(row):
            rep["targets"][sid] = {"status": "NOT_IN_SHORTLIST"}
            continue
        t = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.iloc[0].to_dict().items()}
        t["source_id"] = sid
        res: dict = {}
        rep["targets"][sid] = res
        print(f"[leadvet] {sid}", flush=True)
        run = _runner(res, rep, out_dir)

        run("gaia", gaia_block, t)
        run("vizier", vizier_rows, t, timeout=1500.0)
        tic = None
        tr = _row(res["vizier"], "IV/39/tic82")
        if tr:
            tic = str(tr.get("TIC") or tr.get("ID") or "") or None
        run("neowise", neowise_block, t, timeout=1500.0)
        run("sed", sed_block, t, res["vizier"], deep.get(sid, {}))
        run("desi", desi_block, t)
        run("tess", tess_block, t, tic, timeout=1500.0)
        run("ztf", ztf_block, t)
        run("kinematics", kinematics_block, t)
        run("dust", dust_block, t)
        run("psf", psf_block, t, timeout=1500.0)
        run("literature", literature_block, t, timeout=1500.0)
    # a DESI comparison: the same core-over-model ratios on another G dwarf of the
    # shortlist with a DESI spectrum (1276273278883611264, Teff 5860 K, FLAME
    # 10.9 Gyr), so a template mismatch at Ca II K is not read as activity
    comp = {}
    for csid in ("1276273278883611264",):
        crow = sl[sl["source_id"] == csid]
        if len(crow):
            ct = {k: (v.item() if hasattr(v, "item") else v) for k, v in crow.iloc[0].to_dict().items()}
            try:
                comp[csid] = call_with_timeout(desi_block, 600.0, ct)
            except Exception as exc:                      # noqa: BLE001
                comp[csid] = {"status": "UNTESTED", "error": repr(exc)[:200]}
    rep["desi_comparison"] = comp
    lead = sl[sl["source_id"] == LEADS[0]].iloc[0].to_dict()
    rep["occurrence"] = occurrence_block(out_dir, lead)
    rep["elapsed_s"] = round(_time.monotonic() - t0, 1)
    (out_dir / "leadvet.json").write_text(json.dumps(rep, indent=1, default=_json_default))
    print(json.dumps(rep["occurrence"], indent=1, default=_json_default))
    return rep


def main(argv=None) -> int:
    import argparse  # noqa: PLC0415
    p = argparse.ArgumentParser(prog="seti.cradle.leadvet")
    p.add_argument("--out-dir", default="results/cradle")
    p.add_argument("--leads", default=",".join(LEADS))
    a = p.parse_args(argv)
    run_leadvet(a.out_dir, tuple(a.leads.split(",")))
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
