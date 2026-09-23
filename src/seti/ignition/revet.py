"""Re-vet of a finished IGNITION sweep: is each candidate a star, or the survey?

Run 35740159635 (tiles mode, 650 tiles, 169,749 stars screened) returned nine
``IGNITION_CANDIDATE`` stars.  Read off ``candidates.csv`` they share a
pattern that the screen was never built to see:

* every rise is **small** (0.026--0.083 mag over ten years) -- the same size as
  the survey drift the screen subtracted (+0.02 W1, +0.06 W2 mag per decade);
* every rise is **grey** (``W2 rise - W1 rise`` between -0.016 and +0.012 mag),
  where warm dust on a K dwarf adds more to W2 than to W1;
* in several stars the **raw** W2 slope is not significant at all
  (6383068554366896000: 0.16 sigma raw, 5.8 sigma after the correction) -- the
  rise is *made* by the ensemble correction;
* the candidates sit at the **bright end** of the sample (W1 9.4--10.3, one at
  11.2), and the ensemble offsets are medians dominated by fainter stars.

The ensemble correction (``seti.ignition.ensemble``) assumes the NEOWISE drift
is the same at every brightness.  If bright stars drift *less* than the
median star, subtracting the median drift makes every bright star "rise" by
the difference, smoothly, monotonically and nearly grey -- precisely the
signature the screen was asked to find.  This module tests that directly.

(Outcome on run 35740159635, run 35860165284: the drift IS brightness
dependent -- W1 0.24 -> 1.45 mmag/yr and W2 1.7 -> 3.5 mmag/yr from W1 8-9 to
11.5-12 -- but at the candidates' brightness the over-correction is only
~0.2-0.5 mmag/yr, an order of magnitude below their 2-11 mmag/yr slopes.
The hypothesis explains none of them outright; the candidates are 6-29 MAD
outliers of their own brightness/|beta| population.  See docs/ignition.md 7.4.)

1. ``drift_by_magnitude`` -- the raw and corrected slope distributions of the
   whole screened population in W1-magnitude bins (and ecliptic-latitude
   bands).  A drift that depends on brightness shows up as a trend of the
   median raw slope with magnitude, and as a pile-up of corrected "risers" in
   the bins where the median correction overshoots.
2. ``stratified_offsets`` / ``rescreen`` -- the same ensemble correction, but
   computed per (magnitude bin x |beta| band) stratum, pooled over every shard,
   and the full rise test re-run.  A star that still passes when it is
   compared with stars of its own brightness and sky cadence is a star; one
   that does not was the survey.
3. ``matched_percentile`` -- for each candidate, the fraction of its matched
   neighbours (same brightness, same |beta| band) whose corrected slope is at
   least as large in both bands.
4. ``archive_checks`` (runner only) -- Gaia DR3 neighbours within 30" with
   proper motions propagated over 2014--2024 (a neighbour drifting into the
   6" beam), Gaia DR3 variability / QSO-candidate / astrophysical-parameter
   (activity, H-alpha) rows, SIMBAD and VSX within 10"/30", Milliquas within
   10", AllWISE neighbours within 20", and ZTF light curves where the sky
   allows.  Every query is time-boxed; a failure is recorded as a failure and
   never as a pass.

Nothing here changes a threshold of the rise test.  It changes what the test
is compared with.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .acquire import ecliptic_latitude_deg, epochs_to_series
from .ensemble import BETA_EDGES, MAG_EDGES, stratified_offsets
from .rise import assess_series
from .vet import _bnu_ratio, dust_colour_test  # noqa: F401  (re-exported)

GAIA_EPOCH = 2016.0


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ---------------------------------------------------------------------------
# population
# ---------------------------------------------------------------------------
def load_population(out: Path, n_shards: int) -> pd.DataFrame:
    frames = []
    for i in range(n_shards):
        p = out / f"stars_s{i}of{n_shards}.csv"
        if p.exists():
            d = pd.read_csv(p, dtype={"source_id": str}, low_memory=False)
            d["shard_tag"] = f"s{i}of{n_shards}"
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates("source_id")
    df["abs_beta"] = np.abs(ecliptic_latitude_deg(_num(df["ra"]), _num(df["dec"])))
    return df


def _mad_sigma(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def drift_by_magnitude(stars: pd.DataFrame, mag_edges=MAG_EDGES,
                       key: str = "w1_median") -> list[dict]:
    """Per magnitude bin: n, median raw and corrected slope per band, risers."""
    rows = []
    m = _num(stars[key])
    for lo, hi in zip(mag_edges[:-1], mag_edges[1:], strict=False):
        g = stars[(m >= lo) & (m < hi)]
        rec = {"mag_lo": lo, "mag_hi": hi, "n": int(len(g))}
        if len(g):
            for b in ("w1", "w2"):
                raw = _num(g.get(f"{b}_slope_raw_mag_yr"))
                cor = _num(g.get(f"{b}_slope_mag_yr"))
                rec[f"{b}_raw_slope_median_mmag_yr"] = round(1e3 * float(np.nanmedian(raw)), 3)
                rec[f"{b}_raw_slope_mad_mmag_yr"] = round(1e3 * _mad_sigma(raw), 3)
                rec[f"{b}_corr_slope_median_mmag_yr"] = round(1e3 * float(np.nanmedian(cor)), 3)
            s1 = _num(g.get("w1_slope_sigma"))
            s2 = _num(g.get("w2_slope_sigma"))
            rec["n_corr_two_band_rising_5sig"] = int(((s1 >= 5) & (s2 >= 5)).sum())
            rec["n_candidates"] = int((g.get("screen_verdict") == "IGNITION_CANDIDATE").sum())
            rec["frac_corr_two_band_rising_5sig"] = round(rec["n_corr_two_band_rising_5sig"]
                                                          / len(g), 6)
        rows.append(rec)
    return rows


def drift_by_beta(stars: pd.DataFrame, beta_edges=BETA_EDGES) -> list[dict]:
    rows = []
    for lo, hi in zip(beta_edges[:-1], beta_edges[1:], strict=False):
        g = stars[(stars["abs_beta"] >= lo) & (stars["abs_beta"] < hi)]
        rec = {"abs_beta_lo": lo, "abs_beta_hi": hi, "n": int(len(g))}
        if len(g):
            for b in ("w1", "w2"):
                rec[f"{b}_raw_slope_median_mmag_yr"] = round(
                    1e3 * float(np.nanmedian(_num(g.get(f"{b}_slope_raw_mag_yr")))), 3)
                rec[f"{b}_corr_slope_median_mmag_yr"] = round(
                    1e3 * float(np.nanmedian(_num(g.get(f"{b}_slope_mag_yr")))), 3)
            rec["n_candidates"] = int((g.get("screen_verdict") == "IGNITION_CANDIDATE").sum())
        rows.append(rec)
    return rows


def matched_percentile(stars: pd.DataFrame, cand: pd.Series, dmag: float = 0.25) -> dict:
    """Among stars of the candidate's brightness and |beta| band, how many rise as much?"""
    m = _num(stars["w1_median"])
    band = np.digitize([float(cand["abs_beta"])], BETA_EDGES)[0]
    sb = np.digitize(stars["abs_beta"].to_numpy(float), BETA_EDGES)
    sel = ((m - float(cand["w1_median"])).abs() < dmag) & (sb == band) & \
          (stars["source_id"] != cand["source_id"])
    g = stars[sel]
    out = {"n_matched": int(len(g))}
    if not len(g):
        return out
    for b in ("w1", "w2"):
        raw = _num(g[f"{b}_slope_raw_mag_yr"])
        out[f"{b}_matched_raw_slope_median_mmag_yr"] = round(1e3 * float(np.nanmedian(raw)), 3)
        out[f"{b}_matched_raw_slope_mad_mmag_yr"] = round(1e3 * _mad_sigma(raw), 3)
        c_raw = float(cand.get(f"{b}_slope_raw_mag_yr", np.nan))
        out[f"{b}_raw_slope_minus_matched_mmag_yr"] = round(
            1e3 * (c_raw - float(np.nanmedian(raw))), 3)
        mad = _mad_sigma(raw)
        out[f"{b}_raw_slope_z_vs_matched"] = (round((float(np.nanmedian(raw)) - c_raw) / mad, 2)
                                              if mad and np.isfinite(mad) and mad > 0
                                              else None)
    s1 = _num(g["w1_slope_sigma"])
    s2 = _num(g["w2_slope_sigma"])
    c1 = float(cand["w1_slope_sigma"])
    c2 = float(cand["w2_slope_sigma"])
    out["frac_matched_both_bands_ge_candidate_sigma"] = round(
        float(((s1 >= c1) & (s2 >= c2)).mean()), 6)
    out["frac_matched_both_bands_ge_5sig"] = round(float(((s1 >= 5) & (s2 >= 5)).mean()), 6)
    return out


# ---------------------------------------------------------------------------
# stratified ensemble + re-screen
# ---------------------------------------------------------------------------
def load_epochs(out: Path, n_shards: int) -> pd.DataFrame:
    frames = []
    for i in range(n_shards):
        p = out / f"epochs_s{i}of{n_shards}.csv"
        if p.exists():
            d = pd.read_csv(p, dtype={"source_id": str},
                            usecols=lambda c: c in ("source_id", "band", "t_yr", "mag", "err"))
            frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["source_id", "band", "t_yr", "mag", "err"])
    ep = pd.concat(frames, ignore_index=True)
    ep["t_yr"] = _num(ep["t_yr"])
    ep["mag"] = _num(ep["mag"])
    ep["err"] = _num(ep["err"])
    ep = ep.dropna(subset=["t_yr", "mag"])
    # a resumed shard's CSV is append-only: one star's epochs must appear once
    return ep.drop_duplicates(["source_id", "band", "t_yr"])


def _screen_one(args):
    sid, g_records, rc = args
    g = pd.DataFrame.from_records(g_records)
    series = epochs_to_series(g)
    per_band, verdict = assess_series(series, rc)
    rec = {"source_id": sid, "verdict": verdict.verdict, "is_candidate": bool(verdict.is_candidate)}
    for b in ("W1", "W2"):
        if b in per_band:
            pb = per_band[b]
            rec[f"{b.lower()}_slope_mag_yr"] = pb.slope_mag_yr
            rec[f"{b.lower()}_slope_sigma"] = pb.slope_sigma
            rec[f"{b.lower()}_rise_mag"] = pb.rise_mag
            rec[f"{b.lower()}_reasons"] = ";".join(pb.reasons)
    return rec


def _quick_slopes(ep: pd.DataFrame, err_floor: float) -> pd.DataFrame:
    """Vectorised weighted linear slope and its sigma per (star, band)."""
    d = ep.copy()
    d["w"] = 1.0 / np.maximum(d["err"].fillna(err_floor).to_numpy(float), err_floor) ** 2
    g = d.groupby(["source_id", "band"])
    sw = g["w"].transform("sum")
    tm = (d["w"] * d["t_yr"]).groupby([d["source_id"], d["band"]]).transform("sum") / sw
    mm = (d["w"] * d["mag"]).groupby([d["source_id"], d["band"]]).transform("sum") / sw
    d["dt"] = d["t_yr"] - tm
    d["sxy"] = d["w"] * d["dt"] * (d["mag"] - mm)
    d["sxx"] = d["w"] * d["dt"] ** 2
    a = d.groupby(["source_id", "band"])[["sxy", "sxx"]].sum()
    a["slope"] = a["sxy"] / a["sxx"]
    a["sigma"] = -a["slope"] * np.sqrt(a["sxx"])
    return a[["slope", "sigma"]].unstack("band")


def rescreen(ep_corr: pd.DataFrame, rc: dict, *, prefilter_sigma: float = 3.0,
             workers: int = 4, force_ids=()) -> tuple[pd.DataFrame, dict]:
    """The full rise test on the stratified-corrected epochs.

    The full test (ramp + scan fit, Kendall tau, step-decay BIC) is run on every
    star whose bare weighted linear slope is >= ``prefilter_sigma`` in both
    bands (the candidate rule needs >= 5 sigma from the ramp fit in both, and
    the bare slope and the ramp slope agree to well inside 2 sigma for a
    ramp), plus ``force_ids`` (the original candidates, always re-tested).
    """
    qs = _quick_slopes(ep_corr, float(rc.get("err_floor_mag", 0.005)))
    s1 = qs[("sigma", "W1")] if ("sigma", "W1") in qs.columns else pd.Series(dtype=float)
    s2 = qs[("sigma", "W2")] if ("sigma", "W2") in qs.columns else pd.Series(dtype=float)
    pre = set(qs.index[(s1 >= prefilter_sigma) & (s2 >= prefilter_sigma)].astype(str))
    ids = sorted(pre | {str(x) for x in force_ids})
    sub = ep_corr[ep_corr["source_id"].isin(ids)]
    jobs = [(sid, g[["band", "t_yr", "mag", "err"]].to_dict("records"), rc)
            for sid, g in sub.groupby("source_id")]
    rows = []
    if workers and workers > 1 and len(jobs) > 50:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_screen_one, jobs, chunksize=64))
    else:
        rows = [_screen_one(j) for j in jobs]
    df = pd.DataFrame(rows)
    rep = {"n_stars_quick_slopes": int(len(qs)),
           "n_prefilter_two_band_ge_sigma": int(len(pre)),
           "prefilter_sigma": prefilter_sigma,
           "n_full_tested": int(len(df)),
           "n_candidates": int(df["is_candidate"].sum()) if len(df) else 0,
           "verdicts": (df["verdict"].value_counts().to_dict() if len(df) else {})}
    return df, rep


# ---------------------------------------------------------------------------
# archive checks (runner only)
# ---------------------------------------------------------------------------
def _timed(fn, timeout_s: float = 120.0):
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FTE

    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=timeout_s), None
    except FTE:
        return None, f"timeout {timeout_s:.0f}s"
    except Exception as exc:                          # noqa: BLE001
        return None, repr(exc)[:300]
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _tap(url: str, adql: str) -> pd.DataFrame:
    import pyvo

    return pyvo.dal.TAPService(url).run_sync(adql).to_table().to_pandas()


def _sep_arcsec(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
    s = (np.sin((dec2 - dec1) / 2) ** 2
         + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2)
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(s, 0, 1)))) * 3600.0


def _propagate(ra, dec, pmra, pmdec, epoch):
    dt = epoch - GAIA_EPOCH
    pmra = np.nan_to_num(np.asarray(pmra, float))
    pmdec = np.nan_to_num(np.asarray(pmdec, float))
    dec2 = np.asarray(dec, float) + pmdec * dt / 3.6e6
    ra2 = np.asarray(ra, float) + pmra * dt / 3.6e6 / np.cos(np.radians(dec))
    return ra2, dec2


def gaia_neighbours(c: pd.Series, radius_arcsec: float = 30.0) -> dict:
    ra, dec = float(c["ra"]), float(c["dec"])
    q = ("SELECT g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.parallax, g.phot_g_mean_mag, "
         "g.bp_rp, g.phot_variable_flag, g.ruwe, g.non_single_star FROM gaiadr3.gaia_source AS g "
         f"WHERE 1=CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {ra}, {dec}, "
         f"{radius_arcsec / 3600.0}))")
    df, err = _timed(lambda: _tap("https://gea.esac.esa.int/tap-server/tap", q), 180)
    if df is None:
        return {"status": "FAILED", "error": err}
    df.columns = [x.lower() for x in df.columns]
    df["source_id"] = df["source_id"].astype(str)
    tgt = df[df["source_id"] == str(c["source_id"])]
    if not len(tgt):
        return {"status": "TARGET_NOT_FOUND", "n": int(len(df))}
    t = tgt.iloc[0]
    others = df[df["source_id"] != str(c["source_id"])].copy()
    gt = float(t["phot_g_mean_mag"])
    recs = []
    for _, o in others.iterrows():
        seps = {}
        for ep in (2010.5, 2014.0, 2019.0, 2024.5):
            ra_t, de_t = _propagate(t["ra"], t["dec"], t["pmra"], t["pmdec"], ep)
            ra_o, de_o = _propagate(o["ra"], o["dec"], o["pmra"], o["pmdec"], ep)
            seps[ep] = float(_sep_arcsec(ra_t, de_t, ra_o, de_o))
        recs.append({"source_id": o["source_id"], "g": float(o["phot_g_mean_mag"])
                     if np.isfinite(o["phot_g_mean_mag"]) else None,
                     "dg": (float(o["phot_g_mean_mag"]) - gt)
                     if np.isfinite(o["phot_g_mean_mag"]) else None,
                     "bp_rp": float(o["bp_rp"]) if np.isfinite(o["bp_rp"]) else None,
                     "pm_total": float(np.hypot(np.nan_to_num(o["pmra"]),
                                                np.nan_to_num(o["pmdec"]))),
                     "sep_2010": round(seps[2010.5], 2), "sep_2014": round(seps[2014.0], 2),
                     "sep_2024": round(seps[2024.5], 2),
                     "approach_2014_2024": round(seps[2014.0] - seps[2024.5], 2)})
    recs.sort(key=lambda r: r["sep_2024"])
    inside = [r for r in recs if min(r["sep_2014"], r["sep_2024"]) < 9.0]
    # A neighbour dG mag fainter carries ~10^(-0.4 dG) of the target's optical
    # flux; in W1 a red neighbour can be relatively brighter, so the kill is
    # loose (dG < 6) and is judged together with its approach.
    blend_risk = [r for r in inside if r["dg"] is None or r["dg"] < 6.0]
    approaching = [r for r in blend_risk if r["approach_2014_2024"] > 0.5]
    return {"status": "OK", "n_neighbours_30as": int(len(recs)),
            "target_ruwe": float(t["ruwe"]), "target_non_single_star": int(t["non_single_star"]),
            "target_pm_total": float(np.hypot(t["pmra"], t["pmdec"])),
            "neighbours_within_9as": inside, "n_blend_risk": int(len(blend_risk)),
            "n_blend_approaching": int(len(approaching)), "nearest5": recs[:5]}


def gaia_extras(sids: list[str]) -> dict:
    ids = ",".join(str(s) for s in sids)
    out = {}
    for name, q in (
        ("vari_summary", f"SELECT * FROM gaiadr3.vari_summary WHERE source_id IN ({ids})"),
        ("qso_candidates", f"SELECT source_id FROM gaiadr3.qso_candidates WHERE source_id IN ({ids})"),
        ("astrophysical_parameters",
         "SELECT source_id, teff_gspphot, logg_gspphot, mh_gspphot, classprob_dsc_combmod_star, "
         "activityindex_espcs, activityindex_espcs_uncertainty, ew_espels_halpha, "
         "ew_espels_halpha_flag, flags_flame, age_flame, age_flame_lower, age_flame_upper "
         f"FROM gaiadr3.astrophysical_parameters WHERE source_id IN ({ids})"),
    ):
        df, err = _timed(lambda q=q: _tap("https://gea.esac.esa.int/tap-server/tap", q), 180)
        if df is None:
            out[name] = {"status": "FAILED", "error": err}
            continue
        df.columns = [x.lower() for x in df.columns]
        df["source_id"] = df["source_id"].astype(str)
        out[name] = {"status": "OK", "rows": json.loads(df.to_json(orient="records"))}
    return out


def simbad_cone(c: pd.Series, radius_arcsec: float = 10.0) -> dict:
    q = ("SELECT main_id, otype, sp_type, ra, dec FROM basic WHERE "
         f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {float(c['ra'])}, {float(c['dec'])}, "
         f"{radius_arcsec / 3600.0})) = 1")
    df, err = _timed(lambda: _tap("https://simbad.cds.unistra.fr/simbad/sim-tap", q), 180)
    if df is None:                                    # one retry: SIMBAD TAP is bursty
        df, err = _timed(lambda: _tap("https://simbad.cds.unistra.fr/simbad/sim-tap", q), 180)
    if df is None:
        return {"status": "FAILED", "error": err}
    return {"status": "OK", "rows": json.loads(df.astype(str).to_json(orient="records"))}


def _vizier_asu(source: str, c: pd.Series, radius_arcsec: float, out_cols: str) -> dict:
    import io

    import requests

    params = {"-source": source, "-c": f"{float(c['ra'])} {float(c['dec'])}",
              "-c.rs": str(radius_arcsec), "-out": out_cols, "-out.add": "_r",
              "-out.max": "50"}
    last = None
    for base in ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
                 "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv"):
        try:
            r = requests.get(base, params=params, timeout=60)
            r.raise_for_status()
            lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith("#")]
            if len(lines) < 3:
                return {"status": "OK", "rows": [], "endpoint": base}
            body = "\n".join([lines[0]] + lines[3:])   # drop units + dashes rows
            df = pd.read_csv(io.StringIO(body), sep="\t", dtype=str)
            return {"status": "OK", "rows": json.loads(df.to_json(orient="records")),
                    "endpoint": base}
        except Exception as exc:                      # noqa: BLE001
            last = repr(exc)[:300]
    return {"status": "FAILED", "error": last}


def allwise_neighbours(c: pd.Series, radius_arcsec: float = 20.0) -> dict:
    q = ("SELECT designation, ra, dec, w1mpro, w2mpro, w3mpro, ph_qual, cc_flags, ext_flg, nb, na "
         "FROM allwise_p3as_psd WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', "
         f"{float(c['ra'])}, {float(c['dec'])}, {radius_arcsec / 3600.0})) = 1")
    df, err = _timed(lambda: _tap("https://irsa.ipac.caltech.edu/TAP", q), 180)
    if df is None:
        return {"status": "FAILED", "error": err}
    df.columns = [x.lower() for x in df.columns]
    df["sep_arcsec"] = _sep_arcsec(float(c["ra"]), float(c["dec"]), df["ra"], df["dec"])
    df = df.sort_values("sep_arcsec")
    return {"status": "OK", "rows": json.loads(df.astype(str).to_json(orient="records"))}


def ztf_lightcurve(c: pd.Series, vet_conf: dict | None = None) -> dict:
    import io

    import requests

    from .vet import optical_flatness

    if float(c["dec"]) < -29.0:
        return {"status": "NOT_IN_FOOTPRINT"}
    url = ("https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?"
           f"POS=CIRCLE%20{float(c['ra'])}%20{float(c['dec'])}%200.0004&BAD_CATFLAGS_MASK=32768"
           "&FORMAT=csv")
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as exc:                          # noqa: BLE001
        return {"status": "FAILED", "error": repr(exc)[:300]}
    if not len(df):
        return {"status": "NO_ROWS"}
    out = {"status": "OK", "n_rows": int(len(df))}
    for flt, g in df.groupby("filtercode"):
        g = g[(g["catflags"] == 0)] if "catflags" in g else g
        # One ZTF object id per band: the one with the most points.
        if "oid" in g and len(g):
            g = g[g["oid"] == g["oid"].value_counts().idxmax()]
        t = 2000.0 + (g["mjd"].to_numpy(float) - 51544.5) / 365.25
        fl = optical_flatness(t, g["mag"].to_numpy(float), g["magerr"].to_numpy(float),
                              vet_conf)
        fl["median_mag"] = float(np.median(g["mag"])) if len(g) else None
        fl["t_span_yr"] = float(t.max() - t.min()) if len(t) else None
        fl["saturation_risk"] = bool(len(g) and np.median(g["mag"]) < 13.0)
        out[str(flt)] = fl
    return out


def _camera_slope(t, m, e, cam) -> dict:
    """mag = offset[camera] + slope * t, weighted, 5-sigma clipped once."""
    t = np.asarray(t, float)
    m = np.asarray(m, float)
    e = np.maximum(np.asarray(e, float), 0.005)
    cam = np.asarray(cam).astype(str)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e)
    t, m, e, cam = t[ok], m[ok], e[ok], cam[ok]
    res = {"n": int(t.size)}
    for _ in range(2):
        cams = sorted(set(cam))
        if t.size < 20 or not cams:
            res["status"] = "insufficient"
            return res
        X = np.column_stack([(cam == c).astype(float) for c in cams] + [t - 2019.0])
        w = 1.0 / e
        beta, *_ = np.linalg.lstsq(X * w[:, None], m * w, rcond=None)
        r = m - X @ beta
        chi2 = float(np.sum((r / e) ** 2) / max(t.size - X.shape[1], 1))
        keep = np.abs(r / e) < 5.0 * np.sqrt(max(chi2, 1.0))
        if keep.all():
            break
        t, m, e, cam = t[keep], m[keep], e[keep], cam[keep]
    cov = np.linalg.pinv((X * w[:, None]).T @ (X * w[:, None]))
    se = float(np.sqrt(cov[-1, -1]) * np.sqrt(max(chi2, 1.0)))
    res.update({"status": "OK", "n_used": int(t.size), "n_cameras": len(cams),
                "t_span_yr": round(float(t.max() - t.min()), 2),
                "slope_mmag_yr": round(1e3 * float(beta[-1]), 3),
                "slope_err_mmag_yr": round(1e3 * se, 3),
                "slope_sigma": round(float(beta[-1]) / se, 2) if se > 0 else None,
                "rms_mag": round(float(np.std(r)), 4), "chi2_red": round(chi2, 2),
                "median_mag": round(float(np.median(m)), 3)})
    return res


def asassn_lightcurve(c: pd.Series, radius_arcsec: float = 5.0) -> dict:
    """ASAS-SN Sky Patrol (V 2012-2018, g 2017-): a camera-offset linear trend per filter."""
    try:
        from pyasassn.client import SkyPatrolClient
    except Exception as exc:                          # noqa: BLE001
        return {"status": "CLIENT_MISSING", "error": repr(exc)[:200]}

    def _q():
        cl = SkyPatrolClient()
        return cl.cone_search(ra_deg=float(c["ra"]), dec_deg=float(c["dec"]),
                              radius=radius_arcsec / 3600.0, catalog="master_list",
                              download=True, threads=1)
    lcs, err = _timed(_q, 300)
    if lcs is None:
        return {"status": "FAILED", "error": err}
    d = getattr(lcs, "data", None)
    if d is None or not len(d):
        return {"status": "NO_ROWS"}
    d = d.copy()
    if "quality" in d:
        d = d[d["quality"].astype(str) == "G"]
    if "mag_err" in d:
        d = d[pd.to_numeric(d["mag_err"], errors="coerce") < 0.1]
    ids = d["asas_sn_id"].value_counts() if "asas_sn_id" in d else None
    if ids is not None and len(ids):
        d = d[d["asas_sn_id"] == ids.index[0]]
    out = {"status": "OK", "n_rows": int(len(d)), "n_ids": int(len(ids)) if ids is not None else None}
    jd = pd.to_numeric(d["jd"], errors="coerce").to_numpy(float)
    t = 2000.0 + (jd - 2451545.0) / 365.25
    for flt, g in d.assign(t=t).groupby("phot_filter"):
        out[str(flt)] = _camera_slope(g["t"], pd.to_numeric(g["mag"], errors="coerce"),
                                      pd.to_numeric(g["mag_err"], errors="coerce"),
                                      g["camera"] if "camera" in g else np.zeros(len(g)))
    return out


def gaia_variability_proxy(targets: pd.DataFrame, stars: pd.DataFrame, n_comp: int = 300,
                           seed: int = 20260923) -> dict:
    """Gaia DR3 G-band excess scatter of each target against its own G / BP-RP peers.

    A_G = sqrt(phot_g_n_obs) / phot_g_mean_flux_over_error is the fractional
    per-observation flux scatter over the DR3 window (2014.6-2017.4).  A star
    brightening by D mag across that window has an rms of ~D/sqrt(12) from the
    trend alone, so a star whose optical follows a 5-10 mmag/yr IR rise (or
    exceeds it, as spot/activity brightening does) sits above its peers.  The
    peers are screened stars of the same sample within 0.1 mag in G and 0.15
    in BP-RP; the result is the target's percentile among them.
    """
    rng = np.random.default_rng(seed)
    g_all = _num(stars["phot_g_mean_mag"])
    c_all = _num(stars["bp_rp"])
    ids_t = [str(x) for x in targets["source_id"]]
    comp: dict[str, list[str]] = {}
    pool: set[str] = set(ids_t)
    for _, t in targets.iterrows():
        sel = ((g_all - float(t["phot_g_mean_mag"])).abs() < 0.1) & \
              ((c_all - float(t["bp_rp"])).abs() < 0.15) & (stars["source_id"] != str(t["source_id"]))
        ids = stars.loc[sel, "source_id"].astype(str).to_numpy()
        if len(ids) > n_comp:
            ids = rng.choice(ids, n_comp, replace=False)
        comp[str(t["source_id"])] = list(ids)
        pool.update(ids)
    rows = []
    pool_l = sorted(pool)
    for i in range(0, len(pool_l), 400):
        chunk = ",".join(pool_l[i:i + 400])
        q = ("SELECT source_id, phot_g_mean_mag, phot_g_n_obs, phot_g_mean_flux_over_error, "
             "phot_bp_n_obs, phot_bp_mean_flux_over_error, phot_rp_n_obs, "
             "phot_rp_mean_flux_over_error FROM gaiadr3.gaia_source "
             f"WHERE source_id IN ({chunk})")
        df, err = _timed(lambda q=q: _tap("https://gea.esac.esa.int/tap-server/tap", q), 240)
        if df is None:
            return {"status": "FAILED", "error": err}
        df.columns = [x.lower() for x in df.columns]
        rows.append(df)
    d = pd.concat(rows, ignore_index=True)
    d["source_id"] = d["source_id"].astype(str)
    for b in ("g", "bp", "rp"):
        d[f"a_{b}"] = np.sqrt(_num(d[f"phot_{b}_n_obs"])) / _num(d[f"phot_{b}_mean_flux_over_error"])
    d = d.set_index("source_id")
    out = {"status": "OK", "note": "A = sqrt(n_obs)/flux_over_error, fractional per-obs scatter"}
    for sid in ids_t:
        if sid not in d.index:
            out[sid] = {"status": "TARGET_NOT_RETURNED"}
            continue
        peers = d.reindex([x for x in comp[sid] if x in d.index])
        rec = {"n_peers": int(len(peers))}
        for b in ("g", "bp", "rp"):
            a_t = float(d.at[sid, f"a_{b}"])
            a_p = peers[f"a_{b}"].dropna().to_numpy(float)
            rec[f"a_{b}_mmag"] = round(1085.7 * a_t, 2)
            if a_p.size:
                rec[f"a_{b}_peers_median_mmag"] = round(1085.7 * float(np.median(a_p)), 2)
                rec[f"a_{b}_percentile"] = round(100.0 * float(np.mean(a_p < a_t)), 1)
        out[sid] = rec
    return out


def archive_checks(cands: pd.DataFrame) -> dict:
    res = {}
    asassn_dead = None
    for _, c in cands.iterrows():
        sid = str(c["source_id"])
        if asassn_dead:
            asa = {"status": "SKIPPED", "error": f"service unreachable earlier: {asassn_dead}"}
        else:
            asa = asassn_lightcurve(c)
            if asa.get("status") in ("FAILED", "CLIENT_MISSING") and \
                    any(s in str(asa.get("error")) for s in ("ConnectTimeout", "Max retries",
                                                             "No module", "CLIENT")):
                asassn_dead = str(asa.get("error"))[:120]
        r = {"gaia_neighbours": gaia_neighbours(c),
             "simbad_10as": simbad_cone(c),
             "vsx_30as": _vizier_asu("B/vsx/vsx", c, 30, "Name,Type,max,min,Period"),
             "milliquas_10as": _vizier_asu("VII/294/catalog", c, 10, "Name,Type,z"),
             "allwise_20as": allwise_neighbours(c),
             "ztf": ztf_lightcurve(c),
             "asassn": asa}
        res[sid] = r
        print(f"[revet] archive {sid}: " + ", ".join(f"{k}={v.get('status')}"
                                                     for k, v in r.items()), flush=True)
    res["_gaia_extras"] = gaia_extras(list(cands["source_id"].astype(str)))
    return res


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------
def stage_revet(conf: dict, out: Path, n_shards: int, *, run_id: str = "",
                online: bool = True, workers: int = 4) -> dict:
    t0 = time.monotonic()
    rep: dict = {"stage": "revet", "generated_utc": _now(), "source_run_id": run_id,
                 "n_shards": n_shards}
    stars = load_population(out, n_shards)
    rep["n_stars_population"] = int(len(stars))
    if not len(stars):
        rep["verdict"] = "NO_POPULATION"
        return rep
    cands = stars[stars["screen_verdict"] == "IGNITION_CANDIDATE"].copy()
    rep["n_original_candidates"] = int(len(cands))
    rep["population_counts"] = stars["screen_verdict"].value_counts().to_dict()
    rep["drift_by_w1_mag"] = drift_by_magnitude(stars)
    rep["drift_by_abs_beta"] = drift_by_beta(stars)
    rep["population_w1_median_quantiles"] = {
        q: round(float(np.nanquantile(_num(stars["w1_median"]), q)), 3)
        for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95)}

    # stratified ensemble and the full re-screen
    ep = load_epochs(out, n_shards)
    rep["n_epoch_rows"] = int(len(ep))
    beta = stars.set_index("source_id")["abs_beta"]
    ep_corr, srep = stratified_offsets(ep, beta)
    del ep
    rep["stratified_ensemble"] = srep
    rc = conf["rise"]
    rs, rsrep = rescreen(ep_corr, rc, workers=workers, force_ids=cands["source_id"])
    rep["rescreen"] = rsrep
    rs.to_csv(out / "revet_rescreen.csv", index=False)
    # the raw (uncorrected) re-screen of the candidates alone, for the record
    raw = ep_corr[ep_corr["source_id"].isin(
        set(cands["source_id"]) | set(rs.loc[rs["is_candidate"], "source_id"].astype(str))
    )].copy()
    raw["mag"] = raw["mag_raw"]
    raw_rows = []
    for sid, g in raw.groupby("source_id"):
        _, v = assess_series(epochs_to_series(g), rc)
        raw_rows.append({"source_id": sid, "raw_verdict": v.verdict})
    raw_v = pd.DataFrame(raw_rows)

    new_ids = sorted(set(rs.loc[rs["is_candidate"], "source_id"].astype(str))
                     - set(cands["source_id"]))
    targets = pd.concat([cands.assign(original=True),
                         stars[stars["source_id"].isin(new_ids)].assign(original=False)],
                        ignore_index=True)
    rows = []
    for _, c in targets.iterrows():
        sid = str(c["source_id"])
        rec = {"source_id": sid, "original_candidate": bool(c["original"]),
               "ra": float(c["ra"]), "dec": float(c["dec"]),
               "abs_beta": round(float(c["abs_beta"]), 2),
               "g": float(c["phot_g_mean_mag"]), "bp_rp": float(c["bp_rp"]),
               "w1_median": float(c["w1_median"]), "w2_median": float(c["w2_median"]),
               "w1_rise_mag": float(c["w1_rise_mag"]), "w2_rise_mag": float(c["w2_rise_mag"]),
               "w2_minus_w1_rise": float(c["star_rise_w2_minus_w1_mag"]),
               "w1_slope_raw_mmag_yr": round(1e3 * float(c["w1_slope_raw_mag_yr"]), 3),
               "w1_slope_raw_sigma": float(c["w1_slope_raw_sigma"]),
               "w2_slope_raw_mmag_yr": round(1e3 * float(c["w2_slope_raw_mag_yr"]), 3),
               "w2_slope_raw_sigma": float(c["w2_slope_raw_sigma"]),
               "w1_slope_shardcorr_sigma": float(c["w1_slope_sigma"]),
               "w2_slope_shardcorr_sigma": float(c["w2_slope_sigma"])}
        rec.update(matched_percentile(stars, c))
        r = rs[rs["source_id"] == sid]
        if len(r):
            r = r.iloc[0]
            rec.update({"strat_verdict": r["verdict"],
                        "w1_slope_strat_mmag_yr": round(1e3 * float(r.get("w1_slope_mag_yr")), 3),
                        "w1_slope_strat_sigma": float(r.get("w1_slope_sigma")),
                        "w2_slope_strat_mmag_yr": round(1e3 * float(r.get("w2_slope_mag_yr")), 3),
                        "w2_slope_strat_sigma": float(r.get("w2_slope_sigma")),
                        "w1_strat_reasons": r.get("w1_reasons"),
                        "w2_strat_reasons": r.get("w2_reasons")})
            teff = float(c.get("teff_gspphot", np.nan)) if "teff_gspphot" in c else np.nan
            dc = dust_colour_test(r.get("w1_slope_mag_yr"), r.get("w1_slope_sigma"),
                                  r.get("w2_slope_mag_yr"), r.get("w2_slope_sigma"),
                                  teff if np.isfinite(teff) and teff > 2500 else 5000.0)
            rec.update({f"colour_{k}": v for k, v in dc.items()})
        v = raw_v[raw_v["source_id"] == sid]
        rec["raw_verdict"] = v.iloc[0]["raw_verdict"] if len(v) else None
        rows.append(rec)
    cdf = pd.DataFrame(rows)
    rep["candidates"] = json.loads(cdf.to_json(orient="records"))
    rep["n_survive_stratified"] = int((cdf.get("strat_verdict") == "IGNITION_CANDIDATE").sum()) \
        if len(cdf) else 0
    rep["new_candidates_stratified"] = new_ids
    if online and len(targets):
        rep["archive"] = archive_checks(targets)
        rep["gaia_variability_proxy"] = gaia_variability_proxy(targets, stars)
    rep["elapsed_s"] = round(time.monotonic() - t0, 1)
    cdf.to_csv(out / "revet_candidates.csv", index=False)
    (out / "revet.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"[revet] {len(cands)} original candidates; {rep['n_survive_stratified']} survive the "
          f"stratified ensemble; new under stratified: {len(rep['new_candidates_stratified'])}; "
          f"{rep['elapsed_s']} s")
    return rep


def main(argv=None) -> int:
    import argparse

    from .run import load_ignition_config

    ap = argparse.ArgumentParser(description="IGNITION re-vet of a finished sweep")
    ap.add_argument("--out", default="results/ignition")
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    conf = load_ignition_config()
    stage_revet(conf, Path(a.out), a.shards, run_id=a.run_id, online=not a.offline,
                workers=a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
