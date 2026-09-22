"""Target selection --- runner-only (VizieR, DASCH).

Two populations per field, in this order:

1. **Catalogued periodic variables** (AAVSO VSX through VizieR ``B/vsx/vsx``,
   GCVS as the fallback) bright enough for the plates and with a period the
   plates can resolve.  These carry the cessation test.
2. **Bright stars** (``stdmag`` below a limit in the DASCH ``apass`` reference
   catalogue, from ``querycat`` tiles across the field).  These carry the
   fade and rising-scatter tests.

Each field is first measured for plate density with ``queryexps`` at its
centre, and the pre/post Menzel-gap plate counts are recorded, so the run
summary says which fields the archive actually covers thickly.

Every query is logged with its row count or its error; an empty answer and a
failed query are different facts and are kept different.
"""

from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd

from ..knell.acquire import AcquisitionLog, fetch_gcvs_region, fetch_vsx_region
from .api import numeric, pick_column, querycat, queryexps
from .lightcurve import any_time_to_year
from .step import MENZEL_GAP_END, MENZEL_GAP_START
from .vet import is_lpv_type, is_periodic_type

REF_COLS_RA = ("ra_deg", "ra", "raj2000", "ra_icrs")
REF_COLS_DEC = ("dec_deg", "dec", "dej2000", "de_icrs")
REF_COLS_GBI = ("gsc_bin_index", "gscbinindex", "gsc_bin")
REF_COLS_REFNUM = ("ref_number", "refnumber", "ref_num", "refnum")
REF_COLS_MAG = ("stdmag", "std_mag", "mag", "v_mag", "b_mag", "vmag", "bmag")
REF_COLS_NDET = ("n_detections", "ndet", "ndets", "nobs", "n_obs", "num_detections",
                 "nsum", "n_sum")
REF_COLS_PMRA = ("pm_ra_masyr", "pmra", "pm_ra")
REF_COLS_PMDEC = ("pm_dec_masyr", "pmdec", "pm_dec")
REF_COLS_COLOR = ("color", "colour", "b_v", "bv")
# Verified against a live ``queryexps`` answer on the runner (run 35738717013):
#   series,platenum,scannum,mosnum,expnum,solnum,class,ra,dec,exptime,expdate,
#   epoch,wcssource,scandate,mosdate,centerdist,edgedist,limMagApass,
#   limMagAtlas,medianColortermApass,medianColortermAtlas,nMagd...
#
# ``epoch`` IS NOT THE OBSERVATION DATE.  An earlier revision assumed it was a
# decimal year and put it first because it is numeric; run 35748748365 then
# reported all six fields as ``1890..1992 -> 2000.0..2000.0``, every plate
# post-gap and none pre-gap, across 96,909 exposures.  It is the coordinate
# epoch, constant at 2000.0.  ``daschlab/exposures.py`` says as much by
# omission: its ``_COLTYPES`` parses ``expdate`` and leaves ``epoch``
# commented out.
#
# ``expdate`` is the observation timestamp, in DASCH's own dialect --
# hyphens in the time field as well as the date (``1899-07-04T12-34-56``) and
# occasionally a ``:60.0`` seconds field -- so only its leading ``YYYY-MM-DD``
# is parsed, which is four orders of magnitude finer than the two-year blocks
# it feeds.
EXP_DATE_COLS = ("expdate", "exp_date", "date_obs", "dateobs")
EXP_TIME_COLS = ("date_jd", "jd", "mjd", "hjd", "time", "date")
EXP_LIM_COLS = ("limmagapass", "limmagatlas", "limiting_mag", "limmag")


def dasch_dates_to_year(values) -> np.ndarray:
    """Decimal years from DASCH ``expdate`` strings; NaN where unparseable.

    Only the leading ``YYYY-MM-DD`` is read: DASCH writes the time field with
    hyphens too (``1899-07-04T12-34-56``) and sometimes a ``:60.0`` seconds
    value, and a day is already four orders of magnitude finer than the
    two-year blocks this feeds.
    """
    s = pd.Series(values, dtype="object").astype(str).str.strip().str.slice(0, 10)
    dt = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    out = np.full(len(s), np.nan)
    ok = dt.notna().to_numpy()
    if ok.any():
        d = dt[ok]
        year = d.dt.year.to_numpy(dtype=float)
        start = pd.to_datetime(d.dt.year.astype(str) + "-01-01")
        frac = ((d - start).dt.days.to_numpy(dtype=float)
                / np.where(d.dt.is_leap_year.to_numpy(), 366.0, 365.0))
        out[ok] = year + frac
    return out


def exposure_years(df: pd.DataFrame) -> tuple[np.ndarray, str | None]:
    """Decimal year per ``queryexps`` row, and the column it came from.

    ``expdate`` first (the observation timestamp), then any numeric JD/MJD
    column.  Never ``epoch``, which is the coordinate epoch and is 2000.0 for
    every plate --- see ``EXP_TIME_COLS``.
    """
    dcol = pick_column(df, EXP_DATE_COLS)
    if dcol is not None:
        yr = dasch_dates_to_year(df[dcol].to_numpy())
        if np.isfinite(yr).any():
            return yr, str(dcol)
    tcol = pick_column(df, EXP_TIME_COLS)
    if tcol is not None:
        yr = any_time_to_year(numeric(df, tcol))
        if np.isfinite(yr).any():
            return yr, str(tcol)
    return np.full(len(df), np.nan), None


def field_tag(ra: float, dec: float, radius_deg: float) -> str:
    return f"ra{ra:.2f}_dec{dec:+.2f}_r{radius_deg:.2f}"


EXPOSURE_KEY_COLS = ("series", "platenum", "mosnum", "expnum")


def exposure_table(df: pd.DataFrame) -> pd.DataFrame:
    """The exposure DURATION of every plate, keyed for a join onto a light curve.

    DR7 light curves do **not** carry an exposure time --- the served columns
    are fixed by ``_COLTYPES`` in ``daschlab/photometry.py`` and ``exptime`` is
    not among them --- while ``queryexps`` does, in minutes.  The channel needs
    it because the exposure smears a periodic signal by ``|sinc(f t_exp)|``,
    and Harvard exposure lengths are a property of the *plate series*, which is
    clustered in calendar time.  Unmodelled, an efficiency measured in the
    post-gap blocks of a star whose late plates are longer exposures overstates
    how well those plates could have seen the period --- which is the one
    number the cessation claim rests on.

    One ``queryexps`` per FIELD serves every star in it, so this costs no extra
    requests.  Returns a frame with ``series, platenum, mosnum, expnum,
    exptime_min`` and no duplicate keys.
    """
    if df is None or not len(df):
        return pd.DataFrame(columns=[*EXPOSURE_KEY_COLS, "exptime_min"])
    ecol = pick_column(df, ("exptime", "exposure_time", "exp_time"))
    scol = pick_column(df, ("series", "plate_series"))
    if ecol is None or scol is None:
        return pd.DataFrame(columns=[*EXPOSURE_KEY_COLS, "exptime_min"])
    out = pd.DataFrame({"series": df[scol].astype(str).str.strip().str.lower()})
    for key, cands in (("platenum", ("platenum", "plate_number")),
                       ("mosnum", ("mosnum", "mosaic_number")),
                       ("expnum", ("expnum", "exposure_number"))):
        c = pick_column(df, cands)
        out[key] = (pd.to_numeric(df[c], errors="coerce").astype("Int64") if c is not None
                    else pd.array([pd.NA] * len(df), dtype="Int64"))
    out["exptime_min"] = numeric(df, ecol)
    out = out[np.isfinite(out["exptime_min"].to_numpy(dtype=float))
              & (out["exptime_min"].to_numpy(dtype=float) > 0)]
    return out.drop_duplicates(subset=list(EXPOSURE_KEY_COLS), keep="first")


def plate_density(ra: float, dec: float, log: AcquisitionLog | None = None,
                  exposures_out: list | None = None, **kw) -> dict:
    """``queryexps`` at a position: plate counts overall and by gap segment.

    ``exposures_out``, when given, receives this field's `exposure_table` --- so
    the one request that measures the field's plate density also supplies the
    exposure durations the light curves lack.
    """
    r = queryexps(ra, dec, **kw)
    out = {"n_plates": 0, "n_pre_gap": 0, "n_post_gap": 0, "year_min": float("nan"),
           "year_max": float("nan"), "ok": bool(r.ok), "columns": r.columns[:40],
           "elapsed_s": round(r.elapsed_s, 1)}
    if log:
        log.record("queryexps", f"DASCH queryexps ({ra:.5f},{dec:.5f})",
                   rows=(r.n_rows if r.ok else None), error=(None if r.ok else r.error),
                   extra={"status": r.status, "body_head": r.body_head[:200]})
    if not r.ok or r.frame is None or not len(r.frame):
        return out
    df = r.frame
    out["n_plates"] = int(len(df))
    if exposures_out is not None:
        et = exposure_table(df)
        out["n_exptimes"] = int(len(et))
        if len(et):
            out["exptime_min_median"] = float(np.nanmedian(et["exptime_min"].to_numpy(float)))
            exposures_out.append(et)
    yr, tcol = exposure_years(df)
    out["time_column"] = tcol
    out["n_undated"] = int(np.sum(~np.isfinite(yr)))
    yr = yr[np.isfinite(yr)]
    if yr.size:
        out["year_min"], out["year_max"] = float(np.min(yr)), float(np.max(yr))
        out["n_pre_gap"] = int(np.sum(yr < MENZEL_GAP_START))
        out["n_post_gap"] = int(np.sum(yr >= MENZEL_GAP_END))
        out["n_in_gap"] = int(np.sum((yr >= MENZEL_GAP_START) & (yr < MENZEL_GAP_END)))
    lcol = pick_column(df, EXP_LIM_COLS)
    out["lim_column"] = str(lcol) if lcol is not None else None
    if lcol is not None:
        lm = numeric(df, lcol)
        lm = lm[np.isfinite(lm)]
        if lm.size:
            out["lim_median"] = float(np.median(lm))
            out["lim_p90"] = float(np.percentile(lm, 90))
    scol = pick_column(df, ("series",))
    if scol is not None:
        vals, cnt = np.unique(df[scol].astype(str).to_numpy(), return_counts=True)
        top = np.argsort(-cnt)[:12]
        out["series_top"] = {str(vals[i]): int(cnt[i]) for i in top}
    return out


def _vsx_amplitude(df: pd.DataFrame) -> np.ndarray:
    """VSX ``min`` is an amplitude when ``f_min`` carries ``(``; else min - max."""
    mx = numeric(df, pick_column(df, ("mag_max", "max")))
    mn = numeric(df, pick_column(df, ("mag_min", "min")))
    fcol = pick_column(df, ("f_min",))
    amp = mn - mx
    if fcol is not None:
        f = df[fcol].astype(str).to_numpy()
        is_amp = np.array(["(" in s for s in f])
        amp = np.where(is_amp, mn, amp)
    return amp


def select_variables(ra: float, dec: float, radius_deg: float, *, log: AcquisitionLog,
                     mag_max: float = 13.0, period_min: float = 0.2, period_max: float = 100.0,
                     amp_min: float = 0.3, max_targets: int = 400) -> pd.DataFrame:
    """VSX (GCVS fallback) periodic variables in a cone, filtered for the plates."""
    df = fetch_vsx_region(ra, dec, radius_deg, log=log)
    src = "vsx"
    if not len(df):
        df = fetch_gcvs_region(ra, dec, radius_deg, log=log)
        src = "gcvs"
        if len(df):
            df = df.rename(columns={"raj2000": "ra", "dej2000": "dec", "vartype": "vtype",
                                    "magmax": "mag_max", "magmin": "mag_min", "gcvs": "name"})
    if not len(df):
        return pd.DataFrame()
    df = df.rename(columns={c: c.lower() for c in df.columns})
    ra_c = pick_column(df, ("ra", "raj2000", "_ra"))
    de_c = pick_column(df, ("dec", "dej2000", "_de"))
    if ra_c is None or de_c is None:
        return pd.DataFrame()
    per = numeric(df, pick_column(df, ("period",)))
    mx = numeric(df, pick_column(df, ("mag_max", "max", "magmax")))
    amp = _vsx_amplitude(df)
    vt = (df[pick_column(df, ("vtype", "type", "vartype"))].astype(str).to_numpy()
          if pick_column(df, ("vtype", "type", "vartype")) else np.full(len(df), ""))
    keep = (np.isfinite(per) & (per >= period_min) & (per <= period_max)
            & np.isfinite(mx) & (mx <= mag_max)
            & (~np.isfinite(amp) | (amp >= amp_min))
            & np.array([is_periodic_type(v) and not is_lpv_type(v) for v in vt]))
    out = pd.DataFrame({
        "name": df[pick_column(df, ("name",))].astype(str).to_numpy() if pick_column(df, ("name",))
        else [f"{src}_{i}" for i in range(len(df))],
        "ra": numeric(df, ra_c), "dec": numeric(df, de_c), "kind": "variable",
        "vtype": vt, "period_cat": per, "mag_cat": mx, "amp_cat": amp, "source": src,
    })[keep]
    out = out[np.isfinite(out["ra"]) & np.isfinite(out["dec"])]
    log.record("select_variables", f"VSX/GCVS filter mag<={mag_max} P[{period_min},{period_max}] "
               f"amp>={amp_min} periodic types", rows=int(len(out)),
               extra={"n_raw": int(len(df)), "source": src})
    return out.sort_values("mag_cat").head(int(max_targets)).reset_index(drop=True)


def _tile_centres(ra: float, dec: float, radius_deg: float, tile_arcmin: float) -> list:
    """A hexagonal-ish grid of tile centres covering the cone."""
    step = tile_arcmin / 60.0 * 1.6
    cosd = max(np.cos(np.radians(dec)), 0.05)
    out = []
    n = int(np.ceil(radius_deg / step)) + 1
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            dra = i * step / cosd + (0.5 * step / cosd if j % 2 else 0.0)
            ddec = j * step * 0.87
            if np.hypot(dra * cosd, ddec) <= radius_deg:
                out.append((ra + dra, dec + ddec))
    return out


def select_bright(ra: float, dec: float, radius_deg: float, *, log: AcquisitionLog,
                  mag_max: float = 13.0, mag_min: float = 8.0, refcat: str = "apass",
                  tile_arcmin: float = 10.0, max_targets: int = 200, min_ndet: int = 50,
                  pause_s: float = 0.3, time_budget_s: float = 600.0) -> pd.DataFrame:
    """Bright reference-catalogue stars across the field from ``querycat`` tiles.

    Each tile answer also carries the DASCH identifiers every star needs for
    its light curve, so the frame returned is directly fetchable.
    """
    frames = []
    t0 = _time.monotonic()
    n_tiles = n_ok = 0
    for tra, tdec in _tile_centres(ra, dec, radius_deg, tile_arcmin):
        if _time.monotonic() - t0 > time_budget_s:
            break
        n_tiles += 1
        r = querycat(tra, tdec, tile_arcmin * 60.0 * 0.85, refcat=refcat)
        if not r.ok or r.frame is None or not len(r.frame):
            if n_tiles == 1:
                log.record("querycat_tile", f"DASCH querycat ({tra:.4f},{tdec:.4f}) "
                           f"r={tile_arcmin * 60 * 0.85:.0f}\" {refcat}",
                           rows=(0 if r.ok else None), error=(None if r.ok else r.error),
                           extra={"status": r.status, "body_head": r.body_head[:300]})
            _time.sleep(pause_s)
            continue
        n_ok += 1
        frames.append(r.frame)
        _time.sleep(pause_s)
    if not frames:
        log.record("select_bright", f"{n_tiles} tiles, none answered", rows=0)
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = normalise_refcat(df)
    if not len(df):
        log.record("select_bright", "refcat frame lacked ra/dec/ids", rows=0,
                   extra={"columns": frames[0].columns.tolist()[:40]})
        return pd.DataFrame()
    df = df.drop_duplicates(subset=["gsc_bin_index", "ref_number"])
    keep = np.isfinite(df["mag_cat"]) & (df["mag_cat"] <= mag_max) & (df["mag_cat"] >= mag_min)
    # The min_ndet cut is INERT against the DR7 refcat and is recorded as such.
    # ``_COLTYPES`` in daschlab/refcat.py serves ref_text, ref_number,
    # gsc_bin_index, ra_deg, dec_deg, dra_asec, ddec_asec, pos_epoch,
    # pm_*_masyr, u_pm_*_masyr, stdmag, color, class, v_flag, mag_flag,
    # num_matches --- and no detection count.  ``num_matches`` counts catalogue
    # cross-matches, not plate detections, so it is deliberately NOT mapped
    # onto n_det_cat: cutting on it would discard stars for a reason unrelated
    # to how many plates saw them.  The real detection count only exists once
    # the light curve is fetched, where ``lightcurve.min_detections`` applies
    # it.  A silently inert guard is the failure this channel has already been
    # bitten by once, so it is named here and in the acquisition log.
    ndet_available = bool("n_det_cat" in df.columns and np.isfinite(df["n_det_cat"]).any())
    if ndet_available:
        keep &= ~(np.isfinite(df["n_det_cat"]) & (df["n_det_cat"] < min_ndet))
    df = df[keep].copy()
    df["kind"] = "bright"
    df["name"] = [f"DASCH_{g}_{r}" for g, r in zip(df["gsc_bin_index"], df["ref_number"],
                                                   strict=False)]
    df["vtype"], df["period_cat"], df["amp_cat"], df["source"] = "", np.nan, 0.0, refcat
    log.record("select_bright", f"{n_ok}/{n_tiles} querycat tiles, mag in [{mag_min},{mag_max}]",
               rows=int(len(df)),
               extra={"columns": frames[0].columns.tolist()[:40],
                      "min_ndet_applied": ndet_available, "min_ndet": float(min_ndet)})
    return df.sort_values("mag_cat").head(int(max_targets)).reset_index(drop=True)


def normalise_refcat(df: pd.DataFrame) -> pd.DataFrame:
    """Standard columns from a ``querycat`` answer: ra, dec, gsc_bin_index,
    ref_number, mag_cat, n_det_cat, pm_total_masyr, colour."""
    if df is None or not len(df):
        return pd.DataFrame()
    ra_c, de_c = pick_column(df, REF_COLS_RA), pick_column(df, REF_COLS_DEC)
    g_c, r_c = pick_column(df, REF_COLS_GBI), pick_column(df, REF_COLS_REFNUM)
    if ra_c is None or de_c is None or g_c is None or r_c is None:
        return pd.DataFrame()
    out = pd.DataFrame({
        "ra": numeric(df, ra_c), "dec": numeric(df, de_c),
        "gsc_bin_index": pd.to_numeric(df[g_c], errors="coerce"),
        "ref_number": pd.to_numeric(df[r_c], errors="coerce"),
        "mag_cat": numeric(df, pick_column(df, REF_COLS_MAG)),
        "n_det_cat": numeric(df, pick_column(df, REF_COLS_NDET)),
        "colour": numeric(df, pick_column(df, REF_COLS_COLOR)),
    })
    pmra = numeric(df, pick_column(df, REF_COLS_PMRA))
    pmde = numeric(df, pick_column(df, REF_COLS_PMDEC))
    out["pm_total_masyr"] = np.hypot(pmra, pmde)
    out = out[np.isfinite(out["gsc_bin_index"]) & np.isfinite(out["ref_number"])]
    out["gsc_bin_index"] = out["gsc_bin_index"].astype(np.int64)
    out["ref_number"] = out["ref_number"].astype(np.int64)
    return out.reset_index(drop=True)


def match_refcat(ra: float, dec: float, *, radius_arcsec: float = 15.0, refcat: str = "apass",
                 mag_hint: float = float("nan"), **kw) -> tuple[dict | None, object]:
    """The DASCH reference source for a catalogue position: nearest, unless a
    brighter-and-close source better matches the catalogue magnitude."""
    r = querycat(ra, dec, radius_arcsec, refcat=refcat, **kw)
    if not r.ok or r.frame is None or not len(r.frame):
        return None, r
    df = normalise_refcat(r.frame)
    if not len(df):
        return None, r
    cosd = np.cos(np.radians(dec))
    sep = 3600.0 * np.hypot((df["ra"] - ra) * cosd, df["dec"] - dec)
    df = df.assign(sep_arcsec=sep)
    score = sep / max(radius_arcsec, 1e-3)
    if np.isfinite(mag_hint):
        dm = np.abs(df["mag_cat"] - mag_hint)
        score = score + np.where(np.isfinite(dm), dm / 1.5, 1.0)
    i = int(np.argmin(score.to_numpy()))
    row = df.iloc[i].to_dict()
    return row, r


__all__ = ["EXPOSURE_KEY_COLS", "dasch_dates_to_year", "exposure_table",
           "exposure_years", "field_tag", "match_refcat", "normalise_refcat",
           "plate_density", "select_bright", "select_variables"]
