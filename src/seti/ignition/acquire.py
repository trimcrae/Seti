"""NEOWISE single-exposure photometry for IGNITION.

Runner-only where it touches the network (the sandbox has no archive egress);
the frame cleaning, epoch binning and per-star grouping are pure and are what
the offline suite exercises.

What run 35039105536 taught (2026-09-16; ``results/ignition/acquire_s*.json``)
--------------------------------------------------------------------------
The channel's first full run reached its 846-star parent and then lost 314 of
the 336 stars it attempted, with two shards producing nothing at all.  The
per-query ledger says exactly why, and none of it was the sky:

* **The field route cannot be run at 1 degree near the ecliptic poles.**  A
  ``W1 < 13`` cone of 1 degree at the NEP holds 2.4--4.7 million single
  exposures (every source there has ~4,000 epochs, not ~250).  The VOTable is
  ~1.5 GB; the synchronous transfer broke with ``IncompleteRead(1075248610
  bytes read, 642678956 more expected)`` and ``RemoteDisconnected``, the proxy
  answered ``502`` on the retries, one query that did complete took 3,880 s and
  was **truncated at ``TOP 4000000``** (so even its 8 stars had holes in their
  decade), and the shards that never got a first field through were killed at
  the 350-minute job limit with no checkpoint, because the field route only
  checkpoints after a whole field.  A 1-degree field pulls ~25x more rows than
  its ~40 parent stars need.
* **The upload join was refused with ``QUERY must be set``.**  ``pyvo`` sends
  an inline ``TAP_UPLOAD`` as one ``multipart/form-data`` body with ``QUERY``
  as a form field; IRSA's async endpoint does not read it from there.  That is
  a *transport* failure, not a service one, so it is now a ladder
  (:data:`UPLOAD_TRANSPORTS`): pyvo's synchronous form, then a raw ``POST``
  with every parameter in the URL and only the table in the body (sync, then
  async with polling), then IRSA's Gator multi-object search, which takes an
  IPAC table of positions and has never needed ``TAP_UPLOAD`` at all.  The
  probe walks the ladder on two stars and records which rung answered; the
  acquire stage starts from that rung.
* **The per-star cone works**: 4,110 rows in 15.6 s for the probe star, no
  ``COUNT(*)``.  It is the guaranteed fallback --- for a chunk whose upload
  failed, and for the whole shard when no upload transport answers --- and it
  now runs ``cone_workers`` cones concurrently.

What VIGIL taught, still true here
----------------------------------
* **Proper motion is propagated to the mission mid-epoch** and every match
  radius is widened by half the star's mission-long sweep; a 200 mas/yr star
  otherwise drifts out of its own aperture.
* **Every completed batch is written immediately** (CSV append + a progress
  file listing finished ``source_id``\\s), so a killed shard resumes.
* Rows are assigned to stars **locally, by exact angular separation** on unit
  vectors (:func:`group_by_star`), never by trusting a service's own join
  column: the same code serves the upload, Gator and field routes, and a
  star's radius is its own (cone + half its sweep), not the chunk's.
"""

from __future__ import annotations

import io
import json
import time as _time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..vigil.acquire import (
    GAIA_EPOCH,
    IRSA_TAP,
    NEOWISE_END,
    NEOWISE_MID_EPOCH,
    NEOWISE_START,
    QueryResult,
    pm_sweep_arcsec,
    propagate_pm,
    run_tap,
)

NEOWISE_TABLE = "neowiser_p1bs_psd"
NEOWISE_COLS = ("ra", "dec", "mjd", "w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro",
                "qual_frame", "saa_sep", "moon_masked", "cc_flags", "ph_qual", "nb", "na")
IRSA_GATOR = "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"

#: The upload ladder, in the order it is walked.  ``pyvo_sync`` is the library
#: form that failed on the async queue (``QUERY must be set``) tried on the sync
#: one; ``http_sync`` / ``http_async`` put every TAP parameter in the URL and
#: only the VOTable in the multipart body; ``gator`` is IRSA's own multi-object
#: search (``spatial=Upload`` with an IPAC table), which is not TAP at all.
UPLOAD_TRANSPORTS: tuple[str, ...] = ("pyvo_sync", "http_sync", "http_async", "gator")

DEFAULT_ACQUIRE: dict = {
    "route": "auto",              # auto | upload | field | cone
    "cone_radius_arcsec": 2.5,
    "match_tol_arcsec": 2.5,
    "field_w1_max": 13.0,
    "field_max_rows": 4_000_000,
    "upload_chunk": 200,          # stars per upload query, the CEILING
    "upload_target_rows": 300_000,  # chunk shrinks so a query returns about this many rows
    "upload_max_rows": 3_000_000,   # TOP on the upload join; a chunk at the cap is split
    "upload_transport": "auto",   # auto (from probe.json) | one of UPLOAD_TRANSPORTS
    "upload_timeout_s": 900.0,    # per upload query
    "upload_fallback_cone": True, # a chunk no transport answered goes to per-star cones
    "cone_workers": 3,            # concurrent per-star cones
    "checkpoint_every": 50,       # cone route: stars per CSV append
    "epoch_gap_days": 90.0,       # > 90 d gap starts a new epoch (visits are ~183 d apart)
    "min_exp_per_epoch": 5,
    "epoch_err_floor_mag": 0.005,
    "clip_sigma": 5.0,
    "qual_frame_min": 1,
    "cc_flags_ok": ["0000"],
    "ph_qual_ok": ["A", "B"],
    "time_budget_s": 18000.0,
}


def mjd_to_year(mjd) -> np.ndarray:
    """Julian year from MJD."""
    return 2000.0 + (np.asarray(mjd, float) - 51544.5) / 365.25


# --------------------------------------------------------------------------
# Pure: frame cleaning, quality bookkeeping, epoch binning
# --------------------------------------------------------------------------
def clean_frames(df: pd.DataFrame, conf: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """NEOWISE frame-quality cuts, identical in both bands, with a ledger.

    Keeps ``qual_frame > 0``, ``saa_sep > 0``, ``moon_masked == '00'``,
    ``cc_flags`` in the allowed set (default ``'0000'``) and ``ph_qual`` A/B in
    *both* bands.  ``nb`` / ``na`` (blend count, active deblending) are not cut
    here --- they are counted per star and handed to the vet, because a star
    whose every frame is deblended is a different statement from one bad frame.
    """
    c = {**DEFAULT_ACQUIRE, **(conf or {})}
    d = df.copy()
    d.columns = [str(x).lower() for x in d.columns]
    ledger = {"n_raw": int(len(d))}
    if "qual_frame" in d:
        q = pd.to_numeric(d["qual_frame"], errors="coerce").fillna(0)
        keep = q >= int(c["qual_frame_min"])
        ledger["cut_qual_frame"] = int((~keep).sum())
        d = d[keep]
    if "saa_sep" in d:
        sep = pd.to_numeric(d["saa_sep"], errors="coerce")
        keep = sep.isna() | (sep > 0)
        ledger["cut_saa_sep"] = int((~keep).sum())
        d = d[keep]
    if "moon_masked" in d:
        mm = d["moon_masked"].astype(str).str.strip()
        keep = ~mm.str.contains("1")
        ledger["cut_moon_masked"] = int((~keep).sum())
        d = d[keep]
    if "cc_flags" in d:
        ok = {str(x) for x in c["cc_flags_ok"]}
        cc = d["cc_flags"].astype(str).str.strip()
        keep = cc.isin(ok) | (cc.str[:2] == "00")
        ledger["cut_cc_flags"] = int((~keep).sum())
        d = d[keep]
    if "ph_qual" in d:
        ok = {str(x).upper() for x in c["ph_qual_ok"]}
        pq = d["ph_qual"].astype(str).str.upper().str.strip()
        keep = pq.str[:1].isin(ok) & pq.str[1:2].isin(ok)
        ledger["cut_ph_qual"] = int((~keep).sum())
        d = d[keep]
    for col in ("mjd", "w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro"):
        if col in d:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    need = [x for x in ("mjd", "w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro") if x in d]
    if need:
        d = d.dropna(subset=need)
    ledger["n_clean"] = int(len(d))
    return d.reset_index(drop=True), ledger


def star_quality(raw: pd.DataFrame) -> dict:
    """Per-star blend / deblend fractions, measured on the *raw* frames."""
    out = {"n_exp_raw": int(len(raw)), "frac_nb_gt1": float("nan"),
           "frac_na_gt0": float("nan")}
    if not len(raw):
        return out
    d = raw.copy()
    d.columns = [str(x).lower() for x in d.columns]
    if "nb" in d:
        nb = pd.to_numeric(d["nb"], errors="coerce")
        out["frac_nb_gt1"] = float(np.nanmean((nb > 1).to_numpy(dtype=float)))
    if "na" in d:
        na = pd.to_numeric(d["na"], errors="coerce")
        out["frac_na_gt0"] = float(np.nanmean((na > 0).to_numpy(dtype=float)))
    return out


def exposures_to_epochs(mjd, mag, magerr, gap_days: float = 90.0, min_exp: int = 5,
                        clip_sigma: float = 5.0, err_floor: float = 0.005) -> pd.DataFrame:
    """Group single exposures into epochs separated by gaps > ``gap_days``.

    NEOWISE returns to a field roughly every six months, so a 90-day gap cuts
    cleanly between visits everywhere except the ecliptic poles, where the
    visits themselves stretch --- and there the binning still holds because the
    gap between successive pole visits remains > 90 d.  Per epoch: the median
    magnitude and a robust error, ``max(1.4826 * MAD / sqrt(n), err_floor)``,
    after a MAD clip of single bad frames.  Epochs with fewer than ``min_exp``
    surviving exposures are dropped.
    """
    t = np.asarray(mjd, float)
    m = np.asarray(mag, float)
    e = np.asarray(magerr, float)
    ok = np.isfinite(t) & np.isfinite(m)
    t, m, e = t[ok], m[ok], e[ok]
    cols = ["epoch", "t_mjd", "t_yr", "mag", "err", "n_exp", "scatter"]
    if t.size == 0:
        return pd.DataFrame(columns=cols)
    order = np.argsort(t)
    t, m, e = t[order], m[order], e[order]
    edges = np.nonzero(np.diff(t) > gap_days)[0] + 1
    rows = []
    for k, g in enumerate(np.split(np.arange(t.size), edges)):
        if g.size < min_exp:
            continue
        mv, tv = m[g], t[g]
        med = float(np.median(mv))
        mad = 1.4826 * float(np.median(np.abs(mv - med)))
        if mad > 0:
            keep = np.abs(mv - med) <= clip_sigma * mad
            if keep.sum() >= min_exp:
                mv, tv = mv[keep], tv[keep]
                med = float(np.median(mv))
                mad = 1.4826 * float(np.median(np.abs(mv - med)))
        n = int(mv.size)
        if mad <= 0:
            # Degenerate scatter (quantised magnitudes): fall back on the quoted errors.
            mad = float(np.nanmedian(e[g])) if np.isfinite(np.nanmedian(e[g])) else err_floor
        err = max(mad / np.sqrt(n), err_floor)
        rows.append({"epoch": k, "t_mjd": float(np.mean(tv)),
                     "t_yr": float(mjd_to_year(np.mean(tv))), "mag": med,
                     "err": float(err), "n_exp": n, "scatter": float(mad)})
    return pd.DataFrame(rows, columns=cols)


def star_epochs(frames: pd.DataFrame, conf: dict | None = None) -> pd.DataFrame:
    """Both bands of one star's cleaned frames -> long epoch table."""
    c = {**DEFAULT_ACQUIRE, **(conf or {})}
    out = []
    for band, mcol, ecol in (("W1", "w1mpro", "w1sigmpro"), ("W2", "w2mpro", "w2sigmpro")):
        if mcol not in frames:
            continue
        ep = exposures_to_epochs(frames["mjd"], frames[mcol], frames.get(ecol, np.nan),
                                 gap_days=float(c["epoch_gap_days"]),
                                 min_exp=int(c["min_exp_per_epoch"]),
                                 clip_sigma=float(c["clip_sigma"]),
                                 err_floor=float(c["epoch_err_floor_mag"]))
        ep.insert(0, "band", band)
        out.append(ep)
    if not out:
        return pd.DataFrame(columns=["band", "epoch", "t_mjd", "t_yr", "mag", "err",
                                     "n_exp", "scatter"])
    return pd.concat(out, ignore_index=True)


def epochs_to_series(ep: pd.DataFrame) -> dict:
    """Long epoch table for one star -> ``{"W1": (t_yr, mag, err), "W2": ...}``."""
    series = {}
    for band, g in ep.groupby("band"):
        g = g.sort_values("t_yr")
        series[str(band)] = (g["t_yr"].to_numpy(float), g["mag"].to_numpy(float),
                             g["err"].to_numpy(float))
    return series


def positions_at_neowise_epoch(stars: pd.DataFrame, from_epoch: float = GAIA_EPOCH,
                               to_epoch: float = NEOWISE_MID_EPOCH) -> pd.DataFrame:
    """PM-propagated positions plus the per-star match radius widening."""
    pmra = pd.to_numeric(stars.get("pmra", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
    pmdec = pd.to_numeric(stars.get("pmdec", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
    ra_m, dec_m = propagate_pm(stars["ra"].to_numpy(float), stars["dec"].to_numpy(float),
                               pmra, pmdec, from_epoch, to_epoch)
    sweep = np.hypot(pmra, pmdec) * (NEOWISE_END - NEOWISE_START) / 1000.0
    return pd.DataFrame({"source_id": stars["source_id"].astype(str).to_numpy(),
                         "ra_mid": np.asarray(ra_m, float), "dec_mid": np.asarray(dec_m, float),
                         "sweep_arcsec": sweep})


def _unit_vectors(ra_deg, dec_deg) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def group_by_star(frames: pd.DataFrame, stars: pd.DataFrame,
                  tol_arcsec: float = 2.5) -> dict[str, pd.DataFrame]:
    """Assign exposures to stars by exact angular separation, PM propagated.

    Each star's Gaia position is moved to the NEOWISE mid-epoch and its radius is
    ``tol_arcsec`` plus half its mission-long sweep.  The match is a KD-tree on
    **unit vectors** (chord length ``2 sin(sep/2)``), which is exact at every
    declination --- a flat ``(ra cos dec, dec)`` tree with one ``cos dec`` for a
    whole chunk is 10 % off in RA across a 4-degree tile at dec 60.  Pure.
    """
    from scipy.spatial import cKDTree

    out: dict[str, pd.DataFrame] = {}
    if frames is None or not len(frames) or stars is None or not len(stars):
        return out
    d = frames.copy()
    d.columns = [str(c).lower() for c in d.columns]
    if "ra" not in d or "dec" not in d:
        return out
    ra = pd.to_numeric(d["ra"], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(d["dec"], errors="coerce").to_numpy(float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    d = d[ok].reset_index(drop=True)
    if not len(d):
        return out
    tree = cKDTree(_unit_vectors(ra[ok], dec[ok]))
    pos = positions_at_neowise_epoch(stars)
    xyz = _unit_vectors(pos["ra_mid"], pos["dec_mid"])
    rad = (float(tol_arcsec) + 0.5 * pos["sweep_arcsec"].to_numpy(float)) / 3600.0
    chord = 2.0 * np.sin(np.radians(rad) / 2.0)
    for i in range(len(pos)):
        idx = tree.query_ball_point(xyz[i], r=float(chord[i]))
        if len(idx) >= 3:
            out[str(pos["source_id"].iloc[i])] = d.iloc[sorted(idx)].reset_index(drop=True)
    return out


def ecliptic_latitude_deg(ra_deg, dec_deg) -> np.ndarray:
    """Ecliptic latitude (J2000 obliquity), for the NEOWISE cadence estimate."""
    eps = np.radians(23.4392911)
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    sb = np.sin(dec) * np.cos(eps) - np.cos(dec) * np.sin(eps) * np.sin(ra)
    return np.degrees(np.arcsin(np.clip(sb, -1.0, 1.0)))


def expected_rows_per_star(ra_deg, dec_deg) -> np.ndarray:
    """Rough NEOWISE single-exposure count per source over the mission.

    ~12 exposures per visit, two visits a year for 10.7 years away from the
    ecliptic poles; the visits lengthen as ``1 / cos(beta)`` toward the poles
    (the probe star at beta = 87.6 deg had 4,110 rows).  Capped at 6,000.  Used
    only to size upload chunks, never for science.
    """
    beta = np.radians(ecliptic_latitude_deg(ra_deg, dec_deg))
    return np.clip(260.0 / np.maximum(np.cos(beta), 0.04), 260.0, 6000.0)


# --------------------------------------------------------------------------
# Runner: the three routes
# --------------------------------------------------------------------------
def _cols() -> str:
    return ", ".join(NEOWISE_COLS)


def fetch_neowise_cone(ra: float, dec: float, pmra: float = 0.0, pmdec: float = 0.0,
                       radius_arcsec: float = 2.5, retries: int = 3) -> QueryResult:
    """One PM-propagated cone, *without* the ``COUNT(*)`` that doubled VIGIL's cost."""
    ra_m, dec_m = propagate_pm(ra, dec, pmra, pmdec, GAIA_EPOCH, NEOWISE_MID_EPOCH)
    rad = float(radius_arcsec + 0.5 * pm_sweep_arcsec(pmra, pmdec))
    q = (f"SELECT {_cols()} FROM {NEOWISE_TABLE} WHERE "
         f"1 = CONTAINS(POINT('ICRS', ra, dec), "
         f"CIRCLE('ICRS', {float(ra_m):.7f}, {float(dec_m):.7f}, {rad / 3600.0:.9f}))")
    return run_tap(IRSA_TAP, q, label=f"neowise_cone_pm{pm_sweep_arcsec(pmra, pmdec):.2f}as",
                   retries=retries, async_first=False)


def upload_query(radius_arcsec: float = 2.5, max_rows: int | None = None,
                 radius_column: bool = False) -> str:
    """The ``TAP_UPLOAD`` join.

    The radius is one constant per query (the chunk's widest star), not a
    per-row ``p.rad``: a column inside ``CIRCLE`` is not something every ADQL
    planner can use an index for, and the per-star radius is applied locally by
    :func:`group_by_star` anyway.  ``radius_column=True`` keeps the old form.
    """
    cols = ", ".join(f"n.{c}" for c in NEOWISE_COLS)
    top = f"TOP {int(max_rows)} " if max_rows else ""
    rad = "p.rad" if radius_column else f"{float(radius_arcsec) / 3600.0:.9f}"
    return (f"SELECT {top}p.sid, {cols} FROM {NEOWISE_TABLE} AS n, TAP_UPLOAD.pos AS p "
            f"WHERE 1 = CONTAINS(POINT('ICRS', n.ra, n.dec), "
            f"CIRCLE('ICRS', p.ra, p.dec, {rad}))")


def _upload_table(stars: pd.DataFrame, radius_arcsec: float):
    """The positions table (astropy) and the chunk-wide radius in arcsec."""
    from astropy.table import Table

    pos = positions_at_neowise_epoch(stars)
    rad = (float(radius_arcsec) + 0.5 * pos["sweep_arcsec"]).astype(float)
    tbl = Table.from_pandas(pd.DataFrame({
        "sid": pos["source_id"].astype(str), "ra": pos["ra_mid"].astype(float),
        "dec": pos["dec_mid"].astype(float), "rad": rad / 3600.0}))
    return tbl, float(rad.max()) if len(rad) else float(radius_arcsec)


def _votable_bytes(tbl) -> bytes:
    from astropy.io.votable import from_table, writeto

    buf = io.BytesIO()
    writeto(from_table(tbl), buf)
    return buf.getvalue()


def _ipac_bytes(tbl) -> bytes:
    buf = io.StringIO()
    tbl[["sid", "ra", "dec"]].write(buf, format="ascii.ipac")
    return buf.getvalue().encode("utf-8")


def _parse_votable(content: bytes) -> pd.DataFrame:
    """A TAP VOTable -> DataFrame; a ``QUERY_STATUS = ERROR`` raises with its text."""
    from astropy.io.votable import parse

    vot = parse(io.BytesIO(content), verify="ignore")
    for res in vot.resources:
        for info in res.infos:
            if str(info.name).upper() == "QUERY_STATUS" and str(info.value).upper() == "ERROR":
                raise RuntimeError(f"TAP QUERY_STATUS=ERROR: {str(info.content)[:500]}")
    tab = vot.get_first_table().to_table(use_names_over_ids=True)
    df = tab.to_pandas()
    df.columns = [str(c).lower() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.decode() if isinstance(v, bytes) else v)
    return df


def _raise_if_error_text(text: str, where: str) -> None:
    head = text[:4000]
    if "QUERY_STATUS" in head and 'value="ERROR"' in head:
        raise RuntimeError(f"{where}: {head[:600]}")
    if head.lstrip().startswith("<!DOCTYPE") or "<html" in head[:200].lower():
        raise RuntimeError(f"{where}: HTML answer, not a table: {head[:300]}")


def _t_pyvo_sync(q: str, tbl, timeout_s: float) -> pd.DataFrame:
    import pyvo  # noqa: PLC0415  runner-only

    svc = pyvo.dal.TAPService(IRSA_TAP)
    df = svc.run_sync(q, uploads={"pos": tbl}).to_table().to_pandas()
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _tap_params(q: str) -> dict:
    return {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": q,
            "UPLOAD": "pos,param:pos"}


def _t_http_sync(q: str, tbl, timeout_s: float) -> pd.DataFrame:
    """Every TAP parameter in the URL; only the VOTable in the multipart body."""
    import requests  # noqa: PLC0415

    r = requests.post(f"{IRSA_TAP}/sync", params=_tap_params(q),
                      files={"pos": ("pos.xml", _votable_bytes(tbl),
                                     "application/x-votable+xml")},
                      timeout=(60.0, float(timeout_s)))
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} from {IRSA_TAP}/sync: {r.text[:400]}")
    return _parse_votable(r.content)


def _t_http_async(q: str, tbl, timeout_s: float) -> pd.DataFrame:
    """The same request on the UWS queue, then poll the phase and fetch the result."""
    import requests  # noqa: PLC0415

    r = requests.post(f"{IRSA_TAP}/async", params=_tap_params(q),
                      files={"pos": ("pos.xml", _votable_bytes(tbl),
                                     "application/x-votable+xml")},
                      timeout=(60.0, 300.0), allow_redirects=False)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} from {IRSA_TAP}/async: {r.text[:400]}")
    job = r.headers.get("Location") or r.url
    if "/async/" not in job:
        raise RuntimeError(f"no UWS job URL in the async answer: {r.status_code} {job}")
    job = job.rstrip("/")
    r = requests.post(f"{job}/phase", data={"PHASE": "RUN"}, timeout=(60.0, 120.0),
                      allow_redirects=False)
    t0 = _time.monotonic()
    nap = 4.0
    while True:
        ph = requests.get(f"{job}/phase", timeout=(60.0, 120.0)).text.strip().upper()
        if ph == "COMPLETED":
            break
        if ph in ("ERROR", "ABORTED"):
            err = requests.get(f"{job}/error", timeout=(60.0, 120.0)).text[:600]
            raise RuntimeError(f"UWS job {ph}: {err}")
        if _time.monotonic() - t0 > float(timeout_s):
            try:
                requests.post(f"{job}/phase", data={"PHASE": "ABORT"}, timeout=(30.0, 60.0))
            finally:
                pass
            raise RuntimeError(f"UWS job still {ph} after {timeout_s:.0f} s")
        _time.sleep(nap)
        nap = min(nap * 1.5, 30.0)
    r = requests.get(f"{job}/results/result", timeout=(60.0, float(timeout_s)))
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} fetching the UWS result: {r.text[:400]}")
    return _parse_votable(r.content)


def _t_gator(q: str, tbl, timeout_s: float, radius_arcsec: float = 2.5) -> pd.DataFrame:
    """IRSA's Gator multi-object search: an IPAC table of positions, one radius."""
    import requests  # noqa: PLC0415
    from astropy.io import ascii as _ascii  # noqa: PLC0415

    data = {"catalog": NEOWISE_TABLE, "spatial": "Upload", "radius": f"{float(radius_arcsec):.3f}",
            "radunits": "arcsec", "outfmt": "1", "selcols": ",".join(NEOWISE_COLS)}
    r = requests.post(IRSA_GATOR, data=data,
                      files={"uploadfile": ("pos.tbl", _ipac_bytes(tbl), "text/plain")},
                      timeout=(60.0, float(timeout_s)))
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} from Gator: {r.text[:400]}")
    text = r.text
    _raise_if_error_text(text, "Gator")
    if "\\" not in text[:2000] and "|" not in text[:4000]:
        raise RuntimeError(f"Gator answered no IPAC table: {text[:300]}")
    tab = _ascii.read(text, format="ipac")
    df = tab.to_pandas()
    df.columns = [str(c).lower() for c in df.columns]
    # The uploaded columns come back suffixed (ra_01 / dec_01 ...); the catalogue's
    # own ra/dec are the ones the local assignment needs, and they are unsuffixed.
    return df


_TRANSPORT_FNS = {"pyvo_sync": _t_pyvo_sync, "http_sync": _t_http_sync,
                  "http_async": _t_http_async, "gator": _t_gator}


def transport_order(preferred: str | None = None) -> list[str]:
    """The ladder, with ``preferred`` (the probe's answer) moved to the front."""
    order = list(UPLOAD_TRANSPORTS)
    if preferred in order:
        order.remove(preferred)
        order.insert(0, preferred)
    return order


def fetch_neowise_upload(stars: pd.DataFrame, radius_arcsec: float = 2.5, *,
                         transport: str | None = None, transports: dict | None = None,
                         timeout_s: float = 900.0, max_rows: int | None = 3_000_000,
                         ladder: bool = True) -> QueryResult:
    """Batched positional search through the upload ladder (probed, never assumed).

    Walks :func:`transport_order` (``transport`` first) until one rung returns
    a table.  The result's ``label`` is ``neowise_upload[<rung>]_<n_stars>`` so
    the ledger says which transport served the chunk; ``transports`` injects
    the rungs for the offline suite.  Rows are *not* grouped here --- the
    caller assigns them to stars by position (:func:`group_by_star`).
    """
    tbl, rad = _upload_table(stars, radius_arcsec)
    q = upload_query(rad, max_rows=max_rows)
    fns = dict(_TRANSPORT_FNS) if transports is None else dict(transports)
    order = transport_order(transport) if ladder else [transport or UPLOAD_TRANSPORTS[0]]
    t0 = _time.monotonic()
    errors: list[str] = []
    for name in order:
        fn = fns.get(name)
        if fn is None:
            continue
        ta = _time.monotonic()
        try:
            if name == "gator":
                df = fn(q, tbl, timeout_s, radius_arcsec=rad)
            else:
                df = fn(q, tbl, timeout_s)
            n = int(len(df))
            trunc = bool(max_rows and n >= int(max_rows))
            print(f"[ignition] upload[{name}] {len(tbl)} stars -> {n} rows in "
                  f"{_time.monotonic() - ta:.0f} s" + (" TRUNCATED" if trunc else ""),
                  flush=True)
            return QueryResult(label=f"neowise_upload[{name}]_{len(tbl)}", service=IRSA_TAP,
                               status="OK" if n else "QUERY_RETURNED_ZERO_ROWS", n_rows=n,
                               truncated=trunc, query=q, elapsed_s=_time.monotonic() - t0,
                               error="; ".join(errors), data=df, n_rows_raw=n)
        except Exception as exc:                       # noqa: BLE001
            err = f"{name}: {exc!r}"[:600]
            errors.append(err)
            print(f"[ignition] upload[{name}] failed after {_time.monotonic() - ta:.0f} s: "
                  f"{err[:300]}", flush=True)
    return QueryResult(label=f"neowise_upload[none]_{len(tbl)}", service=IRSA_TAP,
                       status="QUERY_FAILED", query=q, error=" | ".join(errors),
                       elapsed_s=_time.monotonic() - t0)


def fetch_neowise_field(ra: float, dec: float, radius_deg: float, w1_max: float = 13.0,
                        max_rows: int = 4_000_000) -> QueryResult:
    """VIGIL's field-wide cone (one query per field, grouped locally).

    Kept for explicit ``--route field`` only.  Run 35039105536 measured it at
    2.4--4.7 M rows per 1-degree cone near the ecliptic poles, 20--65 minutes a
    query, broken transfers and ``TOP`` truncation; it is no longer recommended
    by the probe.
    """
    from ..vigil.acquire import fetch_neowise_field as _field

    return _field(ra, dec, radius_deg=radius_deg, w1_max=w1_max, max_rows=max_rows)


# --------------------------------------------------------------------------
# Checkpointed epoch store
# --------------------------------------------------------------------------
@dataclass
class EpochStore:
    """Append-only per-shard store: epochs CSV + star-quality CSV + progress JSON.

    Every completed batch lands on disk before the next query is issued, so a
    killed shard loses one batch.  ``done`` is the set of source_ids already
    written; the acquire loop skips them on resume.
    """

    epochs_path: Path
    stars_path: Path
    progress_path: Path
    done: set = field(default_factory=set)
    ledger: list = field(default_factory=list)
    n_stars_written: int = 0

    @classmethod
    def open(cls, out: Path, tag: str) -> EpochStore:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        st = cls(out / f"epochs_{tag}.csv", out / f"neowise_stars_{tag}.csv",
                 out / f"acquire_{tag}.json")
        if st.progress_path.exists():
            try:
                p = json.loads(st.progress_path.read_text())
                st.done = set(str(x) for x in p.get("done", []))
                st.ledger = list(p.get("ledger", []))
                st.n_stars_written = int(p.get("n_stars_written", len(st.done)))
                print(f"[ignition] resuming {tag}: {len(st.done)} stars already done")
            except Exception as exc:                   # noqa: BLE001
                print(f"[ignition] progress file unreadable ({exc!r}); starting fresh")
        return st

    def write_batch(self, epochs: list[pd.DataFrame], stars: list[dict],
                    done_ids: list[str]) -> None:
        if epochs:
            ep = pd.concat(epochs, ignore_index=True)
            ep.to_csv(self.epochs_path, mode="a", index=False,
                      header=not self.epochs_path.exists())
        if stars:
            sd = pd.DataFrame(stars)
            sd.to_csv(self.stars_path, mode="a", index=False,
                      header=not self.stars_path.exists())
        self.done.update(str(x) for x in done_ids)
        self.n_stars_written += len(stars)
        self.flush()

    def flush(self, extra: dict | None = None) -> None:
        rec = {"done": sorted(self.done), "n_done": len(self.done),
               "n_stars_written": self.n_stars_written, "ledger": self.ledger[-500:]}
        if extra:
            rec.update(extra)
        self.progress_path.write_text(json.dumps(rec, indent=1, default=str))


def reduce_star(sid: str, raw: pd.DataFrame, conf: dict | None = None
                ) -> tuple[pd.DataFrame, dict]:
    """Raw frames of one star -> (epoch rows, star record)."""
    clean, led = clean_frames(raw, conf)
    ep = star_epochs(clean, conf)
    ep.insert(0, "source_id", str(sid))
    q = star_quality(raw)
    rec = {"source_id": str(sid), **q, "n_exp_clean": int(len(clean)),
           "n_epochs_w1": int((ep["band"] == "W1").sum()) if len(ep) else 0,
           "n_epochs_w2": int((ep["band"] == "W2").sum()) if len(ep) else 0,
           "w1_median": float(clean["w1mpro"].median()) if len(clean) else float("nan"),
           "w2_median": float(clean["w2mpro"].median()) if len(clean) else float("nan"),
           "cut_cc_flags": led.get("cut_cc_flags", 0), "cut_ph_qual": led.get("cut_ph_qual", 0),
           "cut_qual_frame": led.get("cut_qual_frame", 0),
           "cut_moon_masked": led.get("cut_moon_masked", 0)}
    return ep, rec


def _accepts(fn, name: str) -> bool:
    """Whether ``fn`` takes a keyword ``name`` (injected fetchers may be simpler)."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def _ledger_slim(r: QueryResult) -> dict:
    d = r.to_ledger()
    d["query"] = (d.get("query") or "")[:400]
    return d


def upload_chunks(todo: pd.DataFrame, chunk_max: int, target_rows: int) -> list[pd.DataFrame]:
    """Split the to-do list into upload chunks sized by the expected row count.

    Near the ecliptic poles a source has ~4,000 exposures rather than ~250, so a
    200-star chunk there would return ~1 M rows (a 250 MB VOTable, the size that
    broke the field route).  The chunk shrinks so each query returns roughly
    ``target_rows``; sorted by declination first so chunks are compact.
    """
    if not len(todo):
        return []
    t = todo.sort_values(["dec", "ra"]).reset_index(drop=True)
    est = expected_rows_per_star(t["ra"].to_numpy(float), t["dec"].to_numpy(float))
    out: list[pd.DataFrame] = []
    i = 0
    while i < len(t):
        k = int(np.clip(float(target_rows) / max(float(est[i]), 1.0), 10, max(int(chunk_max), 10)))
        out.append(t.iloc[i:i + k])
        i += k
    return out


def acquire_stars(stars: pd.DataFrame, store: EpochStore, conf: dict | None = None, *,
                  route: str = "cone", fields: list[dict] | None = None,
                  cone_fn=None, upload_fn=None, field_fn=None,
                  time_budget_s: float | None = None, upload_transport: str | None = None,
                  deadline: float | None = None) -> dict:
    """Fetch, clean, bin and checkpoint every star not already in ``store``.

    ``route``: ``upload`` (batched positional join, per-chunk fallback to
    cones), ``field`` (one cone per entry of ``fields``, grouped locally),
    ``cone`` (one cone per star, ``cone_workers`` at a time).  The fetchers are
    injectable so the loop runs offline.  ``deadline`` is a ``time.monotonic``
    instant (the shard's wall clock) that overrides the per-call budget.
    """
    c = {**DEFAULT_ACQUIRE, **(conf or {})}
    budget = float(c["time_budget_s"] if time_budget_s is None else time_budget_s)
    t_start = _time.monotonic()
    stop_at = t_start + budget if deadline is None else min(t_start + budget, float(deadline))
    stars = stars.copy()
    stars["source_id"] = stars["source_id"].astype(str)
    todo = stars[~stars["source_id"].isin(store.done)].reset_index(drop=True)
    n_ok = n_zero = n_fail = n_fallback = 0
    tol = float(c["match_tol_arcsec"])
    rad = float(c["cone_radius_arcsec"])
    stopped = False

    def _over() -> bool:
        return _time.monotonic() > stop_at

    def _finish(sid, raw, eps, recs, ids):
        nonlocal n_ok, n_zero
        if raw is None or not len(raw):
            n_zero += 1
            recs.append({"source_id": str(sid), "n_exp_raw": 0, "n_exp_clean": 0,
                         "n_epochs_w1": 0, "n_epochs_w2": 0, "neowise_status": "ZERO_ROWS"})
            ids.append(sid)
            return
        ep, rec = reduce_star(sid, raw, c)
        rec["neowise_status"] = "OK"
        n_ok += 1
        if len(ep):
            eps.append(ep)
        recs.append(rec)
        ids.append(sid)

    def _cones(sub: pd.DataFrame, fallback: bool = False) -> None:
        """Per-star cones over ``sub``, ``cone_workers`` at a time, checkpointed."""
        nonlocal n_fail, n_fallback, stopped
        fn = cone_fn or fetch_neowise_cone
        every = max(int(c["checkpoint_every"]), 1)
        workers = max(int(c.get("cone_workers") or 1), 1)

        def _one(s):
            try:
                return fn(float(s["ra"]), float(s["dec"]),
                          float(s.get("pmra", 0.0) or 0.0), float(s.get("pmdec", 0.0) or 0.0),
                          radius_arcsec=rad)
            except Exception as exc:                   # noqa: BLE001
                return QueryResult(label="neowise_cone", service=IRSA_TAP,
                                   status="QUERY_FAILED", error=repr(exc))

        rows = [s for _, s in sub.iterrows()]
        for i in range(0, len(rows), every):
            if _over():
                store.ledger.append({"label": "time_budget", "status": "STOPPED"})
                stopped = True
                return
            batch = rows[i:i + every]
            if workers > 1 and len(batch) > 1:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    results = list(ex.map(_one, batch))
            else:
                results = [_one(s) for s in batch]
            eps, recs, ids = [], [], []
            for s, r in zip(batch, results, strict=False):
                sid = str(s["source_id"])
                if r.status == "QUERY_FAILED":
                    n_fail += 1
                    store.ledger.append(_ledger_slim(r) | {"source_id": sid,
                                                           "fallback": bool(fallback)})
                    continue
                if fallback:
                    n_fallback += 1
                _finish(sid, r.data, eps, recs, ids)
            store.write_batch(eps, recs, ids)

    if route == "field":
        if field_fn is None:
            field_fn = fetch_neowise_field
        for f in (fields or []):
            if _over():
                store.ledger.append({"label": "time_budget", "status": "STOPPED"})
                stopped = True
                break
            ra, dec, rdeg = float(f["ra"]), float(f["dec"]), float(f["radius_deg"])
            members = todo[_in_cone(todo, ra, dec, rdeg)]
            if not len(members):
                continue
            try:
                r = field_fn(ra, dec, rdeg, w1_max=float(c["field_w1_max"]),
                             max_rows=int(c["field_max_rows"]))
            except Exception as exc:                   # noqa: BLE001
                r = QueryResult(label="neowise_field", service=IRSA_TAP,
                                status="QUERY_FAILED", error=repr(exc))
            store.ledger.append(_ledger_slim(r) | {"field": f})
            if r.status == "QUERY_FAILED" or r.data is None:
                n_fail += len(members)
                continue
            grouped = group_by_star(r.data, members, tol_arcsec=tol) if len(r.data) else {}
            eps, recs, ids = [], [], []
            for _, s in members.iterrows():
                sid = str(s["source_id"])
                _finish(sid, grouped.get(sid), eps, recs, ids)
            store.write_batch(eps, recs, ids)
            todo = todo[~todo["source_id"].isin(store.done)]
    elif route == "upload":
        if upload_fn is None:
            upload_fn = fetch_neowise_upload
        preferred = upload_transport or (c.get("upload_transport") if
                                         c.get("upload_transport") != "auto" else None)
        chunks = upload_chunks(todo, int(c["upload_chunk"]), int(c["upload_target_rows"]))
        pending = list(chunks)
        splits = 0
        while pending:
            if _over():
                store.ledger.append({"label": "time_budget", "status": "STOPPED"})
                stopped = True
                break
            sub = pending.pop(0)
            try:
                if _accepts(upload_fn, "transport"):
                    r = upload_fn(sub, radius_arcsec=rad, transport=preferred,
                                  timeout_s=float(c["upload_timeout_s"]),
                                  max_rows=int(c["upload_max_rows"]))
                else:
                    r = upload_fn(sub, radius_arcsec=rad)
            except Exception as exc:                   # noqa: BLE001
                r = QueryResult(label="neowise_upload[error]", service=IRSA_TAP,
                                status="QUERY_FAILED", error=repr(exc))
            store.ledger.append(_ledger_slim(r) | {"n_stars": int(len(sub))})
            if r.status == "OK" and "[" in r.label:
                preferred = r.label.split("[", 1)[1].split("]", 1)[0]
            if r.status == "QUERY_FAILED" or r.data is None:
                if c.get("upload_fallback_cone", True) and not _over():
                    print(f"[ignition] upload chunk of {len(sub)} failed on every transport; "
                          f"falling back to per-star cones", flush=True)
                    _cones(sub, fallback=True)
                else:
                    n_fail += len(sub)
                continue
            if r.truncated and len(sub) > 10 and splits < 64:
                # The chunk hit the row cap: its stars have holes in their decade.
                # Halve it and try again rather than keep a truncated series.
                half = len(sub) // 2
                pending[:0] = [sub.iloc[:half], sub.iloc[half:]]
                splits += 1
                store.ledger.append({"label": "upload_truncated_split", "n_stars": int(len(sub))})
                continue
            grouped = group_by_star(r.data, sub, tol_arcsec=tol) if len(r.data) else {}
            eps, recs, ids = [], [], []
            for sid in sub["source_id"]:
                _finish(sid, grouped.get(str(sid)), eps, recs, ids)
            store.write_batch(eps, recs, ids)
    else:
        _cones(todo)

    roll = {"route": route, "n_attempted": int(len(todo)), "n_ok": n_ok, "n_zero_rows": n_zero,
            "n_failed": n_fail, "n_fallback_cone": n_fallback, "n_done_total": len(store.done),
            "stopped_on_budget": bool(stopped),
            "elapsed_s": round(_time.monotonic() - t_start, 1)}
    store.flush({"rollup": roll})
    return roll


def _in_cone(df: pd.DataFrame, ra: float, dec: float, radius_deg: float) -> np.ndarray:
    ra1 = np.radians(df["ra"].to_numpy(float))
    dec1 = np.radians(df["dec"].to_numpy(float))
    ra0, dec0 = np.radians(ra), np.radians(dec)
    cos_sep = (np.sin(dec1) * np.sin(dec0)
               + np.cos(dec1) * np.cos(dec0) * np.cos(ra1 - ra0))
    return np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0))) <= radius_deg


__all__ = ["DEFAULT_ACQUIRE", "IRSA_GATOR", "NEOWISE_COLS", "NEOWISE_TABLE", "UPLOAD_TRANSPORTS",
           "EpochStore", "acquire_stars", "clean_frames", "ecliptic_latitude_deg",
           "epochs_to_series", "expected_rows_per_star", "exposures_to_epochs",
           "fetch_neowise_cone", "fetch_neowise_field", "fetch_neowise_upload", "group_by_star",
           "mjd_to_year", "positions_at_neowise_epoch", "reduce_star", "star_epochs",
           "star_quality", "transport_order", "upload_chunks", "upload_query"]
