"""Runner-only acquisition for RING (S63): four host classes, one prior.

Nothing here runs in the sandbox (no archive egress); every fetcher is
injectable so the loops are exercised offline.  Every leg records the ROUTE
that served it and degrades to an explicit status rather than to an empty
table that reads like a null.

White dwarfs
------------
Route A is the Gaia archive's own copy of the Gentile Fusillo et al. (2021)
catalogue joined in-archive to ``allwise_best_neighbour`` and the AllWISE and
2MASS mirrors -- the official cross-match already propagates proper motion to
the AllWISE epoch (Marrese et al. 2019), which a positional match on nearby
white dwarfs (hundreds of mas/yr) silently gets wrong.  Route B pulls the
parent from VizieR and joins it to the same archive cross-match through a
table upload; route C is a CDS X-Match at positions propagated by hand.  The
query builder and column probe are the OSSUARY ones, because this channel is
"the same acquisition, a different temperature prior".

Pulsars
-------
The ATNF catalogue through ``psrqpy`` (which reads the catalogue tarball),
through the tarball parsed here, or through VizieR ``B/psr`` (older, smaller)
as the last resort.  Positions are propagated from ``POSEPOCH`` to the
AllWISE and CatWISE epochs, matched through CDS X-Match, and the SAME upload
carries two rings of eight offset positions per pulsar so the chance-match
rate is measured on the sky next to each target rather than assumed.

Brown dwarfs and free-floating planets
--------------------------------------
VizieR tables whose columns are resolved at run time by role, never by an
assumed name; the NEOWISE single-exposure series come through IGNITION's
checkpointed store (proper-motion-propagated cones, count-verified).
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..ossuary.acquire import _ALLWISE_WANT, _TMASS_WANT, _TMASS_XMATCH_ID, probe_columns
from ..ossuary.acquire import _run_query as gaia_query
from ..vigil.acquire import propagate_pm

# --------------------------------------------------------------------------
# White dwarfs: the Gaia-archive join (route A)
# --------------------------------------------------------------------------

WD_WANT = {
    "wd_name": ["wd_name", "wdj_name", "name"],
    "wd_source_id": ["source_id", "gaiaedr3", "gaia_edr3", "gaia_id"],
    "pwd": ["pwd"],
    "teff_h": ["teff_h"], "e_teff_h": ["eteff_h", "e_teff_h", "teff_h_error"],
    "logg_h": ["logg_h"], "mass_h": ["mass_h"], "chisq_h": ["chisq_h", "chi2_h"],
    "teff_he": ["teff_he"], "logg_he": ["logg_he"], "mass_he": ["mass_he"],
    "chisq_he": ["chisq_he", "chi2_he"],
    "meanav": ["meanav", "mean_av", "av"],
}

_DEC_EDGES = list(range(-90, 91, 15))


def probe_gaia_wd_table(candidates, probe=probe_columns) -> tuple[str | None, dict]:
    """First Gaia-archive table among ``candidates`` that exposes the WD columns."""
    for t in candidates:
        cols = probe(t, WD_WANT, tag="ring/wd")
        if cols.get("wd_source_id") and cols.get("pwd") and cols.get("teff_h"):
            return t, cols
    return None, {}


def build_wd_query(table: str, wd_cols: dict, wise_cols: dict, tmass_cols: dict,
                   tmass_id: str | None, dec_lo: float, dec_hi: float, *,
                   pwd_min: float, poe_min: float, limit: int) -> str:
    wsel = ", ".join(f"w.{a} AS {logical.lower()}"
                     for logical, a in wise_cols.items() if logical != "designation")
    wdsel = ", ".join(f"wd.{a} AS {logical}" for logical, a in wd_cols.items()
                      if logical != "wd_source_id")
    tsel, tjoin = "", ""
    if tmass_cols and tmass_id:
        tsel = ", " + ", ".join(
            f"t.{a} AS {logical.lower()}" for logical, a in tmass_cols.items()
            if logical != "tmass_designation")
        tjoin = f"""
  LEFT OUTER JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xt ON xt.source_id = g.source_id
  LEFT OUTER JOIN gaiadr1.tmass_original_valid AS t ON t.designation = xt.{tmass_id}"""
    return f"""
SELECT TOP {int(limit)}
  g.source_id, g.ra, g.dec, g.l, g.b, g.parallax, g.parallax_error,
  g.parallax_over_error, g.pmra, g.pmra_error, g.pmdec, g.pmdec_error,
  g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.bp_rp, g.ruwe,
  g.astrometric_excess_noise, g.phot_bp_rp_excess_factor,
  {wdsel},
  xw.angular_distance AS wise_angdist, xw.number_of_neighbours, xw.number_of_mates,
  {wsel}{tsel}
FROM {table} AS wd
  JOIN gaiadr3.gaia_source AS g ON g.source_id = wd.{wd_cols['wd_source_id']}
  JOIN gaiadr3.allwise_best_neighbour AS xw ON xw.source_id = g.source_id
  JOIN gaiadr1.allwise_original_valid AS w
       ON w.{wise_cols['designation']} = xw.original_ext_source_id{tjoin}
WHERE g.dec >= {dec_lo} AND g.dec < {dec_hi}
  AND wd.{wd_cols['pwd']} >= {pwd_min}
  AND g.parallax_over_error > {poe_min}
"""


def fetch_wd_gaia(out_dir: Path, cfg: dict, *, query=gaia_query, probe=probe_columns,
                  dec_band: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Route A, checkpointed per declination band.  Raises if the table is absent."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w = cfg["wd"]
    table, wd_cols = probe_gaia_wd_table(w["gaia_table_candidates"], probe)
    if table is None:
        raise RuntimeError("no Gaia-archive white-dwarf table among "
                           f"{w['gaia_table_candidates']}")
    wise_cols = probe("gaiadr1.allwise_original_valid", _ALLWISE_WANT, tag="ring/wd")
    if not wise_cols.get("designation") or not wise_cols.get("W1mag"):
        raise RuntimeError("AllWISE mirror unusable: no designation/W1 column")
    tmass_cols = probe("gaiadr1.tmass_original_valid", _TMASS_WANT, tag="ring/wd")
    xt = probe("gaiadr3.tmass_psc_xsc_best_neighbour", {"id": _TMASS_XMATCH_ID}, tag="ring/wd")
    tmass_id = xt.get("id")
    width = int(w.get("dec_band_deg", 15))
    edges = list(range(-90, 91, width))
    bands = list(zip(edges[:-1], edges[1:], strict=False))
    if dec_band is not None:
        bands = [bands[int(dec_band)]]
    frames, meta = [], {"route": "gaia_archive_join", "table": table, "bands": [],
                        "tmass_degraded": False}
    for lo, hi in bands:
        part = out_dir / f"wd_dec_{lo:+04d}_{hi:+04d}.parquet"
        if part.exists():
            df = pd.read_parquet(part)
            frames.append(df)
            meta["bands"].append({"dec": [lo, hi], "n": int(len(df)), "cached": True})
            continue
        kw = dict(pwd_min=float(w["pwd_min"]), poe_min=float(w["parallax_over_error_min"]),
                  limit=int(w["limit_per_band"]))
        t0 = time.monotonic()
        try:
            q = build_wd_query(table, wd_cols, wise_cols, tmass_cols, tmass_id, lo, hi, **kw)
            df = query(q, tag="ring/wd")
        except Exception as exc:                        # noqa: BLE001
            print(f"[ring/wd] dec [{lo},{hi}) with 2MASS failed ({exc!r}); "
                  f"retrying without the 2MASS join", flush=True)
            meta["tmass_degraded"] = True
            q = build_wd_query(table, wd_cols, wise_cols, {}, None, lo, hi, **kw)
            df = query(q, tag="ring/wd")
        df["row_limit_hit"] = len(df) >= int(w["limit_per_band"])
        df["dec_band_lo"] = lo
        df.to_parquet(part, index=False)
        meta["bands"].append({"dec": [lo, hi], "n": int(len(df)),
                              "elapsed_s": round(time.monotonic() - t0, 1),
                              "row_limit_hit": bool(len(df) >= int(w["limit_per_band"]))})
        print(f"[ring/wd] dec [{lo:+d},{hi:+d}): {len(df)} white dwarfs "
              f"({time.monotonic() - t0:.0f} s)", flush=True)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).drop_duplicates("source_id") if frames \
        else pd.DataFrame()
    return out, meta


# --------------------------------------------------------------------------
# White dwarfs: VizieR parent + archive upload join (route B) / X-Match (route C)
# --------------------------------------------------------------------------

# Logical column -> candidate VizieR spellings, in priority order.  Run
# 35752692549 lost the whole white-dwarf leg to two of these: the hard-coded
# SELECT named "chi2H" / "chi2He", which J/MNRAS/508/3877/maincat does not
# have, and TAPVizieR refuses a query with ANY unknown column.  The columns are
# therefore resolved against the table's own TAP_SCHEMA listing at run time;
# an optional column that is absent is simply not selected (the H/He choice
# then falls back to "H unless only He is fitted"), and only a missing REQUIRED
# column fails the route -- with the listing it saw in the error.
WD_VIZIER_COLUMNS = {
    "source_id": ["GaiaEDR3", "Source", "source_id", "GaiaDR3"],
    "ra": ["RA_ICRS", "RAJ2000", "RAdeg"],
    "dec": ["DE_ICRS", "DEJ2000", "DEdeg"],
    "parallax": ["Plx", "plx"],
    "parallax_error": ["e_Plx", "e_plx"],
    "pmra": ["pmRA"],
    "pmdec": ["pmDE"],
    "pwd": ["Pwd", "PWD", "pwd"],
    "teff_h": ["TeffH", "Teff-H", "TeffH2"],
    "logg_h": ["loggH", "logg-H"],
    "mass_h": ["MassH", "Mass-H"],
    "chisq_h": ["chi2H", "chisqH", "Chi2H", "chiH", "chi2-H"],
    "teff_he": ["TeffHe", "Teff-He"],
    "logg_he": ["loggHe", "logg-He"],
    "mass_he": ["MassHe", "Mass-He"],
    "chisq_he": ["chi2He", "chisqHe", "Chi2He", "chiHe", "chi2-He"],
    "phot_g_mean_mag": ["Gmag", "GmagCorr"],
    "phot_bp_mean_mag": ["BPmag"],
    "phot_rp_mean_mag": ["RPmag"],
    "ruwe": ["RUWE", "ruwe"],
}
WD_VIZIER_REQUIRED = ("source_id", "ra", "dec", "parallax", "parallax_error", "pwd", "teff_h")


def resolve_wd_vizier_columns(columns) -> dict:
    """Map each logical WD column to the spelling this VizieR table actually has."""
    have = [str(c).strip().strip('"') for c in columns]
    exact = set(have)
    low = {c.lower(): c for c in have}
    out = {}
    for logical, cands in WD_VIZIER_COLUMNS.items():
        for c in cands:
            if c in exact:
                out[logical] = c
                break
            if c.lower() in low:
                out[logical] = low[c.lower()]
                break
    return out


def fetch_wd_vizier_parent(cfg: dict, *, query_fn=None, columns_fn=None,
                           max_rows: int | None = None) -> pd.DataFrame:
    """Gentile Fusillo+2021 parent from VizieR, Pwd-filtered, in declination bands.

    Columns come from the table's own listing (``columns_fn``), never from an
    assumed spelling; the pull is split into declination bands so no single
    TAP response has to carry the ~360 k-row parent, and each band's row count
    is kept so a server-side row cap would be visible rather than silent.
    """
    from ..metronome.acquire import table_columns, tap_query

    query_fn = query_fn or tap_query
    w = cfg["wd"]
    table = w["vizier_parent"]
    if columns_fn is None:
        def columns_fn(t):
            return table_columns(t, query_fn=query_fn)
    cols = resolve_wd_vizier_columns(columns_fn(table))
    missing = [k for k in WD_VIZIER_REQUIRED if k not in cols]
    if missing:
        raise RuntimeError(f"{table}: required columns {missing} not in the table listing "
                           f"(resolved: {sorted(cols)})")
    sel = ", ".join(f'"{c}"' for c in cols.values())
    width = int(w.get("vizier_dec_band_deg", 30))
    edges = list(range(-90, 91, width))
    if edges[-1] != 90:
        edges.append(90)
    frames, bands = [], []
    dec_c = cols["dec"]
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        top = f"TOP {int(max_rows)} " if max_rows else ""
        upper = f'"{dec_c}" <= {hi}' if hi == 90 else f'"{dec_c}" < {hi}'
        adql = (f'SELECT {top}{sel} FROM "{table}" WHERE "{cols["pwd"]}" >= '
                f'{float(w["pwd_min"])} AND "{dec_c}" >= {lo} AND {upper}')
        raw = query_fn(adql)
        raw = raw.rename(columns={c: str(c).strip('"') for c in raw.columns})
        bands.append({"dec": [lo, hi], "n": int(len(raw)),
                      "route": str(getattr(raw, "attrs", {}).get("route", "vizier"))})
        print(f"[ring/wd] VizieR parent dec [{lo:+d},{hi:+d}): {len(raw)} rows", flush=True)
        frames.append(raw)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    inv = {v: k for k, v in cols.items()}
    out = raw.rename(columns={c: inv[c] for c in raw.columns if c in inv})
    if len(out):
        out = out.drop_duplicates("source_id").reset_index(drop=True)
    if {"parallax", "parallax_error"} <= set(out.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            out["parallax_over_error"] = pd.to_numeric(out["parallax"], errors="coerce") / \
                pd.to_numeric(out["parallax_error"], errors="coerce")
    if {"phot_bp_mean_mag", "phot_rp_mean_mag"} <= set(out.columns):
        out["bp_rp"] = pd.to_numeric(out["phot_bp_mean_mag"], errors="coerce") - \
            pd.to_numeric(out["phot_rp_mean_mag"], errors="coerce")
    out.attrs["route"] = bands[0]["route"] if bands else "vizier"
    out.attrs["bands"] = bands
    out.attrs["columns"] = cols
    return out


def gaia_upload_join_allwise(ids: pd.DataFrame, *, query=gaia_query, probe=probe_columns,
                             chunk: int = 50_000) -> pd.DataFrame:
    """Route B: the archive's PM-propagated AllWISE cross-match for uploaded ids."""
    from astropy.table import Table

    wise_cols = probe("gaiadr1.allwise_original_valid", _ALLWISE_WANT, tag="ring/wd")
    if not wise_cols.get("designation"):
        raise RuntimeError("AllWISE mirror unusable")
    wsel = ", ".join(f"w.{a} AS {logical.lower()}"
                     for logical, a in wise_cols.items() if logical != "designation")
    # 2MASS through the archive's own best-neighbour table, as in route A: the
    # J/H/Ks anchor carries half the photosphere systematic of the Gaia one.
    try:
        tmass_cols = probe("gaiadr1.tmass_original_valid", _TMASS_WANT, tag="ring/wd")
        tmass_id = probe("gaiadr3.tmass_psc_xsc_best_neighbour", {"id": _TMASS_XMATCH_ID},
                         tag="ring/wd").get("id")
    except Exception:                                   # noqa: BLE001
        tmass_cols, tmass_id = {}, None
    tsel, tjoin = "", ""
    if tmass_cols and tmass_id:
        tsel = ", " + ", ".join(f"t.{a} AS {logical.lower()}"
                                for logical, a in tmass_cols.items()
                                if logical != "tmass_designation")
        tjoin = f"""
  LEFT OUTER JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xt ON xt.source_id = g.source_id
  LEFT OUTER JOIN gaiadr1.tmass_original_valid AS t ON t.designation = xt.{tmass_id}"""

    def _q(ts, tj):
        return f"""
SELECT u.source_id, g.l, g.b, g.ra, g.dec, g.pmra, g.pmdec, g.pmra_error, g.pmdec_error,
       g.phot_g_mean_mag, g.bp_rp, g.ruwe, g.astrometric_excess_noise,
       xw.angular_distance AS wise_angdist, xw.number_of_neighbours, xw.number_of_mates,
       {wsel}{ts}
FROM tap_upload.ids AS u
  JOIN gaiadr3.gaia_source AS g ON g.source_id = u.source_id
  JOIN gaiadr3.allwise_best_neighbour AS xw ON xw.source_id = g.source_id
  JOIN gaiadr1.allwise_original_valid AS w
       ON w.{wise_cols['designation']} = xw.original_ext_source_id{tj}
"""
    frames = []
    for start in range(0, len(ids), chunk):
        sub = ids.iloc[start:start + chunk][["source_id"]].astype({"source_id": "int64"})
        tbl = Table.from_pandas(sub)
        try:
            df = query(_q(tsel, tjoin), tag="ring/wd-upload", upload=tbl, upload_name="ids")
        except Exception as exc:                        # noqa: BLE001
            if not tsel:
                raise
            print(f"[ring/wd] upload join with 2MASS failed ({exc!r}); retrying without",
                  flush=True)
            tsel, tjoin = "", ""
            df = query(_q("", ""), tag="ring/wd-upload", upload=tbl, upload_name="ids")
        frames.append(df)
        print(f"[ring/wd] upload join {start + len(sub)}/{len(ids)}: {len(df)} matched",
              flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def xmatch_at_epoch(positions: pd.DataFrame, vizier_table: str, radius_arcsec: float,
                    from_epoch: float, to_epoch: float, xmatch_fn=None) -> pd.DataFrame:
    """Route C: CDS X-Match at positions propagated from ``from_epoch`` to ``to_epoch``.

    The proper motion is applied BEFORE the upload: a 500 mas/yr white dwarf
    moves 2.75 arcsec between the Gaia and AllWISE epochs, more than any sane
    match radius, and the previous channel that skipped this step returned
    nothing for exactly its fastest stars.
    """
    from ..acquire.science import _xmatch

    xmatch_fn = xmatch_fn or _xmatch
    pos = positions.copy()
    pmra = pd.to_numeric(pos.get("pmra", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
    pmdec = pd.to_numeric(pos.get("pmdec", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
    ra, dec = propagate_pm(pos["ra"].to_numpy(float), pos["dec"].to_numpy(float),
                           pmra, pmdec, from_epoch, to_epoch)
    up = pd.DataFrame({"source_id": pos["source_id"].astype(str).to_numpy(),
                       "ra": ra, "dec": dec})
    return xmatch_fn(up, vizier_table, radius_arcsec)


ALLWISE_XMATCH_RENAME = {
    "AllWISE": "designation", "RAJ2000": "ra_wise", "DEJ2000": "dec_wise",
    "W1mag": "W1mag", "e_W1mag": "e_W1mag", "W2mag": "W2mag", "e_W2mag": "e_W2mag",
    "W3mag": "W3mag", "e_W3mag": "e_W3mag", "W4mag": "W4mag", "e_W4mag": "e_W4mag",
    "ccf": "cc_flags", "qph": "ph_qual", "ex": "ext_flag", "nb": "number_of_neighbours",
    "na": "number_of_mates", "pmRA": "pmra_wise", "pmDE": "pmdec_wise",
    "e_pmRA": "e_pmra_wise", "e_pmDE": "e_pmdec_wise", "angDist": "match_dist_arcsec",
    "d2d": "match_dist_arcsec", "Jmag": "Jmag", "Hmag": "Hmag", "Kmag": "Ksmag",
    "e_Jmag": "e_Jmag", "e_Hmag": "e_Hmag", "e_Kmag": "e_Ksmag",
}
CATWISE_XMATCH_RENAME = {
    "Name": "catwise_name", "RA_ICRS": "ra_catwise", "DE_ICRS": "dec_catwise",
    "RAPMdeg": "ra_catwise", "DEPMdeg": "dec_catwise",
    "W1mproPM": "W1mag_cat", "e_W1mproPM": "e_W1mag_cat",
    "W2mproPM": "W2mag_cat", "e_W2mproPM": "e_W2mag_cat",
    "pmRA": "pmra_catwise", "pmDE": "pmdec_catwise", "e_pmRA": "e_pmra_catwise",
    "e_pmDE": "e_pmdec_catwise", "ccf": "cc_flags_cat", "qph": "ph_qual_cat",
    "angDist": "match_dist_arcsec", "d2d": "match_dist_arcsec",
}


def normalise_xmatch(raw: pd.DataFrame, rename: dict) -> pd.DataFrame:
    """Rename an X-Match result to the pipeline schema, keeping the nearest match."""
    if raw is None or not len(raw):
        return pd.DataFrame()
    out = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns}).copy()
    if "match_dist_arcsec" in out.columns:
        out = out.sort_values("match_dist_arcsec").drop_duplicates("source_id")
    for col in ("pmra_wise", "pmdec_wise", "e_pmra_wise", "e_pmdec_wise",
                "pmra_catwise", "pmdec_catwise", "e_pmra_catwise", "e_pmdec_catwise"):
        if col in out.columns:
            # CatWISE2020 / AllWISE on VizieR report proper motions in arcsec/yr.
            out[col] = pd.to_numeric(out[col], errors="coerce") * 1000.0
    return out.reset_index(drop=True)


def fetch_wd_leg(out_dir: Path, cfg: dict, *, query=gaia_query, probe=probe_columns,
                 vizier_fn=None, columns_fn=None, xmatch_fn=None, dec_band: int | None = None
                 ) -> tuple[pd.DataFrame, dict]:
    """The white-dwarf sample with AllWISE photometry, by the first route that works."""
    out_dir = Path(out_dir)
    meta: dict = {"routes_tried": []}
    try:
        df, m = fetch_wd_gaia(out_dir, cfg, query=query, probe=probe, dec_band=dec_band)
        meta.update(m)
        meta["routes_tried"].append({"route": "gaia_archive_join", "status": "OK",
                                     "n": int(len(df))})
        if len(df):
            return df, meta
    except Exception as exc:                            # noqa: BLE001
        meta["routes_tried"].append({"route": "gaia_archive_join", "status": "FAILED",
                                     "error": repr(exc)[:300]})
        print(f"[ring/wd] route A failed: {exc!r}", flush=True)
    if dec_band is not None:
        # A sharded call has no business re-pulling the whole parent per shard.
        meta["route"] = "none"
        return pd.DataFrame(), meta

    try:
        parent = fetch_wd_vizier_parent(cfg, query_fn=vizier_fn, columns_fn=columns_fn)
        meta["routes_tried"].append({"route": "vizier_parent", "status": "OK",
                                     "n": int(len(parent)),
                                     "served_by": parent.attrs.get("route"),
                                     "columns": parent.attrs.get("columns"),
                                     "bands": parent.attrs.get("bands")})
    except Exception as exc:                            # noqa: BLE001
        meta["routes_tried"].append({"route": "vizier_parent", "status": "FAILED",
                                     "error": repr(exc)[:300]})
        meta["route"] = "none"
        return pd.DataFrame(), meta
    parent = parent[pd.to_numeric(parent.get("parallax_over_error"), errors="coerce")
                    > float(cfg["wd"]["parallax_over_error_min"])]
    parent.to_parquet(out_dir / "wd_parent_vizier.parquet", index=False)

    try:
        joined = gaia_upload_join_allwise(parent[["source_id"]].dropna(), query=query,
                                          probe=probe)
        if len(joined):
            # Keep the parent's copy of every shared column (Gaia photometry,
            # astrometry): a plain merge would suffix both to _x/_y and the
            # harmoniser would then find no G magnitude at all.
            joined = joined.assign(source_id=pd.to_numeric(joined["source_id"],
                                                           errors="coerce").astype("Int64"))
            parent = parent.assign(source_id=pd.to_numeric(parent["source_id"],
                                                           errors="coerce").astype("Int64"))
            df = parent.merge(joined.drop(columns=[c for c in joined.columns
                                                   if c in parent.columns and c != "source_id"]),
                              on="source_id", how="inner")
            meta["routes_tried"].append({"route": "gaia_upload_join", "status": "OK",
                                         "n": int(len(df))})
            meta["route"] = "vizier_parent+gaia_upload_join"
            df.to_parquet(out_dir / "wd_routeB.parquet", index=False)
            return df, meta
        meta["routes_tried"].append({"route": "gaia_upload_join", "status": "ZERO_ROWS"})
    except Exception as exc:                            # noqa: BLE001
        meta["routes_tried"].append({"route": "gaia_upload_join", "status": "FAILED",
                                     "error": repr(exc)[:300]})
        print(f"[ring/wd] route B failed: {exc!r}", flush=True)

    try:
        ep = cfg["epochs"]
        raw = xmatch_at_epoch(parent, "vizier:II/328/allwise", 3.0, float(ep["gaia"]),
                              float(ep["allwise"]), xmatch_fn=xmatch_fn)
        wise = normalise_xmatch(raw, ALLWISE_XMATCH_RENAME)
        if len(wise):
            wise["source_id"] = wise["source_id"].astype(str)
            p = parent.copy()
            p["source_id"] = p["source_id"].astype(str)
            df = p.merge(wise, on="source_id", how="inner")
            meta["routes_tried"].append({"route": "cds_xmatch_propagated", "status": "OK",
                                         "n": int(len(df))})
            meta["route"] = "vizier_parent+cds_xmatch_propagated"
            df.to_parquet(out_dir / "wd_routeC.parquet", index=False)
            return df, meta
        meta["routes_tried"].append({"route": "cds_xmatch_propagated", "status": "ZERO_ROWS"})
    except Exception as exc:                            # noqa: BLE001
        meta["routes_tried"].append({"route": "cds_xmatch_propagated", "status": "FAILED",
                                     "error": repr(exc)[:300]})
    meta["route"] = "none"
    return pd.DataFrame(), meta


def _numcol(df: pd.DataFrame, col: str) -> pd.Series:
    """Float Series for ``col``; all-NaN when absent (route B has no chi-squares)."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def harmonise_wd(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline schema: WISE/2MASS/Gaia magnitude columns, a single ``teff``."""
    out = df.rename(columns={
        "w1mag": "W1mag", "e_w1mag": "e_W1mag", "w2mag": "W2mag", "e_w2mag": "e_W2mag",
        "w3mag": "W3mag", "e_w3mag": "e_W3mag", "w4mag": "W4mag", "e_w4mag": "e_W4mag",
        "jmag": "Jmag", "e_jmag": "e_Jmag", "hmag": "Hmag", "e_hmag": "e_Hmag",
        "ksmag": "Ksmag", "e_ksmag": "e_Ksmag",
    }).copy()
    for src, dst in (("phot_g_mean_mag", "Gmag"), ("phot_bp_mean_mag", "BPmag"),
                     ("phot_rp_mean_mag", "RPmag")):
        if src in out.columns and dst not in out.columns:
            out[dst] = pd.to_numeric(out[src], errors="coerce")
    for b in ("G", "BP", "RP"):
        if f"{b}mag" in out.columns and f"e_{b}mag" not in out.columns:
            out[f"e_{b}mag"] = 0.02
    th = _numcol(out, "teff_h")
    the = _numcol(out, "teff_he")
    ch = _numcol(out, "chisq_h")
    che = _numcol(out, "chisq_he")
    use_he = (the.notna() & (th.isna() | (che.notna() & ch.notna() & (che < 0.5 * ch))))
    out["atmosphere"] = np.where(use_he, "He", "H")
    out["teff"] = np.where(use_he, the, th)
    out["logg"] = np.where(use_he, _numcol(out, "logg_he"),
                           _numcol(out, "logg_h"))
    out["mass"] = np.where(use_he, _numcol(out, "mass_he"),
                           _numcol(out, "mass_h"))
    plx = _numcol(out, "parallax")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["dist_pc"] = np.where(plx > 0, 1000.0 / plx, np.nan)
    return out


# --------------------------------------------------------------------------
# Pulsars: ATNF
# --------------------------------------------------------------------------

_PSR_FIELDS = ("PSRJ", "PSRB", "RAJ", "DECJ", "ELONG", "ELAT", "PMRA", "PMDEC", "PMELONG",
               "PMELAT", "POSEPOCH", "PEPOCH", "P0", "P1", "F0", "F1", "DIST", "DIST_A",
               "DIST_DM", "DIST_DM1", "DIST1", "ASSOC", "BINARY", "BINCOMP", "TYPE", "PX")


def _sexagesimal(s: str, hours: bool) -> tuple[float, float]:
    """``hh:mm:ss.s`` -> (degrees, error unit of the last digit in arcsec)."""
    parts = s.strip().split(":")
    vals = [float(p) for p in parts]
    sign = -1.0 if s.strip().startswith("-") else 1.0
    vals = [abs(v) for v in vals]
    deg = vals[0] + (vals[1] if len(vals) > 1 else 0.0) / 60.0 \
        + (vals[2] if len(vals) > 2 else 0.0) / 3600.0
    if hours:
        deg *= 15.0
    last = parts[-1]
    nd = len(last.split(".")[1]) if "." in last else 0
    if len(parts) == 3:
        unit = 10.0 ** (-nd) * (15.0 if hours else 1.0)
    elif len(parts) == 2:
        unit = 60.0 * (15.0 if hours else 1.0)
    else:
        unit = 3600.0 * (15.0 if hours else 1.0)
    return sign * deg, unit


def parse_psrcat_db(text: str) -> pd.DataFrame:
    """Parse ``psrcat.db``: ``PARAM value [error] [ref]`` lines, ``@---`` separators."""
    rows, cur = [], {}
    for line in text.splitlines():
        if line.startswith("@-"):
            if cur:
                rows.append(cur)
            cur = {}
            continue
        if not line.strip() or line.startswith("#"):
            continue
        tok = line.split()
        key = tok[0].upper()
        if key not in _PSR_FIELDS or len(tok) < 2:
            continue
        cur[key] = tok[1]
        if len(tok) >= 3 and re.fullmatch(r"[0-9.]+(E[-+]?\d+)?", tok[2], re.I):
            cur[key + "_ERR"] = tok[2]
    if cur:
        rows.append(cur)
    return pd.DataFrame(rows)


def _decimal_unit_deg(s: pd.Series) -> np.ndarray:
    """Value of one unit in the last quoted decimal place of a degree string."""
    out = []
    for v in s.astype(str):
        v = v.strip()
        if not v or v.lower() in ("nan", "none", "<na>"):
            out.append(np.nan)
            continue
        mant = re.split(r"[eEdD]", v)[0]
        nd = len(mant.split(".")[1]) if "." in mant else 0
        out.append(10.0 ** (-nd))
    return np.asarray(out, float)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("D", "E", regex=False), errors="coerce")


def normalise_pulsars(raw: pd.DataFrame, route: str) -> pd.DataFrame:
    """Uniform pulsar table: degrees, mas/yr, seconds, kpc, Julian-year epochs."""
    cols = {c.upper(): c for c in raw.columns}

    def col(name, default=np.nan):
        c = cols.get(name.upper())
        return raw[c] if c is not None else pd.Series(default, index=raw.index)

    out = pd.DataFrame(index=raw.index)
    out["jname"] = col("PSRJ", "").astype(str) if "PSRJ" in cols else \
        col("JNAME", "").astype(str)
    out["bname"] = col("PSRB", "").astype(str) if "PSRB" in cols else \
        col("BNAME", "").astype(str)
    out["bname"] = out["bname"].replace({"nan": "", "None": ""})

    ra = np.full(len(raw), np.nan)
    dec = np.full(len(raw), np.nan)
    ra_unit = np.full(len(raw), np.nan)
    dec_unit = np.full(len(raw), np.nan)
    ecl_err = np.full(len(raw), np.nan)
    if "RAJD" in cols and "DECJD" in cols:               # psrqpy: already degrees
        ra = _num(col("RAJD")).to_numpy(float)
        dec = _num(col("DECJD")).to_numpy(float)
    if "RAJ" in cols and "DECJ" in cols:
        for i, (r, d) in enumerate(zip(col("RAJ"), col("DECJ"), strict=False)):
            try:
                if isinstance(r, str) and ":" in r and isinstance(d, str) and ":" in d:
                    ra[i], ra_unit[i] = _sexagesimal(r, hours=True)
                    dec[i], dec_unit[i] = _sexagesimal(d, hours=False)
            except ValueError:
                continue
    if "ELONG" in cols and "ELAT" in cols:
        missing = ~np.isfinite(ra)
        if missing.any():
            try:
                from astropy import units as u
                from astropy.coordinates import (
                    ICRS,
                    BarycentricTrueEcliptic,
                    SkyCoord,
                )
                el = _num(col("ELONG")).to_numpy(float)
                eb = _num(col("ELAT")).to_numpy(float)
                ok = missing & np.isfinite(el) & np.isfinite(eb)
                if ok.any():
                    c = SkyCoord(el[ok] * u.deg, eb[ok] * u.deg,
                                 frame=BarycentricTrueEcliptic(equinox="J2000"))
                    ic = c.transform_to(ICRS())
                    ra[ok] = ic.ra.deg
                    dec[ok] = ic.dec.deg
                    # The error is the catalogue's own: ELONG/ELAT are quoted in
                    # degrees to as many decimals as the timing solution
                    # supports, with an optional error in units of the last
                    # digit.  Run 35752692549 assumed 0.01 deg (36 arcsec) for
                    # every ecliptic position, which gave 244 timed pulsars --
                    # the MSP J1453+1902 among them -- a 49 arcsec error, an
                    # 8 arcsec match radius and a spurious "counterpart".
                    el_u = _decimal_unit_deg(col("ELONG")) * 3600.0
                    eb_u = _decimal_unit_deg(col("ELAT")) * 3600.0
                    el_e = _num(col("ELONG_ERR")).to_numpy(float) if "ELONG_ERR" in cols \
                        else np.full(len(raw), np.nan)
                    eb_e = _num(col("ELAT_ERR")).to_numpy(float) if "ELAT_ERR" in cols \
                        else np.full(len(raw), np.nan)
                    el_as = np.where(np.isfinite(el_e), el_e * el_u, el_u) \
                        * np.cos(np.radians(np.nan_to_num(eb)))
                    eb_as = np.where(np.isfinite(eb_e), eb_e * eb_u, eb_u)
                    ecl_err[ok] = np.hypot(el_as, eb_as)[ok]
            except Exception as exc:                    # noqa: BLE001
                print(f"[ring/psr] ecliptic conversion skipped: {exc!r}", flush=True)
    out["ra"], out["dec"] = ra, dec

    # Position error: the quoted last-digit error times its unit, else the unit
    # itself (a position given to 0.1 s of time is known to ~1.5 arcsec).
    ra_err = _num(col("RAJ_ERR")).to_numpy(float) if "RAJ_ERR" in cols else \
        np.full(len(raw), np.nan)
    dec_err = _num(col("DECJ_ERR")).to_numpy(float) if "DECJ_ERR" in cols else \
        np.full(len(raw), np.nan)
    if route == "psrqpy":
        # psrqpy reports RAJ_ERR in seconds of time and DECJ_ERR in arcsec.
        ra_as = ra_err * 15.0 * np.cos(np.radians(np.nan_to_num(dec)))
        dec_as = dec_err
    else:
        ra_as = np.where(np.isfinite(ra_err), ra_err * ra_unit, ra_unit) \
            * np.cos(np.radians(np.nan_to_num(dec)))
        dec_as = np.where(np.isfinite(dec_err), dec_err * dec_unit, dec_unit)
    out["pos_err_arcsec"] = np.where(
        np.isfinite(ecl_err), ecl_err,
        np.hypot(np.nan_to_num(ra_as, nan=1.0), np.nan_to_num(dec_as, nan=1.0)))

    out["pmra"] = _num(col("PMRA")).to_numpy(float)
    out["pmdec"] = _num(col("PMDEC")).to_numpy(float)
    pe = _num(col("POSEPOCH")).to_numpy(float)
    pe2 = _num(col("PEPOCH")).to_numpy(float)
    mjd = np.where(np.isfinite(pe), pe, pe2)
    out["posepoch_yr"] = 2000.0 + (mjd - 51544.5) / 365.25
    out.loc[~np.isfinite(out["posepoch_yr"]), "posepoch_yr"] = 2000.0
    p0 = _num(col("P0")).to_numpy(float)
    p1 = _num(col("P1")).to_numpy(float)
    f0 = _num(col("F0")).to_numpy(float)
    f1 = _num(col("F1")).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p0 = np.where(np.isfinite(p0), p0, 1.0 / f0)
        p1 = np.where(np.isfinite(p1), p1, -f1 / f0 ** 2)
    out["p0_s"], out["p1"] = p0, p1
    dist = np.full(len(raw), np.nan)
    for name in ("DIST", "DIST_A", "DIST_DM1", "DIST_DM", "DIST1"):
        if name in cols:
            v = _num(col(name)).to_numpy(float)
            dist = np.where(np.isfinite(dist), dist, v)
    out["dist_kpc"] = dist
    for name, dst in (("ASSOC", "assoc"), ("BINARY", "binary"), ("BINCOMP", "bincomp"),
                      ("TYPE", "ptype")):
        out[dst] = col(name, "").astype(str).replace({"nan": "", "None": "", "*": ""})
    out["route"] = route
    out = out[np.isfinite(out["ra"]) & np.isfinite(out["dec"]) & (out["jname"] != "")]
    return out.drop_duplicates("jname").reset_index(drop=True)


def fetch_pulsars_psrqpy() -> pd.DataFrame:
    from psrqpy import QueryATNF

    q = QueryATNF(params=["JNAME", "BNAME", "RAJ", "DECJ", "RAJD", "DECJD", "PMRA", "PMDEC",
                          "POSEPOCH", "PEPOCH", "P0", "P1", "F0", "F1", "DIST", "DIST_A",
                          "DIST_DM", "ASSOC", "BINARY", "BINCOMP", "TYPE", "PX"])
    df = q.table.to_pandas()
    df = df.rename(columns={"JNAME": "PSRJ", "BNAME": "PSRB"})
    # psrqpy exposes the version as a property on current releases and as a
    # method on old ones; calling the property string raised "'str' object is
    # not callable" in run 35752692549 and threw away a successful query.
    v = getattr(q, "get_version", "")
    df.attrs["catalogue_version"] = str(v() if callable(v) else v)
    return df


def fetch_pulsars_tarball(url: str, fetch_bytes=None) -> pd.DataFrame:
    """Download the ATNF catalogue package and parse ``psrcat.db`` from it."""
    if fetch_bytes is None:
        import requests

        def fetch_bytes(u):
            r = requests.get(u, timeout=300)
            r.raise_for_status()
            return r.content
    blob = fetch_bytes(url)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        member = next((m for m in tf.getmembers() if m.name.endswith("psrcat.db")), None)
        if member is None:
            raise RuntimeError("psrcat.db not in the tarball")
        text = tf.extractfile(member).read().decode("utf-8", errors="replace")
    df = parse_psrcat_db(text)
    m = re.search(r"CATALOGUE_VERSION\s+([\d.]+)", text)
    df.attrs["catalogue_version"] = m.group(1) if m else ""
    return df


def fetch_pulsars_vizier(table: str, query_fn=None) -> pd.DataFrame:
    from ..metronome.acquire import tap_query

    query_fn = query_fn or tap_query
    df = query_fn(f'SELECT * FROM "{table}"')
    ren = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("psrj", "jname"):
            ren[c] = "PSRJ"
        elif cl in ("psrb", "bname"):
            ren[c] = "PSRB"
        elif cl in ("raj2000", "_ra", "ra_icrs", "radeg"):
            ren[c] = "RAJD"
        elif cl in ("dej2000", "_de", "de_icrs", "dedeg"):
            ren[c] = "DECJD"
        elif cl == "pmra":
            ren[c] = "PMRA"
        elif cl in ("pmde", "pmdec"):
            ren[c] = "PMDEC"
        elif cl in ("p0", "per", "period"):
            ren[c] = "P0"
        elif cl in ("p1", "pdot"):
            ren[c] = "P1"
        elif cl in ("dist", "distance", "d"):
            ren[c] = "DIST"
        elif cl in ("assoc", "association"):
            ren[c] = "ASSOC"
        elif cl == "binary":
            ren[c] = "BINARY"
        elif cl == "type":
            ren[c] = "TYPE"
    return df.rename(columns=ren)


def fetch_pulsars(cfg: dict, *, fetchers: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """The pulsar host list by the first route that answers, with the route named."""
    p = cfg["pulsar"]
    fetchers = fetchers or {
        "psrqpy": fetch_pulsars_psrqpy,
        "tarball": lambda: fetch_pulsars_tarball(p["tarball_url"]),
        "vizier": lambda: fetch_pulsars_vizier(p["vizier_table"]),
    }
    meta: dict = {"routes_tried": []}
    for route in p["routes"]:
        fn = fetchers.get(route)
        if fn is None:
            continue
        t0 = time.monotonic()
        try:
            raw = fn()
            df = normalise_pulsars(raw, route)
            meta["routes_tried"].append({"route": route, "status": "OK" if len(df) else
                                         "ZERO_ROWS", "n": int(len(df)),
                                         "elapsed_s": round(time.monotonic() - t0, 1),
                                         "catalogue_version":
                                         str(getattr(raw, "attrs", {}).get(
                                             "catalogue_version", ""))})
            if len(df):
                meta["route"] = route
                return df, meta
        except Exception as exc:                        # noqa: BLE001
            meta["routes_tried"].append({"route": route, "status": "FAILED",
                                         "error": repr(exc)[:300],
                                         "elapsed_s": round(time.monotonic() - t0, 1)})
            print(f"[ring/psr] route {route} failed: {exc!r}", flush=True)
    meta["route"] = "none"
    return pd.DataFrame(), meta


def pulsar_match_positions(psr: pd.DataFrame, cfg: dict, to_epoch: float) -> pd.DataFrame:
    """Target positions at ``to_epoch`` plus two rings of eight control offsets.

    The control positions are the chance-match measurement: a match rate at
    60-90 arcsec from each pulsar, on the same sky, in the same catalogue, at
    the same radius, is the only honest denominator for a counterpart at a
    position known to a few arcsec.  ``source_id`` carries ``<jname>|t`` for the
    target and ``<jname>|c<k>`` for the controls.
    """
    p = cfg["pulsar"]
    pmra = pd.to_numeric(psr["pmra"], errors="coerce").fillna(0.0).to_numpy(float)
    pmdec = pd.to_numeric(psr["pmdec"], errors="coerce").fillna(0.0).to_numpy(float)
    # Each pulsar has its own position epoch, so the propagation is per object.
    dt = float(to_epoch) - pd.to_numeric(psr["posepoch_yr"], errors="coerce") \
        .fillna(2000.0).to_numpy(float)
    ra0 = psr["ra"].to_numpy(float)
    dec0 = psr["dec"].to_numpy(float)
    cosd = np.cos(np.radians(dec0))
    cosd = np.where(np.abs(cosd) < 1e-6, 1e-6, cosd)
    ra = ra0 + (pmra * dt / 3.6e6) / cosd
    dec = dec0 + pmdec * dt / 3.6e6
    rows = [pd.DataFrame({"source_id": psr["jname"].astype(str) + "|t", "ra": ra, "dec": dec})]
    k = 0
    for off in p["control_offsets_arcsec"]:
        for ang in np.arange(0.0, 360.0, 45.0):
            dra = off * np.cos(np.radians(ang)) / 3600.0 / np.cos(np.radians(dec))
            dde = off * np.sin(np.radians(ang)) / 3600.0
            rows.append(pd.DataFrame({"source_id": psr["jname"].astype(str) + f"|c{k}",
                                      "ra": ra + dra, "dec": dec + dde}))
            k += 1
    return pd.concat(rows, ignore_index=True)


def fetch_pulsar_counterparts(psr: pd.DataFrame, cfg: dict, *, xmatch_fn=None
                              ) -> tuple[dict, dict]:
    """AllWISE and CatWISE2020 matches for every pulsar and its control positions."""
    from ..acquire.science import _xmatch

    xmatch_fn = xmatch_fn or _xmatch
    p, ep = cfg["pulsar"], cfg["epochs"]
    r = float(p["xmatch_radius_arcsec"])
    out, meta = {}, {}
    for key, table, epoch, ren in (("allwise", "vizier:II/328/allwise", float(ep["allwise"]),
                                    ALLWISE_XMATCH_RENAME),
                                   ("catwise", "vizier:II/365/catwise", float(ep["catwise"]),
                                    CATWISE_XMATCH_RENAME)):
        pos = pulsar_match_positions(psr, cfg, epoch)
        t0 = time.monotonic()
        try:
            raw = xmatch_fn(pos, table, r)
            df = raw.rename(columns={k: v for k, v in ren.items() if k in raw.columns}) \
                if raw is not None and len(raw) else pd.DataFrame()
            for col in ("pmra_wise", "pmdec_wise", "e_pmra_wise", "e_pmdec_wise",
                        "pmra_catwise", "pmdec_catwise", "e_pmra_catwise", "e_pmdec_catwise"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce") * 1000.0
            out[key] = df
            meta[key] = {"status": "OK" if len(df) else "ZERO_ROWS", "n_rows": int(len(df)),
                         "n_positions": int(len(pos)), "radius_arcsec": r,
                         "elapsed_s": round(time.monotonic() - t0, 1)}
        except Exception as exc:                        # noqa: BLE001
            out[key] = pd.DataFrame()
            meta[key] = {"status": "FAILED", "error": repr(exc)[:300],
                         "n_positions": int(len(pos)),
                         "elapsed_s": round(time.monotonic() - t0, 1)}
            print(f"[ring/psr] {key} X-Match failed: {exc!r}", flush=True)
    return out, meta


# --------------------------------------------------------------------------
# VizieR tables resolved by role (brown dwarfs, free-floating planets)
# --------------------------------------------------------------------------

BD_ROLES = {
    "name": [r"^name$", r"^object$", r"^id$", r"^desig", r"^cwise", r"^wise", r"^src$",
             r"^objname$", r"^simbadname$", r"^shortname$"],
    # CatWISE positions (epoch 2015.4) first: the NEOWISE cone propagates from
    # the Gaia epoch, and a VizieR ``_RA``/``_DE`` at J2000 is 16 years of
    # proper motion (tens of arcsec for a Y dwarf) away from it.
    "ra": [r"^racdeg$", r"^raj2000$", r"^_ra$", r"^ra_icrs$", r"^radeg$", r"^ra$",
           r"^_raj2000$"],
    "dec": [r"^decdeg$", r"^dej2000$", r"^_de$", r"^de_icrs$", r"^dedeg$", r"^dec$", r"^de$",
            r"^_dej2000$"],
    "pmra": [r"^pmra$", r"^pmra"],
    "pmdec": [r"^pmde$", r"^pmdec$", r"^pmde"],
    "plx": [r"^plx$", r"^parallax$", r"^pi$", r"^plxw$", r"^plxg$"],
    # The near-infrared type first: late-T and Y dwarfs essentially never have
    # an OPTICAL type, and run 35752692549 resolved "SpTO" on the 20 pc census,
    # found 0 of 682 objects at >= T6, and fell back to a table without proper
    # motions (28 of 232 targets then returned NEOWISE epochs).
    "spt": [r"^sptir$", r"^spt$", r"^sptype", r"^spectype", r"^sp$", r"^type$",
            r"^spt.*adopt", r"^spt"],
    "spt_opt": [r"^spto$", r"^optspt"],
    "w1": [r"^w1mag$", r"^w1$", r"^w1mpro", r"^w1mag"],
    "w2": [r"^w2mag$", r"^w2$", r"^w2mpro", r"^w2mag"],
    "epoch": [r"^epoch", r"^ep$", r"^mjd"],
}
FFP_ROLES = {
    # Faherty+2016 names every object by its 2MASS designation, in a column
    # called "2MASS": none of the old patterns matched it, so no table carried
    # the required (name, lbol) pair and the leg reached nothing (run 35752692549).
    "name": [r"^name$", r"^object$", r"^id$", r"^desig", r"^objname$", r"^source$",
             r"^2mass$"],
    "spt": [r"^spt$", r"^sptype", r"^spectype", r"^sp$", r"^optspt", r"^nirspt", r"^spt"],
    "group": [r"^group$", r"^grp$", r"^assoc", r"^ymg", r"^member", r"^mg$", r"^mm$",
              r"^gbii$", r"^gl$"],
    "age": [r"^age"],
    "lbol": [r"^logl", r"^lbol", r"^log\(?l", r"^loglbol"],
    "mass": [r"^mass$", r"^m$", r"^mass"],
    "teff": [r"^teff"],
    "ra": [r"^raj2000$", r"^_ra$", r"^ra_icrs$", r"^radeg$", r"^ra$"],
    "dec": [r"^dej2000$", r"^_de$", r"^de_icrs$", r"^dedeg$", r"^dec$", r"^de$"],
}


def resolve_roles(columns, patterns: dict) -> dict:
    """First column (case-insensitive regex, in pattern priority order) per role."""
    cols = [str(c) for c in columns]
    low = [c.lower() for c in cols]
    out = {}
    for role, pats in patterns.items():
        for pat in pats:
            hit = next((cols[i] for i, c in enumerate(low) if re.search(pat, c)), None)
            if hit is not None and hit not in out.values():
                out[role] = hit
                break
    return out


def discover_vizier_table(catalogue: str, patterns: dict, required, *, query_fn=None,
                          fetch_fn=None) -> dict:
    """List the tables under ``catalogue``; pick the one resolving the most roles."""
    from ..metronome.acquire import list_tables, table_columns, tap_query

    query_fn = query_fn or tap_query
    board = []
    try:
        tabs = list_tables(catalogue, query_fn=query_fn, fetch_fn=fetch_fn)
    except Exception as exc:                            # noqa: BLE001
        return {"catalogue": catalogue, "table": None, "status": "QUERY_FAILED",
                "error": repr(exc)[:300], "scoreboard": board}
    best = None
    for _, row in tabs.iterrows():
        t = str(row["table_name"])
        try:
            cols = table_columns(t, query_fn=query_fn, fetch_fn=fetch_fn)
        except Exception as exc:                        # noqa: BLE001
            board.append({"table": t, "error": repr(exc)[:200]})
            continue
        roles = resolve_roles(cols, patterns)
        n_req = sum(1 for r in required if r in roles)
        entry = {"table": t, "roles": roles, "n_required": n_req, "n_columns": len(cols),
                 "columns": [str(c) for c in cols][:60]}
        board.append(entry)
        if n_req == len(required) and (best is None or len(roles) > len(best["roles"])):
            best = entry
    if best is None:
        return {"catalogue": catalogue, "table": None, "status": "NO_TABLE_WITH_ROLES",
                "required": list(required), "scoreboard": board}
    return {"catalogue": catalogue, "table": best["table"], "roles": best["roles"],
            "columns": best["columns"], "status": "OK", "scoreboard": board}


def fetch_vizier_roles(disc: dict, *, query_fn=None, max_rows: int | None = None
                       ) -> pd.DataFrame:
    """Pull the resolved role columns of a discovered table, renamed to roles."""
    from ..metronome.acquire import tap_query

    query_fn = query_fn or tap_query
    if not disc.get("table") or not disc.get("roles"):
        return pd.DataFrame()
    roles = dict(disc["roles"])
    sel = ", ".join(f'"{c}"' for c in dict.fromkeys(roles.values()))
    top = f"TOP {int(max_rows)} " if max_rows else ""
    df = query_fn(f'SELECT {top}{sel} FROM "{disc["table"]}"')
    inv = {v: k for k, v in roles.items()}
    out = df.rename(columns={c: inv[str(c)] for c in df.columns if str(c) in inv})
    for r in ("ra", "dec", "pmra", "pmdec", "plx", "w1", "w2", "age", "lbol", "mass", "teff",
              "epoch"):
        if r in out.columns:
            out[r] = pd.to_numeric(out[r], errors="coerce")
    out.attrs["route"] = str(getattr(df, "attrs", {}).get("route", "vizier"))
    return out


_SPT_CLASS = {"M": 0.0, "L": 10.0, "T": 20.0, "Y": 30.0}


def spt_to_numeric(s) -> float:
    """``'T7.5'`` -> 27.5; ``'Y0'`` -> 30; ``'>=Y1'`` -> 31; NaN when unreadable."""
    if s is None:
        return np.nan
    m = re.search(r"([MLTY])\s*(\d+(\.\d+)?)", str(s).upper())
    if not m:
        return np.nan
    return _SPT_CLASS[m.group(1)] + float(m.group(2))


def fetch_bd_targets(cfg: dict, *, query_fn=None, fetch_fn=None) -> tuple[pd.DataFrame, dict]:
    """Y and late-T dwarfs with positions and proper motions, from VizieR."""
    b = cfg["bd"]
    meta: dict = {"catalogues": []}
    for cat in [b["vizier_catalogue"]] + list(b.get("vizier_fallbacks", [])):
        disc = discover_vizier_table(cat, BD_ROLES, ("name", "ra", "dec", "spt"),
                                     query_fn=query_fn, fetch_fn=fetch_fn)
        entry = {"catalogue": cat, "status": disc.get("status"), "table": disc.get("table"),
                 "roles": disc.get("roles", {})}
        meta["catalogues"].append(entry)
        if disc.get("status") != "OK":
            continue
        try:
            df = fetch_vizier_roles(disc, query_fn=query_fn)
        except Exception as exc:                        # noqa: BLE001
            entry["status"] = "QUERY_FAILED"
            entry["error"] = repr(exc)[:300]
            continue
        entry["n_rows"] = int(len(df))
        ir = [spt_to_numeric(x) for x in df.get("spt", pd.Series(np.nan, index=df.index))]
        opt = [spt_to_numeric(x) for x in df.get("spt_opt", pd.Series(np.nan, index=df.index))]
        df["spt_num"] = [a if np.isfinite(a) else b for a, b in zip(ir, opt, strict=False)]
        entry["n_with_pm"] = int(pd.to_numeric(
            df.get("pmra", pd.Series(np.nan, index=df.index)), errors="coerce").notna().sum())
        late = df[df["spt_num"] >= float(b["spt_min_numeric"])].copy()
        entry["n_late"] = int(len(late))
        if len(late):
            late["source_id"] = late["name"].astype(str).str.strip()
            late = late.drop_duplicates("source_id").head(int(b["max_targets"]))
            meta["route"] = cat
            return late.reset_index(drop=True), meta
    meta["route"] = "none"
    return pd.DataFrame(), meta


def fetch_ffp_targets(cfg: dict, *, query_fn=None, fetch_fn=None) -> tuple[pd.DataFrame, dict]:
    """Young ultracool dwarfs with bolometric luminosities and group membership."""
    f = cfg["ffp"]
    disc = discover_vizier_table(f["vizier_catalogue"], FFP_ROLES, ("name", "lbol"),
                                 query_fn=query_fn, fetch_fn=fetch_fn)
    meta = {"catalogue": f["vizier_catalogue"], "status": disc.get("status"),
            "table": disc.get("table"), "roles": disc.get("roles", {}),
            "scoreboard": disc.get("scoreboard", [])[:20]}
    if disc.get("status") != "OK":
        meta["route"] = "none"
        return pd.DataFrame(), meta
    try:
        df = fetch_vizier_roles(disc, query_fn=query_fn)
    except Exception as exc:                            # noqa: BLE001
        meta["status"] = "QUERY_FAILED"
        meta["error"] = repr(exc)[:300]
        meta["route"] = "none"
        return pd.DataFrame(), meta
    meta["n_rows"] = int(len(df))
    meta["route"] = "vizier"
    # The luminosity table carries no group (hence no age) in Faherty+2016:
    # membership lives in a sibling table keyed by the same designation.  Join
    # the first sibling that resolves (name, group); without it every object is
    # untestable, and the screen says so.
    if "group" not in df.columns and "name" in df.columns:
        sib = [e for e in disc.get("scoreboard", [])
               if e.get("table") and e["table"] != disc.get("table")
               and {"name", "group"} <= set(e.get("roles") or {})]
        meta["membership_table"] = None
        for e in sib:
            try:
                mem = fetch_vizier_roles({"table": e["table"],
                                          "roles": {"name": e["roles"]["name"],
                                                    "group": e["roles"]["group"]}},
                                         query_fn=query_fn)
            except Exception as exc:                    # noqa: BLE001
                meta.setdefault("membership_errors", []).append(
                    {"table": e["table"], "error": repr(exc)[:200]})
                continue
            if not len(mem):
                continue
            mem["name"] = mem["name"].astype(str).str.strip()
            mem = mem.drop_duplicates("name")
            df = df.assign(name=df["name"].astype(str).str.strip()).merge(
                mem[["name", "group"]], on="name", how="left")
            meta["membership_table"] = e["table"]
            meta["membership_column"] = e["roles"]["group"]
            meta["n_with_group"] = int(df["group"].notna().sum())
            break
    return df, meta


def fetch_bd_neowise(targets: pd.DataFrame, out_dir: Path, cfg: dict, *, shard: int = 0,
                     n_shards: int = 1, cone_fn=None, time_budget_s: float | None = None
                     ) -> dict:
    """NEOWISE per-epoch W1/W2 for the targets, through IGNITION's checkpointed store."""
    from ..ignition.acquire import EpochStore, acquire_stars

    b = cfg["bd"]["neowise"]
    stars = targets.copy()
    stars["source_id"] = stars["source_id"].astype(str)
    stars = stars.iloc[shard::max(1, n_shards)].reset_index(drop=True)
    for c in ("pmra", "pmdec"):
        if c not in stars.columns:
            stars[c] = 0.0
    store = EpochStore.open(Path(out_dir), tag=f"bd_shard{shard}")
    conf = {"cone_radius_arcsec": float(b["cone_radius_arcsec"]),
            "cone_workers": int(b.get("cone_workers", 1)),
            "epoch_gap_days": float(b["epoch_gap_days"]),
            "min_exp_per_epoch": int(b["min_exp_per_epoch"]),
            "checkpoint_every": int(b["checkpoint_every"]),
            "time_budget_s": float(b["time_budget_s"])}
    roll = acquire_stars(stars, store, conf, route=str(b.get("route", "cone")),
                         cone_fn=cone_fn, time_budget_s=time_budget_s)
    roll["shard"] = shard
    roll["n_shards"] = n_shards
    roll["epochs_path"] = str(store.epochs_path)
    return roll


def write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


__all__ = ["WD_WANT", "probe_gaia_wd_table", "build_wd_query", "fetch_wd_gaia",
           "fetch_wd_vizier_parent", "gaia_upload_join_allwise", "xmatch_at_epoch",
           "normalise_xmatch", "fetch_wd_leg", "harmonise_wd", "parse_psrcat_db",
           "normalise_pulsars", "fetch_pulsars_psrqpy", "fetch_pulsars_tarball",
           "fetch_pulsars_vizier", "fetch_pulsars", "pulsar_match_positions",
           "fetch_pulsar_counterparts", "resolve_roles", "discover_vizier_table",
           "fetch_vizier_roles", "spt_to_numeric", "fetch_bd_targets", "fetch_ffp_targets",
           "fetch_bd_neowise", "write_json", "ALLWISE_XMATCH_RENAME", "CATWISE_XMATCH_RENAME"]
