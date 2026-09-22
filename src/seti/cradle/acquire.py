"""Runner-only archive access for CRADLE.  Every call is injectable for the tests.

Routes (all measured, none assumed):

* **ESA Gaia TAP** --- the parent sample, one HEALPix ``source_id`` range per
  query, over IGNITION's transport ladder (:func:`seti.ignition.sample.run_gaia_query`:
  astroquery/pyvo, async first, time-boxed, ``QUERY_FAILED`` never confused
  with ``QUERY_RETURNED_ZERO_ROWS``).  A unit that times out is split into its
  four NESTED children and each retried, down to ``healpix_split_max_level``.
  The controls are three small cones on the same join, with the W3/W4 cut off.
* **ESA upload join** --- Gaia neighbours of the shortlist in one query per
  chunk (``TAP_UPLOAD``), per-star cones as the fallback.
* **IRSA TAP** --- the AllWISE profile-fit columns (``w3rchi2``, ``w4rchi2``,
  ``nb``, ``na``) the ESA mirror lacks, by designation; NEOWISE single
  exposures per star (IGNITION's PM-propagated cone); the ``irs_enhv211``
  Spitzer/IRS catalogue (whole table, matched locally); the AllWISE box of a
  unit as the fallback parent route.
* **IRSA product store** --- the IRS Enhanced spectra, through a URL ladder
  whose every rung is recorded (no scheme is asserted; the first that answers
  with an IPAC table wins).
* **VizieR ASU** --- known-excess catalogues (Cotten & Song 2016, Kennedy &
  Wyatt 2013, McDonald+2012) and rotation-period catalogues (McQuillan+2014,
  Santos+2021) as per-star cones with percent-encoded, signed positions
  (the IGNITION lesson).
* **SIMBAD** (astroquery) for object types; **IRSA DUST** for SFD E(B-V).
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from ..ignition.acquire import fetch_neowise_cone, reduce_star
from ..ignition.irsa_route import IRSA_TAP, IRSA_TRANSPORTS
from ..ignition.sample import (
    QUERY_FAILED,
    QUERY_TIMED_OUT,
    QUERY_ZERO,
    GaiaQueryFailed,
    run_gaia_query,
    unwrap_result,
)
from .sample import (
    CONTROLS,
    DEFAULT_SAMPLE,
    GAIA_TAP,
    JOINED_SHAPES,
    SHAPE_GAIA_ONLY,
    build_query,
    unit_children,
    unit_label,
)

STATUS_OK = "OK"
STATUS_ZERO = QUERY_ZERO
STATUS_FAILED = QUERY_FAILED
STATUS_TIMED_OUT = QUERY_TIMED_OUT

IRSA_ALLWISE_TABLE = "allwise_p3as_psd"
IRSA_ALLWISE_EXTRA = ("designation", "w1rchi2", "w2rchi2", "w3rchi2", "w4rchi2", "nb", "na",
                      "w3flg", "w4flg", "w3nm", "w3m", "w4nm", "w4m", "w3snr", "w4snr",
                      "ext_flg", "cc_flags", "ph_qual", "tmass_key", "n_2mass")
IRS_TABLE = "irs_enhv211"
IRS_PRODUCT_BASE = "https://irsa.ipac.caltech.edu/data/SPITZER/Enhanced/IRS/"

#: VizieR tables (ASSERTED, verified by the probe with asu_table_exists).
DEFAULT_VIZIER_TABLES: dict = {
    "known_disks": {"cotten_song_2016": "J/ApJS/225/15/table3",
                    "kennedy_wyatt_2013": "J/MNRAS/433/2334/table1",
                    "mcdonald_2012": "J/MNRAS/427/343/table3"},
    "rotation": {"mcquillan_2014": "J/ApJS/211/24/table1",
                 "santos_2021": "J/ApJS/255/17/table1"},
    "match_radius_arcsec": 5.0,
}


@dataclass
class Backends:
    """Every network-touching callable, injectable.  ``None`` means the real one."""

    gaia: object = None          # adql -> (df, rec)   (ESA)
    gaia_upload: object = None   # (adql, table) -> df (ESA TAP_UPLOAD)
    irsa: object = None          # adql -> (df, rec)   (IRSA)
    asu: object = None           # url -> text         (VizieR ASU)
    simbad: object = None        # (ra, dec) -> otype  (SIMBAD)
    simbad_resolve: object = None  # name -> (ra, dec) or None
    ebv: object = None           # DataFrame(source_id, ra, dec) -> DataFrame(source_id, ebv_sfd)
    neowise: object = None       # (ra, dec, pmra, pmdec, radius) -> QueryResult
    http: object = None          # url -> (status, text)
    ledger: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------
def esa_query_fn(conf: dict | None = None, *, deadline: float | None = None,
                 allow_sync: bool = False, label: str = "cradle"):
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    timeout = float(c.get("query_timeout_s") or 1200.0)

    def _fn(adql: str):
        df, rec = run_gaia_query(adql, label=label, timeout_s=timeout, allow_sync=allow_sync,
                                 deadline=deadline, tag="cradle")
        if rec["status"] in (QUERY_FAILED, QUERY_TIMED_OUT):
            raise GaiaQueryFailed(rec)
        return df, rec
    return _fn


def irsa_query_fn(*, timeout_s: float = 900.0, deadline: float | None = None,
                  label: str = "cradle_irsa"):
    def _fn(adql: str):
        df, rec = run_gaia_query(adql, label=label, timeout_s=timeout_s, deadline=deadline,
                                 allow_sync=True, transports=IRSA_TRANSPORTS, tag="cradle")
        if rec["status"] in (QUERY_FAILED, QUERY_TIMED_OUT):
            raise GaiaQueryFailed(rec)
        return df, rec
    return _fn


def ask(fn, adql: str, *, label: str, service: str) -> tuple[pd.DataFrame, dict]:
    """One call, never raises; ``status`` separates failed from empty."""
    rec: dict = {"label": label, "service": service, "query": adql.strip()[:3000]}
    t0 = _time.monotonic()
    try:
        df, qrec = unwrap_result(fn(adql))
    except GaiaQueryFailed as exc:
        rec.update(status=str(exc.record.get("status") or STATUS_FAILED), n_rows=0,
                   error=str(exc.record.get("error") or exc), seconds=round(_time.monotonic() - t0, 1))
        return pd.DataFrame(), rec
    except Exception as exc:                               # noqa: BLE001
        rec.update(status=STATUS_FAILED, n_rows=0, error=repr(exc),
                   seconds=round(_time.monotonic() - t0, 1))
        return pd.DataFrame(), rec
    df = df if df is not None else pd.DataFrame()
    df = df.rename(columns={x: str(x).lower() for x in df.columns})
    for k in ("transport", "queue"):
        if qrec.get(k) is not None:
            rec[k] = qrec[k]
    rec.update(status=STATUS_OK if len(df) else STATUS_ZERO, n_rows=int(len(df)),
               seconds=round(_time.monotonic() - t0, 1))
    return df, rec


def _real_http(url: str, timeout: float = 120.0) -> tuple[int, str]:
    import requests  # noqa: PLC0415  runner-only

    r = requests.get(url, timeout=timeout,
                     headers={"User-Agent": "seti-cradle/1.0 (+github actions; astronomy)"})
    return int(r.status_code), r.text


def _real_simbad_resolve(name: str):
    from astroquery.simbad import Simbad  # noqa: PLC0415

    t = Simbad.query_object(name)
    if t is None or not len(t):
        return None
    for ra_k, dec_k in (("ra", "dec"), ("RA", "DEC"), ("RA_d", "DEC_d")):
        if ra_k in t.colnames and dec_k in t.colnames:
            ra, dec = t[ra_k][0], t[dec_k][0]
            try:
                return float(ra), float(dec)
            except (TypeError, ValueError):
                from astropy.coordinates import SkyCoord  # noqa: PLC0415
                sc = SkyCoord(str(ra), str(dec), unit=("hourangle", "deg"))
                return float(sc.ra.deg), float(sc.dec.deg)
    return None


def _real_simbad_type(ra: float, dec: float) -> str:
    from ..vigil.acquire import fetch_simbad_type  # noqa: PLC0415

    return fetch_simbad_type(ra, dec, radius_arcsec=5.0)


def _real_ebv(pos: pd.DataFrame) -> pd.DataFrame:
    from ..ossuary.acquire import fetch_ebv  # noqa: PLC0415

    return fetch_ebv(pos, tag="cradle")


def _real_gaia_upload(adql: str, table: pd.DataFrame) -> pd.DataFrame:
    from astropy.table import Table  # noqa: PLC0415
    from astroquery.gaia import Gaia  # noqa: PLC0415

    job = Gaia.launch_job_async(adql, upload_resource=Table.from_pandas(table),
                                upload_table_name="t")
    df = job.get_results().to_pandas()
    return df.rename(columns={c: str(c).lower() for c in df.columns})


def _real_asu(url: str) -> str:
    from ..metronome.acquire import _asu_http_text  # noqa: PLC0415

    return _asu_http_text(url, timeout=180.0)


# ---------------------------------------------------------------------------
# Column probes
# ---------------------------------------------------------------------------
def probe_columns(fn, table: str, *, label: str = "columns") -> tuple[list[str], dict]:
    df, rec = ask(fn, f"SELECT TOP 1 * FROM {table}", label=f"{label}:{table}", service="tap")
    return [str(c) for c in df.columns], rec


def resolve_names(columns: list[str], want: dict[str, list[str]]) -> tuple[dict, list[str]]:
    have = {c.lower(): c for c in columns}
    got, missing = {}, []
    for logical, cands in want.items():
        for cnd in cands:
            if cnd.lower() in have:
                got[logical] = have[cnd.lower()]
                break
        else:
            missing.append(logical)
    return got, missing


# ---------------------------------------------------------------------------
# The parent, one unit at a time, with adaptive splitting
# ---------------------------------------------------------------------------
def fetch_unit(conf: dict, unit: dict, fn, *, shapes: tuple[str, ...] | list[str] = JOINED_SHAPES,
               count: bool = True, max_level: int | None = None,
               ledger: list | None = None,
               deadline: float | None = None) -> tuple[pd.DataFrame, dict]:
    """One HEALPix unit over the shape ladder; timed-out units split recursively.

    Returns ``(rows, record)``.  ``record["status"]`` is ``OK`` /
    ``QUERY_RETURNED_ZERO_ROWS`` / ``QUERY_FAILED`` / ``PARTIAL`` (some children
    failed); ``record["n_parent"]`` is the Gaia-only ``COUNT(*)`` when measured.

    ``deadline`` is a :func:`time.monotonic` instant that bounds the whole unit,
    recursion included.  Without it one pathological level-3 pixel can cost
    ``1 + 4 + 16 + 64`` queries at ``query_timeout_s`` each --- 28 h at the
    configured 1200 s --- which is longer than the job it is running in, so the
    shard would lose every unit it had not yet reached.  Past the deadline the
    split is abandoned and the record carries ``deadline_exceeded``.
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    max_level = int(c.get("healpix_split_max_level", 6) if max_level is None else max_level)
    label = unit_label(unit)
    rec: dict = {"unit": dict(unit), "label": label, "attempts": [], "children": []}
    if deadline is not None and _time.monotonic() > deadline:
        rec.update(status=STATUS_FAILED, n_rows=0, deadline_exceeded=True)
        if ledger is not None:
            ledger.append({k: v for k, v in rec.items() if k != "children"})
        return pd.DataFrame(), rec
    if count:
        cdf, crec = ask(fn, build_query(c, unit=unit, shape=SHAPE_GAIA_ONLY, count_only=True),
                        label=f"count_{label}", service=GAIA_TAP)
        rec["count"] = {k: crec.get(k) for k in ("status", "seconds", "error")}
        rec["n_parent"] = int(pd.to_numeric(cdf.iloc[0, 0])) if len(cdf) else None
    else:
        rec["n_parent"] = None
    last_status = STATUS_FAILED
    for sh in shapes:
        if deadline is not None and _time.monotonic() > deadline:
            rec["deadline_exceeded"] = True
            break
        df, qrec = ask(fn, build_query(c, unit=unit, shape=sh), label=f"{label}:{sh}",
                       service=GAIA_TAP)
        rec["attempts"].append({"shape": sh, **{k: qrec.get(k) for k in
                                                 ("status", "n_rows", "seconds", "error",
                                                  "transport", "queue")}})
        last_status = qrec["status"]
        if qrec["status"] in (STATUS_OK, STATUS_ZERO):
            rec.update(status=qrec["status"], shape=sh, n_rows=int(len(df)),
                       seconds=qrec.get("seconds"))
            if len(df):
                df = df.copy()
                df["unit"] = label
                df["query_shape"] = sh
            if ledger is not None:
                ledger.append({k: v for k, v in rec.items() if k != "children"})
            return df, rec
        if qrec["status"] == STATUS_TIMED_OUT:
            break                      # a slower shape will not help; split instead
    if last_status == STATUS_TIMED_OUT and int(unit["level"]) < max_level:
        frames, kids = [], []
        n_fail = 0
        for child in unit_children(unit):
            cdf, crec = fetch_unit(c, child, fn, shapes=shapes, count=False,
                                   max_level=max_level, ledger=ledger, deadline=deadline)
            if crec.get("deadline_exceeded"):
                rec["deadline_exceeded"] = True
            kids.append({k: crec.get(k) for k in ("label", "status", "n_rows", "shape")})
            if crec.get("status") in (STATUS_OK, STATUS_ZERO):
                if len(cdf):
                    frames.append(cdf)
            else:
                n_fail += 1
        rec["children"] = kids
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        st = STATUS_FAILED if n_fail == 4 else ("PARTIAL" if n_fail else
                                                 (STATUS_OK if len(df) else STATUS_ZERO))
        rec.update(status=st, n_rows=int(len(df)), split=True, n_children_failed=n_fail)
        if ledger is not None:
            ledger.append({k: v for k, v in rec.items() if k != "children"})
        return df, rec
    rec.update(status=last_status, n_rows=0)
    if ledger is not None:
        ledger.append({k: v for k, v in rec.items() if k != "children"})
    return pd.DataFrame(), rec


def resolve_control(ctrl: dict, *, resolve_fn=None) -> dict:
    """Position of a control: SIMBAD first, the asserted coordinates second."""
    out = {"name": ctrl["name"], "ra": None, "dec": None, "method": None, "error": None}
    fn = resolve_fn or _real_simbad_resolve
    for alias in ctrl.get("aliases") or [ctrl["name"]]:
        try:
            got = fn(alias)
        except Exception as exc:                           # noqa: BLE001
            out["error"] = repr(exc)
            got = None
        if got:
            out.update(ra=float(got[0]), dec=float(got[1]), method=f"simbad:{alias}")
            return out
    if ctrl.get("ra") is not None:
        out.update(ra=float(ctrl["ra"]), dec=float(ctrl["dec"]), method="asserted")
    return out


def fetch_controls(conf: dict, fn, *, resolve_fn=None, radius_arcsec: float = 5.0,
                   shapes=JOINED_SHAPES) -> tuple[pd.DataFrame, list[dict]]:
    """The three known mature EDDs through the same join, W3/W4 cut off."""
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    frames, recs = [], []
    for ctrl in CONTROLS:
        pos = resolve_control(ctrl, resolve_fn=resolve_fn)
        rec = {**pos, "note": ctrl.get("note"), "attempts": []}
        if pos["ra"] is None:
            rec["status"] = "UNRESOLVED"
            recs.append(rec)
            continue
        cone = {"ra": pos["ra"], "dec": pos["dec"], "radius_deg": radius_arcsec / 3600.0}
        for sh in shapes:
            df, qrec = ask(fn, build_query(c, cone=cone, shape=sh, controls=True),
                           label=f"control:{ctrl['name']}:{sh}", service=GAIA_TAP)
            rec["attempts"].append({"shape": sh, "status": qrec.get("status"),
                                    "n_rows": qrec.get("n_rows"), "error": qrec.get("error")})
            if qrec["status"] in (STATUS_OK, STATUS_ZERO):
                rec["status"] = qrec["status"]
                rec["shape"] = sh
                if len(df):
                    df = df.copy()
                    df["is_control"] = True
                    df["control_name"] = ctrl["name"]
                    df["unit"] = "control"
                    df["query_shape"] = sh
                    frames.append(df)
                break
        else:
            rec["status"] = STATUS_FAILED
        recs.append(rec)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), recs


# ---------------------------------------------------------------------------
# Fallback parent route: gaia_only at ESA x AllWISE box at IRSA
# ---------------------------------------------------------------------------
def fetch_unit_irsa(conf: dict, unit: dict, gaia_fn, irsa_fn, *, columns: dict | None = None
                    ) -> tuple[pd.DataFrame, dict]:
    """The Gaia half by ``source_id`` range, the AllWISE half as an IRSA box.

    The box is the extent of the Gaia rows actually returned (plus a margin),
    so no HEALPix geometry is asserted.  W3/W4 SNR cuts go to IRSA (they are
    the science selection, not a match-affecting colour cut); the match is
    PM-propagated 2016.0 -> 2010.5 nearest neighbour within 3 arcsec.
    """
    from ..ignition.vizier_route import match_allwise  # noqa: PLC0415

    c = {**DEFAULT_SAMPLE, **(conf or {})}
    label = unit_label(unit)
    rec: dict = {"unit": dict(unit), "label": label, "route": "irsa_tap"}
    gdf, grec = ask(gaia_fn, build_query(c, unit=unit, shape=SHAPE_GAIA_ONLY),
                    label=f"{label}:gaia_only", service=GAIA_TAP)
    rec["gaia"] = {k: grec.get(k) for k in ("status", "n_rows", "seconds", "error")}
    if grec["status"] != STATUS_OK:
        rec.update(status=grec["status"], n_rows=0)
        return pd.DataFrame(), rec
    ra = pd.to_numeric(gdf["ra"], errors="coerce")
    dec = pd.to_numeric(gdf["dec"], errors="coerce")
    pad = 0.02
    e3 = 2.5 / np.log(10.0) / float(c["w3_snr_min"])
    e4 = 2.5 / np.log(10.0) / float(c["w4_snr_min"])
    ra_lo, ra_hi = float(ra.min()) - pad, float(ra.max()) + pad
    dec_lo, dec_hi = float(dec.min()) - pad, float(dec.max()) + pad
    wrap = (ra_hi - ra_lo) > 180.0
    ra_pred = (f"(ra >= {ra_lo:.5f} AND ra <= {ra_hi:.5f})" if not wrap
               else f"(ra >= {float(ra[ra > 180].min()) - pad:.5f} OR ra <= {float(ra[ra <= 180].max()) + pad:.5f})")
    cols = ("designation, ra, dec, w1mpro, w1sigmpro, w2mpro, w2sigmpro, w3mpro, w3sigmpro, "
            "w4mpro, w4sigmpro, cc_flags, ext_flg, var_flg, ph_qual, w3rchi2, w4rchi2, nb, na")
    adql = (f"SELECT {cols} FROM {IRSA_ALLWISE_TABLE} WHERE {ra_pred} "
            f"AND dec >= {dec_lo:.5f} AND dec <= {dec_hi:.5f} "
            f"AND w3sigmpro < {e3:.4f} AND w4sigmpro < {e4:.4f}")
    wdf, wrec = ask(irsa_fn, adql, label=f"{label}:irsa_allwise_box", service=IRSA_TAP)
    rec["allwise"] = {k: wrec.get(k) for k in ("status", "n_rows", "seconds", "error")}
    if wrec["status"] == STATUS_FAILED:
        rec.update(status=STATUS_FAILED, n_rows=0)
        return pd.DataFrame(), rec
    if not len(wdf):
        rec.update(status=STATUS_ZERO, n_rows=0)
        return pd.DataFrame(), rec
    wdf = wdf.rename(columns={"w1sigmpro": "w1mpro_error", "w2sigmpro": "w2mpro_error",
                              "w3sigmpro": "w3mpro_error", "w4sigmpro": "w4mpro_error",
                              "ext_flg": "ext_flag", "var_flg": "var_flag",
                              "designation": "allwise_designation"})
    joined, mrec = match_allwise(gdf, wdf, {"vizier": {"match_radius_arcsec": 3.0}})
    rec["match"] = mrec
    if len(joined):
        joined = joined.copy()
        joined["unit"] = label
        joined["query_shape"] = "irsa_route"
        joined["allwise_n_mates"] = np.nan
    rec.update(status=STATUS_OK if len(joined) else STATUS_ZERO, n_rows=int(len(joined)))
    return joined, rec


# ---------------------------------------------------------------------------
# Enrichment: Gaia neighbours (upload join, per-star cone fallback)
# ---------------------------------------------------------------------------
NEIGHBOUR_COLS = ("source_id", "ra", "dec", "parallax", "parallax_error", "pmra", "pmdec",
                  "phot_g_mean_mag", "bp_rp", "ruwe")


def neighbour_upload_adql() -> str:
    cols = ", ".join(f"g.{c} AS nb_{c}" for c in NEIGHBOUR_COLS)
    return (f"SELECT t.sid AS sid, {cols}, "
            "DISTANCE(POINT('ICRS', t.ra, t.dec), POINT('ICRS', g.ra, g.dec)) * 3600.0 AS sep_arcsec "
            "FROM TAP_UPLOAD.t AS t JOIN gaiadr3.gaia_source AS g "
            "ON 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', t.ra, t.dec, t.rad_deg))")


def neighbour_cone_adql(ra: float, dec: float, rad_deg: float) -> str:
    cols = ", ".join(f"g.{c} AS nb_{c}" for c in NEIGHBOUR_COLS)
    return (f"SELECT {cols}, DISTANCE(POINT('ICRS', {ra:.7f}, {dec:.7f}), "
            f"POINT('ICRS', g.ra, g.dec)) * 3600.0 AS sep_arcsec FROM gaiadr3.gaia_source AS g "
            f"WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, "
            f"{rad_deg:.7f}))")


def fetch_gaia_neighbours(stars: pd.DataFrame, radius_arcsec: np.ndarray, *, gaia_fn=None,
                          upload_fn=None, chunk: int = 200, ledger: list | None = None
                          ) -> pd.DataFrame:
    """All Gaia sources within each star's own radius (the star itself included)."""
    frames = []
    sid = stars["source_id"].astype(str).to_numpy()
    ra = pd.to_numeric(stars["ra"], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(stars["dec"], errors="coerce").to_numpy(float)
    rad = np.asarray(radius_arcsec, float) / 3600.0
    upload_fn = upload_fn or _real_gaia_upload
    done = np.zeros(len(stars), bool)
    for i in range(0, len(stars), chunk):
        sl = slice(i, min(i + chunk, len(stars)))
        tbl = pd.DataFrame({"sid": sid[sl], "ra": ra[sl], "dec": dec[sl], "rad_deg": rad[sl]})
        t0 = _time.monotonic()
        try:
            df = upload_fn(neighbour_upload_adql(), tbl)
            df = df.rename(columns={c: str(c).lower() for c in df.columns})
            df["sid"] = df["sid"].astype(str)
            frames.append(df)
            done[sl] = True
            if ledger is not None:
                ledger.append({"label": f"gaia_neighbours_upload[{i}:{sl.stop}]", "status": STATUS_OK,
                               "n_rows": int(len(df)), "seconds": round(_time.monotonic() - t0, 1)})
        except Exception as exc:                           # noqa: BLE001
            if ledger is not None:
                ledger.append({"label": f"gaia_neighbours_upload[{i}:{sl.stop}]",
                               "status": STATUS_FAILED, "error": repr(exc)[:300]})
    if not done.all() and gaia_fn is not None:
        for j in np.nonzero(~done)[0]:
            df, rec = ask(gaia_fn, neighbour_cone_adql(ra[j], dec[j], rad[j]),
                          label=f"gaia_neighbours_cone:{sid[j]}", service=GAIA_TAP)
            if ledger is not None:
                ledger.append({k: rec.get(k) for k in ("label", "status", "n_rows", "seconds", "error")})
            if rec["status"] in (STATUS_OK, STATUS_ZERO):
                df = df.copy()
                df["sid"] = sid[j]
                frames.append(df)
                done[j] = True
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["sid"])
    out.attrs["n_done"] = int(done.sum())
    out.attrs["n_stars"] = int(len(stars))
    return out


def summarise_neighbours(stars: pd.DataFrame, nb: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-star beam-neighbour counts and wide-companion counts from a neighbour table."""
    from .vet import DEFAULT_VET  # noqa: PLC0415

    c = {**DEFAULT_VET, **(cfg or {})}
    out = stars.copy()
    n = len(out)
    n6 = np.full(n, np.nan)
    n12 = np.full(n, np.nan)
    nwide = np.full(n, np.nan)
    wide_sep = np.full(n, np.nan)
    if not len(nb) or "sid" not in nb:
        for k, v in (("n_gaia_beam_neighbours", n6), ("n_gaia_w4beam_neighbours", n12),
                     ("n_wide_companions", nwide), ("wide_companion_sep_arcsec", wide_sep)):
            out[k] = v
        return out
    g = nb.groupby("sid")
    sids = out["source_id"].astype(str).to_numpy()
    plx = pd.to_numeric(out["parallax"], errors="coerce").to_numpy(float)
    pmra = pd.to_numeric(out["pmra"], errors="coerce").to_numpy(float)
    pmdec = pd.to_numeric(out["pmdec"], errors="coerce").to_numpy(float)
    gmag = pd.to_numeric(out["phot_g_mean_mag"], errors="coerce").to_numpy(float)
    done = set(str(s) for s in nb["sid"].unique())
    for i, s in enumerate(sids):
        if s not in done:
            continue
        rows = g.get_group(s)
        others = rows[rows["nb_source_id"].astype(str) != s]
        sep = pd.to_numeric(others["sep_arcsec"], errors="coerce").to_numpy(float)
        dg = pd.to_numeric(others["nb_phot_g_mean_mag"], errors="coerce").to_numpy(float) - gmag[i]
        n6[i] = int(((sep <= float(c["beam_neighbour_radius_arcsec"]))
                     & (dg <= float(c["beam_neighbour_dg_max"]))).sum())
        n12[i] = int(((sep <= float(c["w4_beam_radius_arcsec"]))
                      & (dg <= float(c["w4_beam_dg_max"]))).sum())
        rad_wide = float(c["wide_companion_au"]) / (1000.0 / plx[i]) if plx[i] > 0 else np.nan
        nplx = pd.to_numeric(others["nb_parallax"], errors="coerce").to_numpy(float)
        npmra = pd.to_numeric(others["nb_pmra"], errors="coerce").to_numpy(float)
        npmdec = pd.to_numeric(others["nb_pmdec"], errors="coerce").to_numpy(float)
        comp = ((sep <= rad_wide) & (np.abs(nplx - plx[i]) <= float(c["wide_companion_plx_frac"]) * plx[i])
                & (np.hypot(npmra - pmra[i], npmdec - pmdec[i]) <= float(c["wide_companion_pm_tol_mas_yr"])))
        nwide[i] = int(np.nansum(comp))
        if nwide[i] > 0:
            wide_sep[i] = float(np.nanmin(sep[comp]))
    out["n_gaia_beam_neighbours"] = n6
    out["n_gaia_w4beam_neighbours"] = n12
    out["n_wide_companions"] = nwide
    out["wide_companion_sep_arcsec"] = wide_sep
    return out


# ---------------------------------------------------------------------------
# Enrichment: IRSA AllWISE profile-fit columns by designation
# ---------------------------------------------------------------------------
def fetch_irsa_allwise_rows(designations, irsa_fn, *, chunk: int = 100,
                            ledger: list | None = None) -> pd.DataFrame:
    frames = []
    des = [str(d).strip() for d in designations if str(d).strip() and str(d) != "nan"]
    cols = ", ".join(IRSA_ALLWISE_EXTRA)
    for i in range(0, len(des), chunk):
        sub = des[i:i + chunk]
        lst = ", ".join("'" + d.replace("'", "''") + "'" for d in sub)
        df, rec = ask(irsa_fn, f"SELECT {cols} FROM {IRSA_ALLWISE_TABLE} WHERE designation IN ({lst})",
                      label=f"irsa_allwise_rows[{i}:{i + len(sub)}]", service=IRSA_TAP)
        if ledger is not None:
            ledger.append({k: rec.get(k) for k in ("label", "status", "n_rows", "seconds", "error")})
        if rec["status"] == STATUS_OK:
            frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(IRSA_ALLWISE_EXTRA))
    return out


# ---------------------------------------------------------------------------
# Enrichment: NEOWISE variability descriptor
# ---------------------------------------------------------------------------
def neowise_descriptor(star: dict, *, cone_fn=None, conf: dict | None = None) -> dict:
    """Epoch-binned W1/W2 variability of one star (IGNITION's cleaning + binning)."""
    cone_fn = cone_fn or fetch_neowise_cone
    rec = {"neowise_measured": False, "neowise_status": None, "neowise_n_epochs_w1": 0,
           "neowise_n_epochs_w2": 0, "neowise_chi2red_w1": np.nan, "neowise_chi2red_w2": np.nan,
           "neowise_amp_w1": np.nan, "neowise_amp_w2": np.nan, "neowise_variable": False}
    try:
        r = cone_fn(float(star["ra"]), float(star["dec"]), float(star.get("pmra") or 0.0),
                    float(star.get("pmdec") or 0.0), radius_arcsec=2.5)
    except Exception as exc:                               # noqa: BLE001
        rec["neowise_status"] = f"QUERY_FAILED: {exc!r}"[:200]
        return rec
    rec["neowise_status"] = getattr(r, "status", None)
    data = getattr(r, "data", None)
    if data is None or not len(data):
        return rec
    ep, srec = reduce_star(str(star.get("source_id")), data, conf)
    rec["neowise_measured"] = True
    rec["neowise_n_exp_clean"] = int(srec.get("n_exp_clean", 0))
    for band in ("W1", "W2"):
        e = ep[ep["band"] == band]
        rec[f"neowise_n_epochs_{band.lower()}"] = int(len(e))
        if len(e) >= 3:
            m = e["mag"].to_numpy(float)
            s = e["err"].to_numpy(float)
            med = np.median(m)
            chi2 = float(np.sum(((m - med) / s) ** 2) / max(len(m) - 1, 1))
            rec[f"neowise_chi2red_{band.lower()}"] = chi2
            rec[f"neowise_amp_{band.lower()}"] = float(np.max(m) - np.min(m))
    c1, c2 = rec["neowise_chi2red_w1"], rec["neowise_chi2red_w2"]
    a1, a2 = rec["neowise_amp_w1"], rec["neowise_amp_w2"]
    rec["neowise_variable"] = bool(np.isfinite(c1) and np.isfinite(c2) and c1 > 3 and c2 > 3
                                   and np.isfinite(a1) and np.isfinite(a2) and min(a1, a2) > 0.05)
    return rec


# ---------------------------------------------------------------------------
# Enrichment: Spitzer/IRS Enhanced Products
# ---------------------------------------------------------------------------
def fetch_irs_catalog(irsa_fn, *, ledger: list | None = None, max_rows: int = 40000
                      ) -> tuple[pd.DataFrame, dict]:
    df, rec = ask(irsa_fn, f"SELECT TOP {int(max_rows)} * FROM {IRS_TABLE}", label="irs_catalog",
                  service=IRSA_TAP)
    if ledger is not None:
        ledger.append({k: rec.get(k) for k in ("label", "status", "n_rows", "seconds", "error")})
    rec["columns"] = [str(c) for c in df.columns]
    return df, rec


def match_irs(stars: pd.DataFrame, irs: pd.DataFrame, radius_arcsec: float = 5.0) -> pd.DataFrame:
    """Nearest IRS entry within ``radius_arcsec`` per star (positions as catalogued)."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    out = stars.copy()
    n = len(out)
    out["irs_reqkey"] = np.nan
    out["irs_object"] = ""
    out["irs_sep_arcsec"] = np.nan
    out["irs_n_matches"] = 0
    if not len(irs) or not n or "ra" not in irs or "dec" not in irs:
        return out

    def _uv(ra, dec):
        r = np.radians(np.asarray(ra, float))
        d = np.radians(np.asarray(dec, float))
        return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])

    ira = pd.to_numeric(irs["ra"], errors="coerce").to_numpy(float)
    idec = pd.to_numeric(irs["dec"], errors="coerce").to_numpy(float)
    ok = np.isfinite(ira) & np.isfinite(idec)
    tree = cKDTree(_uv(ira[ok], idec[ok]))
    pts = _uv(pd.to_numeric(out["ra"], errors="coerce"), pd.to_numeric(out["dec"], errors="coerce"))
    chord = 2.0 * np.sin(np.radians(radius_arcsec / 3600.0) / 2.0)
    dist, idx = tree.query(pts, k=1, distance_upper_bound=chord)
    hit = np.isfinite(dist) & (idx < ok.sum())
    irs_ok = irs[ok].reset_index(drop=True)
    for i in np.nonzero(hit)[0]:
        row = irs_ok.iloc[idx[i]]
        out.loc[out.index[i], "irs_reqkey"] = row.get("reqkey", np.nan)
        out.loc[out.index[i], "irs_object"] = str(row.get("object", ""))
        out.loc[out.index[i], "irs_sep_arcsec"] = float(np.degrees(2 * np.arcsin(dist[i] / 2)) * 3600)
        out.loc[out.index[i], "irs_n_matches"] = len(tree.query_ball_point(pts[i], chord))
        for col in ("irs16", "irs22", "mips24", "irac8", "iras12", "iras25"):
            if col in irs_ok.columns:
                out.loc[out.index[i], f"irs_{col}"] = row.get(col, np.nan)
    return out


def irs_spectrum_urls(row: dict, columns: list[str] | None = None) -> list[str]:
    """Candidate product URLs for one ``irs_enhv211`` row --- a ladder, not a claim.

    Any column whose name says file/url/path is used first (absolute, or joined
    to the Enhanced Products base); then the naming patterns seen in IRSA's
    Enhanced Products tree.  Every rung is tried and recorded by the caller.
    """
    urls: list[str] = []
    for k, v in (row or {}).items():
        kl = str(k).lower()
        if any(t in kl for t in ("url", "file", "fname", "path", "spec_")) and isinstance(v, str) \
                and v.strip() and v.strip().lower() not in ("nan", "null", "none"):
            s = v.strip()
            urls.append(s if s.startswith("http") else IRS_PRODUCT_BASE + s.lstrip("/"))
    try:
        rk = int(float(row.get("reqkey")))
        tn = int(float(row.get("tn", 1) or 1))
    except (TypeError, ValueError):
        return list(dict.fromkeys(urls))
    k8 = f"{rk:08d}"
    stems = [f"SPITZER_S5_{rk}_{tn:02d}_merge.tbl", f"SPITZER_S5_{rk}_{tn:d}_merge.tbl",
             f"SPITZER_S5_{k8}_{tn:02d}_merge.tbl", f"SPITZER_S5_{rk}_merge.tbl"]
    dirs = [f"spectra/{k8[0]}/{k8[:4]}/{k8}/", f"spectra/{k8[:4]}/{k8}/", f"spectra/{rk}/",
            f"spectra/{k8}/", f"spectra/{k8[0]}/{k8[:4]}/{rk}/"]
    for d in dirs:
        for s in stems:
            urls.append(IRS_PRODUCT_BASE + d + s)
    return list(dict.fromkeys(urls))


def fetch_irs_spectrum(row: dict, *, http_fn=None, max_tries: int = 12) -> tuple[pd.DataFrame, dict]:
    """Walk the URL ladder; the first IPAC table with a wavelength column wins."""
    from .mineralogy import parse_ipac_table, spectrum_from_table  # noqa: PLC0415

    http_fn = http_fn or _real_http
    rec: dict = {"tried": [], "url": None, "status": "SPECTRUM_NOT_RETRIEVED"}
    for url in irs_spectrum_urls(row)[:max_tries]:
        try:
            status, text = http_fn(url)
        except Exception as exc:                           # noqa: BLE001
            rec["tried"].append({"url": url, "error": repr(exc)[:160]})
            continue
        rec["tried"].append({"url": url, "http": int(status)})
        if int(status) != 200:
            continue
        df = parse_ipac_table(text)
        w, f, _e = spectrum_from_table(df)
        if len(w) >= 10:
            rec.update(url=url, status="OK", n_points=int(len(w)))
            return df, rec
        rec["tried"][-1]["note"] = f"200 but no spectrum columns ({list(df.columns)[:8]})"
    return pd.DataFrame(), rec


# ---------------------------------------------------------------------------
# Enrichment: VizieR cones (known excess catalogues, rotation periods)
# ---------------------------------------------------------------------------
def vizier_cone(table: str, ra: float, dec: float, radius_arcsec: float, *, asu_fn=None,
                max_rows: int = 20) -> tuple[pd.DataFrame, dict]:
    from ..metronome.acquire import (  # noqa: PLC0415
        VizierRouteError,
        asu_position_spellings,
        asu_rows,
    )

    spelling = dict(asu_position_spellings(ra, dec, radius_arcsec / 3600.0))["decimal_signed"]
    rec = {"table": table, "constraints": spelling}
    try:
        df, attempts = asu_rows(table, max_rows=max_rows, fetch_fn=asu_fn or _real_asu,
                                constraints=spelling)
        rec.update(status=STATUS_OK if len(df) else STATUS_ZERO, n_rows=int(len(df)),
                   endpoint=df.attrs.get("endpoint"))
        return df, rec
    except VizierRouteError as exc:
        rec.update(status=STATUS_FAILED, error=str(exc)[:300])
    except Exception as exc:                               # noqa: BLE001
        rec.update(status=STATUS_FAILED, error=repr(exc)[:300])
    return pd.DataFrame(), rec


def vizier_matches(stars: pd.DataFrame, tables: dict, radius_arcsec: float, *, asu_fn=None,
                   ledger: list | None = None, skip_tables: set | None = None) -> pd.DataFrame:
    """Per-star match counts against each catalogue; a rotation period where one has it."""
    out = stars.copy()
    n = len(out)
    n_known = np.zeros(n, int)
    known_names = [[] for _ in range(n)]
    prot = np.full(n, np.nan)
    prot_src = np.full(n, "", dtype=object)
    skip = set(skip_tables or [])
    for i, (_, s) in enumerate(out.iterrows()):
        ra, dec = float(s["ra"]), float(s["dec"])
        for name, table in (tables.get("known_disks") or {}).items():
            if table in skip:
                continue
            df, rec = vizier_cone(table, ra, dec, radius_arcsec, asu_fn=asu_fn)
            if ledger is not None:
                ledger.append({"label": f"vizier:{name}:{s['source_id']}", **{k: rec.get(k) for k in ("status", "n_rows", "error")}})
            if rec.get("status") == STATUS_OK:
                n_known[i] += int(len(df))
                known_names[i].append(name)
        for name, table in (tables.get("rotation") or {}).items():
            if table in skip or np.isfinite(prot[i]):
                continue
            df, rec = vizier_cone(table, ra, dec, radius_arcsec, asu_fn=asu_fn)
            if ledger is not None:
                ledger.append({"label": f"vizier:{name}:{s['source_id']}", **{k: rec.get(k) for k in ("status", "n_rows", "error")}})
            if rec.get("status") == STATUS_OK:
                col = next((c for c in df.columns if str(c).lower() in ("prot", "per", "period", "p_rot", "prot1")), None)
                if col is not None:
                    v = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(v):
                        prot[i] = float(v.iloc[0])
                        prot_src[i] = name
    out["n_known_disk_matches"] = n_known
    out["known_disk_catalogues"] = ["+".join(k) for k in known_names]
    out["prot_d"] = prot
    out["prot_source"] = prot_src
    return out


# ---------------------------------------------------------------------------
# Checkpointed unit store for the acquire stage
# ---------------------------------------------------------------------------
@dataclass
class UnitStore:
    """Append-only per-unit checkpoint: one CSV of rows, one JSON of progress.

    The CSV is appended to **without a header** after the first write, so every
    frame must be written in the SAME column order.  It is not automatic: a
    unit's rows come out of :func:`fetch_unit` as ``[gaia..., unit,
    query_shape]`` and then gain ``is_control``/``control_name``, while the
    control cones come out of :func:`fetch_controls` with those four columns in
    the other order.  Appending them raw silently shifts every field by two
    columns --- which reads back as *every star is a control*.  So the header
    is pinned on the first write, later frames are reindexed onto it, and a
    genuinely new column (a fallback route with a different product shape)
    widens the header by rewriting what is already on disk.
    """

    rows_path: Path
    progress_path: Path
    done: set = field(default_factory=set)
    records: list = field(default_factory=list)
    columns: list | None = None

    @classmethod
    def open(cls, out: Path, tag: str) -> UnitStore:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        st = cls(out / f"parent_{tag}.csv", out / f"acquire_{tag}.json")
        if st.progress_path.exists():
            try:
                p = json.loads(st.progress_path.read_text())
                st.done = set(str(x) for x in p.get("done", []))
                st.records = list(p.get("records", []))
                cols = p.get("columns")
                st.columns = [str(c) for c in cols] if cols else None
                print(f"[cradle] resuming {tag}: {len(st.done)} units already done")
            except Exception as exc:                       # noqa: BLE001
                print(f"[cradle] progress unreadable ({exc!r}); starting fresh")
        if st.columns is None and st.rows_path.exists():
            try:
                st.columns = [str(c) for c in pd.read_csv(st.rows_path, nrows=0).columns]
            except Exception:                              # noqa: BLE001
                pass
        return st

    def write(self, label: str, rows: pd.DataFrame, record: dict) -> None:
        if len(rows):
            if self.columns is None:
                self.columns = [str(c) for c in rows.columns]
            else:
                added = [str(c) for c in rows.columns if str(c) not in self.columns]
                if added:
                    self.columns = list(self.columns) + added
                    record["columns_added"] = added[:40]
                    if self.rows_path.exists():
                        old = pd.read_csv(self.rows_path, low_memory=False)
                        old.reindex(columns=self.columns).to_csv(self.rows_path, index=False)
            missing = [c for c in self.columns if c not in set(str(x) for x in rows.columns)]
            if missing:
                record["columns_missing"] = missing[:40]
            rows = rows.rename(columns={c: str(c) for c in rows.columns}) \
                       .reindex(columns=self.columns)
            rows.to_csv(self.rows_path, mode="a", index=False, header=not self.rows_path.exists())
        self.done.add(label)
        self.records.append(record)
        self.flush()

    def flush(self, extra: dict | None = None) -> None:
        rec = {"done": sorted(self.done), "n_done": len(self.done), "columns": self.columns,
               "records": self.records[-2000:]}
        if extra:
            rec.update(extra)
        self.progress_path.write_text(json.dumps(rec, indent=1, default=str))


def quote_url(s: str) -> str:
    return quote(s, safe="/:?=&+_.-")


__all__ = ["Backends", "DEFAULT_VIZIER_TABLES", "IRSA_ALLWISE_EXTRA", "IRSA_ALLWISE_TABLE",
           "IRSA_TAP", "IRS_PRODUCT_BASE", "IRS_TABLE", "NEIGHBOUR_COLS", "STATUS_FAILED",
           "STATUS_OK", "STATUS_TIMED_OUT", "STATUS_ZERO", "UnitStore", "ask", "esa_query_fn",
           "fetch_controls", "fetch_gaia_neighbours", "fetch_irs_catalog", "fetch_irs_spectrum",
           "fetch_irsa_allwise_rows", "fetch_unit", "fetch_unit_irsa", "irs_spectrum_urls",
           "irsa_query_fn", "match_irs", "neighbour_cone_adql", "neighbour_upload_adql",
           "neowise_descriptor", "probe_columns", "resolve_control", "resolve_names",
           "summarise_neighbours", "vizier_cone", "vizier_matches"]
