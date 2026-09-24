"""ANTIPHASE archive access (runner only; every fetcher is injectable).

* **ZTF** light curves, g and r, from IRSA's ZTF light-curve API
  (``nph_light_curves``, one PM-propagated cone per star, ``catflags == 0``,
  the dominant object id per filter).  Threads, a job-wide time budget, and a
  per-star status so a failed star is a counted failure, never an empty series.
* **NEOWISE** single exposures for the control objects: IGNITION's cone
  (``seti.ignition.acquire.fetch_neowise_cone``) and its frame cleaning and
  epoch binning (``reduce_star``), unchanged.  The survey stars' NEOWISE
  epochs are *not* re-fetched: they come from IGNITION's shard artifacts.
* **Gaia alerts** light curves (G band) for controls outside the ZTF footprint.
* **Sesame** (CDS) name resolution, to check every control's coordinates.
* **Gaia DR3 neighbours** from the VizieR mirror ``I/355/gaiadr3`` over the
  non-TAP ASU interface --- the route that answered while the ESA archive was
  dark (docs/ignition.md 7.6) --- for the blend model.
"""

from __future__ import annotations

import io
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

ZTF_API = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
ZTF_DEC_MIN = -31.0
ZTF_EPOCH = 2021.0          # mid-survey epoch the cone is propagated to
ZTF_SPAN = (2018.2, 2025.0)
GAIA_EPOCH = 2016.0
SESAME = ("https://cds.unistra.fr/cgi-bin/nph-sesame/-oI/SNV?",
          "https://vizier.cfa.harvard.edu/viz-bin/nph-sesame/-oI/SNV?")
VIZIER_ASU = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
              "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv")
GAIA_ALERTS = "http://gsaweb.ast.cam.ac.uk/alerts/alert/{name}/lightcurve.csv"


def _propagate(ra, dec, pmra, pmdec, from_ep, to_ep):
    dt = float(to_ep) - float(from_ep)
    pmra = float(pmra) if pmra is not None and np.isfinite(float(pmra)) else 0.0
    pmdec = float(pmdec) if pmdec is not None and np.isfinite(float(pmdec)) else 0.0
    dec2 = float(dec) + pmdec * dt / 3.6e6
    ra2 = float(ra) + pmra * dt / 3.6e6 / np.cos(np.radians(float(dec)))
    return ra2, dec2


def parse_ztf_csv(text: str, bright_limit: float = 12.5) -> dict:
    """IRSA ZTF CSV -> ``{"g": {...}, "r": {...}}``, one object id per filter.

    Keeps ``catflags == 0``; per filter the ``oid`` with the most good points.
    Each band carries ``mjd, mag, err`` arrays, ``oid``, ``n``, ``median_mag``
    and ``saturation_risk`` (median brighter than ``bright_limit``).
    """
    df = pd.read_csv(io.StringIO(text))
    out: dict = {}
    if not len(df) or "filtercode" not in df:
        return out
    if "catflags" in df:
        df = df[pd.to_numeric(df["catflags"], errors="coerce") == 0]
    for flt, g in df.groupby("filtercode"):
        band = {"zg": "g", "zr": "r", "zi": "i"}.get(str(flt), str(flt))
        if band not in ("g", "r"):
            continue
        if "oid" in g and len(g):
            g = g[g["oid"] == g["oid"].value_counts().idxmax()]
        mjd = pd.to_numeric(g["mjd"], errors="coerce").to_numpy(float)
        mag = pd.to_numeric(g["mag"], errors="coerce").to_numpy(float)
        err = pd.to_numeric(g["magerr"], errors="coerce").to_numpy(float)
        ok = np.isfinite(mjd) & np.isfinite(mag)
        if not ok.any():
            continue
        med = float(np.median(mag[ok]))
        out[band] = {"mjd": mjd[ok], "mag": mag[ok], "err": err[ok],
                     "oid": str(g["oid"].iloc[0]) if "oid" in g and len(g) else "",
                     "n": int(ok.sum()), "median_mag": med,
                     "saturation_risk": bool(med < bright_limit)}
    return out


def fetch_ztf(ra: float, dec: float, pmra: float = 0.0, pmdec: float = 0.0,
              radius_arcsec: float = 1.5, timeout_s: float = 90.0, retries: int = 2,
              session=None, bright_limit: float = 12.5) -> dict:
    """One star's ZTF g/r light curve from the IRSA API.  ``status`` is one of
    ``OK``, ``NO_ROWS``, ``NOT_IN_FOOTPRINT``, ``FAILED``."""
    import requests

    if float(dec) < ZTF_DEC_MIN:
        return {"status": "NOT_IN_FOOTPRINT"}
    ra2, dec2 = _propagate(ra, dec, pmra, pmdec, GAIA_EPOCH, ZTF_EPOCH)
    mu = float(np.hypot(pmra or 0.0, pmdec or 0.0)) if np.isfinite(pmra or 0.0) else 0.0
    rad = float(radius_arcsec) + 0.5 * mu * (ZTF_SPAN[1] - ZTF_SPAN[0]) / 1000.0
    params = {"POS": f"CIRCLE {ra2:.7f} {dec2:.7f} {rad / 3600.0:.7f}",
              "BANDNAME": "g,r", "BAD_CATFLAGS_MASK": "32768", "FORMAT": "csv"}
    s = session or requests
    last = None
    for k in range(int(retries) + 1):
        try:
            r = s.get(ZTF_API, params=params, timeout=timeout_s)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            txt = r.text
            if txt.lstrip().startswith("<") and "ERROR" in txt.upper():
                raise RuntimeError(txt[:200])
            bands = parse_ztf_csv(txt, bright_limit)
            if not bands:
                return {"status": "NO_ROWS", "radius_arcsec": rad}
            return {"status": "OK", "bands": bands, "radius_arcsec": rad}
        except Exception as exc:                        # noqa: BLE001
            last = repr(exc)[:300]
            _time.sleep(2.0 * (k + 1))
    return {"status": "FAILED", "error": last}


def fetch_ztf_many(stars: pd.DataFrame, *, workers: int = 4, budget_s: float = 3600.0,
                   fetch=None, on_result=None, **kw) -> dict:
    """ZTF for every row of ``stars`` (``source_id, ra, dec, pmra, pmdec``).

    Calls ``on_result(sid, rec)`` as each star finishes (the checkpoint hook).
    Stops *submitting* at ``budget_s``; returns status counts and the ids not
    attempted.
    """
    import requests

    fetch = fetch or fetch_ztf
    t0 = _time.monotonic()
    counts: dict = {}
    not_attempted: list[str] = []
    rows = stars.to_dict("records")
    sess = requests.Session() if fetch is fetch_ztf else None
    kw2 = dict(kw)
    if sess is not None:
        kw2["session"] = sess
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as ex:
        futs = {}
        it = iter(rows)
        exhausted = False

        def _submit_one():
            nonlocal exhausted
            try:
                r = next(it)
            except StopIteration:
                exhausted = True
                return
            f = ex.submit(fetch, float(r["ra"]), float(r["dec"]),
                          float(r.get("pmra") or 0.0), float(r.get("pmdec") or 0.0), **kw2)
            futs[f] = str(r["source_id"])

        for _ in range(max(int(workers), 1) * 2):
            _submit_one()
        while futs:
            done = next(as_completed(list(futs)))
            sid = futs.pop(done)
            try:
                rec = done.result()
            except Exception as exc:                    # noqa: BLE001
                rec = {"status": "FAILED", "error": repr(exc)[:300]}
            counts[rec.get("status", "FAILED")] = counts.get(rec.get("status", "FAILED"), 0) + 1
            if on_result is not None:
                on_result(sid, rec)
            if not exhausted and _time.monotonic() - t0 < budget_s:
                _submit_one()
        if not exhausted:
            not_attempted = [str(r["source_id"]) for r in it]
    return {"counts": counts, "n_not_attempted": len(not_attempted),
            "not_attempted": not_attempted[:50], "elapsed_s": round(_time.monotonic() - t0, 1)}


# ---------------------------------------------------------------------------
# The batched ZTF route (run 36006324981: the per-star POS query took ~52 s a
# star per worker -- the API runs a positional TAP query of its own each time --
# so 11k stars a shard would need ~40 h).  Object ids come from ONE TAP_UPLOAD
# positional join against IRSA's ZTF objects table per chunk (the transport
# IGNITION proved for NEOWISE), and light curves from the API's multi-ID form,
# which skips the positional step.
# ---------------------------------------------------------------------------
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP"


def find_ztf_objects_table(run_sync=None) -> dict:
    """The newest ZTF objects table IRSA's own TAP_SCHEMA lists (never assumed)."""
    import re

    q = ("SELECT table_name FROM TAP_SCHEMA.tables WHERE "
         "table_name LIKE '%ztf_objects%'")
    try:
        if run_sync is None:
            import pyvo

            df = pyvo.dal.TAPService(IRSA_TAP).run_sync(q).to_table().to_pandas()
        else:
            df = run_sync(q)
    except Exception as exc:                            # noqa: BLE001
        return {"status": "FAILED", "error": repr(exc)[:300]}
    names = [str(x.decode() if isinstance(x, bytes) else x) for x in df.iloc[:, 0]] if len(df) else []
    ranked = []
    for n in names:
        m = re.search(r"dr(\d+)", n)
        ranked.append((int(m.group(1)) if m else -1, n))
    if not ranked:
        return {"status": "CATALOGUE_NOT_FOUND", "names": names}
    ranked.sort()
    return {"status": "OK", "table": ranked[-1][1], "names": names}


def _ztf_upload_table(stars: pd.DataFrame, radius_arcsec: float):
    from astropy.table import Table

    ra2, de2, rad = [], [], []
    for r in stars.to_dict("records"):
        pmra = float(r.get("pmra") or 0.0) if np.isfinite(float(r.get("pmra") or 0.0)) else 0.0
        pmde = float(r.get("pmdec") or 0.0) if np.isfinite(float(r.get("pmdec") or 0.0)) else 0.0
        a, d = _propagate(r["ra"], r["dec"], pmra, pmde, GAIA_EPOCH, ZTF_EPOCH)
        ra2.append(a)
        de2.append(d)
        rad.append(float(radius_arcsec) + 0.5 * np.hypot(pmra, pmde)
                   * (ZTF_SPAN[1] - ZTF_SPAN[0]) / 1000.0)
    t = Table()
    t["sid"] = np.arange(len(stars), dtype="int64")
    t["ra"] = np.asarray(ra2, float)
    t["dec"] = np.asarray(de2, float)
    return t, np.asarray(rad, float)


def match_oids(rows: pd.DataFrame, stars: pd.DataFrame, radii) -> dict:
    """Per star and filter, the object id with the most good epochs within its radius (pure)."""
    out: dict = {}
    if rows is None or not len(rows):
        return out
    d = rows.copy()
    d.columns = [str(c).lower() for c in d.columns]
    for c in ("sid", "ngoodobsrel", "ra", "dec", "ra_p", "dec_p"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["sep"] = 3600.0 * np.hypot((d["ra"] - d["ra_p"]) * np.cos(np.radians(d["dec_p"])),
                                 d["dec"] - d["dec_p"])
    ids = stars["source_id"].astype(str).to_numpy()
    for (k, flt), g in d.groupby(["sid", "filtercode"]):
        k = int(k)
        if k < 0 or k >= len(ids):
            continue
        g = g[g["sep"] <= float(radii[k]) + 1e-6]
        if not len(g):
            continue
        band = {"zg": "g", "zr": "r", "zi": "i"}.get(str(flt).strip(), str(flt).strip())
        if band not in ("g", "r"):
            continue
        g = g.sort_values(["ngoodobsrel", "sep"], ascending=[False, True])
        out.setdefault(ids[k], {})[band] = str(int(g["oid"].iloc[0]))
    return out


def parse_ztf_multi(text: str) -> dict:
    """Multi-ID light-curve CSV -> ``{oid: {"mjd","mag","err","filtercode"}}`` (catflags == 0)."""
    df = pd.read_csv(io.StringIO(text))
    out: dict = {}
    if not len(df) or "oid" not in df:
        return out
    if "catflags" in df:
        df = df[pd.to_numeric(df["catflags"], errors="coerce") == 0]
    for oid, g in df.groupby("oid"):
        mjd = pd.to_numeric(g["mjd"], errors="coerce").to_numpy(float)
        mag = pd.to_numeric(g["mag"], errors="coerce").to_numpy(float)
        err = pd.to_numeric(g["magerr"], errors="coerce").to_numpy(float)
        ok = np.isfinite(mjd) & np.isfinite(mag)
        out[str(int(oid))] = {"mjd": mjd[ok], "mag": mag[ok], "err": err[ok],
                              "filtercode": str(g["filtercode"].iloc[0]) if "filtercode" in g
                              else ""}
    return out


def fetch_ztf_ids(oids: list[str], timeout_s: float = 180.0, retries: int = 2,
                  session=None) -> dict:
    import requests

    s = session or requests
    qs = "&".join(f"ID={o}" for o in oids)
    url = f"{ZTF_API}?{qs}&BAD_CATFLAGS_MASK=32768&FORMAT=csv"
    last = None
    for k in range(int(retries) + 1):
        try:
            r = s.get(url, timeout=timeout_s)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return {"status": "OK", "lcs": parse_ztf_multi(r.text)}
        except Exception as exc:                        # noqa: BLE001
            last = repr(exc)[:300]
            _time.sleep(3.0 * (k + 1))
    return {"status": "FAILED", "error": last}


def fetch_ztf_batched(stars: pd.DataFrame, *, table: str, workers: int = 4,
                      budget_s: float = 3600.0, chunk: int = 400, ids_per_request: int = 40,
                      radius_arcsec: float = 1.5, bright_limit: float = 12.5,
                      on_result=None, upload_fn=None, ids_fn=None) -> dict:
    """ZTF g/r for every star via upload-join ids + multi-ID light curves."""
    from ..ignition.acquire import _t_http_sync, _t_pyvo_sync

    t0 = _time.monotonic()
    counts: dict = {}
    ledger: list = []
    n_done = 0
    rows = stars.reset_index(drop=True)
    q = (f"SELECT p.sid, p.ra AS ra_p, p.dec AS dec_p, o.oid, o.ra, o.dec, o.filtercode, "
         f"o.ngoodobsrel FROM {table} AS o, TAP_UPLOAD.pos AS p WHERE "
         f"1 = CONTAINS(POINT('ICRS', o.ra, o.dec), CIRCLE('ICRS', p.ra, p.dec, {{r}}))")
    import requests

    sess = requests.Session()
    for c0 in range(0, len(rows), int(chunk)):
        if _time.monotonic() - t0 > budget_s:
            break
        sub = rows.iloc[c0:c0 + int(chunk)].reset_index(drop=True)
        tbl, radii = _ztf_upload_table(sub, radius_arcsec)
        qq = q.replace("{r}", f"{float(radii.max()) / 3600.0:.9f}")
        ta = _time.monotonic()
        got, err = None, None
        fns = [upload_fn] if upload_fn else [_t_pyvo_sync, _t_http_sync]
        for fn in fns:
            try:
                got = fn(qq, tbl, 900.0)
                break
            except Exception as exc:                    # noqa: BLE001
                err = repr(exc)[:300]
        led = {"chunk": c0, "n_stars": len(sub), "upload_s": round(_time.monotonic() - ta, 1),
               "n_rows": None if got is None else int(len(got)), "error": err}
        if got is None and c0 == 0 and not counts:
            # the route itself does not work (table/columns/transport): say so
            # and let the caller fall back, rather than failing every star
            ledger.append(led)
            return {"counts": counts, "route_failed": True, "error": err, "ledger": ledger,
                    "elapsed_s": round(_time.monotonic() - t0, 1), "route": "batched",
                    "table": table, "n_not_attempted": int(len(rows))}
        if got is None:
            for sid in sub["source_id"].astype(str):
                rec = {"status": "FAILED", "error": f"upload: {err}"}
                counts["FAILED"] = counts.get("FAILED", 0) + 1
                if on_result:
                    on_result(sid, rec)
            ledger.append(led)
            continue
        omap = match_oids(got, sub, radii)
        all_ids = sorted({o for v in omap.values() for o in v.values()})
        lcs: dict = {}
        failed_ids: set = set()
        tb = _time.monotonic()
        batches = [all_ids[k:k + int(ids_per_request)]
                   for k in range(0, len(all_ids), int(ids_per_request))]
        fetch = ids_fn or (lambda ids: fetch_ztf_ids(ids, session=sess))
        with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as ex:
            for ids, res in zip(batches, ex.map(fetch, batches), strict=False):
                if res.get("status") == "OK":
                    lcs.update(res.get("lcs") or {})
                else:
                    failed_ids.update(ids)
        led.update({"n_oids": len(all_ids), "lc_s": round(_time.monotonic() - tb, 1),
                    "n_batches": len(batches), "n_failed_ids": len(failed_ids)})
        ledger.append(led)
        print(f"[antiphase] ztf chunk {c0}: {len(sub)} stars, {led['n_rows']} object rows, "
              f"{len(all_ids)} oids, upload {led['upload_s']} s, lcs {led['lc_s']} s, "
              f"failed ids {len(failed_ids)}", flush=True)
        for sid in sub["source_id"].astype(str):
            m = omap.get(sid, {})
            if not m:
                rec = {"status": "NO_ROWS"}
            elif any(o in failed_ids for o in m.values()):
                rec = {"status": "FAILED", "error": "light-curve batch failed"}
            else:
                bands = {}
                for b, o in m.items():
                    v = lcs.get(o)
                    if v is None or not len(v["mjd"]):
                        continue
                    med = float(np.median(v["mag"]))
                    bands[b] = {"mjd": v["mjd"], "mag": v["mag"], "err": v["err"], "oid": o,
                                "n": int(len(v["mjd"])), "median_mag": med,
                                "saturation_risk": bool(med < bright_limit)}
                rec = {"status": "OK", "bands": bands} if bands else {"status": "NO_ROWS"}
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            n_done += 1
            if on_result:
                on_result(sid, rec)
    return {"counts": counts, "n_not_attempted": int(len(rows) - sum(counts.values())),
            "elapsed_s": round(_time.monotonic() - t0, 1), "ledger": ledger[:400],
            "route": "batched", "table": table}


def vsx_query(vtype: str, dec_min: float = -30.0, max_lo: float = 12.5, max_hi: float = 17.0,
              limit: int = 2000, timeout_s: float = 120.0) -> dict:
    """VSX (VizieR ``B/vsx/vsx``) variables of one type in the ZTF footprint.

    The type constraint is sent as given and re-applied locally (a type field
    such as ``RCB:`` or ``M+EA`` is kept if its first component is ``vtype``,
    uncertainty colon stripped), so a VizieR string-match quirk cannot widen it.
    """
    import requests

    params = {"-source": "B/vsx/vsx", "-out.max": str(int(limit)),
              "-out": "OID Name RAJ2000 DEJ2000 Type max n_max min Period",
              "Type": f"{vtype}*", "DEJ2000": f">{float(dec_min)}",
              "max": f"{float(max_lo)}..{float(max_hi)}"}
    last = None
    for base in VIZIER_ASU:
        try:
            r = requests.get(base, params=params, timeout=timeout_s)
            r.raise_for_status()
            df = parse_asu_tsv(r.text)
            if not len(df):
                return {"status": "NO_ROWS", "endpoint": base, "rows": []}
            rows = []
            for _, x in df.iterrows():
                t = str(x.get("Type", "")).strip()
                first = t.replace(":", "").split("+")[0].split("|")[0].strip()
                if first != vtype:
                    continue
                try:
                    ra, dec = float(x["RAJ2000"]), float(x["DEJ2000"])
                except (TypeError, ValueError):
                    continue
                rows.append({"name": str(x.get("Name", "")).strip(), "ra": ra, "dec": dec,
                             "vsx_type": t, "max": str(x.get("max", "")).strip(),
                             "period": str(x.get("Period", "")).strip()})
            return {"status": "OK", "rows": rows, "endpoint": base, "n_raw": int(len(df))}
        except Exception as exc:                        # noqa: BLE001
            last = repr(exc)[:300]
    return {"status": "FAILED", "error": last}


def neowise_epochs_many(objs: pd.DataFrame, radius_arcsec: float = 2.5,
                        chunk: int = 150) -> tuple[dict, dict]:
    """NEOWISE epochs for many objects via IGNITION's upload ladder and local grouping."""
    from ..ignition.acquire import fetch_neowise_upload, group_by_star, reduce_star

    eps: dict = {}
    led = {"chunks": [], "n_ok": 0}
    for c0 in range(0, len(objs), int(chunk)):
        sub = objs.iloc[c0:c0 + int(chunk)]
        qr = fetch_neowise_upload(sub, radius_arcsec=radius_arcsec)
        led["chunks"].append({"chunk": c0, "status": qr.status, "n_rows": qr.n_rows,
                              "error": (qr.error or "")[:200]})
        if qr.data is None or not len(qr.data):
            continue
        for sid, raw in group_by_star(qr.data, sub, tol_arcsec=radius_arcsec).items():
            ep, _rec = reduce_star(sid, raw)
            if len(ep):
                eps[sid] = ep
                led["n_ok"] += 1
    return eps, led


def neowise_epochs_cone(ra: float, dec: float, pmra: float = 0.0, pmdec: float = 0.0,
                        radius_arcsec: float = 2.5) -> tuple[pd.DataFrame, dict]:
    """IGNITION's NEOWISE cone -> (epoch table, status record) for one object."""
    from ..ignition.acquire import fetch_neowise_cone, reduce_star

    qr = fetch_neowise_cone(ra, dec, pmra, pmdec, radius_arcsec=radius_arcsec)
    rec = {"status": qr.status, "n_rows": qr.n_rows, "error": (qr.error or "")[:300]}
    if qr.data is None or not len(qr.data):
        return pd.DataFrame(), rec
    ep, star = reduce_star("control", qr.data)
    rec.update({k: star[k] for k in ("n_epochs_w1", "n_epochs_w2", "w1_median", "w2_median",
                                     "frac_nb_gt1", "frac_na_gt0") if k in star})
    return ep, rec


def sesame_resolve(name: str, timeout_s: float = 30.0) -> dict:
    """CDS Sesame: J2000 position (and PM if given) for a name."""
    import urllib.parse

    import requests

    last = None
    for base in SESAME:
        try:
            r = requests.get(base + urllib.parse.quote(name), timeout=timeout_s)
            r.raise_for_status()
            ra = dec = pmra = pmdec = None
            for ln in r.text.splitlines():
                if ln.startswith("%J ") and ra is None:
                    p = ln.split()
                    ra, dec = float(p[1]), float(p[2])
                elif ln.startswith("%P ") and pmra is None:
                    p = ln.split()
                    try:
                        pmra, pmdec = float(p[1]), float(p[2])
                    except (IndexError, ValueError):
                        pass
            if ra is None:
                return {"status": "NOT_RESOLVED", "endpoint": base, "text": r.text[:300]}
            return {"status": "OK", "ra": ra, "dec": dec, "pmra": pmra, "pmdec": pmdec,
                    "endpoint": base}
        except Exception as exc:                        # noqa: BLE001
            last = repr(exc)[:300]
    return {"status": "FAILED", "error": last}


def parse_gaia_alert_csv(text: str) -> pd.DataFrame:
    """Gaia alerts light curve (``Date,JD,averagemag``) -> ``mjd, mag``; 'null'/'untrusted' dropped."""
    rows = []
    for ln in text.splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) < 3:
            continue
        try:
            jd, mag = float(p[1]), float(p[2])
        except ValueError:
            continue
        if np.isfinite(jd) and np.isfinite(mag) and 5.0 < mag < 25.0:
            rows.append({"mjd": jd - 2400000.5, "mag": mag})
    return pd.DataFrame(rows, columns=["mjd", "mag"])


def fetch_gaia_alert(name: str, timeout_s: float = 60.0) -> dict:
    import requests

    try:
        r = requests.get(GAIA_ALERTS.format(name=name), timeout=timeout_s)
        r.raise_for_status()
        d = parse_gaia_alert_csv(r.text)
    except Exception as exc:                            # noqa: BLE001
        return {"status": "FAILED", "error": repr(exc)[:300]}
    if not len(d):
        return {"status": "NO_ROWS"}
    # Gaia G per-transit errors are not in the file; 0.01 mag at G~13-16 is the
    # documented alert-photometry scatter, and the binning uses the MAD anyway.
    return {"status": "OK", "bands": {"G": {"mjd": d["mjd"].to_numpy(float),
                                            "mag": d["mag"].to_numpy(float),
                                            "err": np.full(len(d), 0.01), "n": int(len(d))}}}


def fetch_asassn(ra: float, dec: float, radius_arcsec: float = 5.0, timeout_s: float = 240.0) -> dict:
    """ASAS-SN Sky Patrol raw points per filter (V, g); needs ``pyasassn``."""
    try:
        from pyasassn.client import SkyPatrolClient
    except Exception as exc:                            # noqa: BLE001
        return {"status": "CLIENT_MISSING", "error": repr(exc)[:200]}
    from ..ignition.revet import _timed

    def _q():
        return SkyPatrolClient().cone_search(ra_deg=float(ra), dec_deg=float(dec),
                                             radius=radius_arcsec / 3600.0, catalog="master_list",
                                             download=True, threads=1)
    lcs, err = _timed(_q, timeout_s)
    if lcs is None:
        return {"status": "FAILED", "error": err}
    d = getattr(lcs, "data", None)
    if d is None or not len(d):
        return {"status": "NO_ROWS"}
    d = d.copy()
    if "quality" in d:
        d = d[d["quality"].astype(str) == "G"]
    if "asas_sn_id" in d and len(d):
        d = d[d["asas_sn_id"] == d["asas_sn_id"].value_counts().index[0]]
    out = {"status": "OK", "bands": {}}
    for flt, g in d.groupby("phot_filter"):
        mjd = pd.to_numeric(g["jd"], errors="coerce").to_numpy(float) - 2400000.5
        mag = pd.to_numeric(g["mag"], errors="coerce").to_numpy(float)
        err = pd.to_numeric(g["mag_err"], errors="coerce").to_numpy(float)
        ok = np.isfinite(mjd) & np.isfinite(mag) & (err < 0.2)
        out["bands"][f"asassn_{flt}"] = {"mjd": mjd[ok], "mag": mag[ok], "err": err[ok],
                                         "n": int(ok.sum())}
    return out


def parse_asu_tsv(text: str) -> pd.DataFrame:
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame()
    body = "\n".join([lines[0]] + lines[3:])       # header, units, dashes
    return pd.read_csv(io.StringIO(body), sep="\t", dtype=str)


def gaia_neighbours_vizier(ra: float, dec: float, radius_arcsec: float = 20.0,
                           timeout_s: float = 60.0) -> dict:
    """Gaia DR3 sources in a cone from VizieR ``I/355/gaiadr3`` (epoch 2016.0)."""
    import requests

    params = {"-source": "I/355/gaiadr3", "-c": f"{float(ra):.7f} {float(dec):+.7f}",
              "-c.rs": f"{float(radius_arcsec):.1f}", "-out.max": "200",
              "-out": "Source RA_ICRS DE_ICRS pmRA pmDE Plx Gmag BP-RP", "-out.add": "_r"}
    last = None
    for base in VIZIER_ASU:
        try:
            r = requests.get(base, params=params, timeout=timeout_s)
            r.raise_for_status()
            df = parse_asu_tsv(r.text)
            rows = []
            for _, x in df.iterrows():
                def f(k, x=x):
                    try:
                        return float(str(x.get(k, "")).strip())
                    except ValueError:
                        return float("nan")
                rows.append({"source_id": str(x.get("Source", "")).strip(), "ra": f("RA_ICRS"),
                             "dec": f("DE_ICRS"), "pmra": f("pmRA"), "pmdec": f("pmDE"),
                             "parallax": f("Plx"), "phot_g_mean_mag": f("Gmag"),
                             "bp_rp": f("BP-RP"), "sep_arcsec": f("_r")})
            return {"status": "OK", "rows": rows, "endpoint": base}
        except Exception as exc:                        # noqa: BLE001
            last = repr(exc)[:300]
    return {"status": "FAILED", "error": last}


def allwise_vizier(ra: float, dec: float, radius_arcsec: float = 3.0, timeout_s: float = 60.0) -> dict:
    """Nearest AllWISE source (VizieR ``II/328/allwise``): W1/W2/W3 magnitudes."""
    import requests

    params = {"-source": "II/328/allwise", "-c": f"{float(ra):.7f} {float(dec):+.7f}",
              "-c.rs": f"{float(radius_arcsec):.1f}", "-out.max": "10", "-sort": "_r",
              "-out": "AllWISE W1mag W2mag W3mag ccf qph", "-out.add": "_r"}
    last = None
    for base in VIZIER_ASU:
        try:
            r = requests.get(base, params=params, timeout=timeout_s)
            r.raise_for_status()
            df = parse_asu_tsv(r.text)
            if not len(df):
                return {"status": "NO_ROWS", "endpoint": base}
            x = df.iloc[0]

            def f(k, x=x):
                try:
                    return float(str(x.get(k, "")).strip())
                except ValueError:
                    return float("nan")
            return {"status": "OK", "w1mpro": f("W1mag"), "w2mpro": f("W2mag"),
                    "w3mpro": f("W3mag"), "sep_arcsec": f("_r"),
                    "cc_flags": str(x.get("ccf", "")).strip(),
                    "ph_qual": str(x.get("qph", "")).strip(), "endpoint": base}
        except Exception as exc:                        # noqa: BLE001
            last = repr(exc)[:300]
    return {"status": "FAILED", "error": last}


__all__ = ["ZTF_DEC_MIN", "allwise_vizier", "fetch_asassn", "fetch_gaia_alert", "fetch_ztf", "fetch_ztf_many",
           "gaia_neighbours_vizier", "neowise_epochs_cone", "parse_asu_tsv",
           "parse_gaia_alert_csv", "parse_ztf_csv", "sesame_resolve"]
