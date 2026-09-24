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
