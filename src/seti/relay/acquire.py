"""Runner-only archive access for RELAY, every transport injectable.

Four services, four transports, none of them opened by a test:

* **Gaia DR3** (``gea.esac.esa.int`` TAP, parallax shells; VizieR ``I/355``
  and the GCNS ``J/A+A/649/A6`` as the non-ESA routes) --- ``tap_fn(adql,
  endpoint) -> DataFrame`` for ESA, ``query_fn(adql) -> DataFrame`` for VizieR
  (:func:`seti.metronome.acquire.tap_query`, which carries the mirror ladder,
  the circuit breaker and the non-TAP ASU fallback).
* **Breakthrough Listen open data** (``seti.berkeley.edu/opendata/api/...``)
  --- ``fetch_json_fn(url, params) -> object``.  The routes are the ones the
  archive's own README documents; every response's top-level shape and the
  keys of its first record are written to the probe verbatim, and the
  per-file fields (telescope, MJD, frequency, file type) are resolved by
  regex over whatever keys the service actually returns, never assumed.
* **SIMBAD** TAP --- names to Gaia DR3 ids through the ``ident`` table, so a
  BL target string becomes a ``source_id`` without a positional guess.
* **VizieR hit catalogues** --- discovery by id prefix and by keyword over
  ``TAP_SCHEMA``, columns read at runtime, roles resolved by exact regex.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..metronome.acquire import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_ZERO,
    AcquisitionLog,
    _canon,
    list_tables,
    resolve_columns,
    search_tables,
    table_columns,
    tap_query,
    unquote_table,
)

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"
SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap"
BL_BASE = "http://seti.berkeley.edu/opendata"

ROUTE_ESA = "esa_gaia_tap"
ROUTE_VIZIER_DR3 = "vizier_I/355"
ROUTE_GCNS = "vizier_gcns"

STATUS_PARTIAL = "PARTIAL_SAMPLE"

GAIA_ESA_QUERY = """
SELECT source_id, ra, dec, parallax, parallax_error, parallax_over_error,
       pmra, pmra_error, pmdec, pmdec_error,
       radial_velocity, radial_velocity_error, rv_template_teff, grvs_mag,
       phot_g_mean_mag, bp_rp, ruwe, non_single_star
FROM gaiadr3.gaia_source
WHERE parallax >= {plx_lo} AND parallax < {plx_hi}
  AND parallax_over_error > {poe}
  AND ruwe < {ruwe}
"""

# Standard column -> candidate spellings on the non-ESA routes (canonicalised).
GAIA_ROLE_PATTERNS: dict[str, list[str]] = {
    "source_id": [r"^source_id$", r"^source$", r"^gaiadr3$", r"^gaiaedr3$", r"^dr3name$",
                  r"^gaia$", r"^gaiaid$"],
    "ra": [r"^ra$", r"^ra_icrs$", r"^raj2000$", r"^ra_deg$"],
    "dec": [r"^dec$", r"^de_icrs$", r"^dej2000$", r"^dec_deg$", r"^de$"],
    "parallax": [r"^parallax$", r"^plx$"],
    "parallax_error": [r"^parallax_error$", r"^e_plx$"],
    "pmra": [r"^pmra$", r"^pm_ra$"],
    "pmdec": [r"^pmdec$", r"^pmde$", r"^pm_dec$"],
    "radial_velocity": [r"^radial_velocity$", r"^rv$", r"^rvdr2$", r"^rv_dr2$", r"^hrv$"],
    "radial_velocity_error": [r"^radial_velocity_error$", r"^e_rv$", r"^e_rvdr2$", r"^e_hrv$"],
    "phot_g_mean_mag": [r"^phot_g_mean_mag$", r"^gmag$"],
    "bp_rp": [r"^bp_rp$", r"^bp-rp$", r"^bprp$"],
    "ruwe": [r"^ruwe$"],
    "grvs_mag": [r"^grvs_mag$", r"^grvsmag$"],
    "rv_template_teff": [r"^rv_template_teff$", r"^tefftemp$"],
}
GAIA_STANDARD = ["source_id", "ra", "dec", "parallax", "parallax_error", "pmra", "pmdec",
                 "radial_velocity", "radial_velocity_error", "phot_g_mean_mag", "bp_rp",
                 "ruwe", "grvs_mag", "rv_template_teff"]


# ---------------------------------------------------------------------------
# transports
# ---------------------------------------------------------------------------
def pyvo_tap(adql: str, endpoint: str, *, maxrec: int | None = None):
    """Default TAP transport: async first, sync on failure; raises on both."""
    import pyvo  # noqa: PLC0415  runner-only

    svc = pyvo.dal.TAPService(endpoint)
    try:
        job = svc.run_async(adql, maxrec=maxrec)
        return job.to_table().to_pandas()
    except Exception as exc:                              # noqa: BLE001
        print(f"[relay/acquire] async TAP failed at {endpoint} ({exc!r}); trying sync")
        return svc.search(adql, maxrec=maxrec).to_table().to_pandas()


def requests_json(url: str, params: dict | None = None, *, timeout: float = 60.0):
    """Default HTTP-JSON transport for the BL open-data API."""
    import requests  # noqa: PLC0415

    r = requests.get(url, params=params or None, timeout=timeout,
                     headers={"User-Agent": "seti-relay/1.0 (technosignature research)"})
    r.raise_for_status()
    try:
        return r.json()
    except ValueError:
        return {"_non_json_body": r.text[:2000], "_status": r.status_code}


class QueryTimeout(TimeoutError):
    """An archive call that did not answer inside its clock."""


def with_timeout(fn, timeout_s: float | None):
    """Wrap a transport so one hung archive call cannot eat the run.

    pyvo's ``run_async`` polls a remote job with no time limit of its own: a
    VizieR job that never answers holds the whole stage until the workflow cap
    kills it and NOTHING is written (the failure mode `docs/arc.md` records).
    The call runs on a daemon thread --- daemon so a wedged one cannot keep the
    interpreter alive at exit --- and is abandoned after ``timeout_s``.  The
    caller then sees an ordinary failed query, which every stage already
    records as a failure, never as a zero-row answer about the sky.
    """
    if not timeout_s or float(timeout_s) <= 0:
        return fn

    def _wrapped(*args, **kwargs):
        box: dict = {}

        def _go():
            try:
                box["value"] = fn(*args, **kwargs)
            except BaseException as exc:                  # noqa: BLE001
                box["error"] = exc

        th = threading.Thread(target=_go, daemon=True)
        th.start()
        th.join(float(timeout_s))
        if th.is_alive():
            raise QueryTimeout(f"no answer within {float(timeout_s):.0f} s")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    return _wrapped


def _retry(fn, retries: int = 3, label: str = "call", base_sleep: float = 3.0):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:                          # noqa: BLE001
            last = exc
            print(f"[relay/acquire] {label} attempt {attempt + 1}/{retries} failed: {exc!r}")
            if attempt < retries - 1:
                time.sleep(base_sleep * 2 ** attempt)
    raise RuntimeError(f"{label} failed after {retries} attempts: {last!r}")


# ---------------------------------------------------------------------------
# Gaia: the 100 pc sample
# ---------------------------------------------------------------------------
@dataclass
class GaiaAcquisition:
    df: pd.DataFrame
    route: str
    status: str
    shells: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.df
        return {"route": self.route, "status": self.status, "n_rows": int(len(d)),
                "n_with_rv": int(np.isfinite(pd.to_numeric(d.get("radial_velocity"),
                                                            errors="coerce")).sum())
                if len(d) and "radial_velocity" in d else 0,
                "shells": self.shells, "errors": self.errors[:20]}


def canonicalise_gaia(df: pd.DataFrame) -> pd.DataFrame:
    """Map any route's columns onto the standard names; missing ones are NaN."""
    if df is None or not len(df):
        return pd.DataFrame(columns=GAIA_STANDARD)
    roles = resolve_columns(list(df.columns), GAIA_ROLE_PATTERNS)
    out = pd.DataFrame(index=df.index)
    for std in GAIA_STANDARD:
        col = roles.get(std)
        if col is None:
            out[std] = np.nan
        elif std == "source_id":
            out[std] = df[col].map(_clean_source_id)
        else:
            out[std] = pd.to_numeric(df[col], errors="coerce")
    out = out[out["source_id"].astype(str).str.len() > 0]
    out = out.drop_duplicates("source_id").reset_index(drop=True)
    out["source_id"] = out["source_id"].astype(str)
    return out


def _clean_source_id(v) -> str:
    s = str(v).strip()
    if s.lower() in ("", "nan", "none", "<na>"):
        return ""
    s = re.sub(r"(?i)^gaia\s*(e?dr3)?\s*", "", s)
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def fetch_gaia_esa(conf: dict, *, tap_fn=None, log: AcquisitionLog | None = None,
                   max_rows: int | None = None) -> GaiaAcquisition:
    """The 100 pc sample from ESA in parallax shells; a lost shell is PARTIAL."""
    tap_fn = tap_fn or pyvo_tap
    log = log or AcquisitionLog(prefix="relay/acquire")
    edges = [float(x) for x in conf["parallax_shells_mas"]]
    frames, shells, errors = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        adql = GAIA_ESA_QUERY.format(plx_lo=lo, plx_hi=hi, poe=conf["parallax_over_error_min"],
                                     ruwe=conf["ruwe_max"])
        cap = int(conf.get("max_rows_per_shell", 400000))
        if max_rows:
            cap = min(cap, int(max_rows))
            adql = adql.replace("SELECT source_id", f"SELECT TOP {cap} source_id", 1)
        rec = {"plx_lo": lo, "plx_hi": hi, "status": STATUS_FAILED, "n_rows": 0}
        try:
            df = _retry(lambda adql=adql, cap=cap: tap_fn(adql, GAIA_TAP, maxrec=cap),
                        retries=3, label=f"ESA shell [{lo},{hi}) mas")
            df = df.rename(columns={c: c.lower() for c in df.columns})
            rec.update(status=STATUS_OK if len(df) else STATUS_ZERO, n_rows=int(len(df)),
                       truncated=bool(len(df) >= cap))
            log.record("gaia_esa_shell", adql, rows=len(df), extra={"plx_lo": lo, "plx_hi": hi})
            frames.append(df)
        except Exception as exc:                          # noqa: BLE001
            rec["error"] = repr(exc)[:500]
            errors.append(f"shell [{lo},{hi}): {exc!r}")
            log.record("gaia_esa_shell", adql, error=repr(exc), extra={"plx_lo": lo, "plx_hi": hi})
        shells.append(rec)
        if max_rows and sum(s["n_rows"] for s in shells) >= max_rows:
            break
    df = canonicalise_gaia(pd.concat(frames, ignore_index=True)) if frames else canonicalise_gaia(None)
    n_fail = sum(1 for s in shells if s["status"] == STATUS_FAILED)
    status = STATUS_FAILED if not len(df) else (STATUS_PARTIAL if n_fail else STATUS_OK)
    return GaiaAcquisition(df, ROUTE_ESA, status, shells, errors)


def fetch_gaia_vizier(conf: dict, *, query_fn=None, log: AcquisitionLog | None = None,
                      table: str = "I/355/gaiadr3", max_rows: int | None = None,
                      fetch_fn=None) -> GaiaAcquisition:
    """The same sample over VizieR's DR3 mirror (or the GCNS table), one query per shell."""
    query_fn = query_fn or tap_query
    log = log or AcquisitionLog(prefix="relay/acquire")
    route = ROUTE_GCNS if "649/A6" in table else ROUTE_VIZIER_DR3
    try:
        cols = table_columns(table, query_fn=query_fn, fetch_fn=fetch_fn)
    except Exception as exc:                              # noqa: BLE001
        log.record("gaia_vizier_columns", table, error=repr(exc))
        return GaiaAcquisition(canonicalise_gaia(None), route, STATUS_FAILED, [],
                               [f"columns: {exc!r}"])
    roles = resolve_columns(cols, GAIA_ROLE_PATTERNS)
    need = ("source_id", "ra", "dec", "parallax", "pmra", "pmdec")
    missing = [r for r in need if r not in roles]
    if missing:
        log.record("gaia_vizier_columns", table, rows=len(cols),
                   extra={"missing_roles": missing, "columns": cols[:60]})
        return GaiaAcquisition(canonicalise_gaia(None), route, STATUS_FAILED, [],
                               [f"{table} lacks roles {missing}"])
    sel = ", ".join(f'"{roles[r]}"' for r in GAIA_STANDARD if r in roles)
    plx, ruwe = roles["parallax"], roles.get("ruwe")
    edges = [float(x) for x in conf["parallax_shells_mas"]]
    frames, shells, errors = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        cap = int(conf.get("max_rows_per_shell", 400000))
        if max_rows:
            cap = min(cap, int(max_rows))
        where = f'"{plx}" >= {lo} AND "{plx}" < {hi}'
        if ruwe:
            where += f' AND "{ruwe}" < {conf["ruwe_max"]}'
        if roles.get("parallax_error"):
            where += f' AND "{plx}" > {conf["parallax_over_error_min"]} * "{roles["parallax_error"]}"'
        adql = f'SELECT TOP {cap} {sel} FROM "{table}" WHERE {where}'
        rec = {"plx_lo": lo, "plx_hi": hi, "status": STATUS_FAILED, "n_rows": 0}
        try:
            df = query_fn(adql)
            df = df if df is not None else pd.DataFrame()
            rec.update(status=STATUS_OK if len(df) else STATUS_ZERO, n_rows=int(len(df)),
                       truncated=bool(len(df) >= cap), route=str(df.attrs.get("route", "")))
            log.record("gaia_vizier_shell", adql, rows=len(df))
            frames.append(df)
        except Exception as exc:                          # noqa: BLE001
            rec["error"] = repr(exc)[:500]
            errors.append(f"shell [{lo},{hi}): {exc!r}")
            log.record("gaia_vizier_shell", adql, error=repr(exc))
        shells.append(rec)
        if max_rows and sum(s["n_rows"] for s in shells) >= max_rows:
            break
    df = canonicalise_gaia(pd.concat(frames, ignore_index=True)) if frames else canonicalise_gaia(None)
    n_fail = sum(1 for s in shells if s["status"] == STATUS_FAILED)
    status = STATUS_FAILED if not len(df) else (STATUS_PARTIAL if n_fail else STATUS_OK)
    return GaiaAcquisition(df, route, status, shells, errors)


def fetch_gaia_sample(conf: dict, *, tap_fn=None, query_fn=None, log=None,
                      max_rows: int | None = None, routes=("esa", "vizier", "gcns"),
                      fetch_fn=None) -> GaiaAcquisition:
    """ESA first; VizieR DR3 when ESA fails or is partial; GCNS last."""
    log = log or AcquisitionLog(prefix="relay/acquire")
    best: GaiaAcquisition | None = None
    for route in routes:
        if route == "esa":
            acq = fetch_gaia_esa(conf, tap_fn=tap_fn, log=log, max_rows=max_rows)
        elif route == "vizier":
            acq = fetch_gaia_vizier(conf, query_fn=query_fn, log=log, max_rows=max_rows,
                                    fetch_fn=fetch_fn)
        else:
            acq = fetch_gaia_vizier(conf, query_fn=query_fn, log=log, max_rows=max_rows,
                                    table="J/A+A/649/A6/gcns", fetch_fn=fetch_fn)
        print(f"[relay/acquire] gaia route {acq.route}: {acq.status} {len(acq.df)} rows")
        if acq.status == STATUS_OK:
            return acq
        if best is None or len(acq.df) > len(best.df):
            best = acq
    return best if best is not None else GaiaAcquisition(canonicalise_gaia(None), "none",
                                                          STATUS_FAILED)


def gaia_count(conf: dict, *, tap_fn=None) -> dict:
    tap_fn = tap_fn or pyvo_tap
    adql = ("SELECT COUNT(*) AS n FROM gaiadr3.gaia_source WHERE parallax > "
            f"{conf['parallax_min_mas']} AND parallax_over_error > {conf['parallax_over_error_min']}"
            f" AND ruwe < {conf['ruwe_max']}")
    try:
        df = tap_fn(adql, GAIA_TAP, maxrec=1)
        n = int(pd.to_numeric(df.iloc[0, 0]))
        return {"status": STATUS_OK, "n": n, "adql": adql}
    except Exception as exc:                              # noqa: BLE001
        return {"status": STATUS_FAILED, "n": None, "adql": adql, "error": repr(exc)[:500]}


# ---------------------------------------------------------------------------
# Breakthrough Listen open data
# ---------------------------------------------------------------------------
_KEYS = {
    "telescope": [r"telescope", r"^tel$", r"observatory", r"facility"],
    "target": [r"^target(_?name)?$", r"^source(_?name)?$", r"^name$", r"^object$"],
    "mjd": [r"^mjd$", r"tstart", r"^t_?start$", r"start_?mjd", r"^time$", r"utc", r"date", r"obs_?time"],
    "freq_lo": [r"freq.*(start|lo|min|low)", r"^fch1$", r"f_?start", r"lower"],
    "freq_hi": [r"freq.*(end|hi|max|high)", r"f_?end", r"upper"],
    "freq_centre": [r"cent(er|re).*freq", r"^freq(uency)?$", r"^fc$", r"obs_?freq"],
    "file_type": [r"file_?type", r"^type$", r"format", r"^ext(ension)?$"],
    "url": [r"^url$", r"^path$", r"^md5.*", r"^file(name|_name|path|_path)?$", r"^href$"],
    "size": [r"size"],
    "receiver": [r"receiver", r"^rcvr$", r"backend", r"^band$"],
    "ra": [r"^ra(_?deg)?$", r"^ra_?j2000$"],
    "dec": [r"^dec(_?deg)?$", r"^de[c]?_?j2000$"],
}

_MJD_IN_NAME = re.compile(r"(?<!\d)(5[0-9]{4}|6[0-9]{4})_(\d{5})(?!\d)")
_BAND_IN_NAME = re.compile(r"(?i)(?:^|[_./-])(L|S|C|X|UHF|Ku)(?:band)?(?:[_./-]|$)")


def _pick(rec: dict, role: str):
    """First key of ``rec`` matching the role's regexes (canonicalised, ordered)."""
    keys = {k: _canon(k) for k in rec}
    for pat in _KEYS[role]:
        for k, ck in keys.items():
            if re.search(pat, ck):
                return rec[k]
    return None


def _as_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def mjd_from_string(s) -> float:
    """MJD from a BL file name (``..._57650_67573_...`` = MJD + seconds of day)."""
    if s is None:
        return float("nan")
    m = _MJD_IN_NAME.search(str(s))
    if not m:
        return float("nan")
    return float(m.group(1)) + float(m.group(2)) / 86400.0


def _mjd_from_value(v) -> float:
    if v is None:
        return float("nan")
    x = _as_float(v)
    if np.isfinite(x):
        if 40000 < x < 80000:
            return x
        if 2440000 < x < 2480000:      # JD
            return x - 2400000.5
        if 1e9 < x < 2e9:              # unix seconds
            return x / 86400.0 + 40587.0
        return float("nan")
    try:
        ts = pd.Timestamp(str(v))
        if pd.isna(ts):
            return float("nan")
        return float(ts.to_julian_date() - 2400000.5)
    except (ValueError, TypeError):
        return float("nan")


def normalise_bl_file(rec) -> dict:
    """One BL file record -> {telescope, target, mjd, freq_lo/hi_ghz, file_type, name, band}."""
    if not isinstance(rec, dict):
        rec = {"file": str(rec)}
    name = _pick(rec, "url")
    mjd = _mjd_from_value(_pick(rec, "mjd"))
    if not np.isfinite(mjd):
        mjd = mjd_from_string(name)
    flo, fhi, fc = _as_float(_pick(rec, "freq_lo")), _as_float(_pick(rec, "freq_hi")), \
        _as_float(_pick(rec, "freq_centre"))
    # units: BL headers quote MHz; anything above 100 is MHz, above 1e5 is Hz
    flo, fhi, fc = (_to_ghz(x) for x in (flo, fhi, fc))
    if not np.isfinite(fc) and np.isfinite(flo) and np.isfinite(fhi):
        fc = 0.5 * (flo + fhi)
    band = None
    m = _BAND_IN_NAME.search(str(name or ""))
    if m:
        band = m.group(1).upper()
    return {"telescope": _clean_str(_pick(rec, "telescope")), "target": _clean_str(_pick(rec, "target")),
            "mjd": mjd, "freq_lo_ghz": flo, "freq_hi_ghz": fhi, "freq_centre_ghz": fc,
            "file_type": _clean_str(_pick(rec, "file_type")), "name": _clean_str(name),
            "receiver": _clean_str(_pick(rec, "receiver")), "band_hint": band,
            "size": _as_float(_pick(rec, "size")),
            "ra": _as_float(_pick(rec, "ra")), "dec": _as_float(_pick(rec, "dec"))}


def _to_ghz(x: float) -> float:
    if not np.isfinite(x) or x <= 0:
        return float("nan")
    if x > 1e5:
        return x / 1e9
    if x > 100:
        return x / 1e3
    return x


def _clean_str(v):
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def unwrap_records(payload) -> list:
    """The list inside a BL API response, whatever it is wrapped in."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("data", "results", "files", "targets", "items", "records", "rows"):
            if isinstance(payload.get(k), list):
                return payload[k]
        for v in payload.values():
            if isinstance(v, list) and v and isinstance(v[0], (dict, str)):
                return v
    return []


def target_names_from(payload) -> list[str]:
    names = []
    for item in unwrap_records(payload):
        if isinstance(item, str):
            names.append(item.strip())
        elif isinstance(item, dict):
            v = _pick(item, "target")
            if v is None and len(item) == 1:
                v = next(iter(item.values()))
            if v is not None:
                names.append(str(v).strip())
    return [n for n in dict.fromkeys(names) if n]


class BLOpenData:
    """Thin client over the documented routes; every call is recorded."""

    def __init__(self, conf: dict, *, fetch_json_fn=None, log: AcquisitionLog | None = None,
                 sleep_fn=time.sleep):
        self.conf = conf
        self.base = str(conf.get("base_url") or BL_BASE).rstrip("/")
        self.routes = conf.get("routes") or {}
        self.fetch = fetch_json_fn or (lambda url, params: requests_json(
            url, params, timeout=float(conf.get("timeout_s", 60))))
        self.log = log or AcquisitionLog(prefix="relay/bl")
        self.sleep = sleep_fn
        self.calls: list[dict] = []

    def _url(self, key: str) -> str:
        return f"{self.base}/{self.routes.get(key, key).lstrip('/')}"

    def get(self, key: str, params: dict | None = None) -> tuple[object, dict]:
        url = self._url(key)
        rec = {"url": url, "params": params or {}, "status": STATUS_FAILED}
        try:
            payload = self.fetch(url, params or {})
            recs = unwrap_records(payload)
            rec.update(status=STATUS_OK if recs else STATUS_ZERO, n_records=len(recs),
                       shape=_shape(payload))
            self.log.record(f"bl:{key}", json.dumps({"url": url, "params": params or {}}),
                            rows=len(recs))
        except Exception as exc:                          # noqa: BLE001
            payload = None
            rec["error"] = repr(exc)[:500]
            self.log.record(f"bl:{key}", json.dumps({"url": url, "params": params or {}}),
                            error=repr(exc))
        self.calls.append(rec)
        return payload, rec

    def list_targets(self) -> tuple[list[str], dict]:
        payload, rec = self.get("targets")
        names = target_names_from(payload) if payload is not None else []
        rec["n_targets"] = len(names)
        rec["head"] = _head(payload)
        return names, rec

    def list_simple(self, key: str) -> tuple[list[str], dict]:
        payload, rec = self.get(key)
        vals = []
        for item in unwrap_records(payload) if payload is not None else []:
            vals.append(str(item if not isinstance(item, dict) else
                            (item.get("name") or next(iter(item.values())))).strip())
        rec["head"] = _head(payload)
        return [v for v in dict.fromkeys(vals) if v], rec

    def query_files(self, target: str, *, limit: int | None = None, **extra) -> tuple[list[dict], dict]:
        params = {"target": target, "limit": int(limit or self.conf.get("per_target_limit", 200))}
        params.update({k: v for k, v in extra.items() if v is not None})
        payload, rec = self.get("query_files", params)
        recs = unwrap_records(payload) if payload is not None else []
        rec["head"] = _head(payload)
        rec["keys"] = sorted(recs[0].keys()) if recs and isinstance(recs[0], dict) else []
        files = [normalise_bl_file(r) for r in recs]
        for f in files:
            f.setdefault("target", target)
            if not f["target"]:
                f["target"] = target
        return files, rec

    def query_cone(self, ra: float, dec: float, radius_deg: float, *, limit: int = 500
                   ) -> tuple[list[dict], dict]:
        params = {"target": "", "pos-ra": float(ra), "pos-dec": float(dec),
                  "pos-rad": float(radius_deg), "limit": int(limit)}
        payload, rec = self.get("query_files", params)
        recs = unwrap_records(payload) if payload is not None else []
        rec["head"] = _head(payload)
        return [normalise_bl_file(r) for r in recs], rec


def _shape(payload) -> str:
    if isinstance(payload, list):
        inner = type(payload[0]).__name__ if payload else "empty"
        return f"list[{inner}] n={len(payload)}"
    if isinstance(payload, dict):
        return "dict{" + ",".join(list(payload.keys())[:12]) + "}"
    return type(payload).__name__


def _head(payload, limit: int = 1200) -> str:
    try:
        return json.dumps(payload, default=str)[:limit]
    except Exception:                                     # noqa: BLE001
        return str(payload)[:limit]


# ---------------------------------------------------------------------------
# SIMBAD name resolution -> Gaia DR3 id
# ---------------------------------------------------------------------------
_PREFIX_RE = re.compile(r"^([A-Za-z]+[A-Za-z*]*)[\s_-]*(\d.*)$")


def name_variants(name: str) -> list[str]:
    """Spellings SIMBAD may know for a BL target string."""
    s = str(name).strip()
    out = [s, s.replace("_", " ")]
    m = _PREFIX_RE.match(s.replace("_", " "))
    if m:
        pre, num = m.group(1), m.group(2).strip()
        out += [f"{pre} {num}", f"{pre}{num}"]
        up = pre.upper()
        if up in ("HIP", "HD", "GJ", "GL", "LHS", "TIC", "KIC", "TOI", "KOI", "HR", "BD", "LP",
                  "WISE", "NLTT", "G", "L", "LTT", "TYC", "2MASS", "WOLF", "ROSS", "LSPM",
                  "LUYTEN", "KEPLER", "EPIC"):
            out += [f"{up} {num}"]
        if up in ("GL", "GLIESE"):
            out += [f"GJ {num}"]
        if up == "KEPLER":
            out += [f"Kepler-{num}"]
        if up == "TOI":
            out += [f"TOI-{num}"]
    return [v for v in dict.fromkeys(v.strip() for v in out) if v]


def _sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def resolve_names_simbad(names, *, tap_fn=None, endpoint: str = SIMBAD_TAP, batch: int = 120,
                         log: AcquisitionLog | None = None) -> pd.DataFrame:
    """asked name -> (main_id, ra, dec, plx, gaia_dr3) via SIMBAD's ident table."""
    tap_fn = tap_fn or pyvo_tap
    log = log or AcquisitionLog(prefix="relay/resolve")
    names = [str(n) for n in names if str(n).strip()]
    variant_of: dict[str, str] = {}
    for n in names:
        for v in name_variants(n):
            variant_of.setdefault(v, n)
    variants = list(variant_of)
    rows = []
    for i in range(0, len(variants), int(batch)):
        chunk = variants[i:i + int(batch)]
        in_list = ", ".join(_sql_str(v) for v in chunk)
        adql = ("SELECT i.id AS asked, b.main_id, b.ra, b.dec, b.plx_value, b.pmra, b.pmdec, "
                "b.rvz_radvel, g.id AS gaia FROM ident AS i JOIN basic AS b ON i.oidref = b.oid "
                "LEFT JOIN ident AS g ON g.oidref = b.oid AND g.id LIKE 'Gaia DR3 %' "
                f"WHERE i.id IN ({in_list})")
        cap = 10 * len(chunk)
        try:
            df = _retry(lambda adql=adql, cap=cap: tap_fn(adql, endpoint, maxrec=cap),
                        retries=3, label="SIMBAD ident batch")
            df = df.rename(columns={c: c.lower() for c in df.columns})
            log.record("simbad_ident", adql[:300], rows=len(df))
            rows.append(df)
        except Exception as exc:                          # noqa: BLE001
            log.record("simbad_ident", adql[:300], error=repr(exc))
    if not rows:
        return pd.DataFrame(columns=["asked", "main_id", "ra", "dec", "plx", "gaia_dr3", "name"])
    df = pd.concat(rows, ignore_index=True)
    df["asked"] = df["asked"].astype(str).str.strip()
    df["name"] = df["asked"].map(lambda v: variant_of.get(v, variant_of.get(v.strip(), "")))
    df["gaia_dr3"] = df["gaia"].map(_clean_source_id) if "gaia" in df else ""
    df = df.rename(columns={"plx_value": "plx"})
    df["main_id"] = df["main_id"].astype(str).str.strip()
    # one row per asked name: prefer the row that carries a Gaia id
    df["_has_gaia"] = df["gaia_dr3"].astype(str).str.len() > 0
    df = (df.sort_values(["name", "_has_gaia"], ascending=[True, False])
            .drop_duplicates("name").drop(columns=["_has_gaia"]).reset_index(drop=True))
    keep = [c for c in ("name", "asked", "main_id", "ra", "dec", "plx", "pmra", "pmdec",
                        "rvz_radvel", "gaia_dr3") if c in df]
    return df[keep]


# ---------------------------------------------------------------------------
# published hit / target catalogues on VizieR
# ---------------------------------------------------------------------------
HIT_ROLE_PATTERNS: dict[str, list[str]] = {
    "target": [r"^target$", r"^name$", r"^star$", r"^source$", r"^object$", r"^id$", r"^hip$",
               r"^tic$", r"^simbad$", r"^sname$", r"^src$", r"^field$"],
    "ra": [r"^ra_icrs$", r"^raj2000$", r"^_?ra$", r"^ra_?deg$", r"^radeg$"],
    "dec": [r"^de_icrs$", r"^dej2000$", r"^_?dec?$", r"^dec_?deg$", r"^dedeg$"],
    "freq_mhz": [r"^freq(uency)?$", r"^freq_?mhz$", r"^f_?mhz$", r"^nu$", r"^fstart$",
                 r"^fcen(ter|tre)?$", r"^freq_?start$", r"^f$", r"^freq_?ghz$", r"^f_?ghz$"],
    "drift": [r"^drift(_?rate)?$", r"^drate$", r"^dr$", r"^drift_?hz_?s$", r"^df_?dt$", r"^fdot$",
              r"^ddot$"],
    "snr": [r"^snr$", r"^s_?n$", r"^sn$", r"^sigma$"],
    "mjd": [r"^mjd$", r"^tstart$", r"^t_?start$", r"^obs_?mjd$", r"^epoch$", r"^date$",
            r"^obsdate$", r"^time$", r"^mjd_?start$"],
    "telescope": [r"^tel(escope)?$", r"^obs(ervatory)?$", r"^inst(rument)?$", r"^facility$"],
    "hip": [r"^hip$", r"^hipparcos$"],
    "gaia_id": [r"^gaia$", r"^gaiadr[23]$", r"^dr[23]_?id$", r"^source_?id$", r"^gaia_?id$"],
    "dist": [r"^dist$", r"^distance$", r"^d$", r"^dist_?pc$", r"^plx$", r"^parallax$"],
    "band": [r"^band$", r"^rcvr$", r"^receiver$"],
    "verdict": [r"^class$", r"^flag$", r"^type$", r"^rfi$", r"^note$", r"^comment$", r"^verdict$"],
    "n_hits": [r"^n_?hits?$", r"^nhit$", r"^hits$", r"^nsig$", r"^n_?events?$"],
}

KIND_HITS = "hits"
KIND_TARGETS = "targets"
KIND_OTHER = "other"


def classify_hit_table(columns) -> tuple[str, dict]:
    roles = resolve_columns(columns, HIT_ROLE_PATTERNS)
    if "freq_mhz" in roles and "drift" in roles:
        return KIND_HITS, roles
    if "target" in roles or ("ra" in roles and "dec" in roles) or "hip" in roles:
        return KIND_TARGETS, roles
    return KIND_OTHER, roles


def discover_hit_tables(seeds: dict, keywords, *, query_fn=None, fetch_fn=None,
                        log: AcquisitionLog | None = None, deadline=None) -> list[dict]:
    """Every table under each seed id, plus keyword hits, with real columns and roles."""
    query_fn = query_fn or tap_query
    log = log or AcquisitionLog(prefix="relay/hits")
    found: dict[str, dict] = {}

    def _spent() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    for key, spec in seeds.items():
        if _spent():
            found[f"{key}:not_probed"] = {"seed": key, "table": None, "status": "NOT_PROBED",
                                           "reason": "probe wall clock spent"}
            continue
        cid = str(spec["id"])
        try:
            tabs = list_tables(cid, query_fn=query_fn, fetch_fn=fetch_fn)
            names = [unquote_table(t) for t in tabs["table_name"].tolist()] if len(tabs) else []
            log.record("hit_tables_list", cid, rows=len(names))
        except Exception as exc:                          # noqa: BLE001
            log.record("hit_tables_list", cid, error=repr(exc))
            found[f"{key}:{cid}"] = {"seed": key, "id": cid, "table": None, "status": STATUS_FAILED,
                                     "error": repr(exc)[:400], **_seed_meta(spec)}
            continue
        if not names:
            found[f"{key}:{cid}"] = {"seed": key, "id": cid, "table": None, "status": STATUS_ZERO,
                                     "note": "no table under this id on the reached mirror",
                                     **_seed_meta(spec)}
            continue
        for t in names:
            if not t.startswith(cid):
                continue
            found[t] = _describe(t, key, spec, query_fn, fetch_fn, log)
    for kw in keywords or ():
        if _spent():
            break
        try:
            tabs = search_tables([kw], query_fn=query_fn)
            log.record("hit_tables_search", kw, rows=len(tabs))
        except Exception as exc:                          # noqa: BLE001
            log.record("hit_tables_search", kw, error=repr(exc))
            continue
        for t in [unquote_table(x) for x in tabs["table_name"].tolist()] if len(tabs) else []:
            if t in found or t.startswith(("I/", "II/", "III/", "IV/", "V/", "VI/", "VII/",
                                           "VIII/", "IX/", "B/")):
                continue
            found[t] = _describe(t, f"keyword:{kw}", {}, query_fn, fetch_fn, log)
    return list(found.values())


def _seed_meta(spec: dict) -> dict:
    return {"telescope": spec.get("telescope"), "band": spec.get("band"), "note": spec.get("note")}


def _describe(table: str, seed: str, spec: dict, query_fn, fetch_fn, log) -> dict:
    try:
        cols = table_columns(table, query_fn=query_fn, fetch_fn=fetch_fn)
    except Exception as exc:                              # noqa: BLE001
        return {"seed": seed, "table": table, "status": STATUS_FAILED, "error": repr(exc)[:400],
                "columns": [], "roles": {}, "kind": KIND_OTHER, **_seed_meta(spec)}
    kind, roles = classify_hit_table(cols)
    return {"seed": seed, "table": table, "status": STATUS_OK, "columns": cols, "roles": roles,
            "kind": kind, **_seed_meta(spec)}


def fetch_catalogue_rows(desc: dict, *, query_fn=None, max_rows: int = 200000,
                         log: AcquisitionLog | None = None) -> pd.DataFrame:
    """Rows of one discovered table, only the role columns, standardised names."""
    query_fn = query_fn or tap_query
    log = log or AcquisitionLog(prefix="relay/hits")
    roles = desc.get("roles") or {}
    if not roles:
        return pd.DataFrame()
    sel = ", ".join(f'"{c}"' for c in dict.fromkeys(roles.values()))
    adql = f'SELECT TOP {int(max_rows)} {sel} FROM "{desc["table"]}"'
    try:
        df = query_fn(adql)
    except Exception as exc:                              # noqa: BLE001
        log.record("hit_rows", adql, error=repr(exc))
        return pd.DataFrame()
    df = df if df is not None else pd.DataFrame()
    log.record("hit_rows", adql, rows=len(df))
    out = pd.DataFrame(index=df.index)
    for role, col in roles.items():
        real = col if col in df.columns else next((c for c in df.columns
                                                   if _canon(c) == _canon(col)), None)
        out[role] = df[real] if real is not None else np.nan
    out["table"] = desc["table"]
    out["seed"] = desc.get("seed")
    out["catalogue_telescope"] = desc.get("telescope")
    out["catalogue_band"] = desc.get("band")
    out.attrs["route"] = df.attrs.get("route")
    return out.reset_index(drop=True)


def standardise_hits(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric frequency (MHz), drift (Hz/s), SNR, MJD from whatever the table used."""
    out = df.copy()
    f = pd.to_numeric(out.get("freq_mhz"), errors="coerce")
    # GHz tables (< 100) -> MHz; Hz tables (> 1e7) -> MHz
    f = np.where(f < 100, f * 1e3, np.where(f > 1e7, f / 1e6, f))
    out["freq_mhz"] = f
    out["drift_hz_s"] = pd.to_numeric(out.get("drift"), errors="coerce")
    out["snr"] = pd.to_numeric(out.get("snr"), errors="coerce")
    out["mjd"] = [_mjd_from_value(v) for v in out.get("mjd", pd.Series([None] * len(out)))]
    for c in ("ra", "dec"):
        out[c] = pd.to_numeric(out.get(c), errors="coerce")
    out["target"] = out.get("target", pd.Series([None] * len(out))).map(_clean_str)
    return out


__all__ = [
    "BLOpenData", "GAIA_TAP", "GaiaAcquisition", "HIT_ROLE_PATTERNS", "KIND_HITS", "KIND_OTHER",
    "KIND_TARGETS", "ROUTE_ESA", "ROUTE_GCNS", "ROUTE_VIZIER_DR3", "SIMBAD_TAP", "STATUS_PARTIAL",
    "canonicalise_gaia", "classify_hit_table", "discover_hit_tables", "fetch_catalogue_rows",
    "fetch_gaia_esa", "fetch_gaia_sample", "fetch_gaia_vizier", "gaia_count", "mjd_from_string",
    "name_variants", "normalise_bl_file", "pyvo_tap", "requests_json", "resolve_names_simbad",
    "standardise_hits", "target_names_from", "unwrap_records",
]
