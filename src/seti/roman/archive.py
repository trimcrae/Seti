"""Archive discovery for ROMAN -- runner-only (the sandbox has no egress).

Roman is pre-launch, so nothing here assumes an answer.  Three stages, each
of which degrades honestly instead of raising:

:func:`probe`           asks every endpoint in ``config/roman.yaml → archive``
                        (IRSA TAP/SIA2/data root, the public S3 buckets, MAST
                        CAOM, the IPAC simulation pages) and the runner's own
                        Python environment what exists, records status /
                        latency / what it saw per endpoint, and derives one
                        ``data_state`` with the evidence that drove it.  A
                        network failure is a recorded ``NO_ARCHIVE_REACHED``,
                        never an exception.
:func:`inventory`       turns the listings into :class:`RomanProduct` rows
                        (level / kind / survey / size / simulated) from the
                        filename patterns Roman's pipeline is documented to use,
                        lists the S3 buckets deeper (paginated, capped), and
                        writes a shard plan.  A simulated product is stamped
                        ``simulated=True`` so no downstream stage can report it
                        as a sky result.
:func:`fetch_to_cache`  pulls one product (``s3://`` translated to HTTPS) with
                        retries and a size cap, leaving a ``.provenance.json``
                        beside it.

Every HTTP call goes through an injectable ``session`` (anything with
``.get``/``.head``) so the offline tests stub the archive; the default is a
``requests.Session`` with a User-Agent.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote, urlencode, urljoin, urlparse

from .schema import RomanProduct, json_safe, utc_now

USER_AGENT = "seti-roman/0.1 (technosignature intake; github.com/trimcrae/Seti)"
DATA_EXTENSIONS = (".fits", ".asdf", ".parquet", ".csv", ".tar", ".gz", ".h5", ".txt")
PROBE_PACKAGES = ("asdf", "roman_datamodels", "gwcs", "fsspec", "s3fs")
MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_ROOT_LINKS = 50
MAX_SIM_LINKS = 200
S3_PAGE_KEYS = 1000

_HREF_RE = re.compile(r"""href\s*=\s*["']?([^"' >]+)""", re.I)
_FILTER_RE = re.compile(r"(?<![a-z0-9])(f\d{3})(?![0-9])")
_SIM_TOKEN_RE = re.compile(r"(?:^|[^a-z])sim")


# --------------------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------------------

def default_session():
    """A ``requests.Session`` with the project User-Agent; ``None`` if requests is absent."""
    try:
        import requests
    except Exception:  # noqa: BLE001
        return None
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _endpoint(name: str, kind: str, url: str) -> dict:
    return {"name": name, "kind": kind, "url": url, "status": "failed", "http_status": None,
            "latency_s": None, "error": None}


def _http(session, method: str, url: str, timeout: float, rec: dict, **kw):
    """One request, its latency and HTTP status recorded into ``rec``.

    Returns the response or ``None``; the exception text lands in ``rec["error"]``
    so the probe record explains every hole.
    """
    t0 = time.time()
    try:
        fn = getattr(session, method)
        r = fn(url, timeout=timeout, **kw)
        rec["latency_s"] = round(time.time() - t0, 3)
        rec["http_status"] = int(getattr(r, "status_code", 0) or 0)
        return r
    except Exception as exc:  # noqa: BLE001
        rec["latency_s"] = round(time.time() - t0, 3)
        rec["error"] = repr(exc)[:300]
        return None


def _body(r, cap: int = MAX_PAGE_BYTES) -> bytes:
    """Up to ``cap`` bytes of a response body, streaming when the client can."""
    it = getattr(r, "iter_content", None)
    if callable(it):
        chunks, n = [], 0
        try:
            for ch in it(chunk_size=65536):
                if not ch:
                    continue
                chunks.append(ch)
                n += len(ch)
                if n >= cap:
                    break
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
        if chunks:
            return b"".join(chunks)[:cap]
    content = getattr(r, "content", None)
    if content is None:
        content = str(getattr(r, "text", "") or "").encode("utf-8", "replace")
    return bytes(content)[:cap]


def _text(r, cap: int = MAX_PAGE_BYTES) -> str:
    txt = getattr(r, "text", None)
    if isinstance(txt, str):
        return txt[:cap]
    return _body(r, cap).decode("utf-8", "replace")


def _headers(r) -> dict:
    try:
        return {str(k).lower(): str(v) for k, v in dict(r.headers or {}).items()}
    except Exception:  # noqa: BLE001
        return {}


def _tap_url(tap: str, query: str) -> str:
    return f"{tap.rstrip('/')}/sync?" + urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                                    "FORMAT": "json", "QUERY": query})


def parse_tap_json(text: str) -> list[dict]:
    """Rows of a TAP ``FORMAT=json`` answer as dicts, tolerant of the dialects
    IRSA (``metadata``/``data``), MAST (``fields``/``data``) and plain
    lists-of-dicts use.  Falls back to CSV with a header row.  Empty on anything else.
    """
    try:
        obj = json.loads(text)
    except Exception:  # noqa: BLE001
        lines = [ln for ln in str(text).splitlines() if ln.strip()]
        if len(lines) >= 2 and "," in lines[0]:
            import csv
            import io
            return list(csv.DictReader(io.StringIO("\n".join(lines))))
        return []
    if isinstance(obj, list):
        return [dict(x) for x in obj if isinstance(x, dict)]
    if not isinstance(obj, dict):
        return []
    cols = None
    for key in ("metadata", "fields", "columns"):
        c = obj.get(key)
        if isinstance(c, list) and c:
            cols = [str(x.get("name") if isinstance(x, dict) else x) for x in c]
            break
    data = obj.get("data")
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            return [dict(x) for x in data]
        if cols:
            return [dict(zip(cols, row, strict=False)) for row in data
                    if isinstance(row, (list, tuple))]
    return []


def _first_value(row: dict):
    for v in row.values():
        return v
    return None


# --------------------------------------------------------------------------------------
# Endpoint probes
# --------------------------------------------------------------------------------------

def _is_simulated_name(text: str) -> bool:
    t = str(text or "").lower()
    return bool(_SIM_TOKEN_RE.search(t) or "openuniverse" in t or "challenge" in t
                or "mock" in t or "theory" in t)


def _probe_irsa_tap(conf: dict, session, timeout: float) -> dict:
    irsa = conf["archive"]["irsa"]
    kws = [str(k).lower() for k in irsa.get("table_keywords") or ["roman"]]
    clauses = [f"LOWER(table_name) LIKE '%{k}%' OR LOWER(description) LIKE '%{k}%'" for k in kws]
    q = "SELECT table_name, description FROM TAP_SCHEMA.tables WHERE " + " OR ".join(clauses)
    rec = _endpoint("irsa_tap", "irsa_tap", _tap_url(irsa["tap"], q))
    rec["tables"] = []
    r = _http(session, "get", rec["url"], timeout, rec)
    if r is None:
        return rec
    if rec["http_status"] != 200:
        rec["status"] = "http_error"
        rec["error"] = _text(r, 300)
        return rec
    try:
        rows = parse_tap_json(_text(r))
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"parse: {exc!r}"
        return rec
    for row in rows:
        low = {str(k).lower(): v for k, v in row.items()}
        name = str(low.get("table_name") or _first_value(row) or "")
        desc = str(low.get("description") or "")
        if not name:
            continue
        rec["tables"].append({"table_name": name, "description": desc[:300],
                              "simulated": _is_simulated_name(name + " " + desc)})
    rec["status"] = "ok"
    return rec


def _probe_irsa_sia2(conf: dict, session, timeout: float) -> dict:
    sia = conf["archive"]["irsa"]["sia2"]
    url = f"{sia.rstrip('/')}?" + urlencode({"POS": "circle 266.4 -28.9 0.01", "MAXREC": "1"})
    rec = _endpoint("irsa_sia2", "irsa_sia2", url)
    r = _http(session, "get", url, timeout, rec)
    if r is None:
        return rec
    rec["content_type"] = _headers(r).get("content-type")
    rec["status"] = "ok" if rec["http_status"] == 200 else "http_error"
    return rec


def _probe_data_root(conf: dict, session, timeout: float) -> dict:
    url = conf["archive"]["irsa"]["data_root"]
    rec = _endpoint("irsa_data_root", "irsa_data_root", url)
    rec["links"] = []
    r = _http(session, "head", url, timeout, rec, allow_redirects=True)
    rec["head_status"] = rec["http_status"]
    r = _http(session, "get", url, timeout, rec, stream=True)
    if r is None:
        return rec
    rec["content_type"] = _headers(r).get("content-type")
    if rec["http_status"] == 200:
        rec["links"] = extract_links(_text(r), url, MAX_ROOT_LINKS)
        rec["status"] = "ok"
    else:
        rec["status"] = "http_error"
    return rec


def extract_links(html: str, base: str, limit: int, data_only: bool = False) -> list[str]:
    """Absolute hrefs found in ``html`` (optionally only data files), first ``limit``."""
    out, seen = [], set()
    for m in _HREF_RE.finditer(html or ""):
        href = m.group(1).strip()
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        absu = urljoin(base, href)
        if data_only and not urlparse(absu).path.lower().endswith(DATA_EXTENSIONS):
            continue
        if absu in seen:
            continue
        seen.add(absu)
        out.append(absu)
        if len(out) >= limit:
            break
    return out


def s3_listing_url(bucket: str, prefix: str = "", delimiter: str | None = "/",
                   max_keys: int = 200, token: str | None = None) -> str:
    params = {"list-type": "2", "prefix": prefix, "max-keys": str(int(max_keys))}
    if delimiter:
        params["delimiter"] = delimiter
    if token:
        params["continuation-token"] = token
    return f"https://{bucket}.s3.amazonaws.com/?" + urlencode(params)


def parse_s3_listing(xml_text: str) -> dict:
    """``CommonPrefixes`` / ``Contents`` (key, size) / truncation of a ListObjectsV2 answer.

    Namespaces are ignored on purpose: AWS answers with the 2006 namespace and
    S3-compatible stores answer without one.  An error document (``<Error>``)
    comes back as ``{"error": <Code>}``.
    """
    out = {"prefixes": [], "contents": [], "is_truncated": False, "next_token": None,
           "error": None}
    root = ET.fromstring(xml_text)
    tag = root.tag.split("}")[-1]
    if tag == "Error":
        code = root.findtext("Code") or ""
        msg = root.findtext("Message") or ""
        out["error"] = f"{code}: {msg}".strip(": ")
        return out
    for el in root.iter():
        t = el.tag.split("}")[-1]
        if t == "CommonPrefixes":
            p = None
            for ch in el:
                if ch.tag.split("}")[-1] == "Prefix":
                    p = ch.text
            if p:
                out["prefixes"].append(p)
        elif t == "Contents":
            key, size = None, None
            for ch in el:
                ct = ch.tag.split("}")[-1]
                if ct == "Key":
                    key = ch.text
                elif ct == "Size":
                    try:
                        size = int(ch.text)
                    except Exception:  # noqa: BLE001
                        size = None
            if key:
                out["contents"].append({"key": key, "size": size})
        elif t == "IsTruncated":
            out["is_truncated"] = str(el.text).strip().lower() == "true"
        elif t == "NextContinuationToken":
            out["next_token"] = el.text
    return out


def _probe_s3(entry: dict, session, timeout: float) -> dict:
    bucket, prefix = str(entry["bucket"]), str(entry.get("prefix") or "")
    url = s3_listing_url(bucket, prefix, "/", 200)
    rec = _endpoint(f"s3:{entry.get('name', bucket)}", "s3", url)
    rec.update(bucket=bucket, prefix=prefix, s3_kind=str(entry.get("kind") or ""),
               prefixes=[], contents=[], exists=None, is_truncated=False)
    r = _http(session, "get", url, timeout, rec)
    if r is None:
        return rec
    body = _text(r)
    try:
        lst = parse_s3_listing(body) if body.strip() else {"error": "empty body"}
    except Exception as exc:  # noqa: BLE001
        lst = {"error": f"xml: {exc!r}"}
    if rec["http_status"] == 200 and not lst.get("error"):
        rec["prefixes"] = lst["prefixes"]
        rec["contents"] = lst["contents"]
        rec["is_truncated"] = bool(lst["is_truncated"])
        rec["exists"] = True
        rec["status"] = "ok"
    else:
        rec["status"] = "http_error"
        rec["error"] = lst.get("error") or body[:200]
        rec["exists"] = False if rec["http_status"] == 404 else None
    return rec


def _probe_mast(conf: dict, session, timeout: float) -> list[dict]:
    mast = conf["archive"]["mast"]
    recs = []
    for coll in mast.get("collections") or []:
        c = str(coll).replace("'", "''")
        q = f"SELECT COUNT(*) AS n FROM dbo.CaomObservation WHERE obs_collection='{c}'"
        rec = _endpoint(f"mast:{coll}", "mast_caom", _tap_url(mast["tap"], q))
        rec.update(collection=str(coll), count=None, simulated=_is_simulated_name(coll))
        r = _http(session, "get", rec["url"], timeout, rec)
        if r is not None:
            if rec["http_status"] == 200:
                try:
                    rows = parse_tap_json(_text(r))
                    v = _first_value(rows[0]) if rows else 0
                    rec["count"] = int(float(v)) if v is not None else 0
                    rec["status"] = "ok"
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = f"parse: {exc!r}"
            else:
                rec["status"] = "http_error"
                rec["error"] = _text(r, 200)
        recs.append(rec)
    return recs


def _probe_simulation_page(entry: dict, session, timeout: float) -> dict:
    url = str(entry["url"])
    rec = _endpoint(f"sim:{entry.get('name', url)}", "simulation_page", url)
    rec.update(sim_kind=str(entry.get("kind") or ""), content_type=None, bytes=None,
               data_links=[])
    r = _http(session, "head", url, timeout, rec, allow_redirects=True)
    rec["head_status"] = rec["http_status"]
    r = _http(session, "get", url, timeout, rec, stream=True)
    if r is None:
        return rec
    h = _headers(r)
    rec["content_type"] = h.get("content-type")
    body = _body(r)
    rec["bytes"] = int(h["content-length"]) if str(h.get("content-length", "")).isdigit() \
        else len(body)
    if rec["http_status"] == 200:
        rec["data_links"] = extract_links(body.decode("utf-8", "replace"), url, MAX_SIM_LINKS,
                                          data_only=True)
        rec["status"] = "ok"
    else:
        rec["status"] = "http_error"
    return rec


def probe_packages(names=PROBE_PACKAGES) -> dict:
    """Importability and version of each runner package; never raises."""
    out = {}
    for name in names:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", None)
            if ver is None:
                try:
                    from importlib.metadata import version
                    ver = version(name)
                except Exception:  # noqa: BLE001
                    ver = None
            out[name] = {"importable": True, "version": str(ver) if ver else None}
        except Exception as exc:  # noqa: BLE001
            out[name] = {"importable": False, "version": None, "error": repr(exc)[:200]}
    return out


# --------------------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------------------

def _reached(rec: dict) -> bool:
    return rec.get("http_status") is not None


def derive_data_state(endpoints: dict) -> tuple[str, list[str]]:
    """``data_state`` and the observations that drove it.

    Mission data: an IRSA table carrying ``roman``/``wfi`` that is not a
    simulation, ``.asdf`` keys in the mission bucket, or a MAST Roman
    (non-simulation) collection with a count.  Simulations only: a simulation
    bucket that lists, a simulation-labelled IRSA table or MAST collection, or a
    GBTDS-/HLTDS-like simulation page that answers.  Nothing reached: no endpoint
    returned an HTTP status at all.
    """
    mission, sim = [], []
    for name, rec in endpoints.items():
        kind = rec.get("kind")
        if kind == "irsa_tap" and rec.get("status") == "ok":
            for t in rec.get("tables") or []:
                n = str(t.get("table_name", "")).lower()
                if ("roman" in n or "wfi" in n) and not t.get("simulated"):
                    mission.append(f"IRSA TAP table {t['table_name']} (not a simulation)")
                elif t.get("simulated"):
                    sim.append(f"IRSA TAP simulation table {t['table_name']}")
        elif kind == "s3" and rec.get("status") == "ok":
            keys = [c["key"] for c in rec.get("contents") or []]
            n_asdf = sum(1 for k in keys if str(k).lower().endswith(".asdf"))
            if rec.get("s3_kind") == "mission" and n_asdf:
                mission.append(f"{name}: {n_asdf} .asdf key(s) in the mission bucket "
                               f"{rec.get('bucket')}")
            elif rec.get("s3_kind") == "simulation" and (keys or rec.get("prefixes")):
                sim.append(f"{name}: simulation bucket {rec.get('bucket')}/{rec.get('prefix')} "
                           f"lists {len(rec.get('prefixes') or [])} prefix(es), {len(keys)} key(s)")
        elif kind == "mast_caom" and rec.get("status") == "ok" and (rec.get("count") or 0) > 0:
            if rec.get("simulated"):
                sim.append(f"MAST collection {rec.get('collection')}: {rec['count']} obs (simulation)")
            else:
                mission.append(f"MAST collection {rec.get('collection')}: {rec['count']} observations")
        elif kind == "simulation_page" and rec.get("status") == "ok" \
                and rec.get("sim_kind") in ("gbtds_like", "hltds_like"):
            sim.append(f"{name}: simulation page answers ({len(rec.get('data_links') or [])} "
                       f"data link(s))")
    n_reached = sum(1 for r in endpoints.values() if _reached(r))
    if mission:
        return "MISSION_DATA_PRESENT", mission
    if sim:
        return "SIMULATIONS_ONLY", sim
    if n_reached == 0:
        return "NO_ARCHIVE_REACHED", ["no endpoint returned an HTTP status"]
    return "NOT_YET_PUBLIC", [f"{n_reached} endpoint(s) answered; none lists a Roman product"]


def probe(conf: dict, session=None, timeout_s: float | None = None) -> dict:
    """Ask every archive endpoint and the local environment what exists.

    Each endpoint is tried inside its own try/except and recorded with status,
    latency and error text; the function never raises, so a run always has a
    ``probe.json`` to explain itself with.  ``data_state`` is derived by
    :func:`derive_data_state` from the evidence listed in ``data_state_evidence``.
    """
    arch = conf.get("archive") or {}
    timeout = float(timeout_s if timeout_s is not None else arch.get("timeout_s", 60))
    session = session if session is not None else default_session()
    endpoints: dict[str, dict] = {}

    def run(label, fn):
        try:
            res = fn()
        except Exception as exc:  # noqa: BLE001
            res = _endpoint(label, label, "")
            res["error"] = f"probe step raised: {exc!r}"[:300]
        recs = res if isinstance(res, list) else [res]
        for rec in recs:
            endpoints[rec["name"]] = rec

    if session is None:
        endpoints["session"] = {**_endpoint("session", "session", ""),
                                "error": "requests not importable and no session given"}
    else:
        run("irsa_tap", lambda: _probe_irsa_tap(conf, session, timeout))
        run("irsa_sia2", lambda: _probe_irsa_sia2(conf, session, timeout))
        run("irsa_data_root", lambda: _probe_data_root(conf, session, timeout))
        for entry in arch.get("s3") or []:
            run(f"s3:{entry.get('name')}", lambda e=entry: _probe_s3(e, session, timeout))
        run("mast", lambda: _probe_mast(conf, session, timeout))
        for entry in arch.get("simulations") or []:
            run(f"sim:{entry.get('name')}",
                lambda e=entry: _probe_simulation_page(e, session, timeout))
    packages = probe_packages()
    state, evidence = derive_data_state(endpoints)
    rec = {"written_utc": utc_now(), "data_state": state, "data_state_evidence": evidence,
           "endpoints": endpoints, "packages": packages,
           "n_endpoints_reached": sum(1 for r in endpoints.values() if _reached(r)),
           "n_endpoints_tried": len(endpoints), "timeout_s": timeout}
    print(f"[roman] probe: data_state={state}; {rec['n_endpoints_reached']}/{len(endpoints)} "
          f"endpoints answered; packages="
          f"{[k for k, v in packages.items() if v['importable']]}")
    return json_safe(rec)


# --------------------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------------------

_KNOWN_FILTERS = {"f062", "f087", "f106", "f129", "f146", "f158", "f184", "f213"}
_LC_RE = re.compile(r"(?:^|[/_.-])lc(?:[_.-]|$)|light_?curves?")


def _tokens(path: str) -> list[str]:
    return [t for t in re.split(r"[/_\-.\s]+", path.lower()) if t]


def survey_from_path(path: str) -> str:
    """GBTDS / HLTDS / HLWAS from the path tokens (``''`` when nothing matches).

    The order matters: ``gbtds`` ends in ``tds`` and ``hlwas`` ends in ``was``,
    so the bulge tokens are tested first and the exact survey names before the
    generic suffixes (``RomanTDS`` / ``RomanWAS`` are the OpenUniverse layouts).
    """
    toks = _tokens(path)
    for t in toks:
        if "gbtds" in t or "bulge" in t or t.startswith("microlens"):
            return "GBTDS"
    for t in toks:
        if "hltds" in t:
            return "HLTDS"
        if "hlwas" in t:
            return "HLWAS"
    for t in toks:
        if t == "tds" or t.endswith("tds"):
            return "HLTDS"
        if t == "was" or t.endswith("was") or t == "wide":
            return "HLWAS"
    return ""


def band_from_path(path: str) -> str | None:
    low = path.lower()
    for m in _FILTER_RE.finditer(low):
        if m.group(1) in _KNOWN_FILTERS:
            return m.group(1).upper()
    if re.search(r"(?<![a-z0-9])g150(?![0-9])", low):
        return "G150"
    if re.search(r"(?<![a-z0-9])p127(?![0-9])", low):
        return "P127"
    return None


def classify_product(uri: str, origin: str = "", simulated_hint: bool | None = None,
                     size_bytes: int | None = None, meta: dict | None = None) -> RomanProduct:
    """One key / href / table name to a :class:`RomanProduct` by filename pattern.

    Level and kind come from the documented Roman suffixes (``_uncal`` L1,
    ``_cal`` L2, ``coadd``/``mosaic``/``_i2d`` L3, light-curve and spectral
    tokens L4, ``catalog``/``cat_``, ``cgi``); anything else is ``unknown`` and
    stays in the inventory so the runner can look at what it did not recognise.
    ``simulated`` is True when the source is a simulation or the path says so.
    """
    path = urlparse(uri).path if "://" in uri else uri
    low = path.lower()
    base = low.rsplit("/", 1)[-1]
    is_table = base.endswith((".parquet", ".csv", ".ecsv", ".fits", ".asdf", ".h5", ".txt"))
    spectral = any(t in low for t in ("x1d", "spec", "grism", "prism"))
    instrument = "WFI"
    level, kind = "unknown", "unknown"
    if "cgi" in _tokens(low) or "/cgi" in low or base.startswith("cgi"):
        instrument, kind = "CGI", "cgi"
        level = "L1" if "_uncal" in base else "L2" if "_cal" in base else "unknown"
    elif "_uncal" in base:
        level, kind = "L1", "uncal"
    elif _LC_RE.search(base) or ("lightcurve" in low and (is_table or "." not in base)):
        level, kind = "L4", "lightcurve"
    elif spectral:
        if "_cal" in base and ("grism" in low or "prism" in low):
            level, kind = "L2", "spectrum_2d"
        else:
            level, kind = "L4", "spectrum_1d"
    elif "coadd" in low or "mosaic" in low or "_i2d" in base:
        level, kind = "L3", "coadd"
    elif "_cal" in base:
        level, kind = "L2", "cal"
    elif "catalog" in low or "cat_" in base or base.startswith("cat"):
        level, kind = "catalog", "catalog"
    simulated = bool(simulated_hint) or _is_simulated_name(low)
    return RomanProduct(uri=uri, level=level, kind=kind, origin=origin,
                        survey=survey_from_path(path), instrument=instrument,
                        band=band_from_path(path), size_bytes=size_bytes, simulated=simulated,
                        meta=dict(meta or {}))


def list_s3_deep(session, bucket: str, prefix: str, timeout: float, max_keys: int = 5000,
                 page: int = S3_PAGE_KEYS, max_requests: int = 200) -> dict:
    """Every key under ``prefix`` (no delimiter, continuation-token pagination),
    capped at ``max_keys`` and ``max_requests``; failures stop the walk and are
    recorded, never raised."""
    out = {"contents": [], "n_requests": 0, "truncated": False, "error": None}
    token = None
    while out["n_requests"] < max_requests:
        url = s3_listing_url(bucket, prefix, None, min(page, max_keys - len(out["contents"])),
                             token)
        rec = {}
        r = _http(session, "get", url, timeout, rec)
        out["n_requests"] += 1
        if r is None or rec.get("http_status") != 200:
            out["error"] = rec.get("error") or f"http {rec.get('http_status')}"
            break
        try:
            lst = parse_s3_listing(_text(r))
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"xml: {exc!r}"
            break
        if lst.get("error"):
            out["error"] = lst["error"]
            break
        out["contents"].extend(lst["contents"])
        if len(out["contents"]) >= max_keys:
            out["truncated"] = bool(lst["is_truncated"] or len(out["contents"]) > max_keys)
            out["contents"] = out["contents"][:max_keys]
            break
        if not lst["is_truncated"] or not lst["next_token"]:
            break
        token = lst["next_token"]
    else:
        out["truncated"] = True
    return out


def plan_shards(products, n_shards: int, kinds=None) -> list[list[str]]:
    """Round-robin the products (largest first) over ``n_shards``; returns URI lists.

    ``kinds`` restricts the plan to those product kinds.  Sorting by size before
    dealing keeps every shard's byte load within one product of the others, so
    the wall clock is one shard, not the sum.
    """
    rows = [p.as_dict() if hasattr(p, "as_dict") else dict(p) for p in products]
    if kinds:
        want = {str(k) for k in kinds}
        rows = [r for r in rows if r.get("kind") in want]
    rows.sort(key=lambda r: -(int(r.get("size_bytes") or 0)))
    n = max(1, int(n_shards))
    n = min(n, len(rows)) if rows else 1
    shards: list[list[str]] = [[] for _ in range(n)]
    for i, r in enumerate(rows):
        shards[i % n].append(str(r["uri"]))
    return shards


def inventory(conf: dict, probe_record: dict, session=None, max_listing: int = 5000,
              n_shards: int = 8, deep: bool = True, timeout_s: float | None = None) -> dict:
    """Every product the probe (and a deeper S3 walk) can see, classified and sharded.

    Sources: S3 listings (probe contents, then a paginated walk of each bucket
    that answered, up to ``max_listing`` keys), the simulation-page data links,
    the IRSA data-root links, and IRSA tables (as table products).  MAST counts
    are carried as counts, not products.  ``verdict`` is ``NO_DATA_REACHED``
    when nothing at all was listed; a simulation is never a sky product.
    """
    arch = conf.get("archive") or {}
    timeout = float(timeout_s if timeout_s is not None else arch.get("timeout_s", 60))
    endpoints = (probe_record or {}).get("endpoints") or {}
    products: dict[str, RomanProduct] = {}
    listings: dict[str, dict] = {}
    mast_counts: dict[str, int | None] = {}

    def add(p: RomanProduct):
        if p.uri not in products:
            products[p.uri] = p

    if deep and session is None:
        session = default_session()
    for name, rec in endpoints.items():
        kind = rec.get("kind")
        if kind == "s3":
            bucket, prefix = str(rec.get("bucket") or ""), str(rec.get("prefix") or "")
            sim_hint = rec.get("s3_kind") == "simulation"
            origin = "irsa_s3"
            contents = list(rec.get("contents") or [])
            walk = None
            if rec.get("status") == "ok" and deep and session is not None and bucket:
                walk = list_s3_deep(session, bucket, prefix, timeout, max_keys=int(max_listing))
                contents = walk["contents"] or contents
            listings[name] = {"bucket": bucket, "prefix": prefix, "n_keys": len(contents),
                              "walk": None if walk is None else
                              {k: v for k, v in walk.items() if k != "contents"}}
            for c in contents:
                key = str(c.get("key") or "")
                if not key or key.endswith("/"):
                    continue
                add(classify_product(f"s3://{bucket}/{key}", origin, sim_hint,
                                     c.get("size"), {"bucket": bucket, "endpoint": name}))
        elif kind == "simulation_page" and rec.get("status") == "ok":
            for href in rec.get("data_links") or []:
                add(classify_product(str(href), "ipac_sim_page", True, None,
                                     {"endpoint": name, "sim_kind": rec.get("sim_kind")}))
        elif kind == "irsa_data_root" and rec.get("status") == "ok":
            for href in rec.get("links") or []:
                if urlparse(str(href)).path.lower().endswith(DATA_EXTENSIONS):
                    add(classify_product(str(href), "irsa_http", None, None, {"endpoint": name}))
        elif kind == "irsa_tap" and rec.get("status") == "ok":
            for t in rec.get("tables") or []:
                tn = str(t.get("table_name") or "")
                p = classify_product(f"irsa-tap://{tn}", "irsa_tap", bool(t.get("simulated")),
                                     None, {"description": t.get("description")})
                if p.kind == "unknown":
                    p.kind, p.level = "catalog", "catalog"
                add(p)
        elif kind == "mast_caom":
            mast_counts[str(rec.get("collection"))] = rec.get("count")

    rows = [p.as_dict() for p in products.values()]
    by_kind: dict[str, int] = {}
    by_level: dict[str, int] = {}
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        by_level[r["level"]] = by_level.get(r["level"], 0) + 1
    n_sim = sum(1 for r in rows if r["simulated"])
    out = {"written_utc": utc_now(), "data_state": (probe_record or {}).get("data_state"),
           "products": rows, "counts_by_kind": by_kind, "counts_by_level": by_level,
           "n_products": len(rows), "n_simulated": n_sim, "n_mission": len(rows) - n_sim,
           "total_bytes": int(sum(int(r["size_bytes"] or 0) for r in rows)),
           "listings": listings, "mast_counts": mast_counts,
           "shards": plan_shards(rows, n_shards),
           "verdict": "INVENTORIED" if rows else "NO_DATA_REACHED"}
    print(f"[roman] inventory: {len(rows)} products ({n_sim} simulated); kinds={by_kind}")
    return json_safe(out)


# --------------------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------------------

def to_https(uri: str) -> str:
    """``s3://bucket/key`` -> the anonymous HTTPS form; anything else unchanged."""
    if str(uri).startswith("s3://"):
        rest = str(uri)[5:]
        bucket, _, key = rest.partition("/")
        return f"https://{bucket}.s3.amazonaws.com/{quote(key)}"
    return str(uri)


def cache_path_for(uri: str, cache_dir: Path) -> Path:
    u = urlparse(to_https(uri))
    host = u.netloc or "local"
    rel = u.path.lstrip("/") or "index"
    return Path(cache_dir) / host / rel


def _sha256_first_mb(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(1024 * 1024))
    return h.hexdigest()


def fetch_to_cache(uri: str, cache_dir: Path, session=None, max_bytes: int | None = None,
                   retries: int = 3, pause: float = 1.0, timeout: float = 300.0) -> Path | None:
    """Download one product into ``cache_dir`` (mirroring host/path) with retries.

    A local path is returned as is (with provenance written).  ``max_bytes``
    aborts an oversize transfer and removes the partial file.  A cached file
    whose provenance matches is not refetched.  Returns ``None`` on failure and
    says why on stdout; never raises.
    """
    cache_dir = Path(cache_dir)
    src = str(uri)
    if "://" not in src:
        p = Path(src)
        if p.exists():
            _write_provenance(p, src, None, None)
            return p
        print(f"[roman] fetch: local path {src} does not exist")
        return None
    url = to_https(src)
    dest = cache_path_for(src, cache_dir)
    prov = _provenance_path(dest)
    if dest.exists() and prov.exists():
        try:
            old = json.loads(prov.read_text())
            if int(old.get("bytes") or -1) == dest.stat().st_size:
                return dest
        except Exception:  # noqa: BLE001
            pass
    session = session if session is not None else default_session()
    if session is None:
        print("[roman] fetch: requests unavailable")
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last = None
    for i in range(max(1, int(retries))):
        rec = {}
        r = _http(session, "get", url, timeout, rec, stream=True)
        if r is None or rec.get("http_status") != 200:
            last = rec.get("error") or f"http {rec.get('http_status')}"
            if rec.get("http_status") in (401, 403, 404):
                break
            time.sleep(pause * (i + 1))
            continue
        n = 0
        too_big = False
        try:
            with open(tmp, "wb") as fh:
                it = getattr(r, "iter_content", None)
                chunks = it(chunk_size=1 << 20) if callable(it) else [_body(r, 1 << 62)]
                for ch in chunks:
                    if not ch:
                        continue
                    n += len(ch)
                    if max_bytes is not None and n > int(max_bytes):
                        too_big = True
                        break
                    fh.write(ch)
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
            tmp.unlink(missing_ok=True)
            time.sleep(pause * (i + 1))
            continue
        finally:
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
        if too_big:
            tmp.unlink(missing_ok=True)
            print(f"[roman] fetch: {src} exceeds max_bytes={max_bytes}; aborted")
            return None
        tmp.replace(dest)
        _write_provenance(dest, src, rec.get("http_status"), _headers(r).get("content-type"))
        return dest
    print(f"[roman] fetch: {src} failed: {last}")
    return None


def _provenance_path(path: Path) -> Path:
    return path.with_name(path.name + ".provenance.json")


def _write_provenance(path: Path, uri: str, http_status, content_type) -> None:
    try:
        rec = {"uri": uri, "url": to_https(uri), "bytes": path.stat().st_size,
               "utc": utc_now(), "sha256_first_mb": _sha256_first_mb(path),
               "http_status": http_status, "content_type": content_type}
        _provenance_path(path).write_text(json.dumps(rec, indent=1))
    except Exception as exc:  # noqa: BLE001
        print(f"[roman] provenance for {path} not written: {exc!r}")


__all__ = ["probe", "inventory", "plan_shards", "fetch_to_cache", "classify_product",
           "survey_from_path", "band_from_path", "derive_data_state", "parse_s3_listing",
           "parse_tap_json", "list_s3_deep", "extract_links", "s3_listing_url", "to_https",
           "cache_path_for", "probe_packages", "default_session"]
