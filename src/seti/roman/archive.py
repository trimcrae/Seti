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
                        MAST CAOM is asked with ``SELECT TOP 1`` first and the
                        ``COUNT(*)`` form second, and every error body (first
                        3000 characters, plus the VOTable ``<INFO>`` text) is
                        kept; ``kind: index`` simulation pages are crawled one
                        level deep (:func:`crawl_index`).
:func:`inventory`       turns the listings into :class:`RomanProduct` rows
                        (level / kind / survey / size / simulated) from the
                        filename patterns Roman's pipeline is documented to use
                        and the OpenUniverse 2024 layouts
                        (:func:`openuniverse_match`), walks each S3 bucket's
                        directory TREE with delimiter listings
                        (:func:`list_s3_tree`; per-directory sample keys,
                        truncation and subdirectories recorded), and writes a
                        shard plan.  A simulated product is stamped
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
from collections import deque
from pathlib import Path
from urllib.parse import quote, unquote, urlencode, urljoin, urlparse

from .schema import RomanProduct, json_safe, utc_now

USER_AGENT = "seti-roman/0.1 (technosignature intake; github.com/trimcrae/Seti)"
DATA_EXTENSIONS = (".fits", ".asdf", ".parquet", ".csv", ".tar", ".gz", ".h5", ".txt")
PROBE_PACKAGES = ("asdf", "roman_datamodels", "gwcs", "fsspec", "s3fs")
MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_CRAWL_PAGE_BYTES = 1024 * 1024
MAX_ERROR_BODY = 3000
MAX_ROOT_LINKS = 50
MAX_SIM_LINKS = 200
S3_PAGE_KEYS = 1000
CRAWL_MAX_PAGES = 15
CRAWL_LINK_PATTERN = r"sim|challenge|microlens|openuniverse|data|roman"
# The tokens that make a crawl link worth its budget, strongest first: a link
# matching only ``roman`` on roman.ipac.caltech.edu is every link on the site.
_CRAWL_PRIORITY = ("openuniverse", "microlens", "challenge", "sim", "data", "roman")

# OpenUniverse 2024 (Roman-Rubin) band tokens -> WFI filter names; the config key
# ``archive.openuniverse_band_map`` overrides this at inventory time.
OPENUNIVERSE_BAND_MAP = {"R062": "F062", "Z087": "F087", "Y106": "F106", "J129": "F129",
                         "H158": "F158", "F184": "F184", "K213": "F213", "W146": "F146"}

_HREF_RE = re.compile(r"""href\s*=\s*["']?([^"' >]+)""", re.I)
_ANCHOR_RE = re.compile(r"""<a\b[^>]*?href\s*=\s*["']?([^"' >]+)[^>]*>(.*?)</a>""", re.I | re.S)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_FILTER_RE = re.compile(r"(?<![a-z0-9])(f\d{3})(?![0-9])")
# "sim" at the start of a token (simple_model, sims/) or at its end (RomanSim, wfisim);
# not inside a word (prism).
_SIM_TOKEN_RE = re.compile(r"(?:^|[^a-z])sim|sim(?:$|[^a-z])")
_OU_TOKEN_RE = re.compile(r"^[A-Z]\d{3}$")
# Roman_TDS_simple_model_F184_10307_17.fits.gz / Roman_WAS_truth_H158_1062_3.fits.gz /
# Roman_TDS_index_F184_10307_17.txt
_OU_IMAGE_RE = re.compile(r"^roman_(tds|was)_(simple_model|truth|index)_([a-z]\d{3})_(\d+)_(\d+)"
                          r"\.(fits(?:\.gz)?|txt)$", re.I)
_OU_CAT_RE = re.compile(r"^(pointsource|galaxy|snana)(?:_(flux|sed))?_(\d+)\.(parquet|hdf5|h5)$",
                        re.I)
_OU_SNANA_RE = re.compile(r"^(.*)_(head|phot)\.fits(\.gz)?$", re.I)
_OU_OBSEQ_RE = re.compile(r"_obseq_.*\.fits(\.gz)?$", re.I)
# RomanWAS/images/coadd/<BAND>/prod_F_00_03_map.fits.gz (full) and
# RomanWAS/images/coadds/<BAND>/Row12/prod_F_12_07_map.fits.gz (preview)
_OU_COADD_RE = re.compile(r"^prod_([a-z0-9]+)_(\d+)_(\d+)_map\.fits(\.gz)?$", re.I)
_DOC_EXTENSIONS = (".dump", ".list", ".readme", ".yaml", ".yml", ".pdf", ".png", ".md", ".html")


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
    """ListObjectsV2 URL for the anonymous bucket endpoint.

    The prefix is percent-encoded with ``quote(prefix, safe="/")`` so a literal
    ``+`` travels as ``%2B``: S3 decodes a bare ``+`` in the query string as a
    space, and ``openuniverse2024/roman/full/ROMAN+LSST_LARGE_SNIa-normal/``
    then lists as empty (verified 2026-09-09).
    """
    parts = ["list-type=2", "prefix=" + quote(str(prefix or ""), safe="/"),
             f"max-keys={int(max_keys)}"]
    if delimiter:
        parts.append("delimiter=" + quote(delimiter, safe=""))
    if token:
        parts.append("continuation-token=" + quote(str(token), safe=""))
    return f"https://{bucket}.s3.amazonaws.com/?" + "&".join(parts)


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


def votable_info(text: str) -> list[str]:
    """The ``<INFO>`` elements of a VOTable (error) document as ``name=value: text`` strings.

    TAP services put the reason for a 400 here (``QUERY_STATUS=ERROR``); the
    namespace is ignored and a regex fallback covers a truncated document.
    """
    out: list[str] = []
    src = str(text or "")
    try:
        root = ET.fromstring(src)
        for el in root.iter():
            if el.tag.split("}")[-1] != "INFO":
                continue
            name, value = el.get("name", ""), el.get("value", "")
            body = " ".join((el.text or "").split())
            out.append(f"{name}={value}: {body}".strip(": ").strip())
    except Exception:  # noqa: BLE001
        for m in re.finditer(r"<INFO\b([^>]*)>(.*?)</INFO>", src, re.I | re.S):
            attrs = dict(re.findall(r"""(\w+)\s*=\s*["']([^"']*)["']""", m.group(1)))
            body = " ".join(_TAG_RE.sub(" ", m.group(2)).split())
            out.append(f"{attrs.get('name', '')}={attrs.get('value', '')}: {body}"
                       .strip(": ").strip())
    return [s[:500] for s in out if s]


def mast_query_forms(collection: str) -> list[tuple[str, str]]:
    """The ADQL forms tried against MAST CAOM for one collection, in order.

    ``TOP 1`` first: the 2026-09-09 run got HTTP 400 for the ``COUNT(*) AS n``
    form on every collection and the reason was cut off, so the probe now tries
    the simplest possible statement, then the count, and keeps the whole error.
    """
    c = str(collection).replace("'", "''")
    return [("top1", f"SELECT TOP 1 obs_collection FROM dbo.CaomObservation "
                     f"WHERE obs_collection = '{c}'"),
            ("count", f"SELECT COUNT(*) AS n FROM dbo.CaomObservation WHERE obs_collection='{c}'")]


def _mast_attempt(session, tap: str, form: str, query: str, timeout: float) -> tuple[dict, list]:
    """One TAP request; returns (attempt record, parsed rows or None)."""
    att = {"form": form, "url": _tap_url(tap, query), "http_status": None, "latency_s": None,
           "error": None, "error_body": None, "votable_info": []}
    r = _http(session, "get", att["url"], timeout, att)
    if r is None:
        return att, None
    text = _text(r)
    if att["http_status"] != 200:
        att["error_body"] = text[:MAX_ERROR_BODY]
        att["votable_info"] = votable_info(text)
        att["error"] = (att["votable_info"][0] if att["votable_info"]
                        else text[:300]) or f"http {att['http_status']}"
        return att, None
    try:
        return att, parse_tap_json(text)
    except Exception as exc:  # noqa: BLE001
        att["error"] = f"parse: {exc!r}"
        return att, None


def _probe_mast(conf: dict, session, timeout: float) -> list[dict]:
    mast = conf["archive"]["mast"]
    recs = []
    for coll in mast.get("collections") or []:
        forms = mast_query_forms(str(coll))
        rec = _endpoint(f"mast:{coll}", "mast_caom", _tap_url(mast["tap"], forms[0][1]))
        rec.update(collection=str(coll), count=None, present=None, query_form=None,
                   attempts=[], simulated=_is_simulated_name(coll))
        for form, query in forms:
            att, rows = _mast_attempt(session, mast["tap"], form, query, timeout)
            rec["attempts"].append(att)
            rec["url"], rec["http_status"], rec["latency_s"] = att["url"], att["http_status"], \
                att["latency_s"]
            if rows is None:
                rec["error"] = att["error"]
                rec["status"] = "http_error" if att["http_status"] is not None else "failed"
                continue
            rec["status"], rec["error"], rec["query_form"] = "ok", None, form
            if form == "top1":
                vals = [_first_value(row) for row in rows]
                rec["present"] = any(v is not None and str(v).strip() != "" for v in vals)
                rec["count"] = None if rec["present"] else 0
                if rec["present"]:
                    # the number is worth one more request, but its failure is not a failure
                    att2, rows2 = _mast_attempt(session, mast["tap"], "count", forms[1][1], timeout)
                    rec["attempts"].append(att2)
                    if rows2:
                        v = _first_value(rows2[0])
                        try:
                            rec["count"] = int(float(v)) if v is not None else None
                        except Exception:  # noqa: BLE001
                            rec["count"] = None
                        if rec["count"] is not None:      # a parsed count is authoritative
                            rec["present"] = rec["count"] > 0
            else:
                v = _first_value(rows[0]) if rows else 0
                try:
                    rec["count"] = int(float(v)) if v is not None else 0
                except Exception:  # noqa: BLE001
                    rec["count"] = None
                rec["present"] = bool(rec["count"])
            break
        if rec["status"] != "ok" and rec["attempts"]:
            infos = [i for a in rec["attempts"] for i in (a.get("votable_info") or [])]
            rec["votable_info"] = infos
            rec["error_body"] = next((a.get("error_body") for a in rec["attempts"]
                                      if a.get("error_body")), None)
        recs.append(rec)
    return recs


def extract_anchors(html: str, base: str) -> list[tuple[str, str]]:
    """``(absolute url, anchor text)`` for every ``<a href>`` in ``html``, in order,
    de-duplicated by URL; fragments, mailto and javascript links dropped."""
    out, seen = [], set()
    for m in _ANCHOR_RE.finditer(html or ""):
        href = m.group(1).strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        absu = urljoin(base, href).split("#", 1)[0]
        if not absu.startswith(("http://", "https://")) or absu in seen:
            continue
        seen.add(absu)
        text = " ".join(_TAG_RE.sub(" ", m.group(2)).split())
        out.append((absu, text[:200]))
    return out


def page_title(html: str) -> str | None:
    m = _TITLE_RE.search(html or "")
    if not m:
        return None
    return " ".join(_TAG_RE.sub(" ", m.group(1)).split())[:200] or None


def _crawl_priority(url: str, text: str) -> tuple:
    low = (url + " " + text).lower()
    hits = [i for i, tok in enumerate(_CRAWL_PRIORITY) if tok in low]
    strongest = hits[0] if hits else len(_CRAWL_PRIORITY)
    return (strongest, -len(hits))


def crawl_index(html: str, base_url: str, session, timeout: float,
                max_pages: int = CRAWL_MAX_PAGES, pattern: str = CRAWL_LINK_PATTERN) -> dict:
    """Follow the links of an index page one level deep and record what is there.

    Every href is extracted (relative URLs resolved); those whose URL or anchor
    text matches ``pattern`` (case-insensitive) are kept, ordered so the links
    naming a simulation / data challenge come before the ones matching only
    ``roman`` (which, on roman.ipac.caltech.edu, is all of them), and the first
    ``max_pages`` are fetched: HEAD (following redirects) then GET capped at
    :data:`MAX_CRAWL_PAGE_BYTES`.  A link that is itself a data file is
    recorded as such and not downloaded.  Per page: url, anchor text, HTTP
    status, content type, title, and the data-file hrefs on it.
    """
    rx = re.compile(pattern, re.I)
    anchors = extract_anchors(html, base_url)
    base_norm = base_url.rstrip("/")
    matched = [(u, t) for u, t in anchors
               if u.rstrip("/") != base_norm and (rx.search(u) or rx.search(t))]
    matched.sort(key=lambda ut: _crawl_priority(*ut))
    out = {"n_links_total": len(anchors), "n_links_matched": len(matched),
           "n_pages_fetched": 0, "capped": len(matched) > int(max_pages),
           "links_matched": [u for u, _ in matched[:MAX_SIM_LINKS]],
           "pages": [], "data_links": []}
    for url, text in matched[:int(max_pages)]:
        page = {"url": url, "anchor_text": text, "http_status": None, "head_status": None,
                "content_type": None, "title": None, "is_data_file": False, "data_links": [],
                "error": None}
        if urlparse(url).path.lower().endswith(DATA_EXTENSIONS):
            page["is_data_file"] = True
            rec = {}
            _http(session, "head", url, timeout, rec, allow_redirects=True)
            page.update(head_status=rec.get("http_status"), http_status=rec.get("http_status"),
                        error=rec.get("error"))
            page["data_links"] = [url]
            out["pages"].append(page)
            out["data_links"].append(url)
            continue
        rec = {}
        _http(session, "head", url, timeout, rec, allow_redirects=True)
        page["head_status"] = rec.get("http_status")
        rec = {}
        r = _http(session, "get", url, timeout, rec, stream=True)
        page["http_status"], page["error"] = rec.get("http_status"), rec.get("error")
        out["n_pages_fetched"] += 1
        if r is None:
            out["pages"].append(page)
            continue
        page["content_type"] = _headers(r).get("content-type")
        body = _body(r, MAX_CRAWL_PAGE_BYTES).decode("utf-8", "replace")
        page["title"] = page_title(body)
        if page["http_status"] == 200:
            page["data_links"] = extract_links(body, url, MAX_SIM_LINKS, data_only=True)
            for d in page["data_links"]:
                if d not in out["data_links"]:
                    out["data_links"].append(d)
        out["pages"].append(page)
    return out


def _probe_simulation_page(entry: dict, session, timeout: float, arch: dict | None = None) -> dict:
    url = str(entry["url"])
    arch = arch or {}
    rec = _endpoint(f"sim:{entry.get('name', url)}", "simulation_page", url)
    rec.update(sim_kind=str(entry.get("kind") or ""), content_type=None, bytes=None,
               title=None, data_links=[], crawl=None)
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
    html = body.decode("utf-8", "replace")
    rec["title"] = page_title(html)
    if rec["http_status"] == 200:
        rec["data_links"] = extract_links(html, url, MAX_SIM_LINKS, data_only=True)
        rec["status"] = "ok"
        if rec["sim_kind"] == "index":
            rec["crawl"] = crawl_index(
                html, url, session, timeout,
                max_pages=int(arch.get("crawl_max_pages", CRAWL_MAX_PAGES)),
                pattern=str(arch.get("crawl_link_pattern") or CRAWL_LINK_PATTERN))
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
        elif kind == "mast_caom" and rec.get("status") == "ok" \
                and ((rec.get("count") or 0) > 0 or rec.get("present")):
            n = rec["count"] if rec.get("count") is not None else "TOP 1 row present, count unknown"
            if rec.get("simulated"):
                sim.append(f"MAST collection {rec.get('collection')}: {n} obs (simulation)")
            else:
                mission.append(f"MAST collection {rec.get('collection')}: {n} observations")
        elif kind == "simulation_page" and rec.get("status") == "ok" \
                and rec.get("sim_kind") in ("gbtds_like", "hltds_like"):
            sim.append(f"{name}: simulation page answers ({len(rec.get('data_links') or [])} "
                       f"data link(s))")
        elif kind == "simulation_page" and rec.get("status") == "ok" \
                and rec.get("sim_kind") == "index":
            crawl = rec.get("crawl") or {}
            n_links = len(rec.get("data_links") or []) + len(crawl.get("data_links") or [])
            if n_links:
                sim.append(f"{name}: index page and its crawl expose {n_links} data link(s)")
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
                lambda e=entry: _probe_simulation_page(e, session, timeout, arch))
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
        if "hltds" in t or "roman+lsst" in t:     # the OpenUniverse SNANA (HLTDS-like) sim
            return "HLTDS"
        if "hlwas" in t:
            return "HLWAS"
    for t in toks:
        if t == "tds" or t.endswith("tds"):
            return "HLTDS"
        if t == "was" or t.endswith("was") or t == "wide":
            return "HLWAS"
    return ""


def band_from_path(path: str, band_map: dict | None = None) -> str | None:
    """WFI filter / disperser named in the path: the ``Fnnn`` names first, then
    the OpenUniverse band tokens (``R062``/``Z087``/``Y106``/``J129``/``H158``/
    ``K213``/``W146``) through ``band_map``."""
    low = path.lower()
    for m in _FILTER_RE.finditer(low):
        if m.group(1) in _KNOWN_FILTERS:
            return m.group(1).upper()
    bmap = band_map if band_map is not None else OPENUNIVERSE_BAND_MAP
    for t in re.split(r"[/_\-.\s]+", path):
        if _OU_TOKEN_RE.match(t) and t.upper() in bmap:
            return str(bmap[t.upper()])
    if re.search(r"(?<![a-z0-9])g150(?![0-9])", low):
        return "G150"
    if re.search(r"(?<![a-z0-9])p127(?![0-9])", low):
        return "P127"
    return None


def _ou_root(dirname: str, marker: str) -> str | None:
    """``dirname`` cut just after the OpenUniverse survey root (``.../RomanTDS/``),
    located as the last occurrence of ``marker`` (``/images/simple_model/`` ...)."""
    i = dirname.lower().rfind(marker)
    return dirname[:i + 1] if i >= 0 else None


def _swap_token(base: str, old: str, new: str) -> str:
    """Replace ``old`` by ``new`` in ``base`` once, matching and keeping its case."""
    m = re.search(re.escape(old), base, re.I)
    if not m:
        return base
    tok = m.group(0)
    rep = new.upper() if tok.isupper() else new.lower() if tok.islower() else new
    return base[:m.start()] + rep + base[m.end():]


def openuniverse_match(uri: str, band_map: dict | None = None) -> dict | None:
    """Level / kind / band / meta for an OpenUniverse 2024 key, or ``None``.

    Patterns (verified against ``nasa-irsa-simulations`` on 2026-09-09):

    * ``<run>_HEAD.FITS[.gz]`` / ``<run>_PHOT.FITS[.gz]``  SNANA light-curve pairs
      (``ROMAN+LSST_LARGE_SNIa-normal/``): the HEAD is the ingestible
      ``lightcurve`` and names its ``phot_sibling``; the PHOT is
      ``lightcurve_phot`` and is never ingested on its own.
    * ``Roman_{TDS,WAS}_simple_model_<BAND>_<pointing>_<sca>.fits.gz``  the
      simulated L2 image (``cal``) with its ``truth_index`` (per-SCA truth
      table under ``<root>/truth/<BAND>/<pointing>/``) and ``truth_image``.
    * ``Roman_*_truth_*.fits.gz``  noiseless truth image; ``Roman_*_index_*.txt``
      truth index (``catalog``, format ``truth_index``).
    * ``pointsource_<pix>``/``galaxy_<pix>``/``snana_<pix>`` (``_flux``/``_sed``
      variants) the Roman-Rubin truth catalogues per HEALPix pixel.
    * ``prod_F_<row>_<col>_map.fits.gz`` under ``images/coadd[s]/<BAND>/``  the
      RomanWAS coadd tiles (L3 ``coadd``, format ``openuniverse_coadd``).
    * ``*_obseq_*.fits``  observing sequence; ``.DUMP/.LIST/.README/.yaml/.pdf/.png`` docs.

    Sibling keys are returned in the same form as ``uri`` (``s3://`` stays ``s3://``).
    """
    bmap = band_map if band_map is not None else OPENUNIVERSE_BAND_MAP
    base = uri.rsplit("/", 1)[-1]
    dirname = uri[:len(uri) - len(base)]
    m = _OU_IMAGE_RE.match(base)
    if m:
        sv, what, band_tok, pointing, sca, ext = m.groups()
        sv, what, band_tok = sv.upper(), what.lower(), band_tok.upper()
        band = bmap.get(band_tok) or band_from_path(base, bmap)
        common = {"band_token": band_tok, "pointing": int(pointing), "sca": int(sca),
                  "openuniverse_survey": f"Roman{sv}"}
        marker = {"simple_model": "/images/simple_model/", "truth": "/images/truth/",
                  "index": "/truth/"}[what]
        root = _ou_root(dirname, marker)
        img_ext = ext if what != "index" else "fits.gz"

        def under(sub, name):
            return f"{root}{sub}/{band_tok}/{pointing}/{name}" if root else None
        index_key = under("truth", f"Roman_{sv}_index_{band_tok}_{pointing}_{sca}.txt")
        image_key = under("images/simple_model",
                          f"Roman_{sv}_simple_model_{band_tok}_{pointing}_{sca}.{img_ext}")
        truth_key = under("images/truth", f"Roman_{sv}_truth_{band_tok}_{pointing}_{sca}.{img_ext}")
        if what == "simple_model":
            return {"level": "L2", "kind": "cal", "band": band,
                    "meta": {"format": "openuniverse_image", "truth_index": index_key,
                             "truth_image": truth_key, **common}}
        if what == "truth":
            return {"level": "L2", "kind": "truth_image", "band": band,
                    "meta": {"format": "openuniverse_truth_image", "image": image_key,
                             "truth_index": index_key, **common}}
        return {"level": "catalog", "kind": "catalog", "band": band,
                "meta": {"format": "truth_index", "image": image_key, "truth_image": truth_key,
                         **common}}
    m = _OU_SNANA_RE.match(base)
    if m:
        which = m.group(2).lower()
        if which == "head":
            return {"level": "L4", "kind": "lightcurve", "band": None,
                    "meta": {"format": "snana_head",
                             "phot_sibling": dirname + _swap_token(base, "_HEAD", "_PHOT")}}
        return {"level": "L4", "kind": "lightcurve_phot", "band": None,
                "meta": {"format": "snana_phot",
                         "head_sibling": dirname + _swap_token(base, "_PHOT", "_HEAD")}}
    m = _OU_CAT_RE.match(base)
    if m:
        what, variant, pix, _ext = m.groups()
        what = what.lower()
        fmt = {"pointsource": "pointsource", "galaxy": "galaxy", "snana": "snana_truth"}[what]
        return {"level": "catalog", "kind": "catalog", "band": None,
                "meta": {"format": fmt, "variant": (variant or "main").lower(),
                         "healpix": int(pix)}}
    m = _OU_COADD_RE.match(base)
    if m:
        sv = "RomanWAS" if "romanwas" in uri.lower() else \
            "RomanTDS" if "romantds" in uri.lower() else None
        return {"level": "L3", "kind": "coadd", "band": band_from_path(dirname, bmap),
                "meta": {"format": "openuniverse_coadd", "tile_row": int(m.group(2)),
                         "tile_col": int(m.group(3)), "openuniverse_survey": sv}}
    if _OU_OBSEQ_RE.search(base):
        return {"level": "sim", "kind": "obseq", "band": None,
                "meta": {"format": "obseq",
                         "variant": "radec" if "_radec" in base.lower() else "pointings"}}
    if base.lower().endswith(_DOC_EXTENSIONS):
        return {"level": "sim", "kind": "doc", "band": None,
                "meta": {"format": "doc", "extension": base.lower().rsplit(".", 1)[-1]}}
    return None


def classify_product(uri: str, origin: str = "", simulated_hint: bool | None = None,
                     size_bytes: int | None = None, meta: dict | None = None,
                     band_map: dict | None = None) -> RomanProduct:
    """One key / href / table name to a :class:`RomanProduct` by filename pattern.

    Level and kind come from the documented Roman suffixes (``_uncal`` L1,
    ``_cal`` L2, ``coadd``/``mosaic``/``_i2d`` L3, light-curve and spectral
    tokens L4, ``catalog``/``cat_``, ``cgi``) and from the OpenUniverse 2024
    layouts (:func:`openuniverse_match`); anything else is ``unknown`` and
    stays in the inventory so the runner can look at what it did not recognise.
    ``simulated`` is True when the source is a simulation or the path says so.
    ``meta`` is merged with what the pattern derives (``format``, siblings).
    """
    path = urlparse(uri).path if "://" in uri else uri
    low = path.lower()
    base = low.rsplit("/", 1)[-1]
    is_table = base.endswith((".parquet", ".csv", ".ecsv", ".fits", ".asdf", ".h5", ".txt"))
    spectral = any(t in low for t in ("x1d", "spec", "grism", "prism"))
    instrument = "WFI"
    level, kind = "unknown", "unknown"
    band = band_from_path(path, band_map)
    out_meta = dict(meta or {})
    is_cgi = "cgi" in _tokens(low) or "/cgi" in low or base.startswith("cgi")
    ou = None if is_cgi else openuniverse_match(uri, band_map)
    if ou is not None:
        level, kind = ou["level"], ou["kind"]
        band = ou.get("band") or band
        out_meta.update(ou["meta"])
        if kind in ("doc", "obseq") and not (bool(simulated_hint) or _is_simulated_name(low)):
            level = "unknown"
    elif is_cgi:
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
                        band=band, size_bytes=size_bytes, simulated=simulated, meta=out_meta)


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


def _is_numeric_dir(prefix: str) -> bool:
    """A subdirectory whose name is all digits (a pointing, a HEALPix pixel, a visit)."""
    return prefix.rstrip("/").rsplit("/", 1)[-1].isdigit()


def _list_s3_dir(bucket: str, prefix: str, session, timeout: float, max_keys: int,
                 max_pages: int, page_keys: int, max_subdirs: int | None = None) -> dict:
    """One directory (delimiter ``/``): up to ``max_keys`` sample keys over at
    most ``max_pages`` continuation pages, the subdirectories seen, and whether
    more was left unlisted.  A page that carried only subdirectories, once
    ``max_subdirs`` of them are known, ends the listing (``subdirs_truncated``):
    the ``truth/<BAND>/`` directories of OpenUniverse hold 2 000-5 000 numbered
    pointing directories and paging through them tells nothing new.  An HTTP /
    XML failure is recorded, never raised."""
    d = {"path": prefix, "n_keys_listed": 0, "n_subdirs": 0, "n_subdirs_numeric": 0,
         "truncated": False, "subdirs_truncated": False, "total_bytes_listed": 0, "keys": [],
         "subdirs": [], "n_requests": 0, "http_status": None, "error": None, "error_code": None}
    token = None
    while d["n_requests"] < max(1, int(max_pages)):
        url = s3_listing_url(bucket, prefix, "/", min(int(page_keys), 1000), token)
        rec: dict = {}
        r = _http(session, "get", url, timeout, rec)
        d["n_requests"] += 1
        d["http_status"] = rec.get("http_status")
        if r is None:
            d["error"] = rec.get("error") or "no response"
            break
        body = _text(r)
        try:
            lst = parse_s3_listing(body) if body.strip() else {"error": "empty body"}
        except Exception as exc:  # noqa: BLE001
            lst = {"error": f"xml: {exc!r}"}
        if rec.get("http_status") != 200 or lst.get("error"):
            err = lst.get("error") or body[:200] or f"http {rec.get('http_status')}"
            d["error"] = str(err)[:300]
            d["error_code"] = str(err).split(":", 1)[0].strip() if lst.get("error") else None
            break
        for p in lst["prefixes"]:
            if p not in d["subdirs"]:
                d["subdirs"].append(p)
        n_page_keys = 0
        for c in lst["contents"]:
            key = str(c.get("key") or "")
            if not key or key == prefix or key.endswith("/"):
                continue
            d["keys"].append({"key": key, "size": c.get("size")})
            n_page_keys += 1
        d["truncated"] = bool(lst["is_truncated"])
        if len(d["keys"]) >= int(max_keys) or not lst["is_truncated"] or not lst["next_token"]:
            break
        if n_page_keys == 0 and max_subdirs is not None and len(d["subdirs"]) >= int(max_subdirs):
            d["subdirs_truncated"] = True        # more subdirectories exist than were listed
            break
        token = lst["next_token"]
    if len(d["keys"]) > int(max_keys):
        d["keys"] = d["keys"][:int(max_keys)]
        d["truncated"] = True                    # the sample stopped short of what the page held
    d["n_keys_listed"] = len(d["keys"])
    d["total_bytes_listed"] = int(sum(int(k.get("size") or 0) for k in d["keys"]))
    d["n_subdirs"] = len(d["subdirs"])
    d["n_subdirs_numeric"] = sum(1 for p in d["subdirs"] if _is_numeric_dir(p))
    return d


def list_s3_tree(bucket: str, root_prefix: str, session, max_depth: int = 6,
                 max_keys_per_dir: int = 200, max_dirs: int = 400, timeout_s: float | None = None,
                 max_subdirs_per_dir: int = 12, max_numeric_subdirs_per_dir: int = 2,
                 max_pages_per_dir: int = 5, page_keys: int = S3_PAGE_KEYS) -> dict:
    """Delimiter-based, breadth-first listing of the directory TREE under ``root_prefix``.

    Each directory is recorded with ``n_keys_listed`` (sample keys with sizes,
    up to ``max_keys_per_dir``, continuation tokens honoured up to
    ``max_pages_per_dir`` pages), ``truncated`` (S3's ``IsTruncated`` on the
    last page read: more entries exist), ``total_bytes_listed`` and its
    ``subdirs``.  The walk descends into the first ``max_subdirs_per_dir``
    *named* subdirectories of every directory (bands, surveys, product types
    are all wanted) but only the first ``max_numeric_subdirs_per_dir``
    *numbered* ones (pointings, HEALPix pixels: instances of one layout, of
    which OpenUniverse has thousands per band), breadth-first so the budget is
    spread across bands, at most ``max_depth`` levels below the root and
    ``max_dirs`` directories in all.  Prefixes are percent-encoded with
    ``quote(prefix, safe="/")`` (a literal ``+`` becomes ``%2B``).  A 403 / 404
    / ``NoSuchBucket`` answer is recorded on the root (``exists``/``error``)
    and the walk stops; nothing raises.
    """
    timeout = float(timeout_s if timeout_s is not None else 60.0)
    root = str(root_prefix or "")
    if root and not root.endswith("/"):
        root += "/"
    out = {"bucket": bucket, "root_prefix": root, "dirs": [], "n_dirs_listed": 0,
           "n_dirs_seen": 0, "n_requests": 0, "n_keys_listed": 0, "total_bytes_listed": 0,
           "n_dirs_truncated": 0, "max_depth_reached": 0, "dirs_capped": False,
           "depth_capped": False, "exists": None, "http_status": None, "error": None,
           "limits": {"max_depth": int(max_depth), "max_keys_per_dir": int(max_keys_per_dir),
                      "max_dirs": int(max_dirs), "max_subdirs_per_dir": int(max_subdirs_per_dir),
                      "max_numeric_subdirs_per_dir": int(max_numeric_subdirs_per_dir),
                      "max_pages_per_dir": int(max_pages_per_dir)}}
    queue: deque = deque([(root, 0)])
    seen = {root}
    while queue:
        if out["n_dirs_listed"] >= int(max_dirs):
            out["dirs_capped"] = True
            break
        prefix, depth = queue.popleft()
        d = _list_s3_dir(bucket, prefix, session, timeout, max_keys_per_dir, max_pages_per_dir,
                         page_keys, max_subdirs=max(int(max_subdirs_per_dir),
                                                    int(max_numeric_subdirs_per_dir)))
        d["depth"] = depth
        out["n_requests"] += d["n_requests"]
        out["dirs"].append(d)
        if prefix == root:
            out["http_status"] = d["http_status"]
            if d["error"]:
                out["error"] = d["error"]
                code = (d.get("error_code") or "").lower()
                out["exists"] = False if (d["http_status"] == 404 or "nosuchbucket" in code) \
                    else None
                break
            out["exists"] = True
        if d["error"]:
            continue
        out["n_dirs_listed"] += 1
        out["n_keys_listed"] += d["n_keys_listed"]
        out["total_bytes_listed"] += d["total_bytes_listed"]
        out["n_dirs_truncated"] += int(bool(d["truncated"]))
        out["max_depth_reached"] = max(out["max_depth_reached"], depth)
        if depth >= int(max_depth):
            if d["subdirs"]:
                out["depth_capped"] = True
            d["subdirs_descended"] = 0
            continue
        fresh = [s for s in d["subdirs"] if s not in seen]
        named = [s for s in fresh if not _is_numeric_dir(s)][:max(0, int(max_subdirs_per_dir))]
        numeric = [s for s in fresh if _is_numeric_dir(s)][:max(0, int(max_numeric_subdirs_per_dir))]
        descend = [s for s in fresh if s in set(named) | set(numeric)]
        d["subdirs_descended"] = len(descend)
        for sub in descend:
            seen.add(sub)
            queue.append((sub, depth + 1))
    out["n_dirs_seen"] = len(seen)
    return out


def tree_summary(tree: dict) -> list[dict]:
    """The per-directory summary of a :func:`list_s3_tree` result (no key lists)."""
    return [{"path": d.get("path"), "depth": d.get("depth"), "n_keys_listed": d.get("n_keys_listed"),
             "n_subdirs": d.get("n_subdirs"), "n_subdirs_numeric": d.get("n_subdirs_numeric"),
             "subdirs_descended": d.get("subdirs_descended"),
             "subdirs_truncated": d.get("subdirs_truncated"), "truncated": d.get("truncated"),
             "total_bytes_listed": d.get("total_bytes_listed"), "error": d.get("error")}
            for d in (tree or {}).get("dirs") or []]


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
    """Every product the probe (and a tree walk of each S3 bucket) can see, classified and sharded.

    Sources: S3 -- the probe's root contents, then :func:`list_s3_tree` over
    each bucket that answered (``archive.s3_tree`` limits; the products are
    the sampled keys of every directory, at most ``max_listing`` of them) --
    the simulation-page data links (including those found by the index
    crawl), the IRSA data-root links, and IRSA tables (as table products).
    MAST counts are carried as counts, not products.  A bucket that did not
    answer (the not-yet-existing mission bucket) contributes no products but a
    recorded status.  ``tree`` is the per-directory summary (path, keys
    listed, truncation, subdirectory count); ``counts_by_format`` counts the
    ``meta.format`` the classifier derived.  ``verdict`` is ``NO_DATA_REACHED``
    when nothing at all was listed; a simulation is never a sky product.
    """
    arch = conf.get("archive") or {}
    timeout = float(timeout_s if timeout_s is not None else arch.get("timeout_s", 60))
    tree_conf = dict(arch.get("s3_tree") or {})
    band_map = dict(arch.get("openuniverse_band_map") or OPENUNIVERSE_BAND_MAP)
    endpoints = (probe_record or {}).get("endpoints") or {}
    products: dict[str, RomanProduct] = {}
    listings: dict[str, dict] = {}
    mast_counts: dict[str, int | None] = {}
    tree_rows: list[dict] = []

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
            listing = {"bucket": bucket, "prefix": prefix, "s3_kind": rec.get("s3_kind"),
                       "status": rec.get("status"), "http_status": rec.get("http_status"),
                       "exists": rec.get("exists"), "error": rec.get("error"),
                       "n_keys": 0, "n_keys_capped": False, "tree": None}
            if rec.get("status") == "ok" and deep and session is not None and bucket:
                tree = list_s3_tree(bucket, prefix, session, timeout_s=timeout,
                                    **{k: int(v) for k, v in tree_conf.items()
                                       if k in ("max_depth", "max_keys_per_dir", "max_dirs",
                                                "max_subdirs_per_dir",
                                                "max_numeric_subdirs_per_dir",
                                                "max_pages_per_dir")})
                walked = [c for d in tree["dirs"] for c in d["keys"]]
                if walked or not tree.get("error"):
                    contents = walked
                if len(contents) > int(max_listing):
                    contents = contents[:int(max_listing)]
                    listing["n_keys_capped"] = True
                listing["tree"] = {k: v for k, v in tree.items() if k != "dirs"}
                listing["tree"]["n_dirs"] = len(tree["dirs"])
                for d in tree_summary(tree):
                    tree_rows.append({"endpoint": name, "bucket": bucket, **d})
            listing["n_keys"] = len(contents)
            listings[name] = listing
            for c in contents:
                key = str(c.get("key") or "")
                if not key or key.endswith("/"):
                    continue
                add(classify_product(f"s3://{bucket}/{key}", origin, sim_hint, c.get("size"),
                                     {"bucket": bucket, "endpoint": name}, band_map=band_map))
        elif kind == "simulation_page" and rec.get("status") == "ok":
            for href in rec.get("data_links") or []:
                add(classify_product(str(href), "ipac_sim_page", True, None,
                                     {"endpoint": name, "sim_kind": rec.get("sim_kind")},
                                     band_map=band_map))
            for page in (rec.get("crawl") or {}).get("pages") or []:
                for href in page.get("data_links") or []:
                    add(classify_product(str(href), "ipac_sim_page", True, None,
                                         {"endpoint": name, "sim_kind": rec.get("sim_kind"),
                                          "crawled_from": page.get("url")}, band_map=band_map))
        elif kind == "irsa_data_root" and rec.get("status") == "ok":
            for href in rec.get("links") or []:
                if urlparse(str(href)).path.lower().endswith(DATA_EXTENSIONS):
                    add(classify_product(str(href), "irsa_http", None, None, {"endpoint": name},
                                         band_map=band_map))
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
    by_format: dict[str, int] = {}
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        by_level[r["level"]] = by_level.get(r["level"], 0) + 1
        fmt = str((r.get("meta") or {}).get("format") or "none")
        by_format[fmt] = by_format.get(fmt, 0) + 1
    n_sim = sum(1 for r in rows if r["simulated"])
    out = {"written_utc": utc_now(), "data_state": (probe_record or {}).get("data_state"),
           "products": rows, "counts_by_kind": by_kind, "counts_by_level": by_level,
           "counts_by_format": by_format,
           "n_products": len(rows), "n_simulated": n_sim, "n_mission": len(rows) - n_sim,
           "total_bytes": int(sum(int(r["size_bytes"] or 0) for r in rows)),
           "listings": listings, "tree": tree_rows, "mast_counts": mast_counts,
           "shards": plan_shards(rows, n_shards),
           "verdict": "INVENTORIED" if rows else "NO_DATA_REACHED"}
    print(f"[roman] inventory: {len(rows)} products ({n_sim} simulated); kinds={by_kind}; "
          f"formats={by_format}; {len(tree_rows)} director(ies) listed")
    return json_safe(out)


# --------------------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------------------

def to_https(uri: str) -> str:
    """``s3://bucket/key`` -> the anonymous HTTPS form; anything else unchanged.

    The key is percent-encoded with ``/`` kept, so a literal ``+`` (the
    OpenUniverse ``ROMAN+LSST_...`` directory) travels as ``%2B`` -- S3 would
    otherwise read it as a space and answer 404.
    """
    if str(uri).startswith("s3://"):
        rest = str(uri)[5:]
        bucket, _, key = rest.partition("/")
        return f"https://{bucket}.s3.amazonaws.com/{quote(key, safe='/')}"
    return str(uri)


def cache_path_for(uri: str, cache_dir: Path) -> Path:
    """Local mirror path ``<cache_dir>/<host>/<decoded path>`` (a ``%2B`` key lands as ``+``)."""
    u = urlparse(to_https(uri))
    host = u.netloc or "local"
    rel = unquote(u.path).lstrip("/") or "index"
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
           "openuniverse_match", "survey_from_path", "band_from_path", "derive_data_state",
           "parse_s3_listing", "parse_tap_json", "list_s3_tree", "tree_summary", "list_s3_deep",
           "extract_links", "extract_anchors", "page_title", "crawl_index", "votable_info",
           "mast_query_forms", "s3_listing_url", "to_https", "cache_path_for", "probe_packages",
           "default_session", "OPENUNIVERSE_BAND_MAP"]
