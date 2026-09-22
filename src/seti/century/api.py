"""DASCH DR7 REST client --- runner-only, schema-tolerant.

The DR7 web APIs are RESTful JSON endpoints under
``https://api.starglass.cfa.harvard.edu/public/dasch/dr7/``; the ``daschlab``
Python package wraps them.  Three are used here:

``querycat``    sources of a photometric reference catalogue (``apass`` or
                ``atlas``) around a position, with each source's DASCH
                identifiers (``gsc_bin_index``, ``ref_number``);
``queryexps``   every plate exposure covering a position --- the plate density
                of a field, and its pre/post Menzel-gap split;
``lightcurve``  the DASCH photometry of one source: per-plate calibrated
                magnitude, local RMS, **per-plate limiting magnitude at the
                source position**, plate series, exposure time, AFLAGS/BFLAGS.

Nothing here is trusted until the runner has answered.  Every request records
its status, elapsed time and, on a 4xx, the response body --- a FastAPI-style
validation error names the fields the endpoint wanted, which is how the probe
stage corrects a wrong payload shape without a second round trip of guessing.
Payloads are tried as an ordered list of variants; the first 200 wins and the
variant that worked is written into the acquisition log.

The sandbox has no egress; every function here runs on a GitHub Actions runner.
"""

from __future__ import annotations

import io
import json
import time as _time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

API_BASE = "https://api.starglass.cfa.harvard.edu/public"
ENDPOINTS = {
    "querycat": "/dasch/dr7/querycat",
    "queryexps": "/dasch/dr7/queryexps",
    "lightcurve": "/dasch/dr7/lightcurve",
}
DOC_URLS = {
    "web_apis": "https://dasch.cfa.harvard.edu/dr7/web-apis/",
    "lc_columns": "https://dasch.cfa.harvard.edu/dr7/lightcurve-columns/",
    "dr7_index": "https://dasch.cfa.harvard.edu/dr7/",
    "api_docs": "https://docs.api.starglass.cfa.harvard.edu/",
    "openapi_1": "https://api.starglass.cfa.harvard.edu/openapi.json",
    "openapi_2": "https://api.starglass.cfa.harvard.edu/public/openapi.json",
    "openapi_3": "https://docs.api.starglass.cfa.harvard.edu/openapi.json",
}
DASCHLAB_RAW = "https://raw.githubusercontent.com/pkgw/daschlab/main/daschlab/"
# ``photometry.py`` carries ``class AFlags(IntFlag)`` / ``class BFlags(IntFlag)``
# --- the per-detection blend and reject bits.  ``lightcurves.py`` only imports
# them.  The first probe (run 35738717013) fetched lightcurves.py alone, parsed
# nothing, and wrote an empty flag_bits.json, which silently disables the
# blending kill; photometry.py is therefore first in the list and is what the
# flag resolution reads.
DASCHLAB_FILES = ("photometry.py", "lightcurves.py", "refcat.py", "exposures.py",
                  "query.py", "__init__.py", "apiclient.py", "series.py")
FLAG_SOURCE_FILES = ("photometry.py", "lightcurves.py")

# Payload variants, most likely first.  The daschlab client (v1.0) posts the
# first shape of each; the others cover the obvious renamings so a first run is
# not lost to a key name.
QUERYCAT_VARIANTS = (
    lambda ra, dec, r, refcat: {"refcat": refcat, "ra_deg": ra, "dec_deg": dec,
                                "radius_arcsec": r},
    lambda ra, dec, r, refcat: {"refcat": refcat, "ra": ra, "dec": dec, "radius": r},
    lambda ra, dec, r, refcat: {"refcat": refcat, "ra_deg": ra, "dec_deg": dec,
                                "radius_deg": r / 3600.0},
)
QUERYEXPS_VARIANTS = (
    lambda ra, dec: {"ra_deg": ra, "dec_deg": dec},
    lambda ra, dec: {"ra": ra, "dec": dec},
)
LIGHTCURVE_VARIANTS = (
    lambda refcat, gbi, rn: {"refcat": refcat, "gsc_bin_index": int(gbi),
                             "ref_number": int(rn)},
    lambda refcat, gbi, rn: {"refcat": refcat, "gsc_bin_index": str(gbi),
                             "ref_number": str(rn)},
    lambda refcat, gbi, rn: {"refcat": refcat, "ref_number": int(rn)},
)


@dataclass
class ApiResponse:
    """One HTTP exchange, with everything a post-mortem needs."""

    endpoint: str
    status: int | None
    ok: bool
    elapsed_s: float
    payload: dict
    error: str = ""
    body_head: str = ""
    n_rows: int = 0
    columns: list[str] = field(default_factory=list)
    frame: pd.DataFrame | None = None
    method: str = "POST"

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "frame"}
        d["payload"] = {k: (v if isinstance(v, (int, float, str)) else str(v))
                        for k, v in self.payload.items()}
        return d


def csv_lines_to_frame(lines) -> pd.DataFrame:
    """DR7's actual wire format: a JSON **array of CSV lines**, header first.

    Verified on the runner, 2026-09-22 (run 35738717013): ``querycat`` answers
    ``["ref_text,ref_number,gsc_bin_index,ra_deg,dec_deg,...", "N0303...,...",
    ...]`` and ``queryexps`` answers
    ``["series,platenum,...,exptime,expdate,epoch,...,limMagApass,...", ...]``.
    Read as plain JSON this is a single column of strings called ``value``,
    which is exactly what the first probe reported --- 18,429 plates and not
    one usable column.
    """
    txt = "\n".join(str(x) for x in lines)
    try:
        df = pd.read_csv(io.StringIO(txt), engine="python", on_bad_lines="skip")
    except Exception:                                     # noqa: BLE001
        return pd.DataFrame({"value": list(lines)})
    if not len(df.columns):
        return pd.DataFrame({"value": list(lines)})
    return df


def _looks_like_csv(lines) -> bool:
    if not lines or not isinstance(lines[0], str):
        return False
    head = lines[0]
    return head.count(",") >= 2 and "\n" not in head


def to_frame(obj) -> pd.DataFrame:
    """Coerce a JSON answer into a DataFrame, whatever its orientation.

    Accepts a **list of CSV lines with a header first** (what DR7 actually
    returns), a dict of equal-length column arrays, a list of row dicts, a list
    of lists with a ``columns`` sibling, or a ``{"data": ...}`` /
    ``{"rows": ...}`` wrapper.  Anything else becomes an empty frame --- and
    the caller records the head of the body so the shape can be read off the
    artefact.
    """
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, dict):
        for key in ("data", "rows", "results", "sources", "exposures", "lightcurve"):
            if key in obj and isinstance(obj[key], (list, dict)):
                inner = to_frame(obj[key])
                if len(inner) or not any(isinstance(v, list) for v in obj.values()):
                    if "columns" in obj and isinstance(obj["columns"], list) \
                            and isinstance(obj[key], list) and obj[key] \
                            and isinstance(obj[key][0], list):
                        return pd.DataFrame(obj[key], columns=obj["columns"])
                    return inner
        cols = {k: v for k, v in obj.items() if isinstance(v, list)}
        if cols:
            n = {len(v) for v in cols.values()}
            if len(n) == 1:
                return pd.DataFrame(cols)
            # Ragged: keep the longest consistent set.
            m = max(n)
            return pd.DataFrame({k: v for k, v in cols.items() if len(v) == m})
        return pd.DataFrame()
    if isinstance(obj, list):
        if not obj:
            return pd.DataFrame()
        if isinstance(obj[0], dict):
            return pd.DataFrame(obj)
        if _looks_like_csv(obj):
            return csv_lines_to_frame(obj)
        if isinstance(obj[0], list):
            return pd.DataFrame(obj)
        return pd.DataFrame({"value": obj})
    return pd.DataFrame()


def _post(url: str, payload: dict, timeout_s: float, retries: int, backoff_s: float,
          session=None, method: str = "POST") -> tuple[int | None, object, str, float]:
    """Send JSON (or query params for GET); retry 5xx / transport errors, never
    4xx.  Returns ``(status, parsed_or_None, body_head, elapsed)``."""
    import requests

    sess = session or requests
    t0 = _time.monotonic()
    last_err = ""
    for attempt in range(int(max(retries, 1))):
        try:
            if str(method).upper() == "GET":
                resp = sess.get(url, params=payload, timeout=timeout_s,
                                headers={"Accept": "application/json"})
            else:
                resp = sess.post(url, json=payload, timeout=timeout_s,
                                 headers={"Accept": "application/json"})
        except Exception as exc:                          # noqa: BLE001
            last_err = repr(exc)[:300]
            _time.sleep(backoff_s * (2 ** attempt))
            continue
        head = resp.text[:600]
        if resp.status_code >= 500:
            last_err = f"HTTP {resp.status_code}: {head[:200]}"
            _time.sleep(backoff_s * (2 ** attempt))
            continue
        parsed = None
        try:
            parsed = resp.json()
        except Exception:                                 # noqa: BLE001
            parsed = None
        return resp.status_code, parsed, head, _time.monotonic() - t0
    return None, None, last_err, _time.monotonic() - t0


def post_variants(endpoint: str, variants: list[dict], *, timeout_s: float = 120.0,
                  retries: int = 3, backoff_s: float = 2.0, session=None,
                  base: str = API_BASE, methods=("POST", "GET")) -> ApiResponse:
    """Try each payload variant in order; return the first 200 (or the last failure).

    A 400/422 is a *shape* error --- a FastAPI validation body names the fields
    it wanted --- so the next key spelling is tried.  A 404/405/415 says the
    route does not accept this **verb**, so the whole variant list is retried
    with the next method rather than abandoned; a first run must not be lost to
    a POST-vs-GET guess when the queue for a runner is twenty minutes.
    """
    url = base + ENDPOINTS[endpoint]
    last: ApiResponse | None = None
    for method in methods:
        wrong_verb = False
        for payload in variants:
            status, parsed, head, dt = _post(url, payload, timeout_s, retries, backoff_s,
                                             session, method)
            if status == 200:
                df = to_frame(parsed)
                return ApiResponse(endpoint, status, True, dt, payload, "", head[:300],
                                   int(len(df)), [str(c) for c in df.columns], df, method)
            err = ("no response" if status is None else f"HTTP {status}")
            last = ApiResponse(endpoint, status, False, dt, payload, err, head, 0, [], None,
                               method)
            if status in (404, 405, 415):
                wrong_verb = True
                break
            # Anything that is not a shape error (403, 429, no response) will
            # not be cured by renaming keys or by changing the verb.
            if status not in (400, 422):
                return last
        if not wrong_verb:
            break
    return last if last is not None else ApiResponse(endpoint, None, False, 0.0, {},
                                                     "no variants", "", 0, [], None, "POST")


def querycat(ra: float, dec: float, radius_arcsec: float, *, refcat: str = "apass",
             **kw) -> ApiResponse:
    """Reference-catalogue sources around a position, with DASCH identifiers."""
    return post_variants("querycat", [v(float(ra), float(dec), float(radius_arcsec), refcat)
                                      for v in QUERYCAT_VARIANTS], **kw)


def queryexps(ra: float, dec: float, **kw) -> ApiResponse:
    """All plate exposures covering a position."""
    return post_variants("queryexps", [v(float(ra), float(dec)) for v in QUERYEXPS_VARIANTS],
                         **kw)


def lightcurve(refcat: str, gsc_bin_index, ref_number, **kw) -> ApiResponse:
    """The DASCH light curve of one reference-catalogue source."""
    return post_variants("lightcurve", [v(refcat, gsc_bin_index, ref_number)
                                        for v in LIGHTCURVE_VARIANTS], **kw)


def fetch_text(url: str, timeout_s: float = 60.0) -> tuple[int | None, str, str]:
    """GET a page; returns ``(status, text, error)``.  Runner-only."""
    import requests

    try:
        r = requests.get(url, timeout=timeout_s, headers={"User-Agent": "seti-century/0.1"})
    except Exception as exc:                              # noqa: BLE001
        return None, "", repr(exc)[:300]
    return r.status_code, r.text, ""


def html_to_text(html: str) -> str:
    """Crude but dependency-free HTML -> text, enough to read an API reference."""
    import re

    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    s = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</h\d>|</tr>", "\n", s)
    s = re.sub(r"(?i)</td>|</th>", " | ", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = (s.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">")
         .replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'"))
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()


def pick_column(df: pd.DataFrame, candidates, required: bool = False) -> str | None:
    """First column of ``df`` whose lower-cased name is in ``candidates``."""
    if df is None or not len(df.columns):
        if required:
            raise KeyError(f"none of {candidates} present (empty frame)")
        return None
    low = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    if required:
        raise KeyError(f"none of {candidates} in {list(df.columns)[:40]}")
    return None


def numeric(df: pd.DataFrame, col: str | None, default=np.nan) -> np.ndarray:
    """A float column, or a filled default when the column is absent."""
    if col is None or col not in df.columns:
        return np.full(len(df), float(default), dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def jsonable(obj):
    """Recursively convert numpy scalars/arrays so ``json.dumps`` accepts them."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def dumps(obj, **kw) -> str:
    return json.dumps(jsonable(obj), **kw)


__all__ = ["API_BASE", "ApiResponse", "DASCHLAB_FILES", "DASCHLAB_RAW", "DOC_URLS",
           "ENDPOINTS", "FLAG_SOURCE_FILES",
           "csv_lines_to_frame", "dumps", "fetch_text", "html_to_text", "jsonable",
           "lightcurve", "numeric", "pick_column", "post_variants", "querycat",
           "queryexps", "to_frame"]
