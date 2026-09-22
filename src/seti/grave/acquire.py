"""GRAVE acquisition: SGP, EarthChem and GEOROC, with runtime schema discovery.

What the record established before this module was written
-----------------------------------------------------------
* ``results/necrofrontier/probe.json``: ``sgp-search.io`` answers the GitHub
  runner but ``GET /api/v1/samples`` serves the single-page app -- the API
  path asserted in the brief does not exist.  ``env.js`` names
  ``https://archive.sgp-search.io`` as the bulk-download side.  EarthChem and
  GEOROC answer with HTML only at their landing pages.
* GitHub code search (this session): the SGP front end and the Macrostrat
  integration both call ``POST https://sgp-search.io/api/frontend/post-paged``
  with a JSON body ``{"type": "samples" | "nhhxrf", "count": N, "page": k,
  "filters": {"interpreted_age": [lo, hi]}, "show": [field codes]}`` and read
  ``response.rows``, whose keys are *display names* ("site latitude",
  "interpreted age", "Mo (ppm)").  Stockey et al. 2024 (Nature Geoscience)
  published their exact call with the codes ``height_meters, section_name,
  fe, fe_hr_fe_t, fe_py_fe_hr, toc, alu, mo, u, fe_t_al, site_type,
  coord_lat, coord_long, basin_type, meta_bin, strat_name, strat_name_long,
  environment_bin, interpreted_age, max_age, min_age, lithology_name``.  The
  full code list is not published, so the probe *learns* it: the app bundle
  is scanned for API paths and field codes, and every candidate code is
  tested with a two-row request.
* EarthChem REST (``portal.earthchem.org/restsearchservice``): GET with
  ``searchtype=count|rowdata|distinctitems``, ``outputtype=json``, filters
  ``minage/maxage`` (Ma), ``level1..level4``, ``material``, ``keyword``;
  ``startrow/endrow`` pages of at most 50 rows; ``standarditems=yes`` adds the
  standard chemical columns.
* GEOROC (DIGIS) precompiled files live on the Goettingen Dataverse:
  ``api/search?q=...&subtree=digis&type=dataset``, ``api/datasets/:persistentId``
  lists files, ``api/access/datafile/<id>?format=original`` serves a CSV whose
  header row follows a citation preamble.  GEOROC is igneous: it is the
  volcanic-ash reference, not a candidate source.

Every request is recorded (URL, status, bytes, elapsed, error) in the
acquisition ledger; nothing is fabricated; a source that produces no rows is
``NO_DATA_REACHED`` for that source.  ``fetch_fn`` is injectable so the tests
script every route.
"""

from __future__ import annotations

import json
import re
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import references as R

USER_AGENT = "seti-grave/1.0 (+technosignature search; S56)"
STATUS_OK, STATUS_NO_DATA = "OK", "NO_DATA_REACHED"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
@dataclass
class FetchResult:
    url: str
    status: int | None = None
    content: bytes = b""
    error: str | None = None
    elapsed_s: float = 0.0
    content_type: str = ""
    method: str = "GET"

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300 and not self.error

    @property
    def text(self) -> str:
        try:
            return self.content.decode("utf-8", errors="replace")
        except Exception:                                  # noqa: BLE001
            return ""

    def json(self):
        try:
            return json.loads(self.text)
        except Exception:                                  # noqa: BLE001
            return None

    def record(self) -> dict:
        return {"url": self.url, "method": self.method, "status": self.status,
                "bytes": len(self.content), "elapsed_s": round(self.elapsed_s, 2),
                "content_type": self.content_type, "error": self.error}


def http_fetch(url: str, *, method: str = "GET", json_body=None, params=None, timeout: float = 120.0,
               headers=None, retries: int = 2) -> FetchResult:
    """One HTTP call with a small retry; never raises."""
    import requests  # noqa: PLC0415

    hdr = {"User-Agent": USER_AGENT, "Accept": "application/json, text/plain, */*"}
    hdr.update(headers or {})
    last = FetchResult(url=url, method=method)
    for attempt in range(retries + 1):
        t0 = _time.time()
        try:
            r = requests.request(method, url, json=json_body, params=params, timeout=timeout, headers=hdr)
            last = FetchResult(url=str(r.url), status=r.status_code, content=r.content,
                               elapsed_s=_time.time() - t0, content_type=r.headers.get("Content-Type", ""),
                               method=method)
            if r.status_code < 500:
                return last
        except Exception as exc:                          # noqa: BLE001
            last = FetchResult(url=url, error=f"{type(exc).__name__}: {exc}"[:300],
                               elapsed_s=_time.time() - t0, method=method)
        _time.sleep(2.0 * (attempt + 1))
    return last


# ---------------------------------------------------------------------------
# Column resolution (shared by every source)
# ---------------------------------------------------------------------------
_ELEMENTS = set(R.ATOMIC_MASS)
_UNIT_SCALE = {"ppm": 1.0, "mg/kg": 1.0, "ug/g": 1.0, "µg/g": 1.0, "ppb": 1e-3, "ng/g": 1e-3,
               "ppt": 1e-6, "wt%": 1e4, "wt.%": 1e4, "%": 1e4, "wt %": 1e4, "wt. %": 1e4}
_RE_EL_UNIT = re.compile(r"^\s*([A-Za-z]{1,3})\s*[\(\[]\s*([A-Za-z.µ/ %]+?)\s*[\)\]]\s*$")
_RE_OX_UNIT = re.compile(r"^\s*([A-Za-z]{1,2}\d?[Oo]\d?[Tt]?(?:otal)?)\s*[\(\[]\s*([A-Za-z.µ/ %]+?)\s*[\)\]]\s*$")
_RE_BARE = re.compile(r"^\s*([A-Za-z]{1,3})\s*$")

#: Short codes the SGP app uses that are not plain symbols.
CODE_ALIASES: dict[str, str] = {"alu": "Al", "toc": "TOC", "tic": "TIC", "tot_c": "TC", "s_tot": "S"}

#: Metadata roles, regex over a lower-cased, whitespace-collapsed key.
META_ROLES: dict[str, list[str]] = {
    "sample_id": [r"^sample identifier$", r"^sample[_ ]?id$", r"^sampleid$", r"^unique[_ ]id$", r"^id$",
                  r"^sample[_ ]name$", r"^sample$"],
    "section": [r"^section[_ ]name$", r"^section$", r"^site[_ ]name$", r"^location$"],
    "site_type": [r"^site[_ ]type$"],
    "lat": [r"^site latitude$", r"^coord[_ ]lat$", r"^latitude$", r"^lat$", r"^latitude[_ ]\(min\.?\)$"],
    "lon": [r"^site longitude$", r"^coord[_ ]long$", r"^longitude$", r"^lon$", r"^long$",
            r"^longitude[_ ]\(min\.?\)$"],
    "age": [r"^interpreted[_ ]age$", r"^age$", r"^age[_ ]?\(ma\)$", r"^age[_ ]ma$"],
    "min_age": [r"^min[_ ]age$", r"^minimum age$", r"^min\.? age \(yrs\.?\)$", r"^age[_ ]min$"],
    "max_age": [r"^max[_ ]age$", r"^maximum age$", r"^max\.? age \(yrs\.?\)$", r"^age[_ ]max$"],
    "lithology": [r"^lithology[_ ]name$", r"^lithology$", r"^rock[_ ]name$", r"^rock[_ ]type$", r"^level4$",
                  r"^rock name$"],
    "strat": [r"^strat[_ ]name(?:[_ ]long)?$", r"^stratigraphy name$", r"^long stratigraphy name$",
              r"^formation$", r"^geological[_ ]unit$"],
    "basin": [r"^basin[_ ]type$", r"^basin$"],
    "environment": [r"^environment(?:al)?[_ ]bin$", r"^depositional[_ ]environment$", r"^environment$"],
    "meta_bin": [r"^meta(?:morphic)?[_ ]bin$"],
    "height": [r"^height[_ ]meters$", r"^height/depth$", r"^height$", r"^depth$", r"^height \(m\)$"],
    "reference": [r"^ref(?:erence)?(?:[_ ]short|[_ ]long|s)?$", r"^citations?$", r"^source$", r"^data[_ ]source$",
                  r"^original[_ ]source$", r"^doi$"],
    "country": [r"^country$"],
    "analytical_method": [r"^method$", r"^analytical[_ ]method$", r"^analysis[_ ]type$"],
    # SGP serves no citation, DOI or method field (run 35738860553), so these
    # two are most of what provenance there is and they are kept as roles.
    "collector": [r"^collector$", r"^collected[_ ]by$"],
    "state_province": [r"^state/province$", r"^state[_ ]province$", r"^state$", r"^province$"],
}


def _norm_key(k) -> str:
    return re.sub(r"\s+", " ", str(k).strip().lower())


def _element_from_token(tok: str) -> str | None:
    t = tok.strip()
    if t.lower() in CODE_ALIASES:
        return CODE_ALIASES[t.lower()]
    if t.lower() in ("toc", "tic", "tc"):
        return t.upper()
    cap = t[:1].upper() + t[1:].lower()
    return cap if cap in _ELEMENTS else None


def _oxide_lookup(tok: str) -> tuple[str, float] | None:
    t = re.sub(r"total$", "T", tok.strip(), flags=re.I)
    for k, v in R.OXIDE_TO_ELEMENT.items():
        if k.lower() == t.lower():
            return v
    return None


def resolve_columns(columns, *, source: str = "") -> dict:
    """Map raw column names to roles.

    Returns ``{"elements": {El: [(col, scale, kind), ...]}, "meta": {role: col},
    "unresolved": [...], "assumed_units": {col: unit}}``.  ``scale`` converts
    the column to ppm.  A bare code ("mo", "alu") is given the unit its
    class implies (wt% for majors and organic carbon, ppm otherwise) and that
    assumption is recorded so the numbers can be checked against medians.
    """
    elements: dict[str, list] = {}
    meta: dict[str, str] = {}
    unresolved: list[str] = []
    assumed: dict[str, str] = {}
    for col in columns:
        key = _norm_key(col)
        role = None
        for r, pats in META_ROLES.items():
            if r in meta:
                continue
            if any(re.match(p, key) for p in pats):
                role = r
                break
        if role:
            meta[role] = col
            continue
        m = _RE_OX_UNIT.match(str(col))
        if m and _oxide_lookup(m.group(1)):
            el, frac = _oxide_lookup(m.group(1))
            unit = m.group(2).strip().lower().replace("wt.%", "wt%").replace("wt %", "wt%")
            sc = _UNIT_SCALE.get(unit)
            if sc:
                elements.setdefault(el, []).append((col, sc * frac, "oxide"))
                continue
        m = _RE_EL_UNIT.match(str(col))
        if m and _element_from_token(m.group(1)):
            el = _element_from_token(m.group(1))
            unit = m.group(2).strip().lower().replace("wt.%", "wt%").replace("wt %", "wt%")
            sc = _UNIT_SCALE.get(unit)
            if sc:
                if el in ("TOC", "TIC", "TC"):
                    sc = sc / 1e4                      # organic / inorganic carbon stays in wt%
                elements.setdefault(el, []).append((col, sc, "element"))
                continue
        m = _RE_BARE.match(str(col))
        if m:
            tok = m.group(1)
            if tok.lower() in ("toc", "tic", "tc"):
                elements.setdefault(tok.upper(), []).append((col, 1.0, "wt%_assumed"))
                assumed[col] = "wt%"
                continue
            el = _element_from_token(tok)
            if el:
                major = el in R.MAJORS or el == "Si"
                elements.setdefault(el, []).append((col, 1e4 if major else 1.0, "bare_assumed"))
                assumed[col] = "wt%" if major else "ppm"
                continue
        unresolved.append(str(col))
    # oxide/total first so Fe from Fe2O3T beats Fe from FeO
    for el in elements:
        elements[el].sort(key=lambda t: (0 if t[2] == "oxide" and t[0].lower().rstrip("t") != t[0].lower() else 1))
    return {"elements": elements, "meta": meta, "unresolved": unresolved, "assumed_units": assumed,
            "source": source}


def _to_num(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.strip().str.replace(",", "", regex=False)
        s = s.str.replace(r"^[<>]\s*", "", regex=True)      # "<0.5" -> 0.5 (a limit; the DL mask catches repeats)
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan, "n.d.": np.nan, "bdl": np.nan})
    return pd.to_numeric(s, errors="coerce")


def canonicalise(raw: pd.DataFrame, res: dict, *, source: str) -> pd.DataFrame:
    """One row per sample: metadata columns + one ppm column per element.

    Non-positive values become NaN.  When two raw columns feed one element
    the first non-null in priority order wins.  Ages in years (GEOROC) are
    converted to Ma when the median exceeds 10^5.
    """
    out = pd.DataFrame(index=raw.index)
    out["source"] = source
    for role, col in res["meta"].items():
        out[role] = raw[col]
    for role in ("age", "min_age", "max_age"):
        if role in out.columns:
            v = _to_num(out[role])
            if v.notna().any() and float(np.nanmedian(v)) > 1e5:
                v = v / 1e6
            out[role] = v
    for role in ("lat", "lon", "height"):
        if role in out.columns:
            out[role] = _to_num(out[role])
    for el, cands in res["elements"].items():
        vals = None
        for col, scale, _kind in cands:
            v = _to_num(raw[col]) * scale
            v = v.where(v > 0)
            vals = v if vals is None else vals.fillna(v)
        out[el] = vals
    if "sample_id" not in out.columns:
        out["sample_id"] = [f"{source}_{i}" for i in range(len(out))]
    out["sample_id"] = out["sample_id"].astype(str)
    return out


def element_census(canon: pd.DataFrame) -> dict[str, int]:
    return {el: int(canon[el].notna().sum()) for el in canon.columns
            if el in _ELEMENTS or el in ("TOC", "TIC", "TC")}


# ---------------------------------------------------------------------------
# SGP
# ---------------------------------------------------------------------------
_RE_API_PATH = re.compile(r"/api/[A-Za-z0-9_./-]{2,80}")
_RE_BUNDLE = re.compile(r"(/static/js/main\.[0-9a-f]+\.js)")
_RE_VALUE_LABEL = re.compile(r"""value\s*:\s*["']([a-z0-9_]{1,40})["']\s*,\s*label\s*:\s*["']([^"']{1,60})["']""")
_RE_LABEL_VALUE = re.compile(r"""label\s*:\s*["']([^"']{1,60})["']\s*,\s*value\s*:\s*["']([a-z0-9_]{1,40})["']""")


def sgp_body(conf: dict, *, count: int, page: int, show: list[str], filters: dict | None = None,
             kind: str | None = None) -> dict:
    return {"type": kind or conf.get("type", "samples"), "count": int(count), "page": int(page),
            "filters": dict(filters or {}), "show": list(show)}


def _post(fetch, url, body, timeout):
    return fetch(url, method="POST", json_body=body, timeout=timeout)


def _rows_of(fr: FetchResult) -> list[dict] | None:
    if not fr.ok:
        return None
    j = fr.json()
    if isinstance(j, dict):
        for k in ("rows", "data", "results", "samples"):
            if isinstance(j.get(k), list):
                return j[k]
        return None
    if isinstance(j, list):
        return j
    return None


def sgp_probe(conf: dict, *, fetch=http_fetch) -> dict:
    """Learn the SGP API on the runner: paths from the bundle, codes by trial.

    Returns a ledger with ``hosts`` (per host: env.js, bundle paths, API paths
    seen in the bundle, label/value pairs), ``base_call`` (the Stockey body's
    outcome and the row keys it returned), ``type_variants``, and
    ``accepted_codes`` -- ``{code: new row key}`` for every candidate field
    code that the API honoured, plus ``rejected_codes`` and
    ``silently_dropped``.
    """
    s = conf["sgp"]
    timeout = float(s.get("timeout_s", 120))
    window = [float(x) for x in (s.get("probe_age_window") or [0.0, 4000.0])]
    ledger: dict = {"generated_utc": _now(), "hosts": {}, "requests": [], "probe_age_window": window}

    def rec(fr: FetchResult, why: str) -> FetchResult:
        d = fr.record()
        d["why"] = why
        ledger["requests"].append(d)
        return fr

    codes_from_bundle: dict[str, str] = {}
    for host in s.get("hosts", []):
        h: dict = {"env_js": None, "bundles": [], "api_paths": [], "label_value_pairs": {}}
        fr = rec(fetch(f"{host}/env.js", timeout=timeout), "env.js declares the API base URL")
        h["env_js"] = fr.text[:400] if fr.ok else None
        idx = rec(fetch(f"{host}/", timeout=timeout), "index: find the app bundle")
        bundles = sorted(set(_RE_BUNDLE.findall(idx.text))) if idx.ok else []
        for b in bundles[:2]:
            bf = rec(fetch(f"{host}{b}", timeout=timeout), "app bundle: API paths and field codes")
            if not bf.ok:
                continue
            txt = bf.text
            h["bundles"].append({"path": b, "bytes": len(bf.content)})
            h["api_paths"] = sorted(set(h["api_paths"]) | set(_RE_API_PATH.findall(txt)))[:80]
            pairs = {v: lab for v, lab in _RE_VALUE_LABEL.findall(txt)}
            pairs.update({v: lab for lab, v in _RE_LABEL_VALUE.findall(txt)})
            h["label_value_pairs"] = dict(list(pairs.items())[:400])
            codes_from_bundle.update(pairs)
        ledger["hosts"][host] = h

    api = s["post_paged_url"]
    base_show = list(s.get("base_show", []))
    fr = rec(_post(fetch, api, sgp_body(s, count=3, page=1, show=base_show,
                                         filters={"interpreted_age": window}),
                   timeout), "base call: the published Stockey et al. body")
    rows = _rows_of(fr)
    j = fr.json() if fr.ok else None
    ledger["base_call"] = {"status": fr.status, "n_rows": len(rows) if rows is not None else None,
                           "row_keys": sorted(rows[0].keys()) if rows else [],
                           "response_keys": sorted(j.keys()) if isinstance(j, dict) else None,
                           "head": fr.text[:300]}
    ledger["type_variants"] = {}
    for kind in s.get("type_variants", ["samples", "nhhxrf"]):
        fr2 = rec(_post(fetch, api, sgp_body(s, count=2, page=1, show=base_show, kind=kind,
                                              filters={"interpreted_age": window}), timeout),
                  f"type variant {kind!r}")
        r2 = _rows_of(fr2)
        ledger["type_variants"][kind] = {"status": fr2.status, "n_rows": len(r2) if r2 is not None else None,
                                         "row_keys": sorted(r2[0].keys()) if r2 else []}
    # which codes does the API honour?  Each candidate rides alone next to the
    # anchor codes; a new key in the rows is the acceptance test.
    anchor = list(s.get("anchor_show", ["interpreted_age", "coord_lat"]))
    fr0 = rec(_post(fetch, api, sgp_body(s, count=2, page=1, show=anchor,
                                          filters={"interpreted_age": window}), timeout), "anchor-only keys")
    r0 = _rows_of(fr0) or []
    base_keys = set(r0[0].keys()) if r0 else set()
    j0 = fr0.json() if fr0.ok else None
    ledger["anchor_call"] = {"status": fr0.status, "n_rows": len(r0),
                             "response_keys": sorted(j0.keys()) if isinstance(j0, dict) else None,
                             "head": fr0.text[:300]}
    # how many rows does the service actually hand back when asked for many?
    # A bin must not be judged finished by a page that the SERVER capped.
    cap = int(s.get("page_size_probe", 5000))
    frp = rec(_post(fetch, api, sgp_body(s, count=cap, page=1, show=anchor,
                                         filters={"interpreted_age": window}), timeout),
              f"page-size probe: ask for {cap} rows and see how many come back")
    rp = _rows_of(frp) or []
    jp = frp.json() if frp.ok else None
    ledger["page_size_probe"] = {"asked": cap, "returned": len(rp), "status": frp.status,
                                 "response_keys": sorted(jp.keys()) if isinstance(jp, dict) else None,
                                 "total_field": {k: jp.get(k) for k in ("total", "count", "n", "totalCount")
                                                 if isinstance(jp, dict) and k in jp}}
    candidates = list(dict.fromkeys(list(s.get("show_candidates", [])) + list(codes_from_bundle)))
    accepted, rejected, dropped = {}, {}, []
    max_tests = int(s.get("max_code_tests", 200))
    for code in candidates[:max_tests]:
        if code in anchor:
            continue
        frc = _post(fetch, api, sgp_body(s, count=2, page=1, show=anchor + [code],
                                          filters={"interpreted_age": window}), timeout)
        rc = _rows_of(frc)
        if rc is None:
            rejected[code] = frc.status if frc.status is not None else (frc.error or "no response")
            continue
        new = sorted(set(rc[0].keys()) - base_keys) if rc else []
        if new:
            accepted[code] = new[0]
        else:
            dropped.append(code)
    ledger["anchor_keys"] = sorted(base_keys)
    ledger["accepted_codes"] = accepted
    ledger["rejected_codes"] = rejected
    ledger["silently_dropped"] = dropped
    ledger["n_candidates_tested"] = min(len(candidates), max_tests)
    # The API answers the ANCHOR call and the per-code trials even when the
    # published composite body is refused (run 35738860553: the Stockey body
    # 400s on `fe_t_al`, "not a valid attribute in this search type", while 93
    # codes are individually accepted).  "Reached" is therefore about rows, not
    # about that one body.
    ledger["reached"] = bool(rows) or bool(base_keys) or bool(accepted) \
        or any(v["n_rows"] for v in ledger["type_variants"].values())
    # the service's own attribute listings, if it publishes any
    ledger["attribute_endpoints"] = {}
    for host in s.get("hosts", []):
        for path in s.get("attribute_paths", []):
            for method, body in (("GET", None), ("POST", {"type": s.get("type", "samples")})):
                fr3 = rec(fetch(f"{host}{path}", method=method, json_body=body,
                                timeout=min(timeout, 30.0), retries=0),
                          f"attribute listing {path} ({method})")
                # the SPA answers 200 with its index for every unknown path, so
                # an HTML body is "this is a client-side route", not a listing
                is_html = "text/html" in (fr3.content_type or "") or fr3.text.lstrip()[:9].lower() == "<!doctype"
                if fr3.ok and len(fr3.content) > 40 and not is_html:
                    ledger["attribute_endpoints"][f"{method} {host}{path}"] = {
                        "status": fr3.status, "bytes": len(fr3.content), "head": fr3.text[:4000]}
                    break
    return ledger


def sgp_acquire(conf: dict, *, fetch=http_fetch, show: list[str], out_dir: Path | None = None,
                kind: str | None = None, max_rows: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Page every age bin through ``post-paged``; checkpoint each bin to CSV.

    ``show`` is the accepted code list from the probe.

    A bin ends when a page comes back **empty**, or when it repeats the
    previous page's sample identifiers, or at ``max_pages_per_bin``.  It does
    *not* end merely because a page is shorter than ``count``: if the service
    caps a page at, say, 1000 rows, that rule would silently truncate every bin
    at its first page and the run would look like a sparse record instead of a
    capped request.  The cap the service actually applies is measured by the
    probe (``page_size_probe``) and recorded here as ``observed_page_size``.

    The ledger carries every request and the per-bin row counts; ``status`` is
    ``NO_DATA_REACHED`` if no bin returned a row.
    """
    s = conf["sgp"]
    timeout = float(s.get("timeout_s", 180))
    count = int(s.get("page_count", 5000))
    max_pages = int(s.get("max_pages_per_bin", 200))
    id_key = s.get("id_key", "sample identifier")
    edges = list(s.get("age_bin_edges_ma", [0, 4000]))
    ledger: dict = {"generated_utc": _now(), "source": "sgp", "url": s["post_paged_url"], "type": kind or s.get("type"),
                    "show": list(show), "bins": [], "requests": []}
    frames: list[pd.DataFrame] = []
    total = 0
    observed_page = 0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        nb = 0
        bin_frames = []
        seen_ids: set[str] = set()
        stop = ""
        for page in range(1, max_pages + 1):
            body = sgp_body(s, count=count, page=page, show=show, kind=kind,
                            filters={"interpreted_age": [float(lo), float(hi)]})
            fr = _post(fetch, s["post_paged_url"], body, timeout)
            d = fr.record()
            d.update({"bin": [lo, hi], "page": page})
            rows = _rows_of(fr)
            d["n_rows"] = len(rows) if rows is not None else None
            jr = fr.json() if fr.ok else None
            if isinstance(jr, dict) and "count" in jr:
                d["service_count"] = jr["count"]      # rows the service says match this filter
            ledger["requests"].append(d)
            if not rows:
                stop = "empty_page" if rows is not None else f"no_rows_status_{fr.status}"
                break
            observed_page = max(observed_page, len(rows))
            ids = {str(r.get(id_key)) for r in rows if r.get(id_key) is not None}
            if ids and ids <= seen_ids:
                stop = "page_repeated_previous_ids"
                break
            seen_ids |= ids
            df = pd.DataFrame(rows)
            bin_frames.append(df)
            nb += len(df)
            if max_rows and total + nb >= max_rows:
                stop = "max_rows"
                break
        else:
            stop = "max_pages_per_bin"
        svc = [r.get("service_count") for r in ledger["requests"]
               if r.get("bin") == [lo, hi] and r.get("service_count") is not None]
        ledger["bins"].append({"age_lo": lo, "age_hi": hi, "n_rows": nb, "n_pages": len(bin_frames),
                               "stopped_because": stop, "service_count": svc[0] if svc else None,
                               "complete": (bool(svc) and nb >= int(svc[0])) if svc else None})
        if bin_frames:
            bdf = pd.concat(bin_frames, ignore_index=True)
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                bdf.to_csv(out_dir / f"sgp_bin_{lo:g}_{hi:g}.csv", index=False)
            frames.append(bdf)
            total += nb
        if max_rows and total >= max_rows:
            ledger["truncated_at_max_rows"] = max_rows
            break
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    ledger["n_rows_before_dedupe"] = int(len(raw))
    if id_key in raw.columns:
        raw = raw.drop_duplicates(subset=[id_key])
    ledger["observed_page_size"] = observed_page
    ledger["n_rows"] = int(len(raw))
    ledger["status"] = STATUS_OK if len(raw) else STATUS_NO_DATA
    return raw, ledger


# ---------------------------------------------------------------------------
# EarthChem
# ---------------------------------------------------------------------------
def _ec_count(fr: FetchResult) -> int | None:
    j = fr.json() if fr.ok else None
    if isinstance(j, dict):
        for k in ("Count", "count", "COUNT", "total", "totalResults", "numFound"):
            if k in j:
                try:
                    return int(j[k])
                except (TypeError, ValueError):
                    return None
    if isinstance(j, list):
        return len(j)
    if fr.ok and fr.text.strip().isdigit():
        return int(fr.text.strip())
    return None


def earthchem_endpoint_ladder(conf: dict, *, fetch=http_fetch) -> dict:
    """Which EarthChem base URL, if any, actually serves a search.

    Run 35738860553 established that ``portal.earthchem.org/restsearchservice``
    -- the path the brief asserts and the one every citation of the EarthChem
    REST API gives -- now answers **404 from Apache** for every parameter
    spelling.  The service moved.  This ladder asks each candidate base in turn
    with the shape it expects, and records exactly what each served, so the
    verdict is "this endpoint no longer exists", not "EarthChem has no shale".
    """
    e = conf["earthchem"]
    timeout = float(e.get("timeout_s", 120))
    out: dict = {"tried": [], "working": None}
    for cand in e.get("rest_url_candidates", []):
        url = cand["url"] if isinstance(cand, dict) else str(cand)
        params = dict((cand.get("params") if isinstance(cand, dict) else None) or
                      {"searchtype": "count", "outputtype": "json", "keyword": "shale"})
        method = (cand.get("method") if isinstance(cand, dict) else None) or "GET"
        body = cand.get("json") if isinstance(cand, dict) else None
        # one attempt, short timeout: a ladder rung that does not resolve must
        # cost seconds, not the three-attempt retry ladder of a real request
        fr = fetch(url, method=method, params=(None if body else params), json_body=body,
                   timeout=min(timeout, float(e.get("ladder_timeout_s", 25))), retries=0)
        cnt = _ec_count(fr)
        rec = {"url": url, "method": method, "params": (body or params), "status": fr.status,
               "bytes": len(fr.content), "count": cnt, "content_type": fr.content_type,
               "head": fr.text[:300], "error": fr.error}
        out["tried"].append(rec)
        if fr.ok and (cnt or len(fr.content) > 200) and "text/html" not in (fr.content_type or ""):
            out["working"] = url
            break
    return out


def earthchem_probe(conf: dict, *, fetch=http_fetch) -> dict:
    """Find a live EarthChem search endpoint, then count and shape it."""
    e = conf["earthchem"]
    timeout = float(e.get("timeout_s", 120))
    ladder = earthchem_endpoint_ladder(conf, fetch=fetch)
    url = ladder.get("working") or e["rest_url"]
    ledger: dict = {"generated_utc": _now(), "url": url, "endpoint_ladder": ladder,
                    "counts": {}, "distinct": {}, "requests": []}
    for name, params in (e.get("count_queries") or {}).items():
        p = dict(params)
        p.update({"searchtype": "count", "outputtype": "json"})
        fr = fetch(url, params=p, timeout=timeout)
        d = fr.record()
        d["query"] = name
        ledger["requests"].append(d)
        cnt = _ec_count(fr)
        ledger["counts"][name] = {"params": params, "status": fr.status, "count": cnt, "head": fr.text[:120]}
    for item in e.get("distinct_items", []):
        p = {"searchtype": "distinctitems", "outputtype": "json", "outputitems": item}
        p.update(e.get("distinct_base_params") or {})
        fr = fetch(url, params=p, timeout=timeout)
        d = fr.record()
        d["distinct"] = item
        ledger["requests"].append(d)
        ledger["distinct"][item] = {"status": fr.status, "head": fr.text[:600]}
    # one rowdata page, to see the row keys
    best = max(((k, v) for k, v in ledger["counts"].items() if v.get("count")), key=lambda kv: kv[1]["count"],
               default=None)
    ledger["best_query"] = best[0] if best else None
    if best:
        p = dict(best[1]["params"])
        p.update({"searchtype": "rowdata", "outputtype": "json", "standarditems": "yes", "startrow": 0, "endrow": 4})
        fr = fetch(url, params=p, timeout=timeout)
        ledger["requests"].append(fr.record())
        j = fr.json() if fr.ok else None
        rows = j if isinstance(j, list) else (j.get("rows") if isinstance(j, dict) else None)
        ledger["row_keys"] = sorted(rows[0].keys()) if rows else []
        ledger["rowdata_head"] = fr.text[:300]
    ledger["reached"] = bool(any(v.get("count") for v in ledger["counts"].values()) or ledger.get("row_keys"))
    return ledger


def earthchem_acquire(conf: dict, *, fetch=http_fetch, query_params: dict, max_rows: int,
                      windows: list[dict] | None = None, out_dir: Path | None = None,
                      url: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Page rowdata in 50-row steps: boundary windows first, then the general pull."""
    e = conf["earthchem"]
    url = url or e["rest_url"]
    timeout = float(e.get("timeout_s", 120))
    page = int(e.get("page_size", 50))
    ledger: dict = {"generated_utc": _now(), "source": "earthchem", "url": url, "query_params": query_params,
                    "windows": [], "requests": [], "n_rows": 0}
    frames = []

    def pull(params: dict, cap: int, tag: str) -> int:
        got = 0
        start = 0
        empty_streak = 0
        while got < cap:
            p = dict(params)
            p.update({"searchtype": "rowdata", "outputtype": "json", "standarditems": "yes",
                      "startrow": start, "endrow": start + page - 1})
            fr = fetch(url, params=p, timeout=timeout)
            d = fr.record()
            d.update({"tag": tag, "startrow": start})
            j = fr.json() if fr.ok else None
            rows = j if isinstance(j, list) else (j.get("rows") if isinstance(j, dict) else None)
            d["n_rows"] = len(rows) if rows else 0
            ledger["requests"].append(d)
            if not rows:
                empty_streak += 1
                if empty_streak >= 2 or not fr.ok:
                    break
                start += page
                continue
            empty_streak = 0
            df = pd.DataFrame(rows)
            df["_grave_window"] = tag
            frames.append(df)
            got += len(df)
            if len(df) < page:
                break
            start += page
        return got

    for w in windows or []:
        p = dict(query_params)
        p.update({"minage": w["age_lo"], "maxage": w["age_hi"]})
        n = pull(p, int(e.get("max_rows_per_window", 1000)), w["key"])
        ledger["windows"].append({"key": w["key"], "age_lo": w["age_lo"], "age_hi": w["age_hi"], "n_rows": n})
    n_general = pull(dict(query_params), max_rows, "general")
    ledger["windows"].append({"key": "general", "n_rows": n_general})
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(raw) and "sample_id" in raw.columns:
        raw = raw.drop_duplicates(subset=["sample_id"])
    if out_dir is not None and len(raw):
        out_dir.mkdir(parents=True, exist_ok=True)
        raw.to_csv(out_dir / "earthchem_raw.csv", index=False)
    ledger["n_rows"] = int(len(raw))
    ledger["status"] = STATUS_OK if len(raw) else STATUS_NO_DATA
    return raw, ledger


# ---------------------------------------------------------------------------
# GEOROC (Dataverse)
# ---------------------------------------------------------------------------
def georoc_probe(conf: dict, *, fetch=http_fetch) -> dict:
    """List DIGIS datasets and the files that match the sedimentary / tephra regex."""
    g = conf["georoc"]
    api = g["dataverse_api"].rstrip("/")
    timeout = float(g.get("timeout_s", 120))
    ledger: dict = {"generated_utc": _now(), "api": api, "datasets": [], "files": [], "requests": []}
    start, total = 0, None
    per = 100
    while True:
        fr = fetch(f"{api}/search", params={"q": g.get("search_q", "*"), "subtree": g.get("subtree", "digis"),
                                            "type": "dataset", "per_page": per, "start": start}, timeout=timeout)
        ledger["requests"].append(fr.record())
        j = fr.json() if fr.ok else None
        data = (j or {}).get("data") if isinstance(j, dict) else None
        if not data:
            break
        total = data.get("total_count")
        for it in data.get("items", []):
            ledger["datasets"].append({"name": it.get("name"), "doi": it.get("global_id"),
                                       "published": it.get("published_at")})
        start += per
        if total is None or start >= total or start >= int(g.get("max_datasets", 500)):
            break
    ledger["n_datasets"] = len(ledger["datasets"])
    ds_re = re.compile(g.get("dataset_regex", "(?i)rock types|sediment"))
    f_re = re.compile(g.get("file_regex", "(?i)sediment|tuff|tephra|ash|pyroclast"))
    for ds in ledger["datasets"]:
        if not ds.get("doi") or not ds_re.search(ds.get("name") or ""):
            continue
        fr = fetch(f"{api}/datasets/:persistentId/", params={"persistentId": ds["doi"]}, timeout=timeout)
        ledger["requests"].append(fr.record())
        j = fr.json() if fr.ok else None
        files = ((((j or {}).get("data") or {}).get("latestVersion") or {}).get("files")) or []
        ds["n_files"] = len(files)
        names = []
        for f in files:
            df = f.get("dataFile") or {}
            name = df.get("filename") or f.get("label") or ""
            names.append(name)
            if f_re.search(name):
                ledger["files"].append({"dataset": ds["name"], "doi": ds["doi"], "id": df.get("id"),
                                        "filename": name, "filesize": df.get("filesize"),
                                        "url": f"{api}/access/datafile/{df.get('id')}?format=original"})
        ds["filenames"] = sorted(names)[: int(g.get("max_filenames_recorded", 150))]
    ledger["n_matching_files"] = len(ledger["files"])
    ledger["reached"] = ledger["n_datasets"] > 0
    return ledger


def parse_georoc_csv(text: str) -> pd.DataFrame:
    """GEOROC precompiled CSV: a citation preamble, then the header row."""
    import io  # noqa: PLC0415

    lines = text.splitlines()
    hdr = 0
    for i, ln in enumerate(lines[:200]):
        u = ln.upper()
        if ("SAMPLE NAME" in u or "UNIQUE_ID" in u or "CITATIONS" in u) and "," in ln:
            hdr = i
            break
    body = "\n".join(lines[hdr:])
    # the file may carry a references block after the data; keep rows with the header's width
    df = pd.read_csv(io.StringIO(body), dtype=str, on_bad_lines="skip", engine="python")
    df = df.dropna(how="all")
    return df


def georoc_acquire(conf: dict, *, fetch=http_fetch, files: list[dict], out_dir: Path | None = None
                   ) -> tuple[pd.DataFrame, dict]:
    g = conf["georoc"]
    timeout = float(g.get("timeout_s", 300))
    max_bytes = int(g.get("max_total_bytes", 200_000_000))
    ledger: dict = {"generated_utc": _now(), "source": "georoc", "files": [], "requests": []}
    frames, got = [], 0
    for f in files[: int(g.get("max_files", 20))]:
        if f.get("filesize") and got + int(f["filesize"]) > max_bytes:
            ledger["files"].append({"filename": f["filename"], "skipped": "byte cap"})
            continue
        fr = fetch(f["url"], timeout=timeout)
        ledger["requests"].append(fr.record())
        if not fr.ok:
            ledger["files"].append({"filename": f["filename"], "status": fr.status, "error": fr.error})
            continue
        try:
            df = parse_georoc_csv(fr.text)
        except Exception as exc:                          # noqa: BLE001
            ledger["files"].append({"filename": f["filename"], "parse_error": repr(exc)[:200]})
            continue
        df["_grave_file"] = f["filename"]
        frames.append(df)
        got += len(fr.content)
        ledger["files"].append({"filename": f["filename"], "n_rows": int(len(df)), "bytes": len(fr.content)})
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / f"georoc_{re.sub(r'[^A-Za-z0-9_.-]', '_', f['filename'])}", index=False)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    ledger["n_rows"] = int(len(raw))
    ledger["status"] = STATUS_OK if len(raw) else STATUS_NO_DATA
    return raw, ledger


@dataclass
class Acquisition:
    """What the acquire stage hands to the screen: canonical tables per source."""

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    ledgers: dict[str, dict] = field(default_factory=dict)
    resolutions: dict[str, dict] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return STATUS_OK if any(len(t) for t in self.tables.values()) else STATUS_NO_DATA


__all__ = [
    "Acquisition", "CODE_ALIASES", "FetchResult", "META_ROLES", "STATUS_NO_DATA", "STATUS_OK",
    "canonicalise", "earthchem_acquire", "earthchem_probe", "element_census", "georoc_acquire",
    "georoc_probe", "http_fetch", "parse_georoc_csv", "resolve_columns", "sgp_acquire", "sgp_body",
    "sgp_probe",
]
