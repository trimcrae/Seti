"""CRYPT acquisition: reach the Diviner, Mini-RF, ShadowCam and LOLA products honestly.

The probe on ``main`` (``results/necrofrontier/probe.json``, three runs)
established that the PDS Geosciences node answers the runner with IIS
directory listings, that ``lro-l-dlre-4-rdr-v1/`` carries two volumes —
``lrodlr_1001`` (RDR by year) and ``lrodlr_1002`` (the level-3/4 products) —
that the UCLA mirror fails TLS verification (not bypassed), that the ODE PCP
description page answers, and that the ShadowCam archive and SIS answer.
What it did NOT establish is the product naming, the label dialect or the
sizes inside ``lrodlr_1002``: those are *learned on the runner* by this
module and committed in ``probe.json`` so the next iteration can fix the
classification patterns from evidence.

Everything network-facing goes through one injectable ``fetch`` callable so
the offline tests script the archive.  Nothing here fabricates a raster: a
product that cannot be reached is recorded as such and the stage verdict
becomes ``NO_DATA_REACHED``.
"""

from __future__ import annotations

import json
import math
import re
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

USER_AGENT = "seti-crypt/1.0 (+technosignature search; S55 lunar PSR thermal/radar)"
LABEL_SUFFIXES = (".lbl", ".xml")
IMAGE_SUFFIXES = (".img", ".tif", ".tiff", ".cub", ".raw")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------
@dataclass
class FetchResult:
    url: str
    status: int | None = None
    content: bytes = b""
    path: str | None = None          # when streamed to disk
    final_url: str | None = None
    error: str | None = None
    elapsed_s: float = 0.0
    n_bytes: int = 0
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and self.status is not None and 200 <= self.status < 300

    def as_dict(self) -> dict:
        return {"url": self.url, "status": self.status, "final_url": self.final_url,
                "error": self.error, "elapsed_s": round(self.elapsed_s, 2), "n_bytes": self.n_bytes,
                "truncated": self.truncated, "path": self.path}


def http_fetch(url: str, *, dest: Path | None = None, max_bytes: int = 200_000_000,
               timeout: float = 120.0, retries: int = 2, headers: dict | None = None) -> FetchResult:
    """GET ``url``; into memory, or streamed to ``dest`` when given.  Retries
    with backoff on connection errors and 5xx; never disables TLS verification."""
    import requests  # noqa: PLC0415

    hdr = {"User-Agent": USER_AGENT, **(headers or {})}
    last = FetchResult(url=url)
    for attempt in range(retries + 1):
        t0 = _time.time()
        res = FetchResult(url=url)
        try:
            with requests.get(url, headers=hdr, timeout=timeout, stream=True, allow_redirects=True) as r:
                res.status = r.status_code
                res.final_url = r.url
                if r.status_code >= 500 and attempt < retries:
                    res.error = f"HTTP {r.status_code}"
                    last = res
                    _time.sleep(2.0 * (attempt + 1))
                    continue
                n = 0
                if dest is not None and r.status_code == 200:
                    dest = Path(dest)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    with tmp.open("wb") as fh:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            if not chunk:
                                continue
                            n += len(chunk)
                            if n > max_bytes:
                                res.truncated = True
                                break
                            fh.write(chunk)
                    if res.truncated:
                        tmp.unlink(missing_ok=True)
                        res.error = f"exceeded max_bytes={max_bytes}"
                    else:
                        tmp.replace(dest)
                        res.path = str(dest)
                else:
                    buf = bytearray()
                    for chunk in r.iter_content(chunk_size=1 << 18):
                        if not chunk:
                            continue
                        n += len(chunk)
                        if n > max_bytes:
                            res.truncated = True
                            break
                        buf.extend(chunk)
                    res.content = bytes(buf)
                res.n_bytes = n
        except Exception as exc:  # noqa: BLE001
            res.error = f"{type(exc).__name__}: {str(exc)[:300]}"
        res.elapsed_s = _time.time() - t0
        last = res
        if res.error is None or res.truncated:
            return res
        if attempt < retries:
            _time.sleep(2.0 * (attempt + 1))
    return last


# ---------------------------------------------------------------------------
# directory listings (IIS on pds-geosciences, Apache elsewhere)
# ---------------------------------------------------------------------------
@dataclass
class Entry:
    name: str
    url: str
    is_dir: bool
    size: int | None = None

    def as_dict(self) -> dict:
        return {"name": self.name, "url": self.url, "is_dir": self.is_dir, "size": self.size}


class _AnchorParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items: list[tuple[str, str, str]] = []   # (href, text, trailing text)
        self._href = None
        self._text = []
        self._trail = []
        self._after = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self._flush()
            self._href = dict(attrs).get("href")
            self._text = []
            self._after = False

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._href is not None:
            self._after = True

    def handle_data(self, data):
        if self._href is None:
            return
        if self._after:
            self._trail.append(data)
        else:
            self._text.append(data)

    def _flush(self):
        if self._href is not None:
            self.items.append((self._href, "".join(self._text).strip(), "".join(self._trail)))
        self._href, self._text, self._trail, self._after = None, [], [], False

    def close(self):
        self._flush()
        super().close()


_SIZE_RX = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*([KMG]i?B?)?(?![\w.])")


def _parse_size(text: str) -> int | None:
    # IIS puts the size BEFORE the anchor; Apache after.  Take the last plain
    # integer token in the text (dates are split by / and : so they do not match).
    best = None
    for m in _SIZE_RX.finditer(text):
        num, unit = m.group(1), (m.group(2) or "").upper()
        try:
            v = float(num)
        except ValueError:
            continue
        if unit.startswith("K"):
            v *= 1024
        elif unit.startswith("M"):
            v *= 1024**2
        elif unit.startswith("G"):
            v *= 1024**3
        elif "." in num:
            continue
        best = int(v)
    return best


def parse_listing(html: str, base_url: str) -> list[Entry]:
    """Every file/directory anchor of a directory index page, resolved to
    absolute URLs, with sizes when the page prints them."""
    p = _AnchorParser()
    p.feed(html)
    p.close()
    base_path = urlparse(base_url).path.rstrip("/") + "/"
    out: list[Entry] = []
    seen = set()
    # IIS: "date time size <A HREF=..>": the size precedes; capture per line
    lines = html.replace("\r", "").split("\n")
    pre_size: dict[str, int | None] = {}
    for ln in lines:
        m = re.search(r'([\d,]+)\s+<A HREF="([^"]+)"', ln, flags=re.I)
        if m:
            try:
                pre_size[m.group(2)] = int(m.group(1).replace(",", ""))
            except ValueError:
                pass
    for href, _text, trail in p.items:
        if not href or href.startswith(("?", "#", "mailto:", "javascript:")):
            continue
        url = urljoin(base_url, href)
        path = urlparse(url).path
        if not path.startswith(base_path) or path.rstrip("/") == base_path.rstrip("/"):
            continue      # parent / sibling links
        rel = path[len(base_path):]
        if not rel or "/" in rel.rstrip("/"):
            continue      # deeper than one level (absolute links elsewhere)
        if url in seen:
            continue
        seen.add(url)
        is_dir = rel.endswith("/") or "&lt;dir&gt;" in trail or "<dir>" in trail
        name = rel.rstrip("/")
        size = pre_size.get(href) if not is_dir else None
        if size is None and not is_dir:
            size = _parse_size(trail)
        out.append(Entry(name=name, url=url, is_dir=is_dir, size=size))
    return out


def crawl(fetch, root_url: str, *, max_depth: int = 4, max_entries: int = 6000,
          dir_regex: str | None = None, timeout: float = 60.0, log: list | None = None) -> list[Entry]:
    """Breadth-first listing of ``root_url`` to ``max_depth``; directories
    are descended only if ``dir_regex`` (when given) matches their name."""
    out: list[Entry] = []
    queue = [(root_url.rstrip("/") + "/", 0)]
    drx = re.compile(dir_regex) if dir_regex else None
    visited = set()
    while queue and len(out) < max_entries:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        res = fetch(url, timeout=timeout, max_bytes=20_000_000)
        if log is not None:
            log.append({"url": url, "depth": depth, **{k: v for k, v in res.as_dict().items() if k != "url"}})
        if not res.ok:
            continue
        try:
            html = res.content.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for e in parse_listing(html, url):
            out.append(e)
            if e.is_dir and depth + 1 <= max_depth and (drx is None or drx.search(e.name)):
                queue.append((e.url, depth + 1))
            if len(out) >= max_entries:
                break
    return out


# ---------------------------------------------------------------------------
# product classification
# ---------------------------------------------------------------------------
DEFAULT_PATTERNS: dict = {
    "pole": {
        "north": r"(?i)(^|[_\-.])(n|np|north|npole|80n|85n|75n|\d{2,4}n)([_\-.]|$)|north\s*pol",
        "south": r"(?i)(^|[_\-.])(s|sp|south|spole|80s|85s|75s|\d{2,4}s)([_\-.]|$)|south\s*pol",
    },
    "channel": {
        "tbol": r"(?i)(^|[_\-.])(tbol|tb|bol|bolo|bolometric)([_\-.]|$)|bolometric",
        "3": r"(?i)(^|[_\-.])(t3|tb3|ch3|c3|ch03|t03|b3)([_\-.]|$)|channel\s*0?3\b",
        "4": r"(?i)(^|[_\-.])(t4|tb4|ch4|c4|ch04|t04|b4)([_\-.]|$)|channel\s*0?4\b",
        "5": r"(?i)(^|[_\-.])(t5|tb5|ch5|c5|ch05|t05|b5)([_\-.]|$)|channel\s*0?5\b",
        "6": r"(?i)(^|[_\-.])(t6|tb6|ch6|c6|ch06|t06|b6)([_\-.]|$)|channel\s*0?6\b",
        "7": r"(?i)(^|[_\-.])(t7|tb7|ch7|c7|ch07|t07|b7)([_\-.]|$)|channel\s*0?7\b",
        "8": r"(?i)(^|[_\-.])(t8|tb8|ch8|c8|ch08|t08|b8)([_\-.]|$)|channel\s*0?8\b",
        "9": r"(?i)(^|[_\-.])(t9|tb9|ch9|c9|ch09|t09|b9)([_\-.]|$)|channel\s*0?9\b",
    },
    "stat": {
        "avg": r"(?i)(^|[_\-.])(avg|ave|mean|average)([_\-.]|$)|average",
        "max": r"(?i)(^|[_\-.])(max|maximum)([_\-.]|$)|maximum",
        "min": r"(?i)(^|[_\-.])(min|minimum)([_\-.]|$)|minimum",
        "count": r"(?i)(^|[_\-.])(cnt|count|num|nobs|counts)([_\-.]|$)|number of obs",
        "std": r"(?i)(^|[_\-.])(std|stdev|sd|sigma)([_\-.]|$)|standard dev",
    },
    "season": {
        "summer": r"(?i)(^|[_\-.])(sum|summer|smr)([_\-.]|$)|summer",
        "winter": r"(?i)(^|[_\-.])(win|winter|wtr)([_\-.]|$)|winter",
        "all": r"(?i)(^|[_\-.])(all|yr|year|annual|cum|cumulative|total|tot)([_\-.]|$)|all seasons|cumulative",
    },
    "resolution_m": r"(?i)(^|[_\-.])(\d{2,4})m([_\-.]|$)",
    "subsolar": r"(?i)(^|[_\-.])(ssl|sslon|lon|sl|sub)\d{2,3}([_\-.]|$)|sub[- ]?solar",
}


def classify_name(name: str, patterns: dict | None = None, extra_text: str = "") -> dict:
    """Which (pole, channel, stat, season, resolution) a product name (and,
    to fill gaps, its label text) declares.  Unknowns are ``None``."""
    pt = patterns or DEFAULT_PATTERNS
    stem = re.sub(r"\.(img|lbl|xml|tif|tiff|cub|jp2|tab|csv)$", "", name, flags=re.I)
    out: dict = {"name": name}
    for axis in ("pole", "channel", "stat", "season"):
        hit = None
        for key, rx in pt[axis].items():
            if re.search(rx, stem):
                hit = key
                break
        if hit is None and extra_text:
            for key, rx in pt[axis].items():
                if re.search(rx, extra_text):
                    hit = key
                    break
        out[axis] = hit
    m = re.search(pt["resolution_m"], stem)
    out["resolution_m"] = int(m.group(2)) if m else None
    out["subsolar_split"] = bool(re.search(pt["subsolar"], stem))
    return out


def pair_products(entries: list[Entry]) -> dict[str, dict]:
    """Group listing entries by stem: ``{stem: {"label": Entry, "image": Entry}}``."""
    groups: dict[str, dict] = {}
    for e in entries:
        if e.is_dir:
            continue
        low = e.name.lower()
        stem = re.sub(r"\.[a-z0-9]+$", "", low)
        g = groups.setdefault(stem, {"stem": stem, "label": None, "image": None, "other": []})
        if low.endswith(LABEL_SUFFIXES):
            g["label"] = e
        elif low.endswith(IMAGE_SUFFIXES):
            g["image"] = e
        else:
            g["other"].append(e)
    return groups


def select_needed(groups: dict, need: list[dict], patterns: dict | None = None,
                  label_texts: dict | None = None) -> dict:
    """For every needed (pole, season, channel, stat) pick the product whose
    classification matches; prefers no sub-solar split and the 240 m grid.

    Returns ``{key: {"stem", "class", "label", "image"} | None}`` with the
    key ``"pole/season/channel/stat"``.
    """
    classified = []
    for stem, g in groups.items():
        name = (g["image"] or g["label"]).name if (g["image"] or g["label"]) else stem
        txt = (label_texts or {}).get(stem, "")
        c = classify_name(name, patterns, extra_text=txt)
        classified.append((stem, g, c))
    out: dict = {}
    for nd in need:
        key = f"{nd['pole']}/{nd['season']}/{nd['channel']}/{nd['stat']}"
        cands = []
        for stem, g, c in classified:
            if c["pole"] == nd["pole"] and c["season"] == nd["season"] and \
               c["channel"] == nd["channel"] and c["stat"] == nd["stat"] and g["image"] is not None:
                score = (c["subsolar_split"], 0 if c["resolution_m"] in (None, 240) else 1,
                         g["image"].size or 0)
                cands.append((score, stem, g, c))
        if cands:
            cands.sort(key=lambda t: t[0])
            _s, stem, g, c = cands[0]
            out[key] = {"stem": stem, "class": c, "label": g["label"].as_dict() if g["label"] else None,
                        "image": g["image"].as_dict()}
        else:
            out[key] = None
    return out


# ---------------------------------------------------------------------------
# downloads
# ---------------------------------------------------------------------------
def download(fetch, url: str, dest: Path, *, max_bytes: int, timeout: float = 600.0,
             expected_size: int | None = None) -> FetchResult:
    """Download to ``dest`` unless a complete copy is already there."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and (expected_size is None or dest.stat().st_size == expected_size):
        r = FetchResult(url=url, status=200, path=str(dest), n_bytes=dest.stat().st_size, final_url=url)
        r.error = None
        return r
    return fetch(url, dest=dest, max_bytes=max_bytes, timeout=timeout)


# ---------------------------------------------------------------------------
# ODE REST (footprint search over the PDS holdings ODE indexes)
# ---------------------------------------------------------------------------
def ode_url(base: str, params: dict) -> str:
    from urllib.parse import urlencode  # noqa: PLC0415
    return base.rstrip("?") + "?" + urlencode({k: v for k, v in params.items() if v is not None})


def ode_query(fetch, base: str, params: dict, *, timeout: float = 90.0) -> dict:
    """One ODE REST call, returned as a dict with the raw head kept when the
    body is not JSON, so a wrong parameter name is visible in the record."""
    url = ode_url(base, {"output": "JSON", **params})
    res = fetch(url, timeout=timeout, max_bytes=30_000_000)
    rec = {"url": url, **{k: v for k, v in res.as_dict().items() if k != "url"}, "json": None,
           "head": None}
    if not res.ok:
        return rec
    try:
        rec["json"] = json.loads(res.content.decode("utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"not JSON: {exc!r}"
        rec["head"] = res.content[:1500].decode("utf-8", errors="replace")
    return rec


def ode_products(rec: dict) -> list[dict]:
    """The product list inside an ODE ``query=product`` answer (any nesting)."""
    j = rec.get("json")
    if not isinstance(j, dict):
        return []
    node = j.get("ODEResults", j)
    prods = node.get("Products", {}) if isinstance(node, dict) else {}
    if isinstance(prods, dict):
        p = prods.get("Product", [])
    else:
        p = prods
    if isinstance(p, dict):
        p = [p]
    return [x for x in p if isinstance(x, dict)]


def ode_count(rec: dict) -> int | None:
    j = rec.get("json")
    if not isinstance(j, dict):
        return None
    node = j.get("ODEResults", j)
    for k in ("Count", "count", "Total", "total"):
        if isinstance(node, dict) and k in node:
            try:
                return int(node[k])
            except (TypeError, ValueError):
                pass
    return None


def ode_footprint_params(ihid: str, iid: str, pt: str | None, lon: float, lat: float,
                         half_deg_lat: float = 0.02, limit: int = 25) -> dict:
    """Footprint-intersection query box around (lon_east, lat)."""
    cosl = max(0.02, abs(math.cos(math.radians(lat))))
    dlon = half_deg_lat / cosl
    return {"target": "moon", "query": "product", "results": "fmp", "ihid": ihid, "iid": iid,
            "pt": pt, "minlat": round(lat - half_deg_lat, 5), "maxlat": round(lat + half_deg_lat, 5),
            "westernlon": round((lon - dlon) % 360.0, 5), "easternlon": round((lon + dlon) % 360.0, 5),
            "loc": "f", "limit": limit}


# ---------------------------------------------------------------------------
# a scripted fetcher for tests
# ---------------------------------------------------------------------------
@dataclass
class ScriptedFetcher:
    """``fetch(url, ...)`` answering from a table: ``routes[url] = (status, bytes)``;
    a prefix match wins when no exact entry exists; unknown → connection error."""

    routes: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)

    def __call__(self, url: str, *, dest=None, max_bytes: int = 10**9, timeout: float = 30.0,
                 retries: int = 0, headers=None) -> FetchResult:
        self.calls.append(url)
        hit = self.routes.get(url)
        if hit is None:
            for k, v in self.routes.items():
                if k.endswith("*") and url.startswith(k[:-1]):
                    hit = v
                    break
        res = FetchResult(url=url, final_url=url)
        if hit is None:
            res.error = "ConnectionError: scripted archive has no such route"
            return res
        status, body = hit
        if callable(body):
            body = body(url)
        res.status = status
        if isinstance(body, str):
            body = body.encode("utf-8")
        res.n_bytes = len(body)
        if len(body) > max_bytes:
            res.truncated = True
            res.error = f"exceeded max_bytes={max_bytes}"
            return res
        if dest is not None and status == 200:
            dest = Path(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            res.path = str(dest)
        else:
            res.content = body
        return res


__all__ = ["DEFAULT_PATTERNS", "Entry", "FetchResult", "IMAGE_SUFFIXES", "LABEL_SUFFIXES",
           "ScriptedFetcher", "classify_name", "crawl", "download", "http_fetch", "ode_count",
           "ode_footprint_params", "ode_products", "ode_query", "ode_url", "pair_products",
           "parse_listing", "select_needed"]
