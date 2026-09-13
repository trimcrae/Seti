"""ISOTOPE acquisition: reach the Presolar Grain Database SiC table, honestly.

The table (Stephan et al. 2024, ApJS 270, 27; 20,230 SiC grains) is a
spreadsheet DOI'd through the EarthChem Library, not an API.  The
``necrofrontier-probe`` runs (``results/necrofrontier/probe.json`` on
``main``) established that ``presolar.physics.wustl.edu`` does **not** answer
the GitHub runner on either port (connect timeout at 90 s, three runs), that
the DOI asserted from memory was wrong, that ADS sits behind a WAF, and that
``ecl.earthchem.org`` answers.  So the routes, in order:

1. ``--table-url`` — a URL a human found (workflow input);
2. ``--table-path`` — a local file (or a cached download);
3. crawl the wustl landing page(s) for a link ending .xlsx/.xls/.csv/.zip
   (120 s timeout, so a host that is merely slow is not declared dead);
4. search the EarthChem Library HTML for "Presolar Grain Database" and
   follow any download link a record page offers;
5. look the DOI up at DataCite (the release is IEDA/EarthChem-minted) and
   follow the landing page's download links.

Every route writes an outcome record; if nothing is reached the acquisition
status is ``NO_DATA_REACHED`` and the ledger says what was tried.  Nothing
here fabricates rows.

Two more things live here because they are *schema*, not physics:

* table readers — csv, xlsx (``openpyxl`` if importable, else a stdlib
  ``zipfile`` + XML reader, because the venv does not carry openpyxl), and
  zip containers; a header row is *found* (best role-pattern score in the
  first rows), not assumed at row 0;
* header → role resolution by regex over a normalised header, with unit
  scales read off the raw header (``26Al/27Al (×10⁻³)``).  A role that is not
  found is ``None`` and is reported as such.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time as _time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd

USER_AGENT = "seti-isotope/1.0 (+technosignature search; S46)"

ROLES = ("grain_id", "meteorite", "grain_type", "d29si", "d30si", "c12c13", "n14n15",
         "al26al27", "ti44ti48", "d44ca")
NUMERIC_ROLES = ("d29si", "d30si", "c12c13", "n14n15", "al26al27", "ti44ti48", "d44ca")
PARTNER_ROLES = ("c12c13", "n14n15", "al26al27", "ti44ti48", "d44ca")
STATUS_OK, STATUS_NO_DATA = "OK", "NO_DATA_REACHED"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# header normalisation and role resolution
# ---------------------------------------------------------------------------
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉", "01234567890123456789")


def normalise_header(h) -> str:
    """``"err[δ(29Si/28Si)] (‰)"`` → ``"errd29si28sipermil"``."""
    s = str(h).strip().lower().translate(_SUP)
    for a, b in (("δ", "d"), ("∆", "d"), ("Δ", "d"), ("±", "pm"), ("‰", "permil"),
                 ("σ", "sig"), ("ς", "sig"), ("μ", "u"), ("µ", "u")):
        s = s.replace(a, b)
    s = re.sub(r"delta", "d", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


@dataclass
class RoleResolution:
    """Which original column carries each role, and how it was decided."""

    roles: dict = field(default_factory=dict)      # role -> original header or None
    scales: dict = field(default_factory=dict)     # role -> multiplier applied (1.0 default)
    normalised: dict = field(default_factory=dict)  # original header -> normalised
    n_columns: int = 0

    def as_dict(self) -> dict:
        return {"roles": dict(self.roles), "scales": dict(self.scales),
                "n_columns": int(self.n_columns),
                "found": sorted(k for k, v in self.roles.items() if v is not None),
                "missing": sorted(k for k, v in self.roles.items() if v is None)}


def _unit_scale(raw_header: str, role: str, unit_scales) -> float:
    h = str(raw_header).translate(_SUP).replace("⁻", "-")
    for spec in unit_scales or ():
        only = spec.get("only_roles")
        if only and role not in only:
            continue
        try:
            if re.search(str(spec["regex"]), h):
                return float(spec["scale"])
        except re.error:
            continue
    return 1.0


def resolve_roles(columns, conf: dict) -> RoleResolution:
    """Regex-resolve every role (and ``<role>_err``) against normalised headers.

    First pattern with any match wins; among its matches the shortest
    normalised header (the most canonical spelling).  Value roles exclude
    headers carrying the error marker; error roles require it.  A column is
    used for one role only.
    """
    patterns: dict = conf.get("roles") or {}
    err_rx = re.compile(str(conf.get("error_marker") or "(err|sig|unc)"))
    cols = [str(c) for c in columns]
    norm = {c: normalise_header(c) for c in cols}
    taken: set[str] = set()
    res = RoleResolution(n_columns=len(cols), normalised=dict(norm))

    def pick(role: str, want_err: bool) -> str | None:
        for pat in patterns.get(role) or ():
            try:
                rx = re.compile(str(pat))
            except re.error:
                continue
            hits = [c for c in cols if c not in taken and rx.search(norm[c])
                    and (bool(err_rx.search(norm[c])) == want_err)]
            if hits:
                hits.sort(key=lambda c: (len(norm[c]), cols.index(c)))
                return hits[0]
        return None

    for role in ROLES:
        hit = pick(role, False)
        res.roles[role] = hit
        if hit is not None:
            taken.add(hit)
            if role in NUMERIC_ROLES:
                res.scales[role] = _unit_scale(hit, role, conf.get("unit_scales"))
    for role in NUMERIC_ROLES:
        hit = pick(role, True)
        res.roles[f"{role}_err"] = hit
        if hit is not None:
            taken.add(hit)
            res.scales[f"{role}_err"] = res.scales.get(role, 1.0)
    return res


_NUM_RX = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def parse_numeric(x) -> tuple[float, float]:
    """``(value, upper_limit)``: ``"<0.001"`` → ``(nan, 0.001)``; ``"1,234"`` → ``(1234, nan)``."""
    if x is None:
        return np.nan, np.nan
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        v = float(x)
        return (v if np.isfinite(v) else np.nan), np.nan
    s = str(x).strip().replace(",", "").replace("‰", "").replace("−", "-").replace("–", "-")
    if not s or s.lower() in ("nan", "na", "n.d.", "nd", "n/a", "-", "--", "—", "none", "null"):
        return np.nan, np.nan
    limit = s.startswith("<") or s.startswith("≤")
    m = _NUM_RX.search(s)
    if not m:
        return np.nan, np.nan
    try:
        v = float(m.group(0))
    except ValueError:
        return np.nan, np.nan
    if limit:
        return np.nan, v
    return v, np.nan


def canonicalise(raw: pd.DataFrame, res: RoleResolution) -> pd.DataFrame:
    """The canonical grain table: one column per role, NaN where the role is absent.

    Numeric roles get ``<role>``, ``<role>_err`` and (for the ratios) a
    ``<role>_limit`` column for ``"<x"`` entries.  String roles are carried as
    strings.  The unit scale from the header is applied to value, error and
    limit alike.
    """
    n = len(raw)
    out = pd.DataFrame(index=range(n))
    for role in ("grain_id", "meteorite", "grain_type"):
        col = res.roles.get(role)
        if col is not None and col in raw.columns:
            out[role] = raw[col].astype(object).map(lambda v: "" if v is None or
                                                    (isinstance(v, float) and np.isnan(v))
                                                    else str(v).strip()).to_numpy()
        else:
            out[role] = ["" for _ in range(n)] if role != "grain_id" else [str(i) for i in range(n)]
    for role in NUMERIC_ROLES:
        for suffix in ("", "_err"):
            key = f"{role}{suffix}"
            col = res.roles.get(key)
            scale = float(res.scales.get(key, 1.0))
            vals = np.full(n, np.nan)
            lims = np.full(n, np.nan)
            if col is not None and col in raw.columns:
                parsed = [parse_numeric(v) for v in raw[col].tolist()]
                vals = np.array([p[0] for p in parsed], dtype=float) * scale
                lims = np.array([p[1] for p in parsed], dtype=float) * scale
            out[key] = vals
            if suffix == "":
                out[f"{role}_limit"] = lims
    # errors are magnitudes
    for role in NUMERIC_ROLES:
        out[f"{role}_err"] = np.abs(out[f"{role}_err"].to_numpy(dtype=float))
    return out


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def http_fetch(url: str, *, timeout: float = 120.0, retries: int = 2,
               max_bytes: int | None = None, method: str = "GET") -> dict:
    """One request → ``{status, url, content_type, bytes, content, error}``.

    Connection-level failures are retried with backoff; HTTP errors are not
    (a 404 is an answer).  The body is streamed and capped at ``max_bytes``.
    """
    import requests  # noqa: PLC0415  runner-only path; keeps the module importable

    last = None
    for attempt in range(max(int(retries), 1)):
        try:
            r = requests.request(method, url, timeout=timeout, stream=True,
                                 headers={"User-Agent": USER_AGENT}, allow_redirects=True)
            buf = io.BytesIO()
            n = 0
            if method != "HEAD":
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    buf.write(chunk)
                    n += len(chunk)
                    if max_bytes and n > int(max_bytes):
                        r.close()
                        return {"status": int(r.status_code), "url": str(r.url),
                                "content_type": r.headers.get("content-type"), "bytes": n,
                                "content": b"", "error": f"body exceeded {max_bytes} bytes"}
            return {"status": int(r.status_code), "url": str(r.url),
                    "content_type": r.headers.get("content-type"), "bytes": n,
                    "content": buf.getvalue(), "error": None}
        except Exception as exc:                          # noqa: BLE001
            last = exc
            print(f"[isotope/acquire] {method} {url} attempt {attempt + 1}/{retries} failed: {exc!r}")
            _time.sleep(3.0 * (attempt + 1))
    return {"status": None, "url": url, "content_type": None, "bytes": 0, "content": b"",
            "error": repr(last)}


class _LinkParser(HTMLParser):
    """Every ``href`` / ``src`` / form action on a page, with its anchor text."""

    def __init__(self):
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self.forms: list[dict] = []
        self._cur = None
        self._text: list[str] = []
        self._form = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and a.get("href"):
            self._cur = a["href"]
            self._text = []
        elif tag == "form":
            self._form = {"action": a.get("action") or "", "method": (a.get("method") or "get").lower(),
                          "inputs": []}
        elif tag == "input" and self._form is not None:
            self._form["inputs"].append({"name": a.get("name"), "type": (a.get("type") or "text").lower(),
                                         "value": a.get("value")})

    def handle_data(self, data):
        if self._cur is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._cur is not None:
            self.links.append((self._cur, " ".join(" ".join(self._text).split())))
            self._cur = None
        elif tag == "form" and self._form is not None:
            self.forms.append(self._form)
            self._form = None


def extract_links(html: str, base_url: str) -> list[tuple[str, str]]:
    """Absolute ``(href, text)`` pairs from a page (tolerant of broken HTML)."""
    p = _LinkParser()
    try:
        p.feed(html)
    except Exception:                                     # noqa: BLE001
        pass
    out = []
    seen = set()
    for href, text in p.links:
        try:
            u = urljoin(base_url, href.strip())
        except Exception:                                 # noqa: BLE001
            continue
        if u not in seen:
            seen.add(u)
            out.append((u, text))
    return out


def extract_search_forms(html: str, base_url: str) -> list[dict]:
    p = _LinkParser()
    try:
        p.feed(html)
    except Exception:                                     # noqa: BLE001
        pass
    forms = []
    for f in p.forms:
        text_inputs = [i for i in f["inputs"] if i.get("name") and i["type"] in ("text", "search")]
        if text_inputs:
            forms.append({"action": urljoin(base_url, f["action"] or base_url), "method": f["method"],
                          "field": text_inputs[0]["name"]})
    return forms


# ---------------------------------------------------------------------------
# table readers
# ---------------------------------------------------------------------------
def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref.upper())
    if not letters:
        return 0
    n = 0
    for ch in letters.group(0):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_xlsx_rows_stdlib(content: bytes) -> list[tuple[str, list[list]]]:
    """Every worksheet as rows, with only ``zipfile`` and ElementTree.

    Handles shared strings, inline strings, formula strings, booleans and
    numbers; enough for a data release.  Cell positions come from the ``r``
    attribute so sparse rows keep their columns.
    """
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
          "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    zf = zipfile.ZipFile(io.BytesIO(content))
    names = set(zf.namelist())
    shared: list[str] = []
    if "xl/sharedStrings.xml" in names:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        for si in root.findall("m:si", ns):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
    sheets: list[tuple[str, str]] = []
    if "xl/workbook.xml" in names:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = {}
        if "xl/_rels/workbook.xml.rels" in names:
            rr = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
            for rel in rr:
                rid, target = rel.get("Id"), rel.get("Target") or ""
                if rid:
                    rels[rid] = target.lstrip("/")
                    if not rels[rid].startswith("xl/"):
                        rels[rid] = "xl/" + rels[rid]
        for s in wb.iter(f"{{{ns['m']}}}sheet"):
            rid = s.get(f"{{{ns['r']}}}id")
            target = rels.get(rid)
            if target and target in names:
                sheets.append((s.get("name") or target, target))
    if not sheets:
        sheets = [(n, n) for n in sorted(names) if re.match(r"xl/worksheets/sheet\d*\.xml$", n)]
    out = []
    for sname, target in sheets:
        root = ET.fromstring(zf.read(target))
        rows: list[list] = []
        for row in root.iter(f"{{{ns['m']}}}row"):
            cells: list = []
            for c in row.findall("m:c", ns):
                idx = _col_index(c.get("r") or "")
                t = c.get("t")
                v = c.find("m:v", ns)
                val = None
                if t == "s" and v is not None and v.text is not None:
                    try:
                        val = shared[int(v.text)]
                    except (ValueError, IndexError):
                        val = v.text
                elif t == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter(f"{{{ns['m']}}}t"))
                elif t == "b" and v is not None:
                    val = bool(int(v.text or 0))
                elif v is not None and v.text is not None:
                    if t == "str":
                        val = v.text
                    else:
                        try:
                            val = float(v.text)
                        except ValueError:
                            val = v.text
                while len(cells) <= idx:
                    cells.append(None)
                cells[idx] = val
            rows.append(cells)
        out.append((sname, rows))
    return out


def read_xlsx_rows(content: bytes) -> list[tuple[str, list[list]]]:
    """openpyxl when importable (faster, broader), else the stdlib reader."""
    try:
        import openpyxl  # noqa: PLC0415
    except Exception:                                     # noqa: BLE001
        return read_xlsx_rows_stdlib(content)
    wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        out.append((ws.title, [list(r) for r in ws.iter_rows(values_only=True)]))
    return out


def read_csv_rows(content: bytes) -> list[list]:
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return []
    sample = text[:20000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    return [row for row in csv.reader(io.StringIO(text), dialect)]


def rows_from_bytes(content: bytes, name_hint: str = "") -> list[tuple[str, list[list]]]:
    """Dispatch on magic/extension: xlsx, zip container (recursed), csv/tsv/txt."""
    hint = (name_hint or "").lower().split("?")[0]
    if content[:2] == b"PK":
        try:
            zf = zipfile.ZipFile(io.BytesIO(content))
            names = zf.namelist()
        except zipfile.BadZipFile:
            return []
        if "[Content_Types].xml" in names or any(n.startswith("xl/") for n in names):
            return read_xlsx_rows(content)
        out = []
        for n in names:
            if n.endswith("/"):
                continue
            if re.search(r"(?i)\.(xlsx|csv|tsv|txt)$", n):
                out.extend((f"{n}:{s}", rows) for s, rows in rows_from_bytes(zf.read(n), n))
        return out
    if hint.endswith(".xls") and content[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        try:
            import xlrd  # noqa: PLC0415
        except Exception:                                 # noqa: BLE001
            print("[isotope/acquire] legacy .xls needs xlrd, which is not installed")
            return []
        book = xlrd.open_workbook(file_contents=content)
        return [(sh.name, [sh.row_values(i) for i in range(sh.nrows)]) for sh in book.sheets()]
    rows = read_csv_rows(content)
    return [("csv", rows)] if rows else []


def _role_score(cells, conf: dict) -> int:
    res = resolve_roles([("" if c is None else str(c)) for c in cells], conf)
    return sum(1 for v in res.roles.values() if v is not None)


def frame_from_rows(rows: list[list], conf: dict, *, max_header_scan: int = 60
                    ) -> tuple[pd.DataFrame, dict]:
    """Find the header row by role-pattern score, then build the raw frame."""
    best_i, best_s = 0, -1
    for i, r in enumerate(rows[:max_header_scan]):
        if not r or all(c is None or str(c).strip() == "" for c in r):
            continue
        s = _role_score(r, conf)
        if s > best_s:
            best_i, best_s = i, s
    header = rows[best_i] if rows else []
    width = max([len(header)] + [len(r) for r in rows[best_i + 1:]]) if rows else 0
    names: list[str] = []
    seen: dict[str, int] = {}
    for j in range(width):
        h = header[j] if j < len(header) else None
        name = str(h).strip() if h is not None and str(h).strip() else f"col{j}"
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    data = []
    for r in rows[best_i + 1:]:
        if not r or all(c is None or str(c).strip() == "" for c in r):
            continue
        data.append(list(r) + [None] * (width - len(r)))
    df = pd.DataFrame(data, columns=names) if data else pd.DataFrame(columns=names)
    return df, {"header_row_index": int(best_i), "header_role_score": int(best_s),
                "n_columns": int(width), "n_rows": int(len(df))}


def best_frame(sheets: list[tuple[str, list[list]]], conf: dict) -> tuple[pd.DataFrame, dict]:
    """Of every sheet in a workbook, the one whose header resolves most roles."""
    best = (pd.DataFrame(), {"sheet": None, "header_role_score": -1, "n_rows": 0})
    for sname, rows in sheets:
        df, meta = frame_from_rows(rows, conf)
        meta["sheet"] = sname
        key = (meta["header_role_score"], meta["n_rows"])
        if key > (best[1]["header_role_score"], best[1]["n_rows"]):
            best = (df, meta)
    return best


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@dataclass
class Acquisition:
    status: str = STATUS_NO_DATA
    route_used: str | None = None
    source: str | None = None
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    table_meta: dict = field(default_factory=dict)
    routes: list[dict] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    generated_utc: str = field(default_factory=_now)

    @property
    def n_rows(self) -> int:
        return int(len(self.table))

    def record(self, route: str, url: str | None, outcome: str, **extra) -> None:
        rec = {"route": route, "url": url, "outcome": outcome, "at_utc": _now()}
        rec.update({k: v for k, v in extra.items() if v is not None})
        self.routes.append(rec)
        self.log.append(f"{route}: {outcome}" + (f" ({url})" if url else ""))
        print(f"[isotope/acquire] {route}: {outcome}" + (f" {url}" if url else ""))

    def as_dict(self) -> dict:
        return {"status": self.status, "route_used": self.route_used, "source": self.source,
                "n_rows": self.n_rows, "n_columns": int(len(self.table.columns)),
                "columns": [str(c) for c in self.table.columns][:400],
                "table_meta": dict(self.table_meta), "routes": list(self.routes),
                "routes_tried": [r["route"] for r in self.routes],
                "log": list(self.log), "generated_utc": self.generated_utc}


def _table_from_content(content: bytes, name_hint: str, conf: dict) -> tuple[pd.DataFrame, dict]:
    sheets = rows_from_bytes(content, name_hint)
    if not sheets:
        return pd.DataFrame(), {"error": "no table could be parsed from the bytes"}
    return best_frame(sheets, conf)


def _looks_like_grain_table(df: pd.DataFrame, meta: dict, conf: dict) -> bool:
    if not len(df):
        return False
    res = resolve_roles(df.columns, conf)
    return res.roles.get("d29si") is not None or res.roles.get("c12c13") is not None


def _try_url(acq: Acquisition, route: str, url: str, conf: dict, fetch_fn) -> bool:
    a = conf.get("acquire") or {}
    r = fetch_fn(url, timeout=float(a.get("timeout_s", 120)), retries=int(a.get("retries", 2)),
                 max_bytes=int(a.get("max_bytes", 400_000_000)))
    if r.get("error") or not r.get("status") or int(r["status"]) >= 400 or not r.get("content"):
        acq.record(route, url, "FETCH_FAILED", status=r.get("status"), error=r.get("error"),
                   bytes=r.get("bytes"))
        return False
    df, meta = _table_from_content(r["content"], r.get("url") or url, conf)
    if not _looks_like_grain_table(df, meta, conf):
        acq.record(route, url, "REACHED_NO_TABLE", status=r["status"], bytes=r.get("bytes"),
                   content_type=r.get("content_type"), table_meta=meta)
        return False
    acq.table, acq.table_meta = df, meta
    acq.status, acq.route_used, acq.source = STATUS_OK, route, str(r.get("url") or url)
    acq.record(route, url, "OK", status=r["status"], bytes=r.get("bytes"),
               content_type=r.get("content_type"), n_rows=int(len(df)), table_meta=meta)
    return True


def _route_local(acq: Acquisition, path: str | Path, conf: dict) -> bool:
    p = Path(path)
    if not p.exists():
        acq.record("local_path", str(p), "NOT_FOUND")
        return False
    try:
        content = p.read_bytes()
    except Exception as exc:                              # noqa: BLE001
        acq.record("local_path", str(p), "READ_FAILED", error=repr(exc))
        return False
    df, meta = _table_from_content(content, p.name, conf)
    if not _looks_like_grain_table(df, meta, conf):
        acq.record("local_path", str(p), "REACHED_NO_TABLE", table_meta=meta, bytes=len(content))
        return False
    acq.table, acq.table_meta = df, meta
    acq.status, acq.route_used, acq.source = STATUS_OK, "local_path", str(p)
    acq.record("local_path", str(p), "OK", n_rows=int(len(df)), table_meta=meta, bytes=len(content))
    return True


def _route_crawl(acq: Acquisition, conf: dict, fetch_fn) -> bool:
    a = conf.get("acquire") or {}
    suffix = re.compile(str(a.get("link_suffix_regex") or r"(?i)\.(xlsx|xls|csv|zip)$"))
    for url in a.get("crawl_urls") or ():
        r = fetch_fn(url, timeout=float(a.get("timeout_s", 120)), retries=int(a.get("retries", 2)),
                     max_bytes=20_000_000)
        if r.get("error") or not r.get("status") or int(r["status"]) >= 400:
            acq.record("wustl_crawl", url, "FETCH_FAILED", status=r.get("status"), error=r.get("error"))
            continue
        html = r["content"].decode("utf-8", "replace")
        links = [(u, t) for u, t in extract_links(html, r.get("url") or url) if suffix.search(u)]
        acq.record("wustl_crawl", url, "REACHED", status=r["status"], bytes=r.get("bytes"),
                   n_table_links=len(links), table_links=[u for u, _ in links][:20])
        for u, _t in links:
            if _try_url(acq, "wustl_crawl_link", u, conf, fetch_fn):
                return True
    return False


def _route_ecl(acq: Acquisition, conf: dict, fetch_fn) -> bool:
    a = conf.get("acquire") or {}
    timeout, retries = float(a.get("timeout_s", 120)), int(a.get("retries", 2))
    query = str(a.get("ecl_query") or "Presolar Grain Database")
    rec_rx = re.compile(str(a.get("ecl_record_regex") or r"(?i)view\.php\?"))
    dl_rx = re.compile(str(a.get("ecl_download_regex") or r"(?i)(download|\.xlsx|\.csv|\.zip)"))
    search_urls = list(a.get("ecl_search_urls") or ())
    # discover the site's own search form from the home page and add it first
    home = str(a.get("ecl_home") or "https://ecl.earthchem.org/home.php")
    r = fetch_fn(home, timeout=timeout, retries=retries, max_bytes=5_000_000)
    if r.get("error") or not r.get("status") or int(r["status"]) >= 400:
        acq.record("ecl_home", home, "FETCH_FAILED", status=r.get("status"), error=r.get("error"))
    else:
        html = r["content"].decode("utf-8", "replace")
        forms = extract_search_forms(html, r.get("url") or home)
        from urllib.parse import quote_plus  # noqa: PLC0415
        for f in forms:
            if f["method"] == "get":
                sep = "&" if "?" in f["action"] else "?"
                search_urls.insert(0, f"{f['action']}{sep}{f['field']}={quote_plus(query)}")
        acq.record("ecl_home", home, "REACHED", status=r["status"], n_forms=len(forms),
                   forms=forms[:5])
    seen_records: list[str] = []
    for su in dict.fromkeys(search_urls):
        r = fetch_fn(su, timeout=timeout, retries=retries, max_bytes=10_000_000)
        if r.get("error") or not r.get("status") or int(r["status"]) >= 400:
            acq.record("ecl_search", su, "FETCH_FAILED", status=r.get("status"), error=r.get("error"))
            continue
        html = r["content"].decode("utf-8", "replace")
        links = extract_links(html, r.get("url") or su)
        hits = [(u, t) for u, t in links
                if re.search(r"(?i)presolar", t or "") or re.search(r"(?i)presolar", u)]
        records = [u for u, _ in hits if rec_rx.search(u)] or [u for u, _ in hits]
        downloads = [u for u, t in links if dl_rx.search(u) or dl_rx.search(t or "")]
        acq.record("ecl_search", su, "REACHED", status=r["status"], n_links=len(links),
                   n_presolar_hits=len(hits), presolar_hits=[u for u, _ in hits][:20],
                   n_download_links=len(downloads))
        for u in downloads:
            if re.search(r"(?i)presolar", u) and _try_url(acq, "ecl_download", u, conf, fetch_fn):
                return True
        for u in records[:10]:
            if u in seen_records:
                continue
            seen_records.append(u)
            rr = fetch_fn(u, timeout=timeout, retries=retries, max_bytes=10_000_000)
            if rr.get("error") or not rr.get("status") or int(rr["status"]) >= 400:
                acq.record("ecl_record", u, "FETCH_FAILED", status=rr.get("status"), error=rr.get("error"))
                continue
            rhtml = rr["content"].decode("utf-8", "replace")
            dls = [x for x, t in extract_links(rhtml, rr.get("url") or u)
                   if dl_rx.search(x) or dl_rx.search(t or "")]
            acq.record("ecl_record", u, "REACHED", status=rr["status"], n_download_links=len(dls),
                       download_links=dls[:20])
            for x in dls:
                if _try_url(acq, "ecl_download", x, conf, fetch_fn):
                    return True
    return False


def _route_datacite(acq: Acquisition, conf: dict, fetch_fn) -> bool:
    a = conf.get("acquire") or {}
    timeout, retries = float(a.get("timeout_s", 120)), int(a.get("retries", 2))
    api = str(a.get("datacite_api") or "https://api.datacite.org/dois")
    dl_rx = re.compile(str(a.get("ecl_download_regex") or r"(?i)(download|\.xlsx|\.csv|\.zip)"))
    from urllib.parse import quote  # noqa: PLC0415
    dois: list[tuple[str, str, str]] = []
    for q in a.get("datacite_queries") or ():
        url = f"{api}?query={quote(str(q))}&page[size]={int(a.get('datacite_page_size', 25))}"
        r = fetch_fn(url, timeout=timeout, retries=retries, max_bytes=20_000_000)
        if r.get("error") or not r.get("status") or int(r["status"]) >= 400:
            acq.record("datacite_search", url, "FETCH_FAILED", status=r.get("status"), error=r.get("error"))
            continue
        try:
            js = json.loads(r["content"].decode("utf-8", "replace"))
        except Exception as exc:                          # noqa: BLE001
            acq.record("datacite_search", url, "REACHED_NOT_JSON", status=r["status"], error=repr(exc))
            continue
        found = []
        for d in js.get("data") or []:
            at = d.get("attributes") or {}
            titles = " | ".join(str(t.get("title", "")) for t in (at.get("titles") or []))
            if re.search(r"(?i)presolar", titles):
                found.append((str(at.get("doi") or d.get("id") or ""), str(at.get("url") or ""), titles))
        acq.record("datacite_search", url, "REACHED", status=r["status"], n_records=len(js.get("data") or []),
                   n_presolar=len(found), presolar=[{"doi": x, "url": y, "title": z[:200]} for x, y, z in found][:20])
        dois.extend(found)
    tried: set[str] = set()
    for doi, landing, _title in dois:
        for u in (landing, f"{a.get('doi_resolver', 'https://doi.org/')}{doi}" if doi else ""):
            if not u or u in tried:
                continue
            tried.add(u)
            r = fetch_fn(u, timeout=timeout, retries=retries, max_bytes=10_000_000)
            if r.get("error") or not r.get("status") or int(r["status"]) >= 400:
                acq.record("datacite_landing", u, "FETCH_FAILED", status=r.get("status"), error=r.get("error"))
                continue
            ctype = str(r.get("content_type") or "")
            if re.search(r"(?i)spreadsheet|excel|csv|zip|octet", ctype):
                df, meta = _table_from_content(r["content"], r.get("url") or u, conf)
                if _looks_like_grain_table(df, meta, conf):
                    acq.table, acq.table_meta = df, meta
                    acq.status, acq.route_used, acq.source = STATUS_OK, "datacite_landing", str(r.get("url") or u)
                    acq.record("datacite_landing", u, "OK", status=r["status"], n_rows=int(len(df)), table_meta=meta)
                    return True
            html = r["content"].decode("utf-8", "replace")
            dls = [x for x, t in extract_links(html, r.get("url") or u) if dl_rx.search(x) or dl_rx.search(t or "")]
            acq.record("datacite_landing", u, "REACHED", status=r["status"], doi=doi,
                       n_download_links=len(dls), download_links=dls[:20])
            for x in dls:
                if _try_url(acq, "datacite_download", x, conf, fetch_fn):
                    return True
    return False


def acquire_table(conf: dict, *, table_url: str | None = None, table_path: str | Path | None = None,
                  fetch_fn=None, cache_path: str | Path | None = None,
                  routes: tuple[str, ...] = ("url", "path", "crawl", "ecl", "datacite")) -> Acquisition:
    """Try every route in order; the first table with a Si or C role wins.

    ``fetch_fn(url, timeout=, retries=, max_bytes=) -> dict`` is the injection
    point for the offline suite.  A table reached over the network is written
    to ``cache_path`` (raw csv) so later stages and re-runs need not refetch.
    """
    fetch_fn = fetch_fn or http_fetch
    acq = Acquisition()
    for route in routes:
        if route == "url":
            if table_url:
                if _try_url(acq, "human_url", str(table_url), conf, fetch_fn):
                    break
            else:
                acq.record("human_url", None, "SKIPPED_NO_URL")
        elif route == "path":
            if table_path:
                if _route_local(acq, table_path, conf):
                    break
            elif cache_path and Path(cache_path).exists():
                if _route_local(acq, cache_path, conf):
                    acq.route_used = "cache"
                    break
            else:
                acq.record("local_path", None, "SKIPPED_NO_PATH")
        elif route == "crawl":
            if _route_crawl(acq, conf, fetch_fn):
                break
        elif route == "ecl":
            if _route_ecl(acq, conf, fetch_fn):
                break
        elif route == "datacite":
            if _route_datacite(acq, conf, fetch_fn):
                break
    if acq.status == STATUS_OK and cache_path and acq.route_used not in ("local_path", "cache"):
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            acq.table.to_csv(cache_path, index=False)
            acq.log.append(f"cached raw table to {cache_path}")
        except Exception as exc:                          # noqa: BLE001
            acq.log.append(f"cache write failed: {exc!r}")
    if acq.status != STATUS_OK:
        acq.log.append("NO_DATA_REACHED: no route produced a grain table; routes tried: "
                       + ", ".join(dict.fromkeys(r["route"] for r in acq.routes)))
    return acq


def probe_routes(conf: dict, *, fetch_fn=None, table_url: str | None = None) -> dict:
    """Reachability only (no download beyond a few MB): every URL a route would touch."""
    fetch_fn = fetch_fn or http_fetch
    a = conf.get("acquire") or {}
    urls = []
    if table_url:
        urls.append(("human_url", str(table_url)))
    urls += [("wustl_crawl", u) for u in a.get("crawl_urls") or ()]
    urls.append(("ecl_home", str(a.get("ecl_home") or "https://ecl.earthchem.org/home.php")))
    urls += [("ecl_search", u) for u in a.get("ecl_search_urls") or ()]
    urls.append(("datacite_search", str(a.get("datacite_api") or "https://api.datacite.org/dois")
                 + "?query=titles.title:%22Presolar%20Grain%20Database%22&page[size]=5"))
    out = []
    for route, u in urls:
        r = fetch_fn(u, timeout=float(a.get("timeout_s", 120)), retries=1, max_bytes=3_000_000)
        reached = bool(r.get("status")) and not r.get("error") and int(r["status"]) < 400
        out.append({"route": route, "url": u, "reached": reached, "status": r.get("status"),
                    "bytes": r.get("bytes"), "content_type": r.get("content_type"),
                    "error": r.get("error"), "final_url": r.get("url")})
        print(f"[isotope/probe] {route} {u}: {'REACHED' if reached else 'NOT_REACHED'} "
              f"status={r.get('status')} bytes={r.get('bytes')}")
    return {"stage": "probe", "generated_utc": _now(), "endpoints": out,
            "n_reached": int(sum(1 for o in out if o["reached"])), "n_probed": len(out)}


__all__ = [
    "NUMERIC_ROLES", "PARTNER_ROLES", "ROLES", "Acquisition", "RoleResolution", "acquire_table",
    "best_frame", "canonicalise", "extract_links", "extract_search_forms", "frame_from_rows",
    "http_fetch", "normalise_header", "parse_numeric", "probe_routes", "read_csv_rows",
    "read_xlsx_rows", "read_xlsx_rows_stdlib", "resolve_roles", "rows_from_bytes",
]
