"""Published narrowband hit tables straight out of the papers' e-print source.

Why this module exists
----------------------
The RELAY re-cut needs the *events* the Breakthrough Listen and allied
narrowband surveys actually reported: a frequency, a drift rate, and the star
they came from.  The VizieR route (``acquire.discover_hit_tables``) probes the
catalogue ids for those papers and comes back with **target** tables only --- the
observed-star lists are deposited, the event lists are not (run 35738937745:
73 tables discovered, 0 of kind ``hits``).  The event tables exist only inside
the papers, as ``deluxetable``/``longtable`` environments or as AAS
machine-readable tables shipped in the same e-print tarball.

So this module fetches the e-print source from arXiv and parses the tables out
of it.  Three disciplines make that safe to build a search on:

* **The paper is verified, not assumed.**  An arXiv id hint is never used on
  its own: the Atom metadata is fetched and the returned title must contain
  every phrase the seed declares.  A mismatch is recorded as
  ``TITLE_MISMATCH`` with both strings verbatim and the seed is dropped, so a
  mistyped id cannot quietly substitute another paper's numbers.
* **Provenance travels with every row.**  Each parsed row carries the arXiv id,
  the resolved title, the source file inside the tarball, the table's caption
  and the header text of the columns it was read from.  Any candidate can be
  traced back to a line of LaTeX.
* **A table is a hit table only if it proves it.**  Both a frequency column and
  a drift-rate column must resolve, and at least one row of each must parse as
  a finite number.  Everything else is recorded with its header and ignored.

Nothing here opens a socket by itself: ``get_fn(url) -> bytes`` and
``get_text_fn(url) -> str`` are injected, and the offline suite drives the whole
parser off fixture bytes.
"""

from __future__ import annotations

import gzip
import io
import re
import tarfile
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..metronome.acquire import STATUS_FAILED, STATUS_OK, STATUS_ZERO, AcquisitionLog

ARXIV_API = "https://export.arxiv.org/api/query"
EPRINT_URLS = ("https://arxiv.org/e-print/{id}", "https://export.arxiv.org/e-print/{id}")
ATOM = "{http://www.w3.org/2005/Atom}"

STATUS_TITLE_MISMATCH = "TITLE_MISMATCH"
STATUS_UNRESOLVED = "UNRESOLVED"
STATUS_NO_SOURCE = "NO_SOURCE"
STATUS_NO_TABLES = "NO_TABLES"

MAX_EPRINT_BYTES = 60_000_000
TEXT_SUFFIXES = (".tex", ".txt", ".mrt", ".dat", ".tab", ".table")

# ---------------------------------------------------------------------------
# LaTeX table extraction
# ---------------------------------------------------------------------------
TABLE_ENVS = ("deluxetable*", "deluxetable", "longdeluxetable*", "longdeluxetable",
              "splitdeluxetable*", "splitdeluxetable", "longtable*", "longtable",
              "table*", "table", "sidewaystable*", "sidewaystable")

_ROW_SPLIT = re.compile(r"\\\\(?:\s*\[[^\]]*\])?")
_COMMENT = re.compile(r"(?<!\\)%.*")
# macros whose argument is CONTENT (kept) and macros whose argument is
# APPARATUS (dropped: a footnote marker glued to a number would corrupt it).
_MACRO_KEEP = re.compile(r"\\(?:mathrm|mathit|math|text|textrm|textit|textbf|textsf|texttt|"
                         r"emph|colhead|nocolhead|multicolumn\s*\{[^}]*\}\s*\{[^}]*\})\s*\{")
_MACRO_DROP = re.compile(r"\\(?:tablenotemark|tablenotetext|tablerefs|tablecomments|footnote|"
                         r"label|ref|eqref|citep|citet|citealt|cite|phantom|hphantom|vphantom|"
                         r"textsuperscript|textsubscript)\s*\{")
_SIMPLE_MACRO = re.compile(r"\\[A-Za-z@]+\*?")
_WS = re.compile(r"\s+")


def _strip_comments(text: str) -> str:
    return "\n".join(_COMMENT.sub("", ln) for ln in text.splitlines())


def _balanced(text: str, start: int) -> tuple[str, int]:
    """Contents of the brace group that opens at ``text[start] == '{'``."""
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif c == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    return text[start + 1:], len(text)


def _macro_arg(text: str, macro: str) -> str | None:
    m = re.search(re.escape("\\" + macro) + r"\s*\{", text)
    if not m:
        return None
    body, _ = _balanced(text, m.end() - 1)
    return clean_cell(body)


def clean_cell(cell: str) -> str:
    """One table cell as plain text: macros dropped, arguments kept, math flattened."""
    s = cell.replace("\n", " ")
    for pattern, keep in ((_MACRO_DROP, False), (_MACRO_KEEP, True)):
        while True:
            m = pattern.search(s)
            if not m:
                break
            inner, end = _balanced(s, s.index("{", m.end() - 1))
            s = s[:m.start()] + (inner if keep else "") + s[end:]
    s = s.replace("$-$", "-").replace("$+$", "+").replace("\\pm", " +/- ")
    s = s.replace("$\\ldots$", "").replace("\\ldots", "").replace("\\dots", "")
    s = s.replace("\\times10", "e").replace("\\times 10", "e")
    s = s.replace("^{", "^").replace("_{", "_")
    s = s.replace("$", " ").replace("~", " ").replace("\\&", "&")
    s = _SIMPLE_MACRO.sub(" ", s)
    s = s.replace("{", " ").replace("}", " ").replace("\\", " ")
    return _WS.sub(" ", s).strip()


def _split_cells(row: str) -> list[str]:
    """Split a LaTeX row on unescaped ``&`` outside brace groups."""
    out, buf, depth = [], [], 0
    i = 0
    while i < len(row):
        c = row[i]
        if c == "\\" and i + 1 < len(row):
            buf.append(row[i:i + 2])
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if c == "&" and depth <= 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    out.append("".join(buf))
    return out


def _environments(text: str) -> list[tuple[str, str]]:
    """``(env, body)`` for every table-like environment, outermost first."""
    found = []
    for env in TABLE_ENVS:
        pat = re.compile(r"\\begin\{" + re.escape(env) + r"\}")
        end = "\\end{" + env + "}"
        pos = 0
        while True:
            m = pat.search(text, pos)
            if not m:
                break
            e = text.find(end, m.end())
            if e < 0:
                break
            found.append((env, m.start(), text[m.end():e]))
            pos = e + len(end)
    found.sort(key=lambda t: t[1])
    # drop an environment fully contained in an earlier one (table wrapping tabular)
    keep: list[tuple[str, str]] = []
    spans: list[tuple[int, int]] = []
    for env, start, body in found:
        stop = start + len(body)
        if any(s <= start and stop <= e for s, e in spans):
            continue
        spans.append((start, stop))
        keep.append((env, body))
    return keep


def _header_from_tablehead(body: str) -> list[str]:
    m = re.search(r"\\tablehead\s*\{", body)
    if not m:
        return []
    head, _ = _balanced(body, m.end() - 1)
    cols = []
    for cm in re.finditer(r"\\(?:col|nocol|two|multicol)head\s*\{", head):
        inner, _ = _balanced(head, cm.end() - 1)
        cols.append(clean_cell(inner))
    if cols:
        return cols
    return [clean_cell(c) for c in _split_cells(head)]


def _body_rows(body: str) -> list[str]:
    m = re.search(r"\\startdata", body)
    if m:
        e = body.find("\\enddata", m.end())
        return _rows(body[m.end():e if e > 0 else len(body)])
    tm = re.search(r"\\begin\{tabular[*x]?\}", body)
    if tm:
        # skip the column spec argument
        b = body.find("{", tm.end() - 1)
        _, after = _balanced(body, b) if b >= 0 else ("", tm.end())
        e = body.find("\\end{tabular", after)
        return _rows(body[after:e if e > 0 else len(body)])
    return _rows(body)


def _rows(chunk: str) -> list[str]:
    out = []
    for raw in _ROW_SPLIT.split(chunk):
        r = raw.strip()
        r = re.sub(r"\\(?:hline|toprule|midrule|bottomrule|tableline|cline\{[^}]*\}|"
                   r"noalign\{[^}]*\}|cutinhead\{[^}]*\}|sidehead\{[^}]*\})", " ", r)
        if r.strip(" &"):
            out.append(r)
    return out


@dataclass
class ParsedTable:
    source: str
    env: str
    caption: str
    header: list[str]
    rows: list[list[str]]

    def as_report(self) -> dict:
        return {"source": self.source, "env": self.env, "caption": self.caption[:300],
                "header": self.header, "n_rows": len(self.rows)}


def latex_tables(text: str, source: str) -> list[ParsedTable]:
    """Every table-like environment in one .tex file, header and rows as text."""
    text = _strip_comments(text)
    out = []
    for env, body in _environments(text):
        header = _header_from_tablehead(body)
        rows = [[clean_cell(c) for c in _split_cells(r)] for r in _body_rows(body)]
        if not header and rows:
            # a plain tabular: the first row that is all non-numeric is the header
            first = rows[0]
            if first and not any(_num(c) == _num(c) for c in first):
                header, rows = first, rows[1:]
        caption = _macro_arg(body, "tablecaption") or _macro_arg(body, "caption") or ""
        rows = [r for r in rows if any(c for c in r)]
        if header or rows:
            out.append(ParsedTable(source, env, caption, header, rows))
    return out


# ---------------------------------------------------------------------------
# AAS machine-readable tables
# ---------------------------------------------------------------------------
_MRT_BYTES = re.compile(r"^\s*(\d+)\s*-?\s*(\d*)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$")


def mrt_table(text: str, source: str) -> ParsedTable | None:
    """An AAS machine-readable table: byte ranges from the header, then the data."""
    lines = text.splitlines()
    rule = [i for i, ln in enumerate(lines) if ln.startswith("---") and len(ln.strip()) > 10]
    if len(rule) < 3:
        return None
    spec_lines = lines[rule[-2] + 1:rule[-1]]
    cols: list[tuple[int, int, str, str]] = []
    for ln in spec_lines:
        m = _MRT_BYTES.match(ln)
        if not m:
            continue
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        unit = m.group(4)
        label = m.group(5)
        cols.append((lo, hi, label, unit))
    if not cols:
        return None
    title = next((ln.strip() for ln in lines[:rule[0]] if ln.strip()), "")
    header = [f"{lab} ({u})" if u not in ("---", "") else lab for _, _, lab, u in cols]
    rows = []
    for ln in lines[rule[-1] + 1:]:
        if not ln.strip():
            continue
        rows.append([ln[lo - 1:hi].strip() for lo, hi, _, _ in cols])
    if not rows:
        return None
    return ParsedTable(source, "mrt", title, header, rows)


# ---------------------------------------------------------------------------
# column roles, from human-readable headers
# ---------------------------------------------------------------------------
ROLE_RULES: list[tuple[str, list[str], list[str]]] = [
    # role, must-contain (any), must-not-contain
    ("drift", ["drift", "dfdt", "df/dt", "fdot", "f dot", "chirp"], ["rate limit", "eirp"]),
    ("freq_mhz", ["frequency", "freq", "nu obs", "f obs", "centre freq", "center freq"],
     ["resolution", "range", "coverage", "band edge", "sampling"]),
    ("snr", ["snr", "s/n", "signal-to-noise", "signal to noise"], []),
    ("mjd", ["mjd", "utc date", "obs date", "date-obs", "epoch", "observation date"], []),
    ("target", ["target", "source name", "source", "star", "object", "name", "hip", "tic",
                "gj ", "identifier"], ["source count", "namespace"]),
    ("ra", ["r.a.", "ra (", "ra(", "raj2000", "right ascension", "_ra"], []),
    ("dec", ["decl", "dec (", "dec(", "dej2000", "declination"], []),
    ("telescope", ["telescope", "observatory", "facility", "instrument", "receiver"], []),
    ("verdict", ["classification", "rfi", "status", "flag", "note", "comment", "disposition"], []),
    ("drift_unit", [], []),
]
_UNIT = re.compile(r"[\(\[]\s*([^)\]]{1,24})\s*[\)\]]")


def _norm_header(h: str) -> str:
    return _WS.sub(" ", h.lower().replace("\u2212", "-")).strip()


def header_roles(header) -> dict[str, int]:
    """Role -> column index, resolved on human-readable header text."""
    roles: dict[str, int] = {}
    norm = [_norm_header(h) for h in header]
    for role, wants, nots in ROLE_RULES:
        if not wants:
            continue
        for i, h in enumerate(norm):
            if i in roles.values() and role != "target":
                continue
            if any(w in h for w in wants) and not any(n in h for n in nots):
                roles.setdefault(role, i)
                break
    return roles


def header_unit(header_text: str) -> str | None:
    m = _UNIT.search(header_text or "")
    return m.group(1).strip().lower() if m else None


_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?")


def _num(cell) -> float:
    """First number in a cell, or NaN.  ``123 +/- 4`` -> 123."""
    if cell is None:
        return float("nan")
    s = str(cell).replace("\u2212", "-").replace(",", "")
    s = s.split("+/-")[0]
    m = _NUM.search(s)
    if not m:
        return float("nan")
    try:
        return float(m.group(0).replace("D", "E").replace("d", "e"))
    except ValueError:
        return float("nan")


def freq_to_mhz(values: np.ndarray, unit: str | None) -> np.ndarray:
    """Frequency column to MHz using the header's unit, or the magnitude if it has none."""
    v = np.asarray(values, float)
    u = (unit or "").lower()
    if "ghz" in u:
        return v * 1e3
    if "khz" in u:
        return v * 1e-3
    if "mhz" in u:
        return v
    if re.fullmatch(r"\s*hz\s*", u or ""):
        return v * 1e-6
    # no usable unit: L/S/C/X band numbers are MHz in the 300-30000 range,
    # GHz below 100, Hz above 1e7.  Recorded as freq_unit_assumed.
    return np.where(v < 100, v * 1e3, np.where(v > 1e7, v * 1e-6, v))


def drift_to_hz_s(values: np.ndarray, unit: str | None) -> np.ndarray:
    v = np.asarray(values, float)
    u = (unit or "").lower()
    if "nhz" in u:
        return v * 1e-9
    if "mhz" in u and "/" in u:
        return v * 1e6
    if "khz" in u:
        return v * 1e3
    return v


def table_hits(t: ParsedTable, meta: dict, *, max_rows: int = 20000) -> tuple[pd.DataFrame, dict]:
    """Hit rows from one parsed table, or an empty frame with the reason why not."""
    rep = t.as_report() | {"seed": meta.get("seed"), "arxiv_id": meta.get("arxiv_id")}
    roles = header_roles(t.header)
    rep["roles"] = dict(roles)
    if "freq_mhz" not in roles or "drift" not in roles:
        rep["kind"] = "other" if "target" not in roles else "targets"
        return pd.DataFrame(), rep
    ncol = len(t.header)
    rows = [r for r in t.rows if len(r) == ncol][:max_rows]
    rep["n_rows_shaped"] = len(rows)
    if not rows:
        rep["kind"] = "other"
        rep["reason"] = "no row matched the header width"
        return pd.DataFrame(), rep
    col = {role: [r[i] for r in rows] for role, i in roles.items()}
    f_unit = header_unit(t.header[roles["freq_mhz"]])
    d_unit = header_unit(t.header[roles["drift"]])
    freq = freq_to_mhz(np.array([_num(c) for c in col["freq_mhz"]]), f_unit)
    drift = drift_to_hz_s(np.array([_num(c) for c in col["drift"]]), d_unit)
    ok = np.isfinite(freq) & np.isfinite(drift)
    rep["n_rows_numeric"] = int(ok.sum())
    rep["freq_unit"] = f_unit
    rep["freq_unit_assumed"] = f_unit is None
    rep["drift_unit"] = d_unit
    if not ok.any():
        rep["kind"] = "other"
        rep["reason"] = "frequency and drift columns present but no row parsed as numbers"
        return pd.DataFrame(), rep
    rep["kind"] = "hits"
    # `drift` is the raw-role name acquire.standardise_hits reads; `drift_hz_s`
    # is the converted one.  Both are written so either path gives the same
    # number instead of a silent column of NaN.
    out = pd.DataFrame({"freq_mhz": freq[ok], "drift": drift[ok], "drift_hz_s": drift[ok]})
    idx = np.flatnonzero(ok)
    for role in ("target", "snr", "mjd", "ra", "dec", "telescope", "verdict"):
        if role not in roles:
            continue
        vals = [col[role][i] for i in idx]
        # numeric roles are parsed here; `mjd` stays a string because a table may
        # print a calendar date and acquire.standardise_hits knows both forms.
        out[role] = [_num(v) for v in vals] if role in ("snr", "ra", "dec") else vals
    out["table"] = f"arXiv:{meta.get('arxiv_id')}:{t.source}#{t.env}"
    out["seed"] = meta.get("seed")
    out["catalogue_telescope"] = meta.get("telescope")
    out["catalogue_band"] = meta.get("band")
    out["paper_title"] = meta.get("title")
    out["caption"] = t.caption[:200]
    out["freq_header"] = t.header[roles["freq_mhz"]]
    out["drift_header"] = t.header[roles["drift"]]
    return out.reset_index(drop=True), rep


# ---------------------------------------------------------------------------
# arXiv transports
# ---------------------------------------------------------------------------
def default_get_bytes(url: str, *, timeout: float = 120.0) -> bytes:
    import requests  # noqa: PLC0415

    r = requests.get(url, timeout=timeout, stream=True,
                     headers={"User-Agent": "seti-relay/1.0 (technosignature research; "
                                            "contact via github.com/trimcrae/Seti)"})
    r.raise_for_status()
    buf = io.BytesIO()
    for chunk in r.iter_content(65536):
        buf.write(chunk)
        if buf.tell() > MAX_EPRINT_BYTES:
            raise ValueError(f"e-print larger than {MAX_EPRINT_BYTES} bytes")
    return buf.getvalue()


def default_get_text(url: str, *, timeout: float = 120.0) -> str:
    import requests  # noqa: PLC0415

    r = requests.get(url, timeout=timeout,
                     headers={"User-Agent": "seti-relay/1.0 (technosignature research)"})
    r.raise_for_status()
    return r.text


def _api_entries(xml_text: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out = []
    for e in root.findall(f"{ATOM}entry"):
        idt = (e.findtext(f"{ATOM}id") or "").strip()
        title = _WS.sub(" ", (e.findtext(f"{ATOM}title") or "").strip())
        published = (e.findtext(f"{ATOM}published") or "").strip()
        m = re.search(r"abs/([^v\s]+)(v\d+)?$", idt)
        out.append({"arxiv_id": m.group(1) if m else idt, "title": title,
                    "published": published})
    return out


def search_url(query: str, max_results: int = 12, *, start: int = 0) -> str:
    """The arXiv query URL, with the query string URL-encoded exactly once."""
    q = urllib.parse.quote(query, safe=":+")
    return f"{ARXIV_API}?search_query={q}&start={int(start)}&max_results={int(max_results)}"


def _matches(title: str, wants) -> bool:
    t = title.lower().replace("\u2212", "-")
    return all(w.lower() in t for w in (wants or ()))


def resolve_paper(key: str, spec: dict, *, get_text_fn, log: AcquisitionLog,
                  timeout: float = 60.0) -> dict:
    """The arXiv id for one seed, with its title verified against the seed's phrases."""
    wants = spec.get("title_contains") or []
    rep = {"seed": key, "title_contains": wants}
    hint = spec.get("arxiv_id")
    if hint:
        url = f"{ARXIV_API}?id_list={hint}&max_results=1"
        try:
            entries = _api_entries(get_text_fn(url, timeout=timeout))
            log.record("arxiv_id_lookup", url, rows=len(entries))
        except Exception as exc:                          # noqa: BLE001
            entries = []
            log.record("arxiv_id_lookup", url, error=repr(exc))
            rep["id_lookup_error"] = repr(exc)[:300]
        if entries:
            t = entries[0]
            rep |= {"arxiv_id": t["arxiv_id"], "title": t["title"],
                    "published": t["published"], "route": "id_hint"}
            if _matches(t["title"], wants):
                rep["status"] = STATUS_OK
                return rep
            rep["status"] = STATUS_TITLE_MISMATCH
            rep["note"] = "the id hint resolved to a different paper; falling back to search"
    query = spec.get("query")
    if not query:
        rep.setdefault("status", STATUS_UNRESOLVED)
        return rep
    url = search_url(query, int(spec.get("max_results", 12)))
    try:
        entries = _api_entries(get_text_fn(url, timeout=timeout))
        log.record("arxiv_search", url, rows=len(entries))
    except Exception as exc:                              # noqa: BLE001
        log.record("arxiv_search", url, error=repr(exc))
        rep["status"] = STATUS_FAILED
        rep["error"] = repr(exc)[:300]
        return rep
    rep["n_search_results"] = len(entries)
    for e in entries:
        if _matches(e["title"], wants):
            return rep | {"arxiv_id": e["arxiv_id"], "title": e["title"],
                          "published": e["published"], "route": "search", "status": STATUS_OK}
    rep["status"] = STATUS_UNRESOLVED
    rep["titles_seen"] = [e["title"] for e in entries[:8]]
    return rep


def eprint_sources(arxiv_id: str, *, get_fn, log: AcquisitionLog,
                   timeout: float = 120.0) -> tuple[dict[str, str], dict]:
    """Every text file in an e-print package, keyed by its name in the tarball."""
    rep = {"arxiv_id": arxiv_id, "urls": []}
    blob = None
    for tmpl in EPRINT_URLS:
        url = tmpl.format(id=arxiv_id)
        try:
            blob = get_fn(url, timeout=timeout)
            log.record("arxiv_eprint", url, rows=len(blob or b""))
            rep["urls"].append({"url": url, "status": STATUS_OK, "bytes": len(blob or b"")})
            break
        except Exception as exc:                          # noqa: BLE001
            log.record("arxiv_eprint", url, error=repr(exc))
            rep["urls"].append({"url": url, "status": STATUS_FAILED, "error": repr(exc)[:300]})
    if not blob:
        rep["status"] = STATUS_NO_SOURCE
        return {}, rep
    files = unpack_sources(blob)
    rep["status"] = STATUS_OK if files else STATUS_NO_SOURCE
    rep["n_files"] = len(files)
    rep["files"] = sorted(files)[:60]
    return files, rep


def unpack_sources(blob: bytes) -> dict[str, str]:
    """tar.gz / gz / plain e-print bytes -> {name: text} for the text members."""
    files: dict[str, str] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile() or m.size > 20_000_000:
                    continue
                if not m.name.lower().endswith(TEXT_SUFFIXES):
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                files[m.name] = fh.read().decode("utf-8", "replace")
        if files:
            return files
    except (tarfile.TarError, EOFError, OSError, ValueError):
        pass
    for candidate in (lambda: gzip.decompress(blob), lambda: blob):
        try:
            text = candidate().decode("utf-8", "replace")
        except (OSError, ValueError, AttributeError):
            continue
        if "\\begin{" in text or "\\startdata" in text or text.lstrip().startswith("Title:"):
            files["main"] = text
            return files
    return files


def tables_from_sources(files: dict[str, str]) -> list[ParsedTable]:
    out: list[ParsedTable] = []
    for name, text in sorted(files.items()):
        if name.lower().endswith((".mrt", ".txt", ".dat", ".tab", ".table")):
            t = mrt_table(text, name)
            if t is not None:
                out.append(t)
                continue
        if "\\begin{" in text:
            out.extend(latex_tables(text, name))
    return out


# ---------------------------------------------------------------------------
# the stage entry point
# ---------------------------------------------------------------------------
@dataclass
class PaperHarvest:
    hits: pd.DataFrame = field(default_factory=pd.DataFrame)
    papers: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)

    @property
    def n_hit_tables(self) -> int:
        return sum(1 for t in self.tables if t.get("kind") == "hits")


def harvest_papers(conf_arxiv: dict, *, get_text_fn=None, get_fn=None,
                   log: AcquisitionLog | None = None, deadline: float | None = None
                   ) -> PaperHarvest:
    """Resolve, fetch and parse every seed paper; return the hit rows it yielded."""
    get_text_fn = get_text_fn or default_get_text
    get_fn = get_fn or default_get_bytes
    log = log or AcquisitionLog(prefix="relay/papers")
    timeout = float(conf_arxiv.get("timeout_s", 120))
    max_rows = int(conf_arxiv.get("max_rows_per_table", 20000))
    harvest = PaperHarvest()
    frames: list[pd.DataFrame] = []
    for key, spec in (conf_arxiv.get("papers") or {}).items():
        if deadline is not None and time.monotonic() >= deadline:
            harvest.papers.append({"seed": key, "status": "NOT_PROBED",
                                   "reason": "arxiv wall clock spent"})
            continue
        prep = resolve_paper(key, spec, get_text_fn=get_text_fn, log=log, timeout=timeout)
        harvest.papers.append(prep)
        if prep.get("status") != STATUS_OK or not prep.get("arxiv_id"):
            continue
        files, srep = eprint_sources(prep["arxiv_id"], get_fn=get_fn, log=log, timeout=timeout)
        prep["source"] = srep
        if not files:
            prep["status"] = STATUS_NO_SOURCE
            continue
        meta = {"seed": key, "arxiv_id": prep["arxiv_id"], "title": prep.get("title"),
                "telescope": spec.get("telescope"), "band": spec.get("band")}
        n_hits = 0
        parsed = tables_from_sources(files)
        for t in parsed:
            df, rep = table_hits(t, meta, max_rows=max_rows)
            harvest.tables.append(rep)
            if len(df):
                frames.append(df)
                n_hits += len(df)
        prep["n_tables_parsed"] = len(parsed)
        prep["n_hit_rows"] = n_hits
        if not n_hits:
            prep["status"] = STATUS_NO_TABLES if parsed else STATUS_ZERO

    # --- discovery sweep ----------------------------------------------------
    # The seed list is only as good as one person's memory of which paper
    # printed which table.  The sweep asks arXiv instead: every paper its
    # full-text index returns for the technosignature/drift-rate queries is
    # fetched and parsed on the same rules, and a table that carries a
    # frequency AND a drift rate becomes a hit table whoever wrote it.
    seen = {p.get("arxiv_id") for p in harvest.papers if p.get("arxiv_id")}
    cap = int(conf_arxiv.get("sweep_max_papers", 0) or 0)
    for query in (conf_arxiv.get("sweep_queries") or ()):
        if cap <= 0 or (deadline is not None and time.monotonic() >= deadline):
            break
        url = search_url(query, int(conf_arxiv.get("sweep_results_per_query", 40)))
        try:
            entries = _api_entries(get_text_fn(url, timeout=timeout))
            log.record("arxiv_sweep", url, rows=len(entries))
        except Exception as exc:                          # noqa: BLE001
            log.record("arxiv_sweep", url, error=repr(exc))
            harvest.papers.append({"seed": f"sweep:{query}", "status": STATUS_FAILED,
                                   "error": repr(exc)[:300]})
            continue
        for e in entries:
            if cap <= 0 or (deadline is not None and time.monotonic() >= deadline):
                break
            aid = e["arxiv_id"]
            if aid in seen:
                continue
            seen.add(aid)
            cap -= 1
            prep = {"seed": f"sweep:{query}", "arxiv_id": aid, "title": e["title"],
                    "published": e["published"], "route": "sweep", "status": STATUS_OK}
            harvest.papers.append(prep)
            files, srep = eprint_sources(aid, get_fn=get_fn, log=log, timeout=timeout)
            prep["source"] = srep
            if not files:
                prep["status"] = STATUS_NO_SOURCE
                continue
            meta = {"seed": prep["seed"], "arxiv_id": aid, "title": e["title"],
                    "telescope": None, "band": None}
            parsed = tables_from_sources(files)
            n_hits = 0
            for t in parsed:
                df, rep = table_hits(t, meta, max_rows=max_rows)
                harvest.tables.append(rep)
                if len(df):
                    frames.append(df)
                    n_hits += len(df)
            prep["n_tables_parsed"] = len(parsed)
            prep["n_hit_rows"] = n_hits
            if not n_hits:
                prep["status"] = STATUS_NO_TABLES if parsed else STATUS_ZERO

    harvest.hits = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return harvest


__all__ = [
    "ARXIV_API", "EPRINT_URLS", "MAX_EPRINT_BYTES", "STATUS_NO_SOURCE", "STATUS_NO_TABLES",
    "STATUS_TITLE_MISMATCH", "STATUS_UNRESOLVED", "PaperHarvest", "ParsedTable", "clean_cell",
    "default_get_bytes", "default_get_text", "drift_to_hz_s", "eprint_sources", "freq_to_mhz",
    "harvest_papers", "header_roles", "header_unit", "latex_tables", "mrt_table",
    "resolve_paper", "search_url", "table_hits", "tables_from_sources", "unpack_sources",
]
