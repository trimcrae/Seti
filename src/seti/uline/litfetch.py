"""Reach the microwave literature from the runner and read constants out of it.

The five species this channel most wants — CF₂Cl₂, CFCl₃, SO₂F₂, CHClF₂ and
CF₂ — have no JPL or CDMS entry, so their rotational spectra have to be
computed (:mod:`seti.uline.rotorpred`).  A computed spectrum is only as good as
its constants, and the offline assets file says so in every block: the A, B, C
are reconstructed from the literature and the **quartic centrifugal-distortion
constants are mostly unknown**, which is what makes four of the five
unsearchable at survey precision rather than merely uncatalogued.

This module is the attempt to fix that on the runner, where there is egress.
It walks a ladder of open, key-free services per species, records **what each
one answered** (status, bytes, elapsed, the first kilobyte), and runs the
parsers over whatever came back:

* ``jpl_doc`` / ``cdms_doc`` — a catalogue's own documentation file, whose SPFIT
  parameter block gives the constants in Watson's own reduction
  (:func:`seti.uline.rotor.parse_jpl_doc`);
* ``nist_triatomic`` — the NIST Diatomic/Triatomic Spectral Database page for a
  species, which carries both the fitted constants and a table of **measured
  transition frequencies**; the latter is the more valuable half, because
  :func:`seti.uline.rotor.fit_constants` can fit the A-reduced Hamiltonian to it
  directly and so *derive* the quartic constants rather than quote them;
* ``cccbdb`` — NIST's computational chemistry comparison benchmark, experimental
  rotational-constant tables;
* ``openalex`` / ``europepmc`` / ``crossref`` / ``semanticscholar`` / ``arxiv`` —
  bibliographic APIs that return abstracts, which sometimes quote the constants
  in the abstract itself; and
* ``text`` / ``html`` / ``pdf`` — anything else, scraped generically.

Nothing fetched ever overwrites an embedded constant by itself.  Every value the
parsers find is reported **next to** the embedded one in
``results/uline/literature.json``, with the URL it came from; promoting one is a
commit somebody can read, which is the whole point of the ``verify`` flag.
"""

from __future__ import annotations

import json
import re
import time
from urllib.parse import quote_plus

from .acquire import STATUS_FAILED, STATUS_OK, AcquisitionLog, fetch_text
from .rotor import QUARTIC, SEXTIC, parse_jpl_doc

#: Greek and ASCII spellings of the constants, mapped to the asset keys.
_NAME_MAP = {
    "a": "A", "b": "B", "c": "C",
    "dj": "DJ", "deltaj": "DJ", "Δj": "DJ", "d_j": "DJ",
    "djk": "DJK", "deltajk": "DJK", "Δjk": "DJK", "d_jk": "DJK",
    "dk": "DK", "deltak": "DK", "Δk": "DK", "d_k": "DK",
    "dlj": "dJ", "deltalj": "dJ", "δj": "dJ", "smalldeltaj": "dJ",
    "dlk": "dK", "deltalk": "dK", "δk": "dK", "smalldeltak": "dK",
    "hj": "HJ", "phij": "HJ", "Φj": "HJ",
    "hjk": "HJK", "hkj": "HKJ", "hk": "HK",
}

#: ``A = 4118.90(12) MHz`` / ``DJ = 0.46 kHz`` / ``Δ_J = 1.234 5 kHz`` and the
#: tabular spelling ``A  4118.90(12)``.  The unit is optional: MHz is assumed
#: for A/B/C and kHz for the quartic constants, which is the near-universal
#: convention in the microwave literature, and the assumption is reported.
_ASSIGN_RE = re.compile(
    r"(?<![A-Za-z0-9])(Delta_?J_?K|Delta_?J|Delta_?K|delta_?J|delta_?K|"
    r"A|B|C|D_?J_?K|D_?J|D_?K|d_?J|d_?K|[ΔΦ]_?J_?K|[ΔΦ]_?J|[ΔΦ]_?K|"
    r"δ_?J|δ_?K|H_?J_?K|H_?K_?J|H_?J|H_?K)\s*(?:\(MHz\)|\(kHz\)|\(Hz\))?\s*[=:]\s*"
    r"(-?\d+(?:[.,]\d+)?(?:\s*\(\s*\d+\s*\))?(?:[eE][-+]?\d+)?)\s*"
    r"(MHz|kHz|Hz|GHz|cm-1|cm\^-1)?", re.IGNORECASE)

_UNIT_SCALE = {"ghz": 1e3, "mhz": 1.0, "khz": 1e-3, "hz": 1e-6,
               "cm-1": 29979.2458, "cm^-1": 29979.2458}


def _canon_name(raw: str) -> str | None:
    """``Δ_JK``, ``D_JK``, ``Delta_JK`` → ``DJK``; ``δ_J``, ``d_J``, ``delta_J`` → ``dJ``.

    **The case of the first letter is the whole distinction** between Watson's
    two families (``Δ_J`` the symmetric quartic constant, ``δ_J`` the asymmetry
    one), and the spelled-out ``Delta``/``delta`` carries it the same way.  The
    match is case-insensitive so both spellings are found, and the case is read
    back here rather than being thrown away by the regex.
    """
    s = str(raw).strip().replace("_", "").replace(" ", "")
    s = s.replace("Δ", "D").replace("Φ", "H").replace("δ", "d")
    lower_first = s[:1].islower()
    if s.lower().startswith("delta"):
        s = ("d" if lower_first else "D") + s[5:]
    if s in ("A", "B", "C"):
        return s
    upper = s.upper()
    if lower_first and upper in ("DJ", "DK"):
        return {"DJ": "dJ", "DK": "dK"}[upper]
    table = {"DJ": "DJ", "DJK": "DJK", "DK": "DK", "HJ": "HJ", "HJK": "HJK",
             "HKJ": "HKJ", "HK": "HK"}
    return table.get(upper)


def scrape_constants(text: str, *, max_hits: int = 400) -> dict:
    """Every ``name = value unit`` assignment of a rotational constant in ``text``.

    Returns ``{"constants": {name: MHz}, "hits": [{name, value, unit, raw}]}``.
    The FIRST occurrence of a name wins (papers quote the fitted value before
    the comparisons), values in parentheses are the standard error and are
    dropped, and a missing unit is filled in by convention — MHz for A, B, C and
    kHz for a distortion constant — with ``unit_assumed`` set so a reader can
    see the assumption rather than inherit it.
    """
    out: dict = {"constants": {}, "hits": []}
    for m in _ASSIGN_RE.finditer(str(text or "")):
        if len(out["hits"]) >= max_hits:
            break
        name = _canon_name(m.group(1))
        if name is None:
            continue
        raw = m.group(2)
        val = re.sub(r"\(\s*\d+\s*\)", "", raw).replace(",", "").strip()
        try:
            v = float(val)
        except ValueError:
            continue
        unit = (m.group(3) or "").strip().lower()
        assumed = not unit
        if assumed:
            unit = "mhz" if name in ("A", "B", "C") else "khz"
        scale = _UNIT_SCALE.get(unit)
        if scale is None:
            continue
        mhz = v * scale
        hit = {"name": name, "value_mhz": mhz, "unit": unit, "unit_assumed": assumed,
               "raw": m.group(0)[:120]}
        out["hits"].append(hit)
        out["constants"].setdefault(name, mhz)
    return out


# ---------------------------------------------------------------------------
# NIST Diatomic / Triatomic Spectral Database
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r"<[^>]+>")
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)


def html_rows(html: str) -> list[list[str]]:
    """Every ``<tr>`` of an HTML page as a list of de-tagged cell strings."""
    rows = []
    for r in _ROW_RE.finditer(str(html or "")):
        cells = [_TAG_RE.sub(" ", c) for c in _CELL_RE.findall(r.group(1))]
        cells = [re.sub(r"\s+", " ", c).replace("&nbsp;", " ").strip() for c in cells]
        if cells:
            rows.append(cells)
    return rows


_QN_RE = re.compile(r"^\s*(\d{1,3})\s*[\(\[ _,]\s*(\d{1,3})\s*[, _]\s*(\d{1,3})\s*[\)\]]?\s*$")


def parse_measured_lines(html: str, *, min_freq_mhz: float = 1.0,
                         max_freq_mhz: float = 3.0e6) -> list[dict]:
    """``[{"up": (J,Ka,Kc), "lo": (J,Ka,Kc), "freq_mhz": ν}]`` from an HTML table.

    Written for the NIST Triatomic Spectral Database's measured-transition
    tables, whose rows carry the two ``J(Ka,Kc)`` labels and a frequency, but
    deliberately shape-agnostic: any row with two parsable ``J(Ka,Kc)`` cells
    and one numeric cell in a plausible frequency range is taken, and the unit
    is decided by magnitude (a rotational transition quoted in MHz is ≥ 1000,
    in GHz < 1000, and cm⁻¹ values are converted).  A table that yields nothing
    yields nothing — no row is ever invented from a partial parse.
    """
    out = []
    for cells in html_rows(html):
        qns = []
        nums = []
        for c in cells:
            m = _QN_RE.match(c)
            if m:
                qns.append(tuple(int(x) for x in m.groups()))
                continue
            t = c.replace(",", "").strip()
            t = re.sub(r"\(\s*\d+\s*\)$", "", t).strip()
            try:
                nums.append(float(t))
            except ValueError:
                continue
        if len(qns) < 2 or not nums:
            continue
        freq = None
        for v in nums:
            for scale in (1.0, 1e3, 29979.2458):
                f = v * scale
                if min_freq_mhz <= f <= max_freq_mhz and (
                        scale == 1.0 and v >= 1000.0 or scale == 1e3 and v < 1000.0
                        or scale > 1e3 and v < 100.0):
                    freq = f
                    break
            if freq is not None:
                break
        if freq is None:
            continue
        out.append({"up": qns[0], "lo": qns[1], "freq_mhz": float(freq)})
    return out


# ---------------------------------------------------------------------------
# bibliographic APIs (open, key-free)
# ---------------------------------------------------------------------------
def openalex_url(query: str) -> str:
    return ("https://api.openalex.org/works?per-page=10&search=" + quote_plus(query))


def europepmc_url(query: str) -> str:
    return ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json"
            "&resultType=core&pageSize=10&query=" + quote_plus(query))


def crossref_url(query: str) -> str:
    return "https://api.crossref.org/works?rows=10&query=" + quote_plus(query)


def semanticscholar_url(query: str) -> str:
    return ("https://api.semanticscholar.org/graph/v1/paper/search?limit=10"
            "&fields=title,abstract,year,venue,externalIds&query=" + quote_plus(query))


def arxiv_url(query: str) -> str:
    return ("http://export.arxiv.org/api/query?max_results=10&search_query=all:"
            + quote_plus(f'"{query}"'))


BIBLIO_BUILDERS = {"openalex": openalex_url, "europepmc": europepmc_url,
                   "crossref": crossref_url, "semanticscholar": semanticscholar_url,
                   "arxiv": arxiv_url}


def _invert_abstract(idx: dict) -> str:
    if not isinstance(idx, dict):
        return ""
    pos: dict[int, str] = {}
    for word, places in idx.items():
        for p in places or []:
            pos[int(p)] = str(word)
    return " ".join(pos[k] for k in sorted(pos))


def biblio_text(kind: str, body: str) -> str:
    """Titles + abstracts of a bibliographic response, as one searchable string.

    The constants are sometimes quoted in the abstract itself (``B0 = 2466.2
    MHz``), which is why this exists: the scraper then runs over prose that a
    paywall would otherwise hide.  A body that does not parse returns the raw
    text, so the generic scraper still sees it.
    """
    try:
        if kind == "arxiv":
            return re.sub(r"<[^>]+>", " ", body)
        data = json.loads(body)
    except Exception:                                          # noqa: BLE001
        return str(body)
    parts: list[str] = []
    if kind == "openalex":
        for w in (data.get("results") or []):
            parts.append(str(w.get("title") or ""))
            parts.append(_invert_abstract(w.get("abstract_inverted_index") or {}))
    elif kind == "europepmc":
        for w in ((data.get("resultList") or {}).get("result") or []):
            parts.append(str(w.get("title") or ""))
            parts.append(str(w.get("abstractText") or ""))
    elif kind == "crossref":
        for w in ((data.get("message") or {}).get("items") or []):
            parts.append(" ".join(w.get("title") or []))
            parts.append(str(w.get("abstract") or ""))
    elif kind == "semanticscholar":
        for w in (data.get("data") or []):
            parts.append(str(w.get("title") or ""))
            parts.append(str(w.get("abstract") or ""))
    else:
        return str(body)
    return re.sub(r"<[^>]+>", " ", "\n".join(p for p in parts if p))


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------
def fetch_one(spec: dict, *, fetch_fn=None, timeout: float = 90.0, retries: int = 2,
              log: AcquisitionLog | None = None) -> dict:
    """Fetch one literature source and parse it; never raises.

    The record is the deliverable even when the fetch fails: ``status``,
    ``error``, ``elapsed_s``, ``n_bytes`` and the first kilobyte of whatever
    came back, so a route that answered with a login page, a 403 or an empty
    body is told apart from one that was never tried.
    """
    url = str(spec.get("url") or "")
    kind = str(spec.get("kind") or "text")
    rec = {"name": str(spec.get("name") or url[:60]), "species": spec.get("species"),
           "kind": kind, "url": url, "status": STATUS_FAILED}
    t0 = time.time()
    try:
        body = fetch_text(url, fetch_fn=fetch_fn, retries=int(retries), timeout=float(timeout))
    except Exception as exc:                                   # noqa: BLE001
        rec.update({"error": repr(exc)[:1500], "elapsed_s": round(time.time() - t0, 2)})
        if log:
            log.record("litfetch", url, error=repr(exc))
        return rec
    if not body:
        # fetch_text RETURNS None after exhausting its retries; it does not
        # raise.  A route that came back empty is a failed route, and calling it
        # OK would put an empty ledger entry next to a real one.
        rec.update({"error": "no body after retries (fetch_text returned None/empty)",
                    "elapsed_s": round(time.time() - t0, 2), "n_bytes": 0})
        if log:
            log.record("litfetch", url, error="empty body")
        return rec
    rec.update({"status": STATUS_OK, "elapsed_s": round(time.time() - t0, 2),
                "n_bytes": len(body or ""), "head": (body or "")[:1000]})
    text = body or ""
    if kind in BIBLIO_BUILDERS:
        text = biblio_text(kind, body)
        rec["biblio_text_chars"] = len(text)
        rec["head"] = text[:1000]
    if kind in ("jpl_doc", "cdms_doc"):
        parsed = parse_jpl_doc(text)
        rec["constants_mhz"] = parsed["constants"]
        rec["spfit_codes"] = parsed["codes"]
        rec["parser"] = "parse_jpl_doc"
    else:
        sc = scrape_constants(text)
        rec["constants_mhz"] = sc["constants"]
        rec["constant_hits"] = sc["hits"][:40]
        rec["parser"] = "scrape_constants"
    if kind in ("nist_triatomic", "html"):
        lines = parse_measured_lines(text)
        rec["n_measured_lines"] = len(lines)
        rec["measured_lines"] = lines[:4000]
    if log:
        log.record("litfetch", url, rows=rec.get("n_bytes"),
                   extra={"kind": kind, "constants": sorted(rec.get("constants_mhz") or {})})
    return rec


def species_query_sources(species: str, *, queries: list[str] | None = None,
                          kinds=("openalex", "europepmc", "crossref", "semanticscholar")
                          ) -> list[dict]:
    """The bibliographic ladder for one species, as ``fetch_one`` specs.

    Two queries per service: the species' own name with "rotational spectrum",
    and the same with "centrifugal distortion", which is the phrase a paper that
    *reports the quartic constants* almost always carries in its abstract.
    """
    qs = list(queries or [f"{species} rotational spectrum",
                          f"{species} centrifugal distortion constants millimeter wave"])
    out = []
    for kind in kinds:
        build = BIBLIO_BUILDERS.get(kind)
        if build is None:
            continue
        for i, q in enumerate(qs):
            out.append({"name": f"{kind}_{species}_{i}", "species": species, "kind": kind,
                        "url": build(q), "query": q})
    return out


def run_litfetch(sources: list[dict], *, fetch_fn=None, log: AcquisitionLog | None = None,
                 timeout: float = 90.0, retries: int = 2) -> dict:
    """Every source in order; ``{"sources": [...], "by_species": {...}}``.

    ``by_species`` collects, per species and per constant, **every** value any
    route produced with the URL it came from — a side-by-side ledger, not a
    decision.  Choosing between them is a commit to the assets file
    (:func:`seti.uline.rotorpred.merge_fetched_constants`), never something a
    fetch does on its own.
    """
    recs = [fetch_one(s, fetch_fn=fetch_fn, log=log, timeout=timeout, retries=retries)
            for s in (sources or [])]
    by_species: dict[str, dict] = {}
    for r in recs:
        sp = str(r.get("species") or "")
        if not sp:
            continue
        blk = by_species.setdefault(sp, {"constants": {}, "measured_lines": 0, "routes_ok": 0,
                                         "routes_failed": 0})
        if r.get("status") == STATUS_OK:
            blk["routes_ok"] += 1
        else:
            blk["routes_failed"] += 1
        for k, v in (r.get("constants_mhz") or {}).items():
            if k in ("A", "B", "C", *QUARTIC, *SEXTIC):
                blk["constants"].setdefault(k, []).append(
                    {"value_mhz": float(v), "from": r.get("name"), "url": r.get("url")})
        blk["measured_lines"] += int(r.get("n_measured_lines") or 0)
    n_ok = sum(1 for r in recs if r.get("status") == STATUS_OK)
    return {"n_sources": len(recs), "n_ok": n_ok, "n_failed": len(recs) - n_ok,
            "sources": recs, "by_species": by_species}


__all__ = ["BIBLIO_BUILDERS", "arxiv_url", "biblio_text", "crossref_url", "europepmc_url",
           "fetch_one", "html_rows", "openalex_url", "parse_measured_lines", "run_litfetch",
           "scrape_constants", "semanticscholar_url", "species_query_sources"]
