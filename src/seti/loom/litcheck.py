"""Is a flagged object already explained in the literature?

The question that decides whether the six vetted exceedances mean anything.  They
were called "unexplained" only in the sense of *not being in a seven-object list*
— Seligman et al. (2023)'s original dark comets — and that list is known to be
incomplete: Farnocchia & Seligman (2024, PNAS) extended the population and found
two distinct classes.  An object already catalogued there is explained by hidden
outgassing and is not a lead.

So this module asks the literature directly, two ways, because either alone can
miss:

* **Targeted full-text search.**  Every designation the object goes by is searched
  on arXiv.  A hit in *any* paper about non-gravitational acceleration is the
  answer, whoever wrote it.
* **Reading the specific papers.**  The dark-comet papers are fetched and searched
  for each designation, which catches objects that appear only in a table and are
  therefore invisible to abstract-level search.

The matching is deliberately strict.  A designation like ``2006 VC`` is four
characters of very common text, and a substring match against a PDF would find it
inside "2006 VCs" or a citation year followed by initials.  Every pattern here is
anchored on word boundaries and every hit is returned **with its surrounding
context**, so a human can see what actually matched rather than trusting a count.

Runner-only: arXiv is not reachable from the development sandbox.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

ARXIV_API = "http://export.arxiv.org/api/query"

# The dark-comet and anomalous-acceleration literature, as arXiv identifiers.
# These are the papers whose object tables would already explain an exceedance.
# Identifiers gathered by the prior-art sweep (results/vnprobelit*/) and verified
# there against their real titles before being cited.
DARK_COMET_PAPERS: tuple[tuple[str, str], ...] = (
    ("2212.08115", "Seligman et al. 2023, PSJ 4, 35 -- the original seven dark comets"),
    ("2412.07603", "dark comets follow-up"),
    ("2407.01839", "dark comets follow-up"),
    ("2310.02733", "dark comets / seasonal outgassing counter-explanation"),
    ("2410.06874", "Seligman et al. 2024 -- strong non-grav accelerations and "
                   "misidentification of NEOs"),
    ("1612.06920", "Hui & Jewitt -- non-gravitational accelerations in active asteroids"),
    ("1805.05947", "Del Vigna et al. 2018 -- reliable Yarkovsky detections"),
    ("1708.05513", "Greenberg et al. 2020 -- 247 Yarkovsky detections"),
)


def designation_patterns(name: str) -> list[re.Pattern]:
    """Every form a designation might appear in, as boundary-anchored patterns.

    ``452639 (2005 UY6)`` has to match a paper that writes ``2005 UY6``,
    ``2005UY6``, ``(452639)`` or ``452639``.  Boundaries matter more than breadth:
    a bare substring search for ``2006 VC`` matches inside ``2006 VCs`` and inside
    ordinary prose, and a search for ``139359`` would match any six digits.
    """
    pats: list[re.Pattern] = []
    seen: set[str] = set()

    def add(regex: str) -> None:
        if regex not in seen:
            seen.add(regex)
            pats.append(re.compile(regex, re.IGNORECASE))

    # Provisional designations: YYYY LLnnn.  The separator is optional EVERYWHERE,
    # including between the letter pair and the trailing number, because the papers
    # render the order subscript in LaTeX and it reaches a PDF or an HTML dump as
    # "2001 ME 1" rather than "2001 ME1".  Requiring the digits to sit adjacent to
    # the letters missed exactly that form -- and the two objects the first live
    # search DID find were matched by their PERMANENT NUMBER, so an unnumbered
    # object would have been silently declared absent from a paper it appears in.
    # That is the false negative this whole module is built to avoid.
    _SEP = r"[\s~\u00a0]*"
    for m in re.finditer(r"\b((?:1[89]|20)\d{2})[\s~\u00a0]*([A-Z]{2})(\d*)\b", name):
        year, letters, digits = m.group(1), m.group(2), m.group(3)
        add(rf"\b{year}{_SEP}{letters}{_SEP}{digits}\b" if digits
            else rf"\b{year}{_SEP}{letters}\b")
    # Permanent numbers, ONLY parenthesised.  A bare run of digits in a PDF is a
    # page number, a count, a year or a table entry far more often than it is a
    # designation, and the cost of a false match here is the worst available: it
    # declares a real lead "already explained" and discards it silently.  The
    # provisional designation above is the reliable key; this is a supplement.
    for m in re.finditer(r"\((\d{3,7})\)|^\s*(\d{3,7})\b", name):
        num = m.group(1) or m.group(2)
        add(rf"\(\s*{num}\s*\)")
    return pats


@dataclass
class ObjectCheck:
    """What the literature says about one object."""

    name: str
    hits: list = field(default_factory=list)
    searched: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def explained(self) -> bool:
        return bool(self.hits)

    def as_dict(self) -> dict:
        return {"name": self.name, "explained_in_literature": self.explained,
                "n_hits": len(self.hits), "hits": self.hits,
                "searched": self.searched, "errors": self.errors,
                "verdict": ("EXPLAINED_IN_LITERATURE" if self.explained
                            else "NOT_FOUND_IN_SEARCHED_LITERATURE")}


def find_in_text(text: str, name: str, source: str,
                 context: int = 160) -> list[dict]:
    """Every boundary-anchored match of ``name`` in ``text``, with context.

    Context is returned rather than a count because a count cannot be checked. A
    reader has to be able to see that the match is the object and not a year, a
    citation, or a fragment of another designation.
    """
    out: list[dict] = []
    for pat in designation_patterns(name):
        for m in pat.finditer(text):
            lo = max(0, m.start() - context)
            hi = min(len(text), m.end() + context)
            snippet = re.sub(r"\s+", " ", text[lo:hi]).strip()
            out.append({"source": source, "matched": m.group(0),
                        "context": snippet})
            if len(out) >= 6:
                return out
    return out


# ---------------------------------------------------------------------------
# Network (runner only)
# ---------------------------------------------------------------------------
def fetch_arxiv_search(query: str, timeout: float = 60.0,
                       max_results: int = 25) -> dict:
    """arXiv API search, returning title+abstract text per result."""
    import requests

    params = {"search_query": query, "max_results": str(max_results),
              "sortBy": "relevance"}
    out: dict = {"query": query, "entries": []}
    try:
        resp = requests.get(ARXIV_API, params=params, timeout=timeout)
        out["status"] = resp.status_code
        if resp.status_code != 200:
            out["body"] = resp.text[:300]
            return out
        body = resp.text
        for m in re.finditer(r"<entry>(.*?)</entry>", body, re.S):
            e = m.group(1)
            out["entries"].append({"id": _xml_field(e, "id"),
                                   "title": _xml_field(e, "title"),
                                   "summary": _xml_field(e, "summary")})
    except Exception as exc:                                  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def fetch_paper_text(arxiv_id: str, timeout: float = 90.0) -> dict:
    """Full text of one arXiv paper, HTML first and PDF as the fallback.

    Object designations usually live in a *table*, not an abstract, so
    abstract-level search alone would miss exactly the objects this is checking.
    """
    import requests

    out: dict = {"arxiv_id": arxiv_id, "text": "", "source": None}
    for label, url in (("html", f"https://arxiv.org/abs/{arxiv_id}"),
                       ("html_v1", f"https://arxiv.org/html/{arxiv_id}v1"),
                       ("pdf", f"https://arxiv.org/pdf/{arxiv_id}")):
        try:
            resp = requests.get(url, timeout=timeout,
                                headers={"User-Agent": "seti-loom/1.0"})
            if resp.status_code != 200:
                out.setdefault("attempts", {})[label] = resp.status_code
                continue
            if label == "pdf":
                try:
                    import io

                    from pypdf import PdfReader
                    reader = PdfReader(io.BytesIO(resp.content))
                    text = "\n".join((p.extract_text() or "")
                                     for p in reader.pages)
                except Exception as exc:                      # noqa: BLE001
                    out.setdefault("attempts", {})[label] = f"pdf_unreadable: {exc}"[:160]
                    continue
            else:
                text = re.sub(r"<[^>]+>", " ", resp.text)
            if len(text) > len(out["text"]):
                out["text"], out["source"] = text, label
            # An HTML full text is as good as a PDF; stop paying for the PDF.
            if label.startswith("html") and len(text) > 20000:
                break
        except Exception as exc:                              # noqa: BLE001
            out.setdefault("attempts", {})[label] = f"{type(exc).__name__}: {exc}"[:160]
    out["n_chars"] = len(out["text"])
    return out


def check_objects(names: list[str], papers=DARK_COMET_PAPERS,
                  pause: float = 3.0, on_result=None) -> dict:
    """Search the literature for every named object.  Runner-only.

    Returns a per-object verdict plus the evidence.  ``NOT_FOUND_IN_SEARCHED_
    LITERATURE`` is deliberately not called "unexplained": it means these searches
    did not find it, which is a statement about the searches as much as about the
    object, and the searched set is reported alongside so the claim can be judged.
    """
    out: dict = {"papers": {}, "objects": {},
                 "papers_searched": [p[0] for p in papers]}

    # Read the papers once; every object is then checked against all of them.
    texts: dict[str, str] = {}
    for arxiv_id, label in papers:
        rec = fetch_paper_text(arxiv_id)
        rec["label"] = label
        texts[arxiv_id] = rec.pop("text", "")
        rec["n_chars"] = len(texts[arxiv_id])
        out["papers"][arxiv_id] = rec
        if on_result is not None:
            on_result(f"paper:{arxiv_id}", rec)
        time.sleep(pause)

    for name in names:
        chk = ObjectCheck(name=name)
        for arxiv_id, label in papers:
            text = texts.get(arxiv_id, "")
            chk.searched.append(f"{arxiv_id} ({len(text)} chars)")
            if not text:
                continue
            for hit in find_in_text(text, name, f"arXiv:{arxiv_id} -- {label}"):
                chk.hits.append(hit)

        # And a targeted search, which catches papers not in the fixed list.
        for pat in _search_terms(name):
            res = fetch_arxiv_search(f'all:"{pat}"')
            chk.searched.append(f'arxiv_search all:"{pat}"')
            if "error" in res:
                chk.errors.append(res["error"])
                continue
            for e in res.get("entries", []):
                blob = f"{e.get('title', '')} {e.get('summary', '')}"
                for hit in find_in_text(blob, name,
                                        f"arXiv search -> {e.get('id', '?')}"):
                    hit["title"] = e.get("title", "")
                    chk.hits.append(hit)
            time.sleep(pause)

        out["objects"][name] = chk.as_dict()
        if on_result is not None:
            on_result(f"object:{name}", out["objects"][name])

    explained = [n for n, v in out["objects"].items()
                 if v["explained_in_literature"]]
    out["n_objects"] = len(names)
    out["n_explained"] = len(explained)
    out["n_not_found"] = len(names) - len(explained)
    out["explained"] = explained
    out["verdict"] = ("ALL_EXPLAINED" if len(explained) == len(names)
                      else "SOME_NOT_FOUND" if explained else "NONE_FOUND")
    return out


def _xml_field(entry: str, tag: str) -> str:
    """One tag's text out of an arXiv API <entry>, whitespace collapsed."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", entry, re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def _search_terms(name: str) -> list[str]:
    """The literal strings worth searching arXiv for, from a full designation."""
    terms: list[str] = []
    for m in re.finditer(r"\b((?:1[89]|20)\d{2})[\s~ ]*([A-Z]{2}\d*)\b", name):
        terms.append(f"{m.group(1)} {m.group(2)}")
    return terms or [name.strip()]
