#!/usr/bin/env python3
"""VIGIL novelty check: fetch the primary sources on the GitHub runner.

The sandbox blocks arxiv.org / OpenAlex (``CONNECT tunnel failed, response
403``); the Actions runner has egress.  This script establishes, **from the
record rather than from memory**, two things that the VIGIL channel stands on:

1. **Has anyone ever framed mid-infrared variability with optical constancy as a
   technosignature?**  The proposed observable is a *conjunction*: a source whose
   W1/W2 flux moves while its optical flux does not.  Each half separately is a
   large and old literature --- mid-IR variability is routine for YSOs, extreme
   debris disks, and AGN; optical-transit megastructure searches are a decade
   old --- so a keyword sweep that does not separate the conjunction from those
   four decoys will return a false "occupied" every time.  The scan below is
   built to make a null *interpretable* rather than merely empty.
2. **How is the unTimely mid-IR variable catalogue actually reached?**  A
   candidate list is worthless if the parent table cannot be downloaded, so the
   abstracts and full texts are regex-mined for URLs, DOIs, Zenodo deposits,
   VizieR catalogue identifiers and NOIRLab Astro Data Lab table names, for both
   the variable catalogue and the parent unTimely Catalog (Meisner et al. 2023)
   it is built from.

**Citations are verified by TITLE, not by ID.**  Earlier agents in this repo
recorded arXiv identifiers that resolved to unrelated physics --- a plausible
number is not a citation.  Every decisive ID below is fetched, its ``<title>``
extracted, and the match against the expected title fragment recorded in
``id_verification`` and echoed at the top of ``summary.json``.  A failed
verification is meant to be loud.  For the unTimely variable catalogue there is
additionally a title-based *search* fallback, so that a bad ID still yields the
real paper's identifier (or an explicit statement that no such paper was found).

The third leg is Project Hephaistos (Suazo et al.), whose Dyson-sphere search
explicitly *discards* variable stars.  If that is true it is the single most
useful fact for VIGIL: the nearest prior art threw away exactly the population
this channel selects.  The full text is pulled and the sentences around
``G_var``, ``variab`` and ``absorbing elements`` are recorded verbatim, because
this claim must be quotable, not remembered.

Outputs under ``results/vigillit/``:
  arxiv_id_<name>.atom       arXiv API metadata for a specific claimed ID
  arxiv_q_<name>.atom        arXiv API keyword search (discovery / sweep)
  txt_<name>.txt             extracted plain text (PDF -> pdftotext)
  oa_<name>.json             OpenAlex work record (best of DOI / title routes)
  oa_pick_<name>.json        which OpenAlex route won and what each returned
  id_verification.json       title-level verification of every claimed ID
  hephaistos_variability.json  verbatim sentences around the variability cut
  data_access_routes.json    URLs / DOIs / VizieR / Data Lab tables per paper
  concept_scan.json          decoy-aware conjunction scan over every abstract
  summary.json               verdict + fetch status for every URL
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/vigillit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-vigillit/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []
PAUSE = 3.0

ARXIV_API = "http://export.arxiv.org/api/query"
OA_API = "https://api.openalex.org/works"
MAX_RESULTS = 60


def get(url: str, dest: pathlib.Path, tries: int = 3, pause: float = PAUSE) -> bool:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "dest": dest.name, "ok": True, "bytes": len(data)})
            print(f"  ok  {len(data):>9,}B  {dest.name}")
            time.sleep(pause)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  try {i + 1}/{tries} failed: {exc!r}")
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "dest": dest.name, "ok": False})
    return False


# --------------------------------------------------------------------------
# arXiv / OpenAlex primitives
# --------------------------------------------------------------------------
def arxiv_id(name: str, aid: str) -> bool:
    url = f"{ARXIV_API}?search_query=&id_list={aid}&start=0&max_results=1"
    return get(url, OUT / f"arxiv_id_{name}.atom")


def arxiv_query(name: str, q: str, max_results: int = MAX_RESULTS) -> bool:
    url = (f"{ARXIV_API}?search_query={urllib.parse.quote(q)}"
           f"&start=0&max_results={max_results}"
           f"&sortBy=relevance&sortOrder=descending")
    return get(url, OUT / f"arxiv_q_{name}.atom")


def fulltext(name: str, aid: str) -> pathlib.Path | None:
    """Fetch the PDF and convert to text; return the text path if it exists."""
    pdf = OUT / f"pdf_{name}.pdf"
    txt = OUT / f"txt_{name}.txt"
    if not get(f"https://arxiv.org/pdf/{aid}", pdf):
        return None
    try:
        subprocess.run(["pdftotext", str(pdf), str(txt)], check=False, timeout=180)
        pdf.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  pdftotext failed for {name}: {exc!r}")
    return txt if txt.exists() else None


def entries(path: pathlib.Path) -> list[dict]:
    """Parse an arXiv Atom response into {arxiv, title, summary} records."""
    if not path.exists():
        return []
    text = path.read_text(errors="ignore")
    out: list[dict] = []
    for m in re.finditer(r"<entry>(.*?)</entry>", text, re.S):
        e = m.group(1)
        aid = (re.search(r"<id>(.*?)</id>", e, re.S) or [None, ""])[1].strip()
        title = " ".join((re.search(r"<title>(.*?)</title>", e, re.S) or [None, ""])[1].split())
        summ = " ".join((re.search(r"<summary>(.*?)</summary>", e, re.S) or [None, ""])[1].split())
        out.append({"arxiv": aid, "title": title, "summary": summ})
    return out


def short_id(url_or_id: str) -> str:
    """``http://arxiv.org/abs/2405.08657v2`` -> ``2405.08657v2``."""
    return str(url_or_id).rstrip("/").rsplit("/", 1)[-1]


def openalex(name: str, title: str, aid: str) -> None:
    """Work record plus citation count, by DOI *and* by title.

    The arXiv DOI ``10.48550/arXiv.<id>`` resolves to the *preprint stub*, which
    OpenAlex frequently reports with ``cited_by_count = 0`` even for
    heavily-cited papers.  So we also search by title and keep whichever record
    has the larger citation count, recording in ``oa_pick_<name>.json`` which
    route won --- an uninformative leg must be visibly uninformative.
    """
    cands: list[dict] = []
    if get(f"{OA_API}/https://doi.org/10.48550/arXiv.{aid}", OUT / f"oa_doi_{name}.json"):
        try:
            cands.append(json.loads((OUT / f"oa_doi_{name}.json").read_text()))
        except Exception:  # noqa: BLE001
            pass
    q = urllib.parse.quote(title)
    if get(f"{OA_API}?filter=title.search:{q}&per-page=10&sort=cited_by_count:desc",
           OUT / f"oa_search_{name}.json"):
        try:
            res = json.loads((OUT / f"oa_search_{name}.json").read_text()).get("results", [])
            cands.extend(res[:5])
        except Exception:  # noqa: BLE001
            pass
    cands = [c for c in cands if isinstance(c, dict) and c.get("id")]
    if not cands:
        (OUT / f"oa_pick_{name}.json").write_text(json.dumps(
            {"resolved": False, "note": "no OpenAlex record found"}, indent=2))
        return
    best = max(cands, key=lambda c: int(c.get("cited_by_count") or 0))
    (OUT / f"oa_{name}.json").write_text(json.dumps(best, indent=2))
    (OUT / f"oa_pick_{name}.json").write_text(json.dumps({
        "resolved": True, "chosen_id": best.get("id"),
        "chosen_cited_by_count": int(best.get("cited_by_count") or 0),
        "candidates": [{"id": c.get("id"), "title": c.get("display_name"),
                        "cited_by_count": int(c.get("cited_by_count") or 0),
                        "type": c.get("type")} for c in cands],
        "warning": ("cited_by_count == 0 here means the citation leg is "
                    "UNINFORMATIVE for this paper, not that nobody cited it"),
    }, indent=2))


# --------------------------------------------------------------------------
# 1. Verify the decisive citations BY TITLE
# --------------------------------------------------------------------------
# Every one of these is load-bearing for the VIGIL argument, and every one of
# them was written down by an agent that could not reach arxiv.org.  ``mode``
# is "all" when each fragment must appear in the fetched title, "any" when one
# suffices.
CLAIMED_IDS: dict[str, dict] = {
    "untimely_variable": {
        "arxiv_id": "2511.22071",
        "fragments": ["untimely", "variable"],
        "mode": "all",
        "claimed_title": "A Catalogue of Mid-infrared Variable Sources from unTimely",
        "claim": ("the parent catalogue of mid-IR variables that VIGIL selects "
                  "from; if this ID is wrong the whole channel has no input table"),
    },
    "moor_extreme_debris": {
        "arxiv_id": "2103.00568",
        "fragments": ["extreme debris disk"],
        "mode": "all",
        "claimed_title": ("Unusual Variability of Warm Extreme Debris Disks / AllWISE "
                          "monitoring of extreme debris disks (Moor et al.)"),
        "claim": ("claimed to state that the monitored stars were photometrically "
                  "stable with flat optical light curves while the mid-IR varied "
                  "--- i.e. the principal natural confounder for VIGIL"),
    },
    "wright2020_dyson": {
        "arxiv_id": "2006.16734",
        "fragments": ["dyson"],
        "mode": "all",
        "claimed_title": "Dyson Spheres (Wright 2020, SerAJ 200, 1)",
        "claim": "the review that fixes what a Dyson-sphere signature is taken to be",
    },
    "contardo_hogg": {
        "arxiv_id": "2403.18941",
        "fragments": ["infrared excess", "dyson"],
        "mode": "any",
        "claimed_title": "Contardo & Hogg, infrared-excess / Dyson-sphere search",
        "claim": "the most recent static IR-excess baseline VIGIL must move beyond",
    },
}


def verify_claimed_ids() -> list[dict]:
    """Fetch each claimed ID and compare the *fetched title* to expectation."""
    recs: list[dict] = []
    for name, spec in CLAIMED_IDS.items():
        aid, frags, mode = spec["arxiv_id"], spec["fragments"], spec["mode"]
        print(f"-- verifying {name} ({aid})")
        ok = arxiv_id(name, aid)
        ents = entries(OUT / f"arxiv_id_{name}.atom") if ok else []
        fetched = ents[0]["title"] if ents else ""
        fetched_abs = ents[0]["summary"] if ents else ""
        low = fetched.lower()
        # The arXiv API answers a nonexistent id_list with a single entry whose
        # title is literally "Error"; that is a failed lookup, not a mismatch.
        is_error = (not fetched) or low.strip().startswith("error")
        if mode == "all":
            matched = all(f.lower() in low for f in frags)
        else:
            matched = any(f.lower() in low for f in frags)
        verified = bool(ok and ents and not is_error and matched)
        recs.append({
            "name": name,
            "arxiv_id": aid,
            "expected_title_fragment": frags,
            "fragment_match_mode": mode,
            "claimed_title": spec["claimed_title"],
            "why_it_matters": spec["claim"],
            "fetch_ok": bool(ok),
            "fetched_title": fetched,
            "fetched_abstract_head": fetched_abs[:400],
            "verified": verified,
            "failure_mode": ("" if verified else
                             ("fetch_failed" if not ok else
                              "id_does_not_resolve" if is_error else
                              "title_mismatch_ID_IS_WRONG")),
        })
        flag = "VERIFIED" if verified else "*** FAILED ***"
        print(f"   {flag}  {aid}  -> {fetched[:90]!r}")
    return recs


def untimely_title_search() -> dict:
    """Title-based fallback: what *is* the unTimely variable catalogue's ID?"""
    queries = {
        "untimely_var_ti": 'ti:"unTimely" AND ti:"variable"',
        "untimely_var_all": 'all:"unTimely" AND all:"variable" AND all:"mid-infrared"',
        "untimely_any": 'all:"unTimely"',
    }
    cands: list[dict] = []
    for name, q in queries.items():
        arxiv_query(name, q, max_results=40)
        for e in entries(OUT / f"arxiv_q_{name}.atom"):
            low = e["title"].lower()
            if "untimely" not in low and "untimely" not in e["summary"].lower():
                continue
            cands.append({"arxiv": short_id(e["arxiv"]), "title": e["title"],
                          "looks_like_variable_catalogue": bool(
                              re.search(r"variab", low) and re.search(r"catalog", low)),
                          "source_query": name})
    seen, uniq = set(), []
    for c in cands:
        if c["arxiv"] in seen:
            continue
        seen.add(c["arxiv"])
        uniq.append(c)
    best = [c for c in uniq if c["looks_like_variable_catalogue"]]
    return {
        "n_untimely_papers_found": len(uniq),
        "candidates": uniq,
        "best_guess_variable_catalogue": best[0] if best else None,
        "note": ("If the claimed ID 2511.22071 failed verification, "
                 "best_guess_variable_catalogue is the real target --- and if it "
                 "is null, no unTimely variable catalogue exists on arXiv and the "
                 "VIGIL input table has to be rebuilt from the unTimely Catalog "
                 "epoch photometry directly."),
    }


# --------------------------------------------------------------------------
# 2. Project Hephaistos: does the nearest prior art discard variable stars?
# --------------------------------------------------------------------------
HEPH_QUERIES = {
    "hephaistos": 'all:"Project Hephaistos"',
    "hephaistos_suazo": 'au:"Suazo" AND all:"Dyson sphere"',
}
HEPH_KNOWN_ID = "2405.08657"  # Hephaistos II, recorded elsewhere in this repo

PROBES = {
    "G_var": re.compile(r"G[_\s]?var", re.I),
    "variab": re.compile(r"variab", re.I),
    "absorbing elements": re.compile(r"absorbing\s+elements", re.I),
}


def sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", flat) if s.strip()]


def probe_text(path: pathlib.Path, limit: int = 30) -> dict:
    """Record verbatim sentences around each probe string."""
    if not path.exists():
        return {"text_available": False}
    sents = sentences(path.read_text(errors="ignore"))
    out: dict = {"text_available": True, "n_sentences": len(sents)}
    for label, rx in PROBES.items():
        hits = [s for s in sents if rx.search(s)]
        out[label] = {"n": len(hits), "sentences": [s[:700] for s in hits[:limit]]}
    return out


def hephaistos_variability_cut() -> dict:
    """Find the Hephaistos papers and quote their variability rejection."""
    found: dict[str, str] = {}
    for name, q in HEPH_QUERIES.items():
        arxiv_query(name, q, max_results=30)
        for e in entries(OUT / f"arxiv_q_{name}.atom"):
            low = e["title"].lower()
            if "hephaistos" in low or ("dyson" in low and "suazo" in q.lower()):
                found[short_id(e["arxiv"])] = e["title"]
    if HEPH_KNOWN_ID not in {k.split("v")[0] for k in found}:
        found[HEPH_KNOWN_ID] = "Project Hephaistos II (ID recorded in-repo; unverified)"

    per_paper: dict[str, dict] = {}
    for i, (aid, title) in enumerate(sorted(found.items())[:4]):
        tag = f"heph{i}_{aid.replace('.', '_')}"
        txt = fulltext(tag, aid)
        rec = probe_text(txt) if txt else {"text_available": False}
        rec["title"] = title
        rec["arxiv"] = aid
        per_paper[aid] = rec

    any_var = any(p.get("variab", {}).get("n", 0) for p in per_paper.values()
                  if isinstance(p.get("variab"), dict))
    return {
        "papers_found": found,
        "per_paper": per_paper,
        "claim_under_test": ("Project Hephaistos explicitly REJECTS variable stars "
                             "from its Dyson-sphere search (e.g. a cut on the Gaia "
                             "G-band variability metric), which would mean the "
                             "nearest prior art discards exactly the population "
                             "VIGIL selects."),
        "probe_terms_found": any_var,
        "verdict_note": ("Read the quoted sentences before asserting the claim; "
                         "'variab' matching is necessary, not sufficient --- the "
                         "sentence has to be a SELECTION CUT, not a discussion."),
    }


# --------------------------------------------------------------------------
# 3. Decoy-aware novelty sweep
# --------------------------------------------------------------------------
QUERIES = {
    "mir_var_techno": 'all:"mid-infrared variability" AND all:"technosignature"',
    "dyson_var_waste_heat": 'all:"Dyson sphere" AND all:"variability" AND all:"waste heat"',
    "ir_var_optically_constant": 'all:"infrared variable" AND all:"optically constant"',
    "wise_var_seti": 'all:"WISE" AND all:"variability" AND all:"SETI"',
    "waste_heat_duty_cycle": 'all:"waste heat" AND all:"duty cycle" AND all:"megastructure"',
    "neowise_var_techno": 'all:"NEOWISE" AND all:"variability" AND all:"technosignature"',
    "megastructure_thermal_var": 'all:"megastructure" AND all:"thermal" AND all:"variability"',
    # phrasings of the conjunction itself
    "no_optical_counterpart_var": 'all:"no optical variability" AND all:"infrared"',
    "ir_var_optical_stable": 'all:"infrared" AND all:"variable" AND all:"optically stable"',
    "variable_ir_excess": 'all:"variable infrared excess" AND cat:astro-ph.SR',
    "dyson_variability": 'all:"Dyson" AND all:"variability"',
    "techno_time_domain_ir": 'all:"technosignature" AND all:"time-domain" AND all:"infrared"',
    # the four decoys, fetched deliberately so the scan can recognise them
    "decoy_yso_mir_var": 'all:"mid-infrared variability" AND all:"young stellar object"',
    "decoy_extreme_debris": 'all:"extreme debris disk" AND all:"variability"',
    "decoy_agn_mir_var": 'all:"mid-infrared variability" AND cat:astro-ph.GA',
    "decoy_optical_megastructure": 'all:"megastructure" AND all:"transit"',
    "decoy_boyajian": 'all:"KIC 8462852" OR all:"Boyajian"',
}

MIR = re.compile(r"mid-?infrared|mid-?IR|\bWISE\b|NEOWISE|unTimely|unWISE|CatWISE|"
                 r"\bW1\b|\bW2\b|3\.4\s*(?:micron|um|μm)|4\.6\s*(?:micron|um|μm)|"
                 r"12\s*(?:micron|um|μm)|22\s*(?:micron|um|μm)|\bSpitzer\b|\bIRAC\b", re.I)
VAR = re.compile(r"variab|variable|light\s*curve|photometric monitoring|flux change|"
                 r"dimming|brightening|time-?domain|epoch photometry", re.I)
OPT_CONST = re.compile(r"optically\s+(?:constant|stable|quiescent|unchanged|steady)|"
                       r"no\s+optical\s+variab|flat\s+optical|constant\s+in\s+the\s+optical|"
                       r"absence\s+of\s+optical\s+variab|without\s+optical\s+variab|"
                       r"optical(?:ly)?\s+(?:light\s*curves?\s+)?(?:were|are|was|is)?\s*"
                       r"(?:remained\s+)?(?:flat|constant|stable)", re.I)
TECHNO = re.compile(r"technosignature|techno-?signature|Dyson|SETI|megastructure|"
                    r"extraterrestrial|Kardashev|alien|waste heat", re.I)

DECOYS = {
    "yso_protostar": re.compile(r"young stellar object|\bYSO\b|protostar|pre-?main-?sequence|"
                                r"T Tauri|\bclass\s+[0I]{1,2}\b|dipper|UX Ori|FU Ori|EX Lup|"
                                r"star-?forming region|embedded protostellar", re.I),
    "extreme_debris_disk": re.compile(r"debris disk|debris disc|extreme debris|"
                                      r"collisional (?:cascade|avalanche)|giant impact|"
                                      r"circumstellar dust|warm dust|zodiacal", re.I),
    "agn_blazar": re.compile(r"\bAGN\b|active galactic|blazar|quasar|\bQSO\b|Seyfert|"
                             r"tidal disruption|dust echo|reverberation|accretion disk", re.I),
    "optical_transit_megastructure": re.compile(r"KIC\s*8462852|Boyajian|Tabby|transit(?:ing)?\s+"
                                                r"(?:megastructure|alien)|Kepler light curve|"
                                                r"aperiodic dip|dipping star", re.I),
}


def scan() -> dict:
    """Score every fetched abstract on the four-way conjunction, tag the decoys."""
    n_abs = 0
    rows: list[dict] = []
    for p in sorted(OUT.glob("arxiv_*.atom")):
        for e in entries(p):
            n_abs += 1
            blob = f"{e['title']} {e['summary']}"
            groups = {
                "mid_infrared": bool(MIR.search(blob)),
                "variability": bool(VAR.search(blob)),
                "optically_constant": bool(OPT_CONST.search(blob)),
                "technosignature": bool(TECHNO.search(blob)),
            }
            score = sum(groups.values())
            if score < 3:
                continue
            tags = [k for k, rx in DECOYS.items() if rx.search(blob)]
            rows.append({
                "arxiv": short_id(e["arxiv"]),
                "title": e["title"],
                "concept_groups": groups,
                "score": score,
                "decoys": tags,
                "source_query": p.name,
                "abstract": e["summary"][:1500],
            })
    # de-duplicate on arXiv id, keeping the highest-scoring appearance
    byid: dict[str, dict] = {}
    for r in rows:
        cur = byid.get(r["arxiv"])
        if cur is None or r["score"] > cur["score"]:
            byid[r["arxiv"]] = r
    rows = sorted(byid.values(), key=lambda r: -r["score"])

    full = [r for r in rows if r["score"] == 4]
    genuine = [r for r in full if not r["decoys"]]
    decoyed = [r for r in full if r["decoys"]]
    partial = [r for r in rows if r["score"] == 3 and not r["decoys"]
               and r["concept_groups"]["technosignature"]]

    counts: dict[str, int] = {k: 0 for k in DECOYS}
    for r in rows:
        for t in r["decoys"]:
            counts[t] += 1

    out = {
        "n_abstracts_scanned": n_abs,
        "n_scored_3_or_4": len(rows),
        "n_full_conjunction": len(full),
        "n_genuine_prior_art_hits": len(genuine),
        "genuine_prior_art": genuine,
        "full_conjunction_but_decoyed": decoyed,
        "near_miss_partial_hits": partial,
        "decoy_tag_counts": counts,
        "target_concept": ("(mid-infrared OR WISE OR NEOWISE) AND (variability OR "
                           "variable) AND (optically constant OR no optical "
                           "variability) AND (technosignature OR Dyson OR SETI OR "
                           "megastructure)"),
        "interpretation": (
            "A GENUINE prior-art hit is an abstract matching all four concept "
            "groups and carrying none of the four decoy tags. The decoys are NOT "
            "prior art: (a) yso_protostar --- mid-IR variability of young stellar "
            "objects is ordinary accretion/extinction astrophysics; (b) "
            "extreme_debris_disk --- a real and large literature, and the physical "
            "CONFOUNDER this channel must reject, not a technosignature claim; "
            "(c) agn_blazar --- extragalactic mid-IR variability; (d) "
            "optical_transit_megastructure --- Boyajian's star and successors are "
            "OPTICAL transit searches, a different observable. "
            "near_miss_partial_hits are technosignature papers matching three of "
            "the four groups and are the closest thing to competition; read them "
            "individually before claiming the niche is unoccupied."),
    }
    (OUT / "concept_scan.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------
# 4. Data access routes for the unTimely tables
# --------------------------------------------------------------------------
URL_RX = re.compile(r"https?://[^\s<>\"')\]]+")
DOI_RX = re.compile(r"\b10\.\d{4,9}/[^\s<>\"')\],;]+")
ZENODO_RX = re.compile(r"zenodo\.org/[^\s<>\"')\]]+|10\.5281/zenodo\.\d+", re.I)
VIZIER_RX = re.compile(r"\bJ/[A-Za-z]+/\d+/\w+(?:/\w+)?")
DATALAB_RX = re.compile(r"(?:astro\s+data\s+lab|datalab\.noirlab|noirlab|data\s+lab)"
                        r"[^.\n]{0,160}", re.I)
TABLE_RX = re.compile(r"\b(?:untimely|unwise|catwise|allwise|neowise)[a-z0-9_]*\."
                      r"[a-z0-9_]+\b", re.I)


def strip_punct(s: str) -> str:
    return s.rstrip(".,;:)]}'\"")


def access_routes(label: str, text: str) -> dict:
    def uniq(seq: list[str]) -> list[str]:
        seen, out = set(), []
        for x in seq:
            x = strip_punct(x)
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    urls = uniq(URL_RX.findall(text))
    interesting = [u for u in urls if re.search(
        r"zenodo|vizier|cds|noirlab|datalab|dataverse|figshare|mast|irsa|ipac|"
        r"github|dataset|catalog|\.fits|\.csv|\.parquet|\.tar", u, re.I)]
    return {
        "label": label,
        "n_chars_scanned": len(text),
        "urls_data_like": interesting,
        "urls_all": urls[:200],
        "dois": uniq(DOI_RX.findall(text)),
        "zenodo": uniq(ZENODO_RX.findall(text)),
        "vizier_catalogues": uniq(VIZIER_RX.findall(text)),
        "datalab_mentions": [" ".join(m.split())[:300]
                             for m in uniq(DATALAB_RX.findall(text))][:20],
        "table_names": uniq(TABLE_RX.findall(text))[:40],
    }


def untimely_access(var_id: str | None) -> dict:
    """Access routes for the variable catalogue AND its parent unTimely Catalog."""
    routes: dict[str, dict] = {}

    # (a) the variable catalogue, if we have a believable ID
    if var_id:
        txt = fulltext("untimely_variable", var_id)
        blob = txt.read_text(errors="ignore") if txt and txt.exists() else ""
        abs_path = OUT / "arxiv_id_untimely_variable.atom"
        for e in entries(abs_path):
            blob += "\n" + e["summary"]
        routes["untimely_variable_catalogue"] = access_routes(
            f"unTimely variable catalogue (arXiv:{var_id})", blob)
    else:
        routes["untimely_variable_catalogue"] = {
            "label": "unTimely variable catalogue",
            "note": "no verified arXiv ID --- nothing fetched",
        }

    # (b) the parent unTimely Catalog (Meisner et al. 2023)
    arxiv_query("untimely_parent", 'all:"unTimely Catalog" AND all:"unWISE"', max_results=30)
    parent = None
    for e in entries(OUT / "arxiv_q_untimely_parent.atom"):
        if "untimely" in e["title"].lower():
            parent = e
            break
    if parent is None:
        arxiv_query("untimely_parent2", 'all:"unTimely" AND all:"time-domain" AND all:"unWISE"',
                    max_results=30)
        for e in entries(OUT / "arxiv_q_untimely_parent2.atom"):
            if "untimely" in e["title"].lower():
                parent = e
                break
    if parent:
        pid = short_id(parent["arxiv"])
        txt = fulltext("untimely_parent", pid)
        blob = (txt.read_text(errors="ignore") if txt and txt.exists() else "")
        blob += "\n" + parent["summary"]
        rec = access_routes(f"unTimely Catalog parent (arXiv:{pid})", blob)
        rec["arxiv"] = pid
        rec["title"] = parent["title"]
        routes["untimely_catalog_parent"] = rec
    else:
        routes["untimely_catalog_parent"] = {
            "label": "unTimely Catalog parent",
            "note": "not found on arXiv by title search",
        }

    routes["note"] = ("The variable catalogue is expected to be distributed "
                      "alongside the parent unTimely Catalog (Astro Data Lab / "
                      "unwise.me / Zenodo). If only the parent has a route, the "
                      "VIGIL input table has to be derived from the parent's "
                      "per-epoch coadd photometry.")
    (OUT / "data_access_routes.json").write_text(json.dumps(routes, indent=2))
    return routes


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    print("== 1. verify decisive citations BY TITLE ==")
    idver = verify_claimed_ids()
    fallback = untimely_title_search()
    (OUT / "id_verification.json").write_text(json.dumps(
        {"id_verification": idver, "untimely_title_search_fallback": fallback}, indent=2))

    untimely_rec = next(r for r in idver if r["name"] == "untimely_variable")
    untimely_ok = bool(untimely_rec["verified"])
    var_id: str | None = untimely_rec["arxiv_id"] if untimely_ok else None
    if not untimely_ok and fallback.get("best_guess_variable_catalogue"):
        var_id = fallback["best_guess_variable_catalogue"]["arxiv"]
        print(f"   claimed unTimely ID FAILED; title search suggests {var_id}")

    print("== 2. Hephaistos variability rejection ==")
    heph = hephaistos_variability_cut()

    print("== 3. novelty sweep ==")
    for name, q in QUERIES.items():
        print(f"-- {name}")
        arxiv_query(name, q)

    print("== 4. unTimely data access routes ==")
    routes = untimely_access(var_id)

    print("== OpenAlex context for the load-bearing papers ==")
    for name in ("moor_extreme_debris", "wright2020_dyson", "contardo_hogg"):
        rec = next(r for r in idver if r["name"] == name)
        if rec["verified"]:
            openalex(name, rec["fetched_title"], rec["arxiv_id"])

    print("== decoy-aware concept scan ==")
    res = scan()

    n_ok = sum(1 for s in STATUS if s["ok"])
    n_query_atoms = sum(1 for p in OUT.glob("arxiv_q_*.atom") if p.stat().st_size > 0)
    n_genuine = res["n_genuine_prior_art_hits"]

    if n_ok == 0 or n_query_atoms == 0 or res["n_abstracts_scanned"] == 0:
        verdict = "UNDETERMINED_FETCH_FAILED"
    elif n_genuine == 0 and not res["near_miss_partial_hits"]:
        verdict = "UNOCCUPIED"
    elif n_genuine == 0:
        verdict = "PARTIALLY_OCCUPIED"
    elif n_genuine <= 2:
        verdict = "PARTIALLY_OCCUPIED"
    else:
        verdict = "OCCUPIED"

    failed_ids = [r["arxiv_id"] for r in idver if not r["verified"]]
    summary = {
        "VERIFICATION_HEADLINE": (
            "ALL CLAIMED arXiv IDs VERIFIED BY TITLE" if not failed_ids else
            "*** CITATION VERIFICATION FAILED for " + ", ".join(failed_ids) +
            " --- these IDs must NOT be cited until replaced ***"),
        "untimely_id_verified": untimely_ok,
        "untimely_id_used": var_id,
        "untimely_title_search_fallback": fallback.get("best_guess_variable_catalogue"),
        "id_verification": idver,
        "novelty_verdict": verdict,
        "n_genuine_prior_art_hits": n_genuine,
        "n_near_miss_partial_hits": len(res["near_miss_partial_hits"]),
        "n_abstracts_scanned": res["n_abstracts_scanned"],
        "decoy_tag_counts": res["decoy_tag_counts"],
        "hephaistos_variability_cut": heph,
        "data_access_summary": {
            k: {kk: vv for kk, vv in v.items() if kk != "urls_all"}
            for k, v in routes.items() if isinstance(v, dict)
        },
        "verdict_semantics": (
            "UNOCCUPIED = the fetches worked and nothing matched the conjunction; "
            "PARTIALLY_OCCUPIED = adjacent work exists (near misses or 1-2 genuine "
            "hits) and must be read before any novelty claim; OCCUPIED = the idea "
            "is published; UNDETERMINED_FETCH_FAILED = we could not look, which is "
            "NOT the same statement as finding nothing."),
        "n_urls": len(STATUS),
        "n_ok": n_ok,
        "n_failed": len(STATUS) - n_ok,
        "status": STATUS,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + summary["VERIFICATION_HEADLINE"])
    print(f"novelty_verdict = {verdict}  "
          f"(genuine prior art: {n_genuine}, near misses: "
          f"{len(res['near_miss_partial_hits'])})")
    print(f"{n_ok}/{len(STATUS)} fetches ok -> {OUT}")


if __name__ == "__main__":
    main()
