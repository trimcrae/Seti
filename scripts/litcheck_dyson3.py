#!/usr/bin/env python3
"""Dyson/IR-excess literature survey, pass 3: exhaustive OpenAlex + Crossref recall.

arXiv misses a large slice of Dysonian SETI, which is published in the
International Journal of Astrobiology, Acta Astronautica, JBIS, Research in
Astronomy and Astrophysics and similar venues that do not always post preprints.
OpenAlex indexes title+abstract for essentially all of them, and supports cursor
paging, so this pass walks EVERY matching record rather than a relevance top-N.

Outputs under results/litcheck_dyson/oa3/:
  oa_<slug>_p<N>.json   - each page of each OpenAlex query
  cr_<slug>.json        - Crossref bibliographic search
  summary3.json         - fetch status
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/litcheck_dyson/oa3")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-litcheck-dyson3/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []


def fetch(url: str, dest: pathlib.Path, tries: int = 3, pause: float = 3.0):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:38s} {len(data):8d}B", flush=True)
            return json.loads(data.decode("utf-8", errors="replace"))
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {url} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url}", flush=True)
    return None


OA = "https://api.openalex.org/works"

# title_and_abstract.search gives precision; plain `search` also hits full text
# for some records. Walk every page with a cursor.
OA_FILTERS = {
    "dyson_sphere": 'title_and_abstract.search:"dyson sphere"',
    "dyson_spheres": 'title_and_abstract.search:"dyson spheres"',
    "dyson_swarm": 'title_and_abstract.search:"dyson swarm"',
    "dysonian": 'title_and_abstract.search:dysonian',
    "megastructure": 'title_and_abstract.search:megastructure',
    "kardashev": 'title_and_abstract.search:kardashev',
    "technosignature_ir": 'title_and_abstract.search:technosignature AND title_and_abstract.search:infrared',
    "waste_heat_civ": 'title_and_abstract.search:"waste heat" AND title_and_abstract.search:civilization',
    "irexcess_wise": 'title_and_abstract.search:"infrared excess" AND title_and_abstract.search:WISE',
    "debrisdisk_wise": 'title_and_abstract.search:"debris disk" AND title_and_abstract.search:WISE',
    "debrisdisc_wise": 'title_and_abstract.search:"debris disc" AND title_and_abstract.search:WISE',
    "disk_detective": 'title_and_abstract.search:"disk detective"',
    "allwise_excess": 'title_and_abstract.search:AllWISE AND title_and_abstract.search:excess',
    "unwise_neowise": 'title_and_abstract.search:unWISE OR title_and_abstract.search:CatWISE',
    "irexcess_census": 'title_and_abstract.search:"infrared excess" AND title_and_abstract.search:catalog',
    "extraterrestrial_ir": 'title_and_abstract.search:extraterrestrial AND title_and_abstract.search:"infrared"',
}

for slug, filt in OA_FILTERS.items():
    cursor = "*"
    page = 0
    while cursor and page < 12:
        url = OA + "?" + urllib.parse.urlencode({
            "filter": filt, "per-page": "200", "cursor": cursor,
            "select": "id,doi,title,publication_year,authorships,primary_location,"
                      "cited_by_count,type,ids",
            "mailto": "trimcrae@gmail.com"})
        d = fetch(url, OUT / f"oa_{slug}_p{page}.json")
        if not d:
            break
        n = len(d.get("results") or [])
        cursor = (d.get("meta") or {}).get("next_cursor")
        print(f"     {slug} page {page}: {n} results, total="
              f"{(d.get('meta') or {}).get('count')}", flush=True)
        page += 1
        if n == 0:
            break
        time.sleep(1.2)

# Crossref: catches records OpenAlex indexes late, and gives clean journal refs.
CR = "https://api.crossref.org/works"
CR_QUERIES = {
    "dyson_sphere": "Dyson sphere search infrared",
    "dyson_megastructure": "Dyson sphere megastructure technosignature",
    "waste_heat_seti": "waste heat search extraterrestrial civilizations infrared",
    "wise_debris": "WISE debris disk infrared excess survey main-sequence",
    "irexcess_census": "census infrared excess stars nearby WISE",
    "dysonian_seti": "Dysonian SETI artefact search",
}
for slug, q in CR_QUERIES.items():
    fetch(CR + "?" + urllib.parse.urlencode(
        {"query.bibliographic": q, "rows": "100",
         "select": "DOI,title,author,issued,container-title,abstract,"
                   "is-referenced-by-count",
         "mailto": "trimcrae@gmail.com"}),
        OUT / f"cr_{slug}.json")
    time.sleep(1.5)

(OUT.parent / "summary3.json").write_text(json.dumps(STATUS, indent=2))
n_ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n{n_ok}/{len(STATUS)} fetches succeeded")
