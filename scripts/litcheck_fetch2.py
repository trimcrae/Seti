#!/usr/bin/env python3
"""HERDSMAN literature novelty check, pass 2: targets discovered during pass 1.

Adds: Vidal's stellivore/spider-engine kinematic line (arXiv + Zenodo),
Kezerashvili/Matloff/Long JBIS 2021 anomalous-stellar-acceleration (via the
Centauri Dreams summary; JBIS itself is paywalled), LaForge 2023, the
SETI-in-20XX annual reviews, Voros 2014, Loeb 2018, and citation lists.
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/litcheck")
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Seti-litcheck/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []


def get(url: str, dest: pathlib.Path, tries: int = 4, pause: float = 3.0) -> bool:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:44s} {len(data):8d}B")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {url} -> {e}")
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    return False


ARXIV = "http://export.arxiv.org/api/query?"

IDS2 = [
    "2411.05038",  # Vidal 2024 - Spider stellar engine (JBIS)
    "2306.11989",  # LaForge 2023 - new kind of stellar engine
    "2410.08253",  # SETI in 2022 (annual review)
    "1412.4011",   # Voros 2014 - galactic-scale macro-engineering
    "1607.06114",  # Life, Intelligence and Multiverse (Lanier star-rearrangement mention)
    "2504.21151",  # Lacki 2025 - Kardashev II megaswarms
    "2606.08373",  # 2026 - Dust to Dust: passive technosignatures / relics
    "1907.07830",  # Technosignatures in Transit
    "2309.06564",  # Dyson spheres thermodynamics observational consequences
    "2103.01536",  # Concepts for future missions to search for technosignatures
    "1812.08681",  # NASA Technosignatures Workshop report 2018
    "1806.07170",  # Loeb 2018 - Securing fuel for our frigid cosmic future
    "2603.00203",  # unknown 2026 paper surfaced in searches - identify
    "2605.06072",  # re-fetch (phase-space crystallization) in case pass 1 missed
    "2605.21093",  # re-fetch review
    "2607.07781",  # re-fetch J-harvesting
]
get(ARXIV + urllib.parse.urlencode({"id_list": ",".join(IDS2), "max_results": len(IDS2)}),
    OUT / "arxiv_ids2.atom")
time.sleep(3)

Q2 = {
    "seti_in_2023": 'ti:"SETI in 2023"',
    "seti_in_2024": 'ti:"SETI in 2024"',
    "seti_in_2025": 'ti:"SETI in 2025"',
    "parenago": 'all:"Parenago"',
    "volitional": 'all:"volitional star" OR all:"stellar volition" OR all:"star consciousness"',
    "anomacc_engine": 'abs:"anomalous acceleration" AND abs:"stellar engine"',
    "laforge": 'all:"stellar engine" AND au:LaForge',
    "vidal_all": 'au:"Vidal_C" AND (all:stellivore OR all:"stellar engine" OR all:pulsar)',
    "herd": 'abs:"herd" AND abs:stars AND (abs:SETI OR abs:technosignature OR abs:civilization)',
    "tug_boat": 'abs:"stellar propulsion" AND (abs:SETI OR abs:technosignature OR abs:civilization)',
}
for name, q in Q2.items():
    get(ARXIV + urllib.parse.urlencode(
        {"search_query": q, "max_results": "40", "sortBy": "relevance"}),
        OUT / f"arxiv_q2_{name}.atom")
    time.sleep(3.2)

# Zenodo records for Vidal's Gaia DR3 stellivore kinematic tests
get("https://zenodo.org/api/records/6757507", OUT / "zenodo_6757507.json")
time.sleep(1.5)
get("https://zenodo.org/api/records/8052918", OUT / "zenodo_8052918.json")
time.sleep(1.5)

# Centauri Dreams summaries (JBIS content is paywalled; these carry the substance)
CD = {
    "cd_anomalous_accel": "https://www.centauri-dreams.org/2021/08/20/how-to-explain-unusual-stellar-acceleration/",
    "cd_new_stellar_engine": "https://www.centauri-dreams.org/2023/10/13/seti-a-new-kind-of-stellar-engine/",
    "cd_cosmic_engineering": "https://www.centauri-dreams.org/2018/06/21/cosmic-engineering-and-the-movement-of-stars/",
    "cd_evolving_strategies": "https://www.centauri-dreams.org/2026/05/30/evolving-strategies-in-the-search-for-extraterrestrial-civilizations/",
}
for name, url in CD.items():
    get(url, OUT / f"{name}.html")
    time.sleep(2)

# Semantic Scholar: Vidal spider engine + Loeb 2018 citations; Matloff resolution
S2 = "https://api.semanticscholar.org/graph/v1/paper/"
S2FIELDS = "title,year,abstract,externalIds,venue,authors"
for name, ident in {
    "vidal_spider": "arXiv:2411.05038",
    "loeb_fuel18": "arXiv:1806.07170",
}.items():
    get(S2 + urllib.parse.quote(ident) + f"?fields={S2FIELDS},citationCount",
        OUT / f"s2_{name}.json", pause=6.0)
    time.sleep(2.5)
    get(S2 + urllib.parse.quote(ident) + f"/citations?fields={S2FIELDS}&limit=300",
        OUT / f"s2_{name}_citations.json", pause=6.0)
    time.sleep(2.5)

for name, q in {
    "kezerashvili_jbis": "Anomalous Stellar Acceleration Causes and Consequences",
    "matloff_conscious": "Star consciousness an alternative to dark matter",
    "matloff_parenago": "Parenago discontinuity volitional",
    "laforge_engine": "LaForge stellar engine",
    "hooper_moving": "Dyson spheres accelerate stars civilization dark energy",
}.items():
    get(S2 + "search?" + urllib.parse.urlencode(
        {"query": q, "fields": S2FIELDS, "limit": "5"}),
        OUT / f"s2_search2_{name}.json", pause=6.0)
    time.sleep(2.5)

(OUT / "summary2.json").write_text(json.dumps(STATUS, indent=2))
print(f"\n{sum(1 for s in STATUS if s.get('ok'))}/{len(STATUS)} fetches succeeded")
