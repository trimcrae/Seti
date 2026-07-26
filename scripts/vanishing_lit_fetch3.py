#!/usr/bin/env python3
"""Literature survey fetcher, pass 3: gap-fill for specific IDs and critiques."""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/vanishlit3")
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Seti-vanishlit/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []


def get(url: str, dest: pathlib.Path, tries: int = 3, pause: float = 3.0) -> bool:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:44s} {len(data):8d}B", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {url} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url}", flush=True)
    return False


ARXIV = "http://export.arxiv.org/api/query?"

IDS = [
    "2303.00010",  # Search for SN Progenitor Stars with ZTF and LSST
    "2305.11936",  # Kochanek 2024 Transients obscured by dusty discs
    "2106.11780",  # Villarroel+ nine simultaneous transients 1950
    "2507.15896",  # Villarroel/Solano/Marcy image profiles of POSS transients
    "2204.06091",  # Villarroel+ geosynchronous high-albedo objects
    "2401.14471",  # Lucas+ most variable VVV sources / dipping giants
    "1601.00135",  # Rodriguez+ extreme eps Aur analogue (69 yr)
    "2510.01177",  # Vanishing acts: DSNB constraint on BH formation
    "2509.16308",  # Antoni, Jiang & Quataert low-energy explosions
    "1710.01735",  # Fernandez+ mass ejection in failed SNe vs progenitor
    "2604.04810",  # Hayes 2026 independent POSS-I vanishing pipeline
    "2604.13711",  # Beasor+ 2026 SN 2010da surviving star hidden by dust
    "1304.1539",   # Piro 2013 Taking the Un out of Unnovae
    "0908.0701",   # Fryer+ 2009 spectra and light curves of failed SNe
    "2001.07216",  # Kochanek 2020 On the Red Supergiant Problem
]
CH = 8
for i in range(0, len(IDS), CH):
    chunk = IDS[i:i + CH]
    get(ARXIV + urllib.parse.urlencode({"id_list": ",".join(chunk),
                                        "max_results": len(chunk)}),
        OUT / f"arxiv_ids3_{i//CH}.atom")
    time.sleep(3.2)

QUERIES = {
    "villarroel2016": 'ti:"Our Sky Now and Then" OR abs:"lost stars" AND abs:"advanced extraterrestrial"',
    "schaefer_critique": 'au:Schaefer AND (abs:vanishing OR abs:"Palomar" OR abs:plate)',
    "vasco_critique": 'abs:"vanishing sources" AND (abs:artefact OR abs:artifact OR abs:contamination OR abs:plate defects)',
    "poss_defects": 'abs:"photographic plate" AND (abs:defects OR abs:artefacts) AND abs:transient',
    "ztf_progenitor_search": 'abs:ZTF AND abs:progenitor AND abs:"supernova" AND abs:search',
    "lsst_failed_sn": 'abs:"Rubin" OR abs:LSST AND abs:"failed supernova"',
    "neowise_transient_search": 'abs:NEOWISE AND abs:transient AND (abs:M31 OR abs:"nearby galaxies")',
    "disappearing_survey_future": 'abs:"disappearing star" OR abs:"vanishing star" AND abs:survey',
}
for name, q in QUERIES.items():
    get(ARXIV + urllib.parse.urlencode(
        {"search_query": q, "max_results": "40", "sortBy": "relevance"}),
        OUT / f"arxiv_q3_{name}.atom")
    time.sleep(3.2)

(OUT / "summary.json").write_text(json.dumps(STATUS, indent=2))
print(f"\n{sum(1 for s in STATUS if s.get('ok'))}/{len(STATUS)} fetches succeeded")
