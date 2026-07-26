#!/usr/bin/env python3
"""Literature survey fetcher, pass 2: breadth sweep.

Covers the sub-topics pass 1 did not: VASCO / wide-field vanishing searches,
SN impostors that faded (SN 2008S, NGC 300-OT), PHL 293B rebuttals, SPIRITS,
M31-2014-DS1, failed-SN theory, and recurrent-obscuration analogues (RCB,
eps Aur, Gaia17bpp).

Outputs under results/vanishlit2/.
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/vanishlit2")
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
            print(f"OK   {dest.name:46s} {len(data):8d}B", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {url} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url}", flush=True)
    return False


ARXIV = "http://export.arxiv.org/api/query?"

# --- 1. arXiv searches: the breadth sweep ------------------------------------
QUERIES = {
    # --- wide-field / catalogue-scale vanishing searches (topic 5) ---
    "vasco1": 'all:"vanishing and appearing sources"',
    "vasco2": 'all:VASCO AND all:sources',
    "villarroel": 'au:Villarroel AND (all:vanishing OR all:transients OR all:"sky surveys")',
    "usno_sdss_vanish": 'abs:"USNO-B" AND (abs:vanish OR abs:disappear OR abs:missing)',
    "poss_panstarrs_vanish": 'abs:"POSS" AND abs:"Pan-STARRS" AND (abs:vanish OR abs:transient)',
    "plate_transients_1950": 'abs:"photographic plate" AND abs:transients AND abs:1950',
    "missing_star_xmatch": 'abs:"missing" AND abs:"cross-match" AND abs:catalog AND abs:star',
    "faded_permanently": 'abs:"faded" AND abs:"never" AND abs:recover AND abs:star',
    # --- SN impostors / faded eruptions (topic 3) ---
    "sn2008s": 'all:"SN 2008S"',
    "ngc300ot": 'all:"NGC 300 OT" OR all:"NGC300-OT" OR all:"NGC 300 transient"',
    "impostor_survivor": 'abs:"supernova impostor" AND (abs:survivor OR abs:survived OR abs:progenitor)',
    "intermediate_luminosity_transient": 'abs:"intermediate luminosity" AND abs:transient AND abs:dust',
    "sn2009ip": 'all:"SN 2009ip"',
    "ugc2773": 'all:"UGC 2773"',
    "sn1997bs": 'all:"SN 1997bs"',
    "sn2010da": 'all:"SN 2010da"',
    "eta_car_analog": 'abs:"giant eruption" AND abs:"luminous blue variable" AND abs:dust',
    # --- PHL 293B rebuttal literature ---
    "phl293b_all": 'all:"PHL 293B"',
    "kinman": 'all:"Kinman" AND abs:dwarf',
    "burke_vanishing": 'au:Burke AND (abs:vanishing OR abs:"PHL 293")',
    # --- M31-2014-DS1 and M31/M33 disappearing stars ---
    "m31ds1": 'all:"M31-2014-DS1" OR all:"M31 2014 DS1"',
    "m31_disappear": 'abs:"M31" AND (abs:"disappearing" OR abs:"vanished") AND abs:star',
    "m33_disappear": 'abs:"M33" AND abs:"disappear"',
    # --- SPIRITS / IR transient surveys ---
    "spirits": 'all:SPIRITS AND (all:Spitzer OR all:"infrared transients")',
    "jencson": 'au:Jencson AND abs:infrared AND abs:transient',
    "kasliwal_ir": 'au:Kasliwal AND abs:SPIRITS',
    "obscured_sn_survey": 'abs:"dust-obscured" AND abs:supernova AND abs:survey',
    # --- theory: what a failed SN looks like, IR persistence (topic 6) ---
    "nadezhin": 'all:"Nadezhin" OR abs:"neutrino mass loss" AND abs:"red supergiant"',
    "lovegrove_woosley": 'au:Lovegrove AND au:Woosley',
    "very_low_energy_sn": 'abs:"very low energy" AND abs:supernova AND abs:"shock breakout"',
    "failed_sn_signature": 'abs:"failed supernova" AND (abs:signature OR abs:"light curve" OR abs:emission)',
    "bh_formation_disappearance": 'abs:"black hole formation" AND abs:"disappearance"',
    "fallback_accretion_dust": 'abs:"fallback accretion" AND abs:"black hole" AND abs:luminosity',
    "sukhbold_explodability": 'au:Sukhbold AND abs:"core-collapse"',
    "rsg_problem2": 'abs:"red supergiant problem" OR abs:"missing red supergiants"',
    "dust_shell_evolution": 'abs:"dust shell" AND abs:"expanding" AND abs:"infrared" AND abs:"supergiant"',
    "obscured_progenitor_ir": 'abs:"circumstellar dust" AND abs:progenitor AND abs:"infrared excess"',
    # --- vanished-and-returned / recurrent obscuration (topic 7) ---
    "gaia17bpp_all": 'all:"Gaia17bpp"',
    "long_dimming_binary": 'abs:"long-duration" AND abs:dimming AND (abs:binary OR abs:disk)',
    "rcb_stars": 'abs:"R Coronae Borealis" AND abs:"dust"',
    "eps_aur": 'abs:"epsilon Aurigae"',
    "tyc2505": 'all:"TYC 2505-672-1" OR abs:"longest period eclipsing"',
    "asassn_dipper": 'abs:"ASAS-SN" AND abs:"dimming"',
    "boyajian_analogues": 'abs:"Boyajian" AND abs:dimming',
    "occultation_disk_star": 'abs:"occulting" AND abs:disk AND abs:star AND abs:eclipse',
    "betelgeuse_dimming": 'abs:Betelgeuse AND abs:"Great Dimming"',
    "vvv_dipper": 'abs:"VVV" AND (abs:"deep dimming" OR abs:"eclipsing") AND abs:giant',
}
for name, q in QUERIES.items():
    get(ARXIV + urllib.parse.urlencode(
        {"search_query": q, "max_results": "40", "sortBy": "relevance"}),
        OUT / f"arxiv_q_{name}.atom")
    time.sleep(3.2)

# --- 2. Specific IDs worth pulling verbatim ----------------------------------
IDS = [
    "1006.4177",   # Kochanek+ 2010? impostor/dust - verify
    "0809.0510",   # Prieto+ 2008 SN 2008S progenitor - verify
    "0811.3929",   # Thompson+ 2009 new class of transients - verify
    "1109.2497",   # Kochanek 2011 dust and SN 2008S - verify
    "2306.11800",  # possible SPIRITS/failed SN - verify
    "1010.3799",   # Smartt red supergiant problem review - verify
    "1502.05408",  # Smartt 2015 progenitors review - verify
    "1512.00021",  # Adams+ 2016? - verify
    "2404.19017",  # De+ 2024 M31-2014-DS1 - verify
    "2211.16234",  # Tzanidakis+ Gaia17bpp - verify
]
CH = 5
for i in range(0, len(IDS), CH):
    chunk = IDS[i:i + CH]
    get(ARXIV + urllib.parse.urlencode({"id_list": ",".join(chunk),
                                        "max_results": len(chunk)}),
        OUT / f"arxiv_ids2_{i//CH}.atom")
    time.sleep(3.2)

# --- 3. OpenAlex: citation lists + resolution (no key, reliable) -------------
OA = "https://api.openalex.org/works"
OA_SEARCHES = {
    "allan2020": "possible disappearance of a massive star low-metallicity galaxy PHL 293B",
    "adams2017": "search for failed supernovae Large Binocular Telescope confirmation disappearing star",
    "beasor2024": "JWST luminous infrared source failed supernova candidate N6946-BH1",
    "kochanek2024": "mid-infrared counterpart N6946-BH1 failed supernovae",
    "neustadt2021": "failed supernovae Large Binocular Telescope failed SN fraction 11 yr",
    "kochanek2008": "Survey About Nothing monitoring million supergiants failed supernovae",
    "villarroel2016": "our sky now and then vanishing sources USNO SDSS",
    "villarroel2020": "VASCO citizen science project vanishing appearing sources",
    "m31ds1": "M31-2014-DS1 disappearing dusty star failed supernova",
    "gaia17bpp": "Gaia17bpp long duration dimming event giant star",
}
for name, q in OA_SEARCHES.items():
    get(OA + "?" + urllib.parse.urlencode({"search": q, "per-page": "8"}),
        OUT / f"oa_{name}.json", pause=2.0)
    time.sleep(1.5)

# OpenAlex: works CITING the key papers (catches 2024-2026 follow-ups).
# DOIs are the stable handles.
CITING = {
    "adams2017_cited_by": "10.1093/mnras/stx816",
    "neustadt2021_cited_by": "10.1093/mnras/stab2605",
    "beasor2024_cited_by": "10.3847/1538-4357/ad21fa",
    "kochanek2024_cited_by": "10.3847/1538-4357/ad18d7",
    "allan2020_cited_by": "10.1093/mnras/staa1629",
    "kochanek2008_cited_by": "10.1086/590053",
}
for name, doi in CITING.items():
    get(OA + "?" + urllib.parse.urlencode(
        {"filter": f"cites:doi:{doi}", "per-page": "100",
         "select": "id,doi,title,publication_year,authorships"}),
        OUT / f"oa_{name}.json", pause=2.0)
    time.sleep(1.5)

# --- 4. Full-text HTML of argument-critical / recent papers ------------------
for aid in ["2604.05019", "2601.14497", "2404.19017", "2309.16121", "2310.01514"]:
    if not get(f"https://arxiv.org/html/{aid}v1", OUT / f"html_{aid}.html", tries=2):
        get(f"https://arxiv.org/html/{aid}", OUT / f"html_{aid}.html", tries=2)
    time.sleep(3)

(OUT / "summary.json").write_text(json.dumps(STATUS, indent=2))
n_ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n{n_ok}/{len(STATUS)} fetches succeeded")
