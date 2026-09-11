#!/usr/bin/env python3
"""GOO contact-graph literature fetch, run on the GitHub runner.

The contact-graph section of paper/goo/ asks how long it takes for every
system in the Galaxy to be touched by material from a system that was itself
touched: impact ejecta, interstellar objects, free-floating planets, stellar
encounters and Oort-cloud exchange, interstellar dust, supernova ejecta,
birth-cluster exchange.  Every rate in config/goo.yaml's `contact` block must
come from a paper, so this script fetches the record (verbatim arXiv abstracts;
Crossref for the rest) and extracts the sentences that carry numbers for the
quantities the model needs.

Reuses the fetch and parse helpers of goolit_fetch.py; writes to
results/goolit_contact/ (GOOLIT_OUT is set by the workflow).

Outputs:
  arxiv_id_<key>.atom / arxiv_t_<key>.atom / crossref_<key>.json  verbatim
  numbers.json     for each reference: the abstract sentences containing a
                   number next to a rate / density / cross-section / fraction word
  verification.json, REPORT.md, summary.json  as in goolit_fetch.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import sys

os.environ.setdefault("GOOLIT_OUT", "results/goolit_contact")
_spec = importlib.util.spec_from_file_location("goolit_fetch", pathlib.Path(__file__).with_name("goolit_fetch.py"))
G = importlib.util.module_from_spec(_spec)
sys.modules["goolit_fetch"] = G
_spec.loader.exec_module(G)
OUT = G.OUT

REFS: dict[str, dict] = {
    # --- interstellar objects: number density, origin, capture, impacts ---
    "Do2018": {"mode": "arxiv_id", "id": "1801.02821", "expect": "Interstellar Interlopers: Number Density and Origin of 'Oumuamua-like Objects"},
    "JewittSeligman2023": {"mode": "arxiv_id", "id": "2209.08182", "expect": "The Interstellar Interlopers"},
    "PortegiesZwart2018": {"mode": "arxiv_id", "id": "1711.03558", "expect": "The origin of interstellar asteroidal objects like 1I/2017 U1 'Oumuamua"},
    "LingamLoeb2018": {"mode": "arxiv_id", "id": "1801.10254", "expect": "Implications of Captured Interstellar Objects for Panspermia and Extraterrestrial Life"},
    "SirajLoeb2019trapped": {"mode": "arxiv_id", "id": "1811.09632", "expect": "Identifying Interstellar Objects Trapped in the Solar System through Their Orbital Parameters"},
    "HandsDehnen2020": {"mode": "arxiv_title", "title": "Capture of interstellar objects: a source of long-period comets", "expect": "Capture of interstellar objects: a source of long-period comets"},
    "Napier2021capture": {"mode": "arxiv_title", "title": "On the Capture of Interstellar Objects by our Solar System", "expect": "On the Capture of Interstellar Objects by our Solar System"},
    "Dehnen2022": {"mode": "arxiv_title", "title": "Capture of interstellar objects II: by the Solar system", "expect": "Capture of interstellar objects II"},
    "SirajLoeb2022meteors": {"mode": "arxiv_title", "title": "Interstellar Meteors are Outliers in Material Strength", "expect": "Interstellar Meteors are Outliers in Material Strength"},
    "MoroMartin2022": {"mode": "arxiv_title", "title": "Interstellar Interlopers: Number Density and Origin of Oumuamua-like Objects", "expect": "Interstellar"},
    # --- rocks between systems ---
    "AdamsNapier2022": {"mode": "arxiv_title", "title": "Transfer of Rocks Between Planetary Systems: Panspermia Revisited", "expect": "Transfer of Rocks Between Planetary Systems: Panspermia Revisited"},
    "Belbruno2012": {"mode": "arxiv_id", "id": "1205.1059", "expect": "Chaotic Exchange of Solid Material between Planetary Systems: Implications for Lithopanspermia"},
    "Worth2013": {"mode": "arxiv_title", "title": "Seeding Life on the Moons of the Outer Planets via Lithopanspermia", "expect": "Seeding Life on the Moons of the Outer Planets via Lithopanspermia"},
    "AdamsSpergel2005": {"mode": "arxiv_id", "id": "astro-ph/0504648", "expect": "Lithopanspermia in Star Forming Clusters"},
    "Ginsburg2018": {"mode": "arxiv_id", "id": "1810.04307", "expect": "Galactic Panspermia"},
    "Krijt2017": {"mode": "arxiv_title", "title": "Fast litho-panspermia in the habitable zone of the TRAPPIST-1 system", "expect": "Fast litho-panspermia in the habitable zone of the TRAPPIST-1 system"},
    "SirajLoeb2020grazing": {"mode": "arxiv_id", "id": "2001.02235", "expect": "Possible Transfer of Life by Earth-Grazing Objects to Exoplanetary Systems"},
    "Melosh2003": {"mode": "crossref", "q": "Melosh 2003 Exchange of meteorites (and impact ejecta) between stellar systems Astrobiology", "expect": "Exchange of Meteorites (and Impact Ejecta) between Stellar Systems"},
    "Napier2004": {"mode": "crossref", "q": "Napier 2004 A mechanism for interstellar panspermia Monthly Notices", "expect": "A mechanism for interstellar panspermia"},
    "WallisWickramasinghe2004": {"mode": "crossref", "q": "Wallis Wickramasinghe 2004 Interstellar transfer of planetary microbiota Monthly Notices", "expect": "Interstellar transfer of planetary microbiota"},
    "Gladman2005": {"mode": "crossref", "q": "Gladman Dones Levison Burns 2005 Impact seeding and reseeding in the inner solar system Astrobiology", "expect": "Impact Seeding and Reseeding in the Inner Solar System"},
    "Mileikowsky2000": {"mode": "crossref", "q": "Mileikowsky 2000 Natural transfer of viable microbes in space Icarus", "expect": "Natural Transfer of Viable Microbes in Space"},
    # --- free-floating planets and their capture ---
    "Sumi2023": {"mode": "arxiv_id", "id": "2303.08280", "expect": "Free-Floating planet Mass Function from MOA-II 9-year survey towards the Galactic Bulge"},
    "Mroz2017": {"mode": "arxiv_id", "id": "1707.07634", "expect": "No large population of unbound or wide-orbit Jupiter-mass planets"},
    "GoulinskiRibak2018": {"mode": "arxiv_id", "id": "1705.10332", "expect": "Capture of free-floating planets by planetary systems"},
    "PeretsKouwenhoven2012": {"mode": "arxiv_id", "id": "1202.2362", "expect": "On the origin of planets at very wide orbits from the re-capture of free floating planets"},
    # --- stellar encounters and Oort-cloud exchange ---
    "BailerJones2018": {"mode": "arxiv_id", "id": "1805.07581", "expect": "New stellar encounters discovered in the second Gaia data release"},
    "BailerJones2015": {"mode": "arxiv_id", "id": "1412.3648", "expect": "Close encounters of the stellar kind"},
    "GarciaSanchez2001": {"mode": "crossref", "q": "Garcia-Sanchez 2001 Stellar encounters with the solar system Astronomy and Astrophysics", "expect": "Stellar encounters with the solar system"},
    "Levison2010": {"mode": "crossref", "q": "Levison Duncan Brasser Kaufmann 2010 Capture of the Sun's Oort Cloud from Stars in Its Birth Cluster Science", "expect": "Capture of the Sun's Oort Cloud from Stars in Its Birth Cluster"},
    "Rickman2008": {"mode": "crossref", "q": "Rickman Fouchard Froeschle Valsecchi 2008 Injection of Oort Cloud comets: the fundamental role of stellar perturbations Celestial Mechanics", "expect": "Injection of Oort Cloud comets: the fundamental role of stellar perturbations"},
    "Vokrouhlicky2019": {"mode": "arxiv_title", "title": "Origin and Evolution of Long-period Comets", "expect": "Origin and Evolution of Long-period Comets"},
    # --- interstellar dust and supernova ejecta reaching planets ---
    "Grun1993": {"mode": "crossref", "q": "Grun 1993 Discovery of Jovian dust streams and interstellar grains by the Ulysses spacecraft Nature", "expect": "Discovery of Jovian dust streams and interstellar grains by the Ulysses spacecraft"},
    "Landgraf2000": {"mode": "crossref", "q": "Landgraf 2000 Modeling the motion and distribution of interstellar dust inside the heliosphere Journal of Geophysical Research", "expect": "Modeling the motion and distribution of interstellar dust inside the heliosphere"},
    "Kruger2019": {"mode": "arxiv_title", "title": "Interstellar dust in the solar system: model versus in situ spacecraft data", "expect": "Interstellar dust in the solar system"},
    "Wallner2016": {"mode": "crossref", "q": "Wallner 2016 Recent near-Earth supernovae probed by global deposition of interstellar radioactive 60Fe Nature", "expect": "Recent near-Earth supernovae probed by global deposition of interstellar radioactive 60Fe"},
    "Knie2004": {"mode": "crossref", "q": "Knie 2004 60Fe anomaly in a deep-sea manganese crust and implications for a nearby supernova source Physical Review Letters", "expect": "60Fe Anomaly in a Deep-Sea Manganese Crust and Implications for a Nearby Supernova Source"},
    "Fields2019": {"mode": "arxiv_title", "title": "Supernova triggers for end-Devonian extinctions", "expect": "Supernova triggers for end-Devonian extinctions"},
    "Koll2019": {"mode": "crossref", "q": "Koll Korschinek Faestermann 2019 Interstellar 60Fe in Antarctica Physical Review Letters", "expect": "Interstellar 60Fe in Antarctica"},
    # --- galactic mixing and the birth cluster ---
    "Frankel2018": {"mode": "arxiv_id", "id": "1805.09198", "expect": "Measuring radial orbit migration in the Galactic disk"},
    "PortegiesZwart2009": {"mode": "arxiv_id", "id": "0903.0237", "expect": "The lost siblings of the Sun"},
    "Adams2010": {"mode": "arxiv_id", "id": "1001.5444", "expect": "The Birth Environment of the Solar System"},
    "Pfalzner2013": {"mode": "arxiv_title", "title": "Early evolution of the birth cluster of the solar system", "expect": "Early evolution of the birth cluster of the solar system"},
    "Lada2003": {"mode": "arxiv_id", "id": "astro-ph/0301540", "expect": "Embedded Clusters in Molecular Clouds"},
}

NUMBER_WORDS = r"(rate|per (year|yr|Myr|Gyr|million years)|number density|density|cross[- ]section|probability|fraction|capture|impact|encounter|ejected|per star|AU\^?-?3|pc\^?-?3|km ?s|timescale|within)"


def extract_numbers(text: str) -> list[str]:
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if re.search(r"\d", sent) and re.search(NUMBER_WORDS, sent, re.I):
            out.append(re.sub(r"\s+", " ", sent).strip())
    return out


def main() -> None:
    ver = []
    numbers = {}
    for key, spec in REFS.items():
        rec = G.verify_ref(key, spec)
        ver.append(rec)
        # pull the abstract back out of the saved atom, if any
        for stem in (f"arxiv_id_{key}", f"arxiv_t_{key}"):
            p = OUT / f"{stem}.atom"
            if p.exists():
                ents = G._atom_entries(p.read_text("utf-8", "ignore"))
                best = max(ents, key=lambda e: G._overlap(spec.get("expect", ""), e["title"]), default=None)
                if best:
                    numbers[key] = {"title": best["title"], "arxiv": best["id"].rsplit("/", 1)[-1],
                                    "sentences": extract_numbers(best["summary"])}
        (OUT / "verification.json").write_text(json.dumps(
            {"n": len(ver), "counts": {s: sum(1 for r in ver if r["status"] == s)
                                      for s in ("VERIFIED", "MISMATCH", "NOT_FOUND", "ASSERTED")},
             "refs": ver}, indent=1))
        (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1))
    L = ["# GOO contact-graph literature: verification and the sentences that carry numbers", "",
         f"URLs fetched: {sum(1 for s in G.STATUS if s['ok'])} ok / {len(G.STATUS)} total.", "",
         "## Verification", "", "| key | status | record title | authors / venue |", "|---|---|---|---|"]
    for r in ver:
        who = ", ".join(r.get("authors", [])[:3]) if r.get("authors") else r.get("citation", "")
        ven = r.get("journal_ref") or r.get("journal") or ""
        L.append(f"| {r['key']} | {r['status']} | {r.get('record_title', '')} | {who} {ven} |")
    L += ["", "## Sentences carrying numbers (verbatim from the abstracts)", ""]
    for key, rec in numbers.items():
        L.append(f"### {key} — {rec['title']} (arXiv:{rec['arxiv']})")
        for s in rec["sentences"]:
            L.append(f"- {s}")
        L.append("")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print({s: sum(1 for r in ver if r["status"] == s) for s in ("VERIFIED", "MISMATCH", "NOT_FOUND", "ASSERTED")})


if __name__ == "__main__":
    main()
