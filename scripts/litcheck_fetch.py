#!/usr/bin/env python3
"""HERDSMAN literature novelty check: fetch primary-source evidence on the runner.

The sandbox egress policy blocks arxiv.org / Semantic Scholar / ADS, so this
script runs on the GitHub Actions runner (per CLAUDE.md acquisition pattern) and
commits verbatim abstracts + citation lists back to the branch. Everything here
is a plain GET of public APIs; no keys required.

Outputs under results/litcheck/:
  arxiv_ids.atom            - arXiv API metadata (title/authors/abstract) for the
                              specific candidate-overlap papers, by ID
  arxiv_q_<name>.atom       - arXiv API search results for each query
  s2_<name>.json            - Semantic Scholar paper record + citations
  openalex_<name>.json      - OpenAlex fallback citation lists
  html_<id>.html            - arXiv full-text HTML of the 2026 candidate papers
                              (to inspect their own related-work sections)
  summary.json              - fetch status for every URL
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/litcheck")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-litcheck/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []


def get(url: str, dest: pathlib.Path, tries: int = 4, pause: float = 3.0) -> bool:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:40s} {len(data):8d}B  {url}")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {url} -> {e}")
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url}")
    return False


ARXIV = "http://export.arxiv.org/api/query?"

# --- 1. Specific candidate-overlap papers, by arXiv ID (verbatim abstracts) ---
IDS = [
    "1806.05203",  # Hooper 2018 - Life vs Dark Energy (stellar harvesting)
    "2009.08874",  # Lingam & Loeb 2020 - 0.01c stellar engines constraints
    "1306.1672",   # Forgan 2013 - detecting Class A stellar engines w/ transits
    "1804.08351",  # Zackrisson+ 2018 - SETI with Gaia (Dyson spheres)
    "2605.06072",  # 2026 - Phase-space crystallization in globular clusters
    "2607.07781",  # 2026 - Stellar J-harvesting (Kepler)
    "2605.21093",  # 2026 - The Search for Technosignatures: a Review
    "2201.11123",  # Suazo+ 2022 Project Hephaistos I
    "2405.02927",  # Suazo+ 2024 Project Hephaistos II (Gaia DR3 Dyson candidates)
    "1908.02765",  # (candidate ID) Caplan 2019 stellar engines - verify via title
    "2107.07512",  # Wright - SETI in 2020 annual review
    "2206.04092",  # Davenport+ SETI Ellipsoid with Gaia
    "1704.03910",  # Wright - exoplanet/SETI review (heuristics) - verify
    "2111.14183",  # (candidate) - anomaly detection / technosignature axes?
]
get(ARXIV + urllib.parse.urlencode({"id_list": ",".join(IDS), "max_results": len(IDS)}),
    OUT / "arxiv_ids.atom")
time.sleep(3)

# --- 2. arXiv API searches (Atom) ---
QUERIES = {
    "stellar_engine": 'all:"stellar engine" OR all:"stellar engines"',
    "shkadov": 'all:"Shkadov"',
    "star_lifting": 'all:"star lifting" OR all:"starlifting"',
    "astrom_techno": 'all:astrometric AND all:technosignature',
    "astrom_seti": 'all:astrometry AND all:SETI',
    "kinematic_techno": 'all:kinematic AND all:technosignature',
    "gaia_techno": 'abs:Gaia AND abs:technosignature',
    "gaia_seti": 'abs:Gaia AND abs:SETI',
    "hvs_techno": 'all:hypervelocity AND (all:technosignature OR all:SETI)',
    "moving_stars_civ": 'abs:"moving stars" AND (abs:civilization OR abs:SETI OR abs:technosignature)',
    "moved_star": 'abs:"artificially moved" OR abs:"stellar herding" OR abs:"herding stars"',
    "galactic_engineering": 'all:"galactic engineering" OR all:"astroengineering" OR all:"macro-engineering"',
    "stellivore": 'all:stellivore',
    "star_tug": 'all:"star tug" OR all:"Caplan thruster"',
    "dyson_propermotion": 'abs:"Dyson sphere" AND abs:"proper motion"',
    "anomalous_accel": 'abs:"anomalous acceleration" AND (abs:star OR abs:stellar) AND (abs:SETI OR abs:technosignature OR abs:artificial)',
    "velocity_divergence": 'abs:"velocity divergence" AND (abs:technosignature OR abs:artificial OR abs:SETI)',
    "wandering_engineered": 'abs:"engineered" AND abs:"stellar orbits"',
    "orbit_convergence": 'abs:"orbital convergence" OR abs:"converging orbits" AND abs:stars',
    "seti_proper_motion": 'abs:SETI AND abs:"proper motion"',
    "technosig_dynamics": 'abs:technosignature AND (abs:dynamics OR abs:dynamical)',
    "phase_space_techno": 'abs:"phase space" AND abs:technosignature',
}
for name, q in QUERIES.items():
    get(ARXIV + urllib.parse.urlencode(
        {"search_query": q, "max_results": "60", "sortBy": "relevance"}),
        OUT / f"arxiv_q_{name}.atom")
    time.sleep(3.2)  # arXiv API politeness

# --- 3. Semantic Scholar: resolve core papers, pull their citation lists ------
S2 = "https://api.semanticscholar.org/graph/v1/paper/"
S2FIELDS = "title,year,abstract,externalIds,venue,authors"


def s2_paper(name: str, ident: str) -> None:
    get(S2 + urllib.parse.quote(ident) + f"?fields={S2FIELDS},citationCount",
        OUT / f"s2_{name}.json", pause=6.0)
    time.sleep(2.5)
    get(S2 + urllib.parse.quote(ident) +
        f"/citations?fields={S2FIELDS}&limit=500",
        OUT / f"s2_{name}_citations.json", pause=6.0)
    time.sleep(2.5)


s2_paper("hooper2018", "arXiv:1806.05203")
s2_paper("lingamloeb2020", "arXiv:2009.08874")
s2_paper("forgan2013", "arXiv:1306.1672")
s2_paper("zackrisson2018", "arXiv:1804.08351")

# Title-resolved (no arXiv ID known a priori)
SEARCHES = {
    "caplan2019": "Stellar engines Design considerations for maximizing acceleration",
    "badescu2000": "Stellar engines for Kardashev's type II civilisations",
    "svoronos": "The Star Tug active stellar engine",
    "shkadov1987": "Possibility of controlling solar system motion in the galaxy",
    "vidal_stellivore": "stellivore binary stars living systems",
    "smeti": "Migrating extraterrestrial civilizations and interstellar colonization implications for SETI and SETA",
    "loeb_fuel": "Securing Fuel for Our Frigid Cosmic Future",
}
for name, q in SEARCHES.items():
    get(S2 + "search?" + urllib.parse.urlencode(
        {"query": q, "fields": S2FIELDS, "limit": "5"}),
        OUT / f"s2_search_{name}.json", pause=6.0)
    time.sleep(2.5)

# Citations of the resolved concept papers (fetched after inspecting search
# results is ideal; here we optimistically try known S2 corpus routes)
for name, ident in {
    "caplan2019_c": "DOI:10.1016/j.actaastro.2019.08.030",
    "badescu2000_c": "DOI:10.1016/S0094-5765(00)00075-1",
}.items():
    get(S2 + urllib.parse.quote(ident) + f"/citations?fields={S2FIELDS}&limit=500",
        OUT / f"s2_{name}.json", pause=6.0)
    time.sleep(2.5)

# --- 4. OpenAlex fallback citation lists -------------------------------------
OA = "https://api.openalex.org/works"
get(OA + "?" + urllib.parse.urlencode(
    {"search": "Life versus dark energy advanced civilization accelerating expansion",
     "per-page": "5"}), OUT / "openalex_hooper_resolve.json")
time.sleep(1.5)
get(OA + "?" + urllib.parse.urlencode(
    {"search": "stellar engines Milky Way constraints abundance",
     "per-page": "5"}), OUT / "openalex_lingam_resolve.json")
time.sleep(1.5)

# --- 5. Full-text HTML of the 2026 near-neighbours (related-work mining) -----
for aid in ["2605.06072", "2607.07781", "2605.21093"]:
    ok = get(f"https://arxiv.org/html/{aid}v1", OUT / f"html_{aid}.html")
    if not ok:
        get(f"https://arxiv.org/html/{aid}", OUT / f"html_{aid}.html")
    time.sleep(3)

(OUT / "summary.json").write_text(json.dumps(STATUS, indent=2))
n_ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n{n_ok}/{len(STATUS)} fetches succeeded")
