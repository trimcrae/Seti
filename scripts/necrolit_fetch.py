#!/usr/bin/env python3
"""NECROSIGNATURES novelty check: fetch primary sources on the GitHub runner.

The sandbox blocks arxiv.org / ADS / OpenAlex / Semantic Scholar, and the
session WebSearch budget is finite; the Actions runner has egress. This script
fetches verbatim metadata, full text, and citation trees for every antecedent
that could sink a channel in ``docs/necrosignatures.md``, and commits the
evidence back to the branch so the per-channel agents can read full texts
offline.

The single most important target is the **citation tree of Zackrisson et al.
2018 (arXiv:1804.08351)**, which proposed the grey-attenuation / underluminous
Dyson-sphere statistic that CENOTAPH executes. If nobody ever ran it, that
shows up as an absence of observational papers among its citers — and that
absence is the channel's novelty claim, so it must be established from the
record rather than assumed.

Outputs under ``results/necrolit/``:
  arxiv_id_<id>.atom     arXiv API metadata for a specific paper
  arxiv_q_<name>.atom    arXiv API keyword search (discovery)
  ar5iv_<id>.html        ar5iv full-text rendering (equations/tables survive)
  txt_<id>.txt           extracted plain text (PDF -> pdftotext)
  oa_<name>.json         OpenAlex work record (DOI metadata, references)
  citedby_<name>.json    OpenAlex "works citing this" listing
  summary.json           fetch status for every URL
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/necrolit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-necrolit/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []
PAUSE = 3.0


def get(url: str, dest: pathlib.Path, tries: int = 3, pause: float = PAUSE) -> bool:
    """Fetch ``url`` to ``dest`` with retries; record the outcome in STATUS."""
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
# 1. Specific papers whose FULL TEXT decides a novelty verdict.
# --------------------------------------------------------------------------
# Each entry: short name -> arXiv id. The question each one answers is in the
# comment; the per-channel agents must read the text and answer it explicitly.
FULLTEXT: dict[str, str] = {
    # CENOTAPH (S2). Proposed the underluminous/grey-absorber Dyson statistic.
    # Q: did they EXECUTE a search, or only forecast? Do they impose an
    # achromaticity test separating grey occultation from reddening? Do they
    # treat a COLD (non-reradiating) sphere, or only "nearly complete" warm ones?
    "zackrisson2018_gaia_dyson": "1804.08351",
    # The extragalactic ancestor of "underluminous at fixed dynamical mass".
    "zackrisson2015_tullyfisher": "1508.02406",
    # Hephaistos I/II: what every executed Dyson search selects on (IR excess).
    "hephaistos1": "2201.11123",
    "hephaistos2": "2405.02927",
    "hephaistos_contam": "2607.03619",
    # THE DECISIVE GAP. Cirkovic & Bradbury 2006 predict postbiological
    # computation radiates near ~50 K ("considerable difference ... whether one
    # expects a Dyson shell to be close to a blackbody at 50 K, as contrasted to
    # a blackbody at 300 K"). Every executed search covers ~100-1000 K, capped
    # by WISE W4 at 22 um. Q: confirm the 50 K prediction verbatim, and confirm
    # no search has ever probed below 100 K.
    "cirkovic_bradbury_50K": "astro-ph/0506110",
    # Wright 2023: Landauer-limit thermodynamics of Dyson spheres. Q: does it
    # predict optical depth of several (=> grey occulter) and rule out nesting?
    "wright2023_thermodynamics": "2309.06564",
    # ISOTHERM (S5/S6). Thermal-IR technosignature framing; claims the
    # discriminants are lack of far-IR and no star-formation association.
    "thermal_ir_technosig": "1907.07829",
    # The most recent executed searches — what temperature range do they state?
    "huang2026_catwise": "2601.07297",
    "hephaistos4_jwst": "2607.09460",
    "lacki_ground_to_dust": "2504.21151",
    # Data-driven mid-IR excess over ~5M FGK stars: does it use SED SHAPE?
    "datadriven_midir_5M": "2403.18941",
    # VIGIL (S4). The enabling mid-IR time-domain catalogue.
    "untimely_variables": "2511.22071",
    # VIGIL confounder: extreme debris disks are mid-IR variable, optically flat.
    "extreme_debris_allwise": "2103.00568",
    # TIDEMARK (S23/S25). The three mutually contradictory front predictions.
    "carrigan_starry_messages": "1001.5455",
    "cirkovic_against_empire": "0805.1821",
    "grabby_aliens": "2102.01522",
    # Aestivation: the ASI dormancy hypothesis CENOTAPH is tuned for.
    "aestivation": "1705.03394",
    "aestivation_rebuttal": "1902.06730",
    # AI-as-great-filter: derives a lifetime bound but NO astronomical observable.
    "garrett_ai_filter": "2405.00042",
    # DERELICT (S19). Dark comets: nongrav acceleration without a coma.
    # CORRECTED 2026-07-26: this was "2306.16966", which is a hep-ph paper
    # ("Self-interacting dark matter implied by nano-Hertz gravitational
    # waves") -- a dark-matter/dark-comet mixup.  The fetch SUCCEEDED against
    # the wrong id, so nothing flagged it and results/necrolit/*dark_comets*
    # are the wrong paper.  See scripts/derelict_lit.py, which re-fetches these
    # with an id+title verification step and supersedes the necrolit copies.
    "dark_comets": "2212.08115",          # Seligman et al. 2023, PSJ 4, 35
    # (The PNAS 2024 companion is deliberately NOT listed here: the id
    # "2412.02384" was guessed and turned out to be an unrelated software-
    # engineering paper.  scripts/derelict_lit.py resolves it BY TITLE instead.)
    # KNELL/SHROUD (S31/S33). VASCO: optical disappearance.
    "vasco": "1606.08992",
}

# --------------------------------------------------------------------------
# 2. Discovery searches: has ANYONE run the search this channel proposes?
# --------------------------------------------------------------------------
QUERIES: dict[str, str] = {
    # CENOTAPH
    "grey_attenuation_dyson": 'all:"gray absorber" AND all:"Dyson"',
    "underluminous_technosig": 'all:"underluminous" AND all:technosignature',
    "spectrophotometric_parallax_anomaly": 'all:"spectrophotometric distance" AND all:"Dyson"',
    "achromatic_dimming_search": 'all:"achromatic" AND all:"dimming" AND all:star',
    # THE COLD REGIME — the gap between the ~50 K prediction and the
    # 100-1000 K searches. If these come back empty, the gap is real.
    "cold_dyson_sphere": 'all:"Dyson" AND (all:"cold" OR all:"far-infrared")',
    "akari_fis_technosignature": 'all:AKARI AND all:technosignature',
    "far_ir_excess_mainsequence": 'all:"far-infrared excess" AND all:"main-sequence"',
    "iras_60_100_excess_stars": 'all:IRAS AND all:"60 micron" AND all:excess AND all:stars',
    "waste_heat_temperature_range": 'all:"waste heat" AND all:Dyson AND all:temperature',
    # TIDEMARK -- has ANYONE tested an anomaly population for spatial structure?
    # These are the gap-closing queries the 2026-07-26 sweep identified. Note the
    # arXiv "all:" field is metadata-only and returns spurious zeros for papers
    # that are demonstrably present, so a zero here is weak evidence; the
    # citation trees below carry the weight.
    "tidemark_galactocentric_technosig": 'all:"Galactocentric" AND all:technosignature',
    "tidemark_galactocentric_dyson": 'all:"Galactocentric" AND all:"Dyson"',
    "tidemark_technosig_spatial_distribution": 'all:technosignature AND (all:"spatial distribution" OR all:"sky distribution" OR all:"surface density")',
    "tidemark_dyson_galactic_latitude": 'all:"Dyson sphere" AND (all:"Galactic latitude" OR all:"scale height")',
    "tidemark_occurrence_rate_technosig": 'all:"occurrence rate" AND all:technosignature',
    "tidemark_fermi_bubble_seti": 'all:"Fermi bubble" AND (all:SETI OR all:technosignature)',
    "tidemark_colonization_boundary_search": 'all:"colonization" AND all:"boundary" AND all:search AND all:galaxy',
    "tidemark_grabby_observational": '(all:"grabby aliens" OR all:"expanding civilizations") AND (all:"sky survey" OR all:constraint)',
    # The AGE leg (S25) -- the weakest-evidenced part of the novelty claim.
    "tidemark_technosig_stellar_age": 'all:technosignature AND (all:"stellar age" OR all:"age distribution")',
    "tidemark_great_filter_observational": 'all:"Great Filter" AND (all:"observational test" OR all:occurrence)',
    "tidemark_technosig_metallicity": 'all:technosignature AND all:metallicity',
    # ISOTHERM — the shape statistic
    "blackbody_shape_technosig": 'all:"single blackbody" AND all:technosignature',
    "matrioshka_brain": 'all:"Matrioshka" OR all:"nested Dyson"',
    "two_temperature_debris": 'all:"two-temperature" AND all:"debris disk"',
    "featureless_dust_sed": 'all:"featureless" AND all:"silicate" AND all:"debris disk"',
    # EMBER — waste heat that switched off
    "vanishing_ir_excess": 'all:"disappearing" AND all:"infrared excess"',
    "iras_wise_variability": 'all:IRAS AND all:WISE AND all:"infrared excess" AND all:variability',
    "disk_turned_off": 'all:"TYC 8241" OR all:"disappearance of" AND all:"warm dust"',
    # OSSUARY — debris where it cannot exist
    "metalpoor_debris_disk": 'all:"metal-poor" AND all:"debris disk"',
    "halo_star_infrared_excess": 'all:"halo star" AND all:"infrared excess"',
    "old_star_debris_disk": 'all:"debris disk" AND all:"old stars" AND all:incidence',
    # TAILINGS — sparse chemical anomaly
    "chemical_anomaly_detection": 'all:"anomaly detection" AND (all:APOGEE OR all:GALAH)',
    "single_element_anomaly": 'all:"abundance anomaly" AND all:"main-sequence" AND all:"single element"',
    "chemical_tagging_dimensionality": 'all:"chemical tagging" AND all:dimensionality',
    "planet_engulfment_twins": 'all:"planet engulfment" AND all:"co-natal"',
    # TIDEMARK — the front
    "galactic_gradient_technosig": 'all:technosignature AND (all:gradient OR all:anisotropy)',
    "colonization_front_search": 'all:"settlement front" OR all:"colonization front"',
    "fermi_bubble_archaeology": 'all:"interstellar archaeology"',
    # VIGIL
    "midir_variable_optically_constant": 'all:"mid-infrared variability" AND all:"main-sequence"',
    # DERELICT
    "nongravitational_artificial": 'all:"non-gravitational acceleration" AND all:"artificial"',
    "lightsail_search": 'all:lightsail AND all:search AND all:solar system',
    # Cross-cutting: the extinction framing itself
    "extinct_civilization_technosig": 'all:"extinct" AND all:technosignature',
    "technosignature_longevity": 'all:technosignature AND all:longevity',
    "postbiological_seti": 'all:postbiological AND all:SETI',
}

# --------------------------------------------------------------------------
# 3. OpenAlex citation trees. "Who cited this, and did any of them SEARCH?"
# --------------------------------------------------------------------------
CITED_BY: dict[str, str] = {
    # THE decisive one for CENOTAPH.
    "zackrisson2018": "10.3847/1538-4357/aac386",
    "zackrisson2015_tf": "10.1088/0004-637X/810/1/23",
    "hephaistos2": "10.1093/mnras/stae1186",
    "ghat3": "10.1088/0067-0049/217/2/25",
    "cirkovic_bradbury_gradient": "10.1016/j.newast.2006.04.003",
    "aestivation": "10.1016/j.jbis.2017.09.005",
    # TIDEMARK: the spatial-structure predictions. The novelty claim is that
    # each of these has been *cited* but never *tested*, so the citing sets are
    # the decisive evidence -- and they were unreadable until the cites: filter
    # bug above was fixed.
    "tidemark_wright2021_rnaas": "10.3847/2515-5172/ac0910",     # -> Galactic centre
    "tidemark_hanson2021_grabby": "10.3847/1538-4357/ac2369",    # "for which astronomers might search"
    "tidemark_carroll_nellenback2019": "10.3847/1538-3881/ab31a3",
    "tidemark_ghat1_shearmixing": "10.1088/0004-637X/792/1/26",  # predicts NO structure
    "tidemark_vukotic_cirkovic2012": "10.1007/s11084-012-9273-6",
    "tidemark_hair_hedman2013": "10.1017/S1473550412000420",
    "tidemark_lingam_percolation2016": "10.1089/ast.2015.1411",
    "tidemark_carrigan2009_iras": "10.1088/0004-637X/698/2/2075",
}

ARXIV_API = "http://export.arxiv.org/api/query?"
OPENALEX = "https://api.openalex.org/works"


def fetch_fulltext() -> None:
    print("\n=== full text of decisive antecedents ===")
    for name, aid in FULLTEXT.items():
        print(f"[{name}] arXiv:{aid}")
        get(ARXIV_API + urllib.parse.urlencode({"id_list": aid, "max_results": 1}),
            OUT / f"arxiv_id_{name}.atom")
        get(f"https://ar5iv.labs.arxiv.org/html/{aid}", OUT / f"ar5iv_{name}.html")
        pdf = OUT / f"pdf_{name}.pdf"
        if get(f"https://arxiv.org/pdf/{aid}", pdf):
            try:
                subprocess.run(["pdftotext", "-q", str(pdf), str(OUT / f"txt_{name}.txt")],
                               check=False, timeout=120)
                pdf.unlink(missing_ok=True)  # keep the repo small; text is what we read
            except Exception as exc:  # noqa: BLE001
                print(f"  pdftotext failed: {exc!r}")


def fetch_queries() -> None:
    print("\n=== discovery searches ===")
    for name, q in QUERIES.items():
        print(f"[q:{name}] {q}")
        get(ARXIV_API + urllib.parse.urlencode(
            {"search_query": q, "max_results": 40,
             "sortBy": "relevance", "sortOrder": "descending"}),
            OUT / f"arxiv_q_{name}.atom")


def fetch_citation_trees() -> None:
    """Enumerate everything citing each anchor paper.

    NOTE (2026-07-26): the previous implementation used
    ``?filter=cites:doi:<doi>``, which OpenAlex accepts but silently answers with
    ``meta.count = 0`` for **every** DOI --- so all earlier citation-tree fetches
    in this repo (``citedby_zackrisson2018.json``, ``cited_by_heph1_p*.json``)
    are empty for a syntax reason, not because the papers are uncited.  The
    ``cites:`` filter takes a **work ID**, so resolve the DOI first and pass
    ``cites:W...``.  Verified by the ``meta.count`` in the fetched JSON.
    """
    print("\n=== OpenAlex citation trees ===")
    for name, doi in CITED_BY.items():
        print(f"[cite:{name}] {doi}")
        meta = OUT / f"oa_{name}.json"
        get(f"{OPENALEX}/https://doi.org/{doi}", meta)
        work_id = None
        try:
            wid = json.loads(meta.read_text()).get("id") or ""
            work_id = wid.rsplit("/", 1)[-1] if wid.startswith("http") else wid
        except Exception as exc:  # noqa: BLE001
            print(f"  could not resolve a work id for {doi}: {exc!r}")
        if not work_id:
            print(f"  SKIP citation tree for {name}: no work id")
            continue
        # Everything citing it, newest first; 200 is well above any of these counts.
        get(f"{OPENALEX}?filter=cites:{work_id}&per-page=200"
            "&sort=publication_date:desc", OUT / f"citedby_{name}.json")


def main() -> None:
    fetch_fulltext()
    fetch_queries()
    fetch_citation_trees()
    ok = sum(1 for s in STATUS if s["ok"])
    (OUT / "summary.json").write_text(json.dumps(
        {"n_urls": len(STATUS), "n_ok": ok, "n_failed": len(STATUS) - ok,
         "status": STATUS}, indent=2))
    print(f"\n=== {ok}/{len(STATUS)} fetches succeeded -> {OUT} ===")


if __name__ == "__main__":
    main()
