#!/usr/bin/env python3
"""CENOTAPH recon: far-IR (>25um) technosignature prior-art + catalogue schemas.

The sandbox egress policy allows GitHub ONLY (arxiv/VizieR/IRSA/ADS/JAXA all
return 403 at the proxy), so every network fetch for the CENOTAPH channel has to
happen on the GitHub Actions runner (CLAUDE.md acquisition pattern).

Two jobs:

  TASK A - prior art. Does ANY executed Dyson/waste-heat search use a far-IR
           (>25um) all-sky catalogue?  Plus the exact temperature ranges of the
           executed searches, and full text for the four motivation papers
           (Cirkovic & Bradbury 2006, Zackrisson 2015, Annis 1999, Wright 2023).

  TASK B - data availability. EXACT VizieR/IRSA table identifiers and column
           names for AKARI FIS BSC, AKARI IRC PSC, IRAS PSC, IRAS FSC, AllWISE,
           Herschel HPPSC/SPSC, Planck PCCS2.  A wrong column name costs a whole
           runner job, so we pull the authoritative ReadMe (byte-by-byte column
           definitions), the TAP_SCHEMA rows, AND a live 5-row sample whose
           header is ground truth for the actual served column names.

Outputs under results/cenotaph/.
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/cenotaph")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-cenotaph/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []
ARXIV = "http://export.arxiv.org/api/query?"


def get(url: str, name: str, tries: int = 3, pause: float = 3.0) -> bool:
    dest = OUT / name
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": name, "bytes": len(data), "ok": True})
            print(f"OK   {name:52s} {len(data):9d}B", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i + 1}) {name} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": name, "ok": False})
    print(f"FAIL {name}  <- {url}", flush=True)
    return False


def arxiv_query(name: str, search: str, n: int = 60) -> None:
    q = urllib.parse.urlencode(
        {"search_query": search, "max_results": n,
         "sortBy": "relevance", "sortOrder": "descending"}
    )
    get(ARXIV + q, f"A_arxiv_q_{name}.atom")
    time.sleep(3)


def arxiv_ids(name: str, ids: list[str]) -> None:
    q = urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})
    get(ARXIV + q, f"A_arxiv_ids_{name}.atom")
    time.sleep(3)


# =====================================================================
# TASK A -- PRIOR ART
# =====================================================================

# A1/A2/A3: has anyone used a far-IR (>25um) all-sky catalogue for SETI?
PRIOR_ART = {
    "akari_dyson":      'all:"AKARI" AND all:"Dyson"',
    "akari_seti":       'all:"AKARI" AND (all:"SETI" OR all:"technosignature")',
    "akari_waste_heat": 'all:"AKARI" AND all:"waste heat"',
    "farir_dyson":      'all:"far-infrared" AND all:"Dyson sphere"',
    "farir_waste_heat": 'all:"waste heat" AND (all:"far-infrared" OR all:"far infrared")',
    "cold_dyson":       'abs:"Dyson" AND (abs:"cold" OR abs:"low temperature")',
    "planck_seti":      'all:"Planck" AND (all:"technosignature" OR all:"Dyson sphere")',
    "submm_technosig":  '(all:"submillimetre" OR all:"submillimeter" OR all:"millimetre") AND all:"technosignature"',
    "herschel_seti":    'all:"Herschel" AND (all:"SETI" OR all:"Dyson sphere" OR all:"technosignature")',
    "spire_pacs_techno": '(all:"SPIRE" OR all:"PACS") AND all:"technosignature"',
    "iras_dyson":       'all:"IRAS" AND all:"Dyson sphere"',
    "iras_60_100_seti": 'all:"IRAS" AND all:"Dyson" AND (all:"60 micron" OR all:"100 micron")',
    "matrioshka":       'all:"Matrioshka brain" OR all:"postbiological"',
    "kardashev_III":    'all:"Kardashev" AND all:"type III"',
    "technosig_recent": 'abs:"technosignature" AND abs:"infrared"',
    "dyson_candidates": 'abs:"Dyson sphere" AND abs:"candidates"',
    "waste_heat_seti":  'abs:"waste heat" AND abs:"extraterrestrial"',
    "dysonian_review":  'all:"Dysonian SETI"',
    "landauer_seti":    '(all:"Landauer" OR all:"Bremermann" OR all:"Brillouin") AND all:"extraterrestrial"',
    "cirkovic_all":     'au:"Cirkovic_M" AND (all:"SETI" OR all:"postbiological")',
    "suazo_all":        'au:"Suazo_M"',
    "zackrisson_seti":  'au:"Zackrisson_E" AND (all:"Dyson" OR all:"SETI")',
    "annis_kardashev":  'all:"Annis" AND all:"Kardashev"',
    "carrigan_all":     'au:"Carrigan_R"',
    "wright_dyson":     'au:"Wright_J" AND all:"Dyson sphere"',
}

# A2/A3/A4/A5/A6: exact papers whose full text we need
PAPER_IDS = {
    # executed searches -- temperature ranges
    "carrigan":   ["0811.2376", "1001.5455"],
    "ghat":       ["1408.1133", "1408.1134", "1504.03418", "1510.04606"],
    "hephaistos": ["2201.11123", "2405.02927", "2405.14921", "2501.05152"],
    "huang2026":  ["2601.07297"],
    # motivation papers
    "motivation": ["astro-ph/0506110", "1508.02406", "2309.06564"],
    "annis":      ["astro-ph/9901322"],
    "garrett":    ["1508.02624", "1604.07844"],
}

# full text: ar5iv renders LaTeX -> HTML; arxiv /abs for the abstract page
FULLTEXT = [
    "astro-ph/0506110",  # Cirkovic & Bradbury 2006
    "1508.02406",        # Zackrisson 2015 Tully-Fisher extragalactic SETI
    "2309.06564",        # Wright 2023 (optical depth of "complete" Dyson spheres)
    "0811.2376",         # Carrigan 2009 IRAS LRS
    "2405.02927",        # Hephaistos II
    "2601.07297",        # Huang, Tao & Zhang 2026
]


def task_a() -> None:
    print("\n########## TASK A: PRIOR ART ##########\n", flush=True)
    for name, q in PRIOR_ART.items():
        arxiv_query(name, q)
    for name, ids in PAPER_IDS.items():
        arxiv_ids(name, ids)
    for aid in FULLTEXT:
        slug = aid.replace("/", "_")
        get(f"https://ar5iv.labs.arxiv.org/html/{aid}", f"A_ar5iv_{slug}.html")
        time.sleep(2)
        get(f"https://arxiv.org/abs/{aid}", f"A_abs_{slug}.html")
        time.sleep(2)
    # Annis 1999 is JBIS -- try to confirm it exists and get any arXiv record
    get("https://api.semanticscholar.org/graph/v1/paper/search"
        "?query=Placing+a+limit+on+star-fed+Kardashev+type+III+civilisations"
        "&fields=title,abstract,year,authors,externalIds,citationCount",
        "A_s2_annis.json")
    get("https://api.semanticscholar.org/graph/v1/paper/search"
        "?query=Galactic+gradients+postbiological+evolution+apparent+failure+SETI"
        "&fields=title,abstract,year,authors,externalIds,citationCount",
        "A_s2_cirkovic.json")
    # OpenAlex full-text-ish search for the far-IR SETI negative
    for nm, qq in {
        "akari_dyson": "AKARI Dyson sphere",
        "farir_technosig": "far-infrared technosignature waste heat",
        "planck_dyson": "Planck compact sources Dyson sphere SETI",
    }.items():
        get("https://api.openalex.org/works?per-page=50&search="
            + urllib.parse.quote(qq), f"A_oa_{nm}.json")
        time.sleep(2)


# =====================================================================
# TASK B -- CATALOGUE SCHEMAS
# =====================================================================

VIZ_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP/sync"

# (short name, VizieR catalogue dir for ReadMe, VizieR table id for TAP/asu)
VIZ_CATS = [
    ("akari_fis",  "II/298", "II/298/fis"),
    ("akari_irc",  "II/297", "II/297/irc"),
    ("iras_psc",   "II/125", "II/125/main"),
    ("iras_fsc",   "II/156A", "II/156A/main"),
    ("allwise",    "II/328", "II/328/allwise"),
]


def viz_tap(query: str, name: str, fmt: str = "csv") -> None:
    q = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": fmt, "QUERY": query}
    )
    get(f"{VIZ_TAP}?{q}", name)
    time.sleep(2)


def task_b() -> None:
    print("\n########## TASK B: CATALOGUE SCHEMAS ##########\n", flush=True)

    for short, cdir, tid in VIZ_CATS:
        # 1. ReadMe = authoritative byte-by-byte column definitions + units
        get(f"https://cdsarc.cds.unistra.fr/ftp/{cdir}/ReadMe", f"B_readme_{short}.txt")
        time.sleep(1)
        # 2. TAP_SCHEMA columns
        viz_tap(
            "SELECT column_name, datatype, unit, ucd, description "
            f"FROM TAP_SCHEMA.columns WHERE table_name = '{tid}'",
            f"B_tapcols_{short}.csv",
        )
        # 3. VOTable header for one row: FIELD name/unit/ucd as actually served
        get(f"https://vizier.cds.unistra.fr/viz-bin/votable?-source={urllib.parse.quote(tid)}"
            "&-out.max=2&-out.all", f"B_votable_{short}.xml")
        time.sleep(2)
        # 4. live 5-row TSV sample -- header is ground truth for column names
        get(f"https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source={urllib.parse.quote(tid)}"
            "&-out.max=5&-out.all", f"B_sample_{short}.tsv")
        time.sleep(2)
        # 5. row count
        viz_tap(f"SELECT COUNT(*) AS n FROM \"{tid}\"", f"B_count_{short}.csv")

    # enumerate every AKARI / Herschel / Planck catalogue VizieR knows about
    for nm, words in {
        "akari": "AKARI", "herschel": "Herschel",
        "planck_pccs": "Planck compact sources", "iras": "IRAS",
    }.items():
        get("https://vizier.cds.unistra.fr/viz-bin/votable?-meta.all&-source="
            f"&-words={urllib.parse.quote(words)}&-meta.max=200",
            f"B_vizsearch_{nm}.xml")
        time.sleep(2)
        get("https://vizier.cds.unistra.fr/viz-bin/asu-txt?-meta&-source="
            f"&-words={urllib.parse.quote(words)}&-meta.max=200",
            f"B_vizsearch_{nm}.txt")
        time.sleep(2)

    # TAP_SCHEMA hunt for Herschel HPPSC / SPIRE SPSC / Planck PCCS2 table ids
    for nm, pat in {
        "hppsc": "%HPPSC%", "spsc": "%SPSC%", "herschel": "%erschel%",
        "pccs": "%PCCS%", "akari": "%II/29%",
    }.items():
        viz_tap(
            "SELECT table_name, description FROM TAP_SCHEMA.tables "
            f"WHERE table_name LIKE '{pat}' OR description LIKE '{pat}'",
            f"B_tapfind_{nm}.csv",
        )

    # ---- IRSA ----
    for nm, tbl in {
        "allwise": "allwise_p3as_psd",
        "wise_allsky": "wise_allsky_4band_p3as_psd",
        "seip": "slphotdr4",
    }.items():
        q = urllib.parse.urlencode({
            "QUERY": "SELECT column_name, datatype, unit, description "
                     f"FROM TAP_SCHEMA.columns WHERE table_name = '{tbl}'",
            "FORMAT": "csv"})
        get(f"{IRSA_TAP}?{q}", f"B_irsa_cols_{nm}.csv")
        time.sleep(2)
    get(f"{IRSA_TAP}?" + urllib.parse.urlencode(
        {"QUERY": "SELECT table_name, description FROM TAP_SCHEMA.tables",
         "FORMAT": "csv"}), "B_irsa_tables.csv")
    time.sleep(2)
    # IRSA Gator data dictionary (human-readable, authoritative for AllWISE)
    get("https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd?catalog=allwise_p3as_psd&mode=ascii",
        "B_irsa_gator_allwise.txt")
    time.sleep(2)

    # ---- instrument / release notes: beams, sensitivity, flags, cirrus ----
    docs = {
        "akari_fis_rn": "https://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/RN/AKARI-FIS_BSC_V1_RN.pdf",
        "akari_irc_rn": "https://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/RN/AKARI-IRC_PSC_V1_RN.pdf",
        "akari_fis_rn2": "https://darts.isas.jaxa.jp/astro/akari/data/AKARI-FIS_BSC_V1_RN.pdf",
        "iras_exp_psc": "https://irsa.ipac.caltech.edu/IRASdocs/exp.sup/ch2/C.html",
        "iras_exp_cirrus": "https://irsa.ipac.caltech.edu/IRASdocs/exp.sup/ch6/C3.html",
        "iras_exp_pos": "https://irsa.ipac.caltech.edu/IRASdocs/exp.sup/ch6/B.html",
        "allwise_sens": "https://wise2.ipac.caltech.edu/docs/release/allwise/expsup/sec2_4a.html",
        "allwise_cols": "https://wise2.ipac.caltech.edu/docs/release/allwise/expsup/sec2_1a.html",
        "wise_zeropoints": "https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html",
        "herschel_psc": "https://www.cosmos.esa.int/web/herschel/point-source-catalogues",
    }
    for nm, u in docs.items():
        get(u, f"B_doc_{nm}." + ("pdf" if u.endswith(".pdf") else "html"))
        time.sleep(2)

    # instrument papers with beam sizes / sensitivity / confusion
    for nm, ids in {
        "kawada_fis": ["0708.0110"],
        "yamamura_akari": ["0912.3717"],
        "doi_fis_maps": ["1503.02958"],
        "ishihara_irc": ["1003.0270"],
        "pccs2": ["1507.02058"],
        "planck_pccs1": ["1303.5088"],
    }.items():
        arxiv_ids(f"instr_{nm}", ids)


def main() -> None:
    try:
        task_a()
    except Exception as e:  # noqa: BLE001
        print(f"TASK A aborted: {e}", flush=True)
    try:
        task_b()
    except Exception as e:  # noqa: BLE001
        print(f"TASK B aborted: {e}", flush=True)
    (OUT / "status.json").write_text(json.dumps(STATUS, indent=1))
    ok = sum(1 for s in STATUS if s.get("ok"))
    print(f"\n=== {ok}/{len(STATUS)} fetches OK ===", flush=True)
    for s in STATUS:
        if not s.get("ok"):
            print(f"  MISSING: {s['file']}", flush=True)


if __name__ == "__main__":
    main()
