#!/usr/bin/env python3
"""CHEMLIT: literature + atomic-data evidence for chemical technosignatures.

Domain: stellar photospheric composition as a technosignature ("chemical
tombstones") -- Whitmire & Wright (1980) nuclear-waste spectra, Przybylski's
Star, Tc/Pm in unevolved stars, chemical peculiarity in cool dwarfs, abundance
-space dimensionality, sparse abundance-anomaly detection, and planet-engulfment
signatures in co-natal binaries.

The sandbox egress policy blocks arxiv.org / ADS / Semantic Scholar / OpenAlex /
Crossref / NIST; this script runs on the GitHub Actions runner (per CLAUDE.md
acquisition pattern) and commits verbatim metadata back to the branch.
Everything here is a plain GET of a public API; no keys required.

Outputs under results/chemlit/:
  arxiv_q_<name>.atom     arXiv API search results (title/authors/abstract)
  s2_<name>.json          Semantic Scholar paper record
  s2_<name>_citations.json  ... and who cites it (the prior-art tree)
  oa_<name>.json          OpenAlex fallback / cited_by lists
  crossref_<name>.json    Crossref records for pre-arXiv papers
  nist_<species>_<fmt>.txt  raw NIST ASD line lists (feasibility table input)
  summary.json            fetch status for every URL
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/chemlit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-chemlit/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []


def get(url: str, dest: pathlib.Path, tries: int = 3, pause: float = 3.0) -> bool:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:44s} {len(data):9d}B", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {dest.name} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {dest.name}  {url}", flush=True)
    return False


# ===========================================================================
# 1. arXiv API searches  (Atom: title + authors + full abstract + ID + date)
# ===========================================================================
ARXIV = "http://export.arxiv.org/api/query?"

QUERIES = {
    # --- (1) Whitmire & Wright and the nuclear-waste / artificial-element idea
    "ww_nuclear_waste": 'all:"nuclear waste" AND (all:extraterrestrial OR all:civilization OR all:technosignature)',
    "fission_photosphere": 'all:"fission products" AND (all:photosphere OR all:stellar)',
    "artificial_element_star": 'abs:artificial AND abs:element AND (abs:technosignature OR abs:SETI)',
    "techno_spectroscopy": 'abs:technosignature AND (abs:spectroscopy OR abs:abundance OR abs:composition)',
    "seti_stellar_spectra": 'all:SETI AND (all:"stellar spectra" OR all:"stellar spectrum")',
    "techno_white_dwarf": 'abs:"white dwarf" AND (abs:technosignature OR abs:SETI OR abs:artificial)',
    "polluted_wd_techno": 'abs:polluted AND abs:"white dwarf" AND abs:technosignature',
    "stellar_doping": 'abs:"stellar doping" OR abs:"seeding a star" OR abs:"marker element"',
    "techno_review": 'ti:technosignature AND ti:review',

    # --- (2) Przybylski's Star / Ap-star exotica
    "przybylski": 'all:Przybylski',
    "hd101065": 'all:"HD 101065"',
    "promethium": 'all:promethium',
    "actinide_star": 'all:actinide AND (all:star OR all:stellar)',
    "roap_rare_earth": 'abs:roAp AND abs:"rare earth"',
    "ap_stratification": 'abs:"chemically peculiar" AND abs:stratification AND abs:abundance',

    # --- (3) Technetium
    "technetium": 'all:technetium',
    "technetium_agb": 'all:technetium AND (all:AGB OR all:"S star" OR all:"dredge-up")',
    "technetium_dwarf": 'all:technetium AND (all:"main sequence" OR all:dwarf OR all:unevolved)',
    "tc_lines": 'abs:"Tc I" OR abs:"technetium lines"',
    "th_u_chronometer": 'abs:thorium AND abs:uranium AND abs:"metal-poor"',
    "actinide_boost": 'abs:"actinide boost"',

    # --- (4) diffusion / chemical peculiarity in cool stars
    "atomic_diffusion": 'abs:"atomic diffusion" AND abs:abundances AND abs:"main sequence"',
    "radiative_levitation": 'abs:"radiative levitation"',
    "diffusion_globular": 'abs:diffusion AND abs:"globular cluster" AND abs:"turn-off" AND abs:abundances',
    "diffusion_m67": 'abs:diffusion AND abs:M67 AND abs:abundances',
    "cp_dwarfs": 'abs:"chemically peculiar" AND (abs:dwarfs OR abs:"cool stars")',
    "barium_dwarf": 'abs:"barium dwarf" OR abs:"CH subgiant" OR abs:"barium star" AND abs:dwarf',
    "cemp_dwarfs": 'abs:CEMP AND abs:dwarf AND abs:abundances',
    "lambda_boo": 'abs:"lambda Bootis"',
    "convective_envelope_mass": 'abs:"convective envelope" AND abs:mass AND abs:"main-sequence" AND abs:abundance',
    "thermohaline": 'abs:thermohaline AND abs:mixing AND abs:abundance',
    "nlte_corrections": 'abs:"non-LTE" AND abs:abundance AND abs:corrections AND abs:survey',

    # --- (5) dimensionality of abundance space / chemical tagging
    "abundance_dimensionality": 'abs:"abundance space" AND (abs:dimensionality OR abs:dimensions)',
    "chemical_tagging": 'abs:"chemical tagging"',
    "two_process_model": 'abs:"two-process" AND abs:abundances',
    "chemical_cartography": 'ti:"chemical cartography"',
    "doppelganger": 'abs:doppelganger AND abs:stars',
    "information_content_spectra": 'ti:"information content" AND ti:spectra',
    "intrinsic_scatter_abundances": 'abs:"intrinsic scatter" AND abs:abundances AND (abs:APOGEE OR abs:GALAH)',
    "residual_abundances": 'abs:residual AND abs:abundances AND abs:APOGEE',

    # --- (6) anomaly / outlier detection in abundance & spectral catalogs
    "anomaly_detection_stellar": 'abs:"anomaly detection" AND (abs:stellar OR abs:spectra OR abs:stars)',
    "outlier_detection_spectra": 'abs:"outlier detection" AND abs:spectra',
    "unsupervised_apogee": 'abs:unsupervised AND abs:APOGEE',
    "weird_spectra_sdss": 'abs:"weird" AND abs:spectra AND abs:SDSS',
    "rare_object_lamost": 'abs:LAMOST AND (abs:rare OR abs:unusual OR abs:peculiar) AND (abs:"machine learning" OR abs:search)',
    "autoencoder_spectra": 'abs:autoencoder AND abs:spectra AND abs:stellar',
    "single_element_anomaly": 'abs:"single element" AND abs:abundance AND (abs:anomaly OR abs:outlier OR abs:peculiar)',
    "phosphorus_rich": 'abs:"phosphorus-rich" OR abs:"phosphorus rich stars"',
    "k_rich_stars": 'abs:potassium AND abs:enhanced AND abs:stars',
    "silicon_rich": 'abs:"silicon-rich" AND abs:stars AND abs:APOGEE',
    "chemically_unusual_survey": 'abs:"chemically unusual" OR abs:"chemically anomalous" AND abs:survey',

    # --- (7) planet engulfment / co-natal binaries
    "engulfment": 'abs:engulfment AND (abs:planet OR abs:planetary) AND abs:abundances',
    "conatal_binaries": 'abs:"co-natal" OR abs:conatal',
    "binary_abundance_difference": 'abs:binary AND abs:"abundance difference" AND abs:twins',
    "solar_twins_tcond": 'abs:"solar twins" AND abs:"condensation temperature"',
    "wide_binary_chemistry": 'abs:"wide binaries" AND abs:chemical',
    "wide_binary_catalog_gaia": 'ti:"wide binaries" AND abs:Gaia AND abs:catalog',
    "kronos_krios": 'abs:"HD 240430" OR abs:Kronos AND abs:Krios',
    "planet_signature_persistence": 'abs:engulfment AND (abs:signature OR abs:detectability) AND abs:convective',

    # --- survey capability papers (for the feasibility table)
    "galah_dr": 'ti:GALAH AND ti:"data release"',
    "apogee_dr17": 'ti:APOGEE AND abs:"Data Release 17"',
    "aspcap": 'ti:ASPCAP OR abs:"ASPCAP" AND abs:pipeline',
    "lamost_mrs": 'abs:LAMOST AND abs:"medium-resolution" AND abs:survey',
    "lamost_abundances": 'abs:LAMOST AND abs:abundances AND abs:"data-driven"',
    "gaia_rvs_spectra": 'abs:Gaia AND abs:RVS AND abs:spectra AND abs:"data release 3"',
    "rave_dr6": 'ti:RAVE AND abs:"data release"',
    "fourmost": 'abs:4MOST AND abs:survey AND abs:spectrograph',
    "weave_survey": 'abs:WEAVE AND abs:survey AND abs:spectrograph',
    "sdss5_mwm": 'abs:"Milky Way Mapper" OR abs:"SDSS-V" AND abs:survey',
    "harps_archive_mining": 'abs:HARPS AND abs:archive AND (abs:mining OR abs:"large sample")',
}

for name, q in QUERIES.items():
    get(ARXIV + urllib.parse.urlencode(
        {"search_query": q, "max_results": "50", "sortBy": "relevance"}),
        OUT / f"arxiv_q_{name}.atom")
    time.sleep(3.2)  # arXiv API politeness

# ===========================================================================
# 2. Semantic Scholar: resolve key papers and pull their CITATION trees.
#    The Whitmire & Wright tree is the central prior-art question:
#    has anyone ever *executed* the 1980 proposal?
# ===========================================================================
S2 = "https://api.semanticscholar.org/graph/v1/paper/"
S2FIELDS = "title,year,abstract,externalIds,venue,authors,citationCount"


def s2_paper(name: str, ident: str, citations: bool = True) -> None:
    get(S2 + urllib.parse.quote(ident) + f"?fields={S2FIELDS}",
        OUT / f"s2_{name}.json", pause=6.0)
    time.sleep(3.0)
    if citations:
        get(S2 + urllib.parse.quote(ident) + f"/citations?fields={S2FIELDS}&limit=500",
            OUT / f"s2_{name}_citations.json", pause=6.0)
        time.sleep(3.0)


# Papers with known, stable DOIs (pre-arXiv or journal-only)
S2_BY_DOI = {
    "whitmire_wright_1980": "DOI:10.1016/0019-1035(80)90253-5",   # Icarus 42, 149
    "michaud_1970": "DOI:10.1086/150442",                          # ApJ 160, 641
}
for name, ident in S2_BY_DOI.items():
    s2_paper(name, ident)

# Title searches (S2 resolves these to IDs + abstracts; no ID guessing)
S2_SEARCHES = {
    "ww1980": "Nuclear waste spectrum as evidence of technological extraterrestrial civilizations",
    "merrill1952": "Spectroscopic observations of stars of class S technetium",
    "przybylski1961": "Przybylski peculiar star HD 101065 unusual spectrum",
    "cowley_promethium": "promethium HD 101065 Przybylski identification",
    "cowley_actinides": "actinides chemically peculiar stars identification spectra",
    "andrievsky_tc": "technetium Przybylski star spectrum reanalysis",
    "vaneck_jorissen": "Tc-rich and Tc-poor S stars technetium",
    "shetye_s_stars": "S stars s-process Gaia era technetium third dredge-up",
    "uttenthaler_tc": "technetium AGB stars Mira variables third dredge up",
    "cayrel2001": "Measurement of stellar age from uranium decay CS 31082-001",
    "michaud_book": "Atomic diffusion in stars Michaud Alecian Richer",
    "richer2000": "Turcotte Richer Michaud radiative accelerations abundance anomalies main sequence",
    "korn2007": "atomic diffusion and mixing in old stars NGC 6397",
    "souto_m67": "chemical abundances M67 atomic diffusion APOGEE",
    "ting2012": "principal component analysis chemical abundance space",
    "pricejones_bovy": "well-defined dimensionality for chemical abundance space",
    "weinberg2019": "chemical cartography APOGEE two-process",
    "weinberg2022": "two-process model abundances APOGEE residual",
    "griffith_residual": "residual abundances two-process GALAH APOGEE",
    "ness_doppelganger": "galactic doppelgangers chemical similarity stars",
    "baron_poznanski": "weirdest SDSS galaxies outlier detection random forest",
    "reis_apogee": "unsupervised outlier detection APOGEE spectra",
    "masseron_phosphorus": "phosphorus-rich stars nucleosynthesis APOGEE",
    "spina2021": "chemical evidence planetary ingestion sun-like stars",
    "liu2024": "planetary engulfment signatures co-natal binary stars",
    "behmard_engulfment": "planet engulfment signatures stellar convective zone detectability",
    "oh_kronos": "Kronos Krios stellar twins abundance difference planet engulfment",
    "melendez2009": "peculiar solar abundance pattern condensation temperature solar twins",
    "bedell2018": "chemical homogeneity solar twins abundances",
    "elbadry_wide": "million binaries Gaia eDR3 wide binary catalog",
    "hawkins_comoving": "chemical homogeneity comoving pairs stars",
    "espinozarojas": "chemical homogeneity wide binaries GALAH",
    "galah_dr3": "GALAH survey third data release",
    "galah_dr4": "GALAH survey fourth data release",
    "apogee_dr17": "SDSS-IV APOGEE Data Release 17 summary",
    "xiang_ddpayne": "abundance estimates LAMOST spectra data-driven Payne",
}
for name, q in S2_SEARCHES.items():
    get(S2 + "search?" + urllib.parse.urlencode(
        {"query": q, "fields": S2FIELDS, "limit": "6"}),
        OUT / f"s2_search_{name}.json", pause=6.0)
    time.sleep(3.0)

# ===========================================================================
# 3. OpenAlex: independent citation tree (S2 sometimes truncates old papers)
# ===========================================================================
OA = "https://api.openalex.org/works"

# Resolve Whitmire & Wright, then list EVERYTHING that cites it.
get(OA + "/doi:10.1016/0019-1035(80)90253-5", OUT / "oa_ww1980.json")
time.sleep(1.5)
# cited_by via filter (W-id filled in by the parse step if needed); also do a
# direct search fallback.
get(OA + "?" + urllib.parse.urlencode(
    {"filter": "cites:doi:10.1016/0019-1035(80)90253-5", "per-page": "200"}),
    OUT / "oa_ww1980_citedby.json")
time.sleep(1.5)
get(OA + "?" + urllib.parse.urlencode(
    {"search": "nuclear waste spectrum technological extraterrestrial civilizations",
     "per-page": "10"}), OUT / "oa_ww1980_search.json")
time.sleep(1.5)

for name, q in {
    "przybylski_promethium": "promethium Przybylski star HD 101065",
    "technetium_peculiar": "technetium chemically peculiar stars detection",
    "engulfment_binaries": "planetary engulfment co-natal binaries abundance",
    "sparse_abundance_outlier": "single element abundance outlier survey stars",
    "techno_composition": "technosignature stellar photosphere composition",
}.items():
    get(OA + "?" + urllib.parse.urlencode({"search": q, "per-page": "25"}),
        OUT / f"oa_{name}.json")
    time.sleep(1.5)

# ===========================================================================
# 4. Crossref: bibliographic ground truth for the pre-arXiv references
# ===========================================================================
CR = "https://api.crossref.org/works"
for name, q in {
    "ww1980": "Nuclear waste spectrum as evidence of technological extraterrestrial civilizations",
    "merrill1952": "Technetium in the stars Merrill",
    "przybylski1961": "Przybylski unusual spectrum HD 101065",
    "cowley2004": "Cowley promethium Przybylski star",
    "butcher1987": "Butcher thorium age of the Galaxy nucleocosmochronology",
}.items():
    get(CR + "?" + urllib.parse.urlencode(
        {"query.bibliographic": q, "rows": "8",
         "mailto": "trimcrae@gmail.com"}), OUT / f"crossref_{name}.json")
    time.sleep(2.0)

# ===========================================================================
# 5. NIST Atomic Spectra Database: the FEASIBILITY TABLE input.
#    For every species of interest, pull ALL catalogued lines over the full
#    optical->H-band range so we can answer, from primary atomic data, which
#    public survey bands could possibly contain a usable line.
#    Raw responses are committed verbatim and parsed offline.
# ===========================================================================
NIST = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?"

NIST_SPECIES = [
    "Tc I", "Tc II", "Pm I", "Pm II", "Th II", "U II", "Pu I", "Np I",
    # reference/comparison species used in the feasibility discussion
    "Fe I", "Ba II", "Eu II", "Nd II", "Ce II", "La II", "Sr II", "Zr I",
    "Nb I", "Mo I", "Ru I", "P I", "K I", "Li I",
]

# Bands: full optical+NIR sweep, plus each survey band explicitly so a null
# result inside a survey band is recorded as its own file (no ambiguity).
NIST_BANDS = {
    "full": (3000, 18000),
    "blue": (3000, 5000),          # Tc/Pm/Th/U resonance region; GALAH B, BOSS
    "galah_b": (4713, 4903),
    "galah_v": (5648, 5873),
    "galah_r": (6478, 6737),
    "galah_i": (7585, 7887),
    "rvs": (8460, 8700),           # Gaia RVS / RAVE Ca-triplet region
    "apogee": (15100, 17000),      # APOGEE H band
}


def nist_url(species: str, lo: float, hi: float, fmt: int) -> str:
    params = {
        "spectra": species, "low_w": f"{lo:.2f}", "upp_w": f"{hi:.2f}",
        "unit": 0,              # Angstrom
        "format": fmt,          # 0=HTML 1=ASCII 2=tab-delimited 3=comma-delimited
        "line_out": 0, "en_unit": 0, "output": 0, "bibrefs": 0,
        "page_size": 15, "show_obs_wl": 1, "show_calc_wl": 1,
        "unc_out": 0, "order_out": 0, "show_av": 2, "tsb_value": 0,
        "A_out": 0, "intens_out": "on", "allowed_out": 1, "forbid_out": 1,
        "conf_out": "on", "term_out": "on", "enrg_out": "on", "J_out": "on",
        "g_out": "on", "remove_js": "on",
    }
    return NIST + urllib.parse.urlencode(params)


for species in NIST_SPECIES:
    tag = species.replace(" ", "")
    # The full sweep in the machine-readable format is the primary product.
    for fmt, ext in ((3, "csv"), (1, "ascii")):
        get(nist_url(species, 3000, 18000, fmt), OUT / f"nist_{tag}_full.{ext}")
        time.sleep(1.5)

# Per-survey-band pulls for the radionuclides only (keeps the run bounded):
for species in ("Tc I", "Tc II", "Pm I", "Pm II", "Th II", "U II"):
    tag = species.replace(" ", "")
    for band, (lo, hi) in NIST_BANDS.items():
        if band in ("full", "blue"):
            continue
        get(nist_url(species, lo, hi, 3), OUT / f"nist_{tag}_{band}.csv")
        time.sleep(1.5)

(OUT / "summary.json").write_text(json.dumps(STATUS, indent=2))
n_ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n{n_ok}/{len(STATUS)} fetches succeeded", flush=True)
