#!/usr/bin/env python3
"""Przybylski's star (HD 101065) literature survey: primary-source fetch on the runner.

The sandbox egress policy blocks arxiv / ADS / A&A / OUP / Springer / Crossref /
OpenAlex, so this runs on the GitHub Actions runner (CLAUDE.md acquisition
pattern) and commits verbatim primary sources back to the branch.

Everything is a plain GET of a public endpoint; no API keys required.

Outputs under results/przybylski_lit/:
  arxiv_ids.atom              arXiv metadata for named IDs (title/authors/abstract)
  arxiv_q_<name>.atom         arXiv API search results per query
  txt_<name>.txt              pdftotext of key papers (A&A open access, arXiv,
                              ADS scanned-article service)
  simbad_<name>.json/.txt     SIMBAD TAP bibliography + basic data + bibcode checks
  openalex_<name>.json        OpenAlex work records and citing-work lists
  crossref_<name>.json        Crossref bibliographic metadata (DOI verification)
  s2_<name>.json              Semantic Scholar records
  web_<name>.html             popular / blog discussion pages
  summary.json                per-URL fetch status
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/przybylski_lit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36 Seti-litsurvey/1.0 (mailto:trimcrae@gmail.com)"
    ),
    "Accept": "*/*",
}
STATUS: list[dict] = []


def get(url: str, dest: pathlib.Path, tries: int = 3, pause: float = 3.0,
        quiet_fail: bool = False) -> bytes | None:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:44s} {len(data):9d}B  {url[:110]}", flush=True)
            return data
        except Exception as e:  # noqa: BLE001
            if not quiet_fail:
                print(f"RETRY({i+1}) {url[:110]} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url[:130]}", flush=True)
    return None


def get_pdf_text(url: str, name: str, tries: int = 3) -> bool:
    """Fetch a PDF, convert to text, keep only the text (repo stays light)."""
    pdf = OUT / f"_tmp_{name}.pdf"
    data = get(url, pdf, tries=tries)
    if data is None:
        return False
    if not data[:5].startswith(b"%PDF"):
        # Not a PDF (login wall / HTML error page). Keep first 4 kB as evidence.
        (OUT / f"txt_{name}.txt").write_bytes(b"[NOT A PDF] " + data[:4000])
        pdf.unlink(missing_ok=True)
        print(f"     -> {name}: response was not a PDF", flush=True)
        return False
    try:
        txt = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            capture_output=True, timeout=300).stdout.decode("utf-8", "replace")
        (OUT / f"txt_{name}.txt").write_text(txt)
        print(f"     -> txt_{name}.txt  {len(txt)} chars", flush=True)
        ok = len(txt) > 500
    except Exception as e:  # noqa: BLE001
        print(f"     -> pdftotext failed for {name}: {e}", flush=True)
        ok = False
    pdf.unlink(missing_ok=True)
    return ok


# =====================================================================
# 1. arXiv metadata by ID  (arXiv returns the real title -> no guessing)
# =====================================================================
ARXIV = "http://export.arxiv.org/api/query?"

IDS = [
    "2304.13623",      # Andrievsky+ 2023 - is there promethium?
    "2308.04479",      # Andrievsky+ 2023 - technetium revisited
    "1804.07260",      # Hubrig+ 2018 - magnetic & pulsational variability
    "1004.0246",       # Shulyak+ 2010 - realistic model atmosphere HD 101065
    "astro-ph/0610611",  # Goriely 2007 - diffusion/accretion/nuclear reactions
    "astro-ph/0307464",  # Shavrina+ 2003 - Li I 6708 in HD 101065
    "1612.03632",      # Mathys - Ap stars w/ resolved magnetically split lines
    "astro-ph/9806290",  # How to drive roAp stars
    "2002.08075",      # Extended calculations lanthanide ions
    "2302.02487",      # (unidentified; resolve by fetching)
]
get(ARXIV + urllib.parse.urlencode({"id_list": ",".join(IDS), "max_results": 60}),
    OUT / "arxiv_ids.atom")
time.sleep(3.2)

# =====================================================================
# 2. arXiv searches
# =====================================================================
QUERIES = {
    "przybylski": 'all:"Przybylski"',
    "hd101065": 'all:"HD 101065" OR all:"HD101065"',
    "promethium_star": 'all:promethium',
    "technetium_ap": 'all:technetium AND (all:"Ap star" OR all:"peculiar")',
    "hd965": 'all:"HD 965"',
    "hd25354": 'all:"HD 25354" OR all:"HD25354"',
    "hr465": 'all:"HR 465" OR all:"HD 9996"',
    "actinide_cp": 'all:actinide AND (all:star OR all:stellar)',
    "roap_prototype": 'abs:"rapidly oscillating Ap"',
    "kurtz_roap": 'au:Kurtz AND abs:"rapidly oscillating"',
    "radiative_levitation_ap": 'abs:"radiative levitation" OR abs:"radiative diffusion" AND abs:"Ap star"',
    "stratification_roap": 'abs:stratification AND abs:"Ap star"',
    "ree_ap_abundance": 'abs:"rare earth" AND abs:"Ap stars"',
    "ryabchikova_101065": 'au:Ryabchikova',
    "mashonkina_nlte_nd": 'au:Mashonkina AND abs:"Nd"',
    "kochukhov_roap": 'au:Kochukhov AND abs:"roAp"',
    "nuclear_waste_seti": 'all:"nuclear waste" AND (all:SETI OR all:technosignature OR all:extraterrestrial OR all:civilization)',
    "whitmire_wright": 'all:Whitmire',
    "techno_spectroscopy": 'abs:technosignature AND (abs:spectrum OR abs:spectroscopy OR abs:atmosphere)',
    "artificial_abundance": 'abs:"artificial" AND abs:"chemical" AND (abs:civilization OR abs:technosignature)',
    "island_of_stability_star": 'all:"island of stability" AND (all:star OR all:stellar OR all:spectrum)',
    "th_u_ap": 'abs:thorium AND abs:"peculiar star"',
    "cowley_peculiar": 'au:Cowley_C AND abs:peculiar',
    "magnetic_hd101065": 'abs:"magnetic field" AND abs:"roAp"',
    "diffusion_ap_michaud": 'abs:diffusion AND abs:"chemically peculiar" AND abs:magnetic',
    "convection_a_star": 'abs:convection AND abs:"A-type star" AND abs:atmosphere',
    "polluted_wd_techno": 'abs:"white dwarf" AND abs:technosignature',
    "fivet_promethium": 'au:Fivet',
    "quinet_promethium": 'au:Quinet AND abs:promethium',
    "gopka": 'au:Gopka',
    "yushchenko": 'au:Yushchenko',
    "andrievsky_przybylski": 'au:Andrievsky AND abs:Przybylski',
}
for name, q in QUERIES.items():
    get(ARXIV + urllib.parse.urlencode(
        {"search_query": q, "max_results": "80", "sortBy": "relevance"}),
        OUT / f"arxiv_q_{name}.atom")
    time.sleep(3.2)

# =====================================================================
# 3. Full text: open-access publisher PDFs + arXiv PDFs + ADS scans
# =====================================================================
# --- A&A is fully open access -----------------------------------------------
AANDA = {
    # Cowley, Bidelman, Hubrig, Mathys, Bord 2004, A&A 419, 1087 (the Pm paper)
    "cowley2004_promethium": "https://www.aanda.org/articles/aa/pdf/2004/21/aa0726.pdf",
    # A&A 474, 307 (2007) atomic data for radioactive elements Ra/Ac/Th ...
    "aa2007_radioactive_atomicdata": "https://www.aanda.org/articles/aa/pdf/2007/40/aa8082-07.pdf",
}
for name, url in AANDA.items():
    get_pdf_text(url, name)
    time.sleep(2.5)

# --- arXiv PDFs (full text of the modern re-analyses) -----------------------
ARXIV_PDF = {
    "andrievsky2023_pm": "https://arxiv.org/pdf/2304.13623",
    "andrievsky2023_tc": "https://arxiv.org/pdf/2308.04479",
    "hubrig2018": "https://arxiv.org/pdf/1804.07260",
    "shulyak2010": "https://arxiv.org/pdf/1004.0246",
    "goriely2007": "https://arxiv.org/pdf/astro-ph/0610611",
    "shavrina2003_li": "https://arxiv.org/pdf/astro-ph/0307464",
}
for name, url in ARXIV_PDF.items():
    get_pdf_text(url, name)
    time.sleep(3.0)

# --- arXiv HTML (v2 renderings, for the newest papers) ----------------------
for aid in ["2304.13623", "2308.04479"]:
    if not get(f"https://arxiv.org/html/{aid}v1", OUT / f"html_{aid}.html", tries=2,
               quiet_fail=True):
        get(f"https://arxiv.org/html/{aid}", OUT / f"html_{aid}.html", tries=2,
            quiet_fail=True)
    time.sleep(3)

# --- ADS scanned-article service: probes double as BIBCODE VERIFICATION -----
# If the PDF comes back and its text matches the expected paper, the bibcode is
# confirmed; if it 404s the bibcode is NOT confirmed and must be reported so.
ADS_SCAN = {
    "przybylski1961_natur": "1961Natur.189..739P",
    "przybylski1963": "1963MNRAS.126..123P",
    "przybylski1966_natur": "1966Natur.210...20P",
    "przybylski1977": "1977MNRAS.178...71P",
    "kurtz1978_ibvs": "1978IBVS.1436....1K",
    "kurtz1982_mnras": "1982MNRAS.200..807K",
    "michaud1970_apj": "1970ApJ...160..641M",
    "wegner_petford1974": "1974MNRAS.168..557W",
    "cowley2000_hd101065": "2000A&A...358L..45C",
    "cowley1977_pm": "1977ApJ...214..470C",
    "bidelman1966": "1966ARA&A...4..357B",
    "kurtz1990_araa": "1990ARA&A..28..607K",
}
for name, bib in ADS_SCAN.items():
    get_pdf_text(f"https://articles.adsabs.harvard.edu/pdf/{urllib.parse.quote(bib)}",
                 f"adsscan_{name}__{bib.replace('/', '_')}", tries=2)
    time.sleep(2.5)

# --- IBVS is free at Konkoly ------------------------------------------------
for url in ["https://www.konkoly.hu/pub/ibvs/1401/1436.pdf",
            "https://konkoly.hu/pub/ibvs/1401/1436.pdf"]:
    if get_pdf_text(url, "ibvs1436_kurtz1978", tries=2):
        break
    time.sleep(2)

# =====================================================================
# 4. SIMBAD: object bibliography (verified bibcodes+DOIs) and basic data
# =====================================================================
SIMBAD_HOSTS = ["https://simbad.u-strasbg.fr/simbad", "https://simbad.cds.unistra.fr/simbad"]


def simbad_tap(name: str, adql: str) -> bool:
    for host in SIMBAD_HOSTS:
        url = host + "/sim-tap/sync?" + urllib.parse.urlencode(
            {"request": "doQuery", "lang": "adql", "format": "json",
             "maxrec": "5000", "query": adql})
        data = get(url, OUT / f"simbad_{name}.json", tries=2, quiet_fail=True)
        if data and b'"data"' in data:
            return True
        time.sleep(2)
    return False


# schema first, so the column names can be checked if a query fails
simbad_tap("schema_ref",
           "SELECT column_name, datatype FROM TAP_SCHEMA.columns "
           "WHERE table_name = 'ref'")
time.sleep(2)
simbad_tap("schema_hasref",
           "SELECT table_name, column_name FROM TAP_SCHEMA.columns "
           "WHERE table_name IN ('has_ref','ref','basic','allfluxes')")
time.sleep(2)

for star, tag in [("HD 101065", "hd101065"), ("HD 965", "hd965"),
                  ("HD 9996", "hd9996"), ("HD 25354", "hd25354")]:
    ok = simbad_tap(
        f"bib_{tag}",
        "SELECT r.bibcode, r.year, r.journal, r.volume, r.page, r.doi, r.title "
        "FROM ref AS r JOIN has_ref AS h ON h.oidbibref = r.oidbib "
        "JOIN ident AS i ON i.oidref = h.oidref "
        f"WHERE i.id = '{star}' ORDER BY r.year")
    if not ok:
        simbad_tap(
            f"bib_{tag}_alt",
            "SELECT r.bibcode, r.year, r.journal, r.volume, r.page, r.doi, r.title "
            "FROM ref AS r, has_ref AS h, ident AS i "
            "WHERE h.oidbibref = r.oidbibref AND i.oidref = h.oidref "
            f"AND i.id = '{star}' ORDER BY r.year")
    time.sleep(2)

simbad_tap("basic_hd101065",
           "SELECT b.main_id, b.ra, b.dec, b.coo_bibcode, b.sp_type, b.sp_bibcode, "
           "b.plx_value, b.plx_err, b.plx_bibcode, b.rvz_radvel, b.otype_txt "
           "FROM basic AS b JOIN ident AS i ON i.oidref = b.oid "
           "WHERE i.id = 'HD 101065'")
time.sleep(2)
simbad_tap("flux_hd101065",
           "SELECT * FROM allfluxes AS f JOIN ident AS i ON i.oidref = f.oidref "
           "WHERE i.id = 'HD 101065'")
time.sleep(2)

# SIMBAD ASCII bibliography (independent of TAP schema drift)
for star, tag in [("HD+101065", "hd101065"), ("HD+965", "hd965"),
                  ("HD+25354", "hd25354")]:
    get(SIMBAD_HOSTS[0] + f"/sim-id?Ident={star}&output.format=ASCII"
        "&obj.bibsel=on&bibyear1=1850&bibyear2=2030&bibdisplay=refsum&biborder=year",
        OUT / f"simbad_ascii_{tag}.txt", tries=2)
    time.sleep(2.5)

# Bibcode verification via SIMBAD's reference resolver
CANDIDATE_BIBCODES = [
    "1961Natur.189..739P", "1963MNRAS.126..123P", "1966Natur.210...20P",
    "1977MNRAS.178...71P", "1978IBVS.1436....1K", "1982MNRAS.200..807K",
    "2004A&A...419.1087C", "2007MNRAS.380..771F", "2023AN....34430056A",
    "2018MNRAS.477.3791H", "2010A&A...520A..88S", "2000A&A...358L..45C",
    "1970ApJ...160..641M", "1990ARA&A..28..607K", "2008KPCB...24...89G",
]
bibcheck = {}
for bib in CANDIDATE_BIBCODES:
    dest = OUT / ("simbad_refchk_" + re.sub(r"[^A-Za-z0-9]", "_", bib) + ".txt")
    d = get(SIMBAD_HOSTS[0] + "/sim-ref?" + urllib.parse.urlencode(
        {"bibcode": bib, "output.format": "ASCII"}), dest, tries=2, quiet_fail=True)
    bibcheck[bib] = bool(d and b"Bibcode" in d)
    time.sleep(2)
(OUT / "simbad_bibcode_check.json").write_text(json.dumps(bibcheck, indent=2))

# =====================================================================
# 5. OpenAlex: authoritative DOIs + full citing-work lists
# =====================================================================
OA = "https://api.openalex.org"
MAIL = "mailto=trimcrae@gmail.com"


def oa(name: str, path: str) -> dict | None:
    sep = "&" if "?" in path else "?"
    d = get(f"{OA}{path}{sep}{MAIL}", OUT / f"openalex_{name}.json", tries=3)
    if d:
        try:
            return json.loads(d)
        except Exception:  # noqa: BLE001
            return None
    return None


OA_SEARCHES = {
    "przybylski_all": "/works?search=Przybylski%20star&per-page=200",
    "przybylski_title": "/works?filter=title.search:Przybylski&per-page=200",
    "hd101065_title": "/works?filter=title.search:101065&per-page=200",
    "promethium_stellar": "/works?filter=title.search:promethium&per-page=200",
    "nuclear_waste_eti": "/works?search=nuclear%20waste%20extraterrestrial%20civilization%20spectrum&per-page=50",
    "whitmire_wright1980": "/works?filter=title.search:nuclear%20waste&per-page=100",
    "actinide_ap": "/works?search=actinides%20chemically%20peculiar%20star%20spectrum&per-page=50",
    "hd25354_radioactive": "/works?search=HD%2025354%20radioactive%20elements%20island%20of%20stability&per-page=25",
    "technetium_przybylski": "/works?search=technetium%20Przybylski&per-page=50",
    "roap_kurtz1982": "/works?search=rapidly%20oscillating%20Ap%20stars%20Kurtz&per-page=25",
}
for name, path in OA_SEARCHES.items():
    oa(name, path)
    time.sleep(1.2)

# Resolve seed works by DOI/title and pull their complete citing-work lists.
SEED_TITLES = {
    "ww1980": "Nuclear waste spectrum as evidence of technological extraterrestrial civilization",
    "cowley2004": "On the possible presence of promethium in the spectra of HD 101065 and HD 965",
    "gopka2008": "Identification of absorption lines of short half-life actinides in the spectrum of Przybylski's star",
    "fivet2007": "Transition probabilities in singly ionized promethium and the identification of Pm II lines",
    "andrievsky2023pm": "An enigma of Przybylski's star: is there promethium on its surface",
    "andrievsky2023tc": "Abundance of radioactive technetium in Przybylski's star revisited",
    "hubrig2018": "Magnetic and pulsational variability of Przybylski's star",
    "kurtz1982": "Rapidly oscillating Ap stars",
    "michaud1970": "Diffusion processes in peculiar A stars",
    "shulyak2010": "Realistic model atmosphere and revised abundances of the coolest Ap star HD 101065",
}
for name, title in SEED_TITLES.items():
    rec = oa(name + "_resolve",
             "/works?filter=title.search:" + urllib.parse.quote(title) + "&per-page=10")
    time.sleep(1.2)
    wid = None
    if rec and rec.get("results"):
        wid = rec["results"][0].get("id", "").rsplit("/", 1)[-1]
    if wid:
        oa(name + "_citedby",
           f"/works?filter=cites:{wid}&per-page=200&select=id,doi,title,publication_year,"
           "primary_location,authorships,type,cited_by_count")
        time.sleep(1.2)

# =====================================================================
# 6. Crossref: DOI-level verification of the citations used in the survey
# =====================================================================
CR = "https://api.crossref.org/works?"
CR_QUERIES = {
    "ww1980_icarus": "Nuclear waste spectrum evidence technological extraterrestrial civilization Icarus 1980",
    "cowley2004": "possible presence of promethium spectra HD 101065 Przybylski star HD 965",
    "fivet2007": "Transition probabilities singly ionized promethium identification Pm II Przybylski HR 465",
    "andrievsky2023pm": "enigma of Przybylski star is there promethium on its surface",
    "andrievsky2023tc": "Abundance of radioactive technetium in Przybylski star revisited",
    "gopka2008": "Identification absorption lines short half-life actinides spectrum Przybylski star",
    "hubrig2018": "Magnetic and pulsational variability of Przybylski star HD 101065",
    "shulyak2010": "Realistic model atmosphere revised abundances coolest Ap star HD 101065",
    "kurtz1982": "Rapidly oscillating Ap stars Kurtz 1982 Monthly Notices",
    "michaud1970": "Diffusion processes in peculiar A stars Michaud 1970 Astrophysical Journal",
    "hd25354_galaxies2024": "radioactive elements atmosphere HD 25354 island of stability symmetric decay",
    "cpd62_thorium": "Confident detection of doubly ionized thorium extreme Ap star CPD-62 2717",
    "mashonkina2005": "Non-LTE line formation for Pr and Nd in roAp stars",
    "ryabchikova2004": "Rare earth elements in the atmospheres of roAp stars stratification",
    "przybylski1961": "Przybylski peculiar star HD 101065 Nature 1961",
    "kurtz1978_ibvs": "12.15 minute light variations in Przybylski star HD 101065",
}
for name, q in CR_QUERIES.items():
    get(CR + urllib.parse.urlencode({"query.bibliographic": q, "rows": "6",
                                     "mailto": "trimcrae@gmail.com"}),
        OUT / f"crossref_{name}.json", tries=3)
    time.sleep(1.5)

# =====================================================================
# 7. Semantic Scholar (abstracts + reference/citation graph)
# =====================================================================
S2 = "https://api.semanticscholar.org/graph/v1/paper/"
S2F = "title,year,abstract,externalIds,venue,authors,citationCount"
for name, q in {
    "przybylski_promethium": "promethium Przybylski star HD 101065",
    "przybylski_actinides": "actinides Przybylski star short half-life",
    "nuclear_waste_eti": "nuclear waste spectrum technological extraterrestrial civilization",
    "roap_discovery": "rapidly oscillating Ap stars discovery Kurtz",
    "hd101065_atmosphere": "HD 101065 model atmosphere abundances",
}.items():
    get(S2 + "search?" + urllib.parse.urlencode({"query": q, "fields": S2F, "limit": "20"}),
        OUT / f"s2_{name}.json", tries=3, pause=6.0)
    time.sleep(3.5)

# =====================================================================
# 8. Popular / non-refereed but serious discussion (Q5: refereed vs popular)
# =====================================================================
WEB = {
    "wikipedia_przybylski": "https://en.wikipedia.org/wiki/Przybylski%27s_Star",
    "wikipedia_przybylski_raw": "https://en.wikipedia.org/w/index.php?title=Przybylski%27s_Star&action=raw",
    "astrowright_search": "https://sites.psu.edu/astrowright/?s=Przybylski",
    "astrowright_iv": "https://sites.psu.edu/astrowright/2017/03/16/przybylskis-star-iv-or/",
    "centauri_dreams_2026": "https://www.centauri-dreams.org/2026/05/15/przybylskis-star-still-bizarre-after-all-these-years/",
}
for name, url in WEB.items():
    get(url, OUT / f"web_{name}.html", tries=2, quiet_fail=True)
    time.sleep(2)

# AstroWright series: crawl the search page for the other parts
try:
    h = (OUT / "web_astrowright_search.html").read_text(errors="replace")
    links = sorted(set(re.findall(
        r"https://sites\.psu\.edu/astrowright/\d{4}/\d{2}/\d{2}/[a-z0-9\-]*przybylski[a-z0-9\-]*/", h)))
    print(f"AstroWright Przybylski posts found: {links}", flush=True)
    for i, u in enumerate(links[:8]):
        get(u, OUT / f"web_astrowright_{i}.html", tries=2, quiet_fail=True)
        time.sleep(2)
except Exception as e:  # noqa: BLE001
    print(f"astrowright crawl skipped: {e}", flush=True)

# =====================================================================
(OUT / "summary.json").write_text(json.dumps(STATUS, indent=2))
n_ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n{n_ok}/{len(STATUS)} fetches succeeded", flush=True)
for s in STATUS:
    if not s.get("ok"):
        print(f"  MISSING: {s['file']}  <- {s['url'][:120]}", flush=True)
