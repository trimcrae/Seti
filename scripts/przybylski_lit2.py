#!/usr/bin/env python3
"""Przybylski's star literature survey, pass 2: open-access full text + gap filling.

Pass 1 (scripts/przybylski_lit.py) pulled arXiv/SIMBAD/OpenAlex/Crossref metadata
and the A&A + ADS-scan PDFs. Pass 2 goes after the remaining primary sources that
are open access at the publisher, plus deeper citation enumeration for the
technosignature question (who, if anyone, has floated an artificial explanation
for HD 101065 in the refereed literature).
"""
from __future__ import annotations

import json
import pathlib
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
            print(f"OK   {dest.name:48s} {len(data):9d}B  {url[:105]}", flush=True)
            return data
        except Exception as e:  # noqa: BLE001
            if not quiet_fail:
                print(f"RETRY({i+1}) {url[:105]} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url[:130]}", flush=True)
    return None


def get_pdf_text(url: str, name: str, tries: int = 3) -> bool:
    pdf = OUT / f"_tmp2_{name}.pdf"
    data = get(url, pdf, tries=tries)
    if data is None:
        return False
    if not data[:5].startswith(b"%PDF"):
        (OUT / f"txt_{name}.txt").write_bytes(b"[NOT A PDF] " + data[:4000])
        pdf.unlink(missing_ok=True)
        return False
    try:
        txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                             capture_output=True, timeout=300).stdout.decode("utf-8", "replace")
        (OUT / f"txt_{name}.txt").write_text(txt)
        print(f"     -> txt_{name}.txt  {len(txt)} chars", flush=True)
        ok = len(txt) > 500
    except Exception as e:  # noqa: BLE001
        print(f"     -> pdftotext failed {name}: {e}", flush=True)
        ok = False
    pdf.unlink(missing_ok=True)
    return ok


# =====================================================================
# 1. Open-access publisher full text
# =====================================================================
OA_PDFS = {
    # A&A (fully open access)
    "goriely2007_aa466_619": "https://www.aanda.org/articles/aa/pdf/2007/17/aa6583-06.pdf",
    "shulyak2010_aa520_A88": "https://www.aanda.org/articles/aa/pdf/2010/12/aa14803-10.pdf",
    "cowley2000_aa358_L45": "https://www.aanda.org/articles/aa/full/2000/23/aagg214/aagg214.pdf",
    "mashonkina2005_aa441_309": "https://www.aanda.org/articles/aa/pdf/2005/38/aa2957-05.pdf",
    "mashonkina2009_aa495_297": "https://www.aanda.org/articles/aa/pdf/2009/07/aa10527-08.pdf",
    "ryabchikova2004_aa423_705": "https://www.aanda.org/articles/aa/pdf/2004/32/aa0959-04.pdf",
    # Frontiers in Chemistry 2020 promethium review (open access)
    "frontiers2020_promethium": "https://www.frontiersin.org/journals/chemistry/articles/10.3389/fchem.2020.00588/pdf",
    # MDPI Galaxies 2024: radioactive elements in HD 25354 (open access)
    "galaxies2024_hd25354": "https://www.mdpi.com/2075-4434/12/5/57/pdf",
    # Proceedings of Science (open access) - Gopka/Panov on Przybylski
    "pos_nic9_198": "https://pos.sissa.it/028/198/pdf",
}
for name, url in OA_PDFS.items():
    get_pdf_text(url, name)
    time.sleep(2.5)

# HTML fallbacks where the PDF path guess may be wrong
OA_HTML = {
    "frontiers2020_promethium_html":
        "https://www.frontiersin.org/journals/chemistry/articles/10.3389/fchem.2020.00588/full",
    "galaxies2024_hd25354_html": "https://www.mdpi.com/2075-4434/12/5/57",
    "aanda_search_cowley":
        "https://www.aanda.org/component/search/?searchword=Przybylski&searchphrase=all&Itemid=178",
}
for name, url in OA_HTML.items():
    get(url, OUT / f"web_{name}.html", tries=2, quiet_fail=True)
    time.sleep(2.5)

# =====================================================================
# 2. More ADS scanned articles (each success also VERIFIES the bibcode)
# =====================================================================
ADS_SCAN2 = {
    "przybylski1961_natur_alt": "1961Natur.189..739P",
    "przybylski1962": "1962Natur.194..760P",
    "przybylski1965": "1965Natur.205..163P",
    "przybylski1966_natur": "1966Natur.210...20P",
    "przybylski1967_mnras": "1967MNRAS.135...79P",
    "przybylski_kodaira1970": "1970PASJ...22..429P",
    "wegner1976": "1976MNRAS.177..99W",
    "cowley1973_pm_hr465": "1973ApJ...180..121C",
    "aller_cowley1970": "1970ApJ...162L.145A",
    "kurtz1980_mnras": "1980MNRAS.191..115K",
    "kurtz1982_mnras": "1982MNRAS.200..807K",
    "kurtz1990_araa": "1990ARA&A..28..607K",
    "michaud1970_apj": "1970ApJ...160..641M",
    "michaud1976_apj": "1976ApJ...210..447M",
    "babcock1958": "1958ApJS....3..141B",
    "cowley2000_mnras317": "2000MNRAS.317..299C",
    "whitmire_wright1980_icarus": "1980Icar...42..149W",
    "wegner_petford1974": "1974MNRAS.168..557W",
}
for name, bib in ADS_SCAN2.items():
    get_pdf_text(f"https://articles.adsabs.harvard.edu/pdf/{urllib.parse.quote(bib)}",
                 f"adsscan_{name}__{bib.replace('/', '_')}", tries=2)
    time.sleep(2.0)

# =====================================================================
# 3. Technosignature question: exhaustive citation enumeration
# =====================================================================
OA = "https://api.openalex.org"
MAIL = "mailto=trimcrae@gmail.com"

OA_PATHS = {
    # every OpenAlex work whose title/abstract mentions Przybylski
    "przybylski_fulltext": "/works?filter=default.search:Przybylski&per-page=200",
    "przybylski_abstract": "/works?filter=abstract.search:Przybylski&per-page=200",
    "hd101065_abstract": "/works?filter=abstract.search:101065&per-page=200",
    # the technosignature side
    "ww1980_by_doi": "/works/doi:10.1016/0019-1035(80)90253-5",
    "techno_nuclear_waste": "/works?filter=default.search:nuclear%20waste%20technosignature&per-page=100",
    "eti_stellar_pollution": "/works?filter=abstract.search:extraterrestrial%20AND%20stellar%20AND%20pollution&per-page=100",
    # actinide / transuranic side
    "transuranic_stellar": "/works?filter=default.search:transuranic%20stellar%20spectrum&per-page=100",
    "einsteinium_star": "/works?filter=default.search:einsteinium%20star%20spectrum&per-page=50",
}
for name, path in OA_PATHS.items():
    sep = "&" if "?" in path else "?"
    get(f"{OA}{path}{sep}{MAIL}", OUT / f"openalex2_{name}.json", tries=3)
    time.sleep(1.3)

# Crossref: resolve remaining refs precisely
CR = "https://api.crossref.org/works?"
CR_Q = {
    "ww1980_exact": "Whitmire Wright nuclear waste spectrum evidence technological extraterrestrial civilization",
    "cowley2000_mnras": "Cowley Ryabchikova Kupka Bord Mathys Bidelman abundances Przybylski star",
    "kurtz1990_araa": "Kurtz rapidly oscillating Ap stars Annual Review Astronomy Astrophysics 1990",
    "goriely2007": "Goriely interplay diffusion accretion nuclear reactions atmospheres Sirius Przybylski",
    "przybylski1961_nature": "Przybylski new type of peculiar star Nature 1961",
    "ryabchikova_stratification": "Ryabchikova stratification rare earth elements atmospheres roAp stars",
    "kochukhov_shulyak": "Kochukhov Shulyak model atmospheres magnetic chemically peculiar stars",
    "alecian_stift": "Alecian Stift diffusion of elements in magnetic stellar atmospheres",
    "theado_diffusion": "Theado Vauclair Alecian Le Blanc atomic diffusion magnetic Ap stars",
}
for name, q in CR_Q.items():
    get(CR + urllib.parse.urlencode({"query.bibliographic": q, "rows": "8",
                                     "mailto": "trimcrae@gmail.com"}),
        OUT / f"crossref2_{name}.json", tries=3)
    time.sleep(1.5)

# =====================================================================
(OUT / "summary2.json").write_text(json.dumps(STATUS, indent=2))
n_ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\npass2: {n_ok}/{len(STATUS)} fetches succeeded", flush=True)
for s in STATUS:
    if not s.get("ok"):
        print(f"  MISSING: {s['file']}  <- {s['url'][:120]}", flush=True)
