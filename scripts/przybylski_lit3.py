#!/usr/bin/env python3
"""Przybylski's star literature survey, pass 3: gap fill via ADS link_gateway.

Passes 1-2 left specific gaps. This pass resolves publisher/eprint PDFs *by
bibcode* through the public ADS link_gateway (no API key needed), which has the
useful side effect of verifying each bibcode: a gateway hit that yields the
expected paper confirms the identifier, a miss means the bibcode is unverified
and must be reported as such.

Targets: the historical Pm chain (Aller & Cowley 1970; Hartoog et al. 1973 WCS;
Cowley et al. 1977), the roAp discovery/analysis chain (Kurtz & Wegner 1979,
Martinez & Kurtz 1990, Mkrtichian et al. 2008), the diffusion/stratification
theory chain (Michaud 1970; Ryabchikova, Mashonkina, Kochukhov), Goriely 2007,
the HD 25354 island-of-stability paper, and the SETI-side texts (Wright's
Handbook of Exoplanets chapter, the Breakthrough Listen Exotica Catalog).
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
            print(f"OK   {dest.name:52s} {len(data):9d}B", flush=True)
            return data
        except Exception as e:  # noqa: BLE001
            if not quiet_fail:
                print(f"RETRY({i+1}) {url[:100]} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url[:120]}", flush=True)
    return None


def to_text(data: bytes, name: str) -> bool:
    pdf = OUT / f"_tmp3_{name}.pdf"
    pdf.write_bytes(data)
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


def ads_pdf(bibcode: str, name: str) -> bool:
    """Try every ADS link_gateway PDF route for a bibcode; keep the first hit."""
    enc = urllib.parse.quote(bibcode, safe="")
    for route in ("PUB_PDF", "EPRINT_PDF", "ARTICLE", "ADS_PDF"):
        data = get(f"https://ui.adsabs.harvard.edu/link_gateway/{enc}/{route}",
                   OUT / f"_probe_{name}.bin", tries=2, quiet_fail=True)
        if data and data[:5].startswith(b"%PDF"):
            print(f"  [{bibcode}] via {route}", flush=True)
            ok = to_text(data, f"{name}__{bibcode.replace('/', '_')}")
            (OUT / f"_probe_{name}.bin").unlink(missing_ok=True)
            return ok
        time.sleep(1.5)
    (OUT / f"_probe_{name}.bin").unlink(missing_ok=True)
    print(f"  [{bibcode}] NO PDF ROUTE (bibcode unverified by this method)", flush=True)
    return False


BIBCODES = {
    # --- historical promethium chain ---
    "aller_cowley1970_hr465": "1970ApJ...162L.145A",
    "hartoog1973_wcs": "1973ApJ...182..847H",
    "cowley1977_hd101065_lineid": "1977ApJ...216...37C",
    "cowley2000_abundances": "2000MNRAS.317..299C",
    "fivet2007_pmii": "2007MNRAS.380..771F",
    "fivet2007_erratum": "2007MNRAS.382..944F",
    # --- roAp discovery / pulsation ---
    "kurtz1978_ibvs": "1978IBVS.1436....1K",
    "kurtz_wegner1979": "1979ApJ...232..510K",
    "kurtz1982_roap": "1982MNRAS.200..807K",
    "martinez_kurtz1990": "1990MNRAS.242..636M",
    "mkrtichian2008_seismo": "2008A&A...490.1109M",
    "kurtz1990_araa": "1990ARA&A..28..607K",
    # --- diffusion / stratification theory ---
    "michaud1970_diffusion": "1970ApJ...160..641M",
    "ryabchikova2004_roap_ree": "2004A&A...423..705R",
    "mashonkina2005_nd": "2005A&A...441..309M",
    "mashonkina2009_pr": "2009A&A...495..297M",
    "cowley2001_corewing": "2001A&A...374L..19C",
    "kochukhov2002_corewing": "2002A&A...389..420K",
    # --- nucleosynthesis / actinides ---
    "goriely2007_accelparticles": "2007A&A...466..619G",
    "dzuba2017_island": "2017PhRvA..95f2515D",
    "arnould_goriely2020": "2020PrPNP.11203766A",
    "chojnowski2023_thoriumIII": "2023MNRAS.522.5931C",
    "gopka2008_actinides": "2008KPCB...24...89G",
    "pyper_hartoog1975_hd25354": "1975ApJ...198..555P",
    # --- magnetic field history ---
    "wolff_hagen1976": "1976PASP...88..119W",
    "mathys1997_modulus": "1997A&AS..123..353M",
    "scholler2012_multiplicity": "2012A&A...545A..38S",
    # --- SETI side ---
    "wright2018_exoplanets_seti": "2018haex.bookE.186W",
    "lacki2021_exotica": "2021ApJS..257...42L",
}
for name, bib in BIBCODES.items():
    ads_pdf(bib, name)
    time.sleep(2.0)

# --- direct arXiv PDFs for the SETI-side and review texts -------------------
ARXIV_PDF = {
    "wright2018_exoplanets_seti_arxiv": "1707.02175",
    "arnould_goriely2020_arxiv": "2001.11228",
    "dzuba2017_arxiv": "1703.04250",
    "gopka2007_neutronstar_arxiv": "0712.2409",
    "andrievsky2023_deuterium": "2302.02487",
    "cowley2000_corewing_arxiv": "astro-ph/0012102",
    "gopka2003_abundpatterns": "astro-ph/0308339",
}
for name, aid in ARXIV_PDF.items():
    d = get(f"https://arxiv.org/pdf/{aid}", OUT / f"_tmp3_{name}.pdf", tries=2)
    if d and d[:5].startswith(b"%PDF"):
        to_text(d, name)
    (OUT / f"_tmp3_{name}.pdf").unlink(missing_ok=True)
    time.sleep(3)

# Breakthrough Listen Exotica Catalog: resolve the arXiv ID, then pull the PDF
ARXIV = "http://export.arxiv.org/api/query?"
get(ARXIV + urllib.parse.urlencode(
    {"search_query": 'ti:"Exotica Catalog" OR abs:"Exotica Catalog"',
     "max_results": "20"}), OUT / "arxiv_q_exotica.atom")
time.sleep(3.2)
get(ARXIV + urllib.parse.urlencode(
    {"search_query": 'abs:"Przybylski" AND (abs:SETI OR abs:technosignature OR abs:artificial)',
     "max_results": "40"}), OUT / "arxiv_q_przybylski_seti.atom")
time.sleep(3.2)

# --- MDPI Galaxies 2024 (HD 25354, island of stability), open access --------
for u, n in [("https://www.mdpi.com/2075-4434/12/5/57/pdf", "galaxies2024_hd25354"),
             ("https://www.mdpi.com/2075-4434/12/5/57", "galaxies2024_hd25354_html")]:
    d = get(u, OUT / f"_tmp3_{n}.bin", tries=2, quiet_fail=True)
    if d and d[:5].startswith(b"%PDF"):
        to_text(d, n)
    elif d:
        (OUT / f"web_{n}.html").write_bytes(d)
    (OUT / f"_tmp3_{n}.bin").unlink(missing_ok=True)
    time.sleep(2.5)

# --- Crossref: remaining conference/less-indexed references ------------------
CR = "https://api.crossref.org/works?"
for name, q in {
    "aller_cowley1970": "Aller Cowley possible presence of promethium HR 465 Astrophysical Journal Letters 1970",
    "hartoog1973_wcs": "Hartoog Cowley wavelength coincidence statistics peculiar stars 1973",
    "bidelman2005_tc": "Bidelman Tc and other unstable elements in Przybylski star cosmic abundances records stellar evolution",
    "ryabchikova2008_coska": "Ryabchikova promethium technetium Przybylski Contributions Astronomical Observatory Skalnate Pleso 2008",
    "gopka2004_iaus224": "Gopka Yushchenko Shavrina Mkrtichian Hatzes A-Star Puzzle IAU Symposium 224 Przybylski",
    "kurtz2002_iauc187": "Kurtz HD 101065 Przybylski star a most peculiar star",
    "elkin2015_roap_rv": "Elkin Kurtz Nesvacil radial velocity pulsation amplitudes rare earth roAp",
    "mathys2017_longperiod": "Mathys Ap stars with resolved magnetically split lines rotation periods",
}.items():
    get(CR + urllib.parse.urlencode({"query.bibliographic": q, "rows": "6",
                                     "mailto": "trimcrae@gmail.com"}),
        OUT / f"crossref3_{name}.json", tries=3)
    time.sleep(1.5)

(OUT / "summary3.json").write_text(json.dumps(STATUS, indent=2))
n_ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\npass3: {n_ok}/{len(STATUS)} fetches succeeded", flush=True)
