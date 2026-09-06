#!/usr/bin/env python3
"""FALLOUT novelty check: fetch the fission-pattern prior art on the runner.

The sandbox blocks arxiv.org / OpenAlex / Crossref (``CONNECT tunnel failed,
response 403``); the GitHub Actions runner has egress. This script pulls
**verbatim** metadata, abstracts, full text and citation trees for everything
that could sink or subsume the FALLOUT channel and commits the evidence to
``results/falloutlit/`` so the novelty adjudication in ``docs/fallout.md`` can
be done offline, from primary sources, and never from memory.

Modelled on ``scripts/tailingslit_fetch.py``, including the lesson it encodes:
a hardcoded arXiv identifier that resolves cleanly is **no evidence** that it
is the right paper. Every decisive paper is resolved by title search and the
returned title is verified token-by-token before anything is written; slugs
that fail verification get no file and must not be cited.

The questions the fetch has to answer:

1. Has anyone searched survey abundances for the **multi-element fission-yield
   pattern** (as opposed to Tc/Pm lines, or a single-element anomaly)?
   Whitmire & Wright 1980 predicted Pr and Nd as the most overabundant
   products; did anyone ever look for the vector?
2. What do the executed chemical-technosignature searches (Huang, Tao & Zhang
   2026 on polluted white dwarfs; the TAILINGS lineage) actually test, and does
   any of them include a fission template?
3. What are the current s-/r-process decompositions and their uncertainties
   (Arlandini 1999, Bisterzo 2014, Prantzos 2020), and what is the observed
   [Nd/Ba]-[Eu/Nd] distribution of GALAH dwarfs? (Sets how far outside natural
   space the fission vector sits.)
4. What is known about **anomalous Nd/Ba and Ce/Ba ratios** in dwarfs, young-star
   Ba over-estimation, and Ba NLTE/saturation -- the vetoes.

Outputs under ``results/falloutlit/``:
  arxiv_id_<name>.atom   arXiv API metadata (verbatim title + abstract)
  arxiv_q_<name>.atom    arXiv API keyword search (discovery), verbatim abstracts
  ar5iv_<name>.html      ar5iv full-text rendering
  txt_<name>.txt         extracted plain text (PDF -> pdftotext)
  oa_<name>.json         OpenAlex work record
  citedby_<name>_pN.json OpenAlex "works citing this"
  crossref_<name>.json   Crossref record (pre-arXiv literature)
  verification.json      the title-verification ledger
  summary.json           fetch status for every URL
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/falloutlit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-falloutlit/1.0 (mailto:trimcrae@gmail.com)"}
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
# 1. Papers whose FULL TEXT decides the novelty verdict: (title, must-tokens).
# --------------------------------------------------------------------------
FULLTEXT: dict[str, tuple[str, tuple[str, ...]]] = {
    # The one executed chemical technosignature search on photospheres.
    "huang2026_wd_technosignature": (
        "A Calibrated Bayesian Search for Potential Chemical Technosignatures "
        "in Polluted White Dwarf",
        ("technosignature", "white dwarf"),
    ),
    # The 2026 flagship review: its "Stellar Pollution" paragraph.
    "technosig_review_2026": (
        "The Search for Technosignatures: a Review of Possibilities",
        ("technosignature", "review"),
    ),
    # --- s/r decomposition: the templates' provenance.
    "arlandini1999_neutron_capture": (
        "Neutron Capture in Low-Mass Asymptotic Giant Branch Stars: Cross "
        "Sections and Abundance Signatures",
        ("neutron capture", "asymptotic"),
    ),
    "bisterzo2014_s_process_gce": (
        "Galactic Chemical Evolution and Solar s-process Abundances: Dependence "
        "on the 13C-pocket Structure",
        ("s-process", "chemical evolution"),
    ),
    "prantzos2020_neutron_capture_gce": (
        "Chemical evolution with rotating massive star yields II. A new "
        "assessment of the solar s- and r-process components",
        ("s-", "r-process"),
    ),
    "sneden2008_r_process_review": (
        "Neutron-Capture Elements in the Early Galaxy",
        ("neutron-capture", "early galaxy"),
    ),
    # --- the natural classes the vetoes must reject.
    "rekhi2025_s_process_dwarfs": (
        "s-process-enhanced dwarfs",
        ("s-process", "dwarf"),
    ),
    "karinkuzhi2021_lamost_barium_followup": (
        "Sr and Ba enrichment in barium stars",
        ("barium",),
    ),
    "hansen2018_r_process_alliance": (
        "The R-Process Alliance: First Release from the Southern Search for "
        "R-process-enhanced Stars",
        ("r-process", "alliance"),
    ),
    # --- the young-star barium problem and Ba NLTE.
    "dorazi2009_young_ba": (
        "Enhanced production of barium in low-mass stars: evidence from open clusters",
        ("barium", "open cluster"),
    ),
    "baratella2020_young_ba_gaia_eso": (
        "The Gaia-ESO Survey: a new approach to chemically characterising young "
        "open clusters",
        ("gaia-eso", "young"),
    ),
    "korotin2015_ba_nlte": (
        "Non-LTE barium abundance in dwarfs and subgiants",
        ("barium", "non-lte"),
    ),
    # --- surveys and their n-capture panels.
    "galah_dr4": ("The GALAH Survey: Data Release 4", ("galah",)),
    "galah_dr3_ncapture": (
        "The GALAH survey: chemical tagging and chrono-chemodynamics",
        ("galah",),
    ),
    "griffith2022_galah_residuals": (
        "Residual Abundances in GALAH DR3: Implications for Nucleosynthesis and "
        "Identification of Unique Stellar Populations",
        ("galah", "residual"),
    ),
    "apogee_dr17": (
        "The Seventeenth Data Release of the Sloan Digital Sky Surveys",
        ("data release", "sloan"),
    ),
    # --- the Ap-star claims that are the only prior "fission" spectroscopy.
    "andrievsky2023_przybylski_tc": (
        "Technetium in Przybylski's star",
        ("przybylski",),
    ),
    "cowley2004_hd101065_actinides": (
        "Detection of short-lived radioactive elements in Przybylski's star",
        ("przybylski",),
    ),
}

VERIFY: list[dict] = []


def resolve_by_title(title: str, must: tuple[str, ...]) -> tuple[str | None, str]:
    """Find an arXiv id by title search and VERIFY the title that comes back."""
    q = urllib.parse.quote(f'ti:"{title}"')
    url = f"http://export.arxiv.org/api/query?search_query={q}&max_results=8"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=90) as r:
            xml = r.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return None, f"title search failed: {exc!r}"
    time.sleep(PAUSE)
    entries = re.findall(r"<entry>(.*?)</entry>", xml, re.S)
    for e in entries:
        m_id = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", e)
        m_ti = re.search(r"<title>(.*?)</title>", e, re.S)
        if not (m_id and m_ti):
            continue
        got = " ".join(m_ti.group(1).split())
        low = got.lower()
        if all(tok.lower() in low for tok in must):
            return m_id.group(1).split("v")[0], got
    return None, f"no title match among {len(entries)} results"


# --------------------------------------------------------------------------
# 2. Discovery searches: has ANYONE looked for the fission VECTOR?
#    Every hit's title and abstract is stored verbatim in the Atom file.
# --------------------------------------------------------------------------
ARXIV_Q: dict[str, str] = {
    # The direct question, several phrasings.
    "fission_products_stellar_photosphere": 'all:"fission products" AND all:"stellar photosphere"',
    "fission_products_photosphere_abundance": 'all:"fission" AND all:"photospher" AND all:"abundance"',
    "fission_yield_pattern_star": 'all:"fission yield" AND all:"star" AND all:"abundance"',
    "nuclear_waste_technosignature_star": 'all:"nuclear waste" AND all:"star"',
    "nuclear_waste_technosignature": 'all:"nuclear waste" AND all:"technosignature"',
    "whitmire_wright_1980_nuclear_waste": 'all:"Whitmire" AND all:"nuclear waste"',
    "artificial_abundance_pattern_technosignature": 'all:"technosignature" AND all:"abundance pattern"',
    "artificial_abundance_technosignature": 'abs:"technosignature" AND abs:"abundances"',
    "stellar_pollution_civilization": 'all:"pollution" AND all:"civilization" AND all:"star"',
    "praseodymium_neodymium_overabundance": 'abs:"praseodymium" AND abs:"neodymium" AND abs:"overabund"',
    # The vetoes.
    "anomalous_neodymium_barium_ratio_dwarf": 'abs:"neodymium" AND abs:"barium" AND abs:"dwarf"',
    "nd_ba_ratio_anomaly": 'abs:"[Nd/Ba]"',
    "ce_ba_la_ba_ratios_dwarfs": 'abs:"[La/Ba]" OR abs:"[Ce/Ba]"',
    "young_stars_barium_enhancement": 'abs:"barium" AND abs:"young" AND abs:"enhance" AND abs:"cluster"',
    "barium_nlte_dwarfs": 'abs:"barium" AND abs:"NLTE" AND abs:"dwarfs"',
    "barium_lines_saturation": 'abs:"Ba II" AND abs:"saturat"',
    "barium_dwarfs_mass_transfer": 'abs:"barium dwarfs" OR abs:"dwarf barium stars"',
    "r_process_enhanced_dwarfs": 'abs:"r-process enhanced" AND abs:"dwarf"',
    # s/r decomposition and GALAH n-capture.
    "s_process_r_process_decomposition_galah": 'all:"s-process" AND all:"r-process" AND all:"GALAH"',
    "solar_s_process_fractions": 'abs:"s-process" AND abs:"solar" AND abs:"fractions"',
    "galah_neutron_capture_dwarfs": 'abs:"GALAH" AND abs:"neutron-capture"',
    "galah_ba_eu_ratio": 'abs:"GALAH" AND abs:"[Eu/Ba]"',
    "apogee_cerium_neodymium": 'abs:"APOGEE" AND abs:"cerium" AND abs:"neodymium"',
    "abundance_pattern_template_fit_stars": 'abs:"abundance pattern" AND abs:"template" AND abs:"stars" AND abs:"fit"',
    # Ap-star radionuclide claims (the only prior fission-adjacent spectroscopy).
    "przybylski_technetium_promethium": 'all:"Przybylski" AND (all:"technetium" OR all:"promethium")',
    "actinides_ap_stars": 'abs:"actinide" AND abs:"Ap star"',
}

# --------------------------------------------------------------------------
# 3. OpenAlex / Crossref: pre-arXiv literature and citation trees.
# --------------------------------------------------------------------------
# Whitmire & Wright 1980, Icarus 42, 149-156. bibcode 1980Icar...42..149W.
WW80_DOI = "10.1016/0019-1035(80)90253-5"

OPENALEX_DOI: dict[str, str] = {
    "whitmire_wright_1980": WW80_DOI,
    # Arlandini et al. 1999, ApJ 525, 886.
    "arlandini1999": "10.1086/307938",
    # Bisterzo et al. 2014, ApJ 787, 10.
    "bisterzo2014": "10.1088/0004-637X/787/1/10",
    # England & Rider 1994 fission yield evaluation (LA-UR-94-3106) has no DOI;
    # the ENDF/B-VII.1 paper of record:
    "endf_b_vii_1": "10.1016/j.nds.2011.11.002",
    # JEFF-3.1.1 fission yields (Kellett, Bersillon & Mills 2009) has no DOI;
    # JEFF-3.3 paper of record:
    "jeff_3_3": "10.1140/epja/s10050-020-00141-9",
    # Asplund, Amarsi & Grevesse 2021, A&A 653, A141.
    "asplund2021": "10.1051/0004-6361/202140445",
    # Lodders, Palme & Gail 2009.
    "lodders2009": "10.1007/978-3-540-88055-4_34",
}

# OpenAlex "cited by" trees: did anyone EXECUTE the fission-pattern search?
CITEDBY: dict[str, str] = {
    "whitmire_wright_1980": WW80_DOI,
    "huang2026_wd_technosignature": "10.48550/arXiv.2605.29811",
}


def main() -> None:
    print("=" * 70)
    print("FALLOUT literature evidence fetch")
    print("=" * 70)

    print("\n[1] Resolve by TITLE, verify, then fetch full text")
    for name, (title, must) in FULLTEXT.items():
        aid, info = resolve_by_title(title, must)
        VERIFY.append({"slug": name, "wanted_title": title, "must": list(must),
                       "arxiv_id": aid, "returned_title": info,
                       "verified": aid is not None})
        if aid is None:
            print(f"-- {name}: UNRESOLVED ({info}) -- nothing fetched, nothing citable")
            continue
        print(f"-- {name} -> {aid}  |  {info[:70]}")
        get(
            f"http://export.arxiv.org/api/query?id_list={urllib.parse.quote(aid)}&max_results=1",
            OUT / f"arxiv_id_{name}.atom",
        )
        get(f"https://ar5iv.labs.arxiv.org/html/{aid}", OUT / f"ar5iv_{name}.html")
        pdf = OUT / f"pdf_{name}.pdf"
        if get(f"https://arxiv.org/pdf/{aid}", pdf):
            try:
                subprocess.run(
                    ["pdftotext", "-q", str(pdf), str(OUT / f"txt_{name}.txt")],
                    check=False,
                    timeout=180,
                )
                pdf.unlink(missing_ok=True)  # keep the repo small; text is enough
            except Exception as exc:  # noqa: BLE001
                print(f"  pdftotext failed: {exc!r}")

    n_ver = sum(1 for v in VERIFY if v["verified"])
    (OUT / "verification.json").write_text(json.dumps(
        {"n_requested": len(VERIFY), "n_verified": n_ver,
         "unverified": [v for v in VERIFY if not v["verified"]],
         "all": VERIFY}, indent=2))
    print(f"\n[1] title-verified {n_ver}/{len(VERIFY)}; "
          "UNVERIFIED slugs have no files and must not be cited")

    print("\n[2] arXiv discovery searches (verbatim abstracts in the Atom files)")
    for name, q in ARXIV_Q.items():
        print(f"-- {name}")
        url = (
            "http://export.arxiv.org/api/query?search_query="
            + urllib.parse.quote(q)
            + "&start=0&max_results=60&sortBy=relevance"
        )
        get(url, OUT / f"arxiv_q_{name}.atom")

    print("\n[3] OpenAlex work records")
    for name, doi in OPENALEX_DOI.items():
        print(f"-- {name}")
        get(
            f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi, safe='')}"
            "?mailto=trimcrae@gmail.com",
            OUT / f"oa_{name}.json",
        )

    print("\n[4] Crossref record for the pre-arXiv Whitmire & Wright 1980")
    get(
        f"https://api.crossref.org/works/{urllib.parse.quote(WW80_DOI, safe='')}",
        OUT / "crossref_whitmire_wright_1980.json",
    )

    print("\n[5] OpenAlex citation trees (did anyone EXECUTE the pattern search?)")
    for name, doi in CITEDBY.items():
        print(f"-- cited-by {name}")
        rec = OUT / f"oa_resolve_{name}.json"
        if not get(
            f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi, safe='')}"
            "?mailto=trimcrae@gmail.com",
            rec,
        ):
            continue
        try:
            oid = json.loads(rec.read_text())["id"].rsplit("/", 1)[-1]
        except Exception as exc:  # noqa: BLE001
            print(f"  could not resolve OpenAlex id: {exc!r}")
            continue
        for page in (1, 2, 3):
            ok = get(
                f"https://api.openalex.org/works?filter=cites:{oid}"
                f"&per-page=200&page={page}&mailto=trimcrae@gmail.com",
                OUT / f"citedby_{name}_p{page}.json",
            )
            if not ok:
                break
            try:
                if len(json.loads((OUT / f"citedby_{name}_p{page}.json").read_text())["results"]) < 200:
                    break
            except Exception:  # noqa: BLE001
                break

    n_ok = sum(1 for s in STATUS if s["ok"])
    (OUT / "summary.json").write_text(
        json.dumps(
            {"n_urls": len(STATUS), "n_ok": n_ok, "n_failed": len(STATUS) - n_ok,
             "n_title_verified": sum(1 for v in VERIFY if v["verified"]),
             "n_title_requested": len(VERIFY), "verification": VERIFY,
             "status": STATUS},
            indent=2,
        )
    )
    print(f"\n{n_ok}/{len(STATUS)} fetches ok -> {OUT}")


if __name__ == "__main__":
    main()
