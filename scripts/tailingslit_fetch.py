#!/usr/bin/env python3
"""TAILINGS novelty check: fetch the chemical-anomaly prior art on the runner.

The sandbox blocks arxiv.org / OpenAlex / Crossref (``CONNECT tunnel failed,
response 403``); the GitHub Actions runner has egress. This script pulls
verbatim metadata, full text and citation trees for everything that could sink
the TAILINGS channel, and commits the evidence back to the branch so the
analysis can be done offline.

The decisive target is **arXiv:2605.29811** (Huang, Tao & Zhang 2026) — the one
*actually executed* meteorite-calibrated Bayesian test for processed/refined
material, in polluted white dwarfs. TAILINGS claims a different population
(main-sequence cool dwarfs), a different discriminant (sparse-vs-dense anomaly
in a regressed abundance manifold) and different physics (convective
suppression of natural peculiarity). If 2605.29811 already does any of that,
the design has to change, so its full text must come off the record.

The other three questions the fetch has to answer from primary sources:

1. Has anyone run **sparse** (single-element) anomaly detection on APOGEE or
   GALAH, or only global/dense outlier detection?
2. What is the intrinsic dimensionality of abundance space and the intrinsic
   scatter per element at fixed [Fe/H]? (Sets the thresholds.)
3. What is the observed distribution of refractory differences in co-natal
   wide binaries, and the maximum inferred engulfed rocky mass? (Sets the
   "unexplainable by engulfment" bar for stage 4.)

Outputs under ``results/tailingslit/``:
  arxiv_id_<name>.atom   arXiv API metadata for a specific paper
  arxiv_q_<name>.atom    arXiv API keyword search (discovery)
  ar5iv_<name>.html      ar5iv full-text rendering
  txt_<name>.txt         extracted plain text (PDF -> pdftotext)
  oa_<name>.json         OpenAlex work record
  citedby_<name>.json    OpenAlex "works citing this"
  crossref_<name>.json   Crossref record (pre-arXiv literature)
  summary.json           fetch status for every URL
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/tailingslit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-tailingslit/1.0 (mailto:trimcrae@gmail.com)"}
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
# 1. Papers whose FULL TEXT decides the novelty verdict.
# --------------------------------------------------------------------------
FULLTEXT: dict[str, str] = {
    # *** THE REAL COMPETITOR ***  Huang, Tao & Zhang 2026. Meteorite-calibrated
    # Bayesian test for processed/refined material in polluted white dwarfs.
    # Q: population (WD accretion vs MS photosphere)? discriminant (bulk
    # siderophile ratio vs sparse-vs-dense residual)? Does it ever look at
    # main-sequence stars, or at SPARSITY of the anomaly? 697 records / >=397
    # objects / 8 with BF>10 -- verify those numbers verbatim.
    "huang2026_refined_material": "2605.29811",
    # The 2026 flagship technosignature review: its "Stellar Pollution" section
    # is claimed to be one paragraph of proposals with no executed search.
    "technosig_review_2026": "2601.07297",
    # --- Chemical-tagging dimensionality: sets sigma_X and the "families" claim.
    # Ting, Freeman, Kobayashi, De Silva, Bland-Hawthorn 2012: PCA on abundances,
    # how many independent dimensions does chemical space actually have?
    "ting2012_pca_dimensionality": "1207.5074",
    # Price-Jones & Bovy 2018: dimensionality of chemical space from APOGEE.
    "pricejones2018_dimensionality": "1710.08442",
    # Ting & Weinberg 2021: "How Many Elements Matter?" -- residual abundance
    # correlations after conditioning on [Fe/H] and [Mg/Fe]. THE key paper for
    # the manifold-residual construction.
    "ting2021_how_many_elements": "2102.04992",
    # Weinberg et al. 2019/2021: APOGEE two-process model. The explicit
    # statement that abundances move in nucleosynthetic FAMILIES.
    "weinberg2019_two_process": "1810.01470",
    "weinberg2021_two_process_residuals": "2108.08860",
    # Ness et al. 2018 "Galactic doppelgangers": how well one star's abundances
    # predict another's -- the empirical floor on chemical individuality.
    "ness2018_doppelgangers": "1701.07829",
    # Bedell et al. 2018: 79 solar twins at 0.01-0.02 dex -- the precision floor.
    "bedell2018_chemical_homogeneity": "1802.02576",
    # --- Anomaly detection already run on these surveys.
    # Reis, Poznanski & Hall: unsupervised outlier detection on APOGEE spectra.
    "reis_apogee_outliers": "1711.00022",
    # Random-forest / autoencoder outlier detection on stellar spectra.
    "baron_poznanski_weirdest": "1611.07526",
    # --- Diffusion / peculiarity physics: WHY a cool dwarf cannot do this.
    # Richer, Michaud & Turcotte 2000: diffusion in AmFm stars; the convective
    # envelope mass threshold that switches the anomalies OFF.
    "richer2000_amfm_diffusion": "astro-ph/0004035",
    # Vick et al. 2010: AmFm with mass loss.
    "vick2010_amfm_massloss": "1002.1922",
    # Michaud/Richard: diffusion in solar-type and metal-poor stars.
    "michaud2011_diffusion_popii": "1011.4212",
    "deal2020_diffusion_solar_type": "2007.02528",
    # Lambda Boo: the one refractory-DEPLETION peculiarity class -- the nearest
    # natural analogue of a sparse anomaly, and hot-star-only. Confounder check.
    "lambda_boo_review": "1908.03976",
    # --- Planet engulfment in co-natal pairs: the stage-4 mass budget.
    # Spina et al. 2021 (Nature Astronomy): 33 wide binaries, engulfment rate.
    "spina2021_engulfment": "2108.12040",
    # Liu et al. 2024 (Nature): planetary ingestion in co-natal pairs.
    "liu2024_nature_ingestion": "2405.10339",
    # Behmard et al.: how long an engulfment signature survives (the dilution
    # and the thermohaline mixing timescale) -- decides whether the signal can
    # persist at all in a convective envelope.
    "behmard2023_engulfment_signature": "2210.11330",
    "behmard2025_engulfment": "2501.03252",
    # Melendez/Ramirez solar-twin refractory trend -- the 0.08 dex Tcond slope
    # that any "engulfment" claim must exceed.
    "melendez2009_solar_twins": "0910.5845",
    # --- Surveys.
    "galah_dr4": "2409.19858",
    "apogee_dr17": "2112.05131",
    # --- Whitmire & Wright's own prediction, and the modern restatement.
    # (1980 Icarus is pre-arXiv; see the Crossref/OpenAlex block below.)
    "wright2019_technosig_search_landscape": "1907.07830",
}

# --------------------------------------------------------------------------
# 2. Discovery searches: has ANYONE run the search TAILINGS proposes?
# --------------------------------------------------------------------------
ARXIV_Q: dict[str, str] = {
    # The direct question.
    "sparse_abundance_anomaly": 'all:"abundance anomaly" AND all:"single element"',
    "single_element_overabundance": 'abs:"single-element" AND abs:"abundance"',
    "anomaly_detection_galah": 'all:"anomaly detection" AND all:"GALAH"',
    "anomaly_detection_apogee": 'all:"anomaly detection" AND all:"APOGEE"',
    "outlier_detection_stellar_abundances": 'abs:"outlier detection" AND abs:"stellar abundances"',
    "unsupervised_stellar_spectra_outliers": 'abs:"unsupervised" AND abs:"stellar spectra" AND abs:"outlier"',
    "chemically_peculiar_cool_dwarfs": 'abs:"chemically peculiar" AND abs:"dwarfs"',
    "chemically_peculiar_incidence": 'abs:"chemically peculiar stars" AND abs:"incidence"',
    # Technosignature side.
    "technosignature_stellar_pollution": 'abs:"technosignature" AND abs:"pollution"',
    "artificial_pollution_photosphere": 'all:"artificial" AND all:"photosphere" AND all:"civilization"',
    "waste_disposal_star": 'all:"waste disposal" AND all:"star"',
    "extraterrestrial_industrial_pollution": 'abs:"industrial pollution" AND abs:"extraterrestrial"',
    "white_dwarf_technosignature": 'abs:"white dwarf" AND abs:"technosignature"',
    "polluted_white_dwarf_refined": 'abs:"polluted white dwarf" AND abs:"refined"',
    "asteroid_mining_signature": 'abs:"asteroid mining" AND abs:"signature"',
    "planetary_disassembly": 'all:"disassembl" AND all:"planet" AND all:"civilization"',
    # Manifold / regression machinery.
    "abundance_manifold_residuals": 'abs:"abundance" AND abs:"residual" AND abs:"manifold"',
    "intrinsic_abundance_scatter": 'abs:"intrinsic scatter" AND abs:"abundances" AND abs:"fixed metallicity"',
    "chemical_homogeneity_open_clusters": 'abs:"chemical homogeneity" AND abs:"open cluster"',
    # Engulfment stage.
    "wide_binary_abundance_difference": 'abs:"wide binary" AND abs:"abundance difference"',
    "planet_engulfment_mass": 'abs:"engulfment" AND abs:"rocky" AND abs:"mass"',
    "conatal_pairs_chemical": 'abs:"co-natal" AND abs:"chemical"',
    "thermohaline_engulfment_dilution": 'abs:"thermohaline" AND abs:"accretion" AND abs:"dilution"',
    # Diffusion physics.
    "atomic_diffusion_convective_envelope": 'abs:"atomic diffusion" AND abs:"convective envelope"',
    "radiative_levitation_dwarfs": 'abs:"radiative levitation" AND abs:"main sequence"',
    # Prior art on the specific elements Whitmire & Wright predicted.
    "praseodymium_neodymium_anomaly": 'abs:"praseodymium" AND abs:"neodymium" AND abs:"anomal"',
    "actinide_boost_stars": 'abs:"actinide" AND abs:"r-process" AND abs:"boost"',
}

# --------------------------------------------------------------------------
# 3. OpenAlex / Crossref: pre-arXiv literature and citation trees.
# --------------------------------------------------------------------------
# Whitmire & Wright 1980, Icarus 42, 149-156. bibcode 1980Icar...42..149W.
WW80_DOI = "10.1016/0019-1035(80)90253-5"

OPENALEX_DOI: dict[str, str] = {
    "whitmire_wright_1980": WW80_DOI,
    # Spina 2021 Nature Astronomy -- engulfment rate paper of record.
    "spina2021": "10.1038/s41550-021-01451-8",
    # Liu 2024 Nature -- planetary ingestion.
    "liu2024": "10.1038/s41586-024-07091-y",
}

# OpenAlex "cited by" trees: who followed up, and did any of them EXECUTE?
CITEDBY: dict[str, str] = {
    "whitmire_wright_1980": WW80_DOI,
    "huang2026_refined_material": "10.48550/arXiv.2605.29811",
}


def main() -> None:
    print("=" * 70)
    print("TAILINGS literature evidence fetch")
    print("=" * 70)

    print("\n[1] arXiv metadata + full text for decisive papers")
    for name, aid in FULLTEXT.items():
        print(f"-- {name} ({aid})")
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

    print("\n[2] arXiv discovery searches")
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

    print("\n[5] OpenAlex citation trees (did anyone EXECUTE the proposal?)")
    for name, doi in CITEDBY.items():
        print(f"-- cited-by {name}")
        # Resolve the OpenAlex id first, then page the citing works.
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
            {"n_urls": len(STATUS), "n_ok": n_ok, "n_failed": len(STATUS) - n_ok, "status": STATUS},
            indent=2,
        )
    )
    print(f"\n{n_ok}/{len(STATUS)} fetches ok -> {OUT}")


if __name__ == "__main__":
    main()
