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
import re
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
# RESOLVED BY TITLE, NOT BY HARDCODED ID, AND VERIFIED AFTER FETCH.
#
# The first version of this file used hardcoded arXiv identifiers. Thirteen of
# twenty-four resolved to unrelated papers -- a neutron-star precession paper
# standing in for Richer's AmFm diffusion work, an LHC dark-matter paper for
# Vick, a condensed-matter paper for the lambda Boo review -- and every one of
# them fetched *successfully*. A successful fetch is no evidence at all that
# the paper is the right one, and citing from those files would have put
# fabricated attributions into the channel documentation.
#
# So each entry is now (exact-ish title, tokens that MUST appear in the title
# of whatever comes back). The id is discovered by title search and the result
# is checked before anything is written; failures are recorded in
# verification.json rather than silently accepted.
FULLTEXT: dict[str, tuple[str, tuple[str, ...]]] = {
    # *** THE REAL COMPETITOR *** Huang, Tao & Zhang 2026: the one executed
    # meteorite-calibrated test for processed material, in polluted WDs.
    "huang2026_refined_material": (
        "A Calibrated Bayesian Search for Potential Chemical Technosignatures "
        "in Polluted White Dwarf",
        ("technosignature", "white dwarf"),
    ),
    # The 2026 flagship review; its "Stellar Pollution" section is one paragraph.
    "technosig_review_2026": (
        "The Search for Technosignatures: a Review of Possibilities",
        ("technosignature", "review"),
    ),
    # --- Chemical-tagging dimensionality: sets sigma_X and the families claim.
    "ting2012_pca_dimensionality": (
        "Principal component analysis on chemical abundances spaces",
        ("principal component", "abundance"),
    ),
    "pricejones2018_dimensionality": (
        "Blind Chemical Tagging with Density Estimation",
        ("chemical tagging",),
    ),
    "ting2021_how_many_elements": (
        "How Many Elements Matter?",
        ("elements", "matter"),
    ),
    "weinberg2019_two_process": (
        "Chemical Cartography with APOGEE: Multi-element Abundance Ratios",
        ("apogee", "abundance"),
    ),
    "weinberg2021_two_process_residuals": (
        "Chemical Cartography with APOGEE: Two-process Parameters and "
        "Residual Abundances",
        ("apogee", "residual"),
    ),
    "ness2018_doppelgangers": (
        "Galactic Doppelgangers: The Chemical Similarity Among Field Stars "
        "and Among Stars with a Common Birth Origin",
        ("doppelganger",),
    ),
    "bedell2018_chemical_homogeneity": (
        "The Chemical Homogeneity of Sun-like Stars in the Solar Neighborhood",
        ("chemical", "sun-like"),
    ),
    # --- Anomaly detection already run on these surveys.
    "reis_apogee_outliers": (
        "Detecting Outliers and Learning Complex Structures with Large "
        "Spectroscopic Surveys",
        ("outlier", "spectroscopic"),
    ),
    "baron_poznanski_weirdest": (
        "The weirdest SDSS galaxies: results from an outlier detection "
        "algorithm",
        ("weirdest", "outlier"),
    ),
    # --- Diffusion / peculiarity physics: WHY a cool dwarf cannot do this.
    "richer_amfm_diffusion": (
        "Abundance anomalies in main sequence A stars",
        ("abundance", "main sequence"),
    ),
    "vick_amfm_massloss": (
        "Abundance anomalies in AmFm stars: mass loss",
        ("amfm", "mass loss"),
    ),
    "michaud_diffusion_popii": (
        "Models for metal poor stars with gravitational settling and "
        "radiative accelerations",
        ("metal poor", "radiative"),
    ),
    "deal_diffusion_solar_type": (
        "Chemical mixing in low mass stars",
        ("mixing", "low mass"),
    ),
    "xiang_peculiar_boundary": (
        "Chemically peculiar A and F stars with enhanced s-process and "
        "iron-peak elements",
        ("peculiar", "s-process"),
    ),
    "matrozis_thin_envelope": (
        "Constraining the thermohaline mixing efficiency and the "
        "accretion history",
        ("accretion", "carbon-enhanced"),
    ),
    "lambda_boo_review": (
        "The lambda Bootis stars",
        ("bootis",),
    ),
    "karinkuzhi_sr_only": (
        "Sr and Ba enrichment in barium stars",
        ("barium",),
    ),
    # --- Planet engulfment in co-natal pairs: the stage-4 mass budget.
    "spina2021_engulfment": (
        "Chemical evidence for planetary ingestion in a quarter of Sun-like "
        "stars",
        ("planetary", "ingestion"),
    ),
    "liu2024_nature_ingestion": (
        "At least one in a dozen stars exhibits evidence of planetary "
        "ingestion",
        ("planetary", "ingestion"),
    ),
    "behmard_engulfment_signature": (
        "Planet Engulfment Detections are Rare",
        ("engulfment",),
    ),
    "melendez_solar_twins": (
        "The peculiar solar composition and its possible relation to planet "
        "formation",
        ("solar", "composition"),
    ),
    "griffith_na_rich": (
        "Chemical Cartography with APOGEE: Mapping Disk Populations",
        ("apogee",),
    ),
    "sit_residual_abundances": (
        "Chemical Cartography with APOGEE: Mapping the Disk",
        ("apogee",),
    ),
    "manea_doppelganger_followup": (
        "Chemical doppelgangers",
        ("doppelganger",),
    ),
    # --- Surveys.
    "galah_dr4": ("The GALAH Survey: Data Release 4", ("galah",)),
    "apogee_dr17": (
        "The Seventeenth Data Release of the Sloan Digital Sky Surveys",
        ("data release", "sloan"),
    ),
}

VERIFY: list[dict] = []


def resolve_by_title(title: str, must: tuple[str, ...]) -> tuple[str | None, str]:
    """Find an arXiv id by title search and VERIFY the title that comes back.

    Returns ``(arxiv_id, returned_title)``, or ``(None, reason)``. A hit is
    accepted only when every token in ``must`` appears in the returned title,
    case-insensitively -- which is what the hardcoded-id version never did.
    """
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
