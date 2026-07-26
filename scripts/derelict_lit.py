#!/usr/bin/env python3
"""DERELICT novelty check: fetch the primary sources on the GitHub runner.

WHY THIS EXISTS.  The repository's earlier NECROSIGNATURES sweep believed it had
fetched the dark-comet paper, but ``scripts/necrolit_fetch.py`` carried the wrong
arXiv id -- ``2306.16966`` is *"Self-interacting dark matter implied by
nano-Hertz gravitational waves"* (hep-ph), not Seligman et al.  The fetch
succeeded, so nothing flagged it, and ``results/necrolit/{ar5iv,txt}_dark_comets.*``
are a particle-physics paper.  The DERELICT novelty verdict rests on what the
dark-comet papers actually selected on, so it must be re-established from the
real sources.

The four questions this script must answer from primary text
------------------------------------------------------------
Q1  Did Seligman et al. select their dark comets on **non-radial** (A2/A3)
    acceleration?  If yes, the complement set (A1-only) is genuinely unexamined
    and this channel is novel.  If they selected on total or radial
    acceleration, the novelty claim collapses and must be rewritten.
Q2  Do they anywhere compute an implied area-to-mass ratio or radiation-pressure
    beta for their objects?
Q3  Did Bialy & Loeb 2018 generalise the A1 -> beta -> AMR -> lightsail inference
    beyond 1I/'Oumuamua?
Q4  **The live risk.**  Loeb & Cloete 2025 (arXiv:2503.03552) argue the dark
    comet 2005 VL1 is the Venera 2 spacecraft.  Did they compute an AMR/beta,
    and did they do it as a *population* test or for one object?  If the former,
    this channel's novelty narrows to the systematic complement-set search and
    the docs must say so.

Plus the coverage gap the offline sweep identified: the space-situational-
awareness literature on **HAMR** (high area-to-mass ratio) debris, which lives
outside the SETI corpus and returned zero hits locally.

Outputs under ``results/derelictlit/``: ``arxiv_id_*.atom`` (metadata),
``ar5iv_*.html`` (full text with equations/tables), ``txt_*.txt`` (extracted
text), ``arxiv_q_*.atom`` (keyword sweeps), ``summary.json`` (fetch status +
the per-target question each source must answer).

**Every fetch is verified**: the returned arXiv id and title are checked against
what was requested, and a mismatch is recorded as ``id_mismatch`` in the summary
rather than silently written to disk.  That is the exact failure this script
exists to correct.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/derelictlit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-derelict-lit/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []
PAUSE = 3.0


def get(url: str, dest: pathlib.Path, tries: int = 3) -> bytes | None:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "dest": dest.name, "ok": True,
                           "bytes": len(data)})
            print(f"  ok  {len(data):>9,}B  {dest.name}")
            time.sleep(PAUSE)
            return data
        except Exception as exc:  # noqa: BLE001
            print(f"  try {i + 1}/{tries} failed: {exc!r}")
            time.sleep(PAUSE * (i + 1))
    STATUS.append({"url": url, "dest": dest.name, "ok": False})
    return None


# --------------------------------------------------------------------------
# Papers whose FULL TEXT decides the verdict.  Each carries the question it
# must answer and the title fragment used to VERIFY the right paper arrived.
# --------------------------------------------------------------------------
FULLTEXT: dict[str, dict] = {
    "seligman2023_dark_comets": {
        "arxiv": "2212.08115",
        "expect_title": "dark comet",
        "question": "Q1/Q2: what did they select on -- A1 (radial) or A2/A3 "
                    "(non-radial)?  Do they compute AMR or beta anywhere?",
    },
    "seligman2024_two_populations": {
        "arxiv": "2412.02384",
        "expect_title": "dark comet",
        "question": "Q1/Q2 for the 14-object PNAS sample (PNAS 121, "
                    "e2406424121). Same questions as 2023.",
    },
    "bialy_loeb_2018": {
        "arxiv": "1810.11490",
        "expect_title": "radiation pressure",
        "question": "Q3: is the A1 -> beta -> AMR -> lightsail inference applied "
                    "ONLY to 1I/'Oumuamua, or generalised to a catalogue?",
    },
    "loeb_cloete_2025_venera": {
        "arxiv": "2503.03552",
        "expect_title": "Venera",
        "question": "Q4 (the live novelty risk): did they compute an AMR/beta "
                    "for 2005 VL1, and is it one object or a population test?",
    },
    "micheli2018_oumuamua_nongrav": {
        "arxiv": "1811.05519",
        "expect_title": "",
        "question": "the calibration anchor: the published A1 / radial "
                    "acceleration at 1 au used to validate our conversions",
    },
    "seligman_2021_darkcomet_precursor": {
        "arxiv": "2104.10184",
        "expect_title": "",
        "question": "context: earlier non-grav-acceleration work by the same group",
    },
}

# --------------------------------------------------------------------------
# Keyword sweeps.  The first block probes the SETI framing; the second probes
# the space-situational-awareness (HAMR) corpus, which the offline sweep never
# covered and which is the most likely place for undiscovered prior art.
# --------------------------------------------------------------------------
QUERIES: dict[str, str] = {
    # --- is the complement set occupied? ---
    "a1_only_nongrav_asteroids":
        'all:"non-gravitational acceleration" AND all:"radial" AND all:asteroid',
    "radiation_pressure_asteroid_catalog":
        'all:"radiation pressure" AND all:"area-to-mass" AND all:asteroid',
    "srp_detection_neo_astrometry":
        'abs:"solar radiation pressure" AND abs:"near-Earth" AND abs:astrometry',
    "amr_small_body_survey":
        'all:"area-to-mass ratio" AND all:"small body"',
    "dark_comet_population":
        'abs:"dark comet"',
    # --- the technosignature framing ---
    "interstellar_object_technosignature_survey":
        'abs:"interstellar object" AND abs:technosignature',
    "lightsail_technosignature_detection":
        'abs:"light sail" AND abs:technosignature',
    "artificial_object_solar_system_search":
        'abs:"artificial" AND abs:"solar system" AND abs:"search" AND abs:"probe"',
    "seta_artifact_search":
        'abs:"SETA" OR abs:"search for extraterrestrial artifacts"',
    # --- the coverage gap: HAMR space-debris literature ---
    "hamr_debris":
        'all:"high area-to-mass ratio" AND all:debris',
    "hamr_geo_objects":
        'all:"area-to-mass" AND all:"geosynchronous"',
    "rocket_body_asteroid_designation":
        'all:"rocket body" AND all:"minor planet"',
    "artificial_satellite_misidentified_asteroid":
        'all:"artificial" AND all:"asteroid" AND all:"designation"',
}


def arxiv_id_query(name: str, arxiv_id: str, expect_title: str) -> bool:
    """Fetch arXiv metadata and VERIFY the returned id/title match the request."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
    data = get(url, OUT / f"arxiv_id_{name}.atom")
    if not data:
        return False
    text = data.decode("utf-8", errors="replace")
    got_id = re.search(r"<id>http://arxiv\.org/abs/([\d.v]+)</id>", text)
    got_title = re.search(r"<title>(.*?)</title>", text[text.find("<entry>"):], re.S)
    got_id_s = got_id.group(1) if got_id else ""
    got_title_s = " ".join(got_title.group(1).split()) if got_title else ""
    ok_id = got_id_s.startswith(arxiv_id)
    ok_title = (expect_title.lower() in got_title_s.lower()) if expect_title else True
    rec = {"name": name, "requested": arxiv_id, "returned_id": got_id_s,
           "returned_title": got_title_s, "id_ok": ok_id, "title_ok": ok_title}
    if not (ok_id and ok_title):
        rec["id_mismatch"] = True
        print(f"  !! MISMATCH for {name}: asked {arxiv_id}, got {got_id_s} "
              f"'{got_title_s[:70]}'")
    STATUS.append(rec)
    return ok_id and ok_title


def fetch_fulltext(name: str, arxiv_id: str) -> None:
    get(f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}", OUT / f"ar5iv_{name}.html")
    pdf = OUT / f"pdf_{name}.pdf"
    if get(f"https://arxiv.org/pdf/{arxiv_id}", pdf):
        try:
            subprocess.run(["pdftotext", "-layout", str(pdf),
                            str(OUT / f"txt_{name}.txt")], check=True, timeout=180)
            pdf.unlink(missing_ok=True)   # keep the text, not the 10 MB PDF
        except Exception as exc:  # noqa: BLE001
            print(f"  pdftotext failed for {name}: {exc!r}")


def main() -> None:
    print("== full texts (identity-verified) ==")
    for name, spec in FULLTEXT.items():
        print(f"-- {name} ({spec['arxiv']}): {spec['question']}")
        verified = arxiv_id_query(name, spec["arxiv"], spec.get("expect_title", ""))
        if not verified:
            print("   fetching full text anyway, but flagged as UNVERIFIED")
        fetch_fulltext(name, spec["arxiv"])

    print("== keyword sweeps ==")
    for name, q in QUERIES.items():
        url = ("http://export.arxiv.org/api/query?search_query="
               + urllib.parse.quote(q)
               + "&max_results=60&sortBy=relevance")
        get(url, OUT / f"arxiv_q_{name}.atom")

    # NASA ADS and Semantic Scholar need no key for these public endpoints.
    for name, q in (("dark_comet_ads", "dark comets non-gravitational acceleration"),
                    ("amr_technosignature", "area-to-mass ratio technosignature")):
        url = ("https://api.semanticscholar.org/graph/v1/paper/search?query="
               + urllib.parse.quote(q)
               + "&limit=50&fields=title,abstract,year,externalIds,citationCount")
        get(url, OUT / f"s2_{name}.json")

    mismatches = [s for s in STATUS if s.get("id_mismatch")]
    summary = {
        "n_urls": len([s for s in STATUS if "url" in s]),
        "n_ok": len([s for s in STATUS if s.get("ok")]),
        "n_failed": len([s for s in STATUS if s.get("ok") is False]),
        "n_id_mismatch": len(mismatches),
        "id_mismatches": mismatches,
        "targets": {k: v for k, v in FULLTEXT.items()},
        "queries": QUERIES,
        "status": STATUS,
        "note": "results/necrolit/*dark_comets* is arXiv:2306.16966, a hep-ph "
                "paper -- the wrong source. This directory supersedes it for the "
                "DERELICT novelty verdict.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{summary['n_ok']} ok / {summary['n_failed']} failed / "
          f"{summary['n_id_mismatch']} id mismatches -> {OUT}")


if __name__ == "__main__":
    main()
