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
    # NOTE: no arXiv id is given for the PNAS 2024 paper ON PURPOSE.  The first
    # run guessed "2412.02384" and the verification step caught that it is
    # actually "Theory building for empirical software engineering in
    # qualitative research" -- the second wrong-id incident in this repository.
    # It is now resolved BY TITLE at fetch time (see TITLE_RESOLVED) so no id is
    # ever guessed again.
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
    # NOTE: `micheli2018_oumuamua_nongrav` and `seligman_2021_darkcomet_precursor`
    # USED to live here with guessed ids 1811.05519 and 2104.10184 and an EMPTY
    # `expect_title`.  Both fetched the wrong paper -- 1811.05519 returns
    # "Acoplanarity of Lepton Pair to Probe the Electromagnetic Property of Quark
    # Matter" (nucl-th) and 2104.10184 returns "Finite-size evaporating droplets
    # in weakly compressible homogeneous shear turbulence" (physics.flu-dyn) --
    # and BOTH were recorded `id_ok: true, title_ok: true`, because an empty
    # `expect_title` made the title check vacuously true.  That is the third and
    # fourth wrong-paper incidents in this repository, and the first two that the
    # verification step was supposed to catch and did not.
    #
    # Two fixes: an empty `expect_title` is now an explicit UNVERIFIED state
    # rather than a pass (see `arxiv_id_query`), and Micheli et al. moves to
    # TITLE_RESOLVED below so no id is guessed for it.  The Seligman "precursor"
    # entry is REMOVED outright: there is no verified id for it, it was never
    # decisive for the novelty argument, and guessing a fifth time is exactly the
    # behaviour that produced this comment.
}

#: Papers resolved by TITLE rather than by a guessed id.  Guessing an arXiv id
#: has now produced two wrong-paper incidents in this repository (2306.16966 in
#: necrolit, 2412.02384 here), and both fetches SUCCEEDED, so nothing flagged
#: them.  Searching by title and reading the id back off the match cannot fail
#: that way.
TITLE_RESOLVED: dict[str, dict] = {
    "seligman2024_two_populations": {
        "title": "Two distinct populations of dark comets delineated by orbits and sizes",
        "expect_title": "dark comet",
        "question": "Q1/Q2 for the 14-object PNAS sample (PNAS 121, "
                    "e2406424121). Same questions as 2023.",
    },
    "micheli2018_oumuamua_nongrav": {
        "title": "Non-gravitational acceleration in the trajectory of 1I/2017 U1 (Oumuamua)",
        # The verification fragment is the load-bearing part: if the search
        # returns anything whose title does not contain "Oumuamua", NOTHING is
        # written and an id_mismatch is recorded.  A title search that fails is
        # reported; it cannot fetch the wrong paper the way a guessed id can.
        "expect_title": "Oumuamua",
        "question": "the calibration anchor: the published A1 / radial "
                    "acceleration at 1 au used to validate our conversions "
                    "(docs/derelict.md section 1.3 asserts 4.92e-6 m/s^2)",
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
    # An EMPTY expect_title used to make this vacuously True, and that is exactly
    # how two wrong papers were fetched, recorded `title_ok: true`, and believed
    # (arXiv 1811.05519 -> a quark-matter paper; 2104.10184 -> a fluid-dynamics
    # paper).  "The title was never checked" and "the title checked out" are
    # different statements and must not share a value, so an absent fragment now
    # yields `title_ok: null` and an explicit `title_unverified` flag.
    ok_title = (expect_title.lower() in got_title_s.lower()) if expect_title else None
    rec = {"name": name, "requested": arxiv_id, "returned_id": got_id_s,
           "returned_title": got_title_s, "id_ok": ok_id, "title_ok": ok_title}
    if ok_title is None:
        rec["title_unverified"] = True
        print(f"  ?? UNVERIFIED TITLE for {name}: no expect_title fragment, so "
              f"the returned paper '{got_title_s[:70]}' was NOT checked")
    if not ok_id or ok_title is False:
        rec["id_mismatch"] = True
        print(f"  !! MISMATCH for {name}: asked {arxiv_id}, got {got_id_s} "
              f"'{got_title_s[:70]}'")
    STATUS.append(rec)
    # An unverified title is not a verification failure, but it IS a reason not
    # to treat the fetched text as evidence; the caller still stores it, and the
    # summary counts it separately so it can never pass as confirmed.
    return ok_id and ok_title is not False


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


def resolve_by_title(name: str, title: str, expect_title: str) -> str | None:
    """Find a paper's arXiv id by searching its exact title.

    Returns the id, or ``None`` if no match whose title actually contains
    ``expect_title`` is found.  This is the guess-free path: the id comes from
    the search result rather than from memory.
    """
    q = 'ti:"' + title + '"'
    url = ("http://export.arxiv.org/api/query?search_query="
           + urllib.parse.quote(q) + "&max_results=5")
    data = get(url, OUT / f"arxiv_title_{name}.atom")
    if not data:
        return None
    text = data.decode("utf-8", errors="replace")
    for entry in re.findall(r"<entry>(.*?)</entry>", text, re.S):
        idm = re.search(r"<id>http://arxiv\.org/abs/([\d.]+)v?\d*</id>", entry)
        tm = re.search(r"<title>(.*?)</title>", entry, re.S)
        got_title = " ".join(tm.group(1).split()) if tm else ""
        if idm and expect_title.lower() in got_title.lower():
            STATUS.append({"name": name, "resolved_by": "title",
                           "resolved_id": idm.group(1), "resolved_title": got_title})
            print(f"  resolved {name} -> {idm.group(1)} '{got_title[:70]}'")
            return idm.group(1)
    STATUS.append({"name": name, "resolved_by": "title", "resolved_id": None,
                   "id_mismatch": True,
                   "note": f"no arXiv hit whose title contains {expect_title!r}"})
    print(f"  !! could not resolve {name} by title")
    return None


def main() -> None:
    print("== full texts (identity-verified) ==")
    for name, spec in FULLTEXT.items():
        print(f"-- {name} ({spec['arxiv']}): {spec['question']}")
        verified = arxiv_id_query(name, spec["arxiv"], spec.get("expect_title", ""))
        if not verified:
            print("   fetching full text anyway, but flagged as UNVERIFIED")
        fetch_fulltext(name, spec["arxiv"])

    print("== full texts (resolved by title, no id guessed) ==")
    for name, spec in TITLE_RESOLVED.items():
        print(f"-- {name}: {spec['question']}")
        arxiv_id = resolve_by_title(name, spec["title"], spec.get("expect_title", ""))
        if arxiv_id:
            fetch_fulltext(name, arxiv_id)
        else:
            print("   UNRESOLVED -- the keyword sweeps below are the fallback")

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
    unverified = [s for s in STATUS if s.get("title_unverified")]
    summary = {
        "n_urls": len([s for s in STATUS if "url" in s]),
        "n_ok": len([s for s in STATUS if s.get("ok")]),
        "n_failed": len([s for s in STATUS if s.get("ok") is False]),
        "n_id_mismatch": len(mismatches),
        "id_mismatches": mismatches,
        # Counted SEPARATELY from mismatches: a paper whose title was never
        # checked is not verified, and must not be able to hide inside a
        # zero-mismatch summary the way 1811.05519 and 2104.10184 did.
        "n_title_unverified": len(unverified),
        "title_unverified": unverified,
        "targets": {k: v for k, v in FULLTEXT.items()},
        "title_resolved_targets": TITLE_RESOLVED,
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
