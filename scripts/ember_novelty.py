#!/usr/bin/env python3
"""Runner-side novelty verification for EMBER (waste heat that switched off).

The sandbox has no arXiv/OpenAlex egress and the session WebSearch budget is
spent, so the record has to be established on the Actions runner. Two prior
sweeps (``results/disaplit*``, ``results/seamlit``, ``results/hephlit``,
``results/litcheck*``, ``results/necrolit``) already answered most of the
question; this script closes the specific gaps those sweeps identified, and
fetches the full text of the papers that must be *cited and distinguished*
rather than merely absent.

The claim being tested, stated precisely enough to be falsifiable:

    No published work has searched, blind and at catalogue scale, for the
    disappearance of a mid-infrared excess between the IRAS/AKARI and WISE
    epochs, nor framed such a disappearance as a technosignature.

The three findings that qualify it, and which this script must re-verify:

1. **Kim et al. 2015 (1501.05721)** ran the identical IRAS+AKARI+WISE
   cross-catalogue comparison -- but only in the *brightening* direction,
   finding 4 sources all-sky. Methodological antecedent, opposite sign.
2. **Melis et al. 2023 (2306.11945)** ran it downward, but targeted at R CrB
   stars, where factor-10 inter-epoch swings are the known behaviour.
3. **Sedgwick & Serjeant 2022 (2207.09985)** built the IRAS-AKARI all-sky
   cross-match over a 23.4-year baseline -- for *proper motion*, not flux.
   Repointing that machinery at flux change is the novel move, and the paper is
   also the best available source of systematics lore for this cross-match.

Outputs under ``results/emberlit/``.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/emberlit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-ember/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []
PAUSE = 3.0


def get(url: str, dest: pathlib.Path, tries: int = 3, pause: float = PAUSE) -> bool:
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
# 1. Full texts that decide the verdict or must be cited and distinguished.
# --------------------------------------------------------------------------
FULLTEXT: dict[str, str] = {
    # THE METHODOLOGICAL ANTECEDENT. Same three catalogues, same cross-epoch
    # comparison, opposite direction. Q: does it search for fading anywhere?
    # Q: what is its all-sky yield, and what systematics does it report?
    "kim2015_iras_akari_wise_brightening": "1501.05721",
    # THE MACHINERY. IRAS x AKARI all-sky cross-match, 23.4-yr baseline, built
    # to find outer-Solar-System planets by proper motion. Q: what match radius,
    # what false-match rate, what flux-consistency filtering did they apply?
    "sedgwick2022_iras_akari_xmatch": "2207.09985",
    # NEAREST PRIOR ART on the fading side, but targeted at a class known to
    # swing by factors of ten between exactly these epochs.
    "melis2023_rcrb_iras_akari_wise": "2306.11945",
    # IRAS-detected debris hosts re-examined with WISE, framed as *calibration*
    # rather than time-domain disappearance. ~1/1000 the scale.
    "liu2020_rhee_iras_wise": "2008.12611",
    # THE DOMINANT CONTAMINANT, quantified: ~8,000/180,000 apparent IRAS
    # excesses correlate with the 100-micron background; a <5 MJy/sr cut leaves
    # 271. This single number sets EMBER's most valuable cut.
    "kennedy_wyatt2012_iras_excess_spurious": "1207.0521",
    # THE EXISTENCE PROOF: IRAS 1983 excess, stable ~25 yr, fell by ~30x
    # between 2008 and 2010, still unexplained.
    "melis2012_tyc8241": "1207.1162",
    "gunther2017_tyc8241_followup": "1611.01371",
    # THE FALSE-POSITIVE FLOOR: 14 of 17 extreme debris disks varied at
    # 3-5 micron between 2010 and 2019.
    "moor2021_extreme_debris_variability": "2103.00568",
    # THE EMPIRICAL STABILITY FLOOR: HD 172555 stable to 4% over 27 years,
    # IRAS 1983 -> WISE 2010. The only such measurement in the literature.
    "hd172555_27yr_stability": "1210.6258",
    # THE THEORETICAL OBJECTION that must be answered in docs/ember.md:
    # technosignature duration is separable from civilization age, and only the
    # longest-lived signatures are likely to be detected.
    "balbi_cirkovic2021_duration": "2103.02923",
    # The decay physics that would have to supply the fade: megaswarms subject
    # to collisional cascade without upkeep -- but on 1e5-1e7 yr timescales.
    "lacki2025_ground_to_dust": "2504.21151",
    # The duty-cycle motivation, with no observational counterpart proposed.
    "blanco2026_collapse_recovery": "2604.13774",
    # The field's stated prior, i.e. WHY the gap exists: "the signature is as
    # long-lived as the underlying technology".
    "wright2019_thermal_ir_whitepaper": "1907.07829",
    # Solar-system moving objects across exactly IRAS/AKARI/WISE.
    "usui2014_solar_system_iras_akari_wise": "1403.7854",
    # Single-epoch baselines to confirm the absence of any cessation test.
    "carrigan2009_iras_dyson": "0811.2376",
    "hephaistos2": "2405.02927",
    "hephaistos_contamination": "2607.03619",
    "hephaistos4_jwst": "2607.09460",
}

# --------------------------------------------------------------------------
# 2. Discovery searches. Empty results here ARE the novelty evidence.
# --------------------------------------------------------------------------
QUERIES: dict[str, str] = {
    # The direct question, phrased six ways.
    "ir_excess_disappeared_search": 'all:"infrared excess" AND (all:disappeared OR all:vanished OR all:"switched off")',
    "excess_faded_survey": 'all:"infrared excess" AND all:faded AND all:survey',
    "cross_epoch_iras_wise_flux": 'all:IRAS AND all:WISE AND all:"flux" AND (all:epoch OR all:"time domain")',
    "iras_akari_wise_variability": 'all:IRAS AND all:AKARI AND all:WISE AND all:variab',
    "midir_excess_time_domain_survey": 'all:"mid-infrared excess" AND all:"time domain"',
    "disappearing_debris_disk_survey": 'all:"debris disk" AND (all:disappear OR all:vanish) AND all:survey',
    # The technosignature framing.
    "technosignature_ceased": 'all:technosignature AND (all:ceased OR all:cessation OR all:"switched off")',
    "dyson_sphere_time_variable": 'all:"Dyson sphere" AND (all:variable OR all:"time domain" OR all:epoch)',
    "waste_heat_disappearance": 'all:"waste heat" AND (all:disappear OR all:cease OR all:decline)',
    "dead_civilization_infrared": 'all:"extinct civilization" OR (all:"dead civilization" AND all:infrared)',
    "megastructure_decay_observable": 'all:megastructure AND (all:decay OR all:collapse) AND all:observable',
    # Systematics lore for the cross-match itself.
    "iras_faint_source_reliability": 'all:IRAS AND all:"faint source" AND (all:reliability OR all:spurious)',
    "iras_confusion_beam": 'all:IRAS AND all:confusion AND all:beam',
    "iras_wise_photometric_comparison": 'all:IRAS AND all:WISE AND all:photometr AND all:compar',
    "akari_irc_calibration": 'all:AKARI AND all:IRC AND all:calibration',
    "wise_saturation_bright_sources": 'all:WISE AND all:saturation AND all:"bright sources"',
    "eddington_bias_flux_limited_ir": 'all:"Eddington bias" AND all:infrared AND all:"flux-limited"',
    # The rate nobody has measured.
    "rate_of_infrared_variability_survey": 'all:"mid-infrared" AND all:variability AND all:rate AND all:survey',
}

# --------------------------------------------------------------------------
# 3. Citation trees: did anyone follow up the antecedents in the fading direction?
# --------------------------------------------------------------------------
CITED_BY: dict[str, str] = {
    # Everyone who used the IRAS+AKARI+WISE cross-epoch method after Kim 2015.
    "kim2015": "10.1088/0004-6256/149/2/50",
    # Everyone who used the IRAS x AKARI all-sky cross-match machinery.
    "sedgwick2022": "10.1093/mnras/stac2044",
    # Everyone who followed up TYC 8241 2652 1.
    "melis2012_tyc8241": "10.1038/nature11210",
    # The cirrus false-excess result that supplies EMBER's best cut.
    "kennedy_wyatt2012": "10.1111/j.1365-2966.2012.21474.x",
}

ARXIV_API = "http://export.arxiv.org/api/query?"
OPENALEX = "https://api.openalex.org/works"


def fetch_fulltext() -> None:
    print("\n=== full text of decisive antecedents ===")
    for name, aid in FULLTEXT.items():
        print(f"[{name}] arXiv:{aid}")
        get(ARXIV_API + urllib.parse.urlencode({"id_list": aid, "max_results": 1}),
            OUT / f"arxiv_id_{name}.atom")
        get(f"https://ar5iv.labs.arxiv.org/html/{aid}", OUT / f"ar5iv_{name}.html")
        pdf = OUT / f"pdf_{name}.pdf"
        if get(f"https://arxiv.org/pdf/{aid}", pdf):
            try:
                subprocess.run(["pdftotext", "-q", str(pdf), str(OUT / f"txt_{name}.txt")],
                               check=False, timeout=120)
                pdf.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  pdftotext failed: {exc!r}")


def fetch_queries() -> None:
    print("\n=== discovery searches (empty results are the evidence) ===")
    for name, q in QUERIES.items():
        print(f"[q:{name}] {q}")
        get(ARXIV_API + urllib.parse.urlencode(
            {"search_query": q, "max_results": 60,
             "sortBy": "relevance", "sortOrder": "descending"}),
            OUT / f"arxiv_q_{name}.atom")


def fetch_citation_trees() -> None:
    print("\n=== OpenAlex citation trees ===")
    for name, doi in CITED_BY.items():
        print(f"[cite:{name}] {doi}")
        get(f"{OPENALEX}/https://doi.org/{doi}", OUT / f"oa_{name}.json")
        get(f"{OPENALEX}?filter=cites:doi:{doi}&per-page=200&sort=publication_date:desc",
            OUT / f"citedby_{name}.json")


def summarise() -> None:
    """Count hits per discovery query so the null results are legible at a glance."""
    import re

    counts = {}
    for path in sorted(OUT.glob("arxiv_q_*.atom")):
        try:
            text = path.read_text(errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        m = re.search(r"opensearch:totalResults[^>]*>(\d+)<", text)
        counts[path.stem.replace("arxiv_q_", "")] = int(m.group(1)) if m else -1
    (OUT / "query_counts.json").write_text(json.dumps(counts, indent=2))
    print("\n=== discovery-query hit counts ===")
    for k, v in sorted(counts.items(), key=lambda kv: kv[1]):
        print(f"  {v:>5}  {k}")


def main() -> None:
    fetch_fulltext()
    fetch_queries()
    fetch_citation_trees()
    summarise()
    ok = sum(1 for s in STATUS if s["ok"])
    (OUT / "summary.json").write_text(json.dumps(
        {"n_urls": len(STATUS), "n_ok": ok, "n_failed": len(STATUS) - ok,
         "claim": ("No published work has searched, blind and at catalogue scale, "
                   "for the disappearance of a mid-infrared excess between the "
                   "IRAS/AKARI and WISE epochs, nor framed such a disappearance "
                   "as a technosignature."),
         "must_cite_and_distinguish": [
             "1501.05721 Kim+2015 -- same three catalogues, brightening only",
             "2207.09985 Sedgwick & Serjeant 2022 -- same cross-match, proper motion",
             "2306.11945 Melis+2023 -- fading, but targeted at R CrB stars",
             "2008.12611 Liu 2020 -- IRAS vs WISE framed as calibration"],
         "status": STATUS}, indent=2))
    print(f"\n=== {ok}/{len(STATUS)} fetches succeeded -> {OUT} ===")


if __name__ == "__main__":
    main()
