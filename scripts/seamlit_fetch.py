#!/usr/bin/env python3
"""Waste-heat SETI "seam" literature survey: fetch on the runner.

Targets the STRUCTURAL-ASSUMPTION questions that the Hephaistos / G-hat / WISE
long-tail fetchers do NOT cover, i.e. the places where an existing search could
be evaded:

  S-A  attenuation-only detection (grey/achromatic dimming, no IR re-emission)
  S-B  Boyajian's star + the dipper / deep-dimming / secular-fading literature
  S-C  cold re-radiation: far-IR (IRAS 60/100um, AKARI FIS, Herschel, Planck)
  S-D  IR excess around OLD / METAL-POOR / KINEMATICALLY-HALO stars
  S-E  time-variable IR excess at catalog scale (NEOWISE 2010-2025; appeared
       or disappeared), plus the known natural cases
  S-F  IRAS(1983) vs WISE(2010+) -- excesses that vanished over ~40 yr
  S-G  non-main-sequence hosts (white dwarfs, subdwarfs, neutron stars, BDs)
  S-H  SED SHAPE as a discriminant (blackbody narrowness, multi-temperature
       decomposition, emissivity law, silicate features)

The sandbox egress policy blocks arxiv.org / ADS / IOP / OUP / Semantic Scholar,
so this runs on the GitHub Actions runner (CLAUDE.md acquisition pattern) and
commits verbatim abstracts back to the branch.

Outputs under results/seamlit/:
  arxiv_q_<name>.atom  - arXiv API search results per query
  arxiv_ids_<n>.atom   - arXiv metadata for specific known IDs
  summary.json         - fetch status for every URL
  digest.txt           - flat title+abstract digest of everything retrieved
"""
from __future__ import annotations

import json
import pathlib
import re
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/seamlit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-seamlit/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []
ARXIV = "http://export.arxiv.org/api/query?"


def get(url: str, dest: pathlib.Path, tries: int = 3, pause: float = 3.0) -> bool:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:46s} {len(data):8d}B", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i + 1}) {url} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url}", flush=True)
    return False


# --- 1. arXiv topic searches -------------------------------------------------
QUERIES: dict[str, str] = {
    # S-A: attenuation-only / achromatic dimming
    "achromatic_dim": 'all:"achromatic" AND all:"dimming" AND abs:"star"',
    "grey_extinction": 'abs:"grey dust" OR abs:"gray extinction" AND abs:"star"',
    "wavelength_indep_dip": 'abs:"wavelength-independent" AND abs:"transit"',
    "opaque_megastructure": 'abs:"megastructure" AND abs:"transit"',
    "occulter_seti": 'abs:"artificial" AND abs:"occultation" AND abs:"SETI"',
    "transit_technosig": 'abs:"transit" AND abs:"technosignature"',
    "nonspherical_transit": 'abs:"nonspherical" OR abs:"non-spherical" AND abs:"transiting"',
    # S-B: Boyajian / dippers / secular fading
    "boyajian": 'abs:"KIC 8462852"',
    "tabbys_star": 'abs:"Boyajian\'s Star"',
    "dipper_search": 'abs:"dipper" AND abs:"survey"',
    "deep_dimming": 'abs:"deep dimming" OR abs:"dimming events" AND abs:"survey"',
    "secular_dimming": 'abs:"secular dimming" OR abs:"long-term dimming"',
    "dasch": 'abs:"DASCH" OR abs:"photographic plates" AND abs:"century"',
    "vanishing_stars": 'abs:"vanishing" AND abs:"sources" AND abs:"survey"',
    "vasco": 'abs:"VASCO" OR ti:"Vanishing and Appearing Sources"',
    "disappearing_star": 'abs:"disappearing star" OR abs:"failed supernova" AND abs:"survey"',
    "ztf_dipper": 'abs:"ZTF" AND abs:"dipping"',
    "tess_long_dimming": 'abs:"TESS" AND abs:"occultation" AND abs:"long-duration"',
    # S-C: cold re-radiation / far-IR
    "farir_technosig": 'abs:"far-infrared" AND abs:"technosignature"',
    "akari_fis_excess": 'abs:"AKARI" AND abs:"far-infrared" AND abs:"excess"',
    "herschel_debris": 'abs:"Herschel" AND abs:"debris disc" AND abs:"survey"',
    "planck_cold": 'abs:"Planck" AND abs:"cold" AND abs:"compact sources"',
    "iras_excess_60": 'abs:"IRAS" AND abs:"60 micron" AND abs:"excess"',
    "cold_dyson": 'abs:"Dyson" AND abs:"cold"',
    "kuiper_belt_analog": 'abs:"Kuiper belt analog" AND abs:"survey"',
    # S-D: old / metal-poor / halo IR excess
    "debris_metallicity": 'abs:"debris disk" AND abs:"metallicity"',
    "debris_metalpoor": 'abs:"debris" AND abs:"metal-poor"',
    "excess_halo_stars": 'abs:"infrared excess" AND abs:"halo stars"',
    "debris_old_stars": 'abs:"debris disc" AND abs:"old stars" OR abs:"age"',
    "excess_thickdisk": 'abs:"infrared excess" AND abs:"thick disk"',
    "debris_subdwarf": 'abs:"subdwarf" AND abs:"infrared excess"',
    "excess_high_velocity": 'abs:"high proper motion" AND abs:"infrared excess"',
    # S-E: time-variable IR excess
    "variable_excess": 'abs:"variable" AND abs:"infrared excess" AND abs:"debris"',
    "extreme_debris_var": 'abs:"extreme debris disk" AND abs:"variability"',
    "disappearing_disk": 'abs:"disappearance" AND abs:"dust" AND abs:"disk"',
    "neowise_var": 'abs:"NEOWISE" AND abs:"variability"',
    "wise_midir_var": 'abs:"WISE" AND abs:"mid-infrared variability" AND abs:"survey"',
    "tyc8241": 'abs:"TYC 8241"',
    "meng_su_dust": 'abs:"collisional" AND abs:"dust" AND abs:"variability" AND abs:"warm"',
    # S-F: IRAS vs WISE longitudinal
    "iras_wise_compare": 'abs:"IRAS" AND abs:"WISE" AND abs:"comparison"',
    "iras_wise_xmatch": 'abs:"IRAS" AND abs:"cross-match" AND abs:"WISE"',
    "faded_ir_source": 'abs:"faded" AND abs:"infrared source"',
    # S-G: exotic hosts
    "wd_dyson": 'abs:"Dyson" AND abs:"white dwarf"',
    "wd_ir_excess": 'abs:"white dwarf" AND abs:"infrared excess" AND abs:"survey"',
    "wired_survey": 'abs:"WIRED" AND abs:"white dwarf"',
    "polluted_wd": 'abs:"polluted white dwarf" AND abs:"accretion"',
    "wd_transit": 'abs:"white dwarf" AND abs:"transiting" AND abs:"debris"',
    "pulsar_dyson": 'abs:"Dyson" AND abs:"pulsar" OR abs:"neutron star"',
    "bd_excess": 'abs:"brown dwarf" AND abs:"infrared excess"',
    "sdb_excess": 'abs:"hot subdwarf" AND abs:"infrared excess"',
    "blackhole_seti": 'abs:"black hole" AND abs:"technosignature"',
    # S-H: SED shape discriminants
    "sed_shape_disc": 'abs:"spectral energy distribution" AND abs:"Dyson"',
    "silicate_feature_debris": 'abs:"silicate" AND abs:"debris disk" AND abs:"feature"',
    "blackbody_narrow": 'abs:"blackbody" AND abs:"technosignature"',
    "emissivity_debris": 'abs:"emissivity" AND abs:"debris disc" AND abs:"modified blackbody"',
    "multitemp_decomp": 'abs:"two-temperature" AND abs:"debris disk"',
    # framing / reviews
    "dysonian_review": 'abs:"Dysonian" AND abs:"SETI"',
    "technosig_review": 'ti:"technosignature" AND abs:"review"',
    "waste_heat": 'abs:"waste heat" AND abs:"civilization"',
}

for name, q in QUERIES.items():
    url = ARXIV + urllib.parse.urlencode(
        {
            "search_query": q,
            "start": 0,
            "max_results": 40,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    get(url, OUT / f"arxiv_q_{name}.atom")
    time.sleep(3.2)  # arXiv API courtesy rate limit


# --- 2. arXiv metadata for specific known IDs --------------------------------
IDS = [
    # Boyajian / KIC 8462852 core
    "1509.03622",  # Boyajian+2016 Planet Hunters IX (discovery)
    "1601.03256",  # Schaefer 2016 DASCH century-long fading
    "1602.03256",  # Hippke+ rebuttal (verify)
    "1605.09022",  # Hippke & Angerhausen (verify)
    "1608.01316",  # Montet & Simon 2016 Kepler secular dimming
    "1801.00732",  # Boyajian+2018 first post-Kepler dips (CHROMATIC)
    "1901.06582",  # Schmidt 2019 search for analogs
    "1511.01122",  # Marengo+ / Lisse+ IR follow-up (verify)
    "1512.03693",  # Thompson+2016 submm limits (verify)
    "1705.10361",  # Wyatt+ modelling (verify)
    "1612.03170",  # Meng+ Swift photometry (verify)
    # Hephaistos + refutations
    "2405.02927",  # Hephaistos II
    "2405.14921",  # Background contamination (hot DOGs)
    "2501.05152",  # Radio imaging of candidate G
    # G-hat
    "1504.03418",  # G-hat III reddest extended sources
    # theory / temperature
    "2309.06564",  # Wright 2023 thermodynamics of Dyson spheres
    "2602.23270",  # Amiri 2026 Dyson spheres on H-R diagram
    "1907.07829",  # Technosignatures in the thermal infrared
    "1908.02683",  # Nine axes of merit
    "2103.01536",  # Concepts for future technosignature missions
    # variable / extreme debris disks
    "2103.00568",  # warm extreme debris disks from ALLWISE
    "2108.02901",  # V488 Per extreme variability
    "2605.19059",  # VarWISE
    # VASCO
    "1911.05068",  # VASCO I
    "2009.10813",  # VASCO citizen science
    "2602.15171",  # Villarroel response to Watters+2026
    "2604.04810",  # independent recovery POSS-I
    # metallicity / debris
    "1604.07403",  # Gaspar+2016 metallicity-debris disk mass
    # WISE debris catalogs
    "1403.3435",  # Patel/Metchev/Heinze warm debris disks
    "1308.3848",  # Bright 22um excess candidates
    "1308.5593",  # AKARI FIS bright debris disk candidates
    "1211.6365",  # AKARI IRC 18um warm debris disks
]
for i in range(0, len(IDS), 12):
    chunk = IDS[i : i + 12]
    url = ARXIV + urllib.parse.urlencode(
        {"id_list": ",".join(chunk), "max_results": len(chunk)}
    )
    get(url, OUT / f"arxiv_ids_{i // 12}.atom")
    time.sleep(3.2)


# --- 3. flat digest ----------------------------------------------------------
def strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s)).strip()


entries: dict[str, dict] = {}
for f in sorted(OUT.glob("*.atom")):
    try:
        txt = f.read_text(errors="replace")
    except Exception:  # noqa: BLE001
        continue
    for m in re.finditer(r"<entry>(.*?)</entry>", txt, re.S):
        e = m.group(1)
        idm = re.search(r"<id>(.*?)</id>", e, re.S)
        tim = re.search(r"<title>(.*?)</title>", e, re.S)
        sm = re.search(r"<summary>(.*?)</summary>", e, re.S)
        pm = re.search(r"<published>(.*?)</published>", e, re.S)
        jm = re.search(r'<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>', e, re.S)
        dm = re.search(r'<arxiv:doi[^>]*>(.*?)</arxiv:doi>', e, re.S)
        auth = [strip_tags(a) for a in re.findall(r"<name>(.*?)</name>", e, re.S)]
        if not idm:
            continue
        aid = strip_tags(idm.group(1)).rsplit("/", 1)[-1]
        entries[aid] = {
            "arxiv": aid,
            "title": strip_tags(tim.group(1)) if tim else "",
            "authors": auth,
            "published": strip_tags(pm.group(1)) if pm else "",
            "journal_ref": strip_tags(jm.group(1)) if jm else "",
            "doi": strip_tags(dm.group(1)) if dm else "",
            "abstract": strip_tags(sm.group(1)) if sm else "",
            "found_in": f.name,
        }

(OUT / "entries.json").write_text(json.dumps(list(entries.values()), indent=1))
with (OUT / "digest.txt").open("w") as fh:
    for e in sorted(entries.values(), key=lambda x: x["published"], reverse=True):
        fh.write(f"=== arXiv:{e['arxiv']}  [{e['published'][:10]}]\n")
        fh.write(f"TITLE  : {e['title']}\n")
        fh.write(f"AUTHORS: {', '.join(e['authors'][:8])}\n")
        if e["journal_ref"]:
            fh.write(f"JOURNAL: {e['journal_ref']}\n")
        if e["doi"]:
            fh.write(f"DOI    : {e['doi']}\n")
        fh.write(f"ABS    : {e['abstract']}\n\n")

(OUT / "summary.json").write_text(json.dumps(STATUS, indent=1))
ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n=== {ok}/{len(STATUS)} fetches OK; {len(entries)} unique arXiv entries ===")
