#!/usr/bin/env python3
"""Prior-art adjudication for five proposed disappearance-class searches.

arXiv's API only indexes title+abstract and misses journal-only papers
(MNRAS/A&A/AJ items never posted to arXiv, e.g. the Hambly & Blair critique and
Solano & Villarroel 2022). OpenAlex indexes abstracts for essentially the whole
literature and exposes citation counts and referenced works, so it is the better
recall instrument for "has anyone ever done X". Crossref is the tie-breaker for
exact bibliographic metadata.

Adjudicates:
  S1  IRAS(1983)/AKARI(2006) vs WISE(2010)/NEOWISE(2024) cross-epoch search for
      infrared excesses that VANISHED.
  S31 NVSS/FIRST radio sources with STELLAR (Gaia) counterparts absent in VLASS.
  S32 Variable stars whose periodic signal CEASED.
  S33 VASCO-vanished optical sources that HAVE a WISE/AllWISE IR counterpart.
  S9  Stars whose photometric scatter is SECULARLY INCREASING over the
      ZTF/ATLAS/ASAS-SN decade.

Runs on the GitHub Actions runner; the sandbox blocks these hosts.

Outputs under results/disaplit3/:
  oa_<sid>_<name>.json   - OpenAlex work lists per query, per prior-art question
  cr_<name>.json         - Crossref metadata for named papers
  priorart_digest.txt    - flattened title/year/venue/DOI/citations per query
  summary.json           - fetch status for every URL
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/disaplit3")
OUT.mkdir(parents=True, exist_ok=True)

MAIL = "trimcrae@gmail.com"
UA = {"User-Agent": f"Seti-disaplit3/1.0 (mailto:{MAIL})"}
STATUS: list[dict] = []


def get(url: str, dest: pathlib.Path, tries: int = 3, pause: float = 2.5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:52s} {len(data):8d}B", flush=True)
            try:
                return json.loads(data)
            except Exception:  # noqa: BLE001
                return None
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {url} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url}", flush=True)
    return None


OA = "https://api.openalex.org/works?"


def oa_search(sid: str, name: str, phrase: str, extra: str = "") -> list[dict]:
    """OpenAlex title+abstract search, most-cited first."""
    params = {
        "search": phrase,
        "per-page": "50",
        "sort": "cited_by_count:desc",
        "mailto": MAIL,
    }
    if extra:
        params["filter"] = extra
    d = get(OA + urllib.parse.urlencode(params), OUT / f"oa_{sid}_{name}.json")
    time.sleep(1.2)
    return (d or {}).get("results", []) or []


# Each prior-art question gets a battery of phrasings. Recall matters more than
# precision here: a single missed paper flips a verdict.
QUERIES: dict[str, dict[str, str]] = {
    # ---- S1: vanished infrared excesses across mission epochs ----
    "S1": {
        "iras_wise_excess_compare": "IRAS WISE infrared excess comparison epochs",
        "disappearing_debris_disk": "disappearing debris disk infrared excess",
        "disk_vanished": "debris disk vanished dust depletion rapid",
        "ir_excess_variability_survey": "infrared excess variability survey main sequence stars",
        "extreme_debris_disk_var": "extreme debris disk variability",
        "iras_counterpart_wise": "IRAS source WISE counterpart identification cross-match",
        "akari_wise_compare": "AKARI WISE point source catalogue comparison photometry",
        "iras_reliability": "IRAS Faint Source Catalog reliability spurious confusion cirrus",
        "warm_dust_disappear": "warm dust disappearance circumstellar rapid decline",
        "dyson_ir_excess_search": "Dyson sphere infrared excess search candidates",
    },
    # ---- S31: radio sources absent in VLASS with stellar counterparts ----
    "S31": {
        "nvss_vlass_fade": "NVSS VLASS fading radio sources comparison",
        "radio_transient_survey_compare": "radio transient survey comparing epochs NVSS FIRST",
        "vanishing_radio": "vanishing disappearing radio sources survey",
        "radio_stars_gaia_xmatch": "radio stars Gaia cross-match survey counterparts",
        "vlass_stellar_counterpart": "VLASS stellar counterparts radio emission stars",
        "remnant_radio_galaxy": "remnant dying radio galaxy switched off AGN",
        "first_variability": "variable transient sources FIRST survey",
        "askap_vast_variables": "ASKAP VAST variable radio sources survey",
        "radio_star_census": "census of radio emitting stars low frequency survey",
        "nvss_resolution_bias": "NVSS FIRST flux discrepancy resolved out extended emission",
    },
    # ---- S32: periodic signals that ceased ----
    "S32": {
        "ceased_pulsation": "star ceased pulsating pulsation stopped amplitude decline",
        "variability_ceased": "cessation of variability star no longer variable",
        "eclipses_disappeared": "eclipsing binary eclipses disappeared ceased inclination precession",
        "mode_switch_rrl": "RR Lyrae mode switching pulsation cessation",
        "cepheid_amplitude_decline": "Cepheid pulsation amplitude decline ceasing",
        "crossepoch_variable_compare": "cross-epoch comparison variable star catalogues survey missing",
        "gcvs_verification": "General Catalogue of Variable Stars verification constant stars modern survey",
        "disappearing_transits": "disappearing transits vanished planet transit signal",
        "mira_period_change": "Mira period change thermal pulse evolution variability",
        "pulsation_amplitude_variable_wd": "white dwarf pulsation amplitude variability disappearing modes",
    },
    # ---- S33: VASCO vanished sources vs infrared ----
    "S33": {
        "vasco": "vanishing and appearing sources during a century of observations",
        "vanished_star_infrared": "vanished star infrared counterpart obscured dust enshrouded",
        "poss_transient_critique": "Palomar sky survey plate transient emulsion defect artefact",
        "missing_star_usno": "USNO objects missing modern sky surveys vanishing candidates",
        "plate_transient_replication": "photographic plate transients independent replication analysis",
        "obscured_not_destroyed": "star disappeared optical brightened infrared dust obscuration event",
    },
    # ---- S9: secularly increasing photometric scatter ----
    "S9": {
        "increasing_variability": "increasing variability amplitude stars over time secular",
        "onset_of_variability": "onset of variability newly variable star became variable",
        "long_term_variability_asassn": "long term slow variability search ASAS-SN survey",
        "secular_dimming_stars": "secular dimming long term brightness decline stars survey",
        "photometric_scatter_evolution": "photometric scatter evolution time survey stars trend",
        "dasch_century": "DASCH century photographic light curves long term variability",
        "plate_vs_modern_photometry": "photographic plate magnitudes compared modern survey photometry systematic",
        "ztf_secular_trends": "ZTF long term secular trends stellar light curves",
    },
}

digest: list[str] = []
for sid, qs in QUERIES.items():
    digest.append(f"\n{'#'*78}\n# {sid}\n{'#'*78}")
    for name, phrase in qs.items():
        res = oa_search(sid, name, phrase)
        digest.append(f"\n--- [{sid}] {name}: \"{phrase}\"  ({len(res)} results)")
        for w in res[:25]:
            loc = (w.get("primary_location") or {}).get("source") or {}
            digest.append(
                f"  {w.get('publication_year')}  cits={w.get('cited_by_count'):>5}  "
                f"{(loc.get('display_name') or '?')[:34]:34s}  "
                f"{(w.get('doi') or '')[:44]:44s}  {(w.get('title') or '')[:120]}"
            )

# --- Crossref: exact metadata for papers whose citation I must get right ------
CR = "https://api.crossref.org/works?"
CR_QUERIES = {
    "hambly_blair_poss": "Hambly Palomar sky survey transient emulsion",
    "solano_villarroel_2022": "Solano Villarroel vanishing objects POSS Virtual Observatory",
    "villarroel_bruehl_pasp": "Villarroel aligned multiple transient events Palomar",
    "melis_2012_disk": "Melis rapid disappearance warm dusty circumstellar disk",
    "dasch_dr7": "DASCH Digital Access Sky Century photographic data",
}
for name, q in CR_QUERIES.items():
    get(CR + urllib.parse.urlencode({"query.bibliographic": q, "rows": 12,
                                     "mailto": MAIL}),
        OUT / f"cr_{name}.json")
    time.sleep(1.5)

(OUT / "priorart_digest.txt").write_text("\n".join(digest))
(OUT / "summary.json").write_text(json.dumps(STATUS, indent=2))
ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n=== disaplit3 done: {ok}/{len(STATUS)} OK; digest lines={len(digest)} ===")
