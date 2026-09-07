#!/usr/bin/env python3
"""LZ 248 keV nuclear-recoil candidate: fetch the record on the GitHub runner.

The sandbox blocks arxiv.org, inspirehep.net, hepdata.net, zenodo.org and
lz.lbl.gov (``CONNECT tunnel failed``); the Actions runner has egress.  This
script pulls, verbatim, everything the LZ-event analysis needs to be built
from the record rather than from memory:

  1. arXiv:2609.02823 (LZ, extended nuclear-recoil window, one event at
     248 +- 23 +- 23 keV, 2.84 t yr) as PDF -> text and as e-print source;
  2. every follow-up paper that cites it (INSPIRE ``refersto`` plus arXiv
     keyword sweeps sorted by submission date), because the novelty of any
     further analysis is decided against those, not against the LZ paper;
  3. the prior LZ / XENON / PandaX / LUX high-energy, EFT and inelastic
     searches, for the cross-experiment exposure accounting;
  4. the local-halo kinematics record: escape speed from Gaia, SHM++, the
     Sausage, the S1 stream and shards, the LMC-boosted high-speed tail --
     the astrophysics that sets the kinematic edge an endothermic
     interpretation of a 248 keV recoil lives on;
  5. the seasonal-background record (underground muon flux at SURF) that is
     the degenerate alternative to an annually modulated signal;
  6. the LZ data releases: HEPData tables for every LZ record, the Zenodo
     listing, and the collaboration's own preprint / press pages.

Nothing is paraphrased.  Every asserted arXiv id is title-checked against the
fetched metadata and the mismatches are written to ``id_title_check.json``;
every URL's HTTP status is in ``summary.json``.  PDFs are converted with
``pdftotext -layout`` and only the text is committed.

Outputs under ``results/lzlit/``:
  papers.json               id -> {title, authors, published, updated, abstract,
                            categories, source_key, text_ok, src_ok}
  followups.json            the citing / keyword-found follow-ups, newest first
  id_title_check.json       asserted id vs fetched title; title-search hits
  text/<id>.txt             pdftotext of the arXiv PDF
  src/<id>/...              text-like files of the e-print source (tex, bib,
                            dat, csv, txt, json), size-capped
  meta/*.atom|*.json|*.html verbatim API responses and pages
  data/hepdata/<ins>/*.csv  every table of every LZ HEPData record
  summary.json              HTTP status of every URL, counts, timing
"""
from __future__ import annotations

import gzip
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

OUT = pathlib.Path(os.environ.get("LZLIT_OUT", "results/lzlit"))
TEXT = OUT / "text"
SRC = OUT / "src"
META = OUT / "meta"
DATA = OUT / "data"
for d in (OUT, TEXT, SRC, META, DATA):
    d.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-lzlit/1.0 (mailto:trimcrae@gmail.com)"}
PAUSE = float(os.environ.get("LZLIT_PAUSE", "3.0"))
TRIES = int(os.environ.get("LZLIT_TRIES", "3"))
DRYRUN = os.environ.get("LZLIT_DRYRUN", "") == "1"
ARXIV_API = "http://export.arxiv.org/api/query"
T0 = time.time()
STATUS: list[dict] = []
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
MAIN_ID = "2609.02823"

# ---------------------------------------------------------------------------
# What to fetch.  ``want_source`` pulls the e-print (for tables / numbers that
# the PDF text mangles).  ``title`` is the expected title fragment, checked
# case-insensitively against what arXiv returns -- a mismatch is recorded, never
# silently trusted.
# ---------------------------------------------------------------------------
BY_ID: dict[str, dict] = {
    # --- the result and the collaboration's own record -----------------------
    "lz2026_extended_nr": {"id": "2609.02823", "src": True,
        "title": "extended nuclear recoil energy window"},
    "lz2024_ws2024": {"id": "2410.17036", "src": True,
        "title": "4.2 Tonne-Years"},
    "lz2022_first": {"id": "2207.03764", "src": True,
        "title": "First Dark Matter Search Results from the LUX-ZEPLIN"},
    "lz2024_eft_sr1": {"id": "2404.17666", "src": True,
        "title": "Effective Field Theory"},
    "lz2020_projected_backgrounds": {"id": "2001.09363", "src": False,
        "title": "backgrounds"},
    "lz2025_er_lowenergy": {"id": "2511.17350", "src": False,
        "title": "Electron Recoils"},
    "lz2025_crboosted": {"id": "2503.18158", "src": False,
        "title": "boosted"},
    # --- follow-ups known at the time of writing (2026-09-07) --------------
    "fu_2609.01475_idm_lz_cresst": {"id": "2609.01475", "src": True,
        "title": "Inelastic Dark Matter Signature at High Recoil Energy"},
    "fu_2609.01583_higgsino": {"id": "2609.01583", "src": True,
        "title": "Higgsino Dark Matter Interpretation"},
    "fu_2609.01592_fermionic_absorption": {"id": "2609.01592", "src": True,
        "title": "Fermionic Dark Matter Absorption"},
    "fu_2609.02608_kinematic_edge": {"id": "2609.02608", "src": True,
        "title": "Kinematic Edge"},
    "fu_2609.02807_pq_inelastic": {"id": "2609.02807", "src": True,
        "title": "Peccei"},
    "fu_2609.02868_dark_photon_gc": {"id": "2609.02868", "src": True,
        "title": "Dark Photon"},
    "fu_2609.04175_higgsino_sideband": {"id": "2609.04175", "src": True,
        "title": "Sideband"},
    "fu_2609.04181_seasonal_mccabe": {"id": "2609.04181", "src": True,
        "title": "Seasonal dark matter"},
    "fu_2609.04185_atm_nu_upscatter": {"id": "2609.04185", "src": True,
        "title": "Atmospheric neutrino"},
    "fu_2609.04186_axion_portal": {"id": "2609.04186", "src": True,
        "title": "Axion Portal"},
    # --- other experiments' high-energy / EFT / inelastic searches ----------
    "xenon100_eft_high_energy": {"id": "1705.02614", "src": False,
        "title": "high-energy nuclear recoils"},
    "xenonnt_wimp_modeling": {"id": "2406.13638", "src": False,
        "title": "Signal"},
    "xenonnt_lowE_ionization": {"id": "2601.11296", "src": False,
        "title": "Ionization"},
    "pandax4t_light_idm": {"id": "2508.13062", "src": False,
        "title": "Inelastic"},
    "bramante2016_inelastic_frontier": {"id": "1608.02662", "src": True,
        "title": "Inelastic frontier"},
    "reporting_conventions_2021": {"id": "2105.00599", "src": False,
        "title": "Recommended conventions"},
    # --- local halo kinematics --------------------------------------------
    "besla2019_lmc_highest_speed": {"id": "1909.04140", "src": True,
        "title": "highest-speed local dark matter particles"},
    "smithorlik2023_lmc_dd": {"id": "2302.04281", "src": True,
        "title": "Large Magellanic Cloud"},
    "smithorlik2024_lmc_lowmass": {"id": "2409.09119", "src": True,
        "title": "Large Magellanic Cloud"},
    "astro_uncertainties_2025": {"id": "2505.07924", "src": True,
        "title": "Astrophysical Uncertainties"},
    "extragalactic_flux_2025": {"id": "2507.01190", "src": True,
        "title": "Extragalactic Dark Matter"},
    "roche2024_vesc_gaia_dr3": {"id": "2402.00108", "src": True,
        "title": "Escape Velocity"},
    "evans2019_shmpp": {"id": "1810.11468", "src": True,
        "title": "Refinement of the standard halo model"},
    "necib2019_substructure_sdss_gaia": {"id": "1807.02519", "src": False,
        "title": "Kinematic Substructure"},
    "ohare2018_hurricane_s1": {"id": "1807.09004", "src": True,
        "title": "hurricane"},
    "ohare2020_dark_shards": {"id": "1909.04684", "src": True,
        "title": "Shards"},
    "koppelman2021_vesc_pm": {"id": "2006.16283", "src": False,
        "title": "escape velocity"},
    "deason2019_high_velocity_tail": {"id": "1901.02016", "src": True,
        "title": "escape speed"},
    "monari2018_vesc_dr2": {"id": "1807.04565", "src": False,
        "title": "escape speed"},
    "bozorgnia2016_simulated_mw": {"id": "1601.04707", "src": False,
        "title": "Simulated Milky Way analogues"},
    "herzog2018_empirical_metal_poor": {"id": "1704.04499", "src": False,
        "title": "metal-poor stars"},
    "freese2013_modulation_review": {"id": "1209.3339", "src": False,
        "title": "Annual modulation"},
    "lee2013_gravitational_focusing": {"id": "1308.1953", "src": False,
        "title": "focusing"},
    # --- seasonal backgrounds ---------------------------------------------
    "majorana2017_surf_muon_flux": {"id": "1602.07742", "src": False,
        "title": "Muon Flux"},
    "borexino2019_muon_modulation": {"id": "1808.04207", "src": False,
        "title": "muon"},
    # --- added 2026-09-07 after the first sweep ------------------------------
    "lmc_high_speed_flux_high_mass_2025": {"id": "2511.21841", "src": True,
        "title": "High Speed Flux"},
    "lmc_annihilation_2025": {"id": "2509.13540", "src": False,
        "title": "Large Magellanic Cloud"},
    "garavito2019_lmc_wake": {"id": "1902.05089", "src": False,
        "title": "Wake"},
    "lux_eft_2020": {"id": "2003.11141", "src": False,
        "title": "Effective Field Theory"},
    "lmc_directional_2026": {"id": "2606.12535", "src": True,
        "title": "Directional dark matter signatures of the Large Magellanic Cloud"},
    "empirical_speed_distribution_2025": {"id": "2510.21914", "src": True,
        "title": "Empirical Speed Distribution"},
    "necib_lin2022_substructure_ii": {"id": "2102.02211", "src": True,
        "title": "Local Escape Velocity"},
    "donaldson2022_lmc_local": {"id": "2111.15440", "src": True,
        "title": "Large Magellanic Cloud"},
    "lamost2025_escape_curve": {"id": "2510.18227", "src": False,
        "title": "Escape Velocity"},
}

# Papers whose arXiv id is not certain: resolved by title search, so the id
# cannot be wrong; the search hit is recorded in id_title_check.json.
BY_TITLE: dict[str, dict] = {
    "xenon1t_eft_idm_2024": {"src": True,
        "title": "Effective field theory and inelastic dark matter results from XENON1T"},
    "xenon1t_inelastic_2021": {"src": False,
        "title": "Search for inelastic scattering of WIMP dark matter in XENON1T"},
    "pandax4t_2024_1p54ty": {"src": False,
        "title": "Dark Matter Search Results from 1.54 Tonne Year Exposure of PandaX-4T"},
    "lux_eft_2021": {"src": False,
        "title": "Effective field theory analysis of the first LUX dark matter search"},
    "journey_periodic_table_idm": {"src": True,
        "title": "Pushing the frontier of WIMPy inelastic dark matter: Journey to the end of the periodic table"},
    "cosine100_inelastic_iodine": {"src": False,
        "title": "Search for inelastic WIMP-iodine scattering with COSINE-100"},
    "necib_lin_2022_chasing_dark": {"src": True,
        "title": "Chasing the Dark: Gaia's escape velocity"},
    "donaldson2022_lmc_local_dm": {"src": True,
        "title": "Effects of the Large Magellanic Cloud on the local dark matter"},
    "lz_detector_2019": {"src": False,
        "title": "The LUX-ZEPLIN (LZ) Experiment"},
    "lz_muon_veto_or_outer_detector": {"src": False,
        "title": "LUX-ZEPLIN outer detector"},
    "xenonnt_sr0_sr1_wimp_2025": {"src": False,
        "title": "WIMP Dark Matter Search using a 3.1 tonne x year Exposure of the XENONnT Experiment"},
    "besla_lmc_reflex_motion": {"src": False,
        "title": "Dark matter dynamics and the reflex motion of the Milky Way"},
}

# Keyword sweeps, newest first, to catch follow-ups posted after this file was
# written (and the ones INSPIRE has not yet indexed).
KEYWORD: dict[str, str] = {
    "lz_248kev": 'all:"LUX-ZEPLIN" AND all:248',
    "lz_high_energy_event": 'abs:"LUX-ZEPLIN" AND abs:"nuclear recoil" AND abs:event',
    "lz_high_recoil": 'all:"LZ" AND all:"high-recoil"',
    "lz_high_energy_event2": 'all:"LZ" AND all:"high-energy event"',
    "inelastic_dm_2026": 'abs:"inelastic dark matter" AND abs:"LUX-ZEPLIN"',
    "xenonnt_eft": 'all:XENONnT AND all:"effective field theory"',
    "xenonnt_inelastic": 'all:XENONnT AND all:inelastic AND all:"nuclear recoil"',
    "pandax4t_eft": 'all:"PandaX-4T" AND all:"effective field theory"',
    "pandax4t_inelastic_nr": 'all:"PandaX-4T" AND all:inelastic AND all:"nuclear recoil"',
    "lmc_direct_detection": 'abs:"Large Magellanic Cloud" AND abs:"direct detection"',
    "escape_velocity_gaia_dr3": 'abs:"escape velocity" AND abs:"Gaia DR3" AND abs:"Milky Way"',
    "muon_seasonal_surf": 'abs:muon AND abs:seasonal AND abs:"Sanford Underground"',
    "annual_modulation_inelastic_high_energy": 'abs:"annual modulation" AND abs:inelastic AND abs:"dark matter" AND abs:"high energy"',
}

INSPIRE_QUERIES: dict[str, str] = {
    "refersto_main": f"refersto:arxiv:{MAIN_ID}",
    "record_main": f"arxiv:{MAIN_ID}",
    "lz_collab_2026": 'collaboration:LZ and date>2025',
    "lz_248_fulltext": 'fulltext:"248 keV" and fulltext:"LUX-ZEPLIN" and date>2026',
}

HEPDATA_SEARCHES: dict[str, str] = {
    "lz": "collaboration:LZ",
    "lux_zeplin": '"LUX-ZEPLIN"',
    "main_arxiv": f"arxiv:{MAIN_ID}",
    "xenonnt": "collaboration:XENON",
    "pandax": "collaboration:PandaX",
}

ZENODO_QUERIES: dict[str, str] = {
    "lux_zeplin": '"LUX-ZEPLIN"',
    "lz_dark_matter_data_release": 'LZ AND "dark matter" AND "data release"',
    "xenonnt_release": 'XENONnT AND "data release"',
}

PAGES: dict[str, str] = {
    "lz_lbl_home.html": "https://lz.lbl.gov/",
    "lz_lbl_press.html": "https://lz.lbl.gov/press/",
    "lz_lbl_publications.html": "https://lz.lbl.gov/publications/",
    "lz_lbl_data.html": "https://lz.lbl.gov/data/",
    "lz_lbl_data_release.html": "https://lz.lbl.gov/data-release/",
    "lz_lbl_preprint_260901.pdf": "https://lz.lbl.gov/wp-content/uploads/sites/6/2026/08/LZ_Preprint_260901_Dark_Matter_EFT_Nuclear_Recoil_Search_at_Higher_Energies.pdf",
    "hepdata_ins2841863.json": "https://www.hepdata.net/record/ins2841863?format=json",
}


def _write_summary(extra: dict | None = None) -> None:
    payload = {
        "n_urls": len(STATUS),
        "n_ok": sum(1 for s in STATUS if s["ok"]),
        "n_failed": sum(1 for s in STATUS if not s["ok"]),
        "elapsed_s": round(time.time() - T0, 1),
        "status": STATUS,
    }
    if extra:
        payload.update(extra)
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2))


def get(url: str, *, tries: int = TRIES, pause: float = PAUSE, timeout: int = 180,
        label: str = "") -> bytes | None:
    """Fetch one URL verbatim; every attempt's HTTP status is recorded."""
    if DRYRUN:
        print(f"  DRYRUN {url}")
        STATUS.append({"url": url, "label": label, "ok": False, "http_status": None, "dryrun": True})
        return None
    attempts: list[dict] = []
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                code = int(getattr(r, "status", 0) or 0)
                final = r.geturl()
            attempts.append({"http_status": code, "bytes": len(data)})
            STATUS.append({"url": url, "label": label, "ok": True, "http_status": code,
                           "bytes": len(data), "final_url": final, "attempts": attempts})
            print(f"  ok  HTTP {code} {len(data):>10,}B  {label or url}")
            _write_summary()
            time.sleep(pause)
            return data
        except urllib.error.HTTPError as exc:
            attempts.append({"http_status": int(exc.code), "error": repr(exc)})
            print(f"  try {i + 1}/{tries} HTTP {exc.code}: {url}")
            if exc.code in (403, 404):
                break
            time.sleep(pause * (i + 1))
        except Exception as exc:  # noqa: BLE001
            attempts.append({"http_status": None, "error": repr(exc)})
            print(f"  try {i + 1}/{tries} failed: {exc!r}  {url}")
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "label": label, "ok": False,
                   "http_status": attempts[-1]["http_status"] if attempts else None,
                   "attempts": attempts})
    _write_summary()
    return None


# ---------------------------------------------------------------------------
# arXiv metadata
# ---------------------------------------------------------------------------
def parse_atom(data: bytes) -> list[dict]:
    root = ET.fromstring(data)
    out = []
    for e in root.findall("a:entry", NS):
        idurl = (e.findtext("a:id", default="", namespaces=NS) or "").strip()
        m = re.search(r"abs/([^v]+)(v\d+)?$", idurl)
        aid = m.group(1) if m else idurl
        ver = m.group(2) if m and m.group(2) else ""
        out.append({
            "id": aid, "version": ver,
            "title": re.sub(r"\s+", " ", (e.findtext("a:title", default="", namespaces=NS) or "")).strip(),
            "abstract": re.sub(r"\s+", " ", (e.findtext("a:summary", default="", namespaces=NS) or "")).strip(),
            "published": (e.findtext("a:published", default="", namespaces=NS) or "").strip(),
            "updated": (e.findtext("a:updated", default="", namespaces=NS) or "").strip(),
            "authors": [(a.findtext("a:name", default="", namespaces=NS) or "").strip()
                        for a in e.findall("a:author", NS)],
            "categories": [c.get("term") for c in e.findall("a:category", NS)],
            "comment": (e.findtext("arxiv:comment", default="", namespaces=NS) or "").strip(),
            "journal_ref": (e.findtext("arxiv:journal_ref", default="", namespaces=NS) or "").strip(),
        })
    return out


def arxiv_ids_meta(ids: list[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for n in range(0, len(ids), 20):
        chunk = ids[n:n + 20]
        url = f"{ARXIV_API}?id_list={','.join(chunk)}&max_results={len(chunk)}"
        data = get(url, label=f"arxiv id_list chunk {n // 20}")
        if not data:
            continue
        (META / f"arxiv_ids_{n // 20}.atom").write_bytes(data)
        for ent in parse_atom(data):
            found[ent["id"]] = ent
    return found


def arxiv_search(query: str, name: str, max_results: int = 50, newest: bool = False) -> list[dict]:
    q = urllib.parse.quote(query, safe="")
    url = f"{ARXIV_API}?search_query={q}&max_results={max_results}"
    if newest:
        url += "&sortBy=submittedDate&sortOrder=descending"
    data = get(url, label=f"arxiv search {name}")
    if not data:
        return []
    (META / f"arxiv_q_{name}.atom").write_bytes(data)
    return parse_atom(data)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


# ---------------------------------------------------------------------------
# Full text
# ---------------------------------------------------------------------------
def pdf_to_text(pdf: pathlib.Path, txt: pathlib.Path) -> bool:
    if shutil.which("pdftotext"):
        r = subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)],
                           capture_output=True, text=True)
        if r.returncode == 0 and txt.exists() and txt.stat().st_size > 1000:
            return True
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(pdf))
        txt.write_text("\n\f\n".join((p.extract_text() or "") for p in reader.pages))
        return txt.stat().st_size > 1000
    except Exception as exc:  # noqa: BLE001
        print(f"  pdf->text failed for {pdf.name}: {exc!r}")
        return False


def fetch_pdf_text(aid: str) -> bool:
    txt = TEXT / f"{aid}.txt"
    if txt.exists() and txt.stat().st_size > 1000:
        return True
    data = get(f"https://arxiv.org/pdf/{aid}", label=f"pdf {aid}")
    if not data or not data.startswith(b"%PDF"):
        return False
    tmp = OUT / "_tmp"
    tmp.mkdir(exist_ok=True)
    pdf = tmp / f"{aid}.pdf"
    pdf.write_bytes(data)
    ok = pdf_to_text(pdf, txt)
    pdf.unlink(missing_ok=True)
    return ok


TEXT_EXT = {".tex", ".bib", ".bbl", ".txt", ".dat", ".csv", ".json", ".md", ".sty", ".cls", ".tsv"}
SRC_CAP = 3_000_000


def fetch_source(aid: str) -> bool:
    dest = SRC / aid
    if dest.exists() and any(dest.iterdir()):
        return True
    data = get(f"https://arxiv.org/e-print/{aid}", label=f"e-print {aid}")
    if not data:
        return False
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile() or m.size > SRC_CAP:
                    continue
                ext = pathlib.Path(m.name).suffix.lower()
                if ext not in TEXT_EXT:
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                target = dest / pathlib.Path(m.name).name
                target.write_bytes(f.read())
                n += 1
    except tarfile.TarError:
        try:
            raw = gzip.decompress(data)
            if raw.startswith(b"%PDF"):
                (dest / "SOURCE_IS_PDF_ONLY").write_text("e-print is a PDF; text is in text/")
            else:
                (dest / f"{aid}.tex").write_bytes(raw[:SRC_CAP])
                n = 1
        except Exception:  # noqa: BLE001
            if data.startswith(b"%PDF"):
                (dest / "SOURCE_IS_PDF_ONLY").write_text("e-print is a PDF; text is in text/")
            else:
                (dest / "SOURCE_UNREADABLE").write_text(f"{len(data)} bytes, not tar/gz/pdf")
    return n > 0


# ---------------------------------------------------------------------------
# HEPData / Zenodo / INSPIRE / pages
# ---------------------------------------------------------------------------
def hepdata() -> dict:
    report: dict = {"searches": {}, "records": {}}
    inspire_ids: set[str] = set()
    for name, q in HEPDATA_SEARCHES.items():
        url = f"https://www.hepdata.net/search/?q={urllib.parse.quote(q, safe='')}&format=json&size=50"
        data = get(url, label=f"hepdata search {name}")
        if not data:
            continue
        (DATA / f"hepdata_search_{name}.json").write_bytes(data)
        try:
            j = json.loads(data)
        except Exception:  # noqa: BLE001
            continue
        results = j.get("results", []) if isinstance(j, dict) else []
        rows = []
        for r in results:
            ins = r.get("inspire_id") or r.get("record", {}).get("inspire_id")
            rows.append({"inspire_id": ins, "title": r.get("title"), "collaborations": r.get("collaborations"),
                         "year": r.get("year"), "arxiv": r.get("arxiv_id") or r.get("arxiv")})
            if name in ("lz", "lux_zeplin", "main_arxiv") and ins:
                inspire_ids.add(str(ins))
        report["searches"][name] = rows
    inspire_ids.add("2841863")  # LZ WS2024 (4.2 t yr), the record the web search surfaced
    for ins in sorted(inspire_ids):
        rec_url = f"https://www.hepdata.net/record/ins{ins}?format=json"
        data = get(rec_url, label=f"hepdata record ins{ins}")
        if not data:
            continue
        rdir = DATA / "hepdata" / f"ins{ins}"
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "record.json").write_bytes(data)
        try:
            j = json.loads(data)
        except Exception:  # noqa: BLE001
            continue
        tables = j.get("data_tables", []) or []
        version = j.get("version", 1) or 1
        names = []
        for t in tables[:120]:
            tname = t.get("name")
            if not tname:
                continue
            names.append(tname)
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", tname)[:80]
            for fmt in ("csv", "json"):
                turl = (f"https://www.hepdata.net/download/table/ins{ins}/"
                        f"{urllib.parse.quote(tname, safe='')}/{version}/{fmt}")
                td = get(turl, label=f"hepdata table ins{ins} {tname} {fmt}", pause=1.0)
                if td:
                    (rdir / f"{safe}.{fmt}").write_bytes(td)
        report["records"][ins] = {"title": (j.get("record") or {}).get("title"), "n_tables": len(tables),
                                  "tables": names, "version": version}
    return report


def zenodo() -> dict:
    out = {}
    for name, q in ZENODO_QUERIES.items():
        url = f"https://zenodo.org/api/records?q={urllib.parse.quote(q, safe='')}&size=50&sort=mostrecent"
        data = get(url, label=f"zenodo {name}")
        if not data:
            continue
        (DATA / f"zenodo_{name}.json").write_bytes(data)
        try:
            j = json.loads(data)
            hits = j.get("hits", {}).get("hits", [])
            out[name] = [{"id": h.get("id"), "doi": h.get("doi"), "title": h.get("metadata", {}).get("title"),
                          "date": h.get("metadata", {}).get("publication_date"),
                          "files": [(f.get("key"), f.get("size")) for f in (h.get("files") or [])][:30]}
                         for h in hits]
        except Exception:  # noqa: BLE001
            out[name] = "unparsed"
    return out


def inspire() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    fields = "titles,arxiv_eprints,abstracts,authors.full_name,citation_count,earliest_date,dois"
    for name, q in INSPIRE_QUERIES.items():
        url = (f"https://inspirehep.net/api/literature?q={urllib.parse.quote(q, safe='')}"
               f"&size=100&fields={fields}&sort=mostrecent")
        data = get(url, label=f"inspire {name}")
        if not data:
            continue
        (META / f"inspire_{name}.json").write_bytes(data)
        rows = []
        try:
            j = json.loads(data)
            for h in j.get("hits", {}).get("hits", []):
                md = h.get("metadata", {})
                rows.append({
                    "inspire_id": h.get("id"),
                    "arxiv": [e.get("value") for e in md.get("arxiv_eprints", [])],
                    "title": (md.get("titles") or [{}])[0].get("title"),
                    "authors": [a.get("full_name") for a in md.get("authors", [])][:6],
                    "date": md.get("earliest_date"),
                    "citation_count": md.get("citation_count"),
                    "abstract": (md.get("abstracts") or [{}])[0].get("value"),
                })
        except Exception as exc:  # noqa: BLE001
            rows.append({"error": repr(exc)})
        out[name] = rows
    return out


def pages() -> dict:
    out = {}
    for fname, url in PAGES.items():
        data = get(url, label=f"page {fname}")
        if not data:
            out[fname] = "failed"
            continue
        if fname.endswith(".pdf"):
            tmp = OUT / "_tmp"
            tmp.mkdir(exist_ok=True)
            pdf = tmp / fname
            pdf.write_bytes(data)
            ok = pdf_to_text(pdf, TEXT / (fname[:-4] + ".txt"))
            pdf.unlink(missing_ok=True)
            out[fname] = "text" if ok else "pdf_unreadable"
        else:
            (META / fname).write_bytes(data)
            # every link on the page, so a data-release URL cannot hide in HTML
            links = sorted(set(re.findall(rb'href="([^"]+)"', data)))
            (META / (fname + ".links.txt")).write_text("\n".join(x.decode("utf-8", "ignore") for x in links))
            out[fname] = {"bytes": len(data), "n_links": len(links),
                          "data_like_links": [x.decode("utf-8", "ignore") for x in links
                                              if re.search(rb"data|release|zenodo|hepdata|\.csv|\.txt|\.root|\.h5|\.json", x, re.I)][:60]}
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    print(f"lzlit: out={OUT} dryrun={DRYRUN}")
    id_check: dict = {"by_id": {}, "by_title": {}, "n_id_mismatch": 0, "n_title_search_miss": 0}

    # 1. metadata for the asserted ids, title-checked
    ids = [v["id"] for v in BY_ID.values()]
    meta = arxiv_ids_meta(ids)
    papers: dict[str, dict] = {}
    for key, spec in BY_ID.items():
        ent = meta.get(spec["id"])
        ok = bool(ent) and norm(spec["title"]) in norm(ent["title"])
        id_check["by_id"][key] = {"id": spec["id"], "expected_fragment": spec["title"],
                                  "fetched_title": ent["title"] if ent else None, "match": ok}
        if not ok:
            id_check["n_id_mismatch"] += 1
        if ent:
            papers[spec["id"]] = dict(ent, source_key=key, want_src=spec["src"], id_title_match=ok)

    # 2. title searches
    for key, spec in BY_TITLE.items():
        hits = arxiv_search(f'ti:"{spec["title"]}"', f"title_{key}", max_results=5)
        best = None
        for h in hits:
            if norm(spec["title"])[:40] in norm(h["title"]):
                best = h
                break
        if best is None and hits:
            best = hits[0]
        found = bool(best) and norm(spec["title"])[:40] in norm(best["title"])
        id_check["by_title"][key] = {"expected_title": spec["title"],
                                     "resolved_id": best["id"] if best else None,
                                     "resolved_title": best["title"] if best else None,
                                     "match": found}
        if not found:
            id_check["n_title_search_miss"] += 1
        if best and found:
            papers.setdefault(best["id"], dict(best, source_key=key, want_src=spec["src"], id_title_match=True))

    # 3. keyword sweeps, newest first
    kw_hits: dict[str, list[dict]] = {}
    for name, q in KEYWORD.items():
        hits = arxiv_search(q, name, max_results=60, newest=True)
        kw_hits[name] = [{"id": h["id"], "title": h["title"], "published": h["published"]} for h in hits]
        for h in hits:
            if h["published"] >= "2026-08-25":
                papers.setdefault(h["id"], dict(h, source_key=f"kw:{name}", want_src=True, id_title_match=True))

    # 4. INSPIRE citations of the LZ paper: the complete follow-up list
    insp = inspire()
    for row in insp.get("refersto_main", []):
        for aid in row.get("arxiv") or []:
            if aid not in papers:
                papers[aid] = {"id": aid, "title": row.get("title"), "abstract": row.get("abstract"),
                               "published": row.get("date"), "authors": row.get("authors"),
                               "source_key": "inspire:refersto", "want_src": True, "id_title_match": True}
    # metadata for anything INSPIRE / keywords added that we have no atom entry for
    missing = [a for a, p in papers.items() if not p.get("updated")]
    if missing:
        for a, ent in arxiv_ids_meta(missing).items():
            papers[a].update(ent)

    followups = sorted(
        [p for p in papers.values() if (p.get("published") or "") >= "2026-08-25" and p["id"] != MAIN_ID],
        key=lambda p: p.get("published") or "", reverse=True)
    (OUT / "followups.json").write_text(json.dumps(
        [{k: p.get(k) for k in ("id", "title", "authors", "published", "updated", "abstract", "source_key")}
         for p in followups], indent=2))
    (OUT / "id_title_check.json").write_text(json.dumps(id_check, indent=2))
    (META / "keyword_hits.json").write_text(json.dumps(kw_hits, indent=2))
    (META / "inspire_parsed.json").write_text(json.dumps(insp, indent=2))
    _write_summary({"n_papers": len(papers), "n_followups": len(followups)})

    # 5. full text and sources -- the LZ paper and the follow-ups first
    order = sorted(papers.values(),
                   key=lambda p: (p["id"] != MAIN_ID, not str(p.get("source_key", "")).startswith(("fu_", "kw:", "inspire")),
                                  p.get("published") or ""))
    for p in order:
        aid = p["id"]
        p["text_ok"] = fetch_pdf_text(aid)
        p["src_ok"] = fetch_source(aid) if p.get("want_src") else None
        (OUT / "papers.json").write_text(json.dumps(papers, indent=2, sort_keys=True))

    # 6. data releases and pages
    report = {"hepdata": hepdata(), "zenodo": zenodo(), "pages": pages()}
    (DATA / "data_release_report.json").write_text(json.dumps(report, indent=2))
    shutil.rmtree(OUT / "_tmp", ignore_errors=True)
    _write_summary({"n_papers": len(papers), "n_followups": len(followups),
                    "n_text_ok": sum(1 for p in papers.values() if p.get("text_ok")),
                    "n_src_ok": sum(1 for p in papers.values() if p.get("src_ok")),
                    "hepdata_records": list(report["hepdata"].get("records", {}).keys())})
    print(f"lzlit: done in {time.time() - T0:.0f}s; {len(papers)} papers, {len(followups)} follow-ups")


if __name__ == "__main__":
    main()
