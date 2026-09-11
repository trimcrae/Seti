#!/usr/bin/env python3
"""GOO prior-art sweep and reference verification, run on the GitHub runner.

The sandbox blocks arxiv.org, Crossref and ADS (``CONNECT tunnel failed,
response 403``); the Actions runner has egress.  Two jobs:

  1. NOVELTY.  Does any paper compute how far / how fast an UNCONTROLLED
     self-replicator ("grey goo") propagates beyond its planet, search for its
     astronomical signature, or draw the implication for the civilisation
     asking?  Phrase queries against the arXiv API, verbatim atom feeds saved,
     a decoy-aware regex scan over every abstract fetched.

  2. VERIFICATION.  Every reference the manuscript (paper/goo/main.tex,
     paper/replicators/main.tex) cites is resolved from the record, never from
     memory: arXiv ids are title-checked, titles are searched, journal papers
     without an arXiv posting are resolved through the Crossref API by
     bibliographic query, and the fetched title is compared with the expected
     one.  Books, reports and pre-1991 journal papers that neither index
     carries are recorded as ASSERTED, which the manuscript's bibliography then
     marks.  Nothing is silently trusted: results/goolit/verification.json
     lists every reference with its status and the record's own title.

Outputs under results/goolit/:
  arxiv_q_<name>.atom          novelty phrase query (verbatim)
  arxiv_id_<key>.atom          per-reference id lookup (verbatim)
  arxiv_t_<key>.atom           per-reference title search (verbatim)
  crossref_<key>.json          per-reference Crossref bibliographic query (verbatim)
  concept_scan.json            novelty scan over every fetched abstract
  verification.json            per-reference status: VERIFIED / MISMATCH / NOT_FOUND / ASSERTED
  REPORT.md                    the human-readable summary of both
  summary.json                 HTTP status of every URL
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request

OUT = pathlib.Path(os.environ.get("GOOLIT_OUT", "results/goolit"))
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Seti-goolit/1.0 (mailto:trimcrae@gmail.com)"}
PAUSE = float(os.environ.get("GOOLIT_PAUSE", "3.0"))
TRIES = int(os.environ.get("GOOLIT_TRIES", "3"))
ARXIV_API = "http://export.arxiv.org/api/query"
CROSSREF = "https://api.crossref.org/works"
STATUS: list[dict] = []


def _write_summary() -> None:
    (OUT / "summary.json").write_text(json.dumps(
        {"n_urls": len(STATUS), "n_ok": sum(1 for s in STATUS if s["ok"]),
         "n_failed": sum(1 for s in STATUS if not s["ok"]), "status": STATUS}, indent=2))


def get(url: str, dest: pathlib.Path) -> bool:
    attempts: list[dict] = []
    for i in range(TRIES):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
                code = int(getattr(r, "status", 0) or 0)
            dest.write_bytes(data)
            attempts.append({"http_status": code, "bytes": len(data)})
            STATUS.append({"url": url, "dest": dest.name, "ok": True, "http_status": code,
                           "bytes": len(data), "attempts": attempts})
            print(f"  ok  HTTP {code}  {len(data):>8,}B  {dest.name}")
            _write_summary()
            time.sleep(PAUSE)
            return True
        except urllib.error.HTTPError as exc:
            attempts.append({"http_status": int(exc.code), "error": repr(exc)})
            print(f"  try {i + 1}/{TRIES} HTTP {exc.code}: {exc!r}")
            time.sleep(PAUSE * (i + 1))
        except Exception as exc:  # noqa: BLE001
            attempts.append({"http_status": None, "error": repr(exc)})
            print(f"  try {i + 1}/{TRIES} failed: {exc!r}")
            time.sleep(PAUSE * (i + 1))
    STATUS.append({"url": url, "dest": dest.name, "ok": False,
                   "http_status": attempts[-1]["http_status"] if attempts else None, "attempts": attempts})
    _write_summary()
    return False


# ----------------------------------------------------------------------------
# 1. NOVELTY queries.  Target phrases and decoys, fetched on purpose.
# ----------------------------------------------------------------------------
NOVELTY_QUERIES: dict[str, str] = {
    "grey_goo_all": 'all:"grey goo"',
    "gray_goo_all": 'all:"gray goo"',
    "ecophagy": 'all:ecophagy',
    "grey_goo_fermi": 'all:"grey goo" AND all:Fermi',
    "gray_goo_fermi": 'all:"gray goo" AND all:Fermi',
    "goo_technosignature": '(all:"grey goo" OR all:"gray goo") AND all:technosignature',
    "goo_seti": '(all:"grey goo" OR all:"gray goo") AND all:SETI',
    "goo_interstellar": '(all:"grey goo" OR all:"gray goo") AND all:interstellar',
    "goo_exoplanet": '(all:"grey goo" OR all:"gray goo") AND all:exoplanet',
    "runaway_replicat_galaxy": 'all:runaway AND all:replicat* AND all:galaxy AND all:Fermi',
    "uncontrolled_self_replicating": 'all:"uncontrolled" AND all:"self-replicating"',
    "self_replicating_fermi": 'all:"self-replicating" AND all:"Fermi paradox"',
    "von_neumann_probe_malfunction": 'all:"von Neumann probe" AND (all:malfunction OR all:mutation OR all:"error catastrophe")',
    "berserker_probes": 'all:berserker AND all:probe',
    "nanobots_dark_matter": 'all:nanobots AND all:"dark matter"',
    "replicator_technosignature": 'all:replicator AND all:technosignature',
    "self_replicating_technosignature": 'all:"self-replicating" AND all:technosignature',
    "self_destructive_civilisation_signature": 'all:"self-destructive" AND all:civilisation AND all:signature',
    "nanotechnology_fermi_paradox": 'all:nanotechnology AND all:"Fermi paradox"',
    "nanotechnology_great_filter": 'all:nanotechnology AND all:"great filter"',
    "replicator_anthropic_shadow": 'all:replicat* AND all:"anthropic shadow"',
    "dyson_swarm_collisional_cascade": 'all:"collisional cascade" AND all:(Dyson OR megaswarm)',
    "radiation_pressure_blowout_technosignature": 'all:"radiation pressure" AND all:blowout AND all:technosignature',
    "interstellar_dust_artificial": 'all:"interstellar dust" AND all:artificial AND all:(probe OR technosignature)',
    # decoys
    "decoy_von_neumann_probe": 'all:"von Neumann probe"',
    "decoy_self_replicating_probe": 'all:"self-replicating probe"',
    "decoy_galactic_colonization_timescale": 'all:"galactic colonization" AND all:timescale',
    "decoy_dyson_sphere_search": 'all:"Dyson sphere" AND all:search',
}

CONCEPT_PATTERNS = {
    "goo_named": r"gr[ae]y goo|ecophag",
    "uncontrolled_replication": r"(uncontrolled|runaway|out of control|escaped?)\s+(self-)?replicat",
    "propagation_beyond_planet": r"(interstellar|galactic|galaxy-wide|intergalactic)\s+(spread|propagat|dispers|expansion|colon)",
    "observational_signature": r"(technosignature|observational signature|detect(able|ion)|infrared excess|waste heat)",
    "own_risk_implication": r"(our own|humanity|human civili[sz]ation).{0,80}(risk|extinction|filter)",
}


# ----------------------------------------------------------------------------
# 2. REFERENCES.  key -> {mode, id/title/query, expect}.  'expect' is the title
#    the manuscript relies on; the record's title decides.
# ----------------------------------------------------------------------------
REFS: dict[str, dict] = {
    # --- expansion / probes ---
    "Hart1975": {"mode": "crossref", "q": "Hart 1975 An explanation for the absence of extraterrestrials on Earth Quarterly Journal Royal Astronomical Society", "expect": "An Explanation for the Absence of Extraterrestrials on Earth"},
    "Tipler1980": {"mode": "crossref", "q": "Tipler 1980 Extraterrestrial intelligent beings do not exist Quarterly Journal Royal Astronomical Society", "expect": "Extraterrestrial intelligent beings do not exist"},
    "Freitas1980": {"mode": "crossref", "q": "Freitas 1980 A self-reproducing interstellar probe Journal of the British Interplanetary Society", "expect": "A self-reproducing interstellar probe"},
    "SaganNewman1983": {"mode": "crossref", "q": "Sagan Newman 1983 The solipsist approach to extraterrestrial intelligence", "expect": "The solipsist approach to extraterrestrial intelligence"},
    "NewmanSagan1981": {"mode": "crossref", "q": "Newman Sagan 1981 Galactic civilizations: Population dynamics and interstellar diffusion Icarus", "expect": "Galactic civilizations: Population dynamics and interstellar diffusion"},
    "Jones1981": {"mode": "crossref", "q": "Jones 1981 Discrete calculations of interstellar migration and settlement Icarus", "expect": "Discrete calculations of interstellar migration and settlement"},
    "Landis1998": {"mode": "crossref", "q": "Landis 1998 The Fermi paradox: an approach based on percolation theory JBIS", "expect": "The Fermi paradox: an approach based on percolation theory"},
    "Bjork2007": {"mode": "arxiv_id", "id": "astro-ph/0701238", "expect": "Exploring the Galaxy using space probes"},
    "CottaMorales2009": {"mode": "arxiv_id", "id": "0907.0345", "expect": "A computational analysis of galactic exploration with space probes: implications for the Fermi paradox"},
    "Wiley2011": {"mode": "arxiv_id", "id": "1111.6131", "expect": "The Fermi Paradox, Self-Replicating Probes, and the Interstellar Transportation Bandwidth"},
    "NicholsonForgan2013": {"mode": "arxiv_id", "id": "1307.1648", "expect": "Slingshot Dynamics for Self Replicating Probes and the Effect on Exploration Timescales"},
    "ArmstrongSandberg2013": {"mode": "crossref", "q": "Armstrong Sandberg 2013 Eternity in six hours: Intergalactic spreading of intelligent life and sharpening the Fermi paradox Acta Astronautica", "expect": "Eternity in six hours: Intergalactic spreading of intelligent life and sharpening the Fermi paradox"},
    "Olson2015": {"mode": "arxiv_id", "id": "1411.4359", "expect": "Homogeneous cosmology with aggressively expanding civilizations"},
    "Olson2017": {"mode": "arxiv_id", "id": "1507.05969", "expect": "Estimates for the number of visible galaxy-spanning civilizations and the cosmological expansion of life"},
    "Hanson2021": {"mode": "arxiv_id", "id": "2102.01522", "expect": "If Loud Aliens Explain Human Earliness, Quiet Aliens Are Also Rare"},
    "CarrollNellenback2019": {"mode": "arxiv_id", "id": "1902.04450", "expect": "The Fermi Paradox and the Aurora Effect: Exo-civilization Settlement, Expansion and Steady States"},
    "Osmanov2019": {"mode": "arxiv_id", "id": "1909.05078", "expect": "On the interstellar Von Neumann micro self-reproducing probes"},
    "Hein2020": {"mode": "arxiv_id", "id": "2005.12303", "expect": "Near-Term Self-replicating Probes - A Concept Design"},
    "Kowald2015": {"mode": "arxiv_id", "id": "1605.02169", "expect": "Why is there no von Neumann probe on Ceres? Error catastrophe can explain the Fermi-Hart Paradox"},
    "Chen2022": {"mode": "arxiv_id", "id": "2209.14244", "expect": "Lotka-Volterra Models for Extraterrestrial Self-Replicating Probes"},
    "Ellery2025": {"mode": "arxiv_id", "id": "2510.00082", "expect": "Technosignatures of Self-Replicating Probes in the Solar System"},
    "Ellery2022a": {"mode": "crossref", "q": "Ellery 2022 Self-replicating probes are imminent implications for SETI International Journal of Astrobiology", "expect": "Self-replicating probes are imminent – implications for SETI"},
    "Ellery2022b": {"mode": "crossref", "q": "Ellery 2022 Curbing the fruitfulness of self-replicating machines International Journal of Astrobiology", "expect": "Curbing the fruitfulness of self-replicating machines"},
    "Stevens2016": {"mode": "arxiv_id", "id": "1507.08530", "expect": "Observational Signatures of Self-Destructive Civilisations"},
    "Cirkovic2004": {"mode": "arxiv_id", "id": "astro-ph/0309769", "expect": "Cosmic Irony: SETI Optimism from Catastrophes?"},
    "Cirkovic2018lotka": {"mode": "arxiv_id", "id": "1810.03088", "expect": "Fermi and Lotka: The Long Odds of Survival in a Dangerous Universe"},
    "Cirkovic2010": {"mode": "crossref", "q": "Cirkovic Sandberg Bostrom 2010 Anthropic shadow: observation selection effects and human extinction risks Risk Analysis", "expect": "Anthropic Shadow: Observation Selection Effects and Human Extinction Risks"},
    "Sandberg2018": {"mode": "arxiv_id", "id": "1806.02404", "expect": "Dissolving the Fermi Paradox"},
    "SnyderBeattie2019": {"mode": "crossref", "q": "Snyder-Beattie Ord Bonsall 2019 An upper bound for the background rate of human extinction Scientific Reports", "expect": "An upper bound for the background rate of human extinction"},
    "Kipping2020": {"mode": "arxiv_id", "id": "2005.09008", "expect": "An objective Bayesian analysis of life's early start and our late arrival"},
    "Olum2004": {"mode": "crossref", "q": "Olum 2004 Conflict between anthropic reasoning and observation Analysis", "expect": "Conflict between anthropic reasoning and observation"},
    # --- waste-heat / galaxy surveys ---
    "Wright2014a": {"mode": "arxiv_id", "id": "1408.1133", "expect": "The G Infrared Search for Extraterrestrial Civilizations with Large Energy Supplies. I. Background and Justification"},
    "Wright2014b": {"mode": "arxiv_id", "id": "1408.1134", "expect": "The G Infrared Search for Extraterrestrial Civilizations with Large Energy Supplies. II. Framework, Strategy, and First Result"},
    "Griffith2015": {"mode": "arxiv_id", "id": "1504.03418", "expect": "The G Infrared Search for Extraterrestrial Civilizations with Large Energy Supplies. III. The Reddest Extended Sources in WISE"},
    "Garrett2015": {"mode": "arxiv_id", "id": "1508.02624", "expect": "Application of the mid-IR radio correlation to the G sample and the search for advanced extraterrestrial civilisations"},
    "Zackrisson2015": {"mode": "arxiv_id", "id": "1508.02406", "expect": "Extragalactic SETI: The Tully-Fisher relation as a probe of Dysonian astroengineering in disk galaxies"},
    "Carrigan2009": {"mode": "arxiv_id", "id": "0811.2376", "expect": "IRAS-based Whole-Sky Upper Limit on Dyson Spheres"},
    "Lacki2016": {"mode": "arxiv_id", "id": "1604.07844", "expect": "Type III Societies (Apparently) Do Not Exist"},
    "Suazo2022": {"mode": "arxiv_id", "id": "2201.11123", "expect": "Project Hephaistos I. Upper limits on partial Dyson spheres in the Milky Way"},
    "Suazo2024": {"mode": "arxiv_id", "id": "2405.02927", "expect": "Project Hephaistos II. Dyson sphere candidates from Gaia DR3, 2MASS, and WISE"},
    "Ren2024": {"mode": "arxiv_id", "id": "2405.14921", "expect": "Background Contamination of the Project Hephaistos Dyson Spheres Candidates"},
    "Zackrisson2026": {"mode": "arxiv_id", "id": "2607.09460", "expect": "Project Hephaistos IV. James Webb Space Telescope Observations of Two Dyson Sphere Candidates"},
    "Lacki2025": {"mode": "arxiv_id", "id": "2504.21151", "expect": "Ground to Dust: Collisional Cascades and the Fate of Kardashev II Megaswarms"},
    "Lacki2026": {"mode": "arxiv_id", "id": "2606.08373", "expect": "Dust to Dust: Prospects for Passive Technosignatures as Relics of ETI"},
    # --- debris discs ---
    "Wyatt2007": {"mode": "arxiv_id", "id": "astro-ph/0610102", "expect": "Transience of hot dust around sun-like stars"},
    "Wyatt2008": {"mode": "crossref", "q": "Wyatt 2008 Evolution of Debris Disks Annual Review of Astronomy and Astrophysics", "expect": "Evolution of Debris Disks"},
    "Trilling2008": {"mode": "arxiv_id", "id": "0710.5498", "expect": "Debris disks around Sun-like stars"},
    "Eiroa2013": {"mode": "arxiv_id", "id": "1305.0155", "expect": "DUst around NEarby Stars. The survey observational results"},
    "Sibthorpe2018": {"mode": "arxiv_id", "id": "1803.00072", "expect": "Analysis of the Herschel DEBRIS Sun-like star sample"},
    "KennedyWyatt2013": {"mode": "arxiv_id", "id": "1305.6607", "expect": "The bright end of the exo-Zodi luminosity function: disc evolution and implications for exo-Earth detectability"},
    "KennedyWyatt2012": {"mode": "arxiv_id", "id": "1207.0521", "expect": "Confusion limited surveys: using WISE to quantify the rarity of warm dust around Kepler stars"},
    "Moor2021": {"mode": "arxiv_id", "id": "2103.00568", "expect": "A new sample of warm extreme debris disks from the ALLWISE catalog"},
    "Balog2009": {"mode": "arxiv_title", "title": "Spitzer/IRAC-MIPS Survey of NGC 2451A and B: Debris Disks at 50-80 Million years", "expect": "Spitzer/IRAC-MIPS Survey of NGC 2451A and B"},
    "Meng2015": {"mode": "arxiv_title", "title": "Planetary Collisions outside the Solar System: Time Domain Characterization of Extreme Debris Disks", "expect": "Planetary Collisions outside the Solar System"},
    # --- dust dynamics / ISM ---
    "Burns1979": {"mode": "crossref", "q": "Burns Lamy Soter 1979 Radiation forces on small particles in the solar system Icarus", "expect": "Radiation forces on small particles in the solar system"},
    "Grun1993": {"mode": "crossref", "q": "Grun 1993 Discovery of Jovian dust streams and interstellar grains by the Ulysses spacecraft Nature", "expect": "Discovery of Jovian dust streams and interstellar grains by the Ulysses spacecraft"},
    "Landgraf2000": {"mode": "crossref", "q": "Landgraf 2000 Modeling the motion and distribution of interstellar dust inside the heliosphere Journal of Geophysical Research", "expect": "Modeling the motion and distribution of interstellar dust inside the heliosphere"},
    "Kruger2015": {"mode": "arxiv_title", "title": "Sixteen years of Ulysses interstellar dust measurements in the solar system. III. Simulations and data unveil new insights into local interstellar dust", "expect": "Sixteen years of Ulysses interstellar dust measurements"},
    "Sterken2012": {"mode": "arxiv_title", "title": "The flow of interstellar dust into the solar system", "expect": "The flow of interstellar dust into the solar system"},
    "Jones1996": {"mode": "crossref", "q": "Jones Tielens Hollenbach 1996 Grain shattering in shocks: the interstellar grain size distribution Astrophysical Journal", "expect": "Grain Shattering in Shocks: The Interstellar Grain Size Distribution"},
    "Slavin2015": {"mode": "arxiv_id", "id": "1502.00929", "expect": "Destruction of Interstellar Dust in Evolving Supernova Remnant Shock Waves"},
    "Mileikowsky2000": {"mode": "crossref", "q": "Mileikowsky 2000 Natural transfer of viable microbes in space 1. From Mars to Earth and Earth to Mars Icarus", "expect": "Natural Transfer of Viable Microbes in Space"},
    "Melosh2003": {"mode": "crossref", "q": "Melosh 2003 Exchange of meteorites (and impact ejecta) between stellar systems Astrobiology", "expect": "Exchange of Meteorites (and Impact Ejecta) between Stellar Systems"},
    "AdamsSpergel2005": {"mode": "arxiv_id", "id": "astro-ph/0504648", "expect": "Lithopanspermia in Star Forming Clusters"},
    "Worth2013": {"mode": "arxiv_title", "title": "Seeding Life on the Moons of the Outer Planets via Lithopanspermia", "expect": "Seeding Life on the Moons of the Outer Planets via Lithopanspermia"},
    "Ginsburg2018": {"mode": "arxiv_id", "id": "1810.04307", "expect": "Galactic Panspermia"},
    "LingamLoeb2017": {"mode": "arxiv_id", "id": "1703.00878", "expect": "Enhanced interplanetary panspermia in the TRAPPIST-1 system"},
    # --- galactic dynamics ---
    "SellwoodBinney2002": {"mode": "arxiv_id", "id": "astro-ph/0203510", "expect": "Radial mixing in galactic discs"},
    "Frankel2018": {"mode": "arxiv_id", "id": "1805.09198", "expect": "Measuring radial orbit migration in the Galactic disk"},
    "BlandHawthorn2016": {"mode": "arxiv_id", "id": "1602.07702", "expect": "The Galaxy in Context: Structural, Kinematic and Integrated Properties"},
    "DehnenBinney1998": {"mode": "arxiv_id", "id": "astro-ph/9710077", "expect": "Local stellar kinematics from Hipparcos data"},
    # --- planet-scale inputs ---
    "PhoenixDrexler2004": {"mode": "crossref", "q": "Phoenix Drexler 2004 Safe exponential manufacturing Nanotechnology", "expect": "Safe exponential manufacturing"},
    "BarOn2018": {"mode": "crossref", "q": "Bar-On Phillips Milo 2018 The biomass distribution on Earth PNAS", "expect": "The biomass distribution on Earth"},
    "RudnickGao2003": {"mode": "crossref", "q": "Rudnick Gao 2003 Composition of the continental crust Treatise on Geochemistry", "expect": "Composition of the Continental Crust"},
    "Kreidberg2019": {"mode": "arxiv_id", "id": "1908.06834", "expect": "Absence of a thick atmosphere on the terrestrial exoplanet LHS 3844b"},
    "Greene2023": {"mode": "arxiv_id", "id": "2303.14849", "expect": "Thermal Emission from the Earth-sized Exoplanet TRAPPIST-1 b using JWST"},
    "Zieba2023": {"mode": "arxiv_id", "id": "2306.10150", "expect": "No thick carbon dioxide atmosphere on the rocky exoplanet TRAPPIST-1 c"},
    "Eigen1971": {"mode": "crossref", "q": "Eigen 1971 Selforganization of matter and the evolution of biological macromolecules Naturwissenschaften", "expect": "Selforganization of matter and the evolution of biological macromolecules"},
    # --- other replicators (companion paper) ---
    "Adamala2024": {"mode": "crossref", "q": "Adamala 2024 Confronting risks of mirror life Science", "expect": "Confronting risks of mirror life"},
    "Millett2017": {"mode": "crossref", "q": "Millett Snyder-Beattie 2017 Existential risk and cost-effective biosecurity Health Security", "expect": "Existential Risk and Cost-Effective Biosecurity"},
    "Noble2018": {"mode": "crossref", "q": "Noble 2018 Current CRISPR gene drive systems are likely to be highly invasive in wild populations eLife", "expect": "Current CRISPR gene drive systems are likely to be highly invasive in wild populations"},
    "Esvelt2014": {"mode": "crossref", "q": "Esvelt Smidler Catteruccia Church 2014 Concerning RNA-guided gene drives for the alteration of wild populations eLife", "expect": "Concerning RNA-guided gene drives for the alteration of wild populations"},
    "Kriegman2021": {"mode": "crossref", "q": "Kriegman Blackiston Levin Bongard 2021 Kinematic self-replication in reconfigurable organisms PNAS", "expect": "Kinematic self-replication in reconfigurable organisms"},
    "Moore2003": {"mode": "crossref", "q": "Moore Paxson Savage Shannon Staniford Weaver 2003 Inside the Slammer worm IEEE Security Privacy", "expect": "Inside the Slammer worm"},
    "Spafford1989": {"mode": "crossref", "q": "Spafford 1989 The internet worm program: an analysis Computer Communication Review", "expect": "The internet worm program: an analysis"},
    "Pan2024": {"mode": "arxiv_id", "id": "2412.12140", "expect": "Frontier AI systems have surpassed the self-replicating red line"},
    "Bengio2024": {"mode": "crossref", "q": "Bengio 2024 Managing extreme AI risks amid rapid progress Science", "expect": "Managing extreme AI risks amid rapid progress"},
    "Zykov2005": {"mode": "crossref", "q": "Zykov Mytilinaios Adams Lipson 2005 Self-reproducing machines Nature", "expect": "Self-reproducing machines"},
    "Moses2020": {"mode": "crossref", "q": "Moses Chirikjian 2020 Robotic self-replication Annual Review of Control Robotics and Autonomous Systems", "expect": "Robotic Self-Replication"},
    "Metzger2013": {"mode": "crossref", "q": "Metzger Muscatello Mueller Mantovani 2013 Affordable, rapid bootstrapping of the space industry and solar system civilization Journal of Aerospace Engineering", "expect": "Affordable, Rapid Bootstrapping of the Space Industry and Solar System Civilization"},
    "Hastings2005": {"mode": "crossref", "q": "Hastings 2005 The spatial spread of invasions: new developments in theory and evidence Ecology Letters", "expect": "The spatial spread of invasions: new developments in theory and evidence"},
    "Umbrello2018": {"mode": "crossref", "q": "Umbrello Baum 2018 Evaluating future nanotechnology: The net societal impacts of atomically precise manufacturing Futures", "expect": "Evaluating future nanotechnology: The net societal impacts of atomically precise manufacturing"},
    "Bostrom2002": {"mode": "crossref", "q": "Bostrom 2002 Existential risks: analyzing human extinction scenarios and related hazards Journal of Evolution and Technology", "expect": "Existential Risks: Analyzing Human Extinction Scenarios and Related Hazards"},
    "Joy2000": {"mode": "assert", "citation": "Joy, B. 2000, Wired 8.04, 'Why the future doesn't need us'"},
    "Drexler1986": {"mode": "assert", "citation": "Drexler, K. E. 1986, Engines of Creation: The Coming Era of Nanotechnology (Anchor/Doubleday)"},
    "Freitas2000": {"mode": "assert", "citation": "Freitas, R. A. 2000, 'Some limits to global ecophagy by biovorous nanoreplicators, with public policy recommendations', Foresight Institute"},
    "FreitasMerkle2004": {"mode": "assert", "citation": "Freitas, R. A. & Merkle, R. C. 2004, Kinematic Self-Replicating Machines (Landes Bioscience)"},
    "Freitas1980b": {"mode": "assert", "citation": "Freitas, R. A. & Gilbreath, W. P. (eds.) 1982, Advanced Automation for Space Missions, NASA CP-2255"},
    "Rees2003": {"mode": "assert", "citation": "Rees, M. 2003, Our Final Hour (Basic Books)"},
    "Ord2020": {"mode": "assert", "citation": "Ord, T. 2020, The Precipice: Existential Risk and the Future of Humanity (Hachette)"},
    "Sandberg2008": {"mode": "assert", "citation": "Sandberg, A. & Bostrom, N. 2008, Global Catastrophic Risks Survey, FHI Technical Report 2008-1"},
    "Hanson1998": {"mode": "assert", "citation": "Hanson, R. 1998, 'The Great Filter: are we almost past it?', online essay"},
    "Bostrom2008": {"mode": "assert", "citation": "Bostrom, N. 2008, 'Where are they? Why I hope the search for extraterrestrial life finds nothing', MIT Technology Review, May/June"},
    "Bostrom2014": {"mode": "assert", "citation": "Bostrom, N. 2014, Superintelligence: Paths, Dangers, Strategies (OUP)"},
    "Staniford2002": {"mode": "assert", "citation": "Staniford, S., Paxson, V. & Weaver, N. 2002, 'How to 0wn the Internet in your spare time', USENIX Security Symposium"},
    "Esvelt2022": {"mode": "assert", "citation": "Esvelt, K. M. 2022, Delay, Detect, Defend: Preparing for a Future in which Thousands Can Release New Pandemics, Geneva Centre for Security Policy"},
    "RoyalSociety2004": {"mode": "assert", "citation": "Royal Society & Royal Academy of Engineering 2004, Nanoscience and nanotechnologies: opportunities and uncertainties"},
    "Cirkovic2018": {"mode": "assert", "citation": "Cirkovic, M. M. 2018, The Great Silence: The Science and Philosophy of Fermi's Paradox (OUP)"},
    "Omohundro2008": {"mode": "assert", "citation": "Omohundro, S. 2008, 'The basic AI drives', in Artificial General Intelligence 2008 (IOS Press)"},
}


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\$[^$]*\$", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> set[str]:
    stop = {"the", "of", "a", "an", "and", "in", "on", "for", "to", "with", "by", "its", "from", "at", "is", "are", "as", "vs"}
    return {t for t in _norm(s).split() if t not in stop and len(t) > 2}


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def _atom_entries(text: str) -> list[dict]:
    out = []
    for m in re.finditer(r"<entry>(.*?)</entry>", text, re.S):
        e = m.group(1)
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        i = re.search(r"<id>(.*?)</id>", e, re.S)
        s = re.search(r"<summary>(.*?)</summary>", e, re.S)
        authors = re.findall(r"<name>(.*?)</name>", e, re.S)
        pub = re.search(r"<published>(.*?)</published>", e, re.S)
        jr = re.search(r"<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>", e, re.S)
        doi = re.search(r"<arxiv:doi[^>]*>(.*?)</arxiv:doi>", e, re.S)
        out.append({"title": re.sub(r"\s+", " ", t.group(1)).strip() if t else "",
                    "id": i.group(1).strip() if i else "",
                    "summary": re.sub(r"\s+", " ", s.group(1)).strip() if s else "",
                    "authors": authors, "published": pub.group(1) if pub else "",
                    "journal_ref": jr.group(1).strip() if jr else "",
                    "doi": doi.group(1).strip() if doi else ""})
    return out


def verify_ref(key: str, spec: dict) -> dict:
    mode = spec["mode"]
    rec: dict = {"key": key, "mode": mode, "expect": spec.get("expect", ""), "status": "NOT_FOUND"}
    if mode == "assert":
        rec.update({"status": "ASSERTED", "citation": spec["citation"],
                    "note": "book, report or pre-arXiv paper not carried by the arXiv or Crossref APIs; asserted by the manuscript"})
        return rec
    if mode == "arxiv_id":
        dest = OUT / f"arxiv_id_{key}.atom"
        url = f"{ARXIV_API}?id_list={urllib.parse.quote(spec['id'])}&max_results=1"
        if get(url, dest):
            ents = _atom_entries(dest.read_text("utf-8", "ignore"))
            if ents and ents[0]["title"]:
                e = ents[0]
                ov = _overlap(spec["expect"], e["title"])
                rec.update({"arxiv_id": spec["id"], "record_title": e["title"], "authors": e["authors"],
                            "published": e["published"], "journal_ref": e["journal_ref"], "doi": e["doi"],
                            "overlap": round(ov, 2), "status": "VERIFIED" if ov >= 0.6 else "MISMATCH"})
        return rec
    if mode == "arxiv_title":
        dest = OUT / f"arxiv_t_{key}.atom"
        q = 'ti:"' + spec["title"] + '"'
        url = f"{ARXIV_API}?search_query={urllib.parse.quote(q)}&max_results=5"
        if get(url, dest):
            ents = _atom_entries(dest.read_text("utf-8", "ignore"))
            best = max(ents, key=lambda e: _overlap(spec["expect"], e["title"]), default=None)
            if best:
                ov = _overlap(spec["expect"], best["title"])
                rec.update({"arxiv_id": best["id"], "record_title": best["title"], "authors": best["authors"],
                            "published": best["published"], "journal_ref": best["journal_ref"], "doi": best["doi"],
                            "overlap": round(ov, 2), "status": "VERIFIED" if ov >= 0.6 else "MISMATCH"})
        return rec
    if mode == "crossref":
        dest = OUT / f"crossref_{key}.json"
        url = (f"{CROSSREF}?query.bibliographic={urllib.parse.quote(spec['q'])}&rows=5"
               "&select=DOI,title,author,issued,container-title,volume,page,type")
        if get(url, dest):
            try:
                items = json.loads(dest.read_text("utf-8", "ignore"))["message"]["items"]
            except Exception:  # noqa: BLE001
                items = []
            best, bov = None, 0.0
            for it in items:
                t = (it.get("title") or [""])[0]
                ov = _overlap(spec["expect"], t)
                if ov > bov:
                    best, bov = it, ov
            if best:
                auth = [f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in best.get("author", [])]
                year = None
                try:
                    year = best["issued"]["date-parts"][0][0]
                except Exception:  # noqa: BLE001
                    pass
                rec.update({"doi": best.get("DOI"), "record_title": (best.get("title") or [""])[0],
                            "authors": auth, "year": year, "journal": (best.get("container-title") or [""])[0],
                            "volume": best.get("volume"), "page": best.get("page"),
                            "overlap": round(bov, 2), "status": "VERIFIED" if bov >= 0.6 else "MISMATCH"})
        return rec
    return rec


def novelty_scan() -> dict:
    records = []
    seen = set()
    for p in sorted(OUT.glob("arxiv_q_*.atom")):
        qname = p.stem[len("arxiv_q_"):]
        for e in _atom_entries(p.read_text("utf-8", "ignore")):
            aid = e["id"].rsplit("/", 1)[-1]
            text = e["title"] + " " + e["summary"]
            hits = {k: bool(re.search(pat, text, re.I)) for k, pat in CONCEPT_PATTERNS.items()}
            rec = {"query": qname, "arxiv": aid, "title": e["title"], "published": e["published"][:10],
                   "authors": e["authors"][:4], "hits": hits, "n_hits": sum(hits.values())}
            records.append(rec)
            seen.add(aid)
    full = [r for r in records if r["hits"]["goo_named"] and r["hits"]["propagation_beyond_planet"]]
    goo_astro = [r for r in records if r["hits"]["goo_named"] and (r["hits"]["observational_signature"] or r["hits"]["propagation_beyond_planet"])]
    out = {"n_records": len(records), "n_unique_arxiv_ids": len(seen),
           "per_query_counts": {q: sum(1 for r in records if r["query"] == q) for q in NOVELTY_QUERIES},
           "goo_named_anywhere": sorted({r["arxiv"] for r in records if r["hits"]["goo_named"]}),
           "goo_named_and_beyond_planet": [r for r in full],
           "goo_named_and_astronomical": [r for r in goo_astro],
           "records": records}
    (OUT / "concept_scan.json").write_text(json.dumps(out, indent=1))
    return out


def report(ver: list[dict], scan: dict) -> None:
    L = ["# GOO literature sweep: novelty and reference verification", "",
         f"URLs fetched: {sum(1 for s in STATUS if s['ok'])} ok / {len(STATUS)} total.", "",
         "## Novelty", "",
         f"{scan['n_records']} abstracts ({scan['n_unique_arxiv_ids']} unique arXiv ids) scanned across {len(NOVELTY_QUERIES)} phrase queries.", "",
         "| query | entries |", "|---|---|"]
    for q, n in scan["per_query_counts"].items():
        L.append(f"| `{NOVELTY_QUERIES[q]}` | {n} |")
    L += ["", f"Abstracts naming grey/gray goo or ecophagy: {len(scan['goo_named_anywhere'])}", "",
          "Abstracts naming goo AND interstellar/galactic propagation (the paper's question):", ""]
    if scan["goo_named_and_beyond_planet"]:
        for r in scan["goo_named_and_beyond_planet"]:
            L.append(f"- {r['arxiv']} ({r['published']}) {r['title']} — {', '.join(r['authors'])}")
    else:
        L.append("- none")
    L += ["", "Abstracts naming goo AND an observational signature or propagation:", ""]
    if scan["goo_named_and_astronomical"]:
        for r in scan["goo_named_and_astronomical"]:
            L.append(f"- {r['arxiv']} ({r['published']}) {r['title']} — {', '.join(r['authors'])}")
    else:
        L.append("- none")
    L += ["", "## Reference verification", "",
          "| key | status | record title | authors / venue |", "|---|---|---|---|"]
    for r in ver:
        who = ", ".join(r.get("authors", [])[:3]) if r.get("authors") else r.get("citation", "")
        ven = r.get("journal_ref") or r.get("journal") or ""
        if r.get("year"):
            ven = f"{ven} {r.get('volume') or ''} {r.get('page') or ''} ({r['year']})"
        L.append(f"| {r['key']} | {r['status']} | {r.get('record_title', '')} | {who} {ven} |")
    n = {s: sum(1 for r in ver if r["status"] == s) for s in ("VERIFIED", "MISMATCH", "NOT_FOUND", "ASSERTED")}
    L += ["", f"Totals: {n}", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")


def main() -> None:
    print("== novelty queries ==")
    for name, q in NOVELTY_QUERIES.items():
        url = f"{ARXIV_API}?search_query={urllib.parse.quote(q)}&max_results=60"
        get(url, OUT / f"arxiv_q_{name}.atom")
    scan = novelty_scan()
    print("== reference verification ==")
    ver = []
    for key, spec in REFS.items():
        ver.append(verify_ref(key, spec))
        (OUT / "verification.json").write_text(json.dumps(
            {"n": len(ver), "counts": {s: sum(1 for r in ver if r["status"] == s)
                                      for s in ("VERIFIED", "MISMATCH", "NOT_FOUND", "ASSERTED")},
             "refs": ver}, indent=1))
    report(ver, scan)
    print(json.dumps({s: sum(1 for r in ver if r["status"] == s) for s in ("VERIFIED", "MISMATCH", "NOT_FOUND", "ASSERTED")}))


if __name__ == "__main__":
    main()
