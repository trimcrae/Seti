#!/usr/bin/env python3
"""NECROFRONTIER data-source probe: do the archives the new signatures need exist,
answer, and carry the column or product each signature reads?

Runner-only (every endpoint below is egress-blocked in the sandbox; CLAUDE.md
acquisition pattern).  ``docs/necrofrontier.md`` proposes signatures S46-S6x on
data sources this repository has never touched --- SPHEREx, DASCH DR7, the
Presolar Grain Database, PEWDD, the Sedimentary Geochemistry and
Paleoenvironments Project, CASSIS, Diviner PSR maps, CDMS/JPL line lists,
interferometric exozodi tables, the Kepler/TESS depth tables, the Cassini CDA
archive.  A shortlist of "data sources" is worth nothing until each one has
been asked, from a machine with egress, whether it (a) answers, (b) serves the
specific product the signature needs, and (c) does so in bulk without a human
in the loop.  This script asks and commits the answer verbatim.

Nothing here is a measurement of the sky.  The verdict per source is one of

  REACHED_WITH_PRODUCT   the endpoint answered AND the expected token / table /
                         column was present in what it served
  REACHED_NO_PRODUCT     the endpoint answered but the expected product was not
                         found in the response (wrong URL, product not yet
                         public, schema differs from the brief) --- a finding,
                         not a failure: the brief must be corrected
  NOT_REACHED            no HTTP answer (timeout, DNS, TLS, 4xx/5xx)
  NOT_PROBED             skipped (e.g. needs a token the runner does not hold)

Outputs under ``results/necrofrontier/``:
  probe.json    every endpoint: URL, status, bytes, head, expected-token hits,
                verdict, the signature(s) that depend on it
  PROBE.md      one line per source, for a human

The pure functions (``classify``, ``summarise``) are offline-tested in
``tests/test_necrofrontier.py``; the network is touched only in ``main``.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import sys

UA = {"User-Agent": "Seti-necrofrontier/1.0 (mailto:trimcrae@gmail.com)"}

# --------------------------------------------------------------------------
# The endpoint table.  Each REST entry: name -> dict(url, expect=[regex...],
# signatures=[...], why).  ``expect`` is a list of regexes at least ONE of which
# must match the response body for REACHED_WITH_PRODUCT.  TAP entries carry a
# service URL and an ADQL string; their ``expect`` regexes are matched against
# the JSON-serialised rows.
# --------------------------------------------------------------------------
REST: dict[str, dict] = {
    # ---- S46 ISOTOPE: the Presolar Grain Database (isotopic purity) ----
    "pgd_landing": {
        "url": "https://presolar.physics.wustl.edu/presolar-grain-database/",
        "expect": [r"(?i)silicon\s+carbide", r"(?i)download", r"(?i)\.xlsx|\.csv|\.zip"],
        "signatures": ["S46"],
        "why": "Landing page of the Presolar Grain Database (Stephan+2024, 20,230 SiC "
               "grains). The signature needs the per-grain delta29Si/delta30Si, 12C/13C, "
               "14N/15N, 26Al/27Al, 44Ti/48Ti columns as a bulk download.",
    },
    "pgd_wustl_root_http": {
        "url": "http://presolar.physics.wustl.edu/",
        "expect": [r"(?i)presolar"],
        "signatures": ["S46"],
        "why": "Run 1: the https landing page timed out at connect.  Plain-http root of the "
               "same host, to separate a TLS/port problem from a dead host.",
    },
    "pgd_ecl_search": {
        "url": "https://ecl.earthchem.org/home.php",
        "expect": [r"(?i)earthchem", r"(?i)library"],
        "signatures": ["S46"],
        "why": "EarthChem Library, where the 2024 SiC release is DOI'd (the DOI asserted in "
               "run 1 was wrong: 'DOI Not Found').  The record must be found by title "
               "search here, not guessed.",
    },
    "pgd_ads_record": {
        "url": "https://ui.adsabs.harvard.edu/abs/2024ieda.data...96S/abstract",
        "expect": [r"(?i)presolar", r"(?i)doi"],
        "signatures": ["S46"],
        "why": "The ADS record of the dataset release carries the real DOI.",
    },
    # ---- S47 FORGE: hot exozodi as a hot swarm ----
    "jmmc_oidb": {
        "url": "http://oidb.jmmc.fr/",
        "expect": [r"(?i)oidb", r"(?i)tap|search"],
        "signatures": ["S47"],
        "why": "JMMC OiDB: the optical-interferometry data base (PIONIER, FLUOR, MATISSE "
               "OIFITS).  The swarm test needs per-star visibility deficits in H/K/L.",
    },
    # ---- S48 SPARK: SPHEREx single-band stellar excess (industrial NIR line) ----
    "spherex_irsa_docs": {
        "url": "https://irsa.ipac.caltech.edu/data/SPHEREx/docs/",
        "expect": [r"(?i)spherex", r"(?i)quick\s*release|QR|spectral\s+image"],
        "signatures": ["S48", "S49"],
        "why": "IRSA SPHEREx holdings page: are Quick Release spectral images (and any "
               "per-source spectra) public, and under what product names?",
    },
    "spherex_ibe": {
        "url": "https://irsa.ipac.caltech.edu/ibe/data/spherex/",
        "expect": [r"(?i)qr|quick|spherex"],
        "signatures": ["S48", "S49"],
        "why": "IRSA IBE directory listing for SPHEREx image products (bulk access path).",
    },
    # ---- S50 CENTURY: DASCH DR7 ----
    "dasch_home": {
        "url": "https://dasch.cfa.harvard.edu/",
        "expect": [r"(?i)dasch", r"(?i)dr\s*7|data\s+release", r"(?i)starglass|daschlab|api"],
        "signatures": ["S50"],
        "why": "DASCH (Harvard plates 1885-1992): century-baseline light curves for the "
               "cessation / decay / fade signatures.  Needs the DR7 API.",
    },
    "dasch_dr7_web_apis": {
        "url": "https://dasch.cfa.harvard.edu/dr7/web-apis/",
        "expect": [r"(?i)querycat", r"(?i)lightcurve"],
        "signatures": ["S50"],
        "why": "DASCH DR7 web-API documentation: the querycat cone search and the "
               "POST lightcurve endpoint that daschlab wraps.",
    },
    "dasch_dr7_columns": {
        "url": "https://dasch.cfa.harvard.edu/dr7/lightcurve-columns/",
        "expect": [r"(?i)lim_mag|limiting", r"(?i)magcal|mag"],
        "signatures": ["S50"],
        "why": "Light-curve column documentation (per-exposure limiting magnitude, "
               "blending / plate-quality flags the cessation tests must read).",
    },
    "daschlab_pypi": {
        "url": "https://pypi.org/pypi/daschlab/json",
        "expect": [r"(?i)\"name\":\s*\"daschlab\"", r"(?i)version"],
        "signatures": ["S50"],
        "why": "The DASCH DR7 Python client exists on PyPI (its source carries the real "
               "API paths).",
    },
    # ---- S51 SLAG-WD: polluted white dwarf abundance vectors ----
    "pewdd_github": {
        "url": "https://api.github.com/repos/jamietwilliams/PEWDD",
        "expect": [r"(?i)pewdd|white\s+dwarf"],
        "signatures": ["S51"],
        "why": "Planetary Enriched White Dwarf Database (Williams+2024, A&A 691, A352) --- "
               "repository named in Huang+2026's data-availability statement.",
    },
    "pyllutedwd_github": {
        "url": "https://api.github.com/repos/andrewmbuchan4/PyllutedWD_Public",
        "expect": [r"(?i)pylluted|white\s+dwarf|\"name\""],
        "signatures": ["S51"],
        "why": "PyllutedWD (Harrison/Buchan forward model of natural pollution) --- the "
               "natural-process family the misfit test is calibrated against.",
    },
    # ---- S52 CRADLE: extreme debris disks around mature stars ----
    "cassis_atlas": {
        "url": "https://cassis.sirtf.com/atlas/",
        "expect": [r"(?i)cassis", r"(?i)IRS|spitzer"],
        "signatures": ["S52", "S53"],
        "why": "CASSIS: every Spitzer/IRS low-resolution spectrum, the mineralogy corpus "
               "for silica-vs-silicate-vs-metal decomposition of warm debris.  Run 1: read "
               "timeout at 90 s; the root is probed below with a longer budget.",
    },
    "cassis_root": {
        "url": "https://cassis.sirtf.com/",
        "expect": [r"(?i)cassis"],
        "signatures": ["S52", "S53"],
        "why": "CASSIS site root (run 1 timed out on /atlas/).",
    },
    "irsa_seip_docs": {
        "url": "https://irsa.ipac.caltech.edu/data/SPITZER/Enhanced/SEIP/",
        "expect": [r"(?i)spitzer", r"(?i)enhanced"],
        "signatures": ["S52", "S53"],
        "why": "IRSA Spitzer Enhanced Imaging Products page; the IRS Enhanced Products "
               "(extracted low-res spectra) are the IRSA-side mirror of the CASSIS corpus.",
    },
    # ---- S54 ULINE: artificial molecules in the ISM ----
    "cdms_entries": {
        "url": "https://cdms.astro.uni-koeln.de/classic/entries/",
        "expect": [r"(?i)CDMS"] + ['\\bCH3F\\b', '\\bCHF3\\b', '\\bNF3\\b', '\\bCOF2\\b', '\\bSO2F2\\b', '\\bCH2F2\\b', '\\bCF3CN\\b', '\\bCF3Cl\\b', '\\bCF2Cl2\\b', '\\bCFCl3\\b', '\\bCHClF2\\b', '\\bCF2\\b', '\\bCH3Cl\\b'],
        "signatures": ["S54"],
        "why": "Cologne Database for Molecular Spectroscopy: which industrial species "
               "(NF3, CHF3, CH3F, CH2F2, CF3Cl, CF2Cl2, COF2, SO2F2, CF3CN) have "
               "rotational line lists.",
    },
    "jpl_catdir": {
        "url": "https://spec.jpl.nasa.gov/ftp/pub/catalog/catdir.cat",
        "expect": ['\\bCH3F\\b', '\\bCHF3\\b', '\\bNF3\\b', '\\bCOF2\\b', '\\bSO2F2\\b', '\\bCH2F2\\b', '\\bCF3CN\\b', '\\bCF3Cl\\b', '\\bCF2Cl2\\b', '\\bCFCl3\\b', '\\bCHClF2\\b', '\\bCF2\\b', '\\bCH3Cl\\b'],
        "signatures": ["S54"],
        "why": "JPL molecular spectroscopy catalogue directory: the species list.  One "
               "expectation per industrial species, so expect_hits IS the inventory of "
               "which of them have a JPL line list (CH3Cl is the natural baseline).",
    },
    "splatalogue": {
        "url": "https://splatalogue.online/",
        "expect": [r"(?i)splatalogue"],
        "signatures": ["S54"],
        "why": "Splatalogue front end (also carries the CDMS/JPL/Lovas lists and the "
               "'unidentified' tags).",
    },
    # ---- S55 CRYPT: lunar permanently shadowed regions (Diviner / ShadowCam) ----
    "diviner_pds": {
        "url": "https://pds-geosciences.wustl.edu/missions/lro/diviner.htm",
        "expect": [r"(?i)diviner", r"(?i)polar|GDR|RDR|temperature"],
        "signatures": ["S55"],
        "why": "LRO Diviner products at the PDS Geosciences node: the polar bolometric "
               "temperature maps and their minimum-temperature products.",
    },
    "diviner_pcp_ode": {
        "url": "https://ode.rsl.wustl.edu/moon/pagehelp/Content/Missions_Instruments/LRO/DIVINER/PCP.htm",
        "expect": [r"(?i)polar\s+cumulative", r"(?i)240"],
        "signatures": ["S55"],
        "why": "The Diviner Polar Cumulative Product (PCP) description: 240 m/px, "
               "80-90 deg, channels 3-9, split by season and sub-solar longitude.",
    },
    "diviner_pcp_ucla": {
        "url": "http://luna1.diviner.ucla.edu/~jpierre/diviner/level4_polar/pds/",
        "expect": [r"(?i)polar", r"(?i)\.lbl|\.tab|\.img|href"],
        "signatures": ["S55"],
        "why": "UCLA mirror of the level-4 polar products (directory listing = bulk path).  "
               "Run 1: the https endpoint fails certificate verification (never disabled "
               "here); the http listing is tried instead.",
    },
    "diviner_pds_volumes": {
        "url": "https://pds-geosciences.wustl.edu/lro/",
        "expect": [r"(?i)dlre", r"(?i)lrodlr"],
        "signatures": ["S55"],
        "why": "PDS Geosciences LRO directory: which Diviner volumes exist (the polar "
               "products live in one of the lrodlr_100x volumes).",
    },
    "diviner_rdr_volume_root": {
        "url": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/",
        "expect": [r"(?i)lrodlr_100\d", r"(?i)lrodlr_1002|lrodlr_1003"],
        "signatures": ["S55"],
        "why": "Run 2: only the EDR and RDR datasets exist at the PDS node; the level-3/4 "
               "polar maps must be a second volume of the RDR dataset.  List its volumes.",
    },
    "diviner_pds_data_dir": {
        "url": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/",
        "expect": [r"(?i)gdr|pcp|prp|polar|href"],
        "signatures": ["S55"],
        "why": "Diviner RDR volume data directory listing.",
    },
    "shadowcam_archive": {
        "url": "https://shadowcam.im-ldi.com/archive",
        "expect": [r"(?i)shadowcam", r"(?i)release|pds"],
        "signatures": ["S55"],
        "why": "ShadowCam PDS4 archive page: quarterly releases, calibrated map-projected COGs.",
    },
    "shadowcam_pds_sis": {
        "url": "https://pds.shadowcam.im-ldi.com/document/archsis.pdf",
        "expect": [r"%PDF"],
        "signatures": ["S55"],
        "why": "ShadowCam archive SIS (product structure for range reads).",
    },
    "minirf_pds": {
        "url": "https://pds-geosciences.wustl.edu/missions/lro/mrf.htm",
        "expect": [r"(?i)mini-rf|mini rf", r"(?i)stokes|cpr|circular"],
        "signatures": ["S55"],
        "why": "Mini-RF S-band Stokes products (CPR maps) at the PDS Geosciences node.",
    },
    # ---- S56 GRAVE: the terrestrial record (SGP / EarthChem / GEOROC) ----
    "sgp_search": {
        "url": "https://sgp-search.io/",
        "expect": [r"(?i)sedimentary\s+geochemistry|SGP"],
        "signatures": ["S56"],
        "why": "Sedimentary Geochemistry and Paleoenvironments Project: tens of thousands "
               "of shale analyses with ages --- the fission-vector test at extinction "
               "boundaries.",
    },
    "sgp_api": {
        "url": "https://sgp-search.io/api/v1/samples?limit=5",
        "expect": [r"(?i)sample|age|interpreted_age|\"data\""],
        "signatures": ["S56"],
        "why": "SGP API (path asserted; corrected from the site if NO_PRODUCT).",
    },
    "earthchem_portal": {
        "url": "https://portal.earthchem.org/",
        "expect": [r"(?i)earthchem"],
        "signatures": ["S56"],
        "why": "EarthChem Portal (PetDB, GEOROC mirror, SedDB legacy).  Run 1: the host "
               "ecp.iedadata.org no longer resolves; the current portal host is tried.",
    },
    "earthchem_home": {
        "url": "https://www.earthchem.org/",
        "expect": [r"(?i)earthchem"],
        "signatures": ["S56"],
        "why": "EarthChem home (links to the portal and the library).",
    },
    "sgp_env_js": {
        "url": "https://sgp-search.io/env.js",
        "expect": [r"(?i)api|url|http"],
        "signatures": ["S56"],
        "why": "Run 1: /api/v1/samples served the single-page app, so the API path is "
               "unknown.  The app's env.js declares its API base URL; read it here.",
    },
    "sgp_archive": {
        "url": "https://archive.sgp-search.io/",
        "expect": [r"(?i)sgp|sedimentary|csv|zip|download|archive"],
        "signatures": ["S56"],
        "why": "Run 2: the app's env.js names https://archive.sgp-search.io as the "
               "REACT_APP_ARCHIVE_URL --- the bulk-download side of SGP.",
    },
    "sgp_github": {
        "url": "https://api.github.com/search/repositories?q=sgp-search+in:name",
        "expect": [r"(?i)sgp"],
        "signatures": ["S56"],
        "why": "Locate the SGP search application's repository (its README documents the API).",
    },
    "georoc": {
        "url": "https://georoc.eu/",
        "expect": [r"(?i)georoc"],
        "signatures": ["S56"],
        "why": "GEOROC: precompiled downloadable compilation files.",
    },
    # ---- S57 GROWTH: Kepler->TESS transit-depth drift ----
    "exoarchive_koi_depth": {
        "url": ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
                "select+top+5+kepoi_name,koi_depth,koi_depth_err1,koi_period,koi_disposition"
                "+from+cumulative&format=json"),
        "expect": [r"(?i)koi_depth"],
        "signatures": ["S57"],
        "why": "Kepler KOI cumulative table: koi_depth (ppm) per candidate, the 2009-2013 "
               "epoch of the depth-drift test.",
    },
    "exoarchive_toi_depth": {
        "url": ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
                "select+top+5+toi,tid,pl_trandep,pl_trandeperr1,pl_orbper,tfopwg_disp"
                "+from+toi&format=json"),
        "expect": [r"(?i)pl_trandep"],
        "signatures": ["S57"],
        "why": "TESS TOI table: pl_trandep (ppm) per TOI, the 2018-2026 epoch.",
    },
    "exoarchive_tess_kepler_xmatch": {
        "url": ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
                "select+top+5+pl_name,hostname,disc_facility,pl_trandep,pl_orbper,tic_id"
                "+from+ps+where+disc_facility+like+'%25Kepler%25'&format=json"),
        "expect": [r"(?i)tic_id"],
        "signatures": ["S57"],
        "why": "Planetary Systems table carries tic_id for Kepler planets: the join key "
               "to TESS light curves and TOIs.",
    },
    # ---- S58 SPARKLE: in-situ interstellar dust composition ----
    "cda_pds_sbn": {
        "url": "https://sbn.psi.edu/pds/resource/cocda.html",
        "expect": [r"(?i)cosmic\s+dust\s+analy|CDA"],
        "signatures": ["S58"],
        "why": "Cassini CDA archive at the PDS Small Bodies Node: are the impact-"
               "ionisation mass spectra (composition) of the interstellar grains served?",
    },
    "ulysses_dust_pds": {
        "url": "https://sbn.psi.edu/pds/resource/ulyssesdust.html",
        "expect": [r"(?i)ulysses", r"(?i)dust"],
        "signatures": ["S58"],
        "why": "Ulysses DUST detector archive (flux/direction/mass, 1990-2007).  Run 1: "
               "'ulydust.html' was a 404; alternative resource name tried.",
    },
    "sbn_dust_holdings": {
        "url": "https://sbn.psi.edu/pds/archive/dust.html",
        "expect": [r"(?i)ulysses", r"(?i)cassini", r"(?i)galileo", r"(?i)helios", r"(?i)new horizons",
                   r'(?i)href="[^"]*uly[^"]*"', r'(?i)href="[^"]*cda[^"]*"'],
        "signatures": ["S58"],
        "why": "SBN dust-archive listing: the authoritative index of in-situ dust datasets.  "
               "One expectation per mission and the two link patterns, so the record names "
               "the real Ulysses resource path (run 1 and 2 guessed it wrong twice).",
    },
    # ---- S59 ARC: superflares above the spot-energy ceiling ----
    "vizier_tap_home": {
        "url": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/tables",
        "expect": [r"(?i)tableset|schema|table"],
        "signatures": ["S59", "S47", "S52", "S51"],
        "why": "VizieR TAP: flare, rotation, exozodi, debris-disk and polluted-WD "
               "catalogues are read through it (probed below by ADQL).",
    },
    # ---- S60 RELAY: beam-spillover geometry / Breakthrough Listen open data ----
    "bl_opendata": {
        "url": "http://seti.berkeley.edu/opendata",
        "expect": [r"(?i)breakthrough|open\s*data|listen"],
        "signatures": ["S60"],
        "why": "Breakthrough Listen open data portal: which targets have public "
               "filterbank/h5 products for a geometry re-cut.",
    },
    "bl_bldata": {
        "url": "https://bldata.berkeley.edu/",
        "expect": [r"(?i)breakthrough|GBT|parkes|meerkat|\.h5|\.fil"],
        "signatures": ["S60"],
        "why": "The BL data server (directory listings of public products).",
    },
    # ---- S61 IGNITION: NEOWISE decade light curves (rising W1/W2) ----
    "irsa_neowise_tap": {
        "url": ("https://irsa.ipac.caltech.edu/TAP/sync?QUERY="
                "SELECT+TOP+3+ra,dec,mjd,w1mpro,w2mpro+FROM+neowiser_p1bs_psd"
                "+WHERE+CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',10.6847,41.2687,0.01))=1"
                "&FORMAT=json"),
        "expect": [r"(?i)w1mpro"],
        "signatures": ["S61"],
        "why": "NEOWISE single-exposure source table (the decade W1/W2 series).",
    },
    # ---- eROSITA (activity context for S59) ----
    "erosita_dr1": {
        "url": "https://erosita.mpe.mpg.de/dr1/",
        "expect": [r"(?i)erosita", r"(?i)DR1|catalog"],
        "signatures": ["S59"],
        "why": "eROSITA-DE DR1: X-ray activity for flare-star vetting.",
    },
    # ---- Meteoritical Bulletin (S46 / S58 context) ----
    "metbull": {
        "url": "https://www.lpi.usra.edu/meteor/metbull.php",
        "expect": [r"(?i)meteoritical\s+bulletin"],
        "signatures": ["S46"],
        "why": "Meteoritical Bulletin Database: 'ungrouped / anomalous' classifications.",
    },
}

TAP: dict[str, dict] = {
    "vizier_flare_rotation_tables": {
        "service": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
        "adql": ("SELECT table_name, description FROM TAP_SCHEMA.tables "
                 "WHERE table_name LIKE '%J/ApJS/241/29/%' OR table_name LIKE '%J/ApJS/211/24/%' "
                 "OR table_name LIKE '%J/ApJS/253/35/%' OR table_name LIKE '%J/ApJ/935/143/%' "
                 "OR table_name LIKE '%J/ApJS/225/15/%' OR table_name LIKE '%J/A+A/570/A128/%' "
                 "OR table_name LIKE '%J/AJ/159/177/%' OR table_name LIKE '%J/MNRAS/467/4970/%' "
                 "OR table_name LIKE '%J/A+A/555/A104/%' OR table_name LIKE '%J/A+A/608/A113/%' "
                 "OR table_name LIKE '%J/A+A/651/A45/%' OR table_name LIKE '%J/ApJ/876/58/%' "
                 "OR table_name LIKE '%J/ApJ/829/23/%' OR table_name LIKE '%J/AJ/159/60/%'"),
        "expect": [r"J/ApJS/241/29", r"J/ApJS/211/24", r"J/A\+A/570/A128", r"J/A\+A/555/A104",
                   r"J/AJ/159/177", r"J/ApJS/225/15"],
        "signatures": ["S59", "S47", "S52", "S51"],
        "why": "Named VizieR tables the briefs assert exist: Yang & Liu 2019 flares, "
               "McQuillan 2014 rotation, TESS flare catalogues, Cotten & Song 2016 debris "
               "census, Ertel 2014 PIONIER exozodi, Ertel 2020 HOSTS, Hollands 2017 DZ.",
    },
    "vizier_search_exozodi": {
        "service": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
        "adql": ("SELECT table_name, description FROM TAP_SCHEMA.tables "
                 "WHERE description LIKE '%exozodi%' OR description LIKE '%PIONIER%' "
                 "OR description LIKE '%FLUOR%' OR description LIKE '%hot dust%'"),
        "expect": [r"(?i)exozodi|pionier|fluor|hot dust"],
        "signatures": ["S47"],
        "why": "Any interferometric exozodi survey table VizieR serves.",
    },
    "vizier_search_polluted_wd": {
        "service": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
        "adql": ("SELECT table_name, description FROM TAP_SCHEMA.tables "
                 "WHERE table_name LIKE '%J/A+A/691/A352%' "
                 "OR description LIKE '%polluted%white dwarf%' "
                 "OR description LIKE '%DZ white dwarf%' OR description LIKE '%PEWDD%'"),
        "expect": [r"J/A\+A/691/A352", r"(?i)white dwarf"],
        "signatures": ["S51"],
        "why": "PEWDD at VizieR (J/A+A/691/A352) and any other polluted-WD abundance table.",
    },
    "vizier_moor2021_edd": {
        "service": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
        "adql": ("SELECT table_name, description FROM TAP_SCHEMA.tables "
                 "WHERE table_name LIKE '%J/ApJ/910/27/%' OR table_name LIKE '%J/MNRAS/433/2334/%' "
                 "OR table_name LIKE '%J/ApJS/225/15/%' OR table_name LIKE '%J/ApJ/805/77/%' "
                 "OR (description LIKE '%Moor%' AND description LIKE '%debris%')"),
        "expect": [r"J/ApJ/910/27|J/MNRAS/433/2334|J/ApJS/225/15|J/ApJ/805/77"],
        "signatures": ["S52"],
        "why": "Moór+2021 EDD table, Kennedy & Wyatt 2013 warm-dust LF, Cotten & Song 2016 "
               "census, Meng+2015 EDD photometry (run 1 found only the last, by description).",
    },
    "irsa_irs_enhanced_columns": {
        "service": "https://irsa.ipac.caltech.edu/TAP",
        "adql": ("SELECT column_name, datatype, description FROM TAP_SCHEMA.columns "
                 "WHERE table_name = 'irs_enhv211'"),
        "expect": [r"(?i)ra|wave|flux|aor|object"],
        "signatures": ["S52", "S53"],
        "why": "Run 2 found IRSA's 'IRS Enhanced Products' table (irs_enhv211): the "
               "extracted Spitzer/IRS low-res spectra, reachable when cassis.sirtf.com is "
               "not (it timed out from the runner in both runs).  Its columns decide "
               "whether S53 can be built on it.",
    },
    "irsa_irs_enhanced_count": {
        "service": "https://irsa.ipac.caltech.edu/TAP",
        "adql": "SELECT COUNT(*) AS n FROM irs_enhv211",
        "expect": [r"\"n\""],
        "signatures": ["S52", "S53"],
        "why": "How many IRS Enhanced spectra IRSA serves.",
    },
    "irsa_splices_rows": {
        "service": "https://irsa.ipac.caltech.edu/TAP",
        "adql": "SELECT COUNT(*) AS n FROM splices",
        "expect": [r"\"n\""],
        "signatures": ["S48"],
        "why": "How many SPLICES seed sources the TAP table actually holds (run 1 confirmed "
               "the table exists).",
    },
    "irsa_splices_columns": {
        "service": "https://irsa.ipac.caltech.edu/TAP",
        "adql": ("SELECT column_name, datatype, description FROM TAP_SCHEMA.columns "
                 "WHERE table_name = 'splices'"),
        "expect": [r"(?i)ra|dec|mag|flux"],
        "signatures": ["S48"],
        "why": "SPLICES columns: the seed-list photometry and identifiers the forced "
               "spectrophotometry needs.",
    },
    "irsa_spherex_obscore_count": {
        "service": "https://irsa.ipac.caltech.edu/TAP",
        "adql": "SELECT COUNT(*) AS n FROM spherex.obscore",
        "expect": [r"\"n\""],
        "signatures": ["S48", "S61"],
        "why": "How many SPHEREx spectral-image products are public today (the denominator "
               "for sky coverage per pass).",
    },
    "vizier_superflare_tables": {
        "service": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
        "adql": ("SELECT table_name, description FROM TAP_SCHEMA.tables "
                 "WHERE table_name LIKE '%J/ApJ/876/58%' OR table_name LIKE '%J/ApJ/906/72%' "
                 "OR table_name LIKE '%J/ApJS/253/35%' OR description LIKE '%superflare%'"),
        "expect": [r"(?i)superflare|J/ApJ/906/72|J/ApJ/876/58"],
        "signatures": ["S59"],
        "why": "Notsu+2019 / Okamoto+2021 / Tu+2021 superflare tables carrying per-star "
               "spot amplitude AND per-flare energy (the two columns the spot-ceiling "
               "residual needs).",
    },
    "vizier_search_debris_extreme": {
        "service": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
        "adql": ("SELECT table_name, description FROM TAP_SCHEMA.tables "
                 "WHERE description LIKE '%extreme debris%' OR description LIKE '%warm debris%' "
                 "OR description LIKE '%debris disk%census%' OR description LIKE '%debris disc%census%'"),
        "expect": [r"(?i)debris"],
        "signatures": ["S52"],
        "why": "Debris-disk census / extreme-debris tables.",
    },
    "irsa_spherex_tables": {
        "service": "https://irsa.ipac.caltech.edu/TAP",
        "adql": ("SELECT table_name, description FROM TAP_SCHEMA.tables "
                 "WHERE table_name LIKE '%spherex%' OR description LIKE '%SPHEREx%' "
                 "OR table_name LIKE '%splices%'"),
        "expect": [r"(?i)spherex"],
        "signatures": ["S48", "S49"],
        "why": "Is any SPHEREx table served through IRSA TAP yet (spherex.plane / "
               "spherex.artifact image metadata; the SPLICES 9-million-source seed list)?",
    },
    "irsa_spherex_plane_columns": {
        "service": "https://irsa.ipac.caltech.edu/TAP",
        "adql": ("SELECT column_name, datatype, description FROM TAP_SCHEMA.columns "
                 "WHERE table_name = 'spherex.plane'"),
        "expect": [r"(?i)energy_bandpassname|obs|time"],
        "signatures": ["S48", "S49"],
        "why": "The columns of spherex.plane (obs time, bandpass name, footprint polygon) "
               "asserted from the IRSA tutorial notebooks.",
    },
    "irsa_spherex_recent_exposures": {
        "service": "https://irsa.ipac.caltech.edu/TAP",
        "adql": ("SELECT TOP 5 p.planeid, p.energy_bandpassname, a.uri "
                 "FROM spherex.plane p JOIN spherex.artifact a ON p.planeid = a.planeid"),
        "expect": [r"(?i)uri"],
        "signatures": ["S48", "S49"],
        "why": "Can the join that yields cutout URIs be executed (the bulk path to "
               "do-it-yourself 102-channel spectra on the SPLICES seeds)?",
    },
    "irsa_splices_count": {
        "service": "https://irsa.ipac.caltech.edu/TAP",
        "adql": ("SELECT table_name FROM TAP_SCHEMA.tables "
                 "WHERE table_name LIKE '%splices%' OR description LIKE '%SPLICES%' "
                 "OR description LIKE '%List of Ices%'"),
        "expect": [r"(?i)splices|ices"],
        "signatures": ["S48"],
        "why": "The SPLICES seed catalogue (added to IRSA Catalog Search May 2026) as a TAP table.",
    },
    "irsa_euclid_spe_lines": {
        "service": "https://irsa.ipac.caltech.edu/TAP",
        "adql": ("SELECT column_name, datatype FROM TAP_SCHEMA.columns "
                 "WHERE table_name = 'euclid_q1_spe_lines_line_features'"),
        "expect": [r"(?i)spe_line_snr|spe_line_name"],
        "signatures": ["S48"],
        "why": "Euclid Q1 line-feature table: per-source detected lines with SNR and "
               "wavelength --- a pre-computed narrow-feature list over 4.3 M NISP spectra "
               "in the Er-fibre band, if the columns match the tutorial.",
    },
    "irsa_euclid_tables": {
        "service": "https://irsa.ipac.caltech.edu/TAP",
        "adql": ("SELECT table_name, description FROM TAP_SCHEMA.tables "
                 "WHERE table_name LIKE '%euclid%'"),
        "expect": [r"(?i)euclid"],
        "signatures": ["S48"],
        "why": "Euclid Q1/DR1 tables at IRSA (NISP spectra for the 1.2-1.9 um industrial "
               "line, the Er-fibre band).",
    },
    "irsa_cassis_tables": {
        "service": "https://irsa.ipac.caltech.edu/TAP",
        "adql": ("SELECT table_name, description FROM TAP_SCHEMA.tables "
                 "WHERE table_name LIKE '%cassis%' OR table_name LIKE '%irs%' "
                 "OR description LIKE '%IRS %' OR description LIKE '%Spitzer%Enhanced%'"),
        "expect": [r"(?i)cassis|irs"],
        "signatures": ["S52", "S53"],
        "why": "CASSIS / IRS spectral products through IRSA TAP.",
    },
    "gaia_pairs_geometry_feasibility": {
        "service": "https://gea.esac.esa.int/tap-server/tap",
        "adql": ("SELECT COUNT(*) AS n FROM gaiadr3.gaia_source "
                 "WHERE parallax > 10 AND parallax_over_error > 10 AND ruwe < 1.4"),
        "expect": [r"\"n\""],
        "signatures": ["S60"],
        "why": "Denominator for the beam-spillover pair geometry: stars within 100 pc "
               "with good astrometry.",
    },
}


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Pure functions (offline-tested)
# --------------------------------------------------------------------------
def classify(rec: dict, expect: list[str]) -> str:
    """One verdict per endpoint from what it served.

    ``rec`` carries ``status`` (int or None), ``body`` (str or None) and
    optionally ``skipped``.  ``expect`` is a list of regexes; any one matching
    the body is enough for REACHED_WITH_PRODUCT.
    """
    if rec.get("skipped"):
        return "NOT_PROBED"
    status = rec.get("status")
    body = rec.get("body") or ""
    if status is None or not (200 <= int(status) < 300):
        return "NOT_REACHED"
    hits, matched = [], []
    for p in expect:
        m = re.search(p, body)
        if m:
            hits.append(p)
            matched.append(m.group(0)[:80])
    rec["expect_hits"] = hits
    rec["expect_matched_text"] = matched
    return "REACHED_WITH_PRODUCT" if hits else "REACHED_NO_PRODUCT"


def summarise(records: dict[str, dict]) -> dict:
    """Per-verdict counts and the per-signature readiness map.

    A signature is READY if every endpoint it depends on reached its product,
    PARTIAL if at least one did, BLOCKED if none did.
    """
    counts: dict[str, int] = {}
    per_sig: dict[str, dict] = {}
    for name, rec in records.items():
        v = rec["verdict"]
        counts[v] = counts.get(v, 0) + 1
        for sig in rec.get("signatures", []):
            d = per_sig.setdefault(sig, {"endpoints": [], "with_product": 0, "total": 0})
            d["endpoints"].append(name)
            d["total"] += 1
            if v == "REACHED_WITH_PRODUCT":
                d["with_product"] += 1
    for sig, d in per_sig.items():
        if d["with_product"] == d["total"] and d["total"] > 0:
            d["readiness"] = "READY"
        elif d["with_product"] > 0:
            d["readiness"] = "PARTIAL"
        else:
            d["readiness"] = "BLOCKED"
    n = len(records)
    reached = counts.get("REACHED_WITH_PRODUCT", 0) + counts.get("REACHED_NO_PRODUCT", 0)
    if n == 0:
        verdict = "NOTHING_PROBED"
    elif reached == 0:
        verdict = "NO_DATA_SOURCE_REACHED"
    elif counts.get("REACHED_WITH_PRODUCT", 0) == n:
        verdict = "ALL_SOURCES_REACHED_WITH_PRODUCT"
    else:
        verdict = "PARTIAL_REACH"
    return {"verdict": verdict, "n_endpoints": n, "counts": counts,
            "per_signature": dict(sorted(per_sig.items()))}


def render_md(summary: dict, records: dict[str, dict]) -> str:
    lines = ["# NECROFRONTIER data-source probe", "",
             f"Verdict: **{summary['verdict']}** over {summary['n_endpoints']} endpoints "
             f"({json.dumps(summary['counts'])}).", "",
             "| endpoint | verdict | HTTP | bytes | signatures | hits |", "|---|---|---|---|---|---|"]
    for name, rec in records.items():
        lines.append(f"| `{name}` | {rec['verdict']} | {rec.get('status')} | "
                     f"{rec.get('bytes', 0)} | {', '.join(rec.get('signatures', []))} | "
                     f"{'; '.join(rec.get('expect_hits', []))[:80]} |")
    lines += ["", "| signature | readiness | endpoints |", "|---|---|---|"]
    for sig, d in summary["per_signature"].items():
        lines.append(f"| {sig} | {d['readiness']} | {', '.join(d['endpoints'])} |")
    lines += ["", "REACHED_NO_PRODUCT is a finding about the *brief*, not the sky: the URL, "
              "product name or table name asserted in `docs/necrofrontier.md` must be "
              "corrected from what the endpoint actually served (its `head` is in "
              "`probe.json`)."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Network (runner only)
# --------------------------------------------------------------------------
def _get(url: str, timeout: float) -> dict:
    import requests
    rec: dict = {"url": url}
    try:
        resp = requests.get(url, timeout=timeout, headers=UA, allow_redirects=True)
        rec["status"] = resp.status_code
        rec["final_url"] = resp.url
        body = resp.text or ""
        rec["bytes"] = len(body)
        rec["body"] = body
        rec["head"] = body[:2500]
    except Exception as exc:  # noqa: BLE001
        rec["status"] = None
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return rec


def _tap(service: str, adql: str, timeout: float, maxrec: int = 200) -> dict:
    rec: dict = {"url": service, "adql": adql}
    try:
        import pyvo
        svc = pyvo.dal.TAPService(service)
        tab = svc.search(adql, maxrec=maxrec).to_table()
        rows = [{c: str(r[c]) for c in tab.colnames} for r in tab]
        rec["status"] = 200
        rec["rows"] = len(rows)
        body = json.dumps(rows)
        rec["bytes"] = len(body)
        rec["body"] = body
        rec["head"] = body[:2500]
    except Exception as exc:  # noqa: BLE001
        rec["status"] = None
        rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/necrofrontier")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--only", default="", help="comma-separated endpoint names")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    only = {s for s in args.only.split(",") if s}

    records: dict[str, dict] = {}
    for name, spec in REST.items():
        if only and name not in only:
            continue
        print(f"[probe] {name} ...", flush=True)
        rec = _get(spec["url"], args.timeout)
        rec["verdict"] = classify(rec, spec["expect"])
        rec.update({"signatures": spec["signatures"], "why": spec["why"], "kind": "rest"})
        rec.pop("body", None)
        print(f"        {rec['verdict']} status={rec.get('status')} bytes={rec.get('bytes', 0)}")
        records[name] = rec
    for name, spec in TAP.items():
        if only and name not in only:
            continue
        print(f"[probe] {name} (TAP) ...", flush=True)
        rec = _tap(spec["service"], spec["adql"], args.timeout)
        rec["verdict"] = classify(rec, spec["expect"])
        rec.update({"signatures": spec["signatures"], "why": spec["why"], "kind": "tap"})
        rec.pop("body", None)
        print(f"        {rec['verdict']} rows={rec.get('rows')} err={rec.get('error', '')[:100]}")
        records[name] = rec

    summary = summarise(records)
    payload = {"probed_at_utc": _utc(), "summary": summary, "endpoints": records}
    (out / "probe.json").write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    (out / "PROBE.md").write_text(render_md(summary, records))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
