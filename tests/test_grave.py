"""Offline test suite for GRAVE --- a technological extinction in the sedimentary record (S56).

No network anywhere (``conftest.py`` raises on any socket).  Per
``docs/channel-brief.md`` §5 the suite:

* recovers a fission-yield vector injected on a PAAS baseline (LR above the
  floor, leave-one-out survives, peak coherence holds) and classes the sample
  FISSION_CANDIDATE through the full vet;
* returns LR ~ 0 on a black-shale redox enrichment (Mo-U-V-Ni-Cu-Zn-Cd-Ag up)
  and, when a redox sample is forced through the vet, trips ``redox_conditioned``;
* classes a chondritic, Ir-anchored PGE spike ``impact`` and an IPGE-rich
  pattern ``ultramafic``; a Pt-Pd-Rh spike with Ir at background ``refined_pge``;
  a Ru-Rh-Pd spike with neither Ir nor Pt ``fission_like_pge``;
* classes Ta without Nb ``refined_ta``, W with hydrothermal partners
  ``hydrothermal_w``;
* trips ``single_element_driver``, ``monazite_th``, ``heavy_mineral_zr_hf``,
  ``detection_limit_driver``;
* age-stacks: two independent sections at one boundary is a
  STRATIGRAPHIC_CLUSTER, one section is ``single_section``;
* resolves SGP display keys, EarthChem keys and GEOROC oxide headers at
  runtime, with unit scales;
* walks the probe (bundle -> codes) and the paged acquisition with a scripted
  fetcher, and reports NO_DATA_REACHED end-to-end when every source is empty.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from seti.grave import acquire as A
from seti.grave import agestack as G
from seti.grave import references as R
from seti.grave import vectors as V
from seti.grave.run import grave_run, load_grave_config, stage_assess, stage_screen

CONF = load_grave_config()
CFG = V.GraveConfig()
SKIP = ("Ru", "Rh", "Pd", "Te", "I", "Os", "Ir", "Pt", "Au", "Re", "Se", "Hg", "Ag", "Cd",
        "In", "Sn", "Sb", "Tl", "Bi", "B", "Be", "Li")
ELS = [e for e in R.FIT_ELEMENTS if e in R.PAAS and e not in SKIP]
D = V.build_design(ELS, CFG)
PHI = V.fission_mass_vector(CFG)


def synth(n, *, a=0.0, mix=None, seed=0, noise=0.8, override=None):
    """PAAS + optional reservoir mix + fission at amplitude a, log-normal noise."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, len(D.elements)))
    for i in range(n):
        f = {"paas": 1.0}
        if mix:
            f.update(mix)
        c = sum(fr * np.array([R.RESERVOIRS[k].get(e, 0.0) for e in D.elements]) for k, fr in f.items())
        c = c + a * np.array([PHI.get(e, 0.0) for e in D.elements])
        c = c * 10 ** (rng.normal(0, D.sigma * noise))
        if override:
            for e, v in override.items():
                c[D.elements.index(e)] = v
        out[i] = c
    return out


def rowdict(c):
    return {e: float(c[k]) for k, e in enumerate(D.elements) if np.isfinite(c[k])}


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------
def test_fission_mass_vector_is_normalised_to_paas_nd_and_ranks_ru_first():
    assert PHI["Nd"] == pytest.approx(R.PAAS["Nd"])
    disc = {r["element"]: r["rel_to_anchor"] for r in V.fission_discriminants(CFG)}
    assert disc["Ru"] > 1e4 and disc["Mo"] > 10 and disc["Cs"] > 1
    # the absent partners: no Hf with the Zr, no Th, no Gd beyond the cliff
    assert "Hf" not in PHI and "Th" not in PHI and "U" not in PHI
    assert disc["Gd"] < 0.05 and disc["Ba"] < 0.05


def test_injected_fission_vector_is_recovered_and_controls_are_null():
    X = np.vstack([synth(200), synth(30, a=1.0, seed=1)])
    fit = V.fit_mixture(X, D, CFG)
    lr = fit["fission_lr"].to_numpy()
    assert (lr[:200] < CFG.lr_min).all()
    assert np.median(lr[200:]) > 20
    assert (lr[200:] >= CFG.lr_min).mean() > 0.9
    assert np.median(fit["a_fission"].to_numpy()[200:]) == pytest.approx(1.0, abs=0.25)
    assert (lr >= 0).all()


def test_injected_sample_survives_the_full_vet():
    X = synth(1, a=1.5, seed=3)
    fit = V.fit_mixture(X, D, CFG)
    res = V.assess_row(rowdict(X[0]), X[0], D, CFG, D.sigma, fit=fit.iloc[0].to_dict(), threshold=CFG.lr_min)
    assert res["class"] == V.FISSION_CANDIDATE, res
    assert res["lr_loo_min"] >= CFG.lr_min
    assert res["peak"]["coherent"]


@pytest.mark.parametrize("mix", [{"authigenic": 1.0}, {"chondrite": 0.01}, {"monazite": 0.001},
                                 {"femn_oxide": 0.05}, {"zircon": 0.002}, {"apatite": 0.2}])
def test_natural_confounders_are_absorbed_by_the_family(mix):
    X = synth(40, mix=mix, seed=5)
    fit = V.fit_mixture(X, D, CFG)
    assert (fit["fission_lr"] < CFG.lr_min).all(), mix
    assert (fit["reduced_chi2_natural"] < CFG.max_reduced_chi2).all()


def test_black_shale_redox_trips_redox_conditioned_when_mo_drives():
    # a Mo-only spike on an anoxic shale: LR may rise, the vet must catch it
    X = synth(1, mix={"authigenic": 1.5}, seed=7, override={"Mo": 400.0})
    fit = V.fit_mixture(X, D, CFG)
    assert float(fit["fission_lr"].iat[0]) < CFG.lr_min          # the family absorbs the shale
    res = V.assess_row(rowdict(X[0]), X[0], D, CFG, D.sigma, fit=fit.iloc[0].to_dict(), threshold=CFG.lr_min)
    assert res["class"] != V.FISSION_CANDIDATE
    assert res["redox"]["anoxic"] and res["vetoes"]
    # the rule itself: an anoxic sample whose preference rests on Mo is redox-conditioned
    row = rowdict(X[0])
    st = V.redox_state(row, CFG)
    assert st["anoxic"] and st["ef_u"] >= CFG.redox_ef_u
    # a plain PAAS shale is not anoxic
    assert not V.redox_state(rowdict(synth(1, seed=8)[0]), CFG)["anoxic"]


def test_single_element_driver_and_monazite_and_zircon_vetoes():
    # a lone Nd spike
    X = synth(1, seed=11, override={"Nd": 400.0})
    fit = V.fit_mixture(X, D, CFG)
    res = V.assess_row(rowdict(X[0]), X[0], D, CFG, D.sigma, fit=fit.iloc[0].to_dict(), threshold=0.0)
    assert res["class"] != V.FISSION_CANDIDATE
    assert "single_element_driver" in res["vetoes"] or "peak_incoherent" in res["vetoes"]
    # LREE with Th up: monazite (forced driver)
    row = rowdict(synth(1, seed=12)[0])
    row.update({"La": 400, "Ce": 800, "Nd": 300, "Th": 200})
    c = np.array([row.get(e, np.nan) for e in D.elements])
    fit = V.fit_mixture(c[None, :], D, CFG)
    res = V.assess_row(row, c, D, CFG, D.sigma, fit=fit.iloc[0].to_dict(), threshold=0.0)
    assert res["class"] != V.FISSION_CANDIDATE
    # Zr with Hf at the zircon ratio
    ef_th = V.enrichment_factor(row, "Th")
    assert ef_th > CFG.monazite_th_ef
    row2 = rowdict(synth(1, seed=13)[0])
    row2.update({"Zr": 3000.0, "Hf": 3000.0 / 38.0})
    assert CFG.zr_hf_natural[0] <= row2["Zr"] / row2["Hf"] <= CFG.zr_hf_natural[1]


def test_detection_limit_mask_flags_a_repeated_floor():
    df = pd.DataFrame({"Mo": [0.5] * 30 + list(np.linspace(0.6, 5, 20)), "U": np.linspace(1, 5, 50)})
    mask, ledger = V.detection_limit_mask(df, CFG)
    assert mask["Mo"].sum() == 30 and "Mo" in ledger and "U" not in ledger


def test_pge_classes():
    base = rowdict(synth(1, seed=21)[0])
    # chondritic spike anchored on Ir (K-Pg like: 0.01 % chondrite)
    imp = dict(base)
    imp.update({e: R.CI_CHONDRITE[e] * 1e-3 for e in R.PGE})
    assert V.classify_pge(imp, CFG)["pge_class"] == V.PGE_IMPACT
    # IPGE-rich: Os, Ir, Ru up, Pd/Ir far below chondritic
    um = dict(base)
    um.update({"Os": 0.02, "Ir": 0.02, "Ru": 0.03, "Pt": 0.004, "Pd": 0.001})
    assert V.classify_pge(um, CFG)["pge_class"] == V.PGE_ULTRAMAFIC
    # refined: Pt-Pd-Rh up, Ir at background
    ref = dict(base)
    ref.update({"Ir": R.UCC["Ir"], "Pt": 0.05, "Pd": 0.05, "Rh": 0.01})
    assert V.classify_pge(ref, CFG)["pge_class"] == V.PGE_REFINED
    # fission-like: Ru-Rh-Pd up, Ir and Pt at background
    fis = dict(base)
    fis.update({"Ir": R.UCC["Ir"], "Pt": R.UCC["Pt"], "Ru": 0.05, "Rh": 0.01, "Pd": 0.05})
    assert V.classify_pge(fis, CFG)["pge_class"] == V.PGE_FISSION_LIKE
    # background
    bg = dict(base)
    bg.update({e: R.UCC[e] for e in R.PGE})
    assert V.classify_pge(bg, CFG)["pge_class"] == V.PGE_BACKGROUND
    # Pt/Pd only cannot exclude impact
    pp = dict(base)
    pp.update({"Pt": 0.05, "Pd": 0.05})
    assert V.classify_pge(pp, CFG)["pge_class"] == V.PGE_UNCLASSIFIED


def test_alloy_classes():
    base = rowdict(synth(1, seed=22)[0])
    ta = dict(base)
    ta.update({"Ta": 100.0, "Nb": 20.0})
    assert V.classify_alloy(ta, CFG)["alloy_class"] == V.ALLOY_REFINED_TA
    nat = dict(base)
    nat.update({"Ta": 100.0, "Nb": 1500.0})
    assert V.classify_alloy(nat, CFG)["alloy_class"] == V.ALLOY_NATURAL_TA
    w = dict(base)
    w.update({"W": 500.0, "Sn": 200.0, "Mo": 50.0})
    assert V.classify_alloy(w, CFG)["alloy_class"] == V.ALLOY_NATURAL_W
    w2 = dict(base)
    w2.update({"W": 500.0})
    assert V.classify_alloy(w2, CFG)["alloy_class"] == V.ALLOY_REFINED_W


def test_brief_ratios_have_the_signs_the_brief_asserts():
    # the vector's own imposition, read off the yields, not asserted
    X = synth(1, a=2.0, seed=41)
    m, _ = V.natural_model(X[0], D, D.sigma)
    br = V.brief_ratios(rowdict(X[0]), m, D, CFG)
    assert br["[Nd/Ba]"]["fission_dex"] > 1.0          # Nd/Ba strongly up
    assert br["[Eu/Nd]"]["fission_dex"] < 0.0          # Eu/Nd slightly down
    assert br["[Mo/Zr]"]["fission_dex"] > 1.0          # Mo/Zr strongly up
    # and an injected sample shows them against the best natural mixture
    assert br["[Nd/Ba]"]["vs_model"] is not None and br["[Mo/Zr]"]["vs_model"] > 0
    # a plain shale does not
    m0, _ = V.natural_model(synth(1, seed=42)[0], D, D.sigma)
    br0 = V.brief_ratios(rowdict(synth(1, seed=42)[0]), m0, D, CFG)
    assert abs(br0["[Mo/Zr]"]["vs_model"]) < abs(br["[Mo/Zr]"]["vs_model"])


def test_shuffled_null_and_error_floors_run():
    X = synth(120, seed=31)
    fl = V.error_floors(X, D, CFG, max_rows=120)
    assert fl["Nd"]["n"] == 120 and fl["Nd"]["floor_dex"] is not None
    nl = V.shuffled_null(X, D, CFG, max_rows=60)
    assert nl["n"] == 60 and nl["lr_quantile"] is not None


# ---------------------------------------------------------------------------
# age stack
# ---------------------------------------------------------------------------
def test_age_stack_cluster_vs_single_section():
    b = G.boundary_table(CONF)
    rows = []
    rng = np.random.default_rng(0)
    for i in range(400):
        age = float(rng.uniform(0, 600))
        rows.append({"age": age, "section_key": f"S{i % 80}", "is_candidate": False,
                     "boundary": G.assign_boundary(age, None, None, b)})
    # K-Pg: three sections carry a candidate
    for s in ("S1", "S2", "S3"):
        rows.append({"age": 66.0, "section_key": s, "is_candidate": True, "boundary": "k_pg"})
    for s in ("S4", "S5"):
        rows.append({"age": 66.0, "section_key": s, "is_candidate": False, "boundary": "k_pg"})
    rows.append({"age": 252.0, "section_key": "S9", "is_candidate": True, "boundary": "end_permian"})
    df = pd.DataFrame(rows)
    st = G.age_stack(df, b)
    assert st["boundaries"]["k_pg"]["status"] == "STRATIGRAPHIC_CLUSTER"
    assert st["boundaries"]["end_permian"]["status"] == "single_section"
    assert st["boundaries"]["toarcian"]["status"] == "no_candidate"
    assert G.assign_boundary(66.5, None, None, b) == "k_pg"
    assert G.assign_boundary(70.0, 64.0, 72.0, b) == "k_pg"        # bracket overlaps
    assert G.assign_boundary(300.0, None, None, b) == ""
    # the promotion is family-wise: the raw p is paid for over the windows that
    # were testable at all, and the corrected p is what the status reads
    kpg = st["boundaries"]["k_pg"]
    assert st["multiple_testing"] == "holm" and st["cluster_p_is_family_wise"] is True
    assert 0 < st["n_boundaries_tested"] <= len(b)
    assert kpg["p_family"] >= kpg["p_hypergeom"] and kpg["p_family"] < st["cluster_p"]


def test_holm_adjustment_is_monotone_and_pays_for_the_catalogue():
    assert G.holm_adjust([]) == []
    # smallest p is scaled by m, the next by m-1, and the sequence never falls
    adj = G.holm_adjust([0.001, 0.02, 0.5, 0.04])
    assert adj[0] == pytest.approx(0.004)         # smallest, x4
    assert adj[1] == pytest.approx(0.06)          # next, x3
    assert adj[3] == pytest.approx(0.08)          # next, x2
    assert adj[2] == pytest.approx(0.5)           # largest, x1
    # monotone step-down: read in order of raw p, the adjusted values never fall
    raw = [0.001, 0.30, 0.012, 0.9]
    seq = [a for _, a in sorted(zip(raw, G.holm_adjust(raw), strict=True))]
    assert seq == sorted(seq)
    assert all(x <= 1.0 for x in G.holm_adjust([0.9, 0.95, 1.0]))


def test_a_window_significant_only_before_correction_is_not_promoted():
    """Two candidate sections that a per-window 0.01 rule would have promoted,
    but which the catalogue-wide correction does not: the status must be
    ``multi_section_at_background_rate``, not a cluster."""
    b = G.boundary_table(CONF)
    # 300 sections, 5 of them carrying a candidate; the K-Pg window holds 10
    # sections, 2 of which are candidate sections.  Hypergeom(300, 5, 10) puts
    # P(X >= 2) at 0.0095 -- under a per-window 0.01, over it once paid for.
    others = ("end_permian", "toarcian", "oae2", "petm", "kellwasser", "capitanian")
    rows = [{"age": 300.0, "section_key": f"S{i}", "is_candidate": i in (10, 11, 12),
             "boundary": others[i % len(others)]} for i in range(10, 300)]
    rows += [{"age": 66.0, "section_key": f"S{i}", "is_candidate": i < 2, "boundary": "k_pg"}
             for i in range(10)]
    df = pd.DataFrame(rows)
    st = G.age_stack(df, b)
    kpg = st["boundaries"]["k_pg"]
    assert st["n_candidate_sections_total"] == 5 and st["n_boundaries_tested"] == 7
    assert kpg["n_candidate_sections"] == 2 and kpg["n_sections"] == 10
    assert kpg["p_hypergeom"] < 0.01 < kpg["p_family"], kpg
    assert kpg["status"] == "multi_section_at_background_rate"


# ---------------------------------------------------------------------------
# schema resolution
# ---------------------------------------------------------------------------
def test_resolve_sgp_display_keys_earthchem_keys_and_georoc_oxides():
    sgp = ["sample identifier", "section name", "site type", "site latitude", "site longitude",
           "interpreted age", "max age", "min age", "lithology name", "Al (wt%)", "Mo (ppm)", "Nd (ppm)",
           "TOC (wt%)", "Ir (ppb)", "ref_short"]
    res = A.resolve_columns(sgp, source="sgp")
    assert res["meta"]["sample_id"] == "sample identifier" and res["meta"]["age"] == "interpreted age"
    assert res["elements"]["Al"][0][1] == 1e4 and res["elements"]["Ir"][0][1] == 1e-3
    assert "TOC" in res["elements"] and res["meta"]["reference"] == "ref_short"
    codes = ["alu", "mo", "nd", "toc", "coord_lat", "coord_long", "interpreted_age", "section_name"]
    res2 = A.resolve_columns(codes, source="sgp")
    assert res2["elements"]["Al"][0][1] == 1e4 and res2["assumed_units"]["alu"] == "wt%"
    assert res2["elements"]["Mo"][0][1] == 1.0 and res2["meta"]["lat"] == "coord_lat"
    geo = ["UNIQUE_ID", "SAMPLE NAME", "CITATIONS", "AL2O3(WT%)", "FEOT(WT%)", "FEO(WT%)", "ZR(PPM)",
           "LA(PPM)", "IR(PPB)", "MIN. AGE (YRS.)", "MAX. AGE (YRS.)", "LATITUDE (MIN.)", "LONGITUDE (MIN.)"]
    res3 = A.resolve_columns(geo, source="georoc")
    assert res3["elements"]["Al"][0][1] == pytest.approx(1e4 * 0.5293)
    assert res3["elements"]["Fe"][0][0] == "FEOT(WT%)"
    assert res3["meta"]["min_age"] == "MIN. AGE (YRS.)"
    raw = pd.DataFrame({"UNIQUE_ID": ["1"], "SAMPLE NAME": ["x"], "CITATIONS": ["[1]"], "AL2O3(WT%)": ["15"],
                        "FEOT(WT%)": [""], "FEO(WT%)": ["5"], "ZR(PPM)": ["100"], "LA(PPM)": ["<0.5"],
                        "IR(PPB)": ["2"], "MIN. AGE (YRS.)": ["66000000"], "MAX. AGE (YRS.)": ["67000000"],
                        "LATITUDE (MIN.)": ["1"], "LONGITUDE (MIN.)": ["2"]})
    canon = A.canonicalise(raw, res3, source="georoc")
    assert canon["Al"].iat[0] == pytest.approx(15 * 1e4 * 0.5293)
    assert canon["Fe"].iat[0] == pytest.approx(5 * 1e4 * 0.7773)      # FeOT empty -> FeO
    assert canon["Ir"].iat[0] == pytest.approx(0.002)
    assert canon["min_age"].iat[0] == pytest.approx(66.0)


def test_parse_georoc_csv_skips_preamble():
    txt = "GEOROC Compilation: Rock Types\nfile generated 2026\n\nUNIQUE_ID,SAMPLE NAME,CITATIONS,ZR(PPM)\n1,a,[1],100\n2,b,[1],200\n"
    df = A.parse_georoc_csv(txt)
    assert list(df.columns) == ["UNIQUE_ID", "SAMPLE NAME", "CITATIONS", "ZR(PPM)"] and len(df) == 2


# ---------------------------------------------------------------------------
# scripted sources: the probe, the paged pull, the end-to-end run
# ---------------------------------------------------------------------------
SGP_KEYS = {"interpreted_age": "interpreted age", "coord_lat": "site latitude", "coord_long": "site longitude",
            "section_name": "section name", "site_type": "site type", "lithology_name": "lithology name",
            "max_age": "max age", "min_age": "min age", "alu": "Al (wt%)", "mo": "Mo (ppm)", "u": "U (ppm)",
            "toc": "TOC (wt%)", "ref_short": "ref short"}
for _e in ("fe", "ti", "mn", "mg", "ca", "na", "k", "p"):
    SGP_KEYS[_e] = f"{_e.capitalize()} (wt%)"
for _e in ("rb", "sr", "y", "zr", "cs", "ba", "la", "ce", "pr", "nd", "sm", "eu", "gd", "tb", "dy", "sc", "v",
           "cr", "co", "ni", "cu", "zn", "ga", "as", "nb", "hf", "ta", "w", "pb", "th", "ho", "er", "tm", "yb",
           "lu", "ir", "pt", "pd", "ru", "rh", "os"):
    SGP_KEYS[_e] = f"{_e.capitalize()} (ppm)"


def _sample_rows(n, seed, *, inject=None):
    """SGP-shaped rows: PAAS-like shale spread over a continuum of ages.

    The background ages are uniform over the Phanerozoic-and-older column, so
    a +/- 1-2 Myr boundary window holds a small minority of the sections --
    which is what the sedimentary record looks like and what makes the
    hypergeometric section test in ``agestack`` informative.  ``inject``
    places named sections at chosen ages, with (``a`` > 0) or without a
    fission component.
    """
    rng = np.random.default_rng(seed)
    X = synth(n, seed=seed)
    rows = []
    for i in range(n):
        c = X[i].copy()
        age = float(rng.uniform(5.0, 800.0))
        sec = f"SEC{i % 25}"
        override = {}
        if inject and i in inject:
            spec = inject[i]
            c = synth(1, a=spec.get("a", 0.0), seed=1000 + i)[0]
            age, sec = spec.get("age", age), spec.get("section", sec)
            override = dict(spec.get("override") or {})
        row = {"sample identifier": f"SGP{i}", "section name": sec, "site type": "outcrop",
               "site latitude": 40.0 + i % 25, "site longitude": -100.0, "interpreted age": age,
               "max age": age + 2, "min age": age - 2, "lithology name": "shale", "ref short": "Test 2020",
               "TOC (wt%)": 1.0}
        for k, e in enumerate(D.elements):
            key = SGP_KEYS.get(e.lower())
            if key:
                row[key] = c[k] / 1e4 if "wt%" in key else c[k]
        for e, v in override.items():                 # may add elements outside the test design (PGE)
            key = SGP_KEYS[e.lower()]
            row[key] = v / 1e4 if "wt%" in key else v
        rows.append(row)
    return rows


class ScriptedSGP:
    """Answers the SGP API like the real one (bundle, post-paged pages)."""

    def __init__(self, rows, *, page_count=50, fail=False):
        self.rows = rows
        self.page_count = page_count
        self.fail = fail
        self.calls = []

    def __call__(self, url, method="GET", json_body=None, params=None, timeout=None, **kw):
        self.calls.append((url, method, json_body, params))
        if self.fail:
            return A.FetchResult(url=url, status=503, content=b"down", method=method)
        if url.endswith("/env.js"):
            return A.FetchResult(url=url, status=200, content=b'window.react_app_env = {REACT_APP_HOST_PATH: "https://sgp-search.io"}')
        if url.endswith("/") and "sgp-search" in url:
            return A.FetchResult(url=url, status=200, content=b'<script src="/static/js/main.abc123.js"></script>')
        if "/static/js/main" in url:
            js = b'fetch("/api/frontend/post-paged"); [{value:"mo",label:"Mo (ppm)"},{label:"Nd (ppm)",value:"nd"},{value:"bogus_code",label:"Bogus"}]'
            return A.FetchResult(url=url, status=200, content=js)
        if "post-paged" in url and method == "POST":
            body = json_body or {}
            if body.get("type") == "nhhxrf":
                return A.FetchResult(url=url, status=200, content=b'{"rows": []}', method=method)
            show = body.get("show") or []
            if any(s not in SGP_KEYS and s not in ("height_meters", "fe_hr_fe_t", "fe_py_fe_hr", "fe_t_al",
                                                    "basin_type", "meta_bin", "strat_name", "strat_name_long",
                                                    "environment_bin") for s in show):
                return A.FetchResult(url=url, status=400, content=b'{"error":"unknown field"}', method=method)
            lo, hi = (body.get("filters") or {}).get("interpreted_age", [0, 1e9])
            keys = {"sample identifier"} | {SGP_KEYS[s] for s in show if s in SGP_KEYS}
            sel = [{k: v for k, v in r.items() if k in keys} for r in self.rows if lo <= r["interpreted age"] <= hi]
            cnt, page = int(body.get("count", 50)), int(body.get("page", 1))
            chunk = sel[(page - 1) * cnt: page * cnt]
            return A.FetchResult(url=url, status=200, content=json.dumps({"rows": chunk, "total": len(sel)}).encode(),
                                 method=method)
        if "restsearchservice" in url:
            p = params or {}
            if p.get("searchtype") == "count":
                return A.FetchResult(url=url, status=200, content=b'{"Count": 0}')
            return A.FetchResult(url=url, status=200, content=b"no results found")
        if "goettingen" in url:
            return A.FetchResult(url=url, status=200, content=b'{"status":"OK","data":{"total_count":0,"items":[]}}')
        return A.FetchResult(url=url, status=404, content=b"")


def _small_conf(tmp_path):
    conf = json.loads(json.dumps(CONF))
    conf["sgp"]["page_count"] = 40
    conf["sgp"]["age_bin_edges_ma"] = [0, 200, 600, 1000]
    conf["sgp"]["show_candidates"] = list(SGP_KEYS) + ["bogus_code"]
    conf["sgp"]["probe_age_window"] = [0, 4000]
    conf["screen"]["min_element_count"] = 10
    conf["screen"]["min_panel_element_count"] = 3
    conf["screen"]["floor_max_rows"] = 200
    conf["screen"]["null_max_rows"] = 100
    conf["earthchem"]["max_rows"] = 100
    return conf


def test_probe_learns_codes_from_bundle_and_trials(tmp_path):
    rows = _sample_rows(60, seed=1)
    fetch = ScriptedSGP(rows)
    conf = _small_conf(tmp_path)
    rep = grave_run(conf, stage="probe", out_dir=tmp_path, fetch_fn=fetch)
    sgp = rep["sources"]["sgp"]
    assert sgp["reached"]
    assert "/api/frontend/post-paged" in sgp["hosts"]["https://sgp-search.io"]["api_paths"]
    assert sgp["accepted_codes"]["mo"] == "Mo (ppm)" and sgp["accepted_codes"]["nd"] == "Nd (ppm)"
    assert "bogus_code" in sgp["rejected_codes"]
    assert sgp["type_variants"]["nhhxrf"]["n_rows"] == 0 and sgp["type_variants"]["samples"]["n_rows"] == 2
    assert "nd" in rep["sgp_show_recommended"]
    assert rep["reached"] == {"sgp": True, "earthchem": False, "georoc": False}


#: Six independent sections carry the injected vector at the K-Pg, two more
#: sections are sampled at the same level and carry nothing (so the cluster
#: test cannot be satisfied by sampling density alone), and one lone section
#: carries it at the end-Permian (the contamination hypothesis).
KPG_INJECT = {3: "SEC_A", 7: "SEC_B", 11: "SEC_C", 14: "SEC_E", 17: "SEC_F", 23: "SEC_G"}


def _cluster_inject(a=1.5):
    inj = {i: {"a": a, "age": 66.0 + 0.1 * (k - 2), "section": s}
           for k, (i, s) in enumerate(KPG_INJECT.items())}
    inj[31] = {"a": 0.0, "age": 66.0, "section": "SEC_BG1"}      # boundary, no vector
    inj[35] = {"a": 0.0, "age": 65.9, "section": "SEC_BG2"}
    inj[20] = {"a": a, "age": 252.0, "section": "SEC_D"}          # one section only
    return inj


def test_end_to_end_recovers_an_injected_cluster_and_flags_single_sections(tmp_path):
    inject = _cluster_inject()
    rows = _sample_rows(300, seed=2, inject=inject)
    fetch = ScriptedSGP(rows)
    conf = _small_conf(tmp_path)
    grave_run(conf, stage="probe,acquire", out_dir=tmp_path, fetch_fn=fetch)
    acq = json.loads((tmp_path / "acquisition.json").read_text())
    assert acq["status"] == "OK" and acq["sources"]["sgp"]["n_rows"] == 300
    assert acq["sources"]["sgp"]["resolution"]["meta"]["age"] == "interpreted age"
    assert "earthchem:NO_DATA_REACHED" in acq["degraded"]
    scr = stage_screen(conf, tmp_path)
    assert scr["status"] == "OK" and scr["funnel"]["survivors"] >= 4
    # both nulls are reported, and the threshold names the one that bound it
    assert scr["control_population_null"]["n"] > 200
    assert scr["threshold_bound_by"] in ("lr_min", "shuffled_null", "control_population")
    assert scr["threshold"] >= CFG.lr_min
    s = stage_assess(conf, tmp_path)
    assert s["verdict"].endswith("FISSION_VECTOR_STRATIGRAPHIC_CLUSTER"), s["verdict"]
    assert s["verdict"].startswith("DEGRADED_SOURCE")
    kpg = s["age_stack"]["boundaries"]["k_pg"]
    assert kpg["status"] == "STRATIGRAPHIC_CLUSTER" and kpg["n_candidate_sections"] >= 4
    assert "SEC_BG1" not in kpg["candidate_sections"] and "SEC_BG2" not in kpg["candidate_sections"]
    assert s["age_stack"]["boundaries"]["end_permian"]["status"] == "single_section"
    ids = {c["sample_id"] for c in s["survivors"]}
    assert len({f"SGP{i}" for i in KPG_INJECT} & ids) >= 4
    assert "SGP20" in ids
    assert (tmp_path / "REPORT.md").exists() and (tmp_path / "samples.csv").exists()


def test_end_to_end_clean_population_is_a_count_not_a_candidate(tmp_path):
    rows = _sample_rows(150, seed=4)
    fetch = ScriptedSGP(rows)
    conf = _small_conf(tmp_path)
    grave_run(conf, stage="probe,acquire,screen,assess", out_dir=tmp_path, fetch_fn=fetch)
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["verdict"].endswith("NO_FISSION_VECTOR") and s["n_survivors"] == 0
    assert s["refined_verdict"] == "NO_REFINED_PARTICULATE"


def test_impact_layer_is_the_positive_control_in_the_age_stack(tmp_path):
    """The K-Pg iridium, rebuilt: five sections with a chondritic PGE panel at
    one level, five more sampled at that level without one.  The channel must
    class the ejecta ``impact`` (never fission) and recover the stratigraphic
    cluster -- on a panel carried by 5 of 130 analyses, which is why the
    refined pass may not be cut on measurement count."""
    chond = {e: R.CI_CHONDRITE[e] * 1e-3 for e in R.PGE}
    inject = {i: {"age": 66.0, "section": f"IMP{i}", "override": chond} for i in (2, 5, 8, 12, 16)}
    inject.update({i: {"age": 66.0, "section": f"BG{i}"} for i in (21, 24, 27, 30, 33)})
    rows = _sample_rows(130, seed=6, inject=inject)
    fetch = ScriptedSGP(rows)
    conf = _small_conf(tmp_path)
    grave_run(conf, stage="probe,acquire,screen,assess", out_dir=tmp_path, fetch_fn=fetch)
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["verdict"].endswith("NO_FISSION_VECTOR")
    assert s["refined_counts"]["pge"].get("impact") == 5
    kpg = s["impact_positive_control"]["boundaries"]["k_pg"]
    assert kpg["status"] == "STRATIGRAPHIC_CLUSTER", kpg
    assert kpg["n_candidate_sections"] == 5 and kpg["n_sections"] >= 10


def test_every_source_empty_is_no_data_reached(tmp_path):
    fetch = ScriptedSGP([], fail=True)
    conf = _small_conf(tmp_path)
    grave_run(conf, stage="all", out_dir=tmp_path, fetch_fn=fetch)
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["verdict"] == "NO_DATA_REACHED" and s["n_samples"] == 0
    p = json.loads((tmp_path / "probe.json").read_text())
    assert p["verdict"] == "NO_DATA_REACHED"
    assert "ACCESS statement" in s["note"]
