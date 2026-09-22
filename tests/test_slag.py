"""Offline test suite for SLAG-WD (S51) --- polluted white dwarfs beyond the natural family.

No network anywhere (``conftest.py`` raises on any socket).  Per
``docs/channel-brief.md`` §5 the suite:

* recovers an injected refinery vector (a Ti excess on an otherwise natural
  panel) through Tier 1 (calibrated misfit p) AND Tier 2 (z_pair > 4, rest
  natural, no kill) and returns z = 0 / a natural p on the same panel without
  the injection;
* shows the dominant confounders --- a declining-phase natural panel, a
  crustal parcel, a core parcel, a Be-spallation object with its Li partner
  --- come out natural;
* trips every kill (INFORMATION_LIMITED, COOL_HE_LINE_PHYSICS,
  CARBONATE_CRUST_POSSIBLE, MULTI_REFERENCE_DISAGREEMENT, REST_OF_PANEL_NOT_NATURAL);
* checks column-role resolution on VizieR-style and GitHub-style headers,
  atmosphere resolution from the spectral type, limits and assumed errors;
* degrades honestly: a dead archive gives NO_DATA_REACHED with the failures
  recorded, never a candidate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seti.slag import acquire as acq
from seti.slag.family import (
    TC_FRACTIONATION_OVERRIDE_K,
    fractionation_factor,
    load_family,
    ratio_envelope,
)
from seti.slag.misfit import FitSettings, Panel, calibrate_misfit, fit_panel
from seti.slag.pairs import (
    FLAG_ALLOY,
    FLAG_BE,
    FLAG_FE_MG,
    FLAG_PURE_SI,
    KILL_CARBONATE,
    KILL_COOL_HE,
    KILL_INFORMATION_LIMITED,
    KILL_MULTI_REF,
    KILL_REST_NOT_NATURAL,
    apply_kills,
    fit_two_parcel,
    pair_residuals,
    refinery_flags,
)
from seti.slag.run import (
    VERDICT_CANDIDATE,
    VERDICT_LIST,
    VERDICT_NO_DATA,
    load_slag_config,
    stage_acquire,
    stage_assess,
    stage_probe,
    stage_screen,
)
from seti.slag.sinking import (
    PHASE_DECLINING,
    PHASE_EARLY,
    PHASE_STEADY,
    TimescaleModel,
    parse_timescale_table,
    phase_label,
    phase_log_factor,
)

FAM = load_family()
TSM = TimescaleModel(FAM)
FAST = FitSettings(n_random=250, n_refine=1, refine_maxiter=300, t_cut_range=(0.0, 1400.0))
ELS9 = ["Mg", "Al", "Si", "Ca", "Ti", "Cr", "Fe", "Ni", "Na"]


# ---------------------------------------------------------------------------
# synthetic panels
# ---------------------------------------------------------------------------
def natural_panel(endmember="mantle_BSE", elements=ELS9, atm="He", teff=15000.0, t_acc=10.0,
                  t_dec=0.0, err=0.1, seed=1, name="nat", **kw) -> Panel:
    X = FAM.column(endmember)
    lt, _, _ = TSM.log_tau_rel(elements, atm, teff)
    y = np.log10(np.array([X[FAM.index(e)] for e in elements])) \
        + phase_log_factor(lt, t_acc, t_dec) - 6.0
    rng = np.random.default_rng(seed)
    errs = np.full(len(elements), err)
    return Panel(name, list(elements), y + rng.normal(size=len(elements)) * errs, errs, atm, teff,
                 8.0, **kw)


def with_excess(p: Panel, element: str, dex: float) -> Panel:
    q = Panel(p.name + f"+{element}", list(p.elements), p.values.copy(), p.errors.copy(),
              p.atmosphere, p.teff, p.logg, p.reference, list(p.limit_elements),
              p.limit_values.copy(), dict(p.meta))
    q.values[q.elements.index(element)] += dex
    return q


# ---------------------------------------------------------------------------
# family and sinking
# ---------------------------------------------------------------------------
def test_family_loads_with_presence_mask_and_oxygen_override():
    assert "core" in FAM.endmembers and "water_ice" in FAM.endmembers
    assert not FAM.has("Mg", "core") and FAM.has("Fe", "core")
    assert not FAM.has("Ti", "water_ice") and FAM.has("O", "water_ice")
    assert FAM.tc(["O"])[0] == TC_FRACTIONATION_OVERRIDE_K["O"]
    assert FAM.Tc[FAM.index("O")] == 180.0
    np.testing.assert_allclose(FAM.number.sum(axis=0), 1.0)


def test_orthogonal_pairs_are_tight_and_the_open_ones_are_reported_wide():
    tight = {("Ti", "Al"): 0.4, ("Ca", "Al"): 0.9, ("Sc", "Ca"): 0.9}
    for (a, b), w in tight.items():
        e = ratio_envelope(FAM, a, b, t_cut_max=1400.0)
        assert e["width_dex"] < w, (a, b, e)
        assert "water_ice" not in e["per_endmember"] and "core" not in e["per_endmember"]
    # crust and core-formation open these up: the brief's 0.3 dex is not what the compilations say
    for a, b in (("Mn", "Cr"), ("Ni", "Co"), ("Sr", "Ca")):
        assert ratio_envelope(FAM, a, b, t_cut_max=1400.0)["width_dex"] > 1.0


def test_fractionation_depletes_volatiles_only():
    f = fractionation_factor(np.array([600.0, 1000.0, 1600.0]), t_cut=1000.0, depth_dex=2.0)
    assert f[0] < 0.02 and 0.05 < f[1] < 0.2 and f[2] > 0.98


def test_phase_factor_limits():
    lt = np.array([0.2, 0.0, -0.3])
    early = phase_log_factor(lt, 1e-4, 0.0)
    np.testing.assert_allclose(early, 0.0, atol=1e-3)          # the parcel itself
    steady = phase_log_factor(lt, 1e3, 0.0)
    np.testing.assert_allclose(steady, lt - lt.mean(), atol=1e-6)  # x tau_rel
    dec = phase_log_factor(lt, 1e3, 2.0)
    assert dec[2] - dec[0] < steady[2] - steady[0]             # the heavy one keeps sinking
    assert phase_label(0.05, 0.0) == PHASE_EARLY
    assert phase_label(10.0, 0.0) == PHASE_STEADY
    assert phase_label(10.0, 1.0) == PHASE_DECLINING


def test_timescale_scatter_is_tighter_for_the_tabulated_block():
    _, sig, src = TSM.log_tau_rel(["Mg", "Fe", "Be", "Sr"], "He", 12000.0)
    assert src == "embedded_mass_scaling"
    assert sig[0] == sig[1] < sig[2] == sig[3]


def test_parse_timescale_table_reads_a_koester_like_block():
    text = "# tau in yr\nTeff logg Ca Mg Fe Si\n10000 8.0 1e5 1.2e5 8e4 1.1e5\n20000 8.0 3e4 3.5e4 2.5e4 3.2e4\n"
    tab = parse_timescale_table(text)
    assert tab is not None and list(tab.columns) == ["Teff", "logg", "Ca", "Mg", "Fe", "Si"]
    tsm = TimescaleModel(FAM, tables={"He": tab})
    lt, sig, src = tsm.log_tau_rel(["Mg", "Fe"], "He", 15000.0)
    assert src == "fetched_table" and lt[0] > 0 > lt[1]
    assert parse_timescale_table("nothing here\n1 2 3\n") is None


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------
def test_natural_panels_fit_with_low_misfit():
    for em, t_acc, t_dec in (("mantle_BSE", 10.0, 0.0), ("CI", 0.05, 0.0), ("crust_cont", 10.0, 1.5),
                              ("core", 10.0, 0.0)):
        els = [e for e in ELS9 if FAM.has(e, em)]
        p = natural_panel(em, els, t_acc=t_acc, t_dec=t_dec, seed=3)
        f = fit_panel(FAM, p, TSM, FAST)
        assert f.chi2 / max(f.n_measured - 1, 1) < 2.5, (em, f.chi2)


def test_injected_excess_is_recovered_by_the_calibrated_misfit():
    p = natural_panel(seed=5)
    q = with_excess(p, "Ti", 1.5)
    f0 = fit_panel(FAM, p, TSM, FAST)
    f1 = fit_panel(FAM, q, TSM, FAST)
    assert f1.chi2 > f0.chi2 + 15
    c0 = calibrate_misfit(FAM, p, TSM, f0, FAST, n_draws=25)
    c1 = calibrate_misfit(FAM, q, TSM, f1, FAST, n_draws=25)
    assert c0["p_misfit"] > 0.2
    assert c1["p_misfit"] < 0.1
    assert c1["draw_mode"] == "posterior" and c1["n_draws"] == 25


def test_upper_limits_penalise_only_when_violated():
    p = natural_panel(seed=2)
    ok = Panel(p.name, p.elements, p.values, p.errors, p.atmosphere, p.teff, 8.0,
               limit_elements=["Sc"], limit_values=np.array([0.0]))       # a limit far above
    bad = Panel(p.name, p.elements, p.values, p.errors, p.atmosphere, p.teff, 8.0,
                limit_elements=["Sc"], limit_values=np.array([-20.0]))    # impossible limit
    f_ok, f_bad = fit_panel(FAM, ok, TSM, FAST), fit_panel(FAM, bad, TSM, FAST)
    assert f_ok.limit_violation < 1e-6
    assert f_bad.limit_violation > 10


# ---------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------
def test_pair_residual_recovers_injected_ti_and_is_zero_on_natural():
    p = natural_panel(seed=7)
    nat = pair_residuals(FAM, TSM, p, FAST)
    assert {r["pair"] for r in nat} == {"Ti/Al", "Ca/Al"}
    assert all(r["z_pair"] == 0.0 and not r["exceeds"] for r in nat)
    q = with_excess(p, "Ti", 1.5)
    res = {r["pair"]: r for r in pair_residuals(FAM, TSM, q, FAST)}
    assert res["Ti/Al"]["exceeds"] and res["Ti/Al"]["z_pair"] > 4
    assert res["Ti/Al"]["rest_natural"] is True
    assert res["Ti/Al"]["phase_constraint"] == "rest_of_panel_fit"
    two = fit_two_parcel(FAM, q, TSM, FAST)
    ff = fit_panel(FAM, q, TSM, FitSettings(n_random=250, n_refine=1, refine_maxiter=300))
    fr = fit_panel(FAM, q, TSM, FAST)
    kills = apply_kills(q, res["Ti/Al"], ff, fr, None, two)
    assert kills == [], kills


def test_declining_phase_natural_panel_does_not_trigger_the_pair_test():
    p = natural_panel("CI", t_acc=10.0, t_dec=2.5, seed=11)
    res = pair_residuals(FAM, TSM, p, FAST)
    assert all(not r["exceeds"] for r in res), res


def test_pure_si_flag_fires_only_with_o_mg_fe_all_depleted():
    ps = Panel("puresi", ["Si", "Ca", "Fe"], np.array([-5.0, -8.5, -8.0]), np.full(3, 0.1), "He",
               12000.0, 8.0, limit_elements=["Mg", "O"], limit_values=np.array([-8.0, -6.0]))
    fl = {f["flag"]: f for f in refinery_flags(FAM, TSM, ps, FAST)}
    assert fl[FLAG_PURE_SI]["fired"]
    ps2 = Panel("rock", ["Si", "Ca", "Fe", "Mg"], np.array([-5.0, -6.0, -5.2, -5.0]),
                np.full(4, 0.1), "He", 12000.0, 8.0, limit_elements=["O"], limit_values=np.array([-4.0]))
    fl2 = {f["flag"]: f for f in refinery_flags(FAM, TSM, ps2, FAST)}
    assert not fl2[FLAG_PURE_SI]["fired"]


def test_be_flag_needs_the_spallation_partners_absent():
    base = dict(elements=["Be", "Ca", "Mg", "Fe", "Si"], values=np.array([-7.0, -6.5, -5.7, -5.8, -5.6]),
                errors=np.full(5, 0.1), atmosphere="H", teff=12000.0, logg=8.0)
    no_li = Panel("be", **base, limit_elements=["Li"], limit_values=np.array([-9.5]))
    fl = {f["flag"]: f for f in refinery_flags(FAM, TSM, no_li, FAST)}
    assert fl[FLAG_BE]["fired"]
    with_li = Panel("be_li", elements=base["elements"] + ["Li"],
                    values=np.append(base["values"], -6.5), errors=np.full(6, 0.1),
                    atmosphere="H", teff=12000.0, logg=8.0)
    fl2 = {f["flag"]: f for f in refinery_flags(FAM, TSM, with_li, FAST)}
    assert not fl2[FLAG_BE]["fired"] and "spallation" in fl2[FLAG_BE]["note"]


def test_alloy_and_fe_mg_flags():
    alloy = Panel("alloy", ["Ti", "Al", "V", "Fe", "Mg", "Si"],
                  np.array([-5.5, -5.0, -6.0, -8.5, -7.5, -7.5]), np.full(6, 0.1), "He", 15000.0, 8.0)
    fl = {f["flag"]: f for f in refinery_flags(FAM, TSM, alloy, FAST)}
    assert fl[FLAG_ALLOY]["fired"] and fl[FLAG_ALLOY]["has_V"]
    nat = natural_panel("crust_cont", ["Ti", "Al", "V", "Fe", "Mg", "Si"], seed=4)
    fln = {f["flag"]: f for f in refinery_flags(FAM, TSM, nat, FAST)}
    assert not fln[FLAG_ALLOY]["fired"]
    femg = Panel("nofe", ["Mg", "Si", "Ca", "Al", "Fe"], np.array([-5.0, -5.1, -6.2, -6.4, -8.5]),
                 np.full(5, 0.1), "He", 15000.0, 8.0)
    flf = {f["flag"]: f for f in refinery_flags(FAM, TSM, femg, FAST)}
    assert flf[FLAG_FE_MG]["fired"] and flf[FLAG_FE_MG]["caveat"] == "EXOTIC_MANTLE_POSSIBLE"
    bse = natural_panel("mantle_BSE", ["Mg", "Si", "Ca", "Al", "Fe"], seed=6)
    assert not {f["flag"]: f for f in refinery_flags(FAM, TSM, bse, FAST)}[FLAG_FE_MG]["fired"]


def test_every_kill_trips():
    p = natural_panel(seed=9, atm="He", teff=5500.0)
    q = with_excess(p, "Ca", 1.2)
    res = {r["pair"]: r for r in pair_residuals(FAM, TSM, q, FAST)}
    pair = res["Ca/Al"]
    fr = fit_panel(FAM, q, TSM, FAST)
    kills = apply_kills(q, pair, None, fr, None, None)
    assert KILL_COOL_HE in kills
    small = Panel("small", ["Ca", "Al", "Mg", "Fe"], q.values[[3, 1, 0, 6]], q.errors[:4], "He",
                  15000.0, 8.0)
    assert KILL_INFORMATION_LIMITED in apply_kills(small, pair, None, fr, None, None)
    carb = Panel("carb", q.elements + ["C"], np.append(q.values, q.values[3] + 0.2),
                 np.append(q.errors, 0.1), "He", 15000.0, 8.0)
    assert KILL_CARBONATE in apply_kills(carb, pair, None, fr, None, None)
    other = with_excess(p, "Ca", -1.0)
    other.teff = 15000.0
    hot = Panel("hot", q.elements, q.values, q.errors, "He", 15000.0, 8.0)
    assert KILL_MULTI_REF in apply_kills(hot, pair, None, fr, None, None, other_panels=[other])
    bad_rest = dict(pair, rest_p_naive=1e-4)
    assert KILL_REST_NOT_NATURAL in apply_kills(hot, bad_rest, None, fr, None, None)


# ---------------------------------------------------------------------------
# acquisition: roles, atmosphere, panels
# ---------------------------------------------------------------------------
VIZIER_COLS = ["Name", "RAJ2000", "DEJ2000", "SpType", "Teff", "e_Teff", "logg", "logCa", "e_logCa",
               "l_logCa", "logMg", "e_logMg", "l_logMg", "logFe", "e_logFe", "l_logFe", "logSi",
               "e_logSi", "logTi", "e_logTi", "l_logTi", "logAl", "e_logAl", "logSc", "e_logSc", "Ref"]
VIZIER_DESC = {"logCa": "log(Ca/H(e)) abundance", "logMg": "log(Mg/H(e)) abundance",
               "logFe": "log(Fe/H(e))", "logSi": "log(Si/H(e))", "logTi": "log(Ti/H(e))",
               "logAl": "log(Al/H(e))", "logSc": "log(Sc/H(e))", "Ref": "Reference (bibcode)",
               "Name": "White dwarf name", "SpType": "Spectral type"}


def test_resolve_roles_on_vizier_style_columns():
    roles = acq.resolve_roles(VIZIER_COLS, ["Ca", "Mg", "Fe", "Si", "Ti", "Al", "Sc", "C", "Na"],
                              descriptions=VIZIER_DESC)
    assert roles["name"] == "Name" and roles["spt"] == "SpType" and roles["teff"] == "Teff"
    assert roles["logg"] == "logg" and roles["ref"] == "Ref"
    assert roles["elements"]["Ca"] == {"value": "logCa", "error": "e_logCa", "limit": "l_logCa",
                                       "unit": "", "description": "log(Ca/H(e)) abundance"}
    assert roles["elements"]["Si"]["limit"] is None
    assert "C" not in roles["elements"] and "Na" not in roles["elements"]   # never guessed
    assert roles["reference_convention"] == ["H(e)"]
    assert roles["n_elements_resolved"] == 7


def test_resolve_roles_on_github_style_columns():
    cols = ["WD Name", "Spectral Type", "Teff", "log g", "Reference", "Ca", "Ca error",
            "Ca upper limit", "Fe", "Fe error", "Mg", "Mg error", "Ni", "Ni error", "Co", "Co error"]
    roles = acq.resolve_roles(cols, ["Ca", "Fe", "Mg", "Ni", "Co", "Sc"])
    assert roles["elements"]["Ca"]["value"] == "Ca" and roles["elements"]["Ca"]["error"] == "Ca error"
    assert roles["elements"]["Ca"]["limit"] == "Ca upper limit"
    assert roles["elements"]["Co"]["error"] == "Co error"
    assert roles["name"] == "WD Name" and roles["logg"] == "log g" and roles["spt"] == "Spectral Type"
    assert "Sc" not in roles["elements"]


def test_atmosphere_from_spectral_type():
    assert acq.atmosphere_from_spt("DAZ") == "H"
    assert acq.atmosphere_from_spt("DBAZ") == "He"
    assert acq.atmosphere_from_spt("DZA") == "He"
    assert acq.atmosphere_from_spt("DAB") == "H"
    assert acq.atmosphere_from_spt("DQ") == "He"
    assert acq.atmosphere_from_spt("") == "unknown"
    assert acq.atmosphere_from_spt("WD") == "unknown"


def synthetic_table() -> pd.DataFrame:
    rows = []
    def row(name, spt, teff, ref, vals, limits=(), errs=0.1):
        r = {"Name": name, "RAJ2000": 0.0, "DEJ2000": 0.0, "SpType": spt, "Teff": teff, "e_Teff": 100,
             "logg": 8.0, "Ref": ref}
        for el, v in vals.items():
            r[f"log{el}"] = v
            r[f"e_log{el}"] = errs
            if f"l_log{el}" in VIZIER_COLS:
                r[f"l_log{el}"] = ""
        for el, v in limits:
            r[f"log{el}"] = v
            r[f"l_log{el}"] = "<"
            r[f"e_log{el}"] = np.nan
        rows.append(r)
    els = ["Mg", "Al", "Si", "Ca", "Ti", "Fe", "Sc"]
    for k, em in enumerate(["mantle_BSE", "CI", "H", "crust_cont"]):
        p = natural_panel(em, els, seed=20 + k)
        row(f"WD NAT{k}", "DBZ", 15000.0, "2020A&A...1..1A", dict(zip(els, p.values, strict=True)))
    # the refinery: a Ti excess on a natural panel, from two sources that agree
    p = with_excess(natural_panel("mantle_BSE", els, seed=31), "Ti", 1.5)
    row("GD 362", "DBZ", 15000.0, "2020A&A...1..1A", dict(zip(els, p.values, strict=True)))
    row("GD 362", "DBZ", 15000.0, "2021ApJ...1..1B", dict(zip(els, p.values + 0.05, strict=True)))
    # information limited: three elements plus a limit
    row("WD SMALL", "DAZ", 10000.0, "2019MNRAS.1..1C", {"Ca": -8.0, "Mg": -7.2, "Fe": -7.4},
        limits=[("Ti", -9.5)])
    # a row with no error column values (assumed) and a cool He Ca-high object
    q = with_excess(natural_panel("CI", els, seed=40, atm="He", teff=5000.0), "Ca", 2.2)
    row("LHS 2534", "DZ", 5000.0, "2024ApJ...1..1K", dict(zip(els, q.values, strict=True)), errs=np.nan)
    df = pd.DataFrame(rows)
    for c in VIZIER_COLS:
        if c not in df:
            df[c] = np.nan
    return df[VIZIER_COLS]


def test_build_panels_reads_limits_assumed_errors_and_atmospheres():
    df = synthetic_table()
    roles = acq.resolve_roles(list(df.columns), ["Mg", "Al", "Si", "Ca", "Ti", "Fe", "Sc"],
                              descriptions=VIZIER_DESC)
    panels, diag = acq.build_panels(df, roles, default_error_dex=0.2)
    assert len(panels) == 8
    small = next(p for p in panels if p.name == "WD SMALL")
    assert small.n_measured == 3 and small.limit_elements == ["Ti"] and small.atmosphere == "H"
    assert small.meta["atmosphere_how"] == "spectral_type"
    lhs = next(p for p in panels if p.name == "LHS 2534")
    assert lhs.atmosphere == "He" and set(lhs.meta["errors_assumed_for"]) == set(lhs.elements)
    assert np.all(lhs.errors == 0.2)
    assert acq.normalise_name("GD 362") == acq.normalise_name("gd362") == "GD362"


def test_pick_pewdd_csv_and_timescale_files():
    entries = [{"path": "README.md", "size": 10, "type": "blob"},
               {"path": "data/PEWDD.csv", "size": 500000, "type": "blob"},
               {"path": "data/other.csv", "size": 100, "type": "blob"},
               {"path": "timescales/DB_diffusion_timescales.dat", "size": 2000, "type": "blob"},
               {"path": "src/timescale_interpolator.py", "size": 2000, "type": "blob"}]
    assert [e["path"] for e in acq.pick_pewdd_csv(entries)] == ["data/PEWDD.csv", "data/other.csv"]
    assert [e["path"] for e in acq.pick_timescale_files(entries)] == ["timescales/DB_diffusion_timescales.dat"]


# ---------------------------------------------------------------------------
# end to end, offline
# ---------------------------------------------------------------------------
def _cfg(tmp_path: Path) -> dict:
    cfg = load_slag_config()
    cfg["fit"].update(n_random=250, n_refine=1, refine_maxiter=300, n_cal=20)
    cfg["run"]["results_dir"] = str(tmp_path)
    return cfg


def test_end_to_end_recovers_the_injected_refinery_and_lands_the_controls(tmp_path):
    cfg = _cfg(tmp_path)
    df = synthetic_table()
    csv = tmp_path / "table.csv"
    df.to_csv(csv, index=False)
    s1 = stage_screen(cfg, tmp_path, shard="1/2", input_csv=str(csv))
    s2 = stage_screen(cfg, tmp_path, shard="2/2", input_csv=str(csv))
    assert s1["status"] == "OK" and s2["status"] == "OK"
    assert s1["n_panels"] + s2["n_panels"] == 8
    summ = stage_assess(cfg, tmp_path)
    assert summ["verdict"] == VERDICT_CANDIDATE
    f = summ["funnel"]
    assert f["panels_total"] == 8 and f["objects_total"] == 7
    assert f["panels_information_limited"] == 1
    assert f["candidates_surviving"] >= 1
    surv = {(c["name"], c["what"]) for c in summ["survivors"]}
    assert ("GD 362", "Ti/Al") in surv
    # the cool He Ca-high object is killed, not a survivor
    assert all(c["name"] != "LHS 2534" for c in summ["survivors"])
    cand = pd.read_csv(tmp_path / "candidates.csv")
    lhs = cand[cand["name"] == "LHS 2534"]
    assert len(lhs) and all(KILL_COOL_HE in str(k) for k in lhs["kills"])
    controls = {c["name"]: c for c in json.loads((tmp_path / "controls.json").read_text())["controls"]}
    assert controls["GD 362"]["status"] == "FOUND" and controls["GD 362"]["n_sources"] == 2
    assert controls["LHS 2534"]["status"] == "FOUND"
    assert controls["PG 1225-079"]["status"] == "CONTROL_NOT_FOUND"
    ml = pd.read_csv(tmp_path / "misfit_list.csv")
    assert set(ml.iloc[:2]["name"]) == {"GD 362", "LHS 2534"}   # the injected objects top the list
    assert "WD SMALL" not in set(ml["name"])         # never in the calibrated list
    assert (tmp_path / "pairs.csv").exists() and (tmp_path / "flags.csv").exists()


def test_end_to_end_without_the_injection_is_a_list_not_a_candidate(tmp_path):
    cfg = _cfg(tmp_path)
    df = synthetic_table()
    df = df[~df["Name"].isin(["GD 362", "LHS 2534"])]
    csv = tmp_path / "table.csv"
    df.to_csv(csv, index=False)
    stage_screen(cfg, tmp_path, input_csv=str(csv))
    summ = stage_assess(cfg, tmp_path)
    assert summ["verdict"] == VERDICT_LIST
    assert summ["funnel"]["candidates_surviving"] == 0
    assert summ["funnel"]["misfit_unexplained"] == 0


def test_dead_archive_degrades_to_no_data(tmp_path):
    cfg = _cfg(tmp_path)

    def dead(url):
        raise ConnectionError(f"dead: {url}")

    def dead_q(adql):
        raise RuntimeError("TAP down")

    def dead_tap(adql, url):
        raise RuntimeError("TAP down")

    pr = stage_probe(cfg, tmp_path, fetch_fn=dead, query_fn=dead_q)
    assert pr["pewdd"]["status"] == "QUERY_FAILED" and pr["pewdd"]["roles"] is None
    assert pr["acquisition"]["any_query_failed"]
    aq = stage_acquire(cfg, tmp_path, fetch_fn=dead, tap_fn=dead_tap)
    assert aq["status"] == "QUERY_FAILED" and aq["pewdd"]["n_rows"] == 0
    assert aq["pewdd"]["routes_tried"][0]["served_by"] == "none"
    sc = stage_screen(cfg, tmp_path)
    assert sc["status"] == "QUERY_FAILED" and sc["panels"] == []
    summ = stage_assess(cfg, tmp_path)
    assert summ["verdict"] == VERDICT_NO_DATA
    assert summ["funnel"]["candidates_surviving"] == 0
    assert "acquisition:QUERY_FAILED" in summ["degraded"]
    log = json.loads((tmp_path / "acquisition_log.json").read_text())
    assert log["n_query_failed"] >= 1


def test_probe_resolves_roles_from_served_metadata(tmp_path):
    cfg = _cfg(tmp_path)
    meta = ("#Column\tName\t(a20)\tWhite dwarf name\t[ucd=meta.id]\n"
            "#Column\tSpType\t(a6)\tSpectral type\t[ucd=src.spType]\n"
            "#Column\tTeff\t(I5)\tEffective temperature\t[ucd=phys.temperature]\n"
            "#Column\tlogCa\t(F6.2)\tlog(Ca/H(e)) abundance\t[ucd=phys.abund]\n"
            "#Column\te_logCa\t(F5.2)\tError on logCa\t[ucd=stat.error]\n"
            "#Column\tlogMg\t(F6.2)\tlog(Mg/H(e)) abundance\t[ucd=phys.abund]\n"
            "#Column\tlogFe\t(F6.2)\tlog(Fe/H(e)) abundance\t[ucd=phys.abund]\n"
            "#Column\tRef\t(a19)\tReference bibcode\t[ucd=meta.bib]\n"
            "Name\tSpType\tTeff\tlogCa\te_logCa\tlogMg\tlogFe\tRef\n"
            "\t\tK\t[-]\t[-]\t[-]\t[-]\t\n"
            "----\t----\t----\t----\t----\t----\t----\t----\n"
            "GD 362\tDBZ\t10500\t-6.24\t0.10\t-5.98\t-5.65\t2007ApJ...1..1Z\n")

    def fetch(url):
        if "api.github.com" in url:
            return json.dumps({"tree": [{"path": "PEWDD.csv", "size": 100, "type": "blob"}]})
        if "asu-tsv" in url:
            return meta
        raise ConnectionError(url)

    def q(adql):
        raise RuntimeError("TAP down")

    pr = stage_probe(cfg, tmp_path, fetch_fn=fetch, query_fn=q)
    roles = pr["pewdd"]["roles"]
    assert pr["pewdd"]["status"] == "OK"
    assert set(roles["elements"]) == {"Ca", "Mg", "Fe"}
    assert roles["elements"]["Ca"]["error"] == "e_logCa"
    assert roles["reference_convention"] == ["H(e)"]
    assert pr["pewdd"]["github"]["csv_files"][0]["path"] == "PEWDD.csv"
    assert "Ti/Al" in pr["pair_envelopes_restricted"]


@pytest.mark.parametrize("atm,teff", [("He", 15000.0), ("H", 12000.0)])
def test_screen_panel_records_are_json_serialisable(tmp_path, atm, teff):
    from seti.slag.run import screen_panel
    cfg = _cfg(tmp_path)
    p = natural_panel(atm=atm, teff=teff, seed=13)
    rec = screen_panel(FAM, TSM, p, cfg, n_cal=10)
    json.dumps(rec, default=str)
    assert rec["status"] == "SCREENED" and rec["misfit_class"] in ("NATURAL", "WATCH", "UNEXPLAINED")
