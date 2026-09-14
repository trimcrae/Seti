"""Offline test suite for ARC --- superflares above the starspot energy ceiling.

No network anywhere (``conftest.py`` raises on any socket).  Per
``docs/channel-brief.md`` §5 the suite:

* recovers an injected signal: a star with a 1 % rotational amplitude and two
  independent flares at 10x the f = 1 bound has xi > 0 and is a candidate once
  the Gaia context answers and every veto passes;
* returns a clean null on the confounder: the same energies on a 5 %
  amplitude star sit below the conservative ceiling (watch at most);
* trips every rejection rule: RUWE = 2 -> companion_suspect, logg < 4 ->
  evolved, a bright 12" neighbour -> blend, a Gaia ECL class -> pulsator_or_eb,
  a catalogue flag -> catalogue_doubtful;
* degrades honestly: a failed or empty VizieR yields NO_DATA_REACHED, never a
  candidate;
* checks the Berdyugina and spot-area formulas against hand computations.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seti.arc import acquire as acq
from seti.arc.ceiling import (
    R_SUN_CM,
    energy_to_bolometric,
    independent_flares,
    magnetic_energy,
    normalise_amplitude,
    spot_area,
    spot_contrast,
    spot_temperature,
    star_ceiling,
    xi,
)
from seti.arc.run import (
    VERDICT_CANDIDATES,
    VERDICT_NO_DATA,
    VERDICT_NONE,
    arc_run,
    build_star_context,
    load_arc_config,
    screen_catalogue,
    stage_acquire,
    stage_assess,
    stage_probe,
    stage_screen,
)
from seti.arc.vet import (
    HARD_VETO_ORDER,
    REPORT_FLAGS,
    assign_tiers,
    blend,
    catalogue_doubtful,
    companion_suspect,
    evolved,
    funnel,
    kepler_quarter_of,
    pulsator_or_eb,
    rejection_counters,
    vet_star,
)

T_SUN = 5772.0


# ---------------------------------------------------------------------------
# the physics against hand computations
# ---------------------------------------------------------------------------
def test_berdyugina_spot_temperature_by_hand():
    # dT = 3.58e-5 T^2 + 0.249 T - 808 at T = 5772 K
    dt = 3.58e-5 * T_SUN ** 2 + 0.249 * T_SUN - 808.0
    assert math.isclose(dt, 1821.87, rel_tol=1e-4)
    assert math.isclose(float(spot_temperature(T_SUN)), T_SUN - dt, rel_tol=1e-9)
    # a cooler K dwarf has a smaller contrast, an F star a larger one
    assert float(spot_temperature(4500.0)) > 4500.0 - dt
    assert float(spot_temperature(6500.0)) < 6500.0 - dt


def test_spot_area_and_bound_by_hand():
    ts = T_SUN - (3.58e-5 * T_SUN ** 2 + 0.249 * T_SUN - 808.0)
    contrast = 1.0 - (ts / T_SUN) ** 4
    assert math.isclose(float(spot_contrast(T_SUN)), contrast, rel_tol=1e-12)
    a = 0.01 * math.pi * R_SUN_CM ** 2 / contrast
    assert math.isclose(float(spot_area(0.01, 1.0, T_SUN)), a, rel_tol=1e-12)
    assert math.isclose(float(spot_area(0.01, 1.0, T_SUN, geometric_factor=3.0)), 3 * a,
                        rel_tol=1e-12)
    # ~2e20 cm^2 for a 1 % amplitude on the Sun: a large spot group, not a monster
    assert 1.5e20 < a < 2.5e20
    e = 3000.0 ** 2 / (8 * math.pi) * a ** 1.5
    assert math.isclose(float(magnetic_energy(a)), e, rel_tol=1e-12)
    assert math.isclose(float(magnetic_energy(a, f=0.1)), 0.1 * e, rel_tol=1e-12)
    # Shibayama+2013 eq. 1 normalisation: 7e32 erg (f/0.1)(B/1 kG)^2 (A/3e19 cm^2)^1.5
    ref = 7e32 * (0.1 / 0.1) * (1000.0 / 1000.0) ** 2 * (3e19 / 3e19) ** 1.5
    assert math.isclose(float(magnetic_energy(3e19, b_gauss=1000.0, f=0.1)), ref, rel_tol=0.1)
    # xi is a log residual: E = 10 E_mag -> 1
    assert math.isclose(float(xi(10 * e, e)), 1.0, abs_tol=1e-12)
    assert np.isnan(float(xi(-1.0, e)))


def test_amplitude_units_are_normalised_and_the_guess_is_reported():
    v, u = normalise_amplitude([8000.0, 12000.0], "auto")
    assert u == "ppm" and np.allclose(v, [0.008, 0.012])
    v, u = normalise_amplitude([1.0, 3.0], "auto")
    assert u == "percent" and np.allclose(v, [0.01, 0.03])
    v, u = normalise_amplitude([0.01, 0.03], "auto")
    assert u == "fraction" and np.allclose(v, [0.01, 0.03])
    v, u = normalise_amplitude([0.5], "ppm")
    assert u == "ppm" and np.isclose(v[0], 5e-7)
    v, u = normalise_amplitude([10.0], "mmag")
    assert u == "mmag" and np.isclose(v[0], 1 - 10 ** (-0.004))


def test_energy_conversion_detects_log_columns_and_handles_equivalent_duration():
    e, info = energy_to_bolometric([34.0, 35.0], kind="bolometric")
    assert info["log10_input"] and np.allclose(e, [1e34, 1e35])
    e, info = energy_to_bolometric([1e34, 1e35], kind="bolometric", factor=2.0)
    assert not info["log10_input"] and np.allclose(e, [2e34, 2e35])
    # ED = 100 s on the Sun in a band carrying 35 % of L_bol
    e, info = energy_to_bolometric([100.0], kind="equivalent_duration", radius_rsun=1.0,
                                   t_star_k=T_SUN, band_fraction=0.35, log10=False)
    lbol = 4 * math.pi * R_SUN_CM ** 2 * 5.670374e-5 * T_SUN ** 4
    assert math.isclose(lbol, 3.83e33, rel_tol=0.01)
    assert math.isclose(float(e[0]), 100.0 * lbol * 0.35, rel_tol=1e-9)
    with pytest.raises(ValueError):
        energy_to_bolometric([100.0], kind="equivalent_duration")


def test_independent_flares_cluster_within_the_gap():
    t = [10.0, 10.1, 10.3, 20.0, 20.4, 40.0]
    e = [1e34, 5e34, 2e34, 1e33, 3e33, 7e33]
    ee, tt = independent_flares(t, e, gap_days=0.5)
    assert list(ee) == [5e34, 3e33, 7e33] and list(tt) == [10.0, 20.0, 40.0]
    ee, tt = independent_flares(None, e)
    assert len(ee) == 6 and np.isnan(tt).all()


def _bound(amp: float, geom: float = 3.0, teff: float = T_SUN, rad: float = 1.0) -> float:
    return float(magnetic_energy(spot_area(amp, rad, teff, geometric_factor=geom)))


# ---------------------------------------------------------------------------
# injection / recovery and the confounder
# ---------------------------------------------------------------------------
def test_star_above_the_ceiling_has_positive_xi_on_two_flares():
    e_big = 10.0 * _bound(0.01)
    rec = star_ceiling([1e34, e_big, 2e34, e_big, 1e34], [100.0, 200.0, 250.0, 400.0, 500.0],
                       0.01, 1.0, T_SUN)
    assert rec["assessable"]
    assert math.isclose(rec["xi_conservative_max"], 1.0, abs_tol=1e-9)
    assert math.isclose(rec["xi_nominal_max"], 1.0 + 1.5 * math.log10(3.0), abs_tol=1e-9)
    assert rec["n_above_conservative"] == 2 and rec["n_independent"] == 5
    assert [f["t_peak"] for f in rec["flares_above"]] == [200.0, 400.0]


def test_same_energy_on_a_five_percent_amplitude_star_is_below_the_ceiling():
    e_big = 10.0 * _bound(0.01)
    rec = star_ceiling([e_big, e_big], [200.0, 400.0], 0.05, 1.0, T_SUN)
    assert rec["xi_conservative_max"] < 0.0 and rec["n_above_conservative"] == 0
    # (5 %)^1.5 = 11.2x the 1 % bound: the nominal bound is still exceeded, so
    # the star is a watch, never a candidate
    assert rec["xi_nominal_max"] > 0.0
    v = vet_star(dict(rec, mission="kepler"), FULL_CTX)
    assert v["tier"] == "watch" and v["first_veto"] == "below_ceiling"


def test_unassessable_star_is_never_a_zero():
    rec = star_ceiling([1e36], [100.0], float("nan"), 1.0, T_SUN)
    assert not rec["assessable"] and np.isnan(rec["xi_conservative_max"])
    v = vet_star(dict(rec, mission="kepler"), FULL_CTX)
    assert v["tier"] == "none" and v["first_veto"] == "no_ceiling"


FULL_CTX = {"mission": "kepler", "gaia_reached": True, "ruwe": 1.0, "nss": 0, "gmag": 12.0,
            "neighbours": [(8.0, 18.5)], "vari_class": None, "logg": 4.4, "radius_rsun": 1.0}


def _above(**kw) -> dict:
    e_big = 10.0 * _bound(0.01)
    rec = star_ceiling([e_big, 1e34, e_big], [200.0, 300.0, 400.0], 0.01, 1.0, T_SUN)
    rec.update({"mission": "kepler", "star_key": "kepler:1", "star_id": "1",
                "params_assumed": False, "has_peak_times": True, "catalogue_flag": ""})
    rec.update(kw)
    return rec


def test_candidate_when_every_veto_is_applied_and_passed():
    v = vet_star(_above(), FULL_CTX)
    assert v["tier"] == "candidate" and v["first_veto"] is None and v["flags"] == []
    assert v["centroid"] == "not_checked"
    assert [p["quarter"] for p in v["stage2_pulls"]] == [2, 4]
    assert all(p["product"] == "kepler_tpf" for p in v["stage2_pulls"])


def test_ruwe_two_star_is_vetoed_as_companion_suspect():
    v = vet_star(_above(), dict(FULL_CTX, ruwe=2.0))
    assert v["first_veto"] == "companion_suspect" and v["tier"] == "none"
    assert "RUWE=2.00" in v["veto_detail"]["companion_suspect"]
    v = vet_star(_above(), dict(FULL_CTX, nss=1))
    assert v["first_veto"] == "companion_suspect"
    v = vet_star(_above(), dict(FULL_CTX, catalogue_binary=True))
    assert v["first_veto"] == "companion_suspect"
    assert not companion_suspect(1.3, 0)[0] and companion_suspect(1.41, 0)[0]


def test_evolved_star_is_vetoed():
    v = vet_star(_above(), dict(FULL_CTX, logg=3.5))
    assert v["first_veto"] == "evolved" and v["tier"] == "none"
    v = vet_star(_above(), dict(FULL_CTX, radius_rsun=2.0))
    assert v["first_veto"] == "evolved"
    assert not evolved(4.3, 1.5)[0] and evolved(3.99, 1.0)[0] and evolved(4.5, 1.61)[0]


def test_bright_neighbour_is_a_blend_and_a_faint_one_is_not():
    v = vet_star(_above(), dict(FULL_CTX, neighbours=[(5.0, 13.5)]))
    assert v["first_veto"] == "blend"
    v = vet_star(_above(), dict(FULL_CTX, neighbours=[(5.0, 16.0), (20.0, 10.0)]))
    assert v["tier"] == "candidate"
    assert blend([(3.0, 14.9)], 12.0)[0] and not blend([(3.0, 15.1)], 12.0)[0]
    # unknown target magnitude: any neighbour inside the radius is a blend
    assert blend([(3.0, 14.9)], float("nan"))[0]


def test_gaia_pulsator_or_eclipsing_class_is_vetoed_but_spot_classes_are_not():
    v = vet_star(_above(), dict(FULL_CTX, vari_class="ECL"))
    assert v["first_veto"] == "pulsator_or_eb"
    for ok in ("RS", "SOLAR_LIKE", "", None, "nan"):
        assert not pulsator_or_eb(ok)[0]
    assert pulsator_or_eb("DSCT|GDOR|SXPHE")[0] and pulsator_or_eb("RR")[0]


def test_catalogue_flag_marks_doubtful_flares():
    v = vet_star(_above(catalogue_flag="doubtful"), FULL_CTX)
    assert v["first_veto"] == "catalogue_doubtful"
    assert catalogue_doubtful("binary")[0] and catalogue_doubtful("F")[0]
    for ok in ("", "0", "ok", "1", None, "nan"):
        assert not catalogue_doubtful(ok)[0]


def test_single_flare_or_unreached_gaia_or_assumed_params_is_interest_not_candidate():
    e_big = 10.0 * _bound(0.01)
    one = star_ceiling([e_big, 1e34], [200.0, 300.0], 0.01, 1.0, T_SUN)
    one.update({"mission": "kepler", "params_assumed": False, "has_peak_times": True})
    v = vet_star(one, FULL_CTX)
    assert v["tier"] == "interest" and "single_flare_above" in v["flags"]
    v = vet_star(_above(), dict(FULL_CTX, gaia_reached=False))
    assert v["tier"] == "interest" and "gaia_unreached" in v["flags"]
    v = vet_star(_above(params_assumed=True), FULL_CTX)
    assert v["tier"] == "interest" and "stellar_params_assumed" in v["flags"]
    v = vet_star(_above(has_peak_times=False, amplitude_unit_guessed=True), FULL_CTX)
    assert v["tier"] == "candidate"
    assert {"no_peak_times", "amplitude_unit_guessed"} <= set(v["flags"])


def test_kepler_quarter_lookup():
    assert kepler_quarter_of(200.0) == 2 and kepler_quarter_of(1000.0) == 10
    assert kepler_quarter_of(float("nan")) is None and kepler_quarter_of(5000.0) is None


def test_every_rejection_rule_has_a_counter_and_the_funnel_counts_in_order():
    recs = [dict(_above(), star_key="k:1"), dict(_above(), star_key="k:2"),
            dict(_above(), star_key="k:3"), dict(_above(), star_key="k:4"),
            dict(_above(catalogue_flag="doubtful"), star_key="k:5"),
            dict(_above(), star_key="k:6")]
    ctx = {"k:1": dict(FULL_CTX, ruwe=3.0), "k:2": dict(FULL_CTX, neighbours=[(2.0, 12.5)]),
           "k:3": dict(FULL_CTX, logg=3.0), "k:4": dict(FULL_CTX, vari_class="ECL"),
           "k:5": FULL_CTX, "k:6": FULL_CTX}
    vetted = assign_tiers(recs, ctx)
    counters = rejection_counters(vetted)
    for rule in HARD_VETO_ORDER:
        assert counters["first_veto"][rule] == 1, (rule, counters)
    assert counters["tiers"]["candidate"] == 1
    for f in REPORT_FLAGS:
        assert f in counters["flags_raised"]
    fun = funnel(vetted)
    assert fun["xi_conservative_positive"] == 6
    assert fun["after_catalogue_doubtful"] == 5 and fun["after_companion_suspect"] == 4
    assert fun["after_blend"] == 3 and fun["after_evolved"] == 2
    assert fun["after_pulsator_or_eb"] == 1 and fun["candidate"] == 1


# ---------------------------------------------------------------------------
# schema discovery (pure)
# ---------------------------------------------------------------------------
def test_column_roles_distinguish_flare_from_rotational_amplitude():
    okamoto_like = ["recno", "KIC", "Teff", "logg", "Rad", "Prot", "BVAmp", "Tpeak", "E", "Dur"]
    r = acq.resolve_arc_columns(okamoto_like, "flares")
    assert r["star_id"] == "KIC" and r["energy"] == "E" and r["t_peak"] == "Tpeak"
    assert r["rot_amplitude"] == "BVAmp" and r["radius"] == "Rad" and r["teff"] == "Teff"
    yang_like = ["KIC", "Tstart", "Tpeak", "Tend", "Amp", "E"]
    r = acq.resolve_arc_columns(yang_like, "flares")
    assert r["flare_amplitude"] == "Amp" and "rot_amplitude" not in r
    mcq_like = ["KIC", "Prot", "e_Prot", "Rper", "LPH", "w"]
    r = acq.resolve_arc_columns(mcq_like, "stars")
    assert r["rot_amplitude"] == "Rper" and r["prot"] == "Prot"
    star_amp = ["KIC", "Amp", "Prot"]
    assert acq.resolve_arc_columns(star_amp, "stars")["rot_amplitude"] == "Amp"
    # a config override wins when the column exists, and is ignored when it does not
    r = acq.resolve_arc_columns(yang_like, "flares", {"rot_amplitude": "Amp"})
    assert r["rot_amplitude"] == "Amp"
    r = acq.resolve_arc_columns(yang_like, "flares", {"rot_amplitude": "Nope"})
    assert "rot_amplitude" not in r
    # "Perr" must not resolve as a period
    assert "prot" not in acq.resolve_arc_columns(["KIC", "E", "Perr"], "flares")


def test_score_table_rejects_tables_without_the_needed_roles():
    s, _, why = acq.score_table(["KIC", "Tpeak", "Amp"], "flares")
    assert s == 0 and "no energy" in why
    s, _, why = acq.score_table(["KIC", "Teff", "logg"], "stars")
    assert s == 0 and "rotational amplitude" in why
    assert acq.score_table(["KIC", "E"], "flares")[0] > 0
    assert acq.score_table(["KIC", "Rper"], "stars")[0] > 0
    assert acq.score_table(["KIC", "EDmax"], "flares")[0] > 0


def test_parse_gaia_cone_separates_target_from_neighbours():
    df = pd.DataFrame({"RA_ICRS": [290.0, 290.0 + 6 / 3600 / math.cos(math.radians(44.0))],
                       "DE_ICRS": [44.0, 44.0], "RUWE": [1.1, 3.0], "NSS": [0, 1],
                       "Plx": [5.0, 1.0], "Gmag": [12.0, 14.0], "Source": [111, 222]})
    c = acq.parse_gaia_cone(df, 290.0, 44.0)
    assert c["target_found"] and c["ruwe"] == 1.1 and c["nss"] == 0 and c["gmag"] == 12.0
    assert c["gaia_source"] == "111" and len(c["neighbours"]) == 1
    sep, g = c["neighbours"][0]
    assert abs(sep - 6.0) < 0.05 and g == 14.0
    c = acq.parse_gaia_cone(df, 290.0, 44.5)
    assert not c["target_found"] and len(c["neighbours"]) == 2
    assert not acq.parse_gaia_cone(pd.DataFrame(), 0.0, 0.0)["target_found"]


# ---------------------------------------------------------------------------
# a scripted VizieR: tables in a dict, the ADQL the channel emits emulated
# ---------------------------------------------------------------------------
class _FakeTAP:
    """``mode`` = 'fail' | 'zero' | 'ok'; ``tables`` maps a table name to a frame."""

    def __init__(self, mode: str, tables: dict | None = None):
        self.mode, self.tables, self.calls = mode, tables or {}, []

    def __call__(self, adql: str):
        self.calls.append(adql)
        if self.mode == "fail":
            raise RuntimeError("CONNECT tunnel failed, response 403")
        if "TAP_SCHEMA.tables" in adql:
            if self.mode == "zero" or "description LIKE" in adql:
                return pd.DataFrame(columns=["table_name", "description"])
            pat = re.search(r"table_name LIKE '%(.*?)%'", adql).group(1)
            names = [t for t in self.tables if pat in t]
            return pd.DataFrame({"table_name": [f'"{t}"' for t in names],
                                 "description": ["" for _ in names]})
        if "TAP_SCHEMA.columns" in adql:
            t = re.search(r"table_name = '([^']*)'", adql).group(1)
            cols = list(self.tables[t].columns) if t in self.tables else []
            return pd.DataFrame({"column_name": cols})
        m = re.search(r'FROM "([^"]+)"', adql)
        t = m.group(1)
        df = self.tables[t]
        if "COUNT(*)" in adql:
            return pd.DataFrame({"n": [len(df)]})
        sel = re.search(r"SELECT\s+(?:TOP\s+(\d+)\s+)?(.*?)\s+FROM", adql, re.S)
        top, cols = sel.group(1), [c.strip().strip('"') for c in sel.group(2).split(",")]
        out = df[cols].copy()
        w = re.search(r'WHERE "([^"]+)" IN \((.*)\)', adql)
        if w:
            ids = [s.strip().strip("'") for s in w.group(2).split(",")]
            out = out[out[w.group(1)].astype(str).isin(ids)]
        w = re.search(r"WHERE recno BETWEEN (\d+) AND (\d+)", adql)
        if w:
            out = out[(df["recno"] >= int(w.group(1))) & (df["recno"] <= int(w.group(2)))]
        if top:
            out = out.head(int(top))
        return out.reset_index(drop=True)


def _synthetic_tables() -> tuple[dict, dict]:
    """Okamoto-like flare table (per flare, with Teff / Rad / Prot per row and NO
    amplitude), a McQuillan-like star table (Rper in ppm), a Berger-like
    parameter table.  Truth: which star is what."""
    e_big = 10.0 * _bound(0.01, teff=5800.0)
    rows, stars = [], []
    truth = {"candidate": "1000001", "confounder_5pct": "1000002", "ruwe2": "1000003",
             "evolved": "1000004", "single": "1000005"}

    def add(sid, times, energies, rper_ppm, logg=4.4, ruwe=1.0):
        for t, e in zip(times, energies, strict=True):
            rows.append((sid, t, e, 5800.0, 1.0, 12.0))
        stars.append((sid, 12.0, rper_ppm, logg, ruwe))

    big = [1e34, e_big, 2e34, e_big, 1e34]
    tt = [150.0, 200.0, 250.0, 400.0, 500.0]
    add("1000001", tt, big, 10000.0)                   # 1 % amplitude: the candidate
    add("1000002", tt, big, 50000.0)                   # 5 % amplitude: the confounder
    add("1000003", tt, big, 10000.0, ruwe=2.0)         # RUWE 2: companion suspect
    add("1000004", tt, big, 10000.0, logg=3.5)         # subgiant
    add("1000005", tt[:2], [1e34, e_big], 10000.0)     # one flare above
    for i in range(10):
        add(str(2000000 + i), [100.0 + 30 * i, 300.0 + 30 * i], [1e34, 3e34], 10000.0)
    flares = pd.DataFrame(rows, columns=["KIC", "Tpeak", "E", "Teff", "Rad", "Prot"])
    flares.insert(0, "recno", np.arange(1, len(flares) + 1))
    mcq = pd.DataFrame([(s, p, r) for s, p, r, _, _ in stars], columns=["KIC", "Prot", "Rper"])
    berger = pd.DataFrame([(s, 290.0 + 0.01 * i, 44.0, 5800.0, 1.0, lg)
                           for i, (s, _, _, lg, _) in enumerate(stars)],
                          columns=["KIC", "RAJ2000", "DEJ2000", "Teff", "R*", "logg"])
    ruwe_by_ra = {290.0 + 0.01 * i: ru for i, (_, _, _, _, ru) in enumerate(stars)}
    tables = {"J/ApJ/906/72/table2": flares, "J/ApJS/211/24/table1": mcq,
              "J/AJ/159/280/table1": berger}
    return tables, dict(truth, ruwe_by_ra=ruwe_by_ra)


def _conf() -> dict:
    conf = load_arc_config()
    conf["catalogues"] = {"kepler_synth": {"mission": "kepler", "kind": "flares",
                                           "preferred": "J/ApJ/906/72/table2",
                                           "keywords": ["superflare"],
                                           "energy": {"kind": "bolometric", "factor": 1.0,
                                                      "log10": "auto"}}}
    conf["star_catalogues"] = {"kepler": [{"name": "mcquillan2014",
                                           "preferred": "J/ApJS/211/24/table1",
                                           "keywords": [], "amplitude_unit": "ppm"}]}
    conf["param_tables"] = {"kepler": ["J/AJ/159/280/table1"]}
    return conf


def test_run_with_unreachable_archive_is_no_data_reached(tmp_path):
    out = tmp_path / "arc"
    rep = arc_run("all", out_dir=out, query_fn=_FakeTAP("fail"), offline=True, conf=_conf())
    assert rep["verdict"] == VERDICT_NO_DATA and rep["data_status"] == "QUERY_FAILED"
    s = json.loads((out / "summary.json").read_text())
    assert s["n_stars"] == 0 and s["tiers"]["candidate"] == 0 and s["candidates"] == []
    assert "NOT a null result" in s["note"]
    probe = json.loads((out / "probe.json").read_text())
    assert probe["n_usable"] == 0
    assert all(v["status"] == "QUERY_FAILED" for v in probe["catalogues"].values())
    assert (out / "candidates.csv").exists() and (out / "xi_table.csv").exists()


def test_run_with_empty_archive_is_no_data_reached_with_zero_rows_status(tmp_path):
    out = tmp_path / "arc"
    rep = arc_run("all", out_dir=out, query_fn=_FakeTAP("zero"), offline=True, conf=_conf())
    assert rep["verdict"] == VERDICT_NO_DATA
    assert rep["data_status"] == "QUERY_RETURNED_ZERO_ROWS"
    assert rep["degraded"] == ["kepler_synth:QUERY_RETURNED_ZERO_ROWS"]


def test_end_to_end_synthetic_run(tmp_path):
    tables, truth = _synthetic_tables()
    fake = _FakeTAP("ok", tables)
    out = tmp_path / "arc"
    conf = _conf()

    probe = stage_probe(conf, out, query_fn=fake)
    assert probe["n_usable"] == 1 and probe["n_star_tables_usable"] == 1
    d = probe["catalogues"]["kepler_synth"]
    assert d["roles"]["energy"] == "E" and d["roles"]["t_peak"] == "Tpeak"
    assert "rot_amplitude" not in d["roles"]
    assert probe["star_catalogues"]["kepler_mcquillan2014"]["roles"]["rot_amplitude"] == "Rper"

    acq_rep = stage_acquire(conf, out, query_fn=fake)
    assert acq_rep["catalogues"]["kepler_synth"]["status"] == "OK"
    assert acq_rep["catalogues"]["kepler_synth"]["time_system_guess"] == "BKJD"
    assert acq_rep["star_catalogues"]["kepler_mcquillan2014"]["status"] == "OK"
    assert (out / "data" / "kepler_synth_flares.parquet").exists()

    screen = stage_screen(conf, out)
    rep = screen["kepler_synth"]
    assert rep["n_stars"] == 15 and rep["n_stars_assessable"] == 15
    assert rep["n_xi_conservative_positive"] == 4          # 1, 3, 4, 5 (not the 5 % star)
    assert rep["n_xi_nominal_positive"] == 5
    assert rep["amplitude_sources"] == {"mcquillan2014": 15}
    xi_df = pd.read_csv(out / "xi_kepler_synth.csv", dtype={"star_id": str})
    row = xi_df.set_index("star_id").loc[truth["candidate"]]
    assert abs(row["xi_conservative_max"] - 1.0) < 1e-6 and row["n_above_conservative"] == 2
    assert row["amplitude_unit"] == "ppm" and not row["amplitude_unit_guessed"]
    assert row["prot"] == 12.0 and row["prot_source"] == "own"
    assert xi_df.set_index("star_id").loc[truth["confounder_5pct"], "xi_conservative_max"] < 0

    # offline assess: Gaia not reached -> interest, never candidate; the
    # subgiant is not caught because logg only arrives with the params
    s = stage_assess(conf, out, offline=True)
    assert s["verdict"] == VERDICT_CANDIDATES
    assert s["n_candidates"] == 0 and s["n_interest"] == 4 and s["n_watch"] == 1
    assert "gaia_context:offline" in s["degraded"]
    assert s["funnel"]["xi_conservative_positive"] == 4

    # full assess with the scripted Gaia cone: RUWE 2 and the subgiant fall,
    # the single-flare star stays at interest, the 1 % star is the candidate
    calls = []

    def cone(table, ra, dec, r):
        calls.append(table)
        if table.startswith("I/358"):
            return pd.DataFrame()
        return pd.DataFrame({"RA_ICRS": [ra], "DE_ICRS": [dec], "RUWE": [truth["ruwe_by_ra"][ra]],
                             "NSS": [0], "Plx": [5.0], "Gmag": [12.0], "Source": [1]})

    s2 = stage_assess(conf, out, offline=False, query_fn=fake, cone_fn=cone)
    assert s2["verdict"] == VERDICT_CANDIDATES
    assert s2["n_candidates"] == 1 and s2["n_interest"] == 1 and s2["n_watch"] == 1
    assert s2["rejection_counters"]["first_veto"]["companion_suspect"] == 1
    assert s2["rejection_counters"]["first_veto"]["evolved"] == 1
    assert s2["gaia_reached_fraction_of_shortlist"] == 1.0
    assert not s2["degraded"]
    c = json.loads((out / "candidates.json").read_text())
    top = c["candidates"][0]
    assert top["star_id"] == truth["candidate"] and top["tier"] == "candidate"
    assert top["centroid"] == "not_checked"
    assert [p["quarter"] for p in top["stage2_pulls"]] == [2, 4]
    assert c["candidates"][1]["star_id"] == truth["single"]
    assert c["watch"][0]["star_id"] == truth["confounder_5pct"]
    cands = pd.read_csv(out / "candidates.csv", dtype={"star_id": str})
    assert list(cands["star_id"]) == [truth["candidate"], truth["single"]]
    assert (out / "xi_table.csv").exists() and (out / "summary.json").exists()
    fun = s2["funnel"]
    assert fun["xi_conservative_positive"] == 4 and fun["after_companion_suspect"] == 3
    assert fun["after_evolved"] == 2 and fun["candidate"] == 1 and fun["interest"] == 1


def test_no_ceiling_excess_when_every_star_is_below(tmp_path):
    tables, _ = _synthetic_tables()
    fl = tables["J/ApJ/906/72/table2"].copy()
    fl["E"] = 1e34
    fake = _FakeTAP("ok", dict(tables, **{"J/ApJ/906/72/table2": fl}))
    out = tmp_path / "arc"
    rep = arc_run("all", out_dir=out, query_fn=fake, offline=True, conf=_conf())
    assert rep["verdict"] == VERDICT_NONE
    assert rep["n_stars_assessable"] == 15 and rep["funnel"]["xi_conservative_positive"] == 0
    assert rep["xi_distribution"]["conservative"]["max"] < 0
    assert "count, not an occurrence limit" in rep["note"]


def test_amplitude_is_the_max_over_sources_and_own_columns_come_first():
    fl = pd.DataFrame({"star_id": ["a", "a", "b"], "energy": [1e34, 2e34, 1e34],
                       "rot_amplitude": [0.005, 0.005, np.nan], "teff": [5000.0, 5000.0, np.nan],
                       "radius": [0.9, 0.9, np.nan]})
    mcq = pd.DataFrame({"star_id": ["a", "b"], "rot_amplitude": [12000.0, 3000.0],
                        "prot": [10.0, 20.0], "teff": [5500.0, 6000.0]})
    ctx = build_star_context(fl, [("mcq", mcq, "ppm")], {"amplitude_min_frac": 1e-4}
                             ).set_index("star_id")
    assert np.isclose(ctx.loc["a", "amplitude_frac"], 0.012)      # 12000 ppm > 0.5 %
    assert ctx.loc["a", "amplitude_source"] == "mcq"
    assert ctx.loc["a", "teff"] == 5000.0 and ctx.loc["a", "teff_source"] == "own"
    assert ctx.loc["b", "teff"] == 6000.0 and ctx.loc["b", "teff_source"] == "mcq"
    assert np.isclose(ctx.loc["b", "amplitude_frac"], 0.003) and ctx.loc["b", "prot"] == 20.0
    assert np.isnan(ctx.loc["b", "radius"])


def test_screen_assumes_solar_parameters_only_with_a_flag():
    fl = pd.DataFrame({"star_id": ["a", "a"], "energy": [35.0, 37.0], "t_peak": [100.0, 300.0]})
    mcq = pd.DataFrame({"star_id": ["a"], "rot_amplitude": [10000.0]})
    recs, rep = screen_catalogue(fl, [("mcq", mcq, "ppm")], "s", "kepler", load_arc_config())
    r = recs[0]
    assert r["params_assumed"] and r["teff_k"] == 5777.0 and r["radius_rsun"] == 1.0
    assert rep["energy"]["log10_input"] and r["e_flare_max_erg"] == 1e37
    assert r["xi_conservative_max"] > 0     # 1e37 erg against a ~5e36 erg conservative bound
    assert rep["n_stars_params_assumed"] == 1
    v = vet_star(r, FULL_CTX)
    assert v["tier"] != "candidate" and "stellar_params_assumed" in v["flags"]


def test_screen_without_an_energy_column_reports_it():
    fl = pd.DataFrame({"star_id": ["a"], "t_peak": [100.0]})
    recs, rep = screen_catalogue(fl, [], "s", "kepler", load_arc_config())
    assert recs == [] and rep["status"] == "NO_ENERGY_COLUMN"


def test_config_workflow_and_doc_exist():
    conf = load_arc_config()
    for k in ("physics", "catalogues", "star_catalogues", "param_tables", "gaia", "vet",
              "shortlist", "acquire"):
        assert k in conf
    assert conf["physics"]["b_gauss"] == 3000.0 and conf["physics"]["geometric_factor"] == 3.0
    assert conf["vet"]["ruwe_max"] == 1.4 and conf["vet"]["min_flares_above"] == 2
    for name, spec in conf["catalogues"].items():
        assert spec["mission"] in ("kepler", "tess") and spec["preferred"], name
        assert spec["energy"]["kind"] in ("bolometric", "band", "equivalent_duration"), name
    assert Path("config/arc.yaml").exists()
    assert Path(".github/workflows/arc.yml").exists()
    assert Path("docs/arc.md").exists()


def test_tap_query_tries_every_mirror_before_failing(monkeypatch):
    """A 503 from one VizieR endpoint is routine and must not end the query.

    ARC's first dispatch (run 34787802564) failed all six discovery queries
    with 503 from the plain-http endpoint while the https one was answering;
    the helper now walks VIZIER_TAP_MIRRORS and names every endpoint it tried.
    """
    import types

    from seti.metronome import acquire as macq

    assert macq.VIZIER_TAP.startswith("https://")
    assert len(macq.VIZIER_TAP_MIRRORS) >= 2

    tried: list[str] = []

    class _Svc:
        def __init__(self, url):
            tried.append(url)
            self._url = url

        def run_async(self, adql):
            raise RuntimeError(f"503 Service Unavailable for url: {self._url}")

        def search(self, adql):
            raise RuntimeError(f"503 Service Unavailable for url: {self._url}")

    monkeypatch.setitem(
        __import__("sys").modules, "pyvo",
        types.SimpleNamespace(dal=types.SimpleNamespace(TAPService=_Svc)))
    monkeypatch.setattr(macq._time, "sleep", lambda *_a, **_k: None)
    macq.reset_route_state()          # the circuit breaker must not skip a host here

    with pytest.raises(RuntimeError) as err:
        macq.tap_query("SELECT 1", retries=1, allow_non_tap=False)
    assert tried == list(macq.VIZIER_TAP_MIRRORS)
    for endpoint in macq.VIZIER_TAP_MIRRORS:
        assert endpoint in str(err.value)
    macq.reset_route_state()


# ---------------------------------------------------------------------------
# TAP down, VizieR up: ARC gets its rows through the shared route ladder
# ---------------------------------------------------------------------------
def _vizier_without_tap(monkeypatch, *, meta: str, rows: str):
    """A world where every TAPVizieR host 503s and the non-TAP ASU route works."""
    import types

    from seti.metronome import acquire as macq

    class _Svc:
        def __init__(self, url):
            self._url = url

        def _fail(self, adql):
            raise RuntimeError(f"DALServiceError: 503 Server Error: Service Unavailable "
                               f"for url: {self._url}/sync")

        run_async = search = _fail

    monkeypatch.setitem(
        __import__("sys").modules, "pyvo",
        types.SimpleNamespace(dal=types.SimpleNamespace(TAPService=_Svc)))
    monkeypatch.setattr(macq._time, "sleep", lambda *_a, **_k: None)
    urls: list[str] = []

    def fetch(url, timeout=180.0):
        urls.append(url)
        if "-meta.all" in url:
            return meta
        if url.endswith("/ReadMe"):
            raise RuntimeError("404 Not Found")
        return rows

    monkeypatch.setattr(macq, "_asu_http_text", fetch)
    macq.reset_route_state()
    return macq, urls


ARC_ASU_META = "\n".join([
    "#RESOURCE=yCat_J/ApJS/241/29",
    "#Name: J/ApJS/241/29",
    "#Title: Kepler flare catalogue (Yang+, 2019)",
    "#Table\tJ_ApJS_241_29_table4:",
    "#Name: J/ApJS/241/29/table4",
    "#Title: Flare list",
    "#Column\tKIC\t()\tKepler Input Catalog identifier",
    "#Column\tTpeak\t(d)\tFlare peak time",
    "#Column\tEbol\t(erg)\tBolometric flare energy",
    "#Column\tRper\t(ppm)\tRotational amplitude",
])

ARC_ASU_ROWS = "\n".join([
    "#RESOURCE=yCat_J/ApJS/241/29",
    "#Name: J/ApJS/241/29/table4",
    '#Column\t"KIC"\t()\tid',
    '"KIC"\tTpeak\tEbol\tRper',
    "\td\terg\tppm",
    "--------\t--------\t--------\t--------",
    "1234567\t120.5\t1.0e34\t9500",
    "1234567\t123.5\t2.0e34\t9500",
    "7654321\t130.5\t3.0e34\t4200",
])


def test_tap_down_but_asu_up_gives_arc_rows_not_query_failed(monkeypatch):
    """ARC picks the route ladder up for free through the shared helper.

    ``seti.arc.acquire`` does not know the ASU interface exists: it calls
    ``list_tables`` / ``table_columns`` / ``count_rows`` / ``tap_query`` from
    :mod:`seti.metronome.acquire`, and those degrade to the non-TAP route.  So
    a dispatch made while TAPVizieR is 503ing returns REAL ROWS with a recorded
    route instead of the ``QUERY_FAILED`` -> ``NO_DATA_REACHED`` that this
    repository refuses to publish as a sky result.
    """
    macq, urls = _vizier_without_tap(monkeypatch, meta=ARC_ASU_META, rows=ARC_ASU_ROWS)
    log = macq.AcquisitionLog()
    disc = acq.discover_table("kepler_yang2019", "J/ApJS/241/29", "flares", ("flare", "Kepler"),
                              query_fn=macq.tap_query, log=log)
    assert disc.status == "OK" and disc.table == "J/ApJS/241/29/table4"
    assert disc.roles["star_id"] == "KIC" and disc.roles["energy"] == "Ebol"
    assert disc.roles["t_peak"] == "Tpeak" and disc.roles["rot_amplitude"] == "Rper"
    assert disc.n_rows is None                    # COUNT(*) is unknown without TAP, not failed
    df = acq.fetch_table(disc, query_fn=macq.tap_query, log=log)
    assert len(df) == 3 and list(df["star_id"]) == ["1234567", "1234567", "7654321"]
    assert np.allclose(df["energy"].to_numpy(), [1.0e34, 2.0e34, 3.0e34], rtol=1e-12)
    assert df["t_peak"].tolist() == [120.5, 123.5, 130.5]
    # the route that served the rows is recorded, with its endpoint
    summary = macq.route_log_summary()
    assert summary["served_by"] == "asu_tsv"
    assert summary["routes"]["asu_tsv"]["n_ok"] >= 2
    assert summary["routes"]["tap"]["n_failed"] >= len(macq.VIZIER_TAP_MIRRORS)
    assert any("-meta.all" in u for u in urls) and any("-out.form=TSV" in u for u in urls)
    macq.reset_route_state()


def test_arc_with_every_vizier_route_down_is_still_honestly_query_failed(monkeypatch):
    """The other half of the contract: no route, no rows, no invention."""
    def dead(url, timeout=180.0):
        raise RuntimeError("503 Server Error: Service Unavailable")

    macq, _urls = _vizier_without_tap(monkeypatch, meta="", rows="")
    monkeypatch.setattr(macq, "_asu_http_text", dead)
    monkeypatch.setattr(macq, "astroquery_rows", lambda *a, **k: (_ for _ in ()).throw(
        macq.VizierRouteError("astroquery: no route to host")))
    log = macq.AcquisitionLog()
    disc = acq.discover_table("kepler_yang2019", "J/ApJS/241/29", "flares", ("flare", "Kepler"),
                              query_fn=macq.tap_query, log=log)
    assert disc.status == "QUERY_FAILED" and disc.table is None
    assert not len(acq.fetch_table(disc, query_fn=macq.tap_query, log=log))
    assert log.as_dict()["any_query_failed"] and log.as_dict()["total_rows"] == 0
    named = " ".join(str(s.get("error", "")) for s in log.stages)
    assert "503" in named
    macq.reset_route_state()
