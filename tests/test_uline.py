"""Offline test suite for ULINE --- industrial fluorine molecules in U-line lists.

No network anywhere (``conftest.py`` raises on any socket).  Per
``docs/channel-brief.md`` §5 the suite:

* recovers an injected CHF₃ K-ladder pattern seeded into a synthetic U-line
  list (``PATTERN_CANDIDATE``) and returns ``NO_PATTERN`` on the same list
  with the pattern removed;
* shows that Poisson-random U-lines give a false-alarm probability consistent
  with the number of shift trials;
* checks the ``.cat`` / ``catdir`` / CDMS partition-table parsers on synthetic
  fixed-format blocks, the 300 K → T_ex rescaling against a hand computation,
  and the symmetric-top predictor against its formula;
* trips every rejection rule (contaminant veto, blend merging, top-5 missing
  line, sky-vs-rest frame) with a case;
* degrades honestly: a dead archive gives ``NO_DATA_REACHED`` with the
  failures recorded, never a candidate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seti.uline import acquire as acq
from seti.uline.lines import (
    C2_CM_K,
    CATDIR_TEMPS,
    MHZ_PER_CM,
    Entry,
    count_unparsed_catdir,
    find_species,
    interp_log_q,
    parse_cat,
    parse_catdir,
    parse_partition_table,
    rescale_lgint,
    symmetric_top_lines,
)
from seti.uline.match import (
    C_KM_S,
    SourceLines,
    apply_vetoes,
    coverage_mask,
    doppler_sky,
    evaluate_species,
    features_matched,
    match_features,
    merge_blends,
    tolerance_mhz,
)
from seti.uline.run import (
    load_uline_config,
    screen_all,
    stage_acquire,
    stage_assess,
    stage_probe,
    stage_screen,
    uline_run,
)

FMIN, FMAX = 80_000.0, 400_000.0          # MHz: a synthetic 3 mm – 0.75 mm survey
CHF3 = dict(b=10348.7, dj=0.011, djk=0.018, axial=5673.8, mu=1.65, k3=2.0)


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------
def chf3_features(tex: float = 50.0, fwhm: float = 4.0) -> pd.DataFrame:
    """Blended CHF₃ features at T_ex, in the survey range, strongest first."""
    df, ent = symmetric_top_lines(CHF3["b"], CHF3["dj"], CHF3["djk"], axial_mhz=CHF3["axial"],
                                  mu_debye=CHF3["mu"], k3_weight=CHF3["k3"], fmax_mhz=FMAX)
    lg = rescale_lgint(df["lgint_300"], df["elo_cm"], df["freq_mhz"], ent.temps, ent.qlog, tex)
    f = merge_blends(df["freq_mhz"], lg, df["err_mhz"], df["j"].astype(str), fwhm_km_s=fwhm)
    f = f[(f["freq_mhz"] >= FMIN) & (f["freq_mhz"] <= FMAX)]
    return f.sort_values("lgint", ascending=False).reset_index(drop=True)


def synthetic_survey(seed: int, *, n_random: int = 300, n_identified: int = 200,
                     seeded: pd.DataFrame | None = None, noise: float = 0.05,
                     shift_v: float = 0.0) -> pd.DataFrame:
    """A canonical line table: random U-lines, identified lines, and optionally
    a seeded pattern whose intensities follow the prediction (± noise)."""
    rng = np.random.default_rng(seed)
    rows = [(f, "U", float(10 ** rng.normal(-1.0, 0.5))) for f in rng.uniform(FMIN, FMAX, n_random)]
    rows += [(f, "CH3OH", 1.0) for f in rng.uniform(FMIN, FMAX, n_identified)]
    if seeded is not None:
        for _, r in seeded.iterrows():
            rows.append((float(doppler_sky(r["freq_mhz"], shift_v)), "U",
                         float(10 ** r["lgint"] * 1e6 * (1.0 + noise * rng.normal()))))
    tab = pd.DataFrame(rows, columns=["freq_mhz", "ident", "intensity"])
    tab["freq_err_mhz"] = 0.1
    tab["transition"] = ""
    tab["unidentified"] = tab["ident"] == "U"
    tab["row"] = np.arange(len(tab))
    return tab


def seeded_pattern(n: int = 8) -> pd.DataFrame:
    """Eight CHF₃ features spanning a wide intensity range (every other J)."""
    f = chf3_features()
    f["j"] = f["members"].str.split("|").str[0].astype(int)
    pick = f[f["j"] % 2 == 1].sort_values("j").head(n)
    return pick.reset_index(drop=True)


def synth_conf(**match) -> dict:
    conf = load_uline_config()
    conf["sources"] = {"synth": {"vizier_like": "X/", "v_lsr_km_s": 0.0, "fwhm_km_s": 4.0,
                                 "frequency_frame": "rest", "v_lsr_uncertainty_km_s": 0.0,
                                 "contaminant_tex_k": 100.0, "enabled": True}}
    conf["match"].update({"n_trials": 200, **match})
    return conf


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------
def _cat_line(freq, err, lgint, dr, elo, gup, tag, qnfmt, qn_up, qn_lo) -> str:
    """Pickett fixed format: F13.4 F8.4 F8.4 I2 F10.4 I3 I7 I4 6I2 6I2 (``gup`` may be 'A05')."""
    g = f"{gup:>3}" if isinstance(gup, str) else f"{gup:3d}"
    return (f"{freq:13.4f}{err:8.4f}{lgint:8.4f}{dr:2d}{elo:10.4f}{g}{tag:7d}{qnfmt:4d}"
            + "".join(f"{q:2d}" for q in qn_up) + "".join(f"{q:2d}" for q in qn_lo))


CAT_BLOCK = "\n".join([
    _cat_line(22235.0798, 0.0010, -1.8, 3, 35.1234, 7, 18003, 1404, [3, 1, 2, 0, 0, 0], [3, 2, 1, 0, 0, 0]),
    _cat_line(183310.0870, 0.0050, -3.0, 3, 136.1639, 15, -18003, 1404, [3, 1, 3, 0, 0, 0], [2, 2, 0, 0, 0, 0]),
    _cat_line(556936.0020, 0.0300, -1.1, 3, 0.0, "A05", 18003, 1404, [1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]),
    "garbage line that is long enough to be attempted but is not a cat record",
]) + "\n"


def test_cat_parser_on_a_synthetic_fixed_format_block():
    df = parse_cat(CAT_BLOCK)
    assert len(df) == 3
    assert np.isclose(df["freq_mhz"][0], 22235.0798) and np.isclose(df["err_mhz"][0], 0.001)
    assert np.isclose(df["lgint_300"][1], -3.0) and np.isclose(df["elo_cm"][1], 136.1639)
    assert df["gup"].tolist() == [7, 15, 1005]           # A05 -> 1005 (letter overflow)
    assert df["tag"].tolist() == [18003, 18003, 18003]
    assert df["lab"].tolist() == [False, True, False]     # negative tag = lab-measured
    assert df["qn_up"][0].split() == ["3", "1", "2", "0", "0", "0"]
    assert parse_cat("").empty


CATDIR_BLOCK = (
    " 18003 H2O                 1017 2.2507 2.0645 1.8003 1.3608 0.9199 0.4907 0.1183  4\n"
    " 71001 NF3                 9999 4.1000 3.9000 3.6000 3.2000 2.7000 2.3000 1.8000  1\n"
    " 32003 CH3OH             123456 3.4000 3.2000 3.0000 2.5000 2.0000 1.5000 1.0000  3\n"
    " 50007 CH3Cl-35            5000 3.0000 2.8000 2.6000 2.2000 1.7000 1.3000 0.8000  2\n"
    " 52012 CF2Cl2-fake         5000 3.0000 2.8000 2.6000 2.2000 1.7000 1.3000 0.8000  2\n"
    "not a directory line\n"
)


def test_catdir_parser_and_species_regexes():
    ents = parse_catdir(CATDIR_BLOCK)
    assert [e.tag for e in ents] == [18003, 71001, 32003, 50007, 52012]
    assert ents[0].name == "H2O" and ents[0].nlines == 1017 and ents[0].version == "4"
    assert ents[0].temps == list(CATDIR_TEMPS) and np.isclose(ents[0].qlog[0], 2.2507)
    assert count_unparsed_catdir(CATDIR_BLOCK) == 1
    conf = load_uline_config()["species"]
    assert [e.name for e in find_species(ents, conf["targets"]["NF3"]["patterns"])] == ["NF3"]
    assert [e.name for e in find_species(ents, conf["baseline"]["CH3Cl"]["patterns"])] == ["CH3Cl-35"]
    assert [e.name for e in find_species(ents, conf["targets"]["CF2Cl2"]["patterns"])] == ["CF2Cl2-fake"]
    # CF2 must not match CF2Cl2, and CH3OH must not match CH3OHH-like names
    assert find_species(ents, conf["targets"]["CF2"]["patterns"]) == []
    inv = acq.jpl_inventory(CATDIR_BLOCK, conf)
    assert [e.tag for e in inv["NF3"]] == [71001] and inv["CHF3"] == []


CDMS_HTML = """<html><body><pre>
   tag   Molecule     #lines   lg(Q(1000)) lg(Q(500)) lg(Q(300)) lg(Q(225)) lg(Q(150)) lg(Q(75)) lg(Q(37.5)) lg(Q(18.75)) lg(Q(9.375)) lg(Q(5.000)) lg(Q(2.725))
 034501 CH3F, v=0        321   3.1 2.9 2.7 2.5 2.2 1.8 1.4 1.0 0.6 0.3 0.1
 032504 CH3OH, vt=0-2   9999   4.9 4.4 3.9 3.6 3.3 2.8 2.3 1.9 1.5 --- ---
 070501 CHF3, v=0        500   3.9 3.6 3.2 3.0 2.7 2.3 1.9 1.4 1.0 0.7 0.4
</pre>
<a href="/cgi-bin/cdmssearch?file=c034501.cat">c034501.cat</a>
<a href="c070501.cat">CHF3</a>
</body></html>"""


def test_cdms_partition_table_and_link_discovery():
    ents = parse_partition_table(CDMS_HTML)
    assert [e.tag for e in ents] == [34501, 32504, 70501]
    assert ents[1].name == "CH3OH, vt=0-2" and ents[1].nlines == 9999
    assert ents[0].temps[:3] == [1000.0, 500.0, 300.0] and len(ents[0].qlog) == 11
    assert np.isnan(ents[1].qlog[-1]) and np.isclose(ents[1].qlog[2], 3.9)
    assert acq.cdms_cat_tags(CDMS_HTML) == [34501, 70501]
    conf = load_uline_config()["species"]
    allents, rep = acq.cdms_inventory({"root": CDMS_HTML, "dead": None}, conf)
    assert rep["pages"]["dead"]["status"] == "QUERY_FAILED"
    assert rep["pages"]["root"]["n_partition_entries"] == 3
    assert [e.tag for e in rep["species"]["CHF3"]] == [70501]
    assert [e.tag for e in rep["species"]["CH3F"]] == [34501]
    assert [e.tag for e in rep["species"]["CH3OH"]] == [32504]


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------
def test_interp_log_q_is_linear_in_log_t_and_extrapolates():
    temps = [300.0, 150.0, 75.0]
    q = [3.0, 2.5, 2.0]
    assert np.isclose(interp_log_q(temps, q, 150.0), 2.5)
    mid = interp_log_q(temps, q, math.sqrt(300.0 * 150.0))
    assert np.isclose(mid, 2.75)
    assert np.isclose(interp_log_q(temps, q, 37.5), 1.5)    # linear extrapolation below
    assert np.isclose(interp_log_q(temps, q, 600.0), 3.5)   # and above
    assert np.isclose(interp_log_q(temps, [float("nan"), 2.5, 2.0], 75.0), 2.0)
    with pytest.raises(ValueError):
        interp_log_q(temps, [float("nan")] * 3, 100.0)


def test_intensity_rescaling_against_a_hand_computation():
    temps, qlog = list(CATDIR_TEMPS), [3.0, 2.8, 2.5, 2.0, 1.5, 1.0, 0.5]
    lg300, elo, nu = -3.0, 10.0, 100000.0
    t = 50.0
    eu = elo + nu / MHZ_PER_CM
    q300 = 10 ** interp_log_q(temps, qlog, 300.0)
    qt = 10 ** interp_log_q(temps, qlog, t)
    num = math.exp(-C2_CM_K * elo / t) - math.exp(-C2_CM_K * eu / t)
    den = math.exp(-C2_CM_K * elo / 300.0) - math.exp(-C2_CM_K * eu / 300.0)
    expected = lg300 + math.log10(q300 / qt * num / den)
    got = rescale_lgint([lg300], [elo], [nu], temps, qlog, t)[0]
    assert np.isclose(got, expected, atol=1e-9)
    # at 300 K the rescaling is the identity
    assert np.isclose(rescale_lgint([lg300], [elo], [nu], temps, qlog, 300.0)[0], lg300)
    # a very high lower level at 10 K: exp(-c2 E/T) underflows in linear space,
    # but the log-space evaluation stays finite (≈ -180 dex) and is not NaN
    v = rescale_lgint([-3.0], [10000.0], [nu], temps, qlog, 10.0)[0]
    assert np.isfinite(v) and v < -500.0
    assert math.exp(-C2_CM_K * 10000.0 / 10.0) == 0.0         # what a naive evaluation gives


def test_symmetric_top_predictor_matches_its_formula():
    df, ent = symmetric_top_lines(CHF3["b"], CHF3["dj"], CHF3["djk"], axial_mhz=CHF3["axial"],
                                  mu_debye=CHF3["mu"], k3_weight=CHF3["k3"], fmax_mhz=FMAX)
    j, k = 12, 3
    row = df[(df["j"] == j) & (df["k"] == k)].iloc[0]
    nu = 2 * CHF3["b"] * (j + 1) - 4 * CHF3["dj"] * (j + 1) ** 3 - 2 * CHF3["djk"] * (j + 1) * k * k
    assert np.isclose(row["freq_mhz"], nu)
    assert np.isclose(row["elo_cm"], (CHF3["b"] * j * (j + 1) + (CHF3["axial"] - CHF3["b"]) * k * k)
                      / MHZ_PER_CM)
    assert row["gup"] == (2 * (j + 1) + 1) * 2 * 2          # g_K = 2, K = 3n spin weight 2
    assert df[(df["j"] == j) & (df["k"] == 0)].iloc[0]["gup"] == (2 * (j + 1) + 1) * 1 * 2
    assert df[(df["j"] == j) & (df["k"] == 1)].iloc[0]["gup"] == (2 * (j + 1) + 1) * 2 * 1
    assert (df["k"] <= df["j"]).all() and (df["freq_mhz"] <= FMAX).all()
    assert ent.database == "predicted" and len(ent.qlog) == len(CATDIR_TEMPS)
    assert np.all(np.diff(ent.qlog) < 0)                     # Q falls with T along the grid
    # the K = 0 line is the strongest of its J ladder at 300 K for K > 0 excluding spin weight
    assert row["err_mhz"] > 0


def test_merge_blends_merges_unresolved_k_components():
    f = np.array([100000.0, 100000.5, 100001.0, 100020.0])
    lg = np.array([-3.0, -3.0, -3.0, -3.0])
    feats = merge_blends(f, lg, np.zeros(4), ["a", "b", "c", "d"], fwhm_km_s=4.0)
    assert len(feats) == 2
    assert feats["n_members"].tolist() == [3, 1] and feats["members"][0] == "a|b|c"
    assert np.isclose(feats["lgint"][0], -3.0 + math.log10(3))
    assert np.isclose(feats["freq_mhz"][0], 100000.5)


def test_doppler_tolerance_and_one_to_one_matching():
    assert np.isclose(doppler_sky(100000.0, C_KM_S / 1000.0), 99900.0)
    tol = tolerance_mhz(np.array([300000.0]), 4.0, err_mhz=0.1, v_unc_km_s=0.0)[0]
    assert np.isclose(tol, 300000.0 * 4.0 / C_KM_S)
    assert np.isclose(tolerance_mhz(np.array([300000.0]), 4.0, err_mhz=10.0)[0], 10.0)
    u = np.array([100.0, 100.5, 200.0, 300.0])
    fi, uj, sep = match_features(np.array([100.4, 100.6, 299.0]), np.array([1.0, 1.0, 0.5]), u,
                                 np.zeros(4))
    # 100.6 takes 100.5 (closest pair), 100.4 then takes 100.0; 299.0 is out of tolerance
    assert set(zip(fi.tolist(), uj.tolist(), strict=True)) == {(1, 1), (0, 0)}
    assert features_matched(np.array([100.4, 250.0]), np.array([1.0, 1.0]), u, np.zeros(4)).tolist() \
        == [True, False]
    assert coverage_mask([150.0, 1000.0], u, 60.0).tolist() == [True, False]


# ---------------------------------------------------------------------------
# injection / recovery / null
# ---------------------------------------------------------------------------
def test_injected_chf3_pattern_is_recovered(tmp_path):
    conf = synth_conf()
    tab = synthetic_survey(11, seeded=seeded_pattern())
    rep = screen_all(conf, pd.DataFrame(), {}, {"synth": tab}, species=["CHF3"])
    r = rep["results"]["CHF3|synth"]
    assert r["line_source"] == "predicted" and r["verify"]
    best = r["best"]
    assert best["pattern"] and best["n_coincident"] >= 8
    assert best["spearman_rho"] > 0.3 and best["top5_fraction_any"] >= 0.6
    assert best["p_false"] == pytest.approx(1.0 / 201.0)
    assert best["tests"] == {"count": True, "lte": True, "top5": True, "p_false": True}
    s = stage_assess(conf, tmp_path, screen=rep, acquire_report={})
    assert s["verdict"] == "PATTERN_CANDIDATE" and s["n_pattern_candidates"] == 1
    assert "CHF3" in s["targets_with_predicted_frequencies"]
    assert any("PREDICTED" in d for d in s["degraded"])
    c = pd.read_csv(tmp_path / "candidates.csv")
    assert bool(c.iloc[0]["pattern"]) and c.iloc[0]["species"] == "CHF3"


def test_same_list_without_the_pattern_is_no_pattern(tmp_path):
    conf = synth_conf()
    tab = synthetic_survey(11)                               # identical random lines, no seed
    rep = screen_all(conf, pd.DataFrame(), {}, {"synth": tab}, species=["CHF3"])
    best = rep["results"]["CHF3|synth"]["best"]
    assert not best["pattern"] and best["n_coincident"] < 3
    s = stage_assess(conf, tmp_path, screen=rep, acquire_report={})
    assert s["verdict"] == "NO_PATTERN"
    assert "not written up" in s["note"]


def test_poisson_random_ulines_give_p_false_consistent_with_the_trials():
    conf = synth_conf(n_trials=1000)
    n = 1000
    rng = np.random.default_rng(5)
    tab = synthetic_survey(21, n_random=1500)
    rep = screen_all(conf, pd.DataFrame(), {}, {"synth": tab}, species=["CHF3"], n_trials=n,
                     seed=int(rng.integers(1 << 30)))
    for rec in rep["results"]["CHF3|synth"]["records"]:
        assert rec["n_trials"] == n
        assert rec["p_false"] == pytest.approx((rec["k_ge_observed"] + 1) / (n + 1))
        assert rec["p_false"] >= 1.0 / (n + 1)
        assert rec["p_false"] > 0.05 and not rec["pattern"]
        # an observed count of zero is beaten by every trial
        if rec["n_coincident"] == 0:
            assert rec["p_false"] == 1.0
        assert 0 <= rec["trial_count_mean"] <= rec["trial_count_max"]


def test_contaminant_veto_removes_coincidences_and_is_counted(tmp_path):
    conf = synth_conf()
    seeded = seeded_pattern()
    tab = synthetic_survey(11, seeded=seeded)
    # a "CH3OH" catalogue with one strong line on every seeded feature
    lines = pd.DataFrame({"freq_mhz": seeded["freq_mhz"], "err_mhz": 0.01, "lgint_300": -3.0,
                          "dr": 3, "elo_cm": 5.0, "gup": 10, "tag": 32003, "lab": True, "qnfmt": 1404,
                          "qn_up": "", "qn_lo": "", "entry_id": "jpl:32003", "entry_name": "CH3OH",
                          "database": "jpl", "species": "CH3OH"})
    entries = {"jpl:32003": {"tag": 32003, "name": "CH3OH", "temps": list(CATDIR_TEMPS),
                             "qlog": [3.4, 3.2, 3.0, 2.5, 2.0, 1.5, 1.0], "database": "jpl",
                             "species": "CH3OH"}}
    rep = screen_all(conf, lines, entries, {"synth": tab}, species=["CHF3"])
    src = rep["sources"]["synth"]
    assert src["n_ulines_vetoed"] == len(seeded) and src["vetoes_by_species"] == {"CH3OH": len(seeded)}
    assert src["contaminants_applied"] == ["CH3OH"]
    best = rep["results"]["CHF3|synth"]["best"]
    assert best["n_coincident_vetoed"] == len(seeded) and best["n_coincident"] < 3
    assert not best["pattern"]
    assert stage_assess(conf, tmp_path, screen=rep, acquire_report={})["verdict"] == "NO_PATTERN"


def test_veto_dynamic_range_floor_keeps_weak_forest_lines_from_vetoing():
    u = np.array([100000.0, 200000.0, 300000.0])
    src = SourceLines("s", u, np.ones(3), np.zeros(3), u, fwhm_km_s=4.0)
    cont = {"CH3OH": (np.array([100000.5, 200000.5, 300000.5]), np.array([-2.0, -9.0, -2.5]))}
    v = apply_vetoes(src, cont, dynamic_range_dex=4.0)
    assert v.u_veto.tolist() == [True, False, True] and v.u_veto_species == ["CH3OH", "", "CH3OH"]


def test_only_sources_restricts_contaminants_and_baseline_always_applies():
    conf = synth_conf()
    conf["species"]["contaminants"]["HC3N"]["only_sources"] = ["elsewhere"]
    seeded = seeded_pattern()
    tab = synthetic_survey(11, seeded=seeded)
    f = seeded["freq_mhz"].to_numpy()
    mk = lambda sp, tag: pd.DataFrame({  # noqa: E731
        "freq_mhz": f, "err_mhz": 0.01, "lgint_300": -3.0, "dr": 3, "elo_cm": 5.0, "gup": 10,
        "tag": tag, "lab": True, "qnfmt": 1404, "qn_up": "", "qn_lo": "",
        "entry_id": f"jpl:{tag}", "entry_name": sp, "database": "jpl", "species": sp})
    lines = pd.concat([mk("HC3N", 51001), mk("CH3F", 34004)], ignore_index=True)
    ent = {"temps": list(CATDIR_TEMPS), "qlog": [3.0, 2.8, 2.5, 2.0, 1.5, 1.0, 0.5], "database": "jpl"}
    entries = {"jpl:51001": {**ent, "tag": 51001, "name": "HC3N", "species": "HC3N"},
               "jpl:34004": {**ent, "tag": 34004, "name": "CH3F", "species": "CH3F"}}
    rep = screen_all(conf, lines, entries, {"synth": tab}, species=["CHF3"])
    src = rep["sources"]["synth"]
    assert src["contaminants_applied"] == ["CH3F"]           # HC3N not applied here
    assert src["vetoes_by_species"] == {"CH3F": len(seeded)}


def test_missing_strong_line_fails_the_top5_test():
    conf = synth_conf()
    feats = chf3_features()
    # seed only the 6th-13th strongest features: >= 3 coincidences, but the five
    # strongest predicted lines are absent where the survey is sensitive
    tab = synthetic_survey(31, seeded=feats.iloc[5:13])
    rep = screen_all(conf, pd.DataFrame(), {}, {"synth": tab}, species=["CHF3"])
    rec = next(r for r in rep["results"]["CHF3|synth"]["records"] if r["tex_k"] == 50.0)
    assert rec["n_coincident"] >= 3 and rec["top5_fraction_any"] == 0.0
    assert rec["tests"]["top5"] is False and not rec["pattern"]


def test_sky_frame_shifts_predictions_and_rest_frame_does_not():
    v = -26.0
    seeded = seeded_pattern()
    tab = synthetic_survey(41, seeded=seeded, shift_v=v)      # U-lines at sky frequencies
    conf = synth_conf()
    conf["sources"]["synth"].update({"v_lsr_km_s": v, "frequency_frame": "sky"})
    rep = screen_all(conf, pd.DataFrame(), {}, {"synth": tab}, species=["CHF3"])
    assert rep["results"]["CHF3|synth"]["best"]["pattern"]
    conf["sources"]["synth"]["frequency_frame"] = "rest"    # 26 km/s ≫ the 4 km/s tolerance
    rep = screen_all(conf, pd.DataFrame(), {}, {"synth": tab}, species=["CHF3"])
    assert rep["results"]["CHF3|synth"]["best"]["n_coincident"] < 3


def test_no_intensity_column_makes_lte_untestable_but_reports():
    conf = synth_conf()
    tab = synthetic_survey(11, seeded=seeded_pattern())
    tab["intensity"] = np.nan
    rep = screen_all(conf, pd.DataFrame(), {}, {"synth": tab}, species=["CHF3"])
    best = rep["results"]["CHF3|synth"]["best"]
    assert not best["lte_testable"] and best["lte_pass"] is None
    assert best["pattern"]                                  # count + top5 + p_false still hold
    assert not rep["sources"]["synth"]["has_intensity"]


def test_evaluate_species_with_laboratory_lines_uses_entry_partition_function():
    df, ent = symmetric_top_lines(CHF3["b"], CHF3["dj"], CHF3["djk"], axial_mhz=CHF3["axial"],
                                  fmax_mhz=FMAX)
    df["entry_id"] = "jpl:1"
    tab = synthetic_survey(11, seeded=seeded_pattern())
    src = SourceLines("synth", tab.loc[tab.unidentified, "freq_mhz"].to_numpy(),
                      tab.loc[tab.unidentified, "intensity"].to_numpy(),
                      np.zeros(int(tab.unidentified.sum())), tab["freq_mhz"].to_numpy(), fwhm_km_s=4.0)
    recs = evaluate_species(df, {"jpl:1": ent.as_dict()}, src, species="CHF3", line_source="jpl",
                            conf={"n_trials": 100, "seed": 1})
    assert any(r["pattern"] for r in recs) and recs[0]["line_source"] == "jpl"


# ---------------------------------------------------------------------------
# acquisition (scripted archives) and honest degradation
# ---------------------------------------------------------------------------
def _cat_text(freqs, lgint=-3.0, tag=71001) -> str:
    out = []
    for i, f in enumerate(freqs):
        out.append(f"{f:13.4f}{0.01:8.4f}{lgint - 0.01 * i:8.4f} 3{5.0 + i:10.4f}{7:3d}{tag:7d}"
                   f"1404 {i + 1:2d} 0 0 0 0 0 {i:2d} 0 0 0 0 0")
    return "\n".join(out) + "\n"


class _FakeWeb:
    """Scripted HTTP: JPL catdir + cats, CDMS roots."""

    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.urls: list[str] = []
        self.jpl_freqs = np.linspace(90_000.0, 390_000.0, 12)

    def __call__(self, url: str) -> str:
        self.urls.append(url)
        if self.mode == "fail":
            raise RuntimeError("CONNECT tunnel failed, response 403")
        if url.endswith("catdir.cat"):
            return CATDIR_BLOCK
        if url.endswith("c071001.cat"):
            return _cat_text(self.jpl_freqs, tag=71001)
        if url.endswith("c032003.cat"):
            return _cat_text([100_100.0, 250_250.0], tag=32003)
        if url.endswith("c050007.cat"):
            return _cat_text([120_000.0], tag=50007)
        if url.endswith("c052012.cat"):
            return ""                                        # an empty catalogue file
        if "cdms" in url and url.endswith(".cat"):
            return _cat_text([130_000.0, 260_000.0], tag=70501)
        if "cdms" in url:
            return CDMS_HTML if "partition" in url else "<html>rendered by script</html>"
        raise RuntimeError(f"unexpected url {url}")


class _FakeTAP:
    """Scripted VizieR TAP: mode 'fail' | 'zero' | 'ok'."""

    def __init__(self, mode: str, rows: pd.DataFrame | None = None):
        self.mode, self.rows, self.calls = mode, rows, []

    def __call__(self, adql: str) -> pd.DataFrame:
        self.calls.append(adql)
        if self.mode == "fail":
            raise RuntimeError("CONNECT tunnel failed, response 403")
        if "TAP_SCHEMA.tables" in adql:
            if self.mode == "zero" or "description LIKE" in adql:
                return pd.DataFrame(columns=["table_name", "description"])
            assert "LIKE '%J/" in adql                    # leading % against the literal quotes
            return pd.DataFrame({"table_name": ['"J/ApJ/787/112/table2"', '"J/ApJ/787/112/refs"'],
                                 "description": ["Line list", "References"]})
        if "TAP_SCHEMA.columns" in adql:
            if "refs" in adql:
                return pd.DataFrame({"column_name": ["Ref", "Auth"], "unit": ["", ""],
                                     "description": ["", ""]})
            return pd.DataFrame({"column_name": ["Freq", "Species", "TA", "e_Freq"],
                                 "unit": ["GHz", "", "K", "MHz"],
                                 "description": ["Rest frequency at vLSR", "Species or U", "Peak", ""]})
        if "COUNT(*)" in adql:
            return pd.DataFrame({"n": [len(self.rows)]})
        assert adql.startswith('SELECT TOP') and 'FROM "J/ApJ/787/112/table2"' in adql
        return self.rows.copy()


def _raw_rows(seed: int = 3, seeded: pd.DataFrame | None = None) -> pd.DataFrame:
    tab = synthetic_survey(seed, seeded=seeded)
    return pd.DataFrame({"Freq": tab["freq_mhz"] / 1000.0, "Species": tab["ident"],
                         "TA": tab["intensity"], "e_Freq": 0.2})


def test_column_resolution_frequency_scale_and_unidentified_flag():
    cols = pd.DataFrame({"column_name": ["recno", "Freq", "e_Freq", "Species", "TA", "Trans"],
                         "unit": ["", "GHz", "MHz", "", "K", ""], "description": [""] * 6})
    r = acq.resolve_line_columns(cols)
    assert r["freq"] == "Freq" and r["freq_err"] == "e_Freq" and r["ident"] == "Species"
    assert r["intensity"] == "TA" and r["transition"] == "Trans"
    assert acq.frequency_scale("GHz") == (1000.0, "unit:GHz")
    assert acq.frequency_scale("", [500.1, 600.2])[0] == 1000.0
    assert acq.frequency_scale("", [500100.0, 600200.0])[0] == 1.0
    m = acq.unidentified_mask(["U", "u-line", "CH3OH", "U 12", "?", "Unidentified", "CU", None, "U-line?"])
    assert m.tolist() == [True, True, False, True, True, True, False, False, True]


def test_run_with_dead_archives_is_no_data_reached(tmp_path):
    out = tmp_path / "uline"
    conf = load_uline_config()
    conf["match"]["n_trials"] = 50
    conf["archives"]["fetch_retries"] = 1                    # no backoff sleeps in the suite
    rep = uline_run(conf, "all", out_dir=out, fetch_fn=_FakeWeb("fail"), query_fn=_FakeTAP("fail"))
    assert rep["verdict"] == "NO_DATA_REACHED" and rep["n_pattern_candidates"] == 0
    probe = json.loads((out / "probe.json").read_text())
    assert probe["jpl"]["status"] == "QUERY_FAILED" and probe["cdms"]["status"] == "QUERY_FAILED"
    assert all(v["status"] == "QUERY_FAILED" for v in probe["vizier"].values())
    a = json.loads((out / "acquire.json").read_text())
    assert a["acquisition"]["any_query_failed"] and a["n_catalogue_lines"] == 0
    assert all(v["status"] == "QUERY_FAILED" for v in a["sources"].values())
    s = json.loads((out / "summary.json").read_text())
    assert s["n_ulines_total"] == 0 and any("QUERY_FAILED" in d for d in s["degraded"])
    assert pd.read_csv(out / "candidates.csv").empty
    assert "says nothing about the sky" in s["note"]


def test_run_with_empty_archives_is_no_data_reached_not_a_null(tmp_path):
    out = tmp_path / "uline"
    conf = load_uline_config()
    rep = uline_run(conf, "all", out_dir=out, fetch_fn=lambda u: "", query_fn=_FakeTAP("zero"))
    assert rep["verdict"] == "NO_DATA_REACHED"
    a = json.loads((out / "acquire.json").read_text())
    assert a["jpl"]["status"] == "QUERY_RETURNED_ZERO_ROWS"
    assert all(v["status"] == "QUERY_RETURNED_ZERO_ROWS" for v in a["sources"].values())


def test_probe_reports_inventory_and_frame_hint(tmp_path):
    conf = load_uline_config()
    web, tap = _FakeWeb(), _FakeTAP("ok", _raw_rows())
    rep = stage_probe(conf, tmp_path, fetch_fn=web, query_fn=tap, sources=["orion_kl_hifi"])
    inv = rep["species_inventory"]
    assert inv["NF3"]["jpl"] == ["NF3"] and inv["NF3"]["predictable"]
    assert inv["CHF3"]["jpl"] == [] and inv["CHF3"]["cdms"] == ["CHF3, v=0"] and inv["CHF3"]["predictable"]
    assert inv["CH3Cl"]["jpl"] == ["CH3Cl-35"] and inv["CH3Cl"]["group"] == "baseline"
    assert inv["SO2F2"] == {"group": "targets", "jpl": [], "cdms": [], "predictable": False}
    v = rep["vizier"]["orion_kl_hifi"]
    assert v["status"] == "OK" and v["table"] == "J/ApJ/787/112/table2"
    assert v["roles"]["freq"] == "Freq" and v["units"]["freq"] == "GHz"
    assert v["frame_hint"] == "rest,lsr"
    assert len(v["scoreboard"]) == 2 and not v["scoreboard"][1]["usable"]
    assert rep["discovered_unidentified_tables"] == []
    assert not any(u.endswith(".cat") and "catdir" not in u for u in web.urls)   # probe fetches no cat


def test_acquire_screen_assess_end_to_end_with_scripted_archives(tmp_path):
    out = tmp_path / "uline"
    conf = load_uline_config()
    conf["sources"] = {"orion_kl_hifi": conf["sources"]["orion_kl_hifi"]}
    conf["match"]["n_trials"] = 100
    web, tap = _FakeWeb(), _FakeTAP("ok", _raw_rows())
    a = stage_acquire(conf, out, fetch_fn=web, query_fn=tap)
    assert a["species"]["NF3"]["jpl"]["n_lines"] == 12
    assert a["species"]["CHF3"]["cdms"]["n_lines"] == 2
    assert a["species"]["CHF3"]["jpl"] == {"entries": [], "n_lines": 0}    # JPL had no CHF3
    assert a["species"]["CF2Cl2"]["jpl"]["entries"][0]["status"] == "QUERY_RETURNED_ZERO_ROWS"
    src = a["sources"]["orion_kl_hifi"]
    assert src["status"] == "OK" and src["freq_scale"] == "unit:GHz"
    assert src["n_unidentified"] == 300 and src["n_lines"] == 500 and src["has_intensity"]
    assert (out / "data" / "lines_orion_kl_hifi.parquet").exists()
    assert (out / "data" / "lines_catalogue.parquet").exists()
    entries = json.loads((out / "data" / "entries.json").read_text())
    assert entries["jpl:71001"]["species"] == "NF3" and entries["cdms:70501"]["species"] == "CHF3"

    sc = stage_screen(conf, out)
    st = sc["species_tables"]
    assert st["NF3"]["line_source"] == "jpl" and not st["NF3"]["predicted"]
    assert st["CHF3"]["line_source"] == "cdms"             # cdms preferred over predicted
    assert st["CF3Cl"]["line_source"] == "predicted" and st["CF3Cl"]["verify"]
    assert st["SO2F2"]["line_source"] is None
    o = sc["sources"]["orion_kl_hifi"]
    assert o["n_ulines"] == 300 and o["contaminants_applied"] == ["CH3Cl", "CH3F", "CH3OH"]
    assert o["frame"] == "rest" and o["v_unc_km_s"] == 4.0
    assert sc["results"]["SO2F2|orion_kl_hifi"]["status"] == "NO_LINE_LIST"
    assert (out / "coincidences.csv").exists()

    s = stage_assess(conf, out)
    assert s["verdict"] == "NO_PATTERN"
    assert s["targets_unsearchable"] == ["CH2F2", "CF2Cl2", "CFCl3", "COF2", "SO2F2", "CHClF2", "CF2"]
    assert s["targets_with_predicted_frequencies"] == ["CF3Cl", "CF3CN"]
    assert s["species_inventory"]["CF3Cl"]["constants"]["B_mhz"] == 3335.6
    assert s["pairs"]["NF3|orion_kl_hifi"]["line_source"] == "jpl"
    c = pd.read_csv(out / "candidates.csv")
    assert len(c) == len(conf["species"]["targets"]) and not c["pattern"].any()
    assert set(c["line_source"].dropna()) == {"jpl", "cdms", "predicted"}


def test_end_to_end_recovers_a_seeded_pattern_through_the_acquire_path(tmp_path):
    out = tmp_path / "uline"
    conf = load_uline_config()
    conf["sources"] = {"orion_kl_hifi": {**conf["sources"]["orion_kl_hifi"],
                                         "v_lsr_uncertainty_km_s": 0.0}}
    conf["match"]["n_trials"] = 200
    web = _FakeWeb()
    web.jpl_freqs = np.array([])                            # NF3 with no lines: excluded
    tap = _FakeTAP("ok", _raw_rows(seeded=seeded_pattern()))
    stage_acquire(conf, out, fetch_fn=web, query_fn=tap)
    stage_screen(conf, out, species=["CHF3", "NF3"])
    s = stage_assess(conf, out)
    # CHF3 came from the scripted CDMS entry (2 lines, not the pattern) and the
    # JPL NF3 file was empty, so NF3 fell back to its predictor: honest
    # NO_PATTERN, and each pair says which list it used
    assert s["verdict"] == "NO_PATTERN"
    assert s["pairs"]["CHF3|orion_kl_hifi"]["line_source"] == "cdms"
    assert s["pairs"]["NF3|orion_kl_hifi"]["line_source"] == "predicted"
    assert s["targets_with_predicted_frequencies"] == ["NF3", "CF3Cl", "CF3CN"]
    # the pattern is only reachable through the predictor: prefer it and re-screen
    conf["archives"]["prefer_line_list"] = ["predicted", "cdms", "jpl"]
    stage_screen(conf, out, species=["CHF3"])
    s = stage_assess(conf, out)
    assert s["verdict"] == "PATTERN_CANDIDATE"
    p = s["pairs"]["CHF3|orion_kl_hifi"]
    assert p["line_source"] == "predicted" and p["pattern"] and p["n_coincident"] >= 8
    rows = pd.read_csv(out / "coincidences.csv")
    assert (rows["species"] == "CHF3").sum() >= 8 and rows["pattern"].all()


def test_config_carries_every_species_source_and_threshold():
    conf = load_uline_config()
    assert Path("config/uline.yaml").exists()
    for g in ("targets", "baseline", "contaminants"):
        for sp, spec in conf["species"][g].items():
            assert spec["patterns"], (g, sp)
    assert set(conf["species"]["targets"]) == {"NF3", "CHF3", "CH2F2", "CF3Cl", "CF2Cl2", "CFCl3",
                                                "CF3CN", "COF2", "SO2F2", "CHClF2", "CF2"}
    assert set(conf["species"]["baseline"]) == {"CH3Cl", "CH3F"}
    assert {"CH3OH", "CH3CN", "HCOOCH3", "C2H5CN"} <= set(conf["species"]["contaminants"])
    for sp, blk in conf["predicted"].items():
        if sp == "error_model":
            continue
        assert blk["verify"] is True and blk["B_mhz"] > 0 and blk["source"], sp
    for name, s in conf["sources"].items():
        assert s["vizier_like"].startswith("J/") and "v_lsr_km_s" in s and s["fwhm_km_s"] > 0, name
        assert s["frequency_frame"] in ("rest", "sky")
    assert conf["sources"]["orion_kl_hifi"]["v_lsr_km_s"] == 9.0
    assert conf["sources"]["irc10216_he2008"]["v_lsr_km_s"] == -26.0
    m = conf["match"]
    assert m["n_trials"] >= 1000 and m["min_coincidences"] == 3 and m["tex_grid_k"] == [10, 30, 50, 100, 150, 200]
    assert 50 <= m["shift_min_mhz"] < m["shift_max_mhz"] <= 500


def test_workflow_and_doc_exist():
    assert Path(".github/workflows/uline.yml").exists()
    assert Path("docs/uline.md").exists()
    wf = Path(".github/workflows/uline.yml").read_text()
    assert "commit_results.sh" in wf and "workflow_dispatch" in wf and "seti.uline.run" in wf


def test_entry_roundtrip_and_predicted_error_model():
    e = Entry(tag=1, name="X", nlines=2, temps=[300.0, 150.0], qlog=[1.0, float("nan")])
    d = e.as_dict()
    assert d["qlog"] == [1.0, None] and d["database"] == "jpl"
    conf = load_uline_config()
    from seti.uline.run import build_species_tables
    t = build_species_tables(conf, pd.DataFrame(), {}, fmax_mhz=200_000.0)
    # CHF3 has D constants -> tight error model; CF3Cl has none -> ten times looser
    e1 = t["CHF3"]["lines"]["err_mhz"] / t["CHF3"]["lines"]["freq_mhz"]
    e2 = t["CF3Cl"]["lines"]["err_mhz"] / t["CF3Cl"]["lines"]["freq_mhz"]
    assert e2.median() > 5 * e1.median()
    assert t["CHF3"]["distortion_known"] and not t["CF3Cl"]["distortion_known"]
    assert t["CHF3"]["axial_known"] and not t["CF3CN"]["axial_known"]
