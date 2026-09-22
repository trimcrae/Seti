"""Offline test suite for ULINE --- industrial fluorine molecules in U-line lists.

No network anywhere (``conftest.py`` raises on any socket).  Per
``docs/channel-brief.md`` §5 the suite:

* recovers an injected CHF₃ K-ladder pattern seeded into a synthetic U-line
  list, and — because CHF₃'s constants are placeholders — reports it as
  ``PATTERN_CANDIDATE_VERIFY_CONSTANTS`` rather than ``PATTERN_CANDIDATE``,
  while returning ``NO_PATTERN`` on the same list with the pattern removed;
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
import re
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
    formula_key,
    interp_log_q,
    match_species,
    normalise_name,
    parse_cat,
    parse_catdir,
    parse_catdir_report,
    parse_partition_table,
    rescale_lgint,
    species_match_route,
    symmetric_top_lines,
    unparsed_catdir_lines,
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
    inv, rep = acq.jpl_inventory(CATDIR_BLOCK, conf)
    assert [e.tag for e in inv["NF3"]] == [71001] and inv["CHF3"] == []
    assert rep["n_entries"] == 5 and rep["n_unparsed_lines"] == 1
    assert rep["unparsed_sample"] == ["not a directory line"]


# The layout JPL actually publishes (necrofrontier probe, endpoint jpl_catdir):
# a "2*" / "1* " version column, names with spaces, commas, hyphens and "+",
# and a version that is sometimes missing.  The old rigid regex dropped 171 of
# 403 non-blank lines here; every line below except the last must parse.
CATDIR_AWKWARD = (
    "  1001 H-atom            1 0.6021 0.6021 0.6021 0.6021 0.6021 0.6021 0.6021 1\n"
    "  4001 H2D+             32 1.8834 1.6986 1.4401 0.9882 0.4919 0.0846 0.0016 2*\n"
    " 14003 13CH            648 2.3562 2.2249 2.0368 1.7027 1.3423 0.9177 0.3133 1* \n"
    " 18005 H2O v2,2v2,v   8608 2.2507 2.0645 1.8040 1.3649 0.9335 0.4819 0.0994 4* \n"
    " 14002 N-atom-D-st       6 1.4700 1.4676 1.4629 1.4495 1.4246 1.3827 1.3247 3\n"
    " 71001 NF3            9999 4.1000 3.9000 3.6000 3.2000 2.7000 2.3000 1.8000\n"
    " 52011 CH2F2-v4       7808 4.7435 4.5317 4.2551 3.8010 3.3502 2.9003 2.4522 1 extra\n"
    " 99999 Broken --- not a catdir row at all\n"
)


def test_catdir_parser_tolerates_the_real_jpl_layout():
    ents, unparsed = parse_catdir_report(CATDIR_AWKWARD)
    assert [e.tag for e in ents] == [1001, 4001, 14003, 18005, 14002, 71001, 52011]
    by_tag = {e.tag: e for e in ents}
    assert by_tag[4001].name == "H2D+" and by_tag[4001].version == "2*"       # the "2*" version
    assert by_tag[14003].name == "13CH" and by_tag[14003].version == "1*"     # trailing blanks
    assert by_tag[18005].name == "H2O v2,2v2,v" and by_tag[18005].nlines == 8608  # spaces in name
    assert by_tag[14002].name == "N-atom-D-st" and by_tag[14002].version == "3"
    assert by_tag[71001].version == ""                                        # version absent
    assert by_tag[52011].version == "1 extra"                                 # extra columns kept
    assert by_tag[1001].temps == list(CATDIR_TEMPS) and len(by_tag[1001].qlog) == 7
    assert np.isclose(by_tag[4001].qlog[-1], 0.0016)
    # a line that still will not parse is counted and kept verbatim, never dropped
    assert count_unparsed_catdir(CATDIR_AWKWARD) == 1
    assert unparsed == [" 99999 Broken --- not a catdir row at all"]
    assert unparsed_catdir_lines(CATDIR_AWKWARD) == unparsed
    assert len(unparsed_catdir_lines(CATDIR_AWKWARD * 40, limit=20)) == 20


def test_normalised_formula_matching_beats_the_anchored_regexes():
    # every spelling JPL and CDMS actually use for the same molecule. "H3C-Cl"
    # reverses the atom order and so needs a DECLARED pattern: an undeclared
    # same-formula name is a near miss, because atom counts cannot tell a
    # reversed spelling from an isomer.
    import yaml as _yaml
    _pat = _yaml.safe_load(Path("config/uline.yaml").read_text())
    _ch3cl = [re.compile(p) for p in _pat["species"]["baseline"]["CH3Cl"]["patterns"]]
    for name in ("CH3Cl", "CH3CL", "CH3-35Cl", "CH3-37Cl", "CH3Cl, v=0", "CH3-35Cl, v=0",
                 "H3C-Cl", "CH3(35)Cl", "CH 3 Cl", "CH3Cl-35"):
        assert species_match_route(name, "CH3Cl", _ch3cl) is not None, name
    assert species_match_route("H3C-Cl", "CH3Cl") is None        # undeclared: a near miss
    assert species_match_route("CH3-35Cl, v=0", "CH3Cl") == "decorated"
    assert species_match_route("13CH3OH", "CH3OH") == "isotopologue"
    assert formula_key("HCCCN") == formula_key("HC3N") == "C3H1N1"
    assert formula_key("SiCC") == formula_key("SiC2") == "C2Si1"
    assert formula_key("F2CO") == formula_key("CF2O") == formula_key("COF2") == "C1F2O1"
    assert formula_key("CO") == "C1O1" and formula_key("Co") == "Co1"   # carbon+oxygen vs cobalt
    assert formula_key("CF+") != formula_key("CF")                      # charge is kept
    assert formula_key("H-atom") is None                                # not a formula: no match
    # and it must NOT over-match
    assert species_match_route("CF2Cl2", "CF2") is None
    assert species_match_route("CH3CN", "CH3OH") is None
    assert normalise_name("CH3-35Cl, v=0") == "ch335clv0"


def test_near_miss_names_are_recorded_when_nothing_matches():
    ents = [Entry(tag=1, name="CF2Cl2, v=0", nlines=1), Entry(tag=2, name="CF2Cl2-fake", nlines=1),
            Entry(tag=3, name="CH3OH", nlines=1)]
    m = match_species(ents, "CF2", patterns=[r"^CF2\b"])
    assert m.entries == [] and m.matched_names == []
    assert m.near_miss_names == ["CF2Cl2, v=0", "CF2Cl2-fake"]    # the catalogue's own spellings
    m2 = match_species(ents, "CF2Cl2", patterns=[])
    assert [e.tag for e in m2.entries] == [1]                    # the v=0 state IS CF2Cl2
    assert [d["route"] for d in m2.matched_names] == ["decorated"]
    assert m2.near_miss_names == ["CF2Cl2-fake"]                 # a suffix nobody has explained yet
    # the configured regex stays an ADDITIONAL route
    m3 = match_species([Entry(tag=4, name="weird-name-for-NF3", nlines=1)], "NF3",
                       patterns=[r"NF3$"])
    assert [d["route"] for d in m3.matched_names] == ["regex"]


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
    # The signal is recovered — and CHF3's constants are placeholders carrying
    # `verify: true`, so the verdict says the pattern rests on them.  That is
    # the point of the split: the search must find an injected pattern, and it
    # must not dress a reconstructed frequency up as a catalogued one.
    assert s["verdict"] == "PATTERN_CANDIDATE_VERIFY_CONSTANTS"
    assert s["n_pattern_candidates"] == 0
    assert s["n_pattern_candidates_verify_constants"] == 1
    assert s["pattern_candidates_verify_constants"] == ["CHF3|synth"]
    assert "CHF3" in s["targets_with_predicted_frequencies"]
    assert any("PREDICTED" in d for d in s["degraded"])
    c = pd.read_csv(tmp_path / "candidates.csv")
    assert bool(c.iloc[0]["pattern"]) and c.iloc[0]["species"] == "CHF3"
    assert bool(c.iloc[0]["verify_constants"])


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


class _QuotedSchemaTAP:
    """TAPVizieR as it really is: ``table_name`` carries literal double quotes.

    A LIKE without a leading ``%`` therefore matches nothing, and a FROM
    without quotes is a syntax error — the two mistakes this class refuses.
    """

    def __init__(self, rows: pd.DataFrame | None = None):
        self.rows = rows if rows is not None else _raw_rows()
        self.calls: list[str] = []

    def __call__(self, adql: str) -> pd.DataFrame:
        self.calls.append(adql)
        if "TAP_SCHEMA.tables" in adql:
            if "description LIKE" in adql:
                return pd.DataFrame(columns=["table_name", "description"])
            if "LIKE '%J/ApJ/787/112/%'" not in adql:        # no leading % -> no rows, as in reality
                return pd.DataFrame(columns=["table_name", "description"])
            return pd.DataFrame({"table_name": ['"J/ApJ/787/112/table2"'],
                                 "description": ["Unidentified lines in the Orion KL survey"]})
        if "TAP_SCHEMA.columns" in adql:
            if "'\"J/ApJ/787/112/table2\"'" not in adql:      # the quoted spelling must be tried
                return pd.DataFrame(columns=["column_name", "unit", "description"])
            return pd.DataFrame({"column_name": ["Freq", "Species", "TA"],
                                 "unit": ["GHz", "", "K"],
                                 "description": ["Rest frequency", "Species or U", "Peak"]})
        assert 'FROM "J/ApJ/787/112/table2"' in adql          # quoted when selected from
        if "COUNT(*)" in adql:
            return pd.DataFrame({"n": [len(self.rows)]})
        return self.rows.copy()


def test_tapvizier_quoted_table_names_are_found_and_selected_from():
    tap = _QuotedSchemaTAP()
    d = acq.discover_line_table("orion_kl_hifi", "J/ApJ/787/112/", query_fn=tap)
    assert d.status == "OK" and d.table == "J/ApJ/787/112/table2"     # unquoted for downstream use
    assert d.roles["freq"] == "Freq" and d.roles["ident"] == "Species"
    assert acq.tables_like_adql("J/ApJ/787/112/").count("LIKE '%") == 1
    assert d.queries[0]["adql"] == acq.tables_like_adql("J/ApJ/787/112/")
    assert "LIKE '%J/ApJ/787/112/%'" in d.queries[0]["adql"]          # LEADING %
    assert not d.errors and d.fallback == {}
    df = acq.fetch_line_table(d, query_fn=tap)
    assert len(df) == 500 and df["unidentified"].sum() == 300
    assert any('FROM "J/ApJ/787/112/table2"' in c for c in tap.calls)


class _FailingTAP:
    def __init__(self, message: str):
        self.message, self.calls = message, []

    def __call__(self, adql: str):
        self.calls.append(adql)
        raise RuntimeError(self.message)


def test_a_failed_query_records_its_adql_and_its_error_text():
    msg = "DALServiceError: 503 Server Error: Service Unavailable for url: .../TAPVizieR/tap/sync"
    tap = _FailingTAP(msg)
    d = acq.discover_line_table("orion_kl_hifi", "J/ApJ/787/112/", query_fn=tap,
                                fallback_terms_all=["unidentified"],
                                fallback_terms_any=["Orion", "line survey"])
    assert d.status == "QUERY_FAILED"
    assert len(d.errors) == 2                                 # the id search and the fallback
    first = d.errors[0]
    assert first["adql"] == acq.tables_like_adql("J/ApJ/787/112/") and msg in first["error"]
    assert d.fallback["status"] == "QUERY_FAILED" and msg in d.fallback["error"]
    assert "description LIKE '%unidentified%'" in d.fallback["adql"]


def test_description_fallback_records_every_table_when_the_asserted_id_is_absent():
    tables = pd.DataFrame({"table_name": ['"J/ApJS/177/275/table1"', '"J/other/1/2/ulines"'],
                           "description": ["Unidentified lines, IRC+10216 line survey",
                                           "Unidentified features"]})

    def tap(adql: str) -> pd.DataFrame:
        if "TAP_SCHEMA.tables" not in adql:
            raise AssertionError(adql)
        return tables.copy() if "description LIKE" in adql else tables.iloc[:0].copy()

    d = acq.discover_line_table("irc10216_he2008", "J/ApJS/177/275/", query_fn=tap,
                                fallback_terms_all=["unidentified"],
                                fallback_terms_any=["IRC+10216", "line survey"])
    assert d.status == "QUERY_RETURNED_ZERO_ROWS" and d.table is None
    fb = d.fallback
    adql = fb["adql"]
    assert adql.count("LIKE '%") == adql.count("LIKE '")       # every LIKE has a leading %
    for w in ("%unidentified%", "%Unidentified%", "%IRC+10216%", "%line survey%"):
        assert f"LIKE '{w}'" in adql, w
    assert " AND " in adql                                     # 'unidentified' AND (Orion | survey)
    assert fb["n_tables"] == 2
    assert [t["table_name"] for t in fb["tables"]] == ["J/ApJS/177/275/table1",
                                                       "J/other/1/2/ulines"]
    # diagnostic only: nothing is selected from the fallback automatically
    assert d.table is None and d.roles == {}


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
    # a bare QUERY_FAILED is undiagnosable: every failure carries its ADQL and error
    for v in a["sources"].values():
        assert v["errors"], v
        assert "TAP_SCHEMA.tables" in v["errors"][0]["adql"] and "LIKE '%" in v["errors"][0]["adql"]
        assert "403" in v["errors"][0]["error"]
        assert v["fallback"]["status"] == "QUERY_FAILED" and "403" in v["fallback"]["error"]
    s = json.loads((out / "summary.json").read_text())
    assert s["n_ulines_total"] == 0 and any("QUERY_FAILED" in d for d in s["degraded"])
    assert any("403" in d for d in s["degraded"])              # the reason, not just the status
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
    # SO2F2 is in neither catalogue, and is now PREDICTABLE from the rotor
    # assets (src/seti/data_assets/rotor_constants.yaml) rather than
    # unsearchable: the probe names the constants it would use.
    assert inv["SO2F2"]["jpl"] == [] and inv["SO2F2"]["cdms"] == []
    assert inv["SO2F2"]["group"] == "targets" and inv["SO2F2"]["predictable"]
    assert inv["SO2F2"]["rotor_constants"]["role"] == "target"
    assert [i["name"] for i in inv["SO2F2"]["rotor_constants"]["isotopologues"]] == ["SO2F2 v=0"]
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
    assert a["species"]["CHF3"]["jpl"]["entries"] == []                    # JPL had no CHF3
    assert a["species"]["CHF3"]["jpl"]["n_lines"] == 0
    assert a["species"]["CF2Cl2"]["jpl"]["entries"][0]["status"] == "QUERY_RETURNED_ZERO_ROWS"
    # a species that matched nothing still names the catalogue's own spellings
    assert a["species"]["CF2"]["jpl"]["entries"] == []
    assert a["species"]["CF2"]["jpl"]["near_miss_names"] == ["CF2Cl2-fake"]
    assert a["species"]["CH3Cl"]["jpl"]["matched_names"][0]["name"] == "CH3Cl-35"
    assert a["jpl"]["unparsed_sample"] == ["not a directory line"]
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
    assert st["SO2F2"]["line_source"] == "rotor" and st["SO2F2"]["verify"]
    o = sc["sources"]["orion_kl_hifi"]
    assert o["n_ulines"] == 300 and o["contaminants_applied"] == ["CH3Cl", "CH3F", "CH3OH"]
    assert o["frame"] == "rest" and o["v_unc_km_s"] == 4.0
    assert sc["results"]["SO2F2|orion_kl_hifi"]["status"] == "OK"
    assert sc["results"]["SO2F2|orion_kl_hifi"]["line_source"] == "rotor"
    # and it says, in the same record, that the prediction is not sharp enough
    # to be matched at this source's linewidth - the quartic constants are
    # unknown, and that is the limit, not the absence of a catalogue entry
    assert sc["results"]["SO2F2|orion_kl_hifi"]["searchability"]["status"] in (
        "DEGRADED", "FREQUENCY_LIMITED")
    assert (out / "coincidences.csv").exists()

    s = stage_assess(conf, out)
    assert s["verdict"] == "NO_PATTERN"
    # NOTHING is unsearchable any more: the five species with no catalogue
    # entry are predicted from the rotor assets, and CH2F2/COF2 from the
    # validation blocks of the same file.
    assert s["targets_unsearchable"] == []
    assert set(s["targets_with_predicted_frequencies"]) == {
        "CH2F2", "CF3Cl", "CF2Cl2", "CFCl3", "CF3CN", "COF2", "SO2F2", "CHClF2", "CF2"}
    assert s["species_inventory"]["CF3Cl"]["constants"]["B_mhz"] == 3335.6
    assert s["pairs"]["NF3|orion_kl_hifi"]["line_source"] == "jpl"
    c = pd.read_csv(out / "candidates.csv")
    assert len(c) == len(conf["species"]["targets"]) and not c["pattern"].any()
    # `rotor` is the fourth line source, and its presence is the point: the five
    # species with no catalogue entry (CF2Cl2, CFCl3, SO2F2, CHClF2, CF2) are
    # predicted from src/seti/data_assets/rotor_constants.yaml rather than
    # reported unsearchable.  A candidates.csv with only the old three would
    # mean the asymmetric-top predictor had silently stopped reaching them.
    assert set(c["line_source"].dropna()) == {"jpl", "cdms", "predicted", "rotor"}
    rotor_rows = c[c["line_source"] == "rotor"]
    # The five species no catalogue carries must ALL be there.  CH2F2 and COF2
    # may join them: the assets hold validation blocks for both, and in this
    # scripted archive their JPL entries come back empty, so the predictor is
    # the correct fallback rather than `unsearchable`.
    assert {"CF2Cl2", "CFCl3", "SO2F2", "CHClF2", "CF2"} <= set(rotor_rows["species"])
    assert set(rotor_rows["species"]) <= {"CF2Cl2", "CFCl3", "SO2F2", "CHClF2", "CF2",
                                          "CH2F2", "COF2"}
    assert rotor_rows["verify_constants"].all()   # never a catalogued line list


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
    assert set(s["targets_with_predicted_frequencies"]) >= {"NF3", "CF3Cl", "CF3CN"}
    # the pattern is only reachable through the predictor: prefer it and re-screen
    conf["archives"]["prefer_line_list"] = ["predicted", "cdms", "jpl"]
    stage_screen(conf, out, species=["CHF3"])
    s = stage_assess(conf, out)
    # found, and flagged: the predictor's CHF3 block is `verify`, so this is a
    # reason to obtain the laboratory line list, not a candidate
    assert s["verdict"] == "PATTERN_CANDIDATE_VERIFY_CONSTANTS"
    assert s["n_pattern_candidates"] == 0
    assert "CHF3|orion_kl_hifi" in s["pattern_candidates_verify_constants"]
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
        # Every asserted catalogue id carries a DESCRIPTION fallback, so an id
        # absent from TAP_SCHEMA names the real catalogue next run. The terms
        # go in `any`, not `all`: requiring "unidentified" in a TABLE
        # description returned 0 rows in run 35038662696 because that word is
        # a COLUMN description (see column_census).
        terms = list(s.get("fallback_description_all") or []) + \
            list(s.get("fallback_description_any") or [])
        assert "unidentified" in terms, name
        assert "line survey" in terms, name
    assert conf["archives"]["tap_retries"] >= 3
    txt = Path("config/uline.yaml").read_text()
    assert "verify:" in txt and "NOT yet been confirmed against TAP_SCHEMA" in txt
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


# ---------------------------------------------------------------------------
# The VizieR route ladder (TAPVizieR 503'd every query on 2026-09-13)
# ---------------------------------------------------------------------------
ULINE_ASU_META = "\n".join([
    "#RESOURCE=yCat_J/ApJ/787/112",
    "#Name: J/ApJ/787/112",
    "#Title: Herschel/HIFI survey of Orion KL (Crockett+, 2014)",
    "#Table\tJ_ApJ_787_112_table2:",
    "#Name: J/ApJ/787/112/table2",
    "#Title: Line list, including unidentified features",
    "#Column\tFreq\t(GHz)\tRest frequency at the assumed vLSR",
    "#Column\te_Freq\t(MHz)\tFrequency uncertainty",
    "#Column\tSpecies\t()\tSpecies or U for unidentified",
    "#Column\tTA\t(K)\tPeak antenna temperature",
    "#Table\tJ_ApJ_787_112_refs:",
    "#Name: J/ApJ/787/112/refs",
    "#Title: References",
    "#Column\tRef\t()\tReference code",
])


def _uline_asu_rows(n: int = 6) -> str:
    rows = _raw_rows().head(n)
    body = ["#RESOURCE=yCat_J/ApJ/787/112",
            "#Name: J/ApJ/787/112/table2",
            '#Column\t"Freq"\t(GHz)\tRest frequency',
            "#Column\tSpecies\t()\tSpecies or U",
            "#Column\tTA\t(K)\tPeak",
            "#Column\te_Freq\t(MHz)\tError",
            '"Freq"\tSpecies\tTA\te_Freq',
            "GHz\t\tK\tMHz",
            "--------\t--------\t--------\t--------"]
    for _, r in rows.iterrows():
        body.append(f"{r['Freq']:.6f}\t{r['Species']}\t{r['TA']:.4f}\t{r['e_Freq']}")
    return "\n".join(body)


def _uline_asu_web(meta=ULINE_ASU_META, rows=None):
    """VizieR answering over ASU while every TAP host is 503ing."""
    rows = _uline_asu_rows() if rows is None else rows
    seen: list[str] = []

    def fetch(url: str) -> str:
        seen.append(url)
        if "-meta.all" in url:
            if meta is None:
                raise RuntimeError("503 Server Error: Service Unavailable")
            return meta
        if url.endswith("/ReadMe"):
            raise RuntimeError("404 Not Found")
        if rows is None:
            raise RuntimeError("503 Server Error: Service Unavailable")
        return rows

    fetch.urls = seen
    return fetch


def test_uline_discovery_and_fetch_fall_through_tap_503_to_the_asu_route():
    """TAP down, ASU up: real rows, and the record says which route served them.

    This is the run that produced ``NO_DATA_REACHED`` on 2026-09-13 for an
    infrastructure reason --- TAPVizieR 503'd five attempts on each of two
    spellings.  With the ladder, the same dispatch reads the catalogue over
    VizieR's non-TAP ASU interface instead.
    """
    msg = "DALServiceError: 503 Server Error: Service Unavailable for url: .../TAPVizieR/tap/sync"
    tap, web = _FailingTAP(msg), _uline_asu_web()
    log = acq.AcquisitionLog()
    d = acq.discover_line_table("orion_kl_hifi", "J/ApJ/787/112/", query_fn=tap, fetch_fn=web,
                                log=log, fallback_terms_all=["unidentified"],
                                fallback_terms_any=["Orion"])
    assert d.status == "OK" and d.route == "asu_tsv"
    assert d.table == "J/ApJ/787/112/table2"
    assert d.roles["freq"] == "Freq" and d.roles["ident"] == "Species"
    assert d.units["freq"] == "GHz" and "rest" in d.frame_hint
    assert d.fallback == {}                      # a table was found: no diagnostic search needed
    # the TAP failure is still on the record, with its ADQL and its error text
    assert d.errors and msg in d.errors[0]["error"]
    assert any(r["route"] == "asu_tsv" and r["status"] == "OK" for r in d.routes)
    assert any(q.get("route") == "asu_tsv" for q in d.queries)
    assert "-meta.all" in web.urls[0]

    df = acq.fetch_line_table(d, query_fn=tap, fetch_fn=web, log=log)
    assert len(df) == 6 and df.attrs["route"] == "asu_tsv"
    assert df["freq_mhz"].min() > 1000.0         # GHz -> MHz off the non-TAP unit metadata
    assert df["unidentified"].sum() >= 1
    stages = {s["stage"]: s for s in log.as_dict()["stages"]}
    assert stages["fetch_orion_kl_hifi_non_tap"]["route"] == "asu_tsv"
    assert stages["fetch_orion_kl_hifi_non_tap"]["status"] == "OK"
    assert "-out.form=TSV" in web.urls[-1] and "-out=Freq" in web.urls[-1]


def test_uline_route_ladder_reports_every_endpoint_when_every_route_fails():
    """No route, no rows, no invention --- and a diagnosable record."""
    acq.reset_route_state()
    tap, web = _FailingTAP("503 Server Error: Service Unavailable"), _uline_asu_web(meta=None,
                                                                                   rows=None)
    log = acq.AcquisitionLog()
    d = acq.discover_line_table("irc10216_he2008", "J/ApJS/177/275/", query_fn=tap, fetch_fn=web,
                                log=log, fallback_terms_all=["unidentified"],
                                fallback_terms_any=["IRC+10216"])
    assert d.status == "QUERY_FAILED" and d.table is None and d.route == "none"
    assert len(d.errors) == 2                    # the id search and the description fallback
    named = " ".join(f"{r['endpoint']} {r.get('error', '')}" for r in d.routes)
    for endpoint in acq.VIZIER_ASU_MIRRORS:
        assert endpoint in named, endpoint
    assert "ReadMe" in named and "503" in named
    assert not len(acq.fetch_line_table(d, query_fn=tap, fetch_fn=web, log=log))
    assert acq.route_log_summary()["served_by"] == "none"
    acq.reset_route_state()


def test_uline_tap_endpoints_put_the_configured_host_first():
    conf = load_uline_config()
    eps = acq.tap_endpoints(conf["archives"]["vizier_tap"])
    assert eps[0] == conf["archives"]["vizier_tap"]
    assert set(eps) == set(acq.VIZIER_TAP_MIRRORS)
    assert len(eps) == len(set(eps)) >= 2         # a second HOST, not just a second spelling
    assert len({e.split("//")[1].split("/")[0] for e in eps}) >= 2


def test_config_route_ladder_matches_the_module_constants():
    conf = load_uline_config()
    arch = conf["archives"]
    assert tuple(arch["vizier_tap_mirrors"]) == acq.VIZIER_TAP_MIRRORS
    assert tuple(arch["vizier_asu_tsv"]) == acq.VIZIER_ASU_MIRRORS
    assert arch["vizier_tap"] == acq.VIZIER_TAP_MIRRORS[0]
    txt = Path("config/uline.yaml").read_text()
    assert txt.count("# verify:") >= 2 and "ASSERTED" in txt
    assert "asu-tsv" in txt and "-meta.all" in txt


# ---------------------------------------------------------------------------
# The column census: "unidentified" is a COLUMN description, not a table one
# ---------------------------------------------------------------------------
def test_column_census_searches_tap_schema_columns_not_tables():
    """Run 35038662696 got 0 tables for "unidentified" and that is an artefact.

    A VizieR TABLE description is a one-line title ("Spectral survey of Orion
    KL"); the word "unidentified" is in the description of a COLUMN — the flag
    that marks a line unassigned. Searching the wrong table reads as "VizieR
    has no U-line catalogues at all".
    """
    from seti.uline import acquire as A

    adql = A.columns_described_adql(["unidentified"], limit=400)
    assert "FROM TAP_SCHEMA.columns" in adql
    assert "TAP_SCHEMA.tables" not in adql
    # case variants OR-ed: TAPVizieR's LIKE is case-sensitive
    for v in ("%unidentified%", "%Unidentified%", "%UNIDENTIFIED%"):
        assert v in adql
    assert "TOP 400" in adql


def test_column_census_groups_matching_columns_by_table():
    from seti.uline import acquire as A

    def query_fn(adql: str):
        assert "TAP_SCHEMA.columns" in adql
        return pd.DataFrame({
            "table_name": ['"J/A+A/517/A96/table2"', '"J/A+A/517/A96/table2"',
                           '"J/ApJS/259/30/ulines"'],
            "column_name": ["U", "Note", "Uflag"],
            "description": ["Unidentified line flag", "unidentified/blend note",
                            "Unidentified feature"],
        })

    rep = A.column_census(["unidentified"], query_fn=query_fn, source="orion_kl_hifi")
    assert rep["status"] == A.STATUS_OK
    assert rep["n_tables"] == 2 and rep["n_columns"] == 3
    names = [t["table_name"] for t in rep["tables"]]
    assert names == ["J/A+A/517/A96/table2", "J/ApJS/259/30/ulines"]   # quotes stripped
    assert len(rep["tables"][0]["columns"]) == 2


def test_column_census_records_a_failure_rather_than_an_empty_sky():
    from seti.uline import acquire as A

    def query_fn(adql: str):
        raise RuntimeError("503 Service Unavailable")

    rep = A.column_census(["unidentified"], query_fn=query_fn)
    assert rep["status"] == A.STATUS_FAILED and "503" in rep["error"]
    assert rep["n_tables"] == 0
    # and a census with nothing to ask for says so, rather than querying
    assert A.column_census([])["status"] == "NOT_ATTEMPTED"


def test_orion_kl_is_recorded_as_measured_absent_with_its_three_routes():
    """One silent route is not an absence claim; three agreeing routes are."""
    raw = Path("config/uline.yaml").read_text()
    assert "MEASURED ABSENT" in raw
    for evidence in ("HTTP 404", "-meta.all", "0 rows"):
        assert evidence in raw, evidence


def test_the_census_found_source_is_asserted_with_the_same_star_it_already_models():
    """The census turned up a second IRC+10216 U-line list; nothing new is assumed.

    Run 35039345593's table search returned 34 VizieR tables whose description
    is a U-line list and the column census 78 carrying a U-line flag. Only one
    lands on a source this channel already models, so its velocity handling is
    the existing entry's — the survey is new, the star is not.
    """
    import yaml

    conf = yaml.safe_load(Path("config/uline.yaml").read_text())
    new = conf["sources"]["irc10216_cernicharo2000"]
    old = conf["sources"]["irc10216_he2008"]
    assert new["vizier_like"].startswith("J/A+AS/142/181")
    assert new["enabled"] is True
    # same star => same systemic velocity, same expansion-broadened width
    assert new["v_lsr_km_s"] == old["v_lsr_km_s"]
    assert new["fwhm_km_s"] == old["fwhm_km_s"]
    assert new["frequency_frame"] == "rest"
    raw = Path("config/uline.yaml").read_text()
    # the two the census found that are NOT enabled name what blocks them
    assert "J/A+A/681/A50" in raw and "per-row source column" in raw
    assert "J/A+A/564/L2" in raw and "nucleus, not the LSR" in raw


# ---------------------------------------------------------------------------
# Identical atom counts do not identify a molecule (probe run 35039720676)
# ---------------------------------------------------------------------------
def test_isomers_are_never_accepted_as_the_target_species():
    """The three the atom-count route accepted, and what they really are.

    Methyl formate and glycolaldehyde are both C2H4O2 with entirely different
    rotational spectra; methyl cyanide and methyl isocyanide are both C2H3N;
    cyanoacetylene and isocyanoacetylene are both C3HN. Their frequencies were
    being used to VETO unidentified lines under the target's name, so a veto
    could be justified by a molecule that is not the one named and need not
    even be present in the source. Over-vetoing discards exactly what this
    search looks for.
    """
    from seti.uline.lines import same_formula_not_matched, species_match_route

    for cat, target in (("HCOCH2OH", "HCOOCH3"),      # glycolaldehyde vs methyl formate
                        ("CH3NC", "CH3CN"),           # methyl isocyanide vs methyl cyanide
                        ("HCCNC", "HC3N"),            # isocyanoacetylene vs cyanoacetylene
                        ("HNCCC", "HC3N")):
        assert species_match_route(cat, target, []) is None, f"{cat} accepted as {target}"
        assert same_formula_not_matched(cat, target), cat   # same formula, and recorded as such


def test_a_same_formula_entry_is_a_near_miss_that_names_its_reason():
    ents = [Entry(tag=60003, name="CH3OCHO", nlines=1),
            Entry(tag=60006, name="HCOCH2OH", nlines=1)]
    # with no declared pattern, BOTH are near misses — neither is guessed at
    m = match_species(ents, "HCOOCH3", patterns=[])
    assert m.entries == []
    assert all("isomer or unconfigured alias" in n for n in m.near_miss_names)
    assert any("HCOCH2OH" in n for n in m.near_miss_names)
    # the declared spelling is matched, and the isomer still is not
    m2 = match_species(ents, "HCOOCH3", patterns=[r"^CH3OCHO"])
    assert [e.tag for e in m2.entries] == [60003]
    assert [d["route"] for d in m2.matched_names] == ["regex"]
    assert any("HCOCH2OH" in n for n in m2.near_miss_names)


def test_states_and_isotopologues_still_match_through_the_name():
    """The decorated route is a transformation of the NAME, so it cannot
    turn one molecule into another the way an atom-count key can."""
    from seti.uline.lines import species_match_route, undecorate

    assert undecorate("CH3-35Cl, v=0") == "CH3Cl"
    assert undecorate("C-13-H3OH") == "CH3OH"
    assert undecorate("CH2F2-v4") == "CH2F2"
    for cat, target in (("CH3-35Cl, v=0", "CH3Cl"), ("C-13-H3OH", "CH3OH"),
                        ("CH2F2-v4", "CH2F2"), ("CF2Cl2, v=0", "CF2Cl2")):
        assert species_match_route(cat, target, []) == "decorated", cat
    # and it does not undecorate its way into an isomer
    assert undecorate("HCOCH2OH") == "HCOCH2OH"


def test_the_config_says_a_same_formula_alias_must_be_declared():
    raw = Path("config/uline.yaml").read_text()
    assert "MUST BE DECLARED HERE" in raw
    assert "glycolaldehyde" in raw and "isomer" in raw.lower()


# ---------------------------------------------------------------------------
# a `verify` line list may never, on its own, make a candidate
# ---------------------------------------------------------------------------
def _verify_screen(verify: bool) -> dict:
    """A minimal screen report with exactly one PASSING species x source pair."""
    best = {"tex_k": 50.0, "n_features_tested": 40, "n_coincident": 5,
            "n_coincident_vetoed": 0, "spearman_rho": 0.8, "lte_testable": True,
            "lte_pass": True, "top5_fraction_any": 1.0, "top5_fraction_uline": 1.0,
            "p_false": 0.001, "p_false_full": 0.001, "n_trials": 1000, "pattern": True,
            "tests": {"count": True, "lte": True, "top5": True, "p_false": True},
            "coincident_freq_mhz": [1.0e5]}
    return {"sources": {"synth": {"status": "OK", "n_ulines": 40, "has_intensity": True}},
            "species_tables": {"SO2F2": {"group": "targets", "line_source": "rotor",
                                         "databases": [], "n_lines": 400,
                                         "predicted": True, "verify": verify}},
            "results": {"SO2F2|synth": {"species": "SO2F2", "source": "synth",
                                        "line_source": "rotor", "status": "OK",
                                        "predicted": True, "verify": verify,
                                        "records": [{"status": "OK"}], "best": best}},
            "match": {}}


def test_a_pattern_on_verify_constants_is_not_a_pattern_candidate(tmp_path):
    """A predicted frequency is only as good as its constants.

    Every rotor block in ``src/seti/data_assets/rotor_constants.yaml`` is
    ``verify``: reconstructed from the literature, not read off a laboratory
    line list.  A pattern resting on those frequencies alone is a reason to
    obtain the laboratory list; it must never be counted alongside a candidate
    found on a catalogued list.
    """
    conf = synth_conf()
    s = stage_assess(conf, tmp_path, screen=_verify_screen(True), acquire_report={})
    assert s["verdict"] == "PATTERN_CANDIDATE_VERIFY_CONSTANTS"
    assert s["n_pattern_candidates"] == 0
    assert s["n_pattern_candidates_verify_constants"] == 1
    assert s["pattern_candidates_verify_constants"] == ["SO2F2|synth"]
    assert "reconstructed" in s["note"]
    # the same pattern on a laboratory list IS a candidate
    s2 = stage_assess(conf, tmp_path, screen=_verify_screen(False), acquire_report={})
    assert s2["verdict"] == "PATTERN_CANDIDATE" and s2["n_pattern_candidates"] == 1
    assert s2["n_pattern_candidates_verify_constants"] == 0


# ---------------------------------------------------------------------------
# the U-line census asks for U-LINES, not for everything unidentified
# ---------------------------------------------------------------------------
def test_the_uline_census_runs_once_on_its_own_phrases(tmp_path):
    """Run 35039345593's census inherited the SOURCE's description terms
    ("Orion", "IRC+10216", "line survey") and came back with 78 tables of which
    ~70 were Orion star catalogues and Chandra "unidentified sources".  The
    census now carries its own phrase list and runs once, not per source."""
    conf = load_uline_config()
    terms = conf["archives"]["uline_column_census_terms"]
    assert "unidentified line" in terms and not any("rion" in t for t in terms)

    seen: list[str] = []
    base = _FakeTAP("ok", _raw_rows())

    def query_fn(adql: str):
        seen.append(adql)
        if "TAP_SCHEMA.columns" in adql and "unidentified line" in adql:
            return pd.DataFrame({
                "table_name": ['"J/ApJ/787/112/table3"'],
                "column_name": ["Note"],
                "description": ["Unidentified line, peak T(MB)"]})
        return base(adql)

    rep = stage_probe(conf, tmp_path, fetch_fn=_FakeWeb(), query_fn=query_fn,
                      sources=["orion_kl_hifi"])
    census = rep["uline_column_census"]
    assert census["status"] == "OK" and census["n_tables"] == 1
    assert census["tables"][0]["table_name"] == "J/ApJ/787/112/table3"
    # exactly ONE census query, and it asked TAP_SCHEMA.columns
    asked = [a for a in seen if "unidentified line" in a]
    assert len(asked) == 1 and "TAP_SCHEMA.columns" in asked[0]


# ---------------------------------------------------------------------------
# a species whose quartic constants are unknown is not searchable, and says so
# ---------------------------------------------------------------------------
def test_frequency_limited_species_are_rolled_up_into_the_summary(tmp_path):
    """The predicted error of a rotor with unknown quartic constants is tens to
    hundreds of MHz, while an Orion linewidth at 150 GHz is ~2 MHz.  The match
    tolerance is then the prediction's own ignorance, chance alignments rise
    with it, and the FAP gate can never be reached.  That is a property of the
    CONSTANTS and no amount of data repairs it, so the summary names it."""
    conf = synth_conf()
    screen = _verify_screen(True)
    screen["results"]["SO2F2|synth"]["best"]["pattern"] = False
    screen["results"]["SO2F2|synth"]["searchability"] = {
        "status": "FREQUENCY_LIMITED", "median_err_mhz": 146.0, "tolerance_mhz": 2.0,
        "err_over_tolerance": 73.0, "quartic_known": False,
        "remedy": "the quartic centrifugal-distortion constants are unknown"}
    s = stage_assess(conf, tmp_path, screen=screen, acquire_report={})
    assert s["verdict"] == "NO_PATTERN"
    lim = s["targets_frequency_limited"]
    assert set(lim) == {"SO2F2"} and lim["SO2F2"]["err_over_tolerance"] == 73.0
    assert any("FREQUENCY-LIMITED" in d and "quartic" in d for d in s["degraded"])
    assert s["pairs"]["SO2F2|synth"]["searchability"]["status"] == "FREQUENCY_LIMITED"


# ---------------------------------------------------------------------------
# the LTE test lives or dies on the intensity column resolving
# ---------------------------------------------------------------------------
def test_an_all_unidentified_table_needs_no_identification_column():
    """A table whose every row is a U-line has NO identification column,
    because there is nothing to identify.

    Run 35752177872 lost the comet C/2013 R1 (Lovejoy) U-line table to exactly
    that contradiction: `J/A+A/564/L2/table4` resolved `Freq` and `T(MB)dv`
    cleanly, was declared `all_unidentified: true` in the config, and was still
    marked `usable: False` for want of a name column it cannot have.  It cost
    the channel its tightest tolerance — a coma line is ~1.5 km/s wide against
    IRC+10216's 30 — and its only non-stellar environment.
    """
    from seti.uline import acquire as A
    from seti.uline.run import _column_patterns

    cols = _column_patterns(load_uline_config())
    # the comet table's real columns, from probe.json's scoreboard
    comet = ["recno", "Freq", "T(MB)dv", "e_T(MB)dv", "Dv", "e_Dv", "SNR", "Note"]
    roles = A.resolve_line_columns(comet, cols)
    assert roles["freq"] == "Freq" and roles["intensity"] == "T(MB)dv"
    assert not roles["ident"]          # there is none, and there cannot be

    def query_fn(adql: str):
        if "TAP_SCHEMA.tables" in adql:
            return pd.DataFrame({"table_name": ['"J/A+A/564/L2/table4"'],
                                 "description": ["Unidentified lines in comet Lovejoy"]})
        if "TAP_SCHEMA.columns" in adql:
            return pd.DataFrame({"column_name": comet,
                                 "description": ["" for _ in comet],
                                 "unit": ["" for _ in comet]})
        return pd.DataFrame({"Freq": [251766.0, 264753.0], "T(MB)dv": [0.1, 0.2]})

    strict = A.discover_line_table("comet", "J/A+A/564/L2/", query_fn=query_fn,
                                   column_patterns=cols, allow_non_tap=False)
    assert strict.table is None        # the old behaviour: dropped

    loose = A.discover_line_table("comet", "J/A+A/564/L2/", query_fn=query_fn,
                                  column_patterns=cols, allow_non_tap=False,
                                  ident_required=False)
    assert loose.table == "J/A+A/564/L2/table4"
    assert loose.roles["freq"] == "Freq" and loose.roles["intensity"] == "T(MB)dv"


def test_two_views_of_one_table_are_not_two_U_line_samples(tmp_path):
    """Run 35752177872 read J/A+AS/142/181 table2 — whose `Mol` column flags 63
    lines unidentified — AND table3, which is those same 63 as a standalone
    U-line table.  Summing per source reported 143 U-lines where there are 80.

    The sample size is the one number a reader quotes, so it is counted on the
    frequencies themselves and the overlap is named rather than hidden.
    """
    shared = [float(100000 + 7 * i) for i in range(63)]
    other = [float(150000 + 11 * i) for i in range(17)]
    conf = synth_conf()
    screen = {
        "sources": {
            "cernicharo_table2": {"status": "OK", "n_ulines": 63, "has_intensity": True,
                                  "uline_freqs_mhz": list(shared)},
            "cernicharo_table3": {"status": "OK", "n_ulines": 63, "has_intensity": True,
                                  "uline_freqs_mhz": list(shared)},
            "he2008": {"status": "OK", "n_ulines": 17, "has_intensity": True,
                       "uline_freqs_mhz": list(other)},
        },
        "species_tables": {}, "results": {}, "match": {}}
    s = stage_assess(conf, tmp_path, screen=screen, acquire_report={})
    assert s["n_ulines_per_source_sum"] == 143      # what the naive sum said
    assert s["n_ulines_total"] == 80                # what is actually there
    ov = s["uline_source_overlaps"]
    assert len(ov) == 1
    assert sorted(ov[0]["sources"]) == ["cernicharo_table2", "cernicharo_table3"]
    assert ov[0]["n_shared"] == 63 and ov[0]["fraction_of_smaller"] == 1.0


def test_a_diverged_refit_is_never_published_as_a_hamiltonian_floor():
    """Run 35752177872 fitted every parseable SO2 line — including the excited
    states a ground-state Hamiltonian does not model — and reported a 4 GHz
    post-fit rms beside a MEASURED 2.7 MHz median residual on the lines that
    did match.  Quoting that as "the Hamiltonian's floor" says the predictor is
    worthless when it is good to a few MHz."""
    from seti.uline.run import partition_refits

    good, bad = partition_refits([
        # SO2 exactly as run 35752177872 had it: the refit IMPROVED (15 GHz ->
        # 4 GHz) yet landed three orders of magnitude above the 2.7 MHz the
        # unfitted constants already reach line by line.  Improvement on a
        # polluted starting point is not a floor.
        {"comparison": {"residual_mhz": {"median_abs": 2.7}},
         "refit": {"rms_before_mhz": 15126.0, "rms_after_mhz": 4015.0, "n_fit_lines": 4317}},
        # a genuine floor: comparable to the measured residual
        {"comparison": {"residual_mhz": {"median_abs": 3.0}},
         "refit": {"rms_before_mhz": 90.0, "rms_after_mhz": 0.9}},
        {"comparison": {}, "refit": {"rms_after_mhz": 5.0}},     # no reference: taken as given
        {"comparison": {"residual_mhz": {"median_abs": 2.0}},
         "refit": {"error": "scipy missing"}},                   # not a fit at all
    ])
    assert good == [0.9, 5.0]
    assert len(bad) == 1
    assert bad[0]["rms_after_mhz"] == 4015.0
    assert bad[0]["measured_median_abs_mhz"] == 2.7
    assert "not this Hamiltonian's floor" in bad[0]["why"]
    assert partition_refits([]) == ([], [])


def test_the_probe_clock_stops_a_slow_source_and_names_the_ones_it_skipped(tmp_path):
    """A budget checked only BETWEEN sources decides whether to *start* the next
    one and can never stop the one already running.

    One slow catalogue runs its whole route ladder — four TAP hosts, each
    retried, then ASU, then astroquery — plus a column query and a row count
    for every table it lists.  ULINE walks ten sources, so against a loaded
    VizieR that is how a job reaches its `timeout-minutes`, and a job killed
    that way skips the `if: always()` commit-back and leaves NO results.
    """
    import time as _t

    conf = load_uline_config()
    conf["archives"]["probe_budget_s"] = 0.6
    calls = {"n": 0}

    def slow_tap(adql: str):
        calls["n"] += 1
        _t.sleep(0.25)
        return pd.DataFrame()

    names = ["orion_kl_hifi", "irc10216_he2008", "sgrb2_nummelin1998"]
    t0 = _t.monotonic()
    rep = stage_probe(conf, tmp_path, fetch_fn=_FakeWeb(), query_fn=slow_tap, sources=names)
    elapsed = _t.monotonic() - t0

    # it stopped rather than running the full ladder for every source
    assert elapsed < 20.0
    statuses = {k: v.get("status") for k, v in rep["vizier"].items()}
    assert set(statuses) == set(names)
    stopped = [k for k, s in statuses.items()
               if s in ("DISCOVERY_TIMED_OUT", "DISCOVERY_NOT_ATTEMPTED")]
    assert stopped, f"no source was stopped by the clock: {statuses}"
    for k in stopped:
        assert rep["vizier"][k]["error"]            # the reason, always
    # a clock is not a sky result
    assert all(s != "OK" or rep["vizier"][k].get("table") for k, s in statuses.items())


def test_a_species_whose_hamiltonian_is_wrong_says_so_in_the_summary(tmp_path):
    """SO2F2 is accidentally near-spherical (A ~ B ~ C), and Sarka, Demaison,
    Margules et al. found Watson's A-REDUCTION FAILS for it — an unreduced
    Hamiltonian with all six quartic constants was needed.  This predictor is
    A-reduced, so for SO2F2 alone the published quartic set would not make the
    prediction right.  A model limit is not a constants limit and must not be
    reported as one."""
    from seti.uline.rotorpred import load_rotor_assets

    cav = load_rotor_assets()["species"]["SO2F2"]["hamiltonian_caveat"]
    assert "A-reduction fails" in cav["reduction"]
    assert "model-limited" in cav["consequence"]

    conf = synth_conf()
    screen = _verify_screen(True)
    screen["results"]["SO2F2|synth"]["best"]["pattern"] = False
    screen["species_tables"]["SO2F2"]["rotor"] = {"hamiltonian_caveat": cav}
    s = stage_assess(conf, tmp_path, screen=screen, acquire_report={})
    assert s["targets_hamiltonian_caveats"]["SO2F2"]["reduction"] == cav["reduction"]
    assert any("HAMILTONIAN CAVEAT SO2F2" in d for d in s["degraded"])


def test_the_real_column_sets_of_the_acquired_surveys_resolve_an_intensity():
    """Runs 35039822190 and 35041128720 reported `lte_testable: false` for every
    pair, and the channel recorded "no intensity column" as a property of the
    SURVEYS.  It was a column-matching failure: probe.json shows Cernicharo+2000
    tables 2 and 3 carrying `T(MB)dv` and He+2008 table 4 carrying `Iint`, both
    integrated line intensities — exactly the quantity the Spearman test wants,
    since it is what the column is proportional to in the optically thin LTE
    limit.  These are the archive's OWN column names, read off the probe's
    scoreboard, pinned so the regex list cannot silently lose them again.
    """
    from seti.uline.acquire import resolve_line_columns
    from seti.uline.run import _column_patterns

    cols = _column_patterns(load_uline_config())
    cases = {
        "J/A+AS/142/181/table2": (
            ["Mol", "Trans", "n_Freq", "Freq", "e_Freq", "nFreq", "Freqc", "u_Freqc",
             "T(MB)dv", "e_T(MB)dv", "Vexp", "e_Vexp", "Notes"],
            {"freq": "Freq", "ident": "Mol", "intensity": "T(MB)dv"}),
        "J/A+AS/142/181/table3": (
            ["Name", "Trans", "Freq", "e_Freq", "Freqc", "u_Freqc", "T(MB)dv", "e_T(MB)dv",
             "Vexp", "e_Vexp", "Notes"],
            {"freq": "Freq", "ident": "Name", "intensity": "T(MB)dv"}),
        "J/ApJS/177/275/table4": (
            ["Species", "Trans", "Freq", "e_Freq", "Iint", "e_Iint", "Vexp", "Notes"],
            {"freq": "Freq", "ident": "Species", "intensity": "Iint"}),
    }
    for table, (columns, want) in cases.items():
        got = resolve_line_columns(columns, cols)
        for role, col in want.items():
            assert got[role] == col, f"{table}: {role} resolved to {got[role]!r}, want {col!r}"
        # and the ERROR column never masquerades as the intensity
        assert not str(got["intensity"]).startswith("e_"), table
