"""Offline tests for CONFLUENCE (no network)."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from seti.confluence import core, harvest, registry, stats, tagger
from seti.confluence.registry import ChannelSpec

F = frozenset


# ---------------------------------------------------------------------------
# the exact null
# ---------------------------------------------------------------------------


def _brute_pair(n, a, b):
    """P(|A∩B|=k) by enumerating every B for a fixed A."""
    A = set(range(a))
    counts = np.zeros(min(a, b) + 1)
    for B in itertools.combinations(range(n), b):
        counts[len(A & set(B))] += 1
    return counts / counts.sum()


def test_pair_pmf_matches_enumeration():
    for n, a, b in [(7, 3, 4), (6, 2, 2), (8, 5, 6)]:
        assert np.allclose(stats.pair_cell_pmf(n, a, b), _brute_pair(n, a, b))


def test_triple_pmf_matches_enumeration():
    n, a, b, c = 6, 3, 3, 2
    A = set(range(a))
    counts = np.zeros(min(a, b, c) + 1)
    for B in itertools.combinations(range(n), b):
        for C in itertools.combinations(range(n), c):
            counts[len(A & set(B) & set(C))] += 1
    assert np.allclose(stats.triple_cell_pmf(n, a, b, c), counts / counts.sum())


def test_stratified_null_is_the_convolution_of_cells():
    # two cells; analytic expectation = sum a_c b_c / n_c
    idx = [f"s{i}" for i in range(20)]
    t = pd.DataFrame({"A": [True] * 3 + [False] * 7 + [True] * 5 + [False] * 5,
                      "B": [True, False] * 5 + [True] * 2 + [False] * 8}, index=idx)
    cells = pd.Series(["x"] * 10 + ["y"] * 10, index=idx)
    r = stats.stratified_overlap(t, cells, ("A", "B"))
    assert r.expected == pytest.approx(3 * 5 / 10 + 5 * 2 / 10)
    assert r.observed == int((t.A & t.B).sum())
    assert 0 <= r.p_value <= 1


def test_top_q_tail_counts_and_nan():
    s = pd.Series([np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0])
    t = stats.top_q_tail(s, 0.2)
    assert t.sum() == 2 and not t.iloc[0]
    assert stats.top_q_tail(pd.Series([np.nan, np.nan]), 0.1).sum() == 0


def test_bh_monotone():
    q = stats.benjamini_hochberg([0.01, 0.04, 0.03, 0.5, np.nan])
    assert np.isnan(q[-1]) and q[0] <= q[2] <= q[1] <= q[3]


# ---------------------------------------------------------------------------
# synthetic channels
# ---------------------------------------------------------------------------


def _specs():
    return {
        "A": ChannelSpec("A", "fa", F({"ztf"}), F(), "a", "p"),
        "B": ChannelSpec("B", "fb", F({"wise"}), F(), "b", "p"),
        "C": ChannelSpec("C", "fc", F({"apogee"}), F(), "c", "p"),
        "D": ChannelSpec("D", "fd", F({"wise"}), F({"2mass"}), "d", "p"),
    }


def _long(scores: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for ch, s in scores.items():
        rows.append(pd.DataFrame({"channel": ch, "star": s.index, "score": s.to_numpy(),
                                  "flag": False, "source_id": pd.NA, "ra": np.nan,
                                  "dec": np.nan}))
    return core.add_tails(pd.concat(rows, ignore_index=True))


def _cov(stars, g):
    return pd.DataFrame({"star": stars, "phot_g_mean_mag": g})


def test_injected_confluence_is_recovered():
    rng = np.random.default_rng(1)
    n = 4000
    stars = [f"gaia:{i}" for i in range(n)]
    g = rng.uniform(10, 18, n)
    L = _long({"A": pd.Series(rng.normal(size=n), index=stars),
               "B": pd.Series(rng.normal(size=n), index=stars)})
    out = core.injection_trials(L, _cov(stars, g), ("A", "B"), k_values=(10, 20),
                                q=0.05, alert_q=0.005, n_trials=15,
                                covariates=["phot_g_mean_mag"])
    by = {r["k"]: r for r in out["by_k"]}
    assert by[20]["recovery_rate"] >= 0.8
    assert by[20]["mean_frac_planted_in_members"] > 0.95
    assert by[20]["planted_in_alert_tail"] == 0     # every plant was sub-threshold


def test_shared_covariate_confounder_is_nulled_by_stratification():
    """Faint stars score high in BOTH channels (a shared S/N systematic).  The raw
    overlap is hugely significant; the covariate-matched one is not."""
    rng = np.random.default_rng(2)
    n = 6000
    stars = [f"gaia:{i}" for i in range(n)]
    g = rng.uniform(8, 19, n)
    bump = (g > 17.5) * 3.0
    L = _long({"A": pd.Series(rng.normal(size=n) + bump, index=stars),
               "B": pd.Series(rng.normal(size=n) + bump, index=stars)})
    cov = _cov(stars, g)
    raw = core.run_tests(L, cov, _specs(), modes=("0.05",), covariates=[], triples=False)
    strat = core.run_tests(L, cov, _specs(), modes=("0.05",),
                           covariates=["phot_g_mean_mag"], triples=False)
    assert raw.iloc[0]["p_value"] < 1e-6
    assert strat.iloc[0]["p_value"] > 0.01


def test_null_is_calibrated_under_independence():
    """Independent channels with a covariate that shifts BOTH tails' rates: the
    propensity-stratified p-value must not over-reject."""
    rng = np.random.default_rng(4)
    n, trials, hits = 2500, 60, 0
    stars = [f"gaia:{i}" for i in range(n)]
    for _ in range(trials):
        g = rng.uniform(8, 19, n)
        # rate depends on G in both channels, but draws are independent given G
        L = _long({"A": pd.Series(rng.normal(size=n) + 0.3 * g, index=stars),
                   "B": pd.Series(rng.normal(size=n) + 0.3 * g, index=stars)})
        r = core.run_tests(L, _cov(stars, g), _specs(), modes=("0.05",),
                           covariates=["phot_g_mean_mag"], triples=False)
        hits += r.iloc[0]["p_value"] < 0.05
    assert hits / trials <= 0.12


def test_same_instrument_pairs_are_never_tested():
    specs = _specs()
    assert not registry.independent(specs["B"], specs["D"])            # shared WISE
    assert registry.independent(specs["A"], specs["B"])
    same_fam = ChannelSpec("A2", "fa", F({"euclid"}), F(), "x", "p")
    assert not registry.independent(specs["A"], same_fam)               # same channel
    sup = ChannelSpec("E", "fe", F({"akari"}), F({"2mass"}), "x", "p")
    assert not registry.independent(specs["D"], sup, "strict")          # shared support
    assert registry.independent(specs["D"], sup, "relaxed")
    pairs = core.channel_pairs(specs, ["A", "B", "C", "D"])
    assert ("B", "D") not in pairs and ("A", "B") in pairs


def test_triples_need_every_pair_independent():
    rng = np.random.default_rng(3)
    n = 500
    stars = [f"gaia:{i}" for i in range(n)]
    L = _long({c: pd.Series(rng.normal(size=n), index=stars) for c in "ABCD"})
    res = core.run_tests(L, _cov(stars, rng.uniform(10, 16, n)), _specs(),
                         modes=("0.05",), covariates=[])
    groups = set(res["channels"])
    assert "A+B+C" in groups and "A+B+D" not in groups


def test_joint_too_small_is_reported_not_tested():
    L = _long({"A": pd.Series([1.0, 2, 3], index=["x", "y", "z"]),
               "B": pd.Series([1.0, 2, 3], index=["x", "y", "w"])})
    res = core.run_tests(L, _cov(["x", "y", "z", "w"], [10] * 4), _specs(), triples=False)
    assert (res["status"] == "JOINT_TOO_SMALL").all()


# ---------------------------------------------------------------------------
# identity, tags, trace
# ---------------------------------------------------------------------------


def test_unify_matches_positions_to_gaia_and_to_each_other():
    fa = pd.DataFrame({"key": ["gaia:11", "gaia:12"], "source_id": pd.array([11, 12], "Int64"),
                       "ra": [10.0, 20.0], "dec": [5.0, -5.0], "score": [1.0, 2.0],
                       "flag": [False, False]})
    fb = pd.DataFrame({"key": ["ztf:1", "ztf:2"], "source_id": pd.array([pd.NA, pd.NA], "Int64"),
                       "ra": [10.0 + 0.5 / 3600, 50.0], "dec": [5.0, 1.0],
                       "score": [3.0, 4.0], "flag": [False, False]})
    fc = pd.DataFrame({"key": ["kic:9"], "source_id": pd.array([pd.NA], "Int64"),
                       "ra": [50.0 + 0.3 / 3600], "dec": [1.0], "score": [5.0], "flag": [False]})
    L, rep = core.unify({"A": fa, "B": fb, "C": fc})
    star_b1 = L[(L.channel == "B") & (L.score == 3.0)]["star"].iloc[0]
    assert star_b1 == "gaia:11"
    sb2 = L[(L.channel == "B") & (L.score == 4.0)]["star"].iloc[0]
    sc = L[L.channel == "C"]["star"].iloc[0]
    assert sb2 == sc and sb2.startswith("pos:")


def test_tagger_families():
    assert "YSO" in tagger.families_for(simbad_otype="TT*")
    assert "EB" in tagger.families_for(vsx_type="EA")
    assert "EB" in tagger.families_for(vsx_type="EA+ROT") and \
        "ACTIVE_ROT" in tagger.families_for(vsx_type="EA+ROT")
    assert "LPV_AGB" in tagger.families_for(gaia_class="LPV")
    assert "BE_EM" in tagger.families_for(simbad_otype="Be*")
    assert "CV_WDBIN" in tagger.families_for(simbad_otype="CV*")
    assert "BINARY" in tagger.families_for(non_single_star=1)
    assert tagger.families_for(simbad_otype="PM*") == set()     # high PM is not a class
    assert tagger.families_for(simbad_otype="*") == set()
    assert tagger.families_for(vsx_type="ZZZ") == {"OTHER_VAR"}
    t = tagger.tag_table(pd.DataFrame({"star": ["a", "b"], "otype": ["Y*O", "*"]}))
    assert t["known_class"].tolist() == [True, False]


def test_trace_verdicts():
    L = _long({"A": pd.Series([5.0, 4.0, 3.0], index=["gaia:1", "gaia:2", "gaia:3"]),
               "B": pd.Series([5.0, 4.0, 3.0], index=["gaia:1", "gaia:2", "gaia:3"])})
    cov = pd.DataFrame({"star": ["gaia:1", "gaia:2", "gaia:3"], "ruwe": [2.5, 1.0, 1.0],
                        "phot_g_mean_mag": [12, 12, 12]})
    tags = tagger.tag_table(pd.DataFrame({"star": ["gaia:1", "gaia:2", "gaia:3"],
                                          "otype": ["*", "EB*", "*"]}))
    tr = core.trace_members(L, cov, tags, ["gaia:1", "gaia:2", "gaia:3"]).set_index("star")
    assert tr.loc["gaia:1", "trace_verdict"].startswith("SYSTEMATIC:RUWE")
    assert tr.loc["gaia:2", "trace_verdict"] == "KNOWN_CLASS:EB"
    assert tr.loc["gaia:3", "trace_verdict"] == "UNEXPLAINED"


# ---------------------------------------------------------------------------
# harvest extractors on synthetic artifact layouts
# ---------------------------------------------------------------------------


def test_harvest_extractors(tmp_path):
    root = tmp_path / "h"
    (root / "cenotaph" / "1" / "cenotaph-analysis").mkdir(parents=True)
    pd.DataFrame({"source_id": [1, 2, 3], "grey_sigma": [4.0, 1.0, 5.0],
                  "grey_verdict": ["ok", "ok", "ok"],
                  "midir_excess": [False, False, True]}).to_parquet(
        root / "cenotaph" / "1" / "cenotaph-analysis" / "greyfit.parquet")
    (root / "ignition" / "2" / "ignition-shard-0").mkdir(parents=True)
    pd.DataFrame({"source_id": [5, 6], "w1_slope_sigma": [3.0, 9.0],
                  "w2_slope_sigma": [4.0, 1.0], "star_is_candidate": [True, False],
                  "ra": [1.0, 2.0], "dec": [0.0, 0.0]}).to_csv(
        root / "ignition" / "2" / "ignition-shard-0" / "stars_s0of12.csv", index=False)
    (root / "ring_wd" / "3" / "ring-solo" / "wd").mkdir(parents=True)
    pd.DataFrame({"source_id": [7], "chi_W1": [9.0], "chi_W2": [9.0]}).to_csv(
        root / "ring_wd" / "3" / "ring-solo" / "wd" / "excess.csv", index=False)
    (root / "ring_wd" / "3" / "ring-solo" / "wd" / "screen.json").write_text(
        json.dumps({"n_input": 25932}))
    (root / "mystery" / "4").mkdir(parents=True)
    rep = harvest.harvest(root, tmp_path / "scores")
    ch = rep["channels"]
    assert ch["cenotaph"]["status"] == "OK" and ch["cenotaph"]["n_flag"] == 1   # 3 vetoed
    assert ch["ignition"]["status"] == "OK"
    ig = pd.read_parquet(tmp_path / "scores" / "ignition.parquet").set_index("source_id")
    assert ig.loc[6, "score"] == 1.0          # one-band rise scores as its weaker band
    assert ch["ring_wd"]["status"] == "TAIL_ONLY"   # 1 of 25,932: a tail, not a parent
    assert ch["mystery"]["status"] == "NO_EXTRACTOR"
    assert not (tmp_path / "scores" / "ring_wd.parquet").exists()


def test_harvest_schema_not_recognised(tmp_path):
    d = tmp_path / "h" / "cradle" / "1" / "cradle-screen"
    d.mkdir(parents=True)
    pd.DataFrame({"foo": [1]}).to_csv(d / "parent_screened.csv", index=False)
    rep = harvest.harvest(tmp_path / "h", tmp_path / "s")
    assert rep["channels"]["cradle"]["status"] == "SCHEMA_NOT_RECOGNISED"


# ---------------------------------------------------------------------------
# degradation
# ---------------------------------------------------------------------------


def test_empty_archive_degrades_honestly(tmp_path, monkeypatch):
    from seti.confluence import run as crun
    monkeypatch.setattr(crun, "load_frames", lambda prefer_harvest=True: {})
    out = tmp_path / "conf"
    meta = crun.stage_joint(out)
    assert meta["verdict"] == "NO_DATA_REACHED"
    rep = harvest.harvest(tmp_path / "nothing_downloaded", tmp_path / "s")
    assert rep["channels"] == {}


def test_assess_end_to_end_on_synthetic(tmp_path, monkeypatch):
    """Two independent channels with a planted class that lights both up; the class
    is tagged, so it must lead the overlaps and vanish from the residual."""
    from seti.confluence import run as crun
    rng = np.random.default_rng(7)
    n = 3000
    sid = np.arange(1, n + 1)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    yso = sid[:40]
    a[:40] += 4
    b[:40] += 4
    fr = {}
    for name, s in (("dimming_dip", a), ("tailings", b)):
        fr[name] = pd.DataFrame({"key": [f"gaia:{i}" for i in sid],
                                 "source_id": pd.array(sid, "Int64"),
                                 "ra": rng.uniform(0, 360, n), "dec": rng.uniform(-60, 60, n),
                                 "score": s, "flag": False})
    monkeypatch.setattr(crun, "load_frames", lambda prefer_harvest=True: fr)
    out = tmp_path / "conf"
    crun.stage_joint(out)
    J = pd.read_parquet(out / "joint_long.parquet")
    stars = J["star"].unique()
    tags = pd.DataFrame({"star": stars, "tag_attempted": True, "in_baseline_sample": True,
                         "otype": ["TT*" if int(s.split(":")[1]) in set(yso) else "*"
                                   for s in stars]})
    tags.to_parquet(out / "tags.parquet")
    s = crun.stage_assess(out, n_inject=10)
    assert s["control_known_classes"]["status"] == "PASS"
    assert s["control_injection"] == "PASS"
    assert s["verdict"] == "CONFLUENCE_KNOWN_CLASSES_ONLY"
    tr = pd.read_csv(out / "members.csv")
    assert (tr["trace_verdict"] == "KNOWN_CLASS:YSO").sum() >= 30
