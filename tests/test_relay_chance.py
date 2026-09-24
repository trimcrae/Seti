"""RELAY: the chance model, hit hygiene, and the nearest-neighbour target list.

Run 35745111146 reported 22 "pair-line drift matches".  They were 11 hits, 3
of them read from a "Frequency rank" column, and the count sat at the chance
expectation because the drift window is set by pipeline resolution and the
Earth term, not by anything Gaia measures.  These tests pin the machinery that
says so, and the stage that asks a question which is not empty by
construction.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from seti.relay import chance as ch
from seti.relay import geometry as geo
from seti.relay import papers as pap
from seti.relay import targetlist as tl
from seti.relay.run import load_relay_config

DEG = np.pi / 180


@pytest.fixture(scope="module")
def conf():
    return load_relay_config()


# ---------------------------------------------------------------------------
# chance arithmetic
# ---------------------------------------------------------------------------
def test_poisson_binomial_tail_equals_brute_force():
    p = [0.1, 0.5, 0.8, 0.3]
    for k in range(6):
        brute = 0.0
        for bits in itertools.product([0, 1], repeat=4):
            if sum(bits) >= k:
                brute += np.prod([q if b else 1 - q for q, b in zip(p, bits, strict=True)])
        assert np.isclose(ch.poisson_binomial_tail(p, k), brute)


def test_drift_null_at_chance_when_every_hit_has_a_small_drift():
    # 10 hits all within +/-0.05 Hz/s, windows +/-0.18: every other hit falls inside
    d = np.linspace(-0.05, 0.05, 10)
    idx = np.arange(10)
    r = ch.drift_null(d, idx, idx[:4], d[:4], np.zeros(4), np.full(4, 0.06), 3.0)
    assert r["n_drift_match"] == 4 and np.isclose(r["n_expected_drift_match"], 4.0)
    assert r["p_value_drift_match"] > 0.99


def test_drift_null_is_significant_when_the_pool_is_spread():
    rng = np.random.default_rng(1)
    pool = rng.uniform(-4, 4, 200)
    d = np.concatenate([[0.0, 0.01, -0.01], pool])
    idx = np.arange(len(d))
    r = ch.drift_null(d, idx, idx[:3], d[:3], np.zeros(3), np.full(3, 0.02), 3.0)
    assert r["n_drift_match"] == 3
    assert r["n_expected_drift_match"] < 0.2 and r["p_value_drift_match"] < 1e-3


# ---------------------------------------------------------------------------
# hygiene and RFI rules
# ---------------------------------------------------------------------------
def _hits():
    return pd.DataFrame({
        "target": ["HIP 1", "HIP 2", "HIP 2", "HIP 3", "HIP 4", "HIP 5", "HIP 6"],
        "freq_mhz": [989.0, 1420.2, 1420.2, 1380.91, 1380.93, 1381.10, 1450.0],
        "drift_hz_s": [0.1, 0.2, 0.2, -0.1, -0.2, -0.3, 0.5],
        "table": ["arXiv:2505.03927:ms.tex#table", "arXiv:1901.04057:a.tex#t", "arXiv:1901.04057:a.tex#t",
                  "arXiv:1709.03491:b.tex#t", "arXiv:1709.03491:b.tex#t", "arXiv:1709.03491:b.tex#t",
                  "arXiv:2011.05265:c.tex#t"],
        "freq_header": ["Frequency rank", "Frequency (Hz)", "Frequency (Hz)", "Frequency", "Frequency",
                        "Frequency", "FREQ (Hz)"],
        "caption": ["Promising candidates", "Top", "Top", "Events", "Events", "Events",
                    "Properties of artificial signals used for the signal injection"],
    })


def test_hygiene_flags_rank_columns_injections_and_duplicates():
    h = ch.hit_hygiene(_hits())
    assert h["artefact_not_frequency"].tolist() == [True] + [False] * 6
    assert h["artefact_injection_table"].tolist() == [False] * 6 + [True]
    assert h["duplicate_row"].tolist() == [False, False, True, False, False, False, False]
    assert h["valid_hit"].sum() == 4


def test_known_bands_and_wide_recurrence(conf):
    bands = conf["assess"]["known_rfi_bands_mhz"]
    got = ch.known_rfi_band([1381.0, 1528.46, 1621.2, 1420.4, 1575.42], bands)
    assert got[0].startswith("GPS L3") and got[1].startswith("Inmarsat") and got[2].startswith("Iridium")
    assert got[3] is None and got[4].startswith("GPS L1")
    h = ch.hit_hygiene(_hits())
    w = ch.wide_recurrence(h, 500.0, 3, valid=h["valid_hit"])
    # three unrelated sightlines within 190 kHz: all three flagged, the 1 kHz rule would miss them
    assert w.tolist() == [False, False, False, True, True, True, False]


def test_assess_chance_reproduces_the_run_shape():
    h = ch.hit_hygiene(_hits())
    b = "beamx"
    h[f"{b}:on_pair_line"] = True
    h[f"{b}:drift_centre_hz_s"] = 0.0
    h[f"{b}:halfwidth_tight_hz_s"] = 0.2
    h["rfi_flag"] = False
    r = ch.assess_chance(h, [b], k_sig=3.0)[b]
    assert r["n_trials"] == 4                       # hygiene removed three rows
    assert r["n_drift_match"] == 4                  # every drift inside +/-0.6
    assert np.isclose(r["n_expected_drift_match"], 4.0)


# ---------------------------------------------------------------------------
# the parser no longer makes those rows
# ---------------------------------------------------------------------------
def test_frequency_rank_is_not_a_frequency_column():
    roles = pap.header_roles(["ID", "Target", "Drift rate [Hz/s]", "SNR", "Frequency rank", "Similarity rank"])
    assert "drift" in roles and "freq_mhz" not in roles
    roles = pap.header_roles(["Source", "Frequency (MHz)", "Drift Rate", "S/N"])
    assert roles["freq_mhz"] == 1


def test_injection_table_is_refused():
    t = pap.ParsedTable(source="c.tex", env="table",
                        caption="Properties of artificial signals used for the signal injection and recovery",
                        header=["NAME", "FREQ (Hz)", "DFDT (Hz/s)", "SNR"],
                        rows=[["HIP 1", "1400000000", "0.1", "20"]])
    df, rep = pap.table_hits(t, {"seed": "x", "arxiv_id": "2011.05265"})
    assert df.empty and rep["kind"] == "injection"


# ---------------------------------------------------------------------------
# the target list
# ---------------------------------------------------------------------------
def _tl_sample(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    d = np.clip(100.0 * rng.random(n) ** (1 / 3), 10.0, 100.0)
    df = pd.DataFrame({"source_id": [str(i) for i in range(n)],
                       "ra": np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 360,
                       "dec": np.degrees(np.arcsin(u[:, 2])), "parallax": 1000.0 / d,
                       "parallax_error": 0.02, "pmra": rng.normal(0, 60, n), "pmdec": rng.normal(0, 60, n)})
    # (A) a resolved short link pointing at Earth: T at 50 pc, R at 48 pc, 2 arcsec apart
    #     -> alpha = 48 * 2" / 2 pc ~ 48", inside every radio beam
    # (B) the same geometry but parallaxes indistinguishable (sigma 1 mas): unresolved
    # (C) a comoving pair (same proper motion) 0.1 pc apart on the sky, apparent depth 2 pc
    def pair(sid, ra, dec, dt, dr, sep_arcsec, e, pm):
        return [dict(source_id=sid + "T", ra=ra, dec=dec, parallax=1000 / dt, parallax_error=e,
                     pmra=pm[0], pmdec=pm[1]),
                dict(source_id=sid + "R", ra=ra + sep_arcsec / 3600 / np.cos(np.radians(dec)), dec=dec,
                     parallax=1000 / dr, parallax_error=e, pmra=pm[2], pmdec=pm[3])]
    # (A) is 10" apart at a 5 pc depth (alpha ~ 45 * 10" / 5 pc ~ 90"): wider than
    # the close-pair astrometry cut, still inside every radio beam
    rows = (pair("A", 30.0, 10.0, 50.0, 45.0, 10.0, 0.02, (40, -10, -25, 60))
            + pair("B", 120.0, -30.0, 50.0, 48.0, 2.0, 1.0, (40, -10, -25, 60))
            + pair("C", 250.0, 40.0, 40.0, 38.0, 500.0, 0.02, (80, 20, 80.3, 20.1)))
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def test_p_iso_matches_the_cap_fraction():
    th = 8.85 / 60 * DEG
    assert np.isclose(tl.p_iso(th), (th / 4) ** 2, rtol=1e-3)


def test_neighbour_links_drop_self_and_binaries():
    xyz = np.array([[10.0, 0, 0], [10.01, 0, 0], [12.0, 0, 0], [30.0, 0, 0]])
    t, r, d, k = tl.neighbour_links(xyz, 2, 10.0, 0.05)
    pairs = set(zip(t.tolist(), r.tolist(), strict=True))
    assert (0, 1) not in pairs and (1, 0) not in pairs        # 0.01 pc: a binary
    assert (0, 2) in pairs and (2, 0) in pairs
    assert not any(3 in p for p in pairs)                     # 18 pc away: beyond link_max
    assert all(a != b for a, b in pairs)


def test_targetlist_ranks_the_resolved_link_and_excludes_the_confounders(conf):
    s = _tl_sample()
    beams = geo.beam_grid(conf["beams"])
    tconf = dict(conf["targetlist"], n_mc=200)
    ia = int(np.flatnonzero(s["source_id"] == "AT")[0])
    lt, rep = tl.build_targetlist(s, beams, tconf, bl_idx=[ia], bl_names_unresolved=7)
    a = lt[(lt["t_source_id"] == "AT") & (lt["r_source_id"] == "AR")].iloc[0]
    assert a["rankable"] and a["p_in_beam:radio_100m_1p4ghz"] > 0.95
    assert a["narrowest_beam_p50"] == "radio_100m_1p4ghz" and a["t_bl_observed"]
    assert a["geometry"] == geo.GEOM_SPILLOVER and a["flux_ratio_earth_over_receiver"] < 0.02
    # the reverse link (R -> T) points away from Earth
    assert lt[(lt["t_source_id"] == "AR") & (lt["r_source_id"] == "AT")].empty or \
        lt[(lt["t_source_id"] == "AR") & (lt["r_source_id"] == "AT")]["p_in_beam:overfilled_5deg"].max() < 0.05
    b = lt[lt["t_source_id"] == "BT"]
    assert len(b) and not b["radial_separation_resolved"].any() and not b["rankable"].any()
    c = lt[lt["t_source_id"] == "CT"]
    assert c.empty or (c["comoving_likely_bound"].all() and not c["rankable"].any())
    # the ranked list leads with the resolved link, and the report prices it
    assert lt.iloc[0]["t_source_id"] == "AT"
    r100 = rep["beams"]["radio_100m_1p4ghz"]
    assert r100["n_expected_isotropic"] < 0.1 and r100["n_links_p50_rankable"] >= 1
    assert r100["n_links_p50_rankable_t_unobserved"] == r100["n_links_p50_rankable"] - 1
    assert "optical_10m_1um" not in rep["beams"] and rep["bl_names_unresolved"] == 7


def test_close_bound_pairs_are_not_ranked(conf):
    # run 35992222801's first list: 1-2" pairs with dv_tan 3-5 km/s (orbital
    # motion at 20-150 AU) and 3-6 sigma parallax differences (close-pair bias)
    s = _tl_sample()
    extra = pd.DataFrame([
        dict(source_id="DT", ra=200.0, dec=5.0, parallax=1000 / 40.0, parallax_error=0.02, pmra=50.0, pmdec=0.0),
        dict(source_id="DR", ra=200.0 + 1.5 / 3600, dec=5.0, parallax=1000 / 39.0, parallax_error=0.02,
             pmra=70.0, pmdec=0.0)])
    s = pd.concat([s, extra], ignore_index=True)
    lt, rep = tl.build_targetlist(s, geo.beam_grid(conf["beams"]), dict(conf["targetlist"], n_mc=100))
    d = lt[lt["t_source_id"] == "DT"]
    assert len(d) and d["close_pair_astrometry"].all() and not d["rankable"].any()
    assert rep["n_excluded_close_pair"] >= 1


def test_targetlist_stage_without_a_sample_is_no_data(tmp_path, conf):
    from seti.relay.run import stage_targetlist

    rep = stage_targetlist(conf, tmp_path)
    assert rep["verdict"].startswith("NO_DATA_REACHED")
