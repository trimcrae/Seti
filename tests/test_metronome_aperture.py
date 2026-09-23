"""METRONOME: the aperture-scale contamination cone, per-star vet files, and
the state-based rebuild of summary.json.  Offline; no socket is opened.

The case every test here is anchored to is the measured one (STATUS.md,
2026-09-22): kepler:5879574's flare clock at 0.4232741 d was KIC 5879583, an
RR Lyrae 13.3" away with P = 0.4232946 d in VSX and ZTF (Chen+2020) and class
RR in Gaia DR3 -- outside the 3" identity cone the assess stage ran.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from seti.metronome import acquire as acq
from seti.metronome.vet import HARD_VETO_ORDER, aperture_contamination, vet_star

P_CLOCK = 0.4232740931317277          # stars_vetted.csv, kepler:5879574
P_RRLYR = 0.4232946                   # VSX / ZTF, KIC 5879583
# ZTFJ193127.18+410759.8 is KIC 5879583; the target is put 13.3" due north
RR_RA, RR_DEC = 292.863250, 41.133278
TGT_RA, TGT_DEC = RR_RA, RR_DEC - 13.3 / 3600.0


def _sep(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    s = math.sin((d2 - d1) / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2
    return math.degrees(2 * math.asin(math.sqrt(s))) * 3600.0


def _sky_rows():
    """What the three variability catalogues hold around the target."""
    return {
        "B/vsx/vsx": [
            {"Name": "KIC 5879574", "Type": "ROT", "Period": 11.107,
             "RAJ2000": TGT_RA, "DEJ2000": TGT_DEC},
            {"Name": "KIC 5879583", "Type": "RR", "Period": P_RRLYR,
             "RAJ2000": RR_RA, "DEJ2000": RR_DEC},
            {"Name": "far RR", "Type": "RRAB", "Period": P_RRLYR,
             "RAJ2000": RR_RA, "DEJ2000": RR_DEC + 300 / 3600.0},
        ],
        "I/358/vclassre": [
            {"Source": "2053563953175635712", "Class": "RR",
             "RA_ICRS": RR_RA, "DE_ICRS": RR_DEC},
        ],
        "J/ApJS/249/18": [
            {"ID": "ZTFJ193127.18+410759.8", "Type": "RR", "Per": P_RRLYR,
             "RAJ2000": RR_RA, "DEJ2000": RR_DEC},
        ],
    }


def sky_cone(table, ra, dec, radius_arcsec):
    """A cone that honours its radius, as VizieR does."""
    rows = [r for r in _sky_rows().get(table, [])
            if _sep(ra, dec, r.get("RAJ2000", r.get("RA_ICRS")),
                    r.get("DEJ2000", r.get("DE_ICRS"))) <= radius_arcsec]
    return pd.DataFrame(rows)


VARI = {"vsx": {"table": "B/vsx/vsx", "period_patterns": ["^period$", "^per$"],
                "type_patterns": ["^type$"]},
        "gaia_dr3_vari": {"table": "I/358/vclassre",
                          "period_patterns": ["^period$", "^pf$"],
                          "type_patterns": ["^class$"]},
        "ztf_chen2020": {"table": "J/ApJS/249/18", "period_patterns": ["^per$"],
                         "type_patterns": ["^type$"]}}


# ---------------------------------------------------------------------------
# the cone
# ---------------------------------------------------------------------------
def test_the_identity_cone_alone_misses_the_rr_lyrae():
    """The 3" cone is what assess ran: it sees the target's own ROT entry and
    nothing at the clock period.  This is the failure being fixed."""
    pos = pd.DataFrame({"star_id": ["5879574"], "ra": [TGT_RA], "dec": [TGT_DEC]})
    v, reached = acq.fetch_variable_context(pos, VARI, cone_fn=sky_cone, radius_arcsec=3.0)
    periods = [p for (_s, p, _t) in v.get("5879574", []) if np.isfinite(p)]
    assert periods == [11.107]
    assert reached["5879574"] == set(VARI)
    from seti.metronome.vet import periodic_variable
    assert periodic_variable(P_CLOCK, v["5879574"])[0] is False


def test_the_aperture_cone_finds_kic_5879583_and_not_the_target():
    pos = pd.DataFrame({"star_id": ["5879574"], "ra": [TGT_RA], "dec": [TGT_DEC]})
    nb, reached = acq.fetch_aperture_neighbours(
        pos, VARI, radius_arcsec_by_mission={"kepler": 20.0, "tess": 120.0},
        mission="kepler", identity_radius_arcsec=3.0, cone_fn=sky_cone)
    rows = nb["5879574"]
    names = sorted(r["name"] for r in rows)
    # the target's own ROT entry is inside 3" and is left to the identity veto;
    # the RR Lyrae 300" away is outside a Kepler aperture
    assert names == ["2053563953175635712", "KIC 5879583", "ZTFJ193127.18+410759.8"]
    for r in rows:
        assert r["sep_arcsec"] == pytest.approx(13.3, abs=0.05)
        assert r["sep_measured"]
    assert reached["5879574"] == set(VARI)


def test_the_kic_5879574_case_is_flagged_as_contamination():
    """The acceptance test the fix was specified by."""
    pos = pd.DataFrame({"star_id": ["5879574"], "ra": [TGT_RA], "dec": [TGT_DEC]})
    nb, reached = acq.fetch_aperture_neighbours(
        pos, VARI, radius_arcsec_by_mission={"kepler": 20.0}, mission="kepler",
        cone_fn=sky_cone)
    rec = {"status": "scanned", "fdr_significant": True, "fdr_watch": True,
           "period": P_CLOCK, "Q": 0.97, "jitter": 0.005, "f_in_window": 1.0,
           "gap_integer_frac": 0.95, "n_gaps_used": 40, "p_window": 1e-12,
           "p_shuffle": 0.01, "p_pool": 0.005, "pn_n_trials": 200, "cycles_span": 3000.0,
           "n_events": 21, "mission": "kepler"}
    ctx = {"mission": "kepler", "prot": 11.107, "catalogued_periods": [],
           "variability_catalogues_reached": True,
           "aperture_neighbours": nb["5879574"], "aperture_catalogues_reached": True}
    v = vet_star(rec, ctx)
    assert v["first_veto"] == "aperture_contaminating_variable"
    assert v["tier"] == "none"
    d = v["veto_detail"]["aperture_contaminating_variable"]
    best = d["matches"][0]
    assert best["name"] in ("KIC 5879583", "ZTFJ193127.18+410759.8")
    assert best["harmonic"] == 1.0
    assert best["frac_diff"] == pytest.approx(abs(P_CLOCK / P_RRLYR - 1), rel=1e-6)
    assert best["sep_arcsec"] == pytest.approx(13.3, abs=0.05)
    # VSX, Gaia and ZTF are one star: one distinct periodic neighbour
    assert d["n_distinct_with_period"] == 1
    assert 0.0 < d["p_chance_any"] < 0.02
    # it is a contamination veto, placed after the identity one
    assert HARD_VETO_ORDER.index("aperture_contaminating_variable") == \
        HARD_VETO_ORDER.index("periodic_variable") + 1


def test_the_radius_is_the_missions_aperture():
    """90" is inside a TESS aperture (21"/px) and outside a Kepler one."""
    nb_ra, nb_dec = TGT_RA, TGT_DEC + 90.0 / 3600.0

    def cone(table, ra, dec, r):
        if table != "B/vsx/vsx":
            return pd.DataFrame()
        rows = [{"Name": "EB", "Type": "EA", "Period": 2.0 * 3.0,
                 "RAJ2000": nb_ra, "DEJ2000": nb_dec}]
        return pd.DataFrame([x for x in rows
                             if _sep(ra, dec, x["RAJ2000"], x["DEJ2000"]) <= r])

    pos = pd.DataFrame({"star_id": ["1"], "ra": [TGT_RA], "dec": [TGT_DEC]})
    radii = {"kepler": 20.0, "tess": 120.0}
    kep, _ = acq.fetch_aperture_neighbours(pos, {"vsx": VARI["vsx"]},
                                           radius_arcsec_by_mission=radii,
                                           mission="kepler", cone_fn=cone)
    tess, _ = acq.fetch_aperture_neighbours(pos, {"vsx": VARI["vsx"]},
                                            radius_arcsec_by_mission=radii,
                                            mission="tess", cone_fn=cone)
    assert "1" not in kep
    assert tess["1"][0]["sep_arcsec"] == pytest.approx(90.0, abs=0.1)
    # an EB at 6 d contaminates a 3 d clock (its P/2) -- a low harmonic
    hit, d = aperture_contamination(3.0, tess["1"])
    assert hit and d["matches"][0]["harmonic"] == 0.5


def test_an_unrelated_neighbour_is_reported_not_vetoed():
    nb = [{"source": "vsx", "name": "V1", "period": 17.3, "vtype": "ROT", "sep_arcsec": 50.0}]
    hit, d = aperture_contamination(3.137, nb)
    assert not hit and d["n_with_period"] == 1 and d["matches"] == []
    rec = {"status": "scanned", "fdr_significant": True, "fdr_watch": True, "period": 3.137,
           "Q": 0.97, "jitter": 0.005, "f_in_window": 1.0, "gap_integer_frac": 0.95,
           "n_gaps_used": 40, "p_window": 1e-9, "p_shuffle": 0.02, "p_pool": 0.005,
           "pn_n_trials": 200, "cycles_span": 300.0, "n_events": 120,
           "jitter_floor": 1e-6, "mission": "kepler"}
    ctx = {"mission": "kepler", "prot": 11.0, "variability_catalogues_reached": True,
           "aperture_neighbours": nb, "aperture_catalogues_reached": True}
    v = vet_star(rec, ctx)
    assert v["first_veto"] is None
    assert "aperture_variable_neighbour" in v["flags"]
    assert v["tier"] == "candidate"


def test_an_unreached_aperture_cone_caps_the_tier_at_interest():
    rec = {"status": "scanned", "fdr_significant": True, "fdr_watch": True, "period": 3.137,
           "Q": 0.97, "jitter": 0.005, "f_in_window": 1.0, "gap_integer_frac": 0.95,
           "n_gaps_used": 40, "p_window": 1e-9, "p_shuffle": 0.02, "p_pool": 0.005,
           "pn_n_trials": 200, "cycles_span": 300.0, "n_events": 120,
           "jitter_floor": 1e-6, "mission": "kepler"}
    ctx = {"mission": "kepler", "prot": 11.0, "variability_catalogues_reached": True,
           "aperture_neighbours": [], "aperture_catalogues_reached": False}
    v = vet_star(rec, ctx)
    assert "aperture_catalogue_unreached" in v["flags"]
    assert v["tier"] == "interest"


def test_a_row_without_coordinates_is_kept_not_taken_for_the_target():
    def cone(table, ra, dec, r):
        return pd.DataFrame({"Name": ["X"], "Type": ["RR"], "Period": [P_RRLYR]})

    pos = pd.DataFrame({"star_id": ["1"], "ra": [TGT_RA], "dec": [TGT_DEC]})
    nb, _ = acq.fetch_aperture_neighbours(pos, {"vsx": VARI["vsx"]},
                                          radius_arcsec_by_mission={"kepler": 20.0},
                                          mission="kepler", cone_fn=cone)
    assert nb["1"][0]["sep_measured"] is False
    assert aperture_contamination(P_CLOCK, nb["1"])[0] is True


def test_vizier_sep_column_is_used_when_there_are_no_coordinates():
    roles = acq.resolve_columns(["Name", "_sep_arcsec"], {"ra": acq._RA_PATTERNS,
                                                          "dec": acq._DE_PATTERNS})
    row = pd.Series({"Name": "X", "_sep_arcsec": 12.5})
    assert acq.row_separation_arcsec(row, 10.0, 10.0, roles) == pytest.approx(12.5)


# ---------------------------------------------------------------------------
# positions: VizieR's TIC returned nothing for every TESS star on the runner
# ---------------------------------------------------------------------------
def test_tess_positions_fall_back_to_mast():
    def vizier(adql):
        if "TAP_SCHEMA.columns" in adql:
            return pd.DataFrame({"column_name": ["TIC", "RAJ2000", "DEJ2000"]})
        return pd.DataFrame(columns=["TIC", "RAJ2000", "DEJ2000"])     # zero rows

    seen = []

    def mast(ids):
        seen.extend(ids)
        return pd.DataFrame({"ID": ids, "ra": [100.0] * len(ids), "dec": [-20.0] * len(ids)})

    pos = acq.fetch_positions(["260128333", "350479496"], "tess", query_fn=vizier, mast_fn=mast)
    assert sorted(pos["star_id"]) == ["260128333", "350479496"]
    assert sorted(seen) == ["260128333", "350479496"]
    # an offline caller (injected VizieR, no MAST stand-in) never opens a socket
    pos2 = acq.fetch_positions(["1"], "tess", query_fn=vizier)
    assert len(pos2) == 0


# ---------------------------------------------------------------------------
# the assess stage end to end, with the aperture cone
# ---------------------------------------------------------------------------
def _records():
    base = {"status": "scanned", "Q": 0.97, "jitter": 0.005, "f_in_window": 1.0,
            "gap_integer_frac": 0.95, "n_gaps_used": 40, "p_shuffle": 0.01,
            "p_window_source": "empirical", "energy_phase_p": 0.4,
            "wn_truncated_by_budget": False, "p_pool": 0.005, "pn_n_trials": 200,
            "pn_n_exceed": 0, "jitter_floor": 1e-6, "phase_spacing": 1e-5,
            "cycles_span": 3000.0, "cycles_hit": 40, "n_events": 120, "mission": "kepler",
            "catalogue": "kepler_yang2019"}
    recs = [dict(base, star_key="kepler:5879574", star_id="5879574", period=P_CLOCK,
                 p_window=1e-12)]
    rng = np.random.default_rng(1)
    for i in range(40):
        recs.append(dict(base, star_key=f"kepler:{9000 + i}", star_id=str(9000 + i),
                         period=float(10 ** rng.uniform(-0.3, 1.5)), p_window=0.5))
    return recs


class _KicTAP:
    def __call__(self, adql):
        if "TAP_SCHEMA.columns" in adql:
            return pd.DataFrame({"column_name": ["KIC", "RAJ2000", "DEJ2000"]})
        ids = [s.strip() for s in adql.split("IN (")[1].rstrip(")").split(",")]
        return pd.DataFrame({"KIC": ids, "RAJ2000": [TGT_RA] * len(ids),
                             "DEJ2000": [TGT_DEC] * len(ids)})


def test_assess_demotes_kic_5879574_on_the_aperture_cone(tmp_path):
    from seti.metronome.reconcile import check_consistency
    from seti.metronome.run import load_metronome_config, stage_assess

    conf = load_metronome_config()
    conf["variability_catalogues"] = VARI
    s = stage_assess(conf, tmp_path, offline=False, query_fn=_KicTAP(), cone_fn=sky_cone,
                     records=_records())
    ap = s["aperture_contamination"]
    assert ap["enabled"] and ap["n_shortlist"] == 1
    assert ap["n_flagged_contaminating"] == 1
    f = ap["flagged_contaminating"][0]
    assert f["star_key"] == "kepler:5879574"
    assert f["first_veto"] == "aperture_contaminating_variable"
    assert s["n_candidates"] == 0 and s["n_interest"] == 0
    assert "NO_CLOCK_CANDIDATES" in s["verdict"]
    vet = pd.read_csv(tmp_path / "stars_vetted.csv", dtype={"star_key": str})
    row = vet[vet["star_key"] == "kepler:5879574"].iloc[0]
    assert row["first_veto"] == "aperture_contaminating_variable"
    assert check_consistency(tmp_path) == []


def test_assess_without_the_rr_lyrae_keeps_the_star(tmp_path):
    from seti.metronome.run import load_metronome_config, stage_assess

    def empty(table, ra, dec, r):
        return pd.DataFrame()

    conf = load_metronome_config()
    conf["variability_catalogues"] = VARI
    s = stage_assess(conf, tmp_path, offline=False, query_fn=_KicTAP(), cone_fn=empty,
                     records=_records())
    assert s["aperture_contamination"]["n_flagged_contaminating"] == 0
    assert s["aperture_contamination"]["frac_shortlist_reached_all_sources"] == 1.0
    # rotation unknown in this synthetic run, so interest rather than candidate
    assert s["n_candidates"] + s["n_interest"] == 1


# ---------------------------------------------------------------------------
# per-star vet files and the state-based rebuild
# ---------------------------------------------------------------------------
FIVE = {  # the five tess_tu2022 vets of 2026-09-23, verdicts as they landed
    "tess:350479496": "DEGRADED(gaia:gaia_source); NO_MUNDANE_EXPLANATION_FOUND [P=4.194131 d]",
    "tess:260128333": "MUNDANE_EXPLANATION_FOUND(NARROW_DIP_AT_P:width=0.05) [P=2.921710 d]",
    "tess:398943781": "MUNDANE_EXPLANATION_FOUND(NARROW_DIP_AT_P:width=0.03) [P=3.052465 d]",
    "tess:63834969": "MUNDANE_EXPLANATION_FOUND(NARROW_DIP_AT_P:width=0.03) [P=2.055465 d]",
    "tess:32449963": "MUNDANE_EXPLANATION_FOUND(EVENTS_CLUSTER_MORE_TIGHTLY_AT_2P) [P=6.520392 d]",
}


def _state(tmp_path):
    keys = ["kepler:5879574"] + list(FIVE)
    pd.DataFrame({"star_key": keys + ["kepler:1"],
                  "tier": ["candidate"] + ["interest"] * 5 + ["watch"],
                  "first_veto": [None] * 7, "flags": ["p_extrapolated"] * 7}).to_csv(
        tmp_path / "stars_vetted.csv", index=False)
    (tmp_path / "candidates.json").write_text(json.dumps({
        "generated_utc": "2026-09-21T21:42:31Z", "verdict": "stale",
        "candidates": [{"star_key": k, "tier": "interest", "flags": "p_extrapolated"}
                       for k in keys],
        "watch": [{"star_key": "kepler:1", "tier": "watch"}]}))
    (tmp_path / "summary.json").write_text(json.dumps({
        "verdict": "x", "degraded": [], "funnel": {},
        "generated_utc": "2026-09-21T21:42:31Z"}))


def test_parallel_vets_do_not_erase_each_other(tmp_path):
    """Five runs, five files, and the rebuild applies all five whichever order
    they land in -- the measured failure left one of five demotions."""
    from seti.metronome.reconcile import check_consistency, reconcile_all, vetstar_filename
    from seti.metronome.vetstar import reconcile_vetstar

    _state(tmp_path)
    for k, v in FIVE.items():
        rep = {"star_key": k, "verdict": v, "generated_utc": "2026-09-23T00:35:00Z"}
        reconcile_vetstar(tmp_path, rep)
        assert (tmp_path / vetstar_filename(k)).exists()
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["funnel"]["stars_demoted_by_vetstar"] == 4
    assert s["verdict"].count("VETSTAR_DEMOTED") == 1
    assert "VETSTAR_DEMOTED_4" in s["verdict"]
    # kepler:5879574 was never vetted in this fixture and 350479496's vet found
    # nothing: the two stay at their assess tiers
    assert s["n_candidates"] == 1 and s["n_interest"] == 1
    assert s["vetstar"]["n_vetted"] == 5
    assert check_consistency(tmp_path) == []
    # a second rebuild is a fixed point
    again = reconcile_all(tmp_path)
    assert again["verdict"] == s["verdict"]
    c = json.loads((tmp_path / "candidates.json").read_text())
    assert c["verdict"] == s["verdict"] and c["generated_utc"] == s["generated_utc"]


def test_an_extra_demotion_token_is_never_appended_twice(tmp_path):
    """'VETSTAR_DEMOTED_2; VETSTAR_DEMOTED_1' -- the invocation's own count
    beside the records' -- cannot be produced any more."""
    from seti.metronome.redetect import rebuild_summary

    _state(tmp_path)
    c = json.loads((tmp_path / "candidates.json").read_text())
    c["candidates"][0].update(tier="none", first_veto="vet_contaminating_variable_at_p")
    c["candidates"][5].update(tier="none", first_veto="vet_events_cluster_more_tightly_at_2p")
    (tmp_path / "candidates.json").write_text(json.dumps(c))
    s = rebuild_summary(tmp_path, {"verdict": "x", "degraded": [], "funnel": {}},
                        stage="vetstar", extra_tokens=["VETSTAR_DEMOTED_1"])
    assert "VETSTAR_DEMOTED_2" in s["verdict"]
    assert "VETSTAR_DEMOTED_1" not in s["verdict"]


def test_consistency_check_catches_a_patched_summary(tmp_path):
    from seti.metronome.reconcile import check_consistency, reconcile_all

    _state(tmp_path)
    reconcile_all(tmp_path)
    s = json.loads((tmp_path / "summary.json").read_text())
    s["tiers"]["candidate"] = 1
    s["n_candidates"] = 0
    s["verdict"] += "; VETSTAR_DEMOTED_1"
    (tmp_path / "summary.json").write_text(json.dumps(s))
    probs = check_consistency(tmp_path)
    assert any("n_candidates" in p for p in probs)
    assert any("VETSTAR_DEMOTED_1" in p for p in probs)


def test_a_rebuild_restores_a_demotion_whose_file_was_removed(tmp_path):
    """State, not increments: a demotion exists exactly while its record does."""
    from seti.metronome.reconcile import reconcile_all, vetstar_filename
    from seti.metronome.vetstar import reconcile_vetstar

    _state(tmp_path)
    k = "tess:398943781"
    reconcile_vetstar(tmp_path, {"star_key": k, "verdict": FIVE[k]})
    assert json.loads((tmp_path / "summary.json").read_text())["n_interest"] == 4
    (tmp_path / vetstar_filename(k)).unlink()
    reconcile_all(tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text())["n_interest"] == 5


# ---------------------------------------------------------------------------
# the vet's own defects, found in the five recovered reports
# ---------------------------------------------------------------------------
def test_long_period_on_a_short_span_still_gets_a_control_null():
    """tess:350479496 (P = 4.19 d) and tess:32449963 (P = 6.52 d) came back
    with n_control = 0 and a 24-29%-deep narrow eclipse passed as
    NO_MUNDANE_EXPLANATION_FOUND."""
    from seti.metronome.vetstar import fold_amplitude_significance

    rng = np.random.default_rng(0)
    t = np.concatenate([np.arange(0, 27, 0.0208), np.arange(215, 242, 0.0208)])
    P = 6.520392
    ph = np.mod(t / P, 1.0)
    y = rng.normal(0, 1e-3, len(t)) - 0.29 * (np.abs(ph - 0.5) < 0.01)
    d = fold_amplitude_significance(t, y, P, n_control=60)
    assert d["n_control"] > 0
    assert d["band_used"] > 0.35
    assert d["p_empirical"] < 0.05


def test_an_empty_null_is_unreached_not_a_pass():
    from seti.metronome.vetstar import vet_verdict

    rep = {"period": 4.19, "status": "analysed",
           "fold_significance": {"n_control": 0}, "fold_significance_flares_masked":
           {"n_control": 0}, "fold_significance_2p": {"n_control": 0},
           "gaia": {"status": "OK"}}
    v, _ = vet_verdict(rep)
    assert v.startswith("DEGRADED(")
    assert "control_null:fold_significance" in rep["unreached"]


def test_the_crest_needs_the_events_to_cluster():
    """tess:260128333 was called EVENTS_ON_THE_CREST with Rayleigh p = 0.91."""
    from seti.metronome.vetstar import vet_verdict

    base = {"period": 2.92, "status": "analysed", "gaia": {"status": "OK"},
            "fold_significance_flares_masked": {"p_empirical": 0.005, "z_control": 19.7,
                                                "n_control": 200},
            "fold_significance": {"n_control": 200}, "fold_significance_2p": {"n_control": 200},
            "event_phase_offset_from_photometric_max": 0.064}
    v1, _ = vet_verdict(dict(base, event_phase_rayleigh_at_p={"p": 0.91, "rbar": 0.09}))
    assert "EVENTS_ON_THE_CREST" not in v1
    v2, _ = vet_verdict(dict(base, event_phase_rayleigh_at_p={"p": 3e-28, "rbar": 0.96}))
    assert "EVENTS_ON_THE_CREST" in v2


def test_vet_star_tolerates_nan_counts_from_old_shards():
    """The 2026-09-21 shards have no pool-null columns; read back from CSV
    they are NaN, and int(nan or 0) crashed the re-assess (run 35862296965)."""
    rec = {"status": "scanned", "fdr_significant": True, "fdr_watch": True,
           "period": 3.0, "pn_n_trials": float("nan"), "n_gaps_used": float("nan"),
           "n_core": float("nan"), "n_gaps_core": float("nan"), "n_events": float("nan")}
    v = vet_star(rec, {"mission": "tess"})
    assert "pool_null_unreached" in v["flags"]


def test_stage_aperture_flags_the_shortlist_and_reconciles(tmp_path):
    from seti.metronome.aperture import stage_aperture
    from seti.metronome.reconcile import check_consistency
    from seti.metronome.run import load_metronome_config

    pd.DataFrame({
        "star_key": ["kepler:5879574", "kepler:42", "kepler:43"],
        "star_id": ["5879574", "42", "43"], "mission": ["kepler"] * 3,
        "catalogue": ["kepler_yang2019"] * 3,
        "period": [P_CLOCK, 2.5, 7.0], "tier": ["candidate", "interest", "none"],
        "first_veto": [None, None, "not_significant"], "flags": ["", "", ""],
        "fdr_watch": [True, True, False]}).to_csv(tmp_path / "stars_vetted.csv", index=False)
    (tmp_path / "candidates.json").write_text(json.dumps({
        "candidates": [{"star_key": "kepler:5879574", "tier": "candidate"},
                       {"star_key": "kepler:42", "tier": "interest"}], "watch": []}))
    (tmp_path / "summary.json").write_text(json.dumps({
        "verdict": "x", "degraded": [], "funnel": {}, "generated_utc": "2026-09-21T21:42:31Z"}))

    class Tap:
        def __call__(self, adql):
            if "TAP_SCHEMA.columns" in adql:
                return pd.DataFrame({"column_name": ["KIC", "RAJ2000", "DEJ2000"]})
            ids = [s.strip() for s in adql.split("IN (")[1].rstrip(")").split(",")]
            # kepler:42 is far from the RR Lyrae
            return pd.DataFrame({"KIC": ids,
                                 "RAJ2000": [TGT_RA if i == "5879574" else 10.0 for i in ids],
                                 "DEJ2000": [TGT_DEC if i == "5879574" else 10.0 for i in ids]})

    conf = load_metronome_config()
    conf["variability_catalogues"] = VARI
    rep = stage_aperture(conf, tmp_path, query_fn=Tap(), cone_fn=sky_cone)
    assert rep["n_shortlist"] == 2 and rep["n_positioned"] == 2
    assert rep["frac_aperture_reached"] == 1.0 and rep["frac_identity_reached"] == 1.0
    assert rep["flagged"] == ["kepler:5879574"]
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["n_candidates"] == 0 and s["n_interest"] == 1
    assert "APERTURE_DEMOTED_1" in s["verdict"]
    assert s["funnel"]["stars_demoted_by_aperture"] == 1
    assert s["aperture_contamination"]["demoted"] == [
        "kepler:5879574:aperture_contaminating_variable"]
    c = json.loads((tmp_path / "candidates.json").read_text())
    row = c["candidates"][0]
    assert row["first_veto"] == "aperture_contaminating_variable"
    assert row["aperture"]["contamination_detail"]["matches"][0]["sep_arcsec"] == \
        pytest.approx(13.3, abs=0.05)
    assert check_consistency(tmp_path) == []


def test_one_product_per_sector():
    from seti.metronome.vetstar import one_product_per_segment

    def seg(sector, cad_s, n):
        return {"sector": sector, "exptime_s": cad_s, "time": np.arange(n) * cad_s / 86400.0,
                "flux": np.ones(n)}

    segs = [seg(27, 20, 5000), seg(27, 120, 1000), seg(28, 120, 900), seg(28, 20, 4000),
            seg(29, 120, 800)]
    kept, n_dup = one_product_per_segment(segs)
    assert n_dup == 2
    assert [(s["sector"], s["exptime_s"]) for s in kept] == [(27, 120), (28, 120), (29, 120)]


def test_the_vet_reads_the_stars_own_catalogue_from_the_shortlist(tmp_path):
    from seti.metronome.vetstar import shortlist_catalogue

    (tmp_path / "candidates.json").write_text(json.dumps({"candidates": [
        {"star_key": "tess:260128333", "catalogue": "tess_tu2022", "tier": "interest"}]}))
    assert shortlist_catalogue(tmp_path, "tess:260128333") == "tess_tu2022"
    assert shortlist_catalogue(tmp_path, "tess:1") is None
