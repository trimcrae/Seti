"""Offline tests for the ROMAN orchestration: the object store, the verdict
rules of ``assess`` (no sky null without data, no candidate tier on simulated
inputs), the readiness diff, and the alerts hook.  No network (conftest)."""

from __future__ import annotations

import json

import numpy as np

from seti.roman import run as R
from seti.roman.schema import DQCutout, Funnel, LightCurve, Spectrum, load_roman_config


def _lc(star="s1", band="F146", n=50, zp=27.7):
    t = 61500 + np.arange(n) / 120.0
    f = 1000 + np.zeros(n)
    return LightCurve(star, 268.0, -29.0, band, t, f, 10 + np.zeros(n), survey="GBTDS",
                      flux_zp_ab=zp, dq=np.zeros(n, dtype=int))


def test_config_loads_and_carries_the_verify_flags():
    conf = load_roman_config()
    assert conf["config_status"] == "loaded"
    assert conf["surveys"]["GBTDS"]["verify"] is True
    assert conf["lens"]["density_floor_g_cc"] > 0
    assert "Nd:YAG 1064" in [r["name"] for r in conf["lines"]["industrial_lines_um"]]


def test_object_store_round_trips_each_structure(tmp_path):
    store = tmp_path / "normalized"
    lc = _lc()
    row = R.save_object(lc, "lightcurve", store, simulated=True, source_uri="s3://x/lc.parquet")
    back = R.load_object(row, store)
    assert isinstance(back, LightCurve)
    assert back.star_id == "s1" and back.n == 50 and back.flux_zp_ab == 27.7
    assert np.array_equal(back.dq, lc.dq)

    sp = Spectrum("g1", 10.0, 20.0, "G150", np.linspace(1.0, 1.9, 300), np.ones(300), np.ones(300) * 0.1,
                  resolving_power=461 * np.linspace(1.0, 1.9, 300), is_point_source=True)
    row = R.save_object(sp, "spectrum", store, simulated=False)
    back = R.load_object(row, store)
    assert isinstance(back, Spectrum) and back.r_at(1.5) > 600 and back.is_point_source is True

    cut = DQCutout("img1", np.zeros((16, 16), dtype=np.uint32), [{"star_id": "a", "x": 3.0, "y": 4.0}],
                   band="F146", mjd=61500.0)
    row = R.save_object(cut, "dq", store, simulated=False)
    back = R.load_object(row, store)
    assert isinstance(back, DQCutout) and back.stars[0]["star_id"] == "a" and back.shape == (16, 16)


def _write_ckpt(out_dir, channel, name, rec):
    d = out_dir / "checkpoints" / channel
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(rec))


def test_assess_with_nothing_screened_is_not_a_sky_null(tmp_path):
    conf = load_roman_config()
    (tmp_path / "probe.json").write_text(json.dumps({"data_state": "NOT_YET_PUBLIC"}))
    top = R.assess(tmp_path, conf)
    assert top["verdict"] == "ROMAN_NOT_YET_PUBLIC"
    assert top["n_products_analysed"] == 0
    for ch in R.CHANNELS:
        s = json.loads((tmp_path / ch / "summary.json").read_text())
        assert s["verdict"] == "ROMAN_NOT_YET_PUBLIC"
        c = json.loads((tmp_path / ch / "candidates.json").read_text())
        assert c["n"] == 0


def test_assess_never_writes_a_candidate_tier_on_simulated_inputs(tmp_path):
    conf = load_roman_config()
    (tmp_path / "probe.json").write_text(json.dumps({"data_state": "SIMULATIONS_ONLY"}))
    tiers = conf["lens"]["tiers"]
    _write_ckpt(tmp_path, "lens", "a", {"object_id": "a", "simulated": True, "status": "screened",
                                        "tier": tiers["candidate"], "funnel": Funnel().as_dict()})
    top = R.assess(tmp_path, conf, channels=("lens",))
    assert top["verdict"] == "SIMULATION_PACES_OK"
    s = json.loads((tmp_path / "lens" / "summary.json").read_text())
    assert s["verdict"] == "SIMULATION_PACES_OK" and s["simulated_inputs"] is True
    c = json.loads((tmp_path / "lens" / "candidates.json").read_text())
    assert c["n"] == 0


def test_assess_reports_a_candidate_tier_on_flight_data_and_errors_as_degraded(tmp_path):
    conf = load_roman_config()
    (tmp_path / "probe.json").write_text(json.dumps({"data_state": "MISSION_DATA_PRESENT"}))
    tiers = conf["lens"]["tiers"]
    _write_ckpt(tmp_path, "lens", "a", {"object_id": "a", "simulated": False, "status": "screened",
                                        "tier": tiers["candidate"], "funnel": Funnel().as_dict()})
    _write_ckpt(tmp_path, "lens", "b", {"object_id": "b", "simulated": False, "status": "screened",
                                        "tier": "LENSING_NO_OCCULTATION", "funnel": Funnel().as_dict()})
    top = R.assess(tmp_path, conf, channels=("lens",))
    assert top["verdict"] == "CANDIDATES_PENDING_VET"
    c = json.loads((tmp_path / "lens" / "candidates.json").read_text())
    assert c["n"] == 1 and c["candidates"][0]["object_id"] == "a"

    # Half the shard errored: the channel is DEGRADED, not clean.
    _write_ckpt(tmp_path, "lens", "c", {"object_id": "c", "simulated": False, "status": "error",
                                        "error": "boom"})
    _write_ckpt(tmp_path, "lens", "d", {"object_id": "d", "simulated": False, "status": "error",
                                        "error": "boom"})
    top = R.assess(tmp_path, conf, channels=("lens",))
    s = json.loads((tmp_path / "lens" / "summary.json").read_text())
    assert s["verdict"] == "DEGRADED_SOURCE" and s["n_errors"] == 2


def test_readiness_diff_flags_the_first_sight_of_mission_data(tmp_path, monkeypatch):
    conf = load_roman_config()
    states = iter(["NOT_YET_PUBLIC", "SIMULATIONS_ONLY", "MISSION_DATA_PRESENT"])

    def fake_probe(out_dir, conf):
        rec = {"data_state": next(states), "data_state_evidence": ["stub"], "packages": {},
               "n_endpoints_reached": 1}
        (out_dir / "probe.json").write_text(json.dumps(rec))
        return rec

    def fake_inventory(out_dir, conf, n_shards=4, probe_record=None):
        return {"products": [], "counts_by_kind": {}, "verdict": "NO_DATA_REACHED"}

    monkeypatch.setattr(R, "probe", fake_probe)
    monkeypatch.setattr(R, "inventory", fake_inventory)
    r1 = R.readiness(tmp_path, conf)
    assert r1["data_state"] == "NOT_YET_PUBLIC" and r1["state_changed"] is False
    r2 = R.readiness(tmp_path, conf)
    assert r2["first_seen_simulations"] is True and r2["state_changed"] is True
    r3 = R.readiness(tmp_path, conf)
    assert r3["first_seen_mission_data"] is True
    assert len(r3["history"]) == 3

    from seti.alerts import roman_alerts
    root = tmp_path / "root"
    (root / "results" / "roman").mkdir(parents=True)
    (root / "results" / "roman" / "readiness.json").write_text(json.dumps(r3))
    alerts = roman_alerts(root)
    assert [a.key for a in alerts] == ["roman:milestone:mission_data_present"]
    assert alerts[0].severity == "milestone"


def test_roman_candidate_alert_only_on_flight_data(tmp_path):
    from seti.alerts import roman_alerts
    d = tmp_path / "results" / "roman" / "flash"
    d.mkdir(parents=True)
    d2 = tmp_path / "results" / "roman"
    (d2 / "readiness.json").write_text(json.dumps({"data_state": "NOT_YET_PUBLIC"}))
    (d / "summary.json").write_text(json.dumps({"verdict": "SUB_EXPOSURE_FLASH_CANDIDATES_PENDING_VET",
                                                "simulated_inputs": True, "funnel": {}}))
    assert roman_alerts(tmp_path) == []
    (d / "summary.json").write_text(json.dumps({"verdict": "SUB_EXPOSURE_FLASH_CANDIDATES_PENDING_VET",
                                                "simulated_inputs": False, "funnel": {"a": 1}}))
    got = roman_alerts(tmp_path)
    assert len(got) == 1 and got[0].severity == "candidate" and got[0].channel == "roman"


def test_cli_registers_roman_passthrough():
    import pytest

    from seti.cli import main
    with pytest.raises(SystemExit):
        main(["roman", "--help"])


def test_the_two_openuniverse_band_maps_agree():
    """`archive.openuniverse_band_map` (inventory) and `openuniverse.band_tokens`
    (readers) name the same filters; a drift between them would classify an
    image under one band and read it under another."""
    conf = load_roman_config()
    assert conf["archive"]["openuniverse_band_map"] == conf["openuniverse"]["band_tokens"]
