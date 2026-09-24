"""Offline pins for four shard-consistency defects found by the 2026-09-24 sweep
(docs/channel-brief.md §0.7).

1. TIDEMARK ran a literal matrix [0, 1] while its acquire command fell back to
   n_shards=4: an empty input fetched half the interleaved sky.
2. RING bd shards > 0 read the COMMITTED targets.csv while shard 0 fetched a
   fresh one: one run's shards could partition two different target lists.
3. GROWTH vet-gather pooled every vet/shard_* on disk, including shard dirs
   an earlier, wider vet left on the branch.
4. METRONOME assess read stars_redetect.csv (a downstream table) as screen
   output.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import yaml

WF = Path(".github/workflows")


def _wf(name: str) -> dict:
    return yaml.safe_load((WF / name).read_text())


# ---------------------------------------------------------------- 1. tidemark
def test_tidemark_matrix_is_derived_from_n_shards():
    doc = _wf("tidemark.yml")
    jobs = doc["jobs"]
    acq = jobs["acquire"]
    assert "plan" in (acq["needs"] if isinstance(acq["needs"], list) else [acq["needs"]])
    assert acq["strategy"]["matrix"]["shard"] == "${{ fromJSON(needs.plan.outputs.shards) }}"
    run = next(s["run"] for s in acq["steps"] if "tidemark-acquire" in (s.get("run") or ""))
    assert '--n-shards "${{ needs.plan.outputs.n }}"' in run
    assert "|| '4'" not in (WF / "tidemark.yml").read_text()
    # The plan's script: the same width both places, for any input.
    plan_run = jobs["plan"]["steps"][0]["run"]
    import textwrap
    body = textwrap.dedent(plan_run.split("<<'PY'", 1)[1].split("\n", 1)[1].rsplit("PY", 1)[0])
    for n_in, want in (("4", [0, 1, 2, 3]), ("1", [0]), ("", [0, 1])):
        import io
        import os
        from contextlib import redirect_stdout

        os.environ["N"] = n_in
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(compile(body, "plan", "exec"), {})
        out = dict(line.split("=", 1) for line in buf.getvalue().splitlines())
        assert json.loads(out["shards"]) == want and int(out["n"]) == len(want)


# ---------------------------------------------------------------- 2. ring bd
def _handoff():
    spec = importlib.util.spec_from_file_location("ring_bd_handoff",
                                                  "scripts/ring_bd_handoff.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ring_cfg():
    from seti.ring.run import load_ring_config
    return load_ring_config()


def test_ring_bd_every_shard_acquires_the_one_handed_off_list(tmp_path):
    h = _handoff()
    cfg = _ring_cfg()
    fresh = pd.DataFrame({"source_id": ["a", "b", "c", "d", "e", "f"],
                          "ra": range(6), "dec": range(6)})
    handoff = tmp_path / "handoff"
    h.fetch_targets(cfg, handoff, fetch=lambda _c: (fresh, {"catalogues": ["X"]}))
    seen = {}

    def neowise(t, d, c, shard, n):
        seen[shard] = list(t["source_id"].astype(str).iloc[shard::n])
        return {"n": len(seen[shard])}

    for shard in range(3):
        out = tmp_path / f"runner{shard}"
        bd = out / "bd"
        bd.mkdir(parents=True)
        # What a runner checkout holds: an EARLIER run's committed list.
        pd.DataFrame({"source_id": ["stale1", "stale2"], "ra": [0, 1], "dec": [0, 1]}) \
            .to_csv(bd / "targets.csv", index=False)
        (bd / "targets.csv").unlink()          # the workflow purges it first
        h.acquire_shard(cfg, out, shard, 3, handoff, fetchers={"bd_neowise": neowise})
    assert seen == {0: ["a", "d"], 1: ["b", "e"], 2: ["c", "f"]}


def test_ring_bd_missing_handoff_is_no_data_not_a_fresh_fetch(tmp_path):
    h = _handoff()
    rep = h.acquire_shard(_ring_cfg(), tmp_path / "r", 1, 3, tmp_path / "absent",
                          fetchers={"bd_neowise": lambda *a: {"n": 0}})
    assert rep["status"] == "NO_DATA_REACHED"
    assert "missing" in rep.get("error", "")


def test_ring_workflow_hands_the_bd_list_to_every_shard():
    jobs = _wf("ring.yml")["jobs"]
    assert "bd-targets" in jobs and "bd-targets" in jobs["bd"]["needs"]
    text = (WF / "ring.yml").read_text()
    bd_steps = jobs["bd"]["steps"]
    purge = next(s["run"] for s in bd_steps if s.get("name", "").startswith("Purge"))
    assert "results/ring/bd/targets.csv" in purge
    assert any((s.get("with") or {}).get("name") == "ring-bd-targets" for s in bd_steps)
    assert "scripts/ring_bd_handoff.py acquire" in text
    up = next(s for s in bd_steps if "upload-artifact" in str(s.get("uses")))
    assert "targets.csv" not in up["with"]["path"]


# ---------------------------------------------------------------- 3. growth
def _vet_shard(vet: Path, idx: int, n: int, names: list[str]):
    d = vet / f"shard_{idx:02d}"
    d.mkdir(parents=True)
    pd.DataFrame({"kepoi_name": names, "survives_vet": [True] * len(names)}) \
        .to_csv(d / "vetted.csv", index=False)
    (d / "summary.json").write_text(json.dumps({"stage": "vet", "shard": idx, "n_shards": n}))


def test_growth_vet_gather_ignores_shards_of_a_wider_earlier_vet(tmp_path):
    from seti.growth.direct import direct_vet_gather

    vet = tmp_path / "vet"
    for i in range(4):
        _vet_shard(vet, i, 4, [f"K{i}"])
    _vet_shard(vet, 4, 6, ["STALE4"])          # left on the branch by a 6-shard vet
    _vet_shard(vet, 5, 6, ["STALE5"])
    pd.DataFrame({"kepoi_name": ["K0", "K1", "K2", "K3"]}).to_csv(tmp_path / "candidates.csv",
                                                                   index=False)
    rep = direct_vet_gather({}, tmp_path, n_shards=4)
    got = set(pd.read_csv(vet / "vetted.csv")["kepoi_name"])
    assert got == {"K0", "K1", "K2", "K3"}
    assert rep["n_shards_found"] == 4 and rep["n_shards_expected"] == 4
    assert {x["dir"] for x in rep["shards_ignored"]} == {"shard_04", "shard_05"}


def test_growth_vet_gather_ignores_an_in_range_dir_from_another_width(tmp_path):
    from seti.growth.direct import direct_vet_gather

    vet = tmp_path / "vet"
    _vet_shard(vet, 0, 2, ["K0"])
    _vet_shard(vet, 1, 6, ["STALE1"])          # index fits, width does not
    rep = direct_vet_gather({}, tmp_path, n_shards=2)
    assert set(pd.read_csv(vet / "vetted.csv")["kepoi_name"]) == {"K0"}
    assert rep["shards_ignored"][0]["dir"] == "shard_01"


def test_growth_vet_gather_workflow_reads_only_this_runs_shards():
    job = _wf("growth_direct.yml")["jobs"]["vet_gather"]
    runs = [s.get("run") or "" for s in job["steps"]]
    i_purge = next(i for i, r in enumerate(runs) if "rm -rf results/growth/direct/vet/shard_*" in r)
    i_dl = next(i for i, s in enumerate(job["steps"]) if "download-artifact" in str(s.get("uses")))
    assert i_purge < i_dl
    assert any("--vet-shards" in r for r in runs)


# ---------------------------------------------------------------- 4. metronome
def test_metronome_assess_reads_only_screen_star_tables(tmp_path):
    from seti.metronome.run import screen_star_files

    conf = {"catalogues": {"kepler_yang2019": {}, "tess_gunther2020": {}}}
    for n in ("stars_kepler_yang2019_s0of8.csv", "stars_kepler_yang2019_s7of8.csv",
              "stars_tess_gunther2020.csv", "stars_redetect.csv", "stars_vetted.csv",
              "stars_somethingelse.csv"):
        (tmp_path / n).write_text("star_key\nx\n")
    got = [p.name for p in screen_star_files(conf, tmp_path)]
    assert got == ["stars_kepler_yang2019_s0of8.csv", "stars_kepler_yang2019_s7of8.csv",
                   "stars_tess_gunther2020.csv"]
    # Without a catalogue list the downstream tables are still excluded.
    got = {p.name for p in screen_star_files({}, tmp_path)}
    assert "stars_redetect.csv" not in got and "stars_vetted.csv" not in got


def test_metronome_assess_loads_screen_records_through_the_filter(tmp_path, monkeypatch):
    """stage_assess's own loader goes through screen_star_files: a redetect
    row never becomes a screen record.  The run is stopped right after the
    load (the rest of assess is covered by tests/test_metronome.py)."""
    import seti.metronome.run as R

    pd.DataFrame({"star_key": ["kepler:1"], "star_id": ["1"]}) \
        .to_csv(tmp_path / "stars_kepler_yang2019_s0of1.csv", index=False)
    pd.DataFrame({"star_key": ["kepler:REDETECT"], "star_id": ["R"]}) \
        .to_csv(tmp_path / "stars_redetect.csv", index=False)
    seen = {}
    real = R.screen_star_files

    class Stop(Exception):
        pass

    def spy(conf, out):
        seen["files"] = [p.name for p in real(conf, out)]
        return real(conf, out)

    def stop(*_a, **_k):                    # the screen_*.json glob right after the load
        raise Stop()

    monkeypatch.setattr(R, "screen_star_files", spy)
    monkeypatch.setattr(R.glob, "glob", stop)
    try:
        R.stage_assess(R.load_metronome_config(), tmp_path, offline=True)
    except Stop:
        pass
    assert seen["files"] == ["stars_kepler_yang2019_s0of1.csv"]
