"""Run 35862579322: every CENTURY sweep shard died on its first star.

Root cause: DR7 serves some light-curve rows with an EMPTY mosaic/exposure
number.  Under pandas 3 (what the runner installs) ``Series.astype(str)`` keeps
a missing value as NaN instead of writing ``"nan"``, so building the plate
label with ``"_".join`` raised ``TypeError: sequence item 3: expected str
instance, float found``.  The exception escaped ``stage_acquire`` before it
wrote ``acquire_summary.json``, so ``assess`` saw no acquire reports and said
NO_SHARDS_PRESENT --- which pointed at the workflow, not at the exception.

These tests pin the three fixes.  Offline; run them under pandas 3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from seti.century.api import ApiResponse, to_frame  # noqa: E402
from seti.century.lightcurve import attach_exptime, from_api_frame, str_values  # noqa: E402
from seti.century.run import stage_assess  # noqa: E402
from seti.century.targets import exposure_table  # noqa: E402
from test_century import _conf, _dr7_lightcurve_lines, synth_plates  # noqa: E402


def _blanked_lines(rng, exptimes):
    truth = synth_plates(rng, period=0.57, amp=0.5, n_per_year=10)
    lines = _dr7_lightcurve_lines(truth, exptimes)
    hdr = lines[0].split(",")
    i_mos, i_exp, i_ser = (hdr.index("mosaic_number"), hdr.index("exposure_number"),
                           hdr.index("series"))
    for k in (1, 5, 9, 40):
        f = lines[k].split(",")
        f[i_mos] = ""
        f[i_exp] = ""
        lines[k] = ",".join(f)
    f = lines[12].split(",")
    f[i_ser] = ""
    lines[12] = ",".join(f)
    return lines


def test_blank_plate_identifiers_do_not_crash_the_parser():
    exptimes: dict[str, float] = {}
    df = to_frame(_blanked_lines(np.random.default_rng(5), exptimes))
    assert df["mosaic_number"].isna().sum() >= 4          # the condition that crashed
    lc = from_api_frame(df)                               # must not raise
    assert lc is not None and lc.n_det + lc.n_nd == len(df)
    assert all(isinstance(p, str) and "nan" not in p for p in np.r_[lc.plate, lc.plate_nd])
    # The exposure join still places every plate it can (blank series in
    # queryexps too).
    exps = ["series,platenum,scannum,mosnum,expnum,solnum,exptime,epoch,limMagApass"]
    for key, et in exptimes.items():
        ser, pnum = key.split(":")
        exps.append(f"{ser},{pnum},0,0,0,1,{et:.1f},1900.0,14.5")
    exps.append(",999999,0,,,1,30.0,1900.0,14.5")
    prov = attach_exptime(lc, exposure_table(to_frame(exps)))
    assert prov["matched_det"] + prov["matched_nd"] >= len(df) - 6


def test_str_values_is_version_proof():
    s = pd.Series([1.0, np.nan, None, "ai", 7], dtype=object)
    assert list(str_values(s)) == ["1", "", "", "ai", "7"]
    assert list(str_values(pd.Series(["a", None], dtype="string"))) == ["a", ""]
    assert list(str_values(pd.Series([3.0, np.nan]))) == ["3", ""]


def _resp(frame):
    return ApiResponse(endpoint="x", status=200, ok=True, elapsed_s=0.0, payload={},
                       n_rows=len(frame), columns=list(frame.columns), frame=frame)


def test_one_unparseable_light_curve_costs_one_star_not_the_shard(tmp_path, monkeypatch):
    import seti.century.run as R

    targets = pd.DataFrame([{"target_id": i, "name": f"s{i}", "ra": 10.0 + i, "dec": 20.0,
                             "kind": "bright", "vtype": "", "period_cat": np.nan,
                             "mag_cat": 11.0, "amp_cat": 0.0, "source": "apass", "field": "f",
                             "gsc_bin_index": 1, "ref_number": i} for i in range(3)])
    targets.to_csv(tmp_path / "targets.csv", index=False)
    pd.DataFrame(columns=["series", "platenum", "mosnum", "expnum", "exptime_min"]).to_csv(
        tmp_path / "plate_exptime.csv", index=False)
    good = to_frame(_dr7_lightcurve_lines(synth_plates(np.random.default_rng(9),
                                                       n_per_year=8), {}))
    real = R.from_api_frame
    calls = {"n": 0}

    def flaky(frame, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TypeError("sequence item 3: expected str instance, float found")
        return real(frame, *a, **k)

    monkeypatch.setattr(R, "querycat", lambda *a, **k: _resp(good.iloc[:1]))
    monkeypatch.setattr(R, "lightcurve", lambda *a, **k: _resp(good))
    monkeypatch.setattr(R, "from_api_frame", flaky)
    monkeypatch.setattr(R, "_runner_flagdefs",
                        lambda conf, log: (R.FlagDefs("aflags"), R.FlagDefs("bflags"), "none"))
    monkeypatch.setattr(R, "_shard_exposure_table", lambda *a, **k: (pd.DataFrame(), "none"))
    rep = R.stage_acquire(_conf(), tmp_path, (0, 1), pause_s=0.0)
    assert rep["n_parse_failed"] == 1 and rep["n_fetched"] == 2, rep
    sd = next((tmp_path / "shards").iterdir())
    assert (sd / "acquire_summary.json").exists()
    st = [json.loads(x) for x in (sd / "acquire.jsonl").read_text().splitlines()]
    assert [r["status"] for r in st].count("lightcurve_parse_failed") == 1


def test_shards_that_crashed_in_acquire_are_not_reported_as_absent(tmp_path):
    pd.DataFrame([{"target_id": 0, "name": "s0", "ra": 10.0, "dec": 20.0, "kind": "pulsator",
                   "field": "allsky_pulsators"}]).to_csv(tmp_path / "targets.csv", index=False)
    (tmp_path / "targets_summary.json").write_text(json.dumps({"n_targets": 1, "fields": {}}))
    sd = tmp_path / "shards" / "0_of_1"
    sd.mkdir(parents=True)
    (sd / "screen_summary.json").write_text(json.dumps({"stage": "screen", "n_lightcurves": 0}))
    s = stage_assess(_conf(), tmp_path, confirm=False, gaia=False)
    assert s["verdict_code"] == "ACQUIRE_CRASHED", s["verdict_code"]


def test_century_commits_go_through_the_verified_retrying_script():
    """Run 35992721863 finished its whole sweep and lost the commit to two
    GitHub 500s on push, with one inline retry.  Every results commit now goes
    through scripts/commit_results.sh (retries, remote-ref verification)."""
    import yaml

    wf_path = Path(__file__).resolve().parents[1] / ".github/workflows/century.yml"
    wf = yaml.safe_load(wf_path.read_text())
    for job in ("assess", "assess-only"):
        steps = [s for s in wf["jobs"][job]["steps"] if s.get("name") == "Commit results"]
        assert steps, job
        run = steps[0]["run"]
        assert "scripts/commit_results.sh" in run and "git push" not in run, job
        assert int(steps[0]["env"]["RESULTS_PUSH_ATTEMPTS"]) >= 5
