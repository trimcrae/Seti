"""OCCULT stage orchestration: ``python -m seti.occult.run <stage> [...]``.

Stages
------
probe      what OGLE / KMTNet / MOA serve to a runner (results/occult/probe.json)
catalog    both surveys' event lists -> search units (catalog.json.gz)
controls   named finite-source and binary-lens events + injections into their
           baselines; the gate everything downstream reads (controls.json)
screen     one shard of units: fetch, fit, gate, analytic sensitivity and
           injection trials (shards/screen_s<i>of<n>.jsonl, inj_s<i>of<n>.jsonl)
assess     all shards -> funnel, empirical null, efficiency map, candidates,
           summary.json
selftest   the offline synthetic battery (no network)
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import detect as D
from . import inject as INJ

OUT = Path("results/occult")
CONFIG = Path("config/occult.yaml")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def load_config(path: Path = CONFIG) -> dict:
    if not path.exists():
        return {}
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return os.environ.get("GITHUB_SHA")


def _stamp() -> dict:
    return {"generated_utc": _now(), "run_id": os.environ.get("GITHUB_RUN_ID"), "git_sha": _git_sha()}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=1))


def read_catalog(out: Path) -> list:
    p = out / "catalog.json.gz"
    if not p.exists():
        return []
    with gzip.open(p, "rt") as fh:
        return json.load(fh)["units"]


def unit_hint(unit: dict) -> dict:
    r = unit.get("kmt") or unit.get("ogle") or {}
    return {"t0": r.get("t0"), "tE": r.get("tE"), "u0": r.get("u0")}


def unit_event(s, unit: dict) -> tuple[D.Event, dict]:
    from .acquire import fetch_unit_series

    got = fetch_unit_series(s, unit)
    ref = unit.get("kmt") or unit.get("ogle") or {}
    ev = D.build_event(unit["unit"], got["series"],
                       meta={"unit": unit["unit"], "ra": ref.get("ra"), "dec": ref.get("dec")})
    return ev, got["status"]


# --------------------------------------------------------------------------------------
# catalog
# --------------------------------------------------------------------------------------

def stage_catalog(out: Path) -> dict:
    from .acquire import build_units, fetch_catalogues, public_seasons, session

    s = session()
    kmt_seasons = public_seasons()
    cat = fetch_catalogues(s, kmt_seasons)
    units = build_units(cat["rows"])
    n_ogle = sum(1 for r in cat["rows"] if r["survey"] == "OGLE")
    n_kmt = sum(1 for r in cat["rows"] if r["survey"] == "KMT")
    joint = sum(1 for u in units if u["kmt"] and u["ogle"])
    no_coord = [r["name"] for r in cat["rows"] if r.get("ra") is None or r.get("dec") is None]
    no_t0 = [r["name"] for r in cat["rows"] if r.get("t0") is None]
    verdict = "OK" if units else "NO_DATA_REACHED"
    meta = {**_stamp(), "verdict": verdict, "kmt_seasons": kmt_seasons,
            "season_status": cat["status"], "n_rows_ogle": n_ogle, "n_rows_kmt": n_kmt,
            "n_units": len(units), "n_units_joint": joint,
            "n_rows_no_coords": len(no_coord), "rows_no_coords_examples": no_coord[:10],
            "n_rows_no_t0": len(no_t0), "rows_no_t0_examples": no_t0[:10],
            "moa": "UNREACHED (probe: alert pages 404 / host unresolvable from a runner)"}
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "catalog.json.gz", "wt") as fh:
        json.dump(_jsonable({"meta": meta, "units": units}), fh)
    write_json(out / "catalog_meta.json", meta)
    return meta


# --------------------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------------------

def find_unit(units: list, name: str) -> dict | None:
    for u in units:
        if u["unit"] == name:
            return u
        if u.get("ogle") and u["ogle"]["name"] == name:
            return u
        if u.get("kmt") and u["kmt"]["name"] == name:
            return u
    return None


def controls_verdict(records: list, gate: dict, n_listed: int) -> dict:
    """The gate: no named control is an occultation; strong injections are recovered.

    False positives count only among the NAMED controls (literature-classified
    finite-source and binary-lens events).  The recovery test pools the
    injections into the finite-source controls and into the ordinary-event
    baselines that have >= 2 sites (a single-site event cannot pass the
    two-site rule by construction, so it would test the rule, not the
    pipeline) and whose Fisher-expected Delta chi^2 exceeds 500.
    """
    named = [r for r in records if r["class"] in ("finite_source_single_lens", "binary_lens")]
    reached = [r for r in named if r.get("reached")]
    frac = len(reached) / n_listed if n_listed else 0.0
    false_pos = [r["name"] for r in reached if r.get("tier") == D.TIER_CANDIDATE]
    hosts = [r for r in records if r.get("reached")
             and r["class"] in ("finite_source_single_lens", "baseline")
             and len([x for x in (r.get("sites") or []) if x != "?"]) >= 2]
    strong = [t for r in hosts for t in r.get("injections", []) if t["expected_dchi2"] > 500]
    rec_frac = (sum(t["recovered"] for t in strong) / len(strong)) if strong else None
    if not reached:
        verdict = "NO_DATA_REACHED"
    elif false_pos:
        verdict = "CONTROLS_FAIL_FALSE_POSITIVE"
    elif rec_frac is None:
        verdict = "CONTROLS_DEGRADED_NO_STRONG_INJECTIONS"
    elif rec_frac < gate.get("min_recovery_strong", 0.7):
        verdict = "CONTROLS_FAIL_LOW_RECOVERY"
    elif frac < gate.get("min_controls_reached_frac", 0.7):
        verdict = "CONTROLS_PASS_DEGRADED_COVERAGE"
    else:
        verdict = "CONTROLS_PASS"
    by_rho: dict = {}
    for t in strong:
        k = str(t.get('rho_l_inj'))
        by_rho.setdefault(k, [0, 0])
        by_rho[k][0] += int(t["recovered"])
        by_rho[k][1] += 1
    return {"verdict": verdict, "n_listed": n_listed, "n_reached": len(reached),
            "reached_frac": frac, "false_positives": false_pos,
            "strong_recovery_by_rho_l": {k: {"k": v[0], "n": v[1]} for k, v in by_rho.items()},
            "n_baseline_hosts": sum(1 for r in hosts if r["class"] == "baseline"),
            "n_strong_injections": len(strong), "strong_recovery_frac": rec_frac,
            "passed": verdict.startswith("CONTROLS_PASS")}


def baseline_hosts(units: list, n: int) -> list:
    """A fixed, reproducible sample of ordinary well-covered events to host injections.

    Joint KMTNet + OGLE units whose catalogue solution has 0 < u0 < 0.5 and
    5 < tE < 120 d, ordered by a hash of the name (no selection on the light
    curve itself), first ``n``.
    """
    pool = []
    for u in units:
        k = u.get("kmt")
        if not (k and u.get("ogle")):
            continue
        u0, te = k.get("u0"), k.get("tE")
        if u0 is None or te is None or not (0 < u0 < 0.5 and 5 < te < 120):
            continue
        pool.append((zlib.crc32(u["unit"].encode()), u))
    pool.sort(key=lambda x: x[0])
    return [u for _, u in pool[:n]]


def stage_controls(out: Path, cfg: dict) -> dict:
    from .acquire import session

    units = read_catalog(out)
    conf = D.conf_with(cfg.get("detect"))
    ctl = cfg.get("controls", {})
    gate = ctl.get("gate", {})
    s = session()
    records = []
    listed = [(cls, item) for cls in ("finite_source_single_lens", "binary_lens")
              for item in ctl.get(cls, [])]
    n_named = len(listed)
    listed += [("baseline", {"name": u["unit"], "ref": "ordinary joint KMT+OGLE event (injection host)"})
               for u in baseline_hosts(units, int(gate.get("n_baselines", 30)))]
    for cls, item in listed:
        name = item["name"]
        rec = {"name": name, "class": cls, "ref": item.get("ref"), "reached": False}
        u = find_unit(units, name)
        if u is None:
            rec["why"] = "not in catalogue"
            records.append(rec)
            continue
        rec["unit"] = u["unit"]
        ev, status = unit_event(s, u)
        rec["download"] = status
        if ev.t.size == 0:
            rec["why"] = "no photometry"
            records.append(rec)
            continue
        rec["reached"] = True
        t = time.time()
        res = D.assess_event(ev, conf, unit_hint(u))
        rec["seconds"] = round(time.time() - t, 1)
        for k in ("tier", "dchi2", "dchi2_anti", "regime", "rejections", "peak_snr", "fspl",
                  "occult", "n_points", "sites", "bands", "redchi2_fspl"):
            rec[k] = res.get(k)
        rec["injections"] = INJ.injection_trials(ev, conf, unit_hint(u),
                                                 gate.get("injection_rho_l", [0.5, 0.7, 0.85]))
        records.append(rec)
        print(f"[controls] {name}: {rec['tier']} dchi2={rec.get('dchi2')} "
              f"inj={[(t['rho_l_inj'], t['recovered']) for t in rec['injections']]}", flush=True)
    v = controls_verdict(records, gate, n_named)
    res = {**_stamp(), **v, "records": records}
    write_json(out / "controls.json", res)
    return res


# --------------------------------------------------------------------------------------
# screen
# --------------------------------------------------------------------------------------

def _done_units(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        try:
            done.add(json.loads(line)["unit"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def save_lc(lc_dir: Path, ev: D.Event) -> str:
    lc_dir.mkdir(parents=True, exist_ok=True)
    p = lc_dir / f"{ev.name}.csv.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("t,f,e,dataset,site,band\n")
        for i in range(ev.t.size):
            ds = ev.datasets[ev.ds[i]]
            fh.write(f"{ev.t[i]:.5f},{ev.f[i]:.6g},{ev.e[i]:.4g},{ds['name']},{ds['site']},{ds['band']}\n")
    return str(p)


SCREEN_KEEP = ("tier", "dchi2", "dchi2_anti", "dbic", "regime", "u_c", "rejections", "peak_snr",
               "lensing_dchi2", "fspl", "occult", "n_points", "n_clipped", "sites", "bands",
               "redchi2_fspl", "alpha", "alpha_by_site", "alpha_by_band", "sites_seeing_step",
               "sites_consistency", "bands_consistency", "symmetry", "jackknife", "steps",
               "n_in_hole", "hole_below_baseline", "positive_bump", "binned_redchi2_after",
               "colour_tested", "scan_best_dchi2", "scan_best_anti_dchi2", "hollow_centre",
               "u_max_observed", "error", "err_scales")


def screen_unit(s, unit: dict, conf: dict, cfg_screen: dict, idx: int, lc_dir: Path) -> tuple[dict, list]:
    ev, status = unit_event(s, unit)
    rec = {"unit": unit["unit"], "download": status, "n_points_fetched": int(ev.t.size),
           "has_kmt": bool(unit.get("kmt")), "has_ogle": bool(unit.get("ogle"))}
    trials = []
    if ev.t.size == 0:
        rec["tier"] = D.TIER_NO_DATA
        return rec, trials
    t = time.time()
    try:
        res = D.assess_event(ev, conf, unit_hint(unit))
    except Exception as exc:  # noqa: BLE001
        res = {"tier": "ERROR", "error": repr(exc)[:300]}
    rec["seconds"] = round(time.time() - t, 2)
    rec.update({k: res.get(k) for k in SCREEN_KEEP if k in res})
    # analytic sensitivity at this event's own solution, and injection trials
    if res.get("fspl") and res.get("tier") not in (D.TIER_NO_DATA, D.TIER_NOT_LENSING, "ERROR"):
        try:
            ev2, e, f0, _ = D.fit_fspl_clean(ev, conf, unit_hint(unit))
            if ev2 is not None:
                rec["expected"] = INJ.expected_map(ev2, f0, e)
        except Exception as exc:  # noqa: BLE001
            rec["expected_error"] = repr(exc)[:200]
        every = int(cfg_screen.get("inject_every", 8))
        if every > 0 and idx % every == 0:
            rls = INJ.pick_rho_ls(zlib.crc32(unit["unit"].encode()),
                                  int(cfg_screen.get("inject_per_unit", 3)))
            trials = INJ.injection_trials(ev, conf, unit_hint(unit), rls)
    if res.get("tier") == D.TIER_CANDIDATE or \
            (res.get("dchi2") or 0) >= float(cfg_screen.get("lc_save_dchi2_any", 1e12)):
        rec["lc_file"] = save_lc(lc_dir, ev)
    return rec, trials


_WORKER: dict = {}


def _worker_init(conf, cfg_screen, lc_dir):
    from .acquire import session

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _WORKER.update(session=session(), conf=conf, cfg_screen=cfg_screen, lc_dir=Path(lc_dir))


def _worker_run(arg):
    i, u = arg
    try:
        return screen_unit(_WORKER["session"], u, _WORKER["conf"], _WORKER["cfg_screen"], i,
                           _WORKER["lc_dir"])
    except Exception as exc:  # noqa: BLE001  (a network or parse failure: recorded, not fatal)
        return {"unit": u["unit"], "tier": "ERROR", "error": repr(exc)[:300]}, []


def stage_screen(out: Path, cfg: dict, shard: int, n_shards: int, budget_s: float,
                 max_units: int = 0, workers: int = 2) -> dict:
    """One shard: units i with i % n_shards == shard, in a process pool.

    Every finished unit is appended to ``screen_s<i>of<n>.jsonl`` at once (a
    killed shard loses minutes); a re-run resumes past the units already in
    that file.  No new unit is started after ``budget_s``.
    """
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    units = read_catalog(out)
    sdir = out / "shards"
    if not units:
        res = {**_stamp(), "verdict": "NO_DATA_REACHED", "why": "no catalogue", "shard": shard}
        write_json(sdir / f"screen_s{shard}of{n_shards}.meta.json", res)
        return res
    units = sorted(units, key=lambda u: u["unit"])
    mine = [(i, u) for i, u in enumerate(units) if i % n_shards == shard]
    if max_units:
        mine = mine[:max_units]
    conf = D.conf_with(cfg.get("detect"))
    cfg_screen = cfg.get("screen", {})
    sdir.mkdir(parents=True, exist_ok=True)
    spath = sdir / f"screen_s{shard}of{n_shards}.jsonl"
    ipath = sdir / f"inj_s{shard}of{n_shards}.jsonl"
    lc_dir = sdir / f"lc_s{shard}of{n_shards}"
    done = _done_units(spath)
    todo = [(i, u) for i, u in mine if u["unit"] not in done]
    t_start = time.time()
    n_new = 0
    stopped_for_budget = False
    nw = max(1, int(workers))
    if nw == 1:                       # inline: the offline tests and a debugging run
        _worker_init(conf, cfg_screen, str(lc_dir))
        with spath.open("a") as fs, ipath.open("a") as fi:
            for arg in todo:
                if time.time() - t_start >= budget_s:
                    stopped_for_budget = True
                    break
                rec, trials = _worker_run(arg)
                fs.write(json.dumps(_jsonable(rec)) + "\n")
                for tr in trials:
                    fi.write(json.dumps(_jsonable(tr)) + "\n")
                n_new += 1
        todo = []
    with ProcessPoolExecutor(max_workers=nw, initializer=_worker_init,
                             initargs=(conf, cfg_screen, str(lc_dir))) as pool, \
            spath.open("a") as fs, ipath.open("a") as fi:
        it = iter(todo)
        pending = set()
        for _ in range(nw * 2):
            nxt = next(it, None)
            if nxt is None:
                break
            pending.add(pool.submit(_worker_run, nxt))
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                rec, trials = fut.result()
                fs.write(json.dumps(_jsonable(rec)) + "\n")
                fs.flush()
                for tr in trials:
                    fi.write(json.dumps(_jsonable(tr)) + "\n")
                fi.flush()
                n_new += 1
                if n_new % 25 == 0:
                    print(f"[screen s{shard}] {n_new}/{len(todo)} in {time.time() - t_start:.0f}s",
                          flush=True)
                if time.time() - t_start < budget_s:
                    nxt = next(it, None)
                    if nxt is not None:
                        pending.add(pool.submit(_worker_run, nxt))
                else:
                    stopped_for_budget = True
    done_after = _done_units(spath)
    res = {**_stamp(), "shard": shard, "n_shards": n_shards, "n_assigned": len(mine),
           "n_done": len(done_after & {u["unit"] for _, u in mine}), "n_new": n_new,
           "stopped_for_budget": stopped_for_budget, "elapsed_s": round(time.time() - t_start, 1)}
    write_json(sdir / f"screen_s{shard}of{n_shards}.meta.json", res)
    return res


# --------------------------------------------------------------------------------------
# assess
# --------------------------------------------------------------------------------------

def _read_jsonl(paths) -> list:
    rows, seen = [], set()
    for p in paths:
        for line in Path(p).read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r.get("unit") or r.get("event"), r.get("rho_l_inj"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    return rows


def empirical_threshold(recs: list, floor: float) -> dict:
    """The Delta chi^2 bar from the anti-occultation null over the same events.

    The anti template has the same number of parameters and the same
    sensitivity to every systematic, and no physics produces it.  The bar is
    the largest anti Delta chi^2 among events whose own fits did not fail
    (a 1/N false-alarm level), never below ``floor``.
    """
    anti = np.array([r.get("dchi2_anti") or 0.0 for r in recs
                     if r.get("tier") not in (D.TIER_NO_DATA, D.TIER_NOT_LENSING, "ERROR", None)])
    if anti.size == 0:
        return {"threshold": floor, "n_null": 0}
    q = {f"q{int(p * 1000)}": float(np.quantile(anti, p)) for p in (0.5, 0.9, 0.99, 0.999)}
    return {"threshold": float(max(floor, anti.max())), "n_null": int(anti.size),
            "anti_max": float(anti.max()), **q}


def stage_assess(out: Path, cfg: dict, n_shards: int) -> dict:
    sdir = out / "shards"
    spaths = sorted(sdir.glob(f"screen_s*of{n_shards}.jsonl"))
    ipaths = sorted(sdir.glob(f"inj_s*of{n_shards}.jsonl"))
    recs = _read_jsonl(spaths)
    trials = _read_jsonl(ipaths)
    units = read_catalog(out)
    controls = json.loads((out / "controls.json").read_text()) if (out / "controls.json").exists() else {}
    conf = D.conf_with(cfg.get("detect"))
    tiers: dict = {}
    for r in recs:
        tiers[r.get("tier")] = tiers.get(r.get("tier"), 0) + 1
    thr = empirical_threshold(recs, conf["dchi2_floor"])
    rej_counts: dict = {}
    for r in recs:
        for x in r.get("rejections") or []:
            rej_counts[x] = rej_counts.get(x, 0) + 1
    cands = [r for r in recs if r.get("tier") == D.TIER_CANDIDATE]
    above = [r for r in cands if (r.get("dchi2") or 0) >= thr["threshold"]]
    below = [r for r in cands if (r.get("dchi2") or 0) < thr["threshold"]]
    eff = INJ.efficiency_table(trials)
    eff_exp = INJ.efficiency_vs_expected(trials)
    # analytic map over every assessed event
    grid_keys = [f"{x:g}" for x in INJ.RHO_L_GRID]
    exp_frac = {}
    for k in grid_keys:
        vals = [r["expected"][k] for r in recs if r.get("expected") and k in r["expected"]]
        exp_frac[k] = {"n": len(vals),
                       "frac_expected_gt_threshold": (sum(v > thr["threshold"] for v in vals) / len(vals))
                       if vals else None}
    screened = [r for r in recs if r.get("tier") not in (D.TIER_NO_DATA, None)]
    n_units = len(units)
    funnel = {
        "units_in_catalogue": n_units,
        "units_screened_records": len(recs),
        "units_with_photometry": len(screened),
        "not_lensing_or_too_faint": tiers.get(D.TIER_NOT_LENSING, 0),
        "errors": tiers.get("ERROR", 0),
        "no_occultation": tiers.get(D.TIER_NO_OCC, 0),
        "occultation_rejected_by_gates": tiers.get(D.TIER_REJECTED, 0),
        "gate_passing_below_null_threshold": len(below),
        "gate_passing_above_null_threshold": len(above),
    }
    consistency = []
    if sum(tiers.values()) != len(recs):
        consistency.append("tier counts do not sum to records")
    if len(recs) > n_units and n_units:
        consistency.append("more records than catalogue units")
    ctl_ok = bool(controls.get("passed"))
    if not recs:
        verdict = "NO_DATA_REACHED"
    elif not ctl_ok:
        verdict = f"CONTROLS_NOT_PASSED({controls.get('verdict')}) -- nothing downstream is believed"
    elif above:
        verdict = f"CANDIDATES_TO_VET:{len(above)}"
    else:
        verdict = "NO_SURVIVOR"
    summary = {
        **_stamp(), "verdict": verdict, "controls_verdict": controls.get("verdict"),
        "funnel": funnel, "tiers": tiers, "rejection_counts": rej_counts,
        "null_threshold": thr, "coverage": {
            "units_screened_of_catalogue": f"{len(recs)}_of_{n_units}",
            "with_kmt": sum(1 for r in screened if r.get("has_kmt")),
            "with_ogle": sum(1 for r in screened if r.get("has_ogle")),
            "with_both": sum(1 for r in screened if r.get("has_kmt") and r.get("has_ogle")),
            "colour_tested_candidates": sum(1 for r in cands if r.get("colour_tested")),
            "moa": "unreached"},
        "sensitivity": {"injection": eff, "injection_vs_expected": eff_exp,
                        "analytic_frac_events_above_threshold": exp_frac},
        "candidates": [{k: r.get(k) for k in ("unit", "dchi2", "dchi2_anti", "regime", "u_c",
                                              "occult", "alpha", "alpha_by_site", "alpha_by_band",
                                              "sites_seeing_step", "jackknife", "steps",
                                              "symmetry", "lc_file")} for r in above],
        "below_threshold_gate_passing": [{"unit": r["unit"], "dchi2": r.get("dchi2"),
                                          "rho_l": (r.get("occult") or {}).get("rho_l")}
                                         for r in below][:200],
        "self_consistency": {"ok": not consistency, "problems": consistency},
        "n_shard_files": len(spaths),
    }
    # candidate photometry travels out of the shard folders into lc/
    import shutil

    for r in above:
        src = Path(r.get("lc_file") or "")
        if src.name and src.exists():
            (out / "lc").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / "lc" / src.name)
            r["lc_file"] = str(out / "lc" / src.name)
    write_json(out / "summary.json", summary)
    write_json(out / "candidates.json", {"generated_utc": summary["generated_utc"],
                                         "threshold": thr["threshold"], "candidates": above,
                                         "gate_passing_below_threshold": below})
    # compact per-unit table and every injection trial (the full shard files are artifacts)
    compact_keys = ("unit", "tier", "dchi2", "dchi2_anti", "regime", "u_c", "rejections",
                    "peak_snr", "n_points", "sites", "bands", "has_kmt", "has_ogle", "seconds",
                    "error", "expected")
    with gzip.open(out / "screen_compact.jsonl.gz", "wt") as fh:
        for r in recs:
            row = {k: r.get(k) for k in compact_keys if k in r}
            f = r.get("fspl") or {}
            row.update({"t0": f.get("t0"), "tE": f.get("tE"), "u0": f.get("u0"),
                        "rho_star": f.get("rho_star"),
                        "rho_l": (r.get("occult") or {}).get("rho_l")})
            fh.write(json.dumps(_jsonable(row)) + "\n")
    with gzip.open(out / "injections.jsonl.gz", "wt") as fh:
        for t in trials:
            fh.write(json.dumps(_jsonable(t)) + "\n")
    return summary


# --------------------------------------------------------------------------------------
# selftest (offline)
# --------------------------------------------------------------------------------------

def stage_selftest(out: Path) -> dict:
    rows = []
    for name, kw in (("fspl", {"rho_l": None}), ("occ0.6", {"rho_l": 0.6}), ("occ0.5", {"rho_l": 0.5}),
                     ("hole1.3", {"rho_l": 1.3})):
        rec = D.assess_event(D.synth_event(seed=11, **kw))
        rows.append({"case": name, "tier": rec["tier"], "dchi2": rec.get("dchi2"),
                     "rho_l": (rec.get("occult") or {}).get("rho_l"), "rejections": rec["rejections"]})
    res = {**_stamp(), "rows": rows}
    write_json(out / "selftest.json", res)
    return res


# --------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="seti occult")
    ap.add_argument("stage", choices=["probe", "catalog", "controls", "screen", "assess", "selftest"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--budget-s", type=float, default=4.5 * 3600)
    ap.add_argument("--max-units", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    out = Path(a.out)
    cfg = load_config(Path(a.config))
    if a.stage == "probe":
        from .probe import run_probe

        s = run_probe(out)
        print(json.dumps({k: v for k, v in s.items() if k != "records"}, indent=1))
        return 0
    if a.stage == "catalog":
        m = stage_catalog(out)
        print(json.dumps(_jsonable(m), indent=1))
        return 0 if m["verdict"] == "OK" else 1
    if a.stage == "controls":
        r = stage_controls(out, cfg)
        print(json.dumps({k: v for k, v in _jsonable(r).items() if k != "records"}, indent=1))
        return 0
    if a.stage == "screen":
        r = stage_screen(out, cfg, a.shard, a.n_shards, a.budget_s, a.max_units, a.workers)
        print(json.dumps(_jsonable(r), indent=1))
        return 0
    if a.stage == "assess":
        r = stage_assess(out, cfg, a.n_shards)
        print(json.dumps({k: v for k, v in _jsonable(r).items()
                          if k in ("verdict", "funnel", "null_threshold", "controls_verdict")}, indent=1))
        return 0
    if a.stage == "selftest":
        print(json.dumps(_jsonable(stage_selftest(out)), indent=1))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
