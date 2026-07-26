"""Staged HERDSMAN pipeline: fetch / scan / reduce as separate runner jobs.

The monolithic ``herdsman_run`` (run.py) does everything in one process — fine
for tests and small samples, but on a CI runner a single timeout loses hours of
work.  The staged pipeline makes the unit of loss one scan (~minutes):

* **fetch** — acquire + preprocess once, save the detection table as parquet
  (an artifact shared by every scan job) with a sidecar meta JSON;
* **scan** — one job per shard: ``mode=real`` runs both time directions on the
  real velocities; ``mode=mock`` runs a disjoint slice of the mock indices.
  Every completed scan is written to its own JSON *immediately*, so a killed
  shard still delivers its finished mocks (the workflow uploads shard output
  with ``if: always()``);
* **reduce** — aggregates whatever scan files exist (tolerant of lost shards),
  vets the real candidates, computes global p-values against the assembled
  mock null, and writes the same outputs the monolith produces.

Mock seeding uses the *global* mock index, so shard decomposition never changes
the realizations — 12 shards x 4 mocks equals one job running 48.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..galactic.orbits import heliocentric_to_galactocentric
from .convergence import ConvergenceParams, detect_convergences
from .mocks import global_p_value, shuffle_velocities
from .run import _candidates_csv, _jsonable, _write_report, build_detection_table
from .vet import vet_candidate

MOCK_SEED = 20260725


def _gc_state(df: pd.DataFrame):
    pos_kpc, vel = heliocentric_to_galactocentric(
        df["X_pc"].to_numpy(float), df["Y_pc"].to_numpy(float),
        df["Z_pc"].to_numpy(float), df["U_kms"].to_numpy(float),
        df["V_kms"].to_numpy(float), df["W_kms"].to_numpy(float))
    return pos_kpc, vel, df["sigv_kms"].to_numpy(float)


def fetch_stage(cfg: Config | None = None, table: pd.DataFrame | None = None,
                **kwargs) -> Path:
    """Stage 1: build the detection table and save it for the scan fleet."""
    cfg = cfg or load_config()
    out_dir = cfg.root / "results" / "herdsman"
    out_dir.mkdir(parents=True, exist_ok=True)
    df, meta = build_detection_table(table=table, **kwargs)
    path = out_dir / "sample.parquet"
    df.to_parquet(path, index=False)
    (out_dir / "sample_meta.json").write_text(json.dumps(_jsonable(meta), indent=2))
    print(f"[herdsman] fetch stage: {meta['final']} stars -> {path}")
    return path


def scan_stage(cfg: Config | None = None, mode: str = "real", shard: int = 0,
               mocks_per_shard: int = 4, mock_cell_pc: float = 40.0,
               params: ConvergenceParams | None = None,
               sample_path: Path | None = None) -> Path:
    """Stage 2: one scan job.  Writes each completed scan to disk immediately."""
    cfg = cfg or load_config()
    out_dir = cfg.root / "results" / "herdsman"
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    p = params or ConvergenceParams()
    df = pd.read_parquet(sample_path or out_dir / "sample.parquet")
    pos_kpc, vel, sigv = _gc_state(df)

    if mode == "real":
        for tag, direction in (("forward", +1), ("backward", -1)):
            res = detect_convergences(pos_kpc, vel, sigv, direction, p)
            out = shards_dir / f"real_{tag}.json"
            out.write_text(json.dumps(_jsonable(res), indent=2))   # checkpoint
            print(f"[herdsman] real {tag}: horizon {res['t_horizon_myr']:.1f} "
                  f"Myr, {len(res['candidates'])} candidates -> {out.name}")
        return shards_dir
    if mode != "mock":
        raise ValueError(f"unknown scan mode {mode!r}")

    pos_pc = np.asarray(pos_kpc, float) * 1000.0
    for j in range(mocks_per_shard):
        k = shard * mocks_per_shard + j          # global mock index
        for tag, direction in (("forward", +1), ("backward", -1)):
            rng = np.random.default_rng(
                MOCK_SEED + 1000 * k + (0 if direction > 0 else 1))
            vel_s, sig_s = shuffle_velocities(pos_pc, vel, sigv,
                                              mock_cell_pc, rng)
            res = detect_convergences(pos_kpc, vel_s, sig_s, direction, p,
                                      rng=np.random.default_rng(MOCK_SEED + k))
            best = max((c["surprise"] for c in res["candidates"]), default=0.0)
            out = shards_dir / f"mock_{k:03d}_{tag}.json"
            out.write_text(json.dumps({                         # checkpoint
                "mock_index": k, "direction": tag,
                "n_candidates": len(res["candidates"]),
                "max_surprise": float(best),
                "n_raw": res["n_raw_detections"],
                "t_horizon_myr": res["t_horizon_myr"]}, indent=2))
            print(f"[herdsman] mock {k} {tag}: max surprise {best:.2f} "
                  f"-> {out.name}")
    return shards_dir


def reduce_stage(cfg: Config | None = None, n_mocks_expected: int | None = None,
                 astro_floor_kms: float = 0.3, vet_top: int = 50) -> dict:
    """Stage 3: aggregate whatever shards exist; vet; write final outputs.

    Vetting (chemistry + rendezvous Monte Carlo) runs only on the ``vet_top``
    highest-surprise candidates per direction — the first reduce attempt
    timed out MC-vetting an unbounded list, and candidates below the mock
    null's best surprise carry no detection weight anyway.  The rest are kept
    in the output unvetted (``vetted: false``) with all detector statistics.
    """
    cfg = cfg or load_config()
    out_dir = cfg.root / "results" / "herdsman"
    shards_dir = out_dir / "shards"
    df = pd.read_parquet(out_dir / "sample.parquet")
    meta = json.loads((out_dir / "sample_meta.json").read_text())

    results, mock_stats = {}, {}
    for tag, direction in (("forward", +1), ("backward", -1)):
        real_path = shards_dir / f"real_{tag}.json"
        if not real_path.exists():
            print(f"[herdsman] WARNING: missing real_{tag}.json — "
                  "real scan shard was lost; rerun scan-real")
            continue
        res = json.loads(real_path.read_text())
        p = ConvergenceParams(**{k: v for k, v in res["params"].items()})
        ordered = sorted(res["candidates"], key=lambda c: -c["surprise"])
        vetted = []
        for i, c in enumerate(ordered):
            if i < vet_top:
                v = vet_candidate(c, df, direction, dt_myr=p.dt_myr,
                                  astro_floor_kms=astro_floor_kms)
                v["vetted"] = True
            else:
                v = dict(c)
                v["vetted"] = False
            vetted.append(v)
            if (i + 1) % 10 == 0 and i < vet_top:
                print(f"[herdsman] {tag}: vetted {i + 1}/"
                      f"{min(vet_top, len(ordered))} candidates")
        res["candidates"] = vetted
        res["n_unvetted"] = max(0, len(ordered) - vet_top)
        if res["n_unvetted"]:
            print(f"[herdsman] {tag}: {res['n_unvetted']} low-surprise "
                  f"candidates kept unvetted (vet_top={vet_top})")
        results[tag] = res

        mocks = sorted(shards_dir.glob(f"mock_*_{tag}.json"))
        per_mock = [json.loads(m.read_text()) for m in mocks]
        maxes = [m["max_surprise"] for m in per_mock]
        best = max((c["surprise"] for c in vetted), default=0.0)
        stat = {"n_mocks": len(per_mock), "per_mock": per_mock,
                "max_surprise_dist": maxes,
                "observed_best_surprise": float(best)}
        stat["p_global"] = global_p_value(best, stat) if per_mock else None
        if n_mocks_expected and len(per_mock) < n_mocks_expected:
            stat["warning"] = (f"only {len(per_mock)}/{n_mocks_expected} mock "
                               f"scans present ({tag}); lost shards should be "
                               "rerun before quoting p_global")
            print(f"[herdsman] WARNING: {stat['warning']}")
        mock_stats[tag] = stat

    for tag in results:
        (out_dir / f"candidates_{tag}.json").write_text(
            json.dumps(_jsonable(results[tag]), indent=2))
        _candidates_csv(results[tag]["candidates"],
                        out_dir / f"candidates_{tag}.csv")
    (out_dir / "mocks.json").write_text(json.dumps(_jsonable(mock_stats), indent=2))

    summary = {
        "sample": meta,
        "params": results[next(iter(results))]["params"] if results else {},
        "directions": {
            tag: {"t_horizon_myr": results[tag]["t_horizon_myr"],
                  "epochs_scanned": results[tag]["epochs_scanned"],
                  "n_raw_detections": results[tag]["n_raw_detections"],
                  "n_candidates": len(results[tag]["candidates"]),
                  "best_surprise": max((c["surprise"] for c in
                                        results[tag]["candidates"]), default=0.0),
                  "n_mocks": mock_stats.get(tag, {}).get("n_mocks", 0),
                  "p_global": mock_stats.get(tag, {}).get("p_global")}
            for tag in results},
        "staged": True,
    }
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    if results:
        _write_report(out_dir, summary, results, mock_stats)
    print("[herdsman] reduce:", json.dumps(_jsonable(summary.get("directions", {}))))
    return summary


__all__ = ["build_detection_table", "fetch_stage", "scan_stage", "reduce_stage",
           "MOCK_SEED"]
