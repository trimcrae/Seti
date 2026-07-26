"""Runner: the HERDSMAN convergence search end-to-end.

Stages (all parameters surfaced through the CLI/workflow):

1. acquire the Gaia DR3 6D quality sample (or accept an offline table in tests),
   apply the RV zero-point correction, build 6D phase space and per-star scalar
   velocity errors, and cut to the precision subset that actually carries
   sensitivity;
2. collapse resolved co-moving pairs (<0.5 pc, dv < 3 km/s) to their brighter
   member so binaries cannot chain into fake meetings;
3. run the convergence detector forward (+) and backward (-) in time — the
   backward scan is both a science channel (heterogeneous past rendezvous) and
   the matched time-reversal control for the forward one;
4. calibrate chance with velocity-shuffled mocks per direction;
5. vet every candidate (chemistry heterogeneity + rendezvous Monte Carlo);
6. write ``results/herdsman/``: summary.json, candidates_{fwd,bwd}.{json,csv},
   mocks.json, REPORT.md.

Never a bare count: every number that matters (horizon, densities, mock nulls)
is written next to the candidates so the result is interpretable standalone.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..config import Config, load_config
from ..galactic.orbits import heliocentric_to_galactocentric
from ..panspermia.kinematics import phase_space_6d
from .acquire import apply_rv_zero_point, fetch_sample, scalar_velocity_error
from .convergence import ConvergenceParams, detect_convergences
from .mocks import global_p_value, run_mocks
from .vet import vet_candidate


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    return x


def _collapse_comoving_pairs(df: pd.DataFrame, sep_pc: float = 0.5,
                             dv_kms: float = 3.0) -> pd.DataFrame:
    """Drop the fainter member of resolved co-moving (binary) pairs."""
    pos = df[["X_pc", "Y_pc", "Z_pc"]].to_numpy(float)
    vel = df[["U_kms", "V_kms", "W_kms"]].to_numpy(float)
    pairs = cKDTree(pos).query_pairs(sep_pc, output_type="ndarray")
    if len(pairs) == 0:
        return df
    dv = np.sqrt(((vel[pairs[:, 0]] - vel[pairs[:, 1]]) ** 2).sum(-1))
    g = pd.to_numeric(df.get("phot_g_mean_mag"), errors="coerce")\
        .fillna(99.0).to_numpy(float)
    drop = set()
    for (i, j), d in zip(pairs, dv):
        if d < dv_kms:
            drop.add(int(i) if g[i] > g[j] else int(j))
    if drop:
        print(f"[herdsman] collapsed {len(drop)} co-moving pair members")
        df = df.drop(df.index[sorted(drop)]).reset_index(drop=True)
    return df


def _candidates_csv(cands: list[dict], path) -> None:
    rows = []
    for c in cands:
        rows.append({
            "t_myr": c["t_myr"], "m": c["m"], "m_eff": c["m_eff"],
            "surprise": round(c["surprise"], 2),
            "lambda": c["lambda"], "r_ball_pc": round(c["r_ball_pc"], 2),
            "focus": round(c["focus"], 2),
            "rms_now_pc": round(c["rms_now_pc"], 2),
            "rms_meet_pc": round(c["rms_meet_pc"], 2),
            "med_now_pc": round(c["med_now_pc"], 2),
            "sig_v_internal_kms": round(c["sig_v_internal_kms"], 2),
            "n_epochs_seen": c["n_epochs_seen"],
            "mh_mad_dex": c.get("chemistry", {}).get("mh_mad_dex"),
            "heterogeneous": c.get("chemistry", {}).get("heterogeneous"),
            "co_natal_possible": c.get("chemistry", {}).get("co_natal_possible"),
            "rms_min_pc_p50": c.get("rendezvous_mc", {}).get("rms_min_pc_p50"),
            "p_rms_lt_5pc": c.get("rendezvous_mc", {}).get("p_rms_lt_5pc"),
            "member_source_ids": ";".join(str(s) for s in
                                          c.get("member_source_ids", [])),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def build_detection_table(d_max_pc: float = 300.0, g_max: float = 14.5,
                          rv_err_max_kms: float = 1.5, sigv_max_kms: float = 0.8,
                          astro_floor_kms: float = 0.3,
                          table: pd.DataFrame | None = None
                          ) -> tuple[pd.DataFrame, dict]:
    """Fetch (unless ``table`` given) and preprocess to the detection table.

    Shared by the monolithic runner below and the staged pipeline
    (``stages.py``) so the two paths can never drift apart.
    """
    raw = table if table is not None else fetch_sample(
        d_max_pc=d_max_pc, rv_err_max_kms=rv_err_max_kms, g_max=g_max)
    n_raw = len(raw)
    raw = apply_rv_zero_point(raw)
    df = phase_space_6d(raw)
    good = np.isfinite(df[["U_kms", "V_kms", "W_kms", "X_pc"]].to_numpy(float))\
        .all(axis=1)
    df = df[good].reset_index(drop=True)
    df["sigv_kms"] = scalar_velocity_error(df, astro_floor_kms=astro_floor_kms)
    n_6d = len(df)
    df = df[df["sigv_kms"] <= sigv_max_kms].reset_index(drop=True)
    n_precise = len(df)
    df = _collapse_comoving_pairs(df)
    meta = {"fetched": n_raw, "with_6d": n_6d, "precise": n_precise,
            "final": int(len(df)), "d_max_pc": d_max_pc, "g_max": g_max,
            "rv_err_max_kms": rv_err_max_kms, "sigv_max_kms": sigv_max_kms,
            "astro_floor_kms": astro_floor_kms,
            "sigv_median_kms": float(df["sigv_kms"].median()) if len(df) else None}
    return df, meta


def herdsman_run(cfg: Config | None = None, d_max_pc: float = 300.0,
                 g_max: float = 14.5, rv_err_max_kms: float = 1.5,
                 sigv_max_kms: float = 0.8, astro_floor_kms: float = 0.3,
                 t_max_myr: float = 20.0, dt_myr: float = 0.25,
                 rec_every: int = 2, r0_pc: float = 1.0, kappa: float = 1.0,
                 lambda_cap: float = 0.5, n_min: int = 4,
                 r_now_min_pc: float = 20.0, focus_min: float = 3.0,
                 surprise_min: float = 3.0, n_mocks: int = 24,
                 mock_cell_pc: float = 40.0,
                 table: pd.DataFrame | None = None) -> dict:
    """Run the full search; ``table`` may be injected for offline tests."""
    cfg = cfg or load_config()
    out_dir = cfg.root / "results" / "herdsman"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df, meta = build_detection_table(
        d_max_pc=d_max_pc, g_max=g_max, rv_err_max_kms=rv_err_max_kms,
        sigv_max_kms=sigv_max_kms, astro_floor_kms=astro_floor_kms, table=table)
    n_raw, n_6d, n_precise = meta["fetched"], meta["with_6d"], meta["precise"]
    print(f"[herdsman] sample: {n_raw} fetched -> {n_6d} with 6D -> "
          f"{n_precise} with sigma_v <= {sigv_max_kms} km/s -> "
          f"{len(df)} after pair collapse ({time.time() - t0:.0f}s)")

    pos_kpc, vel = heliocentric_to_galactocentric(
        df["X_pc"].to_numpy(float), df["Y_pc"].to_numpy(float),
        df["Z_pc"].to_numpy(float), df["U_kms"].to_numpy(float),
        df["V_kms"].to_numpy(float), df["W_kms"].to_numpy(float))
    sigv = df["sigv_kms"].to_numpy(float)

    params = ConvergenceParams(
        t_max_myr=t_max_myr, dt_myr=dt_myr, rec_every=rec_every, r0_pc=r0_pc,
        kappa=kappa, lambda_cap=lambda_cap, n_min=n_min,
        r_now_min_pc=r_now_min_pc, focus_min=focus_min,
        surprise_min=surprise_min)

    results = {}
    mock_stats = {}
    for tag, direction in (("forward", +1), ("backward", -1)):
        print(f"[herdsman] === detect {tag} (direction {direction:+d}) ===")
        res = detect_convergences(pos_kpc, vel, sigv, direction, params)
        print(f"[herdsman] {tag}: horizon {res['t_horizon_myr']:.1f} Myr, "
              f"{res['epochs_scanned']} epochs, {res['n_raw_detections']} raw, "
              f"{len(res['candidates'])} deduped candidates")
        vetted = [vet_candidate(c, df, direction, dt_myr=dt_myr,
                                astro_floor_kms=astro_floor_kms)
                  for c in res["candidates"]]
        res["candidates"] = vetted
        results[tag] = res
        if n_mocks > 0:
            mock_stats[tag] = run_mocks(pos_kpc, vel, sigv, direction, params,
                                        n_mocks, cell_pc=mock_cell_pc)
            best = max((c["surprise"] for c in vetted), default=0.0)
            mock_stats[tag]["observed_best_surprise"] = float(best)
            mock_stats[tag]["p_global"] = global_p_value(best, mock_stats[tag])

    for tag in results:
        cands = results[tag]["candidates"]
        (out_dir / f"candidates_{tag}.json").write_text(
            json.dumps(_jsonable(results[tag]), indent=2))
        _candidates_csv(cands, out_dir / f"candidates_{tag}.csv")
    if mock_stats:
        (out_dir / "mocks.json").write_text(json.dumps(_jsonable(mock_stats),
                                                       indent=2))

    summary = {
        "sample": {"fetched": n_raw, "with_6d": n_6d,
                   "precise": n_precise, "final": int(len(df)),
                   "d_max_pc": d_max_pc, "g_max": g_max,
                   "rv_err_max_kms": rv_err_max_kms,
                   "sigv_max_kms": sigv_max_kms,
                   "sigv_median_kms": float(np.median(sigv)) if len(df) else None},
        "params": _jsonable(params.__dict__),
        "directions": {
            tag: {"t_horizon_myr": results[tag]["t_horizon_myr"],
                  "epochs_scanned": results[tag]["epochs_scanned"],
                  "n_raw_detections": results[tag]["n_raw_detections"],
                  "n_candidates": len(results[tag]["candidates"]),
                  "best_surprise": max((c["surprise"] for c in
                                        results[tag]["candidates"]), default=0.0),
                  "p_global": mock_stats.get(tag, {}).get("p_global")}
            for tag in results},
        "runtime_s": round(time.time() - t0, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    _write_report(out_dir, summary, results, mock_stats)
    print("[herdsman]", json.dumps(_jsonable(summary["directions"])))
    return summary


def _write_report(out_dir, summary, results, mock_stats) -> None:
    s = summary
    lines = [
        "# HERDSMAN run report", "",
        f"Sample: {s['sample']['final']} stars (fetched {s['sample']['fetched']}, "
        f"6D {s['sample']['with_6d']}, sigma_v <= {s['sample']['sigv_max_kms']} "
        f"km/s: {s['sample']['precise']}); median sigma_v = "
        f"{s['sample']['sigv_median_kms']:.2f} km/s."
        if s["sample"]["sigv_median_kms"] is not None else "Sample: empty.", "",
        "Direction summaries (the backward scan doubles as the time-reversal "
        "control for the forward scan — phase-mixed dynamics is statistically "
        "time-symmetric, deliberate future assembly is not):", "",
        "| direction | horizon (Myr) | epochs | raw | candidates | best surprise | p_global |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag in results:
        d = s["directions"][tag]
        pg = d["p_global"]
        lines.append(
            f"| {tag} | {d['t_horizon_myr']:.1f} | {d['epochs_scanned']} | "
            f"{d['n_raw_detections']} | {d['n_candidates']} | "
            f"{d['best_surprise']:.2f} | "
            f"{('%.3f' % pg) if pg is not None else 'n/a'} |")
    lines += ["", "## Candidates", ""]
    any_c = False
    for tag in results:
        for c in results[tag]["candidates"]:
            any_c = True
            ch = c.get("chemistry", {})
            mc = c.get("rendezvous_mc", {})
            lines += [
                f"### {tag} t = {c['t_myr']:+.1f} Myr — {c['m']} stars, "
                f"surprise {c['surprise']:.2f}",
                f"- members (Gaia DR3): "
                f"{', '.join(str(x) for x in c.get('member_source_ids', []))}",
                f"- now: rms {c['rms_now_pc']:.1f} pc, median pairwise "
                f"{c['med_now_pc']:.1f} pc; meeting: rms {c['rms_meet_pc']:.1f} pc "
                f"in ball {c['r_ball_pc']:.1f} pc (lambda {c['lambda']:.3g})",
                f"- focus x{c['focus']:.1f}; internal sigma_v "
                f"{c['sig_v_internal_kms']:.1f} km/s; seen at "
                f"{c['n_epochs_seen']} epochs",
                f"- chemistry: MAD {ch.get('mh_mad_dex')}, heterogeneous "
                f"{ch.get('heterogeneous')}, co-natal-possible "
                f"{ch.get('co_natal_possible')}",
                f"- rendezvous MC: rms_min p50 {mc.get('rms_min_pc_p50')}, "
                f"P(rms<5pc) {mc.get('p_rms_lt_5pc')}", "",
            ]
    if not any_c:
        lines += ["None above threshold in either direction at these settings.",
                  ""]
    lines += [
        "## Honest sensitivity statement", "",
        "The scan is sensitive only inside the self-computed horizon above "
        "(where an error-matched meeting ball still holds < "
        f"{results[next(iter(results))]['params']['lambda_cap']} field stars by "
        "chance). Herds converging beyond that horizon, herds of fewer than "
        f"{results[next(iter(results))]['params']['n_min']} members, and herds "
        "whose members fall outside the precision subset are NOT probed. "
        "A null here is a statement about this domain only and is not to be "
        "written up as a result; it is a reason to deepen the sample "
        "(better RVs, DR4) or change the question.", "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines))


__all__ = ["herdsman_run"]
