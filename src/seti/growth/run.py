"""Stage orchestration for GROWTH.  Writes ``results/growth/``.

Stages
------
``probe``    a ``TOP 5`` query on each of the three Exoplanet Archive tables:
             reachability, the real column names, nothing else.  Writes
             ``probe.json``.
``acquire``  pull ``cumulative`` (KOI), ``toi`` and the Kepler rows of ``ps``;
             join Kepler <-> TESS over four routes (``name_planet``,
             ``name_host``, ``position_tic``, then ``position_period``), each
             counted on its own; Gaia DR3 neighbours within one TESS pixel for
             every joined star (chunked upload crossmatch, falling back to
             per-target ``pyvo`` cones).  Writes ``data/koi.csv``,
             ``data/toi.csv``, ``data/ps_kepler.csv``, ``data/joined_raw.csv``,
             ``data/neighbours.csv``, ``data/neighbour_status.csv``,
             ``acquire.json``, ``acquisition_log.json``.
``screen``   the per-planet drift statistic (pure): limb-darkening band
             ratio, dilution, corrected log ratio, duration test.  Writes
             ``screened.csv``.
``assess``   vetoes, the population offset, classes, the long-period list,
             the verdict.  Writes ``joined.csv``, ``candidates.csv``,
             ``long_period.csv``, ``summary.json``.

Verdict vocabulary (``summary.json["verdict"]``)
------------------------------------------------
``NO_DATA_REACHED``             no joined planet could be measured (an archive
                                did not answer, or the join was empty; the
                                ``reason`` field says which)
``NO_DEPTH_DRIFT_CANDIDATE``    planets were measured; none reached the
                                candidate gate --- a count, not a limit
``DEPTH_DRIFT_CANDIDATES``      >= 1 ``GROWTH_CANDIDATE`` (pending stage 2)
``degraded`` lists every partial failure separately; the verdict string itself
is never decorated.  None of these is written up as a result.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .drift import CLASS_CANDIDATE, DriftParams, planet_drift
from .vet import VetParams, rejection_counters, vet_table

VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_NONE = "NO_DEPTH_DRIFT_CANDIDATE"
VERDICT_CANDIDATES = "DEPTH_DRIFT_CANDIDATES"
STAGES = ("probe", "acquire", "screen", "assess")

DEFAULTS: dict = {
    "archive": {"exoarchive_tap": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
                "retries": 3, "timeout_s": 900},
    "join": {"pos_radius_arcsec": 2.0, "period_tol": 1e-3, "alias_max": 4,
             "routes": ["name_planet", "name_host", "position_tic", "position_period"]},
    "gaia": {"neighbour_radius_arcsec": 21.0, "chunk": 300, "retries": 3,
             "tap_url": "https://gea.esac.esa.int/tap-server/tap", "fallback_cones": True,
             "cone_retries": 2, "checkpoint": True},
    "aperture": {"radius_arcsec": 21.0, "psf_sigma_arcsec": 10.5},
    "drift": {"sigma_sys": 0.05, "sigma_dur_sys": 0.05, "n_consistent": 3.0,
              "n_candidate": 5.0, "n_duration": 3.0, "apply_dilution_to_tess": True,
              "han2025_expected_deficit": 0.12, "subtract_population_median": True},
    "long_period": {"min_period_days": 30.0},
    "limb_darkening": None,
    "vet": {"ttv_systems": [], "toi_candidate_dispositions": ["PC", "CP", "KP"],
            "koi_candidate_dispositions": ["CONFIRMED", "CANDIDATE"]},
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _deep_update(base: dict, extra: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_growth_config(path: Path | None = None) -> dict:
    """``config/growth.yaml`` over :data:`DEFAULTS`; a missing file degrades."""
    try:
        import yaml  # noqa: PLC0415
        if path is None:
            path = Path(__file__).resolve().parents[3] / "config" / "growth.yaml"
        if not Path(path).exists():
            return _deep_update(DEFAULTS, {})
        return _deep_update(DEFAULTS, yaml.safe_load(Path(path).read_text()) or {})
    except Exception as exc:                              # noqa: BLE001
        print(f"[growth] config not loaded ({exc!r}); using defaults")
        return _deep_update(DEFAULTS, {})


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:                                     # noqa: BLE001
        return pd.DataFrame()


def _query_fn(conf: dict, query_fn):
    if query_fn is not None:
        return query_fn
    from .acquire import tap_sync  # noqa: PLC0415
    a = conf.get("archive") or {}
    url = str(a.get("exoarchive_tap") or DEFAULTS["archive"]["exoarchive_tap"])
    return lambda adql: tap_sync(adql, url=url, retries=int(a.get("retries", 3)),
                                 timeout=float(a.get("timeout_s", 900)))


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def stage_probe(conf: dict, out: Path, *, query_fn=None, log=None) -> dict:
    from .acquire import KOI_COLUMNS, PS_COLUMNS, TOI_COLUMNS, AcquisitionLog  # noqa: PLC0415

    log = log or AcquisitionLog()
    qf = _query_fn(conf, query_fn)
    tables = (conf.get("archive") or {}).get("tables") or {"koi": "cumulative", "toi": "toi",
                                                           "ps": "ps"}
    specs = {"koi": (tables.get("koi", "cumulative"), KOI_COLUMNS, None),
             "toi": (tables.get("toi", "toi"), TOI_COLUMNS, None),
             "ps": (tables.get("ps", "ps"), PS_COLUMNS,
                    (conf.get("archive") or {}).get("ps_where") or "disc_facility like '%Kepler%'")}
    found = {}
    for name, (table, cols, where) in specs.items():
        adql = f"select top 5 {','.join(cols)} from {table}" + (f" where {where}" if where else "")
        try:
            df = qf(adql)
            n = int(len(df)) if df is not None else 0
            log.record(f"probe_{name}", adql, rows=n)
            found[name] = {"table": table, "status": "OK" if n else "QUERY_RETURNED_ZERO_ROWS",
                           "columns": [str(c) for c in (df.columns if df is not None else [])],
                           "n_rows_peek": n}
        except Exception as exc:                          # noqa: BLE001
            log.record(f"probe_{name}", adql, error=repr(exc))
            found[name] = {"table": table, "status": "QUERY_FAILED", "error": repr(exc)[:300]}
    rep = {"stage": "probe", "generated_utc": _now(), "tables": found,
           "n_reachable": sum(1 for d in found.values() if d["status"] == "OK"),
           "acquisition": log.as_dict()}
    _write(out / "probe.json", rep)
    print(f"[growth] probe: {rep['n_reachable']}/3 tables reachable")
    return rep


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------
def _gaia_errors(log, *, max_attempts: int = 12) -> list[dict]:
    """Every Gaia attempt's exception TEXT, lifted out of the acquisition log.

    Run 34787801172 reported ``n_targets_failed: 108`` and nothing else in
    ``acquire.json["gaia"]``, which is what made the failure undiagnosable.
    The text now sits beside the count.
    """
    out = []
    for s in getattr(log, "stages", []) or []:
        if not str(s.get("stage", "")).startswith("gaia_"):
            continue
        att = [a for a in (s.get("attempts") or []) if not a.get("ok")]
        if not att and not s.get("error"):
            continue
        out.append({"stage": s["stage"], "status": s.get("status"),
                    "route": s.get("route"), "error": s.get("error"),
                    "n_targets": s.get("n_targets"), "n_targets_ok": s.get("n_targets_ok"),
                    "n_targets_failed": s.get("n_targets_failed"),
                    "n_failed_attempts": len(att), "attempts": att[:max_attempts]})
    return out


def stage_acquire(conf: dict, out: Path, *, query_fn=None, gaia_fn=None, log=None,
                  skip_gaia: bool = False) -> dict:
    from .acquire import (  # noqa: PLC0415
        JOIN_ROUTES,
        PS_WHERE_KEPLER,
        STATUS_OK,
        AcquisitionLog,
        fetch_gaia_neighbours,
        fetch_koi,
        fetch_ps_kepler,
        fetch_toi,
        gaia_neighbours_cones,
        join_kepler_tess,
    )

    log = log or AcquisitionLog()
    qf = _query_fn(conf, query_fn)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    arch = conf.get("archive") or {}
    koi, s_koi = fetch_koi(query_fn=qf, log=log)
    toi, s_toi = fetch_toi(query_fn=qf, log=log)
    ps, s_ps = fetch_ps_kepler(
        query_fn=qf, log=log,
        where=str(arch.get("ps_where") or PS_WHERE_KEPLER),
        require_kepler=bool(arch.get("ps_require_kepler_in_disc_facility", True)))
    koi.to_csv(data / "koi.csv", index=False)
    toi.to_csv(data / "toi.csv", index=False)
    ps.to_csv(data / "ps_kepler.csv", index=False)
    rep: dict = {"stage": "acquire", "generated_utc": _now(),
                 "tables": {"koi": {"status": s_koi, "n_rows": int(len(koi))},
                            "toi": {"status": s_toi, "n_rows": int(len(toi))},
                            "ps": {"status": s_ps, "n_rows": int(len(ps))}}}
    j = conf.get("join") or {}
    joined, jrep = join_kepler_tess(koi, toi, ps if s_ps == STATUS_OK else None,
                                    pos_radius_arcsec=float(j.get("pos_radius_arcsec", 2.0)),
                                    period_tol=float(j.get("period_tol", 1e-3)),
                                    alias_max=int(j.get("alias_max", 4)),
                                    routes=tuple(j.get("routes") or JOIN_ROUTES))
    joined.to_csv(data / "joined_raw.csv", index=False)
    rep["join"] = jrep
    g = conf.get("gaia") or {}
    if len(joined) and not skip_gaia:
        tap_url = str(g.get("tap_url") or "https://gea.esac.esa.int/tap-server/tap")
        cone_retries = int(g.get("cone_retries", 2))
        fallback = bool(g.get("fallback_cones", True))
        cone_fn = None
        if fallback and gaia_fn is None:
            # Bound the fallback to the SAME endpoint the config names, and give
            # it its own retry budget; a cone that never answers leaves its
            # target QUERY_FAILED (-> not_checked), never "isolated".
            def cone_fn(part, radius, *, attempts=None):  # noqa: ANN001
                return gaia_neighbours_cones(part, radius, retries=cone_retries,
                                             tap_url=tap_url, attempts=attempts)
        neigh, status = fetch_gaia_neighbours(
            joined[["planet_key", "ra", "dec"]], gaia_fn=gaia_fn, cone_fn=cone_fn, log=log,
            radius_arcsec=float(g.get("neighbour_radius_arcsec", 21.0)),
            chunk=int(g.get("chunk", 300)),
            checkpoint_dir=(data / "gaia_checkpoint") if g.get("checkpoint", True) else None)
        neigh.to_csv(data / "neighbours.csv", index=False)
        status.to_csv(data / "neighbour_status.csv", index=False)
        rep["gaia"] = {"n_targets": int(len(status)),
                       "n_targets_ok": int((status["neighbours_status"] == STATUS_OK).sum()),
                       "n_targets_failed": int((status["neighbours_status"] != STATUS_OK).sum()),
                       "n_neighbour_rows": int(len(neigh)),
                       "radius_arcsec": float(g.get("neighbour_radius_arcsec", 21.0)),
                       "tap_url": tap_url, "fallback_cones": bool(cone_fn is not None),
                       "by_route": (status["neighbours_route"].fillna("").value_counts().to_dict()
                                    if "neighbours_route" in status else {})}
        rep["gaia"]["errors"] = _gaia_errors(log)
    else:
        rep["gaia"] = {"n_targets": 0, "n_targets_ok": 0, "n_targets_failed": int(len(joined)),
                       "n_neighbour_rows": 0, "skipped": bool(skip_gaia), "errors": [],
                       "by_route": {}}
        pd.DataFrame(columns=["planet_key", "neighbours_status", "neighbours_route"]).to_csv(
            data / "neighbour_status.csv", index=False)
    rep["acquisition"] = log.as_dict()
    _write(out / "acquire.json", rep)
    log.write(out / "acquisition_log.json")
    print(f"[growth] acquire: koi={s_koi}({len(koi)}) toi={s_toi}({len(toi)}) ps={s_ps}({len(ps)})")
    print(f"[growth] join: {jrep.get('join_statement', '')}")
    if rep["gaia"].get("errors"):
        for e in rep["gaia"]["errors"]:
            print(f"[growth] gaia {e['stage']}: route={e.get('route')} "
                  f"{e.get('n_targets_failed')}/{e.get('n_targets')} failed; "
                  f"first error: {(e.get('attempts') or [{}])[0].get('error')}")
    return rep


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------
def screen_table(joined: pd.DataFrame, neighbours: pd.DataFrame | None,
                 status: pd.DataFrame | None, conf: dict) -> pd.DataFrame:
    """The drift statistic for every joined planet (pure; used by the tests)."""
    if not len(joined):
        return pd.DataFrame()
    params = DriftParams.from_config(conf)
    by_key: dict[str, pd.DataFrame] = {}
    if neighbours is not None and len(neighbours) and "planet_key" in neighbours:
        by_key = {str(k): g for k, g in neighbours.groupby(neighbours["planet_key"].astype(str))}
    st: dict[str, str] = {}
    if status is not None and len(status):
        st = dict(zip(status["planet_key"].astype(str), status["neighbours_status"].astype(str),
                      strict=True))
    recs = []
    for row in joined.to_dict(orient="records"):
        key = str(row.get("planet_key"))
        d = planet_drift(row, by_key.get(key), params)
        d["neighbours_status"] = st.get(key, "QUERY_FAILED" if st else "not_checked")
        recs.append({**row, **d})
    return pd.DataFrame(recs)


def stage_screen(conf: dict, out: Path, *, joined: pd.DataFrame | None = None,
                 neighbours: pd.DataFrame | None = None, status: pd.DataFrame | None = None
                 ) -> dict:
    data = out / "data"
    if joined is None:
        joined = _read_csv(data / "joined_raw.csv")
    if neighbours is None:
        neighbours = _read_csv(data / "neighbours.csv")
    if status is None:
        status = _read_csv(data / "neighbour_status.csv")
    scr = screen_table(joined, neighbours, status, conf)
    scr.to_csv(out / "screened.csv", index=False)
    rep = {"stage": "screen", "generated_utc": _now(), "n_joined": int(len(joined)),
           "n_screened": int(len(scr)),
           "n_measured": int(np.isfinite(pd.to_numeric(scr.get("ln_ratio_corr", pd.Series(
               dtype=float)), errors="coerce")).sum()) if len(scr) else 0}
    _write(out / "screen.json", rep)
    print(f"[growth] screen: {rep['n_screened']} planets, {rep['n_measured']} with a measured ratio")
    return rep


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------
CANDIDATE_COLUMNS = (
    "planet_key", "kepoi_name", "kepler_name", "kepid", "tic_id", "toi", "join_route",
    "period_alias", "koi_period", "pl_orbper", "koi_disposition", "tfopwg_disp",
    "depth_kepler_ppm", "depth_kepler_err_ppm", "depth_tess_ppm", "depth_tess_err_ppm",
    "k_kepler", "b_kepler", "a_rs_kepler", "ld_band_ratio", "contam_applied", "contam_max",
    "n_gaia_neighbours", "ln_ratio_raw", "ln_ratio_corr", "sigma_ln_ratio", "population_offset",
    "z_ratio", "t14_ratio_observed", "t14_ratio_expected_fixed_b", "z_duration",
    "b_implied_by_duration", "duration_verdict", "class", "vetoes", "first_veto", "flags",
    "would_be_candidate_without_vetoes", "koi_tce_delivname",
)


def _gaia_statement(gaia_rep: dict) -> str:
    """One plain sentence about the dilution term's reachability."""
    n = int(gaia_rep.get("n_targets", 0) or 0)
    ok = int(gaia_rep.get("n_targets_ok", 0) or 0)
    bad = int(gaia_rep.get("n_targets_failed", 0) or 0)
    by = gaia_rep.get("by_route") or {}
    routes = ", ".join(f"{k or 'unreached'}={v}" for k, v in sorted(by.items())) or "none"
    tail = ("" if not bad else
            f"; {bad} star(s) reached NO Gaia route and are 'not_checked' — none of them may "
            "be called isolated")
    return f"{ok} of {n} joined stars have Gaia neighbours (by route: {routes}){tail}"


def long_period_list(joined: pd.DataFrame, toi: pd.DataFrame, min_period_days: float
                     ) -> pd.DataFrame:
    """Every joined or TESS-only planet with P > ``min_period_days`` (stage 2 input)."""
    rows = []
    used = set()
    if len(joined):
        p = pd.to_numeric(joined.get("koi_period"), errors="coerce")
        for _, r in joined[p > min_period_days].iterrows():
            rows.append({"key": str(r.get("planet_key")), "source": "joined",
                         "period_days": float(r.get("koi_period")), "toi": r.get("toi"),
                         "tic_id": r.get("tic_id"), "kepoi_name": r.get("kepoi_name")})
        used = {str(t) for t in joined.get("toi", pd.Series(dtype=str)).dropna().astype(str)}
    if len(toi):
        p = pd.to_numeric(toi.get("pl_orbper"), errors="coerce")
        for _, r in toi[p > min_period_days].iterrows():
            if str(r.get("toi")) in used:
                continue
            rows.append({"key": f"TOI-{r.get('toi')}", "source": "tess_only",
                         "period_days": float(r.get("pl_orbper")), "toi": r.get("toi"),
                         "tic_id": r.get("tid"), "kepoi_name": None})
    return pd.DataFrame(rows, columns=["key", "source", "period_days", "toi", "tic_id",
                                       "kepoi_name"])


def stage_assess(conf: dict, out: Path, *, screened: pd.DataFrame | None = None,
                 acquire_report: dict | None = None, toi: pd.DataFrame | None = None) -> dict:
    if screened is None:
        screened = _read_csv(out / "screened.csv")
    if acquire_report is None and (out / "acquire.json").exists():
        try:
            acquire_report = json.loads((out / "acquire.json").read_text())
        except Exception:                                 # noqa: BLE001
            acquire_report = None
    if toi is None:
        toi = _read_csv(out / "data" / "toi.csv")
    acq_tables = (acquire_report or {}).get("tables") or {}
    join_rep = (acquire_report or {}).get("join") or {}
    gaia_rep = (acquire_report or {}).get("gaia") or {}
    lp = long_period_list(screened, toi,
                          float((conf.get("long_period") or {}).get("min_period_days", 30.0)))
    lp.to_csv(out / "long_period.csv", index=False)

    n_measured = int(np.isfinite(pd.to_numeric(screened.get("ln_ratio_corr", pd.Series(
        dtype=float)), errors="coerce")).sum()) if len(screened) else 0
    degraded: list[str] = []
    for k, v in acq_tables.items():
        if v.get("status") != "OK":
            degraded.append(f"{k}:{v.get('status')}")
    if gaia_rep.get("n_targets_failed"):
        degraded.append(f"gaia_neighbours_not_checked:{gaia_rep['n_targets_failed']}")

    if n_measured == 0:
        # Nothing about depth drift was measured.  Say which fact this is.
        if not acq_tables:
            reason = "no_acquire_report"
        elif any(v.get("status") == "QUERY_FAILED" for v in acq_tables.values()):
            reason = "archive_query_failed"
        elif any(v.get("status") != "OK" for v in acq_tables.values()):
            reason = "archive_returned_zero_rows"
        elif not len(screened):
            reason = "join_returned_zero_rows"
        else:
            reason = "no_joined_planet_has_both_depths"
        summary = {"verdict": VERDICT_NO_DATA, "reason": reason, "generated_utc": _now(),
                   "n_joined": int(len(screened)), "n_measured": 0,
                   "join": join_rep, "join_statement": join_rep.get("join_statement", ""),
                   "gaia": gaia_rep, "classes": rejection_counters(
                       pd.DataFrame())["classes"],
                   "rejection_counters": rejection_counters(pd.DataFrame()),
                   "n_long_period": int(len(lp)), "degraded": degraded,
                   "acquisition_tables": acq_tables,
                   "note": ("no planet's depth ratio was measured, so nothing about depth "
                            "drift was tested; this is NOT a null result and must not be "
                            "reported as one")}
        _write(out / "summary.json", summary)
        pd.DataFrame(columns=CANDIDATE_COLUMNS).to_csv(out / "candidates.csv", index=False)
        screened.to_csv(out / "joined.csv", index=False)
        print(f"[growth] assess: {VERDICT_NO_DATA} ({reason})")
        return summary

    vp = VetParams.from_config(conf)
    vetted, pop = vet_table(screened, vp)
    counters = rejection_counters(vetted)
    vetted.to_csv(out / "joined.csv", index=False)
    cands = vetted[vetted["class"] == CLASS_CANDIDATE].copy()
    cands = cands.sort_values("z_ratio", ascending=False)
    cols = [c for c in CANDIDATE_COLUMNS if c in cands.columns]
    cands[cols].to_csv(out / "candidates.csv", index=False)
    verdict = VERDICT_CANDIDATES if len(cands) else VERDICT_NONE
    ln_corr = pd.to_numeric(vetted["ln_ratio_corr"], errors="coerce")
    z = pd.to_numeric(vetted["z_ratio"], errors="coerce")
    summary = {
        "verdict": verdict, "generated_utc": _now(),
        "n_joined": int(len(vetted)), "n_measured": n_measured,
        "n_candidates": int(len(cands)),
        "join": join_rep,
        # The plain statement the 2026-09-13 report could not make: how many of
        # the KOIs reached a TESS counterpart at all, and by which route.
        "join_statement": join_rep.get("join_statement", ""),
        "gaia": gaia_rep,
        "gaia_statement": _gaia_statement(gaia_rep),
        "classes": counters["classes"],
        "rejection_counters": counters,
        "population": {**pop,
                       "ln_ratio_corr_median": float(np.nanmedian(ln_corr)),
                       "ln_ratio_corr_p16": float(np.nanpercentile(ln_corr, 16)),
                       "ln_ratio_corr_p84": float(np.nanpercentile(ln_corr, 84)),
                       "z_abs_median": float(np.nanmedian(np.abs(z))),
                       "n_z_above_3": int((z >= 3).sum()), "n_z_below_minus_3": int((z <= -3).sum()),
                       "n_z_above_5": int((z >= 5).sum())},
        "n_long_period": int(len(lp)),
        "long_period_by_source": lp["source"].value_counts().to_dict() if len(lp) else {},
        "degraded": degraded,
        "acquisition_tables": acq_tables,
        "config": {"drift": conf.get("drift"), "aperture": conf.get("aperture"),
                   "join": conf.get("join"), "gaia": conf.get("gaia"),
                   "long_period": conf.get("long_period"),
                   "ld_source": (conf.get("limb_darkening") or {}).get("source")},
        "checks_not_performed": ["multi_sector_tess_depth_scatter (stage 2)",
                                 "per_epoch_k(t)_fit (stage 2)",
                                 "ingress_egress_asymmetry_long_period (stage 2)"],
        "note": ("a GROWTH_CANDIDATE here is a CATALOGUE statement about two heterogeneous "
                 "pipeline depths and is pending the stage-2 light-curve k(t) fit; "
                 "NO_DEPTH_DRIFT_CANDIDATE is a count, not an occurrence limit, and is "
                 "not written up (CLAUDE.md)"),
    }
    _write(out / "summary.json", summary)
    print(f"[growth] assess: {verdict} — {len(cands)} candidate of {n_measured} measured "
          f"({counters['classes']}); population offset {pop['population_offset']:+.3f}")
    return summary


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def growth_run(stage: str = "all", *, out_dir=None, conf: dict | None = None, query_fn=None,
               gaia_fn=None, skip_gaia: bool = False) -> dict:
    """Run one stage, a comma list, or all.  Returns the last stage's report."""
    conf = conf if conf is not None else load_growth_config()
    out = Path(out_dir) if out_dir else Path("results") / "growth"
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage in ("all", "", None) else tuple(s.strip() for s in stage.split(","))
    rep: dict = {}
    for s in stages:
        if s == "probe":
            rep = stage_probe(conf, out, query_fn=query_fn)
        elif s == "acquire":
            rep = stage_acquire(conf, out, query_fn=query_fn, gaia_fn=gaia_fn, skip_gaia=skip_gaia)
        elif s == "screen":
            rep = stage_screen(conf, out)
        elif s == "assess":
            rep = stage_assess(conf, out)
        else:
            raise SystemExit(f"unknown stage {s!r}; choose from {STAGES}")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(prog="seti growth",
                                description="GROWTH (S57): Kepler -> TESS transit-depth drift")
    p.add_argument("--stage", default="all", choices=("probe", "acquire", "screen", "assess", "all"))
    p.add_argument("--out-dir", default="results/growth", help="results directory")
    p.add_argument("--skip-gaia", action="store_true",
                   help="acquire without the Gaia neighbour cones (every star then not_checked)")
    a = p.parse_args(argv)
    rep = growth_run(a.stage, out_dir=a.out_dir, skip_gaia=a.skip_gaia)
    v = rep.get("verdict") if isinstance(rep, dict) else None
    if v:
        print(f"[growth] verdict: {v}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())


__all__ = ["CANDIDATE_COLUMNS", "DEFAULTS", "STAGES", "VERDICT_CANDIDATES", "VERDICT_NONE",
           "VERDICT_NO_DATA", "growth_run", "load_growth_config", "long_period_list", "main",
           "screen_table", "stage_acquire", "stage_assess", "stage_probe", "stage_screen"]
