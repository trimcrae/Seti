"""Stage orchestration for LOOM.  Writes ``results/loom/``.

Stages
------
``probe``
    Runner-only, cheap, and **run first**.  Records the live solar-system schema
    verbatim and — the query that actually decides the channel's architecture —
    measures the null fraction of the non-gravitational columns.  Every field in
    ``lsst_mpc_orbits``'s ``yarkovsky``/``srp``/``a1``/``a2``/``a3`` block is
    nullable with ``default: null``, and MPC fits those terms for only a small
    minority of objects.  If they are empty in the mirror, Path A is dead and the
    channel runs on per-detection residuals alone; that is a fact about the data
    that must be established before anything is built on top of it, not
    discovered from an inexplicable empty result three weeks later.
``screen``
    The recurring stage.  Pulls the quality-gated parent population, screens it
    against the momentum ceiling, builds a shortlist, and pulls the per-detection
    residual time series only for the shortlist.
``assess``
    Offline.  Runs the population-structure tests and the positive-control
    validation over everything accumulated so far.  Separated from ``screen`` so
    the statistics can be re-derived under a changed threshold without touching
    the network.

Degradation is explicit everywhere.  An unreachable service writes
``verdict: NO_DATA_REACHED``; an empty non-gravitational block writes
``NONGRAV_COLUMNS_EMPTY``.  Neither is written as an empty candidate table, which
would read as a clean null on a search that never ran.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..tocsin.brokers import ALERCE_TAP
from .acquire import AlerceSSO
from .calibrate import fetch_sbdb, summarise_epsilon, verify_yarkovsky_unit
from .control import control_index, normalise_designation, validate
from .litcheck import build_corpus, check_objects
from .nongrav import anomaly_ratio, calibration_table, ceiling_ratio, fit_envelope
from .replication import replication_tests
from .residuals import (
    along_cross_partition,
    apparition_trend,
    arcsec_to_km,
    breakpoint_scan,
    decompose_offset,
    drift_fit,
    fit_common_timing,
    law_discrimination,
    per_object_rate_correlation,
    quality_independence,
    sky_coherence,
    usable_range,
)
from .screen import Thresholds, assign_tier, screen_orbits

DEFAULTS: dict = {
    "acquire": {
        "alerce_tap_url": ALERCE_TAP,
        "timeout_s": 900,
        "maxrec": 2_000_000,
        # MEASURED 2026-07-30: solar-system `detection` rows carry sid = 2.  This
        # was `null` (clause dropped) until the probe settled it, because a wrong
        # guess returns an empty result rather than an error.
        "sid_ssobject": 2,
        "join_on": "measurement_id",
        "nearest_only": True,
        # Quality cuts on the parent population.  These are the cuts Del Vigna
        # et al.'s reliability criteria imply; a blind non-gravitational search
        # without them returns a majority of spurious detections.
        "h_max": 26.0,
        "normalized_rms_max": 1.5,
        "min_oppositions": 2,
        "min_arc_days": 180.0,
        "require_nongrav": False,
        # Residual path.
        "residual_window_days": 900.0,
        # Raised from 6: the along-track rotation needs several epochs, and the
        # first live run failed the drift fit on 273 of 400 objects for want of
        # epochs rather than for want of signal.
        "min_detections_for_summary": 12,
        # Covers the whole eligible population; batching made a complete screen
        # affordable, and it removes a selection step on the observable itself.
        "shortlist_size": 20000,
        # Objects per residual query.  The first live run used one query per object
        # at 1.4 s each; batching turns 400 queries into 20 and the wall-clock
        # budget then buys an order of magnitude more objects.
        "residual_batch_size": 100,
        "max_run_seconds": 5400.0,
    },
    "screen": {
        "albedo_generous": 0.25, "rho_generous_kg_m3": 1000.0,
        "albedo_typical": 0.15, "rho_typical_kg_m3": 2000.0,
        "rho_min_kg_m3": 500.0,
        "epsilon_realistic": 0.1, "epsilon_hard": 1.0, "epsilon_inviolable": 2.0,
        "min_snr_a2": 3.0, "max_normalized_rms": 1.5,
        "min_arc_days": 180.0, "min_oppositions": 2,
        "amr_artificial_floor": 1.0e-3, "min_amr_ceiling_ratio": 1.0,
        "min_detections_for_drift": 8, "min_accel_snr": 5.0,
        "max_timing_correlation": 0.5, "min_delta_chi2_law": 9.0,
        "max_sky_variance_explained": 0.3,
        "population_n_null": 2000, "population_seed": 20260730,
    },
    "assess": {
        "empirical_envelope_quantile": 0.999,
        "empirical_envelope_min_per_bin": 200,
        "collisional_slope": 0.5,
        "breakpoint_n_null": 500,
        # Rank correlation between the anomaly score and any observation-quality
        # covariate above which the population result is not interpretable.  0.3
        # shares ~9% of the rank variance; the first working run reached 0.475
        # against detection count -- 23% -- and slipped under a 0.5 threshold.
        "max_quality_correlation": 0.3,
    },
    "report": {"results_dir": "results/loom", "max_rows_written": 200_000},
}

MJD_UNIX_EPOCH = 40587.0


# ---------------------------------------------------------------------------
# config / io helpers
# ---------------------------------------------------------------------------
def _deep_update(base: dict, extra: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def load_loom_config(cfg=None) -> dict:
    """Read ``config/loom.yaml`` over :data:`DEFAULTS`; a missing file degrades."""
    try:
        import yaml
        root = Path(cfg.root) if cfg is not None else _repo_root()
        p = root / "config" / "loom.yaml"
        if not p.exists():
            return _deep_update(DEFAULTS, {})
        return _deep_update(DEFAULTS, yaml.safe_load(p.read_text()) or {})
    except Exception as exc:                              # noqa: BLE001
        print(f"[loom] config/loom.yaml not loaded ({exc!r}); using defaults")
        return _deep_update(DEFAULTS, {})


def thresholds_from_config(conf: dict) -> Thresholds:
    """Build :class:`~seti.loom.screen.Thresholds` from the config's screen block.

    Only keys the dataclass actually declares are passed through, so a stray or
    renamed config entry raises here rather than being silently ignored — the
    failure mode that let TOCSIN's ``Thresholds`` drift away from its YAML and
    screen a full backlog with a stale reliability cut.
    """
    fields = set(Thresholds().__dict__)
    sconf = dict(conf.get("screen") or {})
    unknown = sorted(set(sconf) - fields)
    if unknown:
        raise ValueError(f"config/loom.yaml screen: unknown keys {unknown}")
    return Thresholds(**{k: v for k, v in sconf.items() if k in fields})


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config").is_dir():
            return parent
    return here.parents[3]


def _now_mjd() -> float:
    return MJD_UNIX_EPOCH + datetime.now(timezone.utc).timestamp() / 86400.0


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=1, sort_keys=True,
                               allow_nan=False) + "\n")


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _fz(v) -> float:
    """Like :func:`_f`, but exactly zero is missing --- see ``screen._fz``.

    This mirror writes 0.0 where a quantity was not determined, and `ephrate` is
    identically zero for every solar-system detection in it.  Reading that as a
    measured rate would make the timing-degeneracy regression fit a column of
    zeros and report a confident null.
    """
    x = _f(v)
    return float("nan") if x == 0.0 else x


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def probe(cfg=None, out_dir: str | Path | None = None) -> dict:
    """Live schema, null fractions and diagnostics for the solar-system tables.

    Writes ``probe.json`` after every query, so a job timeout part-way through
    leaves every answer already obtained on disk instead of losing the pass.
    """
    conf = load_loom_config(cfg)
    root = Path(cfg.root) if cfg is not None else _repo_root()
    out = Path(out_dir) if out_dir else root / conf["report"]["results_dir"]
    aconf = conf["acquire"]
    rec: dict = {"probed_at_utc": _utc(), "now_mjd": round(_now_mjd(), 5),
                 "alerce": {"url": aconf["alerce_tap_url"], "reached": False},
                 # The theory calibration travels with every probe so a reader
                 # can see what the ceiling is anchored to without leaving the
                 # results directory.
                 "momentum_ceiling_calibration": calibration_table()}
    sso = AlerceSSO(aconf["alerce_tap_url"], timeout=float(aconf["timeout_s"]))

    def checkpoint() -> None:
        rec["verdict"] = "OK" if rec["alerce"].get("reached") else "NO_DATA_REACHED"
        _write_json(out / "probe.json", rec)

    checkpoint()
    try:
        rec["alerce"]["schema"] = sso.describe()
        rec["alerce"]["reached"] = True
    except Exception as exc:                              # noqa: BLE001
        rec["alerce"]["error"] = f"{type(exc).__name__}: {exc}"[:800]
    checkpoint()

    if rec["alerce"]["reached"]:
        # The decisive measurement: are the non-gravitational and offset columns
        # populated, and in what units?  Everything downstream branches on it.
        try:
            rec["alerce"]["null_fractions"] = sso.null_fractions()
        except Exception as exc:                          # noqa: BLE001
            rec["alerce"]["null_fractions"] = {
                "error": f"{type(exc).__name__}: {exc}"[:800]}
        checkpoint()

        try:
            diag: dict = {}
            rec["alerce"]["diagnostics"] = diag

            def _record(name, result):
                diag[name] = result
                checkpoint()

            sso.diagnostics(on_result=_record)
        except Exception as exc:                          # noqa: BLE001
            rec["alerce"]["diagnostics"] = {
                "error": f"{type(exc).__name__}: {exc}"[:800]}
        checkpoint()

        try:
            rec["alerce"]["frontier_mjd"] = sso.max_available_mjd(
                join_on=str(aconf["join_on"]))
        except Exception as exc:                          # noqa: BLE001
            rec["alerce"]["frontier_error"] = f"{type(exc).__name__}: {exc}"[:400]
        checkpoint()

    rec["calls"] = sso.calls
    checkpoint()
    print(f"[loom] probe verdict={rec['verdict']} calls={rec['calls']}")
    return rec


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------
def screen(cfg=None, out_dir: str | Path | None = None,
           max_run_seconds: float | None = None, conf: dict | None = None) -> dict:
    """Pull the parent population, screen it, and work the shortlist's residuals.

    The wall-clock budget is not a nicety.  A GitHub Actions job that is cancelled
    on timeout never runs its commit step, so everything the run learned is
    discarded — which is how a three-hour TOCSIN backfill produced nothing at all.
    This yields voluntarily and writes what it has.
    """
    t0 = time.time()
    conf = conf if conf is not None else load_loom_config(cfg)
    root = Path(cfg.root) if cfg is not None else _repo_root()
    out = Path(out_dir) if out_dir else root / conf["report"]["results_dir"]
    aconf, th = conf["acquire"], thresholds_from_config(conf)
    budget = float(max_run_seconds if max_run_seconds is not None
                   else aconf["max_run_seconds"])
    rec: dict = {"screened_at_utc": _utc(), "verdict": "NOT_RUN",
                 "budget_seconds": budget, "timings": {}}

    def elapsed() -> float:
        return time.time() - t0

    def out_of_time(margin: float = 120.0) -> bool:
        return elapsed() > (budget - margin)

    def checkpoint() -> None:
        rec["elapsed_seconds"] = round(elapsed(), 1)
        _write_json(out / "screen.json", rec)

    checkpoint()
    sso = AlerceSSO(aconf["alerce_tap_url"], timeout=float(aconf["timeout_s"]),
                    maxrec=int(aconf["maxrec"]))

    # -- the parent population -------------------------------------------
    t = time.time()
    orbits = sso.orbits(h_max=_none_or_float(aconf["h_max"]),
                        normalized_rms_max=_none_or_float(aconf["normalized_rms_max"]),
                        min_oppositions=_none_or_int(aconf["min_oppositions"]),
                        min_arc_days=_none_or_float(aconf["min_arc_days"]),
                        require_nongrav=bool(aconf["require_nongrav"]))
    rec["timings"]["orbits_s"] = round(time.time() - t, 1)
    rec["orbits_verdict"] = orbits.verdict
    rec["n_orbits"] = len(orbits.rows)
    rec["notes"] = list(orbits.notes)
    checkpoint()
    if not orbits.rows:
        rec["verdict"] = ("NO_DATA_REACHED" if orbits.verdict == "NO_DATA_REACHED"
                          else "EMPTY_PARENT_POPULATION")
        checkpoint()
        print(f"[loom] screen verdict={rec['verdict']}")
        return rec

    recs, funnel = screen_orbits(orbits.rows, th)
    rec["funnel_path_a"] = funnel
    checkpoint()

    # -- the residual path ------------------------------------------------
    frontier = sso.max_available_mjd(join_on=str(aconf["join_on"]))
    rec["frontier_mjd"] = frontier
    window = float(aconf["residual_window_days"])
    mjd_hi = frontier + 1.0 if frontier is not None else None
    mjd_lo = frontier - window if frontier is not None else None
    rec["residual_window"] = {"mjd_lo": mjd_lo, "mjd_hi": mjd_hi,
                              "days": window}
    checkpoint()

    summaries = []
    if not out_of_time():
        t = time.time()
        # The SAME quality cuts as the parent-population query, applied inside the
        # aggregate.  Without this the shortlist is drawn from a different
        # population than the screen, and the expensive per-object stage is spent
        # outside it -- 370 of 400 objects in the first live run.
        summary = sso.object_residual_summary(
            mjd_lo=mjd_lo, mjd_hi=mjd_hi,
            min_detections=int(aconf["min_detections_for_summary"]),
            join_on=str(aconf["join_on"]),
            nearest_only=bool(aconf["nearest_only"]),
            require_orbit=True,
            h_max=_none_or_float(aconf["h_max"]),
            normalized_rms_max=_none_or_float(aconf["normalized_rms_max"]),
            min_oppositions=_none_or_int(aconf["min_oppositions"]),
            min_arc_days=_none_or_float(aconf["min_arc_days"]))
        rec["timings"]["residual_summary_s"] = round(time.time() - t, 1)
        rec["residual_summary_verdict"] = summary.verdict
        summaries = summary.rows
        rec["n_residual_summaries"] = len(summaries)
        rec["notes"].extend(summary.notes)
    else:
        rec["residual_summary_verdict"] = "SKIPPED_OUT_OF_TIME"
    checkpoint()

    by_key = {r.key: r for r in recs}
    for s in summaries:
        key = str(s.get("ssobjectid", "") or "")
        r = by_key.get(key)
        if r is None:
            continue
        r.n_detections = int(_f(s.get("n_det")) if math.isfinite(_f(s.get("n_det")))
                             else 0)

    # The sample-wide timing systematic, fitted before any per-object claim.  A
    # shutter or clock offset is degenerate with acceleration in a single object
    # and identifiable only across the population, so it is measured here and
    # subtracted, not merely noted.
    timing = fit_common_timing(
        [_f(s.get("mean_along")) if math.isfinite(_f(s.get("mean_along")))
         else _f(s.get("mean_offset")) for s in summaries],
        [_fz(s.get("mean_ephrate")) for s in summaries])
    rec["common_timing"] = timing.as_dict()
    if not timing.ok:
        rec["notes"].append(
            "the common timing offset could NOT be fitted: ephrate is zero-filled "
            "for every solar-system detection in this mirror, so the sample's "
            "residuals are not timing-decontaminated and the per-object timing "
            "veto is untestable.  A shutter or clock offset would be "
            "indistinguishable from a real along-track acceleration here.")
    checkpoint()

    # Shortlist: the objects whose aggregate along-track offset is largest after
    # the common timing term is removed.  A full time-series pull for every object
    # a survey detects is not affordable and is not necessary.
    shortlist = _shortlist(summaries, timing.dt_seconds,
                           int(aconf["shortlist_size"]))
    # FORCE THE POSITIVE CONTROLS IN, whatever their ranking.  The first working
    # run had 2020 SO (the 1966 Surveyor 2 Centaur) and 2007 VN84 (the Rosetta
    # spacecraft) in the parent population and measured neither, because neither
    # ranked into the top 2000 by offset -- so the one falsifiable check the channel
    # has went unexercised.  A control you do not measure is not a control.
    forced = _control_keys(summaries, by_key)
    n_forced = 0
    for key in forced:
        if key not in shortlist:
            shortlist.append(key)
            n_forced += 1
    rec["n_shortlist"] = len(shortlist)
    rec["n_controls_forced_into_shortlist"] = n_forced
    checkpoint()

    # BATCHED, not one query per object.  The first live run spent 1.4 s per object
    # on 400 separate queries; the same rows come back in one query per batch, so
    # the wall-clock budget buys an order of magnitude more objects.  The rows are
    # then split by `ssobjectid` locally, which costs nothing.
    residual_tests: dict = {}
    n_done, n_batches = 0, 0
    batch_size = int(aconf["residual_batch_size"])
    for start in range(0, len(shortlist), batch_size):
        if out_of_time():
            rec["notes"].append(
                f"residual time series stopped after {n_done}/{len(shortlist)} "
                f"shortlisted objects: wall-clock budget reached.  The remainder "
                f"are NOT screened and must not be counted as trials.")
            break
        batch = shortlist[start:start + batch_size]
        series = sso.residual_detections(
            mjd_lo=mjd_lo, mjd_hi=mjd_hi, ss_keys=batch,
            join_on=str(aconf["join_on"]),
            sid_ssobject=_none_or_int(aconf["sid_ssobject"]),
            nearest_only=bool(aconf["nearest_only"]))
        n_batches += 1
        by_object: dict[str, list[dict]] = {}
        for row in series.rows:
            by_object.setdefault(str(row.get("ssobjectid", "") or ""), []).append(row)
        for key in batch:
            rows_for = by_object.get(str(key)) or []
            if not rows_for:
                continue
            tests = analyse_series(rows_for, th,
                                   n_null=int(conf["assess"]["breakpoint_n_null"]))
            residual_tests[str(key)] = tests
            r = by_key.get(str(key))
            if r is not None:
                _apply_series(r, tests, th)
                assign_tier(r, th)
            n_done += 1
        rec["n_residual_series_done"] = n_done
        rec["n_residual_batches"] = n_batches
        checkpoint()
    rec["n_residual_series_done"] = n_done
    rec["n_residual_batches"] = n_batches
    # How many of the analysed objects actually reached a screened record?  The
    # first live run scored 30 of 400 here and reported a clean null on the other
    # 370 without ever testing them; anything well below 1.0 means the shortlist and
    # the parent population have drifted apart again.
    rec["shortlist_in_parent_fraction"] = (
        float(sum(1 for k in residual_tests if str(k) in by_key)) / len(residual_tests)
        if residual_tests else None)
    if (rec["shortlist_in_parent_fraction"] is not None
            and rec["shortlist_in_parent_fraction"] < 0.9):
        rec["notes"].append(
            f"only {rec['shortlist_in_parent_fraction']:.1%} of the analysed "
            f"objects have a row in the screened parent population, so the rest "
            f"were measured and DISCARDED.  The shortlist query and the parent "
            f"query are selecting different populations -- fix that before reading "
            f"any funnel below as a result.")

    # The photometric axis.  Pulled for the shortlist only, and only after the
    # dynamical work is done, because it is an *independent* test rather than part
    # of the selection: nothing in ssObject enters the anomaly cut, so a
    # homogeneity statistic on it is not a restatement of how the set was chosen.
    if shortlist and not out_of_time(margin=60.0):
        t = time.time()
        photo = sso.ss_objects(ss_keys=shortlist)
        rec["timings"]["ss_object_s"] = round(time.time() - t, 1)
        rec["ss_object_verdict"] = photo.verdict
        rec["notes"].extend(photo.notes)
        merged = 0
        for row in photo.rows:
            r = by_key.get(str(row.get("ssobjectid", "") or ""))
            if r is None:
                continue
            r.h_g, r.h_r = _f(row.get("g_h")), _f(row.get("r_h"))
            r.h_i, r.h_z = _f(row.get("i_h")), _f(row.get("z_h"))
            r.g12_r = _f(row.get("r_g12"))
            r.extendedness_median = _f(row.get("medianextendedness"))
            r.moid_earth = _f(row.get("moidearth"))
            r.tisserand_j = _f(row.get("tisserandj"))
            merged += 1
            assign_tier(r, th)
        rec["n_ss_object_merged"] = merged
    else:
        rec["ss_object_verdict"] = "SKIPPED"
    rec["timings"]["total_s"] = round(elapsed(), 1)

    # COVERAGE: how long a baseline did this run actually have?
    #
    # This is the number that decides whether the channel can say anything at all,
    # and it changes every week as the survey accumulates.  Recording it makes the
    # run self-describing: a reader a year from now can see at a glance that the
    # 2026-07-30 runs had a median residual baseline of days, not years, and that
    # every object had one apparition -- which is why the law-discrimination and
    # apparition-trend tests, the channel's actual novelty, could not run.  Without
    # it a future reader would have to reconstruct the survey's age to interpret an
    # empty candidate list, and would probably read it as a null instead.
    arcs = np.array([_f(r.residual_arc_days) for r in recs
                     if math.isfinite(_f(r.residual_arc_days))])
    apps = np.array([r.n_apparitions for r in recs if r.n_apparitions > 0])
    cov: dict = {"n_with_residual_fit": int(arcs.size)}
    if arcs.size:
        cov.update({
            "residual_arc_days_median": float(np.median(arcs)),
            "residual_arc_days_max": float(arcs.max()),
            "residual_arc_days_p90": float(np.percentile(arcs, 90)),
            "n_above_min_arc": int((arcs >= th.min_residual_arc_days).sum()),
            "min_residual_arc_days": float(th.min_residual_arc_days),
        })
    if apps.size:
        cov.update({"apparitions_median": float(np.median(apps)),
                    "apparitions_max": int(apps.max()),
                    "n_multi_apparition": int((apps >= 2).sum())})
    cov["law_discrimination_available"] = bool(
        arcs.size and (arcs >= th.min_residual_arc_days).sum() > 0
        and apps.size and (apps >= 2).sum() > 0)
    if not cov["law_discrimination_available"]:
        cov["note"] = (
            "no object has both a baseline above the minimum and two apparitions, "
            "so the heliocentric-distance law test -- the discriminant that "
            "separates an engineered acceleration from a dark comet, and the "
            "channel's central novelty claim -- could not run on ANY object.  An "
            "empty candidate list from this run is a statement about the survey's "
            "age, not about the solar system.")
    rec["coverage"] = cov

    # Funnel after both paths.
    funnel_b: dict = {"n_shortlist": len(shortlist), "n_analysed": n_done}
    for tier in ("untestable", "ordinary", "watch", "interest", "candidate"):
        funnel_b[f"n_{tier}"] = sum(1 for r in recs if r.tier == tier)
    rec["funnel_final"] = funnel_b
    rec["calls"] = sso.calls
    rec["verdict"] = "OK"

    # -- write ------------------------------------------------------------
    df = pd.DataFrame([r.as_dict() for r in recs])
    for col in ("fit_reasons", "reasons"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: "|".join(v) if isinstance(v, list) else v)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "objects.csv", index=False)
    _write_json(out / "residual_tests.json", residual_tests)
    checkpoint()
    print(f"[loom] screen verdict=OK n_orbits={rec['n_orbits']} "
          f"shortlist={len(shortlist)} analysed={n_done} "
          f"candidates={funnel_b['n_candidate']}")
    return rec


def _none_or_float(v):
    return None if v is None else float(v)


def _none_or_int(v):
    return None if v is None else int(v)


def _control_keys(summaries: list[dict], by_key: dict) -> list:
    """Keys of every positive-control object that has a residual aggregate.

    Matched on the screened record's designation, which is the *unpacked* form —
    the detections carry the packed form, and matching that against a control list
    written as "2020 SO" finds nothing.
    """
    idx = control_index()
    out: list = []
    for s in summaries:
        key = s.get("ssobjectid")
        if key is None:
            continue
        rec = by_key.get(str(key))
        if rec is None or not rec.designation:
            continue
        if normalise_designation(rec.designation) in idx:
            out.append(key)
    return out


def _shortlist(summaries: list[dict], dt_seconds: float, size: int) -> list:
    """Rank objects by timing-corrected mean offset, on the best column available.

    Prefers ``mean_along`` — the signed along-track mean is the physically right
    quantity — and falls back to ``mean_offset``, the scalar magnitude, which is the
    only one this mirror populates (measured 2026-07-30: along/cross are NULL for
    all 961,558 rows).  Without the fallback the shortlist would be empty and the
    channel would report a clean null having screened nothing, which is the worst
    outcome available.

    Correcting for the fitted common timing offset *before* ranking matters where
    ``ephrate`` carries a measurement: an uncorrected ranking is dominated by
    whichever objects were moving fastest, which is a property of the survey's
    cadence and of where the object was in its apparition, not of the object.  In
    this mirror ``ephrate`` is identically zero, so the correction is a no-op and
    the ranking is on raw magnitude — which must be remembered when reading it,
    because it means the shortlist is not timing-decontaminated.
    """
    scored: list[tuple[float, object]] = []
    dt = dt_seconds if math.isfinite(_f(dt_seconds)) else 0.0
    for s in summaries:
        key = s.get("ssobjectid")
        if key is None:
            continue
        value = _f(s.get("mean_along"))
        if not math.isfinite(value):
            value = _f(s.get("mean_offset"))
        if not math.isfinite(value):
            continue
        rate = _fz(s.get("mean_ephrate"))
        corrected = value - (rate * dt / 60.0 if math.isfinite(rate) else 0.0)
        scored.append((abs(corrected), key))
    scored.sort(key=lambda t: -t[0])
    return [k for _, k in scored[:int(size)]]


# ---------------------------------------------------------------------------
# per-object residual analysis
# ---------------------------------------------------------------------------
def analyse_series(rows: list[dict], th: Thresholds, n_null: int = 500) -> dict:
    """Run every residual test on one object's detection series.

    The astrometric uncertainty per detection is not in ``lsst_ss_detection``, so
    it is taken as the LSST single-epoch specification with the documented
    systematic floor added in quadrature, and the *scatter of the series itself*
    is used as a floor on that — an object whose residuals scatter by 1 arcsec is
    not being measured to 10 mas, whatever the specification says, and using the
    specification alone would inflate every significance in the channel.
    """
    mjd = np.array([_f(r.get("mjd")) for r in rows])
    along = np.array([_f(r.get("ephoffsetalongtrack")) for r in rows])
    cross = np.array([_f(r.get("ephoffsetcrosstrack")) for r in rows])
    # `_fz`: `ephrate` is identically zero for every solar-system detection in this
    # mirror, so reading it as a measured rate would make the timing-degeneracy
    # regression fit a column of zeros and report a confident zero offset.
    rate = np.array([_fz(r.get("ephrate")) for r in rows])
    helio = np.array([_f(r.get("heliorange")) for r in rows])
    topo = np.array([_f(r.get("toporange")) for r in rows])
    ra = np.array([_f(r.get("ra")) for r in rows])
    dec = np.array([_f(r.get("dec")) for r in rows])

    out: dict = {"n_rows": len(rows)}

    # The decomposition the mirror does not carry.  Measured 2026-07-30:
    # ephoffsetalongtrack/crosstrack are NULL for all 961,558 solar-system
    # detections, so where the columns are absent the split is RECONSTRUCTED from
    # the observed position, the predicted position, and the object's own track
    # direction between neighbouring epochs (see residuals.decompose_offset).
    if not np.any(np.isfinite(along)):
        eph_ra = np.array([_f(r.get("ephra")) for r in rows])
        eph_dec = np.array([_f(r.get("ephdec")) for r in rows])
        # `ephoffsetra`/`ephoffsetdec` ARE populated (measured 2026-07-30) and
        # reproduce `ephoffset` in quadrature exactly, so only the ROTATION is
        # missing, not the offset.  Use the survey's own vector where it exists and
        # fall back to differencing the positions, which was verified against those
        # columns to better than a milliarcsecond on live rows.
        off_ra = np.array([_f(r.get("ephoffsetra")) for r in rows])
        off_dec = np.array([_f(r.get("ephoffsetdec")) for r in rows])
        used_cols = bool(np.any(np.isfinite(off_ra)) and np.any(np.isfinite(off_dec)))
        along, cross, total = decompose_offset(mjd, ra, dec, eph_ra, eph_dec,
                                               off_ra=off_ra, off_dec=off_dec)
        out["decomposition"] = ("rotated_from_ephoffset_ra_dec" if used_cols
                                else "reconstructed_from_positions_and_track")
        # Free validation: the reconstructed magnitude must agree with the alert's
        # own `ephoffset`, which IS populated.  A disagreement means the frame or
        # the sign convention is wrong and nothing downstream should be believed.
        col_total = np.array([_f(r.get("ephoffset")) for r in rows])
        both = np.isfinite(total) & np.isfinite(col_total) & (col_total > 0)
        if both.sum() >= 3:
            resid = np.abs(total[both] - col_total[both])
            out["reconstruction_check"] = {
                "n": int(both.sum()),
                "median_abs_difference_arcsec": float(np.median(resid)),
                "median_ephoffset_arcsec": float(np.median(col_total[both])),
                "agrees": bool(np.median(resid)
                               <= 0.1 * max(float(np.median(col_total[both])), 1e-6)),
            }
            if not out["reconstruction_check"]["agrees"]:
                out["verdict"] = "RECONSTRUCTION_DISAGREES_WITH_EPHOFFSET"
                return out
        else:
            out["reconstruction_check"] = {"n": int(both.sum()),
                                           "agrees": None,
                                           "reason": "too_few_epochs_to_validate"}
    else:
        out["decomposition"] = "from_alert_columns"

    good = np.isfinite(mjd) & np.isfinite(along)
    out["n_usable"] = int(good.sum())
    if good.sum() < th.min_detections_for_drift:
        out["verdict"] = "TOO_FEW_DETECTIONS"
        return out

    # Only meaningful where `ephrate` carries a measurement; NaN otherwise, which
    # `assign_tier` reads as "this veto could not be evaluated" rather than "passed".
    out["timing_correlation"] = per_object_rate_correlation(along, rate)

    # THE PER-POINT UNCERTAINTY, and it must not be the standard error of the mean.
    #
    # The first working run used `scatter / sqrt(n)` here.  That is the uncertainty
    # on the MEAN, not on a single epoch, so for an object with 25 detections it
    # understated the per-point error five-fold, inflating every acceleration
    # signal-to-noise by five and every delta-chi-squared by twenty-five.  It
    # promoted 150 objects to `interest` and put two population statistics at the
    # randomisation floor.  It also injected an explicit n-dependence into the
    # selection, which is precisely the "ranks objects by how well they were
    # observed" failure the channel is built to avoid.
    #
    # The honest estimate is two-pass: fit with the instrumental floor, then take
    # the RMS of the residuals ABOUT THE FITTED MODEL -- not about the mean, which
    # would absorb the very trend being measured -- and refit with that.  Anything
    # the model does not explain is error, whatever its origin, and unmodelled
    # perturbers and photocentre wobble are real contributors no formal error
    # carries.
    spec = math.sqrt(0.010 ** 2 + 0.007 ** 2)
    topo_use = usable_range(topo)
    helio_use = usable_range(helio)
    out["n_with_usable_geometry"] = int(np.isfinite(topo_use).sum())
    along_km = arcsec_to_km(along, topo_use)

    sigma_km = np.abs(arcsec_to_km(np.full_like(along, spec), topo_use))
    first = drift_fit(mjd, along_km, sigma_km)
    sigma_arcsec = spec
    if first.get("verdict") == "OK" and math.isfinite(_f(first.get("chi2_reduced_quadratic"))):
        # sqrt(reduced chi-squared) is exactly the factor by which the assumed
        # error is too small, given the fitted model.  Never allowed below the
        # instrumental floor: an object whose residuals happen to sit inside the
        # spec has not been measured better than the instrument can measure.
        scale = max(math.sqrt(max(_f(first["chi2_reduced_quadratic"]), 0.0)), 1.0)
        sigma_arcsec = spec * scale
        sigma_km = sigma_km * scale
        out["sigma_inflation_from_fit"] = scale
    out["sigma_arcsec_used"] = sigma_arcsec
    out["residual_scatter_arcsec"] = (float(np.nanstd(along[good]))
                                      if good.sum() > 2 else float("nan"))

    out["drift"] = drift_fit(mjd, along_km, sigma_km)
    out["law"] = law_discrimination(mjd, along_km, sigma_km, helio_use)
    out["breakpoint"] = breakpoint_scan(mjd, along_km, sigma_km, n_null=n_null)
    out["sky_along"] = sky_coherence(ra, dec, along)
    out["sky_cross"] = sky_coherence(ra, dec, cross)
    # Structure, not amplitude.  A residual of 0.1-1 arcsec is routine -- it is
    # Rubin's 10 mas position minus a prediction from an orbit fitted to decades of
    # heterogeneous astrometry with star-catalogue biases reaching 175 mas.  These
    # two ask whether the residual has the *geometry* and the *time behaviour* an
    # acceleration produces, which is the only question the amplitude cannot answer.
    out["geometry"] = along_cross_partition(along, cross)
    out["apparitions"] = apparition_trend(mjd, along)
    out["mjd_min"] = float(np.nanmin(mjd[good]))
    out["mjd_max"] = float(np.nanmax(mjd[good]))
    out["verdict"] = "OK"
    return out


def _apply_series(rec, tests: dict, th: Thresholds) -> None:
    """Fold one object's residual tests into its screened record."""
    rec.path = "mpc_orbits+residuals" if rec.path == "mpc_orbits" else "residuals"
    rec.timing_correlation = _f(tests.get("timing_correlation"))
    drift = tests.get("drift") or {}
    rec.accel_au_day2_residual = _f(drift.get("accel_au_per_day2"))
    rec.accel_snr_residual = _f(drift.get("accel_snr"))
    if math.isfinite(rec.h) and math.isfinite(rec.accel_au_day2_residual):
        rec.ratio_hard_residual = float(ceiling_ratio(
            rec.h, rec.accel_au_day2_residual, albedo=th.albedo_generous,
            rho_kg_m3=th.rho_generous_kg_m3, epsilon=th.epsilon_hard))
    rec.mjd_min = _f(tests.get("mjd_min"))
    rec.mjd_max = _f(tests.get("mjd_max"))
    law = tests.get("law") or {}
    if law.get("verdict") == "LAW_PREFERRED":
        rec.best_law = law.get("best_law")
        # The SCALED statistic: the per-detection astrometric error is a floor and
        # is underestimated for some objects, so an unscaled delta-chi-squared would
        # show a decisive law preference for anything with a slightly wrong sigma.
        rec.delta_chi2_law = _f(law.get("delta_chi2_scaled"))
    geom = tests.get("geometry") or {}
    rec.along_cross_power_ratio = _f(geom.get("power_ratio"))
    rec.mean_along_arcsec = _f(geom.get("mean_along_arcsec"))
    rec.mean_cross_arcsec = _f(geom.get("mean_cross_arcsec"))
    app = tests.get("apparitions") or {}
    rec.n_apparitions = int(app.get("n_apparitions") or 0)
    rec.apparition_spearman = _f(app.get("spearman"))
    bp = tests.get("breakpoint") or {}
    rec.breakpoint_p = _f(bp.get("p_value"))
    rec.breakpoint_mjd = _f(bp.get("break_mjd"))
    sky = tests.get("sky_along") or {}
    rec.sky_variance_explained = _f(sky.get("variance_explained_by_sky_bin"))
    rec.residual_arc_days = _f(drift.get("arc_days"))
    # A residual-path acceleration is only believable above a signal-to-noise
    # floor; below it the quadratic coefficient is fitting noise and must not
    # contribute to the tier.
    if (math.isfinite(rec.accel_snr_residual)
            and rec.accel_snr_residual < th.min_accel_snr):
        rec.ratio_hard_residual = float("nan")
        rec.reasons.append(f"residual_accel_snr_{rec.accel_snr_residual:.1f}_"
                           f"below_{th.min_accel_snr}")
    # BASELINE, which is the binding constraint while the survey is young.  A
    # quadratic fitted over a two-week series has no leverage on curvature, so the
    # fitted acceleration is an extrapolation: the first clean run promoted four
    # objects with 2-to-29-day series and implied accelerations 10^4 to 10^8 times
    # the momentum ceiling.  Those are fit blow-ups, not candidates, and the
    # correct response is to record the object as UNTESTABLE on this path rather
    # than to score it either way.
    if (not math.isfinite(rec.residual_arc_days)
            or rec.residual_arc_days < th.min_residual_arc_days):
        rec.ratio_hard_residual = float("nan")
        rec.reasons.append(
            f"residual_series_spans_{rec.residual_arc_days:.0f}d_below_"
            f"{th.min_residual_arc_days:.0f}d__acceleration_not_constrained")
    if rec.n_apparitions < th.min_apparitions_for_promotion:
        rec.ratio_hard_residual = float("nan")
        rec.reasons.append(
            f"only_{rec.n_apparitions}_apparition__an_acceleration_cannot_be_"
            f"separated_from_an_orbit_fit_error")


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------
def assess(cfg=None, out_dir: str | Path | None = None) -> dict:
    """Population-structure tests and positive-control validation.  Offline.

    This is where the channel actually decides.  A per-object anomaly is a
    contamination problem; the question replication asks is whether the *set* of
    anomalies has structure no natural population produces, and that question is
    immune to any contaminant that does not itself cluster in orbital elements and
    orbital pole.
    """
    conf = load_loom_config(cfg)
    root = Path(cfg.root) if cfg is not None else _repo_root()
    out = Path(out_dir) if out_dir else root / conf["report"]["results_dir"]
    th = thresholds_from_config(conf)
    aconf = conf["assess"]
    rec: dict = {"assessed_at_utc": _utc(), "verdict": "NOT_RUN"}

    path = out / "objects.csv"
    if not path.exists():
        rec["verdict"] = "NO_SCREENED_SAMPLE"
        rec["note"] = f"{path} does not exist; run the screen stage first"
        _write_json(out / "assessment.json", rec)
        print(f"[loom] assess verdict={rec['verdict']} ({rec['note']})")
        return rec
    df = pd.read_csv(path)
    rec["n_objects"] = int(len(df))
    rows = df.to_dict("records")

    # The empirical envelope, as a cross-check on the momentum ceiling.  A
    # survey-wide calibration error in H or in the offset units moves the
    # empirical envelope and leaves the ceiling alone, so a disagreement between
    # the two is diagnostic rather than an inconvenience.
    h = df["h"].to_numpy(dtype=float) if "h" in df else np.array([])
    a2 = (df["a2_au_day2"].to_numpy(dtype=float) if "a2_au_day2" in df
          else np.array([]))
    env = fit_envelope(h, a2, quantile=float(aconf["empirical_envelope_quantile"]),
                       min_per_bin=int(aconf["empirical_envelope_min_per_bin"]))
    rec["empirical_envelope"] = {
        "ok": env.ok, "reason": env.reason, "notes": env.notes,
        "quantile": env.quantile, "edges": env.edges.tolist(),
        "level": env.level.tolist(), "n_per_bin": env.n_per_bin.tolist()}
    if env.ok:
        ratio = anomaly_ratio(h, a2, env)
        rec["empirical_envelope"]["n_above"] = int(np.nansum(ratio > 1.0))
        with np.errstate(invalid="ignore"):
            hard = df.get("ratio_hard")
            if hard is not None:
                hard = hard.to_numpy(dtype=float)
                both = np.isfinite(ratio) & np.isfinite(hard)
                rec["empirical_envelope"]["agreement_with_ceiling"] = (
                    float(np.mean((ratio[both] > 1.0) == (hard[both] > 1.0)))
                    if both.sum() else None)

    # The anomalous set: above the hard ceiling on either path, or anomalous in
    # area-to-mass ratio.  Tier is not used as the mask, because the tier ladder
    # already applies the systematic vetoes and the population test should see the
    # magnitude-selected set -- the vetoes are per-object arguments, and folding
    # them into the mask would make the population statistic conditional on them.
    mask = np.zeros(len(df), dtype=bool)
    for col, thresh in (("ratio_hard", 1.0), ("ratio_hard_residual", 1.0),
                        ("amr_ratio", float(th.min_amr_ceiling_ratio))):
        if col in df:
            v = df[col].to_numpy(dtype=float)
            mask |= np.isfinite(v) & (v >= thresh)
    rec["n_anomalous"] = int(mask.sum())

    # Is the anomaly statistic just a measure of how badly each orbit is observed?
    # This runs BEFORE the population tests and can invalidate them on its own: a
    # statistic that ranks objects by arc length has manufactured any structure the
    # population tests then find, because arc length is not uniform across the belt.
    score = (df["score"].to_numpy(dtype=float) if "score" in df
             else np.array([]))
    quality = {name: df[name].to_numpy(dtype=float)
               for name in ("arc_days", "n_opp", "normalized_rms", "n_detections",
                            "h")
               if name in df}
    # `h` is reported but does NOT gate: the momentum ceiling is a function of H by
    # construction, so an anomalous set being systematically faint is the
    # signature's own shape, and the matched null already stratifies on it.
    rec["quality_independence"] = quality_independence(
        score, quality,
        gate_keys=[k for k in quality if k != "h"])
    qi = rec["quality_independence"]
    qthresh = float(conf["assess"]["max_quality_correlation"])
    if qi.get("verdict") == "OK" and qi.get("max_abs_correlation", 0.0) > qthresh:
        rec["quality_independence"]["warning"] = (
            f"the anomaly score correlates with {qi['max_correlated_with']} at "
            f"rho={qi['max_abs_correlation']:.2f} (threshold {qthresh}); "
            f"the channel is partly ranking "
            f"objects by observation quality and no population structure found "
            f"below is interpretable until that is removed")

    labels = _labels_from_frame(df)
    rec["replication"] = replication_tests(
        rows, mask, labels, n_null=int(th.population_n_null),
        collisional_slope=float(aconf["collisional_slope"]),
        seed=int(th.population_seed))

    # Positive controls.  Reported whatever the outcome, including the common and
    # honest case of no control object being present at all.
    flag = np.zeros(len(df), dtype=bool)
    if "tier" in df:
        flag = df["tier"].astype(str).isin(("interest", "candidate")).to_numpy()
    for i, r in enumerate(rows):
        r["flagged"] = bool(flag[i])
    rec["controls"] = validate(rows, score_key="score", flagged_key="flagged")

    tiers = (df["tier"].astype(str).value_counts().to_dict()
             if "tier" in df else {})
    rec["tiers"] = {k: int(v) for k, v in tiers.items()}
    rec["verdict"] = rec["replication"].get("verdict", "NO_STRUCTURE")
    # A detection cannot stand on a statistic that tracks observation quality, so
    # the verdict is downgraded rather than reported alongside a caveat nobody
    # reads.  The underlying p-values stay in the record.
    if (rec["verdict"] == "REPLICATION_STRUCTURE_DETECTED"
            and "warning" in rec["quality_independence"]):
        rec["verdict"] = "STRUCTURE_CONFOUNDED_BY_OBSERVATION_QUALITY"
    if (rec["verdict"] == "REPLICATION_STRUCTURE_DETECTED"
            and rec["controls"]["verdict"] == "SCREEN_INSENSITIVE"):
        rec["verdict"] = "STRUCTURE_FOUND_BUT_SCREEN_FAILS_ITS_POSITIVE_CONTROL"
    _write_json(out / "assessment.json", rec)
    print(f"[loom] assess verdict={rec['verdict']} n_anomalous={rec['n_anomalous']} "
          f"controls={rec['controls']['verdict']}")
    return rec


def calibrate(cfg=None, out_dir: str | Path | None = None) -> dict:
    """Measure the ceiling's realised efficiency on the whole published population.

    Runner-only.  Independent of how old the Rubin survey is, which is why it can
    run while the residual path waits for a second apparition.

    Writes ``calibration.json``: the distribution of realised thermal-recoil
    efficiency over every asteroid with a fitted ``A2`` in JPL's Small-Body
    Database, and a measurement — not an assumption — of the unit of
    ``lsst_mpc_orbits.yarkovsky``.
    """
    conf = load_loom_config(cfg)
    root = Path(cfg.root) if cfg is not None else _repo_root()
    out = Path(out_dir) if out_dir else root / conf["report"]["results_dir"]
    rec: dict = {"calibrated_at_utc": _utc(), "verdict": "NOT_RUN",
                 "anchor_before": calibration_table()}

    def checkpoint() -> None:
        _write_json(out / "calibration.json", rec)

    checkpoint()
    fetched = fetch_sbdb()
    rec["sbdb"] = {k: v for k, v in fetched.items() if k != "rows"}
    rec["verdict"] = fetched["verdict"]
    checkpoint()
    if not fetched["rows"]:
        rec["note"] = ("JPL SBDB returned no rows under any candidate constraint; "
                       "the ceiling remains anchored on the ten objects in "
                       "`anchor_before` and the yarkovsky unit remains assumed")
        checkpoint()
        print(f"[loom] calibrate verdict={rec['verdict']}")
        return rec

    # The efficiency distribution, under two density assumptions.  Density is the
    # one input that is neither measured nor bounded for most objects, so the
    # result is reported across the plausible range rather than at a point: if the
    # conclusion flips between them, it is a statement about density.
    # ASTEROIDS AND COMETS SEPARATELY, because they are bound by different physics.
    # A comet accelerates by shedding mass, so it is not subject to the radiation
    # momentum budget at all and exceeds it by orders of magnitude; an asteroid
    # accelerates by re-radiating sunlight, so it is.  Reporting them together gave
    # a headline of "92 objects above the hard ceiling", which reads as the gate
    # failing when every one of those 92 was a periodic comet -- i.e. the gate
    # separating the two populations exactly as designed.
    rec["epsilon"] = {}
    for kind in ("asteroid", "comet"):
        for label, rho in (("rho_2000", 2000.0), ("rho_1000_generous", 1000.0)):
            rec["epsilon"][f"{kind}_{label}"] = summarise_epsilon(
                fetched["rows"], rho_kg_m3=rho, kind=kind).as_dict()
    checkpoint()

    # And the unit check, against the mirror's own twelve objects.
    aconf = conf["acquire"]
    try:
        sso = AlerceSSO(aconf["alerce_tap_url"], timeout=float(aconf["timeout_s"]))
        mirror = sso.orbits(require_nongrav=True)
        rec["mirror_nongrav_rows"] = len(mirror.rows)
        rec["yarkovsky_unit"] = verify_yarkovsky_unit(mirror.rows, fetched["rows"])
    except Exception as exc:                                  # noqa: BLE001
        rec["yarkovsky_unit"] = {"verdict": "MIRROR_UNREACHABLE",
                                 "error": f"{type(exc).__name__}: {exc}"[:400]}
    checkpoint()

    eps = rec["epsilon"]["asteroid_rho_2000"]
    com = rec["epsilon"]["comet_rho_2000"]
    if eps.get("ok"):
        rec["verdict"] = "OK"
        rec["headline"] = (
            f"ASTEROIDS with a fitted A2 (n={eps['n']}): realised thermal-recoil "
            f"efficiency median {eps['quantiles'].get('p50'):.3g}, "
            f"p99 {eps['quantiles'].get('p99'):.3g}, "
            f"max {eps['quantiles'].get('max'):.3g}; "
            f"{eps['n_above_hard_1.0']} above the hard ceiling.  "
            f"COMETS (n={com.get('n', 0)}): "
            f"{com.get('n_above_hard_1.0', 0)} above it -- which is the ceiling "
            f"working, since a comet accelerates by shedding mass and is not bound "
            f"by the radiation budget at all.")
    checkpoint()
    print(f"[loom] calibrate verdict={rec['verdict']} "
          f"n_sbdb={fetched['n_rows']} "
          f"unit={rec.get('yarkovsky_unit', {}).get('verdict')}")
    return rec


def litcheck(cfg=None, out_dir: str | Path | None = None,
             names: list[str] | None = None) -> dict:
    """Are the flagged exceedances already explained in the literature?  Runner-only.

    The step that decides whether the vetted exceedances mean anything.  They are
    "unexplained" only relative to a seven-object list from 2023, and that
    population has since been extended — so an object already catalogued as a dark
    comet is explained by hidden outgassing and is not a lead.

    Reads the survivors from ``calibration.json`` unless ``names`` is given, and
    writes ``litcheck.json`` after every query so a job timeout leaves the answers
    already obtained on disk.
    """
    conf = load_loom_config(cfg)
    root = Path(cfg.root) if cfg is not None else _repo_root()
    out = Path(out_dir) if out_dir else root / conf["report"]["results_dir"]
    rec: dict = {"checked_at_utc": _utc(), "verdict": "NOT_RUN"}

    if names is None:
        path = out / "calibration.json"
        if not path.exists():
            rec["verdict"] = "NO_CALIBRATION"
            rec["note"] = f"{path} does not exist; run loom-calibrate first"
            _write_json(out / "litcheck.json", rec)
            print(f"[loom] litcheck verdict={rec['verdict']}")
            return rec
        cal = json.loads(path.read_text())
        names = [s["name"] for s in
                 (cal.get("epsilon", {}).get("asteroid_rho_2000", {})
                  .get("survivors") or [])]
    rec["objects_checked"] = list(names)
    if not names:
        rec["verdict"] = "NOTHING_TO_CHECK"
        _write_json(out / "litcheck.json", rec)
        print("[loom] litcheck verdict=NOTHING_TO_CHECK")
        return rec

    progress: dict = {}
    rec["progress"] = progress

    def _record(key, value):
        progress[key] = {k: v for k, v in value.items() if k != "text"}
        _write_json(out / "litcheck.json", rec)

    # Build the corpus first: eight hand-picked papers only tests whether an
    # object is in the papers I thought of.  Every paper the topic searches return
    # is then full-texted, because a designation in a table is invisible to the
    # metadata search that found the paper.
    corpus = build_corpus(on_result=_record)
    rec["corpus"] = {"n_papers": corpus["n_papers"], "queries": corpus["queries"]}
    _write_json(out / "litcheck.json", rec)
    result = check_objects(names, on_result=_record, extra_ids=corpus["ids"])
    rec.update(result)
    rec["verdict"] = result["verdict"]
    _write_json(out / "litcheck.json", rec)
    print(f"[loom] litcheck verdict={rec['verdict']} "
          f"explained={result['n_explained']}/{result['n_objects']}")
    return rec


def _labels_from_frame(df: pd.DataFrame) -> np.ndarray:
    """Matched-null strata from the screened table, mirroring ``covariate_labels``."""
    cols = []
    for name in ("h", "arc_days", "n_opp", "normalized_rms", "n_detections"):
        if name not in df:
            continue
        v = df[name].to_numpy(dtype=float)
        fin = np.isfinite(v)
        if fin.sum() < 5:
            continue
        edges = np.nanquantile(v[fin], [0.25, 0.5, 0.75])
        idx = np.where(fin, np.digitize(v, edges), -1)
        cols.append(idx.astype(int))
    if not cols:
        return np.zeros(len(df), dtype=int)
    lab = np.zeros(len(df), dtype=np.int64)
    for c in cols:
        lab = lab * (int(c.max()) + 2) + (c + 1)
    return lab
