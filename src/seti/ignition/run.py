"""Stage orchestration for IGNITION.  Writes ``results/ignition/``.

Stages
------
``probe``    one minimal live call per route: the AllWISE mirror's column
             names, the Gaia x AllWISE join (``TOP 5``) tried as each candidate
             query shape in turn, a NEOWISE cone on a resolved star, and the
             ``TAP_UPLOAD`` join on two stars.  Every call is time-boxed
             (``config/ignition.yaml`` -> ``probe``: ``budget_s`` per shape,
             ``total_budget_s`` for the stage); a shape that overruns is
             recorded ``TIMED_OUT`` with its elapsed seconds and the probe moves
             on.  Writes ``probe.json`` with ``gaia_shapes`` (each shape's
             status, verbatim error and the exact ADQL sent),
             ``gaia_shape_working`` --- which the ``sample`` stage then reuses
             --- and ``neowise_route_recommended``.  The workflow commits it:
             run 34787803862 went green, took 25 minutes and committed nothing.
``sample``   the parent sample (``parent.parquet`` + ``sample.json`` with the
             archive ``COUNT(*)`` denominator and the subsample fraction).
``acquire``  shard ``i/n`` of the parent: NEOWISE frames -> cleaning -> epochs,
             checkpointed to ``epochs_s{i}of{n}.csv`` /
             ``neowise_stars_s{i}of{n}.csv`` / ``acquire_s{i}of{n}.json``.
``screen``   per shard: the rise statistics, the two-band rule, the injection
             sensitivity.  Writes ``stars_s{i}of{n}.csv`` + ``screen_s{i}of{n}.json``.
``assess``   merge every shard, the contaminant ladder, ``candidates.csv``,
             ``summary.json``.

Verdict vocabulary (``summary.json["verdict"]``)
------------------------------------------------
``NO_DATA_REACHED``        nothing usable came back from the archives (the
                           ``reason`` field says which archive and how)
``NO_IGNITION_CANDIDATE``  stars were screened; none passed --- a count over
                           ``n_screened``, never an occurrence limit
``IGNITION_CANDIDATES``    >= 1 star passes the rise test and the ladder

The lesson written into ``assess``: VIGIL's ``summary.json`` reported
``NO_DATA_REACHED`` while its eight ``field_summary.json`` files said
``SEARCHED`` --- the aggregator ran on a checkout that lacked the per-field
artefacts and reported their absence as a science verdict.  Here ``assess``
records ``shards.expected`` against ``shards.found``; a shortfall is a
``DEGRADED`` prefix and zero-found is ``NO_DATA_REACHED`` with reason
``no_shard_outputs_found`` (never a clean null), and the workflow fails when
that happens after an acquire matrix actually ran.
"""

from __future__ import annotations

import argparse
import glob
import json
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .acquire import (
    DEFAULT_ACQUIRE,
    EpochStore,
    acquire_stars,
    epochs_to_series,
)
from .rise import DEFAULT_RISE, assess_series, sensitivity_from_injections
from .sample import (
    DEFAULT_SAMPLE,
    QUERY_FAILED,
    QUERY_TIMED_OUT,
    ROUTE_ESA,
    ROUTE_VIZIER,
    SHAPES,
    QueryTimeout,
    build_query,
    call_with_timeout,
    fetch_parent,
    run_gaia_query,
    unwrap_result,
)
from .vet import DEFAULT_VET, load_optical_series, summarise, vet_star

VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_NONE = "NO_IGNITION_CANDIDATE"
VERDICT_CANDIDATES = "IGNITION_CANDIDATES"
DEGRADED = "DEGRADED"
STAGES = ("probe", "sample", "acquire", "screen", "assess")

#: The probe's wall-clock discipline.  Run 34787803862 spent 1,490 s on four
#: retries of a query that could not succeed; a broken plan must cost minutes.
DEFAULT_PROBE: dict = {
    "budget_s": 480.0,              # per candidate query shape
    "total_budget_s": 1500.0,       # the whole probe stage
    "columns_timeout_s": 120.0,     # the AllWISE column peek
    "neowise_timeout_s": 300.0,     # each NEOWISE route test
    "cap": 5,                       # TOP n for the join test
    "shapes": list(SHAPES),
    "vizier_timeout_s": 300.0,      # the SECOND route's probe (VizieR ASU)
}

DEFAULTS: dict = {
    "sample": dict(DEFAULT_SAMPLE),
    "acquire": dict(DEFAULT_ACQUIRE),
    "rise": dict(DEFAULT_RISE),
    "vet": dict(DEFAULT_VET),
    "probe": dict(DEFAULT_PROBE),
    "sensitivity": {"amps_mag": [0.1, 0.2, 0.4], "over_yr": 10.0, "max_stars": 200},
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


def load_ignition_config(path: Path | str | None = None) -> dict:
    """``config/ignition.yaml`` over :data:`DEFAULTS`; a missing file degrades."""
    try:
        import yaml
        if path is None:
            path = Path(__file__).resolve().parents[3] / "config" / "ignition.yaml"
        path = Path(path)
        if not path.exists():
            return _deep_update(DEFAULTS, {})
        return _deep_update(DEFAULTS, yaml.safe_load(path.read_text()) or {})
    except Exception as exc:                           # noqa: BLE001
        print(f"[ignition] config not loaded ({exc!r}); using defaults")
        return _deep_update(DEFAULTS, {})


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return None if not np.isfinite(v) else v
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


def _budget(want: float | None, left: float | None) -> float | None:
    """The smaller of a step's own timeout and what is left of the stage budget."""
    want = float(want) if want else None
    if left is None:
        return want
    left = max(float(left), 0.0)
    return left if want is None else min(want, left)


def _tag(shard: int, n_shards: int) -> str:
    return f"s{int(shard)}of{int(n_shards)}"


def parse_shard(s: str | None) -> tuple[int, int]:
    """``"2/8"`` -> (2, 8); ``None`` / ``""`` -> (0, 1)."""
    if not s:
        return 0, 1
    a, b = str(s).split("/")
    i, n = int(a), int(b)
    if n < 1 or not 0 <= i < n:
        raise SystemExit(f"bad --shard {s!r}: want i/n with 0 <= i < n")
    return i, n


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def _probe_query(query_fn, adql: str, timeout_s: float | None,
                 label: str) -> tuple[pd.DataFrame, dict]:
    """One time-boxed archive call.  Never raises, never hangs, always records.

    The returned record carries ``status`` (``OK`` / ``QUERY_RETURNED_ZERO_ROWS``
    / ``QUERY_FAILED`` / ``TIMED_OUT``), ``elapsed_s``, the verbatim ``error``,
    and --- when the default transport ran it --- which queue served it.
    """
    t0 = _time.monotonic()
    meta: dict = {"timeout_s": (round(float(timeout_s), 1) if timeout_s else None)}

    def _default(q: str):
        return run_gaia_query(q, label=label, timeout_s=timeout_s, tag="ignition",
                              deadline=(t0 + float(timeout_s)) if timeout_s else None)

    try:
        res = call_with_timeout(query_fn or _default, adql, timeout_s)
    except QueryTimeout as exc:
        meta.update(status=QUERY_TIMED_OUT, error=repr(exc),
                    elapsed_s=round(_time.monotonic() - t0, 1))
        return pd.DataFrame(), meta
    except Exception as exc:                           # noqa: BLE001
        meta.update(status=QUERY_FAILED, error=repr(exc),
                    elapsed_s=round(_time.monotonic() - t0, 1))
        return pd.DataFrame(), meta
    df, qrec = unwrap_result(res)
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    meta["elapsed_s"] = round(_time.monotonic() - t0, 1)
    for k in ("transport", "queue", "attempts"):
        if qrec.get(k) is not None:
            meta[k] = qrec[k]
    if qrec.get("status") in (QUERY_FAILED, QUERY_TIMED_OUT):
        meta.update(status=qrec["status"], error=qrec.get("error"))
        return pd.DataFrame(), meta
    meta.update(status=("OK" if len(df) else "QUERY_RETURNED_ZERO_ROWS"), n_rows=int(len(df)))
    return df, meta


def _timeboxed(fn, timeout_s: float | None, label: str):
    """Call ``fn()`` under the same time-box; return ``(value, record_or_None)``."""
    t0 = _time.monotonic()
    try:
        val = call_with_timeout(lambda _: fn(), None, timeout_s)
    except QueryTimeout as exc:
        return None, {"status": QUERY_TIMED_OUT, "error": repr(exc), "label": label,
                      "elapsed_s": round(_time.monotonic() - t0, 1),
                      "timeout_s": (round(float(timeout_s), 1) if timeout_s else None)}
    except Exception as exc:                           # noqa: BLE001
        return None, {"status": QUERY_FAILED, "error": repr(exc), "label": label,
                      "elapsed_s": round(_time.monotonic() - t0, 1)}
    return val, None


def stage_probe(conf: dict, out: Path, *, query_fn=None, cone_fn=None,
                upload_fn=None, asu_fetch_fn=None) -> dict:
    """One minimal call per route; decides the acquire architecture.

    The Gaia join is tried as each of :data:`~seti.ignition.sample.SHAPES` in
    order, under a per-shape wall-clock budget, and the shape that answered is
    written to ``probe.json`` as ``gaia_shape_working`` so the sample stage does
    not re-derive it.  Every shape that did not answer keeps its verbatim error
    and the exact ADQL that was sent.  ``probe.json`` is written whatever
    happens --- a green run that commits nothing is a lost run.

    **BOTH parent routes are probed, always** --- the ESA archive first and then
    VizieR's non-TAP ASU interface, even when ESA answered --- so the next run's
    ``probe.json`` names the transport that works instead of re-deriving it.
    (Probing is not using: the sample stage's fallback still runs only where ESA
    fails.)  When ESA returns nothing and VizieR does, the NEOWISE route tests
    below are run on a VizieR-supplied star rather than skipped.
    """
    t_stage = _time.monotonic()
    pc = {**DEFAULT_PROBE, **(conf.get("probe") or {})}
    total_budget = float(pc.get("total_budget_s") or 0.0) or None
    shape_budget = float(pc.get("budget_s") or 0.0) or None
    deadline = (t_stage + total_budget) if total_budget else None

    def _left() -> float | None:
        return None if deadline is None else deadline - _time.monotonic()

    rep: dict = {"stage": "probe", "generated_utc": _now(),
                 "budget": {"per_shape_s": shape_budget, "total_s": total_budget}}
    sc = conf["sample"]

    # 1. AllWISE mirror column names (ext_flag vs ext_flg etc.).
    peek, meta = _probe_query(query_fn, "SELECT TOP 1 * FROM gaiadr1.allwise_original_valid",
                              _budget(pc.get("columns_timeout_s"), _left()), "allwise_columns")
    if meta["status"] in ("OK", "QUERY_RETURNED_ZERO_ROWS"):
        have = {str(c).lower() for c in peek.columns}
        rep["allwise_columns_seen"] = sorted(have)[:80]
        resolved = dict(sc.get("allwise_columns") or {})
        for logical, cands in {"ext_flag": ["ext_flag", "ext_flg"],
                               "cc_flags": ["cc_flags", "ccf"],
                               "ph_qual": ["ph_qual", "qph"],
                               "designation": ["designation", "allwise"]}.items():
            for cnd in cands:
                if cnd in have:
                    resolved[logical] = cnd
                    break
        rep["allwise_columns_resolved"] = resolved
        sc = {**sc, "allwise_columns": resolved}
    else:
        rep["allwise_columns_resolved"] = meta

    # 2. The join itself, TOP n, on the first unit --- one candidate shape at a time.
    fields = list(sc.get("fields") or [])
    unit = {"field": fields[0]} if (sc.get("mode", "fields") == "fields" and fields) else {}
    cap = int(pc.get("cap") or 5)
    shapes = [s for s in (pc.get("shapes") or SHAPES) if s in SHAPES] or list(SHAPES)
    shape_recs: list[dict] = []
    working: str | None = None
    df = pd.DataFrame()
    for sh in shapes:
        q = build_query(sc, cap=cap, shape=sh, **unit)
        entry: dict = {"shape": sh, "query": q[:4000]}
        left = _left()
        if left is not None and left <= 0:
            entry.update(status="NOT_ATTEMPTED",
                         reason=f"probe total budget of {total_budget:.0f} s exhausted",
                         elapsed_s=round(_time.monotonic() - t_stage, 1))
            shape_recs.append(entry)
            print(f"[ignition] probe: shape={sh} NOT_ATTEMPTED (budget exhausted)")
            continue
        d, meta = _probe_query(query_fn, q, _budget(shape_budget, left), f"gaia_join[{sh}]")
        entry.update(meta)
        shape_recs.append(entry)
        print(f"[ignition] probe: shape={sh} {entry['status']} "
              f"in {entry.get('elapsed_s')} s (queue={entry.get('queue')})")
        if entry["status"] == "OK":
            working, df = sh, d
            entry["columns"] = [str(c) for c in d.columns][:60]
            break
        # A shape that answered with zero rows has still answered; the next shape
        # is strictly more permissive, so it is worth the remaining budget.

    rep["gaia_shapes"] = shape_recs
    rep["gaia_shape_working"] = working
    rep["gaia_shapes_tried"] = [s["shape"] for s in shape_recs]
    ok = [s for s in shape_recs if s.get("status") == "OK"]
    zero = [s for s in shape_recs if s.get("status") == "QUERY_RETURNED_ZERO_ROWS"]
    rep["gaia_join"] = (ok or zero or shape_recs or [{"status": "NOT_ATTEMPTED"}])[0]

    # 2b. The SECOND route to the same parent sample: VizieR's non-TAP ASU
    # interface.  Probed on every run, whatever ESA did, because the point of a
    # probe is to say which transports are alive tonight.
    from .vizier_route import probe_route

    vz_val, vz_err = _timeboxed(
        lambda: probe_route(sc, fetch_fn=asu_fetch_fn, cap=cap),
        _budget(pc.get("vizier_timeout_s"), _left()), "vizier_asu")
    if vz_val is None:
        rep["vizier_asu"] = dict(vz_err or {"status": QUERY_FAILED, "error": "no record"})
        vz_frame = pd.DataFrame()
    else:
        rep["vizier_asu"], vz_frame = vz_val
    rep["vizier_asu"].setdefault("usable", False)
    vizier_ok = bool(rep["vizier_asu"].get("usable"))
    print(f"[ignition] probe: vizier_asu {rep['vizier_asu'].get('status')} "
          f"rows={rep['vizier_asu'].get('n_rows')}")
    if not len(df) and len(vz_frame):
        # ESA reached nothing; the NEOWISE tests below still deserve a real star.
        df = vz_frame.rename(columns={c: str(c).lower() for c in vz_frame.columns})
        rep["neowise_star_from_route"] = ROUTE_VIZIER
    elif len(df):
        rep["neowise_star_from_route"] = ROUTE_ESA

    # 3. NEOWISE per-star cone on a resolved star (a bare coordinate tests nothing).
    if cone_fn is None:
        from .acquire import fetch_neowise_cone as cone_fn
    nw_budget = _budget(pc.get("neowise_timeout_s"), _left())
    if len(df):
        s = df.iloc[0]
        r, err = _timeboxed(
            lambda: cone_fn(float(s["ra"]), float(s["dec"]), float(s.get("pmra", 0.0) or 0.0),
                            float(s.get("pmdec", 0.0) or 0.0),
                            radius_arcsec=float(conf["acquire"]["cone_radius_arcsec"])),
            nw_budget, "neowise_cone")
        rep["neowise_cone"] = err or (r.to_ledger() | {"source_id": str(s["source_id"])})
    else:
        rep["neowise_cone"] = {"status": "NOT_ATTEMPTED",
                               "reason": "no Gaia row to resolve "
                                         f"(no query shape answered: {rep['gaia_shapes_tried']})"}

    # 4. The TAP_UPLOAD join, two stars.
    if upload_fn is None:
        from .acquire import fetch_neowise_upload as upload_fn
    if len(df):
        r, err = _timeboxed(
            lambda: upload_fn(df.head(2),
                              radius_arcsec=float(conf["acquire"]["cone_radius_arcsec"])),
            _budget(pc.get("neowise_timeout_s"), _left()), "neowise_upload")
        rep["neowise_upload"] = err or r.to_ledger()
    else:
        rep["neowise_upload"] = {"status": "NOT_ATTEMPTED", "reason": "no Gaia row to upload"}

    cone_ok = rep["neowise_cone"].get("status") == "OK"
    up_ok = rep["neowise_upload"].get("status") == "OK"
    gaia_ok = bool(working)
    rep["neowise_route_recommended"] = ("upload" if up_ok else
                                        "field" if (cone_ok and fields) else
                                        "cone" if cone_ok else "none")
    # Which SOURCE the next run should expect the parent from.  ESA stays first
    # whenever it answers: it is authoritative and owns the in-archive match.
    rep["parent_route_recommended"] = (ROUTE_ESA if gaia_ok else
                                       ROUTE_VIZIER if vizier_ok else "none")
    rep["parent_routes_tried"] = [ROUTE_ESA, ROUTE_VIZIER]
    rep["verdict"] = ("ALL_ROUTES_REACHABLE" if (gaia_ok and cone_ok and up_ok) else
                      "GAIA_AND_NEOWISE_REACHABLE" if (gaia_ok and cone_ok) else
                      "GAIA_ONLY" if gaia_ok else
                      "VIZIER_PARENT_AND_NEOWISE" if (vizier_ok and cone_ok) else
                      "VIZIER_PARENT_ONLY" if vizier_ok else
                      "NEOWISE_ONLY" if cone_ok else VERDICT_NO_DATA)
    rep["elapsed_s"] = round(_time.monotonic() - t_stage, 1)
    _write(out / "probe.json", rep)
    print(f"[ignition] probe: {rep['verdict']}; route={rep['neowise_route_recommended']}; "
          f"parent_route={rep['parent_route_recommended']}; "
          f"gaia_shape={working}; {rep['elapsed_s']} s")
    return rep


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------
def stage_sample(conf: dict, out: Path, *, n_shards: int = 1, max_stars: int | None = None,
                 query_fn=None, mode: str | None = None, asu_fetch_fn=None,
                 vizier: bool = True) -> dict:
    sc = dict(conf["sample"])
    probe_p = out / "probe.json"
    shape = None
    probe_parent_route = None
    if probe_p.exists():
        try:
            pr = json.loads(probe_p.read_text())
            res = pr.get("allwise_columns_resolved")
            if isinstance(res, dict) and "status" not in res:
                sc["allwise_columns"] = res
            # The probe already paid for finding a plan that returns; use it.
            if pr.get("gaia_shape_working") in SHAPES:
                shape = pr["gaia_shape_working"]
            probe_parent_route = pr.get("parent_route_recommended")
        except Exception:                              # noqa: BLE001
            pass
    stars, rep = fetch_parent(sc, mode=mode, n_shards=n_shards,
                              cap_per_shard=max_stars or None, query_fn=query_fn, shape=shape,
                              vizier=vizier, vizier_fetch_fn=asu_fetch_fn)
    rep = {"stage": "sample", "generated_utc": _now(), "n_shards_planned": int(n_shards),
           "query_shape_from_probe": shape,
           "parent_route_from_probe": probe_parent_route, **rep}
    out.mkdir(parents=True, exist_ok=True)
    if len(stars):
        stars.to_parquet(out / "parent.parquet", index=False)
        rep["path"] = str(out / "parent.parquet")
    _write(out / "sample.json", rep)
    print(f"[ignition] sample: {rep['status']} {rep['n_after_local_cuts']} stars "
          f"(parent COUNT(*) = {rep['parent_count']}); routes={rep.get('routes')}"
          + (f"; DEGRADED {rep['degraded']}" if rep.get("degraded") else ""))
    return rep


def _load_parent(out: Path) -> pd.DataFrame:
    p = out / "parent.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["source_id"] = df["source_id"].astype(str)
    return df


def shard_rows(df: pd.DataFrame, shard: int, n_shards: int,
               max_stars: int | None = None) -> pd.DataFrame:
    """Deterministic partition: row ``k`` belongs to shard ``k mod n``."""
    if not len(df):
        return df
    idx = np.arange(len(df))
    sub = df[idx % max(int(n_shards), 1) == int(shard)]
    if max_stars:
        sub = sub.head(int(max_stars))
    return sub.reset_index(drop=True)


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------
def stage_acquire(conf: dict, out: Path, *, shard: int = 0, n_shards: int = 1,
                  max_stars: int | None = None, route: str | None = None,
                  cone_fn=None, upload_fn=None, field_fn=None,
                  stars: pd.DataFrame | None = None) -> dict:
    ac = conf["acquire"]
    tag = _tag(shard, n_shards)
    if stars is None:
        stars = shard_rows(_load_parent(out), shard, n_shards, max_stars)
    if route in (None, "", "auto"):
        route = str(ac.get("route", "auto"))
    if route == "auto":
        route = "cone"
        p = out / "probe.json"
        if p.exists():
            try:
                rec = json.loads(p.read_text()).get("neowise_route_recommended", "cone")
                route = rec if rec in ("upload", "field", "cone") else "cone"
            except Exception:                          # noqa: BLE001
                pass
    fields = list(conf["sample"].get("fields") or [])
    if route == "field" and not fields:
        print("[ignition] route=field but no fields configured; using per-star cones")
        route = "cone"
    store = EpochStore.open(out, tag)
    rep = {"stage": "acquire", "tag": tag, "shard": int(shard), "n_shards": int(n_shards),
           "generated_utc": _now(), "n_stars_in_shard": int(len(stars)), "route": route}
    if not len(stars):
        rep["status"] = "NO_PARENT_ROWS"
        store.flush({"rollup": rep})
        print(f"[ignition] acquire {tag}: no parent rows (was `sample` run?)")
        return rep
    roll = acquire_stars(stars, store, ac, route=route, fields=fields, cone_fn=cone_fn,
                         upload_fn=upload_fn, field_fn=field_fn)
    rep.update(roll)
    rep["status"] = ("OK" if roll["n_ok"] else
                     "QUERY_FAILED" if roll["n_failed"] and not roll["n_zero_rows"] else
                     "QUERY_RETURNED_ZERO_ROWS")
    rep["ledger_tail"] = store.ledger[-20:]
    store.flush({"rollup": rep})
    print(f"[ignition] acquire {tag}: {rep['status']} ok={roll['n_ok']} "
          f"zero={roll['n_zero_rows']} failed={roll['n_failed']} route={route}")
    return rep


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------
def screen_epochs(epochs: pd.DataFrame, conf: dict, *, quality: pd.DataFrame | None = None,
                  parent: pd.DataFrame | None = None, seed: int = 20260913
                  ) -> tuple[pd.DataFrame, dict]:
    """Rise statistics per star, the two-band rule, and the injection sensitivity."""
    rc = conf["rise"]
    sens_conf = conf.get("sensitivity") or DEFAULTS["sensitivity"]
    rows: list[dict] = []
    series_for_injection: list[dict] = []
    if len(epochs):
        epochs = epochs.copy()
        epochs["source_id"] = epochs["source_id"].astype(str)
        for sid, g in epochs.groupby("source_id"):
            series = epochs_to_series(g)
            per_band, verdict = assess_series(series, rc)
            rec = {"source_id": str(sid)}
            for b in ("W1", "W2"):
                if b in per_band:
                    rec.update(per_band[b].as_dict(prefix=f"{b.lower()}_"))
                else:
                    rec[f"{b.lower()}_n_epochs"] = 0
            rec.update({f"star_{k}": v for k, v in verdict.as_dict().items()})
            rec["is_candidate"] = bool(verdict.is_candidate)
            rec["screen_verdict"] = verdict.verdict
            rows.append(rec)
            # Injections go on stars that are measurable and not already rising.
            if ("W1" in series and "W2" in series and not verdict.is_candidate
                    and verdict.verdict not in ("INSUFFICIENT_EPOCHS", "SHORT_BASELINE")):
                series_for_injection.append(series)
    df = pd.DataFrame(rows)
    if len(df):
        if quality is not None and len(quality):
            q = quality.copy()
            q["source_id"] = q["source_id"].astype(str)
            df = df.merge(q.drop_duplicates("source_id"), on="source_id", how="left")
        if parent is not None and len(parent):
            p = parent.copy()
            p["source_id"] = p["source_id"].astype(str)
            df = df.merge(p.drop_duplicates("source_id"), on="source_id", how="left",
                          suffixes=("", "_parent"))
    counts = (df["screen_verdict"].value_counts().to_dict() if len(df) else {})
    sens = sensitivity_from_injections(series_for_injection, sens_conf["amps_mag"], rc,
                                       over_yr=float(sens_conf["over_yr"]),
                                       max_stars=int(sens_conf["max_stars"]), seed=seed)
    rep = {"n_stars_screened": int(len(df)),
           "n_rise_candidates": int(df["is_candidate"].sum()) if len(df) else 0,
           "screen_counts": {str(k): int(v) for k, v in counts.items()},
           "sensitivity": sens}
    return df, rep


def stage_screen(conf: dict, out: Path, *, shard: int = 0, n_shards: int = 1,
                 seed: int = 20260913) -> dict:
    tag = _tag(shard, n_shards)
    ep_p = out / f"epochs_{tag}.csv"
    q_p = out / f"neowise_stars_{tag}.csv"
    rep = {"stage": "screen", "tag": tag, "shard": int(shard), "n_shards": int(n_shards),
           "generated_utc": _now()}
    acq_p = out / f"acquire_{tag}.json"
    acq_status = None
    if acq_p.exists():
        try:
            acq_status = json.loads(acq_p.read_text()).get("rollup", {})
        except Exception:                              # noqa: BLE001
            acq_status = None
    rep["acquire_rollup"] = acq_status
    epochs = pd.read_csv(ep_p, dtype={"source_id": str}) if ep_p.exists() else pd.DataFrame()
    quality = pd.read_csv(q_p, dtype={"source_id": str}) if q_p.exists() else pd.DataFrame()
    parent = shard_rows(_load_parent(out), shard, n_shards)
    rep["n_stars_in_shard"] = int(len(parent))
    rep["n_stars_with_neowise"] = (int((quality.get("neowise_status", pd.Series(dtype=str))
                                        == "OK").sum()) if len(quality) else 0)
    if not len(epochs):
        rep.update({"status": "NO_EPOCHS", "n_stars_screened": 0, "n_rise_candidates": 0,
                    "screen_counts": {}, "sensitivity": {"n_stars": 0, "amps": {}}})
        _write(out / f"screen_{tag}.json", rep)
        print(f"[ignition] screen {tag}: NO_EPOCHS")
        return rep
    df, srep = screen_epochs(epochs, conf, quality=quality, parent=parent, seed=seed + shard)
    df.to_csv(out / f"stars_{tag}.csv", index=False)
    rep.update(srep)
    rep["status"] = "OK"
    _write(out / f"screen_{tag}.json", rep)
    print(f"[ignition] screen {tag}: {srep['n_stars_screened']} screened, "
          f"{srep['n_rise_candidates']} rise candidates; counts={srep['screen_counts']}")
    return rep


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------
def _aggregate_sensitivity(screens: list[dict]) -> dict:
    agg: dict = {}
    n_stars = 0
    for s in screens:
        sens = s.get("sensitivity") or {}
        n_stars += int(sens.get("n_stars", 0) or 0)
        for a, r in (sens.get("amps") or {}).items():
            cur = agg.setdefault(a, {"n": 0, "n_recovered": 0})
            cur["n"] += int(r.get("n", 0) or 0)
            cur["n_recovered"] += int(r.get("n_recovered", 0) or 0)
    for r in agg.values():
        r["fraction"] = (r["n_recovered"] / r["n"]) if r["n"] else None
    return {"n_stars": n_stars, "amps": agg,
            "note": "linear ramps of the stated amplitude over 10 yr injected on both bands "
                    "of screened, non-rising stars; fraction recovered as IGNITION_CANDIDATE "
                    "by the rise test alone (before the contaminant ladder)"}


def stage_assess(conf: dict, out: Path, *, n_shards_expected: int | None = None,
                 optical_dir: Path | str | None = None, offline: bool = True) -> dict:
    vc = conf["vet"]
    sample = {}
    if (out / "sample.json").exists():
        try:
            sample = json.loads((out / "sample.json").read_text())
        except Exception:                              # noqa: BLE001
            sample = {}
    screens = []
    for fp in sorted(glob.glob(str(out / "screen_s*of*.json"))):
        try:
            screens.append(json.loads(Path(fp).read_text()))
        except Exception:                              # noqa: BLE001
            continue
    acquires = []
    for fp in sorted(glob.glob(str(out / "acquire_s*of*.json"))):
        try:
            acquires.append(json.loads(Path(fp).read_text()).get("rollup") or {})
        except Exception:                              # noqa: BLE001
            continue
    frames = []
    for fp in sorted(glob.glob(str(out / "stars_s*of*.csv"))):
        try:
            d = pd.read_csv(fp, dtype={"source_id": str})
        except Exception:                              # noqa: BLE001
            continue
        if len(d):
            tagcol = pd.Series(Path(fp).name, index=d.index, name="shard_file")
            frames.append(pd.concat([d, tagcol], axis=1))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df):
        df = df.drop_duplicates("source_id")

    n_exp = int(n_shards_expected or sample.get("n_shards_planned") or
                max([int(s.get("n_shards", 0) or 0) for s in screens] + [0]))
    found_tags = sorted({str(s.get("tag")) for s in screens})
    shards = {"expected": n_exp, "found": len(found_tags), "found_tags": found_tags,
              "missing": [f"s{i}of{n_exp}" for i in range(n_exp)
                          if f"s{i}of{n_exp}" not in found_tags] if n_exp else []}
    degraded: list[str] = []
    if n_exp and shards["found"] < n_exp:
        degraded.append(f"shards_missing:{len(shards['missing'])}/{n_exp}")
    if sample.get("n_units_failed"):
        degraded.append(f"sample_units_failed:{sample['n_units_failed']}/{sample.get('n_units')}")
    # A parent drawn from two different sources is DEGRADED, never silently mixed:
    # the ESA route carries the archive's own cross-match, the VizieR route a
    # positional one bounded by an -out.max row cap.
    for d in (sample.get("degraded") or []):
        if d not in degraded:
            degraded.append(str(d))
    n_acq_failed = int(sum(int(a.get("n_failed", 0) or 0) for a in acquires))
    if n_acq_failed:
        degraded.append(f"neowise_queries_failed:{n_acq_failed}")

    n_attempted = int(sum(int(a.get("n_attempted", 0) or 0) for a in acquires))
    n_with_nw = int(sum(int(a.get("n_ok", 0) or 0) for a in acquires))
    n_screened = int(len(df))
    denominators = {
        "parent_count_archive": sample.get("parent_count"),
        "n_rows_pulled": sample.get("n_rows_pulled"),
        "n_stars_in_parent_file": sample.get("n_after_local_cuts"),
        "subsample_fraction": sample.get("subsample_fraction"),
        "n_stars_attempted_neowise": n_attempted,
        "n_stars_with_neowise_rows": n_with_nw,
        "n_stars_screened": n_screened,
        "sample_mode": sample.get("mode"),
        "local_cut_counters": sample.get("local_cut_counters"),
        # Which SOURCE the parent came from, and in what proportion.
        "parent_routes": sample.get("routes"),
        "parent_route_fractions": sample.get("route_fractions"),
        "parent_route_used": sample.get("route_used"),
        "parent_routes_mixed": bool(sample.get("mixed_routes")),
        "vizier_capped_units": sample.get("vizier_capped_units"),
        "route_note": sample.get("route_note"),
    }

    # --- nothing screened: say which archive did not answer ------------------
    if n_screened == 0:
        if not sample:
            reason = "no_sample_json: the sample stage did not run or did not write"
        elif sample.get("status") == "QUERY_FAILED":
            reason = "gaia_QUERY_FAILED"
        elif sample.get("status") == "QUERY_RETURNED_ZERO_ROWS":
            reason = "gaia_QUERY_RETURNED_ZERO_ROWS"
        elif not screens and not acquires:
            reason = "no_shard_outputs_found"
        elif n_with_nw == 0 and n_acq_failed:
            reason = "neowise_all_queries_failed"
        else:
            reason = "neowise_returned_no_usable_epochs"
        summary = {"verdict": VERDICT_NO_DATA, "reason": reason, "generated_utc": _now(),
                   "denominators": denominators, "shards": shards, "degraded": degraded,
                   "veto_counters": {"screen": {}, "vet": {"verdicts": {}, "flags": {}}},
                   "stage_counts": {"screened": 0, "rise_candidates": 0, "vetted": 0,
                                    "survivors": 0},
                   "sensitivity": _aggregate_sensitivity(screens),
                   "note": ("nothing was screened, so nothing about ignitions was measured; "
                            "this is NOT a null result and must not be reported as one")}
        _write(out / "summary.json", summary)
        pd.DataFrame(columns=["source_id"]).to_csv(out / "candidates.csv", index=False)
        print(f"[ignition] assess: {VERDICT_NO_DATA} ({reason})")
        return summary

    # --- the ladder on the rise candidates -------------------------------------
    cands = df[df["is_candidate"].astype(bool)].copy() if "is_candidate" in df else df.iloc[0:0]
    vets = []
    for _, r in cands.iterrows():
        row = r.to_dict()
        opt = load_optical_series(optical_dir, str(row["source_id"]))
        v = vet_star(row, vc, optical=opt)
        vd = v.as_dict()
        vd["source_id"] = str(row["source_id"])
        vets.append(vd)
    vdf = pd.DataFrame(vets)
    if len(vdf):
        vetted = cands.merge(vdf, on="source_id", how="left", suffixes=("", "_vet"))
        vetted = vetted.rename(columns={"verdict": "vet_verdict"})
    else:
        vetted = cands.assign(vet_verdict=pd.Series(dtype=str))
    survivors = vetted[vetted["vet_verdict"].isin(["clean", "clean_optical_untested"])] \
        if len(vetted) else vetted
    gold = vetted[vetted["vet_verdict"] == "clean"] if len(vetted) else vetted
    vetted.to_csv(out / "stars_vetted.csv", index=False)
    slim = [c for c in ("source_id", "ra", "dec", "parallax", "parallax_over_error", "pmra",
                        "pmdec", "phot_g_mean_mag", "bp_rp", "abs_g", "v_tan_kms",
                        "kinematically_old", "w1w2_2010", "w1_median", "w2_median",
                        "w1_n_epochs", "w1_baseline_yr", "w1_slope_mag_yr", "w1_slope_sigma",
                        "w1_tau_rise", "w1_tau_p", "w1_rise_mag", "w1_rise_sigma",
                        "w1_mono_frac", "w1_delta_bic", "w1_scan_amp_mag",
                        "w2_n_epochs", "w2_baseline_yr", "w2_slope_mag_yr", "w2_slope_sigma",
                        "w2_tau_rise", "w2_tau_p", "w2_rise_mag", "w2_rise_sigma",
                        "w2_mono_frac", "w2_delta_bic", "w2_scan_amp_mag",
                        "star_rise_w2_minus_w1_mag", "vet_verdict", "flags", "optical",
                        "optical_slope_sigma", "untested_checks", "shard_file")
            if c in survivors.columns]
    survivors[slim].to_csv(out / "candidates.csv", index=False)

    screen_counts: dict = {}
    for s in screens:
        for k, v in (s.get("screen_counts") or {}).items():
            screen_counts[k] = screen_counts.get(k, 0) + int(v)
    verdict = VERDICT_CANDIDATES if len(survivors) else VERDICT_NONE
    if degraded:
        verdict = f"{DEGRADED} ({'; '.join(degraded)}); {verdict}"
    summary = {
        "verdict": verdict, "generated_utc": _now(),
        "n_candidates": int(len(survivors)), "n_clean": int(len(gold)),
        "n_candidates_optical_untested": int(len(survivors) - len(gold)),
        "denominators": denominators,
        "stage_counts": {"screened": n_screened, "rise_candidates": int(len(cands)),
                         "vetted": int(len(vetted)), "survivors": int(len(survivors)),
                         "clean": int(len(gold))},
        "veto_counters": {"screen": screen_counts, "vet": summarise(vets)},
        "sensitivity": _aggregate_sensitivity(screens),
        "shards": shards, "degraded": degraded, "offline": bool(offline),
        "optical_dir": str(optical_dir) if optical_dir else None,
        "config": {"rise": conf["rise"], "vet": {k: v for k, v in vc.items()
                                                  if k != "star_forming_boxes"}},
        "note": ("a candidate here is a star whose W1 and W2 both rise monotonically over "
                 ">= 5 yr at >= 5 sigma with a ramp preferred over an impulsive born excess, "
                 "and that survived the contaminant ladder; NO_IGNITION_CANDIDATE is a count "
                 "over n_stars_screened, not an occurrence limit, and is not written up "
                 "(CLAUDE.md)"),
    }
    _write(out / "summary.json", summary)
    print(f"[ignition] assess: {verdict} — {len(survivors)} survivors of {len(cands)} rise "
          f"candidates from {n_screened} screened")
    return summary


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def ignition_run(stage: str = "all", *, out_dir: Path | str | None = None, shard: int = 0,
                 n_shards: int = 1, max_stars: int | None = None, route: str | None = None,
                 mode: str | None = None, optical_dir: Path | str | None = None,
                 seed: int = 20260913, conf: dict | None = None, config_path=None,
                 query_fn=None, cone_fn=None, upload_fn=None, field_fn=None) -> dict:
    """Run one stage, a comma list, or all of them.  Returns the last report."""
    conf = conf if conf is not None else load_ignition_config(config_path)
    out = Path(out_dir) if out_dir else Path("results") / "ignition"
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage in ("all", "", None) else tuple(s.strip() for s in stage.split(","))
    rep: dict = {}
    t0 = _time.monotonic()
    for s in stages:
        if s == "probe":
            rep = stage_probe(conf, out, query_fn=query_fn, cone_fn=cone_fn, upload_fn=upload_fn)
        elif s == "sample":
            rep = stage_sample(conf, out, n_shards=n_shards, max_stars=max_stars,
                               query_fn=query_fn, mode=mode)
        elif s == "acquire":
            rep = stage_acquire(conf, out, shard=shard, n_shards=n_shards, max_stars=max_stars,
                                route=route, cone_fn=cone_fn, upload_fn=upload_fn,
                                field_fn=field_fn)
        elif s == "screen":
            rep = stage_screen(conf, out, shard=shard, n_shards=n_shards, seed=seed)
        elif s == "assess":
            rep = stage_assess(conf, out, n_shards_expected=n_shards if stage == "all" else None,
                               optical_dir=optical_dir)
        else:
            raise SystemExit(f"unknown stage {s!r}; choose from {STAGES + ('all',)}")
    print(f"[ignition] {stage}: done in {_time.monotonic() - t0:.0f}s")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(prog="seti ignition",
                                description="IGNITION (S61): an infrared excess being born on "
                                            "an old star")
    p.add_argument("--stage", default="all",
                   help="probe|sample|acquire|screen|assess|all or a comma list")
    p.add_argument("--shard", default="0/1", help="i/n: this shard of n (acquire, screen)")
    p.add_argument("--shards", type=int, default=0,
                   help="number of shards the run is planned for (sample, assess); "
                        "defaults to the n of --shard")
    p.add_argument("--max-stars", type=int, default=0,
                   help="cap per shard (sample: rows per shard; acquire: stars fetched)")
    p.add_argument("--out-dir", default="", help="results directory (default results/ignition)")
    p.add_argument("--route", default="auto", help="auto|upload|field|cone (acquire)")
    p.add_argument("--mode", default="", help="fields|allsky (sample; default from config)")
    p.add_argument("--optical-dir", default="",
                   help="directory of <source_id>.csv optical series (ZTF/ASAS-SN) for assess")
    p.add_argument("--config", default="", help="alternative config yaml")
    p.add_argument("--seed", type=int, default=20260913)
    a = p.parse_args(argv)
    shard, n = parse_shard(a.shard)
    n_shards = a.shards or n
    rep = ignition_run(a.stage, out_dir=a.out_dir or None, shard=shard, n_shards=n_shards,
                       max_stars=a.max_stars or None, route=a.route or None,
                       mode=a.mode or None, optical_dir=a.optical_dir or None, seed=a.seed,
                       config_path=a.config or None)
    v = rep.get("verdict") if isinstance(rep, dict) else None
    if v:
        print(f"[ignition] verdict: {v}")
    return 0


if __name__ == "__main__":                            # pragma: no cover
    raise SystemExit(main())


__all__ = ["DEFAULTS", "DEFAULT_PROBE", "STAGES", "ignition_run", "load_ignition_config", "main", "parse_shard",
           "screen_epochs", "shard_rows", "stage_acquire", "stage_assess", "stage_probe",
           "stage_sample", "stage_screen"]
