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
             --- ``parent_routes_tried`` / ``parent_route_recommended`` over all
             three parent routes (``esa_gaia``, ``irsa_tap``, ``vizier_asu``),
             and ``neowise_route_recommended``.  The workflow commits it:
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
import threading
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .acquire import (
    DEFAULT_ACQUIRE,
    EpochStore,
    _accepts,
    acquire_stars,
    epochs_to_series,
)
from .rise import DEFAULT_RISE, assess_series, fit_linear_slope, sensitivity_from_injections
from .sample import (
    DEFAULT_SAMPLE,
    JOINED_SHAPES,
    QUERY_FAILED,
    QUERY_TIMED_OUT,
    ROUTE_ESA,
    ROUTE_IRSA,
    ROUTE_VIZIER,
    SHAPES,
    QueryTimeout,
    build_query,
    call_with_timeout,
    fetch_parent,
    run_gaia_query,
    unwrap_result,
)
from .tiles import DEFAULT_TILES, owns, sky_tiles, tile_field, tiles_for_shard
from .vet import DEFAULT_VET, load_optical_series, summarise, vet_star

VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_NONE = "NO_IGNITION_CANDIDATE"
VERDICT_CANDIDATES = "IGNITION_CANDIDATES"
DEGRADED = "DEGRADED"
STAGES = ("probe", "sample", "acquire", "screen", "assess")
#: ``sweep`` is the tiles-mode shard stage: sample + acquire, tile by tile, under
#: one wall clock, then the screen.  It is not part of ``all`` (which is the
#: fields-mode pipeline); the workflow runs it per matrix job.
EXTRA_STAGES = ("sweep",)

#: The ensemble zero-point correction applied by the screen (``config ->
#: screen``).  NEOWISE's W1/W2 zero points are not constant over the mission,
#: and run 35039105536 labelled 97 of 172 stars FADING: a common drift is the
#: survey, not the stars.  Per band, the median of (mag - star median) over the
#: shard's stars in each time bin is subtracted when the bin holds enough
#: stars; the offsets are recorded, and the raw linear slope significance is
#: kept per star so the correction's effect is on the record.
DEFAULT_SCREEN: dict = {
    "ensemble_correct": True,
    "ensemble_bin_yr": 0.25,
    "ensemble_min_stars": 8,
    # "stratified" (per W1/W2 magnitude bin x |beta| band, falling back to the
    # magnitude bin, then to the global median) or "global" (the pre-2026-09-23
    # correction, which over-corrects bright stars: docs/ignition.md 7.4).
    "ensemble_mode": "stratified",
    "stratum_min_stars": 40,
}

#: The tiles-mode shard clock and the per-tile sample time-box.
DEFAULT_SWEEP: dict = {
    "time_budget_s": 9000.0,        # the shard stops starting new tiles after this
    "sample_timeout_s": 300.0,      # one tile's parent query, per attempt
    "sample_unit_budget_s": 600.0,  # one tile's parent query, across the WHOLE ladder
    "max_tiles": 0,                 # 0 = every tile of the shard
    "sample_prefetch": 2,           # parent queries run this many tiles ahead
}

#: The probe's wall-clock discipline.  Run 34787803862 spent 1,490 s on four
#: retries of a query that could not succeed; a broken plan must cost minutes.
DEFAULT_PROBE: dict = {
    "budget_s": 480.0,              # per candidate query shape
    "total_budget_s": 2400.0,       # the whole probe stage
    "columns_timeout_s": 120.0,     # the AllWISE column peek
    "neowise_timeout_s": 300.0,     # each NEOWISE route test
    "cap": 5,                       # TOP n for the join test
    "shapes": list(SHAPES),
    "irsa_timeout_s": 600.0,        # the SECOND route's probe (gaia_only x IRSA)
    "vizier_timeout_s": 300.0,      # the THIRD route's probe (VizieR ASU)
}

DEFAULTS: dict = {
    "sample": dict(DEFAULT_SAMPLE),
    "acquire": dict(DEFAULT_ACQUIRE),
    "rise": dict(DEFAULT_RISE),
    "vet": dict(DEFAULT_VET),
    "probe": dict(DEFAULT_PROBE),
    "sensitivity": {"amps_mag": [0.1, 0.2, 0.4], "over_yr": 10.0, "max_stars": 200},
    "tiles": dict(DEFAULT_TILES),
    "sweep": dict(DEFAULT_SWEEP),
    "screen": dict(DEFAULT_SCREEN),
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
                upload_fn=None, asu_fetch_fn=None, irsa_fetch_fn=None) -> dict:
    """One minimal call per route; decides the acquire architecture.

    The Gaia join is tried as each of :data:`~seti.ignition.sample.SHAPES` in
    order, under a per-shape wall-clock budget, and the shape that answered is
    written to ``probe.json`` as ``gaia_shape_working`` so the sample stage does
    not re-derive it.  Every shape that did not answer keeps its verbatim error
    and the exact ADQL that was sent.  ``probe.json`` is written whatever
    happens --- a green run that commits nothing is a lost run.

    **ALL THREE parent routes are probed, always** --- the ESA archive's joined
    shapes first, then ``gaia_only`` x IRSA (:mod:`seti.ignition.irsa_route`),
    then VizieR's non-TAP ASU interface, even when ESA answered --- so the next
    run's ``probe.json`` names the transport that works instead of re-deriving
    it.  (Probing is not using: the sample stage's fallbacks still run only
    where the route before them failed.)  When ESA returns nothing and a
    fallback does, the NEOWISE route tests below are run on a star that fallback
    supplied rather than skipped.
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
    # Only the JOINED shapes can serve the parent on their own, so only they are
    # candidates for `gaia_shape_working`.  `gaia_only` is the IRSA route's own
    # half and is probed there, below, where its AllWISE partner is also probed.
    shapes = ([s for s in (pc.get("shapes") or SHAPES) if s in JOINED_SHAPES]
              or list(JOINED_SHAPES))
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

    # 2b. The SECOND route to the same parent sample: the `gaia_only` shape at
    # ESA (no AllWISE table anywhere in it --- the one thing every shape above
    # has in common is the table that timed out) joined to the AllWISE
    # catalogue at IRSA's own TAP service.  Probed on every run, whatever ESA
    # did, and it settles the asserted IRSA table/column names against the
    # service's own TAP_SCHEMA in the same call.
    from .irsa_route import probe_route as irsa_probe_route

    ir_val, ir_err = _timeboxed(
        lambda: irsa_probe_route(sc, query_fn=query_fn, fetch_fn=irsa_fetch_fn, cap=cap),
        _budget(pc.get("irsa_timeout_s"), _left()), ROUTE_IRSA)
    if ir_val is None:
        rep[ROUTE_IRSA] = dict(ir_err or {"status": QUERY_FAILED, "error": "no record"})
        ir_frame = pd.DataFrame()
    else:
        rep[ROUTE_IRSA], ir_frame = ir_val
    rep[ROUTE_IRSA].setdefault("usable", False)
    irsa_ok = bool(rep[ROUTE_IRSA].get("usable"))
    print(f"[ignition] probe: {ROUTE_IRSA} {rep[ROUTE_IRSA].get('status')} "
          f"rows={rep[ROUTE_IRSA].get('n_rows')} "
          f"table={rep[ROUTE_IRSA].get('table')} "
          f"gaia_only={rep[ROUTE_IRSA].get('gaia_only_status')}")

    # 2c. The THIRD route: VizieR's non-TAP ASU interface.  Probed on every run
    # too, because the point of a probe is to say which transports are alive
    # tonight, not which one the last run happened to use.
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
    if len(df):
        rep["neowise_star_from_route"] = ROUTE_ESA
    elif len(ir_frame):
        # ESA's joined shapes reached nothing; the NEOWISE tests below still
        # deserve a real star, and this is the route nearest to the archive's.
        df = ir_frame.rename(columns={c: str(c).lower() for c in ir_frame.columns})
        rep["neowise_star_from_route"] = ROUTE_IRSA
    elif len(vz_frame):
        df = vz_frame.rename(columns={c: str(c).lower() for c in vz_frame.columns})
        rep["neowise_star_from_route"] = ROUTE_VIZIER

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

    # 4. The upload ladder, two stars.  Run 35039105536's probe recorded the one
    # rung it had (pyvo's multipart form on the async queue) refused with
    # `QUERY must be set`; every rung is walked now and the one that answered
    # is written down for the acquire stage to start from.
    if upload_fn is None:
        from .acquire import fetch_neowise_upload as upload_fn
    if len(df):
        radius = float(conf["acquire"]["cone_radius_arcsec"])
        nw_left = _budget(pc.get("neowise_timeout_s"), _left())

        def _up():
            if _accepts(upload_fn, "transport"):
                return upload_fn(df.head(2), radius_arcsec=radius, transport=None,
                                 timeout_s=float(nw_left or 300.0), ladder=True)
            return upload_fn(df.head(2), radius_arcsec=radius)

        r, err = _timeboxed(_up, nw_left, "neowise_upload")
        rep["neowise_upload"] = err or r.to_ledger()
        label = str(rep["neowise_upload"].get("label") or "")
        rep["neowise_upload_transport"] = (label.split("[", 1)[1].split("]", 1)[0]
                                           if "[" in label and rep["neowise_upload"].get("status")
                                           == "OK" else None)
    else:
        rep["neowise_upload"] = {"status": "NOT_ATTEMPTED", "reason": "no Gaia row to upload"}
        rep["neowise_upload_transport"] = None

    cone_ok = rep["neowise_cone"].get("status") == "OK"
    up_ok = rep["neowise_upload"].get("status") == "OK"
    gaia_ok = bool(working)
    # `field` is never recommended any more: at 1 degree near the ecliptic poles
    # it is 2.4-4.7 M rows a query and broke every way a transfer can break
    # (results/ignition/acquire_s*.json, 2026-09-16).  The per-star cone is the
    # proven fallback and now runs concurrently.
    rep["neowise_route_recommended"] = ("upload" if up_ok else
                                        "cone" if cone_ok else "none")
    # Which SOURCE the next run should expect the parent from.  ESA stays first
    # whenever it answers: it is authoritative and owns the in-archive match.
    # IRSA comes next --- it keeps ESA's own Gaia cuts in SQL and gives up only
    # the cross-match --- and the VizieR mirror last.
    rep["parent_route_recommended"] = (ROUTE_ESA if gaia_ok else
                                       ROUTE_IRSA if irsa_ok else
                                       ROUTE_VIZIER if vizier_ok else "none")
    rep["parent_routes_tried"] = [ROUTE_ESA, ROUTE_IRSA, ROUTE_VIZIER]
    rep["verdict"] = ("ALL_ROUTES_REACHABLE" if (gaia_ok and cone_ok and up_ok) else
                      "GAIA_AND_NEOWISE_REACHABLE" if (gaia_ok and cone_ok) else
                      "GAIA_ONLY" if gaia_ok else
                      "IRSA_PARENT_AND_NEOWISE" if (irsa_ok and cone_ok) else
                      "IRSA_PARENT_ONLY" if irsa_ok else
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
                 vizier: bool = True, irsa_fetch_fn=None, irsa: bool = True) -> dict:
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
            # JOINED_SHAPES, not SHAPES: `gaia_only` returns no W1/W2 and is
            # never a parent shape on its own (seti.ignition.irsa_route).
            if pr.get("gaia_shape_working") in JOINED_SHAPES:
                shape = pr["gaia_shape_working"]
            probe_parent_route = pr.get("parent_route_recommended")
        except Exception:                              # noqa: BLE001
            pass
    stars, rep = fetch_parent(sc, mode=mode, n_shards=n_shards,
                              cap_per_shard=max_stars or None, query_fn=query_fn, shape=shape,
                              vizier=vizier, vizier_fetch_fn=asu_fetch_fn,
                              irsa=irsa, irsa_fetch_fn=irsa_fetch_fn)
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


def _load_parent(out: Path, tag: str | None = None) -> pd.DataFrame:
    """The parent frame: the shard's own (tiles mode) if it exists, else the run's."""
    cands = ([out / f"parent_{tag}.parquet"] if tag else []) + [out / "parent.parquet"]
    for p in cands:
        if p.exists():
            df = pd.read_parquet(p)
            df["source_id"] = df["source_id"].astype(str)
            return df
    return pd.DataFrame()


def _probe_record(out: Path) -> dict:
    p = out / "probe.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:                                  # noqa: BLE001
        return {}


def _resolve_route(conf: dict, out: Path, route: str | None) -> tuple[str, str | None]:
    """The NEOWISE route and the upload transport, from the flag or ``probe.json``."""
    ac = conf["acquire"]
    if route in (None, "", "auto"):
        route = str(ac.get("route", "auto"))
    pr = _probe_record(out)
    transport = pr.get("neowise_upload_transport") or None
    if str(ac.get("upload_transport", "auto")) != "auto":
        transport = str(ac["upload_transport"])
    if route == "auto":
        rec = pr.get("neowise_route_recommended", "cone")
        route = rec if rec in ("upload", "field", "cone") else "cone"
    return route, transport


def _shard_parent(out: Path, shard: int, n_shards: int) -> pd.DataFrame:
    """The shard's stars: its own tiles-mode parent if present, else its rows of the run's.

    A shard that swept tiles is identified by its ``sweep_{tag}.json``, and for
    such a shard the run-level ``parent.parquet`` is **not** a fallback: that
    file belongs to a fields/allsky sample over different sky, and a shard whose
    every tile failed must screen *nothing*, not somebody else's stars.  An
    empty frame here is the honest answer and the screen reports ``NO_EPOCHS``.
    """
    tag = _tag(shard, n_shards)
    if (out / f"parent_{tag}.parquet").exists():
        return _load_parent(out, tag)
    if (out / f"sweep_{tag}.json").exists():
        return pd.DataFrame()
    return shard_rows(_load_parent(out), shard, n_shards)


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
    route, transport = _resolve_route(conf, out, route)
    fields = list(conf["sample"].get("fields") or [])
    if route == "field" and not fields:
        print("[ignition] route=field but no fields configured; using per-star cones")
        route = "cone"
    store = EpochStore.open(out, tag)
    rep = {"stage": "acquire", "tag": tag, "shard": int(shard), "n_shards": int(n_shards),
           "generated_utc": _now(), "n_stars_in_shard": int(len(stars)), "route": route,
           "upload_transport_preferred": transport}
    if not len(stars):
        rep["status"] = "NO_PARENT_ROWS"
        store.flush({"rollup": rep})
        print(f"[ignition] acquire {tag}: no parent rows (was `sample` run?)")
        return rep
    roll = acquire_stars(stars, store, ac, route=route, fields=fields, cone_fn=cone_fn,
                         upload_fn=upload_fn, field_fn=field_fn, upload_transport=transport)
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
# sweep: tiles mode --- sample + acquire, tile by tile, under one wall clock
# ---------------------------------------------------------------------------
def _sweep_checkpoint(out: Path, tag: str) -> dict:
    p = out / f"sweep_{tag}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:                              # noqa: BLE001
            pass
    return {"tiles": []}


def stage_sweep(conf: dict, out: Path, *, shard: int = 0, n_shards: int = 1,
                route: str | None = None, query_fn=None, cone_fn=None, upload_fn=None,
                asu_fetch_fn=None, irsa_fetch_fn=None, vizier: bool = True, irsa: bool = True,
                time_budget_s: float | None = None, max_tiles: int | None = None) -> dict:
    """One shard of the all-sky sweep: its tiles, sampled and acquired in turn.

    Every tile is a checkpoint (``sweep_s{i}of{n}.json``): its parent query,
    ownership cut, and NEOWISE acquisition are recorded before the next tile
    starts, and a re-run (or a later dispatch resuming from this run's
    artifact) skips finished tiles.  The shard stops *starting* tiles at
    ``time_budget_s``; a tile whose acquisition the clock interrupted is
    ``PARTIAL`` and is re-run next time (its finished stars are in the store).
    The parent grows in ``parent_s{i}of{n}.parquet``; the screen reads it.
    """
    tag = _tag(shard, n_shards)
    sw = {**DEFAULT_SWEEP, **(conf.get("sweep") or {})}
    tc = {**DEFAULT_TILES, **(conf.get("tiles") or {})}
    tc["abs_b_min_deg"] = float(conf["sample"].get("abs_b_min_deg", tc["abs_b_min_deg"]))
    budget = float(time_budget_s or sw["time_budget_s"])
    t0 = _time.monotonic()
    deadline = t0 + budget
    tiles_all = sky_tiles(tc)
    mine = tiles_for_shard(tiles_all, shard, n_shards)
    cap_tiles = int(max_tiles or sw.get("max_tiles") or 0)
    if cap_tiles:
        mine = mine.head(cap_tiles)
    ck = _sweep_checkpoint(out, tag)
    done_tiles = {str(t.get("tile")) for t in ck.get("tiles", [])
                  if t.get("status") in ("OK", "QUERY_RETURNED_ZERO_ROWS")}
    if done_tiles:
        print(f"[ignition] sweep {tag}: resuming, {len(done_tiles)} tiles already done")
    parent_p = out / f"parent_{tag}.parquet"
    parent = _load_parent(out, tag) if parent_p.exists() else pd.DataFrame()
    route, transport = _resolve_route(conf, out, route)
    if route == "field":
        route = "upload" if transport else "cone"
    pr = _probe_record(out)
    shape = pr.get("gaia_shape_working") if pr.get("gaia_shape_working") in JOINED_SHAPES else None
    sc = dict(conf["sample"])
    if isinstance(pr.get("allwise_columns_resolved"), dict) and \
            "status" not in pr["allwise_columns_resolved"]:
        sc["allwise_columns"] = pr["allwise_columns_resolved"]
    sc["count_parent"] = False          # an uncapped tile's row count IS its count
    sc["query_timeout_s"] = float(sw["sample_timeout_s"])
    # What one tile's parent query may cost in total, across the whole route
    # ladder.  Without it a single tile can spend three ESA shapes plus both
    # fallback routes -- tens of minutes -- and in a time-limited dispatch that
    # is paid for in tiles never reached.  The ladder's ORDER is untouched.
    unit_budget = float(sw.get("sample_unit_budget_s") or 0.0) or None
    top = int(sc.get("cap_per_shard") or 20000)
    store = EpochStore.open(out, tag)
    ac = conf["acquire"]
    stopped = False
    n_new = 0
    tiles_rec: list[dict] = [t for t in ck.get("tiles", []) if str(t.get("tile")) in done_tiles]

    def _save(extra: dict | None = None) -> None:
        rec = {"stage": "sweep", "tag": tag, "shard": int(shard), "n_shards": int(n_shards),
               "generated_utc": _now(), "route": route, "upload_transport": transport,
               "tiles_assigned": int(len(mine)), "tiles_in_sky": int(len(tiles_all)),
               "tile_deg": float(tc["dec_step_deg"]), "budget_s": budget,
               "tiles": tiles_rec}
        rec.update(_sweep_rollup(tiles_rec, len(mine), len(tiles_all), tiles_all))
        rec.update(extra or {})
        _write(out / f"sweep_{tag}.json", rec)

    # Parent queries run AHEAD of the tile being acquired.  Run 35859572295
    # measured 150-280 s per ESA parent query and 100-190 s per acquire, done
    # one after the other: ~22 tiles per shard in 150 min where the previous
    # night did ~60.  The next `sample_prefetch` tiles' queries now run on
    # daemon threads while the current tile is acquired; tiles are still
    # PROCESSED (acquired, checkpointed) strictly in order, so the checkpoint
    # and resume semantics are unchanged.  A prefetched query whose tile is
    # never reached before the deadline is simply dropped.
    todo = [t for _, t in mine.iterrows() if str(t["tile"]) not in done_tiles]
    ahead = max(0, int(sw.get("sample_prefetch", 2) or 0))
    pending: dict[int, tuple] = {}

    def _start(i: int) -> None:
        box: dict = {}
        fld_i = tile_field(todo[i])

        def _go():
            t_s = _time.monotonic()
            try:
                box["res"] = fetch_parent(sc, mode="fields", fields=[fld_i], n_shards=1,
                                          cap_per_shard=top, query_fn=query_fn, shape=shape,
                                          vizier=vizier, vizier_fetch_fn=asu_fetch_fn,
                                          irsa=irsa, irsa_fetch_fn=irsa_fetch_fn,
                                          unit_budget_s=unit_budget)
            except BaseException as exc:               # noqa: BLE001
                box["error"] = exc
            box["dt"] = _time.monotonic() - t_s

        th = threading.Thread(target=_go, daemon=True)
        th.start()
        pending[i] = (th, box)

    for i, t in enumerate(todo):
        tile_id = str(t["tile"])
        if _time.monotonic() > deadline:
            stopped = True
            break
        for j in range(i, min(len(todo), i + 1 + ahead)):
            if j not in pending and (j == i or _time.monotonic() < deadline):
                _start(j)
        th, box = pending.pop(i)
        th.join()
        if "error" in box:
            raise box["error"]
        stars, srep = box["res"]
        n_cone = int(srep.get("n_rows_pulled") or 0)
        if len(stars):
            stars = stars[owns(t, stars["ra"].to_numpy(float),
                               stars["dec"].to_numpy(float))].reset_index(drop=True)
        rec = {"tile": tile_id, "order": int(t["order"]), "ra_c": float(t["ra_c"]),
               "dec_c": float(t["dec_c"]), "area_deg2": float(t["area_deg2"]),
               "status": srep.get("status"), "parent_route": srep.get("route_used"),
               "shape": srep.get("query_shape_used"), "n_cone_rows": n_cone,
               "n_parent": int(len(stars)), "capped": bool(n_cone >= top),
               "sample_s": round(float(box.get("dt", 0.0)), 1),
               "errors": [e for e in (srep.get("route_errors") or [])[:3]]}
        if srep.get("status") == "QUERY_FAILED":
            tiles_rec.append(rec)
            _save()
            print(f"[ignition] sweep {tag}: tile {tile_id} QUERY_FAILED", flush=True)
            continue
        if len(stars):
            stars = stars.copy()
            stars["tile"] = tile_id
            # A resumed parent comes back from parquet with source_id as str
            # (_load_parent) while a fresh tile's is int64; concatenating the two
            # gives a mixed object column that pyarrow refuses to write (run
            # 35859572295 lost every shard's resume to that, two minutes in).
            # The stored parent keys on str; `stars` itself is left as it is.
            new = stars.assign(source_id=stars["source_id"].astype(str))
            if len(parent):
                parent = parent.assign(source_id=parent["source_id"].astype(str))
                parent = pd.concat([parent, new], ignore_index=True).drop_duplicates("source_id")
            else:
                parent = new
            parent.to_parquet(parent_p, index=False)
            roll = acquire_stars(stars, store, ac, route=route, cone_fn=cone_fn,
                                 upload_fn=upload_fn, upload_transport=transport,
                                 deadline=deadline)
            rec["acquire"] = {k: roll.get(k) for k in ("n_attempted", "n_ok", "n_zero_rows",
                                                       "n_failed", "n_fallback_cone",
                                                       "elapsed_s", "stopped_on_budget")}
            if roll.get("stopped_on_budget"):
                rec["status"] = "PARTIAL"
                stopped = True
        n_new += 1
        tiles_rec.append(rec)
        _save()
        print(f"[ignition] sweep {tag}: tile {tile_id} {rec['status']} parent={rec['n_parent']} "
              f"acquire={rec.get('acquire')} ({_time.monotonic() - t0:.0f} s elapsed)", flush=True)
        if stopped:
            break
    rep = {"stopped_on_budget": bool(stopped), "tiles_new_this_run": int(n_new),
           "elapsed_s": round(_time.monotonic() - t0, 1),
           "n_parent_total": int(len(parent)), "n_done_total": int(len(store.done))}
    _save(rep)
    out_rep = json.loads((out / f"sweep_{tag}.json").read_text())
    print(f"[ignition] sweep {tag}: {out_rep['tiles_done']}/{out_rep['tiles_assigned']} tiles, "
          f"{out_rep['n_parent_total']} parent stars, {len(store.done)} acquired, "
          f"route={route}[{transport}], {'STOPPED on budget' if stopped else 'complete'}")
    return out_rep


def _sweep_rollup(tiles_rec: list[dict], n_assigned: int, n_sky: int,
                  tiles_all: pd.DataFrame | None = None) -> dict:
    ok = [t for t in tiles_rec if t.get("status") in ("OK", "QUERY_RETURNED_ZERO_ROWS")]
    failed = [t for t in tiles_rec if t.get("status") == "QUERY_FAILED"]
    partial = [t for t in tiles_rec if t.get("status") == "PARTIAL"]
    area_done = float(sum(float(t.get("area_deg2") or 0.0) for t in ok))
    area_sky = float(tiles_all["area_deg2"].sum()) if tiles_all is not None else None
    acq = [t.get("acquire") or {} for t in tiles_rec]
    return {"tiles_done": len(ok), "tiles_failed": len(failed), "tiles_partial": len(partial),
            "tiles_capped": int(sum(1 for t in ok if t.get("capped"))),
            "area_done_deg2": round(area_done, 1), "area_sky_deg2": area_sky,
            "n_parent_done_tiles": int(sum(int(t.get("n_parent") or 0) for t in ok)),
            "n_stars_attempted": int(sum(int(a.get("n_attempted") or 0) for a in acq)),
            "n_stars_ok": int(sum(int(a.get("n_ok") or 0) for a in acq)),
            "n_stars_failed": int(sum(int(a.get("n_failed") or 0) for a in acq)),
            "n_stars_fallback_cone": int(sum(int(a.get("n_fallback_cone") or 0) for a in acq))}


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------
def screen_epochs(epochs: pd.DataFrame, conf: dict, *, quality: pd.DataFrame | None = None,
                  parent: pd.DataFrame | None = None, seed: int = 20260913
                  ) -> tuple[pd.DataFrame, dict]:
    """Rise statistics per star, the two-band rule, and the injection sensitivity."""
    rc = conf["rise"]
    sens_conf = conf.get("sensitivity") or DEFAULTS["sensitivity"]
    scr = {**DEFAULT_SCREEN, **(conf.get("screen") or {})}
    rows: list[dict] = []
    series_for_injection: list[dict] = []
    ens_offsets: dict = {}
    strat_rep: dict = {}
    if len(epochs):
        epochs = epochs.copy()
        epochs["source_id"] = epochs["source_id"].astype(str)
        # The ensemble zero-point correction (seti.ignition.ensemble): what the
        # rise test sees is the star relative to the shard's constant stars.
        if scr.get("ensemble_correct", True):
            from .ensemble import apply_ensemble, ensemble_offsets, stratified_offsets

            # The global per-bin offsets are always measured (they are the
            # recorded drift); what is APPLIED is the stratified correction
            # unless the config asks for the old global one.
            ens_offsets = ensemble_offsets(epochs, bin_yr=float(scr["ensemble_bin_yr"]),
                                           min_stars=int(scr["ensemble_min_stars"]))
            if str(scr.get("ensemble_mode", "stratified")) == "stratified":
                beta = None
                if parent is not None and len(parent) and {"ra", "dec"} <= set(parent.columns):
                    from .acquire import ecliptic_latitude_deg

                    pp = parent.drop_duplicates("source_id")
                    beta = pd.Series(np.abs(ecliptic_latitude_deg(
                        pd.to_numeric(pp["ra"], errors="coerce"),
                        pd.to_numeric(pp["dec"], errors="coerce"))),
                        index=pp["source_id"].astype(str).to_numpy())
                epochs, strat_rep = stratified_offsets(
                    epochs, beta, bin_yr=float(scr["ensemble_bin_yr"]),
                    min_stars=int(scr.get("stratum_min_stars", 40)))
            else:
                epochs = apply_ensemble(epochs, ens_offsets,
                                        bin_yr=float(scr["ensemble_bin_yr"]))
        else:
            epochs["mag_raw"] = pd.to_numeric(epochs["mag"], errors="coerce")
            epochs["ensemble_applied"] = False
        for sid, g in epochs.groupby("source_id"):
            series = epochs_to_series(g)
            per_band, verdict = assess_series(series, rc)
            rec = {"source_id": str(sid)}
            for b in ("W1", "W2"):
                if b in per_band:
                    rec.update(per_band[b].as_dict(prefix=f"{b.lower()}_"))
                else:
                    rec[f"{b.lower()}_n_epochs"] = 0
                # The uncorrected linear slope, so the correction is auditable per star.
                gb = g[g["band"] == b].sort_values("t_yr")
                if len(gb) >= 3:
                    s_raw, e_raw = fit_linear_slope(gb["t_yr"], gb["mag_raw"], gb["err"],
                                                    err_floor=float(rc.get("err_floor_mag",
                                                                           0.005)))
                    rec[f"{b.lower()}_slope_raw_mag_yr"] = s_raw
                    rec[f"{b.lower()}_slope_raw_sigma"] = (float(-s_raw / e_raw)
                                                          if e_raw and e_raw > 0
                                                          else float("nan"))
                    rec[f"{b.lower()}_ensemble_frac"] = float(gb["ensemble_applied"].mean())
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
    from .ensemble import summarise_offsets

    raw_fading = raw_rising = 0
    if len(df) and "w1_slope_raw_sigma" in df and "w2_slope_raw_sigma" in df:
        s1 = pd.to_numeric(df["w1_slope_raw_sigma"], errors="coerce")
        s2 = pd.to_numeric(df["w2_slope_raw_sigma"], errors="coerce")
        thr = float(rc.get("slope_sigma_min", 5.0))
        raw_fading = int(((s1 <= -thr) & (s2 <= -thr)).sum())
        raw_rising = int(((s1 >= thr) & (s2 >= thr)).sum())
    rep = {"n_stars_screened": int(len(df)),
           "n_rise_candidates": int(df["is_candidate"].sum()) if len(df) else 0,
           "screen_counts": {str(k): int(v) for k, v in counts.items()},
           "ensemble": {"applied": bool(scr.get("ensemble_correct", True)),
                        "mode": (str(scr.get("ensemble_mode", "stratified"))
                                 if scr.get("ensemble_correct", True) else "none"),
                        "stratified": strat_rep,
                        "bin_yr": float(scr["ensemble_bin_yr"]),
                        "min_stars": int(scr["ensemble_min_stars"]),
                        "drift": summarise_offsets(ens_offsets),
                        "offsets": ens_offsets,
                        "raw_two_band_fading_5sigma": raw_fading,
                        "raw_two_band_rising_5sigma": raw_rising,
                        "note": ("offsets are the per-bin median of (mag - star median) over "
                                 "the shard's stars, subtracted before the rise test; "
                                 "raw_* count stars whose UNCORRECTED bare linear slope is "
                                 ">= slope_sigma_min in both bands")},
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
    parent = _shard_parent(out, shard, n_shards)
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


def _aggregate_ensemble(screens: list[dict]) -> dict:
    """The zero-point drift each shard measured, and the raw two-band 5-sigma counts."""
    per = []
    raw_f = raw_r = 0
    for s in screens:
        e = s.get("ensemble") or {}
        if not e:
            continue
        raw_f += int(e.get("raw_two_band_fading_5sigma") or 0)
        raw_r += int(e.get("raw_two_band_rising_5sigma") or 0)
        per.append({"tag": s.get("tag"), "drift": e.get("drift")})
    return {"per_shard_drift": per, "raw_two_band_fading_5sigma": raw_f,
            "raw_two_band_rising_5sigma": raw_r,
            "note": ("per-shard median zero-point offsets by 0.25-yr bin were subtracted "
                     "before the rise test (seti.ignition.ensemble); raw_* count stars whose "
                     "UNCORRECTED bare linear slope is >= 5 sigma in both bands")}


def _coverage(sweeps: list[dict], conf: dict) -> dict:
    """Which part of the sky the tiles-mode sweep actually finished."""
    tc = {**DEFAULT_TILES, **(conf.get("tiles") or {})}
    tc["abs_b_min_deg"] = float(conf["sample"].get("abs_b_min_deg", tc["abs_b_min_deg"]))
    tiles_all = sky_tiles(tc)
    n_sky = int(len(tiles_all))
    area_sky = float(tiles_all["area_deg2"].sum())
    n_shards = max([int(s.get("n_shards") or 0) for s in sweeps] + [0])
    tiles_seen: dict[str, dict] = {}
    routes: dict[str, int] = {}
    for s in sweeps:
        for t in s.get("tiles") or []:
            tiles_seen[str(t.get("tile"))] = t
            if t.get("status") in ("OK", "QUERY_RETURNED_ZERO_ROWS") and t.get("parent_route"):
                routes[str(t["parent_route"])] = routes.get(str(t["parent_route"]), 0) + 1
    recs = list(tiles_seen.values())
    roll = _sweep_rollup(recs, sum(int(s.get("tiles_assigned") or 0) for s in sweeps),
                         n_sky, tiles_all)
    done = [t for t in recs if t.get("status") in ("OK", "QUERY_RETURNED_ZERO_ROWS")]
    shards_reporting = sorted({str(s.get("tag")) for s in sweeps})
    return {"mode": "tiles", "tile_deg": float(tc["dec_step_deg"]),
            "tiles_in_sky": n_sky, "area_sky_deg2": round(area_sky, 1),
            "tiles_assigned": int(sum(int(s.get("tiles_assigned") or 0) for s in sweeps)),
            "sky_fraction_done": round(roll["area_done_deg2"] / area_sky, 4) if area_sky else None,
            "n_shards": n_shards, "shards_reporting": shards_reporting,
            "shards_stopped_on_budget": [str(s.get("tag")) for s in sweeps
                                        if s.get("stopped_on_budget")],
            "parent_routes": routes,
            "route_used": (next(iter(routes)) if len(routes) == 1 else
                           ("mixed" if routes else None)),
            "mixed_routes": len(routes) > 1,
            "dec_range_done": ([round(min(float(t["dec_c"]) for t in done), 1),
                                round(max(float(t["dec_c"]) for t in done), 1)] if done else None),
            **roll,
            "note": ("a DONE tile is a complete selection over its box (its parent query was "
                     "uncapped unless `tiles_capped`), so every count here is exact for the "
                     "area covered and says nothing about the rest of the sky")}


def _write_summary(out: Path, summary: dict) -> None:
    """``summary.json`` plus a mode-tagged copy that a later dispatch cannot erase.

    The channel is run in two modes against different sky --- ``fields`` over
    the 20 ecliptic-pole cones, ``tiles`` over the whole ``|b| > 15`` sky ---
    and both write the channel's one ``summary.json``.  Two dispatches in
    flight at once would therefore overwrite each other's verdict with a
    verdict about a different sample.  ``summary.json`` stays the channel's
    current verdict (that is where every reader looks), and
    ``summary_<mode>.json`` keeps each mode's own record beside it.
    """
    _write(out / "summary.json", summary)
    mode = str(((summary.get("denominators") or {}).get("sample_mode")) or "")
    if mode in ("fields", "allsky", "tiles"):
        _write(out / f"summary_{mode}.json", summary)


def stage_assess(conf: dict, out: Path, *, n_shards_expected: int | None = None,
                 optical_dir: Path | str | None = None, offline: bool = True,
                 online_vet: bool = False, vet_fetchers: dict | None = None) -> dict:
    vc = conf["vet"]
    sample = {}
    if (out / "sample.json").exists():
        try:
            sample = json.loads((out / "sample.json").read_text())
        except Exception:                              # noqa: BLE001
            sample = {}

    # Which shard files belong to THIS run.  The checkout carries the committed
    # files of earlier dispatches (a different matrix leaves s0of8 next to
    # s0of16), so when the expected shard count is known only `*of{n}` files are
    # read; summing a stale shard's failures into this run's verdict would be a
    # disguised report of another run.
    n_hint = int(n_shards_expected or sample.get("n_shards_planned") or 0)
    sweep_paths = sorted(glob.glob(str(out / "sweep_s*of*.json")))
    if sweep_paths and not n_shards_expected:
        # Tiles mode: the sweeps say how many shards there were.
        ns = []
        for fp in sweep_paths:
            try:
                ns.append(int(json.loads(Path(fp).read_text()).get("n_shards") or 0))
            except Exception:                          # noqa: BLE001
                continue
        n_hint = max(ns + [0])

    def _paths(pattern: str) -> list[str]:
        ps = sorted(glob.glob(str(out / pattern)))
        if n_hint:
            ps = [p for p in ps if Path(p).name.split(".")[0].endswith(f"of{n_hint}")]
        return ps

    def _load_json(fp: str) -> dict | None:
        try:
            return json.loads(Path(fp).read_text())
        except Exception:                              # noqa: BLE001
            return None

    screens = [d for d in (_load_json(fp) for fp in _paths("screen_s*of*.json")) if d]
    acquires = [(d.get("rollup") or {}) for d in
                (_load_json(fp) for fp in _paths("acquire_s*of*.json")) if d]
    sweeps = [d for d in (_load_json(fp) for fp in _paths("sweep_s*of*.json")) if d]
    coverage = _coverage(sweeps, conf) if sweeps else None
    if sweeps:
        # Tiles mode has no run-level sample.json (a sample.json in the checkout
        # is an earlier fields-mode run's and is ignored): the denominators are
        # the union of what the shards' finished tiles selected, and they are
        # complete selections over that area (no TOP cap hit unless `capped`).
        sample = {"status": "OK" if coverage["n_parent_done_tiles"] else
                  ("QUERY_FAILED" if coverage["tiles_failed"] and not coverage["tiles_done"]
                   else "QUERY_RETURNED_ZERO_ROWS"),
                  "mode": "tiles", "parent_count": coverage["n_parent_done_tiles"],
                  "n_rows_pulled": coverage["n_parent_done_tiles"],
                  "n_after_local_cuts": coverage["n_parent_done_tiles"],
                  "subsample_fraction": 1.0, "n_shards_planned": coverage["n_shards"],
                  "routes": coverage["parent_routes"], "route_used": coverage["route_used"],
                  "mixed_routes": coverage["mixed_routes"],
                  "degraded": (["mixed_parent_routes:" + "+".join(sorted(coverage["parent_routes"]))]
                               if coverage["mixed_routes"] else []),
                  "route_endpoints": {ROUTE_ESA: "https://gea.esac.esa.int/tap-server/tap"}}
    frames = []
    for fp in _paths("stars_s*of*.csv"):
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

    n_exp = int(n_hint or max([int(s.get("n_shards", 0) or 0) for s in screens] + [0]))
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
    if sweeps:
        # In tiles mode the acquire rollup is per tile inside the sweep record.
        n_acq_failed = int(coverage["n_stars_failed"])
        n_attempted = int(coverage["n_stars_attempted"])
        n_with_nw = int(coverage["n_stars_ok"])
        if coverage["tiles_failed"]:
            degraded.append(f"tiles_failed:{coverage['tiles_failed']}")
        if coverage["tiles_capped"]:
            degraded.append(f"tiles_capped:{coverage['tiles_capped']}")
    else:
        n_acq_failed = int(sum(int(a.get("n_failed", 0) or 0) for a in acquires))
        n_attempted = int(sum(int(a.get("n_attempted", 0) or 0) for a in acquires))
        n_with_nw = int(sum(int(a.get("n_ok", 0) or 0) for a in acquires))
    if n_acq_failed:
        degraded.append(f"neowise_queries_failed:{n_acq_failed}")
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
    ensemble = _aggregate_ensemble(screens)

    # --- nothing screened: say which archive did not answer ------------------
    probe: dict = {}
    if (out / "probe.json").exists():
        try:
            probe = json.loads((out / "probe.json").read_text())
        except Exception:                              # noqa: BLE001
            probe = {}
    # Every endpoint that was asked, whether or not it answered.  With both
    # parent routes dark this is the whole of what the run learned, so it must
    # survive into summary.json rather than only into a log.
    endpoints = {
        "parent": sample.get("route_endpoints") or {ROUTE_ESA: None, ROUTE_IRSA: {},
                                                    ROUTE_VIZIER: []},
        "parent_route_recommended_by_probe": probe.get("parent_route_recommended"),
        "parent_routes_tried": probe.get("parent_routes_tried"),
        "irsa_tap_probe": {k: (probe.get(ROUTE_IRSA) or {}).get(k)
                           for k in ("status", "usable", "endpoints", "table",
                                     "table_requested", "gaia_only_status", "error",
                                     "n_rows")},
        "vizier_asu_probe": {k: (probe.get("vizier_asu") or {}).get(k)
                             for k in ("status", "usable", "endpoints", "catalogues",
                                       "error", "n_rows")},
        "gaia_shapes_tried": probe.get("gaia_shapes_tried"),
        "errors": (sample.get("route_errors") or [])[:40],
    }
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
                   "endpoints": endpoints, "coverage": coverage, "ensemble": ensemble,
                   "denominators": denominators, "shards": shards, "degraded": degraded,
                   "veto_counters": {"screen": {}, "vet": {"verdicts": {}, "flags": {}}},
                   "stage_counts": {"screened": 0, "rise_candidates": 0, "vetted": 0,
                                    "survivors": 0},
                   "sensitivity": _aggregate_sensitivity(screens),
                   "note": ("nothing was screened, so nothing about ignitions was measured; "
                            "this is NOT a null result and must not be reported as one")}
        _write_summary(out, summary)
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
    # --- the archive rungs (approaching neighbour, optical trend, Gaia scatter)
    # on the stars the pure ladder left standing.  Unreachable is a
    # degradation and an untested check, never a pass.
    online_rec: dict = {}
    if online_vet and vets:
        from .vet import KILL_ORDER
        from .vet_online import online_vet as _online_vet

        alive = [v for v in vets if str(v["verdict"]).startswith("clean")]
        if alive:
            ids = {v["source_id"] for v in alive}
            sub = cands[cands["source_id"].astype(str).isin(ids)]
            online_rec, unreach = _online_vet(sub, df, vc, fetchers=vet_fetchers)
            for k, n in unreach.items():
                if n:
                    degraded.append(f"vet_unreachable:{k}:{n}/{len(sub)}")
            for v in alive:
                o = online_rec.get(v["source_id"]) or {}
                flags = [x for x in str(v.get("flags") or "").split(";") if x] + o.get("flags", [])
                untested = [x for x in str(v.get("untested_checks") or "").split(";") if x]
                untested = [x for x in untested if x != "optical_flatness"] + o.get("untested", [])
                ov = (o.get("optical") or {}).get("status")
                v["optical"] = ov or v.get("optical")
                v["neighbour"] = (o.get("neighbour") or {}).get("status")
                v["neighbour_pred_dmag"] = (o.get("neighbour") or {}).get("pred_dmag")
                v["gaia_scatter"] = (o.get("gaia_scatter") or {}).get("status")
                v["flags"] = ";".join(dict.fromkeys(flags))
                v["untested_checks"] = ";".join(dict.fromkeys(untested))
                verdict = None
                for k in KILL_ORDER:
                    if k in flags:
                        verdict = f"rejected_{k}"
                        break
                if verdict is None:
                    if "optical_flatness" in untested:
                        verdict = "clean_optical_untested"
                    elif untested:
                        verdict = "clean_checks_untested"
                    else:
                        verdict = "clean"
                v["verdict"] = verdict
    vdf = pd.DataFrame(vets)
    if len(vdf):
        vetted = cands.merge(vdf, on="source_id", how="left", suffixes=("", "_vet"))
        vetted = vetted.rename(columns={"verdict": "vet_verdict"})
    else:
        vetted = cands.assign(vet_verdict=pd.Series(dtype=str))
    survivors = vetted[vetted["vet_verdict"].astype(str).str.startswith("clean")] \
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
                        "star_rise_w2_minus_w1_mag", "rise_colour", "rise_colour_ratio",
                        "rise_colour_err", "rise_colour_z", "neighbour", "neighbour_pred_dmag",
                        "gaia_scatter", "vet_verdict", "flags", "optical",
                        "optical_slope_sigma", "untested_checks", "shard_file")
            if c in survivors.columns]
    survivors[slim].to_csv(out / "candidates.csv", index=False)

    screen_counts: dict = {}
    for s in screens:
        for k, v in (s.get("screen_counts") or {}).items():
            screen_counts[k] = screen_counts.get(k, 0) + int(v)
    verdict = VERDICT_CANDIDATES if len(survivors) else VERDICT_NONE
    if online_rec:
        (out / "vet_online.json").write_text(json.dumps(online_rec, indent=1, default=str))
    if degraded:
        verdict = f"{DEGRADED} ({'; '.join(degraded)}); {verdict}"
    summary = {
        "verdict": verdict, "generated_utc": _now(), "endpoints": endpoints,
        "n_candidates": int(len(survivors)), "n_clean": int(len(gold)),
        "n_candidates_optical_untested": int(len(survivors) - len(gold)),
        "denominators": denominators,
        "stage_counts": {"screened": n_screened, "rise_candidates": int(len(cands)),
                         "vetted": int(len(vetted)), "survivors": int(len(survivors)),
                         "clean": int(len(gold))},
        "veto_counters": {"screen": screen_counts, "vet": summarise(vets)},
        "sensitivity": _aggregate_sensitivity(screens),
        "coverage": coverage, "ensemble": ensemble,
        "shards": shards, "degraded": degraded, "offline": bool(offline),
        "online_vet": bool(online_vet),
        "optical_dir": str(optical_dir) if optical_dir else None,
        "config": {"rise": conf["rise"], "vet": {k: v for k, v in vc.items()
                                                  if k != "star_forming_boxes"}},
        "note": ("a candidate here is a star whose W1 and W2 both rise monotonically over "
                 ">= 5 yr at >= 5 sigma with a ramp preferred over an impulsive born excess, "
                 "and that survived the contaminant ladder; NO_IGNITION_CANDIDATE is a count "
                 "over n_stars_screened, not an occurrence limit, and is not written up "
                 "(CLAUDE.md)"),
    }
    _write_summary(out, summary)
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
                 query_fn=None, cone_fn=None, upload_fn=None, field_fn=None,
                 asu_fetch_fn=None, vizier: bool = True, irsa_fetch_fn=None,
                 irsa: bool = True, n_shards_expected: int | None = None,
                 time_budget_s: float | None = None, max_tiles: int | None = None,
                 tile_deg: float | None = None, online_vet: bool = False,
                 vet_fetchers: dict | None = None) -> dict:
    """Run one stage, a comma list, or all of them.  Returns the last report."""
    conf = conf if conf is not None else load_ignition_config(config_path)
    if tile_deg:
        conf = _deep_update(conf, {"tiles": {"dec_step_deg": float(tile_deg)}})
    out = Path(out_dir) if out_dir else Path("results") / "ignition"
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage in ("all", "", None) else tuple(s.strip() for s in stage.split(","))
    rep: dict = {}
    t0 = _time.monotonic()
    for s in stages:
        if s == "sweep":
            rep = stage_sweep(conf, out, shard=shard, n_shards=n_shards, route=route,
                              query_fn=query_fn, cone_fn=cone_fn, upload_fn=upload_fn,
                              asu_fetch_fn=asu_fetch_fn, irsa_fetch_fn=irsa_fetch_fn,
                              vizier=vizier, irsa=irsa, time_budget_s=time_budget_s,
                              max_tiles=max_tiles)
        elif s == "probe":
            rep = stage_probe(conf, out, query_fn=query_fn, cone_fn=cone_fn,
                              upload_fn=upload_fn, asu_fetch_fn=asu_fetch_fn,
                              irsa_fetch_fn=irsa_fetch_fn)
        elif s == "sample":
            rep = stage_sample(conf, out, n_shards=n_shards, max_stars=max_stars,
                               query_fn=query_fn, mode=mode, asu_fetch_fn=asu_fetch_fn,
                               vizier=vizier, irsa_fetch_fn=irsa_fetch_fn, irsa=irsa)
        elif s == "acquire":
            rep = stage_acquire(conf, out, shard=shard, n_shards=n_shards, max_stars=max_stars,
                                route=route, cone_fn=cone_fn, upload_fn=upload_fn,
                                field_fn=field_fn)
        elif s == "screen":
            rep = stage_screen(conf, out, shard=shard, n_shards=n_shards, seed=seed)
        elif s == "assess":
            n_exp = n_shards_expected or (n_shards if stage == "all" else None)
            rep = stage_assess(conf, out, n_shards_expected=n_exp, optical_dir=optical_dir,
                               online_vet=online_vet, vet_fetchers=vet_fetchers)
        else:
            raise SystemExit(f"unknown stage {s!r}; choose from "
                             f"{STAGES + EXTRA_STAGES + ('all',)}")
    print(f"[ignition] {stage}: done in {_time.monotonic() - t0:.0f}s")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(prog="seti ignition",
                                description="IGNITION (S61): an infrared excess being born on "
                                            "an old star")
    p.add_argument("--stage", default="all",
                   help="probe|sample|acquire|screen|assess|all or a comma list; "
                        "`sweep` is the tiles-mode shard stage (sample + acquire per tile)")
    p.add_argument("--budget-s", type=float, default=0.0,
                   help="sweep: stop starting tiles after this many seconds "
                        "(default config sweep.time_budget_s)")
    p.add_argument("--max-tiles", type=int, default=0, help="sweep: cap on tiles per shard")
    p.add_argument("--tile-deg", type=float, default=0.0,
                   help="tiles: box side in degrees (default config tiles.dec_step_deg)")
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
    p.add_argument("--no-irsa", action="store_true",
                   help="do not fall back to the gaia_only x IRSA route when no ESA query "
                        "shape answers (the ESA route is always tried first either way)")
    p.add_argument("--no-vizier", action="store_true",
                   help="do not fall back to the VizieR ASU route when neither the ESA "
                        "archive nor the IRSA route answers")
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--online-vet", action="store_true",
                   help="assess: run the archive rungs of the ladder (Gaia neighbours, "
                        "ZTF/ASAS-SN optical trend, Gaia scatter) on the survivors")
    a = p.parse_args(argv)
    shard, n = parse_shard(a.shard)
    n_shards = a.shards or n
    rep = ignition_run(a.stage, out_dir=a.out_dir or None, shard=shard, n_shards=n_shards,
                       max_stars=a.max_stars or None, route=a.route or None,
                       mode=a.mode or None, optical_dir=a.optical_dir or None, seed=a.seed,
                       config_path=a.config or None, vizier=not a.no_vizier,
                       irsa=not a.no_irsa, n_shards_expected=(a.shards or None),
                       time_budget_s=(a.budget_s or None), max_tiles=(a.max_tiles or None),
                       tile_deg=(a.tile_deg or None), online_vet=a.online_vet)
    v = rep.get("verdict") if isinstance(rep, dict) else None
    if v:
        print(f"[ignition] verdict: {v}")
    return 0


if __name__ == "__main__":                            # pragma: no cover
    raise SystemExit(main())


__all__ = ["DEFAULTS", "DEFAULT_PROBE", "DEFAULT_SCREEN", "DEFAULT_SWEEP", "EXTRA_STAGES",
           "STAGES", "ignition_run", "load_ignition_config", "main", "parse_shard",
           "screen_epochs", "shard_rows", "stage_acquire", "stage_assess", "stage_probe",
           "stage_sample", "stage_screen", "stage_sweep"]
