"""Stage orchestration for METRONOME.  Writes ``results/metronome/``.

Stages
------
``probe``    schema discovery on every configured catalogue: the real table,
             its real column names, the resolved roles, the row count, the
             time-system guess.  Writes ``probe.json``.  Nothing is fetched.
``acquire``  pull the flare event tables and the rotation-period tables
             (chunked, retried, logged).  Writes ``data/<cat>_events.parquet``,
             ``data/rotation_<mission>.parquet`` and ``acquisition_log.json``.
``screen``   per catalogue (shardable): cross-star coincidence removal, the
             observing-window model, the per-star scan + nulls.  Writes
             ``stars_<cat>[_s<i>of<n>].csv`` and ``screen_<cat>[...].json``.
``assess``   merge every ``stars_*.csv``, BH-FDR, vetting context (rotation
             from the acquired tables; VSX / Gaia vari / ZTF cones for the
             shortlist), tiers, calibration.  Writes ``summary.json`` and
             ``candidates.json``.

Verdict vocabulary (``summary.json["verdict"]``)
------------------------------------------------
``NO_DATA_REACHED``               no catalogue produced events
``QUERY_RETURNED_ZERO_ROWS``      every catalogue answered, with nothing usable
``NO_CLOCK_CANDIDATES``           stars were scanned; nothing reached interest
``CLOCK_CANDIDATES_PENDING_VET``  >= 1 star at interest or candidate tier
A ``DEGRADED_SOURCE`` prefix marks a run in which some catalogue failed or a
veto catalogue could not be reached.  None of these is ever written up as a
result; a null is a reason to change the question (CLAUDE.md).

Entry points
------------
``metronome_run(cfg=None, stage="all", ...)`` and ``main(argv=None)`` so the
module runs as ``python -m seti.metronome.run --stage probe``.  The CLI wiring
in ``seti.cli`` is the parent session's.
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

from .clock import DEFAULT_NULL, DEFAULT_SCAN, analyze_star, cross_star_coincidence
from .vet import DEFAULT_VET, assign_tiers, calibrate_jitter, rejection_counters
from .windows import (
    guess_time_system,
    kepler_quarter_windows,
    star_windows,
    windows_from_events,
)

DEFAULTS: dict = {
    "catalogues": {},
    "rotation_catalogues": {},
    "position_tables": {"kepler": ["V/133/kic"], "tess": ["IV/39/tic82", "IV/38/tic"]},
    "variability_catalogues": {},
    "cone_radius_arcsec": 3.0,
    "windows": {"bin_days": 0.1, "min_gap_days": 0.5, "min_expected_in_gap": 20.0,
                "min_events_for_data_driven": 2000,
                "pad_days": 0.5, "drop_expected": 5.0,
                "cadence_days": {"kepler": 0.020434, "tess": 0.0013889}},
    "cross_star": {"bin_days": {"kepler": 0.0409, "tess": 0.0069}, "min_stars": 5,
                   "tail_p": 1e-6},
    "scan": dict(DEFAULT_SCAN),
    "null": dict(DEFAULT_NULL),
    "vet": dict(DEFAULT_VET),
    "acquire": {"chunk_rows": 50000, "max_rows": 0, "rotation_max_rows": 0},
}

VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_ZERO = "QUERY_RETURNED_ZERO_ROWS"
VERDICT_NONE = "NO_CLOCK_CANDIDATES"
VERDICT_PENDING = "CLOCK_CANDIDATES_PENDING_VET"
DEGRADED = "DEGRADED_SOURCE"


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


def load_metronome_config(cfg=None, path: Path | None = None) -> dict:
    """``config/metronome.yaml`` over :data:`DEFAULTS`; a missing file degrades."""
    try:
        import yaml
        if path is None:
            root = Path(cfg.root) if cfg is not None else Path(__file__).resolve().parents[3]
            path = root / "config" / "metronome.yaml"
        if not Path(path).exists():
            return _deep_update(DEFAULTS, {})
        return _deep_update(DEFAULTS, yaml.safe_load(Path(path).read_text()) or {})
    except Exception as exc:                              # noqa: BLE001
        print(f"[metronome] config not loaded ({exc!r}); using defaults")
        return _deep_update(DEFAULTS, {})


def _out_root(cfg, out_root) -> Path:
    if out_root is not None:
        return Path(out_root)
    return (Path(cfg.root) if cfg is not None else Path(".")) / "results" / "metronome"


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _enabled_catalogues(conf: dict, catalogues=None) -> dict:
    cats = {k: v for k, v in (conf.get("catalogues") or {}).items()
            if v.get("enabled", True)}
    if catalogues:
        want = {c.strip() for c in catalogues if c.strip()}
        cats = {k: v for k, v in cats.items() if k in want}
    return cats


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def stage_probe(conf: dict, out: Path, *, catalogues=None, query_fn=None,
                log=None) -> dict:
    from .acquire import AcquisitionLog, discover_event_table, tap_query

    log = log or AcquisitionLog()
    query_fn = query_fn or tap_query
    cats = _enabled_catalogues(conf, catalogues)
    found = {}
    for name, spec in cats.items():
        disc = discover_event_table(name, spec["preferred"], tuple(spec.get("keywords") or ()),
                                    query_fn=query_fn, log=log)
        d = disc.as_dict()
        d["mission"] = spec.get("mission")
        d["preferred"] = spec["preferred"]
        # One-row peek to guess the time system, only through a verified column.
        tcol = disc.roles.get("t_peak") or disc.roles.get("t_start")
        if disc.table and tcol:
            try:
                peek = query_fn(f'SELECT TOP 200 "{tcol}" FROM "{disc.table}"')
                vals = pd.to_numeric(peek.iloc[:, 0], errors="coerce").to_numpy()
                d["time_system_guess"] = guess_time_system(vals, spec.get("mission", ""))
                d["time_median"] = float(np.nanmedian(vals)) if len(vals) else None
                log.record(f"peek_{name}", f'SELECT TOP 200 "{tcol}" FROM "{disc.table}"',
                           rows=int(len(peek)))
            except Exception as exc:                      # noqa: BLE001
                d["time_system_guess"] = "unknown"
                log.record(f"peek_{name}", f'SELECT TOP 200 "{tcol}" FROM "{disc.table}"',
                           error=repr(exc))
        found[name] = d
    rep = {"stage": "probe", "generated_utc": _now(), "catalogues": found,
           "n_usable": sum(1 for d in found.values() if d["status"] == "OK"),
           "acquisition": log.as_dict()}
    _write(out / "probe.json", rep)
    print(f"[metronome] probe: {rep['n_usable']}/{len(found)} catalogues usable")
    return rep


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------
MISSION_TIME_OFFSET = {"kepler": ("BKJD", 2454833.0), "tess": ("BTJD", 2457000.0)}


def normalise_time_system(df: pd.DataFrame, guess: str, mission: str
                          ) -> tuple[pd.DataFrame, str]:
    """Bring ``t_peak / t_start / t_end`` to the mission-native offset system.

    The data-driven windows work in any consistent system, but the published
    Kepler quarter fallback is in BKJD, and a BJD catalogue would otherwise
    miss every quarter silently.  Only the guessed BJD / MJD cases are shifted;
    the guess and the resulting system are both recorded in ``acquire.json``.
    """
    native, off = MISSION_TIME_OFFSET.get(mission.lower(), (guess, 0.0))
    shift = 0.0
    if guess == "BJD":
        shift = -off
    elif guess == "MJD":
        shift = 2400000.5 - off
    else:
        return df, guess
    out = df.copy()
    for c in ("t_peak", "t_start", "t_end"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") + shift
    return out, native


def stage_acquire(conf: dict, out: Path, *, catalogues=None, query_fn=None,
                  max_rows: int | None = None, log=None) -> dict:
    from .acquire import (
        AcquisitionLog,
        discover_and_fetch_rotation,
        discover_event_table,
        fetch_events,
        tap_query,
    )

    log = log or AcquisitionLog()
    query_fn = query_fn or tap_query
    acq = conf.get("acquire") or {}
    max_rows = int(acq.get("max_rows") or 0) if max_rows is None else int(max_rows)
    cats = _enabled_catalogues(conf, catalogues)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    per_cat = {}
    for name, spec in cats.items():
        disc = discover_event_table(name, spec["preferred"], tuple(spec.get("keywords") or ()),
                                    query_fn=query_fn, log=log)
        rec = {"table": disc.table, "roles": disc.roles, "discovery_status": disc.status,
               "route": disc.route, "mission": spec.get("mission"), "n_rows_catalogue": disc.n_rows}
        if disc.table is None:
            rec.update({"status": disc.status, "n_events": 0, "n_stars": 0})
            per_cat[name] = rec
            continue
        df = fetch_events(disc, query_fn=query_fn, log=log,
                          chunk_rows=int(acq.get("chunk_rows", 50000)),
                          max_rows=max_rows or None)
        if not len(df) or "t_peak" not in df.columns or "star_id" not in df.columns:
            failed = any(s["stage"] == f"fetch_{name}" and s["status"] == "QUERY_FAILED"
                         for s in log.stages)
            rec.update({"status": "QUERY_FAILED" if failed else "QUERY_RETURNED_ZERO_ROWS",
                        "n_events": 0, "n_stars": 0})
            per_cat[name] = rec
            continue
        df["mission"] = spec.get("mission")
        df["catalogue"] = name
        guess = guess_time_system(df["t_peak"].to_numpy(), spec.get("mission", ""))
        df, native = normalise_time_system(df, guess, str(spec.get("mission", "")))
        path = data / f"{name}_events.parquet"
        df.to_parquet(path, index=False)
        rec.update({"status": "OK", "n_events": int(len(df)),
                    "n_stars": int(df["star_id"].nunique()), "path": str(path),
                    "time_system_guess": guess, "time_system_native": native,
                    "t_peak_source": str(df["t_peak_source"].iloc[0])
                    if "t_peak_source" in df else "t_peak"})
        per_cat[name] = rec

    rot = {}
    missions = {spec.get("mission") for spec in cats.values()}
    for mission in sorted(m for m in missions if m):
        frames, recs = [], []
        for rspec in (conf.get("rotation_catalogues") or {}).get(mission, []) or []:
            df, r = discover_and_fetch_rotation(
                rspec["name"], rspec.get("preferred") or "", tuple(rspec.get("keywords") or ()),
                query_fn=query_fn, log=log,
                max_rows=int(acq.get("rotation_max_rows") or 0) or None)
            recs.append(r)
            if len(df):
                frames.append(df)
        if frames:
            rdf = pd.concat(frames, ignore_index=True).dropna(subset=["prot"])
            rdf = rdf.drop_duplicates("star_id", keep="first")
            rdf.to_parquet(data / f"rotation_{mission}.parquet", index=False)
            rot[mission] = {"n_rows": int(len(rdf)), "sources": recs}
        else:
            rot[mission] = {"n_rows": 0, "sources": recs}

    rep = {"stage": "acquire", "generated_utc": _now(), "catalogues": per_cat,
           "rotation": rot, "acquisition": log.as_dict()}
    _write(out / "acquire.json", rep)
    log.write(out / "acquisition_log.json")
    n_ok = sum(1 for r in per_cat.values() if r.get("status") == "OK")
    print(f"[metronome] acquire: {n_ok}/{len(per_cat)} catalogues fetched")
    return rep


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------
def build_mission_windows(events: pd.DataFrame, mission: str, conf: dict):
    """Data-driven windows from the whole catalogue, or the published fallback."""
    w = conf.get("windows") or {}
    cad = float((w.get("cadence_days") or {}).get(mission, 0.020434))
    t = events["t_peak"].to_numpy(dtype=float)
    kw = dict(bin_days=float(w.get("bin_days", 0.5)),
              min_gap_days=float(w.get("min_gap_days", 0.5)),
              min_expected_in_gap=float(w.get("min_expected_in_gap", 20.0)),
              cadence_days=cad)
    if len(t) >= int(w.get("min_events_for_data_driven", 2000)):
        return windows_from_events(t, label=f"{mission}_data_driven", **kw)
    if mission == "kepler":
        return kepler_quarter_windows(cadence_days=cad)
    # TESS with too few events: the catalogue's own event density is still the
    # best available window model, only coarser; label it so a reader knows.
    return windows_from_events(t, label=f"{mission}_sparse_data_driven", **kw)


def screen_catalogue(events: pd.DataFrame, name: str, mission: str, conf: dict, *,
                     shard: int = 0, n_shards: int = 1, max_stars: int | None = None,
                     seed: int = 20260906, progress_every: int = 200) -> tuple[list[dict], dict]:
    """Cross-star removal, windows, then the per-star scan and nulls."""
    ev = events.dropna(subset=["t_peak", "star_id"]).copy()
    ev["star_id"] = ev["star_id"].astype(str)
    n_events_in = int(len(ev))
    cs = conf.get("cross_star") or {}
    bin_days = float((cs.get("bin_days") or {}).get(mission, 0.04))
    cc = cross_star_coincidence(ev["star_id"].to_numpy(), ev["t_peak"].to_numpy(dtype=float),
                                bin_days=bin_days, min_stars=int(cs.get("min_stars", 5)),
                                tail_p=float(cs.get("tail_p", 1e-6)))
    removed_by_star = ev.loc[cc["remove"], "star_id"].value_counts()
    ev = ev.loc[~cc["remove"]]
    mission_w = build_mission_windows(ev, mission, conf)
    wconf = conf.get("windows") or {}
    sc, nc = conf.get("scan") or {}, conf.get("null") or {}
    n_min = int(sc.get("n_min", 8))
    counts = ev["star_id"].value_counts()
    stars = sorted(counts[counts >= n_min].index.tolist())
    n_stars_all = int(len(counts))
    stars = [s for i, s in enumerate(stars) if i % max(int(n_shards), 1) == int(shard)]
    if max_stars:
        stars = stars[:int(max_stars)]
    rng = np.random.default_rng(int(seed) + int(shard))
    grouped = ev.groupby("star_id")
    records: list[dict] = []
    t_start = _time.monotonic()
    for i, sid in enumerate(stars):
        g = grouped.get_group(sid)
        t = g["t_peak"].to_numpy(dtype=float)
        e = g["energy"].to_numpy(dtype=float) if "energy" in g else None
        w = star_windows(t, mission_w, pad_days=float(wconf.get("pad_days", 0.5)),
                         drop_expected=float(wconf.get("drop_expected", 5.0)))
        rec = analyze_star(t, w, e, sc, nc, rng)
        rec.pop("windows", None)
        rec.update({"star_key": f"{mission}:{sid}", "star_id": sid, "catalogue": name,
                    "mission": mission, "n_windows": w.n, "observed_days": round(w.total, 3),
                    "n_removed_cross_star": int(removed_by_star.get(sid, 0)),
                    "prot_catalogue": float(pd.to_numeric(g["prot"], errors="coerce").median())
                    if "prot" in g else float("nan")})
        records.append(rec)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[metronome] {name} shard {shard}/{n_shards}: {i + 1}/{len(stars)} stars, "
                  f"{_time.monotonic() - t_start:.0f}s")
    rep = {"stage": "screen", "catalogue": name, "mission": mission, "shard": int(shard),
           "n_shards": int(n_shards), "generated_utc": _now(),
           "n_events_in": n_events_in, "n_events_after_cross_star": int(len(ev)),
           "cross_star": {k: v for k, v in cc.items() if k != "remove"},
           "n_stars_in_catalogue": n_stars_all,
           "n_stars_with_n_min": int((counts >= n_min).sum()),
           "n_stars_scanned_this_shard": int(len(records)),
           "n_null_computed": int(sum(1 for r in records if r.get("null_computed"))),
           "mission_windows": mission_w.as_dict(),
           "elapsed_s": round(_time.monotonic() - t_start, 1)}
    return records, rep


def stage_screen(conf: dict, out: Path, *, catalogues=None, shard: int = 0, n_shards: int = 1,
                 max_stars: int | None = None, seed: int = 20260906,
                 events_by_catalogue: dict | None = None) -> dict:
    cats = _enabled_catalogues(conf, catalogues)
    reports = {}
    for name, spec in cats.items():
        mission = str(spec.get("mission", "kepler"))
        if events_by_catalogue is not None and name in events_by_catalogue:
            ev = events_by_catalogue[name]
        else:
            path = out / "data" / f"{name}_events.parquet"
            if not path.exists():
                reports[name] = {"stage": "screen", "catalogue": name, "status": "NO_EVENTS_FILE",
                                 "generated_utc": _now()}
                continue
            ev = pd.read_parquet(path)
        if not len(ev):
            reports[name] = {"stage": "screen", "catalogue": name, "status": "NO_EVENTS",
                             "generated_utc": _now()}
            continue
        recs, rep = screen_catalogue(ev, name, mission, conf, shard=shard, n_shards=n_shards,
                                     max_stars=max_stars, seed=seed)
        tag = f"{name}" + (f"_s{shard}of{n_shards}" if n_shards > 1 else "")
        pd.DataFrame(recs).to_csv(out / f"stars_{tag}.csv", index=False)
        rep["status"] = "OK"
        _write(out / f"screen_{tag}.json", rep)
        reports[name] = rep
        print(f"[metronome] screen {tag}: {rep['n_stars_scanned_this_shard']} stars scanned, "
              f"{rep['n_null_computed']} with nulls, {rep['elapsed_s']}s")
    return reports


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------
def _load_rotation(out: Path, mission: str) -> pd.DataFrame:
    p = out / "data" / f"rotation_{mission}.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:                                 # noqa: BLE001
            return pd.DataFrame()
    return pd.DataFrame()


def _prot_for(row, rot: pd.DataFrame) -> tuple[float, str | None]:
    """Rotation period: acquired rotation catalogue first, flare table's own second."""
    sid = str(row.get("star_id"))
    if len(rot) and "star_id" in rot:
        hit = rot[rot["star_id"] == sid]
        if len(hit):
            p = float(hit["prot"].iloc[0])
            if np.isfinite(p) and p > 0:
                return p, str(hit["prot_source"].iloc[0]) if "prot_source" in hit else "rotation"
    p = float(row.get("prot_catalogue", np.nan))
    if np.isfinite(p) and p > 0:
        return p, "flare_catalogue"
    return float("nan"), None


def stage_assess(conf: dict, out: Path, *, offline: bool = False, query_fn=None, cone_fn=None,
                 records: list[dict] | None = None, log=None,
                 acquire_report: dict | None = None) -> dict:
    from .acquire import AcquisitionLog

    log = log or AcquisitionLog()
    vconf = conf.get("vet") or {}
    if records is None:
        frames = []
        for fp in sorted(glob.glob(str(out / "stars_*.csv"))):
            if Path(fp).name == "stars_vetted.csv":
                continue
            try:
                d = pd.read_csv(fp, dtype={"star_id": str, "star_key": str})
            except Exception:                             # noqa: BLE001
                continue
            if len(d):
                frames.append(d)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(df) and "star_key" in df:
            df = df.drop_duplicates("star_key")
        records = df.to_dict(orient="records") if len(df) else []
    screens = []
    for fp in sorted(glob.glob(str(out / "screen_*.json"))):
        try:
            screens.append(json.loads(Path(fp).read_text()))
        except Exception:                                 # noqa: BLE001
            continue
    if acquire_report is None and (out / "acquire.json").exists():
        try:
            acquire_report = json.loads((out / "acquire.json").read_text())
        except Exception:                                 # noqa: BLE001
            acquire_report = None

    # --- verdict when nothing was scanned -----------------------------------
    acq_cats = (acquire_report or {}).get("catalogues") or {}
    statuses = {str(v.get("status")) for v in acq_cats.values()}
    n_scanned = sum(1 for r in records if r.get("status") == "scanned")
    if n_scanned == 0:
        # Three different facts, kept apart: nothing answered (an archive-access
        # statement); everything answered with nothing (a statement about the
        # catalogues); events arrived but no star had n_min of them (a statement
        # about the sample).  None is a statement about clocks.
        if "OK" in statuses:
            verdict = "NO_STAR_REACHED_N_MIN"
        elif acq_cats and statuses <= {"QUERY_RETURNED_ZERO_ROWS"}:
            verdict = VERDICT_ZERO
        else:
            verdict = VERDICT_NO_DATA
        summary = {"verdict": verdict, "generated_utc": _now(),
                   "n_stars_scanned": 0, "n_stars_records": int(len(records)),
                   "funnel": {"stars_records": int(len(records)), "scanned": 0},
                   "rejection_counters": rejection_counters([]),
                   "tiers": {"none": 0, "watch": 0, "interest": 0, "candidate": 0},
                   "acquisition_per_catalogue": {k: {"status": v.get("status"),
                                                     "n_events": v.get("n_events", 0),
                                                     "n_stars": v.get("n_stars", 0),
                                                     "table": v.get("table")}
                                                 for k, v in acq_cats.items()},
                   "acquisition": (acquire_report or {}).get("acquisition", log.as_dict()),
                   "note": ("no star was scanned, so nothing about clocks was measured; "
                            "this is NOT a null result and must not be reported as one")}
        _write(out / "summary.json", summary)
        _write(out / "candidates.json", {"generated_utc": summary["generated_utc"],
                                         "verdict": verdict, "candidates": []})
        print(f"[metronome] assess: {verdict}")
        return summary

    # --- vetting context ------------------------------------------------------
    df = pd.DataFrame(records)
    contexts: dict[str, dict] = {}
    rot_by_mission = {m: _load_rotation(out, m) for m in df["mission"].dropna().unique()}
    # Preliminary BH at the watch level decides who gets the (network) cone crossmatch.
    prelim = assign_tiers(records, {}, vconf)
    shortlist = [r for r in prelim if r.get("fdr_watch")]
    vari: dict = {}
    reached_by_star: dict = {}
    vari_sources = set(conf.get("variability_catalogues") or {})
    if shortlist and not offline:
        from .acquire import fetch_positions_by_id, fetch_variable_context, tap_query
        query_fn = query_fn or tap_query
        for mission in sorted({str(r["mission"]) for r in shortlist}):
            ids = sorted({str(r["star_id"]) for r in shortlist if str(r["mission"]) == mission})
            pos = fetch_positions_by_id(ids, mission, query_fn=query_fn, log=log,
                                        tables=conf.get("position_tables"))
            if not len(pos):
                continue
            v, rch = fetch_variable_context(pos, conf.get("variability_catalogues") or {},
                                            cone_fn=cone_fn, log=log,
                                            radius_arcsec=float(conf.get("cone_radius_arcsec", 3.0)))
            for sid, lst in v.items():
                vari.setdefault(f"{mission}:{sid}", []).extend(lst)
            for sid, srcs in rch.items():
                reached_by_star.setdefault(f"{mission}:{sid}", set()).update(srcs)
    short_keys = {r.get("star_key") for r in shortlist}
    # Per-source fraction of the shortlist whose cone answered, for the summary.
    reached = {k: (float(np.mean([k in reached_by_star.get(s, set()) for s in short_keys]))
                   if short_keys else float("nan")) for k in sorted(vari_sources)}
    for r in records:
        key = r.get("star_key")
        prot, src = _prot_for(r, rot_by_mission.get(str(r.get("mission")), pd.DataFrame()))
        contexts[key] = {"mission": r.get("mission"), "prot": prot, "prot_source": src,
                         "catalogued_periods": vari.get(key, []),
                         "variability_catalogues_reached": bool(vari_sources) and
                         vari_sources <= reached_by_star.get(key, set())}
    vetted = assign_tiers(records, contexts, vconf)
    counters = rejection_counters(vetted)
    calib = calibrate_jitter(vetted, vconf)

    vdf = pd.DataFrame(vetted)
    keep_cols = [c for c in vdf.columns if c != "veto_detail"]
    vdf[keep_cols].to_csv(out / "stars_vetted.csv", index=False)
    cands = [r for r in vetted if r.get("tier") in ("candidate", "interest")]
    watch = [r for r in vetted if r.get("tier") == "watch"]
    cands.sort(key=lambda r: float(r.get("p_window", 1.0)))
    watch.sort(key=lambda r: float(r.get("p_window", 1.0)))

    degraded = []
    for k, v in acq_cats.items():
        if v.get("status") != "OK":
            degraded.append(f"{k}:{v.get('status')}")
    if not offline and shortlist:
        unreached = [k for k in sorted(vari_sources)
                     if not (np.isfinite(reached.get(k, np.nan)) and reached[k] >= 1.0)]
        if unreached:
            degraded.append("variability_catalogues:" + ",".join(unreached))
    for m, rdf in rot_by_mission.items():
        if not len(rdf):
            degraded.append(f"rotation_{m}:none")

    verdict = VERDICT_PENDING if cands else VERDICT_NONE
    if degraded:
        verdict = f"{DEGRADED} ({'; '.join(degraded)}); {verdict}"

    n_events_in = sum(int(s.get("n_events_in", 0)) for s in screens)
    n_removed = sum(int((s.get("cross_star") or {}).get("n_removed_events", 0)) for s in screens)
    coverage = {
        "catalogues": {k: {"status": v.get("status"), "n_events": v.get("n_events", 0),
                           "n_stars": v.get("n_stars", 0), "table": v.get("table"),
                           "time_system_guess": v.get("time_system_guess")}
                       for k, v in acq_cats.items()},
        "n_stars_in_catalogues": int(sum(int(s.get("n_stars_in_catalogue", 0)) for s in screens)),
        "n_stars_with_n_min": int(sum(int(s.get("n_stars_with_n_min", 0)) for s in screens)),
        "min_period_used_days": {m: float(vdf.loc[vdf["mission"] == m, "min_period_used"].median())
                                 for m in vdf["mission"].dropna().unique()
                                 if "min_period_used" in vdf},
        "observed_days_median": float(vdf["observed_days"].median()) if "observed_days" in vdf
        else float("nan"),
        "note": ("coverage is bounded by each catalogue's own flare-detection threshold and "
                 "cadence: a clock whose ticks fall below the catalogue's amplitude/energy "
                 "threshold, or shorter than ~10 cadences, is invisible here by construction"),
    }
    summary = {
        "verdict": verdict, "generated_utc": _now(),
        "n_candidates": int(sum(1 for r in cands if r["tier"] == "candidate")),
        "n_interest": int(sum(1 for r in cands if r["tier"] == "interest")),
        "n_watch": int(len(watch)),
        "funnel": {
            "events_in": n_events_in, "events_removed_cross_star": n_removed,
            "stars_records": int(len(records)), "stars_scanned": n_scanned,
            "stars_null_computed": int(sum(1 for r in records if r.get("null_computed"))),
            "stars_fdr_watch": int(sum(1 for r in vetted if r.get("fdr_watch"))),
            "stars_fdr_significant": int(sum(1 for r in vetted if r.get("fdr_significant"))),
            "stars_watch": int(len(watch)),
            "stars_interest": int(sum(1 for r in cands if r["tier"] == "interest")),
            "stars_candidate": int(sum(1 for r in cands if r["tier"] == "candidate")),
        },
        "rejection_counters": counters,
        "tiers": counters["tiers"],
        "jitter_calibration": calib,
        "coverage": coverage,
        "degraded": degraded,
        "offline": bool(offline),
        "variability_catalogues_reached": reached,
        "cross_star_epochs": {s.get("catalogue"): (s.get("cross_star") or {}).get("bad_bins", [])[:50]
                              for s in screens},
        "acquisition_per_catalogue": coverage["catalogues"],
        "acquisition": {"acquire": (acquire_report or {}).get("acquisition"),
                        "assess": log.as_dict()},
        "config": {"scan": conf.get("scan"), "null": conf.get("null"), "vet": vconf},
        "note": ("a clock candidate here is a STATISTICAL statement about catalogued peak "
                 "times and is pending light-curve inspection; NO_CLOCK_CANDIDATES is a "
                 "count, not an occurrence limit, and is not written up (CLAUDE.md)"),
    }
    _write(out / "summary.json", summary)
    slim = ("star_key", "star_id", "catalogue", "mission", "tier", "flags", "first_veto",
            "n_events", "period", "Q", "jitter", "f_in_window", "gap_integer_frac",
            "n_gaps_used", "cycle_occupancy", "h_max", "p_window", "p_window_source",
            "p_shuffle", "energy_phase_rho", "energy_phase_p", "t0", "mean_phase",
            "prot_catalogue", "wn_n_trials", "wn_n_exceed", "veto_detail")
    _write(out / "candidates.json", {
        "generated_utc": summary["generated_utc"], "verdict": verdict,
        "candidates": [{k: r.get(k) for k in slim} | {"context": contexts.get(r.get("star_key"))}
                       for r in cands],
        "watch": [{k: r.get(k) for k in slim} for r in watch[:200]],
    })
    print(f"[metronome] assess: {verdict} — {summary['n_candidates']} candidate, "
          f"{summary['n_interest']} interest, {summary['n_watch']} watch of {n_scanned} scanned")
    return summary


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
STAGES = ("probe", "acquire", "screen", "assess")


def metronome_run(cfg=None, stage: str = "all", catalogues=None, *, shard: int = 0,
                  n_shards: int = 1, max_stars: int | None = None, max_rows: int | None = None,
                  offline: bool = False, seed: int = 20260906, out_root=None,
                  query_fn=None, cone_fn=None) -> dict:
    """Run one stage or all of them.  Returns the last stage's report."""
    conf = load_metronome_config(cfg)
    out = _out_root(cfg, out_root)
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage in ("all", "", None) else tuple(s.strip() for s in stage.split(","))
    rep: dict = {}
    for s in stages:
        if s == "probe":
            rep = stage_probe(conf, out, catalogues=catalogues, query_fn=query_fn)
        elif s == "acquire":
            rep = stage_acquire(conf, out, catalogues=catalogues, query_fn=query_fn,
                                max_rows=max_rows)
        elif s == "screen":
            rep = stage_screen(conf, out, catalogues=catalogues, shard=shard, n_shards=n_shards,
                               max_stars=max_stars, seed=seed)
        elif s == "assess":
            rep = stage_assess(conf, out, offline=offline, query_fn=query_fn, cone_fn=cone_fn)
        else:
            raise SystemExit(f"unknown stage {s!r}; choose from {STAGES}")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(prog="seti metronome",
                                description="METRONOME: clocks in stellar flare timing (S28)")
    p.add_argument("--stage", default="all", help="probe|acquire|screen|assess|all or a comma list")
    p.add_argument("--catalogues", default="",
                   help="comma-separated catalogue keys from config/metronome.yaml (default all)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--max-stars", type=int, default=0, help="cap per catalogue shard (0 = none)")
    p.add_argument("--max-rows", type=int, default=-1, help="acquire row cap (-1 = config)")
    p.add_argument("--offline", action="store_true", help="assess without network crossmatches")
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--out-root", default="", help="results directory (default results/metronome)")
    a = p.parse_args(argv)
    cats = [c for c in a.catalogues.split(",") if c.strip()] or None
    from ..config import load_config
    cfg = load_config()
    rep = metronome_run(cfg, stage=a.stage, catalogues=cats, shard=a.shard, n_shards=a.n_shards,
                        max_stars=a.max_stars or None, max_rows=None if a.max_rows < 0 else a.max_rows,
                        offline=a.offline, seed=a.seed, out_root=a.out_root or None)
    v = rep.get("verdict") if isinstance(rep, dict) else None
    if v:
        print(f"[metronome] verdict: {v}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())


__all__ = ["DEFAULTS", "STAGES", "load_metronome_config", "main", "metronome_run",
           "screen_catalogue", "stage_acquire", "stage_assess", "stage_probe",
           "stage_screen"]
