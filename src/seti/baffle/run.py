"""Stage orchestration for BAFFLE.  Writes ``results/baffle/``.

Stages
------
``probe``    schema probe of the archive tables (``TOP 1 *`` each) + one timed
             ``TOP 100`` query per track over a 1° band.  Writes ``probe.json``.
``acquire``  both tracks, sharded on 5° declination bands (``--shard``/
             ``--n-shards`` round-robin, or ``--dec-band-index`` explicit),
             checkpointed per leaf to ``chunks/*.parquet``; the per-chunk
             ledger goes to ``acquisition_ledger[_s<i>of<n>].json``.
``screen``   assemble chunks -> ``sample.parquet`` / ``missing_sample.parquet``,
             fit the locus (``locus.json``), residuals, the named-veto screen,
             the missing-track screen, the tail asymmetry and the injected
             sensitivity.  Writes ``candidates.csv``, ``vetoed.csv``,
             ``deferred_lpv.csv``, ``missing_candidates.csv``, ``screen.json``
             and ``summary.json``.
``vet``      per-candidate archive checks (``seti.baffle.vet``): Gaia
             neighbours, AllWISE proper, CatWISE / unWISE against the same
             locus, W3 consistency; the missing-track direct match.  Writes
             ``vet.json``, ``vet_table.csv``, ``vetted_candidates.csv``,
             ``missing_vet.json``, ``missing_vet.csv``.
``patch``    ``from .patch import run_patch_stage`` (another agent's module);
             ``run_patch_stage(candidates_df, out_dir, cfg) -> dict`` merged
             under ``summary["patch"]`` and written to ``patch.json``.
``radio``    ``from .radio import run_radio_stage`` (another agent's module);
             ``run_radio_stage(cfg, out_dir) -> dict`` -> ``summary["radio"]``.
``assess``   rewrite ``summary.json`` from what is on disk and print it.
``all``      every stage in order.

Verdict vocabulary (``summary.json["verdict"]``): ``NO_DATA_REACHED``,
``DEGRADED_SOURCE (...)`` prefix, ``NO_MIDIR_DEFICIT_SURVIVOR`` /
``MIDIR_DEFICIT_CANDIDATES_PENDING_VET (n=..)`` for the deficit track and
``NO_MISSING_COUNTERPART_SURVIVOR`` /
``MISSING_COUNTERPART_CANDIDATES_PENDING_VET (n=..)`` for the missing track,
joined with `` | ``.  None of these is ever written up as a result: a null is a
reason to change the question (CLAUDE.md).

Entry points: ``baffle_run(cfg=None, stage="all", ...)`` and ``main(argv)`` so
``python -m seti.baffle.run --stage screen`` and ``python -m seti.cli baffle
--stage screen`` both work.
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

from . import acquire as acq
from . import screen as scr
from .locus import (
    DEFAULT_LOCUS_CFG,
    Locus,
    fit_locus,
    locus_quality_mask,
    residuals,
    tail_asymmetry,
)

CHANNEL = "baffle"
STAGES = ("probe", "acquire", "screen", "vet", "patch", "radio", "assess")
TRACKS = ("deficit", "missing")

DEFAULTS: dict = {
    "acquire": {
        "g_max": 15.0, "g_max_missing": 13.0, "pre_cut_w1": -0.15, "pre_cut_w2": -0.15,
        "locus_random_index_max": acq.LOCUS_RANDOM_INDEX_MAX,
        "gaia_random_index_max": acq.GAIA_RANDOM_INDEX_MAX,
        "dec_band_deg": 5.0, "top": 2_000_000, "max_split_depth": 6,
        "query_timeout_s": 1500, "count_timeout_s": 900, "retries_per_transport": 2,
        "transport_cooldown_queries": 25, "probe_band_deg": 1.0,
        "neighbour_radius_arcmin": 3.0,
    },
    "locus": dict(DEFAULT_LOCUS_CFG),
    "screen": {k: (dict(v) if isinstance(v, dict) else v)
               for k, v in scr.DEFAULT_SCREEN_CFG.items()},
    "sensitivity": {"inject_mags": [0.2, 0.3, 0.5, 1.0], "max_stars": 20000},
    "output": {"max_vetoed_rows": 50000},
    "vet": {},                      # seti.baffle.vet.DEFAULTS fills every key
    "radio": {},
    "patch": {"max_objects": 200},
}

_COMPACT_COLS = (
    "source_id", "ra", "dec", "l", "b", "ecl_lat", "parallax", "parallax_over_error",
    "pmra", "pmdec", "ruwe", "phot_g_mean_mag", "bp_rp", "phot_variable_flag",
    "non_single_star", "ipd_frac_multi_peak", "is_locus_sample", "lum_class", "jk",
    "j_m", "h_m", "ks_m", "ks_msigcom", "tmass_ph_qual", "w1mpro", "w1mpro_error", "w1snr",
    "w1rchi2", "w2mpro", "w2mpro_error", "w2snr", "w2rchi2", "w3mpro", "w3snr", "w4mpro",
    "w4snr", "cc_flags", "ext_flag", "var_flag", "ph_qual", "wise_angular_distance",
    "wise_number_of_neighbours", "wise_number_of_mates", "resid_w1", "sig_w1", "resid_w2",
    "sig_w2", "resid_w3", "sig_w3", "w3_status", "w4_status", "first_veto", "vetoes",
    "bad_astrometry", "high_pm_epoch_risk", "pm_total_mas_yr", "neighbours_checked",
    "etz", "nearby", "distance_pc", "dec_band_index",
)


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


def load_baffle_config(cfg=None, path: Path | None = None) -> dict:
    """``config/baffle.yaml`` over :data:`DEFAULTS`; a missing file degrades."""
    try:
        import yaml
        if path is None:
            root = Path(cfg.root) if cfg is not None else Path(__file__).resolve().parents[3]
            path = root / "config" / f"{CHANNEL}.yaml"
        if not Path(path).exists():
            return _deep_update(DEFAULTS, {})
        return _deep_update(DEFAULTS, yaml.safe_load(Path(path).read_text()) or {})
    except Exception as exc:                                        # noqa: BLE001
        print(f"[baffle] config not loaded ({exc!r}); using defaults")
        return _deep_update(DEFAULTS, {})


def _out_root(cfg, out_root) -> Path:
    if out_root is not None:
        return Path(out_root)
    return (Path(cfg.root) if cfg is not None else Path(".")) / "results" / CHANNEL


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:                                               # noqa: BLE001
        return None


def _parse_tracks(tracks) -> list[str]:
    if tracks is None:
        return list(TRACKS)
    if isinstance(tracks, str):
        tracks = tracks.split(",")
    out = [t.strip() for t in tracks if t and t.strip()]
    bad = [t for t in out if t not in TRACKS]
    if bad:
        raise SystemExit(f"unknown track(s) {bad}; choose from {TRACKS}")
    return out or list(TRACKS)


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def stage_probe(conf: dict, out: Path, *, tracks=None, runner=None) -> dict:
    a = conf["acquire"]
    runner = runner or acq.run_gaia_query
    tables = {}
    for key, table in acq.TABLES.items():
        t0 = _time.time()
        try:
            cols = acq.probe_columns(table, runner, timeout_s=float(a["count_timeout_s"]))
            tables[key] = {"table": table, "status": "OK", "n_columns": len(cols),
                           "columns": cols, "seconds": round(_time.time() - t0, 1)}
        except Exception as exc:                                    # noqa: BLE001
            tables[key] = {"table": table, "status": "FAILED", "error": repr(exc),
                           "seconds": round(_time.time() - t0, 1)}
    try:
        cm = acq.resolve_columns(runner, timeout_s=float(a["count_timeout_s"]))
        columns = cm.as_dict()
    except Exception as exc:                                        # noqa: BLE001
        cm = acq.default_columns()
        columns = dict(cm.as_dict(), error=repr(exc))
    width = float(a.get("probe_band_deg", 1.0))
    lo, hi = 0.0, width
    timed = {}
    for track in _parse_tracks(tracks):
        build = acq.build_deficit_query if track == "deficit" else acq.build_missing_query
        q = build(lo, hi, a, cm, top=100)
        t0 = _time.time()
        df, rec = runner(q, label=f"probe {track}", expect_rows=None,
                         timeout_s=float(a["query_timeout_s"]))
        timed[track] = {"status": rec["status"], "n_rows": int(rec.get("n_rows") or 0),
                        "seconds": round(_time.time() - t0, 1), "transport": rec.get("transport"),
                        "error": rec.get("error"), "dec_band": [lo, hi],
                        "columns_returned": [str(c) for c in df.columns][:120],
                        "query": rec.get("query")}
        if track == "missing":
            qd = acq.build_missing_denominator_query(lo, hi, a, cm)
            t0 = _time.time()
            dd, rd = runner(qd, label="probe missing-denominator", expect_rows=None,
                            timeout_s=float(a["count_timeout_s"]))
            timed["missing_denominator"] = {
                "status": rd["status"], "n_rows": int(rd.get("n_rows") or 0),
                "seconds": round(_time.time() - t0, 1), "error": rd.get("error"),
                "grouped_ok": bool({"babs_bin", "g_bin", "n"} <= set(dd.columns)),
                "query": rd.get("query")}
    n_ok = sum(1 for t in tables.values() if t["status"] == "OK")
    rep = {"stage": "probe", "generated_utc": _now(), "tables": tables, "columns": columns,
           "timed_queries": timed, "n_tables_ok": n_ok, "n_tables": len(tables),
           "status": ("PROBE_OK" if n_ok == len(tables) and all(
               v["status"] in (acq.QUERY_OK, acq.QUERY_ZERO) for k, v in timed.items()
               if k in TRACKS) else "PROBE_DEGRADED" if n_ok else "PROBE_FAILED")}
    _write(out / "probe.json", rep)
    print(f"[baffle] probe: {n_ok}/{len(tables)} tables reachable; "
          + "; ".join(f"{k}={v['status']} {v['n_rows']} rows in {v['seconds']} s"
                      for k, v in timed.items()), flush=True)
    return rep


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------
def stage_acquire(conf: dict, out: Path, *, tracks=None, shard: int = 0, n_shards: int = 1,
                  dec_band_index=None, runner=None, columns: acq.ColumnMap | None = None,
                  assume_columns: bool = False) -> dict:
    a = conf["acquire"]
    runner = runner or acq.run_gaia_query
    bands = acq.dec_bands(float(a["dec_band_deg"]))
    if dec_band_index is not None:
        idx = [int(i) for i in (dec_band_index if isinstance(dec_band_index, (list, tuple))
                                else str(dec_band_index).split(","))
               if str(i).strip() != ""]
    else:
        idx = acq.bands_for_shard(bands, shard, n_shards)
    if columns is None:
        if assume_columns:
            columns = acq.default_columns()
        else:
            try:
                columns = acq.resolve_columns(runner, timeout_s=float(a["count_timeout_s"]))
            except Exception as exc:                                # noqa: BLE001
                print(f"[baffle] column resolution failed ({exc!r}); assuming names",
                      flush=True)
                columns = acq.default_columns()
    _write(out / (f"columns_s{shard}of{n_shards}.json" if n_shards > 1 else "columns.json"),
           columns.as_dict())
    ledger = acq.AcquisitionLedger()
    ledger_path = out / (f"acquisition_ledger_s{shard}of{n_shards}.json" if n_shards > 1
                         else "acquisition_ledger.json")
    chunk_dir = out / "chunks"
    per_track = {}
    for track in _parse_tracks(tracks):
        print(f"[baffle] acquire {track}: bands {idx} (shard {shard}/{n_shards})", flush=True)
        df = acq.fetch_track(track, a, chunk_dir, ledger, band_indices=idx, cols=columns,
                             runner=runner, screen_missing=conf["screen"].get("missing"))
        ledger.save(ledger_path)          # checkpoint the ledger after every track
        per_track[track] = {"n_rows": int(len(df)),
                            "summary": acq.summarise_ledger(ledger.entries, track)}
        print(f"[baffle] acquire {track}: {len(df)} rows; "
              f"{per_track[track]['summary']['acquisition_verdict']}", flush=True)
    rep = {"stage": "acquire", "generated_utc": _now(), "shard": shard, "n_shards": n_shards,
           "bands": [{"index": i, "dec_lo": bands[i][0], "dec_hi": bands[i][1]} for i in idx],
           "tracks": per_track, "columns": columns.as_dict(), "ledger": str(ledger_path.name)}
    ledger.save(ledger_path)
    _write(out / (f"acquire_s{shard}of{n_shards}.json" if n_shards > 1 else "acquire.json"), rep)
    return rep


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------
def merged_ledger(out: Path) -> list[dict]:
    entries: list[dict] = []
    for p in sorted(glob.glob(str(out / "acquisition_ledger*.json"))):
        doc = _read_json(p)
        if isinstance(doc, dict):
            entries.extend(doc.get("entries") or [])
        elif isinstance(doc, list):
            entries.extend(doc)
    return entries


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in _COMPACT_COLS if c in df.columns]
    return df[cols] if cols else df


def assemble_sample(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chunk_dir = out / "chunks"
    deficit = acq.load_chunks(chunk_dir, "deficit")
    missing = acq.load_chunks(chunk_dir, "missing")
    den = acq.load_denominators(chunk_dir) if chunk_dir.exists() else pd.DataFrame()
    if len(deficit):
        deficit.to_parquet(out / "sample.parquet", index=False)
    if len(missing):
        missing.to_parquet(out / "missing_sample.parquet", index=False)
    return deficit, missing, den


def stage_screen(conf: dict, out: Path, *, neighbours: pd.DataFrame | None = None,
                 run_sensitivity: bool = True) -> dict:
    lc, sc = conf["locus"], conf["screen"]
    entries = merged_ledger(out)
    acq_summary = {t: acq.summarise_ledger(entries, t) for t in TRACKS}
    deficit, missing, den = assemble_sample(out)
    rep: dict = {"stage": "screen", "generated_utc": _now(), "acquisition": acq_summary,
                 "config": {"locus": lc, "screen": sc}}

    # ---- deficit track ----------------------------------------------------
    d_acq = acq_summary["deficit"]["acquisition_verdict"]
    if len(deficit) and not entries:
        d_acq = "COMPLETE_UNLEDGERED"      # chunks present, ledger lost: still data
    if len(deficit):
        locus = fit_locus(deficit, lc)
        locus.save(out / "locus.json")
        ok, _ = locus_quality_mask(deficit, lc)
        r = residuals(deficit, locus, lc)
        res = scr.screen_deficit(r, sc, neighbours)
        cand, vet, deferred = res["candidates"], res["vetoed"], res["deferred_lpv"]
        cand.to_csv(out / "candidates.csv", index=False)
        max_v = int(conf.get("output", {}).get("max_vetoed_rows", 50000))
        vet_c = _compact(vet)
        if len(vet_c) > max_v and "sig_w2" in vet_c.columns:
            vet_c = vet_c.sort_values("sig_w2").head(max_v)
        vet_c.to_csv(out / "vetoed.csv", index=False)
        _compact(deferred).to_csv(out / "deferred_lpv.csv", index=False)
        tails = tail_asymmetry(r[ok], lc)
        sens = (scr.sensitivity(deficit, locus, lc, sc,
                                tuple(conf["sensitivity"]["inject_mags"]),
                                max_stars=int(conf["sensitivity"]["max_stars"]))
                if run_sensitivity else {"skipped": True})
        rep["deficit"] = {
            "n_rows": int(len(deficit)), "n_locus_grade": int(ok.sum()),
            "locus_meta": {k: v for k, v in locus.meta.items() if k != "config"},
            "locus_classes": {c: {b: len(v["centers"]) for b, v in bands.items()}
                              for c, bands in locus.bins.items()},
            "funnel": res["funnel"], "veto_counters": res["counters"],
            "first_veto_counters": res["counters_first_veto"],
            "report_flags": res["report_flags"],
            "neighbours_not_checked": res["neighbours_not_checked"],
            "denominators": res["denominators"], "tail_asymmetry": tails,
            "sensitivity": sens,
            "n_vetoed_rows_written": int(len(vet_c)), "n_vetoed_rows_total": int(len(vet)),
        }
        rep["verdict_deficit"] = scr.deficit_verdict(d_acq, int(len(deficit)),
                                                     int(res["funnel"]["n_candidates"]))
        rep["candidates_preview"] = _compact(cand).head(50).to_dict(orient="records")
    else:
        rep["deficit"] = {"n_rows": 0, "funnel": {"n_screened": 0, "n_candidates": 0}}
        rep["verdict_deficit"] = scr.deficit_verdict(d_acq, 0, 0)
        for name in ("candidates.csv", "vetoed.csv", "deferred_lpv.csv"):
            pd.DataFrame().to_csv(out / name, index=False)

    # ---- missing track ----------------------------------------------------
    m_acq = acq_summary["missing"]["acquisition_verdict"]
    if len(missing) and not entries:
        m_acq = "COMPLETE_UNLEDGERED"
    if len(missing) or m_acq not in ("NO_DATA_REACHED", "NO_QUERY_ATTEMPTED"):
        mres = scr.screen_missing(missing, sc, den)
        mres["candidates"].to_csv(out / "missing_candidates.csv", index=False)
        rep["missing"] = {"n_rows": int(len(missing)), "funnel": mres["funnel"],
                          "counters": mres["counters"], "fractions": mres["fractions"],
                          "denominators": mres["denominators"],
                          "n_denominator_bands": int(den["dec_band_index"].nunique())
                          if len(den) else 0}
        rep["verdict_missing"] = scr.missing_verdict(m_acq, int(len(missing)),
                                                     int(mres["funnel"]["n_candidates"]))
    else:
        rep["missing"] = {"n_rows": 0, "funnel": {"n_missing_rows": 0, "n_candidates": 0}}
        rep["verdict_missing"] = scr.missing_verdict(m_acq, 0, 0)
        pd.DataFrame().to_csv(out / "missing_candidates.csv", index=False)

    rep["verdict"] = scr.combine_verdicts(rep["verdict_deficit"], rep["verdict_missing"])
    _write(out / "screen.json", rep)
    stage_assess(conf, out, quiet=True)
    print(f"[baffle] screen: deficit rows {rep['deficit']['n_rows']} -> "
          f"{rep['deficit']['funnel'].get('n_candidates', 0)} candidates; missing rows "
          f"{rep['missing']['n_rows']} -> {rep['missing']['funnel'].get('n_candidates', 0)}; "
          f"verdict {rep['verdict']}", flush=True)
    return rep


# ---------------------------------------------------------------------------
# vet
# ---------------------------------------------------------------------------
def stage_vet(conf: dict, out: Path, *, gaia_fetcher=None, matchers=None) -> dict:
    from .vet import run_vet_stage

    rep = run_vet_stage(conf, out, gaia_fetcher=gaia_fetcher, matchers=matchers,
                        locus_cfg=conf.get("locus"))
    stage_assess(conf, out, quiet=True)
    return rep


# ---------------------------------------------------------------------------
# patch / radio (other agents' modules, guarded)
# ---------------------------------------------------------------------------
def _load_candidates(out: Path) -> tuple[pd.DataFrame, str]:
    """``vetted_candidates.csv`` when the vet has run, else ``candidates.csv``."""
    for name in ("vetted_candidates.csv", "candidates.csv"):
        p = out / name
        if not p.exists():
            continue
        if p.stat().st_size == 0:
            return pd.DataFrame(), name
        try:
            return pd.read_csv(p), name
        except pd.errors.EmptyDataError:
            return pd.DataFrame(), name
    return pd.DataFrame(), "none"


def stage_patch(conf: dict, out: Path, *, max_objects: int | None = None) -> dict:
    try:
        from .patch import run_patch_stage
    except ImportError as exc:
        rep = {"stage": "patch", "status": "MODULE_MISSING", "generated_utc": _now(),
               "error": (f"seti.baffle.patch.run_patch_stage is not available ({exc!r}); "
                         "the patch stage is owned by another module — nothing was run")}
        _write(out / "patch.json", rep)
        print(f"[baffle] patch: {rep['error']}", flush=True)
        return rep
    cands, cand_source = _load_candidates(out)
    cfg = _deep_update(conf, {"patch": {"max_objects": int(max_objects)}}) \
        if max_objects is not None else conf
    if max_objects is not None and len(cands) > int(max_objects):
        cands = cands.head(int(max_objects))
    t0 = _time.time()
    try:
        rep = dict(run_patch_stage(cands, out, cfg) or {})
        rep.setdefault("status", "OK")
    except Exception as exc:                                        # noqa: BLE001
        rep = {"status": "FAILED", "error": repr(exc)}
    rep.update(stage="patch", generated_utc=_now(), n_candidates_in=int(len(cands)),
               candidates_source=cand_source, seconds=round(_time.time() - t0, 1))
    _write(out / "patch.json", rep)
    stage_assess(conf, out, quiet=True)
    return rep


def stage_radio(conf: dict, out: Path) -> dict:
    try:
        from .radio import run_radio_stage
    except ImportError as exc:
        rep = {"stage": "radio", "status": "MODULE_MISSING", "generated_utc": _now(),
               "error": (f"seti.baffle.radio.run_radio_stage is not available ({exc!r}); "
                         "the radio stage is owned by another module — nothing was run")}
        _write(out / "radio.json", rep)
        print(f"[baffle] radio: {rep['error']}", flush=True)
        return rep
    # The radio module writes its own candidates.csv / summary.json: give it a
    # subdirectory so it can never clobber the deficit-track files.  An EMPTY
    # ``radio:`` placeholder in config/baffle.yaml must not shadow the module's
    # own config/baffle_radio.yaml, so it is dropped before the hand-off.
    radio_out = out / "radio"
    cfg_radio = dict(conf)
    if not cfg_radio.get("radio"):
        cfg_radio.pop("radio", None)
    t0 = _time.time()
    try:
        rep = dict(run_radio_stage(cfg_radio, radio_out) or {})
        rep.setdefault("status", "OK")
        rep.setdefault("out_dir", str(radio_out))
    except Exception as exc:                                        # noqa: BLE001
        rep = {"status": "FAILED", "error": repr(exc)}
    rep.update(stage="radio", generated_utc=_now(), seconds=round(_time.time() - t0, 1))
    _write(out / "radio.json", rep)
    stage_assess(conf, out, quiet=True)
    return rep


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------
def stage_assess(conf: dict, out: Path, *, quiet: bool = False) -> dict:
    screen = _read_json(out / "screen.json") if (out / "screen.json").exists() else None
    probe = _read_json(out / "probe.json") if (out / "probe.json").exists() else None
    entries = merged_ledger(out)
    acq_summary = {t: acq.summarise_ledger(entries, t) for t in TRACKS}
    summary: dict = {"channel": CHANNEL, "generated_utc": _now(),
                     "acquisition": acq_summary,
                     "probe_status": (probe or {}).get("status")}
    if screen:
        for k in ("deficit", "missing", "verdict_deficit", "verdict_missing",
                  "candidates_preview", "config"):
            if k in screen:
                summary[k] = screen[k]
        summary["screened_utc"] = screen.get("generated_utc")
        verdict = screen.get("verdict") or scr.combine_verdicts(
            screen.get("verdict_deficit"), screen.get("verdict_missing"))
    else:
        verdict = scr.combine_verdicts(
            scr.deficit_verdict(acq_summary["deficit"]["acquisition_verdict"], 0, 0),
            scr.missing_verdict(acq_summary["missing"]["acquisition_verdict"], 0, 0))
        summary["note"] = "no screen.json on disk: nothing has been screened"
    for sect in ("vet", "missing_vet", "patch", "radio"):
        p = out / f"{sect}.json"
        if p.exists():
            summary[sect] = _read_json(p)
    summary["verdict_screen"] = verdict
    vet = summary.get("vet") or {}
    if isinstance(vet, dict) and vet.get("verdict_deficit_after_vet"):
        # the ledger keeps the query log; the summary keeps the verdict and counters
        summary["vet"] = {k: v for k, v in vet.items() if k != "ledger"}
        mv = (summary.get("missing_vet") or {}).get("missing_vet_verdict") if isinstance(
            summary.get("missing_vet"), dict) else None
        summary["verdict_after_vet"] = scr.combine_verdicts(vet["verdict_deficit_after_vet"], mv)
        verdict = summary["verdict_after_vet"]
    summary["verdict"] = verdict
    summary["files"] = sorted(p.name for p in out.iterdir()
                              if p.is_file() and p.suffix in (".json", ".csv"))
    _write(out / "summary.json", summary)
    if not quiet:
        show = {k: v for k, v in summary.items()
                if k not in ("candidates_preview", "config", "acquisition")}
        print(json.dumps(show, indent=2, default=_json_default)[:8000])
        print(f"[baffle] verdict: {verdict}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def baffle_run(cfg=None, stage: str = "all", tracks=None, *, shard: int = 0, n_shards: int = 1,
               dec_band_index=None, g_max: float | None = None, out_root=None,
               max_patch_objects: int | None = None, runner=None, neighbours=None,
               assume_columns: bool = False, run_sensitivity: bool = True,
               gaia_fetcher=None, matchers=None) -> dict:
    """Run one stage, a comma list, or all of them.  Returns the last report."""
    conf = load_baffle_config(cfg)
    if g_max is not None:
        conf["acquire"]["g_max"] = float(g_max)
    out = _out_root(cfg, out_root)
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage in ("all", "", None) else tuple(s.strip() for s in stage.split(","))
    rep: dict = {}
    for s in stages:
        if s == "probe":
            rep = stage_probe(conf, out, tracks=tracks, runner=runner)
        elif s == "acquire":
            rep = stage_acquire(conf, out, tracks=tracks, shard=shard, n_shards=n_shards,
                                dec_band_index=dec_band_index, runner=runner,
                                assume_columns=assume_columns)
        elif s == "screen":
            rep = stage_screen(conf, out, neighbours=neighbours, run_sensitivity=run_sensitivity)
        elif s == "vet":
            rep = stage_vet(conf, out, gaia_fetcher=gaia_fetcher, matchers=matchers)
        elif s == "patch":
            rep = stage_patch(conf, out, max_objects=max_patch_objects)
        elif s == "radio":
            rep = stage_radio(conf, out)
        elif s == "assess":
            rep = stage_assess(conf, out)
        else:
            raise ValueError(f"unknown stage {s!r}; choose from {STAGES + ('all',)}")
    return rep


def _stage_arg(value: str) -> str:
    toks = [t.strip() for t in str(value).split(",") if t.strip()]
    bad = [t for t in toks if t not in STAGES + ("all",)]
    if bad or not toks:
        raise argparse.ArgumentTypeError(
            f"unknown stage(s) {bad or [value]}; choose from {STAGES + ('all',)}")
    return ",".join(toks)


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The stage flags, shared by ``python -m seti.baffle.run`` and ``seti baffle``."""
    p.add_argument("--stage", default="all", type=_stage_arg, metavar="STAGE",
                   help="probe|acquire|screen|vet|patch|radio|assess|all or a comma list")
    p.add_argument("--tracks", default="deficit,missing",
                   help="comma-separated: deficit,missing")
    p.add_argument("--shard", type=int, default=0, help="acquire shard index")
    p.add_argument("--n-shards", type=int, default=1,
                   help="acquire: declination bands are dealt round-robin over this many shards")
    p.add_argument("--dec-band-index", default="",
                   help="acquire: explicit comma list of 5-degree band indices (0..35); "
                        "overrides --shard/--n-shards")
    p.add_argument("--g-max", type=float, default=None, help="override acquire.g_max")
    p.add_argument("--max-patch-objects", type=int, default=None,
                   help="cap on candidates handed to the patch stage")
    p.add_argument("--assume-columns", action="store_true",
                   help="acquire without the runtime column probe (offline dry runs)")
    p.add_argument("--no-sensitivity", action="store_true",
                   help="screen: skip the injected-deficit recovery check")
    p.add_argument("--out-root", default="", help="results directory (default results/baffle)")
    return p


def build_parser(prog: str = "seti baffle") -> argparse.ArgumentParser:
    return add_arguments(argparse.ArgumentParser(
        prog=prog, description="BAFFLE: reciprocal mid-IR absorbing screens on Sun-star lines"))


def _cmd_baffle(a, cfg) -> int:
    """``seti baffle ...`` entry (``seti.cli``) and the tail of :func:`main`."""
    rep = baffle_run(cfg, stage=a.stage, tracks=a.tracks, shard=a.shard, n_shards=a.n_shards,
                     dec_band_index=(a.dec_band_index or None), g_max=a.g_max,
                     out_root=a.out_root or None, max_patch_objects=a.max_patch_objects,
                     assume_columns=a.assume_columns, run_sensitivity=not a.no_sensitivity)
    v = rep.get("verdict") if isinstance(rep, dict) else None
    if v:
        print(f"[baffle] verdict: {v}")
    return 0


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    from ..config import load_config
    try:
        cfg = load_config()
    except Exception:                                               # noqa: BLE001
        cfg = None
    return _cmd_baffle(a, cfg)


if __name__ == "__main__":                                          # pragma: no cover
    raise SystemExit(main())


__all__ = ["CHANNEL", "DEFAULTS", "STAGES", "TRACKS", "Locus", "add_arguments", "assemble_sample",
           "baffle_run", "build_parser", "load_baffle_config", "main", "merged_ledger", "stage_acquire",
           "stage_assess", "stage_patch", "stage_probe", "stage_radio", "stage_screen", "stage_vet"]
