"""Stage orchestration for CRADLE.  Writes ``results/cradle/``.

Stages
------
``probe``    one minimal live call per route: the ESA mirrors' column names,
             the join in each shape (``TOP 5`` on one HEALPix unit, time-boxed),
             the three control stars through the same join, IRSA's AllWISE and
             ``irs_enhv211`` columns, a NEOWISE cone, the VizieR tables' existence.
             Writes ``probe.json`` (committed by the workflow).
``acquire``  shard ``i/n`` of the HEALPix units -> ``parent_s{i}of{n}.csv`` +
             ``acquire_s{i}of{n}.json``, checkpointed per unit; shard 0 also
             pulls the controls.
``screen``   every shard merged -> the empirical loci -> W3/W4 excess -> T_bb,
             f, log(f/f_max) -> ``shortlist.csv`` + ``screen.json``.
``ages``     shard ``i/n`` of the shortlist: Gaia neighbours, IRSA profile fits,
             SIMBAD, E(B-V), NEOWISE, VizieR catalogues, Spitzer/IRS + S53,
             then the age indicators -> ``enriched_s{i}of{n}.csv``.
``assess``   the kill list, the classes, ``candidates.csv``, ``controls.json``,
             ``summary.json``.

Verdict vocabulary (``summary.json["verdict"]``)
------------------------------------------------
``NO_DATA_REACHED``       no parent rows came back (``reason`` names the route)
``NO_CRADLE_CANDIDATE``   stars were screened; none is in the cell, mature and
                          unkilled --- a COUNT over what was searched, with the
                          in-cell/age-undetermined/killed breakdown, never a limit
``CRADLE_CANDIDATES``     >= 1 star in the cell, >= 2 old indicators, no kill
A ``DEGRADED (...)`` prefix names missing shards, failed units, untested vetoes.
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
from .ages import DEFAULT_AGES, assess_ages
from .excess import (
    DEFAULT_EXCESS,
    excess_table,
    fit_disk,
    fit_loci,
    harmonise,
    loci_summary,
    locus_reference_mask,
)
from .mineralogy import DEFAULT_MINERALOGY, score_spectrum, slag_verdict, spectrum_from_table
from .sample import (
    CONTROLS,
    DEFAULT_SAMPLE,
    JOINED_SHAPES,
    SHAPE_GAIA_ONLY,
    _as_bool,
    build_query,
    healpix_units,
    select_parent,
    shard_units,
    unit_label,
)
from .vet import DEFAULT_VET, apply_rules, as_bool, classify, not_excluded

STAGES: tuple[str, ...] = ("probe", "acquire", "screen", "ages", "assess")

DEFAULT_PROBE: dict = {
    "unit_level": 3, "unit_k": 0, "top": 5, "shape_timeout_s": 480.0,
    "total_budget_s": 2400.0, "control_radius_arcsec": 5.0,
}
DEFAULT_ENRICH: dict = {
    "neowise": True, "neowise_time_budget_s": 5400.0, "vizier": True, "irs": True,
    "simbad": True, "ebv": True, "gaia_neighbours": True, "irsa_rows": True,
    "max_stars_per_shard": 0,
}

DEFAULTS: dict = {"sample": DEFAULT_SAMPLE, "excess": DEFAULT_EXCESS, "ages": DEFAULT_AGES,
                  "vet": DEFAULT_VET, "mineralogy": DEFAULT_MINERALOGY, "probe": DEFAULT_PROBE,
                  "enrich": DEFAULT_ENRICH, "vizier_tables": acq.DEFAULT_VIZIER_TABLES,
                  "acquire": {"time_budget_s": 18000.0, "irsa_fallback": True, "max_units": 0}}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _deep_update(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_cradle_config(path: Path | str | None = None) -> dict:
    import yaml  # noqa: PLC0415

    p = Path(path) if path else Path("config") / "cradle.yaml"
    conf = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
    if p.exists():
        got = yaml.safe_load(p.read_text()) or {}
        conf = _deep_update(conf, got)
    return conf


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def _write(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=_json_default))


def _tag(shard: int, n_shards: int) -> str:
    return f"s{int(shard)}of{int(n_shards)}"


def parse_shard(s: str | None) -> tuple[int, int]:
    if not s:
        return 0, 1
    a, b = str(s).split("/")
    i, n = int(a), int(b)
    if n < 1 or i < 0 or i >= n:
        raise ValueError(f"bad shard {s!r}")
    return i, n


def _backends(b: acq.Backends | None) -> acq.Backends:
    return b if b is not None else acq.Backends()


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def stage_probe(conf: dict, out: Path, *, backends: acq.Backends | None = None) -> dict:
    b = _backends(backends)
    pc = {**DEFAULT_PROBE, **(conf.get("probe") or {})}
    sc = {**DEFAULT_SAMPLE, **(conf.get("sample") or {})}
    t0 = _time.monotonic()
    deadline = t0 + float(pc["total_budget_s"])
    gaia = b.gaia or acq.esa_query_fn(sc | {"query_timeout_s": float(pc["shape_timeout_s"])},
                                      deadline=deadline, allow_sync=True, label="cradle_probe")
    irsa = b.irsa or acq.irsa_query_fn(timeout_s=300.0, deadline=deadline)
    rep: dict = {"stage": "probe", "generated_utc": _now(), "esa": {}, "irsa": {}, "vizier": {},
                 "controls": [], "gaia_shapes": [], "shape_working": None}

    # 1. mirror columns
    for name, table in (("allwise", "gaiadr1.allwise_original_valid"),
                        ("tmass", "gaiadr1.tmass_original_valid"),
                        ("tmass_xmatch", "gaiadr3.tmass_psc_xsc_best_neighbour"),
                        ("rotation", str(sc.get("rotation_table"))),
                        ("astrophysical_parameters", "gaiadr3.astrophysical_parameters")):
        cols, rec = acq.probe_columns(gaia, table, label="esa")
        rep["esa"][name] = {"table": table, "status": rec.get("status"), "n_columns": len(cols),
                            "columns": cols[:120], "error": rec.get("error")}
    aw = rep["esa"]["allwise"]["columns"]
    if aw:
        got, missing = acq.resolve_names(aw, {"ext_flag": ["ext_flag", "ext_flg"],
                                             "w4mpro_error": ["w4mpro_error", "w4sigmpro"],
                                             "w3mpro_error": ["w3mpro_error", "w3sigmpro"]})
        rep["esa"]["allwise"]["resolved"] = got
        rep["esa"]["allwise"]["missing"] = missing
    xm = rep["esa"]["tmass_xmatch"]["columns"]
    if xm:
        got, _m = acq.resolve_names(xm, {"id": ["original_ext_source_id", "original_psc_source_id"]})
        rep["esa"]["tmass_xmatch"]["id_column"] = got.get("id")

    # 2. the join, shape by shape, on one unit
    unit = next(u for u in healpix_units(int(pc["unit_level"])) if int(u["k"]) == int(pc["unit_k"]))
    rep["probe_unit"] = unit
    for sh in JOINED_SHAPES + (SHAPE_GAIA_ONLY,):
        if _time.monotonic() > deadline:
            rep["gaia_shapes"].append({"shape": sh, "status": "SKIPPED_BUDGET"})
            continue
        q = build_query(sc, unit=unit, shape=sh, top=int(pc["top"]))
        df, rec = acq.ask(gaia, q, label=f"probe:{unit_label(unit)}:{sh}", service=acq.GAIA_TAP)
        rep["gaia_shapes"].append({"shape": sh, "status": rec.get("status"), "n_rows": rec.get("n_rows"),
                                   "seconds": rec.get("seconds"), "transport": rec.get("transport"),
                                   "error": (rec.get("error") or "")[:400], "query": q[:3000],
                                   "columns": [str(c) for c in df.columns][:80]})
        if rec.get("status") in (acq.STATUS_OK, acq.STATUS_ZERO) and sh in JOINED_SHAPES \
                and rep["shape_working"] is None:
            rep["shape_working"] = sh
    rep["gaia_only_status"] = next((s["status"] for s in rep["gaia_shapes"]
                                    if s["shape"] == SHAPE_GAIA_ONLY), None)

    # 3. controls through the same join
    shapes = ([rep["shape_working"]] + [s for s in JOINED_SHAPES if s != rep["shape_working"]]) \
        if rep["shape_working"] else list(JOINED_SHAPES)
    cdf, crecs = acq.fetch_controls(sc, gaia, resolve_fn=b.simbad_resolve,
                                    radius_arcsec=float(pc["control_radius_arcsec"]), shapes=shapes)
    rep["controls"] = crecs
    rep["controls_found"] = int(len(cdf))
    if len(cdf):
        keep = [c for c in ("control_name", "source_id", "ra", "dec", "parallax", "phot_g_mean_mag",
                            "bp_rp", "ks_m", "w1mpro", "w2mpro", "w3mpro", "w3mpro_error", "w4mpro",
                            "w4mpro_error", "allwise_sep_arcsec", "age_flame", "age_flame_lower",
                            "age_flame_upper", "radial_velocity", "gaia_rot_period_d",
                            "non_single_star", "ruwe") if c in cdf.columns]
        rep["controls_rows"] = json.loads(cdf[keep].to_json(orient="records"))

    # 4. IRSA
    for name, table in (("allwise", acq.IRSA_ALLWISE_TABLE), ("irs", acq.IRS_TABLE)):
        df, rec = acq.ask(irsa, f"SELECT column_name FROM TAP_SCHEMA.columns WHERE table_name = '{table}'",
                          label=f"irsa_columns:{table}", service=acq.IRSA_TAP)
        cols = [str(x) for x in df.iloc[:, 0]] if len(df) else []
        if not cols:
            cols, rec2 = acq.probe_columns(irsa, table, label="irsa")
            rec = {**rec, "fallback": {k: rec2.get(k) for k in ("status", "error")}}
        rep["irsa"][name] = {"table": table, "status": rec.get("status"), "n_columns": len(cols),
                             "columns": cols[:120], "error": rec.get("error")}
    if rep["irsa"]["irs"]["columns"]:
        rep["irsa"]["irs"]["file_like_columns"] = [c for c in rep["irsa"]["irs"]["columns"]
                                                    if any(t in c.lower() for t in ("url", "file", "fname", "path", "spec"))]
    df, rec = acq.ask(irsa, f"SELECT COUNT(*) AS n FROM {acq.IRS_TABLE}", label="irs_count",
                      service=acq.IRSA_TAP)
    rep["irsa"]["irs_count"] = int(pd.to_numeric(df.iloc[0, 0])) if len(df) else None
    # a NEOWISE cone on the first resolved control
    ctrl_pos = next(((c["ra"], c["dec"]) for c in crecs if c.get("ra") is not None), None)
    if ctrl_pos is not None:
        try:
            r = (b.neowise or acq.fetch_neowise_cone)(ctrl_pos[0], ctrl_pos[1], 0.0, 0.0, radius_arcsec=2.5)
            rep["irsa"]["neowise"] = {"status": getattr(r, "status", None), "n_rows": getattr(r, "n_rows", None),
                                      "elapsed_s": getattr(r, "elapsed_s", None),
                                      "error": (getattr(r, "error", "") or "")[:200]}
        except Exception as exc:                           # noqa: BLE001
            rep["irsa"]["neowise"] = {"status": acq.STATUS_FAILED, "error": repr(exc)[:200]}

    # 5. VizieR tables
    vt = conf.get("vizier_tables") or acq.DEFAULT_VIZIER_TABLES
    try:
        from ..metronome.acquire import asu_table_exists  # noqa: PLC0415
        for group in ("known_disks", "rotation"):
            for name, table in (vt.get(group) or {}).items():
                try:
                    ok, attempts = asu_table_exists(table, fetch_fn=b.asu)
                    rep["vizier"][name] = {"table": table, "exists": bool(ok),
                                           "last": (attempts or [{}])[-1] if attempts else None}
                except Exception as exc:                   # noqa: BLE001
                    rep["vizier"][name] = {"table": table, "exists": None, "error": repr(exc)[:200]}
    except Exception as exc:                               # noqa: BLE001
        rep["vizier"]["error"] = repr(exc)[:200]

    joined_ok = rep["shape_working"] is not None
    rep["parent_route"] = "esa_gaia" if joined_ok else (
        "irsa_tap" if (rep["gaia_only_status"] in (acq.STATUS_OK, acq.STATUS_ZERO)
                       and rep["irsa"]["allwise"]["status"] == acq.STATUS_OK) else None)
    rep["verdict"] = ("PARENT_ROUTE_REACHABLE" if rep["parent_route"] else "NO_DATA_REACHED")
    rep["elapsed_s"] = round(_time.monotonic() - t0, 1)
    _write(out / "probe.json", rep)
    print(f"[cradle] probe: {rep['verdict']} shape={rep['shape_working']} route={rep['parent_route']} "
          f"controls_found={rep['controls_found']} in {rep['elapsed_s']} s")
    return rep


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------
def _shape_order(conf: dict, out: Path) -> list[str]:
    want = str((conf.get("sample") or {}).get("query_shape") or "auto")
    pref: list[str] = []
    p = out / "probe.json"
    if p.exists():
        try:
            w = json.loads(p.read_text()).get("shape_working")
            if w in JOINED_SHAPES:
                pref.append(w)
        except Exception:                                  # noqa: BLE001
            pass
    if want in JOINED_SHAPES and want not in pref:
        pref.insert(0, want)
    return pref + [s for s in JOINED_SHAPES if s not in pref]


def stage_acquire(conf: dict, out: Path, *, shard: int = 0, n_shards: int = 1,
                  backends: acq.Backends | None = None, max_units: int | None = None) -> dict:
    b = _backends(backends)
    sc = {**DEFAULT_SAMPLE, **(conf.get("sample") or {})}
    ac = {**DEFAULTS["acquire"], **(conf.get("acquire") or {})}
    t0 = _time.monotonic()
    budget = float(ac.get("time_budget_s") or 18000.0)
    gaia = b.gaia or acq.esa_query_fn(sc, label="cradle_acquire")
    irsa = b.irsa or acq.irsa_query_fn()
    tag = _tag(shard, n_shards)
    store = acq.UnitStore.open(out, tag)
    units = shard_units(healpix_units(int(sc["healpix_level"])), shard, n_shards)
    cap = int(max_units if max_units is not None else (ac.get("max_units") or 0))
    if cap:
        units = units[:cap]
    shapes = _shape_order(conf, out)
    n_ok = n_zero = n_fail = n_partial = n_rows = 0
    n_parent = 0
    parent_measured = 0
    stopped = False
    for u in units:
        label = unit_label(u)
        if label in store.done:
            continue
        if _time.monotonic() - t0 > budget:
            stopped = True
            break
        df, rec = acq.fetch_unit(sc, u, gaia, shapes=shapes, count=bool(sc.get("count_parent", True)))
        if rec.get("status") in (acq.STATUS_FAILED, acq.STATUS_TIMED_OUT) and ac.get("irsa_fallback", True):
            idf, irec = acq.fetch_unit_irsa(sc, u, gaia, irsa)
            rec["irsa_fallback"] = {k: irec.get(k) for k in ("status", "n_rows", "gaia", "allwise")}
            if irec.get("status") in (acq.STATUS_OK, acq.STATUS_ZERO):
                df, rec["status"], rec["route"] = idf, irec["status"], "irsa_tap"
        st = rec.get("status")
        if st == acq.STATUS_OK:
            n_ok += 1
        elif st == acq.STATUS_ZERO:
            n_zero += 1
        elif st == "PARTIAL":
            n_partial += 1
        else:
            n_fail += 1
        if rec.get("n_parent") is not None:
            n_parent += int(rec["n_parent"])
            parent_measured += 1
        n_rows += int(len(df))
        if len(df):
            df = df.copy()
            df["is_control"] = False
            df["control_name"] = ""
        store.write(label, df, {k: rec.get(k) for k in ("label", "status", "shape", "n_rows",
                                                       "n_parent", "seconds", "route", "split",
                                                       "n_children_failed", "attempts")})
        print(f"[cradle] {tag} {label}: {st} rows={len(df)} parent={rec.get('n_parent')} "
              f"shape={rec.get('shape')} {rec.get('seconds')}s")
    controls: list[dict] = []
    if int(shard) == 0 and "controls" not in store.done and not stopped:
        cdf, controls = acq.fetch_controls(sc, gaia, resolve_fn=b.simbad_resolve, shapes=shapes)
        store.write("controls", cdf, {"label": "controls", "status": "OK" if len(cdf) else "NONE",
                                      "n_rows": int(len(cdf)), "records": controls})
    roll = {"stage": "acquire", "shard": int(shard), "n_shards": int(n_shards), "generated_utc": _now(),
            "healpix_level": int(sc["healpix_level"]), "n_units_planned": len(units),
            "n_units_done": len([x for x in store.done if x != "controls"]),
            "n_ok": n_ok, "n_zero": n_zero, "n_partial": n_partial, "n_failed": n_fail,
            "n_rows_this_run": n_rows, "n_parent_gaia_only": n_parent,
            "n_units_parent_measured": parent_measured, "shapes": shapes,
            "stopped_on_budget": stopped, "controls": controls,
            "elapsed_s": round(_time.monotonic() - t0, 1)}
    store.flush({"rollup": roll})
    print(f"[cradle] acquire {tag}: units ok={n_ok} zero={n_zero} partial={n_partial} failed={n_fail} "
          f"rows={n_rows} in {roll['elapsed_s']} s")
    return roll


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------
def _read_parents(out: Path) -> tuple[pd.DataFrame, dict]:
    files = sorted(glob.glob(str(out / "parent_s*of*.csv")))
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    frames = [f for f in frames if len(f)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    acq_files = sorted(glob.glob(str(out / "acquire_s*of*.json")))
    cov = {"n_parent_files": len(files), "n_acquire_files": len(acq_files), "n_units_done": 0,
           "n_units_planned": 0, "n_units_failed": 0, "n_parent_gaia_only": 0, "shards": []}
    for f in acq_files:
        try:
            r = json.loads(Path(f).read_text()).get("rollup") or {}
        except Exception:                                  # noqa: BLE001
            continue
        cov["n_units_done"] += int(r.get("n_units_done") or 0)
        cov["n_units_planned"] += int(r.get("n_units_planned") or 0)
        cov["n_units_failed"] += int(r.get("n_failed") or 0)
        cov["n_parent_gaia_only"] += int(r.get("n_parent_gaia_only") or 0)
        cov["shards"].append({k: r.get(k) for k in ("shard", "n_shards", "n_units_planned", "n_units_done",
                                                    "n_failed", "n_rows_this_run", "stopped_on_budget")})
    if len(df) and "source_id" in df:
        df["source_id"] = df["source_id"].astype(str)
        if "is_control" in df:
            df["is_control"] = _as_bool(df["is_control"])
            df["control_name"] = df.get("control_name", "").fillna("")
            # a control that also fell in its unit: keep one row, marked as the control
            ctrl = df[df["is_control"]].drop_duplicates("source_id")
            rest = df[~df["is_control"] & ~df["source_id"].isin(ctrl["source_id"])].drop_duplicates("source_id")
            df = pd.concat([rest, ctrl], ignore_index=True)
        else:
            df = df.drop_duplicates("source_id")
    return df, cov


SHORTLIST_COLS = ("source_id", "control_name", "is_control", "unit", "ra", "dec", "l", "b", "parallax",
                  "parallax_error", "parallax_over_error", "pmra", "pmra_error", "pmdec", "pmdec_error",
                  "radial_velocity", "radial_velocity_error", "phot_g_mean_mag", "bp_rp", "abs_g", "ruwe",
                  "ipd_frac_multi_peak", "phot_variable_flag", "non_single_star", "teff", "teff_gspphot",
                  "logg_gspphot", "mh_gspphot", "ag_gspphot", "age_flame", "age_flame_lower",
                  "age_flame_upper", "mass_flame", "lum_flame", "evolstage_flame", "flags_flame",
                  "teff_gspspec", "logg_gspspec", "mh_gspspec", "alphafe_gspspec",
                  "alphafe_gspspec_lower", "alphafe_gspspec_upper", "flags_gspspec",
                  "activityindex_espcs", "gaia_rot_period_d", "allwise_designation", "allwise_ra",
                  "allwise_dec", "allwise_sep_arcsec", "allwise_n_neighbours", "allwise_n_mates",
                  "w1mpro", "w1mpro_error", "w2mpro", "w2mpro_error", "w3mpro", "w3mpro_error",
                  "w4mpro", "w4mpro_error", "cc_flags", "ext_flag", "var_flag", "ph_qual",
                  "tmass_designation", "ks_m", "ks_msigcom", "tmass_ph_qual", "tmass_sep_arcsec",
                  "w3_snr", "w4_snr", "ks_w1", "w1_w2", "v_tan_kms",
                  "W1_excess_jy", "W1_excess_err_jy", "chi_W1", "W2_excess_jy", "W2_excess_err_jy",
                  "chi_W2", "W3_excess_jy", "W3_excess_err_jy", "chi_W3", "W4_excess_jy",
                  "W4_excess_err_jy", "chi_W4", "W3_excess_mag", "W4_excess_mag", "excess_significant",
                  "t_bb_k", "t_bb_lo_k", "t_bb_hi_k", "omega_dust_sr", "bb_fit_chi2", "lum_lsun",
                  "lum_source", "mass_msun", "mass_source", "f_ir", "f_ir_lo", "f_ir_hi", "r_bb_au",
                  "fmax_1gyr", "log_f_fmax_1gyr", "log_f_fmax_1gyr_lo", "log_f_fmax_1gyr_hi",
                  "in_t_cell", "above_fmax_3dex", "above_fmax_3dex_16pct")


def stage_screen(conf: dict, out: Path, *, backends: acq.Backends | None = None) -> dict:
    sc = {**DEFAULT_SAMPLE, **(conf.get("sample") or {})}
    ec = {**DEFAULT_EXCESS, **(conf.get("excess") or {})}
    t0 = _time.monotonic()
    raw, cov = _read_parents(out)
    rep: dict = {"stage": "screen", "generated_utc": _now(), "coverage": cov, "n_raw": int(len(raw))}
    if not len(raw):
        rep.update(verdict="NO_DATA_REACHED", reason="no parent rows in any acquire shard",
                   n_parent=0, n_shortlist=0)
        _write(out / "screen.json", rep)
        pd.DataFrame(columns=list(SHORTLIST_COLS)).to_csv(out / "shortlist.csv", index=False)
        print("[cradle] screen: NO_DATA_REACHED (no parent rows)")
        return rep
    parent, counters = select_parent(raw, sc)
    rep["local_cut_counters"] = counters
    rep["n_parent"] = int(len(parent))
    rep["n_controls"] = int(parent["is_control"].sum()) if "is_control" in parent else 0
    d = harmonise(parent)
    d["ks_present"] = pd.to_numeric(d["Ksmag"], errors="coerce").notna()
    rep["n_ks_present"] = int(d["ks_present"].sum())
    ref = locus_reference_mask(d, ec)
    rep["n_locus_reference"] = int(ref.sum())
    loci = fit_loci(d, ec)
    rep["loci"] = loci_summary(loci)
    ex = excess_table(d, loci, ec)
    fit = fit_disk(ex, ec)
    ks_w1_ok = (pd.to_numeric(fit["ks_w1"], errors="coerce") < float(ec["ks_w1_max"])).fillna(False)
    sig = _as_bool(fit["excess_significant"])
    above = _as_bool(fit["above_fmax_3dex"])
    t = pd.to_numeric(fit["t_bb_k"], errors="coerce")
    slo, shi = (float(x) for x in ec["shortlist_t_k"])
    tlo, thi = (float(x) for x in ec["t_cell_k"])
    ctrl = _as_bool(fit["is_control"])
    short = (sig & above & ks_w1_ok & t.between(slo, shi)) | ctrl
    fit["shortlisted"] = short
    rep["funnel"] = {
        "n_parent": int(len(fit)), "n_ks_present": int(d["ks_present"].sum()),
        "n_ks_w1_photospheric": int((ks_w1_ok & d["ks_present"]).sum()),
        "n_excess_significant": int((sig & ks_w1_ok).sum()),
        "n_above_fmax_3dex": int((sig & above & ks_w1_ok).sum()),
        "n_above_fmax_16pct": int((sig & fit["above_fmax_3dex_16pct"].fillna(False) & ks_w1_ok).sum()),
        "n_in_t_cell_photometric": int((sig & above & ks_w1_ok & t.between(tlo, thi)).sum()),
        "n_above_fmax_hot": int((sig & above & ks_w1_ok & (t > thi)).sum()),
        "n_above_fmax_cold": int((sig & above & ks_w1_ok & (t < tlo)).sum()),
        "n_shortlist": int(short.sum()), "n_controls": int(ctrl.sum()),
    }
    tt = t[sig & above & ks_w1_ok]
    if len(tt):
        edges = [0, 100, 150, 200, 250, 300, 350, 400, 500, 700, 1000, 1500, 3000]
        hist, _ = np.histogram(tt.to_numpy(float), bins=edges)
        rep["t_bb_histogram_above_fmax"] = {"edges_k": edges, "counts": [int(x) for x in hist]}
    cols = [c for c in SHORTLIST_COLS if c in fit.columns]
    fit[cols + ["shortlisted"]].to_csv(out / "parent_screened.csv", index=False)
    # Enrichment is the expensive stage (a NEOWISE cone is ~90 s a star) and it
    # runs under a wall-clock budget, so the shortlist is written in PRIORITY
    # order: the controls, then the stars already inside the strict T cell,
    # then everything else by how far above f_max it sits.  Sharding is `i mod
    # n` over this order, so every shard starts on its own best stars and a
    # budget that runs out costs the least interesting ones.
    sl = fit.loc[short, cols].copy()
    if len(sl):
        rank = pd.DataFrame({
            "c": (~ctrl[short]).astype(int).to_numpy(),
            "t": (~t[short].between(tlo, thi).fillna(False)).astype(int).to_numpy(),
            "f": -pd.to_numeric(fit.loc[short, "log_f_fmax_1gyr"], errors="coerce")
            .fillna(-np.inf).to_numpy(),
        }, index=sl.index)
        sl = sl.loc[rank.sort_values(["c", "t", "f"], kind="stable").index]
    sl.to_csv(out / "shortlist.csv", index=False)
    rep["n_shortlist"] = int(short.sum())
    rep["verdict"] = "SCREENED"
    rep["elapsed_s"] = round(_time.monotonic() - t0, 1)
    _write(out / "screen.json", rep)
    print(f"[cradle] screen: parent={len(fit)} sig={rep['funnel']['n_excess_significant']} "
          f"above_fmax={rep['funnel']['n_above_fmax_3dex']} in_cell={rep['funnel']['n_in_t_cell_photometric']} "
          f"shortlist={rep['n_shortlist']} in {rep['elapsed_s']} s")
    return rep


# ---------------------------------------------------------------------------
# ages (enrichment + indicators), sharded over the shortlist
# ---------------------------------------------------------------------------
def _shard_rows(df: pd.DataFrame, shard: int, n_shards: int) -> pd.DataFrame:
    if not len(df):
        return df
    idx = np.arange(len(df))
    return df[(idx % int(n_shards)) == int(shard)].reset_index(drop=True)


def stage_ages(conf: dict, out: Path, *, shard: int = 0, n_shards: int = 1,
               backends: acq.Backends | None = None) -> dict:
    b = _backends(backends)
    en = {**DEFAULT_ENRICH, **(conf.get("enrich") or {})}
    vc = {**DEFAULT_VET, **(conf.get("vet") or {})}
    agc = {**DEFAULT_AGES, **(conf.get("ages") or {})}
    mc = {**DEFAULT_MINERALOGY, **(conf.get("mineralogy") or {})}
    vt = conf.get("vizier_tables") or acq.DEFAULT_VIZIER_TABLES
    t0 = _time.monotonic()
    tag = _tag(shard, n_shards)
    ledger: list = []
    rep: dict = {"stage": "ages", "shard": int(shard), "n_shards": int(n_shards), "generated_utc": _now()}
    p = out / "shortlist.csv"
    short = pd.read_csv(p, low_memory=False) if p.exists() else pd.DataFrame()
    rows = _shard_rows(short, shard, n_shards)
    cap = int(en.get("max_stars_per_shard") or 0)
    if cap:
        rows = rows.iloc[:cap].reset_index(drop=True)
    rep["n_shortlist_total"] = int(len(short))
    rep["n_rows"] = int(len(rows))
    if not len(rows):
        rows.to_csv(out / f"enriched_{tag}.csv", index=False)
        rep.update(status="EMPTY_SHARD", elapsed_s=round(_time.monotonic() - t0, 1))
        _write(out / f"ages_{tag}.json", rep)
        print(f"[cradle] ages {tag}: empty shard")
        return rep
    rows["source_id"] = rows["source_id"].astype(str)
    gaia = b.gaia or acq.esa_query_fn(conf.get("sample") or {}, label="cradle_ages", allow_sync=True)
    irsa = b.irsa or acq.irsa_query_fn()

    # Gaia neighbours: the larger of the W4 beam and 1000 AU
    if en.get("gaia_neighbours", True):
        plx = pd.to_numeric(rows["parallax"], errors="coerce").to_numpy(float)
        rad = np.maximum(float(vc["w4_beam_radius_arcsec"]),
                         float(vc["wide_companion_au"]) * plx / 1000.0)
        nb = acq.fetch_gaia_neighbours(rows, rad, gaia_fn=gaia, upload_fn=b.gaia_upload, ledger=ledger)
        rep["gaia_neighbours"] = {"n_done": nb.attrs.get("n_done"), "n_stars": nb.attrs.get("n_stars"),
                                  "n_rows": int(len(nb))}
        rows = acq.summarise_neighbours(rows, nb, vc)
    # IRSA profile-fit columns
    if en.get("irsa_rows", True) and "allwise_designation" in rows:
        extra = acq.fetch_irsa_allwise_rows(rows["allwise_designation"].tolist(), irsa, ledger=ledger)
        rep["irsa_rows"] = {"n_rows": int(len(extra))}
        if len(extra) and "designation" in extra:
            extra = extra.rename(columns={"designation": "allwise_designation", "ext_flg": "irsa_ext_flg",
                                          "cc_flags": "irsa_cc_flags", "ph_qual": "irsa_ph_qual"})
            extra = extra.drop_duplicates("allwise_designation")
            rows = rows.merge(extra, on="allwise_designation", how="left")
    # SIMBAD
    if en.get("simbad", True):
        fn = b.simbad or acq._real_simbad_type
        types = []
        for _, s in rows.iterrows():
            try:
                types.append(str(fn(float(s["ra"]), float(s["dec"])) or ""))
            except Exception as exc:                       # noqa: BLE001
                ledger.append({"label": f"simbad:{s['source_id']}", "status": acq.STATUS_FAILED,
                               "error": repr(exc)[:200]})
                types.append("")
        rows["simbad_otype"] = types
        rep["simbad"] = {"n_typed": int(sum(1 for t in types if t))}
    # E(B-V)
    if en.get("ebv", True):
        fn = b.ebv or acq._real_ebv
        try:
            e = fn(rows[["source_id", "ra", "dec"]])
            e["source_id"] = e["source_id"].astype(str)
            rows = rows.merge(e[["source_id", "ebv_sfd"]], on="source_id", how="left")
            rep["ebv"] = {"n_finite": int(pd.to_numeric(rows["ebv_sfd"], errors="coerce").notna().sum())}
        except Exception as exc:                           # noqa: BLE001
            ledger.append({"label": "ebv", "status": acq.STATUS_FAILED, "error": repr(exc)[:200]})
            rows["ebv_sfd"] = np.nan
    # NEOWISE
    if en.get("neowise", True):
        recs = []
        tb = float(en.get("neowise_time_budget_s") or 5400.0)
        tn = _time.monotonic()
        for _, s in rows.iterrows():
            if _time.monotonic() - tn > tb:
                recs.append({"neowise_measured": False, "neowise_status": "BUDGET_EXHAUSTED"})
                continue
            recs.append(acq.neowise_descriptor(s.to_dict(), cone_fn=b.neowise))
        nd = pd.DataFrame(recs)
        for c in nd.columns:
            rows[c] = nd[c].to_numpy()
        rep["neowise"] = {"n_measured": int(nd.get("neowise_measured", pd.Series(dtype=bool)).fillna(False).sum()),
                          "n_variable": int(nd.get("neowise_variable", pd.Series(dtype=bool)).fillna(False).sum())}
    # VizieR
    if en.get("vizier", True):
        skip = set()
        pp = out / "probe.json"
        if pp.exists():
            try:
                for _name, r in (json.loads(pp.read_text()).get("vizier") or {}).items():
                    if isinstance(r, dict) and r.get("exists") is False:
                        skip.add(r.get("table"))
            except Exception:                              # noqa: BLE001
                pass
        rows = acq.vizier_matches(rows, vt, float(vt.get("match_radius_arcsec", 5.0)), asu_fn=b.asu,
                                  ledger=ledger, skip_tables=skip)
        rep["vizier"] = {"n_known_disk_matched": int((rows["n_known_disk_matches"] > 0).sum()),
                         "n_prot": int(pd.to_numeric(rows["prot_d"], errors="coerce").notna().sum()),
                         "tables_skipped": sorted(str(x) for x in skip)}
    # Spitzer/IRS + S53
    rows["irs_spectrum_scored"] = False
    rows["irs_verdict"] = ""
    rows["slag_verdict"] = ""
    if en.get("irs", True):
        cat, crec = acq.fetch_irs_catalog(irsa, ledger=ledger)
        rep["irs"] = {"catalog_status": crec.get("status"), "n_catalog": int(len(cat)),
                      "columns": crec.get("columns", [])[:90]}
        rows = acq.match_irs(rows, cat, radius_arcsec=5.0)
        n_spec = 0
        spectra_rec = []
        for i in np.nonzero(pd.to_numeric(rows["irs_reqkey"], errors="coerce").notna().to_numpy())[0]:
            s = rows.iloc[i]
            crow = cat[pd.to_numeric(cat.get("reqkey"), errors="coerce") == float(s["irs_reqkey"])]
            row = crow.iloc[0].to_dict() if len(crow) else {"reqkey": s["irs_reqkey"]}
            sdf, srec = acq.fetch_irs_spectrum(row, http_fn=b.http)
            srec["source_id"] = str(s["source_id"])
            spectra_rec.append(srec)
            if srec.get("status") == "OK":
                w, f, e = spectrum_from_table(sdf)
                sc_ = score_spectrum(w, f, e, mc)
                rows.loc[rows.index[i], "irs_spectrum_scored"] = True
                rows.loc[rows.index[i], "irs_verdict"] = str(sc_.get("verdict"))
                rows.loc[rows.index[i], "irs_feat10_contrast"] = sc_.get("feat10", {}).get("contrast", np.nan)
                rows.loc[rows.index[i], "irs_feat10_snr"] = sc_.get("feat10_snr", np.nan)
                rows.loc[rows.index[i], "irs_crystalline_ratio"] = sc_.get("crystalline_ratio_11p3_over_9p8", np.nan)
                rows.loc[rows.index[i], "irs_silica_contrast"] = sc_.get("silica_contrast", np.nan)
                rows.loc[rows.index[i], "irs_fes_contrast"] = sc_.get("fes_contrast", np.nan)
                rows.loc[rows.index[i], "slag_verdict"] = slag_verdict(sc_, bool(s.get("excess_significant", False)))
                n_spec += 1
            else:
                rows.loc[rows.index[i], "irs_verdict"] = "SPECTRUM_NOT_RETRIEVED"
        rep["irs"].update(n_matched=int(pd.to_numeric(rows["irs_reqkey"], errors="coerce").notna().sum()),
                          n_spectra_scored=n_spec, spectra=spectra_rec[:40])

    rows = assess_ages(rows, agc)
    rows.to_csv(out / f"enriched_{tag}.csv", index=False)
    rep["age_classes"] = {str(k): int(v) for k, v in rows["age_class"].value_counts().items()}
    rep["ledger_failed"] = [x for x in ledger if x.get("status") == acq.STATUS_FAILED][:100]
    rep["n_ledger"] = len(ledger)
    rep["status"] = "OK"
    rep["elapsed_s"] = round(_time.monotonic() - t0, 1)
    _write(out / f"ages_{tag}.json", rep)
    print(f"[cradle] ages {tag}: {len(rows)} stars, classes={rep['age_classes']} in {rep['elapsed_s']} s")
    return rep


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------
def stage_assess(conf: dict, out: Path, *, n_shards_expected: int | None = None) -> dict:
    vc = {**DEFAULT_VET, **(conf.get("vet") or {})}
    ec = {**DEFAULT_EXCESS, **(conf.get("excess") or {})}
    vc = {**vc, "t_cell_k": ec["t_cell_k"], "log_f_fmax_min": ec["log_f_fmax_min"]}
    t0 = _time.monotonic()
    rep: dict = {"stage": "assess", "generated_utc": _now(), "channel": "cradle",
                 "signatures": ["S52", "S53"]}
    sp = out / "screen.json"
    screen = json.loads(sp.read_text()) if sp.exists() else {}
    rep["screen"] = {k: screen.get(k) for k in ("verdict", "reason", "n_raw", "n_parent", "funnel",
                                                 "coverage", "n_shortlist", "t_bb_histogram_above_fmax")}
    files = sorted(glob.glob(str(out / "enriched_s*of*.csv")))
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    frames = [f for f in frames if len(f)]
    enriched = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    n_found = len(files)
    rep["shards"] = {"expected": n_shards_expected, "found": n_found}
    degraded: list[str] = []
    if n_shards_expected and n_found < int(n_shards_expected):
        degraded.append(f"ages_shards_missing:{n_found}/{n_shards_expected}")
    cov = screen.get("coverage") or {}
    if cov.get("n_units_failed"):
        degraded.append(f"acquire_units_failed:{cov['n_units_failed']}/{cov.get('n_units_planned')}")
    if cov.get("n_units_planned") and cov.get("n_units_done", 0) < cov["n_units_planned"]:
        degraded.append(f"acquire_units_incomplete:{cov.get('n_units_done')}/{cov['n_units_planned']}")

    if screen.get("verdict") == "NO_DATA_REACHED" or not screen:
        rep.update(verdict="NO_DATA_REACHED", reason=screen.get("reason") or "no screen.json",
                   n_candidates=0, degraded=degraded, stage_counts={"screened": 0})
        _write(out / "summary.json", rep)
        print(f"[cradle] assess: {rep['verdict']} ({rep['reason']})")
        return rep
    if not len(enriched):
        n_short = int(screen.get("n_shortlist") or 0)
        if n_short > 0 and n_found == 0:
            rep.update(verdict="NO_DATA_REACHED", reason="shortlist exists but no ages shard output reached assess",
                       n_candidates=0, degraded=degraded + ["no_ages_shard_outputs"],
                       stage_counts={"screened": int(screen.get("n_parent") or 0), "shortlisted": n_short})
            _write(out / "summary.json", rep)
            print(f"[cradle] assess: {rep['verdict']} ({rep['reason']})")
            return rep
        enriched = pd.DataFrame()
    if len(enriched):
        vetted, counters = apply_rules(enriched, vc)
        vetted = classify(vetted, vc)
    else:
        vetted, counters = pd.DataFrame(), {"n_in": 0, "kills": {}, "flags": {}, "untested": {}, "n_killed": 0}
    rep["vet_counters"] = counters
    classes = {str(k): int(v) for k, v in vetted["cradle_class"].value_counts().items()} if len(vetted) else {}
    rep["classes"] = classes
    ctrl_mask = pd.Series(as_bool(vetted, "is_control"), index=vetted.index) \
        if len(vetted) else pd.Series(dtype=bool)
    sci = vetted[~ctrl_mask] if len(vetted) else vetted
    cand = sci[sci["cradle_class"] == "CANDIDATE"] if len(sci) else sci
    in_cell = sci[as_bool(sci, "in_cell")] if len(sci) else sci
    rep["n_candidates"] = int(len(cand))
    funnel = dict(screen.get("funnel") or {})
    funnel.update({
        "n_enriched": int(len(sci)),
        "n_in_cell": int(len(in_cell)),
        "n_in_cell_killed": int((in_cell["cradle_class"] == "IN_CELL_KILLED").sum()) if len(in_cell) else 0,
        "n_in_cell_age_undetermined": int((in_cell["cradle_class"] == "IN_CELL_AGE_UNDETERMINED").sum()) if len(in_cell) else 0,
        "n_in_cell_mature": int(len(cand)),
        "n_candidates": int(len(cand)),
    })
    if len(in_cell):
        reasons: dict[str, int] = {}
        for r in in_cell["kill_reasons"].fillna(""):
            for name in str(r).split(";"):
                if name:
                    reasons[name] = reasons.get(name, 0) + 1
        funnel["in_cell_kill_reasons"] = reasons
        funnel["in_cell_age_classes"] = {str(k): int(v) for k, v in in_cell["age_class"].value_counts().items()}
    rep["funnel"] = funnel
    rep["stage_counts"] = {"screened": int(screen.get("n_parent") or 0),
                           "shortlisted": int(screen.get("n_shortlist") or 0),
                           "enriched": int(len(sci)), "in_cell": int(len(in_cell)),
                           "candidates": int(len(cand))}
    # untested vetoes across the in-cell rows
    if len(in_cell):
        ut: dict[str, int] = {}
        for r in in_cell["untested"].fillna(""):
            for name in str(r).split(";"):
                if name:
                    ut[name] = ut.get(name, 0) + 1
        if ut:
            degraded.append("vetoes_untested_in_cell:" + ",".join(f"{k}={v}" for k, v in sorted(ut.items())))
    # candidate table: every in-cell row, candidates first
    keep = [c for c in ("cradle_class", "source_id", "allwise_designation", "ra", "dec", "l", "b", "parallax",
                        "phot_g_mean_mag", "bp_rp", "teff", "ks_m", "w1mpro", "w2mpro", "w3mpro", "w4mpro",
                        "ks_w1", "chi_W3", "chi_W4", "t_bb_k", "t_bb_lo_k", "t_bb_hi_k", "f_ir", "r_bb_au",
                        "log_f_fmax_1gyr", "log_f_fmax_1gyr_lo", "log_f_fmax_adopted", "lum_lsun",
                        "mass_msun", "age_class", "n_old_indicators", "old_indicators",
                        "age_iso_verdict", "age_kin_verdict", "age_alpha_verdict", "age_gyro_verdict",
                        "age_act_verdict", "age_flame", "age_flame_lower", "age_flame_upper",
                        "kin_log10_lr", "kin_n_components", "age_adopted_gyr", "young_group",
                        "young_region", "kill_reasons", "flags", "untested", "n_gaia_beam_neighbours",
                        "n_wide_companions", "w3rchi2", "w4rchi2", "simbad_otype", "ebv_sfd",
                        "neowise_measured", "neowise_variable", "neowise_chi2red_w1", "neowise_chi2red_w2",
                        "n_known_disk_matches", "known_disk_catalogues", "irs_reqkey", "irs_object",
                        "irs_verdict", "slag_verdict", "irs_feat10_contrast", "irs_crystalline_ratio",
                        "control_name") if c in vetted.columns]
    if len(vetted):
        order = {"CANDIDATE": 0, "IN_CELL_AGE_UNDETERMINED": 1, "IN_CELL_KILLED": 2}
        tbl = vetted[keep].copy()
        tbl["_o"] = tbl["cradle_class"].map(lambda s: order.get(str(s).replace("CONTROL:", ""), 3))
        tbl = tbl.sort_values(["_o", "log_f_fmax_1gyr"], ascending=[True, False]).drop(columns="_o")
        tbl.to_csv(out / "shortlist_vetted.csv", index=False)
        tbl[tbl["cradle_class"].isin(["CANDIDATE", "IN_CELL_AGE_UNDETERMINED", "IN_CELL_KILLED"])] \
            .to_csv(out / "candidates.csv", index=False)
    else:
        pd.DataFrame(columns=keep).to_csv(out / "candidates.csv", index=False)
    rep["candidates"] = []
    for _, r in cand.iterrows():
        rep["candidates"].append({k: r.get(k) for k in keep if k in r} | {"not_excluded": not_excluded(r)})
    rep["in_cell_age_undetermined"] = [
        {k: r.get(k) for k in ("source_id", "t_bb_k", "log_f_fmax_1gyr", "old_indicators", "age_class",
                                "kill_reasons", "flags")}
        for _, r in in_cell[in_cell["cradle_class"] == "IN_CELL_AGE_UNDETERMINED"].iterrows()] if len(in_cell) else []
    # controls: where each one lands
    controls = []
    for ctrl in CONTROLS:
        row = vetted[ctrl_mask & (vetted["control_name"].astype(str) == ctrl["name"])] if len(vetted) else pd.DataFrame()
        if len(row):
            r = row.iloc[0]
            controls.append({"name": ctrl["name"], "found": True, "expected": ctrl.get("note"),
                             **{k: r.get(k) for k in ("cradle_class", "source_id", "t_bb_k", "t_bb_lo_k",
                                                      "t_bb_hi_k", "f_ir", "log_f_fmax_1gyr",
                                                      "log_f_fmax_adopted", "chi_W3", "chi_W4", "ks_w1",
                                                      "age_class", "old_indicators", "age_flame",
                                                      "kin_log10_lr", "young_group", "kill_reasons",
                                                      "flags", "neowise_variable", "irs_verdict",
                                                      "slag_verdict") if k in r}})
        else:
            controls.append({"name": ctrl["name"], "found": False, "expected": ctrl.get("note")})
    rep["controls"] = controls
    _write(out / "controls.json", {"generated_utc": _now(), "controls": controls})
    if len(cand):
        verdict = "CRADLE_CANDIDATES"
    else:
        verdict = "NO_CRADLE_CANDIDATE"
    if degraded:
        verdict = f"DEGRADED ({'; '.join(degraded)}); {verdict}"
    rep["degraded"] = degraded
    rep["verdict"] = verdict
    rep["note"] = ("NO_CRADLE_CANDIDATE is a count over the stars actually searched (stage_counts), "
                   "never an occurrence limit; an IN_CELL_AGE_UNDETERMINED star is in the cell "
                   "photometrically and awaits a second independent age indicator.")
    rep["elapsed_s"] = round(_time.monotonic() - t0, 1)
    _write(out / "summary.json", rep)
    print(f"[cradle] assess: {verdict}; classes={classes}")
    return rep


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def cradle_run(stage: str = "all", *, out_dir: Path | str | None = None, shard: int = 0,
               n_shards: int = 1, max_units: int | None = None, conf: dict | None = None,
               config_path=None, backends: acq.Backends | None = None) -> dict:
    conf = conf if conf is not None else load_cradle_config(config_path)
    out = Path(out_dir) if out_dir else Path("results") / "cradle"
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage in ("all", "", None) else tuple(s.strip() for s in stage.split(","))
    rep: dict = {}
    t0 = _time.monotonic()
    for s in stages:
        if s == "probe":
            rep = stage_probe(conf, out, backends=backends)
        elif s == "acquire":
            rep = stage_acquire(conf, out, shard=shard, n_shards=n_shards, backends=backends,
                                max_units=max_units)
        elif s == "screen":
            rep = stage_screen(conf, out, backends=backends)
        elif s == "ages":
            rep = stage_ages(conf, out, shard=shard, n_shards=n_shards, backends=backends)
        elif s == "assess":
            rep = stage_assess(conf, out, n_shards_expected=int(n_shards) if n_shards else None)
        else:
            raise SystemExit(f"unknown stage {s!r}; choose from {STAGES + ('all',)}")
    print(f"[cradle] {stage}: done in {_time.monotonic() - t0:.0f}s")
    return rep


def add_arguments(p) -> None:
    """The channel's flags, so `seti cradle --stage probe` works as itself."""
    p.add_argument("--stage", default="all", help="probe|acquire|screen|ages|assess|all or a comma list")
    p.add_argument("--shard", default="0/1", help="i/n: this shard of n (acquire, ages)")
    p.add_argument("--shards", type=int, default=0,
                   help="number of shards the run is planned for (assess); defaults to the n of --shard")
    p.add_argument("--max-units", type=int, default=0, help="cap on HEALPix units per acquire shard")
    p.add_argument("--max-stars-per-shard", type=int, default=0,
                   help="cap on shortlist stars enriched per ages shard (0 = all)")
    p.add_argument("--out-dir", default="", help="results directory (default results/cradle)")
    p.add_argument("--config", default="", help="alternative config yaml")


def run_from_args(a, _cfg=None) -> int:
    shard, n = parse_shard(a.shard)
    n_shards = a.shards or n
    conf = load_cradle_config(a.config or None)
    if a.max_stars_per_shard:
        conf["enrich"] = {**(conf.get("enrich") or {}),
                          "max_stars_per_shard": int(a.max_stars_per_shard)}
    rep = cradle_run(a.stage, out_dir=a.out_dir or None, shard=shard, n_shards=n_shards,
                     max_units=a.max_units or None, conf=conf)
    v = rep.get("verdict") if isinstance(rep, dict) else None
    if v:
        print(f"[cradle] verdict: {v}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="seti cradle",
                                description="CRADLE (S52/S53): warm debris at the habitable-zone "
                                            "radius of a mature star, above the collisional maximum")
    add_arguments(p)
    return run_from_args(p.parse_args(argv))


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())


__all__ = ["DEFAULTS", "DEFAULT_ENRICH", "DEFAULT_PROBE", "SHORTLIST_COLS", "STAGES",
           "add_arguments", "cradle_run", "load_cradle_config", "main", "parse_shard",
           "run_from_args", "stage_acquire", "stage_ages", "stage_assess", "stage_probe",
           "stage_screen"]
