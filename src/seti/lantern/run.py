"""Stage orchestration for LANTERN.  Writes ``results/lantern/``.

Stages (``python -m seti.lantern.run <stage> ...``, or via :func:`register`):

``probe``      is the archive reachable from here?  Exoplanet Archive row count,
               one MAST ``query_criteria`` per instrument, no downloads.
``inventory``  every JWST time-series observation cross-matched to a transiting
               planet host, with its ``x1dints`` products (sizes, data rights,
               segments) and each planet's ephemeris -> ``inventory.json`` and
               the shard plan.
``screen``     one shard of targets: stream each exposure's ``x1dints`` one
               product at a time (download, read, delete), label the
               integrations by phase, search for narrow features, run the
               eclipse discriminant, and checkpoint one JSON per exposure
               immediately -- a killed shard loses minutes.
``assess``     gather every checkpoint, apply the recurrence-across-targets
               veto and BH-FDR, assign tiers, write ``summary.json``.
``selftest``   the synthetic injection/rejection battery through the same
               analysis path (an offline gate that runs on the runner too).

Verdicts: ``NO_DATA_REACHED`` (nothing analysed), ``NO_VANISHING_LINE``,
``VANISHING_LINE_CANDIDATES_PENDING_VET``, ``DEGRADED_SOURCE`` (data reached but
the eclipse discriminant could not be run on any observation, or most downloads
failed).  A verdict never reads as a science null when nothing was tested.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .line import (
    assess_feature,
    bh_fdr,
    cosmic_ray_driven,
    eclipse_discriminant,
    is_recurrent,
    known_artefact,
    line_flux_series,
    narrow_feature_search,
    recurrent_wavelengths,
    time_average_spectrum,
    transit_consistency,
    vanish_pvalue,
)
from .phase import Ephemeris, ephemeris_from_archive_row, label_integrations

VERDICTS = ("NO_DATA_REACHED", "NO_VANISHING_LINE",
            "VANISHING_LINE_CANDIDATES_PENDING_VET", "DEGRADED_SOURCE")

DEFAULTS: dict = {
    "instruments": {"default": {"R": 1000, "samples_per_resel": 2}},
    "artefacts": {"edge_tolerance_um": 0.01},
    "phase": {}, "line": {}, "discriminant": {},
    "recurrence": {"bin_um": 0.004, "min_targets": 3},
    "fdr": {"alpha": 0.05},
    "acquire": {"instruments": ["NIRSPEC", "NIRISS", "NIRCAM", "MIRI"],
                "match_radius_arcsec": 30.0, "max_file_bytes": 3.0e9,
                "product_batch": 15, "retries": 3, "retry_pause_s": 5.0},
}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _deep_update(base: dict, extra: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_lantern_config(root: Path | None = None) -> dict:
    """``config/lantern.yaml`` over :data:`DEFAULTS`; a missing file degrades."""
    try:
        import yaml
        p = (root or repo_root()) / "config" / "lantern.yaml"
        if not p.exists():
            return _deep_update(DEFAULTS, {})
        return _deep_update(DEFAULTS, yaml.safe_load(p.read_text()) or {})
    except Exception as exc:  # noqa: BLE001
        print(f"[lantern] config/lantern.yaml not loaded ({exc!r}); using defaults")
        return _deep_update(DEFAULTS, {})


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, Ephemeris):
        return _json_safe(o.__dict__)
    return o


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_safe(obj), indent=1, default=str))
    os.replace(tmp, path)


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_") or "unknown"


# --- instrument profile -------------------------------------------------------------------
def instrument_profile(conf: dict, instrument: str | None, grating: str | None,
                       filt: str | None = None, pupil: str | None = None) -> dict:
    """Resolving power, sampling and artefact rows for an exposure's mode."""
    inst = str(instrument or "").upper()
    table = conf.get("instruments", {})
    modes = table.get(inst, {}) if isinstance(table.get(inst), dict) else {}
    mode_key = None
    for cand in (grating, pupil, filt):
        c = str(cand or "").upper()
        if c and c in modes:
            mode_key = c
            break
    if mode_key is None:
        if inst == "NIRISS" and any("GR700" in str(x or "").upper() for x in (pupil, filt, grating)):
            mode_key = "SOSS" if "SOSS" in modes else None
        elif inst == "NIRCAM" and "GRISM" in modes:
            mode_key = "GRISM"
        elif inst == "MIRI" and "LRS" in modes:
            mode_key = "LRS"
        elif inst == "NIRISS" and "SOSS" in modes:
            mode_key = "SOSS"
    prof = dict(modes.get(mode_key) or table.get("default") or {"R": 1000, "samples_per_resel": 2})
    art = conf.get("artefacts", {})
    rows = []
    if isinstance(art.get(inst), dict) and mode_key in art[inst]:
        rows = list(art[inst][mode_key] or [])
    prof.update(instrument=inst, mode=mode_key or "unknown", artefacts=rows,
                edge_tolerance_um=float(art.get("edge_tolerance_um", 0.01)))
    return prof


# --- the per-exposure analysis (pure; the runner and the selftest share it) --------------------
def analyse_stack(stack: dict, ephemerides: list[Ephemeris], conf: dict, target: str,
                  obs_meta: dict | None = None) -> dict:
    """Phase-label, search, and test one exposure against each planet's ephemeris.

    ``stack`` carries ``wavelength``, ``flux`` (n_int, n_wl), ``flux_err`` (or
    None), ``times`` (BJD_TDB), ``time_source`` and ``meta``.  Returns the
    per-exposure record that ``screen`` checkpoints: phase classes per planet,
    the feature list with every statistic and veto, and coverage numbers.
    """
    meta = stack.get("meta") or {}
    prof = instrument_profile(conf, meta.get("INSTRUME") or stack.get("instrument"),
                              meta.get("GRATING") or stack.get("grating"),
                              meta.get("FILTER"), meta.get("PUPIL"))
    pcfg, lcfg, dcfg = conf.get("phase", {}), conf.get("line", {}), conf.get("discriminant", {})
    wl = np.asarray(stack["wavelength"], float)
    flux = np.asarray(stack["flux"], float)
    err = stack.get("flux_err")
    err = np.asarray(err, float) if err is not None else None
    times = np.asarray(stack["times"], float)
    n_int = flux.shape[0]
    extra_sig = 0.0
    if stack.get("time_source") not in ("int_times_bjd_tdb", None):
        extra_sig = float(pcfg.get("header_time_uncertainty_days", 0.006))
    if stack.get("time_source") == "index_only":
        extra_sig = np.inf
    rec = {
        "target": target, "exposure": obs_meta or {}, "instrument": prof["instrument"],
        "mode": prof["mode"], "R": prof.get("R"), "samples_per_resel": prof.get("samples_per_resel"),
        "n_integrations": int(n_int), "n_wavelength": int(wl.size),
        "wavelength_range_um": [float(np.nanmin(wl)), float(np.nanmax(wl))],
        "time_source": stack.get("time_source"),
        "time_span_days": float(np.nanmax(times) - np.nanmin(times)) if np.all(np.isfinite(times)) else None,
        "cadence_days": float(np.nanmedian(np.diff(times))) if n_int > 1 else None,
        "planets": [], "features": [], "n_scanned": 0, "ew_5sigma_limit_um": None,
        "velocity_width_kms": (299792.458 / float(prof["R"])) if prof.get("R") else None,
        "guard_counters": {}, "status": "analysed",
    }
    if n_int < 4:
        rec["status"] = "too_few_integrations"
        return rec
    # Phase labels per planet; keep the planets whose events are in the window.
    labelled = []
    for eph in ephemerides:
        lab = label_integrations(times, eph, pcfg, extra_timing_sigma=extra_sig)
        lab["cadence_days"] = rec["cadence_days"]
        labelled.append((eph, lab))
        rec["planets"].append({"planet": eph.name, "phase_class": lab["phase_class"],
                               "coverage": lab["coverage"], "notes": lab["notes"] + eph.notes,
                               "timing_sigma_days": lab.get("timing_sigma"),
                               "n_transits": len(lab["transits"]),
                               "n_eclipses": len(lab["eclipses"])})
    order = {"both": 0, "eclipse": 1, "transit": 2, "phase_unresolved": 3}
    labelled.sort(key=lambda el: order.get(el[1]["phase_class"], 9))
    if not labelled:
        eph, lab = None, None
        phase_class = "phase_unresolved"
    else:
        eph, lab = labelled[0]
        phase_class = lab["phase_class"]
    rec["phase_class"] = phase_class
    rec["planet"] = eph.name if eph else None
    # Integrations used for the time-averaged spectrum: everything outside
    # eclipse and outside every contact window.
    if lab is not None:
        avg_mask = lab["out_eclipse"] & ~lab["transit_contact"]
        if avg_mask.sum() < 4:
            avg_mask = np.ones(n_int, bool)
    else:
        avg_mask = np.ones(n_int, bool)
    avg = time_average_spectrum(flux, avg_mask, err, float(lcfg.get("clip_sigma", 5.0)))
    scan = narrow_feature_search(wl, avg["spec"], avg["spec_err"],
                                 float(prof.get("samples_per_resel", 2)), lcfg)
    rec["n_scanned"] = int(scan["n_scanned"])
    rec["ew_5sigma_limit_um"] = scan["ew_5sigma_limit"]
    rec["noise_median_norm"] = scan["noise_median"]
    rec["guard_counters"] = dict(scan["counters"])
    rec["n_averaged_integrations"] = int(avg_mask.sum())
    for f in scan["features"]:
        ser = line_flux_series(flux, f["left"], f["right"], err, lcfg)
        disc = tr = None
        if lab is not None and phase_class in ("eclipse", "both"):
            disc = eclipse_discriminant(ser["line"], ser["line_err"], ser["cont"], lab, times, dcfg)
        if lab is not None and phase_class in ("transit", "both"):
            tr = transit_consistency(ser["line"], ser["line_err"], ser["cont"], lab)
        art = known_artefact(f["wavelength"], prof["artefacts"], prof["edge_tolerance_um"])
        cr = cosmic_ray_driven(flux, avg_mask, f["left"], f["right"],
                               float(lcfg.get("sigma_min", 6.0)),
                               int(dcfg.get("cosmic_ray_top_n", 2)),
                               float(prof.get("samples_per_resel", 2)), lcfg, err)
        a = assess_feature(f, disc, tr, phase_class, lab, artefact=art, cosmic=cr, cfg=dcfg)
        entry = {**f, "artefact": art, "cosmic_ray": cr, "eclipse": disc, "transit": tr,
                 "tier_local": a["tier"], "vetoes_local": a["vetoes"],
                 "eclipse_tested": a["eclipse_tested"]}
        if disc is not None:
            entry["p_vanish"] = vanish_pvalue(disc.get("eclipse_vanish_snr", np.nan))
        # Compact window for the record (the spectra themselves are not kept).
        lo, hi = max(0, f["index"] - 12), min(wl.size, f["index"] + 13)
        entry["window"] = {"wavelength": [round(float(x), 5) for x in wl[lo:hi]],
                           "spec_norm": [round(float(x), 6) if np.isfinite(x) else None
                                         for x in avg["spec"][lo:hi]]}
        if disc is not None:
            entry["line_series_binned"] = _bin_series(ser["line"], 25)
            entry["cont_series_binned"] = _bin_series(ser["cont"], 25)
        rec["features"].append(entry)
    return rec


def _bin_series(y, n_bins: int) -> list:
    y = np.asarray(y, float)
    if y.size == 0:
        return []
    edges = np.linspace(0, y.size, min(n_bins, y.size) + 1).astype(int)
    return [round(float(np.nanmean(y[a:b])), 4) if b > a else None
            for a, b in zip(edges[:-1], edges[1:], strict=True)]


# --- probe --------------------------------------------------------------------------------
def probe(out_dir: Path, conf: dict) -> dict:
    from . import acquire
    rec = {"generated_utc": _utc(), "exoplanet_archive": {}, "mast": {}}
    t = time.time()
    planets = acquire.fetch_transiting_planets(conf["acquire"]["retries"],
                                               conf["acquire"]["retry_pause_s"])
    rec["exoplanet_archive"] = {"reached": bool(len(planets)), "n_transiting": int(len(planets)),
                                "seconds": round(time.time() - t, 1)}
    for inst in conf["acquire"]["instruments"]:
        t = time.time()
        obs = acquire.query_jwst_timeseries([inst], conf["acquire"]["retries"],
                                            conf["acquire"]["retry_pause_s"])
        rec["mast"][inst] = {"reached": bool(len(obs)), "n_timeseries_obs": int(len(obs)),
                             "seconds": round(time.time() - t, 1)}
    rec["verdict"] = ("REACHED" if rec["exoplanet_archive"]["reached"]
                      and any(v["reached"] for v in rec["mast"].values()) else "NO_DATA_REACHED")
    _write_json(out_dir / "probe.json", rec)
    print("[lantern] probe:", json.dumps(_json_safe(rec)))
    return rec


# --- inventory -----------------------------------------------------------------------------
def inventory(out_dir: Path, conf: dict, targets: list[str] | None = None,
              instruments: list[str] | None = None, n_shards: int = 8,
              fetch_planets_fn=None, query_tso_fn=None, list_products_fn=None) -> dict:
    """Enumerate observations x planets x products.  Injectable fetchers for tests."""
    from . import acquire
    acq = conf["acquire"]
    fetch_planets_fn = fetch_planets_fn or (lambda: acquire.fetch_transiting_planets(
        acq["retries"], acq["retry_pause_s"]))
    query_tso_fn = query_tso_fn or (lambda insts: acquire.query_jwst_timeseries(
        insts, acq["retries"], acq["retry_pause_s"]))
    list_products_fn = list_products_fn or (lambda obs_df: _list_products_for(obs_df, acq))
    insts = instruments or acq["instruments"]
    inv = {"generated_utc": _utc(), "instruments": insts, "targets": {},
           "funnel": {}, "verdict": "NO_DATA_REACHED"}
    planets = fetch_planets_fn()
    inv["funnel"]["transiting_planets"] = int(len(planets))
    if not len(planets):
        inv["note"] = "exoplanet archive unreachable"
        _write_json(out_dir / "inventory.json", inv)
        _write_json(out_dir / "shards.json", [])
        return inv
    obs = query_tso_fn(insts)
    inv["funnel"]["jwst_timeseries_observations"] = int(len(obs))
    if not len(obs):
        inv["note"] = "MAST unreachable or returned no timeseries observations"
        _write_json(out_dir / "inventory.json", inv)
        _write_json(out_dir / "shards.json", [])
        return inv
    matched = acquire.match_observations(obs, planets, float(acq["match_radius_arcsec"]))
    inv["funnel"]["observations_matched_to_hosts"] = int(len(matched))
    if targets:
        want = {t.strip().lower() for t in targets if t.strip()}
        matched = matched[matched["hostname"].str.lower().isin(want)].reset_index(drop=True)
        inv["funnel"]["observations_after_target_filter"] = int(len(matched))
    if not len(matched):
        inv["note"] = "no JWST timeseries observation matched a transiting-planet host"
        inv["verdict"] = "NO_DATA_REACHED"
        _write_json(out_dir / "inventory.json", inv)
        _write_json(out_dir / "shards.json", [])
        return inv
    prods = list_products_fn(matched)
    inv["funnel"]["x1dints_products"] = int(len(prods))
    by_obs = {}
    if len(prods):
        key = "parent_obsid" if "parent_obsid" in prods.columns else "obsID"
        for k, g in prods.groupby(prods[key].astype(str)):
            by_obs[k] = g
    n_bytes_total, n_public, n_proprietary = 0, 0, 0
    for host, g in matched.groupby("hostname"):
        pl = planets[planets["hostname"] == host]
        eph = [ephemeris_from_archive_row(r, r.get("pl_name")) for r in pl.to_dict("records")]
        entry = {"host": host, "ra": float(pl["ra"].median()), "dec": float(pl["dec"].median()),
                 "planets": [_json_safe(e.__dict__) for e in eph], "observations": [],
                 "total_bytes": 0, "n_products": 0}
        for r in g.to_dict("records"):
            oid = str(r.get("obsid"))
            p = by_obs.get(oid)
            exposures = {}
            if p is not None and len(p):
                for ek, pg in p.groupby("exposure_key"):
                    items = []
                    for pr in pg.to_dict("records"):
                        size = int(pr.get("size") or 0)
                        rights = str(pr.get("dataRights") or r.get("dataRights") or "").upper()
                        items.append({"filename": pr.get("productFilename"),
                                      "uri": pr.get("dataURI"), "size": size,
                                      "calib_level": pr.get("calib_level"),
                                      "dataRights": rights})
                        n_bytes_total += size
                        entry["total_bytes"] += size
                        entry["n_products"] += 1
                        if rights == "PUBLIC" or rights == "":
                            n_public += 1
                        else:
                            n_proprietary += 1
                    exposures[ek] = sorted(items, key=lambda d: str(d["filename"]))
            entry["observations"].append({
                "obsid": oid, "obs_id": r.get("obs_id"), "instrument": r.get("instrument_name"),
                "filters": r.get("filters"), "target_name": r.get("target_name"),
                "t_min": r.get("t_min"), "t_max": r.get("t_max"), "t_exptime": r.get("t_exptime"),
                "calib_level": r.get("calib_level"), "dataRights": r.get("dataRights"),
                "proposal_id": r.get("proposal_id"), "sep_arcsec": r.get("sep_arcsec"),
                "exposures": exposures})
        inv["targets"][host] = entry
    inv["funnel"].update(hosts=len(inv["targets"]), products_public=n_public,
                         products_proprietary_or_unknown=n_proprietary,
                         total_bytes=int(n_bytes_total))
    inv["verdict"] = "INVENTORIED" if inv["targets"] else "NO_DATA_REACHED"
    # Shard plan: round-robin over hosts sorted by bytes (balances wall clock).
    hosts = sorted(inv["targets"], key=lambda h: -inv["targets"][h]["total_bytes"])
    n_shards = max(1, min(int(n_shards), len(hosts)))
    shards = [hosts[i::n_shards] for i in range(n_shards)]
    inv["shards"] = shards
    _write_json(out_dir / "inventory.json", inv)
    _write_json(out_dir / "shards.json", shards)
    print(f"[lantern] inventory: {json.dumps(_json_safe(inv['funnel']))}; {n_shards} shards")
    return inv


def _list_products_for(obs_df: pd.DataFrame, acq: dict) -> pd.DataFrame:
    """Product lists need the astropy table; re-query the matched obsids."""
    from astroquery.mast import Observations

    from . import acquire
    ids = [str(x) for x in obs_df["obsid"].tolist()]
    frames = []
    b = int(acq["product_batch"])
    for s in range(0, len(ids), b):
        chunk = ids[s:s + b]
        try:
            tab = Observations.query_criteria(obsid=chunk)
        except Exception as exc:  # noqa: BLE001
            print(f"[lantern] query_criteria(obsid) batch {s} failed: {exc!r}")
            continue
        if tab is None or len(tab) == 0:
            continue
        p = acquire.list_x1dints(tab, b, acq["retries"], acq["retry_pause_s"])
        if len(p):
            frames.append(p)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --- screen ----------------------------------------------------------------------------------
def screen(out_dir: Path, conf: dict, shard: int = 0, n_shards: int = 1,
           inventory_path: Path | None = None, max_file_bytes: float | None = None,
           max_products_per_target: int = 0, work_dir: Path | None = None,
           download_fn=None, read_fn=None) -> dict:
    """Analyse one shard of targets, checkpointing one JSON per exposure."""
    from . import acquire
    inv_path = inventory_path or (out_dir / "inventory.json")
    if not Path(inv_path).exists():
        rec = {"generated_utc": _utc(), "shard": shard, "verdict": "NO_DATA_REACHED",
               "note": f"no inventory at {inv_path}"}
        _write_json(out_dir / f"screen_shard{shard}.json", rec)
        return rec
    inv = json.loads(Path(inv_path).read_text())
    shards = inv.get("shards") or []
    hosts = shards[shard] if shard < len(shards) else []
    if n_shards != len(shards) and shards:        # explicit re-sharding
        allh = sorted(inv["targets"], key=lambda h: -inv["targets"][h]["total_bytes"])
        hosts = allh[shard::max(1, n_shards)]
    acq = conf["acquire"]
    cap = float(max_file_bytes or acq["max_file_bytes"])
    work = Path(work_dir or (out_dir / "_work"))
    work.mkdir(parents=True, exist_ok=True)
    download_fn = download_fn or (lambda uri, local: acquire.download_x1dints(
        uri, local, acq["retries"], acq["retry_pause_s"]))
    read_fn = read_fn or acquire.read_x1dints
    log = {"generated_utc": _utc(), "shard": shard, "hosts": hosts, "exposures": [],
           "counts": {"analysed": 0, "skipped_checkpoint": 0, "proprietary": 0,
                      "too_large": 0, "download_failed": 0, "read_failed": 0,
                      "no_products": 0}}
    for host in hosts:
        tgt = inv["targets"][host]
        eph = [Ephemeris(**{k: v for k, v in e.items() if k in Ephemeris.__dataclass_fields__})
               for e in tgt["planets"]]
        n_done = 0
        for ob in tgt["observations"]:
            if not ob["exposures"]:
                log["counts"]["no_products"] += 1
                continue
            for ek, items in ob["exposures"].items():
                if max_products_per_target and n_done >= max_products_per_target:
                    break
                ck = out_dir / "obs" / slug(host) / f"{slug(ek)}.json"
                if ck.exists():
                    log["counts"]["skipped_checkpoint"] += 1
                    continue
                status, stacks, notes = "analysed", [], []
                total = sum(int(i.get("size") or 0) for i in items)
                big = [i for i in items if int(i.get("size") or 0) > cap]
                if big:
                    status = "too_large"
                    notes.append(f"{len(big)} product(s) above the {cap / 1e9:.1f} GB cap")
                elif any(str(i.get("dataRights", "")).upper() not in ("PUBLIC", "") for i in items):
                    status = "proprietary"
                else:
                    for it in items:
                        local = work / str(it["filename"])
                        t0 = time.time()
                        st = download_fn(it["uri"], local)
                        if st == "PROPRIETARY":
                            status = "proprietary"
                            break
                        if st != "COMPLETE":
                            status = "download_failed"
                            notes.append(st)
                            break
                        s = read_fn(local)
                        try:
                            local.unlink()
                        except OSError:
                            pass
                        if s is None:
                            status = "read_failed"
                            break
                        stacks.append(s)
                        print(f"[lantern] {host} {it['filename']} {int(it.get('size') or 0) / 1e6:.0f} MB "
                              f"in {time.time() - t0:.0f}s: {s['flux'].shape}")
                rec = {"generated_utc": _utc(), "target": host, "exposure_key": ek,
                       "obsid": ob["obsid"], "obs_id": ob["obs_id"], "instrument_name": ob["instrument"],
                       "filters": ob["filters"], "proposal_id": ob["proposal_id"],
                       "n_products": len(items), "total_bytes": total, "status": status,
                       "notes": notes}
                if status == "analysed":
                    stack = acquire.concatenate_segments(stacks)
                    if stack is None:
                        rec["status"] = "read_failed"
                    else:
                        t0 = time.time()
                        try:
                            rec.update(analyse_stack(stack, eph, conf, host,
                                                     {"exposure_key": ek, "obsid": ob["obsid"],
                                                      "obs_id": ob["obs_id"]}))
                        except Exception as exc:  # noqa: BLE001
                            rec["status"] = "analysis_failed"
                            rec["notes"].append(repr(exc))
                        rec["analysis_seconds"] = round(time.time() - t0, 1)
                        del stack, stacks
                _write_json(ck, rec)
                log["counts"][rec["status"] if rec["status"] in log["counts"] else "read_failed"] = \
                    log["counts"].get(rec["status"], 0) + 1
                log["exposures"].append({"host": host, "exposure_key": ek, "status": rec["status"],
                                         "n_features": len(rec.get("features", []))})
                n_done += 1
                print(f"[lantern] checkpoint {ck.relative_to(out_dir)}: {rec['status']}, "
                      f"{len(rec.get('features', []))} features, phase {rec.get('phase_class')}")
    log["verdict"] = "SCREENED" if log["counts"]["analysed"] else "NO_DATA_REACHED"
    _write_json(out_dir / f"screen_shard{shard}.json", log)
    return log


# --- assess -----------------------------------------------------------------------------------
def assess(out_dir: Path, conf: dict) -> dict:
    """Gather every checkpoint, apply the population-level vetoes, write summary.json."""
    files = sorted(glob.glob(str(out_dir / "obs" / "*" / "*.json")))
    recs = []
    for f in files:
        try:
            recs.append(json.loads(Path(f).read_text()))
        except Exception as exc:  # noqa: BLE001
            print(f"[lantern] unreadable checkpoint {f}: {exc!r}")
    inv_path = out_dir / "inventory.json"
    inv = json.loads(inv_path.read_text()) if inv_path.exists() else {}
    rcfg, fcfg, dcfg = conf["recurrence"], conf["fdr"], conf.get("discriminant", {})
    analysed = [r for r in recs if r.get("status") == "analysed"]
    status_counts = {}
    for r in recs:
        status_counts[r.get("status", "?")] = status_counts.get(r.get("status", "?"), 0) + 1
    # Recurrence across targets over every feature that passed the local guards.
    entries = [{"wavelength": f["wavelength"], "target": r["target"]}
               for r in analysed for f in r.get("features", [])]
    rec_bins = recurrent_wavelengths(entries, float(rcfg["bin_um"]), int(rcfg["min_targets"]))
    rejections = {"known_artefact_wavelength": 0, "recurrent_across_targets": 0,
                  "tracks_continuum": 0, "ramp_correlated": 0,
                  "cosmic_ray_single_integration": 0, "insufficient_phase_coverage": 0,
                  "low_snr": 0, "single_pixel_spike": 0, "drop_not_at_eclipse": 0,
                  "transit_inconsistent": 0, "fdr_not_significant": 0}
    guard_totals = {}
    rows = []
    for r in analysed:
        for k, v in (r.get("guard_counters") or {}).items():
            guard_totals[k] = guard_totals.get(k, 0) + int(v)
        for f in r.get("features", []):
            recurrent = is_recurrent(f["wavelength"], rec_bins, float(rcfg["bin_um"]))
            vetoes = list(f.get("vetoes_local", []))
            if recurrent and "recurrent_across_targets" not in vetoes:
                vetoes.append("recurrent_across_targets")
            tier = f.get("tier_local", "none")
            if recurrent and tier != "none":
                tier = "none"
            rows.append({"target": r["target"], "planet": r.get("planet"),
                         "exposure_key": r.get("exposure_key"), "instrument": r.get("instrument"),
                         "mode": r.get("mode"), "phase_class": r.get("phase_class"),
                         "wavelength_um": f["wavelength"], "snr": f["snr"],
                         "fwhm_samples": f.get("fwhm_samples"), "width_resel": f.get("width_resel"),
                         "equivalent_width_um": f.get("equivalent_width"),
                         "eclipse_vanish_snr": (f.get("eclipse") or {}).get("eclipse_vanish_snr"),
                         "out_positive_snr": (f.get("eclipse") or {}).get("out_positive_snr"),
                         "in_eclipse_sigma": (f.get("eclipse") or {}).get("in_eclipse_sigma"),
                         "ramp_correlation": (f.get("eclipse") or {}).get("ramp_correlation"),
                         "continuum_correlation": (f.get("eclipse") or {}).get("continuum_correlation"),
                         "transit_constancy": (f.get("transit") or {}).get("transit_constancy"),
                         "p_vanish": f.get("p_vanish"), "tier": tier, "vetoes": vetoes})
    # BH-FDR over the eclipse-tested features with the FULL trial count.
    m_total = int(sum(int(r.get("n_scanned") or 0) for r in analysed))
    tested = [i for i, row in enumerate(rows) if row["p_vanish"] is not None
              and row["tier"] in ("interest", "candidate")]
    if tested:
        reject, thresh = bh_fdr([rows[i]["p_vanish"] for i in tested], m_total,
                                float(fcfg["alpha"]))
        for ok, i in zip(reject, tested, strict=True):
            rows[i]["fdr_pass"] = bool(ok)
            if not ok and rows[i]["tier"] == "candidate":
                rows[i]["tier"] = "interest"
                rows[i]["vetoes"].append("fdr_not_significant")
    else:
        thresh = None
    for row in rows:
        for v in row["vetoes"]:
            rejections[v] = rejections.get(v, 0) + 1
    tiers = {t: sum(1 for r in rows if r["tier"] == t) for t in ("none", "watch", "interest", "candidate")}
    n_ecl = sum(1 for r in analysed if r.get("phase_class") in ("eclipse", "both"))
    n_tr = sum(1 for r in analysed if r.get("phase_class") in ("transit", "both"))
    n_unres = sum(1 for r in analysed if r.get("phase_class") == "phase_unresolved")
    n_fail = sum(v for k, v in status_counts.items() if k in ("download_failed", "read_failed",
                                                              "analysis_failed"))
    if not analysed:
        verdict = "NO_DATA_REACHED"
    elif tiers["candidate"] or tiers["interest"]:
        verdict = "VANISHING_LINE_CANDIDATES_PENDING_VET"
    elif n_ecl == 0 or (n_fail > len(analysed)):
        verdict = "DEGRADED_SOURCE"
    else:
        verdict = "NO_VANISHING_LINE"
    per_target = {}
    for r in recs:
        t = per_target.setdefault(r["target"], {"exposures": 0, "analysed": 0, "instruments": set(),
                                                "n_integrations": 0, "phase_classes": {},
                                                "total_bytes": 0, "statuses": {}})
        t["exposures"] += 1
        t["total_bytes"] += int(r.get("total_bytes") or 0)
        t["statuses"][r.get("status")] = t["statuses"].get(r.get("status"), 0) + 1
        if r.get("status") == "analysed":
            t["analysed"] += 1
            t["instruments"].add(f"{r.get('instrument')}/{r.get('mode')}")
            t["n_integrations"] += int(r.get("n_integrations") or 0)
            pc = r.get("phase_class")
            t["phase_classes"][pc] = t["phase_classes"].get(pc, 0) + 1
    for t in per_target.values():
        t["instruments"] = sorted(t["instruments"])
    summary = {
        "generated_utc": _utc(), "channel": "lantern", "verdict": verdict,
        "funnel": {
            **{k: v for k, v in (inv.get("funnel") or {}).items()},
            "exposure_checkpoints": len(recs), "exposures_analysed": len(analysed),
            "exposure_statuses": status_counts,
            "exposures_eclipse_class": n_ecl, "exposures_transit_class": n_tr,
            "exposures_phase_unresolved": n_unres,
            "integrations_analysed": int(sum(int(r.get("n_integrations") or 0) for r in analysed)),
            "resolution_elements_scanned": m_total,
            "narrow_features": len(rows),
            "features_eclipse_tested": sum(1 for r in rows if r["p_vanish"] is not None),
            "tiers": tiers,
        },
        "rejections": rejections, "search_guard_counters": guard_totals,
        "recurrent_wavelength_bins": sorted(float(b * float(rcfg["bin_um"])) for b in rec_bins),
        "fdr": {"alpha": fcfg["alpha"], "m_total": m_total, "threshold": thresh},
        "targets": per_target,
        "candidates": [r for r in rows if r["tier"] in ("interest", "candidate")],
        "watch": [r for r in rows if r["tier"] == "watch"][:200],
        "sensitivity": {r.get("exposure_key"): {"target": r["target"], "instrument": r.get("instrument"),
                                                 "mode": r.get("mode"), "R": r.get("R"),
                                                 "velocity_width_kms": r.get("velocity_width_kms"),
                                                 "ew_5sigma_limit_um": r.get("ew_5sigma_limit_um"),
                                                 "n_integrations": r.get("n_integrations")}
                        for r in analysed},
        "thresholds": {"discriminant": dcfg, "line": conf.get("line", {}),
                       "phase": conf.get("phase", {}), "recurrence": rcfg},
        "verdict_vocabulary": list(VERDICTS),
        "limitations": (
            "A detection-level screen: line fluxes come from a fixed-window sum over a local "
            "continuum with no systematics model; the eclipse test is a two-state comparison "
            "on the archive ephemeris propagated to the epoch (timing uncertainty widens the "
            "contact exclusions). JWST resolution (R<=2700) means any 'narrow' feature is "
            ">=100 km/s wide, so a true laser is unresolved and the sensitivity is the "
            "per-exposure 5-sigma equivalent-width limit quoted here. NO_VANISHING_LINE is a "
            "count at that sensitivity over the exposures actually analysed -- not an "
            "occurrence limit and not a statement about phases or wavelengths not covered."),
    }
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "candidates.json", {"generated_utc": summary["generated_utc"],
                                              "rows": rows})
    print("[lantern] assess:", json.dumps(_json_safe({k: summary[k] for k in
                                                       ("verdict", "funnel", "rejections")})))
    return summary


# --- selftest --------------------------------------------------------------------------------
def selftest(out_dir: Path, conf: dict) -> dict:
    """Synthetic injection/rejection battery through :func:`analyse_stack`."""
    from .synth import synthesise_timeseries
    cases = {
        "planet_line_vanishes": (dict(line_amp=0.02), "candidate"),
        "stellar_line_constant": (dict(line_amp=0.02, line_vanishes=False), "none"),
        "no_line": (dict(line_amp=0.0), None),
        "line_ramp_late_eclipse": (dict(line_amp=0.02, line_vanishes=False, line_ramp_amp=3.0,
                                        ramp_tau=60.0, centre_shift_h=-1.8), "none"),
        "planet_line_transit_only": (dict(line_amp=0.02, centre="transit"), "watch"),
    }
    out = {"generated_utc": _utc(), "cases": {}, "all_as_expected": True}
    for name, (kw, expect) in cases.items():
        s = synthesise_timeseries(**kw)
        s["meta"] = {"INSTRUME": "NIRSPEC", "GRATING": "G395H"}
        s["time_source"] = "int_times_bjd_tdb"
        rec = analyse_stack(s, [s["ephemeris"]], conf, "synthetic")
        near = [f for f in rec["features"] if abs(f["wavelength"] - 4.05) < 0.01]
        got = near[0]["tier_local"] if near else None
        ok = (got == expect) if expect is not None else (got is None)
        out["cases"][name] = {"expected": expect, "got": got, "phase_class": rec["phase_class"],
                              "n_features": len(rec["features"]), "ok": ok,
                              "vetoes": near[0]["vetoes_local"] if near else None}
        out["all_as_expected"] &= ok
    _write_json(out_dir / "selftest.json", out)
    print("[lantern] selftest:", json.dumps(_json_safe(out)))
    return out


# --- entry points --------------------------------------------------------------------------
def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("stage", choices=["probe", "inventory", "screen", "assess", "selftest"])
    p.add_argument("--out-dir", default=None, help="default results/lantern")
    p.add_argument("--targets", default="", help="semicolon-separated host names (inventory)")
    p.add_argument("--instruments", default="", help="comma-separated (inventory/probe)")
    p.add_argument("--n-shards", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--inventory", default=None, help="path to inventory.json (screen)")
    p.add_argument("--max-file-gb", type=float, default=None)
    p.add_argument("--max-products-per-target", type=int, default=0)
    p.add_argument("--work-dir", default=None)


def run_stage(args) -> dict:
    conf = load_lantern_config()
    out_dir = Path(args.out_dir) if args.out_dir else repo_root() / "results" / "lantern"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.instruments:
        conf["acquire"]["instruments"] = [s.strip().upper() for s in args.instruments.split(",")
                                          if s.strip()]
    if args.stage == "probe":
        return probe(out_dir, conf)
    if args.stage == "inventory":
        targets = [t for t in args.targets.split(";") if t.strip()] or None
        return inventory(out_dir, conf, targets=targets, n_shards=args.n_shards)
    if args.stage == "screen":
        return screen(out_dir, conf, shard=args.shard, n_shards=args.n_shards,
                      inventory_path=Path(args.inventory) if args.inventory else None,
                      max_file_bytes=(args.max_file_gb * 1e9) if args.max_file_gb else None,
                      max_products_per_target=args.max_products_per_target,
                      work_dir=Path(args.work_dir) if args.work_dir else None)
    if args.stage == "assess":
        return assess(out_dir, conf)
    return selftest(out_dir, conf)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="seti.lantern", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_args(p)
    args = p.parse_args(argv)
    run_stage(args)
    return 0


def register(sub) -> None:
    """Wire ``lantern <stage>`` into the main CLI's subparsers."""
    p = sub.add_parser("lantern", help="LANTERN: narrow emission line that vanishes at "
                                       "secondary eclipse, across every public JWST "
                                       "exoplanet time series (probe/inventory/screen/assess)")
    _add_args(p)
    p.set_defaults(func=lambda args, cfg=None: run_stage(args))


if __name__ == "__main__":
    raise SystemExit(main())
