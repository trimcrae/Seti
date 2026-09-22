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
    difference_spectrum,
    drift_control_masks,
    eclipse_discriminant,
    feature_snr_in_mask,
    is_recurrent,
    known_artefact,
    line_flux_series,
    narrow_feature_search,
    recurrent_wavelengths,
    residual_z,
    time_average_spectrum,
    transit_consistency,
    vanish_pvalue,
)
from .phase import (
    Ephemeris,
    ephemeris_from_archive_row,
    ingress_duration,
    label_integrations,
    predict_phase_class,
)

VERDICTS = ("NO_DATA_REACHED", "NO_VANISHING_LINE",
            "VANISHING_LINE_CANDIDATES_PENDING_VET", "DEGRADED_SOURCE")

# Bumped whenever the reader or the analysis changes in a way that makes an
# older checkpoint wrong.  `screen` re-analyses any checkpoint carrying a
# different version; `assess` reports them as `stale_checkpoint`.  Version 1
# was the run that read the table-per-segment x1dints layout as one row per
# HDU (docs/lantern.md section 3.1).
# Version 2 searched only the out-of-eclipse time-averaged spectrum, which on
# real x1d products is limited at ~1% of the continuum by the static pixel
# pattern; version 3 searches the out-minus-in difference, where that pattern
# cancels (docs/lantern.md section 3.8).
CHECKPOINT_VERSION = 3
# Time sources that already carry the barycentric correction (no extra timing
# sigma): the per-row TDB-MID column and the INT_TIMES BJD_TDB column.
BARYCENTRIC_TIME_SOURCES = ("int_times_bjd_tdb", "row_bjd_tdb")

DEFAULTS: dict = {
    "instruments": {"default": {"R": 1000, "samples_per_resel": 2}},
    "artefacts": {"edge_tolerance_um": 0.01},
    "phase": {}, "line": {}, "discriminant": {},
    "recurrence": {"bin_um": 0.004, "min_targets": 3},
    "fdr": {"alpha": 0.05},
    "acquire": {"instruments": ["NIRSPEC", "NIRISS", "NIRCAM", "MIRI"],
                "match_radius_arcsec": 30.0, "max_file_bytes": 12.0e9,
                "product_batch": 15, "retries": 3, "retry_pause_s": 5.0,
                # In-memory cap on n_int x n_wl per grid before integrations
                # are co-added (float64 working copies: 40e6 -> ~320 MB each).
                "max_samples_in_memory": 40.0e6,
                # A bin never exceeds this many minutes, nor a quarter of the
                # shortest ingress of the host's planets.
                "max_bin_minutes": 3.0,
                # Planner: prefer the level-3 product (segments + detectors
                # combined) over its level-2 segments when it holds at least
                # this fraction of their bytes.
                "level3_min_byte_fraction": 0.7,
                "plan_cadence_minutes": 1.0},
    "verify": {"cases": [], "min_depth_snr": 5.0, "depth_range": [1e-4, 2e-2],
               "injection_amp": 0.02, "window_fraction_of_period": 0.25},
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
    if stack.get("time_source") not in BARYCENTRIC_TIME_SOURCES and stack.get("time_source") is not None:
        extra_sig = float(pcfg.get("header_time_uncertainty_days", 0.006))
    if stack.get("time_source") == "index_only":
        extra_sig = np.inf
    rec = {
        "target": target, "exposure": obs_meta or {}, "instrument": prof["instrument"],
        "mode": prof["mode"], "R": prof.get("R"), "samples_per_resel": prof.get("samples_per_resel"),
        "n_integrations": int(n_int), "n_wavelength": int(wl.size),
        "n_integrations_raw": int(stack.get("n_int_raw") or n_int),
        "binned_by": int(stack.get("binned_by") or 1),
        "grid": stack.get("grid"),
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
    # Per-feature tests (cosmic-ray drop-out, in-eclipse spectrum SNR) re-average
    # the stack; on a 2e4 x 2048 stack that is seconds per feature and there
    # can be 200 features.  They run on a +-160-sample window around the
    # feature, which holds the 31-sample continuum window, the 64-sample
    # noise blocks and the side windows many times over.
    half_win = 160
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
    clip = float(lcfg.get("clip_sigma", 5.0))
    spr = float(prof.get("samples_per_resel", 2))
    avg = time_average_spectrum(flux, avg_mask, err, clip)
    scan = narrow_feature_search(wl, avg["spec"], avg["spec_err"], spr, lcfg)
    rec["n_scanned"] = int(scan["n_scanned"])
    rec["ew_5sigma_limit_out_um"] = scan["ew_5sigma_limit"]
    rec["ew_5sigma_limit_um"] = scan["ew_5sigma_limit"]
    rec["noise_median_norm"] = scan["noise_median"]
    rec["guard_counters"] = dict(scan["counters"])
    rec["n_averaged_integrations"] = int(avg_mask.sum())
    # --- the difference search -------------------------------------------------
    # The out-of-eclipse spectrum is static-pattern limited (see
    # line.difference_spectrum); the out-minus-in difference is not, and it is
    # where an eclipse-gated line has to show up anyway.  For an eclipse-class
    # exposure the difference is searched too, with the drift null
    # (out-before minus out-after) as the matched control.
    features = [dict(f, found_in="out_spectrum") for f in scan["features"]]
    in_mask = out_diff_mask = ctrl_a = ctrl_b = None
    diff_z = ctrl_z = None
    diff_kind = None
    if lab is not None and phase_class in ("eclipse", "both"):
        # The eclipse difference: the vanishing test itself.
        diff_kind = "difference"
        in_mask, out_diff_mask = lab["in_eclipse"], avg_mask
    elif lab is not None and phase_class == "transit":
        # No eclipse in this visit, so the vanishing test cannot run -- but the
        # out-of-transit minus in-transit difference cancels the same static
        # pattern, so a narrow feature that CHANGED when the planet crossed the
        # star is reachable at the photon limit instead of at ~8% of the
        # continuum.  Such a feature can only ever be 'watch': it is not the
        # eclipse-gated signature and assess_feature keeps it off the candidate
        # ladder through insufficient_phase_coverage.
        diff_kind = "transit_difference"
        in_mask = lab["in_transit"]
        out_diff_mask = lab["out_transit"] & ~lab["eclipse_contact"]
        if in_mask.sum() < 4 or out_diff_mask.sum() < 8:
            diff_kind = None
    if diff_kind is not None:
        dif = difference_spectrum(flux, out_diff_mask, in_mask, err, clip)
        dscan = narrow_feature_search(wl, dif["spec"], dif["spec_err"], spr, lcfg)
        diff_z = residual_z(dif["spec"], dif["spec_err"], spr, lcfg)["z"]
        rec["n_scanned"] = int(rec["n_scanned"] + dscan["n_scanned"])
        rec["ew_5sigma_limit_diff_um"] = dscan["ew_5sigma_limit"]
        rec["noise_median_difference"] = dscan["noise_median"]
        rec["difference_kind"] = diff_kind
        rec["n_in_difference_integrations"] = int(np.count_nonzero(in_mask))
        # The difference is the detection channel whenever it runs, so it sets
        # the quoted sensitivity.
        if dscan["ew_5sigma_limit"] is not None:
            rec["ew_5sigma_limit_um"] = dscan["ew_5sigma_limit"]
        for k, v in (dscan["counters"] or {}).items():
            rec["guard_counters"][f"difference_{k}"] = int(v)
        ctrl_a, ctrl_b = drift_control_masks(out_diff_mask, in_mask)
        if ctrl_a.sum() >= 4 and ctrl_b.sum() >= 4:
            cd = difference_spectrum(flux, ctrl_a, ctrl_b, err, clip)
            ctrl_z = residual_z(cd["spec"], cd["spec_err"], spr, lcfg)["z"]
            rec["drift_control"] = {"n_before": int(ctrl_a.sum()), "n_after": int(ctrl_b.sum())}
        else:
            rec["drift_control"] = {"n_before": int(ctrl_a.sum()), "n_after": int(ctrl_b.sum()),
                                    "note": "too few out-of-event integrations on one side"}
        by_index = {f["index"]: f for f in features}
        for f in dscan["features"]:
            near = next((by_index[j] for j in range(f["index"] - 2, f["index"] + 3)
                         if j in by_index), None)
            if near is not None:
                near["found_in"] = "both"
                near["snr_difference"] = float(f["snr"])
                near["equivalent_width_difference"] = float(f["equivalent_width"])
            else:
                g = dict(f, found_in=diff_kind)
                g["snr_difference"] = float(f["snr"])
                g["equivalent_width_difference"] = float(f["equivalent_width"])
                features.append(g)
                by_index[g["index"]] = g
        features.sort(key=lambda d: -float(d.get("snr_difference") or d["snr"]))
        features = features[: int(lcfg.get("max_features_per_spectrum", 200))]
        for f in features:
            i = int(f["index"])
            if ctrl_z is not None and np.isfinite(ctrl_z[i]):
                f["drift_control_snr"] = float(ctrl_z[i])
            if diff_z is not None and np.isfinite(diff_z[i]):
                f["difference_spectrum_snr"] = float(diff_z[i])
    _DIFF_KINDS = ("difference", "transit_difference", "both")
    rec["n_features_out_spectrum"] = sum(1 for f in features
                                         if f["found_in"] in ("out_spectrum", "both"))
    rec["n_features_difference"] = sum(1 for f in features if f["found_in"] in _DIFF_KINDS)
    for f in features:
        ser = line_flux_series(flux, f["left"], f["right"], err, lcfg)
        lo_w = max(0, f["index"] - half_win)
        hi_w = min(wl.size, f["index"] + half_win + 1)
        fw = flux[:, lo_w:hi_w]
        ew_ = err[:, lo_w:hi_w] if err is not None else None
        disc = tr = None
        if lab is not None and phase_class in ("eclipse", "both"):
            disc = eclipse_discriminant(ser["line"], ser["line_err"], ser["cont"], lab, times, dcfg)
            # 'Consistent with zero' is the residual at the line's own centre
            # sample on the in-eclipse averaged spectrum -- one z value, so a
            # 2-sigma threshold means 2 sigma (a max over +-1 sample demoted a
            # quarter of injected candidates to 'interest' on pure noise).
            disc["in_eclipse_spectrum_snr"] = feature_snr_in_mask(
                fw, lab["in_eclipse"], f["index"] - lo_w, ew_, spr, lcfg, halfwidth=0)
            disc["out_eclipse_spectrum_snr"] = float(f["snr"])
            disc["difference_spectrum_snr"] = f.get("difference_spectrum_snr")
            disc["drift_control_snr"] = f.get("drift_control_snr")
            # The null control matched to HOW the feature was found.
            disc["null_control_snr"] = (disc["drift_control_snr"]
                                        if f.get("found_in") == "difference"
                                        else disc["in_eclipse_spectrum_snr"])
        if lab is not None and phase_class in ("transit", "both"):
            tr = transit_consistency(ser["line"], ser["line_err"], ser["cont"], lab)
        art = known_artefact(f["wavelength"], prof["artefacts"], prof["edge_tolerance_um"])
        diff_found = f.get("found_in") in ("difference", "transit_difference")
        cr_in = in_mask if diff_found else None
        cr_out = out_diff_mask if diff_found else avg_mask
        cr = cosmic_ray_driven(fw, cr_out, f["left"] - lo_w, f["right"] - lo_w,
                               float(lcfg.get("sigma_min", 6.0)),
                               int(dcfg.get("cosmic_ray_top_n", 2)),
                               spr, lcfg, ew_, in_mask=cr_in)
        a = assess_feature(f, disc, tr, phase_class, lab, artefact=art, cosmic=cr, cfg=dcfg)
        entry = {**f, "artefact": art, "cosmic_ray": cr, "eclipse": disc, "transit": tr,
                 "tier_local": a["tier"], "vetoes_local": a["vetoes"],
                 "eclipse_tested": a["eclipse_tested"]}
        if disc is not None:
            entry["p_vanish"] = vanish_pvalue(disc.get("eclipse_vanish_snr", np.nan))
        # A compact window and binned series are kept ONLY for features that
        # matter (a non-'none' tier, or an eclipse test above the interest
        # threshold): a forest-rich M dwarf yields ~100 stellar pseudo-peaks per
        # exposure, and carrying detail for all of them makes the checkpoints
        # hundreds of MB across the archive.
        keep_detail = a["tier"] != "none" or (
            disc is not None and np.isfinite(disc.get("eclipse_vanish_snr", np.nan))
            and disc["eclipse_vanish_snr"] >= float(dcfg.get("vanish_snr_interest", 3.0)))
        if keep_detail:
            lo, hi = max(0, f["index"] - 12), min(wl.size, f["index"] + 13)
            entry["window"] = {"wavelength": [round(float(x), 5) for x in wl[lo:hi]],
                               "spec_norm": [round(float(x), 6) if np.isfinite(x) else None
                                             for x in avg["spec"][lo:hi]]}
            if diff_z is not None:
                entry["window"]["difference_z"] = [
                    round(float(x), 3) if np.isfinite(x) else None for x in diff_z[lo:hi]]
            if ctrl_z is not None:
                entry["window"]["drift_control_z"] = [
                    round(float(x), 3) if np.isfinite(x) else None for x in ctrl_z[lo:hi]]
            if disc is not None:
                entry["line_series_binned"] = _bin_series(ser["line"], 25)
                entry["cont_series_binned"] = _bin_series(ser["cont"], 25)
        rec["features"].append(entry)
    return rec


def _f(v) -> float:
    """None-safe float for max() over JSON-roundtripped values."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return -np.inf
    return x if np.isfinite(x) else -np.inf


def _bin_series(y, n_bins: int) -> list:
    y = np.asarray(y, float)
    if y.size == 0:
        return []
    edges = np.linspace(0, y.size, min(n_bins, y.size) + 1).astype(int)
    return [round(float(np.nanmean(y[a:b])), 4) if b > a else None
            for a, b in zip(edges[:-1], edges[1:], strict=True)]


# --- multi-grid exposure analysis ---------------------------------------------------------
def _bin_factor(stack: dict, ephemerides: list[Ephemeris], acq: dict) -> int:
    """Co-add factor that keeps ``n_int x n_wl`` under the memory cap without a
    bin longer than ``max_bin_minutes`` or a quarter of the shortest ingress."""
    f = np.asarray(stack["flux"])
    n_int, n_wl = f.shape
    cap = float(acq.get("max_samples_in_memory", 40.0e6))
    if n_int * n_wl <= cap or n_int < 8:
        return 1
    want = int(np.ceil(n_int * n_wl / cap))
    t = np.asarray(stack["times"], float)
    cad = float(np.nanmedian(np.diff(t))) if n_int > 1 else np.nan
    if not np.isfinite(cad) or cad <= 0:
        return 1
    limit_days = float(acq.get("max_bin_minutes", 3.0)) / 1440.0
    for eph in ephemerides:
        if eph.valid():
            limit_days = min(limit_days, 0.25 * ingress_duration(eph))
    allowed = int(max(1, np.floor(limit_days / cad)))
    return int(max(1, min(want, allowed, n_int // 8)))


def analyse_exposure(grids: list[dict], ephemerides: list[Ephemeris], conf: dict,
                     target: str, obs_meta: dict | None = None) -> dict:
    """Run :func:`analyse_stack` on every wavelength grid of one exposure
    (NIRISS orders, NIRSpec detectors) and merge into one record: the largest
    grid is primary; features carry their ``grid`` index; ``n_scanned`` sums.
    Long stacks are co-added first (:func:`_bin_factor`, recorded)."""
    from .acquire import bin_integrations
    acq = conf.get("acquire", {})
    recs = []
    for gi, g in enumerate(grids):
        k = _bin_factor(g, ephemerides, acq)
        stack = bin_integrations(g, k) if k > 1 else g
        r = analyse_stack(stack, ephemerides, conf, target, obs_meta)
        r["grid_index"] = gi
        for f in r["features"]:
            f["grid"] = gi
        recs.append(r)
        del stack
    if not recs:
        return {"status": "read_failed", "features": [], "grids": []}
    analysed = [r for r in recs if r["status"] == "analysed"]
    primary = analysed[0] if analysed else recs[0]
    rec = dict(primary)
    rec["features"] = [f for r in analysed for f in r["features"]]
    rec["n_scanned"] = int(sum(int(r.get("n_scanned") or 0) for r in analysed))
    rec["grids"] = [{"grid_index": r["grid_index"], "status": r["status"],
                     "grid": r.get("grid"), "n_integrations": r.get("n_integrations"),
                     "n_integrations_raw": r.get("n_integrations_raw"),
                     "binned_by": r.get("binned_by"), "n_wavelength": r.get("n_wavelength"),
                     "wavelength_range_um": r.get("wavelength_range_um"),
                     "phase_class": r.get("phase_class"), "n_scanned": r.get("n_scanned"),
                     "ew_5sigma_limit_um": r.get("ew_5sigma_limit_um"),
                     "n_features": len(r.get("features", [])),
                     "time_source": r.get("time_source")} for r in recs]
    rec["n_grids"] = len(recs)
    return rec


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
_L3_RE = re.compile(r"^jw(\d{5})-o(\d{3})", re.I)
_L2_RE = re.compile(r"^jw(\d{5})(\d{3})(\d{3})_", re.I)


def _program_obs(exposure_key: str) -> tuple[str, str] | None:
    m = _L3_RE.match(str(exposure_key))
    if m:
        return m.group(1), m.group(2)
    m = _L2_RE.match(str(exposure_key))
    if m:
        return m.group(1), m.group(2)
    return None


def _ephemerides_of(entry: dict) -> list[Ephemeris]:
    """Ephemerides from an inventory entry (JSON nulls become NaN / 0)."""
    out = []
    for e in entry.get("planets", []):
        d = {k: v for k, v in e.items() if k in Ephemeris.__dataclass_fields__}
        for k in ("period", "t0", "duration"):
            if d.get(k) is None:
                d[k] = np.nan
        for k in ("period_err", "t0_err"):
            if d.get(k) is None:
                d[k] = 0.0
        d["notes"] = list(d.get("notes") or [])
        out.append(Ephemeris(**d))
    return out


def plan_units(inv: dict, conf: dict, n_shards: int = 8) -> dict:
    """Turn ``inv['targets']`` into the flat, prioritised list of work units.

    One unit = one exposure key of one host, with its FITS products merged
    across every MAST observation row that listed them.  Rules:

    * ``.png`` previews are never units;
    * a level-3 product (segments and detectors combined by ``calwebb_tso3``)
      supersedes the level-2 segments of the same program+observation when it
      holds >= ``level3_min_byte_fraction`` of their bytes -- otherwise the
      segments are used and the level-3 file skipped (never both: that would
      count one detector twice);
    * a unit with any non-public product is recorded, never scheduled;
    * each scheduled unit gets a predicted phase class from its MAST window
      (:func:`predict_phase_class`) and a rank: eclipse-class 0, transit 1,
      unresolved 2.  Units are sorted by (rank, bytes) and dealt round-robin
      into ``n_shards`` shards, so every shard works eclipses first and, within
      a rank, the cheapest exposures first.
    """
    acq = conf.get("acquire", {})
    pcfg = conf.get("phase", {})
    frac_min = float(acq.get("level3_min_byte_fraction", 0.7))
    cadence = float(acq.get("plan_cadence_minutes", 1.0)) / 1440.0
    units: list[dict] = []
    for host, entry in inv.get("targets", {}).items():
        eph = _ephemerides_of(entry)
        per_key: dict[str, dict] = {}
        for ob in entry.get("observations", []):
            for ek, items in (ob.get("exposures") or {}).items():
                fits_items = [i for i in items if str(i.get("filename", "")).lower().endswith(".fits")]
                if not fits_items:
                    continue
                u = per_key.setdefault(ek, {
                    "host": host, "exposure_key": ek, "obsid": ob.get("obsid"),
                    "obs_id": ob.get("obs_id"), "instrument": ob.get("instrument"),
                    "filters": ob.get("filters"), "proposal_id": ob.get("proposal_id"),
                    "t_min": None, "t_max": None, "items": {}, "obs_rows": []})
                for i in fits_items:
                    u["items"][str(i["filename"])] = i
                u["obs_rows"].append(ob.get("obs_id"))
                for k, fn in (("t_min", min), ("t_max", max)):
                    v = ob.get(k)
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(v):
                        u[k] = v if u[k] is None else fn(u[k], v)
                # Prefer the level-3 observation row's identity (its window is
                # the whole visit; a level-2 row covers one segment).
                if ob.get("calib_level") == 3:
                    u["obsid"], u["obs_id"] = ob.get("obsid"), ob.get("obs_id")
                    u["instrument"], u["filters"] = ob.get("instrument"), ob.get("filters")
                    u["proposal_id"] = ob.get("proposal_id")
        for u in per_key.values():
            items = sorted(u["items"].values(), key=lambda d: str(d["filename"]))
            u["items"] = [{"filename": i["filename"], "uri": i["uri"], "size": int(i.get("size") or 0),
                           "calib_level": i.get("calib_level"), "dataRights": i.get("dataRights")}
                          for i in items]
            u["total_bytes"] = int(sum(i["size"] for i in u["items"]))
            u["n_products"] = len(u["items"])
            levels = [int(i["calib_level"]) for i in u["items"] if i.get("calib_level") is not None]
            u["level"] = max(levels) if levels else None
            u["public"] = all(str(i.get("dataRights") or "").upper() in ("PUBLIC", "")
                              for i in u["items"])
            u["program_obs"] = _program_obs(u["exposure_key"])
            del u["obs_rows"]
        # Level-3 vs level-2 within each program+observation.
        groups: dict[tuple, list[dict]] = {}
        for u in per_key.values():
            groups.setdefault(u["program_obs"] or ("?", u["exposure_key"]), []).append(u)
        for g in groups.values():
            l3 = [u for u in g if u["level"] == 3]
            l2 = [u for u in g if u["level"] != 3]
            if l3 and l2:
                b3 = sum(u["total_bytes"] for u in l3)
                b2 = sum(u["total_bytes"] for u in l2)
                if b3 >= frac_min * b2 or not all(u["public"] for u in l2):
                    for u in l2:
                        u["superseded"] = "level3_product_holds_these_segments"
                else:
                    for u in l3:
                        u["superseded"] = f"level3_bytes_{b3 / max(b2, 1):.2f}_of_segments_use_level2"
        for u in per_key.values():
            u["scheduled"] = bool(u["public"] and not u.get("superseded"))
            u["skip_reason"] = (None if u["scheduled"] else
                                ("proprietary" if not u["public"] else u["superseded"]))
            jd0 = (u["t_min"] + 2400000.5) if u["t_min"] is not None else None
            jd1 = (u["t_max"] + 2400000.5) if u["t_max"] is not None else None
            u["predicted"] = predict_phase_class(eph, jd0, jd1, pcfg, cadence_days=cadence)
            u["rank"] = int(u["predicted"]["rank"])
            units.append(u)
    # Eclipse-class first, then transit, then unresolved; WITHIN a rank the
    # cheapest exposure first.  Every exposure is an independent target/epoch,
    # so cost-ascending order maximises the number of independent eclipse tests
    # a bounded dispatch reaches (the 10 GB phase curves are reached by the
    # dispatches that follow, since checkpoints accumulate).
    units.sort(key=lambda u: (0 if u["scheduled"] else 1, u["rank"], u["total_bytes"],
                              u["host"], u["exposure_key"]))
    for i, u in enumerate(units):
        u["unit"] = i
    sched = [u["unit"] for u in units if u["scheduled"]]
    n_shards = max(1, min(int(n_shards), max(1, len(sched))))
    shards = [sched[i::n_shards] for i in range(n_shards)]
    by_class: dict[str, dict] = {}
    for u in units:
        key = u["predicted"]["phase_class"] if u["scheduled"] else f"skipped_{u['skip_reason']}"
        d = by_class.setdefault(key, {"units": 0, "bytes": 0})
        d["units"] += 1
        d["bytes"] += u["total_bytes"]
    inv["units"] = units
    inv["shards"] = shards
    inv["plan"] = {"n_units": len(units), "n_scheduled": len(sched),
                   "scheduled_bytes": int(sum(u["total_bytes"] for u in units if u["scheduled"])),
                   "by_class": by_class, "n_shards": n_shards,
                   "checkpoint_version": CHECKPOINT_VERSION}
    return inv


def inventory(out_dir: Path, conf: dict, targets: list[str] | None = None,
              instruments: list[str] | None = None, n_shards: int = 8,
              fetch_planets_fn=None, query_tso_fn=None, list_products_fn=None,
              replan_from: Path | None = None) -> dict:
    """Enumerate observations x planets x products, then plan the work.

    ``replan_from`` re-plans an existing ``inventory.json`` offline (no archive
    calls): the committed inventory is re-used and only the unit list, the
    predicted phase classes and the shard plan are rebuilt.
    """
    from . import acquire
    if replan_from is not None:
        inv = json.loads(Path(replan_from).read_text())
        inv["replanned_utc"] = _utc()
        inv["replanned_from"] = str(replan_from)
        if targets:
            want = {t.strip().lower() for t in targets if t.strip()}
            inv["targets"] = {h: e for h, e in inv["targets"].items() if h.lower() in want}
        plan_units(inv, conf, n_shards)
        _write_plan(out_dir, inv)
        print(f"[lantern] inventory (replan): {json.dumps(_json_safe(inv['plan']))}")
        return inv
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
                        if not str(pr.get("productFilename", "")).lower().endswith(".fits"):
                            continue
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
                    if items:
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
    plan_units(inv, conf, n_shards)
    _write_plan(out_dir, inv)
    print(f"[lantern] inventory: {json.dumps(_json_safe(inv['funnel']))}; "
          f"plan {json.dumps(_json_safe(inv['plan']))}")
    return inv


def _write_plan(out_dir: Path, inv: dict) -> None:
    """``inventory.json`` (targets, funnel, plan summary, shards) and
    ``plan.json`` (the unit list the shards work from) -- kept apart so the
    committed inventory stays small."""
    _write_json(out_dir / "inventory.json", {k: v for k, v in inv.items() if k != "units"})
    _write_json(out_dir / "plan.json", {"generated_utc": _utc(), "plan": inv.get("plan"),
                                        "shards": inv.get("shards"), "units": inv.get("units", [])})
    _write_json(out_dir / "shards.json", inv.get("shards", []))


def _load_units(inv_path: Path, inv: dict, conf: dict, n_shards: int) -> list[dict]:
    plan_path = Path(inv_path).parent / "plan.json"
    if "units" in inv:
        return inv["units"]
    if plan_path.exists():
        return json.loads(plan_path.read_text()).get("units", [])
    plan_units(inv, conf, n_shards)
    return inv["units"]


def _list_products_for(obs_df: pd.DataFrame, acq: dict) -> pd.DataFrame:
    """``x1dints`` products for the matched observations, by obsid, in batches
    (``get_product_list`` accepts obsid strings directly)."""
    from . import acquire
    ids = [str(x) for x in obs_df["obsid"].tolist()]
    return acquire.list_x1dints(ids, int(acq["product_batch"]), acq["retries"],
                                acq["retry_pause_s"])


# --- screen ----------------------------------------------------------------------------------
def _checkpoint_current(path: Path) -> bool:
    try:
        return int(json.loads(path.read_text()).get("checkpoint_version", 0)) == CHECKPOINT_VERSION
    except Exception:  # noqa: BLE001
        return False


def _as_grids(read_result) -> list[dict]:
    if read_result is None:
        return []
    if isinstance(read_result, dict):
        return [read_result]
    return [g for g in read_result if g is not None]


def screen(out_dir: Path, conf: dict, shard: int = 0, n_shards: int = 1,
           inventory_path: Path | None = None, max_file_bytes: float | None = None,
           max_products_per_target: int = 0, work_dir: Path | None = None,
           download_fn=None, read_fn=None, deadline_minutes: float = 0.0,
           unit_filter=None) -> dict:
    """Analyse one shard of work units, checkpointing one JSON per exposure.

    Units come from ``inventory['shards'][shard]`` (or a re-slice of the
    scheduled units when ``n_shards`` differs from the plan).  An existing
    checkpoint at the current :data:`CHECKPOINT_VERSION` is skipped, so
    successive dispatches accumulate; a stale one is re-analysed.  No new unit
    is started after ``deadline_minutes`` (0 = no deadline).
    """
    from . import acquire
    inv_path = inventory_path or (out_dir / "inventory.json")
    if not Path(inv_path).exists():
        rec = {"generated_utc": _utc(), "shard": shard, "verdict": "NO_DATA_REACHED",
               "note": f"no inventory at {inv_path}"}
        _write_json(out_dir / f"screen_shard{shard}.json", rec)
        return rec
    inv = json.loads(Path(inv_path).read_text())
    units = _load_units(inv_path, inv, conf, n_shards)
    shards = inv.get("shards") or []
    if shards and n_shards == len(shards):
        my = shards[shard] if shard < len(shards) else []
    else:
        sched = [u["unit"] for u in units if u.get("scheduled")]
        my = sched[shard::max(1, n_shards)]
    if unit_filter is not None:
        my = [i for i in my if unit_filter(units[i])]
    acq = conf["acquire"]
    cap = float(max_file_bytes or acq["max_file_bytes"])
    work = Path(work_dir or (out_dir / "_work"))
    work.mkdir(parents=True, exist_ok=True)
    download_fn = download_fn or (lambda uri, local: acquire.download_x1dints(
        uri, local, acq["retries"], acq["retry_pause_s"]))
    read_fn = read_fn or acquire.read_x1dints_grids
    t_start = time.time()
    deadline = t_start + 60.0 * float(deadline_minutes) if deadline_minutes else None
    log = {"generated_utc": _utc(), "shard": shard, "n_units": len(my), "exposures": [],
           "counts": {"analysed": 0, "skipped_checkpoint": 0, "stale_checkpoint_redone": 0,
                      "proprietary": 0, "too_large": 0, "download_failed": 0,
                      "read_failed": 0, "too_few_integrations": 0, "analysis_failed": 0,
                      "deadline_deferred": 0},
           "bytes_downloaded": 0}
    per_target: dict[str, int] = {}
    for ui in my:
        u = units[ui]
        host = u["host"]
        tgt = inv["targets"][host]
        eph = _ephemerides_of(tgt)
        ek = u["exposure_key"]
        if max_products_per_target and per_target.get(host, 0) >= max_products_per_target:
            continue
        ck = out_dir / "obs" / slug(host) / f"{slug(ek)}.json"
        if ck.exists():
            if _checkpoint_current(ck):
                log["counts"]["skipped_checkpoint"] += 1
                continue
            log["counts"]["stale_checkpoint_redone"] += 1
        if deadline is not None and time.time() > deadline:
            log["counts"]["deadline_deferred"] += 1
            continue
        items = u["items"]
        status, stacks, notes = "analysed", [], []
        total = int(u.get("total_bytes") or 0)
        big = [i for i in items if int(i.get("size") or 0) > cap]
        if big:
            status = "too_large"
            notes.append(f"{len(big)} product(s) above the {cap / 1e9:.1f} GB cap")
        elif not u.get("public", True):
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
                log["bytes_downloaded"] += int(it.get("size") or 0)
                t1 = time.time()
                grids = _as_grids(read_fn(local))
                try:
                    local.unlink()
                except OSError:
                    pass
                if not grids:
                    status = "read_failed"
                    notes.append(f"no spectral table recognised in {it['filename']}")
                    break
                stacks.extend(grids)
                print(f"[lantern] {host} {it['filename']} {int(it.get('size') or 0) / 1e6:.0f} MB "
                      f"dl {t1 - t0:.0f}s read {time.time() - t1:.0f}s: "
                      + ", ".join(f"{g['flux'].shape}{'/o' + str(g['grid'].get('sporder')) if g.get('grid') else ''}"
                                  for g in grids))
        rec = {"generated_utc": _utc(), "checkpoint_version": CHECKPOINT_VERSION,
               "target": host, "exposure_key": ek, "unit": ui,
               "obsid": u.get("obsid"), "obs_id": u.get("obs_id"), "instrument_name": u.get("instrument"),
               "filters": u.get("filters"), "proposal_id": u.get("proposal_id"),
               "level": u.get("level"), "predicted": u.get("predicted"),
               "n_products": len(items), "total_bytes": total, "status": status, "notes": notes}
        if status == "analysed":
            grids = acquire.group_segments_by_grid(stacks)
            del stacks
            if not grids:
                rec["status"] = "read_failed"
                rec["notes"].append("segments could not be joined on any grid")
            else:
                t0 = time.time()
                try:
                    out = analyse_exposure(grids, eph, conf, host,
                                           {"exposure_key": ek, "obsid": u.get("obsid"),
                                            "obs_id": u.get("obs_id")})
                    out.pop("notes", None)
                    rec.update(out)
                except Exception as exc:  # noqa: BLE001
                    rec["status"] = "analysis_failed"
                    rec["notes"].append(repr(exc))
                rec["analysis_seconds"] = round(time.time() - t0, 1)
                del grids
        _write_json(ck, rec)
        key = rec["status"] if rec["status"] in log["counts"] else "read_failed"
        log["counts"][key] += 1
        log["exposures"].append({"host": host, "exposure_key": ek, "status": rec["status"],
                                 "phase_class": rec.get("phase_class"),
                                 "n_features": len(rec.get("features", []))})
        per_target[host] = per_target.get(host, 0) + 1
        print(f"[lantern] checkpoint {ck.relative_to(out_dir)}: {rec['status']}, "
              f"{len(rec.get('features', []))} features, phase {rec.get('phase_class')} "
              f"(predicted {(u.get('predicted') or {}).get('phase_class')}), "
              f"{rec.get('n_integrations')} int, elapsed {(time.time() - t_start) / 60:.0f} min")
    log["verdict"] = "SCREENED" if log["counts"]["analysed"] else "NO_DATA_REACHED"
    log["elapsed_minutes"] = round((time.time() - t_start) / 60.0, 1)
    _write_json(out_dir / f"screen_shard{shard}.json", log)
    return log


# --- assess -----------------------------------------------------------------------------------
def assess(out_dir: Path, conf: dict) -> dict:
    """Gather every checkpoint, apply the population-level vetoes, write summary.json."""
    files = sorted(glob.glob(str(out_dir / "obs" / "*" / "*.json")))
    recs, stale = [], 0
    for f in files:
        try:
            r = json.loads(Path(f).read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"[lantern] unreadable checkpoint {f}: {exc!r}")
            continue
        if int(r.get("checkpoint_version", 0)) != CHECKPOINT_VERSION:
            stale += 1
            continue
        recs.append(r)
    inv_path = out_dir / "inventory.json"
    inv = json.loads(inv_path.read_text()) if inv_path.exists() else {}
    rcfg, fcfg, dcfg = conf["recurrence"], conf["fdr"], conf.get("discriminant", {})
    analysed = [r for r in recs if r.get("status") == "analysed"]
    status_counts = {}
    for r in recs:
        status_counts[r.get("status", "?")] = status_counts.get(r.get("status", "?"), 0) + 1
    if stale:
        status_counts["stale_checkpoint"] = stale
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
                         "mode": r.get("mode"), "grid": f.get("grid"),
                         "phase_class": r.get("phase_class"),
                         "wavelength_um": f["wavelength"], "snr": f["snr"],
                         "found_in": f.get("found_in"),
                         "snr_difference": f.get("snr_difference"),
                         "drift_control_snr": (f.get("eclipse") or {}).get("drift_control_snr"),
                         "fwhm_samples": f.get("fwhm_samples"), "width_resel": f.get("width_resel"),
                         "equivalent_width_um": f.get("equivalent_width"),
                         "eclipse_vanish_snr": (f.get("eclipse") or {}).get("eclipse_vanish_snr"),
                         "out_positive_snr": (f.get("eclipse") or {}).get("out_positive_snr"),
                         "in_eclipse_sigma": (f.get("eclipse") or {}).get("in_eclipse_sigma"),
                         "in_eclipse_spectrum_snr": (f.get("eclipse") or {}).get("in_eclipse_spectrum_snr"),
                         "line_fractional_drop": (f.get("eclipse") or {}).get("line_fractional_drop"),
                         "continuum_fractional_drop": (f.get("eclipse") or {}).get("continuum_fractional_drop"),
                         "ramp_correlation": (f.get("eclipse") or {}).get("ramp_correlation"),
                         "continuum_correlation": (f.get("eclipse") or {}).get("continuum_correlation"),
                         "free_step_offset_days": (f.get("eclipse") or {}).get("free_step_offset_days"),
                         "transit_constancy": (f.get("transit") or {}).get("transit_constancy"),
                         "transit_excess_sigma": (f.get("transit") or {}).get("transit_excess_sigma"),
                         "p_vanish": f.get("p_vanish"), "tier": tier, "vetoes": vetoes,
                         "eclipse_tested": bool(f.get("eclipse_tested"))})
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
            t["n_integrations"] += int(r.get("n_integrations_raw") or r.get("n_integrations") or 0)
            pc = r.get("phase_class")
            t["phase_classes"][pc] = t["phase_classes"].get(pc, 0) + 1
    for t in per_target.values():
        t["instruments"] = sorted(t["instruments"])
    # Coverage by instrument mode and by phase class, and predicted-vs-found.
    by_mode: dict[str, dict] = {}
    pred_vs: dict[str, int] = {}
    for r in analysed:
        m = f"{r.get('instrument')}/{r.get('mode')}"
        d = by_mode.setdefault(m, {"exposures": 0, "eclipse_class": 0, "transit_class": 0,
                                   "phase_unresolved": 0, "integrations": 0,
                                   "resolution_elements": 0, "features": 0,
                                   "features_difference": 0, "eclipse_tested": 0,
                                   "ew_5sigma_limit_um_median": [],
                                   "ew_5sigma_limit_out_um_median": []})
        d["exposures"] += 1
        pc = r.get("phase_class")
        if pc in ("eclipse", "both"):
            d["eclipse_class"] += 1
        if pc in ("transit", "both"):
            d["transit_class"] += 1
        if pc == "phase_unresolved":
            d["phase_unresolved"] += 1
        d["integrations"] += int(r.get("n_integrations_raw") or r.get("n_integrations") or 0)
        d["resolution_elements"] += int(r.get("n_scanned") or 0)
        d["features"] += len(r.get("features", []))
        d["features_difference"] += sum(1 for f in r.get("features", [])
                                        if f.get("found_in") in ("difference", "both"))
        d["eclipse_tested"] += sum(1 for f in r.get("features", []) if f.get("eclipse_tested"))
        if r.get("ew_5sigma_limit_um") is not None:
            d["ew_5sigma_limit_um_median"].append(float(r["ew_5sigma_limit_um"]))
        if r.get("ew_5sigma_limit_out_um") is not None:
            d["ew_5sigma_limit_out_um_median"].append(float(r["ew_5sigma_limit_out_um"]))
        pk = f"{(r.get('predicted') or {}).get('phase_class')}->{pc}"
        pred_vs[pk] = pred_vs.get(pk, 0) + 1
    for d in by_mode.values():
        for k in ("ew_5sigma_limit_um_median", "ew_5sigma_limit_out_um_median"):
            v = d.pop(k)
            d[k] = float(np.median(v)) if v else None
    verify_path = out_dir / "verify.json"
    verify = None
    if verify_path.exists():
        try:
            v = json.loads(verify_path.read_text())
            verify = {"verdict": v.get("verdict"), "generated_utc": v.get("generated_utc"),
                      "cases": {k: {"passed": c.get("passed"), "phase_class": c.get("phase_class"),
                                    "depth": c.get("depth"), "depth_snr": c.get("depth_snr")}
                                for k, c in (v.get("cases") or {}).items()}}
        except Exception:  # noqa: BLE001
            verify = None
    plan = inv.get("plan") or {}
    summary = {
        "generated_utc": _utc(), "channel": "lantern", "verdict": verdict,
        "checkpoint_version": CHECKPOINT_VERSION,
        "funnel": {
            **{k: v for k, v in (inv.get("funnel") or {}).items()},
            "units_planned": plan.get("n_units"), "units_scheduled": plan.get("n_scheduled"),
            "scheduled_bytes": plan.get("scheduled_bytes"),
            "planned_by_predicted_class": plan.get("by_class"),
            "exposure_checkpoints": len(recs), "exposures_analysed": len(analysed),
            "exposure_statuses": status_counts,
            "exposures_eclipse_class": n_ecl, "exposures_transit_class": n_tr,
            "exposures_phase_unresolved": n_unres,
            "predicted_vs_found_class": pred_vs,
            "integrations_analysed": int(sum(int(r.get("n_integrations_raw") or r.get("n_integrations") or 0)
                                             for r in analysed)),
            "bytes_analysed": int(sum(int(r.get("total_bytes") or 0) for r in analysed)),
            "resolution_elements_scanned": m_total,
            "narrow_features": len(rows),
            "features_eclipse_tested": sum(1 for r in rows if r["eclipse_tested"]),
            "features_vanish_snr_ge_3": sum(1 for r in rows if r["eclipse_tested"]
                                            and _f(r["eclipse_vanish_snr"]) >= 3.0),
            "tiers": tiers,
        },
        "by_mode": by_mode,
        "rejections": rejections, "search_guard_counters": guard_totals,
        "recurrent_wavelength_bins": sorted(float(b * float(rcfg["bin_um"])) for b in rec_bins),
        "fdr": {"alpha": fcfg["alpha"], "m_total": m_total, "threshold": thresh},
        "verify": verify,
        "targets": per_target,
        "candidates": [r for r in rows if r["tier"] in ("interest", "candidate")],
        "best_eclipse_tested": sorted([r for r in rows if r["eclipse_tested"]],
                                      key=lambda r: -_f(r["eclipse_vanish_snr"]))[:50],
        "watch": [r for r in rows if r["tier"] == "watch"][:200],
        "sensitivity": {r.get("exposure_key"): {"target": r["target"], "instrument": r.get("instrument"),
                                                 "mode": r.get("mode"), "R": r.get("R"),
                                                 "phase_class": r.get("phase_class"),
                                                 "velocity_width_kms": r.get("velocity_width_kms"),
                                                 "ew_5sigma_limit_um": r.get("ew_5sigma_limit_um"),
                                                 "ew_5sigma_limit_out_um": r.get("ew_5sigma_limit_out_um"),
                                                 "ew_5sigma_limit_diff_um": r.get("ew_5sigma_limit_diff_um"),
                                                 "n_integrations": r.get("n_integrations_raw") or r.get("n_integrations")}
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
    # One compact row per exposure: the committed record of what was analysed
    # (the full checkpoints stay in the run artifact for reduce_only_run_id).
    exposures = [{"target": r.get("target"), "exposure_key": r.get("exposure_key"),
                  "obs_id": r.get("obs_id"), "status": r.get("status"), "level": r.get("level"),
                  "instrument": r.get("instrument"), "mode": r.get("mode"),
                  "planet": r.get("planet"), "phase_class": r.get("phase_class"),
                  "predicted_class": (r.get("predicted") or {}).get("phase_class"),
                  "n_integrations": r.get("n_integrations_raw") or r.get("n_integrations"),
                  "binned_by": r.get("binned_by"), "n_grids": r.get("n_grids"),
                  "n_products": r.get("n_products"),
                  "total_bytes": r.get("total_bytes"), "time_source": r.get("time_source"),
                  "coverage": next((p.get("coverage") for p in r.get("planets", [])
                                    if p.get("planet") == r.get("planet")), None),
                  "n_features": len(r.get("features", [])),
                  "n_features_difference": r.get("n_features_difference"),
                  "n_features_eclipse_tested": sum(1 for f in r.get("features", [])
                                                   if f.get("eclipse_tested")),
                  "best_vanish_snr": max((_f((f.get("eclipse") or {}).get("eclipse_vanish_snr"))
                                          for f in r.get("features", [])), default=None),
                  "ew_5sigma_limit_um": r.get("ew_5sigma_limit_um"),
                  "ew_5sigma_limit_out_um": r.get("ew_5sigma_limit_out_um"),
                  "ew_5sigma_limit_diff_um": r.get("ew_5sigma_limit_diff_um"),
                  "analysis_seconds": r.get("analysis_seconds"),
                  "notes": r.get("notes")} for r in recs]
    for e in exposures:
        if e["best_vanish_snr"] is not None and not np.isfinite(e["best_vanish_snr"]):
            e["best_vanish_snr"] = None
    _write_json(out_dir / "exposures.json", {"generated_utc": summary["generated_utc"],
                                             "rows": exposures})
    print("[lantern] assess:", json.dumps(_json_safe({k: summary[k] for k in
                                                       ("verdict", "funnel", "rejections")})))
    return summary


# --- verify: known eclipses through the real reader and labeller ------------------------------
def _continuum_series(stack: dict) -> np.ndarray:
    """Broad-band light curve: per-integration median over the interior 60% of
    the finite samples, normalised to its own median."""
    f = np.asarray(stack["flux"], float)
    n_wl = f.shape[1]
    fin = np.flatnonzero(np.isfinite(np.nanmedian(f[: min(64, f.shape[0])], axis=0)))
    if fin.size < 8:
        fin = np.arange(n_wl)
    lo, hi = fin[int(0.2 * fin.size)], fin[int(0.8 * fin.size) - 1]
    with np.errstate(invalid="ignore"):
        c = np.nanmedian(f[:, lo:hi + 1], axis=1)
    med = np.nanmedian(c)
    return c / med if np.isfinite(med) and med != 0 else c


def verify_eclipse_stack(stack: dict, ephemerides: list[Ephemeris], conf: dict,
                         injection_amp: float | None = None) -> dict:
    """Does the continuum light curve of a KNOWN eclipse observation drop while
    the labeller says the planet is occulted?

    The broad-band series is detrended by a straight line fitted to the
    out-of-eclipse integrations, then compared in-vs-out (depth, its error
    from the out-of-eclipse scatter), and the free two-level step scan of
    :func:`eclipse_discriminant` reports where the drop actually sits relative
    to the predicted ingress.  Optionally a vanishing line of amplitude
    ``injection_amp`` (fraction of the local continuum) is injected at the
    centre of the grid and the full :func:`analyse_stack` chain must recover
    it as a ``candidate`` -- the end-to-end check on real noise.
    """
    pcfg, dcfg = conf.get("phase", {}), conf.get("discriminant", {})
    vcfg = {**DEFAULTS["verify"], **(conf.get("verify") or {})}
    times = np.asarray(stack["times"], float)
    extra = 0.0 if stack.get("time_source") in BARYCENTRIC_TIME_SOURCES \
        else float(pcfg.get("header_time_uncertainty_days", 0.006))
    best = None
    for eph in ephemerides:
        lab = label_integrations(times, eph, pcfg, extra_timing_sigma=extra)
        if best is None or lab["coverage"]["n_in_eclipse"] > best[1]["coverage"]["n_in_eclipse"]:
            best = (eph, lab)
    if best is None:
        return {"passed": False, "reason": "no_ephemeris"}
    eph, lab = best
    lab["cadence_days"] = float(np.nanmedian(np.diff(times))) if times.size > 1 else None
    out = {"planet": eph.name, "phase_class": lab["phase_class"], "coverage": lab["coverage"],
           "notes": lab["notes"], "time_source": stack.get("time_source"),
           "n_integrations": int(times.size), "time_span_days": float(times.max() - times.min()),
           "eclipses": [{k: e[k] for k in ("mid", "t1", "t2", "t3", "t4", "timing_sigma", "phase_reason")}
                        for e in lab["eclipses"]],
           "transits": [{k: e[k] for k in ("mid", "t1", "t4", "timing_sigma")} for e in lab["transits"]],
           "window_bjd": [float(times.min()), float(times.max())]}
    inn = lab["in_eclipse"]
    outm = lab["out_eclipse"] & ~lab["in_transit"] & ~lab["transit_contact"]
    if inn.sum() < 4 or outm.sum() < 8:
        out.update(passed=False, reason="labeller_placed_no_eclipse_in_window")
        return out
    # A phase curve (WASP-43 b: two eclipses AND a transit in 1.1 d) is not a
    # two-level light curve: a free step fitted over the whole visit locks onto
    # the transit, which is deeper than the eclipse, and the timing check then
    # fails on an observation whose eclipse is perfectly well placed.  Restrict
    # the verification to a window around ONE predicted eclipse -- wide enough
    # that the free step can still land far from the prediction (the window is
    # ~10x the tolerance), narrow enough that the transit and most of the
    # thermal phase variation are outside it.
    ecl_all = [e for e in lab["eclipses"] if not e.get("unplaceable")]
    win_frac = float(vcfg.get("window_fraction_of_period", 0.25))
    if ecl_all:
        best_e = max(ecl_all, key=lambda e: int(np.count_nonzero(
            (times >= e["t2"]) & (times <= e["t3"]))))
        half = max(win_frac * eph.period, 2.0 * eph.duration)
        win = np.abs(times - best_e["mid"]) <= half
        win &= ~lab["in_transit"] & ~lab["transit_contact"]
        if np.count_nonzero(win & inn) >= 4 and np.count_nonzero(win & outm) >= 8:
            inn = inn & win
            outm = outm & win
            out["verify_window_days"] = [float(best_e["mid"] - half), float(best_e["mid"] + half)]
            out["verify_window_n_integrations"] = int(np.count_nonzero(win))
    c = _continuum_series(stack)
    ok = np.isfinite(c) & (inn | outm)
    # Detrend with a line through the out-of-eclipse points (a settling ramp
    # or slope must not masquerade as, or hide, the eclipse).
    x = times - times.mean()
    coef = np.polyfit(x[outm & ok], c[outm & ok], 1)
    d = c - np.polyval(coef, x)
    s_out = float(np.nanstd(d[outm & ok], ddof=1))
    mu_in, mu_out = float(np.nanmean(d[inn & ok])), float(np.nanmean(d[outm & ok]))
    depth = mu_out - mu_in
    depth_err = s_out * np.sqrt(1.0 / (inn & ok).sum() + 1.0 / (outm & ok).sum())
    # Restrict the discriminant to the same window and phase groups.
    lab_w = dict(lab)
    lab_w["in_eclipse"], lab_w["out_eclipse"] = inn, outm
    disc = eclipse_discriminant(np.where(ok, d, np.nan), np.full(d.size, s_out), c,
                                lab_w, times, dcfg)
    ecl = [e for e in lab["eclipses"] if not e.get("unplaceable")]
    e_ref = best_e if ecl_all else (ecl[0] if ecl else None)
    tol_days = ((float(dcfg.get("timing_tolerance_ingress_units", 2.0)) * e_ref["ingress_duration"]
                 + e_ref["timing_sigma"] + 2.0 * (lab["cadence_days"] or 0.0))
                if e_ref else np.nan)
    off = disc.get("free_step_offset_days")
    lo_d, hi_d = vcfg["depth_range"]
    checks = {
        "depth_positive_and_significant": bool(depth > 0 and depth / depth_err >= float(vcfg["min_depth_snr"])),
        "depth_in_planetary_range": bool(lo_d <= depth <= hi_d),
        "free_step_at_predicted_ingress": bool(off is not None and np.isfinite(off)
                                               and abs(off) <= tol_days),
        "step_beats_flat": bool(np.isfinite(disc.get("chi2_flat", np.nan))
                                and disc["chi2_step_predicted"] < disc["chi2_flat"] - 25.0),
    }
    out.update(depth=depth, depth_err=depth_err, depth_snr=depth / depth_err if depth_err > 0 else None,
               out_scatter=s_out, detrend_slope_per_day=float(coef[0]),
               free_step_offset_days=off, timing_tolerance_days=tol_days,
               chi2_flat=disc.get("chi2_flat"), chi2_step_predicted=disc.get("chi2_step_predicted"),
               chi2_step_free=disc.get("chi2_step_free"), checks=checks,
               continuum_binned=_bin_series(d, 60), in_eclipse_binned=_bin_series(inn.astype(float), 60),
               passed=all(checks.values()))
    if injection_amp:
        f = np.asarray(stack["flux"], float).copy()
        wl = np.asarray(stack["wavelength"], float)
        fin = np.flatnonzero(np.isfinite(wl) & np.all(np.isfinite(f[: min(32, f.shape[0])]), axis=0))
        j = int(fin[fin.size // 2]) if fin.size else f.shape[1] // 2
        prof = np.exp(-0.5 * ((np.arange(f.shape[1]) - j) / 0.9) ** 2)
        # The injected line vanishes at EVERY eclipse in the visit (the
        # verification window above restricts only the continuum check; the
        # analysis chain below sees the whole exposure and the full labels).
        vis = np.where(lab["in_eclipse"], 0.0, np.where(lab["eclipse_contact"], 0.5, 1.0))
        cont_j = np.nanmedian(f[:, max(0, j - 10): j + 11], axis=1)
        f += (injection_amp * cont_j * vis)[:, None] * prof[None, :]
        s2 = dict(stack)
        s2["flux"] = f
        rec = analyse_stack(s2, [eph], conf, "verify")
        near = [x for x in rec["features"] if abs(x["index"] - j) <= 2]
        out["injection"] = {"wavelength_um": float(wl[j]), "amp": injection_amp,
                            "found_in": near[0].get("found_in") if near else None,
                            "snr_difference": near[0].get("snr_difference") if near else None,
                            "drift_control_snr": ((near[0].get("eclipse") or {})
                                                  .get("drift_control_snr") if near else None),
                            "ew_5sigma_limit_out_um": rec.get("ew_5sigma_limit_out_um"),
                            "ew_5sigma_limit_diff_um": rec.get("ew_5sigma_limit_diff_um"),
                            "recovered": bool(near), "tier": near[0]["tier_local"] if near else None,
                            "vetoes": near[0]["vetoes_local"] if near else None,
                            "snr": near[0]["snr"] if near else None,
                            "eclipse_vanish_snr": (near[0].get("eclipse") or {}).get("eclipse_vanish_snr")
                            if near else None,
                            "in_eclipse_spectrum_snr": (near[0].get("eclipse") or {}).get("in_eclipse_spectrum_snr")
                            if near else None,
                            "n_features_total": len(rec["features"]),
                            "ew_5sigma_limit_um": rec.get("ew_5sigma_limit_um")}
        # 'candidate', or 'interest' with no veto (the in-eclipse residual at
        # the line centre above 2 sigma on noise alone, ~1 in 8 on the
        # synthetic forest): either is the line coming back clean.
        clean = bool(near and near[0]["tier_local"] in ("candidate", "interest")
                     and not near[0]["vetoes_local"])
        out["passed"] = bool(out["passed"] and clean)
        out["checks"]["injected_line_recovered_clean"] = clean
    return out


def verify(out_dir: Path, conf: dict, work_dir: Path | None = None,
           download_fn=None, read_fn=None, inventory_path: Path | None = None) -> dict:
    """Run :func:`verify_eclipse_stack` on the configured known-eclipse cases
    (``config/lantern.yaml: verify.cases``), writing ``verify.json``.  The
    ephemerides come from the inventory (or the archive when absent)."""
    from . import acquire
    vcfg = {**DEFAULTS["verify"], **(conf.get("verify") or {})}
    acq = conf["acquire"]
    inv_path = inventory_path or (out_dir / "inventory.json")
    inv = json.loads(Path(inv_path).read_text()) if Path(inv_path).exists() else {"targets": {}}
    work = Path(work_dir or (out_dir / "_work"))
    work.mkdir(parents=True, exist_ok=True)
    download_fn = download_fn or (lambda uri, local: acquire.download_x1dints(
        uri, local, acq["retries"], acq["retry_pause_s"]))
    read_fn = read_fn or acquire.read_x1dints_grids
    res = {"generated_utc": _utc(), "checkpoint_version": CHECKPOINT_VERSION, "cases": {}}
    planets_df = None
    for case in vcfg.get("cases") or []:
        host = case["host"]
        name = case.get("name") or f"{host}:{case.get('exposure_key')}"
        entry = inv.get("targets", {}).get(host)
        if entry is None:
            if planets_df is None:
                planets_df = acquire.fetch_transiting_planets(acq["retries"], acq["retry_pause_s"])
            pl = planets_df[planets_df["hostname"] == host] if len(planets_df) else []
            eph = [ephemeris_from_archive_row(r, r.get("pl_name")) for r in
                   (pl.to_dict("records") if len(pl) else [])]
        else:
            eph = _ephemerides_of(entry)
        c = {"host": host, "exposure_key": case.get("exposure_key"), "expected": case.get("expect"),
             "uris": list(case.get("uris") or []), "passed": False}
        if not eph:
            c["reason"] = "no_ephemeris_for_host"
            res["cases"][name] = c
            continue
        stacks = []
        for uri in c["uris"]:
            local = work / os.path.basename(uri)
            st = download_fn(uri, local)
            if st != "COMPLETE":
                c["reason"] = f"download: {st}"
                break
            stacks.extend(_as_grids(read_fn(local)))
            try:
                local.unlink()
            except OSError:
                pass
        grids = acquire.group_segments_by_grid(stacks)
        if not grids:
            c.setdefault("reason", "no grid read")
            res["cases"][name] = c
            continue
        g = grids[0]
        k = _bin_factor(g, eph, acq)
        g = acquire.bin_integrations(g, k) if k > 1 else g
        c["grid"] = g.get("grid")
        c["hdu_layout"] = (g.get("meta") or {}).get("hdu_layout")
        c["cal_ver"] = (g.get("meta") or {}).get("CAL_VER")
        c["binned_by"] = k
        try:
            c.update(verify_eclipse_stack(g, eph, conf, float(vcfg.get("injection_amp") or 0.0)))
        except Exception as exc:  # noqa: BLE001
            c["reason"] = f"verify raised {exc!r}"
        if case.get("expect") and c.get("phase_class") not in (case["expect"], "both"):
            c["passed"] = False
            c["reason"] = f"expected {case['expect']} got {c.get('phase_class')}"
        res["cases"][name] = c
        print(f"[lantern] verify {name}: {json.dumps(_json_safe({k: v for k, v in c.items() if k not in ('continuum_binned', 'in_eclipse_binned', 'hdu_layout')}))}")
    n_pass = sum(1 for c in res["cases"].values() if c.get("passed"))
    res["n_cases"], res["n_passed"] = len(res["cases"]), n_pass
    res["verdict"] = ("PHASE_VERIFIED" if res["cases"] and n_pass == len(res["cases"])
                      else ("PHASE_PARTIALLY_VERIFIED" if n_pass else "PHASE_NOT_VERIFIED"))
    _write_json(out_dir / "verify.json", res)
    print(f"[lantern] verify: {res['verdict']} ({n_pass}/{len(res['cases'])})")
    return res


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
        # With the ~1% static pixel pattern that real x1d products carry, the
        # out-of-eclipse spectrum cannot see a 2% line at all; the difference
        # must still return it, and must still reject the same confounders.
        "pattern_planet_line_vanishes": (dict(line_amp=0.02, fixed_pattern_amp=0.01),
                                         "candidate"),
        "pattern_stellar_line_constant": (dict(line_amp=0.02, line_vanishes=False,
                                               fixed_pattern_amp=0.01), None),
        "pattern_no_line": (dict(line_amp=0.0, fixed_pattern_amp=0.01), None),
        "pattern_line_ramp": (dict(line_amp=0.02, line_vanishes=False, line_ramp_amp=3.0,
                                   ramp_tau=60.0, fixed_pattern_amp=0.01), "none"),
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
                              "found_in": near[0].get("found_in") if near else None,
                              "n_features_out_spectrum": rec.get("n_features_out_spectrum"),
                              "n_features_difference": rec.get("n_features_difference"),
                              "ew_5sigma_limit_out_um": rec.get("ew_5sigma_limit_out_um"),
                              "ew_5sigma_limit_diff_um": rec.get("ew_5sigma_limit_diff_um"),
                              "vetoes": near[0]["vetoes_local"] if near else None}
        out["all_as_expected"] &= ok
    _write_json(out_dir / "selftest.json", out)
    print("[lantern] selftest:", json.dumps(_json_safe(out)))
    return out


# --- entry points --------------------------------------------------------------------------
def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("stage", choices=["probe", "inventory", "screen", "assess", "selftest", "verify"])
    p.add_argument("--out-dir", default=None, help="default results/lantern")
    p.add_argument("--targets", default="", help="semicolon-separated host names (inventory)")
    p.add_argument("--instruments", default="", help="comma-separated (inventory/probe)")
    p.add_argument("--n-shards", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--inventory", default=None, help="path to inventory.json (screen/verify)")
    p.add_argument("--replan", default=None,
                   help="inventory: re-plan this existing inventory.json offline (no archive calls)")
    p.add_argument("--max-file-gb", type=float, default=None)
    p.add_argument("--max-products-per-target", type=int, default=0)
    p.add_argument("--deadline-minutes", type=float, default=0.0,
                   help="screen: start no new exposure after this many minutes")
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
        return inventory(out_dir, conf, targets=targets, n_shards=args.n_shards,
                         replan_from=Path(args.replan) if args.replan else None)
    if args.stage == "screen":
        return screen(out_dir, conf, shard=args.shard, n_shards=args.n_shards,
                      inventory_path=Path(args.inventory) if args.inventory else None,
                      max_file_bytes=(args.max_file_gb * 1e9) if args.max_file_gb else None,
                      max_products_per_target=args.max_products_per_target,
                      work_dir=Path(args.work_dir) if args.work_dir else None,
                      deadline_minutes=args.deadline_minutes)
    if args.stage == "assess":
        return assess(out_dir, conf)
    if args.stage == "verify":
        return verify(out_dir, conf, work_dir=Path(args.work_dir) if args.work_dir else None,
                      inventory_path=Path(args.inventory) if args.inventory else None)
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
