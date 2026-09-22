"""Stage orchestration for SLAG-WD.  Writes ``results/slag/``.

Stages
------
``probe``    what VizieR and GitHub hold for PEWDD: the tables under the
             catalogue id, the served column names / units / descriptions,
             the resolved roles (name, spectral type, Teff, log g, reference,
             atmosphere, one value / error / limit column per element), the
             reference convention the descriptions state, and the PyllutedWD
             repository's timescale files.  Writes ``probe.json``.
``acquire``  the PEWDD rows over the VizieR ladder (GitHub second), the
             timescale tables if any parse.  Writes ``acquire.json``,
             ``acquisition_log.json``, ``data/``.
``screen``   per panel (row): Tier 1 restricted and free-fractionation fits,
             the calibrated misfit p (panels with >= 5 measured elements),
             Tier 2 pair residuals with the rest-of-panel phase set, the
             refinery flags, the two-parcel kill where something exceeded,
             the multi-reference check.  Shardable.  Writes ``screen_<i>of<n>.json``.
``assess``   merges the shards: the misfit list, the pair table, the flags,
             the controls, the funnel and the verdict.  Writes ``summary.json``,
             ``misfit_list.csv``, ``pairs.csv``, ``flags.csv``, ``candidates.csv``,
             ``controls.json``.

Verdict vocabulary (``summary.json["verdict"]``)
------------------------------------------------
``NO_DATA_REACHED``              no PEWDD rows were acquired
``INFORMATION_LIMITED_ONLY``     rows, but no panel with >= 5 measured elements
``MISFIT_LIST_PRODUCED``         the calibrated list exists; no pair or flag survives its kills
``REFINERY_VECTOR_CANDIDATE``    >= 1 pair residual or flag survives every kill --- pending vet
Degradation (a failed route, a missing error column, an unresolved
atmosphere, an embedded rather than fetched timescale law) is a first-class
field, never folded into the verdict string.
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

from . import acquire as A
from .family import (
    load_family,
    load_measured_meteorites,
    measured_suite_report,
    ratio_envelope,
)
from .misfit import FitSettings, Panel, calibrate_misfit, fit_panel, naive_p
from .pairs import (
    KILL_INFORMATION_LIMITED,
    apply_kills,
    fit_two_parcel,
    pair_residuals,
    refinery_flags,
)
from .sinking import SOURCE_SCALING, TimescaleModel, relative_timescale_library

VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_INFO_LIMITED = "INFORMATION_LIMITED_ONLY"
VERDICT_LIST = "MISFIT_LIST_PRODUCED"
VERDICT_CANDIDATE = "REFINERY_VECTOR_CANDIDATE"

DEFAULTS: dict = {
    "sources": {
        "pewdd": {"vizier_catalogue": "J/A+A/691/A352", "vizier_table": "J/A+A/691/A352/pewdd",
                  "github_repo": "jamietwilliams/PEWDD", "max_rows": 20000,
                  "meteorite_patterns": [r"(?i)meteorite[^/]*\.csv$",
                                         r"(?i)mass_fractions\.csv$"],
                  "max_meteorite_files": 8},
        "pyllutedwd": {"github_repo": "andrewmbuchan4/PyllutedWD_Public", "max_files": 12,
                       "koester_urls": []},
    },
    "elements": {"all": ["Li", "Be", "B", "C", "N", "O", "Na", "Mg", "Al", "Si", "P", "S", "K",
                         "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Sr",
                         "Sn", "Ba", "Pb"],
                 "trace_panel": ["Sc", "V", "Co", "Cu", "Sr", "Zn", "Be", "Li", "K"]},
    "panels": {"min_elements": 5, "default_error_dex": 0.2, "error_floor_dex": 0.02},
    "fit": {"sigma_sys": 0.15, "n_random": 400, "n_refine": 2, "refine_maxiter": 500,
            "n_cal_meteorite": 50,
            "t_cut_max_restricted": 1400.0, "depth_range": [-1.5, 4.0],
            "t_acc_range": [0.01, 30.0], "t_dec_range": [0.0, 3.0],
            "fractionation_width_K": 100.0, "phase_delta": 4.0, "n_cal": 150,
            "draw_mode": "posterior", "calibrate_full_below_p": 0.10},
    "tier2": {"pairs": [["Ti", "Al"], ["Sc", "Ca"], ["Ca", "Al"], ["Sr", "Ca"], ["Mn", "Cr"],
                        ["Ni", "Co"]],
              "z_threshold": 4.0, "rest_p_min": 0.01, "nsig_flags": 4.0,
              "exotic_mantle_margin_dex": 0.5, "teff_cool_he_K": 7000.0, "multi_ref_sigma": 3.0,
              "free_fractionation_p_min": 0.05},
    "thresholds": {"p_misfit_unexplained": 0.01, "p_misfit_watch": 0.05},
    "controls": [],
    "run": {"results_dir": "results/slag", "fetch_retries": 3, "fetch_timeout_s": 180},
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _deep_update(base: dict, upd: dict) -> dict:
    out = dict(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "config").is_dir():
            return p
    return here.parents[3]


def load_slag_config(path: Path | None = None) -> dict:
    """``config/slag.yaml`` over :data:`DEFAULTS`; a missing file degrades."""
    try:
        import yaml
        path = Path(path) if path is not None else _repo_root() / "config" / "slag.yaml"
        if not path.exists():
            return _deep_update(DEFAULTS, {})
        return _deep_update(DEFAULTS, yaml.safe_load(path.read_text()) or {})
    except Exception as exc:                                  # noqa: BLE001
        print(f"[slag] config not loaded ({exc!r}); using defaults")
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
    if isinstance(o, (Path, set)):
        return str(o) if isinstance(o, Path) else sorted(o)
    return str(o)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def fit_settings(cfg: dict, *, restricted: bool) -> FitSettings:
    f = cfg["fit"]
    s = FitSettings(sigma_sys=float(f["sigma_sys"]), n_random=int(f["n_random"]),
                    n_refine=int(f["n_refine"]), refine_maxiter=int(f["refine_maxiter"]),
                    depth_range=tuple(float(x) for x in f["depth_range"]),
                    t_acc_range=tuple(float(x) for x in f["t_acc_range"]),
                    t_dec_range=tuple(float(x) for x in f["t_dec_range"]),
                    fractionation_width_K=float(f["fractionation_width_K"]),
                    phase_delta=float(f.get("phase_delta", 4.0)))
    return s.restricted(float(f["t_cut_max_restricted"])) if restricted else s


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_probe(cfg: dict, out_dir: Path, *, fetch_fn=None, query_fn=None) -> dict:
    A.reset_route_state()
    log = A.AcquisitionLog()
    t0 = _time.time()
    pew = A.probe_pewdd(cfg, fetch_fn=fetch_fn, query_fn=query_fn, log=log)
    ts = A.discover_timescale_tables(cfg, out_dir, fetch_fn=fetch_fn, log=log)
    fam = load_family()
    measured = _attach_measured_meteorites(fam, out_dir, cfg)
    envs = {}
    for a, b in cfg["tier2"]["pairs"]:
        e = ratio_envelope(fam, a, b, t_cut_max=float(cfg["fit"]["t_cut_max_restricted"]))
        envs[f"{a}/{b}"] = {k: e[k] for k in ("lo", "hi", "width_dex", "endmember_lo",
                                              "endmember_hi", "measured_n", "measured_lo",
                                              "measured_hi", "measured_trim_lo",
                                              "measured_trim_hi", "envelope_source",
                                              "unfractionated_lo",
                                              "unfractionated_hi") if k in e}
    out = {"generated_utc": _now(), "elapsed_s": round(_time.time() - t0, 1), "pewdd": pew,
           "timescales": ts, "pair_envelopes_restricted": envs,
           "measured_meteorites": measured,
           "family_endmembers": fam.endmembers, "family_elements": fam.elements,
           "acquisition": log.as_dict()}
    _write_json(out_dir / "probe.json", out)
    return out


def stage_acquire(cfg: dict, out_dir: Path, *, fetch_fn=None, tap_fn=None) -> dict:
    A.reset_route_state()
    log = A.AcquisitionLog()
    t0 = _time.time()
    pew = A.fetch_pewdd(cfg, out_dir, fetch_fn=fetch_fn, tap_fn=tap_fn, log=log)
    ts = A.discover_timescale_tables(cfg, out_dir, fetch_fn=fetch_fn, log=log)
    met = A.fetch_meteorite_tables(cfg, out_dir, fetch_fn=fetch_fn, log=log)
    roles = None
    if pew.get("columns"):
        # descriptions travel from the probe when it ran in this checkout
        descs, units = {}, {}
        pp = out_dir / "probe.json"
        if pp.exists():
            try:
                pj = json.loads(pp.read_text())
                if pj["pewdd"]["vizier"].get("columns"):
                    descs = pj["pewdd"]["vizier"].get("descriptions") or {}
                    units = pj["pewdd"]["vizier"].get("units") or {}
            except Exception:                                 # noqa: BLE001
                pass
        roles = A.resolve_roles(pew["columns"], cfg["elements"]["all"], units=units,
                                descriptions=descs)
    out = {"generated_utc": _now(), "elapsed_s": round(_time.time() - t0, 1), "pewdd": pew,
           "timescales": ts, "meteorites": met, "roles": roles,
           "status": pew["status"] if pew.get("n_rows") else A.STATUS_FAILED}
    _write_json(out_dir / "acquire.json", out)
    log.write(out_dir / "acquisition_log.json")
    return out


def _load_table(out_dir: Path, cfg: dict, input_csv: str | None = None) -> tuple[pd.DataFrame | None, dict | None, dict]:
    """The acquired PEWDD rows and roles (or an injected CSV), with provenance."""
    prov: dict = {"source": None}
    if input_csv:
        df = pd.read_csv(input_csv, low_memory=False)
        roles = A.resolve_roles([str(c) for c in df.columns], cfg["elements"]["all"])
        prov["source"] = f"input_csv:{input_csv}"
        return df, roles, prov
    ap = out_dir / "acquire.json"
    if not ap.exists():
        return None, None, {"source": None, "error": "no acquire.json"}
    aj = json.loads(ap.read_text())
    path = aj.get("pewdd", {}).get("path")
    if not path or not Path(path).exists():
        # acquire.json records the absolute path of the checkout that fetched
        # the rows (on a runner, /home/runner/work/...).  A later stage, a
        # different checkout or a local replay sees the same file under its own
        # results directory, so fall back on the basename before giving up.
        cands = []
        if path:
            name = Path(str(path)).name
            cands += [out_dir / "data" / name, out_dir / name,
                      _repo_root() / "results" / "slag" / "data" / name,
                      _repo_root() / str(path).lstrip("/")]
        found = next((c for c in cands if c.exists()), None)
        if found is not None:
            path = str(found)
        else:
            return None, None, {"source": None, "error": "acquired table missing",
                                "acquire_status": aj.get("status"),
                                "recorded_path": path,
                                "searched": [str(c) for c in cands]}
    df = pd.read_csv(path, low_memory=False)
    roles = aj.get("roles")
    if not roles:
        roles = A.resolve_roles([str(c) for c in df.columns], cfg["elements"]["all"])
    prov.update(source=aj["pewdd"].get("route"), path=path, n_rows=int(len(df)))
    return df, roles, prov



def _attach_measured_meteorites(fam, out_dir: Path, cfg: dict) -> dict:
    """Attach PEWDD's own meteorite compilations to the family, if they landed.

    They widen the Tier 2 envelopes to what nature has been MEASURED to do
    (docs/slag.md): the compiled end-members are eighteen averaged vectors,
    and individual stones and irons reach a long way past them.  When nothing
    parses, the run says so and the envelopes come from the end-members alone
    --- which is a narrower, more permissive test and is recorded as such.
    """
    rep: dict = {}
    paths = sorted(glob.glob(str(out_dir / "data" / "meteorite*.csv")))
    paths += sorted(glob.glob(str(out_dir / "data" / "meteorites_*.csv")))
    ap = out_dir / "acquire.json"
    if ap.exists():
        try:
            for f in (json.loads(ap.read_text()).get("meteorites") or {}).get("files", []):
                lp = f.get("local")
                if lp:
                    cand = Path(lp)
                    if not cand.exists():
                        cand = out_dir / "data" / cand.name
                    if cand.exists():
                        paths.append(str(cand))
        except Exception:                                     # noqa: BLE001
            pass
    paths = sorted(set(paths))
    suite = load_measured_meteorites(paths, report=rep) if paths else None
    fam.measured = suite
    out = {"attached": suite is not None, "paths": paths, **rep}
    out.update(measured_suite_report(suite, [tuple(p) for p in cfg["tier2"]["pairs"]]))
    return out


def _timescale_model(fam, out_dir: Path, *, df=None, roles: dict | None = None) -> TimescaleModel:
    """The sinking lever, from the best source this run reached.

    Priority, highest first: the object's OWN published timescales (per panel,
    applied inside :class:`~seti.slag.misfit.PanelModel`); a timescale table
    that was fetched and parsed; the relative-timescale library built from
    every catalogue row that publishes timescales; the embedded mass-scaling
    law.  The library is built here because it needs the acquired table.
    """
    tsm = TimescaleModel(fam)
    if df is not None and roles:
        lib = relative_timescale_library(df, roles.get("sinking_time_columns") or {},
                                         reference=tsm.reference,
                                         atmosphere_column=roles.get("atm"))
        if lib.get("elements"):
            tsm.library = lib["elements"]
            tsm.library_beta = lib.get("beta")
            tsm.source = "pewdd_relative_library"
            tsm.library_meta = lib
    ap = out_dir / "acquire.json"
    if not ap.exists():
        return tsm
    try:
        parsed = json.loads(ap.read_text()).get("timescales", {}).get("parsed", {})
    except Exception:                                         # noqa: BLE001
        return tsm
    for atm, paths in parsed.items():
        if atm not in ("H", "He"):
            continue
        for p in paths:
            try:
                tab = pd.read_csv(p)
            except Exception:                                 # noqa: BLE001
                continue
            if "Teff" in tab.columns and tsm.reference in tab.columns:
                tsm.tables[atm] = tab
                tsm.source = "fetched_table"
                break
    return tsm


def _limit_bookkeeping(panels: list[Panel]) -> dict:
    """Check the parsed detections/limits against the catalogue's own counts.

    PEWDD marks an upper limit with a negative error and separately publishes
    ``total_detections`` / ``total_upper_limits``.  Agreement is the proof
    that the convention was read correctly; a disagreement is reported, never
    silently absorbed.  (Rows whose limits fall on elements outside the
    channel's element list disagree by construction, so the DETECTION count is
    the strict test and the limit count is informational.)
    """
    n_det_checked = n_det_agree = n_lim_checked = n_lim_agree = 0
    examples = []
    for p in panels:
        sd = p.meta.get("stated_n_detections")
        if sd is not None:
            n_det_checked += 1
            if int(sd) == p.n_measured:
                n_det_agree += 1
            elif len(examples) < 10:
                examples.append({"name": p.name, "reference": p.reference, "stated": int(sd),
                                 "parsed": p.n_measured, "kind": "detections"})
        sl = p.meta.get("stated_n_upper_limits")
        if sl is not None:
            n_lim_checked += 1
            if int(sl) == len(p.limit_elements):
                n_lim_agree += 1
    return {"detections_checked": n_det_checked, "detections_agree": n_det_agree,
            "upper_limits_checked": n_lim_checked, "upper_limits_agree": n_lim_agree,
            "disagreements": examples,
            "note": "the catalogue publishes its own detection/limit counts; these are the "
                    "cross-check on the negative-error upper-limit convention"}


def screen_panel(fam, tsm, panel: Panel, cfg: dict, *, others: list[Panel] | None = None,
                 n_cal: int | None = None, rng=None) -> dict:
    """Everything the channel computes for one panel."""
    s_res = fit_settings(cfg, restricted=True)
    s_full = fit_settings(cfg, restricted=False)
    t2 = cfg["tier2"]
    th = cfg["thresholds"]
    min_el = int(cfg["panels"]["min_elements"])
    n_cal = int(cfg["fit"]["n_cal"]) if n_cal is None else int(n_cal)
    rec: dict = {"name": panel.name, "name_key": panel.meta.get("name_key"),
                 "object_key": panel.meta.get("object_key") or panel.meta.get("name_key"),
                 "object_label": panel.meta.get("object_label") or panel.name,
                 "object_designations": panel.meta.get("object_designations") or [],
                 "ra": panel.meta.get("ra"), "dec": panel.meta.get("dec"),
                 "reference": panel.reference, "atmosphere": panel.atmosphere,
                 "atmosphere_how": panel.meta.get("atmosphere_how"), "teff": panel.teff,
                 "logg": panel.logg, "spt": panel.meta.get("spt"), "n_measured": panel.n_measured,
                 "n_limits": len(panel.limit_elements), "elements": list(panel.elements),
                 "values": [float(v) for v in panel.values],
                 "errors": [float(e) for e in panel.errors],
                 "limit_elements": list(panel.limit_elements),
                 "limit_values": [float(v) for v in panel.limit_values],
                 "errors_assumed_for": panel.meta.get("errors_assumed_for", []),
                 "trace_elements_measured": [e for e in cfg["elements"]["trace_panel"]
                                             if e in panel.elements],
                 "information_limited": bool(panel.n_measured < min_el),
                 "star_raw": panel.meta.get("star_raw"),
                 "n_row_sinking_times": len(panel.meta.get("sinking_times_s") or {}),
                 "provenance": {k: panel.meta.get(k) for k in
                                ("identifier", "binary", "binary_sep", "ir_excess", "gas_disc",
                                 "bfield", "t_since_acc", "comment")
                                if panel.meta.get(k) is not None},
                 "status": "INFORMATION_LIMITED" if panel.n_measured < min_el else "SCREENED"}
    if panel.n_measured < 2:
        rec["status"] = "INFORMATION_LIMITED"
        rec["fit_restricted"] = None
        rec["fit_full"] = None
        return rec
    t0 = _time.time()
    fr = fit_panel(fam, panel, tsm, s_res, rng=rng)
    ff = fit_panel(fam, panel, tsm, s_full, rng=rng)
    rec["fit_restricted"] = fr.as_dict(fam.endmembers)
    rec["fit_full"] = ff.as_dict(fam.endmembers)
    rec["timescale_source"] = fr.timescale_source
    rec["fit_restricted"]["p_naive"] = naive_p(fr.chi2, fr.n_measured - 1)
    rec["fit_full"]["p_naive"] = naive_p(ff.chi2, ff.n_measured - 1)
    rec["misfit"] = None
    rec["misfit_full"] = None
    rec["misfit_meteorite"] = None
    rec["pairs"] = []
    rec["flags"] = []
    rec["two_parcel"] = None
    rec["candidate_pairs"] = []
    rec["candidate_flags"] = []
    if panel.n_measured >= min_el:
        rec["misfit"] = calibrate_misfit(fam, panel, tsm, fr, s_res, n_draws=n_cal, rng=rng,
                                         draw_mode=str(cfg["fit"]["draw_mode"]))
        if rec["misfit"]["p_misfit"] < float(cfg["fit"].get("calibrate_full_below_p", 0.1)):
            rec["misfit_full"] = calibrate_misfit(fam, panel, tsm, ff, s_full, n_draws=n_cal,
                                                  rng=rng, draw_mode=str(cfg["fit"]["draw_mode"]))
        # the second, harder calibration: real measured meteorites as the null
        n_met = int(cfg["fit"].get("n_cal_meteorite", 0) or 0)
        if n_met > 0:
            rec["misfit_meteorite"] = calibrate_misfit(fam, panel, tsm, fr, s_res,
                                                       n_draws=n_met, rng=rng,
                                                       draw_mode="meteorite")
        pairs = pair_residuals(fam, tsm, panel, s_res,
                               pairs=[tuple(p) for p in t2["pairs"]],
                               t_cut_max=float(cfg["fit"]["t_cut_max_restricted"]),
                               z_threshold=float(t2["z_threshold"]),
                               rest_p_min=float(t2["rest_p_min"]))
        flags = refinery_flags(fam, tsm, panel, s_res, nsig=float(t2["nsig_flags"]),
                               t_cut_max=float(cfg["fit"]["t_cut_max_restricted"]),
                               exotic_mantle_margin_dex=float(t2["exotic_mantle_margin_dex"]))
        need_two = any(p["exceeds"] for p in pairs) or any(f.get("fired") for f in flags)
        if need_two:
            rec["two_parcel"] = fit_two_parcel(fam, panel, tsm, s_res, rng=rng)
        for p in pairs:
            if p["exceeds"]:
                p["kills"] = apply_kills(panel, p, ff, fr, rec["misfit"], rec["two_parcel"],
                                         teff_cool_he=float(t2["teff_cool_he_K"]),
                                         other_panels=others,
                                         multi_ref_sigma=float(t2["multi_ref_sigma"]),
                                         min_elements=min_el, rest_p_min=float(t2["rest_p_min"]),
                                         free_fractionation_p_min=float(t2["free_fractionation_p_min"]))
                p["survives"] = bool(not p["kills"])
                rec["candidate_pairs"].append(p["pair"])
            else:
                p["kills"] = []
                p["survives"] = False
        for f in flags:
            if f.get("fired"):
                fake_pair = {"a": "", "b": "", "rest_p_naive": None}
                f["kills"] = [k for k in apply_kills(panel, fake_pair, ff, fr, rec["misfit"],
                                                     rec["two_parcel"],
                                                     teff_cool_he=float(t2["teff_cool_he_K"]),
                                                     min_elements=min_el,
                                                     rest_p_min=float(t2["rest_p_min"]),
                                                     free_fractionation_p_min=float(
                                                         t2["free_fractionation_p_min"]))]
                f["survives"] = bool(not f["kills"])
                rec["candidate_flags"].append(f["flag"])
        rec["pairs"] = pairs
        rec["flags"] = flags
        p = rec["misfit"]["p_misfit"]
        rec["misfit_class"] = ("UNEXPLAINED" if p < float(th["p_misfit_unexplained"])
                               else "WATCH" if p < float(th["p_misfit_watch"]) else "NATURAL")
    else:
        rec["misfit_class"] = KILL_INFORMATION_LIMITED
    rec["elapsed_s"] = round(_time.time() - t0, 2)
    return rec


def stage_screen(cfg: dict, out_dir: Path, *, shard: str = "1/1", input_csv: str | None = None,
                 n_cal: int | None = None, names: list[str] | None = None,
                 max_panels: int | None = None) -> dict:
    fam = load_family()
    measured = _attach_measured_meteorites(fam, out_dir, cfg)
    df, roles, prov = _load_table(out_dir, cfg, input_csv)
    tsm = _timescale_model(fam, out_dir, df=df, roles=roles)
    i, n = (int(x) for x in shard.split("/"))
    out: dict = {"generated_utc": _now(), "shard": shard, "provenance": prov,
                 "timescale_source": tsm.source, "measured_meteorites": measured,
                 "panels": [], "status": A.STATUS_FAILED}
    if df is None or roles is None or not roles.get("elements"):
        out["error"] = prov.get("error", "no table or no element columns resolved")
        _write_json(out_dir / f"screen_{i}of{n}.json", out)
        return out
    grouping: dict = {}
    panels, diag = A.build_panels(df, roles, default_error_dex=float(cfg["panels"]["default_error_dex"]),
                                  error_floor_dex=float(cfg["panels"]["error_floor_dex"]),
                                  elements=cfg["elements"]["all"],
                                  sinking=roles.get("sinking_time_columns") or {},
                                  match_arcsec=float(cfg["panels"].get("object_match_arcsec",
                                                                       A.OBJECT_MATCH_ARCSEC)),
                                  grouping_out=grouping)
    out["n_rows"] = len(panels)
    out["object_grouping"] = grouping
    out["roles"] = {k: v for k, v in roles.items() if k != "elements"}
    out["element_columns"] = {e: r["value"] for e, r in roles["elements"].items()}
    out["timescale_library"] = tsm.library_meta
    out["limit_bookkeeping"] = _limit_bookkeeping(panels)
    # The shard unit and the "other sources for this object" set are the
    # reconciled OBJECT (sky position), not the name a given paper used.
    by_key: dict[str, list[Panel]] = {}
    for p in panels:
        by_key.setdefault(p.meta.get("object_key") or p.meta["name_key"], []).append(p)
    keys = sorted(by_key)
    if names:
        want = {A.normalise_name(x) for x in names}
        keys = [k for k in keys
                if k in want or any((q.meta.get("name_key") in want) for q in by_key[k])]
    keys = [k for j, k in enumerate(keys) if j % n == (i - 1)]
    if max_panels:
        keys = keys[: int(max_panels)]
    rng = np.random.default_rng(20260921 + i)
    t0 = _time.time()
    last_ckpt = t0
    for j, k in enumerate(keys, start=1):
        group = by_key[k]
        for p in group:
            others = [q for q in group if q is not p]
            try:
                rec = screen_panel(fam, tsm, p, cfg, others=others, n_cal=n_cal, rng=rng)
            except Exception as exc:                          # noqa: BLE001
                rec = {"name": p.name, "name_key": k, "reference": p.reference,
                       "n_measured": p.n_measured, "status": "SCREEN_ERROR",
                       "error": repr(exc)[:500]}
            rec["n_sources_for_object"] = len(group)
            out["panels"].append(rec)
        # Checkpoint on the OBJECT counter and on the clock, never on the panel
        # count: a group can add several panels at once and step straight over a
        # modulus, which on a multi-hour shard means the file is written far
        # less often than intended (or, for a shard whose groups are all large,
        # never).  A killed shard must lose minutes, not hours.
        if j % 25 == 0 or (_time.time() - last_ckpt) > 300.0:
            out["n_objects_done"] = j
            out["n_objects_total"] = len(keys)
            out["elapsed_s"] = round(_time.time() - t0, 1)
            # a checkpoint is a partial shard, not a failed query: say so, so a
            # shard killed mid-flight is never merged as if its table had failed
            out["status"] = "IN_PROGRESS"
            _write_json(out_dir / f"screen_{i}of{n}.json", out)
            last_ckpt = _time.time()
    out["elapsed_s"] = round(_time.time() - t0, 1)
    out["n_objects"] = len(keys)
    out["n_panels"] = len(out["panels"])
    out["status"] = A.STATUS_OK if out["panels"] else A.STATUS_ZERO
    _write_json(out_dir / f"screen_{i}of{n}.json", out)
    return out


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------
def _match_controls(cfg: dict, panels: list[dict]) -> list[dict]:
    keys = {}
    for r in panels:
        for k in {r.get("object_key"), r.get("name_key"),
                  A.normalise_name(r.get("name", ""))} | set(r.get("object_designations") or []):
            if k:
                keys.setdefault(k, []).append(r)
    all_keys = sorted(keys)
    out = []
    for c in cfg.get("controls", []):
        aliases = [c["name"]] + list(c.get("aliases", []))
        norm = [A.normalise_name(a) for a in aliases]
        hit = None
        for nk in norm:
            if nk in keys:
                hit = nk
                break
        if hit is None:
            # substring match on the alias core (digits and sign)
            for nk in norm:
                core = nk.replace("WD", "").replace("J", "")
                cands = [k for k in all_keys if core and core in k]
                if len(cands) == 1:
                    hit = cands[0]
                    break
        rec = {"name": c["name"], "expectation": c.get("expectation", ""), "aliases_tried": aliases}
        if hit is None:
            near = [k for k in all_keys if any(nk[:5] in k for nk in norm if len(nk) >= 5)][:8]
            rec.update(status="CONTROL_NOT_FOUND", nearest_keys=near)
        else:
            # the alias may have matched one designation of a star the sky
            # match reconciled with others; report the whole object
            rows = keys[hit]
            okey = (max(rows, key=lambda r: r.get("n_measured", 0)).get("object_key") or hit)
            rows = keys.get(okey, rows)
            best = max(rows, key=lambda r: r.get("n_measured", 0))
            rec.update(status="FOUND", matched_key=hit, object_key=okey,
                       object_label=best.get("object_label"),
                       designations=sorted({r.get("name") for r in rows}),
                       ra=best.get("ra"), dec=best.get("dec"),
                       teff=best.get("teff"), atmosphere=best.get("atmosphere"),
                       n_sources=len(rows),
                       best_reference=best.get("reference"), n_measured=best.get("n_measured"),
                       elements=best.get("elements"), misfit_class=best.get("misfit_class"),
                       p_misfit=(best.get("misfit") or {}).get("p_misfit"),
                       p_misfit_full=(best.get("misfit_full") or {}).get("p_misfit"),
                       p_misfit_meteorite=(best.get("misfit_meteorite") or {}).get("p_misfit"),
                       meteorite_cal_status=(best.get("misfit_meteorite") or {}).get("status"),
                       chi2_restricted=(best.get("fit_restricted") or {}).get("chi2"),
                       phase=(best.get("fit_restricted") or {}).get("phase"),
                       dominant_endmember=(best.get("fit_restricted") or {}).get("dominant_endmember"),
                       pairs=[{k: p.get(k) for k in ("pair", "obs_log_ratio", "env_lo_total",
                                                     "env_hi_total", "z_pair", "exceeds",
                                                     "rest_natural", "kills", "survives")}
                              for p in best.get("pairs", [])],
                       flags=[{k: f.get(k) for k in ("flag", "fired", "z", "kills", "survives",
                                                     "note")} for f in best.get("flags", [])],
                       max_abs_residual_sigma=(best.get("fit_restricted") or {}).get(
                           "max_abs_residual_sigma"))
        out.append(rec)
    return out


def _p_distribution(misfit_df: pd.DataFrame) -> dict:
    """Is the calibrated misfit p CALIBRATED?  The one statistic the list rests on.

    Each object's p is the fraction of its own natural draws that fit worse
    than it does.  If the natural model is adequate for the population, those
    p values are roughly uniform on (0, 1].  A pile-up at low p means the
    model is too rigid for *everyone* --- the "unexplainable" list would then
    be a statement about the model, not about any star --- and a pile-up at
    high p means it is too loose to reject anything.  Both are reported; the
    Kolmogorov-Smirnov distance from uniform is the summary number.
    """
    if not len(misfit_df) or "p_misfit" not in misfit_df:
        return {"n": 0}
    p = np.sort(misfit_df["p_misfit"].to_numpy(dtype=float))
    p = p[np.isfinite(p)]
    n = len(p)
    if n == 0:
        return {"n": 0}
    ks = float(np.max(np.abs(np.arange(1, n + 1) / n - p))) if n else float("nan")
    return {
        "n": int(n),
        "quantiles": {q: float(np.quantile(p, q / 100.0)) for q in (5, 25, 50, 75, 95)},
        "fraction_below_0.01": float(np.mean(p < 0.01)),
        "fraction_below_0.05": float(np.mean(p < 0.05)),
        "fraction_below_0.50": float(np.mean(p < 0.50)),
        "ks_distance_from_uniform": ks,
        "expected_below_0.05_if_uniform": 0.05,
        "note": "roughly uniform p means the natural model is neither too rigid nor too "
                "loose for the population; a pile-up at low p is a statement about the "
                "model, not about any star",
    }


def _count_by(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for r in rows:
        v = str(r.get(key))
        out[v] = out.get(v, 0) + 1
    return out


def stage_assess(cfg: dict, out_dir: Path) -> dict:
    shards = sorted(glob.glob(str(out_dir / "screen_*of*.json")))
    panels: list[dict] = []
    shard_meta = []
    ts_source = None
    ts_library: dict = {}
    obj_grouping: dict = {}
    measured_rep: dict = {}
    limit_book: dict = {}
    for s in shards:
        try:
            d = json.loads(Path(s).read_text())
        except Exception as exc:                              # noqa: BLE001
            shard_meta.append({"file": s, "error": repr(exc)[:300]})
            continue
        shard_meta.append({"file": s, "shard": d.get("shard"), "status": d.get("status"),
                           "n_panels": d.get("n_panels"), "n_objects": d.get("n_objects"),
                           "provenance": d.get("provenance"), "error": d.get("error")})
        ts_source = d.get("timescale_source", ts_source)
        obj_grouping = d.get("object_grouping") or obj_grouping
        measured_rep = d.get("measured_meteorites") or measured_rep
        ts_library = d.get("timescale_library") or ts_library
        if d.get("limit_bookkeeping"):
            lb = d["limit_bookkeeping"]
            for k in ("detections_checked", "detections_agree", "upper_limits_checked",
                      "upper_limits_agree"):
                limit_book[k] = limit_book.get(k, 0) + int(lb.get(k, 0))
            limit_book.setdefault("disagreements", []).extend(lb.get("disagreements", [])[:5])
        panels.extend(d.get("panels", []))
    acq = {}
    ap = out_dir / "acquire.json"
    if ap.exists():
        try:
            aj = json.loads(ap.read_text())
            acq = {"status": aj.get("status"), "route": aj.get("pewdd", {}).get("route"),
                   "n_rows": aj.get("pewdd", {}).get("n_rows"),
                   "endpoint": aj.get("pewdd", {}).get("endpoint"),
                   "reference_convention": (aj.get("roles") or {}).get("reference_convention"),
                   "n_elements_resolved": (aj.get("roles") or {}).get("n_elements_resolved"),
                   "timescale_files_parsed": aj.get("timescales", {}).get("parsed")}
        except Exception as exc:                              # noqa: BLE001
            acq = {"error": repr(exc)[:300]}
    min_el = int(cfg["panels"]["min_elements"])
    n_panels = len(panels)
    objects = {}
    for r in panels:
        objects.setdefault(r.get("object_key") or r.get("name_key") or r.get("name"),
                           []).append(r)
    screened = [r for r in panels if r.get("status") == "SCREENED" and r.get("misfit")]
    errors = [r for r in panels if r.get("status") == "SCREEN_ERROR"]
    # the misfit list: best panel per object among those calibrated
    misfit_rows = []
    for rows in objects.values():
        cal = [r for r in rows if r.get("misfit")]
        if not cal:
            continue
        best = min(cal, key=lambda r: (r["misfit"]["p_misfit"], -r["n_measured"]))
        fr = best["fit_restricted"]
        misfit_rows.append({
            "name": best["name"], "reference": best["reference"], "atmosphere": best["atmosphere"],
            "teff": best["teff"], "n_measured": best["n_measured"], "n_limits": best["n_limits"],
            "elements": " ".join(best["elements"]),
            "p_misfit": best["misfit"]["p_misfit"],
            "p_misfit_full": (best.get("misfit_full") or {}).get("p_misfit"),
            "p_misfit_meteorite": (best.get("misfit_meteorite") or {}).get("p_misfit"),
            "meteorite_cal_status": (best.get("misfit_meteorite") or {}).get("status"),
            "n_bodies_covering_panel": (best.get("misfit_meteorite") or {}).get(
                "n_bodies_covering_panel"),
            "chi2": fr["chi2"], "chi2_per_dof": fr["chi2_per_dof"], "p_naive": fr.get("p_naive"),
            "phase": fr["phase"], "t_dec": fr["t_dec_over_tau_ref"],
            "dominant_endmember": fr["dominant_endmember"], "dominant_weight": fr["dominant_weight"],
            "t_cut_K": fr["t_cut_K"], "depth_dex": fr["depth_dex"],
            "max_abs_residual_sigma": fr["max_abs_residual_sigma"],
            "worst_element": best["elements"][int(np.argmax(np.abs(fr["residual_sigma"])))]
            if best["elements"] else "",
            "misfit_class": best.get("misfit_class"),
            "n_sources": len(rows), "trace_measured": " ".join(best.get("trace_elements_measured", [])),
            "n_pairs_exceeding": sum(1 for p in best.get("pairs", []) if p.get("exceeds")),
            "n_flags_fired": sum(1 for f in best.get("flags", []) if f.get("fired")),
            "errors_assumed": " ".join(best.get("errors_assumed_for", [])),
        })
    misfit_df = pd.DataFrame(misfit_rows)
    if len(misfit_df):
        misfit_df = misfit_df.sort_values(["p_misfit", "chi2_per_dof"], ascending=[True, False])
        misfit_df.to_csv(out_dir / "misfit_list.csv", index=False)
    # pairs and flags
    pair_rows, flag_rows = [], []
    for r in screened:
        for p in r.get("pairs", []):
            pair_rows.append({"name": r["name"], "reference": r["reference"],
                              "atmosphere": r["atmosphere"], "teff": r["teff"],
                              "n_measured": r["n_measured"], **{k: p.get(k) for k in (
                                  "pair", "obs_log_ratio", "sigma", "env_lo", "env_hi", "phase_lo",
                                  "phase_hi", "env_lo_total", "env_hi_total", "z_pair", "exceeds",
                                  "phase_constraint", "rest_p_naive", "rest_natural",
                                  "rest_t_dec_max")},
                              "kills": ";".join(p.get("kills", [])), "survives": p.get("survives")})
        for f in r.get("flags", []):
            flag_rows.append({"name": r["name"], "reference": r["reference"],
                              "atmosphere": r["atmosphere"], "teff": r["teff"],
                              "n_measured": r["n_measured"], "flag": f.get("flag"),
                              "element": f.get("element"), "fired": f.get("fired"),
                              "z": f.get("z"), "phase_constraint": f.get("phase_constraint"),
                              "note": f.get("note"), "kills": ";".join(f.get("kills", [])),
                              "survives": f.get("survives")})
    pairs_df = pd.DataFrame(pair_rows)
    flags_df = pd.DataFrame(flag_rows)
    if len(pairs_df):
        pairs_df.sort_values("z_pair", key=lambda s: -s.abs()).to_csv(out_dir / "pairs.csv", index=False)
    if len(flags_df):
        flags_df.to_csv(out_dir / "flags.csv", index=False)
    # candidates: anything that exceeded / fired and survives every kill
    cand = []
    for r in screened:
        for p in r.get("pairs", []):
            if p.get("exceeds"):
                cand.append({"name": r["name"], "reference": r["reference"], "kind": "pair",
                             "what": p["pair"], "z": p["z_pair"], "kills": ";".join(p.get("kills", [])),
                             "survives": p.get("survives"), "n_measured": r["n_measured"],
                             "p_misfit": r["misfit"]["p_misfit"], "atmosphere": r["atmosphere"],
                             "teff": r["teff"]})
        for f in r.get("flags", []):
            if f.get("fired"):
                cand.append({"name": r["name"], "reference": r["reference"], "kind": "flag",
                             "what": f["flag"] + (f":{f['element']}" if f.get("element") else ""),
                             "z": f.get("z"), "kills": ";".join(f.get("kills", [])),
                             "survives": f.get("survives"), "n_measured": r["n_measured"],
                             "p_misfit": r["misfit"]["p_misfit"], "atmosphere": r["atmosphere"],
                             "teff": r["teff"]})
    cand_df = pd.DataFrame(cand)
    if len(cand_df):
        cand_df.to_csv(out_dir / "candidates.csv", index=False)
    survivors = [c for c in cand if c.get("survives")]
    controls = _match_controls(cfg, panels)
    _write_json(out_dir / "controls.json", {"generated_utc": _now(), "controls": controls})
    # funnel
    n_objects = len(objects)
    n_obj_ge_min = sum(1 for rows in objects.values() if max(r.get("n_measured", 0) for r in rows) >= min_el)
    unexplained = misfit_df[misfit_df["misfit_class"] == "UNEXPLAINED"] if len(misfit_df) else misfit_df
    watch = misfit_df[misfit_df["misfit_class"] == "WATCH"] if len(misfit_df) else misfit_df
    kill_counts: dict[str, int] = {}
    for c in cand:
        for k in (c["kills"].split(";") if c["kills"] else []):
            kill_counts[k] = kill_counts.get(k, 0) + 1
    pair_stats = {}
    if len(pairs_df):
        for pr, g in pairs_df.groupby("pair"):
            pair_stats[pr] = {"n_tested": int(len(g)), "n_exceeding": int(g["exceeds"].sum()),
                              "env_width_dex_median": float((g["env_hi_total"] - g["env_lo_total"]).median()),
                              "max_abs_z": float(g["z_pair"].abs().max())}
    degraded = []
    if acq.get("status") != A.STATUS_OK:
        degraded.append(f"acquisition:{acq.get('status')}")
    if ts_source == SOURCE_SCALING:
        degraded.append("timescales:embedded_mass_scaling_law")
    n_assumed = sum(1 for r in screened if r.get("errors_assumed_for"))
    if n_assumed:
        degraded.append(f"errors_assumed_on_{n_assumed}_panels")
    n_unres = sum(1 for r in panels if r.get("atmosphere_how") in ("unresolved", None) and r.get("status") == "SCREENED")
    if n_unres:
        degraded.append(f"atmosphere_unresolved_on_{n_unres}_panels")
    if errors:
        degraded.append(f"screen_errors:{len(errors)}")
    if n_panels == 0:
        verdict = VERDICT_NO_DATA
    elif not screened:
        verdict = VERDICT_INFO_LIMITED
    elif survivors:
        verdict = VERDICT_CANDIDATE
    else:
        verdict = VERDICT_LIST
    summary = {
        "generated_utc": _now(), "verdict": verdict, "degraded": degraded,
        "acquisition": acq, "timescale_source": ts_source,
        "timescale_library": ts_library, "limit_bookkeeping": limit_book,
        "object_grouping": obj_grouping,
        "measured_meteorites": measured_rep,
        "timescale_source_per_panel": _count_by(screened, "timescale_source"),
        "panels_with_own_timescales": sum(1 for r in screened
                                          if int(r.get("n_row_sinking_times") or 0) > 0),
        "funnel": {"panels_total": n_panels, "objects_total": n_objects,
                   "objects_with_ge_min_elements": n_obj_ge_min,
                   "panels_screened_and_calibrated": len(screened),
                   "panels_information_limited": sum(1 for r in panels if r.get("information_limited")),
                   "panels_screen_error": len(errors),
                   "misfit_unexplained": int(len(unexplained)), "misfit_watch": int(len(watch)),
                   "pairs_tested": int(len(pairs_df)), "pairs_exceeding": int(pairs_df["exceeds"].sum()) if len(pairs_df) else 0,
                   "flags_fired": int(flags_df["fired"].sum()) if len(flags_df) else 0,
                   "candidates_before_kills": len(cand), "candidates_surviving": len(survivors),
                   "kills": kill_counts},
        "misfit_p_distribution": _p_distribution(misfit_df),
        "pair_statistics": pair_stats,
        "misfit_list_top": misfit_df.head(25).to_dict("records") if len(misfit_df) else [],
        "survivors": survivors,
        "controls": [{k: c.get(k) for k in ("name", "status", "matched_key", "designations",
                                             "ra", "dec", "teff", "atmosphere", "n_sources",
                                             "n_measured",
                                             "p_misfit", "p_misfit_full", "p_misfit_meteorite",
                                             "meteorite_cal_status", "misfit_class", "phase",
                                             "dominant_endmember", "max_abs_residual_sigma")}
                     for c in controls],
        "shards": shard_meta,
        "min_elements": min_el,
        "thresholds": cfg["thresholds"], "tier2": {k: v for k, v in cfg["tier2"].items()},
        "note": ("The misfit list is the deliverable: a calibrated ranking of every >= 5-element "
                 "panel by how badly the best natural parcel reproduces it. A low p is a "
                 "measurement about the natural family's reach, not a technosignature; a pair "
                 "residual or flag is a candidate only after every kill in the ledger fails."),
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def slag_run(stage: str, *, config_path=None, results_dir=None, shard: str = "1/1",
             input_csv: str | None = None, n_cal: int | None = None, names=None,
             max_panels: int | None = None, fetch_fn=None, query_fn=None, tap_fn=None) -> dict:
    cfg = load_slag_config(config_path)
    out_dir = Path(results_dir) if results_dir else _repo_root() / cfg["run"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    stages = ["probe", "acquire", "screen", "assess"] if stage == "all" else [stage]
    for st in stages:
        if st == "probe":
            out["probe"] = stage_probe(cfg, out_dir, fetch_fn=fetch_fn, query_fn=query_fn)
        elif st == "acquire":
            out["acquire"] = stage_acquire(cfg, out_dir, fetch_fn=fetch_fn, tap_fn=tap_fn)
        elif st == "screen":
            out["screen"] = stage_screen(cfg, out_dir, shard=shard, input_csv=input_csv,
                                         n_cal=n_cal, names=names, max_panels=max_panels)
        elif st == "assess":
            out["assess"] = stage_assess(cfg, out_dir)
        else:
            raise SystemExit(f"unknown stage {st!r}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="seti slag",
                                 description="SLAG-WD (S51): polluted white dwarfs beyond the natural family")
    ap.add_argument("--stage", default="all", choices=["probe", "acquire", "screen", "assess", "all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--shard", default="1/1", help="i/n for the screen stage")
    ap.add_argument("--input-csv", default=None, help="screen an injected table instead of the acquired one")
    ap.add_argument("--n-cal", type=int, default=None, help="injected natural draws per panel")
    ap.add_argument("--names", default="", help="comma-separated object names to screen (controls first)")
    ap.add_argument("--max-panels", type=int, default=None)
    a = ap.parse_args(argv)
    names = [x.strip() for x in a.names.split(",") if x.strip()] or None
    out = slag_run(a.stage, config_path=a.config, results_dir=a.results_dir, shard=a.shard,
                   input_csv=a.input_csv, n_cal=a.n_cal, names=names, max_panels=a.max_panels)
    if "assess" in out:
        s = out["assess"]
        print(f"[slag] verdict={s['verdict']} funnel={json.dumps(s['funnel'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
