"""Stage orchestration for FORGE.  Writes ``results/forge/``.

Stages
------
``probe``    schema discovery on every published table in ``config/forge.yaml``
             (the real table, its columns, the resolved roles, the row count),
             plus a reachability check of Simbad TAP and the ESA Gaia archive.
             Writes ``probe.json`` after every table.  Nothing is fetched.
``acquire``  pull every discovered excess table into per-star measurements;
             verify the embedded rows against them; resolve positions and
             spectral types (Simbad); companion context (WDS, Gaia DR3); 2MASS
             and AllWISE for the sample; the Gaia x 2MASS x AllWISE < 30 pc
             population (checkpointed parquet, not committed).  Writes
             ``data/*``, ``acquire.json``, ``acquisition_log.json`` and
             ``forge_excess_verified.csv``.
``screen``   per star: the merged H/K/L/N measurement set, the grey-vs-nano
             likelihood ratio, the variability term, the tier.  The broadband
             leg on the sample and the population.  Writes ``star_table.csv``,
             ``screen.json``, ``broadband_sample.csv``, ``broadband_population.json``.
``assess``   the verdict, the funnel, the coverage, the candidates with the
             systematics each still carries.  Writes ``summary.json``,
             ``candidates.json``.

Verdict vocabulary (``summary.json["verdict"]``)
------------------------------------------------
``NO_DATA_REACHED``                    no archive table produced usable rows and
                                       nothing embedded was verified
``PLANCK_CONSISTENT_CANDIDATES``       >= 1 star at candidate tier
``NO_PLANCK_CONSISTENT_OUTLIER``       stars with an N-band test, none a candidate:
                                       A COUNT, NOT A LIMIT (the number of
                                       N_UNTESTED stars says how much of the
                                       population the test never touched)
``degraded`` lists every source that failed and every veto that could not be
applied.  A null is a reason to change the question (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import acquire as acq
from .physics import (
    BROADBAND_HOT,
    DEFAULT_PHYSICS,
    TIER_CANDIDATE,
    TIER_COMPANION_T,
    TIER_INTEREST,
    TIER_KNOWN_COMPANION,
    TIER_N_UNTESTED,
    TIER_NANO,
    TIER_NO_NIR,
    TIER_UNVERIFIED,
    StarContext,
    assess_star,
    broadband_classify,
    colour_shift,
    f_k_for_shift,
    locus_residual,
    running_locus,
)
from .tables import (
    ASSET_DIR,
    TargetTable,
    embedded_measurements,
    load_asset,
    load_targets,
    merge_measurements,
    teff_from_sptype,
    write_asset_with_verification,
)

VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_CANDIDATES = "PLANCK_CONSISTENT_CANDIDATES"
VERDICT_NONE = "NO_PLANCK_CONSISTENT_OUTLIER"
STAGES = ("probe", "acquire", "screen", "assess")

DEFAULTS: dict = {
    "physics": dict(DEFAULT_PHYSICS),
    "tables": {},
    "positions": {"simbad_tap": acq.SIMBAD_TAP, "chunk": 60},
    "companions": {"wds_table": "B/wds/wds", "wds_radius_arcsec": 3.0, "wds_sep_max_arcsec": 1.5,
                   "wds_dmag_max": 6.0, "gaia_table": "I/355/gaiadr3", "gaia_radius_arcsec": 12.0,
                   "gaia_match_arcsec": 3.0, "ruwe_max": 1.4},
    "broadband": {"sample_tmass_table": "II/246/out", "sample_allwise_table": "II/328/allwise",
                  "cone_arcsec": 5.0,
                  "population": {"parallax_min_mas": 33.333, "parallax_over_error_min": 10.0,
                                 "ruwe_max": 1.4, "g_max": 15.0, "bp_rp_range": [0.3, 3.5],
                                 "dec_band_deg": 30, "top_per_band": 200000},
                  "saturation": {"ks_min": 4.5, "w1_min": 8.0, "w2_min": 7.0, "w3_min": 3.8},
                  "locus_bins": 25, "sigma_min": 3.0, "hot_t_k": 1500.0},
    "probe": {"budget_s": 1500.0},
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


def load_forge_config(path: Path | None = None) -> dict:
    """``config/forge.yaml`` over :data:`DEFAULTS`; a missing file degrades."""
    try:
        import yaml
        if path is None:
            path = Path(__file__).resolve().parents[3] / "config" / "forge.yaml"
        if not Path(path).exists():
            return _deep_update(DEFAULTS, {})
        return _deep_update(DEFAULTS, yaml.safe_load(Path(path).read_text()) or {})
    except Exception as exc:                              # noqa: BLE001
        print(f"[forge] config not loaded ({exc!r}); using defaults")
        return _deep_update(DEFAULTS, {})


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _read(path: Path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text())


def _enabled_tables(conf: dict, names=None) -> dict:
    tabs = {k: v for k, v in (conf.get("tables") or {}).items() if v.get("enabled", True)}
    if names:
        want = {n.strip() for n in names if n.strip()}
        tabs = {k: v for k, v in tabs.items() if k in want}
    return tabs


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def stage_probe(out_dir: Path, conf: dict, *, names=None, query_fn=None, fetch_fn=None,
                simbad_fn=None, gaia_fn=None) -> dict:
    out_dir = Path(out_dir)
    log = acq.AcquisitionLog(prefix="forge/probe")
    budget = float((conf.get("probe") or {}).get("budget_s") or 0) or None
    t0 = time.time()
    tabs = _enabled_tables(conf, names)
    rec: dict = {"stage": "probe", "started": _now(), "tables": {}, "endpoints": {}}
    for name, spec in tabs.items():
        if budget and time.time() - t0 > budget:
            rec["tables"][name] = {"status": acq.STATUS_NOT_ATTEMPTED,
                                   "note": f"probe budget {budget:.0f}s exhausted before {name}"}
            _write(out_dir / "probe.json", rec)
            continue
        role = str(spec.get("role", "excess"))
        disc = acq.discover_table(name, str(spec.get("preferred", "")), role,
                                  tuple(spec.get("keywords", []) or []), query_fn=query_fn,
                                  log=log, overrides=spec.get("columns"), fetch_fn=fetch_fn)
        d = disc.as_dict()
        d["preferred"] = spec.get("preferred")
        d["band"] = spec.get("band")
        rec["tables"][name] = d
        _write(out_dir / "probe.json", rec)
    # endpoints the acquire stage will lean on
    try:
        pos = acq.resolve_positions(["HD 172167", "HD 10700", "HD 102647"], tap_fn=simbad_fn,
                                    url=str(conf["positions"].get("simbad_tap", acq.SIMBAD_TAP)),
                                    log=log)
        rec["endpoints"]["simbad_tap"] = {"status": acq.STATUS_OK if len(pos) else acq.STATUS_ZERO,
                                          "n_resolved": int(len(pos))}
    except Exception as exc:                              # noqa: BLE001
        rec["endpoints"]["simbad_tap"] = {"status": acq.STATUS_FAILED, "error": repr(exc)[:300]}
    try:
        cols = acq.probe_columns("gaiadr1.allwise_original_valid", acq._WISE_WANT,
                                 gaia_fn or acq._gaia_run)
        rec["endpoints"]["gaia_archive"] = {"status": acq.STATUS_OK if cols else acq.STATUS_FAILED,
                                            "allwise_columns": cols}
    except Exception as exc:                              # noqa: BLE001
        rec["endpoints"]["gaia_archive"] = {"status": acq.STATUS_FAILED, "error": repr(exc)[:300]}
    rec["n_tables_usable"] = sum(1 for d in rec["tables"].values() if d.get("status") == acq.STATUS_OK)
    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["finished"] = _now()
    rec["log"] = log.as_dict()
    _write(out_dir / "probe.json", rec)
    return rec


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------
def stage_acquire(out_dir: Path, conf: dict, *, names=None, query_fn=None, fetch_fn=None,
                  simbad_fn=None, cone_fn=None, gaia_fn=None, probe: dict | None = None,
                  skip_population: bool = False, skip_sample_phot: bool = False) -> dict:
    out_dir = Path(out_dir)
    data = out_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    log = acq.AcquisitionLog(prefix="forge/acquire")
    t0 = time.time()
    tabs = _enabled_tables(conf, names)
    probe = probe or _read(out_dir / "probe.json", {}) or {}
    targets = load_targets()
    excess_asset = load_asset("excess")
    rec: dict = {"stage": "acquire", "started": _now(), "tables": {}, "verification": {},
                 "positions": {}, "companions": {}, "sample_photometry": {}, "population": {}}
    archive: dict[str, list] = {}
    reached: dict[str, str] = {}
    for name, spec in tabs.items():
        p = (probe.get("tables") or {}).get(name)
        if not p or p.get("status") != acq.STATUS_OK:
            disc = acq.discover_table(name, str(spec.get("preferred", "")),
                                      str(spec.get("role", "excess")),
                                      tuple(spec.get("keywords", []) or []), query_fn=query_fn,
                                      log=log, overrides=spec.get("columns"), fetch_fn=fetch_fn)
        else:
            disc = acq.DiscoveredTable(name, str(p.get("role", "excess")), p.get("table"),
                                       list(p.get("columns", [])), dict(p.get("roles", {})),
                                       p.get("n_rows"), acq.STATUS_OK, route=p.get("route", ""))
        role = str(spec.get("role", "excess"))
        if role != "excess":
            # polarimetry and model tables are recorded, not folded into the fit
            rec["tables"][name] = {"role": role, "table": disc.table, "status": disc.status,
                                   "roles": disc.roles, "note": "recorded only"}
            reached[name] = disc.status
            continue
        meas, r = acq.fetch_excess_table(disc, spec, targets, query_fn=query_fn, log=log)
        rec["tables"][name] = {**r, "roles": disc.roles, "route": disc.route}
        reached[name] = r["status"]
        for k, ms in meas.items():
            archive.setdefault(k, []).extend(ms)
    rec["n_archive_stars"] = len(archive)
    rec["n_archive_measurements"] = int(sum(len(v) for v in archive.values()))
    _write(data / "archive_measurements.json",
           {k: [m.as_dict() for m in v] for k, v in archive.items()})
    # --- verification of the embedded rows ---------------------------------
    rows, counts = acq.verify_embedded(excess_asset, archive, reached, targets)
    rec["verification"] = counts
    write_asset_with_verification(out_dir / "forge_excess_verified.csv", rows)
    # --- positions ------------------------------------------------------------
    keys = list(targets.rows["key"])
    try:
        pos = acq.resolve_positions(keys, tap_fn=simbad_fn,
                                    url=str(conf["positions"].get("simbad_tap", acq.SIMBAD_TAP)),
                                    chunk=int(conf["positions"].get("chunk", 60)), log=log)
    except Exception as exc:                              # noqa: BLE001
        log.record("simbad_positions", "resolve", error=repr(exc))
        pos = pd.DataFrame(columns=["key", "main_id", "ra", "dec", "sptype", "plx_mas"])
    rec["positions"] = {"n_keys": len(keys), "n_resolved": int(len(pos)),
                        "status": acq.STATUS_OK if len(pos) else acq.STATUS_FAILED}
    pos.to_csv(data / "positions.csv", index=False)
    targets.rows.to_csv(data / "targets.csv", index=False)
    # --- companions --------------------------------------------------------------
    comp = acq.companion_context(pos, conf["companions"], cone_fn=cone_fn, log=log,
                                 embedded=load_asset("companions"))
    rec["companions"] = {"n_stars": len(comp),
                         "n_assessed": sum(1 for c in comp.values() if c["companion_assessed"]),
                         "n_known_companion": sum(1 for c in comp.values() if c["known_companion"])}
    _write(data / "companions.json", comp)
    # --- broadband: sample --------------------------------------------------------
    if not skip_sample_phot:
        sp = acq.sample_photometry(pos, conf["broadband"], cone_fn=cone_fn, log=log)
        sp.to_csv(data / "sample_photometry.csv", index=False)
        rec["sample_photometry"] = {"n_stars": int(len(sp)),
                                    "n_2mass": int(sp["tmass_reached"].sum()) if len(sp) else 0,
                                    "n_allwise": int(sp["wise_reached"].sum()) if len(sp) else 0}
    else:
        rec["sample_photometry"] = {"status": "SKIPPED"}
    # --- broadband: population --------------------------------------------------------
    if not skip_population:
        popdf, prec = acq.population_photometry(conf["broadband"], data / "population",
                                                run_query=gaia_fn, log=log)
        rec["population"] = prec
        if len(popdf):
            popdf.to_parquet(data / "population.parquet", index=False)
    else:
        rec["population"] = {"status": "SKIPPED"}
    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["finished"] = _now()
    _write(out_dir / "acquire.json", rec)
    log.write(out_dir / "acquisition_log.json")
    return rec


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------
def _load_archive(data: Path) -> dict[str, list]:
    from .physics import Measurement
    raw = _read(data / "archive_measurements.json", {}) or {}
    out = {}
    for k, ms in raw.items():
        out[k] = [Measurement(m["band"], float(m["wl_um"]), float(m["value_pct"]),
                              float(m["err_pct"]), m.get("kind", "meas"), m.get("instrument", ""),
                              m.get("epoch", ""), m.get("source", ""), bool(m.get("verified")),
                              m.get("origin", "archive"),
                              m.get("survey", m.get("source", ""))) for m in ms]
    return out


def _embedded_from_verified(out_dir: Path, targets: TargetTable) -> dict[str, list]:
    """The embedded rows with the runner's verification column, when this run
    (or an earlier one on the branch) wrote it; else the raw asset."""
    p = out_dir / "forge_excess_verified.csv"
    df = pd.read_csv(p, dtype=str, keep_default_na=False) if p.exists() else load_asset("excess")
    return embedded_measurements(df, targets)


def build_contexts(targets: TargetTable, positions: pd.DataFrame, companions: dict
                   ) -> dict[str, StarContext]:
    pos = positions.set_index("key") if len(positions) else pd.DataFrame()
    out = {}
    for _, r in targets.rows.iterrows():
        key = str(r["key"])
        teff = targets.teff(key)
        if key in pos.index:
            sp = str(pos.loc[key, "sptype"]) if "sptype" in pos.columns else ""
            if sp and sp not in ("nan", ""):
                teff = teff_from_sptype(sp, teff)
        c = companions.get(key, {})
        out[key] = StarContext(key=key, teff_k=float(teff),
                               known_companion=bool(c.get("known_companion", False)),
                               companion_note=str(c.get("companion_note", "")),
                               companion_assessed=bool(c.get("companion_assessed", False)))
    return out


def screen_stars(measurements: dict[str, list], contexts: dict[str, StarContext],
                 physics: dict) -> list[dict]:
    rows = []
    for key in sorted(measurements):
        ctx = contexts.get(key) or StarContext(key=key)
        r = assess_star(measurements[key], ctx, physics)
        r["companion_note"] = ctx.companion_note
        r["companion_assessed"] = ctx.companion_assessed
        rows.append(r)
    return rows


def _star_row(r: dict) -> dict:
    g = r.get("grey") or {}
    nb = r.get("n_band") or {}
    a = r.get("anchor") or {}
    return {"key": r["key"], "tier": r.get("tier"), "teff_k": r.get("teff_k"),
            "n_meas": r.get("n_meas"),
            "bands": "|".join(sorted({m["band"] for m in r.get("measurements", [])})),
            "nir_anchor_band": a.get("band"), "nir_anchor_pct": a.get("value_pct"),
            "nir_anchor_err_pct": a.get("err_pct"), "nir_anchor_source": a.get("source"),
            "planck_1500k_N_pred_pct": (r.get("planck_1500k_prediction_pct") or {}).get("N"),
            "n_obs_pct": (nb.get("measurement") or {}).get("value_pct"),
            "n_err_pct": (nb.get("measurement") or {}).get("err_pct"),
            "n_source": (nb.get("measurement") or {}).get("source"),
            "n_grey_pred_pct": nb.get("grey_prediction_pct"),
            "n_nano_ceiling_pct": nb.get("nano_ceiling_pct"),
            "n_separation_sigma": nb.get("separation_sigma"),
            "grey_t_k": g.get("t_k"), "grey_f_k_pct": g.get("f_ref_pct"),
            "delta_chi2": r.get("delta_chi2"),
            "delta_chi2_unrestricted": r.get("delta_chi2_unrestricted"),
            "grey_fit_p": r.get("grey_fit_p"), "variability_z": (r.get("variability") or {}).get("z_max"),
            "variable": (r.get("variability") or {}).get("variable"),
            "driving_verified": (r.get("gates") or {}).get("driving_values_verified"),
            "companion_assessed": r.get("companion_assessed"),
            "reason": r.get("reason")}


def broadband_sample(sp: pd.DataFrame, contexts: dict[str, StarContext], conf: dict) -> pd.DataFrame:
    """The broadband classes of the interferometric sample.  These stars are
    mostly brighter than the 2MASS / WISE saturation limits, and the class
    says so rather than reading a saturated colour as an excess."""
    if sp is None or not len(sp):
        return pd.DataFrame()
    sat = conf.get("saturation", {})
    t_hot = float(conf.get("hot_t_k", 1500.0))
    rows = []
    for _, r in sp.iterrows():
        key = str(r["key"])
        ctx = contexts.get(key) or StarContext(key=key)
        ks, w1, w2, w3 = (pd.to_numeric(r.get(c), errors="coerce") for c in ("ks", "w1", "w2", "w3"))
        rec = {"key": key, "ks": ks, "w1": w1, "w2": w2, "w3": w3, "teff_k": ctx.teff_k}
        if not (np.isfinite(ks) and np.isfinite(w1) and np.isfinite(w2)):
            rec["class"] = "INSUFFICIENT_PHOTOMETRY"
            rows.append(rec)
            continue
        saturated = (ks < float(sat.get("ks_min", 4.5)) or w1 < float(sat.get("w1_min", 8.0))
                     or w2 < float(sat.get("w2_min", 7.0)))
        cc = str(r.get("wise_cc", "") or "")
        contaminated = bool(cc and cc.strip("0"))
        # without a population locus the sample uses the photospheric colours
        # of a blackbody at Teff as the baseline: (K-W1, W1-W2) ~ 0 for a
        # bare photosphere in Vega magnitudes to ~0.05 mag
        s = 0.05
        cls = broadband_classify(ctx.teff_k, float(ks - w1), s, float(w1 - w2), s,
                                 float(w2 - w3) if np.isfinite(w3) else None, 0.1,
                                 saturated=saturated, contaminated=contaminated,
                                 sigma_min=float(conf.get("sigma_min", 3.0)), t_k=t_hot)
        rec.update({"class": cls["class"], "k_w1": float(ks - w1), "w1_w2": float(w1 - w2),
                    "f_k_sensitivity_pct": cls.get("f_k_sensitivity_pct"),
                    "f_k_est_pct": cls.get("f_k_est_pct")})
        rows.append(rec)
    return pd.DataFrame(rows)


def broadband_population(pop: pd.DataFrame, conf: dict) -> tuple[dict, pd.DataFrame]:
    """The < 30 pc population: empirical loci of K-W1, W1-W2 and W2-W3 against
    BP-RP, residuals per star, the hot-component class, and the sensitivity
    (the K excess a 3 sigma K-W1 residual corresponds to) per colour bin."""
    rec: dict = {"n_input": int(len(pop)) if pop is not None else 0}
    if pop is None or not len(pop):
        rec["status"] = "NO_POPULATION"
        return rec, pd.DataFrame()
    df = pop.copy()
    df.columns = [str(c).lower() for c in df.columns]
    for c in ("ks", "w1", "w2", "w3", "bp_rp", "parallax", "phot_g_mean_mag", "e_w1", "e_w2",
              "e_w3", "e_ks", "teff_gspphot"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    sat = conf.get("saturation", {})
    lo, hi = conf.get("population", {}).get("bp_rp_range", [0.3, 3.5])
    ok = df["ks"].notna() & df["w1"].notna() & df["w2"].notna() & df["bp_rp"].between(lo, hi)
    # dwarfs only: absolute G above the giant branch (M_G > 2.5 at BP-RP > 0.8 is generous)
    if "parallax" in df and "phot_g_mean_mag" in df:
        mg = df["phot_g_mean_mag"] + 5 * np.log10(df["parallax"] / 100.0)
        df["abs_g"] = mg
        ok &= (mg > 2.0)
    df = df[ok].copy()
    rec["n_dwarfs_with_photometry"] = int(len(df))
    if len(df) < 50:
        rec["status"] = "TOO_FEW"
        return rec, pd.DataFrame()
    df["k_w1"] = df["ks"] - df["w1"]
    df["w1_w2"] = df["w1"] - df["w2"]
    df["w2_w3"] = df["w2"] - df["w3"]
    unsat = ((df["ks"] > float(sat.get("ks_min", 4.5))) & (df["w1"] > float(sat.get("w1_min", 8.0)))
             & (df["w2"] > float(sat.get("w2_min", 7.0))))
    clean = unsat.copy()
    if "wise_cc" in df:
        clean &= df["wise_cc"].astype(str).str.strip("0").eq("")
    if "wise_ext" in df:
        clean &= pd.to_numeric(df["wise_ext"], errors="coerce").fillna(0).eq(0)
    nb = int(conf.get("locus_bins", 25))
    loc1 = running_locus(df.loc[clean, "bp_rp"], df.loc[clean, "k_w1"], nb)
    loc2 = running_locus(df.loc[clean, "bp_rp"], df.loc[clean, "w1_w2"], nb)
    w3ok = clean & df["w3"].notna() & (df["w3"] > float(sat.get("w3_min", 3.8)))
    loc3 = running_locus(df.loc[w3ok, "bp_rp"], df.loc[w3ok, "w2_w3"], nb)
    r1, s1 = locus_residual(df["bp_rp"], df["k_w1"], loc1)
    r2, s2 = locus_residual(df["bp_rp"], df["w1_w2"], loc2)
    r3, s3 = locus_residual(df["bp_rp"], df["w2_w3"], loc3)
    df["r_k_w1"], df["s_k_w1"], df["r_w1_w2"], df["s_w1_w2"] = r1, s1, r2, s2
    df["r_w2_w3"], df["s_w2_w3"] = r3, s3
    t_hot = float(conf.get("hot_t_k", 1500.0))
    smin = float(conf.get("sigma_min", 3.0))
    classes, fest, fsens = [], [], []
    teff_col = df["teff_gspphot"] if "teff_gspphot" in df else pd.Series(np.nan, index=df.index)
    for i, (_, r) in enumerate(df.iterrows()):
        teff = float(teff_col.iloc[i]) if np.isfinite(teff_col.iloc[i]) else _teff_from_bp_rp(r["bp_rp"])
        w3_known = bool(w3ok.iloc[i]) and np.isfinite(r3[i])
        cls = broadband_classify(teff, float(r1[i]), float(s1[i]), float(r2[i]), float(s2[i]),
                                 float(r3[i]) if w3_known else None, float(s3[i]) if w3_known else None,
                                 saturated=not bool(unsat.iloc[i]), contaminated=not bool(clean.iloc[i]) and bool(unsat.iloc[i]),
                                 sigma_min=smin, t_k=t_hot)
        classes.append(cls["class"])
        fest.append(cls.get("f_k_est_pct", float("nan")))
        fsens.append(cls.get("f_k_sensitivity_pct", float("nan")))
    df["class"] = classes
    df["f_k_est_pct"] = fest
    df["f_k_sensitivity_pct"] = fsens
    rec["status"] = "OK"
    rec["n_clean"] = int(clean.sum())
    rec["n_unsaturated"] = int(unsat.sum())
    rec["classes"] = {k: int(v) for k, v in df["class"].value_counts().items()}
    rec["locus"] = {"k_w1": {"bp_rp": loc1["x"].round(3).tolist(), "median": loc1["med"].round(4).tolist(),
                             "sigma": loc1["sig"].round(4).tolist(), "n": loc1["n"].tolist()},
                    "w1_w2": {"bp_rp": loc2["x"].round(3).tolist(), "median": loc2["med"].round(4).tolist(),
                              "sigma": loc2["sig"].round(4).tolist(), "n": loc2["n"].tolist()},
                    "w2_w3": {"bp_rp": loc3["x"].round(3).tolist(), "median": loc3["med"].round(4).tolist(),
                              "sigma": loc3["sig"].round(4).tolist(), "n": loc3["n"].tolist()}}
    # the sensitivity in the interferometric unit, per locus bin
    sens = []
    for x, s in zip(loc1["x"], loc1["sig"], strict=True):
        teff = _teff_from_bp_rp(float(x))
        sens.append({"bp_rp": round(float(x), 3), "teff_k": round(teff),
                     "sigma_k_w1_mag": round(float(s), 4),
                     "f_k_3sigma_pct": round(f_k_for_shift(teff, smin * float(s), t_hot), 2),
                     "unit_shift_1pct": {k: round(v, 4) for k, v in colour_shift(teff, 1.0, t_hot).items()
                                         if k != "excess_pct"}})
    rec["sensitivity"] = sens
    med_sens = float(np.median([s["f_k_3sigma_pct"] for s in sens])) if sens else float("nan")
    rec["median_f_k_3sigma_pct"] = med_sens
    flagged = df[df["class"] == BROADBAND_HOT].copy()
    rec["n_hot_excess"] = int(len(flagged))
    keep = [c for c in ("source_id", "ra", "dec", "parallax", "phot_g_mean_mag", "bp_rp", "abs_g", "ks",
                        "w1", "w2", "w3", "e_w1", "e_w2", "e_w3", "k_w1", "w1_w2", "w2_w3", "r_k_w1",
                        "s_k_w1", "r_w1_w2", "s_w1_w2", "r_w2_w3", "s_w2_w3", "class", "f_k_est_pct",
                        "f_k_sensitivity_pct", "wise_cc", "wise_ext", "wise_qual", "tmass_qual",
                        "non_single_star", "ruwe") if c in flagged.columns]
    return rec, flagged[keep]


def _teff_from_bp_rp(bp_rp: float) -> float:
    """A coarse main-sequence Teff from BP-RP (Pecaut & Mamajek scale, F to M)."""
    x = float(bp_rp)
    pts = [(0.3, 7300), (0.5, 6600), (0.7, 6100), (0.82, 5770), (1.0, 5400), (1.2, 4900),
           (1.5, 4400), (2.0, 3900), (2.5, 3500), (3.0, 3200), (3.5, 3000)]
    xs, ys = zip(*pts, strict=True)
    return float(np.interp(x, xs, ys))


def stage_screen(out_dir: Path, conf: dict) -> dict:
    out_dir = Path(out_dir)
    data = out_dir / "data"
    t0 = time.time()
    targets = load_targets()
    tpath = data / "targets.csv"
    if tpath.exists():
        extra = pd.read_csv(tpath, dtype=str, keep_default_na=False)
        for _, r in extra.iterrows():
            if str(r["key"]) not in set(targets.rows["key"]):
                targets.add(str(r["key"]), **{c: r[c] for c in extra.columns if c != "key"})
    archive = _load_archive(data)
    embedded = _embedded_from_verified(out_dir, targets)
    meas = merge_measurements(embedded, archive)
    ppath = data / "positions.csv"
    pos = pd.read_csv(ppath) if ppath.exists() else pd.DataFrame(columns=["key", "ra", "dec", "sptype"])
    comp = _read(data / "companions.json", {}) or {}
    contexts = build_contexts(targets, pos, comp)
    rows = screen_stars(meas, contexts, conf["physics"])
    table = pd.DataFrame([_star_row(r) for r in rows])
    table.to_csv(out_dir / "star_table.csv", index=False)
    # every sample star with no measurement at all is still a row: untested
    tested = set(table["key"]) if len(table) else set()
    untested = [k for k in targets.rows["key"] if k not in tested]
    # --- broadband --------------------------------------------------------------------
    spp = data / "sample_photometry.csv"
    sp = pd.read_csv(spp) if spp.exists() else pd.DataFrame()
    bs = broadband_sample(sp, contexts, conf["broadband"])
    if len(bs):
        bs.to_csv(out_dir / "broadband_sample.csv", index=False)
    popp = data / "population.parquet"
    pop = pd.read_parquet(popp) if popp.exists() else pd.DataFrame()
    brec, flagged = broadband_population(pop, conf["broadband"])
    if len(flagged):
        flagged.to_csv(out_dir / "broadband_population_flagged.csv", index=False)
    _write(out_dir / "broadband_population.json", brec)
    tiers = {t: int((table["tier"] == t).sum()) for t in sorted(set(table["tier"]))} if len(table) else {}
    rec = {"stage": "screen", "started": _now(), "n_targets": int(len(targets.rows)),
           "n_with_measurements": int(len(table)), "n_without_measurements": len(untested),
           "tiers": tiers,
           "n_verified_driving": int(table["driving_verified"].fillna(False).astype(bool).sum()) if len(table) else 0,
           "broadband_sample_classes": ({k: int(v) for k, v in bs["class"].value_counts().items()}
                                        if len(bs) else {}),
           "broadband_population": {k: v for k, v in brec.items() if k not in ("locus", "sensitivity")},
           "stars": rows, "untested_keys": untested,
           "elapsed_s": round(time.time() - t0, 1), "finished": _now()}
    _write(out_dir / "screen.json", rec)
    return rec


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------
def stage_assess(out_dir: Path, conf: dict) -> dict:
    out_dir = Path(out_dir)
    screen = _read(out_dir / "screen.json", {}) or {}
    acquire = _read(out_dir / "acquire.json", {}) or {}
    probe = _read(out_dir / "probe.json", {}) or {}
    stars = screen.get("stars", [])
    tiers = screen.get("tiers", {})
    tables = acquire.get("tables", {})
    reached = {k: v.get("status") for k, v in tables.items()}
    n_tables_ok = sum(1 for v in reached.values() if v == acq.STATUS_OK)
    ver = acquire.get("verification", {})
    n_verified_rows = int(ver.get("verified", 0))
    n_with_n = sum(1 for s in stars if (s.get("gates") or {}).get("n_measured"))
    n_candidates = int(tiers.get(TIER_CANDIDATE, 0))
    n_interest = int(tiers.get(TIER_INTEREST, 0))
    degraded = []
    for k, v in tables.items():
        if v.get("status") != acq.STATUS_OK:
            degraded.append(f"{k}: {v.get('status')}")
    if (acquire.get("positions") or {}).get("status") != acq.STATUS_OK:
        degraded.append("simbad positions: " + str((acquire.get("positions") or {}).get("status")))
    comp = acquire.get("companions") or {}
    if comp and comp.get("n_assessed", 0) < comp.get("n_stars", 0):
        degraded.append(f"companion veto unapplied on {comp.get('n_stars', 0) - comp.get('n_assessed', 0)} stars")
    if (acquire.get("population") or {}).get("status") != acq.STATUS_OK:
        degraded.append("population photometry: " + str((acquire.get("population") or {}).get("status")))
    if n_tables_ok == 0 and n_verified_rows == 0:
        verdict = VERDICT_NO_DATA
    elif n_candidates:
        verdict = VERDICT_CANDIDATES
    else:
        verdict = VERDICT_NONE
    if degraded:
        verdict += " | DEGRADED (" + "; ".join(degraded)[:600] + ")"
    cands = [s for s in stars if s.get("tier") in (TIER_CANDIDATE, TIER_INTEREST, TIER_UNVERIFIED)]
    cand_out = []
    for s in cands:
        systematics = []
        if not s.get("companion_assessed"):
            systematics.append("companion veto unapplied (no WDS / Gaia answer)")
        systematics.append("faint companion at the 1 % level not excluded by closure phases (Tsishchankava+2025)")
        systematics.append("K-vs-N cross-instrument calibration (floors in config/forge.yaml)")
        if s.get("degenerate_with_cool_nano_grains"):
            systematics.append("a nano-grain population below {:.0f} K reproduces the K/N ratio; "
                               "needs L/M band".format(conf["physics"]["nano_t_range_k"][0]))
        if not (s.get("gates") or {}).get("driving_values_verified"):
            systematics.append("driving values embedded, not archive-verified")
        cand_out.append({"key": s["key"], "tier": s["tier"], "reason": s.get("reason"),
                         "teff_k": s.get("teff_k"), "anchor": s.get("anchor"),
                         "n_band": s.get("n_band"), "grey": s.get("grey"),
                         "delta_chi2": s.get("delta_chi2"),
                         "delta_chi2_by_nano_t_floor": s.get("delta_chi2_by_nano_t_floor"),
                         "variability": s.get("variability"),
                         "systematics_not_excluded": systematics})
    brec = screen.get("broadband_population", {})
    summary = {
        "channel": "forge", "signature": "S47", "generated": _now(),
        "verdict": verdict,
        "funnel": {
            "n_tables_configured": len(conf.get("tables") or {}),
            "n_tables_reached": n_tables_ok,
            "n_targets": screen.get("n_targets", 0),
            "n_with_any_excess_measurement": screen.get("n_with_measurements", 0),
            "n_with_nir_detection": sum(1 for s in stars if (s.get("gates") or {}).get("nir_detected")),
            "n_with_n_band": n_with_n,
            "n_N_UNTESTED": int(tiers.get(TIER_N_UNTESTED, 0)),
            "n_NO_NIR_EXCESS": int(tiers.get(TIER_NO_NIR, 0)),
            "n_nano_preferred": int(tiers.get(TIER_NANO, 0)),
            "n_companion_temperature": int(tiers.get(TIER_COMPANION_T, 0)),
            "n_known_companion": int(tiers.get(TIER_KNOWN_COMPANION, 0)),
            "n_unverified_input": int(tiers.get(TIER_UNVERIFIED, 0)),
            "n_interest": n_interest,
            "n_candidates": n_candidates,
            "n_variable": sum(1 for s in stars if (s.get("variability") or {}).get("variable")),
        },
        "tiers": tiers,
        "verification": ver,
        "coverage": {
            "tables": reached,
            "positions": acquire.get("positions"),
            "companions": comp,
            "sample_photometry": acquire.get("sample_photometry"),
            "population": {k: v for k, v in (acquire.get("population") or {}).items() if k != "bands"},
            "broadband_population": brec,
            "broadband_sample_classes": screen.get("broadband_sample_classes"),
        },
        "n_stars_assessable": n_with_n,
        "n_candidates": n_candidates,
        "n_interest": n_interest,
        "degraded": degraded,
        "probe_n_tables_usable": probe.get("n_tables_usable"),
        "candidates": cand_out,
        "reading": (
            "A COUNT, NOT A LIMIT. The Planck test needs an N-band measurement; every star "
            "without one is N_UNTESTED and says nothing. The broadband leg's sensitivity is "
            "reported in the same unit (the K excess) and is an order of magnitude worse."),
    }
    _write(out_dir / "summary.json", summary)
    _write(out_dir / "candidates.json", {"generated": _now(), "candidates": cand_out})
    return summary


# ---------------------------------------------------------------------------
def forge_run(stage: str = "all", out_dir: Path | None = None, conf: dict | None = None,
              names=None, **kw) -> dict:
    conf = conf or load_forge_config()
    out_dir = Path(out_dir) if out_dir else Path("results") / "forge"
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage == "all" else (stage,)
    result: dict = {}
    probe_kw = {k: kw[k] for k in ("query_fn", "fetch_fn", "simbad_fn", "gaia_fn") if k in kw}
    acq_kw = {k: kw[k] for k in ("query_fn", "fetch_fn", "simbad_fn", "cone_fn", "gaia_fn",
                                  "skip_population", "skip_sample_phot") if k in kw}
    for s in stages:
        if s == "probe":
            result["probe"] = stage_probe(out_dir, conf, names=names, **probe_kw)
        elif s == "acquire":
            result["acquire"] = stage_acquire(out_dir, conf, names=names,
                                              probe=result.get("probe"), **acq_kw)
        elif s == "screen":
            result["screen"] = stage_screen(out_dir, conf)
        elif s == "assess":
            result["assess"] = stage_assess(out_dir, conf)
        else:
            raise SystemExit(f"unknown stage {s!r}; one of {STAGES + ('all',)}")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="seti forge",
                                 description="FORGE (S47): hot exozodis as ~1500 K swarm candidates")
    ap.add_argument("--stage", default="all", choices=STAGES + ("all",))
    ap.add_argument("--out-dir", default="results/forge")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tables", default="", help="comma-separated table keys from config/forge.yaml")
    ap.add_argument("--skip-population", action="store_true",
                    help="skip the Gaia < 30 pc population pull in the acquire stage")
    ap.add_argument("--skip-sample-phot", action="store_true",
                    help="skip the 2MASS / AllWISE cones on the sample")
    args = ap.parse_args(argv)
    conf = load_forge_config(Path(args.config) if args.config else None)
    names = [t for t in args.tables.split(",") if t.strip()] or None
    res = forge_run(args.stage, Path(args.out_dir), conf, names=names,
                    skip_population=args.skip_population, skip_sample_phot=args.skip_sample_phot)
    if "assess" in res:
        s = res["assess"]
        print(json.dumps({"verdict": s["verdict"], "funnel": s["funnel"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ASSET_DIR", "DEFAULTS", "STAGES", "VERDICT_CANDIDATES", "VERDICT_NONE", "VERDICT_NO_DATA",
           "broadband_population", "broadband_sample", "build_contexts", "forge_run",
           "load_forge_config", "main", "screen_stars", "stage_acquire", "stage_assess",
           "stage_probe", "stage_screen"]
