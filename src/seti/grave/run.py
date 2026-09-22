"""Stage orchestration for GRAVE.  Writes ``results/grave/``.

Stages
------
``probe``    learn every source's real shape on the runner: SGP bundle paths and
             field codes (each tested), EarthChem count queries and row keys,
             GEOROC dataset / file listing.  Writes ``probe.json``.
``acquire``  pull the tables with what the probe learned; resolve columns to
             roles at runtime; canonicalise to one ppm column per element.
             Writes ``acquisition.json`` and ``data/<source>_canonical.csv``
             (artifact only).
``screen``   design over the elements present, detection-limit mask, error
             floors on the control population, shuffled null, the mixture fit
             on every sample, the full vet on every above-threshold sample,
             the refined-particulate classes on every sample with a PGE or
             alloy panel.  Writes ``samples.csv``, ``candidates.json``,
             ``refined.json``, ``screen.json``.
``assess``   the age stack (fission candidates, and the impact class as the
             built-in positive control), provenance, ``summary.json``,
             ``REPORT.md``.

Verdict vocabulary (``summary.json["verdict"]``)
------------------------------------------------
``NO_DATA_REACHED``                          no source produced a canonical row
``DEGRADED_SOURCE (...)``                    prefix: an enabled source failed or a role is missing
``NO_FISSION_VECTOR``                        samples screened; 0 survive the vet -- a count, never written up
``FISSION_VECTOR_SINGLE_SECTION``            survivors exist but none recurs across sections at a boundary
``FISSION_VECTOR_STRATIGRAPHIC_CLUSTER``     survivors recur in >= 2 independent sections at one boundary
                                             beyond the population rate -- PENDING provenance / re-analysis
``refined_verdict``: ``NO_REFINED_PARTICULATE`` | ``REFINED_PARTICULATE_CANDIDATES_PENDING_VET``
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import acquire as A
from . import agestack as G
from . import references as R
from . import vectors as V

STAGES = ("probe", "acquire", "screen", "assess")
VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_NONE = "NO_FISSION_VECTOR"
VERDICT_SINGLE = "FISSION_VECTOR_SINGLE_SECTION"
VERDICT_CLUSTER = "FISSION_VECTOR_STRATIGRAPHIC_CLUSTER"
REFINED_NONE = "NO_REFINED_PARTICULATE"
REFINED_CANDS = "REFINED_PARTICULATE_CANDIDATES_PENDING_VET"

META_COLS = ("source", "sample_id", "section", "site_type", "lat", "lon", "age", "min_age", "max_age",
             "lithology", "strat", "basin", "environment", "meta_bin", "height", "reference", "country",
             "analytical_method", "collector", "state_province")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _deep_update(base: dict, extra: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


DEFAULTS: dict = {
    "sources": ["sgp", "earthchem", "georoc"],
    "sgp": {"hosts": ["https://sgp-search.io"], "post_paged_url": "https://sgp-search.io/api/frontend/post-paged",
            "type": "samples", "type_variants": ["samples", "nhhxrf"], "timeout_s": 180,
            "base_show": ["section_name", "coord_lat", "coord_long", "interpreted_age", "max_age", "min_age",
                          "lithology_name", "site_type", "alu", "mo", "u"],
            "anchor_show": ["interpreted_age", "coord_lat"], "show_candidates": [], "max_code_tests": 200,
            "probe_age_window": [0, 4000],
            "page_count": 5000, "max_pages_per_bin": 60, "age_bin_edges_ma": [0, 4000]},
    "earthchem": {"rest_url": "https://portal.earthchem.org/restsearchservice", "timeout_s": 120,
                  "page_size": 50, "max_rows": 5000, "max_rows_per_window": 500, "window_half_width_myr": 5.0,
                  "count_queries": {"keyword_shale": {"keyword": "shale"}}, "distinct_items": []},
    "georoc": {"dataverse_api": "https://data.goettingen-research-online.de/api", "subtree": "digis",
               "search_q": "*", "max_datasets": 500, "dataset_regex": "(?i)rock types|sediment",
               "file_regex": "(?i)sediment|tuff|tephra|ash", "max_files": 12, "max_total_bytes": 150_000_000,
               "timeout_s": 300},
    "detector": {},
    "screen": {"min_element_count": 50, "min_panel_element_count": 5, "unmeasured_floor_dex": 0.30,
               "floor_max_rows": 4000, "null_max_rows": 2000, "max_assess": 5000,
               "min_age_ma": 0.0, "max_age_ma": 4000.0,
               "provenance_flag_regex": r"(?i)core|drill|well|borehole|mine|quarry|tailing"},
    "agestack": {"width_scale": 1.0, "max_span_myr": 20.0, "cluster_p": 0.05, "histogram_bin_myr": 10.0},
}


def load_grave_config(path: Path | None = None) -> dict:
    try:
        import yaml  # noqa: PLC0415
        path = Path(path) if path is not None else repo_root() / "config" / "grave.yaml"
        if not path.exists():
            print(f"[grave] config {path} not found; using defaults")
            return _deep_update(DEFAULTS, {})
        return _deep_update(DEFAULTS, yaml.safe_load(path.read_text()) or {})
    except Exception as exc:                              # noqa: BLE001
        print(f"[grave] config not loaded ({exc!r}); using defaults")
        return _deep_update(DEFAULTS, {})


def detector_config(conf: dict) -> V.GraveConfig:
    d = dict(conf.get("detector") or {})
    for k in ("zr_hf_natural", "nb_ta_natural", "heavy_set", "light_set", "ash_drivers"):
        if k in d and isinstance(d[k], list):
            d[k] = tuple(d[k])
    fields = V.GraveConfig.__dataclass_fields__
    kw = {}
    for k, v in d.items():
        if k not in fields:
            continue
        typ = fields[k].type
        # PyYAML reads "1.0e6" as a string (YAML 1.1 wants 1.0e+6); coerce numerics
        if typ == "float" and isinstance(v, str):
            v = float(v)
        elif typ == "int" and isinstance(v, str):
            v = int(float(v))
        elif typ == "tuple[float, float]":
            v = tuple(float(x) for x in v)
        kw[k] = v
    return V.GraveConfig(**kw)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=2, default=_json_default))


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except Exception:                                     # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def stage_probe(conf: dict, out: Path, *, fetch_fn=None) -> dict:
    fetch = fetch_fn or A.http_fetch
    rep: dict = {"generated_utc": _now(), "sources": {}}
    for src in conf.get("sources", []):
        try:
            if src == "sgp":
                rep["sources"]["sgp"] = A.sgp_probe(conf, fetch=fetch)
            elif src == "earthchem":
                rep["sources"]["earthchem"] = A.earthchem_probe(conf, fetch=fetch)
            elif src == "georoc":
                rep["sources"]["georoc"] = A.georoc_probe(conf, fetch=fetch)
        except Exception as exc:                          # noqa: BLE001
            rep["sources"][src] = {"reached": False, "error": repr(exc)[:300]}
    rep["reached"] = {k: bool(v.get("reached")) for k, v in rep["sources"].items()}
    sgp = rep["sources"].get("sgp") or {}
    # What acquire may ask for, by what the probe PROVED about each code:
    #   accepted        -> the API answered and the column appeared: send it.
    #   silently dropped-> the API answered 200 and ignored it.  Harmless to
    #                      send, and NOT proof the code is invalid: a valid
    #                      column that is null in the two sampled rows looks
    #                      exactly like this.  Send it.
    #   rejected (400)  -> "... is not a valid attribute in this search type",
    #                      and one such code fails the WHOLE request.  Never.
    #   anchor          -> proven by the anchor call itself, which is why the
    #                      code loop skips them; they must still be requested
    #                      or the pull comes back with no age and no latitude.
    acc = set(sgp.get("accepted_codes", {})) | set(sgp.get("silently_dropped", []))
    acc |= set(conf["sgp"].get("anchor_show", []))
    acc -= set(sgp.get("rejected_codes", {}))
    rep["sgp_show_recommended"] = sorted(acc) or sorted(conf["sgp"].get("base_show", []))
    rep["sgp_show_source"] = "probe_accepted_plus_dropped_plus_anchor" if acc else "config_base_show_fallback"
    rep["verdict"] = "REACHED" if any(rep["reached"].values()) else VERDICT_NO_DATA
    _write(out / "probe.json", rep)
    print(f"[grave] probe: {rep['reached']}; SGP codes accepted: {len(sgp.get('accepted_codes', {}))}")
    return rep


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------
def _canon_source(raw: pd.DataFrame, source: str, ledger: dict, out: Path) -> tuple[pd.DataFrame, dict]:
    res = A.resolve_columns(raw.columns, source=source)
    canon = A.canonicalise(raw, res, source=source)
    medians = {}
    for el in res["elements"]:
        if el in canon.columns and canon[el].notna().any():
            medians[el] = float(np.nanmedian(canon[el]))
    resolution = {"meta": res["meta"], "elements": {el: [c[0] for c in v] for el, v in res["elements"].items()},
                  "unresolved": res["unresolved"][:200], "assumed_units": res["assumed_units"],
                  "median_ppm": {k: round(v, 5) for k, v in medians.items()},
                  "census": A.element_census(canon)}
    (out / "data").mkdir(parents=True, exist_ok=True)
    canon.to_csv(out / "data" / f"{source}_canonical.csv", index=False)
    return canon, resolution


def stage_acquire(conf: dict, out: Path, *, fetch_fn=None, max_rows: int | None = None,
                  sources: list[str] | None = None) -> dict:
    fetch = fetch_fn or A.http_fetch
    probe = _read_json(out / "probe.json") or {}
    rep: dict = {"generated_utc": _now(), "stage": "acquire", "sources": {}, "degraded": []}
    srcs = sources or conf.get("sources", [])
    for src in srcs:
        try:
            if src == "sgp":
                psgp = (probe.get("sources") or {}).get("sgp") or {}
                show = probe.get("sgp_show_recommended") or list(conf["sgp"].get("base_show", []))
                kind = None
                tv = psgp.get("type_variants") or {}
                if tv and not (tv.get(conf["sgp"].get("type"), {}) or {}).get("n_rows"):
                    for k, v in tv.items():
                        if v.get("n_rows"):
                            kind = k
                            break
                raw, ledger = A.sgp_acquire(conf, fetch=fetch, show=show, out_dir=out / "data" / "sgp_raw",
                                            kind=kind, max_rows=max_rows)
            elif src == "earthchem":
                pec = (probe.get("sources") or {}).get("earthchem") or {}
                best = pec.get("best_query")
                params = dict(conf["earthchem"]["count_queries"].get(best) or {}) if best else {}
                if not params:
                    params = dict(next(iter(conf["earthchem"]["count_queries"].values())))
                hw = float(conf["earthchem"].get("window_half_width_myr", 5.0))
                windows = [{"key": b["key"], "age_lo": b["age_ma"] - hw, "age_hi": b["age_ma"] + hw}
                           for b in G.boundary_table(conf)]
                raw, ledger = A.earthchem_acquire(conf, fetch=fetch, query_params=params,
                                                  max_rows=int(max_rows or conf["earthchem"].get("max_rows", 5000)),
                                                  windows=windows, out_dir=out / "data",
                                                  url=pec.get("url"))
            elif src == "georoc":
                pg = (probe.get("sources") or {}).get("georoc") or {}
                files = pg.get("files") or []
                if not files:
                    pg = A.georoc_probe(conf, fetch=fetch)
                    files = pg.get("files") or []
                raw, ledger = A.georoc_acquire(conf, fetch=fetch, files=files, out_dir=out / "data")
            else:
                continue
        except Exception as exc:                          # noqa: BLE001
            rep["sources"][src] = {"status": A.STATUS_NO_DATA, "error": repr(exc)[:300]}
            rep["degraded"].append(f"{src}:exception")
            continue
        entry = {"status": ledger.get("status"), "n_rows_raw": int(len(raw)),
                 "ledger": {k: v for k, v in ledger.items() if k != "requests"},
                 "requests": ledger.get("requests", [])[:400],
                 "n_requests": len(ledger.get("requests", []))}
        if len(raw):
            canon, resolution = _canon_source(raw, src, ledger, out)
            entry["resolution"] = resolution
            entry["n_rows"] = int(len(canon))
            missing = [r for r in ("age", "lat", "lon") if r not in resolution["meta"]]
            if missing and src != "georoc":
                rep["degraded"].append(f"{src}:missing_roles:{','.join(missing)}")
        else:
            entry["n_rows"] = 0
            rep["degraded"].append(f"{src}:{ledger.get('status')}")
        rep["sources"][src] = entry
    rep["status"] = A.STATUS_OK if any(v.get("n_rows") for v in rep["sources"].values()) else A.STATUS_NO_DATA
    _write(out / "acquisition.json", rep)
    print("[grave] acquire: " + ", ".join(f"{k}={v.get('n_rows', 0)}" for k, v in rep["sources"].items())
          + f"; degraded={rep['degraded']}")
    return rep


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------
def _load_canonical(out: Path, sources) -> pd.DataFrame:
    frames = []
    for src in sources:
        p = out / "data" / f"{src}_canonical.csv"
        if p.exists():
            try:
                df = pd.read_csv(p, low_memory=False)
                df["source"] = src
                frames.append(df)
            except Exception as exc:                      # noqa: BLE001
                print(f"[grave] could not read {p}: {exc!r}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def stage_screen(conf: dict, out: Path, *, table: pd.DataFrame | None = None) -> dict:
    cfg = detector_config(conf)
    sc = conf.get("screen") or {}
    if table is None:
        table = _load_canonical(out, conf.get("sources", []))
    if table is None or not len(table):
        rep = {"stage": "screen", "generated_utc": _now(), "status": VERDICT_NO_DATA, "n_samples": 0}
        _write(out / "screen.json", rep)
        print("[grave] screen: NO_DATA_REACHED")
        return rep
    df = table.reset_index(drop=True)
    for c in META_COLS:
        if c not in df.columns:
            df[c] = np.nan
    # candidate-source rows (GEOROC is the reference population only)
    df["is_reference_only"] = df["source"].astype(str).eq("georoc")
    # Elements in the design.  Two thresholds, because the elements that matter
    # most are the rarest ones: Ru, Rh, Pd, Te, Ir, Pt are measured on a per-cent
    # minority of analyses, and they are exactly the light-peak / PGE
    # discriminants.  A common element needs ``min_element_count`` measurements
    # (enough to measure its own error floor on the control population); an
    # element of the fission / PGE / alloy panel enters at
    # ``min_panel_element_count`` and carries the wider ``unmeasured_floor_dex``
    # sigma, since its scatter could not be measured.
    n_meas = {e: int(pd.to_numeric(df[e], errors="coerce").notna().sum())
              for e in R.FIT_ELEMENTS if e in df.columns}
    panel = set(R.FISSION_ELEMENTS) | set(R.PGE) | set(R.ALLOY_PANEL)
    min_common = int(sc.get("min_element_count", 50))
    min_panel = int(sc.get("min_panel_element_count", 5))
    elements = [e for e in R.FIT_ELEMENTS
                if n_meas.get(e, 0) >= (min_panel if e in panel else min_common)]
    if not elements:
        rep = {"stage": "screen", "generated_utc": _now(), "status": VERDICT_NO_DATA, "n_samples": int(len(df)),
               "note": "no element column with enough measurements"}
        _write(out / "screen.json", rep)
        return rep
    # every element column present at all, for the kills (redox, Fe-Mn, PGE,
    # alloy) which must see the measurement even when the design cannot use it
    all_elements = [e for e in R.FIT_ELEMENTS if n_meas.get(e, 0) > 0]
    # ``copy=True`` is not optional: under pandas 3's copy-on-write,
    # ``DataFrame.to_numpy()`` hands back a READ-ONLY view, and the next line
    # (masking non-positive values to NaN) raises "assignment destination is
    # read-only".  The sandbox runs pandas 2, the runner installs pandas 3, so
    # this only ever failed on the runner -- at the top of the screen stage,
    # after the whole acquisition had been paid for.
    full_df = df[all_elements].apply(pd.to_numeric, errors="coerce")
    full = full_df.to_numpy(dtype=float, copy=True)
    full[full <= 0] = np.nan
    conc_df = df[elements].apply(pd.to_numeric, errors="coerce")
    dl_mask, dl_ledger = V.detection_limit_mask(conc_df, cfg)
    conc = conc_df.to_numpy(dtype=float, copy=True)
    conc[conc <= 0] = np.nan
    D0 = V.build_design(elements, cfg)
    # boundaries and sections
    bounds = G.boundary_table(conf)
    ag = conf.get("agestack") or {}
    df["boundary"] = [G.assign_boundary(a, lo, hi, bounds, max_span_myr=float(ag.get("max_span_myr", 20.0)),
                                        width_scale=float(ag.get("width_scale", 1.0)))
                      for a, lo, hi in zip(pd.to_numeric(df["age"], errors="coerce"),
                                           pd.to_numeric(df["min_age"], errors="coerce"),
                                           pd.to_numeric(df["max_age"], errors="coerce"), strict=True)]
    df["section_key"] = [G.section_key(r) for r in df[["section", "lat", "lon", "sample_id"]].to_dict("records")]
    control = (df["boundary"] == "") & ~df["is_reference_only"]
    floors = V.error_floors(conc[control.to_numpy()] if control.any() else conc, D0, cfg,
                            max_rows=int(sc.get("floor_max_rows", 4000)))
    D = V.apply_floors(D0, floors, unmeasured_dex=float(sc.get("unmeasured_floor_dex", 0.30)))
    null = V.shuffled_null(conc[control.to_numpy()] if control.any() else conc, D, cfg,
                           max_rows=int(sc.get("null_max_rows", 2000)))
    fit = V.fit_mixture(conc, D, cfg)
    df = pd.concat([df, fit], axis=1)
    # Second, empirical null: the SAME lithology away from every boundary, which
    # is the comparison the brief asks for.  The shuffled null destroys element
    # identity and so does not carry sediments' real correlated deviations; the
    # control population does, at the cost of being contaminated if a residue
    # exists off-boundary too.  The threshold takes whichever is higher, and
    # both are reported so it is visible which one bound it.
    ctrl_lr = pd.to_numeric(df.loc[control, "fission_lr"], errors="coerce").dropna()
    control_null = {"n": int(len(ctrl_lr)), "quantile": cfg.null_quantile,
                    "lr_quantile": round(float(ctrl_lr.quantile(cfg.null_quantile)), 3) if len(ctrl_lr) >= 200 else None,
                    "lr_max": round(float(ctrl_lr.max()), 3) if len(ctrl_lr) else None,
                    "note": "non-boundary samples of the candidate sources; None when fewer than 200"}
    threshold = max(cfg.lr_min, float(null.get("lr_quantile") or 0.0),
                    float(control_null.get("lr_quantile") or 0.0))
    df["threshold_bound_by"] = ("lr_min" if threshold == cfg.lr_min else
                                ("control_population" if threshold == (control_null.get("lr_quantile") or -1)
                                 else "shuffled_null"))
    df["above_threshold"] = (df["fission_lr"] >= threshold) & ~df["is_reference_only"]
    df["ambiguous"] = (df["fission_lr"] >= cfg.ambiguity_margin) & ~df["above_threshold"] & ~df["is_reference_only"]
    df["class"] = np.where(df["n_measured"] < cfg.min_elements, V.INSUFFICIENT, V.NORMAL)
    df.loc[df["ambiguous"], "class"] = V.FISSION_AMBIGUOUS
    df["first_veto"] = ""
    # the full vet on the above-threshold samples, highest LR first
    idx = df.index[df["above_threshold"]].tolist()
    idx = sorted(idx, key=lambda i: -float(df.at[i, "fission_lr"]))[: int(sc.get("max_assess", 5000))]
    vet_counts = dict.fromkeys(V.VETOES, 0)
    cands: list[dict] = []
    prov_re = sc.get("provenance_flag_regex", "")
    print(f"[grave] screen: {len(df)} samples, {len(D.elements)} design elements, "
          f"threshold {threshold:.2f}, {int(df['above_threshold'].sum())} above it, vetting {len(idx)}",
          flush=True)
    for n_done, i in enumerate(idx):
        if n_done and n_done % 100 == 0:
            print(f"[grave]   vetted {n_done}/{len(idx)}", flush=True)
        # the kills see every element the sample carries, not only the design's
        row = {e: float(full[i, k]) for k, e in enumerate(all_elements) if np.isfinite(full[i, k])}
        for extra in ("TOC",):
            if extra in df.columns:
                row[extra] = pd.to_numeric(df.at[i, extra], errors="coerce")
        dl_flags = {e for e in D.elements if e in dl_mask.columns and bool(dl_mask[e].iat[i])}
        res = V.assess_row(row, conc[i], D, cfg, D.sigma, fit=df.loc[i].to_dict(), threshold=threshold,
                           dl_flags=dl_flags)
        df.at[i, "class"] = res["class"]
        df.at[i, "first_veto"] = res["first_veto"]
        for v in res["vetoes"]:
            vet_counts[v] += 1
        rec = {c: _scalar(df.at[i, c]) for c in META_COLS}
        rec.update({"boundary": df.at[i, "boundary"], "section_key": df.at[i, "section_key"],
                    "provenance_flag": bool(prov_re and pd.notna(df.at[i, "site_type"])
                                            and pd.Series([str(df.at[i, "site_type"])]).str.contains(prov_re).iat[0]),
                    "n_measured": int(df.at[i, "n_measured"]), "measured": {e: round(v, 5) for e, v in row.items()
                                                                            if isinstance(v, float) and np.isfinite(v)}})
        rec.update(res)
        cands.append(rec)
    survivors = [c for c in cands if c["class"] == V.FISSION_CANDIDATE]
    df["is_candidate"] = df["class"].eq(V.FISSION_CANDIDATE)
    # refined-particulate classes on every sample with a panel
    refined = _refined_pass(df, full, all_elements, cfg)
    # outputs
    slim = df[[c for c in META_COLS if c in df.columns] + ["boundary", "section_key", "n_measured", "fission_lr",
                                                             "a_fission", "reduced_chi2_natural",
                                                             "reduced_chi2_fission", "dominant_reservoir", "class",
                                                             "first_veto", "pge_class", "alloy_class"]]
    slim.to_csv(out / "samples.csv", index=False)
    _write(out / "candidates.json", {"generated_utc": _now(), "threshold": threshold, "n_assessed": len(cands),
                                     "n_survivors": len(survivors), "survivors": survivors,
                                     "assessed": cands[:400]})
    _write(out / "refined.json", refined)
    rep = {"stage": "screen", "generated_utc": _now(), "status": "OK", "n_samples": int(len(df)),
           "n_candidate_source_samples": int((~df["is_reference_only"]).sum()),
           "n_reference_only": int(df["is_reference_only"].sum()),
           "elements_in_design": D.elements, "n_elements": len(D.elements),
           "elements_measured": {e: n_meas.get(e, 0) for e in all_elements},
           "design": D.to_dict(), "error_floors": floors, "shuffled_null": null,
           "control_population_null": control_null, "threshold": threshold,
           "threshold_bound_by": str(df["threshold_bound_by"].iat[0]) if len(df) else "",
           "detection_limits": dl_ledger,
           "funnel": {"samples": int(len(df)), "with_age": int(pd.to_numeric(df["age"], errors="coerce").notna().sum()),
                      "sufficient_panel": int((df["n_measured"] >= cfg.min_elements).sum()),
                      "fission_lr_positive": int((df["fission_lr"] > 0).sum()),
                      "ambiguous": int(df["ambiguous"].sum()), "above_threshold": int(df["above_threshold"].sum()),
                      "assessed": len(cands), "survivors": len(survivors)},
           "vetoes": vet_counts, "class_counts": df["class"].value_counts().to_dict(),
           "lr_distribution": _quantiles(df.loc[~df["is_reference_only"], "fission_lr"]),
           "boundary_counts": df.loc[df["boundary"] != "", "boundary"].value_counts().to_dict(),
           "refined_counts": refined.get("counts"),
           "fission_discriminants": V.fission_discriminants(cfg),
           "sources": df["source"].value_counts().to_dict()}
    _write(out / "screen.json", rep)
    df.to_pickle(out / "data" / "screened.pkl")
    print(f"[grave] screen: {len(df)} samples, {len(D.elements)} elements, threshold {threshold:.2f}, "
          f"{len(cands)} assessed, {len(survivors)} survivors; refined={refined.get('counts')}")
    return rep


def _scalar(v):
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.integer, int)):
        return int(v)
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def _quantiles(s: pd.Series) -> dict:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return {}
    return {"n": int(len(v)), "p50": round(float(v.quantile(0.5)), 3), "p90": round(float(v.quantile(0.9)), 3),
            "p99": round(float(v.quantile(0.99)), 3), "p999": round(float(v.quantile(0.999)), 3),
            "max": round(float(v.max()), 3)}


def _refined_pass(df: pd.DataFrame, conc: np.ndarray, elements: list[str], cfg: V.GraveConfig) -> dict:
    """PGE and alloy classes over every element column present.

    Deliberately *not* restricted to the mixture design: a chondritic Ir-Os-Ru
    -Pt-Pd panel is reported on a per-cent minority of analyses, so a design
    cut on measurement count would delete the impact positive control and the
    refined-catalyst test together.
    """
    idx = {e: k for k, e in enumerate(elements)}
    has_pge = [e for e in R.PGE if e in idx]
    has_alloy = [e for e in ("Ta", "W") if e in idx]
    pge_cls = np.full(len(df), V.PGE_INSUFFICIENT, dtype=object)
    alloy_cls = np.full(len(df), V.ALLOY_NONE, dtype=object)
    records = []
    for i in range(len(df)):
        row = {e: float(conc[i, idx[e]]) for e in idx if np.isfinite(conc[i, idx[e]])}
        touched = False
        if has_pge and sum(1 for e in has_pge if e in row) >= 2:
            p = V.classify_pge(row, cfg)
            pge_cls[i] = p["pge_class"]
            touched = p["pge_class"] not in (V.PGE_INSUFFICIENT, V.PGE_BACKGROUND)
            pge_rec = p
        else:
            pge_rec = None
        if has_alloy and any(e in row for e in has_alloy):
            a = V.classify_alloy(row, cfg)
            alloy_cls[i] = a["alloy_class"]
            touched = touched or a["alloy_class"] not in (V.ALLOY_NONE,)
            alloy_rec = a
        else:
            alloy_rec = None
        if touched and len(records) < 2000:
            rec = {c: _scalar(df.at[i, c]) for c in META_COLS}
            rec.update({"boundary": df.at[i, "boundary"], "section_key": df.at[i, "section_key"],
                        "pge": pge_rec, "alloy": alloy_rec})
            records.append(rec)
    df["pge_class"] = pge_cls
    df["alloy_class"] = alloy_cls
    counts = {"pge": pd.Series(pge_cls).value_counts().to_dict(), "alloy": pd.Series(alloy_cls).value_counts().to_dict()}
    return {"generated_utc": _now(), "pge_elements_present": has_pge, "alloy_elements_present": has_alloy,
            "counts": counts, "records": records}


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------
def stage_assess(conf: dict, out: Path) -> dict:
    acq = _read_json(out / "acquisition.json") or {}
    scr = _read_json(out / "screen.json") or {}
    cands = _read_json(out / "candidates.json") or {}
    refined = _read_json(out / "refined.json") or {}
    pkl = out / "data" / "screened.pkl"
    df = pd.read_pickle(pkl) if pkl.exists() else pd.DataFrame()
    degraded = list(acq.get("degraded") or [])
    if scr.get("status") != "OK" or not len(df):
        summary = {"verdict": VERDICT_NO_DATA, "generated_utc": _now(), "n_samples": 0,
                   "funnel": {}, "acquisition": {k: {kk: vv for kk, vv in v.items() if kk not in ("requests",)}
                                                 for k, v in (acq.get("sources") or {}).items()},
                   "degraded": degraded,
                   "note": ("no canonical sedimentary chemistry was reached, so nothing about a fission or "
                            "refined-particulate residue was measured; an ACCESS statement, not a null, never "
                            "written up.  The probe ledger names every endpoint tried.")}
        _write(out / "summary.json", summary)
        (out / "REPORT.md").write_text(_report(summary, scr, cands, refined, None))
        print(f"[grave] assess: {VERDICT_NO_DATA}")
        return summary
    bounds = G.boundary_table(conf)
    cluster_p = float((conf.get("agestack") or {}).get("cluster_p", 0.05))
    main = df[~df["is_reference_only"]].copy()
    stack = G.age_stack(main, bounds, candidate_col="is_candidate", cluster_p=cluster_p)
    main["is_impact"] = main["pge_class"].eq(V.PGE_IMPACT)
    impact_stack = G.age_stack(main, bounds, candidate_col="is_impact", cluster_p=cluster_p) \
        if main["is_impact"].any() else None
    main["is_refined"] = main["pge_class"].isin([V.PGE_REFINED, V.PGE_FISSION_LIKE]) | \
        main["alloy_class"].isin([V.ALLOY_REFINED_TA, V.ALLOY_REFINED_W])
    refined_stack = G.age_stack(main, bounds, candidate_col="is_refined", cluster_p=cluster_p) \
        if main["is_refined"].any() else None
    hist = G.age_histogram(main, bin_myr=float((conf.get("agestack") or {}).get("histogram_bin_myr", 10.0)))
    clusters = [k for k, v in stack["boundaries"].items() if v["status"] == "STRATIGRAPHIC_CLUSTER"]
    n_surv = int(cands.get("n_survivors", 0))
    if n_surv == 0:
        verdict = VERDICT_NONE
    elif clusters:
        verdict = VERDICT_CLUSTER
    else:
        verdict = VERDICT_SINGLE
    if degraded:
        verdict = f"DEGRADED_SOURCE ({'; '.join(degraded)}); {verdict}"
    n_ref = int(main["is_refined"].sum())
    refined_verdict = REFINED_CANDS if n_ref else REFINED_NONE
    survivors = cands.get("survivors") or []
    summary = {
        "verdict": verdict, "refined_verdict": refined_verdict, "generated_utc": _now(),
        "n_samples": int(scr.get("n_samples", 0)), "n_candidate_source_samples": int(len(main)),
        "funnel": scr.get("funnel"), "vetoes": scr.get("vetoes"), "class_counts": scr.get("class_counts"),
        "threshold": scr.get("threshold"), "shuffled_null": scr.get("shuffled_null"),
        "control_population_null": scr.get("control_population_null"),
        "threshold_bound_by": scr.get("threshold_bound_by"),
        "lr_distribution": scr.get("lr_distribution"), "elements_in_design": scr.get("elements_in_design"),
        "error_floors": scr.get("error_floors"), "detection_limits": scr.get("detection_limits"),
        "age_stack": stack, "impact_positive_control": impact_stack, "refined_stack": refined_stack,
        "stratigraphic_clusters": clusters, "age_histogram": hist,
        "n_survivors": n_surv, "survivors": survivors[:100],
        "refined_counts": refined.get("counts"), "n_refined": n_ref,
        "refined_records": (refined.get("records") or [])[:100],
        "acquisition": {k: {kk: vv for kk, vv in v.items() if kk not in ("requests",)}
                        for k, v in (acq.get("sources") or {}).items()},
        "degraded": degraded, "sources": scr.get("sources"),
        "reservoirs": R.RESERVOIR_SOURCES, "paas_from_ucc": list(R.PAAS_FROM_UCC),
        "fission_discriminants": scr.get("fission_discriminants"),
        "boundaries": bounds,
        "note": ("A FISSION_CANDIDATE is a sample whose chemistry the natural reservoir family cannot build "
                 "without a fission-product component, surviving leave-one-out, peak coherence and the named "
                 "kills; it becomes a boundary-level claim only as a STRATIGRAPHIC_CLUSTER (>= 2 independent "
                 "sections at one boundary beyond the population rate).  NO_FISSION_VECTOR is a count, not "
                 "an occurrence limit, and is not written up (CLAUDE.md).  Every survivor is PENDING "
                 "provenance (core vs outcrop, laboratory, reference) and re-analysis of the sample."),
    }
    _write(out / "summary.json", summary)
    (out / "REPORT.md").write_text(_report(summary, scr, cands, refined, stack))
    print(f"[grave] assess: {verdict} | {refined_verdict}; survivors={n_surv}, clusters={clusters}, refined={n_ref}")
    return summary


def _report(s: dict, scr: dict, cands: dict, refined: dict, stack: dict | None) -> str:
    L = [f"# GRAVE report — {s['generated_utc']}", "",
         f"**Verdict:** `{s['verdict']}`  ", f"**Refined-particulate:** `{s.get('refined_verdict', '')}`", ""]
    if s["verdict"] == VERDICT_NO_DATA:
        L += [s.get("note", ""), "", "## Acquisition", ""]
        for k, v in (s.get("acquisition") or {}).items():
            L.append(f"- **{k}**: {v.get('status')} ({v.get('n_rows', 0)} rows) {v.get('error', '')}")
        return "\n".join(L) + "\n"
    f = s.get("funnel") or {}
    L += ["## Funnel", "", "| stage | n |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in f.items()]
    L += ["", f"Threshold ln LR = {s.get('threshold'):.2f} (floor {scr.get('design', {}) and ''}"
          f"shuffled-null q{(s.get('shuffled_null') or {}).get('quantile')} = "
          f"{(s.get('shuffled_null') or {}).get('lr_quantile')}).", ""]
    L += ["## Vetoes", "", "| veto | n |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in (s.get("vetoes") or {}).items()]
    L += ["", "## Age stack (fission candidates)", "",
          "| boundary | age | samples | sections | cand. | cand. sections | expected | p_raw | p_Holm | status |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for b in ((s.get("age_stack") or {}).get("boundaries") or {}).values():
        L.append(f"| {b['name']} | {b['age_ma']} | {b['n_samples']} | {b['n_sections']} | {b['n_candidates']} | "
                 f"{b['n_candidate_sections']} | {b['expected_candidate_sections']} | "
                 f"{b.get('p_hypergeom')} | {b.get('p_family')} | {b['status']} |")
    _st = s.get("age_stack") or {}
    L += ["", f"Promotion reads the Holm-corrected `p_family` over the {_st.get('n_boundaries_tested')} "
              f"testable windows, at a family-wise threshold of {_st.get('cluster_p')}."]
    ic = s.get("impact_positive_control")
    L += ["", "## Impact class as positive control (chondritic PGE, Ir-anchored)", ""]
    if ic:
        for b in ic["boundaries"].values():
            if b["n_candidates"]:
                L.append(f"- {b['name']}: {b['n_candidates']} impact-class samples in {b['n_candidate_sections']} "
                         f"sections, p_raw = {b.get('p_hypergeom')}, p_Holm = {b.get('p_family')} "
                         f"({b['status']})")
    else:
        L.append("No sample carries a PGE panel that classes as impact — the Ir positive control could not run "
                 "on this corpus (state which elements were present: "
                 f"{(refined or {}).get('pge_elements_present')}).")
    L += ["", "## Survivors", ""]
    for c in (s.get("survivors") or [])[:50]:
        L.append(f"- `{c.get('source')}:{c.get('sample_id')}` age {c.get('age')} Ma, {c.get('section')} "
                 f"({c.get('lithology')}, {c.get('site_type')}), boundary `{c.get('boundary') or '-'}`, "
                 f"ln LR {c.get('fission_lr')} (LOO min {c.get('lr_loo_min')} on {c.get('lr_loo_driver')}), "
                 f"a_f {c.get('a_fission')}, heavy {c.get('peak', {}).get('heavy_up')} light {c.get('peak', {}).get('light_up')}, "
                 f"ref {c.get('reference')}, provenance_flag {c.get('provenance_flag')}")
    if not s.get("survivors"):
        L.append("none")
    L += ["", "## Refined-particulate classes", "", f"{s.get('refined_counts')}", ""]
    for r in (s.get("refined_records") or [])[:30]:
        L.append(f"- `{r.get('source')}:{r.get('sample_id')}` age {r.get('age')} Ma {r.get('section')}: "
                 f"pge {(r.get('pge') or {}).get('pge_class')} {(r.get('pge') or {}).get('pge_ratios_vs_ci')} "
                 f"alloy {(r.get('alloy') or {}).get('alloy_class')}")
    L += ["", "## Degradation", "", f"{s.get('degraded')}", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def grave_run(conf: dict | None = None, stage: str = "all", *, out_dir=None, fetch_fn=None,
              max_rows: int | None = None, sources: list[str] | None = None) -> dict:
    conf = conf if conf is not None else load_grave_config()
    if sources:
        conf = dict(conf)
        conf["sources"] = list(sources)
    out = Path(out_dir) if out_dir else repo_root() / "results" / "grave"
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage in ("all", "", None) else tuple(s.strip() for s in stage.split(","))
    rep: dict = {}
    for s in stages:
        if s == "probe":
            rep = stage_probe(conf, out, fetch_fn=fetch_fn)
        elif s == "acquire":
            rep = stage_acquire(conf, out, fetch_fn=fetch_fn, max_rows=max_rows)
        elif s == "screen":
            rep = stage_screen(conf, out)
        elif s == "assess":
            rep = stage_assess(conf, out)
        else:
            raise SystemExit(f"unknown stage {s!r}; choose from {STAGES}")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(prog="seti grave",
                                description="GRAVE (S56): a technological extinction in Earth's sedimentary record")
    p.add_argument("--stage", default="all", help="probe|acquire|screen|assess|all or a comma list")
    p.add_argument("--sources", default="", help="comma list of sources (default: config)")
    p.add_argument("--max-rows", type=int, default=0, help="cap rows per source (0 = config)")
    p.add_argument("--out-dir", default="", help="results directory (default results/grave)")
    p.add_argument("--config", default="", help="alternative config yaml")
    a = p.parse_args(argv)
    conf = load_grave_config(Path(a.config) if a.config else None)
    rep = grave_run(conf, stage=a.stage, out_dir=a.out_dir or None, max_rows=a.max_rows or None,
                    sources=[s.strip() for s in a.sources.split(",") if s.strip()] or None)
    v = rep.get("verdict") if isinstance(rep, dict) else None
    if v:
        print(f"[grave] verdict: {v}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())


__all__ = ["DEFAULTS", "STAGES", "VERDICT_CLUSTER", "VERDICT_NONE", "VERDICT_NO_DATA", "VERDICT_SINGLE",
           "detector_config", "grave_run", "load_grave_config", "main", "stage_acquire", "stage_assess",
           "stage_probe", "stage_screen"]
