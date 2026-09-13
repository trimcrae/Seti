"""Stage orchestration for ULINE.  Writes ``results/uline/``.

Stages
------
``probe``    what the archives hold: JPL ``catdir.cat`` species matches per
             configured regex, CDMS root pages mined for entries, VizieR
             tables under each source id with their resolved columns, row
             counts and frequency-frame hints, plus any VizieR table whose
             description mentions "unidentified".  Writes ``probe.json``.
``acquire``  fetch every matched ``.cat`` (targets, baseline, contaminants)
             from JPL and CDMS and every source's full line table (identified
             and U) through verified columns.  Writes ``data/lines_<db>.parquet``,
             ``data/entries.json``, ``data/lines_<source>.parquet``,
             ``acquire.json``, ``acquisition_log.json``.
``screen``   per source: vetoes (baseline + contaminants), then per target
             species × T_ex the coincidence count, the LTE tests and the
             shift-trial false-alarm probability.  Writes ``screen.json`` and
             ``coincidences.csv``.
``assess``   the verdict, the species inventory (laboratory vs predicted),
             per-source counts, the best record per species × source.
             Writes ``summary.json`` and ``candidates.csv``.

Verdict vocabulary (``summary.json["verdict"]``)
------------------------------------------------
``NO_DATA_REACHED``     no U-line list was acquired, or no species line list
``NO_PATTERN``          lists and lines were compared; nothing met the pattern test
``PATTERN_CANDIDATE``   ≥ 1 species × source met every test — pending vet
Degradation (a failed archive, a species with no line list, predicted rather
than laboratory frequencies) is a first-class field, never folded into the
verdict string.  None of these is ever written up as a result (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import acquire as A
from .lines import (
    CATDIR_TEMPS,
    Entry,
    count_unparsed_catdir,
    parse_catdir,
    rescale_lgint,
    symmetric_top_lines,
)
from .match import DEFAULT_MATCH, SourceLines, apply_vetoes, best_record, evaluate_species

VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_NONE = "NO_PATTERN"
VERDICT_PATTERN = "PATTERN_CANDIDATE"

DEFAULTS: dict = {
    "archives": {
        "jpl_catdir_url": "https://spec.jpl.nasa.gov/ftp/pub/catalog/catdir.cat",
        "jpl_cat_url": "https://spec.jpl.nasa.gov/ftp/pub/catalog/c{tag:06d}.cat",
        "cdms_roots": ["https://cdms.astro.uni-koeln.de/classic/entries/",
                       "https://cdms.astro.uni-koeln.de/classic/predictions/catalog/"],
        "cdms_cat_url": "https://cdms.astro.uni-koeln.de/classic/predictions/catalog/c{tag:06d}.cat",
        "vizier_tap": A.VIZIER_TAP,
        "prefer_line_list": ["cdms", "jpl", "predicted"],
        "fetch_retries": 3, "fetch_timeout_s": 180, "max_rows": 200000,
        "discover_description_word": "nidentified",
    },
    "species": {"targets": {}, "baseline": {}, "contaminants": {}},
    "predicted": {"error_model": {"base_mhz": 0.5, "rel_with_distortion": 1e-5,
                                  "rel_without_distortion": 1e-4}},
    "sources": {},
    "match": dict(DEFAULT_MATCH),
    "columns": {},
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


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "config").is_dir():
            return p
    return here.parents[3]


def load_uline_config(path: Path | None = None) -> dict:
    """``config/uline.yaml`` over :data:`DEFAULTS`; a missing file degrades."""
    try:
        import yaml
        path = Path(path) if path is not None else _repo_root() / "config" / "uline.yaml"
        if not path.exists():
            return _deep_update(DEFAULTS, {})
        return _deep_update(DEFAULTS, yaml.safe_load(path.read_text()) or {})
    except Exception as exc:                                  # noqa: BLE001
        print(f"[uline] config not loaded ({exc!r}); using defaults")
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
    if isinstance(o, Entry):
        return o.as_dict()
    return str(o)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _column_patterns(conf: dict) -> dict:
    pats = dict(A.DEFAULT_COLUMN_PATTERNS)
    for k, v in (conf.get("columns") or {}).items():
        if k != "uline_values" and v:
            pats[k] = list(v)
    return pats


def _uline_patterns(conf: dict) -> list[str]:
    return list((conf.get("columns") or {}).get("uline_values") or A.DEFAULT_ULINE_PATTERNS)


def _enabled_sources(conf: dict, sources=None) -> dict:
    src = {k: v for k, v in (conf.get("sources") or {}).items() if v.get("enabled", True)}
    if sources:
        want = {s.strip() for s in sources if s.strip()}
        src = {k: v for k, v in src.items() if k in want}
    return src


def _species_group(conf: dict, sp: str) -> str:
    for g in ("targets", "baseline", "contaminants"):
        if sp in (conf.get("species") or {}).get(g, {}):
            return g
    return "unknown"


# ---------------------------------------------------------------------------
# archive inventories (shared by probe and acquire)
# ---------------------------------------------------------------------------
def _jpl_inventory(conf: dict, *, fetch_fn=None, log: A.AcquisitionLog) -> tuple[dict, dict]:
    arc = conf["archives"]
    text = A.fetch_text(arc["jpl_catdir_url"], fetch_fn=fetch_fn,
                        retries=int(arc.get("fetch_retries", 3)),
                        timeout=float(arc.get("fetch_timeout_s", 180)), log=log, stage="jpl_catdir")
    if text is None:
        return {}, {"status": A.STATUS_FAILED, "url": arc["jpl_catdir_url"], "n_entries": 0}
    inv = A.jpl_inventory(text, conf.get("species") or {})
    n_entries = len(parse_catdir(text))
    rep = {"status": A.STATUS_OK if n_entries else A.STATUS_ZERO, "url": arc["jpl_catdir_url"],
           "n_entries": n_entries, "n_unparsed_lines": count_unparsed_catdir(text),
           "species": {sp: [e.as_dict() for e in ents] for sp, ents in inv.items()}}
    return inv, rep


def _cdms_inventory(conf: dict, *, fetch_fn=None, log: A.AcquisitionLog) -> tuple[dict, dict]:
    arc = conf["archives"]
    pages = {}
    for url in arc.get("cdms_roots") or []:
        pages[url] = A.fetch_text(url, fetch_fn=fetch_fn, retries=int(arc.get("fetch_retries", 3)),
                                  timeout=float(arc.get("fetch_timeout_s", 180)), log=log,
                                  stage="cdms_root")
    entries, rep = A.cdms_inventory(pages, conf.get("species") or {})
    inv = rep.pop("species")
    rep["status"] = (A.STATUS_FAILED if all(v is None for v in pages.values()) and pages
                     else (A.STATUS_OK if entries else A.STATUS_ZERO))
    rep["species"] = {sp: [e.as_dict() for e in ents] for sp, ents in inv.items()}
    return inv, rep


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def stage_probe(conf: dict, out: Path, *, fetch_fn=None, query_fn=None, sources=None,
                log: A.AcquisitionLog | None = None) -> dict:
    log = log or A.AcquisitionLog(prefix="uline/probe")
    query_fn = query_fn or A.tap_query
    _, jpl = _jpl_inventory(conf, fetch_fn=fetch_fn, log=log)
    _, cdms = _cdms_inventory(conf, fetch_fn=fetch_fn, log=log)
    cols = _column_patterns(conf)
    vizier = {}
    for name, spec in _enabled_sources(conf, sources).items():
        d = A.discover_line_table(name, spec["vizier_like"], query_fn=query_fn, log=log,
                                  column_patterns=cols)
        vizier[name] = d.as_dict()
    word = conf["archives"].get("discover_description_word")
    discovered = []
    if word:
        try:
            df = A.list_tables_described(str(word), query_fn=query_fn)
            log.record("tables_described", str(word), rows=int(len(df)))
            discovered = df.to_dict(orient="records")
        except Exception as exc:                              # noqa: BLE001
            log.record("tables_described", str(word), error=repr(exc))
    predicted = {sp: {k: v for k, v in blk.items()}
                 for sp, blk in (conf.get("predicted") or {}).items() if sp != "error_model"}
    inventory = {}
    for sp in list((conf.get("species") or {}).get("targets", {})) + \
            list((conf.get("species") or {}).get("baseline", {})) + \
            list((conf.get("species") or {}).get("contaminants", {})):
        inventory[sp] = {"group": _species_group(conf, sp),
                         "jpl": [e["name"] for e in jpl.get("species", {}).get(sp, [])],
                         "cdms": [e["name"] for e in cdms.get("species", {}).get(sp, [])],
                         "predictable": sp in predicted}
    rep = {"stage": "probe", "generated_utc": _now(), "jpl": jpl, "cdms": cdms,
           "vizier": vizier, "discovered_unidentified_tables": discovered,
           "species_inventory": inventory, "acquisition": log.as_dict()}
    _write(out / "probe.json", rep)
    n_ok = sum(1 for v in vizier.values() if v["status"] == A.STATUS_OK)
    print(f"[uline] probe: JPL {jpl['status']} ({jpl['n_entries']} entries), CDMS {cdms['status']} "
          f"({cdms.get('n_entries', 0)} entries), VizieR {n_ok}/{len(vizier)} sources usable")
    return rep


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------
def stage_acquire(conf: dict, out: Path, *, fetch_fn=None, query_fn=None, sources=None,
                  log: A.AcquisitionLog | None = None) -> dict:
    log = log or A.AcquisitionLog()
    query_fn = query_fn or A.tap_query
    arc = conf["archives"]
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    jpl_inv, jpl_rep = _jpl_inventory(conf, fetch_fn=fetch_fn, log=log)
    cdms_inv, cdms_rep = _cdms_inventory(conf, fetch_fn=fetch_fn, log=log)

    entries: dict[str, dict] = {}
    per_species: dict[str, dict] = {}
    frames = []
    for db, inv, tmpl in (("jpl", jpl_inv, arc["jpl_cat_url"]),
                          ("cdms", cdms_inv, arc["cdms_cat_url"])):
        for sp, ents in inv.items():
            table, reps = A.fetch_cats(ents, tmpl, fetch_fn=fetch_fn, log=log,
                                       retries=int(arc.get("fetch_retries", 3)), database=db)
            for e in ents:
                entries[f"{db}:{e.tag}"] = {**e.as_dict(), "species": sp}
            rec = per_species.setdefault(sp, {"group": _species_group(conf, sp)})
            rec[db] = {"entries": reps, "n_lines": int(len(table))}
            if len(table):
                table["species"] = sp
                frames.append(table)
    lines = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(lines):
        lines.to_parquet(data / "lines_catalogue.parquet", index=False)
    _write(data / "entries.json", entries)

    per_source: dict[str, dict] = {}
    cols = _column_patterns(conf)
    for name, spec in _enabled_sources(conf, sources).items():
        d = A.discover_line_table(name, spec["vizier_like"], query_fn=query_fn, log=log,
                                  column_patterns=cols)
        rec = {"table": d.table, "roles": d.roles, "units": d.units, "frame_hint": d.frame_hint,
               "discovery_status": d.status, "n_rows_catalogue": d.n_rows}
        if d.table is None:
            rec.update({"status": d.status, "n_lines": 0, "n_unidentified": 0})
            per_source[name] = rec
            continue
        df = A.fetch_line_table(d, query_fn=query_fn, log=log,
                                max_rows=int(arc.get("max_rows", 200000)),
                                uline_patterns=_uline_patterns(conf))
        if not len(df):
            failed = any(s["stage"] == f"fetch_{name}" and s["status"] == A.STATUS_FAILED
                         for s in log.stages)
            rec.update({"status": A.STATUS_FAILED if failed else A.STATUS_ZERO,
                        "n_lines": 0, "n_unidentified": 0})
            per_source[name] = rec
            continue
        df.to_parquet(data / f"lines_{name}.parquet", index=False)
        rec.update({"status": A.STATUS_OK, "n_lines": int(len(df)),
                    "n_unidentified": int(df["unidentified"].sum()),
                    "freq_scale": df.attrs.get("freq_scale"),
                    "fmin_mhz": float(df["freq_mhz"].min()), "fmax_mhz": float(df["freq_mhz"].max()),
                    "has_intensity": bool(np.isfinite(df["intensity"]).any()),
                    "ident_values_sample": sorted(df["ident"].astype(str).unique().tolist())[:30]})
        per_source[name] = rec

    rep = {"stage": "acquire", "generated_utc": _now(),
           "jpl": {k: v for k, v in jpl_rep.items() if k != "species"},
           "cdms": {k: v for k, v in cdms_rep.items() if k != "species"},
           "species": per_species, "sources": per_source,
           "n_catalogue_lines": int(len(lines)), "acquisition": log.as_dict()}
    _write(out / "acquire.json", rep)
    log.write(out / "acquisition_log.json")
    n_src = sum(1 for r in per_source.values() if r.get("status") == A.STATUS_OK)
    print(f"[uline] acquire: {len(lines)} catalogue lines for {len(per_species)} species; "
          f"{n_src}/{len(per_source)} U-line sources fetched")
    return rep


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------
def build_species_tables(conf: dict, lines: pd.DataFrame, entries: dict, *,
                         fmax_mhz: float) -> dict[str, dict]:
    """{species: {lines, entries, line_source, databases, predicted}} for MATCHING.

    One line list per species, chosen by ``archives.prefer_line_list``; a
    species absent from both databases but present in ``predicted:`` gets the
    symmetric-top predictor.  Contaminant vetoes take the *union* of databases
    (see :func:`contaminant_lines`).
    """
    prefer = list(conf["archives"].get("prefer_line_list") or ["cdms", "jpl", "predicted"])
    pred_conf = conf.get("predicted") or {}
    em = pred_conf.get("error_model") or {}
    out: dict[str, dict] = {}
    species_all = [s for g in ("targets", "baseline", "contaminants")
                   for s in (conf.get("species") or {}).get(g, {})]
    for sp in species_all:
        have = {}
        if len(lines) and "species" in lines:
            sub = lines[lines["species"] == sp]
            for db in ("jpl", "cdms"):
                d = sub[sub["database"] == db]
                if len(d):
                    have[db] = d
        chosen = None
        for db in prefer:
            if db in have:
                chosen = db
                break
            if db == "predicted" and sp in pred_conf and sp != "error_model":
                chosen = "predicted"
                break
        rec = {"group": _species_group(conf, sp), "databases": sorted(have),
               "line_source": chosen, "predicted": chosen == "predicted", "verify": False}
        if chosen in ("jpl", "cdms"):
            d = have[chosen].reset_index(drop=True)
            rec["lines"] = d
            rec["entries"] = {eid: entries[eid] for eid in d["entry_id"].unique() if eid in entries}
            missing = [eid for eid in d["entry_id"].unique() if eid not in entries]
            if missing:
                rec["entries_missing_partition_function"] = missing
                rec["lines"] = d[~d["entry_id"].isin(missing)].reset_index(drop=True)
        elif chosen == "predicted":
            blk = pred_conf[sp]
            has_d = blk.get("DJ_mhz") is not None and blk.get("DJK_mhz") is not None
            rel = float(em.get("rel_with_distortion", 1e-5) if has_d
                        else em.get("rel_without_distortion", 1e-4))
            df, ent = symmetric_top_lines(float(blk["B_mhz"]), float(blk.get("DJ_mhz") or 0.0),
                                          float(blk.get("DJK_mhz") or 0.0),
                                          axial_mhz=blk.get("axial_mhz"),
                                          mu_debye=float(blk.get("mu_debye") or 1.0),
                                          k3_weight=float(blk.get("k3_weight") or 1.0),
                                          fmax_mhz=float(fmax_mhz), temps=CATDIR_TEMPS,
                                          name=f"{sp} (predicted)",
                                          err_base_mhz=float(em.get("base_mhz", 0.5)), err_rel=rel)
            df["entry_id"] = f"predicted:{sp}"
            rec["lines"] = df
            rec["entries"] = {f"predicted:{sp}": {**ent.as_dict(), "species": sp}}
            rec["verify"] = bool(blk.get("verify", True))
            rec["constants"] = {k: blk.get(k) for k in ("B_mhz", "DJ_mhz", "DJK_mhz", "axial_mhz",
                                                        "mu_debye", "k3_weight", "source")}
            rec["axial_known"] = blk.get("axial_mhz") is not None
            rec["distortion_known"] = has_d
        else:
            rec["lines"] = pd.DataFrame()
            rec["entries"] = {}
        rec["n_lines"] = int(len(rec["lines"]))
        out[sp] = rec
    return out


def contaminant_lines(conf: dict, lines: pd.DataFrame, entries: dict, species_tables: dict,
                      *, source: str, tex_k: float, frame: str, v_lsr: float
                      ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{species: (freq at the survey frame, lgint at tex)} for baseline + contaminants (union)."""
    from .match import doppler_sky

    out = {}
    groups = conf.get("species") or {}
    for g in ("baseline", "contaminants"):
        for sp, spec in (groups.get(g) or {}).items():
            only = spec.get("only_sources")
            if only and source not in only:
                continue
            frames = []
            if len(lines) and "species" in lines:
                sub = lines[lines["species"] == sp]
                for eid, grp in sub.groupby("entry_id"):
                    ent = entries.get(eid)
                    if not ent or not ent.get("qlog"):
                        continue
                    lg = rescale_lgint(grp["lgint_300"], grp["elo_cm"], grp["freq_mhz"],
                                       ent["temps"], ent["qlog"], tex_k)
                    frames.append((grp["freq_mhz"].to_numpy(dtype=float), lg))
            if not frames and species_tables.get(sp, {}).get("predicted"):
                st = species_tables[sp]
                ent = next(iter(st["entries"].values()))
                lg = rescale_lgint(st["lines"]["lgint_300"], st["lines"]["elo_cm"],
                                   st["lines"]["freq_mhz"], ent["temps"], ent["qlog"], tex_k)
                frames.append((st["lines"]["freq_mhz"].to_numpy(dtype=float), lg))
            if not frames:
                continue
            f = np.concatenate([x[0] for x in frames])
            lg = np.concatenate([np.asarray(x[1], dtype=float) for x in frames])
            if frame == "sky":
                f = doppler_sky(f, v_lsr)
            out[sp] = (f, lg)
    return out


def source_from_table(name: str, spec: dict, df: pd.DataFrame) -> SourceLines:
    u = df[df["unidentified"].astype(bool)]
    return SourceLines(name=name, u_freq=u["freq_mhz"].to_numpy(dtype=float),
                       u_int=u["intensity"].to_numpy(dtype=float) if "intensity" in u
                       else np.full(len(u), np.nan),
                       u_err=u["freq_err_mhz"].to_numpy(dtype=float) if "freq_err_mhz" in u
                       else np.zeros(len(u)),
                       all_freq=df["freq_mhz"].to_numpy(dtype=float),
                       fwhm_km_s=float(spec.get("fwhm_km_s", 5.0)),
                       v_lsr_km_s=float(spec.get("v_lsr_km_s", 0.0)),
                       frame=str(spec.get("frequency_frame", "rest")),
                       v_unc_km_s=float(spec.get("v_lsr_uncertainty_km_s", 0.0)),
                       fmin_mhz=spec.get("fmin_mhz"), fmax_mhz=spec.get("fmax_mhz"),
                       u_id=u["row"].to_numpy() if "row" in u else None)


def screen_all(conf: dict, lines: pd.DataFrame, entries: dict, source_tables: dict[str, pd.DataFrame],
               *, n_trials: int | None = None, seed: int | None = None,
               species: list[str] | None = None) -> dict:
    """The whole screen on in-memory tables (what the tests drive)."""
    mconf = dict(conf.get("match") or {})
    if n_trials is not None:
        mconf["n_trials"] = int(n_trials)
    if seed is not None:
        mconf["seed"] = int(seed)
    fmax = max([float(df["freq_mhz"].max()) for df in source_tables.values() if len(df)] or [2.0e6])
    tables = build_species_tables(conf, lines, entries, fmax_mhz=fmax * 1.001)
    targets = [s for s in (conf.get("species") or {}).get("targets", {})
               if not species or s in species]
    rng = np.random.default_rng(int(mconf.get("seed", 20260913)))
    per_source: dict = {}
    results: dict = {}
    for name, spec in _enabled_sources(conf).items():
        df = source_tables.get(name)
        if df is None or not len(df):
            per_source[name] = {"status": "NO_TABLE", "n_lines": 0, "n_ulines": 0}
            continue
        src = source_from_table(name, spec, df)
        cont = contaminant_lines(conf, lines, entries, tables, source=name,
                                 tex_k=float(spec.get("contaminant_tex_k", 100.0)),
                                 frame=src.frame, v_lsr=src.v_lsr_km_s)
        src = apply_vetoes(src, cont, dynamic_range_dex=float(mconf.get("veto_dynamic_range_dex", 4.0)))
        veto_counts: dict[str, int] = {}
        for s in src.u_veto_species:
            for sp in (s.split(";") if s else []):
                veto_counts[sp] = veto_counts.get(sp, 0) + 1
        per_source[name] = {"status": "OK", "n_lines": int(len(df)), "n_ulines": src.n_ulines,
                            "n_ulines_clean": src.n_clean,
                            "n_ulines_vetoed": int(src.n_ulines - src.n_clean),
                            "vetoes_by_species": veto_counts,
                            "contaminants_applied": sorted(cont),
                            "fmin_mhz": src.fmin_mhz, "fmax_mhz": src.fmax_mhz,
                            "frame": src.frame, "v_lsr_km_s": src.v_lsr_km_s,
                            "fwhm_km_s": src.fwhm_km_s, "v_unc_km_s": src.v_unc_km_s,
                            "has_intensity": bool(np.isfinite(src.u_int).any())}
        for sp in targets:
            st = tables.get(sp) or {}
            key = f"{sp}|{name}"
            if not st.get("line_source"):
                results[key] = {"species": sp, "source": name, "line_source": None,
                                "status": "NO_LINE_LIST", "records": [], "best": None}
                continue
            recs = evaluate_species(st["lines"], st["entries"], src, species=sp,
                                    line_source=st["line_source"], conf=mconf, rng=rng)
            results[key] = {"species": sp, "source": name, "line_source": st["line_source"],
                            "predicted": bool(st.get("predicted")), "verify": bool(st.get("verify")),
                            "status": "OK", "records": recs, "best": best_record(recs)}
    inventory = {sp: {k: v for k, v in st.items() if k not in ("lines", "entries")}
                 for sp, st in tables.items()}
    return {"stage": "screen", "generated_utc": _now(), "match": mconf, "sources": per_source,
            "species_tables": inventory, "results": results}


def _load_data(out: Path, conf: dict) -> tuple[pd.DataFrame, dict, dict]:
    data = out / "data"
    lines = pd.DataFrame()
    if (data / "lines_catalogue.parquet").exists():
        lines = pd.read_parquet(data / "lines_catalogue.parquet")
    entries = {}
    if (data / "entries.json").exists():
        entries = json.loads((data / "entries.json").read_text())
    tables = {}
    for name in _enabled_sources(conf):
        p = data / f"lines_{name}.parquet"
        if p.exists():
            tables[name] = pd.read_parquet(p)
    return lines, entries, tables


def _coincidence_rows(screen: dict) -> list[dict]:
    rows = []
    for r in (screen.get("results") or {}).values():
        b = r.get("best")
        if not b or b.get("status") != "OK":
            continue
        for i, f in enumerate(b.get("coincident_freq_mhz") or []):
            rows.append({"species": r["species"], "source": r["source"],
                         "line_source": r["line_source"], "tex_k": b["tex_k"],
                         "pred_freq_mhz": f,
                         "uline_freq_mhz": b["coincident_uline_freq_mhz"][i],
                         "sep_mhz": b["coincident_sep_mhz"][i],
                         "members": b["coincident_members"][i],
                         "pred_lgint": b["coincident_pred_lgint"][i],
                         "obs_intensity": b["coincident_obs_int"][i],
                         "pattern": b["pattern"], "p_false": b["p_false"]})
    return rows


def stage_screen(conf: dict, out: Path, *, n_trials: int | None = None, seed: int | None = None,
                 species: list[str] | None = None, data=None) -> dict:
    if data is None:
        lines, entries, tables = _load_data(out, conf)
    else:
        lines, entries, tables = data
    rep = screen_all(conf, lines, entries, tables, n_trials=n_trials, seed=seed, species=species)
    _write(out / "screen.json", rep)
    rows = _coincidence_rows(rep)
    pd.DataFrame(rows, columns=["species", "source", "line_source", "tex_k", "pred_freq_mhz",
                                "uline_freq_mhz", "sep_mhz", "members", "pred_lgint",
                                "obs_intensity", "pattern", "p_false"]
                 ).to_csv(out / "coincidences.csv", index=False)
    n_pat = sum(1 for r in rep["results"].values() if (r.get("best") or {}).get("pattern"))
    print(f"[uline] screen: {len(rep['results'])} species×source evaluated, {n_pat} pattern(s)")
    return rep


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------
CANDIDATE_COLUMNS = ("species", "source", "line_source", "predicted_frequencies", "verify_constants",
                     "tex_k", "n_features_tested", "n_coincident", "n_coincident_vetoed",
                     "spearman_rho", "lte_testable", "lte_pass", "top5_fraction_any",
                     "top5_fraction_uline", "p_false", "p_false_full", "n_trials", "pattern",
                     "coincident_freq_mhz")


def stage_assess(conf: dict, out: Path, *, screen: dict | None = None,
                 acquire_report: dict | None = None) -> dict:
    if screen is None and (out / "screen.json").exists():
        screen = json.loads((out / "screen.json").read_text())
    if acquire_report is None and (out / "acquire.json").exists():
        acquire_report = json.loads((out / "acquire.json").read_text())
    screen = screen or {}
    acq = acquire_report or {}
    sources = screen.get("sources") or {}
    results = screen.get("results") or {}
    tables = screen.get("species_tables") or {}

    n_ulines = sum(int(v.get("n_ulines", 0)) for v in sources.values())
    evaluated = [r for r in results.values() if r.get("status") == "OK"
                 and any(x.get("status") == "OK" for x in r.get("records", []))]
    patterns = [r for r in evaluated if (r.get("best") or {}).get("pattern")]
    if n_ulines == 0 or not evaluated:
        verdict = VERDICT_NO_DATA
    elif patterns:
        verdict = VERDICT_PATTERN
    else:
        verdict = VERDICT_NONE

    degraded = []
    for k, v in (acq.get("sources") or {}).items():
        if v.get("status") != A.STATUS_OK:
            degraded.append(f"source {k}: {v.get('status')}")
    for k, v in sources.items():
        if v.get("status") != "OK":
            degraded.append(f"source {k}: {v.get('status')}")
        elif not v.get("has_intensity"):
            degraded.append(f"source {k}: no intensity column — LTE Spearman test untestable")
    if acq.get("jpl", {}).get("status") not in (None, A.STATUS_OK):
        degraded.append(f"jpl: {acq['jpl'].get('status')}")
    if acq.get("cdms", {}).get("status") not in (None, A.STATUS_OK):
        degraded.append(f"cdms: {acq['cdms'].get('status')}")
    targets = list((conf.get("species") or {}).get("targets", {}))
    inventory = {}
    for sp, st in tables.items():
        inventory[sp] = {"group": st.get("group"), "line_source": st.get("line_source"),
                         "databases": st.get("databases"), "n_lines": st.get("n_lines"),
                         "predicted": bool(st.get("predicted")), "verify": bool(st.get("verify")),
                         "constants": st.get("constants")}
    unsearchable = [sp for sp in targets if not (tables.get(sp) or {}).get("line_source")]
    predicted_used = [sp for sp in targets if (tables.get(sp) or {}).get("predicted")]
    if unsearchable:
        degraded.append("no line list (unsearchable this run): " + ", ".join(unsearchable))
    if predicted_used:
        degraded.append("PREDICTED (not laboratory) frequencies: " + ", ".join(predicted_used))

    rows = []
    for r in results.values():
        b = r.get("best") or {}
        rows.append({"species": r["species"], "source": r["source"], "line_source": r.get("line_source"),
                     "predicted_frequencies": bool(r.get("predicted")),
                     "verify_constants": bool(r.get("verify")),
                     "tex_k": b.get("tex_k"), "n_features_tested": b.get("n_features_tested"),
                     "n_coincident": b.get("n_coincident", 0),
                     "n_coincident_vetoed": b.get("n_coincident_vetoed", 0),
                     "spearman_rho": b.get("spearman_rho"), "lte_testable": b.get("lte_testable"),
                     "lte_pass": b.get("lte_pass"), "top5_fraction_any": b.get("top5_fraction_any"),
                     "top5_fraction_uline": b.get("top5_fraction_uline"),
                     "p_false": b.get("p_false"), "p_false_full": b.get("p_false_full"),
                     "n_trials": b.get("n_trials"), "pattern": bool(b.get("pattern")),
                     "coincident_freq_mhz": ";".join(str(x) for x in (b.get("coincident_freq_mhz") or []))})
    cands = pd.DataFrame(rows, columns=list(CANDIDATE_COLUMNS))
    if len(cands):
        cands = cands.sort_values(["pattern", "n_coincident", "p_false"],
                                  ascending=[False, False, True])
    cands.to_csv(out / "candidates.csv", index=False)

    per_pair = {k: {"line_source": r.get("line_source"), "status": r.get("status"),
                    "best_tex_k": (r.get("best") or {}).get("tex_k"),
                    "n_coincident": (r.get("best") or {}).get("n_coincident", 0),
                    "n_coincident_vetoed": (r.get("best") or {}).get("n_coincident_vetoed", 0),
                    "spearman_rho": (r.get("best") or {}).get("spearman_rho"),
                    "lte_pass": (r.get("best") or {}).get("lte_pass"),
                    "top5_fraction_any": (r.get("best") or {}).get("top5_fraction_any"),
                    "p_false": (r.get("best") or {}).get("p_false"),
                    "p_false_full": (r.get("best") or {}).get("p_false_full"),
                    "pattern": bool((r.get("best") or {}).get("pattern")),
                    "tests": (r.get("best") or {}).get("tests"),
                    "per_tex": [{"tex_k": x.get("tex_k"), "n_coincident": x.get("n_coincident"),
                                 "p_false": x.get("p_false"), "pattern": x.get("pattern")}
                                for x in r.get("records", [])]}
                for k, r in results.items()}
    summary = {
        "verdict": verdict, "generated_utc": _now(),
        "n_pattern_candidates": len(patterns),
        "n_pairs_evaluated": len(evaluated), "n_pairs_total": len(results),
        "species_inventory": inventory,
        "targets_unsearchable": unsearchable,
        "targets_with_predicted_frequencies": predicted_used,
        "sources": {k: {kk: vv for kk, vv in v.items()} for k, v in sources.items()},
        "n_ulines_total": n_ulines,
        "pairs": per_pair,
        "degraded": degraded,
        "match": screen.get("match"),
        "acquisition": {"acquire": acq.get("acquisition")},
        "note": ("PATTERN_CANDIDATE is a coincidence statement pending a line-by-line vet "
                 "(isotopologues, vibrational states, blends, instrumental features) and a "
                 "matched-filter stack; NO_PATTERN is a count, not an abundance limit, and is "
                 "not written up (CLAUDE.md); NO_DATA_REACHED says nothing about the sky"),
    }
    _write(out / "summary.json", summary)
    print(f"[uline] assess: {verdict} — {len(patterns)} pattern(s) in {len(evaluated)} "
          f"species×source pairs, {n_ulines} U-lines")
    return summary


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
STAGES = ("probe", "acquire", "screen", "assess")


def uline_run(conf: dict | None = None, stage: str = "all", *, out_dir=None, fetch_fn=None,
              query_fn=None, sources=None, species=None, n_trials: int | None = None,
              seed: int | None = None) -> dict:
    conf = conf or load_uline_config()
    out = Path(out_dir) if out_dir else Path("results") / "uline"
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage in ("all", "", None) else tuple(s.strip() for s in stage.split(","))
    rep: dict = {}
    for s in stages:
        if s == "probe":
            rep = stage_probe(conf, out, fetch_fn=fetch_fn, query_fn=query_fn, sources=sources)
        elif s == "acquire":
            rep = stage_acquire(conf, out, fetch_fn=fetch_fn, query_fn=query_fn, sources=sources)
        elif s == "screen":
            rep = stage_screen(conf, out, n_trials=n_trials, seed=seed, species=species)
        elif s == "assess":
            rep = stage_assess(conf, out)
        else:
            raise SystemExit(f"unknown stage {s!r}; choose from {STAGES}")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(prog="seti uline",
                                description="ULINE (S54): industrial fluorine molecules in "
                                            "public unidentified-line lists")
    p.add_argument("--stage", default="all", choices=list(STAGES) + ["all"],
                   help="probe|acquire|screen|assess|all")
    p.add_argument("--out-dir", default="results/uline")
    p.add_argument("--sources", default="", help="comma-separated source keys (default all)")
    p.add_argument("--species", default="", help="comma-separated target species (default all)")
    p.add_argument("--n-trials", type=int, default=0, help="shift trials (0 = config)")
    p.add_argument("--seed", type=int, default=0, help="rng seed (0 = config)")
    a = p.parse_args(argv)
    src = [s for s in a.sources.split(",") if s.strip()] or None
    spc = [s for s in a.species.split(",") if s.strip()] or None
    rep = uline_run(load_uline_config(), stage=a.stage, out_dir=a.out_dir, sources=src, species=spc,
                    n_trials=a.n_trials or None, seed=a.seed or None)
    v = rep.get("verdict") if isinstance(rep, dict) else None
    if v:
        print(f"[uline] verdict: {v}")
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())


__all__ = ["DEFAULTS", "STAGES", "VERDICT_NONE", "VERDICT_NO_DATA", "VERDICT_PATTERN",
           "build_species_tables", "contaminant_lines", "load_uline_config", "main",
           "screen_all", "source_from_table", "stage_acquire", "stage_assess", "stage_probe",
           "stage_screen", "uline_run"]
