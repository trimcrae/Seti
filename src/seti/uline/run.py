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
    rescale_lgint,
    symmetric_top_lines,
)
from .match import DEFAULT_MATCH, SourceLines, apply_vetoes, best_record, evaluate_species
from .rotorpred import (
    literature_sources,
    load_rotor_assets,
    predict_species_lines,
    rotor_error_model,
    searchability,
    summarise_assets,
    validation_targets,
)

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
                                  "rel_without_distortion": 1e-4,
                                  "distortion_omega_cm": 300.0}},
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
        return {}, {"status": A.STATUS_FAILED, "url": arc["jpl_catdir_url"], "n_entries": 0,
                    "n_unparsed_lines": 0, "unparsed_sample": [], "matched_names": {},
                    "near_miss_names": {}}
    inv, rep = A.jpl_inventory(text, conf.get("species") or {})
    rep["status"] = A.STATUS_OK if rep["n_entries"] else A.STATUS_ZERO
    rep["url"] = arc["jpl_catdir_url"]
    rep["species"] = {sp: [e.as_dict() for e in ents] for sp, ents in inv.items()}
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


def _tap_query_fn(conf: dict, query_fn=None):
    """The configured TAP endpoint and retry budget (tests inject ``query_fn``)."""
    if query_fn is not None:
        return query_fn
    arc = conf.get("archives") or {}
    url = str(arc.get("vizier_tap") or A.VIZIER_TAP)
    retries = int(arc.get("tap_retries", 5))
    return lambda adql: A.tap_query(adql, url=url, retries=retries)


def _discover(conf: dict, name: str, spec: dict, *, query_fn, log, cols) -> A.LineTableDiscovery:
    return A.discover_line_table(
        name, spec["vizier_like"], query_fn=query_fn, log=log, column_patterns=cols,
        fallback_terms_all=spec.get("fallback_description_all")
        or [conf["archives"].get("discover_description_word") or "nidentified"],
        fallback_terms_any=spec.get("fallback_description_any") or [])


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def stage_probe(conf: dict, out: Path, *, fetch_fn=None, query_fn=None, sources=None,
                log: A.AcquisitionLog | None = None) -> dict:
    log = log or A.AcquisitionLog(prefix="uline/probe")
    query_fn = _tap_query_fn(conf, query_fn)
    _, jpl = _jpl_inventory(conf, fetch_fn=fetch_fn, log=log)
    _, cdms = _cdms_inventory(conf, fetch_fn=fetch_fn, log=log)
    cols = _column_patterns(conf)
    vizier = {}
    for name, spec in _enabled_sources(conf, sources).items():
        d = _discover(conf, name, spec, query_fn=query_fn, log=log, cols=cols)
        vizier[name] = d.as_dict()
    word = conf["archives"].get("discover_description_word")
    discovered = []
    if word:
        adql = A.tables_described_adql([str(word)], [])
        try:
            df = A.list_tables_described(str(word), query_fn=query_fn)
            log.record("tables_described", str(word), rows=int(len(df)), extra={"adql": adql})
            discovered = df.to_dict(orient="records")
        except Exception as exc:                              # noqa: BLE001
            log.record("tables_described", str(word), error=repr(exc), extra={"adql": adql})
    predicted = {sp: {k: v for k, v in blk.items()}
                 for sp, blk in (conf.get("predicted") or {}).items() if sp != "error_model"}
    assets = load_rotor_assets()
    rotor_summary = summarise_assets(assets)
    inventory = {}
    for sp in list((conf.get("species") or {}).get("targets", {})) + \
            list((conf.get("species") or {}).get("baseline", {})) + \
            list((conf.get("species") or {}).get("contaminants", {})):
        inventory[sp] = {"group": _species_group(conf, sp),
                         "jpl": [e["name"] for e in jpl.get("species", {}).get(sp, [])],
                         "cdms": [e["name"] for e in cdms.get("species", {}).get(sp, [])],
                         "jpl_near_miss_names": (jpl.get("near_miss_names") or {}).get(sp, []),
                         "cdms_near_miss_names": (cdms.get("near_miss_names") or {}).get(sp, []),
                         "predictable": sp in predicted or sp in rotor_summary,
                         "rotor_constants": rotor_summary.get(sp)}
    rep = {"stage": "probe", "generated_utc": _now(), "jpl": jpl, "cdms": cdms,
           "vizier": vizier, "discovered_unidentified_tables": discovered,
           "species_inventory": inventory, "rotor_assets": rotor_summary,
           "acquisition": log.as_dict()}
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
    query_fn = _tap_query_fn(conf, query_fn)
    arc = conf["archives"]
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    jpl_inv, jpl_rep = _jpl_inventory(conf, fetch_fn=fetch_fn, log=log)
    cdms_inv, cdms_rep = _cdms_inventory(conf, fetch_fn=fetch_fn, log=log)

    entries: dict[str, dict] = {}
    per_species: dict[str, dict] = {}
    frames = []
    for db, inv, tmpl, dbrep in (("jpl", jpl_inv, arc["jpl_cat_url"], jpl_rep),
                                 ("cdms", cdms_inv, arc["cdms_cat_url"], cdms_rep)):
        for sp, ents in inv.items():
            table, reps = A.fetch_cats(ents, tmpl, fetch_fn=fetch_fn, log=log,
                                       retries=int(arc.get("fetch_retries", 3)), database=db)
            for e in ents:
                entries[f"{db}:{e.tag}"] = {**e.as_dict(), "species": sp}
            rec = per_species.setdefault(sp, {"group": _species_group(conf, sp)})
            # near_miss_names: catalogue entries whose normalised name merely
            # CONTAINS the formula.  When `entries` is empty this is what says
            # how the archive actually spells the species.
            rec[db] = {"entries": reps, "n_lines": int(len(table)),
                       "matched_names": (dbrep.get("matched_names") or {}).get(sp, []),
                       "near_miss_names": (dbrep.get("near_miss_names") or {}).get(sp, [])}
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
        d = _discover(conf, name, spec, query_fn=query_fn, log=log, cols=cols)
        rec = {"table": d.table, "roles": d.roles, "units": d.units, "frame_hint": d.frame_hint,
               "discovery_status": d.status, "n_rows_catalogue": d.n_rows,
               "queries": d.queries, "errors": d.errors, "fallback": d.fallback,
               "scoreboard": d.scoreboard}
        df = pd.DataFrame()
        if d.table is not None:
            df = A.fetch_line_table(d, query_fn=query_fn, log=log,
                                    max_rows=int(arc.get("max_rows", 200000)),
                                    uline_patterns=_uline_patterns(conf))
        if not len(df) and spec.get("text_routes"):
            # VizieR has no usable table for this source.  Some published line
            # lists were never deposited at CDS at all (Crockett+2014 is
            # measured absent from J/ApJ/787/112 on three routes), and live only
            # as machine-readable tables attached to the article.  Same format,
            # different door.
            from .textlists import fetch_text_line_table
            tr = fetch_text_line_table(dict(spec["text_routes"]), fetch_fn=fetch_fn,
                                       column_patterns=cols,
                                       uline_patterns=_uline_patterns(conf), log=log,
                                       timeout=float(arc.get("fetch_timeout_s", 180)),
                                       retries=int(arc.get("fetch_retries", 3)))
            df = tr.pop("table", pd.DataFrame())
            rec["text_route"] = tr
            if len(df):
                rec["route"] = "text"
        if not len(df):
            failed = any(s["stage"] == f"fetch_{name}" and s["status"] == A.STATUS_FAILED
                         for s in log.stages)
            rec.update({"status": (d.status if d.table is None
                                   else (A.STATUS_FAILED if failed else A.STATUS_ZERO)),
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

    drop = ("species", "matched_names", "near_miss_names")     # reported per species instead
    rep = {"stage": "acquire", "generated_utc": _now(),
           "jpl": {k: v for k, v in jpl_rep.items() if k not in drop},
           "cdms": {k: v for k, v in cdms_rep.items() if k not in drop},
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
                         fmax_mhz: float, fmin_mhz: float = 0.0,
                         assets: dict | None = None) -> dict[str, dict]:
    """{species: {lines, entries, line_source, databases, predicted}} for MATCHING.

    One line list per species, chosen by ``archives.prefer_line_list``.  A
    species absent from both databases is predicted, preferring the **rotor
    assets** (``src/seti/data_assets/rotor_constants.yaml``: a full asymmetric
    top, every isotopologue, the published constants with their provenance) over
    the old single-constant symmetric-top block in ``config/uline.yaml``, which
    remains only as a fallback for a species the assets do not carry.
    Contaminant vetoes take the *union* of databases
    (see :func:`contaminant_lines`).
    """
    prefer = list(conf["archives"].get("prefer_line_list") or ["cdms", "jpl", "predicted"])
    pred_conf = conf.get("predicted") or {}
    em = pred_conf.get("error_model") or {}
    assets = load_rotor_assets() if assets is None else assets
    rotor_species = set(assets.get("species") or {})
    emr = rotor_error_model(conf)
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
            if db == "predicted" and sp != "error_model" and (sp in rotor_species
                                                              or sp in pred_conf):
                chosen = "rotor" if sp in rotor_species else "predicted"
                break
        rec = {"group": _species_group(conf, sp), "databases": sorted(have),
               "line_source": chosen, "predicted": chosen in ("predicted", "rotor"),
               "verify": False}
        if chosen in ("jpl", "cdms"):
            d = have[chosen].reset_index(drop=True)
            rec["lines"] = d
            rec["entries"] = {eid: entries[eid] for eid in d["entry_id"].unique() if eid in entries}
            missing = [eid for eid in d["entry_id"].unique() if eid not in entries]
            if missing:
                rec["entries_missing_partition_function"] = missing
                rec["lines"] = d[~d["entry_id"].isin(missing)].reset_index(drop=True)
        elif chosen == "rotor":
            df, ents, meta = predict_species_lines(
                assets, sp, fmin_mhz=float(fmin_mhz), fmax_mhz=float(fmax_mhz),
                temps=CATDIR_TEMPS, err_base_mhz=float(emr["base_mhz"]),
                err_rel=float(emr["rel"]),
                distortion_omega_cm=float(emr["distortion_omega_cm"]))
            rec["lines"] = df
            rec["entries"] = ents
            # EVERY rotor block is `verify`: the constants are reconstructed
            # from the literature, not read off a catalogue.  A candidate that
            # rests on these lines alone is flagged, never promoted.
            rec["verify"] = True
            rec["rotor"] = {k: v for k, v in meta.items() if k != "isotopologues"}
            rec["rotor_isotopologues"] = meta.get("isotopologues")
            rec["constants"] = {"assets": "src/seti/data_assets/rotor_constants.yaml",
                                "quality": meta.get("quality"),
                                "quartic_known": meta.get("quartic_known"),
                                "n_isotopologues": meta.get("n_isotopologues")}
            rec["distortion_known"] = bool(meta.get("quartic_known"))
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
                sl = st["lines"]
                # A rotor prediction carries one entry (and one partition
                # function) PER ISOTOPOLOGUE, so rescale each group with its own.
                groups = (sl.groupby("entry_id") if "entry_id" in sl
                          else [(next(iter(st["entries"]), None), sl)])
                for eid, grp in groups:
                    ent = st["entries"].get(eid) if eid is not None else None
                    if not ent:
                        continue
                    lg = rescale_lgint(grp["lgint_300"], grp["elo_cm"], grp["freq_mhz"],
                                       ent["temps"], ent["qlog"], tex_k)
                    frames.append((grp["freq_mhz"].to_numpy(dtype=float), lg))
            if not frames:
                continue
            f = np.concatenate([x[0] for x in frames])
            lg = np.concatenate([np.asarray(x[1], dtype=float) for x in frames])
            if frame == "sky":
                f = doppler_sky(f, v_lsr)
            out[sp] = (f, lg)
    return out


def source_from_table(name: str, spec: dict, df: pd.DataFrame,
                      coverage: pd.DataFrame | None = None) -> SourceLines:
    """One survey's :class:`SourceLines`.

    ``all_unidentified`` in the source spec declares that every row of the
    table is a U-line — the honest reading of a VizieR table whose *published
    description* is "Unidentified lines", and one that does not depend on
    parsing a label column whose spelling ("U 130156.3") no regex anticipated.
    ``coverage`` supplies the survey's full catalogued line list when the
    U-lines live in a table of their own: the coverage mask asks "was the survey
    sensitive here?", which the 63 U-lines cannot answer and the 399 identified
    ones can.
    """
    if bool(spec.get("all_unidentified")):
        df = df.copy()
        df["unidentified"] = True
    u = df[df["unidentified"].astype(bool)]
    all_freq = df["freq_mhz"].to_numpy(dtype=float)
    if coverage is not None and len(coverage):
        all_freq = np.concatenate([all_freq, coverage["freq_mhz"].to_numpy(dtype=float)])
    return SourceLines(name=name, u_freq=u["freq_mhz"].to_numpy(dtype=float),
                       u_int=u["intensity"].to_numpy(dtype=float) if "intensity" in u
                       else np.full(len(u), np.nan),
                       u_err=u["freq_err_mhz"].to_numpy(dtype=float) if "freq_err_mhz" in u
                       else np.zeros(len(u)),
                       all_freq=all_freq,
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
    fmin = min([float(df["freq_mhz"].min()) for df in source_tables.values() if len(df)] or [0.0])
    # Predicting only inside the surveys' own band saves the J range that no
    # survey can see; the 2 % margin keeps the coverage mask honest at the edges.
    tables = build_species_tables(conf, lines, entries, fmax_mhz=fmax * 1.02,
                                  fmin_mhz=max(0.0, fmin * 0.98))
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
        cov_key = spec.get("coverage_from")
        cov = source_tables.get(str(cov_key)) if cov_key else None
        src = source_from_table(name, spec, df, coverage=cov)
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
                            "all_unidentified": bool(spec.get("all_unidentified")),
                            "coverage_from": (str(cov_key) if cov_key else None),
                            "n_coverage_lines": int(len(cov)) if cov is not None else 0,
                            "has_intensity": bool(np.isfinite(src.u_int).any()),
                            "n_ulines_with_intensity": int(np.isfinite(src.u_int).sum())}
        for sp in targets:
            st = tables.get(sp) or {}
            key = f"{sp}|{name}"
            if not st.get("line_source"):
                results[key] = {"species": sp, "source": name, "line_source": None,
                                "status": "NO_LINE_LIST", "records": [], "best": None}
                continue
            recs = evaluate_species(st["lines"], st["entries"], src, species=sp,
                                    line_source=st["line_source"], conf=mconf, rng=rng)
            rr = {"species": sp, "source": name, "line_source": st["line_source"],
                  "predicted": bool(st.get("predicted")), "verify": bool(st.get("verify")),
                  "status": "OK", "records": recs, "best": best_record(recs)}
            if st.get("rotor"):
                # Is the prediction sharp enough to be matched HERE?  The
                # source's own linewidth is the tolerance the pattern test
                # allows; a predicted error far above it means a coincidence
                # carries no information, and that is a property of the
                # constants, not of the sky.
                tol = max(1e-6, src.fwhm_km_s / 299792.458 * float(np.median(
                    st["lines"]["freq_mhz"]) if len(st["lines"]) else fmax))
                rr["searchability"] = searchability(st["rotor"], tolerance_mhz=tol)
            results[key] = rr
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
            errs = [str(e.get("error", ""))[:200] for e in (v.get("errors") or [])]
            why = f" — {errs[0]}" if errs else ""
            fb = v.get("fallback") or {}
            if fb.get("n_tables"):
                why += (f" — description fallback found {fb['n_tables']} table(s): "
                        + ", ".join(t["table_name"] for t in fb.get("tables", [])[:5]))
            degraded.append(f"source {k}: {v.get('status')}{why}")
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
# validate — the predictor against catalogues that DO exist
# ---------------------------------------------------------------------------
def validate_rotor(assets: dict, cats: dict[str, pd.DataFrame], *, j_max: int = 40,
                   fmax_mhz: float = 1.0e6, docs: dict | None = None,
                   refit: bool = True) -> dict:
    """Predict SO₂, CH₂F₂ and COF₂ and compare line by line with their catalogues.

    ``cats`` maps ``"<database>:<tag>"`` to that catalogue's line table.  For
    each validation species the embedded constants are run through
    :func:`seti.uline.rotorpred.predict_species_lines`, matched to the
    catalogue by quantum numbers, and the residuals reported — and then, when
    ``refit`` and SciPy are available, A, B, C and the quartic constants are
    **refitted to the catalogue's own frequencies**.

    The two residuals answer different questions and both are needed:
    ``residual_mhz`` before the refit measures *the constants*, which are
    reconstructed from the literature and expected to be off; ``rms_after_mhz``
    measures *the Hamiltonian*, and is the number behind any claim that this
    predictor reproduces a catalogue.  ``distortion_truncation`` isolates the
    third question — how far the frequencies move when the quartic constants
    are switched off — which is exactly the error a species with unknown
    quartic constants carries.
    """
    from .rotor import compare_with_cat, fit_constants
    from .rotorpred import species_isotopologues

    out: dict = {"generated_utc": _now(), "species": {}, "j_max": int(j_max)}
    for va in validation_targets(assets):
        sp, db, tag = va["species"], va["database"], va["tag"]
        key = f"{db}:{tag}"
        cat = cats.get(key)
        rec: dict = {"species": sp, "database": db, "tag": tag,
                     "catalogue_lines": int(len(cat)) if cat is not None else 0}
        if cat is None or not len(cat):
            rec["status"] = "NO_CATALOGUE"
            out["species"][key] = rec
            continue
        isos = species_isotopologues(assets, sp)
        if not isos:
            rec["status"] = "NO_CONSTANTS"
            out["species"][key] = rec
            continue
        iso = isos[0]
        c = iso.constants
        rec["constants_embedded"] = c.as_dict()
        rec["quality"] = iso.quality
        try:
            from .rotor import predict_lines
            pred, _ = predict_lines(c, fmin_mhz=0.0, fmax_mhz=float(fmax_mhz), j_max=int(j_max),
                                    lgint_floor=-12.0, err_base_mhz=0.0, err_rel=0.0)
        except Exception as exc:                               # noqa: BLE001
            rec.update({"status": "PREDICTION_FAILED", "error": repr(exc)[:500]})
            out["species"][key] = rec
            continue
        rec["n_predicted"] = int(len(pred))
        rec["comparison"] = compare_with_cat(pred, cat, j_max=int(j_max))
        # The distortion-truncation error: the same constants with the quartic
        # terms zeroed.  This is the size of the error a species whose quartic
        # constants are UNKNOWN carries at the same J — measured, not asserted.
        if c.any_quartic_known:
            rigid = RotorConstantsRigid(c)
            try:
                pred0, _ = predict_lines(rigid, fmin_mhz=0.0, fmax_mhz=float(fmax_mhz),
                                         j_max=int(j_max), lgint_floor=-12.0,
                                         err_base_mhz=0.0, err_rel=0.0)
                rec["distortion_truncation"] = compare_with_cat(pred0, cat, j_max=int(j_max))
            except Exception as exc:                           # noqa: BLE001
                rec["distortion_truncation"] = {"error": repr(exc)[:300]}
        doc = (docs or {}).get(key) or (docs or {}).get(sp)
        if doc:
            rec["constants_from_documentation"] = doc
        obs = []
        from .rotor import _qn_triplet
        for r in cat.itertuples():
            qu, ql = _qn_triplet(r.qn_up), _qn_triplet(r.qn_lo)
            if qu is None or ql is None or qu[0] > j_max or ql[0] > j_max:
                continue
            obs.append((qu, ql, float(r.freq_mhz)))
        rec["n_fit_lines"] = len(obs)
        if refit and len(obs) >= 8:
            try:
                fitted, report = fit_constants(c, obs, j_max=int(j_max))
                rec["refit"] = report
                rec["constants_refitted"] = fitted.as_dict()
            except Exception as exc:                           # noqa: BLE001
                rec["refit"] = {"error": repr(exc)[:500]}
        rec["status"] = "OK"
        out["species"][key] = rec
    ok = [r for r in out["species"].values() if r.get("status") == "OK"]
    res = [r["comparison"]["residual_mhz"]["median_abs"] for r in ok
           if (r.get("comparison") or {}).get("residual_mhz")]
    fit = [r["refit"]["rms_after_mhz"] for r in ok
           if isinstance(r.get("refit"), dict) and "rms_after_mhz" in r["refit"]]
    out["headline"] = {
        "n_species_validated": len(ok),
        "median_abs_residual_mhz_embedded_constants": float(np.median(res)) if res else None,
        "hamiltonian_floor_rms_mhz_after_refit": float(np.median(fit)) if fit else None,
        "reading": ("the first number measures the CONSTANTS (reconstructed from the "
                    "literature); the second measures the HAMILTONIAN itself, refitted to the "
                    "catalogue's own frequencies"),
    }
    return out


def RotorConstantsRigid(c):                                   # noqa: N802
    """``c`` with every distortion constant set to zero (a rigid rotor)."""
    from .rotor import QUARTIC, SEXTIC, RotorConstants
    d = c.as_dict()
    for k in (*QUARTIC, *SEXTIC):
        d[k] = 0.0
    return RotorConstants.from_dict(d)


def stage_validate(conf: dict, out: Path, *, fetch_fn=None, assets: dict | None = None,
                   j_max: int = 40, log: A.AcquisitionLog | None = None) -> dict:
    """Fetch the validation species' catalogues and documentation, then compare."""
    log = log or A.AcquisitionLog(prefix="uline/validate")
    assets = load_rotor_assets() if assets is None else assets
    arc = conf["archives"]
    cats: dict[str, pd.DataFrame] = {}
    docs: dict[str, dict] = {}
    fetch_report = []
    for va in validation_targets(assets):
        db, tag, sp = va["database"], int(va["tag"]), va["species"]
        key = f"{db}:{tag}"
        tmpl = arc["jpl_cat_url"] if db == "jpl" else arc["cdms_cat_url"]
        ent = Entry(tag=tag, name=sp, nlines=0, temps=list(CATDIR_TEMPS), qlog=[], database=db)
        try:
            table, reps = A.fetch_cats([ent], tmpl, fetch_fn=fetch_fn, log=log,
                                       retries=int(arc.get("fetch_retries", 3)), database=db)
        except Exception as exc:                               # noqa: BLE001
            fetch_report.append({"key": key, "status": A.STATUS_FAILED, "error": repr(exc)[:500]})
            continue
        fetch_report.append({"key": key, "status": A.STATUS_OK if len(table) else A.STATUS_ZERO,
                             "n_lines": int(len(table)), "entries": reps})
        if len(table):
            cats[key] = table
        if db == "jpl":
            url = str(arc["jpl_cat_url"]).replace("c{tag:06d}.cat", f"doc/d{tag:06d}.cat")
            try:
                from .litfetch import fetch_one
                rec = fetch_one({"name": f"jpl_doc_{sp}", "species": sp, "kind": "jpl_doc",
                                 "url": url}, fetch_fn=fetch_fn, log=log)
                docs[key] = rec
            except Exception as exc:                           # noqa: BLE001
                docs[key] = {"status": A.STATUS_FAILED, "error": repr(exc)[:300], "url": url}
    rep = validate_rotor(assets, cats, j_max=int(j_max), docs=docs)
    rep["fetch"] = fetch_report
    rep["assets"] = summarise_assets(assets)
    rep["acquisition"] = log.as_dict()
    _write(out / "rotor_validation.json", rep)
    h = rep.get("headline") or {}
    print(f"[uline] validate: {h.get('n_species_validated')} species; embedded-constant residual "
          f"{h.get('median_abs_residual_mhz_embedded_constants')} MHz, Hamiltonian floor "
          f"{h.get('hamiltonian_floor_rms_mhz_after_refit')} MHz")
    return rep


# ---------------------------------------------------------------------------
# litfetch — the microwave literature, from the runner
# ---------------------------------------------------------------------------
def stage_litfetch(conf: dict, out: Path, *, fetch_fn=None, assets: dict | None = None,
                   species: list[str] | None = None,
                   log: A.AcquisitionLog | None = None) -> dict:
    """Walk every literature route for the target species and record what answered."""
    from .litfetch import run_litfetch, species_query_sources

    log = log or A.AcquisitionLog(prefix="uline/litfetch")
    assets = load_rotor_assets() if assets is None else assets
    targets = [sp for sp, blk in (assets.get("species") or {}).items()
               if str(blk.get("role", "")) == "target" and (not species or sp in species)]
    srcs = [s for s in literature_sources(assets)
            if not species or str(s.get("species")) in species]
    for sp in targets:
        srcs.extend(species_query_sources(sp))
    rep = run_litfetch(srcs, fetch_fn=fetch_fn, log=log,
                       timeout=float(conf["archives"].get("fetch_timeout_s", 180)),
                       retries=int(conf["archives"].get("fetch_retries", 3)))
    rep["generated_utc"] = _now()
    rep["targets"] = targets
    rep["embedded"] = summarise_assets(assets)
    rep["acquisition"] = log.as_dict()
    rep["note"] = ("nothing fetched here overwrites an embedded constant: the two are reported "
                   "side by side and promoting one is a commit to "
                   "src/seti/data_assets/rotor_constants.yaml")
    _write(out / "literature.json", rep)
    print(f"[uline] litfetch: {rep['n_ok']}/{rep['n_sources']} routes answered; constants found "
          f"for {sorted(k for k, v in rep['by_species'].items() if v.get('constants'))}")
    return rep


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
STAGES = ("probe", "validate", "litfetch", "acquire", "screen", "assess")
#: ``--stage all``.  ``validate`` and ``litfetch`` are not in it: they are about
#: the PREDICTOR, not about the sky, they are slow (dozens of HTTP round trips),
#: and their outputs change only when the constants do.  Run them explicitly.
DEFAULT_STAGES = ("probe", "acquire", "screen", "assess")


def uline_run(conf: dict | None = None, stage: str = "all", *, out_dir=None, fetch_fn=None,
              query_fn=None, sources=None, species=None, n_trials: int | None = None,
              seed: int | None = None) -> dict:
    conf = conf or load_uline_config()
    out = Path(out_dir) if out_dir else Path("results") / "uline"
    out.mkdir(parents=True, exist_ok=True)
    stages = (DEFAULT_STAGES if stage in ("all", "", None)
              else tuple(s.strip() for s in stage.split(",")))
    rep: dict = {}
    for s in stages:
        if s == "probe":
            rep = stage_probe(conf, out, fetch_fn=fetch_fn, query_fn=query_fn, sources=sources)
        elif s == "validate":
            rep = stage_validate(conf, out, fetch_fn=fetch_fn)
        elif s == "litfetch":
            rep = stage_litfetch(conf, out, fetch_fn=fetch_fn, species=species)
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
                   help="probe|validate|litfetch|acquire|screen|assess|all "
                        "(all = probe,acquire,screen,assess)")
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


__all__ = ["DEFAULTS", "DEFAULT_STAGES", "STAGES", "VERDICT_NONE", "VERDICT_NO_DATA",
           "VERDICT_PATTERN", "build_species_tables", "contaminant_lines", "load_uline_config",
           "main", "screen_all", "source_from_table", "stage_acquire", "stage_assess",
           "stage_litfetch", "stage_probe", "stage_screen", "stage_validate", "uline_run",
           "validate_rotor"]
