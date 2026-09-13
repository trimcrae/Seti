"""Stage orchestration for ISOTOPE.  Writes ``results/isotope/``.

Stages
------
``probe``    reachability of every URL a route would touch (no download).
             Writes ``probe.json``.
``acquire``  the route ledger: human URL → local path → wustl crawl → EarthChem
             Library search → DataCite lookup.  Writes ``acquire.json`` (every
             route's outcome, the header→role resolution) and caches the raw
             table as ``data/pgd_raw.csv``.
``screen``   canonicalise, build the empirical X-grain envelope, classify every
             grain.  Writes ``grains.csv`` and ``screen.json``.
``assess``   ``summary.json`` (verdict, counts per class, role resolution,
             envelope numbers, candidates), ``candidates.csv``, ``frontier.csv``.

Verdict vocabulary (``summary.json["verdict"]``)
------------------------------------------------
``NO_DATA_REACHED``       no route produced a grain table — the run says which
                          routes were tried; this is an access statement
``NO_PURITY_CANDIDATE``   grains were classified; none is a candidate — a count,
                          not an occurrence limit, and never written up
``PURITY_CANDIDATES``     >= 1 PURITY_CANDIDATE or CARBON_PURITY_CANDIDATE,
                          pending contamination vetting

Entry points
------------
``isotope_run(conf=None, stage="all", ...)`` and ``main(argv=None)`` so the
module runs as ``python -m seti.isotope.run --stage all``.  The CLI wiring in
``seti.cli`` belongs to the parent session.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import acquire as A
from . import purity as P

VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_NONE = "NO_PURITY_CANDIDATE"
VERDICT_CANDIDATES = "PURITY_CANDIDATES"
STAGES = ("probe", "acquire", "screen", "assess")

#: Minimal defaults so a missing ``config/isotope.yaml`` degrades rather than
#: crashes; the yaml is the authority and carries the full pattern lists.
DEFAULTS: dict = {
    "acquire": {"timeout_s": 120, "retries": 2, "max_bytes": 400_000_000,
                "crawl_urls": ["https://presolar.physics.wustl.edu/presolar-grain-database/",
                               "http://presolar.physics.wustl.edu/presolar-grain-database/"],
                "link_suffix_regex": r"(?i)\.(xlsx|xls|csv|zip)(\?.*)?$",
                "ecl_home": "https://ecl.earthchem.org/home.php",
                "ecl_search_urls": ["https://ecl.earthchem.org/search.php?title=Presolar+Grain+Database"],
                "ecl_query": "Presolar Grain Database",
                "datacite_api": "https://api.datacite.org/dois",
                "datacite_queries": ['titles.title:"Presolar Grain Database"'],
                "doi_resolver": "https://doi.org/"},
    "roles": {
        "grain_id": ["^pgdid", "^grainid$", "^grainlabel$", "^id$", "^grain$", "^label$"],
        "meteorite": ["^meteorite"],
        "grain_type": ["^pgdtype$", "^type$", "^graintype$", "^class$", "^classification$", "type$"],
        "d29si": ["^d29si", "29si28si", "29si"],
        "d30si": ["^d30si", "30si28si", "30si"],
        "c12c13": ["^12c13c", "12c13c", "c12c13"],
        "n14n15": ["^14n15n", "14n15n", "n14n15"],
        "al26al27": ["^26al27al", "26al27al", "al26al27"],
        "ti44ti48": ["^44ti48ti", "44ti48ti", "ti44ti48"],
        "d44ca": ["^d44ca", "44ca40ca"],
    },
    "error_marker": r"(err|sig|unc|stdev|(^|[^p])sd|(^|[^p])pm|uncert|error)",
    "unit_scales": [],
    "x_type_regex": r"(?i)^\s*x",
    "natural_type_regex": r"(?i)^\s*(m|ms|mainstream|x[0-9a-z]*|y|z|ab[0-9]?|n|nova|c)\b",
    "solar": dict(P.DEFAULT_SOLAR),
    "x_package": dict(P.DEFAULT_PACKAGE),
    "x_package_min_n": 10,
    "envelope_fallback": dict(P.DEFAULT_ENVELOPE_FALLBACK),
    "thresholds": dict(P.DEFAULT_THRESHOLDS),
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_isotope_config(path: Path | None = None) -> dict:
    """``config/isotope.yaml`` over :data:`DEFAULTS`; a missing file degrades."""
    try:
        import yaml  # noqa: PLC0415
        path = Path(path) if path is not None else repo_root() / "config" / "isotope.yaml"
        if not path.exists():
            print(f"[isotope] config {path} not found; using defaults")
            return _deep_update(DEFAULTS, {})
        return _deep_update(DEFAULTS, yaml.safe_load(path.read_text()) or {})
    except Exception as exc:                              # noqa: BLE001
        print(f"[isotope] config not loaded ({exc!r}); using defaults")
        return _deep_update(DEFAULTS, {})


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _clean(obj):
    """NaN → None recursively so the JSON is standard."""
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
# stages
# ---------------------------------------------------------------------------
def stage_probe(conf: dict, out: Path, *, fetch_fn=None, table_url: str | None = None) -> dict:
    rep = A.probe_routes(conf, fetch_fn=fetch_fn, table_url=table_url)
    rep["note"] = ("reachability only; the wustl host timed out on the runner in the three "
                   "necrofrontier-probe runs, so a NOT_REACHED there is expected and the "
                   "EarthChem / DataCite routes carry the search")
    _write(out / "probe.json", rep)
    print(f"[isotope] probe: {rep['n_reached']}/{rep['n_probed']} endpoints reached")
    return rep


def stage_acquire(conf: dict, out: Path, *, fetch_fn=None, table_url: str | None = None,
                  table_path: str | None = None) -> dict:
    cache = out / "data" / "pgd_raw.csv"
    acq = A.acquire_table(conf, table_url=table_url, table_path=table_path, fetch_fn=fetch_fn,
                          cache_path=cache)
    rep = acq.as_dict()
    rep["stage"] = "acquire"
    rep["inputs"] = {"table_url": table_url, "table_path": str(table_path) if table_path else None}
    if acq.status == A.STATUS_OK:
        res = A.resolve_roles(acq.table.columns, conf)
        rep["role_resolution"] = res.as_dict()
        rep["cache_path"] = str(cache) if acq.route_used not in ("local_path",) else str(acq.source)
    else:
        rep["role_resolution"] = None
        rep["verdict"] = VERDICT_NO_DATA
    _write(out / "acquire.json", rep)
    print(f"[isotope] acquire: {acq.status} via {acq.route_used} ({acq.n_rows} rows)")
    return rep


def _load_raw(out: Path, conf: dict, acquire_report: dict | None) -> tuple[pd.DataFrame, str | None]:
    """The raw table the acquire stage left behind (cache or the local path it used)."""
    cands = []
    if acquire_report:
        for k in ("cache_path", "source"):
            v = acquire_report.get(k)
            if v and Path(str(v)).exists():
                cands.append(Path(str(v)))
    cands.append(out / "data" / "pgd_raw.csv")
    for p in cands:
        if p.exists():
            try:
                if p.suffix.lower() == ".csv" and p == out / "data" / "pgd_raw.csv":
                    return pd.read_csv(p, dtype=str, keep_default_na=False), str(p)
                sheets = A.rows_from_bytes(p.read_bytes(), p.name)
                df, _meta = A.best_frame(sheets, conf)
                if len(df):
                    return df, str(p)
            except Exception as exc:                      # noqa: BLE001
                print(f"[isotope] could not read {p}: {exc!r}")
    return pd.DataFrame(), None


def stage_screen(conf: dict, out: Path, *, raw: pd.DataFrame | None = None) -> dict:
    acq_rep = _read_json(out / "acquire.json")
    source = None
    if raw is None:
        raw, source = _load_raw(out, conf, acq_rep)
    if raw is None or not len(raw):
        rep = {"stage": "screen", "generated_utc": _now(), "status": VERDICT_NO_DATA,
               "n_grains": 0, "counts": {}, "roles": None, "envelope": None,
               "routes_tried": (acq_rep or {}).get("routes_tried", []),
               "note": "no raw table on disk; nothing was screened"}
        _write(out / "screen.json", rep)
        print("[isotope] screen: NO_DATA_REACHED (no raw table)")
        return rep
    res = A.resolve_roles(raw.columns, conf)
    canon = A.canonicalise(raw, res)
    results, env, counts = P.classify_table(canon, conf, roles=res.roles)
    flat = P.flatten_for_csv(results)
    flat.to_csv(out / "grains.csv", index=False)
    rep = {"stage": "screen", "generated_utc": _now(), "status": "OK", "source": source,
           "n_grains": int(len(results)), "counts": counts, "roles": res.as_dict(),
           "envelope": env, "grain_types": _type_counts(canon)}
    _write(out / "screen.json", rep)
    print(f"[isotope] screen: {len(results)} grains; " +
          ", ".join(f"{c}={counts.get(c, 0)}" for c in P.CLASSES))
    return rep


def _type_counts(canon: pd.DataFrame, top: int = 40) -> dict:
    if "grain_type" not in canon.columns or not len(canon):
        return {}
    vc = canon["grain_type"].astype(str).str.strip().replace("", "(blank)").value_counts()
    return {str(k): int(v) for k, v in vc.head(top).items()}


def stage_assess(conf: dict, out: Path) -> dict:
    thr = conf.get("thresholds") or {}
    acq_rep = _read_json(out / "acquire.json") or {}
    scr = _read_json(out / "screen.json") or {}
    grains = pd.DataFrame()
    if (out / "grains.csv").exists():
        try:
            grains = pd.read_csv(out / "grains.csv", dtype={"grain_id": str, "meteorite": str,
                                                            "grain_type": str, "flags": str,
                                                            "grade": str, "reason": str})
        except Exception as exc:                          # noqa: BLE001
            print(f"[isotope] grains.csv unreadable: {exc!r}")
    n = int(len(grains))
    routes_tried = list(dict.fromkeys(acq_rep.get("routes_tried") or scr.get("routes_tried") or []))
    if n == 0 or scr.get("status") != "OK":
        summary = {
            "verdict": VERDICT_NO_DATA, "generated_utc": _now(), "n_grains": 0,
            "counts": {c: 0 for c in P.CLASSES}, "n_candidates": 0, "candidates": [],
            "role_resolution": acq_rep.get("role_resolution"), "envelope": None,
            "acquisition": {"status": acq_rep.get("status", VERDICT_NO_DATA),
                            "route_used": acq_rep.get("route_used"), "source": acq_rep.get("source"),
                            "routes_tried": routes_tried, "routes": acq_rep.get("routes", []),
                            "inputs": acq_rep.get("inputs")},
            "note": ("no grain table was reached, so nothing about isotopic purity was measured; "
                     "this is an ACCESS statement, NOT a null result, and is not written up. "
                     "Routes tried: " + (", ".join(routes_tried) or "none") + ". The table "
                     "(Stephan et al. 2024, ApJS 270, 27) must be located in the EarthChem Library "
                     "or requested from the authors and passed as --table-url / --table-path."),
        }
        _write(out / "summary.json", summary)
        pd.DataFrame(columns=["grain_id", "class"]).to_csv(out / "candidates.csv", index=False)
        print(f"[isotope] assess: {VERDICT_NO_DATA}")
        return summary

    is_cand = grains["class"].isin([P.CLASS_CANDIDATE, P.CLASS_CARBON])
    cands = grains[is_cand].copy()
    if len(cands):
        cands["grade"] = cands["grade"].fillna("")
        cands = cands.sort_values(["grade", "d29si"], na_position="last")
    cands.to_csv(out / "candidates.csv", index=False)
    fr = P.frontier(grains, int(thr.get("frontier_n", 25)))
    fr.to_csv(out / "frontier.csv", index=False)
    counts = scr.get("counts") or {c: int((grains["class"] == c).sum()) for c in P.CLASSES}
    degraded = list(counts.get("tests_not_run") or [])
    env = scr.get("envelope") or {}
    if str(env.get("source", "")).startswith("config_fallback"):
        degraded.append("envelope:config_fallback_no_classified_x_grain")
    verdict = VERDICT_CANDIDATES if len(cands) else VERDICT_NONE
    slim_cols = [c for c in ("grain_id", "meteorite", "grain_type", "class", "grade", "reason",
                             "d29si", "d29si_err", "d30si", "d30si_err", "margin29_sigma",
                             "margin30_sigma", "n_partners_measured", "partners_measured",
                             "partners_solar", "partners_package_excluded", "flags",
                             "c12c13", "c12c13_err", "n14n15", "n14n15_err", "al26al27",
                             "al26al27_err", "ti44ti48", "ti44ti48_err", "d44ca", "d44ca_err",
                             "mixing_line_distance") if c in cands]
    summary = {
        "verdict": verdict, "generated_utc": _now(), "n_grains": n,
        "n_candidates": int(len(cands)),
        "n_contamination_suspect": int(counts.get(P.FLAG_CONTAMINATION, 0)),
        "counts": {c: int(counts.get(c, 0)) for c in P.CLASSES},
        "candidates_by_grade": counts.get("candidates_by_grade"),
        "n_beyond_envelope": int(counts.get("n_beyond_envelope", 0)),
        "role_resolution": scr.get("roles") or acq_rep.get("role_resolution"),
        "envelope": {k: v for k, v in env.items() if k != "thresholds"},
        "grain_types": scr.get("grain_types"),
        "degraded": degraded,
        "candidates": _clean(cands[slim_cols].head(200).to_dict(orient="records")) if len(cands) else [],
        "frontier": _clean(fr.to_dict(orient="records")),
        "acquisition": {"status": acq_rep.get("status"), "route_used": acq_rep.get("route_used"),
                        "source": acq_rep.get("source"), "routes_tried": routes_tried,
                        "inputs": acq_rep.get("inputs")},
        "thresholds": thr,
        "note": ("a PURITY_CANDIDATE is a grain whose 28Si (or 12C) purity lies beyond the "
                 "database's own most extreme classified X grain in BOTH silicon ratios by > 3σ "
                 "AND whose measured nucleosynthetic partners are all consistent with solar; it is "
                 "PENDING contamination vetting (wafer fragment, SIMS artefact, mislabelled X grain). "
                 "NO_PURITY_CANDIDATE is a count, not an occurrence limit, and is not written up "
                 "(CLAUDE.md). Every partner test that could not run is listed in 'degraded'."),
    }
    _write(out / "summary.json", summary)
    print(f"[isotope] assess: {verdict} — {len(cands)} candidates of {n} grains; "
          f"{summary['n_contamination_suspect']} contamination-suspect; degraded={degraded}")
    return summary


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def isotope_run(conf: dict | None = None, stage: str = "all", *, table_url: str | None = None,
                table_path: str | None = None, out_dir=None, fetch_fn=None) -> dict:
    """Run one stage, a comma list, or all.  Returns the last stage's report."""
    conf = conf if conf is not None else load_isotope_config()
    out = Path(out_dir) if out_dir else repo_root() / "results" / "isotope"
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage in ("all", "", None) else tuple(s.strip() for s in stage.split(","))
    rep: dict = {}
    for s in stages:
        if s == "probe":
            rep = stage_probe(conf, out, fetch_fn=fetch_fn, table_url=table_url)
        elif s == "acquire":
            rep = stage_acquire(conf, out, fetch_fn=fetch_fn, table_url=table_url, table_path=table_path)
        elif s == "screen":
            rep = stage_screen(conf, out)
        elif s == "assess":
            rep = stage_assess(conf, out)
        else:
            raise SystemExit(f"unknown stage {s!r}; choose from {STAGES}")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(prog="seti isotope",
                                description="ISOTOPE (S46): isotopic purity without a "
                                            "nucleosynthetic package in the Presolar Grain Database")
    p.add_argument("--stage", default="all", help="probe|acquire|screen|assess|all or a comma list")
    p.add_argument("--table-url", default="", help="URL of the PGD SiC table (xlsx/csv/zip) found by a human")
    p.add_argument("--table-path", default="", help="local path to the PGD SiC table")
    p.add_argument("--out-dir", default="", help="results directory (default results/isotope)")
    p.add_argument("--config", default="", help="alternative config yaml")
    a = p.parse_args(argv)
    conf = load_isotope_config(Path(a.config) if a.config else None)
    rep = isotope_run(conf, stage=a.stage, table_url=a.table_url or None,
                      table_path=a.table_path or None, out_dir=a.out_dir or None)
    v = rep.get("verdict") if isinstance(rep, dict) else None
    if v:
        print(f"[isotope] verdict: {v}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())


__all__ = ["DEFAULTS", "STAGES", "VERDICT_CANDIDATES", "VERDICT_NONE", "VERDICT_NO_DATA",
           "isotope_run", "load_isotope_config", "main", "stage_acquire", "stage_assess",
           "stage_probe", "stage_screen"]
