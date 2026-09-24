"""Harvest per-star scores from other channels' run artifacts (runner only).

Most channels commit only their survivors, but their runs upload the whole
scored parent as a workflow artifact (cenotaph ``greyfit.parquet``, ignition
``stars_s*.csv``, cradle ``parent_screened.csv``, tailings' checkpointed
catalogues, ...).  An artifact is readable from a later run of ANY workflow in
the repository for 90 days, so the per-star scores can be recovered without
re-running the channel.  The sandbox cannot reach the artifact store, so this
module runs on the runner after ``gh run download`` has laid the artifacts out
as ``<harvest_dir>/<channel>/<run_id>/<artifact>/...``.

Every extractor discovers the schema it is given and refuses to guess: a file
that lacks the columns it needs is reported (``SCHEMA_NOT_RECOGNISED`` with the
columns it did find) and nothing is emitted for that channel.  ``manifest``
records every file seen, so the next dispatch can be fixed against the real
layout rather than an assumed one.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import STD_COLUMNS, standardise

TABLE_EXT = (".csv", ".csv.gz", ".parquet", ".tsv")


def _is_table(p: Path) -> bool:
    return any(str(p).endswith(e) for e in TABLE_EXT)


def _read(p: Path, columns=None) -> pd.DataFrame:
    if str(p).endswith(".parquet"):
        return pd.read_parquet(p, columns=columns)
    return pd.read_csv(p, low_memory=False, usecols=columns,
                       sep="\t" if str(p).endswith(".tsv") else ",")


def _columns(p: Path) -> tuple[list[str], int | None]:
    try:
        if str(p).endswith(".parquet"):
            import pyarrow.parquet as pq
            f = pq.ParquetFile(p)
            return list(f.schema_arrow.names), int(f.metadata.num_rows)
        head = pd.read_csv(p, nrows=5, low_memory=False,
                           sep="\t" if str(p).endswith(".tsv") else ",")
        n = sum(1 for _ in open(p, "rb")) - 1 if not str(p).endswith(".gz") else None
        return list(head.columns), n
    except Exception as e:  # noqa: BLE001 -- a manifest records, it does not fail
        return [f"<unreadable: {type(e).__name__}: {str(e)[:120]}>"], None


def manifest(root: Path) -> list[dict]:
    """Every file under ``root``: path, bytes, and for tables the columns and rows."""
    out = []
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        rec = {"path": str(p.relative_to(root)), "bytes": p.stat().st_size}
        if _is_table(p):
            cols, n = _columns(p)
            rec["n_rows"] = n
            rec["columns"] = cols[:400]
            rec["n_columns"] = len(cols)
        out.append(rec)
    return out


def _find(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in Path(root).rglob(pattern) if p.is_file())


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _truthy(s: pd.Series) -> np.ndarray:
    if s.dtype == bool:
        return s.to_numpy()
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"]).to_numpy()


COV_COLS = ("source_id", "ra", "dec", "l", "b", "ecl_lat", "phot_g_mean_mag", "bp_rp", "ruwe",
            "parallax", "ipd_frac_multi_peak", "phot_variable_flag", "non_single_star",
            "visibility_periods_used", "astrometric_matched_transits")


def side_covariates(d: pd.DataFrame) -> pd.DataFrame | None:
    """The Gaia columns a channel's own table already carries (null covariates
    that need no further archive call)."""
    cols = [c for c in COV_COLS if c in d.columns]
    if "source_id" not in cols or len(cols) < 3:
        return None
    return d[cols].drop_duplicates("source_id")


# ---------------------------------------------------------------------------
# extractors: each returns (standard frame or None, report dict)
# ---------------------------------------------------------------------------


def extract_cenotaph(root: Path) -> tuple[pd.DataFrame | None, dict]:
    fs = _find(root, "greyfit.parquet")
    rep = {"files": [str(f) for f in fs]}
    if not fs:
        rep["status"] = "FILE_NOT_FOUND"
        return None, rep
    g = pd.read_parquet(fs[0])
    need = {"source_id", "grey_sigma"}
    if not need <= set(g.columns):
        rep.update(status="SCHEMA_NOT_RECOGNISED", columns=list(g.columns))
        return None, rep
    g["_s"] = _num(g["grey_sigma"])
    ok = g["grey_verdict"].astype(str).eq("ok") if "grey_verdict" in g else True
    # Leg 1 (grey > 3 sigma) and leg 2 (no mid-IR excess) -- the channel's own tail.
    mid = _truthy(g["midir_excess"]) if "midir_excess" in g else np.zeros(len(g), bool)
    g["_f"] = (g["_s"].to_numpy() > 3.0) & ~mid & np.asarray(ok)
    rep.update(status="OK", n_rows=int(len(g)), n_flag=int(g["_f"].sum()),
               n_verdict_ok=int(np.asarray(ok).sum()) if not isinstance(ok, bool) else len(g))
    return standardise(g, sid="source_id", score="_s", flag="_f"), rep


def extract_ignition(root: Path) -> tuple[pd.DataFrame | None, dict]:
    fs = _find(root, "stars_s*of*.csv")
    rep = {"n_files": len(fs)}
    if not fs:
        rep["status"] = "FILE_NOT_FOUND"
        return None, rep
    parts = []
    for f in fs:
        try:
            parts.append(pd.read_csv(f, low_memory=False))
        except Exception as e:  # noqa: BLE001
            rep.setdefault("unreadable", []).append(f"{f.name}: {e}")
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    need = {"source_id", "w1_slope_sigma", "w2_slope_sigma"}
    if not need <= set(d.columns):
        rep.update(status="SCHEMA_NOT_RECOGNISED", columns=list(d.columns)[:200])
        return None, rep
    s = pd.concat([_num(d["w1_slope_sigma"]), _num(d["w2_slope_sigma"])], axis=1).min(axis=1)
    # A rise in only one band cannot enter IGNITION's tail (ONE_BAND_ONLY veto):
    # the min of the two band significances is the channel's own two-band rule.
    d["_s"] = s
    flag = np.zeros(len(d), bool)
    for c in ("star_is_candidate", "is_candidate"):
        if c in d:
            flag |= _truthy(d[c])
    d["_f"] = flag
    rep["_cov"] = side_covariates(d)
    rep.update(status="OK", n_rows_raw=int(len(d)), n_unique=int(d["source_id"].nunique()),
               n_flag=int(flag.sum()))
    return standardise(d, sid="source_id", score="_s", flag="_f"), rep


def extract_cradle(root: Path) -> tuple[pd.DataFrame | None, dict]:
    fs = _find(root, "parent_screened.csv")
    rep = {"files": [str(f) for f in fs]}
    if not fs:
        rep["status"] = "FILE_NOT_FOUND"
        return None, rep
    d = pd.read_csv(fs[0], low_memory=False)
    if "source_id" not in d:
        rep.update(status="SCHEMA_NOT_RECOGNISED", columns=list(d.columns))
        return None, rep
    if "log_f_fmax_1gyr" in d:
        d["_s"] = _num(d["log_f_fmax_1gyr"])
        rep["score"] = "log_f_fmax_1gyr"
    elif {"chi_W3", "chi_W4"} <= set(d.columns):
        d["_s"] = pd.concat([_num(d["chi_W3"]), _num(d["chi_W4"])], axis=1).min(axis=1)
        rep["score"] = "min(chi_W3, chi_W4)"
    else:
        rep.update(status="SCHEMA_NOT_RECOGNISED", columns=list(d.columns))
        return None, rep
    # an insignificant excess is not in the channel's tail whatever f/f_max says
    if "excess_significant" in d:
        d.loc[~_truthy(d["excess_significant"]), "_s"] = -np.inf
    d["_f"] = _truthy(d["shortlisted"]) if "shortlisted" in d else False
    if "is_control" in d:   # injected/known controls are not sky stars
        d = d[~_truthy(d["is_control"])]
    rep["_cov"] = side_covariates(d)
    rep.update(status="OK", n_rows=int(len(d)), n_flag=int(np.sum(d["_f"])))
    return standardise(d, sid="source_id", score="_s", flag="_f"), rep


def extract_ring(root: Path) -> tuple[pd.DataFrame | None, dict]:
    """White-dwarf leg: needs the SCREENED parent, not the flagged excess list."""
    rep = {"tables": []}
    best = None
    for f in _find(root, "*"):
        if not _is_table(f):
            continue
        cols, n = _columns(f)
        rep["tables"].append({"path": str(f.relative_to(root)), "n_rows": n,
                              "has_chi": {"chi_W1", "chi_W2"} <= set(cols)})
        if "source_id" in cols and {"chi_W1", "chi_W2"} <= set(cols) and "wd" in str(f):
            if best is None or (n or 0) > (best[1] or 0):
                best = (f, n)
    if best is None:
        rep["status"] = "FILE_NOT_FOUND"
        return None, rep
    d = _read(best[0])
    screen = next(iter(_find(root, "screen.json")), None)
    n_input = None
    if screen is not None:
        try:
            n_input = json.loads(screen.read_text()).get("n_input")
        except Exception:  # noqa: BLE001
            pass
    rep.update(file=str(best[0].relative_to(root)), n_rows=int(len(d)), n_screen_input=n_input)
    # The committed excess.csv is only the flagged 4,694 of ~25,932: a tail,
    # not a parent.  Accept the table as a parent only if it is the screen input.
    if n_input and len(d) < 0.9 * n_input:
        rep["status"] = "TAIL_ONLY"
        return None, rep
    d["_s"] = pd.concat([_num(d["chi_W1"]), _num(d["chi_W2"])], axis=1).min(axis=1)
    d["_f"] = _truthy(d["ring_candidate"]) if "ring_candidate" in d else False
    rep["status"] = "OK"
    rep["_cov"] = side_covariates(d)
    return standardise(d, sid="source_id", score="_s", flag="_f"), rep


def extract_ossuary(root: Path) -> tuple[pd.DataFrame | None, dict]:
    rep = {"tables": []}
    best = None
    for f in _find(root, "*"):
        if not _is_table(f):
            continue
        cols, n = _columns(f)
        rep["tables"].append({"path": str(f.relative_to(root)), "n_rows": n,
                              "n_cols": len(cols)})
        if "source_id" in cols and {"chi_W1", "chi_W2", "chi_W3"} <= set(cols):
            if best is None or (n or 0) > (best[1] or 0):
                best = (f, n)
    if best is None:
        rep["status"] = "NO_SCORED_TABLE"
        return None, rep
    d = _read(best[0])
    rep.update(file=str(best[0].relative_to(root)), n_rows=int(len(d)))
    if "excess_flagged" in str(best[0]):
        rep["status"] = "TAIL_ONLY"
        return None, rep
    d["_s"] = d[["chi_W1", "chi_W2", "chi_W3"]].apply(pd.to_numeric, errors="coerce").max(axis=1)
    d["_f"] = _truthy(d["candidate"]) if "candidate" in d else False
    rep["status"] = "OK"
    return standardise(d, sid="source_id", score="_s", flag="_f"), rep


def extract_tailings(root: Path) -> tuple[pd.DataFrame | None, dict]:
    """Re-run TAILINGS' own reduce on its checkpointed catalogues (no network)."""
    rep = {"surveys": {}}
    fs = [f for f in _find(root, "stars_*.parquet") if "wide" not in f.name]
    if not fs:
        rep["status"] = "FILE_NOT_FOUND"
        return None, rep
    from ..tailings.run import _cfg_block, load_config, reduce_survey
    block = _cfg_block(load_config())
    frames = []
    with tempfile.TemporaryDirectory() as td:
        for f in fs:
            sv = f.stem.replace("stars_", "").upper()
            try:
                r = reduce_survey(pd.read_parquet(f), survey=sv, block=block, out_dir=Path(td))
                rep["surveys"][sv] = {k: r.get(k) for k in ("n_stars", "n_sparse", "n_vetted",
                                                           "verdict")}
            except Exception as e:  # noqa: BLE001
                rep["surveys"][sv] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
                continue
            p = Path(td) / f"per_star_{sv.lower()}.parquet"
            if p.exists():
                x = pd.read_parquet(p)
                x["_key"] = sv.lower() + ":" + x["star_id"].astype(str)
                x["_f"] = x["classification"].astype(str).str.lower().eq("sparse") \
                    if "classification" in x else False
                frames.append(standardise(x, sid=None, key_ns="tailings", key_col="_key",
                                          score="z_max", flag="_f"))
    if not frames:
        rep["status"] = "NO_PER_STAR_OUTPUT"
        return None, rep
    rep["status"] = "OK"
    return pd.concat(frames, ignore_index=True)[STD_COLUMNS], rep


EXTRACTORS = {
    "cenotaph": extract_cenotaph,
    "ignition": extract_ignition,
    "cradle": extract_cradle,
    "ring_wd": extract_ring,
    "ossuary": extract_ossuary,
    "tailings": extract_tailings,
}


def harvest(harvest_dir: Path, scores_dir: Path) -> dict:
    """Manifest every downloaded artifact and run each channel's extractor."""
    harvest_dir, scores_dir = Path(harvest_dir), Path(scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    report = {"channels": {}}
    import time as _t
    dirs = sorted((p for p in harvest_dir.iterdir() if p.is_dir()),
                  key=lambda p: (p.name == "tailings", p.name)) if harvest_dir.exists() else []
    for ch_dir in dirs:     # tailings (a full re-reduce) last: slowest
        ch = ch_dir.name
        t0 = _t.time()
        print(f"[confluence] harvest {ch}: start", flush=True)
        rec = {"manifest": manifest(ch_dir)}
        fn = EXTRACTORS.get(ch)
        if fn is None:
            rec["status"] = "NO_EXTRACTOR"
        else:
            try:
                frame, rep = fn(ch_dir)
            except Exception as e:  # noqa: BLE001 -- one channel must not sink the rest
                frame, rep = None, {"status": "EXTRACTOR_ERROR",
                                    "error": f"{type(e).__name__}: {str(e)[:500]}"}
            cov = rep.pop("_cov", None)
            if cov is not None and len(cov):
                cov.to_parquet(scores_dir / f"_cov_{ch}.parquet", index=False)
                rec["n_side_covariates"] = int(len(cov))
                rec["side_covariate_columns"] = list(cov.columns)
            rec.update(rep)
            if frame is not None and len(frame):
                frame.to_parquet(scores_dir / f"{ch}.parquet", index=False)
                rec["n_emitted"] = int(len(frame))
                rec["n_scored"] = int(np.isfinite(frame["score"].to_numpy(float)).sum())
        rec["elapsed_s"] = round(_t.time() - t0, 1)
        report["channels"][ch] = rec
        print(f"[confluence] harvest {ch}: {rec.get('status')} emitted={rec.get('n_emitted')} "
              f"({rec['elapsed_s']} s)", flush=True)
        # partial report after every channel, so a killed step still says what it did
        (scores_dir / "_harvest_partial.json").write_text(
            json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "manifest"}
                        for k, v in report["channels"].items()}, default=str, indent=1))
    return report
