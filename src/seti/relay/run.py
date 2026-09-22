"""RELAY stage orchestration: probe -> targets -> geometry -> recut -> assess.

Every stage writes its own JSON as soon as it has something (a killed run
loses minutes, not the stage), and every verdict that depends on an archive
carries the archive's answer as a first-class field.  ``NO_DATA_REACHED`` is
a statement about the run, never about the sky.

Outputs (``results/relay/``):

* ``probe.json``            every endpoint's answer, verbatim heads, the BL
                            API's real response shapes and keys
* ``targets.csv/json``      the BL open-data target list resolved to Gaia DR3
* ``gaia_sample.json``      the 100 pc sample's provenance (rows travel as an
                            artifact parquet, never committed)
* ``geometry.json``         directed-pair counts per beam and geometry, the
                            analytic theta^2 N^2 expectation beside each, the
                            fitted scaling slope, RV completeness
* ``pairs_full_<beam>.csv.gz``   the whole directed-pair catalogue of a beam
                            narrow enough to commit
* ``pairs_targets.csv.gz``  every pair whose transmitter is a BL target or an
                            in-beam neighbour of one, with its kinematics
* ``recut.json``, ``recut_pointings.csv``   which observed stars sit on pair
                            lines, per telescope/band, with the drift window
* ``hits.json``, ``hits_crossmatch.csv``    the published hits against the
                            geometry and the prior
* ``summary.json``          verdict, funnel, coverage
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..herdsman.acquire import apply_rv_zero_point
from ..metronome.acquire import STATUS_FAILED, STATUS_OK, STATUS_ZERO, AcquisitionLog, tap_query
from . import acquire as acq
from . import geometry as geo

STAGES = ("probe", "targets", "geometry", "recut", "assess")

V_GEOMETRY = "GEOMETRY_COMPUTED"
V_GEOMETRY_PARTIAL = "GEOMETRY_COMPUTED_ON_PARTIAL_SAMPLE"
V_NO_DATA = "NO_DATA_REACHED"
V_TARGETS = "TARGETS_RESOLVED"
V_RECUT = "RECUT_COMPLETE"
V_NO_HIT_CATALOGUE = "NO_HIT_CATALOGUE_REACHED"
V_NO_PAIRLINE_HIT = "NO_PAIRLINE_HIT"
V_PAIRLINE_OFF_PRIOR = "PAIRLINE_HITS_OFF_PRIOR"
V_PAIRLINE_MATCH = "PAIRLINE_DRIFT_MATCH"

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "relay.yaml"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_relay_config(path: Path | None = None) -> dict:
    p = Path(path) if path else CONFIG_PATH
    with open(p) as fh:
        return yaml.safe_load(fh)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _write(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def _read(path: Path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def _write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        df.to_csv(fh, index=False)


def _read_csv_gz(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    with gzip.open(p, "rt") as fh:
        return pd.read_csv(fh)


def name_key(name) -> str:
    """Canonical key for matching target strings across services."""
    s = re.sub(r"[\s_\-]+", "", str(name or "")).upper()
    s = re.sub(r"^GL(?=\d)", "GJ", s)
    s = re.sub(r"^GAIA(E?DR3)?", "", s)
    return s


def _beam_name(b: dict) -> str:
    return str(b["name"])


def _band_of(freq_ghz: float, hint, conf_tel: dict) -> str | None:
    if hint:
        return str(hint).upper()
    if freq_ghz is None or not np.isfinite(freq_ghz):
        return None
    for row in conf_tel.get("band_from_frequency_ghz") or []:
        if float(row["lo"]) <= freq_ghz < float(row["hi"]):
            return str(row["band"])
    return None


def _telescope_key(name, conf_tel: dict) -> str | None:
    if not name:
        return None
    s = str(name).lower()
    for k in conf_tel:
        if k in ("in_beam_max_arcmin", "band_from_frequency_ghz"):
            continue
        if k.lower() in s or s in k.lower():
            return k
    if "green" in s or "gbt" in s:
        return "GBT"
    if "parkes" in s or "murriyang" in s:
        return "Parkes"
    if "meer" in s:
        return "MeerKAT"
    if "ata" in s or "allen" in s:
        return "ATA"
    return None


def _hpbw_arcmin(tel_key, band, conf_tel: dict) -> float:
    t = conf_tel.get(tel_key) if tel_key else None
    if not t:
        return float("nan")
    h = t.get("hpbw_arcmin") or {}
    if band and band in h:
        return float(h[band])
    return float(h.get(t.get("default_band"), float("nan")))


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def stage_probe(conf: dict, out: Path, *, bl_fetch=None, tap_fn=None, query_fn=None,
                fetch_fn=None, budget_s: float | None = None, log=None) -> dict:
    """Reachability and product check of every service, heads recorded verbatim."""
    log = log or AcquisitionLog(prefix="relay/probe")
    started = time.monotonic()
    budget = float(budget_s or (conf.get("probe") or {}).get("budget_s", 1500))
    deadline = started + budget
    rep = {"stage": "probe", "generated_utc": _now(), "endpoints": {}, "hit_tables": []}

    def _save():
        rep["elapsed_s"] = round(time.monotonic() - started, 1)
        rep["acquisition"] = log.as_dict()
        _write(out / "probe.json", rep)

    # --- Breakthrough Listen open data ------------------------------------
    bl = acq.BLOpenData(conf["bl_opendata"], fetch_json_fn=bl_fetch, log=log)
    names, rec = bl.list_targets()
    rec["sample_names"] = names[:15]
    rep["endpoints"]["bl_list_targets"] = rec
    tels, rec = bl.list_simple("telescopes")
    rec["values"] = tels[:40]
    rep["endpoints"]["bl_list_telescopes"] = rec
    ftypes, rec = bl.list_simple("file_types")
    rec["values"] = ftypes[:40]
    rep["endpoints"]["bl_list_file_types"] = rec
    if names:
        files, rec = bl.query_files(names[0], limit=5)
        rec["normalised_sample"] = files[:3]
        rep["endpoints"]["bl_query_files_sample"] = rec
    for tel in tels:
        if "meer" in tel.lower():
            files, rec = bl.query_files("", limit=20, telescopes=tel)
            rec["normalised_sample"] = files[:3]
            rep["endpoints"]["bl_query_files_meerkat"] = rec
            break
    rep["bl_targets_n"] = len(names)
    rep["bl_telescopes"] = tels
    rep["bl_file_types"] = ftypes
    _save()

    # --- Gaia ---------------------------------------------------------------
    rep["endpoints"]["gaia_esa_count"] = acq.gaia_count(conf["sample"], tap_fn=tap_fn)
    _save()

    # --- SIMBAD --------------------------------------------------------------
    try:
        df = acq.resolve_names_simbad(["HIP 71683", "GJ 1002"], tap_fn=tap_fn,
                                      endpoint=conf["resolver"]["simbad_tap"], log=log)
        rep["endpoints"]["simbad_ident"] = {"status": STATUS_OK if len(df) else STATUS_ZERO,
                                            "n_rows": int(len(df)),
                                            "rows": df.head(3).to_dict("records")}
    except Exception as exc:                              # noqa: BLE001
        rep["endpoints"]["simbad_ident"] = {"status": STATUS_FAILED, "error": repr(exc)[:400]}
    _save()

    # --- VizieR hit / target catalogues -------------------------------------
    hc = conf["hit_catalogues"]
    tables = acq.discover_hit_tables(hc["seeds"], hc.get("keywords") or (), query_fn=query_fn,
                                     fetch_fn=fetch_fn, log=log, deadline=deadline)
    rep["hit_tables"] = tables
    rep["n_hit_tables"] = sum(1 for t in tables if t.get("kind") == acq.KIND_HITS)
    rep["n_target_tables"] = sum(1 for t in tables if t.get("kind") == acq.KIND_TARGETS)
    rep["verdict"] = _probe_verdict(rep)
    _save()
    print(f"[relay] probe: {rep['verdict']} -- BL targets {len(names)}, telescopes {tels}, "
          f"hit tables {rep['n_hit_tables']}, target tables {rep['n_target_tables']}")
    return rep


def _probe_verdict(rep: dict) -> str:
    parts = []
    parts.append("BL_API_REACHED" if rep.get("bl_targets_n") else "BL_API_NOT_REACHED")
    g = rep["endpoints"].get("gaia_esa_count") or {}
    parts.append("GAIA_REACHED" if g.get("status") == STATUS_OK else "GAIA_NOT_REACHED")
    parts.append(f"HIT_TABLES={rep.get('n_hit_tables', 0)}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# targets: the observed-star list, resolved to Gaia DR3
# ---------------------------------------------------------------------------
def stage_targets(conf: dict, out: Path, *, bl_fetch=None, tap_fn=None, query_fn=None,
                  fetch_fn=None, log=None, probe: dict | None = None) -> dict:
    log = log or AcquisitionLog(prefix="relay/targets")
    bl = acq.BLOpenData(conf["bl_opendata"], fetch_json_fn=bl_fetch, log=log)
    names, rec = bl.list_targets()
    route = "bl_opendata_api"
    tbl_rows = pd.DataFrame()
    if not names:
        # second route: the published target tables (Isaacson+2017, Price+2020 ...)
        probe = probe or _read(out / "probe.json") or {}
        for t in probe.get("hit_tables") or []:
            if t.get("kind") != acq.KIND_TARGETS or t.get("status") != STATUS_OK:
                continue
            rows = acq.fetch_catalogue_rows(t, query_fn=query_fn,
                                            max_rows=conf["hit_catalogues"]["max_rows"], log=log)
            if len(rows):
                tbl_rows = pd.concat([tbl_rows, rows], ignore_index=True)
        if len(tbl_rows) and "target" in tbl_rows:
            names = [str(v) for v in tbl_rows["target"].dropna().astype(str).unique()]
            route = "vizier_target_tables"
            if "hip" in tbl_rows and not names:
                names = [f"HIP {int(v)}" for v in pd.to_numeric(tbl_rows["hip"], errors="coerce").dropna()]
    rep = {"stage": "targets", "generated_utc": _now(), "route": route, "n_names": len(names),
           "bl_list_targets": rec}
    if not names:
        rep["verdict"] = f"{V_NO_DATA} (bl_opendata list-targets and VizieR target tables)"
        rep["n_resolved"] = 0
        rep["acquisition"] = log.as_dict()
        _write(out / "targets.json", rep)
        pd.DataFrame(columns=["name", "main_id", "ra", "dec", "plx", "gaia_dr3", "resolved_by"]
                     ).to_csv(out / "targets.csv", index=False)
        print(f"[relay] targets: {rep['verdict']}")
        return rep
    res = acq.resolve_names_simbad(names, tap_fn=tap_fn, endpoint=conf["resolver"]["simbad_tap"],
                                   batch=int(conf["resolver"].get("batch", 120)), log=log)
    df = pd.DataFrame({"name": names})
    df["key"] = df["name"].map(name_key)
    if len(res):
        res = res.copy()
        res["key"] = res["name"].map(name_key)
        res = res.drop_duplicates("key")
        df = df.merge(res.drop(columns=["name"]), on="key", how="left")
    for c in ("main_id", "ra", "dec", "plx", "gaia_dr3"):
        if c not in df:
            df[c] = np.nan
    df["resolved_by"] = np.where(df["gaia_dr3"].astype(str).str.len().gt(0)
                                 & df["gaia_dr3"].notna(), "simbad_gaia_id",
                                 np.where(df["ra"].notna(), "simbad_position", "unresolved"))
    # positions from the VizieR target tables for names SIMBAD did not know
    if len(tbl_rows) and {"ra", "dec"} <= set(tbl_rows.columns):
        tbl_rows = tbl_rows.copy()
        tbl_rows["key"] = tbl_rows["target"].map(name_key)
        pos = tbl_rows.drop_duplicates("key").set_index("key")
        miss = df["resolved_by"] == "unresolved"
        for i in df.index[miss]:
            k = df.at[i, "key"]
            if k in pos.index:
                df.at[i, "ra"] = pd.to_numeric(pos.at[k, "ra"], errors="coerce")
                df.at[i, "dec"] = pd.to_numeric(pos.at[k, "dec"], errors="coerce")
                df.at[i, "resolved_by"] = "catalogue_position"
    df = df.drop(columns=["key", "asked"], errors="ignore")
    df.to_csv(out / "targets.csv", index=False)
    counts = df["resolved_by"].value_counts().to_dict()
    rep.update({"n_resolved": int((df["resolved_by"] != "unresolved").sum()),
                "resolved_by": counts, "n_with_gaia_id": int(counts.get("simbad_gaia_id", 0)),
                "unresolved_sample": df.loc[df["resolved_by"] == "unresolved", "name"].head(30).tolist(),
                "acquisition": log.as_dict()})
    rep["verdict"] = f"{V_TARGETS} ({rep['n_resolved']}/{len(names)} via {route})"
    _write(out / "targets.json", rep)
    print(f"[relay] targets: {rep['verdict']} {counts}")
    return rep


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _prepare_sample(df: pd.DataFrame, conf: dict) -> pd.DataFrame:
    """Standard columns, RV zero-point, quality cut on RVs, velocities attached."""
    df = df.copy().reset_index(drop=True)
    for c in acq.GAIA_STANDARD:
        if c not in df:
            df[c] = np.nan
    if "grvs_mag" in df and df["grvs_mag"].notna().any():
        df = apply_rv_zero_point(df)
    rv = pd.to_numeric(df["radial_velocity"], errors="coerce").to_numpy(float)
    rve = pd.to_numeric(df["radial_velocity_error"], errors="coerce").to_numpy(float)
    teff = pd.to_numeric(df["rv_template_teff"], errors="coerce").to_numpy(float)
    bad = (~np.isfinite(rv)) | (np.isfinite(rve) & (rve > float(conf["sample"]["rv_err_max_kms"])))
    bad |= np.isfinite(teff) & (teff > float(conf["sample"]["rv_template_teff_max_k"]))
    rv = np.where(bad, np.nan, rv)
    df["rv_used_kms"] = rv
    df["has_rv"] = np.isfinite(rv)
    d = 1000.0 / pd.to_numeric(df["parallax"], errors="coerce").to_numpy(float)
    df["d_pc"] = d
    return df


def _sample_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    xyz = geo.positions_pc(df["ra"].to_numpy(float), df["dec"].to_numpy(float),
                           df["parallax"].to_numpy(float))
    vel = geo.space_velocities(df["ra"].to_numpy(float), df["dec"].to_numpy(float),
                               df["parallax"].to_numpy(float), df["pmra"].to_numpy(float),
                               df["pmdec"].to_numpy(float), df["rv_used_kms"].to_numpy(float))
    return xyz, vel


def match_targets_to_sample(targets: pd.DataFrame, sample: pd.DataFrame, *,
                            match_arcsec: float = 5.0) -> pd.DataFrame:
    """Gaia row index per target: by DR3 id, else by J2000 position after PM propagation."""
    out = targets.copy()
    out["gaia_idx"] = -1
    out["match_by"] = "none"
    if not len(sample):
        return out
    by_id = {str(s): i for i, s in enumerate(sample["source_id"].astype(str))}
    ids = out.get("gaia_dr3")
    if ids is not None:
        for i, v in enumerate(ids):
            k = acq._clean_source_id(v)
            if k and k in by_id:
                out.iat[i, out.columns.get_loc("gaia_idx")] = by_id[k]
                out.iat[i, out.columns.get_loc("match_by")] = "gaia_id"
    need = (out["gaia_idx"] < 0) & pd.to_numeric(out.get("ra"), errors="coerce").notna() \
        & pd.to_numeric(out.get("dec"), errors="coerce").notna()
    if need.any():
        from scipy.spatial import cKDTree

        # propagate the Gaia (2016.0) positions to J2000 for the SIMBAD/catalogue positions
        dt = 2000.0 - 2016.0
        dec = sample["dec"].to_numpy(float)
        pmra = np.nan_to_num(sample["pmra"].to_numpy(float))
        pmdec = np.nan_to_num(sample["pmdec"].to_numpy(float))
        ra2000 = sample["ra"].to_numpy(float) + dt * pmra / 3.6e6 / np.maximum(np.cos(np.radians(dec)), 1e-6)
        dec2000 = dec + dt * pmdec / 3.6e6
        tree = cKDTree(geo.unit_vectors(ra2000, dec2000))
        q = geo.unit_vectors(pd.to_numeric(out.loc[need, "ra"]).to_numpy(float),
                             pd.to_numeric(out.loc[need, "dec"]).to_numpy(float))
        dist, idx = tree.query(q, k=1)
        ok = dist <= geo.chord(match_arcsec * geo.ARCSEC)
        rows = np.flatnonzero(need.to_numpy())
        for r, i, o in zip(rows, idx, ok, strict=False):
            if o:
                out.iat[r, out.columns.get_loc("gaia_idx")] = int(i)
                out.iat[r, out.columns.get_loc("match_by")] = "position"
    return out


def in_beam_neighbours(sample: pd.DataFrame, centre_idx, radius_arcmin: float) -> dict[int, list[int]]:
    """Sample rows within ``radius_arcmin`` of each centre row (centre excluded)."""
    from scipy.spatial import cKDTree

    u = geo.unit_vectors(sample["ra"].to_numpy(float), sample["dec"].to_numpy(float))
    tree = cKDTree(u)
    out = {}
    for c in centre_idx:
        c = int(c)
        if c < 0:
            continue
        near = tree.query_ball_point(u[c], geo.chord(radius_arcmin * geo.ARCMIN))
        out[c] = sorted(int(i) for i in near if i != c)
    return out


def stage_geometry(conf: dict, out: Path, *, tap_fn=None, query_fn=None, fetch_fn=None,
                   gaia_df: pd.DataFrame | None = None, max_stars: int | None = None,
                   beams=None, log=None, progress_every: float = 60.0) -> dict:
    """Acquire the 100 pc sample and enumerate every qualifying directed pair per beam."""
    log = log or AcquisitionLog(prefix="relay/geometry")
    gconf = conf["geometry"]
    started = time.monotonic()

    # --- the sample ---------------------------------------------------------
    if gaia_df is None:
        acqn = acq.fetch_gaia_sample(conf["sample"], tap_fn=tap_fn, query_fn=query_fn, log=log,
                                     max_rows=max_stars, fetch_fn=fetch_fn)
        gaia_df, sample_rec = acqn.df, acqn.as_dict()
    else:
        sample_rec = {"route": "injected", "status": STATUS_OK if len(gaia_df) else STATUS_FAILED,
                      "n_rows": int(len(gaia_df))}
    if max_stars and len(gaia_df) > max_stars:
        gaia_df = gaia_df.iloc[:int(max_stars)]
    sample_rec["generated_utc"] = _now()
    _write(out / "gaia_sample.json", sample_rec)
    if not len(gaia_df):
        rep = {"stage": "geometry", "generated_utc": _now(), "sample": sample_rec,
               "verdict": f"{V_NO_DATA} (gaia: {sample_rec.get('route')})", "beams": {},
               "n_stars": 0, "acquisition": log.as_dict()}
        _write(out / "geometry.json", rep)
        print(f"[relay] geometry: {rep['verdict']}")
        return rep
    sample = _prepare_sample(acq.canonicalise_gaia(gaia_df) if "d_pc" not in gaia_df else gaia_df,
                             conf)
    d_ok = np.isfinite(sample["d_pc"]) & (sample["d_pc"] > 0) & \
        (sample["d_pc"] <= float(conf["sample"]["d_max_pc"]))
    sample = sample[d_ok].reset_index(drop=True)
    try:
        sample.to_parquet(out / "gaia_sample.parquet", index=False)
    except Exception as exc:                              # noqa: BLE001
        print(f"[relay] parquet unavailable ({exc!r}); keeping the sample as csv.gz")
        _write_csv_gz(sample, out / "gaia_sample.csv.gz")
    n = len(sample)
    xyz, vel = _sample_arrays(sample)

    # --- targets and the keep set ---------------------------------------------
    tpath = out / "targets.csv"
    targets = pd.read_csv(tpath) if tpath.exists() else pd.DataFrame()
    keep = None
    matched = pd.DataFrame()
    n_targets_matched = 0
    neighbours: dict[int, list[int]] = {}
    if len(targets):
        matched = match_targets_to_sample(targets, sample,
                                          match_arcsec=float(conf["resolver"].get("match_arcsec", 5)))
        matched.to_csv(out / "targets_matched.csv", index=False)
        tidx = [int(i) for i in matched["gaia_idx"] if i >= 0]
        n_targets_matched = len(tidx)
        neighbours = in_beam_neighbours(sample, tidx,
                                        float(conf["telescopes"].get("in_beam_max_arcmin", 8.0)))
        keep = set(tidx) | {j for js in neighbours.values() for j in js}
    beams = beams or geo.beam_grid(conf["beams"])
    # narrowest first: the cheap, scientifically central beams must always finish,
    # and a wide beam that overruns the clock is recorded, never guessed at
    beams = sorted(beams, key=lambda b: float(b["theta_rad"]))
    budget = float(gconf.get("budget_s", 9000))
    deadline = started + budget
    rep = {"stage": "geometry", "generated_utc": _now(), "sample": sample_rec, "n_stars": n,
           "n_with_rv": int(sample["has_rv"].sum()),
           "rv_completeness": float(sample["has_rv"].mean()) if n else 0.0,
           "d_max_pc": float(conf["sample"]["d_max_pc"]),
           "n_targets": int(len(targets)), "n_targets_matched": n_targets_matched,
           "n_keep_transmitters": int(len(keep)) if keep is not None else n,
           "min_separation_pc": float(gconf.get("min_separation_pc", 0.0)),
           "beam_budget_s": budget, "beams": {}, "committed_files": []}

    def _save():
        rep["elapsed_s"] = round(time.monotonic() - started, 1)
        rep["acquisition"] = log.as_dict()
        _write(out / "geometry.json", rep)

    _save()
    target_frames = []
    for b in beams:
        name = _beam_name(b)
        th = float(b["theta_rad"])
        t0 = time.monotonic()
        if t0 >= deadline:
            rep["beams"][name] = {**dict(b), "status": "NOT_COMPUTED",
                                  "reason": f"geometry wall clock spent ({budget:.0f} s)"}
            print(f"[relay] {name}: NOT COMPUTED -- geometry wall clock spent")
            _save()
            continue
        last = [t0]

        def _prog(done, total, ns, nb, name=name, last=last):
            if time.monotonic() - last[0] >= progress_every:
                last[0] = time.monotonic()
                print(f"[relay] {name}: {done}/{total} receivers, {ns} spillover, {nb} between")

        res = geo.find_pairs(xyz, th, d_max_pc=float(conf["sample"]["d_max_pc"]),
                             keep_transmitters=keep, row_cap=int(gconf.get("row_cap", 4_000_000)),
                             chunk_candidates=int(gconf.get("chunk_candidates", 3_000_000)),
                             min_separation_pc=float(gconf.get("min_separation_pc", 0.0)),
                             progress=_prog, deadline=deadline)
        pairs = res.pairs
        if len(pairs):
            kin = geo.pair_kinematics(xyz, vel, pairs["t_idx"].to_numpy(int), pairs["r_idx"].to_numpy(int))
            pairs = pd.concat([pairs.reset_index(drop=True), kin], axis=1)
            pairs["t_source_id"] = sample["source_id"].to_numpy()[pairs["t_idx"].to_numpy(int)]
            pairs["r_source_id"] = sample["source_id"].to_numpy()[pairs["r_idx"].to_numpy(int)]
            pairs["beam"] = name
            f_ref = float(conf["drift"]["reference_frequency_ghz"]) * 1e9
            pairs["drift_kin_hz_s_at_ref"] = geo.drift_hz_s(f_ref, pairs["a_kin_m_s2"].to_numpy(float))
        exp = geo.analytic_expectation(th, n)
        per_t = res.per_transmitter_spill + res.per_transmitter_between
        brec = {**{k: v for k, v in b.items()}, **res.as_dict(),
                "status": "COMPUTED" if res.counts_complete else "PARTIAL_COUNTS",
                "analytic_expectation": exp,
                # a count cut short by the clock covers only part of the receiver
                # list, so it is NOT comparable with the whole-sample expectation
                "measured_over_analytic": {
                    "spillover": (res.n_spillover / exp["spillover"])
                    if (exp["spillover"] and res.counts_complete) else None,
                    "between": (res.n_between / exp["between"])
                    if (exp["between"] and res.counts_complete) else None},
                "n_transmitters_with_any_pair": int((per_t > 0).sum()),
                "n_transmitters_spillover": int((res.per_transmitter_spill > 0).sum()),
                "n_transmitters_between": int((res.per_transmitter_between > 0).sum()),
                "n_kept_with_kinematics": int(pairs["kinematics_complete"].sum()) if len(pairs) else 0,
                "elapsed_s": round(time.monotonic() - t0, 1)}
        if len(pairs):
            brec["alpha_over_half_theta_p50"] = float(pairs["alpha_over_half_theta"].median())
            brec["flux_ratio_p50_spillover"] = _median(pairs, "flux_ratio_earth_over_receiver",
                                                       pairs["geometry"] == geo.GEOM_SPILLOVER)
            brec["flux_ratio_p50_between"] = _median(pairs, "flux_ratio_earth_over_receiver",
                                                     pairs["geometry"] == geo.GEOM_BETWEEN)
            brec["a_kin_abs_p50_m_s2"] = float(np.nanmedian(np.abs(pairs["a_kin_m_s2"])))
            brec["drift_kin_abs_p99_hz_s_at_ref"] = float(np.nanpercentile(
                np.abs(pairs["drift_kin_hz_s_at_ref"].dropna()), 99)) if pairs["kinematics_complete"].any() else None
        if keep is not None and len(matched):
            tset = set(int(i) for i in matched["gaia_idx"] if i >= 0)
            brec["n_bl_targets_as_transmitter"] = int(sum(1 for i in tset if per_t[i] > 0))
            brec["n_bl_targets_spillover"] = int(sum(1 for i in tset if res.per_transmitter_spill[i] > 0))
            brec["n_bl_targets_between"] = int(sum(1 for i in tset if res.per_transmitter_between[i] > 0))
        rep["beams"][name] = brec
        # per-transmitter counts for every star travel in the artifact
        try:
            pairs.to_parquet(out / f"pairs_{name}.parquet", index=False)
        except Exception:                                 # noqa: BLE001
            _write_csv_gz(pairs, out / f"pairs_{name}.csv.gz")
        # commit the whole catalogue of a beam narrow enough to be small
        total = res.n_spillover + res.n_between
        if keep is None and not res.truncated and total <= int(gconf["commit_full_catalogue_max_rows"]):
            _write_csv_gz(pairs, out / f"pairs_full_{name}.csv.gz")
            rep["committed_files"].append(f"pairs_full_{name}.csv.gz")
        elif len(pairs) and total <= int(gconf["commit_full_catalogue_max_rows"]) and not res.truncated:
            # the kept rows ARE the whole catalogue only when the keep set covers it;
            # otherwise re-run for the narrow beam without a keep set (cheap)
            res_all = geo.find_pairs(xyz, th, d_max_pc=float(conf["sample"]["d_max_pc"]),
                                     row_cap=int(gconf["commit_full_catalogue_max_rows"]),
                                     min_separation_pc=float(gconf.get("min_separation_pc", 0.0)))
            full = res_all.pairs
            if len(full):
                full["t_source_id"] = sample["source_id"].to_numpy()[full["t_idx"].to_numpy(int)]
                full["r_source_id"] = sample["source_id"].to_numpy()[full["r_idx"].to_numpy(int)]
                kin = geo.pair_kinematics(xyz, vel, full["t_idx"].to_numpy(int), full["r_idx"].to_numpy(int))
                full = pd.concat([full.reset_index(drop=True), kin], axis=1)
            _write_csv_gz(full, out / f"pairs_full_{name}.csv.gz")
            rep["committed_files"].append(f"pairs_full_{name}.csv.gz")
        if len(pairs) and keep is not None:
            target_frames.append(pairs)
        per_t_df = pd.DataFrame({"source_id": sample["source_id"], "beam": name,
                                 "n_spillover": res.per_transmitter_spill,
                                 "n_between": res.per_transmitter_between})
        per_t_df = per_t_df[(per_t_df["n_spillover"] > 0) | (per_t_df["n_between"] > 0)]
        try:
            per_t_df.to_parquet(out / f"per_transmitter_{name}.parquet", index=False)
        except Exception:                                 # noqa: BLE001
            _write_csv_gz(per_t_df, out / f"per_transmitter_{name}.csv.gz")
        print(f"[relay] {name} ({b['theta_label']}): {res.n_spillover} spillover + "
              f"{res.n_between} between directed pairs (analytic {exp['spillover']:.3g} + "
              f"{exp['between']:.3g}); kept {len(pairs)} rows in {brec['elapsed_s']} s")
        _save()

    # the committed BL-target pair table, widest beams trimmed first
    if target_frames:
        tp = pd.concat(target_frames, ignore_index=True)
        cap = int(gconf["commit_target_pairs_max_rows"])
        if len(tp) > cap:
            keep_rows = []
            room = cap
            for b in beams:
                sub = tp[tp["beam"] == _beam_name(b)]
                take = sub.iloc[:room]
                keep_rows.append(take)
                room -= len(take)
                if room <= 0:
                    break
            tp_c = pd.concat(keep_rows, ignore_index=True)
            rep["target_pairs_committed_truncated"] = True
        else:
            tp_c = tp
        _write_csv_gz(tp_c, out / "pairs_targets.csv.gz")
        rep["committed_files"].append("pairs_targets.csv.gz")
        rep["n_target_pair_rows"] = int(len(tp))
    rep["scaling"] = _scaling_fit(rep["beams"])
    n_done = sum(1 for v in rep["beams"].values() if v.get("status") in (None, "COMPUTED"))
    rep["n_beams_computed"] = n_done
    rep["verdict"] = (V_GEOMETRY_PARTIAL if sample_rec.get("status") == acq.STATUS_PARTIAL
                      else V_GEOMETRY) + f" ({n} stars, {n_done}/{len(beams)} beams complete)"
    if n_done < len(beams):
        rep["verdict"] += " | BEAMS_INCOMPLETE (geometry wall clock)"
    _save()
    print(f"[relay] geometry: {rep['verdict']} in {rep['elapsed_s']} s")
    return rep


def _median(df, col, mask) -> float | None:
    v = df.loc[mask, col]
    return float(v.median()) if len(v) else None


def _scaling_fit(beams: dict) -> dict:
    """Slope of log N vs log theta over the beams with non-zero counts (expect ~2)."""
    out = {}
    for geom_key in ("n_spillover", "n_between"):
        xs, ys = [], []
        for b in beams.values():
            # a partial or uncomputed beam's count is not on the theta^2 curve
            if b.get("status") not in (None, "COMPUTED"):
                continue
            if b.get(geom_key, 0) > 0:
                xs.append(math.log10(b["theta_rad"]))
                ys.append(math.log10(b[geom_key]))
        if len(xs) >= 2:
            slope = float(np.polyfit(xs, ys, 1)[0])
            out[geom_key] = {"slope_log_n_log_theta": slope, "n_points": len(xs)}
        else:
            out[geom_key] = {"slope_log_n_log_theta": None, "n_points": len(xs)}
    return out


# ---------------------------------------------------------------------------
# recut: which observed stars sit on pair lines, and the drift window
# ---------------------------------------------------------------------------
def _load_pairs(out: Path, beams) -> dict[str, pd.DataFrame]:
    res = {}
    for b in beams:
        name = _beam_name(b)
        p = out / f"pairs_{name}.parquet"
        if p.exists():
            res[name] = pd.read_parquet(p)
        elif (out / f"pairs_{name}.csv.gz").exists():
            res[name] = _read_csv_gz(out / f"pairs_{name}.csv.gz")
        else:
            res[name] = pd.DataFrame()
    return res


def _load_sample(out: Path) -> pd.DataFrame:
    p = out / "gaia_sample.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return _read_csv_gz(out / "gaia_sample.csv.gz")


def pointing_windows(pairs: pd.DataFrame, t_indices, *, freq_hz: float, mjd: float,
                     site: dict | None, alpha_col: str = "alpha_rad", conf_drift: dict,
                     sigma_drift: float, sample: pd.DataFrame) -> dict:
    """Pair counts and the drift window for one pointing over a set of transmitters."""
    sub = pairs[pairs["t_idx"].isin(list(t_indices))] if len(pairs) else pairs
    known = bool(np.isfinite(mjd)) and site is not None
    rec = {"n_pairs": int(len(sub)),
           "n_spillover": int((sub["geometry"] == geo.GEOM_SPILLOVER).sum()) if len(sub) else 0,
           "n_between": int((sub["geometry"] == geo.GEOM_BETWEEN).sum()) if len(sub) else 0,
           "earth_term_known": known}
    if not len(sub):
        return rec
    ra = sample["ra"].to_numpy(float)[sub["t_idx"].to_numpy(int)]
    dec = sample["dec"].to_numpy(float)[sub["t_idx"].to_numpy(int)]
    if known:
        e = geo.earth_acceleration_los(mjd, site["lat_deg"], site["lon_deg"], ra, dec)
        a_e = e["a_los_m_s2"]
    else:
        a_e = geo.earth_term_bounds(site["lat_deg"] if site else 0.0, dec)
    w = geo.drift_window(freq_hz, sub["a_kin_m_s2"].to_numpy(float), a_e,
                         sub[alpha_col].to_numpy(float),
                         a_local_max_m_s2=float(conf_drift["a_local_max_m_s2"]),
                         sigma_drift_hz_s=sigma_drift, earth_term_known=known)
    rec.update({
        "drift_centre_hz_s_min": float(np.min(w["centre_hz_s"])),
        "drift_centre_hz_s_max": float(np.max(w["centre_hz_s"])),
        "halfwidth_tight_hz_s_min": float(np.min(w["halfwidth_tight_hz_s"])),
        "halfwidth_tight_hz_s_max": float(np.max(w["halfwidth_tight_hz_s"])),
        "halfwidth_loose_hz_s_max": float(np.max(w["halfwidth_loose_hz_s"])),
        "n_pairs_with_kinematics": int(sub["kinematics_complete"].sum()),
        "alpha_rad_min": float(sub[alpha_col].min()),
        "flux_ratio_max": float(sub["flux_ratio_earth_over_receiver"].max()),
    })
    return rec


def stage_recut(conf: dict, out: Path, *, bl_fetch=None, log=None, beams=None,
                budget_s: float | None = None) -> dict:
    log = log or AcquisitionLog(prefix="relay/recut")
    beams = beams or geo.beam_grid(conf["beams"])
    started = time.monotonic()
    budget = float(budget_s or conf["bl_opendata"].get("files_budget_s", 5400))
    tconf = conf["telescopes"]
    sample = _load_sample(out)
    mpath = out / "targets_matched.csv"
    matched = pd.read_csv(mpath) if mpath.exists() else pd.DataFrame()
    pairs = _load_pairs(out, beams)
    geom_beams = ((_read(out / "geometry.json") or {}).get("beams") or {})
    rep = {"stage": "recut", "generated_utc": _now(), "n_targets": int(len(matched)),
           "n_targets_in_sample": int((matched["gaia_idx"] >= 0).sum()) if len(matched) else 0,
           "beams": {}, "pointings": {}}
    if not len(sample) or not len(matched) or not any(len(p) for p in pairs.values()):
        missing = [k for k, v in (("gaia_sample", len(sample)), ("targets", len(matched)),
                                  ("pairs", sum(len(p) for p in pairs.values()))) if not v]
        rep["verdict"] = f"{V_NO_DATA} (recut inputs missing: {', '.join(missing)})"
        rep["acquisition"] = log.as_dict()
        _write(out / "recut.json", rep)
        pd.DataFrame().to_csv(out / "recut_pointings.csv", index=False)
        print(f"[relay] recut: {rep['verdict']}")
        return rep
    in_sample = matched[matched["gaia_idx"] >= 0].reset_index(drop=True)
    tidx = in_sample["gaia_idx"].astype(int).tolist()
    neighbours = in_beam_neighbours(sample, tidx, float(tconf.get("in_beam_max_arcmin", 8.0)))

    # --- per target: pair-line membership as transmitter, per beam ------------
    per_target_rows = []
    for _, t in in_sample.iterrows():
        i = int(t["gaia_idx"])
        row = {"name": t["name"], "gaia_idx": i, "source_id": sample["source_id"].iat[i],
               "ra": sample["ra"].iat[i], "dec": sample["dec"].iat[i], "d_pc": sample["d_pc"].iat[i],
               "has_rv": bool(sample["has_rv"].iat[i]), "n_in_beam_neighbours": len(neighbours.get(i, []))}
        for b in beams:
            name = _beam_name(b)
            p = pairs[name]
            if len(p):
                mine = p[p["t_idx"] == i]
                row[f"{name}:n_spillover"] = int((mine["geometry"] == geo.GEOM_SPILLOVER).sum())
                row[f"{name}:n_between"] = int((mine["geometry"] == geo.GEOM_BETWEEN).sum())
                nb = neighbours.get(i, [])
                theirs = p[p["t_idx"].isin(nb)] if nb else p.iloc[0:0]
                row[f"{name}:n_in_beam_neighbour_pairs"] = int(len(theirs))
            else:
                row[f"{name}:n_spillover"] = row[f"{name}:n_between"] = 0
                row[f"{name}:n_in_beam_neighbour_pairs"] = 0
        per_target_rows.append(row)
    per_target = pd.DataFrame(per_target_rows)
    for b in beams:
        name = _beam_name(b)
        sp, bt = per_target.get(f"{name}:n_spillover", 0), per_target.get(f"{name}:n_between", 0)
        nb = per_target.get(f"{name}:n_in_beam_neighbour_pairs", 0)
        rep["beams"][name] = {
            "theta_label": b["theta_label"],
            # a beam the geometry stage never finished has no pairs to recut on:
            # its zeros are the absence of a search, not the absence of a pair
            "geometry_status": geom_beams.get(name, {}).get("status", "COMPUTED"),
            "n_targets_as_transmitter_spillover": int((sp > 0).sum()),
            "n_targets_as_transmitter_between": int((bt > 0).sum()),
            "n_targets_as_transmitter_any": int(((sp > 0) | (bt > 0)).sum()),
            "n_targets_with_in_beam_transmitter": int((nb > 0).sum()),
            "n_target_pairs_spillover": int(sp.sum()), "n_target_pairs_between": int(bt.sum()),
        }
    _write(out / "recut.json", rep)

    # --- the BL files per target: telescope, band, epoch --------------------------
    bl = acq.BLOpenData(conf["bl_opendata"], fetch_json_fn=bl_fetch, log=log)
    files_rows = []
    n_asked = n_answered = 0
    cap = int(conf["bl_opendata"].get("max_targets_for_files", 2500))
    sleep_s = float(conf["bl_opendata"].get("request_sleep_s", 0.1))
    order = per_target.copy()
    any_cols = [c for c in order.columns if c.endswith(":n_spillover") or c.endswith(":n_between")]
    order["_score"] = order[any_cols].sum(axis=1) if any_cols else 0
    order = order.sort_values("_score", ascending=False)
    for _, t in order.iterrows():
        if n_asked >= cap or time.monotonic() - started > budget:
            break
        n_asked += 1
        files, rec = bl.query_files(str(t["name"]))
        if rec.get("status") == STATUS_FAILED:
            continue
        n_answered += 1
        for f in files:
            f["gaia_idx"] = int(t["gaia_idx"])
            f["name"] = f.get("name")
            f["target"] = str(t["name"])
            files_rows.append(f)
        if sleep_s:
            bl.sleep(sleep_s)
    files_df = pd.DataFrame(files_rows)
    rep["bl_files"] = {"n_targets_asked": n_asked, "n_targets_answered": n_answered,
                       "n_files": int(len(files_df)),
                       "elapsed_s": round(time.monotonic() - started, 1)}
    if len(files_df):
        files_df["telescope_key"] = files_df["telescope"].map(lambda v: _telescope_key(v, tconf))
        files_df["band"] = [_band_of(fc, hint, tconf) for fc, hint in
                            zip(files_df["freq_centre_ghz"], files_df["band_hint"], strict=False)]
        _write_csv_gz(files_df, out / "bl_files.csv.gz")
        rep["bl_files"]["telescopes"] = files_df["telescope_key"].fillna("unknown").value_counts().to_dict()
        rep["bl_files"]["bands"] = files_df["band"].fillna("unknown").value_counts().to_dict()
        rep["bl_files"]["n_with_mjd"] = int(np.isfinite(files_df["mjd"]).sum())
        rep["bl_files"]["telescope_names_raw"] = files_df["telescope"].fillna("").value_counts().head(20).to_dict()
    _write(out / "recut.json", rep)

    # --- pointings: (target, telescope, band, session) with pair counts and windows --
    pointing_rows = []
    if len(files_df):
        files_df["session"] = np.floor(files_df["mjd"].fillna(-1) * 24.0) / 24.0
        grp = files_df.groupby(["target", "gaia_idx", "telescope_key", "band", "session"], dropna=False)
        for (target, gidx, tel, band, session), g in grp:
            gidx = int(gidx)
            tel = tel if isinstance(tel, str) else None
            band = band if isinstance(band, str) else None
            site = tconf.get(tel) if tel else None
            hp = _hpbw_arcmin(tel, band, tconf)
            fc = float(np.nanmedian(g["freq_centre_ghz"])) if np.isfinite(g["freq_centre_ghz"]).any() \
                else float(conf["drift"]["reference_frequency_ghz"])
            mjd = float(session) if session is not None and session >= 0 else float("nan")
            base = {"target": target, "gaia_idx": gidx, "source_id": sample["source_id"].iat[gidx],
                    "telescope": tel, "band": band, "mjd_session": mjd, "n_files": int(len(g)),
                    "freq_centre_ghz": fc, "hpbw_arcmin": hp}
            near = [j for j in neighbours.get(gidx, [])
                    if np.isfinite(hp) and _sep_arcmin(sample, gidx, j) <= 0.5 * hp]
            base["n_in_beam_neighbours"] = len(near)
            tset = [gidx] + near
            for b in beams:
                name = _beam_name(b)
                w = pointing_windows(pairs[name], tset, freq_hz=fc * 1e9, mjd=mjd, site=site,
                                     conf_drift=conf["drift"],
                                     sigma_drift=float(conf["drift"]["sigma_drift_default_hz_s"]),
                                     sample=sample)
                for k, v in w.items():
                    base[f"{name}:{k}"] = v
            pointing_rows.append(base)
    pointings = pd.DataFrame(pointing_rows)
    pointings.to_csv(out / "recut_pointings.csv", index=False)
    per_target.to_csv(out / "recut_targets.csv", index=False)
    for b in beams:
        name = _beam_name(b)
        if len(pointings):
            col = f"{name}:n_pairs"
            rep["beams"][name]["n_pointings_on_pair_line"] = int((pointings[col] > 0).sum())
            rep["beams"][name]["n_pointings_with_epoch"] = int(
                ((pointings[col] > 0) & pointings[f"{name}:earth_term_known"].fillna(False)).sum())
        else:
            rep["beams"][name]["n_pointings_on_pair_line"] = 0
    rep["n_pointings"] = int(len(pointings))
    if len(files_df):
        rep["verdict"] = (f"{V_RECUT} ({n_answered} targets with files, {len(files_df)} files, "
                          f"{len(pointings)} pointings)")
    else:
        rep["verdict"] = (f"{V_RECUT}_GEOMETRY_ONLY ({V_NO_DATA} for BL files: "
                          f"{n_asked} asked, {n_answered} answered)")
    rep["elapsed_s"] = round(time.monotonic() - started, 1)
    rep["acquisition"] = log.as_dict()
    _write(out / "recut.json", rep)
    print(f"[relay] recut: {rep['verdict']}")
    return rep


def _sep_arcmin(sample: pd.DataFrame, i: int, j: int) -> float:
    u = geo.unit_vectors(sample["ra"].to_numpy(float)[[i, j]], sample["dec"].to_numpy(float)[[i, j]])
    return float(geo.sky_separation_deg(u[0], u[1]) * 60.0)


# ---------------------------------------------------------------------------
# assess: the published hits against the geometry and the prior
# ---------------------------------------------------------------------------
def stage_assess(conf: dict, out: Path, *, query_fn=None, fetch_fn=None, tap_fn=None, log=None,
                 beams=None, hits_df: pd.DataFrame | None = None, probe: dict | None = None
                 ) -> dict:
    log = log or AcquisitionLog(prefix="relay/assess")
    beams = beams or geo.beam_grid(conf["beams"])
    tconf = conf["telescopes"]
    aconf = conf["assess"]
    started = time.monotonic()
    sample = _load_sample(out)
    pairs = _load_pairs(out, beams)
    mpath = out / "targets_matched.csv"
    matched = pd.read_csv(mpath) if mpath.exists() else pd.DataFrame()
    geom = _read(out / "geometry.json") or {}
    recut = _read(out / "recut.json") or {}

    # --- the hit catalogues ---------------------------------------------------
    tables = []
    if hits_df is None:
        probe = probe or _read(out / "probe.json") or {}
        tables = [t for t in probe.get("hit_tables") or []
                  if t.get("kind") == acq.KIND_HITS and t.get("status") == STATUS_OK]
        frames = []
        for t in tables:
            rows = acq.fetch_catalogue_rows(t, query_fn=query_fn or tap_query,
                                            max_rows=int(conf["hit_catalogues"]["max_rows"]), log=log)
            t["n_rows_fetched"] = int(len(rows))
            if len(rows):
                frames.append(acq.standardise_hits(rows))
        hits_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        hits_df = acq.standardise_hits(hits_df) if "drift_hz_s" not in hits_df else hits_df
    rep = {"stage": "assess", "generated_utc": _now(), "hit_tables": tables,
           "n_hits": int(len(hits_df)), "beams": {}}
    if not len(hits_df):
        rep["verdict"] = V_NO_HIT_CATALOGUE
        rep["acquisition"] = log.as_dict()
        _write(out / "hits.json", rep)
        pd.DataFrame().to_csv(out / "hits_crossmatch.csv", index=False)
        return _finish(conf, out, rep, geom, recut, beams, started)
    if not len(sample) or not any(len(p) for p in pairs.values()):
        rep["verdict"] = f"{V_NO_DATA} (hits reached but no geometry to cut them on)"
        rep["acquisition"] = log.as_dict()
        _write(out / "hits.json", rep)
        hits_df.to_csv(out / "hits_crossmatch.csv", index=False)
        return _finish(conf, out, rep, geom, recut, beams, started)

    # --- hit -> star ------------------------------------------------------------
    hits = hits_df.copy().reset_index(drop=True)
    hits["gaia_idx"] = -1
    hits["match_by"] = "none"
    key_to_idx = {}
    if len(matched):
        for _, r in matched.iterrows():
            if r["gaia_idx"] >= 0:
                key_to_idx.setdefault(name_key(r["name"]), int(r["gaia_idx"]))
                if isinstance(r.get("main_id"), str):
                    key_to_idx.setdefault(name_key(r["main_id"]), int(r["gaia_idx"]))
    for i, v in enumerate(hits["target"]):
        k = name_key(v)
        if k and k in key_to_idx:
            hits.at[i, "gaia_idx"] = key_to_idx[k]
            hits.at[i, "match_by"] = "target_name"
    unresolved = hits[(hits["gaia_idx"] < 0) & hits["target"].notna()]["target"].astype(str).unique()
    if len(unresolved) and tap_fn is not None:
        res = acq.resolve_names_simbad(list(unresolved), tap_fn=tap_fn,
                                       endpoint=conf["resolver"]["simbad_tap"], log=log)
        if len(res):
            tmp = match_targets_to_sample(res, sample, match_arcsec=float(conf["resolver"].get("match_arcsec", 5)))
            for _, r in tmp.iterrows():
                if r["gaia_idx"] >= 0:
                    key_to_idx.setdefault(name_key(r["name"]), int(r["gaia_idx"]))
            for i, v in enumerate(hits["target"]):
                if hits.at[i, "gaia_idx"] < 0 and name_key(v) in key_to_idx:
                    hits.at[i, "gaia_idx"] = key_to_idx[name_key(v)]
                    hits.at[i, "match_by"] = "simbad"
    need = (hits["gaia_idx"] < 0) & hits["ra"].notna() & hits["dec"].notna()
    if need.any():
        pos = match_targets_to_sample(hits.loc[need, ["ra", "dec"]].assign(name=""), sample,
                                      match_arcsec=float(tconf.get("in_beam_max_arcmin", 8.0)) * 60.0 / 2)
        for r, gi in zip(np.flatnonzero(need.to_numpy()), pos["gaia_idx"], strict=False):
            if gi >= 0:
                hits.at[r, "gaia_idx"] = int(gi)
                hits.at[r, "match_by"] = "position"
    rep["n_hits_matched_to_sample"] = int((hits["gaia_idx"] >= 0).sum())
    rep["match_by"] = hits["match_by"].value_counts().to_dict()

    # --- RFI rules: zero drift, recurrence across sightlines ---------------------------
    zero = float(conf["hit_catalogues"].get("zero_drift_rfi_hz_s", 0.0093))
    hits["rfi_zero_drift"] = np.abs(hits["drift_hz_s"].fillna(0.0)) < zero
    bins = np.round(hits["freq_mhz"].to_numpy(float) * 1e3 / float(aconf["recurrence_khz"]))
    hits["_fbin"] = bins
    n_sight = hits.groupby("_fbin")["target"].nunique()
    hits["rfi_recurrent"] = hits["_fbin"].map(n_sight).fillna(0) >= int(aconf["recurrence_min_sightlines"])
    hits["rfi_flag"] = hits["rfi_zero_drift"] | hits["rfi_recurrent"]

    # --- the geometry and the prior per hit per beam -------------------------------------
    k_sig = float(aconf["match_window_sigma"])
    sigma0 = float(conf["drift"]["sigma_drift_default_hz_s"])
    neighbours = in_beam_neighbours(sample, sorted(set(int(i) for i in hits["gaia_idx"] if i >= 0)),
                                    float(tconf.get("in_beam_max_arcmin", 8.0)))
    for b in beams:
        name = _beam_name(b)
        p = pairs[name]
        on, match, cand = [], [], []
        centres, halfs = [], []
        for _, h in hits.iterrows():
            gi = int(h["gaia_idx"])
            if gi < 0 or not len(p):
                _append(on, match, cand, centres, halfs, False, False, False, np.nan, np.nan)
                continue
            tel = _telescope_key(h.get("telescope") or h.get("catalogue_telescope"), tconf)
            band = h.get("band") if isinstance(h.get("band"), str) else h.get("catalogue_band")
            hp = _hpbw_arcmin(tel, band, tconf)
            near = [j for j in neighbours.get(gi, []) if np.isfinite(hp) and _sep_arcmin(sample, gi, j) <= 0.5 * hp]
            f_hz = float(h["freq_mhz"]) * 1e6 if np.isfinite(h["freq_mhz"]) else float(conf["drift"]["reference_frequency_ghz"]) * 1e9
            w = pointing_windows(p, [gi] + near, freq_hz=f_hz, mjd=float(h["mjd"]) if np.isfinite(h["mjd"]) else float("nan"),
                                 site=tconf.get(tel) if tel else None, conf_drift=conf["drift"],
                                 sigma_drift=sigma0, sample=sample)
            is_on = w["n_pairs"] > 0
            if not is_on:
                _append(on, match, cand, centres, halfs, False, False, False, np.nan, np.nan)
                continue
            # the window for this hit: centre range and the widest tight half-width
            c_lo, c_hi = w["drift_centre_hz_s_min"], w["drift_centre_hz_s_max"]
            hw = w["halfwidth_tight_hz_s_max"]
            d = float(h["drift_hz_s"]) if np.isfinite(h["drift_hz_s"]) else np.nan
            inside = np.isfinite(d) and (c_lo - k_sig * hw) <= d <= (c_hi + k_sig * hw)
            _append(on, match, cand, centres, halfs, True, bool(inside),
                    bool(inside and not h["rfi_flag"]), 0.5 * (c_lo + c_hi), hw)
        hits[f"{name}:on_pair_line"] = on
        hits[f"{name}:drift_match"] = match
        hits[f"{name}:candidate"] = cand
        hits[f"{name}:drift_centre_hz_s"] = centres
        hits[f"{name}:halfwidth_tight_hz_s"] = halfs
        n_on = int(sum(on))
        n_match = int(sum(match))
        n_cand = int(sum(cand))
        # chance rate: the fraction of OFF-pair-line hits whose drift falls inside a
        # window of the same width placed at the same Earth-term centre
        off = hits[(~np.asarray(on)) & (hits["gaia_idx"] >= 0)]
        exp_chance = None
        if len(off) and n_on:
            hw_med = float(np.nanmedian(halfs)) if np.isfinite(halfs).any() else sigma0
            c_med = float(np.nanmedian(centres)) if np.isfinite(centres).any() else 0.0
            frac = float(np.mean(np.abs(off["drift_hz_s"].to_numpy(float) - c_med) <= k_sig * hw_med))
            exp_chance = frac * n_on
        rep["beams"][name] = {"theta_label": b["theta_label"], "n_hits_on_pair_line": n_on,
                              "n_drift_match": n_match, "n_candidates_after_rfi": n_cand,
                              "n_expected_by_chance": exp_chance,
                              "n_trials": n_on}
    hits = hits.drop(columns=["_fbin"])
    hits.to_csv(out / "hits_crossmatch.csv", index=False)
    rep["n_rfi_zero_drift"] = int(hits["rfi_zero_drift"].sum())
    rep["n_rfi_recurrent"] = int(hits["rfi_recurrent"].sum())
    cands = []
    for b in beams:
        name = _beam_name(b)
        sub = hits[hits[f"{name}:candidate"]]
        for _, h in sub.iterrows():
            cands.append({"beam": name, "target": h["target"], "source_id": sample["source_id"].iat[int(h["gaia_idx"])],
                          "freq_mhz": h["freq_mhz"], "drift_hz_s": h["drift_hz_s"], "snr": h["snr"],
                          "mjd": h["mjd"], "table": h.get("table"),
                          "drift_centre_hz_s": h[f"{name}:drift_centre_hz_s"],
                          "halfwidth_tight_hz_s": h[f"{name}:halfwidth_tight_hz_s"],
                          "systematics_not_excluded": [
                              "RFI at non-zero drift (single-dish, no on/off cadence re-examined here)",
                              "drift window widened by unknown observation epoch" if not np.isfinite(h["mjd"]) else None,
                              "trials over every pair-line hit (see n_trials)"]})
    rep["candidates"] = cands
    any_on = sum(v["n_hits_on_pair_line"] for v in rep["beams"].values())
    if not any_on:
        rep["verdict"] = f"{V_NO_PAIRLINE_HIT} ({len(hits)} hits, {rep['n_hits_matched_to_sample']} on sample stars)"
    elif not cands:
        rep["verdict"] = f"{V_PAIRLINE_OFF_PRIOR} ({any_on} pair-line hit tests, 0 inside the prior after RFI)"
    else:
        rep["verdict"] = f"{V_PAIRLINE_MATCH} ({len(cands)} hit-beam matches; see n_expected_by_chance per beam)"
    rep["acquisition"] = log.as_dict()
    _write(out / "hits.json", rep)
    return _finish(conf, out, rep, geom, recut, beams, started)


def _append(on, match, cand, centres, halfs, o, m, c, ce, hw) -> None:
    on.append(o)
    match.append(m)
    cand.append(c)
    centres.append(ce)
    halfs.append(hw)


def _finish(conf, out, assess_rep, geom, recut, beams, started) -> dict:
    """summary.json: verdict, funnel, coverage."""
    targets = _read(out / "targets.json") or {}
    funnel = {"n_stars_100pc": geom.get("n_stars", 0),
              "n_stars_with_rv": geom.get("n_with_rv", 0),
              "n_bl_targets": targets.get("n_names", 0),
              "n_bl_targets_resolved": targets.get("n_resolved", 0),
              "n_bl_targets_in_sample": recut.get("n_targets_in_sample", geom.get("n_targets_matched", 0)),
              "n_bl_files": (recut.get("bl_files") or {}).get("n_files", 0),
              "n_pointings": recut.get("n_pointings", 0),
              "n_hits": assess_rep.get("n_hits", 0),
              "n_hits_on_sample_stars": assess_rep.get("n_hits_matched_to_sample", 0)}
    per_beam = {}
    for b in beams:
        name = _beam_name(b)
        g = (geom.get("beams") or {}).get(name, {})
        r = (recut.get("beams") or {}).get(name, {})
        a = (assess_rep.get("beams") or {}).get(name, {})
        per_beam[name] = {"theta_label": b.get("theta_label"),
                          "n_spillover": g.get("n_spillover"), "n_between": g.get("n_between"),
                          "analytic": g.get("analytic_expectation"),
                          "n_bl_targets_as_transmitter": g.get("n_bl_targets_as_transmitter"),
                          "n_pointings_on_pair_line": r.get("n_pointings_on_pair_line"),
                          "n_hits_on_pair_line": a.get("n_hits_on_pair_line"),
                          "n_candidates": a.get("n_candidates_after_rfi"),
                          "n_expected_by_chance": a.get("n_expected_by_chance")}
    verdict = " | ".join(v for v in (geom.get("verdict"), targets.get("verdict"),
                                     recut.get("verdict"), assess_rep.get("verdict")) if v)
    summary = {"channel": "relay", "signature": "S60", "generated_utc": _now(),
               "verdict": verdict, "geometry_verdict": geom.get("verdict"),
               "targets_verdict": targets.get("verdict"), "recut_verdict": recut.get("verdict"),
               "assess_verdict": assess_rep.get("verdict"), "funnel": funnel, "per_beam": per_beam,
               "scaling": geom.get("scaling"), "candidates": assess_rep.get("candidates", []),
               "sample_route": (geom.get("sample") or {}).get("route"),
               "sample_status": (geom.get("sample") or {}).get("status"),
               "rv_completeness": geom.get("rv_completeness"),
               "elapsed_assess_s": round(time.monotonic() - started, 1)}
    _write(out / "summary.json", summary)
    print(f"[relay] verdict: {verdict}")
    return summary


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def relay_run(stage: str = "all", *, out_dir=None, conf: dict | None = None, bl_fetch=None,
              tap_fn=None, query_fn=None, fetch_fn=None, gaia_df=None, max_stars=None,
              hits_df=None, log=None) -> dict:
    conf = conf or load_relay_config()
    out = Path(out_dir) if out_dir else Path("results") / "relay"
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage in ("all", "", None) else tuple(s.strip() for s in stage.split(","))
    log = log or AcquisitionLog(prefix="relay")
    rep: dict = {}
    for s in stages:
        if s == "probe":
            rep = stage_probe(conf, out, bl_fetch=bl_fetch, tap_fn=tap_fn, query_fn=query_fn,
                              fetch_fn=fetch_fn, log=log)
        elif s == "targets":
            rep = stage_targets(conf, out, bl_fetch=bl_fetch, tap_fn=tap_fn, query_fn=query_fn,
                                fetch_fn=fetch_fn, log=log)
        elif s == "geometry":
            rep = stage_geometry(conf, out, tap_fn=tap_fn, query_fn=query_fn, fetch_fn=fetch_fn,
                                 gaia_df=gaia_df, max_stars=max_stars, log=log)
        elif s == "recut":
            rep = stage_recut(conf, out, bl_fetch=bl_fetch, log=log)
        elif s == "assess":
            rep = stage_assess(conf, out, query_fn=query_fn, fetch_fn=fetch_fn, tap_fn=tap_fn,
                               log=log, hits_df=hits_df)
        else:
            raise SystemExit(f"unknown stage {s!r}; choose from {STAGES}")
        log.write(out / "acquisition_log.json")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(prog="seti relay",
                                description="RELAY (S60): intercepting node-to-node beams by geometry")
    p.add_argument("--stage", default="all", help="comma-separated subset of "
                   + ",".join(STAGES) + " or 'all'")
    p.add_argument("--out-dir", default="results/relay")
    p.add_argument("--max-stars", type=int, default=0, help="debug cap on the Gaia sample (0 = none)")
    p.add_argument("--config", default="", help="alternative config/relay.yaml")
    a = p.parse_args(argv)
    conf = load_relay_config(a.config) if a.config else None
    rep = relay_run(a.stage, out_dir=a.out_dir, conf=conf, max_stars=a.max_stars or None)
    v = rep.get("verdict") if isinstance(rep, dict) else None
    if v:
        print(f"[relay] {v}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())


__all__ = ["STAGES", "V_GEOMETRY", "V_NO_DATA", "V_NO_HIT_CATALOGUE", "V_NO_PAIRLINE_HIT",
           "V_PAIRLINE_MATCH", "V_PAIRLINE_OFF_PRIOR", "V_RECUT", "V_TARGETS",
           "in_beam_neighbours", "load_relay_config", "main", "match_targets_to_sample",
           "name_key", "pointing_windows", "relay_run", "stage_assess", "stage_geometry",
           "stage_probe", "stage_recut", "stage_targets"]
