"""Stage orchestration for SPARK.  Writes ``results/spark/``.

Stages
------
``probe``    what the archives actually serve: the Euclid Q1 table schemas
             and counts, a TOP-5 of the line×MER join (the wavelength scale
             is inferred from real values), the Gaia routes, the SPHEREx
             obscore count at the first box, one full level-2 product
             (HDUs, WCS keys, the wavelength map and its fingerprint), one
             IBE cutout of it (HDUs, offset keys, whether the map survives),
             and the cutout-tool page.  ``probe.json``.
``euclid``   shard ``i/n`` of the (field, dec-strip) units: the line×MER
             strip, the spectra×MER strip (denominator), the Gaia field
             cone; checkpointed per unit; then the screen for the shard.
``spherex``  shard ``i/n`` of the seed boxes: Gaia seeds + isolation,
             obscore products, cutouts, forced photometry, the per-star
             statistic; checkpointed per box.
``screen``   re-run the pure screens over whatever shard files are local.
``assess``   merge every shard, redo the cross-shard recurrence and the
             trials over the full denominator, write ``summary.json``,
             ``candidates.csv`` / ``candidates.json``.

Verdict vocabulary (``summary.json["verdict"]``)
------------------------------------------------
``NO_DATA_REACHED``                 no star was tested in either survey
``NO_SPARK_SURVIVOR``               stars were tested; no feature survived
``SPARK_CANDIDATES_PENDING_VET``    ≥ 1 survivor — the vet is what is next
prefixed by ``DEGRADED (...)`` when a route failed, a strip or box was
lost, or a veto could not be applied.  A null changes the question.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import euclid as E
from . import gaia as G
from . import lines as L
from . import spherex as S

VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_NONE = "NO_SPARK_SURVIVOR"
VERDICT_CANDIDATES = "SPARK_CANDIDATES_PENDING_VET"
STAGES = ("probe", "euclid", "spherex", "screen", "assess", "all")

DEFAULTS: dict = {
    "services": {"irsa_tap": "https://irsa.ipac.caltech.edu/TAP",
                 "irsa_base": "https://irsa.ipac.caltech.edu/",
                 "gaia_tap": "https://gea.esac.esa.int/tap-server/tap",
                 "irsa_gaia_tables": ["gaia_dr3_source"]},
    "euclid": {"tables": {"lines": "euclid_q1_spe_lines_line_features", "mer": "euclid_q1_mer_catalogue",
                          "spectra": ["euclid_q1_spectro_zcatalog_spe_classification"]},
               "fields": {"EDF-N": {"ra": 269.73, "dec": 66.01, "radius_deg": 2.7}},
               "strip_deg": 0.5, "snr_acquire_min": 3.0, "band_um": [1.21, 1.89], "edge_margin_um": 0.02,
               "resolving_power": 450.0, "wavelength_unit": "auto", "snr_min": 5.0, "n_dith_min": 2,
               "max_fwhm_resel": 2.5, "stellar_tol_resel": 1.5, "redshift_z_tol": 0.003,
               "redshift_min_snr": 3.0, "point_like_prob_min": 0.5, "require_point_like": False,
               "gaia_match_arcsec": 1.0, "gaia_epoch": 2016.0, "euclid_epoch": 2024.3,
               "plx_over_error_min": 5.0, "gaia_g_max": 21.5, "blend_radius_arcsec": 6.0, "blend_dmag": 2.0,
               "dispersion_neighbour_radius_arcsec": 140.0, "dispersion_neighbour_halfwidth_arcsec": 3.0,
               "recurrence_bin_resel": 1.0, "recurrence_min_stars": 3, "trials_alpha": 0.01,
               "max_rows_per_strip": 400000, "time_budget_s": 15000},
    "spherex": {"obscore_table": "spherex.obscore", "plane_table": "spherex.plane",
                "artifact_table": "spherex.artifact", "splices_table": "splices",
                "detector_bands_um": {"D1": [0.75, 1.11], "D2": [1.11, 1.64], "D3": [1.64, 2.42],
                                      "D4": [2.42, 3.82], "D5": [3.82, 4.42], "D6": [4.42, 5.00]},
                "boxes": [{"name": "NEP-0", "ra": 270.0, "dec": 66.56}], "box_arcmin": 15.0,
                "seed_g_min": 9.0, "seed_g_max": 15.0, "seed_plx_over_error_min": 5.0, "neighbour_g_max": 19.0,
                "isolation_radius_arcsec": 20.0, "isolation_dmag": 3.0, "inner_radius_arcsec": 9.0,
                "inner_dmag": 6.0, "max_seeds_per_box": 200, "max_images_per_detector": 400,
                "cutout_route": "auto", "max_full_downloads": 12, "request_timeout_s": 120,
                "time_budget_s": 15000, "pixel_arcsec": 6.15, "aperture_radius_px": 2.0, "annulus_px": [4.0, 7.0],
                "subtract_zodi_plane": True, "flag_mask_any": True, "pass_gap_days": 30.0,
                "min_pixel_separation_px": 20.0, "channel_width_resel": 1.0, "excess_sigma_total": 5.0,
                "excess_sigma_single": 2.5, "neighbour_channel_sigma_max": 2.0, "edge_margin_resel": 2.0,
                "min_samples_per_detector": 12, "continuum_window_resel": 5.0, "stellar_tol_resel": 1.0,
                "recurrence_min_stars": 3, "pixel_recurrence_px": 3.0, "pixel_recurrence_min_stars": 2,
                "trials_alpha": 0.01},
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


def load_spark_config(path: Path | None = None) -> dict:
    """``config/spark.yaml`` over :data:`DEFAULTS`; a missing file degrades to the defaults."""
    try:
        import yaml
        if path is None:
            path = Path(__file__).resolve().parents[3] / "config" / "spark.yaml"
        if not Path(path).exists():
            return _deep_update(DEFAULTS, {})
        return _deep_update(DEFAULTS, yaml.safe_load(Path(path).read_text()) or {})
    except Exception as exc:                            # noqa: BLE001
        print(f"[spark] config not loaded ({exc!r}); using defaults")
        return _deep_update(DEFAULTS, {})


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    os.replace(tmp, path)


def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:                                   # noqa: BLE001
        return default


def _parse_shard(s: str | None) -> tuple[int, int]:
    if not s:
        return 0, 1
    a, b = str(s).split("/")
    return int(a), max(1, int(b))


# --------------------------------------------------------------------------
# Runner-side query callables (never used by the offline suite)
# --------------------------------------------------------------------------
def make_irsa_query(service: str, *, retries: int = 3, job_timeout_s: float = 1800.0, tag: str = "spark"):
    """``(adql, maxrec=None) -> DataFrame``; async first, sync last; raises after the ladder."""
    def _q(adql: str, maxrec: int | None = None) -> pd.DataFrame:
        import pyvo
        last = ""
        for attempt in range(retries):
            try:
                svc = pyvo.dal.TAPService(service)
                if attempt < retries - 1:
                    job = svc.submit_job(adql, maxrec=maxrec)
                    job.run()
                    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=job_timeout_s)
                    if job.phase != "COMPLETED":
                        msg = ""
                        try:
                            msg = str(job.errorsummary)
                        except Exception:               # noqa: BLE001
                            pass
                        try:
                            job.delete()
                        except Exception:               # noqa: BLE001
                            pass
                        raise RuntimeError(f"async job phase={job.phase} {msg[:300]}")
                    tbl = job.fetch_result().to_table()
                    try:
                        job.delete()
                    except Exception:                   # noqa: BLE001
                        pass
                else:
                    tbl = svc.search(adql, maxrec=maxrec).to_table()
                df = tbl.to_pandas()
                df.columns = [str(c).lower() for c in df.columns]
                return df
            except Exception as exc:                    # noqa: BLE001
                last = repr(exc)
                print(f"[{tag}] IRSA attempt {attempt + 1}/{retries} failed: {last[:300]}", flush=True)
                _time.sleep(3.0 * (attempt + 1))
        raise RuntimeError(last)
    return _q


def make_esa_query(tag: str = "spark"):
    from ..ignition.sample import gaia_query

    def _q(adql: str, maxrec: int | None = None) -> pd.DataFrame:
        return gaia_query(adql, tag=tag, label="spark_gaia", allow_sync=True, timeout_s=1500.0)
    return _q


def make_fetcher(timeout_s: float = 120.0):
    import requests
    sess = requests.Session()

    def _f(url: str):
        return S.fetch_fits(url, timeout=timeout_s, session=sess)
    return _f


# --------------------------------------------------------------------------
# Probe
# --------------------------------------------------------------------------
def _try(fn, *a, **k):
    t0 = _time.monotonic()
    try:
        v = fn(*a, **k)
        return {"status": "OK", "seconds": round(_time.monotonic() - t0, 1)}, v
    except Exception as exc:                            # noqa: BLE001
        return {"status": "FAILED", "error": repr(exc)[:500], "seconds": round(_time.monotonic() - t0, 1)}, None


def probe(conf: dict, out_dir: Path, *, irsa_query=None, esa_query=None, fetch_fn=None,
          http_get=None, skip_spherex: bool = False) -> dict:
    """What the archives serve.  Every callable is injected (runner) or absent (offline)."""
    ec, sc, sv = conf["euclid"], conf["spherex"], conf["services"]
    rec: dict = {"generated_at": _now(), "euclid": {}, "gaia": {}, "spherex": {}}
    t0 = _time.monotonic()
    if irsa_query is None:
        rec["status"] = "NO_QUERY_FUNCTION"
        _write_json(out_dir / "probe.json", rec)
        return rec

    # ---- Euclid schemas ----
    tables = ec["tables"]
    schemas: dict = {}
    for key in ("lines", "mer", "classification", "quality", "star_candidates", "galaxy_candidates", "spectrofile"):
        t = tables.get(key)
        if not t:
            continue
        names, r = E.discover_columns(t, irsa_query)
        schemas[key] = r
    for t in tables.get("spectra", []):
        if t not in [v.get("table") for v in schemas.values()]:
            names, r = E.discover_columns(t, irsa_query)
            schemas[f"spectra:{t}"] = r
    rec["euclid"]["schemas"] = schemas
    roles_line = E.resolve_roles(schemas.get("lines", {}).get("columns", []), E.LINE_ROLES)
    roles_mer = E.resolve_roles(schemas.get("mer", {}).get("columns", []), E.MER_ROLES)
    rec["euclid"]["roles_line"] = roles_line
    rec["euclid"]["roles_mer"] = roles_mer
    rec["euclid"]["missing_line_roles"] = E.missing_required(roles_line, E.LINE_REQUIRED)
    rec["euclid"]["missing_mer_roles"] = E.missing_required(roles_mer, E.MER_REQUIRED)
    # spectra denominator table: the first whose schema answered with object_id
    spec_choice = None
    for t in tables.get("spectra", []):
        cols = None
        for v in schemas.values():
            if v.get("table") == t and v.get("columns"):
                cols = v["columns"]
        if cols and E.resolve_roles(cols, E.SPECTRA_ROLES).get("object_id"):
            spec_choice = t
            break
    rec["euclid"]["spectra_table"] = spec_choice
    # counts
    for key, t in (("lines", tables["lines"]), ("spectra", spec_choice)):
        if not t:
            continue
        r, df = _try(irsa_query, f"SELECT COUNT(*) AS n FROM {t}")
        if df is not None and len(df):
            r["n"] = int(pd.to_numeric(df.iloc[0, 0], errors="coerce"))
        rec["euclid"][f"count_{key}"] = r
    if roles_line.get("snr"):
        r, df = _try(irsa_query, f"SELECT COUNT(*) AS n FROM {tables['lines']} WHERE {roles_line['snr']} >= {float(ec['snr_min']):.1f}")
        if df is not None and len(df):
            r["n"] = int(pd.to_numeric(df.iloc[0, 0], errors="coerce"))
        rec["euclid"]["count_lines_snr_ge_min"] = r
    # a TOP-5 of the real join, for the wavelength scale and the value ranges
    if not rec["euclid"]["missing_line_roles"] and not rec["euclid"]["missing_mer_roles"]:
        fields = ec["fields"]
        fname = next(iter(fields))
        unit = E.units_for_shard({fname: fields[fname]}, float(ec["strip_deg"]))[len(E.field_strips(fields[fname], float(ec["strip_deg"]))) // 2]
        adql = E.strip_adql(tables, roles_line, roles_mer, unit, snr_min=float(ec["snr_min"]), top=5)
        r, df = _try(irsa_query, adql)
        r["adql"] = adql[:1500]
        if df is not None:
            r["n_rows"] = int(len(df))
            r["columns"] = list(map(str, df.columns))
            r["head"] = df.head(5).to_dict(orient="records")
            if len(df):
                feat, meta = E.normalise_features(df, fname, ec.get("wavelength_unit", "auto"))
                r["wavelength"] = meta
                r["wl_um_head"] = [None if not np.isfinite(x) else round(float(x), 5) for x in feat["wl_um"].head(5)]
        rec["euclid"]["join_top5"] = r
        # the line-name vocabulary (what the pipeline calls things)
        if roles_line.get("line_name"):
            r, df = _try(irsa_query, f"SELECT {roles_line['line_name']} AS nm, COUNT(*) AS n FROM {tables['lines']} "
                                      f"WHERE {roles_line['snr']} >= {float(ec['snr_min']):.1f} GROUP BY {roles_line['line_name']}")
            if df is not None:
                r["names"] = df.head(60).to_dict(orient="records")
            rec["euclid"]["line_names"] = r

    # ---- Gaia routes ----
    r, df = _try(irsa_query, "SELECT table_name FROM TAP_SCHEMA.tables WHERE table_name LIKE '%gaia%'")
    if df is not None and len(df):
        r["tables"] = [str(x) for x in df.iloc[:, 0].tolist()]
    rec["gaia"]["irsa_tables"] = r
    irsa_gaia = [t for t in sv.get("irsa_gaia_tables", []) if t in set(rec["gaia"]["irsa_tables"].get("tables", []))]
    rec["gaia"]["irsa_gaia_table_usable"] = irsa_gaia
    if irsa_gaia:
        names, r2 = E.discover_columns(irsa_gaia[0], irsa_query)
        rec["gaia"]["irsa_gaia_columns"] = r2
        have = {n.lower() for n in names}
        rec["gaia"]["irsa_gaia_missing_columns"] = [c for c in G.GAIA_COLS if c not in have]
        if rec["gaia"]["irsa_gaia_missing_columns"]:
            irsa_gaia = []
    box0 = sc["boxes"][0]
    log: list = []
    df, grec = G.fetch_gaia_cone(float(box0["ra"]), float(box0["dec"]), 0.05, 19.0,
                                 irsa_tables=irsa_gaia, irsa_query_fn=irsa_query, esa_query_fn=esa_query,
                                 label="probe_cone", log=log)
    rec["gaia"]["cone_test"] = grec
    rec["gaia"]["route_recommended"] = grec.get("route")

    # ---- SPHEREx ----
    if not skip_spherex:
        sx: dict = {}
        r, df = _try(irsa_query, S.obscore_count_adql(sc["obscore_table"], float(box0["ra"]), float(box0["dec"])))
        if df is not None and len(df):
            r["n"] = int(pd.to_numeric(df.iloc[0, 0], errors="coerce"))
        sx["obscore_count_box0"] = r
        names, r2 = E.discover_columns(sc["obscore_table"], irsa_query)
        sx["obscore_columns"] = r2
        have = {n.lower() for n in names}
        cols = [c for c in S.OBSCORE_COLS if c in have]
        sx["obscore_missing_columns"] = [c for c in S.OBSCORE_COLS if c not in have]
        r, df = _try(irsa_query, S.obscore_adql(sc["obscore_table"], float(box0["ra"]), float(box0["dec"]), cols, top=12))
        if df is not None:
            r["n_rows"] = int(len(df))
            r["head"] = df.head(12).to_dict(orient="records")
            if len(df):
                sel = S.select_images(df, sc["detector_bands_um"], 0)
                r["detectors"] = {str(k): int(v) for k, v in sel["detector"].value_counts(dropna=False).items()}
        sx["obscore_top"] = r
        # a plane->artifact join (the URI path), for the record
        r, dfa = _try(irsa_query, f"SELECT TOP 6 p.planeid, p.energy_bandpassname, a.uri FROM {sc['plane_table']} p "
                                   f"JOIN {sc['artifact_table']} a ON p.planeid = a.planeid")
        if dfa is not None:
            r["head"] = dfa.head(6).to_dict(orient="records")
        sx["plane_artifact_top"] = r
        if http_get is not None:
            r, txt = _try(http_get, sv["irsa_base"].rstrip("/") + "/data/SPHEREx/docs/cutout_tool.html")
            if txt is not None:
                r["text_head"] = str(txt)[:4000]
            sx["cutout_tool_page"] = r
        # one full product, one cutout
        url = None
        if df is not None and len(df) and "access_url" in df:
            url = S.product_url(sv["irsa_base"], str(df["access_url"].iloc[0]))
        elif dfa is not None and len(dfa):
            url = S.product_url(sv["irsa_base"], str(dfa["uri"].iloc[0]))
        sx["product_url"] = url
        if url and fetch_fn is not None:
            r, hd = _try(fetch_fn, url)
            if hd is not None:
                r["hdus"] = S.describe_hdul(hd)
                try:
                    img = S._find_image_hdu(hd)
                    r["crpix"] = [float(img.header.get("CRPIX1", np.nan)), float(img.header.get("CRPIX2", np.nan))]
                    r["mjd"] = S.mjd_of(img.header)
                    r["bunit"] = str(img.header.get("BUNIT", ""))
                except Exception as exc:                # noqa: BLE001
                    r["image_error"] = repr(exc)[:300]
                rw, wm = _try(S.WaveMap.from_hdul, hd)
                if wm is not None:
                    rw["fingerprint"] = wm.fingerprint()
                r["wavemap"] = rw
                try:
                    hd.close()
                except Exception:                       # noqa: BLE001
                    pass
            sx["full_product"] = r
            curl = S.cutout_url(url, float(box0["ra"]), float(box0["dec"]), float(sc["box_arcmin"]) * 60.0)
            sx["cutout_url"] = curl
            r, hc = _try(fetch_fn, curl)
            if hc is not None:
                r["hdus"] = S.describe_hdul(hc)
                try:
                    img = S._find_image_hdu(hc)
                    r["shape"] = list(img.data.shape)
                    full_crpix = tuple(sx.get("full_product", {}).get("crpix") or ()) or None
                    r["offset_from_ltv_or_crpix"] = S.cutout_offset(img.header, full_crpix)
                    w = S.celestial_wcs(img.header)
                    x, y = w.all_world2pix([float(box0["ra"])], [float(box0["dec"])], 0)
                    r["box_centre_pixel"] = [float(x[0]), float(y[0])]
                except Exception as exc:                # noqa: BLE001
                    r["image_error"] = repr(exc)[:300]
                rw, wm = _try(S.WaveMap.from_hdul, hc)
                if wm is not None:
                    rw["fingerprint"] = wm.fingerprint()
                r["wavemap"] = rw
            sx["cutout"] = r
            # a second product of the same detector: is the map a property of the detector?
            if df is not None and len(df) > 1 and "access_url" in df and sx.get("full_product", {}).get("wavemap", {}).get("status") == "OK":
                sel = S.select_images(df, sc["detector_bands_um"], 0)
                det0 = sel["detector"].iloc[0]
                same = sel[(sel["detector"] == det0)]
                if len(same) > 1:
                    url2 = S.product_url(sv["irsa_base"], str(same["access_url"].iloc[-1]))
                    r, hd2 = _try(fetch_fn, url2)
                    if hd2 is not None:
                        rw, wm2 = _try(S.WaveMap.from_hdul, hd2)
                        if wm2 is not None:
                            rw["fingerprint"] = wm2.fingerprint()
                            f1 = sx["full_product"]["wavemap"]["fingerprint"]["lambda_um_at"]
                            f2 = rw["fingerprint"]["lambda_um_at"]
                            diffs = [abs(a[2] - b[2]) for a, b in zip(f1, f2, strict=True) if a[2] is not None and b[2] is not None]
                            rw["max_abs_diff_um_vs_first"] = max(diffs) if diffs else None
                        r["wavemap"] = rw
                        r["url"] = url2
                    sx["second_product_same_detector"] = r
        rec["spherex"] = sx
    rec["elapsed_s"] = round(_time.monotonic() - t0, 1)
    rec["status"] = "OK"
    _write_json(out_dir / "probe.json", rec)
    return rec


# --------------------------------------------------------------------------
# Euclid stage
# --------------------------------------------------------------------------
def _gaia_for_field(conf: dict, field: dict, *, irsa_query, esa_query, irsa_tables, cache_dir: Path,
                    name: str, log: list) -> tuple[pd.DataFrame, dict]:
    p = cache_dir / f"gaia_{name}.csv"
    if p.exists():
        df = pd.read_csv(p)
        return df, {"route": "cache", "n_rows": int(len(df)), "status": "OK" if len(df) else "QUERY_RETURNED_ZERO_ROWS"}
    ec = conf["euclid"]
    df, rec = G.fetch_gaia_cone(float(field["ra"]), float(field["dec"]), float(field["radius_deg"]) + 0.05,
                                float(ec["gaia_g_max"]), irsa_tables=irsa_tables, irsa_query_fn=irsa_query,
                                esa_query_fn=esa_query, label=f"gaia_{name}", log=log)
    if len(df):
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
    return df, rec


def euclid_stage(conf: dict, out_dir: Path, *, shard: int = 0, n_shards: int = 1, irsa_query=None,
                 esa_query=None, fields: list[str] | None = None, max_units: int | None = None,
                 probe_rec: dict | None = None, with_denominator: bool = True) -> dict:
    """Acquire the shard's units (checkpointed), then screen the shard."""
    ec = conf["euclid"]
    edir = out_dir / "euclid"
    edir.mkdir(parents=True, exist_ok=True)
    t0 = _time.monotonic()
    budget = float(ec.get("time_budget_s", 15000))
    led: dict = {"stage": "euclid", "shard": shard, "n_shards": n_shards, "generated_at": _now(),
                 "units": [], "gaia": {}, "queries": [], "degraded": []}
    if irsa_query is None:
        led["status"] = VERDICT_NO_DATA
        led["degraded"].append("no IRSA query function")
        _write_json(edir / f"euclid_s{shard}.json", led)
        return led
    probe_rec = probe_rec or _read_json(out_dir / "probe.json", {}) or {}
    tables = dict(ec["tables"])
    # roles: from the probe when it answered, else discovered now
    roles_line = (probe_rec.get("euclid") or {}).get("roles_line") or {}
    roles_mer = (probe_rec.get("euclid") or {}).get("roles_mer") or {}
    if not roles_line.get("wl") or not roles_mer.get("ra"):
        names, _ = E.discover_columns(tables["lines"], irsa_query)
        roles_line = E.resolve_roles(names, E.LINE_ROLES)
        names, _ = E.discover_columns(tables["mer"], irsa_query)
        roles_mer = E.resolve_roles(names, E.MER_ROLES)
    led["roles_line"], led["roles_mer"] = roles_line, roles_mer
    miss = E.missing_required(roles_line, E.LINE_REQUIRED) + E.missing_required(roles_mer, E.MER_REQUIRED)
    if miss:
        led["status"] = VERDICT_NO_DATA
        led["degraded"].append(f"schema: missing roles {miss}")
        _write_json(edir / f"euclid_s{shard}.json", led)
        return led
    spec_table = (probe_rec.get("euclid") or {}).get("spectra_table") or (tables.get("spectra") or [None])[0]
    roles_spec = {}
    if with_denominator and spec_table:
        names, _ = E.discover_columns(spec_table, irsa_query)
        roles_spec = E.resolve_roles(names, E.SPECTRA_ROLES)
        if not roles_spec.get("object_id"):
            led["degraded"].append(f"denominator table {spec_table} has no object_id; stars-with-spectra not measured")
            spec_table = None
    led["spectra_table"] = spec_table
    irsa_tables = (probe_rec.get("gaia") or {}).get("irsa_gaia_table_usable") or []
    all_fields = ec["fields"]
    if fields:
        all_fields = {k: v for k, v in all_fields.items() if k in set(fields)}
    units = E.units_for_shard(all_fields, float(ec["strip_deg"]), shard, n_shards)
    if max_units:
        units = units[:int(max_units)]
    led["n_units"] = len(units)
    maxrec = int(ec.get("max_rows_per_strip", 400000))
    gaia_cache: dict[str, pd.DataFrame] = {}
    for u in units:
        if _time.monotonic() - t0 > budget:
            led["degraded"].append(f"time budget exhausted before unit {u['field']}:{u['k']}")
            break
        tag = f"{u['field']}_{u['k']}"
        urec = {"field": u["field"], "k": u["k"], "dec_lo": u["dec_lo"], "dec_hi": u["dec_hi"]}
        strip_csv = edir / f"strip_{tag}.csv"
        done_json = edir / f"strip_{tag}.json"
        if done_json.exists():
            urec.update(_read_json(done_json, {}))
            urec["resumed"] = True
            led["units"].append(urec)
            continue
        # Gaia for the field (once per shard)
        if u["field"] not in gaia_cache:
            gdf, grec = _gaia_for_field(conf, all_fields[u["field"]], irsa_query=irsa_query, esa_query=esa_query,
                                        irsa_tables=irsa_tables, cache_dir=edir, name=u["field"], log=led["queries"])
            gaia_cache[u["field"]] = gdf
            led["gaia"][u["field"]] = {k: grec.get(k) for k in ("route", "status", "n_rows")}
            if not len(gdf):
                led["degraded"].append(f"Gaia cone for {u['field']} returned nothing ({grec.get('status')})")
        # the line x MER strip
        adql_fn = lambda unit, _t=tables, _rl=roles_line, _rm=roles_mer: E.strip_adql(  # noqa: E731
            _t, _rl, _rm, unit, snr_min=float(ec["snr_acquire_min"]))
        raw, r = E.fetch_strip(adql_fn, u, irsa_query, maxrec=maxrec, label="lines", log=led["queries"])
        urec["lines_status"] = r.get("status")
        urec["n_raw_rows"] = int(len(raw))
        if r.get("status") == "QUERY_FAILED":
            led["degraded"].append(f"strip {tag}: lines query failed")
        feat, meta = E.normalise_features(raw, u["field"], ec.get("wavelength_unit", "auto")) if len(raw) else (pd.DataFrame(columns=list(E.STANDARD_COLS)), {})
        urec["wavelength"] = meta
        feat.to_csv(strip_csv, index=False)
        # the denominator strip (objects with a spectrum) -> stars with spectra
        if spec_table:
            adql_d = lambda unit, _s=spec_table, _rs=roles_spec, _t=tables, _rm=roles_mer: E.denominator_adql(  # noqa: E731
                _s, _rs, _t, _rm, unit)
            drow, rd = E.fetch_strip(adql_d, u, irsa_query, maxrec=maxrec, label="spectra", log=led["queries"])
            urec["spectra_status"] = rd.get("status")
            urec["n_objects_with_spectra"] = int(len(drow))
            if rd.get("status") == "QUERY_FAILED":
                led["degraded"].append(f"strip {tag}: spectra (denominator) query failed")
            gdf = gaia_cache.get(u["field"], pd.DataFrame())
            gstar = gdf[pd.to_numeric(gdf.get("parallax_over_error", np.nan), errors="coerce") > float(ec["plx_over_error_min"])] if len(gdf) else gdf
            n_star, matched = E.stars_with_spectra(drow, gstar, ec) if len(drow) else (0, pd.DataFrame())
            urec["n_stars_with_spectra"] = int(n_star)
            if len(matched):
                keep = [c for c in matched.columns if c.startswith("m_") or c.startswith("s_") or c.startswith("gaia_")]
                matched[keep].to_csv(edir / f"denominator_{tag}.csv", index=False)
        urec["seconds"] = round(_time.monotonic() - t0, 1)
        _write_json(done_json, urec)
        led["units"].append(urec)
        print(f"[spark] euclid {tag}: {urec.get('n_raw_rows')} feature rows, "
              f"{urec.get('n_objects_with_spectra', '?')} spectra, {urec.get('n_stars_with_spectra', '?')} stars", flush=True)
    led["elapsed_s"] = round(_time.monotonic() - t0, 1)
    led["status"] = "OK" if any(u.get("n_raw_rows", 0) for u in led["units"]) else VERDICT_NO_DATA
    _write_json(edir / f"euclid_s{shard}.json", led)
    scr = screen_euclid_local(conf, out_dir, shard=shard)
    led["screen"] = {k: scr.get(k) for k in ("n_features_on_stars", "n_stars_with_features", "n_survivors")}
    _write_json(edir / f"euclid_s{shard}.json", led)
    return led


def screen_euclid_local(conf: dict, out_dir: Path, *, shard: int | None = None) -> dict:
    """Screen the strip CSVs present locally (a shard's, or all), per field."""
    ec = conf["euclid"]
    edir = out_dir / "euclid"
    edir.mkdir(parents=True, exist_ok=True)
    strips = sorted(glob.glob(str(edir / "strip_*.csv")))
    funnel: dict = {"generated_at": _now(), "shard": shard, "n_strip_files": len(strips), "fields": {}}
    all_screened: list[pd.DataFrame] = []
    by_field: dict[str, list[pd.DataFrame]] = {}
    for p in strips:
        try:
            df = pd.read_csv(p)
        except Exception:                               # noqa: BLE001
            continue
        if not len(df):
            continue
        by_field.setdefault(str(df["field"].iloc[0]), []).append(df)
    n_stars_total = 0
    for dj in glob.glob(str(edir / "strip_*.json")):
        j = _read_json(dj, {}) or {}
        n_stars_total += int(j.get("n_stars_with_spectra", 0) or 0)
    funnel["n_stars_with_spectra"] = n_stars_total
    for field, parts in by_field.items():
        feat = pd.concat(parts, ignore_index=True)
        gp = edir / f"gaia_{field}.csv"
        gaia_field = pd.read_csv(gp) if gp.exists() else pd.DataFrame(columns=list(G.GAIA_COLS))
        gstar = gaia_field[pd.to_numeric(gaia_field.get("parallax_over_error", np.nan), errors="coerce") > float(ec["plx_over_error_min"])] if len(gaia_field) else gaia_field
        n_stars_field = sum(int((_read_json(dj, {}) or {}).get("n_stars_with_spectra", 0) or 0)
                            for dj in glob.glob(str(edir / f"strip_{field}_*.json")))
        scr, f = E.screen_features(feat, gstar, ec, n_stars_with_spectra=n_stars_field or None, gaia_field=gaia_field)
        f["n_gaia_field"] = int(len(gaia_field))
        f["n_gaia_stars"] = int(len(gstar))
        funnel["fields"][field] = f
        if len(scr):
            scr["field"] = field
            all_screened.append(scr)
    screened = pd.concat(all_screened, ignore_index=True) if all_screened else pd.DataFrame()
    tag = f"_s{shard}" if shard is not None else ""
    screened.to_csv(edir / f"features{tag}.csv", index=False)
    funnel["n_features_on_stars"] = int(len(screened))
    funnel["n_stars_with_features"] = int(screened["gaia_source_id"].nunique()) if len(screened) else 0
    funnel["n_survivors"] = int(screened["survivor"].sum()) if len(screened) else 0
    _write_json(edir / f"screen{tag}.json", funnel)
    return funnel


# --------------------------------------------------------------------------
# SPHEREx stage
# --------------------------------------------------------------------------
def _boxes_for_shard(boxes: list[dict], shard: int, n_shards: int) -> list[dict]:
    return [b for i, b in enumerate(boxes) if i % max(1, n_shards) == shard]


def spherex_box(conf: dict, out_dir: Path, box: dict, *, irsa_query, esa_query, fetch_fn, irsa_tables,
                probe_rec: dict, deadline: float, max_images: int | None = None,
                max_seeds: int | None = None, log: list) -> dict:
    """One seed box end to end, checkpointed per image."""
    sc, sv = conf["spherex"], conf["services"]
    sdir = out_dir / "spherex"
    sdir.mkdir(parents=True, exist_ok=True)
    name = str(box["name"])
    ra0, dec0 = float(box["ra"]), float(box["dec"])
    half = float(sc["box_arcmin"]) / 2.0
    rec: dict = {"box": name, "ra": ra0, "dec": dec0, "degraded": []}
    t0 = _time.monotonic()
    # ---- seeds ----
    seeds_csv = sdir / f"seeds_{name}.csv"
    if seeds_csv.exists():
        seeds = pd.read_csv(seeds_csv)
        rec["seeds"] = {"route": "cache", "n_seeds": int(len(seeds))}
    else:
        gdf, grec = G.fetch_gaia_cone(ra0, dec0, S.box_radius_deg(float(sc["box_arcmin"])) + 20.0 / 3600.0,
                                      float(sc["neighbour_g_max"]), irsa_tables=irsa_tables,
                                      irsa_query_fn=irsa_query, esa_query_fn=esa_query, label=f"seeds_{name}", log=log)
        rec["seeds"] = {k: grec.get(k) for k in ("route", "status", "n_rows")}
        if not len(gdf):
            rec["degraded"].append(f"Gaia cone at {name} returned nothing ({grec.get('status')})")
            rec["status"] = VERDICT_NO_DATA
            return rec
        g = pd.to_numeric(gdf["phot_g_mean_mag"], errors="coerce")
        pe = pd.to_numeric(gdf["parallax_over_error"], errors="coerce")
        cand = gdf[(g >= float(sc["seed_g_min"])) & (g <= float(sc["seed_g_max"])) & (pe > float(sc["seed_plx_over_error_min"]))]
        cand = S.seed_in_box(cand, ra0, dec0, half)
        iso = G.isolation(cand, gdf, outer_arcsec=float(sc["isolation_radius_arcsec"]), outer_dmag=float(sc["isolation_dmag"]),
                          inner_arcsec=float(sc["inner_radius_arcsec"]), inner_dmag=float(sc["inner_dmag"]))
        rec["seeds"].update(n_in_box_mag_plx=int(len(cand)), n_isolated=int(iso["isolated"].sum()))
        seeds = iso[iso["isolated"]].copy()
        cap = int(max_seeds or sc.get("max_seeds_per_box", 200))
        if cap and len(seeds) > cap:
            seeds = seeds.sort_values("phot_g_mean_mag").head(cap)
        seeds = seeds.reset_index(drop=True)
        seeds.to_csv(seeds_csv, index=False)
        rec["seeds"]["n_seeds"] = int(len(seeds))
    if not len(seeds):
        rec["status"] = "NO_SEEDS"
        return rec
    # SPHEREx epochs are 2025-26: propagate the Gaia positions
    seeds = G.propagate(seeds, 2016.0, 2025.9)
    # ---- images ----
    imgs_csv = sdir / f"images_{name}.csv"
    if imgs_csv.exists():
        imgs = pd.read_csv(imgs_csv)
        rec["images"] = {"route": "cache", "n_selected": int(len(imgs))}
    else:
        r, cnt = _try(irsa_query, S.obscore_count_adql(sc["obscore_table"], ra0, dec0))
        rec["images"] = {"count_query": r}
        if cnt is not None and len(cnt):
            rec["images"]["n_products_covering_centre"] = int(pd.to_numeric(cnt.iloc[0, 0], errors="coerce"))
        cols = [c for c in S.OBSCORE_COLS if c not in set((probe_rec.get("spherex") or {}).get("obscore_missing_columns") or [])]
        r, df = _try(irsa_query, S.obscore_adql(sc["obscore_table"], ra0, dec0, cols), 2_000_000)
        rec["images"]["list_query"] = r
        if df is None or not len(df):
            rec["degraded"].append(f"obscore returned no products at {name}")
            rec["status"] = VERDICT_NO_DATA
            return rec
        imgs = S.select_images(df, sc["detector_bands_um"], int(max_images or sc["max_images_per_detector"]))
        imgs.to_csv(imgs_csv, index=False)
        rec["images"].update(n_products=int(len(df)), n_selected=int(len(imgs)),
                             per_detector={str(k): int(v) for k, v in imgs["detector"].value_counts(dropna=False).items()})
    # ---- cutouts + photometry, checkpointed ----
    prog_p = sdir / f"progress_{name}.json"
    prog = _read_json(prog_p, {"done": [], "failed": [], "route": None, "wavemap": {}}) or {}
    done = set(prog.get("done", []))
    samples_csv = sdir / f"samples_{name}.csv"
    wavemaps: dict[str, S.WaveMap] = {}
    full_crpix: dict[str, tuple[float, float]] = {}
    n_full = 0
    route = prog.get("route") or sc.get("cutout_route", "auto")
    size_arcsec = float(sc["box_arcmin"]) * 60.0
    stats = {"n_images_tried": 0, "n_images_ok": 0, "n_images_failed": 0, "n_samples": 0, "n_samples_ok": 0,
             "route_counts": {}, "wavemap_methods": {}, "fail_reasons": {}}
    for _, im in imgs.iterrows():
        if _time.monotonic() > deadline:
            rec["degraded"].append("time budget exhausted during cutouts")
            break
        oid = str(im.get("obs_id") or im.get("access_url"))
        if oid in done:
            continue
        det = str(im.get("detector"))
        url = S.product_url(sv["irsa_base"], str(im.get("access_url")))
        stats["n_images_tried"] += 1
        hd, used, err = None, None, ""
        ladder = ["ibe_cutout", "full"] if route == "auto" else [route]
        for rt in ladder:
            if rt == "full" and n_full >= int(sc.get("max_full_downloads", 12)):
                err = "full-download cap reached"
                continue
            try:
                hd = fetch_fn(S.cutout_url(url, ra0, dec0, size_arcsec) if rt == "ibe_cutout" else url)
                used = rt
                if rt == "full":
                    n_full += 1
                break
            except Exception as exc:                    # noqa: BLE001
                err = repr(exc)[:200]
                hd = None
        if hd is None:
            stats["n_images_failed"] += 1
            stats["fail_reasons"][err[:60]] = stats["fail_reasons"].get(err[:60], 0) + 1
            prog.setdefault("failed", []).append({"obs_id": oid, "error": err})
            continue
        stats["route_counts"][used] = stats["route_counts"].get(used, 0) + 1
        try:
            planes = S.image_planes(hd)
            hdr = planes["image_header"]
            wcs_cel = S.celestial_wcs(hdr)
            mjd = S.mjd_of(hdr)
            if not np.isfinite(mjd) and "t_min" in im and pd.notna(im["t_min"]):
                mjd = float(im["t_min"])
            local_map, offset = None, (0.0, 0.0)
            # wavelength: from this product if its WCS decodes, else the detector map + offset
            try:
                local_map = S.WaveMap.from_hdul(hd)
                if used == "ibe_cutout":
                    offset = S.cutout_offset(hdr, full_crpix.get(det)) or (0.0, 0.0)
            except Exception:                           # noqa: BLE001
                local_map = None
            if local_map is None:
                if det not in wavemaps:
                    if n_full >= int(sc.get("max_full_downloads", 12)):
                        raise RuntimeError("no wavelength map and the full-download cap is reached")
                    hf = fetch_fn(url)
                    n_full += 1
                    wavemaps[det] = S.WaveMap.from_hdul(hf)
                    fh = S._find_image_hdu(hf).header
                    full_crpix[det] = (float(fh.get("CRPIX1")), float(fh.get("CRPIX2")))
                    prog.setdefault("wavemap", {})[det] = {**wavemaps[det].fingerprint(), "crpix": list(full_crpix[det])}
                    try:
                        hf.close()
                    except Exception:                   # noqa: BLE001
                        pass
                offset = S.cutout_offset(hdr, full_crpix.get(det)) if used == "ibe_cutout" else (0.0, 0.0)
                if offset is None:
                    raise RuntimeError("cutout carries neither LTV nor a CRPIX shift; pixel offset unknown")
            method = local_map.method if local_map is not None else wavemaps[det].method
            stats["wavemap_methods"][method] = stats["wavemap_methods"].get(method, 0) + 1
            rows = S.samples_from_cutout(planes, wcs_cel, seeds, wavemaps.get(det), offset, sc,
                                         meta={"obs_id": oid, "detector": det, "mjd": mjd, "route": used,
                                               "wavemap_method": method},
                                         local_wavemap=local_map)
            if rows:
                pd.DataFrame(rows).to_csv(samples_csv, mode="a", header=not samples_csv.exists(), index=False)
            stats["n_samples"] += len(rows)
            stats["n_samples_ok"] += int(sum(1 for r in rows if r["ok"]))
            stats["n_images_ok"] += 1
            done.add(oid)
        except Exception as exc:                        # noqa: BLE001
            stats["n_images_failed"] += 1
            e = repr(exc)[:200]
            stats["fail_reasons"][e[:60]] = stats["fail_reasons"].get(e[:60], 0) + 1
            prog.setdefault("failed", []).append({"obs_id": oid, "error": e})
        finally:
            try:
                hd.close()
            except Exception:                           # noqa: BLE001
                pass
        if stats["n_images_tried"] % 25 == 0:
            prog["done"] = sorted(done)
            prog["route"] = used
            _write_json(prog_p, prog)
            print(f"[spark] {name}: {stats['n_images_ok']} ok / {stats['n_images_tried']} tried, "
                  f"{stats['n_samples_ok']} good samples, {S.elapsed(t0)} s", flush=True)
    prog["done"] = sorted(done)
    prog["route"] = route if route != "auto" else (max(stats["route_counts"], key=stats["route_counts"].get) if stats["route_counts"] else None)
    _write_json(prog_p, prog)
    rec["cutouts"] = stats
    rec["n_full_downloads"] = n_full
    rec["seconds"] = S.elapsed(t0)
    if stats["n_images_tried"] and not stats["n_images_ok"]:
        rec["degraded"].append("every image failed: " + "; ".join(list(stats["fail_reasons"])[:3]))
    rec["status"] = "OK" if samples_csv.exists() else VERDICT_NO_DATA
    return rec


def spherex_stage(conf: dict, out_dir: Path, *, shard: int = 0, n_shards: int = 1, irsa_query=None,
                  esa_query=None, fetch_fn=None, boxes: list[str] | None = None, max_images: int | None = None,
                  max_seeds: int | None = None, probe_rec: dict | None = None) -> dict:
    sc = conf["spherex"]
    sdir = out_dir / "spherex"
    sdir.mkdir(parents=True, exist_ok=True)
    t0 = _time.monotonic()
    led: dict = {"stage": "spherex", "shard": shard, "n_shards": n_shards, "generated_at": _now(),
                 "boxes": [], "queries": [], "degraded": []}
    if irsa_query is None or fetch_fn is None:
        led["status"] = VERDICT_NO_DATA
        led["degraded"].append("no query / fetch function")
        _write_json(sdir / f"spherex_s{shard}.json", led)
        return led
    probe_rec = probe_rec or _read_json(out_dir / "probe.json", {}) or {}
    irsa_tables = (probe_rec.get("gaia") or {}).get("irsa_gaia_table_usable") or []
    all_boxes = list(sc["boxes"])
    if boxes:
        all_boxes = [b for b in all_boxes if str(b["name"]) in set(boxes)]
    mine = _boxes_for_shard(all_boxes, shard, n_shards)
    deadline = t0 + float(sc.get("time_budget_s", 15000))
    for b in mine:
        r = spherex_box(conf, out_dir, b, irsa_query=irsa_query, esa_query=esa_query, fetch_fn=fetch_fn,
                        irsa_tables=irsa_tables, probe_rec=probe_rec, deadline=deadline, max_images=max_images,
                        max_seeds=max_seeds, log=led["queries"])
        led["boxes"].append(r)
        led["degraded"].extend(f"{b['name']}: {x}" for x in r.get("degraded", []))
        _write_json(sdir / f"spherex_s{shard}.json", led)
    led["elapsed_s"] = S.elapsed(t0)
    led["status"] = "OK" if any(r.get("status") == "OK" for r in led["boxes"]) else VERDICT_NO_DATA
    _write_json(sdir / f"spherex_s{shard}.json", led)
    scr = screen_spherex_local(conf, out_dir, shard=shard)
    led["screen"] = {k: scr.get(k) for k in ("n_stars_with_samples", "n_channels_tested", "n_survivors")}
    _write_json(sdir / f"spherex_s{shard}.json", led)
    return led


def screen_spherex_local(conf: dict, out_dir: Path, *, shard: int | None = None) -> dict:
    sc = conf["spherex"]
    sdir = out_dir / "spherex"
    sdir.mkdir(parents=True, exist_ok=True)
    parts, seeds_parts = [], []
    for p in sorted(glob.glob(str(sdir / "samples_*.csv"))):
        try:
            d = pd.read_csv(p)
        except Exception:                               # noqa: BLE001
            continue
        if len(d):
            d["box"] = Path(p).stem.replace("samples_", "")
            parts.append(d)
    for p in sorted(glob.glob(str(sdir / "seeds_*.csv"))):
        try:
            seeds_parts.append(pd.read_csv(p))
        except Exception:                               # noqa: BLE001
            continue
    samples = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    seeds = pd.concat(seeds_parts, ignore_index=True) if seeds_parts else pd.DataFrame()
    funnel: dict = {"generated_at": _now(), "shard": shard, "n_sample_files": len(parts),
                    "n_samples_raw": int(len(samples))}
    if len(samples):
        ok = samples["ok"].astype(str).str.lower().isin(["true", "1"]) if samples["ok"].dtype == object else samples["ok"].astype(bool)
        funnel["n_samples_flagged_or_failed"] = int((~ok).sum())
        funnel["fail_reasons"] = {str(k): int(v) for k, v in samples.loc[~ok, "reason"].value_counts().head(10).items()}
        samples = samples[ok]
    chans, f = S.screen_spherex(samples, seeds, sc, sc["detector_bands_um"]) if len(samples) else (pd.DataFrame(), {"n_seeds": int(len(seeds)), "n_stars_with_samples": 0, "n_channels_tested": 0, "n_survivors": 0})
    funnel.update(f)
    tag = f"_s{shard}" if shard is not None else ""
    chans.to_csv(sdir / f"channels{tag}.csv", index=False)
    _write_json(sdir / f"screen{tag}.json", funnel)
    return funnel


# --------------------------------------------------------------------------
# Assess
# --------------------------------------------------------------------------
def _merge_csv(pattern: str) -> pd.DataFrame:
    parts = []
    for p in sorted(glob.glob(pattern)):
        try:
            d = pd.read_csv(p)
        except Exception:                               # noqa: BLE001
            continue
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def assess(conf: dict, out_dir: Path) -> dict:
    """Merge every shard, redo the cross-shard recurrence and trials, write the summary."""
    ec, sc = conf["euclid"], conf["spherex"]
    edir, sdir = out_dir / "euclid", out_dir / "spherex"
    summary: dict = {"generated_at": _now(), "channel": "spark", "signatures": ["S48", "S49"],
                     "degraded": [], "euclid": {}, "spherex": {}}
    # ---- Euclid ----
    feats = _merge_csv(str(edir / "features_s*.csv"))
    if not len(feats):
        feats = _merge_csv(str(edir / "features.csv"))
    ledgers = [_read_json(p, {}) or {} for p in sorted(glob.glob(str(edir / "euclid_s*.json")))]
    n_stars_spec = sum(int(u.get("n_stars_with_spectra", 0) or 0) for led in ledgers for u in led.get("units", []))
    n_units = sum(len(led.get("units", [])) for led in ledgers)
    n_units_failed = sum(1 for led in ledgers for u in led.get("units", []) if u.get("lines_status") == "QUERY_FAILED")
    for led in ledgers:
        summary["degraded"].extend(f"euclid s{led.get('shard')}: {x}" for x in led.get("degraded", []))
    eu: dict = {"n_shard_ledgers": len(ledgers), "n_units": n_units, "n_units_failed": n_units_failed,
                "n_stars_with_spectra": n_stars_spec,
                "n_objects_with_spectra": sum(int(u.get("n_objects_with_spectra", 0) or 0) for led in ledgers for u in led.get("units", [])),
                "n_feature_rows_raw": sum(int(u.get("n_raw_rows", 0) or 0) for led in ledgers for u in led.get("units", [])),
                "gaia_routes": {k: v for led in ledgers for k, v in (led.get("gaia") or {}).items()},
                "wavelength_units_seen": sorted({str((u.get("wavelength") or {}).get("wavelength_unit")) for led in ledgers for u in led.get("units", []) if u.get("wavelength")})}
    if len(feats):
        if feats["field"].nunique() if "field" in feats else 0:
            pass
        d = feats.copy()
        d = d.drop_duplicates(["object_id", "wl_um"]) if "object_id" in d else d
        # cross-shard recurrence
        R = float(ec["resolving_power"])
        bin_um = float(ec["recurrence_bin_resel"])
        minst = int(ec["recurrence_min_stars"])
        strong = d[~d["low_snr"].astype(bool)]
        counts = {}
        if len(strong):
            b = np.floor(np.log(strong["wl_um"].to_numpy(float)) * R / bin_um).astype(int)
            counts = pd.DataFrame({"bin": b, "sid": strong["gaia_source_id"].to_numpy()}).groupby("bin")["sid"].nunique().to_dict()
        allb = np.floor(np.log(d["wl_um"].to_numpy(float)) * R / bin_um).astype(int)
        d["recurrence_n_stars"] = [max(int(counts.get(x, 0)), int(counts.get(x - 1, 0)), int(counts.get(x + 1, 0))) for x in allb]
        d["recurrent_wavelength"] = d["recurrence_n_stars"] >= minst
        vet = list(E.VETO_NAMES)
        d["vetoes"] = ["|".join(v for v in vet if bool(r[v])) for _, r in d[vet].iterrows()]
        d["survivor"] = d["vetoes"] == ""
        nres = E.n_resels(ec["band_um"], R)
        nstars = n_stars_spec or int(d["gaia_source_id"].nunique())
        ntrials = max(1, nstars * nres)
        d["p_global"] = np.minimum(1.0, d["p_single"] * ntrials)
        d["significant_after_trials"] = d["p_global"] < float(ec["trials_alpha"])
        d = d.sort_values(["survivor", "snr"], ascending=[False, False]).reset_index(drop=True)
        d.to_csv(edir / "features_all.csv", index=False)
        eu.update(n_features_on_stars=int(len(d)), n_stars_with_features=int(d["gaia_source_id"].nunique()),
                  n_pass_snr=int((~d["low_snr"].astype(bool)).sum()),
                  veto_counts={v: int(d[v].astype(bool).sum()) for v in vet},
                  veto_counts_among_snr_pass={v: int((d[v].astype(bool) & ~d["low_snr"].astype(bool)).sum()) for v in vet},
                  n_survivors=int(d["survivor"].sum()),
                  n_survivors_significant=int((d["survivor"] & d["significant_after_trials"]).sum()),
                  n_resels=nres, n_trials=ntrials,
                  recurrent_bins=[{"wl_um": round(float(math.exp((k + 0.5) * bin_um / R)), 4), "n_stars": int(v)} for k, v in counts.items() if v >= minst],
                  industrial_flag_counts={str(k): int(v) for k, v in d.loc[d["industrial_flag"].fillna("") != "", "industrial_flag"].value_counts().items()},
                  survivors=d[d["survivor"]].head(50).to_dict(orient="records"))
        eu["verdict"] = VERDICT_CANDIDATES if eu["n_survivors"] else VERDICT_NONE
    else:
        eu.update(n_features_on_stars=0, n_stars_with_features=0, n_survivors=0)
        eu["verdict"] = VERDICT_NO_DATA if not n_stars_spec and not eu["n_feature_rows_raw"] else VERDICT_NONE
        if eu["n_feature_rows_raw"] and not n_stars_spec:
            summary["degraded"].append("euclid: feature rows were acquired but no Gaia star matched any (Gaia route or epoch problem?)")
    summary["euclid"] = eu

    # ---- SPHEREx ----
    chans = _merge_csv(str(sdir / "channels_s*.csv"))
    if not len(chans):
        chans = _merge_csv(str(sdir / "channels.csv"))
    sled = [_read_json(p, {}) or {} for p in sorted(glob.glob(str(sdir / "spherex_s*.json")))]
    screens = [_read_json(p, {}) or {} for p in sorted(glob.glob(str(sdir / "screen_s*.json")))]
    for led in sled:
        summary["degraded"].extend(f"spherex s{led.get('shard')}: {x}" for x in led.get("degraded", []))
    sx: dict = {"n_shard_ledgers": len(sled),
                "n_boxes": sum(len(led.get("boxes", [])) for led in sled),
                "n_seeds": sum(int((b.get("seeds") or {}).get("n_seeds", 0) or 0) for led in sled for b in led.get("boxes", [])),
                "n_images_ok": sum(int((b.get("cutouts") or {}).get("n_images_ok", 0) or 0) for led in sled for b in led.get("boxes", [])),
                "n_images_tried": sum(int((b.get("cutouts") or {}).get("n_images_tried", 0) or 0) for led in sled for b in led.get("boxes", [])),
                "n_products_covering": {str(b.get("box")): (b.get("images") or {}).get("n_products_covering_centre") for led in sled for b in led.get("boxes", [])},
                "routes": {k: v for led in sled for b in led.get("boxes", []) for k, v in ((b.get("cutouts") or {}).get("route_counts") or {}).items()},
                "wavemap_methods": {k: v for led in sled for b in led.get("boxes", []) for k, v in ((b.get("cutouts") or {}).get("wavemap_methods") or {}).items()},
                "n_samples_good": sum(int((b.get("cutouts") or {}).get("n_samples_ok", 0) or 0) for led in sled for b in led.get("boxes", [])),
                "n_stars_with_samples": sum(int(s.get("n_stars_with_samples", 0) or 0) for s in screens),
                "n_channels_tested": sum(int(s.get("n_channels_tested", 0) or 0) for s in screens),
                "n_stars_insufficient": sum(int(s.get("n_stars_insufficient", 0) or 0) for s in screens)}
    if len(chans):
        d = chans.copy()
        s_tot = float(sc["excess_sigma_total"])
        hot = d[d["z_combined"].fillna(-99) >= s_tot]
        counts = hot.groupby(["detector", "channel_bin"])["source_id"].nunique().to_dict()
        key = list(zip(d["detector"], d["channel_bin"], strict=True))
        d["recurrence_n_stars"] = [int(counts.get(k, 0)) for k in key]
        d["recurrent_channel"] = d["recurrence_n_stars"] >= int(sc["recurrence_min_stars"])
        vet = ["stellar_line", "recurrent_channel", "recurrent_pixel"]
        d["vetoes"] = ["|".join(v for v in vet if bool(r[v])) for _, r in d[vet].iterrows()]
        d["survivor"] = (d["tier"] == "candidate") & (d["vetoes"] == "")
        ntrials = max(1, int(sx["n_channels_tested"]))
        d["p_global"] = np.minimum(1.0, d["p_single"] * ntrials)
        d["significant_after_trials"] = d["p_global"] < float(sc["trials_alpha"])
        d = d.sort_values(["survivor", "z_combined"], ascending=[False, False]).reset_index(drop=True)
        d.to_csv(sdir / "channels_all.csv", index=False)
        sx.update(n_channels=int(len(d)), n_channels_above_total=int(len(hot)),
                  n_candidates_before_vetoes=int((d["tier"] == "candidate").sum()),
                  n_watch=int((d["tier"] == "watch").sum()),
                  veto_counts={v: int((d[v].astype(bool) & (d["tier"] == "candidate")).sum()) for v in vet},
                  n_survivors=int(d["survivor"].sum()),
                  n_survivors_significant=int((d["survivor"] & d["significant_after_trials"]).sum()),
                  n_trials=ntrials,
                  recurrent_channels=[{"detector": k[0], "bin": int(k[1]), "n_stars": int(v)} for k, v in counts.items() if v >= int(sc["recurrence_min_stars"])],
                  industrial_flag_counts={str(k): int(v) for k, v in d.loc[(d["industrial_flag"].fillna("") != "") & (d["tier"] != "none"), "industrial_flag"].value_counts().items()},
                  survivors=d[d["survivor"]].head(50).to_dict(orient="records"),
                  watch=d[d["tier"] == "watch"].head(20).to_dict(orient="records"))
        sx["verdict"] = VERDICT_CANDIDATES if sx["n_survivors"] else VERDICT_NONE
    else:
        sx.update(n_channels=0, n_survivors=0)
        sx["verdict"] = VERDICT_NO_DATA if not sx["n_stars_with_samples"] else VERDICT_NONE
    summary["spherex"] = sx

    # ---- combined ----
    tested_e = int(eu.get("n_stars_with_features", 0) or 0) + int(eu.get("n_stars_with_spectra", 0) or 0)
    tested_s = int(sx.get("n_channels_tested", 0) or 0)
    n_surv = int(eu.get("n_survivors", 0) or 0) + int(sx.get("n_survivors", 0) or 0)
    if tested_e == 0 and tested_s == 0:
        verdict = VERDICT_NO_DATA
    elif n_surv:
        verdict = VERDICT_CANDIDATES
    else:
        verdict = VERDICT_NONE
    summary["degraded"] = sorted(set(summary["degraded"]))
    if summary["degraded"] and verdict != VERDICT_NO_DATA:
        verdict = f"DEGRADED ({len(summary['degraded'])} notes); {verdict}"
    summary["verdict"] = verdict
    summary["stage_counts"] = {
        "euclid_stars_with_spectra": eu.get("n_stars_with_spectra", 0),
        "euclid_stars_with_features": eu.get("n_stars_with_features", 0),
        "euclid_features_on_stars": eu.get("n_features_on_stars", 0),
        "euclid_survivors": eu.get("n_survivors", 0),
        "spherex_seeds": sx.get("n_seeds", 0), "spherex_stars_with_samples": sx.get("n_stars_with_samples", 0),
        "spherex_channels_tested": sx.get("n_channels_tested", 0), "spherex_survivors": sx.get("n_survivors", 0)}
    summary["coverage"] = {"euclid_units": f"{n_units - n_units_failed}_of_{n_units}_strips",
                           "spherex_images": f"{sx.get('n_images_ok', 0)}_of_{sx.get('n_images_tried', 0)}_images"}
    summary["not_a_limit"] = ("Counts of what was tested, not an occurrence limit; a null changes the question "
                              "(CLAUDE.md).  Every survivor is PENDING the vet in docs/spark.md §6.")
    _write_json(out_dir / "summary.json", summary)
    # candidates
    cands = []
    for r in eu.get("survivors", []):
        cands.append({"survey": "euclid", "gaia_source_id": r.get("gaia_source_id"), "object_id": r.get("object_id"),
                      "field": r.get("field"), "lambda_um": r.get("wl_um"), "snr": r.get("snr"), "n_dith": r.get("n_dith"),
                      "fwhm_resel": r.get("fwhm_resel"), "industrial_flag": r.get("industrial_flag"),
                      "single_line_z": r.get("single_line_z"), "p_global": r.get("p_global"),
                      "significant_after_trials": r.get("significant_after_trials"), "gaia_g": r.get("gaia_g"),
                      "gaia_plx_over_err": r.get("gaia_plx_over_err"), "gaia_ruwe": r.get("gaia_ruwe"),
                      "n_dispersion_neighbours": r.get("n_dispersion_neighbours"), "names": r.get("names"),
                      "ra": r.get("ra"), "dec": r.get("dec")})
    for r in sx.get("survivors", []):
        cands.append({"survey": "spherex", "gaia_source_id": r.get("source_id"), "detector": r.get("detector"),
                      "lambda_um": r.get("lambda_um"), "z_combined": r.get("z_combined"), "n_passes": r.get("n_passes"),
                      "n_positions": r.get("n_positions"), "industrial_flag": r.get("industrial_flag"),
                      "p_global": r.get("p_global"), "significant_after_trials": r.get("significant_after_trials"),
                      "excess_fraction": r.get("excess_fraction")})
    pd.DataFrame(cands).to_csv(out_dir / "candidates.csv", index=False)
    _write_json(out_dir / "candidates.json", {"generated_at": _now(), "n": len(cands), "candidates": cands})
    pd.DataFrame(L.line_table()).to_csv(out_dir / "lines.csv", index=False)
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def spark_run(stage: str = "all", out_dir: Path | str = "results/spark", *, shard: str | None = None,
              config: dict | None = None, fields: list[str] | None = None, boxes: list[str] | None = None,
              max_images: int | None = None, max_seeds: int | None = None, max_units: int | None = None,
              no_denominator: bool = False, offline: bool = False) -> dict:
    conf = config or load_spark_config()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    i, n = _parse_shard(shard)
    if offline:
        irsa = esa = fetch = http = None
    else:
        irsa = make_irsa_query(conf["services"]["irsa_tap"])
        esa = make_esa_query()
        fetch = make_fetcher(float(conf["spherex"].get("request_timeout_s", 120)))

        def http(url):
            import requests
            r = requests.get(url, timeout=60, headers={"User-Agent": "seti-spark/1.0"})
            return r.text if r.status_code == 200 else f"HTTP {r.status_code}"
    result: dict = {}
    if stage in ("probe", "all"):
        result["probe"] = probe(conf, out, irsa_query=irsa, esa_query=esa, fetch_fn=fetch, http_get=http)
    if stage in ("euclid", "all"):
        result["euclid"] = euclid_stage(conf, out, shard=i, n_shards=n, irsa_query=irsa, esa_query=esa,
                                        fields=fields, max_units=max_units, with_denominator=not no_denominator)
    if stage in ("spherex", "all"):
        result["spherex"] = spherex_stage(conf, out, shard=i, n_shards=n, irsa_query=irsa, esa_query=esa,
                                          fetch_fn=fetch, boxes=boxes, max_images=max_images, max_seeds=max_seeds)
    if stage == "screen":
        result["euclid_screen"] = screen_euclid_local(conf, out, shard=i if shard else None)
        result["spherex_screen"] = screen_spherex_local(conf, out, shard=i if shard else None)
    if stage in ("assess", "all"):
        result["summary"] = assess(conf, out)
    return result


def _add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--out-dir", default="results/spark")
    p.add_argument("--shard", default=None, help="i/n for the euclid and spherex stages")
    p.add_argument("--fields", default="", help="comma-separated Euclid field names (default: all)")
    p.add_argument("--boxes", default="", help="comma-separated SPHEREx box names (default: all)")
    p.add_argument("--max-images", type=int, default=None, help="per detector per box")
    p.add_argument("--max-seeds", type=int, default=None, help="per box")
    p.add_argument("--max-units", type=int, default=None, help="cap on (field, strip) units per shard")
    p.add_argument("--no-denominator", action="store_true", help="skip the stars-with-spectra strip queries")
    p.add_argument("--offline", action="store_true", help="no network (writes NO_DATA_REACHED ledgers)")


def _cmd_spark(args, _cfg=None):
    res = spark_run(args.stage, args.out_dir, shard=args.shard,
                    fields=[x for x in args.fields.split(",") if x] or None,
                    boxes=[x for x in args.boxes.split(",") if x] or None,
                    max_images=args.max_images, max_seeds=args.max_seeds, max_units=args.max_units,
                    no_denominator=args.no_denominator, offline=args.offline)
    if "summary" in res:
        s = res["summary"]
        print(json.dumps({"verdict": s.get("verdict"), "stage_counts": s.get("stage_counts"),
                          "degraded": s.get("degraded")}, indent=2, default=_json_default))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="seti spark", description=__doc__.split("\n\n")[0])
    _add_arguments(p)
    return _cmd_spark(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
