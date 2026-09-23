"""Stage orchestration for CENTURY.  Writes ``results/century/``.

Stages (``century --stage ... [--shard i/n]``):

``probe``    learn the real shapes of the DASCH DR7 API on the runner: the
             documentation pages as text, the OpenAPI schema if served, the
             daschlab source (for the AFLAGS/BFLAGS bits), one live
             ``queryexps`` / ``querycat`` / ``lightcurve`` chain on known
             stars, and the plate density of every configured field.
``targets``  per field: plate density, VSX/GCVS periodic variables bright
             enough for the plates, and bright reference-catalogue stars.
``acquire``  (sharded) fetch every target's light curve, checkpointed.
``screen``   (sharded) the three statistics per star; per-star annual and
             season tables written for the assess stage's field ensemble.
``assess``   field common mode, confirmation pass at full injection depth,
             Gaia context, the gauntlet, ``summary.json``.
``all``      targets (if missing) + acquire + screen for one shard.

Verdict discipline: a run that could not reach data says ``NO_DATA_REACHED``;
one that reached data but could test nothing says ``NO_TESTABLE_LIGHT_CURVES``;
one that tested and found nothing says ``NO_CANDIDATES`` --- a count, not an
occurrence limit --- and every degradation (flags not applied, ensemble not
applied, Gaia not reached, shard truncated) is a first-class field.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import re
import sys
import tarfile
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from ..knell.acquire import AcquisitionLog
from .api import (
    DASCHLAB_FILES,
    DASCHLAB_RAW,
    DOC_URLS,
    FLAG_SOURCE_FILES,
    dumps,
    fetch_text,
    html_to_text,
    jsonable,
    lightcurve,
    querycat,
)
from .cease import analyze_century, same_series_check
from .fade import analyze_fade, fade_from_annual
from .flags import FlagDefs, resolve_flagdefs
from .lightcurve import CenturyLC, attach_exptime, from_api_frame
from .scatter import (
    analyze_scatter,
    scatter_from_table,
    season_scatter_from_dict,
    season_scatter_to_dict,
)
from .step import MENZEL_GAP_END, MENZEL_GAP_START, fit_step_slope
from .targets import (
    EXPOSURE_KEY_COLS,
    PULSATOR_FIELD,
    field_tag,
    match_refcat,
    plate_density,
    select_bright,
    select_pulsators,
    select_variables,
)
from .vet import vet_row

DEFAULTS: dict = {
    "fields": [
        {"ra": 290.0, "dec": 44.5, "name": "kepler"},
        {"ra": 130.1, "dec": 19.7, "name": "praesepe"},
        {"ra": 299.6, "dec": 35.2, "name": "cygnus"},
        {"ra": 197.9, "dec": 29.4, "name": "sa57_ngp"},
        {"ra": 84.0, "dec": -1.2, "name": "orion"},
        {"ra": 10.7, "dec": 41.3, "name": "m31"},
    ],
    "radius_deg": 1.0,
    "gap": {"start": MENZEL_GAP_START, "end": MENZEL_GAP_END},
    "targets": {"mag_max": 13.0, "bright_mag_min": 8.0, "period_min": 0.2,
                "period_max": 100.0, "amp_min": 0.3, "max_variables_per_field": 0,
                "max_bright_per_field": 80, "refcat": "apass", "match_radius_arcsec": 15.0,
                "tile_arcmin": 10.0, "min_ndet_bright": 50},
    # The cessation population (docs/century.md §6.4): catalogued pulsators,
    # all sky.  Per-class period windows and caps live in pulsators.py
    # (DEFAULT_CLASS_LIMITS) and are overridden under ``classes``.
    "pulsators": {"enabled": True, "b_mean_max": 14.0, "b_mean_min": 8.0, "amp_min": 0.3,
                  "max_total": 2400, "catalogue_timeout_s": 900.0, "classes": {}},
    "acquire": {"pause_s": 0.5, "time_budget_s": 10800, "timeout_s": 180,
                "max_consecutive_failures": 25, "checkpoint_every": 25},
    "lightcurve": {"default_err": 0.15, "min_detections": 40, "min_span_yr": 20.0},
    "blocks": {"block_years": 2.0, "origin_year": 1880.0, "min_epochs_block": 20,
               "min_blocks": 4},
    "cease": {"half_width_frac": 0.005, "oversample": 5.0, "fap": 0.01, "n_null": 200,
              "n_trials": 120, "n_null_trial": 80, "n_trials_confirm": 300,
              "n_null_trial_confirm": 150, "eta_min": 0.90, "p_persist_max": 0.01,
              "min_pre_detected": 2, "min_post_informative": 2, "min_post_span_yr": 10.0,
              "mean_shift_max_mag": 0.10, "post_amp_sigma_max": 3.0, "drop_sigma_min": 5.0,
              "pdm_p_pre_max": 0.01, "pdm_p_post_min": 0.05, "pdm_null": 100,
              "var_drop_frac": 0.35, "blend_jump_max": 0.30, "vanish_margin_mag": 0.5,
              "vanish_rate_min": 0.3, "smear": True},
    "fade": {"margin": 1.0, "deep_extra": 1.0, "min_per_year": 3, "min_years": 15,
             "min_span_yr": 30.0, "fade_sigma_min": 5.0, "pre_sigma_min": 3.0,
             "series_min_years": 10, "series_min_span_yr": 25.0},
    "scatter": {"margin": 1.0, "deep_extra": 1.0, "season_years": 1.0,
                "min_epochs_season": 8, "min_seasons": 6, "slope_sigma_min": 3.0,
                "pre_rank_p_max": 0.01, "pre_slope_sigma_min": 3.0, "pre_loo_min": 2.0,
                "robust_sigma_min": 2.0},
    "ensemble": {"min_stars": 8, "mag_bin": 1.0},
    # A wall clock for the screen stage, inside the job's timeout-minutes, so an
    # overrun still uploads the stars it did screen instead of being killed.
    "screen": {"time_budget_s": 7200},
    "vet": {"pm_max_masyr": 50.0, "bright_limit_mag": 8.0, "lpv_colour_min": 1.5,
            "mean_shift_max_mag": 0.10, "gaia_radius_arcsec": 5.0},
    "flag_bits": {"aflags": {}, "bflags": {}},
    "probe_stars": [
        {"name": "RR Lyr", "ra": 291.36631, "dec": 42.78435, "mag": 7.6},
        {"name": "SW And", "ra": 5.92967, "dec": 29.40099, "mag": 9.6},
        {"name": "XZ Cyg", "ra": 293.12290, "dec": 56.38810, "mag": 9.7},
        {"name": "RT Aur", "ra": 97.29296, "dec": 30.49364, "mag": 5.4},
    ],
}


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


def load_century_config(root: Path | None = None) -> dict:
    root = Path(root) if root is not None else repo_root()
    p = root / "config" / "century.yaml"
    try:
        import yaml
        if p.exists():
            return _deep_update(DEFAULTS, yaml.safe_load(p.read_text()) or {})
    except Exception as exc:                              # noqa: BLE001
        print(f"[century] config not loaded ({exc!r}); using defaults")
    return _deep_update(DEFAULTS, {})


def _gap(conf: dict) -> tuple[float, float]:
    g = conf.get("gap", {})
    return float(g.get("start", MENZEL_GAP_START)), float(g.get("end", MENZEL_GAP_END))


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(obj, indent=2))


def _shard_dir(out_root: Path, shard: tuple[int, int]) -> Path:
    return out_root / "shards" / f"{shard[0]}_of_{shard[1]}"


def parse_shard(s: str | None) -> tuple[int, int]:
    if not s:
        return 0, 1
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", str(s))
    if not m:
        raise ValueError(f"--shard must be i/n, got {s!r}")
    i, n = int(m.group(1)), int(m.group(2))
    if n < 1 or not (0 <= i < n):
        raise ValueError(f"--shard {s!r} out of range")
    return i, n


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def _fetch_daschlab_source(out_dir: Path, log: AcquisitionLog) -> dict[str, str]:
    """daschlab source files from GitHub, else the PyPI sdist.  Saved as text."""
    got: dict[str, str] = {}
    for fn in DASCHLAB_FILES:
        st, txt, err = fetch_text(DASCHLAB_RAW + fn)
        if st == 200 and txt:
            got[fn] = txt
            (out_dir / "daschlab_src").mkdir(parents=True, exist_ok=True)
            (out_dir / "daschlab_src" / fn).write_text(txt)
    log.record("daschlab_github", DASCHLAB_RAW, rows=len(got),
               error=(None if got else "no file fetched"))
    if "lightcurves.py" in got:
        return got
    # Fallback: the sdist on PyPI.
    st, txt, err = fetch_text("https://pypi.org/pypi/daschlab/json")
    if st != 200:
        log.record("daschlab_pypi_meta", "pypi json", error=err or f"HTTP {st}")
        return got
    try:
        meta = json.loads(txt)
        urls = [u for u in meta.get("urls", []) if u.get("packagetype") == "sdist"]
        if not urls:
            log.record("daschlab_pypi_sdist", "no sdist url", rows=0)
            return got
        import requests
        r = requests.get(urls[0]["url"], timeout=120)
        r.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
            for mem in tf.getmembers():
                base = mem.name.split("/")[-1]
                if "/daschlab/" in mem.name and base in DASCHLAB_FILES and mem.isfile():
                    f = tf.extractfile(mem)
                    if f is not None:
                        got[base] = f.read().decode("utf-8", "replace")
                        (out_dir / "daschlab_src").mkdir(parents=True, exist_ok=True)
                        (out_dir / "daschlab_src" / base).write_text(got[base])
        log.record("daschlab_pypi_sdist", urls[0]["url"], rows=len(got))
    except Exception as exc:                              # noqa: BLE001
        log.record("daschlab_pypi_sdist", "sdist download", error=repr(exc))
    return got


def _openapi_schemas(obj: dict) -> dict:
    """Request/response schema fragments for the three endpoints, if present."""
    out = {}
    paths = obj.get("paths", {}) if isinstance(obj, dict) else {}
    for p, spec in paths.items():
        if "dasch" not in p:
            continue
        for method, op in (spec or {}).items():
            if not isinstance(op, dict):
                continue
            body = op.get("requestBody", {})
            schema = None
            try:
                schema = body["content"]["application/json"]["schema"]
            except Exception:                             # noqa: BLE001
                schema = None
            out[f"{method.upper()} {p}"] = {"summary": op.get("summary", ""),
                                            "request_schema": schema,
                                            "parameters": op.get("parameters", [])}
    comps = obj.get("components", {}).get("schemas", {}) if isinstance(obj, dict) else {}
    keep = {k: v for k, v in comps.items()
            if re.search(r"(?i)querycat|queryexps|lightcurve|refcat|dasch", k)}
    return {"endpoints": out, "components": keep}


def stage_probe(conf: dict, out_root: Path) -> dict:
    out_dir = out_root / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "docs").mkdir(exist_ok=True)
    log = AcquisitionLog()
    rep: dict = {"stage": "probe", "started_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                 _time.gmtime())}

    # 1. documentation pages -> text
    docs: dict = {}
    for key, url in DOC_URLS.items():
        st, txt, err = fetch_text(url)
        docs[key] = {"url": url, "status": st, "bytes": len(txt), "error": err}
        log.record(f"doc_{key}", url, rows=(len(txt) if st == 200 else None),
                   error=(err or None if st == 200 else (err or f"HTTP {st}")))
        if st == 200 and txt:
            if "openapi" in key:
                try:
                    obj = json.loads(txt)
                    (out_dir / f"{key}.json").write_text(json.dumps(obj, indent=1)[:2_000_000])
                    docs[key]["schemas"] = _openapi_schemas(obj)
                except Exception as exc:                  # noqa: BLE001
                    docs[key]["parse_error"] = repr(exc)[:200]
                    (out_dir / "docs" / f"{key}.txt").write_text(txt[:200_000])
            else:
                text = html_to_text(txt)
                (out_dir / "docs" / f"{key}.txt").write_text(text[:300_000])
                docs[key]["text_bytes"] = len(text)
                if key == "dr7_index":
                    links = sorted(set(re.findall(r'href="(/dr7/[^"#]+/?)"', txt)))
                    docs[key]["links"] = links
                    for ln in links[:30]:
                        slug = ln.strip("/").replace("/", "_")
                        if slug in ("dr7", "dr7_web-apis", "dr7_lightcurve-columns"):
                            continue
                        st2, t2, e2 = fetch_text("https://dasch.cfa.harvard.edu" + ln)
                        if st2 == 200 and t2:
                            (out_dir / "docs" / f"{slug}.txt").write_text(
                                html_to_text(t2)[:300_000])
                            docs[f"link_{slug}"] = {"url": ln, "status": st2, "bytes": len(t2)}
    rep["docs"] = docs

    # 2. daschlab source -> flag bits
    src = _fetch_daschlab_source(out_dir, log)
    a, b = resolve_flagdefs(conf, [src.get(fn) for fn in FLAG_SOURCE_FILES])
    rep["flag_bits"] = {"aflags": a.as_dict(), "bflags": b.as_dict()}
    _write_json(out_dir / "flag_bits.json", rep["flag_bits"])
    # When the bits did not parse, commit the EVIDENCE rather than a silent
    # degradation: the flag-bearing lines of every daschlab file fetched, and
    # of the DR7 light-curve column documentation.  The bits can then be read
    # off the artefact and put into config/century.yaml as the fallback,
    # instead of guessing them from memory.
    if not (a.available or b.available):
        ev: dict = {}
        for fn, text in src.items():
            hits = [ln.rstrip()[:200] for ln in text.splitlines()
                    if re.search(r"(?i)flag", ln)]
            if hits:
                ev[f"daschlab::{fn}"] = hits[:200]
        doc_txt_path = out_dir / "docs" / "lc_columns.txt"
        if doc_txt_path.exists():
            dt = doc_txt_path.read_text()
            ev["doc::lc_columns"] = [ln.rstrip()[:200] for ln in dt.splitlines()
                                     if re.search(r"(?i)flag", ln)][:200]
        rep["flag_bit_evidence"] = ev
        rep["flag_bits_degraded"] = True
        _write_json(out_dir / "flag_bit_evidence.json", ev)
    if "lightcurves.py" in src:
        api_lines = [ln.strip() for ln in src["lightcurves.py"].splitlines()
                     if re.search(r"(?i)api|url|requests\.|payload|json=|refcat", ln)]
        rep["daschlab_api_lines_lightcurves"] = api_lines[:80]
    for fn in ("refcat.py", "query.py", "exposures.py", "apiclient.py", "__init__.py"):
        if fn in src:
            rep[f"daschlab_api_lines_{fn}"] = [
                ln.strip() for ln in src[fn].splitlines()
                if re.search(r"(?i)_API|api\.starglass|requests\.|payload|json=|\"refcat\"|radius", ln)
            ][:60]

    # 3. plate density of the configured fields
    dens = {}
    for f in conf["fields"]:
        tag = f.get("name") or field_tag(f["ra"], f["dec"], conf["radius_deg"])
        dens[tag] = plate_density(float(f["ra"]), float(f["dec"]), log=log)
        _time.sleep(float(conf["acquire"]["pause_s"]))
    rep["field_density"] = dens

    # 4. one live chain per probe star
    chains = []
    for ps in conf.get("probe_stars", [])[:4]:
        ch: dict = {"star": ps}
        r = querycat(ps["ra"], ps["dec"], 30.0, refcat=conf["targets"]["refcat"])
        ch["querycat"] = r.as_dict()
        log.record("probe_querycat", f"{ps['name']} r=30\"", rows=(r.n_rows if r.ok else None),
                   error=(None if r.ok else r.error))
        if r.ok and r.frame is not None and len(r.frame):
            ch["querycat_head"] = jsonable(r.frame.head(5).to_dict(orient="list"))
            row, _ = match_refcat(ps["ra"], ps["dec"], radius_arcsec=30.0,
                                  refcat=conf["targets"]["refcat"], mag_hint=ps.get("mag", np.nan))
            ch["matched"] = jsonable(row)
            if row is not None:
                lr = lightcurve(conf["targets"]["refcat"], row["gsc_bin_index"],
                                row["ref_number"], timeout_s=float(conf["acquire"]["timeout_s"]))
                ch["lightcurve"] = lr.as_dict()
                log.record("probe_lightcurve", f"{ps['name']} gbi={row['gsc_bin_index']} "
                           f"ref={row['ref_number']}", rows=(lr.n_rows if lr.ok else None),
                           error=(None if lr.ok else lr.error))
                if lr.ok and lr.frame is not None and len(lr.frame):
                    df = lr.frame
                    ch["lightcurve_head"] = jsonable(df.head(3).to_dict(orient="list"))
                    ch["lightcurve_dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
                    lc = from_api_frame(df, a, b)
                    if lc is not None:
                        yr = lc.year
                        ch["normalised"] = {
                            "n_raw": lc.n_raw, "n_det": lc.n_det, "n_nd": lc.n_nd,
                            "n_good": int(lc.good.sum()), "flags_applied": lc.flags_applied,
                            "exptime_unit": lc.exptime_unit,
                            "exptime_median_min": float(np.nanmedian(lc.exptime_min))
                            if np.isfinite(lc.exptime_min).any() else None,
                            "year_min": float(np.min(yr)) if lc.n_det else None,
                            "year_max": float(np.max(yr)) if lc.n_det else None,
                            "n_pre_gap": int(np.sum(yr < _gap(conf)[0])),
                            "n_post_gap": int(np.sum(yr >= _gap(conf)[1])),
                            "median_mag": lc.median_mag(),
                            "median_err": float(np.median(lc.err)) if lc.n_det else None,
                            "median_lim": float(np.nanmedian(lc.lim)) if lc.n_det else None,
                            "blend_fraction": lc.blend_fraction(),
                            "series_counts": {k: int(v) for k, v in zip(
                                *np.unique(lc.series.astype(str), return_counts=True),
                                strict=False)},
                        }
        chains.append(ch)
        _time.sleep(float(conf["acquire"]["pause_s"]))
    rep["chains"] = chains

    n_lc = sum(1 for c in chains if c.get("normalised", {}).get("n_det", 0) > 0)
    n_api = sum(1 for c in chains if c.get("querycat", {}).get("ok"))
    rep["verdict"] = ("API_REACHED_WITH_LIGHTCURVE" if n_lc else
                      "API_REACHED_NO_LIGHTCURVE" if n_api else "API_NOT_REACHED")
    rep["acquisition"] = log.as_dict()
    _write_json(out_dir / "probe.json", rep)
    print(f"[century/probe] verdict={rep['verdict']} chains_with_lc={n_lc}")
    return rep


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


def stage_targets(conf: dict, out_root: Path, *, fields=None, radius_deg: float | None = None,
                  max_variables: int | None = None, max_bright: int | None = None,
                  include_bright: bool = True, max_pulsators: int | None = None,
                  include_pulsators: bool | None = None,
                  pulsator_catalogue_fn=None) -> pd.DataFrame:
    """Three populations, each carrying the question it can answer.

    * **pulsators** (all sky, by catalogued TYPE) carry the cessation test ---
      an oscillator can stop, an eclipse cannot (docs/century.md §6.4);
    * **bright** refcat stars in the configured fields carry the fade and
      rising-scatter tests, which need the field ensemble;
    * field **variables** (the old per-field VSX cone, any periodic type) are
      off by default (``max_variables_per_field: 0``): run 35748748365 showed
      they are eclipsers, whose geometric period cannot cease.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    tc = conf["targets"]
    fields = fields if fields is not None else conf["fields"]
    radius = float(conf["radius_deg"] if radius_deg is None else radius_deg)
    mv = int(tc["max_variables_per_field"] if max_variables is None else max_variables)
    mb = int(tc["max_bright_per_field"] if max_bright is None else max_bright)
    log = AcquisitionLog()
    frames = []
    dens = {}
    # The same queryexps that measures each field's plate density also carries
    # the EXPOSURE TIME of every plate, which the light curves do not.  It is
    # collected here and written beside targets.csv so the shards can join it
    # on without spending a request of their own.
    exp_tables: list[pd.DataFrame] = []
    for f in fields:
        ra, dec = float(f["ra"]), float(f["dec"])
        tag = f.get("name") or field_tag(ra, dec, radius)
        dens[tag] = plate_density(ra, dec, log=log, exposures_out=exp_tables)
        v = select_variables(ra, dec, radius, log=log, mag_max=tc["mag_max"],
                             period_min=tc["period_min"], period_max=tc["period_max"],
                             amp_min=tc["amp_min"], max_targets=mv)
        if len(v):
            v["field"] = tag
            frames.append(v)
        if include_bright and mb > 0:
            bstars = select_bright(ra, dec, radius, log=log, mag_max=tc["mag_max"],
                                   mag_min=tc["bright_mag_min"], refcat=tc["refcat"],
                                   tile_arcmin=tc["tile_arcmin"], max_targets=mb,
                                   min_ndet=tc["min_ndet_bright"],
                                   pause_s=float(conf["acquire"]["pause_s"]))
            if len(bstars):
                bstars["field"] = tag
                frames.append(bstars)
        _time.sleep(float(conf["acquire"]["pause_s"]))
    pc = conf.get("pulsators", {}) or {}
    do_puls = bool(pc.get("enabled", True)) if include_pulsators is None else bool(include_pulsators)
    puls_funnel: dict = {"enabled": do_puls}
    if do_puls:
        pdf, pf = select_pulsators(conf, log=log, max_total=max_pulsators,
                                   catalogue_fn=pulsator_catalogue_fn)
        puls_funnel.update(pf)
        if len(pdf):
            # A pulsator that also fell in a field cone (as a field variable, or
            # as a bright refcat star) is measured once, as a pulsator --- and a
            # large-amplitude variable must not sit in the bright stars' field
            # ensemble, whose job is to describe the PLATES.
            if frames:
                prev = pd.concat(frames, ignore_index=True)
                pv = prev[prev["kind"].isin(["variable", "bright"])] if "kind" in prev \
                    else prev.iloc[0:0]
                if len(pv):
                    cosd = np.cos(np.radians(pdf["dec"].to_numpy(float)))[:, None]
                    sep = 3600.0 * np.hypot(
                        (pdf["ra"].to_numpy(float)[:, None] - pv["ra"].to_numpy(float)[None, :])
                        * cosd,
                        pdf["dec"].to_numpy(float)[:, None] - pv["dec"].to_numpy(float)[None, :])
                    dup_v = (sep <= 5.0).any(axis=0)
                    if dup_v.any():
                        drop_names = set(pv["name"].astype(str).to_numpy()[dup_v])
                        frames = [f[~(f["kind"].isin(["variable", "bright"])
                                      & f["name"].astype(str).isin(drop_names))]
                                  for f in frames]
                        puls_funnel["n_field_stars_superseded"] = int(dup_v.sum())
            frames.append(pdf)
    cols = ["target_id", "name", "ra", "dec", "kind", "vtype", "period_cat", "mag_cat",
            "amp_cat", "source", "field", "gsc_bin_index", "ref_number", "pm_total_masyr",
            "colour", "n_det_cat", "pulsator_class", "mag_max_cat", "band_cat"]
    if frames:
        df = pd.concat(frames, ignore_index=True)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df["target_id"] = np.arange(len(df))
        df = df[cols]
    else:
        df = pd.DataFrame(columns=cols)
    df.to_csv(out_root / "targets.csv", index=False)
    if exp_tables:
        et = pd.concat(exp_tables, ignore_index=True).drop_duplicates(
            subset=list(EXPOSURE_KEY_COLS), keep="first")
        et.to_csv(out_root / "plate_exptime.csv", index=False)
    else:
        et = pd.DataFrame(columns=[*EXPOSURE_KEY_COLS, "exptime_min"])
        et.to_csv(out_root / "plate_exptime.csv", index=False)
    summ = {"n_plate_exptimes": int(len(et)),
            "n_targets": int(len(df)),
            "n_variables": int((df["kind"] == "variable").sum()) if len(df) else 0,
            "n_bright": int((df["kind"] == "bright").sum()) if len(df) else 0,
            "n_pulsators": int((df["kind"] == "pulsator").sum()) if len(df) else 0,
            "n_pulsators_by_class": ({str(k): int(v) for k, v in
                                      df.loc[df["kind"] == "pulsator", "pulsator_class"]
                                      .value_counts().items()} if len(df) else {}),
            "pulsators": puls_funnel,
            "fields": dens, "radius_deg": radius, "acquisition": log.as_dict()}
    _write_json(out_root / "targets_summary.json", summ)
    print(f"[century/targets] {summ['n_targets']} targets "
          f"({summ['n_pulsators']} pulsators {summ['n_pulsators_by_class']}, "
          f"{summ['n_variables']} field variables, {summ['n_bright']} bright)")
    return df


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------


def _runner_flagdefs(conf: dict, log: AcquisitionLog) -> tuple[FlagDefs, FlagDefs, str]:
    """Flag bits from the live daschlab source, else the config, else none."""
    texts, fetched = [], {}
    for fn in FLAG_SOURCE_FILES:
        st, txt, err = fetch_text(DASCHLAB_RAW + fn)
        fetched[fn] = int(st) if st is not None else 0
        if st == 200 and txt:
            texts.append(txt)
    a, b = resolve_flagdefs(conf, texts)
    src = "daschlab_source" if (a.source == "daschlab_source" or b.source == "daschlab_source") \
        else ("config" if (a.available or b.available) else "none")
    log.record("flag_definitions", DASCHLAB_RAW + ",".join(FLAG_SOURCE_FILES),
               rows=len(a.bits) + len(b.bits), extra={"source": src, "http": fetched})
    return a, b, src


def _shard_exposure_table(mine: pd.DataFrame, conf: dict,
                          log: AcquisitionLog) -> tuple[pd.DataFrame, str]:
    """Rebuild the plate exposure-time table from this shard's own fields.

    Exposure time is a property of the PLATE, so one ``queryexps`` at the
    median position of the shard's stars in a field covers every star in it.
    Returns ``(table, source)``; an empty table and ``"none"`` when no field
    answered, which is a degradation the caller records, never a guess.
    """
    if mine is not None and len(mine) and "kind" in mine.columns:
        # All-sky pulsators get their own queryexps at their own position in
        # stage_acquire; a "field centre" of stars scattered over the sky is
        # a point no plate list describes.
        mine = mine[mine["kind"].astype(str) != "pulsator"]
    if mine is None or not len(mine) or "ra" not in mine.columns:
        return pd.DataFrame(), "none"
    frames: list[pd.DataFrame] = []
    tried = ok = 0
    fields = (mine.groupby("field")[["ra", "dec"]].median().reset_index()
              if "field" in mine.columns
              else pd.DataFrame([{"field": "all", "ra": float(mine["ra"].median()),
                                  "dec": float(mine["dec"].median())}]))
    for _, f in fields.iterrows():
        ra, dec = float(f["ra"]), float(f["dec"])
        if not (np.isfinite(ra) and np.isfinite(dec)):
            continue
        tried += 1
        collected: list[pd.DataFrame] = []
        plate_density(ra, dec, log=log, exposures_out=collected,
                      timeout_s=float(conf["acquire"]["timeout_s"]))
        if collected:
            ok += 1
            frames.extend(collected)
        _time.sleep(float(conf["acquire"]["pause_s"]))
    if not frames:
        log.record("plate_exptime_rebuild", f"{tried} field centres, none answered", rows=0)
        return pd.DataFrame(), "none"
    et = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=list(EXPOSURE_KEY_COLS), keep="first")
    log.record("plate_exptime_rebuild", f"{ok}/{tried} field centres answered", rows=int(len(et)))
    return et, "acquire_stage_rebuild"


def _save_shard_lcs(path: Path, store: dict[str, dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for tid, d in store.items():
        for k, v in d.items():
            arrays[f"s{tid}__{k}"] = v
    np.savez_compressed(path, **arrays)
    _write_json(path.with_name("lightcurves_meta.json"), meta)


def load_shard_lcs(shard_dir: Path) -> tuple[dict[str, CenturyLC], dict]:
    p = shard_dir / "lightcurves.npz"
    m = shard_dir / "lightcurves_meta.json"
    if not p.exists() or not m.exists():
        return {}, {}
    meta = json.loads(m.read_text())
    z = np.load(p, allow_pickle=False)
    out: dict[str, CenturyLC] = {}
    per = meta.get("stars", {})
    for tid, info in per.items():
        d = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(f"s{tid}__")}
        if "t" not in d:
            continue
        out[str(tid)] = CenturyLC.from_arrays(
            d, flags_applied=bool(info.get("flags_applied", False)),
            n_raw=int(info.get("n_raw", 0)), columns=info.get("columns", []),
            exptime_unit=str(info.get("exptime_unit", "unknown")))
    return out, meta


def stage_acquire(conf: dict, out_root: Path, shard: tuple[int, int], *,
                  time_budget_s: float | None = None, max_targets: int | None = None,
                  pause_s: float | None = None) -> dict:
    ac = conf["acquire"]
    tc = conf["targets"]
    budget = float(ac["time_budget_s"] if time_budget_s is None else time_budget_s)
    pause = float(ac["pause_s"] if pause_s is None else pause_s)
    sd = _shard_dir(out_root, shard)
    sd.mkdir(parents=True, exist_ok=True)
    log = AcquisitionLog()
    tpath = out_root / "targets.csv"
    if not tpath.exists():
        log.record("targets_csv", str(tpath), error="missing")
        rep = {"stage": "acquire", "shard": list(shard), "n_targets": 0, "n_fetched": 0,
               "verdict": "NO_TARGETS", "acquisition": log.as_dict()}
        _write_json(sd / "acquire_summary.json", rep)
        return rep
    targets = pd.read_csv(tpath)
    mine = targets.iloc[shard[0]::shard[1]].copy()
    if max_targets:
        mine = mine.head(int(max_targets))
    a, b, flag_src = _runner_flagdefs(conf, log)
    # Exposure durations from the targets stage's queryexps.  DR7 light curves
    # carry no exposure time and a long exposure smears a short period, so
    # without this the injection efficiency in the post-gap blocks is computed
    # as if the later plates smeared exactly as much as the earlier ones.
    epath = out_root / "plate_exptime.csv"
    exp_tab = pd.read_csv(epath) if epath.exists() else pd.DataFrame()
    exp_source = "targets_stage" if len(exp_tab) else "none"
    if not len(exp_tab):
        # The shard builds it itself rather than doing without.  The file
        # travels between jobs as an artifact, and an artifact list is part of
        # the workflow file, which is fixed when a run is dispatched — so a
        # sweep launched from an older workflow, or a reduce-only re-run, would
        # otherwise silently lose the exposure times and leave every smear
        # unmodelled.  One queryexps per field the shard actually has stars in,
        # at the median position of those stars, is six requests, not six
        # hundred: exposure time is a property of the plate, not of the star.
        exp_tab, exp_source = _shard_exposure_table(mine, conf, log)
        if len(exp_tab):
            epath.parent.mkdir(parents=True, exist_ok=True)
            exp_tab.to_csv(epath, index=False)
    log.record("plate_exptime", str(epath), rows=len(exp_tab),
               error=(None if len(exp_tab) else "absent and not rebuildable"),
               extra={"source": exp_source})
    meta_exptime = {"table_rows": int(len(exp_tab)), "source": exp_source,
                    "n_matched_stars": 0, "match_keys": {}, "unmatched_stars": 0}

    # Resume from a checkpoint if one exists.
    store: dict[str, dict] = {}
    meta: dict = {"shard": list(shard), "stars": {}, "flag_source": flag_src,
                  "refcat": tc["refcat"]}
    prev, prev_meta = load_shard_lcs(sd)
    for tid, lc in prev.items():
        store[tid] = lc.to_arrays()
    if prev_meta.get("stars"):
        meta["stars"].update(prev_meta["stars"])
    status_path = sd / "acquire.jsonl"
    done = set(meta["stars"].keys())
    if status_path.exists():
        for ln in status_path.read_text().splitlines():
            try:
                done.add(str(json.loads(ln)["target_id"]))
            except Exception:                             # noqa: BLE001
                pass

    # Service probe, recorded as its own stage: a proxy refusal must not look
    # like an empty sky.
    pr = querycat(180.0, 20.0, 60.0, refcat=tc["refcat"], timeout_s=float(ac["timeout_s"]))
    log.record("dasch_service_probe", "querycat (180,+20) r=60\"",
               rows=(pr.n_rows if pr.ok else None), error=(None if pr.ok else pr.error),
               extra={"status": pr.status, "body_head": pr.body_head[:200]})

    t0 = _time.monotonic()
    n_fetched = n_failed = n_empty = n_skipped = 0
    consecutive = 0
    truncated = False
    since_ckpt = 0
    for _, row in mine.iterrows():
        tid = str(int(row["target_id"]))
        if tid in done:
            n_skipped += 1
            continue
        if _time.monotonic() - t0 > budget:
            truncated = True
            break
        if consecutive >= int(ac["max_consecutive_failures"]):
            log.record("abort", f"{consecutive} consecutive failures", error="aborted")
            truncated = True
            break
        rec = {"target_id": int(tid), "name": str(row["name"]), "kind": str(row["kind"])}
        gbi, rn = row.get("gsc_bin_index"), row.get("ref_number")
        if not (np.isfinite(gbi) and np.isfinite(rn)):
            m, qr = match_refcat(float(row["ra"]), float(row["dec"]),
                                 radius_arcsec=float(tc["match_radius_arcsec"]),
                                 refcat=tc["refcat"], mag_hint=float(row.get("mag_cat", np.nan)),
                                 timeout_s=float(ac["timeout_s"]))
            if m is None:
                rec.update({"status": "no_refcat_match" if qr.ok else "querycat_failed",
                            "error": qr.error, "http": qr.status})
                n_failed += int(not qr.ok)
                n_empty += int(qr.ok)
                consecutive = consecutive + 1 if not qr.ok else 0
                _append(status_path, rec)
                _time.sleep(pause)
                continue
            gbi, rn = m["gsc_bin_index"], m["ref_number"]
            rec.update({"sep_arcsec": m.get("sep_arcsec"), "refcat_mag": m.get("mag_cat"),
                        "pm_total_masyr": m.get("pm_total_masyr"), "colour": m.get("colour"),
                        "n_det_cat": m.get("n_det_cat")})
            _time.sleep(pause)
        lr = lightcurve(tc["refcat"], int(gbi), int(rn), timeout_s=float(ac["timeout_s"]))
        rec.update({"gsc_bin_index": int(gbi), "ref_number": int(rn), "http": lr.status,
                    "elapsed_s": round(lr.elapsed_s, 2)})
        if not lr.ok:
            rec.update({"status": "lightcurve_failed", "error": lr.error,
                        "body_head": lr.body_head[:200]})
            n_failed += 1
            consecutive += 1
            _append(status_path, rec)
            _time.sleep(pause)
            continue
        consecutive = 0
        lc = from_api_frame(lr.frame, a, b, default_err=float(conf["lightcurve"]["default_err"]))
        if lc is None or lc.n_det == 0:
            rec.update({"status": "empty_lightcurve", "n_rows": lr.n_rows,
                        "columns": lr.columns[:30]})
            n_empty += 1
            _append(status_path, rec)
            _time.sleep(pause)
            continue
        star_tab = exp_tab
        pos_plates: dict = {}
        if str(row.get("kind")) == "pulsator":
            # An all-sky pulsator shares no field with anything: its exposure
            # durations come from a queryexps at ITS position (the plates that
            # cover it), never from a neighbouring field's plate list, which
            # would leave most of its plates unmatched and so unsmeared.
            collected: list[pd.DataFrame] = []
            dens_star = plate_density(float(row["ra"]), float(row["dec"]), log=None,
                                      exposures_out=collected,
                                      timeout_s=float(ac["timeout_s"]))
            star_tab = collected[0] if collected else pd.DataFrame()
            pos_plates = {k: dens_star.get(k) for k in ("ok", "n_plates", "n_pre_gap",
                                                        "n_post_gap", "lim_median", "lim_p90",
                                                        "n_exptimes")}
            rec["queryexps_ok"] = bool(dens_star.get("ok"))
            rec["n_plates_at_position"] = int(dens_star.get("n_plates") or 0)
            _time.sleep(pause)
        eprov = attach_exptime(lc, star_tab) if len(star_tab) else {"key": "none",
                                                                    "matched_det": 0}
        if eprov.get("matched_det"):
            meta_exptime["n_matched_stars"] += 1
            k = str(eprov.get("key"))
            meta_exptime["match_keys"][k] = meta_exptime["match_keys"].get(k, 0) + 1
        else:
            meta_exptime["unmatched_stars"] += 1
        store[tid] = lc.to_arrays()
        meta["stars"][tid] = {"n_raw": lc.n_raw, "n_det": lc.n_det, "n_nd": lc.n_nd,
                              "flags_applied": lc.flags_applied, "columns": lc.columns[:40],
                              "exptime_unit": lc.exptime_unit, "exptime": eprov,
                              "gsc_bin_index": int(gbi), "ref_number": int(rn),
                              "pm_total_masyr": rec.get("pm_total_masyr"),
                              "colour": rec.get("colour"),
                              "plates_at_position": pos_plates}
        rec.update({"status": "ok", "n_rows": lr.n_rows, "n_det": lc.n_det, "n_nd": lc.n_nd})
        n_fetched += 1
        since_ckpt += 1
        _append(status_path, rec)
        if since_ckpt >= int(ac["checkpoint_every"]):
            _save_shard_lcs(sd / "lightcurves.npz", store, meta)
            since_ckpt = 0
        _time.sleep(pause)
    _save_shard_lcs(sd / "lightcurves.npz", store, meta)
    log.record("lightcurves", f"shard {shard[0]}/{shard[1]}: {len(mine)} targets",
               rows=n_fetched, extra={"n_failed": n_failed, "n_empty": n_empty,
                                      "n_skipped_done": n_skipped, "truncated": truncated,
                                      "elapsed_s": round(_time.monotonic() - t0, 1)})
    rep = {"stage": "acquire", "shard": list(shard), "n_targets": int(len(mine)),
           "n_fetched": n_fetched, "n_failed": n_failed, "n_empty": n_empty,
           "n_resumed": len(prev), "truncated": truncated, "flag_source": flag_src,
           "service_probe_ok": bool(pr.ok), "exptime": meta_exptime,
           "verdict": ("NO_DATA_REACHED" if (n_fetched + len(prev)) == 0 and not pr.ok
                       else "NO_LIGHTCURVES" if (n_fetched + len(prev)) == 0
                       else "LIGHTCURVES_FETCHED"),
           "acquisition": log.as_dict()}
    _write_json(sd / "acquire_summary.json", rep)
    print(f"[century/acquire] shard {shard}: fetched={n_fetched} failed={n_failed} "
          f"empty={n_empty} truncated={truncated} flags={flag_src}")
    return rep


def _append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------


def screen_star(lc: CenturyLC, target: dict, conf: dict, *, rng=None,
                confirm: bool = False) -> dict:
    """All three statistics for one star.  Pure given the light curve."""
    rng = np.random.default_rng(rng)
    gap = _gap(conf)
    bc, cc, fc, sc, lcc = (conf["blocks"], conf["cease"], conf["fade"], conf["scatter"],
                           conf["lightcurve"])
    row: dict = {"target_id": target.get("target_id"), "name": target.get("name"),
                 "kind": target.get("kind"), "field": target.get("field"),
                 "vtype": target.get("vtype", ""), "period_cat": target.get("period_cat"),
                 "pulsator_class": ("" if target.get("pulsator_class") is None
                                    or (isinstance(target.get("pulsator_class"), float)
                                        and np.isnan(target.get("pulsator_class")))
                                    else str(target.get("pulsator_class"))),
                 "amp_cat": target.get("amp_cat"), "mag_cat": target.get("mag_cat"),
                 "ra": target.get("ra"), "dec": target.get("dec"),
                 "n_raw": lc.n_raw, "n_det": lc.n_det, "n_nd": lc.n_nd,
                 "n_good": int(lc.good.sum()), "blend_fraction": lc.blend_fraction(),
                 "flags_applied": lc.flags_applied, "median_mag": lc.median_mag(),
                 "span_yr": lc.span_years(), "exptime_unit": lc.exptime_unit}
    yr = lc.year[lc.good]
    row["n_pre_gap"] = int(np.sum(yr < gap[0])) if yr.size else 0
    row["n_post_gap"] = int(np.sum(yr >= gap[1])) if yr.size else 0
    row["series_dominant"], row["series_dominant_frac"] = lc.dominant_series()
    usable = (row["n_good"] >= int(lcc["min_detections"])
              and row["span_yr"] >= float(lcc["min_span_yr"]))
    row["usable"] = bool(usable)
    if not usable:
        row["cess_status"] = "not_run"
        row["fade_status"] = row["rust_status"] = "insufficient_data"
        return row

    period = float(target.get("period_cat") or np.nan)
    amp_cat = float(target.get("amp_cat") or 0.0)
    if not np.isfinite(amp_cat):
        amp_cat = 0.0
    f_ref = float("nan")
    if np.isfinite(period) and period > 0:
        n_trials = int(cc["n_trials_confirm"] if confirm else cc["n_trials"])
        n_null_trial = int(cc["n_null_trial_confirm"] if confirm else cc["n_null_trial"])
        res = analyze_century(
            lc, period, block_years=bc["block_years"], origin_year=bc["origin_year"],
            min_epochs_block=bc["min_epochs_block"], min_blocks=bc["min_blocks"],
            half_width_frac=cc["half_width_frac"], oversample=cc["oversample"], fap=cc["fap"],
            n_null=cc["n_null"], n_trials=n_trials, n_null_trial=n_null_trial,
            eta_min=cc["eta_min"], p_persist_max=cc["p_persist_max"],
            min_pre_detected=cc["min_pre_detected"],
            min_post_informative=cc["min_post_informative"],
            min_post_span_yr=cc["min_post_span_yr"], mean_shift_max_mag=cc["mean_shift_max_mag"],
            post_amp_sigma_max=cc["post_amp_sigma_max"], drop_sigma_min=cc["drop_sigma_min"],
            pdm_p_pre_max=cc["pdm_p_pre_max"], pdm_p_post_min=cc["pdm_p_post_min"],
            pdm_null=cc["pdm_null"], var_drop_frac=cc["var_drop_frac"],
            blend_jump_max=cc["blend_jump_max"], vanish_margin_mag=cc["vanish_margin_mag"],
            vanish_rate_min=cc["vanish_rate_min"], gap=gap, smear=bool(cc["smear"]),
            blind_check=True, rng=rng)
        row.update({f"cess_{k}": v for k, v in res.as_dict().items()})
        f_ref = res.f_ref
        if res.status == "cessation" or (confirm and res.status in ("cessation", "low_efficiency")):
            row.update(same_series_check(
                lc, res, min_epochs_block=bc["min_epochs_block"], block_years=bc["block_years"],
                origin_year=bc["origin_year"], rng=rng, half_width_frac=cc["half_width_frac"],
                fap=cc["fap"], n_null=cc["n_null"], n_trials=max(n_trials // 2, 40),
                n_null_trial=n_null_trial, eta_min=cc["eta_min"], gap=gap,
                smear=bool(cc["smear"])))
    else:
        row["cess_status"] = "no_catalogue_period"

    fres, tab = analyze_fade(lc, amp_cat=amp_cat, margin=fc["margin"],
                             deep_extra=fc["deep_extra"], min_per_year=fc["min_per_year"],
                             min_years=fc["min_years"], min_span_yr=fc["min_span_yr"],
                             fade_sigma_min=fc["fade_sigma_min"], pre_sigma_min=fc["pre_sigma_min"],
                             series_min_years=fc["series_min_years"],
                             series_min_span_yr=fc["series_min_span_yr"], gap=gap)
    row.update({f"fade_{k}": v for k, v in fres.as_dict().items()})
    row["annual"] = tab.to_dict(orient="list") if len(tab) else {}

    sres, ss = analyze_scatter(lc, f_ref=f_ref, amp_cat=amp_cat, margin=sc["margin"],
                               deep_extra=sc["deep_extra"], season_years=sc["season_years"],
                               min_epochs_season=sc["min_epochs_season"],
                               min_seasons=sc["min_seasons"], slope_sigma_min=sc["slope_sigma_min"],
                               pre_rank_p_max=sc["pre_rank_p_max"],
                               pre_slope_sigma_min=sc["pre_slope_sigma_min"],
                               pre_loo_min=sc["pre_loo_min"],
                               robust_sigma_min=sc["robust_sigma_min"], gap=gap)
    row.update({f"rust_{k}": v for k, v in sres.as_dict().items()})
    row["season"] = season_scatter_to_dict(ss) if ss is not None else {}
    return row


def stage_screen(conf: dict, out_root: Path, shard: tuple[int, int], *, seed: int = 20260921,
                 max_stars: int | None = None, time_budget_s: float | None = None) -> dict:
    """Screen every fetched light curve in the shard, under a wall clock.

    The cessation statistic is a censored injection-efficiency measurement in
    every block of a century-long light curve; a single bright RR Lyrae costs
    of order a minute.  Without a clock a shard that drew more periodic
    variables than expected runs past the job's ``timeout-minutes`` and the
    runner is killed with its screen.jsonl uploaded by nobody.  The budget is
    checked BEFORE each star (screen_star is not interruptible), screen.jsonl
    is appended per star, and ``truncated`` says the shard is incomplete --- so
    a re-dispatch resumes from the checkpoint instead of starting over.  One
    star is always screened, whatever the budget: a clock that can stop a run
    before its first star is a shard that never finishes.
    """
    sd = _shard_dir(out_root, shard)
    sd.mkdir(parents=True, exist_ok=True)
    lcs, meta = load_shard_lcs(sd)
    tpath = out_root / "targets.csv"
    targets = pd.read_csv(tpath) if tpath.exists() else pd.DataFrame()
    tmap = {str(int(r["target_id"])): r.to_dict() for _, r in targets.iterrows()} \
        if len(targets) else {}
    out_path = sd / "screen.jsonl"
    done = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            try:
                done.add(str(json.loads(ln)["target_id"]))
            except Exception:                             # noqa: BLE001
                pass
    rng = np.random.default_rng(seed + shard[0])
    n = n_cess = n_fade = n_rust = 0
    t0 = _time.monotonic()
    budget = conf.get("screen", {}).get("time_budget_s") if time_budget_s is None \
        else time_budget_s
    budget = float(budget) if budget else 0.0
    truncated = False
    n_remaining = 0
    for tid, lc in lcs.items():
        if tid in done:
            continue
        if max_stars and n >= int(max_stars):
            break
        # ``n > 0``: one star always runs.  A budget smaller than a single
        # star's cost must not make every re-dispatch screen nothing --- that
        # is a shard that can never finish, however many runs it is given.
        if budget > 0 and n > 0 and _time.monotonic() - t0 > budget:
            truncated = True
            n_remaining = sum(1 for k in lcs if k not in done) - n
            print(f"[century/screen] shard {shard}: wall-clock budget {budget:.0f}s spent "
                  f"after {n} stars; {n_remaining} not screened (resumable).")
            break
        tgt = tmap.get(tid, {"target_id": int(tid), "name": tid, "kind": "unknown"})
        info = meta.get("stars", {}).get(tid, {})
        tgt = dict(tgt)
        if info.get("pm_total_masyr") is not None and not np.isfinite(
                float(tgt.get("pm_total_masyr", np.nan) or np.nan)):
            tgt["pm_total_masyr"] = info.get("pm_total_masyr")
        ts = _time.monotonic()
        row = screen_star(lc, tgt, conf, rng=rng)
        row["pm_total_masyr"] = tgt.get("pm_total_masyr")
        row["colour"] = tgt.get("colour", info.get("colour"))
        row["screen_elapsed_s"] = round(_time.monotonic() - ts, 2)
        _append(out_path, row)
        n += 1
        n_cess += int(row.get("cess_status") == "cessation")
        n_fade += int(bool(row.get("fade_is_fade")))
        n_rust += int(bool(row.get("rust_is_rust")))
    rep = {"stage": "screen", "shard": list(shard), "n_lightcurves": len(lcs),
           "n_screened_now": n, "n_previously_done": len(done), "n_cessation": n_cess,
           "n_fade": n_fade, "n_rust": n_rust, "elapsed_s": round(_time.monotonic() - t0, 1),
           "truncated": bool(truncated), "n_not_screened": int(n_remaining),
           "time_budget_s": budget}
    _write_json(sd / "screen_summary.json", rep)
    print(f"[century/screen] shard {shard}: screened={n} cess={n_cess} fade={n_fade} "
          f"rust={n_rust} in {rep['elapsed_s']}s")
    return rep


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------


def _load_rows(out_root: Path) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(str(out_root / "shards" / "*" / "screen.jsonl"))):
        for ln in Path(p).read_text().splitlines():
            try:
                rows.append(json.loads(ln))
            except Exception:                             # noqa: BLE001
                continue
    # Dedupe on target_id (a re-run shard could append twice).
    seen = {}
    for r in rows:
        seen[str(r.get("target_id"))] = r
    return list(seen.values())


def _in_field_ensemble(r: dict) -> bool:
    """Whether a star may describe its field's PLATES.  All-sky pulsators may
    not: they share no field, and ``allsky_pulsators`` is a label, not a patch
    of sky --- a "common mode" over it would be the whole sky's median, applied
    to every field.  (They are also large-amplitude variables, which is the
    last thing an ensemble of the plates' own history should contain.)"""
    return str(r.get("kind")) != "pulsator" and str(r.get("field")) != PULSATOR_FIELD


def field_common_mode(rows: list[dict], *, min_stars: int = 8, mag_bin: float = 1.0) -> dict:
    """Per field, per magnitude bin, per year: the median offset of every star's
    annual median from its own median.  The plates' shared history."""
    acc: dict = {}
    for r in rows:
        if not _in_field_ensemble(r):
            continue
        ann = r.get("annual") or {}
        if not ann.get("year"):
            continue
        fld = str(r.get("field"))
        mb = int(np.floor(float(r.get("median_mag", np.nan)) / mag_bin)) \
            if np.isfinite(float(r.get("median_mag", np.nan) or np.nan)) else None
        if mb is None:
            continue
        star_med = float(np.median(ann["med"]))
        for y, m in zip(ann["year"], ann["med"], strict=False):
            acc.setdefault(fld, {}).setdefault(mb, {}).setdefault(int(y), []).append(
                float(m) - star_med)
    out: dict = {}
    for fld, bins in acc.items():
        for mb, years in bins.items():
            cm = {y: float(np.median(v)) for y, v in years.items() if len(v) >= min_stars}
            if cm:
                out.setdefault(fld, {})[mb] = cm
    return out


def apply_common_mode(ann: dict, cm: dict | None) -> pd.DataFrame:
    tab = pd.DataFrame(ann)
    if cm and len(tab):
        corr = np.array([cm.get(int(y), np.nan) for y in tab["year"]])
        ok = np.isfinite(corr)
        tab = tab[ok].copy()
        tab["med"] = tab["med"].to_numpy(dtype=float) - corr[ok]
    return tab


def _gaia_context(rows: list[dict], conf: dict, log: AcquisitionLog) -> dict:
    """Gaia DR3 pm / colour / RUWE for a shortlist, via VizieR (runner-only)."""
    out = {}
    if not rows:
        return out
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier
        v = Vizier(columns=["RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "Gmag", "BP-RP", "RUWE"],
                   row_limit=20)
        rad = float(conf["vet"]["gaia_radius_arcsec"])
        n_ok = 0
        for r in rows:
            try:
                res = v.query_region(SkyCoord(float(r["ra"]) * u.deg, float(r["dec"]) * u.deg),
                                     radius=rad * u.arcsec, catalog="I/355/gaiadr3")
            except Exception as exc:                      # noqa: BLE001
                out[str(r["target_id"])] = {"error": repr(exc)[:200]}
                continue
            if res is None or not len(res) or not len(res[0]):
                out[str(r["target_id"])] = {"n": 0}
                continue
            t = res[0].to_pandas()
            i = int(np.argmin(np.abs(t["Gmag"].to_numpy(dtype=float)
                                     - float(r.get("mag_cat") or r.get("median_mag") or 12.0))))
            g = t.iloc[i]
            out[str(r["target_id"])] = {
                "n": int(len(t)),
                "pm_total_masyr": float(np.hypot(g.get("pmRA", np.nan), g.get("pmDE", np.nan))),
                "bp_rp": float(g.get("BP-RP", np.nan)), "ruwe": float(g.get("RUWE", np.nan)),
                "gmag": float(g.get("Gmag", np.nan))}
            n_ok += 1
        log.record("gaia_context", f"VizieR I/355/gaiadr3 {rad}\" for {len(rows)} candidates",
                   rows=n_ok)
    except Exception as exc:                              # noqa: BLE001
        log.record("gaia_context", "VizieR I/355/gaiadr3", error=repr(exc))
    return out


def stage_assess(conf: dict, out_root: Path, *, confirm: bool = True, gaia: bool = True,
                 seed: int = 20260921) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    log = AcquisitionLog()
    gap = _gap(conf)
    rows = _load_rows(out_root)
    degraded: list[str] = []

    # Acquisition roll-up across shards.
    acq_reps = [json.loads(Path(p).read_text())
                for p in sorted(glob.glob(str(out_root / "shards" / "*" / "acquire_summary.json")))]
    n_fetched = sum(int(r.get("n_fetched", 0)) + int(r.get("n_resumed", 0)) for r in acq_reps)
    n_failed = sum(int(r.get("n_failed", 0)) for r in acq_reps)
    n_empty = sum(int(r.get("n_empty", 0)) for r in acq_reps)
    n_attempted = sum(int(r.get("n_targets", 0)) for r in acq_reps)
    any_service_ok = any(bool(r.get("service_probe_ok")) for r in acq_reps)
    if any(bool(r.get("truncated")) for r in acq_reps):
        degraded.append("some_shards_truncated")
    flag_sources = sorted({str(r.get("flag_source", "none")) for r in acq_reps})
    if not acq_reps or flag_sources == ["none"]:
        degraded.append("no_flag_definitions_applied")
    tsum_p = out_root / "targets_summary.json"
    tsum = json.loads(tsum_p.read_text()) if tsum_p.exists() else {}
    n_targets = int(tsum.get("n_targets", 0))

    # Field common mode on the annual tables -> corrected fade.
    ens = conf["ensemble"]
    cm = field_common_mode(rows, min_stars=int(ens["min_stars"]), mag_bin=float(ens["mag_bin"]))
    n_cm = sum(len(b) for f in cm.values() for b in f.values())
    if not n_cm:
        degraded.append("field_ensemble_not_applied_too_few_stars")
    field_steps: dict = {}
    for fld, bins in cm.items():
        for mb, yrs in bins.items():
            ys = np.array(sorted(yrs))
            fit = fit_step_slope(ys + 0.5, np.array([yrs[y] for y in ys]), None, gap=gap)
            field_steps[f"{fld}|{mb}"] = {"step": fit.step if fit.ok else float("nan"),
                                          "slope_per_century": (fit.slope * 100.0 if fit.ok
                                                                else float("nan")),
                                          "n_years": int(fit.n)}
    fc = conf["fade"]
    for r in rows:
        ann = r.get("annual") or {}
        mm = float(r.get("median_mag", np.nan) or np.nan)
        mb = int(np.floor(mm / float(ens["mag_bin"]))) if np.isfinite(mm) else None
        cmap = cm.get(str(r.get("field")), {}).get(mb) if mb is not None else None
        r["ensemble_applied"] = bool(cmap)
        if ann.get("year") and cmap:
            tab = apply_common_mode(ann, cmap)
            fit = fade_from_annual(tab, min_years=fc["min_years"], min_span_yr=fc["min_span_yr"],
                                   gap=gap)
            r["fade_corr_slope_mag_per_century"] = fit.slope * 100.0 if fit.ok else float("nan")
            r["fade_corr_slope_sigma"] = fit.slope_sigma if fit.ok else float("nan")
            r["fade_corr_slope_pre_sigma"] = fit.slope_pre_sigma if fit.ok else float("nan")
            r["fade_corr_step_mag"] = fit.step if fit.ok else float("nan")
            r["fade_corr_is_fade"] = bool(
                fit.ok and fit.slope > 0 and fit.slope_sigma >= fc["fade_sigma_min"]
                and np.isfinite(fit.slope_pre_sigma) and fit.slope_pre_sigma >= fc["pre_sigma_min"]
                and fit.slope_pre > 0 and bool(r.get("fade_is_fade")))
            fs = field_steps.get(f"{r.get('field')}|{mb}", {})
            if r.get("cess_mean_shift_across_gap") and np.isfinite(float(fs.get("step", np.nan))):
                r["cess_mean_shift_corrected"] = (float(r.get("cess_mean_shift_mag", np.nan))
                                                  - float(fs["step"]))
        else:
            r["fade_corr_is_fade"] = bool(r.get("fade_is_fade")) and False
            if r.get("fade_is_fade"):
                r.setdefault("assess_notes", "")
                r["assess_notes"] += "fade_uncorrected_no_ensemble;"
    # Season-scatter ensemble (rust.trend), field as the shared "CCD".
    from ..rust.trend import ensemble_detrend_scatter
    srows = []
    for r in rows:
        ss = season_scatter_from_dict(r.get("season") or {})
        if ss is not None and _in_field_ensemble(r):
            srows.append({"_ss": ss, "_ccd": str(r.get("field")), "_row": r})
    ens_diag = ensemble_detrend_scatter(srows, min_stars=int(ens["min_stars"]), min_seasons=4) \
        if srows else {"ensemble_verdict": "NOT_APPLIED_NO_ROWS"}
    sc = conf["scatter"]
    for sr in srows:
        r = sr["_row"]
        fit, st_pre, _ = scatter_from_table(sr["_ss"], gap=gap)
        r["rust_corr_slope_sigma"] = fit.slope_sigma if fit.ok else float("nan")
        r["rust_corr_slope_positive"] = bool(fit.ok and fit.slope > 0)
        r["rust_corr_pre_rank_p"] = st_pre.rank_p if st_pre is not None else float("nan")
        r["rust_corr_pre_slope_sigma"] = (st_pre.slope_sigma if st_pre is not None
                                          and st_pre.slope_var_yr > 0 else float("nan"))
        r["rust_corr_is_rust"] = bool(
            r.get("rust_is_rust") and fit.ok and fit.slope > 0
            and fit.slope_sigma >= sc["slope_sigma_min"] and st_pre is not None
            and st_pre.slope_var_yr > 0 and st_pre.rank_p <= sc["pre_rank_p_max"])
    if ens_diag.get("ensemble_verdict") != "APPLIED":
        degraded.append("scatter_ensemble_not_applied")

    # Confirmation pass at full injection depth for cessation candidates.
    cands = [r for r in rows if r.get("cess_status") == "cessation"]
    if confirm and cands:
        tpath = out_root / "targets.csv"
        targets = pd.read_csv(tpath) if tpath.exists() else pd.DataFrame()
        tmap = {str(int(t["target_id"])): t.to_dict() for _, t in targets.iterrows()} \
            if len(targets) else {}
        by_shard: dict[str, dict[str, CenturyLC]] = {}
        rng = np.random.default_rng(seed)
        for r in cands:
            tid = str(r["target_id"])
            lc = None
            for sdp in sorted(glob.glob(str(out_root / "shards" / "*"))):
                if sdp not in by_shard:
                    by_shard[sdp], _ = load_shard_lcs(Path(sdp))
                if tid in by_shard[sdp]:
                    lc = by_shard[sdp][tid]
                    break
            if lc is None:
                r["confirm_status"] = "lightcurve_unavailable"
                continue
            tgt = dict(tmap.get(tid, {"target_id": int(tid), "name": r.get("name")}))
            c = screen_star(lc, tgt, conf, rng=rng, confirm=True)
            r["confirm_status"] = c.get("cess_status")
            r["confirm_p_persist_upper"] = c.get("cess_p_persist_upper")
            r["confirm_eta_min_post"] = c.get("cess_eta_min_post")
            r["confirm_flags"] = c.get("cess_flags")
            for k in ("same_series_status", "same_series_name"):
                if k in c:
                    r[k] = c[k]
            if c.get("cess_status") != "cessation":
                r["cess_status_screen"] = "cessation"
                r["cess_status"] = f"confirm_failed:{c.get('cess_status')}"

    # Gaia context for anything still a candidate, then the gauntlet.
    any_cand = [r for r in rows if (r.get("cess_status") == "cessation"
                                    or (_in_field_ensemble(r)
                                        and (r.get("fade_corr_is_fade") or r.get("fade_is_fade")
                                             or r.get("rust_corr_is_rust")
                                             or r.get("rust_is_rust"))))]
    gctx = _gaia_context(any_cand, conf, log) if (gaia and any_cand) else {}
    if any_cand and not gctx:
        degraded.append("gaia_context_not_reached")
    vc = conf["vet"]
    for r in rows:
        g = gctx.get(str(r.get("target_id")), {})
        if g.get("pm_total_masyr") is not None:
            r["pm_total_masyr"] = g["pm_total_masyr"]
            r["bp_rp"], r["ruwe"], r["gaia_gmag"] = g.get("bp_rp"), g.get("ruwe"), g.get("gmag")
        vr = dict(r)
        # The corrected verdicts supersede the raw ones where the ensemble ran.
        vr["fade_is_fade"] = bool(r.get("fade_corr_is_fade")) if r.get("ensemble_applied") \
            else bool(r.get("fade_is_fade"))
        vr["rust_is_rust"] = bool(r.get("rust_corr_is_rust")) if srows else bool(r.get("rust_is_rust"))
        if not _in_field_ensemble(r):
            # The fade and rising-scatter questions are asked of the BRIGHT
            # field sample, where the field ensemble can take the plates' own
            # history out.  A pulsator's raw fade / scatter statistics stay in
            # the screen table, but they are not candidacies: without an
            # ensemble a pulsator "fade" is the Hippke/Lund failure mode.
            vr["fade_is_fade"] = False
            vr["rust_is_rust"] = False
            r["fade_scatter_scope"] = "not_scored_pulsator"
        r.update(vet_row(vr, vc))

    # Funnel.
    def cnt(pred) -> int:
        return int(sum(1 for r in rows if pred(r)))

    funnel = {
        "n_targets": n_targets, "n_fetch_attempted": n_attempted, "n_lc_fetched": n_fetched,
        "n_lc_failed": n_failed, "n_lc_empty": n_empty, "n_screened": len(rows),
        "n_usable": cnt(lambda r: r.get("usable")),
        "n_with_period": cnt(lambda r: r.get("usable") and r.get("cess_status")
                             not in ("no_catalogue_period", "not_run", None)),
        "n_cess_blocks_ok": cnt(lambda r: int(r.get("cess_n_blocks", 0) or 0) > 0),
        "n_cess_period_recovered": cnt(lambda r: int(r.get("cess_n_detected_blocks", 0) or 0) > 0),
        "n_cess_still_periodic": cnt(lambda r: r.get("cess_status") == "still_periodic"),
        "n_cess_transition": cnt(lambda r: r.get("cess_status") in
                                 ("cessation", "low_efficiency", "variance_conserved",
                                  "faded_or_brightened", "vanished_not_ceased", "rejected")
                                 or str(r.get("cess_status", "")).startswith("confirm_failed")),
        "n_cess_low_efficiency": cnt(lambda r: r.get("cess_status") == "low_efficiency"),
        "n_cess_vanished": cnt(lambda r: r.get("cess_status") == "vanished_not_ceased"),
        "n_cess_candidates_screen": cnt(lambda r: r.get("cess_status") == "cessation"
                                        or r.get("cess_status_screen") == "cessation"),
        "n_cess_candidates_confirmed": cnt(lambda r: r.get("cess_status") == "cessation"),
        "n_fade_testable": cnt(lambda r: r.get("fade_status") not in ("insufficient_data", None)),
        # Whether a star's fade was judged against the FIELD's own plate history
        # or only against its own.  Without the ensemble the gauntlet falls back
        # to the raw fade, so this is the number that says which of
        # n_fade_candidates_raw / _corrected the verdict actually rests on.
        "n_ensemble_applied": cnt(lambda r: bool(r.get("ensemble_applied"))),
        "n_fade_candidates_raw": cnt(lambda r: bool(r.get("fade_is_fade"))),
        "n_fade_candidates_corrected": cnt(lambda r: bool(r.get("fade_corr_is_fade"))),
        "n_fade_naive_trend_was_gap": cnt(lambda r: "naive_trend_was_the_gap"
                                          in str(r.get("fade_flags", ""))),
        "n_rust_testable": cnt(lambda r: r.get("rust_status") not in ("insufficient_data", None)),
        "n_rust_candidates_raw": cnt(lambda r: bool(r.get("rust_is_rust"))),
        "n_rust_candidates_corrected": cnt(lambda r: bool(r.get("rust_corr_is_rust"))),
        "n_candidates_any": cnt(lambda r: r.get("verdict") not in ("not_candidate", None)),
        "n_survivors": cnt(lambda r: r.get("verdict") == "survivor"),
        # The cessation population, by class: how many pulsators reached a
        # usable light curve, how many were still periodic across the century,
        # and how many cessation candidates each class produced.
        "pulsators": {
            cls: {"n_screened": cnt(lambda r, c=cls: r.get("pulsator_class") == c),
                  "n_usable": cnt(lambda r, c=cls: r.get("pulsator_class") == c
                                  and bool(r.get("usable"))),
                  "n_still_periodic": cnt(lambda r, c=cls: r.get("pulsator_class") == c
                                          and r.get("cess_status") == "still_periodic"),
                  "n_cess_candidates_confirmed": cnt(
                      lambda r, c=cls: r.get("pulsator_class") == c
                      and r.get("cess_status") == "cessation")}
            for cls in sorted({str(r.get("pulsator_class")) for r in rows
                               if r.get("kind") == "pulsator" and r.get("pulsator_class")})},
    }
    kills: dict[str, int] = {}
    for r in rows:
        v = str(r.get("verdict", ""))
        if v.startswith("killed:"):
            kills[v[7:]] = kills.get(v[7:], 0) + 1

    # A reduce with NO shard directory at all has not failed to reach the
    # archive --- nobody asked it to.  It happens when the sweep was cancelled
    # or never ran, or when a reduce-only re-run is pointed at a run id whose
    # artifacts are gone.  Saying NO_LIGHTCURVES_RETURNED there would read as
    # "DASCH returned nothing for 384 targets", which is a statement about the
    # archive that no request was ever made to support.  Observed 2026-09-22:
    # run 35748748365 was cancelled between its targets and sweep jobs, and
    # its `assess` job --- guarded by `if: always()` --- was still scheduled.
    if not acq_reps:
        verdict = "NO_SHARDS_PRESENT"
    elif n_attempted == 0 and n_targets == 0:
        verdict = "NO_TARGETS"
    elif n_fetched == 0 and not any_service_ok:
        verdict = "NO_DATA_REACHED"
    elif n_fetched == 0:
        verdict = "NO_LIGHTCURVES_RETURNED"
    elif funnel["n_usable"] == 0:
        verdict = "NO_TESTABLE_LIGHT_CURVES"
    elif funnel["n_candidates_any"] == 0:
        verdict = "NO_CANDIDATES"
    elif funnel["n_survivors"] == 0:
        verdict = "CANDIDATES_ALL_TRACED"
    else:
        verdict = "SURVIVORS_FOR_FOLLOWUP"
    if degraded and verdict not in ("NO_DATA_REACHED", "NO_TARGETS", "NO_SHARDS_PRESENT"):
        verdict_full = f"{verdict} — DEGRADED ({', '.join(degraded)})"
    else:
        verdict_full = verdict

    # Coverage.
    fields = {}
    for r in rows:
        f = str(r.get("field"))
        d = fields.setdefault(f, {"n_screened": 0, "n_usable": 0, "n_variables": 0,
                                  "median_n_det": [], "median_span_yr": []})
        d["n_screened"] += 1
        d["n_usable"] += int(bool(r.get("usable")))
        d["n_variables"] += int(r.get("kind") in ("variable", "pulsator"))
        d["median_n_det"].append(float(r.get("n_good", 0) or 0))
        d["median_span_yr"].append(float(r.get("span_yr", 0) or 0))
    for d in fields.values():
        d["median_n_det"] = float(np.median(d["median_n_det"])) if d["median_n_det"] else 0.0
        d["median_span_yr"] = float(np.median(d["median_span_yr"])) if d["median_span_yr"] else 0.0

    summary = {
        "channel": "century", "signature": "S50",
        "verdict": verdict_full, "verdict_code": verdict, "degraded": degraded,
        "funnel": funnel, "kills": kills, "fields": fields,
        "field_density": tsum.get("fields", {}),
        "ensemble": {"annual_common_mode_cells": n_cm, "field_steps": field_steps,
                     "scatter": ens_diag},
        "flag_sources": flag_sources, "menzel_gap": {"start": gap[0], "end": gap[1]},
        "acquisition": log.as_dict(),
        "generated_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
    }
    _write_json(out_root / "summary.json", summary)

    # Tables.
    flat = []
    for r in rows:
        flat.append({k: v for k, v in r.items() if k not in ("annual", "season")})
    df = pd.DataFrame(flat)
    if len(df):
        with gzip.open(out_root / "century_screen.csv.gz", "wt") as fh:
            df.to_csv(fh, index=False)
        cand = df[df["verdict"].astype(str) != "not_candidate"] if "verdict" in df else df.iloc[0:0]
        cand.to_csv(out_root / "candidates.csv", index=False)
        surv = df[df["verdict"].astype(str) == "survivor"] if "verdict" in df else df.iloc[0:0]
        surv.to_csv(out_root / "survivors.csv", index=False)
    print(f"[century/assess] verdict={verdict_full}")
    print(json.dumps(jsonable(funnel), indent=1))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stage", required=True,
                   choices=["probe", "targets", "acquire", "screen", "assess", "all"])
    p.add_argument("--shard", default="0/1", help="i/n (0-based)")
    p.add_argument("--fields", default=None,
                   help="semicolon-separated ra,dec[,name]; default from config")
    p.add_argument("--radius-deg", type=float, default=None)
    p.add_argument("--max-variables", type=int, default=None, help="per field")
    p.add_argument("--max-bright", type=int, default=None, help="per field (0 = none)")
    p.add_argument("--max-pulsators", type=int, default=None,
                   help="cap on all-sky pulsators (0 = none; default pulsators.max_total)")
    p.add_argument("--max-targets", type=int, default=None, help="cap per shard (acquire)")
    p.add_argument("--max-stars", type=int, default=None, help="cap per shard (screen)")
    p.add_argument("--time-budget-s", type=float, default=None,
                   help="acquire: DASCH wall-clock budget (s)")
    p.add_argument("--screen-budget-s", type=float, default=None,
                   help="screen: wall-clock budget (s); 0 = no clock")
    p.add_argument("--pause-s", type=float, default=None)
    p.add_argument("--no-confirm", action="store_true")
    p.add_argument("--no-gaia", action="store_true")
    p.add_argument("--out-root", default=None)
    p.add_argument("--seed", type=int, default=20260921)


def _parse_fields(s: str | None):
    if not s:
        return None
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        bits = [b.strip() for b in part.split(",")]
        f = {"ra": float(bits[0]), "dec": float(bits[1])}
        if len(bits) > 2 and bits[2]:
            f["name"] = bits[2]
        out.append(f)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="century", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    args = p.parse_args(argv)
    return run_args(args)


def run_args(args) -> int:
    conf = load_century_config()
    out_root = Path(args.out_root) if args.out_root else (repo_root() / "results" / "century")
    shard = parse_shard(args.shard)
    fields = _parse_fields(args.fields)
    st = args.stage
    if st == "probe":
        stage_probe(conf, out_root)
    if st == "targets" or (st == "all" and not (out_root / "targets.csv").exists()):
        stage_targets(conf, out_root, fields=fields, radius_deg=args.radius_deg,
                      max_variables=args.max_variables, max_bright=args.max_bright,
                      include_bright=(args.max_bright is None or args.max_bright > 0),
                      max_pulsators=args.max_pulsators,
                      include_pulsators=(None if args.max_pulsators is None
                                         else args.max_pulsators > 0))
    if st in ("acquire", "all"):
        stage_acquire(conf, out_root, shard, time_budget_s=args.time_budget_s,
                      max_targets=args.max_targets, pause_s=args.pause_s)
    if st in ("screen", "all"):
        stage_screen(conf, out_root, shard, seed=args.seed, max_stars=args.max_stars,
                     time_budget_s=getattr(args, "screen_budget_s", None))
    if st == "assess":
        stage_assess(conf, out_root, confirm=not args.no_confirm, gaia=not args.no_gaia,
                     seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
