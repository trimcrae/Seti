"""Stage orchestration for CRYPT (S55).  Writes ``results/crypt/``.

Stages
------
``probe``    reach every root; crawl the Diviner level-3/4 volume, download
             the labels (small), classify every product by name + label text,
             and record the inventory and the selection the screen would
             make; list ODE instrument/product types; crawl Mini-RF; reach
             ShadowCam and the PSR routes.  Writes ``probe.json``.
``acquire``  download the selected rasters for this shard's pole(s) (label +
             image, size-capped, cached under ``results/crypt/data/``), read
             the labels, and — if a Mini-RF polar CPR product was classified
             — that too, plus the external PSR raster.  Writes
             ``acquire_<pole>.json``.
``screen``   build the layer store per pole, run the thermal screen, the
             injection-recovery sensitivity and the radar screen.  Writes
             ``screen_<pole>.json``, ``flags_<pole>.csv``,
             ``sensitivity_<pole>.json``, ``radar_<pole>.json``.
``assess``   merge the poles, vet every candidate (radar reading, ODE
             footprint coverage), write ``summary.json``, ``candidates.json``,
             ``candidates.csv``.

Verdict vocabulary (``summary.json["verdict"]``)
------------------------------------------------
``NO_DATA_REACHED``             no Diviner polar product was read; an access
                                statement with the routes tried
``NO_ANISOTHERMAL_SURVIVOR``    pixels screened, none survives — a count with
                                its measured floor, never written up
``ANISOTHERMAL_CANDIDATES (n)`` n pixels survive every rule, pending the vet
Each may carry ``DEGRADED (...)`` naming what was missing (single season, no
count layer, no radar, thermal-only PSR definition).

``--shard i/n`` splits the pole list; ``--synthetic`` runs screen + assess on
a synthetic pole (offline smoke; verdict prefixed ``SYNTHETIC_``).
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import acquire as A
from . import radar as R
from . import thermal as T
from . import vet as V
from .labels import RasterMeta, read_label, read_raster

STAGES = ("probe", "acquire", "screen", "assess")
VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_NONE = "NO_ANISOTHERMAL_SURVIVOR"
VERDICT_CANDIDATES = "ANISOTHERMAL_CANDIDATES"


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
    "acquire": {"timeout_s": 120, "listing_timeout_s": 60, "download_timeout_s": 900, "retries": 2,
                "max_bytes_per_file": 400_000_000, "max_total_bytes": 6_000_000_000,
                "max_labels_probe": 400, "label_max_bytes": 400_000, "crawl_max_depth": 5,
                "crawl_max_entries": 8000,
                "diviner_volume_roots": ["https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/"],
                "diviner_data_subdirs": ["data/"], "diviner_dir_regex": None, "diviner_pds4_roots": [],
                "minirf_roots": [], "minirf_dir_regex": None, "minirf_product_regex": "(?i)cpr",
                "shadowcam_roots": [], "ode_rest": "https://oderest.rsl.wustl.edu/live2/",
                "ode_instruments": {}, "ode_footprint_half_deg": 0.02, "ode_limit": 25,
                "psr_routes": [], "psr_product_regex": "(?i)(psr|shadow|illum)"},
    "products": {"poles": ["north", "south"], "seasons": ["summer", "winter"],
                 "channels": ["6", "7", "8", "9"], "stats": ["avg", "count"],
                 "mask_products": [{"season": "all", "channel": "tbol", "stat": "max"},
                                   {"season": "summer", "channel": "tbol", "stat": "max"}],
                 "optional": []},
    "patterns": dict(A.DEFAULT_PATTERNS),
    "thermal": dict(T.DEFAULT_THRESHOLDS),
    "sensitivity": {"areas_m2": [1, 10, 100, 1000, 10000], "t_hot_K": [300.0], "n_per_area": 20, "seed": 11},
    "radar": dict(R.DEFAULT_RADAR),
    "hardware": [],
    "vet": {"max_candidates_vet": 200},
}


def load_crypt_config(path: Path | None = None) -> dict:
    """``config/crypt.yaml`` over :data:`DEFAULTS`; a missing file degrades."""
    try:
        import yaml  # noqa: PLC0415
        path = Path(path) if path is not None else repo_root() / "config" / "crypt.yaml"
        if not path.exists():
            print(f"[crypt] config {path} not found; using defaults")
            return _deep_update(DEFAULTS, {})
        return _deep_update(DEFAULTS, yaml.safe_load(path.read_text()) or {})
    except Exception as exc:  # noqa: BLE001
        print(f"[crypt] config not loaded ({exc!r}); using defaults")
        return _deep_update(DEFAULTS, {})


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
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
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=2, default=_json_default))


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return None


def parse_shard(s: str | None) -> tuple[int, int]:
    if not s:
        return 0, 1
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", str(s))
    if not m:
        raise SystemExit(f"--shard must look like i/n, got {s!r}")
    i, n = int(m.group(1)), int(m.group(2))
    if n < 1 or not (0 <= i < n):
        raise SystemExit(f"--shard {s!r}: need 0 <= i < n")
    return i, n


def shard_poles(conf: dict, shard: str | None, poles=None) -> list[str]:
    allp = list(poles or conf["products"]["poles"])
    i, n = parse_shard(shard)
    return [p for k, p in enumerate(allp) if k % n == i]


def needed_products(conf: dict, pole: str) -> list[dict]:
    pr = conf["products"]
    need = []
    for s in pr["seasons"]:
        for ch in pr["channels"]:
            for st in pr["stats"]:
                need.append({"pole": pole, "season": s, "channel": str(ch), "stat": st, "role": "screen"})
    for m in pr.get("mask_products", []):
        need.append({"pole": pole, "role": "mask", **{k: str(v) for k, v in m.items()}})
    for m in pr.get("optional", []):
        need.append({"pole": pole, "role": "optional", **{k: str(v) for k, v in m.items()}})
    return need


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def _reach(fetch, url: str, timeout: float, max_bytes: int = 5_000_000) -> dict:
    res = fetch(url, timeout=timeout, max_bytes=max_bytes)
    d = res.as_dict()
    d["head"] = res.content[:1200].decode("utf-8", errors="replace") if res.content else None
    d["reached"] = bool(res.ok)
    return d


def _label_summary(meta: RasterMeta) -> dict:
    return {"dialect": meta.dialect, "lines": meta.lines, "samples": meta.samples, "bands": meta.bands,
            "dtype": meta.dtype, "scaling_factor": meta.scaling_factor, "offset": meta.offset,
            "missing": meta.missing, "unit": meta.unit, "product_id": meta.product_id,
            "description": meta.description[:300], "georef": meta.georef.as_dict(),
            "data_file": Path(meta.data_file).name, "byte_offset": meta.byte_offset}


def crawl_diviner(conf: dict, fetch, log: list) -> tuple[list[A.Entry], list[dict]]:
    acq = conf["acquire"]
    entries: list[A.Entry] = []
    roots_tried = []
    for root in list(acq.get("diviner_volume_roots", [])) + list(acq.get("diviner_pds4_roots", [])):
        for sub in acq.get("diviner_data_subdirs", ["data/"]):
            url = root.rstrip("/") + "/" + sub.lstrip("/")
            before = len(log)
            got = A.crawl(fetch, url, max_depth=int(acq.get("crawl_max_depth", 5)),
                          max_entries=int(acq.get("crawl_max_entries", 8000)),
                          dir_regex=acq.get("diviner_dir_regex"), timeout=float(acq.get("listing_timeout_s", 60)),
                          log=log)
            ok = any(x.get("status") == 200 for x in log[before:])
            roots_tried.append({"url": url, "reached": ok, "n_entries": len(got),
                                "n_files": sum(1 for e in got if not e.is_dir)})
            entries.extend(got)
        # Stop only once the crawl has found products that actually carry the
        # axes the screen needs.  ``lrodlr_1002/data/`` answers 200 with two
        # PDS4 collection files and nothing else — treating that as "reached"
        # is what made the first run report NO_DATA_REACHED (results/crypt at
        # 2026-09-22T01:25Z).
        if _n_useful(conf, entries) > 0:
            break
    # de-duplicate by URL
    seen, uniq = set(), []
    for e in entries:
        if e.url not in seen:
            seen.add(e.url)
            uniq.append(e)
    return uniq, roots_tried


def _n_useful(conf: dict, entries: list[A.Entry]) -> int:
    """Files whose NAME already places them on a pole and a channel — the
    minimum for the selector to have anything to choose between."""
    n = 0
    for e in entries:
        if e.is_dir:
            continue
        c = A.classify_name(e.name, conf["patterns"])
        if c.get("pole") and c.get("channel"):
            n += 1
    return n


def discover_diviner_ode(conf: dict, fetch) -> tuple[list[A.Entry], dict]:
    """Diviner gridded products through ODE's own index.

    The PDS directory layout is not knowable from the sandbox and the first
    run's guess was wrong; ODE lists (ihid=LRO, iid=DLRE) product types
    including PCP (Polar Cumulative Products) and GDR_L3, and ``results=f``
    hands back each product's absolute file URLs.  Returns listing entries so
    they join the crawled ones before classification.
    """
    acq = conf["acquire"]
    base = acq.get("ode_rest")
    spec = acq.get("diviner_ode") or {}
    rep: dict = {"queries": [], "n_files": 0, "n_products": 0}
    if not base or not spec.get("product_types"):
        rep["status"] = "NOT_CONFIGURED"
        return [], rep
    entries: list[A.Entry] = []
    for pt in spec["product_types"]:
        try:
            r = A.ode_dataset_files(fetch, base, spec.get("ihid", "LRO"), spec.get("iid", "DLRE"), pt,
                                    limit=int(spec.get("limit", 1000)),
                                    timeout=float(acq.get("listing_timeout_s", 60)))
        except Exception as exc:  # noqa: BLE001
            rep["queries"].append({"pt": pt, "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        entries.extend(A.entries_from_ode_files(r["files"]))
        rep["queries"].append({k: v for k, v in r.items() if k != "files"} |
                              {"example_files": [f["name"] for f in r["files"][:40]]})
        rep["n_products"] += int(r.get("n_products") or 0)
    rep["n_files"] = len(entries)
    rep["status"] = "OK" if entries else "NO_FILES"
    return entries, rep


def stage_probe(conf: dict, out: Path, *, fetch=None) -> dict:
    fetch = fetch or A.http_fetch
    acq = conf["acquire"]
    t_list = float(acq.get("listing_timeout_s", 60))
    rep: dict = {"stage": "probe", "generated_utc": _now(), "roots": {}, "diviner": {},
                 "minirf": {}, "shadowcam": {}, "ode": {}, "psr": {}, "crawl_log": []}
    # 1. Diviner: crawl, read labels, classify, select.  Two independent
    # routes — the HTTP directory crawl and ODE's index — are merged, so a
    # wrong volume path cannot by itself produce a no-data verdict.
    entries, roots_tried = crawl_diviner(conf, fetch, rep["crawl_log"])
    ode_entries, ode_rep = discover_diviner_ode(conf, fetch)
    have = {e.url for e in entries}
    entries = entries + [e for e in ode_entries if e.url not in have]
    files = [e for e in entries if not e.is_dir]
    groups = A.pair_products(files)
    label_texts: dict[str, str] = {}
    label_meta: dict[str, dict] = {}
    n_lab = 0
    lab_dir = out / "data" / "labels"
    # labels of the products the NAME already places on every axis first, so
    # the ones the screen will select are read before the cap bites
    def _rank(item):
        stem, g = item
        name = (g["image"] or g["label"]).name if (g["image"] or g["label"]) else stem
        c = A.classify_name(name, conf["patterns"])
        return sum(1 for ax in ("pole", "channel", "stat", "season") if c.get(ax) is None)
    for stem, g in sorted(groups.items(), key=_rank):
        if g["label"] is None or n_lab >= int(acq.get("max_labels_probe", 400)):
            continue
        lp = lab_dir / g["label"].name
        res = A.download(fetch, g["label"].url, lp, max_bytes=int(acq.get("label_max_bytes", 400_000)),
                         timeout=t_list)
        n_lab += 1
        if not res.ok:
            label_meta[stem] = {"error": res.error, "status": res.status}
            continue
        try:
            txt = lp.read_bytes()[:20000].decode("latin-1", errors="replace")
            label_texts[stem] = txt
            meta = read_label(lp)
            label_meta[stem] = _label_summary(meta)
        except Exception as exc:  # noqa: BLE001
            label_meta[stem] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    classified = {}
    for stem, g in groups.items():
        name = (g["image"] or g["label"]).name if (g["image"] or g["label"]) else stem
        classified[stem] = A.classify_name(name, conf["patterns"], extra_text=label_texts.get(stem, ""))
        classified[stem]["image_size"] = g["image"].size if g["image"] else None
        classified[stem]["has_label"] = g["label"] is not None
        classified[stem]["has_image"] = g["image"] is not None
        classified[stem]["url"] = (g["image"] or g["label"]).url if (g["image"] or g["label"]) else None
    selection = {}
    for pole in conf["products"]["poles"]:
        selection[pole] = A.select_needed(groups, needed_products(conf, pole), conf["patterns"], label_texts)
    axis_counts = {ax: {} for ax in ("pole", "channel", "stat", "season")}
    for c in classified.values():
        for ax in axis_counts:
            axis_counts[ax][str(c.get(ax))] = axis_counts[ax].get(str(c.get(ax)), 0) + 1
    rep["diviner"] = {
        "roots_tried": roots_tried, "ode": ode_rep, "n_entries": len(entries), "n_files": len(files),
        "n_products": len(groups), "n_labels_read": n_lab,
        "directories": sorted({e.url for e in entries if e.is_dir})[:400],
        "inventory": [{"stem": s, **{k: v for k, v in c.items() if k != "name"}} for s, c in
                      list(classified.items())[:3000]],
        "labels": label_meta,
        "axis_counts": axis_counts,
        "selection": selection,
        "n_selected": {p: sum(1 for v in sel.values() if v) for p, sel in selection.items()},
        "n_needed": {p: len(sel) for p, sel in selection.items()},
        "example_names": [e.name for e in files[:60]],
        "n_useful": _n_useful(conf, entries),
        "useful_example_names": [s for s, c in classified.items()
                                 if c.get("pole") and c.get("channel")][:80],
        "unplaced_example_names": [s for s, c in classified.items()
                                   if not (c.get("pole") and c.get("channel"))][:80],
    }
    # 2. Mini-RF
    mr_entries: list[A.Entry] = []
    mr_roots = []
    for root in acq.get("minirf_roots", []):
        before = len(rep["crawl_log"])
        got = A.crawl(fetch, root, max_depth=4, max_entries=4000, dir_regex=acq.get("minirf_dir_regex"),
                      timeout=t_list, log=rep["crawl_log"])
        mr_roots.append({"url": root, "reached": any(x.get("status") == 200 for x in rep["crawl_log"][before:]),
                         "n_entries": len(got)})
        mr_entries.extend(got)
    prx = re.compile(str(acq.get("minirf_product_regex", "(?i)cpr")))
    mr_files = [e for e in mr_entries if not e.is_dir]
    mr_hits = [e for e in mr_files if prx.search(e.name)]
    rep["minirf"] = {"roots": mr_roots, "n_files": len(mr_files), "n_matching": len(mr_hits),
                     "directories": sorted({e.url for e in mr_entries if e.is_dir})[:300],
                     "matching": [e.as_dict() for e in mr_hits[:300]],
                     # every file in a mosaic directory, unfiltered: the product
                     # codes that actually exist are read off this, not guessed
                     "mosaic_names": sorted({e.name for e in mr_files
                                             if "mosaic" in e.url.lower()})[:400],
                     "example_names": [e.name for e in mr_files[:60]]}
    # 3. ShadowCam roots
    rep["shadowcam"] = {u: _reach(fetch, u, t_list) for u in acq.get("shadowcam_roots", [])}
    # 4. ODE: instrument / product-type listing and one footprint query per instrument
    base = acq.get("ode_rest")
    if base:
        iipy = A.ode_query(fetch, base, {"target": "moon", "query": "iipy"}, timeout=t_list)
        j = iipy.get("json")
        rep["ode"]["iipy"] = {"status": iipy.get("status"), "error": iipy.get("error"),
                              "head": iipy.get("head"), "url": iipy.get("url"),
                              "json_head": json.dumps(j)[:6000] if j is not None else None}
        # every instrument host / instrument / product type ODE lists for the Moon
        types = []
        if isinstance(j, dict):
            for e in _walk_dicts(j):
                if "IID" in e or "iid" in e or "InstrumentId" in e:
                    types.append({k: v for k, v in e.items() if isinstance(v, (str, int, float))})
        rep["ode"]["instrument_types"] = types[:400]
        probe_pos = {"north": (0.0, 89.5), "south": (0.0, -89.5)}
        rep["ode"]["footprint_probe"] = {}
        for pole, (lon, lat) in probe_pos.items():
            rep["ode"]["footprint_probe"][pole] = V.ode_coverage(conf, fetch, lon, lat)
    # 5. PSR routes: the HTTP pages, plus ODE's index of the LOLA GDR
    # permanently-shadowed map (LRO/LOLA/GDRPSR) which names the rasters.
    rep["psr"] = {u: _reach(fetch, u, t_list) for u in acq.get("psr_routes", [])}
    pspec = acq.get("psr_ode") or {}
    if base and pspec.get("product_types"):
        rep["psr_ode"] = []
        for pt in pspec["product_types"]:
            try:
                r = A.ode_dataset_files(fetch, base, pspec.get("ihid", "LRO"), pspec.get("iid", "LOLA"),
                                        pt, limit=int(pspec.get("limit", 500)), timeout=t_list)
            except Exception as exc:  # noqa: BLE001
                rep["psr_ode"].append({"pt": pt, "error": f"{type(exc).__name__}: {exc}"[:300]})
                continue
            rep["psr_ode"].append({k: v for k, v in r.items() if k != "files"} |
                                  {"files": r["files"][:60]})
    rep["crawl_log"] = rep["crawl_log"][:600]
    reached = {"diviner": rep["diviner"]["n_files"] > 0,
               "minirf": rep["minirf"]["n_files"] > 0,
               "shadowcam": any(v.get("reached") for v in rep["shadowcam"].values()),
               "ode": rep["ode"].get("iipy", {}).get("status") == 200,
               "psr": any(v.get("reached") for v in rep["psr"].values())}
    rep["reached"] = reached
    rep["verdict"] = ("DIVINER_PRODUCTS_LISTED" if reached["diviner"] else "DIVINER_NOT_LISTED")
    _write(out / "probe.json", rep)
    print(f"[crypt] probe: {rep['verdict']}; diviner files={rep['diviner']['n_files']} "
          f"labels={n_lab} selected={rep['diviner']['n_selected']} of {rep['diviner']['n_needed']}; "
          f"minirf files={rep['minirf']['n_files']} matching={rep['minirf']['n_matching']}; "
          f"reached={reached}")
    return rep


def _walk_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_dicts(v)


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------
def stage_acquire(conf: dict, out: Path, *, fetch=None, shard: str | None = None, poles=None) -> dict:
    fetch = fetch or A.http_fetch
    acq = conf["acquire"]
    probe = _read_json(out / "probe.json")
    if not probe or not probe.get("diviner", {}).get("selection"):
        print("[crypt] acquire: no probe.json selection on disk; probing first")
        probe = stage_probe(conf, out, fetch=fetch)
    sel_all = probe["diviner"]["selection"]
    total = 0
    reports = {}
    for pole in shard_poles(conf, shard, poles):
        sel = sel_all.get(pole, {})
        ddir = out / "data" / pole
        rep = {"stage": "acquire", "pole": pole, "generated_utc": _now(), "products": {}, "n_ok": 0,
               "n_missing": 0, "bytes": 0}
        for key, spec in sel.items():
            if spec is None:
                rep["products"][key] = {"status": "NOT_IN_INVENTORY"}
                rep["n_missing"] += 1
                continue
            img, lbl = spec["image"], spec.get("label")
            prec = {"status": "PENDING", "stem": spec["stem"], "class": spec["class"],
                    "image_url": img["url"], "image_size": img.get("size"),
                    "label_url": lbl["url"] if lbl else None}
            if total + (img.get("size") or 0) > int(acq.get("max_total_bytes", 6e9)):
                prec["status"] = "SKIPPED_TOTAL_CAP"
                rep["products"][key] = prec
                rep["n_missing"] += 1
                continue
            ip = ddir / img["name"]
            r2 = A.download(fetch, img["url"], ip, max_bytes=int(acq.get("max_bytes_per_file", 4e8)),
                            timeout=float(acq.get("download_timeout_s", 900)), expected_size=img.get("size"))
            prec["image_fetch"] = {k: v for k, v in r2.as_dict().items() if k != "url"}
            lp = None
            if lbl:
                lp = ddir / lbl["name"]
                r1 = A.download(fetch, lbl["url"], lp, max_bytes=int(acq.get("label_max_bytes", 4e5)),
                                timeout=float(acq.get("listing_timeout_s", 60)))
                prec["label_fetch"] = {k: v for k, v in r1.as_dict().items() if k != "url"}
                if not r1.ok:
                    lp = None
            if not r2.ok:
                prec["status"] = "IMAGE_NOT_REACHED"
                rep["n_missing"] += 1
                rep["products"][key] = prec
                continue
            try:
                meta = read_label(lp if lp is not None else ip)
                if lp is None:
                    meta.data_file = str(ip)
                prec["label"] = _label_summary(meta)
                prec["label_path"] = str(lp if lp is not None else ip)
                prec["image_path"] = str(ip)
                prec["status"] = "OK"
                rep["n_ok"] += 1
                total += r2.n_bytes
                rep["bytes"] += r2.n_bytes
            except Exception as exc:  # noqa: BLE001
                prec["status"] = "LABEL_UNREADABLE"
                prec["error"] = f"{type(exc).__name__}: {exc}"[:300]
                rep["n_missing"] += 1
            rep["products"][key] = prec
        # Mini-RF polar CPR (and S1) if the probe classified any
        rep["minirf"] = _acquire_minirf(conf, fetch, probe, pole, ddir)
        rep["status"] = "OK" if rep["n_ok"] > 0 else VERDICT_NO_DATA
        _write(out / f"acquire_{pole}.json", rep)
        print(f"[crypt] acquire {pole}: {rep['status']} ok={rep['n_ok']} missing={rep['n_missing']} "
              f"bytes={rep['bytes']} minirf={rep['minirf'].get('status')}")
        reports[pole] = rep
    return reports


def _acquire_minirf(conf: dict, fetch, probe: dict, pole: str, ddir: Path) -> dict:
    acq = conf["acquire"]
    hits = probe.get("minirf", {}).get("matching", [])
    if not hits:
        return {"status": "NO_MINIRF_PRODUCT_LISTED"}
    pole_rx = re.compile(conf["patterns"]["pole"][pole])
    groups = A.pair_products([A.Entry(**h) for h in hits])
    rec = {"status": "NO_POLAR_CPR_MATCH", "tried": []}
    for stem, g in groups.items():
        if g["image"] is None or not pole_rx.search(g["image"].name):
            continue
        low = g["image"].name.lower()
        # Mini-RF PDS3 mosaic names carry the level and the product code in one
        # token: lsz_xxxxx_3s1_pfu_90n000_v1 → level 3, Stokes 1.  The CPR code
        # is "cp".  Both therefore sit against a digit, not a separator.
        kind = None
        if "cpr" in low or re.search(r"(^|[_\-.]|\d)cp([_\-.]|$)", low):
            kind = "cpr"
        elif re.search(r"(^|[_\-.]|\d)s1([_\-.]|$)", low):
            kind = "s1"
        if kind is None:
            continue
        if g["image"].size and g["image"].size > int(acq.get("max_bytes_per_file", 4e8)):
            rec["tried"].append({"stem": stem, "outcome": "too_large", "size": g["image"].size})
            continue
        ip = ddir / "minirf" / g["image"].name
        r2 = A.download(fetch, g["image"].url, ip, max_bytes=int(acq.get("max_bytes_per_file", 4e8)),
                        timeout=float(acq.get("download_timeout_s", 900)), expected_size=g["image"].size)
        lp = None
        if g["label"] is not None:
            lp = ddir / "minirf" / g["label"].name
            r1 = A.download(fetch, g["label"].url, lp, max_bytes=int(acq.get("label_max_bytes", 4e5)))
            if not r1.ok:
                lp = None
        entry = {"stem": stem, "kind": kind, "outcome": "OK" if r2.ok else f"image: {r2.error}"}
        if r2.ok:
            try:
                meta = read_label(lp if lp is not None else ip)
                if lp is None:
                    meta.data_file = str(ip)
                entry["label"] = _label_summary(meta)
                entry["label_path"] = str(lp if lp is not None else ip)
                rec[kind] = entry
                rec["status"] = "OK"
            except Exception as exc:  # noqa: BLE001
                entry["outcome"] = f"label: {type(exc).__name__}: {exc}"[:300]
        rec["tried"].append(entry)
    return rec


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------
def build_layers(conf: dict, pole: str, acq_rep: dict) -> tuple[T.PoleLayers | None, dict]:
    """The layer store from an acquire report; rasters on a different grid
    from the first are nearest-resampled onto it (recorded)."""
    notes: dict = {"loaded": [], "resampled": [], "failed": [], "skipped": []}
    layers = None
    ref_georef = None
    for key, prec in (acq_rep.get("products") or {}).items():
        if prec.get("status") != "OK":
            notes["skipped"].append({"key": key, "status": prec.get("status")})
            continue
        try:
            meta = read_label(prec["label_path"])
            if Path(prec["label_path"]).suffix.lower() in (".img", ".tif"):
                meta.data_file = prec["image_path"]
            img = read_raster(meta)
        except Exception as exc:  # noqa: BLE001
            notes["failed"].append({"key": key, "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        _p, season, channel, stat = key.split("/")
        if layers is None:
            ref_georef = meta.georef
            layers = T.PoleLayers(pole, ref_georef)
        if img.shape != layers.shape and layers.arrays:
            img = _resample(img, meta.georef, ref_georef)
            notes["resampled"].append(key)
        # units: temperatures must be kelvin; counts are counts
        unit = (meta.unit or "").upper()
        if stat != "count" and unit and unit not in ("K", "KELVIN", "DEGK", "DEG K", "KELVINS"):
            notes["skipped"].append({"key": key, "status": f"unit {meta.unit!r} is not kelvin"})
            continue
        layers.put(season, channel, stat, img, source=prec.get("image_url"))
        notes["loaded"].append({"key": key, "finite_frac": float(np.isfinite(img).mean()),
                                "p50": float(np.nanmedian(img)) if np.isfinite(img).any() else None,
                                "shape": list(img.shape)})
    return layers, notes


def _resample(img: np.ndarray, src, dst) -> np.ndarray:
    ii, jj = np.mgrid[0:dst.lines, 0:dst.samples]
    lon, lat = dst.pix_to_lonlat(ii, jj)
    si, sj = src.lonlat_to_pix(lon, lat)
    si = np.rint(si).astype(int)
    sj = np.rint(sj).astype(int)
    ok = (si >= 0) & (si < img.shape[0]) & (sj >= 0) & (sj < img.shape[1])
    out = np.full((dst.lines, dst.samples), np.nan, dtype=np.float32)
    out[ok] = img[si[ok], sj[ok]]
    return out


def _load_minirf(acq_rep: dict) -> dict | None:
    mr = acq_rep.get("minirf") or {}
    if mr.get("status") != "OK" or "cpr" not in mr:
        return None
    try:
        meta = read_label(mr["cpr"]["label_path"])
        cpr = read_raster(meta)
        out = {"cpr": cpr, "georef": meta.georef, "s1": None}
        if "s1" in mr:
            m1 = read_label(mr["s1"]["label_path"])
            s1 = read_raster(m1)
            if s1.shape == cpr.shape:
                out["s1"] = s1
        return out
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}


def screen_one_pole(conf: dict, out: Path, pole: str, layers: T.PoleLayers, *, fetch=None,
                    radar_layers: dict | None = None, external_psr=None, psr_ledger=None,
                    run_sensitivity: bool = True) -> dict:
    thr = conf["thermal"]
    rep = T.screen_pole(layers, thr, hardware=conf.get("hardware"), external_psr=external_psr)
    flags = rep.pop("flags")
    rep["psr_route"] = psr_ledger
    rep["layers"] = sorted(f"{s}/{c}/{t}" for (s, c, t) in layers.arrays)
    rep["georef"] = layers.georef.as_dict()
    rep["grid"] = {"shape": list(layers.shape), "pixel_area_m2": layers.georef.pixel_area_m2}
    if rep.get("mask", {}).get("n_interior"):
        rep["mask"]["area_km2_interior"] = rep["mask"]["n_interior"] * layers.georef.pixel_area_m2 / 1e6
        rep["mask"]["area_km2_mask"] = rep["mask"]["n_mask"] * layers.georef.pixel_area_m2 / 1e6
    if len(flags):
        flags = flags.sort_values("z_primary_min", ascending=False)
        flags.head(5000).to_csv(out / f"flags_{pole}.csv", index=False)
        rep["top_flags"] = _clean(flags.head(50).to_dict(orient="records"))
        cands = _clean(flags[flags["class"] == "candidate"].head(500).to_dict(orient="records"))
        # the radar reading is taken HERE, where the rasters are (the assess
        # job runs without the data directory)
        for c in cands:
            if radar_layers and radar_layers.get("cpr") is not None:
                c["radar"] = R.sample_at(radar_layers["cpr"], radar_layers["georef"], c["lon"], c["lat"])
                if radar_layers.get("s1") is not None:
                    c["radar_s1"] = R.sample_at(radar_layers["s1"], radar_layers["georef"], c["lon"], c["lat"])
            else:
                c["radar"] = {"status": "NO_RADAR_LAYER"}
        rep["candidates"] = cands
    else:
        pd.DataFrame(columns=["pole", "line", "sample", "lon", "lat", "class", "reasons"]).to_csv(
            out / f"flags_{pole}.csv", index=False)
        rep["top_flags"], rep["candidates"] = [], []
    _write(out / f"screen_{pole}.json", rep)
    print(f"[crypt] screen {pole}: {rep.get('status')} interior={rep.get('mask', {}).get('n_interior')} "
          f"flagged={rep.get('n_flagged_any_season')} counts={rep.get('counts')}")
    # sensitivity
    if run_sensitivity and rep.get("status") == "OK":
        sc = conf["sensitivity"]
        sens = {"pole": pole, "generated_utc": _now(), "by_t_hot": {}}
        for th in sc.get("t_hot_K", [300.0]):
            sens["by_t_hot"][str(th)] = T.sensitivity(layers, thr, sc["areas_m2"], float(th),
                                                       n_per=int(sc.get("n_per_area", 20)),
                                                       seed=int(sc.get("seed", 11)),
                                                       hardware=conf.get("hardware"), external_psr=external_psr)
        sens["floor_statement"] = floor_statement(rep, sens)
        _write(out / f"sensitivity_{pole}.json", sens)
        print(f"[crypt] sensitivity {pole}: {sens['floor_statement']}")
        rep["sensitivity"] = sens["floor_statement"]
    # radar
    rrep = {"pole": pole, "status": "NO_RADAR_LAYER"}
    if radar_layers and radar_layers.get("cpr") is not None:
        try:
            m = T.psr_mask(layers, {**T.DEFAULT_THRESHOLDS, **thr}, external=external_psr)
            rmask = R.resample_mask(m["mask"], layers.georef, radar_layers["georef"])
            rr = R.screen_radar(radar_layers["cpr"], rmask, radar_layers["georef"], conf["radar"],
                                s1=radar_layers.get("s1"))
            rf = rr.pop("flags")
            rr["top"] = _clean(rf.sort_values("cpr", ascending=False).head(100).to_dict(orient="records")) if len(rf) else []
            rr["radar_candidates"] = _clean(rf[rf["class"] == "radar_candidate"].head(500).to_dict(orient="records")) if len(rf) else []
            rrep = {"pole": pole, **rr, "georef": radar_layers["georef"].as_dict()}
        except Exception as exc:  # noqa: BLE001
            rrep = {"pole": pole, "status": "RADAR_SCREEN_FAILED", "error": f"{type(exc).__name__}: {exc}"[:300]}
    elif radar_layers and radar_layers.get("error"):
        rrep = {"pole": pole, "status": "RADAR_LAYER_UNREADABLE", "error": radar_layers["error"]}
    _write(out / f"radar_{pole}.json", rrep)
    print(f"[crypt] radar {pole}: {rrep.get('status')} counts={rrep.get('counts')}")
    return rep


def floor_statement(screen_rep: dict, sens: dict) -> dict:
    """The smallest injected 300 K area recovered in ≥ 50 % of trials, per T_hot,
    and the median per-bin 5σ floor of the primary channel per season."""
    out = {"recovered_half_area_m2": {}, "per_season_primary_floor_m2_median": {}}
    for th, s in sens.get("by_t_hot", {}).items():
        rows = [r for r in s.get("rows", []) if r.get("n_injected")]
        half = [r["area_m2"] for r in rows if r.get("recovered_frac", 0) >= 0.5]
        out["recovered_half_area_m2"][th] = min(half) if half else None
    for season, r in screen_rep.get("per_season", {}).items():
        prim = r.get("primary")
        if prim and prim in r.get("noise", {}):
            a = [b["A_min_m2"] for b in r["noise"][prim]["bins"] if b.get("A_min_m2") is not None
                 and np.isfinite(b["A_min_m2"])]
            out["per_season_primary_floor_m2_median"][season] = float(np.median(a)) if a else None
    return out


def stage_screen(conf: dict, out: Path, *, fetch=None, shard: str | None = None, poles=None,
                 synthetic: bool = False, run_sensitivity: bool = True) -> dict:
    fetch = fetch or A.http_fetch
    reports = {}
    for pole in shard_poles(conf, shard, poles):
        if synthetic:
            layers = T.synthetic_pole(160, pole=pole, seed=3 if pole == "south" else 4)
            layers = T.inject(layers, [(80, 80)], 300.0 / layers.georef.pixel_area_m2, 300.0)
            layers = T.inject(layers, [(60, 80)], 3000.0 / layers.georef.pixel_area_m2, 300.0, seasons=["summer"])
            cpr, s1, _m = R.synthetic_radar(160, seed=5)
            cpr[80, 80] = 1.8
            s1[80, 80] = 8.0
            rl = {"cpr": cpr, "s1": s1, "georef": layers.georef}
            reports[pole] = screen_one_pole(conf, out, pole, layers, radar_layers=rl,
                                            psr_ledger={"status": "SYNTHETIC"}, run_sensitivity=run_sensitivity)
            continue
        acq_rep = _read_json(out / f"acquire_{pole}.json")
        if not acq_rep or acq_rep.get("status") != "OK":
            rep = {"pole": pole, "status": VERDICT_NO_DATA, "generated_utc": _now(),
                   "note": "no acquire report with OK products for this pole",
                   "acquire_status": (acq_rep or {}).get("status"), "counts": {c: 0 for c in T.CLASSES}}
            _write(out / f"screen_{pole}.json", rep)
            print(f"[crypt] screen {pole}: {VERDICT_NO_DATA}")
            reports[pole] = rep
            continue
        layers, notes = build_layers(conf, pole, acq_rep)
        if layers is None or not layers.arrays:
            rep = {"pole": pole, "status": VERDICT_NO_DATA, "generated_utc": _now(), "layers_notes": notes,
                   "counts": {c: 0 for c in T.CLASSES}}
            _write(out / f"screen_{pole}.json", rep)
            print(f"[crypt] screen {pole}: {VERDICT_NO_DATA} (no readable raster)")
            reports[pole] = rep
            continue
        psr = V.load_external_psr(conf, fetch, out / "data" / pole, layers.georef, pole)
        rl = _load_minirf(acq_rep)
        rep = screen_one_pole(conf, out, pole, layers, fetch=fetch, radar_layers=rl,
                              external_psr=psr["mask"], psr_ledger={k: v for k, v in psr.items() if k != "mask"},
                              run_sensitivity=run_sensitivity)
        rep["layers_notes"] = notes
        _write(out / f"screen_{pole}.json", rep)
        reports[pole] = rep
    return reports


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------
def stage_assess(conf: dict, out: Path, *, fetch=None, synthetic: bool = False, do_vet: bool = True) -> dict:
    fetch = fetch or A.http_fetch
    poles = conf["products"]["poles"]
    screens = {p: _read_json(out / f"screen_{p}.json") for p in poles}
    radars = {p: _read_json(out / f"radar_{p}.json") for p in poles}
    sens = {p: _read_json(out / f"sensitivity_{p}.json") for p in poles}
    acqs = {p: _read_json(out / f"acquire_{p}.json") for p in poles}
    probe = _read_json(out / "probe.json") or {}
    ok_poles = [p for p in poles if screens.get(p) and screens[p].get("status") == "OK"]
    degraded: list[str] = []
    counts = {c: 0 for c in T.CLASSES}
    counts["below_threshold_other_season"] = 0
    n_interior, n_flagged, area_km2 = 0, 0, 0.0
    cands: list[dict] = []
    for p in ok_poles:
        s = screens[p]
        for c, n in (s.get("counts") or {}).items():
            counts[c] = counts.get(c, 0) + int(n)
        n_interior += int(s.get("mask", {}).get("n_interior", 0))
        area_km2 += float(s.get("mask", {}).get("area_km2_interior", 0.0) or 0.0)
        n_flagged += int(s.get("n_flagged_any_season", 0))
        for d in s.get("degraded", []):
            degraded.append(f"{p}:{d}")
        if (s.get("psr_route") or {}).get("status") != "EXTERNAL_PSR_USED":
            degraded.append(f"{p}:psr_from_thermal_definition_only")
        if not radars.get(p) or radars[p].get("status") != "OK":
            degraded.append(f"{p}:no_radar_layer")
        cands.extend(s.get("candidates", []))
    radar_layers = {}
    if do_vet and not synthetic:
        for p in ok_poles:
            a = acqs.get(p)
            if a:
                rl = _load_minirf(a)
                if rl and rl.get("cpr") is not None:
                    radar_layers[p] = rl
    vetted = V.vet_candidates(cands, conf, None if (synthetic or not do_vet) else fetch,
                              radar_layers=radar_layers)
    missing_poles = [p for p in poles if p not in ok_poles]
    for p in missing_poles:
        degraded.append(f"{p}:{(screens.get(p) or {}).get('status') or 'NOT_SCREENED'}")
    if not ok_poles:
        verdict = VERDICT_NO_DATA
    elif vetted:
        verdict = f"{VERDICT_CANDIDATES} ({len(vetted)})"
    else:
        verdict = VERDICT_NONE
    if degraded and ok_poles:
        verdict = f"{verdict}; DEGRADED ({', '.join(sorted(set(degraded)))})"
    if synthetic:
        verdict = "SYNTHETIC_" + verdict
    routes = {"diviner_roots": [r.get("url") for r in probe.get("diviner", {}).get("roots_tried", [])],
              "diviner_reached": probe.get("reached", {}).get("diviner"),
              "n_products_listed": probe.get("diviner", {}).get("n_products"),
              "n_selected": probe.get("diviner", {}).get("n_selected"),
              "n_needed": probe.get("diviner", {}).get("n_needed")}
    summary = {
        "verdict": verdict, "generated_utc": _now(), "synthetic": bool(synthetic),
        "poles_screened": ok_poles, "poles_missing": missing_poles,
        "n_psr_interior_px": n_interior, "psr_interior_area_km2": area_km2,
        "n_flagged_any_season": n_flagged, "counts": counts, "n_candidates": len(vetted),
        "candidates": vetted[:200], "degraded": sorted(set(degraded)),
        "floor": {p: (sens.get(p) or {}).get("floor_statement") for p in ok_poles},
        "mask": {p: screens[p].get("mask") for p in ok_poles},
        "per_season": {p: {s: {k: v for k, v in r.items() if k in ("ref", "primary", "coverage", "status")}
                           for s, r in (screens[p].get("per_season") or {}).items()} for p in ok_poles},
        "radar": {p: {k: v for k, v in (radars.get(p) or {}).items() if k in ("status", "counts", "n_mask_px",
                                                                               "n_valid_px", "n_high_cpr")}
                  for p in poles},
        "acquisition": {p: {"status": (acqs.get(p) or {}).get("status"), "n_ok": (acqs.get(p) or {}).get("n_ok"),
                            "n_missing": (acqs.get(p) or {}).get("n_missing"),
                            "minirf": ((acqs.get(p) or {}).get("minirf") or {}).get("status")} for p in poles},
        "routes": routes,
        "thresholds": conf["thermal"],
        "note": ("A candidate is a PSR-interior pixel whose primary short-channel (6 or 7) brightness "
                 "temperature exceeds the long-channel reference by >= z_min empirical sigma in BOTH the "
                 "summer and the winter product, with a season-independent excess radiance, a compact "
                 "footprint, no row/column stripe, an adequate observation count and a two-component "
                 "spectrum that beats a single temperature. It is PENDING the optical/radar vet listed "
                 "per candidate under unexcluded_systematics. NO_ANISOTHERMAL_SURVIVOR is a count at the "
                 "stated floor, not an occurrence limit, and is not written up (CLAUDE.md). "
                 "NO_DATA_REACHED is an access statement."),
    }
    _write(out / "summary.json", summary)
    _write(out / "candidates.json", {"generated_utc": _now(), "n": len(vetted), "candidates": vetted})
    if vetted:
        flat = [{k: v for k, v in c.items() if not isinstance(v, (dict, list))} for c in vetted]
        pd.DataFrame(flat).to_csv(out / "candidates.csv", index=False)
    else:
        pd.DataFrame(columns=["pole", "line", "sample", "lon", "lat", "class"]).to_csv(out / "candidates.csv", index=False)
    print(f"[crypt] assess: {verdict}; interior px={n_interior} ({area_km2:.0f} km2); "
          f"flagged={n_flagged}; counts={counts}")
    return summary


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def crypt_run(conf: dict | None = None, stage: str = "all", *, shard: str | None = None, poles=None,
              out_dir=None, fetch=None, synthetic: bool = False, run_sensitivity: bool = True,
              do_vet: bool = True) -> dict:
    conf = conf if conf is not None else load_crypt_config()
    out = Path(out_dir) if out_dir else repo_root() / "results" / "crypt"
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage in ("all", "", None) else tuple(s.strip() for s in stage.split(","))
    rep: dict = {}
    for s in stages:
        if s == "probe":
            if synthetic:
                continue
            rep = stage_probe(conf, out, fetch=fetch)
        elif s == "acquire":
            if synthetic:
                continue
            rep = stage_acquire(conf, out, fetch=fetch, shard=shard, poles=poles)
        elif s == "screen":
            rep = stage_screen(conf, out, fetch=fetch, shard=shard, poles=poles, synthetic=synthetic,
                               run_sensitivity=run_sensitivity)
        elif s == "assess":
            rep = stage_assess(conf, out, fetch=fetch, synthetic=synthetic, do_vet=do_vet)
        else:
            raise SystemExit(f"unknown stage {s!r}; choose from {STAGES}")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(prog="seti crypt",
                                description="CRYPT (S55): anisothermal hot components and compact radar "
                                            "anomalies inside lunar permanently shadowed regions")
    p.add_argument("--stage", default="all", help="probe|acquire|screen|assess|all or a comma list")
    p.add_argument("--shard", default="", help="i/n: this job's share of the poles")
    p.add_argument("--poles", default="", help="comma list (default from config)")
    p.add_argument("--out-dir", default="", help="results directory (default results/crypt)")
    p.add_argument("--config", default="", help="alternative config yaml")
    p.add_argument("--synthetic", action="store_true", help="offline smoke run on a synthetic pole")
    p.add_argument("--no-sensitivity", action="store_true", help="skip the injection-recovery table")
    p.add_argument("--no-vet", action="store_true", help="skip the ODE coverage queries at assess")
    a = p.parse_args(argv)
    conf = load_crypt_config(Path(a.config) if a.config else None)
    poles = [x.strip() for x in a.poles.split(",") if x.strip()] or None
    rep = crypt_run(conf, stage=a.stage, shard=a.shard or None, poles=poles, out_dir=a.out_dir or None,
                    synthetic=a.synthetic, run_sensitivity=not a.no_sensitivity, do_vet=not a.no_vet)
    v = rep.get("verdict") if isinstance(rep, dict) else None
    if v:
        print(f"[crypt] verdict: {v}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["DEFAULTS", "STAGES", "VERDICT_CANDIDATES", "VERDICT_NONE", "VERDICT_NO_DATA", "build_layers",
           "crypt_run", "load_crypt_config", "main", "needed_products", "parse_shard", "shard_poles",
           "stage_acquire", "stage_assess", "stage_probe", "stage_screen"]
