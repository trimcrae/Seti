"""Stage orchestration for ROMAN.  Writes ``results/roman/``.

Stages (``python -m seti.roman.run <stage> ...`` or ``seti roman <stage> ...``):

``probe``      what does the archive serve today?  IRSA TAP/SIA, the S3
               buckets, MAST, the simulation pages, the Python packages ->
               ``probe.json`` with a single ``data_state``.
``inventory``  every product the probe could see, classified into level /
               kind / survey / simulated, plus a shard plan -> ``inventory.json``.
``ingest``     pull a bounded set of products into the four structures of
               :mod:`seti.roman.schema` (cached under ``data/cache/roman/``)
               -> ``ingest.json`` and a manifest.
``screen``     one shard of one channel (``lens``, ``lines``, ``flash``,
               ``statite``, ``paces``) over the ingested objects; one JSON
               checkpoint per object, written immediately.
``assess``     gather checkpoints, run each channel's cross-object vetoes,
               write ``<channel>/summary.json`` + ``candidates.json`` and the
               top-level ``summary.json``.
``selftest``   the synthetic injection/rejection battery through the SAME
               screen and assess code paths (offline gate; the runner runs it
               too before touching the archive).
``readiness``  ``probe`` + ``inventory`` + the diff against the previous
               readiness record -> ``readiness.json`` (what ``alerts.py`` reads).

Two rules every verdict obeys.  A verdict never reads as a sky null when
nothing real was analysed (``NO_DATA_REACHED`` / ``ROMAN_NOT_YET_PUBLIC``).
And a run on simulated inputs can never carry a candidate tier: it says
``SIMULATION_PACES_OK`` or ``SIMULATION_PACES_FAILED`` -- did the data flow
through every stage -- and nothing else.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np

from .schema import (
    DQCutout,
    Funnel,
    LightCurve,
    Spectrum,
    load_roman_config,
    read_json,
    repo_root,
    utc_now,
    write_json,
)

CHANNELS = ("lens", "lines", "flash", "statite", "paces")
KINDS = ("lightcurve", "dq", "spectrum", "cgi", "obseq", "catalog")
CHANNEL_KINDS = {"lens": ("lightcurve",), "paces": ("lightcurve", "spectrum"),
                 "lines": ("spectrum",), "flash": ("dq",), "statite": ("cgi",)}
VERDICTS = ("NO_DATA_REACHED", "ROMAN_NOT_YET_PUBLIC", "SIMULATION_PACES_OK",
            "SIMULATION_PACES_FAILED", "NO_CANDIDATE", "DEGRADED_SOURCE")


def default_out_dir() -> Path:
    return repo_root() / "results" / "roman"


def default_cache_dir() -> Path:
    return repo_root() / "data" / "cache" / "roman"


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")[:120] or "unknown"


def _log(msg: str) -> None:
    print(f"[roman] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Normalised-object store (ingest writes, screen reads).  npz + json sidecar so
# that the screen shards never need the archive or the reader packages.
# ---------------------------------------------------------------------------

def _obj_id(kind: str, name: str) -> str:
    h = hashlib.sha1(f"{kind}:{name}".encode()).hexdigest()[:10]
    return f"{slug(name)}_{h}"


def save_object(obj, kind: str, store: Path, simulated: bool, source_uri: str = "") -> dict:
    """Persist one schema object; returns the manifest row."""
    store = Path(store)
    name = getattr(obj, "star_id", None) or getattr(obj, "source_id", None) or getattr(obj, "image_id", None)
    oid = _obj_id(kind, f"{name}:{getattr(obj, 'band', '')}:{getattr(obj, 'mode', '')}:{source_uri}")
    d = store / kind
    d.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    scalars: dict = {"kind": kind, "class": type(obj).__name__, "simulated": bool(simulated),
                     "source_uri": source_uri}
    for k, v in obj.__dict__.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        elif isinstance(v, (list, tuple)) and k in ("stars",):
            scalars[k] = json.loads(json.dumps(v, default=str))
        else:
            try:
                json.dumps(v)
                scalars[k] = v
            except TypeError:
                scalars[k] = str(v)
    np.savez_compressed(d / f"{oid}.npz", **arrays)
    write_json(d / f"{oid}.json", scalars)
    return {"id": oid, "kind": kind, "class": scalars["class"], "simulated": bool(simulated),
            "source_uri": source_uri, "name": str(name)}


def load_object(row: dict, store: Path):
    """Rebuild the schema object a manifest row points at."""
    d = Path(store) / row["kind"]
    scalars = read_json(d / f"{row['id']}.json", {}) or {}
    arrays = {}
    p = d / f"{row['id']}.npz"
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    cls = {"LightCurve": LightCurve, "Spectrum": Spectrum, "DQCutout": DQCutout}.get(scalars.get("class"))
    fields = {k: v for k, v in scalars.items() if k not in ("kind", "class", "simulated", "source_uri")}
    fields.update(arrays)
    if cls is None:
        if scalars.get("class") == "CGISource":
            from .statite import CGISource
            return CGISource(**fields)
        raise ValueError(f"unknown stored class {scalars.get('class')!r}")
    return cls(**fields)


# ---------------------------------------------------------------------------
# probe / inventory / readiness
# ---------------------------------------------------------------------------

def probe(out_dir: Path, conf: dict) -> dict:
    from .archive import probe as _probe
    rec = _probe(conf)
    rec["stage"] = "probe"
    write_json(out_dir / "probe.json", rec)
    _log(f"probe: data_state={rec.get('data_state')} reached={rec.get('n_endpoints_reached')}")
    for e in rec.get("data_state_evidence") or []:
        _log(f"  evidence: {e}")
    return rec


def inventory(out_dir: Path, conf: dict, n_shards: int = 4, probe_record: dict | None = None) -> dict:
    from .archive import inventory as _inventory
    from .archive import plan_shards
    prec = probe_record or read_json(out_dir / "probe.json") or probe(out_dir, conf)
    inv = _inventory(conf, prec)
    products = inv.get("products") or []
    inv["shards"] = plan_shards(products, n_shards, KINDS) if products else []
    inv["stage"] = "inventory"
    inv["data_state"] = prec.get("data_state")
    inv["written_utc"] = utc_now()
    write_json(out_dir / "inventory.json", inv)
    write_json(out_dir / "shards.json", inv["shards"])
    _log(f"inventory: {inv.get('verdict')} products={len(products)} "
         f"mission={inv.get('n_mission')} simulated={inv.get('n_simulated')}")
    return inv


def readiness(out_dir: Path, conf: dict) -> dict:
    """Probe + inventory, diffed against the last readiness record."""
    prev = read_json(out_dir / "readiness.json") or {}
    prec = probe(out_dir, conf)
    inv = inventory(out_dir, conf, probe_record=prec)
    state = prec.get("data_state")
    rec = {
        "stage": "readiness", "written_utc": utc_now(), "data_state": state,
        "data_state_evidence": prec.get("data_state_evidence"),
        "previous_data_state": prev.get("data_state"),
        "state_changed": (prev.get("data_state") is not None and prev.get("data_state") != state),
        "n_products": len(inv.get("products") or []),
        "n_products_previous": prev.get("n_products"),
        "counts_by_kind": inv.get("counts_by_kind"),
        "first_seen_simulations": (state == "SIMULATIONS_ONLY" and prev.get("data_state")
                                   not in ("SIMULATIONS_ONLY", "MISSION_DATA_PRESENT")),
        "first_seen_mission_data": (state == "MISSION_DATA_PRESENT"
                                    and prev.get("data_state") != "MISSION_DATA_PRESENT"),
        "milestone_state": (conf.get("readiness") or {}).get("milestone_state"),
        "packages": prec.get("packages"),
        "history": ([*(prev.get("history") or [])][-23:]
                    + [{"utc": utc_now(), "data_state": state,
                        "n_products": len(inv.get("products") or [])}]),
    }
    write_json(out_dir / "readiness.json", rec)
    _log(f"readiness: {state} (previous {prev.get('data_state')}), products {rec['n_products']}")
    return rec


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

# Product kinds that only make sense as a sibling of another product (fetched with it).
_SIBLING_FORMATS = ("snana_phot", "openuniverse_truth_index")
# OpenUniverse ``meta.format`` -> ingest kind.
_FORMAT_KINDS = {"snana_head": "lightcurve", "openuniverse_image": "dq", "obseq": "obseq",
                 "pointsource": "catalog", "galaxy_catalog": "catalog", "snana_catalog": "catalog"}


def _ou_meta(product: dict) -> dict:
    """The product's ``meta`` with the OpenUniverse facts (``format``, siblings, band)
    filled in from the filename when the inventory did not record them."""
    meta = dict(product.get("meta") or {})
    if meta.get("format"):
        return meta
    from .openuniverse import classify_openuniverse_uri
    inferred = classify_openuniverse_uri(str(product.get("uri") or ""))
    if inferred:
        merged = dict(inferred)
        merged.update(meta)
        return merged
    return meta


def _kind_of(product: dict) -> str | None:
    fmt = str(_ou_meta(product).get("format") or "")
    if fmt in _FORMAT_KINDS:
        return _FORMAT_KINDS[fmt]
    if fmt in _SIBLING_FORMATS:
        return None
    k = str(product.get("kind") or "")
    if k == "lightcurve":
        return "lightcurve"
    if k in ("spectrum_1d",):
        return "spectrum"
    if k == "cal":
        return "dq"
    if k == "cgi":
        return "cgi"
    if k == "obseq":
        return "obseq"
    return None


def _merge_calibration(out_dir: Path, updates: dict) -> dict:
    """Create or update ``results/roman/calibration.json`` (the measured-vs-assumed
    ledger); other keys are kept, ``written_utc`` is refreshed."""
    path = Path(out_dir) / "calibration.json"
    cal = read_json(path, {}) or {}
    cal.update(updates)
    cal["written_utc"] = utc_now()
    write_json(path, cal)
    return cal


def _fetch_sibling(ctx: dict, uri: str) -> Path | None:
    """Fetch a product's sibling (PHOT table, truth index) through the ingest's
    fetcher, charging its size to the download budget; local paths pass through."""
    if not uri:
        return None
    fetch = ctx.get("fetch")
    budget = ctx.get("budget") or {}
    if fetch is None:
        p = Path(str(uri))
        return p if p.exists() else None
    remaining = None
    if budget:
        remaining = max(0.0, float(budget.get("max", 0.0)) - float(budget.get("spent", 0.0)))
        if remaining <= 0:
            return None
    local = fetch(uri, remaining)
    if local is None:
        return None
    try:
        if budget:
            budget["spent"] = float(budget.get("spent", 0.0)) + float(Path(local).stat().st_size)
    except Exception:  # noqa: BLE001
        pass
    return Path(local)


def ingest(out_dir: Path, conf: dict, kinds=KINDS, limit: int = 200, cache_dir: Path | None = None,
           inventory_record: dict | None = None, session=None,
           max_objects_per_product: int | None = None) -> dict:
    """Pull up to ``limit`` products per kind and normalise them.

    Mission products are preferred over simulated ones; a product whose
    reader is unavailable (no ``asdf`` on this machine, an unknown table
    layout) is recorded under ``unreadable`` with the reason -- the count is
    part of the result, never silently dropped.  Multi-object products (an
    SNANA HEAD/PHOT pair, an image full of stars) stop after
    ``max_objects_per_product`` objects (``archive.max_objects_per_product``).
    OpenUniverse pointing sequences are not objects: their cadence record and
    the per-image dq-flag census go to ``calibration.json``; the input
    catalogues are recorded as ``catalog_only`` and not fetched.
    """
    from . import products as P
    from .archive import fetch_to_cache

    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    store = cache_dir / "normalized"
    inv = inventory_record or read_json(out_dir / "inventory.json") or {}
    products = inv.get("products") or []
    funnel = Funnel()
    manifest: list[dict] = []
    unreadable: list[dict] = []
    catalog_only: list[dict] = []
    per_kind: dict[str, int] = {}
    n_by_kind: dict[str, int] = {}
    flags, flags_src = P.dq_flags(conf)
    arch = conf.get("archive") or {}
    max_bytes = float(arch.get("max_download_bytes_per_run", 2e10))
    if max_objects_per_product is None:
        max_objects_per_product = int(arch.get("max_objects_per_product", 200))
    budget = {"spent": 0.0, "max": max_bytes}
    calib_updates: dict = {}
    census_sum: dict = {"n_images": 0, "n_pixels": 0, "n_nonzero": 0, "per_flag": {}, "images": [],
                        "dq_flags_source": flags_src}
    zp_implied: dict[str, list[float]] = {}

    def fetch(uri, remaining=None):
        kw = {"session": session}
        if remaining is not None:
            kw["max_bytes"] = int(remaining)
        return fetch_to_cache(uri, cache_dir / "raw", **kw)

    for prod in sorted(products, key=lambda r: (bool(r.get("simulated")), r.get("size_bytes") or 0)):
        kind = _kind_of(prod)
        if kind is None or kind not in kinds:
            continue
        ou_meta = _ou_meta(prod)
        if kind == "catalog":
            catalog_only.append({"uri": prod["uri"], "format": ou_meta.get("format"),
                                 "size_bytes": prod.get("size_bytes")})
            funnel.bump("catalog_only")
            continue
        if per_kind.get(kind, 0) >= int(limit):
            continue
        funnel.bump(f"selected_{kind}")
        size = float(prod.get("size_bytes") or 0)
        if budget["spent"] + size > max_bytes:
            funnel.reject("download_budget_exhausted")
            continue
        local = fetch(prod["uri"])
        if local is None:
            unreadable.append({"uri": prod["uri"], "reason": "fetch_failed"})
            funnel.reject("fetch_failed")
            continue
        budget["spent"] += size
        ctx = {"fetch": fetch, "budget": budget, "max_objects": max_objects_per_product, "ou_meta": ou_meta}
        if kind == "obseq":
            try:
                from .openuniverse import read_obseq
                rec = read_obseq(local, conf)
                rec["uri"] = prod["uri"]
                calib_updates["survey_cadence_sim"] = rec
                funnel.bump("normalised_obseq")
                per_kind[kind] = per_kind.get(kind, 0) + 1
            except Exception as exc:  # noqa: BLE001
                unreadable.append({"uri": prod["uri"], "reason": f"{type(exc).__name__}: {exc}", "needs": []})
                funnel.reject("reader_unavailable")
            continue
        try:
            objs = _normalise(kind, local, prod, conf, P, flags, ctx)
        except Exception as exc:  # noqa: BLE001
            objs = P.ReaderUnavailable(reason=f"{type(exc).__name__}: {exc}", uri=prod["uri"], needs=[])
        rec = ctx.get("image_record")
        if rec and rec.get("dq_flag_census"):
            c = rec["dq_flag_census"]
            census_sum["n_images"] += 1
            census_sum["n_pixels"] += int(c.get("n_pixels", 0))
            census_sum["n_nonzero"] += int(c.get("n_nonzero", 0))
            for name, n in (c.get("per_flag") or {}).items():
                census_sum["per_flag"][name] = int(census_sum["per_flag"].get(name, 0)) + int(n)
            if len(census_sum["images"]) < 200:
                census_sum["images"].append({k: rec.get(k) for k in (
                    "image_id", "band", "detector", "mjd", "exptime", "zptmag", "n_stars_in_image",
                    "n_stars_used", "star_source", "truth_xy_origin", "truth_xy_offset",
                    "zp_ab_implied_per_e_s", "dq_all_zero")} | {"n_nonzero": int(c.get("n_nonzero", 0)),
                                                                   "uri": prod["uri"]})
            if rec.get("zp_ab_implied_per_e_s") is not None and rec.get("band"):
                zp_implied.setdefault(str(rec["band"]), []).append(float(rec["zp_ab_implied_per_e_s"]))
        if isinstance(objs, P.ReaderUnavailable):
            unreadable.append({"uri": prod["uri"], "reason": objs.reason, "needs": list(objs.needs)})
            funnel.reject("reader_unavailable")
            continue
        for o in objs:
            simulated = bool(prod.get("simulated")) or bool((getattr(o, "meta", None) or {}).get("simulated"))
            manifest.append(save_object(o, kind, store, simulated, prod["uri"]))
            funnel.bump(f"normalised_{kind}")
            n_by_kind[kind] = n_by_kind.get(kind, 0) + 1
        per_kind[kind] = per_kind.get(kind, 0) + 1
    if census_sum["n_images"]:
        calib_updates["dq_flag_census_sim"] = census_sum
    if zp_implied:
        ftab = ((conf.get("instruments") or {}).get("WFI", {}).get("filters") or {})
        calib_updates["zero_points_sim"] = {
            b: {"implied_ab_per_e_s_median": float(np.median(v)), "n_images": len(v),
                "config_zp_ab": (ftab.get(b) or {}).get("zp_ab")} for b, v in zp_implied.items()}
    if calib_updates:
        _merge_calibration(out_dir, calib_updates)
    rec = {"stage": "ingest", "written_utc": utc_now(), "kinds": list(kinds), "limit": int(limit),
           "max_objects_per_product": int(max_objects_per_product),
           "n_products_selected": int(sum(per_kind.values())), "per_kind": per_kind,
           "n_objects": len(manifest), "n_objects_by_kind": n_by_kind,
           "n_unreadable": len(unreadable), "unreadable": unreadable[:200],
           "catalog_only": catalog_only[:200], "calibration_records": sorted(calib_updates),
           "dq_flags_source": flags_src, "bytes_downloaded": budget["spent"],
           "simulated_inputs": (all(m["simulated"] for m in manifest) if manifest else None),
           "funnel": funnel.as_dict(), "store": str(store)}
    write_json(out_dir / "ingest.json", rec)
    write_json(cache_dir / "manifest.json", {"written_utc": utc_now(), "objects": manifest})
    _log(f"ingest: {len(manifest)} objects from {rec['n_products_selected']} products; "
         f"{len(unreadable)} unreadable; {len(catalog_only)} catalog-only")
    return rec


def _normalise(kind: str, local: Path, prod: dict, conf: dict, P, flags: dict, ctx: dict | None = None) -> list:
    """One fetched product -> schema objects (or ``ReaderUnavailable``).

    Dispatches first on the OpenUniverse ``meta.format`` (SNANA HEAD + its PHOT
    sibling; galsim image + its truth index), then on the generic kind.  ``ctx``
    carries the ingest's fetcher / budget / ``max_objects`` and receives
    ``image_record`` (header facts and the dq census) for an image.
    """
    ctx = ctx if ctx is not None else {}
    meta = ctx.get("ou_meta") or _ou_meta(prod)
    fmt = str(meta.get("format") or "")
    max_objects = ctx.get("max_objects")
    if fmt == "snana_head":
        from .openuniverse import iter_snana_lightcurves
        sib = meta.get("phot_sibling")
        if not sib:
            return P.ReaderUnavailable(reason="SNANA HEAD without a PHOT sibling", uri=str(local),
                                       needs=["phot_sibling"])
        phot = _fetch_sibling(ctx, str(sib))
        if phot is None:
            return P.ReaderUnavailable(reason=f"PHOT sibling not fetched: {sib}", uri=str(local),
                                       needs=["phot_sibling"])
        return list(iter_snana_lightcurves(local, phot, conf, max_objects=max_objects, roman_only=True))
    if fmt == "openuniverse_image":
        from .openuniverse import read_openuniverse_image
        cands = list(meta.get("truth_index_candidates") or [])
        if meta.get("truth_index") and meta["truth_index"] not in cands:
            cands.insert(0, meta["truth_index"])
        truth = None
        for c in cands:
            truth = _fetch_sibling(ctx, str(c))
            if truth is not None:
                break
        oc = conf.get("openuniverse") or {}
        rec: dict = {"truth_index": str(truth) if truth else None, "truth_index_candidates": cands}
        ctx["image_record"] = rec
        cuts = read_openuniverse_image(local, truth, conf, max_stars=int(oc.get("max_stars_per_image", 2000)),
                                       mag_limit=float(oc.get("star_mag_limit", 24.0)), image_record=rec,
                                       flags=flags)
        if isinstance(cuts, P.ReaderUnavailable):
            return cuts
        if truth is None:
            return P.ReaderUnavailable(reason="no truth index reachable and no star catalogue given",
                                       uri=str(local), needs=["truth_index"])
        return cuts[:int(max_objects)] if max_objects is not None else cuts
    if kind == "lightcurve":
        return P.read_lightcurve_table(local, conf=conf, band=prod.get("band"), survey=prod.get("survey") or "")
    if kind == "spectrum":
        return P.read_spectrum_table(local, conf=conf)
    if kind == "dq":
        arrs = P.read_asdf_arrays(local, keys=("dq",), lazy=True)
        if isinstance(arrs, P.ReaderUnavailable):
            return arrs
        meta = arrs.get("meta") or {}
        stars = list(meta.get("stars") or [])   # a catalogue join is the runner's job; recorded if present
        band = str(meta.get("optical_element") or prod.get("band") or "")
        fwhm = float((conf["instruments"]["WFI"].get("psf_fwhm_px") or {}).get(band, 1.2))
        return P.dq_cutouts_from_image(arrs["dq"], stars, image_id=str(prod["uri"]), band=band,
                                       detector=str(meta.get("detector") or ""), psf_fwhm_px=fwhm,
                                       mjd=meta.get("start_time_mjd"))
    if kind == "cgi":
        return P.ReaderUnavailable(reason="CGI product layout unknown before release", uri=str(local),
                                   needs=["cgi_reader"])
    return P.ReaderUnavailable(reason=f"no reader for kind {kind}", uri=str(local), needs=[])


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------

def _screen_one(channel: str, obj, conf: dict, ctx: dict) -> dict:
    """Dispatch one object to one channel's pure screen function."""
    if channel == "lens":
        from .lens import screen_lightcurve
        colour = ctx.get("colour", {}).get(getattr(obj, "star_id", None))
        return screen_lightcurve(obj, conf, colour_lc=colour)
    if channel == "paces":
        from .bridge import pace_lightcurve, pace_spectrum
        if isinstance(obj, Spectrum):
            return pace_spectrum(obj, conf)
        colour = ctx.get("colour", {}).get(getattr(obj, "star_id", None))
        return pace_lightcurve(obj, conf, colour_lc=colour)
    if channel == "lines":
        from .lines import screen_spectrum
        return screen_spectrum(obj, conf)
    if channel == "flash":
        from .flash import screen_dq_cutout
        return screen_dq_cutout(obj, conf, ctx["flags"], hot_ledger=ctx.get("ledger"))
    if channel == "statite":
        from .statite import screen_source
        return screen_source(obj, conf)
    raise ValueError(f"unknown channel {channel!r}")


def _colour_index(rows: list[dict], store: Path, conf: dict) -> dict:
    """F087 light curves keyed by star, for the achromaticity tests."""
    colour_band = str(((conf.get("surveys") or {}).get("GBTDS") or {}).get("color_band", "F087"))
    out = {}
    for r in rows:
        if r["kind"] != "lightcurve":
            continue
        scal = read_json(Path(store) / "lightcurve" / f"{r['id']}.json", {}) or {}
        if str(scal.get("band", "")).upper() == colour_band.upper():
            try:
                out[str(scal.get("star_id"))] = load_object(r, store)
            except Exception:  # noqa: BLE001
                continue
    return out


def screen(out_dir: Path, conf: dict, channel: str, shard: int = 0, n_shards: int = 1,
           cache_dir: Path | None = None, objects: list | None = None,
           simulated: bool | None = None) -> dict:
    """One shard of one channel; a checkpoint per object, written immediately."""
    from . import products as P

    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    store = cache_dir / "normalized"
    ckpt_dir = out_dir / "checkpoints" / channel
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    flags, flags_src = P.dq_flags(conf)
    ctx: dict = {"flags": flags}
    if channel == "flash":
        from .flash import HotPixelLedger
        led = read_json(out_dir / "flash" / "hot_ledger.json")
        ctx["ledger"] = HotPixelLedger.from_dict(led) if isinstance(led, dict) else HotPixelLedger()
    funnel = Funnel()
    kinds = CHANNEL_KINDS[channel]
    todo: list[tuple[str, object, bool]] = []
    if objects is not None:
        for i, o in enumerate(objects):
            todo.append((f"obj{i}", o, bool(simulated)))
    else:
        man = read_json(cache_dir / "manifest.json") or {}
        rows = [r for r in (man.get("objects") or []) if r["kind"] in kinds]
        colour_band = str(((conf.get("surveys") or {}).get("GBTDS") or {}).get("color_band", "F087"))
        ctx["colour"] = _colour_index(rows, store, conf)
        for i, r in enumerate(rows):
            if i % max(1, int(n_shards)) != int(shard):
                continue
            if r["kind"] == "lightcurve":
                scal = read_json(store / "lightcurve" / f"{r['id']}.json", {}) or {}
                if str(scal.get("band", "")).upper() == colour_band.upper():
                    funnel.bump("colour_curves_reserved")
                    continue
            try:
                todo.append((r["id"], load_object(r, store), bool(r.get("simulated"))))
            except Exception as exc:  # noqa: BLE001
                funnel.reject("object_unloadable")
                _log(f"  cannot load {r['id']}: {exc!r}")
    n_done = n_err = 0
    any_sim = None
    for oid, obj, sim in todo:
        any_sim = (sim if any_sim is None else (any_sim or sim))
        p = ckpt_dir / f"{slug(oid)}.json"
        if p.exists() and objects is None:
            funnel.bump("checkpoint_reused")
            continue
        try:
            rec = _screen_one(channel, obj, conf, ctx)
            rec = dict(rec or {})
            rec.update({"object_id": oid, "simulated": sim, "status": rec.get("status", "screened")})
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            rec = {"object_id": oid, "simulated": sim, "status": "error",
                   "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]}
            n_err += 1
        write_json(p, rec)
        funnel.bump("screened")
    if channel == "flash" and ctx.get("ledger") is not None:
        write_json(out_dir / "flash" / "hot_ledger.json", ctx["ledger"].as_dict())
    rec = {"stage": "screen", "channel": channel, "shard": int(shard), "n_shards": int(n_shards),
           "written_utc": utc_now(), "n_objects": len(todo), "n_screened": n_done, "n_errors": n_err,
           "simulated_inputs": any_sim, "dq_flags_source": flags_src, "funnel": funnel.as_dict()}
    write_json(out_dir / f"screen_{channel}_shard{int(shard)}.json", rec)
    _log(f"screen {channel} shard {shard}/{n_shards}: {n_done} screened, {n_err} errors")
    return rec


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------

def _lens_assess(records: list[dict], conf: dict) -> dict:
    tiers = (conf.get("lens") or {}).get("tiers") or {}
    cand_tier = tiers.get("candidate", "OPAQUE_LENS_CANDIDATE")
    int_tier = tiers.get("interest", "OCCULTING_PREFERRED_PENDING_VET")
    counts: dict[str, int] = {}
    for r in records:
        t = str(r.get("tier") or r.get("status") or "unknown")
        counts[t] = counts.get(t, 0) + 1
    n_cand = counts.get(cand_tier, 0)
    n_int = counts.get(int_tier, 0)
    if not records:
        verdict = "NO_DATA_REACHED"
    elif n_cand:
        verdict = "OPAQUE_LENS_CANDIDATES_PENDING_VET"
    elif n_int:
        verdict = "OCCULTING_FITS_PENDING_VET"
    else:
        verdict = "NO_CANDIDATE"
    cands = [r for r in records if str(r.get("tier")) in (cand_tier, int_tier)]
    return {"verdict": verdict, "tiers": counts, "n_events": len(records),
            "n_lensing": sum(1 for r in records if str(r.get("tier")) != "NOT_LENSING"),
            "candidates": cands}


def _channel_assess(channel: str, records: list[dict], conf: dict) -> dict:
    if channel == "lens":
        return _lens_assess(records, conf)
    if channel == "flash":
        from .flash import assess_candidates
        flat = [c for r in records for c in (r.get("candidates") or [])]
        out = assess_candidates(flat, conf)
        tiers = set(((conf.get("flash") or {}).get("tiers") or {}).values())
        out.setdefault("candidates", [r for r in (out.get("records") or []) if str(r.get("tier")) in tiers])
        out.setdefault("n_cutouts", len(records))
        out.setdefault("labels", _sum_dicts(r.get("labels") for r in records))
        return out
    if channel == "lines":
        from .lines import assess_features
        out = assess_features(records, conf)
        out.setdefault("verdict", out.get("tier", "NO_DATA_REACHED"))
        return out
    if channel == "statite":
        from .statite import assess_sources
        return assess_sources(records, conf)
    if channel == "paces":
        from .bridge import assess_paces
        return assess_paces(records, conf)
    raise ValueError(channel)


def _sum_dicts(dicts) -> dict:
    out: dict[str, int] = {}
    for d in dicts:
        for k, v in (d or {}).items():
            out[k] = out.get(k, 0) + int(v)
    return out


def assess(out_dir: Path, conf: dict, channels=CHANNELS) -> dict:
    """Cross-object vetoes per channel, then the verdict rules of the module docstring."""
    probe_rec = read_json(out_dir / "probe.json") or {}
    data_state = probe_rec.get("data_state")
    top = {"stage": "assess", "written_utc": utc_now(), "data_state": data_state,
           "channels": {}, "n_products_analysed": 0, "simulated_inputs": None,
           "config_status": conf.get("config_status")}
    any_sim: bool | None = None
    for ch in channels:
        files = sorted(glob.glob(str(out_dir / "checkpoints" / ch / "*.json")))
        records = [r for r in (read_json(f) for f in files) if isinstance(r, dict)]
        ok = [r for r in records if r.get("status") != "error"]
        errors = [r for r in records if r.get("status") == "error"]
        sim_flags = {bool(r.get("simulated")) for r in records}
        ch_sim = (None if not records else (True if sim_flags == {True} else False if sim_flags == {False} else True))
        try:
            res = _channel_assess(ch, ok, conf) if ok else {"verdict": "NO_DATA_REACHED"}
        except Exception as exc:  # noqa: BLE001
            res = {"verdict": "DEGRADED_SOURCE", "assess_error": f"{type(exc).__name__}: {exc}"}
        verdict = str(res.get("verdict") or "NO_DATA_REACHED")
        if records and errors and len(errors) >= max(1, len(records) // 2):
            verdict = "DEGRADED_SOURCE"
        if ch_sim:
            # A simulated run tests the pipeline, never the sky.
            flowed = bool(ok) and not errors
            verdict = "SIMULATION_PACES_OK" if flowed else "SIMULATION_PACES_FAILED"
        elif not records and data_state in ("NOT_YET_PUBLIC", "SIMULATIONS_ONLY"):
            verdict = "ROMAN_NOT_YET_PUBLIC"
        summ = {"channel": ch, "written_utc": utc_now(), "verdict": verdict, "data_state": data_state,
                "simulated_inputs": ch_sim, "n_records": len(records), "n_errors": len(errors),
                "funnel": _sum_dicts([(r.get("funnel") or {}).get("counts") for r in ok]),
                "rejections": _sum_dicts([(r.get("funnel") or {}).get("rejections") for r in ok]),
                "assessment": {k: v for k, v in res.items() if k != "candidates"},
                "first_errors": [e.get("error") for e in errors[:5]]}
        write_json(out_dir / ch / "summary.json", summ)
        cands = [] if ch_sim else list(res.get("candidates") or [])
        write_json(out_dir / ch / "candidates.json", {"channel": ch, "written_utc": utc_now(),
                                                      "n": len(cands), "candidates": cands})
        top["channels"][ch] = {"verdict": verdict, "n_records": len(records), "n_errors": len(errors),
                               "simulated_inputs": ch_sim}
        top["n_products_analysed"] += len(ok)
        if ch_sim is not None:
            any_sim = ch_sim if any_sim is None else (any_sim or ch_sim)
    top["simulated_inputs"] = any_sim
    verdicts = [c["verdict"] for c in top["channels"].values()]
    if top["n_products_analysed"] == 0:
        top["verdict"] = ("ROMAN_NOT_YET_PUBLIC" if data_state in ("NOT_YET_PUBLIC", "SIMULATIONS_ONLY")
                          else "NO_DATA_REACHED")
    elif any_sim:
        top["verdict"] = ("SIMULATION_PACES_FAILED" if "SIMULATION_PACES_FAILED" in verdicts
                          else "SIMULATION_PACES_OK")
    elif any("CANDIDATE" in v and "NO_CANDIDATE" not in v for v in verdicts):
        top["verdict"] = "CANDIDATES_PENDING_VET"
    elif all(v in ("NO_CANDIDATE", "ROMAN_NOT_YET_PUBLIC", "NO_DATA_REACHED") for v in verdicts):
        top["verdict"] = "NO_CANDIDATE"
    else:
        top["verdict"] = "MIXED: " + ", ".join(f"{k}={v['verdict']}" for k, v in top["channels"].items())
    write_json(out_dir / "summary.json", top)
    _log(f"assess: {top['verdict']} ({top['n_products_analysed']} products; simulated={any_sim})")
    return top


# ---------------------------------------------------------------------------
# selftest -- the synthetic battery through screen() and assess()
# ---------------------------------------------------------------------------

def selftest(out_dir: Path, conf: dict, seed: int = 7) -> dict:
    """Inject each signature and its dominant confounder; run the real stages.

    Every expectation is asserted and recorded; the stage returns ``status``
    ``PASS`` or ``FAIL`` and the workflow gates on it before touching the
    archive.  Everything here is simulated, so the assess verdicts must come
    out ``SIMULATION_PACES_OK`` and no candidate file may be non-empty: that
    rule is itself one of the checks.
    """
    rng = np.random.default_rng(seed)
    st_dir = out_dir / "selftest_work"
    checks: list[dict] = []

    def check(name: str, ok: bool, detail=None):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        _log(f"  selftest {'PASS' if ok else 'FAIL'}: {name}")

    # --- S40 lens ---------------------------------------------------------------------
    try:
        from .lens import screen_lightcurve, synthesise_event
        occ = synthesise_event(rho_l=0.6, u0=0.2, tE=20.0, seed=seed, star_id="syn_occult")
        pac = synthesise_event(rho_l=None, u0=0.3, tE=15.0, seed=seed + 1, star_id="syn_paczynski")
        r_occ = screen_lightcurve(occ, conf)
        r_pac = screen_lightcurve(pac, conf)
        tiers = (conf.get("lens") or {}).get("tiers") or {}
        check("lens: injected occulting event preferred",
              str(r_occ.get("tier")) in (tiers.get("interest"), tiers.get("candidate")), r_occ.get("tier"))
        check("lens: plain Paczynski not occulting", str(r_pac.get("tier")) == "LENSING_NO_OCCULTATION",
              r_pac.get("tier"))
        lens_objs = [occ, pac]
    except Exception as exc:  # noqa: BLE001
        check("lens: module ran", False, f"{type(exc).__name__}: {exc}")
        lens_objs = []

    # --- S42 flash --------------------------------------------------------------------
    try:
        from . import products as P
        from .flash import screen_dq_cutout, synthesise_dq_image
        flags, _ = P.dq_flags(conf)
        stars = [{"star_id": f"s{i}", "x": float(x), "y": float(y)}
                 for i, (x, y) in enumerate(rng.uniform(20, 236, size=(40, 2)))]
        dq = synthesise_dq_image((256, 256), stars, 1.1, flash_stars=["s0", "s1", "s2"], n_cosmic_rays=150,
                                 n_snowballs=2, rng=rng, flags=flags)
        cut = DQCutout("syn_dq", dq, stars, band="F146", psf_fwhm_px=1.1, mjd=61500.0)
        r = screen_dq_cutout(cut, conf, flags)
        got = {c.get("star_id") for c in r.get("candidates") or []}
        check("flash: the three flash stars are psf_on_star", got == {"s0", "s1", "s2"}, sorted(got))
        flash_objs = [cut]
    except Exception as exc:  # noqa: BLE001
        check("flash: module ran", False, f"{type(exc).__name__}: {exc}")
        flash_objs = []

    # --- S41 lines --------------------------------------------------------------------
    try:
        from .lines import screen_spectrum, synthesise_spectrum
        sp = synthesise_spectrum("G150", conf, snr_continuum=20, line_um=1.55, line_snr=15,
                                 source_id="syn_line", rng=rng)
        sp0 = synthesise_spectrum("G150", conf, snr_continuum=20, source_id="syn_clean", rng=rng)
        r1 = screen_spectrum(sp, conf, neighbours=[], prior_bright_pixels=[])
        r0 = screen_spectrum(sp0, conf, neighbours=[], prior_bright_pixels=[])
        found = [f for f in (r1.get("features") or []) if abs(float(f.get("lambda_um", 0)) - 1.55) < 0.005]
        check("lines: injected 1.55 um line found", bool(found), [f.get("lambda_um") for f in r1.get("features") or []][:5])
        check("lines: industrial flag Er fibre", bool(found) and "Er" in str(found[0].get("industrial_flag")),
              found[0].get("industrial_flag") if found else None)
        check("lines: clean spectrum has no surviving feature",
              not any(f.get("survives") for f in (r0.get("features") or [])), r0.get("n_features"))
        line_objs = [sp, sp0]
    except Exception as exc:  # noqa: BLE001
        check("lines: module ran", False, f"{type(exc).__name__}: {exc}")
        line_objs = []

    # --- S43 statite ------------------------------------------------------------------
    try:
        from .statite import screen_source, synthesise_source
        fixed = synthesise_source("fixed", n_epochs=5, span_yr=1.5, sep_mas=300.0, star_mass=1.0,
                                  distance_pc=10.0, err_mas=2.0, rng=rng)
        kep = synthesise_source("kepler", n_epochs=5, span_yr=1.5, sep_mas=300.0, star_mass=1.0,
                                distance_pc=10.0, err_mas=2.0, rng=rng)
        rf, rk = screen_source(fixed, conf), screen_source(kep, conf)
        vf = (rf.get("astrometry") or {}).get("verdict")
        vk = (rk.get("astrometry") or {}).get("verdict")
        check("statite: fixed source flagged non-Keplerian", vf == "FIXED_PREFERRED", vf)
        check("statite: Keplerian source not flagged", vk != "FIXED_PREFERRED", vk)
        stat_objs = [fixed, kep]
    except Exception as exc:  # noqa: BLE001
        check("statite: module ran", False, f"{type(exc).__name__}: {exc}")
        stat_objs = []

    # --- paces ------------------------------------------------------------------------
    try:
        from .bridge import pace_lightcurve, synthesise_gbtds_lightcurve
        dip = synthesise_gbtds_lightcurve("syn_dip", rng=rng, inject={"kind": "dip", "depth": 0.2, "t": 61530.0, "dur_d": 2.0})
        flat = synthesise_gbtds_lightcurve("syn_flat", rng=rng)
        rd, rf2 = pace_lightcurve(dip, conf), pace_lightcurve(flat, conf)
        check("paces: injected dip flagged by dips", "dips" in (rd.get("flags") or []), rd.get("flags"))
        check("paces: constant star raises no flag", not (rf2.get("flags") or []), rf2.get("flags"))
        pace_objs = [dip, flat]
    except Exception as exc:  # noqa: BLE001
        check("paces: module ran", False, f"{type(exc).__name__}: {exc}")
        pace_objs = []

    # --- the same objects through screen() and assess(), as simulated inputs ---------
    try:
        import shutil
        if st_dir.exists():
            shutil.rmtree(st_dir)
        st_dir.mkdir(parents=True)
        (st_dir / "probe.json").write_text(json.dumps({"data_state": "SIMULATIONS_ONLY"}))
        for ch, objs in (("lens", lens_objs), ("flash", flash_objs), ("lines", line_objs),
                         ("statite", stat_objs), ("paces", pace_objs)):
            if objs:
                screen(st_dir, conf, ch, objects=objs, simulated=True)
        top = assess(st_dir, conf)
        check("stages: simulated run reports SIMULATION_PACES_OK", top.get("verdict") == "SIMULATION_PACES_OK",
              top.get("verdict"))
        n_cand = sum(int((read_json(st_dir / ch / "candidates.json") or {}).get("n", 0)) for ch in CHANNELS)
        check("stages: no candidate tier is written on simulated inputs", n_cand == 0, n_cand)
    except Exception as exc:  # noqa: BLE001
        check("stages: screen/assess ran on the synthetic objects", False, f"{type(exc).__name__}: {exc}")

    status = "PASS" if all(c["ok"] for c in checks) else "FAIL"
    rec = {"stage": "selftest", "written_utc": utc_now(), "status": status, "seed": seed,
           "n_checks": len(checks), "n_failed": sum(1 for c in checks if not c["ok"]), "checks": checks}
    write_json(out_dir / "selftest.json", rec)
    _log(f"selftest: {status} ({rec['n_failed']} of {len(checks)} failed)")
    return rec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("stage", choices=["probe", "inventory", "ingest", "screen", "assess", "selftest",
                                     "readiness"])
    p.add_argument("--out-dir", default=None, help="default results/roman")
    p.add_argument("--cache-dir", default=None, help="default data/cache/roman")
    p.add_argument("--channel", default="paces", choices=list(CHANNELS), help="screen: which channel")
    p.add_argument("--channels", default=",".join(CHANNELS), help="assess: comma-separated channels")
    p.add_argument("--kinds", default=",".join(KINDS), help="ingest: comma-separated product kinds")
    p.add_argument("--limit", type=int, default=200, help="ingest: products per kind")
    p.add_argument("--max-objects-per-product", type=int, default=None,
                   help="ingest: objects normalised per multi-object product "
                        "(default archive.max_objects_per_product)")
    p.add_argument("--n-shards", type=int, default=4)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)


def run_stage(args) -> dict:
    conf = load_roman_config()
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if args.stage == "probe":
        return probe(out_dir, conf)
    if args.stage == "inventory":
        return inventory(out_dir, conf, n_shards=args.n_shards)
    if args.stage == "readiness":
        return readiness(out_dir, conf)
    if args.stage == "ingest":
        kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
        return ingest(out_dir, conf, kinds=kinds, limit=args.limit, cache_dir=cache_dir,
                      max_objects_per_product=args.max_objects_per_product)
    if args.stage == "screen":
        return screen(out_dir, conf, args.channel, shard=args.shard, n_shards=args.n_shards,
                      cache_dir=cache_dir)
    if args.stage == "assess":
        chans = tuple(c.strip() for c in args.channels.split(",") if c.strip())
        return assess(out_dir, conf, channels=chans)
    return selftest(out_dir, conf, seed=args.seed)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="seti.roman", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_args(p)
    args = p.parse_args(argv)
    rec = run_stage(args)
    if args.stage == "selftest" and rec.get("status") != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
