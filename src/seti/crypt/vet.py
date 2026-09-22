"""The CRYPT kill ledger applied to every surviving pixel, plus the external
PSR raster route.

* ``load_external_psr``: try each configured LOLA / PGDA route for a raster
  of permanently shadowed pixels on (or re-sampled onto) the Diviner grid;
  every route's outcome is recorded.  When none serves, the thermal cold-trap
  definition stands alone and the summary says so.
* ``vet_candidates``: for each candidate — the radar reading at its position
  (CPR value and neighbourhood from the Mini-RF layer when one was acquired),
  and ODE footprint searches for ShadowCam, LROC NAC and Mini-RF products
  covering it (product ids and counts; ``NO_DATA_REACHED`` when ODE does not
  answer).  Nothing here changes a class: the vet *adds* evidence and the
  summary reports what is still unexcluded.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import acquire as A
from . import radar as R
from .labels import Georef, read_label, read_raster


def _same_grid(a: Georef, b: Georef) -> bool:
    return (a.projection == b.projection and a.lines == b.lines and a.samples == b.samples
            and abs(a.map_scale_m - b.map_scale_m) < 1e-3 and a.center_lat == b.center_lat)


def load_external_psr(conf: dict, fetch, dest_dir: Path, target: Georef, pole: str) -> dict:
    """Try the PSR routes; return ``{"mask": ndarray|None, "routes": [...]}``."""
    acq = conf.get("acquire", {})
    routes = []
    mask = None
    rx = re.compile(str(acq.get("psr_product_regex", r"(?i)(psr|shadow|illum)")))
    pole_rx = re.compile(A.DEFAULT_PATTERNS["pole"][pole])
    for url in acq.get("psr_routes", []):
        rec = {"url": url, "outcome": "NOT_REACHED", "products": []}
        res = fetch(url, timeout=float(acq.get("listing_timeout_s", 60)), max_bytes=20_000_000)
        rec.update({k: v for k, v in res.as_dict().items() if k in ("status", "error", "n_bytes")})
        if not res.ok:
            routes.append(rec)
            continue
        html = res.content.decode("utf-8", errors="replace")
        entries = A.parse_listing(html, url)
        if not entries:
            # a landing page: keep any raster-looking links at all
            for m in re.finditer(r'href="([^"]+\.(?:img|lbl|xml|tif|tiff|zip))"', html, flags=re.I):
                entries.append(A.Entry(name=m.group(1).rsplit("/", 1)[-1],
                                       url=A.urljoin(url, m.group(1)), is_dir=False))
        cands = [e for e in entries if not e.is_dir and rx.search(e.name) and pole_rx.search(e.name)]
        rec["outcome"] = "REACHED_NO_PRODUCT" if not cands else "REACHED_WITH_PRODUCT"
        rec["n_entries"] = len(entries)
        rec["products"] = [e.as_dict() for e in cands[:20]]
        groups = A.pair_products(cands)
        for stem, g in groups.items():
            if g["image"] is None or g["label"] is None:
                continue
            if g["image"].size and g["image"].size > int(acq.get("max_bytes_per_file", 4e8)):
                continue
            try:
                lp = dest_dir / "psr" / g["label"].name
                ip = dest_dir / "psr" / g["image"].name
                r1 = A.download(fetch, g["label"].url, lp, max_bytes=int(acq.get("label_max_bytes", 4e5)))
                r2 = A.download(fetch, g["image"].url, ip, max_bytes=int(acq.get("max_bytes_per_file", 4e8)),
                                expected_size=g["image"].size)
                if not (r1.ok and r2.ok):
                    rec.setdefault("errors", []).append({"stem": stem, "label": r1.error, "image": r2.error})
                    continue
                meta = read_label(lp)
                img = read_raster(meta)
                m = np.isfinite(img) & (img > 0)
                if _same_grid(meta.georef, target):
                    mask = m
                    rec["used"] = {"stem": stem, "resampled": False, "n_true": int(m.sum())}
                elif meta.georef.projection == "polar_stereographic" and target.projection == "polar_stereographic":
                    mask = R.resample_mask(m, meta.georef, target)
                    rec["used"] = {"stem": stem, "resampled": True, "n_true": int(mask.sum()),
                                   "source_georef": meta.georef.as_dict()}
                else:
                    rec.setdefault("errors", []).append({"stem": stem, "error": "grid not usable",
                                                         "georef": meta.georef.as_dict()})
                    continue
                break
            except Exception as exc:  # noqa: BLE001
                rec.setdefault("errors", []).append({"stem": stem, "error": f"{type(exc).__name__}: {exc}"[:300]})
        routes.append(rec)
        if mask is not None:
            break
    return {"mask": mask, "routes": routes,
            "status": "EXTERNAL_PSR_USED" if mask is not None else "THERMAL_DEFINITION_ONLY"}


def ode_coverage(conf: dict, fetch, lon: float, lat: float, instruments: dict | None = None) -> dict:
    """Footprint searches around one position for every configured instrument."""
    acq = conf.get("acquire", {})
    base = acq.get("ode_rest", "https://oderest.rsl.wustl.edu/live2/")
    half = float(acq.get("ode_footprint_half_deg", 0.02))
    out = {}
    for name, spec in (instruments or acq.get("ode_instruments", {})).items():
        params = A.ode_footprint_params(spec["ihid"], spec["iid"], spec.get("pt"), lon, lat,
                                        half_deg_lat=half, limit=int(acq.get("ode_limit", 25)))
        rec = A.ode_query(fetch, base, params)
        prods = A.ode_products(rec)
        ids = []
        for p in prods:
            pid = p.get("pdsid") or p.get("PDSID") or p.get("ode_id") or p.get("ODE_ID") or p.get("LabelFileName")
            if pid:
                ids.append(str(pid))
        out[name] = {"status": "OK" if rec.get("json") is not None else "NO_DATA_REACHED",
                     "http": rec.get("status"), "error": rec.get("error"), "count": A.ode_count(rec),
                     "n_listed": len(prods), "product_ids": ids[:25], "url": rec.get("url"),
                     "head": rec.get("head")}
    return out


def vet_candidates(cands: list[dict], conf: dict, fetch, *, radar_layers: dict | None = None) -> list[dict]:
    """Attach the radar reading and the optical/radar coverage to each candidate."""
    out = []
    vconf = conf.get("vet", {})
    cap = int(vconf.get("max_candidates_vet", 200))
    for n, c in enumerate(cands):
        row = dict(c)
        pole = c.get("pole")
        rl = (radar_layers or {}).get(pole)
        already = isinstance(c.get("radar"), dict) and c["radar"].get("in_raster")
        if rl is not None and rl.get("cpr") is not None and not already:
            row["radar"] = R.sample_at(rl["cpr"], rl["georef"], c["lon"], c["lat"])
            if rl.get("s1") is not None:
                row["radar_s1"] = R.sample_at(rl["s1"], rl["georef"], c["lon"], c["lat"])
        elif not already:
            row["radar"] = {"status": "NO_RADAR_LAYER"}
        if n < cap and fetch is not None:
            row["coverage"] = ode_coverage(conf, fetch, c["lon"], c["lat"])
        else:
            row["coverage"] = {"status": "NOT_QUERIED"}
        row["unexcluded_systematics"] = unexcluded(row)
        out.append(row)
    return out


def unexcluded(row: dict) -> list[str]:
    """What the record has NOT yet ruled out for a surviving pixel."""
    items = []
    rad = row.get("radar") or {}
    if not rad.get("in_raster"):
        items.append("radar: no Mini-RF reading at the position (CPR unknown)")
    elif rad.get("value") is None:
        items.append("radar: Mini-RF pixel is no-data")
    cov = row.get("coverage") or {}
    sc = cov.get("shadowcam") or {}
    if sc.get("status") != "OK":
        items.append("optical: ShadowCam coverage not established (ODE not answering)")
    elif not sc.get("n_listed") and not sc.get("count"):
        items.append("optical: no ShadowCam product covers the position")
    else:
        items.append("optical: ShadowCam frames exist but have not been inspected")
    items.append("geolocation on steep walls inside the 240 m footprint")
    items.append("Diviner channel-6/7 calibration at the noise floor (needs the per-orbit RDR stack)")
    return items


__all__ = ["load_external_psr", "ode_coverage", "unexcluded", "vet_candidates"]
