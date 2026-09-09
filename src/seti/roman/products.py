"""Readers: Roman products -> the four structures in :mod:`seti.roman.schema`.

This is the only module allowed to know what a Roman file looks like.  Level 1
``_uncal.asdf`` ramps, Level 2 ``_cal.asdf`` exposures (``dq`` read alone by
range request), and Level 4 light-curve / spectrum tables in Parquet, CSV,
ECSV, FITS or ASDF all come out as :class:`LightCurve`, :class:`Spectrum`,
:class:`DQCutout` or :class:`Ramp`.

Every optional dependency (``asdf``, ``roman_datamodels``, ``gwcs``,
``fsspec``, ``astropy``, ``pandas``, ``pyarrow``) is imported inside the
reader that needs it.  A missing package, an unreadable file or a table that
lacks a role the structure requires yields a :class:`ReaderUnavailable`
record naming the reason and what would fix it -- never an exception and
never a fabricated field.  The funnel then records the dependent test as
*not run* (the TOCSIN lesson).

Column roles (``time``, ``flux``, ``band`` ...) are resolved at run time from
a candidate list per role, overridable by ``colmap``; the unit and zero point
come from the table metadata when present, else from ``config/roman.yaml``,
and ``meta["zp_source"]`` says which.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .schema import DQ_FLAGS_FALLBACK, DQCutout, LightCurve, Ramp, Spectrum, json_safe

MAG_ZP_DERIVED = 25.0
_ANGSTROM_PER_UM = 1.0e4
_NM_PER_UM = 1.0e3
_JD_MJD_OFFSET = 2400000.5
# The "HJD - 2450000" convention of microlensing tables (OGLE, MOA, the Roman
# Microlensing Data Challenge notebooks): values in this range under a JD-like
# column name are reduced Julian dates.
_REDUCED_JD_OFFSET = 2450000.0
_REDUCED_JD_RANGE = (4000.0, 9999.0)
# Band names as simulations / challenges write them -> WFI filter names; the config's
# ``instruments.WFI.band_aliases`` overrides this table at read time.
WFI_BAND_ALIASES = {"W149": "F146", "W146": "F146", "R062": "F062", "Z087": "F087", "Y106": "F106",
                    "J129": "F129", "H158": "F158", "K213": "F213", "R": "F062", "Z": "F087",
                    "Y": "F106", "J": "F129", "W": "F146", "H": "F158", "F": "F184", "K": "F213"}


@dataclass
class ReaderUnavailable:
    """Why a reader could not produce its structure (missing package, bad file,
    absent role).  ``needs`` lists what would make it possible."""

    reason: str
    uri: str = ""
    needs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"reader_unavailable": True, "reason": self.reason, "uri": self.uri,
                "needs": list(self.needs)}


# --------------------------------------------------------------------------------------
# dq flag table
# --------------------------------------------------------------------------------------

def dq_flags(conf: dict | None = None) -> tuple[dict[str, int], str]:
    """The WFI pixel dq bit table and where it came from.

    ``roman_datamodels.dqflags.pixel`` when importable (an IntFlag or a dict
    depending on the release), else the fallback table overlaid with the config's
    ``dq_flags`` block -- the summary records which, because the two may drift.
    """
    try:
        from roman_datamodels import dqflags
        pix = getattr(dqflags, "pixel", None)
        table: dict[str, int] = {}
        if pix is not None:
            members = getattr(pix, "__members__", None)
            if members:
                table = {str(k): int(v) for k, v in members.items()}
            elif isinstance(pix, dict):
                table = {str(k): int(v) for k, v in pix.items()}
        if table:
            return table, "roman_datamodels.dqflags"
    except Exception:  # noqa: BLE001
        pass
    table = dict(DQ_FLAGS_FALLBACK)
    for k, v in ((conf or {}).get("dq_flags") or {}).items():
        try:
            table[str(k)] = int(v)
        except Exception:  # noqa: BLE001
            continue
    return table, "config/roman.yaml"


def lightcurve_dq_mask(conf: dict, table: dict[str, int]) -> int:
    """OR of the bits named in ``lightcurve_dq_reject``; names absent from the
    table are skipped (they cannot be rejected on, and the caller's table source
    says why)."""
    mask = 0
    for name in (conf or {}).get("lightcurve_dq_reject") or []:
        v = table.get(str(name))
        if v is not None:
            mask |= int(v)
    return int(mask)


# --------------------------------------------------------------------------------------
# ASDF access
# --------------------------------------------------------------------------------------

def _is_remote(uri: str) -> bool:
    return str(uri).startswith(("http://", "https://", "s3://"))


def _open_asdf(uri, lazy: bool = True, fs=None):
    """(asdf_file, tree) for a path, URL or ``s3://`` URI; the caller closes it.

    Remote URIs go through ``fsspec`` (anonymous S3) so that lazy block loading
    becomes HTTP range reads and only the requested arrays travel.
    """
    import asdf
    kw = {"lazy_load": bool(lazy)}
    try:
        import inspect
        params = inspect.signature(asdf.open).parameters
        if "memmap" in params:
            kw["memmap"] = False
        elif "copy_arrays" in params:
            kw["copy_arrays"] = True
    except Exception:  # noqa: BLE001
        pass
    if _is_remote(str(uri)):
        if fs is None:
            import fsspec
            proto = str(uri).split("://", 1)[0]
            fs = fsspec.filesystem(proto, anon=True) if proto == "s3" else fsspec.filesystem(proto)
        fh = fs.open(str(uri), "rb")
        af = asdf.open(fh, **kw)
    else:
        af = asdf.open(str(uri), **kw)
    return af, af.tree


def _lookup(tree, key: str):
    """``key`` (dotted allowed) under ``tree["roman"]`` first, then the root."""
    for base in (tree.get("roman") if isinstance(tree, dict) else None, tree):
        if not isinstance(base, dict):
            continue
        node = base
        ok = True
        for part in str(key).split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif hasattr(node, part) and not isinstance(node, (str, bytes)):
                node = getattr(node, part)
            else:
                ok = False
                break
        if ok:
            return node
    return None


def flatten_meta(node, prefix: str = "", depth: int = 0, max_depth: int = 6) -> dict:
    """A JSON-safe ``{"a.b.c": value}`` view of a nested metadata mapping.

    Arrays and unknown objects become their string form (an ASDF ``Time`` or a
    WCS must not end up in a summary as an object); depth is capped so a
    self-referential tree cannot recurse forever.
    """
    out: dict = {}
    if depth > max_depth:
        return out
    items = None
    if isinstance(node, dict):
        items = node.items()
    elif hasattr(node, "items") and callable(node.items):
        try:
            items = list(node.items())
        except Exception:  # noqa: BLE001
            items = None
    if items is None:
        out[prefix or "value"] = _json_scalar(node)
        return out
    for k, v in items:
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) or (hasattr(v, "items") and not isinstance(v, (str, bytes))
                                   and not hasattr(v, "shape")):
            out.update(flatten_meta(v, key, depth + 1, max_depth))
        else:
            out[key] = _json_scalar(v)
    return out


def _json_scalar(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return json_safe(v)
    if isinstance(v, (np.generic,)):
        return json_safe(v)
    if isinstance(v, (list, tuple)):
        try:
            return [_json_scalar(x) for x in v][:64]
        except Exception:  # noqa: BLE001
            return str(v)[:200]
    if hasattr(v, "shape"):
        try:
            shp = tuple(int(s) for s in v.shape)
        except Exception:  # noqa: BLE001
            shp = None
        return f"<array shape={shp}>"
    return str(v)[:200]


def read_asdf_arrays(uri, keys=("dq",), lazy: bool = True, fs=None) -> dict | ReaderUnavailable:
    """Requested arrays (numpy) and the flattened ``meta`` from one ASDF product.

    Keys are searched under ``tree["roman"]`` then the root; a key that is
    absent is listed in ``missing`` rather than invented.  Degrades to
    :class:`ReaderUnavailable` when ``asdf`` (or ``fsspec`` for a remote URI) is
    not importable or the file cannot be opened.
    """
    try:
        import asdf  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return ReaderUnavailable(f"asdf not importable: {exc!r}", str(uri), ["asdf"])
    if _is_remote(str(uri)) and fs is None:
        try:
            import fsspec  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            needs = ["fsspec"] + (["s3fs"] if str(uri).startswith("s3://") else [])
            return ReaderUnavailable(f"fsspec not importable: {exc!r}", str(uri), needs)
    try:
        af, tree = _open_asdf(uri, lazy=lazy, fs=fs)
    except Exception as exc:  # noqa: BLE001
        return ReaderUnavailable(f"asdf open failed: {exc!r}", str(uri), [])
    out: dict = {"uri": str(uri), "missing": [], "meta": {}}
    try:
        for key in keys:
            node = _lookup(tree, key)
            if node is None:
                out["missing"].append(str(key))
                continue
            try:
                out[str(key)] = np.asarray(node)
            except Exception as exc:  # noqa: BLE001
                out["missing"].append(str(key))
                out.setdefault("errors", {})[str(key)] = repr(exc)[:200]
        meta = _lookup(tree, "meta")
        if meta is not None:
            out["meta"] = flatten_meta(meta)
    finally:
        try:
            af.close()
        except Exception:  # noqa: BLE001
            pass
    return out


# --------------------------------------------------------------------------------------
# dq cutouts and pixel coordinates
# --------------------------------------------------------------------------------------

def stars_to_pixels(wcs_like, ra, dec) -> tuple[np.ndarray, np.ndarray]:
    """Detector pixel coordinates from sky coordinates through a gwcs / astropy
    WCS (``world_to_pixel_values``) or any callable ``(ra, dec) -> (x, y)``."""
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    fn = getattr(wcs_like, "world_to_pixel_values", None)
    if callable(fn):
        x, y = fn(ra, dec)
    elif callable(wcs_like):
        x, y = wcs_like(ra, dec)
    else:
        raise TypeError("wcs_like needs world_to_pixel_values or must be callable")
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def dq_cutouts_from_image(dq: np.ndarray, stars: list[dict], image_id: str, box: int = 32,
                          margin: int = 4, group: bool = True, **kw) -> list[DQCutout]:
    """Cut a ``box``-sized dq window around each catalogued star.

    Stars whose pixel lies inside an earlier cutout by at least ``margin``
    pixels join that cutout (so a dense bulge field does not produce one
    cutout per star); every star is copied with ``x``/``y`` shifted into the
    cutout frame and ``x_det``/``y_det`` kept.  Stars off the detector are
    skipped and counted in ``meta["n_off_image"]``.  Extra keywords
    (``detector``, ``band``, ``mjd``, ``psf_fwhm_px`` ...) pass to
    :class:`DQCutout`.
    """
    dq = np.asarray(dq)
    if dq.ndim != 2:
        raise ValueError("dq must be 2-D")
    ny, nx = dq.shape
    bw, bh = min(int(box), nx), min(int(box), ny)
    meta_extra = dict(kw.pop("meta", {}) or {})
    valid, n_off = [], 0
    for s in stars:
        try:
            x, y = float(s["x"]), float(s["y"])
        except Exception:  # noqa: BLE001
            n_off += 1
            continue
        if not (math.isfinite(x) and math.isfinite(y)) or x < -0.5 or y < -0.5 \
                or x > nx - 0.5 or y > ny - 0.5:
            n_off += 1
            continue
        valid.append((x, y, s))
    assigned = [False] * len(valid)
    cutouts: list[DQCutout] = []
    for i, (x, y, _s) in enumerate(valid):
        if assigned[i]:
            continue
        x0 = int(min(max(int(round(x)) - bw // 2, 0), nx - bw))
        y0 = int(min(max(int(round(y)) - bh // 2, 0), ny - bh))
        members = []
        for j, (xj, yj, sj) in enumerate(valid):
            if assigned[j]:
                continue
            inside = (x0 + margin <= xj <= x0 + bw - 1 - margin
                      and y0 + margin <= yj <= y0 + bh - 1 - margin)
            if j == i or (group and inside):
                assigned[j] = True
                row = dict(sj)
                row["x_det"], row["y_det"] = xj, yj
                row["x"], row["y"] = xj - x0, yj - y0
                row.setdefault("star_id", f"star_{j}")
                members.append(row)
        meta = dict(meta_extra)
        meta.update(n_stars=len(members), star_ids=[m["star_id"] for m in members],
                    n_off_image=n_off)
        cutouts.append(DQCutout(image_id=str(image_id), dq=dq[y0:y0 + bh, x0:x0 + bw],
                                stars=members, x0=x0, y0=y0, meta=meta, **kw))
    return cutouts


# --------------------------------------------------------------------------------------
# Level 1 ramps
# --------------------------------------------------------------------------------------

def resultant_mid_times(read_pattern, frame_time_s: float) -> np.ndarray:
    """Mid-time of each resultant: the mean frame index of its group x frame time."""
    return np.array([float(np.mean([float(f) for f in grp])) * float(frame_time_s)
                     for grp in read_pattern], dtype=float)


def _read_pattern_list(rp) -> list[list[int]] | None:
    if rp is None:
        return None
    try:
        out = [[int(f) for f in np.atleast_1d(np.asarray(g)).tolist()] for g in rp]
    except Exception:  # noqa: BLE001
        return None
    return out if out and all(len(g) for g in out) else None


def read_ramp(uri_or_tree, box=None, conf: dict | None = None) -> Ramp | ReaderUnavailable:
    """A Level 1 up-the-ramp cube for the pixel box ``(x0, x1, y0, y1)``.

    From an ``_uncal.asdf`` (``data`` cube; ``meta.exposure.read_pattern`` and
    ``meta.exposure.frame_time``, the latter falling back to the config's
    ``frame_time_s``) or from an in-memory dict ``{"data", "read_pattern",
    "frame_time"}``.  With lazy loading only the box is materialised.  A cube
    without a read pattern has no time axis and is reported, not guessed.
    """
    wfi = ((conf or {}).get("instruments") or {}).get("WFI") or {}
    conf_ft = wfi.get("frame_time_s")
    gain = wfi.get("gain_e_per_dn")
    af = None
    try:
        if isinstance(uri_or_tree, dict) and "data" in uri_or_tree:
            tree = uri_or_tree
            data = tree["data"]
            rp = tree.get("read_pattern")
            ft = tree.get("frame_time", conf_ft)
            image_id = str(tree.get("image_id") or "memory")
            meta = {k: _json_scalar(v) for k, v in tree.items()
                    if k not in ("data", "dq", "read_pattern")}
            dq = tree.get("dq")
        else:
            uri = str(uri_or_tree)
            try:
                import asdf  # noqa: F401
            except Exception as exc:  # noqa: BLE001
                return ReaderUnavailable(f"asdf not importable: {exc!r}", uri, ["asdf"])
            try:
                af, tree = _open_asdf(uri, lazy=True)
            except Exception as exc:  # noqa: BLE001
                return ReaderUnavailable(f"asdf open failed: {exc!r}", uri, [])
            data = _lookup(tree, "data")
            if data is None:
                return ReaderUnavailable("no data cube in file", uri, [])
            rp = _lookup(tree, "meta.exposure.read_pattern")
            if rp is None:
                rp = _lookup(tree, "read_pattern")
            ft = _lookup(tree, "meta.exposure.frame_time")
            if ft is None:
                ft = conf_ft
            m = _lookup(tree, "meta")
            meta = flatten_meta(m) if m is not None else {}
            image_id = str(meta.get("filename") or Path(uri).name)
            dq = _lookup(tree, "dq")
        if box is not None:
            x0, x1, y0, y1 = (int(v) for v in box)
        else:
            x0, y0 = 0, 0
            x1, y1 = int(data.shape[2]), int(data.shape[1])
        cube = np.asarray(data[:, y0:y1, x0:x1], dtype=float)
        if cube.ndim != 3:
            return ReaderUnavailable(f"data is not a 3-D cube (shape {cube.shape})",
                                     str(uri_or_tree)[:80], [])
        pattern = _read_pattern_list(rp)
        if pattern is None:
            return ReaderUnavailable("no read_pattern: resultant times unknown",
                                     image_id, ["meta.exposure.read_pattern"])
        if ft is None:
            return ReaderUnavailable("no frame_time (file or config)", image_id,
                                     ["meta.exposure.frame_time"])
        if len(pattern) != cube.shape[0]:
            return ReaderUnavailable(f"read_pattern has {len(pattern)} groups but the cube "
                                     f"has {cube.shape[0]} resultants", image_id, [])
        dq_cube = None
        if dq is not None:
            try:
                dq_cube = np.asarray(dq[:, y0:y1, x0:x1])
                if dq_cube.shape != cube.shape:
                    dq_cube = None
            except Exception:  # noqa: BLE001
                dq_cube = None
        meta = dict(meta)
        meta["frame_time_source"] = "file" if (isinstance(uri_or_tree, dict) and
                                               "frame_time" in uri_or_tree) or \
            (not isinstance(uri_or_tree, dict) and _lookup(tree, "meta.exposure.frame_time")
             is not None) else "config/roman.yaml"
        return Ramp(image_id=image_id, resultants=cube,
                    times_s=resultant_mid_times(pattern, float(ft)), x0=x0, y0=y0,
                    read_pattern=pattern, frame_time_s=float(ft), unit="DN",
                    gain_e_per_dn=None if gain is None else float(gain), dq=dq_cube, meta=meta)
    except Exception as exc:  # noqa: BLE001
        return ReaderUnavailable(f"ramp read failed: {exc!r}", str(uri_or_tree)[:80], [])
    finally:
        if af is not None:
            try:
                af.close()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------

LC_ROLE_CANDIDATES: dict[str, list[str]] = {
    "star_id": ["star_id", "source_id", "id", "objid", "object_id", "sourceid", "star",
                "gaia_id", "roman_id", "lc_id", "name", "event_id", "event", "event_name"],
    "time": ["mjd", "MJD", "time", "t", "bjd", "BJD_TDB", "bjd_tdb", "jd", "JD", "hjd", "HJD",
             "obsmjd", "mjd_obs", "epoch", "mjd_tdb", "tdb"],
    "flux": ["flux", "f", "flux_e_s", "psf_flux", "aperture_flux", "FLUX", "counts", "rate"],
    "flux_err": ["flux_err", "ferr", "e_flux", "fluxerr", "psf_flux_err", "flux_error",
                 "err", "sigma", "aperture_flux_err", "FLUX_ERR", "counts_err", "rate_err"],
    "band": ["band", "filter", "filt", "optical_element", "bandpass"],
    "dq": ["dq", "flag", "flags", "quality", "qual"],
    "ra": ["ra", "RA", "ra_deg", "right_ascension"],
    "dec": ["dec", "DEC", "dec_deg", "declination"],
    "mag": ["mag", "magnitude", "MAG", "psf_mag", "aperture_mag", "m"],
    "mag_err": ["mag_err", "magerr", "e_mag", "mag_error", "psf_mag_err", "MAGERR", "merr"],
    "exptime": ["exptime", "exposure_time", "exposure", "texp"],
}

SPEC_ROLE_CANDIDATES: dict[str, list[str]] = {
    "source_id": ["source_id", "id", "objid", "object_id", "star_id", "name"],
    "wavelength": ["wave", "wavelength", "lambda", "wl", "lam", "wave_um", "WAVELENGTH",
                   "wavelength_um", "wave_angstrom"],
    "flux": ["flux", "f", "FLUX", "flux_density", "fnu", "flam"],
    "flux_err": ["flux_err", "ferr", "e_flux", "fluxerr", "flux_error", "err", "error",
                 "sigma", "FLUX_ERROR", "ivar"],
    "dq": ["dq", "flag", "flags", "quality", "mask"],
    "contam": ["contam", "contamination", "contam_frac", "contam_flux", "contamination_flux"],
    "ra": ["ra", "RA", "ra_deg"],
    "dec": ["dec", "DEC", "dec_deg"],
}


def resolve_roles(columns, candidates: dict[str, list[str]],
                  colmap: dict | None = None) -> dict[str, str | None]:
    """Column name per role: the ``colmap`` override first, else the first candidate
    present (case-insensitively).  Roles with no match map to ``None``."""
    cols = [str(c) for c in columns]
    low = {c.lower(): c for c in cols}
    out: dict[str, str | None] = {}
    for role, cands in candidates.items():
        chosen = None
        override = (colmap or {}).get(role)
        if override is not None and str(override) in cols:
            chosen = str(override)
        elif override is not None and str(override).lower() in low:
            chosen = low[str(override).lower()]
        else:
            for c in cands:
                if c in cols:
                    chosen = c
                    break
                if c.lower() in low:
                    chosen = low[c.lower()]
                    break
        out[role] = chosen
    return out


def _meta_get(meta: dict, *names, default=None):
    low = {str(k).lower(): v for k, v in (meta or {}).items()}
    for n in names:
        if str(n).lower() in low and low[str(n).lower()] is not None:
            return low[str(n).lower()]
    return default


def load_table(path_or_df, meta: dict | None = None, asdf_key: str | None = None):
    """(DataFrame, meta dict) from a DataFrame, an astropy Table, or a path
    (.parquet / .csv / .ecsv / .fits / .asdf).  Returns :class:`ReaderUnavailable`
    when the reader for that format is missing or the file will not parse."""
    try:
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        return ReaderUnavailable(f"pandas not importable: {exc!r}", "", ["pandas"])
    meta = dict(meta or {})
    if isinstance(path_or_df, pd.DataFrame):
        m = dict(getattr(path_or_df, "attrs", {}) or {})
        m.update(meta)
        return path_or_df, m
    if hasattr(path_or_df, "to_pandas") and hasattr(path_or_df, "colnames"):
        m = dict(getattr(path_or_df, "meta", {}) or {})
        m.update(meta)
        return _table_to_pandas(path_or_df), m
    path = Path(str(path_or_df))
    suf = path.suffix.lower()
    uri = str(path)
    try:
        if suf in (".parquet", ".pq"):
            try:
                import pyarrow.parquet as pq
            except Exception as exc:  # noqa: BLE001
                return ReaderUnavailable(f"pyarrow not importable: {exc!r}", uri, ["pyarrow"])
            t = pq.read_table(str(path))
            m = {}
            for k, v in (t.schema.metadata or {}).items():
                ks = k.decode("utf-8", "replace") if isinstance(k, bytes) else str(k)
                if ks == "pandas":
                    continue
                vs = v.decode("utf-8", "replace") if isinstance(v, bytes) else v
                m[ks] = _coerce_meta_value(vs)
            m.update(meta)
            return t.to_pandas(), m
        if suf in (".csv", ".txt", ".dat"):
            comments = _header_comments(path)
            df, how = _read_delimited(path, pd)
            comments.update(how)
            comments.update(meta)
            return df, comments
        if suf == ".ecsv":
            try:
                from astropy.table import Table
            except Exception as exc:  # noqa: BLE001
                return ReaderUnavailable(f"astropy not importable: {exc!r}", uri, ["astropy"])
            t = Table.read(str(path), format="ascii.ecsv")
            m = dict(t.meta or {})
            m.update(_column_units(t))
            m.update(meta)
            return _table_to_pandas(t), m
        if suf in (".fits", ".fit", ".fz"):
            try:
                from astropy.table import Table
            except Exception as exc:  # noqa: BLE001
                return ReaderUnavailable(f"astropy not importable: {exc!r}", uri, ["astropy"])
            t = Table.read(str(path))
            m = dict(t.meta or {})
            m.update(_column_units(t))
            m.update(meta)
            return _table_to_pandas(t), m
        if suf == ".asdf":
            try:
                import asdf  # noqa: F401
            except Exception as exc:  # noqa: BLE001
                return ReaderUnavailable(f"asdf not importable: {exc!r}", uri, ["asdf"])
            af, tree = _open_asdf(uri, lazy=False)
            try:
                node = _lookup(tree, asdf_key) if asdf_key else _find_table_node(tree)
                if node is None:
                    return ReaderUnavailable("no table-like node in ASDF tree", uri, [])
                if hasattr(node, "to_pandas") and hasattr(node, "colnames"):
                    df = _table_to_pandas(node)
                    m = dict(getattr(node, "meta", {}) or {})
                else:
                    df = pd.DataFrame({str(k): np.asarray(v) for k, v in node.items()
                                       if hasattr(v, "__len__")})
                    m = {}
                mm = _lookup(tree, "meta")
                if mm is not None:
                    m.update(flatten_meta(mm))
                m.update(meta)
                return df, m
            finally:
                af.close()
        return ReaderUnavailable(f"unrecognised table format {suf!r}", uri, [])
    except ReaderUnavailable as ru:  # pragma: no cover - defensive
        return ru
    except Exception as exc:  # noqa: BLE001
        return ReaderUnavailable(f"table read failed: {exc!r}", uri, [])


def _first_lines(path: Path, n: int = 200) -> tuple[list[str], str | None]:
    """The leading ``#`` comment lines and the first non-comment, non-blank line."""
    comments: list[str] = []
    first = None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for _ in range(n):
                line = fh.readline()
                if not line:
                    break
                if line.startswith("#"):
                    comments.append(line.rstrip("\n"))
                    continue
                if line.strip():
                    first = line.rstrip("\n")
                    break
    except Exception:  # noqa: BLE001
        pass
    return comments, first


def _looks_numeric(tok) -> bool:
    try:
        float(str(tok))
        return True
    except Exception:  # noqa: BLE001
        return False


def _read_delimited(path: Path, pd):
    """A ``.csv`` / ``.txt`` / ``.dat`` table with ``#`` comments: comma-separated
    when the first data line carries commas, else whitespace-delimited (the
    ``pd.read_csv(sep=r"\\s+", comment="#")`` form the RMDC26 notebooks use).

    A file without a header row (its first non-comment line carries a number
    -- a header row carries none) takes its column names from the last ``#``
    line when that line has as many tokens as the row (``# HJD mag mag_err
    band``), else ``col0..colN``.  Returns ``(df, {"delimiter", "header_source"})``.
    """
    comments, first = _first_lines(path)
    sep = "," if (first is not None and "," in first) else r"\s+"
    how = {"delimiter": "comma" if sep == "," else "whitespace", "header_source": "first_row"}
    kw = {"comment": "#", "sep": sep} if sep == "," else {"comment": "#", "sep": sep, "engine": "python"}
    first_toks = [t for t in re.split(r"[\s,]+", (first or "").strip()) if t]
    if first_toks and any(_looks_numeric(t) for t in first_toks):
        names = None
        for line in reversed(comments):
            toks = [t for t in re.split(r"[\s,]+", line.lstrip("#").strip()) if t]
            if len(toks) == len(first_toks) and not any(_looks_numeric(t) for t in toks):
                names = toks
                break
        if names is None:
            names = [f"col{i}" for i in range(len(first_toks))]
            how["header_source"] = "none"
        else:
            how["header_source"] = "comment_line"
        return pd.read_csv(path, header=None, names=names, **kw), how
    return pd.read_csv(path, **kw), how


def _coerce_meta_value(v):
    if isinstance(v, str):
        s = v.strip()
        try:
            return float(s) if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s) else s
        except Exception:  # noqa: BLE001
            return s
    return v


def _header_comments(path: Path) -> dict:
    """``# key = value`` / ``# key: value`` lines at the top of a CSV as metadata."""
    out: dict = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for _ in range(200):
                line = fh.readline()
                if not line:
                    break
                if not line.startswith("#"):
                    break
                m = re.match(r"#\s*([A-Za-z_][\w.\-]*)\s*[=:]\s*(.+?)\s*$", line)
                if m:
                    out[m.group(1)] = _coerce_meta_value(m.group(2))
    except Exception:  # noqa: BLE001
        pass
    return out


def _column_units(t) -> dict:
    out = {}
    try:
        for c in t.colnames:
            u = getattr(t[c], "unit", None)
            if u is not None:
                out[f"unit.{c}"] = str(u)
    except Exception:  # noqa: BLE001
        pass
    return out


def _table_to_pandas(t):
    df = t.to_pandas()
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = df[c].map(lambda v: v.decode() if isinstance(v, bytes) else v)
            except Exception:  # noqa: BLE001
                pass
    return df


def _find_table_node(tree):
    for key in ("lightcurve", "lightcurves", "light_curve", "spectrum", "spectra", "table",
                "data", "catalog"):
        node = _lookup(tree, key)
        if node is None:
            continue
        if hasattr(node, "colnames") or (isinstance(node, dict)
                                         and any(hasattr(v, "__len__") for v in node.values())):
            return node
    base = tree.get("roman", tree) if isinstance(tree, dict) else tree
    if isinstance(base, dict):
        for v in base.values():
            if hasattr(v, "colnames"):
                return v
    return None


def _time_system(colname: str | None, meta: dict) -> str:
    c = (colname or "").lower()
    sysm = str(_meta_get(meta, "time_system", "timesys", "timesystem", default="") or "").upper()
    if "bjd" in c or "BJD" in sysm:
        return "BJD_TDB"
    if "hjd" in c:
        return "HJD"
    if "tdb" in c or "TDB" in sysm:
        return "TDB"
    if sysm in ("UTC", "TT", "TAI"):
        return f"MJD_{sysm}"
    return "unknown"


def band_aliases(conf: dict | None = None) -> dict[str, str]:
    """``instruments.WFI.band_aliases`` (upper-cased keys) over :data:`WFI_BAND_ALIASES`."""
    out = {str(k).upper(): str(v).upper() for k, v in WFI_BAND_ALIASES.items()}
    cfg = (((conf or {}).get("instruments") or {}).get("WFI") or {}).get("band_aliases") or {}
    for k, v in cfg.items():
        out[str(k).upper()] = str(v).upper()
    return out


def times_to_mjd(t: np.ndarray, colname: str | None) -> tuple[np.ndarray, str]:
    """Times in MJD and the convention that was assumed.

    * a median above 2.4e6 is a full Julian date whatever the column name
      (``jd_to_mjd``);
    * a JD-like column name (``jd`` / ``hjd`` / ``bjd``, not ``mjd``) whose
      median lies in 4000-9999 is a reduced ``JD - 2450000`` (the microlensing
      convention; ``reduced_jd_to_mjd``);
    * a JD-like name with values already in the MJD range is left as it is
      (``mjd_range_as_is``); anything else is untouched (``as_is``).
    """
    t = np.asarray(t, dtype=float)
    med = float(np.nanmedian(t)) if t.size and np.any(np.isfinite(t)) else float("nan")
    c = (colname or "").lower()
    jd_like = "jd" in c and "mjd" not in c
    if np.isfinite(med) and med > 2.4e6:
        return t - _JD_MJD_OFFSET, "jd_to_mjd"
    if jd_like and np.isfinite(med) and _REDUCED_JD_RANGE[0] <= med <= _REDUCED_JD_RANGE[1]:
        return t + (_REDUCED_JD_OFFSET - _JD_MJD_OFFSET), "reduced_jd_to_mjd"
    if jd_like:
        return t, "mjd_range_as_is"
    return t, "as_is"


def _first_finite(series, default=float("nan")) -> float:
    try:
        arr = np.asarray(series, dtype=float)
        ok = np.isfinite(arr)
        return float(arr[ok][0]) if np.any(ok) else default
    except Exception:  # noqa: BLE001
        return default


def _zero_point(meta: dict, conf: dict | None, band: str, flux_unit: str) -> tuple[float | None, str]:
    zp = _meta_get(meta, "zp", "zeropoint", "ZP_AB", "zp_ab", "magzp", "zero_point")
    if isinstance(zp, dict):
        zp = zp.get(band, zp.get(str(band).upper()))
    if zp is not None:
        try:
            return float(zp), "table_meta"
        except Exception:  # noqa: BLE001
            pass
    if flux_unit.lower().replace(" ", "") in ("e-/s", "e/s", "electron/s", "electrons/s",
                                              "dn/s", "count/s", "counts/s"):
        row = (((conf or {}).get("instruments") or {}).get("WFI") or {}).get("filters") or {}
        r = row.get(str(band).upper())
        if isinstance(r, dict) and r.get("zp_ab") is not None:
            return float(r["zp_ab"]), "config/roman.yaml"
    return None, "none"


def read_lightcurve_table(path_or_df, colmap: dict | None = None, conf: dict | None = None,
                          band: str | None = None, survey: str = "", star_id=None,
                          meta: dict | None = None,
                          max_objects: int | None = None) -> list[LightCurve] | ReaderUnavailable:
    """Level 4 light-curve table(s) -> one :class:`LightCurve` per (star, band).

    Roles are resolved by :func:`resolve_roles`; a table with only magnitudes is
    converted to flux at zero point 25 (``flux_unit="mag_derived"``).  Missing
    ``time`` or (``flux``, ``flux_err``) yields :class:`ReaderUnavailable`: a
    curve without errors cannot be screened and must not be padded.  The zero
    point comes from the table metadata, else the config filter table (e-/s),
    else ``None`` (relative flux), with ``meta["zp_source"]`` saying which.

    Times become MJD by :func:`times_to_mjd` (full JD, reduced ``JD - 2450000``,
    or already MJD) and ``meta["time_convention"]`` records which.  Band values
    are upper-cased and passed through ``instruments.WFI.band_aliases``
    (``W149`` -> ``F146`` ...) before grouping; ``meta["band_raw"]`` keeps the
    token as written and ``meta["band_alias_applied"]`` is ``{raw: alias}`` or
    ``None``.  Curves come out in order of first appearance of their
    (star, band) pair; ``max_objects`` keeps the first that many, and every
    returned curve carries ``meta["capped"]`` (True when curves were dropped),
    ``meta["n_curves_in_table"]`` and ``meta["max_objects"]``.
    """
    loaded = load_table(path_or_df, meta)
    if isinstance(loaded, ReaderUnavailable):
        return loaded
    df, tmeta = loaded
    uri = str(path_or_df) if not hasattr(path_or_df, "columns") else "<dataframe>"
    roles = resolve_roles(df.columns, LC_ROLE_CANDIDATES, colmap)
    if roles["time"] is None:
        return ReaderUnavailable("no time column", uri, ["time"])
    use_mag = roles["flux"] is None
    if use_mag and roles["mag"] is None:
        return ReaderUnavailable("no flux or mag column", uri, ["flux"])
    if use_mag and roles["mag_err"] is None:
        return ReaderUnavailable("mag column without mag_err", uri, ["mag_err"])
    if not use_mag and roles["flux_err"] is None:
        return ReaderUnavailable("flux column without flux_err", uri, ["flux_err"])
    if len(df) == 0:
        return []

    t, time_convention = times_to_mjd(np.asarray(df[roles["time"]], dtype=float), roles["time"])
    time_system = _time_system(roles["time"], tmeta)
    if use_mag:
        mag = np.asarray(df[roles["mag"]], dtype=float)
        merr = np.asarray(df[roles["mag_err"]], dtype=float)
        flux = 10.0 ** (-0.4 * (mag - MAG_ZP_DERIVED))
        ferr = merr * flux * np.log(10.0) / 2.5
        flux_unit, zp, zp_source = "mag_derived", MAG_ZP_DERIVED, "mag_conversion_zp25"
    else:
        flux = np.asarray(df[roles["flux"]], dtype=float)
        ferr = np.asarray(df[roles["flux_err"]], dtype=float)
        flux_unit = str(_meta_get(tmeta, "flux_unit", f"unit.{roles['flux']}", "unit",
                                  "bunit", default="e-/s"))
        zp = zp_source = None
    dq = None
    if roles["dq"] is not None:
        dq = np.nan_to_num(np.asarray(df[roles["dq"]], dtype=float), nan=0.0).astype(np.int64)
    exp_s = None
    if roles["exptime"] is not None:
        exp_s = float(np.nanmedian(np.asarray(df[roles["exptime"]], dtype=float)))
    else:
        e = _meta_get(tmeta, "exposure_time", "exptime", "exposure_s")
        exp_s = float(e) if isinstance(e, (int, float)) else None

    sid_col, band_col = roles["star_id"], roles["band"]
    sid_default = str(star_id) if star_id is not None else \
        str(_meta_get(tmeta, "star_id", "source_id", "id", default="table"))
    band_default = str(band or _meta_get(tmeta, "band", "filter", "optical_element",
                                         default="UNKNOWN")).upper()
    sids = df[sid_col].astype(str).to_numpy() if sid_col else np.full(len(df), sid_default)
    raw_bands = df[band_col].astype(str).str.strip().str.upper().to_numpy() if band_col \
        else np.full(len(df), band_default)
    aliases = band_aliases(conf)
    alias_of = {b: aliases.get(b, b) for b in np.unique(raw_bands).tolist()}
    bands = np.asarray([alias_of[b] for b in raw_bands.tolist()], dtype=object) \
        if alias_of else raw_bands
    surv = str(survey or _meta_get(tmeta, "survey", default=""))
    out: list[LightCurve] = []
    # (star, band) groups in order of first appearance -- the cap is then deterministic
    import pandas as pd
    groups = pd.DataFrame({"s": sids, "b": bands}).groupby(["s", "b"], sort=False).indices
    keys = list(groups.keys())
    n_keys = len(keys)
    capped = max_objects is not None and n_keys > int(max_objects)
    if capped:
        keys = keys[:max(0, int(max_objects))]
    raw_by_key = {}
    for (sid, bnd), idx in groups.items():
        raw_by_key[(sid, bnd)] = str(raw_bands[idx[0]])
    ra_all = df[roles["ra"]].to_numpy() if roles["ra"] else None
    dec_all = df[roles["dec"]].to_numpy() if roles["dec"] else None
    for sid, bnd in keys:
        sel = np.asarray(groups[(sid, bnd)], dtype=int)      # row positions of this curve
        raw = raw_by_key[(sid, bnd)]
        if use_mag:
            zp_b, src_b = zp, zp_source
        else:
            zp_b, src_b = _zero_point(tmeta, conf, bnd, flux_unit)
        ra = _first_finite(ra_all[sel]) if ra_all is not None else \
            float(_meta_get(tmeta, "ra", default=float("nan")) or float("nan"))
        dec = _first_finite(dec_all[sel]) if dec_all is not None else \
            float(_meta_get(tmeta, "dec", default=float("nan")) or float("nan"))
        m = {"zp_source": src_b, "roles": {k: v for k, v in roles.items() if v},
             "time_column": roles["time"], "time_convention": time_convention, "source": uri,
             "n_rows": int(sel.size), "band_raw": raw,
             "band_alias_applied": ({raw: str(bnd)} if raw != str(bnd) else None),
             "capped": bool(capped), "n_curves_in_table": int(n_keys),
             "max_objects": None if max_objects is None else int(max_objects)}
        for k in ("simulated", "pipeline_version", "survey", "field", "tier"):
            v = _meta_get(tmeta, k)
            if v is not None:
                m[k] = json_safe(v)
        order = np.argsort(t[sel], kind="stable")
        lc = LightCurve(star_id=str(sid), ra=ra, dec=dec, band=str(bnd), mjd=t[sel][order],
                        flux=flux[sel][order], flux_err=ferr[sel][order], survey=surv,
                        flux_unit=flux_unit, flux_zp_ab=zp_b, time_system=time_system,
                        dq=None if dq is None else dq[sel][order], exposure_s=exp_s, meta=m)
        out.append(lc)
    return out


# --------------------------------------------------------------------------------------
# Spectra
# --------------------------------------------------------------------------------------

def wavelength_to_um(wl: np.ndarray, unit_hint: str | None = None) -> tuple[np.ndarray, str]:
    """Wavelengths in microns, from a declared unit or from their magnitude.

    0.3-3 -> already um; 300-3000 -> nm; > 100 otherwise -> Angstrom (the Roman
    bands are 7500-19300 A / 750-1930 nm / 0.75-1.93 um, so the three ranges do
    not overlap for any Roman product).  Returns the array and the unit assumed.
    """
    wl = np.asarray(wl, dtype=float)
    u = str(unit_hint or "").strip().lower()
    if u:
        if u in ("um", "micron", "microns", "micrometer", "micrometre", "µm", "μm"):
            return wl, "um"
        if u in ("nm", "nanometer", "nanometre"):
            return wl / _NM_PER_UM, "nm"
        if u in ("a", "aa", "angstrom", "angstroms", "å", "ang"):
            return wl / _ANGSTROM_PER_UM, "angstrom"
    med = float(np.nanmedian(wl)) if wl.size and np.any(np.isfinite(wl)) else float("nan")
    if 0.3 <= med <= 3.0:
        return wl, "um"
    if 300.0 <= med <= 3000.0:
        return wl / _NM_PER_UM, "nm"
    if med > 100.0:
        return wl / _ANGSTROM_PER_UM, "angstrom"
    return wl, "unknown"


def resolving_power_for(conf: dict | None, mode: str | None,
                        wl_um: np.ndarray) -> float | np.ndarray | None:
    """R(lambda) from the config disperser table: ``R_per_um`` x lambda (grism) or a
    linear ramp of ``R_range`` across ``range_um`` (prism).  ``None`` when unknown."""
    if not conf or not mode:
        return None
    disp = (((conf.get("instruments") or {}).get("WFI") or {}).get("dispersers") or {})
    row = disp.get(str(mode).upper())
    if not isinstance(row, dict):
        cgi = (((conf.get("instruments") or {}).get("CGI") or {}).get("bands") or {})
        crow = cgi.get(str(mode).upper().replace("CGI_", ""))
        if isinstance(crow, dict) and crow.get("R") is not None:
            return float(crow["R"])
        return None
    if row.get("R_per_um") is not None:
        return float(row["R_per_um"]) * np.asarray(wl_um, dtype=float)
    if row.get("R_range") is not None and row.get("range_um") is not None:
        lo, hi = (float(v) for v in row["range_um"])
        r0, r1 = (float(v) for v in row["R_range"])
        return np.interp(np.asarray(wl_um, dtype=float), [lo, hi], [r0, r1])
    return None


def _point_source_from_meta(meta: dict) -> bool | None:
    v = _meta_get(meta, "is_point_source", "point_source")
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    cs = _meta_get(meta, "class_star", "CLASS_STAR")
    if isinstance(cs, (int, float)) and math.isfinite(float(cs)):
        return float(cs) > 0.9
    morph = _meta_get(meta, "morphology", "morph")
    if isinstance(morph, str):
        m = morph.strip().lower()
        if m in ("point", "star", "stellar", "psf"):
            return True
        if m in ("extended", "galaxy"):
            return False
    return None


def read_spectrum_table(path_or_df, colmap: dict | None = None, conf: dict | None = None,
                        mode: str | None = None, source_id=None,
                        meta: dict | None = None) -> list[Spectrum] | ReaderUnavailable:
    """Level 4 extracted spectrum table(s) -> one :class:`Spectrum` per source.

    Wavelengths are converted to vacuum microns from the declared unit or their
    magnitude; ``resolving_power`` comes from the table's ``R`` else from the
    config disperser table for ``mode`` (G150: 461 lambda; P127: 80-180 across
    the band); ``contam`` and ``dq`` are ``None`` when absent so the overlap and
    quality tests are recorded as not run; ``is_point_source`` is taken from
    metadata (``is_point_source`` / ``class_star`` / ``morphology``) else ``None``.
    """
    loaded = load_table(path_or_df, meta)
    if isinstance(loaded, ReaderUnavailable):
        return loaded
    df, tmeta = loaded
    uri = str(path_or_df) if not hasattr(path_or_df, "columns") else "<dataframe>"
    roles = resolve_roles(df.columns, SPEC_ROLE_CANDIDATES, colmap)
    if roles["wavelength"] is None:
        return ReaderUnavailable("no wavelength column", uri, ["wavelength"])
    if roles["flux"] is None:
        return ReaderUnavailable("no flux column", uri, ["flux"])
    if roles["flux_err"] is None:
        return ReaderUnavailable("flux column without flux_err", uri, ["flux_err"])
    if len(df) == 0:
        return []
    wcol = roles["wavelength"]
    unit_hint = _meta_get(tmeta, "wave_unit", "wavelength_unit", f"unit.{wcol}", "waveunit")
    if unit_hint is None:
        wl_low = wcol.lower()
        unit_hint = "um" if wl_low.endswith(("_um", "_micron")) else \
            "angstrom" if "angstrom" in wl_low else "nm" if wl_low.endswith("_nm") else None
    wl_all, unit_used = wavelength_to_um(np.asarray(df[wcol], dtype=float), unit_hint)
    flux_all = np.asarray(df[roles["flux"]], dtype=float)
    err_all = np.asarray(df[roles["flux_err"]], dtype=float)
    if roles["flux_err"].lower() == "ivar":
        with np.errstate(divide="ignore", invalid="ignore"):
            err_all = np.where(err_all > 0, 1.0 / np.sqrt(err_all), np.nan)
    dq_all = None if roles["dq"] is None else \
        np.nan_to_num(np.asarray(df[roles["dq"]], dtype=float), nan=0.0).astype(np.int64)
    contam_all = None if roles["contam"] is None else np.asarray(df[roles["contam"]], dtype=float)
    mode_raw = mode or _meta_get(tmeta, "mode", "disperser", "optical_element", "grating",
                                 "element")
    mode_used = str(mode_raw).upper() if mode_raw else "unknown"
    sid_col = roles["source_id"]
    sid_default = str(source_id) if source_id is not None else \
        str(_meta_get(tmeta, "source_id", "id", "name", default="table"))
    sids = df[sid_col].astype(str).to_numpy() if sid_col else np.full(len(df), sid_default)
    r_meta = _meta_get(tmeta, "R", "resolving_power")
    survey = str(_meta_get(tmeta, "survey", default=""))
    detector = _meta_get(tmeta, "detector", "sca")
    flux_unit = str(_meta_get(tmeta, "flux_unit", f"unit.{roles['flux']}", "bunit",
                              default="arbitrary"))
    out: list[Spectrum] = []
    for sid in sorted(set(sids.tolist())):
        sel = sids == sid
        order = np.argsort(wl_all[sel], kind="stable")
        wl = wl_all[sel][order]
        if r_meta is not None:
            try:
                rp = float(r_meta) if np.isscalar(r_meta) or isinstance(r_meta, str) \
                    else np.asarray(r_meta, dtype=float)
                if isinstance(rp, np.ndarray) and rp.size != wl.size:
                    rp = float(np.nanmedian(rp))
                r_source = "table_meta"
            except Exception:  # noqa: BLE001
                rp, r_source = resolving_power_for(conf, mode_used, wl), "config/roman.yaml"
        else:
            rp = resolving_power_for(conf, mode_used, wl)
            r_source = "config/roman.yaml" if rp is not None else "none"
        ra = _first_finite(df.loc[sel, roles["ra"]]) if roles["ra"] else \
            float(_meta_get(tmeta, "ra", default=float("nan")) or float("nan"))
        dec = _first_finite(df.loc[sel, roles["dec"]]) if roles["dec"] else \
            float(_meta_get(tmeta, "dec", default=float("nan")) or float("nan"))
        m = {"wavelength_unit_in": unit_used, "r_source": r_source, "source": uri,
             "roles": {k: v for k, v in roles.items() if v}, "n_rows": int(sel.sum())}
        for k in ("simulated", "pipeline_version", "class_star", "morphology", "field"):
            v = _meta_get(tmeta, k)
            if v is not None:
                m[k] = json_safe(v)
        out.append(Spectrum(source_id=str(sid), ra=ra, dec=dec, mode=mode_used, wavelength_um=wl,
                            flux=flux_all[sel][order], flux_err=err_all[sel][order],
                            survey=survey, flux_unit=flux_unit, resolving_power=rp,
                            dq=None if dq_all is None else dq_all[sel][order],
                            contam=None if contam_all is None else contam_all[sel][order],
                            is_point_source=_point_source_from_meta(tmeta),
                            detector=None if detector is None else str(detector), meta=m))
    return out


__all__ = ["ReaderUnavailable", "dq_flags", "lightcurve_dq_mask", "read_asdf_arrays",
           "flatten_meta", "dq_cutouts_from_image", "stars_to_pixels", "read_ramp",
           "resultant_mid_times", "read_lightcurve_table", "read_spectrum_table",
           "wavelength_to_um", "resolving_power_for", "resolve_roles", "load_table",
           "band_aliases", "times_to_mjd", "LC_ROLE_CANDIDATES", "SPEC_ROLE_CANDIDATES",
           "MAG_ZP_DERIVED", "WFI_BAND_ALIASES"]
