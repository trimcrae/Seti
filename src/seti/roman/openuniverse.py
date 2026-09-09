"""OpenUniverse 2024 (pre-launch Roman simulation) -> the :mod:`seti.roman.schema` structures.

``s3://nasa-irsa-simulations/openuniverse2024/roman/`` holds the Roman-Rubin
joint simulation: SNANA light-curve files, galsim ``roman_imsim`` images with
per-image truth indices, the pointing sequence and the input point-source
catalogues.  This module is the only place that knows those layouts; every
reader emits :class:`LightCurve` / :class:`DQCutout` (or a
:class:`ReaderUnavailable` naming why not), stamped ``simulated=True``.

Conventions verified against the public files on 2026-09-09 (and re-checked
by the real-file tests whenever the sample files are present):

* ``HEAD.PTROBS_MIN`` / ``PTROBS_MAX`` are **1-based, inclusive** row numbers
  into ``PHOT``; ``NOBS == PTROBS_MAX - PTROBS_MIN + 1`` for every SN.
* ``PHOT`` carries **one separator row per SN** (``MJD == -777``, ``BAND ==
  '-'``, ``SIM_MAGOBS == 99``) *after* ``PTROBS_MAX`` -- outside the pointer
  range, so slicing by the pointers never sees it; the reader still drops any
  ``-777`` row defensively.
* The ``PHOT`` table of the ``ROMAN+LSST_LARGE_SNIa-normal`` release has only
  ``MJD, BAND, SIM_MAGOBS`` (noiseless model magnitudes; no ``FLUXCAL``): flux
  errors are **synthesised** from a depth model and every curve says so.
* Truth-index ``x, y`` are **1-based** (galsim ``GS_XMIN = 1``): the SIP WCS
  with ``origin=1`` reproduces them to 1e-7 px.  The reader measures this per
  image rather than assuming it (``meta["truth_xy_offset"]``).
* The truth-index ``mag`` column is *not* an AB magnitude: it is the galsim
  bandpass zero point (photons / cm^2 / s) minus 2.5 log10(total e-).  The
  image header's ``ZPTMAG`` equals 2.5 log10(EXPTIME x collecting area), so
  ``mag_ab = mag + ZPTMAG``; the implied per-e-/s zero point is recorded next
  to the config value so a drift would show.
* The simulated ``DQ`` plane is all zero in the files inspected: the flash
  channel's ``JUMP_DET`` assumption is a thing the census *records*, not one
  the reader fills in.
"""

from __future__ import annotations

import gzip
import math
import os
import re
import shutil
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .products import ReaderUnavailable, dq_cutouts_from_image, dq_flags, stars_to_pixels
from .schema import DQCutout, LightCurve

FLUXCAL_ZP_AB = 27.5                    # SNANA: FLUXCAL = 10**(-0.4 (mag - 27.5))
SNANA_SEPARATOR_MJD = -777.0
SNANA_SEPARATOR_BAND = "-"

# SNANA single-letter filters -> WFI names; LSST letters keep an ``lsst_`` prefix.
DEFAULT_BAND_LETTERS: dict[str, str] = {
    "R": "F062", "Z": "F087", "Y": "F106", "J": "F129", "W": "F146", "H": "F158",
    "F": "F184", "K": "F213",
    "u": "lsst_u", "g": "lsst_g", "r": "lsst_r", "i": "lsst_i", "z": "lsst_z", "y": "lsst_y",
}
# Image ``FILTER`` / obseq ``filter`` tokens -> WFI names.
DEFAULT_BAND_TOKENS: dict[str, str] = {
    "R062": "F062", "Z087": "F087", "Y106": "F106", "J129": "F129", "W146": "F146",
    "H158": "F158", "F184": "F184", "K213": "F213",
}
_ROMAN_BAND_RE = re.compile(r"^F\d{3}$")

# Filename layouts (bucket names and the local sample names both match).
_IMAGE_RE = re.compile(r"(?:^|_)(simple_model|truth|img)_([A-Za-z]\d{3})_(\d+)_(\d+)\.fits(?:\.gz)?$", re.I)
_INDEX_RE = re.compile(r"(?:^|_)index_([A-Za-z]\d{3})_(\d+)_(\d+)\.txt$", re.I)
_SNANA_RE = re.compile(r"^(.*)_(HEAD|PHOT)\.FITS(\.gz)?$", re.I)
_OBSEQ_RE = re.compile(r"obseq.*\.fits(\.gz)?$", re.I)
_CATALOG_RE = re.compile(r"^(pointsource|galaxy|snana)(?:_(flux|sed))?_(\d+)\.(parquet|hdf5|h5)$", re.I)


# --------------------------------------------------------------------------------------
# Config and band maps
# --------------------------------------------------------------------------------------

def ou_conf(conf: dict | None) -> dict:
    return dict((conf or {}).get("openuniverse") or {})


def band_letter_map(conf: dict | None = None) -> dict[str, str]:
    m = dict(DEFAULT_BAND_LETTERS)
    m.update({str(k): str(v) for k, v in (ou_conf(conf).get("band_letters") or {}).items()})
    return m


def band_token_map(conf: dict | None = None) -> dict[str, str]:
    m = dict(DEFAULT_BAND_TOKENS)
    m.update({str(k).upper(): str(v) for k, v in (ou_conf(conf).get("band_tokens") or {}).items()})
    return m


def is_roman_band(band: str | None) -> bool:
    return bool(band) and bool(_ROMAN_BAND_RE.match(str(band).upper()))


def wfi_band_from_letter(letter: str, conf: dict | None = None) -> str | None:
    """``'F'`` -> ``'F184'``, ``'u'`` -> ``'lsst_u'``; ``None`` for the separator / unknown."""
    s = str(letter).strip()
    if not s or s == SNANA_SEPARATOR_BAND:
        return None
    m = band_letter_map(conf)
    return m.get(s) or m.get(s.upper()) or m.get(s.lower())


def wfi_band_from_token(token: str, conf: dict | None = None) -> str | None:
    """``'R062'`` / ``'F184'`` / ``'f158'`` -> WFI name; ``None`` when unknown."""
    s = str(token).strip().upper()
    if not s:
        return None
    m = band_token_map(conf)
    if s in m:
        return m[s]
    if _ROMAN_BAND_RE.match(s):
        return s
    return None


# --------------------------------------------------------------------------------------
# Filename classification (what the inventory records, and what ingest falls back on)
# --------------------------------------------------------------------------------------

def _basename(uri: str) -> str:
    return str(uri).rstrip("/").rsplit("/", 1)[-1]


def _dirname(uri: str) -> str:
    s = str(uri).rstrip("/")
    return s.rsplit("/", 1)[0] if "/" in s else ""


def classify_openuniverse_uri(uri: str) -> dict | None:
    """Recognise an OpenUniverse product by its filename.

    Returns ``None`` for anything else, otherwise a dict with ``format`` in
    {``snana_head``, ``snana_phot``, ``openuniverse_image``,
    ``openuniverse_truth_index``, ``obseq``, ``pointsource``, ``galaxy_catalog``,
    ``snana_catalog``} plus the sibling URIs a reader needs (``phot_sibling`` for a
    HEAD file; ``truth_index`` / ``truth_index_candidates`` for an image), the
    band / pointing / SCA parsed from the name, and ``simulated: True``.
    """
    base = _basename(uri)
    d = _dirname(uri)
    m = _SNANA_RE.match(base)
    if m:
        stem, which, gz = m.group(1), m.group(2).upper(), m.group(3) or ""
        other = "PHOT" if which == "HEAD" else "HEAD"
        sib = f"{d}/{stem}_{other}.FITS{gz}" if d else f"{stem}_{other}.FITS{gz}"
        rec = {"format": f"snana_{which.lower()}", "simulated": True, "snana_stem": stem}
        rec["phot_sibling" if which == "HEAD" else "head_sibling"] = sib
        return rec
    m = _IMAGE_RE.search(base)
    if m:
        variant, tok, pointing, sca = m.group(1).lower(), m.group(2).upper(), m.group(3), m.group(4)
        band = wfi_band_from_token(tok)
        cands: list[str] = []
        # Bucket layout: .../images/simple_model/<TOK>/<pointing>/Roman_TDS_simple_model_<TOK>_<p>_<s>.fits.gz
        #                .../truth/<TOK>/<pointing>/Roman_TDS_index_<TOK>_<p>_<s>.txt
        mm = re.match(r"^(.*)/images/(?:simple_model|truth)/([^/]+)/([^/]+)$", d)
        if mm:
            root, tokdir, pdir = mm.groups()
            prefix = re.sub(r"_(simple_model|truth|img)_.*$", "", base, flags=re.I)
            cands.append(f"{root}/truth/{tokdir}/{pdir}/{prefix}_index_{tok}_{pointing}_{sca}.txt")
        prefix = re.sub(r"_(simple_model|truth|img)_.*$", "", base, flags=re.I)
        local_names = [f"{prefix}_index_{tok}_{pointing}_{sca}.txt", f"truth_index_{tok}_{pointing}_{sca}.txt"]
        for n in local_names:
            cands.append(f"{d}/{n}" if d else n)
        return {"format": "openuniverse_image", "image_variant": variant, "band": band, "band_token": tok,
                "pointing": int(pointing), "sca": int(sca), "truth_index": cands[0],
                "truth_index_candidates": cands, "simulated": True}
    m = _INDEX_RE.search(base)
    if m:
        tok = m.group(1).upper()
        return {"format": "openuniverse_truth_index", "band": wfi_band_from_token(tok), "band_token": tok,
                "pointing": int(m.group(2)), "sca": int(m.group(3)), "simulated": True}
    if _OBSEQ_RE.search(base):
        return {"format": "obseq", "simulated": True}
    m = _CATALOG_RE.match(base)
    if m:
        kind = m.group(1).lower()
        fmt = "pointsource" if kind == "pointsource" else f"{kind}_catalog"
        return {"format": fmt, "healpix": int(m.group(3)), "catalog_variant": (m.group(2) or "").lower(),
                "simulated": True}
    return None


# --------------------------------------------------------------------------------------
# SNANA light curves
# --------------------------------------------------------------------------------------

def _decode(v):
    if isinstance(v, bytes):
        return v.decode("ascii", "replace")
    return v


def read_snana_head(path: str | Path):
    """The SNANA HEAD table as a DataFrame: ``SNID`` (str), ``RA``, ``DEC``, ``NOBS``,
    ``PTROBS_MIN``/``MAX``, the redshift columns, ``PEAKMJD`` and every ``SIM_*``
    column present.  Gzip is handled by astropy."""
    import pandas as pd
    from astropy.io import fits

    with fits.open(str(path), memmap=False) as hdul:
        hdu = hdul[1]
        names = list(hdu.columns.names)
        keep = [c for c in names if c in ("SNID", "RA", "DEC", "NOBS", "PTROBS_MIN", "PTROBS_MAX",
                                          "SNTYPE", "PEAKMJD", "MWEBV", "FAKE")
                or c.startswith(("REDSHIFT", "HOSTGAL_OBJID", "HOSTGAL_PHOTOZ", "HOSTGAL_SPECZ",
                                 "HOSTGAL_LOGMASS", "HOSTGAL_SNSEP"))
                or c.startswith("SIM_")]
        data = hdu.data
        cols: dict = {}
        for c in keep:
            col = data[c]
            if col.dtype.kind in ("S", "U", "O"):
                cols[c] = [str(_decode(v)).strip() for v in col]
            elif col.ndim == 1:
                arr = np.asarray(col)
                # FITS tables are big-endian; pandas wants native byte order.
                cols[c] = arr.astype(arr.dtype.newbyteorder("=")) if arr.dtype.byteorder in (">", "<") else arr
        df = pd.DataFrame(cols)
    if "SNID" in df:
        df["SNID"] = df["SNID"].astype(str).str.strip()
    for c in ("PTROBS_MIN", "PTROBS_MAX", "NOBS"):
        if c in df:
            df[c] = df[c].astype(np.int64)
    return df


def _materialise_phot(phot_path: Path, cache_dir: Path | None = None, decompress: bool = True) -> Path:
    """A ``.FITS.gz`` PHOT table cannot be memory-mapped; stream-decompress it once
    (next to the source, or under ``cache_dir``) so the 31M-row table is sliced
    from disk.  Falls back to the gzip file itself when nothing can be written."""
    p = Path(phot_path)
    if not decompress or not p.name.lower().endswith(".gz"):
        return p
    target_dir = Path(cache_dir) if cache_dir else p.parent
    target = target_dir / p.name[:-3]
    tmp = target.with_name(target.name + ".tmp")
    try:
        if target.exists() and target.stat().st_size > 0:
            return target
        target_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        os.replace(tmp, target)
        return target
    except Exception:  # noqa: BLE001
        try:
            tmp.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        return p


def snana_error_model(mag: np.ndarray, conf: dict | None = None) -> tuple[np.ndarray, str]:
    """Magnitude error assumed for a noiseless SNANA model magnitude.

    ``err = max(assumed, slope * 10**(0.4 (mag - floor_at)))`` -- a flat floor at
    the bright end and a Poisson-like rise past ``floor_at`` (a crude Roman-depth
    model; every number from ``openuniverse:``).  Returns the array and a string
    naming the model so the curve's ``meta`` can carry it.
    """
    oc = ou_conf(conf)
    assumed = float(oc.get("snana_assumed_mag_err", 0.01))
    floor_at = float(oc.get("snana_err_floor_mag_at", 24.0))
    slope = float(oc.get("snana_err_mag_slope", 0.01))
    mag = np.asarray(mag, dtype=float)
    err = np.maximum(assumed, slope * 10.0 ** (0.4 * (mag - floor_at)))
    desc = f"max({assumed:g}, {slope:g}*10**(0.4*(mag-{floor_at:g}))) mag"
    return err, desc


def _object_class(row) -> str:
    model = str(row.get("SIM_MODEL_NAME", "") or "")
    tname = str(row.get("SIM_TYPE_NAME", "") or "")
    if "SALT" in model.upper():
        return "SNIa_model"
    if tname:
        return f"{tname}_model"
    return "SNANA_model"


def iter_snana_lightcurves(head_path: str | Path, phot_path: str | Path, conf: dict | None = None,
                           bands=None, max_objects: int | None = None, roman_only: bool = True,
                           cache_dir: Path | None = None) -> Iterator[LightCurve]:
    """One :class:`LightCurve` per (SN, band) from an SNANA HEAD/PHOT pair.

    Flux is SNANA ``FLUXCAL`` (zero point 27.5 AB): taken from the ``FLUXCAL`` /
    ``FLUXCALERR`` columns when the file has them, else from ``SIM_MAGOBS`` with
    errors from :func:`snana_error_model` and ``meta["errors_assumed"] = True``.
    Epochs fainter than ``openuniverse.snana_mag_faint_limit`` are dropped
    (``meta["n_dropped_faint"]``).  ``bands`` restricts to those WFI names;
    ``roman_only`` drops the LSST letters.  The PHOT table is opened once,
    memory-mapped (decompressed to disk first when gzipped) and sliced by the
    HEAD pointers; it is never loaded whole into pandas.
    """
    from astropy.io import fits

    oc = ou_conf(conf)
    faint_limit = float(oc.get("snana_mag_faint_limit", 30.0))
    head = read_snana_head(head_path)
    want = {str(b).upper() for b in bands} if bands else None
    phot_local = _materialise_phot(Path(phot_path), cache_dir=cache_dir,
                                   decompress=bool(oc.get("decompress_phot", True)))
    letters = band_letter_map(conf)
    with fits.open(str(phot_local), memmap=True) as hdul:
        tab = hdul[1].data
        names = set(hdul[1].columns.names)
        band_col = "BAND" if "BAND" in names else ("FLT" if "FLT" in names else None)
        if "MJD" not in names or band_col is None or not ({"SIM_MAGOBS", "FLUXCAL"} & names):
            raise ValueError(f"PHOT table lacks MJD/BAND/flux columns: {sorted(names)}")
        has_fluxcal = "FLUXCAL" in names and "FLUXCALERR" in names
        n_done = 0
        for _, row in head.iterrows():
            if max_objects is not None and n_done >= int(max_objects):
                break
            lo, hi = int(row["PTROBS_MIN"]) - 1, int(row["PTROBS_MAX"])
            if lo < 0 or hi <= lo or hi > len(tab):
                continue
            block = tab[lo:hi]
            mjd = np.asarray(block["MJD"], dtype=float)
            bl = np.array([str(_decode(b)).strip() for b in block[band_col]])
            keep = (mjd != SNANA_SEPARATOR_MJD) & (bl != SNANA_SEPARATOR_BAND)
            if has_fluxcal:
                flux_all = np.asarray(block["FLUXCAL"], dtype=float)
                err_all = np.asarray(block["FLUXCALERR"], dtype=float)
                mag_all = np.where(flux_all > 0, FLUXCAL_ZP_AB - 2.5 * np.log10(np.where(flux_all > 0, flux_all, 1.0)),
                                   np.inf)
            else:
                mag_all = np.asarray(block["SIM_MAGOBS"], dtype=float)
                flux_all = 10.0 ** (-0.4 * (mag_all - FLUXCAL_ZP_AB))
                err_mag, err_desc = snana_error_model(mag_all, conf)
                err_all = flux_all * err_mag * (math.log(10.0) / 2.5)
            n_done += 1
            snid = str(row["SNID"])
            meta_common = {
                "simulated": True, "object_class": _object_class(row), "snid": snid,
                "redshift": _f(row.get("REDSHIFT_FINAL")), "redshift_helio": _f(row.get("REDSHIFT_HELIO")),
                "sim_redshift_cmb": _f(row.get("SIM_REDSHIFT_CMB")),
                "peak_mjd": _f(row.get("SIM_PEAKMJD", row.get("PEAKMJD"))),
                "sim_model": str(row.get("SIM_MODEL_NAME", "") or ""),
                "sim_salt2x1": _f(row.get("SIM_SALT2x1")), "sim_salt2c": _f(row.get("SIM_SALT2c")),
                "mwebv": _f(row.get("MWEBV")), "host_logmass": _f(row.get("HOSTGAL_LOGMASS")),
                "flux_source": "FLUXCAL" if has_fluxcal else "SIM_MAGOBS", "noiseless": not has_fluxcal,
                "errors_assumed": not has_fluxcal,
                "error_model": None if has_fluxcal else err_desc,
                "origin": "openuniverse_snana", "ptrobs": [int(row["PTROBS_MIN"]), int(row["PTROBS_MAX"])],
            }
            for letter in sorted(set(bl[keep].tolist())):
                band = letters.get(letter) or letters.get(letter.upper()) or letters.get(letter.lower())
                if band is None:
                    continue
                roman = is_roman_band(band)
                if roman_only and not roman:
                    continue
                if want is not None and band.upper() not in want:
                    continue
                sel = keep & (bl == letter)
                n_raw = int(sel.sum())
                faint = sel & ~(mag_all <= faint_limit)
                sel = sel & (mag_all <= faint_limit)
                if not np.any(sel):
                    continue
                meta = dict(meta_common)
                meta.update(band_letter=letter, roman_band=roman, n_epochs_raw=n_raw,
                            n_dropped_faint=int(faint.sum()), mag_faint_limit=faint_limit,
                            peak_mag=_f(row.get(f"SIM_PEAKMAG_{letter}")))
                yield LightCurve(star_id=f"snana_{snid}", ra=float(row["RA"]), dec=float(row["DEC"]),
                                 band=band, mjd=mjd[sel], flux=flux_all[sel], flux_err=err_all[sel],
                                 survey="HLTDS_sim" if roman else "LSST_sim", flux_unit="fluxcal",
                                 flux_zp_ab=FLUXCAL_ZP_AB, time_system="MJD_sim", dq=None,
                                 exposure_s=None, meta=meta)


def _f(v) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------------------
# Truth index, images
# --------------------------------------------------------------------------------------

TRUTH_COLUMNS = ["object_id", "ra", "dec", "x", "y", "realized_flux", "flux", "mag", "obj_type"]


def read_truth_index(path: str | Path):
    """The per-image truth table (``# object_id ra dec x y realized_flux flux mag obj_type``)."""
    import pandas as pd

    p = Path(path)
    with open(p) as fh:
        first = fh.readline()
    names = TRUTH_COLUMNS
    if first.startswith("#"):
        toks = first.lstrip("#").split()
        if len(toks) >= 5:
            names = toks
    df = pd.read_csv(p, sep=r"\s+", comment="#", header=None, names=names, dtype={"object_id": str})
    if "obj_type" in df:
        df["obj_type"] = df["obj_type"].astype(str).str.strip()
    return df


def read_openuniverse_header(path: str | Path) -> dict | ReaderUnavailable:
    """PRIMARY-header facts of a galsim Roman image: exposure, time, band, ZPTMAG, SCA,
    the WCS (as an ``astropy.wcs.WCS``) and the image shape from the SCI HDU."""
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except Exception as exc:  # noqa: BLE001
        return ReaderUnavailable(reason=f"astropy unavailable: {exc}", uri=str(path), needs=["astropy"])
    p = Path(path)
    try:
        with fits.open(str(p), memmap=False) as hdul:
            hdr = hdul[0].header
            names = [h.name for h in hdul]
            sci = hdul["SCI"] if "SCI" in names else (hdul[1] if len(hdul) > 1 else None)
            shape = None
            if sci is not None:
                shape = (int(sci.header.get("NAXIS2", 0)), int(sci.header.get("NAXIS1", 0)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wcs = WCS(hdr)
            return {"exptime": _f(hdr.get("EXPTIME")), "mjd": _f(hdr.get("MJD-OBS")),
                    "filter_token": str(hdr.get("FILTER", "")).strip(), "zptmag": _f(hdr.get("ZPTMAG")),
                    "sca": hdr.get("SCA_NUM"), "date_obs": hdr.get("DATE-OBS"), "wcs": wcs,
                    "has_sip": wcs.sip is not None, "shape": shape, "hdus": names,
                    "gs_xmin": hdr.get("GS_XMIN"), "gs_ymin": hdr.get("GS_YMIN")}
    except Exception as exc:  # noqa: BLE001
        return ReaderUnavailable(reason=f"{type(exc).__name__}: {exc}", uri=str(p), needs=[])


def _read_dq_plane(path: Path) -> np.ndarray | ReaderUnavailable:
    from astropy.io import fits
    with fits.open(str(path), memmap=False) as hdul:
        names = [h.name for h in hdul]
        if "DQ" not in names:
            return ReaderUnavailable(reason=f"no DQ HDU (HDUs: {names})", uri=str(path), needs=["DQ HDU"])
        dq = np.asarray(hdul["DQ"].data)
    if dq.ndim != 2:
        return ReaderUnavailable(reason=f"DQ HDU is {dq.ndim}-D", uri=str(path), needs=[])
    return dq.astype(np.uint32, copy=False)


def dq_flag_census(dq: np.ndarray, flags: dict) -> dict:
    """Pixels carrying each named flag bit, plus totals and the distinct values seen."""
    d = np.asarray(dq).astype(np.int64, copy=False)
    out: dict = {"n_pixels": int(d.size), "n_nonzero": int(np.count_nonzero(d))}
    per_flag = {}
    for name, bit in (flags or {}).items():
        try:
            b = int(bit)
        except Exception:  # noqa: BLE001
            continue
        per_flag[str(name)] = int(np.count_nonzero(d & b)) if b else 0
    out["per_flag"] = per_flag
    vals, counts = np.unique(d, return_counts=True)
    order = np.argsort(-counts)[:20]
    out["distinct_values"] = {str(int(vals[i])): int(counts[i]) for i in order}
    return out


def _truth_xy_convention(wcs, ra, dec, x, y) -> dict:
    """Compare truth ``x, y`` with the WCS (0-based) on up to 200 stars: the
    median offset tells whether the truth columns are 1-based (galsim) or 0-based."""
    n = min(len(ra), 200)
    if n == 0:
        return {"origin": None, "offset": None, "n": 0, "max_residual_px": None}
    xw, yw = stars_to_pixels(wcs, np.asarray(ra[:n], float), np.asarray(dec[:n], float))
    dx, dy = xw - np.asarray(x[:n], float), yw - np.asarray(y[:n], float)
    mdx, mdy = float(np.nanmedian(dx)), float(np.nanmedian(dy))
    origin: int | None
    if abs(mdx + 1.0) < 0.1 and abs(mdy + 1.0) < 0.1:
        origin = 1
    elif abs(mdx) < 0.1 and abs(mdy) < 0.1:
        origin = 0
    else:
        origin = None
    resid = np.hypot(dx - mdx, dy - mdy)
    return {"origin": origin, "offset": [mdx, mdy], "n": int(n),
            "max_residual_px": float(np.nanmax(resid)) if resid.size else None}


def _wcs_pixels(wcs, ra, dec) -> tuple[np.ndarray, np.ndarray]:
    # astropy WCS.world_to_pixel_values is 0-based and includes SIP.
    return stars_to_pixels(wcs, ra, dec)


def read_openuniverse_image(path: str | Path, truth_index_path: str | Path | None = None,
                            conf: dict | None = None, stars_df=None, box: int = 32,
                            max_stars: int | None = None, mag_limit: float | None = None,
                            margin: int = 4, image_record: dict | None = None,
                            flags: dict | None = None) -> list[DQCutout] | ReaderUnavailable:
    """DQ cutouts around the catalogued stars of one galsim Roman image.

    Stars come from the truth index (``obj_type == 'star'``, inside the image by
    ``margin`` px, pixel origin measured against the WCS and recorded) or, with no
    index, from ``stars_df`` (``ra, dec`` and optionally ``id``/``star_id``,
    ``magnorm``) through the SIP WCS.  ``mag_limit`` is applied to the AB
    magnitude ``truth_mag + ZPTMAG`` (no cut when the star list has no such
    magnitude).  Only the DQ HDU is loaded.  ``image_record``, when given, is
    filled with the header facts, the whole-plane :func:`dq_flag_census` and the
    star counts even when no cutout results.
    """
    p = Path(path)
    hdr = read_openuniverse_header(p)
    if isinstance(hdr, ReaderUnavailable):
        return hdr
    dq = _read_dq_plane(p)
    if isinstance(dq, ReaderUnavailable):
        return dq
    ny, nx = dq.shape
    oc = ou_conf(conf)
    if max_stars is None:
        max_stars = oc.get("max_stars_per_image")
    if mag_limit is None:
        mag_limit = oc.get("star_mag_limit")
    if flags is None:
        flags, _src = dq_flags(conf)
    band = wfi_band_from_token(hdr["filter_token"], conf) or ""
    zpt = hdr["zptmag"]
    exptime = hdr["exptime"]
    wcs = hdr["wcs"]
    stem = re.sub(r"\.fits(\.gz)?$", "", p.name, flags=re.I)
    m = _IMAGE_RE.search(p.name)
    sca = hdr.get("sca")
    if sca is None and m:
        sca = int(m.group(4))
    pointing = int(m.group(3)) if m else None
    zp_cfg = None
    try:
        zp_cfg = float(conf["instruments"]["WFI"]["filters"][band]["zp_ab"])
    except Exception:  # noqa: BLE001
        pass

    rows: list[dict] = []
    conv: dict = {"origin": None, "offset": None, "n": 0, "max_residual_px": None}
    star_source = None
    n_in_index = None
    zp_implied = None
    if truth_index_path is not None:
        ti = read_truth_index(truth_index_path)
        st = ti[ti["obj_type"] == "star"] if "obj_type" in ti else ti
        n_in_index = int(len(st))
        star_source = "truth_index"
        if len(st):
            conv = _truth_xy_convention(wcs, st["ra"].values, st["dec"].values, st["x"].values, st["y"].values)
            if conv["origin"] is None:
                xs, ys = _wcs_pixels(wcs, st["ra"].values, st["dec"].values)
            else:
                xs = st["x"].values.astype(float) - float(conv["origin"])
                ys = st["y"].values.astype(float) - float(conv["origin"])
            mag_truth = st["mag"].values.astype(float) if "mag" in st else np.full(len(st), np.nan)
            flux = st["flux"].values.astype(float) if "flux" in st else np.full(len(st), np.nan)
            mag_ab = mag_truth + zpt if zpt is not None else np.full(len(st), np.nan)
            if zpt is not None and exptime:
                with np.errstate(divide="ignore", invalid="ignore"):
                    zp_i = mag_ab + 2.5 * np.log10(flux / float(exptime))
                zp_i = zp_i[np.isfinite(zp_i)]
                zp_implied = float(np.nanmedian(zp_i)) if zp_i.size else None
            for i in range(len(st)):
                rows.append({"star_id": str(st["object_id"].values[i]), "x": float(xs[i]), "y": float(ys[i]),
                             "ra": float(st["ra"].values[i]), "dec": float(st["dec"].values[i]),
                             "mag": _f(mag_ab[i]), "mag_truth": _f(mag_truth[i]), "flux_e": _f(flux[i]),
                             "realized_flux_e": _f(st["realized_flux"].values[i]) if "realized_flux" in st else None})
    elif stars_df is not None:
        star_source = "catalogue_wcs"
        n_in_index = int(len(stars_df))
        if len(stars_df):
            xs, ys = _wcs_pixels(wcs, stars_df["ra"].values.astype(float), stars_df["dec"].values.astype(float))
            idcol = "star_id" if "star_id" in stars_df else ("id" if "id" in stars_df else None)
            for i in range(len(stars_df)):
                r = stars_df.iloc[i]
                rows.append({"star_id": str(r[idcol]) if idcol else f"cat_{i}", "x": float(xs[i]), "y": float(ys[i]),
                             "ra": float(r["ra"]), "dec": float(r["dec"]),
                             "mag": _f(r["mag"]) if "mag" in stars_df else None,
                             "magnorm": _f(r["magnorm"]) if "magnorm" in stars_df else None})
    else:
        star_source = "none"
    inside = [r for r in rows if (margin <= r["x"] <= nx - 1 - margin and margin <= r["y"] <= ny - 1 - margin)]
    n_inside = len(inside)
    if mag_limit is not None:
        inside = [r for r in inside if r.get("mag") is None or r["mag"] <= float(mag_limit)]
    n_after_mag = len(inside)
    inside.sort(key=lambda r: (r.get("mag") if r.get("mag") is not None else np.inf))
    if max_stars is not None:
        inside = inside[:int(max_stars)]
    census = dq_flag_census(dq, flags)
    fwhm = 1.2
    try:
        fwhm = float((conf["instruments"]["WFI"].get("psf_fwhm_px") or {}).get(band, 1.2))
    except Exception:  # noqa: BLE001
        pass
    meta = {"zptmag": zpt, "simulated": True, "origin": "openuniverse", "n_stars_in_image": n_inside,
            "n_stars_in_index": n_in_index, "n_stars_after_mag_limit": n_after_mag, "n_stars_used": len(inside),
            "star_source": star_source, "truth_xy_offset": conv["offset"], "truth_xy_origin": conv["origin"],
            "truth_xy_max_residual_px": conv["max_residual_px"],
            "mag_convention": "mag_ab = truth_mag + ZPTMAG", "zp_ab_implied_per_e_s": zp_implied,
            "zp_ab_config": zp_cfg, "mag_limit": mag_limit, "band_token": hdr["filter_token"],
            "pointing": pointing, "has_sip": hdr["has_sip"], "image_shape": [ny, nx],
            "dq_all_zero": census["n_nonzero"] == 0}
    if image_record is not None:
        image_record.update({k: v for k, v in hdr.items() if k != "wcs"})
        image_record.update(band=band, dq_flag_census=census, image_id=stem, detector=f"SCA{sca}" if sca else "",
                            **{k: meta[k] for k in ("n_stars_in_image", "n_stars_in_index", "n_stars_used",
                                                   "star_source", "truth_xy_offset", "truth_xy_origin",
                                                   "zp_ab_implied_per_e_s", "zp_ab_config", "dq_all_zero")})
    return dq_cutouts_from_image(dq, inside, image_id=stem, box=int(box), band=band,
                                 detector=f"SCA{sca}" if sca is not None else "", mjd=hdr["mjd"],
                                 exposure_s=exptime, psf_fwhm_px=fwhm, meta=meta)


def read_openuniverse_science(path: str | Path, hdus=("SCI", "ERR", "DQ")) -> dict | ReaderUnavailable:
    """The science / error / dq planes and the primary header of one image (for
    photometry later).  Costs the full float64 plane; nothing in the DQ path calls it."""
    try:
        from astropy.io import fits
    except Exception as exc:  # noqa: BLE001
        return ReaderUnavailable(reason=f"astropy unavailable: {exc}", uri=str(path), needs=["astropy"])
    out: dict = {}
    try:
        with fits.open(str(path), memmap=False) as hdul:
            out["header"] = {k: hdul[0].header[k] for k in hdul[0].header if k not in ("COMMENT", "HISTORY")}
            names = [h.name for h in hdul]
            for name in hdus:
                if name in names:
                    arr = np.asarray(hdul[name].data)
                    out[name.lower()] = arr.astype(arr.dtype.newbyteorder("="), copy=False) \
                        if arr.dtype.byteorder in (">", "<") else arr
        return out
    except Exception as exc:  # noqa: BLE001
        return ReaderUnavailable(reason=f"{type(exc).__name__}: {exc}", uri=str(path), needs=[])


# --------------------------------------------------------------------------------------
# Pointing sequence
# --------------------------------------------------------------------------------------

def _visit_gaps(mjd: np.ndarray, gap_days: float) -> np.ndarray:
    """Gaps between the starts of successive visits, where a visit is a run of
    exposures separated by less than ``gap_days``."""
    t = np.sort(np.asarray(mjd, dtype=float))
    if t.size < 2:
        return np.array([])
    starts = [t[0]]
    for a, b in zip(t[:-1], t[1:], strict=True):
        if b - a > gap_days:
            starts.append(b)
    starts = np.asarray(starts)
    return np.diff(starts)


def read_obseq(path: str | Path, conf: dict | None = None, visit_gap_days: float | None = None) -> dict:
    """The survey-cadence record of a Roman TDS/WAS pointing sequence.

    Counts per filter and exposure time, the MJD span, the number of distinct
    epochs and their median spacing, the per-filter revisit gap (median gap
    between visit starts, visits split at ``visit_gap_days``), the
    ``hltds_like_cadence_days`` estimate (median of the per-filter revisits) and
    the footprint.  Every number is measured from the table.
    """
    from astropy.io import fits

    oc = ou_conf(conf)
    gap = float(visit_gap_days if visit_gap_days is not None else oc.get("obseq_visit_gap_days", 0.5))
    with fits.open(str(path), memmap=False) as hdul:
        t = hdul[1].data
        names = {n.lower(): n for n in hdul[1].columns.names}

        def col(*cands):
            for c in cands:
                if c in names:
                    return np.asarray(t[names[c]])
            return None

        ra, dec = col("ra"), col("dec")
        filt = col("filter", "band")
        expt = col("exptime", "exposure")
        mjd = col("date", "mjd", "mjd_obs")
        pa = col("pa")
    if mjd is None or filt is None:
        raise ValueError(f"obseq table lacks date/filter columns: {sorted(names)}")
    mjd = mjd.astype(float)
    filt = np.array([str(_decode(f)).strip() for f in filt])
    tokens, counts = np.unique(filt, return_counts=True)
    filters = {str(tok): {"wfi_band": wfi_band_from_token(tok, conf), "n_exposures": int(c)}
               for tok, c in zip(tokens, counts, strict=True)}
    exptimes = {}
    if expt is not None:
        ev, ec = np.unique(np.round(expt.astype(float), 3), return_counts=True)
        exptimes = {f"{float(v):g}": int(c) for v, c in zip(ev, ec, strict=True)}
        for tok in filters:
            ev2 = np.unique(np.round(expt[filt == tok].astype(float), 3))
            filters[tok]["exptimes_s"] = [float(v) for v in ev2]
    epochs = np.unique(np.round(mjd, 6))
    per_filter_revisit = {}
    for tok in filters:
        g = _visit_gaps(mjd[filt == tok], gap)
        n_visits = int(g.size + 1) if (filt == tok).any() else 0
        filters[tok].update(n_visits=n_visits,
                            median_revisit_days=float(np.median(g)) if g.size else None,
                            median_gap_between_exposures_days=(
                                float(np.median(np.diff(np.unique(mjd[filt == tok])))) if (filt == tok).sum() > 1 else None))
        if g.size:
            per_filter_revisit[tok] = float(np.median(g))
    cadence = float(np.median(list(per_filter_revisit.values()))) if per_filter_revisit else None
    rec = {"source": Path(path).name, "n_exposures": int(mjd.size), "mjd_min": float(mjd.min()),
           "mjd_max": float(mjd.max()), "span_days": float(mjd.max() - mjd.min()),
           "filters": filters, "exptimes_s": exptimes, "n_distinct_epochs": int(epochs.size),
           "median_gap_between_distinct_epochs_days": float(np.median(np.diff(epochs))) if epochs.size > 1 else None,
           "visit_gap_days": gap, "per_filter_median_revisit_days": per_filter_revisit,
           "hltds_like_cadence_days": cadence,
           "footprint": {"ra_min": float(ra.min()) if ra is not None else None,
                         "ra_max": float(ra.max()) if ra is not None else None,
                         "dec_min": float(dec.min()) if dec is not None else None,
                         "dec_max": float(dec.max()) if dec is not None else None,
                         "n_unique_pointings": (int(len(np.unique(np.round(np.c_[ra, dec], 5), axis=0)))
                                                if ra is not None and dec is not None else None)},
           "pa_values": ([float(v) for v in np.unique(np.round(pa.astype(float), 3))[:12]]
                         if pa is not None else None),
           "simulated": True}
    return rec


# --------------------------------------------------------------------------------------
# Point-source catalogue
# --------------------------------------------------------------------------------------

def read_pointsource_catalog(path: str | Path, conf: dict | None = None, max_rows: int | None = None):
    """Stars (``object_type == 'star'``) of a ``pointsource_<healpix>.parquet``:
    ``id, ra, dec, magnorm, pm_ra, pm_dec, pm, parallax, variability_model``.
    ``magnorm`` is the 500 nm normalising magnitude, not a WFI band magnitude."""
    import pandas as pd

    want = ["object_type", "id", "ra", "dec", "magnorm", "mura", "mudec", "parallax", "variability_model",
            "sed_filepath"]
    try:
        import pyarrow.parquet as pq
        have = set(pq.ParquetFile(str(path)).schema.names)
        df = pd.read_parquet(str(path), columns=[c for c in want if c in have])
    except Exception:  # noqa: BLE001
        df = pd.read_parquet(str(path))
        df = df[[c for c in want if c in df.columns]]
    if "object_type" in df:
        df = df[df["object_type"].astype(str) == "star"]
    df = df.rename(columns={"mura": "pm_ra", "mudec": "pm_dec"})
    if "pm_ra" in df and "pm_dec" in df:
        df["pm"] = np.hypot(df["pm_ra"].astype(float), df["pm_dec"].astype(float))
    if "id" in df:
        df["id"] = df["id"].astype(str)
    if "variability_model" in df:
        df["variability_model"] = df["variability_model"].astype(str).str.strip()
    if max_rows is not None:
        df = df.head(int(max_rows))
    return df.reset_index(drop=True)
