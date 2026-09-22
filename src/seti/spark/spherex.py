"""S48 — the spark: a single-channel excess in SPHEREx QR2 spectral images.

SPHEREx has no per-source spectra yet (the HRSC comes after the third sky
pass); what IRSA serves is the Quick Release level-2 spectral image: one
2040×2040 frame per detector per exposure, 6.15″ pixels, with the linear
variable filter mapping every pixel to its own wavelength through a lookup-
table (``-TAB``) spectral WCS.  A star's spectrum is therefore *assembled*:
every exposure that covers it contributes one (λ, flux) sample at the
wavelength of the pixel it fell on, and the many dithers of the survey walk
the star across the filter.

This module is pure except for the injected fetchers:

* :func:`obscore_adql` — the images covering a box (``spherex.obscore``);
* :func:`cutout_url` — IRSA IBE's ``?center=&size=`` cutout of one product;
* :class:`WaveMap` — the per-detector wavelength / bandwidth map, from the
  3-axis WCS when astropy decodes it and from the ``WCS-WAVE`` extension
  by hand when it does not; the map is a property of the *detector*, so it
  is read once from a full frame and applied to every cutout of that
  detector through the cutout's pixel offset (``LTV`` or ``CRPIX``);
* :func:`forced_photometry` — a fixed circular aperture with a local
  annulus, FLAGS-masked, VARIANCE-propagated, ZODI-plane subtracted;
* :func:`detect_single_channel_excess` — the statistic: per detector a
  robust running continuum in λ, residuals normalised by the empirical
  scatter, one channel = one resolution element; a candidate channel
  exceeds ``excess_sigma_total`` combined, with ≥ 2 independent passes and
  ≥ 2 detector positions each at ``excess_sigma_single``, and BOTH
  neighbouring channels quiet (the excess is confined to one element);
* :func:`screen_spherex` — every star, then the cross-star recurrence
  vetoes (same channel on ≥ 3 stars is instrumental; same detector pixel on
  ≥ 2 stars is a defect or persistence), the stellar-line veto, the
  industrial descriptor and the trials correction.
"""

from __future__ import annotations

import io
import math
import re
import time as _time
import warnings

import numpy as np
import pandas as pd

from . import lines as L
from .gaia import sep_arcsec

OBSCORE_COLS = ("obs_id", "obs_collection", "dataproduct_type", "calib_level", "s_ra", "s_dec",
                "s_fov", "t_min", "t_max", "t_exptime", "em_min", "em_max", "access_url",
                "access_format", "access_estsize", "instrument_name", "facility_name",
                "obs_publisher_did", "target_name")
DETECTOR_URI_RE = re.compile(r"_(\d)D(\d)_", re.IGNORECASE)
BAND_NAME_RE = re.compile(r"D\s*([1-6])", re.IGNORECASE)


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def obscore_adql(table: str, ra: float, dec: float, columns=None, *, radius_deg: float | None = None,
                 top: int | None = None, calib_level: int | None = 2) -> str:
    """Products whose footprint contains the point (or intersects the circle)."""
    cols = ", ".join(columns or OBSCORE_COLS)
    top_s = f"TOP {int(top)} " if top else ""
    if radius_deg:
        where = (f"1 = INTERSECTS(CIRCLE('ICRS', {float(ra):.6f}, {float(dec):.6f}, {float(radius_deg):.6f}), s_region)")
    else:
        where = f"1 = CONTAINS(POINT('ICRS', {float(ra):.6f}, {float(dec):.6f}), s_region)"
    q = f"SELECT {top_s}{cols} FROM {table} WHERE {where}"
    if calib_level is not None:
        q += f" AND calib_level = {int(calib_level)}"
    return q


def obscore_count_adql(table: str, ra: float, dec: float, calib_level: int | None = 2) -> str:
    q = (f"SELECT COUNT(*) AS n FROM {table} WHERE "
         f"1 = CONTAINS(POINT('ICRS', {float(ra):.6f}, {float(dec):.6f}), s_region)")
    if calib_level is not None:
        q += f" AND calib_level = {int(calib_level)}"
    return q


def detector_of(row, bands: dict) -> str | None:
    """``D1``…``D6`` from the bandpass name, the URI pattern, or the em range."""
    for key in ("energy_bandpassname", "instrument_name", "obs_id", "obs_publisher_did"):
        v = row.get(key) if hasattr(row, "get") else None
        if isinstance(v, str):
            m = BAND_NAME_RE.search(v)
            if m and "spherex" in v.lower() or (m and key == "energy_bandpassname"):
                return f"D{m.group(1)}"
    for key in ("access_url", "uri", "obs_publisher_did", "obs_id"):
        v = row.get(key) if hasattr(row, "get") else None
        if isinstance(v, str):
            m = DETECTOR_URI_RE.search(v)
            if m:
                return f"D{m.group(2)}"
    try:
        lo, hi = float(row.get("em_min")), float(row.get("em_max"))
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    # ObsCore em_* are metres; the config bands are microns
    lo_um, hi_um = (lo * 1e6, hi * 1e6) if hi < 1e-3 else (lo, hi)
    mid = 0.5 * (lo_um + hi_um)
    for name, (a, b) in bands.items():
        if float(a) <= mid <= float(b):
            return str(name)
    return None


def select_images(df: pd.DataFrame, bands: dict, max_per_detector: int) -> pd.DataFrame:
    """Per detector, up to ``max_per_detector`` products spread evenly in time."""
    if not len(df):
        out = df.copy()
        out["detector"] = []
        return out
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    d["detector"] = [detector_of(r, bands) for _, r in d.iterrows()]
    if "t_min" in d:
        d["t_min"] = pd.to_numeric(d["t_min"], errors="coerce")
        d = d.sort_values("t_min")
    keep = []
    for _det, g in d.groupby("detector", dropna=False):
        n = len(g)
        if max_per_detector and n > max_per_detector:
            pick = np.unique(np.linspace(0, n - 1, int(max_per_detector)).round().astype(int))
            keep.append(g.iloc[pick])
        else:
            keep.append(g)
    return pd.concat(keep).reset_index(drop=True) if keep else d


def product_url(base: str, uri_or_url: str) -> str:
    u = str(uri_or_url)
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return base.rstrip("/") + "/" + u.lstrip("/")


def cutout_url(url: str, ra: float, dec: float, size_arcsec: float) -> str:
    """IRSA IBE cutout syntax: ``?center=ra,dec&size=<n>arcsec``."""
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}center={float(ra):.6f},{float(dec):.6f}&size={float(size_arcsec):.1f}arcsec&gzip=false"


# --------------------------------------------------------------------------
# FITS access (runner) and description (probe)
# --------------------------------------------------------------------------
def fetch_fits(url: str, timeout: float = 120.0, session=None):
    """Download a FITS product into memory; returns an ``HDUList``.  Raises."""
    import requests
    from astropy.io import fits
    s = session or requests
    r = s.get(url, timeout=timeout, headers={"User-Agent": "seti-spark/1.0"})
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url[:200]}: {r.text[:200]!r}")
    return fits.open(io.BytesIO(r.content), memmap=False)


def describe_hdul(hdul, header_keys=("NAXIS", "NAXIS1", "NAXIS2", "WCSAXES", "CTYPE1", "CTYPE2",
                                     "CTYPE3", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "PS3_0",
                                     "PS3_1", "PS3_2", "PV3_1", "PV3_2", "PV3_3", "LTV1", "LTV2",
                                     "BUNIT", "EXTNAME", "MJD-AVG", "MJD-OBS", "DATE-OBS",
                                     "DETECTOR", "OBSID", "WCSNAME", "CTYPE3W", "PS3_0W")) -> list[dict]:
    """What the product actually is: HDU names, shapes, the WCS keys that matter."""
    out = []
    for i, h in enumerate(hdul):
        rec: dict = {"index": i, "name": str(getattr(h, "name", "")), "type": type(h).__name__}
        data = getattr(h, "data", None)
        try:
            rec["shape"] = list(np.shape(data)) if data is not None else None
        except Exception:                               # noqa: BLE001
            rec["shape"] = "?"
        try:
            if hasattr(h, "columns") and h.columns is not None:
                rec["columns"] = [{"name": c.name, "format": str(c.format), "dim": str(getattr(c, "dim", ""))}
                                  for c in h.columns]
        except Exception:                               # noqa: BLE001
            pass
        hdr = h.header
        rec["header"] = {k: (str(hdr[k]) if not isinstance(hdr[k], (int, float)) else hdr[k])
                         for k in header_keys if k in hdr}
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# The wavelength map
# --------------------------------------------------------------------------
_SPECTRAL_KEY_RE = re.compile(r"^(CTYPE|CUNIT|CRVAL|CRPIX|CDELT|CNAME|CRDER|CSYER)3[A-Z]?$|"
                              r"^(PS|PV)3_\d+[A-Z]?$|^(CD|PC)3_\d[A-Z]?$|^(CD|PC)\d_3[A-Z]?$|"
                              r"^(CTYPE|CUNIT|CRVAL|CRPIX|CDELT)[12]W$|^(CD|PC)[12]_[12]W$|^WCSNAMEW$|^LONPOLEW$|^LATPOLEW$")


def celestial_header(header):
    """A copy of the image header with every spectral (axis-3, alternate-W) key removed."""
    from astropy.io import fits
    h = fits.Header()
    for card in header.cards:
        k = str(card.keyword)
        if _SPECTRAL_KEY_RE.match(k):
            continue
        if k in ("WCSAXES",):
            continue
        try:
            h.append(card)
        except Exception:                               # noqa: BLE001
            continue
    h["WCSAXES"] = 2
    return h


def celestial_wcs(header):
    from astropy.wcs import WCS
    return WCS(celestial_header(header))


def _find_image_hdu(hdul):
    for name in ("IMAGE", "SCI", "PRIMARY"):
        if name in hdul and getattr(hdul[name], "data", None) is not None and np.ndim(hdul[name].data) == 2:
            return hdul[name]
    for h in hdul:
        if getattr(h, "data", None) is not None and np.ndim(h.data) == 2:
            return h
    raise ValueError("no 2-D image HDU")


def image_planes(hdul) -> dict:
    """``{"image", "variance", "flags", "zodi", "psf"}`` arrays where present."""
    out: dict = {"image": None, "variance": None, "flags": None, "zodi": None}
    img = _find_image_hdu(hdul)
    out["image"] = np.asarray(img.data, dtype=float)
    out["image_header"] = img.header
    for key, names in (("variance", ("VARIANCE", "VAR", "ERR")), ("flags", ("FLAGS", "DQ", "MASK")),
                       ("zodi", ("ZODI", "ZODIACAL"))):
        for n in names:
            if n in hdul and getattr(hdul[n], "data", None) is not None:
                arr = np.asarray(hdul[n].data)
                if arr.shape == out["image"].shape:
                    out[key] = arr.astype(float) if key != "flags" else arr.astype(np.int64)
                    break
    return out


class WaveMap:
    """Wavelength and bandwidth (µm) as a function of full-frame pixel (x, y), 0-based.

    Built from a full-frame product; ``method`` records how (``astropy_tab``,
    ``manual_tab``).  Grid-sampled every ``step`` pixels and bilinearly
    interpolated — the LVF varies smoothly, and a 6″ pixel is far below the
    resolution element.
    """

    def __init__(self, xs: np.ndarray, ys: np.ndarray, wave: np.ndarray, bw: np.ndarray | None,
                 method: str, naxis: tuple[int, int], unit_scale: float = 1.0):
        self.xs, self.ys = np.asarray(xs, float), np.asarray(ys, float)
        self.wave = np.asarray(wave, float) * unit_scale
        self.bw = None if bw is None else np.asarray(bw, float) * unit_scale
        self.method = method
        self.naxis = naxis

    # ---- construction ----
    @classmethod
    def from_hdul(cls, hdul, step: int = 8) -> WaveMap:
        img = _find_image_hdu(hdul)
        ny, nx = img.data.shape
        xs = np.arange(0, nx, step, dtype=float)
        ys = np.arange(0, ny, step, dtype=float)
        if xs[-1] != nx - 1:
            xs = np.append(xs, nx - 1)
        if ys[-1] != ny - 1:
            ys = np.append(ys, ny - 1)
        errors = []
        try:
            return cls._from_astropy(hdul, img, xs, ys, (nx, ny))
        except Exception as exc:                        # noqa: BLE001
            errors.append(f"astropy: {exc!r}"[:300])
        try:
            return cls._from_table(hdul, img, xs, ys, (nx, ny))
        except Exception as exc:                        # noqa: BLE001
            errors.append(f"manual: {exc!r}"[:300])
        raise ValueError("no wavelength map: " + "; ".join(errors))

    @classmethod
    def _from_astropy(cls, hdul, img, xs, ys, naxis) -> WaveMap:
        from astropy.wcs import WCS
        w = WCS(img.header, hdul)
        if w.naxis < 3:
            raise ValueError(f"WCS has {w.naxis} axes; no spectral axis")
        xx, yy = np.meshgrid(xs, ys)
        world = w.pixel_to_world_values(xx.ravel(), yy.ravel(), np.zeros(xx.size))
        wave = np.asarray(world[2], float).reshape(xx.shape)
        unit = str(w.wcs.cunit[2]).strip().lower() if len(w.wcs.cunit) > 2 else ""
        scale = {"m": 1e6, "um": 1.0, "micron": 1.0, "nm": 1e-3, "angstrom": 1e-4, "": 1.0}.get(unit, 1.0)
        if scale == 1.0 and np.nanmedian(wave) < 1e-3:
            scale = 1e6                                  # metres, unlabelled
        bw = None
        try:
            wb = WCS(img.header, hdul, key="W")
            worldb = wb.pixel_to_world_values(xx.ravel(), yy.ravel(), np.zeros(xx.size))
            bw = np.asarray(worldb[2], float).reshape(xx.shape)
        except Exception:                               # noqa: BLE001
            bw = None
        if not np.isfinite(wave).any():
            raise ValueError("astropy returned no finite wavelengths")
        # A lookup that does not vary across the detector is a mis-decoded
        # table (the third pixel axis read as the index), not an LVF: refuse it
        # rather than assemble every star's spectrum at one wavelength.
        span = float(np.nanmax(wave) - np.nanmin(wave))
        if span < 0.02 * float(np.nanmedian(wave)):
            raise ValueError(f"astropy map is degenerate (span {span:.3g} of median {np.nanmedian(wave):.3g})")
        return cls(xs, ys, wave, bw, "astropy_tab", naxis, unit_scale=scale)

    @classmethod
    def _from_table(cls, hdul, img, xs, ys, naxis) -> WaveMap:
        hdr = img.header
        ext = str(hdr.get("PS3_0", "WCS-WAVE")).strip()
        col = str(hdr.get("PS3_1", "COORDS")).strip()
        if ext not in hdul:
            cands = [h.name for h in hdul if "WAVE" in str(h.name).upper()]
            if not cands:
                raise ValueError(f"no {ext} extension")
            ext = cands[0]
        tab = hdul[ext]
        names = [c.name for c in tab.columns]
        if col not in names:
            col = next((n for n in names if "WAVE" in n.upper() or "COORD" in n.upper()), names[0])
        arr = np.asarray(tab.data[col][0], float)
        arr = np.squeeze(arr)
        bw_arr = None
        if arr.ndim == 3:                                # (ny', nx', M): M coords per cell
            if arr.shape[-1] >= 2:
                bw_arr = arr[..., 1]
            arr = arr[..., 0]
        elif arr.ndim != 2:
            raise ValueError(f"unexpected lookup shape {arr.shape}")
        for n in names:
            if bw_arr is None and "BAND" in n.upper():
                bw_arr = np.squeeze(np.asarray(tab.data[n][0], float))
                if bw_arr.ndim == 3:
                    bw_arr = bw_arr[..., 0]
        ny_t, nx_t = arr.shape
        nx, ny = naxis
        # lookup cell -> pixel: identity when the table is full-frame, scaled otherwise
        gx = np.linspace(0, nx - 1, nx_t) if nx_t != nx else np.arange(nx, dtype=float)
        gy = np.linspace(0, ny - 1, ny_t) if ny_t != ny else np.arange(ny, dtype=float)
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((gy, gx), arr, bounds_error=False, fill_value=np.nan)
        xx, yy = np.meshgrid(xs, ys)
        wave = f(np.column_stack([yy.ravel(), xx.ravel()])).reshape(xx.shape)
        bw = None
        if bw_arr is not None and bw_arr.shape == arr.shape:
            fb = RegularGridInterpolator((gy, gx), bw_arr, bounds_error=False, fill_value=np.nan)
            bw = fb(np.column_stack([yy.ravel(), xx.ravel()])).reshape(xx.shape)
        unit = str(hdr.get("CUNIT3", "")).strip().lower()
        scale = {"m": 1e6, "um": 1.0, "micron": 1.0, "nm": 1e-3, "angstrom": 1e-4, "": 1.0}.get(unit, 1.0)
        if scale == 1.0 and np.nanmedian(wave) < 1e-3:
            scale = 1e6
        return cls(xs, ys, wave, bw, "manual_tab", naxis, unit_scale=scale)

    @classmethod
    def synthetic(cls, nx: int, ny: int, lam_lo: float, lam_hi: float, R: float = 40.0,
                  axis: str = "y") -> WaveMap:
        """A linear-variable-filter model for the offline suite: λ varies along one axis."""
        xs, ys = np.arange(nx, dtype=float), np.arange(ny, dtype=float)
        xx, yy = np.meshgrid(xs, ys)
        t = (yy / max(ny - 1, 1)) if axis == "y" else (xx / max(nx - 1, 1))
        wave = lam_lo * (lam_hi / lam_lo) ** t
        return cls(xs, ys, wave, wave / R, "synthetic", (nx, ny))

    # ---- evaluation ----
    def _interp(self, arr: np.ndarray, x, y) -> np.ndarray:
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((self.ys, self.xs), arr, bounds_error=False, fill_value=np.nan)
        return f(np.column_stack([np.atleast_1d(np.asarray(y, float)), np.atleast_1d(np.asarray(x, float))]))

    def wavelength_at(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        lam = self._interp(self.wave, x, y)
        if self.bw is not None:
            bw = self._interp(self.bw, x, y)
        else:
            bw = lam / 40.0                              # nominal R if no bandwidth plane
        return lam, bw

    def fingerprint(self) -> dict:
        nx, ny = self.naxis
        pts = [(nx * 0.1, ny * 0.1), (nx * 0.5, ny * 0.5), (nx * 0.9, ny * 0.9), (nx * 0.1, ny * 0.9)]
        lam, bw = self.wavelength_at([p[0] for p in pts], [p[1] for p in pts])
        return {"method": self.method, "naxis": list(self.naxis),
                "lambda_um_at": [[round(p[0]), round(p[1]), None if not np.isfinite(v) else round(float(v), 5)]
                                 for p, v in zip(pts, lam, strict=True)],
                "bandwidth_um_median": None if not np.isfinite(bw).any() else round(float(np.nanmedian(bw)), 5),
                "lambda_um_range": [None if not np.isfinite(self.wave).any() else round(float(np.nanmin(self.wave)), 5),
                                    None if not np.isfinite(self.wave).any() else round(float(np.nanmax(self.wave)), 5)],
                "gradient_axis": self._gradient_axis()}

    def _gradient_axis(self) -> str:
        w = self.wave
        if not np.isfinite(w).any() or w.shape[0] < 2 or w.shape[1] < 2:
            return "?"
        gy = np.nanmedian(np.abs(np.diff(w, axis=0)))
        gx = np.nanmedian(np.abs(np.diff(w, axis=1)))
        return "y" if gy >= gx else "x"


def cutout_offset(cut_header, full_crpix: tuple[float, float] | None) -> tuple[float, float] | None:
    """``(x0, y0)`` such that ``x_full = x_cut + x0`` (0-based), from LTV or CRPIX shift."""
    if "LTV1" in cut_header and "LTV2" in cut_header:
        return -float(cut_header["LTV1"]), -float(cut_header["LTV2"])
    if full_crpix is not None and "CRPIX1" in cut_header and "CRPIX2" in cut_header:
        return float(full_crpix[0]) - float(cut_header["CRPIX1"]), float(full_crpix[1]) - float(cut_header["CRPIX2"])
    return None


def mjd_of(header) -> float:
    for k in ("MJD-AVG", "MJD-OBS", "MJD", "MJDAVG"):
        if k in header:
            try:
                return float(header[k])
            except (TypeError, ValueError):
                continue
    if "DATE-OBS" in header:
        try:
            from astropy.time import Time
            return float(Time(str(header["DATE-OBS"])).mjd)
        except Exception:                               # noqa: BLE001
            return float("nan")
    return float("nan")


# --------------------------------------------------------------------------
# Forced aperture photometry (pure numpy)
# --------------------------------------------------------------------------
def forced_photometry(image: np.ndarray, x: float, y: float, *, variance: np.ndarray | None = None,
                      flags: np.ndarray | None = None, zodi: np.ndarray | None = None,
                      aperture_px: float = 2.0, annulus_px=(4.0, 7.0), pixel_arcsec: float = 6.15,
                      flag_mask_any: bool = True) -> dict:
    """Aperture sum minus a clipped-median local background, in mJy.

    ``image`` in MJy/sr (SPHEREx level 2); ``Ω_pix = (pixel_arcsec/206265)²``
    sr, so ``F[mJy] = Σ(I − bg) · Ω_pix · 1e9``.  Any FLAGS-set pixel inside
    the aperture drops the sample (``ok = False``) when ``flag_mask_any``.
    Off-frame apertures return ``ok = False`` with ``reason``.
    """
    ny, nx = image.shape
    r_out = float(annulus_px[1])
    if not (r_out <= x <= nx - 1 - r_out and r_out <= y <= ny - 1 - r_out):
        return {"ok": False, "reason": "off_frame"}
    img = image - zodi if zodi is not None else image
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - x, yy - y)
    ap = rr <= float(aperture_px)
    ann = (rr >= float(annulus_px[0])) & (rr <= r_out)
    if flags is not None:
        bad = flags != 0
        n_flag_ap = int(np.sum(ap & bad))
        ann = ann & ~bad
    else:
        n_flag_ap = 0
    ap_vals = img[ap]
    ann_vals = img[ann]
    ann_vals = ann_vals[np.isfinite(ann_vals)]
    if ann_vals.size < 8:
        return {"ok": False, "reason": "no_annulus"}
    med = float(np.median(ann_vals))
    mad = float(1.4826 * np.median(np.abs(ann_vals - med)))
    keep = np.abs(ann_vals - med) <= 3.0 * max(mad, 1e-12)
    bg = float(np.median(ann_vals[keep])) if keep.any() else med
    bg_rms = float(1.4826 * np.median(np.abs(ann_vals[keep] - bg))) if keep.any() else mad
    if not np.isfinite(ap_vals).all():
        return {"ok": False, "reason": "nan_in_aperture", "n_flagged": n_flag_ap}
    n_ap = int(ap.sum())
    omega = (float(pixel_arcsec) / 206264.806) ** 2
    signal = float(np.sum(ap_vals - bg))
    if variance is not None:
        v = variance[ap]
        var_sum = float(np.nansum(np.where(np.isfinite(v) & (v > 0), v, bg_rms ** 2)))
    else:
        var_sum = n_ap * bg_rms ** 2
    var_sum += n_ap ** 2 * bg_rms ** 2 / max(int(keep.sum()), 1)     # background-level error
    out = {"ok": True, "flux_mjy": signal * omega * 1e9, "err_mjy": math.sqrt(var_sum) * omega * 1e9,
           "bg": bg, "bg_rms": bg_rms, "n_ap": n_ap, "n_flagged": n_flag_ap,
           "peak": float(np.max(ap_vals) - bg)}
    if flag_mask_any and n_flag_ap > 0:
        out["ok"] = False
        out["reason"] = "flagged"
    return out


# --------------------------------------------------------------------------
# The statistic
# --------------------------------------------------------------------------
def _passes(mjd: np.ndarray, gap_days: float) -> np.ndarray:
    """Pass labels: samples separated by < gap in time share a pass."""
    order = np.argsort(mjd)
    lab = np.zeros(mjd.size, int)
    cur, last = 0, None
    for i in order:
        if last is not None and np.isfinite(mjd[i]) and np.isfinite(last) and mjd[i] - last > gap_days:
            cur += 1
        lab[i] = cur
        last = mjd[i] if np.isfinite(mjd[i]) else last
    return lab


def _positions(x: np.ndarray, y: np.ndarray, min_sep: float) -> np.ndarray:
    """Greedy clustering of detector positions; two clusters are ≥ min_sep apart."""
    lab = np.full(x.size, -1, int)
    centres: list[tuple[float, float]] = []
    for i in range(x.size):
        for k, (cx, cy) in enumerate(centres):
            if math.hypot(x[i] - cx, y[i] - cy) < min_sep:
                lab[i] = k
                break
        else:
            centres.append((float(x[i]), float(y[i])))
            lab[i] = len(centres) - 1
    return lab


def running_continuum(lam: np.ndarray, flux: np.ndarray, err: np.ndarray, window_um: np.ndarray,
                      n_iter: int = 4, degree: int = 4) -> np.ndarray:
    """A smooth continuum: a clipped polynomial in log λ plus a running median of its residuals.

    Stage 1 removes the stellar slope and curvature (a Rayleigh-Jeans tail
    changes by 25 % across a 5-resel window, so a plain running median of a
    randomly sampled steep continuum is biased at the few-percent level —
    above the photometric noise of a bright star).  Stage 2, a running
    median of the *residuals* in λ (window ± ``window_um``), absorbs what
    the polynomial cannot (molecular bands, a companion's SED).  Both
    stages sigma-clip iteratively, so a single-channel excess — far
    narrower than the window — does not lift the continuum under itself,
    while a broad bump does and is therefore *not* a single-channel excess.
    """
    n = lam.size
    order = np.argsort(lam)
    ls, fs = lam[order], flux[order]
    es = np.where(np.isfinite(err[order]) & (err[order] > 0), err[order], np.inf)
    good = np.isfinite(fs) & np.isfinite(es)
    cont_s = np.full(n, np.nan)
    x = np.log(ls) - float(np.nanmean(np.log(ls[good]))) if good.any() else np.log(ls)
    deg = int(max(0, min(degree, good.sum() // 10)))
    for _ in range(n_iter):
        # stage 1: weighted polynomial on the unclipped samples
        try:
            w = 1.0 / np.where(np.isfinite(es), es, np.inf)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coef = np.polyfit(x[good], fs[good], deg, w=w[good]) if good.sum() > deg else np.array([np.nanmedian(fs[good])])
            poly = np.polyval(coef, x)
        except Exception:                               # noqa: BLE001
            poly = np.full(n, np.nanmedian(fs[good]) if good.any() else np.nan)
        r1 = fs - poly
        # stage 2: running median of the residuals
        med = np.zeros(n)
        for k in range(n):
            wk = float(window_um[order][k])
            lo = np.searchsorted(ls, ls[k] - wk, side="left")
            hi = np.searchsorted(ls, ls[k] + wk, side="right")
            sel = good[lo:hi]
            vals = r1[lo:hi][sel]
            med[k] = np.median(vals) if vals.size >= 3 else 0.0
        cont_s = poly + med
        with np.errstate(invalid="ignore"):
            z = (fs - cont_s) / es
        finite = np.isfinite(z)
        mad = 1.4826 * np.nanmedian(np.abs(z[finite] - np.nanmedian(z[finite]))) if finite.sum() >= 5 else 1.0
        scale = max(mad, 1.0)
        good = np.isfinite(fs) & np.isfinite(es) & (np.abs(z) <= 4.0 * scale)
    cont = np.empty(n)
    cont[order] = cont_s
    return cont


def detect_single_channel_excess(samples: pd.DataFrame, conf: dict, bands: dict) -> tuple[list[dict], dict]:
    """The S48 statistic on one star's samples.  Pure.

    ``samples`` columns: ``detector, wl_um, bw_um, flux_mjy, err_mjy, mjd,
    x_full, y_full``.  Returns ``(channels, per_detector)``: every tested
    channel with its combined significance, per-pass and per-position
    breakdown, neighbour-channel significances, edge distance and tier.
    """
    c = conf
    gap = float(c.get("pass_gap_days", 30.0))
    minsep = float(c.get("min_pixel_separation_px", 20.0))
    width = float(c.get("channel_width_resel", 1.0))
    s_tot = float(c.get("excess_sigma_total", 5.0))
    s_one = float(c.get("excess_sigma_single", 2.5))
    s_nb = float(c.get("neighbour_channel_sigma_max", 2.0))
    edge_resel = float(c.get("edge_margin_resel", 2.0))
    min_n = int(c.get("min_samples_per_detector", 12))
    win_resel = float(c.get("continuum_window_resel", 5.0))
    channels: list[dict] = []
    per_det: dict = {}
    if not len(samples):
        return channels, per_det
    d = samples.copy()
    for col in ("wl_um", "bw_um", "flux_mjy", "err_mjy", "mjd", "x_full", "y_full"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d[np.isfinite(d["wl_um"]) & np.isfinite(d["flux_mjy"]) & np.isfinite(d["err_mjy"]) & (d["err_mjy"] > 0)]
    for det, g in d.groupby("detector"):
        rec = {"n_samples": int(len(g)), "status": "ok"}
        if len(g) < min_n:
            rec["status"] = "insufficient_samples"
            per_det[str(det)] = rec
            continue
        lam = g["wl_um"].to_numpy(float)
        bw = g["bw_um"].to_numpy(float)
        bw = np.where(np.isfinite(bw) & (bw > 0), bw, lam / 40.0)
        flux = g["flux_mjy"].to_numpy(float)
        err = g["err_mjy"].to_numpy(float)
        cont = running_continuum(lam, flux, err, win_resel * bw)
        resid = flux - cont
        z = resid / err
        finite = np.isfinite(z)
        mad = 1.4826 * float(np.median(np.abs(z[finite] - np.median(z[finite])))) if finite.sum() >= 5 else 1.0
        scale = max(mad, 1.0)
        zn = z / scale
        R_eff = float(np.nanmedian(lam / bw))
        rec.update(empirical_scale=round(scale, 3), R_eff=round(R_eff, 1),
                   n_passes=int(np.unique(_passes(g["mjd"].to_numpy(float), gap)).size))
        bins = np.floor(np.log(lam) * R_eff / width).astype(int)
        passes = _passes(g["mjd"].to_numpy(float), gap)
        pos = _positions(g["x_full"].to_numpy(float), g["y_full"].to_numpy(float), minsep)
        rec["n_positions"] = int(np.unique(pos).size)
        band = bands.get(str(det))
        lam_lo = float(band[0]) if band else float(np.nanmin(lam))
        lam_hi = float(band[1]) if band else float(np.nanmax(lam))
        by_bin: dict[int, np.ndarray] = {b: np.where(bins == b)[0] for b in np.unique(bins)}

        def _zcomb(idx: np.ndarray, _zn: np.ndarray = zn) -> float:
            zz = _zn[idx]
            zz = zz[np.isfinite(zz)]
            return float(np.sum(zz) / math.sqrt(zz.size)) if zz.size else float("nan")

        n_tested = 0
        for b, idx in by_bin.items():
            if idx.size < 2:
                continue
            lam_c = float(np.exp((b + 0.5) * width / R_eff))
            zc = _zcomb(idx)
            pz = {int(p): _zcomb(idx[passes[idx] == p]) for p in np.unique(passes[idx])}
            qz = {int(p): _zcomb(idx[pos[idx] == p]) for p in np.unique(pos[idx])}
            n_pass_ok = sum(1 for v in pz.values() if np.isfinite(v) and v >= s_one)
            n_pos_ok = sum(1 for v in qz.values() if np.isfinite(v) and v >= s_one)
            nb_lo = _zcomb(by_bin[b - 1]) if (b - 1) in by_bin and by_bin[b - 1].size else float("nan")
            nb_hi = _zcomb(by_bin[b + 1]) if (b + 1) in by_bin and by_bin[b + 1].size else float("nan")
            edge_dist = min(lam_c - lam_lo, lam_hi - lam_c) * R_eff
            tested = len(pz) >= 2 and len(qz) >= 2
            n_tested += int(tested)
            ch = {"detector": str(det), "channel_bin": int(b), "lambda_um": round(lam_c, 5),
                  "resel_um": round(lam_c / R_eff, 5), "n_samples": int(idx.size),
                  "z_combined": round(zc, 3) if np.isfinite(zc) else None,
                  "n_passes": len(pz), "n_passes_above_single": n_pass_ok,
                  "n_positions": len(qz), "n_positions_above_single": n_pos_ok,
                  "z_by_pass": {k: round(v, 2) for k, v in pz.items() if np.isfinite(v)},
                  "z_by_position": {k: round(v, 2) for k, v in qz.items() if np.isfinite(v)},
                  "z_neighbour_lo": round(nb_lo, 2) if np.isfinite(nb_lo) else None,
                  "z_neighbour_hi": round(nb_hi, 2) if np.isfinite(nb_hi) else None,
                  "edge_distance_resel": round(edge_dist, 2), "tested": bool(tested),
                  "mean_flux_mjy": round(float(np.mean(flux[idx])), 5),
                  "mean_excess_mjy": round(float(np.mean(resid[idx])), 5),
                  "excess_fraction": round(float(np.mean(resid[idx]) / max(np.mean(cont[idx]), 1e-12)), 4),
                  "x_full_mean": round(float(np.mean(g["x_full"].to_numpy(float)[idx])), 1),
                  "y_full_mean": round(float(np.mean(g["y_full"].to_numpy(float)[idx])), 1),
                  "mjd_min": round(float(np.nanmin(g["mjd"].to_numpy(float)[idx])), 3) if np.isfinite(g["mjd"].to_numpy(float)[idx]).any() else None,
                  "mjd_max": round(float(np.nanmax(g["mjd"].to_numpy(float)[idx])), 3) if np.isfinite(g["mjd"].to_numpy(float)[idx]).any() else None}
            reasons = []
            if not (np.isfinite(zc) and zc >= s_tot):
                reasons.append("below_total_sigma")
            if n_pass_ok < 2:
                reasons.append("fewer_than_two_passes")
            if n_pos_ok < 2:
                reasons.append("fewer_than_two_positions")
            nb_quiet = all(np.isfinite(v) and abs(v) <= s_nb for v in (nb_lo, nb_hi))
            if not nb_quiet:
                reasons.append("neighbour_channel_not_quiet" if (np.isfinite(nb_lo) and np.isfinite(nb_hi))
                               else "neighbour_channel_unsampled")
            if edge_dist < edge_resel:
                reasons.append("lvf_edge")
            ch["reasons"] = reasons
            ch["tier"] = "candidate" if not reasons else (
                "watch" if (np.isfinite(zc) and zc >= s_tot and set(reasons) <= {"neighbour_channel_unsampled", "fewer_than_two_positions"}) else "none")
            channels.append(ch)
        rec["n_channels_tested"] = n_tested
        rec["n_channels"] = int(len(by_bin))
        per_det[str(det)] = rec
    return channels, per_det


def screen_spherex(samples: pd.DataFrame, seeds: pd.DataFrame, conf: dict, bands: dict) -> tuple[pd.DataFrame, dict]:
    """Every seed star through the statistic, then the cross-star vetoes and the trials.

    Returns ``(channels, funnel)``: one row per tested channel with
    ``tier``, ``vetoes`` (``stellar_line``, ``recurrent_channel``,
    ``recurrent_pixel``), ``industrial_flag``, ``survivor``, ``p_global``.
    """
    c = conf
    rows: list[dict] = []
    stars_done: dict = {}
    for sid, g in samples.groupby("source_id"):
        chans, per_det = detect_single_channel_excess(g, c, bands)
        stars_done[sid] = {"n_samples": int(len(g)), "per_detector": per_det,
                           "n_channels_tested": int(sum(v.get("n_channels_tested", 0) for v in per_det.values()))}
        for ch in chans:
            rows.append({"source_id": sid, **ch})
    funnel: dict = {"n_seeds": int(len(seeds)), "n_stars_with_samples": len(stars_done),
                    "n_samples": int(len(samples)),
                    "n_channels_tested": int(sum(v["n_channels_tested"] for v in stars_done.values())),
                    "n_stars_insufficient": int(sum(1 for v in stars_done.values()
                                                    if v["n_channels_tested"] == 0))}
    # Coverage: what fraction of each detector band the stellar-line veto even
    # leaves available for a single-channel detection.  A detection can only be
    # claimed where one was possible, so this number travels with the verdict.
    cov: dict = {}
    if len(samples):
        s = samples.copy()
        for col in ("wl_um", "bw_um"):
            s[col] = pd.to_numeric(s[col], errors="coerce")
        for det, g in s.groupby("detector"):
            lam = g["wl_um"].to_numpy(float)
            bw = g["bw_um"].to_numpy(float)
            bw = np.where(np.isfinite(bw) & (bw > 0), bw, lam / 40.0)
            ok = np.isfinite(lam) & (lam > 0) & np.isfinite(bw) & (bw > 0)
            if not ok.any():
                continue
            R_eff = float(np.median(lam[ok] / bw[ok]))
            band = bands.get(str(det)) or [float(np.min(lam[ok])), float(np.max(lam[ok]))]
            cov[str(det)] = {"R_eff": round(R_eff, 1),
                             **L.clean_channel_fraction(float(band[0]), float(band[1]), R_eff,
                                                        tol_resel=float(c.get("stellar_tol_resel", 1.0)),
                                                        width_resel=float(c.get("channel_width_resel", 1.0)))}
    funnel["clean_channel_coverage"] = cov
    cols = ["source_id", "detector", "channel_bin", "lambda_um", "resel_um", "n_samples", "z_combined",
            "n_passes", "n_passes_above_single", "n_positions", "n_positions_above_single",
            "z_neighbour_lo", "z_neighbour_hi", "edge_distance_resel", "tested", "mean_flux_mjy",
            "mean_excess_mjy", "excess_fraction", "x_full_mean", "y_full_mean", "mjd_min", "mjd_max",
            "reasons", "tier", "stellar_line", "stellar_line_name", "stellar_line_sep_resel",
            "stellar_band_name", "recurrent_channel",
            "recurrence_n_stars", "recurrent_pixel", "pixel_recurrence_n_stars", "industrial_flag",
            "vetoes", "survivor", "p_single", "p_global", "significant_after_trials"]
    if not rows:
        funnel.update(n_channels_above_total=0, n_candidates_before_vetoes=0, n_survivors=0,
                      veto_counts={"stellar_line": 0, "recurrent_channel": 0, "recurrent_pixel": 0})
        return pd.DataFrame(columns=cols), funnel
    d = pd.DataFrame(rows)
    d["reasons"] = ["|".join(r) for r in d["reasons"]]
    tol = float(c.get("stellar_tol_resel", 1.0)) * d["resel_um"]
    # At R ~ 40 a broad molecular band spans many channels and cannot make a
    # single-channel excess with quiet neighbours; only its edge can.  Discrete
    # lines veto wherever they fall.  (lines.stellar_line_match docstring.)
    sl = [L.stellar_line_match(w, t, bands_edge_only=True) for w, t in zip(d["lambda_um"], tol, strict=True)]
    d["stellar_line"] = [m is not None for m in sl]
    d["stellar_line_name"] = [m.name if m else "" for m in sl]
    d["stellar_line_sep_resel"] = [round(abs(float(w) - m.um) / float(r), 3) if m and m.kind == "line" else None
                                   for w, r, m in zip(d["lambda_um"], d["resel_um"], sl, strict=True)]
    # Descriptor (never a veto): the broad molecular / ice band the channel sits inside.
    sb = [L.stellar_line_match(w, t) for w, t in zip(d["lambda_um"], tol, strict=True)]
    d["stellar_band_name"] = [m.name if (m is not None and m.kind == "band") else "" for m in sb]
    d["industrial_flag"] = [L.industrial_flag(w, t) or "" for w, t in zip(d["lambda_um"], tol, strict=True)]
    # recurrence: the same channel (detector, bin) elevated above the total threshold on >= N stars
    s_tot = float(c.get("excess_sigma_total", 5.0))
    hot = d[(d["z_combined"].fillna(-99) >= s_tot)]
    key = list(zip(d["detector"], d["channel_bin"], strict=True))
    counts = hot.groupby(["detector", "channel_bin"])["source_id"].nunique().to_dict()
    d["recurrence_n_stars"] = [int(counts.get(k, 0)) for k in key]
    d["recurrent_channel"] = d["recurrence_n_stars"] >= int(c.get("recurrence_min_stars", 3))
    # pixel recurrence: hot channels on >= N stars at the same detector pixel
    px = float(c.get("pixel_recurrence_px", 3.0))
    d["pixel_recurrence_n_stars"] = 0
    if len(hot):
        hx, hy, hs, hd = hot["x_full_mean"].to_numpy(float), hot["y_full_mean"].to_numpy(float), hot["source_id"].to_numpy(), hot["detector"].to_numpy()
        vals = []
        for _, r in d.iterrows():
            same = (hd == r["detector"]) & (np.hypot(hx - float(r["x_full_mean"]), hy - float(r["y_full_mean"])) <= px)
            vals.append(int(pd.unique(hs[same]).size))
        d["pixel_recurrence_n_stars"] = vals
    d["recurrent_pixel"] = d["pixel_recurrence_n_stars"] >= int(c.get("pixel_recurrence_min_stars", 2))
    vet = ["stellar_line", "recurrent_channel", "recurrent_pixel"]
    d["vetoes"] = ["|".join(v for v in vet if bool(r[v])) for _, r in d[vet].iterrows()]
    d["survivor"] = (d["tier"] == "candidate") & (d["vetoes"] == "")
    from scipy.stats import norm
    ntrials = max(1, int(funnel["n_channels_tested"]))
    d["p_single"] = [float(norm.sf(z)) if z is not None and np.isfinite(z) else 1.0 for z in d["z_combined"]]
    d["p_global"] = np.minimum(1.0, d["p_single"] * ntrials)
    d["significant_after_trials"] = d["p_global"] < float(c.get("trials_alpha", 0.01))
    funnel.update(n_trials=ntrials,
                  n_channels_above_total=int((d["z_combined"].fillna(-99) >= s_tot).sum()),
                  n_candidates_before_vetoes=int((d["tier"] == "candidate").sum()),
                  n_watch=int((d["tier"] == "watch").sum()),
                  n_survivors=int(d["survivor"].sum()),
                  n_survivors_significant=int((d["survivor"] & d["significant_after_trials"]).sum()),
                  veto_counts={v: int((d[v] & (d["tier"] == "candidate")).sum()) for v in vet},
                  reason_counts={k: int(v) for k, v in pd.Series([x for r in d["reasons"] for x in r.split("|") if x]).value_counts().items()},
                  industrial_flag_counts={k: int(v) for k, v in d.loc[(d["industrial_flag"] != "") & (d["tier"] != "none"), "industrial_flag"].value_counts().items()},
                  recurrent_channels=[{"detector": k[0], "bin": int(k[1]), "n_stars": int(v)} for k, v in counts.items()
                                      if v >= int(c.get("recurrence_min_stars", 3))])
    d = d.sort_values(["survivor", "z_combined"], ascending=[False, False]).reset_index(drop=True)
    return d[[x for x in cols if x in d.columns]], funnel


# --------------------------------------------------------------------------
# Sample assembly for one cutout (pure given the arrays)
# --------------------------------------------------------------------------
def samples_from_cutout(planes: dict, wcs_cel, seeds: pd.DataFrame, wavemap: WaveMap | None,
                        offset: tuple[float, float] | None, conf: dict, *, meta: dict,
                        local_wavemap: WaveMap | None = None) -> list[dict]:
    """Forced photometry of every seed on one cutout → sample rows.

    ``wavemap`` is the detector's full-frame map (needs ``offset``);
    ``local_wavemap`` is a map built from the cutout itself (no offset).
    A seed outside the cutout, or whose wavelength is unknown, yields no row
    but is counted in ``meta``-side ledgers by the caller.
    """
    img = planes["image"]
    ny, nx = img.shape
    out: list[dict] = []
    ra = seeds["ra_ep"].to_numpy(float) if "ra_ep" in seeds else seeds["ra"].to_numpy(float)
    dec = seeds["dec_ep"].to_numpy(float) if "dec_ep" in seeds else seeds["dec"].to_numpy(float)
    try:
        xs, ys = wcs_cel.all_world2pix(ra, dec, 0)
    except Exception:                                   # noqa: BLE001
        xs, ys = wcs_cel.wcs_world2pix(ra, dec, 0)
    for k in range(len(seeds)):
        x, y = float(xs[k]), float(ys[k])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        phot = forced_photometry(img, x, y, variance=planes.get("variance"), flags=planes.get("flags"),
                                 zodi=planes.get("zodi") if conf.get("subtract_zodi_plane", True) else None,
                                 aperture_px=float(conf.get("aperture_radius_px", 2.0)),
                                 annulus_px=tuple(conf.get("annulus_px", (4.0, 7.0))),
                                 pixel_arcsec=float(conf.get("pixel_arcsec", 6.15)),
                                 flag_mask_any=bool(conf.get("flag_mask_any", True)))
        if local_wavemap is not None:
            lam, bw = local_wavemap.wavelength_at(x, y)
            xf, yf = (x + offset[0], y + offset[1]) if offset else (x, y)
        elif wavemap is not None and offset is not None:
            xf, yf = x + offset[0], y + offset[1]
            lam, bw = wavemap.wavelength_at(xf, yf)
        else:
            continue
        row = {"source_id": seeds["source_id"].iloc[k], "x_cut": round(x, 2), "y_cut": round(y, 2),
               "x_full": round(float(xf), 2), "y_full": round(float(yf), 2),
               "wl_um": float(lam[0]), "bw_um": float(bw[0]), "ok": bool(phot.get("ok")),
               "reason": phot.get("reason", ""), "flux_mjy": phot.get("flux_mjy", np.nan),
               "err_mjy": phot.get("err_mjy", np.nan), "bg": phot.get("bg", np.nan),
               "n_flagged": phot.get("n_flagged", 0), "peak": phot.get("peak", np.nan), **meta}
        out.append(row)
    return out


def seed_in_box(seeds: pd.DataFrame, ra0: float, dec0: float, half_arcmin: float) -> pd.DataFrame:
    """Seeds inside the square box (small-angle, cos δ corrected)."""
    if not len(seeds):
        return seeds
    dra = (seeds["ra"].to_numpy(float) - ra0 + 180.0) % 360.0 - 180.0
    dx = dra * math.cos(math.radians(dec0)) * 60.0
    dy = (seeds["dec"].to_numpy(float) - dec0) * 60.0
    return seeds[(np.abs(dx) <= half_arcmin) & (np.abs(dy) <= half_arcmin)].reset_index(drop=True)


def box_radius_deg(box_arcmin: float) -> float:
    return float(box_arcmin) / 60.0 * math.sqrt(2.0) / 2.0 + 0.01


def elapsed(t0: float) -> float:
    return round(_time.monotonic() - t0, 1)


def separation_check(seeds: pd.DataFrame, ra0: float, dec0: float) -> np.ndarray:
    return sep_arcsec(seeds["ra"], seeds["dec"], ra0, dec0)
