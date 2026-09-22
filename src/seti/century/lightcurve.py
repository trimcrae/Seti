"""A DASCH light curve as the channel sees it.

A plate light curve differs from a CCD one in three ways that every statistic
downstream has to respect, so they are made explicit here rather than in each
detector:

1. **Every plate has its own limiting magnitude at the source position**
   (``limiting_mag_local``), and the archive reports the plates on which the
   star was *not* detected together with that limit.  A non-detection is a
   censored measurement, not a missing one.  The light curve therefore carries
   both the detections (``t, mag, err, lim``) and the non-detections
   (``t_nd, lim_nd``), and calendar blocks carry both too, so an injection can
   be censored against the block's real limits.
2. **Plates come in series** --- different telescopes, emulsions and plate
   scales, clustered in calendar time.  Blending, depth and passband are all
   properties of the series, so the series label rides along with every point
   and every statistic can be re-done inside one series.
3. **Exposures are long** (tens of minutes), so a short period is smeared by a
   known factor ``|sinc(f * t_exp)|`` per plate.  The exposure time rides along
   for the same reason.

Nothing here touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .api import numeric, pick_column
from .flags import FlagDefs, apply_masks

JD_MJD_OFFSET = 2400000.5
MJD_J2000 = 51544.5

TIME_COLS = ("time", "date_jd", "jd", "hjd", "date_hjd", "mjd", "date_mjd")
MAG_COLS = ("magcal_magdep", "magcal_local", "mag", "magnitude")
ERR_COLS = ("magcal_local_rms", "magcal_magdep_rms", "magcal_rms", "mag_err", "magerr", "err")
LIM_COLS = ("limiting_mag_local", "limiting_mag", "lim_mag", "limmag", "limiting_magnitude")
SERIES_COLS = ("series", "plate_series")
PLATENUM_COLS = ("platenum", "plate_number", "plate")
MOSNUM_COLS = ("mosnum", "mosaic_number")
EXPNUM_COLS = ("expnum", "exposure_number")
EXPTIME_COLS = ("exptime", "exposure_time", "exp_time", "exposure_minutes")
AFLAG_COLS = ("aflags", "a_flags", "aflag")
BFLAG_COLS = ("bflags", "b_flags", "bflag")
QUALITY_COLS = ("quality",)


def mjd_to_year(mjd) -> np.ndarray:
    """Decimal year (J2000 convention: 2000.0 at MJD 51544.5)."""
    return 2000.0 + (np.asarray(mjd, dtype=float) - MJD_J2000) / 365.25


def year_to_mjd(year) -> np.ndarray:
    return MJD_J2000 + (np.asarray(year, dtype=float) - 2000.0) * 365.25


def _time_to_mjd(vals: np.ndarray, colname: str) -> np.ndarray:
    """Interpret a time column: JD-scale numbers (> 2.3e6) become MJD."""
    v = np.asarray(vals, dtype=float)
    name = colname.lower()
    finite = v[np.isfinite(v)]
    if "mjd" in name and "jd" in name and not name.startswith("date_jd"):
        return v
    if finite.size and np.nanmedian(finite) > 2.0e6:
        return v - JD_MJD_OFFSET
    return v


@dataclass
class CenturyLC:
    """One star's plate photometry: detections, non-detections, and provenance."""

    # Detections
    t: np.ndarray                    # MJD
    mag: np.ndarray
    err: np.ndarray
    lim: np.ndarray                  # per-plate limiting magnitude at the position
    series: np.ndarray               # str
    exptime_min: np.ndarray          # minutes (NaN if unknown)
    blend: np.ndarray                # bool
    reject: np.ndarray               # bool: point must not be used
    plate: np.ndarray                # str identifier (series+platenum[+mosnum+expnum])
    # Non-detections
    t_nd: np.ndarray = field(default_factory=lambda: np.array([], float))
    lim_nd: np.ndarray = field(default_factory=lambda: np.array([], float))
    series_nd: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    exptime_nd: np.ndarray = field(default_factory=lambda: np.array([], float))
    # Provenance
    flags_applied: bool = False
    n_raw: int = 0
    columns: list[str] = field(default_factory=list)
    exptime_unit: str = "unknown"

    # ---- basic views ------------------------------------------------------
    @property
    def n_det(self) -> int:
        return int(len(self.t))

    @property
    def n_nd(self) -> int:
        return int(len(self.t_nd))

    @property
    def year(self) -> np.ndarray:
        return mjd_to_year(self.t)

    @property
    def year_nd(self) -> np.ndarray:
        return mjd_to_year(self.t_nd)

    @property
    def good(self) -> np.ndarray:
        """Usable, unblended detections with finite magnitude and error."""
        return (np.isfinite(self.t) & np.isfinite(self.mag) & np.isfinite(self.err)
                & (self.err > 0) & ~self.blend & ~self.reject)

    @property
    def usable(self) -> np.ndarray:
        """Finite, non-rejected detections (blends kept --- for the blend history)."""
        return (np.isfinite(self.t) & np.isfinite(self.mag) & np.isfinite(self.err)
                & (self.err > 0) & ~self.reject)

    def median_mag(self) -> float:
        g = self.good
        return float(np.median(self.mag[g])) if g.any() else float("nan")

    def blend_fraction(self) -> float:
        u = self.usable
        return float(np.mean(self.blend[u])) if u.any() else float("nan")

    def span_years(self) -> float:
        g = self.good
        return float(np.ptp(self.year[g])) if g.sum() > 1 else 0.0

    def dominant_series(self, mask=None) -> tuple[str, float]:
        m = self.good if mask is None else (self.good & mask)
        if not m.any():
            return "", 0.0
        vals, cnt = np.unique(self.series[m].astype(str), return_counts=True)
        i = int(np.argmax(cnt))
        return str(vals[i]), float(cnt[i] / cnt.sum())

    def margin_mask(self, margin: float, ref_mag: float | None = None) -> np.ndarray:
        """Detections on plates whose limit is at least ``margin`` deeper than the
        STAR'S median magnitude.

        The cut is keyed to the star's typical brightness and the plate's limit
        --- never to the point's own measured magnitude, which would select on
        the very quantity being measured and bias the mean bright on shallow
        plates in a way that tracks the plate-depth history (i.e. calendar time).
        """
        ref = self.median_mag() if ref_mag is None else float(ref_mag)
        lim = np.where(np.isfinite(self.lim), self.lim, -np.inf)
        return self.good & (lim - ref >= float(margin))

    def subset(self, mask: np.ndarray) -> CenturyLC:
        m = np.asarray(mask, dtype=bool)
        return CenturyLC(
            t=self.t[m], mag=self.mag[m], err=self.err[m], lim=self.lim[m],
            series=self.series[m], exptime_min=self.exptime_min[m], blend=self.blend[m],
            reject=self.reject[m], plate=self.plate[m], t_nd=self.t_nd, lim_nd=self.lim_nd,
            series_nd=self.series_nd, exptime_nd=self.exptime_nd,
            flags_applied=self.flags_applied, n_raw=self.n_raw, columns=self.columns,
            exptime_unit=self.exptime_unit,
        )

    # ---- (de)serialisation for the shard archive --------------------------
    def to_arrays(self) -> dict:
        return {
            "t": self.t.astype(float), "mag": self.mag.astype(float),
            "err": self.err.astype(float), "lim": self.lim.astype(float),
            "series": self.series.astype(str), "exptime_min": self.exptime_min.astype(float),
            "blend": self.blend.astype(bool), "reject": self.reject.astype(bool),
            "plate": self.plate.astype(str), "t_nd": self.t_nd.astype(float),
            "lim_nd": self.lim_nd.astype(float), "series_nd": self.series_nd.astype(str),
            "exptime_nd": self.exptime_nd.astype(float),
        }

    @classmethod
    def from_arrays(cls, d: dict, *, flags_applied: bool = False, n_raw: int = 0,
                    columns=None, exptime_unit: str = "unknown") -> CenturyLC:
        g = lambda k, dt: np.asarray(d[k], dtype=dt) if k in d else np.array([], dtype=dt)  # noqa: E731
        return cls(
            t=g("t", float), mag=g("mag", float), err=g("err", float), lim=g("lim", float),
            series=g("series", object), exptime_min=g("exptime_min", float),
            blend=g("blend", bool), reject=g("reject", bool), plate=g("plate", object),
            t_nd=g("t_nd", float), lim_nd=g("lim_nd", float), series_nd=g("series_nd", object),
            exptime_nd=g("exptime_nd", float), flags_applied=bool(flags_applied),
            n_raw=int(n_raw), columns=list(columns or []), exptime_unit=str(exptime_unit),
        )


def from_api_frame(df: pd.DataFrame, aflags: FlagDefs | None = None,
                   bflags: FlagDefs | None = None, *, exptime_unit: str = "auto",
                   default_err: float = 0.15) -> CenturyLC | None:
    """Normalise a raw ``lightcurve`` answer.  ``None`` if no time/magnitude columns."""
    if df is None or not len(df):
        return None
    tcol = pick_column(df, TIME_COLS)
    mcol = pick_column(df, MAG_COLS)
    if tcol is None or mcol is None:
        return None
    t = _time_to_mjd(numeric(df, tcol), str(tcol))
    mag = numeric(df, mcol)
    err = numeric(df, pick_column(df, ERR_COLS))
    lim = numeric(df, pick_column(df, LIM_COLS))
    exptime = numeric(df, pick_column(df, EXPTIME_COLS))
    unit = exptime_unit
    if unit == "auto":
        fin = exptime[np.isfinite(exptime) & (exptime > 0)]
        # Harvard patrol exposures were tens of minutes: a median above ~400
        # can only be seconds.
        unit = ("seconds" if (fin.size and np.median(fin) > 400.0)
                else ("minutes" if fin.size else "unknown"))
    if unit == "seconds":
        exptime = exptime / 60.0
    scol = pick_column(df, SERIES_COLS)
    series = (df[scol].astype(str).to_numpy() if scol is not None
              else np.full(len(df), "unknown", dtype=object))
    pn = pick_column(df, PLATENUM_COLS)
    mn = pick_column(df, MOSNUM_COLS)
    en = pick_column(df, EXPNUM_COLS)
    parts = [series.astype(str)]
    for c in (pn, mn, en):
        if c is not None:
            parts.append(df[c].astype(str).to_numpy())
    plate = np.array(["_".join(p) for p in zip(*parts, strict=False)], dtype=object)

    af = numeric(df, pick_column(df, AFLAG_COLS), default=0.0)
    bf = numeric(df, pick_column(df, BFLAG_COLS), default=0.0)
    a = aflags if aflags is not None else FlagDefs("aflags")
    b = bflags if bflags is not None else FlagDefs("bflags")
    blend, reject, applied = apply_masks(af, bf, a, b)

    # Magnitudes outside any physical range mark non-detections in some
    # encodings; NaN/null in others.  Treat all of them as censored.
    det = np.isfinite(mag) & (mag > 0.0) & (mag < 30.0) & np.isfinite(t)
    nd = ~det & np.isfinite(t)
    err_det = err[det]
    err_det = np.where(np.isfinite(err_det) & (err_det > 0), err_det, float(default_err))

    return CenturyLC(
        t=t[det], mag=mag[det], err=err_det, lim=lim[det], series=series[det].astype(object),
        exptime_min=exptime[det], blend=blend[det], reject=reject[det],
        plate=plate[det], t_nd=t[nd], lim_nd=lim[nd], series_nd=series[nd].astype(object),
        exptime_nd=exptime[nd], flags_applied=applied, n_raw=int(len(df)),
        columns=[str(c) for c in df.columns], exptime_unit=unit,
    )


# ---------------------------------------------------------------------------
# Calendar blocks that carry their censoring
# ---------------------------------------------------------------------------


@dataclass
class CenturyBlock:
    """A calendar block of a plate light curve, with its own non-detections."""

    index: int
    t: np.ndarray
    y: np.ndarray
    e: np.ndarray
    lim: np.ndarray
    exptime_min: np.ndarray
    t_nd: np.ndarray
    lim_nd: np.ndarray
    exptime_nd: np.ndarray
    series: np.ndarray
    blend_frac: float = float("nan")
    year_lo: float = float("nan")
    year_hi: float = float("nan")

    @property
    def n(self) -> int:
        return int(len(self.t))

    @property
    def n_nd(self) -> int:
        return int(len(self.t_nd))

    @property
    def t_mid(self) -> float:
        return float(np.median(self.t)) if self.n else float("nan")

    @property
    def year_mid(self) -> float:
        return float(mjd_to_year(self.t_mid)) if self.n else float("nan")

    @property
    def t_span(self) -> float:
        return float(np.ptp(self.t)) if self.n > 1 else 0.0

    @property
    def mean_mag(self) -> float:
        return float(np.median(self.y)) if self.n else float("nan")

    @property
    def median_err(self) -> float:
        return float(np.median(self.e)) if self.n else float("nan")

    @property
    def median_lim(self) -> float:
        v = np.concatenate([self.lim, self.lim_nd])
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    def dominant_series(self) -> str:
        if not self.n:
            return ""
        vals, cnt = np.unique(self.series.astype(str), return_counts=True)
        return str(vals[int(np.argmax(cnt))])

    def as_knell_block(self):
        from ..knell.blocks import Block
        return Block(index=self.index, t=self.t, y=self.y, e=self.e)


def calendar_blocks(lc: CenturyLC, *, block_years: float = 2.0, origin_year: float = 1880.0,
                    min_epochs_block: int = 20, min_blocks: int = 4,
                    mask: np.ndarray | None = None) -> list[CenturyBlock]:
    """Fixed calendar blocks of ``block_years`` from ``origin_year``.

    Thin blocks are dropped, not merged (merging would smear a transition).
    The non-detections falling in a kept block ride along, so an injection in
    that block can be censored against the plates that *were* taken.  A blend
    fraction per block is computed from all usable detections in the block,
    whether or not they pass ``mask`` --- it is the block's blending history.
    Returns ``[]`` if fewer than ``min_blocks`` survive.
    """
    m = lc.good if mask is None else (lc.good & np.asarray(mask, dtype=bool))
    if not m.any():
        return []
    yr = lc.year
    idx_all = np.floor((yr - float(origin_year)) / float(block_years)).astype(int)
    idx_nd = np.floor((lc.year_nd - float(origin_year)) / float(block_years)).astype(int) \
        if lc.n_nd else np.array([], dtype=int)
    usable = lc.usable
    out: list[CenturyBlock] = []
    for k in np.unique(idx_all[m]):
        sel = m & (idx_all == k)
        if int(sel.sum()) < int(min_epochs_block):
            continue
        order = np.argsort(lc.t[sel])
        nd = idx_nd == k
        ub = usable & (idx_all == k)
        bf = float(np.mean(lc.blend[ub])) if ub.any() else float("nan")
        out.append(CenturyBlock(
            index=int(k), t=lc.t[sel][order], y=lc.mag[sel][order], e=lc.err[sel][order],
            lim=lc.lim[sel][order], exptime_min=lc.exptime_min[sel][order],
            t_nd=lc.t_nd[nd], lim_nd=lc.lim_nd[nd], exptime_nd=lc.exptime_nd[nd],
            series=lc.series[sel][order], blend_frac=bf,
            year_lo=float(origin_year + k * block_years),
            year_hi=float(origin_year + (k + 1) * block_years),
        ))
    if len(out) < int(min_blocks):
        return []
    return out


# ---------------------------------------------------------------------------
# Annual tables --- the input to the fade statistic and the field ensemble
# ---------------------------------------------------------------------------


def annual_table(lc: CenturyLC, *, mask: np.ndarray | None = None, min_per_year: int = 3
                 ) -> pd.DataFrame:
    """Per calendar year: median magnitude, its error, count, median limit,
    blend fraction and dominant series.

    The error on the median is ``1.2533 * sigma / sqrt(n)`` with ``sigma`` the
    robust (MAD) scatter of that year, floored at the median reported error over
    ``sqrt(n)`` --- so a year with three identical values cannot acquire
    infinite weight.
    """
    m = lc.good if mask is None else (lc.good & np.asarray(mask, dtype=bool))
    if not m.any():
        return pd.DataFrame(columns=["year", "n", "med", "err", "mad", "lim_med",
                                     "blend_frac", "series"])
    yr = np.floor(lc.year).astype(int)
    usable = lc.usable
    rows = []
    for y in np.unique(yr[m]):
        sel = m & (yr == y)
        n = int(sel.sum())
        if n < int(min_per_year):
            continue
        v = lc.mag[sel]
        med = float(np.median(v))
        mad = float(1.4826 * np.median(np.abs(v - med)))
        e_rep = float(np.median(lc.err[sel]))
        err = float(max(1.2533 * mad, e_rep) / np.sqrt(n))
        lim = lc.lim[sel]
        lim = lim[np.isfinite(lim)]
        ub = usable & (yr == y)
        bf = float(np.mean(lc.blend[ub])) if ub.any() else float("nan")
        vals, cnt = np.unique(lc.series[sel].astype(str), return_counts=True)
        rows.append({"year": int(y), "n": n, "med": med, "err": err, "mad": mad,
                     "lim_med": float(np.median(lim)) if lim.size else float("nan"),
                     "blend_frac": bf, "series": str(vals[int(np.argmax(cnt))])})
    return pd.DataFrame(rows)


def smear_factor(freq_per_day, exptime_min) -> np.ndarray:
    """``|sinc(f * t_exp)|``: the amplitude a sinusoid of frequency ``f`` keeps
    after a top-hat exposure of ``t_exp``.  NaN exposure -> 1 (unmodelled)."""
    tau = np.asarray(exptime_min, dtype=float) / 1440.0
    x = float(freq_per_day) * tau
    out = np.ones_like(x)
    ok = np.isfinite(x) & (x > 0)
    out[ok] = np.abs(np.sinc(x[ok]))       # numpy sinc is sin(pi x)/(pi x)
    return out


__all__ = ["CenturyBlock", "CenturyLC", "annual_table", "calendar_blocks", "from_api_frame",
           "mjd_to_year", "smear_factor", "year_to_mjd"]
