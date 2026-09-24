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
# The error must belong to the MAGNITUDE that was used.  DR7 serves several
# calibrations side by side (``magcal_magdep``, ``magcal_local``, ``magcal_iso``)
# each with its own ``*_rms``; pairing ``magcal_magdep`` with ``magcal_local_rms``
# attaches one calibration's scatter to another's photometry.  ``ERR_FOR_MAG``
# is consulted first, ``ERR_COLS`` only when the magnitude column is unknown.
ERR_FOR_MAG = {
    "magcal_magdep": ("magcal_magdep_rms",),
    "magcal_local": ("magcal_local_rms", "magcal_local_error"),
    "magcal_iso": ("magcal_iso_rms",),
}
ERR_COLS = ("magcal_magdep_rms", "magcal_local_rms", "magcal_rms", "mag_err", "magerr", "err")
# daschlab masks ``magcal_magdep_rms == 99.0`` (``extra_mq(..., 99.0)`` in
# ``daschlab/photometry.py``): 99 is DASCH's "no rms available" sentinel, not a
# 99-magnitude error bar, and carried through as a number it would give such a
# point essentially zero weight for ever after instead of the default error.
ERR_SENTINELS = (99.0, -99.0, 9999.0)
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


def any_time_to_year(vals) -> np.ndarray:
    """Decimal year from a column that may be a JD, an MJD or already a year.

    DR7 speaks all three: ``queryexps`` returns ``epoch`` as a decimal year
    (verified on the runner, run 35738717013) while the light curves carry a
    JD-scale ``date``.  The scales are two and four orders of magnitude apart,
    so the median decides without ambiguity --- and a column that is none of
    them comes back as NaN rather than as a plausible wrong century.
    """
    v = np.asarray(vals, dtype=float)
    finite = v[np.isfinite(v)]
    if not finite.size:
        return np.full(v.shape, np.nan)
    med = float(np.nanmedian(finite))
    if med > 2.0e6:                       # Julian Date
        return mjd_to_year(v - JD_MJD_OFFSET)
    if 1.0e4 < med < 1.0e6:               # Modified Julian Date
        return mjd_to_year(v)
    if 1700.0 < med < 2200.0:             # already a decimal year
        return v
    return np.full(v.shape, np.nan)


def _time_to_mjd(vals: np.ndarray, colname: str) -> np.ndarray:
    """Interpret a time column: JD-scale numbers (> 2.3e6) become MJD, and a
    decimal year (1700--2200) becomes an MJD too."""
    v = np.asarray(vals, dtype=float)
    name = colname.lower()
    finite = v[np.isfinite(v)]
    if "mjd" in name and "jd" in name and not name.startswith("date_jd"):
        return v
    if finite.size and np.nanmedian(finite) > 2.0e6:
        return v - JD_MJD_OFFSET
    if finite.size and 1700.0 < float(np.nanmedian(finite)) < 2200.0:
        return year_to_mjd(v)
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
    plate_nd: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
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

    def subset(self, mask: np.ndarray, mask_nd: np.ndarray | None = None) -> CenturyLC:
        """The light curve restricted to ``mask`` (detections).

        ``mask_nd`` restricts the NON-detections too, and a caller that
        restricts by plate series must pass it.  The non-detections are the
        censoring: they set the limits an injection is evaluated against, so
        carrying every series' non-detections into a single-series re-run would
        censor that series' injection with another telescope's plate depths,
        which is precisely the confounder the single-series re-run exists to
        remove.
        """
        m = np.asarray(mask, dtype=bool)
        mnd = (np.ones(self.n_nd, dtype=bool) if mask_nd is None
               else np.asarray(mask_nd, dtype=bool))
        return CenturyLC(
            t=self.t[m], mag=self.mag[m], err=self.err[m], lim=self.lim[m],
            series=self.series[m], exptime_min=self.exptime_min[m], blend=self.blend[m],
            reject=self.reject[m], plate=self.plate[m], t_nd=self.t_nd[mnd],
            lim_nd=self.lim_nd[mnd], series_nd=self.series_nd[mnd],
            exptime_nd=self.exptime_nd[mnd],
            plate_nd=(self.plate_nd[mnd] if self.plate_nd.size == self.n_nd
                      else self.plate_nd),
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
            "plate_nd": self.plate_nd.astype(str),
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
            exptime_nd=g("exptime_nd", float), plate_nd=g("plate_nd", object),
            flags_applied=bool(flags_applied),
            n_raw=int(n_raw), columns=list(columns or []), exptime_unit=str(exptime_unit),
        )


def str_values(col) -> np.ndarray:
    """A column as an object array of Python strings, ``""`` where missing.

    Not ``Series.astype(str)``: under pandas 3 that returns the new ``str``
    dtype, which KEEPS missing values as NaN (pandas 2 wrote ``"nan"``).  Run
    35862579322 lost all 14 sweep shards to exactly that --- one DR7 light
    curve with an empty ``mosnum`` made ``"_".join`` meet a float and raise on
    the first star of every shard.  Integer-valued floats (``123.0``, what a
    CSV column with one gap reads back as) are written without the ``.0``.
    """
    out = []
    for v in (col.tolist() if hasattr(col, "tolist") else list(col)):
        if v is None or (isinstance(v, float) and not np.isfinite(v)) or v is pd.NA \
                or v is pd.NaT:
            out.append("")
        elif isinstance(v, (float, np.floating)) and float(v).is_integer():
            out.append(str(int(v)))
        else:
            out.append(str(v))
    return np.array(out, dtype=object)


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
    ecol = pick_column(df, ERR_FOR_MAG.get(str(mcol).lower(), ())) or pick_column(df, ERR_COLS)
    err = numeric(df, ecol)
    for s in ERR_SENTINELS:
        err = np.where(np.isclose(err, s), np.nan, err)
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
    series = (str_values(df[scol]) if scol is not None
              else np.full(len(df), "unknown", dtype=object))
    pn = pick_column(df, PLATENUM_COLS)
    mn = pick_column(df, MOSNUM_COLS)
    en = pick_column(df, EXPNUM_COLS)
    parts = [series.astype(str)]
    for c in (pn, mn, en):
        if c is not None:
            parts.append(str_values(df[c]))
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
        exptime_nd=exptime[nd], plate_nd=plate[nd], flags_applied=applied, n_raw=int(len(df)),
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


def plate_ids(plate_strings) -> pd.DataFrame:
    """``series, platenum, mosnum, expnum`` recovered from a ``plate`` label.

    ``from_api_frame`` builds the label as ``series_platenum[_mosnum[_expnum]]``
    from whichever identifier columns the archive served.
    """
    parts = [str(p).split("_") for p in np.asarray(plate_strings, dtype=object)]

    def _at(i):
        return pd.to_numeric(pd.Series([p[i] if len(p) > i else None for p in parts]),
                             errors="coerce").astype("Int64")

    return pd.DataFrame({"series": [p[0].strip().lower() if p else "" for p in parts],
                         "platenum": _at(1), "mosnum": _at(2), "expnum": _at(3)})


def attach_exptime(lc: CenturyLC, table: pd.DataFrame) -> dict:
    """Fill ``exptime_min`` / ``exptime_nd`` from a ``queryexps`` exposure table.

    DR7 light curves carry **no exposure time** --- the served columns are
    fixed by ``_COLTYPES`` in ``daschlab/photometry.py`` and ``exptime`` is not
    among them --- while ``queryexps`` carries it, in minutes.  The channel
    needs it because a long exposure smears a periodic signal by
    ``|sinc(f t_exp)|``, and Harvard exposure lengths are a property of the
    plate SERIES, which is clustered in calendar time.  Left unmodelled, the
    injection efficiency measured in a star's post-gap blocks would be computed
    as if those plates smeared the signal exactly as much as the pre-gap plates
    did; where the later series exposed longer, that overstates how well the
    late plates could have seen the period --- and that number is the whole
    content of an efficiency-normalised non-detection.

    One ``queryexps`` per FIELD serves every star in it, so this costs no
    request.  The join key is ``(series, platenum, mosnum, expnum)`` where the
    light curve names all four, falling back to ``(series, platenum)`` --- a
    plate all of whose exposures ran the same length.  It never falls back to
    the series alone: that would hand every plate its series' typical exposure
    and so manufacture the series-clustered smear history the channel exists to
    distinguish from a real change in the star.

    Mutates ``lc`` and returns a provenance dict.
    """
    prov = {"n_det": int(lc.n_det), "n_nd": int(lc.n_nd), "matched_det": 0, "matched_nd": 0,
            "key": "none", "table_rows": int(0 if table is None else len(table)),
            "exptime_min_pre": float("nan"), "exptime_min_post": float("nan")}
    if table is None or not len(table) or "exptime_min" not in table.columns:
        return prov
    tab = table.copy()
    tab["series"] = pd.Series(str_values(tab["series"]), index=tab.index).str.strip().str.lower()
    # The table makes a round trip through plate_exptime.csv, and a CSV column
    # with one empty cell reads back as float64 while `plate_ids` produces
    # Int64.  Merging those two dtypes matches nothing and would look like a
    # DASCH coverage gap rather than a dtype mismatch, so both sides are
    # normalised to Int64 here.
    for c in ("platenum", "mosnum", "expnum"):
        if c in tab.columns:
            tab[c] = pd.to_numeric(tab[c], errors="coerce").astype("Int64")
    full = [c for c in ("series", "platenum", "mosnum", "expnum") if c in tab.columns]
    keys = [k for k in (full, ["series", "platenum"]) if len(k) > 1]

    def _join_on(labels, key) -> np.ndarray:
        """``exptime_min`` per label under one key, NaN where it did not match."""
        left = plate_ids(labels)
        if any(c not in left.columns or left[c].isna().all() for c in key if c != "series"):
            return np.full(len(labels), np.nan)
        red = tab.drop_duplicates(subset=key, keep="first")[[*key, "exptime_min"]]
        return left.merge(red, on=key, how="left")["exptime_min"].to_numpy(dtype=float)

    def _best(labels) -> tuple[np.ndarray, str]:
        """The key that matches MOST of ``labels``; ties go to the more specific.

        Taking the first key that matches *anything* would keep the four-part
        key even when the two endpoints spell mosnum/expnum differently and it
        matches a handful of plates, silently discarding the rest.  The
        comparison is on how many plates each key actually places.
        """
        if labels is None or not len(labels):
            return np.array([], dtype=float), "none"
        best_v, best_key, best_n = np.full(len(labels), np.nan), "none", 0
        for key in keys:
            v = _join_on(labels, key)
            n = int(np.isfinite(v).sum())
            if n > best_n:
                best_v, best_key, best_n = v, "+".join(key), n
        return best_v, best_key

    det_exp, key = _best(lc.plate)
    if key == "none":
        return prov
    prov["key"] = key
    # The non-detections are joined on the SAME key as the detections: two
    # different keys would give the two halves of one light curve two different
    # exposure-time provenances, which is a difference the smear would then
    # carry into the efficiency.
    key_cols = key.split("+")

    def _join(labels) -> tuple[np.ndarray, str]:
        if labels is None or not len(labels):
            return np.array([], dtype=float), "none"
        return _join_on(labels, key_cols), key
    if det_exp.size == lc.n_det:
        lc.exptime_min = np.where(np.isfinite(det_exp), det_exp,
                                  lc.exptime_min if lc.exptime_min.size == lc.n_det
                                  else np.nan)
        prov["matched_det"] = int(np.isfinite(det_exp).sum())
    if lc.plate_nd.size == lc.n_nd and lc.n_nd:
        nd_exp, _ = _join(lc.plate_nd)
        if nd_exp.size == lc.n_nd:
            lc.exptime_nd = np.where(np.isfinite(nd_exp), nd_exp,
                                     lc.exptime_nd if lc.exptime_nd.size == lc.n_nd
                                     else np.nan)
            prov["matched_nd"] = int(np.isfinite(nd_exp).sum())
    if prov["matched_det"] or prov["matched_nd"]:
        lc.exptime_unit = "minutes"
        e, yr = lc.exptime_min, lc.year
        pre = np.isfinite(e) & (yr < 1954.0)
        post = np.isfinite(e) & (yr >= 1970.0)
        if pre.any():
            prov["exptime_min_pre"] = float(np.median(e[pre]))
        if post.any():
            prov["exptime_min_post"] = float(np.median(e[post]))
    return prov


def smear_factor(freq_per_day, exptime_min) -> np.ndarray:
    """``|sinc(f * t_exp)|``: the amplitude a sinusoid of frequency ``f`` keeps
    after a top-hat exposure of ``t_exp``.  NaN exposure -> 1 (unmodelled)."""
    tau = np.asarray(exptime_min, dtype=float) / 1440.0
    x = float(freq_per_day) * tau
    out = np.ones_like(x)
    ok = np.isfinite(x) & (x > 0)
    out[ok] = np.abs(np.sinc(x[ok]))       # numpy sinc is sin(pi x)/(pi x)
    return out


__all__ = ["CenturyBlock", "CenturyLC", "annual_table", "any_time_to_year", "attach_exptime",
           "calendar_blocks", "from_api_frame", "mjd_to_year", "plate_ids", "smear_factor",
           "year_to_mjd"]
