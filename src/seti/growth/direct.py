"""GROWTH stage ``direct`` --- the TESS depth of EVERY Kepler planet, from the pixels' light curves.

Why this stage exists
---------------------
Stage 1 (:mod:`seti.growth.drift`) reached **108 of 9,564 KOIs** because it
joined the Kepler catalogue to the TESS era through the TOI *alert* catalogue:
3,055 KOIs resolve a TIC id, but only 94 of their 1,975 distinct stars ever had
a TOI at all.  TOI is not a re-measurement of Kepler's planets, so leaning on it
cost 99 % of the sample **and** imported whichever pipeline produced each alert
--- the heterogeneity that broke stage 1's error model (31 of 108 planets above
5 sigma of the population median).  Stage 2 fixed the error model for ONE
object by measuring both eras from the light curves, and stage 3 tracked that
object (Kepler-718 b) down to a TESS PDC crowding-correction artefact with a
difference-image source 29 arcsec off target.

This stage is the scale step named in ``STATUS.md``: *measuring TESS depths
directly from the TESS-SPOC / QLP light curves fixes both the sample loss and
the heterogeneous error model at once.*  For **every confirmed or candidate
KOI with a TIC id**, it

1. fetches every TESS light-curve product MAST serves for the star (SPOC
   2-minute, TESS-SPOC and QLP FFI), **once per star** so multi-planet systems
   are not re-downloaded per planet;
2. folds on the KOI ephemeris propagated to the TESS epoch with its accumulated
   error, and **searches a small window in phase** for the transit (an
   ephemeris propagated over ~2,000 epochs can drift by more than the
   transit; the offset, its significance and the number of trials are all on
   the record);
3. fits the depth with the **stage-2 fitter** (:func:`seti.growth.stage2.measure_target`
   --- same masked baseline, same exposure-shrunk core, same bootstrap, same
   :func:`seti.growth.stage2.dedupe_sectors`) on BOTH the raw ``SAP_FLUX`` and
   the corrected ``PDCSAP_FLUX`` family, with the spread over pipeline authors
   within each family entering the error as a systematic;
4. profiles the transit **duration** on the detrended fold, so a depth change
   can be tested against the fixed-impact-parameter prediction
   (:func:`seti.growth.drift.duration_test`);
5. compares each family with the KOI depth carried into the TESS band, and
   states **per planet what depth change the comparison could have seen**
   (``*_detectable_ln_ratio``, ``*_detectable_depth_change_ppm``), so a null on
   a star is an honest statement about that star's TESS precision.

The lesson of Kepler-718 b, as a rule
-------------------------------------
A bigger aperture admits more contaminating light, so on the same star the
**raw TESS SAP depth must read shallower than Kepler, never deeper**.  A star
whose ``PDCSAP`` depth grew while its ``SAP`` depth did not is a
**crowding-correction case** (``crowding_correction``), not growth: PDC divided
the raw depth by a crowding factor the TIC got wrong.  A candidate must
therefore change at >= ``n_candidate`` sigma on the TOTAL error in **both** the
SAP and the PDCSAP family, in the same direction, with the duration consistent
with a fixed impact parameter, no odd-even signature, the transit recovered at
its ephemeris, and no KOI false-positive flag.  Every survivor is then handed
to stage 2 (both eras fitted alike, the reduction ensemble) and stage 3 (the
Gaia census and the difference image) by the ``vet`` stage.

What this stage does NOT claim
------------------------------
* A ``growth_candidate`` is a reason to run the vet, not a detection.
* ``consistent`` on a star is a count at that star's stated sensitivity, not a
  limit on construction; ``not_measurable`` says TESS could not have seen the
  KOI depth at all, and neither is written up (CLAUDE.md).
* ``not_measured`` keeps ``QUERY_FAILED`` (the archive errored),
  ``QUERY_RETURNED_ZERO_ROWS`` (it answered with nothing), ``TIC_UNRESOLVED``,
  ``EPHEMERIS_UNAVAILABLE``, ``NO_USABLE_TRANSIT``, ``BUDGET_EXHAUSTED`` and
  ``NOT_REACHED`` apart.  A shard the clock killed leaves its targets
  ``NOT_REACHED``, never ``consistent``.

Everything that touches a service is injectable (``query_fn`` for the
Exoplanet Archive, ``products_fn`` for MAST, ``tic_fn`` for the TIC fallback)
and every network stage carries a wall-clock budget.  Nothing here reaches the
network inside a test.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .acquire import (
    KOI_COLUMNS,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_ZERO,
    TIC_ROUTES,
    AcquisitionLog,
    fetch_ps_kepler,
    koi_name_keys,
    prepare_ps,
    resolve_koi_tics,
    table_query,
    tap_sync,
)
from .drift import DUR_FIXED_B, DUR_INCONCLUSIVE, band_ratio, duration_test, sym_err
from .stage2 import (
    BG_DISAGREE,
    BG_PDC_DEEPER,
    BTJD_OFFSET,
    ERA_TESS,
    Deadline,
    EnsembleParams,
    FitParams,
    combine_transit_depths,
    dedupe_sectors,
    epoch_in_era,
    fit_transits,
    fold,
    mast_probe,
    measure_target,
    propagate_epoch,
    reduction_spread,
    sap_vs_pdcsap,
    total_depth_error,
    trapezoid_transit,
)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
#: The two flux FAMILIES.  ``pdc`` is the pipeline-corrected photometry (PDCSAP
#: for SPOC / TESS-SPOC; KSPSAP or DET for QLP); ``sap`` is the raw aperture sum.
FAMILY_PDC = "pdc"
FAMILY_SAP = "sap"
FAMILIES = (FAMILY_PDC, FAMILY_SAP)
#: Column priority within each family (the first one a product carries wins).
PDC_COLUMNS: tuple[str, ...] = ("PDCSAP_FLUX", "KSPSAP_FLUX", "DET_FLUX")
SAP_COLUMNS: tuple[str, ...] = ("SAP_FLUX",)
ALL_FLUX_COLUMNS: tuple[str, ...] = PDC_COLUMNS + SAP_COLUMNS
FAMILY_COLUMNS = {FAMILY_PDC: PDC_COLUMNS, FAMILY_SAP: SAP_COLUMNS}

#: Pipeline authors at MAST, best cadence first (the dedupe tie-break order).
DEFAULT_AUTHORS: tuple[str, ...] = ("SPOC", "TESS-SPOC", "QLP")

#: Per-target classes.
CLASS_GROWTH = "growth_candidate"
CLASS_SHRINK = "shrink_candidate"
CLASS_DEEPER = "deeper_tess"
CLASS_SHALLOWER = "shallower_tess"
CLASS_CONSISTENT = "consistent"
CLASS_CROWDING = "crowding_correction"
CLASS_NOT_RECOVERED = "transit_not_recovered"
CLASS_NOT_MEASURABLE = "not_measurable"
CLASS_NOT_MEASURED = "not_measured"
CLASSES = (CLASS_GROWTH, CLASS_SHRINK, CLASS_DEEPER, CLASS_SHALLOWER, CLASS_CONSISTENT,
           CLASS_CROWDING, CLASS_NOT_RECOVERED, CLASS_NOT_MEASURABLE, CLASS_NOT_MEASURED)
CANDIDATE_CLASSES = (CLASS_GROWTH, CLASS_SHRINK)

#: ``not_measured`` reasons --- all different facts, none collapsed.
REASON_QUERY_FAILED = STATUS_FAILED
REASON_ZERO_ROWS = STATUS_ZERO
REASON_TIC_UNRESOLVED = "TIC_UNRESOLVED"
REASON_NO_EPHEMERIS = "EPHEMERIS_UNAVAILABLE"
REASON_NO_TRANSIT = "NO_USABLE_TRANSIT"
REASON_BUDGET = "BUDGET_EXHAUSTED"
REASON_NOT_REACHED = "NOT_REACHED"
REASON_NO_REFERENCE = "NO_REFERENCE_DEPTH"
REASONS = (REASON_QUERY_FAILED, REASON_ZERO_ROWS, REASON_TIC_UNRESOLVED, REASON_NO_EPHEMERIS,
           REASON_NO_TRANSIT, REASON_BUDGET, REASON_NOT_REACHED, REASON_NO_REFERENCE)

#: Run verdicts.
RUN_NO_DATA = "NO_DATA_REACHED"
RUN_NONE = "NO_DEPTH_CHANGE_CANDIDATE"
RUN_CANDIDATES = "DEPTH_CHANGE_CANDIDATES"
RUN_VERDICTS = (RUN_NO_DATA, RUN_NONE, RUN_CANDIDATES)

#: Vetoes a would-be candidate can carry.
VETO_ODD_EVEN = "odd_even_significant"
VETO_DURATION = "duration_not_fixed_b"
VETO_DURATION_INCONCLUSIVE = "duration_inconclusive"
VETO_FPFLAG = "koi_fpflag"
VETO_DISPOSITION = "koi_false_positive"
VETO_EPHEMERIS = "ephemeris_not_recovered"
VETO_ONE_FAMILY = "one_family_only"
VETO_LOWER_BOUND = "depth_is_lower_bound"
VETOES = (VETO_ODD_EVEN, VETO_DURATION, VETO_DURATION_INCONCLUSIVE, VETO_FPFLAG,
          VETO_DISPOSITION, VETO_EPHEMERIS, VETO_ONE_FAMILY, VETO_LOWER_BOUND)

#: Extra TIC-resolution routes tried in the shard when the ``ps`` routes failed.
TIC_FALLBACK_ROUTES: tuple[str, ...] = ("tic_kic_crossid", "tic_region_kic", "tic_region_nearest")

STAGES = ("probe", "targets", "measure", "assess", "control", "vet")

#: The KOI columns the direct stage needs on top of stage 1's list: the epoch
#: (``koi_time0bk`` is NOT in the stage-1 list) and the period's second error.
DIRECT_KOI_COLUMNS: tuple[str, ...] = tuple(dict.fromkeys(
    KOI_COLUMNS + ("koi_period_err2", "koi_time0bk", "koi_time0bk_err1", "koi_time0bk_err2")))

ROUTE_LIGHTKURVE = "lightkurve"
ROUTE_MAST_FITS = "astroquery_mast_fits"


# ---------------------------------------------------------------------------
# Parameters (all from config/growth.yaml, ``direct:``)
# ---------------------------------------------------------------------------
def _sub(conf: dict | None, *keys) -> dict:
    d = (conf or {})
    for k in keys:
        d = (d or {}).get(k) or {}
    return d if isinstance(d, dict) else {}


def _set_from(d, m: dict) -> None:
    for k in d.__dataclass_fields__:
        if k in m and m[k] is not None:
            cur = getattr(d, k)
            if isinstance(cur, tuple):
                setattr(d, k, tuple(str(v) for v in m[k]))
            elif isinstance(cur, bool):
                setattr(d, k, bool(m[k]))
            elif isinstance(cur, int):
                setattr(d, k, int(m[k]))
            elif isinstance(cur, float):
                setattr(d, k, float(m[k]))
            elif cur is None:
                setattr(d, k, m[k])
            else:
                setattr(d, k, type(cur)(m[k]))


@dataclass
class TargetParams:
    """Which KOIs are targets."""

    dispositions: tuple[str, ...] = ("CONFIRMED", "CANDIDATE")
    pos_radius_arcsec: float = 2.0
    long_period_min_days: float = 30.0
    archive_timeout_s: float = 600.0
    archive_retries: int = 3
    #: Keep KOIs with no ``ps`` TIC in the list so the shard can try the TIC
    #: fallback routes; False restricts the list to the ``ps``-resolved ones.
    include_unresolved: bool = True

    @classmethod
    def from_config(cls, conf: dict | None) -> TargetParams:
        d = cls()
        _set_from(d, _sub(conf, "direct", "targets"))
        return d


@dataclass
class FetchParams:
    """How the TESS products are reached, and the ceilings on reaching them."""

    authors: tuple[str, ...] = DEFAULT_AUTHORS
    max_products: int = 45
    #: 20-second products are several times the size of the 120-second ones
    #: and carry the same transits; they are skipped, and the skip is counted.
    min_exptime_s: float = 60.0
    quality_bitmask: str = "default"
    per_target_budget_s: float = 300.0
    shard_budget_s: float = 16200.0        # 4.5 h inside a 5 h job
    retries: int = 2
    retry_pause_s: float = 5.0
    download_dir: str | None = None
    clean_downloads: bool = True
    tic_fallback: bool = True
    tic_radius_arcsec: float = 6.0
    tic_mag_tolerance: float = 1.5

    @classmethod
    def from_config(cls, conf: dict | None) -> FetchParams:
        d = cls()
        _set_from(d, _sub(conf, "direct", "fetch"))
        return d


@dataclass
class EpochSearchParams:
    """The window in phase searched for the transit, and how the offset binds."""

    n_sigma: float = 3.0                   # half-width = n_sigma * sigma_T0 + min half-width
    min_halfwidth_durations: float = 0.25
    max_halfwidth_days: float = 0.6
    max_halfwidth_periods: float = 0.3
    coarse_step_durations: float = 0.25
    fine_step_durations: float = 1.0 / 12.0
    max_trials: int = 161
    min_snr: float = 3.0                   # below this the transit is NOT recovered
    #: |offset| beyond this many propagated sigmas is flagged (a real transit
    #: can still sit there --- TTVs --- but the flag is on the record).
    flag_offset_sigma: float = 5.0
    #: Phases (in units of the period) at which the SAME max-over-trials search
    #: is repeated where no transit is --- the look-elsewhere null.  0 and 0.5
    #: are excluded: the transit and the secondary eclipse live there.
    control_phases: tuple = (0.17, 0.27, 0.37, 0.63, 0.73, 0.83)

    @classmethod
    def from_config(cls, conf: dict | None) -> EpochSearchParams:
        d = cls()
        _set_from(d, _sub(conf, "direct", "epoch_search"))
        return d


@dataclass
class DurationParams:
    """The duration profile fit on the detrended fold."""

    window_durations: float = 2.5
    guard_durations: float = 1.25
    factor_min: float = 0.4
    factor_max: float = 2.5
    n_factors: int = 61
    oversample: int = 9
    default_ingress_fraction: float = 0.15
    sigma_dur_sys: float = 0.05
    n_duration: float = 3.0

    @classmethod
    def from_config(cls, conf: dict | None) -> DurationParams:
        d = cls()
        _set_from(d, _sub(conf, "direct", "duration"))
        return d


@dataclass
class ClassifyParams:
    """The gates."""

    n_candidate: float = 5.0
    n_background: float = 3.0              # SAP-vs-PDCSAP disagreement
    sigma_sys_ln: float = 0.05
    min_expected_snr: float = 3.0          # KOI depth / TESS error below this: not_measurable
    min_measured_snr_for_ln: float = 2.0   # below this z falls back to the linear form
    apply_band_ratio: bool = True
    subtract_population_median: bool = True
    scale_by_population_scatter: bool = True
    min_population_for_offset: int = 20
    odd_even_sigma: float = 3.0
    dispositions: tuple[str, ...] = ("CONFIRMED", "CANDIDATE")
    vet_max_targets: int = 12

    @classmethod
    def from_config(cls, conf: dict | None) -> ClassifyParams:
        d = cls()
        _set_from(d, _sub(conf, "direct", "classify"))
        return d


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _b(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in ("true", "1", "yes")


def _s(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def _write(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    os.replace(tmp, path)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.8g")
    os.replace(tmp, path)


def _read_csv(path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or not p.stat().st_size:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:                                     # noqa: BLE001
        return pd.DataFrame()


def shard_of(kepid, n_shards: int) -> int:
    """The shard a star belongs to.  By ``kepid``, so a multi-planet system's
    planets share one shard and one download."""
    try:
        k = int(float(kepid))
    except (TypeError, ValueError):
        k = 0
    return int(k % max(int(n_shards), 1))


# ---------------------------------------------------------------------------
# MAST: every product for a star, ALL flux columns, ONE download each
# ---------------------------------------------------------------------------
def read_tess_lc_fits_all(path, *, columns=ALL_FLUX_COLUMNS) -> dict | None:
    """One TESS light-curve FITS product -> time (BTJD) + EVERY flux column it carries.

    The stage-2 reader returns one column; the direct stage needs ``SAP_FLUX``
    and ``PDCSAP_FLUX`` from the same file without downloading it twice.
    ``TIME`` goes through the file's own ``BJDREFI``/``BJDREFF``; ``QUALITY != 0``
    is masked.
    """
    from astropy.io import fits  # noqa: PLC0415

    with fits.open(str(path), memmap=False) as hdul:
        hdu = None
        for h in hdul:
            try:
                names = [str(n).upper() for n in h.columns.names]
            except AttributeError:
                continue
            if "TIME" in names:
                hdu = h
                break
        if hdu is None:
            return None
        data, hdr = hdu.data, hdu.header
        cols = [str(n).upper() for n in hdu.columns.names]
        present = [c for c in columns if c in cols]
        if not present:
            return None
        bjdrefi = float(hdr.get("BJDREFI", hdul[0].header.get("BJDREFI", BTJD_OFFSET)) or
                        BTJD_OFFSET)
        bjdreff = float(hdr.get("BJDREFF", hdul[0].header.get("BJDREFF", 0.0)) or 0.0)
        t = np.asarray(data["TIME"], dtype=float) + (bjdrefi + bjdreff) - BTJD_OFFSET
        keep = np.isfinite(t)
        if "QUALITY" in cols:
            keep &= (np.asarray(data["QUALITY"]) == 0)
        t = t[keep]
        fluxes = {}
        for c in present:
            f = np.asarray(data[c], dtype=float)[keep]
            ecol = c + "_ERR"
            fe = (np.asarray(data[ecol], dtype=float)[keep] if ecol in cols
                  else np.full(f.shape, np.nan))
            fluxes[c] = (f, fe)
        exptime = float("nan")
        try:
            v = hdr.get("TIMEDEL")
            if v is not None and np.isfinite(float(v)) and float(v) > 0:
                exptime = float(v) * 86400.0
        except Exception:                                 # noqa: BLE001
            pass
        if not np.isfinite(exptime) and t.size > 2:
            exptime = float(np.median(np.diff(np.sort(t)))) * 86400.0
        sector = hdul[0].header.get("SECTOR", hdr.get("SECTOR"))
        return {"sector": int(sector) if sector is not None else None,
                "author": str(hdul[0].header.get("PROCVER", "") or "unknown"),
                "exptime_s": exptime, "time": t, "fluxes": fluxes, "n_points": int(t.size)}


def _author_rank(author: str, authors) -> int:
    a = str(author).upper().replace("_", "-")
    for i, x in enumerate(authors):
        if a == str(x).upper().replace("_", "-"):
            return i
    return len(authors)


def _float_array(x) -> np.ndarray:
    """A plain float array from a Quantity / Column / masked column (masked -> NaN)."""
    try:
        if hasattr(x, "filled"):
            x = x.filled(np.nan)
    except Exception:                                     # noqa: BLE001
        pass
    x = getattr(x, "value", x)
    try:
        if hasattr(x, "filled"):
            x = x.filled(np.nan)
    except Exception:                                     # noqa: BLE001
        pass
    return np.asarray(x, dtype=float)


def lightcurve_to_product(lc, *, fallback_sector=None, fallback_author: str = "") -> dict | None:
    """One ``lightkurve`` LightCurve -> a product record with EVERY flux column it carries.

    The time axis is ``Time.jd - 2457000`` (BTJD) rather than the object's own
    format string; ``SECTOR`` / ``AUTHOR`` / ``TIMEDEL`` come from the meta,
    with the search-table values as the fallback.  A product carrying none of
    the family columns returns ``None``.
    """
    try:
        t = _float_array(lc.time.jd) - BTJD_OFFSET
    except Exception:                                     # noqa: BLE001
        t = _float_array(lc.time.value)
    cols = {str(c).lower(): str(c) for c in getattr(lc, "columns", [])}
    fluxes = {}
    for col in ALL_FLUX_COLUMNS:
        c = col.lower()
        if c not in cols:
            continue
        try:
            f = _float_array(lc[cols[c]])
        except Exception:                                 # noqa: BLE001
            continue
        if f.shape != t.shape or not np.any(np.isfinite(f)):
            continue
        fe = np.full(f.shape, np.nan)
        if c + "_err" in cols:
            try:
                fe = _float_array(lc[cols[c + "_err"]])
            except Exception:                             # noqa: BLE001
                pass
        fluxes[col] = (f, fe)
    if not fluxes:
        return None
    meta = dict(getattr(lc, "meta", {}) or {})
    exptime = meta.get("TIMEDEL")
    exptime_s = float(exptime) * 86400.0 if exptime else float("nan")
    if not np.isfinite(exptime_s) and t.size > 2:
        exptime_s = float(np.median(np.diff(np.sort(t)))) * 86400.0
    sector = meta.get("SECTOR")
    try:
        sector = int(sector) if sector is not None else (
            int(fallback_sector) if fallback_sector is not None else None)
    except (TypeError, ValueError):
        sector = None
    return {"sector": sector, "author": str(meta.get("AUTHOR") or fallback_author or "unknown"),
            "exptime_s": exptime_s, "time": t, "fluxes": fluxes, "n_points": int(t.size)}


def lightkurve_products_fn(tic_id, *, authors=DEFAULT_AUTHORS, max_products: int = 45,
                           min_exptime_s: float = 60.0, download_dir: str | None = None,
                           quality_bitmask: str = "default", **_kw) -> list[dict]:
    """Every TESS light-curve product for one TIC through ``lightkurve`` (runner only).

    The author filter is **strict**: a request for ``SPOC``/``TESS-SPOC``/``QLP``
    that matches nothing returns nothing, never another pipeline's products
    (the ``AUTHOR_NOT_SERVED`` lesson of run ``bcc670c``).  Each product is
    downloaded once and every flux column it carries is kept.
    """
    import lightkurve as lk  # noqa: PLC0415

    sr = lk.search_lightcurve(f"TIC {int(tic_id)}", mission="TESS")
    if sr is None or len(sr) == 0:
        return []
    tbl = sr.table
    auth = np.array([str(a) for a in tbl["author"]])
    try:
        expt = np.array([float(e) for e in tbl["exptime"]])
    except Exception:                                     # noqa: BLE001
        expt = np.full(len(auth), np.nan)
    try:
        sec = np.array([int(s) for s in tbl["sequence_number"]])
    except Exception:                                     # noqa: BLE001
        sec = np.zeros(len(auth), dtype=int)
    want = {str(a).upper() for a in authors}
    keep = np.array([a.upper() in want for a in auth])
    keep &= ~(np.isfinite(expt) & (expt < float(min_exptime_s)))
    idx = [i for i in range(len(auth)) if keep[i]]
    idx.sort(key=lambda i: (_author_rank(auth[i], authors), sec[i], expt[i]))
    idx = idx[:int(max_products)]
    out: list[dict] = []
    for i in idx:
        try:
            lc = sr[i].download(download_dir=download_dir, quality_bitmask=quality_bitmask)
        except Exception:                                 # noqa: BLE001
            continue
        if lc is None:
            continue
        rec = lightcurve_to_product(lc, fallback_sector=sec[i], fallback_author=auth[i])
        if rec is not None:
            out.append(rec)
    return out


def mast_fits_products_fn(tic_id, *, authors=DEFAULT_AUTHORS, max_products: int = 45,
                          min_exptime_s: float = 60.0, download_dir: str | None = None,
                          **_kw) -> list[dict]:
    """The same through ``astroquery.mast`` + FITS when ``lightkurve`` is absent."""
    from astroquery.mast import Observations  # noqa: PLC0415

    obs = Observations.query_criteria(obs_collection=["TESS", "HLSP"],
                                      dataproduct_type="timeseries",
                                      target_name=str(int(tic_id)))
    if obs is None or len(obs) == 0:
        return []
    prod = Observations.get_product_list(obs)
    if prod is None or len(prod) == 0:
        return []
    p = prod.to_pandas()
    if "productSubGroupDescription" in p.columns:
        p = p[p["productSubGroupDescription"].astype(str).str.upper() == "LC"]
    if "provenance_name" in p.columns:
        want = {str(a).upper() for a in authors}
        p = p[p["provenance_name"].astype(str).str.upper().isin(want)]
    if not len(p):
        return []
    p = p.assign(_rank=[_author_rank(a, authors) for a in p["provenance_name"].astype(str)])
    p = p.sort_values(["_rank", "productFilename"]).head(int(max_products))
    root = Path(download_dir or ".") / "mast_lc_direct"
    root.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for _, r in p.iterrows():
        uri = str(r.get("dataURI") or "")
        if not uri:
            continue
        local = root / str(r.get("productFilename") or Path(uri).name)
        try:
            status, _msg, _url = Observations.download_file(uri, local_path=str(local),
                                                            cache=False)
            if str(status).upper() != "COMPLETE" or not local.exists():
                continue
            rec = read_tess_lc_fits_all(local)
        except Exception:                                 # noqa: BLE001
            continue
        if rec is None:
            continue
        if np.isfinite(rec["exptime_s"]) and rec["exptime_s"] < float(min_exptime_s):
            continue
        prov = str(r.get("provenance_name") or "").strip()
        if prov:
            rec["author"] = prov
        out.append(rec)
    return out


def default_products_fn(tic_id, **kw) -> list[dict]:
    """``lightkurve`` if it is installed, else ``astroquery.mast`` + FITS."""
    try:
        import lightkurve  # noqa: PLC0415, F401

        return lightkurve_products_fn(tic_id, **kw)
    except ImportError:
        return mast_fits_products_fn(tic_id, **kw)


def fetch_products(tic_id, *, products_fn=None, params: FetchParams | None = None,
                   log: AcquisitionLog | None = None, deadline: Deadline | None = None,
                   key: str = "") -> tuple[list[dict], str, str]:
    """One star's TESS products, bounded and recorded.  ``(products, status, route)``."""
    params = params or FetchParams()
    log = log or AcquisitionLog(prefix="growth/direct")
    label = f"products_{key or tic_id}"
    query = f"TIC {tic_id}"
    if deadline is not None and deadline.expired():
        log.record(label, query, error="budget_exhausted_before_request")
        return [], STATUS_FAILED, ""
    route = "injected" if products_fn is not None else (mast_probe().get("route_preferred") or "")
    fn = products_fn or default_products_fn
    own = Deadline(budget_s=float(params.per_target_budget_s))
    last = ""
    for attempt in range(max(int(params.retries), 1)):
        if own.expired() or (deadline is not None and deadline.expired()):
            last = "budget_exhausted"
            break
        if attempt and float(params.retry_pause_s) > 0:
            _time.sleep(min(float(params.retry_pause_s) * attempt, own.remaining(),
                            deadline.remaining() if deadline is not None else float("inf")))
        try:
            prods = fn(tic_id, authors=tuple(params.authors), max_products=int(params.max_products),
                       min_exptime_s=float(params.min_exptime_s),
                       download_dir=params.download_dir, quality_bitmask=params.quality_bitmask)
        except Exception as exc:                          # noqa: BLE001
            last = repr(exc)[:400]
            continue
        prods = [p for p in (prods or []) if p is not None and p.get("fluxes")
                 and len(np.asarray(p.get("time"), dtype=float))]
        if not prods:
            log.record(label, query, rows=0, extra={"route": route, "attempt": attempt + 1})
            return [], STATUS_ZERO, route
        log.record(label, query, rows=int(sum(int(p.get("n_points") or 0) for p in prods)),
                   extra={"route": route, "n_products": len(prods), "attempt": attempt + 1})
        return prods, STATUS_OK, route
    log.record(label, query, error=last or "unknown")
    return [], STATUS_FAILED, route


# ---------------------------------------------------------------------------
# TIC fallback (runner only; injectable)
# ---------------------------------------------------------------------------
def astroquery_tic_fn(kepid, ra, dec, kepmag, *, radius_arcsec: float = 6.0,
                      mag_tolerance: float = 1.5) -> tuple[float, str]:
    """A TIC id for a KIC star through the TIC itself.  ``(tic_id, route)``.

    The TIC carries the KIC cross-id, so the first route asks for it by name;
    the second cones the KOI position and keeps the row whose ``KIC`` matches;
    the third takes the nearest source whose ``Tmag`` is within
    ``mag_tolerance`` of ``kepmag`` and is recorded as such (a crowded-field
    nearest match is a weaker identification, and the route says so).
    """
    from astroquery.mast import Catalogs  # noqa: PLC0415

    kid = int(float(kepid))
    try:
        t = Catalogs.query_criteria(catalog="Tic", KIC=kid)
        if t is not None and len(t):
            return float(t["ID"][0]), "tic_kic_crossid"
    except Exception:                                     # noqa: BLE001
        pass
    if not (np.isfinite(_f(ra)) and np.isfinite(_f(dec))):
        return float("nan"), ""
    from astropy import units as u  # noqa: PLC0415
    from astropy.coordinates import SkyCoord  # noqa: PLC0415

    t = Catalogs.query_region(SkyCoord(float(ra), float(dec), unit="deg"),
                              radius=float(radius_arcsec) * u.arcsec, catalog="Tic")
    if t is None or not len(t):
        return float("nan"), ""
    df = t.to_pandas()
    if "KIC" in df.columns:
        kk = pd.to_numeric(df["KIC"], errors="coerce")
        hit = df[kk == kid]
        if len(hit):
            return float(hit["ID"].iloc[0]), "tic_region_kic"
    if "dstArcSec" in df.columns:
        df = df.sort_values("dstArcSec")
    if "Tmag" in df.columns and np.isfinite(_f(kepmag)):
        tm = pd.to_numeric(df["Tmag"], errors="coerce")
        ok = df[(tm - float(kepmag)).abs() <= float(mag_tolerance)]
        if len(ok):
            return float(ok["ID"].iloc[0]), "tic_region_nearest"
    return float("nan"), ""


# ---------------------------------------------------------------------------
# Segments per family
# ---------------------------------------------------------------------------
def family_segments(products, family: str, *, per_column: bool = False) -> list[dict]:
    """Stage-2-shaped segments for one family.

    With ``per_column=False`` each product contributes ONE segment, using the
    first family column it carries (``PDCSAP`` over ``KSPSAP`` over ``DET``);
    that set, deduplicated by sector, is the primary measurement.  With
    ``per_column=True`` every family column a product carries becomes its own
    segment, which is what the per-(author, column) members are built from.
    """
    cols = FAMILY_COLUMNS[family]
    out: list[dict] = []
    for p in products or []:
        fl = p.get("fluxes") or {}
        present = [c for c in cols if c in fl]
        if not present:
            continue
        for c in (present if per_column else present[:1]):
            f, fe = fl[c]
            out.append({"sector": p.get("sector"), "author": p.get("author"),
                        "exptime_s": p.get("exptime_s"), "flux_column": c,
                        "time": np.asarray(p.get("time"), dtype=float),
                        "flux": np.asarray(f, dtype=float),
                        "flux_err": np.asarray(fe, dtype=float),
                        "n_points": int(len(np.asarray(p.get("time"))))})
    return out


# ---------------------------------------------------------------------------
# The detrended fold (for the duration fit)
# ---------------------------------------------------------------------------
def detrended_fold(segments, *, period_days: float, t0_btjd: float, duration_days: float,
                   params: FitParams | None = None, window_durations: float = 2.5,
                   guard_durations: float = 1.25) -> dict:
    """Every point within ``window`` of a predicted transit, divided by its own
    transit-masked local linear baseline --- the stage-2 baseline, wider.

    Returns ``dt`` (days), ``flux`` (ratio to baseline), ``flux_err``,
    ``exptime_days`` per point and the transit count.  The window and guard
    are wider than the depth fitter's because a duration that GREW must not
    have its ingress inside the baseline mask.
    """
    params = params or FitParams()
    dts, fs, fes, exps = [], [], [], []
    n_used = 0
    for s in segments or []:
        t = np.asarray(s.get("time"), dtype=float)
        f = np.asarray(s.get("flux"), dtype=float)
        fe = (np.asarray(s.get("flux_err"), dtype=float) if s.get("flux_err") is not None
              else np.full(t.shape, np.nan))
        ok = np.isfinite(t) & np.isfinite(f)
        t, f, fe = t[ok], f[ok], fe[ok]
        if not t.size:
            continue
        med = float(np.median(f))
        if not np.isfinite(med) or med == 0:
            continue
        f, fe = f / med, fe / med
        exptime_s = _f(s.get("exptime_s"))
        exp_d = exptime_s / 86400.0 if np.isfinite(exptime_s) and exptime_s > 0 else 0.0
        epoch, dt = fold(t, period_days, t0_btjd)
        w_guard = float(guard_durations) * float(duration_days)
        w_out = float(window_durations) * float(duration_days)
        if exp_d > 0:
            w_out = max(w_out, w_guard + (0.5 * int(params.min_baseline_points) + 1.0) * exp_d)
        for ep in np.unique(epoch):
            m = (epoch == ep) & (np.abs(dt) <= w_out)
            if not np.any(m):
                continue
            d_i, f_i, e_i = dt[m], f[m], fe[m]
            base = np.abs(d_i) >= w_guard
            if base.sum() < int(params.min_baseline_points):
                continue
            if params.require_baseline_both_sides and (not np.any(base & (d_i < 0))
                                                       or not np.any(base & (d_i > 0))):
                continue
            wb = np.where(np.isfinite(e_i[base]) & (e_i[base] > 0),
                          1.0 / np.maximum(e_i[base], 1e-30) ** 2, 1.0)
            sw = float(np.sum(wb))
            sx = float(np.sum(wb * d_i[base]))
            sy = float(np.sum(wb * f_i[base]))
            sxx = float(np.sum(wb * d_i[base] ** 2))
            sxy = float(np.sum(wb * d_i[base] * f_i[base]))
            den = sw * sxx - sx * sx
            if abs(den) < 1e-30:
                a, b = sy / sw, 0.0
            else:
                b = (sw * sxy - sx * sy) / den
                a = (sy - b * sx) / sw
            pred = a + b * d_i
            if not np.all(np.isfinite(pred)) or np.any(pred == 0):
                continue
            resid = f_i[base] / pred[base] - 1.0
            mad = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826
            sig = mad if mad > 0 else float(np.std(resid, ddof=1)) if resid.size > 1 else np.nan
            dts.append(d_i)
            fs.append(f_i / pred)
            fes.append(np.full(d_i.shape, sig if np.isfinite(sig) and sig > 0 else np.nan))
            exps.append(np.full(d_i.shape, exp_d))
            n_used += 1
    if not dts:
        return {"dt": np.array([]), "flux": np.array([]), "flux_err": np.array([]),
                "exptime_days": np.array([]), "n_transits": 0}
    return {"dt": np.concatenate(dts), "flux": np.concatenate(fs),
            "flux_err": np.concatenate(fes), "exptime_days": np.concatenate(exps),
            "n_transits": int(n_used)}


# ---------------------------------------------------------------------------
# The ephemeris-offset search
# ---------------------------------------------------------------------------
def _snr_at(segments, *, period_days, t0_btjd, duration_days, params: FitParams) -> tuple:
    rows = []
    for s in segments:
        t = np.asarray(s["time"], dtype=float)
        f = np.asarray(s["flux"], dtype=float)
        fe = np.asarray(s.get("flux_err"), dtype=float) if s.get("flux_err") is not None else None
        ok = np.isfinite(t) & np.isfinite(f)
        if not ok.any():
            continue
        med = float(np.median(f[ok]))
        if not np.isfinite(med) or med == 0:
            continue
        exptime_s = _f(s.get("exptime_s"))
        exp_d = exptime_s / 86400.0 if np.isfinite(exptime_s) else float("nan")
        rows.extend(fit_transits(t[ok], f[ok] / med, (fe[ok] / med) if fe is not None else None,
                                 period_days=period_days, t0_btjd=t0_btjd,
                                 duration_days=duration_days, exptime_days=exp_d,
                                 sector=s.get("sector"), params=params))
    c = combine_transit_depths(rows, params=params)
    d, e = c["depth_ppm"], c["depth_err_analytic_ppm"]
    snr = d / e if np.isfinite(d) and np.isfinite(e) and e > 0 else float("nan")
    return snr, d, e, c["n_transits"]


def search_epoch_offset(segments, *, period_days: float, t0_btjd: float, duration_days: float,
                        sigma_t0_days: float, fit: FitParams | None = None,
                        params: EpochSearchParams | None = None) -> dict:
    """Find the transit near the propagated epoch: a coarse-then-fine scan of the
    epoch offset, scored by the fitted depth's signal-to-noise.

    The search exists because ``koi_time0bk`` propagated over ~2,000 periods
    can miss the TESS transit by more than its duration, and a fold that
    misses the transit reads as a shallow depth --- i.e. as *shrinkage*.  The
    offset, the number of trials, the best signal-to-noise and the depth at the
    nominal ephemeris are all reported.  **The maximum over N trials is
    biased**: a transit is only called recovered above ``min_snr``, and a
    candidate must separately clear the depth gate on the total error.
    """
    fit = fit or FitParams()
    params = params or EpochSearchParams()
    import dataclasses  # noqa: PLC0415

    fast = dataclasses.replace(fit, err_method="analytic", bootstrap_draws=8)
    dur = float(duration_days)
    sig = float(sigma_t0_days) if np.isfinite(sigma_t0_days) else 0.0
    half = params.n_sigma * sig + params.min_halfwidth_durations * dur
    half = max(half, params.min_halfwidth_durations * dur)
    half = min(half, params.max_halfwidth_days, params.max_halfwidth_periods * float(period_days))
    coarse = max(params.coarse_step_durations * dur, 1e-4)
    n_side = int(math.floor(half / coarse))
    if 2 * n_side + 1 > int(params.max_trials):
        n_side = (int(params.max_trials) - 1) // 2
        coarse = half / max(n_side, 1)
    offsets = np.arange(-n_side, n_side + 1) * coarse
    out = {"epoch_offset_days": 0.0, "epoch_offset_minutes": 0.0,
           "epoch_search_halfwidth_minutes": half * 1440.0,
           "epoch_search_n_trials": 0, "epoch_search_snr_best": float("nan"),
           "epoch_search_snr_nominal": float("nan"),
           "depth_at_nominal_ephemeris_ppm": float("nan"),
           "depth_at_best_offset_ppm": float("nan"),
           "ephemeris_recovered": False, "epoch_offset_sigma": float("nan")}
    if not segments:
        return out
    best = (-np.inf, 0.0, float("nan"))
    n_trials = 0
    snr_nom = float("nan")
    for off in offsets:
        snr, d, _e, _n = _snr_at(segments, period_days=period_days, t0_btjd=t0_btjd + off,
                                 duration_days=dur, params=fast)
        n_trials += 1
        if abs(off) < 1e-9:
            snr_nom, out["depth_at_nominal_ephemeris_ppm"] = snr, d
        if np.isfinite(snr) and snr > best[0]:
            best = (snr, float(off), d)
    if not np.isfinite(snr_nom):
        snr_nom, d, _e, _n = _snr_at(segments, period_days=period_days, t0_btjd=t0_btjd,
                                     duration_days=dur, params=fast)
        out["depth_at_nominal_ephemeris_ppm"] = d
        n_trials += 1
    fine = max(params.fine_step_durations * dur, 1e-4)
    if np.isfinite(best[0]) and fine < coarse:
        lo, hi = best[1] - coarse, best[1] + coarse
        for off in np.arange(lo, hi + 0.5 * fine, fine):
            if abs(off) > half + 1e-9 or abs(off - best[1]) < 1e-9:
                continue
            snr, d, _e, _n = _snr_at(segments, period_days=period_days, t0_btjd=t0_btjd + off,
                                     duration_days=dur, params=fast)
            n_trials += 1
            if np.isfinite(snr) and snr > best[0]:
                best = (snr, float(off), d)
    out.update({"epoch_search_n_trials": int(n_trials),
                "epoch_search_snr_best": float(best[0]) if np.isfinite(best[0]) else float("nan"),
                "epoch_search_snr_nominal": float(snr_nom),
                "depth_at_best_offset_ppm": float(best[2])})
    recovered = bool(np.isfinite(best[0]) and best[0] >= float(params.min_snr))
    out["ephemeris_recovered"] = recovered
    if recovered:
        out["epoch_offset_days"] = float(best[1])
        out["epoch_offset_minutes"] = float(best[1]) * 1440.0
        out["epoch_offset_sigma"] = (abs(best[1]) / sig) if sig > 0 else float("nan")
    return out


#: The control-phase verdicts.
CTRL_ABOVE = "ABOVE_CONTROL"
CTRL_WITHIN = "WITHIN_SEARCH_NOISE"
CTRL_UNAVAILABLE = "CONTROL_UNAVAILABLE"
CTRL_VERDICTS = (CTRL_ABOVE, CTRL_WITHIN, CTRL_UNAVAILABLE)


def control_phase_null(products, family: str, *, period_days: float, t0_btjd: float,
                       duration_days: float, sigma_t0_days: float, depth_ppm: float,
                       ref_ppm: float, fit: FitParams | None = None,
                       params: EpochSearchParams | None = None) -> dict:
    """Repeat the epoch search where no transit is, and report what it finds.

    :func:`search_epoch_offset` takes the **maximum** of the fitted signal-to-
    noise over up to ``max_trials`` epoch offsets, and the depth is then fitted
    at the winning offset.  A maximum over trials is biased upward, and the bias
    grows as the signal weakens --- exactly the regime in which a spurious
    *deeper* TESS depth would be manufactured.  Nothing in the depth's formal
    error knows about it.

    So the identical search --- same half-width, same coarse and fine steps,
    same fitter, same segments --- is run centred on ``control_phases`` of the
    period, where the planet is not.  The best signal-to-noise and the deepest
    depth it finds there are what the search produces from noise alone on
    *this* star.  ``snr_excess`` and ``depth_excess_over_control_ppm`` say by
    how much the transit beat its own null.

    Caveats kept on the record, not buried: in a multi-planet system a control
    phase can land on a *sibling's* transit, which makes the null conservative,
    not permissive; and a control phase can land in a data gap, in which case
    that phase returns nothing and is not counted.
    """
    fit = fit or FitParams()
    params = params or EpochSearchParams()
    out = {"control_family": family, "control_n_phases": 0, "control_n_phases_measured": 0,
           "control_phases": "", "control_snr_max": float("nan"),
           "control_snr_median": float("nan"), "control_depth_max_ppm": float("nan"),
           "control_depth_median_ppm": float("nan"), "snr_excess": float("nan"),
           "depth_excess_over_control_ppm": float("nan"),
           "control_verdict": CTRL_UNAVAILABLE}
    segs = family_segments(products, family)
    segs, _dropped = dedupe_sectors(segs)
    phases = tuple(float(p) for p in (params.control_phases or ()))
    out["control_n_phases"] = len(phases)
    out["control_phases"] = ",".join(f"{p:g}" for p in phases)
    if not segs or not phases or not (np.isfinite(period_days) and period_days > 0):
        return out
    snrs, depths = [], []
    for ph in phases:
        es = search_epoch_offset(segs, period_days=period_days,
                                 t0_btjd=t0_btjd + ph * float(period_days),
                                 duration_days=duration_days, sigma_t0_days=sigma_t0_days,
                                 fit=fit, params=params)
        s, d = _f(es.get("epoch_search_snr_best")), _f(es.get("depth_at_best_offset_ppm"))
        if np.isfinite(s):
            snrs.append(float(s))
        if np.isfinite(d):
            depths.append(float(d))
    out["control_n_phases_measured"] = len(snrs)
    if not snrs:
        return out
    out["control_snr_max"] = float(np.max(snrs))
    out["control_snr_median"] = float(np.median(snrs))
    if depths:
        out["control_depth_max_ppm"] = float(np.max(depths))
        out["control_depth_median_ppm"] = float(np.median(depths))
    return out


def apply_control(rec: dict, ctrl: dict, *, family: str) -> dict:
    """Fold one family's control-phase null into the record's own numbers.

    ``snr_excess`` is the transit's search signal-to-noise minus the best the
    same search reached off-transit.  ``depth_excess_over_control_ppm`` is the
    measured depth's excess over the Kepler reference minus the deepest depth
    the search manufactured from noise: a "growth" smaller than that is not a
    growth, it is the search.
    """
    out = dict(ctrl)
    snr_t = _f(rec.get("epoch_search_snr_best"))
    smax = _f(out.get("control_snr_max"))
    if np.isfinite(snr_t) and np.isfinite(smax):
        out["snr_excess"] = float(snr_t - smax)
    d = _f(rec.get(f"{family}_depth_ppm"))
    ref = _f(rec.get("koi_depth_in_tess_band_ppm"))
    dmax = _f(out.get("control_depth_max_ppm"))
    if np.isfinite(d) and np.isfinite(ref) and np.isfinite(dmax):
        out["depth_excess_over_control_ppm"] = float((d - ref) - max(dmax, 0.0))
    if not np.isfinite(smax):
        out["control_verdict"] = CTRL_UNAVAILABLE
    elif np.isfinite(out.get("snr_excess", float("nan"))) and out["snr_excess"] > 0:
        out["control_verdict"] = CTRL_ABOVE
    else:
        out["control_verdict"] = CTRL_WITHIN
    return out


# ---------------------------------------------------------------------------
# The duration profile fit
# ---------------------------------------------------------------------------
def ingress_fraction(k: float, b: float, default: float = 0.15) -> float:
    """``tau / T14`` for a circular orbit: ``(1 - T23/T14) / 2``; ``default`` when unknown."""
    if not (np.isfinite(k) and np.isfinite(b)) or k <= 0:
        return float(default)
    c14 = (1.0 + k) ** 2 - b * b
    c23 = (1.0 - k) ** 2 - b * b
    if c14 <= 0:
        return float(default)
    t23 = math.sqrt(max(c23, 0.0) / c14)
    return float(min(max(0.5 * (1.0 - t23), 0.02), 0.5))


def fit_duration(fold_rec: dict, *, duration_days: float, ingress_frac: float,
                 params: DurationParams | None = None) -> dict:
    """Profile the total duration on the detrended fold: a trapezoid of fixed
    ingress fraction, integrated over each point's exposure, depth solved
    linearly at each trial duration, ``T14`` from the chi-square minimum and
    its error from ``delta chi2 = 1`` on the reduced-chi-square-scaled profile.
    """
    params = params or DurationParams()
    out = {"t14_tess_hours": float("nan"), "t14_tess_err_hours": float("nan"),
           "t14_factor": float("nan"), "t14_factor_err": float("nan"),
           "duration_fit_depth_ppm": float("nan"), "duration_fit_chi2_per_dof": float("nan"),
           "duration_fit_n_points": 0, "duration_fit_status": "not_fitted",
           "duration_at_grid_edge": False}
    dt = np.asarray(fold_rec.get("dt"), dtype=float)
    f = np.asarray(fold_rec.get("flux"), dtype=float)
    fe = np.asarray(fold_rec.get("flux_err"), dtype=float)
    ex = np.asarray(fold_rec.get("exptime_days"), dtype=float)
    ok = np.isfinite(dt) & np.isfinite(f)
    dt, f, fe, ex = dt[ok], f[ok], fe[ok], ex[ok]
    n = int(dt.size)
    out["duration_fit_n_points"] = n
    if n < 20 or not (np.isfinite(duration_days) and duration_days > 0):
        out["duration_fit_status"] = "too_few_points"
        return out
    w = np.where(np.isfinite(fe) & (fe > 0), 1.0 / np.maximum(fe, 1e-30) ** 2, 1.0)
    sub = np.linspace(-0.5, 0.5, max(int(params.oversample), 1))
    exv = np.where(np.isfinite(ex) & (ex > 0), ex, 0.0)
    grid = dt[:, None] + sub[None, :] * exv[:, None]
    factors = np.exp(np.linspace(math.log(params.factor_min), math.log(params.factor_max),
                                 max(int(params.n_factors), 5)))
    chi2 = np.full(factors.shape, np.nan)
    depths = np.full(factors.shape, np.nan)
    y = 1.0 - f
    for i, fac in enumerate(factors):
        T = fac * float(duration_days)
        m = (1.0 - trapezoid_transit(grid, T, 1.0, ingress_frac)).mean(axis=1)
        smm = float(np.sum(w * m * m))
        if smm <= 0:
            continue
        d = float(np.sum(w * y * m) / smm)
        chi2[i] = float(np.sum(w * (y - d * m) ** 2))
        depths[i] = d
    if not np.any(np.isfinite(chi2)):
        out["duration_fit_status"] = "fit_failed"
        return out
    i0 = int(np.nanargmin(chi2))
    dof = max(n - 2, 1)
    scale = max(float(chi2[i0]) / dof, 1.0)
    prof = (chi2 - chi2[i0]) / scale
    inside = np.isfinite(prof) & (prof <= 1.0)
    idx = np.where(inside)[0]
    lo, hi = factors[idx.min()], factors[idx.max()]
    edge = bool(idx.min() == 0 or idx.max() == len(factors) - 1)
    err_fac = 0.5 * (hi - lo)
    if err_fac <= 0:
        # The minimum sits between two grid points: half a grid step.
        step = factors[min(i0 + 1, len(factors) - 1)] - factors[max(i0 - 1, 0)]
        err_fac = 0.5 * float(step)
    out.update({"t14_tess_hours": float(factors[i0] * duration_days * 24.0),
                "t14_tess_err_hours": float(err_fac * duration_days * 24.0),
                "t14_factor": float(factors[i0]), "t14_factor_err": float(err_fac),
                "duration_fit_depth_ppm": float(depths[i0]) * 1e6,
                "duration_fit_chi2_per_dof": float(chi2[i0]) / dof,
                "duration_fit_status": STATUS_OK, "duration_at_grid_edge": edge})
    return out


# ---------------------------------------------------------------------------
# One family, measured
# ---------------------------------------------------------------------------
_FAMILY_FIELDS: tuple[str, ...] = (
    "depth_ppm", "depth_err_ppm", "depth_err_method", "depth_err_analytic_ppm",
    "depth_err_bootstrap_ppm", "n_transits", "n_transits_seen", "oot_scatter_ppm",
    "n_sectors", "n_sectors_measured", "sector_list", "authors", "exptimes_s", "smeared",
    "depth_is_lower_bound", "odd_even_diff_ppm", "odd_even_sigma", "sector_scatter_chi2_per_dof",
    "n_sectors_dropped_duplicate", "flags", "chi2_per_dof",
)


def measure_family(products, family: str, *, period_days: float, t0_btjd: float,
                   duration_days: float, fit: FitParams | None = None,
                   ensemble: EnsembleParams | None = None) -> tuple[dict, list[dict], dict | None]:
    """The primary (deduplicated) depth of one family and its per-(author, column)
    members from the SAME downloaded products.  ``(record, members, measure_target_result)``.

    The primary depth is the stage-2 measurement: one sector counted once, the
    shortest cadence winning.  The members are each pipeline's own reduction of
    the same sectors; their spread (:func:`seti.growth.stage2.reduction_spread`)
    enters the error as a systematic and **never moves the depth**.
    """
    fit = fit or FitParams()
    ensemble = ensemble or EnsembleParams()
    pre = f"{family}_"
    rec: dict = {f"{pre}status": "FLUX_COLUMN_NOT_PRESENT", f"{pre}flux_column": "",
                 f"{pre}primary_author": ""}
    for k in _FAMILY_FIELDS:
        rec[pre + k] = float("nan") if k not in ("depth_err_method", "sector_list", "authors",
                                                  "exptimes_s", "flags") else ""
    for k in ("n_transits", "n_transits_seen", "n_sectors", "n_sectors_measured",
              "n_sectors_dropped_duplicate"):
        rec[pre + k] = 0
    for k in ("smeared", "depth_is_lower_bound"):
        rec[pre + k] = False
    rec.update({pre + k: v for k, v in reduction_spread([], params=ensemble).items()})
    rec[pre + "total_err_ppm"] = float("nan")
    segs = family_segments(products, family)
    if not segs:
        return rec, [], None
    m = measure_target(segs, period_days=period_days, t0_btjd=t0_btjd,
                       duration_days=duration_days, params=fit)
    for k in _FAMILY_FIELDS:
        rec[pre + k] = m.get(k)
    sec = m.get("sectors")
    rec[pre + "flux_column"] = (str(sec["flux_column"].iloc[0]) if sec is not None and len(sec)
                                and "flux_column" in sec else "")
    rec[pre + "primary_author"] = str(m.get("authors") or "").split(",")[0]
    rec[pre + "status"] = STATUS_OK if (m["n_transits"] and np.isfinite(m["depth_ppm"])) \
        else REASON_NO_TRANSIT
    # --- the members: every (author, column) the products carry ------------
    groups: dict = {}
    for s in family_segments(products, family, per_column=True):
        groups.setdefault((str(s["author"]), str(s["flux_column"])), []).append(s)
    members: list[dict] = []
    for (author, col), group in groups.items():
        mm = measure_target(group, period_days=period_days, t0_btjd=t0_btjd,
                            duration_days=duration_days, params=fit)
        okm = bool(mm["n_transits"] and np.isfinite(mm["depth_ppm"]))
        members.append({"family": family, "author": author, "flux_column": col,
                        "status": STATUS_OK if okm else REASON_NO_TRANSIT,
                        "n_segments": int(mm["n_sectors"]), "segment_list": mm["sector_list"],
                        "exptimes_s": mm["exptimes_s"], "n_transits": int(mm["n_transits"]),
                        "depth_ppm": float(mm["depth_ppm"]),
                        "depth_err_ppm": float(mm["depth_err_ppm"]),
                        "depth_err_method": str(mm["depth_err_method"]),
                        "oot_scatter_ppm": float(mm["oot_scatter_ppm"]),
                        "odd_even_sigma": float(mm["odd_even_sigma"]),
                        "smeared": bool(mm["smeared"]),
                        "is_primary_reduction": bool(author == rec[pre + "primary_author"]
                                                     and col == rec[pre + "flux_column"])})
    sp = reduction_spread(members, primary_depth_ppm=float(m["depth_ppm"]), params=ensemble)
    rec.update({pre + k: v for k, v in sp.items()})
    rec[pre + "total_err_ppm"] = total_depth_error(float(m["depth_err_ppm"]),
                                                   _f(sp.get("depth_reduction_spread_ppm")))
    return rec, members, m


# ---------------------------------------------------------------------------
# The comparison with the Kepler era, and the sensitivity
# ---------------------------------------------------------------------------
def compare_family(depth_ppm: float, stat_err_ppm: float, total_err_ppm: float,
                   ref_ppm: float, ref_err_ppm: float, *, params: ClassifyParams | None = None,
                   n_candidate: float | None = None) -> dict:
    """``ln(D / D_ref)`` with its sigma, ``z``, and what this comparison COULD see.

    ``z`` uses the total error.  When the measured depth is too small for the
    log form (``D <= 0`` or ``D / err < min_measured_snr_for_ln``) the linear
    ``z_lin = (D - D_ref) / sigma_ppm`` is used instead and ``z_method`` says
    so --- a vanished transit must still get a number.  The sensitivity is
    stated against the REFERENCE depth (``detectable_ln_ratio =
    n_candidate * sigma_expected`` with the TESS fractional error taken as
    ``err / D_ref``), so a star on which nothing was detected still says what
    change would have been.
    """
    params = params or ClassifyParams()
    ncand = float(params.n_candidate if n_candidate is None else n_candidate)
    out = {"ln_ratio": float("nan"), "sigma_ln": float("nan"), "z": float("nan"),
           "z_lin": float("nan"), "z_method": "", "ratio": float("nan"),
           "measured_snr": float("nan"), "expected_snr": float("nan"),
           "sigma_ln_expected": float("nan"), "detectable_ln_ratio": float("nan"),
           "detectable_depth_change_ppm": float("nan"), "comparison_status": ""}
    ref_ok = np.isfinite(ref_ppm) and ref_ppm > 0
    if not ref_ok:
        out["comparison_status"] = REASON_NO_REFERENCE
        return out
    fr_ref = (ref_err_ppm / ref_ppm) if np.isfinite(ref_err_ppm) and ref_err_ppm > 0 else 0.0
    if np.isfinite(total_err_ppm) and total_err_ppm > 0:
        s_exp = math.sqrt((total_err_ppm / ref_ppm) ** 2 + fr_ref ** 2 + params.sigma_sys_ln ** 2)
        out["sigma_ln_expected"] = s_exp
        out["detectable_ln_ratio"] = ncand * s_exp
        out["detectable_depth_change_ppm"] = ref_ppm * (math.exp(ncand * s_exp) - 1.0)
    if np.isfinite(stat_err_ppm) and stat_err_ppm > 0:
        out["expected_snr"] = ref_ppm / stat_err_ppm
    if not (np.isfinite(depth_ppm) and np.isfinite(total_err_ppm) and total_err_ppm > 0):
        out["comparison_status"] = REASON_NO_TRANSIT
        return out
    out["measured_snr"] = depth_ppm / stat_err_ppm if np.isfinite(stat_err_ppm) and \
        stat_err_ppm > 0 else depth_ppm / total_err_ppm
    out["ratio"] = depth_ppm / ref_ppm
    sig_lin = math.sqrt(total_err_ppm ** 2 + (fr_ref * ref_ppm) ** 2
                        + (params.sigma_sys_ln * ref_ppm) ** 2)
    out["z_lin"] = (depth_ppm - ref_ppm) / sig_lin
    if depth_ppm > 0 and out["measured_snr"] >= float(params.min_measured_snr_for_ln):
        s = math.sqrt((total_err_ppm / depth_ppm) ** 2 + fr_ref ** 2 + params.sigma_sys_ln ** 2)
        out["ln_ratio"] = math.log(depth_ppm / ref_ppm)
        out["sigma_ln"] = s
        out["z"] = out["ln_ratio"] / s
        out["z_method"] = "ln"
    else:
        # The log of a depth consistent with zero is not a number; the linear
        # form is, and it is what a disappeared transit is scored on.
        out["z"] = out["z_lin"]
        out["sigma_ln"] = sig_lin / ref_ppm
        out["ln_ratio"] = out["z"] * out["sigma_ln"]
        out["z_method"] = "linear"
    out["comparison_status"] = STATUS_OK
    return out


# ---------------------------------------------------------------------------
# One target
# ---------------------------------------------------------------------------
def measure_direct_target(entry: dict, products, *, fetch_status: str, fetch_route: str,
                          fit: FitParams | None = None, ensemble: EnsembleParams | None = None,
                          search: EpochSearchParams | None = None,
                          duration: DurationParams | None = None,
                          classify: ClassifyParams | None = None,
                          ld_table: dict | None = None
                          ) -> tuple[dict, list[dict], pd.DataFrame, pd.DataFrame]:
    """Everything the direct stage has to say about one planet, from products
    already fetched for its star.  ``(record, members, sectors_df, fold_df)``.
    """
    fit = fit or FitParams()
    ensemble = ensemble or EnsembleParams()
    search = search or EpochSearchParams()
    duration = duration or DurationParams()
    classify = classify or ClassifyParams()
    rec: dict = {k: entry.get(k) for k in ("kepoi_name", "kepler_name", "kepid", "tic_id",
                                            "tic_route", "koi_disposition", "koi_pdisposition",
                                            "koi_period", "koi_time0bk", "koi_duration",
                                            "koi_depth", "koi_depth_err1", "koi_depth_err2",
                                            "koi_ror", "koi_impact", "koi_dor", "koi_steff",
                                            "koi_slogg", "koi_kepmag", "koi_fpflag_nt",
                                            "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
                                            "koi_count", "ra", "dec")}
    rec.update({"lc_status": fetch_status, "lc_route": fetch_route,
                "n_products": int(len(products or [])),
                "product_authors": ",".join(sorted({str(p.get("author")) for p in (products or [])
                                                    if p.get("author")})),
                "product_sectors": ",".join(str(s) for s in sorted(
                    {int(p["sector"]) for p in (products or []) if p.get("sector") is not None})),
                "not_measured_reason": "", "class_provisional": CLASS_NOT_MEASURED,
                "t0_btjd_nominal": float("nan"), "t0_btjd_used": float("nan"),
                "n_epochs_propagated": 0, "ephemeris_sigma_minutes": float("nan")})
    # --- reference depth in the TESS band ------------------------------------
    period = _f(entry.get("koi_period"))
    t0_bkjd = _f(entry.get("koi_time0bk"))
    dur_h = _f(entry.get("koi_duration"))
    depth_k = _f(entry.get("koi_depth"))
    err_k = sym_err(_f(entry.get("koi_depth_err1")), _f(entry.get("koi_depth_err2")))
    b = _f(entry.get("koi_impact"))
    k = _f(entry.get("koi_ror"))
    fr, ld_meta = band_ratio(_f(entry.get("koi_steff")), _f(entry.get("koi_slogg")),
                             b if np.isfinite(b) else 0.0, ld_table)
    fr_applied = float(fr) if (classify.apply_band_ratio and np.isfinite(fr) and fr > 0) else 1.0
    ref = depth_k * fr_applied if np.isfinite(depth_k) and depth_k > 0 else float("nan")
    ref_err = err_k * fr_applied if np.isfinite(err_k) else float("nan")
    rec.update({"ld_band_ratio": float(fr), "ld_note": ld_meta.get("ld_note", ""),
                "koi_depth_in_tess_band_ppm": ref, "koi_depth_in_tess_band_err_ppm": ref_err})
    empty_sec, empty_fold = pd.DataFrame(), pd.DataFrame()
    for fam in FAMILIES:
        r0, _m0, _x = measure_family([], fam, period_days=1.0, t0_btjd=0.0, duration_days=0.1,
                                     fit=fit, ensemble=ensemble)
        rec.update(r0)
    rec.update({k: v for k, v in search_epoch_offset([], period_days=1.0, t0_btjd=0.0,
                                                     duration_days=0.1, sigma_t0_days=0.0,
                                                     fit=fit, params=search).items()})
    rec.update(fit_duration({}, duration_days=float("nan"), ingress_frac=0.15, params=duration))
    for fam in FAMILIES:
        rec.update({f"{fam}_{k}": v for k, v in compare_family(
            float("nan"), float("nan"), float("nan"), ref, ref_err, params=classify).items()})
    rec.update({k: v for k, v in sap_vs_pdcsap([], params=ensemble).items()
                if k != "sap_vs_pdcsap_pairs"})
    rec.update(duration_test(k, b, float("nan"), dur_h, float("nan"), float("nan"), float("nan")))
    def _unmeasured(reason: str):
        rec["not_measured_reason"] = reason
        rec.update(classify_direct(rec, params=classify))
        return rec, [], empty_sec, empty_fold

    if fetch_status != STATUS_OK:
        return _unmeasured(fetch_status)
    if not (np.isfinite(period) and period > 0 and np.isfinite(t0_bkjd)
            and np.isfinite(dur_h) and dur_h > 0):
        return _unmeasured(REASON_NO_EPHEMERIS)
    if not products:
        return _unmeasured(REASON_ZERO_ROWS)
    dur_d = dur_h / 24.0
    t0_tess = epoch_in_era(t0_bkjd, ERA_TESS)
    t_all = np.concatenate([np.asarray(p["time"], dtype=float) for p in products])
    prop = propagate_epoch(t0_tess, period, float(np.nanmedian(t_all)),
                           t0_err_days=sym_err(_f(entry.get("koi_time0bk_err1")),
                                               _f(entry.get("koi_time0bk_err2"))),
                           period_err_days=sym_err(_f(entry.get("koi_period_err1")),
                                                   _f(entry.get("koi_period_err2"))))
    rec.update({"t0_btjd_nominal": prop["t0_btjd"], "n_epochs_propagated": prop["n_epochs"],
                "ephemeris_sigma_minutes": prop["sigma_minutes"]})
    # --- the phase search, on the corrected family (else the raw one) --------
    search_segs = family_segments(products, FAMILY_PDC) or family_segments(products, FAMILY_SAP)
    search_segs, _dropped = dedupe_sectors(search_segs)
    es = search_epoch_offset(search_segs, period_days=period, t0_btjd=prop["t0_btjd"],
                             duration_days=dur_d, sigma_t0_days=prop["sigma_days"],
                             fit=fit, params=search)
    rec.update(es)
    t0_used = prop["t0_btjd"] + (es["epoch_offset_days"] if es["ephemeris_recovered"] else 0.0)
    rec["t0_btjd_used"] = t0_used
    # --- both families, same fitter, same epoch --------------------------------
    members_all: list[dict] = []
    sec_frames, fold_frames = [], []
    fam_res: dict = {}
    for fam in FAMILIES:
        r, mem, m = measure_family(products, fam, period_days=period, t0_btjd=t0_used,
                                   duration_days=dur_d, fit=fit, ensemble=ensemble)
        rec.update(r)
        members_all.extend(mem)
        fam_res[fam] = m
        if m is not None and len(m["sectors"]):
            sf = m["sectors"].copy()
            sf.insert(0, "family", fam)
            sec_frames.append(sf)
        if m is not None and len(m["fold"]):
            ff = m["fold"].copy()
            ff.insert(0, "family", fam)
            fold_frames.append(ff)
    rec.update({k: v for k, v in sap_vs_pdcsap(members_all, params=ensemble).items()
                if k != "sap_vs_pdcsap_pairs"})
    # --- the comparison with Kepler, per family --------------------------------
    for fam in FAMILIES:
        pre = f"{fam}_"
        cmp = compare_family(_f(rec.get(pre + "depth_ppm")), _f(rec.get(pre + "depth_err_ppm")),
                             _f(rec.get(pre + "total_err_ppm")), ref, ref_err, params=classify)
        rec.update({pre + k: v for k, v in cmp.items()})
    # --- the duration, on the corrected family's detrended fold ---------------
    dur_segs = family_segments(products, FAMILY_PDC) or family_segments(products, FAMILY_SAP)
    dur_segs, _d = dedupe_sectors(dur_segs)
    fr_ing = ingress_fraction(k, b, duration.default_ingress_fraction)
    rec["ingress_fraction_used"] = fr_ing
    fold_rec = detrended_fold(dur_segs, period_days=period, t0_btjd=t0_used, duration_days=dur_d,
                              params=fit, window_durations=duration.window_durations,
                              guard_durations=duration.guard_durations)
    rec.update(fit_duration(fold_rec, duration_days=dur_d, ingress_frac=fr_ing, params=duration))
    ratio_pdc = _f(rec.get("pdc_ratio"))
    rec.update(duration_test(k, b, ratio_pdc if np.isfinite(ratio_pdc) and ratio_pdc > 0
                             else float("nan"),
                             dur_h, sym_err(_f(entry.get("koi_duration_err1")),
                                            _f(entry.get("koi_duration_err2"))),
                             _f(rec.get("t14_tess_hours")), _f(rec.get("t14_tess_err_hours")),
                             a_rs=_f(entry.get("koi_dor")), period_days=period,
                             sigma_dur_sys=duration.sigma_dur_sys, nsigma=duration.n_duration))
    if rec["pdc_status"] != STATUS_OK and rec["sap_status"] != STATUS_OK:
        rec["not_measured_reason"] = REASON_NO_TRANSIT
    rec.update(classify_direct(rec, params=classify))
    sec = pd.concat(sec_frames, ignore_index=True) if sec_frames else empty_sec
    fold_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else empty_fold
    if len(sec):
        sec.insert(0, "kepoi_name", rec["kepoi_name"])
    if len(fold_df):
        fold_df.insert(0, "kepoi_name", rec["kepoi_name"])
    for mrec in members_all:
        mrec["kepoi_name"] = rec["kepoi_name"]
        mrec["tic_id"] = rec["tic_id"]
    return rec, members_all, sec, fold_df


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_direct(rec: dict, *, params: ClassifyParams | None = None,
                    offsets: dict | None = None) -> dict:
    """The class of one record, given the population offsets (``None`` = zero).

    Growth needs BOTH families up at ``n_candidate`` sigma --- the raw SAP on its
    own raw ``z`` (a bigger aperture can only make the raw depth shallower, so
    a SAP depth deeper than Kepler is never dilution) and the PDCSAP on its
    population-corrected, scatter-scaled ``z`` --- with the duration at fixed
    ``b``, no odd-even signature, the transit recovered, and no KOI flag.
    Shrinkage needs both families down on the population-corrected ``z``.  A
    PDCSAP change the SAP does not share, with PDC deeper than SAP beyond
    ``n_background`` sigma, is ``crowding_correction`` --- Kepler-718 b's case.
    """
    params = params or ClassifyParams()
    off = offsets or {}
    out: dict = {"class": CLASS_NOT_MEASURED, "vetoes": "", "would_be_candidate_without_vetoes": False,
                 "pdc_z_pop": float("nan"), "sap_z_pop": float("nan"),
                 "population_offset_pdc": _f(off.get("pdc_offset")) if off else 0.0,
                 "population_offset_sap": _f(off.get("sap_offset")) if off else 0.0,
                 "population_scale_pdc": _f(off.get("pdc_scale")) if off else 1.0,
                 "population_scale_sap": _f(off.get("sap_scale")) if off else 1.0,
                 "flags_classify": ""}
    for fam in FAMILIES:
        o = _f(off.get(f"{fam}_offset")) if off else 0.0
        sc = _f(off.get(f"{fam}_scale")) if off else 1.0
        o = o if np.isfinite(o) and params.subtract_population_median else 0.0
        sc = sc if np.isfinite(sc) and sc > 1.0 and params.scale_by_population_scatter else 1.0
        out[f"population_offset_{fam}"], out[f"population_scale_{fam}"] = o, sc
        ln, s = _f(rec.get(f"{fam}_ln_ratio")), _f(rec.get(f"{fam}_sigma_ln"))
        if np.isfinite(ln) and np.isfinite(s) and s > 0:
            out[f"{fam}_z_pop"] = (ln - o) / (s * sc)
    reason = _s(rec.get("not_measured_reason"))
    if _s(rec.get("lc_status")) != STATUS_OK or reason:
        out["class"] = CLASS_NOT_MEASURED
        return out
    pdc_ok = _s(rec.get("pdc_comparison_status")) == STATUS_OK
    sap_ok = _s(rec.get("sap_comparison_status")) == STATUS_OK
    if not (pdc_ok or sap_ok):
        out["class"] = CLASS_NOT_MEASURED
        return out
    exp_snr = _f(rec.get("pdc_expected_snr")) if pdc_ok else _f(rec.get("sap_expected_snr"))
    if not (np.isfinite(exp_snr) and exp_snr >= float(params.min_expected_snr)):
        out["class"] = CLASS_NOT_MEASURABLE
        return out
    if not _b(rec.get("ephemeris_recovered")):
        out["class"] = CLASS_NOT_RECOVERED
        return out
    zp, zs_pop, zs_raw = out["pdc_z_pop"], out["sap_z_pop"], _f(rec.get("sap_z"))
    n = float(params.n_candidate)
    pdc_up = bool(np.isfinite(zp) and zp >= n)
    pdc_down = bool(np.isfinite(zp) and zp <= -n)
    sap_up = bool(np.isfinite(zs_raw) and zs_raw >= n)
    sap_down = bool(np.isfinite(zs_pop) and zs_pop <= -n)
    vetoes: list[str] = []
    if not (pdc_ok and sap_ok):
        vetoes.append(VETO_ONE_FAMILY)
    for fam in FAMILIES:
        oe = _f(rec.get(f"{fam}_odd_even_sigma"))
        if np.isfinite(oe) and oe >= float(params.odd_even_sigma) and VETO_ODD_EVEN not in vetoes:
            vetoes.append(VETO_ODD_EVEN)
    dv = _s(rec.get("duration_verdict"))
    if dv == DUR_INCONCLUSIVE or not dv:
        vetoes.append(VETO_DURATION_INCONCLUSIVE)
    elif dv != DUR_FIXED_B:
        vetoes.append(VETO_DURATION)
    if any(_f(rec.get(c)) == 1 for c in ("koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co",
                                          "koi_fpflag_ec")):
        vetoes.append(VETO_FPFLAG)
    disp = _s(rec.get("koi_disposition")).upper()
    if disp and disp not in {d.upper() for d in params.dispositions}:
        vetoes.append(VETO_DISPOSITION)
    if _b(rec.get("pdc_depth_is_lower_bound")) or _b(rec.get("sap_depth_is_lower_bound")):
        vetoes.append(VETO_LOWER_BOUND)
    both_up = pdc_up and sap_up
    both_down = pdc_down and sap_down
    out["would_be_candidate_without_vetoes"] = bool(both_up or both_down)
    bg_z = _f(rec.get("sap_minus_pdcsap_z"))
    pdc_deeper = bool(np.isfinite(bg_z) and bg_z <= -float(params.n_background)
                      and _s(rec.get("background_direction")) == BG_PDC_DEEPER)
    flags = []
    if pdc_deeper:
        flags.append("pdc_deeper_than_sap")
    if both_up and not vetoes:
        out["class"] = CLASS_GROWTH
    elif both_down and not vetoes:
        out["class"] = CLASS_SHRINK
        flags.append("dilution_not_excluded")
    elif pdc_up and not sap_up and pdc_deeper:
        out["class"] = CLASS_CROWDING
    elif pdc_up or sap_up:
        out["class"] = CLASS_DEEPER
    elif pdc_down or sap_down:
        out["class"] = CLASS_SHALLOWER
    else:
        out["class"] = CLASS_CONSISTENT
    # Every veto that fired is on the record whether or not a depth gate did:
    # the summary counts them, and would_be_candidate_without_vetoes is the
    # denominator that says how many they actually cost.
    out["vetoes"] = ";".join(vetoes)
    out["flags_classify"] = ";".join(flags)
    return out


def population_offsets(meas: pd.DataFrame, *, params: ClassifyParams | None = None) -> dict:
    """The population's own median ``ln(D/D_ref)`` per family and the robust
    scatter of ``z`` about it --- the stage-1 honesty check, now applied.

    Stage 1's failure was 31 of 108 planets above 5 sigma of the median: the
    quoted errors did not describe the scatter.  Here the scatter of ``z`` is
    measured (``1.4826 * MAD``) and, when it exceeds one, every ``z`` is divided
    by it before the gate, so a candidate is ``n_candidate`` sigma in the
    population's OWN units.  Both numbers are reported per family.
    """
    params = params or ClassifyParams()
    out: dict = {"n_for_offset": {}, "median_ln_ratio": {}, "z_robust_scatter": {},
                 "n_z_above_5": {}, "n_z_below_minus_5": {}, "n_z_above_3": {},
                 "n_z_below_minus_3": {}, "ln_ratio_p16": {}, "ln_ratio_p84": {}}
    for fam in FAMILIES:
        ln = pd.to_numeric(meas.get(f"{fam}_ln_ratio", pd.Series(dtype=float)), errors="coerce")
        s = pd.to_numeric(meas.get(f"{fam}_sigma_ln", pd.Series(dtype=float)), errors="coerce")
        snr = pd.to_numeric(meas.get(f"{fam}_measured_snr", pd.Series(dtype=float)),
                            errors="coerce")
        rec_ok = (meas["ephemeris_recovered"].map(_b) if "ephemeris_recovered" in meas
                  else pd.Series(False, index=meas.index))
        good = (ln.notna() & s.notna() & (s > 0) & (snr >= float(params.min_expected_snr))
                & rec_ok.reindex(ln.index, fill_value=False))
        n = int(good.sum())
        out["n_for_offset"][fam] = n
        off, scale = 0.0, 1.0
        if n >= int(params.min_population_for_offset):
            v = ln[good].to_numpy(float)
            off = float(np.median(v))
            z = (v - off) / s[good].to_numpy(float)
            mad = float(np.median(np.abs(z - np.median(z)))) * 1.4826
            scale = float(mad) if np.isfinite(mad) and mad > 0 else 1.0
            out["ln_ratio_p16"][fam], out["ln_ratio_p84"][fam] = (
                float(x) for x in np.percentile(v, [16, 84]))
            out["n_z_above_5"][fam] = int((z >= 5).sum())
            out["n_z_below_minus_5"][fam] = int((z <= -5).sum())
            out["n_z_above_3"][fam] = int((z >= 3).sum())
            out["n_z_below_minus_3"][fam] = int((z <= -3).sum())
        out["median_ln_ratio"][fam] = off
        out["z_robust_scatter"][fam] = scale
        out[f"{fam}_offset"] = off if params.subtract_population_median else 0.0
        out[f"{fam}_scale"] = (scale if params.scale_by_population_scatter and scale > 1.0
                               else 1.0)
    out["subtract_population_median"] = bool(params.subtract_population_median)
    out["scale_by_population_scatter"] = bool(params.scale_by_population_scatter)
    return out


# ---------------------------------------------------------------------------
# Stage: probe
# ---------------------------------------------------------------------------
def direct_probe(conf: dict, out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"stage": "probe", "generated_utc": _now(), "mast": mast_probe()}
    try:
        from astroquery.mast import Catalogs  # noqa: PLC0415, F401

        rep["tic_fallback_importable"] = True
    except Exception as exc:                              # noqa: BLE001
        rep["tic_fallback_importable"] = False
        rep["tic_fallback_error"] = repr(exc)[:300]
    _write(out / "probe.json", rep)
    print(f"[growth-direct] probe: MAST route = {rep['mast'].get('route_preferred')}")
    return rep


# ---------------------------------------------------------------------------
# Stage: targets
# ---------------------------------------------------------------------------
def build_targets(koi: pd.DataFrame, ps: pd.DataFrame | None, *,
                  params: TargetParams | None = None) -> tuple[pd.DataFrame, dict]:
    """Every confirmed/candidate KOI with the TIC the ``ps`` routes give it (pure)."""
    params = params or TargetParams()
    rep: dict = {"n_koi": int(len(koi))}
    if not len(koi):
        rep.update(n_targets=0, n_with_tic=0, tic_by_route={}, n_without_tic=0)
        return pd.DataFrame(), rep
    k = koi.copy().reset_index(drop=True)
    k.columns = [str(c).strip().lower() for c in k.columns]
    for c in ("koi_period", "ra", "dec", "koi_time0bk", "koi_duration", "koi_depth", "kepid"):
        if c in k:
            k[c] = pd.to_numeric(k[c], errors="coerce")
    disp = k.get("koi_disposition", pd.Series([""] * len(k))).astype(str).str.upper()
    want = {d.upper() for d in params.dispositions}
    keep = disp.isin(want)
    rep["dispositions"] = list(params.dispositions)
    rep["n_by_disposition"] = disp.value_counts().to_dict()
    k = k[keep].reset_index(drop=True)
    rep["n_confirmed_or_candidate"] = int(len(k))
    maps = prepare_ps(ps, rep)
    names, hosts = koi_name_keys(k)
    k["_name"], k["_host"] = names, hosts
    tic, route = resolve_koi_tics(k, maps, routes=TIC_ROUTES,
                                  pos_radius_arcsec=params.pos_radius_arcsec)
    k["tic_id"] = tic
    k["tic_route"] = route
    k = k.drop(columns=["_name", "_host"])
    has_eph = (k["koi_period"].notna() & (k["koi_period"] > 0) & k["koi_time0bk"].notna()
               & k["koi_duration"].notna() & (k["koi_duration"] > 0))
    k["has_ephemeris"] = has_eph
    k["has_depth"] = k["koi_depth"].notna() & (k["koi_depth"] > 0)
    k["long_period"] = k["koi_period"] > float(params.long_period_min_days)
    rep["n_with_tic"] = int(np.isfinite(tic).sum())
    rep["tic_by_route"] = {r: int(sum(1 for x in route if x == r)) for r in TIC_ROUTES}
    rep["n_without_tic"] = int(len(k) - rep["n_with_tic"])
    rep["n_without_ephemeris"] = int((~has_eph).sum())
    rep["n_without_depth"] = int((~k["has_depth"]).sum())
    rep["n_long_period"] = int(k["long_period"].sum())
    rep["n_distinct_stars"] = int(k["kepid"].nunique())
    rep["n_distinct_tics"] = int(pd.Series(tic[np.isfinite(tic)]).nunique())
    if not params.include_unresolved:
        k = k[np.isfinite(k["tic_id"])].reset_index(drop=True)
    k = k.sort_values(["kepid", "kepoi_name"]).reset_index(drop=True)
    rep["n_targets"] = int(len(k))
    rep["targets_statement"] = (
        f"{rep['n_targets']} of {rep['n_koi']} KOIs are targets ({rep['n_confirmed_or_candidate']} "
        f"confirmed/candidate); {rep['n_with_tic']} carry a TIC id from the ps routes "
        + ", ".join(f"{r}={n}" for r, n in rep["tic_by_route"].items())
        + f"; {rep['n_without_tic']} have none and are left to the shard's TIC fallback; "
        f"{rep['n_distinct_stars']} distinct stars; {rep['n_long_period']} with P > "
        f"{params.long_period_min_days:g} d")
    return k, rep


def direct_targets(conf: dict, out: Path, *, query_fn=None,
                   log: AcquisitionLog | None = None) -> dict:
    """Fetch the KOI table and the ``ps`` TIC map; write ``targets.csv``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = log or AcquisitionLog(prefix="growth/direct")
    tp = TargetParams.from_config(conf)
    arch = (conf or {}).get("archive") or {}
    url = str(arch.get("exoarchive_tap") or "https://exoplanetarchive.ipac.caltech.edu/TAP/sync")
    qf = query_fn or (lambda adql: tap_sync(adql, url=url, retries=int(tp.archive_retries),
                                            timeout=float(tp.archive_timeout_s)))
    adql = table_query("cumulative", DIRECT_KOI_COLUMNS)
    try:
        koi = qf(adql)
        koi = koi if koi is not None else pd.DataFrame()
        koi = koi.rename(columns={c: str(c).strip().lower() for c in koi.columns})
        log.record("fetch_koi_cumulative", adql, rows=int(len(koi)))
        s_koi = STATUS_OK if len(koi) else STATUS_ZERO
    except Exception as exc:                              # noqa: BLE001
        log.record("fetch_koi_cumulative", adql, error=repr(exc))
        koi, s_koi = pd.DataFrame(), STATUS_FAILED
    ps, s_ps = fetch_ps_kepler(query_fn=qf, log=log)
    targets, rep = build_targets(koi, ps if s_ps == STATUS_OK else None, params=tp)
    rep.update({"stage": "targets", "generated_utc": _now(), "koi_status": s_koi,
                "ps_status": s_ps, "acquisition": log.as_dict()})
    _write_csv(out / "targets.csv", targets)
    _write(out / "targets.json", rep)
    log.write(out / "acquisition_log.json")
    print(f"[growth-direct] targets: {rep.get('targets_statement', rep)}")
    return rep


# ---------------------------------------------------------------------------
# Stage: measure (one shard)
# ---------------------------------------------------------------------------
#: The per-target columns kept in the shard CSV (and measurements.csv).
MEASUREMENT_COLUMNS: tuple[str, ...] = (
    "kepoi_name", "kepler_name", "kepid", "tic_id", "tic_route", "koi_disposition",
    "koi_pdisposition", "koi_period", "koi_time0bk", "koi_duration", "koi_depth",
    "koi_depth_err1", "koi_depth_err2", "koi_ror", "koi_impact", "koi_dor", "koi_steff",
    "koi_slogg", "koi_kepmag", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
    "koi_count", "ra", "dec", "shard", "lc_status", "lc_route", "not_measured_reason",
    "n_products", "product_authors", "product_sectors", "ld_band_ratio", "ld_note",
    "koi_depth_in_tess_band_ppm", "koi_depth_in_tess_band_err_ppm",
    "t0_btjd_nominal", "t0_btjd_used", "n_epochs_propagated", "ephemeris_sigma_minutes",
    "epoch_offset_minutes", "epoch_offset_sigma", "epoch_search_halfwidth_minutes",
    "epoch_search_n_trials", "epoch_search_snr_best", "epoch_search_snr_nominal",
    "depth_at_nominal_ephemeris_ppm", "depth_at_best_offset_ppm", "ephemeris_recovered",
    # families
    *[f"{fam}_{k}" for fam in FAMILIES for k in (
        "status", "flux_column", "primary_author", "depth_ppm", "depth_err_ppm",
        "depth_err_method", "total_err_ppm", "depth_reduction_spread_ppm",
        "depth_reduction_spread_status", "n_members", "n_members_measured", "ensemble_members",
        "n_transits", "n_sectors", "n_sectors_measured", "sector_list", "authors", "exptimes_s",
        "smeared", "depth_is_lower_bound", "oot_scatter_ppm", "odd_even_diff_ppm",
        "odd_even_sigma", "sector_scatter_chi2_per_dof", "n_sectors_dropped_duplicate", "flags",
        "ln_ratio", "sigma_ln", "z", "z_lin", "z_method", "ratio", "measured_snr",
        "expected_snr", "sigma_ln_expected", "detectable_ln_ratio",
        "detectable_depth_change_ppm", "comparison_status", "z_pop")],
    "sap_vs_pdcsap_verdict", "sap_vs_pdcsap_n_pairs", "sap_vs_pdcsap_authors",
    "sap_minus_pdcsap_ppm", "sap_minus_pdcsap_err_ppm", "sap_minus_pdcsap_z",
    "sap_minus_pdcsap_fraction", "background_direction",
    "ingress_fraction_used", "t14_tess_hours", "t14_tess_err_hours", "t14_factor",
    "t14_factor_err", "duration_fit_depth_ppm", "duration_fit_chi2_per_dof",
    "duration_fit_n_points", "duration_fit_status", "duration_at_grid_edge",
    "t14_ratio_observed", "t14_ratio_expected_fixed_b", "z_duration", "b_implied_by_duration",
    "duration_verdict",
    "class", "class_provisional", "vetoes", "would_be_candidate_without_vetoes",
    "flags_classify", "population_offset_pdc", "population_offset_sap",
    "population_scale_pdc", "population_scale_sap", "elapsed_s",
)

MEMBER_COLUMNS: tuple[str, ...] = (
    "kepoi_name", "tic_id", "family", "author", "flux_column", "status", "n_segments",
    "segment_list", "exptimes_s", "n_transits", "depth_ppm", "depth_err_ppm", "depth_err_method",
    "oot_scatter_ppm", "odd_even_sigma", "smeared", "is_primary_reduction",
)


def _shard_paths(out: Path, shard: int) -> dict:
    d = Path(out) / "shards"
    tag = f"shard_{int(shard):02d}"
    return {"dir": d, "csv": d / f"{tag}.csv", "json": d / f"{tag}.json",
            "members": d / f"{tag}_members.csv", "sectors": d / f"{tag}_sectors.csv",
            "folds": d / f"{tag}_folds", "tmp": d / f"{tag}_downloads"}


def select_shard(targets: pd.DataFrame, shard: int, n_shards: int) -> pd.DataFrame:
    if not len(targets):
        return targets
    s = targets["kepid"].map(lambda k: shard_of(k, n_shards))
    return targets[s == int(shard)].reset_index(drop=True)


def direct_measure(conf: dict, out: Path, *, shard: int = 0, n_shards: int = 1,
                   targets: pd.DataFrame | None = None, products_fn=None, tic_fn=None,
                   resume: bool = True, log: AcquisitionLog | None = None,
                   budget_s: float | None = None) -> dict:
    """Measure every target of one shard, checkpointing the shard CSV after each.

    Products are fetched ONCE per star and shared by its planets.  A target the
    shard's wall clock did not reach is not in the CSV at all, and the aggregate
    stage marks it ``NOT_REACHED``; a target whose fetch failed is
    ``not_measured`` with the archive's own status.  With ``resume``, targets
    already in the committed shard CSV are skipped unless their fetch had been
    cut off by the budget.
    """
    out = Path(out)
    paths = _shard_paths(out, shard)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    log = log or AcquisitionLog(prefix="growth/direct")
    fp = FetchParams.from_config(conf)
    if budget_s is not None:
        fp.shard_budget_s = float(budget_s)
    fit = FitParams.from_config(conf)
    ens = EnsembleParams.from_config(conf)
    search = EpochSearchParams.from_config(conf)
    dur_p = DurationParams.from_config(conf)
    cls_p = ClassifyParams.from_config(conf)
    ld_table = (conf or {}).get("limb_darkening")
    if fp.download_dir is None and products_fn is None:
        fp.download_dir = str(paths["tmp"])
    if targets is None:
        targets = _read_csv(out / "targets.csv")
    mine = select_shard(targets, shard, n_shards)
    done: dict[str, dict] = {}
    if resume:
        prev = _read_csv(paths["csv"])
        for r in (prev.to_dict(orient="records") if len(prev) else []):
            if _s(r.get("not_measured_reason")) in (REASON_BUDGET, REASON_NOT_REACHED):
                continue
            done[str(r.get("kepoi_name"))] = r
    prev_members = _read_csv(paths["members"]) if resume else pd.DataFrame()
    prev_members = prev_members[prev_members["kepoi_name"].astype(str).isin(done)] \
        if len(prev_members) and "kepoi_name" in prev_members else pd.DataFrame()
    # A budget of 0 is a ceiling of zero (everything NOT_REACHED), not "no ceiling".
    deadline = Deadline(budget_s=(float(fp.shard_budget_s)
                                  if fp.shard_budget_s is not None and float(fp.shard_budget_s) >= 0
                                  else None))
    recs: list[dict] = list(done.values())
    members_frames: list[pd.DataFrame] = [prev_members] if len(prev_members) else []
    sector_frames: list[pd.DataFrame] = []
    n_new, n_skipped, n_budget = 0, 0, 0
    tic_fallback_by_route: dict = {}
    fetch_status_counts: dict = {}

    def _checkpoint() -> None:
        df = pd.DataFrame(recs)
        if len(df):
            df = df[[c for c in MEASUREMENT_COLUMNS if c in df.columns]
                    + [c for c in df.columns if c not in MEASUREMENT_COLUMNS]]
        _write_csv(paths["csv"], df)
        if members_frames:
            mdf = pd.concat(members_frames, ignore_index=True)
            _write_csv(paths["members"], mdf[[c for c in MEMBER_COLUMNS if c in mdf.columns]])

    # Group by star: one download per kepid (the TIC is per star too).
    groups: list[tuple] = []
    if len(mine):
        for kepid, g in mine.groupby("kepid", sort=True):
            groups.append((kepid, g))
    for kepid, g in groups:
        rows = [r for r in g.to_dict(orient="records") if str(r.get("kepoi_name")) not in done]
        if not rows:
            n_skipped += len(g)
            continue
        if deadline.expired():
            n_budget += len(rows)
            break
        started = _time.monotonic()
        tic = _f(rows[0].get("tic_id"))
        route = _s(rows[0].get("tic_route"))
        if not (np.isfinite(tic) and tic > 0) and fp.tic_fallback:
            fn = tic_fn or astroquery_tic_fn
            try:
                tic, route = fn(kepid, _f(rows[0].get("ra")), _f(rows[0].get("dec")),
                                _f(rows[0].get("koi_kepmag")),
                                radius_arcsec=float(fp.tic_radius_arcsec),
                                mag_tolerance=float(fp.tic_mag_tolerance))
                log.record(f"tic_fallback_{int(kepid)}", f"KIC {int(kepid)}",
                           rows=1 if np.isfinite(_f(tic)) else 0, extra={"route": route})
            except Exception as exc:                      # noqa: BLE001
                log.record(f"tic_fallback_{int(kepid)}", f"KIC {int(kepid)}", error=repr(exc)[:300])
                tic, route = float("nan"), ""
            tic = _f(tic)
            if np.isfinite(tic):
                tic_fallback_by_route[route] = tic_fallback_by_route.get(route, 0) + 1
        products, status, lc_route = [], REASON_TIC_UNRESOLVED, ""
        if np.isfinite(tic) and tic > 0:
            products, status, lc_route = fetch_products(
                int(tic), products_fn=products_fn, params=fp, log=log, deadline=deadline,
                key=str(rows[0].get("kepoi_name")))
            if status == STATUS_FAILED and deadline.expired():
                status = REASON_BUDGET
        fetch_status_counts[status] = fetch_status_counts.get(status, 0) + 1
        for r in rows:
            entry = dict(r)
            entry["tic_id"] = tic if np.isfinite(tic) else float("nan")
            entry["tic_route"] = route
            t1 = _time.monotonic()
            rec, mem, sec, _fold_df = measure_direct_target(
                entry, products, fetch_status=status, fetch_route=lc_route, fit=fit, ensemble=ens,
                search=search, duration=dur_p, classify=cls_p, ld_table=ld_table)
            rec["shard"] = int(shard)
            rec["elapsed_s"] = round(_time.monotonic() - t1, 1)
            rec["class_provisional"] = rec.get("class")
            recs.append(rec)
            n_new += 1
            if mem:
                members_frames.append(pd.DataFrame(mem))
            if len(sec):
                sector_frames.append(sec)
            if rec.get("class") in (CLASS_GROWTH, CLASS_SHRINK, CLASS_DEEPER, CLASS_SHALLOWER,
                                    CLASS_CROWDING) and len(_fold_df):
                paths["folds"].mkdir(parents=True, exist_ok=True)
                _write_csv(paths["folds"] / f"{str(rec['kepoi_name']).replace('/', '_')}.csv",
                           _fold_df)
            print(f"[growth-direct] {rec['kepoi_name']} (shard {shard}): {rec['lc_status']} "
                  f"{rec['n_products']} products; PDC {rec.get('pdc_depth_ppm')} +/- "
                  f"{rec.get('pdc_total_err_ppm')} ppm (z {rec.get('pdc_z')}); SAP "
                  f"{rec.get('sap_depth_ppm')} +/- {rec.get('sap_total_err_ppm')} ppm "
                  f"(z {rec.get('sap_z')}); ref {rec.get('koi_depth_in_tess_band_ppm')}; "
                  f"offset {rec.get('epoch_offset_minutes')} min "
                  f"[{'recovered' if rec.get('ephemeris_recovered') else 'NOT recovered'}]; "
                  f"T14 x{rec.get('t14_factor')} {rec.get('duration_verdict')}; "
                  f"-> {rec.get('class')} {rec.get('vetoes') or ''}")
        _checkpoint()
        if fp.clean_downloads and fp.download_dir and products_fn is None:
            shutil.rmtree(fp.download_dir, ignore_errors=True)
        print(f"[growth-direct] star {int(kepid)}: {len(rows)} planet(s) in "
              f"{_time.monotonic() - started:.1f}s; shard elapsed {deadline.elapsed():.0f}s "
              f"of {fp.shard_budget_s}")
    _checkpoint()
    if sector_frames:
        sdf = pd.concat(sector_frames, ignore_index=True)
        _write_csv(paths["sectors"], sdf)
    df = pd.DataFrame(recs)
    classes = df["class"].map(str).value_counts().to_dict() if len(df) and "class" in df else {}
    rep = {"stage": "measure", "shard": int(shard), "n_shards": int(n_shards),
           "generated_utc": _now(), "n_targets_in_shard": int(len(mine)),
           "n_stars_in_shard": int(len(groups)), "n_measured_this_run": n_new,
           "n_skipped_already_done": n_skipped, "n_not_reached_budget": n_budget,
           "n_rows": int(len(df)), "budget_s": fp.shard_budget_s,
           "elapsed_s": round(deadline.elapsed(), 1), "budget_exhausted": bool(deadline.expired()),
           "fetch_status_counts": fetch_status_counts,
           "tic_fallback_by_route": tic_fallback_by_route,
           "classes_provisional": classes,
           "lc_routes": (df["lc_route"].map(str).value_counts().to_dict()
                         if len(df) and "lc_route" in df else {}),
           "acquisition": {k: v for k, v in log.as_dict().items() if k != "stages"},
           "acquisition_failures": [s for s in log.stages if s.get("status") == STATUS_FAILED][:50]}
    _write(paths["json"], rep)
    print(f"[growth-direct] shard {shard}/{n_shards}: {n_new} measured, {n_skipped} skipped, "
          f"{n_budget} not reached in {rep['elapsed_s']}s; {classes}")
    return rep


# ---------------------------------------------------------------------------
# Stage: assess (aggregate the shards)
# ---------------------------------------------------------------------------
CANDIDATE_COLUMNS: tuple[str, ...] = (
    "kepoi_name", "kepler_name", "kepid", "tic_id", "tic_route", "koi_disposition", "class",
    "vetoes", "flags_classify", "koi_period", "koi_depth", "koi_depth_in_tess_band_ppm",
    "pdc_depth_ppm", "pdc_total_err_ppm", "pdc_z", "pdc_z_pop", "pdc_ratio",
    "sap_depth_ppm", "sap_total_err_ppm", "sap_z", "sap_z_pop", "sap_ratio",
    "sap_minus_pdcsap_z", "background_direction", "pdc_odd_even_sigma", "sap_odd_even_sigma",
    "duration_verdict", "z_duration", "t14_ratio_observed", "t14_ratio_expected_fixed_b",
    "epoch_offset_minutes", "epoch_offset_sigma", "epoch_search_snr_best",
    "pdc_n_transits", "sap_n_transits", "pdc_sector_list", "pdc_authors", "pdc_exptimes_s",
    "pdc_depth_reduction_spread_ppm", "sap_depth_reduction_spread_ppm",
    "pdc_sector_scatter_chi2_per_dof", "ra", "dec", "shard",
)


def gather_shards(out: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Every ``shards/shard_*.csv`` concatenated, one row per KOI (last wins)."""
    out = Path(out)
    frames, reports = [], []
    for p in sorted(glob.glob(str(out / "shards" / "shard_*.csv"))):
        name = Path(p).name
        if name.endswith("_members.csv") or name.endswith("_sectors.csv"):
            continue
        df = _read_csv(p)
        if len(df):
            frames.append(df)
    for p in sorted(glob.glob(str(out / "shards" / "shard_*.json"))):
        try:
            reports.append(json.loads(Path(p).read_text()))
        except Exception:                                 # noqa: BLE001
            continue
    if not frames:
        return pd.DataFrame(), reports
    df = pd.concat(frames, ignore_index=True)
    if "kepoi_name" in df:
        df["kepoi_name"] = df["kepoi_name"].astype(str)
        df = df.drop_duplicates(subset=["kepoi_name"], keep="last").reset_index(drop=True)
    return df, reports


def direct_assess(conf: dict, out: Path) -> dict:
    """Aggregate every shard, measure the population offsets, classify, write the summary."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cls_p = ClassifyParams.from_config(conf)
    tp = TargetParams.from_config(conf)
    targets = _read_csv(out / "targets.csv")
    tj = {}
    try:
        tj = json.loads((out / "targets.json").read_text()) if (out / "targets.json").exists() \
            else {}
    except Exception:                                     # noqa: BLE001
        tj = {}
    meas, shard_reports = gather_shards(out)
    # Every target not in any shard file was never reached.
    if len(targets):
        seen = set(meas["kepoi_name"].astype(str)) if len(meas) else set()
        missing = targets[~targets["kepoi_name"].astype(str).isin(seen)].copy()
        if len(missing):
            missing["lc_status"] = ""
            missing["not_measured_reason"] = REASON_NOT_REACHED
            missing["class"] = CLASS_NOT_MEASURED
            meas = pd.concat([meas, missing], ignore_index=True) if len(meas) else missing
    if not len(meas):
        summary = {"verdict": RUN_NO_DATA, "reason": "no_targets_or_shards", "generated_utc": _now(),
                   "n_targets": int(len(targets)), "targets": tj}
        _write(out / "summary.json", summary)
        return summary
    offsets = population_offsets(meas, params=cls_p)
    rows = meas.to_dict(orient="records")
    for r in rows:
        r.update(classify_direct(r, params=cls_p, offsets=offsets))
    meas = pd.DataFrame(rows)
    meas = meas[[c for c in MEASUREMENT_COLUMNS if c in meas.columns]
                + [c for c in meas.columns if c not in MEASUREMENT_COLUMNS
                   and not str(c).startswith("_")]]
    _write_csv(out / "measurements.csv", meas)
    cls = meas["class"].map(str)
    classes = {c: int((cls == c).sum()) for c in CLASSES}
    cands = meas[cls.isin(CANDIDATE_CLASSES)].copy()
    cands = cands.assign(_abs=cands["pdc_z_pop"].map(_f).abs()).sort_values("_abs", ascending=False)
    cands = cands.drop(columns=["_abs"])
    _write_csv(out / "candidates.csv", cands[[c for c in CANDIDATE_COLUMNS if c in cands.columns]])
    lp = meas[pd.to_numeric(meas.get("koi_period"), errors="coerce")
              > float(tp.long_period_min_days)].copy()
    _write_csv(out / "long_period.csv", lp[[c for c in CANDIDATE_COLUMNS if c in lp.columns]])
    reasons = (meas.loc[cls == CLASS_NOT_MEASURED, "not_measured_reason"].fillna("").map(_s)
               .replace("", "unknown").value_counts().to_dict())
    lc_status = meas["lc_status"].fillna("").map(_s).replace("", "not_attempted") \
        .value_counts().to_dict()
    vet_counts: dict = {}
    for v in meas["vetoes"].fillna("").map(_s):
        for x in v.split(";"):
            if x:
                vet_counts[x] = vet_counts.get(x, 0) + 1
    flag_counts: dict = {}
    for col in ("pdc_flags", "sap_flags", "flags_classify"):
        if col in meas:
            for v in meas[col].fillna("").map(_s):
                for x in v.split(";"):
                    if x:
                        key = f"{col.split('_')[0]}:{x}" if col != "flags_classify" else x
                        flag_counts[key] = flag_counts.get(key, 0) + 1
    measured = meas[cls.isin((CLASS_GROWTH, CLASS_SHRINK, CLASS_DEEPER, CLASS_SHALLOWER,
                              CLASS_CONSISTENT, CLASS_CROWDING))]
    sens = {}
    for fam in FAMILIES:
        d = pd.to_numeric(measured.get(f"{fam}_detectable_ln_ratio"), errors="coerce").dropna() \
            if len(measured) else pd.Series(dtype=float)
        sens[fam] = ({"n": int(len(d)), "median_detectable_ln_ratio": float(d.median()),
                      "p16": float(d.quantile(0.16)), "p84": float(d.quantile(0.84)),
                      "median_detectable_depth_change_fraction": float(math.exp(d.median()) - 1.0)}
                     if len(d) else {"n": 0})
    n_meas = int(len(measured))
    n_cand = int(len(cands))
    if n_meas == 0:
        verdict, reason = RUN_NO_DATA, "no target reached a measured depth: " + ",".join(
            f"{k}={v}" for k, v in sorted(reasons.items()))
    elif n_cand:
        verdict, reason = RUN_CANDIDATES, f"{n_cand} candidate(s) pending the vet stage"
    else:
        verdict, reason = RUN_NONE, (f"{n_meas} planets measured in both eras, none changed at "
                                     f"{cls_p.n_candidate:g} sigma in both SAP and PDCSAP")
    tic_by_route = meas["tic_route"].fillna("").map(_s).replace("", "unresolved") \
        .value_counts().to_dict()
    funnel = {
        "n_koi_cumulative": tj.get("n_koi"),
        "n_confirmed_or_candidate": tj.get("n_confirmed_or_candidate"),
        "n_targets": int(len(targets)) if len(targets) else int(len(meas)),
        "n_with_tic_from_ps": tj.get("n_with_tic"),
        "n_tic_by_route_after_fallback": tic_by_route,
        "n_in_shard_files": int(len(meas) - int((meas["not_measured_reason"].fillna("").map(_s)
                                                  == REASON_NOT_REACHED).sum())),
        "n_not_reached": int((meas["not_measured_reason"].fillna("").map(_s)
                              == REASON_NOT_REACHED).sum()),
        "lc_status_counts": lc_status,
        "n_lightcurve_ok": int(lc_status.get(STATUS_OK, 0)),
        "n_pdc_measured": int((meas["pdc_comparison_status"].fillna("").map(_s) == STATUS_OK).sum()
                              if "pdc_comparison_status" in meas else 0),
        "n_sap_measured": int((meas["sap_comparison_status"].fillna("").map(_s) == STATUS_OK).sum()
                              if "sap_comparison_status" in meas else 0),
        "n_measurable": int(n_meas + classes[CLASS_NOT_RECOVERED]),
        "n_measured_both_eras": n_meas,
        "n_would_be_candidate_without_vetoes": int(
            meas["would_be_candidate_without_vetoes"].map(_b).sum()),
        "n_candidates": n_cand,
        "classes": classes,
        "not_measured_reasons": reasons,
        "vetoes": vet_counts,
        "flags": flag_counts,
        "n_long_period": int(len(lp)),
        "long_period_classes": lp["class"].map(str).value_counts().to_dict() if len(lp) else {},
    }
    shard_stats = [{k: r.get(k) for k in ("shard", "n_targets_in_shard", "n_rows",
                                          "n_measured_this_run", "n_not_reached_budget",
                                          "elapsed_s", "budget_exhausted", "fetch_status_counts",
                                          "tic_fallback_by_route", "lc_routes")}
                   for r in shard_reports]
    degraded = []
    if funnel["n_not_reached"]:
        degraded.append(f"{funnel['n_not_reached']} target(s) NOT_REACHED by any shard")
    if lc_status.get(STATUS_FAILED):
        degraded.append(f"{lc_status[STATUS_FAILED]} target(s) QUERY_FAILED at MAST")
    if any(r.get("budget_exhausted") for r in shard_reports):
        degraded.append("at least one shard exhausted its wall-clock budget")
    summary = {
        "verdict": verdict, "reason": reason, "generated_utc": _now(),
        "what_was_measured": ("the TESS-era transit depth of every confirmed/candidate KOI with a "
                              "TIC id, fitted from the SPOC / TESS-SPOC / QLP light curves with "
                              "the stage-2 fitter on BOTH the raw SAP and the corrected PDCSAP "
                              "family, against the KOI depth carried into the TESS band"),
        "funnel": funnel,
        "population": offsets,
        "sensitivity": sens,
        "candidates": cands[[c for c in CANDIDATE_COLUMNS if c in cands.columns]]
        .to_dict(orient="records"),
        "crowding_correction_cases": meas.loc[cls == CLASS_CROWDING,
                                              [c for c in CANDIDATE_COLUMNS if c in meas.columns]]
        .head(50).to_dict(orient="records"),
        "transit_not_recovered": meas.loc[cls == CLASS_NOT_RECOVERED,
                                          [c for c in ("kepoi_name", "kepler_name", "koi_period",
                                                       "koi_depth_in_tess_band_ppm",
                                                       "pdc_expected_snr",
                                                       "epoch_search_snr_best",
                                                       "epoch_search_halfwidth_minutes",
                                                       "ephemeris_sigma_minutes",
                                                       "pdc_n_transits")
                                           if c in meas.columns]].head(200).to_dict(orient="records"),
        "shards": shard_stats,
        "degraded": degraded,
        "config": {"classify": {k: getattr(cls_p, k) for k in cls_p.__dataclass_fields__},
                   "targets": {k: getattr(tp, k) for k in tp.__dataclass_fields__}},
        "checks_not_performed": [
            "kepler_era_refit (the Kepler-era depth here is cumulative.koi_depth; the vet stage "
            "refits the Kepler light curve with the same fitter for every candidate)",
            "per_pixel_centroid (the vet stage runs the difference image on every candidate)",
            "achromaticity (one band per era)",
            "gaia_dilution_per_star (shrinkage on the raw SAP is dilution until the Gaia census "
            "in the vet stage excludes it; flagged dilution_not_excluded)",
            "independent_period_search (a transit not recovered inside the phase window is "
            "listed, not searched for at other periods)",
        ],
        "note": ("a growth_candidate here is a reason to run the vet stage (stage 2: both eras "
                 "fitted alike with the reduction ensemble; stage 3: Gaia census and difference "
                 "image), not a detection. NO_DEPTH_CHANGE_CANDIDATE is a count at each star's "
                 "stated sensitivity and is not written up (CLAUDE.md). The raw SAP depth of a "
                 "star must read shallower than Kepler's, never deeper: a PDCSAP-only change is "
                 "a crowding-correction case, which is how Kepler-718 b was tracked down."),
    }
    _write(out / "summary.json", summary)
    print(f"[growth-direct] assess: {verdict} — {reason}; classes {classes}; offsets "
          f"{offsets['median_ln_ratio']} scatter {offsets['z_robust_scatter']}")
    return summary


# ---------------------------------------------------------------------------
# Stage: control (the look-elsewhere null, on everything that changed)
# ---------------------------------------------------------------------------
#: Classes the control stage re-opens.  ``consistent`` needs no null --- the
#: search bias can only push a depth UP, so it cannot manufacture agreement.
CONTROL_CLASSES: tuple[str, ...] = (CLASS_GROWTH, CLASS_SHRINK, CLASS_DEEPER, CLASS_CROWDING)

CONTROL_COLUMNS: tuple[str, ...] = (
    "kepoi_name", "kepler_name", "kepid", "tic_id", "class", "koi_period",
    "koi_depth_in_tess_band_ppm", "epoch_search_snr_best", "epoch_search_n_trials",
    "epoch_offset_minutes", "epoch_offset_sigma", "control_status",
    *[f"{fam}_{k}" for fam in FAMILIES for k in (
        "depth_ppm", "total_err_ppm", "z", "control_n_phases_measured", "control_snr_max",
        "control_snr_median", "control_depth_max_ppm", "control_depth_median_ppm",
        "snr_excess", "depth_excess_over_control_ppm", "control_verdict")],
    "control_verdict",
)


def direct_control(conf: dict, out: Path, *, products_fn=None, tic_fn=None,
                   budget_s: float | None = None, max_targets: int | None = None,
                   classes=None, log: AcquisitionLog | None = None) -> dict:
    """Run the control-phase null on every planet the measurement said changed.

    The epoch search maximises signal-to-noise over up to ``max_trials``
    offsets and the depth is fitted at the winner; that maximum is biased
    upward and its bias is not in the formal error.  This stage re-fetches the
    star and repeats the identical search at ``epoch_search.control_phases``,
    where the planet is not, so every survivor carries the depth the search
    manufactures from that star's own noise.  A planet whose transit does not
    beat its own null (``WITHIN_SEARCH_NOISE``) is not a candidate.
    """
    out = Path(out)
    cdir = out / "control"
    cdir.mkdir(parents=True, exist_ok=True)
    log = log or AcquisitionLog(prefix="growth/direct/control")
    fp = FetchParams.from_config(conf)
    if budget_s is not None:
        fp.shard_budget_s = float(budget_s)
    fit = FitParams.from_config(conf)
    search = EpochSearchParams.from_config(conf)
    want = tuple(classes) if classes else CONTROL_CLASSES
    meas = _read_csv(out / "measurements.csv")
    if not len(meas) or "class" not in meas:
        rep = {"stage": "control", "generated_utc": _now(), "n_selected": 0,
               "note": "no measurements.csv to re-open"}
        _write(cdir / "summary.json", rep)
        return rep
    sel = meas[meas["class"].map(_s).isin(want)].copy()
    sel = sel.assign(_a=pd.to_numeric(sel.get("pdc_z"), errors="coerce").abs()) \
        .sort_values("_a", ascending=False).drop(columns=["_a"])
    cap = int(max_targets) if max_targets is not None else len(sel)
    sel = sel.head(max(cap, 0))
    if fp.download_dir is None and products_fn is None:
        fp.download_dir = str(cdir / "downloads")
    deadline = Deadline(budget_s=(float(fp.shard_budget_s)
                                  if fp.shard_budget_s is not None else None))
    rows: list[dict] = []
    n_within, n_above, n_unavailable = 0, 0, 0
    for kepid, g in sel.groupby("kepid", sort=True):
        if deadline.expired():
            for r in g.to_dict(orient="records"):
                rows.append({"kepoi_name": r.get("kepoi_name"), "kepid": kepid,
                             "class": r.get("class"), "control_status": REASON_NOT_REACHED,
                             "control_verdict": CTRL_UNAVAILABLE})
                n_unavailable += 1
            continue
        tic = _f(g.iloc[0].get("tic_id"))
        products, status, _route = ([], REASON_TIC_UNRESOLVED, "")
        if np.isfinite(tic) and tic > 0:
            products, status, _route = fetch_products(
                int(tic), products_fn=products_fn, params=fp, log=log, deadline=deadline,
                key=f"control_{g.iloc[0].get('kepoi_name')}")
        for r in g.to_dict(orient="records"):
            row = {k: r.get(k) for k in ("kepoi_name", "kepler_name", "kepid", "tic_id", "class",
                                          "koi_period", "koi_depth_in_tess_band_ppm",
                                          "epoch_search_snr_best", "epoch_search_n_trials",
                                          "epoch_offset_minutes", "epoch_offset_sigma")}
            row["control_status"] = status
            period, t0u = _f(r.get("koi_period")), _f(r.get("t0_btjd_used"))
            dur_d = _f(r.get("koi_duration")) / 24.0
            sig_d = _f(r.get("ephemeris_sigma_minutes")) / 1440.0
            verdicts = []
            for fam in FAMILIES:
                row[f"{fam}_depth_ppm"] = r.get(f"{fam}_depth_ppm")
                row[f"{fam}_total_err_ppm"] = r.get(f"{fam}_total_err_ppm")
                row[f"{fam}_z"] = r.get(f"{fam}_z")
                c = {"control_verdict": CTRL_UNAVAILABLE}
                if status == STATUS_OK and products and np.isfinite(period) and period > 0 \
                        and np.isfinite(t0u) and np.isfinite(dur_d) and dur_d > 0:
                    c = apply_control(r, control_phase_null(
                        products, fam, period_days=period, t0_btjd=t0u, duration_days=dur_d,
                        sigma_t0_days=sig_d if np.isfinite(sig_d) else 0.0,
                        depth_ppm=_f(r.get(f"{fam}_depth_ppm")),
                        ref_ppm=_f(r.get("koi_depth_in_tess_band_ppm")),
                        fit=fit, params=search), family=fam)
                for k in ("control_n_phases_measured", "control_snr_max", "control_snr_median",
                          "control_depth_max_ppm", "control_depth_median_ppm", "snr_excess",
                          "depth_excess_over_control_ppm", "control_verdict"):
                    row[f"{fam}_{k}"] = c.get(k)
                verdicts.append(_s(c.get("control_verdict")))
            # The planet's verdict is the WEAKEST family's: a change the search
            # can manufacture in either reduction is not a change.
            if CTRL_UNAVAILABLE in verdicts or not verdicts:
                row["control_verdict"] = CTRL_UNAVAILABLE
                n_unavailable += 1
            elif CTRL_WITHIN in verdicts:
                row["control_verdict"] = CTRL_WITHIN
                n_within += 1
            else:
                row["control_verdict"] = CTRL_ABOVE
                n_above += 1
            rows.append(row)
            print(f"[growth-direct] control {row['kepoi_name']} ({row['class']}): "
                  f"snr {row.get('epoch_search_snr_best')} vs control "
                  f"pdc {row.get('pdc_control_snr_max')} / sap {row.get('sap_control_snr_max')}"
                  f" -> {row['control_verdict']}")
        if fp.clean_downloads and fp.download_dir and products_fn is None:
            shutil.rmtree(fp.download_dir, ignore_errors=True)
    df = pd.DataFrame(rows)
    if len(df):
        df = df[[c for c in CONTROL_COLUMNS if c in df.columns]
                + [c for c in df.columns if c not in CONTROL_COLUMNS]]
    _write_csv(cdir / "control.csv", df)
    rep = {"stage": "control", "generated_utc": _now(), "classes_selected": list(want),
           "n_selected": int(len(sel)), "n_rows": int(len(df)),
           "n_above_control": n_above, "n_within_search_noise": n_within,
           "n_control_unavailable": n_unavailable,
           "control_phases": list(search.control_phases),
           "budget_exhausted": bool(deadline.expired()),
           "elapsed_s": round(deadline.elapsed(), 1),
           "what_this_tests": ("the epoch search maximises S/N over up to "
                               f"{search.max_trials} offsets and the depth is fitted at the "
                               "winner; the same search is repeated at control phases where the "
                               "planet is not, and a planet whose transit does not beat that "
                               "null is not a candidate"),
           "caveats": ["a control phase can land on a sibling planet's transit in a multi-planet "
                       "system, which makes the null conservative, not permissive",
                       "a control phase in a data gap returns nothing and is not counted",
                       "the null is per star: it is not a population statement"],
           "acquisition": {k: v for k, v in log.as_dict().items() if k != "stages"}}
    _write(cdir / "summary.json", rep)
    print(f"[growth-direct] control: {n_above} above control, {n_within} within search noise, "
          f"{n_unavailable} unavailable")
    return rep


# ---------------------------------------------------------------------------
# Stage: vet (stage 2 + stage 3 on every survivor)
# ---------------------------------------------------------------------------
def direct_vet(conf: dict, out: Path, *, query_fn=None, lc_fn=None, kepler_lc_fn=None,
               ensemble_lc_fn=None, kepler_ensemble_lc_fn=None, cone_fn=None, tpf_fn=None,
               max_targets: int | None = None) -> dict:
    """Run stage 2 (both eras, one fitter, the reduction ensemble) and stage 3
    (census + difference image) on every candidate, into ``out/vet/``."""
    from .centroid import centroid_run  # noqa: PLC0415
    from .stage2 import stage2_assess, stage2_measure  # noqa: PLC0415

    out = Path(out)
    cls_p = ClassifyParams.from_config(conf)
    cap = int(cls_p.vet_max_targets if max_targets is None else max_targets)
    cands = _read_csv(out / "candidates.csv")
    vet_dir = out / "vet"
    vet_dir.mkdir(parents=True, exist_ok=True)
    if not len(cands):
        rep = {"stage": "vet", "generated_utc": _now(), "n_candidates": 0, "n_vetted": 0,
               "note": "no candidate to vet"}
        _write(vet_dir / "summary.json", rep)
        print("[growth-direct] vet: nothing to vet")
        return rep
    cands = cands.assign(_abs=pd.to_numeric(cands.get("pdc_z_pop"), errors="coerce").abs()) \
        .sort_values("_abs", ascending=False)
    chosen = cands.head(cap)
    shortlist = pd.DataFrame([{"kepoi_name": str(r.get("kepoi_name")),
                               "kepler_name": r.get("kepler_name"), "kepid": r.get("kepid"),
                               "tic_id": r.get("tic_id"), "toi": float("nan"),
                               "shortlist_source": "direct_candidates"}
                              for r in chosen.to_dict(orient="records")])
    s2_dir = vet_dir / "stage2"
    stage2_measure(conf, s2_dir, query_fn=query_fn, lc_fn=lc_fn, kepler_lc_fn=kepler_lc_fn,
                   ensemble_lc_fn=ensemble_lc_fn, kepler_ensemble_lc_fn=kepler_ensemble_lc_fn,
                   shortlist=shortlist)
    s2 = stage2_assess(conf, s2_dir)
    c3_dir = vet_dir / "centroid"
    c3 = centroid_run("census,difference,assess", out_dir=c3_dir, conf=conf, query_fn=query_fn,
                      cone_fn=cone_fn, tpf_fn=tpf_fn, shortlist=shortlist)
    by2 = {str(t.get("kepoi_name")): t for t in (s2.get("targets") or [])}
    by3 = {str(t.get("kepoi_name")): t for t in (c3.get("targets") or [])}
    # The look-elsewhere null, if the control stage has run.  Absent, it is
    # NOT silently treated as a pass: the column says CONTROL_NOT_RUN and the
    # survivor carries that as an open systematic.
    ctrl_df = _read_csv(out / "control" / "control.csv")
    byc = {str(r.get("kepoi_name")): r for r in (ctrl_df.to_dict(orient="records")
                                                 if len(ctrl_df) else [])}
    per = []
    for r in chosen.to_dict(orient="records"):
        k = str(r.get("kepoi_name"))
        t2, t3 = by2.get(k, {}), by3.get(k, {})
        tc = byc.get(k, {})
        cverd = _s(tc.get("control_verdict")) or "CONTROL_NOT_RUN"
        survives = (str(t2.get("like_for_like_verdict")) == "MEASURED_DEPTH_CHANGED"
                    and str(t3.get("verdict")) == "TRANSIT_ON_TARGET"
                    and str(t2.get("sap_vs_pdcsap_verdict")) != BG_DISAGREE
                    and cverd != CTRL_WITHIN)
        per.append({"kepoi_name": k, "kepler_name": r.get("kepler_name"), "tic_id": r.get("tic_id"),
                    "direct_class": r.get("class"), "direct_pdc_z_pop": r.get("pdc_z_pop"),
                    "direct_sap_z": r.get("sap_z"),
                    "stage2_like_for_like_verdict": t2.get("like_for_like_verdict"),
                    "stage2_z_measured_eras": t2.get("z_measured_eras"),
                    "stage2_depth_kepler_measured_ppm": t2.get("depth_kepler_measured_ppm"),
                    "stage2_depth_tess_measured_ppm": t2.get("depth_tess_measured_ppm"),
                    "stage2_koi_depth_verdict": t2.get("koi_depth_verdict"),
                    "stage2_sap_vs_pdcsap_verdict": t2.get("sap_vs_pdcsap_verdict"),
                    "stage2_background_direction": t2.get("background_direction"),
                    "stage3_verdict": t3.get("verdict"),
                    "stage3_offset_arcsec": t3.get("offset_arcsec"),
                    "stage3_offset_sigma": t3.get("offset_sigma"),
                    "stage3_census_statement": t3.get("census_statement"),
                    "control_verdict": cverd,
                    "control_snr_excess_pdc": tc.get("pdc_snr_excess"),
                    "control_snr_excess_sap": tc.get("sap_snr_excess"),
                    "control_depth_max_pdc_ppm": tc.get("pdc_control_depth_max_ppm"),
                    "survives_vet": bool(survives)})
    pdf = pd.DataFrame(per)
    _write_csv(vet_dir / "vetted.csv", pdf)
    rep = {"stage": "vet", "generated_utc": _now(), "n_candidates": int(len(cands)),
           "n_vetted": int(len(chosen)), "n_not_vetted_over_cap": int(max(len(cands) - cap, 0)),
           "n_survive_vet": int(pdf["survives_vet"].sum()) if len(pdf) else 0,
           "stage2_primary_verdict": s2.get("primary_verdict"),
           "stage3_verdict": c3.get("verdict"), "targets": per,
           "n_control_within_search_noise": int(sum(1 for t in per
                                                    if t.get("control_verdict") == CTRL_WITHIN)),
           "n_control_not_run": int(sum(1 for t in per
                                        if t.get("control_verdict") == "CONTROL_NOT_RUN")),
           "survives_vet_means": ("stage 2 MEASURED_DEPTH_CHANGED on the total error with both "
                                  "eras fitted alike, SAP and PDCSAP not in disagreement, "
                                  "stage 3 TRANSIT_ON_TARGET, and the transit beating its own "
                                  "control-phase null WHERE THAT NULL HAS RUN — a survivor "
                                  "carrying CONTROL_NOT_RUN has the look-elsewhere bias of the "
                                  "epoch search still open against it, and n_control_not_run "
                                  "counts them; still not a detection — achromaticity is "
                                  "untested")}
    _write(vet_dir / "summary.json", rep)
    print(f"[growth-direct] vet: {rep['n_vetted']} vetted, {rep['n_survive_vet']} survive")
    return rep


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def direct_run(stage: str = "all", *, out_dir=None, conf: dict | None = None, shard: int = 0,
               n_shards: int = 1, resume: bool = True, budget_s: float | None = None,
               query_fn=None, products_fn=None, tic_fn=None, **vet_kw) -> dict:
    from .run import load_growth_config  # noqa: PLC0415

    conf = conf if conf is not None else load_growth_config()
    out = Path(out_dir) if out_dir else Path("results") / "growth" / "direct"
    out.mkdir(parents=True, exist_ok=True)
    stages = ("probe", "targets", "measure", "assess") if stage in ("all", "", None) \
        else tuple(s.strip() for s in stage.split(","))
    rep: dict = {}
    for s in stages:
        if s == "probe":
            rep = direct_probe(conf, out)
        elif s == "targets":
            rep = direct_targets(conf, out, query_fn=query_fn)
        elif s == "measure":
            rep = direct_measure(conf, out, shard=shard, n_shards=n_shards, products_fn=products_fn,
                                 tic_fn=tic_fn, resume=resume, budget_s=budget_s)
        elif s == "assess":
            rep = direct_assess(conf, out)
        elif s == "control":
            rep = direct_control(conf, out, products_fn=products_fn, tic_fn=tic_fn,
                                 budget_s=budget_s)
        elif s == "vet":
            rep = direct_vet(conf, out, query_fn=query_fn, **vet_kw)
        else:
            raise SystemExit(f"unknown stage {s!r}; choose from {STAGES}")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="seti growth-direct",
        description="GROWTH direct (S57): the TESS-era depth of EVERY confirmed/candidate KOI "
                    "with a TIC id, fitted from the light curves on both SAP and PDCSAP, "
                    "against the KOI depth — sharded, checkpointed, with per-planet sensitivity")
    p.add_argument("--stage", default="all",
                   help="probe | targets | measure | assess | control | vet | "
                        "all (probe,targets,measure,assess) | a comma list")
    p.add_argument("--out-dir", default="results/growth/direct")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--budget-s", type=float, default=None,
                   help="override direct.fetch.shard_budget_s for this shard")
    a = p.parse_args(argv)
    rep = direct_run(a.stage, out_dir=a.out_dir, shard=a.shard, n_shards=a.n_shards,
                     resume=not a.no_resume, budget_s=a.budget_s)
    if isinstance(rep, dict) and rep.get("verdict"):
        print(f"[growth-direct] verdict: {rep['verdict']}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALL_FLUX_COLUMNS", "CANDIDATE_CLASSES", "CANDIDATE_COLUMNS", "CLASSES", "CLASS_CONSISTENT",
    "CLASS_CROWDING", "CLASS_DEEPER", "CLASS_GROWTH", "CLASS_NOT_MEASURABLE",
    "CLASS_NOT_MEASURED", "CLASS_NOT_RECOVERED", "CLASS_SHALLOWER", "CLASS_SHRINK",
    "DEFAULT_AUTHORS", "DIRECT_KOI_COLUMNS", "FAMILIES", "FAMILY_COLUMNS", "FAMILY_PDC",
    "FAMILY_SAP", "MEASUREMENT_COLUMNS", "MEMBER_COLUMNS", "PDC_COLUMNS", "REASONS",
    "REASON_BUDGET", "REASON_NOT_REACHED", "REASON_NO_EPHEMERIS", "REASON_NO_REFERENCE",
    "REASON_NO_TRANSIT", "REASON_QUERY_FAILED", "REASON_TIC_UNRESOLVED", "REASON_ZERO_ROWS",
    "RUN_CANDIDATES", "RUN_NONE", "RUN_NO_DATA", "RUN_VERDICTS", "SAP_COLUMNS", "STAGES",
    "TIC_FALLBACK_ROUTES", "VETOES", "VETO_DISPOSITION", "VETO_DURATION",
    "VETO_DURATION_INCONCLUSIVE", "VETO_EPHEMERIS", "VETO_FPFLAG", "VETO_LOWER_BOUND",
    "VETO_ODD_EVEN", "VETO_ONE_FAMILY", "ClassifyParams", "DurationParams", "EpochSearchParams",
    "FetchParams", "TargetParams", "astroquery_tic_fn", "build_targets", "classify_direct",
    "CONTROL_CLASSES", "CONTROL_COLUMNS", "CTRL_ABOVE", "CTRL_UNAVAILABLE", "CTRL_WITHIN",
    "CTRL_VERDICTS", "apply_control", "compare_family", "control_phase_null",
    "default_products_fn", "detrended_fold", "direct_assess", "direct_control", "direct_measure",
    "direct_probe", "direct_run", "direct_targets", "direct_vet", "family_segments",
    "fetch_products", "fit_duration", "gather_shards", "ingress_fraction",
    "lightkurve_products_fn", "main", "mast_fits_products_fn", "measure_direct_target",
    "measure_family", "population_offsets", "read_tess_lc_fits_all", "search_epoch_offset",
    "select_shard", "shard_of",
]
