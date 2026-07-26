"""Runner-only archive access for KNELL.

The sandbox has no archive egress (every IRSA/Gaia/VizieR call returns
``CONNECT tunnel failed, response 403``), so everything in this module runs on a
GitHub Actions runner.  The ZTF primitives are inherited rather than
reimplemented --- :func:`seti.rust.acquire.iter_region_2band` already pairs g and
r positionally at field scale and is proven --- and this module adds the two
things KNELL needs on top:

1. an **acquisition log** that distinguishes, per stage, between a query that
   *failed* and a query that *succeeded and returned zero rows*.  Three channels
   in this repository have reported ``NO_DATA_REACHED`` while the data sat on
   disk; the cure is to record the query text and the row count of every stage
   into ``summary.json`` so the failure mode is visible in the artefact rather
   than inferred from silence;
2. the **cross-survey secondary layer** --- catalogued variables from VSX and
   GCVS.  This layer is explicitly secondary: it compares a *different* survey's
   detection to ZTF's, so passband, cadence and pipeline all change at the same
   time as the epoch, which is precisely the confound the intra-survey primary
   search exists to avoid.  Every cross-survey candidate therefore has to carry
   the same injection-measured demonstration that ZTF *would have detected* the
   catalogued period and amplitude (:func:`seti.knell.run.crossmatch_demonstration`);
   without it a VSX-minus-ZTF difference is a statement about two telescopes.
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..rust.acquire import iter_region_2band, pair_bands  # noqa: F401  (re-export)

VSX_TABLE = "B/vsx/vsx"
GCVS_TABLE = "B/gcvs/gcvs_cat"


@dataclass
class AcquisitionLog:
    """Per-stage record of what was asked for and what came back.

    ``status`` is one of ``OK`` (rows returned), ``QUERY_RETURNED_ZERO_ROWS``
    (the service answered, with nothing), or ``QUERY_FAILED`` (it did not
    answer).  These are different facts about the world and the summary must not
    collapse them into one.
    """

    stages: list[dict] = field(default_factory=list)

    def record(self, stage: str, query: str, *, rows: int | None = None,
               error: str | None = None, extra: dict | None = None) -> None:
        if error is not None:
            status = "QUERY_FAILED"
        elif rows is None:
            status = "QUERY_FAILED"
        elif rows == 0:
            status = "QUERY_RETURNED_ZERO_ROWS"
        else:
            status = "OK"
        rec = {"stage": stage, "status": status, "rows": int(rows or 0),
               "query": str(query)[:2000]}
        if error:
            rec["error"] = str(error)[:500]
        if extra:
            rec.update(extra)
        self.stages.append(rec)
        print(f"[knell/acquire] {stage}: {status} rows={rec['rows']}"
              + (f" error={rec.get('error')}" if error else ""))

    def as_dict(self) -> dict:
        n_fail = sum(1 for s in self.stages if s["status"] == "QUERY_FAILED")
        n_zero = sum(1 for s in self.stages if s["status"] == "QUERY_RETURNED_ZERO_ROWS")
        n_ok = sum(1 for s in self.stages if s["status"] == "OK")
        return {
            "stages": self.stages,
            "n_stages": len(self.stages), "n_ok": n_ok,
            "n_query_failed": n_fail, "n_query_returned_zero_rows": n_zero,
            "any_query_failed": bool(n_fail > 0),
            "total_rows": int(sum(s["rows"] for s in self.stages)),
        }

    def write(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.as_dict(), indent=2))


def probe_ztf_service(timeout_s: float = 30.0) -> tuple[bool, str]:
    """One tiny live request to the IRSA ZTF light-curve endpoint.

    This exists because of a real provenance bug found while building this
    channel.  The inherited bulk fetcher (:func:`seti.dimming.acquire.fetch_ztf_region`)
    is deliberately defensive: it catches its own exceptions and returns an empty
    dict.  That makes a **403 from the proxy indistinguishable from a genuinely
    empty sky box**, and a sweep over a dozen boxes would then report
    ``QUERY_RETURNED_ZERO_ROWS`` for a run in which no query ever reached IRSA.
    Probing the service once, separately, and recording the outcome as its own
    stage restores the distinction the summary needs.

    Returns ``(reachable, detail)``.
    """
    import requests

    from ..dimming.acquire import ZTF_LC_URL

    params = {"POS": "CIRCLE 180.00000 20.00000 0.00300", "BANDNAME": "r",
              "FORMAT": "CSV", "BAD_CATFLAGS_MASK": "32768"}
    try:
        resp = requests.get(ZTF_LC_URL, params=params, timeout=timeout_s)
    except Exception as exc:                              # noqa: BLE001
        return False, repr(exc)[:300]
    return (resp.status_code == 200), f"HTTP {resp.status_code}, {len(resp.text)} bytes"


def iter_region_2band_logged(ra: float, dec: float, log: AcquisitionLog, **kw):
    """:func:`seti.rust.acquire.iter_region_2band` with a per-field row count.

    The generator is consumed lazily by the caller, so the log entry is written
    when the generator is exhausted --- which is what makes a truncated (time
    budget) sweep distinguishable in the artefact from an empty one.
    """
    n = 0
    t0 = _time.monotonic()
    err = None
    try:
        for pair in iter_region_2band(ra, dec, **kw):
            n += 1
            yield pair
    except Exception as exc:                              # noqa: BLE001
        err = repr(exc)
        raise
    finally:
        log.record(
            "ztf_region_2band",
            f"IRSA ZTF light curves, g+r, centre=({ra:.5f},{dec:.5f}), "
            f"{ {k: v for k, v in kw.items()} }",
            rows=n if err is None else None, error=err,
            extra={"elapsed_s": round(_time.monotonic() - t0, 1)},
        )


def fetch_vsx_region(ra: float, dec: float, radius_deg: float = 0.5,
                     log: AcquisitionLog | None = None,
                     row_limit: int = 50000) -> pd.DataFrame:
    """AAVSO VSX variables in a cone --- the cross-survey secondary layer's input.

    Returns a frame with ``name, ra, dec, vtype, period, mag_max, mag_min`` (empty
    on failure, with the reason recorded in ``log``).  The distinction between an
    empty answer and a failed query is carried by the log, never by the emptiness
    of the frame.
    """
    q = (f"VizieR {VSX_TABLE} cone ({ra:.5f},{dec:.5f}) r={radius_deg} deg "
         f"limit={row_limit}")
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier

        v = Vizier(columns=["**"], row_limit=int(row_limit))
        res = v.query_region(SkyCoord(ra * u.deg, dec * u.deg),
                             radius=radius_deg * u.deg, catalog=VSX_TABLE)
    except Exception as exc:                              # noqa: BLE001
        if log:
            log.record("vsx_cone", q, error=repr(exc))
        return pd.DataFrame()
    if res is None or len(res) == 0:
        if log:
            log.record("vsx_cone", q, rows=0)
        return pd.DataFrame()
    df = res[0].to_pandas()
    df = df.rename(columns={c: c.lower() for c in df.columns})
    ren = {"raj2000": "ra", "dej2000": "dec", "type": "vtype", "period": "period",
           "max": "mag_max", "min": "mag_min", "name": "name"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    if log:
        log.record("vsx_cone", q, rows=int(len(df)))
    return df


def fetch_gcvs_region(ra: float, dec: float, radius_deg: float = 0.5,
                      log: AcquisitionLog | None = None,
                      row_limit: int = 50000) -> pd.DataFrame:
    """GCVS variables in a cone --- the longest-baseline catalogued sample.

    GCVS matters here for one reason: many of its entries were classified from
    photographic plates decades before ZTF, so a GCVS variable that ZTF cannot
    detect is the widest-baseline cessation candidate reachable without going to
    the plate archives themselves.  It is also the layer with the *worst*
    detectability confound, which is why it is secondary and why every candidate
    from it must carry the injection demonstration.
    """
    q = (f"VizieR {GCVS_TABLE} cone ({ra:.5f},{dec:.5f}) r={radius_deg} deg "
         f"limit={row_limit}")
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier

        v = Vizier(columns=["**"], row_limit=int(row_limit))
        res = v.query_region(SkyCoord(ra * u.deg, dec * u.deg),
                             radius=radius_deg * u.deg, catalog=GCVS_TABLE)
    except Exception as exc:                              # noqa: BLE001
        if log:
            log.record("gcvs_cone", q, error=repr(exc))
        return pd.DataFrame()
    if res is None or len(res) == 0:
        if log:
            log.record("gcvs_cone", q, rows=0)
        return pd.DataFrame()
    df = res[0].to_pandas().rename(columns=str.lower)
    if log:
        log.record("gcvs_cone", q, rows=int(len(df)))
    return df


def fetch_gaia_context(positions: pd.DataFrame, radius_arcsec: float = 5.0,
                       log: AcquisitionLog | None = None) -> pd.DataFrame:
    """Gaia DR3 astrometry + the crowding census for a shortlist.

    ``ruwe`` and ``non_single_star`` are not generic quality flags in this
    channel --- they are the direct astrometric test for the **third body** that
    precesses an eclipsing binary out of the line of sight, which is the one
    published mechanism for eclipse cessation (SS Lacertae).
    """
    from ..rust.acquire import fetch_gaia_context as _rust_gaia

    q = (f"Gaia DR3 cone {radius_arcsec}\" around {len(positions)} positions "
         "(ruwe, non_single_star, astrometric_excess_noise, neighbour census)")
    try:
        df = _rust_gaia(positions, radius_arcsec=radius_arcsec)
    except Exception as exc:                              # noqa: BLE001
        if log:
            log.record("gaia_context", q, error=repr(exc))
        return pd.DataFrame()
    n_ok = int(df["gaia_ok"].sum()) if "gaia_ok" in df.columns else int(len(df))
    if log:
        log.record("gaia_context", q, rows=n_ok)
    return df


def fetch_simbad_context(positions: pd.DataFrame, log: AcquisitionLog | None = None
                         ) -> pd.DataFrame:
    """SIMBAD object types for a shortlist (CV / AGN / YSO / RR Lyr rejection)."""
    q = f"SIMBAD cone around {len(positions)} positions"
    try:
        from ..acquire.science import fetch_simbad_context as _sim
        df = _sim(positions)
    except Exception as exc:                              # noqa: BLE001
        if log:
            log.record("simbad_context", q, error=repr(exc))
        return pd.DataFrame()
    if log:
        log.record("simbad_context", q, rows=int(len(df)))
    return df


def crossmatch_positions(df: pd.DataFrame, ra: np.ndarray, dec: np.ndarray,
                         tol_arcsec: float = 2.0) -> np.ndarray:
    """Nearest-neighbour index of each ``(ra, dec)`` in ``df``; -1 where unmatched."""
    if not len(df) or "ra" not in df or "dec" not in df or not len(ra):
        return np.full(len(ra), -1, dtype=int)
    from scipy.spatial import cKDTree

    dec0 = float(np.median(np.asarray(dec, dtype=float)))
    cosd = max(np.cos(np.radians(dec0)), 1e-3)
    tree = cKDTree(np.column_stack([df["ra"].to_numpy(dtype=float) * cosd,
                                    df["dec"].to_numpy(dtype=float)]))
    d, i = tree.query(np.column_stack([np.asarray(ra, dtype=float) * cosd,
                                       np.asarray(dec, dtype=float)]), k=1,
                      distance_upper_bound=tol_arcsec / 3600.0)
    i = np.asarray(i, dtype=int)
    i[~np.isfinite(d)] = -1
    i[i >= len(df)] = -1
    return i


__all__ = ["AcquisitionLog", "GCVS_TABLE", "VSX_TABLE", "crossmatch_positions",
           "fetch_gaia_context", "fetch_gcvs_region", "fetch_simbad_context",
           "fetch_vsx_region", "iter_region_2band", "iter_region_2band_logged",
           "pair_bands", "probe_ztf_service"]
