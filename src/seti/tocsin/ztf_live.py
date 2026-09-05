"""TOCSIN on the LIVE ZTF public alert stream, while Rubin is dark.

WHY THIS EXISTS.  Rubin has been off sky since the night of 13/14 July 2026
(``docs/rubin-outage.md``).  On 2026-09-05 the same check found the *other*
public wide-field alert stream, ZTF's, **live**: its own nightly archive,
ALeRCE's ZTF API and ANTARES all held the night of 2026-09-04.  The repository
had assumed the opposite from two broker facts that were about brokers.  So the
S30 flash/dip screen --- an unclassified achromatic transient on a catalogued
nearby star, in BOTH difference polarities, promoted only on coherence across
nights --- gets a second stream: northern sky (dec >~ -30), r <~ 20.5 per 30 s
visit, both polarities issued as alerts, reachable with no credential.

WHAT IS REUSED, AND WHAT IS NEW.  The funnel (``screen.screen_alerts``), the
ledger (``ledger.Ledger``) and the target model (``targets``) are the Rubin
channel's, untouched: this module only *acquires* and *normalises*.  Two live
services, each documentation-derived until the probe records their real shapes:

* **ALeRCE's ZTF API** (``api.alerce.online/ztf/v1``) --- a REST service, not
  the TAP mirror the Rubin channel reads (whose non-LSST table froze on
  2026-04-30).  ``/objects`` lists objects by ``lastmjd`` window; a night's
  worth is a few hundred pages, matched locally to the nearby-star list;
  ``/objects/{oid}/detections`` and ``/non_detections`` give the matched stars'
  full alert history and their upper limits.
* **IRSA's ZTF exposure table** (``ztf.ztf_current_meta_sci`` over TAP) ---
  every public science quadrant with its corners and its own 5-sigma limit.
  Point-in-quadrant against the propagated target list is THE DENOMINATOR: it
  says which stars were looked at on which night, and how deep, independently of
  whether any alert was issued.  Strictly better than the Rubin path's 1-degree
  detection-footprint proxy.

THE SWEEP, AND WHY IT IS CORRECT FOR A BACKFILL.  Objects are listed by the
window their *newest* detection falls in.  For the live nightly run that is
exactly "alerted last night".  For a backfill the object is met once, in the
window of its last alert, and its FULL history comes with it; an event on an
earlier night is folded if that night has already been screened (its trials are
in the ledger) and is otherwise left for the sweep to reach.  Numerator and
denominator therefore always cover the same nights.  The window never advances
past the exposure table's own frontier, so no night is folded without its
trials.

Everything network-facing is guarded; a failed leg degrades a run to a named
verdict, never to a quiet null.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from .ledger import Ledger, bin_key, night_of
from .photometry import ab_to_njy
from .run import (
    _finite,
    _now_mjd,
    _repo_root,
    _thresholds,
    _utc,
    _write_events,
    _write_json,
    _write_watchlist,
    load_targets,
    load_tocsin_config,
)
from .schema import NormalizedAlert, night_id
from .screen import ScreenVerdict, screen_alerts
from .targets import GAIA_EPOCH, match_alerts_to_targets, position_uncertainty_arcsec, propagate_pm

ALERCE_ZTF_API = "https://api.alerce.online/ztf/v1"
IRSA_TAP_SYNC = "https://irsa.ipac.caltech.edu/TAP/sync"
IRSA_EXPOSURE_TABLE = "ztf.ztf_current_meta_sci"
#: ZTF filter ids -> the band labels the shared schema and the GSPC baseline
#: columns understand.  ZTF g/r/i are close to SDSS g/r/i, the same standing as
#: the ATLAS c/o mapping (docs/tocsin-altfeeds.md 4).
ZTF_FID_BAND = {1: "g", 2: "r", 3: "i"}
ZTF_START_MJD = 58194.0            # public survey from 2018-03
#: Per-axis astrometric uncertainty assigned to a ZTF difference-image centroid.
#: ZTF alerts carry none, and the schema's 50 mas floor is Rubin's.  MEASURED on
#: the first live run (results/tocsin_ztf, 2026-09-05): the 18 detections of
#: three catalogued nearby stars sat 0.16-1.4 arcsec from the propagated Gaia
#: position, median ~0.5, while ZTF's own `distnr` put each within 0.1 arcsec
#: of its reference source -- the difference-image centroid of a bright star's
#: residual scatters at a few tenths of an arcsecond.  With the Rubin floor all
#: 18 were rejected as astrometric offsets at 3-28 sigma.  0.25 arcsec per axis
#: (0.35 total) makes 1 arcsec the 3-sigma line; `max_sep_arcsec` still bounds.
ZTF_ASTROMETRIC_FLOOR_ARCSEC = 0.25
JD_MINUS_MJD = 2400000.5
LN10_OVER_2P5 = math.log(10.0) / 2.5

#: Overridable under ``ztf:`` in ``config/tocsin.yaml``.
DEFAULTS: dict = {
    "alerce_ztf_api": ALERCE_ZTF_API,
    "irsa_tap_url": IRSA_TAP_SYNC,
    "timeout_s": 120.0,
    "page_size": 1000,
    #: Parallel keyset walks over equal sub-ranges of the window (run 5: ~18 s
    #: of service latency per page, so a night is half an hour serially).
    "sweep_workers": 4,
    #: Northern list: ZTF reaches dec ~ -31.  The Rubin list stops at +15.
    "dec_min": -31.0,
    "dec_max": 90.0,
    #: Where the sweep starts with an empty ledger: MJD 61235 = the night of
    #: 2026-07-14, Rubin's last night.  The ZTF channel exists to cover the
    #: Rubin-dark interval, and at ~8 min a night (four sweep workers) that is
    #: reached in a week of nightly runs; earlier nights can be folded later
    #: with explicit windows, which never move the watermark.
    "backfill_start_mjd": 61235.0,
    "max_nights_per_run": 3.0,
    "lookback_nights": 1.0,
    #: The stream is served within hours; the exposure table lags more, and it
    #: is the exposure table that caps the window (see `frontier`).
    "ingest_lag_days": 0.3,
    "max_run_seconds": 5400.0,
    #: Only public-survey quadrants issue public alerts (IRSA `ipac_gid` = 1).
    "public_gid": 1,
    #: Stop fetching per-object history after this many matched objects in one
    #: window: a night that matches thousands of nearby stars is a bad night
    #: (a reference-image change), not a discovery, and it should be looked at.
    "max_matched_objects": 3000,
    "results_dir": "results/tocsin_ztf",
    "ledger_path": "results/tocsin_ztf/ledger.json",
}


class ZtfLiveError(RuntimeError):
    """A live leg failed in a way the run can name."""


def ztf_config(cfg=None) -> tuple[dict, dict]:
    """``(tocsin_conf, ztf_conf)`` with ``ztf:`` overrides applied over DEFAULTS."""
    conf = load_tocsin_config(cfg)
    z = dict(DEFAULTS)
    z.update(conf.get("ztf") or {})
    return conf, z


def _session(timeout: float):
    import requests

    class _S(requests.Session):
        def request(self, *args, **kwargs):      # noqa: D102
            kwargs.setdefault("timeout", timeout)
            return super().request(*args, **kwargs)

    return _S()


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ---------------------------------------------------------------------------
# ALeRCE's ZTF API
# ---------------------------------------------------------------------------
class AlerceZtfAPI:
    """Thin client for ``api.alerce.online/ztf/v1``: objects by window, per-object history."""

    def __init__(self, base: str = ALERCE_ZTF_API, timeout: float = 120.0,
                 page_size: int = 1000, sleep=None, workers: int = 4):
        self.base = base.rstrip("/")
        self.timeout = float(timeout)
        self.page_size = int(page_size)
        self.workers = max(1, int(workers))
        self.calls = 0
        self.notes: list[str] = []
        self._s = None
        self._sleep = sleep or time.sleep

    def _sess(self):
        if self._s is None:
            self._s = _session(self.timeout)
        return self._s

    #: Back-off between retries of one request, seconds.  MEASURED 2026-09-05:
    #: the service answers HTTP 500 ("overloaded or an error in the application")
    #: under a long paged sweep; three quick retries were not enough.
    RETRY_WAITS_S = (5.0, 15.0, 30.0, 60.0, 120.0)

    def _get(self, path: str, params: list[tuple[str, str]] | dict | None = None,
             retries: int | None = None, sess=None):
        """GET with bounded retries on 429/5xx.  Raises :class:`ZtfLiveError`."""
        url = f"{self.base}/{path.lstrip('/')}"
        last = ""
        retries = len(self.RETRY_WAITS_S) if retries is None else int(retries)
        sess = sess if sess is not None else self._sess()
        for attempt in range(retries + 1):
            try:
                r = sess.get(url, params=params)
                self.calls += 1
            except Exception as exc:                               # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"[:200]
                if attempt < retries:
                    self._sleep(2.0 * (attempt + 1))
                    continue
                raise ZtfLiveError(f"GET {path}: {last}") from exc
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                wait = self.RETRY_WAITS_S[min(attempt, len(self.RETRY_WAITS_S) - 1)]
                ra = r.headers.get("Retry-After")
                if ra and str(ra).isdigit():
                    wait = min(120.0, float(ra))
                self._sleep(wait)
                continue
            if r.status_code >= 400:
                raise ZtfLiveError(f"GET {path}: HTTP {r.status_code} {r.text[:200]}")
            try:
                return r.json()
            except ValueError as exc:
                raise ZtfLiveError(f"GET {path}: not JSON: {r.text[:200]}") from exc
        raise ZtfLiveError(f"GET {path}: gave up after {retries} retries ({last})")

    def objects_in_window(self, mjd_lo: float, mjd_hi: float,
                          deadline: float | None = None,
                          max_pages: int | None = None,
                          workers: int | None = None) -> tuple[list[dict], dict]:
        """Every object whose NEWEST detection falls in ``[mjd_lo, mjd_hi)``.

        KEYSET PAGINATION ON ``lastmjd``, not page offsets.  MEASURED 2026-09-05
        across four live runs: with ``count=false`` the service never numbers
        pages (run 1 stopped after one); ordering by ``oid`` makes every page a
        walk of the object-id index filtered by ``lastmjd``, so a NARROW window
        is SLOWER (a quarter-night drew nginx 504 on page 1, run 4) and a deep
        offset in a wide window drew HTTP 500 after 27 minutes (run 3).  Ordering
        by the filtered column itself and advancing the window's lower bound to
        the last epoch seen turns every request into a short index range scan
        from a fresh start.  Objects sharing the boundary epoch are returned
        twice and de-duplicated by ``oid``; a page whose newest epoch does not
        move the cursor is nudged by a microsecond so the walk cannot stall.

        IN PARALLEL SUB-RANGES.  MEASURED run 5: a page of 1000 takes ~18 s of
        service latency, so a night (~95 pages) is half an hour serially.  The
        window is split into ``workers`` equal sub-ranges, each walked by its own
        keyset cursor on its own session; the results are merged and
        de-duplicated.  Four workers is a modest load on a public API.
        """
        n_workers = max(1, int(workers if workers is not None else self.workers))
        lo, hi = float(mjd_lo), float(mjd_hi)
        edges = np.linspace(lo, hi, n_workers + 1)
        ranges = [(float(edges[i]), float(edges[i + 1])) for i in range(n_workers)
                  if edges[i + 1] > edges[i]]
        stats = {"pages": 0, "truncated": False, "page_size": self.page_size,
                 "pagination": "keyset_lastmjd", "workers": len(ranges)}
        if len(ranges) <= 1:
            rows, part = self._walk(lo, hi, deadline, max_pages, self._sess())
            stats.update({k: v for k, v in part.items() if k != "rows"})
            stats["objects"] = len(rows)
            return rows, stats
        from concurrent.futures import ThreadPoolExecutor
        per_worker_pages = None if max_pages is None else max(1, int(max_pages) // len(ranges))
        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            futures = [pool.submit(self._walk, a, b, deadline, per_worker_pages,
                                   self._new_session()) for a, b in ranges]
            parts = [f.result() for f in futures]
        rows: list[dict] = []
        seen: set[str] = set()
        for part_rows, part in parts:
            stats["pages"] += int(part.get("pages", 0))
            if part.get("truncated"):
                stats["truncated"] = True
                if part.get("error") and not stats.get("error"):
                    stats["error"] = part["error"]
            for r in part_rows:
                oid = str(r.get("oid"))
                if oid not in seen:
                    seen.add(oid)
                    rows.append(r)
        stats["objects"] = len(rows)
        return rows, stats

    def _new_session(self):
        return _session(self.timeout)

    def _walk(self, lo: float, hi: float, deadline, max_pages, sess
              ) -> tuple[list[dict], dict]:
        """One keyset walk over ``[lo, hi)`` on one session (thread-safe by construction)."""
        rows: list[dict] = []
        seen: set[str] = set()
        stats = {"pages": 0, "truncated": False}
        cursor = float(lo)
        hi = float(hi)
        while cursor < hi:
            if deadline is not None and time.monotonic() > deadline:
                stats["truncated"] = True
                self.notes.append(f"objects sweep stopped by the deadline after "
                                  f"{stats['pages']} pages at MJD {cursor:.5f}")
                break
            if max_pages is not None and stats["pages"] >= int(max_pages):
                stats["truncated"] = True
                break
            if stats["pages"] >= self.MAX_SWEEP_PAGES:
                stats["truncated"] = True
                self.notes.append(f"objects sweep stopped at MAX_SWEEP_PAGES="
                                  f"{self.MAX_SWEEP_PAGES}")
                break
            params = [("lastmjd", f"{cursor:.6f}"), ("lastmjd", f"{hi:.6f}"),
                      ("page", "1"), ("page_size", str(self.page_size)),
                      ("order_by", "lastmjd"), ("order_mode", "ASC"), ("count", "false")]
            try:
                payload = self._get("objects", params, sess=sess)
            except ZtfLiveError as exc:
                stats["truncated"] = True
                stats["error"] = str(exc)[:400]
                self.notes.append(f"objects sweep at MJD {cursor:.5f} failed after retries: "
                                  f"{str(exc)[:200]}")
                break
            items = (payload or {}).get("items") if isinstance(payload, dict) else payload
            items = [r for r in (items or []) if isinstance(r, dict)]
            stats["pages"] += 1
            newest = None
            for r in items:
                v = _num(r.get("lastmjd"))
                if v is not None:
                    newest = v if newest is None else max(newest, v)
                oid = str(r.get("oid"))
                if oid not in seen:
                    seen.add(oid)
                    rows.append(r)
            if len(items) < self.page_size or newest is None:
                break
            cursor = newest if newest > cursor else cursor + 1e-6
        stats["cursor_end_mjd"] = cursor
        return rows, stats

    #: A full ZTF night is a few hundred pages of 1000; this is a runaway guard.
    MAX_SWEEP_PAGES = 5000

    def detections(self, oid: str) -> list[dict]:
        payload = self._get(f"objects/{oid}/detections")
        if payload is None:
            return []
        return [r for r in (payload if isinstance(payload, list) else
                            (payload.get("items") or payload.get("detections") or []))
                if isinstance(r, dict)]

    def non_detections(self, oid: str) -> list[dict]:
        payload = self._get(f"objects/{oid}/non_detections")
        if payload is None:
            return []
        return [r for r in (payload if isinstance(payload, list) else
                            (payload.get("items") or payload.get("non_detections") or []))
                if isinstance(r, dict)]

    def frontier(self) -> float | None:
        """Newest ``lastmjd`` the API serves, from one object ordered by it."""
        payload = self._get("objects", [("order_by", "lastmjd"), ("order_mode", "DESC"),
                                        ("page_size", "1"), ("page", "1"), ("count", "false")])
        items = (payload or {}).get("items") if isinstance(payload, dict) else payload
        for r in items or []:
            v = _num((r or {}).get("lastmjd"))
            if v is not None:
                return v
        return None

    def describe(self, now: float | None = None) -> dict:
        """Record the LIVE shapes: a small objects page, one object's history."""
        now = _now_mjd() if now is None else float(now)
        rec: dict = {"url": self.base, "reached": False}
        try:
            rec["frontier_mjd"] = self.frontier()
            rec["reached"] = True
        except Exception as exc:                                   # noqa: BLE001
            rec["frontier_error"] = str(exc)[:300]
        try:
            payload = self._get("objects", [("lastmjd", f"{now - 2.0:.5f}"),
                                            ("lastmjd", f"{now:.5f}"),
                                            ("page", "1"), ("page_size", "3"),
                                            ("order_by", "oid"), ("order_mode", "ASC"),
                                            ("count", "false")])
            rec["reached"] = True
            rec["objects_page_keys"] = sorted((payload or {}).keys()) if isinstance(payload, dict) else "list"
            items = (payload or {}).get("items") if isinstance(payload, dict) else payload
            rec["objects_page_head"] = json.dumps(payload, default=str)[:1500]
            if items:
                first = items[0]
                rec["object_keys"] = sorted(first.keys())
                oid = str(first.get("oid"))
                dets = self.detections(oid)
                nd = self.non_detections(oid)
                rec["sample_oid"] = oid
                rec["detection_keys"] = sorted(dets[0].keys()) if dets else []
                rec["detection_head"] = json.dumps(dets[:2], default=str)[:1500]
                rec["non_detection_keys"] = sorted(nd[0].keys()) if nd else []
                rec["non_detection_head"] = json.dumps(nd[:2], default=str)[:600]
                rec["n_detections_sample"] = len(dets)
                rec["n_non_detections_sample"] = len(nd)
        except Exception as exc:                                   # noqa: BLE001
            rec["error"] = str(exc)[:400]
        rec["calls"] = self.calls
        return rec


# ---------------------------------------------------------------------------
# IRSA's exposure table --- the denominator
# ---------------------------------------------------------------------------
EXPOSURE_COLUMNS = ("obsjd", "fid", "field", "ccdid", "qid", "ra", "dec",
                    "ra1", "dec1", "ra2", "dec2", "ra3", "dec3", "ra4", "dec4",
                    "maglimit", "exptime", "ipac_gid")
#: IRSA's group id for the public survey (1 = public, 2 = partnership, 3 =
#: Caltech).  The metadata table carries `ipac_gid`, not the alert packet's
#: `programid`; the first live run asked for `programid` and got a VOTable error.
PUBLIC_GID_COLUMN = "ipac_gid"


def votable_error(text: str) -> str | None:
    """The QUERY_STATUS error message inside a VOTable error document, or None."""
    if not text or not text.lstrip().startswith("<"):
        return None
    m = re.search(r'<INFO[^>]*name="QUERY_STATUS"[^>]*value="ERROR"[^>]*>(.*?)</INFO>',
                  text, re.S | re.I)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()[:600]
    m = re.search(r"<INFO[^>]*>(.*?)</INFO>", text, re.S | re.I)
    return (re.sub(r"\s+", " ", m.group(1)).strip()[:600] if m else text[:300])


class IrsaZtfExposures:
    """Public ZTF science-quadrant metadata over IRSA's synchronous TAP."""

    def __init__(self, url: str = IRSA_TAP_SYNC, timeout: float = 300.0,
                 table: str = IRSA_EXPOSURE_TABLE):
        self.url = url
        self.timeout = float(timeout)
        self.table = table
        self.calls = 0
        self.notes: list[str] = []
        self._s = None

    def _sess(self):
        if self._s is None:
            self._s = _session(self.timeout)
        return self._s

    def query(self, adql: str, maxrec: int = 2_000_000) -> list[dict]:
        r = self._sess().get(self.url, params={"QUERY": adql, "LANG": "ADQL",
                                               "REQUEST": "doQuery", "FORMAT": "csv",
                                               "MAXREC": str(int(maxrec))})
        self.calls += 1
        if r.status_code >= 400:
            raise ZtfLiveError(f"IRSA TAP HTTP {r.status_code}: {r.text[:300]}")
        text = r.text or ""
        err = votable_error(text)
        if err is not None:
            raise ZtfLiveError(f"IRSA TAP error: {err}")
        reader = csv.DictReader(io.StringIO(text))
        return [{(k or "").strip().lower(): v for k, v in row.items()} for row in reader]

    def exposures(self, mjd_lo: float, mjd_hi: float, public_gid: int | None = 1) -> list[dict]:
        """Public science quadrants in ``[mjd_lo, mjd_hi)``, one TAP query per night.

        A night is ~45k quadrant rows; asking for a window night by night keeps
        every response far inside the service's row limit instead of relying on
        a MAXREC the service may silently clip.
        """
        out: list[dict] = []
        lo = float(mjd_lo)
        while lo < float(mjd_hi):
            hi = min(float(mjd_hi), lo + 1.0)
            where = [f"obsjd >= {lo + JD_MINUS_MJD:.6f}", f"obsjd < {hi + JD_MINUS_MJD:.6f}"]
            if public_gid is not None:
                where.append(f"{PUBLIC_GID_COLUMN} = {int(public_gid)}")
            adql = (f"SELECT {', '.join(EXPOSURE_COLUMNS)} FROM {self.table} "
                    f"WHERE {' AND '.join(where)}")
            out.extend(self.query(adql, maxrec=500_000))
            lo = hi
        return out

    #: Lookbacks for the frontier query, days.  MEASURED 2026-09-05: MAX(obsjd)
    #: over the whole table read-timed out at 240 s twice; bounded by obsjd the
    #: same MAX is an index walk.  Widened only when a slice is empty.
    FRONTIER_LOOKBACKS_D = (3.0, 10.0, 40.0, 120.0, 400.0)

    def frontier(self, public_gid: int | None = 1, now: float | None = None) -> float | None:
        now = _now_mjd() if now is None else float(now)
        for back in self.FRONTIER_LOOKBACKS_D:
            where = [f"obsjd > {now - back + JD_MINUS_MJD:.6f}"]
            if public_gid is not None:
                where.append(f"{PUBLIC_GID_COLUMN} = {int(public_gid)}")
            rows = self.query(f"SELECT MAX(obsjd) AS obsjd_max FROM {self.table} "
                              f"WHERE {' AND '.join(where)}", maxrec=5)
            for r in rows:
                v = _num(r.get("obsjd_max"))
                if v is not None:
                    return v - JD_MINUS_MJD
        return None

    def describe(self, now: float | None = None) -> dict:
        now = _now_mjd() if now is None else float(now)
        rec: dict = {"url": self.url, "table": self.table, "reached": False}
        try:
            fr = self.frontier()
            rec["frontier_mjd"] = fr
            rec["public_gid_column"] = PUBLIC_GID_COLUMN
            rec["frontier_lag_days"] = None if fr is None else round(now - fr, 2)
            rec["reached"] = True
            lo = (fr - 1.0) if fr is not None else now - 3.0
            rows = self.exposures(lo, lo + 1.0)
            rec["sample_rows"] = len(rows)
            rec["sample_columns"] = sorted(rows[0].keys()) if rows else []
            rec["sample_head"] = json.dumps(rows[:2], default=str)[:1200]
            fids = {}
            for r in rows:
                fids[str(r.get("fid"))] = fids.get(str(r.get("fid")), 0) + 1
            rec["sample_fid_counts"] = fids
        except Exception as exc:                                   # noqa: BLE001
            rec["error"] = str(exc)[:400]
        rec["calls"] = self.calls
        return rec


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def _isdiffpos_sign(v) -> float | None:
    """ZTF's ``isdiffpos``: 't'/'1'/True -> +1 (brighter than reference), 'f'/'0'/False -> -1."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return 1.0 if v else -1.0
    s = str(v).strip().lower()
    if s in ("t", "true", "1", "+1", "1.0"):
        return 1.0
    if s in ("f", "false", "0", "-1", "-1.0"):
        return -1.0
    return None


def normalize_alerce_ztf_detections(oid: str, dets: list[dict]) -> list[NormalizedAlert]:
    """ALeRCE ZTF detections -> normalised alerts in nJy, signed by ``isdiffpos``.

    ``magpsf`` is the magnitude of the |difference| flux; ``isdiffpos`` carries
    the sign, and a dip is a real ZTF alert exactly as a flash is.  The star's
    quiescent flux comes from ALeRCE's ``magpsf_corr``, the difference photometry
    combined with the reference-catalogue magnitude at the same position: total
    flux minus the signed difference is the reference flux, in the SAME band and
    system as the difference --- the analogue of Rubin's ``templateFlux``.  Where
    ``corrected`` is false (no reference source within 1.4") that is left None
    and the funnel falls back to the Gaia GSPC baseline with its passband error,
    as it does on the Rubin path.

    Quality: ``drb`` (the deep-learning real/bogus score) is the reliability,
    ``rb`` when ``drb`` is absent; ALeRCE's ``dubious`` flag (a corrected
    magnitude inconsistent with its own reference) is the pixel-flag stand-in.
    ZTF alerts carry no per-detection astrometric error, so the schema's floor
    applies.  Nothing here says solar-system, dipole or trail: those tests are
    recorded as unavailable by the funnel and the ledger's recurrence
    requirement carries the load, as it does for the alternative feeds.
    """
    out: list[NormalizedAlert] = []
    for r in dets:
        mag = _num(r.get("magpsf"))
        sig = _num(r.get("sigmapsf"))
        mjd = _num(r.get("mjd"))
        sign = _isdiffpos_sign(r.get("isdiffpos"))
        if mag is None or sig is None or mjd is None or sign is None:
            continue
        try:
            band = ZTF_FID_BAND.get(int(r.get("fid")), "")
        except (TypeError, ValueError):
            band = ""
        if not band:
            continue
        flux_abs = float(ab_to_njy(mag))
        dflux = sign * flux_abs
        dflux_err = flux_abs * sig * LN10_OVER_2P5
        template = template_err = None
        corrected = r.get("corrected")
        if isinstance(corrected, str):
            corrected = corrected.strip().lower() in ("t", "true", "1")
        mag_corr = _num(r.get("magpsf_corr"))
        if corrected and mag_corr is not None:
            total = float(ab_to_njy(mag_corr))
            ref = total - dflux
            if ref > 0:
                template = ref
                sig_corr = _num(r.get("sigmapsf_corr_ext")) or _num(r.get("sigmapsf_corr"))
                template_err = (total * float(sig_corr) * LN10_OVER_2P5
                                if sig_corr is not None else None)
        rel = _num(r.get("drb"))
        rel_version = "drb"
        if rel is None:
            rel = _num(r.get("rb"))
            rel_version = "rb" if rel is not None else None
        dubious = r.get("dubious")
        if isinstance(dubious, str):
            dubious = dubious.strip().lower() in ("t", "true", "1")
        out.append(NormalizedAlert(
            alert_id=str(r.get("candid") or f"{oid}:{mjd:.6f}:{band}"),
            object_id=str(oid),
            mjd=float(mjd), band=band,
            ra=_num(r.get("ra")) or float("nan"), dec=_num(r.get("dec")) or float("nan"),
            dflux_njy=float(dflux), dflux_err_njy=float(dflux_err),
            broker="alerce-ztf",
            ra_err_arcsec=ZTF_ASTROMETRIC_FLOOR_ARCSEC,
            dec_err_arcsec=ZTF_ASTROMETRIC_FLOOR_ARCSEC,
            template_flux_njy=template, template_flux_err_njy=template_err,
            snr=float(dflux / dflux_err) if dflux_err > 0 else None,
            reliability=rel, reliability_version=rel_version,
            is_negative=(sign < 0),
            pixel_flag_bad=(bool(dubious) if dubious is not None else None),
            raw={"oid": str(oid), "candid": r.get("candid"), "fid": r.get("fid"),
                 "magpsf": mag, "sigmapsf": sig, "magpsf_corr": mag_corr,
                 "corrected": bool(corrected) if corrected is not None else None,
                 "distnr_arcsec": _num(r.get("distnr")),
                 "diffmaglim": _num(r.get("diffmaglim")),
                 "isdiffpos": r.get("isdiffpos"), "rb": _num(r.get("rb")),
                 "drb": _num(r.get("drb")), "survey": "ztf"},
        ))
    return out


def upper_limits(nondets: list[dict]) -> list[tuple[float, str, float]]:
    """ALeRCE non-detections -> ``(mjd, band, diffmaglim)`` --- the epochs the field
    was imaged and this position showed nothing above the 5-sigma limit."""
    out = []
    for r in nondets:
        mjd = _num(r.get("mjd"))
        lim = _num(r.get("diffmaglim"))
        try:
            band = ZTF_FID_BAND.get(int(r.get("fid")), "")
        except (TypeError, ValueError):
            band = ""
        if mjd is None or not band:
            continue
        out.append((float(mjd), band, float(lim) if lim is not None else float("nan")))
    return out


# ---------------------------------------------------------------------------
# Matching objects to the target list
# ---------------------------------------------------------------------------
def _propagated(targets, epoch_jyear: float, th):
    t_ra = np.asarray(targets["ra"], dtype=float)
    t_dec = np.asarray(targets["dec"], dtype=float)
    n = t_ra.size
    pmra = np.asarray(targets["pmra"], dtype=float) if "pmra" in targets else np.zeros(n)
    pmdec = np.asarray(targets["pmdec"], dtype=float) if "pmdec" in targets else np.zeros(n)
    p_ra, p_dec = propagate_pm(t_ra, t_dec, pmra, pmdec, to_epoch=epoch_jyear)
    sig = position_uncertainty_arcsec(
        targets["pmra_error"] if "pmra_error" in targets else np.zeros(n),
        targets["pmdec_error"] if "pmdec_error" in targets else np.zeros(n),
        dt_yr=epoch_jyear - GAIA_EPOCH,
        pm_missing=~(np.isfinite(pmra) & np.isfinite(pmdec)),
        missing_pm_penalty_arcsec=th.missing_pm_penalty_arcsec)
    ids = (np.asarray(targets["source_id"]).astype(str) if "source_id" in targets
           else np.arange(n).astype(str))
    return p_ra, p_dec, sig, ids


def match_objects(objects: list[dict], targets, th, epoch_jyear: float) -> dict[str, str]:
    """``oid -> target_id`` for objects whose mean position sits on a target.

    The object's ``meanra``/``meandec`` is the mean of its detections, so for a
    high-proper-motion star it lags the star by up to the drift over the
    object's lifetime; the match radius is therefore the funnel's
    ``match_radius_arcsec`` plus the propagation uncertainty, and the per-alert
    association inside the funnel (at each night's epoch) is what decides.
    """
    if not objects or targets is None or len(targets) == 0:
        return {}
    ra = np.array([_num(o.get("meanra")) or np.nan for o in objects])
    dec = np.array([_num(o.get("meandec")) or np.nan for o in objects])
    ok = np.isfinite(ra) & np.isfinite(dec)
    if not ok.any():
        return {}
    p_ra, p_dec, sig, ids = _propagated(targets, epoch_jyear, th)
    m = match_alerts_to_targets(ra[ok], dec[ok], p_ra, p_dec,
                                radius_arcsec=max(2.0, th.match_radius_arcsec),
                                target_pos_err_arcsec=sig)
    idx_ok = np.nonzero(ok)[0]
    out: dict[str, str] = {}
    for ai, ti in zip(m.alert_index, m.target_index, strict=True):
        oid = str(objects[int(idx_ok[int(ai)])].get("oid"))
        out[oid] = str(ids[int(ti)])
    return out


# ---------------------------------------------------------------------------
# The denominator: point-in-quadrant against the propagated target list
# ---------------------------------------------------------------------------
def _quad_polygon_contains(ra_c, dec_c, corners_ra, corners_dec, p_ra, p_dec) -> np.ndarray:
    """Ray-casting point-in-polygon on a tangent plane about the quadrant centre.

    ZTF quadrants are ~0.85 degrees on a side, so a gnomonic projection about
    the centre is accurate to far better than a pixel over the whole footprint.
    """
    def proj(ra, dec):
        ra0, dec0 = math.radians(ra_c), math.radians(dec_c)
        r = np.radians(np.asarray(ra, dtype=float))
        d = np.radians(np.asarray(dec, dtype=float))
        cosc = np.sin(dec0) * np.sin(d) + np.cos(dec0) * np.cos(d) * np.cos(r - ra0)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.cos(d) * np.sin(r - ra0) / cosc
            y = (np.cos(dec0) * np.sin(d) - np.sin(dec0) * np.cos(d) * np.cos(r - ra0)) / cosc
        return x, y

    px, py = proj(p_ra, p_dec)
    cx, cy = proj(np.asarray(corners_ra), np.asarray(corners_dec))
    inside = np.zeros(px.shape, dtype=bool)
    n = len(cx)
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = cx[i], cy[i], cx[j], cy[j]
        cond = (yi > py) != (yj > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj - xi) * (py - yi) / (yj - yi) + xi
        inside ^= cond & (px < xint)
        j = i
    return inside


def quadrant_footprint(exposures: list[dict], targets, th, epoch_jyear: float
                       ) -> tuple[set, dict, dict, dict, dict]:
    """Which targets each public quadrant covered, per night, with its limit.

    Returns ``(pairs, observed_bands, limits, visit_epochs, stats)``:

    * ``pairs`` --- ``{(target_id, night)}``: the trials.
    * ``observed_bands`` --- ``(target_id, night) -> {band}``, for the funnel's
      one-sided non-detection test.
    * ``limits`` --- ``(target_id, night, band) -> [maglimit]``, the quadrant's
      own 5-sigma limits on the visits that covered the star.
    * ``visit_epochs`` --- ``target_id -> [mjd]``: every exposure epoch that
      covered the star, the exact per-visit history the ledger's timing null
      wants.
    """
    from scipy.spatial import cKDTree

    stats = {"exposure_rows": len(exposures or [])}
    pairs: set = set()
    bands: dict = {}
    limits: dict = {}
    epochs: dict[str, list[float]] = {}
    if not exposures or targets is None or len(targets) == 0:
        return pairs, bands, limits, epochs, stats
    p_ra, p_dec, _sig, ids = _propagated(targets, epoch_jyear, th)
    from .targets import _unit_vectors
    tree = cKDTree(_unit_vectors(p_ra, p_dec))
    # A quadrant's half-diagonal is ~0.6 degrees; 0.75 with margin.
    chord = 2.0 * math.sin(math.radians(0.75) / 2.0)
    n_quads = 0
    n_bad = 0
    nights: set = set()
    for r in exposures:
        try:
            ra_c, dec_c = float(r["ra"]), float(r["dec"])
            cr = [float(r[f"ra{i}"]) for i in (1, 2, 3, 4)]
            cd = [float(r[f"dec{i}"]) for i in (1, 2, 3, 4)]
            obsjd = float(r["obsjd"])
        except (KeyError, TypeError, ValueError):
            n_bad += 1
            continue
        mjd = obsjd - JD_MINUS_MJD
        try:
            band = ZTF_FID_BAND.get(int(float(r.get("fid"))), "")
        except (TypeError, ValueError):
            band = ""
        lim = _num(r.get("maglimit"))
        n_quads += 1
        cand = tree.query_ball_point(_unit_vectors(np.array([ra_c]), np.array([dec_c]))[0],
                                     r=chord)
        if not cand:
            continue
        cand = np.asarray(cand, dtype=int)
        inside = _quad_polygon_contains(ra_c, dec_c, cr, cd, p_ra[cand], p_dec[cand])
        if not inside.any():
            continue
        night = night_id(mjd)
        nights.add(night)
        for k in cand[inside]:
            tid = str(ids[int(k)])
            key = (tid, night)
            pairs.add(key)
            if band:
                bands.setdefault(key, set()).add(band)
                if lim is not None:
                    limits.setdefault((tid, night, band), []).append(lim)
            epochs.setdefault(tid, []).append(round(mjd, 6))
    stats["quadrants"] = n_quads
    stats["quadrant_rows_unparsable"] = n_bad
    stats["footprint_nights"] = len(nights)
    stats["footprint_star_nights"] = len(pairs)
    stats["footprint_targets"] = len(epochs)
    return pairs, bands, limits, epochs, stats


def proxy_footprint(objects: list[dict], targets, th, epoch_jyear: float,
                    nights: set, bin_deg: float = 1.0) -> tuple[set, dict, dict]:
    """The Rubin path's denominator, for nights IRSA's exposure table has not reached.

    Every object that alerted on a night marks the 1-degree sky bin it lies in
    as observed that night; a target inside an observed bin is a trial.  The
    swept objects are exactly "everything that alerted", so the bins trace
    where the camera pointed, at the cost of no per-visit limit and no band
    coverage --- those nights' events cannot use the one-sided non-detection
    test and the summary says so.  Returns ``(pairs, visit_epochs, stats)``.
    """
    stats = {"proxy_objects": 0, "proxy_bins": 0, "proxy_star_nights": 0}
    pairs: set = set()
    epochs: dict[str, list[float]] = {}
    if not objects or targets is None or len(targets) == 0 or not nights:
        return pairs, epochs, stats
    observed: set = set()
    for o in objects:
        ra, dec, mjd = _num(o.get("meanra")), _num(o.get("meandec")), _num(o.get("lastmjd"))
        if ra is None or dec is None or mjd is None:
            continue
        night = night_id(mjd)
        if night not in nights:
            continue
        observed.add((int(math.floor(ra / bin_deg)), int(math.floor(dec / bin_deg)), night))
        stats["proxy_objects"] += 1
    if not observed:
        return pairs, epochs, stats
    p_ra, p_dec, _sig, ids = _propagated(targets, epoch_jyear, th)
    rab = np.floor(p_ra / bin_deg).astype(int)
    decb = np.floor(p_dec / bin_deg).astype(int)
    by_night: dict[str, set] = {}
    for a, d, n in observed:
        by_night.setdefault(n, set()).add((a, d))
    for night, bins in by_night.items():
        hit = np.fromiter(((int(a), int(d)) in bins for a, d in zip(rab, decb, strict=True)),
                          dtype=bool, count=rab.size)
        try:
            n_int = int(night.lstrip("n"))
        except ValueError:
            continue
        for idx in np.nonzero(hit)[0]:
            tid = str(ids[idx])
            pairs.add((tid, night))
            epochs.setdefault(tid, []).append(round(n_int + 1.1666667, 6))
    stats["proxy_bins"] = len(observed)
    stats["proxy_star_nights"] = len(pairs)
    return pairs, epochs, stats


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------
def probe(cfg=None, out_dir: str | Path | None = None) -> dict:
    """Runner-only: record both services' live shapes before any science claim."""
    conf, z = ztf_config(cfg)
    root = Path(cfg.root) if cfg is not None else _repo_root()
    out = Path(out_dir) if out_dir else root / z["results_dir"]
    now = _now_mjd()
    rec: dict = {"probed_at_utc": _utc(), "now_mjd": round(now, 5), "verdict": "NOT_RUN"}
    try:
        rec["alerce_ztf"] = AlerceZtfAPI(z["alerce_ztf_api"], timeout=float(z["timeout_s"]),
                                         page_size=int(z["page_size"])).describe(now)
    except Exception as exc:                                       # noqa: BLE001
        rec["alerce_ztf"] = {"error": str(exc)[:400], "reached": False}
    try:
        rec["irsa_exposures"] = IrsaZtfExposures(z["irsa_tap_url"],
                                                 timeout=float(z["timeout_s"]) * 2).describe(now)
    except Exception as exc:                                       # noqa: BLE001
        rec["irsa_exposures"] = {"error": str(exc)[:400], "reached": False}
    a_ok = bool(rec["alerce_ztf"].get("reached")) and rec["alerce_ztf"].get("detection_keys")
    i_ok = bool(rec["irsa_exposures"].get("reached")) and rec["irsa_exposures"].get("sample_rows")
    rec["verdict"] = ("OK" if a_ok and i_ok else
                      "PARTIAL" if (a_ok or i_ok) else "NO_SERVICE_REACHED")
    _write_json(out / "probe.json", rec)
    print(f"[tocsin-ztf] probe verdict={rec['verdict']} "
          f"alerce_frontier={rec['alerce_ztf'].get('frontier_mjd')} "
          f"irsa_frontier={rec['irsa_exposures'].get('frontier_mjd')}")
    return rec


# ---------------------------------------------------------------------------
# Targets: the northern list
# ---------------------------------------------------------------------------
def build_ztf_targets(cfg=None, out_path: str | Path | None = None) -> dict:
    """The nearby-star list over ZTF's sky (dec > -31), cached like tocsin's."""
    from .run import build_targets
    conf, z = ztf_config(cfg)
    root = Path(cfg.root) if cfg is not None else _repo_root()
    out = Path(out_path) if out_path else root / ".cache" / "tocsin_ztf" / "targets.parquet"
    return build_targets(cfg, out_path=out, dec_min=float(z["dec_min"]),
                         dec_max=float(z["dec_max"]))


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------
def _merge_verdicts(parts: list[ScreenVerdict]) -> ScreenVerdict:
    v = ScreenVerdict()
    for p in parts:
        v.events.extend(p.events)
        v.rejected.extend(p.rejected)
        for k, n in p.counts.items():
            if isinstance(n, int):
                v.counts[k] = v.counts.get(k, 0) + n
        for tid, mjds in p.visit_history.items():
            v.visit_history.setdefault(tid, []).extend(mjds)
        v.target_positions.update(p.target_positions)
        v.notes.extend(p.notes)
        v.nights.extend(p.nights)
    v.nights = sorted(set(v.nights))
    return v


def screen_window(cfg=None, mjd_lo: float | None = None, mjd_hi: float | None = None,
                  targets_path: str | Path | None = None,
                  out_dir: str | Path | None = None,
                  deadline: float | None = None,
                  api=None, irsa=None) -> dict:
    """Sweep, match, fetch, footprint, screen and ledger one window.  Runner-only
    unless ``api``/``irsa`` are injected (the tests)."""
    conf, z = ztf_config(cfg)
    root = Path(cfg.root) if cfg is not None else _repo_root()
    out = Path(out_dir) if out_dir else root / z["results_dir"]
    th = _thresholds(conf)
    ledger_path = root / z["ledger_path"] if out_dir is None else out / "ledger.json"
    led = Ledger.load(ledger_path)
    explicit = mjd_lo is not None or mjd_hi is not None
    api = api or AlerceZtfAPI(z["alerce_ztf_api"], timeout=float(z["timeout_s"]),
                              page_size=int(z["page_size"]),
                              workers=int(z.get("sweep_workers", 4)))
    irsa = irsa or IrsaZtfExposures(z["irsa_tap_url"], timeout=float(z["timeout_s"]) * 2)
    now = _now_mjd()
    t0 = time.monotonic()
    summary: dict = {"run_at_utc": _utc(), "explicit_window": explicit, "verdict": "NOT_RUN",
                     "counts": {}, "notes": [], "timings_s": {}}

    # THE WINDOW.  Capped at the STREAM's frontier and the wall clock.  MEASURED
    # 2026-09-05: IRSA's public exposure table runs ~60 days behind the stream,
    # so capping there would make a "live" screen two months late.  Nights the
    # exposure table has reached get the exact quadrant denominator; later
    # nights get the detection-footprint proxy (step 4), and the summary says
    # which nights got which.
    frontiers: dict = {}
    try:
        frontiers["irsa_exposures_mjd"] = irsa.frontier(int(z["public_gid"]))
    except Exception as exc:                                       # noqa: BLE001
        summary["notes"].append(f"irsa_frontier_failed: {str(exc)[:200]}")
    if explicit:
        hi = float(mjd_hi) if mjd_hi is not None else now
        lo = float(mjd_lo) if mjd_lo is not None else hi - float(z["lookback_nights"])
    else:
        try:
            frontiers["alerce_ztf_mjd"] = api.frontier()
        except Exception as exc:                                   # noqa: BLE001
            summary["notes"].append(f"alerce_frontier_failed: {str(exc)[:200]}")
        caps = [now - float(z["ingest_lag_days"])]
        if frontiers.get("alerce_ztf_mjd") is not None:
            caps.append(float(frontiers["alerce_ztf_mjd"]))
        hi_cap = min(caps)
        wm = _finite(led.last_mjd_screened)
        lo = wm if wm is not None else float(z["backfill_start_mjd"])
        hi = min(hi_cap, lo + float(z["max_nights_per_run"]))
    summary.update({"mjd_lo": round(lo, 5), "mjd_hi": round(hi, 5), "frontiers": frontiers,
                    "nights": sorted({night_of(lo), night_of(hi)})})
    if hi <= lo:
        summary["verdict"] = "NO_NEW_DATA"
        summary["notes"].append("the watermark has caught up with the newest epoch both "
                                "services hold; nothing to screen until they advance")
        _write_json(out / "summary.json", summary)
        print(f"[tocsin-ztf] NO_NEW_DATA (watermark {lo:.3f} >= frontier {hi:.3f})")
        return summary

    tpath = (Path(targets_path) if targets_path
             else root / ".cache" / "tocsin_ztf" / "targets.parquet")
    targets = load_targets(tpath)
    if targets is None or len(targets) == 0:
        summary["verdict"] = "NO_TARGET_LIST"
        summary["notes"].append(f"target list missing at {tpath}; run tocsin-ztf-targets")
        _write_json(out / "summary.json", summary)
        return summary
    summary["n_targets"] = int(len(targets))
    epoch_jyear = 2000.0 + ((lo + hi) / 2.0 - 51544.5) / 365.25

    # 1. The sweep: every object whose newest detection is in the window.
    t1 = time.monotonic()
    try:
        objects, ostats = api.objects_in_window(lo, hi, deadline=deadline)
    except ZtfLiveError as exc:
        summary["verdict"] = "NO_DATA_REACHED"
        summary["error"] = str(exc)[:600]
        _write_json(out / "summary.json", summary)
        print(f"[tocsin-ztf] NO_DATA_REACHED: {summary['error']}")
        return summary
    summary["timings_s"]["objects_sweep"] = round(time.monotonic() - t1, 1)
    summary["counts"]["objects_in_window"] = len(objects)
    summary["counts"]["objects_pages"] = ostats.get("pages", 0)
    summary["counts"]["objects_sweep_pages"] = ostats.get("pages", 0)
    summary["notes"].extend(getattr(api, "notes", [])[:10])
    if ostats.get("truncated"):
        # An incomplete numerator is not folded: the trials would be counted
        # once and the events partially, and the next run (same window, the
        # watermark unmoved) could not add the missing events to a night whose
        # trials it must not add twice.  Nothing is written; the run says why.
        summary["verdict"] = "SWEEP_TRUNCATED"
        summary["error"] = ostats.get("error")
        summary["notes"].append("objects sweep TRUNCATED (server error, deadline or page "
                                "cap): this window's numerator is incomplete, nothing is "
                                "folded and the watermark is not advanced")
        summary["watermark_mjd"] = _finite(led.last_mjd_screened)
        summary["timings_s"]["total"] = round(time.monotonic() - t0, 1)
        _write_json(out / "summary.json", summary)
        print(f"[tocsin-ztf] SWEEP_TRUNCATED after {ostats.get('pages')} pages: "
              f"{ostats.get('error')}")
        return summary

    # 2. Match to the nearby-star list.
    matched = match_objects(objects, targets, th, epoch_jyear)
    summary["counts"]["objects_matched"] = len(matched)
    if len(matched) > int(z["max_matched_objects"]):
        summary["notes"].append(
            f"{len(matched)} matched objects exceeds max_matched_objects "
            f"{z['max_matched_objects']}: a night that alerts on thousands of "
            f"catalogued stars is a reference-image problem, not a discovery; only "
            f"the first {z['max_matched_objects']} are fetched")
        matched = dict(list(matched.items())[:int(z["max_matched_objects"])])

    # 3. Per-object history: all detections, all upper limits.
    t2 = time.monotonic()
    alerts: list[NormalizedAlert] = []
    ul_epochs: dict[str, list[float]] = {}
    ul_limits: dict = {}
    n_fetched = 0
    n_failed = 0
    for oid, tid in matched.items():
        if deadline is not None and time.monotonic() > deadline:
            summary["notes"].append(f"per-object fetch stopped by the deadline after "
                                    f"{n_fetched} of {len(matched)} objects")
            break
        try:
            dets = api.detections(oid)
            nd = api.non_detections(oid)
        except ZtfLiveError as exc:
            n_failed += 1
            summary["notes"].append(f"{oid}: {str(exc)[:160]}")
            continue
        n_fetched += 1
        alerts.extend(normalize_alerce_ztf_detections(oid, dets))
        for mjd, band, lim in upper_limits(nd):
            ul_epochs.setdefault(tid, []).append(round(mjd, 6))
            if math.isfinite(lim):
                ul_limits.setdefault((tid, night_id(mjd), band), []).append(lim)
    summary["timings_s"]["object_history"] = round(time.monotonic() - t2, 1)
    summary["counts"]["objects_fetched"] = n_fetched
    summary["counts"]["objects_failed"] = n_failed
    summary["counts"]["detections_pulled"] = len(alerts)
    summary["counts"]["upper_limits_pulled"] = sum(len(v) for v in ul_epochs.values())

    # 4. The denominator: the public quadrants that covered each star, for the
    #    nights the exposure table has reached; the detection-footprint proxy
    #    (where tonight's alerted objects lie, in 1-degree bins -- the Rubin
    #    path's own denominator) for the nights it has not.
    t3 = time.monotonic()
    footprint_pairs: set = set()
    observed_bands: dict = {}
    fp_limits: dict = {}
    fp_epochs: dict = {}
    irsa_fr = frontiers.get("irsa_exposures_mjd")
    exact_hi = min(hi, float(irsa_fr)) if irsa_fr is not None else lo
    window_nights = {f"n{n}" for n in range(night_of(lo), night_of(hi) + 1)}
    # A night is exact only if the exposure table has reached its END: a night
    # the table holds half of would count half its exposures as the whole.
    # Night n spans MJD n + 16/24 to n + 1 + 16/24 (schema.NIGHT_BOUNDARY_FRAC).
    exact_nights = ({f"n{n}" for n in range(night_of(lo), night_of(hi) + 1)
                     if irsa_fr is not None and (n + 1.0 + 16.0 / 24.0) <= float(irsa_fr)}
                    if exact_hi > lo else set())
    proxy_nights = window_nights - exact_nights
    summary["denominator_by_night"] = {n: ("quadrant_exact" if n in exact_nights
                                           else "detection_proxy")
                                       for n in sorted(window_nights)}
    try:
        exposures = irsa.exposures(lo, exact_hi, int(z["public_gid"])) if exact_hi > lo else []
        footprint_pairs, observed_bands, fp_limits, fp_epochs, fstats = quadrant_footprint(
            exposures, targets, th, epoch_jyear)
        # Quadrant coverage is authoritative only for the nights the table has
        # reached; a quadrant row that somehow labels a later night is dropped
        # so the two denominators never mix on one night.
        footprint_pairs = {p for p in footprint_pairs if p[1] in exact_nights}
        summary["counts"].update(fstats)
        if proxy_nights:
            ppairs, pepochs, pstats = proxy_footprint(objects, targets, th, epoch_jyear,
                                                      proxy_nights)
            footprint_pairs |= ppairs
            for tid, mjds in pepochs.items():
                fp_epochs.setdefault(tid, []).extend(mjds)
            summary["counts"].update(pstats)
            summary["notes"].append(
                f"{len(proxy_nights)} of {len(window_nights)} nights lie beyond IRSA's "
                f"exposure frontier (MJD {irsa_fr}) and use the detection-footprint "
                f"proxy: 1-degree bins holding any alerted object that night, no "
                f"per-visit limit, no band coverage")
    except Exception as exc:                                       # noqa: BLE001
        # NO DENOMINATOR, NO FOLD.  The first live run folded four nights with
        # zero trials when this query failed, and advanced the watermark past
        # them; the backfill rule then reads those nights as screened.  A night
        # without its trials is not screened, so nothing is written and the
        # next run asks for the same window again.
        summary["notes"].append(f"footprint_query_failed: {str(exc)[:600]}")
        summary["verdict"] = "NO_DENOMINATOR"
        summary["denominator"] = "unavailable"
        summary["timings_s"]["footprint"] = round(time.monotonic() - t3, 1)
        summary["watermark_mjd"] = _finite(led.last_mjd_screened)
        _write_json(out / "summary.json", summary)
        print(f"[tocsin-ztf] NO_DENOMINATOR: the exposure query failed; window "
              f"{lo:.3f}-{hi:.3f} is NOT marked screened")
        return summary
    summary["timings_s"]["footprint"] = round(time.monotonic() - t3, 1)

    # 5. Screen, one night at a time, with that night's own limits.
    by_night: dict[str, list[NormalizedAlert]] = {}
    for a in alerts:
        by_night.setdefault(a.night, []).append(a)
    parts: list[ScreenVerdict] = []
    for night in sorted(by_night):
        batch = by_night[night]
        ej = 2000.0 + (float(np.median([a.mjd for a in batch])) - 51544.5) / 365.25
        band_limits: dict[str, float] = {}
        lims_tonight: dict[str, list[float]] = {}
        for (_tid, n, band), vals in list(fp_limits.items()) + list(ul_limits.items()):
            if n == night:
                lims_tonight.setdefault(band, []).extend(vals)
        for band, vals in lims_tonight.items():
            band_limits[band] = float(ab_to_njy(float(np.median(vals))))
        ob_tonight = {k: v for k, v in observed_bands.items() if k[1] == night}
        for (tid, n, band) in ul_limits:
            if n == night:
                ob_tonight.setdefault((tid, night), set()).add(band)
        parts.append(screen_alerts(batch, targets, th, epoch_jyear=ej,
                                   observed_bands=ob_tonight, band_limits=band_limits))
    verdict = _merge_verdicts(parts)
    summary["counts"].update({k: v for k, v in verdict.counts.items() if isinstance(v, int)})
    summary["notes"].extend(n for n in verdict.notes if n != "no_alerts_in")

    # 6. Which events are folded: those in the window, and those on nights whose
    #    trials are already in the ledger (the backfill case; module docstring).
    folded_nights_before = set(led.nights)
    events_in = [ev for ev in verdict.events
                 if ev.night in window_nights or ev.night in folded_nights_before]
    summary["counts"]["events_kept"] = len(verdict.events)
    summary["counts"]["events_folded"] = len(events_in)
    summary["counts"]["events_deferred_to_sweep"] = len(verdict.events) - len(events_in)

    # 7. Denominator bookkeeping, mirroring `run.screen_night`.
    event_pairs = {(ev.target_id, ev.night) for ev in events_in}
    all_pairs = set(footprint_pairs) | {p for p in event_pairs if p[1] in window_nights}
    trials_by_night: dict[str, int] = {}
    for _tid, night in all_pairs:
        trials_by_night[night] = trials_by_night.get(night, 0) + 1
    tpos = verdict.target_positions
    _ra = np.asarray(targets["ra"], dtype=float)
    _dec = np.asarray(targets["dec"], dtype=float)
    _ids = (np.asarray(targets["source_id"]).astype(str) if "source_id" in targets
            else np.arange(_ra.size).astype(str))
    _pos = dict(zip(_ids, zip(_ra, _dec, strict=True), strict=True))
    bin_trials_tonight: dict[str, dict[str, int]] = {}
    for tid, night in all_pairs:
        rd = tpos.get(tid) or _pos.get(str(tid))
        if not rd:
            continue
        k = bin_key(rd[0], rd[1])
        if k:
            bin_trials_tonight.setdefault(night, {})
            bin_trials_tonight[night][k] = bin_trials_tonight[night].get(k, 0) + 1
    n_fp = len(footprint_pairs)
    n_total = len(all_pairs)
    summary["counts"]["target_nights_screened"] = n_total
    summary["counts"]["target_nights_from_footprint"] = n_fp
    summary["counts"]["target_nights_detection_only"] = len(
        {p for p in event_pairs if p[1] in window_nights} - set(footprint_pairs))
    summary["footprint_coverage_fraction"] = round(n_fp / n_total, 4) if n_total else 0.0
    if n_total == 0:
        summary["denominator"] = "unavailable"
    elif n_fp > 2 * max(len(event_pairs), 1):
        summary["denominator"] = ("quadrant_footprint_exact" if not proxy_nights else
                                  "detection_footprint_proxy" if not exact_nights else
                                  "mixed_quadrant_exact_and_detection_proxy")
    else:
        summary["denominator"] = "detection_dominated_lower_bound"
        summary["notes"].append("the quadrant footprint covers few of the screened "
                                "star-nights; the rate is an UPPER bound")

    # The visit history: every exposure epoch that covered the star, plus the
    # object's own upper limits and detections.
    hist: dict[str, list[float]] = {}
    for tid, mjds in fp_epochs.items():
        hist.setdefault(tid, []).extend(mjds)
    for tid, mjds in ul_epochs.items():
        hist.setdefault(tid, []).extend(mjds)
    for tid, mjds in verdict.visit_history.items():
        hist.setdefault(tid, []).extend(mjds)
    for tid in list(hist):
        hist[tid] = sorted(set(hist[tid]))

    # 8. Fold.
    if not led.opened_utc:
        led.opened_utc = _utc()
    events_by_night: dict[str, list] = {}
    for ev in events_in:
        events_by_night.setdefault(ev.night, []).append(ev)
    all_nights = sorted(set(trials_by_night) | set(events_by_night) | window_nights)
    first = True
    for night in all_nights:
        led.add_night(night, events_by_night.get(night, []),
                      target_visits=trials_by_night.get(night, 0),
                      targets_in_footprint=len(fp_epochs) or len(targets),
                      alerts_seen=len(alerts) if first else 0,
                      visit_history=None, target_positions=verdict.target_positions,
                      bin_trials=bin_trials_tonight.get(night))
        first = False
    led.apply_visit_history(hist)
    led.updated_utc = _utc()
    lconf = conf["ledger"]
    stats = led.assess(alpha_fdr=float(lconf["alpha_fdr"]),
                       min_visits_for_rate=int(lconf["min_visits_for_rate"]),
                       max_duty_cycle=float(lconf["max_duty_cycle"]),
                       n_null_timing=int(lconf["n_null_timing"]),
                       timing_alpha=float(lconf["timing_alpha"]),
                       max_grey_z=float(conf["screen"]["max_grey_z"]),
                       mixed_polarity_requires_grey_both=bool(
                           lconf.get("mixed_polarity_requires_grey_both", True)))
    if not explicit:
        led.last_mjd_screened = float(hi)
    led.save(ledger_path)
    summary["ledger"] = stats
    wm = _finite(led.last_mjd_screened)
    summary["watermark_mjd"] = None if wm is None else round(wm, 5)
    summary["nights_folded"] = all_nights
    summary["verdict"] = "OK" if (objects or footprint_pairs) else "NO_DETECTIONS_IN_WINDOW"
    _write_events(out, verdict, conf)
    _write_watchlist(out, led, conf)
    summary["timings_s"]["total"] = round(time.monotonic() - t0, 1)
    _write_json(out / "summary.json", summary)
    print(f"[tocsin-ztf] {summary['verdict']}: {len(objects)} objects, {len(matched)} on "
          f"targets, {len(alerts)} detections, {len(events_in)} events folded, "
          f"{n_total} star-nights; tiers={stats['tier_counts']}")
    return summary


def screen(cfg=None, chunks: int = 1, max_run_seconds: float | None = None,
           mjd_lo: float | None = None, mjd_hi: float | None = None,
           targets_path: str | Path | None = None, out_dir: str | Path | None = None
           ) -> list[dict]:
    """Walk up to ``chunks`` consecutive windows inside one wall-clock budget."""
    _conf, z = ztf_config(cfg)
    root = Path(cfg.root) if cfg is not None else _repo_root()
    out_path = Path(out_dir) if out_dir else root / z["results_dir"]
    budget = float(max_run_seconds if max_run_seconds is not None else z["max_run_seconds"])
    started = time.monotonic()
    deadline = started + budget
    out: list[dict] = []
    run: dict = {"run_at_utc": _utc(), "budget_s": budget, "chunks": [], "notes": []}
    last_duration = None
    for i in range(max(1, int(chunks))):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            run["notes"].append(f"budget of {budget:.0f}s spent after {i} chunks")
            break
        # DO NOT START A CHUNK THAT CANNOT FINISH.  MEASURED run 5: the second
        # chunk began with 43 minutes left, swept for 39 of them and was
        # truncated by the deadline -- 127 pages of service time that folded
        # nothing.  A chunk is started only if the previous one would fit.
        if last_duration is not None and remaining < 1.1 * last_duration:
            run["notes"].append(f"chunk {i + 1} not started: {remaining:.0f}s left, the "
                                f"previous chunk took {last_duration:.0f}s")
            print(f"[tocsin-ztf] {run['notes'][-1]}")
            break
        t0 = time.monotonic()
        rec = screen_window(cfg, mjd_lo=mjd_lo, mjd_hi=mjd_hi, targets_path=targets_path,
                            out_dir=out_dir, deadline=deadline)
        last_duration = time.monotonic() - t0
        out.append(rec)
        run["chunks"].append({k: rec.get(k) for k in (
            "verdict", "mjd_lo", "mjd_hi", "denominator", "denominator_by_night",
            "watermark_mjd", "counts", "timings_s", "frontiers", "error")}
            | {"ledger_tiers": (rec.get("ledger") or {}).get("tier_counts"),
               "ledger_visits": (rec.get("ledger") or {}).get("cumulative_target_visits"),
               "notes": (rec.get("notes") or [])[:12]})
        if rec.get("verdict") in ("NO_NEW_DATA", "NO_TARGET_LIST", "NO_DATA_REACHED",
                                  "SWEEP_TRUNCATED", "NO_DENOMINATOR"):
            break
        if mjd_lo is not None or mjd_hi is not None:
            break
    run["elapsed_s"] = round(time.monotonic() - started, 1)
    _write_json(out_path / "run.json", run)
    return out


def assess_only(cfg=None, out_dir: str | Path | None = None) -> dict:
    """Recompute tiers over the accumulated ZTF ledger, offline."""
    conf, z = ztf_config(cfg)
    root = Path(cfg.root) if cfg is not None else _repo_root()
    out = Path(out_dir) if out_dir else root / z["results_dir"]
    ledger_path = root / z["ledger_path"] if out_dir is None else out / "ledger.json"
    led = Ledger.load(ledger_path)
    lconf = conf["ledger"]
    stats = led.assess(alpha_fdr=float(lconf["alpha_fdr"]),
                       min_visits_for_rate=int(lconf["min_visits_for_rate"]),
                       max_duty_cycle=float(lconf["max_duty_cycle"]),
                       n_null_timing=int(lconf["n_null_timing"]),
                       timing_alpha=float(lconf["timing_alpha"]),
                       max_grey_z=float(conf["screen"]["max_grey_z"]),
                       mixed_polarity_requires_grey_both=bool(
                           lconf.get("mixed_polarity_requires_grey_both", True)))
    led.save(ledger_path)
    _write_watchlist(out, led, conf)
    rec = {"assessed_at_utc": _utc(), **stats}
    _write_json(out / "assessment.json", rec)
    return rec


__all__ = ["AlerceZtfAPI", "IrsaZtfExposures", "normalize_alerce_ztf_detections",
           "upper_limits", "match_objects", "quadrant_footprint", "proxy_footprint", "probe",
           "build_ztf_targets", "screen_window", "screen", "assess_only"]
