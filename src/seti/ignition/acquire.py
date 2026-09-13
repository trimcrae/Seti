"""NEOWISE single-exposure photometry for IGNITION.

Runner-only where it touches the network (the sandbox has no archive egress);
the frame cleaning, epoch binning and per-star grouping are pure and are what
the offline suite exercises.

What VIGIL taught, and what this module does about it
-----------------------------------------------------
* **Per-star cones are too slow for a catalogue-scale sample.**  VIGIL measured
  ~92 s for one star (a ``COUNT(*)`` plus the cone, both synchronous) on
  ``neowiser_p1bs_psd``.  Its fix --- one field-wide cone per field, exposures
  assigned to stars locally with a KD-tree --- ran at 377 s for 2.8 M rows and
  matched 186/187 stars.  That path is reused here verbatim
  (:func:`seti.vigil.acquire.fetch_neowise_field`,
  :func:`seti.vigil.run.group_neowise_by_star`).
* **Proper motion is propagated to the mission mid-epoch** and every match
  radius is widened by half the star's mission-long sweep, as VIGIL does; a
  200 mas/yr star otherwise drifts out of its own aperture.
* **A batched upload join is tried first** (``TAP_UPLOAD``), because it is the
  only route that scales to an all-sky subsample, but it is *probed*, never
  assumed: the probe stage records whether IRSA accepted it, and the acquire
  stage picks the route from that record.
* **Every completed batch is written immediately** (CSV append + a progress
  file listing finished ``source_id``\\s), so a killed shard resumes.
* **VIGIL's final summary said ``NO_DATA_REACHED`` over eight field summaries
  that said ``SEARCHED``** --- the aggregator ran on a checkout in which the
  per-field artefacts were not present and reported the absence as a science
  verdict.  IGNITION's assess stage records how many shard files it found
  against how many were expected, and the workflow fails loudly when the count
  is zero (see ``run.py`` and ``.github/workflows/ignition.yml``).
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..vigil.acquire import (
    GAIA_EPOCH,
    IRSA_TAP,
    NEOWISE_END,
    NEOWISE_MID_EPOCH,
    NEOWISE_START,
    QueryResult,
    pm_sweep_arcsec,
    propagate_pm,
    run_tap,
)

NEOWISE_TABLE = "neowiser_p1bs_psd"
NEOWISE_COLS = ("ra", "dec", "mjd", "w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro",
                "qual_frame", "saa_sep", "moon_masked", "cc_flags", "ph_qual", "nb", "na")

DEFAULT_ACQUIRE: dict = {
    "route": "auto",              # auto | upload | field | cone
    "cone_radius_arcsec": 2.5,
    "match_tol_arcsec": 2.5,
    "field_w1_max": 13.0,
    "field_max_rows": 4_000_000,
    "upload_chunk": 200,
    "checkpoint_every": 50,
    "epoch_gap_days": 90.0,
    "min_exp_per_epoch": 5,
    "epoch_err_floor_mag": 0.005,
    "clip_sigma": 5.0,
    "qual_frame_min": 1,
    "cc_flags_ok": ["0000"],
    "ph_qual_ok": ["A", "B"],
    "time_budget_s": 18000.0,
}


def mjd_to_year(mjd) -> np.ndarray:
    """Julian year from MJD."""
    return 2000.0 + (np.asarray(mjd, float) - 51544.5) / 365.25


# --------------------------------------------------------------------------
# Pure: frame cleaning, quality bookkeeping, epoch binning
# --------------------------------------------------------------------------
def clean_frames(df: pd.DataFrame, conf: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """NEOWISE frame-quality cuts, identical in both bands, with a ledger.

    Keeps ``qual_frame > 0``, ``saa_sep > 0``, ``moon_masked == '00'``,
    ``cc_flags`` in the allowed set (default ``'0000'``) and ``ph_qual`` A/B in
    *both* bands.  ``nb`` / ``na`` (blend count, active deblending) are not cut
    here --- they are counted per star and handed to the vet, because a star
    whose every frame is deblended is a different statement from one bad frame.
    """
    c = {**DEFAULT_ACQUIRE, **(conf or {})}
    d = df.copy()
    d.columns = [str(x).lower() for x in d.columns]
    ledger = {"n_raw": int(len(d))}
    if "qual_frame" in d:
        q = pd.to_numeric(d["qual_frame"], errors="coerce").fillna(0)
        keep = q >= int(c["qual_frame_min"])
        ledger["cut_qual_frame"] = int((~keep).sum())
        d = d[keep]
    if "saa_sep" in d:
        sep = pd.to_numeric(d["saa_sep"], errors="coerce")
        keep = sep.isna() | (sep > 0)
        ledger["cut_saa_sep"] = int((~keep).sum())
        d = d[keep]
    if "moon_masked" in d:
        mm = d["moon_masked"].astype(str).str.strip()
        keep = ~mm.str.contains("1")
        ledger["cut_moon_masked"] = int((~keep).sum())
        d = d[keep]
    if "cc_flags" in d:
        ok = {str(x) for x in c["cc_flags_ok"]}
        cc = d["cc_flags"].astype(str).str.strip()
        keep = cc.isin(ok) | (cc.str[:2] == "00")
        ledger["cut_cc_flags"] = int((~keep).sum())
        d = d[keep]
    if "ph_qual" in d:
        ok = {str(x).upper() for x in c["ph_qual_ok"]}
        pq = d["ph_qual"].astype(str).str.upper().str.strip()
        keep = pq.str[:1].isin(ok) & pq.str[1:2].isin(ok)
        ledger["cut_ph_qual"] = int((~keep).sum())
        d = d[keep]
    for col in ("mjd", "w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro"):
        if col in d:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    need = [x for x in ("mjd", "w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro") if x in d]
    if need:
        d = d.dropna(subset=need)
    ledger["n_clean"] = int(len(d))
    return d.reset_index(drop=True), ledger


def star_quality(raw: pd.DataFrame) -> dict:
    """Per-star blend / deblend fractions, measured on the *raw* frames."""
    out = {"n_exp_raw": int(len(raw)), "frac_nb_gt1": float("nan"),
           "frac_na_gt0": float("nan")}
    if not len(raw):
        return out
    d = raw.copy()
    d.columns = [str(x).lower() for x in d.columns]
    if "nb" in d:
        nb = pd.to_numeric(d["nb"], errors="coerce")
        out["frac_nb_gt1"] = float(np.nanmean((nb > 1).to_numpy(dtype=float)))
    if "na" in d:
        na = pd.to_numeric(d["na"], errors="coerce")
        out["frac_na_gt0"] = float(np.nanmean((na > 0).to_numpy(dtype=float)))
    return out


def exposures_to_epochs(mjd, mag, magerr, gap_days: float = 90.0, min_exp: int = 5,
                        clip_sigma: float = 5.0, err_floor: float = 0.005) -> pd.DataFrame:
    """Group single exposures into epochs separated by gaps > ``gap_days``.

    NEOWISE returns to a field roughly every six months, so a 90-day gap cuts
    cleanly between visits everywhere except the ecliptic poles, where the
    visits themselves stretch --- and there the binning still holds because the
    gap between successive pole visits remains > 90 d.  Per epoch: the median
    magnitude and a robust error, ``max(1.4826 * MAD / sqrt(n), err_floor)``,
    after a MAD clip of single bad frames.  Epochs with fewer than ``min_exp``
    surviving exposures are dropped.
    """
    t = np.asarray(mjd, float)
    m = np.asarray(mag, float)
    e = np.asarray(magerr, float)
    ok = np.isfinite(t) & np.isfinite(m)
    t, m, e = t[ok], m[ok], e[ok]
    cols = ["epoch", "t_mjd", "t_yr", "mag", "err", "n_exp", "scatter"]
    if t.size == 0:
        return pd.DataFrame(columns=cols)
    order = np.argsort(t)
    t, m, e = t[order], m[order], e[order]
    edges = np.nonzero(np.diff(t) > gap_days)[0] + 1
    rows = []
    for k, g in enumerate(np.split(np.arange(t.size), edges)):
        if g.size < min_exp:
            continue
        mv, tv = m[g], t[g]
        med = float(np.median(mv))
        mad = 1.4826 * float(np.median(np.abs(mv - med)))
        if mad > 0:
            keep = np.abs(mv - med) <= clip_sigma * mad
            if keep.sum() >= min_exp:
                mv, tv = mv[keep], tv[keep]
                med = float(np.median(mv))
                mad = 1.4826 * float(np.median(np.abs(mv - med)))
        n = int(mv.size)
        if mad <= 0:
            # Degenerate scatter (quantised magnitudes): fall back on the quoted errors.
            mad = float(np.nanmedian(e[g])) if np.isfinite(np.nanmedian(e[g])) else err_floor
        err = max(mad / np.sqrt(n), err_floor)
        rows.append({"epoch": k, "t_mjd": float(np.mean(tv)),
                     "t_yr": float(mjd_to_year(np.mean(tv))), "mag": med,
                     "err": float(err), "n_exp": n, "scatter": float(mad)})
    return pd.DataFrame(rows, columns=cols)


def star_epochs(frames: pd.DataFrame, conf: dict | None = None) -> pd.DataFrame:
    """Both bands of one star's cleaned frames -> long epoch table."""
    c = {**DEFAULT_ACQUIRE, **(conf or {})}
    out = []
    for band, mcol, ecol in (("W1", "w1mpro", "w1sigmpro"), ("W2", "w2mpro", "w2sigmpro")):
        if mcol not in frames:
            continue
        ep = exposures_to_epochs(frames["mjd"], frames[mcol], frames.get(ecol, np.nan),
                                 gap_days=float(c["epoch_gap_days"]),
                                 min_exp=int(c["min_exp_per_epoch"]),
                                 clip_sigma=float(c["clip_sigma"]),
                                 err_floor=float(c["epoch_err_floor_mag"]))
        ep.insert(0, "band", band)
        out.append(ep)
    if not out:
        return pd.DataFrame(columns=["band", "epoch", "t_mjd", "t_yr", "mag", "err",
                                     "n_exp", "scatter"])
    return pd.concat(out, ignore_index=True)


def epochs_to_series(ep: pd.DataFrame) -> dict:
    """Long epoch table for one star -> ``{"W1": (t_yr, mag, err), "W2": ...}``."""
    series = {}
    for band, g in ep.groupby("band"):
        g = g.sort_values("t_yr")
        series[str(band)] = (g["t_yr"].to_numpy(float), g["mag"].to_numpy(float),
                             g["err"].to_numpy(float))
    return series


def positions_at_neowise_epoch(stars: pd.DataFrame, from_epoch: float = GAIA_EPOCH,
                               to_epoch: float = NEOWISE_MID_EPOCH) -> pd.DataFrame:
    """PM-propagated positions plus the per-star match radius widening."""
    pmra = pd.to_numeric(stars.get("pmra", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
    pmdec = pd.to_numeric(stars.get("pmdec", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
    ra_m, dec_m = propagate_pm(stars["ra"].to_numpy(float), stars["dec"].to_numpy(float),
                               pmra, pmdec, from_epoch, to_epoch)
    sweep = np.hypot(pmra, pmdec) * (NEOWISE_END - NEOWISE_START) / 1000.0
    return pd.DataFrame({"source_id": stars["source_id"].astype(str).to_numpy(),
                         "ra_mid": ra_m, "dec_mid": dec_m, "sweep_arcsec": sweep})


def group_by_star(frames: pd.DataFrame, stars: pd.DataFrame,
                  tol_arcsec: float = 2.5) -> dict[str, pd.DataFrame]:
    """Assign field-level exposures to stars (PM-propagated KD-tree; VIGIL's)."""
    from ..vigil.run import group_neowise_by_star

    return group_neowise_by_star(frames, stars, tol_arcsec=tol_arcsec)


# --------------------------------------------------------------------------
# Runner: the three routes
# --------------------------------------------------------------------------
def _cols() -> str:
    return ", ".join(NEOWISE_COLS)


def fetch_neowise_cone(ra: float, dec: float, pmra: float = 0.0, pmdec: float = 0.0,
                       radius_arcsec: float = 2.5, retries: int = 3) -> QueryResult:
    """One PM-propagated cone, *without* the ``COUNT(*)`` that doubled VIGIL's cost."""
    ra_m, dec_m = propagate_pm(ra, dec, pmra, pmdec, GAIA_EPOCH, NEOWISE_MID_EPOCH)
    rad = float(radius_arcsec + 0.5 * pm_sweep_arcsec(pmra, pmdec))
    q = (f"SELECT {_cols()} FROM {NEOWISE_TABLE} WHERE "
         f"1 = CONTAINS(POINT('ICRS', ra, dec), "
         f"CIRCLE('ICRS', {float(ra_m):.7f}, {float(dec_m):.7f}, {rad / 3600.0:.9f}))")
    return run_tap(IRSA_TAP, q, label=f"neowise_cone_pm{pm_sweep_arcsec(pmra, pmdec):.2f}as",
                   retries=retries, async_first=False)


def upload_query(radius_arcsec: float = 2.5) -> str:
    """The ``TAP_UPLOAD`` join.  ``rad`` is per star (cone + half the PM sweep)."""
    cols = ", ".join(f"n.{c}" for c in NEOWISE_COLS)
    return (f"SELECT p.sid, {cols} FROM {NEOWISE_TABLE} AS n, TAP_UPLOAD.pos AS p "
            f"WHERE 1 = CONTAINS(POINT('ICRS', n.ra, n.dec), "
            f"CIRCLE('ICRS', p.ra, p.dec, p.rad))")


def fetch_neowise_upload(stars: pd.DataFrame, radius_arcsec: float = 2.5,
                         retries: int = 2, sync: bool = False) -> QueryResult:
    """Batched positional join through a table upload (probed, never assumed)."""
    from astropy.table import Table

    pos = positions_at_neowise_epoch(stars)
    tbl = Table.from_pandas(pd.DataFrame({
        "sid": pos["source_id"].astype(str),
        "ra": pos["ra_mid"].astype(float), "dec": pos["dec_mid"].astype(float),
        "rad": (radius_arcsec + 0.5 * pos["sweep_arcsec"]).astype(float) / 3600.0}))
    q = upload_query(radius_arcsec)
    t0 = _time.monotonic()
    last = ""
    for attempt in range(retries):
        try:
            import pyvo
            svc = pyvo.dal.TAPService(IRSA_TAP)
            if sync:
                res = svc.run_sync(q, uploads={"pos": tbl})
            else:
                res = svc.run_async(q, uploads={"pos": tbl})
            df = res.to_table().to_pandas()
            df.columns = [str(c).lower() for c in df.columns]
            n = int(len(df))
            return QueryResult(label=f"neowise_upload_{len(tbl)}", service=IRSA_TAP,
                               status="OK" if n else "QUERY_RETURNED_ZERO_ROWS", n_rows=n,
                               query=q, elapsed_s=_time.monotonic() - t0, data=df)
        except Exception as exc:                       # noqa: BLE001
            last = repr(exc)
            print(f"[ignition] upload attempt {attempt + 1}/{retries} failed: {last}")
            _time.sleep(3.0 * (attempt + 1))
    return QueryResult(label=f"neowise_upload_{len(tbl)}", service=IRSA_TAP,
                       status="QUERY_FAILED", query=q, error=last,
                       elapsed_s=_time.monotonic() - t0)


def fetch_neowise_field(ra: float, dec: float, radius_deg: float, w1_max: float = 13.0,
                        max_rows: int = 4_000_000) -> QueryResult:
    """VIGIL's proven field-wide cone (one query per field, grouped locally)."""
    from ..vigil.acquire import fetch_neowise_field as _field

    return _field(ra, dec, radius_deg=radius_deg, w1_max=w1_max, max_rows=max_rows)


# --------------------------------------------------------------------------
# Checkpointed epoch store
# --------------------------------------------------------------------------
@dataclass
class EpochStore:
    """Append-only per-shard store: epochs CSV + star-quality CSV + progress JSON.

    Every completed batch lands on disk before the next query is issued, so a
    killed shard loses one batch.  ``done`` is the set of source_ids already
    written; the acquire loop skips them on resume.
    """

    epochs_path: Path
    stars_path: Path
    progress_path: Path
    done: set = field(default_factory=set)
    ledger: list = field(default_factory=list)
    n_stars_written: int = 0

    @classmethod
    def open(cls, out: Path, tag: str) -> EpochStore:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        st = cls(out / f"epochs_{tag}.csv", out / f"neowise_stars_{tag}.csv",
                 out / f"acquire_{tag}.json")
        if st.progress_path.exists():
            try:
                p = json.loads(st.progress_path.read_text())
                st.done = set(str(x) for x in p.get("done", []))
                st.ledger = list(p.get("ledger", []))
                st.n_stars_written = int(p.get("n_stars_written", len(st.done)))
                print(f"[ignition] resuming {tag}: {len(st.done)} stars already done")
            except Exception as exc:                   # noqa: BLE001
                print(f"[ignition] progress file unreadable ({exc!r}); starting fresh")
        return st

    def write_batch(self, epochs: list[pd.DataFrame], stars: list[dict],
                    done_ids: list[str]) -> None:
        if epochs:
            ep = pd.concat(epochs, ignore_index=True)
            ep.to_csv(self.epochs_path, mode="a", index=False,
                      header=not self.epochs_path.exists())
        if stars:
            sd = pd.DataFrame(stars)
            sd.to_csv(self.stars_path, mode="a", index=False,
                      header=not self.stars_path.exists())
        self.done.update(str(x) for x in done_ids)
        self.n_stars_written += len(stars)
        self.flush()

    def flush(self, extra: dict | None = None) -> None:
        rec = {"done": sorted(self.done), "n_done": len(self.done),
               "n_stars_written": self.n_stars_written, "ledger": self.ledger[-500:]}
        if extra:
            rec.update(extra)
        self.progress_path.write_text(json.dumps(rec, indent=1, default=str))


def reduce_star(sid: str, raw: pd.DataFrame, conf: dict | None = None
                ) -> tuple[pd.DataFrame, dict]:
    """Raw frames of one star -> (epoch rows, star record)."""
    clean, led = clean_frames(raw, conf)
    ep = star_epochs(clean, conf)
    ep.insert(0, "source_id", str(sid))
    q = star_quality(raw)
    rec = {"source_id": str(sid), **q, "n_exp_clean": int(len(clean)),
           "n_epochs_w1": int((ep["band"] == "W1").sum()) if len(ep) else 0,
           "n_epochs_w2": int((ep["band"] == "W2").sum()) if len(ep) else 0,
           "w1_median": float(clean["w1mpro"].median()) if len(clean) else float("nan"),
           "w2_median": float(clean["w2mpro"].median()) if len(clean) else float("nan"),
           "cut_cc_flags": led.get("cut_cc_flags", 0), "cut_ph_qual": led.get("cut_ph_qual", 0),
           "cut_qual_frame": led.get("cut_qual_frame", 0),
           "cut_moon_masked": led.get("cut_moon_masked", 0)}
    return ep, rec


def acquire_stars(stars: pd.DataFrame, store: EpochStore, conf: dict | None = None, *,
                  route: str = "cone", fields: list[dict] | None = None,
                  cone_fn=None, upload_fn=None, field_fn=None,
                  time_budget_s: float | None = None) -> dict:
    """Fetch, clean, bin and checkpoint every star not already in ``store``.

    ``route``: ``upload`` (batched TAP_UPLOAD join), ``field`` (one cone per
    entry of ``fields``, grouped locally), ``cone`` (one cone per star).  The
    fetchers are injectable so the loop runs offline.
    """
    c = {**DEFAULT_ACQUIRE, **(conf or {})}
    budget = float(c["time_budget_s"] if time_budget_s is None else time_budget_s)
    t_start = _time.monotonic()
    stars = stars.copy()
    stars["source_id"] = stars["source_id"].astype(str)
    todo = stars[~stars["source_id"].isin(store.done)].reset_index(drop=True)
    n_ok = n_zero = n_fail = 0
    tol = float(c["match_tol_arcsec"])

    def _finish(sid, raw, eps, recs, ids):
        nonlocal n_ok, n_zero
        if raw is None or not len(raw):
            n_zero += 1
            recs.append({"source_id": str(sid), "n_exp_raw": 0, "n_exp_clean": 0,
                         "n_epochs_w1": 0, "n_epochs_w2": 0, "neowise_status": "ZERO_ROWS"})
            ids.append(sid)
            return
        ep, rec = reduce_star(sid, raw, c)
        rec["neowise_status"] = "OK"
        n_ok += 1
        if len(ep):
            eps.append(ep)
        recs.append(rec)
        ids.append(sid)

    if route == "field":
        if field_fn is None:
            field_fn = fetch_neowise_field
        for f in (fields or []):
            if _time.monotonic() - t_start > budget:
                store.ledger.append({"label": "time_budget", "status": "STOPPED"})
                break
            ra, dec, rad = float(f["ra"]), float(f["dec"]), float(f["radius_deg"])
            members = todo[_in_cone(todo, ra, dec, rad)]
            if not len(members):
                continue
            try:
                r = field_fn(ra, dec, rad, w1_max=float(c["field_w1_max"]),
                             max_rows=int(c["field_max_rows"]))
            except Exception as exc:                   # noqa: BLE001
                r = QueryResult(label="neowise_field", service=IRSA_TAP,
                                status="QUERY_FAILED", error=repr(exc))
            store.ledger.append(r.to_ledger() | {"field": f})
            if r.status == "QUERY_FAILED" or r.data is None:
                n_fail += len(members)
                continue
            grouped = group_by_star(r.data, members, tol_arcsec=tol) if len(r.data) else {}
            eps, recs, ids = [], [], []
            for _, s in members.iterrows():
                sid = str(s["source_id"])
                _finish(sid, grouped.get(sid), eps, recs, ids)
            store.write_batch(eps, recs, ids)
            todo = todo[~todo["source_id"].isin(store.done)]
    elif route == "upload":
        if upload_fn is None:
            upload_fn = fetch_neowise_upload
        chunk = int(c["upload_chunk"])
        for i in range(0, len(todo), chunk):
            if _time.monotonic() - t_start > budget:
                store.ledger.append({"label": "time_budget", "status": "STOPPED"})
                break
            sub = todo.iloc[i:i + chunk]
            try:
                r = upload_fn(sub, radius_arcsec=float(c["cone_radius_arcsec"]))
            except Exception as exc:                   # noqa: BLE001
                r = QueryResult(label="neowise_upload", service=IRSA_TAP,
                                status="QUERY_FAILED", error=repr(exc))
            store.ledger.append(r.to_ledger())
            if r.status == "QUERY_FAILED" or r.data is None:
                n_fail += len(sub)
                continue
            d = r.data
            eps, recs, ids = [], [], []
            groups = {str(k): g for k, g in d.groupby("sid")} if len(d) else {}
            for sid in sub["source_id"]:
                _finish(sid, groups.get(str(sid)), eps, recs, ids)
            store.write_batch(eps, recs, ids)
    else:
        if cone_fn is None:
            cone_fn = fetch_neowise_cone
        every = int(c["checkpoint_every"])
        eps, recs, ids = [], [], []
        for _, s in todo.iterrows():
            if _time.monotonic() - t_start > budget:
                store.ledger.append({"label": "time_budget", "status": "STOPPED"})
                break
            sid = str(s["source_id"])
            try:
                r = cone_fn(float(s["ra"]), float(s["dec"]),
                            float(s.get("pmra", 0.0) or 0.0), float(s.get("pmdec", 0.0) or 0.0),
                            radius_arcsec=float(c["cone_radius_arcsec"]))
            except Exception as exc:                   # noqa: BLE001
                r = QueryResult(label="neowise_cone", service=IRSA_TAP,
                                status="QUERY_FAILED", error=repr(exc))
            if r.status == "QUERY_FAILED":
                n_fail += 1
                store.ledger.append(r.to_ledger() | {"source_id": sid})
                continue
            _finish(sid, r.data, eps, recs, ids)
            if len(ids) >= every:
                store.write_batch(eps, recs, ids)
                eps, recs, ids = [], [], []
        if ids:
            store.write_batch(eps, recs, ids)

    roll = {"route": route, "n_attempted": int(len(todo)), "n_ok": n_ok, "n_zero_rows": n_zero,
            "n_failed": n_fail, "n_done_total": len(store.done),
            "elapsed_s": round(_time.monotonic() - t_start, 1)}
    store.flush({"rollup": roll})
    return roll


def _in_cone(df: pd.DataFrame, ra: float, dec: float, radius_deg: float) -> np.ndarray:
    ra1 = np.radians(df["ra"].to_numpy(float))
    dec1 = np.radians(df["dec"].to_numpy(float))
    ra0, dec0 = np.radians(ra), np.radians(dec)
    cos_sep = (np.sin(dec1) * np.sin(dec0)
               + np.cos(dec1) * np.cos(dec0) * np.cos(ra1 - ra0))
    return np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0))) <= radius_deg


__all__ = ["DEFAULT_ACQUIRE", "NEOWISE_COLS", "NEOWISE_TABLE", "EpochStore", "acquire_stars",
           "clean_frames", "epochs_to_series", "exposures_to_epochs", "fetch_neowise_cone",
           "fetch_neowise_field", "fetch_neowise_upload", "group_by_star", "mjd_to_year",
           "positions_at_neowise_epoch", "reduce_star", "star_epochs", "star_quality",
           "upload_query"]
