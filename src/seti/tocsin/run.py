"""Stage orchestration for TOCSIN.  Writes ``results/tocsin/``.

Stages
------
``probe``
    Runner-only, cheap, and **run first**.  Establishes that the brokers are
    reachable from a GitHub Actions runner and records the *live* TAP schema
    verbatim.  The exact column set of a broker in July 2026 cannot be verified
    from inside the sandbox, so the channel discovers it on the runner and
    commits it: a later schema change then shows up as a diff in version
    control instead of as an unexplained null months later.
``targets``
    Runner-only.  Builds the Gaia DR3 nearby-star list in parallax shells.
``screen``
    The nightly stage.  Pulls the night's difference-image detections on nearby
    stars, screens them, folds the survivors into the persistent ledger, and
    writes the night's funnel counts.
``assess``
    Offline.  Recomputes rates, trial-corrected p-values and tiers over the
    whole accumulated ledger.  Separated from ``screen`` so the statistics can
    be re-derived (with a changed threshold, say) without touching the network.

Degradation is explicit everywhere: an unreachable broker writes
``verdict: NO_DATA_REACHED`` and commits that, rather than an empty candidate
table that reads like a clean null.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .brokers import (
    ALERCE_BAND,
    ALERCE_SID_DIAOBJECT,
    ALERCE_TAP,
    AlerceTAP,
    BrokerError,
    normalize_alerce_rows,
)
from .ledger import Ledger, _finite, night_of
from .screen import Thresholds, screen_alerts
from .targets import (
    GAIA_EPOCH,
    build_target_adql,
    match_alerts_to_targets,
    parallax_shells,
    position_uncertainty_arcsec,
    propagate_pm,
)

DEFAULTS: dict = {
    "target": {"d_max_pc": 100.0, "parallax_over_error_min": 10.0,
               "n_parallax_shells": 6, "dec_min": -90.0, "dec_max": 15.0,
               "g_max": 21.0, "gaia_epoch": GAIA_EPOCH},
    "acquire": {"primary_broker": "alerce_tap", "alerce_tap_url": ALERCE_TAP,
                "lookback_nights": 2.0, "parallax_min_mas": 10.0,
                "xmatch_max_arcsec": 1.5, "maxrec": 2000000, "timeout_s": 900,
                "gaia_catid": 1, "sid_diaobject": 1,
                "max_nights_per_run": 14.0, "backfill_start_mjd": 60973.0,
                "use_forced_photometry": False,
                "use_footprint_denominator": True, "footprint_bin_deg": 1.0,
                "max_run_seconds": 5400.0,
                "audit_without_gaia_join_every_n_runs": 14},
    "screen": {"min_abs_snr": 6.0, "min_reliability": 0.0,
               "require_reliability": False, "max_dipole_significance": 3.0,
               "max_extendedness": 0.5, "max_trail_arcsec": 5.0,
               "match_radius_arcsec": 1.5, "max_sep_arcsec": 1.0,
               "max_sep_sigma": 3.0, "max_grey_z": 3.0,
               "baseline_rel_err": 0.03, "missing_pm_penalty_arcsec": 2.0},
    "ledger": {"path": "results/tocsin/ledger.json", "alpha_fdr": 0.05,
               "min_visits_for_rate": 5, "max_duty_cycle": 0.2,
               "n_null_timing": 2000, "timing_alpha": 0.01,
               "mixed_polarity_requires_grey_both": True,
               "population_n_null": 2000, "population_seed": 20260730},
    "report": {"results_dir": "results/tocsin", "max_candidate_rows": 2000},
}

MJD_UNIX_EPOCH = 40587.0


def _deep_update(base: dict, extra: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def load_tocsin_config(cfg=None) -> dict:
    """Read ``config/tocsin.yaml`` over :data:`DEFAULTS`; a missing file degrades."""
    try:
        import yaml
        root = Path(cfg.root) if cfg is not None else _repo_root()
        p = root / "config" / "tocsin.yaml"
        if not p.exists():
            return _deep_update(DEFAULTS, {})
        return _deep_update(DEFAULTS, yaml.safe_load(p.read_text()) or {})
    except Exception as exc:                              # noqa: BLE001
        print(f"[tocsin] config/tocsin.yaml not loaded ({exc!r}); using defaults")
        return _deep_update(DEFAULTS, {})


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config").is_dir():
            return parent
    return here.parents[3]


def _sid(aconf: dict) -> int | None:
    """The configured ALeRCE ``sid`` filter, or ``None`` to drop the clause.

    Kept as a helper so the value the probe measures is applied in exactly one
    place, from config, without editing any query code.
    """
    v = aconf.get("sid_diaobject")
    return None if v is None else int(v)


def _frontier_mjd(aconf: dict) -> float | None:
    """Newest LSST epoch the broker holds, or None if it cannot be determined."""
    try:
        tap = AlerceTAP(aconf["alerce_tap_url"], timeout=float(aconf["timeout_s"]))
        return tap.max_available_mjd()
    except Exception as exc:                              # noqa: BLE001
        print(f"[tocsin] frontier query failed ({exc!r}); falling back to now")
        return None


def _now_mjd() -> float:
    return MJD_UNIX_EPOCH + datetime.now(timezone.utc).timestamp() / 86400.0


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=1, sort_keys=True,
                               allow_nan=False) + "\n")


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def probe(cfg=None, out_dir: str | Path | None = None) -> dict:
    """Reachability + live-schema record for the brokers.  Runner-only.

    Writes ``probe.json``.  Nothing else in the channel should be dispatched
    against real data until this has succeeded once, because every ADQL column
    name in ``brokers.py`` is inferred from the brokers' published source rather
    than from a live query --- which is the best that can be done from a sandbox
    with no egress, and is not the same as verified.
    """
    conf = load_tocsin_config(cfg)
    root = Path(cfg.root) if cfg is not None else _repo_root()
    out = Path(out_dir) if out_dir else root / conf["report"]["results_dir"]
    rec: dict = {"probed_at_utc": _utc(), "now_mjd": round(_now_mjd(), 5),
                 "alerce": {"url": conf["acquire"]["alerce_tap_url"],
                            "reached": False}}
    tap = AlerceTAP(conf["acquire"]["alerce_tap_url"],
                    timeout=float(conf["acquire"]["timeout_s"]))

    def checkpoint() -> None:
        """Write the record after every stage.

        The probe queries a live service with a job timeout above it.  Writing
        only at the end means a single slow query loses the whole pass ---
        including the answers that had already been obtained.  Checkpointing
        costs nothing and makes a timeout partially informative instead of
        totally wasted.
        """
        rec["verdict"] = "OK" if rec["alerce"].get("reached") else "NO_DATA_REACHED"
        _write_json(out / "probe.json", rec)

    try:
        rec["alerce"]["schema"] = tap.describe()
        rec["alerce"]["reached"] = True
    except Exception as exc:                              # noqa: BLE001
        rec["alerce"]["error"] = f"{type(exc).__name__}: {exc}"[:800]
    checkpoint()

    # The diagnostic battery.  A schema dump says a column exists; it does not
    # say whether any row populates it, how current the data are, or what the
    # discriminating keys actually mean.  The first probe returned an empty
    # night window and the schema alone could not say why, so this runs a set of
    # small independent queries and captures each error separately.
    if rec["alerce"]["reached"]:
        try:
            diag: dict = {}
            rec["alerce"]["diagnostics"] = diag

            def _record(name, result):
                diag[name] = result
                checkpoint()

            tap.diagnostics(_now_mjd(),
                            gaia_catid=int(conf["acquire"].get("gaia_catid", 1)),
                            on_result=_record)
        except Exception as exc:                          # noqa: BLE001
            rec["alerce"]["diagnostics"] = {
                "error": f"{type(exc).__name__}: {exc}"[:800]}
        checkpoint()

    # A tiny live slice of the real query, in both variants.
    if rec["alerce"]["reached"]:
        hi = _now_mjd()
        lo = hi - 3.0
        # Variants ordered so that the first one to return rows identifies the
        # clause responsible for an empty result: Gaia join, then the sid filter,
        # then the SNR cut, then the window width.
        variants = (
            ("with_gaia_join", conf["acquire"]["parallax_min_mas"],
             ALERCE_SID_DIAOBJECT, float(conf["screen"]["min_abs_snr"]), 3.0),
            ("without_gaia_join", None,
             ALERCE_SID_DIAOBJECT, float(conf["screen"]["min_abs_snr"]), 3.0),
            ("without_sid_filter", None, None,
             float(conf["screen"]["min_abs_snr"]), 3.0),
            ("without_snr_cut", None, None, 0.0, 3.0),
            ("wide_window_30d", None, None, 0.0, 30.0),
        )
        for label, plx, sid, snr, span in variants:
            try:
                r = tap.night_detections(
                    hi - span, hi, parallax_min_mas=plx,
                    xmatch_max_arcsec=float(conf["acquire"]["xmatch_max_arcsec"]),
                    min_abs_snr=snr, sid_diaobject=sid, maxrec=20)
                rec["alerce"][label] = {
                    "verdict": r.verdict, "rows": len(r.rows),
                    "columns": sorted(r.rows[0].keys()) if r.rows else [],
                    "sample": r.rows[:2],
                    "adql": r.notes[0] if r.notes else "",
                }
            except Exception as exc:                      # noqa: BLE001
                rec["alerce"][label] = {"error": f"{type(exc).__name__}: {exc}"[:800]}
            checkpoint()
        try:
            fp = tap.forced_photometry_night(
                lo, hi, parallax_min_mas=None, sid_diaobject=None, maxrec=20)
            rec["alerce"]["forced_photometry"] = {
                "verdict": fp.verdict, "rows": len(fp.rows),
                "columns": sorted(fp.rows[0].keys()) if fp.rows else []}
        except Exception as exc:                          # noqa: BLE001
            rec["alerce"]["forced_photometry"] = {
                "error": f"{type(exc).__name__}: {exc}"[:800]}

    checkpoint()
    print(f"[tocsin] probe verdict={rec['verdict']} -> {out/'probe.json'}")
    return rec


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
def build_targets(cfg=None, out_path: str | Path | None = None,
                  max_rows_per_shell: int = 500000) -> dict:
    """Fetch the Gaia DR3 nearby-star target list, in parallax shells.  Runner-only.

    Chunking is not optional: a single monolithic Gaia query at this row count
    times out on the runner (``docs/channel-brief.md`` §2).  Shells are equal in
    volume so the row count per query stays roughly flat.
    """
    conf = load_tocsin_config(cfg)
    root = Path(cfg.root) if cfg is not None else _repo_root()
    tconf = conf["target"]
    out = Path(out_path) if out_path else root / ".cache" / "tocsin" / "targets.parquet"
    rec = {"built_at_utc": _utc(), "shells": [], "n_targets": 0,
           "verdict": "NOT_RUN", "path": str(out)}
    try:
        from astroquery.gaia import Gaia
    except ImportError as exc:
        rec["verdict"] = "NO_ASTROQUERY"
        rec["error"] = str(exc)
        return rec

    frames = []
    for lo, hi in parallax_shells(float(tconf["d_max_pc"]),
                                  int(tconf["n_parallax_shells"])):
        adql = build_target_adql(
            lo, hi, dec_max=float(tconf["dec_max"]), dec_min=float(tconf["dec_min"]),
            g_max=float(tconf["g_max"]), max_rows=int(max_rows_per_shell))
        shell = {"parallax_mas": [lo, hi], "rows": 0}
        for attempt in range(3):
            try:
                job = Gaia.launch_job_async(adql)
                df = job.get_results().to_pandas()
                frames.append(df)
                shell["rows"] = len(df)
                break
            except Exception as exc:                      # noqa: BLE001
                shell["error"] = f"{type(exc).__name__}: {exc}"[:300]
                if attempt == 2:
                    # The GSPC join is the likely failure (not every source has
                    # BP/RP synthetic photometry); retry without it so the run
                    # degrades to "no baseline flux" rather than to no targets.
                    try:
                        df = Gaia.launch_job_async(
                            build_target_adql(lo, hi, dec_max=float(tconf["dec_max"]),
                                              dec_min=float(tconf["dec_min"]),
                                              g_max=float(tconf["g_max"]),
                                              max_rows=int(max_rows_per_shell),
                                              require_synthetic=False)
                        ).get_results().to_pandas()
                        frames.append(df)
                        shell["rows"] = len(df)
                        shell["note"] = "fell_back_to_no_synthetic_photometry"
                    except Exception as exc2:             # noqa: BLE001
                        shell["error"] = f"{type(exc2).__name__}: {exc2}"[:300]
        rec["shells"].append(shell)

    if not frames:
        rec["verdict"] = "NO_DATA_REACHED"
        return rec
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    df = df.drop_duplicates(subset=["source_id"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    rec["n_targets"] = int(len(df))
    rec["verdict"] = "OK"
    print(f"[tocsin] targets: {len(df)} nearby stars -> {out}")
    return rec


def load_targets(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.columns = [c.lower() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# screen (the nightly stage)
# ---------------------------------------------------------------------------
def _thresholds(conf: dict) -> Thresholds:
    s = conf["screen"]
    return Thresholds(
        min_abs_snr=float(s["min_abs_snr"]),
        min_reliability=float(s["min_reliability"]),
        require_reliability=bool(s["require_reliability"]),
        max_dipole_significance=float(s["max_dipole_significance"]),
        max_extendedness=float(s["max_extendedness"]),
        max_trail_arcsec=float(s["max_trail_arcsec"]),
        max_sep_sigma=float(s["max_sep_sigma"]),
        max_sep_arcsec=float(s["max_sep_arcsec"]),
        match_radius_arcsec=float(s["match_radius_arcsec"]),
        max_grey_z=float(s["max_grey_z"]),
        baseline_rel_err=float(s["baseline_rel_err"]),
        missing_pm_penalty_arcsec=float(s["missing_pm_penalty_arcsec"]),
    )


def _trials_from_footprint(rows, targets, bin_deg: float, epoch_jyear: float
                           ) -> tuple[set, dict, dict]:
    """Star-nights on which a target's own sky bin was observed.

    This is the denominator that does not depend on the broker populating forced
    photometry (it covered 0% of star-nights in the first backfill).  Target
    positions are proper-motion propagated to the window's epoch before binning,
    for the same reason everything else here is: nearby stars are high-proper-
    motion stars, and binning a 2016 position into a 2026 footprint would
    mis-assign the ones that matter most.
    """
    stats = {"footprint_bins": len(rows or [])}
    if not rows or targets is None or len(targets) == 0:
        return set(), stats, {}
    observed = set()
    nights = set()
    bands_by_bin: dict = {}
    for r in rows:
        try:
            key = (int(r["rab"]), int(r["decb"]), int(r["night"]))
            observed.add(key)
            nights.add(int(r["night"]))
        except (KeyError, TypeError, ValueError):
            continue
        b = r.get("band")
        if b is not None:
            try:
                bands_by_bin.setdefault(key, set()).add(ALERCE_BAND.get(int(b), ""))
            except (TypeError, ValueError):
                pass
    if not observed:
        return set(), stats, {}
    t_ra = np.asarray(targets["ra"], dtype=float)
    t_dec = np.asarray(targets["dec"], dtype=float)
    pmra = np.asarray(targets["pmra"], dtype=float) if "pmra" in targets else np.zeros(t_ra.size)
    pmdec = (np.asarray(targets["pmdec"], dtype=float) if "pmdec" in targets
             else np.zeros(t_ra.size))
    p_ra, p_dec = propagate_pm(t_ra, t_dec, pmra, pmdec, to_epoch=epoch_jyear)
    rab = np.floor(p_ra / bin_deg).astype(int)
    decb = np.floor(p_dec / bin_deg).astype(int)
    ids = (np.asarray(targets["source_id"]).astype(str) if "source_id" in targets
           else np.arange(t_ra.size).astype(str))
    pairs = set()
    bands_by_pair: dict = {}
    for night in sorted(nights):
        # Vectorised membership: build this night's observed-bin set once.
        bins_tonight = {(a, b) for a, b, n in observed if n == night}
        if not bins_tonight:
            continue
        hit = np.fromiter(((int(a), int(b)) in bins_tonight
                           for a, b in zip(rab, decb, strict=True)),
                          dtype=bool, count=rab.size)
        for idx in np.nonzero(hit)[0]:
            tid = str(ids[idx])
            pair = (tid, f"n{night}")
            pairs.add(pair)
            seen = bands_by_bin.get((int(rab[idx]), int(decb[idx]), night))
            if seen:
                bands_by_pair.setdefault(pair, set()).update(b for b in seen if b)
    stats["footprint_nights"] = len(nights)
    stats["footprint_star_nights"] = len(pairs)
    return pairs, stats, bands_by_pair


def _visit_history_from_forced(rows, targets, th: Thresholds, epoch_jyear: float
                               ) -> tuple[dict[str, list[float]], set, dict]:
    """Map forced-photometry rows onto targets and count the night's trials.

    The association uses the *same* proper-motion-propagated matcher as the
    events do.  Using the broker's own cross-match for the denominator and ours
    for the numerator would silently mix two different definitions of "this
    star", and the ratio of the two would not be a rate.
    """
    stats = {"forced_rows": len(rows), "forced_matched": 0}
    if not rows or targets is None or len(targets) == 0:
        return {}, set(), stats
    fra = np.array([float(r.get("ra", np.nan)) for r in rows])
    fdec = np.array([float(r.get("dec", np.nan)) for r in rows])
    fmjd = np.array([float(r.get("mjd", np.nan)) for r in rows])
    ok = np.isfinite(fra) & np.isfinite(fdec) & np.isfinite(fmjd)
    if not np.any(ok):
        return {}, set(), stats
    t_ra = np.asarray(targets["ra"], dtype=float)
    t_dec = np.asarray(targets["dec"], dtype=float)
    pmra = np.asarray(targets["pmra"], dtype=float) if "pmra" in targets else np.zeros(t_ra.size)
    pmdec = (np.asarray(targets["pmdec"], dtype=float) if "pmdec" in targets
             else np.zeros(t_ra.size))
    p_ra, p_dec = propagate_pm(t_ra, t_dec, pmra, pmdec, to_epoch=epoch_jyear)
    t_sig = position_uncertainty_arcsec(
        targets["pmra_error"] if "pmra_error" in targets else np.zeros(t_ra.size),
        targets["pmdec_error"] if "pmdec_error" in targets else np.zeros(t_ra.size),
        dt_yr=epoch_jyear - GAIA_EPOCH,
        pm_missing=~(np.isfinite(pmra) & np.isfinite(pmdec)),
        missing_pm_penalty_arcsec=th.missing_pm_penalty_arcsec)
    m = match_alerts_to_targets(fra[ok], fdec[ok], p_ra, p_dec,
                                radius_arcsec=th.match_radius_arcsec,
                                target_pos_err_arcsec=t_sig)
    ids = (np.asarray(targets["source_id"]).astype(str) if "source_id" in targets
           else np.arange(t_ra.size).astype(str))
    mjds = fmjd[ok]
    hist: dict[str, list[float]] = {}
    star_nights: set[tuple[str, int]] = set()
    for ai, ti in zip(m.alert_index, m.target_index, strict=True):
        tid = str(ids[int(ti)])
        mj = float(mjds[int(ai)])
        hist.setdefault(tid, []).append(mj)
        star_nights.add((tid, night_of(mj)))
    stats["forced_matched"] = int(m.alert_index.size)
    stats["tracked_targets"] = len(hist)
    # Return the (target, night) PAIRS, not a per-night count: the caller has to
    # union these with the star-nights that produced detections, and that union
    # cannot be taken on counts alone.
    pairs = {(tid, f"n{n}") for tid, n in star_nights}
    return ({k: sorted(set(v)) for k, v in hist.items()}, pairs, stats)


def screen(cfg=None, chunks: int = 1, max_run_seconds: float | None = None,
           **kw) -> dict:
    """Run :func:`screen_night` up to ``chunks`` times, stopping when caught up.

    One dispatch can therefore walk several chunks of the backlog instead of
    needing one dispatch per chunk.  Stops early on ``NO_NEW_DATA`` (the
    watermark has reached the broker's frontier) or on any non-OK verdict, so a
    broker failure does not burn the whole budget retrying.

    It also stops on a **wall-clock budget**, well short of the CI job timeout.
    A 16-chunk backfill once ran into the 180-minute job limit and was cancelled
    outright, which discards the whole run: the commit step never executes, so
    every chunk that HAD succeeded is thrown away too.  Yielding voluntarily
    means the watermark and ledger are always committed and the next dispatch
    resumes exactly where this one stopped.
    """
    conf = load_tocsin_config(cfg)
    budget = float(max_run_seconds if max_run_seconds is not None
                   else conf["acquire"].get("max_run_seconds", 5400.0))
    t0 = time.monotonic()
    last: dict = {}
    for i in range(max(1, int(chunks))):
        last = screen_night(cfg, **kw)
        v = last.get("verdict")
        elapsed = time.monotonic() - t0
        print(f"[tocsin] chunk {i + 1}/{chunks}: {v}  ({elapsed:.0f}s elapsed)")
        if v != "OK" and v != "NO_DETECTIONS_IN_WINDOW":
            break
        if kw.get("mjd_lo") is not None or kw.get("mjd_hi") is not None:
            break                       # an explicit window is a single shot
        if elapsed > budget:
            print(f"[tocsin] wall-clock budget {budget:.0f}s reached after "
                  f"{i + 1} chunks; stopping so the ledger is committed")
            last.setdefault("notes", []).append(
                f"stopped after {i + 1} of {chunks} chunks on the "
                f"{budget:.0f}s wall-clock budget; re-dispatch to continue")
            break
    return last


def screen_night(cfg=None, lookback_nights: float | None = None,
                 mjd_lo: float | None = None, mjd_hi: float | None = None,
                 targets_path: str | Path | None = None,
                 out_dir: str | Path | None = None,
                 use_gaia_join: bool = True) -> dict:
    """Pull, screen and ledger one window of the Rubin alert stream.  Runner-only."""
    conf = load_tocsin_config(cfg)
    root = Path(cfg.root) if cfg is not None else _repo_root()
    out = Path(out_dir) if out_dir else root / conf["report"]["results_dir"]
    th = _thresholds(conf)
    aconf = conf["acquire"]

    lconf_path = root / conf["ledger"]["path"]
    led_peek = Ledger.load(lconf_path)
    explicit_window = mjd_lo is not None or mjd_hi is not None
    max_nights = float(aconf.get("max_nights_per_run", 7.0))
    look = float(lookback_nights if lookback_nights is not None
                 else aconf["lookback_nights"])

    if explicit_window:
        hi = float(mjd_hi) if mjd_hi is not None else _now_mjd()
        lo = float(mjd_lo) if mjd_lo is not None else hi - look
        frontier = None
    else:
        # Anchor to what the broker ACTUALLY holds, not to the wall clock.
        # ALeRCE's TAP mirror lagged 15.6 days when measured on 2026-07-30, so a
        # window of "the last two nights" would return nothing every night
        # forever --- and an empty result is indistinguishable from a real null.
        frontier = _frontier_mjd(aconf)
        hi_cap = frontier if frontier is not None else _now_mjd()
        # `_finite`, not `float(...)`: the watermark is written as JSON null
        # when unset (NaN is not valid JSON), so it comes back as None and
        # `float(None)` would raise on the second run of any live ledger.
        wm = _finite(led_peek.last_mjd_screened)
        if wm is not None:
            lo = wm
        else:
            # First run ever.  `backfill_start_mjd` starts the walk at the
            # beginning of the broker's LSST holdings rather than a few nights
            # back: ~262 nights of real data already exist, and the recurrence
            # statistic this channel is built on needs many nights before it can
            # say anything at all.
            start = aconf.get("backfill_start_mjd")
            lo = float(start) if start is not None else hi_cap - look
        # Cap the chunk so the first run against a 262-night backlog does not
        # try to pull all of it in one job.
        hi = min(hi_cap, lo + max_nights)
    summary: dict = {
        "run_at_utc": _utc(), "mjd_lo": round(lo, 5), "mjd_hi": round(hi, 5),
        "broker_frontier_mjd": (round(frontier, 5) if frontier is not None else None),
        "broker_lag_days": (round(_now_mjd() - frontier, 2)
                            if frontier is not None else None),
        "explicit_window": explicit_window,
        "nights": sorted({night_of(lo), night_of(hi)}),
        "use_gaia_join": bool(use_gaia_join), "verdict": "NOT_RUN", "counts": {},
        "notes": [],
    }

    if hi <= lo:
        summary["verdict"] = "NO_NEW_DATA"
        summary["notes"].append(
            "the watermark has caught up with the broker's newest epoch; "
            "nothing to screen until the mirror advances")
        _write_json(out / "summary.json", summary)
        print(f"[tocsin] NO_NEW_DATA (watermark {lo:.3f} >= frontier {hi:.3f})")
        return summary

    tpath = (Path(targets_path) if targets_path
             else root / ".cache" / "tocsin" / "targets.parquet")
    targets = load_targets(tpath)
    if targets is None or len(targets) == 0:
        summary["verdict"] = "NO_TARGET_LIST"
        summary["notes"].append(f"target list missing at {tpath}; run tocsin-targets")
        _write_json(out / "summary.json", summary)
        return summary
    summary["n_targets"] = int(len(targets))

    tap = AlerceTAP(aconf["alerce_tap_url"], timeout=float(aconf["timeout_s"]),
                    maxrec=int(aconf["maxrec"]))
    plx = float(aconf["parallax_min_mas"]) if use_gaia_join else None
    _t0 = time.monotonic()
    timings: dict[str, float] = {}
    summary["timings_s"] = timings
    try:
        det = tap.night_detections(
            lo, hi, parallax_min_mas=plx,
            xmatch_max_arcsec=float(aconf["xmatch_max_arcsec"]),
            gaia_catid=int(aconf.get("gaia_catid", 1)),
            sid_diaobject=_sid(aconf),
            min_abs_snr=float(conf["screen"]["min_abs_snr"]))
    except BrokerError as exc:
        summary["verdict"] = "NO_DATA_REACHED"
        summary["error"] = str(exc)[:800]
        _write_json(out / "summary.json", summary)
        print(f"[tocsin] {summary['verdict']}: {summary['error']}")
        return summary
    timings["night_detections"] = round(time.monotonic() - _t0, 1)
    summary["counts"]["detections_pulled"] = len(det.rows)
    summary["notes"].extend(n for n in det.notes if not n.startswith("adql="))

    alerts = normalize_alerce_rows(det.rows)
    summary["counts"]["normalised"] = len(alerts)

    # Per-band effective alert threshold, measured from this window's OWN
    # detections rather than assumed: the median flux error in a band times the
    # stream's ~5-sigma detection threshold.  Used only to decide whether a
    # silent band's silence is informative.
    band_limits: dict[str, float] = {}
    errs: dict[str, list[float]] = {}
    for a in alerts:
        if a.band and a.dflux_err_njy and np.isfinite(a.dflux_err_njy):
            errs.setdefault(a.band, []).append(float(a.dflux_err_njy))
    for b, vals in errs.items():
        band_limits[b] = 5.0 * float(np.median(vals))
    summary["band_detection_limits_njy"] = {
        b: round(v, 1) for b, v in sorted(band_limits.items())}
    summary["band_limit_n"] = {b: len(v) for b, v in sorted(errs.items())}

    epoch_jyear = 2000.0 + ((lo + hi) / 2.0 - 51544.5) / 365.25

    # The footprint denominator: which targets were LOOKED AT each night,
    # independent of whether the broker populated forced photometry for them.
    footprint_pairs: set = set()
    observed_bands: dict = {}
    if bool(aconf.get("use_footprint_denominator", True)):
        _t1 = time.monotonic()
        try:
            fb = tap.footprint_bins(lo, hi, bin_deg=float(aconf.get("footprint_bin_deg", 1.0)),
                                    sid_diaobject=_sid(aconf))
            footprint_pairs, fpstats, observed_bands = _trials_from_footprint(
                fb.rows, targets, float(aconf.get("footprint_bin_deg", 1.0)),
                epoch_jyear)
            summary["counts"].update(fpstats)
        except BrokerError as exc:
            summary["notes"].append(f"footprint_query_failed: {str(exc)[:300]}")
        timings["footprint"] = round(time.monotonic() - _t1, 1)

    verdict = screen_alerts(alerts, targets, th, epoch_jyear=epoch_jyear,
                            observed_bands=observed_bands, band_limits=band_limits)
    summary["counts"].update(verdict.counts)
    summary["notes"].extend(verdict.notes)

    # OBSERVABILITY.  The first live run produced 62 events in which every
    # single fractional amplitude was untestable, because a wrong Gaia column
    # name had silently disabled the baseline photometry --- and therefore the
    # greyness test, this channel's core discriminator.  Nothing in the summary
    # said so.  These two distributions make that class of failure impossible to
    # miss: if `baseline_sources` is dominated by anything other than
    # `rubin_template`, or `bands_per_event` is dominated by 1, the colour test
    # is not actually running and no result should be believed.
    base_src: dict[str, int] = {}
    bands_hist: dict[str, int] = {}
    grey_tested = 0
    for ev in verdict.events:
        for pb in ev.per_band.values():
            k = str(pb.get("baseline_source", "unknown"))
            base_src[k] = base_src.get(k, 0) + 1
        nb = str(len(ev.bands))
        bands_hist[nb] = bands_hist.get(nb, 0) + 1
        grey_tested += int(bool(ev.grey_tested))
    summary["baseline_sources"] = base_src
    summary["bands_per_event"] = bands_hist
    # The colour test's reach must count the events it REJECTED, not only the
    # survivors.  Counting survivors alone reported "the discriminant did not
    # run at all" in a window where it had just killed five flares — the metric
    # was measuring the wrong population and hiding its own success.
    n_chromatic = int(verdict.counts.get("rejected_chromatic", 0))
    n_considered = len(verdict.events) + n_chromatic
    summary["colour_tested"] = grey_tested + n_chromatic
    summary["colour_rejected_chromatic"] = n_chromatic
    summary["greyness_tested_fraction"] = (
        round((grey_tested + n_chromatic) / n_considered, 4) if n_considered else None)
    if n_considered and (grey_tested + n_chromatic) == 0:
        summary["notes"].append(
            "NO event could be colour-tested: the achromaticity discriminant "
            "did not run at all this window")

    # Forced photometry: OPT-IN, and off by default.  Measured on the live
    # service it took 3151 s and then timed out, while covering 0-0.08% of
    # screened star-nights — the footprint query answers the same question in
    # 5 s and covers all of them.  One stalled forced-photometry call had
    # already burned a 180-minute job to cancellation.  The code path is kept
    # because the epoch-level history it returns is strictly richer than the
    # night-level history the footprint gives; it is simply not worth its cost
    # against this broker today.
    _t2 = time.monotonic()
    hist, forced_pairs = {}, set()
    if not bool(aconf.get("use_forced_photometry", False)):
        summary["notes"].append(
            "forced photometry skipped (use_forced_photometry=false): measured "
            "at 3151 s and 0-0.08% coverage against a 5 s footprint query")
    else:
        try:
            fp = tap.forced_photometry_night(
                lo, hi, parallax_min_mas=plx,
                xmatch_max_arcsec=float(aconf["xmatch_max_arcsec"]),
                gaia_catid=int(aconf.get("gaia_catid", 1)),
                sid_diaobject=_sid(aconf))
            hist, forced_pairs, fstats = _visit_history_from_forced(
                fp.rows, targets, th, epoch_jyear)
            summary["counts"].update(fstats)
        except BrokerError as exc:
            hist, forced_pairs = {}, set()
            summary["notes"].append(f"forced_photometry_failed: {str(exc)[:300]}")
    timings["forced_photometry"] = round(time.monotonic() - _t2, 1)

    # THE DENOMINATOR.  A star-night that produced a detection was, by
    # definition, a star-night that was observed --- so it is a trial whether or
    # not forced photometry happens to cover it.  Taking the union is not a
    # patch: without it the numerator and the denominator are measured over
    # different populations and their ratio is not a rate at all.  The first
    # live run made that concrete: 62 events against 8 forced star-nights gave
    # an "event rate" of 7.75 per star-night.
    #
    # The union guarantees trials >= events.  What it does NOT do is manufacture
    # non-detection information: a trial known only from a detection contributes
    # nothing but itself, so a screen whose forced-photometry coverage is poor
    # simply reports a rate near 1 and promotes nobody --- which is the correct,
    # conservative behaviour, and is reported rather than hidden.
    event_pairs = {(ev.target_id, ev.night) for ev in verdict.events}
    all_pairs = set(forced_pairs) | event_pairs | set(footprint_pairs)
    trials_by_night: dict[str, int] = {}
    for _tid, night in all_pairs:
        trials_by_night[night] = trials_by_night.get(night, 0) + 1
    # Per-bin trial counts for the stratified null (see `Ledger.bin_trials`).
    from .ledger import bin_key as _bin_key
    tpos = verdict.target_positions
    # Keyed by NIGHT, then by bin.  Keying only by bin and handing the whole
    # window's dict to the first night folded meant the ledger's night-dedup
    # guard threw it away whenever that night had already been seen — so
    # bin_trials stayed empty, every local rate came back None, and the
    # stratified null silently fell back to the all-sky rate it was built to
    # replace.  Same failure shape as the visit-history ordering bug.
    bin_trials_tonight: dict[str, dict[str, int]] = {}
    if targets is not None and len(targets):
        _ra = np.asarray(targets["ra"], dtype=float)
        _dec = np.asarray(targets["dec"], dtype=float)
        _ids = (np.asarray(targets["source_id"]).astype(str)
                if "source_id" in targets else np.arange(_ra.size).astype(str))
        _pos = dict(zip(_ids, zip(_ra, _dec, strict=True), strict=True))
        for tid, _night in all_pairs:
            ra_dec = tpos.get(tid) or _pos.get(str(tid))
            if not ra_dec:
                continue
            k = _bin_key(ra_dec[0], ra_dec[1])
            if k:
                bin_trials_tonight.setdefault(_night, {})
                bin_trials_tonight[_night][k] = \
                    bin_trials_tonight[_night].get(k, 0) + 1

    n_forced = len(forced_pairs)
    n_footprint = len(footprint_pairs)
    n_total = len(all_pairs)
    summary["counts"]["target_nights_screened"] = n_total
    summary["counts"]["target_nights_with_forced_photometry"] = n_forced
    summary["counts"]["target_nights_from_footprint"] = n_footprint
    summary["counts"]["target_nights_detection_only"] = len(
        event_pairs - set(forced_pairs) - set(footprint_pairs))
    summary["trials_by_night"] = trials_by_night
    frac = (n_forced / n_total) if n_total else 0.0
    summary["forced_coverage_fraction"] = round(frac, 4)
    summary["footprint_coverage_fraction"] = (
        round(n_footprint / n_total, 4) if n_total else 0.0)
    if n_total == 0:
        summary["denominator"] = "unavailable"
    elif frac >= 0.9:
        summary["denominator"] = "forced_photometry_exact"
    elif n_footprint > 2 * len(event_pairs):
        # The footprint carries genuine non-detection information: far more
        # star-nights were observed than produced events, so the rate is a real
        # rate rather than a tautology.
        summary["denominator"] = "observed_footprint"
    else:
        # Most trials carry no non-detection information, so the rate is an
        # upper bound on the true rate and every p-value derived from it is
        # conservative.  Say so in the committed artefact.
        summary["denominator"] = "detection_dominated_lower_bound"
        summary["notes"].append(
            f"forced photometry covers only {frac:.1%} of screened star-nights; "
            "the ensemble rate is therefore an UPPER bound and promotion is "
            "correspondingly conservative")

    # THE VISIT HISTORY, from the footprint.  `Ledger._set_tier` requires
    # `visits_exact` before a target can reach candidate tier, and that flag is
    # set from this history — so without it nothing could EVER be promoted, and
    # switching forced photometry off would have traded a 52-minute stall for a
    # permanently inert screen.
    #
    # The footprint knows which NIGHTS each target's sky bin was observed, which
    # is exactly the unit the ledger counts in: events are star-nights, trials
    # are star-nights, and the timing null resamples nights.  One representative
    # epoch per observed night is therefore a complete history at the resolution
    # that matters.  (Forced photometry, when enabled, gives true per-visit
    # epochs and is merged on top.)
    for tid, night in footprint_pairs:
        try:
            n = int(str(night).lstrip("n"))
        except (TypeError, ValueError):
            continue
        # Any epoch inside the night works; night_of(n + 1.1667) == n.
        hist.setdefault(tid, []).append(round(n + 1.1666667, 6))
    for tid, mjds in verdict.visit_history.items():
        hist.setdefault(tid, []).extend(mjds)
    for tid in list(hist):
        hist[tid] = sorted(set(hist[tid]))

    lconf = conf["ledger"]
    ledger_path = root / lconf["path"]
    led = Ledger.load(ledger_path)
    if not led.opened_utc:
        led.opened_utc = _utc()
    # Fold NIGHT BY NIGHT.  The run window spans two nights and consecutive runs
    # overlap; only a per-night key keeps the cumulative trial count honest.
    events_by_night: dict[str, list] = {}
    for ev in verdict.events:
        events_by_night.setdefault(ev.night, []).append(ev)
    all_nights = sorted(set(trials_by_night) | set(events_by_night)
                        | {f"n{n}" for n in range(night_of(lo), night_of(hi) + 1)})
    first = True
    for night in all_nights:
        led.add_night(night, events_by_night.get(night, []),
                      target_visits=trials_by_night.get(night, 0),
                      targets_in_footprint=len(hist) or len(targets),
                      # Alerts pulled are a per-window quantity, not per-night;
                      # attribute them once so the running total stays a count
                      # of alerts actually seen.
                      alerts_seen=len(alerts) if first else 0,
                      visit_history=None,
                      target_positions=verdict.target_positions,
                      bin_trials=bin_trials_tonight.get(night))
        first = False
    # AFTER every night is folded, so targets whose record is created by a later
    # night still receive their visit history.
    led.apply_visit_history(hist)
    summary["nights_folded"] = all_nights
    led.updated_utc = _utc()
    stats = led.assess(alpha_fdr=float(lconf["alpha_fdr"]),
                       min_visits_for_rate=int(lconf["min_visits_for_rate"]),
                       max_duty_cycle=float(lconf["max_duty_cycle"]),
                       n_null_timing=int(lconf["n_null_timing"]),
                       timing_alpha=float(lconf["timing_alpha"]),
                       max_grey_z=float(conf["screen"]["max_grey_z"]),
                       mixed_polarity_requires_grey_both=bool(
                           lconf.get("mixed_polarity_requires_grey_both", True)))
    led.save(ledger_path)

    if not explicit_window:
        # Only an auto-derived window advances the watermark: a manual re-run of
        # an old window must not make the screen skip forward over data it has
        # not actually looked at.
        led.last_mjd_screened = float(hi)
        led.save(ledger_path)
    summary["ledger"] = stats
    _wm = _finite(led.last_mjd_screened)
    summary["watermark_mjd"] = None if _wm is None else round(_wm, 5)
    summary["verdict"] = "OK" if det.rows else "NO_DETECTIONS_IN_WINDOW"
    _write_events(out, verdict, conf)
    _write_watchlist(out, led, conf)
    timings["total"] = round(time.monotonic() - _t0, 1)
    _write_json(out / "summary.json", summary)
    print(f"[tocsin] timings(s): {timings}")
    print(f"[tocsin] {summary['verdict']}: {len(alerts)} alerts, "
          f"{len(verdict.events)} events, tiers={stats['tier_counts']}")
    return summary


def _write_events(out: Path, verdict, conf: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cap = int(conf["report"]["max_candidate_rows"])
    rows = [asdict(e) for e in verdict.events]
    truncated = len(rows) > cap
    df = pd.DataFrame(rows[:cap])
    if not df.empty:
        for col in ("bands", "alert_ids", "reasons", "per_band"):
            if col in df:
                df[col] = df[col].map(json.dumps)
    df.to_csv(out / "events_latest.csv", index=False)
    rej = pd.DataFrame(verdict.rejected)
    if not rej.empty:
        # The rejection ledger is the honesty record; keep the *counts* always,
        # and a bounded sample of the rows themselves.
        rej.head(cap).to_csv(out / "rejected_latest.csv", index=False)
    if truncated:
        print(f"[tocsin] events truncated at {cap} rows (of {len(rows)})")


def _write_watchlist(out: Path, led: Ledger, conf: dict) -> None:
    """The cross-night watchlist: everything above ``none``, ranked."""
    order = {"alarm": 0, "candidate": 1, "interest": 2, "watch": 3, "none": 4}
    max_grey_z = float(conf["screen"]["max_grey_z"])
    rows = []
    for tid, rec in led.targets.items():
        if rec.get("tier", "none") == "none":
            continue
        rows.append({
            "target_id": tid, "tier": rec.get("tier"), "ra": rec.get("ra"),
            "dec": rec.get("dec"), "n_events": rec.get("n_events"),
            "n_visits": rec.get("n_visits"), "visits_exact": rec.get("visits_exact"),
            "duty_cycle": rec.get("duty_cycle"),
            "p_binomial": rec.get("p_binomial"), "p_timing": rec.get("p_timing"),
            "timing_period_d": rec.get("timing_period_d"),
            "first_mjd": rec.get("first_mjd"), "last_mjd": rec.get("last_mjd"),
            "polarities": ",".join(sorted({e.get("polarity", "")
                                           for e in rec.get("events", [])})),
            # `_finite`, not `or`: a perfectly grey event has grey_z == 0.0,
            # and `0.0 or fallback` would silently discard the best events in
            # the channel.
            "grey_confirmed": sum(
                1 for e in rec.get("events", [])
                if e.get("grey_tested") and _finite(e.get("grey_z")) is not None
                and abs(_finite(e.get("grey_z"))) <= max_grey_z),
            "notes": ";".join(rec.get("notes", [])),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["_o"] = df["tier"].map(lambda t: order.get(t, 9))
        df = df.sort_values(["_o", "p_binomial"], na_position="last").drop(columns="_o")
    df.to_csv(out / "watchlist.csv", index=False)


def population(cfg=None, out_dir: str | Path | None = None,
               targets_path: str | Path | None = None,
               n_null: int | None = None) -> dict:
    """Population-level structure tests on the accumulated ledger.  Offline.

    Every per-target test in this channel asks "is this star special?", and that
    question died three times on real data (a deep-drilling field, then three
    variable stars).  This stage asks the question that is immune to per-object
    contamination: is the event RATE structured across the screened population in
    a way contaminants cannot produce?  See `seti.tocsin.population`.

    Needs the Gaia target list to rebuild the parent, which the workflow already
    caches, and the ledger's per-bin trial counts to know which targets were
    actually screened.
    """
    from .population import build_parent_population, population_tests

    conf = load_tocsin_config(cfg)
    root = Path(cfg.root) if cfg is not None else _repo_root()
    out = Path(out_dir) if out_dir else root / conf["report"]["results_dir"]
    lconf = conf["ledger"]
    led = Ledger.load(root / lconf["path"])
    tpath = (Path(targets_path) if targets_path
             else root / ".cache" / "tocsin" / "targets.parquet")
    targets = load_targets(tpath)
    rec: dict = {"run_at_utc": _utc(),
                 "n_bins_with_trials": len(led.bin_trials or {})}
    if targets is None or not len(targets):
        rec["verdict"] = "NO_TARGET_LIST"
        rec["note"] = f"target list missing at {tpath}; run tocsin-targets"
        _write_json(out / "population.json", rec)
        return rec
    if not led.bin_trials:
        rec["verdict"] = "NO_BIN_TRIALS"
        rec["note"] = ("the ledger carries no per-bin trial counts, so the "
                       "screened parent cannot be reconstructed; walk at least "
                       "one window with the footprint denominator enabled")
        _write_json(out / "population.json", rec)
        return rec
    rows, mask = build_parent_population(
        targets, led.bin_trials,
        bin_deg=float(conf["acquire"].get("footprint_bin_deg", 1.0)),
        event_targets=led.targets)
    nn = int(n_null if n_null is not None else lconf.get("population_n_null", 2000))
    rec.update(population_tests(rows, mask,
                                n_null=nn,
                                seed=int(lconf.get("population_seed", 20260730))))
    _write_json(out / "population.json", rec)
    print(f"[tocsin] population: {rec.get('verdict')} "
          f"parent={rec.get('n_parent')} anomalies={rec.get('n_anomaly')} "
          f"p_min={rec.get('p_min')}")
    return rec


def assess_only(cfg=None, out_dir: str | Path | None = None) -> dict:
    """Recompute the ledger's statistics offline (no network)."""
    conf = load_tocsin_config(cfg)
    root = Path(cfg.root) if cfg is not None else _repo_root()
    out = Path(out_dir) if out_dir else root / conf["report"]["results_dir"]
    lconf = conf["ledger"]
    path = root / lconf["path"]
    led = Ledger.load(path)
    stats = led.assess(alpha_fdr=float(lconf["alpha_fdr"]),
                       min_visits_for_rate=int(lconf["min_visits_for_rate"]),
                       max_duty_cycle=float(lconf["max_duty_cycle"]),
                       n_null_timing=int(lconf["n_null_timing"]),
                       timing_alpha=float(lconf["timing_alpha"]),
                       max_grey_z=float(conf["screen"]["max_grey_z"]),
                       mixed_polarity_requires_grey_both=bool(
                           lconf.get("mixed_polarity_requires_grey_both", True)))
    led.updated_utc = _utc()
    led.save(path)
    _write_watchlist(out, led, conf)
    _write_json(out / "assessment.json", {"assessed_at_utc": _utc(), **stats})
    print(f"[tocsin] assess: {stats}")
    return stats
