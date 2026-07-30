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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .brokers import (
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
                "audit_without_gaia_join_every_n_runs": 14},
    "screen": {"min_abs_snr": 6.0, "min_reliability": 0.0,
               "require_reliability": False, "max_dipole_significance": 3.0,
               "max_extendedness": 0.5, "max_trail_arcsec": 0.5,
               "match_radius_arcsec": 1.5, "max_sep_arcsec": 1.0,
               "max_sep_sigma": 3.0, "max_grey_z": 3.0,
               "baseline_rel_err": 0.03, "missing_pm_penalty_arcsec": 2.0},
    "ledger": {"path": "results/tocsin/ledger.json", "alpha_fdr": 0.05,
               "min_visits_for_rate": 5, "max_duty_cycle": 0.2,
               "n_null_timing": 2000, "timing_alpha": 0.01},
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
    try:
        rec["alerce"]["schema"] = tap.describe()
        rec["alerce"]["reached"] = True
    except Exception as exc:                              # noqa: BLE001
        rec["alerce"]["error"] = f"{type(exc).__name__}: {exc}"[:800]

    # The diagnostic battery.  A schema dump says a column exists; it does not
    # say whether any row populates it, how current the data are, or what the
    # discriminating keys actually mean.  The first probe returned an empty
    # night window and the schema alone could not say why, so this runs a set of
    # small independent queries and captures each error separately.
    if rec["alerce"]["reached"]:
        try:
            rec["alerce"]["diagnostics"] = tap.diagnostics(
                _now_mjd(), gaia_catid=int(conf["acquire"].get("gaia_catid", 1)))
        except Exception as exc:                          # noqa: BLE001
            rec["alerce"]["diagnostics"] = {
                "error": f"{type(exc).__name__}: {exc}"[:800]}

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
        try:
            fp = tap.forced_photometry_night(
                lo, hi, parallax_min_mas=None, sid_diaobject=None, maxrec=20)
            rec["alerce"]["forced_photometry"] = {
                "verdict": fp.verdict, "rows": len(fp.rows),
                "columns": sorted(fp.rows[0].keys()) if fp.rows else []}
        except Exception as exc:                          # noqa: BLE001
            rec["alerce"]["forced_photometry"] = {
                "error": f"{type(exc).__name__}: {exc}"[:800]}

    rec["verdict"] = "OK" if rec["alerce"].get("reached") else "NO_DATA_REACHED"
    _write_json(out / "probe.json", rec)
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


def _visit_history_from_forced(rows, targets, th: Thresholds, epoch_jyear: float
                               ) -> tuple[dict[str, list[float]], int, dict]:
    """Map forced-photometry rows onto targets and count the night's trials.

    The association uses the *same* proper-motion-propagated matcher as the
    events do.  Using the broker's own cross-match for the denominator and ours
    for the numerator would silently mix two different definitions of "this
    star", and the ratio of the two would not be a rate.
    """
    stats = {"forced_rows": len(rows), "forced_matched": 0}
    if not rows or targets is None or len(targets) == 0:
        return {}, 0, stats
    fra = np.array([float(r.get("ra", np.nan)) for r in rows])
    fdec = np.array([float(r.get("dec", np.nan)) for r in rows])
    fmjd = np.array([float(r.get("mjd", np.nan)) for r in rows])
    ok = np.isfinite(fra) & np.isfinite(fdec) & np.isfinite(fmjd)
    if not np.any(ok):
        return {}, 0, stats
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
    return ({k: sorted(set(v)) for k, v in hist.items()}, len(star_nights), stats)


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

    hi = float(mjd_hi) if mjd_hi is not None else _now_mjd()
    lo = (float(mjd_lo) if mjd_lo is not None
          else hi - float(lookback_nights if lookback_nights is not None
                          else aconf["lookback_nights"]))
    summary: dict = {
        "run_at_utc": _utc(), "mjd_lo": round(lo, 5), "mjd_hi": round(hi, 5),
        "nights": sorted({night_of(lo), night_of(hi)}),
        "use_gaia_join": bool(use_gaia_join), "verdict": "NOT_RUN", "counts": {},
        "notes": [],
    }

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
    try:
        det = tap.night_detections(
            lo, hi, parallax_min_mas=plx,
            xmatch_max_arcsec=float(aconf["xmatch_max_arcsec"]),
            min_abs_snr=float(conf["screen"]["min_abs_snr"]))
    except BrokerError as exc:
        summary["verdict"] = "NO_DATA_REACHED"
        summary["error"] = str(exc)[:800]
        _write_json(out / "summary.json", summary)
        print(f"[tocsin] {summary['verdict']}: {summary['error']}")
        return summary
    summary["counts"]["detections_pulled"] = len(det.rows)
    summary["notes"].extend(n for n in det.notes if not n.startswith("adql="))

    alerts = normalize_alerce_rows(det.rows)
    summary["counts"]["normalised"] = len(alerts)

    epoch_jyear = 2000.0 + ((lo + hi) / 2.0 - 51544.5) / 365.25
    verdict = screen_alerts(alerts, targets, th, epoch_jyear=epoch_jyear)
    summary["counts"].update(verdict.counts)
    summary["notes"].extend(verdict.notes)

    # The denominator: forced photometry on every tracked nearby star tonight.
    try:
        fp = tap.forced_photometry_night(
            lo, hi, parallax_min_mas=plx,
            xmatch_max_arcsec=float(aconf["xmatch_max_arcsec"]))
        hist, star_nights, fstats = _visit_history_from_forced(
            fp.rows, targets, th, epoch_jyear)
        summary["counts"].update(fstats)
        summary["counts"]["target_nights_screened"] = star_nights
        summary["denominator"] = "forced_photometry_exact"
    except BrokerError as exc:
        hist, star_nights = {}, 0
        summary["denominator"] = "unavailable"
        summary["notes"].append(f"forced_photometry_failed: {str(exc)[:300]}")

    # Merge the forced history the events themselves carried (if any broker
    # supplied it) with the bulk history.
    for tid, mjds in verdict.visit_history.items():
        hist.setdefault(tid, []).extend(mjds)
        hist[tid] = sorted(set(hist[tid]))

    lconf = conf["ledger"]
    ledger_path = root / lconf["path"]
    led = Ledger.load(ledger_path)
    if not led.opened_utc:
        led.opened_utc = _utc()
    night_label = f"n{night_of(lo)}-n{night_of(hi)}"
    led.add_night(night_label, verdict.events, target_visits=star_nights,
                  targets_in_footprint=len(hist) or len(targets),
                  alerts_seen=len(alerts), visit_history=hist,
                  target_positions=verdict.target_positions)
    led.updated_utc = _utc()
    stats = led.assess(alpha_fdr=float(lconf["alpha_fdr"]),
                       min_visits_for_rate=int(lconf["min_visits_for_rate"]),
                       max_duty_cycle=float(lconf["max_duty_cycle"]),
                       n_null_timing=int(lconf["n_null_timing"]),
                       timing_alpha=float(lconf["timing_alpha"]),
                       max_grey_z=float(conf["screen"]["max_grey_z"]))
    led.save(ledger_path)

    summary["ledger"] = stats
    summary["verdict"] = "OK" if det.rows else "NO_DETECTIONS_IN_WINDOW"
    _write_events(out, verdict, conf)
    _write_watchlist(out, led, conf)
    _write_json(out / "summary.json", summary)
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
                       max_grey_z=float(conf["screen"]["max_grey_z"]))
    led.updated_utc = _utc()
    led.save(path)
    _write_watchlist(out, led, conf)
    _write_json(out / "assessment.json", {"assessed_at_utc": _utc(), **stats})
    print(f"[tocsin] assess: {stats}")
    return stats
