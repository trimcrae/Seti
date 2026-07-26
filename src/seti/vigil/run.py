"""Stage orchestration for VIGIL.  Writes ``results/vigil/``.

Three stages, deliberately separated so the expensive one shards:

``vigil_probe``   one minimal live call per route.  Establishes whether the
                  unTimely mid-IR variable catalogue is reachable, what it is
                  called, and whether NEOWISE per-epoch photometry comes back ---
                  *before* a multi-hour sweep is spent finding out.
``vigil_sweep``   per sky field: Gaia sample -> PM-propagated NEOWISE epochs ->
                  visit binning -> variability -> photosphere -> excess ->
                  modulation index, morphology, colour -> the cut.
``vigil_vet``     across fields: optical constancy, SIMBAD, AllWISE W3/W4, the
                  contamination gauntlet.

Every stage writes an explicit verdict when it could not reach data, and the
archive ledger --- query text, per-stage row counts, ``COUNT(*)`` comparisons,
``QUERY_FAILED`` versus ``QUERY_RETURNED_ZERO_ROWS`` --- goes into
``summary.json``.  A verdict must never read as a science null when no data was
tested; ``NO_DATA_REACHED`` and ``ALL_REJECTED`` are different words here for
exactly that reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..photometry import mag_to_flux_jy
from .discriminate import colour_stability, discriminate, shape_stats
from .excess import measure_excess, photosphere_from_nir, teff_from_colour
from .variability import (
    bin_visits,
    ensemble_common_mode,
    fit_error_scale,
    midir_variability,
    pair_variability,
    visit_flux_series,
)
from .vet import summarise_verdicts, vet_row


def load_vigil_config(cfg=None) -> dict:
    """Read ``config/vigil.yaml``; a missing file degrades to documented defaults."""
    try:
        import yaml
        root = Path(cfg.root) if cfg is not None else Path(__file__).resolve().parents[3]
        p = root / "config" / "vigil.yaml"
        if not p.exists():
            return {}
        return yaml.safe_load(p.read_text()) or {}
    except Exception as exc:                           # noqa: BLE001
        print(f"[vigil] config/vigil.yaml not loaded ({exc!r}); using defaults")
        return {}


def _root(cfg, out_root) -> Path:
    r = Path(out_root) if out_root is not None else (
        (Path(cfg.root) if cfg is not None else Path(".")) / "results" / "vigil")
    r.mkdir(parents=True, exist_ok=True)
    return r


def _field_tag(ra: float, dec: float, radius_deg: float) -> str:
    return f"ra{ra:.3f}_dec{dec:+.3f}_r{radius_deg:.2f}"


# --------------------------------------------------------------------------
# Stage 0: the probe
# --------------------------------------------------------------------------
# The probe's field centre: the North Ecliptic Pole, where the NEOWISE scan
# pattern piles up and the per-star visit count is highest.
PROBE_RA, PROBE_DEC = 270.0, 66.56


def vigil_probe(cfg=None, out_root: Path | None = None, ra: float = PROBE_RA,
                dec: float = PROBE_DEC) -> dict:
    """One minimal call per route.  Cheap, and it decides the run's architecture.

    The NEOWISE leg deliberately does **not** probe a bare coordinate.  The first
    probe run did, got 32 single exposures across the whole mission --- a marginal
    source detected sporadically --- and binned to ZERO usable visits.  The code
    was right to report zero rather than invent a light curve, but the probe was
    then measuring an empty patch of sky rather than the transport.  So the probe
    now *resolves a real star first*: it asks Gaia for a bright, well-behaved star
    in the field and fetches NEOWISE at that star's PM-propagated position.  A
    zero then means the transport is broken, which is what a probe is for.
    """
    from .acquire import fetch_gaia_field, fetch_neowise_epochs, probe_untimely

    root = _root(cfg, out_root)
    out: dict = {"stage": "probe"}
    try:
        out["untimely"] = probe_untimely()
    except Exception as exc:                           # noqa: BLE001
        out["untimely"] = {"reachable": False, "verdict": "PROBE_FAILED",
                           "error": repr(exc)}

    # Resolve a real, bright star in the field before asking for its photometry.
    tgt = {"ra": ra, "dec": dec, "pmra": 0.0, "pmdec": 0.0, "resolved": False}
    try:
        g = fetch_gaia_field(ra, dec, radius_deg=0.3, g_max=12.0,
                             plx_over_err_min=5.0, max_rows=200)
        out["gaia"] = g.to_ledger()
        if g.status == "OK" and g.data is not None and len(g.data):
            d = g.data.sort_values("phot_g_mean_mag")
            row = d.iloc[len(d) // 2]          # mid-range: bright but not saturated
            tgt = {"ra": float(row["ra"]), "dec": float(row["dec"]),
                   "pmra": float(row.get("pmra") or 0.0),
                   "pmdec": float(row.get("pmdec") or 0.0),
                   "g_mag": float(row.get("phot_g_mean_mag") or float("nan")),
                   "source_id": str(row.get("source_id")), "resolved": True}
    except Exception as exc:                           # noqa: BLE001
        out["gaia"] = {"status": "QUERY_FAILED", "error": repr(exc)}
    out["probe_target"] = tgt

    try:
        r = fetch_neowise_epochs(tgt["ra"], tgt["dec"], tgt["pmra"], tgt["pmdec"],
                                 radius_arcsec=3.0)
        out["neowise"] = r.to_ledger()
        if r.data is not None and len(r.data):
            v = bin_visits(r.data["mjd"], r.data["w1mpro"], r.data["w1sigmpro"])
            out["neowise"]["n_visits_binned"] = len(v)
            out["neowise"]["err_scale_fitted"] = fit_error_scale(v)
            out["neowise"]["exposures_per_visit_median"] = (
                float(np.median([x.n_exp for x in v])) if v else 0.0)
            if not v:
                # Distinguish "the transport failed" from "this position has too
                # few exposures per visit to calibrate the noise", which is a
                # sensitivity statement about the source, not about the archive.
                out["neowise"]["binning_note"] = (
                    "rows returned but no visit reached the minimum exposure "
                    "count; the source is too faint or too sparsely detected "
                    "for the within-visit noise calibration")
    except Exception as exc:                           # noqa: BLE001
        out["neowise"] = {"status": "QUERY_FAILED", "error": repr(exc)}

    reach_ut = bool(out.get("untimely", {}).get("reachable"))
    reach_nw = out.get("neowise", {}).get("status") == "OK"
    out["verdict"] = (
        "BOTH_ROUTES_REACHABLE" if (reach_ut and reach_nw) else
        "NEOWISE_ONLY" if reach_nw else
        "UNTIMELY_ONLY" if reach_ut else "NO_DATA_REACHED")
    out["architecture"] = (
        "untimely_preselect_then_neowise_characterise" if (reach_ut and reach_nw)
        else "neowise_field_sweep" if reach_nw else "none")
    (root / "probe.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "untimely"}, indent=2,
                     default=str))
    return out


# --------------------------------------------------------------------------
# Stage 1: the sweep
# --------------------------------------------------------------------------
def _score_star(star: dict, w1_visits, w2_visits, conf: dict) -> dict | None:
    """All of the physics for one star, given its visit-binned light curves."""
    d = dict(star)
    s1 = fit_error_scale(w1_visits)
    s2 = fit_error_scale(w2_visits)
    v1 = midir_variability(w1_visits, band="W1", err_scale=s1,
                           common_mode_removed=bool(star.get("common_mode_removed")))
    v2 = midir_variability(w2_visits, band="W2", err_scale=s2,
                           common_mode_removed=bool(star.get("common_mode_removed")))
    pair = pair_variability(v1, v2)
    d.update(pair.as_dict())
    if v1 is None or v2 is None:
        d["vigil_verdict_stage"] = "NOT_MEASURED"
        return d
    # Fall back to the NEOWISE mean magnitudes when AllWISE did not match, so the
    # saturation, blend and AGN rules in vet.py still have a colour to work with.
    d.setdefault("w1mpro", v1.mean_mag)
    d.setdefault("w2mpro", v2.mean_mag)
    if not np.isfinite(float(d.get("w1mpro") or np.nan)):
        d["w1mpro"] = v1.mean_mag
    if not np.isfinite(float(d.get("w2mpro") or np.nan)):
        d["w2mpro"] = v2.mean_mag

    # --- photosphere and the excess ---------------------------------------
    teff = star.get("teff_gspphot")
    if teff is None or not np.isfinite(float(teff or np.nan)):
        teff = teff_from_colour(star.get("bp_rp", np.nan))
    d["teff_used_k"] = float(teff) if np.isfinite(float(teff or np.nan)) else np.nan
    pred = photosphere_from_nir(star.get("j_m_2mass", np.nan),
                                star.get("h_m_2mass", np.nan),
                                star.get("k_m_2mass", np.nan), d["teff_used_k"])
    exc1 = exc2 = None
    if pred:
        sys_floor = float(conf.get("excess", {}).get("sys_floor", 0.03))
        exc1 = measure_excess("W1", v1.mean_mag, v1.sigma_typ_mag, pred.get("W1", np.nan),
                              0.0, d["teff_used_k"], sys_floor=sys_floor)
        exc2 = measure_excess("W2", v2.mean_mag, v2.sigma_typ_mag, pred.get("W2", np.nan),
                              0.0, d["teff_used_k"], sys_floor=sys_floor)
        d.update({f"exc_w1_{k}": v for k, v in exc1.as_dict().items() if k != "band"})
        d.update({f"exc_w2_{k}": v for k, v in exc2.as_dict().items() if k != "band"})
        d["chi_w1"] = exc1.chi
        d["chi_w2"] = exc2.chi

    # --- morphology and colour --------------------------------------------
    t2, f2, e2 = visit_flux_series(w2_visits, err_scale=s2)
    shp = shape_stats(t2, f2, e2)
    if shp is not None:
        d.update({f"shape_{k}": v for k, v in shp.as_dict().items()})

    col = None
    if pred and exc1 is not None and exc2 is not None:
        col = _colour_from_visits(w1_visits, w2_visits, pred, s1, s2)
        if col is not None:
            d.update({f"colour_{k}": v for k, v in col.as_dict().items()})

    res = discriminate(v1, v2, exc1, exc2, shape=shp, colour=col,
                       optical_fvar=star.get("optical_fvar"),
                       optical_measured=bool(star.get("optical_measured")),
                       cfg=conf.get("detect", {}))
    d.update(res.as_dict())
    d["vigil_verdict_stage"] = res.verdict
    d["is_candidate"] = bool(res.is_candidate)
    d["amp_ptp"] = res.amp_ptp
    return d


def _colour_from_visits(w1_visits, w2_visits, pred: dict, s1: float, s2: float):
    """Per-epoch excess fluxes in both bands, matched by visit, then the colour test."""
    from .acquire import NEOWISE_START  # noqa: F401  (documents the epoch convention)

    t1 = np.array([v.t_mjd for v in w1_visits])
    t2 = np.array([v.t_mjd for v in w2_visits])
    if t1.size == 0 or t2.size == 0:
        return None
    idx = [int(np.argmin(np.abs(t1 - t))) for t in t2]
    keep = [k for k, j in enumerate(idx) if abs(t1[j] - t2[k]) < 30.0]
    if len(keep) < 4:
        return None
    e1, e2, r1, r2, tt = [], [], [], [], []
    for k in keep:
        j = idx[k]
        a = w1_visits[j]
        b = w2_visits[k]
        fa = float(mag_to_flux_jy(a.mag, "W1"))
        fb = float(mag_to_flux_jy(b.mag, "W2"))
        ea = 0.4 * np.log(10.0) * fa * a.err_quoted * s1
        eb = 0.4 * np.log(10.0) * fb * b.err_quoted * s2
        r1.append(fa - float(pred.get("W1", np.nan)))
        r2.append(fb - float(pred.get("W2", np.nan)))
        e1.append(ea)
        e2.append(eb)
        tt.append(b.t_mjd)
    tt = (np.array(tt) - np.min(tt)) / 365.25
    return colour_stability(tt, np.array(r1), np.array(e1), np.array(r2), np.array(e2))


def preselect_from_untimely(stars: pd.DataFrame, table: str, service: str,
                            ra: float, dec: float, radius_deg: float,
                            tol_arcsec: float = 2.0, fetch=None
                            ) -> tuple[pd.DataFrame, dict]:
    """Restrict a Gaia field sample to sources the unTimely catalogue calls variable.

    This is what makes the catalogue *load-bearing* rather than merely probed: it
    supplies scale (which of the >8M mid-IR variables to characterise), while the
    per-epoch NEOWISE photometry supplies the modulation index, morphology and
    colour statistics that no variability catalogue contains.

    Degrades explicitly.  If the catalogue is unreachable or returns nothing, the
    **full** sample is returned with ``applied: False`` and the transport status
    recorded, because silently searching fewer stars and calling it a
    pre-selection is how a channel misreports its own coverage.
    """
    ledger = {"label": "untimely_preselect", "table": table, "service": service,
              "applied": False, "n_in": int(len(stars)), "n_out": int(len(stars)),
              "n_untimely_rows": 0, "status": "NOT_ATTEMPTED"}
    if not table or not service:
        ledger["status"] = "NO_TABLE_DISCOVERED"
        return stars, ledger
    if fetch is None:
        from .acquire import fetch_untimely_variables as fetch
    try:
        r = fetch(table, service, ra, dec, radius_deg)
    except Exception as exc:                           # noqa: BLE001
        ledger.update({"status": "QUERY_FAILED", "error": repr(exc)})
        return stars, ledger
    ledger.update({"status": r.status, "n_untimely_rows": int(r.n_rows),
                   "count_star": r.count_star, "truncated": r.truncated,
                   "query": r.query[:1000]})
    if r.status != "OK" or r.data is None or not len(r.data):
        return stars, ledger

    d = r.data.copy()
    d.columns = [c.lower() for c in d.columns]
    racol = next((c for c in ("ra", "raj2000", "ra_deg", "ramean") if c in d), None)
    deccol = next((c for c in ("dec", "dej2000", "dec_deg", "decmean") if c in d), None)
    if racol is None or deccol is None:
        ledger["status"] = "NO_POSITION_COLUMNS"
        return stars, ledger

    from scipy.spatial import cKDTree
    cosd = max(np.cos(np.radians(float(dec))), 1e-3)
    tree = cKDTree(np.column_stack([
        pd.to_numeric(d[racol], errors="coerce").to_numpy() * cosd,
        pd.to_numeric(d[deccol], errors="coerce").to_numpy()]))
    q = np.column_stack([stars["ra"].to_numpy() * cosd, stars["dec"].to_numpy()])
    dist, _idx = tree.query(q, k=1, distance_upper_bound=tol_arcsec / 3600.0)
    keep = np.isfinite(dist)
    out = stars[keep].copy()
    ledger.update({"applied": True, "n_out": int(len(out)),
                   "tol_arcsec": tol_arcsec})
    print(f"[vigil] unTimely pre-select: {len(stars)} -> {len(out)} "
          f"(matched against {len(d)} catalogue rows)")
    return out, ledger


def group_neowise_by_star(df: pd.DataFrame, stars: pd.DataFrame,
                          tol_arcsec: float = 2.5,
                          from_epoch: float = 2016.0) -> dict[str, pd.DataFrame]:
    """Assign one field's NEOWISE exposures to stars, with PM propagated.

    Each star's position is moved to the NEOWISE mission mid-epoch before
    matching --- the same correction the per-star path applies --- and the match
    radius is widened per star by half its mission-long proper-motion sweep, so a
    high-PM star's track stays inside its own aperture instead of falling out of
    the sample without explanation.  Pure function: no network.
    """
    from scipy.spatial import cKDTree

    from .acquire import NEOWISE_MID_EPOCH, pm_sweep_arcsec, propagate_pm

    out: dict[str, pd.DataFrame] = {}
    if df is None or not len(df) or not len(stars):
        return out
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]
    if "ra" not in d or "dec" not in d:
        return out
    dec0 = float(np.median(pd.to_numeric(d["dec"], errors="coerce").dropna()))
    cosd = max(np.cos(np.radians(dec0)), 1e-3)
    tree = cKDTree(np.column_stack([
        pd.to_numeric(d["ra"], errors="coerce").to_numpy() * cosd,
        pd.to_numeric(d["dec"], errors="coerce").to_numpy()]))

    for _, s in stars.iterrows():
        pmra = float(s.get("pmra", 0.0) or 0.0)
        pmdec = float(s.get("pmdec", 0.0) or 0.0)
        ra_m, dec_m = propagate_pm(float(s["ra"]), float(s["dec"]), pmra, pmdec,
                                   from_epoch, NEOWISE_MID_EPOCH)
        rad = (tol_arcsec + 0.5 * pm_sweep_arcsec(pmra, pmdec)) / 3600.0
        idx = tree.query_ball_point([float(ra_m) * cosd, float(dec_m)], r=rad)
        if len(idx) >= 3:
            out[str(s["source_id"])] = d.iloc[idx].reset_index(drop=True)
    return out


def vigil_sweep(cfg=None, ra: float = 266.0, dec: float = 65.0,
                radius_deg: float = 0.4, g_max: float = 15.0,
                max_stars: int = 400, time_budget_s: float = 3000.0,
                out_root: Path | None = None, untimely_table: str = "",
                untimely_service: str = "", use_field_query: bool = True,
                w1_max: float = 14.5,
                gaia_fetch=None, neowise_fetch=None, allwise_fetch=None,
                untimely_fetch=None, neowise_field_fetch=None) -> dict:
    """Sweep one sky field for mid-IR variables at low fractional excess.

    ``gaia_fetch`` / ``neowise_fetch`` / ``allwise_fetch`` are injectable so the
    orchestration is exercised offline against synthetic archives.  On the runner
    they default to the real ones.
    """
    import time as _time

    conf = load_vigil_config(cfg)
    root = _root(cfg, out_root)
    tag = _field_tag(ra, dec, radius_deg)
    out_dir = root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger: list[dict] = []

    if gaia_fetch is None:
        from .acquire import fetch_gaia_field as gaia_fetch
    if neowise_fetch is None:
        from .acquire import fetch_neowise_epochs as neowise_fetch
    if allwise_fetch is None:
        from .acquire import fetch_allwise_for as allwise_fetch
    if neowise_field_fetch is None and use_field_query:
        from .acquire import fetch_neowise_field as neowise_field_fetch

    gres = gaia_fetch(ra, dec, radius_deg=radius_deg, g_max=g_max)
    ledger.append(gres.to_ledger())
    if gres.status != "OK" or gres.data is None or not len(gres.data):
        return _write_field(out_dir, {
            "field": tag, "ra": ra, "dec": dec, "radius_deg": radius_deg,
            "verdict": "NO_DATA_REACHED",
            "reason": f"gaia_{gres.status}", "archive_ledger": ledger,
            "n_stars_in_field": 0, "n_stars_with_neowise": 0, "n_scored": 0,
            "n_candidates": 0})

    stars = gres.data.copy()
    # If the unTimely mid-IR variable catalogue was discovered by the probe, use
    # it as a pre-selector: it is what turns this from a blind field sweep into a
    # search over the >8M known mid-IR variables.
    stars, ut_ledger = preselect_from_untimely(
        stars, untimely_table, untimely_service, ra, dec, radius_deg,
        fetch=untimely_fetch)
    ledger.append(ut_ledger)
    stars = stars.head(max_stars).copy()
    aw = pd.DataFrame()
    try:
        aw = allwise_fetch(stars[["source_id", "ra", "dec", "pmra", "pmdec"]])
    except Exception as exc:                           # noqa: BLE001
        print(f"[vigil] AllWISE/2MASS unavailable: {exc!r}")
    if len(aw):
        stars = stars.merge(aw, on="source_id", how="left", suffixes=("", "_aw"))

    # --- per-star NEOWISE, then the field-wide common mode ------------------
    t0 = _time.monotonic()
    per_star_w1: dict[str, list] = {}
    per_star_w2: dict[str, list] = {}
    meta: dict[str, dict] = {}
    n_nw_ok = n_nw_zero = n_nw_fail = 0

    # --- fast path: ONE field query, rows assigned to stars locally ---------
    grouped: dict[str, pd.DataFrame] = {}
    if use_field_query and neowise_field_fetch is not None:
        try:
            fr = neowise_field_fetch(ra, dec, radius_deg=radius_deg, w1_max=w1_max)
            ledger.append(fr.to_ledger())
            if fr.status.startswith("OK") and fr.data is not None and len(fr.data):
                grouped = group_neowise_by_star(fr.data, stars)
                print(f"[vigil] field query: {fr.n_rows} rows -> "
                      f"{len(grouped)}/{len(stars)} stars matched")
        except Exception as exc:                       # noqa: BLE001
            ledger.append({"label": "neowise_field", "status": "QUERY_FAILED",
                           "error": repr(exc)})
    for sid, d in grouped.items():
        v1 = bin_visits(d["mjd"], d["w1mpro"], d["w1sigmpro"])
        v2 = bin_visits(d["mjd"], d["w2mpro"], d["w2sigmpro"])
        if not v1 or not v2:
            continue
        n_nw_ok += 1
        per_star_w1[sid] = v1
        per_star_w2[sid] = v2
        meta[sid] = stars[stars["source_id"].astype(str) == sid].iloc[0].to_dict()

    # --- slow path: per-star cones, for whatever the field query did not cover
    for _, s in stars.iterrows():
        if str(s["source_id"]) in meta:
            continue
        if _time.monotonic() - t0 > time_budget_s:
            print("[vigil] time budget reached during NEOWISE fetch")
            break
        sid = str(s["source_id"])
        try:
            r = neowise_fetch(float(s["ra"]), float(s["dec"]),
                              float(s.get("pmra", 0.0) or 0.0),
                              float(s.get("pmdec", 0.0) or 0.0))
        except Exception as exc:                       # noqa: BLE001
            n_nw_fail += 1
            ledger.append({"label": "neowise_epochs", "status": "QUERY_FAILED",
                           "error": repr(exc), "source_id": sid})
            continue
        if r.status == "QUERY_FAILED":
            n_nw_fail += 1
            ledger.append(r.to_ledger() | {"source_id": sid})
            continue
        if r.status == "QUERY_RETURNED_ZERO_ROWS" or r.data is None or not len(r.data):
            n_nw_zero += 1
            continue
        n_nw_ok += 1
        d = r.data
        v1 = bin_visits(d["mjd"], d["w1mpro"], d["w1sigmpro"])
        v2 = bin_visits(d["mjd"], d["w2mpro"], d["w2sigmpro"])
        if not v1 or not v2:
            continue
        per_star_w1[sid] = v1
        per_star_w2[sid] = v2
        meta[sid] = s.to_dict()

    ledger.append({"label": "neowise_epochs_rollup", "status": "SUMMARY",
                   "n_ok": n_nw_ok, "n_zero_rows": n_nw_zero, "n_failed": n_nw_fail,
                   "n_from_field_query": int(len(grouped)),
                   "n_stars_attempted": int(len(stars))})

    if not meta:
        return _write_field(out_dir, {
            "field": tag, "ra": ra, "dec": dec, "radius_deg": radius_deg,
            "verdict": "NO_DATA_REACHED",
            "reason": ("neowise_all_failed" if n_nw_fail and not n_nw_zero
                       else "neowise_returned_no_usable_epochs"),
            "archive_ledger": ledger, "n_stars_in_field": int(len(stars)),
            "n_stars_with_neowise": 0, "n_scored": 0, "n_candidates": 0,
            "neowise_query_rollup": {"n_ok": n_nw_ok, "n_zero_rows": n_nw_zero,
                                     "n_failed": n_nw_fail,
                                     "n_from_field_query": int(len(grouped))}})

    per_star_w1, cm1 = ensemble_common_mode(per_star_w1)
    per_star_w2, cm2 = ensemble_common_mode(per_star_w2)
    cm_applied = bool(cm1.get("applied") and cm2.get("applied"))

    rows = []
    for sid, m in meta.items():
        m = dict(m)
        m["source_id"] = sid
        m["common_mode_removed"] = cm_applied
        rec = _score_star(m, per_star_w1[sid], per_star_w2[sid], conf)
        if rec is not None:
            rows.append(rec)

    df = pd.DataFrame(rows)
    cand = df[df.get("is_candidate", False) == True] if len(df) else df  # noqa: E712
    if len(df):
        keep = [c for c in df.columns if not c.startswith("_")]
        df[keep].to_csv(out_dir / "vigil_stats.csv", index=False)
    if len(cand):
        cand.to_csv(out_dir / "vigil_candidates.csv", index=False)

    stage_counts = (df["vigil_verdict_stage"].value_counts().to_dict()
                    if len(df) and "vigil_verdict_stage" in df else {})
    summary = {
        "field": tag, "ra": ra, "dec": dec, "radius_deg": radius_deg,
        "verdict": "SEARCHED",
        "n_stars_in_field": int(len(stars)),
        "untimely_preselect": ut_ledger,
        "n_stars_with_neowise": int(len(meta)),
        "n_scored": int(len(df)),
        "n_candidates": int(len(cand)),
        "stage_counts": {str(k): int(v) for k, v in stage_counts.items()},
        "ensemble_common_mode_w1": cm1, "ensemble_common_mode_w2": cm2,
        "ensemble_correction_applied": cm_applied,
        "neowise_query_rollup": {"n_ok": n_nw_ok, "n_zero_rows": n_nw_zero,
                                 "n_failed": n_nw_fail,
                                 "n_from_field_query": int(len(grouped))},
        "archive_ledger": ledger,
        "cadence_bias_note": (
            "per-exposure errors are rescaled per star by the within-visit scatter; "
            "the primary statistic is the unbiased normalised excess variance and "
            "its uncertainty carries the exact N dependence (Vaughan et al. 2003); "
            "a per-field ensemble common mode is fitted and removed"),
        "instrumental_bound": (
            "NEOWISE is W1/W2 only (Wien peaks 852 K and 630 K), so this channel "
            "probes hot material and is structurally blind to 100-300 K"),
    }
    return _write_field(out_dir, summary)


def _write_field(out_dir: Path, summary: dict) -> dict:
    (out_dir / "field_summary.json").write_text(json.dumps(summary, indent=2,
                                                           default=str))
    print(f"[vigil] {summary.get('field')}: {summary.get('verdict')} "
          f"({summary.get('n_scored', 0)} scored, "
          f"{summary.get('n_candidates', 0)} candidates)")
    return summary


# --------------------------------------------------------------------------
# Stage 2: vetting
# --------------------------------------------------------------------------
def vigil_vet(cfg=None, out_root: Path | None = None, max_candidates: int = 200,
              offline: bool = False) -> dict:
    """Aggregate every field, then run the contamination gauntlet on the shortlist."""
    import glob

    root = _root(cfg, out_root)
    n_fields_searched = n_fields_nodata = n_scored = n_stars = 0
    frames = []
    for fp in sorted(glob.glob(str(root / "*" / "field_summary.json"))):
        try:
            s = json.loads(Path(fp).read_text())
        except Exception:                              # noqa: BLE001
            continue
        n_scored += int(s.get("n_scored", 0) or 0)
        n_stars += int(s.get("n_stars_with_neowise", 0) or 0)
        if s.get("verdict") == "SEARCHED":
            n_fields_searched += 1
        else:
            n_fields_nodata += 1
    for fp in sorted(glob.glob(str(root / "*" / "vigil_candidates.csv"))):
        try:
            d = pd.read_csv(fp)
        except Exception:                              # noqa: BLE001
            continue
        if len(d):
            d["field_dir"] = Path(fp).parent.name
            frames.append(d)

    if not frames:
        verdict = "NO_CANDIDATES" if n_fields_searched else "NO_DATA_REACHED"
        summary = {
            "verdict": verdict,
            "n_fields_searched": n_fields_searched,
            "n_fields_no_data": n_fields_nodata,
            "n_stars_with_neowise": n_stars, "n_scored": n_scored,
            "n_candidates": 0, "n_survivors": 0,
            "note": ("no star passed the low-excess/high-modulation gate. This is a "
                     "count over what was actually searched, not an occurrence "
                     "limit; see docs/vigil.md" if verdict == "NO_CANDIDATES" else
                     "no field returned usable data; nothing was tested"),
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return summary

    cand = pd.concat(frames, ignore_index=True).drop_duplicates("source_id")
    sort_key = "modulation_index" if "modulation_index" in cand else "f_var"
    cand = cand.sort_values(sort_key, ascending=False).head(max_candidates)
    print(f"[vigil-vet] vetting {len(cand)} candidates from {n_fields_searched} fields")

    ctxs = []
    for _, r in cand.iterrows():
        ctx = r.to_dict()
        if not offline:
            try:
                from .acquire import fetch_optical_constancy
                ctx.update(fetch_optical_constancy(float(r["ra"]), float(r["dec"])))
            except Exception as exc:                   # noqa: BLE001
                print(f"[vigil-vet] optical constancy failed: {exc!r}")
            try:
                from .acquire import fetch_simbad_type
                ctx["simbad_otype"] = fetch_simbad_type(float(r["ra"]), float(r["dec"]))
            except Exception as exc:                   # noqa: BLE001
                print(f"[vigil-vet] SIMBAD failed: {exc!r}")
        v = vet_row(ctx)
        v["source_id"] = r["source_id"]
        v["optical_measured"] = bool(ctx.get("optical_measured"))
        v["optical_fvar"] = ctx.get("optical_fvar", float("nan"))
        v["simbad_otype"] = ctx.get("simbad_otype", "")
        ctxs.append(v)

    vdf = pd.DataFrame(ctxs)
    vetted = cand.merge(vdf, on="source_id", how="left", suffixes=("", "_vet"))
    vetted.to_csv(root / "vigil_vetted.csv", index=False)

    survivors = vetted[vetted["vigil_verdict"].isin(
        ["clean_duty_cycle", "clean_optical_untested"])]
    gold = vetted[vetted["vigil_verdict"] == "clean_duty_cycle"]
    if len(survivors):
        survivors.to_csv(root / "vigil_survivors.csv", index=False)

    summary = {
        "verdict": ("SURVIVORS" if len(gold) else
                    ("SURVIVORS_OPTICAL_UNTESTED" if len(survivors) else
                     "ALL_REJECTED")),
        "n_fields_searched": n_fields_searched,
        "n_fields_no_data": n_fields_nodata,
        "n_stars_with_neowise": n_stars, "n_scored": n_scored,
        "n_candidates": int(len(cand)), "n_vetted": int(len(vetted)),
        "n_survivors": int(len(survivors)), "n_clean": int(len(gold)),
        "verdict_counts": summarise_verdicts(ctxs),
        "offline": bool(offline),
        "discriminator": ("modulation index m = A_obs / A_max(tau): the fraction of "
                          "the inferred excess that is actually switching. An extreme "
                          "debris disk has tau ~ 1e-2 with m << 1; a duty-cycled "
                          "radiator has low tau with m -> 1"),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


__all__ = ["group_neowise_by_star", "load_vigil_config",
           "preselect_from_untimely", "vigil_probe",
           "vigil_sweep", "vigil_vet"]
