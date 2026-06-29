"""End-to-end anomalous-dimming search: acquire light curves, score, summarise.

Pulls a Gaia stellar sample in a sky field, fetches each star's ZTF light curve,
scores it for the Boyajian's-star signature (deep, irregular, *aperiodic*
dimming) with :func:`seti.dimming.dips.detect_dips`, ranks the dippers, places an
occurrence-rate upper limit on the deep-aperiodic-dipper fraction, and writes
small, committable result files under ``results/dimming/``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..stats.upper_limit import occurrence_upper_limit
from .context import hr_class, resists_mundane
from .dips import detect_dips
from .secular import detect_secular_fade, fit_secular, season_medians


def _is_candidate(stat: dict, depth_min: float, n_dips_min: int,
                  asym_min: float, period_power_max: float,
                  max_events: int = 20, out_rms_max: float = 0.05) -> bool:
    """A dimming candidate: deep + a *few discrete* events on a *quiescent* baseline
    + asymmetric + NOT strongly periodic.

    The event ceiling and the out-of-dip quiescence gate are the decisive
    mundane-rejections: a high-amplitude pulsator or eclipsing binary produces
    hundreds of below-baseline epochs and a large baseline RMS, whereas KIC 8462852
    shows a handful of deep, discrete dips on an otherwise flat light curve.
    """
    n_events = stat.get("n_dip_events", stat.get("n_dips", 0))
    # Depth must come from a *sustained* (multi-epoch) event, and there must be at
    # least one such event --- a lone faint-end outlier no longer qualifies.
    sustained_depth = stat.get("max_event_depth", stat.get("max_depth", 0.0))
    return (sustained_depth >= depth_min
            and n_events >= 1
            and stat.get("n_dips", 0) >= n_dips_min
            and n_events <= max_events
            and stat.get("out_of_dip_rms", 0.0) <= out_rms_max
            and stat.get("asymmetry", 0.0) >= asym_min
            and stat.get("period_power", 1.0) <= period_power_max)


def _attach_gaia_hr(candidates: list[dict], period_power_max: float,
                    match_arcsec: float = 2.0) -> None:
    """Cross-match region-mode candidates to Gaia DR3 and apply the HR cut in place.

    Region mode discovers ZTF sources with no Gaia photometry, so the HR-diagram
    main-sequence cut is deferred to this short candidate list.  For each candidate
    we pull the nearest Gaia source (G, BP-RP, parallax) and recompute ``hr_class``
    and ``resists_mundane``.
    """
    from astroquery.gaia import Gaia

    pts = ", ".join(f"({r['ra']:.6f}, {r['dec']:.6f})" for r in candidates
                    if np.isfinite(r.get("ra", np.nan))
                    and np.isfinite(r.get("dec", np.nan)))
    if not pts:
        return
    # One ADQL cone-per-candidate union via a VALUES-like table is awkward; instead
    # query a small bounding region and match locally (the candidate list is short).
    ras = [r["ra"] for r in candidates if np.isfinite(r.get("ra", np.nan))]
    decs = [r["dec"] for r in candidates if np.isfinite(r.get("dec", np.nan))]
    ra0, ra1, dec0, dec1 = min(ras), max(ras), min(decs), max(decs)
    pad = 0.02
    query = f"""
        SELECT ra, dec, phot_g_mean_mag, bp_rp, parallax, parallax_over_error
        FROM gaiadr3.gaia_source
        WHERE ra BETWEEN {ra0 - pad} AND {ra1 + pad}
          AND dec BETWEEN {dec0 - pad} AND {dec1 + pad}
          AND phot_g_mean_mag IS NOT NULL
    """
    gtab = Gaia.launch_job_async(query).get_results().to_pandas()
    gtab = gtab.rename(columns={c: c.lower() for c in gtab.columns})
    if gtab.empty:
        return
    tol = match_arcsec / 3600.0
    for r in candidates:
        ra, dec = r.get("ra"), r.get("dec")
        if not (np.isfinite(ra) and np.isfinite(dec)):
            continue
        cosd = np.cos(np.radians(dec))
        d2 = ((gtab["ra"] - ra) * cosd) ** 2 + (gtab["dec"] - dec) ** 2
        j = int(np.argmin(d2.to_numpy()))
        if float(d2.iloc[j]) > tol ** 2:
            continue
        g = gtab.iloc[j]
        hr = hr_class(float(g.get("phot_g_mean_mag", np.nan)),
                      float(g.get("bp_rp", np.nan)),
                      float(g.get("parallax", np.nan)),
                      float(g.get("parallax_over_error", 0.0) or 0.0))
        r["g_mag"] = float(g.get("phot_g_mean_mag", np.nan))
        r["bp_rp"] = float(g.get("bp_rp", np.nan))
        r["parallax"] = float(g.get("parallax", np.nan))
        r["hr_class"] = hr
        r["resists_mundane"] = bool(resists_mundane(
            hr, r.get("period_power", 1.0), period_power_max))


def _flag_shared_epoch_dips(rows: list[dict], depth_min: float = 0.10,
                            bin_days: float = 0.5, min_share: int = 5,
                            share_frac: float = 0.02) -> int:
    """Reject dips that fall on epochs shared by many stars in the same field.

    A genuine occultation dims one star on its own dates; a bad reference image or
    a bad-calibration night dims *many* stars in the field at the *same* epoch.
    We bin every star's dip epochs, mark epochs hit by an anomalous number of
    stars as bad, and demote any candidate whose dips are mostly on bad epochs ---
    the dimming analog of the laser recurrent-wavelength cut.  Returns the number
    of candidates demoted.
    """
    from collections import defaultdict
    star_bins: list[tuple[dict, set]] = []
    epoch_counts: dict[int, int] = defaultdict(int)
    for r in rows:
        mjd, mag = r.get("_mjd"), r.get("_mag")
        if mjd is None or not len(mjd):
            continue
        base = float(np.percentile(mag, 20))
        frac = 1.0 - 10.0 ** (-0.4 * (mag - base))
        bins = set(np.round(mjd[frac >= depth_min] / bin_days).astype(int).tolist())
        if not bins:
            continue
        star_bins.append((r, bins))
        for b in bins:
            epoch_counts[b] += 1
    if not star_bins:
        return 0
    n_stars = len(star_bins)
    thresh = max(min_share, int(np.ceil(share_frac * n_stars)))
    bad = {b for b, c in epoch_counts.items() if c >= thresh}
    if not bad:
        return 0
    demoted = 0
    for r, bins in star_bins:
        shared = len(bins & bad)
        r["shared_epoch_frac"] = shared / len(bins)
        if r.get("is_candidate") and shared / len(bins) >= 0.5:
            r["is_candidate"] = False
            r["resists_mundane"] = False
            r["rejected_shared_epoch"] = True
            demoted += 1
    return demoted


def _ensemble_detrend_secular(rows: list[dict]) -> None:
    """Subtract the field common-mode seasonal drift and re-fit the secular fade.

    ZTF zeropoint / reference-image changes impose a shared seasonal magnitude
    offset on *every* star in a field; uncorrected, this manufactures spurious
    "secular fades" (the dominant false positive).  We estimate the common mode as
    the median over all stars of (season median - star median) in each season, and
    subtract it before re-fitting each star.  A star that still fades after the
    field's shared drift is removed is an *intrinsic* fader.
    """
    # Accumulate per-CCD (ZTF zeropoint/reference drift is per readout channel):
    # offsets[ccd][season] -> list of per-star (season median - star median).
    from collections import defaultdict
    offsets: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    glob: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        sm = r.get("_sm")
        if sm is None:
            continue
        labels, _s_t, s_m, _s_w = sm
        omed = r.get("_omed", float("nan"))
        if not np.isfinite(omed):
            continue
        ccd = r.get("_ccd", "x")
        for lab, mm in zip(labels, s_m, strict=False):
            offsets[ccd][int(lab)].append(float(mm) - omed)
            glob[int(lab)].append(float(mm) - omed)
    if not glob:
        return
    # Per-CCD common mode where a season has >=5 stars on that CCD; otherwise fall
    # back to the global field common mode (still better than no correction).
    common = {ccd: {lab: float(np.median(v)) for lab, v in seasons.items()
                    if len(v) >= 5}
              for ccd, seasons in offsets.items()}
    common_glob = {lab: float(np.median(v)) for lab, v in glob.items() if len(v) >= 5}
    for r in rows:
        sm = r.get("_sm")
        if sm is None:
            continue
        labels, s_t, s_m, s_w = sm
        ccd_cm = common.get(r.get("_ccd", "x"), {})
        corr = np.array([ccd_cm.get(int(lab), common_glob.get(int(lab), 0.0))
                         for lab in labels])
        det_m = s_m - corr                 # remove the per-CCD shared drift
        stat = fit_secular(s_t, det_m, s_w, n_epochs=r.get("_nepoch", 0))
        if stat is None:
            r["is_secular_fader"] = False
            continue
        r["secular_slope_mag_yr"] = stat.slope_mag_yr
        r["secular_sigma"] = stat.slope_sigma
        r["secular_total_mag"] = stat.total_change_mag
        r["secular_score"] = stat.score
        r["is_secular_fader"] = bool(stat.score >= 0.5 and stat.slope_sigma >= 4.0)


def dimming_run(
    cfg: Config | None = None,
    ra: float = 270.0,
    dec: float = 30.0,
    radius_deg: float = 1.5,
    g_min: float = 13.0,
    g_max: float = 18.5,
    variable_only: bool = True,
    band: str = "r",
    limit: int = 4000,
    min_epochs: int = 30,
    depth_min: float = 0.10,
    n_dips_min: int = 3,
    asym_min: float = 1.5,
    period_power_max: float = 0.4,
    time_budget_s: float = 1800.0,
    mode: str = "targets",
    box_deg: float = 0.12,
    lightcurves: list[dict] | None = None,
) -> dict:
    """Search a field of stars for deep aperiodic dimming and write results.

    ``lightcurves`` may be passed directly (offline tests): a list of dicts with
    ``source_id``/``ra``/``dec``/``mjd``/``mag``/``magerr``.  Otherwise targets are
    pulled from Gaia and light curves from ZTF on the runner.  Returns the summary.
    """
    cfg = cfg or load_config()

    rows: list[dict] = []
    n_searched = 0

    def _score_one(meta: dict, mjd, mag, magerr) -> None:
        nonlocal n_searched
        stat = detect_dips(np.asarray(mjd, float), np.asarray(mag, float),
                           np.asarray(magerr, float) if magerr is not None else None,
                           depth_min=0.05, min_epochs=min_epochs)
        if stat is None:
            return
        n_searched += 1
        d = stat.as_dict()
        # HR-diagram context: reject the dominant astrophysical mimics (R CrB /
        # Mira giants, white dwarfs, YSO dippers) by demanding the main sequence.
        g_mag = meta.get("phot_g_mean_mag")
        bp_rp = meta.get("bp_rp")
        plx = meta.get("parallax")
        plx_snr = meta.get("parallax_over_error", 0.0)
        hr = hr_class(float(g_mag) if g_mag is not None else float("nan"),
                      float(bp_rp) if bp_rp is not None else float("nan"),
                      float(plx) if plx is not None else float("nan"),
                      float(plx_snr) if plx_snr is not None else 0.0)
        is_cand = _is_candidate(d, depth_min, n_dips_min, asym_min, period_power_max)
        # Second, artifact-robust signature: a significant monotonic multi-year
        # fade (the Schaefer secular dimming of KIC 8462852).  Measured from season
        # medians, so it is immune to the single-epoch artefacts that dominate the
        # dip channel.  A secular fader is its own candidate class.
        mjd_a = np.asarray(mjd, float)
        mag_a = np.asarray(mag, float)
        err_a = np.asarray(magerr, float) if magerr is not None else None
        sec = detect_secular_fade(mjd_a, mag_a, err_a, min_epochs=min_epochs)
        is_fader = bool(sec is not None and sec.score >= 0.5
                        and sec.slope_sigma >= 4.0)
        # Stash season medians so the field common-mode (ZTF zeropoint/reference
        # drift) can be subtracted in an ensemble pass -- the decisive contamination
        # control for the secular channel.
        sm = season_medians(mjd_a, mag_a, err_a, min_epochs=min_epochs)
        d.update({"source_id": meta.get("source_id"), "ra": meta.get("ra"),
                  "dec": meta.get("dec"),
                  "g_mag": g_mag, "bp_rp": bp_rp, "parallax": plx,
                  "hr_class": hr,
                  "secular_slope_mag_yr": sec.slope_mag_yr if sec else float("nan"),
                  "secular_sigma": sec.slope_sigma if sec else float("nan"),
                  "secular_total_mag": sec.total_change_mag if sec else float("nan"),
                  "secular_score": sec.score if sec else 0.0,
                  "is_secular_fader": is_fader,
                  "resists_mundane": bool(is_cand and resists_mundane(
                      hr, d.get("period_power", 1.0), period_power_max)),
                  "is_candidate": is_cand})
        # Keep the light curve only for the strongest dippers (committed JSON).
        d["_mjd"], d["_mag"] = mjd_a, mag_a
        d["_sm"] = sm                      # (labels, s_t, s_m, s_w) or None
        d["_omed"] = float(np.median(mag_a)) if mag_a.size else float("nan")
        d["_nepoch"] = int(mjd_a.size)
        d["_ccd"] = meta.get("ccd", "x")   # ZTF field/CCD/quadrant for detrend
        rows.append(d)

    if lightcurves is not None:
        for lc in lightcurves:
            _score_one(lc, lc["mjd"], lc["mag"], lc.get("magerr"))
    elif mode == "region":
        # Bulk box-sweep: search EVERY ZTF source in the field (no Gaia target
        # bottleneck), 10-100x more stars per run.  HR vetting is applied to the
        # shortlist afterwards by matching candidate positions to Gaia.
        from .acquire import iter_region_lightcurves
        for meta, lc in iter_region_lightcurves(
                ra, dec, radius_deg=radius_deg, box_deg=box_deg, band=band,
                min_epochs=min_epochs, time_budget_s=time_budget_s):
            _score_one(meta, lc["mjd"].to_numpy(), lc["mag"].to_numpy(),
                       lc["magerr"].to_numpy())
            if n_searched and n_searched % 500 == 0:
                nc = sum(1 for r in rows if r["is_candidate"])
                print(f"[dimming] progress: {n_searched} scored, {nc} candidates")
    else:
        from .acquire import fetch_gaia_targets, iter_lightcurves
        targets = fetch_gaia_targets(ra, dec, radius_deg=radius_deg, g_min=g_min,
                                     g_max=g_max, variable_only=variable_only,
                                     limit=limit)
        for meta, lc in iter_lightcurves(targets, band=band,
                                         time_budget_s=time_budget_s):
            _score_one(meta, lc["mjd"].to_numpy(), lc["mag"].to_numpy(),
                       lc["magerr"].to_numpy())
            if n_searched and n_searched % 200 == 0:
                nc = sum(1 for r in rows if r["is_candidate"])
                print(f"[dimming] progress: {n_searched} scored, {nc} candidates")

    # Field-level contamination controls (need the whole field):
    #  (1) shared-epoch dips: reject bad-reference/bad-night artifacts that dim many
    #      stars on the same epoch (the dimming analog of the recurrent-line cut);
    #  (2) ensemble common-mode detrend of the secular channel.
    if len(rows) >= 10:
        n_shared = _flag_shared_epoch_dips(rows)
        if n_shared:
            print(f"[dimming] shared-epoch cut demoted {n_shared} dip candidates")
        _ensemble_detrend_secular(rows)

    candidates = [r for r in rows if r["is_candidate"]]
    # Region mode has no Gaia photometry at detection time: cross-match the (short)
    # candidate list to Gaia DR3 now to apply the HR-diagram main-sequence cut.
    if mode == "region" and candidates:
        try:
            _attach_gaia_hr(candidates, period_power_max)
        except Exception as exc:
            print(f"[dimming] Gaia HR cross-match skipped: {exc!r}")
    # Rank main-sequence aperiodic dippers (those that resist the mundane
    # giant/YSO/eclipse explanations) above the rest, then by dimming score.
    candidates.sort(key=lambda r: (r.get("resists_mundane", False),
                                   r.get("score", 0.0)), reverse=True)
    n_resists = sum(1 for r in candidates if r.get("resists_mundane"))

    # Namespace output by sky field so a multi-field hunt accumulates rather than
    # clobbering: concurrent/sequential runs each write their own subdirectory.
    field_tag = f"f{ra:+06.1f}{dec:+05.1f}".replace(".", "p").replace("+", "p").replace("-", "m")
    out_dir = cfg.root / "results" / "dimming" / field_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the top dippers' light curves so the actual dimming can be examined.
    windows: list[dict] = []
    for r in candidates[:30]:
        windows.append({
            "source_id": r.get("source_id"), "ra": r.get("ra"), "dec": r.get("dec"),
            "score": r.get("score"), "max_depth": r.get("max_depth"),
            "n_dips": r.get("n_dips"), "asymmetry": r.get("asymmetry"),
            "best_period_d": r.get("best_period_d"),
            "period_power": r.get("period_power"),
            "hr_class": r.get("hr_class"),
            "resists_mundane": r.get("resists_mundane"),
            "mjd": r["_mjd"].tolist(), "mag": r["_mag"].tolist()})
    (out_dir / "top_dippers.json").write_text(json.dumps(windows))

    # Secular faders: the second, artifact-robust candidate class.  Rank by the
    # significance of the monotonic multi-year fade.
    raw_faders = [r for r in rows if r.get("is_secular_fader")]
    if mode == "region" and raw_faders:
        try:
            _attach_gaia_hr(raw_faders, period_power_max)
        except Exception as exc:
            print(f"[dimming] Gaia HR cross-match (faders) skipped: {exc!r}")
    # Require a Gaia main-sequence match: the faint hr=unknown population is a
    # magnitude-dependent ZTF systematic (no parallax, clustered per CCD) that no
    # common-mode removes; a real enshrouding candidate must be a characterisable
    # main-sequence star, exactly as for the dippers.
    n_faders_raw = len(raw_faders)
    faders = [r for r in raw_faders if r.get("hr_class") == "main_sequence"]
    print(f"[dimming] secular faders: {n_faders_raw} raw -> {len(faders)} "
          f"main-sequence (faint hr=unknown systematics removed)")
    faders.sort(key=lambda r: r.get("secular_sigma", 0.0), reverse=True)
    fader_windows = [{
        "source_id": r.get("source_id"), "ra": r.get("ra"), "dec": r.get("dec"),
        "secular_slope_mag_yr": r.get("secular_slope_mag_yr"),
        "secular_sigma": r.get("secular_sigma"),
        "secular_total_mag": r.get("secular_total_mag"),
        "secular_score": r.get("secular_score"), "hr_class": r.get("hr_class"),
        "mjd": r["_mjd"].tolist(), "mag": r["_mag"].tolist()} for r in faders[:30]]
    (out_dir / "top_faders.json").write_text(json.dumps(fader_windows))

    # Flat candidate table (drop the bulky light-curve arrays).
    if rows:
        flat = pd.DataFrame([{k: v for k, v in r.items()
                              if not k.startswith("_")} for r in rows])
        flat = flat.sort_values("score", ascending=False)
        flat.to_csv(out_dir / "dimming_scored.csv", index=False)
        # Always (over)write the candidate tables -- even when empty -- so a re-run
        # that now finds nothing clears any stale candidates a prior run committed.
        flat[flat["is_candidate"]].to_csv(
            out_dir / "dimming_candidates.csv", index=False)
        # Only the main-sequence faders (the faint hr=unknown raw faders are a ZTF
        # magnitude-dependent systematic and must not reach the vet stage).
        fader_ids = {r.get("source_id") for r in faders}
        flat[flat["source_id"].isin(fader_ids)].to_csv(
            out_dir / "secular_faders.csv", index=False)

    k = len(candidates)
    lim = occurrence_upper_limit(
        k=k, n_eff=max(n_searched, 1),
        confidence=cfg.thresholds["stats"]["upper_limit_confidence"])

    summary = {
        "field": {"ra": ra, "dec": dec, "radius_deg": radius_deg},
        "band": band, "variable_only": variable_only,
        "n_searched": n_searched,
        "n_candidates": k,
        "n_resists_mundane": n_resists,
        "n_secular_faders": len(faders),
        "n_secular_faders_raw": n_faders_raw,
        "top_faders": [
            {"source_id": r.get("source_id"), "ra": r.get("ra"), "dec": r.get("dec"),
             "secular_slope_mag_yr": r.get("secular_slope_mag_yr"),
             "secular_sigma": r.get("secular_sigma"),
             "secular_total_mag": r.get("secular_total_mag"),
             "hr_class": r.get("hr_class")} for r in faders[:20]],
        "selection": {"depth_min": depth_min, "n_dips_min": n_dips_min,
                      "asym_min": asym_min, "period_power_max": period_power_max},
        "top_candidates": [
            {"source_id": r.get("source_id"), "ra": r.get("ra"), "dec": r.get("dec"),
             "score": r.get("score"), "max_depth": r.get("max_depth"),
             "n_dips": r.get("n_dips"), "n_dip_events": r.get("n_dip_events"),
             "max_event_depth": r.get("max_event_depth"),
             "out_of_dip_rms": r.get("out_of_dip_rms"),
             "asymmetry": r.get("asymmetry"),
             "best_period_d": r.get("best_period_d"),
             "period_power": r.get("period_power"),
             "hr_class": r.get("hr_class"),
             "resists_mundane": r.get("resists_mundane")} for r in candidates[:20]],
        "occurrence_limit": {
            "k_candidates": lim.k, "n_eff": lim.n_eff, "confidence": lim.confidence,
            "f_upper": lim.f_upper, "f_point": lim.f_point},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # SIMBAD vetting of the (short) candidate list.
    if candidates:
        try:
            from ..acquire.science import classify_candidate, fetch_simbad_context
            pos = pd.DataFrame([{"source_id": r.get("source_id"), "ra": r.get("ra"),
                                 "dec": r.get("dec")} for r in candidates[:50]])
            ctx = fetch_simbad_context(pos)
            if ctx is not None and len(ctx):
                ctx = ctx.assign(candidate_class=[
                    classify_candidate(o, s) for o, s in
                    zip(ctx.get("simbad_otype", ""), ctx.get("simbad_sptype", ""),
                        strict=False)])
                ctx.to_csv(out_dir / "dimming_simbad.csv", index=False)
        except Exception as exc:
            print(f"[dimming] SIMBAD vetting skipped: {exc!r}")

    try:
        from .figures import render_dimming
        render_dimming(summary, out_dir / "figures", windows=windows)
    except Exception as exc:
        print(f"[dimming] figures skipped: {exc!r}")

    print("[dimming] summary:", json.dumps({k_: summary[k_] for k_ in
          ("n_searched", "n_candidates", "n_resists_mundane")}))
    print("[dimming] occurrence limit:", json.dumps(summary["occurrence_limit"]))
    return summary


__all__ = ["dimming_run"]
