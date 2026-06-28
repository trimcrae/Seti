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


def _is_candidate(stat: dict, depth_min: float, n_dips_min: int,
                  asym_min: float, period_power_max: float) -> bool:
    """A dimming candidate: deep + several dips + asymmetric + NOT strongly periodic.

    The periodicity gate is the key mundane-rejection: strictly periodic, repeating
    dips are an eclipsing binary, not the aperiodic dimming of KIC 8462852.
    """
    return (stat.get("max_depth", 0.0) >= depth_min
            and stat.get("n_dips", 0) >= n_dips_min
            and stat.get("asymmetry", 0.0) >= asym_min
            and stat.get("period_power", 1.0) <= period_power_max)


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
        d.update({"source_id": meta.get("source_id"), "ra": meta.get("ra"),
                  "dec": meta.get("dec"),
                  "g_mag": g_mag, "bp_rp": bp_rp, "parallax": plx,
                  "hr_class": hr,
                  "resists_mundane": bool(is_cand and resists_mundane(
                      hr, d.get("period_power", 1.0), period_power_max)),
                  "is_candidate": is_cand})
        # Keep the light curve only for the strongest dippers (committed JSON).
        d["_mjd"], d["_mag"] = np.asarray(mjd, float), np.asarray(mag, float)
        rows.append(d)

    if lightcurves is not None:
        for lc in lightcurves:
            _score_one(lc, lc["mjd"], lc["mag"], lc.get("magerr"))
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

    candidates = [r for r in rows if r["is_candidate"]]
    # Rank main-sequence aperiodic dippers (those that resist the mundane
    # giant/YSO/eclipse explanations) above the rest, then by dimming score.
    candidates.sort(key=lambda r: (r.get("resists_mundane", False),
                                   r.get("score", 0.0)), reverse=True)
    n_resists = sum(1 for r in candidates if r.get("resists_mundane"))

    out_dir = cfg.root / "results" / "dimming"
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

    # Flat candidate table (drop the bulky light-curve arrays).
    if rows:
        flat = pd.DataFrame([{k: v for k, v in r.items()
                              if not k.startswith("_")} for r in rows])
        flat = flat.sort_values("score", ascending=False)
        flat.to_csv(out_dir / "dimming_scored.csv", index=False)
        if candidates:
            flat[flat["is_candidate"]].to_csv(
                out_dir / "dimming_candidates.csv", index=False)

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
        "selection": {"depth_min": depth_min, "n_dips_min": n_dips_min,
                      "asym_min": asym_min, "period_power_max": period_power_max},
        "top_candidates": [
            {"source_id": r.get("source_id"), "ra": r.get("ra"), "dec": r.get("dec"),
             "score": r.get("score"), "max_depth": r.get("max_depth"),
             "n_dips": r.get("n_dips"), "asymmetry": r.get("asymmetry"),
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
