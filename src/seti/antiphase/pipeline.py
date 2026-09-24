"""One star through the whole ANTIPHASE ladder.  Pure (no network).

``make_pack`` puts a star's NEOWISE epochs and its raw optical points on one
epoch grid; ``evaluate_pack`` runs coupling -> chromaticity -> energy budget
-> natural classes -> verdict and returns one flat record.  The survey
shards, the positive controls, the null and the injections all call these two
functions, so every one of them is measured by the same ladder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .classify import DEFAULT_CLASSIFY, final_verdict, natural_flags, periodogram_features
from .coupling import DEFAULT_COUPLING, assess_coupling, bin_at_epochs, year_to_mjd
from .energy import energy_budget, star_teff
from .null import align_to


def ir_arrays(ep: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Long epoch table (``band, t_yr, mag, err``) -> W1 epoch grid + aligned W1/W2."""
    if ep is None or not len(ep):
        return np.array([]), {}
    d = ep.copy()
    d["band"] = d["band"].astype(str)
    w1 = d[d["band"] == "W1"].sort_values("t_yr")
    w2 = d[d["band"] == "W2"].sort_values("t_yr")
    if not len(w1) or not len(w2):
        return np.array([]), {}
    t = w1["t_yr"].to_numpy(float)
    ir = {"W1": (w1["mag"].to_numpy(float), w1["err"].to_numpy(float))}
    ir.update(align_to(t, w2["t_yr"].to_numpy(float),
                       {"W2": (w2["mag"].to_numpy(float), w2["err"].to_numpy(float))},
                       tol_yr=0.1))
    return t, ir


def make_pack(ep: pd.DataFrame, optical: dict, meta: dict, conf: dict | None = None) -> dict:
    """``optical`` = ``{band: {"mjd", "mag", "err"}}`` raw points."""
    t, ir = ir_arrays(ep)
    opt = {}
    raw = {}
    for b, v in (optical or {}).items():
        if not isinstance(v, dict) or "mjd" not in v:
            continue
        bb = bin_at_epochs(v["mjd"], v["mag"], v.get("err"), year_to_mjd(t), conf)
        if np.isfinite(bb["mag"]).any():
            opt[b] = (bb["mag"], bb["err"])
        raw[b] = v
    return {"t": t, "opt": opt, "ir": ir, "meta": dict(meta or {}), "raw": raw}


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def evaluate_pack(pack: dict, conf: dict | None = None, *, periodogram: bool = True,
                  blend: dict | None = None, period: dict | None = None) -> dict:
    """The full ladder on one pack.  ``conf`` has ``coupling``/``energy``/``classify``."""
    conf = conf or {}
    cc = {**DEFAULT_COUPLING, **(conf.get("coupling") or {})}
    kc = {**DEFAULT_CLASSIFY, **(conf.get("classify") or {})}
    meta = pack.get("meta", {})
    r = assess_coupling(pack["t"], pack["opt"], pack["ir"], cc)
    cd = r.to_dict()
    en = None
    if r.n_faded:
        teff, tsrc = star_teff(meta.get("teff_gspphot"), meta.get("bp_rp"))
        en = energy_budget(_f(meta.get("phot_g_mean_mag")), teff, r.frac_lost, r.frac_lost_err,
                           r.ir_ref, r.ir_dmag, r.ir_dmag_err, conf.get("energy"))
        en["teff"], en["teff_source"] = teff, tsrc
    per = period
    if per is None and periodogram and r.n_faded:
        raw = pack.get("raw") or {}
        band = max(raw, key=lambda b: len(raw[b].get("mjd", []))) if raw else None
        if band:
            v = raw[band]
            per = periodogram_features(v["mjd"], v["mag"], v.get("err"))
            per["band"] = band
    nat = natural_flags(meta, cd, per, kc)
    fv = final_verdict(cd, en, nat, blend, kc)
    rec = {"verdict": fv["verdict"], "natural_class": fv["natural_class"],
           "reasons": ";".join(fv["reasons"]), "untested": ";".join(sorted(set(fv["untested"]))),
           "coupling_label": r.label, "n_matched": r.n_matched, "n_faded": r.n_faded,
           "optical_bands": ",".join(r.optical_bands),
           "faded_t_yr": ";".join(f"{x:.2f}" for x in r.faded_t_yr),
           "depth_mag": r.depth_mag, "depth_err": r.depth_err, "frac_lost": r.frac_lost,
           "chroma": r.chroma, "k_colour": r.k_colour, "k_colour_err": r.k_colour_err,
           "d_gr_deepest": r.d_gr_deepest, "d_gr_deepest_err": r.d_gr_deepest_err,
           "best_lag": r.best_lag, "corr_lag0": r.corr_lag0, "corr_best": r.corr_best,
           "score": r.score, "ir_two_band": r.ir_two_band, "lag_ir_sigma": r.lag_ir_sigma,
           "ir_rise_max_sigma": r.ir_rise_max_sigma,
           "natural_flags": ";".join(nat["flags"]), "sf_box": nat["sf_box"],
           "abs_g": nat["abs_g"]}
    for b in ("W1", "W2"):
        rec[f"{b.lower()}_dmag"] = r.ir_dmag.get(b, np.nan)
        rec[f"{b.lower()}_dmag_err"] = r.ir_dmag_err.get(b, np.nan)
        rec[f"{b.lower()}_sigma"] = r.ir_sigma.get(b, np.nan)
        rec[f"{b.lower()}_slope_sigma"] = r.slope_sigma.get(b, np.nan)
        rec[f"{b.lower()}_contrast"] = r.ir_contrast.get(b, np.nan)
    for b, v in r.depth_by_band.items():
        rec[f"depth_{b}"] = v[0]
        rec[f"depth_{b}_err"] = v[1]
    if en:
        bb = en.get("blackbody", {})
        rec.update({"energy_verdict": en.get("verdict"), "ratio_best": en.get("ratio_best"),
                    "ratio_lo": en.get("ratio_lo"), "ratio_hi": en.get("ratio_hi"),
                    "t_bb_k": bb.get("t_best_k"), "t_bb_lo_k": bb.get("t_lo_k"),
                    "t_bb_hi_k": bb.get("t_hi_k"), "f_bol_w_m2": en.get("f_bol_w_m2"),
                    "teff": en.get("teff")})
    if per:
        rec.update({"period_d": per.get("period_d"), "period_fap": per.get("fap"),
                    "period_amp_mag": per.get("amp_mag"),
                    "period_cycles_active": per.get("n_cycles_active"),
                    "period_cycles_covered": per.get("n_cycles_covered")})
    return rec


def evaluate_for_injection(pack: dict, conf: dict | None = None) -> dict:
    """What the injection test scores: coupled, grey, balanced (no periodogram)."""
    rec = evaluate_pack(pack, conf, periodogram=False)
    return {"coupled": rec["coupling_label"] == "COUPLED", "grey": rec["chroma"] == "GREY",
            "balanced": rec.get("energy_verdict") == "BALANCED"}


__all__ = ["evaluate_for_injection", "evaluate_pack", "ir_arrays", "make_pack"]
