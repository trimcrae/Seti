"""The ANTIPHASE verdict ladder, the natural classes and the blend model.  Pure.

A coupled event (``coupling.assess_coupling`` -> ``COUPLED``) is not yet
anything.  Every one of the following is a known way for a star to fade in the
optical while its mid-infrared brightens, and each is tested in this order;
the first that applies names the event.

| class | why it couples | test here |
|---|---|---|
| ``YSO_DIPPER`` | accretion-disk clumps occult a young star; the disk's own warm dust varies with it | star-forming-region box, a pre-existing AllWISE excess (W1-W2 > 0.15 or W2-W3 > 0.5), or ``young`` flag |
| ``RCRB_LIKE`` | a carbon cloud condenses on our line of sight; the shell re-radiates | host is a giant/supergiant (M_G < 2.5) *and* the fade is deep (> 1 mag) |
| ``LPV`` | pulsation: optical and IR vary with a phase offset | periodogram of the optical: P in 60 d .. baseline/2.5, significant, amplitude > 0.05 mag, or host is red giant |
| ``EB_DUSTY_DISK`` | a companion's disk eclipses the primary (EE Cep, KH 15D) | a significant short period (< 60 d) or repeated eclipse |
| ``NATURAL_CHROMATIC`` | ordinary grains redden (or scatter-dominated blueing) | ``chroma`` in REDDENING_* / BLUEING |
| ``STELLAR_IR`` | the IR gain is a photosphere (companion, blend), not grains | best-fit T > 1800 K |
| ``GREY_IR_DEFICIT`` | a line-of-sight-only occulter (ASASSN-24fw's circumsecondary disk) | ratio range below 1/3 |
| ``GREY_IR_SURPLUS`` | the IR is not the absorbed optical re-emitted | ratio range above 3 |
| ``LAGGED`` | a collision afterglow whose debris transits later (ASASSN-21qj) | best lag != 0 and clearly better than lag 0 |
| ``BLEND_PREDICTED`` | a Gaia neighbour moving through the WISE beam | the neighbour model predicts >= half the IR gain |
| ``COUPLED_INCOMPLETE`` | none of the above, but achromaticity or the energy budget could not be measured (one optical band, no F_bol) | --- |
| ``ANTIPHASE_CANDIDATE`` | none of the above, grey and balanced measured | --- |

Everything not coupled keeps its coupling label.  A class that could not be
*tested* (no neighbour table, one optical band) is listed in ``untested``;
an untested check is never a pass.
"""

from __future__ import annotations

import numpy as np

DEFAULT_CLASSIFY: dict = {
    "yso_w1w2_min": 0.15,
    "yso_w2w3_min": 0.5,
    "giant_mg_max": 2.5,
    "rcrb_depth_min_mag": 1.0,
    "lpv_pmin_d": 60.0,
    "lpv_amp_min_mag": 0.05,
    "periodic_fap_max": 1e-6,
    "eb_pmax_d": 60.0,
    "lag_corr_margin": 0.2,       # best-lag correlation must beat lag 0 by this
    "blend_frac_kill": 0.5,       # neighbours explain >= this of the IR gain
    "wise_fwhm_arcsec": {"W1": 6.08, "W2": 6.84},
    "gw1_intercept": 0.30,        # G - W1 ~ a + b (BP-RP) for a dwarf neighbour
    "gw1_slope": 1.28,
    "gw1_bright_margin_mag": 0.75,  # neighbour assumed this much brighter in W1 (conservative)
    "star_forming_boxes": [],
}

NATURAL_LABELS = ("YSO_DIPPER", "RCRB_LIKE", "LPV", "EB_DUSTY_DISK", "NATURAL_CHROMATIC",
                  "STELLAR_IR", "GREY_IR_DEFICIT", "GREY_IR_SURPLUS", "LAGGED",
                  "BLEND_PREDICTED")


def in_box(ra: float, dec: float, boxes) -> str:
    for b in boxes or []:
        r0, r1 = b["ra"]
        d0, d1 = b["dec"]
        if r0 <= ra <= r1 and d0 <= dec <= d1:
            return str(b.get("name", "box"))
    return ""


def absolute_g(g: float, parallax_mas: float) -> float:
    try:
        g, p = float(g), float(parallax_mas)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(g) and np.isfinite(p) and p > 0):
        return float("nan")
    return g + 5.0 * np.log10(p) - 10.0


def periodogram_features(t_mjd, mag, err, pmin_d: float = 0.5, pmax_d: float | None = None) -> dict:
    """Lomb-Scargle of one optical band: best period, FAP, semi-amplitude."""
    t = np.asarray(t_mjd, float)
    m = np.asarray(mag, float)
    e = np.asarray(err, float)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[ok], m[ok], e[ok]
    out = {"status": "OK", "n": int(t.size), "period_d": float("nan"), "fap": float("nan"),
           "amp_mag": float("nan"), "baseline_d": float(np.ptp(t)) if t.size else 0.0}
    if t.size < 30:
        out["status"] = "INSUFFICIENT"
        return out
    from astropy.timeseries import LombScargle

    base = float(np.ptp(t))
    pmax = float(pmax_d) if pmax_d else base / 2.5
    if pmax <= pmin_d:
        out["status"] = "SHORT_BASELINE"
        return out
    ls = LombScargle(t, m, e)
    fmin, fmax = 1.0 / pmax, 1.0 / pmin_d
    n_f = int(min(200_000, max(2000, 5 * base * (fmax - fmin))))
    freq = np.linspace(fmin, fmax, n_f)
    pw = ls.power(freq)
    k = int(np.argmax(pw))
    f = float(freq[k])
    try:
        fap = float(ls.false_alarm_probability(pw[k], minimum_frequency=fmin,
                                               maximum_frequency=fmax))
    except Exception:                                   # noqa: BLE001
        fap = float("nan")
    model = ls.model(t, f)
    amp = float(0.5 * (np.percentile(model, 97.5) - np.percentile(model, 2.5)))
    # A single long fade also gives a long-period peak with a tiny FAP.  A
    # periodic variable repeats: count the cycles whose own peak-to-peak
    # reaches the model's.  One active cycle is an event, not a period.
    p = 1.0 / f
    cyc = np.floor((t - t.min()) / p).astype(int)
    active = 0
    n_cov = 0
    for q in np.unique(cyc):
        v = m[cyc == q]
        if v.size < 5:
            continue
        n_cov += 1
        if np.percentile(v, 95) - np.percentile(v, 5) >= 1.0 * amp:
            active += 1
    out.update({"period_d": p, "fap": fap, "power": float(pw[k]), "amp_mag": amp,
                "n_cycles_covered": int(n_cov), "n_cycles_active": int(active)})
    return out


def natural_flags(meta: dict, coupling: dict, period: dict | None, conf: dict | None = None) -> dict:
    """Which natural classes the star's own properties point to (tested / untested)."""
    c = {**DEFAULT_CLASSIFY, **(conf or {})}
    flags: list[str] = []
    untested: list[str] = []
    ra, dec = float(meta.get("ra", np.nan)), float(meta.get("dec", np.nan))
    box = in_box(ra, dec, c["star_forming_boxes"]) if np.isfinite(ra) else ""
    w1, w2 = meta.get("w1mpro"), meta.get("w2mpro")
    w3 = meta.get("w3mpro")
    try:
        w12 = float(w1) - float(w2)
    except (TypeError, ValueError):
        w12 = float("nan")
    try:
        w23 = float(w2) - float(w3)
    except (TypeError, ValueError):
        w23 = float("nan")
    if box or (np.isfinite(w12) and w12 > float(c["yso_w1w2_min"])) or \
            (np.isfinite(w23) and w23 > float(c["yso_w2w3_min"])) or bool(meta.get("young")):
        flags.append("YSO_DIPPER")
    mg = absolute_g(meta.get("phot_g_mean_mag"), meta.get("parallax"))
    depth = float(coupling.get("depth_mag", np.nan))
    if np.isfinite(mg) and mg < float(c["giant_mg_max"]) and np.isfinite(depth) and \
            depth > float(c["rcrb_depth_min_mag"]):
        flags.append("RCRB_LIKE")
    elif not np.isfinite(mg):
        untested.append("host_luminosity")
    if period and period.get("status") == "OK":
        p, fap, amp = period.get("period_d"), period.get("fap"), period.get("amp_mag")
        sig = np.isfinite(fap) and fap < float(c["periodic_fap_max"]) and \
            int(period.get("n_cycles_active", 0)) >= 3 and \
            int(period.get("n_cycles_active", 0)) >= 0.6 * int(period.get("n_cycles_covered", 1))
        if sig and np.isfinite(p):
            if p >= float(c["lpv_pmin_d"]) and amp >= float(c["lpv_amp_min_mag"]):
                flags.append("LPV")
            elif p < float(c["eb_pmax_d"]) and amp >= 0.01:
                flags.append("EB_DUSTY_DISK")
    else:
        untested.append("periodicity")
    if np.isfinite(mg) and mg < float(c["giant_mg_max"]) and \
            float(meta.get("bp_rp", np.nan) or np.nan) > 1.8 and "LPV" not in flags:
        flags.append("LPV")                 # a red giant host: LPV until shown otherwise
    return {"flags": flags, "untested": untested, "sf_box": box, "abs_g": mg,
            "w1_w2_allwise": w12, "w2_w3_allwise": w23}


def blend_prediction(target: dict, neighbours: list[dict], t_yr, conf: dict | None = None,
                     gaia_epoch: float = 2016.0) -> dict:
    """Predicted W1/W2 change of the target's photometry from neighbours' motion.

    Each Gaia neighbour's W1 is estimated from its G and BP-RP with a dwarf
    colour relation and then made ``gw1_bright_margin_mag`` brighter (the
    conservative end: a red neighbour carries more W1 than a dwarf relation
    says).  Its weight in the target's profile-fit flux is the Gaussian PSF at
    the separation of the day, ``exp(-s^2 / 2 sigma^2)``.  Positions are
    propagated with both stars' proper motions.  Returns the predicted
    ``dmag`` per band per epoch relative to the median epoch (negative =
    the blend makes the target look brighter).
    """
    c = {**DEFAULT_CLASSIFY, **(conf or {})}
    t = np.asarray(t_yr, float)
    out = {"n_neighbours": len(neighbours or []), "dmag": {}, "max_brightening": {}}
    tw1 = target.get("w1mpro")
    tw2 = target.get("w2mpro")
    if tw1 is None or not np.isfinite(float(tw1)):
        out["status"] = "NO_TARGET_W1"
        return out
    base = {"W1": float(tw1), "W2": float(tw2) if tw2 is not None and np.isfinite(float(tw2))
            else float(tw1)}

    def pos(r, ep):
        dt = ep - gaia_epoch
        pmra = float(r.get("pmra") or 0.0) if np.isfinite(float(r.get("pmra") or 0.0)) else 0.0
        pmde = float(r.get("pmdec") or 0.0) if np.isfinite(float(r.get("pmdec") or 0.0)) else 0.0
        dec = float(r["dec"]) + pmde * dt / 3.6e6
        ra = float(r["ra"]) + pmra * dt / 3.6e6 / np.cos(np.radians(float(r["dec"])))
        return ra, dec

    for b in ("W1", "W2"):
        sig = float(c["wise_fwhm_arcsec"][b]) / 2.3548
        ft = 10.0 ** (-0.4 * base[b])
        tot = np.full(t.size, ft)
        for n in neighbours or []:
            g = n.get("phot_g_mean_mag")
            if g is None or not np.isfinite(float(g)):
                continue
            bprp = n.get("bp_rp")
            bprp = float(bprp) if bprp is not None and np.isfinite(float(bprp)) else 1.5
            w1n = float(g) - (float(c["gw1_intercept"]) + float(c["gw1_slope"]) * bprp) \
                - float(c["gw1_bright_margin_mag"])
            fn = 10.0 ** (-0.4 * w1n)
            for i, ep in enumerate(t):
                ra_t, de_t = pos(target, ep)
                ra_n, de_n = pos(n, ep)
                s = np.hypot((ra_n - ra_t) * np.cos(np.radians(de_t)), de_n - de_t) * 3600.0
                tot[i] += fn * np.exp(-s ** 2 / (2.0 * sig ** 2))
        m = -2.5 * np.log10(tot)
        d = m - np.median(m)
        out["dmag"][b] = [float(x) for x in d]
        out["max_brightening"][b] = float(-np.min(d)) if d.size else 0.0
    out["status"] = "OK"
    return out


def blend_explains(pred: dict, faded_t_yr, t_yr, observed_dmag: dict, conf: dict | None = None) -> dict:
    """Does the neighbour model explain >= ``blend_frac_kill`` of the IR gain at F?"""
    c = {**DEFAULT_CLASSIFY, **(conf or {})}
    if pred.get("status") != "OK":
        return {"status": "UNTESTED", "explains": False}
    t = np.asarray(t_yr, float)
    sel = np.isin(np.round(t, 3), np.round(np.asarray(faded_t_yr, float), 3))
    fr = {}
    for b in ("W1", "W2"):
        d = np.asarray(pred["dmag"].get(b, []), float)
        obs = float(observed_dmag.get(b, np.nan))
        if d.size != t.size or not sel.any() or not np.isfinite(obs) or obs >= 0:
            fr[b] = float("nan")
            continue
        fr[b] = float(np.mean(d[sel]) / obs)      # both negative for a brightening
    ok = [v for v in fr.values() if np.isfinite(v)]
    explains = bool(ok) and max(ok) >= float(c["blend_frac_kill"])
    return {"status": "OK", "fraction_explained": fr, "explains": explains}


def final_verdict(coupling: dict, energy: dict | None, natural: dict, blend: dict | None = None,
                  conf: dict | None = None) -> dict:
    """The ladder.  Returns ``{"verdict", "natural_class", "untested", "reasons"}``."""
    c = {**DEFAULT_CLASSIFY, **(conf or {})}
    lab = str(coupling.get("label"))
    untested = list(natural.get("untested", []))
    if lab not in ("COUPLED", "LAGGED_COUPLING"):
        return {"verdict": lab, "natural_class": None, "untested": untested, "reasons": []}
    reasons: list[str] = []
    # luminosity/periodicity classes first: an R CrB star or a Mira also
    # carries the pre-existing IR excess that the YSO test keys on
    for f in ("RCRB_LIKE", "LPV", "EB_DUSTY_DISK", "YSO_DIPPER"):
        if f in natural.get("flags", []):
            reasons.append(f)
    chroma = str(coupling.get("chroma"))
    if chroma.startswith("REDDENING") or chroma == "BLUEING":
        reasons.append("NATURAL_CHROMATIC")
    elif chroma in ("UNTESTED", "AMBIGUOUS"):
        untested.append(f"achromaticity:{chroma}")
    ev = (energy or {}).get("verdict")
    if ev == "STELLAR_TEMPERATURE":
        reasons.append("STELLAR_IR")
    elif ev == "IR_DEFICIT":
        reasons.append("GREY_IR_DEFICIT")
    elif ev == "IR_SURPLUS":
        reasons.append("GREY_IR_SURPLUS")
    elif ev != "BALANCED":
        untested.append("energy_budget")
    lag, cb, c0 = coupling.get("best_lag", 0), coupling.get("corr_best"), coupling.get("corr_lag0")
    if lab == "LAGGED_COUPLING" or (
            lag and np.isfinite(cb or np.nan) and np.isfinite(c0 or np.nan)
            and abs(int(lag)) > 1 and cb - c0 >= float(c["lag_corr_margin"])):
        reasons.append("LAGGED")
    if blend is None or blend.get("status") != "OK":
        untested.append("neighbour_blend")
    elif blend.get("explains"):
        reasons.append("BLEND_PREDICTED")
    if reasons:
        return {"verdict": "NATURAL", "natural_class": reasons[0], "untested": untested,
                "reasons": reasons}
    if any(u.startswith("achromaticity") for u in untested) or "energy_budget" in untested:
        # grey-and-balanced is the claim; an event where either could not be
        # measured is coupled but not a candidate (one optical band, no F_bol)
        return {"verdict": "COUPLED_INCOMPLETE", "natural_class": None, "untested": untested,
                "reasons": []}
    return {"verdict": "ANTIPHASE_CANDIDATE", "natural_class": None, "untested": untested,
            "reasons": []}


__all__ = ["DEFAULT_CLASSIFY", "NATURAL_LABELS", "absolute_g", "blend_explains",
           "blend_prediction", "final_verdict", "in_box", "natural_flags", "periodogram_features"]
