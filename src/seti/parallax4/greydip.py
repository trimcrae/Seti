"""The photometric half: grey (achromatic) dips and brightenings in Gaia
per-transit G / BP / RP photometry.  Pure functions, no I/O.

One source at a time:

1. **Baseline.**  Per band, the median flux of the usable transits and a
   robust scatter (1.4826 MAD of the fractional residual).  The part of that
   scatter not explained by the quoted errors is the star's own variability,
   ``s_int``; each transit's significance is ``delta / sqrt(err^2 + s_int^2)``,
   so an ordinary variable star has to beat its own variability to count.
2. **Episodes.**  Transits deviating >= ``k_member`` sigma in G with the same
   sign are chained into episodes when consecutive deviants lie within
   ``episode_gap_d`` (Gaia's sampling: the two FoVs 106.5 min apart, then the
   next 6-h spin).  An episode qualifies when its peak reaches ``k_peak``
   sigma and ``min_frac_depth`` in G.
3. **Three bands.**  BP and RP are combined over the episode's transits; both
   must deviate >= ``k_colour`` sigma with G's sign, else the episode is
   ``G_ONLY`` (single-band anomalies are artefacts until a second band agrees).
4. **Grey.**  The G, BP, RP fractional depths must share one value (chi^2,
   2 dof, measurement errors plus ``calib_floor``) AND the BP/RP depth ratio
   must be within ``ratio_tol`` of 1 with an error <= ``ratio_err_max``.
   Small-grain dust gives BP/RP ~ A_BP/A_RP ~ 1.7; spots, pulsation and flares
   are chromatic too.  A ratio too uncertain to tell 1 from 1.7 is
   ``GREY_UNCONSTRAINED``, never ``GREY``.
5. **Coherence.**  A one-transit episode whose FoV partner (<= ``partner_d``)
   is normal is ``SINGLE_CONTRADICTED`` (shorter than ~2 h, or instrumental);
   a one-transit episode with no partner is ``SINGLE_UNCHECKED``; an episode
   of >= 2 same-sign deviant transits is ``MULTI``.
6. **Periodicity.**  With >= ``min_episodes_periodic`` dip episodes, a period
   search asks whether one period folds every dip episode into one or two
   narrow phase arcs (primary and secondary eclipse) at a false-alarm
   probability below ``period_fap_max``, with the transits inside the arcs
   mostly dips (an eclipsing binary).  Fewer episodes is
   ``PERIOD_UNTESTABLE`` -- stated, not treated as aperiodic.

Grey eclipses are common (twin-temperature eclipsing binaries); the detector
is *not* an EB discriminator on its own and never claims to be.  The grey
test removes stellar-physics and dust confounders; periodicity plus Gaia's own
classification removes EBs; FoV coherence plus cross-source epoch clustering
removes instrument; the neighbour-flux match (vet stage) removes mis-assigned
transits; and DR4's photocentre removes blends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class GreyConfig:
    n_min: int = 20                 # usable G transits per source
    k_member: float = 3.0           # a transit joins an episode at this |z_G|
    k_peak: float = 5.0             # an episode qualifies when one transit reaches this
    min_frac_depth: float = 0.01    # |delta_G| at the peak
    k_colour: float = 3.0           # BP and RP episode deviation, same sign as G
    calib_floor: float = 0.003      # fractional, added in quadrature for the grey test
    grey_p_min: float = 0.01        # chi^2 (2 dof) p-value for a common depth
    ratio_tol: float = 0.25         # |BP/RP depth ratio - 1|
    ratio_err_max: float = 0.20     # ratio must be measured at least this well
    dust_ratio: float = 1.70        # A_BP/A_RP for R_V = 3.1 small-grain dust
    episode_gap_d: float = 1.0
    partner_d: float = 0.10         # the other FoV, 106.5 min away
    normal_z: float = 2.0           # a partner below this |z| contradicts
    min_episodes_periodic: int = 3
    period_min_d: float = 0.2
    period_max_d: float = 600.0
    period_n: int = 250000         # df = 2e-5 /d: phase error <= 0.02 over 1000 d
    period_fap_max: float = 0.01    # chance of folding n unrelated episodes that tightly
    window_purity_min: float = 0.5  # dip transits / all transits inside the phase arc(s)
    max_band_err_frac: float = 0.5  # a band point with err/flux above this is unusable

    @classmethod
    def from_dict(cls, d: dict | None) -> GreyConfig:
        d = dict(d or {})
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


def _band(f, e, ok, cfg: GreyConfig):
    """Fractional residual, measurement sigma and total sigma for one band."""
    f = np.asarray(f, float)
    e = np.asarray(e, float)
    ok = ok & np.isfinite(f) & np.isfinite(e) & (f > 0) & (e > 0) & (e < cfg.max_band_err_frac * np.abs(f))
    if ok.sum() < 3:
        n = len(f)
        return (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), ok, np.nan, np.nan)
    med = float(np.median(f[ok]))
    d = f / med - 1.0
    em = e / med
    mad = MAD_TO_SIGMA * float(np.median(np.abs(d[ok] - np.median(d[ok]))))
    s_int = float(np.sqrt(max(0.0, mad ** 2 - float(np.median(em[ok] ** 2)))))
    sig = np.sqrt(em ** 2 + s_int ** 2)
    return d, em, sig, ok, med, s_int


def _wmean(x, s):
    w = 1.0 / s ** 2
    m = np.isfinite(x) & np.isfinite(w)
    if not m.any():
        return np.nan, np.nan
    return float(np.sum(w[m] * x[m]) / np.sum(w[m])), float(1.0 / np.sqrt(np.sum(w[m])))


def grey_test(dg, sg, db, sb, dr, sr, cfg: GreyConfig) -> dict:
    """Is (dg, db, dr) one achromatic fractional change?"""
    d = np.array([dg, db, dr], float)
    s = np.sqrt(np.array([sg, sb, sr], float) ** 2 + cfg.calib_floor ** 2)
    out = {"depth_common": np.nan, "chi2_grey": np.nan, "p_grey": np.nan,
           "ratio_bp_rp": np.nan, "ratio_err": np.nan, "grey_class": "NO_COLOUR",
           "ratio_vs_dust_sigma": np.nan}
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(s)):
        return out
    w = 1.0 / s ** 2
    m = float(np.sum(w * d) / np.sum(w))
    chi2 = float(np.sum(w * (d - m) ** 2))
    p = float(np.exp(-0.5 * chi2))                       # chi^2 survival, 2 dof
    ratio = db / dr if dr != 0 else np.nan
    rerr = abs(ratio) * np.sqrt((s[1] / db) ** 2 + (s[2] / dr) ** 2) if db != 0 and dr != 0 else np.inf
    out.update(depth_common=m, chi2_grey=chi2, p_grey=p, ratio_bp_rp=float(ratio),
               ratio_err=float(rerr))
    if np.isfinite(ratio) and np.isfinite(rerr) and rerr > 0:
        out["ratio_vs_dust_sigma"] = float((cfg.dust_ratio - ratio) / rerr)
    if p >= cfg.grey_p_min and np.isfinite(ratio) and abs(ratio - 1.0) <= cfg.ratio_tol:
        out["grey_class"] = "GREY" if rerr <= cfg.ratio_err_max else "GREY_UNCONSTRAINED"
    elif p >= cfg.grey_p_min and rerr > cfg.ratio_err_max:
        out["grey_class"] = "GREY_UNCONSTRAINED"
    else:
        if np.isfinite(ratio) and ratio > 1.0:
            out["grey_class"] = "CHROMATIC_BLUE"      # deeper in BP: dust, spots, flares
        else:
            out["grey_class"] = "CHROMATIC_RED"
    return out


def _episodes(t, z, ok, sign, cfg: GreyConfig) -> list[np.ndarray]:
    """Chains of same-sign deviant transits (indices into t, time-ordered)."""
    idx = np.nonzero(ok & (sign * z >= cfg.k_member))[0]
    if not len(idx):
        return []
    idx = idx[np.argsort(t[idx])]
    groups, cur = [], [idx[0]]
    for a, b in zip(idx[:-1], idx[1:], strict=False):
        # a normal usable transit between two deviants breaks the chain
        between = ok & (t > t[a]) & (t < t[b]) & (sign * z < cfg.k_member)
        if (t[b] - t[a]) <= cfg.episode_gap_d and not between.any():
            cur.append(b)
        else:
            groups.append(np.array(cur))
            cur = [b]
    groups.append(np.array(cur))
    return groups


def period_check(t_ok: np.ndarray, is_dip: np.ndarray, ep_times: np.ndarray,
                 cfg: GreyConfig) -> dict:
    """Does one period fold every dip episode into one or two narrow phase
    arcs (primary, or primary + secondary eclipse), with the transits inside
    those arcs mostly dips?

    For each trial frequency the minimal arc(s) covering the n episode phases
    are 1 - g1 (one arc) and 1 - g1 - g2 (two arcs), g1 >= g2 the largest
    phase gaps.  The chance that n unrelated episodes fold that tightly at
    some trial frequency is approximately
    ``K * n * W1^(n-1)`` or ``K * n^2 / 2 * W2^(n-2)``, K = T * (f_max - f_min)
    independent frequencies.  PERIODIC needs that false-alarm probability
    below ``period_fap_max`` AND an in-arc purity >= ``window_purity_min``;
    so three episodes can only be called periodic if they fold very tightly,
    and a real eclipsing binary with many episodes is caught easily.
    """
    n_ep = len(ep_times)
    if n_ep < cfg.min_episodes_periodic:
        return {"period_class": "PERIOD_UNTESTABLE", "period_d": np.nan, "n_dip_episodes": n_ep,
                "period_fap": np.nan}
    tspan = float(np.ptp(t_ok)) if len(t_ok) else 0.0
    pmax = min(cfg.period_max_d, max(tspan, cfg.period_min_d * 2))
    fmin, fmax = 1.0 / pmax, 1.0 / cfg.period_min_d
    freqs = np.linspace(fmin, fmax, cfg.period_n)
    k_indep = max(1.0, tspan * (fmax - fmin))
    def fold(times, fr):
        n = len(times)
        ph_ = (times[None, :] * fr[:, None]) % 1.0
        ph_.sort(axis=1)
        gaps_ = np.diff(np.concatenate([ph_, ph_[:, :1] + 1.0], axis=1), axis=1)
        gs_ = -np.sort(-gaps_, axis=1)
        w1 = np.clip(1.0 - gs_[:, 0], 1e-12, 1.0)
        with np.errstate(divide="ignore"):
            l1 = np.log(k_indep * n) + (n - 1) * np.log(w1)
            if n >= 3:
                w2 = np.clip(1.0 - gs_[:, 0] - gs_[:, 1], 1e-12, 1.0)
                l2 = np.log(k_indep * n * n / 2.0) + (n - 2) * np.log(w2)
            else:
                l2 = np.full_like(l1, np.inf)
        return ph_, gaps_, l1, l2

    # Two passes keep the cost flat in the number of episodes (catalogue
    # scale: ~10^6 eclipsing binaries reach this function): a coarse grid on
    # <= 6 episodes spread through the baseline, then every episode on the
    # fine frequencies around the best coarse ones.  A fold tight for all
    # episodes is tight for any subset, so it survives the first pass.
    # A coarse grid (4x the full-resolution step) on the subset, then the 8
    # fine frequencies around each of the 500 best coarse ones.
    ep_sorted = np.sort(ep_times)
    sub = ep_sorted[np.linspace(0, n_ep - 1, min(n_ep, 6)).round().astype(int)]
    step = (fmax - fmin) / max(cfg.period_n - 1, 1)
    coarse = freqs[::4]
    _, _, s1, s2 = fold(sub, coarse)
    best_c = coarse[np.argsort(np.minimum(s1, s2))[:500]]
    fine = (best_c[:, None] + step * np.arange(-4, 5)[None, :]).ravel()
    freqs = np.unique(fine[(fine >= fmin) & (fine <= fmax)])
    ph, gaps, lf1, lf2 = fold(ep_sorted, freqs)
    lfap = np.minimum(lf1, lf2)
    order = np.argsort(lfap)[:300]
    best = None
    for k in order:
        if lfap[k] > np.log(cfg.period_fap_max):
            break
        f = freqs[k]
        two = lf2[k] < lf1[k]
        phase_all = (t_ok * f) % 1.0
        # arcs = the complement of the one or two largest gaps, padded
        pk = ph[k]
        gk = gaps[k]
        big = np.argsort(-gk)[: (2 if two else 1)]
        inwin = np.ones(len(t_ok), bool)
        for b in big:
            g0 = pk[b] + 0.01
            g1 = pk[(b + 1) % n_ep] + (1.0 if b == n_ep - 1 else 0.0) - 0.01
            if g1 > g0:
                x = phase_all.copy()
                x[x < pk[b]] += 1.0
                inwin &= ~((x > g0) & (x < g1))
        n_in = int(inwin.sum())
        purity = float((inwin & is_dip).sum()) / n_in if n_in else 0.0
        if purity >= cfg.window_purity_min:
            best = (1.0 / f, purity, float(np.exp(lfap[k])), bool(two))
            break
    if best is None:
        return {"period_class": "APERIODIC", "period_d": np.nan, "n_dip_episodes": n_ep,
                "period_fap": float(min(1.0, np.exp(lfap[order[0]]))) if len(order) else np.nan}
    return {"period_class": "PERIODIC", "period_d": float(best[0]), "window_purity": best[1],
            "period_fap": best[2], "two_arcs": best[3], "n_dip_episodes": n_ep}


def detect_source(lc: pd.DataFrame, cfg: GreyConfig | None = None) -> tuple[list[dict], dict]:
    """All qualifying episodes of one source, plus a per-source summary.

    ``lc`` is the normalised photometry frame (``epochs.PHOT_COLUMNS``) for a
    single source_id."""
    cfg = cfg or GreyConfig()
    sid = int(lc["source_id"].iloc[0]) if len(lc) else -1
    lc = lc.sort_values("t")
    t = lc["t"].to_numpy(dtype=float, copy=True)
    okt = np.isfinite(t)
    dg, emg, sgt, okg, medg, sintg = _band(lc["f_g"], lc["e_g"], okt & ~lc["bad_g"].to_numpy(), cfg)
    summary = {"source_id": sid, "n_transits": int(len(lc)), "n_ok_g": int(okg.sum()),
               "med_flux_g": medg, "s_int_g": sintg, "status": "ok"}
    if okg.sum() < cfg.n_min:
        summary["status"] = "too_few_transits"
        return [], summary
    db, emb, sbt, okb, medb, sintb = _band(lc["f_bp"], lc["e_bp"], okt & ~lc["bad_bp"].to_numpy(), cfg)
    dr, emr, srt, okr, medr, sintr = _band(lc["f_rp"], lc["e_rp"], okt & ~lc["bad_rp"].to_numpy(), cfg)
    summary.update(n_ok_bp=int(okb.sum()), n_ok_rp=int(okr.sum()), s_int_bp=sintb, s_int_rp=sintr,
                   mad_g=float(MAD_TO_SIGMA * np.median(np.abs(dg[okg] - np.median(dg[okg])))))
    zg = dg / sgt
    vrej = lc["vrej_g"].to_numpy() | lc["vrej_bp"].to_numpy() | lc["vrej_rp"].to_numpy()
    nobs = lc["n_obs_g"].to_numpy(dtype=float, copy=True)
    tid = lc["transit_id"].to_numpy()
    events: list[dict] = []
    dip_mask = np.zeros(len(t), bool)
    dip_ep_times = []
    for sign, kind in ((-1.0, "DIP"), (1.0, "BRIGHTENING")):
        for ep in _episodes(t, zg, okg, sign, cfg):
            peak = ep[np.argmax(sign * zg[ep])]
            if sign * zg[peak] < cfg.k_peak or abs(dg[peak]) < cfg.min_frac_depth:
                continue
            if sign < 0:
                dip_mask[ep] = True
                dip_ep_times.append(float(np.mean(t[ep])))
            mg, smg = _wmean(dg[ep], emg[ep])
            eb = ep[okb[ep]]
            er = ep[okr[ep]]
            mb, smb = _wmean(db[eb], emb[eb]) if len(eb) else (np.nan, np.nan)
            mr, smr = _wmean(dr[er], emr[er]) if len(er) else (np.nan, np.nan)
            zb_ep = mb / np.sqrt(smb ** 2 + sintb ** 2) if np.isfinite(mb) else np.nan
            zr_ep = mr / np.sqrt(smr ** 2 + sintr ** 2) if np.isfinite(mr) else np.nan
            colour_ok = (np.isfinite(zb_ep) and np.isfinite(zr_ep)
                         and sign * zb_ep >= cfg.k_colour and sign * zr_ep >= cfg.k_colour)
            if colour_ok:
                gt = grey_test(mg, smg, mb, smb, mr, smr, cfg)
            else:
                gt = grey_test(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, cfg)
                gt["grey_class"] = "G_ONLY" if (len(eb) and len(er)) else "NO_COLOUR"
            t0, t1 = float(t[ep].min()), float(t[ep].max())
            near = okg & (np.abs(t - t[peak]) <= cfg.partner_d)
            near[ep] = False
            n_contra = int((near & (np.abs(zg) < cfg.normal_z)).sum())
            if len(ep) >= 2:
                coh = "MULTI"
            elif n_contra:
                coh = "SINGLE_CONTRADICTED"
            else:
                coh = "SINGLE_UNCHECKED"
            events.append({
                "source_id": sid, "kind": kind, "t_peak": float(t[peak]), "t_start": t0, "t_end": t1,
                "transit_id_peak": int(tid[peak]), "n_transits_episode": int(len(ep)),
                "delta_g_peak": float(dg[peak]), "z_g_peak": float(zg[peak]),
                "delta_g": mg, "sigma_g": smg, "delta_bp": mb, "sigma_bp": smb,
                "delta_rp": mr, "sigma_rp": smr, "z_bp_episode": zb_ep, "z_rp_episode": zr_ep,
                "n_bp_episode": int(len(eb)), "n_rp_episode": int(len(er)),
                "coherence": coh, "n_partner_contradicting": n_contra,
                "any_variability_reject": bool(vrej[ep].any()),
                "min_n_obs_g": float(np.nanmin(nobs[ep])) if np.isfinite(nobs[ep]).any() else np.nan,
                **gt,
            })
    # The period search only matters where it could kill something: run it
    # when a GREY episode exists (it is the expensive step at catalogue scale).
    if any(e["grey_class"] == "GREY" for e in events):
        per = period_check(t[okg], dip_mask[okg], np.array(dip_ep_times), cfg)
    else:
        per = {"period_class": "NOT_RUN", "period_d": np.nan, "period_fap": np.nan}
    summary.update(n_dip_episodes=int(len(dip_ep_times)),
                   n_bright_episodes=int(sum(e["kind"] == "BRIGHTENING" for e in events)),
                   **{k: per.get(k) for k in ("period_class", "period_d", "period_fap")})
    # quietness outside the episodes (a deep grey dip on a quiet star is the
    # interesting configuration; a noisy star's episodes are cheap)
    out_ep = okg.copy()
    for e in events:
        out_ep &= ~((t >= e["t_start"] - 1e-6) & (t <= e["t_end"] + 1e-6))
    summary["rms_out_of_episode_g"] = float(np.std(dg[out_ep])) if out_ep.sum() > 3 else np.nan
    for e in events:
        e["period_class"] = per["period_class"]
        e["period_d"] = per.get("period_d", np.nan)
        e["period_fap"] = per.get("period_fap", np.nan)
        e["n_dip_episodes"] = int(len(dip_ep_times))
        e["rms_out_of_episode_g"] = summary["rms_out_of_episode_g"]
        e["s_int_g"] = sintg
        e["n_ok_g"] = int(okg.sum())
        e["med_flux_g"] = medg
    return events, summary


def tier(ev: dict) -> str:
    """The event's tier before the cross-source and vet vetoes.

    A  grey, colour-constrained, multi-transit, not periodic
    B  grey, colour-constrained, single transit not contradicted by its partner
    C  anything else that qualified (kept for the funnel, never promoted)
    """
    grey = ev.get("grey_class") == "GREY"
    if grey and ev.get("coherence") == "MULTI" and ev.get("period_class") != "PERIODIC":
        return "A"
    if grey and ev.get("coherence") == "SINGLE_UNCHECKED" and ev.get("period_class") != "PERIODIC":
        return "B"
    return "C"


def detect_frame(phot: pd.DataFrame, cfg: GreyConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Run ``detect_source`` over every source in a normalised frame."""
    cfg = cfg or GreyConfig()
    events: list[dict] = []
    counts = {"n_sources": 0, "n_too_few": 0, "n_searched": 0, "n_transits_ok_g": 0}
    for _, lc in phot.groupby("source_id", sort=False):
        counts["n_sources"] += 1
        ev, summ = detect_source(lc, cfg)
        if summ["status"] != "ok":
            counts["n_too_few"] += 1
            continue
        counts["n_searched"] += 1
        counts["n_transits_ok_g"] += summ["n_ok_g"]
        for e in ev:
            e["tier"] = tier(e)
        events.extend(ev)
    df = pd.DataFrame(events)
    return df, counts


# ---------------------------------------------------------------------------
# Injection into real light curves (the photometric positive control)
# ---------------------------------------------------------------------------
def inject_episode(lc: pd.DataFrame, depth: float, *, ratio_bp_rp: float = 1.0,
                   n_transits: int = 2, rng: np.random.Generator | None = None,
                   sign: float = -1.0) -> tuple[pd.DataFrame, dict] | None:
    """Multiply ``n_transits`` consecutive usable transits within one Gaia
    visit by (1 + sign*depth) in G and RP and (1 + sign*depth*ratio) in BP
    (RP carries ``depth``; BP carries ``depth * ratio``; G gets the mean of
    the two, which is what a chromatic occulter gives a broad band)."""
    rng = rng or np.random.default_rng(0)
    lc = lc.sort_values("t").reset_index(drop=True).copy()
    t = lc["t"].to_numpy(float)
    ok = (np.isfinite(t) & np.isfinite(lc["f_g"].to_numpy(float)) & ~lc["bad_g"].to_numpy()
          & np.isfinite(lc["f_bp"].to_numpy(float)) & np.isfinite(lc["f_rp"].to_numpy(float)))
    idx = np.nonzero(ok)[0]
    starts = [i for k, i in enumerate(idx[:-n_transits + 1] if n_transits > 1 else idx)
              if n_transits == 1 or (t[idx[k + n_transits - 1]] - t[i]) <= 0.35]
    if not starts:
        return None
    s = int(rng.choice(starts))
    k0 = int(np.nonzero(idx == s)[0][0])
    sel = idx[k0:k0 + n_transits]
    d_rp = depth
    d_bp = depth * ratio_bp_rp
    d_g = 0.5 * (d_rp + d_bp)
    for col, d in (("f_g", d_g), ("f_bp", d_bp), ("f_rp", d_rp)):
        v = lc[col].to_numpy(float, copy=True)
        v[sel] = v[sel] * (1.0 + sign * d)
        lc[col] = v
    return lc, {"t_inj": float(np.mean(t[sel])), "t0": float(t[sel].min()), "t1": float(t[sel].max()),
                "depth": depth, "ratio": ratio_bp_rp, "n_transits": n_transits}


def recovered(events: list[dict], inj: dict, tol_d: float = 0.5) -> dict | None:
    for e in events:
        if e["t_start"] - tol_d <= inj["t_inj"] <= e["t_end"] + tol_d:
            return e
    return None
