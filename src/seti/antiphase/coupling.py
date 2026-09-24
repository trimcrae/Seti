"""ANTIPHASE: an optical fade that the mid-infrared answers with a rise.  Pure.

The observable
--------------
On NEOWISE's own epochs (a visit every ~6 months, 2014--2024), the optical
flux is binned from the ZTF g and r light curves inside a window around each
visit.  At every epoch where all four bands exist we then have

* ``dg, dr`` --- optical change against the star's *unfaded* level (> 0 = fainter),
* ``dW1, dW2`` --- IR change against the IR level *at those same unfaded epochs*
  (< 0 = brighter).

The unfaded level is the median over the optically brightest half of the
matched epochs.  A star with no fade has dg, dr ~ 0 everywhere and nothing
below follows; a star that fades has a set of *faded epochs* F (both g and r
fainter by >= ``n_sigma_fade`` and by >= ``min_depth_mag``), and the question
is what the IR does at F.

Tests (all on the matched epochs)
---------------------------------
* **IR answer at F**: the error-weighted mean of ``-dW`` over F, in sigma, per
  band.  ``COUPLED`` needs >= ``n_sigma_ir`` in **W2**, and W1 not fading by
  >= ``w1_contradict_sigma``.  W2 is required and W1 is not because an
  energy-balanced re-radiator at ~300 K (the habitable-zone case) adds almost
  nothing at 3.4 um; the independent second band is the optical fade itself,
  measured in g **and** r.  A W1 rise as well is recorded as ``ir_two_band``.
* **Specificity** (``ir_contrast``): the W2 brightening at F divided by the
  star's *own* epoch-to-epoch W2 scatter outside F (1.4826 MAD, which holds
  the measurement noise and any unrelated variability) in quadrature with its
  error; >= ``n_sigma_contrast`` is required.  An IR that wanders by as much
  at epochs where nothing faded is not answering the fade.  The weighted
  regression of the IR flux ratio ``10^(-0.4 dW) - 1`` on the optical deficit
  (``slope_sigma``) is reported as well; it is not a gate, because its
  flux-space errors grow with the excess and a large event down-weights
  itself.
* **Lag**: the correlation of deficit and IR excess with the IR series shifted
  by -``max_lag``..+``max_lag`` epochs.  ASASSN-21qj's IR led its dimming by
  ~900 d (Kenworthy et al. 2023): an IR-leads event is a real coupled event,
  but not a *simultaneous* one, and it is labelled so.
* **Chromaticity**: ``dg = k dr`` through the origin with errors in both
  (effective variance), over F and all epochs.  Grey is ``k = 1``; ISM-like
  dust is ``k ~ 1.41`` (ZTF g/r, R_V = 3.1); scattering-dominated blueing is
  ``k < 1``.

Labels (first that applies): ``NO_OPTICAL``, ``NO_IR``, ``INSUFFICIENT_MATCHED``,
``NO_FADE``, ``FADE_IR_FADES`` (IR also fainter at F: an occulter that dims
both, or a secular systematic), ``FADE_IR_FLAT``, ``W1_ONLY``,
``IR_BANDS_DISAGREE``, ``NOT_PROPORTIONAL``, ``COUPLED``; any non-simultaneous
label becomes ``LAGGED_COUPLING`` when the lag scan finds the IR answering the
fade a year or more early or late (best-shift correlation >= 0.5 and >= 0.2
above lag 0, and a >= 3 sigma W2 brightening at the shifted faded epochs).  ``IR_RISE_NO_FADE`` replaces ``NO_FADE``
when the IR rises by >= ``n_sigma_ir`` in both bands at some epoch anyway (an
IGNITION-like star, reported, never a candidate here).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

MJD_J2000 = 51544.5

DEFAULT_COUPLING: dict = {
    "half_window_days": 60.0,     # optical points within +-60 d of a NEOWISE visit
    "min_points_per_bin": 5,
    "bin_err_floor_mag": 0.003,
    "bin_sys_mag": 0.007,         # ZTF season-to-season calibration floor
    "clip_sigma": 5.0,
    "min_matched": 6,             # epochs with every band present
    "min_optical_bands": 2,       # survey: g AND r; controls may run on 1 band
    "n_sigma_fade": 4.0,          # per band, per faded epoch
    "min_depth_mag": 0.02,
    "n_sigma_ir": 3.0,            # mean IR brightening over F, per band
    "n_sigma_contrast": 3.0,      # W2 rise at F against the star's own W2 scatter outside F
    "w1_contradict_sigma": 2.0,   # W1 fainter by this much while W2 rises: disagree
    "max_lag": 6,                 # epochs (~3 yr)
    "lag_min_epochs": 2,          # a lagged coupling is shifted by >= 1 yr ...
    "lag_corr_min": 0.5,          # ... correlates at >= 0.5 at its best shift ...
    "lag_corr_margin": 0.2,       # ... and beats the unshifted correlation by 0.2
    "k_ism": 1.41,                # A_g / A_r for R_V = 3.1 dust in ZTF g, r
    "grey_n_sigma": 2.0,          # |k - 1| within this is grey ...
    "ism_n_sigma": 3.0,           # ... and k must be below k_ism by this much
}


def mjd_to_year(mjd):
    return 2000.0 + (np.asarray(mjd, float) - MJD_J2000) / 365.25


def year_to_mjd(t_yr):
    return MJD_J2000 + (np.asarray(t_yr, float) - 2000.0) * 365.25


# ---------------------------------------------------------------------------
# binning
# ---------------------------------------------------------------------------
def bin_at_epochs(t_mjd, mag, err, epoch_mjd, conf: dict | None = None) -> dict:
    """Bin one optical band at each NEOWISE epoch.

    Per epoch: 5-sigma MAD-clipped median of the points within
    ``half_window_days``; error ``sqrt(max(1.4826 MAD / sqrt(n), floor)^2 + sys^2)``.
    Epochs with fewer than ``min_points_per_bin`` points are NaN.  Returns
    ``{"mag": array, "err": array, "n": array}`` aligned with ``epoch_mjd``.
    """
    c = {**DEFAULT_COUPLING, **(conf or {})}
    t = np.asarray(t_mjd, float)
    m = np.asarray(mag, float)
    e = np.asarray(err, float) if err is not None else np.full_like(m, np.nan)
    ok = np.isfinite(t) & np.isfinite(m)
    t, m, e = t[ok], m[ok], e[ok]
    ep = np.asarray(epoch_mjd, float)
    out_m = np.full(ep.size, np.nan)
    out_e = np.full(ep.size, np.nan)
    out_n = np.zeros(ep.size, int)
    if t.size == 0:
        return {"mag": out_m, "err": out_e, "n": out_n}
    order = np.argsort(t)
    t, m, e = t[order], m[order], e[order]
    hw = float(c["half_window_days"])
    for i, tc in enumerate(ep):
        if not np.isfinite(tc):
            continue
        lo, hi = np.searchsorted(t, tc - hw), np.searchsorted(t, tc + hw, side="right")
        v = m[lo:hi]
        if v.size < int(c["min_points_per_bin"]):
            continue
        med = float(np.median(v))
        mad = 1.4826 * float(np.median(np.abs(v - med)))
        if mad > 0:
            keep = np.abs(v - med) <= float(c["clip_sigma"]) * mad
            if keep.sum() >= int(c["min_points_per_bin"]):
                v = v[keep]
                med = float(np.median(v))
                mad = 1.4826 * float(np.median(np.abs(v - med)))
        if mad <= 0:
            ev = e[lo:hi]
            mad = float(np.nanmedian(ev)) if np.isfinite(np.nanmedian(ev)) else 0.0
        stat = max(mad / np.sqrt(v.size), float(c["bin_err_floor_mag"]))
        out_m[i] = med
        out_e[i] = float(np.hypot(stat, float(c["bin_sys_mag"])))
        out_n[i] = int(v.size)
    return {"mag": out_m, "err": out_e, "n": out_n}


# ---------------------------------------------------------------------------
# the detector
# ---------------------------------------------------------------------------
@dataclass
class CouplingResult:
    label: str
    is_coupled: bool = False
    n_matched: int = 0
    optical_bands: list = field(default_factory=list)
    n_faded: int = 0
    faded_t_yr: list = field(default_factory=list)
    depth_mag: float = float("nan")          # weighted-mean optical change at F
    depth_err: float = float("nan")
    frac_lost: float = float("nan")          # 1 - 10^(-0.4 depth)
    frac_lost_err: float = float("nan")
    depth_by_band: dict = field(default_factory=dict)
    ir_dmag: dict = field(default_factory=dict)      # mean dW over F (neg = brighter)
    ir_dmag_err: dict = field(default_factory=dict)
    ir_sigma: dict = field(default_factory=dict)     # brightening in sigma (pos = brighter)
    slope: dict = field(default_factory=dict)        # IR ratio per unit optical deficit
    slope_sigma: dict = field(default_factory=dict)
    ir_contrast: dict = field(default_factory=dict)  # rise at F / own scatter outside F
    best_lag: int = 0
    corr_lag0: float = float("nan")
    corr_best: float = float("nan")
    k_colour: float = float("nan")                   # dg = k dr
    k_colour_err: float = float("nan")
    d_gr_deepest: float = float("nan")
    d_gr_deepest_err: float = float("nan")
    chroma: str = "UNTESTED"
    ir_rise_max_sigma: float = float("nan")          # IGNITION-like info
    score: float = float("nan")                      # min of the coupling sigmas that apply
    ir_two_band: bool = False                        # W1 rose significantly as well as W2
    ir_ref: dict = field(default_factory=dict)       # IR level at the unfaded epochs
    lag_ir_sigma: float = float("nan")               # W2 rise at the lag-shifted faded epochs
    opt_ref: dict = field(default_factory=dict)      # optical level at the unfaded epochs

    def to_dict(self) -> dict:
        return asdict(self)


def _wmean(x, s):
    x = np.asarray(x, float)
    s = np.asarray(s, float)
    ok = np.isfinite(x) & np.isfinite(s) & (s > 0)
    if not ok.any():
        return float("nan"), float("nan")
    w = 1.0 / s[ok] ** 2
    return float(np.sum(w * x[ok]) / np.sum(w)), float(1.0 / np.sqrt(np.sum(w)))


def _wslope(x, y, sy):
    """Weighted slope of y on x with intercept; error inflated by sqrt(chi2_red)."""
    x, y, sy = (np.asarray(a, float) for a in (x, y, sy))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(sy) & (sy > 0)
    x, y, sy = x[ok], y[ok], sy[ok]
    if x.size < 3 or np.ptp(x) <= 0:
        return float("nan"), float("nan")
    w = 1.0 / sy ** 2
    xm = np.sum(w * x) / np.sum(w)
    ym = np.sum(w * y) / np.sum(w)
    sxx = np.sum(w * (x - xm) ** 2)
    if sxx <= 0:
        return float("nan"), float("nan")
    b = float(np.sum(w * (x - xm) * (y - ym)) / sxx)
    a = ym - b * xm
    chi2r = float(np.sum(w * (y - a - b * x) ** 2) / max(x.size - 2, 1))
    return b, float(np.sqrt(max(chi2r, 1.0) / sxx))


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def colour_slope(dg, sg, dr, sr, k0: float = 1.0) -> tuple[float, float]:
    """``dg = k dr`` through the origin, effective-variance iteration."""
    dg, sg, dr, sr = (np.asarray(a, float) for a in (dg, sg, dr, sr))
    ok = np.isfinite(dg) & np.isfinite(dr) & np.isfinite(sg) & np.isfinite(sr)
    dg, sg, dr, sr = dg[ok], sg[ok], dr[ok], sr[ok]
    if dg.size < 2 or np.sum(dr ** 2) <= 0:
        return float("nan"), float("nan")
    k = float(k0)
    for _ in range(20):
        w = 1.0 / (sg ** 2 + (k * sr) ** 2)
        den = np.sum(w * dr * dr)
        if den <= 0:
            return float("nan"), float("nan")
        k_new = float(np.sum(w * dr * dg) / den)
        if abs(k_new - k) < 1e-6:
            k = k_new
            break
        k = k_new
    w = 1.0 / (sg ** 2 + (k * sr) ** 2)
    den = np.sum(w * dr * dr)
    chi2r = float(np.sum(w * (dg - k * dr) ** 2) / max(dg.size - 1, 1))
    return k, float(np.sqrt(max(chi2r, 1.0) / den))


def classify_chroma(k: float, k_err: float, conf: dict | None = None) -> str:
    c = {**DEFAULT_COUPLING, **(conf or {})}
    if not (np.isfinite(k) and np.isfinite(k_err) and k_err > 0):
        return "UNTESTED"
    kism = float(c["k_ism"])
    if abs(k - 1.0) <= float(c["grey_n_sigma"]) * k_err and \
            (kism - k) >= float(c["ism_n_sigma"]) * k_err:
        return "GREY"
    if k - 1.0 >= 3.0 * k_err:
        return "REDDENING_ISM_LIKE" if abs(k - kism) <= 3.0 * k_err else (
            "REDDENING_STEEP" if k > kism else "REDDENING_SHALLOW")
    if 1.0 - k >= 3.0 * k_err:
        return "BLUEING"
    return "AMBIGUOUS"


def assess_coupling(epoch_t_yr, optical: dict, ir: dict, conf: dict | None = None) -> CouplingResult:
    """The detector.  All arrays are aligned with ``epoch_t_yr`` (NaN = missing).

    ``optical`` = ``{"g": (mag, err), "r": (mag, err), ...}`` binned at the
    epochs; ``ir`` = ``{"W1": (mag, err), "W2": (mag, err)}`` on the same
    epochs (ensemble-corrected upstream).
    """
    c = {**DEFAULT_COUPLING, **(conf or {})}
    t = np.asarray(epoch_t_yr, float)
    opt = {b: (np.asarray(v[0], float), np.asarray(v[1], float)) for b, v in (optical or {}).items()
           if v is not None and np.isfinite(np.asarray(v[0], float)).any()}
    irr = {b: (np.asarray(v[0], float), np.asarray(v[1], float)) for b, v in (ir or {}).items()
           if b in ("W1", "W2") and v is not None and np.isfinite(np.asarray(v[0], float)).any()}
    if len(opt) < int(c["min_optical_bands"]):
        return CouplingResult(label="NO_OPTICAL", optical_bands=sorted(opt))
    if set(irr) != {"W1", "W2"}:
        return CouplingResult(label="NO_IR", optical_bands=sorted(opt))
    bands = sorted(opt)
    have = np.isfinite(t)
    for b in bands:
        have &= np.isfinite(opt[b][0]) & np.isfinite(opt[b][1])
    for b in ("W1", "W2"):
        have &= np.isfinite(irr[b][0]) & np.isfinite(irr[b][1])
    idx = np.nonzero(have)[0]
    res = CouplingResult(label="INSUFFICIENT_MATCHED", optical_bands=bands, n_matched=int(idx.size))
    if idx.size < int(c["min_matched"]):
        return res
    # --- unfaded reference: the optically brightest half -------------------
    om = np.vstack([opt[b][0][idx] for b in bands])
    oe = np.vstack([opt[b][1][idx] for b in bands])
    med_b = np.median(om, axis=1, keepdims=True)
    comb = np.mean(om - med_b, axis=0)                 # combined relative mag
    nref = max(int(np.ceil(idx.size / 2.0)), 3)
    ref_sel = np.argsort(comb)[:nref]                  # brightest half
    ref_o = np.median(om[:, ref_sel], axis=1)
    ref_o_err = 1.2533 * np.sqrt(np.mean(oe[:, ref_sel] ** 2, axis=1) / nref)
    d_o = om - ref_o[:, None]
    s_o = np.sqrt(oe ** 2 + ref_o_err[:, None] ** 2)
    ir_m = {b: irr[b][0][idx] for b in ("W1", "W2")}
    ir_e = {b: irr[b][1][idx] for b in ("W1", "W2")}
    ref_ir = {b: float(np.median(ir_m[b][ref_sel])) for b in ("W1", "W2")}
    ref_ir_err = {b: float(1.2533 * np.sqrt(np.mean(ir_e[b][ref_sel] ** 2) / nref))
                  for b in ("W1", "W2")}
    d_ir = {b: ir_m[b] - ref_ir[b] for b in ("W1", "W2")}
    res.ir_ref = dict(ref_ir)
    res.opt_ref = {b: float(ref_o[j]) for j, b in enumerate(bands)}
    s_ir = {b: np.sqrt(ir_e[b] ** 2 + ref_ir_err[b] ** 2) for b in ("W1", "W2")}
    # the IGNITION-like information: the largest two-band IR rise anywhere
    rise = np.minimum(-d_ir["W1"] / s_ir["W1"], -d_ir["W2"] / s_ir["W2"])
    res.ir_rise_max_sigma = float(np.nanmax(rise)) if rise.size else float("nan")
    # --- faded epochs --------------------------------------------------------
    nsf, mind = float(c["n_sigma_fade"]), float(c["min_depth_mag"])
    faded = np.ones(idx.size, bool)
    for j in range(len(bands)):
        faded &= (d_o[j] >= nsf * s_o[j]) & (d_o[j] >= mind)
    # combined grey deficit per epoch (weighted mean over bands)
    w_o = 1.0 / s_o ** 2
    dm = np.sum(w_o * d_o, axis=0) / np.sum(w_o, axis=0)
    dm_err = 1.0 / np.sqrt(np.sum(w_o, axis=0))
    deficit = 1.0 - 10.0 ** (-0.4 * dm)
    # --- lag scan (always informative) ---------------------------------------
    full_def = np.full(t.size, np.nan)
    full_def[idx] = deficit
    full_ir = np.full(t.size, np.nan)
    # the IR series may extend beyond the optical: lag against every IR epoch
    irok = np.isfinite(irr["W1"][0]) & np.isfinite(irr["W2"][0])
    full_ir[irok] = np.mean([10.0 ** (-0.4 * (irr[b][0][irok] - ref_ir[b])) - 1.0
                             for b in ("W1", "W2")], axis=0)
    lags = {}
    for L in range(-int(c["max_lag"]), int(c["max_lag"]) + 1):
        # positive L: the IR at epoch k+L answers the optical at k (IR lags)
        if L >= 0:
            a, b_ = full_def[: t.size - L], full_ir[L:]
        else:
            a, b_ = full_def[-L:], full_ir[: t.size + L]
        lags[L] = _corr(a, b_)
    res.corr_lag0 = lags.get(0, float("nan"))
    fin = {k: v for k, v in lags.items() if np.isfinite(v)}
    if fin:
        res.best_lag = int(max(fin, key=lambda k: fin[k]))
        res.corr_best = float(fin[res.best_lag])
    # --- chromaticity (needs g and r) ----------------------------------------
    if "g" in bands and "r" in bands:
        jg, jr = bands.index("g"), bands.index("r")
        use = faded if faded.sum() >= 2 else np.ones(idx.size, bool)
        k, ke = colour_slope(d_o[jg][use], s_o[jg][use], d_o[jr][use], s_o[jr][use])
        res.k_colour, res.k_colour_err = k, ke
        deep = int(np.argmax(dm))
        res.d_gr_deepest = float(d_o[jg][deep] - d_o[jr][deep])
        res.d_gr_deepest_err = float(np.hypot(s_o[jg][deep], s_o[jr][deep]))
        # grey only means something where there is a fade to colour
        if faded.any() or np.nanmax(dm / dm_err) >= nsf:
            res.chroma = classify_chroma(k, ke, c)
    # --- regression slopes over all matched epochs ---------------------------
    for b in ("W1", "W2"):
        y = 10.0 ** (-0.4 * d_ir[b]) - 1.0
        sy = 0.4 * np.log(10.0) * s_ir[b] * (1.0 + y)
        sl, se = _wslope(deficit, y, sy)
        res.slope[b] = sl
        res.slope_sigma[b] = sl / se if np.isfinite(sl) and np.isfinite(se) and se > 0 else float("nan")
    res.n_faded = int(faded.sum())
    if not faded.any():
        res.label = "IR_RISE_NO_FADE" if res.ir_rise_max_sigma >= float(c["n_sigma_ir"]) else "NO_FADE"
        return res
    res.faded_t_yr = [round(float(x), 3) for x in t[idx][faded]]
    res.depth_mag, res.depth_err = _wmean(dm[faded], dm_err[faded])
    res.frac_lost = float(1.0 - 10.0 ** (-0.4 * res.depth_mag))
    res.frac_lost_err = float(0.4 * np.log(10.0) * (1.0 - res.frac_lost) * res.depth_err)
    for j, b in enumerate(bands):
        mb, eb = _wmean(d_o[j][faded], s_o[j][faded])
        res.depth_by_band[b] = [mb, eb]
    for b in ("W1", "W2"):
        mb, eb = _wmean(d_ir[b][faded], s_ir[b][faded])
        res.ir_dmag[b], res.ir_dmag_err[b] = mb, eb
        res.ir_sigma[b] = -mb / eb if np.isfinite(mb) and eb > 0 else float("nan")
        rest = -d_ir[b][~faded]
        if rest.size >= 3:
            s_int = 1.4826 * float(np.median(np.abs(rest - np.median(rest))))
            s_int = max(s_int, float(np.median(s_ir[b][~faded])))
        else:
            s_int = float(np.median(s_ir[b]))
        res.ir_contrast[b] = float(-mb / np.hypot(eb, s_int)) if np.isfinite(mb) else float("nan")
    # The IR answer.  W2 is the band that must rise: a re-radiator cooler than
    # ~400 K adds almost nothing at 3.4 um (an energy-balanced 5 % fade
    # re-emitted at 300 K is -0.15 mag in W2 and -0.004 mag in W1), so a rule
    # demanding a W1 rise would reject the habitable-zone case by construction.
    # The second-band confirmation is the optical itself (g AND r, another
    # instrument); W1 must merely not contradict (not fading), and whether it
    # rose as well is recorded (``ir_two_band``) and feeds the temperature.
    nsi, nss = float(c["n_sigma_ir"]), float(c["n_sigma_contrast"])
    z1, z2 = res.ir_sigma["W1"], res.ir_sigma["W2"]
    res.ir_two_band = bool(z1 >= nsi and z2 >= nsi)
    zs = [z2, res.ir_contrast["W2"]] + ([z1, res.ir_contrast["W1"]] if res.ir_two_band else [])
    res.score = float(np.nanmin(zs)) if np.isfinite(zs).any() else float("nan")
    if z1 <= -nsi and z2 <= -nsi:
        res.label = "FADE_IR_FADES"
    elif z2 < nsi and z1 < nsi:
        res.label = "FADE_IR_FLAT"
    elif z2 < nsi:
        res.label = "W1_ONLY"                 # hotter than any grain, or a W1 systematic
    elif z1 <= -float(c["w1_contradict_sigma"]):
        res.label = "IR_BANDS_DISAGREE"       # W2 up while W1 significantly down
    elif not (res.ir_contrast["W2"] >= nss):
        res.label = "NOT_PROPORTIONAL"
    else:
        res.label = "COUPLED"
        res.is_coupled = True
    if not res.is_coupled:
        # A coupled event that is not simultaneous (ASASSN-21qj: the IR led the
        # dimming by ~900 d).  The lag scan's best shift, clearly better than no
        # shift, and a real W2 brightening at the faded epochs moved by that
        # shift (against the star's all-epoch median).  Always natural here:
        # the channel's claim is the simultaneous answer.
        L = int(res.best_lag)
        if abs(L) >= int(c["lag_min_epochs"]) and np.isfinite(res.corr_best) and \
                res.corr_best >= float(c["lag_corr_min"]) and \
                (not np.isfinite(res.corr_lag0) or
                 res.corr_best - res.corr_lag0 >= float(c["lag_corr_margin"])):
            w2m, w2e = irr["W2"]
            okw = np.isfinite(w2m) & np.isfinite(w2e)
            sh = idx[faded] + L
            sh = sh[(sh >= 0) & (sh < t.size)]
            sh = sh[okw[sh]] if sh.size else sh
            if sh.size and okw.sum() >= 6:
                med = float(np.median(w2m[okw]))
                merr = 1.2533 * float(np.sqrt(np.mean(w2e[okw] ** 2) / okw.sum()))
                mb, eb = _wmean(w2m[sh], w2e[sh])
                zl = (med - mb) / float(np.hypot(eb, merr))
                res.lag_ir_sigma = float(zl)
                if zl >= float(c["n_sigma_ir"]):
                    res.label = "LAGGED_COUPLING"
    return res


def ir_rise_prescore(t_yr, ir: dict, t_min: float = 2018.2, band: str = "W2") -> float:
    """The order in which stars fetch ZTF, and the survey's completeness variable.

    The largest ``band`` (default W2) brightening, in sigma, at an epoch after
    ``t_min``, against the larger-offset of two references: the star's
    pre-``t_min`` median and its all-epoch median.  W2 alone, because a ~300 K
    re-radiator moves W2 and not W1.  A COUPLED star needs a W2 rise at its
    faded epochs of >= 3 sigma (``n_sigma_ir``) against its unfaded epochs, so
    every coupled star has a prescore near or above that; the ZTF budget is
    spent from the top of this list down, and the prescore of the last star
    reached is reported as the completeness limit.
    """
    t = np.asarray(t_yr, float)
    if band not in ir:
        return float("nan")
    m, e = (np.asarray(x, float) for x in ir[band])
    ok = np.isfinite(m) & np.isfinite(e)
    pre = (t < t_min) & ok
    post = (t >= t_min) & ok
    if post.sum() < 1 or ok.sum() < 4:
        return float("nan")
    zs = []
    for sel in ((pre if pre.sum() >= 3 else None), ok):
        if sel is None:
            continue
        base = float(np.median(m[sel]))
        berr = 1.2533 * float(np.sqrt(np.mean(e[sel] ** 2) / sel.sum()))
        zs.append(np.where(post, (base - m) / np.sqrt(e ** 2 + berr ** 2), np.nan))
    v = np.vstack(zs) if zs else np.full((1, 1), np.nan)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else float("nan")


__all__ = ["DEFAULT_COUPLING", "CouplingResult", "assess_coupling", "bin_at_epochs",
           "classify_chroma", "colour_slope", "ir_rise_prescore", "mjd_to_year", "year_to_mjd"]
