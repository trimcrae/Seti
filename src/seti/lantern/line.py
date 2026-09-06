"""The LANTERN detector -- pure NumPy, offline-tested.

1. :func:`time_average_spectrum`   robust mean of the out-of-eclipse integrations
2. :func:`narrow_feature_search`   unresolved emission features on that spectrum,
                                   with the guards inherited from
                                   :func:`seti.panspermia.dossier.narrow_feature_scan`
                                   (interior, bounded, 1-3 resolution elements
                                   wide, significance against a local continuum
                                   with a robust noise estimate)
3. :func:`line_flux_series`        line flux and local-continuum flux per integration
4. :func:`eclipse_discriminant`    THE test: in-eclipse line flux consistent with
                                   zero, out-of-eclipse flux significantly positive,
                                   the drop at the predicted contacts and not at a
                                   settling ramp, and the line not merely following
                                   the continuum
5. :func:`transit_consistency`     a planet-origin line does not change in transit
                                   beyond the continuum's own change
6. :func:`assess_feature`          vetoes with counters, and the tier
7. :func:`recurrent_wavelengths` / :func:`bh_fdr`   population-level steps

Nothing here fabricates a phase: an observation without the phase coverage for
the discriminant gets ``watch`` at most, never ``candidate``.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats as _stats

VETO_NAMES = (
    "known_artefact_wavelength", "recurrent_across_targets", "tracks_continuum",
    "ramp_correlated", "cosmic_ray_single_integration",
    "insufficient_phase_coverage", "low_snr", "single_pixel_spike",
    "adjacent_to_gap", "drop_not_at_eclipse", "transit_inconsistent",
)

_DEFAULT_LINE_CFG: dict = {
    "sigma_min": 6.0, "min_from_edge": 8, "min_width_resel": 1.0,
    "max_width_resel": 3.0, "continuum_window": 31, "noise_block": 64,
    "clip_sigma": 5.0, "max_features_per_spectrum": 200,
    "line_pad": 1, "side_inner": 3, "side_outer": 15, "cont_half_width": 150,
}
_DEFAULT_DISC_CFG: dict = {
    "out_positive_snr_min": 5.0, "in_eclipse_zero_sigma_max": 2.0,
    "vanish_snr_interest": 3.0, "vanish_snr_candidate": 5.0,
    "tracks_continuum_sigma": 3.0, "continuum_corr_max": 0.5,
    "continuum_corr_p_max": 0.01, "ramp_taus": [5, 10, 20, 40, 80],
    "ramp_corr_max": 0.5, "timing_tolerance_ingress_units": 2.0,
    "cosmic_ray_top_n": 2, "transit_excess_sigma_max": 3.0,
}


# --- 1. time-averaged spectrum ------------------------------------------------
def time_average_spectrum(flux, mask, flux_err=None, clip_sigma: float = 5.0) -> dict:
    """Robust, continuum-normalised mean spectrum over the integrations in ``mask``.

    Each integration is divided by its own median (removing the continuum's
    time systematics before averaging), then averaged per pixel with one
    ``clip_sigma`` clip.  The per-pixel error is the robust scatter across the
    used integrations divided by ``sqrt(n)``, floored by the propagated
    ``flux_err`` when given.  Returns ``spec``, ``spec_err``, ``n_used``,
    ``scale`` (median absolute continuum level).
    """
    f = np.asarray(flux, float)
    m = np.asarray(mask, bool)
    if f.ndim != 2 or m.shape[0] != f.shape[0]:
        raise ValueError("flux must be (n_int, n_wl) and mask length n_int")
    sub = f[m]
    n_wl = f.shape[1]
    if sub.shape[0] == 0:
        return {"spec": np.full(n_wl, np.nan), "spec_err": np.full(n_wl, np.nan),
                "n_used": 0, "scale": np.nan}
    med = np.nanmedian(sub, axis=1)
    med = np.where(np.isfinite(med) & (med != 0), med, np.nan)
    norm = sub / med[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN (dead) columns
        centre = np.nanmedian(norm, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(norm - centre), axis=0)
    mad = np.where(np.isfinite(mad) & (mad > 0), mad, np.nanmedian(mad[np.isfinite(mad)])
                   if np.any(np.isfinite(mad)) else 1.0)
    keep = np.abs(norm - centre) <= clip_sigma * mad
    keep &= np.isfinite(norm)
    n_used = keep.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        spec = np.where(n_used > 0, np.nansum(np.where(keep, norm, 0.0), axis=0)
                        / np.maximum(n_used, 1), np.nan)
        scatter = np.sqrt(np.nansum(np.where(keep, (norm - spec) ** 2, 0.0), axis=0)
                          / np.maximum(n_used - 1, 1))
        spec_err = scatter / np.sqrt(np.maximum(n_used, 1))
        if flux_err is not None:
            e = np.asarray(flux_err, float)[m] / med[:, None]
            prop = np.sqrt(np.nansum(np.where(keep, e ** 2, 0.0), axis=0)) / np.maximum(n_used, 1)
            spec_err = np.fmax(spec_err, np.where(np.isfinite(prop), prop, 0.0))
    spec_err = np.where(n_used > 1, spec_err, np.nan)
    return {"spec": spec, "spec_err": spec_err,
            "n_used": int(np.nanmedian(n_used)) if n_used.size else 0,
            "scale": float(np.nanmedian(med)) if np.any(np.isfinite(med)) else np.nan}


# --- 2. narrow-feature search ---------------------------------------------------
def running_median(x, window: int) -> np.ndarray:
    """NaN-aware centred running median (window forced odd)."""
    x = np.asarray(x, float)
    w = max(3, int(window) | 1)
    h = w // 2
    pad = np.concatenate([np.full(h, np.nan), x, np.full(h, np.nan)])
    view = np.lib.stride_tricks.sliding_window_view(pad, w)      # (n, w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN windows
        return np.nanmedian(view, axis=1)


def local_poly_continuum(spec, window: int = 31, hole: int = 4, mask=None,
                         deg: int = 2) -> np.ndarray:
    """Notched local-polynomial continuum: at each sample, the least-squares
    polynomial (``deg``) through the ``window`` neighbours EXCLUDING ``+-hole``
    samples around the sample itself (so a narrow feature does not pull its
    own continuum) and any ``mask``-ed samples, evaluated at the sample.  A
    running median fails on a steep continuum (it returns the centre sample
    exactly, and a line shifts its rank); a local line leaves the band
    curvature in the residual; a local quadratic with a hole does neither.
    """
    y = np.asarray(spec, float)
    n = y.size
    h = max(int(window) // 2, hole + deg + 2)
    ok = np.isfinite(y)
    if mask is not None:
        ok &= ~np.asarray(mask, bool)
    pred = np.full(n, np.nan)
    widths = [h, 2 * h, 4 * h]          # widen inside long masked/NaN runs
    for i in range(n):
        use = np.empty(0, int)
        hw = h
        for hw in widths:
            lo, hi = max(0, i - hw), min(n, i + hw + 1)
            idx = np.arange(lo, hi)
            idx = idx[np.abs(idx - i) > hole]
            use = idx[ok[idx]]
            if use.size >= deg + 3:
                break
        if use.size < deg + 3:
            if use.size:
                pred[i] = np.mean(y[use])
            continue
        xl = (use - i) / float(hw)
        try:
            coef = np.polyfit(xl, y[use], deg)
        except (np.linalg.LinAlgError, ValueError):
            pred[i] = np.mean(y[use])
            continue
        pred[i] = coef[-1]          # polynomial at x = 0
    return pred


def block_noise(resid, block: int = 64, spec_err=None) -> np.ndarray:
    """Robust per-pixel noise: block-wise MAD of the residual, interpolated,
    floored by the propagated error."""
    r = np.asarray(resid, float)
    n = r.size
    b = max(8, int(block))
    centres, mads = [], []
    for s in range(0, n, b):
        seg = r[s:s + b]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 5:
            centres.append(s + 0.5 * min(b, n - s))
            mads.append(1.4826 * np.median(np.abs(seg - np.median(seg))))
    if not mads:
        sig = np.full(n, np.nan)
    elif len(mads) == 1:
        sig = np.full(n, mads[0])
    else:
        sig = np.interp(np.arange(n), centres, mads)
    fin = np.isfinite(r)
    floor = 1e-6 * (np.nanmedian(np.abs(r[fin])) if fin.any() else 1.0) + 1e-12
    sig = np.fmax(sig, floor)
    if spec_err is not None:
        e = np.asarray(spec_err, float)
        sig = np.fmax(sig, np.where(np.isfinite(e), e, 0.0))
    return sig


def _half_max_extent(z, i, max_reach: int):
    """Samples above half-peak around ``i`` and whether the feature is bounded."""
    n = z.size
    half = z[i] / 2.0
    left = i
    while left > 0 and i - left < max_reach and np.isfinite(z[left - 1]) and z[left - 1] > half:
        left -= 1
    right = i
    while right < n - 1 and right - i < max_reach and np.isfinite(z[right + 1]) and z[right + 1] > half:
        right += 1
    bounded = (left > 0 and np.isfinite(z[left - 1]) and z[left - 1] <= half
               and right < n - 1 and np.isfinite(z[right + 1]) and z[right + 1] <= half)
    return left, right, bounded


def _width_bounds(spr: float, c: dict) -> tuple[int, int]:
    """Accepted half-max width range in samples for ``spr`` samples per resel.
    An unresolved line at Nyquist sampling has FWHM ~2 samples; a single
    sample above half-peak is sub-resolution (hot pixel / cosmic ray)."""
    min_w = max(1, int(round(c["min_width_resel"] * spr)))
    max_w = max(min_w, int(round(c["max_width_resel"] * spr)))
    min_w = max(min_w, 2) if spr >= 2 else min_w
    return min_w, max_w


def residual_z(spec, spec_err, samples_per_resel: float = 2.0, cfg: dict | None = None) -> dict:
    """Continuum residual and its significance for a normalised spectrum.

    Iterated notched local-quadratic continuum: each pass masks every
    >4-sigma excursion (and its wings) so a strong feature -- or a stellar
    line forest, whose first-pass residual dominates the noise -- cannot bias
    the continuum of the samples beside it; iterate until the mask converges.
    Returns ``z``, ``resid``, ``noise``, ``cont`` and the final ``mask``.
    """
    c = {**_DEFAULT_LINE_CFG, **(cfg or {})}
    s = np.asarray(spec, float)
    n = s.size
    _, max_w = _width_bounds(max(float(samples_per_resel), 1.0), c)
    hole = max_w + 1
    cont = local_poly_continuum(s, c["continuum_window"], hole)
    resid = s - cont
    noise = block_noise(resid, c["noise_block"], spec_err)
    grow = np.zeros(n, bool)
    for _pass in range(4):
        with np.errstate(invalid="ignore", divide="ignore"):
            z0 = resid / noise
        excur = np.isfinite(z0) & (np.abs(z0) > 4.0)
        new = excur.copy()
        for k in (1, 2):                 # the wings of an excursion
            new[k:] |= excur[:-k]
            new[:-k] |= excur[k:]
        new |= grow
        if not new.any() or np.array_equal(new, grow):
            break
        grow = new
        cont = local_poly_continuum(s, c["continuum_window"], hole, mask=grow)
        resid = s - cont
        noise = block_noise(resid, c["noise_block"], spec_err)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = resid / noise
    return {"z": z, "resid": resid, "noise": noise, "cont": cont, "mask": grow}


def feature_snr_in_mask(flux, mask, index: int, flux_err=None, samples_per_resel: float = 2.0,
                        cfg: dict | None = None, halfwidth: int = 1) -> float:
    """Significance of a feature at ``index`` on the time-averaged spectrum of
    the integrations in ``mask`` (max z within ``+-halfwidth``), with the same
    masked-quadratic continuum the search uses.  This is the in-eclipse
    'consistent with zero' test: a series-based in-eclipse mean inherits any
    constant bias of the side-window continuum (a neighbouring absorption line
    offsets it identically in every integration), whereas the spectrum-level
    residual handles curvature and the forest by construction."""
    c = {**_DEFAULT_LINE_CFG, **(cfg or {})}
    avg = time_average_spectrum(flux, mask, flux_err, float(c.get("clip_sigma", 5.0)))
    if avg["n_used"] < 2:
        return np.nan
    z = residual_z(avg["spec"], avg["spec_err"], samples_per_resel, c)["z"]
    lo, hi = max(0, index - halfwidth), min(z.size, index + halfwidth + 1)
    seg = z[lo:hi]
    return float(np.nanmax(seg)) if np.any(np.isfinite(seg)) else np.nan


def _template_fwhm(spec, noise, i, reach: int, fwhm_max: float,
                   exclude=None) -> tuple[float, float]:
    """Best-fitting Gaussian FWHM (samples) of the feature at ``i`` by weighted
    least squares of ``a * template + b + c * x`` over ``+-reach`` samples of
    the (masked-continuum) residual, for templates from a single-pixel spike
    (FWHM 1) to ``fwhm_max``.  The residual is used because the excursion mask
    keeps a broad feature's shape (its >4-sigma core and wings are excluded
    from the continuum fit) while removing the stellar line forest that would
    otherwise mislead a fit on the raw spectrum.  A matched-template width is
    stable where a moment or a half-max count is not.  Returns
    ``(fwhm, sub_pixel_centre)``."""
    n = spec.size
    lo, hi = max(0, i - reach), min(n, i + reach + 1)
    y = spec[lo:hi]
    s = noise[lo:hi]
    ok = np.isfinite(y) & np.isfinite(s) & (s > 0)
    if exclude is not None:
        ok &= ~np.asarray(exclude, bool)[lo:hi]
    if ok.sum() < 5:
        return 1.0, float(i)
    x = np.arange(lo, hi, dtype=float)[ok]
    y, w = y[ok], 1.0 / s[ok] ** 2
    sw = np.sqrt(w)
    grid = np.unique(np.concatenate([[1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
                                     np.linspace(3.0, max(fwhm_max, 3.0), 10)]))
    xl = (x - i) / max(reach, 1)
    best = (np.inf, 1.0, float(i))
    for fwhm in grid:
        sig = fwhm / 2.3548
        for c0 in (i - 0.5, i - 0.25, i, i + 0.25, i + 0.5):
            tmpl = np.exp(-0.5 * ((x - c0) / sig) ** 2)
            A = np.column_stack([tmpl, np.ones_like(x), xl])
            coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
            if coef[0] <= 0:
                continue
            chi2 = float(np.sum(w * (y - A @ coef) ** 2))
            if chi2 < best[0]:
                best = (chi2, float(fwhm), float(c0))
    return best[1], best[2]


def narrow_feature_search(wavelength, spec, spec_err=None, samples_per_resel: float = 2.0,
                          cfg: dict | None = None) -> dict:
    """Unresolved emission features on a (normalised) time-averaged spectrum.

    Continuum = notched local-quadratic fit (:func:`local_poly_continuum`,
    two passes with >4-sigma excursions masked); noise = block-wise MAD of the
    residual floored by ``spec_err``;
    a feature is a local maximum of ``z = resid / noise`` with ``z >= sigma_min``
    that is interior (``min_from_edge`` samples from either end), bounded (falls
    below half-peak on both sides), not adjacent to a NaN gap, and whose
    half-max width lies in ``[min_width_resel, max_width_resel]`` resolution
    elements.  A single-sample spike is a hot pixel / cosmic ray, not a line.

    Returns ``features`` (list of dicts), ``n_scanned`` (independent trials =
    interior finite samples / samples_per_resel), the guard ``counters`` and
    the 5-sigma equivalent-width sensitivity in the spectrum's wavelength unit.
    """
    c = {**_DEFAULT_LINE_CFG, **(cfg or {})}
    wl = np.asarray(wavelength, float)
    s = np.asarray(spec, float)
    n = s.size
    counters = {"single_pixel_spike": 0, "adjacent_to_gap": 0, "too_wide": 0,
                "unbounded": 0, "edge": 0}
    out = {"features": [], "n_scanned": 0, "counters": counters,
           "ew_5sigma_limit": None, "noise_median": None}
    if n < 2 * c["min_from_edge"] + 3 or not np.any(np.isfinite(s)):
        return out
    spr = max(float(samples_per_resel), 1.0)
    min_w, max_w = _width_bounds(spr, c)
    rz = residual_z(s, spec_err, spr, c)
    z, resid, noise, cont, grow = rz["z"], rz["resid"], rz["noise"], rz["cont"], rz["mask"]
    lo, hi = c["min_from_edge"], n - c["min_from_edge"]
    dlam = np.abs(np.gradient(wl)) if n > 1 else np.ones(n)
    feats = []
    for i in range(lo, hi):
        if not np.isfinite(z[i]) or z[i] < c["sigma_min"]:
            continue
        win = z[max(0, i - 2): i + 3]
        if not np.isfinite(win).all() or z[i] != np.max(win):
            if np.isfinite(z[i]) and z[i] == np.nanmax(win) and not np.isfinite(win).all():
                counters["adjacent_to_gap"] += 1
            continue
        left, right, bounded = _half_max_extent(z, i, max_w + 2)
        width = right - left + 1
        reach = max_w + 2
        if not np.isfinite(z[max(0, left - reach): right + reach + 1]).all():
            counters["adjacent_to_gap"] += 1
            continue
        if not bounded:
            counters["unbounded"] += 1
            continue
        # Width by matched template: a Nyquist-sampled unresolved line centred
        # on a pixel has its neighbours at exactly half peak, so a
        # count-above-half-max is unstable there; a template fit is not.  A
        # single-sample spike fits the FWHM~1 template.
        # Masked NEGATIVE residuals (absorption lines of the forest) are kept
        # out of the width fit; masked positive ones are the feature's own wings.
        fwhm, _c0 = _template_fwhm(resid, noise, i, 3 * max_w, 3.0 * max_w,
                                   exclude=grow & (resid < 0))
        if fwhm < 0.75 * min_w:
            counters["single_pixel_spike"] += 1
            continue
        if fwhm > max_w or width > max_w + 1:
            counters["too_wide"] += 1
            continue
        ew = float(np.nansum((resid[left:right + 1] / np.fmax(cont[left:right + 1], 1e-12))
                             * dlam[left:right + 1]))
        feats.append({
            "index": int(i), "wavelength": float(wl[i]), "snr": float(z[i]),
            "width_samples": int(width), "fwhm_samples": float(fwhm),
            "width_resel": float(fwhm / spr),
            "left": int(left), "right": int(right),
            "amplitude_norm": float(resid[i]),
            "equivalent_width": ew, "local_noise": float(noise[i]),
        })
    feats.sort(key=lambda d: -d["snr"])
    feats = feats[: int(c["max_features_per_spectrum"])]
    fin = np.isfinite(z[lo:hi])
    out.update(features=feats, n_scanned=int(np.count_nonzero(fin) / spr))
    if fin.any():
        nm = float(np.nanmedian(noise[lo:hi][fin]))
        dl = float(np.nanmedian(dlam))
        out["noise_median"] = nm
        # 5-sigma EW limit for a line spanning one resolution element.
        out["ew_5sigma_limit"] = float(5.0 * nm * np.sqrt(spr) * dl)
    return out


# --- 3. per-integration line flux ----------------------------------------------
def line_flux_series(flux, left: int, right: int, flux_err=None,
                     cfg: dict | None = None) -> dict:
    """Line flux and local-continuum flux per integration (absolute units).

    Line pixels are ``[left - line_pad, right + line_pad]``; the local continuum
    per integration is the median of two side windows ``side_inner..side_outer``
    samples beyond the line on each side, and ``line`` is the summed excess
    over it.  ``cont`` is the star's broad-band light curve: the median of the
    integration over a wide region (``+-cont_half_width`` samples, the line and
    its side windows excluded) scaled to the line's pixel count -- deliberately
    NOT the subtraction window, whose noise would otherwise be shared with the
    line series and manufacture an anticorrelation.  Errors come from
    ``flux_err`` when given, else from the side-window scatter.
    """
    c = {**_DEFAULT_LINE_CFG, **(cfg or {})}
    f = np.asarray(flux, float)
    n_int, n_wl = f.shape
    p0, p1 = max(0, left - c["line_pad"]), min(n_wl - 1, right + c["line_pad"])
    li = np.arange(p0, p1 + 1)
    ls = np.arange(max(0, p0 - c["side_outer"]), max(0, p0 - c["side_inner"] + 1))
    rs = np.arange(min(n_wl, p1 + c["side_inner"]), min(n_wl, p1 + c["side_outer"] + 1))
    side = np.concatenate([ls, rs])
    if side.size < 4:
        return {"line": np.full(n_int, np.nan), "line_err": np.full(n_int, np.nan),
                "cont": np.full(n_int, np.nan), "n_line_pixels": int(li.size),
                "note": "no side windows"}
    cont_lvl = np.nanmedian(f[:, side], axis=1)
    # Per-integration noise from the TEMPORAL residual of the side pixels
    # (each integration minus the time-median spectrum on those pixels): the
    # spread of the side pixels around their own median is stellar structure
    # (lines, slope), not noise, and would inflate the error many-fold.
    tmed = np.nanmedian(f[:, side], axis=0)
    side_sc = 1.4826 * np.nanmedian(np.abs(f[:, side] - tmed[None, :]), axis=1)
    line = np.nansum(f[:, li] - cont_lvl[:, None], axis=1)
    hw = int(c.get("cont_half_width", 150))
    broad = np.arange(max(0, p0 - hw), min(n_wl, p1 + hw + 1))
    broad = broad[(broad < side.min()) | (broad > side.max())]
    if broad.size < 8:
        broad = side
    cont = np.nanmedian(f[:, broad], axis=1) * li.size
    err = side_sc * np.sqrt(li.size) * np.sqrt(1.0 + li.size / max(side.size, 1))
    if flux_err is not None:
        e = np.asarray(flux_err, float)
        prop = np.sqrt(np.nansum(e[:, li] ** 2, axis=1))
        err = np.fmax(err, np.where(np.isfinite(prop), prop, 0.0))
    return {"line": line, "line_err": err, "cont": cont, "n_line_pixels": int(li.size)}


# --- 4. the eclipse discriminant --------------------------------------------------
def _mean_err(x, e):
    x = np.asarray(x, float)
    e = np.asarray(e, float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return np.nan, np.nan, 0
    mu = float(np.mean(x[ok]))
    n = int(ok.sum())
    sc = float(np.std(x[ok], ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    pr = (float(np.sqrt(np.nansum(e[ok] ** 2)) / n) if np.any(np.isfinite(e[ok])) else np.nan)
    err = np.nanmax([sc, pr]) if (np.isfinite(sc) or np.isfinite(pr)) else np.nan
    return mu, float(err), n


def _fit_scale(y, w, model):
    """Weighted LS of y = a*model + b; returns chi2, a, b."""
    A = np.vstack([model, np.ones_like(model)]).T
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    chi2 = float(np.sum(w * (y - A @ coef) ** 2))
    return chi2, float(coef[0]), float(coef[1])


def eclipse_discriminant(line, line_err, cont, labels: dict, times=None,
                         cfg: dict | None = None) -> dict:
    """The eclipse-vanishing test on one feature's line-flux series.

    Reports ``eclipse_vanish_snr`` (out minus in, over their joint error),
    ``out_positive_snr``, ``in_eclipse_sigma`` (in-eclipse mean over its error),
    the fractional drops of line and continuum, ``continuum_correlation``
    (line vs continuum over OUT-of-eclipse integrations, where a planet line is
    constant and any correlation is systematics), ``ramp_correlation`` (best
    settling-ramp template) with the ramp-vs-step model comparison, and the
    free-step timing offset from the predicted ingress.  Vetoes derived from
    these are decided in :func:`assess_feature`.
    """
    c = {**_DEFAULT_DISC_CFG, **(cfg or {})}
    y = np.asarray(line, float)
    e = np.asarray(line_err, float)
    k = np.asarray(cont, float)
    n = y.size
    inn, out = np.asarray(labels["in_eclipse"], bool), np.asarray(labels["out_eclipse"], bool)
    res: dict = {"n_in": int(inn.sum()), "n_out": int(out.sum())}
    mu_out, s_out, _ = _mean_err(y[out], e[out])
    mu_in, s_in, _ = _mean_err(y[inn], e[inn])
    kc_out, sk_out, _ = _mean_err(k[out], np.full(out.sum(), np.nan))
    kc_in, sk_in, _ = _mean_err(k[inn], np.full(inn.sum(), np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        res["out_mean"], res["out_err"] = mu_out, s_out
        res["in_mean"], res["in_err"] = mu_in, s_in
        res["out_positive_snr"] = float(mu_out / s_out) if s_out > 0 else np.nan
        res["in_eclipse_sigma"] = float(mu_in / s_in) if s_in > 0 else np.nan
        joint = np.hypot(s_out, s_in)
        res["eclipse_vanish_snr"] = float((mu_out - mu_in) / joint) if joint > 0 else np.nan
        res["line_fractional_drop"] = float((mu_out - mu_in) / mu_out) if mu_out else np.nan
        res["line_fractional_drop_err"] = float(joint / abs(mu_out)) if mu_out else np.nan
        res["continuum_fractional_drop"] = (float((kc_out - kc_in) / kc_out) if kc_out else np.nan)
        res["continuum_fractional_drop_err"] = (float(np.hypot(sk_out, sk_in) / abs(kc_out))
                                                if kc_out else np.nan)
    # Out-of-eclipse line-vs-continuum correlation.
    yo, ko = y[out], k[out]
    okc = np.isfinite(yo) & np.isfinite(ko)
    if okc.sum() >= 5 and np.std(yo[okc]) > 0 and np.std(ko[okc]) > 0:
        r, p = _stats.pearsonr(yo[okc], ko[okc])
        res["continuum_correlation"], res["continuum_correlation_p"] = float(r), float(p)
    else:
        res["continuum_correlation"], res["continuum_correlation_p"] = np.nan, np.nan
    # Ramp templates vs the eclipse-step model.
    ok = np.isfinite(y) & np.isfinite(e) & (e > 0)
    idx = np.arange(n, dtype=float)
    res["ramp_correlation"], res["ramp_tau"] = np.nan, None
    res["chi2_ramp"], res["chi2_step_predicted"], res["chi2_flat"] = np.nan, np.nan, np.nan
    if ok.sum() >= 8:
        w = 1.0 / e[ok] ** 2
        yy = y[ok]
        res["chi2_flat"] = _fit_scale(yy, w, np.zeros(ok.sum()))[0]
        best = (np.inf, None, np.nan)
        for tau in c["ramp_taus"]:
            tmpl = np.exp(-idx[ok] / float(tau))
            chi2, a, _ = _fit_scale(yy, w, tmpl)
            if np.std(tmpl) > 0 and np.std(yy) > 0:
                r = float(np.corrcoef(tmpl, yy)[0, 1])
            else:
                r = np.nan
            if chi2 < best[0]:
                best = (chi2, tau, r)
        res["chi2_ramp"], res["ramp_tau"], res["ramp_correlation"] = best
        if inn.any():
            # Eclipse-step model = the full in-eclipse mask (a phase curve may
            # hold two eclipses), fitted as two group means.
            res["chi2_step_predicted"] = _two_group_chi2(yy, w, inn[ok])
            # Free-step scan: slide the whole in-eclipse pattern and ask where
            # the drop actually sits.
            best_chi, best_d = np.inf, None
            i1p = int(np.flatnonzero(inn)[0])
            for d in range(-i1p, n - i1p):
                shifted = np.zeros(n, bool)
                if d >= 0:
                    shifted[d:] = inn[:n - d]
                else:
                    shifted[:n + d] = inn[-d:]
                sm = shifted[ok]
                if sm.sum() < 2 or (~sm).sum() < 2:
                    continue
                chi2 = _two_group_chi2(yy, w, sm)
                if chi2 < best_chi:
                    best_chi, best_d = chi2, d
            res["chi2_step_free"] = float(best_chi)
            res["free_step_offset_integrations"] = int(best_d) if best_d is not None else None
            if times is not None and best_d is not None:
                t = np.asarray(times, float)
                j = min(max(i1p + best_d, 0), n - 1)
                if t.size == n:
                    res["free_step_offset_days"] = float(t[j] - t[i1p])
    return res


def _two_group_chi2(y, w, inside) -> float:
    """Weighted chi-square of a two-level (in/out) model with free levels."""
    chi = 0.0
    for g in (inside, ~inside):
        if g.any():
            sw = np.sum(w[g])
            mu = np.sum(w[g] * y[g]) / sw
            chi += float(np.sum(w[g] * (y[g] - mu) ** 2))
    return chi


# --- 5. transit consistency -----------------------------------------------------
def transit_consistency(line, line_err, cont, labels: dict) -> dict:
    """A planet-origin line does not change in transit beyond the continuum's change.

    Reports the fractional in-vs-out-of-transit change of the line and of the
    continuum, ``transit_constancy`` (line change over its error) and
    ``transit_excess_sigma`` (how much MORE the line changes than the
    continuum, in sigma).  A line that changes more than the continuum is not
    a steady source on the planet.
    """
    y, e, k = (np.asarray(a, float) for a in (line, line_err, cont))
    inn, out = np.asarray(labels["in_transit"], bool), np.asarray(labels["out_transit"], bool)
    mu_out, s_out, _ = _mean_err(y[out], e[out])
    mu_in, s_in, _ = _mean_err(y[inn], e[inn])
    kc_out, sk_out, _ = _mean_err(k[out], np.full(out.sum(), np.nan))
    kc_in, sk_in, _ = _mean_err(k[inn], np.full(inn.sum(), np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        lf = (mu_in - mu_out) / mu_out if mu_out else np.nan
        lf_e = np.hypot(s_in, s_out) / abs(mu_out) if mu_out else np.nan
        cf = (kc_in - kc_out) / kc_out if kc_out else np.nan
        cf_e = np.hypot(sk_in, sk_out) / abs(kc_out) if kc_out else np.nan
        const = abs(lf) / lf_e if lf_e and lf_e > 0 else np.nan
        joint = np.hypot(lf_e, cf_e)
        excess = (abs(lf) - abs(cf)) / joint if joint and joint > 0 else np.nan
    return {"n_in": int(inn.sum()), "n_out": int(out.sum()),
            "line_fractional_change": float(lf), "line_fractional_change_err": float(lf_e),
            "continuum_fractional_change": float(cf),
            "continuum_fractional_change_err": float(cf_e),
            "transit_constancy": float(const), "transit_excess_sigma": float(excess)}


# --- 6. vetoes and tiers -----------------------------------------------------------
def known_artefact(wavelength: float, artefact_rows: list[dict] | None,
                   edge_tol: float = 0.01) -> dict | None:
    """Return the artefact row covering ``wavelength`` (+- ``edge_tol``), else None."""
    for row in artefact_rows or []:
        lo, hi = row.get("range_um", (np.nan, np.nan))
        if lo - edge_tol <= wavelength <= hi + edge_tol:
            return row
    return None


def cosmic_ray_driven(flux, mask, left: int, right: int, sigma_min: float,
                      top_n: int = 2, samples_per_resel: float = 2.0,
                      cfg: dict | None = None, flux_err=None) -> dict:
    """Does the time-averaged feature survive dropping its ``top_n`` brightest
    integrations?  If not, it was a cosmic ray / single-integration event."""
    f = np.asarray(flux, float)
    m = np.asarray(mask, bool).copy()
    series = line_flux_series(f, left, right, flux_err, cfg)["line"]
    order = np.argsort(np.where(m, series, -np.inf))[::-1]
    dropped = [int(i) for i in order[:top_n] if m[i]]
    m[dropped] = False
    avg = time_average_spectrum(f, m, flux_err, (cfg or {}).get("clip_sigma", 5.0))
    scan = narrow_feature_search(np.arange(f.shape[1]), avg["spec"], avg["spec_err"],
                                 samples_per_resel, cfg)
    centre = 0.5 * (left + right)
    snr_after = max((d["snr"] for d in scan["features"] if abs(d["index"] - centre) <= 2),
                    default=0.0)
    out_series = series[np.asarray(mask, bool)]
    med = np.nanmedian(out_series)
    mad = 1.4826 * np.nanmedian(np.abs(out_series - med)) + 1e-12
    n_hi = int(np.count_nonzero((out_series - med) / mad > 5.0))
    return {"snr_after_dropping": float(snr_after), "dropped": dropped,
            "n_integrations_above_5mad": n_hi,
            "cosmic_ray_driven": bool(snr_after < sigma_min)}


def assess_feature(feature: dict, disc: dict | None, transit: dict | None,
                   phase_class: str, labels: dict | None = None,
                   artefact: dict | None = None, recurrent: bool = False,
                   cosmic: dict | None = None, cfg: dict | None = None) -> dict:
    """Apply every veto with a named counter and assign the tier.

    Tiers: ``none`` (vetoed), ``watch`` (a clean narrow feature whose phase
    coverage cannot test vanishing), ``interest`` (vanishes at >= interest
    SNR), ``candidate`` (vanishes at >= candidate SNR, in-eclipse consistent
    with zero, drop timed at the predicted contact, transit-consistent).  FDR
    is applied afterwards across the population (:func:`bh_fdr`).
    """
    c = {**_DEFAULT_DISC_CFG, **(cfg or {})}
    disc = {k: _num(v) for k, v in (disc or {}).items()} if disc is not None else None
    vetoes: list[str] = []
    if artefact is not None:
        vetoes.append("known_artefact_wavelength")
    if recurrent:
        vetoes.append("recurrent_across_targets")
    if cosmic is not None and cosmic.get("cosmic_ray_driven"):
        vetoes.append("cosmic_ray_single_integration")
    if feature.get("fwhm_samples", 2.0) < 1.0:
        vetoes.append("single_pixel_spike")
    eclipse_tested = disc is not None and phase_class in ("eclipse", "both")
    if not eclipse_tested:
        vetoes.append("insufficient_phase_coverage")
    else:
        d = disc
        vs = d.get("eclipse_vanish_snr", np.nan)
        if not np.isfinite(vs) or vs < c["vanish_snr_interest"]:
            vetoes.append("low_snr")
        lf, lfe = d.get("line_fractional_drop", np.nan), d.get("line_fractional_drop_err", np.nan)
        cf = d.get("continuum_fractional_drop", np.nan)
        cfe = d.get("continuum_fractional_drop_err", np.nan)
        if all(np.isfinite(v) for v in (lf, lfe, cf, cfe)):
            joint = np.hypot(lfe, cfe) + 1e-12
            matches_cont = abs(lf - cf) / joint < c["tracks_continuum_sigma"]
            not_total = abs(lf - 1.0) / (lfe + 1e-12) > c["tracks_continuum_sigma"]
            if matches_cont and not_total:
                vetoes.append("tracks_continuum")
        r, p = d.get("continuum_correlation", np.nan), d.get("continuum_correlation_p", np.nan)
        if (np.isfinite(r) and np.isfinite(p) and abs(r) > c["continuum_corr_max"]
                and p < c["continuum_corr_p_max"] and "tracks_continuum" not in vetoes):
            vetoes.append("tracks_continuum")
        rc = d.get("ramp_correlation", np.nan)
        chi_r, chi_s = d.get("chi2_ramp", np.nan), d.get("chi2_step_predicted", np.nan)
        if np.isfinite(rc) and rc > c["ramp_corr_max"] and np.isfinite(chi_r) \
                and np.isfinite(chi_s) and chi_r <= chi_s:
            vetoes.append("ramp_correlated")
        off = d.get("free_step_offset_integrations", np.nan)
        if np.isfinite(off) and labels is not None and np.isfinite(vs) \
                and vs >= c["vanish_snr_interest"]:
            tol = _timing_tolerance_integrations(labels, c)
            chi_f = d.get("chi2_step_free", np.nan)
            if abs(off) > tol and np.isfinite(chi_f) and np.isfinite(chi_s) \
                    and (chi_s - chi_f) > 9.0:
                vetoes.append("drop_not_at_eclipse")
    if transit is not None and phase_class in ("transit", "both"):
        ex = _num(transit.get("transit_excess_sigma"))
        if np.isfinite(ex) and ex > c["transit_excess_sigma_max"]:
            vetoes.append("transit_inconsistent")
    tier = "none"
    if not vetoes:
        vs = disc.get("eclipse_vanish_snr", np.nan)
        # 'Consistent with zero' is judged on the in-eclipse averaged SPECTRUM
        # when available (bias-free), else on the series mean.
        zin = disc.get("in_eclipse_spectrum_snr", np.nan)
        if not np.isfinite(zin):
            zin = abs(disc.get("in_eclipse_sigma", np.nan))
        ok_cand = (vs >= c["vanish_snr_candidate"]
                   and disc.get("out_positive_snr", np.nan) >= c["out_positive_snr_min"]
                   and zin <= c["in_eclipse_zero_sigma_max"])
        tier = "candidate" if ok_cand else "interest"
    elif vetoes == ["insufficient_phase_coverage"]:
        tier = "watch"
    return {"tier": tier, "vetoes": vetoes, "eclipse_tested": eclipse_tested}


def _num(v) -> float:
    """None / non-numeric (e.g. a JSON-roundtripped NaN) -> nan; else float."""
    if v is None or isinstance(v, (str, bytes, list, dict, tuple)):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _timing_tolerance_integrations(labels: dict, c: dict) -> float:
    ecl = [e for e in labels.get("eclipses", []) if not e.get("unplaceable")]
    cad = labels.get("cadence_days")
    if not ecl or not cad:
        return 3.0
    e = ecl[0]
    tol_days = c["timing_tolerance_ingress_units"] * e["ingress_duration"] + e["timing_sigma"]
    return max(3.0, tol_days / cad)


# --- 7. population-level ---------------------------------------------------------
def recurrent_wavelengths(entries: list[dict], bin_um: float = 0.004,
                          min_targets: int = 3) -> set:
    """Wavelength bins hosting features from >= ``min_targets`` distinct targets.

    The empirical instrumental-recurrence veto (``spectra-triage``): a beacon has
    no reason to repeat at one instrumental wavelength across unrelated targets.
    """
    from collections import defaultdict
    by_bin: dict[int, set] = defaultdict(set)
    for e in entries:
        by_bin[int(round(float(e["wavelength"]) / bin_um))].add(str(e["target"]))
    # A cluster straddling a bin edge must not escape: count each bin together
    # with its two neighbours (a +-bin_um window around the bin).
    return {b for b in by_bin
            if len(by_bin[b] | by_bin.get(b - 1, set()) | by_bin.get(b + 1, set())) >= min_targets}


def is_recurrent(wavelength: float, recurrent_bins: set, bin_um: float = 0.004) -> bool:
    return int(round(float(wavelength) / bin_um)) in recurrent_bins


def bh_fdr(pvalues, m_total: int, alpha: float = 0.05):
    """Benjamini-Hochberg with the FULL trial count.

    ``pvalues`` are the tested features' p-values; ``m_total`` is the number of
    trials actually made (every scanned resolution element of every target),
    which is larger than ``len(pvalues)``.  Returns ``(reject, threshold)``.
    """
    p = np.asarray(pvalues, float)
    reject = np.zeros(p.size, bool)
    m = max(int(m_total), p.size, 1)
    fin = np.flatnonzero(np.isfinite(p))
    if fin.size == 0:
        return reject, float("nan")
    order = fin[np.argsort(p[fin])]
    kmax, thresh = -1, float("nan")
    for rank, j in enumerate(order, start=1):
        if p[j] <= alpha * rank / m:
            kmax, thresh = rank, alpha * rank / m
    if kmax > 0:
        reject[order[:kmax]] = True
    return reject, thresh


def vanish_pvalue(snr: float) -> float:
    """One-sided Gaussian p-value of an eclipse-vanish SNR."""
    return float(_stats.norm.sf(snr)) if np.isfinite(snr) else float("nan")


__all__ = ["VETO_NAMES", "time_average_spectrum", "running_median",
           "local_poly_continuum", "block_noise", "residual_z", "feature_snr_in_mask",
           "narrow_feature_search", "line_flux_series", "eclipse_discriminant",
           "transit_consistency", "known_artefact", "cosmic_ray_driven",
           "assess_feature", "recurrent_wavelengths", "is_recurrent", "bh_fdr",
           "vanish_pvalue"]
