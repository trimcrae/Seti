"""S41 -- the industrial line (``docs/roman.md`` section 2.2).

An **unresolved emission line on a stellar point source** in the Roman G150
grism (1.00-1.93 um) or P127 prism (0.75-1.80 um).  Every optical laser search
to date stops near 0.98 um; the band Roman opens is where our own high-power
lasers live (Nd:YAG 1.064 um, Yb fibre 1.03-1.09, Er fibre 1.53-1.57).  The
search is *blind* over the whole band; a match to ``lines.industrial_lines_um``
is a **flag for the reader, never a filter**.

The detector is the existing LSF-matched filter
(:func:`seti.spectra.detect.find_emission_lines`) with the LSF set from R(lambda)
per sample, followed by the slitless-spectroscopy ledger of vetoes.  Every veto
is a pure function returning ``(bool | None, note)``: ``True`` fires, ``False``
passes, ``None`` means *the test could not be run* on this product -- and a
feature with a ``None`` is tier ``PENDING_<test>``, never a survivor (the TOCSIN
honesty rule: absence of a contamination estimate is not absence of
contamination).

Population-level steps (:func:`assess_features`) reuse the LANTERN machinery:
the same wavelength on >= 3 unrelated sources is instrumental, the same
detector pixel on >= 2 is a defect, and Benjamini-Hochberg is applied with the
full trial count.  A 2-D mode (:func:`dispersed_blob_search`) looks for a
compact PSF-like blob at the position the dispersion solution predicts for
lambda -- how an unresolved line on a faint continuum actually looks in
slitless data.

Pure numpy/scipy; no network; every number comes from ``config/roman.yaml``
(``lines:`` and ``instruments.WFI.dispersers``).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable

import numpy as np
from scipy import stats as _stats

from seti.lantern.line import bh_fdr, is_recurrent, recurrent_wavelengths
from seti.roman.schema import Funnel, Spectrum
from seti.spectra.detect import EmissionLine, estimate_continuum, find_emission_lines

FWHM_PER_SIGMA = 2.3548200450309493      # 2 sqrt(2 ln 2)

# Nominal first-order dispersion (micron per detector pixel).  VERIFY against
# the delivered grism/prism WCS: G150 is quoted at R = 461 lambda per 2 px,
# i.e. a constant ~10.9e-4 um/px across the band; the prism is strongly
# non-linear and 6e-3 um/px is a band-average stand-in only.  ``conf["lines"]
# ["dispersion_um_per_px"]`` overrides these when present.
_NOMINAL_DISPERSION_UM_PER_PX = {"G150": 10.9e-4, "P127": 6.0e-3}
_FALLBACK_R = 100.0

VETO_NAMES = ("known_stellar_line", "zeroth_order_contaminated", "overlap_contaminated",
              "persistence_suspect", "possible_high_z_emitter")
RECURRENCE_VETO_NAMES = ("recurrent_wavelength", "recurrent_pixel")


# --------------------------------------------------------------------------------------
# Config access and wavelength bookkeeping
# --------------------------------------------------------------------------------------

def _lines_conf(conf: dict) -> dict:
    return dict((conf or {}).get("lines") or {})


def _dispersers(conf: dict) -> dict:
    return dict(((conf or {}).get("instruments") or {}).get("WFI", {}).get("dispersers") or {})


def air_to_vacuum_um(um):
    """Air -> vacuum wavelength (micron) with the Edlen-type dispersion of standard air.

    ``n - 1 = 1e-8 (8342.13 + 2406030 / (130 - s^2) + 15997 / (38.9 - s^2))``,
    ``s = 1 / lambda_um``; ``lambda_vac = n lambda_air``.  The industrial line
    list in the config is *already* vacuum; this exists so the reader can check
    a literature (air) wavelength against it -- Nd:YAG at 1064.1 nm in air is
    1064.4 nm in vacuum, which is what the config carries.
    """
    lam = np.asarray(um, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = (1.0 / lam) ** 2
        n = 1.0 + 1e-8 * (8342.13 + 2406030.0 / (130.0 - s2) + 15997.0 / (38.9 - s2))
    out = lam * n
    return float(out) if np.ndim(out) == 0 else out


def industrial_lines(conf: dict) -> list[dict]:
    """The ``lines.industrial_lines_um`` list normalised to ``{name, um_lo, um_hi, kind}``.

    A ``line`` entry has ``um_lo == um_hi``; a ``band`` entry (a tunable family)
    carries its two edges.  Wavelengths are vacuum microns as configured.
    Malformed rows are skipped rather than raised on: a bad list entry must not
    take the blind search down with it.
    """
    out: list[dict] = []
    for row in _lines_conf(conf).get("industrial_lines_um") or []:
        if not isinstance(row, dict) or "um" not in row:
            continue
        um = row["um"]
        try:
            if isinstance(um, (list, tuple)):
                lo, hi = sorted(float(v) for v in um[:2])
            else:
                lo = hi = float(um)
        except (TypeError, ValueError):
            continue
        kind = str(row.get("kind") or ("line" if lo == hi else "band"))
        out.append({"name": str(row.get("name", f"{lo:.4f}")), "um_lo": lo, "um_hi": hi,
                    "kind": kind})
    return out


def stellar_lines(conf: dict) -> list[dict]:
    """``lines.stellar_lines_um`` as ``{name, um}`` rows (vacuum microns)."""
    out = []
    for row in _lines_conf(conf).get("stellar_lines_um") or []:
        if isinstance(row, dict) and "um" in row:
            try:
                out.append({"name": str(row.get("name", "")), "um": float(row["um"])})
            except (TypeError, ValueError):
                continue
    return out


# --------------------------------------------------------------------------------------
# Resolving power and the LSF in samples
# --------------------------------------------------------------------------------------

def resolving_power_with_note(spec: Spectrum, conf: dict) -> tuple[np.ndarray, str]:
    """Per-sample R and a provenance note.

    Order of trust: the product's own ``resolving_power`` (scalar or per-sample
    array), then the config disperser for ``spec.mode`` (G150: ``R_per_um *
    lambda``; P127: linear in lambda between ``R_range`` over ``range_um``),
    then a flat fallback of 100 -- never a ``ValueError``: a spectrum in an
    unknown mode is still searched, and the note says the LSF was guessed.
    """
    wl = np.asarray(spec.wavelength_um, dtype=float)
    n = wl.size
    if spec.resolving_power is not None:
        r = np.asarray(spec.resolving_power, dtype=float)
        if r.ndim == 0:
            return np.full(n, float(r)), "spectrum:scalar"
        if r.size == n:
            return r.copy(), "spectrum:array"
    disp = _dispersers(conf).get(str(spec.mode).upper())
    if isinstance(disp, dict):
        if "R_per_um" in disp:
            return float(disp["R_per_um"]) * wl, f"config:{str(spec.mode).upper()}:R_per_um"
        if "R_range" in disp and "range_um" in disp:
            lo, hi = (float(v) for v in disp["range_um"][:2])
            r0, r1 = (float(v) for v in disp["R_range"][:2])
            return np.interp(wl, [lo, hi], [r0, r1]), f"config:{str(spec.mode).upper()}:R_range"
    return np.full(n, _FALLBACK_R), f"fallback:R={_FALLBACK_R:g} (mode {spec.mode!r} unknown)"


def resolving_power(spec: Spectrum, conf: dict) -> np.ndarray:
    """Per-sample resolving power (see :func:`resolving_power_with_note`)."""
    return resolving_power_with_note(spec, conf)[0]


def sample_spacing_um(spec: Spectrum) -> np.ndarray:
    """|d lambda| per sample, with zero/NaN spacings replaced by the median (a
    duplicated wavelength must not make the LSF infinite)."""
    wl = np.asarray(spec.wavelength_um, dtype=float)
    if wl.size < 2:
        return np.ones(wl.size)
    d = np.abs(np.gradient(wl))
    ok = np.isfinite(d) & (d > 0)
    fill = float(np.median(d[ok])) if ok.any() else 1.0
    return np.where(ok, d, fill)


def lsf_sigma_pix(spec: Spectrum, conf: dict) -> np.ndarray:
    """Gaussian LSF sigma in *samples* per sample: ``FWHM = lambda / R``,
    ``sigma_pix = FWHM / (2.3548 d_lambda)``."""
    wl = np.asarray(spec.wavelength_um, dtype=float)
    r = resolving_power(spec, conf)
    with np.errstate(divide="ignore", invalid="ignore"):
        fwhm_um = wl / r
    sig = fwhm_um / (FWHM_PER_SIGMA * sample_spacing_um(spec))
    ok = np.isfinite(sig) & (sig > 0)
    fill = float(np.median(sig[ok])) if ok.any() else 1.0
    return np.where(ok, sig, fill)


# --------------------------------------------------------------------------------------
# Width measurement (the resolved / unresolved discriminant)
# --------------------------------------------------------------------------------------

def _fit_fwhm_samples(resid: np.ndarray, err: np.ndarray, i: int, fwhm_lsf: float,
                      fwhm_max: float) -> tuple[float, float, float]:
    """Gaussian FWHM (samples) of the residual peak at ``i``: ``(best, lo, hi)``.

    Weighted least squares of ``a * template + b + c * x`` over a window of
    ``+-(1.5 fwhm_max + 2)`` samples, on a geometric grid of widths from 0.7
    samples to ``fwhm_max`` and sub-sample centres.  ``[lo, hi]`` is the range
    of grid widths within ``delta chi^2 <= 4`` (2 sigma) of the best.  A
    continuous template width is what the resel test needs: the integer
    half-maximum count (``EmissionLine.fwhm_pix``) is quantised at 1, 2, 3 for
    a Nyquist-sampled line whose neighbours sit at exactly half peak, and the
    noise-weighted second moment (``width_ratio``) is inflated by window
    noise; and on the under-sampled prism (FWHM ~1.6 samples) even the fitted
    point estimate wanders, so the width guards act on what the range
    EXCLUDES.  All three measures are reported for the reader.
    """
    n = resid.size
    reach = int(math.ceil(1.5 * fwhm_max)) + 2
    lo, hi = max(0, i - reach), min(n, i + reach + 1)
    y = resid[lo:hi]
    e = err[lo:hi]
    ok = np.isfinite(y) & np.isfinite(e) & (e > 0)
    nan3 = (float("nan"),) * 3
    if ok.sum() < 6:
        return nan3
    x = np.arange(lo, hi, dtype=float)[ok]
    y = y[ok]
    w = 1.0 / e[ok] ** 2
    sw = np.sqrt(w)
    xl = (x - i) / max(reach, 1)
    grid = np.geomspace(0.7, max(fwhm_max, 1.0), 48)
    best_chi, best_fwhm = np.inf, float("nan")
    chi_by_fwhm: dict[float, float] = {}
    for fwhm in grid:
        sig = fwhm / FWHM_PER_SIGMA
        cbest = np.inf
        for c0 in (i - 0.5, i - 0.25, i, i + 0.25, i + 0.5):
            tmpl = np.exp(-0.5 * ((x - c0) / sig) ** 2)
            a_mat = np.column_stack([tmpl, np.ones_like(x), xl])
            coef, *_ = np.linalg.lstsq(a_mat * sw[:, None], y * sw, rcond=None)
            if coef[0] <= 0:
                continue
            chi2 = float(np.sum(w * (y - a_mat @ coef) ** 2))
            cbest = min(cbest, chi2)
            if chi2 < best_chi:
                best_chi, best_fwhm = chi2, float(fwhm)
        chi_by_fwhm[float(fwhm)] = cbest
    if not np.isfinite(best_chi):
        return nan3
    consistent = [f for f, c2 in chi_by_fwhm.items() if c2 <= best_chi + 4.0]
    return best_fwhm, float(min(consistent)), float(max(consistent))


def _continuum_snr(flux: np.ndarray, err: np.ndarray, cont: np.ndarray, i: int,
                   half_window: int, core: int) -> float:
    """Per-sample S/N of the local continuum around ``i``: the median continuum
    level over ``+-half_window`` (the line core ``+-core`` excluded) divided by
    the larger of the median propagated error and the robust scatter of the
    residual -- an under-reported error must not manufacture a stellar continuum
    where there is only sky."""
    n = flux.size
    lo, hi = max(0, i - half_window), min(n, i + half_window + 1)
    idx = np.arange(lo, hi)
    idx = idx[np.abs(idx - i) > core]
    f, e, c = flux[idx], err[idx], cont[idx]
    ok = np.isfinite(f) & np.isfinite(c)
    if ok.sum() < 5:
        return float("nan")
    level = float(np.median(c[ok]))
    resid = f[ok] - c[ok]
    mad = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    e_ok = e[ok]
    e_ok = e_ok[np.isfinite(e_ok) & (e_ok > 0)]
    med_err = float(np.median(e_ok)) if e_ok.size else 0.0
    noise = max(mad, med_err)
    if not noise > 0:
        return float("nan")
    return level / noise


# --------------------------------------------------------------------------------------
# The 1-D search
# --------------------------------------------------------------------------------------

def _chunk_plan(sig: np.ndarray, pad: int, ratio_max: float = 1.3) -> list[tuple[int, int]]:
    """Index ranges ``[lo, hi)`` (cores) over which the LSF sigma in samples
    varies by less than ``ratio_max``; a single chunk when it already does."""
    n = sig.size
    fin = sig[np.isfinite(sig) & (sig > 0)]
    if fin.size == 0 or n <= 4 * pad:
        return [(0, n)]
    spread = float(fin.max() / fin.min())
    k = 1 if spread <= ratio_max else int(math.ceil(math.log(spread) / math.log(ratio_max)))
    k = max(1, min(k, n // max(2 * pad, 1)))
    edges = np.linspace(0, n, k + 1).astype(int)
    return [(int(edges[j]), int(edges[j + 1])) for j in range(k)]


def _run_matched_filter(spec: Spectrum, sig: np.ndarray, c: dict) -> list[tuple[EmissionLine, float]]:
    """``find_emission_lines`` over chunks of near-constant LSF; each feature is
    returned with the LSF sigma actually used, indices in the full spectrum."""
    wl = np.asarray(spec.wavelength_um, dtype=float)
    f = np.asarray(spec.flux, dtype=float)
    e = np.asarray(spec.flux_err, dtype=float)
    n = f.size
    window = int(c.get("continuum_window", 61))
    pad = window
    found: list[tuple[EmissionLine, float]] = []
    seen: set[int] = set()
    for lo, hi in _chunk_plan(sig, pad):
        a, b = max(0, lo - pad), min(n, hi + pad)
        s_med = float(np.median(sig[lo:hi]))
        lines = find_emission_lines(wl[a:b], f[a:b], e[a:b], lsf_sigma_pix=s_med,
                                    snr_min=float(c.get("snr_min", 8.0)),
                                    continuum_window=window)
        for ln in lines:
            j = ln.index + a
            if lo <= j < hi and j not in seen:
                seen.add(j)
                ln.index = j
                found.append((ln, s_med))
    return found


def find_unresolved_lines(spec: Spectrum, conf: dict, funnel: Funnel | None = None) -> list[dict]:
    """Unresolved, interior, significant emission features of one spectrum.

    The matched filter is :func:`seti.spectra.detect.find_emission_lines`, which
    takes one LSF sigma; the spectrum is therefore searched in chunks inside
    which the per-sample sigma varies by < 30 % (one chunk for G150, whose
    R = 461 lambda gives a constant 2-sample FWHM; a few for a prism on a
    non-linear grid), each with its chunk-median sigma and a
    ``continuum_window`` overlap so chunk edges are not spectrum edges.  Per
    feature the width is re-measured by a Gaussian template fit against the
    *local* LSF (``width_resel = FWHM_fit / (2.3548 sigma_pix[i])``) and the
    feature is kept when the 2-sigma-consistent width range overlaps
    ``[min_width_resel, max_width_resel]`` (see :func:`_fit_fwhm_samples`), it lies ``>= min_from_edge_samples`` from either end, and ``snr >= snr_min``.
    ``continuum_snr`` is the local per-sample continuum S/N over
    ``+-continuum_window`` -- the stellar-continuum requirement lives in
    :func:`single_line_emitter_suspect`, not here, so the reader sees the
    continuum-free features too (as vetoed rows).
    """
    c = _lines_conf(conf)
    fn = funnel if funnel is not None else Funnel()
    n = spec.n
    edge = int(c.get("min_from_edge_samples", 8))
    snr_min = float(c.get("snr_min", 8.0))
    w_lo = float(c.get("min_width_resel", 0.6))
    w_hi = float(c.get("max_width_resel", 2.0))
    window = int(c.get("continuum_window", 61))
    fn.bump("spectra", 1)
    fn.bump("samples", n)
    fn.bump("samples_searched", max(0, n - 2 * edge))
    if n < max(5, 2 * edge + 3):
        fn.note(f"{spec.source_id}: {n} samples, too short to search")
        return []
    wl = np.asarray(spec.wavelength_um, dtype=float)
    f = np.asarray(spec.flux, dtype=float)
    e = np.asarray(spec.flux_err, dtype=float)
    e = np.where(np.isfinite(e) & (e > 0), e, np.inf)
    sig = lsf_sigma_pix(spec, conf)
    r_arr = resolving_power(spec, conf)
    cont = estimate_continuum(f, window)
    resid = f - cont
    raw = _run_matched_filter(spec, sig, c)
    fn.bump("features_raw", len(raw))
    out: list[dict] = []
    for ln, s_used in raw:
        i = int(ln.index)
        if i < edge or i > n - 1 - edge:
            fn.reject("edge")
            continue
        if not np.isfinite(ln.significance) or ln.significance < snr_min:
            fn.reject("snr_below_min")
            continue
        s_loc = float(sig[i])
        fwhm_lsf = FWHM_PER_SIGMA * s_loc
        fwhm_fit, fwhm_lo, fwhm_hi = _fit_fwhm_samples(resid, e, i, fwhm_lsf, 4.0 * w_hi * fwhm_lsf)
        if not (np.isfinite(fwhm_fit) and fwhm_lsf > 0):
            fn.reject("width_unmeasurable")
            continue
        width_resel = fwhm_fit / fwhm_lsf
        # The guards act on the 2-sigma-consistent range: too narrow (a hot
        # pixel / cosmic ray) when even min_width_resel is excluded, resolved
        # when every width up to max_width_resel is excluded.
        if fwhm_hi / fwhm_lsf < w_lo or fwhm_lo / fwhm_lsf > w_hi:
            fn.reject("width_out_of_range")
            continue
        core = int(math.ceil(2.0 * fwhm_lsf * w_hi))
        c_snr = _continuum_snr(f, e, cont, i, window, core)
        feat = {
            "lambda_um": float(wl[i]), "snr": float(ln.significance),
            "width_resel": float(width_resel), "sample_index": i,
            "continuum_snr": float(c_snr) if np.isfinite(c_snr) else None,
            "trace_pixel": (float(spec.trace_pixel[i]) if spec.trace_pixel is not None else None),
            "fwhm_samples": float(fwhm_fit),
            "width_resel_range": [float(fwhm_lo / fwhm_lsf), float(fwhm_hi / fwhm_lsf)],
            "lsf_sigma_pix": s_loc,
            "lsf_sigma_pix_used": float(s_used),
            "resel_um": float(wl[i] / r_arr[i]) if r_arr[i] > 0 else None,
            "amplitude": float(ln.amplitude), "ew_um": float(ln.ew),
            "moment_width_ratio": float(ln.width_ratio),
            "fwhm_halfmax_samples": float(ln.fwhm_pix),
            "min_adjacent": float(ln.min_adjacent), "asymmetry": float(ln.asymmetry),
            "n_samples_searched": int(max(0, n - 2 * edge)),
        }
        out.append(feat)
    fn.bump("features_kept", len(out))
    out.sort(key=lambda d: -d["snr"])
    return out


# --------------------------------------------------------------------------------------
# Vetoes: pure functions returning (bool | None, note)
# --------------------------------------------------------------------------------------

def is_stellar_line(lambda_um: float, conf: dict) -> tuple[bool, str]:
    """Within ``stellar_line_tolerance_um`` of a photospheric / circumstellar line."""
    tol = float(_lines_conf(conf).get("stellar_line_tolerance_um", 0.004))
    best = None
    for row in stellar_lines(conf):
        d = abs(float(lambda_um) - row["um"])
        if d <= tol and (best is None or d < best[0]):
            best = (d, row["name"])
    if best is not None:
        return True, f"known_stellar_line: {best[1]} (|d lambda| = {best[0]:.4f} um)"
    return False, "no stellar line within tolerance"


def industrial_match(lambda_um: float, conf: dict, resel_um: float) -> str | None:
    """Name of the industrial line / band the wavelength falls in, else ``None``.

    A ``line`` entry matches within ``match_tolerance_resel`` resolution elements;
    a ``band`` entry matches when the wavelength lies inside it (padded by the
    same tolerance).  Line matches take precedence over bands -- Nd:YAG 1.064 um
    sits inside the Yb-fibre band and the reader wants the specific name.  This
    is a FLAG for the reader, never a filter on the blind search.
    """
    c = _lines_conf(conf)
    tol = float(c.get("match_tolerance_resel", 1.5)) * float(resel_um or 0.0)
    lam = float(lambda_um)
    rows = industrial_lines(conf)
    for row in [r for r in rows if r["kind"] == "line"] + [r for r in rows if r["kind"] != "line"]:
        if row["um_lo"] - tol <= lam <= row["um_hi"] + tol:
            return row["name"]
    return None


def _trace_y(spec: Spectrum) -> float | None:
    v = (spec.meta or {}).get("trace_y")
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def zeroth_order_contaminated(feature: dict, spec: Spectrum, neighbours: list[dict] | None,
                              conf: dict) -> tuple[bool | None, str]:
    """Does a neighbour's zeroth-order image land on the feature's trace pixel?

    ``neighbours`` are direct-image sources ``{x, y, mag}`` in detector pixels;
    the zeroth order sits ``zeroth_order_offset_px[mode]`` from the direct image
    along the dispersion axis.  A hit within ``zeroth_order_tolerance_px`` along
    the trace and ``trace_cross_dispersion_px`` across it is contamination.
    Without ``spec.trace_pixel``, ``spec.meta['trace_y']`` or a neighbour list
    the test cannot run: ``(None, "zeroth_order_test_not_run")``.
    """
    c = _lines_conf(conf)
    tp = feature.get("trace_pixel")
    ty = _trace_y(spec)
    offsets = c.get("zeroth_order_offset_px") or {}
    off = offsets.get(str(spec.mode).upper(), offsets.get(str(spec.mode)))
    if tp is None or neighbours is None or ty is None or off is None:
        why = ("no trace_pixel" if tp is None else "no neighbours" if neighbours is None
               else "no trace_y" if ty is None else f"no zeroth_order_offset_px for {spec.mode}")
        return None, f"zeroth_order_test_not_run ({why})"
    tol_x = float(c.get("zeroth_order_tolerance_px", 6))
    tol_y = float(c.get("trace_cross_dispersion_px", 4))
    for nb in neighbours:
        try:
            x0 = float(nb["x"]) + float(off)
            y0 = float(nb["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if abs(x0 - float(tp)) <= tol_x and abs(y0 - ty) <= tol_y:
            mag = nb.get("mag")
            return True, (f"zeroth order of neighbour at ({nb['x']}, {nb['y']}, mag {mag}) "
                          f"lands {x0 - float(tp):+.1f} px from the feature")
    return False, f"no zeroth order within {tol_x:g} x {tol_y:g} px of the feature ({len(neighbours)} neighbours)"


def overlap_contaminated(feature: dict, spec: Spectrum, conf: dict) -> tuple[bool | None, str]:
    """Pipeline contamination estimate at the feature above half the flux -> True.

    ``spec.contam`` is the slitless overlap model per sample when the pipeline
    provides one; ``None`` means ``(None, "overlap_test_not_run")``.
    """
    if spec.contam is None:
        return None, "overlap_test_not_run"
    i = int(feature["sample_index"])
    frac_max = float(_lines_conf(conf).get("overlap_contam_fraction_max", 0.5))
    contam = float(spec.contam[i])
    flux = float(spec.flux[i])
    if not np.isfinite(contam):
        return None, "overlap_test_not_run (contam not finite at the feature)"
    denom = max(abs(flux), 1e-30)
    frac = contam / denom
    if frac > frac_max:
        return True, f"contam/flux = {frac:.2f} > {frac_max:g} at the feature"
    return False, f"contam/flux = {frac:.2f} at the feature"


def persistence_suspect(feature: dict, spec: Spectrum,
                        prior_bright_pixels: list | None) -> tuple[bool | None, str]:
    """Same detector pixel as a bright source in the previous exposure -> True.

    ``prior_bright_pixels`` is a list of ``{x, y[, detector]}`` dicts or
    ``(x, y)`` pairs; "same pixel" is within one pixel in each axis (the bright
    source's core), the cross-dispersion comparison only when ``trace_y`` is
    known.  ``None`` without a prior list or without ``trace_pixel``.
    """
    if prior_bright_pixels is None:
        return None, "persistence_test_not_run (no prior exposure list)"
    tp = feature.get("trace_pixel")
    if tp is None:
        return None, "persistence_test_not_run (no trace_pixel)"
    ty = _trace_y(spec)
    for p in prior_bright_pixels:
        try:
            if isinstance(p, dict):
                x, y = float(p["x"]), float(p["y"])
                det = p.get("detector")
            else:
                x, y = float(p[0]), float(p[1])
                det = None
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if det is not None and spec.detector is not None and str(det) != str(spec.detector):
            continue
        if abs(x - float(tp)) <= 1.0 and (ty is None or abs(y - ty) <= 1.0):
            return True, f"bright source at ({x:g}, {y:g}) in the previous exposure on this pixel"
    return False, f"no prior bright source on the feature's pixel ({len(prior_bright_pixels)} checked)"


def single_line_emitter_suspect(feature: dict, spec: Spectrum, conf: dict) -> tuple[bool | None, str]:
    """The classic single-line emitter -- a high-z Halpha/[O III]/Lyalpha galaxy.

    A laser claim needs BOTH a stellar continuum (``continuum_snr >=
    min_continuum_snr_for_stellar``) AND a point source in the direct image.
    True when either fails; ``None`` when the continuum is fine but the
    morphology is unknown (``is_point_source is None``: "morphology_unknown").
    """
    c_min = float(_lines_conf(conf).get("min_continuum_snr_for_stellar", 5.0))
    c_snr = feature.get("continuum_snr")
    cont_ok = c_snr is not None and np.isfinite(c_snr) and c_snr >= c_min
    if not cont_ok:
        shown = "nan" if c_snr is None else f"{c_snr:.1f}"
        return True, f"possible_high_z_emitter: continuum S/N {shown} < {c_min:g}"
    if spec.is_point_source is False:
        return True, "possible_high_z_emitter: not a point source in the direct image"
    if spec.is_point_source is None:
        return None, "morphology_unknown"
    return False, f"stellar continuum (S/N {c_snr:.1f}) on a point source"


# --------------------------------------------------------------------------------------
# One spectrum through the ledger
# --------------------------------------------------------------------------------------

def _pvalues(snr: float, n_trials: int) -> tuple[float, float]:
    p1 = float(_stats.norm.sf(snr)) if np.isfinite(snr) else float("nan")
    return p1, float(min(1.0, p1 * max(int(n_trials), 1)))


def screen_spectrum(spec: Spectrum, conf: dict, neighbours: list[dict] | None = None,
                    prior_bright_pixels: list | None = None) -> dict:
    """Search one spectrum and run every veto on every feature.

    A feature *survives* only when every veto that RAN returned ``False``.  A
    veto that returned ``None`` is listed under ``tests_not_run`` and the feature
    is tier ``PENDING_<test>`` -- not a survivor, not vetoed, waiting for the
    product that carries the missing field.  ``industrial_flag`` is attached
    for the reader and never changes the verdict.
    """
    fn = Funnel()
    r_note = resolving_power_with_note(spec, conf)[1]
    if r_note.startswith("fallback"):
        fn.note(f"{spec.source_id}: {r_note}")
    feats = find_unresolved_lines(spec, conf, fn)
    out_feats = []
    n_surv = n_pend = 0
    for feat in feats:
        vetoes: dict[str, bool | None] = {}
        notes: dict[str, str] = {}
        vetoes["known_stellar_line"], notes["known_stellar_line"] = is_stellar_line(feat["lambda_um"], conf)
        vetoes["zeroth_order_contaminated"], notes["zeroth_order_contaminated"] = \
            zeroth_order_contaminated(feat, spec, neighbours, conf)
        vetoes["overlap_contaminated"], notes["overlap_contaminated"] = overlap_contaminated(feat, spec, conf)
        vetoes["persistence_suspect"], notes["persistence_suspect"] = \
            persistence_suspect(feat, spec, prior_bright_pixels)
        vetoes["possible_high_z_emitter"], notes["possible_high_z_emitter"] = \
            single_line_emitter_suspect(feat, spec, conf)
        fired = [k for k, v in vetoes.items() if v is True]
        not_run = [k for k, v in vetoes.items() if v is None]
        for k in fired:
            fn.reject(k)
        survives = not fired and not not_run
        if survives:
            tier = "UNRESOLVED_LINE_SURVIVOR"
            n_surv += 1
        elif fired:
            tier = "VETOED"
        else:
            tier = "PENDING_" + "+".join(not_run)
            n_pend += 1
            for k in not_run:
                fn.bump(f"pending_{k}")
        p1, ps = _pvalues(feat["snr"], feat["n_samples_searched"])
        rec = dict(feat)
        rec.update({
            "source_id": spec.source_id, "mode": spec.mode, "detector": spec.detector,
            "industrial_flag": industrial_match(feat["lambda_um"], conf, feat.get("resel_um") or 0.0),
            "vetoes": vetoes, "veto_notes": notes, "tests_not_run": not_run,
            "survives": bool(survives), "tier": tier,
            "reasons": fired + [f"not_run:{k}" for k in not_run],
            "p_single": p1, "p_source": ps,
        })
        out_feats.append(rec)
    fn.bump("survivors", n_surv)
    fn.bump("pending", n_pend)
    if not out_feats:
        status = "no_features"
    elif n_surv:
        status = "survivor"
    elif n_pend:
        status = "pending"
    else:
        status = "vetoed"
    return {"source_id": spec.source_id, "mode": spec.mode, "detector": spec.detector,
            "n_samples": spec.n, "resolving_power_note": r_note,
            "n_features": len(out_feats), "n_survivors": n_surv, "n_pending": n_pend,
            "features": out_feats, "funnel": fn.as_dict(), "status": status}


# --------------------------------------------------------------------------------------
# Population level
# --------------------------------------------------------------------------------------

def _recurrent_pixels(feats: list[dict], min_sources: int) -> set[tuple[str, int]]:
    """``(detector, pixel-bin)`` keys hosting features from >= ``min_sources``
    distinct sources within +-1 px -- :func:`recurrent_wavelengths` reused with
    a 1-pixel bin, per detector."""
    by_det: dict[str, list[dict]] = defaultdict(list)
    for f in feats:
        if f.get("trace_pixel") is None:
            continue
        by_det[str(f.get("detector"))].append({"wavelength": float(f["trace_pixel"]),
                                                "target": f["source_id"]})
    out: set[tuple[str, int]] = set()
    for det, rows in by_det.items():
        for b in recurrent_wavelengths(rows, bin_um=1.0, min_targets=min_sources):
            out.add((det, int(b)))
    return out


def assess_features(records: list[dict], conf: dict) -> dict:
    """Cross-source recurrence, FDR, and the population tier.

    Recurrence runs over EVERY detected feature (vetoed ones included -- an
    instrumental wavelength shows up regardless of what else is wrong with the
    spectrum): the same wavelength bin (``recurrence_bin_um``) on
    ``>= recurrence_min_sources`` distinct sources is ``recurrent_wavelength``;
    the same detector pixel (+-1) on ``>= pixel_recurrence_min_sources`` is
    ``recurrent_pixel``.  Survivors of the per-spectrum ledger that are not
    recurrent are then tested with Benjamini-Hochberg on ``p_source`` (the
    Gaussian tail of the S/N times the samples searched in that spectrum), with
    the trial count equal to the number of spectra searched.  Tier is
    ``INDUSTRIAL_LINE_CANDIDATES_PENDING_VET`` when any survivor passes FDR,
    else ``NO_CANDIDATE``; pending features are counted, never promoted.  The
    input records are not mutated: ``features`` in the result carries every
    feature with the recurrence vetoes added.
    """
    c = _lines_conf(conf)
    fn = Funnel()
    feats: list[dict] = []
    for rec in records or []:
        try:
            fn.merge(Funnel(**rec.get("funnel", {})))
        except TypeError:
            pass
        for f in rec.get("features") or []:
            g = dict(f)
            g.setdefault("source_id", rec.get("source_id"))
            g.setdefault("detector", rec.get("detector"))
            g["vetoes"] = dict(g.get("vetoes") or {})
            g["reasons"] = list(g.get("reasons") or [])
            feats.append(g)
    n_sources = len({str(r.get("source_id")) for r in (records or [])})
    bin_um = float(c.get("recurrence_bin_um", 0.003))
    rec_bins = recurrent_wavelengths(
        [{"wavelength": f["lambda_um"], "target": f["source_id"]} for f in feats],
        bin_um=bin_um, min_targets=int(c.get("recurrence_min_sources", 3)))
    rec_px = _recurrent_pixels(feats, int(c.get("pixel_recurrence_min_sources", 2)))
    flag_counts: dict[str, int] = defaultdict(int)
    survivors: list[dict] = []
    n_pending = n_pre = 0
    for f in feats:
        rw = is_recurrent(f["lambda_um"], rec_bins, bin_um)
        rp = (f.get("trace_pixel") is not None
              and (str(f.get("detector")), int(round(float(f["trace_pixel"])))) in rec_px)
        f["vetoes"]["recurrent_wavelength"] = bool(rw)
        f["vetoes"]["recurrent_pixel"] = bool(rp)
        if f.get("industrial_flag"):
            flag_counts[str(f["industrial_flag"])] += 1
        was_survivor = bool(f.get("survives"))
        n_pre += int(was_survivor)
        if rw:
            f["reasons"].append("recurrent_wavelength")
            fn.reject("recurrent_wavelength")
        if rp:
            f["reasons"].append("recurrent_pixel")
            fn.reject("recurrent_pixel")
        if rw or rp:
            f["survives"] = False
            if not str(f.get("tier", "")).startswith("PENDING"):
                f["tier"] = "VETOED"
            elif was_survivor:
                f["tier"] = "VETOED"
        if f.get("tests_not_run") and not (rw or rp):
            n_pending += 1
        if f.get("survives"):
            survivors.append(f)
    m_total = max(len(records or []), 1)
    alpha = float(c.get("fdr_alpha", 0.05))
    p = np.array([float(s.get("p_source", np.nan)) for s in survivors], dtype=float)
    reject, thresh = bh_fdr(p, m_total, alpha) if p.size else (np.zeros(0, bool), float("nan"))
    for s, r in zip(survivors, reject, strict=True):
        s["fdr_pass"] = bool(r)
    passing = [s for s in survivors if s.get("fdr_pass")]
    passing.sort(key=lambda d: -float(d.get("snr", 0.0)))
    tier = "INDUSTRIAL_LINE_CANDIDATES_PENDING_VET" if passing else "NO_CANDIDATE"
    fn.bump("sources_assessed", n_sources)
    fn.bump("survivors_after_recurrence", len(survivors))
    fn.bump("survivors_after_fdr", len(passing))
    return {
        "tier": tier, "n_sources": n_sources, "n_features": len(feats),
        "n_survivors_pre_recurrence": n_pre, "n_survivors": len(survivors),
        "n_fdr_pass": len(passing), "n_pending": n_pending,
        "recurrent_wavelength_bins_um": sorted(float(b * bin_um) for b in rec_bins),
        "recurrent_pixels": sorted([list(k) for k in rec_px]),
        "fdr": {"alpha": alpha, "threshold": None if not np.isfinite(thresh) else float(thresh),
                "m_total": int(m_total), "n_tested": int(p.size), "n_reject": int(len(passing))},
        "industrial_flag_counts": dict(sorted(flag_counts.items())),
        "top_survivors": passing[:20],
        "features": feats,
        "funnel": fn.as_dict(),
    }


# --------------------------------------------------------------------------------------
# 2-D mode: a compact blob where the dispersion solution puts lambda
# --------------------------------------------------------------------------------------

def linear_dispersion(mode: str, conf: dict) -> Callable[[float], tuple[float, float]]:
    """``lambda_um -> (dx, dy)`` pixel offset from the direct image, linear in lambda.

    Nominal values, VERIFY against the delivered grism WCS: the trace starts at
    the direct-image position (``trace_origin_px``, default 0) at the blue edge
    of ``range_um`` and runs along +x at ``dispersion_um_per_px`` (G150 10.9e-4
    um/px from R = 461 lambda per 2 px; P127 6e-3 um/px, a band average for a
    strongly non-linear prism).  ``dy`` is 0.  Both are read from
    ``conf['lines']`` when present.  The real solution is a field-dependent
    polynomial; this is the stand-in the synthetic tests run on.
    """
    m = str(mode).upper()
    c = _lines_conf(conf)
    disp_um = float((c.get("dispersion_um_per_px") or {}).get(m, _NOMINAL_DISPERSION_UM_PER_PX.get(m, 1e-3)))
    origin = float((c.get("trace_origin_px") or {}).get(m, 0.0))
    d = _dispersers(conf).get(m) or {}
    lam0 = float(d["range_um"][0]) if "range_um" in d else 1.0

    def _disp(lam_um: float) -> tuple[float, float]:
        return origin + (float(lam_um) - lam0) / disp_um, 0.0

    return _disp


def _psf_stamp(px: float, py: float, sigma: float, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * (((xs - px) ** 2 + (ys - py) ** 2) / sigma ** 2))


def dispersed_blob_search(image2d: np.ndarray, star_xy: tuple[float, float],
                          dispersion: Callable[[float], tuple[float, float]],
                          lambdas_um, fwhm_px: float, conf: dict, noise=None, *,
                          resel_um: float | None = None) -> list[dict]:
    """PSF-like blobs along one star's trace at the positions predicted for each lambda.

    For every lambda the direct-image position plus ``dispersion(lambda)`` is
    the predicted centre.  The local background is a *profile across the
    trace*: each pixel of the 5x5 box is compared with the median of the side
    pixels (4-8 px along the trace on either side) in the same unit-wide
    cross-dispersion bin.  On a slitless trace the continuum runs through the box,
    so a circular annulus -- or one median over the whole side region, which
    the off-trace pixels dominate -- leaves the continuum in the residual as a
    streak; the row-wise profile removes it whatever the trace angle.  The
    flux is the PSF-weighted sum over the box, its S/N against ``noise``
    (scalar, per-pixel array, or -- when ``None`` -- the robust scatter of the
    side pixels about their profile), and the compactness is the ratio of the
    background-subtracted second moment to the PSF's own over the same box.
    Blobs with ``snr >= blob_snr_min`` and ``compactness <= blob_compactness_max``
    are returned with the industrial flag attached (matched within
    ``resel_um``, default twice the lambda grid step).  Adjacent lambdas that
    fall on the same peak are merged to the highest S/N.
    """
    img = np.asarray(image2d, dtype=float)
    ny, nx = img.shape
    c = _lines_conf(conf)
    snr_min = float(c.get("blob_snr_min", 8.0))
    comp_max = float(c.get("blob_compactness_max", 1.5))
    sigma = max(float(fwhm_px), 0.5) / FWHM_PER_SIGMA
    noise_arr = None if noise is None else np.broadcast_to(np.asarray(noise, dtype=float), img.shape)
    x_star, y_star = float(star_xy[0]), float(star_xy[1])
    lam_grid = np.atleast_1d(np.asarray(lambdas_um, dtype=float))
    if resel_um is None:
        resel_um = 2.0 * float(np.median(np.abs(np.diff(lam_grid)))) if lam_grid.size > 1 else 0.0
    rc = 8
    hits: list[dict] = []
    for lam in lam_grid:
        dx, dy = dispersion(float(lam))
        px, py = x_star + float(dx), y_star + float(dy)
        ix, iy = int(round(px)), int(round(py))
        if ix - rc < 0 or ix + rc >= nx or iy - rc < 0 or iy + rc >= ny:
            continue
        h = 1e-3
        ax, ay = dispersion(float(lam) + h)
        bx, by = dispersion(float(lam) - h)
        ux, uy = float(ax - bx), float(ay - by)
        norm = math.hypot(ux, uy)
        ux, uy = (ux / norm, uy / norm) if norm > 0 else (1.0, 0.0)
        ys, xs = np.mgrid[iy - rc: iy + rc + 1, ix - rc: ix + rc + 1]
        cut = img[iy - rc: iy + rc + 1, ix - rc: ix + rc + 1]
        s = (xs - px) * ux + (ys - py) * uy
        t = -(xs - px) * uy + (ys - py) * ux
        box = (np.abs(xs - ix) <= 2) & (np.abs(ys - iy) <= 2)
        side = (np.abs(s) >= 4.0) & (np.abs(s) <= 8.0) & (np.abs(t) <= 2.5) & ~box
        side &= np.isfinite(cut)
        if side.sum() < 8:
            continue
        side_vals, side_bin = cut[side], np.round(t[side]).astype(int)
        bkg_all = float(np.median(side_vals))
        # Background profile across the trace: one median of the side pixels
        # per unit-wide cross-dispersion bin, applied to the box pixels in the
        # same bin (a bin with fewer than 4 side pixels falls back to the
        # overall side median).
        prof_by_bin: dict[int, float] = {}
        for b in np.unique(side_bin):
            row = side_vals[side_bin == b]
            if row.size >= 4:
                prof_by_bin[int(b)] = float(np.median(row))
        box_bin = np.round(t[box]).astype(int)
        bkg = np.array([prof_by_bin.get(int(b), bkg_all) for b in box_bin])
        if noise_arr is not None:
            sig_px = noise_arr[iy - rc: iy + rc + 1, ix - rc: ix + rc + 1][box]
        else:
            prof = np.array([prof_by_bin.get(int(b), bkg_all) for b in side_bin])
            mad = 1.4826 * float(np.median(np.abs(side_vals - prof)))
            sig_px = np.full(int(box.sum()), max(mad, 1e-12))
        w = _psf_stamp(px, py, sigma, xs, ys)[box]
        d = cut[box] - bkg
        ok = np.isfinite(d) & np.isfinite(sig_px) & (sig_px > 0)
        if ok.sum() < 9:
            continue
        w, d, sig_px = w[ok], d[ok], sig_px[ok]
        num = float(np.sum(w * d))
        snr = num / float(np.sqrt(np.sum(w ** 2 * sig_px ** 2)))
        amp = num / float(np.sum(w ** 2))
        flux = amp * 2.0 * math.pi * sigma ** 2
        r2 = ((xs - px) ** 2 + (ys - py) ** 2)[box][ok]
        dpos = np.clip(d, 0.0, None)
        m2_psf = float(np.sum(w * r2) / np.sum(w))
        m2 = float(np.sum(dpos * r2) / np.sum(dpos)) if np.sum(dpos) > 0 else float("inf")
        compact = m2 / m2_psf if m2_psf > 0 else float("inf")
        if snr >= snr_min and compact <= comp_max:
            hits.append({"lambda_um": float(lam), "x": px, "y": py, "flux": float(flux),
                         "snr": float(snr), "compactness": float(compact),
                         "background": float(np.median(bkg)), "noise_px": float(np.median(sig_px))})
    # Merge runs of adjacent predicted positions on one peak: keep the best S/N.
    merged: list[dict] = []
    for hbl in sorted(hits, key=lambda d: d["lambda_um"]):
        if merged and math.hypot(hbl["x"] - merged[-1]["x"], hbl["y"] - merged[-1]["y"]) <= 2.5:
            if hbl["snr"] > merged[-1]["snr"]:
                merged[-1] = hbl
        else:
            merged.append(hbl)
    for hbl in merged:
        hbl["industrial_flag"] = industrial_match(hbl["lambda_um"], conf, float(resel_um))
    return merged


# --------------------------------------------------------------------------------------
# Synthesis for tests and the selftest
# --------------------------------------------------------------------------------------

def _planck_shape(wl_um: np.ndarray, t_k: float = 5000.0) -> np.ndarray:
    """F_lambda of a blackbody, normalised to its median over the grid."""
    hc_k = 14387.77          # um K
    x = hc_k / (wl_um * t_k)
    b = wl_um ** -5 / np.expm1(x)
    return b / np.median(b)


def synthesise_spectrum(mode: str, conf: dict, snr_continuum: float, line_um: float | None = None,
                        line_snr: float = 0.0, resolved_width_resel: float | None = None,
                        source_id: str = "syn", is_point_source: bool | None = True,
                        rng=None, n: int | None = None, *, trace_origin_px: float = 512.0,
                        trace_y: float = 1024.0, detector: str = "SCA01") -> Spectrum:
    """A stellar-like spectrum in ``mode`` with an optional injected line.

    The wavelength grid spans the disperser's ``range_um`` at the nominal
    dispersion (or ``n`` samples); the continuum is a 5000 K Planck shape with
    three broad (sigma >= 0.015 um) absorption troughs, so a median continuum
    has curvature to cope with; noise is Gaussian at ``flux / snr_continuum``
    per sample.  The line is a Gaussian at the *local LSF* (or
    ``resolved_width_resel`` times it) with the amplitude that makes the
    LSF-matched-filter S/N equal ``line_snr`` (computed numerically for the
    width mismatch of a resolved line).  ``trace_pixel`` is linear from
    ``trace_origin_px``; ``meta['trace_y']`` and ``detector`` are set so the
    geometric vetoes can run.  ``contam`` is left ``None`` on purpose -- the
    test for the honesty rule needs it absent -- set it on the returned object
    when the overlap test should run.
    """
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    m = str(mode).upper()
    d = _dispersers(conf).get(m) or {}
    lo, hi = (float(v) for v in d.get("range_um", [1.0, 1.93])[:2])
    if n is None:
        step = float(_NOMINAL_DISPERSION_UM_PER_PX.get(m, (hi - lo) / 800.0))
        n = int(round((hi - lo) / step)) + 1
    wl = np.linspace(lo, hi, int(n))
    cont = _planck_shape(wl)
    for cen, sig_um, depth in ((lo + 0.28 * (hi - lo), 0.02, 0.10), (lo + 0.55 * (hi - lo), 0.03, 0.07),
                               (lo + 0.80 * (hi - lo), 0.015, 0.12)):
        cont *= 1.0 - depth * np.exp(-0.5 * ((wl - cen) / sig_um) ** 2)
    err = cont / float(snr_continuum)
    flux = cont + rng.normal(0.0, err)
    spec = Spectrum(source_id=source_id, ra=0.0, dec=0.0, mode=m, wavelength_um=wl, flux=flux,
                    flux_err=err, survey="synthetic", is_point_source=is_point_source,
                    detector=detector, trace_pixel=trace_origin_px + np.arange(n, dtype=float),
                    meta={"trace_y": float(trace_y), "synthetic": True})
    if line_um is not None and line_snr > 0:
        sig_pix = lsf_sigma_pix(spec, conf)
        i0 = int(np.argmin(np.abs(wl - float(line_um))))
        s_lsf = float(sig_pix[i0])
        s_line = s_lsf * (float(resolved_width_resel) if resolved_width_resel else 1.0)
        x = (wl - float(line_um)) / float(np.median(np.diff(wl)))
        profile = np.exp(-0.5 * (x / s_line) ** 2)
        # Matched-filter response of this profile to the unit-norm LSF kernel.
        half = max(1, int(math.ceil(4 * s_lsf)))
        k = np.exp(-0.5 * (np.arange(-half, half + 1) / s_lsf) ** 2)
        k /= np.sqrt(np.sum(k ** 2))
        response = float(np.max(np.convolve(profile, k, mode="same")))
        amp = float(line_snr) * float(err[i0]) / max(response, 1e-12)
        spec.flux = spec.flux + amp * profile
        spec.meta.update({"injected_um": float(line_um), "injected_snr": float(line_snr),
                          "injected_width_resel": float(resolved_width_resel or 1.0)})
    return spec


def synthesise_dispersed_image(shape: tuple[int, int], star_xy: tuple[float, float],
                               dispersion: Callable[[float], tuple[float, float]], fwhm_px: float,
                               line_um: float | None, line_flux: float, continuum_per_px: float,
                               rng=None, *, trace_lambdas_um=None, noise_sigma: float = 1.0) -> np.ndarray:
    """A slitless-image cutout: one star's continuum trace, an optional line blob, noise.

    The trace is the sum of PSFs (Gaussian, ``fwhm_px``) placed at
    ``dispersion(lambda)`` for ``trace_lambdas_um`` (default 1.00-1.93 um in
    900 steps), each carrying ``continuum_per_px`` times its step along the
    trace so the flux per pixel of trace is ``continuum_per_px``.  The line is
    one PSF of total flux ``line_flux`` at ``dispersion(line_um)``.  Gaussian
    noise of ``noise_sigma`` per pixel.
    """
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    ny, nx = int(shape[0]), int(shape[1])
    img = np.zeros((ny, nx))
    sigma = max(float(fwhm_px), 0.5) / FWHM_PER_SIGMA
    norm = 2.0 * math.pi * sigma ** 2
    x0, y0 = float(star_xy[0]), float(star_xy[1])
    lams = (np.linspace(1.0, 1.93, 900) if trace_lambdas_um is None
            else np.asarray(trace_lambdas_um, dtype=float))
    half = int(math.ceil(4 * sigma)) + 1

    def _add(px: float, py: float, total: float) -> None:
        ix, iy = int(round(px)), int(round(py))
        xa, xb = max(0, ix - half), min(nx, ix + half + 1)
        ya, yb = max(0, iy - half), min(ny, iy + half + 1)
        if xa >= xb or ya >= yb:
            return
        ys, xs = np.mgrid[ya:yb, xa:xb]
        img[ya:yb, xa:xb] += total / norm * _psf_stamp(px, py, sigma, xs, ys)

    if continuum_per_px > 0 and lams.size:
        prev = None
        for lam in lams:
            dx, dy = dispersion(float(lam))
            p = (x0 + dx, y0 + dy)
            step = 1.0 if prev is None else math.hypot(p[0] - prev[0], p[1] - prev[1])
            _add(p[0], p[1], continuum_per_px * step)
            prev = p
    if line_um is not None and line_flux > 0:
        dx, dy = dispersion(float(line_um))
        _add(x0 + dx, y0 + dy, float(line_flux))
    if noise_sigma > 0:
        img += rng.normal(0.0, float(noise_sigma), img.shape)
    return img


__all__ = [
    "FWHM_PER_SIGMA", "VETO_NAMES", "RECURRENCE_VETO_NAMES",
    "air_to_vacuum_um", "industrial_lines", "stellar_lines",
    "resolving_power", "resolving_power_with_note", "sample_spacing_um", "lsf_sigma_pix",
    "find_unresolved_lines", "is_stellar_line", "industrial_match",
    "zeroth_order_contaminated", "overlap_contaminated", "persistence_suspect",
    "single_line_emitter_suspect", "screen_spectrum", "assess_features",
    "linear_dispersion", "dispersed_blob_search",
    "synthesise_spectrum", "synthesise_dispersed_image",
]
