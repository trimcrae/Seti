"""Per-exposure persistence test for the narrow-line survivors.

A survey coadd is the sum of several individual exposures (SDSS: typically 3-8
x 15 min on one plate; DESI: several 10-20 min exposures across one or more
nights).  A real monochromatic source persists, at the same strength, in every
exposure that built the coadd.  A cosmic ray, a single-exposure read-out
artefact, or a one-off sky-subtraction failure lives in one exposure and is
merely *diluted* into the coadd.  A sky-line residual tracks the sky-line
strength from exposure to exposure.  This is the decisive test that no
single coadded spectrum can provide and the one `spectra-confirm` could not
reach (no survivor had a second SPARCL spectrum).

Two data routes, both public, both runner-side:

* **SDSS-DR17** --- the *full* ``spec-PLATE-MJD-FIBER.fits`` on the SAS: HDU 1 is
  the coadd, HDUs 4+ are the individual spCFrame exposures (one per camera per
  exposure) with ``flux, loglam, ivar, mask, wdisp, sky``.  One file per
  survivor.  Plate/MJD/fiber/run2d come from the SPARCL record (or are decoded
  from ``specobjid``).
* **DESI-DR1** --- the healpix ``coadd`` file's ``EXP_FIBERMAP`` lists the
  (night, expid, petal, fiber) of every exposure that went into the coadd; each
  exposure's sky-subtracted, flux-calibrated spectrum is one row of the
  ``cframe-{band}{petal}-{expid}.fits`` image HDUs (and the sky model one row of
  ``sky-{band}{petal}-{expid}.fits``), read by HTTP range requests where the
  server allows it and by whole-file download where it does not.

Per exposure the line is measured *independently* (local continuum, integrated
excess in a window of +-1.2 LSF FWHM, error from the pipeline inverse variance
rescaled to the empirical continuum scatter), and the set of measurements is
classified: ``persistent``, ``persistent_2exp``, ``transient``, ``sky_residual``,
``absent_in_exposures``, ``inconsistent_strength``, ``partial``, ``untestable``.

The same run also applies the remaining cheap discriminators to every survivor:
a second epoch (any other SPARCL spectrum at the position, measured the same
way), SIMBAD object type, and a comprehensive rest-frame line identification at
the star's own catalogue redshift (``linelist.py``).  Everything is checkpointed
per spectrum so a shard that dies loses minutes, not hours.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np

from .confirm import _find_with_retry, _make_client

SDSS_SAS = "https://data.sdss.org/sas/dr17"
DESI_DR1 = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron"
# Mirrors of the same DR1 tree, tried in order when the primary host does not
# answer.  The 2026-09-22 probe got a connection-level failure (status -1, not a
# 404) from every data.desi.lbl.gov request, so the host being reachable is a
# thing to measure per run, not to assume.
DESI_DR1_BASES = [
    DESI_DR1,
    "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron",
    "https://portal.nersc.gov/cfs/desi/public/dr1/spectro/redux/iron",
]

# --- thresholds ------------------------------------------------------------------
PRESENT_SIG = 2.5          # an exposure "shows" the line at >= this significance
PERSIST_FRAC = 0.8         # ... in >= this fraction of testable exposures
PERSIST_MIN_EXP = 3        # ... with at least this many testable exposures
COMBINED_SIG = 4.0         # weighted-mean per-exposure significance to call it real
DOMINANT_FRAC = 0.7        # one exposure carrying >= this share of the flux = suspect
SKY_LINE_SIG = 5.0         # sky model peak at the wavelength above this = on a sky line
CHI2_P_INCONSISTENT = 0.01

# Bumped whenever a change alters the NUMBERS a checkpoint holds.  run_shard
# reuses a checkpoint only if it carries the current value, so a corrected
# estimator re-measures instead of silently inheriting the superseded run's
# answer -- the failure mode that would have frozen the 2026-09-22 median
# continuum bias into every result committed afterwards.
#   1: median continuum, photon-only error
#   2: sigma-clipped linear continuum fit, continuum error propagated
#   3: per-spectrum offset null (bias and scatter of this estimator where there
#      is no line) subtracted from every exposure and from the coadd, plus the
#      inverse-variance stack of the exposure HDUs as a second reference
#   4: the null's own standard error propagated into the corrected error, and
#      no correction at all from a null too thin to mean anything
#   5: the DESI route's null re-measures the SPARCL coadd at the same offsets,
#      so the DESI coadd significance is calibrated like the SDSS one (it fell
#      back to the uncalibrated number under v4)
CKPT_VERSION = 5

# Offsets used for the in-spectrum null.  24 is enough to place the median to
# ~0.3 sigma and costs nothing once the arrays are in memory.
N_NULL_OFFSETS = 24
# Per-exposure measurements the null must have before its bias is subtracted
# from anything.  Below this the median is not an estimate, it is a draw.
MIN_NULL_MEASUREMENTS = 12

# SDSS SPPIXMASK bits (per-exposure spCFrame masks).
SDSS_BAD_BITS = (1 << 16) | (1 << 18) | (1 << 22) | (1 << 24) | (1 << 25)
SDSS_REJECT_BITS = (1 << 18) | (1 << 25)          # FULLREJECT | COMBINEREJ
# DESI specmask bits.
DESI_BAD_BITS = 2 | 256 | 512                       # ALLBADPIX | NODATA | BADFIBER
DESI_COSMIC_BIT = 4


# ---------------------------------------------------------------------------
# LSF and measurement
# ---------------------------------------------------------------------------

def lsf_fwhm_A(lam: float, release: str, band: str | None = None) -> float:
    """Instrumental FWHM (A) at ``lam`` for a survey / arm."""
    rel = (release or "").upper()
    if rel.startswith("DESI"):
        r = {"b": 2500.0, "r": 3600.0, "z": 4500.0}.get(
            band or desi_bands_for(lam)[0], 3000.0)
    else:
        r = 2000.0
    return float(lam) / r


def _mad_std(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def measure_line(wave, flux, ivar, lam0: float, fwhm_A: float, mode: str = "emission",
                 mask=None, sky=None, bad_bits: int = 0, cosmic_bits: int = 0,
                 cont_win_A: float = 30.0) -> dict:
    """Independent measurement of a narrow line at ``lam0`` in one spectrum.

    Returns integrated line flux ``F`` (positive in the searched sense: excess
    for emission, deficit for absorption), its error, significance, equivalent
    width (A), the peak significance, the local continuum, and --- when a sky
    model is given --- whether the sky has a line there and how bright it is.
    ``testable`` is False when the wavelength is not covered or the line window
    is masked.
    """
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    ivar = np.asarray(ivar, float)
    n = wave.size
    out = {"testable": False, "reason": "", "F": float("nan"), "err": float("nan"),
           "sig": float("nan"), "ew": float("nan"), "peak_sig": float("nan"),
           "cont": float("nan"), "n_line_pix": 0, "n_cosmic": 0, "n_reject": 0,
           "sky_peak_sig": float("nan"), "sky_level": float("nan")}
    if n < 20 or flux.size != n or ivar.size != n:
        out["reason"] = "no_spectrum"
        return out
    if not (np.nanmin(wave) + cont_win_A < lam0 < np.nanmax(wave) - cont_win_A):
        out["reason"] = "not_covered"
        return out
    good = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
    m = None
    if mask is not None:
        m = np.asarray(mask)
        if m.size == n:
            mi = m.astype(np.int64)
            if bad_bits:
                good &= (mi & bad_bits) == 0
        else:
            m = None
    half = 1.2 * fwhm_A
    excl = 2.0 * fwhm_A
    d = wave - lam0
    line = (np.abs(d) <= half)
    cont = (np.abs(d) <= cont_win_A) & (np.abs(d) > excl)
    line_good = line & good
    cont_good = cont & good
    out["n_line_pix"] = int(line_good.sum())
    if m is not None:
        mi = m.astype(np.int64)
        if cosmic_bits:
            out["n_cosmic"] = int(((mi & cosmic_bits) != 0)[line].sum())
        out["n_reject"] = int((mi != 0)[line].sum())
    if line_good.sum() < 2 or line_good.sum() < 0.6 * line.sum():
        out["reason"] = "line_masked"
        return out
    if cont_good.sum() < 8:
        out["reason"] = "continuum_masked"
        return out
    # Continuum: a sigma-clipped LINEAR fit across the annulus, not its median.
    # A median has no slope term, so whenever the annulus is sampled
    # asymmetrically -- which is the rule, not the exception, near a spectrum's
    # blue end or beside a masked region -- it returns the mean level of
    # whichever side survived rather than the continuum *under the line*.  On
    # the steep blue throughput slope of an SDSS exposure that mis-sets the
    # continuum by a per cent or two, which at continuum S/N ~ 20 over ~50
    # annulus pixels is a many-sigma spurious deficit in every spectrum at once.
    cw = wave[cont_good] - lam0
    cf = flux[cont_good]
    keep = np.ones(cw.size, bool)
    coef = np.array([float(np.median(cf)), 0.0])
    for _ in range(3):
        if keep.sum() < 6:
            break
        try:
            coef = np.polyfit(cw[keep], cf[keep], 1)[::-1]
        except (np.linalg.LinAlgError, ValueError):
            break
        r = cf - (coef[0] + coef[1] * cw)
        s = _mad_std(r)
        if not (np.isfinite(s) and s > 0):
            break
        keep = np.abs(r) <= 3.0 * s
    cont_at = coef[0] + coef[1] * (wave - lam0)
    c = float(coef[0])                       # continuum evaluated AT the line
    resid = flux - cont_at
    # How lopsided was the annulus?  A one-sided continuum is still fitted (the
    # slope term handles it) but the imbalance is recorded, because it is the
    # condition under which a continuum error masquerades as a line.
    n_blue = int((cw < 0).sum())
    n_red = int((cw > 0).sum())
    pipe_err = 1.0 / np.sqrt(ivar[cont_good])
    emp = _mad_std(resid[cont_good])
    scale = 1.0
    med_pipe = float(np.median(pipe_err))
    if np.isfinite(emp) and med_pipe > 0 and emp > med_pipe:
        scale = emp / med_pipe
    noise = max(emp if np.isfinite(emp) else 0.0, med_pipe * scale, 1e-12)
    sign = -1.0 if mode == "absorption" else 1.0
    idx = np.where(line_good)[0]
    dl = np.gradient(wave)
    err_i = scale / np.sqrt(ivar[idx])
    F = float(sign * np.sum(resid[idx] * dl[idx]))
    err = float(np.sqrt(np.sum((err_i * dl[idx]) ** 2)))
    # The continuum is estimated, not known: its uncertainty at the line
    # propagates into the integrated flux and must not be left out.
    n_eff = max(int(keep.sum()), 1)
    cont_err = (emp if np.isfinite(emp) else med_pipe) / np.sqrt(n_eff)
    width = float(np.sum(dl[idx]))
    err = float(np.sqrt(err ** 2 + (cont_err * width) ** 2))
    out.update({
        "testable": True, "cont": c, "F": F, "err": err,
        "sig": F / err if err > 0 else float("nan"),
        "ew": F / c if c != 0 else float("nan"),
        "peak_sig": float(sign * np.max(sign * resid[idx]) / noise),
        "err_scale": round(float(scale), 3),
        "cont_slope_per_A": float(coef[1]),
        "cont_n_blue": n_blue, "cont_n_red": n_red,
        "cont_one_sided": bool(min(n_blue, n_red) < 4),
    })
    if sky is not None:
        s = np.asarray(sky, float)
        if s.size == n:
            sc = s[cont_good]
            sl = s[line_good]
            if np.isfinite(sc).sum() >= 8 and np.isfinite(sl).sum() >= 1:
                s_noise = _mad_std(sc)
                s_med = float(np.nanmedian(sc))
                if not (np.isfinite(s_noise) and s_noise > 0):
                    s_noise = max(abs(s_med) * 0.05, 1e-9)
                out["sky_peak_sig"] = float((np.nanmax(sl) - s_med) / s_noise)
                out["sky_level"] = float(np.nanmedian(sl))
    return out


def combine_measurements(meas: list[dict]) -> dict:
    """Inverse-variance combination of the same line measured in several arms
    of ONE exposure (the arm overlap regions)."""
    good = [m for m in meas if m.get("testable")]
    if not good:
        base = dict(meas[0]) if meas else {"testable": False, "reason": "no_spectrum"}
        base["testable"] = False
        return base
    if len(good) == 1:
        return dict(good[0])
    w = np.array([1.0 / m["err"] ** 2 for m in good])
    F = float(np.sum(w * np.array([m["F"] for m in good])) / np.sum(w))
    err = float(1.0 / np.sqrt(np.sum(w)))
    out = dict(good[int(np.argmax(w))])
    out.update({"F": F, "err": err, "sig": F / err,
                "ew": F / out["cont"] if out.get("cont") else float("nan"),
                "n_arms": len(good),
                "sky_peak_sig": float(np.nanmax([m.get("sky_peak_sig", np.nan) for m in good])),
                "n_cosmic": int(sum(m.get("n_cosmic", 0) for m in good))})
    return out


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _null_calibration(null: dict | None) -> dict:
    """What this spectrum's own null says the estimator does where there is no line.

    Returns the per-exposure bias and scatter, the coadd's, and -- the part that
    matters for safety -- the STANDARD ERROR of each bias.  The bias is
    subtracted from every measurement, so it is itself an estimate with an
    uncertainty, and that uncertainty has to be propagated: a null built from a
    handful of offsets can land several sigma from the truth, and subtracting
    such a number from a deficit would MANUFACTURE a detection.  With the
    standard error carried through, a thin null widens the error bar instead of
    inventing signal, and a null too thin to mean anything (fewer than
    MIN_NULL_MEASUREMENTS) is not applied at all.

    The scatter is floored at 1.0: the nominal error already carries unit
    variance, and a null that comes out narrower than nominal is not a licence
    to call a line more significant than the photon noise allows.
    """
    out = {"bias": 0.0, "sd": 1.0, "se_bias": 0.0,
           "coadd_bias": 0.0, "coadd_sd": 1.0, "coadd_se_bias": 0.0,
           "applied": False, "n": 0, "n_coadd": 0}
    if not null:
        return out

    def _v(key, default):
        x = null.get(key)
        try:
            v = float(x)
        except (TypeError, ValueError):
            return default
        return v if np.isfinite(v) else default

    n = int(_v("n_exposure_measurements", 0))
    n_co = int(_v("n_coadd_measurements", 0))
    out["n"], out["n_coadd"] = n, n_co
    if n < MIN_NULL_MEASUREMENTS:
        return out
    mad = max(_v("exposure_sig_mad", 1.0), 1.0)
    out.update({"bias": _v("exposure_sig_median", 0.0), "sd": mad,
                # standard error of a median, 1.2533 * sigma / sqrt(n)
                "se_bias": 1.2533 * mad / np.sqrt(n), "applied": True})
    if n_co >= MIN_NULL_MEASUREMENTS:
        co_mad = max(_v("coadd_sig_mad", 1.0), 1.0)
        out.update({"coadd_bias": _v("coadd_sig_median", 0.0), "coadd_sd": co_mad,
                    "coadd_se_bias": 1.2533 * co_mad / np.sqrt(n_co)})
    return out


def classify_persistence(coadd: dict | None, exposures: list[dict],
                         stack: dict | None = None, null: dict | None = None) -> dict:
    """Classify a set of independent per-exposure measurements of one line.

    ``null`` is the same estimator's reading at wavelengths in this spectrum
    where nothing was found (:func:`offset_null`).  When it is given, every
    per-exposure flux is corrected by the bias it measures and every error is
    widened to the scatter it measures, so the significances below are excesses
    over what this spectrum returns for nothing at all -- not over zero, which
    the SDSS blue frames demonstrably do not deliver.

    ``stack`` is the same line measured in the inverse-variance mean of the
    exposures (:func:`stack_exposures`).  The coadd is supposed to BE that, so a
    coadd feature that the stack does not show is made by the coaddition.
    """
    from scipy import stats

    cal = _null_calibration(null)
    bias, sd, se_bias = cal["bias"], cal["sd"], cal["se_bias"]
    co_bias, co_sd = cal["coadd_bias"], cal["coadd_sd"]
    tested = [e for e in exposures if e.get("testable")]
    n = len(tested)
    res = {"n_exposures": len(exposures), "n_tested": n, "n_present": 0,
           "frac_present": float("nan"), "mean_F": float("nan"),
           "mean_err": float("nan"), "combined_sig": float("nan"),
           "chi2": float("nan"), "chi2_p": float("nan"), "dominant_frac": float("nan"),
           "sig_without_strongest": float("nan"), "max_exposure_sig": float("nan"),
           "on_sky_line": False, "sky_corr": float("nan"), "n_cosmic_flagged": 0,
           "ratio_to_coadd": float("nan"), "coadd_recovered": None,
           "null_exposure_bias_sig": bias, "null_exposure_scatter": sd,
           "null_exposure_bias_se": se_bias, "null_n_measurements": cal["n"],
           "null_coadd_bias_sig": co_bias, "null_calibrated": bool(cal["applied"]),
           "combined_sig_raw": float("nan"), "coadd_sig_cal": float("nan"),
           "stack_sig": float("nan"), "stack_ew": float("nan"),
           "stack_over_coadd_F": float("nan"),
           "persistence_class": "untestable", "basis": ""}
    if coadd is not None and coadd.get("testable"):
        res["coadd_sig_cal"] = (float(coadd["sig"]) - co_bias) / co_sd
        res["coadd_recovered"] = bool(res["coadd_sig_cal"] >= 3.0)
        if np.isfinite(coadd.get("sky_peak_sig", np.nan)) and coadd["sky_peak_sig"] >= SKY_LINE_SIG:
            res["on_sky_line"] = True
    if stack is not None and stack.get("testable"):
        res["stack_sig"] = (float(stack["sig"]) - co_bias) / co_sd
        res["stack_ew"] = float(stack.get("ew", np.nan))
        if coadd is not None and coadd.get("testable") and coadd.get("F"):
            res["stack_over_coadd_F"] = float(stack["F"]) / float(coadd["F"])
    if n == 0:
        reasons = sorted({e.get("reason", "") for e in exposures if e.get("reason")})
        res["basis"] = "no testable exposure" + (f" ({', '.join(reasons)})" if reasons else "")
        return res
    # Correct every exposure by the bias this spectrum's own null shows, and
    # widen every error to the scatter it shows, before anything is compared.
    err_raw = np.array([e["err"] for e in tested], float)
    F_raw = np.array([e["F"] for e in tested], float)
    res["combined_sig_raw"] = float(np.sum(F_raw / err_raw ** 2) /
                                    np.sqrt(np.sum(1.0 / err_raw ** 2)))
    F = F_raw - bias * err_raw
    # The bias is an ESTIMATE.  Its own standard error is a flux uncertainty of
    # se_bias * err_raw and belongs in the error bar; without it a thin null
    # could subtract a spurious deficit and manufacture a detection.
    err = np.sqrt((err_raw * sd) ** 2 + (se_bias * err_raw) ** 2)
    sig = F / err
    present = sig >= PRESENT_SIG
    w = 1.0 / err ** 2
    Fbar = float(np.sum(w * F) / np.sum(w))
    ebar = float(1.0 / np.sqrt(np.sum(w)))
    chi2 = float(np.sum(w * (F - Fbar) ** 2))
    p = float(stats.chi2.sf(chi2, n - 1)) if n > 1 else float("nan")
    pos = F > 0
    dominant = float(F.max() / F[pos].sum()) if pos.any() and F.max() > 0 else 0.0
    k = int(np.argmax(F))
    if n >= 2:
        rest = np.delete(np.arange(n), k)
        wr = w[rest]
        sig_rest = float(np.sum(wr * F[rest]) / np.sum(wr) * np.sqrt(np.sum(wr)))
    else:
        sig_rest = float("nan")
    sky_sig = np.array([e.get("sky_peak_sig", np.nan) for e in tested], float)
    sky_lvl = np.array([e.get("sky_level", np.nan) for e in tested], float)
    if np.isfinite(sky_sig).any() and np.nanmax(sky_sig) >= SKY_LINE_SIG:
        res["on_sky_line"] = True
    corr = float("nan")
    ok = np.isfinite(sky_lvl)
    if ok.sum() >= 3 and np.std(sky_lvl[ok]) > 0 and np.std(F[ok]) > 0:
        corr = float(np.corrcoef(F[ok], sky_lvl[ok])[0, 1])
    n_cos = int(sum(1 for e in tested if e.get("n_cosmic", 0) > 0))
    res.update({
        "n_present": int(present.sum()), "frac_present": float(present.mean()),
        "mean_F": Fbar, "mean_err": ebar, "combined_sig": Fbar / ebar,
        "chi2": chi2, "chi2_p": p, "dominant_frac": dominant,
        "sig_without_strongest": sig_rest, "max_exposure_sig": float(sig.max()),
        # The bias-robust companion to combined_sig.  An inverse-variance
        # combination multiplies any residual per-exposure bias b by sqrt(N),
        # so N=18 exposures each reading a harmless -0.4 sigma combine to -1.7,
        # and each reading -1.0 combine to -4.2 -- with no line and no further
        # bias anywhere.  The median of the per-exposure significances does not
        # do that, and is the number to read when combined_sig is large and
        # negative.  Measured on run 35747997902 shard 0: median per-exposure
        # -0.40 against a median combined -2.60, over 24 lines.
        "median_exposure_sig": float(np.median(sig)),
        "sky_corr": corr, "n_cosmic_flagged": n_cos,
    })
    if coadd is not None and coadd.get("testable") and coadd.get("F"):
        co_F = float(coadd["F"]) - co_bias * float(coadd["err"])
        if co_F:
            res["ratio_to_coadd"] = Fbar / co_F
    frac = float(present.mean())
    cls, basis = "ambiguous", ""
    consistent_all = bool(np.all(np.abs(F - Fbar) <= 2.0 * err))
    if n >= 2 and present.sum() <= 1 and dominant >= 0.8 and sig.max() >= 3.0 \
            and (not np.isfinite(sig_rest) or sig_rest < 2.0):
        cls = "transient"
        basis = (f"one exposure carries {dominant:.0%} of the flux at {sig.max():.1f} sigma; "
                 f"the others combine to {sig_rest:.1f} sigma")
    elif n >= 3 and res["on_sky_line"] and (p < CHI2_P_INCONSISTENT or
                                             (np.isfinite(corr) and corr > 0.7)):
        cls = "sky_residual"
        basis = (f"sky model has a line here (peak {np.nanmax(sky_sig):.1f} sigma); "
                 f"strength varies between exposures (chi2 p={p:.2g}, corr with sky {corr:.2f})")
    elif n >= PERSIST_MIN_EXP and frac >= PERSIST_FRAC and Fbar / ebar >= COMBINED_SIG \
            and dominant < DOMINANT_FRAC:
        cls = "persistent"
        basis = (f"present at >= {PRESENT_SIG} sigma in {int(present.sum())}/{n} exposures; "
                 f"combined {Fbar / ebar:.1f} sigma; chi2 p={p:.2g}")
    elif n >= PERSIST_MIN_EXP and Fbar / ebar >= COMBINED_SIG and p >= CHI2_P_INCONSISTENT \
            and consistent_all and dominant < DOMINANT_FRAC and float((sig >= 1.5).mean()) >= 0.6:
        cls = "persistent"
        basis = (f"individual exposures are low S/N (max {sig.max():.1f} sigma) but every one "
                 f"is consistent with the {Fbar / ebar:.1f}-sigma mean (chi2 p={p:.2g})")
    elif n == 2 and frac == 1.0 and Fbar / ebar >= COMBINED_SIG and dominant < 0.8:
        cls = "persistent_2exp"
        basis = f"both exposures show it ({sig[0]:.1f}, {sig[1]:.1f} sigma); only 2 exposures"
    elif n >= 2 and present.sum() == 0 and Fbar / ebar < 2.0 \
            and (not np.isfinite(res["stack_sig"]) or res["stack_sig"] < 4.0):
        cls = "absent_in_exposures"
        # State the numbers; do not state a conclusion the numbers do not carry.
        # A large negative combined_sig is sqrt(N) times a small residual
        # per-exposure bias, not evidence that the coadd invented a feature --
        # which is why the median per-exposure significance is quoted next to it.
        basis = (f"no exposure shows it (max {sig.max():.1f} sigma, median "
                 f"{np.median(sig):.1f} sigma, combined {Fbar / ebar:.1f} sigma"
                 + (f", stack of the exposures {res['stack_sig']:.1f} sigma"
                    if np.isfinite(res["stack_sig"]) else "")
                 + ")")
    elif n >= 2 and present.sum() == 0 and np.isfinite(res["stack_sig"]) \
            and res["stack_sig"] >= 4.0:
        # The exposures are individually too noisy to show it but their own
        # inverse-variance mean does.  That is not absence, and calling it
        # absence would throw away a line that is in the data.
        cls = "stack_only"
        basis = (f"no single exposure reaches {PRESENT_SIG} sigma (max {sig.max():.1f}) but the "
                 f"inverse-variance stack of the same exposures shows it at "
                 f"{res['stack_sig']:.1f} sigma; combined per-exposure {Fbar / ebar:.1f} sigma")
    elif n >= 3 and frac >= 0.5 and p < 0.001 and dominant < 0.8:
        cls = "inconsistent_strength"
        basis = f"present in {int(present.sum())}/{n} but strengths disagree (chi2 p={p:.2g})"
    elif n >= 2 and 0 < frac < PERSIST_FRAC:
        cls = "partial"
        basis = (f"present in {int(present.sum())}/{n} exposures; combined "
                 f"{Fbar / ebar:.1f} sigma; dominant share {dominant:.0%}")
    elif n == 1:
        cls = "single_exposure_only"
        basis = f"only one testable exposure ({sig[0]:.1f} sigma): persistence undecidable"
    else:
        basis = (f"n={n}, present {int(present.sum())}, combined {Fbar / ebar:.1f} sigma, "
                 f"dominant {dominant:.0%}, chi2 p={p:.2g}")
    res["persistence_class"] = cls
    res["basis"] = basis
    return res


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _session():
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": "seti-spectra-persist/1.0"})
    return s


def http_head(url: str, timeout: float = 60.0, tries: int = 3) -> dict:
    """HEAD (falling back to a 1-byte ranged GET) -> status, length, ranges.

    A connection-level failure is reported as status -1 *with the exception
    text*: "the archive refused us" and "the archive does not have this file"
    are different verdicts and must never be collapsed into one.
    """
    s = _session()
    last = ""
    for k in range(tries):
        try:
            r = s.head(url, allow_redirects=True, timeout=timeout)
            info = {"url": url, "status": r.status_code,
                    "length": int(r.headers.get("Content-Length", 0) or 0),
                    "accept_ranges": r.headers.get("Accept-Ranges", "")}
            if r.status_code in (403, 405):
                r2 = s.get(url, headers={"Range": "bytes=0-0"}, timeout=timeout, stream=True)
                info.update({"status": r2.status_code,
                             "accept_ranges": "bytes" if r2.status_code == 206
                             else info["accept_ranges"]})
                r2.close()
            return info
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
            time.sleep(2.0 * (k + 1))
    return {"url": url, "status": -1, "error": last}


def fetch_bytes(url: str, timeout: float = 300.0, max_bytes: int = 2_000_000_000,
                tries: int = 3) -> bytes | None:
    s = _session()
    last = None
    for k in range(tries):
        try:
            r = s.get(url, timeout=timeout, stream=True)
            if r.status_code != 200:
                r.close()
                return None
            buf = io.BytesIO()
            for chunk in r.iter_content(1 << 20):
                buf.write(chunk)
                if buf.tell() > max_bytes:
                    r.close()
                    raise RuntimeError(f"{url}: exceeds {max_bytes} bytes")
            return buf.getvalue()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3.0 * (k + 1))
    print(f"[persist] fetch failed {url}: {last!r}")
    return None


def fetch_to_file(url: str, dst: Path, timeout: float = 600.0, tries: int = 3) -> bool:
    s = _session()
    for k in range(tries):
        try:
            with s.get(url, timeout=timeout, stream=True) as r:
                if r.status_code != 200:
                    return False
                with open(dst, "wb") as fh:
                    for chunk in r.iter_content(4 << 20):
                        fh.write(chunk)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[persist] download attempt {k + 1} failed {url}: {exc!r}")
            time.sleep(3.0 * (k + 1))
    return False


def open_fits_remote(url: str, workdir: Path, prefer_range: bool = True):
    """Open a remote FITS file lazily (HTTP ranges via fsspec) or after download.

    Returns ``(hdulist, how)`` or ``(None, reason)``.  Range access is only
    attempted on uncompressed files; a gzipped file is always downloaded whole.
    """
    from astropy.io import fits
    if prefer_range and not url.endswith(".gz"):
        try:
            import fsspec  # noqa: F401
            hdul = fits.open(url, use_fsspec=True, lazy_load_hdus=True,
                             fsspec_kwargs={"block_size": 1 << 20})
            return hdul, "range"
        except Exception as exc:  # noqa: BLE001
            print(f"[persist] range open failed ({exc!r}); downloading {url}")
    workdir.mkdir(parents=True, exist_ok=True)
    dst = workdir / url.rsplit("/", 1)[-1]
    if not dst.exists() and not fetch_to_file(url, dst):
        return None, "download_failed"
    try:
        return fits.open(dst, memmap=False), "download"
    except Exception as exc:  # noqa: BLE001
        return None, f"open_failed: {exc!r}"


# ---------------------------------------------------------------------------
# SDSS
# ---------------------------------------------------------------------------

def decode_specobjid(specobjid) -> dict:
    """plate / fiber / mjd / run2d from an SDSS ``specObjID`` (64-bit)."""
    s = int(specobjid)
    plate = (s >> 50) & 0x3FFF
    fiber = (s >> 38) & 0xFFF
    mjd = ((s >> 24) & 0x3FFF) + 50000
    r = (s >> 10) & 0x3FFF
    if r < 100:
        run2d = str(r)
    elif r < 1000:
        run2d = str(r)          # 103, 104
    else:
        n, m, p = r // 10000, (r % 10000) // 100, r % 100
        run2d = f"v{n + 5}_{m}_{p}"
    return {"plate": plate, "fiberid": fiber, "mjd": mjd, "run2d": run2d}


def sdss_spec_urls(plate: int, mjd: int, fiber: int, run2d: str | None) -> list[str]:
    """Candidate SAS URLs for the FULL spec file (per-exposure HDUs), best first."""
    name = f"spec-{int(plate):04d}-{int(mjd)}-{int(fiber):04d}.fits"
    r2d = str(run2d or "").strip()
    urls = []
    if r2d.startswith("v"):
        urls.append(f"{SDSS_SAS}/eboss/spectro/redux/{r2d}/spectra/full/{int(plate):04d}/{name}")
    elif r2d:
        urls.append(f"{SDSS_SAS}/sdss/spectro/redux/{r2d}/spectra/full/{int(plate):04d}/{name}")
        urls.append(f"{SDSS_SAS}/sdss/spectro/redux/{r2d}/spectra/{int(plate):04d}/{name}")
    for alt in ("v5_13_2", "26", "104", "103"):
        if alt == r2d:
            continue
        base = "eboss" if alt.startswith("v") else "sdss"
        urls.append(f"{SDSS_SAS}/{base}/spectro/redux/{alt}/spectra/full/{int(plate):04d}/{name}")
        if base == "sdss":
            urls.append(f"{SDSS_SAS}/{base}/spectro/redux/{alt}/spectra/{int(plate):04d}/{name}")
    return urls


def parse_sdss_spec(hdul) -> dict:
    """Coadd + per-exposure spectra from a full SDSS spec file."""
    from astropy.io import fits  # noqa: F401
    coadd = None
    exposures = []
    for k, h in enumerate(hdul):
        if k == 0 or getattr(h, "data", None) is None:
            continue
        cols = [c.lower() for c in getattr(h, "columns", []).names] if hasattr(h, "columns") else []
        if "loglam" not in cols or "flux" not in cols:
            continue
        d = h.data
        wave = 10.0 ** np.asarray(d["loglam"], float)
        rec = {"wave": wave, "flux": np.asarray(d["flux"], float),
               "ivar": np.asarray(d["ivar"], float)}
        for name in ("mask", "and_mask", "sky", "wdisp"):
            if name in cols:
                rec[name] = np.asarray(d[name])
        if "mask" not in rec and "and_mask" in rec:
            rec["mask"] = rec["and_mask"]
        ext = str(h.header.get("EXTNAME", "")).upper()
        if ext == "COADD" or (coadd is None and k == 1):
            coadd = rec
            continue
        m = re.match(r"^([BR][12])-(\d+)", ext)
        camera = m.group(1).lower() if m else ext.lower()
        expid = int(m.group(2)) if m else k
        rec.update({"camera": camera, "expid": expid,
                    "mjd": h.header.get("MJD"), "exptime": h.header.get("EXPTIME"),
                    "extname": ext})
        exposures.append(rec)
    return {"coadd": coadd, "exposures": exposures}


def stack_exposures(parsed: dict, lam0: float | None = None,
                    half_width_A: float = 120.0) -> dict | None:
    """Inverse-variance mean of the per-exposure HDUs, on the coadd's own grid.

    The archive coadd is *supposed to be* this.  Measuring the same line in the
    stack and in the archive's coadd separates two very different verdicts that
    a per-exposure table alone cannot tell apart:

    * the stack reproduces the coadd  -> the per-exposure *measurement* is what
      disagrees, i.e. our estimator, the window, or the masking;
    * the stack reproduces the exposures -> the coadd does not agree with its
      own inputs, which is the artefact this whole channel exists to catch.

    Only the arms that cover ``lam0`` contribute; the exposures keep their own
    native wavelength solutions and are interpolated onto the coadd grid, so a
    wavelength zero-point difference shows up here as a shifted feature rather
    than as a missing one.
    """
    co = parsed.get("coadd")
    exps = parsed.get("exposures") or []
    if co is None or not exps:
        return None
    w = np.asarray(co["wave"], float)
    if lam0 is not None:
        sel = np.abs(w - float(lam0)) <= half_width_A
        if int(sel.sum()) < 20:
            return None
        w = w[sel]
    num = np.zeros(w.size)
    den = np.zeros(w.size)
    n_used = 0
    for e in exps:
        ew = np.asarray(e["wave"], float)
        ef = np.asarray(e["flux"], float)
        ei = np.asarray(e["ivar"], float)
        ok = np.isfinite(ew) & np.isfinite(ef) & np.isfinite(ei) & (ei > 0)
        if int(ok.sum()) < 20:
            continue
        order = np.argsort(ew[ok])
        ewo, efo, eio = ew[ok][order], ef[ok][order], ei[ok][order]
        inside = (w >= ewo[0]) & (w <= ewo[-1])
        if int(inside.sum()) < 10:
            continue
        num[inside] += np.interp(w[inside], ewo, eio) * np.interp(w[inside], ewo, efo)
        den[inside] += np.interp(w[inside], ewo, eio)
        n_used += 1
    if n_used == 0:
        return None
    flux = np.full(w.size, np.nan)
    good = den > 0
    flux[good] = num[good] / den[good]
    return {"wave": w, "flux": flux, "ivar": den, "n_used": n_used}


def offset_null(measure_at, lam0: float, n: int = 24, lo_A: float = 12.0,
                hi_A: float = 200.0, seed: int = 7) -> dict:
    """The per-exposure estimator, re-run at wavelengths where nothing was found.

    ``absent_in_exposures`` is only a statement about the sky if the estimator
    reads ~0 sigma where there is no line.  The first two SDSS runs returned
    combined significances of -3 to -11 for nearly every survivor, and genuine
    absence gives 0, so the question "is the deficit at the candidate
    wavelength, or everywhere?" has to be answered before any of those verdicts
    means anything.  This repeats the entire measurement -- exposures, coadd,
    combination -- at ``n`` random offsets from the candidate in the *same*
    spectrum.  A combined significance already at -5 sigma there is an estimator
    bias; one centred on 0 there makes the value at ``lam0`` a real statement.

    ``measure_at(lam) -> (coadd_measurement_or_None, [per-exposure measurements])``,
    so one null serves the SDSS route (re-measuring in-memory HDUs) and the DESI
    route (re-measuring cframe rows that are already downloaded).
    """
    rng = np.random.default_rng(seed)
    offs = rng.uniform(lo_A, hi_A, n) * rng.choice([-1.0, 1.0], n)
    comb, comax, co_sig, exp_sig = [], [], [], []
    for o in offs:
        try:
            fc, ex = measure_at(lam0 + float(o))
        except Exception as exc:  # noqa: BLE001
            print(f"[persist] offset null at {lam0 + float(o):.1f} A failed: {exc!r}")
            continue
        cls = classify_persistence(fc, ex)
        v = cls.get("combined_sig", float("nan"))
        if np.isfinite(v):
            comb.append(float(v))
            comax.append(float(cls.get("max_exposure_sig", np.nan)))
        for e in ex or []:
            if e.get("testable") and e.get("err"):
                s = float(e["F"]) / float(e["err"])
                if np.isfinite(s):
                    exp_sig.append(s)
        if fc and fc.get("testable") and np.isfinite(fc.get("sig", np.nan)):
            co_sig.append(float(fc["sig"]))
    out = {"n_offsets": int(n), "n_measured": len(comb), "n_exposure_measurements": len(exp_sig)}
    if comb:
        a = np.asarray(comb, float)
        out.update({
            "combined_sig_median": float(np.median(a)),
            "combined_sig_mean": float(np.mean(a)),
            "combined_sig_std": float(np.std(a)),
            "combined_sig_mad": float(_mad_std(a)),
            "combined_sig_min": float(np.min(a)),
            "combined_sig_max": float(np.max(a)),
            "frac_below_minus2": float(np.mean(a < -2.0)),
            "max_exposure_sig_median": float(np.nanmedian(np.asarray(comax, float))),
        })
    if exp_sig:
        b = np.asarray(exp_sig, float)
        out["exposure_sig_median"] = float(np.median(b))
        out["exposure_sig_mad"] = float(_mad_std(b))
    out["n_coadd_measurements"] = len(co_sig)
    if co_sig:
        c = np.asarray(co_sig, float)
        out["coadd_sig_median"] = float(np.median(c))
        out["coadd_sig_std"] = float(np.std(c))
        out["coadd_sig_mad"] = float(_mad_std(c))
    return out


def sdss_measure_at(parsed: dict, mode: str):
    """A ``measure_at`` callable for :func:`offset_null` over an SDSS spec file."""
    return lambda lam: sdss_exposure_measurements(parsed, lam, mode)


def desi_measure_at(collected: list[dict], mode: str, coadd: dict | None = None):
    """A ``measure_at`` callable over DESI cframe rows already in memory.

    The DESI route downloads tens of MB per exposure, so the null can only be
    afforded if the arrays are reused; ``collected`` is what
    :func:`desi_exposure_measurements` kept while it was reading them.

    ``coadd`` (``wave, flux, ivar`` and optionally ``mask, sky``: the SPARCL
    coadd the DESI route uses as its reference) is re-measured at each offset
    too.  Without it the null has no coadd measurements, and the DESI coadd
    significance could not be calibrated like the SDSS one -- it fell back to
    the raw number, which on SDSS runs a median ~3x the calibrated one.
    """
    def _coadd_at(lam):
        if not coadd or np.asarray(coadd.get("wave", []), float).size < 50:
            return None
        return measure_line(coadd["wave"], coadd["flux"], coadd["ivar"], lam,
                            lsf_fwhm_A(lam, "DESI-DR1"), mode, mask=coadd.get("mask"),
                            sky=coadd.get("sky"))

    def _at(lam):
        out = []
        for d in collected:
            arms = []
            for a in d.get("arms", []):
                fw = lsf_fwhm_A(lam, "DESI-DR1", a.get("band"))
                m = measure_line(a["wave"], a["flux"], a["ivar"], lam, fw, mode,
                                 mask=a.get("mask"), sky=a.get("sky"),
                                 bad_bits=DESI_BAD_BITS, cosmic_bits=DESI_COSMIC_BIT)
                m["band"] = a.get("band")
                arms.append(m)
            if arms:
                c = combine_measurements(arms)
                c["expid"] = d.get("expid")
                out.append(c)
        return _coadd_at(lam), out
    return _at


def wave_lag(co: dict, e: dict, lam0: float, half_width_A: float = 150.0,
             max_shift_A: float = 4.0, step_A: float = 0.05) -> float:
    """Wavelength shift (A) that best aligns one exposure with the coadd.

    A non-zero lag is the ordinary case -- the coadd carries a heliocentric
    correction the native exposure frames do not -- and a lag comparable to the
    line window is on its own enough to move a real line out of the window and
    into the continuum annulus, which turns a line into a *deficit*.
    """
    cw = np.asarray(co["wave"], float)
    cf = np.asarray(co["flux"], float)
    ew = np.asarray(e["wave"], float)
    ef = np.asarray(e["flux"], float)
    ok = np.isfinite(ew) & np.isfinite(ef)
    if int(ok.sum()) < 50:
        return float("nan")
    order = np.argsort(ew[ok])
    ewo, efo = ew[ok][order], ef[ok][order]
    sel = (np.abs(cw - lam0) <= half_width_A) & np.isfinite(cf)
    sel &= (cw >= ewo[0] + max_shift_A) & (cw <= ewo[-1] - max_shift_A)
    if int(sel.sum()) < 50:
        return float("nan")
    a = cf[sel] - np.mean(cf[sel])
    if not np.std(a) > 0:
        return float("nan")
    best, best_c = float("nan"), -2.0
    for s in np.arange(-max_shift_A, max_shift_A + 0.5 * step_A, step_A):
        b = np.interp(cw[sel] + s, ewo, efo)
        if not np.std(b) > 0:
            continue
        c = float(np.corrcoef(a, b - np.mean(b))[0, 1])
        if np.isfinite(c) and c > best_c:
            best_c, best = c, float(s)
    return best


def sdss_exposure_measurements(parsed: dict, lam0: float, mode: str) -> tuple[dict | None, list[dict]]:
    """Measure the line in the file coadd and, per EXPOSURE (arms combined)."""
    fwhm = lsf_fwhm_A(lam0, "SDSS-DR17")
    co = parsed.get("coadd")
    coadd_meas = None
    if co is not None:
        coadd_meas = measure_line(co["wave"], co["flux"], co["ivar"], lam0, fwhm, mode,
                                  mask=co.get("mask"), sky=co.get("sky"),
                                  bad_bits=SDSS_BAD_BITS, cosmic_bits=SDSS_REJECT_BITS)
    by_exp: dict[int, list[dict]] = {}
    meta: dict[int, dict] = {}
    for e in parsed.get("exposures", []):
        wd = e.get("wdisp")
        fw = fwhm
        if wd is not None:
            near = np.abs(e["wave"] - lam0) < 10.0
            if near.any():
                wpx = float(np.nanmedian(np.asarray(wd, float)[near]))
                dl = float(np.nanmedian(np.abs(np.diff(e["wave"][near])))) if near.sum() > 1 else 0.0
                if np.isfinite(wpx) and wpx > 0 and dl > 0:
                    fw = max(0.5 * fwhm, min(2.0 * fwhm, 2.3548 * wpx * dl))
        m = measure_line(e["wave"], e["flux"], e["ivar"], lam0, fw, mode,
                         mask=e.get("mask"), sky=e.get("sky"),
                         bad_bits=SDSS_BAD_BITS, cosmic_bits=SDSS_REJECT_BITS)
        m["camera"] = e.get("camera")
        by_exp.setdefault(e["expid"], []).append(m)
        meta[e["expid"]] = {"mjd": e.get("mjd"), "exptime": e.get("exptime")}
    out = []
    for expid in sorted(by_exp):
        c = combine_measurements(by_exp[expid])
        c.update({"expid": int(expid), "arms": [m.get("camera") for m in by_exp[expid]
                                                if m.get("testable")]})
        c.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                  for k, v in meta[expid].items()})
        out.append(c)
    return coadd_meas, out


# ---------------------------------------------------------------------------
# DESI
# ---------------------------------------------------------------------------

def desi_bands_for(lam: float) -> list[str]:
    bands = []
    if 3600.0 <= lam <= 5800.0:
        bands.append("b")
    if 5760.0 <= lam <= 7620.0:
        bands.append("r")
    if 7520.0 <= lam <= 9824.0:
        bands.append("z")
    return bands or ["r"]


def _desi_bases() -> list[str]:
    """Distinct DR1 host roots, preferred first."""
    seen, out = set(), []
    for b in DESI_DR1_BASES:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def _desi_healpix_dir(base: str, survey: str, program: str, healpix: int) -> str:
    hp = int(healpix)
    return f"{base}/healpix/{survey}/{program}/{hp // 100}/{hp}"


def desi_coadd_urls(survey: str, program: str, healpix: int) -> list[str]:
    hp = int(healpix)
    return [f"{_desi_healpix_dir(b, survey, program, hp)}/coadd-{survey}-{program}-{hp}.fits"
            for b in _desi_bases()]


def desi_coadd_url(survey: str, program: str, healpix: int) -> str:
    return desi_coadd_urls(survey, program, healpix)[0]


def desi_spectra_urls(survey: str, program: str, healpix: int) -> list[str]:
    hp = int(healpix)
    out = []
    for b in _desi_bases():
        stem = f"{_desi_healpix_dir(b, survey, program, hp)}/spectra-{survey}-{program}-{hp}"
        out += [stem + ".fits.gz", stem + ".fits"]
    return out


def desi_frame_urls(kind: str, night: int, expid: int, band: str, petal: int) -> list[str]:
    out = []
    for b in _desi_bases():
        stem = (f"{b}/exposures/{int(night)}/{int(expid):08d}/"
                f"{kind}-{band}{int(petal)}-{int(expid):08d}")
        out += [stem + ".fits", stem + ".fits.gz"]
    return out


def desi_exposure_rows(hdul, targetid: int) -> list[dict]:
    """(night, expid, tileid, petal, fiber) rows for a target from EXP_FIBERMAP."""
    try:
        t = hdul["EXP_FIBERMAP"].data
    except Exception:
        return []
    sel = np.asarray(t["TARGETID"]).astype(np.int64) == int(targetid)
    rows = []
    for r in t[sel]:
        rows.append({"night": int(r["NIGHT"]), "expid": int(r["EXPID"]),
                     "tileid": int(r["TILEID"]), "petal": int(r["PETAL_LOC"]),
                     "fiber": int(r["FIBER"]),
                     "exptime": float(r["EXPTIME"]) if "EXPTIME" in t.names else float("nan"),
                     "fiberstatus": int(r["FIBERSTATUS"]) if "FIBERSTATUS" in t.names else 0})
    return rows


def desi_read_row(hdul, hdu_names: list[str], fiber: int) -> dict | None:
    """One fiber's row from each named image HDU of a cframe / sky file."""
    try:
        fm = hdul["FIBERMAP"].data
        fibers = np.asarray(fm["FIBER"]).astype(int)
        idx = np.where(fibers == int(fiber))[0]
        if idx.size == 0:
            return None
        i = int(idx[0])
    except Exception:
        i = int(fiber) % 500
    out = {"wave": np.asarray(hdul["WAVELENGTH"].data, float)}
    for name in hdu_names:
        try:
            h = hdul[name]
            try:
                arr = h.section[i, :]          # ranged read of one row (remote)
            except Exception:  # noqa: BLE001 -- in-memory HDUs have no section
                arr = h.data[i, :]
            out[name.lower()] = np.asarray(arr, float)
        except Exception as exc:  # noqa: BLE001
            print(f"[persist] DESI read {name} row {i} failed: {exc!r}")
            return None
    return out


def desi_exposure_measurements(rows: list[dict], lam0: float, mode: str, workdir: Path,
                               max_exposures: int = 10, url_cache: dict | None = None,
                               collect: list | None = None) -> list[dict]:
    """Measure the line in each exposure's cframe (and sky) rows.

    ``collect``, when given, is filled with the arrays that were read, so the
    caller can re-measure at other wavelengths -- the offset null -- without
    downloading tens of megabytes of frames a second time.
    """
    bands = desi_bands_for(lam0)
    out = []
    url_cache = url_cache if url_cache is not None else {}
    for r in rows[:max_exposures]:
        arms = []
        kept: list[dict] = []
        for band in bands:
            fw = lsf_fwhm_A(lam0, "DESI-DR1", band)
            key = ("cframe", r["night"], r["expid"], band, r["petal"])
            urls = url_cache.get(key) or desi_frame_urls("cframe", r["night"], r["expid"], band, r["petal"])
            data = None
            for url in urls:
                hd, how = open_fits_remote(url, workdir)
                if hd is None:
                    continue
                try:
                    data = desi_read_row(hd, ["FLUX", "IVAR", "MASK"], r["fiber"])
                finally:
                    hd.close()
                if data is not None:
                    url_cache[key] = [url]
                    data["how"] = how
                    break
            if data is None:
                arms.append({"testable": False, "reason": "cframe_unreachable", "band": band})
                continue
            sky = None
            for url in desi_frame_urls("sky", r["night"], r["expid"], band, r["petal"]):
                hd, _ = open_fits_remote(url, workdir)
                if hd is None:
                    continue
                try:
                    sk = desi_read_row(hd, ["SKY"], r["fiber"])
                finally:
                    hd.close()
                if sk is not None:
                    sky = sk["sky"]
                    break
            m = measure_line(data["wave"], data["flux"], data["ivar"], lam0, fw, mode,
                             mask=data.get("mask"), sky=sky, bad_bits=DESI_BAD_BITS,
                             cosmic_bits=DESI_COSMIC_BIT)
            m["band"] = band
            m["access"] = data.get("how")
            arms.append(m)
            if collect is not None:
                kept.append({"band": band, "wave": data["wave"], "flux": data["flux"],
                             "ivar": data["ivar"], "mask": data.get("mask"), "sky": sky})
        c = combine_measurements(arms)
        c.update({"expid": r["expid"], "night": r["night"], "tileid": r["tileid"],
                  "petal": r["petal"], "fiber": r["fiber"], "exptime": r.get("exptime"),
                  "arms": [a.get("band") for a in arms if a.get("testable")]})
        out.append(c)
        if collect is not None and kept:
            collect.append({"expid": r["expid"], "night": r["night"], "arms": kept})
        # Free downloaded frames as we go (they are tens of MB each).
        for f in workdir.glob("*frame-*.fits*"):
            try:
                f.unlink()
            except OSError:
                pass
        for f in workdir.glob("sky-*.fits*"):
            try:
                f.unlink()
            except OSError:
                pass
    return out


def desi_spectra_file_measurements(url: str, targetid: int, lam0: float, mode: str,
                                   workdir: Path, max_exposures: int = 10) -> list[dict]:
    """Fallback: per-exposure rows of the healpix ``spectra-*`` file."""
    hd, how = open_fits_remote(url, workdir, prefer_range=False)
    if hd is None:
        return []
    out = []
    try:
        fm = hd["FIBERMAP"].data
        tid = np.asarray(fm["TARGETID"]).astype(np.int64)
        idx = np.where(tid == int(targetid))[0][:max_exposures]
        for i in idx:
            arms = []
            for band in desi_bands_for(lam0):
                B = band.upper()
                try:
                    wave = np.asarray(hd[f"{B}_WAVELENGTH"].data, float)
                    flux = np.asarray(hd[f"{B}_FLUX"].section[int(i), :], float)
                    ivar = np.asarray(hd[f"{B}_IVAR"].section[int(i), :], float)
                    mask = np.asarray(hd[f"{B}_MASK"].section[int(i), :])
                except Exception as exc:  # noqa: BLE001
                    arms.append({"testable": False, "reason": f"read_failed: {exc!r}"})
                    continue
                m = measure_line(wave, flux, ivar, lam0, lsf_fwhm_A(lam0, "DESI-DR1", band),
                                 mode, mask=mask, bad_bits=DESI_BAD_BITS,
                                 cosmic_bits=DESI_COSMIC_BIT)
                m["band"] = band
                arms.append(m)
            c = combine_measurements(arms)
            row = fm[int(i)]
            c.update({"expid": int(row["EXPID"]) if "EXPID" in fm.names else int(i),
                      "night": int(row["NIGHT"]) if "NIGHT" in fm.names else None,
                      "tileid": int(row["TILEID"]) if "TILEID" in fm.names else None,
                      "source": "spectra_file"})
            out.append(c)
    finally:
        hd.close()
    return out


# ---------------------------------------------------------------------------
# SPARCL provenance and second epochs
# ---------------------------------------------------------------------------

_WANT_FIELDS = ["sparcl_id", "specid", "ra", "dec", "redshift", "spectype", "subtype",
                "subclass", "data_release", "wavelength", "flux", "ivar", "mask", "sky",
                # The pipeline's own per-pixel LSF width.  Without it the
                # resolved/unresolved test falls back to a nominal R = 2000,
                # which is exactly the approximation that made the triage's
                # width ratios unusable.
                "wave_sigma",
                "plate", "mjd", "fiberid", "run2d", "specobjid", "plateid",
                "targetid", "survey", "program", "healpix", "instrument", "dateobs",
                "dateobs_center", "exptime", "site", "telescope"]


def _rget(rec, key, default=None):
    from .acquire import _rget as rg
    return rg(rec, key, default)


def _records(obj):
    return getattr(obj, "records", obj)


def sparcl_fields(client, release: str) -> list[str]:
    """Field names SPARCL serves for ``release`` (client API differs by version)."""
    for fn in ("get_all_fields", "get_default_fields"):
        f = getattr(client, fn, None)
        if f is None:
            continue
        for call in (lambda g=f: g(dataset_list=[release]), lambda g=f: g([release]),
                     lambda g=f: g()):
            try:
                got = call()
                if got:
                    return [str(x) for x in got]
            except TypeError:
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[persist] {fn}({release}) failed: {exc!r}")
                break
    return []


# DESI healpix grouping of the spectro pipeline: nside 64, NESTED.
DESI_HPX_NSIDE = 64
# (survey, program) combinations that exist in DR1 (iron), most common first.
DESI_SURVEY_PROGRAMS = [("main", "dark"), ("main", "bright"), ("main", "backup"),
                        ("sv3", "dark"), ("sv3", "bright"), ("sv3", "backup"),
                        ("sv1", "dark"), ("sv1", "bright"), ("sv1", "backup"), ("sv1", "other"),
                        ("sv2", "dark"), ("sv2", "bright"), ("sv2", "backup"),
                        ("special", "dark"), ("special", "bright"), ("special", "backup"),
                        ("special", "other"), ("cmx", "other")]


def desi_healpix(ra: float, dec: float) -> int:
    from astropy import units as u
    from astropy_healpix import HEALPix
    hp = HEALPix(nside=DESI_HPX_NSIDE, order="nested")
    return int(hp.lonlat_to_healpix(float(ra) * u.deg, float(dec) * u.deg))


def desi_identity(rec: dict) -> dict:
    """targetid / survey / program / healpix for a SPARCL DESI record.

    SPARCL's ``specid`` for DESI *is* the TARGETID; the healpix follows from the
    position when the record does not carry it; survey/program are enumerated
    later by asking the archive which coadd files hold the target.
    """
    tid = rec.get("targetid")
    if tid is None:
        tid = rec.get("specid")
    try:
        tid = int(tid) if tid is not None else None
    except (TypeError, ValueError):
        tid = None
    hpx = rec.get("healpix")
    if hpx is None and np.isfinite(float(rec.get("ra", np.nan) or np.nan)):
        try:
            hpx = desi_healpix(float(rec["ra"]), float(rec["dec"]))
        except Exception as exc:  # noqa: BLE001
            print(f"[persist] healpix computation failed: {exc!r}")
    return {"targetid": tid, "survey": rec.get("survey"), "program": rec.get("program"),
            "healpix": int(hpx) if hpx is not None else None}


def desi_find_exposures(tid: int, hpx: int, survey=None, program=None,
                        workdir: Path | None = None) -> tuple[list[dict], list[dict]]:
    """Every (night, expid, petal, fiber) of ``tid`` across the DR1 coadd files
    of its healpix.  With survey/program unknown, each combination is tried by
    HEAD and the ones that exist are opened.  Returns (rows, files_tried)."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="desi_"))
    combos = [(survey, program)] if (survey and program) else DESI_SURVEY_PROGRAMS
    rows, tried, seen = [], [], set()
    for sv, pg in combos:
        h, url = None, ""
        for cand in desi_coadd_urls(sv, pg, hpx):
            url = cand
            h = http_head(cand)
            if h.get("status") == 200:
                break
        h = h or {"status": -1, "error": "no candidate URL"}
        entry = {"survey": sv, "program": pg, "url": url, "status": h.get("status"),
                 "n_rows": 0}
        if h.get("error"):
            entry["error"] = str(h["error"])[:300]
        if h.get("status") == 200:
            hd, how = open_fits_remote(url, workdir)
            if hd is not None:
                try:
                    got = desi_exposure_rows(hd, tid)
                finally:
                    hd.close()
                entry["access"] = how
                entry["n_rows"] = len(got)
                for r in got:
                    key = (r["expid"], r["petal"], r["fiber"])
                    if key not in seen:
                        seen.add(key)
                        r["survey"], r["program"] = sv, pg
                        rows.append(r)
            for f in workdir.glob("coadd-*.fits*"):
                try:
                    f.unlink()
                except OSError:
                    pass
        tried.append(entry)
        if rows and (survey and program):
            break
    return rows, tried


def sparcl_retrieve(client, ids: list[str], release: str) -> list[dict]:
    """Retrieve records with every wanted field the release actually has."""
    avail = sparcl_fields(client, release)
    inc = [f for f in _WANT_FIELDS if f in avail] if avail else \
          ["sparcl_id", "specid", "ra", "dec", "redshift", "spectype", "data_release",
           "wavelength", "flux", "ivar", "mask"]
    print(f"[persist] {release}: retrieving fields {inc}")
    out = []
    for k in range(0, len(ids), 100):
        chunk = ids[k:k + 100]
        try:
            got = _find_with_retry(lambda c=chunk: client.retrieve(uuid_list=c, include=inc,
                                                                   dataset_list=[release]))
        except TypeError:
            got = _find_with_retry(lambda c=chunk: client.retrieve(c, include=inc))
        except Exception as exc:  # noqa: BLE001
            print(f"[persist] retrieve failed: {exc!r}; retrying with default fields")
            try:
                got = _find_with_retry(lambda c=chunk: client.retrieve(uuid_list=c))
            except Exception as exc2:  # noqa: BLE001
                print(f"[persist] retrieve failed again: {exc2!r}")
                continue
        for r in _records(got):
            d = {}
            for f in set(inc) | {"sparcl_id", "specid", "wavelength", "flux", "ivar", "mask"}:
                v = _rget(r, f)
                if v is not None:
                    d[f] = v
            out.append(d)
    return out


def sdss_ids_from_record(rec: dict) -> dict | None:
    plate = rec.get("plate") or rec.get("plateid")
    mjd, fiber, run2d = rec.get("mjd"), rec.get("fiberid"), rec.get("run2d")
    if plate and mjd and fiber:
        return {"plate": int(plate), "mjd": int(mjd), "fiberid": int(fiber),
                "run2d": str(run2d) if run2d else None, "source": "sparcl_fields"}
    sid = rec.get("specobjid") or rec.get("specid")
    try:
        if sid is not None and int(sid) > (1 << 50):
            d = decode_specobjid(int(sid))
            d["source"] = "specobjid_decode"
            return d
    except (TypeError, ValueError):
        pass
    return None


def second_epoch(client, ra: float, dec: float, exclude_id: str, lam0: float, mode: str,
                 tol_arcsec: float = 2.0) -> dict:
    """Every other SPARCL spectrum at the position, with the line measured."""
    d = tol_arcsec / 3600.0
    cosd = max(np.cos(np.radians(dec)), 1e-3)
    cons = {"ra": [ra - d / cosd, ra + d / cosd], "dec": [dec - d, dec + d]}
    res = {"n_found_incl_self": 0, "self_found": False, "n_other": 0,
           "other_releases": "", "other_best_sig": float("nan"),
           "other_best_err_rel": float("nan"), "second_epoch": "none", "find_error": ""}
    try:
        found = _find_with_retry(lambda: client.find(
            outfields=["sparcl_id", "ra", "dec", "data_release", "dateobs_center"],
            constraints=cons, limit=50))
    except Exception as exc:  # noqa: BLE001
        res["find_error"] = repr(exc)
        return res
    recs = list(_records(found))
    ids = [str(_rget(r, "sparcl_id") or _rget(r, "id")) for r in recs]
    rels = [str(_rget(r, "data_release", "")) for r in recs]
    res["n_found_incl_self"] = len(ids)
    res["self_found"] = str(exclude_id) in ids
    others = [(i, rl) for i, rl in zip(ids, rels, strict=True) if i != str(exclude_id)]
    res["n_other"] = len(others)
    res["other_releases"] = ",".join(sorted({rl for _, rl in others}))
    if not others:
        return res
    best, best_rel = float("nan"), float("nan")
    for i, rl in others[:10]:
        try:
            got = _find_with_retry(lambda j=i: client.retrieve(
                uuid_list=[j], include=["sparcl_id", "wavelength", "flux", "ivar", "data_release"]))
        except Exception as exc:  # noqa: BLE001
            res["find_error"] = repr(exc)
            continue
        for r in _records(got):
            wave = np.asarray(_rget(r, "wavelength", []), float)
            flux = np.asarray(_rget(r, "flux", []), float)
            iv = np.asarray(_rget(r, "ivar", []), float)
            m = measure_line(wave, flux, iv, lam0, lsf_fwhm_A(lam0, rl), mode)
            if m.get("testable"):
                if not np.isfinite(best) or m["sig"] > best:
                    best, best_rel = m["sig"], m["err"]
    res["other_best_sig"] = best
    res["other_best_err_rel"] = best_rel
    if np.isfinite(best):
        res["second_epoch"] = "confirmed" if best >= 4.0 else "not_seen"
    return res


def sdss_pixel_A(lam: float) -> float:
    """Width in A of one SDSS/BOSS log-lambda pixel (1e-4 dex) at ``lam``."""
    return float(lam) * np.log(10.0) * 1e-4


def lsf_fwhm_measured(wave, wave_sigma, lam0: float, window_A: float = 20.0,
                      release: str = "") -> float:
    """Instrumental FWHM (A) at ``lam0`` from the pipeline's own LSF column.

    SPARCL serves ``wave_sigma`` (and an SDSS spec file a ``wdisp``) per pixel;
    using it instead of a nominal R = 2000 matters here, because SDSS's real
    resolution runs from about 1500 to 2500 across the spectrum and between
    fibres, and a line called "40 % broader than the LSF" against the wrong LSF
    is not an argument about anything.

    **Units.**  For SDSS the column is the pipeline's ``wdisp``, whose unit is
    the log-lambda PIXEL (1e-4 dex), not the angstrom.  Read as angstroms it
    gave 1.7-2.2 A FWHM at 6400-8800 A on every line of the first control run
    (35751666444) -- R = 3400-5100, which the SDSS spectrographs (R ~ 1500-2600)
    cannot deliver -- and every candidate came out "2-4x broader than the LSF".
    Multiplying by the pixel width gives 3.2-3.5 A, i.e. R ~ 2000-2500.  The
    control stage cross-checks this against the SAS file's own ``wdisp`` and
    against sky lines fitted in the same spectrum (``lsf_from_sky``).  For any
    other release the column is taken to be in angstroms.
    """
    w = np.asarray(wave, float)
    s = np.asarray(wave_sigma, float) if wave_sigma is not None else None
    if s is None or s.size != w.size:
        return float("nan")
    near = np.abs(w - float(lam0)) <= window_A
    v = s[near]
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return float("nan")
    sig = float(np.median(v))
    rel = (release or "").upper()
    if rel.startswith("SDSS") or rel.startswith("BOSS") or rel.startswith("EBOSS"):
        sig *= sdss_pixel_A(lam0)
    return float(2.3548 * sig)


def lsf_from_sky(wave, sky, lam0: float, search_A: float = 350.0, max_lines: int = 8,
                 min_sep_A: float = 10.0) -> dict:
    """The instrumental FWHM measured directly on sky lines in the same spectrum.

    Airglow lines are unresolved at R ~ 2000, so a Gaussian fitted to the
    brightest isolated sky-model peaks near ``lam0`` is the LSF with no unit
    convention to get wrong.  (OH Lambda-doublets are split by < 1 A in the red,
    so this runs a few per cent HIGH if anything -- which makes it conservative
    for the question "is the candidate broader than the LSF?".)
    """
    from scipy.optimize import curve_fit
    out = {"lsf_sky_fwhm_A": float("nan"), "lsf_sky_n": 0, "lsf_sky_mad_A": float("nan"),
           "lsf_sky_lines_A": []}
    w = np.asarray(wave, float)
    if sky is None:
        return out
    sk = np.asarray(sky, float)
    if sk.size != w.size or w.size < 50:
        return out
    win = np.isfinite(sk) & (np.abs(w - float(lam0)) <= search_A)
    if win.sum() < 30:
        return out
    idx = np.where(win)[0]
    base = float(np.median(sk[idx]))
    noise = _mad_std(sk[idx]) or 1e-9
    peaks = [i for i in idx[1:-1] if sk[i] >= sk[i - 1] and sk[i] >= sk[i + 1]
             and sk[i] - base > 8.0 * noise]
    peaks.sort(key=lambda i: -sk[i])
    chosen: list[int] = []
    for i in peaks:
        if all(abs(w[i] - w[j]) >= min_sep_A for j in chosen):
            # isolated: no comparably bright peak within min_sep_A
            if not any(abs(w[i] - w[k]) < min_sep_A and k != i and sk[k] - base
                       > 0.3 * (sk[i] - base) for k in peaks):
                chosen.append(i)
        if len(chosen) >= max_lines:
            break

    def _g(x, a, mu, sg, c0, c1):
        return a * np.exp(-0.5 * ((x - mu) / sg) ** 2) + c0 + c1 * x

    fw = []
    for i in chosen:
        sel = np.abs(w - w[i]) <= 0.6 * min_sep_A
        x = w[sel] - w[i]
        y = sk[sel]
        if sel.sum() < 7:
            continue
        pix = float(np.median(np.diff(w[sel])))
        try:
            pp, _ = curve_fit(_g, x, y, p0=[sk[i] - base, 0.0, 1.5 * pix, base, 0.0],
                              bounds=([0, -2 * pix, 0.3 * pix, -np.inf, -np.inf],
                                      [np.inf, 2 * pix, 6 * pix, np.inf, np.inf]),
                              maxfev=5000)
        except Exception:  # noqa: BLE001
            continue
        f = 2.3548 * float(pp[2])
        if np.isfinite(f):
            fw.append(f)
            out["lsf_sky_lines_A"].append(round(float(w[i]), 2))
    if fw:
        a = np.asarray(fw, float)
        out.update({"lsf_sky_fwhm_A": float(np.median(a)), "lsf_sky_n": int(a.size),
                    "lsf_sky_mad_A": float(_mad_std(a)) if a.size > 1 else float("nan")})
    return out


def fit_line_profile(wave, flux, ivar, lam0: float, fwhm_guess: float, mode: str = "emission",
                     fit_win_A: float = 25.0, cont_win_A: float = 60.0) -> dict:
    """Gaussian fit of the feature: centre, width, and width / instrumental width.

    A monochromatic source is by definition UNRESOLVED -- its profile is the
    instrument's LSF, so the fitted FWHM over the instrumental FWHM is 1 within
    the error.  A feature that is resolved cannot be a single narrow line
    whatever else it does, and this is the cheapest way to say so.  It is a
    measurement, not a cut: the fit is reported with its error so a 1.4 +- 0.3
    is not read as a 1.4 +- 0.05.
    """
    from scipy.optimize import curve_fit
    w = np.asarray(wave, float)
    f = np.asarray(flux, float)
    iv = np.asarray(ivar, float)
    out = {"fit_ok": False, "fit_reason": "", "fit_center_A": float("nan"),
           "fit_dv_kms": float("nan"), "fit_fwhm_A": float("nan"),
           "fit_fwhm_err_A": float("nan"), "fit_amp": float("nan"),
           "fit_amp_sig": float("nan"), "fit_at_bound": False,
           "fit_reduced_chi2": float("nan")}
    good = np.isfinite(w) & np.isfinite(f) & np.isfinite(iv) & (iv > 0)
    d = w - float(lam0)
    ann = good & (np.abs(d) <= cont_win_A) & (np.abs(d) > 2.0 * fwhm_guess)
    sel = good & (np.abs(d) <= fit_win_A)
    if int(sel.sum()) < 7 or int(ann.sum()) < 8:
        out["fit_reason"] = "too few pixels"
        return out
    try:
        coef = np.polyfit(w[ann] - lam0, f[ann], 1)[::-1]
    except (np.linalg.LinAlgError, ValueError):
        out["fit_reason"] = "continuum fit failed"
        return out
    r = f[sel] - (coef[0] + coef[1] * d[sel])
    sign = -1.0 if mode == "absorption" else 1.0
    y = sign * r
    sy = 1.0 / np.sqrt(iv[sel])

    def _g(x, amp, mu, sig):
        return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    p0 = [max(float(np.max(y)), 1e-6), 0.0, max(fwhm_guess / 2.3548, 1e-3)]
    # A spectral feature cannot be narrower than the LSF.  The lower bound is
    # set just below it so that a fit which WANTS to go narrower is reported as
    # having hit the bound -- that is a one-pixel defect or a cosmic ray, and it
    # must read as a flag, not as a suspiciously precise width.
    s_lo, s_hi = 0.35 * fwhm_guess / 2.3548, 8.0 * fwhm_guess / 2.3548
    lo = [0.0, -2.0 * fwhm_guess, s_lo]
    hi = [1e6 * abs(p0[0]) + 1e6, 2.0 * fwhm_guess, s_hi]
    try:
        p, cov = curve_fit(_g, d[sel], y, p0=p0, sigma=sy, absolute_sigma=True,
                           bounds=(lo, hi), maxfev=20000)
    except Exception as exc:  # noqa: BLE001
        out["fit_reason"] = repr(exc)[:200]
        return out
    perr = np.sqrt(np.diag(cov)) if cov is not None and np.all(np.isfinite(cov)) \
        else np.full(3, np.nan)
    resid = y - _g(d[sel], *p)
    dof = max(int(sel.sum()) - 3, 1)
    out.update({
        "fit_at_bound": bool(p[2] <= s_lo * 1.01 or p[2] >= s_hi * 0.99),
        "fit_reduced_chi2": float(np.sum((resid / sy) ** 2) / dof),
        "fit_ok": True,
        "fit_center_A": float(lam0 + p[1]),
        "fit_dv_kms": float(p[1] / lam0 * 299792.458),
        "fit_fwhm_A": float(2.3548 * p[2]),
        "fit_fwhm_err_A": float(2.3548 * perr[2]),
        "fit_amp": float(sign * p[0]),
        "fit_amp_sig": float(p[0] / perr[0]) if np.isfinite(perr[0]) and perr[0] > 0
        else float("nan"),
    })
    return out


def _control_stats(meas: list[dict], key: str = "sig") -> dict:
    v = np.array([m[key] for m in meas if m.get("testable") and np.isfinite(m.get(key, np.nan))],
                 float)
    if v.size == 0:
        return {"n_measured": 0}
    return {"n_measured": int(v.size), "sig_median": float(np.median(v)),
            "sig_mad": float(_mad_std(v)), "sig_max": float(np.max(v)),
            "frac_ge3": float(np.mean(v >= 3.0)), "frac_ge5": float(np.mean(v >= 5.0)),
            "frac_ge8": float(np.mean(v >= 8.0))}


def background_galaxy_scan(wave, flux, ivar, lam0: float, release: str,
                           mode: str = "emission", min_sig: float = 3.0) -> dict:
    """Is the line one nebular line of a background galaxy in the same fibre?

    ``galaxy_reject`` already asks this, but it asks it of the CANDIDATE LIST:
    it needs two surviving candidates in one spectrum to land on one redshift.
    A galaxy whose Halpha clears the 8-sigma search threshold while its [N II]
    and [S II] do not therefore leaves exactly one candidate and passes.  That
    is the common case, not the rare one -- [N II] 6584 is typically 0.3 of
    Halpha in a star-forming galaxy, so a 16-sigma Halpha comes with a 5-sigma
    companion that was never a candidate.

    This asks it of the SPECTRUM.  Each strong nebular line is tried as the
    anchor; at the redshift that implies, every OTHER line of the family is
    measured directly in the data, however weak.  A real galaxy answers with a
    family at one redshift, and the line ratios say which kind.
    """
    from .galaxy_reject import GALAXY_LINES
    w = np.asarray(wave, float)
    f = np.asarray(flux, float)
    iv = np.asarray(ivar, float)
    best: dict = {"anchor": "", "z": float("nan"), "n_companions_ge3": 0,
                  "companions": [], "tested": 0}
    if str(mode) == "absorption":
        # Nebular emission cannot explain an absorption survivor.
        best["error"] = "absorption mode: a nebular family is not an explanation"
        return best
    if w.size < 50:
        best["error"] = "no spectrum"
        return best
    results = []
    for anchor, rest in GALAXY_LINES.items():
        z = float(lam0) / float(rest) - 1.0
        if not (-0.002 <= z <= 1.2):
            continue
        comps = []
        for name, r2 in GALAXY_LINES.items():
            if name == anchor:
                continue
            obs = r2 * (1.0 + z)
            # A close doublet partner ([O II] 3727/3729, [N II]/Halpha at high
            # z) can land on the candidate itself, and re-measuring the same
            # feature is not a companion.
            if abs(obs - float(lam0)) < 2.0 * lsf_fwhm_A(float(lam0), release):
                continue
            m = measure_line(w, f, iv, obs, lsf_fwhm_A(obs, release), "emission")
            if not m.get("testable"):
                continue
            comps.append({"line": name, "obs_A": round(obs, 2),
                          "sig": round(float(m["sig"]), 2),
                          "ew_A": round(float(m["ew"]), 3) if np.isfinite(m["ew"]) else None})
        if not comps:
            continue
        n3 = sum(1 for c in comps if c["sig"] >= min_sig)
        strongest = max((c["sig"] for c in comps), default=float("nan"))
        results.append({"anchor": anchor, "z": round(z, 6), "n_companions_ge3": n3,
                        "n_companions_tested": len(comps),
                        "strongest_companion_sig": strongest,
                        "companions": sorted(comps, key=lambda c: -c["sig"])[:6]})
    if not results:
        best["error"] = "no anchor gives a redshift with covered companions"
        return best
    results.sort(key=lambda r: (r["n_companions_ge3"], r["strongest_companion_sig"]),
                 reverse=True)
    out = dict(results[0])
    out["tested"] = len(results)
    out["all_anchors"] = [{k: r[k] for k in ("anchor", "z", "n_companions_ge3",
                                             "strongest_companion_sig")}
                          for r in results]
    return out


# Night-sky and artificial-light emission, VACUUM wavelengths (SDSS and DESI
# spectra are on vacuum grids; the air values the literature quotes are 1.5-2.4 A
# shorter in the optical).  Not the OH forest -- that lives in linelist.py -- but
# the atomic lines a narrow-line search trips over.
SKY_ATOMIC_VAC = {
    "[OI]5577 airglow": 5578.89, "[OI]6300 airglow": 6302.05, "[OI]6364 airglow": 6365.54,
    "NaD2 airglow/lamp": 5891.58, "NaD1 airglow/lamp": 5897.56,
    "Hg 4047 lamp": 4047.71, "Hg 4358 lamp": 4359.56, "Hg 5461 lamp": 5462.27,
    "Hg 5770 lamp": 5771.20, "Hg 5791 lamp": 5792.27,
    "Na HPS 5683 lamp": 5684.20, "Na HPS 5688 lamp": 5689.78,
    "Na HPS 8183 lamp": 8185.50, "Na HPS 8195 lamp": 8197.05,
    "K I 7665 lamp": 7667.02, "K I 7699 lamp": 7701.08,
}


def nearest_sky_atomic(lam: float) -> dict:
    """The nearest listed atomic sky / street-lamp line to an observed wavelength."""
    name, ref = min(SKY_ATOMIC_VAC.items(), key=lambda kv: abs(kv[1] - float(lam)))
    return {"sky_atomic_line": name, "sky_atomic_dA": round(float(lam) - ref, 2)}


def nebular_family_calibrated(spectra: list[dict], lam0: float, release: str,
                              n_null: int = 24, seed: int = 11, min_sig: float = 3.0) -> dict:
    """Background-galaxy test on EVERY epoch at once, calibrated against its own null.

    ``background_galaxy_scan`` measures the companions in one coadd, and a single
    SDSS epoch of an M dwarf is too shallow to decide it: for 0412-51942-0465 the
    companions at the Halpha redshift came out at 1.9-3.9 sigma, suggestive and
    not decisive.  The same object has nine epochs.  Here each companion is
    measured in every spectrum of the position and combined by inverse variance,
    which is sqrt(N) deeper; and because the per-spectrum estimator carries a
    small bias that a combination multiplies by sqrt(N) (the lesson of the
    per-exposure null), the combined significance is re-measured at ``n_null``
    random offsets 12-200 A from each companion in the same spectra and quoted
    against that null: ``sig_cal = (sig - median_null) / max(MAD_null, 1)``.

    ``spectra``: dicts with ``wave, flux, ivar``.  Returns every anchor tried,
    best first by the number of calibrated companions at >= ``min_sig``.
    """
    from .galaxy_reject import GALAXY_LINES
    out: dict = {"n_spectra": len(spectra), "anchors": [], "best": {}}
    sp = [d for d in spectra if np.asarray(d.get("wave", []), float).size >= 50]
    out["n_spectra"] = len(sp)
    if not sp:
        out["error"] = "no spectra"
        return out
    rng = np.random.default_rng(seed)
    offs = rng.uniform(12.0, 200.0, n_null) * rng.choice([-1.0, 1.0], n_null)

    def _comb(lam):
        fw = lsf_fwhm_A(lam, release)
        ms = [measure_line(d["wave"], d["flux"], d["ivar"], lam, fw, "emission") for d in sp]
        ms = [m for m in ms if m.get("testable") and np.isfinite(m.get("err", np.nan))
              and m["err"] > 0]
        if not ms:
            return None, 0
        wts = np.array([1.0 / m["err"] ** 2 for m in ms])
        F = float(np.sum(wts * np.array([m["F"] for m in ms])) / np.sum(wts))
        return F * np.sqrt(np.sum(wts)), len(ms)

    for anchor, rest in GALAXY_LINES.items():
        z = float(lam0) / float(rest) - 1.0
        if not (-0.002 <= z <= 1.2):
            continue
        comps = []
        for name, r2 in GALAXY_LINES.items():
            if name == anchor:
                continue
            obs = r2 * (1.0 + z)
            if abs(obs - float(lam0)) < 2.0 * lsf_fwhm_A(float(lam0), release):
                continue
            sig, n_used = _comb(obs)
            if sig is None:
                continue
            null = [v for v in (_comb(obs + o)[0] for o in offs) if v is not None]
            if len(null) >= 8:
                med = float(np.median(null))
                mad = max(float(_mad_std(np.asarray(null, float))), 1.0)
            else:
                med, mad = 0.0, 1.0
            comps.append({"line": name, "obs_A": round(obs, 2), "n_spectra": n_used,
                          "sig_raw": round(float(sig), 2), "null_median": round(med, 2),
                          "null_mad": round(mad, 2), "n_null": len(null),
                          "sig_cal": round((float(sig) - med) / mad, 2)})
        if not comps:
            continue
        n3 = sum(1 for c in comps if c["sig_cal"] >= min_sig)
        out["anchors"].append({"anchor": anchor, "z": round(z, 6), "n_companions_ge3_cal": n3,
                               "strongest_cal": max(c["sig_cal"] for c in comps),
                               "companions": sorted(comps, key=lambda c: -c["sig_cal"])})
    out["anchors"].sort(key=lambda a: (a["n_companions_ge3_cal"], a["strongest_cal"]),
                        reverse=True)
    if out["anchors"]:
        out["best"] = {k: out["anchors"][0][k] for k in
                       ("anchor", "z", "n_companions_ge3_cal", "strongest_cal")}
    return out


def sdss_lite_urls(plate: int, mjd: int, fiber: int, run2d: str | None) -> list[str]:
    """SAS URLs for the LITE (coadd-only) spec file of one fibre, best first."""
    name = f"spec-{int(plate):04d}-{int(mjd)}-{int(fiber):04d}.fits"
    r2d = str(run2d or "").strip()
    order = [r2d] if r2d else []
    order += [a for a in ("26", "104", "103", "v5_13_2") if a != r2d]
    urls = []
    for a in order:
        base = "eboss" if a.startswith("v") else "sdss"
        urls.append(f"{SDSS_SAS}/{base}/spectro/redux/{a}/spectra/lite/{int(plate):04d}/{name}")
    return urls


def _fetch_sdss_coadd(plate: int, mjd: int, fiber: int, run2d: str | None) -> dict | None:
    """One fibre's coadd (wave, flux, ivar, sky, wdisp) from the SAS lite file."""
    from astropy.io import fits
    urls = sdss_lite_urls(plate, mjd, fiber, run2d) + sdss_spec_urls(plate, mjd, fiber, run2d)[:2]
    for url in urls:
        data = fetch_bytes(url, max_bytes=60_000_000, tries=2)
        if data is None:
            continue
        try:
            with fits.open(io.BytesIO(data), memmap=False) as hd:
                got = parse_sdss_spec(hd)
        except Exception:  # noqa: BLE001
            continue
        if got.get("coadd") is not None:
            got["coadd"]["url"] = url
            return got["coadd"]
    return None


def sdss_fibre_neighbours(plate: int, mjd: int, fiber: int, run2d: str | None, lam0: float,
                          mode: str, lsf_A: float, n_random: int = 24, seed: int = 3,
                          fetch=None) -> dict:
    """The same wavelength in the OTHER fibres of the same plate and night.

    Two things that SPARCL cannot be asked (its ``find`` takes no plate or fibre
    constraint -- the first control run's same-plate sample failed on exactly
    that, ``UnknownField``):

    * **cross-talk**: on the slit head fibre ``k`` sits between ``k-1`` and
      ``k+1``, so a bright emission line in a neighbour leaks a narrow copy into
      its trace at the same wavelength.  The neighbours' own line flux at
      ``lam0`` is measured, and the ratio target / neighbour is what decides it
      (SDSS cross-talk is at the <~1 % level).
    * **a detector column**: a bad column puts a feature at one wavelength in
      many fibres of that plate; ``n_random`` other fibres of the plate are
      measured at the same wavelength.

    Per-fibre coadds are read from the SAS lite files.  ``fetch(plate, mjd,
    fiber, run2d)`` can be injected for tests.
    """
    fetch = fetch or _fetch_sdss_coadd
    per_spec = 320 if str(run2d or "").strip() in ("26", "103", "104") else 500
    n_fib = 2 * per_spec
    block = (int(fiber) - 1) // per_spec
    lo, hi = block * per_spec + 1, (block + 1) * per_spec
    nb = [f for f in (fiber - 3, fiber - 2, fiber - 1, fiber + 1, fiber + 2, fiber + 3)
          if lo <= f <= hi]
    rng = np.random.default_rng(seed + int(plate))
    pool = [f for f in range(1, n_fib + 1) if abs(f - fiber) > 3]
    rand = sorted(int(x) for x in rng.choice(pool, size=min(n_random, len(pool)),
                                             replace=False))
    out = {"plate": int(plate), "mjd": int(mjd), "fiber": int(fiber),
           "fibres_per_spectrograph": per_spec, "target": None,
           "neighbours": [], "same_plate": {"n_measured": 0}, "same_plate_fibres": []}

    def _m(f):
        co = fetch(int(plate), int(mjd), int(f), run2d)
        if co is None:
            return None, None
        m = measure_line(co["wave"], co["flux"], co["ivar"], lam0, lsf_A, mode,
                         sky=co.get("sky"))
        return m, co

    tm, tco = _m(fiber)
    if tm is not None:
        out["target"] = _json_safe({k: tm.get(k) for k in
                                    ("testable", "F", "err", "sig", "ew", "cont",
                                     "sky_peak_sig")})
        if tco is not None and tco.get("wdisp") is not None:
            wd = np.asarray(tco["wdisp"], float)
            near = (np.abs(np.asarray(tco["wave"], float) - lam0) <= 20.0) & \
                np.isfinite(wd) & (wd > 0)
            if near.any():
                out["target_wdisp_pix"] = float(np.median(wd[near]))
                out["lsf_file_wdisp_fwhm_A"] = float(2.3548 * np.median(wd[near])
                                                     * sdss_pixel_A(lam0))
        out["target_file"] = (tco or {}).get("url")
    tF = (tm or {}).get("F")
    for f in nb:
        m, _co = _m(f)
        if m is None:
            out["neighbours"].append({"fiber": f, "offset": f - int(fiber),
                                      "testable": False, "reason": "unreachable"})
            continue
        rec = {"fiber": f, "offset": f - int(fiber)}
        rec.update({k: m.get(k) for k in ("testable", "F", "err", "sig", "ew", "cont")})
        if (m.get("testable") and tF is not None and np.isfinite(tF)
                and np.isfinite(m.get("F", np.nan)) and m["F"] > 0):
            rec["target_over_neighbour_F"] = float(tF / m["F"])
        out["neighbours"].append(_json_safe(rec))
    meas = []
    for f in rand:
        m, _co = _m(f)
        if m is None:
            continue
        meas.append(m)
        out["same_plate_fibres"].append(_json_safe({"fiber": f, "sig": m.get("sig"),
                                                    "testable": m.get("testable")}))
    out["same_plate"] = _control_stats(meas)
    tn = [n for n in out["neighbours"] if n.get("testable")]
    sigs = [float(n["sig"]) for n in tn if n.get("sig") is not None
            and np.isfinite(float(n["sig"]))]
    out["max_neighbour_sig"] = max(sigs) if sigs else None
    # Cross-talk is plausible only if a neighbour HAS a line at this wavelength
    # and it is bright enough that <~2 % of it would make the target's.
    out["crosstalk_plausible"] = bool(any(
        float(n.get("sig") or 0) >= 5.0
        and 0 < float(n.get("target_over_neighbour_F") or 1e9) <= 0.02 for n in tn))
    return out


def _control_measure(client, release: str, ids: list[str], zs: dict, lam0: float, mode: str,
                     z_cand: float) -> dict:
    """Observed-frame and stellar-frame measurements of ``lam0`` in ``ids``."""
    lam_rest = float(lam0) / (1.0 + float(z_cand or 0.0))
    obs, star = [], []
    got = sparcl_retrieve(client, ids, release)
    for r in got:
        wave = np.asarray(r.get("wavelength", []), float)
        flux = np.asarray(r.get("flux", []), float)
        iv = np.asarray(r.get("ivar", []), float)
        if wave.size < 50:
            continue
        obs.append(measure_line(wave, flux, iv, lam0, lsf_fwhm_A(lam0, release), mode))
        try:
            zc = float(zs.get(str(r.get("sparcl_id")), 0.0) or 0.0)
        except (TypeError, ValueError):
            zc = 0.0
        lam_star = lam_rest * (1.0 + zc)
        star.append(measure_line(wave, flux, iv, lam_star, lsf_fwhm_A(lam_star, release),
                                 mode))
    return {"n_retrieved": len(got), "obs_frame": _control_stats(obs),
            "star_frame": _control_stats(star)}


def same_type_sample(client, release: str, lam0: float, mode: str, z_cand: float,
                     subclass: str, n: int = 40, exclude_ids: tuple = (), seed: int = 0,
                     pool: int = 2000) -> dict:
    """``control_sample`` restricted to stars the survey itself typed ``subclass``.

    SPARCL's ``find`` does not take ``subclass`` (run 35751666444: UnknownField),
    so the first control run's "same-type" sample silently fell back to "any
    star" -- with the same random seed, i.e. the SAME forty spectra, and every
    same-type number in that control.json is a copy of the any-star number.
    Here a pool of stars is found, their ``subclass`` is retrieved (a
    metadata-only retrieve), and the sample is drawn from the matches.  Exact
    subclass first (M1), then the same class letter (M) if fewer than 10 match;
    which one was used is on the record.
    """
    out = {"release": release, "wavelength": lam0, "subclass": subclass,
           "n_requested": int(n), "constraint": "", "error": "",
           "obs_frame": {"n_measured": 0}, "star_frame": {"n_measured": 0}}
    try:
        found = _find_with_retry(lambda: client.find(
            outfields=["sparcl_id", "redshift"],
            constraints={"data_release": [release], "spectype": ["STAR"]}, limit=int(pool)))
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"find: {exc!r}"[:300]
        return out
    recs = [r for r in _records(found) if str(_rget(r, "sparcl_id") or "") not in
            set(exclude_ids)]
    zs = {str(_rget(r, "sparcl_id")): _rget(r, "redshift", 0.0) for r in recs}
    ids = list(zs)
    key = "subtype" if release.upper().startswith("DESI") else "subclass"
    typed: dict[str, str] = {}
    for k in range(0, len(ids), 500):
        chunk = ids[k:k + 500]
        try:
            got = _find_with_retry(lambda c=chunk: client.retrieve(
                uuid_list=c, include=["sparcl_id", key], dataset_list=[release]))
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"retrieve {key}: {exc!r}"[:300]
            continue
        for r in _records(got):
            typed[str(_rget(r, "sparcl_id"))] = survey_subclass(_rget(r, key))
    out["n_pool"], out["n_pool_typed"] = len(ids), len(typed)
    exact = [i for i, t in typed.items() if t == subclass]
    letter = [i for i, t in typed.items() if t[:1] == subclass[:1]]
    if len(exact) >= 10:
        chosen, out["constraint"] = exact, f"{key}={subclass} (retrieved, exact)"
    elif len(letter) >= 10:
        chosen, out["constraint"] = letter, f"{key}={subclass[:1]}* (retrieved, class letter)"
    else:
        out["error"] = out["error"] or (f"only {len(exact)} exact / {len(letter)} same-letter "
                                        f"{key} matches in a pool of {len(typed)}")
        return out
    out["n_matching"] = len(chosen)
    rng = np.random.default_rng(seed)
    pick = [chosen[int(i)] for i in rng.permutation(len(chosen))[:int(n)]]
    out.update(_control_measure(client, release, pick, zs, lam0, mode, z_cand))
    return out


def survey_subclass(sptype: str | None) -> str:
    """A SIMBAD spectral type reduced to what a survey pipeline calls a subclass.

    SIMBAD says ``M1V``, ``M4V``, ``K3III``; SDSS's ``subclass`` and DESI's
    ``subtype`` say ``M1``, ``M4``, ``K3``.  Asking the archive for ``M1V``
    returns nothing, the query falls back to "any star", and the same-type
    control silently becomes a second copy of the all-stars control -- which
    would look like a clean result and mean nothing.
    """
    if not sptype:
        return ""
    # Search rather than match: SIMBAD prefixes luminosity/metallicity markers
    # ("dM4e", "sdF8"), and the first class letter is the one that matters.
    m = re.search(r"([OBAFGKMLTY])\s*(\d)?", str(sptype).upper())
    if not m:
        return ""
    return m.group(1) + (m.group(2) or "")


def control_sample(client, release: str, lam0: float, mode: str, z_cand: float,
                   subclass: str | None = None, n: int = 40,
                   exclude_ids: tuple = (), seed: int = 0,
                   extra_constraint: dict | None = None, label: str = "") -> dict:
    """Measure the same line in unrelated spectra of the same kind of star.

    The per-exposure test and a second epoch cannot reject a feature that the
    star's spectral TYPE produces -- a gap between molecular band heads in an M
    dwarf persists in every exposure of that star and in every epoch of it.
    What rejects it is that other stars of the same type show it too.  Two
    controls are measured on the same sample and they separate the two ways a
    candidate can be uninteresting:

    * **observed frame** -- the same observed wavelength in every control star.
      A feature that is atmospheric (an OH line the list does not carry) or
      instrumental sits at a fixed observed wavelength and shows up here.
    * **stellar frame** -- the candidate's rest wavelength, redshifted to each
      control star's own catalogue redshift.  A photospheric or molecular
      feature of that spectral type shows up here.

    The two frames only separate when the velocities differ by more than a
    resolution element: at 6800 A the SDSS line window is +-4 A, so stars within
    ~180 km/s of each other land in the same window and the two numbers say the
    same thing.  For Galactic stars that is the usual case, so the frames are
    reported but the *type* dependence -- the same measurement on a same-type
    sample and on an all-stars sample, which :func:`controls` runs -- is what
    separates "this spectral type does this" from "the sky or the instrument
    does this".

    A candidate that appears in neither control is peculiar to its object.
    """
    out = {"release": release, "wavelength": lam0, "subclass": subclass,
           "n_requested": int(n), "constraint": "", "error": "",
           "obs_frame": {"n_measured": 0}, "star_frame": {"n_measured": 0}}
    cons: dict = {"data_release": [release], "spectype": ["STAR"]}
    fields = ["sparcl_id", "data_release", "redshift", "spectype"]
    if extra_constraint:
        # A same-plate sample is not about stars at all: it asks whether OTHER
        # FIBRES of the same exposure set show the feature at the same
        # wavelength, which is what a bad CCD column does and what nothing
        # upstream can see, because it is in every exposure of that plate and
        # in every repeat observation of it.
        cons = {"data_release": [release], **extra_constraint}
        out["constraint"] = label or str(extra_constraint)
        tries = [(cons, out["constraint"])]
        subclass = None
    else:
        tries = []
    if subclass:
        for key in ("subclass", "subtype"):
            tries.append(({**cons, key: [subclass]}, f"{key}={subclass}"))
    # The label the archive uses must be on the record, so a same-type control
    # that quietly degraded to "any star" can be seen to have done so.
    tries.append((cons, "spectype=STAR (subclass not matched)" if subclass
                  else "spectype=STAR"))
    recs = []
    for c, label in tries:
        try:
            found = _find_with_retry(lambda c=c: client.find(
                outfields=fields, constraints=c, limit=int(max(n * 6, 120))))
        except Exception as exc:  # noqa: BLE001
            out["error"] = repr(exc)[:300]
            continue
        recs = [r for r in _records(found)
                if str(_rget(r, "sparcl_id") or "") not in set(exclude_ids)]
        if recs:
            out["constraint"] = label
            break
    if not recs:
        out["error"] = out["error"] or "no control spectra returned"
        return out
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(recs))[:int(n)]
    ids = [str(_rget(recs[int(i)], "sparcl_id")) for i in idx]
    zs = {str(_rget(r, "sparcl_id")): _rget(r, "redshift", 0.0) for r in recs}
    lam_rest = float(lam0) / (1.0 + float(z_cand or 0.0))
    obs, star = [], []
    got = sparcl_retrieve(client, ids, release)
    out["n_retrieved"] = len(got)
    for r in got:
        wave = np.asarray(r.get("wavelength", []), float)
        flux = np.asarray(r.get("flux", []), float)
        iv = np.asarray(r.get("ivar", []), float)
        if wave.size < 50:
            continue
        fwhm = lsf_fwhm_A(lam0, release)
        obs.append(measure_line(wave, flux, iv, lam0, fwhm, mode))
        try:
            zc = float(zs.get(str(r.get("sparcl_id")), 0.0) or 0.0)
        except (TypeError, ValueError):
            zc = 0.0
        lam_star = lam_rest * (1.0 + zc)
        star.append(measure_line(wave, flux, iv, lam_star,
                                 lsf_fwhm_A(lam_star, release), mode))
    out["obs_frame"] = _control_stats(obs)
    out["star_frame"] = _control_stats(star)
    return out


def epoch_series(client, ra: float, dec: float, exclude_id: str, lam0: float, mode: str,
                 tol_arcsec: float = 2.0, max_epochs: int = 20,
                 keep: list | None = None) -> dict:
    """Every SPARCL spectrum at the position, measured one by one.

    ``second_epoch`` keeps only the best of them, which answers "was it seen
    again" and nothing else.  The series answers the question that follows: a
    line of constant strength across years is a stable property of the star;
    one that varies is either a real variable or a reduction artefact, and
    which of those it is depends on how it varies.  Kept out of the per-exposure
    checkpoints on purpose, so adding it cannot invalidate a run in flight.
    """
    d = tol_arcsec / 3600.0
    cosd = max(np.cos(np.radians(dec)), 1e-3)
    cons = {"ra": [ra - d / cosd, ra + d / cosd], "dec": [dec - d, dec + d]}
    out = {"n_found": 0, "epochs": [], "error": ""}
    try:
        found = _find_with_retry(lambda: client.find(
            outfields=["sparcl_id", "ra", "dec", "data_release", "dateobs_center"],
            constraints=cons, limit=50))
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)[:300]
        return out
    recs = list(_records(found))
    out["n_found"] = len(recs)
    by_rel: dict[str, list[str]] = {}
    when: dict[str, str] = {}
    for r in recs:
        sid = str(_rget(r, "sparcl_id") or "")
        rel = str(_rget(r, "data_release", ""))
        if not sid:
            continue
        when[sid] = str(_rget(r, "dateobs_center", ""))
        by_rel.setdefault(rel, []).append(sid)
    seen_fibres: set = set()
    for rel, ids in by_rel.items():
        for got in (sparcl_retrieve(client, ids[:max_epochs], rel),):
            for r in got:
                sid = str(r.get("sparcl_id"))
                w = np.asarray(r.get("wavelength", []), float)
                if w.size < 50:
                    continue
                lsf = lsf_fwhm_measured(w, r.get("wave_sigma"), lam0, release=rel)
                if not np.isfinite(lsf):
                    lsf = lsf_fwhm_A(lam0, rel)
                m = measure_line(w, np.asarray(r.get("flux", []), float),
                                 np.asarray(r.get("ivar", []), float), lam0, lsf, mode,
                                 sky=r.get("sky"))
                if keep is not None:
                    # the arrays, for the epoch-stacked background-galaxy test
                    keep.append({"spec_id": sid, "wave": w,
                                 "flux": np.asarray(r.get("flux", []), float),
                                 "ivar": np.asarray(r.get("ivar", []), float)})
                fit = fit_line_profile(w, np.asarray(r.get("flux", []), float),
                                       np.asarray(r.get("ivar", []), float), lam0, lsf, mode)
                # WHICH fibre of which plate: SDSS repeat spectra of one object
                # are very often the same plate and fibre on another night, i.e.
                # the same CCD column.  Eight "independent epochs" that are all
                # one fibre confirm a detector defect exactly as well as they
                # confirm a source, and the summary's best-of number cannot say
                # which.
                # SDSS only: a DESI TARGETID is > 2**50 and would be decoded
                # as a specObjID into a fictitious "plate 2048" (control run
                # 35863951810 did exactly that).
                ids_ = (sdss_ids_from_record(r) or {}) if rel.upper().startswith(
                    ("SDSS", "BOSS")) else {}
                fibre_key = (ids_.get("plate"), ids_.get("fiberid"))
                if fibre_key != (None, None):
                    seen_fibres.add(fibre_key)
                out["epochs"].append(_json_safe({
                    "spec_id": sid, "is_self": sid == str(exclude_id),
                    "data_release": rel, "dateobs_center": when.get(sid, ""),
                    "plate": ids_.get("plate"), "mjd": ids_.get("mjd"),
                    "fiberid": ids_.get("fiberid"),
                    "testable": m.get("testable"), "reason": m.get("reason"),
                    "sig": m.get("sig"), "ew": m.get("ew"), "cont": m.get("cont"),
                    "sky_peak_sig": m.get("sky_peak_sig"),
                    "fit_fwhm_A": fit.get("fit_fwhm_A"), "lsf_fwhm_A": lsf,
                    "fwhm_over_lsf": (float(fit["fit_fwhm_A"]) / lsf
                                      if fit.get("fit_ok") and lsf > 0 else None),
                    "fit_dv_kms": fit.get("fit_dv_kms"),
                }))
    ok = [e for e in out["epochs"] if e.get("testable") and e.get("ew") is not None]
    if ok:
        ew = np.array([float(e["ew"]) for e in ok], float)
        sg = np.array([float(e["sig"]) for e in ok], float)
        out["n_measured"] = len(ok)
        out["ew_median"] = float(np.median(ew))
        out["ew_spread_frac"] = (float(_mad_std(ew) / abs(np.median(ew)))
                                 if np.median(ew) else float("nan"))
        out["n_sig_ge4"] = int(np.sum(sg >= 4.0))
        out["sig_min"], out["sig_max"] = float(np.min(sg)), float(np.max(sg))
    out["n_distinct_fibres"] = len(seen_fibres)
    out["distinct_fibres"] = sorted(f"{p}-{f}" for p, f in seen_fibres)
    return out


def controls(root: Path, n: int = 40, classes: tuple = ("persistent", "persistent_2exp",
                                                        "stack_only", "partial"),
             max_seconds: float = 8400.0) -> dict:
    """Run the comparison-sample control on every line still standing.

    Writes ``control.json`` after every line and stops when ``max_seconds`` is
    spent, so a job that runs long commits what it measured instead of being
    killed with nothing: three samples of 40 spectra plus an epoch series per
    line is a lot of SPARCL traffic and the number of lines is not known in
    advance.
    """
    import pandas as pd
    t_start = time.time()
    root = Path(root)
    out_dir = root / "results" / "spectra_persist"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "persistence.csv"
    if not p.exists():
        rep = {"error": "no persistence.csv: run the reduce stage first", "entries": []}
        (out_dir / "control.json").write_text(json.dumps(rep, indent=2))
        return rep
    tab = pd.read_csv(p)
    sel = tab[tab["persistence_class"].astype(str).isin(classes)]
    sel = sel.sort_values("combined_sig", ascending=False)
    print(f"[persist] control sample for {len(sel)} lines in classes {classes}")
    client = _make_client()
    entries = []
    for _, r in sel.iterrows():
        rel = str(r.get("data_release"))
        raw_sp = r.get("simbad_sptype")
        raw_sp = "" if (raw_sp is None or
                        (isinstance(raw_sp, float) and not np.isfinite(raw_sp))) else str(raw_sp)
        sub = survey_subclass(raw_sp) or None
        e = {"spec_id": str(r["spec_id"]), "identifier": r.get("identifier"),
             "wavelength": float(r["wavelength"]), "search_mode": str(r.get("search_mode")),
             "persistence_class": str(r.get("persistence_class")),
             "combined_sig": float(r.get("combined_sig", np.nan)),
             "coadd_ew_A": float(r.get("coadd_ew_A", np.nan)),
             "simbad_otype": r.get("simbad_otype"), "simbad_sptype": raw_sp,
             "survey_subclass": sub}
        try:
            z = float(r.get("redshift", 0.0) or 0.0)
        except (TypeError, ValueError):
            z = 0.0
        # Is the feature even unresolved?  Fit its profile in its own coadd and
        # compare with the pipeline's LSF at that pixel, not with a nominal R.
        try:
            own = sparcl_retrieve(client, [str(r["spec_id"])], rel)
            if own:
                o = own[0]
                w = np.asarray(o.get("wavelength", []), float)
                lam = float(r["wavelength"])
                # Three LSF estimates, all recorded: the SPARCL column in its
                # correct unit, the same column read the way run 35751666444
                # read it (as angstroms -- kept so the correction is visible),
                # and sky lines fitted in this very spectrum.  The sky-line
                # value needs no unit convention and is preferred when >= 3
                # isolated lines were fitted.
                ws = o.get("wave_sigma")
                lsf_ws = lsf_fwhm_measured(w, ws, lam, release=rel)
                e["lsf_wave_sigma_fwhm_A"] = round(float(lsf_ws), 3) \
                    if np.isfinite(lsf_ws) else None
                raw = lsf_fwhm_measured(w, ws, lam)
                e["lsf_wave_sigma_read_as_A_fwhm"] = round(float(raw), 3) \
                    if np.isfinite(raw) else None
                e.update(_json_safe(lsf_from_sky(w, o.get("sky"), lam)))
                if int(e.get("lsf_sky_n") or 0) >= 3:
                    lsf, e["lsf_source"] = float(e["lsf_sky_fwhm_A"]), "sky_lines"
                elif np.isfinite(lsf_ws):
                    lsf, e["lsf_source"] = float(lsf_ws), "wave_sigma"
                else:
                    lsf, e["lsf_source"] = lsf_fwhm_A(lam, rel), "nominal"
                e["lsf_fwhm_A"] = round(float(lsf), 3)
                e["lsf_nominal_fwhm_A"] = round(lsf_fwhm_A(lam, rel), 3)
                fit = fit_line_profile(w, np.asarray(o.get("flux", []), float),
                                       np.asarray(o.get("ivar", []), float), lam, lsf,
                                       str(r.get("search_mode", "emission")))
                e.update(_json_safe(fit))
                if fit.get("fit_ok") and lsf > 0:
                    e["fwhm_over_lsf"] = round(float(fit["fit_fwhm_A"]) / lsf, 3)
                    e["fwhm_over_lsf_err"] = round(float(fit["fit_fwhm_err_A"]) / lsf, 3)
                    for k in ("lsf_wave_sigma_fwhm_A", "lsf_sky_fwhm_A"):
                        v = e.get(k)
                        if v is not None and np.isfinite(float(v)) and float(v) > 0:
                            e["fwhm_over_" + k.replace("_fwhm_A", "")] = round(
                                float(fit["fit_fwhm_A"]) / float(v), 3)
                e.update(nearest_sky_atomic(lam))
                # Is it one nebular line of a background galaxy in the fibre?
                e["background_galaxy"] = _json_safe(background_galaxy_scan(
                    w, np.asarray(o.get("flux", []), float),
                    np.asarray(o.get("ivar", []), float), lam, rel,
                    str(r.get("search_mode", "emission"))))
                # SIMBAD has no spectral type for 10 of the 14 lines the first
                # run left standing, and without one the same-type control
                # silently degrades to the all-stars control.  The survey's own
                # classification is right there in the record.
                if not sub:
                    for key in ("subclass", "subtype"):
                        sub = survey_subclass(o.get(key)) or None
                        if sub:
                            e["survey_subclass"] = sub
                            e["subclass_source"] = f"sparcl {key}={o.get(key)}"
                            break
        except Exception as exc:  # noqa: BLE001
            e["fit_error"] = repr(exc)[:300]
        # Every epoch at the position, one by one, not just the best of them.
        kept: list[dict] = []
        try:
            e["epoch_series"] = epoch_series(
                client, float(r["ra"]), float(r["dec"]), str(r["spec_id"]),
                float(r["wavelength"]), str(r.get("search_mode", "emission")), keep=kept)
        except Exception as exc:  # noqa: BLE001
            e["epoch_series"] = {"error": repr(exc)[:300]}
        # The background-galaxy test on every epoch at once, against its null.
        if str(r.get("search_mode", "emission")) != "absorption" and kept:
            try:
                e["nebular_family_epochs"] = _json_safe(nebular_family_calibrated(
                    kept, float(r["wavelength"]), rel))
            except Exception as exc:  # noqa: BLE001
                e["nebular_family_epochs"] = {"error": repr(exc)[:300]}
        del kept
        # Three samples.  Stars of this object's own type and stars of any type:
        # a feature the spectral TYPE makes appears in the first and not the
        # second; one the sky makes appears in both.  And other FIBRES of the
        # same plate: a bad CCD column puts a narrow feature at one wavelength
        # in every fibre of that exposure set, and it is in every exposure and
        # in every repeat observation of the plate, so nothing upstream sees it.
        # Not \d{4} below: eBOSS plates run past 9999 and a five-digit plate
        # would silently lose its same-plate control.
        ident = str(r.get("identifier") or "")
        mode_ = str(r.get("search_mode", "emission"))
        lam_ = float(r["wavelength"])
        try:
            e["any_star"] = control_sample(client, rel, lam_, mode_, z, subclass=None, n=n,
                                           exclude_ids=(str(r["spec_id"]),))
        except Exception as exc:  # noqa: BLE001
            e["any_star"] = {"error": repr(exc)[:300]}
        if sub:
            # Drawn from a DIFFERENT seed as well as a different pool, so it can
            # never again be a byte-for-byte copy of the any-star sample.
            try:
                e["same_type"] = same_type_sample(client, rel, lam_, mode_, z, sub, n=n,
                                                  exclude_ids=(str(r["spec_id"]),), seed=1)
            except Exception as exc:  # noqa: BLE001
                e["same_type"] = {"error": repr(exc)[:300]}
        else:
            e["same_type"] = {"error": "no spectral type known for this object"}
        # Same plate and adjacent fibres, from the SAS files (SPARCL cannot be
        # asked for a plate or a fibre).
        mm = re.match(r"^(\d+)-(\d+)-(\d+)$", ident)
        if mm and rel.upper().startswith("SDSS"):
            try:
                run2d = None
                try:
                    ck = json.loads((out_dir / "ckpt" / f"{r['spec_id']}.json").read_text())
                    run2d = (ck.get("provenance") or {}).get("run2d")
                except (OSError, ValueError):
                    pass
                nbr = sdss_fibre_neighbours(int(mm.group(1)), int(mm.group(2)),
                                            int(mm.group(3)), run2d, lam_, mode_,
                                            float(e.get("lsf_fwhm_A") or
                                                  lsf_fwhm_A(lam_, rel)))
                e["fibres"] = nbr
                e["same_plate"] = {"constraint": f"plate={mm.group(1)} (SAS lite files)",
                                   "obs_frame": nbr.get("same_plate")}
                # Units check on the LSF column: the SAS file's wdisp is in
                # pixels by the data model; SPARCL's wave_sigma for the same
                # fibre should equal it if SPARCL passes it through unconverted.
                wd = nbr.get("target_wdisp_pix")
                if wd:
                    e["lsf_file_wdisp_fwhm_A"] = round(float(nbr["lsf_file_wdisp_fwhm_A"]), 3)
                    rawv = e.get("lsf_wave_sigma_read_as_A_fwhm")
                    if rawv:
                        e["wave_sigma_over_file_wdisp"] = round(
                            float(rawv) / 2.3548 / float(wd), 3)
                    if e.get("fit_ok") and e.get("fit_fwhm_A"):
                        e["fwhm_over_lsf_file_wdisp"] = round(
                            float(e["fit_fwhm_A"]) / float(nbr["lsf_file_wdisp_fwhm_A"]), 3)
            except Exception as exc:  # noqa: BLE001
                e["same_plate"] = {"error": repr(exc)[:300]}
        else:
            e["same_plate"] = {"error": "no plate in the identifier (not an SDSS route)"}
        sa = (e.get("same_type", {}).get("obs_frame") or {})
        an = (e.get("any_star", {}).get("obs_frame") or {})
        sp = (e.get("same_plate", {}).get("obs_frame") or {})
        es = e.get("epoch_series", {})
        bg = e.get("background_galaxy", {}) or {}
        if bg.get("n_companions_ge3"):
            print(f"[persist]   background-galaxy scan: anchor {bg.get('anchor')} at "
                  f"z={bg.get('z')} gives {bg['n_companions_ge3']} companions >= 3 sigma "
                  f"(strongest {bg.get('strongest_companion_sig')}): "
                  + ", ".join(f"{c['line']}@{c['obs_A']}={c['sig']}"
                              for c in bg.get("companions", [])[:4]))
        print(f"[persist] control {e['identifier']} lam={e['wavelength']:.1f} "
              f"fwhm/lsf={e.get('fwhm_over_lsf')}+-{e.get('fwhm_over_lsf_err')} "
              f"({e.get('lsf_source')}); "
              f"epochs {es.get('n_measured')}/{es.get('n_found')} "
              f"sig {es.get('sig_min')}..{es.get('sig_max')} "
              f"EW spread {es.get('ew_spread_frac')}; "
              f"same-type frac>=3: {sa.get('frac_ge3')} (n={sa.get('n_measured')}), "
              f"any-star frac>=3: {an.get('frac_ge3')} (n={an.get('n_measured')}), "
              f"same-plate frac>=3: {sp.get('frac_ge3')} (n={sp.get('n_measured')})")
        entries.append(_json_safe(e))
        # Commit-what-you-have after every line: the job has a wall-clock cap
        # and the traffic per line is not known in advance.
        rep = {**_provenance(), "n": len(entries), "n_lines_selected": int(len(sel)),
               "n_control_requested": int(n), "elapsed_s": round(time.time() - t_start, 1),
               "stopped_early": False, "entries": entries}
        (out_dir / "control.json").write_text(json.dumps(_json_safe(rep), indent=2))
        if time.time() - t_start > max_seconds:
            rep["stopped_early"] = True
            rep["stopped_reason"] = (f"time budget {max_seconds:.0f}s spent after "
                                     f"{len(entries)} of {len(sel)} lines")
            print(f"[persist] {rep['stopped_reason']}")
            (out_dir / "control.json").write_text(json.dumps(_json_safe(rep), indent=2))
            return rep
        time.sleep(0.5)
    rep = {**_provenance(), "n": len(entries), "n_lines_selected": int(len(sel)),
           "n_control_requested": int(n), "elapsed_s": round(time.time() - t_start, 1),
           "stopped_early": False, "entries": entries}
    (out_dir / "control.json").write_text(json.dumps(_json_safe(rep), indent=2))
    return rep


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _ckpt_current(path: Path) -> bool:
    """True only for a checkpoint written by the current estimator version."""
    if not path.exists():
        return False
    try:
        return int(json.loads(path.read_text()).get("ckpt_version", 1)) == CKPT_VERSION
    except (OSError, ValueError, TypeError):
        return False


def _provenance() -> dict:
    """When, by which workflow run and which commit an output file was written."""
    return {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
            "code_sha": os.environ.get("GITHUB_SHA", "")[:12]}


def _read_ckpt(path: Path) -> dict | None:
    """A checkpoint's contents, or None if the file is not a whole JSON object."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None
    return d if isinstance(d, dict) and d.get("spec_id") else None


def merge_checkpoints(dest: Path, incoming: Path, prefer_sha: str = "") -> dict:
    """Merge per-shard checkpoint directories into ``dest``, one file at a time.

    Run 35758868818 measured all 141 spectra and then lost 49 of them in the
    merge: every shard uploaded the WHOLE ckpt directory (the checkpoints it
    fetched from the branch as well as the ones it wrote), and
    ``download-artifact`` with ``merge-multiple`` unpacked the shard archives
    CONCURRENTLY into one directory.  Where two archives held the same file --
    shard 0's fresh v4 measurement and shard 1's stale v2 copy of it -- the two
    writes raced: 14 files came out as the interleaving of both (not JSON at
    all, and the reduce died on the first) and 35 as the stale copy.

    Here each artifact is unpacked into its own directory under ``incoming``
    and the choice is made per file, deterministically: a file that does not
    parse is never chosen; of the rest the highest ``ckpt_version`` wins, then
    the one measured by ``prefer_sha`` (the dispatched commit), then the one
    already in ``dest``.  ``dest``'s own copy takes part, so a corrupt file
    already committed is replaced by a good one when a shard has it.
    """
    dest, incoming = Path(dest), Path(incoming)
    dest.mkdir(parents=True, exist_ok=True)
    cands: dict[str, list[tuple]] = {}
    for p in sorted(dest.glob("*.json")):
        cands.setdefault(p.name, []).append((p, _read_ckpt(p), True))
    for p in sorted(incoming.rglob("*.json")) if incoming.exists() else []:
        cands.setdefault(p.name, []).append((p, _read_ckpt(p), False))
    stats = {"n_files": len(cands), "n_replaced": 0, "n_unreadable_dropped": 0,
             "n_unreadable_kept": 0, "unreadable_kept": []}
    for name, lst in cands.items():
        good = [(p, d, own) for p, d, own in lst if d is not None]
        stats["n_unreadable_dropped"] += sum(1 for _, d, _o in lst if d is None and not _o)
        if not good:
            stats["n_unreadable_kept"] += 1
            stats["unreadable_kept"].append(name)
            continue

        def _key(t):
            _p, d, own = t
            try:
                v = int(d.get("ckpt_version", 1))
            except (TypeError, ValueError):
                v = 1
            sha = str(d.get("code_sha", "") or "")
            return (v, bool(prefer_sha) and sha[:12] == prefer_sha[:12], own)
        best = max(good, key=_key)
        if not best[2]:
            (dest / name).write_text(best[0].read_text())
            stats["n_replaced"] += 1
    return stats


def load_survivors(root: Path):
    import pandas as pd
    p = root / "results" / "spectra_triage" / "priority_targets.csv"
    df = pd.read_csv(p)
    df = df.drop_duplicates(subset=["spec_id", "wavelength"]).reset_index(drop=True)
    return df


_BULK_KEYS = ("wave", "flux", "ivar", "mask", "sky")


def _is_bulk(v) -> bool:
    """A whole spectrum, as opposed to a short hand-picked excerpt of one."""
    if isinstance(v, np.ndarray):
        return True
    if isinstance(v, (list, tuple)):
        return len(v) > 64
    return False


def _json_safe(o):
    if isinstance(o, dict):
        # Whole spectra are dropped (they would be megabytes per record); a short
        # excerpt stored under the same key -- the pixel window a diagnostic dumps
        # around a line -- is exactly the evidence that is wanted and is kept.
        return {k: _json_safe(v) for k, v in o.items()
                if not (k in _BULK_KEYS and _is_bulk(v))}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bytes):
        return o.decode(errors="replace")
    return o


def process_spectrum(rec: dict, cand_rows: list[dict], release: str, workdir: Path,
                     client=None, max_exposures: int = 10) -> dict:
    """Full persistence + second-epoch treatment of one spectrum's candidates."""
    spec_id = str(rec.get("sparcl_id"))
    out = {"spec_id": spec_id, "data_release": release, "route": "", "file_url": "",
           "provenance": {}, "n_exposures_in_file": 0, "lines": [], "error": ""}
    mode = str(cand_rows[0].get("search_mode", "emission"))
    wave = np.asarray(rec.get("wavelength", []), float)
    flux = np.asarray(rec.get("flux", []), float)
    ivar = np.asarray(rec.get("ivar", []), float)
    sky = rec.get("sky")
    exposures_by_line: dict[float, list[dict]] = {}
    file_coadd_by_line: dict[float, dict | None] = {}
    stack_by_line: dict[float, dict | None] = {}
    null_by_line: dict[float, dict | None] = {}

    if release.upper().startswith("SDSS") or release.upper().startswith("BOSS"):
        ids = sdss_ids_from_record(rec)
        out["provenance"] = ids or {}
        if not ids:
            out["error"] = "no plate/mjd/fiber in SPARCL record"
        else:
            out["route"] = "sdss_full_spec"
            parsed = None
            tried_files = []
            for url in sdss_spec_urls(ids["plate"], ids["mjd"], ids["fiberid"], ids.get("run2d")):
                data = fetch_bytes(url, max_bytes=200_000_000)
                if data is None:
                    continue
                try:
                    from astropy.io import fits
                    with fits.open(io.BytesIO(data), memmap=False) as hd:
                        got = parse_sdss_spec(hd)
                except Exception as exc:  # noqa: BLE001
                    out["error"] = f"parse failed {url}: {exc!r}"
                    continue
                n_exp = len({e["expid"] for e in got["exposures"]})
                tried_files.append({"url": url, "n_bytes": len(data), "n_exposures": n_exp})
                # A reduction whose full spec file carries NO per-exposure HDUs (the
                # legacy run2d=26 files are coadd-only) cannot answer the persistence
                # question.  Keep it only as a last resort and prefer any reduction of
                # the same plate that does carry them.
                if parsed is None or n_exp > out["n_exposures_in_file"]:
                    parsed = got
                    out["file_url"] = url
                    out["n_exposures_in_file"] = n_exp
                if n_exp > 0:
                    break
            out["provenance"]["files_tried"] = tried_files
            if parsed is None:
                out["error"] = out["error"] or "no full spec file reachable"
            elif out["n_exposures_in_file"] == 0:
                out["error"] = ("full spec file has no per-exposure HDUs "
                                "(coadd-only reduction); spCFrame route needed")
            if parsed is not None:
                for c in cand_rows:
                    lam0 = float(c["wavelength"])
                    fc, ex = sdss_exposure_measurements(parsed, lam0, mode)
                    file_coadd_by_line[lam0] = fc
                    exposures_by_line[lam0] = ex
                    if parsed.get("exposures"):
                        st = stack_exposures(parsed, lam0)
                        if st is not None:
                            stack_by_line[lam0] = measure_line(
                                st["wave"], st["flux"], st["ivar"], lam0,
                                lsf_fwhm_A(lam0, release), mode)
                        null_by_line[lam0] = offset_null(
                            sdss_measure_at(parsed, mode), lam0, n=N_NULL_OFFSETS)
    elif release.upper().startswith("DESI"):
        ident = desi_identity(rec)
        tid, hpx = ident["targetid"], ident["healpix"]
        out["provenance"] = dict(ident)
        if tid is None or hpx is None:
            out["error"] = "no targetid/healpix derivable from the SPARCL record"
        else:
            out["route"] = "desi_cframe"
            rows, tried = desi_find_exposures(tid, hpx, ident.get("survey"), ident.get("program"),
                                              workdir)
            out["provenance"]["coadd_files"] = tried
            hit = [t for t in tried if t.get("n_rows")]
            if hit:
                out["file_url"] = hit[0]["url"]
                out["coadd_access"] = hit[0].get("access")
                out["provenance"]["survey"] = ",".join(sorted({t["survey"] for t in hit}))
                out["provenance"]["program"] = ",".join(sorted({t["program"] for t in hit}))
            out["n_exposures_in_file"] = len(rows)
            out["provenance"]["exposures"] = [
                {k: r[k] for k in ("night", "expid", "tileid", "petal", "fiber", "survey", "program")}
                for r in rows]
            if not rows:
                out["error"] = "no DR1 coadd file holds the target (EXP_FIBERMAP) in its healpix"
            for c in cand_rows:
                lam0 = float(c["wavelength"])
                collected: list[dict] = []
                ex = desi_exposure_measurements(rows, lam0, mode, workdir, max_exposures,
                                                collect=collected) if rows else []
                if collected:
                    co_arrays = ({"wave": wave, "flux": flux, "ivar": ivar,
                                  "mask": rec.get("mask"), "sky": sky}
                                 if wave.size else None)
                    null_by_line[lam0] = offset_null(
                        desi_measure_at(collected, mode, coadd=co_arrays), lam0,
                        n=N_NULL_OFFSETS)
                if rows and not any(e.get("testable") for e in ex):
                    # cframes unreachable: try the per-exposure healpix spectra file.
                    for t in hit:
                        for surl in desi_spectra_urls(t["survey"], t["program"], int(hpx)):
                            ex2 = desi_spectra_file_measurements(surl, int(tid), lam0, mode, workdir,
                                                                 max_exposures)
                            if ex2:
                                out["route"] = "desi_spectra_file"
                                ex = ex2
                                break
                        if out["route"] == "desi_spectra_file":
                            break
                exposures_by_line[lam0] = ex
                file_coadd_by_line[lam0] = None
    else:
        out["error"] = f"unsupported release {release}"

    for c in cand_rows:
        lam0 = float(c["wavelength"])
        fwhm = lsf_fwhm_A(lam0, release)
        sparcl_coadd = measure_line(wave, flux, ivar, lam0, fwhm, mode, mask=rec.get("mask"),
                                    sky=sky) if wave.size else None
        fc = file_coadd_by_line.get(lam0)
        ref = fc if (fc and fc.get("testable")) else sparcl_coadd
        ex = exposures_by_line.get(lam0, [])
        st = stack_by_line.get(lam0)
        null = null_by_line.get(lam0)
        cls = classify_persistence(ref, ex, stack=st, null=null)
        entry = {"wavelength": lam0, "search_mode": mode,
                 "triage_significance": float(c.get("significance", np.nan)),
                 "sparcl_coadd": _json_safe(sparcl_coadd) if sparcl_coadd else None,
                 "file_coadd": _json_safe(fc) if fc else None,
                 "stack_coadd": _json_safe(st) if st else None,
                 "offset_null": _json_safe(null) if null else None,
                 "exposures": _json_safe(ex), **cls}
        if client is not None and np.isfinite(float(c.get("ra", np.nan))):
            try:
                entry["second_epoch"] = second_epoch(client, float(c["ra"]), float(c["dec"]),
                                                     spec_id, lam0, mode)
            except Exception as exc:  # noqa: BLE001
                entry["second_epoch"] = {"find_error": repr(exc), "second_epoch": "error"}
            time.sleep(0.5)
        out["lines"].append(entry)
    return out


def run_shard(root: Path, shard: int = 0, n_shards: int = 1, top: int = 0,
              max_exposures: int = 10, release_filter: str = "", offline_records=None,
              workdir: Path | None = None) -> dict:
    """Process every survivor spectrum assigned to this shard (checkpointed)."""
    root = Path(root)
    out_dir = root / "results" / "spectra_persist"
    ckpt = out_dir / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    workdir = workdir or Path(tempfile.mkdtemp(prefix="persist_"))
    df = load_survivors(root)
    if top:
        df = df.sort_values("significance", ascending=False).head(top)
    if release_filter:
        df = df[df["data_release"].str.upper().str.startswith(release_filter.upper())]
    spec_ids = sorted(df["spec_id"].astype(str).unique())
    mine = [s for k, s in enumerate(spec_ids) if k % max(n_shards, 1) == shard]
    print(f"[persist] shard {shard}/{n_shards}: {len(mine)} spectra of {len(spec_ids)}")
    todo = [s for s in mine if not _ckpt_current(ckpt / f"{s}.json")]
    print(f"[persist] {len(mine) - len(todo)} already checkpointed at v{CKPT_VERSION}; "
          f"{len(todo)} to do")
    stats = {"n_assigned": len(mine), "n_done_before": len(mine) - len(todo), "n_processed": 0,
             "n_failed": 0}
    if not todo:
        return stats
    by_release: dict[str, list[str]] = {}
    for s in todo:
        rel = str(df.loc[df["spec_id"] == s, "data_release"].iloc[0])
        by_release.setdefault(rel, []).append(s)
    client = None
    if offline_records is None:
        client = _make_client()
    for rel, ids in by_release.items():
        if offline_records is not None:
            recs = [r for r in offline_records if str(r.get("sparcl_id")) in set(ids)]
        else:
            recs = sparcl_retrieve(client, ids, rel)
        if recs:
            print(f"[persist] {rel}: {len(recs)} records; first keys "
                  f"{sorted(k for k in recs[0] if k not in ('wavelength', 'flux', 'ivar', 'mask', 'sky'))}")
        got = {str(r.get("sparcl_id")): r for r in recs}
        for s in ids:
            rec = got.get(s)
            rows = df[df["spec_id"] == s].to_dict("records")
            t0 = time.time()
            if rec is None:
                res = {"spec_id": s, "data_release": rel, "route": "", "lines": [],
                       "error": "SPARCL retrieve returned no record", "provenance": {}}
                for c in rows:
                    res["lines"].append({"wavelength": float(c["wavelength"]),
                                         "search_mode": c.get("search_mode"),
                                         "persistence_class": "untestable",
                                         "basis": "SPARCL record unavailable", "exposures": []})
                stats["n_failed"] += 1
            else:
                try:
                    res = process_spectrum(rec, rows, rel, workdir, client=client,
                                           max_exposures=max_exposures)
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    res = {"spec_id": s, "data_release": rel, "route": "", "lines": [],
                           "error": f"exception: {exc!r}", "provenance": {}}
                    for c in rows:
                        res["lines"].append({"wavelength": float(c["wavelength"]),
                                             "search_mode": c.get("search_mode"),
                                             "persistence_class": "untestable",
                                             "basis": f"exception: {exc!r}", "exposures": []})
                    stats["n_failed"] += 1
            res["elapsed_s"] = round(time.time() - t0, 1)
            res["ckpt_version"] = CKPT_VERSION
            # Which commit measured this. A sharded run whose jobs queue for an
            # hour can otherwise check out two different estimators and merge
            # them into one summary without leaving a trace.
            res["code_sha"] = os.environ.get("GITHUB_SHA", "")[:12]
            (ckpt / f"{s}.json").write_text(json.dumps(_json_safe(res)))
            stats["n_processed"] += 1
            for ln in res.get("lines", []):
                print(f"[persist] {s[:8]} {rel} lam={ln['wavelength']:.1f} "
                      f"-> {ln.get('persistence_class')} ({ln.get('basis', '')[:90]}) "
                      f"[{res.get('route')}, {res['elapsed_s']}s]"
                      + (f" ERR={res['error']}" if res.get("error") else ""))
    return stats


def _recurrence_counts(root: Path, waves, spec_ids, tol_A: float = 3.0) -> dict:
    """Other sightlines with a candidate at the same wavelength.

    The triage's recurrence cut needed three spectra within 3 A, so pairs came
    through: across the 350 triaged candidates there are 114 pairs at EXACTLY
    the same wavelength, and on a survey's common log-lambda grid the same
    wavelength is the same PIXEL.  Unrelated sightlines do not agree to three
    decimals by accident.  Counted against every triaged candidate, not only
    the survivors, because the ones the triage already removed are evidence
    about the wavelength too.
    """
    import pandas as pd
    out = {"n_other_candidates_within_3A": [0] * len(list(waves)),
           "nearest_other_candidate_dA": [float("nan")] * len(list(waves))}
    p = Path(root) / "results" / "spectra_triage" / "triaged_candidates.csv"
    if not p.exists():
        return out
    try:
        allc = pd.read_csv(p)
    except (OSError, ValueError):
        return out
    aw = pd.to_numeric(allc["wavelength"], errors="coerce").to_numpy(float)
    aid = allc["spec_id"].astype(str).to_numpy()
    n_other, d_near = [], []
    for w, sid in zip(waves, spec_ids, strict=True):
        try:
            wv = float(w)
        except (TypeError, ValueError):
            n_other.append(0)
            d_near.append(float("nan"))
            continue
        other = aid != str(sid)
        d = np.abs(aw - wv)
        n_other.append(int(np.sum(other & np.isfinite(d) & (d <= tol_A))))
        rest = d[other & np.isfinite(d)]
        d_near.append(float(np.min(rest)) if rest.size else float("nan"))
    return {"n_other_candidates_within_3A": n_other,
            "nearest_other_candidate_dA": [round(x, 3) if np.isfinite(x) else None
                                           for x in d_near]}


def plate_context(identifiers, waves, tol_A: float = 1.0) -> dict:
    """How much company each candidate has on its own plate.

    An SDSS plate is one exposure set on one pair of CCDs.  A plate that
    contributes many candidates is telling you about the plate, not about the
    sky, and two FIBRES of one plate with a candidate at the same wavelength
    are telling you about a CCD column: the fibres are different objects but
    the same detector columns.

    On the first run's 71 measured lines this separates the set cleanly --
    plate 2333 alone contributes 11, with two exact-wavelength fibre pairs and
    every one of them below 3.3 sigma once the coadd is measured against its
    own local scatter, while the strongest candidate's plate contributes
    exactly one.
    """
    ids = [str(x) if x is not None else "" for x in identifiers]
    w = [float(x) if x is not None and np.isfinite(float(x)) else float("nan")
         for x in waves]
    plate, fibre = [], []
    for s in ids:
        m = re.match(r"^(\d+)-(\d+)-(\d+)$", s)
        plate.append(m.group(1) if m else "")
        fibre.append(m.group(3) if m else "")
    n_on_plate, n_same_pixel = [], []
    for i, p in enumerate(plate):
        if not p:
            n_on_plate.append(0)
            n_same_pixel.append(0)
            continue
        same = [j for j in range(len(plate)) if plate[j] == p and j != i]
        n_on_plate.append(len(same))
        n_same_pixel.append(sum(
            1 for j in same
            if fibre[j] and fibre[j] != fibre[i]
            and np.isfinite(w[i]) and np.isfinite(w[j]) and abs(w[i] - w[j]) <= tol_A))
    return {"plate_n_other_candidates": n_on_plate,
            "plate_other_fibre_same_wavelength": n_same_pixel}


def _pair_offsets(pix, sid, ra, dec, max_offset: int, tol: float) -> tuple[np.ndarray, int]:
    """Histogram of pixel separations between candidates on DIFFERENT sightlines."""
    counts = np.zeros(max_offset + 1, np.int64)
    n_same_object = 0
    for i in range(pix.size - 1):
        diff = np.abs(pix[i + 1:] - pix[i])
        near = np.zeros(diff.shape, bool)
        if np.isfinite(ra[i]) and np.isfinite(dec[i]):
            cosd = max(np.cos(np.radians(dec[i])), 1e-3)
            near = (np.abs(dec[i + 1:] - dec[i]) <= tol) & \
                   (np.abs(ra[i + 1:] - ra[i]) * cosd <= tol)
        keep = (sid[i + 1:] != sid[i]) & ~near & (diff <= max_offset)
        n_same_object += int(np.sum(near & (diff <= max_offset)))
        diff = diff[keep]
        if diff.size:
            counts += np.bincount(diff, minlength=max_offset + 1)
    return counts, n_same_object


def pixel_coincidence(root: Path, release: str = "SDSS-DR17", max_offset: int = 10,
                      baseline_from: int = 3, n_perm: int = 500, seed: int = 17) -> dict:
    """Do unrelated sightlines put candidates on the SAME PIXEL more than chance?

    A survey coadd lives on one common wavelength grid, so "the same
    wavelength" means "the same pixel index" for every spectrum in the release.
    A detector or reduction feature that produces narrow spikes does so at a
    fixed pixel; a real source does not care which pixel it lands on.

    The test calibrates itself: the number of pairs of candidates from
    DIFFERENT sightlines separated by 0 pixels is compared with the number at
    3-10 pixels, which carries the same clustering of the search's sensitivity
    with wavelength but none of the same-pixel effect.  An excess at 0 (and at
    1, for a feature that straddles two pixels) is instrumental.
    """
    import pandas as pd
    out = {"release": release, "n_candidates": 0, "pairs_by_offset": {}, "baseline": None}
    p = Path(root) / "results" / "spectra_triage" / "triaged_candidates.csv"
    if not p.exists():
        out["error"] = "no triaged_candidates.csv"
        return out
    try:
        t = pd.read_csv(p)
    except (OSError, ValueError) as exc:
        out["error"] = repr(exc)[:200]
        return out
    t = t[t["data_release"].astype(str) == release].drop_duplicates(["spec_id", "wavelength"])
    w = pd.to_numeric(t["wavelength"], errors="coerce").to_numpy(float)
    sid = t["spec_id"].astype(str).to_numpy()
    ra = pd.to_numeric(t.get("ra"), errors="coerce").to_numpy(float) \
        if "ra" in t.columns else np.full(w.shape, np.nan)
    dec = pd.to_numeric(t.get("dec"), errors="coerce").to_numpy(float) \
        if "dec" in t.columns else np.full(w.shape, np.nan)
    ok = np.isfinite(w) & (w > 0)
    w, sid, ra, dec = w[ok], sid[ok], ra[ok], dec[ok]
    if w.size < 20:
        out["error"] = f"only {w.size} candidates for {release}"
        return out
    if release.upper().startswith(("SDSS", "BOSS")):
        pix = np.round(np.log10(w) * 1e4).astype(np.int64)     # 1e-4 dex grid
        out["grid"] = "log10 step 1e-4"
    else:
        pix = np.round(w / 0.8).astype(np.int64)               # DESI 0.8 A grid
        out["grid"] = "linear step 0.8 A"
    out["n_candidates"] = int(w.size)
    # Two spectra of the SAME OBJECT land on the same pixel for an honest
    # reason, so only genuinely different sightlines count.  2 arcsec, the same
    # tolerance the second-epoch search uses.
    tol = 2.0 / 3600.0
    counts, n_same_object = _pair_offsets(pix, sid, ra, dec, max_offset, tol)
    out["n_pairs_same_object_excluded"] = n_same_object
    out["pairs_by_offset"] = {int(k): int(v) for k, v in enumerate(counts)}
    base = float(np.mean(counts[baseline_from:max_offset + 1]))
    out["baseline"] = round(base, 2)
    out["baseline_offsets"] = f"{baseline_from}-{max_offset}"
    if base > 0:
        for k in (0, 1):
            out[f"excess_{k}px"] = int(counts[k] - round(base))
            out[f"z_{k}px"] = round(float((counts[k] - base) / np.sqrt(base)), 2)
        out["excess_0_or_1px"] = int(counts[0] + counts[1] - round(2 * base))
    # The Poisson z above assumes the pair counts are independent draws, and
    # they are not: one candidate is in many pairs.  The null distribution is
    # got instead by sliding each SIGHTLINE's own set of pixels bodily by a
    # random offset far larger than the window counted -- that destroys
    # cross-sightline alignment while keeping how many candidates each sightline
    # has and roughly where they sit.  (A cluster bootstrap is wrong here:
    # resampling sightlines with replacement makes duplicate copies of one
    # sightline, which land on the same pixel by construction.)
    if n_perm and base > 0:
        rng = np.random.default_rng(seed)
        where = {s: (sid == s) for s in set(sid.tolist())}
        obs = float(counts[0]) - base
        null = np.empty(int(n_perm), float)
        for b in range(int(n_perm)):
            shifted = pix.copy()
            for m in where.values():
                shifted[m] += int(rng.integers(-400, 401))
            c, _ = _pair_offsets(shifted, sid, ra, dec, max_offset, tol)
            null[b] = float(c[0]) - float(np.mean(c[baseline_from:max_offset + 1]))
        sd = float(np.std(null))
        out.update({
            "n_permutations": int(n_perm), "n_sightlines": len(where),
            "perm_null_mean": round(float(np.mean(null)), 3),
            "perm_null_sd": round(sd, 3),
            "perm_z_0px": round(float((obs - np.mean(null)) / sd), 2) if sd > 0 else None,
            "perm_p_0px": round(float(np.mean(null >= obs)), 5),
        })
    return out


def reduce_results(root: Path, do_simbad: bool = True, do_nist: bool = True) -> dict:
    """Merge checkpoints into the flat table + summary; add SIMBAD and line IDs."""
    import pandas as pd

    from .linelist import (
        atmospheric_context,
        band_gap_context,
        build_nist_cache,
        identify_rest_frame,
        nist_context,
    )
    root = Path(root)
    out_dir = root / "results" / "spectra_persist"
    ckpt = out_dir / "ckpt"
    df = load_survivors(root)
    rows = []
    per_exp = {}
    n_stale = 0
    code_shas: dict[str, int] = {}
    unreadable: list[str] = []
    for p in sorted(ckpt.glob("*.json")):
        r = _read_ckpt(p)
        if r is None:
            # A truncated or interleaved checkpoint is not a measurement.  It
            # must not take the whole reduce down with it (run 35758868818
            # lost a complete 141-spectrum run that way); it is counted and
            # named in the summary instead, and the next run re-measures it.
            unreadable.append(p.name)
            continue
        sha = str(r.get("code_sha", "") or "unrecorded")
        code_shas[sha] = code_shas.get(sha, 0) + 1
        # A checkpoint from a superseded estimator is not evidence; it is a
        # stale number that would otherwise be merged in as though it were.
        if int(r.get("ckpt_version", 1)) != CKPT_VERSION:
            n_stale += 1
            continue
        prov = r.get("provenance", {}) or {}
        for ln in r.get("lines", []):
            se = ln.get("second_epoch", {}) or {}
            rows.append({
                "spec_id": r["spec_id"], "wavelength": ln["wavelength"],
                "search_mode": ln.get("search_mode"), "data_release": r.get("data_release"),
                "route": r.get("route"), "identifier": _identifier(r.get("data_release"), prov),
                "n_exposures_in_file": r.get("n_exposures_in_file"),
                "n_tested": ln.get("n_tested"), "n_present": ln.get("n_present"),
                "frac_present": ln.get("frac_present"),
                "combined_sig": ln.get("combined_sig"),
                "combined_sig_raw": ln.get("combined_sig_raw"),
                "null_exposure_bias_sig": ln.get("null_exposure_bias_sig"),
                "null_exposure_scatter": ln.get("null_exposure_scatter"),
                "null_calibrated": ln.get("null_calibrated"),
                "stack_sig": ln.get("stack_sig"), "stack_ew_A": ln.get("stack_ew"),
                "stack_over_coadd_F": ln.get("stack_over_coadd_F"),
                "coadd_sig_cal": ln.get("coadd_sig_cal"),
                "mean_F": ln.get("mean_F"),
                "chi2_p": ln.get("chi2_p"),
                "dominant_frac": ln.get("dominant_frac"),
                "max_exposure_sig": ln.get("max_exposure_sig"),
                "ratio_to_coadd": ln.get("ratio_to_coadd"),
                "coadd_recovered": ln.get("coadd_recovered"),
                "on_sky_line": ln.get("on_sky_line"), "sky_corr": ln.get("sky_corr"),
                "n_cosmic_flagged": ln.get("n_cosmic_flagged"),
                "persistence_class": ln.get("persistence_class"), "basis": ln.get("basis"),
                "coadd_ew_A": (ln.get("file_coadd") or ln.get("sparcl_coadd") or {}).get("ew"),
                "coadd_sig": (ln.get("file_coadd") or ln.get("sparcl_coadd") or {}).get("sig"),
                "mean_exposure_ew_A": _mean_ew(ln.get("exposures", [])),
                "n_other_epochs": se.get("n_other"), "self_found": se.get("self_found"),
                "other_releases": se.get("other_releases"),
                "other_best_sig": se.get("other_best_sig"),
                "second_epoch": se.get("second_epoch", "none"),
                "error": r.get("error", ""),
            })
            per_exp[f"{r['spec_id']}@{ln['wavelength']:.1f}"] = [
                {k: e.get(k) for k in ("expid", "night", "mjd", "camera", "arms", "F", "err",
                                       "sig", "ew", "peak_sig", "sky_peak_sig", "sky_level",
                                       "n_cosmic", "testable", "reason")}
                for e in ln.get("exposures", [])]
    # A reduce that arrives after an estimator change holds only superseded
    # checkpoints.  Writing its (empty) summary over a real one would replace a
    # measurement with a no-data verdict and commit that back, which is exactly
    # what a cancelled run's late reduce would have done here on 2026-09-22.
    n_current = len({r["spec_id"] for r in rows})
    if n_stale and n_current == 0 and (out_dir / "summary.json").exists():
        msg = (f"every checkpoint on hand ({n_stale}) is from a superseded estimator "
               f"(need ckpt_version {CKPT_VERSION}); refusing to overwrite the existing "
               f"summary with a no-data verdict")
        print(f"[persist] {msg}")
        prev = json.loads((out_dir / "summary.json").read_text())
        prev["reduce_skipped"] = msg
        return prev

    tab = pd.DataFrame(rows)
    keep = ["spec_id", "wavelength", "significance", "ra", "dec", "redshift", "simbad_id",
            "simbad_otype", "simbad_sptype", "n_lines_in_spectrum"]
    kept_cols = [c for c in keep if c in df.columns]
    tab = df[kept_cols].merge(
        tab, on=["spec_id", "wavelength"], how="left") if len(tab) else df[kept_cols].copy()
    for col in ("persistence_class", "route", "n_other_epochs", "second_epoch", "combined_sig",
                "mean_F", "other_best_err_rel", "search_mode", "identifier", "data_release",
                "coadd_ew_A", "n_tested", "n_present", "known_line_match", "stack_sig",
                "combined_sig_raw", "null_exposure_bias_sig", "coadd_sig_cal", "redshift"):
        if col not in tab.columns:
            tab[col] = np.nan
    # A column created as all-NaN is float64; filling it with a STRING is a
    # dtype change that pandas 2 did silently and pandas 3 does not.  Make the
    # column object first, so the same code runs on both majors (the runner
    # installs pandas 3; this sandbox has 2).
    tab["persistence_class"] = tab["persistence_class"].astype(object).fillna("not_run")
    if "search_mode" in df.columns:
        tab["search_mode"] = tab["search_mode"].astype(object).fillna(tab["spec_id"].map(
            df.drop_duplicates("spec_id").set_index("spec_id")["search_mode"]))

    # Rest-frame identification at the star's own catalogue redshift.
    z_col = pd.to_numeric(tab["redshift"], errors="coerce").fillna(0.0)
    mode_col = tab["search_mode"].astype(object).fillna("emission")
    ids = [identify_rest_frame(w, z, mode=str(m)) for w, z, m in
           zip(tab["wavelength"], z_col, mode_col, strict=True)]
    for k in ids[0].keys() if ids else []:
        tab[k] = [d[k] for d in ids]

    # Where the observed wavelength sits relative to the atmosphere: a telluric
    # band edge and a gap in the OH list inside the forest are the two places a
    # hand-kept sky list is least trustworthy, and five of the six lines left
    # standing after the first run are in that part of the spectrum.
    atm = [atmospheric_context(w) for w in tab["wavelength"]]
    for k in atm[0].keys() if atm else []:
        tab[k] = [d[k] for d in atm]

    # And where it sits relative to the star's own molecular band heads: the
    # flux BETWEEN two heads in a cool star is a relative maximum, which a
    # matched filter on a local linear continuum reads as an emission line --
    # in every exposure and in every epoch, so nothing upstream can see it.
    bg = [band_gap_context(w, z) for w, z in zip(tab["wavelength"], z_col, strict=True)]
    for k in bg[0].keys() if bg else []:
        tab[k] = [d[k] for d in bg]

    # How many OTHER sightlines put a candidate at this same wavelength.  The
    # triage's recurrence cut needed three spectra within 3 A, so PAIRS came
    # through -- and across the 350 triaged candidates there are 114 pairs at
    # EXACTLY the same wavelength, which on a common log-lambda grid means the
    # same pixel.  Unrelated sightlines do not agree to three decimals by
    # accident; that is a fixed feature of the detector or the reduction.
    rec = _recurrence_counts(root, tab["wavelength"], tab["spec_id"])
    for k, v in rec.items():
        tab[k] = v
    # How much company the candidate has on its own plate, and whether another
    # FIBRE of that plate has one at the same wavelength (a CCD column).
    for k, v in plate_context(tab["identifier"], tab["wavelength"]).items():
        tab[k] = v

    # SIMBAD for every unique position (refresh; the triage table has gaps).
    if do_simbad:
        try:
            from ..acquire.science import fetch_simbad_context
            pos = tab.drop_duplicates("spec_id")[["spec_id", "ra", "dec"]].rename(
                columns={"spec_id": "source_id"})
            ctx = fetch_simbad_context(pos, radius_arcsec=3.0)
            if ctx is not None and len(ctx):
                ctx = ctx.rename(columns={"source_id": "spec_id", "simbad_id": "simbad_id_3as",
                                          "simbad_otype": "simbad_otype_3as",
                                          "simbad_sptype": "simbad_sptype_3as"})
                tab = tab.merge(ctx[[c for c in ctx.columns if c == "spec_id" or c.endswith("_3as")]],
                                on="spec_id", how="left")
        except Exception as exc:  # noqa: BLE001
            print(f"[persist] SIMBAD skipped: {exc!r}")
    if do_nist:
        try:
            cache = build_nist_cache(out_dir / "nist_lines.json")
            tab["nist_context"] = [nist_context(cache, w, z) for w, z in
                                   zip(tab["wavelength"], z_col, strict=True)]
        except Exception as exc:  # noqa: BLE001
            print(f"[persist] NIST context skipped: {exc!r}")

    tab["verdict"] = [final_verdict(r) for _, r in tab.iterrows()]
    # A column merged from checkpoints can come back object-typed (None mixed
    # with floats); sorting on that is a comparison pandas 3 refuses.
    tab["combined_sig"] = pd.to_numeric(tab["combined_sig"], errors="coerce")
    tab = tab.sort_values(["verdict", "combined_sig"], ascending=[True, False])
    tab.to_csv(out_dir / "persistence.csv", index=False)
    (out_dir / "exposures.json").write_text(json.dumps(_json_safe(per_exp)))

    counts = tab["persistence_class"].value_counts().to_dict()
    vcounts = tab["verdict"].value_counts().to_dict()
    alive = tab[tab["verdict"] == "ALIVE_persistent_unidentified"]
    summary = {
        # Which run wrote this file.  A summary with no provenance cannot be
        # matched to the artefacts it describes, and on 2026-09-23 this one
        # had silently outlived two later runs.
        **_provenance(),
        "n_survivors_in": int(len(df)), "n_spectra_in": int(df["spec_id"].nunique()),
        "n_checkpointed_spectra": int(len(list(ckpt.glob("*.json")))),
        "ckpt_version": CKPT_VERSION,
        "n_checkpoints_stale_ignored": int(n_stale),
        "n_checkpoints_unreadable": len(unreadable),
        "checkpoints_unreadable": unreadable,
        "n_spectra_measured_current": int(n_current),
        "checkpoint_code_shas": dict(sorted(code_shas.items(), key=lambda kv: -kv[1])),
        "persistence_class_counts": {k: int(v) for k, v in counts.items()},
        "verdict_counts": {k: int(v) for k, v in vcounts.items()},
        "route_counts": {k: int(v) for k, v in
                         tab["route"].astype(object).fillna("").value_counts().items()},
        "n_with_other_fibre_same_wavelength": int(
            (pd.to_numeric(tab["plate_other_fibre_same_wavelength"],
                           errors="coerce").fillna(0) > 0).sum()),
        "most_crowded_plates": {
            str(k): int(v) for k, v in
            tab["identifier"].astype(str).str.extract(r"^(\d+)-")[0]
            .dropna().value_counts().head(5).items()},
        "pixel_coincidence": {rel: pixel_coincidence(root, rel)
                              for rel in sorted(set(df["data_release"].astype(str)))},
        "n_with_another_candidate_within_3A": int(
            (pd.to_numeric(tab["n_other_candidates_within_3A"],
                           errors="coerce").fillna(0) > 0).sum()),
        "n_known_line_rest_frame": int(pd.to_numeric(
            tab["known_line_match"], errors="coerce").fillna(0).sum()),
        "n_second_epoch_available": int((pd.to_numeric(
            tab["n_other_epochs"], errors="coerce").fillna(0) > 0).sum()),
        "n_second_epoch_confirmed": int((tab["second_epoch"] == "confirmed").sum()),
        "n_alive": int(len(alive)),
        "alive": [
            {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v)
             for k, v in r.items()}
            for r in alive[[c for c in
                            ("spec_id", "identifier", "data_release", "ra", "dec", "wavelength",
                             "search_mode", "coadd_ew_A", "coadd_sig_cal", "combined_sig",
                             "combined_sig_raw", "null_exposure_bias_sig", "stack_sig",
                             "n_tested", "n_present", "simbad_otype", "simbad_sptype",
                             "n_lines_in_spectrum", "known_line_label", "known_line_dv_kms",
                             "telluric_band", "oh_gap_A", "oh_density_per_100A",
                             "between_band_heads", "band_blue_label", "band_blue_dA",
                             "band_red_label", "band_red_dA",
                             "n_other_candidates_within_3A", "nearest_other_candidate_dA",
                             "plate_n_other_candidates", "plate_other_fibre_same_wavelength",
                             "second_epoch")
                            if c in alive.columns]].to_dict("records")],
        "verdict": ("PERSISTENT_UNIDENTIFIED_LINES_REMAIN" if len(alive)
                    else ("NO_DATA_REACHED" if not counts or set(counts) <= {"not_run", "untestable"}
                          else "ALL_SURVIVORS_RESOLVED")),
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2))
    print("[persist] summary:", json.dumps(_json_safe(
        {k: summary[k] for k in ("persistence_class_counts", "verdict_counts", "n_alive",
                                 "verdict")})))
    return summary


def _identifier(release, prov: dict) -> str:
    rel = (release or "").upper()
    if rel.startswith(("SDSS", "BOSS")):
        if prov.get("plate"):
            return f"{int(prov['plate']):04d}-{int(prov['mjd'])}-{int(prov['fiberid']):04d}"
        return ""
    if rel.startswith("DESI"):
        if prov.get("targetid") is not None:
            return f"TARGETID {prov['targetid']} ({prov.get('survey')}/{prov.get('program')}/hpx{prov.get('healpix')})"
        return ""
    return ""


def _mean_ew(exps: list[dict]):
    v = [e.get("ew") for e in exps if e.get("testable") and e.get("ew") is not None]
    return float(np.mean(v)) if v else None


def final_verdict(r) -> str:
    """One string per survivor: what kills it, or that it is alive."""
    cls = str(r.get("persistence_class", ""))
    if bool(r.get("known_line_match")):
        return "KILLED_known_line_rest_frame"
    # Two DIFFERENT fibres of one plate with a candidate at the same wavelength
    # are two different objects sharing detector columns.  Both cannot be
    # sources, and a sky or ISM feature at that wavelength would have been taken
    # by the known-line cut above.  Plate 2333 supplies two such pairs.
    try:
        if float(r.get("plate_other_fibre_same_wavelength", 0) or 0) > 0:
            return "KILLED_shared_ccd_column"
    except (TypeError, ValueError):
        pass
    if cls == "absent_in_exposures":
        # Two very different things wear this label.  If the coadd itself does
        # not reach 5 sigma once measured against its own local scatter, the
        # line was never there and "the coadd feature is not in its inputs" is
        # the wrong sentence: there was no coadd feature.  Only a line the
        # CALIBRATED coadd does show, and the exposures and their stack do not,
        # is the coaddition artefact this channel exists to catch.  On the
        # first run's 70 measured lines the calibrated coadd significance is a
        # median 0.34x the triage's and 51% fall below 3 sigma, so the
        # distinction covers most of the class.
        cs = float("nan")
        for key in ("coadd_sig_cal", "coadd_sig"):
            try:
                v = float(r.get(key, float("nan")))
            except (TypeError, ValueError):
                v = float("nan")
            if np.isfinite(v):
                cs = v
                break
        if np.isfinite(cs) and cs < 5.0:
            return "KILLED_not_significant_in_coadd"
        return "KILLED_absent_in_exposures"
    if cls == "transient":
        return "KILLED_transient"
    if cls == "sky_residual":
        return "KILLED_sky_residual"
    if cls == "persistent":
        if str(r.get("second_epoch", "")) == "not_seen" and _other_epoch_sensitive(r):
            return "KILLED_second_epoch_absent"
        return "ALIVE_persistent_unidentified"
    if cls == "persistent_2exp":
        return "OPEN_persistent_2exp"
    if cls in ("partial", "inconsistent_strength", "ambiguous", "single_exposure_only",
               "stack_only"):
        return "OPEN_" + cls
    return "UNTESTED_" + (cls or "unknown")


def _other_epoch_sensitive(r) -> bool:
    """The other epoch would have seen the coadd-strength line at >= 5 sigma."""
    try:
        F = float(r.get("mean_F", np.nan)) if "mean_F" in r else np.nan
        e = float(r.get("other_best_err_rel", np.nan))
        if not np.isfinite(F) or not np.isfinite(e) or e <= 0:
            return False
        return F / e >= 5.0
    except (TypeError, ValueError):
        return False


def probe(root: Path, n_each: int = 2) -> dict:
    """Cheap runner-side reconnaissance: SPARCL fields and archive URL patterns."""
    root = Path(root)
    out_dir = root / "results" / "spectra_persist"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_survivors(root)
    client = _make_client()
    rep: dict = {"releases": {}, "urls": [], "fsspec": False}
    try:
        import aiohttp  # noqa: F401
        import fsspec  # noqa: F401
        rep["fsspec"] = True
    except Exception as exc:  # noqa: BLE001
        rep["fsspec_error"] = repr(exc)
    # Is each archive host reachable at all?  A connection failure and a missing
    # file are different verdicts; record which one we are looking at before any
    # DESI result is interpreted.
    rep["host_reachability"] = [
        http_head(u, tries=2) for u in
        [f"{SDSS_SAS}/", *[f"{b}/healpix/" for b in _desi_bases()]]
    ]
    for h in rep["host_reachability"]:
        print(f"[persist] host {h['url']} -> status {h.get('status')} "
              f"{h.get('error', '')[:200]}")
    for rel in sorted(df["data_release"].unique()):
        fields = sparcl_fields(client, rel)
        ids = df.loc[df["data_release"] == rel, "spec_id"].astype(str).unique()[:n_each].tolist()
        recs = sparcl_retrieve(client, ids, rel)
        info = {"fields": fields, "n_records": len(recs), "samples": []}
        for rec in recs:
            meta = {k: (v if not hasattr(v, "__len__") or isinstance(v, str) else f"<{len(v)}>")
                    for k, v in rec.items()}
            sample = {"meta": meta}
            if rel.upper().startswith(("SDSS", "BOSS")):
                ids_ = sdss_ids_from_record(rec)
                sample["sdss_ids"] = ids_
                if ids_:
                    for url in sdss_spec_urls(ids_["plate"], ids_["mjd"], ids_["fiberid"], ids_.get("run2d")):
                        h = http_head(url)
                        rep["urls"].append(h)
                        sample.setdefault("heads", []).append(h)
                        if h.get("status") == 200:
                            break
            elif rel.upper().startswith("DESI"):
                ident = desi_identity(rec)
                sample["desi_ids"] = ident
                tid, hp = ident["targetid"], ident["healpix"]
                if tid is not None and hp is not None:
                    work = Path(tempfile.mkdtemp(prefix="probe_"))
                    rows, tried = desi_find_exposures(tid, hp, ident.get("survey"),
                                                      ident.get("program"), work)
                    sample["coadd_files"] = tried
                    rep["urls"].extend({k: t.get(k) for k in ("url", "status")} for t in tried)
                    sample["exp_rows"] = rows[:5]
                    sample["n_exp_rows"] = len(rows)
                    hit = [t for t in tried if t.get("n_rows")]
                    if hit:
                        for u in desi_spectra_urls(hit[0]["survey"], hit[0]["program"], int(hp)):
                            hh = http_head(u)
                            rep["urls"].append(hh)
                            sample.setdefault("spectra_heads", []).append(hh)
                            if hh.get("status") == 200:
                                break
                    if rows:
                        r0 = rows[0]
                        for kind in ("cframe", "sky"):
                            for u in desi_frame_urls(kind, r0["night"], r0["expid"], "r", r0["petal"]):
                                hh = http_head(u)
                                rep["urls"].append(hh)
                                sample.setdefault("frame_heads", []).append(hh)
                                if hh.get("status") == 200:
                                    break
                        # One real per-exposure measurement, to prove the row read.
                        lam0 = float(df.loc[df["spec_id"] == str(rec.get("sparcl_id")),
                                            "wavelength"].iloc[0])
                        t0 = time.time()
                        ex = desi_exposure_measurements(rows[:2], lam0, "absorption", work, 2)
                        sample["exposure_test"] = _json_safe(ex)
                        sample["exposure_test_s"] = round(time.time() - t0, 1)
            info["samples"].append(sample)
        rep["releases"][rel] = info
    (out_dir / "probe.json").write_text(json.dumps(_json_safe(rep), indent=2, default=str))
    print(json.dumps(_json_safe(rep), indent=2, default=str)[:20000])
    return rep


def diagnose(root: Path, n: int = 8, release: str = "SDSS") -> dict:
    """Side-by-side of the SPARCL coadd, the SAS file coadd and each exposure.

    The first sharded run returned ``absent_in_exposures`` for every SDSS
    survivor with a *consistently negative* combined significance (-3 to -9
    sigma), not the ~0 sigma that genuine absence produces.  That is the
    signature of a systematic, so this stage prints the raw numbers the
    classifier is built on -- the same line measured in the SPARCL coadd, in
    the file's own COADD HDU and in each exposure HDU, plus the pixel values
    around the line and the correlation between the two coadds -- so the
    disagreement can be attributed instead of guessed at.
    """
    root = Path(root)
    out_dir = root / "results" / "spectra_persist"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_survivors(root)
    if release:
        df = df[df["data_release"].str.upper().str.startswith(release.upper())]
    df = df.sort_values("significance", ascending=False).head(n)
    client = _make_client()
    report = []
    for _, c in df.iterrows():
        spec_id, lam0 = str(c["spec_id"]), float(c["wavelength"])
        rel, mode = str(c["data_release"]), str(c.get("search_mode", "emission"))
        entry = {"spec_id": spec_id, "wavelength": lam0, "search_mode": mode,
                 "triage_significance": float(c.get("significance", np.nan)), "release": rel}
        recs = sparcl_retrieve(client, [spec_id], rel)
        if not recs:
            entry["error"] = "no SPARCL record"
            report.append(entry)
            continue
        rec = recs[0]
        entry["sparcl_keys"] = sorted(rec.keys())
        sw = np.asarray(rec.get("wavelength", []), float)
        sf = np.asarray(rec.get("flux", []), float)
        si = np.asarray(rec.get("ivar", []), float)
        entry["sparcl_n_pix"] = int(sw.size)
        fwhm = lsf_fwhm_A(lam0, rel)
        entry["fwhm_A"] = round(fwhm, 3)
        if sw.size:
            entry["sparcl_coadd"] = _json_safe(measure_line(sw, sf, si, lam0, fwhm, mode))
            k = int(np.argmin(np.abs(sw - lam0)))
            lo, hi = max(k - 6, 0), min(k + 7, sw.size)
            entry["sparcl_window"] = {"wave": [round(float(x), 3) for x in sw[lo:hi]],
                                      "flux": [round(float(x), 4) for x in sf[lo:hi]]}
        ids = sdss_ids_from_record(rec)
        entry["sdss_ids"] = ids
        if not ids:
            report.append(entry)
            continue
        files = []
        for url in sdss_spec_urls(ids["plate"], ids["mjd"], ids["fiberid"], ids.get("run2d")):
            data = fetch_bytes(url, max_bytes=200_000_000)
            if data is None:
                continue
            finfo = {"url": url, "n_bytes": len(data)}
            try:
                from astropy.io import fits
                with fits.open(io.BytesIO(data), memmap=False) as hd:
                    finfo["extnames"] = [str(h.header.get("EXTNAME", f"HDU{i}"))
                                         for i, h in enumerate(hd)]
                    parsed = parse_sdss_spec(hd)
            except Exception as exc:  # noqa: BLE001
                finfo["error"] = repr(exc)
                files.append(finfo)
                continue
            co = parsed.get("coadd")
            finfo["n_exposure_hdus"] = len(parsed["exposures"])
            finfo["n_exposures"] = len({e["expid"] for e in parsed["exposures"]})
            if co is not None:
                finfo["file_coadd"] = _json_safe(
                    measure_line(co["wave"], co["flux"], co["ivar"], lam0, fwhm, mode,
                                 mask=co.get("mask"), sky=co.get("sky"),
                                 bad_bits=SDSS_BAD_BITS, cosmic_bits=SDSS_REJECT_BITS))
                k = int(np.argmin(np.abs(co["wave"] - lam0)))
                lo, hi = max(k - 6, 0), min(k + 7, co["wave"].size)
                finfo["file_window"] = {"wave": [round(float(x), 3) for x in co["wave"][lo:hi]],
                                        "flux": [round(float(x), 4) for x in co["flux"][lo:hi]]}
                # Is the SAS file the same spectrum SPARCL served?
                if sw.size and co["wave"].size:
                    g = (sw > max(sw.min(), co["wave"].min())) & (sw < min(sw.max(), co["wave"].max()))
                    if g.sum() > 100:
                        fi = np.interp(sw[g], co["wave"], co["flux"])
                        a, b = sf[g], fi
                        ok = np.isfinite(a) & np.isfinite(b)
                        if ok.sum() > 100 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                            finfo["corr_with_sparcl"] = round(
                                float(np.corrcoef(a[ok], b[ok])[0, 1]), 4)
                            finfo["median_ratio_to_sparcl"] = round(
                                float(np.median(b[ok] / np.where(a[ok] == 0, np.nan, a[ok]))), 4)
            _, ex = sdss_exposure_measurements(parsed, lam0, mode)
            finfo["exposures"] = [
                {k2: e.get(k2) for k2 in ("expid", "arms", "testable", "reason", "sig", "ew",
                                          "cont", "F", "err", "n_line_pix", "n_cosmic",
                                          "err_scale", "mjd", "peak_sig", "sky_peak_sig",
                                          "sky_level", "cont_slope_per_A", "cont_one_sided")}
                for e in ex]
            # The coadd IS supposed to be the stack of the exposure HDUs.  Measure
            # the line in that stack with the very same estimator: it decides
            # whether a coadd/exposure disagreement lives in our measurement or in
            # the archive's own data.
            st = stack_exposures(parsed, lam0)
            if st is not None:
                finfo["stack_n_used"] = st["n_used"]
                finfo["stack_coadd"] = _json_safe(
                    measure_line(st["wave"], st["flux"], st["ivar"], lam0, fwhm, mode))
                k = int(np.argmin(np.abs(st["wave"] - lam0)))
                lo, hi = max(k - 10, 0), min(k + 11, st["wave"].size)
                finfo["stack_window"] = {
                    "wave": [round(float(x), 3) for x in st["wave"][lo:hi]],
                    "flux": [round(float(x), 4) for x in st["flux"][lo:hi]]}
            # Is the deficit at the candidate wavelength, or everywhere?
            if parsed.get("exposures"):
                finfo["offset_null"] = _json_safe(
                    offset_null(sdss_measure_at(parsed, mode), lam0))
            # A wavelength zero-point difference between the coadd and the native
            # exposure frames moves a real line off the window centre; measure it
            # rather than assume it away.
            if co is not None:
                for e_rec in parsed["exposures"][:6]:
                    ew = np.asarray(e_rec["wave"], float)
                    k2 = int(np.argmin(np.abs(ew - lam0)))
                    lo, hi = max(k2 - 10, 0), min(k2 + 11, ew.size)
                    finfo.setdefault("exposure_windows", []).append({
                        "extname": e_rec.get("extname"),
                        "lag_A": round(float(wave_lag(co, e_rec, lam0)), 3),
                        "d_pix_A": round(float(np.median(np.abs(np.diff(ew[lo:hi])))), 4)
                        if hi - lo > 2 else None,
                        "wave": [round(float(x), 3) for x in ew[lo:hi]],
                        "flux": [round(float(x), 4) for x in
                                 np.asarray(e_rec["flux"], float)[lo:hi]]})
            files.append(_json_safe(finfo))
            if finfo.get("n_exposures"):
                break
        entry["files"] = files
        report.append(entry)
        print(json.dumps(_json_safe(entry), indent=2, default=str)[:6000])
        print("-" * 78)
    out = {"n": len(report), "release": release, "entries": report}
    (out_dir / "diagnose.json").write_text(json.dumps(_json_safe(out), indent=2, default=str))
    return out


def _sdss_lite_specobj(plate: int, mjd: int, fiber: int, run2d: str | None) -> dict:
    """Pipeline CLASS / SUBCLASS / Z of one fibre from its lite file's SPECOBJ HDU."""
    from astropy.io import fits
    for url in sdss_lite_urls(plate, mjd, fiber, run2d):
        data = fetch_bytes(url, max_bytes=60_000_000, tries=2)
        if data is None:
            continue
        try:
            with fits.open(io.BytesIO(data), memmap=False) as hd:
                got = parse_sdss_spec(hd)
                so = {}
                for h in hd[1:]:
                    if str(h.header.get("EXTNAME", "")).upper() == "SPECOBJ":
                        row = h.data[0]
                        names = [n.upper() for n in h.columns.names]
                        for k in ("CLASS", "SUBCLASS", "Z", "Z_ERR", "ZWARNING", "OBJTYPE",
                                  "SN_MEDIAN_ALL"):
                            if k in names:
                                v = row[names.index(k)]
                                so[k.lower()] = v.strip() if isinstance(v, str) else (
                                    float(v) if np.ndim(v) == 0 else None)
                return {"url": url, "coadd": got.get("coadd"), "specobj": _json_safe(so)}
        except Exception:  # noqa: BLE001
            continue
    return {}


def _galaxy_lines_at(lam0: float, z: float, tol_A: float = 4.0) -> list[str]:
    """Which nebular lines land within ``tol_A`` of ``lam0`` at redshift ``z``."""
    from .galaxy_reject import GALAXY_LINES
    return [n for n, r in GALAXY_LINES.items() if abs(r * (1 + z) - lam0) <= tol_A]


def _sky_peak_near(wave, sky, lam0: float, half_A: float = 15.0) -> dict:
    w = np.asarray(wave, float)
    s = np.asarray(sky, float) if sky is not None else None
    if s is None or s.size != w.size:
        return {}
    sel = np.abs(w - lam0) <= half_A
    if sel.sum() < 5:
        return {}
    idx = np.where(sel)[0]
    i = idx[int(np.nanargmax(s[idx]))]
    base = float(np.nanmedian(s[idx]))
    at = np.abs(w - lam0) <= 2.0
    return {"sky_peak_A": round(float(w[i]), 2), "sky_peak_dA": round(float(w[i] - lam0), 2),
            "sky_peak_over_median": round(float(s[i] / base), 2) if base > 0 else None,
            "sky_at_line_over_median": round(float(np.nanmax(s[at]) / base), 2)
            if base > 0 and at.any() else None}


def recheck_line(root: Path, plate: int, mjd: int, fibers: list[int], lam0: float,
                 run2d: str = "26", n_plate: int = 160, n_other: int = 60,
                 max_other_plates: int = 10) -> dict:
    """Close one open line: are the same-plate co-detections galaxies, a shared
    sky-subtraction residual, or a same-night instrument feature?

    For the target fibre and each same-plate co-detection: the pipeline class
    and redshift, whether a nebular line of that redshift sits at ``lam0``, the
    spectrum's own best nebular family, and -- from the FULL spec file -- the
    per-exposure line flux against the per-exposure sky model at ``lam0`` (a
    sky residual scales with the sky), and where the nearest sky-model peak
    actually is on the vacuum grid.  Then ``n_plate`` fibres of the same plate
    for the rate, and ``n_other`` fibres on every other plate observed the same
    MJD (from the SAS platelist) for a same-night feature.
    """
    from astropy.io import fits
    out: dict = {**_provenance(), "plate": plate, "mjd": mjd, "lam0": lam0, "fibres": []}
    fw = lsf_fwhm_A(lam0, "SDSS-DR17")
    for f in fibers:
        rec: dict = {"fiber": int(f)}
        lite = _sdss_lite_specobj(plate, mjd, f, run2d)
        co = lite.get("coadd")
        rec["specobj"] = lite.get("specobj")
        if co is not None:
            m = measure_line(co["wave"], co["flux"], co["ivar"], lam0, fw, "emission",
                             sky=co.get("sky"))
            rec["coadd"] = _json_safe({k: m.get(k) for k in ("sig", "F", "ew", "cont",
                                                             "sky_peak_sig")})
            rec["coadd_sky_peak"] = _sky_peak_near(co["wave"], co.get("sky"), lam0)
            rec["nebular_scan"] = _json_safe(background_galaxy_scan(
                co["wave"], co["flux"], co["ivar"], lam0, "SDSS-DR17"))
            z = (rec["specobj"] or {}).get("z")
            if z is not None:
                rec["lines_at_lam0_for_pipeline_z"] = _galaxy_lines_at(lam0, float(z))
                # at the pipeline z, measure the whole nebular family directly
                rec["nebular_at_pipeline_z"] = _json_safe(nebular_family_calibrated(
                    [co], lam0, "SDSS-DR17")) if rec["lines_at_lam0_for_pipeline_z"] else None
        # per-exposure: line flux vs sky
        for url in sdss_spec_urls(plate, mjd, f, run2d)[:2]:
            data = fetch_bytes(url, max_bytes=200_000_000, tries=2)
            if data is None:
                continue
            try:
                with fits.open(io.BytesIO(data), memmap=False) as hd:
                    parsed = parse_sdss_spec(hd)
            except Exception:  # noqa: BLE001
                continue
            if not parsed.get("exposures"):
                continue
            _fc, ex = sdss_exposure_measurements(parsed, lam0, "emission")
            peaks = [_sky_peak_near(e["wave"], e.get("sky"), lam0)
                     for e in parsed["exposures"] if abs(np.nanmedian(e["wave"]) - lam0) < 3000]
            rec["exposures"] = [_json_safe({k: e.get(k) for k in
                                            ("expid", "sig", "F", "err", "sky_level",
                                             "sky_peak_sig", "n_cosmic")}) for e in ex]
            rec["exposure_sky_peaks"] = [p for p in peaks if p]
            F = np.array([e.get("F", np.nan) for e in ex], float)
            S = np.array([e.get("sky_level", np.nan) for e in ex], float)
            ok = np.isfinite(F) & np.isfinite(S)
            if ok.sum() >= 3 and np.std(S[ok]) > 0 and np.std(F[ok]) > 0:
                rec["corr_F_vs_sky"] = float(np.corrcoef(F[ok], S[ok])[0, 1])
            rec["full_file"] = url
            break
        out["fibres"].append(_json_safe(rec))

    def _rate(pl: int, mj: int, r2d: str, n: int, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        per_spec = 320 if r2d in ("26", "103", "104") else 500
        fibs = sorted(int(x) for x in rng.choice(np.arange(1, 2 * per_spec + 1),
                                                 size=min(n, 2 * per_spec), replace=False))
        sig, hits = [], []
        for f in fibs:
            if pl == plate and f in fibers:
                continue
            co = _fetch_sdss_coadd(pl, mj, f, r2d)
            if co is None:
                continue
            m = measure_line(co["wave"], co["flux"], co["ivar"], lam0, fw, "emission")
            if m.get("testable") and np.isfinite(m["sig"]):
                sig.append(float(m["sig"]))
                if m["sig"] >= 4.5:
                    hits.append({"fiber": f, "sig": round(float(m["sig"]), 2)})
        a = np.asarray(sig, float)
        return {"plate": pl, "mjd": mj, "run2d": r2d, "n_measured": int(a.size),
                "n_ge4p5": int((a >= 4.5).sum()), "n_ge5": int((a >= 5).sum()),
                "sig_median": float(np.median(a)) if a.size else None, "hits": hits}

    out["same_plate_rate"] = _rate(plate, mjd, run2d, n_plate, 101)
    # every other plate observed the same MJD (same night)
    others = []
    tried = []
    for url in (f"{SDSS_SAS}/sdss/spectro/redux/platelist.fits",
                f"{SDSS_SAS}/sdss/spectro/redux/plates-dr17.fits",
                f"{SDSS_SAS}/sdss/spectro/redux/26/platelist.fits",
                f"{SDSS_SAS}/eboss/spectro/redux/platelist.fits",
                "https://data.sdss.org/sas/dr16/sdss/spectro/redux/platelist.fits",
                "https://data.sdss.org/sas/dr12/sdss/spectro/redux/platelist.fits",
                "https://data.sdss.org/sas/dr9/sdss/spectro/redux/platelist.fits"):
        if others:
            break
        data = fetch_bytes(url, max_bytes=300_000_000, tries=2)
        if data is None:
            tried.append(url)
            out["platelist_error"] = f"unreachable: {', '.join(tried)}"
            continue
        out["platelist_url"] = url
        try:
            with fits.open(io.BytesIO(data), memmap=False) as hd:
                d = hd[1].data
                names = [n.upper() for n in hd[1].columns.names]
                P = np.asarray(d[hd[1].columns.names[names.index("PLATE")]]).astype(int)
                M = np.asarray(d[hd[1].columns.names[names.index("MJD")]]).astype(int)
                R = np.asarray(d[hd[1].columns.names[names.index("RUN2D")]]).astype(str)
                sel = (M == int(mjd)) & (P != int(plate))
                seen = set()
                for p_, r_ in zip(P[sel], R[sel], strict=True):
                    if p_ not in seen:
                        seen.add(int(p_))
                        others.append((int(p_), str(r_).strip()))
        except Exception as exc:  # noqa: BLE001
            out["platelist_error"] = repr(exc)[:300]
    out["same_mjd_plates"] = [p for p, _ in others]
    out["same_mjd_rates"] = [_rate(p, mjd, r or run2d, n_other, 200 + k)
                             for k, (p, r) in enumerate(others[:max_other_plates])]
    od = Path(root) / "results" / "spectra_persist"
    od.mkdir(parents=True, exist_ok=True)
    (od / f"recheck_{plate}_{mjd}_{int(round(lam0))}.json").write_text(
        json.dumps(_json_safe(out), indent=1))
    return out


def _parse_lamost_fits(data: bytes) -> dict | None:
    """wave (vacuum), flux, ivar from a LAMOST low-resolution spectrum file.

    Two layouts exist: DR1-DR7 put a 5-row image in the primary HDU (flux,
    invvar, wavelength, andmask, ormask); DR8+ put a binary table in HDU 1 with
    FLUX / IVAR / WAVELENGTH array columns.
    """
    from astropy.io import fits
    try:
        with fits.open(io.BytesIO(data), memmap=False) as hd:
            p = hd[0].data
            if p is not None and np.ndim(p) == 2 and p.shape[0] >= 3:
                return {"wave": np.asarray(p[2], float), "flux": np.asarray(p[0], float),
                        "ivar": np.asarray(p[1], float),
                        "header": {k: str(hd[0].header.get(k)) for k in
                                   ("OBSID", "DATE-OBS", "CLASS", "SUBCLASS", "Z", "SNRR",
                                    "VACUUM") if k in hd[0].header}}
            for h in hd[1:]:
                names = [n.upper() for n in getattr(h, "columns", []).names] \
                    if hasattr(h, "columns") else []
                if "FLUX" in names and "WAVELENGTH" in names:
                    d = h.data
                    cn = h.columns.names
                    return {"wave": np.asarray(d[cn[names.index("WAVELENGTH")]][0], float),
                            "flux": np.asarray(d[cn[names.index("FLUX")]][0], float),
                            "ivar": np.asarray(d[cn[names.index("IVAR")]][0], float)
                            if "IVAR" in names else None,
                            "header": {k: str(hd[0].header.get(k)) for k in
                                       ("OBSID", "DATE-OBS", "CLASS", "SUBCLASS", "Z",
                                        "SNRR", "VACUUM") if k in hd[0].header}}
    except Exception:  # noqa: BLE001
        return None
    return None


def recheck_second_epoch_and_template(root: Path, spec_id: str, ra: float, dec: float,
                                      lam0: float, subclass: str, radius_arcsec: float = 3.0,
                                      n_template: int = 40) -> dict:
    """Second epoch from every reachable archive, and a same-subclass template.

    * SPARCL at ``radius_arcsec`` (SDSS DR16/17, BOSS, eBOSS, DESI DR1): every
      spectrum at the position, the line measured in each.
    * LAMOST low-resolution (R ~ 1800, vacuum wavelengths): VizieR catalogues
      give the obsid; the spectrum is pulled from the LAMOST data release sites.
    * An empirical template: the median of ``n_template`` continuum-normalised
      spectra of the survey's own ``subclass``, the detector run on the template
      at ``lam0`` and on the target divided by it.  A pseudo-continuum peak
      between molecular band heads shows up in the template itself.
    """
    out: dict = {**_provenance(), "spec_id": spec_id, "ra": ra, "dec": dec, "lam0": lam0,
                 "subclass": subclass, "sparcl": [], "lamost": [], "lamost_errors": []}
    fw = lsf_fwhm_A(lam0, "SDSS-DR17")
    client = _make_client()
    # ---- SPARCL, every release, 3 arcsec
    d = radius_arcsec / 3600.0
    cosd = max(np.cos(np.radians(dec)), 1e-3)
    cons = {"ra": [ra - d / cosd, ra + d / cosd], "dec": [dec - d, dec + d]}
    try:
        found = _find_with_retry(lambda: client.find(
            outfields=["sparcl_id", "ra", "dec", "data_release", "dateobs_center"],
            constraints=cons, limit=100))
        recs = list(_records(found))
    except Exception as exc:  # noqa: BLE001
        recs = []
        out["sparcl_error"] = repr(exc)[:300]
    by_rel: dict[str, list[str]] = {}
    for r in recs:
        by_rel.setdefault(str(_rget(r, "data_release", "")), []).append(
            str(_rget(r, "sparcl_id")))
    target = None
    for rel, ids in by_rel.items():
        for r in sparcl_retrieve(client, ids, rel):
            w = np.asarray(r.get("wavelength", []), float)
            if w.size < 50:
                continue
            m = measure_line(w, np.asarray(r.get("flux", []), float),
                             np.asarray(r.get("ivar", []), float), lam0,
                             lsf_fwhm_A(lam0, rel), "emission")
            ids_ = (sdss_ids_from_record(r) or {}) if rel.upper().startswith(
                ("SDSS", "BOSS")) else {}
            out["sparcl"].append(_json_safe({
                "sparcl_id": r.get("sparcl_id"), "data_release": rel,
                "plate": ids_.get("plate"), "mjd": ids_.get("mjd"),
                "fiberid": ids_.get("fiberid"), "sig": m.get("sig"), "ew": m.get("ew"),
                "testable": m.get("testable"), "subclass": r.get("subclass")}))
            if str(r.get("sparcl_id")) == str(spec_id):
                target = r
    # ---- LAMOST
    obsids = []
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier
        viz = Vizier(columns=["**"], row_limit=50)
        c = SkyCoord(ra * u.deg, dec * u.deg)
        for cat in ("V/164", "V/156", "V/153", "V/149", "V/146"):
            try:
                res = viz.query_region(c, radius=radius_arcsec * u.arcsec, catalog=cat)
            except Exception as exc:  # noqa: BLE001
                out["lamost_errors"].append(f"{cat}: {exc!r}"[:200])
                continue
            for t in res:
                cols = {k.lower(): k for k in t.colnames}
                key = cols.get("obsid") or cols.get("obsid_")
                if key is None:
                    continue
                for row in t:
                    try:
                        obsids.append((cat, t.meta.get("name", cat), int(row[key])))
                    except (TypeError, ValueError):
                        continue
                    keep = {}
                    for cn in t.colnames:
                        if cn.lower() in ("obsid", "obsdate", "class", "subclass", "teff",
                                          "logg", "feh", "[fe/h]", "snrr", "snrg", "z",
                                          "rv", "hrv", "_r"):
                            v = row[cn]
                            keep[cn] = v.item() if hasattr(v, "item") else str(v)
                    out.setdefault("lamost_catalogue", []).append(
                        _json_safe({"catalog": t.meta.get("name", cat), **keep}))
    except Exception as exc:  # noqa: BLE001
        out["lamost_errors"].append(f"vizier: {exc!r}"[:300])
    out["lamost_obsids"] = sorted({o[2] for o in obsids})
    for ob in out["lamost_obsids"]:
        got = None
        statuses = []
        urls = []
        for dr, ver in (("dr11", "v1.0"), ("dr10", "v2.0"), ("dr9", "v2.0"), ("dr8", "v2.0"),
                        ("dr7", "v2.0"), ("dr6", "v2.0"), ("dr5", "v3")):
            urls += [f"https://www.lamost.org/{dr}/{ver}/spectrum/fits/{ob}",
                     f"https://www.lamost.org/{dr}/{ver}/lrs/spectrum/fits/{ob}",
                     f"https://{dr}.lamost.org/{ver}/spectrum/fits/{ob}",
                     f"http://{dr}.lamost.org/{ver}/spectrum/fits/{ob}"]
        for url in urls:
            try:
                rr = _session().get(url, timeout=60, allow_redirects=True)
                statuses.append(f"{rr.status_code} {len(rr.content)}B {url}")
                body = rr.content
                if body[:2] == b"\x1f\x8b":          # LAMOST serves .fits.gz
                    import gzip
                    body = gzip.decompress(body)
                data = body if rr.status_code == 200 and body[:6] == b"SIMPLE" else None
            except Exception as exc:  # noqa: BLE001
                statuses.append(f"ERR {type(exc).__name__} {url}")
                data = None
            if not data:
                continue
            sp = _parse_lamost_fits(data)
            if sp is None:
                continue
            got = (url, sp)
            break
        if got is None:
            out["lamost"].append({"obsid": ob, "error": "spectrum unreachable",
                                  "http": statuses})
            continue
        url, sp = got
        iv = sp["ivar"] if sp["ivar"] is not None else np.ones_like(sp["flux"])
        m = measure_line(sp["wave"], sp["flux"], iv, lam0, lam0 / 1800.0, "emission")
        # sensitivity: what an EW 1.6 A line would give here
        exp_sig = None
        if m.get("testable") and m.get("cont") and m.get("err"):
            exp_sig = float(1.6 * abs(m["cont"]) / m["err"])
        fit = fit_line_profile(sp["wave"], sp["flux"], iv, lam0, lam0 / 1800.0, "emission")
        # the strong OH line 9.4 A redward (6863.96 air = 6865.86 vac): how big is
        # the sky residual there in this spectrum?
        m_oh = measure_line(sp["wave"], sp["flux"], iv, 6865.86, lam0 / 1800.0, "emission")
        out["lamost_fit"] = _json_safe({**fit, "oh6866_sig": m_oh.get("sig"),
                                        "oh6866_ew": m_oh.get("ew")})
        out["lamost"].append(_json_safe({"obsid": ob, "url": url, "header": sp["header"],
                                         "sig": m.get("sig"), "ew": m.get("ew"),
                                         "testable": m.get("testable"),
                                         "reason": m.get("reason"),
                                         "expected_sig_for_ew_1p6": exp_sig}))
    # ---- galaxies with a redshift that puts Halpha near lam0, within 20 arcmin
    try:
        zc = lam0 / 6564.61 - 1.0
        dd = 20.0 / 60.0
        found = _find_with_retry(lambda: client.find(
            outfields=["sparcl_id", "ra", "dec", "redshift", "spectype", "data_release"],
            constraints={"ra": [ra - dd / cosd, ra + dd / cosd], "dec": [dec - dd, dec + dd],
                         "redshift": [zc - 0.01, zc + 0.01]}, limit=500))
        grp = []
        for r in _records(found):
            gra, gde = float(_rget(r, "ra")), float(_rget(r, "dec"))
            sep = 3600.0 * np.hypot((gra - ra) * cosd, gde - dec)
            grp.append({"sparcl_id": str(_rget(r, "sparcl_id")), "ra": gra, "dec": gde,
                        "z": float(_rget(r, "redshift")), "spectype": _rget(r, "spectype"),
                        "data_release": _rget(r, "data_release"),
                        "sep_arcsec": round(float(sep), 1),
                        "dv_kms_vs_halpha_z": round(
                            (float(_rget(r, "redshift")) - zc) / (1 + zc) * 299792.458, 0)})
        out["halpha_z"] = zc
        out["galaxies_near_halpha_z"] = sorted(grp, key=lambda g: g["sep_arcsec"])
    except Exception as exc:  # noqa: BLE001
        out["galaxies_error"] = repr(exc)[:300]
    # ---- SDSS photometric objects within 12 arcsec (is there a galaxy under the star?)
    try:
        sql = ("SELECT n.objID, n.distance, p.ra, p.dec, p.type, p.r, p.petroRad_r, p.clean "
               f"FROM dbo.fGetNearbyObjEq({ra}, {dec}, 0.2) n JOIN PhotoObj p "
               "ON n.objID = p.objID ORDER BY n.distance")
        rr = _session().get("https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch",
                            params={"cmd": sql, "format": "json"}, timeout=120)
        js = rr.json()
        rows = js[0].get("Rows", []) if isinstance(js, list) and js else []
        out["sdss_photo_within_12arcsec"] = [
            {**row, "distance_arcsec": round(60.0 * float(row.get("distance", 0)), 2),
             "type_name": {3: "GALAXY", 6: "STAR"}.get(int(row.get("type", 0)), "OTHER")}
            for row in rows]
    except Exception as exc:  # noqa: BLE001
        out["sdss_photo_error"] = repr(exc)[:300]
    # ---- template from same-subclass stars
    try:
        tpl = {"subclass": subclass}
        found = _find_with_retry(lambda: client.find(
            outfields=["sparcl_id", "redshift"],
            constraints={"data_release": ["SDSS-DR17"], "spectype": ["STAR"]}, limit=3000))
        ids = [str(_rget(r, "sparcl_id")) for r in _records(found)
               if str(_rget(r, "sparcl_id")) != str(spec_id)]
        typed = []
        for k in range(0, len(ids), 500):
            got = _find_with_retry(lambda c=ids[k:k + 500]: client.retrieve(
                uuid_list=c, include=["sparcl_id", "subclass"], dataset_list=["SDSS-DR17"]))
            typed += [str(_rget(r, "sparcl_id")) for r in _records(got)
                      if str(_rget(r, "subclass") or "").strip().upper().startswith(
                          subclass.upper())]
        rng = np.random.default_rng(5)
        pick = [typed[int(i)] for i in rng.permutation(len(typed))[:n_template]]
        grid = np.arange(lam0 - 150.0, lam0 + 150.0, 0.5)
        stack = []
        for r in sparcl_retrieve(client, pick, "SDSS-DR17"):
            w = np.asarray(r.get("wavelength", []), float)
            f = np.asarray(r.get("flux", []), float)
            if w.size < 50 or not (w.min() < grid[0] and w.max() > grid[-1]):
                continue
            fi = np.interp(grid, w, f)
            med = np.nanmedian(fi)
            if np.isfinite(med) and med > 0:
                stack.append(fi / med)
        tpl["n_used"] = len(stack)
        tpl["n_typed_pool"] = len(typed)
        if len(stack) >= 10:
            T = np.nanmedian(np.asarray(stack), axis=0)
            iv_t = np.full(grid.size, 1.0 / max(_mad_std(np.diff(T)) / np.sqrt(2), 1e-4) ** 2)
            mt = measure_line(grid, T, iv_t, lam0, fw, "emission")
            tpl["template_line"] = _json_safe({k: mt.get(k) for k in ("sig", "ew", "F")})
            if target is not None:
                w = np.asarray(target["wavelength"], float)
                f = np.asarray(target["flux"], float)
                iv = np.asarray(target["ivar"], float)
                sel = (w > grid[0]) & (w < grid[-1])
                Ti = np.interp(w[sel], grid, T)
                fn = f[sel] / np.nanmedian(f[sel])
                ratio = fn / Ti
                ivr = iv[sel] * (np.nanmedian(f[sel]) * Ti) ** 2
                mr = measure_line(w[sel], ratio, ivr, lam0, fw, "emission")
                mo = measure_line(w[sel], fn, iv[sel] * np.nanmedian(f[sel]) ** 2, lam0, fw,
                                  "emission")
                tpl["target_line"] = _json_safe({k: mo.get(k) for k in ("sig", "ew")})
                tpl["target_over_template_line"] = _json_safe({k: mr.get(k)
                                                               for k in ("sig", "ew")})
        out["template"] = tpl
    except Exception as exc:  # noqa: BLE001
        out["template"] = {"error": repr(exc)[:300]}
    od = Path(root) / "results" / "spectra_persist"
    od.mkdir(parents=True, exist_ok=True)
    (od / f"recheck_epochs_{str(spec_id)[:8]}_{int(round(lam0))}.json").write_text(
        json.dumps(_json_safe(out), indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="seti.spectra.persist")
    ap.add_argument("--stage", choices=["probe", "run", "reduce", "diagnose", "control",
                                        "recheck", "recheck2"],
                    default="run")
    ap.add_argument("--root", default=".")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--top", type=int, default=0, help="0 = every survivor")
    ap.add_argument("--max-exposures", type=int, default=10)
    ap.add_argument("--release", default="", help="only SDSS or DESI survivors")
    ap.add_argument("--n-control", type=int, default=40,
                    help="control-stage: comparison spectra per surviving line")
    ap.add_argument("--incoming", default="",
                    help="reduce: a directory of per-shard checkpoint directories to "
                         "merge into ckpt/ first (see merge_checkpoints)")
    ap.add_argument("--recheck", default="", help="recheck: PLATE:MJD:FIB1,FIB2,...")
    ap.add_argument("--recheck-lam", default="0")
    ap.add_argument("--recheck-run2d", default="26")
    ap.add_argument("--no-simbad", action="store_true")
    ap.add_argument("--no-nist", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.stage == "probe":
        probe(root)
    elif a.stage == "diagnose":
        diagnose(root, n=a.top or 8, release=a.release or "SDSS")
    elif a.stage == "control":
        controls(root, n=a.n_control)
    elif a.stage == "recheck2":
        sid, ra, dec, sub = a.recheck.split(":")
        rep = recheck_second_epoch_and_template(root, sid, float(ra), float(dec),
                                                float(a.recheck_lam), sub)
        print(json.dumps(rep, default=str)[:6000])
    elif a.stage == "recheck":
        pl, mj, fibs = a.recheck.split(":")
        rep = recheck_line(root, int(pl), int(mj), [int(x) for x in fibs.split(",")],
                           float(a.recheck_lam), run2d=a.recheck_run2d)
        print(json.dumps({k: v for k, v in rep.items() if k != "fibres"}, default=str)[:4000])
    elif a.stage == "run":
        st = run_shard(root, a.shard, a.n_shards, a.top, a.max_exposures, a.release)
        print("[persist] shard stats:", json.dumps(st))
    else:
        if a.incoming:
            ms = merge_checkpoints(root / "results" / "spectra_persist" / "ckpt",
                                   Path(a.incoming),
                                   prefer_sha=os.environ.get("GITHUB_SHA", ""))
            print("[persist] checkpoint merge:", json.dumps(ms))
        reduce_results(root, do_simbad=not a.no_simbad, do_nist=not a.no_nist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["measure_line", "combine_measurements", "classify_persistence", "decode_specobjid",
           "sdss_spec_urls", "parse_sdss_spec", "sdss_exposure_measurements",
           "stack_exposures", "wave_lag", "offset_null",
           "desi_bands_for", "desi_coadd_url", "desi_exposure_rows", "process_spectrum",
           "run_shard", "reduce_results", "final_verdict", "probe", "diagnose",
           "control_sample", "controls", "fit_line_profile", "lsf_fwhm_measured",
           "epoch_series", "background_galaxy_scan", "plate_context", "pixel_coincidence",
           "survey_subclass", "merge_checkpoints", "main"]
