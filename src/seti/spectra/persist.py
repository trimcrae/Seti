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
CKPT_VERSION = 2

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

def classify_persistence(coadd: dict | None, exposures: list[dict]) -> dict:
    """Classify a set of independent per-exposure measurements of one line."""
    from scipy import stats

    tested = [e for e in exposures if e.get("testable")]
    n = len(tested)
    res = {"n_exposures": len(exposures), "n_tested": n, "n_present": 0,
           "frac_present": float("nan"), "mean_F": float("nan"),
           "mean_err": float("nan"), "combined_sig": float("nan"),
           "chi2": float("nan"), "chi2_p": float("nan"), "dominant_frac": float("nan"),
           "sig_without_strongest": float("nan"), "max_exposure_sig": float("nan"),
           "on_sky_line": False, "sky_corr": float("nan"), "n_cosmic_flagged": 0,
           "ratio_to_coadd": float("nan"), "coadd_recovered": None,
           "persistence_class": "untestable", "basis": ""}
    if coadd is not None and coadd.get("testable"):
        res["coadd_recovered"] = bool(coadd["sig"] >= 3.0)
        if np.isfinite(coadd.get("sky_peak_sig", np.nan)) and coadd["sky_peak_sig"] >= SKY_LINE_SIG:
            res["on_sky_line"] = True
    if n == 0:
        reasons = sorted({e.get("reason", "") for e in exposures if e.get("reason")})
        res["basis"] = "no testable exposure" + (f" ({', '.join(reasons)})" if reasons else "")
        return res
    F = np.array([e["F"] for e in tested], float)
    err = np.array([e["err"] for e in tested], float)
    sig = F / err
    present = (sig >= PRESENT_SIG) & (F > 0)
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
        "sky_corr": corr, "n_cosmic_flagged": n_cos,
    })
    if coadd is not None and coadd.get("testable") and coadd.get("F"):
        res["ratio_to_coadd"] = Fbar / coadd["F"]
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
    elif n >= 2 and present.sum() == 0 and Fbar / ebar < 2.0:
        cls = "absent_in_exposures"
        basis = (f"no exposure shows it (max {sig.max():.1f} sigma, combined "
                 f"{Fbar / ebar:.1f} sigma): the coadd feature is not in its inputs")
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
                               max_exposures: int = 10, url_cache: dict | None = None) -> list[dict]:
    """Measure the line in each exposure's cframe (and sky) rows."""
    bands = desi_bands_for(lam0)
    out = []
    url_cache = url_cache if url_cache is not None else {}
    for r in rows[:max_exposures]:
        arms = []
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
        c = combine_measurements(arms)
        c.update({"expid": r["expid"], "night": r["night"], "tileid": r["tileid"],
                  "petal": r["petal"], "fiber": r["fiber"], "exptime": r.get("exptime"),
                  "arms": [a.get("band") for a in arms if a.get("testable")]})
        out.append(c)
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
                "data_release", "wavelength", "flux", "ivar", "mask", "sky",
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


def load_survivors(root: Path):
    import pandas as pd
    p = root / "results" / "spectra_triage" / "priority_targets.csv"
    df = pd.read_csv(p)
    df = df.drop_duplicates(subset=["spec_id", "wavelength"]).reset_index(drop=True)
    return df


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items() if k not in ("wave", "flux", "ivar", "mask", "sky")}
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
                ex = desi_exposure_measurements(rows, lam0, mode, workdir, max_exposures) if rows else []
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
        cls = classify_persistence(ref, ex)
        entry = {"wavelength": lam0, "search_mode": mode,
                 "triage_significance": float(c.get("significance", np.nan)),
                 "sparcl_coadd": _json_safe(sparcl_coadd) if sparcl_coadd else None,
                 "file_coadd": _json_safe(fc) if fc else None,
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
            (ckpt / f"{s}.json").write_text(json.dumps(_json_safe(res)))
            stats["n_processed"] += 1
            for ln in res.get("lines", []):
                print(f"[persist] {s[:8]} {rel} lam={ln['wavelength']:.1f} "
                      f"-> {ln.get('persistence_class')} ({ln.get('basis', '')[:90]}) "
                      f"[{res.get('route')}, {res['elapsed_s']}s]"
                      + (f" ERR={res['error']}" if res.get("error") else ""))
    return stats


def reduce_results(root: Path, do_simbad: bool = True, do_nist: bool = True) -> dict:
    """Merge checkpoints into the flat table + summary; add SIMBAD and line IDs."""
    import pandas as pd

    from .linelist import build_nist_cache, identify_rest_frame, nist_context
    root = Path(root)
    out_dir = root / "results" / "spectra_persist"
    ckpt = out_dir / "ckpt"
    df = load_survivors(root)
    rows = []
    per_exp = {}
    n_stale = 0
    for p in sorted(ckpt.glob("*.json")):
        r = json.loads(p.read_text())
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
                "combined_sig": ln.get("combined_sig"), "mean_F": ln.get("mean_F"),
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
    tab = pd.DataFrame(rows)
    keep = ["spec_id", "wavelength", "significance", "ra", "dec", "redshift", "simbad_id",
            "simbad_otype", "simbad_sptype", "n_lines_in_spectrum"]
    tab = df[[c for c in keep if c in df.columns]].merge(
        tab, on=["spec_id", "wavelength"], how="left") if len(tab) else df[keep].copy()
    for col in ("persistence_class", "route", "n_other_epochs", "second_epoch", "combined_sig",
                "mean_F", "other_best_err_rel", "search_mode", "identifier", "data_release",
                "coadd_ew_A", "n_tested", "n_present", "known_line_match"):
        if col not in tab.columns:
            tab[col] = np.nan
    tab["persistence_class"] = tab["persistence_class"].fillna("not_run")
    if "search_mode" in df.columns:
        tab["search_mode"] = tab["search_mode"].fillna(tab["spec_id"].map(
            df.drop_duplicates("spec_id").set_index("spec_id")["search_mode"]))

    # Rest-frame identification at the star's own catalogue redshift.
    ids = [identify_rest_frame(w, z, mode=str(m)) for w, z, m in
           zip(tab["wavelength"], tab["redshift"].fillna(0.0), tab["search_mode"].fillna("emission"),
               strict=True)]
    for k in ids[0].keys() if ids else []:
        tab[k] = [d[k] for d in ids]

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
                                   zip(tab["wavelength"], tab["redshift"].fillna(0.0), strict=True)]
        except Exception as exc:  # noqa: BLE001
            print(f"[persist] NIST context skipped: {exc!r}")

    tab["verdict"] = [final_verdict(r) for _, r in tab.iterrows()]
    tab = tab.sort_values(["verdict", "combined_sig"], ascending=[True, False])
    tab.to_csv(out_dir / "persistence.csv", index=False)
    (out_dir / "exposures.json").write_text(json.dumps(_json_safe(per_exp)))

    counts = tab["persistence_class"].value_counts().to_dict()
    vcounts = tab["verdict"].value_counts().to_dict()
    alive = tab[tab["verdict"] == "ALIVE_persistent_unidentified"]
    summary = {
        "n_survivors_in": int(len(df)), "n_spectra_in": int(df["spec_id"].nunique()),
        "n_checkpointed_spectra": int(len(list(ckpt.glob("*.json")))),
        "ckpt_version": CKPT_VERSION,
        "n_checkpoints_stale_ignored": int(n_stale),
        "persistence_class_counts": {k: int(v) for k, v in counts.items()},
        "verdict_counts": {k: int(v) for k, v in vcounts.items()},
        "route_counts": {k: int(v) for k, v in tab["route"].fillna("").value_counts().items()},
        "n_known_line_rest_frame": int(tab["known_line_match"].sum()),
        "n_second_epoch_available": int((tab["n_other_epochs"].fillna(0) > 0).sum()),
        "n_second_epoch_confirmed": int((tab["second_epoch"] == "confirmed").sum()),
        "n_alive": int(len(alive)),
        "alive": [
            {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v)
             for k, v in r.items()}
            for r in alive[["spec_id", "identifier", "data_release", "ra", "dec", "wavelength",
                            "search_mode", "coadd_ew_A", "combined_sig", "n_tested", "n_present",
                            "simbad_otype", "known_line_label", "known_line_dv_kms",
                            "second_epoch"]].to_dict("records")],
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
    if cls in ("transient", "absent_in_exposures"):
        return "KILLED_" + cls
    if cls == "sky_residual":
        return "KILLED_sky_residual"
    if cls == "persistent":
        if str(r.get("second_epoch", "")) == "not_seen" and _other_epoch_sensitive(r):
            return "KILLED_second_epoch_absent"
        return "ALIVE_persistent_unidentified"
    if cls == "persistent_2exp":
        return "OPEN_persistent_2exp"
    if cls in ("partial", "inconsistent_strength", "ambiguous", "single_exposure_only"):
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
                                          "err_scale", "mjd")}
                for e in ex]
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="seti.spectra.persist")
    ap.add_argument("--stage", choices=["probe", "run", "reduce", "diagnose"], default="run")
    ap.add_argument("--root", default=".")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--top", type=int, default=0, help="0 = every survivor")
    ap.add_argument("--max-exposures", type=int, default=10)
    ap.add_argument("--release", default="", help="only SDSS or DESI survivors")
    ap.add_argument("--no-simbad", action="store_true")
    ap.add_argument("--no-nist", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.stage == "probe":
        probe(root)
    elif a.stage == "diagnose":
        diagnose(root, n=a.top or 8, release=a.release or "SDSS")
    elif a.stage == "run":
        st = run_shard(root, a.shard, a.n_shards, a.top, a.max_exposures, a.release)
        print("[persist] shard stats:", json.dumps(st))
    else:
        reduce_results(root, do_simbad=not a.no_simbad, do_nist=not a.no_nist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["measure_line", "combine_measurements", "classify_persistence", "decode_specobjid",
           "sdss_spec_urls", "parse_sdss_spec", "sdss_exposure_measurements",
           "desi_bands_for", "desi_coadd_url", "desi_exposure_rows", "process_spectrum",
           "run_shard", "reduce_results", "final_verdict", "probe", "diagnose", "main"]
