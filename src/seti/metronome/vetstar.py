"""METRONOME — the single-star vet: is the clock an eclipsing binary?

One star survived every threshold-free test the channel has: its catalogued
flare epochs are real brightenings in its own light curve, an independent
detector re-finds a clock at the catalogued period, and that period is not the
star's DOMINANT photometric periodicity.  That last clause is the weak one.
``period_is_photometric`` asks only whether the Lomb-Scargle peak of the whole
light curve sits at the clock period.  A LOW-AMPLITUDE periodicity — a grazing
or heavily diluted eclipsing binary, a g-mode pulsator, a blended neighbour —
is never the dominant peak and still produces a strict clock, because the
detector's running-median baseline cannot follow a cycle of comparable length
and the crest of every cycle clears the residual sigma.

A period near 0.42 d sits squarely in the contact-binary and fast-rotator
range, so the eclipsing-binary hypothesis is the one that has to be killed
explicitly.  This module does that four ways, and each is reported as a
number rather than a verdict adjective:

1. **What the variability catalogues say.**  Gaia DR3's own variability
   tables (``vari_summary``, ``vari_eclipsing_binary``,
   ``vari_rotation_modulation``, ``vari_classifier_result``), the non-single
   star solutions (``nss_two_body_orbit``), VSX, the ZTF periodic-variable
   catalogue, and — for a Kepler star, the authoritative one — the Kepler
   Eclipsing Binary Catalog.  "Not listed" is recorded as explicitly as a
   listing; an unreached service is recorded as UNREACHED and is never read
   as "not listed".

2. **The folded light curve, at P and at 2P.**  An eclipsing binary folded at
   half its true period is the classic trap: two eclipses per orbit fold on
   top of each other and the period looks half what it is.  So the light curve
   is detrended on a window long compared with the clock period (which leaves
   the clock in and takes the rotation out) and folded at both.  The reported
   quantities are the folded peak-to-peak amplitude, whether the extremum is a
   dip or a crest, how NARROW it is (an eclipse occupies a small fraction of
   the phase; a sinusoid occupies half), the harmonic content, and at 2P the
   depths of the two half-phase minima separately — unequal minima are an
   eclipsing binary and nothing else.

3. **Whether the events ARE the fold.**  The fold is measured twice: once on
   every cadence, and once with the detected events masked out.  If the folded
   amplitude survives the mask, the star carries a photometric oscillation
   independent of the events.  If it collapses, the fold signal *is* the
   events, and the star has no underlying oscillation at that period.  The
   phase offset between the events and the folded maximum settles which way
   round it is: events sitting on the crest of a coherent oscillation are that
   oscillation's peaks being counted as flares.

4. **Where the flares sit in orbital phase.**  Flares concentrated at one
   phase are an interacting binary, not a clock.  Reported as the Rayleigh
   statistic of the event phases at P and at 2P, and — because a clock is BY
   CONSTRUCTION a phase-clustering statement at its own period, which makes
   the P test circular — as the offset between the event phase and the
   photometric extremum, which is not.

Plus the astrometric route (RUWE, the non-single-star flag, radial-velocity
scatter), and the in-aperture neighbour census, since a blended neighbour's
variability contaminates a Kepler aperture and is not a property of the star.

Everything here takes its data through an injectable callable, so the whole
module is exercised offline against synthetic light curves; the runner supplies
the real ones.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from .acquire import AcquisitionLog, clean_star_id
from .redetect import contiguous_runs, find_flares, stitch_segments

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"

#: Gaia DR3 reference epoch (Julian years).
GAIA_EPOCH = 2016.0
#: The epoch KIC / TIC positions are quoted at.
CATALOGUE_EPOCH = 2000.0

STATUS_OK = "OK"
STATUS_ABSENT = "NOT_LISTED"
STATUS_UNREACHED = "UNREACHED"

#: Gaia DR3 single-row tables that are simply asked for the star's source_id.
#: ``SELECT *`` deliberately: these are one-row-per-source tables, and naming
#: columns here would be guessing at a schema this channel has not read.
GAIA_VARI_TABLES = (
    "gaiadr3.vari_summary",
    "gaiadr3.vari_classifier_result",
    "gaiadr3.vari_eclipsing_binary",
    "gaiadr3.vari_rotation_modulation",
    "gaiadr3.vari_short_timescale",
    "gaiadr3.nss_two_body_orbit",
)

#: The gaia_source columns that bear on binarity and blending.
GAIA_SOURCE_COLUMNS = (
    "source_id", "ra", "dec", "pmra", "pmdec", "parallax", "parallax_error",
    "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "ruwe",
    "astrometric_excess_noise", "astrometric_excess_noise_sig",
    "astrometric_gof_al", "ipd_frac_multi_peak", "ipd_gof_harmonic_amplitude",
    "duplicated_source", "non_single_star", "phot_variable_flag",
    "radial_velocity", "radial_velocity_error", "rv_nb_transits",
    "rv_amplitude_robust", "rv_chisq_pvalue", "rv_renormalised_gof",
    "teff_gspphot", "logg_gspphot", "distance_gspphot",
)

#: VizieR tables asked by identifier rather than by cone.  A Kepler star's
#: eclipsing-binary status is a matter of record in the Kepler EB catalogue,
#: which is indexed by KIC.
VIZIER_ID_TABLES = {
    "kepler_eb_kirk2016": {
        "seed": "J/AJ/151/68", "id_column_patterns": [r"^kic$"], "mission": "kepler"},
    "mcquillan2014_rotation": {
        "seed": "J/ApJS/211/24", "id_column_patterns": [r"^kic$"], "mission": "kepler"},
    "santos2021_rotation": {
        "seed": "J/ApJS/255/17", "id_column_patterns": [r"^kic$"], "mission": "kepler"},
}

#: VizieR tables asked by cone.
VIZIER_CONE_TABLES = {
    "vsx": "B/vsx/vsx",
    "gaia_dr3_vclassre": "I/358/vclassre",
    "gaia_dr3_veb": "I/358/veb",
    "ztf_chen2020": "J/ApJS/249/18",
}


#: Low-order rational multiples of the period under test.  A fold at ``2P/3``
#: sorts the cadences of a signal of period ``P`` into three fixed phases per
#: bin, and a per-bin MEDIAN — unlike a mean — does not average them away, so
#: such a trial period inherits part of the real amplitude and has no place in
#: a control null.
RATIONAL_ALIASES = (1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75, 0.8, 1.0, 1.25, 4.0 / 3.0,
                    1.5, 2.0, 2.5, 3.0)


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _id_str(v) -> str:
    """A catalogue identifier as text, never as ``"1.0"``.

    ``DataFrame.iterrows`` upcasts a mixed-dtype row to float, which silently
    turns a Gaia ``source_id`` into a float that cannot round-trip; the column
    is read directly instead.
    """
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if np.isfinite(v) and float(v).is_integer() else str(v)
    return str(v)


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# light-curve shape
# ---------------------------------------------------------------------------
def detrend_fractional(t, f, *, cadence_days: float, window_days: float = 2.0,
                       gap_days: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Fractional residual about a running median, per contiguous run.

    ``window_days`` is chosen LONG compared with the period under test and
    short compared with the rotation period.  A running median of length
    ``W`` attenuates a sinusoid of period ``P`` by roughly ``|sinc(W/P)|``:
    at ``W/P ~ 5`` the clock survives at ~95% of its amplitude while an
    11-day rotation (``W/P ~ 0.2``) is removed almost entirely.  The
    alternative — the 0.5 d window the flare detector uses — would eat the
    very oscillation this test is looking for.
    """
    from scipy.ndimage import median_filter

    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    order = np.argsort(t)
    t, f = t[order], f[order]
    y = np.full(len(t), np.nan)
    size = int(round(float(window_days) / max(float(cadence_days), 1e-6)))
    size = max(3, size | 1)
    for i0, i1 in contiguous_runs(t, gap_days=gap_days):
        seg = f[i0:i1]
        if len(seg) < 5:
            continue
        base = median_filter(seg, size=min(size, len(seg) | 1), mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            y[i0:i1] = np.where(np.abs(base) > 0, seg / base - 1.0, np.nan)
    return t, y


def fold_profile(t, y, period: float, *, t0: float = 0.0, n_bins: int = 100,
                 mask=None) -> dict:
    """Median phase profile at ``period`` and the shape statistics of it.

    ``depth`` / ``height`` are measured from the profile's own median, so the
    question "is the extremum a dip or a crest?" is answered by which is
    larger.  ``frac_below_half_depth`` is the fraction of filled phase bins
    that lie below half the depth: an eclipse is narrow (a contact binary's
    is the widest at roughly a third, a detached one's a few percent), a
    sinusoid is 1/3 by construction, and a crest-only signal is ~0.
    """
    out = {"period": float(period), "n_bins": int(n_bins), "n_points": 0,
           "amplitude_ptp": float("nan"), "depth": float("nan"), "height": float("nan"),
           "frac_below_half_depth": float("nan"), "frac_above_half_height": float("nan"),
           "phase_of_min": float("nan"), "phase_of_max": float("nan"),
           "bin_sigma_median": float("nan"), "amplitude_sigma": float("nan"),
           "extremum_is_dip": None, "profile": [], "profile_err": [], "profile_n": []}
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    if mask is not None:
        ok &= ~np.asarray(mask, dtype=bool)
    t, y = t[ok], y[ok]
    if not (np.isfinite(period) and period > 0) or len(t) < 20:
        return out
    ph = np.mod((t - float(t0)) / float(period), 1.0)
    idx = np.clip((ph * n_bins).astype(int), 0, n_bins - 1)
    prof = np.full(n_bins, np.nan)
    err = np.full(n_bins, np.nan)
    cnt = np.bincount(idx, minlength=n_bins).astype(int)
    # one sort, then contiguous slices: the obvious `y[idx == b]` per bin is a
    # full-array comparison per bin, and the control-period null folds this
    # hundreds of times.
    order = np.argsort(idx, kind="stable")
    ys = y[order]
    edges = np.concatenate([[0], np.cumsum(cnt)])
    for b in range(n_bins):
        sel = ys[edges[b]:edges[b + 1]]
        if len(sel) < 3:
            continue
        m = float(np.median(sel))
        mad = float(np.median(np.abs(sel - m)))
        prof[b] = m
        s = 1.4826 * mad if mad > 0 else float(np.std(sel))
        err[b] = s / math.sqrt(len(sel)) if np.isfinite(s) else np.nan
    filled = np.isfinite(prof)
    if filled.sum() < max(8, n_bins // 8):
        out.update({"n_points": int(len(t)), "profile": prof.tolist(),
                    "profile_err": err.tolist(), "profile_n": cnt.tolist()})
        return out
    p = prof[filled]
    med = float(np.median(p))
    lo, hi = float(np.min(p)), float(np.max(p))
    depth, height = med - lo, hi - med
    centres = (np.arange(n_bins) + 0.5) / n_bins
    sig = float(np.nanmedian(err[filled])) if np.isfinite(err[filled]).any() else float("nan")
    out.update({
        "n_points": int(len(t)),
        "n_bins_filled": int(filled.sum()),
        "amplitude_ptp": hi - lo,
        "depth": depth,
        "height": height,
        "frac_below_half_depth": float(np.mean(p < med - 0.5 * depth)) if depth > 0 else 0.0,
        "frac_above_half_height": float(np.mean(p > med + 0.5 * height)) if height > 0 else 0.0,
        "phase_of_min": float(centres[filled][int(np.argmin(p))]),
        "phase_of_max": float(centres[filled][int(np.argmax(p))]),
        "bin_sigma_median": sig,
        "amplitude_sigma": (hi - lo) / sig if np.isfinite(sig) and sig > 0 else float("nan"),
        "extremum_is_dip": bool(depth > height),
        "profile": [None if not np.isfinite(v) else float(v) for v in prof],
        "profile_err": [None if not np.isfinite(v) else float(v) for v in err],
        "profile_n": cnt.tolist(),
    })
    return out


def fold_amplitude_significance(t, y, period: float, *, n_bins: int = 100, mask=None,
                                n_control: int = 200, band: float = 0.35,
                                rng=None) -> dict:
    """Folded amplitude at ``period`` against folds at CONTROL periods.

    ``amplitude_sigma`` — the folded peak-to-peak over the per-bin standard
    error — is not a usable threshold on real photometry.  With ~600 cadences
    per phase bin the standard error is tiny, and Kepler's red noise clears
    five of them on almost any period.  So the amplitude is judged against an
    EMPIRICAL null built from the same light curve: the identical fold at
    ``n_control`` periods drawn uniformly in frequency across a band around
    the real one, skipping anything within 2% of ``P``, ``2P`` or ``P/2``.
    Those folds carry the star's own red noise, its gaps and its cadence, so
    ``p_empirical`` is the probability that a period of no significance would
    have produced an amplitude this large.
    """
    rng = np.random.default_rng(20260922) if rng is None else rng
    out = {"period": float(period), "amplitude_ptp": float("nan"),
           "control_median": float("nan"), "control_p95": float("nan"),
           "control_max": float("nan"), "n_control": 0,
           "p_empirical": float("nan"), "z_control": float("nan")}
    real = fold_profile(t, y, period, n_bins=n_bins, mask=mask)
    amp = _f(real.get("amplitude_ptp"))
    out["amplitude_ptp"] = amp
    if not np.isfinite(amp) or not (np.isfinite(period) and period > 0):
        return out
    f0 = 1.0 / float(period)
    # A control period too close to the real one inherits its power through the
    # peak's own width, which is ~1/span in FREQUENCY.  The exclusion is
    # therefore the larger of 2% and twenty frequency-resolution elements; on a
    # 1341-day Kepler baseline the first binds, on a short segment the second.
    tt = np.asarray(t, dtype=float)
    tt = tt[np.isfinite(tt)]
    span = float(tt.max() - tt.min()) if len(tt) > 1 else 0.0
    excl = max(0.02, 20.0 / (span * f0)) if span > 0 else 0.02
    out["exclusion_frac"] = float(excl)
    # MEASURED 2026-09-23 (tess:350479496, P = 4.19 d over ~240 d; and
    # tess:32449963, P = 6.52 d): the exclusion reached 0.345 of the 0.35 band,
    # every draw landed inside it or on a rational alias, the null came back
    # EMPTY -- and an empty null was then read as "not significant", so a
    # 24%-deep narrow eclipse passed as NO_MUNDANE_EXPLANATION_FOUND.  The band
    # now widens to leave room outside the exclusion (still in frequency, still
    # the star's own light curve), and ``band_used`` records what was drawn.
    band_used = float(band)
    if 3.0 * excl > band_used:
        band_used = float(min(0.9, 3.0 * excl))
    out["band_used"] = band_used
    amps = []
    tries = 0
    while len(amps) < int(n_control) and tries < 40 * int(n_control):
        tries += 1
        f = f0 * (1.0 + band_used * rng.uniform(-1.0, 1.0))
        if f <= 0:
            continue
        p_try = 1.0 / f
        if any(abs(p_try / (period * h) - 1.0) < excl for h in RATIONAL_ALIASES):
            continue
        a = _f(fold_profile(t, y, p_try, n_bins=n_bins, mask=mask).get("amplitude_ptp"))
        if np.isfinite(a):
            amps.append(a)
    if not amps:
        return out
    arr = np.asarray(amps, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad if mad > 0 else float(np.std(arr))
    n_ge = int(np.sum(arr >= amp))
    out.update({"control_median": med, "control_p95": float(np.quantile(arr, 0.95)),
                "control_max": float(np.max(arr)), "n_control": int(len(arr)),
                "p_empirical": float((n_ge + 1) / (len(arr) + 1)),
                "z_control": (amp - med) / scale if scale > 0 else float("nan")})
    return out


def harmonic_content(t, y, period: float, *, t0: float = 0.0, n_harmonics: int = 4,
                     mask=None) -> dict:
    """Amplitudes of the first ``n_harmonics`` Fourier terms at ``period``.

    A pure sinusoid puts everything in the fundamental.  An eclipse — a narrow
    feature — needs many harmonics; a contact binary folded at its true period
    is dominated by the SECOND harmonic (two minima per orbit), which is
    exactly why it is so often catalogued at half its orbital period.
    """
    out = {"amplitudes": [], "fundamental_fraction": float("nan"),
           "a2_over_a1": float("nan"), "n_points": 0,
           "phase_of_max_fundamental": float("nan"),
           "phase_of_max_model": float("nan"), "phase_of_min_model": float("nan")}
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    if mask is not None:
        ok &= ~np.asarray(mask, dtype=bool)
    t, y = t[ok], y[ok]
    if len(t) < 50 or not (np.isfinite(period) and period > 0):
        return out
    ph = 2.0 * np.pi * np.mod((t - float(t0)) / float(period), 1.0)
    cols = [np.ones(len(t))]
    for k in range(1, int(n_harmonics) + 1):
        cols += [np.cos(k * ph), np.sin(k * ph)]
    design = np.vstack(cols).T
    try:
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:                         # pragma: no cover - degenerate
        return out
    amps = [float(math.hypot(coef[1 + 2 * (k - 1)], coef[2 + 2 * (k - 1)]))
            for k in range(1, int(n_harmonics) + 1)]
    tot = float(np.sum(np.square(amps)))
    # The crest of the FUNDAMENTAL is the phase reference that survives having
    # the crest itself masked out: the argmax of a binned profile whose peak
    # bins have been removed sits at the edge of the hole, the fit does not.
    a1, b1 = float(coef[1]), float(coef[2])
    grid = np.linspace(0.0, 1.0, 1001)[:-1]
    model = np.zeros(len(grid))
    for k in range(1, int(n_harmonics) + 1):
        model += (coef[1 + 2 * (k - 1)] * np.cos(2 * np.pi * k * grid)
                  + coef[2 + 2 * (k - 1)] * np.sin(2 * np.pi * k * grid))
    out.update({"amplitudes": amps, "n_points": int(len(t)),
                "fundamental_fraction": amps[0] ** 2 / tot if tot > 0 else float("nan"),
                "a2_over_a1": amps[1] / amps[0] if len(amps) > 1 and amps[0] > 0
                else float("nan"),
                "phase_of_max_fundamental": float(np.mod(
                    math.atan2(b1, a1) / (2.0 * np.pi), 1.0)),
                "phase_of_max_model": float(grid[int(np.argmax(model))]),
                "phase_of_min_model": float(grid[int(np.argmin(model))])})
    return out


def odd_even_minima(t, y, period_double: float, *, t0: float = 0.0, n_bins: int = 100,
                    mask=None) -> dict:
    """Depths of the two half-phase minima of a fold at ``period_double``.

    If the true orbital period is twice the detected one, the fold at 2P shows
    a PRIMARY and a SECONDARY eclipse half a cycle apart, and for anything but
    an equal-mass pair their depths differ.  ``delta_sigma`` is that difference
    in units of the two depths' own errors; anything above ~3 with a
    significant amplitude is an eclipsing binary and is not a clock.
    """
    prof = fold_profile(t, y, period_double, t0=t0, n_bins=n_bins, mask=mask)
    out = {"period_double": float(period_double),
           "depth_first_half": float("nan"), "depth_second_half": float("nan"),
           "err_first_half": float("nan"), "err_second_half": float("nan"),
           "delta_sigma": float("nan"), "amplitude_ptp": prof.get("amplitude_ptp"),
           "amplitude_sigma": prof.get("amplitude_sigma")}
    p = np.array([np.nan if v is None else v for v in prof.get("profile", [])], dtype=float)
    e = np.array([np.nan if v is None else v for v in prof.get("profile_err", [])], dtype=float)
    if not len(p) or not np.isfinite(p).any():
        return out
    med = float(np.nanmedian(p))
    half = len(p) // 2
    depths, errs = [], []
    for sl in (slice(0, half), slice(half, len(p))):
        seg, seg_e = p[sl], e[sl]
        if not np.isfinite(seg).any():
            depths.append(float("nan"))
            errs.append(float("nan"))
            continue
        i = int(np.nanargmin(seg))
        depths.append(med - float(seg[i]))
        errs.append(float(seg_e[i]) if np.isfinite(seg_e[i]) else float("nan"))
    d1, d2 = depths
    e1, e2 = errs
    denom = math.hypot(e1 if np.isfinite(e1) else 0.0, e2 if np.isfinite(e2) else 0.0)
    out.update({"depth_first_half": d1, "depth_second_half": d2,
                "err_first_half": e1, "err_second_half": e2,
                "delta_sigma": abs(d1 - d2) / denom if denom > 0 else float("nan")})
    return out


def rayleigh(phases) -> dict:
    """Rayleigh test of circular uniformity with the standard small-``n``
    correction to the p-value (Mardia & Jupp 2000, eq. 6.3.5)."""
    ph = np.asarray(phases, dtype=float)
    ph = ph[np.isfinite(ph)]
    n = len(ph)
    out = {"n": int(n), "rbar": float("nan"), "z": float("nan"), "p": float("nan"),
           "mean_phase": float("nan")}
    if n < 3:
        return out
    ang = 2.0 * np.pi * np.mod(ph, 1.0)
    c, s = float(np.mean(np.cos(ang))), float(np.mean(np.sin(ang)))
    rbar = math.hypot(c, s)
    z = n * rbar * rbar
    p = math.exp(-z) * (1.0 + (2.0 * z - z * z) / (4.0 * n)
                        - (24.0 * z - 132.0 * z ** 2 + 76.0 * z ** 3 - 9.0 * z ** 4)
                        / (288.0 * n * n))
    out.update({"rbar": rbar, "z": z, "p": float(min(max(p, 0.0), 1.0)),
                "mean_phase": float(np.mod(math.atan2(s, c) / (2.0 * np.pi), 1.0))})
    return out


def phase_separation(a: float, b: float) -> float:
    """Signed circular separation ``a - b`` folded into ``[-0.5, 0.5)``."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float((a - b + 0.5) % 1.0 - 0.5)


def periodogram_at(t, y, period: float, *, cadence_days: float,
                   min_period_days: float = 0.1, max_period_days: float = 5.0,
                   samples_per_peak: int = 10, mask=None) -> dict:
    """Lomb-Scargle power AT the clock period on the DETRENDED flux, and how
    that peak ranks against every other independent peak in the band.

    ``period_is_photometric`` in the re-detection asks only about the global
    maximum of the undetrended light curve, where an 11-day rotation buries
    everything.  With the rotation removed the question becomes answerable:
    is there coherent power at the clock period, how big is it, and is it the
    largest thing left?
    """
    out = {"period": float(period), "power_at_period": float("nan"),
           "amplitude_frac_at_period": float("nan"), "peak_period": float("nan"),
           "peak_power": float("nan"), "peak_amplitude_frac": float("nan"),
           "rank_of_period": None, "n_points": 0, "fap_at_period": float("nan")}
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    if mask is not None:
        ok &= ~np.asarray(mask, dtype=bool)
    t, y = t[ok], y[ok]
    if len(t) < 100 or not (np.isfinite(period) and period > 0):
        return out
    try:
        from astropy.timeseries import LombScargle
    except Exception:                                     # noqa: BLE001  pragma: no cover
        return out
    span = float(t[-1] - t[0])
    p_min = float(max(min_period_days, 2.5 * cadence_days))
    p_max = float(min(max_period_days, max(span / 3.0, 4.0 * cadence_days)))
    if not (p_max > p_min):
        return out
    ls = LombScargle(t, y)
    df = 1.0 / (float(samples_per_peak) * max(span, 1.0))
    freq = np.arange(1.0 / p_max, 1.0 / p_min, df)
    if len(freq) < 10:
        return out
    power = ls.power(freq)
    f0 = 1.0 / float(period)
    if not (freq[0] <= f0 <= freq[-1]):
        return out
    j = int(np.argmin(np.abs(freq - f0)))
    model = ls.model(t, freq[j])
    amp = float(np.nanmax(model) - np.nanmin(model))
    i = int(np.nanargmax(power))
    model_pk = ls.model(t, freq[i])
    # rank: how many INDEPENDENT frequencies (one per 1/span) beat it
    step = max(1, int(round(1.0 / (df * max(span, 1.0)))))
    coarse = power[::step] if step > 1 else power
    rank = int(np.sum(coarse > power[j])) + 1
    try:
        fap = float(ls.false_alarm_probability(power[j], method="baluev"))
    except Exception:                                     # noqa: BLE001
        fap = float("nan")
    out.update({"power_at_period": float(power[j]), "amplitude_frac_at_period": amp,
                "peak_period": float(1.0 / freq[i]), "peak_power": float(power[i]),
                "peak_amplitude_frac": float(np.nanmax(model_pk) - np.nanmin(model_pk)),
                "rank_of_period": rank, "n_points": int(len(t)), "fap_at_period": fap})
    return out


def per_segment_amplitude(segments, period: float, *, cadence_days: float,
                          window_days: float = 2.0, n_bins: int = 40) -> list[dict]:
    """Folded amplitude and phase of the clock signal in each quarter/sector.

    A Kepler aperture rotates by 90 degrees every quarter, so a signal that
    belongs to a BLENDED NEIGHBOUR rather than to the target changes amplitude
    from quarter to quarter as the neighbour moves in and out of the mask,
    while a signal intrinsic to the target does not.  The phase, by contrast,
    should be stable either way if the period is right.
    """
    rows = []
    for seg in segments or []:
        t = np.asarray(seg.get("time"), dtype=float)
        f = np.asarray(seg.get("flux"), dtype=float)
        ok = np.isfinite(t) & np.isfinite(f)
        t, f = t[ok], f[ok]
        if len(t) < 200:
            continue
        td, yd = detrend_fractional(t, f, cadence_days=cadence_days,
                                    window_days=window_days)
        prof = fold_profile(td, yd, period, n_bins=n_bins)
        harm = harmonic_content(td, yd, period, n_harmonics=2)
        amps = harm.get("amplitudes") or []
        rows.append({"segment": seg.get("sector"), "n_points": int(len(t)),
                     "t_start": float(t[0]), "t_end": float(t[-1]),
                     "amplitude_ptp": prof.get("amplitude_ptp"),
                     "amplitude_sigma": prof.get("amplitude_sigma"),
                     "phase_of_max": prof.get("phase_of_max"),
                     "a1": amps[0] if amps else float("nan")})
    return rows


def roll_season_test(per_segment, *, mission: str = "kepler", key: str = "amplitude_ptp",
                     n_perm: int = 2000, rng=None) -> dict:
    """Is the folded amplitude a function of the SPACECRAFT ROLL, not of time?

    Kepler rolls 90 degrees every quarter, so quarters sharing ``quarter % 4``
    observe the field in the same orientation and put the same neighbours
    inside the target's aperture.  That makes ``quarter % 4`` a label with no
    astrophysical meaning whatever, and it is the sharpest contamination test
    there is on Kepler photometry:

    * a signal INTRINSIC to the target has a fractional amplitude diluted by
      whatever else is in the mask, ``A_true x (1 - crowding)``.  Crowding
      moves with the mask, but by tens of percent, not by factors;
    * a signal belonging to a NEIGHBOUR has a fractional amplitude
      proportional to how much of *that star's* flux the mask happens to
      catch, which changes by factors as the field rotates.

    So an amplitude locked to the roll season places the signal on another
    star.  The statistic is the ratio of between-season to within-season
    scatter, and its p-value is the fraction of ``n_perm`` random relabellings
    of the segments into the same season sizes that reach it — no distribution
    is assumed, and with four seasons and a dozen quarters none could be.

    TESS sectors do not repeat an orientation this way, so the test is Kepler
    only and reports ``NOT_APPLICABLE`` elsewhere.
    """
    out = {"status": "NOT_APPLICABLE", "mission": str(mission), "n_seasons": 0,
           "season_means": {}, "season_n": {}, "f_stat": float("nan"),
           "p_perm": float("nan"), "ratio_max_min": float("nan"), "n_perm": 0}
    if not str(mission).lower().startswith("kep"):
        return out
    rows = [(int(r["segment"]), _f(r.get(key))) for r in (per_segment or [])
            if r.get("segment") is not None and np.isfinite(_f(r.get(key)))
            and _f(r.get(key)) > 0]
    if len(rows) < 6:
        out["status"] = "TOO_FEW_SEGMENTS"
        return out
    labels = np.array([q % 4 for q, _ in rows])
    vals = np.array([a for _, a in rows], dtype=float)
    seasons = sorted(set(labels.tolist()))
    sized = [int(np.sum(labels == s)) for s in seasons]
    if len(seasons) < 2 or sum(1 for n in sized if n >= 2) < 2:
        out["status"] = "TOO_FEW_SEGMENTS"
        return out

    def f_ratio(lab):
        means = np.array([vals[lab == s].mean() for s in seasons])
        within = np.array([vals[lab == s].var(ddof=1) if int(np.sum(lab == s)) > 1
                           else np.nan for s in seasons])
        w = float(np.nanmean(within))
        b = float(np.var(means, ddof=1)) if len(means) > 1 else float("nan")
        return b / w if np.isfinite(w) and w > 0 else float("nan")

    obs = f_ratio(labels)
    rng = np.random.default_rng(20260922) if rng is None else rng
    n_ge = 0
    n_done = 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(labels)
        f = f_ratio(perm)
        if np.isfinite(f):
            n_done += 1
            if f >= obs:
                n_ge += 1
    means = {int(s): float(vals[labels == s].mean()) for s in seasons}
    out.update({"status": STATUS_OK, "n_seasons": len(seasons),
                "season_means": means,
                "season_n": {int(s): int(np.sum(labels == s)) for s in seasons},
                "segments": {int(q): float(a) for q, a in rows},
                "f_stat": obs, "n_perm": int(n_done),
                "p_perm": float((n_ge + 1) / (n_done + 1)) if n_done else float("nan"),
                "ratio_max_min": (max(means.values()) / min(means.values())
                                  if min(means.values()) > 0 else float("nan"))})
    return out


def flare_mask(t, flares, *, pad_days: float = 0.0) -> np.ndarray:
    """Boolean mask over ``t`` covering every detected event (with padding)."""
    t = np.asarray(t, dtype=float)
    m = np.zeros(len(t), dtype=bool)
    if flares is None or not len(flares):
        return m
    for _, r in pd.DataFrame(flares).iterrows():
        a = _f(r.get("t_start")) - float(pad_days)
        b = _f(r.get("t_end")) + float(pad_days)
        if np.isfinite(a) and np.isfinite(b):
            m |= (t >= a) & (t <= b)
    return m


# ---------------------------------------------------------------------------
# archives
# ---------------------------------------------------------------------------
def catalogue_epochs_for_star(catalogue: str, star_id: str, conf: dict, *, query_fn=None,
                              log: AcquisitionLog | None = None) -> tuple[np.ndarray, dict]:
    """This one star's catalogued flare times, straight from VizieR.

    The vet is about a single star, so it has no business depending on a
    previous run's multi-megabyte event parquet being reachable --- and it was
    not: run 35796061650 silently got nothing, because the workflow's
    permissions block had dropped ``actions: read`` and the cross-run artifact
    download 403'd.  One row per flare for one star is a query, not a
    download, so the parquet is now only a shortcut and this is the fallback.

    The table's columns are read from ``TAP_SCHEMA`` first, as everywhere in
    this channel, and the time role resolved from them: ``t_peak`` when the
    catalogue has a peak column, otherwise the midpoint of ``t_start`` and
    ``t_end`` --- which is what the screen stage does with Yang & Liu 2019's
    ``Begin``/``End``.
    """
    from .acquire import list_tables, table_columns, tap_query
    from .acquire import resolve_columns as _resolve

    qf = query_fn or tap_query
    spec = ((conf.get("catalogues") or {}).get(str(catalogue)) or {})
    seed = spec.get("preferred")
    rec = {"catalogue": str(catalogue), "seed": seed, "status": STATUS_UNREACHED,
           "table": None, "query": "", "error": None, "n_rows": 0, "role": None}
    if not seed:
        rec["status"] = "NO_SEED"
        return np.zeros(0), rec
    tables = [seed]
    try:
        found = list_tables(seed, query_fn=qf)
        if found is not None and len(found) and "table_name" in found:
            tables = [str(v) for v in found["table_name"]] or tables
    except Exception as exc:                              # noqa: BLE001
        rec["error"] = repr(exc)
    sid = clean_star_id(star_id)
    for tname in tables:
        try:
            cols = table_columns(tname, query_fn=qf)
        except Exception as exc:                          # noqa: BLE001
            rec["error"] = repr(exc)
            continue
        roles = _resolve(cols)
        if "star_id" not in roles or not ({"t_peak", "t_start"} & set(roles)):
            continue
        want = [roles["star_id"]]
        for r in ("t_peak", "t_start", "t_end"):
            if r in roles and roles[r] not in want:
                want.append(roles[r])
        sel = ", ".join(f'"{c}"' for c in want)
        adql = f'SELECT {sel} FROM "{tname}" WHERE "{roles["star_id"]}" = {sid}'
        rec.update({"table": tname, "query": adql})
        try:
            df = qf(adql)
        except Exception as exc:                          # noqa: BLE001
            rec["error"] = repr(exc)
            if log:
                log.record(f"vetstar_epochs_{catalogue}", adql[:300], error=repr(exc))
            continue
        n = 0 if df is None else int(len(df))
        rec["n_rows"] = n
        rec["status"] = STATUS_OK if n else STATUS_ABSENT
        if log:
            log.record(f"vetstar_epochs_{catalogue}", adql[:300], rows=n)
        if not n:
            continue
        if "t_peak" in roles and roles["t_peak"] in df.columns:
            t = pd.to_numeric(df[roles["t_peak"]], errors="coerce").to_numpy(dtype=float)
            rec["role"] = "t_peak"
        elif "t_start" in roles and "t_end" in roles:
            a = pd.to_numeric(df[roles["t_start"]], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(df[roles["t_end"]], errors="coerce").to_numpy(dtype=float)
            t = 0.5 * (a + b)
            rec["role"] = "midpoint(t_start,t_end)"
        else:
            t = pd.to_numeric(df[roles["t_start"]], errors="coerce").to_numpy(dtype=float)
            rec["role"] = "t_start"
        t = t[np.isfinite(t)]
        rec["n_rows"] = int(len(t))
        return t, rec
    return np.zeros(0), rec


def _gaia_query(adql: str) -> pd.DataFrame:               # pragma: no cover - network
    from astroquery.gaia import Gaia

    job = Gaia.launch_job(adql)
    res = job.get_results()
    return res.to_pandas() if res is not None else pd.DataFrame()


def propagate_position(ra: float, dec: float, pmra_mas_yr: float, pmdec_mas_yr: float,
                       *, from_epoch: float, to_epoch: float) -> tuple[float, float]:
    """Linear proper-motion propagation.  ``pmra`` is mu_alpha* (already
    carrying the ``cos(dec)`` factor, as Gaia quotes it)."""
    dt = float(to_epoch) - float(from_epoch)
    pmra = 0.0 if not np.isfinite(pmra_mas_yr) else float(pmra_mas_yr)
    pmdec = 0.0 if not np.isfinite(pmdec_mas_yr) else float(pmdec_mas_yr)
    d = float(dec) + pmdec * dt / 3.6e6
    cosd = math.cos(math.radians(float(dec)))
    r = float(ra) + (pmra * dt / 3.6e6) / (cosd if abs(cosd) > 1e-9 else 1e-9)
    return r, d


def angular_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    r1, d1, r2, d2 = (math.radians(x) for x in (ra1, dec1, ra2, dec2))
    s = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(min(1.0, max(-1.0, s)))) * 3600.0


def gaia_neighbourhood(ra: float, dec: float, *, radius_arcsec: float = 20.0,
                       epoch: float = CATALOGUE_EPOCH, query_fn=None,
                       log: AcquisitionLog | None = None) -> dict:
    """Every Gaia DR3 source within ``radius_arcsec``, positions propagated
    back to ``epoch`` before the separation is measured, nearest first.

    The nearest becomes the star's counterpart; the rest are the blending
    census — a Kepler pixel is 3.98 arcsec, so anything inside ~10 arcsec can
    contribute flux to the target's aperture.
    """
    query_fn = query_fn or _gaia_query
    cols = ", ".join(GAIA_SOURCE_COLUMNS)
    rad_deg = float(radius_arcsec) / 3600.0
    where = (f"1 = CONTAINS(POINT('ICRS', ra, dec), "
             f"CIRCLE('ICRS', {float(ra)!r}, {float(dec)!r}, {rad_deg!r}))")
    out = {"status": STATUS_UNREACHED, "query": "", "error": None, "n_sources": 0,
           "match": None, "neighbours": [], "radius_arcsec": float(radius_arcsec),
           "epoch": float(epoch)}
    df = None
    for sel in (cols, "*"):
        adql = f"SELECT {sel} FROM gaiadr3.gaia_source WHERE {where}"
        out["query"] = adql
        try:
            df = query_fn(adql)
            out["error"] = None
            break
        except Exception as exc:                          # noqa: BLE001
            out["error"] = repr(exc)
            if log:
                log.record("vetstar_gaia_source", adql[:300], error=repr(exc))
            df = None
    if df is None:
        return out
    out["status"] = STATUS_OK
    if not len(df):
        out["status"] = STATUS_ABSENT
        return out
    rows = []
    ids = df["source_id"].tolist() if "source_id" in df else [None] * len(df)
    for i, (_, r) in enumerate(df.iterrows()):
        pr, pd_ = propagate_position(_f(r.get("ra")), _f(r.get("dec")),
                                     _f(r.get("pmra")), _f(r.get("pmdec")),
                                     from_epoch=GAIA_EPOCH, to_epoch=float(epoch))
        rec = {k: (None if pd.isna(r.get(k)) else
                   (float(r.get(k)) if isinstance(r.get(k), (int, float, np.floating,
                                                             np.integer))
                    else str(r.get(k))))
               for k in df.columns}
        rec["source_id"] = _id_str(ids[i])
        rec["sep_arcsec_at_epoch"] = angular_separation_arcsec(float(ra), float(dec), pr, pd_)
        rec["sep_arcsec_gaia_epoch"] = angular_separation_arcsec(
            float(ra), float(dec), _f(r.get("ra")), _f(r.get("dec")))
        rows.append(rec)
    rows.sort(key=lambda d: d["sep_arcsec_at_epoch"])
    out.update({"n_sources": len(rows), "match": rows[0], "neighbours": rows[1:]})
    if log:
        log.record("vetstar_gaia_source", out["query"][:300], rows=len(rows))
    return out


def _cell(v):
    """One table cell as JSON-able data.  MEASURED 2026-09-23 (run
    35862302623, tess:260128333): ``nss_two_body_orbit`` carries ARRAY-valued
    columns (``corr_vec``), ``pd.isna`` of an array is an array, and the vet
    crashed on the one star Gaia lists as an orbital binary."""
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = np.asarray(v, dtype=object).ravel()
        return [_cell(x) for x in arr[:64]]
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        return str(v)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    return str(v)


def gaia_variability(source_id: str, *, tables=GAIA_VARI_TABLES, query_fn=None,
                     log: AcquisitionLog | None = None) -> dict:
    """One row per Gaia DR3 variability / non-single-star table for the source.

    Reported verbatim: ``NOT_LISTED`` when the table answered with no row,
    ``UNREACHED`` when it did not answer.  The two are never merged; an
    archive that cannot be reached is not a statement about the star.
    """
    query_fn = query_fn or _gaia_query
    out: dict[str, dict] = {}
    sid = str(source_id).strip()
    for table in tables:
        adql = f"SELECT * FROM {table} WHERE source_id = {sid}"
        rec = {"table": table, "query": adql, "status": STATUS_UNREACHED,
               "error": None, "n_rows": 0, "row": None}
        try:
            df = query_fn(adql)
        except Exception as exc:                          # noqa: BLE001
            rec["error"] = repr(exc)
            if log:
                log.record(f"vetstar_{table}", adql[:300], error=repr(exc))
            out[table] = rec
            continue
        n = 0 if df is None else int(len(df))
        rec["n_rows"] = n
        rec["status"] = STATUS_OK if n else STATUS_ABSENT
        if n:
            r = df.iloc[0]
            rec["row"] = {c: _cell(r.get(c)) for c in df.columns}
        if log:
            log.record(f"vetstar_{table}", adql[:300], rows=n)
        out[table] = rec
    return out


def vizier_cone_report(ra: float, dec: float, tables=None, *, radius_arcsec: float = 5.0,
                       cone_fn=None, log: AcquisitionLog | None = None) -> dict:
    """VizieR cones on the periodic-variable catalogues, reported verbatim."""
    from .acquire import _vizier_cone

    cone_fn = cone_fn or _vizier_cone
    tables = dict(VIZIER_CONE_TABLES if tables is None else tables)
    out: dict[str, dict] = {}
    for name, table in tables.items():
        rec = {"table": table, "status": STATUS_UNREACHED, "error": None,
               "n_rows": 0, "rows": [], "radius_arcsec": float(radius_arcsec)}
        try:
            df = cone_fn(table, float(ra), float(dec), float(radius_arcsec))
        except Exception as exc:                          # noqa: BLE001
            rec["error"] = repr(exc)
            if log:
                log.record(f"vetstar_vizier_{name}", f"cone {table}", error=repr(exc))
            out[name] = rec
            continue
        n = 0 if df is None else int(len(df))
        rec["n_rows"] = n
        rec["status"] = STATUS_OK if n else STATUS_ABSENT
        if n:
            rec["rows"] = [{c: (None if pd.isna(v) else
                                (float(v) if isinstance(v, (int, float, np.floating,
                                                            np.integer)) else str(v)))
                            for c, v in row.items()}
                           for _, row in df.head(10).iterrows()]
        if log:
            log.record(f"vetstar_vizier_{name}", f"cone {table} r={radius_arcsec}\"", rows=n)
        out[name] = rec
    return out


def neighbour_context(gaia: dict, *, g_target: float = float("nan"),
                      max_neighbours: int = 4, max_sep_arcsec: float = 20.0,
                      query_fn=None, cone_fn=None, log: AcquisitionLog | None = None
                      ) -> list[dict]:
    """Ask the archives about the neighbours that could be the real source.

    A Kepler pixel is 3.98 arcsec and the optimal aperture is several of them,
    so a variable star a dozen arcseconds away puts its own signal into the
    target's light curve.  The neighbours worth asking about are the ones that
    are bright enough to matter --- brighter than the target, or flagged
    VARIABLE by Gaia --- and each is put to the same variability tables and the
    same VSX cone as the target, so "the signal belongs to that one instead"
    is a checkable claim rather than a hand-wave.
    """
    out: list[dict] = []
    for n in (gaia or {}).get("neighbours") or []:
        if len(out) >= int(max_neighbours):
            break
        sep = _f(n.get("sep_arcsec_at_epoch"))
        if not np.isfinite(sep) or sep > float(max_sep_arcsec):
            continue
        g = _f(n.get("phot_g_mean_mag"))
        variable = str(n.get("phot_variable_flag") or "").upper() == "VARIABLE"
        brighter = np.isfinite(g) and np.isfinite(g_target) and g <= g_target
        if not (variable or brighter):
            continue
        rec = {"source_id": n.get("source_id"), "sep_arcsec": sep,
               "phot_g_mean_mag": g, "phot_variable_flag": n.get("phot_variable_flag"),
               "why": "gaia_variable" if variable else "brighter_than_target",
               "ra": n.get("ra"), "dec": n.get("dec")}
        rec["gaia_variability"] = gaia_variability(str(n.get("source_id")),
                                                   query_fn=query_fn, log=log) \
            if n.get("source_id") else {}
        ra, dec = _f(n.get("ra")), _f(n.get("dec"))
        rec["vizier_cones"] = vizier_cone_report(ra, dec, radius_arcsec=3.0,
                                                 cone_fn=cone_fn, log=log) \
            if np.isfinite(ra) and np.isfinite(dec) else {}
        out.append(rec)
    return out


def neighbour_periods_matching(neighbours, period: float, *, tol: float = 0.01) -> list[dict]:
    """Neighbours a catalogue gives a period equal to the clock's.

    This is the end of the line for a contamination hypothesis: not "the
    amplitude behaves like a blend" but "that star, this far away, is a
    catalogued variable at this period".  Periods are matched against P, 2P
    and P/2, since a catalogue may list either the pulsation or the orbit.
    """
    import re

    out = []
    for n in neighbours or []:
        hits = []
        sources = dict(n.get("vizier_cones") or {})
        for t, rec in (n.get("gaia_variability") or {}).items():
            sources[t] = {"table": t, "status": rec.get("status"),
                          "rows": [rec["row"]] if rec.get("row") else []}
        for name, rec in sources.items():
            for row in (rec.get("rows") or []):
                for k, v in row.items():
                    if not isinstance(v, (int, float)) or not v or not np.isfinite(v):
                        continue
                    if not re.fullmatch(r"per|per-?[a-z]|period|porb|pf|p", str(k), re.I):
                        continue
                    for h in (0.5, 1.0, 2.0):
                        if abs(float(v) / (period * h) - 1.0) <= float(tol):
                            hits.append({"source": name, "column": str(k),
                                         "period": float(v), "harmonic": h,
                                         "type": next((str(x) for kk, x in row.items()
                                                       if isinstance(x, str) and x.strip()
                                                       and re.fullmatch(
                                                           r"type|class|vartype|best_?class_?name",
                                                           str(kk), re.I)), ""),
                                         "name": next((str(x) for kk, x in row.items()
                                                       if isinstance(x, str)
                                                       and re.fullmatch(r"name|id",
                                                                        str(kk), re.I)), "")})
                            break
        if hits:
            out.append({"source_id": n.get("source_id"), "sep_arcsec": n.get("sep_arcsec"),
                        "phot_g_mean_mag": n.get("phot_g_mean_mag"), "matches": hits})
    return out


def vizier_by_identifier(star_id: str, mission: str, specs=None, *, query_fn=None,
                         log: AcquisitionLog | None = None) -> dict:
    """Identifier lookups in catalogues indexed by KIC/TIC.

    Each seed is expanded through ``TAP_SCHEMA`` (never a guessed table name),
    each table's real columns are read, and the one whose name matches the
    identifier role is used.  For a Kepler star the important entry is the
    Kepler Eclipsing Binary Catalog: if it lists the star, the question is
    closed.
    """
    from .acquire import list_tables, table_columns, tap_query
    from .acquire import resolve_columns as _resolve

    qf = query_fn or tap_query
    specs = dict(VIZIER_ID_TABLES if specs is None else specs)
    sid = clean_star_id(star_id)
    out: dict[str, dict] = {}
    for name, spec in specs.items():
        if spec.get("mission") and str(spec["mission"]).lower() != str(mission).lower():
            continue
        rec = {"seed": spec.get("seed"), "status": STATUS_UNREACHED, "error": None,
               "tables_seen": [], "n_rows": 0, "rows": [], "query": ""}
        try:
            tabs = list_tables(spec.get("seed"), query_fn=qf)
        except Exception as exc:                          # noqa: BLE001
            rec["error"] = repr(exc)
            out[name] = rec
            if log:
                log.record(f"vetstar_id_{name}", str(spec.get("seed")), error=repr(exc))
            continue
        names = ([str(v) for v in tabs["table_name"]]
                 if tabs is not None and len(tabs) and "table_name" in tabs else [])
        rec["tables_seen"] = names
        if not names:
            rec["status"] = STATUS_ABSENT
            out[name] = rec
            continue
        answered = False
        for tname in names:
            try:
                cols = table_columns(tname, query_fn=qf)
            except Exception as exc:                      # noqa: BLE001
                rec["error"] = repr(exc)
                continue
            roles = _resolve(cols, {"star_id": spec.get("id_column_patterns")
                                    or [r"^kic$", r"^tic$", r"^id$"]})
            if "star_id" not in roles:
                continue
            adql = f'SELECT * FROM "{tname}" WHERE "{roles["star_id"]}" = {sid}'
            rec["query"] = adql
            try:
                df = qf(adql)
            except Exception as exc:                      # noqa: BLE001
                rec["error"] = repr(exc)
                if log:
                    log.record(f"vetstar_id_{name}", adql[:300], error=repr(exc))
                continue
            answered = True
            n = 0 if df is None else int(len(df))
            if log:
                log.record(f"vetstar_id_{name}", adql[:300], rows=n)
            if n:
                rec["n_rows"] += n
                rec["rows"] += [{c: (None if pd.isna(v) else
                                     (float(v) if isinstance(v, (int, float, np.floating,
                                                                 np.integer)) else str(v)))
                                 for c, v in row.items()}
                                for _, row in df.head(5).iterrows()]
        if answered:
            rec["status"] = STATUS_OK if rec["n_rows"] else STATUS_ABSENT
        out[name] = rec
    return out


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------
def catalogued_binary_hits(catalogue_report: dict, period: float, *,
                           tol: float = 0.03) -> list[dict]:
    """Rows from any catalogue whose type or class names an eclipsing binary,
    with the catalogued period compared against P and 2P."""
    import re

    eb_pat = re.compile(r"\b(EA|EB|EW|E[:/]?|ECL|ELL|W\s*UMa|CONTACT|BETA\s*LYR|"
                        r"ECLIPSINGBINARY|ECLIPSING)\b", re.I)
    hits = []
    for name, rec in (catalogue_report or {}).items():
        for row in (rec.get("rows") or []):
            blob = " ".join(str(v) for v in row.values() if isinstance(v, str))
            typed = bool(eb_pat.search(blob)) or "eclips" in blob.lower()
            periods = [v for k, v in row.items()
                       if isinstance(v, (int, float)) and v and np.isfinite(v)
                       and re.search(r"per|^p$|^pf$|^porb$", str(k), re.I)]
            near = [p for p in periods
                    if np.isfinite(period) and period > 0
                    and (abs(p / period - 1.0) <= tol or abs(p / (2.0 * period) - 1.0) <= tol
                         or abs(p / (0.5 * period) - 1.0) <= tol)]
            if typed or near:
                hits.append({"source": name, "table": rec.get("table") or rec.get("seed"),
                             "typed_eclipsing": typed, "periods": periods,
                             "periods_matching_clock": near, "row": row})
    return hits


def vet_verdict(report: dict) -> tuple[str, list[str]]:
    """The verdict string and the list of surviving mundane explanations.

    Ordered by how cheaply each kills the candidate: a catalogue listing first,
    then eclipse geometry in the star's own fold, then the oscillation route,
    then the astrometric route.
    """
    reasons: list[str] = []
    surviving: list[str] = []
    p = _f(report.get("period"))

    # A neighbour catalogued at the clock period is the end of the line: the
    # signal has an owner, and it is not the target.
    nb = neighbour_periods_matching(report.get("neighbours") or [], p)
    report["neighbour_period_matches"] = nb
    for h in nb:
        best = h["matches"][0]
        reasons.append(
            f"CONTAMINATING_VARIABLE_AT_P:gaia{h['source_id']}@{_f(h['sep_arcsec']):.1f}arcsec,"
            f"{best['name'] or best['source']},type={best['type'] or '?'},"
            f"P={best['period']:.7f},in="
            + "+".join(sorted({m["source"] for m in h["matches"]})))

    hits = report.get("catalogued_binary_hits") or []
    if hits:
        reasons.append("CATALOGUED_ECLIPSING_BINARY:"
                       + ",".join(sorted({str(h["source"]) for h in hits})))

    def _significant(key: str) -> bool:
        d = report.get(key) or {}
        return bool(np.isfinite(_f(d.get("p_empirical"))) and _f(d.get("p_empirical")) <= 0.01
                    and np.isfinite(_f(d.get("z_control"))) and _f(d.get("z_control")) >= 5.0)

    # The odd-even and dip tests read the UNMASKED fold deliberately.  Masking
    # the detected events punches a hole at whatever phase they occupy, which
    # on a crest-shaped oscillation manufactures both a "dip" and an apparent
    # inequality between the two half-phase minima.  Flares are brightenings
    # and the profile is a per-bin MEDIAN, so they cannot fake an eclipse.
    oe = report.get("odd_even") or {}
    if np.isfinite(_f(oe.get("delta_sigma"))) and _f(oe.get("delta_sigma")) >= 3.0 \
            and _significant("fold_significance_2p"):
        reasons.append(f"UNEQUAL_MINIMA_AT_2P:delta={_f(oe.get('delta_sigma')):.1f}sigma")

    f1 = report.get("fold_at_period") or {}
    if f1.get("extremum_is_dip") and _significant("fold_significance") \
            and np.isfinite(_f(f1.get("frac_below_half_depth"))) \
            and _f(f1.get("frac_below_half_depth")) <= 0.25:
        reasons.append("NARROW_DIP_AT_P:"
                       f"width={_f(f1.get('frac_below_half_depth')):.2f}")

    fm = report.get("fold_at_period_flares_masked") or {}
    ctl = report.get("fold_significance_flares_masked") or {}
    amp_masked = _f(fm.get("amplitude_ptp"))
    p_ctl = _f(ctl.get("p_empirical"))
    z_ctl = _f(ctl.get("z_control"))
    oscillates = _significant("fold_significance_flares_masked")
    if oscillates:
        reasons.append(f"COHERENT_OSCILLATION_AT_P:amp={amp_masked:.5f}"
                       f",p_control={p_ctl:.3g},z={z_ctl:.1f}_with_events_masked")
    else:
        surviving.append("no coherent photometric oscillation survives masking the events")

    off = _f(report.get("event_phase_offset_from_photometric_max"))
    # a mean phase of UNclustered events is noise, not a location: MEASURED
    # 2026-09-23, tess:260128333 was called EVENTS_ON_THE_CREST with Rayleigh
    # p = 0.91 over its 11 events
    ray_p = _f((report.get("event_phase_rayleigh_at_p") or {}).get("p"))
    clustered = bool(np.isfinite(ray_p) and ray_p < 0.01)
    if np.isfinite(off) and abs(off) <= 0.15 and oscillates and clustered:
        reasons.append(f"EVENTS_ON_THE_CREST:offset={off:.3f}cycles")

    g = (report.get("gaia") or {}).get("match") or {}
    ruwe = _f(g.get("ruwe"))
    nss = g.get("non_single_star")
    if np.isfinite(ruwe) and ruwe >= 1.4:
        reasons.append(f"GAIA_RUWE={ruwe:.2f}")
    if nss not in (None, 0, 0.0, "0"):
        reasons.append(f"GAIA_NON_SINGLE_STAR={nss}")
    rv_amp = _f(g.get("rv_amplitude_robust"))
    if np.isfinite(rv_amp) and rv_amp >= 20.0:
        reasons.append(f"GAIA_RV_AMPLITUDE={rv_amp:.1f}km/s")

    ray = report.get("event_phase_rayleigh_at_2p") or {}
    if np.isfinite(_f(ray.get("p"))) and _f(ray.get("p")) < 1e-3 \
            and _f(ray.get("rbar")) > _f((report.get("event_phase_rayleigh_at_p")
                                          or {}).get("rbar", 0.0)):
        reasons.append("EVENTS_CLUSTER_MORE_TIGHTLY_AT_2P")

    seg = report.get("per_segment") or []
    amps = [_f(r.get("amplitude_ptp")) for r in seg]
    amps = [a for a in amps if np.isfinite(a) and a > 0]
    if len(amps) >= 4:
        ratio = max(amps) / min(amps)
        report["per_segment_amplitude_ratio"] = ratio
        if ratio >= 5.0:
            reasons.append(f"AMPLITUDE_VARIES_BETWEEN_SEGMENTS:ratio={ratio:.1f}")

    # The sharpest contamination test on Kepler photometry: `quarter % 4` is a
    # label about the spacecraft, not about the sky, so an amplitude that
    # tracks it puts the signal on a neighbouring star.
    roll = report.get("roll_season") or {}
    if roll.get("status") == STATUS_OK and np.isfinite(_f(roll.get("p_perm"))) \
            and _f(roll.get("p_perm")) <= 0.01 \
            and np.isfinite(_f(roll.get("ratio_max_min"))) \
            and _f(roll.get("ratio_max_min")) >= 1.5:
        reasons.append("AMPLITUDE_TRACKS_SPACECRAFT_ROLL:"
                       f"F={_f(roll.get('f_stat')):.2f},p={_f(roll.get('p_perm')):.4f},"
                       f"ratio={_f(roll.get('ratio_max_min')):.2f}")

    unreached = []
    for key in ("gaia_variability", "vizier_cones", "vizier_by_id"):
        for name, rec in (report.get(key) or {}).items():
            if rec.get("status") == STATUS_UNREACHED:
                unreached.append(f"{key}:{name}")
    if (report.get("gaia") or {}).get("status") == STATUS_UNREACHED:
        unreached.append("gaia:gaia_source")
    if report.get("status") == "analysed":
        for key in ("fold_significance", "fold_significance_flares_masked",
                    "fold_significance_2p"):
            d = report.get(key) or {}
            if int(d.get("n_control") or 0) == 0:
                unreached.append(f"control_null:{key}")
    report["unreached"] = unreached

    if reasons:
        verdict = "MUNDANE_EXPLANATION_FOUND(" + "; ".join(reasons) + ")"
    elif unreached:
        verdict = "DEGRADED(" + ",".join(sorted(unreached)) + "); NO_MUNDANE_EXPLANATION_FOUND"
    else:
        verdict = "NO_MUNDANE_EXPLANATION_FOUND"
    if np.isfinite(p):
        verdict += f" [P={p:.6f} d]"
    return verdict, surviving


#: The vet's vetoes, most-mundane-first.  Each one, on its own, ends a star.
VETSTAR_VETOES = ("CONTAMINATING_VARIABLE_AT_P", "CATALOGUED_ECLIPSING_BINARY",
                  "UNEQUAL_MINIMA_AT_2P", "NARROW_DIP_AT_P",
                  "AMPLITUDE_TRACKS_SPACECRAFT_ROLL", "COHERENT_OSCILLATION_AT_P",
                  "EVENTS_ON_THE_CREST", "GAIA_NON_SINGLE_STAR", "GAIA_RUWE",
                  "GAIA_RV_AMPLITUDE", "EVENTS_CLUSTER_MORE_TIGHTLY_AT_2P",
                  "AMPLITUDE_VARIES_BETWEEN_SEGMENTS")


def vetstar_veto(report: dict) -> str | None:
    """The single most mundane reason the vet found, as a flag name.

    A verdict string is for a human; `summary.json` needs one token so the
    funnel and the channel index can be read by a machine.
    """
    import re

    v = str((report or {}).get("verdict") or "")
    if not v.startswith("MUNDANE_EXPLANATION_FOUND"):
        return None
    # the reasons carry their measurements in several spellings --
    # NAME:value, NAME=value, NAME alone -- so match the token, not a
    # separator that happens to follow it
    for name in VETSTAR_VETOES:
        if re.search(rf"\b{re.escape(name)}\b", v):
            return "vet_" + name.lower()
    return "vet_mundane_explanation_found"


def reconcile_vetstar(out: Path, report: dict) -> dict:
    """Fold the vet's answer back into `summary.json` and `candidates.json`.

    Without this the channel's headline artefact would still read
    ``CLOCK_CANDIDATES_PENDING_VET`` while the vet's own file said the
    candidate is a neighbouring RR Lyrae.  **Demotion only ever removes a
    claim.**

    It is a REBUILD from every per-star vet file present, not an edit for this
    star alone: MEASURED 2026-09-23, five parallel vets each demoted their own
    star in their own checkout of ``candidates.json`` and the last to commit
    erased the other four's demotions.  The report must already be on disk as
    its per-star file (``stage_vetstar`` writes it first).
    """
    from .reconcile import reconcile_all, vetstar_filename

    key = str((report or {}).get("star_key") or "")
    veto = vetstar_veto(report)
    if key:
        # the per-star file IS the record the rebuild reads
        report = dict(report)
        report.setdefault("generated_utc", _now())
        (Path(out) / vetstar_filename(key)).write_text(_dumps(report))
    res = reconcile_all(Path(out), stage="vetstar")
    mine = [d for d in (res.get("demoted_vetstar") or []) if d.startswith(key + ":")]
    return {"status": res.get("status"), "n_annotated": res.get("n_rows", 0),
            "demoted": mine, "veto": veto,
            "n_vetstar_files": res.get("n_vetstar_files", 0),
            "all_demoted_vetstar": res.get("demoted_vetstar", [])}


def _json_default_r(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)



# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------
DEFAULT_VETSTAR = {
    "star_key": "kepler:5879574",
    "catalogue": "kepler_yang2019",
    "period_days": 0.42328185409991,
    "detrend_window_days": 2.0,
    "n_bins": 100,
    "gaia_radius_arcsec": 20.0,
    "vizier_radius_arcsec": 5.0,
    "max_quarters": 20,
    "max_neighbours": 4,
    "per_target_budget_s": 1800.0,
    "budget_s": 5400.0,
    "detector": {"detrend_window_days": 0.5, "sigma_lo": 2.5, "sigma_hi": 3.5,
                 "n_consecutive": 3, "gap_days": 0.5},
}


def one_product_per_segment(segments) -> tuple[list, int]:
    """Keep ONE light-curve product per quarter/sector.

    MEASURED 2026-09-23: tess:260128333's fetch returned sectors 27-31 twice
    (a 20-s and a 2-min product), tess:398943781 and tess:63834969 each sector
    twice.  Stitched, the two are interleaved at different cadences; per
    segment they are counted as two independent sectors, which doubles the
    weight of those sectors in the amplitude-variation test.  The product with
    the LONGER cadence is kept (the flare catalogues were built on it and its
    per-point noise is the lower), ties broken by point count.
    """
    from .redetect import _segment_cadence_days

    best: dict = {}
    order: list = []
    loose = []
    for s in segments or []:
        if s is None:
            continue
        key = s.get("sector")
        if key is None:
            loose.append(s)
            continue
        cad = _segment_cadence_days(s)
        n = len(np.asarray(s.get("time"), dtype=float))
        score = (cad if np.isfinite(cad) else -1.0, n)
        if key not in best:
            order.append(key)
            best[key] = (score, s)
        elif score > best[key][0]:
            best[key] = (score, s)
    kept = [best[k][1] for k in order] + loose
    n_in = sum(1 for s in (segments or []) if s is not None)
    return kept, n_in - len(kept)


def analyse_lightcurve(segments, period: float, *, conf: dict | None = None,
                       catalogue_times=None) -> dict:
    """Every light-curve test in one place, on already-fetched segments."""
    vc = dict(DEFAULT_VETSTAR, **(conf or {}))
    det = dict(DEFAULT_VETSTAR["detector"], **(vc.get("detector") or {}))
    segments, n_dup = one_product_per_segment(segments)
    t, f, meta = stitch_segments(segments, cadence="long")
    meta = dict(meta, n_duplicate_products_dropped=int(n_dup))
    out: dict = {"period": float(period), "lightcurve": meta}
    if len(t) < 200:
        out["status"] = "TOO_FEW_POINTS"
        return out
    cad = _f(meta.get("cadence_days"))
    if not (np.isfinite(cad) and cad > 0):
        cad = float(np.median(np.diff(t)))
    out["status"] = "analysed"

    flares = find_flares(t, f, cadence_days=cad,
                         window_days=float(det["detrend_window_days"]),
                         sigma_lo=float(det["sigma_lo"]), sigma_hi=float(det["sigma_hi"]),
                         n_consecutive=int(det["n_consecutive"]),
                         gap_days=float(det["gap_days"]))
    out["n_flares_redetected"] = int(len(flares))
    # four cadences of padding on each side: a Kepler long-cadence flare's
    # exponential decay runs past the last point the detector called part of
    # it, and an unmasked tail folds coherently at the clock period and would
    # be read as an underlying oscillation that is not there.
    mask = flare_mask(t, flares, pad_days=4.0 * cad)
    out["masked_fraction"] = float(np.mean(mask)) if len(mask) else 0.0

    td, yd = detrend_fractional(t, f, cadence_days=cad,
                                window_days=float(vc["detrend_window_days"]))
    ok = np.isfinite(yd)
    # the mask is built on the sorted t; detrend_fractional sorts too
    n_bins = int(vc["n_bins"])
    out["fold_at_period"] = fold_profile(td, yd, period, n_bins=n_bins)
    out["fold_at_period_flares_masked"] = fold_profile(td, yd, period, n_bins=n_bins,
                                                       mask=mask)
    out["fold_at_2p"] = fold_profile(td, yd, 2.0 * period, n_bins=n_bins)
    out["fold_at_2p_flares_masked"] = fold_profile(td, yd, 2.0 * period, n_bins=n_bins,
                                                   mask=mask)
    out["fold_significance"] = fold_amplitude_significance(td, yd, period, n_bins=n_bins)
    out["fold_significance_flares_masked"] = fold_amplitude_significance(
        td, yd, period, n_bins=n_bins, mask=mask)
    out["fold_significance_2p"] = fold_amplitude_significance(
        td, yd, 2.0 * period, n_bins=n_bins)
    out["harmonics_at_period"] = harmonic_content(td, yd, period, mask=mask)
    out["harmonics_at_2p"] = harmonic_content(td, yd, 2.0 * period, mask=mask)
    out["odd_even"] = odd_even_minima(td, yd, 2.0 * period, n_bins=n_bins)
    out["odd_even_flares_masked"] = odd_even_minima(td, yd, 2.0 * period, n_bins=n_bins,
                                                    mask=mask)
    # the band must contain the clock: at max 5 d a 6.52 d clock returned NaN
    p_hi = max(5.0, 3.0 * float(period))
    out["periodogram_detrended"] = periodogram_at(
        td, yd, period, cadence_days=cad, min_period_days=max(0.1, 2.5 * cad),
        max_period_days=p_hi, mask=mask)
    out["periodogram_detrended_unmasked"] = periodogram_at(
        td, yd, period, cadence_days=cad, min_period_days=max(0.1, 2.5 * cad),
        max_period_days=p_hi)
    out["n_points_detrended"] = int(ok.sum())

    # event phases
    tp = np.asarray(flares["t_peak"], dtype=float) if len(flares) else np.zeros(0)
    out["event_phase_rayleigh_at_p"] = rayleigh(np.mod(tp / period, 1.0)) if len(tp) else {}
    out["event_phase_rayleigh_at_2p"] = rayleigh(np.mod(tp / (2.0 * period), 1.0)) \
        if len(tp) else {}
    ct = np.asarray(catalogue_times, dtype=float) if catalogue_times is not None \
        else np.zeros(0)
    ct = ct[np.isfinite(ct)]
    out["catalogue_phase_rayleigh_at_p"] = rayleigh(np.mod(ct / period, 1.0)) if len(ct) else {}
    out["catalogue_phase_rayleigh_at_2p"] = rayleigh(np.mod(ct / (2.0 * period), 1.0)) \
        if len(ct) else {}

    mean_ph = _f((out["event_phase_rayleigh_at_p"] or {}).get("mean_phase"))
    hm = out["harmonics_at_period"] or {}
    out["photometric_max_phase"] = _f(hm.get("phase_of_max_fundamental"))
    out["event_phase_offset_from_photometric_max"] = phase_separation(
        mean_ph, _f(hm.get("phase_of_max_fundamental")))
    out["event_phase_offset_from_photometric_max_binned"] = phase_separation(
        mean_ph, _f((out["fold_at_period_flares_masked"] or {}).get("phase_of_max")))
    out["event_phase_offset_from_photometric_min"] = phase_separation(
        mean_ph, _f((out["fold_at_period_flares_masked"] or {}).get("phase_of_min")))

    out["per_segment"] = per_segment_amplitude(
        segments, period, cadence_days=cad,
        window_days=float(vc["detrend_window_days"]))
    out["roll_season"] = roll_season_test(out["per_segment"],
                                          mission=str(vc.get("mission", "kepler")))
    return out


def shortlist_catalogue(out: Path, star_key: str) -> str | None:
    """The flare catalogue that put ``star_key`` on the shortlist, read from
    ``candidates.json`` (then ``stars_vetted.csv``)."""
    out = Path(out)
    cp = out / "candidates.json"
    if cp.exists():
        try:
            cj = json.loads(cp.read_text())
        except (OSError, ValueError):
            cj = {}
        for bucket in ("candidates", "watch"):
            for row in (cj or {}).get(bucket) or []:
                if str(row.get("star_key")) == str(star_key) and row.get("catalogue"):
                    return str(row["catalogue"])
    vp = out / "stars_vetted.csv"
    if vp.exists():
        try:
            df = pd.read_csv(vp, dtype={"star_key": str},
                             usecols=lambda c: c in ("star_key", "catalogue"))
            hit = df[df["star_key"].astype(str) == str(star_key)]
            if len(hit) and isinstance(hit["catalogue"].iloc[0], str):
                return str(hit["catalogue"].iloc[0])
        except (OSError, ValueError, KeyError):
            pass
    return None


def stage_vetstar(conf: dict, out: Path, *, star_key: str | None = None,
                  period: float | None = None, kepler_lc_fn=None, lc_fn=None,
                  query_fn=None, gaia_query_fn=None, cone_fn=None,
                  position=None, log=None, budget_s: float | None = None,
                  catalogue: str | None = None, mast_fn=None) -> dict:
    """The full single-star vet.

    Writes ``results/metronome/vetstar_<mission>_<id>.json`` (and
    ``vetstar_fold_<mission>_<id>.csv``) -- ONE FILE PER STAR, so parallel vets
    of different stars never write the same path -- then rebuilds
    ``candidates.json`` / ``summary.json`` from every per-star file present
    (:func:`seti.metronome.reconcile.reconcile_all`)."""
    from ..growth.stage2 import Deadline, MastParams, fetch_kepler_lightcurves, fetch_lightcurves

    vc = dict(DEFAULT_VETSTAR, **(conf.get("vetstar") or {}))
    if star_key:
        vc["star_key"] = str(star_key)
    if period is not None and np.isfinite(float(period)) and float(period) > 0:
        vc["period_days"] = float(period)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = log or AcquisitionLog(prefix="metronome/vetstar")
    t_start = _time.monotonic()

    key = str(vc["star_key"])
    mission, _, sid = key.partition(":")
    sid = sid or key
    # The star's OWN catalogue, from the shortlist that named it.  MEASURED
    # 2026-09-23: five TESS stars from Tu+2022 were vetted with the config
    # default `kepler_yang2019`, so their catalogued epochs were looked up by
    # TIC in a KIC table and came back NOT_LISTED -- n_catalogue_epochs = 0.
    cat_src = "supplied" if catalogue else None
    if not catalogue:
        catalogue = shortlist_catalogue(out, key)
        cat_src = "shortlist" if catalogue else None
    if not catalogue and (conf.get("vetstar") or {}).get("catalogue") \
            and str(key) == str((conf.get("vetstar") or {}).get("star_key")):
        catalogue = (conf.get("vetstar") or {}).get("catalogue")
        cat_src = "config"
    if not catalogue and mission.startswith("kep"):
        catalogue, cat_src = DEFAULT_VETSTAR["catalogue"], "default_for_kepler"
    vc["catalogue"] = catalogue
    p = float(vc["period_days"])
    rep: dict = {"stage": "vetstar", "generated_utc": _now(), "star_key": key,
                 "mission": mission, "star_id": sid, "period": p,
                 "period_double": 2.0 * p, "catalogue": catalogue,
                 "catalogue_source": cat_src}

    # ---- position ----------------------------------------------------
    ra = dec = float("nan")
    if position is not None:
        ra, dec = float(position[0]), float(position[1])
        rep["position_source"] = "supplied"
    else:
        try:
            from .acquire import fetch_positions
            tabs = conf.get("position_tables") or None
            pos = fetch_positions([sid], mission, query_fn=query_fn, mast_fn=mast_fn,
                                  log=log, tables=tabs)
            if pos is not None and len(pos):
                ra, dec = float(pos.iloc[0]["ra"]), float(pos.iloc[0]["dec"])
                rep["position_source"] = "vizier_or_mast_tic"
        except Exception as exc:                          # noqa: BLE001
            rep["position_error"] = repr(exc)
        if not (np.isfinite(ra) and np.isfinite(dec)):
            # the flare catalogue's own star table, on disk from an acquire
            try:
                from .run import _fill_positions_from_rotation, _load_rotation
                rot = _load_rotation(out, mission)
                pos = _fill_positions_from_rotation(pd.DataFrame(), [sid], rot)
                if len(pos):
                    ra, dec = float(pos.iloc[0]["ra"]), float(pos.iloc[0]["dec"])
                    rep["position_source"] = "flare_catalogue_star_table"
            except Exception as exc:                      # noqa: BLE001
                rep["position_error_rotation"] = repr(exc)
    rep["ra"] = None if not np.isfinite(ra) else ra
    rep["dec"] = None if not np.isfinite(dec) else dec

    # ---- archives ----------------------------------------------------
    if np.isfinite(ra) and np.isfinite(dec):
        rep["gaia"] = gaia_neighbourhood(ra, dec, radius_arcsec=float(vc["gaia_radius_arcsec"]),
                                         query_fn=gaia_query_fn, log=log)
        src = ((rep["gaia"] or {}).get("match") or {}).get("source_id")
        rep["gaia_variability"] = gaia_variability(src, query_fn=gaia_query_fn, log=log) \
            if src else {}
        rep["vizier_cones"] = vizier_cone_report(
            ra, dec, radius_arcsec=float(vc["vizier_radius_arcsec"]), cone_fn=cone_fn, log=log)
        rep["neighbours"] = neighbour_context(
            rep["gaia"],
            g_target=_f(((rep["gaia"] or {}).get("match") or {}).get("phot_g_mean_mag")),
            max_neighbours=int(vc.get("max_neighbours", 4)),
            max_sep_arcsec=float(vc["gaia_radius_arcsec"]),
            query_fn=gaia_query_fn, cone_fn=cone_fn, log=log)
    else:
        rep["gaia"] = {"status": STATUS_UNREACHED, "error": "no position"}
        rep["gaia_variability"] = {}
        rep["vizier_cones"] = {}
        rep["neighbours"] = []
    rep["vizier_by_id"] = vizier_by_identifier(sid, mission, query_fn=query_fn, log=log)

    merged = dict(rep.get("vizier_cones") or {})
    merged.update(rep.get("vizier_by_id") or {})
    for tname, grec in (rep.get("gaia_variability") or {}).items():
        merged[tname] = {"table": tname, "status": grec.get("status"),
                         "rows": [grec["row"]] if grec.get("row") else []}
    rep["catalogued_binary_hits"] = catalogued_binary_hits(merged, p)

    # ---- light curve -------------------------------------------------
    rc = dict(conf.get("redetect") or {})
    params = MastParams(per_target_budget_s=float(vc["per_target_budget_s"]),
                        kepler_per_target_budget_s=float(vc["per_target_budget_s"]),
                        max_sectors=int(vc["max_quarters"]),
                        kepler_max_quarters=int(vc["max_quarters"]),
                        retries=int(rc.get("retries", 3)),
                        kepler_retries=int(rc.get("retries", 3)),
                        download_dir=str(out / "lc_cache"))
    deadline = Deadline(budget_s=float(budget_s if budget_s is not None else vc["budget_s"]))
    try:
        if mission.startswith("kep"):
            segs, status, route, _prov = fetch_kepler_lightcurves(
                sid, lc_fn=kepler_lc_fn, params=params, log=log, deadline=deadline, key=key)
        else:
            segs, status, route = fetch_lightcurves(
                sid, lc_fn=lc_fn, params=params, log=log, deadline=deadline, key=key)
    except Exception as exc:                              # noqa: BLE001
        segs, status, route = [], "QUERY_FAILED", ""
        log.record(f"vetstar_lightcurves_{key}", sid, error=repr(exc))
    rep["fetch_status"] = status
    rep["fetch_route"] = route

    ct = np.zeros(0)
    try:
        from .redetect import _catalogue_times
        ct = _catalogue_times(out, str(vc.get("catalogue")), sid)
    except Exception as exc:                              # noqa: BLE001
        rep["catalogue_times_error"] = repr(exc)
    rep["catalogue_epochs_source"] = "prior run's event parquet" if len(ct) else None
    if not len(ct):
        ct, erec = catalogue_epochs_for_star(str(vc.get("catalogue")), sid, conf,
                                             query_fn=query_fn, log=log)
        rep["catalogue_epochs_query"] = erec
        rep["catalogue_epochs_source"] = "vizier" if len(ct) else None
    rep["n_catalogue_epochs"] = int(len(ct))

    if status == "OK" and segs:
        rep.update(analyse_lightcurve(segs, p, conf=dict(vc, mission=mission),
                                      catalogue_times=ct))
        rep["period"] = p
    else:
        rep["status"] = "NO_LIGHTCURVE"

    verdict, surviving = vet_verdict(rep)
    rep["verdict"] = verdict
    rep["surviving_explanations"] = surviving
    rep["acquisition"] = log.as_dict() if hasattr(log, "as_dict") else {}
    rep["elapsed_s"] = round(_time.monotonic() - t_start, 1)

    from .reconcile import vetstar_filename

    (out / vetstar_filename(key)).write_text(_dumps(rep))
    rep["reconciliation"] = reconcile_vetstar(out, rep)
    (out / vetstar_filename(key)).write_text(_dumps(rep))
    _write_fold_csv(out, rep)
    print(f"[metronome/vetstar] {key}: {verdict}")
    return rep


def _write_fold_csv(out: Path, rep: dict) -> None:
    rows = []
    for name in ("fold_at_period", "fold_at_period_flares_masked", "fold_at_2p",
                 "fold_at_2p_flares_masked"):
        d = rep.get(name) or {}
        prof = d.get("profile") or []
        err = d.get("profile_err") or []
        cnt = d.get("profile_n") or []
        n = len(prof)
        for i in range(n):
            rows.append({"fold": name, "period_days": d.get("period"),
                         "phase": (i + 0.5) / n,
                         "flux_frac": prof[i],
                         "err": err[i] if i < len(err) else None,
                         "n": cnt[i] if i < len(cnt) else None})
    if rows:
        from .reconcile import vetstar_filename

        name = vetstar_filename(str(rep.get("star_key") or "unknown"), suffix=".csv",
                                prefix="vetstar_fold_")
        pd.DataFrame(rows).to_csv(Path(out) / name, index=False)


def _dumps(obj) -> str:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return v if np.isfinite(v) else None
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        return str(o)

    return json.dumps(obj, indent=1, default=default, allow_nan=True)


__all__ = ["CATALOGUE_EPOCH", "DEFAULT_VETSTAR", "GAIA_EPOCH", "GAIA_SOURCE_COLUMNS",
           "VETSTAR_VETOES",
           "GAIA_TAP", "GAIA_VARI_TABLES", "STATUS_ABSENT", "STATUS_OK", "STATUS_UNREACHED",
           "VIZIER_CONE_TABLES", "VIZIER_ID_TABLES",
           "analyse_lightcurve", "angular_separation_arcsec", "catalogue_epochs_for_star",
           "catalogued_binary_hits",
           "detrend_fractional", "flare_mask", "fold_amplitude_significance",
           "fold_profile", "gaia_neighbourhood",
           "gaia_variability", "harmonic_content", "neighbour_context",
           "neighbour_periods_matching", "odd_even_minima",
           "per_segment_amplitude",
           "periodogram_at", "phase_separation", "propagate_position", "rayleigh",
           "reconcile_vetstar", "vetstar_veto",
           "roll_season_test",
           "stage_vetstar", "vet_verdict", "vizier_by_identifier", "vizier_cone_report"]
