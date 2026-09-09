"""The paces: every existing light-curve and spectrum channel, run on Roman products.

``docs/roman.md`` section 3.  Nothing in this module detects anything itself: it
takes a :class:`~seti.roman.schema.LightCurve` or :class:`~seti.roman.schema.Spectrum`,
turns it into what each detector already consumes (time, magnitude, error; or
wavelength, flux, error), sets the Roman-appropriate parameters from
``config/roman.yaml`` (``paces:``), calls the detector **unchanged**, and wraps
the answer in one record shape so the screen and assess stages can count them.

Detectors reused, and what had to be adapted around each:

* ``dimming.dips.detect_dips`` / ``dimming.glint.detect_glints`` --- report counts
  and scores, not event epochs.  For the F146/F087 achromaticity test the event's
  epochs are re-identified with the detector's own per-epoch criterion (bright
  baseline, depth, significance) and the fractional amplitude in each band is
  compared with ``tocsin.photometry.greyness_z``.  The test needs a colour epoch
  *inside* the event; the F087 cadence (12 h) therefore sets the shortest event
  it can reach, and a shorter event is reported ``no_colour_epoch_in_window``,
  never as achromatic.
* ``dimming.secular.detect_secular_fade`` / ``rust.trend.detect_rust`` --- both
  bin "seasons" as ``floor((t - t0) / season_days)`` from the first epoch.  A
  72-day GBTDS season followed by a ~110-day gap does not survive that binning
  at ``season_days = 72`` (seasons straddle bin edges and split), so the bridge
  finds the seasons from the sampling gaps (:func:`seasons_from_gaps`) and hands
  the detector the bin width that reproduces them one-to-one, checking that it
  does and recording ``season_binning_matches_gaps``.
* ``knell.cease.analyze_band`` --- takes ``block_mode="gap"`` but does not forward
  a gap length, so its blocks split at the detector's own default (90 d); the
  record says whether those blocks coincide with the paces seasons.  Its
  generalised Lomb-Scargle builds a dense ``(n_freq, n_epoch)`` grid: a 72-day
  block at 12-minute cadence searched down to 0.02 d is ~1.5e8 cells, several
  gigabytes.  The bridge bins the series in time until the footprint fits
  ``paces.knell.max_gls_cells`` and raises the shortest period to three bins,
  recording both.  Efficiency injection is off unless the config turns it on.
* ``metronome.clock.analyze_star`` --- wants event *times* and observing
  *windows*.  Roman publishes neither for a star, so :func:`detect_flares` finds
  brightenings in the light curve itself (a crude finder; the timing test is
  the deliverable) and ``windows_from_events`` builds the windows from the
  light curve's own epochs.
* ``spectra.detect.find_emission_lines`` / ``spectra.absorb.find_absorption_lines``
  --- take one LSF width in pixels; the bridge derives it from the product's
  resolving power as ``lambda / (R * 2.3548 * dlambda)`` and uses the median over
  the spectrum, recording the range when R varies (the prism).

Every record carries ``relative_flux`` (no zero point on the product), the
parameters actually used, and a ``status`` in ``{"ran", "insufficient",
"relative_flux_only", "not_applicable", "error"}``.  Nothing here reaches the
network; the synthesis functions at the bottom exist for the tests and the
selftest and stamp ``synthetic: True`` on what they make.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from seti.dimming.dips import detect_dips
from seti.dimming.glint import detect_glints
from seti.dimming.secular import detect_secular_fade
from seti.knell.cease import analyze_band
from seti.metronome.clock import analyze_star
from seti.metronome.windows import windows_from_events
from seti.rust.trend import detect_rust
from seti.spectra.absorb import find_absorption_lines
from seti.spectra.detect import find_emission_lines
from seti.tocsin.photometry import fractional_amplitude, greyness_z

from .schema import Funnel, LightCurve, Spectrum, flag_value, json_safe

MAG_TO_FRAC = 0.4 * math.log(10.0)        # d(flux)/flux per magnitude, small-signal
FWHM_TO_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))   # 2.3548
STATUSES = ("ran", "insufficient", "relative_flux_only", "not_applicable", "error")
CHANNELS = ("dips", "glint", "secular", "rust", "knell", "metronome")

# Interest thresholds for ``pace_lightcurve``'s ``flags`` list.  Overridable from
# ``paces.flags`` in config; these defaults are deliberately conservative
# because a flag here only routes a star to the vet stage, it never tiers it.
DEFAULT_FLAG_LEVELS = {
    "secular_sigma_min": 5.0,
    "rust_p_max": 1e-3,
    "metronome_p_max": 1e-3,
}
MIN_EPOCHS_ANY = 10


# --------------------------------------------------------------------------------------
# Seasons and series
# --------------------------------------------------------------------------------------

def seasons_from_gaps(mjd, gap_days: float) -> np.ndarray:
    """Season label (0..k-1) per epoch, split wherever the sampling gap exceeds ``gap_days``.

    The GBTDS season unit is the ~72-day bulge-visibility window, six per
    field, separated by ~110-day gaps; nothing here is a year.  Labels are
    returned in the *input* order (the epochs need not be sorted) and count up
    in time.  Non-finite times get label ``-1``.
    """
    t = np.asarray(mjd, dtype=float)
    lab = np.full(t.shape, -1, dtype=int)
    ok = np.isfinite(t)
    if not ok.any():
        return lab
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(t[idx], kind="mergesort")]
    ts = t[order]
    split = np.diff(ts) > float(gap_days)
    lab[order] = np.concatenate([[0], np.cumsum(split)]).astype(int)
    return lab


def dq_reject_mask(conf: dict) -> int:
    """OR of the ``lightcurve_dq_reject`` flag names, resolved through :func:`flag_value`."""
    mask = 0
    for name in conf.get("lightcurve_dq_reject") or []:
        try:
            mask |= flag_value(conf, str(name))
        except KeyError:
            continue
    return int(mask)


def to_mag_series(lc: LightCurve, conf: dict) -> dict | None:
    """``{t, mag, magerr, relative_flux, ...}`` on the usable epochs, or ``None``.

    Applies the config dq rejection, sorts in time, and converts through
    ``LightCurve.to_mag``.  ``relative_flux`` is True when the product carries no
    zero point: the magnitudes are then internally consistent (zp 25) but must
    never be compared with anything outside this curve.  ``None`` when fewer
    than ten epochs survive --- below that no bridged detector can say anything.
    """
    if lc is None or lc.n == 0:
        return None
    mask = dq_reject_mask(conf)
    lcs = lc.sorted()
    n_total = lcs.n
    n_dq = int(np.sum((lcs.dq & mask) != 0)) if (lcs.dq is not None and mask) else 0
    tm = lcs.to_mag(mask)
    if tm is None:
        return None
    t, mag, magerr = tm
    if t.size < MIN_EPOCHS_ANY:
        return None
    dt = np.diff(t)
    cadence = float(np.median(dt[dt > 0])) if np.any(dt > 0) else float("nan")
    return {
        "t": t, "mag": mag, "magerr": magerr,
        "relative_flux": lc.flux_zp_ab is None,
        "zp_ab": lc.flux_zp_ab,
        "n_total": int(n_total), "n_good": int(t.size), "n_dq_rejected": n_dq,
        "dq_reject_mask": mask, "cadence_days": cadence,
    }


def _series_flux(ms: dict) -> tuple[np.ndarray, np.ndarray]:
    """Flux (arbitrary scale) and its error from a magnitude series.

    Only *fractional* amplitudes are ever formed from these, so the scale is
    irrelevant and the zero point is not needed.
    """
    f = 10.0 ** (-0.4 * ms["mag"])
    return f, f * MAG_TO_FRAC * ms["magerr"]


def _running_median(x: np.ndarray, window: int) -> np.ndarray:
    """Sliding median with edge replication; ``window`` is forced odd and <= n."""
    n = x.size
    w = int(max(3, window))
    w = w if w % 2 == 1 else w + 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 3:
        return np.full(n, float(np.median(x)))
    half = w // 2
    padded = np.pad(x, half, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, w)
    return np.median(view, axis=1)


def _runs(idx: np.ndarray, t: np.ndarray, merge_gap_d: float) -> list[np.ndarray]:
    """Split flagged epoch indices into runs of adjacent epochs closer than ``merge_gap_d``."""
    if idx.size == 0:
        return []
    runs: list[list[int]] = [[int(idx[0])]]
    for a, b in zip(idx[:-1], idx[1:], strict=False):
        if b != a + 1 or (t[b] - t[a]) > merge_gap_d:
            runs.append([int(b)])
        else:
            runs[-1].append(int(b))
    return [np.asarray(r, dtype=int) for r in runs]


def detect_flares(t, mag, magerr, k_sigma: float = 5.0, bright_min: float = 0.10,
                  *, window_epochs: int = 101, merge_gap_d: float = 0.05) -> np.ndarray:
    """Peak times of brightening events, for METRONOME's timing test.

    A crude finder, on purpose: an epoch counts when it is at least
    ``bright_min`` (fractional flux) *and* ``k_sigma`` (per-epoch error) above a
    running median of ``window_epochs`` epochs; contiguous flagged epochs closer
    than ``merge_gap_d`` form one event; the event's time is the epoch of its
    largest brightening.  No profile is fitted and no energy is estimated.  The
    deliverable is what ``metronome.clock.analyze_star`` then does with these
    times against the survey's own observing windows --- a clock test --- and a
    finder that misses half the flares only weakens that test, it cannot
    fabricate a period.
    """
    t = np.asarray(t, dtype=float)
    m = np.asarray(mag, dtype=float)
    e = np.asarray(magerr, dtype=float)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[ok], m[ok], e[ok]
    if t.size < 5:
        return np.zeros(0)
    order = np.argsort(t, kind="mergesort")
    t, m, e = t[order], m[order], e[order]
    med = _running_median(m, window_epochs)
    dmag = med - m                                   # >0 when brighter than the local median
    frac = 10.0 ** (0.4 * np.clip(dmag, 0.0, None)) - 1.0
    sig = dmag / (MAG_TO_FRAC * e)
    flagged = np.flatnonzero((frac >= float(bright_min)) & (sig >= float(k_sigma)))
    peaks = [float(t[r[int(np.argmax(frac[r]))]]) for r in _runs(flagged, t, merge_gap_d)]
    return np.asarray(peaks, dtype=float)


# --------------------------------------------------------------------------------------
# Season binning for the fixed-width detectors
# --------------------------------------------------------------------------------------

def _season_bin_days(t: np.ndarray, labels: np.ndarray, conf_days: float) -> dict:
    """Bin width for a ``floor((t - t0) / W)`` detector that reproduces the gap seasons.

    Tries the median spacing between season starts first (for a regular
    survey calendar every season then lands in its own bin), then the config
    value.  ``matches`` is True when the mapping season -> bin is one-to-one; if
    neither candidate achieves that the spacing is used and the record says so.
    """
    t0 = float(np.min(t))
    starts = np.array([float(np.min(t[labels == k])) for k in np.unique(labels)])
    spacing = float(np.median(np.diff(starts))) if starts.size > 1 else float(conf_days)
    candidates = [spacing, float(conf_days)]
    for w in candidates:
        if not (np.isfinite(w) and w > 0):
            continue
        bins = np.floor((t - t0) / w).astype(int)
        pairs = {(int(a), int(b)) for a, b in zip(labels, bins, strict=False)}
        one_to_one = (len({a for a, _ in pairs}) == len(pairs)
                      and len({b for _, b in pairs}) == len(pairs))
        if one_to_one:
            return {"season_bin_days": w, "season_binning_matches_gaps": True,
                    "season_start_spacing_days": spacing}
    return {"season_bin_days": spacing, "season_binning_matches_gaps": False,
            "season_start_spacing_days": spacing}


def _bin_series(t: np.ndarray, mag: np.ndarray, magerr: np.ndarray,
                bin_days: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse-variance-weighted time bins; a bin's time is the mean of its epochs."""
    key = np.floor((t - t[0]) / float(bin_days)).astype(int)
    w = 1.0 / np.clip(magerr, 1e-6, None) ** 2
    uniq, inv = np.unique(key, return_inverse=True)
    sw = np.bincount(inv, weights=w, minlength=uniq.size)
    st = np.bincount(inv, weights=w * t, minlength=uniq.size) / sw
    sm = np.bincount(inv, weights=w * mag, minlength=uniq.size) / sw
    return st, sm, 1.0 / np.sqrt(sw)


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------

def _record(channel: str, status: str, result=None, params: dict | None = None,
            notes: list[str] | None = None, **extra) -> dict:
    rec = {"channel": channel, "status": status,
           "result": None if result is None else json_safe(result.as_dict()
                                                            if hasattr(result, "as_dict")
                                                            else result),
           "params": json_safe(params or {}), "notes": list(notes or [])}
    rec.update(json_safe(extra))
    return rec


def _paces(conf: dict, channel: str) -> dict:
    return dict((conf.get("paces") or {}).get(channel) or {})


def _relative_gate(ms: dict, conf: dict, channel: str, params: dict) -> dict | None:
    """The record to return when a season-comparing channel may not run in relative flux."""
    allow = bool((conf.get("paces") or {}).get("allow_relative_flux", True))
    if ms["relative_flux"] and not allow:
        return _record(channel, "relative_flux_only", None, params,
                       ["product carries no zero point and paces.allow_relative_flux is false"],
                       relative_flux=True)
    return None


# --------------------------------------------------------------------------------------
# Achromaticity (F146 event vs the F087 colour series)
# --------------------------------------------------------------------------------------

def _event_epochs(ms: dict, kind: str, depth_min: float, k_sigma: float,
                  merge_gap_d: float) -> np.ndarray | None:
    """Epoch indices of the strongest event, by the detector's own per-epoch rule.

    ``dips`` uses the bright-state (20th percentile) baseline, a fractional
    deficit >= ``depth_min`` and >= ``k_sigma``, and counts only runs of >= 2
    epochs, exactly as ``detect_dips`` does.  ``glint`` uses the median
    baseline and a fractional excess, as ``detect_glints`` does, and admits a
    single epoch since the detector does.
    """
    t, m, e = ms["t"], ms["mag"], ms["magerr"]
    if kind == "dips":
        base = float(np.nanpercentile(m, 20))
        dmag = m - base
        frac = 1.0 - 10.0 ** (-0.4 * dmag)
        min_run = 2
    else:
        base = float(np.median(m))
        dmag = base - m
        frac = 10.0 ** (0.4 * np.clip(dmag, 0.0, None)) - 1.0
        min_run = 1
    sig = dmag / (MAG_TO_FRAC * e + 1e-9)
    flagged = np.flatnonzero((frac >= depth_min) & (sig >= k_sigma))
    runs = [r for r in _runs(flagged, t, merge_gap_d) if r.size >= min_run]
    if not runs:
        return None
    return max(runs, key=lambda r: float(np.max(frac[r])))


def _band_amplitude(t: np.ndarray, f: np.ndarray, fe: np.ndarray, in_event: np.ndarray):
    """Fractional amplitude of the in-event epochs against the out-of-event median."""
    out = ~in_event
    if in_event.sum() == 0 or out.sum() < 5:
        return None
    base = float(np.median(f[out]))
    base_err = 1.4826 * float(np.median(np.abs(f[out] - base))) / math.sqrt(out.sum())
    w = 1.0 / np.clip(fe[in_event], 1e-30, None) ** 2
    f_ev = float(np.sum(w * f[in_event]) / np.sum(w))
    f_ev_err = float(1.0 / math.sqrt(np.sum(w)))
    return fractional_amplitude(f_ev - base, f_ev_err, base, base_err)


def _achromatic_test(ms: dict, colour_lc: LightCurve | None, conf: dict, kind: str,
                     depth_min: float, k_sigma: float, merge_gap_d: float) -> dict:
    """Compare the event's fractional amplitude in F146 with F087 at the same times."""
    out: dict[str, Any] = {"achromatic_test": "not_run", "achromatic_z": None}
    if colour_lc is None:
        out["notes"] = ["no colour series supplied"]
        return out
    ev = _event_epochs(ms, kind, depth_min, k_sigma, merge_gap_d)
    if ev is None:
        out["notes"] = ["detector reported an event but no epoch passes its per-epoch rule"]
        return out
    t146 = ms["t"]
    cad146 = ms["cadence_days"] if np.isfinite(ms["cadence_days"]) else 0.01
    t_start, t_end = float(t146[ev[0]]), float(t146[ev[-1]])
    colour_cadence_h = float((conf.get("surveys") or {}).get("GBTDS", {})
                             .get("color_cadence_h", 12.0))
    out.update({"event_t_start": t_start, "event_t_end": t_end,
                "event_duration_h": 24.0 * (t_end - t_start + cad146),
                "colour_cadence_h": colour_cadence_h, "colour_band": colour_lc.band})
    cms = to_mag_series(colour_lc, conf)
    if cms is None:
        out["achromatic_test"] = "no_colour_epoch_in_window"
        out["notes"] = ["colour series has too few usable epochs"]
        return out
    tc = cms["t"]
    in_win = (tc >= t_start - cad146) & (tc <= t_end + cad146)
    out["n_colour_epochs_in_event"] = int(in_win.sum())
    if not in_win.any():
        out["achromatic_test"] = "no_colour_epoch_in_window"
        out["notes"] = [f"no {colour_lc.band} epoch inside the event; the colour cadence "
                        f"({colour_cadence_h:g} h) is the shortest event this test reaches"]
        return out
    f146, fe146 = _series_flux(ms)
    fc, fec = _series_flux(cms)
    # F146 at the colour epochs: the primary-band epochs within 1.5 cadences of each.
    near = np.zeros(t146.size, dtype=bool)
    for tcc in tc[in_win]:
        near |= np.abs(t146 - tcc) <= 1.5 * cad146
    if not near.any():
        out["achromatic_test"] = "no_colour_epoch_in_window"
        out["notes"] = ["no F146 epoch coincides with the in-event colour epochs"]
        return out
    a146 = _band_amplitude(t146, f146, fe146, near)
    ac = _band_amplitude(tc, fc, fec, in_win)
    if a146 is None or ac is None or not (a146.testable and ac.testable):
        out["achromatic_test"] = "not_run"
        out["notes"] = ["fractional amplitude untestable in one band"]
        return out
    # greyness_z wants the BLUER band first: F087 is bluer than F146.
    z = greyness_z(ac.a, ac.a_err, a146.a, a146.a_err)
    out.update({"achromatic_test": "ran", "achromatic_z": float(z),
                "a_primary": float(a146.a), "a_primary_err": float(a146.a_err),
                "a_colour": float(ac.a), "a_colour_err": float(ac.a_err),
                "n_primary_epochs_matched": int(near.sum())})
    return out


# --------------------------------------------------------------------------------------
# The paces
# --------------------------------------------------------------------------------------

def _insufficient(channel: str, params: dict, ms: dict | None, why: str) -> dict:
    return _record(channel, "insufficient", None, params, [why],
                   relative_flux=None if ms is None else ms["relative_flux"],
                   n_epochs=0 if ms is None else ms["n_good"])


def pace_dips(lc: LightCurve, conf: dict, colour_lc: LightCurve | None = None) -> dict:
    """Deep aperiodic dips (S10 analogue) with the F146/F087 achromaticity test."""
    p = _paces(conf, "dips")
    params = {"depth_min": float(p.get("depth_min", 0.03)), "k_sigma": float(p.get("k_sigma", 4.0)),
              "min_epochs": int(p.get("min_epochs", 200))}
    ms = to_mag_series(lc, conf)
    if ms is None or ms["n_good"] < params["min_epochs"]:
        return _insufficient("dips", params, ms, "fewer usable epochs than paces.dips.min_epochs")
    res = detect_dips(ms["t"], ms["mag"], ms["magerr"], depth_min=params["depth_min"],
                      k_sigma=params["k_sigma"], min_epochs=params["min_epochs"])
    if res is None:
        return _insufficient("dips", params, ms, "detector returned None")
    extra: dict[str, Any] = {"relative_flux": ms["relative_flux"], "n_epochs": ms["n_good"],
                             "achromatic_test": "not_run", "achromatic_z": None}
    notes: list[str] = []
    if res.n_dip_events >= 1:
        ach = _achromatic_test(ms, colour_lc, conf, "dips", params["depth_min"],
                               params["k_sigma"], merge_gap_d=5.0)
        notes.extend(ach.pop("notes", []))
        extra.update(ach)
    return _record("dips", "ran", res, params, notes, **extra)


def pace_glint(lc: LightCurve, conf: dict, colour_lc: LightCurve | None = None) -> dict:
    """Brief specular brightening (S30) with the same colour test."""
    p = _paces(conf, "glint")
    params = {"bright_min": float(p.get("bright_min", 0.20)), "k_sigma": float(p.get("k_sigma", 6.0)),
              "min_epochs": int(p.get("min_epochs", 200)), "max_events": int(p.get("max_events", 6)),
              "merge_gap_d": float(p.get("merge_gap_d", 0.05))}
    ms = to_mag_series(lc, conf)
    if ms is None:
        return _insufficient("glint", params, ms, "no usable epochs")
    # A glint lasts minutes to hours.  Sampled more coarsely than the longest
    # glint, the series cannot resolve one, and every transient in it would be
    # "a brightening confined to a few epochs" -- which is what happened to
    # 1,595 simulated supernovae at the 5-day HLTDS cadence.
    max_dur = float(p.get("max_event_duration_d", 1.0))
    params["max_event_duration_d"] = max_dur
    params["cadence_days"] = ms.get("cadence_days")
    if ms.get("cadence_days") is not None and float(ms["cadence_days"]) > max_dur:
        return _record("glint", "not_applicable", None, params,
                       [f"median cadence {ms['cadence_days']:.3g} d exceeds the longest glint "
                        f"({max_dur:g} d); the series cannot resolve one"],
                       relative_flux=ms["relative_flux"], n_epochs=ms["n_good"])
    if ms["n_good"] < params["min_epochs"]:
        return _insufficient("glint", params, ms, "fewer usable epochs than paces.glint.min_epochs")
    res = detect_glints(ms["t"], ms["mag"], ms["magerr"], bright_min=params["bright_min"],
                        k_sigma=params["k_sigma"], min_epochs=params["min_epochs"],
                        max_events=params["max_events"], merge_gap_d=params["merge_gap_d"])
    if res is None:
        return _insufficient("glint", params, ms, "detector returned None")
    extra: dict[str, Any] = {"relative_flux": ms["relative_flux"], "n_epochs": ms["n_good"],
                             "achromatic_test": "not_run", "achromatic_z": None}
    notes: list[str] = []
    if res.n_glint_events >= 1:
        ach = _achromatic_test(ms, colour_lc, conf, "glint", params["bright_min"],
                               params["k_sigma"], merge_gap_d=params["merge_gap_d"])
        notes.extend(ach.pop("notes", []))
        extra.update(ach)
    return _record("glint", "ran", res, params, notes, **extra)


def pace_secular(lc: LightCurve, conf: dict, colour_lc: LightCurve | None = None) -> dict:
    """Secular fade across GBTDS seasons (the season unit is found from gaps, never 365.25 d)."""
    p = _paces(conf, "secular")
    gap = float((conf.get("paces") or {}).get("season_gap_days", 20.0))
    params = {"min_epochs": int(p.get("min_epochs", 400)), "min_seasons": int(p.get("min_seasons", 3)),
              "season_days": float(p.get("season_days", 72.0)), "season_gap_days": gap}
    ms = to_mag_series(lc, conf)
    if ms is None or ms["n_good"] < params["min_epochs"]:
        return _insufficient("secular", params, ms, "fewer usable epochs than paces.secular.min_epochs")
    gated = _relative_gate(ms, conf, "secular", params)
    if gated is not None:
        return gated
    labels = seasons_from_gaps(ms["t"], gap)
    n_seasons = int(np.unique(labels).size)
    if n_seasons < params["min_seasons"]:
        return _insufficient("secular", params, ms,
                             f"{n_seasons} gap seasons < paces.secular.min_seasons")
    sb = _season_bin_days(ms["t"], labels, params["season_days"])
    params["season_bin_days_used"] = sb["season_bin_days"]
    res = detect_secular_fade(ms["t"], ms["mag"], ms["magerr"], min_epochs=params["min_epochs"],
                              min_seasons=params["min_seasons"], season_days=sb["season_bin_days"])
    if res is None:
        return _insufficient("secular", params, ms, "detector returned None (too few seasons)")
    notes = []
    if ms["relative_flux"]:
        notes.append("relative flux: season medians compared within this curve only")
    if not sb["season_binning_matches_gaps"]:
        notes.append("detector's fixed-width season bins do not reproduce the gap seasons")
    return _record("secular", "ran", res, params, notes, relative_flux=ms["relative_flux"],
                   n_epochs=ms["n_good"], n_seasons=n_seasons, **sb)


def pace_rust(lc: LightCurve, conf: dict, colour_lc: LightCurve | None = None) -> dict:
    """RUST (S9): does the season scatter grow?  One GBTDS season per block."""
    p = _paces(conf, "rust")
    gap = float((conf.get("paces") or {}).get("season_gap_days", 20.0))
    params = {"season_days": float(p.get("season_days", 72.0)),
              "min_epochs_season": int(p.get("min_epochs_season", 100)),
              "min_seasons": int(p.get("min_seasons", 4)), "season_gap_days": gap}
    ms = to_mag_series(lc, conf)
    need = params["min_epochs_season"] * params["min_seasons"]
    if ms is None or ms["n_good"] < need:
        return _insufficient("rust", params, ms, f"fewer than {need} usable epochs")
    gated = _relative_gate(ms, conf, "rust", params)
    if gated is not None:
        return gated
    labels = seasons_from_gaps(ms["t"], gap)
    n_seasons = int(np.unique(labels).size)
    if n_seasons < params["min_seasons"]:
        return _insufficient("rust", params, ms, f"{n_seasons} gap seasons < paces.rust.min_seasons")
    sb = _season_bin_days(ms["t"], labels, params["season_days"])
    params["season_bin_days_used"] = sb["season_bin_days"]
    res = detect_rust(ms["t"], ms["mag"], ms["magerr"], season_days=sb["season_bin_days"],
                      min_epochs_season=params["min_epochs_season"],
                      min_seasons=params["min_seasons"])
    if res is None:
        return _insufficient("rust", params, ms, "detector returned None (too few usable seasons)")
    notes = []
    if ms["relative_flux"]:
        notes.append("relative flux: season scatter compared within this curve only")
    if not sb["season_binning_matches_gaps"]:
        notes.append("detector's fixed-width season bins do not reproduce the gap seasons")
    return _record("rust", "ran", res, params, notes, relative_flux=ms["relative_flux"],
                   n_epochs=ms["n_good"], n_seasons=n_seasons, **sb)


KNELL_DETECTOR_GAP_DAYS = 90.0     # make_blocks' default; analyze_band does not forward it
_BIN_LADDER_MIN = (0.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 240.0, 480.0)


def _knell_footprint(t: np.ndarray, labels: np.ndarray, cadence_d: float, min_period: float,
                     oversample: float, max_cells: float) -> dict:
    """Time bin (minutes) that keeps the detector's dense GLS grid under ``max_cells``."""
    spans = [float(np.ptp(t[labels == k])) for k in np.unique(labels)]
    T = max(spans) if spans else float(np.ptp(t))
    cad = cadence_d if (np.isfinite(cadence_d) and cadence_d > 0) else 1.0 / 120.0
    # bin_min == 0 means "no binning": the series goes in at its own cadence and
    # the config min_period stands.  Real bins raise min_period to three bins.
    ladder = [0.0] + [b for b in _BIN_LADDER_MIN if b / 1440.0 > cad]
    fp = {}
    for bin_min in ladder:
        bin_d = cad if bin_min == 0 else bin_min / 1440.0
        p_min = float(min_period) if bin_min == 0 else max(float(min_period), 3.0 * bin_d)
        cells = float(oversample) * T * T / (p_min * bin_d)
        fp = {"bin_min": bin_min, "min_period_effective": p_min, "gls_cells": cells}
        if cells <= max_cells:
            break
    return fp


def pace_knell(lc: LightCurve, conf: dict, colour_lc: LightCurve | None = None) -> dict:
    """KNELL (S32): a coherent period in the early seasons, gone in the late ones."""
    p = _paces(conf, "knell")
    gap = float((conf.get("paces") or {}).get("season_gap_days", 20.0))
    params = {"season_days": float(p.get("season_days", 72.0)),
              "min_epochs_block": int(p.get("min_epochs_block", 200)),
              "min_blocks": int(p.get("min_blocks", 4)), "block_mode": "gap",
              "min_period": float(p.get("min_period", 0.02)),
              "max_period": float(p.get("max_period", 30.0)),
              "n_null": int(p.get("n_null", 50)), "n_trials": int(p.get("n_trials", 50)),
              "pdm_null": int(p.get("pdm_null", 100)), "oversample": float(p.get("oversample", 5.0)),
              "fap": float(p.get("fap", 0.01)),
              "measure_efficiency": bool(p.get("measure_efficiency", False)),
              "max_gls_cells": float(p.get("max_gls_cells", 2.0e7)),
              "block_gap_days": KNELL_DETECTOR_GAP_DAYS, "seed": int(p.get("seed", 0))}
    for k in ("min_post_span_days", "min_pre_blocks", "min_post_blocks"):
        if k in p:
            params[k] = p[k]
    ms = to_mag_series(lc, conf)
    need = params["min_epochs_block"] * params["min_blocks"]
    if ms is None or ms["n_good"] < need:
        return _insufficient("knell", params, ms, f"fewer than {need} usable epochs")
    labels = seasons_from_gaps(ms["t"], gap)
    n_seasons = int(np.unique(labels).size)
    if n_seasons < params["min_blocks"]:
        return _insufficient("knell", params, ms, f"{n_seasons} gap seasons < paces.knell.min_blocks")
    notes = []
    n_det_blocks = int(np.unique(seasons_from_gaps(ms["t"], KNELL_DETECTOR_GAP_DAYS)).size)
    if n_det_blocks != n_seasons:
        notes.append(f"detector splits blocks at {KNELL_DETECTOR_GAP_DAYS:g} d gaps "
                     f"({n_det_blocks} blocks) but the paces seasons number {n_seasons}")
    fp = _knell_footprint(ms["t"], labels, ms["cadence_days"], params["min_period"],
                          params["oversample"], params["max_gls_cells"])
    t, m, e = ms["t"], ms["mag"], ms["magerr"]
    if fp["bin_min"] > 0:
        t, m, e = _bin_series(t, m, e, fp["bin_min"] / 1440.0)
        notes.append(f"binned to {fp['bin_min']:g} min and min_period raised to "
                     f"{fp['min_period_effective']:.3g} d to bound the detector's GLS grid")
    params.update({"bin_min": fp["bin_min"], "min_period_effective": fp["min_period_effective"],
                   "n_epochs_to_detector": int(t.size)})
    kw = {k: params[k] for k in ("min_post_span_days", "min_pre_blocks", "min_post_blocks")
          if k in params}
    try:
        res = analyze_band(
            t, m, e, band=lc.band, season_days=params["season_days"],
            min_epochs_block=params["min_epochs_block"], min_blocks=params["min_blocks"],
            block_mode="gap", min_period=fp["min_period_effective"],
            max_period=params["max_period"], oversample=params["oversample"], fap=params["fap"],
            n_null=params["n_null"], n_trials=params["n_trials"], pdm_null=params["pdm_null"],
            measure_efficiency=params["measure_efficiency"], rng=params["seed"], **kw)
    except Exception as exc:  # noqa: BLE001 -- the paces must never raise on one star
        return _record("knell", "error", None, params, notes + [f"{type(exc).__name__}: {exc}"],
                       relative_flux=ms["relative_flux"], n_epochs=ms["n_good"],
                       n_seasons=n_seasons)
    status = "insufficient" if res.status == "insufficient_data" else "ran"
    if status == "insufficient":
        notes.append("detector: too few usable blocks")
    if not params["measure_efficiency"]:
        notes.append("efficiency injection off (paces.knell.measure_efficiency); "
                     "is_cessation cannot be True without it")
    return _record("knell", status, res, params, notes, relative_flux=ms["relative_flux"],
                   n_epochs=ms["n_good"], n_seasons=n_seasons, detector_status=res.status)


def pace_metronome(lc: LightCurve, conf: dict, colour_lc: LightCurve | None = None) -> dict:
    """METRONOME (S28): are the star's own flares on a clock, against its own windows?"""
    p = _paces(conf, "metronome")
    params = {"flare_k_sigma": float(p.get("flare_k_sigma", 5.0)),
              "flare_bright_min": float(p.get("flare_bright_min", 0.10)),
              "min_flares": int(p.get("min_flares", 8)),
              "median_window_epochs": int(p.get("median_window_epochs", 101)),
              "flare_merge_gap_d": float(p.get("flare_merge_gap_d", 0.05)),
              "window_bin_days": float(p.get("window_bin_days", 0.1)),
              "window_min_gap_days": float(p.get("window_min_gap_days", 1.0)),
              "run_nulls": bool(p.get("run_nulls", True)), "seed": int(p.get("seed", 0)),
              "scan": dict(p.get("scan") or {}), "null": dict(p.get("null") or {})}
    ms = to_mag_series(lc, conf)
    if ms is None or ms["n_good"] < 2 * params["median_window_epochs"]:
        return _insufficient("metronome", params, ms, "too few usable epochs for a running median")
    flares = detect_flares(ms["t"], ms["mag"], ms["magerr"], k_sigma=params["flare_k_sigma"],
                           bright_min=params["flare_bright_min"],
                           window_epochs=params["median_window_epochs"],
                           merge_gap_d=params["flare_merge_gap_d"])
    extra = {"relative_flux": ms["relative_flux"], "n_epochs": ms["n_good"],
             "n_flares": int(flares.size), "flare_times": flares[:200]}
    if flares.size < params["min_flares"]:
        return _record("metronome", "insufficient", None, params,
                       [f"{flares.size} flares found < paces.metronome.min_flares"], **extra)
    windows = windows_from_events(ms["t"], bin_days=params["window_bin_days"],
                                  min_gap_days=params["window_min_gap_days"],
                                  cadence_days=ms["cadence_days"], label="roman_lightcurve_epochs")
    scan_conf = dict({"n_min": params["min_flares"]}, **params["scan"])
    try:
        res = analyze_star(flares, windows, scan_conf=scan_conf, null_conf=params["null"],
                           rng=params["seed"], run_nulls=params["run_nulls"])
    except Exception as exc:  # noqa: BLE001
        return _record("metronome", "error", None, params, [f"{type(exc).__name__}: {exc}"], **extra)
    status = "insufficient" if res.get("status") == "insufficient_events" else "ran"
    notes = ["flare times from the crude in-curve finder (detect_flares); the timing test "
             "is the deliverable, the finder is not"]
    return _record("metronome", status, res, params, notes, **extra)


PACES = {"dips": pace_dips, "glint": pace_glint, "secular": pace_secular, "rust": pace_rust,
         "knell": pace_knell, "metronome": pace_metronome}


def _flag_levels(conf: dict) -> dict:
    lv = dict(DEFAULT_FLAG_LEVELS)
    lv.update({k: float(v) for k, v in ((conf.get("paces") or {}).get("flags") or {}).items()})
    return lv


def channel_flagged(name: str, rec: dict, conf: dict) -> bool:
    """Conservative interest test on one channel record; False for anything not ``ran``."""
    if rec.get("status") != "ran" or not isinstance(rec.get("result"), dict):
        return False
    r = rec["result"]
    lv = _flag_levels(conf)

    def num(key, default=float("nan")):
        v = r.get(key, default)
        try:
            return float(v) if v is not None else float("nan")
        except (TypeError, ValueError):
            return float("nan")

    if name == "dips":
        depth_min = float(rec.get("params", {}).get("depth_min", 0.03))
        return (num("n_dip_events", 0) >= 1 and num("max_event_depth") >= depth_min
                and num("score") > 0)
    if name == "glint":
        return num("n_glint_events", 0) >= 1
    if name == "secular":
        return num("slope_mag_yr") > 0 and num("slope_sigma") >= lv["secular_sigma_min"]
    if name == "rust":
        return num("rank_p") < lv["rust_p_max"]
    if name == "knell":
        if "is_cessation" in r:
            return bool(r["is_cessation"])
        return str(r.get("status", "")) == "cessation"
    if name == "metronome":
        pw = num("p_window")
        return math.isfinite(pw) and pw < lv["metronome_p_max"]
    return False


def transient_like(ms: dict | None, conf: dict) -> dict:
    """Does one brightening dominate the series, and is it time-asymmetric?

    A supernova rises in days and declines over weeks; a specular glint, a
    flare or an occultation is what the paces look for on a *star*.  The test
    takes the brightest epoch, grows the excursion outward while the flux
    stays ``k_sigma`` above the median, and calls the curve transient-like
    when that excursion spans at least ``min_excursion_epochs`` and the rise
    lasts less than ``rise_decline_ratio_max`` of the decline.  Reported, not
    applied: the flags on such a curve are counted separately by
    :func:`assess_paces`, since a supernova is not a star with an occulter.
    """
    tc = (conf.get("paces") or {}).get("transient") or {}
    k = float(tc.get("k_sigma", 5.0))
    min_n = int(tc.get("min_excursion_epochs", 5))
    ratio_max = float(tc.get("rise_decline_ratio_max", 0.5))
    if ms is None or ms["n_good"] < min_n + 2:
        return {"transient_like": False, "status": "insufficient"}
    t, mag, err = ms["t"], ms["mag"], ms["magerr"]
    med = float(np.median(mag))
    bright = (med - mag) / np.where(err > 0, err, np.nanmedian(err[err > 0]) if np.any(err > 0) else 0.02)
    i = int(np.argmax(bright))
    if bright[i] < k:
        return {"transient_like": False, "status": "no_significant_brightening"}
    lo = i
    while lo - 1 >= 0 and bright[lo - 1] >= k:
        lo -= 1
    hi = i
    while hi + 1 < bright.size and bright[hi + 1] >= k:
        hi += 1
    n_exc = hi - lo + 1
    rise = float(t[i] - t[lo])
    decline = float(t[hi] - t[i])
    ratio = rise / decline if decline > 0 else (np.inf if rise > 0 else 1.0)
    # The excursion must be interior with a measured rise: a secular fade is
    # "brightest at the start", which is an edge, not a transient.
    interior = lo > 0 and hi < bright.size - 1 and lo < i < hi
    return {"transient_like": bool(interior and n_exc >= min_n and ratio < ratio_max),
            "status": "ok", "n_excursion_epochs": int(n_exc), "rise_days": rise,
            "decline_days": decline, "rise_decline_ratio": float(ratio),
            "peak_sigma": float(bright[i])}


def pace_lightcurve(lc: LightCurve, conf: dict, colour_lc: LightCurve | None = None,
                    channels=None) -> dict:
    """Run every bridged light-curve channel (or ``channels``) on one star, one band."""
    names = list(CHANNELS if channels is None else [c for c in channels if c in PACES])
    funnel = Funnel()
    funnel.bump("lightcurves")
    ms = to_mag_series(lc, conf)
    gap = float((conf.get("paces") or {}).get("season_gap_days", 20.0))
    n_seasons = int(np.unique(seasons_from_gaps(ms["t"], gap)).size) if ms else 0
    out: dict[str, Any] = {
        "star_id": lc.star_id, "band": lc.band, "survey": lc.survey,
        "n_epochs": 0 if ms is None else ms["n_good"],
        "n_dq_rejected": 0 if ms is None else ms["n_dq_rejected"],
        "relative_flux": lc.flux_zp_ab is None, "n_seasons": n_seasons,
        "colour_band": None if colour_lc is None else colour_lc.band,
        "channels": {}, "flags": [],
    }
    tr = transient_like(ms, conf)
    out["transient"] = tr
    if tr.get("transient_like"):
        funnel.bump("transient_like")
    for name in names:
        try:
            rec = PACES[name](lc, conf, colour_lc)
        except Exception as exc:  # noqa: BLE001 -- one channel's failure must not stop the star
            rec = _record(name, "error", None, {}, [f"{type(exc).__name__}: {exc}"])
        out["channels"][name] = rec
        funnel.bump(f"{name}_{rec['status']}")
        if rec["status"] != "ran":
            funnel.reject(f"{name}_{rec['status']}")
        if channel_flagged(name, rec, conf):
            out["flags"].append(name)
            funnel.bump(f"{name}_flagged")
    if lc.flux_zp_ab is None:
        funnel.note("relative flux: product carries no zero point")
    out["funnel"] = funnel.as_dict()
    return json_safe(out)


# --------------------------------------------------------------------------------------
# Spectra
# --------------------------------------------------------------------------------------

def lsf_sigma_pix(spec: Spectrum) -> dict:
    """LSF Gaussian sigma in samples from the resolving power: ``lambda / (R * 2.3548 * dlambda)``.

    Per sample when ``R`` varies (the prism); the finders take one number, so the
    median is what they get and the range is recorded beside it.  With no
    resolving power on the product the finders run at 1 sample and the record
    says the LSF was assumed.
    """
    wl = spec.wavelength_um
    if spec.n < 3:
        return {"lsf_sigma_pix": float("nan"), "lsf_source": "too_few_samples"}
    dl = np.gradient(wl)
    dl = np.where(np.isfinite(dl) & (dl > 0), dl, np.nan)
    if spec.resolving_power is None:
        return {"lsf_sigma_pix": 1.0, "lsf_source": "assumed_1px_no_resolving_power"}
    R = (spec.resolving_power if isinstance(spec.resolving_power, np.ndarray)
         else np.full(wl.shape, float(spec.resolving_power)))
    with np.errstate(invalid="ignore", divide="ignore"):
        s = wl / (R * FWHM_TO_SIGMA * dl)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        return {"lsf_sigma_pix": 1.0, "lsf_source": "assumed_1px_unusable_resolving_power"}
    return {"lsf_sigma_pix": float(np.median(s)), "lsf_sigma_pix_range": [float(s.min()), float(s.max())],
            "lsf_source": "resolving_power"}


def pace_spectrum(spec: Spectrum, conf: dict) -> dict:
    """Narrow emission and absorption lines on one extracted spectrum."""
    p = _paces(conf, "spectra")
    params = {"snr_min": float(p.get("snr_min", 8.0)),
              "continuum_window": int(p.get("continuum_window", 61))}
    out: dict[str, Any] = {"source_id": spec.source_id, "mode": spec.mode, "survey": spec.survey,
                           "n_samples": spec.n, "params": params, "notes": []}
    if spec.n < 5:
        out.update({"status": "not_applicable", "n_emission": 0, "n_absorption": 0,
                    "emission": [], "absorption": []})
        out["notes"].append("fewer than 5 samples")
        return json_safe(out)
    lsf = lsf_sigma_pix(spec)
    out.update(lsf)
    if not math.isfinite(lsf["lsf_sigma_pix"]):
        out.update({"status": "insufficient", "n_emission": 0, "n_absorption": 0,
                    "emission": [], "absorption": []})
        return json_safe(out)
    if spec.resolving_power is None:
        out["notes"].append("no resolving power on the product; LSF assumed at 1 sample")
    good = spec.flux_err.copy()
    if spec.dq is not None:
        good = np.where(spec.dq != 0, np.inf, good)      # masked samples carry no weight
    try:
        em = find_emission_lines(spec.wavelength_um, spec.flux, good,
                                 lsf_sigma_pix=lsf["lsf_sigma_pix"], snr_min=params["snr_min"],
                                 continuum_window=params["continuum_window"])
        ab = find_absorption_lines(spec.wavelength_um, spec.flux, good,
                                   lsf_sigma_pix=lsf["lsf_sigma_pix"], snr_min=params["snr_min"],
                                   continuum_window=params["continuum_window"])
    except Exception as exc:  # noqa: BLE001
        out.update({"status": "error", "n_emission": 0, "n_absorption": 0,
                    "emission": [], "absorption": []})
        out["notes"].append(f"{type(exc).__name__}: {exc}")
        return json_safe(out)
    out.update({"status": "ran", "n_emission": len(em), "n_absorption": len(ab),
                "emission": [ln.as_dict() for ln in em], "absorption": [ln.as_dict() for ln in ab],
                "contam_available": spec.contam is not None,
                "is_point_source": spec.is_point_source})
    if spec.contam is None:
        out["notes"].append("overlap_test_not_run: product carries no contamination estimate")
    return json_safe(out)


# --------------------------------------------------------------------------------------
# Assess
# --------------------------------------------------------------------------------------

def assess_paces(records: list[dict], conf: dict) -> dict:
    """Counts of stars per channel status, the flags, and a verdict that never overclaims.

    ``NO_DATA_REACHED`` for an empty list; ``PACES_RAN_NO_FLAGS`` when every
    channel that ran flagged nothing; ``PACES_FLAGS_PENDING_VET`` otherwise ---
    a flag routes a star to vetting, it is not a candidate.
    """
    per_channel: dict[str, dict[str, int]] = {c: {} for c in CHANNELS}
    flags: dict[str, int] = {}
    flags_on_transients: dict[str, int] = {}
    n_transient_like = 0
    flagged_stars: list[dict] = []
    n_lc = n_spec = 0
    n_relative = 0
    spec_status: dict[str, int] = {}
    n_em = n_ab = 0
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        if "channels" in rec:
            n_lc += 1
            n_relative += int(bool(rec.get("relative_flux")))
            for name, ch in (rec.get("channels") or {}).items():
                st = str((ch or {}).get("status", "error"))
                d = per_channel.setdefault(name, {})
                d[st] = d.get(st, 0) + 1
            is_tr = bool((rec.get("transient") or {}).get("transient_like"))
            if is_tr:
                n_transient_like += 1
            for f in rec.get("flags") or []:
                if is_tr:
                    flags_on_transients[f] = flags_on_transients.get(f, 0) + 1
                else:
                    flags[f] = flags.get(f, 0) + 1
            if rec.get("flags"):
                flagged_stars.append({"star_id": rec.get("star_id"), "band": rec.get("band"),
                                      "flags": list(rec["flags"]), "transient_like": is_tr})
        elif "n_emission" in rec:
            n_spec += 1
            st = str(rec.get("status", "error"))
            spec_status[st] = spec_status.get(st, 0) + 1
            n_em += int(rec.get("n_emission") or 0)
            n_ab += int(rec.get("n_absorption") or 0)
            if rec.get("status") == "ran" and (rec.get("n_emission") or rec.get("n_absorption")):
                flags["spectral_lines"] = flags.get("spectral_lines", 0) + 1
                flagged_stars.append({"star_id": rec.get("source_id"), "band": rec.get("mode"),
                                      "flags": ["spectral_lines"]})
    n_ran = sum(v.get("ran", 0) for v in per_channel.values()) + spec_status.get("ran", 0)
    if n_lc + n_spec == 0:
        verdict = "NO_DATA_REACHED"
    elif flags:
        verdict = "PACES_FLAGS_PENDING_VET"
    else:
        verdict = "PACES_RAN_NO_FLAGS"
    return json_safe({
        "verdict": verdict, "n_lightcurves": n_lc, "n_spectra": n_spec,
        "n_relative_flux": n_relative, "n_channel_runs": n_ran,
        "per_channel_status": per_channel, "spectra_status": spec_status,
        "n_emission_lines": n_em, "n_absorption_lines": n_ab,
        "flags": flags, "flagged": flagged_stars[:500],
        "n_transient_like": n_transient_like,
        "flags_on_transient_like": flags_on_transients,
        "flag_levels": _flag_levels(conf),
    })


# --------------------------------------------------------------------------------------
# Synthesis (tests and selftest only; everything made here is stamped synthetic)
# --------------------------------------------------------------------------------------

def _gbtds_epochs(t0: float, n_seasons: int, season_days: float, cadence_min: float,
                  gap_days: float) -> np.ndarray:
    per = int(round(season_days * 1440.0 / cadence_min))
    step = cadence_min / 1440.0
    parts = [t0 + s * (season_days + gap_days) + step * np.arange(per) for s in range(n_seasons)]
    return np.concatenate(parts)


def _signal_factor(t: np.ndarray, inject: dict | None, t0: float, rng=None) -> np.ndarray:
    """Multiplicative flux factor of an injected signal (1.0 everywhere for none/rust)."""
    f = np.ones_like(t)
    if not inject:
        return f
    kind = inject.get("kind")
    if kind == "dip":
        tc, dur = float(inject["t"]), float(inject.get("dur_d", 2.0))
        f[np.abs(t - tc) <= 0.5 * dur] *= 1.0 - float(inject["depth"])
    elif kind == "glint":
        tc = float(inject["t"])
        i = int(np.argmin(np.abs(t - tc)))
        sel = np.zeros_like(f, dtype=bool)
        sel[max(i - 1, 0):i + 2] = True                # three consecutive epochs
        f[sel] *= 1.0 + float(inject["amp"])
    elif kind == "fade":
        f *= 10.0 ** (-0.4 * float(inject["mag_per_yr"]) * (t - t0) / 365.25)
    elif kind == "periodic_cease":
        amp, per = float(inject.get("amp", 0.05)), float(inject["period_d"])
        on = t < float(inject["cease_at"])
        f[on] *= 10.0 ** (-0.4 * amp * np.sin(2.0 * np.pi * t[on] / per))
    elif kind == "clock_flares":
        per, n, amp = float(inject["period_d"]), int(inject.get("n", 20)), float(inject.get("amp", 0.3))
        tau = float(inject.get("tau_d", 0.02))
        ticks = t0 + 0.37 * per + per * np.arange(int(np.ceil((t.max() - t0) / per)) + 1)
        # keep the ticks that fall in observed time, spread over the whole span
        obs = ticks[np.array([np.min(np.abs(t - tk)) < 0.02 for tk in ticks])]
        if obs.size > n:
            obs = obs[np.linspace(0, obs.size - 1, n).astype(int)]
        for tk in obs:
            after = t >= tk
            f[after] *= 1.0 + amp * np.exp(-(t[after] - tk) / tau)
    elif kind == "rust":
        pass                                          # a noise property; see synthesise
    else:
        raise ValueError(f"unknown injection kind {kind!r}")
    return f


def synthesise_gbtds_lightcurve(star_id: str, n_seasons: int = 3, season_days: float = 72,
                                cadence_min: float = 12, gap_days: float = 110, mag: float = 20.0,
                                noise_mag: float = 0.01, zp_ab: float = 27.7, rng=None,
                                inject: dict | None = None, t0: float = 61500.0,
                                band: str = "F146") -> LightCurve:
    """A GBTDS-like F146 series in e-/s with a zero point, optionally with one injected signal.

    ``inject`` is one of ``None``, ``{"kind": "dip", "depth", "t", "dur_d"}``,
    ``{"kind": "glint", "amp", "t"}``, ``{"kind": "fade", "mag_per_yr"}``,
    ``{"kind": "rust", "scatter_growth"}`` (extra Gaussian scatter, in mag, added
    per season as ``scatter_growth * season_index``), ``{"kind": "periodic_cease",
    "period_d", "amp", "cease_at"}`` or ``{"kind": "clock_flares", "period_d", "n",
    "amp"}``.  Times are MJD from ``t0``; ``meta`` carries the injection so
    :func:`colour_companion` can reproduce the same signal in F087.
    """
    rng = np.random.default_rng(0 if rng is None else rng)
    t = _gbtds_epochs(float(t0), int(n_seasons), float(season_days), float(cadence_min),
                      float(gap_days))
    base = 10.0 ** (-0.4 * (float(mag) - float(zp_ab)))
    model = base * _signal_factor(t, inject, float(t0))
    sigma_frac = float(noise_mag) * MAG_TO_FRAC
    flux = model * (1.0 + sigma_frac * rng.standard_normal(t.size))
    if inject and inject.get("kind") == "rust":
        labels = seasons_from_gaps(t, float(gap_days) / 2.0)
        extra = float(inject.get("scatter_growth", 0.01)) * labels * MAG_TO_FRAC
        flux = flux * (1.0 + extra * rng.standard_normal(t.size))
    err = model * sigma_frac
    return LightCurve(star_id=str(star_id), ra=268.0, dec=-29.0, band=band, mjd=t, flux=flux,
                      flux_err=err, survey="GBTDS", flux_unit="e-/s", flux_zp_ab=float(zp_ab),
                      time_system="TDB", dq=np.zeros(t.size, dtype=np.int64), exposure_s=46.96,
                      meta={"synthetic": True, "inject": dict(inject) if inject else None,
                            "t0": float(t0), "mag": float(mag), "noise_mag": float(noise_mag),
                            "season_days": float(season_days), "gap_days": float(gap_days)})


def colour_companion(lc: LightCurve, rng=None, cadence_h: float = 12.0, band: str = "F087",
                     zp_ab: float = 26.5, noise_mag: float | None = None) -> LightCurve:
    """The F087 colour series of a synthetic F146 curve: same seasons, same signal, 12-h cadence.

    The injected signal is rebuilt from ``lc.meta["inject"]`` so the companion
    carries the *same fractional* signal (achromatic, as an occulter or a mirror
    would be) with its own independent noise.
    """
    rng = np.random.default_rng(1 if rng is None else rng)
    meta = lc.meta or {}
    gap = float(meta.get("gap_days", 110.0))
    labels = seasons_from_gaps(lc.mjd, gap / 2.0)
    step = float(cadence_h) / 24.0
    parts = []
    for k in np.unique(labels):
        tt = lc.mjd[labels == k]
        parts.append(np.arange(float(tt.min()) + 0.25 * step, float(tt.max()), step))
    t = np.concatenate(parts) if parts else np.zeros(0)
    nm = float(meta.get("noise_mag", 0.01)) if noise_mag is None else float(noise_mag)
    base = 10.0 ** (-0.4 * (float(meta.get("mag", 20.0)) + 0.5 - float(zp_ab)))   # redder star
    model = base * _signal_factor(t, meta.get("inject"), float(meta.get("t0", lc.mjd.min())))
    sigma_frac = nm * MAG_TO_FRAC
    flux = model * (1.0 + sigma_frac * rng.standard_normal(t.size))
    return LightCurve(star_id=lc.star_id, ra=lc.ra, dec=lc.dec, band=band, mjd=t, flux=flux,
                      flux_err=model * sigma_frac, survey=lc.survey, flux_unit="e-/s",
                      flux_zp_ab=float(zp_ab), time_system=lc.time_system,
                      dq=np.zeros(t.size, dtype=np.int64), exposure_s=lc.exposure_s,
                      meta={"synthetic": True, "companion_of": lc.band,
                            "inject": meta.get("inject")})


def synthesise_spectrum(source_id: str = "syn-spec", mode: str = "G150", n: int = 1500,
                        wl_range_um=(1.00, 1.93), resolving_power: float = 461.0 * 1.4,
                        snr: float = 30.0, line_um: float | None = 1.0644, line_snr: float = 25.0,
                        rng=None) -> Spectrum:
    """A flat-continuum point-source spectrum with one optional unresolved emission line."""
    rng = np.random.default_rng(2 if rng is None else rng)
    wl = np.linspace(float(wl_range_um[0]), float(wl_range_um[1]), int(n))
    cont = np.full(wl.shape, 100.0)
    err = cont / float(snr)
    flux = cont + err * rng.standard_normal(wl.size)
    if line_um is not None:
        dl = float(wl[1] - wl[0])
        sig_pix = float(line_um) / (float(resolving_power) * FWHM_TO_SIGMA * dl)
        x = (wl - float(line_um)) / (sig_pix * dl)
        # unit-norm kernel S/N == line_snr in the matched filter
        peak = float(line_snr) * err[0] / math.sqrt(math.sqrt(math.pi) * sig_pix)
        flux = flux + peak * np.exp(-0.5 * x * x)
    return Spectrum(source_id=source_id, ra=0.0, dec=0.0, mode=mode, wavelength_um=wl,
                    flux=flux, flux_err=err, survey="HLWAS", flux_unit="arbitrary",
                    resolving_power=float(resolving_power), is_point_source=True,
                    meta={"synthetic": True, "line_um": line_um})


__all__ = [
    "CHANNELS", "PACES", "STATUSES", "assess_paces", "channel_flagged", "colour_companion",
    "detect_flares", "dq_reject_mask", "lsf_sigma_pix", "pace_dips", "pace_glint", "pace_knell",
    "pace_lightcurve", "pace_metronome", "pace_rust", "pace_secular", "pace_spectrum",
    "seasons_from_gaps", "synthesise_gbtds_lightcurve", "synthesise_spectrum", "to_mag_series",
]
