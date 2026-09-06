"""Ephemeris propagation and per-integration phase labels (pure, offline).

Every integration of a JWST time series is labelled by where the planet is:

``in_transit``        fully in front of the star (interior of T14, contacts excluded)
``transit_contact``   inside a transit ingress/egress window (excluded from tests)
``out_transit``       outside transit (and its guard band)
``in_eclipse``        fully behind the star
``eclipse_contact``   inside an eclipse ingress/egress window
``out_eclipse``       outside eclipse (and its guard band)

The eclipse time is propagated from the archive transit ephemeris.  For a
circular orbit it is ``T0 + P/2``; for an eccentric orbit with a known argument
of periastron the first-order offset ``(2/pi) e cos(omega) P`` is applied (Winn
2010).  With an eccentricity above ``ecc_max_assume_circular`` and no omega the
eclipse phase is unknown, and the observation is ``phase_unresolved`` -- it can
never host a candidate, only a constant-line entry.

Timing uncertainty is propagated to the observation epoch,
``sigma_t = sqrt(sigma_T0^2 + (N sigma_P)^2)``, and added to every contact
guard; when it exceeds ``max_timing_uncertainty_frac * T14`` the contacts cannot
be placed and the observation is likewise ``phase_unresolved``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_DEFAULT_PHASE_CFG: dict = {
    "ingress_fraction_default": 0.10,
    "out_guard_fraction": 0.05,
    "ecc_max_assume_circular": 0.05,
    "max_timing_uncertainty_frac": 0.50,
    "header_time_uncertainty_days": 0.006,
    "min_in_eclipse": 8,
    "min_out_eclipse": 16,
    "min_in_transit": 8,
    "min_out_transit": 16,
    "min_baseline_before_ingress": 8,
}


@dataclass
class Ephemeris:
    """Transit ephemeris of one planet (days, BJD_TDB)."""

    name: str
    period: float
    t0: float
    duration: float                       # T14 in days
    period_err: float = 0.0
    t0_err: float = 0.0
    ecc: float | None = None
    omega_deg: float | None = None
    rp_rs: float | None = None
    notes: list[str] = field(default_factory=list)

    def valid(self) -> bool:
        return (np.isfinite(self.period) and self.period > 0
                and np.isfinite(self.t0) and np.isfinite(self.duration)
                and self.duration > 0)


def eclipse_offset_fraction(ecc: float | None, omega_deg: float | None,
                            ecc_max_circular: float = 0.05) -> tuple[float | None, str]:
    """Fraction of the period from mid-transit to mid-eclipse, or None.

    Returns ``(fraction, reason)``: ``0.5`` for a circular (or assumed-circular)
    orbit, the first-order eccentric offset when omega is known, and ``None``
    when the eccentric eclipse phase cannot be placed.
    """
    e = float(ecc) if ecc is not None and np.isfinite(ecc) else None
    if e is None or e <= ecc_max_circular:
        return 0.5, ("circular" if e is not None else "eccentricity_unknown_assumed_circular")
    if omega_deg is None or not np.isfinite(omega_deg):
        return None, "eccentric_omega_unknown"
    # t_ecl - t_tr ~= P/2 * (1 + (4/pi) e cos(omega)); the convention of omega
    # (star vs planet) flips the sign, which is why eccentric cases also carry a
    # widened timing uncertainty (see contact_times).
    frac = 0.5 * (1.0 + (4.0 / np.pi) * e * np.cos(np.radians(omega_deg)))
    return float(frac), "eccentric_first_order"


def timing_uncertainty(eph: Ephemeris, t: float) -> float:
    """Propagated 1-sigma timing uncertainty (days) of an event near epoch ``t``."""
    n = np.round((t - eph.t0) / eph.period)
    s_t0 = float(eph.t0_err) if np.isfinite(eph.t0_err) else 0.0
    s_p = float(eph.period_err) if np.isfinite(eph.period_err) else 0.0
    return float(np.hypot(s_t0, n * s_p))


def ingress_duration(eph: Ephemeris, cfg: dict | None = None) -> float:
    """Ingress/egress duration (days): ``T14 * Rp/Rs`` (b=0), else a default fraction."""
    c = {**_DEFAULT_PHASE_CFG, **(cfg or {})}
    if eph.rp_rs is not None and np.isfinite(eph.rp_rs) and 0 < eph.rp_rs < 0.5:
        return float(eph.duration * eph.rp_rs)
    return float(eph.duration * c["ingress_fraction_default"])


def events_in_window(eph: Ephemeris, t_min: float, t_max: float,
                     kind: str, cfg: dict | None = None) -> list[dict]:
    """Predicted transits or eclipses whose T14 window overlaps ``[t_min, t_max]``.

    Each event carries its mid-time, contact times (t1..t4 with the ingress
    duration), the propagated timing uncertainty and, for eclipses, the reason
    the eclipse phase was or was not placeable.
    """
    c = {**_DEFAULT_PHASE_CFG, **(cfg or {})}
    if not eph.valid():
        return []
    if kind == "transit":
        frac, reason = 0.0, "transit_ephemeris"
    else:
        frac, reason = eclipse_offset_fraction(eph.ecc, eph.omega_deg,
                                               c["ecc_max_assume_circular"])
        if frac is None:
            return []
    tau = ingress_duration(eph, c)
    half = 0.5 * eph.duration
    out: list[dict] = []
    n_lo = int(np.floor((t_min - eph.t0) / eph.period - frac)) - 1
    n_hi = int(np.ceil((t_max - eph.t0) / eph.period - frac)) + 1
    for n in range(n_lo, n_hi + 1):
        mid = eph.t0 + (n + frac) * eph.period
        if mid + half < t_min or mid - half > t_max:
            continue
        sig = timing_uncertainty(eph, mid)
        if reason == "eccentric_first_order":
            # The omega convention is ambiguous across catalogues: widen by the
            # full possible first-order offset so a sign error cannot mislabel.
            sig = float(np.hypot(sig, (2.0 / np.pi) * float(eph.ecc) * eph.period))
        out.append({
            "kind": kind, "epoch": int(n), "mid": float(mid),
            "t1": float(mid - half), "t2": float(mid - half + tau),
            "t3": float(mid + half - tau), "t4": float(mid + half),
            "ingress_duration": float(tau), "timing_sigma": float(sig),
            "phase_reason": reason,
        })
    return out


def label_integrations(times, eph: Ephemeris, cfg: dict | None = None,
                       extra_timing_sigma: float = 0.0) -> dict:
    """Phase labels for every integration mid-time (BJD_TDB, days).

    Returns a dict with boolean arrays ``in_transit``, ``transit_contact``,
    ``out_transit``, ``in_eclipse``, ``eclipse_contact``, ``out_eclipse``, the
    orbital ``phase`` (0 = transit), the event lists, ``coverage`` counts, the
    observation ``phase_class`` (``eclipse``, ``transit``, ``both``,
    ``phase_unresolved``) and a list of ``notes`` explaining any degradation.
    """
    c = {**_DEFAULT_PHASE_CFG, **(cfg or {})}
    t = np.asarray(times, float)
    n = t.size
    res = {k: np.zeros(n, bool) for k in ("in_transit", "transit_contact", "out_transit",
                                          "in_eclipse", "eclipse_contact", "out_eclipse")}
    notes: list[str] = []
    res["phase"] = (((t - eph.t0) / eph.period) % 1.0 if eph.valid()
                    else np.full(n, np.nan))
    if not eph.valid() or n == 0 or not np.all(np.isfinite(t)):
        notes.append("invalid_ephemeris_or_times")
        res.update(transits=[], eclipses=[], phase_class="phase_unresolved",
                   coverage=_coverage(res), notes=notes, timing_sigma=None)
        return res
    t_min, t_max = float(np.nanmin(t)), float(np.nanmax(t))
    transits = events_in_window(eph, t_min, t_max, "transit", c)
    eclipses = events_in_window(eph, t_min, t_max, "eclipse", c)
    if not eclipses:
        frac, reason = eclipse_offset_fraction(eph.ecc, eph.omega_deg,
                                               c["ecc_max_assume_circular"])
        if frac is None:
            notes.append(reason)
    if eph.ecc is None:
        notes.append("eccentricity_unknown_assumed_circular")
    guard = c["out_guard_fraction"] * eph.duration
    sig_max = 0.0
    for ev, key in ((transits, "transit"), (eclipses, "eclipse")):
        for e in ev:
            sig = float(np.hypot(e["timing_sigma"], extra_timing_sigma))
            sig_max = max(sig_max, sig)
            if sig > c["max_timing_uncertainty_frac"] * eph.duration:
                notes.append(f"{key}_timing_uncertainty_{sig:.4f}d_exceeds_limit")
                e["unplaceable"] = True
                continue
            e["unplaceable"] = False
            inside = (t >= e["t2"] + sig) & (t <= e["t3"] - sig)
            contact = (t >= e["t1"] - sig - guard) & (t <= e["t4"] + sig + guard) & ~inside
            res[f"in_{key}"] |= inside
            res[f"{key}_contact"] |= contact
    res["out_transit"] = ~res["in_transit"] & ~res["transit_contact"]
    res["out_eclipse"] = ~res["in_eclipse"] & ~res["eclipse_contact"]
    # Baseline before the first eclipse ingress inside the observation.
    n_before = 0
    for e in eclipses:
        if e.get("unplaceable"):
            continue
        n_before = max(n_before, int(np.count_nonzero(t < e["t1"] - guard)))
    cov = _coverage(res)
    cov["n_baseline_before_eclipse_ingress"] = n_before
    has_ecl = (cov["n_in_eclipse"] >= c["min_in_eclipse"]
               and cov["n_out_eclipse"] >= c["min_out_eclipse"])
    has_tr = (cov["n_in_transit"] >= c["min_in_transit"]
              and cov["n_out_transit"] >= c["min_out_transit"])
    if has_ecl and n_before < c["min_baseline_before_ingress"]:
        notes.append("eclipse_without_pre_ingress_baseline")
        has_ecl = False
    if has_ecl and has_tr:
        cls = "both"
    elif has_ecl:
        cls = "eclipse"
    elif has_tr:
        cls = "transit"
    else:
        cls = "phase_unresolved"
    res.update(transits=transits, eclipses=eclipses, phase_class=cls,
               coverage=cov, notes=notes, timing_sigma=sig_max)
    return res


def _coverage(res: dict) -> dict:
    return {f"n_{k}": int(np.count_nonzero(res[k])) for k in
            ("in_transit", "transit_contact", "out_transit",
             "in_eclipse", "eclipse_contact", "out_eclipse")}


def ephemeris_from_archive_row(row: dict, planet_name: str | None = None) -> Ephemeris:
    """Build an :class:`Ephemeris` from a ``pscomppars`` row (dict-like).

    Column semantics: ``pl_orbper`` [d], ``pl_tranmid`` [BJD_TDB],
    ``pl_trandur`` [hours], ``pl_orbeccen``, ``pl_orblper`` [deg], ``pl_ratror``;
    errors ``pl_orbpererr1`` / ``pl_tranmiderr1``.  Missing values degrade
    explicitly through ``notes``.
    """
    def f(k):
        v = row.get(k)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    notes = []
    per, t0, dur_h = f("pl_orbper"), f("pl_tranmid"), f("pl_trandur")
    if per is None or t0 is None:
        notes.append("missing_period_or_t0")
    if dur_h is None:
        notes.append("missing_duration")
    p_err, t0_err = f("pl_orbpererr1"), f("pl_tranmiderr1")
    if p_err is None:
        notes.append("period_err_missing")
    if t0_err is None:
        notes.append("t0_err_missing")
    return Ephemeris(
        name=str(planet_name or row.get("pl_name", "?")),
        period=per if per is not None else np.nan,
        t0=t0 if t0 is not None else np.nan,
        duration=(dur_h / 24.0) if dur_h is not None else np.nan,
        period_err=abs(p_err) if p_err is not None else 0.0,
        t0_err=abs(t0_err) if t0_err is not None else 0.0,
        ecc=f("pl_orbeccen"), omega_deg=f("pl_orblper"), rp_rs=f("pl_ratror"),
        notes=notes)


__all__ = ["Ephemeris", "eclipse_offset_fraction", "timing_uncertainty",
           "ingress_duration", "events_in_window", "label_integrations",
           "ephemeris_from_archive_row"]
