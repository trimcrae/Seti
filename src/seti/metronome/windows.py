"""Observing-window model for METRONOME.

The null hypothesis of this channel is "the events are placed at random in the
time the star was actually being watched".  Everything about the mission
calendar --- Kepler's quarterly rolls and monthly downlinks, TESS's 27-day
sectors and orbit-perigee gaps, the 30-min or 2-min cadence quantisation of a
catalogued peak time --- therefore has to live *inside* the null, or it will be
discovered as a signal.  This module builds the windows.

Two routes, and the second is preferred:

1. **Published mission boundaries** (:func:`kepler_quarter_windows`): the
   Kepler quarter start/stop epochs in BKJD, transcribed approximately from
   the Kepler Data Characteristics Handbook.  Kept as a fallback and for tests.
2. **Data-driven windows** (:func:`windows_from_events`): bin *every* event in
   the catalogue in time; runs of empty bins longer than ``min_gap_days`` are
   gaps, everything else is observed.  With ~10^5 catalogued flares over a
   mission, an empty day is a real gap.  This route needs no transcription and
   is right for whichever sectors a TESS catalogue actually covers.

Per star (:func:`star_windows`), the mission windows are restricted to the
star's own event span and windows in which the star has *no* events but would
have been expected to show ``>= drop_expected`` at its mean rate are dropped as
"presumed unobserved" (a Kepler module failure, a TESS target not on silicon
that sector).  That is an approximation, stated in ``docs/metronome.md``: it
slightly weakens the evidence against a clock that fell silent for a window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Kepler quarter boundaries, BKJD = BJD - 2454833, APPROXIMATE (rounded to the
# day from the Data Characteristics Handbook quarter table).  Q0 was the
# commissioning quarter; Q17 was truncated by the second reaction-wheel failure.
# Superseded at runtime by the data-driven windows whenever the catalogue has
# enough events to define them.
KEPLER_QUARTERS_BKJD: tuple[tuple[int, float, float], ...] = (
    (0, 120.5, 130.3), (1, 131.5, 165.0), (2, 169.5, 258.5), (3, 260.2, 349.5),
    (4, 352.4, 442.2), (5, 443.5, 538.2), (6, 539.5, 629.3), (7, 630.2, 719.6),
    (8, 735.4, 802.4), (9, 808.5, 906.9), (10, 906.9, 1000.3), (11, 1001.2, 1098.4),
    (12, 1099.4, 1182.1), (13, 1182.8, 1273.1), (14, 1274.1, 1371.4),
    (15, 1373.5, 1471.2), (16, 1472.1, 1558.0), (17, 1559.2, 1591.0),
)

KEPLER_LC_CADENCE_DAYS = 29.4244 / 1440.0          # long cadence, 29.42 min
TESS_2MIN_CADENCE_DAYS = 2.0 / 1440.0
TESS_SECTOR_DAYS = 27.4
TESS_ORBIT_DAYS = 13.7


@dataclass
class Windows:
    """A union of disjoint observed intervals plus the sampling cadence."""

    starts: np.ndarray
    stops: np.ndarray
    cadence_days: float = KEPLER_LC_CADENCE_DAYS
    label: str = ""
    # Reference epoch of the cadence grid.  Catalogued Kepler peak times are
    # already quantised to the long-cadence grid; the null has to be too.
    t_ref: float = field(default=0.0)

    def __post_init__(self):
        s = np.asarray(self.starts, dtype=float)
        e = np.asarray(self.stops, dtype=float)
        order = np.argsort(s)
        s, e = s[order], e[order]
        keep = e > s
        self.starts, self.stops = s[keep], e[keep]
        if self.t_ref == 0.0 and len(self.starts):
            self.t_ref = float(self.starts[0])

    # -- geometry ---------------------------------------------------------
    @property
    def n(self) -> int:
        return int(len(self.starts))

    @property
    def lengths(self) -> np.ndarray:
        return self.stops - self.starts

    @property
    def total(self) -> float:
        return float(self.lengths.sum()) if self.n else 0.0

    @property
    def span(self) -> float:
        return float(self.stops[-1] - self.starts[0]) if self.n else 0.0

    def contains(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if not self.n:
            return np.zeros(len(t), dtype=bool)
        i = np.searchsorted(self.starts, t, side="right") - 1
        ok = i >= 0
        out = np.zeros(len(t), dtype=bool)
        out[ok] = t[ok] <= self.stops[i[ok]]
        return out

    def overlaps(self, lo, hi) -> np.ndarray:
        """Does each interval ``[lo, hi]`` intersect the observed union?

        ``contains`` answers for an instant; a clock tick is not an instant
        once it carries a phase window, and a tick whose instant falls in a
        gap can still have had most of its window observed.
        """
        lo = np.atleast_1d(np.asarray(lo, dtype=float))
        hi = np.atleast_1d(np.asarray(hi, dtype=float))
        if not self.n:
            return np.zeros(len(lo), dtype=bool)
        # windows are disjoint and sorted, so the last one starting at or
        # before ``hi`` has the largest stop of any candidate: if its stop is
        # below ``lo`` every earlier stop is too.
        i = np.searchsorted(self.starts, hi, side="right") - 1
        ok = i >= 0
        out = np.zeros(len(lo), dtype=bool)
        out[ok] = self.stops[i[ok]] >= lo[ok]
        return out

    def window_index(self, t) -> np.ndarray:
        """Index of the window containing each time; -1 outside every window."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if not self.n:
            return np.full(len(t), -1, dtype=int)
        i = np.searchsorted(self.starts, t, side="right") - 1
        inside = (i >= 0) & (t <= self.stops[np.clip(i, 0, self.n - 1)])
        i = np.where(inside, i, -1)
        return i

    # -- observed-time coordinates -----------------------------------------
    def observed_time(self, t) -> np.ndarray:
        """Map real time to cumulative *observed* time (gaps removed).

        Times inside a gap map to the end of the preceding window so the map is
        monotone; the shuffle null uses these coordinates so a permuted waiting
        time can never land inside a gap.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        off = np.concatenate([[0.0], np.cumsum(self.lengths)[:-1]]) if self.n else np.zeros(0)
        i = np.clip(np.searchsorted(self.starts, t, side="right") - 1, 0, max(self.n - 1, 0))
        if not self.n:
            return t.copy()
        local = np.clip(t - self.starts[i], 0.0, self.lengths[i])
        return off[i] + local

    def real_time(self, tau) -> np.ndarray:
        """Inverse of :meth:`observed_time`."""
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        if not self.n:
            return tau.copy()
        edges = np.concatenate([[0.0], np.cumsum(self.lengths)])
        i = np.clip(np.searchsorted(edges, tau, side="right") - 1, 0, self.n - 1)
        return self.starts[i] + (tau - edges[i])

    # -- sampling ------------------------------------------------------------
    def quantize(self, t) -> np.ndarray:
        """Snap times to the cadence grid and keep them inside their window."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if not np.isfinite(self.cadence_days) or self.cadence_days <= 0:
            return t
        q = self.t_ref + np.round((t - self.t_ref) / self.cadence_days) * self.cadence_days
        i = self.window_index(t)
        ok = i >= 0
        q[ok] = np.clip(q[ok], self.starts[i[ok]], self.stops[i[ok]])
        return q

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` times uniform in observed time, snapped to the cadence grid."""
        if not self.n or n <= 0:
            return np.zeros(0)
        tau = rng.uniform(0.0, self.total, size=int(n))
        return np.sort(self.quantize(self.real_time(tau)))

    def as_dict(self) -> dict:
        return {"n_windows": self.n, "total_days": round(self.total, 3),
                "span_days": round(self.span, 3),
                "cadence_days": float(self.cadence_days), "label": self.label,
                "starts": [round(float(x), 3) for x in self.starts],
                "stops": [round(float(x), 3) for x in self.stops]}


def kepler_quarter_windows(cadence_days: float = KEPLER_LC_CADENCE_DAYS) -> Windows:
    """The published (approximate) Kepler quarter boundaries as windows, BKJD."""
    return Windows(np.array([q[1] for q in KEPLER_QUARTERS_BKJD]),
                   np.array([q[2] for q in KEPLER_QUARTERS_BKJD]),
                   cadence_days=cadence_days, label="kepler_quarters_published")


def tess_sector_windows(first_start: float, n_sectors: int,
                        sector_days: float = TESS_SECTOR_DAYS,
                        orbit_gap_days: float = 1.0,
                        cadence_days: float = TESS_2MIN_CADENCE_DAYS) -> Windows:
    """Idealised consecutive TESS sectors: two orbits each, a perigee gap between.

    Real sector boundaries drift by a day or two; this is the *synthetic* window
    model for tests.  Real runs use :func:`windows_from_events`.
    """
    starts, stops = [], []
    t = float(first_start)
    half = (sector_days - orbit_gap_days) / 2.0
    for _ in range(int(n_sectors)):
        starts += [t, t + half + orbit_gap_days]
        stops += [t + half, t + sector_days]
        t += sector_days + 0.7           # ~a day of downlink between sectors
    return Windows(np.array(starts), np.array(stops), cadence_days=cadence_days,
                   label="tess_sectors_synthetic")


def windows_from_events(all_times, *, bin_days: float = 0.1, min_gap_days: float = 0.5,
                        min_expected_in_gap: float = 20.0,
                        cadence_days: float = KEPLER_LC_CADENCE_DAYS,
                        label: str = "data_driven") -> Windows:
    """Observing windows from the density of *every* catalogued event.

    Bins of ``bin_days``; a run of empty bins is a gap when it is at least
    ``min_gap_days`` long **and** the catalogue's mean event rate predicts at
    least ``min_expected_in_gap`` events inside it (so an empty run is a
    ``e^-20`` Poisson fluke at worst, never a low-rate stretch).  Shorter or
    less-populated empty runs are bridged.  Kepler's inter-quarter gaps are
    ~1-3 days and its monthly downlinks ~1 day; a mission-scale catalogue at
    ~10^2 events/day resolves both.  A sparse catalogue (a few events/day)
    resolves only its long gaps, and its label says so through the coarser
    ``min_gap`` recorded in the window model.
    """
    t = np.asarray(all_times, dtype=float)
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return Windows(np.zeros(0), np.zeros(0), cadence_days=cadence_days, label=label)
    t0 = np.floor(t.min() / bin_days) * bin_days
    nb = int(np.ceil((t.max() - t0) / bin_days)) + 1
    occ = np.bincount(((t - t0) / bin_days).astype(int), minlength=nb) > 0
    # Events per day over the whole SPAN, gaps included: a lower bound on the
    # true rate, so the gap length it demands is an upper bound and an empty
    # run is only ever called a gap when even the conservative rate predicts
    # >= min_expected_in_gap events in it.  Measured over occupied bins
    # instead, a sparse catalogue (60 events over two TESS sectors) rated
    # itself at ~10/day and cut every two-day lull inside a sector into a
    # "gap"; for a mission-scale catalogue the two definitions agree.
    rate = len(t) / max(float(t.max() - t.min()), bin_days)
    gap_days_needed = max(float(min_gap_days), float(min_expected_in_gap) / max(rate, 1e-9))
    min_run = max(1, int(np.ceil(gap_days_needed / bin_days)))
    label = f"{label}|min_gap={gap_days_needed:.2f}d"
    starts, stops = [], []
    i = 0
    while i < nb:
        if not occ[i]:
            i += 1
            continue
        j = i
        while j < nb:
            if occ[j]:
                j += 1
                continue
            k = j
            while k < nb and not occ[k]:
                k += 1
            if k - j >= min_run:
                break
            j = k                       # bridge a short empty run
        starts.append(t0 + i * bin_days)
        stops.append(t0 + j * bin_days)
        i = j
    return Windows(np.array(starts), np.array(stops), cadence_days=cadence_days,
                   label=label)


def windows_from_sectors(all_times, sectors, *, cadence_days: float = KEPLER_LC_CADENCE_DAYS,
                         pad_days: float = 0.0, label: str = "sector_spans") -> Windows:
    """One window per catalogued sector / quarter: the span of its events.

    A flare catalogue that records the sector (TESS) or quarter (Kepler) of
    every event states, per sector, the earliest and latest time at which
    *anything* was seen; the union of those spans is the observed time to
    within the first and last event of each sector.  It cannot resolve the
    gaps INSIDE a sector (the TESS perigee gap, a Kepler monthly downlink),
    which is what :func:`windows_from_events` is for; the two are intersected
    by :func:`intersect_windows`.
    """
    t = np.asarray(all_times, dtype=float)
    s = np.asarray(sectors)
    ok = np.isfinite(t)
    try:
        sf = np.asarray(s, dtype=float)
        ok &= np.isfinite(sf)
        s = sf
    except (TypeError, ValueError):
        pass
    t, s = t[ok], s[ok]
    if not len(t):
        return Windows(np.zeros(0), np.zeros(0), cadence_days=cadence_days, label=label)
    starts, stops = [], []
    for val in np.unique(s):
        sel = s == val
        starts.append(float(t[sel].min()) - float(pad_days))
        stops.append(float(t[sel].max()) + float(pad_days))
    return Windows(np.array(starts), np.array(stops), cadence_days=cadence_days, label=label)


def intersect_windows(a: Windows, b: Windows, label: str | None = None) -> Windows:
    """The observed time common to two window models."""
    starts, stops = [], []
    j = 0
    for s0, e0 in zip(a.starts, a.stops, strict=False):
        while j < b.n and b.stops[j] <= s0:
            j += 1
        k = j
        while k < b.n and b.starts[k] < e0:
            lo, hi = max(s0, float(b.starts[k])), min(e0, float(b.stops[k]))
            if hi > lo:
                starts.append(lo)
                stops.append(hi)
            k += 1
    return Windows(np.array(starts), np.array(stops), cadence_days=a.cadence_days,
                   t_ref=a.t_ref, label=label if label is not None else f"{a.label}&{b.label}")


def star_windows(times, mission: Windows, *, pad_days: float = 0.5,
                 drop_expected: float = 5.0) -> Windows:
    """Restrict the mission windows to one star.

    * clipped to ``[t_first - pad, t_last + pad]`` --- a star whose catalogued
      events start in Q3 was, as far as this channel can know, not searched
      before Q3;
    * a window with no events, where the star's mean rate over its *other*
      windows predicts ``>= drop_expected`` events, is dropped as presumed
      unobserved.  Documented approximation (see module docstring).
    """
    t = np.sort(np.asarray(times, dtype=float))
    if not len(t) or not mission.n:
        return Windows(np.zeros(0), np.zeros(0), cadence_days=mission.cadence_days,
                       label=mission.label + "|empty")
    lo, hi = t[0] - pad_days, t[-1] + pad_days
    s = np.clip(mission.starts, lo, hi)
    e = np.clip(mission.stops, lo, hi)
    keep = e > s
    s, e = s[keep], e[keep]
    if not len(s):
        return Windows(np.array([lo]), np.array([hi]), cadence_days=mission.cadence_days,
                       label=mission.label + "|span_only")
    w = Windows(s, e, cadence_days=mission.cadence_days, t_ref=mission.t_ref,
                label=mission.label + "|star")
    idx = w.window_index(t)
    counts = np.bincount(idx[idx >= 0], minlength=w.n)
    occupied = counts > 0
    if occupied.sum() == 0:
        return w
    rate = counts[occupied].sum() / max(w.lengths[occupied].sum(), 1e-9)
    expected = rate * w.lengths
    drop = (~occupied) & (expected >= float(drop_expected))
    if drop.any():
        w = Windows(w.starts[~drop], w.stops[~drop], cadence_days=mission.cadence_days,
                    t_ref=mission.t_ref, label=w.label + f"|dropped{int(drop.sum())}")
    return w


def guess_time_system(times, mission: str = "") -> str:
    """Name the time system a catalogue's peak times are most plausibly in.

    Never trusted silently: the probe stage records the guess and the median
    value beside it, so a wrong guess is visible in the artefact.  The BKJD
    (Kepler, 120-1591) and BTJD (TESS, >= 1325) ranges overlap, so the mission
    hint breaks the tie where the numbers alone cannot.
    """
    t = np.asarray(times, dtype=float)
    t = t[np.isfinite(t)]
    if not len(t):
        return "unknown"
    med = float(np.median(t))
    if 2.4e6 < med < 2.5e6:
        return "BJD"
    if 5.0e4 < med < 7.0e4:
        return "MJD"
    m = str(mission).lower()
    if m.startswith("tess") and 1300.0 < med < 6000.0:
        return "BTJD"          # BJD - 2457000 (TESS)
    if m.startswith("kep") and 100.0 < med < 1700.0:
        return "BKJD"          # BJD - 2454833 (Kepler)
    if 100.0 < med < 1300.0:
        return "BKJD"
    if 1300.0 < med < 6000.0:
        return "BTJD"
    return "unknown"


# ---------------------------------------------------------------------------
# The catalogue's OWN time lattice
# ---------------------------------------------------------------------------
def infer_time_grid(times, *, tol_frac: float = 0.12, min_on_grid: float = 0.97,
                    min_unique: int = 40, max_divisor: int = 16,
                    max_steps: float = 5.0e7) -> dict:
    """The spacing of the lattice a catalogue's event times actually lie on.

    A published peak time is not a real number: it is the time stamp of a
    cadence, so every time in a catalogue sits on a grid whose spacing is that
    catalogue's effective cadence.  The spacing is *not* reliably the mission
    cadence.  Tu+2022's TESS peak times are spaced by 0.0069 d (9.94 min), five
    times the 2-min cadence the mission-cadence default assumes; Guenther+2020
    is on the 2-min grid.  The difference is not cosmetic: the window null
    snaps its draws to ``Windows.cadence_days``, so a null quantised finer than
    the data is a null that cannot reproduce the data's own lattice, and every
    period commensurate with that lattice then looks significant.

    The measurement is deliberately assumption-free: candidate spacings are
    built from the *smallest observed gaps* between distinct times (and their
    integer divisors, in case no pair of events ever landed on adjacent
    cadences), and the largest candidate on which at least ``min_on_grid`` of
    the distinct times fall within ``tol_frac`` of an integer step wins.  It is
    then refined by least squares through the origin.

    Returns a dict --- ``grid_days`` is NaN when no lattice is detectable
    (times genuinely continuous, or barycentric corrections smearing one), and
    that is reported rather than papered over.
    """
    t = np.asarray(times, dtype=float)
    t = np.unique(t[np.isfinite(t)])
    out = {"grid_days": float("nan"), "t0": float("nan"), "frac_on_grid": float("nan"),
           "resid_rms_days": float("nan"), "n_unique": int(len(t)), "method": "none"}
    if len(t) < int(min_unique):
        out["method"] = "too_few_times"
        return out
    d = np.diff(t)
    d = d[d > 0]
    if not len(d):
        out["method"] = "no_distinct_times"
        return out
    # Candidates: the smallest distinct gaps (a gap is an integer number of
    # steps, so the smallest few are the best estimates of the step) and their
    # divisors.  Rounded to 1e-9 d to collapse float noise.
    small = np.unique(np.round(np.sort(d)[: max(50, len(d) // 200)], 9))
    cands: set[float] = set()
    for c in small[:50]:
        for m in range(1, int(max_divisor) + 1):
            g = float(c) / m
            if g > 1e-9:
                cands.add(round(g, 12))
    span = float(t[-1] - t[0])
    t0 = float(t[0])
    best: tuple[float, float, float] | None = None
    for g in sorted(cands, reverse=True):
        if span / g > float(max_steps):
            continue
        r = (t - t0) / g
        res = np.abs(r - np.round(r))
        frac = float(np.mean(res <= float(tol_frac)))
        if frac >= float(min_on_grid):
            best = (g, frac, float(np.sqrt(np.mean((res * g) ** 2))))
            break
    if best is None:
        out["method"] = "no_lattice"
        return out
    g, frac, rms = best
    k = np.round((t - t0) / g)
    denom = float(np.sum(k * k))
    if denom > 0:
        g = float(np.sum(k * (t - t0)) / denom)
        r = (t - t0) / g
        res = np.abs(r - np.round(r))
        frac = float(np.mean(res <= float(tol_frac)))
        rms = float(np.sqrt(np.mean((res * g) ** 2)))
    out.update({"grid_days": float(g), "t0": t0, "frac_on_grid": frac,
                "resid_rms_days": rms, "method": "lattice"})
    return out


def lattice_phase_limits(period: float, grid_days: float, *, phase_window: float = 0.05) -> dict:
    """What the time lattice alone can contribute to a phase concentration.

    A catalogue that rounds its peak times to a lattice of spacing ``g`` cannot
    place an event's phase more precisely than one lattice step, so at period
    ``P`` the phases live on a comb of ``P/g`` teeth spaced ``g/P`` cycles
    apart, and *any* clock --- however perfect --- shows an rms phase jitter of
    at least ``g / (P sqrt(12))`` from the rounding alone.

    Two numbers come out of that, and they are the honest statement of what
    quantisation can and cannot do:

    ``jitter_floor``  the rms phase jitter forced by the rounding.  A star
                      whose measured jitter sits at this floor is as tight as
                      its time stamps allow and no tighter; the tightness is
                      then a property of the catalogue's rounding, not
                      evidence about the star, and it is flagged.
    ``phase_spacing`` ``g/P``.  When it exceeds the phase window used for
                      ``f_in_window``, the fraction of the comb that fits
                      inside the window is quantised and systematically above
                      the window width, so ``f_in_window`` is inflated.  For
                      the catalogues in hand this bites only at the short-period
                      end of the scan (TESS, g = 0.0069 d, P < 0.14 d).

    Note what is NOT claimed: a low-denominator rational P/g is *not* by itself
    a problem.  With P/g = a/b in lowest terms the comb has ``a`` teeth, not
    ``b``, so the resonances that matter are small ``a`` --- short periods ---
    and nothing else.
    """
    out = {"n_teeth": float("nan"), "phase_spacing": float("nan"),
           "jitter_floor": float("nan"), "comb_coarser_than_window": False}
    p, g = float(period), float(grid_days)
    if not (np.isfinite(p) and np.isfinite(g) and p > 0 and g > 0):
        return out
    spacing = g / p
    out.update({"n_teeth": p / g, "phase_spacing": spacing,
                "jitter_floor": spacing / np.sqrt(12.0),
                "comb_coarser_than_window": bool(spacing > float(phase_window))})
    return out


__all__ = ["KEPLER_LC_CADENCE_DAYS", "KEPLER_QUARTERS_BKJD", "TESS_2MIN_CADENCE_DAYS",
           "TESS_ORBIT_DAYS", "TESS_SECTOR_DAYS", "Windows", "guess_time_system",
           "infer_time_grid", "intersect_windows", "kepler_quarter_windows",
           "lattice_phase_limits", "star_windows", "tess_sector_windows",
           "windows_from_events", "windows_from_sectors"]
