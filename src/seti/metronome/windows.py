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
    rate = len(t) / max(occ.sum() * bin_days, bin_days)      # events per day, occupied time
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


__all__ = ["KEPLER_LC_CADENCE_DAYS", "KEPLER_QUARTERS_BKJD", "TESS_2MIN_CADENCE_DAYS",
           "TESS_ORBIT_DAYS", "TESS_SECTOR_DAYS", "Windows", "guess_time_system",
           "kepler_quarter_windows", "star_windows", "tess_sector_windows",
           "windows_from_events"]
