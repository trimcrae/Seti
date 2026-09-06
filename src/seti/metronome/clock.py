"""The clock detector --- pure functions, offline-testable.

Given the catalogued peak times of one star's flares and the windows in which
that star was observed, the question is: **is there a period P at which the
events fall at one phase, with a jitter far below anything rotation produces?**

Statistic
---------
For each trial frequency ``f`` the phases ``phi_j = 2*pi*frac(f*t_j)`` are
tested for concentration with the H-test of de Jager, Raubenheimer &
Swanepoel (1989): ``H = max_m (Z2_m - 4m + 4)`` over ``m = 1..m_max`` harmonics,
where ``Z2_m = (2/N) sum_{k<=m} |sum_j exp(i*k*phi_j)|^2`` is the Rayleigh
statistic summed over harmonics.  A strict clock is a delta function in phase
and puts power in every harmonic; rotational modulation of flare visibility
(``rate ∝ 1 + cos(phi)``) is a pure fundamental.  The H-test is the standard
choice for exactly this reason in pulsar timing, and ``m_max`` is a config
value, not a magic number.

At the best period the *clock quality* is then read off the phase distribution
rather than off the periodogram height:

* ``Q = 1 - s / s_0`` where ``s = sqrt(2 (1 - Rbar))`` is the angular deviation
  of the phases (Rbar the mean resultant length) and ``s_0`` its expectation
  for N uniform phases.  ``Q = 1`` is a perfect clock, ``Q = 0`` is noise;
* ``jitter`` --- the rms residual of the event times from the strict clock
  ``t0 + k P``, in units of P.  Rotational modulation gives ~0.15-0.3;
  a 30-min-cadence clock at P = 1 d cannot do better than ~0.006;
* ``f_in_window`` --- the fraction of events within ``±phase_window`` of the
  best phase;
* ``gap_integer_frac`` --- the fraction of consecutive same-window waiting
  times that are an integer number of periods to within ``gap_tol``.  This is
  the property a clock has and *nothing else does*: it is invariant under the
  waiting-time shuffle, which is what makes the second null interpretable.

Nulls
-----
1. **Window-resampled** (:func:`window_null`): N times uniform in the star's
   own observed time, snapped to the mission cadence, scanned identically.  The
   p-value is the sequential Monte-Carlo p of Besag & Clifford (1991): trials
   continue until ``h_stop`` exceedances or ``n_max`` trials.  A star whose
   observed H is ordinary stops after ~20 trials; a star that never gets
   exceeded runs the full budget and its p is quoted at the resolution floor
   with a Gumbel extrapolation from the null's own tail, **flagged as such**.
2. **Waiting-time shuffle** (:func:`shuffle_null`): the star's own waiting
   times, permuted in observed-time coordinates.  Burstiness is preserved;
   long-range phase order is destroyed.  A strict clock survives this null
   (its waiting times *are* the signal, so ``p_shuffle`` ~ 1) --- which is why
   the shuffle p is not a candidate criterion by itself but the joint test
   ``p_shuffle`` large AND ``gap_integer_frac`` small is the ``bursty_random``
   rejection: coherence explained by the waiting-time distribution without the
   waiting times being clock-like.

Cross-star coincidence
----------------------
Event epochs shared by many stars are spacecraft, not sky (Kepler momentum
dumps and Argabrightenings, TESS scattered-light and pointing excursions).
:func:`cross_star_coincidence` removes them from every star *before* any star
is scanned, and counts what it removed.
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import asdict, dataclass, field

import numpy as np

from .windows import Windows

# ---------------------------------------------------------------------------
# Defaults (config/metronome.yaml overrides every one of them)
# ---------------------------------------------------------------------------
DEFAULT_SCAN = {
    "n_min": 8, "min_period_days": 0.2, "max_period_days": 400.0,
    "span_fraction_max": 1.0 / 3.0, "n_cadences_min": 10, "oversample": 5.0,
    "m_max": 4, "chunk": 8192, "refine_points": 41, "refine_halfwidth_steps": 2.0,
    "phase_window": 0.05, "gap_tol": 0.05, "harmonic_walk_tol": 0.02,
    "harmonic_walk_max": 12, "decluster_gap_days": 0.1,
}
DEFAULT_NULL = {
    "h_stop": 10, "n_max": 2000, "n_min_trials": 20, "budget_s": 300.0,
    "screen_rate": 0.3, "screen_p_max": 0.5, "n_shuffle": 100,
    "shuffle_min_gaps": 4,
    # A star whose phase concentration already fails the loose (watch) clock
    # thresholds can never rank above tier `none`, whatever its p; its null is
    # capped here so a strongly rotation-modulated star does not spend the
    # full budget establishing a p-value that cannot change its tier.  The p
    # is still a valid Besag-Clifford p, only coarser.
    "n_max_not_clock": 200, "Q_watch": 0.6, "jitter_watch": 0.12,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def decluster(times, energies=None, gap_days: float = 0.1):
    """Merge events closer than ``gap_days`` into one (keep the first).

    Complex / sympathetic flares are catalogued as several events minutes to
    an hour apart.  They are one energy release, and leaving them in would let
    a burst of three masquerade as three coherent ticks.
    """
    t = np.asarray(times, dtype=float)
    order = np.argsort(t)
    t = t[order]
    e = None if energies is None else np.asarray(energies, dtype=float)[order]
    if len(t) == 0:
        return t, e, 0
    keep = np.ones(len(t), dtype=bool)
    last = t[0]
    for i in range(1, len(t)):
        if t[i] - last < gap_days:
            keep[i] = False
        else:
            last = t[i]
    return t[keep], (None if e is None else e[keep]), int((~keep).sum())


def frequency_grid(span_days: float, *, min_period: float, max_period: float,
                   oversample: float = 5.0, span_fraction_max: float = 1.0 / 3.0
                   ) -> np.ndarray:
    """Uniform frequency grid whose step resolves one cycle over the span.

    The longest trial period is capped at ``span * span_fraction_max``: fewer
    than three cycles is a trend, not a clock.
    """
    span = float(span_days)
    if not np.isfinite(span) or span <= 0:
        return np.zeros(0)
    p_max = min(float(max_period), span * float(span_fraction_max))
    p_min = float(min_period)
    if p_max <= p_min:
        return np.zeros(0)
    df = 1.0 / (float(oversample) * span)
    f_min, f_max = 1.0 / p_max, 1.0 / p_min
    n = int(np.floor((f_max - f_min) / df)) + 1
    return f_min + df * np.arange(n)


def h_statistic(times, freqs, *, m_max: int = 4, chunk: int = 8192) -> np.ndarray:
    """H-test statistic at every trial frequency (vectorised, chunked in f)."""
    t = np.asarray(times, dtype=float)
    f = np.asarray(freqs, dtype=float)
    n = len(t)
    out = np.empty(len(f))
    if n == 0 or len(f) == 0:
        out[:] = 0.0
        return out
    tcol = t[:, None]
    for i in range(0, len(f), int(chunk)):
        fi = f[i:i + int(chunk)][None, :]
        base = np.exp(2j * np.pi * (fi * tcol))
        cur = base
        z = np.zeros(fi.shape[1])
        h = np.full(fi.shape[1], -np.inf)
        for m in range(1, int(m_max) + 1):
            if m > 1:
                cur = cur * base
            s = cur.sum(axis=0)
            z = z + (2.0 / n) * (s.real ** 2 + s.imag ** 2)
            h = np.maximum(h, z - 4.0 * m + 4.0)
        out[i:i + int(chunk)] = h
    return out


def phase_stats(times, period: float, *, phase_window: float = 0.05) -> dict:
    """Clock quality at one period, read off the phase distribution."""
    t = np.asarray(times, dtype=float)
    n = len(t)
    if n == 0 or not np.isfinite(period) or period <= 0:
        return {"n": int(n), "rbar": float("nan"), "Q": float("nan"),
                "jitter": float("nan"), "f_in_window": float("nan"),
                "mean_phase": float("nan"), "t0": float("nan")}
    phi = 2.0 * np.pi * np.mod(t / period, 1.0)
    c, s = np.cos(phi).mean(), np.sin(phi).mean()
    rbar = float(np.hypot(c, s))
    theta = float(np.arctan2(s, c))
    dev = np.sqrt(max(2.0 * (1.0 - rbar), 0.0))
    r0 = math.sqrt(math.pi) / (2.0 * math.sqrt(n))       # E[Rbar | uniform]
    dev0 = math.sqrt(2.0 * (1.0 - min(r0, 0.999)))
    q = float(np.clip(1.0 - dev / dev0, 0.0, 1.0))
    d = np.mod(phi - theta + np.pi, 2.0 * np.pi) - np.pi
    jitter = float(np.sqrt(np.mean(d ** 2)) / (2.0 * np.pi))
    f_in = float(np.mean(np.abs(d) / (2.0 * np.pi) <= float(phase_window)))
    t0 = float((theta / (2.0 * np.pi)) * period)
    return {"n": int(n), "rbar": rbar, "Q": q, "jitter": jitter, "f_in_window": f_in,
            "mean_phase": float(np.mod(theta / (2.0 * np.pi), 1.0)), "t0": t0}


def refine_frequency(times, f0: float, df: float, *, m_max: int = 4,
                     points: int = 41, halfwidth_steps: float = 2.0) -> float:
    """Fine local maximum of H around a grid peak."""
    grid = f0 + np.linspace(-halfwidth_steps * df, halfwidth_steps * df, int(points))
    grid = grid[grid > 0]
    if not len(grid):
        return float(f0)
    h = h_statistic(times, grid, m_max=m_max, chunk=len(grid))
    return float(grid[int(np.argmax(h))])


def fundamental_period(times, period: float, *, tol: float = 0.02, k_max: int = 12,
                       phase_window: float = 0.05, max_period: float | None = None) -> float:
    """Walk to the longest period among ``k*P`` with the same phase concentration.

    A zero-jitter clock folded at ``P/2`` is exactly as concentrated as at ``P``,
    so the grid maximum can land on a sub-harmonic --- ``P/7`` for a clock with
    a low duty cycle, which is why ``k_max`` must exceed the small primes.
    Walking *up* stops at the first multiple where the events split into
    several phases --- ``2P`` for any clock that ticks in consecutive cycles.
    """
    best = float(period)
    q_ref = phase_stats(times, best, phase_window=phase_window)["Q"]
    t = np.asarray(times, dtype=float)
    span = float(np.nanmax(t) - np.nanmin(t)) if len(t) > 1 else float("inf")
    p_cap = span / 3.0 if max_period is None else float(max_period)
    # Multiplicative walk: from the current best, take the LARGEST multiple
    # (2..k_max) that keeps the concentration, and repeat.  A grid maximum on
    # P/6 climbs 6 -> 3 -> 1 (or 6 -> 2 -> 1) rather than stalling at 3.
    for _ in range(12):
        moved = False
        for k in range(int(k_max), 1, -1):
            cand = best * k
            if cand > p_cap:
                continue
            q = phase_stats(times, cand, phase_window=phase_window)["Q"]
            if np.isfinite(q) and q >= q_ref - float(tol):
                best, q_ref, moved = cand, max(q, q_ref), True
                break
        if not moved:
            break
    return best


def gap_integer_fraction(times, period: float, windows: Windows | None = None,
                         *, tol: float = 0.05) -> tuple[float, int]:
    """Fraction of consecutive same-window waiting times that are ~integer periods."""
    t = np.sort(np.asarray(times, dtype=float))
    if len(t) < 2 or not np.isfinite(period) or period <= 0:
        return float("nan"), 0
    gaps = np.diff(t)
    if windows is not None and windows.n:
        wi = windows.window_index(t)
        same = (wi[1:] == wi[:-1]) & (wi[1:] >= 0)
        gaps = gaps[same]
    if len(gaps) == 0:
        return float("nan"), 0
    r = gaps / period
    k = np.round(r)
    ok = (k >= 1) & (np.abs(r - k) <= float(tol))
    return float(ok.mean()), int(len(gaps))


def screen_p_upper(h_obs: float, n_freq: int, rate: float = 0.3) -> float:
    """A deliberately loose (conservative) Bonferroni bound on the scan p-value.

    ``P(H > h)`` for a single trial decays roughly as ``exp(-0.4 h)`` (de Jager &
    Büsching 2010); ``rate = 0.3`` overstates the tail, so a star screened out
    here would not have been significant under the empirical null either.
    """
    if not np.isfinite(h_obs) or n_freq <= 0:
        return 1.0
    return float(min(1.0, n_freq * math.exp(-float(rate) * float(h_obs))))


def gumbel_tail_p(h_obs: float, null_h) -> tuple[float, float, float]:
    """Gumbel extrapolation of the null max-H tail: ``(p, mu, beta)``.

    The maximum over many exponential-tailed trials converges to a Gumbel; a
    method-of-moments fit is stable on a few hundred draws.  Used only when
    the empirical p is pinned at its resolution floor, and flagged.
    """
    x = np.asarray(null_h, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return float("nan"), float("nan"), float("nan")
    beta = float(np.std(x, ddof=1) * math.sqrt(6.0) / math.pi)
    if beta <= 0:
        return float("nan"), float("nan"), float("nan")
    mu = float(np.mean(x) - 0.5772156649 * beta)
    z = (float(h_obs) - mu) / beta
    p = float(-np.expm1(-math.exp(-z))) if z < 700 else 0.0
    return max(p, 0.0), mu, beta


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------
@dataclass
class ScanResult:
    n: int = 0
    n_freq: int = 0
    span_days: float = float("nan")
    min_period_used: float = float("nan")
    max_period_used: float = float("nan")
    h_max: float = float("nan")
    f_best: float = float("nan")
    period_grid: float = float("nan")
    period: float = float("nan")
    Q: float = float("nan")
    jitter: float = float("nan")
    f_in_window: float = float("nan")
    rbar: float = float("nan")
    mean_phase: float = float("nan")
    t0: float = float("nan")
    gap_integer_frac: float = float("nan")
    n_gaps_used: int = 0
    cycles_span: float = float("nan")
    cycles_hit: int = 0
    cycle_occupancy: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)


def scan(times, windows: Windows | None, conf: dict | None = None) -> ScanResult:
    """Full period scan of one event list; every number the tiers need."""
    c = dict(DEFAULT_SCAN, **(conf or {}))
    t = np.sort(np.asarray(times, dtype=float))
    t = t[np.isfinite(t)]
    res = ScanResult(n=int(len(t)))
    if len(t) < 2:
        return res
    span = float(t[-1] - t[0])
    res.span_days = span
    cad = float(windows.cadence_days) if windows is not None else float("nan")
    p_min = float(c["min_period_days"])
    if np.isfinite(cad) and cad > 0:
        p_min = max(p_min, float(c["n_cadences_min"]) * cad)
    freqs = frequency_grid(span, min_period=p_min, max_period=float(c["max_period_days"]),
                           oversample=float(c["oversample"]),
                           span_fraction_max=float(c["span_fraction_max"]))
    res.n_freq = int(len(freqs))
    res.min_period_used = p_min
    res.max_period_used = min(float(c["max_period_days"]), span * float(c["span_fraction_max"]))
    if res.n_freq == 0:
        return res
    h = h_statistic(t, freqs, m_max=int(c["m_max"]), chunk=int(c["chunk"]))
    i = int(np.argmax(h))
    res.h_max = float(h[i])
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0 / (5.0 * span)
    f_ref = refine_frequency(t, float(freqs[i]), df, m_max=int(c["m_max"]),
                             points=int(c["refine_points"]),
                             halfwidth_steps=float(c["refine_halfwidth_steps"]))
    res.f_best = f_ref
    res.period_grid = 1.0 / float(freqs[i])
    p = fundamental_period(t, 1.0 / f_ref, tol=float(c["harmonic_walk_tol"]),
                           k_max=int(c["harmonic_walk_max"]),
                           phase_window=float(c["phase_window"]),
                           max_period=res.max_period_used)
    res.period = float(p)
    ps = phase_stats(t, p, phase_window=float(c["phase_window"]))
    res.Q, res.jitter, res.f_in_window = ps["Q"], ps["jitter"], ps["f_in_window"]
    res.rbar, res.mean_phase, res.t0 = ps["rbar"], ps["mean_phase"], ps["t0"]
    res.gap_integer_frac, res.n_gaps_used = gap_integer_fraction(
        t, p, windows, tol=float(c["gap_tol"]))
    # Cycle bookkeeping: how many ticks of the clock fell in observed time, and
    # how many of those carry an event.  Report-only; a beacon need not tick
    # every cycle, but a reader should see the duty cycle.
    if windows is not None and windows.n:
        k0 = np.floor((windows.starts[0] - res.t0) / p)
        k1 = np.ceil((windows.stops[-1] - res.t0) / p)
        ks = np.arange(k0, k1 + 1)
        ticks = res.t0 + ks * p
        in_obs = windows.contains(ticks)
        res.cycles_span = float(in_obs.sum())
        hit = np.unique(np.round((t - res.t0) / p))
        res.cycles_hit = int(len(hit))
        res.cycle_occupancy = float(len(hit) / max(in_obs.sum(), 1))
    return res


# ---------------------------------------------------------------------------
# Nulls
# ---------------------------------------------------------------------------
@dataclass
class NullResult:
    kind: str
    n_trials: int = 0
    n_exceed: int = 0
    p_empirical: float = float("nan")
    at_floor: bool = False
    stopped_early: bool = False
    truncated_by_budget: bool = False
    p_gumbel: float = float("nan")
    gumbel_mu: float = float("nan")
    gumbel_beta: float = float("nan")
    h_null_median: float = float("nan")
    h_null_p99: float = float("nan")
    h_null_max: float = float("nan")
    null_Q_median: float = float("nan")
    null_jitter_median: float = float("nan")
    elapsed_s: float = 0.0
    h_null: list = field(default_factory=list, repr=False)

    @property
    def p(self) -> float:
        """The p-value to use: empirical unless pinned at the floor."""
        if self.at_floor and np.isfinite(self.p_gumbel):
            return float(min(self.p_empirical, self.p_gumbel))
        return float(self.p_empirical)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("h_null", None)
        d["p"] = self.p
        return d


def _run_null(kind: str, h_obs: float, draw, scan_conf: dict, null_conf: dict,
              windows: Windows | None, *, quality: bool = False) -> NullResult:
    """Besag-Clifford sequential Monte Carlo over ``draw()`` event lists."""
    nc = dict(DEFAULT_NULL, **(null_conf or {}))
    res = NullResult(kind=kind)
    if not np.isfinite(h_obs):
        return res
    t_start = _time.monotonic()
    hs, qs, js = [], [], []
    exceed = 0
    n = 0
    h_stop, n_max = int(nc["h_stop"]), int(nc["n_max"])
    n_min = int(nc["n_min_trials"])
    while n < n_max:
        tt = draw()
        if tt is None or len(tt) < 2:
            break
        r = scan(tt, windows, scan_conf)
        n += 1
        hs.append(r.h_max)
        if quality:
            qs.append(r.Q)
            js.append(r.jitter)
        if np.isfinite(r.h_max) and r.h_max >= h_obs:
            exceed += 1
        if exceed >= h_stop and n >= n_min:
            res.stopped_early = True
            break
        if _time.monotonic() - t_start > float(nc["budget_s"]) and n >= n_min:
            res.truncated_by_budget = True
            break
    res.n_trials, res.n_exceed = n, exceed
    res.elapsed_s = round(_time.monotonic() - t_start, 2)
    if n == 0:
        return res
    if res.stopped_early:
        res.p_empirical = float(exceed / n)
    else:
        res.p_empirical = float((exceed + 1) / (n + 1))
        res.at_floor = exceed == 0
    arr = np.asarray(hs, dtype=float)
    res.h_null = [float(x) for x in arr]
    res.h_null_median = float(np.nanmedian(arr))
    res.h_null_p99 = float(np.nanpercentile(arr, 99)) if len(arr) >= 10 else float("nan")
    res.h_null_max = float(np.nanmax(arr))
    if res.at_floor:
        res.p_gumbel, res.gumbel_mu, res.gumbel_beta = gumbel_tail_p(h_obs, arr)
    if quality and qs:
        res.null_Q_median = float(np.nanmedian(qs))
        res.null_jitter_median = float(np.nanmedian(js))
    return res


def window_null(times, windows: Windows, h_obs: float, scan_conf: dict | None = None,
                null_conf: dict | None = None, rng=None, *, quality: bool = True
                ) -> NullResult:
    """Null 1: N events uniform in observed time, cadence-snapped, same scan."""
    rng = np.random.default_rng(rng)
    n = int(len(np.asarray(times)))

    def draw():
        return windows.sample(n, rng)

    return _run_null("window_resample", h_obs, draw, scan_conf or {}, null_conf or {},
                     windows, quality=quality)


def shuffle_waiting_times(times, windows: Windows, rng) -> np.ndarray | None:
    """One realisation of the waiting-time shuffle in observed-time coordinates."""
    t = np.sort(np.asarray(times, dtype=float))
    if len(t) < 3 or not windows.n:
        return None
    tau = windows.observed_time(t)
    gaps = np.diff(tau)
    gaps = gaps[rng.permutation(len(gaps))]
    total = windows.total
    free = max(total - gaps.sum(), 0.0)
    tau0 = rng.uniform(0.0, free) if free > 0 else 0.0
    tau_new = tau0 + np.concatenate([[0.0], np.cumsum(gaps)])
    tau_new = np.clip(tau_new, 0.0, max(total - 1e-9, 0.0))
    return np.sort(windows.quantize(windows.real_time(tau_new)))


def shuffle_null(times, windows: Windows, h_obs: float, scan_conf: dict | None = None,
                 null_conf: dict | None = None, rng=None) -> NullResult:
    """Null 2: waiting times permuted; burstiness kept, long-range order destroyed."""
    rng = np.random.default_rng(rng)
    nc = dict(DEFAULT_NULL, **(null_conf or {}))
    nc = dict(nc, n_max=int(nc["n_shuffle"]), h_stop=int(nc["n_shuffle"]) + 1)

    def draw():
        return shuffle_waiting_times(times, windows, rng)

    return _run_null("waiting_time_shuffle", h_obs, draw, scan_conf or {}, nc, windows)


# ---------------------------------------------------------------------------
# Energy-phase coherence (report only)
# ---------------------------------------------------------------------------
def energy_phase_correlation(times, energies, period: float, mean_phase: float) -> dict:
    """Spearman rank correlation of flare energy with distance from the clock phase.

    A clock does not care how bright the tick is.  Rotational modulation does:
    the star's visible hemisphere sets which flares are seen and how large they
    look.  Report-only flag ``energy_incoherent``.
    """
    from scipy import stats

    t = np.asarray(times, dtype=float)
    e = np.asarray(energies, dtype=float) if energies is not None else np.zeros(0)
    out = {"energy_phase_rho": float("nan"), "energy_phase_p": float("nan"),
           "n_energy": 0}
    if len(e) != len(t) or len(t) < 6 or not np.isfinite(period) or period <= 0:
        return out
    ok = np.isfinite(e) & (e > 0) if np.nanmin(e) > 0 else np.isfinite(e)
    if ok.sum() < 6 or len(np.unique(e[ok])) < 3:
        return out
    phi = np.mod(t[ok] / period - mean_phase + 0.5, 1.0) - 0.5
    d = np.abs(phi)
    try:
        rho, p = stats.spearmanr(np.log10(e[ok]) if np.nanmin(e[ok]) > 0 else e[ok], d)
    except Exception:                                     # noqa: BLE001
        return out
    out.update({"energy_phase_rho": float(rho), "energy_phase_p": float(p),
                "n_energy": int(ok.sum())})
    return out


# ---------------------------------------------------------------------------
# Cross-star coincidence removal (before any star is scanned)
# ---------------------------------------------------------------------------
def cross_star_coincidence(star_ids, times, *, bin_days: float, min_stars: int = 5,
                           tail_p: float = 1e-6) -> dict:
    """Find epochs at which anomalously many *distinct* stars have an event.

    Threshold per bin = ``max(min_stars, Poisson_isf(tail_p / n_bins, lambda) + 1)``
    with ``lambda`` the mean distinct-star count over occupied bins.  Returns the
    boolean removal mask over events, the bad epochs, and the counts.
    """
    from scipy import stats

    sid = np.asarray(star_ids)
    t = np.asarray(times, dtype=float)
    n = len(t)
    if n == 0 or not np.isfinite(bin_days) or bin_days <= 0:
        return {"remove": np.zeros(n, dtype=bool), "bad_bins": [], "threshold": float("nan"),
                "n_removed_events": 0, "n_bad_bins": 0, "lambda": float("nan")}
    t0 = np.nanmin(t)
    b = np.floor((t - t0) / bin_days).astype(np.int64)
    # distinct stars per bin
    pairs = np.unique(np.stack([b, np.unique(sid, return_inverse=True)[1]], axis=1), axis=0)
    counts = np.bincount(pairs[:, 0], minlength=int(b.max()) + 1)
    occupied = counts[counts > 0]
    lam = float(occupied.mean()) if len(occupied) else 0.0
    n_bins = int(len(counts))
    thr_pois = float(stats.poisson.isf(float(tail_p) / max(n_bins, 1), lam)) + 1.0 \
        if lam > 0 else float("inf")
    thr = max(float(min_stars), thr_pois)
    bad = np.where(counts >= thr)[0]
    remove = np.isin(b, bad)
    return {"remove": remove, "bad_bins": [float(t0 + (k + 0.5) * bin_days) for k in bad],
            "bad_bin_counts": [int(counts[k]) for k in bad],
            "threshold": float(thr), "lambda": lam, "n_bins": n_bins,
            "n_removed_events": int(remove.sum()), "n_bad_bins": int(len(bad))}


# ---------------------------------------------------------------------------
# Per-star analysis
# ---------------------------------------------------------------------------
def analyze_star(times, windows: Windows, energies=None, scan_conf: dict | None = None,
                 null_conf: dict | None = None, rng=None, *, run_nulls: bool = True) -> dict:
    """Scan one star, run the screen, then the nulls only if the screen passes.

    Returns a flat dict of everything the vetting and tier stages need.  The
    null is skipped --- and ``null_computed`` is False --- when the loose
    Bonferroni bound on the scan p already exceeds ``screen_p_max``; that star
    could not have been significant and the cost is spent elsewhere.
    """
    sc = dict(DEFAULT_SCAN, **(scan_conf or {}))
    nc = dict(DEFAULT_NULL, **(null_conf or {}))
    rng = np.random.default_rng(rng)
    t_raw = np.asarray(times, dtype=float)
    t, e, n_declustered = decluster(t_raw, energies, gap_days=float(sc["decluster_gap_days"]))
    rec: dict = {"n_events_raw": int(len(t_raw)), "n_declustered": int(n_declustered),
                 "n_events": int(len(t)), "windows": windows.as_dict()}
    if len(t) < int(sc["n_min"]):
        rec.update({"status": "insufficient_events", "null_computed": False,
                    "p_window": float("nan"), "p_shuffle": float("nan")})
        return rec
    r = scan(t, windows, sc)
    rec.update(r.as_dict())
    rec["p_screen_upper"] = screen_p_upper(r.h_max, r.n_freq, float(nc["screen_rate"]))
    rec["status"] = "scanned"
    if not run_nulls or rec["p_screen_upper"] > float(nc["screen_p_max"]):
        rec.update({"null_computed": False, "p_window": rec["p_screen_upper"],
                    "p_window_source": "bonferroni_screen", "p_shuffle": float("nan")})
    else:
        not_clock = not (np.isfinite(r.Q) and r.Q >= float(nc["Q_watch"])
                         and np.isfinite(r.jitter) and r.jitter <= float(nc["jitter_watch"]))
        nc_run = dict(nc, n_max=min(int(nc["n_max"]), int(nc["n_max_not_clock"]))) \
            if not_clock else nc
        rec["null_budget_mode"] = "not_clock_reduced" if not_clock else "full"
        wn = window_null(t, windows, r.h_max, sc, nc_run, rng)
        rec.update({f"wn_{k}": v for k, v in wn.as_dict().items() if k != "kind"})
        rec["null_computed"] = wn.n_trials > 0
        rec["p_window"] = wn.p if wn.n_trials > 0 else rec["p_screen_upper"]
        rec["p_window_source"] = ("gumbel_extrapolated" if wn.at_floor and np.isfinite(wn.p_gumbel)
                                  else "empirical" if wn.n_trials > 0 else "bonferroni_screen")
        sn = shuffle_null(t, windows, r.h_max, sc, nc, rng) if r.n_gaps_used >= int(
            nc["shuffle_min_gaps"]) else NullResult(kind="waiting_time_shuffle")
        rec.update({f"sn_{k}": v for k, v in sn.as_dict().items() if k != "kind"})
        rec["p_shuffle"] = sn.p_empirical if sn.n_trials > 0 else float("nan")
    rec.update(energy_phase_correlation(t, e, r.period, r.mean_phase))
    return rec


def bh_fdr(pvals, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of rejected hypotheses (NaN never rejected)."""
    p = np.asarray(pvals, dtype=float)
    out = np.zeros(len(p), dtype=bool)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    idx = np.where(ok)[0]
    ps = p[idx]
    order = np.argsort(ps)
    m = len(ps)
    thr = float(alpha) * (np.arange(1, m + 1) / m)
    passed = ps[order] <= thr
    if not passed.any():
        return out
    k = int(np.max(np.where(passed)[0]))
    out[idx[order[:k + 1]]] = True
    return out


__all__ = ["DEFAULT_NULL", "DEFAULT_SCAN", "NullResult", "ScanResult", "analyze_star",
           "bh_fdr", "cross_star_coincidence", "decluster", "energy_phase_correlation",
           "frequency_grid", "fundamental_period", "gap_integer_fraction", "gumbel_tail_p",
           "h_statistic", "phase_stats", "refine_frequency", "scan", "screen_p_upper",
           "shuffle_null", "shuffle_waiting_times", "window_null"]
