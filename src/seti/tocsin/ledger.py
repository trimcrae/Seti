"""The persistent cross-night ledger --- this channel's actual instrument.

A broker filter is stateless: it answers "is this alert interesting tonight?".
The S30/glint observable is not a property of one alert, it is a property of a
*position over time*: does this catalogued star produce grey, unresolved,
non-moving events **again**, and with what timing?  Nothing in the alert stream
answers that; only accumulated state does.

Three disciplines are enforced here, and they are the reason a nightly screen
can be scientifically honest rather than a candidate-generating machine:

1. **Cumulative trials.**  Significance is quoted against the running total of
   ``target x visit`` screenings since the ledger opened, not against tonight's.
   A screen that forgets its own history manufactures a 3-sigma event every few
   weeks by construction.
2. **A real denominator.**  Alerts exist only where there was a *detection*, so
   the alert stream cannot tell you how many times a star was looked at and
   showed nothing.  The visit count comes from the **forced-photometry history**
   carried in the alert packet (``prvDiaForcedSources``), which is measured at
   the position whether or not it was detected.  Where that history is missing
   the denominator is marked ``approximate`` and the target is capped below
   candidate tier --- never silently assumed.
3. **A cadence-matched null.**  Any timing test must draw its null from the
   *actual visit epochs*, because the survey cadence itself has structure
   (in-night pairs, ~3-4 day revisits, seasonal gaps, the lunar cycle).  A
   period that merely matches the cadence is the dominant false positive for
   "structured timing", and sampling the null from real visit times removes it
   by construction --- the same discipline KNELL applies with injection-measured
   efficiency.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

LEDGER_VERSION = 2

# Night boundary at Cerro Pachon local noon, mirroring `schema.night_id` --- kept
# as a float here so the ledger can bin forced-photometry epochs into nights
# without importing the alert schema.
_NIGHT_BOUNDARY_FRAC = 16.0 / 24.0


def night_of(mjd: float) -> int:
    """Integer observing-night label for an MJD (Cerro Pachon local night)."""
    return int(math.floor(float(mjd) - _NIGHT_BOUNDARY_FRAC))

# Promotion tiers, in increasing order of interest.  A tier is a *statement
# about evidence*, not a ranking of excitement: `watch` means "one event, not
# yet testable", `candidate` means "repeated, grey-confirmed, trial-corrected".
TIERS = ("none", "watch", "interest", "candidate", "alarm")


@dataclass
class Event:
    """One screened **star-night**: the unit of both multiplicity and trials.

    The event unit is deliberately ``(target, night)`` and not ``(target, band)``
    or ``(target, alert)``.  A grey flash detected in the night's *g* and *r*
    visits is one event measured twice --- the second band is the colour
    measurement, not a second occurrence.  Counting it twice would inflate the
    multiplicity and, worse, corrupt the binomial denominator, which is counted
    in nights-visited.  Every statistic downstream therefore counts star-nights.
    """

    target_id: str
    night: str
    mjd: float                       # flux-weighted epoch of the night's detections
    polarity: str                    # "flash" | "dip"
    bands: list[str]
    dflux_njy: float                 # strongest band, signed
    dflux_err_njy: float
    strongest_band: str
    a: float                         # fractional amplitude dF/F* in strongest band
    a_err: float
    sep_arcsec: float
    sep_sigma: float
    grey_z: float                    # NaN when fewer than two bands were available
    grey_tested: bool
    colour_temp_k: float             # NaN when untestable (flash mode, >=2 bands)
    verdict: str = "kept"
    alert_ids: list[str] = field(default_factory=list)
    per_band: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass
class TargetRecord:
    """Accumulated state for one catalogued star."""

    target_id: str
    ra: float
    dec: float
    n_events: int = 0
    n_visits: int = 0
    visits_exact: bool = False
    first_mjd: float = float("nan")
    last_mjd: float = float("nan")
    events: list[dict] = field(default_factory=list)
    visit_mjds: list[float] = field(default_factory=list)
    visit_nights: list[int] = field(default_factory=list)
    tier: str = "none"
    p_binomial: float = float("nan")
    p_timing: float = float("nan")
    duty_cycle: float = float("nan")
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def binomial_sf(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, p)``.

    Implemented directly (no scipy dependency in the hot path) via the
    regularised incomplete beta identity, falling back to a stable sum for
    small ``n``.  Returns 1.0 for degenerate inputs so a missing denominator can
    never masquerade as significance.
    """
    if k <= 0:
        return 1.0
    # A degenerate denominator, an impossible k, or a rate outside (0,1) all mean
    # "no evidence": returning 1.0 is the only safe convention, because any other
    # value would let a missing visit history look like a detection.
    if n <= 0 or k > n or not (0.0 < p < 1.0):
        return 1.0
    # Sum the upper tail directly; n here is at most a few thousand visits.
    log_p, log_q = math.log(p), math.log1p(-p)
    total = 0.0
    log_binom = 0.0
    for j in range(0, n + 1):
        if j > 0:
            log_binom += math.log((n - j + 1) / j)
        if j >= k:
            total += math.exp(log_binom + j * log_p + (n - j) * log_q)
    return min(1.0, max(0.0, total))


def benjamini_hochberg(pvalues, alpha: float = 0.05):
    """Benjamini-Hochberg step-up.  Returns ``(reject, p_threshold)``.

    FDR rather than Bonferroni because the screen accumulates tens of thousands
    of targets: Bonferroni at that scale would reject a genuine repeater along
    with the noise.  The threshold is reported so the ledger can state the
    correction that was actually applied on the night in question.
    """
    p = np.asarray(pvalues, dtype=float)
    reject = np.zeros(p.size, dtype=bool)
    finite = np.isfinite(p)
    if not np.any(finite):
        return reject, float("nan")
    idx = np.nonzero(finite)[0]
    order = idx[np.argsort(p[idx])]
    m = order.size
    thresh = float("nan")
    kmax = -1
    for rank, j in enumerate(order, start=1):
        if p[j] <= alpha * rank / m:
            kmax = rank
            thresh = alpha * rank / m
    if kmax > 0:
        reject[order[:kmax]] = True
    return reject, thresh


def timing_structure(event_mjds, visit_mjds, period_grid=None,
                     n_null: int = 2000, rng: np.random.Generator | None = None
                     ) -> tuple[float, float, float]:
    """Test event epochs for a common period, against a **cadence-matched** null.

    Returns ``(best_period_days, rayleigh_r, p_value)``.

    The statistic is the Rayleigh concentration ``R`` of the event phases at the
    best trial period.  The null is generated by drawing the same number of
    events from the star's *own observed visit epochs* --- so a "period" that is
    merely the survey's revisit cadence, an in-night pair separation or a lunar
    alias scores exactly as highly under the null as in the data, and cancels.

    With fewer than three events no period is defined and the p-value is 1.0:
    two points are always perfectly periodic, and admitting them would make
    every second-time repeater look structured.
    """
    ev = np.asarray(sorted(float(m) for m in event_mjds), dtype=float)
    vis = np.asarray(sorted({float(m) for m in visit_mjds}), dtype=float)
    if ev.size < 3 or vis.size < ev.size + 1:
        return float("nan"), float("nan"), 1.0
    span = float(ev[-1] - ev[0])
    if span <= 0:
        return float("nan"), float("nan"), 1.0
    if period_grid is None:
        # The grid must REACH the shortest observed spacing: a beacon that fires
        # every P days has a minimum spacing of exactly P, so a grid starting
        # above it would exclude the very period being searched for.  A third of
        # that allows for up to two missed firings between detections, while
        # still staying well clear of the very-short-period regime where any set
        # of discretely sampled times looks periodic.  The upper end is the full
        # span: beyond it a "period" is indistinguishable from one interval.
        p_min = max(float(np.min(np.diff(ev))) / 3.0, 0.02)
        if p_min >= span:
            return float("nan"), float("nan"), 1.0
        period_grid = np.geomspace(p_min, span, 600)
    rng = rng or np.random.default_rng(20260730)

    def best_r(times: np.ndarray) -> tuple[float, int]:
        ph = 2.0 * np.pi * (times[None, :] / period_grid[:, None])
        r = np.hypot(np.cos(ph).mean(axis=1), np.sin(ph).mean(axis=1))
        return float(np.max(r)), int(np.argmax(r))

    r_obs, j_obs = best_r(ev)
    n_ge = 0
    for _ in range(int(n_null)):
        draw = rng.choice(vis, size=ev.size, replace=False)
        r_null, _ = best_r(np.sort(draw))
        if r_null >= r_obs:
            n_ge += 1
    # Add-one estimator: with n_null trials the smallest resolvable p-value is
    # 1/(n_null+1), and quoting 0 would be a lie about the resolution.
    p_val = (n_ge + 1) / (int(n_null) + 1)
    return float(period_grid[j_obs]), r_obs, p_val


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------
@dataclass
class Ledger:
    """Accumulated screening state.  Serialises to one small JSON file."""

    version: int = LEDGER_VERSION
    opened_utc: str = ""
    updated_utc: str = ""
    nights: list[str] = field(default_factory=list)
    n_target_visits: int = 0          # cumulative trials
    n_targets_screened: int = 0       # union of targets ever in footprint
    n_alerts_seen: int = 0
    n_events_kept: int = 0
    # High-water mark: the newest epoch already screened.  The next run starts
    # here, which is what makes coverage gapless AND non-overlapping regardless
    # of how far the broker's mirror lags behind the wall clock.
    last_mjd_screened: float = float("nan")
    targets: dict[str, dict] = field(default_factory=dict)
    rate_per_visit: float = float("nan")
    fdr_threshold: float = float("nan")
    notes: list[str] = field(default_factory=list)

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> Ledger:
        p = Path(path)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        if int(raw.get("version", 0)) != LEDGER_VERSION:
            led = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
            led.version = LEDGER_VERSION
            led.notes.append(f"migrated_from_version_{raw.get('version')}")
            return led
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # NaN/Inf are not valid JSON and `json` emits them unquoted by default,
        # producing a file that other tools refuse to read.  Non-finite means
        # "not measured" here, so it is written as null.
        p.write_text(json.dumps(_sanitize(asdict(self)), indent=1, sort_keys=True,
                                allow_nan=False) + "\n")

    # -- accumulation ------------------------------------------------------
    def add_night(self, night: str, events: list[Event],
                  target_visits: int, targets_in_footprint: int,
                  alerts_seen: int,
                  visit_history: dict[str, list[float]] | None = None,
                  target_positions: dict[str, tuple[float, float]] | None = None,
                  ) -> None:
        """Fold **one observing night** into the running state.

        ``target_visits`` is that night's trial count (distinct star-nights
        actually observed).  ``visit_history`` maps target id -> visit MJDs
        measured from forced photometry; where a target is absent from it, its
        denominator stays approximate and it cannot reach candidate tier.

        THE UNIT MUST BE ONE NIGHT, NOT ONE RUN.  The screen pulls a window of
        two nights so a missed cron firing is recovered, and it runs daily --- so
        consecutive runs overlap.  Folding a whole window under a single label
        would add the overlapping night's trials twice, which inflates the
        denominator, *deflates* the ensemble event rate, and therefore makes
        every per-target binomial p-value too small.  That error is
        anti-conservative: it manufactures significance.  Events are already
        de-duplicated by star-night; the trial count is made safe the same way,
        by keying on the night and ignoring a night already recorded.
        """
        if night and night not in self.nights:
            self.nights.append(night)
            self.n_target_visits += int(target_visits)
            self.n_targets_screened = max(self.n_targets_screened,
                                          int(targets_in_footprint))
            self.n_alerts_seen += int(alerts_seen)
        for ev in events:
            tid = str(ev.target_id)
            rec = self.targets.get(tid)
            if rec is None:
                pos = (target_positions or {}).get(tid, (float("nan"), float("nan")))
                rec = asdict(TargetRecord(target_id=tid, ra=pos[0], dec=pos[1]))
                self.targets[tid] = rec
            d = asdict(ev)
            # An event is identified by its star-night, which is the unit the
            # statistics count; re-running a night must not duplicate it.
            if not any(e.get("night") == ev.night for e in rec["events"]):
                rec["events"].append(d)
                rec["n_events"] = len(rec["events"])
                self.n_events_kept += 1
            mjds = [e["mjd"] for e in rec["events"]]
            rec["first_mjd"], rec["last_mjd"] = min(mjds), max(mjds)
        self.apply_visit_history(visit_history)

    def apply_visit_history(self, visit_history: dict | None) -> None:
        """Merge visit epochs into whichever target records now exist.

        Kept separate from :meth:`add_night` and called AFTER every night has
        been folded.  Applying it during the first night's fold silently lost
        the history of any target whose record is created by a LATER night —
        which is most of them, since a record only comes into existence when
        that target's event is folded.  The symptom was `visits_exact` staying
        False, and therefore candidate tier being unreachable.
        """
        for tid, mjds in (visit_history or {}).items():
            rec = self.targets.get(tid)
            if rec is None:
                continue
            merged = sorted({round(float(m), 6) for m in list(rec["visit_mjds"]) + list(mjds)})
            rec["visit_mjds"] = merged
            # The denominator must count the same unit the numerator does, and
            # the event unit is the star-NIGHT (see `Event`).  A night with an
            # in-night pair contributes two forced-photometry epochs but only
            # one opportunity to record an event.
            rec["visit_nights"] = sorted({night_of(m) for m in merged})
            rec["n_visits"] = len(rec["visit_nights"])
            rec["visits_exact"] = True

    # -- assessment --------------------------------------------------------
    def assess(self, alpha_fdr: float = 0.05, min_visits_for_rate: int = 5,
               max_duty_cycle: float = 0.2, n_null_timing: int = 2000,
               timing_alpha: float = 0.01, max_grey_z: float = 3.0) -> dict:
        """Recompute rates, p-values and tiers over the whole accumulated state.

        The ensemble event rate per target-visit is the null hypothesis every
        per-target p-value is measured against, and it is measured from *this*
        screen's own history --- not assumed, and not taken from another survey
        whose cadence and depth differ.
        """
        if self.n_target_visits > 0:
            self.rate_per_visit = self.n_events_kept / self.n_target_visits
        else:
            # May be None after a JSON round-trip (NaN is written as null).
            self.rate_per_visit = _finite(self.rate_per_visit) or float("nan")
        # A per-star-night event PROBABILITY cannot exceed 1.  If it does, the
        # numerator and the denominator were measured over different
        # populations and the quotient is not a rate --- publishing it would put
        # a meaningless number into a committed artefact and, worse, invite a
        # reader to treat it as one.  The first live run produced 7.75 this way.
        # Refuse it explicitly rather than letting it flow into p-values.
        _rate = _finite(self.rate_per_visit)
        if _rate is not None and _rate > 1.0:
            _note_list(self.notes,
                       f"INVALID ensemble rate {self.rate_per_visit:.3g} > 1: "
                       f"{self.n_events_kept} events against "
                       f"{self.n_target_visits} star-night trials — the "
                       "denominator does not cover the numerator's population, "
                       "so no per-target p-value is computed")
            self.rate_per_visit = float("nan")
            _rate = None
        rate = _rate if _rate is not None else float("nan")
        ids, pvals = [], []
        for tid, rec in self.targets.items():
            k = int(rec["n_events"])
            n = int(rec["n_visits"])
            rec["duty_cycle"] = (k / n) if n > 0 else float("nan")
            if rec["visits_exact"] and n >= min_visits_for_rate and np.isfinite(rate) \
                    and 0.0 < rate < 1.0:
                rec["p_binomial"] = binomial_sf(k, n, rate)
            else:
                rec["p_binomial"] = float("nan")
                if not rec["visits_exact"]:
                    _note(rec, "denominator_approximate")
            # The timing null draws from the star's own visited nights, one
            # epoch per night, so a "period" that is really the survey's
            # revisit cadence or the in-night pair separation cannot win.
            night_epochs = _one_epoch_per_night(rec["visit_mjds"]) or \
                [e["mjd"] for e in rec["events"]]
            per, r, p_t = timing_structure([e["mjd"] for e in rec["events"]],
                                           night_epochs, n_null=n_null_timing)
            rec["p_timing"] = p_t
            rec["timing_period_d"] = per
            rec["timing_r"] = r
            ids.append(tid)
            pvals.append(rec["p_binomial"])
        reject, thresh = benjamini_hochberg(pvals, alpha=alpha_fdr)
        self.fdr_threshold = thresh
        for tid, rej in zip(ids, reject, strict=True):
            self._set_tier(self.targets[tid], bool(rej), max_duty_cycle,
                           timing_alpha, max_grey_z,
                           min_visits_for_duty=min_visits_for_rate)
        return self.summary()

    def _set_tier(self, rec: dict, fdr_reject: bool, max_duty_cycle: float,
                  timing_alpha: float, max_grey_z: float = 3.0,
                  min_visits_for_duty: int = 5) -> None:
        events = rec["events"]
        k = len(events)
        grey_ok = [e for e in events
                   if e.get("grey_tested") and _finite(e.get("grey_z")) is not None
                   and abs(_finite(e.get("grey_z"))) <= max_grey_z]
        duty = rec.get("duty_cycle", float("nan"))
        # A star that alerts on a large fraction of its visits is a subtraction
        # residual (proper-motion dipole, template defect), not a beacon.  This
        # test costs nothing and removes the single most likely systematic for a
        # high-proper-motion nearby-star sample.
        #
        # It needs a real denominator to mean anything.  A star discovered
        # tonight has one visit and one event, so its duty cycle is 1.0 by
        # arithmetic --- applying the cut there would silently reject EVERY new
        # detection on the night it is found, which is the one night that
        # matters most.  Below `min_visits_for_duty` the statistic is not
        # computed against, and the target waits instead.
        n_visits = int(rec.get("n_visits") or 0)
        if (np.isfinite(duty) and duty > max_duty_cycle and rec["visits_exact"]
                and n_visits >= int(min_visits_for_duty)):
            rec["tier"] = "none"
            _note(rec, "rejected_high_duty_cycle")
            return
        if np.isfinite(duty) and duty > max_duty_cycle and n_visits < int(min_visits_for_duty):
            _note(rec, "duty_cycle_not_yet_testable")
        tier = "none"
        if k >= 1:
            tier = "watch"
        if grey_ok or k >= 2:
            tier = "interest"
        if k >= 2 and grey_ok and fdr_reject and rec["visits_exact"]:
            tier = "candidate"
        if tier == "candidate" and float(rec.get("p_timing", 1.0)) <= timing_alpha:
            tier = "alarm"
        rec["tier"] = tier

    def summary(self) -> dict:
        counts = dict.fromkeys(TIERS, 0)
        for rec in self.targets.values():
            counts[rec.get("tier", "none")] = counts.get(rec.get("tier", "none"), 0) + 1
        return {
            "version": self.version,
            "nights": len(self.nights),
            "cumulative_target_visits": self.n_target_visits,
            "alerts_seen": self.n_alerts_seen,
            "events_kept": self.n_events_kept,
            "ensemble_rate_per_target_visit": self.rate_per_visit,
            "fdr_threshold": self.fdr_threshold,
            "tier_counts": counts,
            "targets_with_events": len(self.targets),
        }


def _finite(value) -> float | None:
    """Coerce to a finite float, or ``None``.

    Saving the ledger writes non-finite numbers as JSON ``null`` (NaN is not
    valid JSON), so every value re-read from disk may be ``None`` where it was
    NaN in memory.  Anything that reads an event field after a round-trip must
    go through here: ``float(None)`` raises, and a truthiness test would treat a
    perfectly grey event (``grey_z == 0.0``) as missing.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _one_epoch_per_night(mjds) -> list[float]:
    """Collapse forced-photometry epochs to one representative epoch per night."""
    by_night: dict[int, list[float]] = {}
    for m in mjds or []:
        by_night.setdefault(night_of(m), []).append(float(m))
    return sorted(float(np.median(v)) for v in by_night.values())


def _note_list(notes: list, text: str) -> None:
    if text not in notes:
        notes.append(text)


def _note(rec: dict, text: str) -> None:
    if text not in rec.setdefault("notes", []):
        rec["notes"].append(text)


def _sanitize(obj):
    """Recursively make a structure JSON-safe: non-finite floats become null."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
