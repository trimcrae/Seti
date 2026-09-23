"""The contamination gauntlet and the tier assignment --- pure functions.

Every rejection is a *named mechanism* with its own counter, applied in the
order most-mundane-first so a star that trips several is reported under the
dullest one:

``insufficient_events``     fewer than ``n_min`` events after declustering and
                            cross-star removal (the cross-star removal itself
                            is counted at event level in the screen stage)
``not_significant``         the window-resampled null explains the coherence
                            (BH-FDR across every star scanned)
``pool_null_explains``      the coherence is not rarer than the same statistic
                            on N times drawn from the OTHER stars' catalogued
                            event times inside this star's own windows.  That
                            null carries the catalogue's real time lattice and
                            epoch structure without modelling either, so a
                            period that only looks sharp because the catalogue
                            quantises its peak times dies here
``cadence_alias``           P at a named instrumental period or its low
                            harmonics (Kepler cadence / momentum-dump / monthly
                            downlink / quarter; TESS orbit / sector / cadences)
``rotation_alias``          P within tolerance of P_rot, P_rot/2, P_rot/3,
                            P_rot/4, 2 P_rot, 3 P_rot --- rotational modulation
                            of flare visibility is the dominant natural
                            quasi-periodicity
``periodic_variable``       the star is a catalogued periodic variable (VSX /
                            Gaia DR3 vari / ZTF) and P sits at its period or a
                            low harmonic: a pulsator's cycles chopped into
                            "flares" by the flare finder
``aperture_contaminating_variable``
                            a catalogued variable NEIGHBOUR inside the
                            photometric aperture (Kepler 20", TESS 120";
                            outside the 3" identity cone) has P at its period
                            or a low harmonic: its cycle leaks into this
                            star's light curve.  A contamination statement,
                            not an identity one (kepler:5879574 / the RR Lyrae
                            KIC 5879583 at 13.3", 2026-09-22)
``population_period``      unrelated stars of the same mission pile up at
                            this period.  A clock is a property of ONE star;
                            a period that many independent stars share is a
                            property of the mission's sampling.  Measured
                            from the run's own scanned population, so it
                            needs no list of instrumental periods and catches
                            the ones nobody wrote down
``few_cycles``              the period repeated too few times inside the
                            observing windows for "recurs" to mean anything
                            --- the long-period tail where P approaches the
                            span/3 grid edge and three sector groups phase up
``bursty_random``           the shuffle null does not beat the observed H
                            (coherence explained by the waiting-time
                            distribution) AND the waiting times are not
                            integer periods --- bursty, not clocked
``jitter_too_large``        the phase concentration fails even the loose
                            ("watch") clock thresholds

Report-only flags never reject: ``energy_incoherent`` (flare energy depends
on clock phase --- what visibility modulation does and a beacon should not),
``quality_uninformative`` (fewer than ``n_quality_informative`` events, so the
strict Q / jitter gates carry no discriminating power on this star and its
case rests on the null alone -- see :func:`calibrate_jitter`),
``rotation_unknown``, ``variability_catalogue_unreached``, ``p_extrapolated``,
``null_truncated_by_budget``, ``quantisation_limited`` (the measured phase
jitter is at the floor the catalogue's own time rounding imposes, so the
tightness is a property of the time stamps rather than evidence about the
star), ``pool_null_unreached`` (too few other-star times inside the windows to
run null 3; the star is NOT credited with passing it).

Tiers
-----
``none``       not significant at the watch FDR, or a hard veto tripped
``watch``      significant at ``fdr_alpha_watch``, no hard veto, loose quality
``interest``   significant at ``fdr_alpha``, strict quality, but a veto could
               not be applied (no P_rot, a variability catalogue was not
               reached, or the pool null could not be run) --- candidate-grade
               statistics with an incomplete vet
``candidate``  significant at ``fdr_alpha``, strict quality, every veto
               applied and passed.  PENDING human/light-curve vet always.
"""

from __future__ import annotations

import math

import numpy as np

from .clock import bh_fdr

DEFAULT_VET: dict = {
    "rotation_ratios": [1.0, 0.5, 1.0 / 3.0, 0.25, 2.0, 3.0],
    "rotation_tol": 0.03,
    "cadence_harmonics": [1.0, 2.0, 3.0, 0.5, 1.0 / 3.0],
    "cadence_tol": 0.02,
    "variable_harmonics": [1.0, 0.5, 1.0 / 3.0, 2.0, 3.0],
    "variable_tol": 0.03,
    # The APERTURE-scale contamination cone (see ``aperture_contamination``):
    # a variable NEIGHBOUR whose period, or a low harmonic of it, is the clock
    # period is putting its own cycle into this star's photometry.  Tighter
    # than ``variable_tol`` because a 1-2 arcmin TESS cone holds several
    # unrelated variables and each one is a chance of a coincidence; the
    # chance probability is reported with every hit.
    "aperture_tol": 0.01,
    "aperture_period_range_days": [0.05, 1000.0],
    "fdr_alpha": 0.05,
    "fdr_alpha_watch": 0.25,
    "shuffle_alpha": 0.05,
    "gap_frac_min": 0.6,
    "gap_min_count": 4,
    "Q_min": 0.85,
    "jitter_max": 0.05,
    "Q_watch": 0.6,
    "jitter_watch": 0.12,
    # The CORE route to the same quality: when a natural flare background is
    # mixed with the ticks, the rms over all events fails while the events at
    # the clock phase are a perfect clock.  Strict: >= f_core_min of all events
    # inside +-phase_window (0.05) cycle AND the core jitter <= jitter_max over
    # >= core_n_min core events; watch: f_core_watch and jitter_watch.
    # Rotational modulation (rate ∝ 1 + cos) has f_in_window ≈ 0.20 and a core
    # jitter ≈ 0.087, below both gates.
    "f_core_min": 0.6,
    "f_core_watch": 0.4,
    "core_n_min": 8,
    "energy_p_max": 0.01,
    # Null 3.  A star must be rarer than `pool_alpha` against the catalogue's
    # own event times resampled inside its windows.  The threshold is loose on
    # purpose: with n_pool = 200 trials the smallest reachable p is ~0.005, and
    # the job of this null is to kill lattice artefacts, not to re-rank real
    # clocks that null 1 has already put at p ~ 1e-20.
    "pool_alpha": 0.05,
    # A catalogue that rounds its peak times to a lattice of spacing g forces
    # an rms phase jitter of at least g / (P sqrt(12)) on ANY clock.  A star
    # whose measured jitter is within `quantisation_factor` of that floor is as
    # tight as its time stamps allow and no tighter, which is a fact about the
    # catalogue, not about the star.  Report-only: it says where to look, the
    # pool null says whether to believe.
    "quantisation_factor": 1.5,
    # ``few_cycles``.  The claim "the brightenings recur on a clock" is only
    # as strong as the number of repeats behind it.  ``cycles_span`` counts
    # the ticks whose phase window had any observing coverage, i.e. the
    # opportunities the clock had; below ``cycles_min`` the star is vetoed.
    # MEASURED on the 2026-09-21 run: every star whose best period exceeded
    # ~span/4 had cycles_span <= 5, and those periods clustered at 243 d
    # (TESS) and 372 d (Kepler) across unrelated stars -- window structure,
    # not clocks.  This bounds the channel's reach to P < span / cycles_min
    # and that bound is reported in the coverage block.
    "cycles_min": 10.0,
    # ``population_period``.  Per mission, over the log10 periods of every
    # scanned star: a star is vetoed when at least ``pop_min_count`` OTHER
    # stars sit within ``pop_tol_dex`` of it and that count is Poisson-rarer
    # than ``pop_alpha`` against the local background density measured over
    # ``pop_bg_dex``.  The tolerance is a fractional period tolerance (0.005
    # dex ~ 1.2%), matched to the period resolution of the scan.
    # ``pop_min_stars`` is the population below which the density estimate
    # says nothing and the veto is not applied at all.
    "pop_tol_dex": 0.005,
    "pop_bg_dex": 0.25,
    "pop_min_count": 4,
    "pop_alpha": 1e-3,
    "pop_min_stars": 50,
    # MEASURED on run 35652897914: with the period free on a ~10^4-point grid
    # the fitted jitter is a function of N before it is a function of the
    # star -- median jitter 0.030 at N = 8-11 rising to 0.217 at N >= 80 --
    # and 83% of the stars that run itself rejected as `rotation_alias`
    # (every one of them N <= 34) pass BOTH strict quality gates.  The gates
    # are a look-elsewhere floor at small N, not a clock criterion, so a star
    # below this many events carries the report flag `quality_uninformative`
    # and its case rests on the null alone.  The null itself is not fooled:
    # the window null's own best fit reaches Q ~ 0.25, jitter ~ 0.18 and is
    # inside the strict gate for under 1% of stars.
    "n_quality_informative": 35,
    "instrumental_periods": {
        "kepler": {"long_cadence": 0.020434, "momentum_dump": 3.0,
                   "monthly_downlink": 31.0, "quarter": 93.0},
        "tess": {"cadence_2min": 0.0013889, "cadence_10min": 0.0069444,
                 "cadence_200s": 0.0023148, "ffi_30min": 0.0208333,
                 "momentum_dump_early": 3.5, "orbit": 13.7, "sector": 27.4},
    },
}

HARD_VETO_ORDER = ("pool_null_explains", "population_period", "few_cycles",
                   "cadence_alias", "rotation_alias",
                   "periodic_variable", "aperture_contaminating_variable",
                   "bursty_random", "jitter_too_large")
REPORT_FLAGS = ("energy_incoherent", "rotation_unknown", "variability_catalogue_unreached",
                "aperture_catalogue_unreached", "aperture_variable_neighbour",
                "p_extrapolated", "null_truncated_by_budget", "quantisation_limited",
                "pool_null_unreached", "quality_uninformative")


def _close(a: float, b: float, tol: float) -> bool:
    return bool(np.isfinite(a) and np.isfinite(b) and b > 0 and abs(a / b - 1.0) <= tol)


def rotation_alias(period: float, prot: float, ratios=None, tol: float = 0.03
                   ) -> tuple[bool, float | None]:
    """Is ``period`` a low harmonic / multiple of the rotation period?"""
    ratios = DEFAULT_VET["rotation_ratios"] if ratios is None else ratios
    if not (np.isfinite(period) and np.isfinite(prot) and prot > 0):
        return False, None
    for r in ratios:
        if _close(period, prot * float(r), tol):
            return True, float(r)
    return False, None


def cadence_alias(period: float, instrumental: dict | None, harmonics=None,
                  tol: float = 0.02) -> tuple[bool, str | None]:
    """Is ``period`` at a named instrumental period (or a low harmonic of one)?"""
    harmonics = DEFAULT_VET["cadence_harmonics"] if harmonics is None else harmonics
    if not instrumental or not np.isfinite(period):
        return False, None
    for name, p in instrumental.items():
        for h in harmonics:
            if _close(period, float(p) * float(h), tol):
                return True, f"{name}x{h:.3g}"
    return False, None


def periodic_variable(period: float, catalogued, harmonics=None, tol: float = 0.03
                      ) -> tuple[bool, str | None]:
    """``catalogued`` is an iterable of ``(source, period, vtype)`` for this star."""
    harmonics = DEFAULT_VET["variable_harmonics"] if harmonics is None else harmonics
    if not catalogued or not np.isfinite(period):
        return False, None
    for src, p, vtype in catalogued:
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(p) and p > 0):
            continue
        for h in harmonics:
            if _close(period, p * float(h), tol):
                return True, f"{src}:{vtype}:P={p:.6g}x{h:.3g}"
    return False, None


def aperture_contamination(period: float, neighbours, harmonics=None, tol: float = 0.01,
                           period_range=None) -> tuple[bool, dict]:
    """Is a variable NEIGHBOUR catalogued at the clock period (or a low harmonic)?

    ``neighbours`` are dicts ``{source, name, period, vtype, sep_arcsec}`` from
    the aperture-scale cone, the target itself already excluded.  Returns
    ``(hit, detail)``; ``detail`` always carries the census (how many variable
    neighbours, how many with a period) and the probability that ANY of them
    would land within ``tol`` of the clock or a harmonic by chance, for
    periods spread log-uniformly over ``period_range`` -- so a hit in a
    crowded TESS aperture is read against the coincidence rate it implies.
    """
    harmonics = DEFAULT_VET["variable_harmonics"] if harmonics is None else harmonics
    lo, hi = (period_range or DEFAULT_VET["aperture_period_range_days"])
    nb = [n for n in (neighbours or []) if isinstance(n, dict)]
    with_p = [n for n in nb if np.isfinite(_fnum(n.get("period")))
              and _fnum(n.get("period")) > 0]
    # distinct neighbours: the same star in VSX, Gaia and ZTF is one star, so
    # count distinct separations (to 0.5") rather than rows
    seps = sorted({round(2.0 * _fnum(n.get("sep_arcsec"))) for n in with_p
                   if np.isfinite(_fnum(n.get("sep_arcsec")))})
    n_unmeasured = sum(1 for n in with_p if not np.isfinite(_fnum(n.get("sep_arcsec"))))
    n_distinct = len(seps) + n_unmeasured
    w = min(1.0, len(harmonics) * 2.0 * float(tol) / math.log(float(hi) / float(lo)))
    detail: dict = {"n_neighbours": len(nb), "n_with_period": len(with_p),
                    "n_distinct_with_period": n_distinct, "tol": float(tol),
                    "p_chance_any": float(1.0 - (1.0 - w) ** n_distinct) if n_distinct else 0.0,
                    "matches": []}
    if not np.isfinite(period) or not with_p:
        return False, detail
    for n in with_p:
        p = _fnum(n.get("period"))
        for h in harmonics:
            if _close(period, p * float(h), tol):
                detail["matches"].append({
                    "source": n.get("source"), "name": n.get("name"),
                    "vtype": n.get("vtype"), "period": p, "harmonic": float(h),
                    "sep_arcsec": _fnum(n.get("sep_arcsec")),
                    "frac_diff": abs(period / (p * float(h)) - 1.0)})
                break
    detail["matches"].sort(key=lambda m: m["frac_diff"])
    return bool(detail["matches"]), detail


def _fnum(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def population_period_stats(records, conf: dict | None = None) -> None:
    """Annotate each scanned record with how crowded its best period is.

    Sets ``pop_n_near`` (other scanned stars of the same mission within
    ``pop_tol_dex``), ``pop_expected`` (the local background density scaled to
    that tolerance) and ``pop_p`` (Poisson survival) **in place**.

    The argument this makes is the one thing a per-star null cannot make: a
    clock belongs to a star, so two unrelated stars agreeing on a period to
    1% is either a coincidence with a computable probability or a property of
    the instrument.  The background is measured locally in log period, so the
    steep rise of the period distribution toward the grid floor is divided
    out and only genuine *narrow* pile-ups are flagged.
    """
    c = dict(DEFAULT_VET, **(conf or {}))
    tol, bg = float(c["pop_tol_dex"]), float(c["pop_bg_dex"])
    by_mission: dict[str, list[dict]] = {}
    for r in records:
        if r.get("status") != "scanned":
            continue
        p = _f(r, "period")
        if not (np.isfinite(p) and p > 0):
            continue
        by_mission.setdefault(str(r.get("mission", "") or "").lower(), []).append(r)
    for _, group in by_mission.items():
        if len(group) < int(c["pop_min_stars"]):
            # too few stars for a density to mean anything; leave the keys
            # unset so vet_star does not apply the veto rather than applying
            # it with a meaningless background
            continue
        lp = np.log10(np.array([_f(r, "period") for r in group], dtype=float))
        srt = np.sort(lp)
        n_near = (np.searchsorted(srt, lp + tol, "right")
                  - np.searchsorted(srt, lp - tol, "left") - 1)
        n_bg = (np.searchsorted(srt, lp + bg, "right")
                - np.searchsorted(srt, lp - bg, "left") - 1)
        # the background band is wider than the test band by bg/tol, and the
        # star itself is excluded from both
        expected = n_bg * (tol / bg)
        for r, k, lam in zip(group, n_near, expected, strict=True):
            r["pop_n_near"] = int(k)
            r["pop_expected"] = float(lam)
            r["pop_p"] = float(_poisson_sf(int(k), float(lam)))


def _poisson_sf(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam), without a SciPy dependency."""
    if k <= 0:
        return 1.0
    lam = max(float(lam), 1e-12)
    # sum the first k terms of the pmf; k is small (a handful to a few tens)
    # and lam is small, so the direct sum is stable in double precision
    term = math.exp(-lam)
    cdf = term
    for i in range(1, k):
        term *= lam / i
        cdf += term
    return float(min(1.0, max(0.0, 1.0 - cdf)))


def _f(rec: dict, key: str) -> float:
    try:
        return float(rec.get(key, np.nan))
    except (TypeError, ValueError):
        return float("nan")


def _int(rec: dict, key: str) -> int:
    """An integer count, 0 when absent.  A record read back from CSV carries
    NaN for a column its shard never wrote (MEASURED 2026-09-23: the
    2026-09-21 shards predate the pool null, and ``int(nan or 0)`` crashed
    the re-assess, because NaN is truthy)."""
    v = _f(rec, key)
    return int(v) if np.isfinite(v) else 0


def core_pass(rec: dict, conf: dict, *, strict: bool) -> tuple[bool, list[str]]:
    """The core route: enough events AT the clock phase, and those events a
    clock.  Robust to a natural flare background the rms route is not."""
    c = dict(DEFAULT_VET, **(conf or {}))
    f_in, jc, nc = _f(rec, "f_in_window"), _f(rec, "jitter_core"), _int(rec, "n_core")
    f_min = float(c["f_core_min"] if strict else c["f_core_watch"])
    j_max = float(c["jitter_max"] if strict else c["jitter_watch"])
    why = []
    if not (np.isfinite(f_in) and f_in >= f_min):
        why.append(f"f_in_window<{f_min}")
    if not (np.isfinite(jc) and jc <= j_max):
        why.append(f"jitter_core>{j_max}")
    if nc < int(c["core_n_min"]):
        why.append(f"n_core<{c['core_n_min']}")
    return (not why), why


def quality_pass(rec: dict, conf: dict, *, strict: bool) -> tuple[bool, list[str]]:
    """Does the phase concentration meet the clock threshold (strict or watch)?

    Two routes, either suffices: the rms route (``Q`` and ``jitter`` over every
    event) and the core route (:func:`core_pass`).  The integer-gap test, on
    the strict tier, accepts the core events' gaps as well as everyone's: a
    background flare between two ticks splits one integer gap into two
    non-integer ones without the ticks having moved.
    """
    c = dict(DEFAULT_VET, **(conf or {}))
    q, j = _f(rec, "Q"), _f(rec, "jitter")
    why = []
    q_min = float(c["Q_min"] if strict else c["Q_watch"])
    j_max = float(c["jitter_max"] if strict else c["jitter_watch"])
    rms_why = []
    if not (np.isfinite(q) and q >= q_min):
        rms_why.append(f"Q<{q_min}")
    if not (np.isfinite(j) and j <= j_max):
        rms_why.append(f"jitter>{j_max}")
    core_ok, core_why = core_pass(rec, c, strict=strict)
    if rms_why and not core_ok:
        why.extend(rms_why)
        why.extend("core:" + w for w in core_why)
    if strict:
        gf, ng = _f(rec, "gap_integer_frac"), _int(rec, "n_gaps_used")
        gfc, ngc = _f(rec, "gap_integer_frac_core"), _int(rec, "n_gaps_core")
        best_n = max(ng, ngc)
        if best_n >= int(c["gap_min_count"]):
            ok_all = ng >= int(c["gap_min_count"]) and np.isfinite(gf) \
                and gf >= float(c["gap_frac_min"])
            ok_core = ngc >= int(c["gap_min_count"]) and np.isfinite(gfc) \
                and gfc >= float(c["gap_frac_min"])
            if not (ok_all or ok_core):
                why.append(f"gap_integer_frac<{c['gap_frac_min']}")
        else:
            why.append("gap_integer_frac_unmeasurable")
    return (not why), why


def vet_star(rec: dict, context: dict | None = None, conf: dict | None = None) -> dict:
    """Apply the gauntlet to one star's record; return flags, first veto, tier.

    ``rec`` is the :func:`seti.metronome.clock.analyze_star` output plus the
    booleans ``fdr_significant`` and ``fdr_watch`` set by :func:`assign_tiers`.
    ``context`` carries ``prot`` (float or NaN), ``catalogued_periods`` (list of
    ``(source, period, vtype)``), ``variability_catalogues_reached`` (bool) and
    ``mission`` (``"kepler"`` / ``"tess"``).
    """
    c = dict(DEFAULT_VET, **(conf or {}))
    ctx = context or {}
    flags: list[str] = []
    detail: dict = {}
    out = {"tier": "none", "first_veto": None, "flags": flags, "veto_detail": detail}

    if rec.get("status") != "scanned":
        out["first_veto"] = "insufficient_events"
        return out
    if not bool(rec.get("fdr_watch", False)):
        out["first_veto"] = "not_significant"
        return out

    period = float(rec.get("period", np.nan))
    mission = str(ctx.get("mission", rec.get("mission", ""))).lower()
    inst = (c.get("instrumental_periods") or {}).get(mission) or {}

    # Null 3 first: it is the most mundane explanation available (the
    # catalogue's own sampling reproduces the coherence), and unlike the
    # window null it needs no model of the cadence to say so.
    p_pool = _f(rec, "p_pool")
    n_pool = _int(rec, "pn_n_trials")
    if n_pool > 0:
        if np.isfinite(p_pool) and p_pool >= float(c["pool_alpha"]):
            flags.append("pool_null_explains")
            detail["pool_null_explains"] = {"p_pool": p_pool, "n_trials": n_pool}
    else:
        flags.append("pool_null_unreached")
    floor, jit = _f(rec, "jitter_floor"), _f(rec, "jitter")
    jit_used = jit if np.isfinite(jit) else _f(rec, "jitter_core")
    if np.isfinite(floor) and floor > 0 and np.isfinite(jit_used) \
            and jit_used <= float(c["quantisation_factor"]) * floor:
        flags.append("quantisation_limited")
        detail["quantisation_limited"] = {"jitter": jit_used, "jitter_floor": floor,
                                          "grid_days": _f(rec, "grid_days"),
                                          "grid_source": rec.get("grid_source")}

    # Unrelated stars sharing a period: a property of the mission, not of any
    # one star.  Needs population_period_stats to have run over every record.
    pop_k = rec.get("pop_n_near")
    if pop_k is not None:
        pop_p = _f(rec, "pop_p")
        if int(pop_k) >= int(c["pop_min_count"]) and np.isfinite(pop_p) \
                and pop_p < float(c["pop_alpha"]):
            flags.append("population_period")
            detail["population_period"] = {"n_near": int(pop_k),
                                           "expected": _f(rec, "pop_expected"),
                                           "p": pop_p,
                                           "tol_dex": float(c["pop_tol_dex"])}

    # Too few repeats for "recurs" to mean anything.
    cyc = _f(rec, "cycles_span")
    if np.isfinite(cyc) and cyc < float(c["cycles_min"]):
        flags.append("few_cycles")
        detail["few_cycles"] = {"cycles_span": cyc, "cycles_min": float(c["cycles_min"]),
                                "period": period, "span_days": _f(rec, "span_days")}

    hit, d = cadence_alias(period, inst, c["cadence_harmonics"], float(c["cadence_tol"]))
    if hit:
        flags.append("cadence_alias")
        detail["cadence_alias"] = d
    prot = float(ctx.get("prot", np.nan)) if ctx.get("prot") is not None else float("nan")
    if np.isfinite(prot) and prot > 0:
        hit, d = rotation_alias(period, prot, c["rotation_ratios"], float(c["rotation_tol"]))
        if hit:
            flags.append("rotation_alias")
            detail["rotation_alias"] = {"prot": prot, "ratio": d,
                                        "source": ctx.get("prot_source")}
    else:
        flags.append("rotation_unknown")
    reached = bool(ctx.get("variability_catalogues_reached", False))
    hit, d = periodic_variable(period, ctx.get("catalogued_periods") or [],
                               c["variable_harmonics"], float(c["variable_tol"]))
    if hit:
        flags.append("periodic_variable")
        detail["periodic_variable"] = d
    if not reached:
        flags.append("variability_catalogue_unreached")

    # The aperture-scale cone.  Its hit is a CONTAMINATION statement -- a
    # neighbour's cycle is in this star's aperture -- not an identity one.
    if "aperture_catalogues_reached" in ctx:
        hit, d = aperture_contamination(period, ctx.get("aperture_neighbours") or [],
                                        c["variable_harmonics"], float(c["aperture_tol"]),
                                        c.get("aperture_period_range_days"))
        if hit:
            flags.append("aperture_contaminating_variable")
            detail["aperture_contaminating_variable"] = d
        elif ctx.get("aperture_neighbours"):
            flags.append("aperture_variable_neighbour")
            detail["aperture_variable_neighbour"] = d
        if not bool(ctx.get("aperture_catalogues_reached")):
            flags.append("aperture_catalogue_unreached")

    p_sh = _f(rec, "p_shuffle")
    gf = _f(rec, "gap_integer_frac")
    ng = _int(rec, "n_gaps_used")
    gfc = _f(rec, "gap_integer_frac_core")
    ngc = _int(rec, "n_gaps_core")
    # clock-like gaps among ALL events or among the CORE events both clear the
    # star of "bursty"; a background flare between two ticks is not burstiness
    gaps_clocklike = (ng >= int(c["gap_min_count"]) and np.isfinite(gf)
                      and gf >= float(c["gap_frac_min"])) or \
        (ngc >= int(c["gap_min_count"]) and np.isfinite(gfc) and gfc >= float(c["gap_frac_min"]))
    if (np.isfinite(p_sh) and p_sh >= float(c["shuffle_alpha"])
            and ng >= int(c["gap_min_count"]) and not gaps_clocklike):
        flags.append("bursty_random")
        detail["bursty_random"] = {"p_shuffle": p_sh, "gap_integer_frac": gf,
                                   "gap_integer_frac_core": gfc}

    ok_watch, why_watch = quality_pass(rec, c, strict=False)
    if not ok_watch:
        flags.append("jitter_too_large")
        detail["jitter_too_large"] = why_watch

    # report-only
    ep = float(rec.get("energy_phase_p", np.nan))
    if np.isfinite(ep) and ep < float(c["energy_p_max"]):
        flags.append("energy_incoherent")
    if str(rec.get("p_window_source", "")) == "gumbel_extrapolated":
        flags.append("p_extrapolated")
    if bool(rec.get("wn_truncated_by_budget", False)):
        flags.append("null_truncated_by_budget")
    if _int(rec, "n_events") < int(c["n_quality_informative"]):
        flags.append("quality_uninformative")

    hard = [f for f in HARD_VETO_ORDER if f in flags]
    if hard:
        out["first_veto"] = hard[0]
        out["tier"] = "none"
        return out

    ok_strict, why_strict = quality_pass(rec, c, strict=True)
    detail["strict_quality"] = why_strict
    if bool(rec.get("fdr_significant", False)) and ok_strict:
        complete = ("rotation_unknown" not in flags
                    and "variability_catalogue_unreached" not in flags
                    and "aperture_catalogue_unreached" not in flags
                    and "pool_null_unreached" not in flags)
        out["tier"] = "candidate" if complete else "interest"
    else:
        out["tier"] = "watch"
    return out


def assign_tiers(records: list[dict], contexts: dict | None = None,
                 conf: dict | None = None) -> list[dict]:
    """BH-FDR across every scanned star, then the gauntlet per star.

    ``contexts`` maps ``star_key`` -> context dict (see :func:`vet_star`).
    Returns new dicts (input untouched) with ``fdr_*``, ``tier``, ``first_veto``
    and ``flags`` added.
    """
    c = dict(DEFAULT_VET, **(conf or {}))
    contexts = contexts or {}
    recs = [dict(r) for r in records]
    population_period_stats(recs, c)
    scanned = [i for i, r in enumerate(recs) if r.get("status") == "scanned"]
    p = np.array([float(recs[i].get("p_window", np.nan)) for i in scanned], dtype=float)
    sig = bh_fdr(p, float(c["fdr_alpha"])) if len(p) else np.zeros(0, dtype=bool)
    watch = bh_fdr(p, float(c["fdr_alpha_watch"])) if len(p) else np.zeros(0, dtype=bool)
    for r in recs:
        r["fdr_significant"] = False
        r["fdr_watch"] = False
    for j, i in enumerate(scanned):
        recs[i]["fdr_significant"] = bool(sig[j])
        recs[i]["fdr_watch"] = bool(watch[j])
    for r in recs:
        v = vet_star(r, contexts.get(r.get("star_key"), {"mission": r.get("mission")}), c)
        r["tier"] = v["tier"]
        r["first_veto"] = v["first_veto"]
        r["flags"] = ";".join(v["flags"])
        r["veto_detail"] = v["veto_detail"]
    return recs


def rejection_counters(vetted: list[dict]) -> dict:
    """Named counters: first veto per star, every flag raised, tiers."""
    first: dict[str, int] = {}
    every: dict[str, int] = {}
    tiers: dict[str, int] = {"none": 0, "watch": 0, "interest": 0, "candidate": 0}
    for r in vetted:
        fv = r.get("first_veto") or "passed"
        first[fv] = first.get(fv, 0) + 1
        for f in str(r.get("flags", "") or "").split(";"):
            if f:
                every[f] = every.get(f, 0) + 1
        t = str(r.get("tier", "none"))
        tiers[t] = tiers.get(t, 0) + 1
    for name in HARD_VETO_ORDER + ("insufficient_events", "not_significant"):
        first.setdefault(name, 0)
    for name in HARD_VETO_ORDER + REPORT_FLAGS:
        every.setdefault(name, 0)
    return {"first_veto": first, "flags_raised": every, "tiers": tiers}


def calibrate_jitter(vetted: list[dict], conf: dict | None = None) -> dict:
    """Where the clock thresholds sit against the *natural* jitter distribution.

    The natural population is every scanned star that reached the watch FDR
    and was rejected as ``rotation_alias`` (rotational modulation, the
    dominant natural quasi-periodicity), with all scanned stars as a broader
    reference.  Reported so a reader can see the thresholds are far below what
    rotation produces rather than take it on trust.
    """
    c = dict(DEFAULT_VET, **(conf or {}))

    def _pct(vals, qs=(5, 16, 50, 84, 95)):
        v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
        if not len(v):
            return {"n": 0}
        d = {"n": int(len(v))}
        d.update({f"p{q}": float(np.percentile(v, q)) for q in qs})
        return d

    rot = [r for r in vetted if r.get("first_veto") == "rotation_alias"]
    sig = [r for r in vetted if r.get("fdr_watch")]
    allr = [r for r in vetted if r.get("status") == "scanned"]
    out = {
        "thresholds": {"jitter_max": c["jitter_max"], "jitter_watch": c["jitter_watch"],
                       "Q_min": c["Q_min"], "Q_watch": c["Q_watch"],
                       "gap_frac_min": c["gap_frac_min"]},
        "rotation_alias_population": {
            "jitter": _pct([float(r.get("jitter", np.nan)) for r in rot]),
            "Q": _pct([float(r.get("Q", np.nan)) for r in rot]),
            "gap_integer_frac": _pct([float(r.get("gap_integer_frac", np.nan)) for r in rot])},
        "watch_significant_population": {
            "jitter": _pct([float(r.get("jitter", np.nan)) for r in sig]),
            "Q": _pct([float(r.get("Q", np.nan)) for r in sig])},
        "all_scanned": {
            "jitter": _pct([float(r.get("jitter", np.nan)) for r in allr]),
            "Q": _pct([float(r.get("Q", np.nan)) for r in allr])},
    }
    jr = [float(r.get("jitter", np.nan)) for r in rot]
    jr = np.asarray([x for x in jr if np.isfinite(x)])
    out["fraction_of_rotation_population_below_jitter_max"] = (
        float((jr <= float(c["jitter_max"])).mean()) if len(jr) else float("nan"))
    qr = np.asarray([float(r.get("Q", np.nan)) for r in rot], dtype=float)
    both = np.isfinite(jr) if len(jr) == len(qr) else None
    out["fraction_of_rotation_population_inside_strict_gate"] = (
        float(((jr <= float(c["jitter_max"])) & (qr >= float(c["Q_min"]))).mean())
        if both is not None and len(jr) else float("nan"))

    # The measured look-elsewhere floor.  With a free period on a 10^4-point
    # frequency grid, a handful of event times phase up whatever they are, so
    # the fitted jitter is a function of N before it is a function of the
    # star.  Two numbers say how badly, and the second is the honest one:
    #   * the observed quality binned by N;
    #   * the quality the star's OWN window null reaches at ITS best period
    #     (wn_null_Q_median, wn_null_jitter_median) -- the same fit on times
    #     that carry no clock at all.
    # A gate the null routinely passes is not a gate.
    def _med(sel, key):
        v = np.asarray([_f(r, key) for r in sel], dtype=float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else float("nan")

    def _bin_stats(lo, hi):
        sel = [r for r in allr if lo <= int(r.get("n_events", 0) or 0) < hi
               and np.isfinite(_f(r, "wn_null_jitter_median"))]
        if not sel:
            return None
        return {"n": len(sel), "jitter_p50": _med(sel, "jitter"),
                "jitter_null_p50": _med(sel, "wn_null_jitter_median"),
                "Q_p50": _med(sel, "Q"), "Q_null_p50": _med(sel, "wn_null_Q_median")}

    edges = [8, 12, 16, 24, 40, 80, 10 ** 9]
    out["by_n_events"] = {f"{a}-{b if b < 10 ** 9 else 'inf'}": _bin_stats(a, b)
                          for a, b in zip(edges[:-1], edges[1:], strict=True)}
    nulls_j = np.asarray([_f(r, "wn_null_jitter_median") for r in allr], dtype=float)
    nulls_q = np.asarray([_f(r, "wn_null_Q_median") for r in allr], dtype=float)
    ok = np.isfinite(nulls_j) & np.isfinite(nulls_q)
    out["window_null_own_best_fit"] = {
        "n": int(ok.sum()),
        "jitter_p50": float(np.median(nulls_j[ok])) if ok.any() else float("nan"),
        "Q_p50": float(np.median(nulls_q[ok])) if ok.any() else float("nan"),
        "fraction_inside_strict_gate": float(
            ((nulls_j[ok] <= float(c["jitter_max"]))
             & (nulls_q[ok] >= float(c["Q_min"]))).mean()) if ok.any() else float("nan"),
    }
    out["n_quality_informative"] = float(c["n_quality_informative"])
    out["note"] = (
        "the fitted jitter falls with N because the period is free: read "
        "by_n_events against window_null_own_best_fit before reading any "
        "quality number as physics.  Stars below n_quality_informative carry "
        "the report flag `quality_uninformative` and rest on the null alone")
    return out


__all__ = ["DEFAULT_VET", "HARD_VETO_ORDER", "REPORT_FLAGS", "assign_tiers",
           "cadence_alias", "calibrate_jitter", "core_pass", "periodic_variable",
           "aperture_contamination",
           "population_period_stats", "quality_pass", "rejection_counters",
           "rotation_alias", "vet_star"]
