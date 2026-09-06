"""The contamination gauntlet and the tier assignment --- pure functions.

Every rejection is a *named mechanism* with its own counter, applied in the
order most-mundane-first so a star that trips several is reported under the
dullest one:

``insufficient_events``     fewer than ``n_min`` events after declustering and
                            cross-star removal (the cross-star removal itself
                            is counted at event level in the screen stage)
``not_significant``         the window-resampled null explains the coherence
                            (BH-FDR across every star scanned)
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
``bursty_random``           the shuffle null does not beat the observed H
                            (coherence explained by the waiting-time
                            distribution) AND the waiting times are not
                            integer periods --- bursty, not clocked
``jitter_too_large``        the phase concentration fails even the loose
                            ("watch") clock thresholds

Report-only flags never reject: ``energy_incoherent`` (flare energy depends
on clock phase --- what visibility modulation does and a beacon should not),
``rotation_unknown``, ``variability_catalogue_unreached``, ``p_extrapolated``,
``null_truncated_by_budget``.

Tiers
-----
``none``       not significant at the watch FDR, or a hard veto tripped
``watch``      significant at ``fdr_alpha_watch``, no hard veto, loose quality
``interest``   significant at ``fdr_alpha``, strict quality, but a veto could
               not be applied (no P_rot, or a variability catalogue was not
               reached) --- candidate-grade statistics with an incomplete vet
``candidate``  significant at ``fdr_alpha``, strict quality, every veto
               applied and passed.  PENDING human/light-curve vet always.
"""

from __future__ import annotations

import numpy as np

from .clock import bh_fdr

DEFAULT_VET: dict = {
    "rotation_ratios": [1.0, 0.5, 1.0 / 3.0, 0.25, 2.0, 3.0],
    "rotation_tol": 0.03,
    "cadence_harmonics": [1.0, 2.0, 3.0, 0.5, 1.0 / 3.0],
    "cadence_tol": 0.02,
    "variable_harmonics": [1.0, 0.5, 1.0 / 3.0, 2.0, 3.0],
    "variable_tol": 0.03,
    "fdr_alpha": 0.05,
    "fdr_alpha_watch": 0.25,
    "shuffle_alpha": 0.05,
    "gap_frac_min": 0.6,
    "gap_min_count": 4,
    "Q_min": 0.85,
    "jitter_max": 0.05,
    "Q_watch": 0.6,
    "jitter_watch": 0.12,
    "energy_p_max": 0.01,
    "instrumental_periods": {
        "kepler": {"long_cadence": 0.020434, "momentum_dump": 3.0,
                   "monthly_downlink": 31.0, "quarter": 93.0},
        "tess": {"cadence_2min": 0.0013889, "cadence_10min": 0.0069444,
                 "cadence_200s": 0.0023148, "ffi_30min": 0.0208333,
                 "momentum_dump_early": 3.5, "orbit": 13.7, "sector": 27.4},
    },
}

HARD_VETO_ORDER = ("cadence_alias", "rotation_alias", "periodic_variable",
                   "bursty_random", "jitter_too_large")
REPORT_FLAGS = ("energy_incoherent", "rotation_unknown", "variability_catalogue_unreached",
                "p_extrapolated", "null_truncated_by_budget")


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


def quality_pass(rec: dict, conf: dict, *, strict: bool) -> tuple[bool, list[str]]:
    """Does the phase concentration meet the clock threshold (strict or watch)?"""
    q, j = float(rec.get("Q", np.nan)), float(rec.get("jitter", np.nan))
    why = []
    if strict:
        if not (np.isfinite(q) and q >= float(conf["Q_min"])):
            why.append(f"Q<{conf['Q_min']}")
        if not (np.isfinite(j) and j <= float(conf["jitter_max"])):
            why.append(f"jitter>{conf['jitter_max']}")
        gf, ng = float(rec.get("gap_integer_frac", np.nan)), int(rec.get("n_gaps_used", 0) or 0)
        if ng >= int(conf["gap_min_count"]):
            if not (np.isfinite(gf) and gf >= float(conf["gap_frac_min"])):
                why.append(f"gap_integer_frac<{conf['gap_frac_min']}")
        else:
            why.append("gap_integer_frac_unmeasurable")
    else:
        if not (np.isfinite(q) and q >= float(conf["Q_watch"])):
            why.append(f"Q<{conf['Q_watch']}")
        if not (np.isfinite(j) and j <= float(conf["jitter_watch"])):
            why.append(f"jitter>{conf['jitter_watch']}")
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

    p_sh = float(rec.get("p_shuffle", np.nan))
    gf = float(rec.get("gap_integer_frac", np.nan))
    ng = int(rec.get("n_gaps_used", 0) or 0)
    if (np.isfinite(p_sh) and p_sh >= float(c["shuffle_alpha"])
            and ng >= int(c["gap_min_count"]) and np.isfinite(gf)
            and gf < float(c["gap_frac_min"])):
        flags.append("bursty_random")
        detail["bursty_random"] = {"p_shuffle": p_sh, "gap_integer_frac": gf}

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

    hard = [f for f in HARD_VETO_ORDER if f in flags]
    if hard:
        out["first_veto"] = hard[0]
        out["tier"] = "none"
        return out

    ok_strict, why_strict = quality_pass(rec, c, strict=True)
    detail["strict_quality"] = why_strict
    if bool(rec.get("fdr_significant", False)) and ok_strict:
        complete = ("rotation_unknown" not in flags
                    and "variability_catalogue_unreached" not in flags)
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
    return out


__all__ = ["DEFAULT_VET", "HARD_VETO_ORDER", "REPORT_FLAGS", "assign_tiers",
           "cadence_alias", "calibrate_jitter", "periodic_variable", "quality_pass",
           "rejection_counters", "rotation_alias", "vet_star"]
