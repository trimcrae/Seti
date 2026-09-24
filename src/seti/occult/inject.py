"""Injection of opaque-lens signals into REAL event photometry, and analytic sensitivity.

An injection keeps the event's real cadence, gaps, noise, systematics and
any real anomaly, and adds only the difference an opaque lens of radius
rho_L would make at the event's own fitted shape:

    f_inj = f + fs_k [A_occ(u; rho_L, rho_*) - A_FSPL(u; rho_*)]

with (t0, tE, u0, rho_*, fs_k) from the event's FSPL fit.  The injected
event then goes through the whole funnel (re-cleaned, re-fitted, gated);
it counts as recovered only if it comes out ``OCCULTATION_CANDIDATE`` with
the right regime and rho_L within ``tol`` --- the same bar a real signal
has to clear.
"""

from __future__ import annotations

import math

import numpy as np

from . import detect as D

# rho_L grid of the sensitivity map: the wing regime where a step is
# measurable (u_c from ~3.3 down to ~0.1) and the central-hole regime.
RHO_L_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.05, 1.3, 2.0)
U0_EDGES = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5)


def inject(ev: D.Event, fit: D.Fit, rho_l: float, ns: int = 64) -> D.Event:
    """``ev`` with an opaque lens of radius ``rho_l`` added at ``fit``'s shape."""
    u = D.traj_fit(ev, fit)
    a0 = D.mag_fspl(u, fit.rho, ns)
    a1 = D.mag_occult(u, rho_l, fit.rho, ns)
    f = ev.f + fit.fs[ev.ds] * (a1 - a0)
    return D.Event(ev.name + f"+inj{rho_l:g}", ev.t, f, ev.e, ev.ds, ev.datasets, dict(ev.meta))


def recovered(rec: dict, rho_l: float, tol: float = 0.05, tol_hole: float = 0.15) -> bool:
    """Candidate tier, right regime, rho_l within tolerance.

    The wing regime pins rho_l through the step times (tol 5 %); in the
    central-hole regime rho_l trades against tE and u0 (the horns constrain a
    combination), and the offline battery shows 5-8 % scatter at KMTNet-like
    cadence, so the bar there is 15 %.
    """
    if rec.get("tier") != D.TIER_CANDIDATE:
        return False
    got = (rec.get("occult") or {}).get("rho_l")
    if got is None:
        return False
    if (got < 1.0) != (rho_l < 1.0):
        return False
    return abs(got - rho_l) <= (tol if rho_l < 1.0 else tol_hole) * rho_l


def expected_map(ev: D.Event, fit: D.Fit, e=None, grid=RHO_L_GRID) -> dict:
    """Fisher-style expected Delta chi^2 for each rho_L on the grid (0 where u0 >= u_c)."""
    out = {}
    for rl in grid:
        uc = D.u_crit(rl)
        if rl < 1.0 and D.u_min_fit(ev, fit) >= uc:
            out[f"{rl:g}"] = 0.0          # blend-degenerate: minor image hidden all the time
            continue
        out[f"{rl:g}"] = round(D.expected_dchi2(ev, fit, rl, e), 2)
    return out


def injection_trials(ev_raw: D.Event, conf: dict, hint: dict | None, rho_ls, seed: int = 0,
                     tol: float = 0.05) -> list:
    """Run the funnel on ``ev_raw`` with each rho_L injected; one record per trial.

    The injection is made into the cleaned, error-renormalised event at its
    own FSPL solution; the injected event is then re-assessed from scratch
    (fresh clipping, renormalisation and fits).
    """
    conf = D.conf_with(conf)
    ev2, e, f0, info = D.fit_fspl_clean(ev_raw, conf, hint)
    if ev2 is None:
        return []
    raw_e = ev2.e
    out = []
    for rl in rho_ls:
        uc = D.u_crit(rl)
        exp = 0.0 if (rl < 1.0 and D.u_min_fit(ev2, f0) >= uc) else D.expected_dchi2(ev2, f0, rl, raw_e)
        inj = inject(ev2, f0, rl, int(conf["n_samples"]))
        try:
            rec = D.assess_event(inj, conf, hint={"t0": f0.t0, "tE": f0.te, "u0": f0.u0})
        except Exception as exc:  # noqa: BLE001
            rec = {"tier": "ERROR", "error": repr(exc)[:200]}
        out.append({
            "event": ev_raw.name, "rho_l_inj": rl, "u0": D.u_min_fit(ev2, f0), "tE": f0.te,
            "parallax": bool(f0.parallax),
            "u_c": uc, "expected_dchi2": round(exp, 2),
            "tier": rec.get("tier"), "rho_l_fit": (rec.get("occult") or {}).get("rho_l"),
            "dchi2": rec.get("dchi2"), "rejections": rec.get("rejections", []),
            "recovered": recovered(rec, rl, tol),
        })
    return out


def efficiency_table(trials: list, grid=RHO_L_GRID, u0_edges=U0_EDGES) -> dict:
    """Recovered fraction per (rho_L, u0) cell, and the sensitive fraction of the plane."""
    cells = {}
    for rl in grid:
        for i in range(len(u0_edges) - 1):
            lo, hi = u0_edges[i], u0_edges[i + 1]
            sel = [t for t in trials if abs(t["rho_l_inj"] - rl) < 1e-9 and lo <= t["u0"] < hi]
            n = len(sel)
            k = sum(1 for t in sel if t["recovered"])
            cells[f"{rl:g}|{lo:g}-{hi:g}"] = {"n": n, "k": k, "eff": (k / n) if n else None}
    measured = [c for c in cells.values() if c["n"] >= 5]
    sens = [c for c in measured if c["eff"] >= 0.5]
    return {"cells": cells, "n_cells": len(cells), "n_cells_measured": len(measured),
            "n_cells_sensitive_eff50": len(sens),
            "fraction_of_measured_plane_sensitive": (len(sens) / len(measured)) if measured else None,
            "fraction_of_whole_plane_sensitive": len(sens) / len(cells) if cells else None,
            "overall_recovered": sum(1 for t in trials if t["recovered"]),
            "overall_trials": len(trials)}


def efficiency_vs_expected(trials: list, edges=(0, 25, 50, 100, 200, 500, 1000, 1e4, 1e9)) -> list:
    """Recovery fraction binned by the Fisher-expected Delta chi^2 (calibrates the analytic map)."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = [t for t in trials if lo <= t["expected_dchi2"] < hi]
        out.append({"lo": lo, "hi": hi if math.isfinite(hi) else None, "n": len(sel),
                    "eff": (sum(t["recovered"] for t in sel) / len(sel)) if sel else None})
    return out


def pick_rho_ls(seed: int, n: int = 3, grid=RHO_L_GRID) -> list:
    rng = np.random.default_rng(seed)
    return [float(x) for x in rng.choice(np.array(grid), size=min(n, len(grid)), replace=False)]
