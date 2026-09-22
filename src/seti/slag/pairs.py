"""Tier 2: the process-orthogonal pair residual, the refinery flags, the kills.

A pair (a, b) is *process-orthogonal* when the four natural levers barely
move log(a/b): the natural ENVELOPE of the ratio --- extreme end-members,
condensation fractionation below ``t_cut_max``, and the sinking phases with
the timescale scatter --- is narrow.  ``z_pair`` is the distance of the
observed ratio OUTSIDE that envelope in units of
sqrt(sigma_a^2 + sigma_b^2 + sigma_sys^2); inside the envelope it is zero.

The sinking phase is ONE number per object.  So the phase shift a pair is
allowed is not the extreme over every phase, it is the extreme over the
phases the REST of the panel accepts (the acceptable phase set of the Tier 1
fit of the panel without the pair) --- the argument Doyle+2021 made against
sinking for the beryllium objects, made mechanical.  A panel too small to
constrain its phase (fewer than three other elements) gets the full grid and
says so.

The envelope width is REPORTED per pair, because the brief's claim that all
six pairs stay within ~0.3 dex is not what the compilations say: Ti/Al, Ca/Al
and Sc/Ca are tight, but Sr/Ca, Mn/Cr and Ni/Co open up by 1-1.5 dex once
continental crust and core-formation residues are in the family (the
incompatible-element and metal-partitioning levers).  A wide envelope makes a
pair a weak test, not a wrong one, and the number is in the record.

Flags are the brief's direct refinery vectors; each is a comparison against
the same envelope machinery with the same rest-of-panel phase set.  Kills
are recorded per candidate and never silently applied: a killed candidate
stays in the table with its kill.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .family import NaturalFamily, combined_ratio_envelope
from .misfit import FitResult, FitSettings, Panel, PanelModel, fit_panel, naive_p
from .sinking import TimescaleModel, phase_grid, phase_log_factor

ORTHOGONAL_PAIRS = [("Ti", "Al"), ("Sc", "Ca"), ("Ca", "Al"), ("Sr", "Ca"), ("Mn", "Cr"),
                    ("Ni", "Co")]

#: Elements whose cool-helium line physics (resonance lines, pressure
#: broadening by neutral He) is where the models are weakest.
COOL_HE_ELEMENTS = {"Ca", "Na", "K", "Li"}

KILL_INFORMATION_LIMITED = "INFORMATION_LIMITED"
KILL_REST_NOT_NATURAL = "REST_OF_PANEL_NOT_NATURAL"
KILL_COOL_HE = "COOL_HE_LINE_PHYSICS"
KILL_ASYNC = "ASYNCHRONOUS_ACCRETION_EXPLAINS"
KILL_CARBONATE = "CARBONATE_CRUST_POSSIBLE"
KILL_MULTI_REF = "MULTI_REFERENCE_DISAGREEMENT"
KILL_FREE_FRACTIONATION = "REFRACTORY_FRACTIONATION_EXPLAINS"
KILL_REAL_METEORITE = "A_REAL_METEORITE_DOES_THIS"
CAVEAT_EXOTIC_MANTLE = "EXOTIC_MANTLE_POSSIBLE"
PHASE_UNCONSTRAINED = "unconstrained_full_grid"
PHASE_FROM_REST = "rest_of_panel_fit"

FLAG_ALLOY = "ALLOY_TI_AL_V_WITHOUT_FE"
FLAG_PURE_SI = "PURE_SI_WITHOUT_O_MG_FE"
FLAG_CU_ZN_SN_PB = "CU_ZN_SN_PB_ENRICHED"
FLAG_BE = "BE_WITHOUT_LI_B"
FLAG_BE_UNMEASURED = "BE_ENRICHED_PARTNERS_UNMEASURED"
FLAG_FE_MG = "FE_MG_BELOW_MANTLE"

MIN_REST_FOR_PHASE = 3


# ---------------------------------------------------------------------------
# the phase set of a panel subset
# ---------------------------------------------------------------------------
def rest_phase_set(fam: NaturalFamily, tsm: TimescaleModel, panel: Panel, drop: list[str],
                   settings: FitSettings) -> tuple[list[tuple[float, float]], str, FitResult | None]:
    """Acceptable (t_acc, t_dec) from the panel without ``drop``; the full grid if too small."""
    rest = panel.subset(drop)
    if rest.n_measured < MIN_REST_FOR_PHASE:
        return phase_grid(settings.t_acc_range, settings.t_dec_range), PHASE_UNCONSTRAINED, None
    fit = fit_panel(fam, rest, tsm, settings)
    phases = list(fit.phase_set) or [(fit.t_acc, fit.t_dec)]
    return phases, PHASE_FROM_REST, fit


def phase_shift_range(tsm: TimescaleModel, a: str, b: str, atmosphere: str, teff, logg,
                      settings: FitSettings, *, phases=None, row_tau=None
                      ) -> tuple[float, float]:
    """Min / max of the sinking shift of log(a/b) over ``phases`` and the pair's tau scatter.

    The scatter is applied to the pair's timescale DIFFERENCE (the model's
    per-atmosphere sigma), not independently to each element: what the
    tables do not pin down is how far apart two neighbouring elements sink,
    and that is one number.
    """
    lt, st, _ = tsm.log_tau_rel([a, b], atmosphere, teff, logg, row_tau=row_tau)
    sig_pair = float(max(st.max(), 0.0))
    if phases is None:
        phases = phase_grid(settings.t_acc_range, settings.t_dec_range)
    lo, hi = np.inf, -np.inf
    for d in (-1.0, 0.0, 1.0):
        ltv = lt + np.array([0.5, -0.5]) * d * sig_pair
        for t_acc, t_dec in phases:
            ph = phase_log_factor(ltv, t_acc, t_dec)
            v = float(ph[0] - ph[1])
            lo, hi = min(lo, v), max(hi, v)
    return float(lo), float(hi)


def natural_interval(fam: NaturalFamily, tsm: TimescaleModel, a: str, b: str, panel: Panel,
                     settings: FitSettings, *, t_cut_max: float = 1400.0, endmembers=None,
                     phases=None, meteorites=None) -> dict:
    """The natural envelope of log(a/b) for this atmosphere, with the phase shift folded in.

    ``meteorites`` is PEWDD's measured compilation; when it constrains the
    pair the envelope is the union of the end-member model and the measured
    spread (``family.combined_ratio_envelope``).  Restricting to an end-member
    subset (the mantle-only Fe/Mg floor) is a statement about the MODEL, so
    the measured envelope is not unioned in that case.
    """
    env = combined_ratio_envelope(fam, a, b, endmembers=endmembers, t_cut_max=t_cut_max,
                                  meteorites=None if endmembers else meteorites)
    if not np.isfinite(env["lo"]):
        return {**env, "phase_lo": 0.0, "phase_hi": 0.0, "lo_total": np.nan, "hi_total": np.nan}
    plo, phi = phase_shift_range(tsm, a, b, panel.atmosphere, panel.teff, panel.logg, settings,
                                 phases=phases, row_tau=panel.meta.get("sinking_times_s"))
    return {**env, "phase_lo": plo, "phase_hi": phi, "lo_total": env["lo"] + plo,
            "hi_total": env["hi"] + phi}


def _z_outside(obs: float, lo: float, hi: float, sigma: float) -> float:
    """Signed distance outside [lo, hi] in sigma; 0 inside; an open end never bounds."""
    if np.isnan(lo) or np.isnan(hi):
        return float("nan")
    if np.isfinite(hi) and obs > hi:
        return float((obs - hi) / sigma)
    if np.isfinite(lo) and obs < lo:
        return float((obs - lo) / sigma)
    return 0.0


# ---------------------------------------------------------------------------
# pair residuals
# ---------------------------------------------------------------------------
def pair_residuals(fam: NaturalFamily, tsm: TimescaleModel, panel: Panel,
                   settings: FitSettings, *, pairs=None, t_cut_max: float = 1400.0,
                   z_threshold: float = 4.0, rest_p_min: float = 0.01,
                   meteorites=None) -> list[dict]:
    """Every orthogonal pair both of whose elements this panel measures."""
    out = []
    idx = {e: i for i, e in enumerate(panel.elements)}
    for a, b in (pairs or ORTHOGONAL_PAIRS):
        if a not in idx or b not in idx:
            continue
        obs = float(panel.values[idx[a]] - panel.values[idx[b]])
        sig = float(np.sqrt(panel.errors[idx[a]] ** 2 + panel.errors[idx[b]] ** 2
                            + settings.sigma_sys ** 2))
        phases, how, rest_fit = rest_phase_set(fam, tsm, panel, [a, b], settings)
        env = natural_interval(fam, tsm, a, b, panel, settings, t_cut_max=t_cut_max,
                               phases=phases, meteorites=meteorites)
        z = _z_outside(obs, env["lo_total"], env["hi_total"], sig)
        met = env.get("meteorite") or {}
        rec = {"name": panel.name, "reference": panel.reference, "pair": f"{a}/{b}", "a": a,
               "b": b, "obs_log_ratio": obs, "sigma": sig, "env_lo": env["lo"],
               "env_hi": env["hi"], "phase_lo": env["phase_lo"], "phase_hi": env["phase_hi"],
               "env_lo_total": env["lo_total"], "env_hi_total": env["hi_total"],
               "env_width_dex": env.get("width_dex"), "z_pair": z,
               "env_source": env.get("source"),
               "env_model_lo": env.get("model_lo"), "env_model_hi": env.get("model_hi"),
               "met_lo": met.get("lo"), "met_hi": met.get("hi"), "met_n": met.get("n"),
               "met_min": met.get("min"), "met_max": met.get("max"),
               "met_fraction_outside_model": met.get("fraction_outside_model"),
               "exceeds": bool(np.isfinite(z) and abs(z) > z_threshold),
               "n_measured": panel.n_measured, "rest_n": panel.n_measured - 2,
               "phase_constraint": how, "n_phases": len(phases),
               "rest_chi2": None, "rest_p_naive": None, "rest_natural": None}
        if rest_fit is not None:
            rec["rest_chi2"] = float(rest_fit.chi2)
            rec["rest_p_naive"] = naive_p(rest_fit.chi2, rest_fit.n_measured - 1)
            rec["rest_natural"] = bool(rec["rest_p_naive"] > rest_p_min)
            rec["rest_phase"] = rest_fit.phase
            rec["rest_t_dec_max"] = float(max(p[1] for p in phases))
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# refinery flags
# ---------------------------------------------------------------------------
def _obs_ratio(panel: Panel, a: str, b: str) -> tuple[float, float, str] | None:
    """(log a/b, sigma, kind) where kind says which side is a limit; None if unconstrained."""
    idx = {e: i for i, e in enumerate(panel.elements)}
    lidx = {e: i for i, e in enumerate(panel.limit_elements)}
    if a in idx and b in idx:
        return (float(panel.values[idx[a]] - panel.values[idx[b]]),
                float(np.hypot(panel.errors[idx[a]], panel.errors[idx[b]])), "measured")
    if a in idx and b in lidx:          # b is an upper limit: the ratio is a LOWER limit
        return (float(panel.values[idx[a]] - panel.limit_values[lidx[b]]),
                float(panel.errors[idx[a]]), "lower_limit")
    if a in lidx and b in idx:          # a is an upper limit: the ratio is an UPPER limit
        return (float(panel.limit_values[lidx[a]] - panel.values[idx[b]]),
                float(panel.errors[idx[b]]), "upper_limit")
    return None


def _above_env(fam, tsm, panel, settings, a, b, *, nsig, t_cut_max, phases, endmembers=None,
               meteorites=None) -> dict | None:
    """Is log(a/b) above the natural envelope by ``nsig``?  None if unconstrained."""
    r = _obs_ratio(panel, a, b)
    if r is None or r[2] == "upper_limit":
        return None
    obs, sig, kind = r
    sig = float(np.hypot(sig, settings.sigma_sys))
    env = natural_interval(fam, tsm, a, b, panel, settings, t_cut_max=t_cut_max,
                           endmembers=endmembers, phases=phases, meteorites=meteorites)
    z = _z_outside(obs, env["lo_total"], env["hi_total"], sig)
    return {"ratio": f"{a}/{b}", "obs": obs, "sigma": sig, "kind": kind,
            "env_hi_total": env["hi_total"], "env_lo_total": env["lo_total"], "z": z,
            "env_source": env.get("source"),
            "above": bool(np.isfinite(z) and z > nsig)}


def _below_env(fam, tsm, panel, settings, a, b, *, nsig, t_cut_max, phases, margin=0.0,
               endmembers=None, meteorites=None) -> dict | None:
    r = _obs_ratio(panel, a, b)
    if r is None or r[2] == "lower_limit":
        return None
    obs, sig, kind = r
    sig = float(np.hypot(sig, settings.sigma_sys))
    env = natural_interval(fam, tsm, a, b, panel, settings, t_cut_max=t_cut_max,
                           endmembers=endmembers, phases=phases, meteorites=meteorites)
    lo = env["lo_total"] - margin
    z = _z_outside(obs, lo, env["hi_total"], sig)
    return {"ratio": f"{a}/{b}", "obs": obs, "sigma": sig, "kind": kind,
            "env_lo_total": lo, "env_hi_total": env["hi_total"], "z": z,
            "below": bool(np.isfinite(z) and z < -nsig)}


def refinery_flags(fam: NaturalFamily, tsm: TimescaleModel, panel: Panel, settings: FitSettings,
                   *, nsig: float = 4.0, t_cut_max: float = 1400.0,
                   exotic_mantle_margin_dex: float = 0.5, meteorites=None,
                   mantle_members=("mantle_BSE", "Moon_BSM", "Mars_BSM", "crust_cont",
                                   "Vesta_eucrite")) -> list[dict]:
    """The brief's direct refinery vectors, each as a comparison against the natural envelope."""
    flags: list[dict] = []
    meas = set(panel.elements)
    lims = set(panel.limit_elements)
    refs_lith = [e for e in ("Ca", "Mg", "Si") if e in meas]
    refs_all = [e for e in ("Fe", "Ca", "Mg", "Si") if e in meas]

    def phases_without(drop):
        ph, how, _ = rest_phase_set(fam, tsm, panel, list(drop), settings)
        return ph, how

    # alloy: Ti and Al (V optional) far above any natural (Ti+Al)/Fe
    if {"Ti", "Al"} <= meas and ("Fe" in meas or "Fe" in lims):
        i = {e: k for k, e in enumerate(panel.elements)}
        num = np.log10(10.0 ** panel.values[i["Ti"]] + 10.0 ** panel.values[i["Al"]])
        if "Fe" in meas:
            obs = float(num - panel.values[i["Fe"]])
            sig, kind = float(np.hypot(panel.errors[i["Al"]], panel.errors[i["Fe"]])), "measured"
        else:
            j = panel.limit_elements.index("Fe")
            obs, sig, kind = float(num - panel.limit_values[j]), float(panel.errors[i["Al"]]), \
                "lower_limit"
        sig = float(np.hypot(sig, settings.sigma_sys))
        vals = []
        for e in fam.endmembers:
            if all(fam.has(x, e) for x in ("Ti", "Al", "Fe")):
                c = fam.column(e)
                vals.append(np.log10((c[fam.index("Ti")] + c[fam.index("Al")])
                                     / c[fam.index("Fe")]))
        phases, how = phases_without(["Ti", "Al", "V", "Fe"])
        _plo, phi = phase_shift_range(tsm, "Al", "Fe", panel.atmosphere, panel.teff, panel.logg,
                                      settings, phases=phases,
                                      row_tau=panel.meta.get("sinking_times_s"))
        hi = float(max(vals)) + phi + 0.3   # +0.3 dex: refractory-enriched natural rocks (CAI-rich)
        z = _z_outside(obs, -np.inf, hi, sig)
        flags.append({"flag": FLAG_ALLOY, "fired": bool(np.isfinite(z) and z > nsig),
                      "ratio": "(Ti+Al)/Fe", "obs": obs, "sigma": sig, "kind": kind,
                      "env_hi_total": hi, "z": z, "has_V": "V" in meas, "phase_constraint": how})

    # pure Si: Mg, O and Fe all below the natural floor relative to Si
    if "Si" in meas:
        phases, how = phases_without(["Si", "Mg", "O", "Fe"])
        parts = {}
        for x in ("Mg", "O", "Fe"):
            r = _below_env(fam, tsm, panel, settings, x, "Si", nsig=nsig, t_cut_max=t_cut_max,
                           phases=phases, meteorites=meteorites)
            if r is not None:
                parts[x] = r
        if len(parts) == 3:
            fired = all(r["below"] for r in parts.values())
            flags.append({"flag": FLAG_PURE_SI, "fired": bool(fired), "parts": parts,
                          "z": float(max(r["z"] for r in parts.values())),
                          "phase_constraint": how})
        elif parts:
            flags.append({"flag": FLAG_PURE_SI, "fired": False, "parts": parts,
                          "note": "Mg, O and Fe must all be constrained relative to Si",
                          "z": float("nan"), "phase_constraint": how})

    # Cu / Zn / Sn / Pb enriched relative to EVERY measured major reference
    for x in ("Cu", "Zn", "Sn", "Pb"):
        if x in meas and refs_all:
            phases, how = phases_without([x])
            parts = {}
            for ref in refs_all:
                r = _above_env(fam, tsm, panel, settings, x, ref, nsig=nsig, t_cut_max=t_cut_max,
                               phases=phases, meteorites=meteorites)
                if r is not None:
                    parts[ref] = r
            if parts:
                flags.append({"flag": FLAG_CU_ZN_SN_PB, "element": x,
                              "fired": bool(all(r["above"] for r in parts.values())),
                              "parts": parts, "z": float(min(r["z"] for r in parts.values())),
                              "phase_constraint": how})

    # Be enriched, and its spallation partners Li / B absent
    if "Be" in meas and (refs_lith or "Fe" in meas):
        phases, how = phases_without(["Be", "Li", "B"])
        parts = {}
        for ref in (refs_lith or ["Fe"]):
            r = _above_env(fam, tsm, panel, settings, "Be", ref, nsig=nsig, t_cut_max=t_cut_max,
                           phases=phases, meteorites=meteorites)
            if r is not None:
                parts[ref] = r
        enriched = bool(parts) and all(r["above"] for r in parts.values())
        partner = {}
        for p in ("Li", "B"):
            r = _obs_ratio(panel, p, "Be")
            if r is None:
                continue
            obs, sig, kind = r
            sig = float(np.hypot(sig, settings.sigma_sys))
            # spallation makes Li and B alongside Be at order-unity ratios (Doyle+2021);
            # a partner BELOW Be by nsig is the refinery reading
            partner[p] = {"obs_log_partner_over_Be": obs, "sigma": sig, "kind": kind,
                          "absent": bool(kind != "lower_limit" and obs < -nsig * sig)}
        zmin = float(min(r["z"] for r in parts.values())) if parts else float("nan")
        if enriched:
            if partner and all(v["absent"] for v in partner.values()):
                flags.append({"flag": FLAG_BE, "fired": True, "parts": parts, "partners": partner,
                              "z": zmin, "phase_constraint": how})
            elif not partner:
                flags.append({"flag": FLAG_BE_UNMEASURED, "fired": False, "parts": parts,
                              "z": zmin, "phase_constraint": how,
                              "note": "Be enriched; Li and B unconstrained, so spallation is untested"})
            else:
                flags.append({"flag": FLAG_BE, "fired": False, "parts": parts, "partners": partner,
                              "z": zmin, "phase_constraint": how,
                              "note": "Be enriched with a spallation partner present: natural (spallation)"})
        elif parts:
            flags.append({"flag": FLAG_BE, "fired": False, "parts": parts, "z": zmin,
                          "phase_constraint": how, "note": "Be within the natural envelope"})

    # Fe/Mg below the most depleted natural mantle (with the exotic-mantle margin)
    if "Mg" in meas and ("Fe" in meas or "Fe" in lims):
        phases, how = phases_without(["Fe", "Mg"])
        r = _below_env(fam, tsm, panel, settings, "Fe", "Mg", nsig=nsig, t_cut_max=t_cut_max,
                       margin=exotic_mantle_margin_dex, endmembers=list(mantle_members),
                       phases=phases)
        if r is not None:
            flags.append({"flag": FLAG_FE_MG, "fired": bool(r["below"]), **r,
                          "caveat": CAVEAT_EXOTIC_MANTLE, "phase_constraint": how,
                          "exotic_mantle_margin_dex": exotic_mantle_margin_dex})
    return flags


# ---------------------------------------------------------------------------
# kills
# ---------------------------------------------------------------------------
def fit_two_parcel(fam: NaturalFamily, panel: Panel, tsm: TimescaleModel,
                   settings: FitSettings, *, rng=None, n_pairs: int = 200,
                   maxiter: int = 800) -> dict:
    """Brouwers+2022 asynchronous accretion: two natural parcels, each with its own phase.

    The photosphere is log10(10^m1 + 10^(m2 + delta)) + c.  If this explains
    the panel where one parcel did not, the anomaly is a time-variable
    composition inside one event, not a refinery.
    """
    rng = np.random.default_rng(settings.seed + 7) if rng is None else rng
    pm = PanelModel(fam, panel, tsm, settings)
    n_par = pm.n_par

    def model_of(theta):
        t1, t2, delta = theta[:n_par], theta[n_par: 2 * n_par], float(theta[-1])
        m1, g1 = pm.predict(*pm.unpack(t1))
        m2, g2 = pm.predict(*pm.unpack(t2))
        m = np.logaddexp(m1 * np.log(10.0), (m2 + delta) * np.log(10.0)) / np.log(10.0)
        f1 = 1.0 / (1.0 + 10.0 ** (m2 + delta - m1))
        g = f1 * g1 + (1 - f1) * g2
        return m, g

    def objective(theta):
        m, g = model_of(theta)
        sig2 = pm.sig_meas2 + (g[: pm.n_meas] * pm.sig_tau[: pm.n_meas]) ** 2
        wt = 1.0 / sig2
        c = float(np.sum(wt * (pm.y - m[: pm.n_meas])) / np.sum(wt))
        chi2 = float(np.sum((pm.y - m[: pm.n_meas] - c) ** 2 / sig2))
        nll2 = chi2 + float(np.sum(np.log(sig2)))
        if len(pm.limits):
            viol = m[pm.n_meas:] + c - pm.limits
            pen = float(np.sum(np.maximum(viol, 0.0) ** 2) / settings.limit_sigma ** 2)
            chi2 += pen
            nll2 += pen
        return nll2, chi2

    starts = pm.random_thetas(rng, max(n_pairs, 2 * pm.n_end))
    k = starts.shape[0]
    i1 = rng.integers(0, k, size=n_pairs)
    i2 = rng.integers(0, k, size=n_pairs)
    deltas = rng.uniform(-2.0, 2.0, size=n_pairs)
    best, best_theta = np.inf, None
    for a, b, d in zip(i1, i2, deltas, strict=True):
        th = np.concatenate([starts[a], starts[b], [d]])
        v = objective(th)[0]
        if v < best:
            best, best_theta = v, th
    r = minimize(lambda th: objective(th)[0], best_theta, method="Nelder-Mead",
                 options={"maxiter": int(maxiter), "xatol": 1e-3, "fatol": 1e-4, "adaptive": True})
    theta = np.asarray(r.x) if float(r.fun) < best else best_theta
    nll2, chi2 = objective(theta)
    return {"nll2_two_parcel": float(nll2), "chi2_two_parcel": float(chi2),
            "p_naive_two_parcel": naive_p(chi2, pm.n_meas - 1)}


def apply_kills(panel: Panel, pair: dict, fit_full: FitResult | None, fit_restricted: FitResult,
                misfit_p: dict | None, two_parcel: dict | None, *, teff_cool_he: float = 7000.0,
                other_panels: list[Panel] | None = None, multi_ref_sigma: float = 3.0,
                min_elements: int = 5, rest_p_min: float = 0.01,
                free_fractionation_p_min: float = 0.05) -> list[str]:
    """The kill ledger for one pair candidate.  Order is not priority; all are recorded."""
    kills: list[str] = []
    if panel.n_measured < min_elements:
        kills.append(KILL_INFORMATION_LIMITED)
    if pair.get("rest_p_naive") is not None and pair["rest_p_naive"] <= rest_p_min:
        kills.append(KILL_REST_NOT_NATURAL)
    if str(panel.atmosphere).upper().startswith("HE") and np.isfinite(panel.teff) \
            and panel.teff < teff_cool_he and ({pair["a"], pair["b"]} & COOL_HE_ELEMENTS):
        kills.append(KILL_COOL_HE)
    if two_parcel is not None and two_parcel.get("p_naive_two_parcel") is not None \
            and two_parcel["p_naive_two_parcel"] > rest_p_min:
        kills.append(KILL_ASYNC)
    if fit_full is not None and \
            naive_p(fit_full.chi2, fit_full.n_measured - 1) > free_fractionation_p_min:
        kills.append(KILL_FREE_FRACTIONATION)
    # The plainest kill there is: a measured rock already sits where this ratio
    # sits.  PEWDD's compilation carries ~1,100 meteorites per common pair, and
    # the full sane min-max of that sample is natural by demonstration, not by
    # model.  (The percentile envelope, which is what z_pair is measured
    # against, is deliberately tighter; this catches a candidate that the
    # percentile cut excluded but a real aubrite, ureilite or pallasite does.)
    if pair.get("met_min") is not None and pair.get("met_max") is not None \
            and pair.get("obs_log_ratio") is not None:
        try:
            if float(pair["met_min"]) <= float(pair["obs_log_ratio"]) <= float(pair["met_max"]):
                kills.append(KILL_REAL_METEORITE)
        except (TypeError, ValueError):
            pass
    if "Ca" in (pair["a"], pair["b"]) and "C" in panel.elements and "Ca" in panel.elements:
        i = panel.elements.index("C")
        j = panel.elements.index("Ca")
        if panel.values[i] - panel.values[j] > -0.5:
            kills.append(KILL_CARBONATE)
    if other_panels:
        for e in (pair["a"], pair["b"]):
            if e not in panel.elements:
                continue
            i = panel.elements.index(e)
            for op in other_panels:
                if e in op.elements:
                    k = op.elements.index(e)
                    d = abs(panel.values[i] - op.values[k])
                    s = float(np.hypot(panel.errors[i], op.errors[k]))
                    if s > 0 and d / s > multi_ref_sigma:
                        kills.append(KILL_MULTI_REF)
                        break
            if KILL_MULTI_REF in kills:
                break
    return kills


__all__ = ["CAVEAT_EXOTIC_MANTLE", "COOL_HE_ELEMENTS", "FLAG_ALLOY", "FLAG_BE",
           "FLAG_BE_UNMEASURED", "FLAG_CU_ZN_SN_PB", "FLAG_FE_MG", "FLAG_PURE_SI",
           "KILL_ASYNC", "KILL_CARBONATE", "KILL_COOL_HE", "KILL_FREE_FRACTIONATION",
           "KILL_INFORMATION_LIMITED", "KILL_MULTI_REF", "KILL_REAL_METEORITE",
           "KILL_REST_NOT_NATURAL",
           "MIN_REST_FOR_PHASE", "ORTHOGONAL_PAIRS", "PHASE_FROM_REST", "PHASE_UNCONSTRAINED",
           "apply_kills", "fit_two_parcel", "natural_interval", "pair_residuals",
           "phase_shift_range", "refinery_flags", "rest_phase_set"]
