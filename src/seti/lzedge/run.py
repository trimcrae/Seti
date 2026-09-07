"""LZEDGE stage orchestration.  Writes results/lzedge/.

Stages
  kinematics  the event's v_min / threshold / E* map over (m_chi, delta) and the
              kinematic ceiling on delta for each halo model on the event date
  scan        timing Bayes factor, modulation phase / on-fraction, and the
              observed-energy density at 248 keV over the (m_chi, delta) grid
              for every halo model in config/lzedge.yaml
  all         both
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import pathlib
import time

import numpy as np
import yaml

from . import kinematics as K
from .earth import extremal_dates, galactic_lb_deg, v_lab_kms, v_lab_speed_kms, year_grid
from .halo import lmc_tail, shm, shm_plus_plus, shm_tail
from .rate import Efficiency
from .timing import (
    EventModel,
    LiveTime,
    modulation_summary,
    profile_likelihood,
    sideband_expectation,
    timing_bayes_factor,
)


def load_cfg(path: str | None = None) -> dict:
    p = pathlib.Path(path) if path else pathlib.Path(__file__).resolve().parents[3] / "config" / "lzedge.yaml"
    return yaml.safe_load(p.read_text())


def build_halo(name: str, spec: dict):
    t = spec.get("type", "shm")
    if t == "shm":
        return shm(v0=spec["v0"], v_esc=spec["v_esc"], name=name)
    if t == "shmpp":
        return shm_plus_plus(v0=spec["v0"], v_esc=spec["v_esc"], eta_s=spec.get("eta_sausage", 0.2),
                             beta=spec.get("beta", 0.9), name=name)
    if t == "shm_tail":
        return shm_tail(v0=spec["v0"], v_esc=spec["v_esc"], tail=spec.get("tail", "sharp"),
                        k=spec.get("k", 2.0), v_join=spec.get("v_join"), name=name)
    if t == "shm_lmc":
        from .halo import LMC_DIRECTION
        base = shm(v0=spec["v0"], v_esc=spec["v_esc"], name="round")
        return lmc_tail(base, speed_kms=spec["lmc_speed"], sigma_kms=spec["lmc_sigma"],
                        fraction=spec["lmc_fraction"], direction=tuple(spec.get("lmc_direction", LMC_DIRECTION)),
                        cut_kms=spec.get("lmc_cut"), name=name)
    raise ValueError(f"unknown halo type {t!r}")


def reachability_calendar(E_keV: float, m_chi_gev: float, delta_keV: float, v_esc: float,
                          v0_kms: float, year: int, n: int = 366) -> dict:
    """Dates on which a recoil of E_keV is kinematically reachable: v_min(E) <= v_esc + v_lab(t).

    Pure kinematics plus the escape-sphere edge — no rate, no efficiency, no
    cross section.  Returns the on-fraction of the year and the first/last
    reachable dates (None if never / always)."""
    vmin = float(K.v_min_kms(E_keV, m_chi_gev, delta_keV))
    grid = year_grid(year, n)
    vmax = np.array([v_esc + v_lab_speed_kms(t, v0_kms) for t in grid])
    ok = vmin <= vmax
    frac = float(np.mean(ok))
    if frac == 0.0:
        return {"fraction": 0.0, "first": None, "last": None, "v_min_kms": vmin}
    if frac == 1.0:
        return {"fraction": 1.0, "first": None, "last": None, "v_min_kms": vmin}
    # the reachable set is one arc around the June peak; report its edges
    idx = np.where(ok)[0]
    return {"fraction": frac, "first": grid[int(idx[0])].date().isoformat(),
            "last": grid[int(idx[-1])].date().isoformat(), "v_min_kms": vmin}


def stage_kinematics(cfg: dict, out: pathlib.Path) -> dict:
    ev = cfg["event"]
    E = float(ev["energy_keV"])
    t_obs = ev["date"]
    rows = []
    halos = {n: (build_halo(n, s), s) for n, s in cfg["halo_models"].items()}
    vlab = v_lab_speed_kms(t_obs, cfg["halo_models"]["shm_lz"]["v0"])
    peak, trough = extremal_dates(int(t_obs[:4]), cfg["halo_models"]["shm_lz"]["v0"])
    for m in cfg["scan"]["m_chi_gev"]:
        d = cfg["scan"]["delta_keV"]
        for delta in np.arange(d["min"], d["max"] + 0.5 * d["step"], d["step"]):
            row = {"m_chi_gev": m, "delta_keV": float(delta),
                   "v_min_at_event_kms": float(K.v_min_kms(E, m, delta)),
                   "v_threshold_kms": K.v_threshold_kms(m, delta),
                   "E_star_keV": K.e_star_keV(m, delta)}
            for name, (h, spec) in halos.items():
                vmax = h.v_max(v_lab_kms(t_obs, spec["v0"]))
                row[f"{name}_v_max_kms"] = vmax
                row[f"{name}_allowed_at_event"] = bool(row["v_min_at_event_kms"] <= vmax)
                row[f"{name}_E_max_keV"] = K.max_recoil_energy_keV(vmax, m, delta)
                cal = reachability_calendar(E, m, float(delta), spec["v_esc"], spec["v0"], int(t_obs[:4]))
                row[f"{name}_reachable_fraction_of_year"] = cal["fraction"]
                row[f"{name}_reachable_first"] = cal["first"]
                row[f"{name}_reachable_last"] = cal["last"]
            rows.append(row)
    ceilings = {}
    for name, (h, spec) in halos.items():
        v = v_lab_kms(t_obs, spec["v0"])
        vmax = h.v_max(v)
        ceilings[name] = {"v_lab_kms": float(np.linalg.norm(v)), "v_max_kms": vmax,
                          "delta_max_any_energy_keV": {str(m): K.max_splitting_keV(vmax, m) for m in cfg["scan"]["m_chi_gev"]},
                          "delta_max_at_event_energy_keV": {str(m): K.max_splitting_at_energy_keV(vmax, E, m) for m in cfg["scan"]["m_chi_gev"]}}
    summary = {
        "event": ev, "v_lab_on_event_date_kms": vlab,
        "v_lab_direction_lb_deg": galactic_lb_deg(v_lab_kms(t_obs, cfg["halo_models"]["shm_lz"]["v0"])),
        "lab_speed_peak_date": peak.date().isoformat(), "lab_speed_trough_date": trough.date().isoformat(),
        "days_after_peak": (dt.date.fromisoformat(t_obs) - peak.date()).days,
        "helm_zeros_keV": K.helm_zeros_keV(),
        "kinematic_ceilings": ceilings,
    }
    with (out / "kinematics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "kinematics_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def stage_scan(cfg: dict, out: pathlib.Path, halos_only: list[str] | None = None) -> dict:
    ev, run, win, sc = cfg["event"], cfg["run"], cfg["window"], cfg["scan"]
    E_obs, t_obs = float(ev["energy_keV"]), ev["date"]
    eff = Efficiency.from_table(win["efficiency"])
    live = LiveTime.uniform(run["start"], run["end"], run["live_days"])
    sig_ref = float(ev["sigma_stat_keV"])

    def sigma_fn(E):
        return K.resolution_sigma_keV(E, sig_ref, E_obs)

    d = sc["delta_keV"]
    deltas = np.arange(d["min"], d["max"] + 0.5 * d["step"], d["step"])
    rows = []
    t0 = time.time()
    for name, spec in cfg["halo_models"].items():
        if halos_only and name not in halos_only:
            continue
        halo = build_halo(name, spec)
        for m in sc["m_chi_gev"]:
            for delta in deltas:
                model = EventModel(halo, float(m), float(delta), eff, v0_kms=spec["v0"],
                                   rho_gev_cm3=spec.get("rho", 0.3))
                tb = timing_bayes_factor(model, live, t_obs, sc.get("livetime_step_days", 3.0))
                ms = modulation_summary(model, int(t_obs[:4]), sc.get("year_samples", 73))
                r_obs = tb["rate_at_event"]
                pE = float(model.observed_energy_density(E_obs, t_obs, sigma_fn)[0]) / r_obs if r_obs > 0 else 0.0
                rows.append({
                    "halo": name, "m_chi_gev": m, "delta_keV": float(delta),
                    "rate_at_event_per_ty": r_obs, "mean_rate_livetime_per_ty": tb["mean_rate_livetime"],
                    "bayes_factor_timing": tb["bayes_factor_timing"],
                    "expected_events_2p84ty_at_1e-45": tb["mean_rate_livetime"] * run["exposure_tonne_year"],
                    "sigma_n_for_one_event_cm2": (1e-45 / (tb["mean_rate_livetime"] * run["exposure_tonne_year"])
                                                  if tb["mean_rate_livetime"] > 0 else None),
                    "energy_density_at_event_per_keV": pE,
                    "peak_date": ms["peak_date"], "trough_date": ms["trough_date"],
                    "fractional_modulation": ms["fractional_modulation"],
                    "fraction_of_year_on": ms["fraction_of_year_on"],
                })
                (out / "scan.json").write_text(json.dumps(rows, indent=1))
        print(f"  {name}: {len(rows)} rows so far, {time.time() - t0:.0f}s")
    with (out / "scan.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    best = sorted([r for r in rows if r["rate_at_event_per_ty"] > 0],
                  key=lambda r: -(r["bayes_factor_timing"] * r["energy_density_at_event_per_keV"]))[:20]
    summary = {"n_rows": len(rows), "top_joint": best, "elapsed_s": round(time.time() - t0, 1)}
    (out / "scan_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def stage_tails(cfg: dict, out: pathlib.Path) -> dict:
    """Tail shape x v_esc x v0 at the event: the astrophysics of the edge, model by model."""
    ev, run, win, sc = cfg["event"], cfg["run"], cfg["window"], cfg["scan"]
    E_obs, t_obs = float(ev["energy_keV"]), ev["date"]
    eff = Efficiency.from_table(win["efficiency"])
    live = LiveTime.uniform(run["start"], run["end"], run["live_days"])
    sig_ref = float(ev["sigma_stat_keV"])

    def sigma_fn(E):
        return K.resolution_sigma_keV(E, sig_ref, E_obs)

    tails = cfg.get("tails", {})
    shapes = tails.get("shapes", [{"tail": "sharp"}, {"tail": "soft"}, {"tail": "power", "k": 1.0},
                                  {"tail": "power", "k": 2.5}])
    vescs = tails.get("v_esc", [500.0, 528.0, 544.0, 580.0])
    v0s = tails.get("v0", [220.0, 238.0, 250.0])
    masses = tails.get("m_chi_gev", [200.0, 1000.0, 10000.0])
    d = tails.get("delta_keV", {"min": 0.0, "max": 480.0, "step": 20.0})
    deltas = np.arange(d["min"], d["max"] + 0.5 * d["step"], d["step"])
    rows = []
    t0 = time.time()
    for shape in shapes:
        for vesc in vescs:
            for v0 in v0s:
                halo = shm_tail(v0=v0, v_esc=vesc, tail=shape["tail"], k=shape.get("k", 2.0))
                label = f"{shape['tail']}" + (f"_k{shape['k']}" if shape["tail"] == "power" else "")
                for m in masses:
                    for delta in deltas:
                        model = EventModel(halo, float(m), float(delta), eff, v0_kms=v0)
                        tb = timing_bayes_factor(model, live, t_obs, sc.get("livetime_step_days", 3.0))
                        ms = modulation_summary(model, int(t_obs[:4]), sc.get("year_samples", 73))
                        r_obs = tb["rate_at_event"]
                        pE = float(model.observed_energy_density(E_obs, t_obs, sigma_fn)[0]) / r_obs if r_obs > 0 else 0.0
                        rows.append({"tail": label, "v_esc": vesc, "v0": v0, "m_chi_gev": m, "delta_keV": float(delta),
                                     "rate_at_event_per_ty": r_obs, "mean_rate_livetime_per_ty": tb["mean_rate_livetime"],
                                     "bayes_factor_timing": tb["bayes_factor_timing"],
                                     "energy_density_at_event_per_keV": pE,
                                     "peak_date": ms["peak_date"], "fractional_modulation": ms["fractional_modulation"],
                                     "fraction_of_year_on": ms["fraction_of_year_on"]})
                print(f"  tails {label} vesc={vesc} v0={v0}: {len(rows)} rows, {time.time() - t0:.0f}s")
                (out / "tails.json").write_text(json.dumps(rows, indent=1))
    with (out / "tails.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"n_rows": len(rows), "elapsed_s": round(time.time() - t0, 1)}
    (out / "tails_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def stage_posterior(cfg: dict, out: pathlib.Path, halos_only: list[str] | None = None) -> dict:
    """Profile likelihood over (m_chi, delta) per halo model, sigma_n profiled, the
    energy-scale systematic marginalised.  Normalised per halo to its maximum, and
    across halos to the global maximum, so the tables read as likelihood ratios."""
    ev, run, win, sc = cfg["event"], cfg["run"], cfg["window"], cfg["scan"]
    E_obs, t_obs = float(ev["energy_keV"]), ev["date"]
    eff = Efficiency.from_table(win["efficiency"])
    live = LiveTime.uniform(run["start"], run["end"], run["live_days"])
    sig_ref, sig_sys = float(ev["sigma_stat_keV"]), float(ev.get("sigma_sys_keV", 0.0))

    def sigma_fn(E):
        return K.resolution_sigma_keV(E, sig_ref, E_obs)

    pc = cfg.get("posterior", {})
    sbc = cfg.get("sideband", {})
    sigma_ceiling = float(pc.get("sigma_ceiling_cm2", 1e-37))
    masses = pc.get("m_chi_gev", sc["m_chi_gev"])
    d = pc.get("delta_keV", sc["delta_keV"])
    deltas = np.arange(d["min"], d["max"] + 0.5 * d["step"], d["step"])
    rows = []
    t0 = time.time()
    for name, spec in cfg["halo_models"].items():
        if halos_only and name not in halos_only:
            continue
        halo = build_halo(name, spec)
        for m in masses:
            for delta in deltas:
                model = EventModel(halo, float(m), float(delta), eff, v0_kms=spec["v0"],
                                   rho_gev_cm3=spec.get("rho", 0.3))
                pl = profile_likelihood(model, live, E_obs, t_obs, sigma_fn, sig_sys,
                                        sc.get("livetime_step_days", 3.0))
                mu1 = pl["mean_rate_livetime"] * run["exposure_tonne_year"]     # events at 1e-45 cm²
                sig1 = 1e-45 / mu1 if mu1 > 0 else None
                # marginal over log sigma with a log-uniform prior up to the ceiling:
                # ∫ dμ/μ e^{-μ} μ p(E,t) ∝ p(E,t) (1 - e^{-μ_max}),  μ_max = events at the ceiling
                mu_max = mu1 * (sigma_ceiling / 1e-45)
                occam = 1.0 - math.exp(-mu_max) if mu_max < 50 else 1.0
                # the empty high-energy sideband: expected counts per window event, Poisson(0)
                sb_per_event = 0.0
                if mu1 > 0 and sbc.get("use", True):
                    sb_rate = sideband_expectation(model, live, sbc["E_lo_keV"], sbc["E_hi_keV"],
                                                   sbc.get("plateau_efficiency", 0.96), sc.get("livetime_step_days", 3.0))
                    sb_per_event = sb_rate / pl["mean_rate_livetime"]
                sb_factor = math.exp(-sb_per_event * sbc.get("observed_window_events", 1.0))
                rows.append({"halo": name, "m_chi_gev": m, "delta_keV": float(delta),
                             "timing": pl["timing"], "energy_per_keV": pl["energy"], "profile": pl["profile"],
                             "sigma_n_for_one_event_cm2": sig1,
                             "events_at_ceiling": mu_max,
                             "sideband_per_window_event": sb_per_event,
                             "sideband_factor": sb_factor,
                             "marginal": pl["profile"] * occam * sb_factor,
                             "physical": bool(sig1 is not None and sig1 <= sigma_ceiling)})
        print(f"  posterior {name}: {len(rows)} rows, {time.time() - t0:.0f}s")
        (out / "posterior.json").write_text(json.dumps(rows, indent=1))
    key = "marginal"
    gmax = max(r[key] for r in rows) if rows else 0.0
    per_halo = {}
    for name in {r["halo"] for r in rows}:
        rr = [r for r in rows if r["halo"] == name]
        hmax = max(r[key] for r in rr)
        best = max(rr, key=lambda r: r[key])
        # delta interval at fixed mass where the marginal is within e^-0.5 (1 sigma) and e^-2 (2 sigma) of the best
        by_m = {}
        for m in masses:
            rm = [r for r in rr if r["m_chi_gev"] == m]
            mm = max(r[key] for r in rm) if rm else 0.0
            if mm <= 0:
                by_m[str(m)] = None
                continue
            d1 = [r["delta_keV"] for r in rm if r[key] >= mm * math.exp(-0.5)]
            d2 = [r["delta_keV"] for r in rm if r[key] >= mm * math.exp(-2.0)]
            bm = max(rm, key=lambda r: r[key])
            by_m[str(m)] = {"best_delta_keV": bm["delta_keV"], "sigma_n_at_best_cm2": bm["sigma_n_for_one_event_cm2"],
                            "delta_1sigma_keV": [min(d1), max(d1)], "delta_2sigma_keV": [min(d2), max(d2)],
                            "max_marginal_rel_global": mm / gmax if gmax > 0 else None}
        per_halo[name] = {"best": best, "max_marginal_rel_global": hmax / gmax if gmax > 0 else None,
                          "by_mass": by_m}
    for r in rows:
        r["marginal_rel_global"] = r[key] / gmax if gmax > 0 else None
    # evidence per halo: mean of the marginal over the (m, delta) grid (log-uniform in m, uniform in delta)
    evid = {}
    for name in per_halo:
        rr = [r for r in rows if r["halo"] == name]
        evid[name] = float(np.mean([r[key] for r in rr])) if rr else 0.0
    emax = max(evid.values()) if evid else 0.0
    for name in per_halo:
        per_halo[name]["evidence"] = evid[name]
        per_halo[name]["evidence_rel_best"] = evid[name] / emax if emax > 0 else None
    with (out / "posterior.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"n_rows": len(rows), "per_halo": per_halo, "elapsed_s": round(time.time() - t0, 1),
               "sigma_sys_keV": sig_sys, "sigma_ceiling_cm2": sigma_ceiling,
               "note": "marginal = profile x (1 - exp(-events at the ceiling)): a log-uniform prior on "
                       "sigma_n up to the ceiling; profile alone is flat in sigma_n and runs to the grid edge"}
    (out / "posterior_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _quantiles(cdf: np.ndarray, grid: np.ndarray, probs) -> list[float]:
    return [float(np.interp(p, cdf, grid)) for p in probs]


def _asym_gaussian_weights(grid: np.ndarray, centre: float, plus: float, minus: float) -> np.ndarray:
    """Two-piece Gaussian weights for a v_esc measurement quoted as centre (+plus / -minus)."""
    sig = np.where(grid >= centre, plus, minus)
    w = np.exp(-0.5 * ((grid - centre) / sig) ** 2)
    return w / np.sum(w)


def stage_vesc_marginal(cfg: dict, out: pathlib.Path) -> dict:
    """The delta posterior with the escape speed integrated over a *measurement*:

        P(delta | event, measurement) ∝ ∫ dv_esc P(v_esc | measurement) L(delta, v_esc),

    for each measurement listed under ``vesc_measurements`` in config, at the
    masses in ``posterior.m_chi_gev``, one-component SHM with the chosen tail
    (config ``vesc_tail``: sharp / soft / power).  Also the probability that
    the event is kinematically reachable at all on its date, per measurement."""
    ev, run, win, sc = cfg["event"], cfg["run"], cfg["window"], cfg["scan"]
    E_obs, t_obs = float(ev["energy_keV"]), ev["date"]
    eff = Efficiency.from_table(win["efficiency"])
    live = LiveTime.uniform(run["start"], run["end"], run["live_days"])
    sig_ref, sig_sys = float(ev["sigma_stat_keV"]), float(ev.get("sigma_sys_keV", 0.0))

    def sigma_fn(E):
        return K.resolution_sigma_keV(E, sig_ref, E_obs)

    pc = cfg.get("posterior", {})
    sbc = cfg.get("sideband", {})
    sigma_ceiling = float(pc.get("sigma_ceiling_cm2", 1e-37))
    masses = pc.get("m_chi_gev", sc["m_chi_gev"])
    d = pc.get("delta_keV", sc["delta_keV"])
    deltas = np.arange(d["min"], d["max"] + 0.5 * d["step"], d["step"])
    tail = cfg.get("vesc_tail", {"tail": "sharp"})
    v0 = float(cfg.get("vesc_v0_kms", 238.0))
    meas = cfg.get("vesc_measurements", {})
    vgrid = np.arange(float(cfg.get("vesc_grid", {}).get("min", 440.0)),
                      float(cfg.get("vesc_grid", {}).get("max", 660.0)) + 0.1,
                      float(cfg.get("vesc_grid", {}).get("step", 10.0)))
    t0 = time.time()
    # likelihood table L[v_esc, m, delta] (marginal over sigma_n with the ceiling), computed once
    like = np.zeros((len(vgrid), len(masses), len(deltas)))
    reach = np.zeros((len(vgrid), len(masses)))
    for i, vesc in enumerate(vgrid):
        halo = shm_tail(v0=v0, v_esc=float(vesc), tail=tail.get("tail", "sharp"), k=tail.get("k", 2.0))
        vmax = float(vesc) + v_lab_speed_kms(t_obs, v0)
        for j, m in enumerate(masses):
            reach[i, j] = 1.0 if float(K.v_min_kms(E_obs, m, 0.0)) <= vmax else 0.0
            for k_, delta in enumerate(deltas):
                model = EventModel(halo, float(m), float(delta), eff, v0_kms=v0)
                pl = profile_likelihood(model, live, E_obs, t_obs, sigma_fn, sig_sys, sc.get("livetime_step_days", 3.0))
                mu1 = pl["mean_rate_livetime"] * run["exposure_tonne_year"]
                mu_max = mu1 * (sigma_ceiling / 1e-45)
                occam = 1.0 - math.exp(-mu_max) if mu_max < 50 else 1.0
                sb_factor = 1.0
                if mu1 > 0 and sbc.get("use", True):
                    sb_rate = sideband_expectation(model, live, sbc["E_lo_keV"], sbc["E_hi_keV"],
                                                   sbc.get("plateau_efficiency", 0.96), sc.get("livetime_step_days", 3.0))
                    sb_factor = math.exp(-(sb_rate / pl["mean_rate_livetime"]) * sbc.get("observed_window_events", 1.0))
                like[i, j, k_] = pl["profile"] * occam * sb_factor
        print(f"  vesc {vesc:.0f}: {time.time() - t0:.0f}s")
    # the inverse question: the event's own likelihood of the escape speed, L(v_esc) = mean over (m, delta)
    ev_v = np.mean(like, axis=(1, 2))
    ev_v_n = ev_v / np.sum(ev_v) if np.sum(ev_v) > 0 else ev_v
    cdf_v = np.cumsum(ev_v_n)
    inverse = {"vesc_grid": [float(x) for x in vgrid], "likelihood_norm": [float(x) for x in ev_v_n],
               "flat_prior_median_kms": _quantiles(cdf_v, vgrid, (0.5,))[0],
               "flat_prior_68_kms": _quantiles(cdf_v, vgrid, (0.16, 0.84)),
               "flat_prior_95_lower_kms": _quantiles(cdf_v, vgrid, (0.05,))[0]}
    results = {}
    for name, spec in meas.items():
        w = _asym_gaussian_weights(vgrid, float(spec["v_esc"]), float(spec["plus"]), float(spec["minus"]))
        post_v = w * ev_v
        zv = float(np.sum(post_v))
        per_mass = {"_vesc_posterior": None}
        if zv > 0:
            pv = post_v / zv
            cv = np.cumsum(pv)
            per_mass["_vesc_posterior"] = {"evidence": zv / float(np.sum(w)) if np.sum(w) > 0 else None,
                                           "median_kms": _quantiles(cv, vgrid, (0.5,))[0],
                                           "v68_kms": _quantiles(cv, vgrid, (0.16, 0.84)),
                                           "posterior": [(float(a), float(b)) for a, b in zip(vgrid, pv, strict=True)]}
        for j, m in enumerate(masses):
            post = np.einsum("i,ik->k", w, like[:, j, :])
            evidence = float(np.sum(post) * (deltas[1] - deltas[0]) if len(deltas) > 1 else np.sum(post))
            if post.max() <= 0:
                per_mass[str(m)] = None
                continue
            pn = post / np.sum(post)
            cdf = np.cumsum(pn)
            q = _quantiles(cdf, deltas, (0.5, 0.16, 0.84, 0.025, 0.975))
            per_mass[str(m)] = {"best_delta_keV": float(deltas[int(np.argmax(post))]),
                                "delta_median_keV": q[0], "delta_68_keV": [q[1], q[2]],
                                "delta_95_keV": [q[3], q[4]],
                                "p_elastic_reachable": float(np.sum(w * reach[:, j])),
                                "evidence": evidence,
                                "posterior": [(float(dd), float(pp)) for dd, pp in zip(deltas, pn, strict=True)]}
        results[name] = {"measurement": spec, "per_mass": per_mass}
    emax = max((r["per_mass"]["_vesc_posterior"] or {}).get("evidence") or 0.0 for r in results.values()) if results else 0.0
    for r in results.values():
        vp = r["per_mass"]["_vesc_posterior"]
        if vp and vp.get("evidence") is not None:
            vp["evidence_rel_best"] = vp["evidence"] / emax if emax > 0 else None
    summary = {"tail": tail, "v0_kms": v0, "vesc_grid": [float(x) for x in vgrid], "masses": masses,
               "sigma_ceiling_cm2": sigma_ceiling, "measurements": results, "inverse": inverse,
               "elapsed_s": round(time.time() - t0, 1)}
    (out / "vesc_marginal.json").write_text(json.dumps(summary, indent=1))
    np.save(out / "vesc_like.npy", like)
    return summary


def stage_sensitivity(cfg: dict, out: pathlib.Path) -> dict:
    """How much the date factor depends on the unknown live-time mask, and a
    cross-check of the sideband count against arXiv:2609.04175 (N_SB = 4.9 per
    window event at delta = 377 keV, SHM 544, Helm, sideband ~350-590 keV)."""
    ev, run, win, sc = cfg["event"], cfg["run"], cfg["window"], cfg["scan"]
    t_obs = ev["date"]
    eff = Efficiency.from_table(win["efficiency"])
    spec = cfg["halo_models"]["shm_lz"]
    halo = build_halo("shm_lz", spec)
    masks = {
        "uniform_27mar2023_1apr2024": LiveTime.uniform(run["start"], run["end"], run["live_days"]),
        "late_start_1jun2023": LiveTime.uniform("2023-06-01", run["end"], run["live_days"]),
        "early_end_31dec2023": LiveTime.uniform(run["start"], "2023-12-31", run["live_days"]),
        "summer_half_duty": LiveTime.from_rows([(run["start"], "2023-05-15", 1.0), ("2023-05-15", "2023-08-15", 0.5),
                                               ("2023-08-15", run["end"], 1.0)]),
        "winter_half_duty": LiveTime.from_rows([(run["start"], "2023-10-15", 1.0), ("2023-10-15", "2024-02-15", 0.5),
                                               ("2024-02-15", run["end"], 1.0)]),
        "calibration_weeks_removed": LiveTime.from_rows([(run["start"], "2023-06-05", 1.0), ("2023-06-05", "2023-06-12", 0.0),
                                                        ("2023-06-12", run["end"], 1.0)]),
    }
    rows = []
    for m in (1000.0,):
        for delta in (300.0, 340.0, 360.0, 370.0, 377.0, 380.0):
            model = EventModel(halo, m, delta, eff, v0_kms=spec["v0"])
            entry = {"m_chi_gev": m, "delta_keV": delta}
            for name, live in masks.items():
                tb = timing_bayes_factor(model, live, t_obs, sc.get("livetime_step_days", 6.0))
                entry[f"timing_{name}"] = tb["bayes_factor_timing"]
            live0 = masks["uniform_27mar2023_1apr2024"]
            r_win = timing_bayes_factor(model, live0, t_obs, sc.get("livetime_step_days", 6.0))["mean_rate_livetime"]
            for lo, hi, tag in ((350.0, 590.0, "04175_350_590"), (350.0, 600.0, "ours_350_600"), (350.0, 680.0, "dent_350_680")):
                sb = sideband_expectation(model, live0, lo, hi, 0.96, sc.get("livetime_step_days", 6.0))
                entry[f"sideband_per_event_{tag}"] = sb / r_win if r_win > 0 else None
            rows.append(entry)
    summary = {"rows": rows, "reference_04175": {"delta1_keV": 377, "N_SB_helm": 4.9, "N_SB_vietze": 3.7,
                                                  "sideband_keV": [350, 590]}}
    (out / "sensitivity.json").write_text(json.dumps(summary, indent=1))
    return summary


def stage_numbers(cfg: dict, out: pathlib.Path, paper_dir: pathlib.Path | None = None) -> dict:
    """Write paper/lzedge/numbers.tex from the committed results (missing stages -> TODO macros)."""
    ev, run = cfg["event"], cfg["run"]
    E, t_obs = float(ev["energy_keV"]), ev["date"]
    v0, vesc = cfg["halo_models"]["shm_lz"]["v0"], cfg["halo_models"]["shm_lz"]["v_esc"]
    peak, trough = extremal_dates(int(t_obs[:4]), v0)
    vmax = vesc + v_lab_speed_kms(t_obs, v0)
    q = math.sqrt(2.0 * K.nucleus_mass_gev(K.XE_A_MEAN) * 1e6 * E)
    macros = {
        "EventEnergy": f"{E:.0f}", "EventDate": t_obs, "Exposure": f"{run['exposure_tonne_year']:.2f}",
        "VminElastic": f"{float(K.v_min_kms(E, 1000.0, 0.0)):.0f}",
        "VminPerHundredKeV": f"{K.C_KMS * 100.0 / q:.0f}",
        "DeltaMaxLZ": f"{K.max_splitting_at_energy_keV(vmax, E, 1000.0):.0f}",
        "VescLZ": f"{vesc:.0f}", "VlabAmp": f"{0.5 * (v_lab_speed_kms(peak, v0) - v_lab_speed_kms(trough, v0)):.1f}",
        "DaysAfterPeak": f"{(dt.date.fromisoformat(t_obs) - peak.date()).days}",
        "TimingBFbest": r"\TODO{run posterior}",
    }
    ps = out / "posterior_summary.json"
    if ps.exists():
        s = json.loads(ps.read_text())
        best = max(s["per_halo"].values(), key=lambda v: v["max_marginal_rel_global"] or 0)["best"]
        macros["TimingBFbest"] = f"{best['timing']:.1f}"
        macros["BestDelta"] = f"{best['delta_keV']:.0f}"
        macros["BestHalo"] = best["halo"].replace("_", r"\_")
        macros["BestSideband"] = f"{best['sideband_per_window_event']:.2f}"
        for name, v in s["per_halo"].items():
            key = "".join(w.capitalize() for w in name.split("_"))
            if v.get("evidence_rel_best") is not None:
                macros[f"Evid{key}"] = f"{v['evidence_rel_best']:.2g}"
            bm = v["by_mass"].get("1000")
            if bm:
                macros[f"DeltaBest{key}"] = f"{bm['best_delta_keV']:.0f}"
                macros[f"DeltaOne{key}"] = f"{bm['delta_1sigma_keV'][0]:.0f}--{bm['delta_1sigma_keV'][1]:.0f}"
    ss = out / "sensitivity.json"
    if ss.exists():
        srows = json.loads(ss.read_text())["rows"]
        row = next((x for x in srows if x["delta_keV"] == 370.0), srows[-1])
        vals = [v for k, v in row.items() if k.startswith("timing_")]
        macros["TimingMaskMin"] = f"{min(vals):.1f}"
        macros["TimingMaskMax"] = f"{max(vals):.1f}"
        macros["SidebandAt377"] = f"{row.get('sideband_per_event_04175_350_590', float('nan')):.1f}" if row["delta_keV"] == 377.0 else \
            f"{next(x for x in srows if x['delta_keV'] == 377.0)['sideband_per_event_04175_350_590']:.1f}"
    vs = out / "vesc_marginal.json"
    if vs.exists():
        sv = json.loads(vs.read_text())
        inv = sv.get("inverse", {})
        if inv:
            macros["VescMedian"] = f"{inv['flat_prior_median_kms']:.0f}"
            macros["VescSixtyEight"] = f"{inv['flat_prior_68_kms'][0]:.0f}--{inv['flat_prior_68_kms'][1]:.0f}"
            macros["VescNinetyFiveLower"] = f"{inv['flat_prior_95_lower_kms']:.0f}"
        for name, res in sv["measurements"].items():
            key = "".join(w.capitalize() for w in name.split("_"))
            pm = res["per_mass"].get("1000")
            if pm:
                macros[f"DeltaMed{key}"] = f"{pm['delta_median_keV']:.0f}"
                macros[f"DeltaSixtyEight{key}"] = f"{pm['delta_68_keV'][0]:.0f}--{pm['delta_68_keV'][1]:.0f}"
            vp = res["per_mass"].get("_vesc_posterior") or {}
            if vp.get("evidence_rel_best") is not None:
                macros[f"EvidVesc{key}"] = f"{vp['evidence_rel_best']:.2g}"
    pdir = paper_dir or (pathlib.Path(__file__).resolve().parents[3] / "paper" / "lzedge")
    pdir.mkdir(parents=True, exist_ok=True)
    lines = ["% generated by `python -m seti.lzedge.run --stage numbers`; do not edit"]
    for k, v in macros.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}")
    (pdir / "numbers.tex").write_text("\n".join(lines) + "\n")
    return macros


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lzedge")
    ap.add_argument("--stage", default="all",
                    choices=["kinematics", "scan", "tails", "posterior", "vesc", "sensitivity", "numbers", "figures", "all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/lzedge")
    ap.add_argument("--halos", default=None, help="comma-separated subset of halo_models")
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("kinematics", "all"):
        s = stage_kinematics(cfg, out)
        print(json.dumps({k: v for k, v in s.items() if k != "kinematic_ceilings"}, indent=1, default=str))
    if a.stage in ("scan", "all"):
        s = stage_scan(cfg, out, a.halos.split(",") if a.halos else None)
        print(f"scan: {s['n_rows']} rows in {s['elapsed_s']}s")
    if a.stage in ("tails", "all"):
        s = stage_tails(cfg, out)
        print(f"tails: {s['n_rows']} rows in {s['elapsed_s']}s")
    if a.stage in ("posterior", "all"):
        s = stage_posterior(cfg, out, a.halos.split(",") if a.halos else None)
        print(f"posterior: {s['n_rows']} rows in {s['elapsed_s']}s")
    if a.stage in ("vesc", "all"):
        s = stage_vesc_marginal(cfg, out)
        print(f"vesc: {len(s['measurements'])} measurements in {s['elapsed_s']}s")
    if a.stage in ("sensitivity", "all"):
        s = stage_sensitivity(cfg, out)
        print("sensitivity:", json.dumps(s["rows"][-2], indent=None))
    if a.stage in ("figures", "all"):
        from .figures import make_all
        made = make_all(cfg, out / "figures")
        print("figures:", ", ".join(p.name for p in made))
    if a.stage in ("numbers", "all"):
        m = stage_numbers(cfg, out)
        print("numbers:", json.dumps(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
