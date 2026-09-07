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
from .halo import shm, shm_plus_plus, shm_tail
from .rate import Efficiency
from .timing import (
    EventModel,
    LiveTime,
    modulation_summary,
    profile_likelihood,
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
                rows.append({"halo": name, "m_chi_gev": m, "delta_keV": float(delta),
                             "timing": pl["timing"], "energy_per_keV": pl["energy"], "profile": pl["profile"],
                             "sigma_n_for_one_event_cm2": sig1,
                             "events_at_ceiling": mu_max,
                             "marginal": pl["profile"] * occam,
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lzedge")
    ap.add_argument("--stage", default="all", choices=["kinematics", "scan", "tails", "posterior", "all"])
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
