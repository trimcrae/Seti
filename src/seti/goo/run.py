"""GOO stage orchestration.  Writes results/goo/ and paper/goo/numbers.tex.

Stages
  planet    the three ceilings on in-place conversion of Earth's biosphere, crust and bulk
  system    belt/planet conversion times, swarm covering fractions and temperatures,
            residue lifetimes, and the event-rate bounds the null searches imply
  passive   blowout / arrival size window, seeding rates, survival scan
  active    front speeds, galaxy fill times, cosmological reach
  bayes     posteriors on the escaping-branch probability with and without the shadow
  figures   the manuscript figures
  numbers   paper/goo/numbers.tex
  all       everything, in that order
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re

import numpy as np
import yaml

from . import active as A
from . import bayes as B
from . import passive as P
from . import planet as PL
from . import system as S
from .constants import GYR, KM, L_SUN, M_EARTH, MYR, R_EARTH, YR, C

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _numeric(obj):
    """PyYAML reads '5.972e24' (no sign in the exponent) as a string; coerce."""
    if isinstance(obj, dict):
        return {k: _numeric(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numeric(v) for v in obj]
    if isinstance(obj, str):
        try:
            return float(obj) if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", obj) else obj
        except ValueError:
            return obj
    return obj


def load_cfg(path: str | None = None) -> dict:
    p = pathlib.Path(path) if path else ROOT / "config" / "goo.yaml"
    return _numeric(yaml.safe_load(p.read_text()))


def out_dir(cfg: dict | None = None) -> pathlib.Path:
    d = ROOT / "results" / "goo"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dump(name: str, obj: dict) -> None:
    (out_dir() / name).write_text(json.dumps(obj, indent=1, default=float))


def _planet(cfg: dict) -> PL.Planet:
    p = cfg["planet"]
    return PL.Planet(mass_kg=p["mass_kg"], radius_m=p["radius_m"], t_eq_K=p["t_eq_K"],
                     absorbed_solar_W=p["absorbed_solar_W"], biomass_kg=p["biomass_dry_kg"],
                     crust_kg=p["crust_kg"], crust_U_Th_kg=p["crust_U_Th_kg"],
                     fission_J_per_kg=p["fission_J_per_kg"], ocean_kg=p["ocean_kg"],
                     deuterium_fraction=p["deuterium_fraction"], dd_fusion_J_per_kg=p["dd_fusion_J_per_kg"])


# --- planet ---------------------------------------------------------------------
def stage_planet(cfg: dict) -> dict:
    r = cfg["replicator"]
    planet = _planet(cfg)
    m_seed = PL.seed_mass_kg(r["radius_um"], r["density_gcc"])
    out = {"seed_mass_kg": m_seed, "e_rep_J_per_kg": r["e_rep_J_per_kg"],
           "energy_reservoirs_J": PL.energy_reservoirs_J(planet),
           "absorbed_solar_W": planet.absorbed_solar_W,
           "radiative_limit_W": {str(T): PL.radiative_power_limit(planet, T) for T in cfg["planet"]["surface_T_max_K"]},
           "tables": {}}
    for tau in r["tau_doubling_s"]:
        for T in cfg["planet"]["surface_T_max_K"]:
            key = f"tau{tau:g}_T{T:g}"
            out["tables"][key] = PL.conversion_table(planet, m_seed, r["e_rep_J_per_kg"], tau, T)
    # disassembly (lifting the mass off the planet) at the full stellar luminosity and at
    # the surface radiative ceiling
    out["disassembly_s"] = {
        "earth_at_L_sun": PL.t_disassembly_s(planet.mass_kg, planet.radius_m, L_SUN),
        "earth_at_absorbed_solar": PL.t_disassembly_s(planet.mass_kg, planet.radius_m, planet.absorbed_solar_W),
        "earth_at_radiative_2000K": PL.t_disassembly_s(planet.mass_kg, planet.radius_m, PL.radiative_power_limit(planet, 2000.0)),
    }
    _dump("planet.json", out)
    return out


# --- system ---------------------------------------------------------------------
def stage_system(cfg: dict) -> dict:
    s = cfg["system"]
    r = cfg["replicator"]
    tau_mid = 1.0e3
    rows = []
    for res in s["reservoirs"]:
        row = dict(res)
        if res["n_bodies_gt_1km"] > 10:
            row["t_conversion_s"] = S.belt_conversion_time_s(
                res["n_bodies_gt_1km"], res["r_AU"], s["hop_speed_kms"] * KM,
                s["branching_per_body"], s["local_saturation_doublings"], tau_mid)
            row["conversion_regime"] = "transport-limited branching front"
        else:
            # planets: mass leaves at the surface radiative ceiling (2000 K) or at the
            # captured luminosity of a complete swarm, whichever is slower is the honest one
            planet = _planet(cfg)
            M = res["mass_kg"] / res["n_bodies_gt_1km"]
            R = planet.radius_m * (M / M_EARTH) ** (1.0 / 3.0) if M < 10 * M_EARTH else 7.0e7
            P_rad = 4 * math.pi * R**2 * 5.67e-8 * 2000.0**4
            row["t_conversion_s"] = S.planet_disassembly_time_s(M, R, min(P_rad, L_SUN))
            row["t_at_full_L_s"] = S.planet_disassembly_time_s(M, R, L_SUN)
            row["conversion_regime"] = "binding energy at the surface radiative ceiling"
        row["T_swarm_K"] = S.swarm_temperature_K(res["r_AU"])
        row["covering_fraction"] = {f"{sig:g}": S.covering_fraction(res["mass_kg"], sig, res["r_AU"])
                                    for sig in s["swarm_areal_density_kg_m2"]}
        row["residue"] = {}
        for size in (0.5, 10.0, 1000.0, 1.0e6):  # um: micron machine, dust, mm, metre
            f = S.covering_fraction_for_size(res["mass_kg"], size, r["density_gcc"], res["r_AU"])
            row["residue"][f"{size:g}um"] = S.residue_lifetime_s(res["r_AU"], max(f, 1e-12), size, r["density_gcc"])
        row["max_visible_residue_s"] = {f"{fm:g}": S.max_visible_residue_s(res["r_AU"], fm) for fm in (1e-3, 1e-2, 1e-1)}
        rows.append(row)
    # event-rate bounds from null searches, per residue lifetime
    bounds = {}
    for t_res_yr in (1e3, 1e4, 1e5, 1e6, 1e8, 1e10):
        bounds[f"{t_res_yr:g}"] = {
            "hephaistos_II_per_star_per_yr": S.rate_bound_per_star_per_yr(s["hephaistos_II_n_stars"], t_res_yr * YR),
            "hephaistos_II_galaxy_per_yr": S.rate_bound_per_star_per_yr(s["hephaistos_II_n_stars"], t_res_yr * YR) * 1e11,
        }
    out = {"reservoirs": rows, "rate_bounds_vs_residue_lifetime_yr": bounds,
           "ghat_III": {"n_galaxies": s["ghat_III_n_galaxies"], "gamma_max": s["ghat_III_gamma_max"],
                        "fraction_bound_95": 3.0 / s["ghat_III_n_galaxies"]}}
    _dump("system.json", out)
    return out


# --- passive --------------------------------------------------------------------
def stage_passive(cfg: dict) -> dict:
    p = cfg["passive"]
    r = cfg["replicator"]
    g = p["galaxy"]
    v_in = p["inflow_speed_kms"] * KM
    s_min, s_max = P.deliverable_size_window_um(r["density_gcc"], v_in, p["target_r_AU"], r["q_pr"])
    b_at = S.beta_solar(r["radius_um"], r["density_gcc"], r["q_pr"])
    out = {
        "blowout_radius_um": P.blowout_radius_um(r["density_gcc"], r["q_pr"]),
        "beta_max_to_reach_1AU": P.beta_max_to_reach(v_in, p["target_r_AU"]),
        "deliverable_window_um": [s_min, s_max],
        "beta_device": b_at,
        "v_inf_device_kms": P.v_infinity_ms(b_at, p["release_r_AU"]) / KM,
        "v_inf_vs_beta": {f"{b:g}": P.v_infinity_ms(b, p["release_r_AU"]) / KM for b in (0.6, 0.75, 1.0, 1.4)},
        "annulus_volume_m3": P.annulus_volume_m3(g["R_sun_kpc"], g["annulus_width_kpc"], g["scale_height_kpc"]),
        "annulus_fill_time_Gyr": P.annulus_fill_time_s(g["rotation_period_Myr"], g["R_sun_kpc"], g["annulus_width_kpc"]) / GYR,
        "capture_cross_section_m2": P.capture_cross_section_m2(R_EARTH, v_in, M_EARTH),
        "focusing_factor": P.capture_cross_section_m2(R_EARTH, v_in, M_EARTH) / (math.pi * R_EARTH**2),
        "blown_fraction_collisional": P.blown_mass_fraction((s_min, s_max), 0.05, 1e6),
        "seeding": {},
    }
    for M in p["release_mass_kg"]:
        for ts in p["t_survive_yr"]:
            for t_el_Gyr in (2.0, 5.0):
                key = f"M{M:g}_ts{ts:g}_t{t_el_Gyr:g}"
                out["seeding"][key] = P.passive_seeding(
                    M, r["radius_um"], r["density_gcc"], 1.0, g, R_EARTH, M_EARTH, v_in,
                    ts * YR, t_el_Gyr * GYR)
    _dump("passive.json", out)
    return out


# --- active ---------------------------------------------------------------------
def stage_active(cfg: dict) -> dict:
    a = cfg["active"]
    grid = {}
    for v in a["v_ship_c"]:
        for tau in a["tau_build_yr"]:
            grid[f"v{v:g}_tau{tau:g}"] = {
                "v_ship_c": v, "tau_build_yr": tau,
                "front_speed_kms": A.front_speed_ms(v * C, tau * YR, a["d_hop_pc"] * 3.086e16) / KM,
                "t_fill_Myr": A.galaxy_fill_time_s(a["R_galaxy_kpc"], v, tau, a["d_hop_pc"]) / MYR,
            }
    cos = a["cosmology"]
    out = {"front_grid": grid,
           "cosmic_reach": A.reach_table(a["v_cosmic_c"] + [1.0], cos["H0"], cos["Om"], a["n_gal_per_Mpc3"]),
           "event_horizon_Gly": A.comoving_reach_Mpc(1.0, cos["H0"], cos["Om"]) * 3.2616e-3}
    _dump("active.json", out)
    return out


# --- bayes ----------------------------------------------------------------------
def stage_bayes(cfg: dict) -> dict:
    inf = cfg["inference"]
    W = inf["window_Gyr"] * GYR
    f_active = (W - inf["t_fill_active_Myr"] * MYR) / W
    f_passive = (W - inf["t_fill_passive_Gyr"] * GYR) / W
    pgrid = B.loguniform_grid(inf["p_prior"]["lo"], inf["p_prior"]["hi"])
    lo, hi, n = inf["lambda_grid_log10"]
    lams = np.logspace(lo, hi, int(n))
    out = {"f_active": f_active, "f_passive": f_passive, "by_lambda": {}, "marginal": {},
           "likelihood_bounds": {}}
    for lam in inf["lambda_points"]:
        out["likelihood_bounds"][f"{lam:g}"] = {
            "own_active": B.likelihood_bound(lam, f_active),
            "own_passive": B.likelihood_bound(lam, f_passive),
            "ext_persist": B.likelihood_bound(lam, 1.0, inf["n_gal_observed"], 1.0),
            "ext_dead_1e-4": B.likelihood_bound(lam, 1.0, inf["n_gal_observed"], 1e-4),
        }
    for lam in inf["lambda_points"]:
        row = {}
        for label, f in (("active", f_active), ("passive", f_passive)):
            for shadow in (False, True):
                for q in inf["p_persist"]:
                    for use_ext in (False, True):
                        post = B.posterior_p(lam, pgrid, f_own=f, shadowed=shadow,
                                             n_gal=inf["n_gal_observed"], q=q, use_ext=use_ext)
                        key = f"{label}_shadow{int(shadow)}_q{q:g}_ext{int(use_ext)}"
                        row[key] = {"p95": B.upper_bound(pgrid, post),
                                    "bf_gt_1e-3": B.bayes_factor_exceeds(pgrid, post, 1e-3),
                                    "bf_gt_1e-1": B.bayes_factor_exceeds(pgrid, post, 1e-1)}
        out["by_lambda"][f"{lam:g}"] = row
    for label, f in (("active", f_active), ("passive", f_passive)):
        for shadow in (False, True):
            for q in inf["p_persist"]:
                for use_ext in (False, True):
                    post = B.marginal_over_lambda(lams, pgrid, f_own=f, shadowed=shadow,
                                                  n_gal=inf["n_gal_observed"], q=q, use_ext=use_ext)
                    key = f"{label}_shadow{int(shadow)}_q{q:g}_ext{int(use_ext)}"
                    out["marginal"][key] = {"p95": B.upper_bound(pgrid, post)}
    # branch mapping: an escaping-branch likelihood ratio applied to elicited planet-scale priors
    out["branch"] = {}
    for p_h in inf["human_prior_p_goo"]:
        for p_esc in (1e-3, 1e-2, 1e-1, 0.5):
            for lr in (1e-3, 1e-2, 0.1):
                out["branch"][f"ph{p_h:g}_pe{p_esc:g}_lr{lr:g}"] = B.branch_update(p_h, p_esc, lr)
    _dump("bayes.json", out)
    return out


# --- numbers --------------------------------------------------------------------
def _sci(x: float, sig: int = 2) -> str:
    """LaTeX 'a \\times 10^{b}' with sig significant figures; plain for 0.01..1000."""
    if x == 0:
        return "0"
    if 0.01 <= abs(x) < 1000:
        return f"{x:.{sig}g}"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10**e
    m_s = f"{m:.{sig - 1}f}"
    if m_s.startswith("10"):
        e += 1
        m_s = "1.0"
    return rf"\ensuremath{{{m_s}\times10^{{{e}}}}}"


def _time(s: float) -> str:
    """Human time: s, d, yr, kyr, Myr, Gyr."""
    if s < 3600:
        return f"{s:.0f}\\,s"
    if s < 86400 * 3:
        return f"{s / 3600:.1f}\\,h"
    if s < YR:
        return f"{s / 86400:.0f}\\,d"
    y = s / YR
    if y < 1e3:
        return f"{y:.0f}\\,yr"
    if y < 1e6:
        return f"{y / 1e3:.0f}\\,kyr"
    if y < 1e9:
        return f"{y / 1e6:.1f}\\,Myr".replace(".0\\,", "\\,")
    return f"{y / 1e9:.1f}\\,Gyr"


def stage_numbers(cfg: dict) -> pathlib.Path:
    d = out_dir()
    pl = json.loads((d / "planet.json").read_text())
    sy = json.loads((d / "system.json").read_text())
    pa = json.loads((d / "passive.json").read_text())
    ac = json.loads((d / "active.json").read_text())
    ba = json.loads((d / "bayes.json").read_text())
    r = cfg["replicator"]
    L: list[str] = ["% generated by `python -m seti.goo.run --stage numbers`; do not edit"]

    def cmd(name: str, val: str) -> None:
        L.append(rf"\newcommand{{\{name}}}{{{val}\xspace}}")

    cmd("SeedMassKg", _sci(pl["seed_mass_kg"]))
    cmd("SeedRadiusUm", f"{r['radius_um']:g}")
    cmd("ERepJkg", _sci(r["e_rep_J_per_kg"], 1))
    t = pl["tables"]["tau1000_T500"]
    cmd("BioDoublings", f"{t['biosphere']['doublings']:.0f}")
    cmd("CrustDoublings", f"{t['crust']['doublings']:.0f}")
    cmd("BulkDoublings", f"{t['bulk_planet']['doublings']:.0f}")
    cmd("BioExpTime", _time(t["biosphere"]["t_exponential_s"]))
    cmd("BioRadTime", _time(t["biosphere"]["t_radiative_s"]))
    cmd("BioSolarTime", _time(t["biosphere"]["t_solar_s"]))
    cmd("CrustRadTime", _time(t["crust"]["t_radiative_s"]))
    cmd("CrustSolarTime", _time(t["crust"]["t_solar_s"]))
    cmd("CrustGovTime", _time(t["crust"]["t_governing_s"]))
    cmd("BulkRadTime", _time(t["bulk_planet"]["t_radiative_s"]))
    cmd("BulkSolarTime", _time(t["bulk_planet"]["t_solar_s"]))
    cmd("BulkGovTime", _time(t["bulk_planet"]["t_governing_s"]))
    t2 = pl["tables"]["tau1000_T2000"]
    cmd("BulkRadTimeHot", _time(t2["bulk_planet"]["t_radiative_s"]))
    cmd("CrustRadTimeHot", _time(t2["crust"]["t_radiative_s"]))
    cmd("RadLimitFiveHundred", _sci(pl["radiative_limit_W"]["500.0"]))
    cmd("RadLimitTwoThousand", _sci(pl["radiative_limit_W"]["2000.0"]))
    cmd("AbsorbedSolar", _sci(pl["absorbed_solar_W"]))
    cmd("CrustFissionJ", _sci(pl["energy_reservoirs_J"]["crustal_fission"], 1))
    cmd("OceanDeuteriumJ", _sci(pl["energy_reservoirs_J"]["oceanic_deuterium"], 1))
    cmd("EarthDisassemblyLsun", _time(pl["disassembly_s"]["earth_at_L_sun"]))
    cmd("EarthDisassemblySolar", _time(pl["disassembly_s"]["earth_at_absorbed_solar"]))
    cmd("EarthDisassemblyRad", _time(pl["disassembly_s"]["earth_at_radiative_2000K"]))
    # tau scan for the biosphere
    for tau, nm in ((100.0, "Hundred"), (1e4, "TenK"), (3.2e7, "Year")):
        cmd(f"BioExpTimeTau{nm}", _time(pl["tables"][f"tau{tau:g}_T500"]["biosphere"]["t_exponential_s"]))
        cmd(f"CrustGovTimeTau{nm}", _time(pl["tables"][f"tau{tau:g}_T500"]["crust"]["t_governing_s"]))

    res = {row["name"]: row for row in sy["reservoirs"]}
    belt = res["asteroid belt"]
    kb = res["Kuiper belt"]
    cmd("BeltMassKg", _sci(belt["mass_kg"]))
    cmd("BeltConvTime", _time(belt["t_conversion_s"]))
    cmd("KuiperConvTime", _time(kb["t_conversion_s"]))
    cmd("BeltTempK", f"{belt['T_swarm_K']:.0f}")
    cmd("KuiperTempK", f"{kb['T_swarm_K']:.0f}")
    cmd("BeltCoverFilm", f"{belt['covering_fraction']['0.001']:.2g}")
    cmd("BeltCoverSheet", f"{belt['covering_fraction']['1']:.2g}")
    cmd("BeltCoverSlab", _sci(belt["covering_fraction"]["1000"]))
    cmd("KuiperCoverSheet", f"{kb['covering_fraction']['1']:.2g}")
    cmd("TerrConvTime", _time(res["terrestrial planets"]["t_conversion_s"]))
    cmd("TerrConvTimeFullL", _time(res["terrestrial planets"]["t_at_full_L_s"]))
    cmd("GiantConvTime", _time(res["giant planets"]["t_conversion_s"]))
    cmd("GiantConvTimeFullL", _time(res["giant planets"]["t_at_full_L_s"]))
    rr = belt["residue"]
    cmd("BeltResidueMicron", _time(rr["0.5um"]["t_residue_s"]))
    cmd("BeltResidueMm", _time(rr["1000um"]["t_residue_s"]))
    cmd("BeltResidueMetre", _time(rr["1e+06um"]["t_residue_s"]))
    cmd("BeltCollTime", _time(rr["0.5um"]["t_collisional_s"]))
    cmd("BeltResidueMmCover", _sci(rr["1000um"]["f_cov"]))
    cmd("BeltResidueMetreCover", _sci(rr["1e+06um"]["f_cov"]))
    cmd("MaxVisibleResidueMilli", _time(belt["max_visible_residue_s"]["0.001"]))
    cmd("MaxVisibleResidueCenti", _time(belt["max_visible_residue_s"]["0.01"]))
    cmd("MaxVisibleResidueDeci", _time(belt["max_visible_residue_s"]["0.1"]))
    cmd("KuiperMaxVisibleResidueCenti", _time(kb["max_visible_residue_s"]["0.01"]))
    b = sy["rate_bounds_vs_residue_lifetime_yr"]
    cmd("HephBoundTenK", _sci(b["10000"]["hephaistos_II_galaxy_per_yr"]))
    cmd("HephBoundMyr", _sci(b["1e+06"]["hephaistos_II_galaxy_per_yr"]))
    cmd("HephBoundTenGyr", _sci(b["1e+10"]["hephaistos_II_galaxy_per_yr"]))
    cmd("GhatFractionBound", _sci(sy["ghat_III"]["fraction_bound_95"]))

    cmd("BlowoutRadiusUm", f"{pa['blowout_radius_um']:.2f}")
    cmd("BetaMaxReach", f"{pa['beta_max_to_reach_1AU']:.2f}")
    cmd("WindowLoUm", f"{pa['deliverable_window_um'][0]:.2f}")
    cmd("WindowHiUm", f"{pa['deliverable_window_um'][1]:.2f}")
    cmd("VinfDevice", f"{pa['v_inf_device_kms']:.0f}")
    cmd("VinfBetaOne", f"{pa['v_inf_vs_beta']['1']:.0f}")
    cmd("AnnulusFillGyr", f"{pa['annulus_fill_time_Gyr']:.1f}")
    cmd("FocusingFactor", f"{pa['focusing_factor']:.1f}")
    cmd("BlownFractionColl", _sci(pa["blown_fraction_collisional"]))
    sd = pa["seeding"]
    k = "M2.4e+21_ts1e+08_t5"
    cmd("BeltDevicesReleased", _sci(sd[k]["N_released"]))
    cmd("BeltArrivalsUnsurvived", _sci(sd[k]["arrivals_per_yr_unsurvived"]))
    for ts, nm in ((1e6, "Myr"), (1e7, "TenMyr"), (1e8, "HundredMyr"), (1e9, "Gyr")):
        kk = f"M2.4e+21_ts{ts:g}_t5"
        cmd(f"BeltCumulative{nm}", _sci(sd[kk]["cumulative_arrivals"]))
    cmd("BioArrivalsUnsurvived", _sci(sd["M1e+15_ts1e+08_t5"]["arrivals_per_yr_unsurvived"]))
    cmd("BioCumulativeGyr", _sci(sd["M1e+15_ts1e+09_t5"]["cumulative_arrivals"]))
    cmd("BioCumulativeHundredMyr", _sci(sd["M1e+15_ts1e+08_t5"]["cumulative_arrivals"]))

    g = ac["front_grid"]
    cmd("FillSlowSlow", _time(g["v0.0001_tau1000"]["t_fill_Myr"] * MYR))
    cmd("FillMidMid", _time(g["v0.01_tau100"]["t_fill_Myr"] * MYR))
    cmd("FillFastFast", _time(g["v0.1_tau10"]["t_fill_Myr"] * MYR))
    cmd("FillVoyagerTen", _time(g["v0.0001_tau10"]["t_fill_Myr"] * MYR))
    cmd("FrontMidMid", f"{g['v0.01_tau100']['front_speed_kms']:.0f}")
    cmd("EventHorizonGly", f"{ac['event_horizon_Gly']:.1f}")
    for row in ac["cosmic_reach"]:
        nm = {0.1: "Tenth", 0.5: "Half", 0.8: "EightTenths", 0.99: "NinetyNine", 1.0: "Light"}[row["v_c"]]
        cmd(f"Reach{nm}Gly", f"{row['reach_Gly']:.1f}")
        cmd(f"Galaxies{nm}", _sci(row["n_galaxies"], 1))

    cmd("WindowGyr", f"{cfg['inference']['window_Gyr']:g}")
    cmd("NGalObserved", _sci(cfg["inference"]["n_gal_observed"], 1))
    bl = ba["by_lambda"]
    for lam, nm in (("1", "One"), ("100", "Hundred"), ("10000", "TenK")):
        row = bl[lam]
        cmd(f"PnaiveLam{nm}", _sci(row["active_shadow0_q1_ext0"]["p95"]))
        cmd(f"PshadowLam{nm}", _sci(row["active_shadow1_q1_ext0"]["p95"]))
        cmd(f"PextPersistLam{nm}", _sci(row["active_shadow1_q1_ext1"]["p95"]))
        cmd(f"PextDeadLam{nm}", _sci(row["active_shadow1_q0.0001_ext1"]["p95"]))
        cmd(f"PpassiveNaiveLam{nm}", _sci(row["passive_shadow0_q1_ext0"]["p95"]))
    lb = ba["likelihood_bounds"]
    for lam, nm in (("1", "One"), ("100", "Hundred"), ("10000", "TenK")):
        cmd(f"LbOwnLam{nm}", _sci(lb[lam]["own_active"]))
        cmd(f"LbOwnPassiveLam{nm}", _sci(lb[lam]["own_passive"]))
        cmd(f"LbExtPersistLam{nm}", _sci(lb[lam]["ext_persist"]))
        cmd(f"LbExtDeadLam{nm}", _sci(lb[lam]["ext_dead_1e-4"]))
    m = ba["marginal"]
    cmd("PmargNaive", _sci(m["active_shadow0_q1_ext0"]["p95"]))
    cmd("PmargShadow", _sci(m["active_shadow1_q1_ext0"]["p95"]))
    cmd("PmargExtPersist", _sci(m["active_shadow1_q1_ext1"]["p95"]))
    cmd("PmargExtDead", _sci(m["active_shadow1_q0.0001_ext1"]["p95"]))
    br = ba["branch"]
    cmd("BranchRatioTypical", f"{br['ph0.001_pe0.01_lr0.001']['ratio']:.3f}")
    cmd("BranchRatioHalf", f"{br['ph0.001_pe0.5_lr0.001']['ratio']:.2f}")
    cmd("BranchRatioTenth", f"{br['ph0.001_pe0.1_lr0.001']['ratio']:.2f}")

    tex = ROOT / "paper" / "goo" / "numbers.tex"
    tex.parent.mkdir(parents=True, exist_ok=True)
    tex.write_text("\n".join(L) + "\n")
    return tex


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["planet", "system", "passive", "active", "bayes", "figures", "numbers", "all"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    stages = [args.stage] if args.stage != "all" else ["planet", "system", "passive", "active", "bayes", "figures", "numbers"]
    for st in stages:
        if st == "planet":
            stage_planet(cfg)
        elif st == "system":
            stage_system(cfg)
        elif st == "passive":
            stage_passive(cfg)
        elif st == "active":
            stage_active(cfg)
        elif st == "bayes":
            stage_bayes(cfg)
        elif st == "figures":
            from .figures import make_all
            make_all(cfg)
        elif st == "numbers":
            stage_numbers(cfg)
        print(f"[goo] stage {st} done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
