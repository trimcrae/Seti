"""The contact graph of the Galaxy: every channel by which mass moves from one
planetary system to another, and how long it takes for everything to be
touched by something that was touched by something else.

Two definitions of "touch":
  STRICT  a carrier lands on an Earth-sized planet of the target system
          (cross-section pi R^2 (1 + v_esc^2/v^2))
  LOOSE   a carrier passes through the target system, within r_loose (~100 AU)
          (cross-section pi r_loose^2 (1 + v_esc(r_loose)^2/v^2)); for a bound
          capture the literature rate is used where it exists.

Two questions:
  NATURAL   every system emits.  The standing carrier density is n = N_sys *
            N_c / V (after mixing), and a target is touched at rate n v sigma.
            The time for *everything* to be touched is t_mix + ln(N_sys) / rate.
  EPIDEMIC  one system is touched first.  Its carriers touch others at rate
            r_1 = (N_c / V) v sigma per target; SI dynamics on N_sys systems has
            logistic rate k = r_1 N_sys and takes ~ 2 ln(N_sys) / k to go from
            one to all, after the mixing delay.  R0 = r_1 N_sys W is the number
            of systems one source touches over the window W; below 1 the chain
            does not propagate.  Functional survival with e-folding time tau
            replaces W by the integral tau (1 - exp(-W/tau)): landings do not
            wait for mixing (survival.py), so the factor is (tau/W)(1 - e^{-W/tau}),
            not exp(-t_mix/tau).

Local channels (stellar encounters, supernova ejecta) do not phase-mix; they
are contact rates per system with a range, and their epidemic spreads as a
front at ~ range / waiting time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import AU, KM, M_EARTH, M_SUN, PC, R_EARTH, YR, G
from .passive import annulus_volume_m3


@dataclass
class ChannelResult:
    name: str
    kind: str
    rate_strict_natural_per_planet_yr: float
    rate_loose_natural_per_system_yr: float
    t_all_touched_strict_yr: float
    t_all_touched_loose_yr: float
    r1_strict_per_planet_yr: float
    r1_loose_per_system_yr: float
    R0_strict: float
    R0_loose: float
    t_epidemic_strict_yr: float
    t_epidemic_loose_yr: float
    R0_strict_functional: dict
    notes: str
    rate_capture_natural_per_system_yr: float = 0.0
    R0_capture: float = 0.0


def sigma_strict_m2(v_ms: float) -> float:
    ve = math.sqrt(2.0 * G * M_EARTH / R_EARTH)
    return math.pi * R_EARTH**2 * (1.0 + (ve / v_ms) ** 2)


def sigma_loose_m2(v_ms: float, r_loose_AU: float) -> float:
    r = r_loose_AU * AU
    ve = math.sqrt(2.0 * G * M_SUN / r)
    return math.pi * r**2 * (1.0 + (ve / v_ms) ** 2)


def _epidemic_time(r1: float, n_targets: float, t_mix: float) -> float:
    """SI logistic from one touched system to all, after the mixing delay (yr)."""
    k = r1 * n_targets
    if k <= 0:
        return float("inf")
    return t_mix + 2.0 * math.log(n_targets) / k


def mixed_channel(ch: dict, cfg: dict, W_yr: float) -> ChannelResult:
    """A channel whose carriers phase-mix over the annulus."""
    V = annulus_volume_m3(cfg["passive"]["galaxy"]["R_sun_kpc"], cfg["passive"]["galaxy"]["annulus_width_kpc"],
                          cfg["passive"]["galaxy"]["scale_height_kpc"])
    c = cfg["contact"]
    N_sys = c["n_systems_annulus"]
    N_pl = N_sys * c["n_earthlike_per_star"]
    v = ch["speed_kms"] * KM
    t_mix = c["t_mix_annulus_yr"]
    ss = sigma_strict_m2(v)
    sl = sigma_loose_m2(v, c["r_loose_AU"])
    t_mass = ch.get("t_mass_survive_yr", 1e12)
    if "standing_flux_per_m2_s" in ch:
        # the natural density is measured directly as a flux F = n v
        n_nat = ch["standing_flux_per_m2_s"] / v
        N_c = n_nat * V / N_sys          # implied carriers per system in the standing population
        kind = "measured flux"
    elif "standing_density_per_AU3" in ch:
        n_nat = ch["standing_density_per_AU3"] / AU**3
        N_c = n_nat * V / N_sys
        kind = "measured density"
    else:
        N_c = ch["carriers_per_source"]
        # standing population after emission: carriers that survive as mass
        n_nat = N_sys * N_c * min(1.0, t_mass / W_yr) / V
        kind = "emitted carriers"
    n_1 = N_c * min(1.0, t_mass / W_yr) / V
    rate_s_nat = n_nat * v * ss * YR
    rate_l_nat = n_nat * v * sl * YR
    r1_s = n_1 * v * ss * YR
    r1_l = n_1 * v * sl * YR
    t_all_s = t_mix + (math.log(N_pl) / rate_s_nat if rate_s_nat > 0 else float("inf"))
    t_all_l = t_mix + (math.log(N_sys) / rate_l_nat if rate_l_nat > 0 else float("inf"))
    R0_s = r1_s * N_pl * (W_yr - t_mix) if W_yr > t_mix else 0.0
    R0_l = r1_l * N_sys * (W_yr - t_mix) if W_yr > t_mix else 0.0
    if ch.get("loose_is_capture_count"):
        # the literature counts bound captures of this system's material by other
        # systems directly (Siraj & Loeb 2020); use it as the loose R0 and derive
        # the strict one through the landing fraction of a captured body
        R0_l = N_c
        R0_s = N_c * 5.0e-11
        r1_l = R0_l / (N_sys * max(W_yr - t_mix, 1.0))
        r1_s = R0_s / (N_pl * max(W_yr - t_mix, 1.0))
    # bound capture, where the literature gives a rate per system
    cap_nat = ch.get("capture_rate_per_yr_natural", 0.0)
    R0_cap = cap_nat / N_sys * N_sys * (W_yr - t_mix) * (N_c / max(N_c, 1.0)) if cap_nat else 0.0
    # single-source share of the natural capture rate: this source's carriers are
    # N_c out of N_sys * N_c in the standing population
    R0_cap = cap_nat * (W_yr - t_mix) if cap_nat else 0.0
    func = {}
    for tf in ch.get("t_func_survive_yr", [1e12]):
        # landings are mixing-independent (survival.py): the window W is replaced by
        # the integral of the survival curve, tau (1 - e^{-W/tau})
        S = (tf / W_yr) * (1.0 - math.exp(-W_yr / tf))
        func[f"{tf:g}"] = R0_s * S
    return ChannelResult(
        name=ch["name"], kind=kind,
        rate_strict_natural_per_planet_yr=rate_s_nat, rate_loose_natural_per_system_yr=rate_l_nat,
        t_all_touched_strict_yr=t_all_s, t_all_touched_loose_yr=t_all_l,
        r1_strict_per_planet_yr=r1_s, r1_loose_per_system_yr=r1_l,
        R0_strict=R0_s, R0_loose=R0_l,
        t_epidemic_strict_yr=_epidemic_time(r1_s, N_pl, t_mix) if R0_s > 1 else float("inf"),
        t_epidemic_loose_yr=_epidemic_time(r1_l, N_sys, t_mix) if R0_l > 1 else float("inf"),
        R0_strict_functional=func,
        notes=ch.get("source", ""),
        rate_capture_natural_per_system_yr=cap_nat,
        R0_capture=R0_cap,
    )


def encounter_channel(ch: dict, cfg: dict, W_yr: float) -> ChannelResult:
    """Stellar flybys: rate within r scales as r^2 from the 1 pc rate.  A flyby is a
    LOOSE touch of both systems by each other's outer material; it lands nothing on a
    planet by itself (strict rate 0).  The epidemic spreads as a front: each touched
    system touches its flyby partners, who are random field stars, so after mixing the
    process is well mixed with per-pair rate = encounter rate / N_sys."""
    c = cfg["contact"]
    N_sys = c["n_systems_annulus"]
    rate_1pc = ch["encounter_rate_per_Myr_within_1pc"] / 1e6
    r_pc = ch["r_encounter_AU"] * AU / PC
    rate_l_nat = rate_1pc * r_pc**2
    r1_l = rate_l_nat / N_sys
    R0_l = r1_l * N_sys * W_yr
    return ChannelResult(
        name=ch["name"], kind="local encounters",
        rate_strict_natural_per_planet_yr=0.0, rate_loose_natural_per_system_yr=rate_l_nat,
        t_all_touched_strict_yr=float("inf"),
        t_all_touched_loose_yr=math.log(N_sys) / rate_l_nat,
        r1_strict_per_planet_yr=0.0, r1_loose_per_system_yr=r1_l,
        R0_strict=0.0, R0_loose=R0_l,
        t_epidemic_strict_yr=float("inf"),
        t_epidemic_loose_yr=_epidemic_time(r1_l, N_sys, 0.0) if R0_l > 1 else float("inf"),
        R0_strict_functional={}, notes=ch.get("source", ""),
    )


def supernova_channel(ch: dict, cfg: dict, W_yr: float) -> ChannelResult:
    """Supernova ejecta reaching a planet's surface: a STRICT touch by the dying
    star's material, at the measured deposition-episode rate.  Not a carrier a
    replicator survives; included for the natural graph."""
    rate = ch["events_per_Myr"] / 1e6
    c = cfg["contact"]
    N_pl = c["n_systems_annulus"] * c["n_earthlike_per_star"]
    return ChannelResult(
        name=ch["name"], kind="local events",
        rate_strict_natural_per_planet_yr=rate, rate_loose_natural_per_system_yr=rate,
        t_all_touched_strict_yr=math.log(N_pl) / rate, t_all_touched_loose_yr=math.log(N_pl) / rate,
        r1_strict_per_planet_yr=0.0, r1_loose_per_system_yr=0.0, R0_strict=0.0, R0_loose=0.0,
        t_epidemic_strict_yr=float("inf"), t_epidemic_loose_yr=float("inf"),
        R0_strict_functional={}, notes=ch.get("source", ""),
    )


def cluster_channel(ch: dict, cfg: dict, W_yr: float) -> ChannelResult:
    """The birth cluster: for its first ~100 Myr a system sits among ~10^3 siblings at
    ~1 km/s relative speed, and weak transfer moves rocks between them with a
    per-rock probability far above the field value.  R0 = rocks * P * (fraction of
    transfers that land on a planet ~ sigma_strict / sigma_loose) over the cluster
    lifetime; only a goo that exists while its star is still in its cluster uses it."""
    R0_l = ch["bodies_transferred_to_nearest_neighbour"]
    R0_s = R0_l * ch["planet_landing_probability_per_captured_body"]
    n_sib = ch["n_siblings"]
    t_cl = ch["t_cluster_Myr"] * 1e6
    return ChannelResult(
        name=ch["name"], kind="birth cluster (first 100 Myr only)",
        rate_strict_natural_per_planet_yr=R0_s * n_sib / t_cl, rate_loose_natural_per_system_yr=R0_l * n_sib / t_cl,
        t_all_touched_strict_yr=float("inf"), t_all_touched_loose_yr=float("inf"),
        r1_strict_per_planet_yr=R0_s / t_cl, r1_loose_per_system_yr=R0_l / t_cl,
        R0_strict=R0_s, R0_loose=R0_l,
        t_epidemic_strict_yr=float("inf"), t_epidemic_loose_yr=float("inf"),
        R0_strict_functional={}, notes=ch.get("source", "") + "; confined to the cluster, so no galaxy-wide epidemic",
    )


def run_contact(cfg: dict, W_yr: float = 5e9) -> list[ChannelResult]:
    out = []
    for ch in cfg["contact"]["channels"]:
        if "encounter_rate_per_Myr_within_1pc" in ch:
            out.append(encounter_channel(ch, cfg, W_yr))
        elif "events_per_Myr" in ch:
            out.append(supernova_channel(ch, cfg, W_yr))
        elif "bodies_transferred_to_nearest_neighbour" in ch:
            out.append(cluster_channel(ch, cfg, W_yr))
        else:
            out.append(mixed_channel(ch, cfg, W_yr))
    return out


def iso_impacts_on_earth_over_age(cfg: dict, age_yr: float = 4.5e9) -> float:
    """Sanity number: interstellar-object impacts on Earth over its history from the
    measured number density (natural, all systems emitting)."""
    for r in run_contact(cfg):
        if r.name.startswith("interstellar objects"):
            return r.rate_strict_natural_per_planet_yr * age_yr
    return float("nan")
