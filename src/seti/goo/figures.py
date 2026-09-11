"""Manuscript figures for the GOO paper.  Reads results/goo/*.json; writes
results/goo/figures/*.pdf and .png.

Palette: one hue per branch, fixed order, never cycled; direct labels; no dual axes.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import bayes as B  # noqa: E402
from .constants import GYR, MYR, YR  # noqa: E402
from .run import load_cfg, out_dir  # noqa: E402

COL = {
    "planet": "#1f5fbf",    # blue
    "system": "#c2571a",    # orange-brown
    "passive": "#2a8a5c",   # green
    "active": "#7a3fa0",    # purple
    "ext": "#8a8a8a",       # grey
    "ink": "#222222",
}
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#dddddd", "grid.linewidth": 0.5, "figure.dpi": 150,
})


def _fig_dir() -> pathlib.Path:
    d = out_dir() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, name: str) -> None:
    d = _fig_dir()
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(d / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


# 1. Timescale ladder: governing conversion / fill time at each scale, passive vs active
def fig_ladder(cfg: dict) -> None:
    d = out_dir()
    pl = json.loads((d / "planet.json").read_text())
    sy = json.loads((d / "system.json").read_text())
    pa = json.loads((d / "passive.json").read_text())
    ac = json.loads((d / "active.json").read_text())
    t = pl["tables"]["tau1000_T500"]
    res = {r["name"]: r for r in sy["reservoirs"]}
    rows = [
        ("biosphere (exponential)", t["biosphere"]["t_exponential_s"], "planet"),
        ("biosphere (radiative ceiling)", t["biosphere"]["t_radiative_s"], "planet"),
        ("crust (governing)", t["crust"]["t_governing_s"], "planet"),
        ("bulk planet (governing)", t["bulk_planet"]["t_governing_s"], "planet"),
        ("asteroid belt", res["asteroid belt"]["t_conversion_s"], "system"),
        ("Kuiper belt", res["Kuiper belt"]["t_conversion_s"], "system"),
        ("terrestrial planets (lift)", res["terrestrial planets"]["t_conversion_s"], "system"),
        ("giant planets (lift)", res["giant planets"]["t_conversion_s"], "system"),
        ("annulus, passive (phase mixing)", pa["annulus_fill_time_Gyr"] * GYR, "passive"),
        ("disc, passive (radial migration)", 10 * GYR, "passive"),
        ("galaxy, active 0.01c", ac["front_grid"]["v0.01_tau100"]["t_fill_Myr"] * MYR, "active"),
        ("galaxy, active 30 km/s", ac["front_grid"]["v0.0001_tau1000"]["t_fill_Myr"] * MYR, "active"),
        ("galaxy, active 0.1c", ac["front_grid"]["v0.1_tau10"]["t_fill_Myr"] * MYR, "active"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    y = np.arange(len(rows))[::-1]
    for yi, (_label, ts, kind) in zip(y, rows, strict=True):
        ax.barh(yi, ts / YR, left=1e-8, color=COL[kind], height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 3e10)
    ax.set_xlabel("time (yr)")
    ax.axvline(13.8e9, color=COL["ink"], lw=0.8, ls="--")
    ax.text(13.8e9, len(rows) - 0.5, " age of the universe", va="top", ha="right", fontsize=7, rotation=90)
    ax.axvline(4.5e9, color=COL["ink"], lw=0.8, ls=":")
    ax.text(4.5e9, len(rows) - 0.5, " age of the Sun", va="top", ha="right", fontsize=7, rotation=90)
    for kind, lab in (("planet", "planet"), ("system", "system"), ("passive", "galaxy, passive"), ("active", "galaxy, active")):
        ax.barh(-10, 1, color=COL[kind], label=lab)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.legend(loc="upper right", frameon=False, ncol=1)
    ax.set_title("Time to convert or fill each scale (tau = 1000 s, 500 K, e_rep = 1e7 J/kg)")
    _save(fig, "fig1_ladder")


# 2. Active front: fill time vs ship speed for three build times
def fig_front(cfg: dict) -> None:
    from . import active as A
    a = cfg["active"]
    v = np.logspace(-4.5, 0, 200)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    shades = ["#b9a3d1", "#7a3fa0", "#3f1f5a"]
    for tau, c in zip(a["tau_build_yr"], shades, strict=True):
        t = np.array([A.galaxy_fill_time_s(a["R_galaxy_kpc"], vi, tau, a["d_hop_pc"]) / YR for vi in v])
        ax.plot(v, t, color=c, lw=2)
        ax.text(v[-1], t[-1], f"  tau = {tau:g} yr", color=c, va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ship speed (c)")
    ax.set_ylabel("time to fill the galaxy (yr)")
    ax.set_xlim(3e-5, 3)
    ax.axhline(4.5e9, color=COL["ink"], lw=0.8, ls=":")
    ax.text(3.5e-5, 4.5e9, "age of the Sun", va="bottom", fontsize=7)
    ax.set_title("Active branch: galaxy fill time")
    _save(fig, "fig2_front")


# 3. Passive seeding: cumulative intact arrivals per planet vs survival time
def fig_survival(cfg: dict) -> None:
    """The survival scan: reach, planets within reach, R0 and the galaxy-crossing time of
    a chain of seedings, against the functional e-folding time in interstellar space,
    from kiloyears to infinity, for the four carrier classes."""
    d = out_dir()
    sv = json.loads((d / "survival.json").read_text())
    shades = ["#2a8a5c", "#1f5fbf", "#c2571a", "#7a3fa0"]
    short = ["sub-micron, blown out", "micron dust-sized", "in 100-m planetesimals", "in impact-ejected rocks"]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5))
    ax_r, ax_R, ax_t = axes
    xinf = 3e10   # where the tau = infinity point is drawn
    for car, col, lab in zip(sv["carriers"], shades, short, strict=True):
        rows = [r for r in car["rows"] if not np.isinf(r["tau_yr"])]
        inf_row = [r for r in car["rows"] if np.isinf(r["tau_yr"])][0]
        tau = np.array([r["tau_yr"] for r in rows])
        # panel 1: planets within reach (all carriers share the same speed except the first)
        npl = np.array([r["planets_within_reach"] for r in rows])
        ax_r.plot(tau, np.maximum(npl, 1e-7), color=col, lw=2)
        if lab in ("sub-micron, blown out", "in impact-ejected rocks"):
            i = int(np.searchsorted(tau, 3e6))
            ax_r.text(tau[i], npl[i] * 4, f"{car['v_inf_kms']:.0f} km/s" +
                      (" (blown out)" if lab.startswith("sub") else " (all other carriers)"),
                      color=col, fontsize=7, rotation=52, rotation_mode="anchor", ha="left", va="bottom")
        # panel 2: R0
        R0 = np.array([r["R0"] for r in rows])
        ax_R.plot(tau, R0, color=col, lw=2)
        ax_R.plot([xinf], [inf_row["R0"]], "o", color=col, ms=4)
        ax_R.text(xinf * 1.6, inf_row["R0"], lab, color=col, fontsize=7, va="center")
        # panel 3: time for the chain to cross the Galaxy
        tg = np.array([r["t_galaxy_yr"] if r["front"] else np.nan for r in rows])
        ax_t.plot(tau, tg, color=col, lw=2)
        if inf_row["front"] and np.isfinite(inf_row["t_galaxy_yr"]):
            ax_t.plot([xinf], [inf_row["t_galaxy_yr"]], "o", color=col, ms=4)
            ax_t.text(xinf * 1.6, inf_row["t_galaxy_yr"], lab, color=col, fontsize=7, va="center")
        first = next((r for r in rows if r["front"]), None)
        if first is not None and first["t_galaxy_yr"] < 1e11:
            ax_t.axvline(first["tau_yr"], color=col, lw=0.6, ls=":")
    hop = sv["carriers"][0]["t_hop_yr"]
    ax_r.axhline(1.0, color=COL["ink"], lw=0.8, ls="--")
    ax_r.text(1.2e3, 1.4, "one planet in reach", fontsize=7)
    ax_r.set_ylabel("Earth-like planets within reach $v_\\infty\\tau$")
    ax_r.set_title("Reach")
    ax_R.axhline(1.0, color=COL["ink"], lw=0.8, ls="--")
    ax_R.text(1.2e3, 1.6, "$R_0=1$", fontsize=7)
    ax_R.set_ylabel("strict landings per source in 5 Gyr, $R_0(\\tau)$")
    ax_R.set_title("Landings (mixing-independent)")
    ax_R.set_ylim(1e-12, 1e23)
    ax_t.axvline(hop, color=COL["ink"], lw=0.8, ls="--")
    ax_t.text(hop * 1.3, 2.5e9, "hop to the\nnearest star", fontsize=7, va="bottom")
    ax_t.set_ylabel("time for a chain of seedings to cross the Galaxy (yr)")
    ax_t.set_title("Front")
    ax_t.set_ylim(1e8, 1e14)
    ax_t.text(3e5, 3e12, "rock-borne: no chain below $\\tau\\sim10^{10}$ yr,\nthen one hop per 2 Gyr", fontsize=7)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e3, 2e11)
        ax.set_xticks([1e3, 1e5, 1e7, 1e9, xinf])
        ax.set_xticklabels(["$10^3$", "$10^5$", "$10^7$", "$10^9$", "$\\infty$"])
        ax.set_xlabel("functional survival e-folding time $\\tau$ (yr)")
    fig.tight_layout()
    _save(fig, "fig3_survival")


# 4. Residue lifetime and the rate bound it implies
def fig_residue(cfg: dict) -> None:
    from . import system as S
    r = cfg["replicator"]
    sizes = np.logspace(-1, 7, 200)  # um
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for M, rad, c, lab in ((2.4e21, 2.7, "#c2571a", "asteroid belt (2.4e21 kg at 2.7 AU)"),
                           (1.2e23, 40.0, "#e2a37a", "Kuiper belt (1.2e23 kg at 40 AU)")):
        t = np.array([S.residue_lifetime_s(rad, S.covering_fraction_for_size(M, sz, r["density_gcc"], rad),
                                           sz, r["density_gcc"])["t_residue_s"] / YR for sz in sizes])
        ax.plot(sizes, t, color=c, lw=2)
        ax.text(sizes[-1], t[-1], "  " + lab, color=c, va="center", fontsize=7)
    # the detectability floor: a residue above f_min lives at most P_orb / f_min
    for fm, ls in ((1e-3, "--"), (1e-2, ":")):
        ax.axhline(S.max_visible_residue_s(2.7, fm) / YR, color=COL["ink"], lw=0.8, ls=ls)
        ax.text(0.12, S.max_visible_residue_s(2.7, fm) / YR * 1.3, f"longest visible above f = {fm:g} at 2.7 AU", fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("device size (um)")
    ax.set_ylabel("dead-swarm residue lifetime (yr)")
    ax.set_title("How long a dead system-scale goo stays visible")
    _save(fig, "fig4_residue")


# 5. Posterior 95 % bounds on the escaping-branch probability vs Lambda
def fig_bayes(cfg: dict) -> None:
    inf = cfg["inference"]
    W = inf["window_Gyr"] * GYR
    f_active = (W - inf["t_fill_active_Myr"] * MYR) / W
    pgrid = B.loguniform_grid(inf["p_prior"]["lo"], inf["p_prior"]["hi"])
    lams = np.logspace(-2, 6, 60)
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    curves = [
        ("own galaxy, no shadow (SIA)", dict(shadowed=False, use_ext=False, q=1.0), "#7a3fa0", "-"),
        ("own galaxy, anthropic shadow (SSA)", dict(shadowed=True, use_ext=False, q=1.0), "#7a3fa0", ":"),
        ("+ 1e5 galaxies, persistent residue", dict(shadowed=True, use_ext=True, q=1.0), "#8a8a8a", "-"),
        ("+ 1e5 galaxies, residue 1e-4 of window", dict(shadowed=True, use_ext=True, q=1e-4), "#8a8a8a", "--"),
    ]
    for lab, kw, c, ls in curves:
        ub = [B.upper_bound(pgrid, B.posterior_p(lam, pgrid, f_own=f_active, n_gal=inf["n_gal_observed"], **kw)) for lam in lams]
        ax.plot(lams, ub, color=c, ls=ls, lw=2, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("expected number of civilisations in the window, Lambda")
    ax.set_ylabel("95 % upper bound on p (escaping goo per civilisation)")
    ax.set_ylim(1e-7, 2)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("What the absence constrains")
    _save(fig, "fig5_bayes")


# 6. Branch map: how an astronomical likelihood ratio on the escaping branch moves p_goo
def fig_branch(cfg: dict) -> None:
    p_esc = np.logspace(-4, 0, 200)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for lr, c in ((1e-1, "#9ccdb3"), (1e-2, "#2a8a5c"), (1e-3, "#155c3b")):
        ratio = [B.branch_update(1e-3, pe, lr)["ratio"] for pe in p_esc]
        ax.plot(p_esc, ratio, color=c, lw=2)
        ax.text(p_esc[-1], ratio[-1], f"  LR = {lr:g}", color=c, va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("fraction of goo accidents that reach other stars, p_escape")
    ax.set_ylabel("posterior / prior for the planet-scale accident probability")
    ax.set_ylim(0, 1.05)
    ax.set_title("The update on our own risk")
    _save(fig, "fig6_branch")


# 7. The contact graph: single-source R0 per channel, strict and loose, against the threshold
def fig_contact(cfg: dict) -> None:
    d = out_dir()
    ct = json.loads((d / "contact.json").read_text())
    rows = [r for r in ct["channels"] if r["R0_loose"] > 0 or r["R0_strict"] > 0]
    labels = {
        "interstellar objects (>100 m)": "interstellar objects",
        "impact ejecta from a planet (rocks > 1 m)": "impact ejecta (rocks)",
        "free-floating planets": "free-floating planets",
        "interstellar dust (ISM grains, 0.1-1 um)": "interstellar dust",
        "radiation-pressure grains from a converted belt": "blown-out belt devices",
        "stellar encounters (within the Oort cloud, 2e4 AU)": "stellar flybys",
        "birth-cluster exchange (first ~10-100 Myr)": "birth-cluster rocks",
    }
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    y = np.arange(len(rows))[::-1]
    floor = 1e-6
    for yi, r in zip(y, rows, strict=True):
        ax.barh(yi + 0.18, max(r["R0_loose"], floor), left=floor, height=0.34, color="#e2a37a")
        ax.barh(yi - 0.18, max(r["R0_strict"], floor), left=floor, height=0.34, color="#c2571a")
    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(r["name"], r["name"]) for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(floor, 1e40)
    ax.axvline(1.0, color=COL["ink"], lw=0.8, ls="--")
    ax.text(1.3, len(rows) - 0.45, "R0 = 1: chain propagates", fontsize=7, va="top")
    ax.barh(-10, 1, color="#e2a37a", label="loose: passes within 100 AU")
    ax.barh(-10, 1, color="#c2571a", label="strict: lands on an Earth-sized planet")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlabel("systems touched per touched system over 5 Gyr (R0), mass survival only")
    ax.set_title("The contact graph: what one system's material reaches")
    _save(fig, "fig7_contact")


def make_all(cfg: dict | None = None) -> None:
    cfg = cfg or load_cfg()
    fig_ladder(cfg)
    fig_front(cfg)
    fig_survival(cfg)
    fig_residue(cfg)
    fig_bayes(cfg)
    fig_branch(cfg)
    fig_contact(cfg)
