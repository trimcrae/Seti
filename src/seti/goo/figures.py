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
def fig_passive(cfg: dict) -> None:
    from . import passive as P
    from .constants import KM, M_EARTH, R_EARTH
    p = cfg["passive"]
    r = cfg["replicator"]
    g = p["galaxy"]
    v_in = p["inflow_speed_kms"] * KM
    ts = np.logspace(5, 10, 120)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    shades = {"1e+15": "#9ccdb3", "2.4e+21": "#2a8a5c", "1.2e+23": "#155c3b"}
    labels = {"1e+15": "a biosphere (1e15 kg)", "2.4e+21": "the asteroid belt", "1.2e+23": "the Kuiper belt"}
    for M in p["release_mass_kg"]:
        cum = np.array([P.passive_seeding(M, r["radius_um"], r["density_gcc"], 1.0, g, R_EARTH, M_EARTH,
                                          v_in, t * YR, 5 * GYR)["cumulative_arrivals"] for t in ts])
        k = f"{M:g}"
        ax.plot(ts, np.maximum(cum, 1e-30), color=shades[k], lw=2)
        ax.text(ts[-1], max(cum[-1], 1e-30), "  " + labels[k], color=shades[k], va="center", fontsize=7)
    ax.axhline(1.0, color=COL["ink"], lw=0.8, ls="--")
    ax.text(1.1e5, 1.3, "one intact device per planet", fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1e18)
    ax.set_xlabel("functional survival e-folding time in the ISM (yr)")
    ax.set_ylabel("intact arrivals per Earth-like planet in 5 Gyr")
    ax.set_title("Passive branch: seeding of the solar annulus")
    _save(fig, "fig3_passive")


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


def make_all(cfg: dict | None = None) -> None:
    cfg = cfg or load_cfg()
    fig_ladder(cfg)
    fig_front(cfg)
    fig_passive(cfg)
    fig_residue(cfg)
    fig_bayes(cfg)
    fig_branch(cfg)
