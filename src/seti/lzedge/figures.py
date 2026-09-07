"""Figures for LZEDGE, drawn from the stage outputs in results/lzedge/.

  tails.png        lab-frame speed distributions of the halo variants on the event
                   date, with v_min(248 keV; delta) marked for a few splittings
  modulation.png   window rate through the year for edge models, the event date
                   and the run span marked
  calendar.png     the reachability calendar: for each delta, the dates on which
                   a 248 keV recoil is kinematically possible (v_min <= v_esc + v_lab(t))
  scan_map.png     timing Bayes factor and energy density at 248 keV over (delta)
                   for each halo model at fixed mass
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import kinematics as K  # noqa: E402
from .earth import v_lab_kms, year_grid  # noqa: E402
from .halo import lmc_tail, shm, shm_plus_plus, shm_tail  # noqa: E402
from .rate import Efficiency  # noqa: E402
from .timing import EventModel  # noqa: E402


def fig_tails(cfg: dict, out: pathlib.Path, deltas=(300.0, 350.0, 380.0), m_chi: float = 1000.0) -> pathlib.Path:
    t_obs = cfg["event"]["date"]
    E = float(cfg["event"]["energy_keV"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    variants = [
        ("SHM sharp, v_esc 544", shm_tail(238.0, 544.0, "sharp"), 238.0),
        ("SHM soft, v_esc 544", shm_tail(238.0, 544.0, "soft"), 238.0),
        ("SHM (v_esc-v)^2.5 tail", shm_tail(238.0, 544.0, "power", k=2.5), 238.0),
        ("SHM++ (Sausage), v_esc 528", shm_plus_plus(), 233.0),
        ("SHM sharp, v_esc 500", shm(238.0, 500.0), 238.0),
        ("SHM sharp, v_esc 580", shm(238.0, 580.0), 238.0),
        ("SHM + LMC 0.26 % (2609.04175)", lmc_tail(shm(238.0, 544.0), 570.0, 100.0, 0.0026, cut_kms=200.0), 238.0),
    ]
    for label, halo, v0 in variants:
        vl = v_lab_kms(t_obs, v0)
        f = halo.lab_speed_distribution(vl)
        ax.plot(halo.v_grid, f, lw=1.4, label=label)
    for d in deltas:
        vm = float(K.v_min_kms(E, m_chi, d))
        ax.axvline(vm, color="k", ls=":", lw=0.9)
        ax.text(vm + 3, 2e-3, f"δ={d:.0f} keV", rotation=90, va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e-2)
    ax.set_xlim(300, 1000)
    ax.set_xlabel("lab-frame speed v [km/s] on " + t_obs)
    ax.set_ylabel("f_lab(v) [s/km]")
    ax.set_title(f"the high-speed tail on the event date; v_min(248 keV) for m_χ = {m_chi:.0f} GeV")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    p = out / "tails.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_modulation(cfg: dict, out: pathlib.Path, deltas=(0.0, 300.0, 360.0, 380.0), m_chi: float = 1000.0) -> pathlib.Path:
    ev, run, win = cfg["event"], cfg["run"], cfg["window"]
    eff = Efficiency.from_table(win["efficiency"])
    halo = shm(238.0, 544.0)
    year = int(ev["date"][:4])
    dates = year_grid(year, 73) + year_grid(year + 1, 73)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for d in deltas:
        model = EventModel(halo, m_chi, d, eff, v0_kms=238.0)
        r = model.rate_curve(dates)
        if r.max() <= 0:
            continue
        ax.plot(dates, r / r.max(), lw=1.5, label=f"δ = {d:.0f} keV (max {r.max():.2e} /t/yr at 1e-45 cm²)")
    t_ev = dt.datetime.fromisoformat(ev["date"]).replace(tzinfo=dt.timezone.utc)
    ax.axvline(t_ev, color="r", lw=1.5, label="event " + ev["date"])
    ax.axvspan(dt.datetime.fromisoformat(run["start"]).replace(tzinfo=dt.timezone.utc),
               dt.datetime.fromisoformat(run["end"]).replace(tzinfo=dt.timezone.utc),
               color="0.85", zorder=0, label="WS2024 span (placeholder edges)")
    ax.set_ylabel("window rate / annual maximum")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_title(f"annual modulation of the {win['E_lo_keV']:.0f}–{win['E_hi_keV']:.0f} keV rate, m_χ = {m_chi:.0f} GeV, SHM")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.autofmt_xdate()
    fig.tight_layout()
    p = out / "modulation.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_calendar(cfg: dict, out: pathlib.Path, m_chi: float = 1000.0) -> pathlib.Path:
    from .run import reachability_calendar
    ev = cfg["event"]
    E = float(ev["energy_keV"])
    year = int(ev["date"][:4])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = {"544": "C0", "528": "C1", "500": "C2", "580": "C3"}
    for vesc, v0 in ((544.0, 238.0), (528.0, 233.0), (500.0, 238.0), (580.0, 238.0)):
        deltas = np.arange(300.0, 470.0, 1.0)
        firsts, lasts, ds = [], [], []
        for d in deltas:
            c = reachability_calendar(E, m_chi, float(d), vesc, v0, year, n=366)
            if c["first"] is not None:
                ds.append(d)
                firsts.append(dt.datetime.fromisoformat(c["first"]))
                lasts.append(dt.datetime.fromisoformat(c["last"]))
        if ds:
            ax.fill_betweenx(ds, firsts, lasts, alpha=0.25, color=colors[str(int(vesc))],
                             label=f"v_esc = {vesc:.0f}, v_0 = {v0:.0f} km/s")
    t_ev = dt.datetime.fromisoformat(ev["date"])
    ax.axvline(t_ev, color="r", lw=1.5, label="event " + ev["date"])
    ax.set_ylabel("mass splitting δ [keV]")
    ax.set_xlabel(f"date in {year} on which a {E:.0f} keV recoil is kinematically reachable")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_title(f"reachability calendar at the kinematic edge, m_χ = {m_chi:.0f} GeV")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    p = out / "calendar.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_scan_map(results: pathlib.Path, out: pathlib.Path, m_chi: float = 1000.0) -> pathlib.Path | None:
    f = results / "scan.json"
    if not f.exists():
        return None
    rows = [r for r in json.loads(f.read_text()) if r["m_chi_gev"] == m_chi]
    if not rows:
        return None
    halos = sorted({r["halo"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    for h in halos:
        rr = sorted([r for r in rows if r["halo"] == h], key=lambda r: r["delta_keV"])
        d = [r["delta_keV"] for r in rr]
        axes[0].plot(d, [r["bayes_factor_timing"] for r in rr], lw=1.5, label=h)
        axes[1].plot(d, [r["energy_density_at_event_per_keV"] for r in rr], lw=1.5, label=h)
    axes[0].set_ylabel("timing Bayes factor R(t_obs)/<R>")
    axes[1].set_ylabel("p(E_obs = 248 keV | t_obs) [1/keV]")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("mass splitting δ [keV]")
        ax.legend(fontsize=8)
    fig.suptitle(f"m_χ = {m_chi:.0f} GeV")
    fig.tight_layout()
    p = out / "scan_map.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_vesc_posterior(results: pathlib.Path, out: pathlib.Path, m_chi: float = 1000.0) -> pathlib.Path | None:
    f = results / "vesc_marginal.json"
    if not f.exists():
        return None
    s = json.loads(f.read_text())
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name, res in s["measurements"].items():
        pm = res["per_mass"].get(str(m_chi)) or res["per_mass"].get(str(int(m_chi)))
        if not pm:
            continue
        d = [x[0] for x in pm["posterior"]]
        p = [x[1] for x in pm["posterior"]]
        ms = res["measurement"]
        ax.plot(d, p, lw=1.6, label=f"{name}: v_esc = {ms['v_esc']:.0f} +{ms['plus']:.0f} -{ms['minus']:.0f}")
    ax.set_xlabel("mass splitting δ [keV]")
    ax.set_ylabel("P(δ | event, v_esc measurement)")
    ax.set_title(f"δ posterior with the escape speed integrated over each measurement, m_χ = {m_chi:.0f} GeV, {s['tail']['tail']} tail")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "vesc_posterior.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_evidence(results: pathlib.Path, out: pathlib.Path) -> pathlib.Path | None:
    f = results / "posterior_summary.json"
    g = results / "vesc_marginal.json"
    if not f.exists():
        return None
    s = json.loads(f.read_text())
    names, vals = [], []
    for name, v in s["per_halo"].items():
        if v.get("evidence_rel_best") is not None:
            names.append(name)
            vals.append(v["evidence_rel_best"])
    if g.exists():
        sv = json.loads(g.read_text())
        for name, res in sv["measurements"].items():
            vp = res["per_mass"].get("_vesc_posterior") or {}
            if vp.get("evidence_rel_best") is not None:
                names.append("v_esc: " + name)
                vals.append(vp["evidence_rel_best"])
    if not names:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 0.45 * len(names) + 1.6))
    y = np.arange(len(names))
    ax.barh(y, vals, color=["C0" if not n.startswith("v_esc") else "C1" for n in names])
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("evidence relative to the best model in its group (energy x date x sideband, σ marginalised)")
    ax.invert_yaxis()
    fig.tight_layout()
    p = out / "evidence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_vesc_inverse(results: pathlib.Path, out: pathlib.Path) -> pathlib.Path | None:
    g = results / "vesc_marginal.json"
    if not g.exists():
        return None
    sv = json.loads(g.read_text())
    inv = sv.get("inverse")
    if not inv:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(inv["vesc_grid"], inv["likelihood_norm"], "k-", lw=2, label="the event's likelihood of v_esc (flat prior)")
    for name, res in sv["measurements"].items():
        vp = res["per_mass"].get("_vesc_posterior") or {}
        if vp.get("posterior"):
            x = [a for a, _ in vp["posterior"]]
            yv = [b for _, b in vp["posterior"]]
            ax.plot(x, yv, lw=1.2, label=f"posterior with {name}")
    ax.set_xlabel("local escape speed v_esc [km/s]")
    ax.set_ylabel("normalised likelihood / posterior")
    ax.set_title(f"the escape speed from the event ({sv['tail']['tail']} tail; m_χ, δ, σ marginalised)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = out / "vesc_inverse.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def make_all(cfg: dict, results: pathlib.Path, out: pathlib.Path | None = None) -> list[pathlib.Path]:
    out = out or (results / "figures")
    out.mkdir(parents=True, exist_ok=True)
    made = [fig_tails(cfg, out), fig_modulation(cfg, out), fig_calendar(cfg, out)]
    for extra in (fig_scan_map(results, out), fig_vesc_posterior(results, out),
                  fig_evidence(results, out), fig_vesc_inverse(results, out)):
        if extra is not None:
            made.append(extra)
    return made
