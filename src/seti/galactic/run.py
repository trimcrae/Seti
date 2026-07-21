"""Runner: long-baseline (few hundred Myr) encounter search for both anchors.

For each biosignature-anchor system (LHS 1140, K2-18) this:

1. resolves the anchor's 6D phase space from Gaia DR3;
2. pulls the RV-complete Gaia sample in a present-day sphere around it;
3. integrates every star's orbit (and the anchor's) back ``t_max`` (default
   300 Myr) in the Galactic potential and records each star's closest approach;
4. shortlists close encounters and Monte-Carlos the top ones to test whether the
   encounter *timing* survives phase mixing (the honest recoverability horizon);
5. cross-matches the shortlist to the NASA Exoplanet Archive and runs the
   signature battery -- the astrometric hidden-companion (techno) screen on every
   encounter star, and the biosignature-detectability (bio) answer on any that
   host a planet.

Writes ``results/galactic/<anchor>.json`` + a combined ``summary.json``.  The
dynamics are unit-tested offline; acquisition + integration run on the runner.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..lhs1140.biosignature import biosignature_detectability, biosignature_verdict
from ..lhs1140.dossier import companion_diagnostics
from ..panspermia.kinematics import phase_space_6d
from ..panspermia.run import _anchor_phase_space, _fetch_shell, _resolve_anchor
from .encounters import closest_approach_from_helio, mc_encounter_orbit

# The two nearby biosignature-anchor systems, with the planet parameters and the
# physically expected atmosphere used for the biosignature-detectability answer.
ANCHORS = [
    {"name": "LHS 1140", "source_id": 2371032916186181760,
     "planet": "LHS 1140 b", "expected_atmosphere": "N2_secondary",
     "bio_params": {"rp_earth": 1.730, "mp_earth": 5.60, "rs_sun": 0.2159,
                    "teq_k": 226.0, "jmag": 9.612, "t_in_hours": 2.0}},
    {"name": "K2-18", "source_id": 3892950081412683520,
     "planet": "K2-18 b", "expected_atmosphere": "H2_rich_cleared",
     "bio_params": {"rp_earth": 2.610, "mp_earth": 8.63, "rs_sun": 0.4445,
                    "teq_k": 272.0, "jmag": 9.763, "t_in_hours": 2.7}},
]


def _shortlist_encounters(anchor_ps: dict, stars: pd.DataFrame,
                          t_max_myr: float, d_cut_pc: float,
                          dt_myr: float = 0.5) -> pd.DataFrame:
    """Orbit-integrated closest approaches; keep past encounters within ``d_cut``."""
    ca = closest_approach_from_helio(anchor_ps, stars, t_max_myr=t_max_myr,
                                     dt_myr=dt_myr)
    sid = ca.get("source_id", pd.Series(-1, index=ca.index)).to_numpy()
    keep = ((ca["d_min_pc"].to_numpy() <= d_cut_pc)
            & (ca["t_enc_myr"].to_numpy() < 0)          # past encounters
            & (sid != anchor_ps.get("source_id", -1)))
    sl = ca[keep].sort_values("d_min_pc").reset_index(drop=True)
    return sl


def _battery_on_hosts(shortlist: pd.DataFrame, planets: pd.DataFrame) -> dict:
    """Techno (companion) screen on all; bio detectability on planet-hosts."""
    from ..panspermia.exohosts import crossmatch_hosts

    # Techno: astrometric hidden-companion screen from the Gaia row (free).
    n_companion = 0
    companion_flags = []
    for r in shortlist.to_dict("records"):
        diag = companion_diagnostics(r)
        if diag.get("companion_flag"):
            n_companion += 1
            companion_flags.append({"source_id": r.get("source_id"),
                                    "d_min_pc": round(float(r.get("d_min_pc", np.nan)), 3),
                                    "reasons": diag["reasons"]})
    # Bio: which encounter stars are known planet hosts.
    hosts = []
    if len(planets):
        matched = crossmatch_hosts(shortlist, planets, radius_arcsec=5.0)
        for r in matched[matched["known_planet_host"]].to_dict("records"):
            hosts.append({"source_id": r.get("source_id"),
                          "host_name": r.get("host_name"),
                          "d_min_pc": round(float(r.get("d_min_pc", np.nan)), 3),
                          "t_enc_myr": round(float(r.get("t_enc_myr", np.nan)), 2),
                          "n_planets": int(r.get("n_planets", 0)),
                          "has_temperate_planet": bool(r.get("has_temperate_planet")),
                          "has_hycean_candidate": bool(r.get("has_hycean_candidate"))})
    return {"n_encounter_stars": int(len(shortlist)),
            "n_companion_flag": n_companion, "companion_flags": companion_flags,
            "n_planet_host_encounters": len(hosts), "planet_host_encounters": hosts}


def _anchor_biosignature(anchor: dict) -> dict:
    """Biosignature-detectability answer for the anchor's own planet."""
    budget = biosignature_detectability(anchor["bio_params"])
    verdict = biosignature_verdict(budget, atmospheres_observed=1,
                                   transits_observed=4,
                                   expected_atmosphere=anchor["expected_atmosphere"])
    return {"planet": anchor["planet"],
            "expected_atmosphere": anchor["expected_atmosphere"],
            "answer": verdict["answer"],
            "min_transits_for_any_biosignature":
                verdict["min_transits_for_any_biosignature"],
            "scale_height_km": verdict["scale_height_km"]}


def galactic_run(cfg: Config | None = None, t_max_myr: float = 300.0,
                 search_pc: float = 250.0, d_cut_pc: float = 3.0,
                 g_max: float = 16.0, limit: int = 150000,
                 mc_top: int = 15) -> dict:
    """Long-baseline encounter search + signature battery for both anchors."""
    cfg = cfg or load_config()
    out_dir = cfg.root / "results" / "galactic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Planet catalogue once (covers the whole encounter volume).
    try:
        from ..panspermia.exohosts import fetch_nearby_planets
        planets = fetch_nearby_planets(max_pc=max(300.0, search_pc + 50.0))
    except Exception as exc:  # noqa: BLE001
        print(f"[galactic] planet catalogue fetch failed: {exc!r}")
        planets = pd.DataFrame()

    anchor_results = []
    for anc in ANCHORS:
        name = anc["name"]
        print(f"[galactic] === {name} (Gaia DR3 {anc['source_id']}) ===")
        anchor = _anchor_phase_space(_resolve_anchor(anc["source_id"]))
        try:
            raw = _fetch_shell(anchor, search_pc, g_max, limit)
        except Exception as exc:  # noqa: BLE001
            print(f"[galactic] shell fetch failed for {name}: {exc!r}")
            raw = pd.DataFrame()
        result = {"anchor": name, "source_id": anc["source_id"],
                  "t_max_myr": t_max_myr, "search_pc": search_pc,
                  "d_cut_pc": d_cut_pc,
                  "anchor_biosignature": _anchor_biosignature(anc)}
        if not len(raw):
            result["n_sample"] = 0
            anchor_results.append(result)
            (out_dir / f"{name.replace(' ', '_')}.json").write_text(
                json.dumps(result, indent=2, default=str))
            continue

        stars = phase_space_6d(raw)
        stars = stars[np.isfinite(stars["U_kms"].to_numpy())].reset_index(drop=True)
        result["n_sample"] = int(len(stars))
        print(f"[galactic] {name}: integrating {len(stars)} orbits back "
              f"{t_max_myr:.0f} Myr")
        shortlist = _shortlist_encounters(anchor, stars, t_max_myr, d_cut_pc)
        result["n_encounters_within_dcut"] = int(len(shortlist))

        # Monte-Carlo the closest encounters for timing recoverability.
        mc = []
        for r in shortlist.head(mc_top).to_dict("records"):
            star = {k: r.get(k) for k in
                    ("ra", "dec", "parallax", "parallax_error", "pmra",
                     "pmra_error", "pmdec", "pmdec_error", "radial_velocity",
                     "radial_velocity_error")}
            m = mc_encounter_orbit(anchor, star, t_max_myr=t_max_myr,
                                   dt_myr=1.0, n=120)
            m["source_id"] = r.get("source_id")
            m["d_min_pc_point"] = round(float(r.get("d_min_pc", np.nan)), 3)
            m["t_enc_myr_point"] = round(float(r.get("t_enc_myr", np.nan)), 2)
            mc.append(m)
        n_recoverable = sum(1 for m in mc if m.get("timing_recoverable"))
        result["mc"] = mc
        result["n_timing_recoverable"] = n_recoverable
        result["recoverability_note"] = (
            f"{n_recoverable}/{len(mc)} of the closest encounters have a "
            f"Monte-Carlo t_enc spread < 25% of the {t_max_myr:.0f} Myr window; "
            "the rest are real close passes whose *timing* is erased by phase "
            "mixing (d_min may still be robust). This is the honest recoverability "
            "horizon of a hundreds-of-Myr encounter search.")

        # Signature battery on the shortlist.
        result["battery"] = _battery_on_hosts(shortlist, planets)
        shortlist.to_csv(out_dir / f"{name.replace(' ', '_')}_encounters.csv",
                         index=False)
        (out_dir / f"{name.replace(' ', '_')}.json").write_text(
            json.dumps(result, indent=2, default=str))
        print(f"[galactic] {name}: {len(shortlist)} encounters < {d_cut_pc} pc, "
              f"{n_recoverable}/{len(mc)} timing-recoverable, "
              f"{result['battery']['n_planet_host_encounters']} planet-host "
              f"encounters, {result['battery']['n_companion_flag']} companion flags")
        anchor_results.append(result)

    summary = {
        "anchors": [{
            "name": r["anchor"],
            "anchor_biosignature": r["anchor_biosignature"],
            "n_sample": r.get("n_sample", 0),
            "n_encounters_within_dcut": r.get("n_encounters_within_dcut", 0),
            "n_timing_recoverable": r.get("n_timing_recoverable", 0),
            "n_planet_host_encounters": r.get("battery", {}).get(
                "n_planet_host_encounters", 0),
            "n_companion_flag": r.get("battery", {}).get("n_companion_flag", 0),
        } for r in anchor_results],
        "t_max_myr": t_max_myr, "search_pc": search_pc, "d_cut_pc": d_cut_pc,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("[galactic]", json.dumps(summary, default=str))
    return summary


__all__ = ["galactic_run", "ANCHORS"]
