"""Runner: back-track the known interstellar objects toward LHS 1140.

For each known ISO (1I/'Oumuamua, 2I/Borisov, 3I/ATLAS) this:

1. builds the ISO's heliocentric Galactic velocity from its asymptotic incoming
   speed + radiant (the direction it came from);
2. resolves LHS 1140's live 6D phase space from Gaia DR3 (degrading to the
   committed literature vector) and transforms it to the Galactocentric frame,
   integrating its orbit back a few Myr for context;
3. integrates the ISO's trajectory **backward through the shared Galactic
   potential** (the ``galactic`` integrator -- not reimplemented) and Monte-Carlos
   its velocity + radiant uncertainty to get the *distribution* of closest-approach
   distance and time to LHS 1140;
4. optionally scans a modest sample of the nearest RV-complete Gaia stars to
   report which star (if any) each ISO best back-tracks toward, purely as context
   for how crowded the track is.

Writes ``results/iso/backtrack.json`` (full per-ISO MC distributions + nearest-
star scan) and ``results/iso/summary.json``.  BOTH carry, as a first-class field,
the necessary-not-sufficient caveat: a close pass does NOT mean an ISO came from
LHS 1140, and the Galactic-disk prior overwhelmingly favours a generic origin.

The dynamics are unit-tested offline (``tests/test_iso.py``); the Gaia acquisition
runs only on the GitHub runner and degrades gracefully when the network is blocked.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..galactic.orbits import (
    heliocentric_to_galactocentric,
    integrate_orbits,
)
from ..panspermia.kinematics import phase_space_6d
from .backtrack import (
    closest_approach_integrated,
    iso_galactocentric,
    mc_backtrack,
    radiant_to_velocity,
)

# --- LHS 1140 (the technosignature anchor) --------------------------------------
# Gaia DR3 astrometry + systemic RV (values from the task brief / Gaia DR3 +
# Lillo-Box et al. 2020).  Used as the fallback if the live Gaia fetch fails.
LHS1140_SOURCE_ID = 2371032916186181760
LHS1140 = {
    "name": "LHS 1140", "source_id": LHS1140_SOURCE_ID,
    "ra": 11.2487, "dec": -15.2742, "parallax": 66.83,
    "pmra": 317.6, "pmdec": -596.6, "radial_velocity": -13.7,
}

# --- Known interstellar objects -------------------------------------------------
# Asymptotic incoming speed v_inf (km/s) and radiant (RA, Dec deg; the direction
# the object CAME FROM).  sigma_v / sigma_radiant_deg encode the uncertainty we
# propagate -- generous, because back-tracking is uncertainty-limited over Myr.
#
# Citations:
#   1I/'Oumuamua  -- v_inf ~26.3 km/s; radiant near the solar apex in Lyra,
#                    RA 279.8 Dec +33.996 (Mamajek 2017, RNAAS; JPL SBDB C/2017 U1).
#   2I/Borisov    -- v_inf ~32 km/s; radiant RA 32.3 Dec +59.4
#                    (JPL SBDB C/2019 Q4; Higuchi & Kokubo 2019).
#   3I/ATLAS      -- v_inf ~58 km/s, retrograde; radiant approx in Sagittarius,
#                    RA ~295 Dec ~-16.  ELEMENTS APPROXIMATE (2025 discovery;
#                    early-arc solution) -- included for context, not precision.
ISOS = [
    {"name": "1I/'Oumuamua", "v_inf_kms": 26.3,
     "ra_radiant": 279.8, "dec_radiant": 33.996,
     "sigma_v": 0.5, "sigma_radiant_deg": 1.0, "approx": False},
    {"name": "2I/Borisov", "v_inf_kms": 32.0,
     "ra_radiant": 32.3, "dec_radiant": 59.4,
     "sigma_v": 0.5, "sigma_radiant_deg": 1.0, "approx": False},
    {"name": "3I/ATLAS", "v_inf_kms": 58.0,
     "ra_radiant": 295.0, "dec_radiant": -16.0,
     "sigma_v": 5.0, "sigma_radiant_deg": 5.0, "approx": True},
]

_CAVEAT = (
    "A close back-track pass is NECESSARY BUT NOT SUFFICIENT for a common origin. "
    "ISO radiants are known to at best ~degree precision (parsecs of transverse "
    "smear at the distance of the nearest stars, growing with lookback time as "
    "phase mixing erases the track); fast-ISO radiants project near the solar "
    "apex, so a large fraction of nearby disk stars lie along the back-track by "
    "projection alone; and the Galactic-disk prior overwhelmingly favours a "
    "generic field-star origin over any one named system. NONE of these results "
    "may be read as a claim that an ISO originated at LHS 1140 (or any listed "
    "star). consistent_with_origin=True means 'not dynamically excluded', not "
    "'came from'."
)

# Nearest-star context scan (RV-complete Gaia within ~25 pc).
_NEARBY_QUERY = """
SELECT TOP {limit}
       source_id, ra, dec, parallax, pmra, pmdec, radial_velocity,
       phot_g_mean_mag, bp_rp, ruwe
FROM gaiadr3.gaia_source
WHERE parallax > {plx_min}
  AND parallax_over_error > 8
  AND radial_velocity IS NOT NULL
  AND ruwe < 1.4
"""


def _resolve_lhs1140() -> dict:
    """LHS 1140's live 6D vector from Gaia DR3, falling back to the committed
    literature values if the fetch fails, returns the wrong source, or lacks an RV
    (the back-track maths is undefined without a 3D velocity)."""
    try:
        from ..panspermia.run import _ANCHOR_QUERY, _run_query
        row = _run_query(_ANCHOR_QUERY.format(source_id=LHS1140_SOURCE_ID))
        if len(row):
            r = row.iloc[0]
            rv = pd.to_numeric(r.get("radial_velocity"), errors="coerce")
            if int(r["source_id"]) == LHS1140_SOURCE_ID and np.isfinite(rv):
                print(f"[iso] resolved LHS 1140 from Gaia DR3: plx={r['parallax']:.3f} "
                      f"pmra={r['pmra']:.2f} pmdec={r['pmdec']:.2f} rv={rv:.3f}")
                out = dict(LHS1140)
                out.update({k: float(r[k]) for k in
                            ("ra", "dec", "parallax", "pmra", "pmdec")})
                out["radial_velocity"] = float(rv)
                return out
        print("[iso] Gaia row unusable for LHS 1140; using literature fallback")
    except Exception as exc:  # noqa: BLE001
        print(f"[iso] LHS 1140 resolve failed ({exc!r}); using literature fallback")
    return dict(LHS1140)


def _anchor_galactocentric(anchor: dict) -> dict:
    """Heliocentric phase space + Galactocentric (pos, vel) for a resolved star."""
    ps = phase_space_6d(pd.DataFrame([anchor])).iloc[0]
    pos, vel = heliocentric_to_galactocentric(
        ps["X_pc"], ps["Y_pc"], ps["Z_pc"],
        ps["U_kms"], ps["V_kms"], ps["W_kms"])
    a = dict(anchor)
    a.update({k: float(ps[k]) for k in ("X_pc", "Y_pc", "Z_pc",
                                        "U_kms", "V_kms", "W_kms", "dist_pc")})
    a["pos_gc"] = pos[0]
    a["vel_gc"] = vel[0]
    return a


def _fetch_nearby(max_pc: float, limit: int) -> pd.DataFrame:
    """RV-complete Gaia sample within ``max_pc`` (context scan; runner-only)."""
    from ..panspermia.run import _run_query
    plx_min = 1000.0 / max_pc
    q = _NEARBY_QUERY.format(limit=int(limit), plx_min=plx_min)
    df = _run_query(q)
    print(f"[iso] nearby-star scan: {len(df)} RV-complete Gaia stars within "
          f"{max_pc:.0f} pc")
    return df.reset_index(drop=True)


def _scan_nearest_stars(iso: dict, stars_gc_pos, stars_gc_vel, meta: pd.DataFrame,
                        t_max_myr: float, dt_myr: float, top: int = 10) -> list:
    """Which nearby stars does this ISO's back-track pass closest to? (context)."""
    v_gal = radiant_to_velocity(iso["v_inf_kms"], iso["ra_radiant"],
                                iso["dec_radiant"])
    iso_pos, iso_vel = iso_galactocentric(v_gal)
    # ISO is the anchor (row 0); every star is a "star" row -> one integration.
    res = closest_approach_integrated(iso_pos, iso_vel, stars_gc_pos, stars_gc_vel,
                                      t_max_myr=t_max_myr, dt_myr=dt_myr,
                                      direction=-1)
    d_min = res["d_min_pc"]
    t_enc = res["t_enc_myr"]
    order = np.argsort(d_min)[:top]
    rows = []
    for i in order:
        m = meta.iloc[int(i)]
        rows.append({
            "source_id": int(m["source_id"]),
            "ra": round(float(m["ra"]), 4), "dec": round(float(m["dec"]), 4),
            "dist_pc": round(1000.0 / float(m["parallax"]), 3),
            "phot_g_mean_mag": (round(float(m["phot_g_mean_mag"]), 2)
                                if "phot_g_mean_mag" in m else None),
            "d_min_pc": round(float(d_min[i]), 3),
            "t_enc_myr": round(float(t_enc[i]), 2),
            "is_lhs1140": int(m["source_id"]) == LHS1140_SOURCE_ID,
        })
    return rows


def iso_run(cfg: Config | None = None, t_max_myr: float = 200.0,
            dt_myr: float = 0.5, n_mc: int = 2000, nearby_pc: float = 25.0,
            nearby_limit: int = 20000, d_close_pc: float = 1.0,
            scan_nearby: bool = True) -> dict:
    """Back-track every known ISO toward LHS 1140 (+ nearest-star context scan)."""
    cfg = cfg or load_config()
    out_dir = cfg.root / "results" / "iso"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- LHS 1140 anchor: 6D + Galactocentric orbit back a few Myr ------------
    anchor = _anchor_galactocentric(_resolve_lhs1140())
    times, traj = integrate_orbits(anchor["pos_gc"][None, :],
                                   anchor["vel_gc"][None, :],
                                   t_max_myr=min(t_max_myr, 20.0), dt_myr=0.5,
                                   direction=-1, record_every=10)
    anchor_orbit = [{"t_myr": round(float(t), 2),
                     "x_kpc": round(float(traj[k, 0, 0]), 5),
                     "y_kpc": round(float(traj[k, 0, 1]), 5),
                     "z_kpc": round(float(traj[k, 0, 2]), 5)}
                    for k, t in enumerate(times)]
    print(f"[iso] LHS 1140 at helio dist {anchor['dist_pc']:.2f} pc; "
          f"integrated orbit back {min(t_max_myr, 20.0):.0f} Myr")

    # --- nearby-star context sample (optional; runner-only) -------------------
    stars_gc_pos = stars_gc_vel = None
    star_meta = pd.DataFrame()
    if scan_nearby:
        try:
            raw = _fetch_nearby(nearby_pc, nearby_limit)
            sp = phase_space_6d(raw)
            sp = sp[np.isfinite(sp["U_kms"].to_numpy())].reset_index(drop=True)
            if len(sp):
                stars_gc_pos, stars_gc_vel = heliocentric_to_galactocentric(
                    sp["X_pc"].to_numpy(float), sp["Y_pc"].to_numpy(float),
                    sp["Z_pc"].to_numpy(float), sp["U_kms"].to_numpy(float),
                    sp["V_kms"].to_numpy(float), sp["W_kms"].to_numpy(float))
                star_meta = sp
        except Exception as exc:  # noqa: BLE001
            print(f"[iso] nearby-star scan skipped ({exc!r})")

    # --- per-ISO back-track ---------------------------------------------------
    iso_results = []
    for iso in ISOS:
        print(f"[iso] === {iso['name']} (v_inf={iso['v_inf_kms']} km/s, "
              f"radiant RA={iso['ra_radiant']} Dec={iso['dec_radiant']}) ===")
        mc = mc_backtrack(iso, {"name": "LHS 1140", "pos_gc": anchor["pos_gc"],
                                "vel_gc": anchor["vel_gc"]},
                          sigma_v=iso["sigma_v"],
                          sigma_radiant_deg=iso["sigma_radiant_deg"],
                          n=n_mc, t_max_myr=t_max_myr, dt_myr=dt_myr,
                          d_close_pc=d_close_pc)
        # Point-estimate geometry (nominal radiant/speed) for reference.
        v_gal = radiant_to_velocity(iso["v_inf_kms"], iso["ra_radiant"],
                                    iso["dec_radiant"])
        rec = {
            "name": iso["name"],
            "elements_approximate": bool(iso.get("approx")),
            "v_inf_kms": iso["v_inf_kms"],
            "radiant": {"ra_deg": iso["ra_radiant"], "dec_deg": iso["dec_radiant"]},
            "helio_velocity_UVW_kms": [round(float(x), 3) for x in v_gal],
            "lhs1140_backtrack": mc,
            "necessary_not_sufficient": _CAVEAT,
        }
        if star_meta is not None and len(star_meta):
            rec["nearest_star_scan"] = _scan_nearest_stars(
                iso, stars_gc_pos, stars_gc_vel, star_meta, t_max_myr, dt_myr)
        d50 = mc["d_min_pc"]["p50"]
        print(f"[iso] {iso['name']}: LHS 1140 closest approach median "
              f"{d50:.2f} pc (2.5-97.5%: {mc['d_min_pc']['p2_5']:.2f}"
              f"-{mc['d_min_pc']['p97_5']:.2f} pc); "
              f"consistent_with_origin={mc['consistent_with_origin']} "
              f"(necessary-not-sufficient)")
        iso_results.append(rec)

    backtrack = {
        "target": "LHS 1140", "target_source_id": LHS1140_SOURCE_ID,
        "target_dist_pc": round(anchor["dist_pc"], 3),
        "t_max_myr": t_max_myr, "n_mc": n_mc, "d_close_pc": d_close_pc,
        "necessary_not_sufficient": _CAVEAT,
        "anchor_orbit_galactocentric": anchor_orbit,
        "isos": iso_results,
    }
    (out_dir / "backtrack.json").write_text(json.dumps(backtrack, indent=2,
                                                       default=str))

    summary = {
        "target": "LHS 1140",
        "necessary_not_sufficient": _CAVEAT,
        "any_consistent_with_origin": bool(
            any(r["lhs1140_backtrack"]["consistent_with_origin"]
                for r in iso_results)),
        "isos": [{
            "name": r["name"],
            "elements_approximate": r["elements_approximate"],
            "lhs1140_d_min_pc_p50": r["lhs1140_backtrack"]["d_min_pc"]["p50"],
            "lhs1140_d_min_pc_p2_5": r["lhs1140_backtrack"]["d_min_pc"]["p2_5"],
            "lhs1140_d_min_pc_p97_5": r["lhs1140_backtrack"]["d_min_pc"]["p97_5"],
            "lhs1140_t_enc_myr_p50": r["lhs1140_backtrack"]["t_enc_myr"]["p50"],
            "frac_within_d_close": r["lhs1140_backtrack"]["frac_within_d_close"],
            "consistent_with_origin": r["lhs1140_backtrack"]["consistent_with_origin"],
            "nearest_star_d_min_pc": (r["nearest_star_scan"][0]["d_min_pc"]
                                      if r.get("nearest_star_scan") else None),
        } for r in iso_results],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("[iso]", json.dumps(summary, default=str))
    return summary


__all__ = ["iso_run", "ISOS", "LHS1140", "LHS1140_SOURCE_ID"]
