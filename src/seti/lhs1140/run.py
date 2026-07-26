"""Runner-side acquisition + orchestration for the LHS 1140 signature sweep.

Three stages, all writing under ``results/lhs1140/``:

1. **System dossier** -- resolve LHS 1140's live Gaia DR3 row (nearest source in a
   proper-motion-tolerant cone), then run the full per-target battery reused from
   the panspermia dossier: hidden-companion astrometry, WISE IR-colour excess,
   NEOWISE mid-IR variability, ZTF + TESS/K2 light curves (which *carry the transit
   signal of planets b and c*), and the Gaia XP narrow-line (laser) scan.
2. **Neighbour sweep** -- the same waste-heat and hidden-companion screens at
   catalogue scale over every Gaia source in the local volume around LHS 1140,
   cross-matched to the NASA Exoplanet Archive so any flagged neighbour that also
   hosts a planet is surfaced.
3. **Biosignature-observation inventory** -- query MAST for every observation of
   the system and record whether atmosphere-capable spectroscopy (JWST/HST/high-res
   RV) exists, i.e. whether a molecular biosignature search is even possible.

Acquisition needs archive egress and runs on the GitHub runner; the scorers are
unit-tested offline.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..panspermia.run import (
    _dossier_ir_variability,
    _dossier_lightcurve,
    _dossier_tess,
    _dossier_xp,
    _fetch_wise_irsa,
    _run_query,
)
from .dossier import (
    LHS1140,
    PLANETS,
    companion_diagnostics,
    dossier_verdict,
    inventory_summary,
    ir_color_excess,
    neighbor_companion_scan,
    neighbor_ir_excess_scan,
)

# Resolve LHS 1140 by the nearest (highest-parallax) Gaia source in a generous
# cone -- LHS 1140 at ~67 mas dominates any arcmin-scale field, so this is robust
# to its large proper motion between the J2000 catalogue position and Gaia 2016.0.
_RESOLVE_QUERY = """
SELECT source_id, ra, dec, parallax, parallax_over_error, pmra, pmdec, ruwe,
       astrometric_excess_noise, astrometric_excess_noise_sig,
       ipd_frac_multi_peak, non_single_star, phot_g_mean_mag, bp_rp,
       phot_variable_flag, radial_velocity, has_xp_sampled
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(POINT('ICRS', ra, dec),
                 CIRCLE('ICRS', {ra}, {dec}, {radius}))
  AND parallax > 30
ORDER BY parallax DESC
"""

# Neighbour cone: every well-measured Gaia source inside a distance sphere around
# LHS 1140.  parallax > {plx_min} selects the local volume; the 3D cut to LHS 1140
# is applied in Python.  Kept single-table + a chunked WISE join (the cluster
# channel's lesson: joining AllWISE inside the cone makes the TAP server drop the
# result).
_NEIGHBOR_QUERY = """
SELECT TOP {limit}
       source_id, ra, dec, parallax, parallax_over_error, pmra, pmdec,
       phot_g_mean_mag, bp_rp, ruwe, astrometric_excess_noise,
       astrometric_excess_noise_sig, ipd_frac_multi_peak, non_single_star,
       has_xp_sampled
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(POINT('ICRS', ra, dec),
                 CIRCLE('ICRS', {ra}, {dec}, {radius}))
  AND parallax > {plx_min}
  AND parallax_over_error > 8
"""



def _resolve_anchor(cone_arcsec: float = 90.0) -> dict:
    """Resolve LHS 1140's live Gaia DR3 row; fall back to the literature values."""
    radius = cone_arcsec / 3600.0
    try:
        df = _run_query(_RESOLVE_QUERY.format(ra=LHS1140["ra"], dec=LHS1140["dec"],
                                              radius=radius))
        if len(df):
            r = df.iloc[0].to_dict()
            print(f"[lhs1140] resolved Gaia DR3 {int(r['source_id'])} "
                  f"parallax={float(r['parallax']):.2f} mas "
                  f"G={float(r['phot_g_mean_mag']):.2f}")
            return r
    except Exception as exc:  # noqa: BLE001
        print(f"[lhs1140] anchor resolve failed ({exc!r}); using fallback")
    return {"source_id": LHS1140["source_id_fallback"], "ra": LHS1140["ra"],
            "dec": LHS1140["dec"], "parallax": LHS1140["parallax_mas"],
            "pmra": LHS1140["pmra"], "pmdec": LHS1140["pmdec"]}


def _system_dossier(cfg: Config, out_dir) -> dict:
    """Full per-target signature battery on LHS 1140 (star == planets b, c)."""
    row = _resolve_anchor()
    sid = int(row["source_id"])
    ra, dec = float(row["ra"]), float(row["dec"])
    pmra, pmdec = row.get("pmra", 0.0), row.get("pmdec", 0.0)

    companion = companion_diagnostics(row)
    wise = _fetch_wise_irsa(ra, dec, pmra, pmdec)
    ir = ir_color_excess(wise)
    lc = _dossier_lightcurve(ra, dec, pmra, pmdec)          # ZTF g+r
    tess = _dossier_tess(ra, dec)                           # TESS/K2 (transits)
    irvar = _dossier_ir_variability(ra, dec, pmra, pmdec)   # NEOWISE mid-IR
    xp = _dossier_xp(sid)                                   # Gaia XP laser scan

    parts = {"companion": companion, "ir_excess": ir, "ir_variability": irvar,
             "lightcurve_ztf": lc, "lightcurve_tess": tess, "xp": xp}
    verdict = dossier_verdict(parts)

    def _has_data(ch, v):
        if not isinstance(v, dict):
            return False
        if "has_data" in v:
            return bool(v["has_data"])
        if ch == "companion":
            return np.isfinite(v.get("ruwe", np.nan))
        if ch == "ir_excess":
            return any(np.isfinite(v.get(k, np.nan)) for k in ("W1_W2", "W1_W3"))
        if ch == "xp":
            return not any("no XP" in r or "failed" in r for r in v.get("reasons", []))
        return True

    coverage = {ch: ("data" if _has_data(ch, v) else "no_data")
                for ch, v in parts.items()}
    coverage["not_covered"] = ["radio (SETI/GBT/VLA)", "X-ray (Chandra/XMM)"]
    n_obs = sum(1 for ch in parts if coverage.get(ch) == "data")
    verdict["channels_with_data"] = n_obs
    verdict["channels_total"] = len(parts)
    verdict["verdict"] = ("ANOMALY_FLAGGED" if verdict["any_signature_flag"]
                          else f"clean_in_{n_obs}_of_{len(parts)}_observed_channels")

    dossier = {"name": LHS1140["name"], "source_id": sid, "ra": ra, "dec": dec,
               "planets": PLANETS,
               "gaia": {k: row.get(k) for k in
                        ("parallax", "phot_g_mean_mag", "bp_rp",
                         "phot_variable_flag", "radial_velocity", "has_xp_sampled")},
               **parts, "coverage": coverage, "verdict": verdict}
    (out_dir / "system_dossier.json").write_text(json.dumps(dossier, indent=2,
                                                            default=str))
    print(f"[lhs1140] system: {verdict['verdict']} "
          f"flags={[k for k, v in verdict['channel_flags'].items() if v]}")
    return {"anchor": row, "dossier": dossier}


def _fetch_neighbor_wise(local: pd.DataFrame, max_cones: int = 400) -> dict:
    """AllWISE W1-W4 (+errors) per neighbour via proper-motion-propagated IRSA cones.

    The Gaia ``allwise_best_neighbour`` join both misses these high-proper-motion
    nearby stars *and* lacks the ``*sigmpro`` error columns, so -- exactly as for
    the anchor -- we go straight to the IRSA ``allwise_p3as_psd`` catalogue at each
    star's WISE-epoch position.  The local volume is small (tens of stars within a
    few pc), so a per-source cone is affordable; ``max_cones`` guards against an
    over-large sphere silently issuing thousands of queries.
    """
    out = {}
    rows = local.to_dict("records")
    if len(rows) > max_cones:
        print(f"[lhs1140] neighbour WISE: {len(rows)} sources exceeds "
              f"max_cones={max_cones}; fetching the nearest {max_cones}")
        rows = sorted(rows, key=lambda r: r.get("sep_from_lhs1140_pc", 1e9))[:max_cones]
    for r in rows:
        try:
            w = _fetch_wise_irsa(float(r["ra"]), float(r["dec"]),
                                 r.get("pmra", 0.0), r.get("pmdec", 0.0))
        except Exception as exc:  # noqa: BLE001
            print(f"[lhs1140] neighbour WISE cone {r['source_id']} failed: {exc!r}")
            w = {}
        if w:
            out[int(r["source_id"])] = w
    return out


def _neighbor_sweep(anchor: dict, out_dir, radius_deg: float = 8.0,
                    sphere_pc: float = 10.0, limit: int = 200000) -> dict:
    """Catalogue-scale technosignature battery over LHS 1140's stellar neighbours.

    Selects every well-measured Gaia source within ``sphere_pc`` of LHS 1140
    (a cone of ``radius_deg`` provides the angular selection; the 3D distance cut
    is applied in Python), fetches AllWISE photometry, and runs the IR-excess and
    hidden-companion screens over all of them.
    """
    plx0 = float(anchor.get("parallax", LHS1140["parallax_mas"]))
    d0 = 1000.0 / plx0                                    # pc
    # Distance sphere -> a parallax floor (nearer than d0 + sphere).
    plx_min = 1000.0 / (d0 + sphere_pc)
    try:
        stars = _run_query(_NEIGHBOR_QUERY.format(
            ra=LHS1140["ra"], dec=LHS1140["dec"], radius=radius_deg,
            plx_min=plx_min, limit=limit))
    except Exception as exc:  # noqa: BLE001
        print(f"[lhs1140] neighbour cone failed: {exc!r}")
        stars = pd.DataFrame()
    result = {"n_cone": int(len(stars)), "sphere_pc": sphere_pc,
              "anchor_dist_pc": d0}
    if not len(stars):
        (out_dir / "neighbors.json").write_text(json.dumps(result, indent=2))
        return result

    # 3D distance cut to LHS 1140 (angular sep small over a few pc; use the simple
    # radial + tangential separation from parallax distances).
    stars = stars.copy()
    stars["dist_pc"] = 1000.0 / stars["parallax"].clip(lower=1e-3)
    cosd = np.cos(np.radians(LHS1140["dec"]))
    dra = (stars["ra"] - LHS1140["ra"]) * cosd
    ddec = stars["dec"] - LHS1140["dec"]
    ang = np.radians(np.sqrt(dra ** 2 + ddec ** 2))
    # Law of cosines on the two heliocentric distances and the angular separation.
    sep = np.sqrt(stars["dist_pc"] ** 2 + d0 ** 2
                  - 2 * stars["dist_pc"] * d0 * np.cos(ang))
    stars["sep_from_lhs1140_pc"] = sep
    local = stars[sep <= sphere_pc].reset_index(drop=True)
    result["n_within_sphere"] = int(len(local))
    print(f"[lhs1140] neighbours: {len(stars)} in cone, {len(local)} within "
          f"{sphere_pc} pc of LHS 1140")

    rows = local.to_dict("records")
    # Astrometric hidden-companion screen (needs only the Gaia row).
    comp = neighbor_companion_scan(rows)
    # IR-excess screen (needs AllWISE; PM-propagated IRSA cone per neighbour).
    wise = _fetch_neighbor_wise(local) if len(local) else {}
    for r in rows:
        r.update(wise.get(int(r["source_id"]), {}))
    ir = neighbor_ir_excess_scan(rows)

    # Cross-match flagged neighbours to the NASA Exoplanet Archive (context: does a
    # flagged neighbour also host a planet?).
    planet_hosts = _crossmatch_planet_hosts(local)
    result.update({"companion_screen": comp, "ir_excess_screen": ir,
                   "n_planet_hosts_in_volume": len(planet_hosts),
                   "planet_hosts": planet_hosts})
    local.to_csv(out_dir / "neighbors.csv", index=False)
    (out_dir / "neighbors.json").write_text(json.dumps(result, indent=2,
                                                       default=str))
    print(f"[lhs1140] neighbours: {ir['n_ir_excess']} IR-excess, "
          f"{comp['n_companion_flag']} companion-flag, "
          f"{len(planet_hosts)} planet hosts in volume")
    return result


def _crossmatch_planet_hosts(local: pd.DataFrame) -> list[dict]:
    """Which neighbours in the local volume are known exoplanet hosts."""
    try:
        from ..panspermia.exohosts import crossmatch_hosts, fetch_nearby_planets
        planets = fetch_nearby_planets(max_pc=60.0)
        matched = crossmatch_hosts(local, planets, radius_arcsec=5.0)
        hosts = matched[matched["known_planet_host"]] if len(matched) else matched
        cols = [c for c in ("source_id", "ra", "dec", "sep_from_lhs1140_pc",
                            "host_name", "n_planets", "has_temperate_planet",
                            "has_hycean_candidate") if c in hosts.columns]
        return hosts[cols].to_dict("records") if len(hosts) else []
    except Exception as exc:  # noqa: BLE001
        print(f"[lhs1140] planet-host crossmatch skipped: {exc!r}")
        return []


def _biosignature_inventory(out_dir, ra: float | None = None,
                            dec: float | None = None,
                            radius_arcsec: float = 30.0) -> dict:
    """Query MAST for every observation of LHS 1140 and summarise coverage.

    ``query_criteria(objectname=..., radius=...)`` is rejected by MAST ("at least
    one non-positional criterion") -- the positional cone search is
    ``query_region``/``query_object``.  We prefer a coordinate cone (robust to name
    resolution) and fall back to the object name."""
    records = []
    try:
        from astroquery.mast import Observations
        obs = None
        if ra is not None and dec is not None:
            import astropy.units as u
            from astropy.coordinates import SkyCoord
            try:
                obs = Observations.query_region(
                    SkyCoord(ra, dec, unit="deg"),
                    radius=radius_arcsec * u.arcsec)
            except Exception as exc:  # noqa: BLE001
                print(f"[lhs1140] MAST query_region failed: {exc!r}")
        if obs is None or len(obs) == 0:
            obs = Observations.query_object("LHS 1140",
                                            radius=f"{radius_arcsec} arcsec")
        if obs is not None and len(obs):
            records = obs.to_pandas().to_dict("records")
    except Exception as exc:  # noqa: BLE001
        print(f"[lhs1140] MAST inventory query failed: {exc!r}")
    summary = inventory_summary(records)
    summary["planets"] = PLANETS
    (out_dir / "biosignature_inventory.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"[lhs1140] inventory: {summary['n_observations']} obs, "
          f"{summary['n_spectroscopic']} spectroscopic, "
          f"atmosphere_capable={summary['atmosphere_capable_spectroscopy']}")
    return summary


# LHS 1140 b literature parameters (Cadieux+2024), a fallback if the live NASA
# Exoplanet Archive fetch fails.  ``t_in_hours`` is the in-transit duration.
_LHS1140B_FALLBACK = {"rp_earth": 1.730, "mp_earth": 5.60, "rs_sun": 0.2159,
                      "teq_k": 226.0, "jmag": 9.612, "t_in_hours": 2.0,
                      "resolution": 50.0}

# Which biosignature gas band each JWST instrument actually covers -- used to turn
# the observed instrument list into a data-grounded coverage statement.
_INSTRUMENT_GASES = {
    "NIRISS": ["H2O", "O2_CIA"],
    "NIRSPEC": ["CH4", "CO2", "CH3Cl"],
    "NIRSPEC/PRISM": ["CH4", "CO2", "CH3Cl"],
    "MIRI": ["O3", "N2O"],
}


def _fetch_planet_params(name: str = "LHS 1140 b") -> dict:
    """Live LHS 1140 b parameters from the NASA Exoplanet Archive (pscomppars)."""
    params = dict(_LHS1140B_FALLBACK)
    try:
        import io

        import requests
        q = ("select pl_rade,pl_bmasse,pl_eqt,st_rad,sy_jmag,pl_trandur "
             f"from pscomppars where pl_name='{name}'")
        r = requests.get("https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
                         params={"query": q, "format": "csv"}, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if len(df):
            row = df.iloc[0]
            def val(k, default):
                v = row.get(k)
                return float(v) if v is not None and np.isfinite(
                    pd.to_numeric(v, errors="coerce")) else default
            params.update({
                "rp_earth": val("pl_rade", params["rp_earth"]),
                "mp_earth": val("pl_bmasse", params["mp_earth"]),
                "rs_sun": val("st_rad", params["rs_sun"]),
                "teq_k": val("pl_eqt", params["teq_k"]),
                "jmag": val("sy_jmag", params["jmag"]),
                # pl_trandur is the full (T14) transit duration in hours; the
                # in-transit constraining window is ~that.
                "t_in_hours": val("pl_trandur", params["t_in_hours"]),
            })
            print(f"[lhs1140] resolved LHS 1140 b params from NASA archive: "
                  f"Rp={params['rp_earth']:.2f} Re Mp={params['mp_earth']:.2f} Me "
                  f"Teq={params['teq_k']:.0f} K J={params['jmag']:.2f}")
    except Exception as exc:  # noqa: BLE001
        print(f"[lhs1140] planet-param fetch failed ({exc!r}); using literature fallback")
    return params


def _biosignature_answer(out_dir, inventory: dict) -> dict:
    """Compute the biosignature detectability answer for LHS 1140 b.

    Grounds the physics budget in (a) live system parameters and (b) which
    biosignature bands were actually observed (from the inventory instrument
    list), and writes the verdict.
    """
    from .biosignature import biosignature_detectability, biosignature_verdict

    params = _fetch_planet_params()
    budget = biosignature_detectability(params)

    # Data-grounded band coverage: map observed JWST instruments -> gases covered.
    per_inst = inventory.get("per_instrument", {}) if inventory else {}
    covered = {}
    for inst, n in per_inst.items():
        for key, gases in _INSTRUMENT_GASES.items():
            if inst.upper().startswith(key.split("/")[0]):
                for gcol in gases:
                    covered.setdefault(gcol, 0)
                    covered[gcol] += int(n)
    # Estimate of independent transit epochs observed.  Raw MAST rows over-count
    # (per-segment products), so this is a deliberately generous upper bound on the
    # *epochs*, clearly labelled -- the conclusion is robust to its exact value.
    jwst_spectro = sum(int(n) for inst, n in per_inst.items()
                       if inst.upper().startswith(("NIRISS", "NIRSPEC", "MIRI")))
    transits_est = max(2, min(20, jwst_spectro // 8))   # generous epoch proxy

    verdict = biosignature_verdict(budget, atmospheres_observed=1,
                                   transits_observed=transits_est)
    result = {
        "planet": "LHS 1140 b", "params": params,
        "biosignature_bands_observed": covered,
        "jwst_spectroscopic_products": jwst_spectro,
        "transits_observed_estimate": transits_est,
        "budget": budget, "verdict": verdict,
    }
    (out_dir / "biosignature.json").write_text(json.dumps(result, indent=2,
                                                          default=str))
    print(f"[lhs1140] biosignature answer: {verdict['answer']} "
          f"(min {verdict['min_transits_for_any_biosignature']} transits needed "
          f"for any biosignature under {verdict['expected_atmosphere']}; "
          f"~{transits_est} epochs observed)")
    return result


def lhs1140_run(cfg: Config | None = None, sphere_pc: float = 10.0) -> dict:
    """Exhaustive LHS 1140 signature sweep: system dossier + neighbours + bio inventory."""
    cfg = cfg or load_config()
    out_dir = cfg.root / "results" / "lhs1140"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[lhs1140] === system dossier ===")
    sysd = _system_dossier(cfg, out_dir)
    print("[lhs1140] === neighbour sweep ===")
    neigh = _neighbor_sweep(sysd["anchor"], out_dir, sphere_pc=sphere_pc)
    print("[lhs1140] === biosignature-observation inventory ===")
    anchor = sysd["anchor"]
    inv = _biosignature_inventory(out_dir, ra=float(anchor.get("ra", LHS1140["ra"])),
                                  dec=float(anchor.get("dec", LHS1140["dec"])))
    print("[lhs1140] === biosignature detectability answer ===")
    bio = _biosignature_answer(out_dir, inv)

    summary = {
        "target": LHS1140["name"], "planets": [p["name"] for p in PLANETS],
        "system_verdict": sysd["dossier"]["verdict"]["verdict"],
        "system_flags": [k for k, v in
                         sysd["dossier"]["verdict"]["channel_flags"].items() if v],
        "system_coverage": {ch: sysd["dossier"]["coverage"][ch] for ch in
                            ("companion", "ir_excess", "ir_variability",
                             "lightcurve_ztf", "lightcurve_tess", "xp")},
        "neighbours": {k: neigh.get(k) for k in
                       ("n_cone", "n_within_sphere", "n_planet_hosts_in_volume")},
        "neighbour_ir_excess": neigh.get("ir_excess_screen", {}).get("n_ir_excess"),
        "neighbour_companion_flags": neigh.get("companion_screen", {}).get(
            "n_companion_flag"),
        "biosignature_inventory": {
            "n_observations": inv.get("n_observations"),
            "n_spectroscopic": inv.get("n_spectroscopic"),
            "atmosphere_capable_spectroscopy": inv.get(
                "atmosphere_capable_spectroscopy"),
            "per_instrument": inv.get("per_instrument"),
        },
        "biosignature_answer": {
            "answer": bio["verdict"]["answer"],
            "expected_atmosphere": bio["verdict"]["expected_atmosphere"],
            "min_transits_for_any_biosignature":
                bio["verdict"]["min_transits_for_any_biosignature"],
            "transits_observed_estimate": bio["transits_observed_estimate"],
            "bands_observed": list(bio["biosignature_bands_observed"].keys()),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("[lhs1140]", json.dumps(summary, default=str))
    return summary


__all__ = ["lhs1140_run"]
