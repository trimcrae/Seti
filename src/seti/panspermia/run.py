"""End-to-end close-encounter search anchored on the hycean biosignature host.

Hypothesis under test.  K2-18 b is the hycean world with a JWST hint of a
biosignature (DMS/DMSO; Madhusudhan et al. 2023, 2025 -- contested, treated here
as the premise, not a result).  *If* something living arose there, the material
that could carry it to another star -- impact ejecta, dormant spores, free-flying
bodies of the 'Oumuamua/Borisov class -- is delivered most efficiently during a
**close, slow stellar encounter**, when two systems' outer reservoirs overlap and
the relative speed is low enough for capture.  Because the stellar neighbourhood
*reshuffles over time* (today's neighbours are not those of a few Myr ago), the
right question is not "who is nearby now" but "who passed close to K2-18, slowly,
in the recent past".

Pipeline (acquisition on the GitHub runner; maths unit-tested offline):

1. resolve K2-18's full 6D phase space from Gaia DR3 (RV essential);
2. pull every Gaia DR3 source with a radial velocity inside a heliocentric
   distance shell that brackets the search volume around K2-18;
3. build 6D phase space, compute each star's linear closest approach to K2-18,
   and score past close/slow encounters for transfer plausibility;
4. write the ranked recipient candidates + co-moving companions.

A surviving candidate is a *specific nearby star* that passed close to K2-18 at
low relative velocity within the viability window -- the star most likely, on
kinematic grounds alone, to have received K2-18-origin material.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config, load_config
from .encounters import closest_approach, flag_comoving, transfer_score
from .kinematics import phase_space_6d

# K2-18 (EPIC 201912552 = 2MASS J11301441+0735371). The runner resolves the live
# Gaia DR3 6D vector by source_id; these are an approximate literature fallback
# only, used if acquisition fails, and are clearly superseded by the fetched row.
K2_18_SOURCE_ID = 3892950081412683520
K2_18_FALLBACK = {
    "source_id": K2_18_SOURCE_ID, "ra": 172.560055, "dec": 7.588391,
    "parallax": 26.30, "pmra": -84.68, "pmdec": -60.93,
    "radial_velocity": 0.35,   # km/s, systemic (literature); superseded by Gaia
}

_ANCHOR_QUERY = """
SELECT source_id, ra, dec, parallax, parallax_over_error, pmra, pmdec,
       radial_velocity, phot_g_mean_mag, bp_rp, ruwe
FROM gaiadr3.gaia_source
WHERE source_id = {source_id}
"""

# All-sky 6D shell: every star with a radial velocity whose heliocentric distance
# brackets the anchor's, so the search volume around the anchor is fully covered.
# The 3D cut to the anchor is applied in Python after the fetch.
_SHELL_QUERY = """
SELECT TOP {limit}
       source_id, ra, dec, parallax, parallax_over_error, pmra, pmdec,
       radial_velocity, phot_g_mean_mag, bp_rp, ruwe
FROM gaiadr3.gaia_source
WHERE parallax > {plx_min} AND parallax < {plx_max}
  AND parallax_over_error > 8
  AND radial_velocity IS NOT NULL
  AND ruwe < 1.4
"""


def _run_query(query: str, retries: int = 4) -> pd.DataFrame:
    """Robust Gaia ADQL (async with exp. backoff, synchronous fallback on the
    last try) -- the same pattern the clustering channel uses to survive the Gaia
    TAP server's intermittent async-result drops."""
    import time

    from astroquery.gaia import Gaia
    last = None
    for attempt in range(retries):
        try:
            if attempt == retries - 1:
                job = Gaia.launch_job(query)
            else:
                job = Gaia.launch_job_async(query)
            df = job.get_results().to_pandas()
            return df.rename(columns={c: c.lower() for c in df.columns})
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[panspermia] query attempt {attempt + 1}/{retries} failed: {exc!r}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gaia query failed after {retries} attempts: {last!r}")


def _resolve_anchor(source_id: int) -> dict:
    """Fetch K2-18's live 6D vector from Gaia DR3; fall back to the committed
    literature value only if the fetch fails or returns no radial velocity (the
    encounter maths is undefined without a 3D velocity)."""
    try:
        row = _run_query(_ANCHOR_QUERY.format(source_id=source_id))
        if len(row) and np.isfinite(pd.to_numeric(row["radial_velocity"],
                                                   errors="coerce").iloc[0]):
            r = row.iloc[0]
            print(f"[panspermia] resolved anchor from Gaia DR3: "
                  f"plx={r['parallax']:.3f} rv={r['radial_velocity']:.3f}")
            return {k: r[k] for k in ("source_id", "ra", "dec", "parallax",
                                      "pmra", "pmdec", "radial_velocity")}
        print("[panspermia] Gaia row lacks a radial velocity; using fallback")
    except Exception as exc:  # noqa: BLE001
        print(f"[panspermia] anchor resolve failed ({exc!r}); using fallback")
    return dict(K2_18_FALLBACK)


def _anchor_phase_space(anchor: dict) -> dict:
    ps = phase_space_6d(pd.DataFrame([anchor])).iloc[0]
    a = dict(anchor)
    a.update({k: float(ps[k]) for k in ("X_pc", "Y_pc", "Z_pc",
                                        "U_kms", "V_kms", "W_kms", "dist_pc")})
    return a


def _fetch_shell(anchor: dict, search_pc: float, g_max: float,
                 limit: int) -> pd.DataFrame:
    """Pull the all-sky 6D shell whose heliocentric distances bracket the
    ``search_pc`` sphere around the anchor."""
    d0 = anchor["dist_pc"]
    d_lo = max(d0 - search_pc, 1.0)
    d_hi = d0 + search_pc
    plx_min = 1000.0 / d_hi
    plx_max = 1000.0 / d_lo
    q = _SHELL_QUERY.format(limit=int(limit), plx_min=plx_min, plx_max=plx_max)
    df = _run_query(q)
    if g_max is not None and "phot_g_mean_mag" in df.columns:
        df = df[pd.to_numeric(df["phot_g_mean_mag"], errors="coerce") < g_max]
    print(f"[panspermia] {len(df)} Gaia 6D stars in the distance shell "
          f"[{d_lo:.1f}, {d_hi:.1f}] pc")
    return df.reset_index(drop=True)


def panspermia_run(cfg: Config | None = None, source_id: int = K2_18_SOURCE_ID,
                   search_pc: float = 40.0, g_max: float = 16.0,
                   limit: int = 400000, t_max_myr: float = 10.0,
                   d_min_max_pc: float = 2.0, anchor: dict | None = None,
                   table: pd.DataFrame | None = None) -> dict:
    """Rank the stars most likely to have received K2-18-origin material.

    ``anchor``/``table`` may be supplied for offline tests instead of querying
    Gaia.  ``search_pc`` is the 3D radius around K2-18 to consider; ``t_max_myr``
    the past-encounter viability window; ``d_min_max_pc`` the closest-approach
    cut that defines the shortlist.
    """
    cfg = cfg or load_config()

    anchor = anchor or _resolve_anchor(source_id)
    anchor = _anchor_phase_space(anchor)

    raw = table if table is not None else _fetch_shell(anchor, search_pc, g_max, limit)
    df = phase_space_6d(raw)
    # Keep only stars with a defined space velocity and inside the 3D search
    # sphere around K2-18 (and drop the anchor itself if present).
    df = closest_approach(anchor, df)
    good = (np.isfinite(df["v_rel_kms"].to_numpy())
            & (df["sep_now_pc"].to_numpy() <= search_pc)
            & (df.get("source_id", pd.Series(-1, index=df.index)).to_numpy()
               != anchor.get("source_id", -1)))
    df = df[good].reset_index(drop=True)

    df = transfer_score(df, t_max_myr=t_max_myr)
    df = flag_comoving(df)

    n_past = int(df["past_encounter"].sum())
    shortlist = df[(df["past_encounter"]) & (df["d_min_pc"] <= d_min_max_pc)].copy()
    shortlist = shortlist.sort_values("transfer_score", ascending=False)
    comoving = df[df["comoving"]].sort_values("v_rel_kms")

    out_dir = cfg.root / "results" / "panspermia"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ("source_id", "ra", "dec", "dist_pc", "phot_g_mean_mag",
                        "bp_rp", "radial_velocity", "sep_now_pc", "v_rel_kms",
                        "t_enc_myr", "d_min_pc", "transfer_score", "comoving")
            if c in df.columns]
    if len(shortlist):
        shortlist[cols].to_csv(out_dir / "recipient_candidates.csv", index=False)
    if len(comoving):
        comoving[cols].to_csv(out_dir / "comoving_companions.csv", index=False)
    # Always write the full scored table (small) so a re-vet needs no re-query.
    df.sort_values("transfer_score", ascending=False)[cols].to_csv(
        out_dir / "encounters_all.csv", index=False)

    def _rec(r) -> dict:
        return {k: (float(r[k]) if k not in ("source_id",) else int(r[k]))
                for k in cols if k in r and pd.notna(r[k]) and k != "comoving"}

    summary = {
        "anchor": {"name": "K2-18", "source_id": int(anchor.get("source_id", 0)),
                   "dist_pc": round(float(anchor["dist_pc"]), 3),
                   "U_kms": round(float(anchor["U_kms"]), 3),
                   "V_kms": round(float(anchor["V_kms"]), 3),
                   "W_kms": round(float(anchor["W_kms"]), 3)},
        "search_pc": search_pc, "t_max_myr": t_max_myr, "d_min_max_pc": d_min_max_pc,
        "n_searched": int(len(df)),
        "n_past_encounter": n_past,
        "n_shortlist": int(len(shortlist)),
        "n_comoving": int(len(comoving)),
        "top_recipients": [_rec(r) for _, r in shortlist.head(20).iterrows()],
        "closest_approach_pc": (round(float(shortlist["d_min_pc"].min()), 4)
                                if len(shortlist) else None),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("[panspermia]", json.dumps({
        "n_searched": summary["n_searched"], "n_past": n_past,
        "n_shortlist": summary["n_shortlist"], "n_comoving": summary["n_comoving"],
        "d_min_min_pc": summary["closest_approach_pc"]}))
    return summary


def targets_run(cfg: Config | None = None, target: str = "hycean",
                crossmatch: bool = False, max_pc: float = 80.0) -> dict:
    """Directed-travel destination ranking over the committed encounter table.

    Reframes the search for a *technological* disperser: reachability is trivial
    at any cruise speed, so rank K2-18's (past-close) neighbours by destination
    quality instead.  ``target`` picks the habitability prior (``"hycean"`` by
    default -- the traveller evolved on a hycean world, so it seeks other hycean
    sub-Neptunes around cool stars, not Earth-analogs).  With ``crossmatch`` the
    runner layers on NASA Exoplanet Archive known-planet / hycean-candidate flags.
    """
    from .reachability import DEFAULT_SPEEDS_C, rank_targets

    cfg = cfg or load_config()
    out_dir = cfg.root / "results" / "panspermia"
    src = out_dir / "encounters_all.csv"
    if not src.exists():
        raise SystemExit(f"no encounter table at {src}; run panspermia-run first")
    enc = pd.read_csv(src)
    ranked = rank_targets(enc, target=target)

    xmatch_note = "not run (offline)"
    if crossmatch:
        try:
            from .exohosts import crossmatch_hosts, fetch_nearby_planets
            planets = fetch_nearby_planets(max_pc=max_pc)
            ranked = crossmatch_hosts(ranked, planets)
            # Boost hosts, boost hycean-candidate hosts most: the sharpest signal.
            ranked["dest_score"] = (ranked["dest_score"]
                                    + 0.5 * ranked["known_planet_host"].astype(float)
                                    + 1.0 * ranked["has_hycean_candidate"].astype(float))
            ranked = ranked.sort_values(["dest_score", "d_min_pc"],
                                        ascending=[False, True])
            xmatch_note = f"{len(planets)} archive planets < {max_pc} pc"
        except Exception as exc:  # noqa: BLE001
            xmatch_note = f"failed: {exc!r}"
            print(f"[panspermia-targets] exoplanet cross-match {xmatch_note}")

    speed_cols = [f"cross_yr_{f:g}c" for f in DEFAULT_SPEEDS_C]
    cols = [c for c in (["source_id", "ra", "dec", "dist_pc", "phot_g_mean_mag",
                         "bp_rp", "sep_now_pc", "v_rel_kms", "t_enc_myr", "d_min_pc",
                         "abs_g", "lum_class", "dest_score"] + speed_cols
                        + ["known_planet_host", "n_planets", "has_temperate_planet",
                           "has_hycean_candidate", "host_name"])
            if c in ranked.columns]
    ranked[cols].to_csv(out_dir / "reachable_targets.csv", index=False)

    ms = ranked[ranked["lum_class"] == "main_sequence"]
    summary = {
        "target_prior": target,
        "crossmatch": xmatch_note,
        "n_candidates": int(len(ranked)),
        "n_main_sequence": int(len(ms)),
        "n_known_hosts": int(ranked.get("known_planet_host", pd.Series(dtype=bool)).sum()),
        "n_hycean_candidate_hosts":
            int(ranked.get("has_hycean_candidate", pd.Series(dtype=bool)).sum()),
        "top_targets": [
            {k: (int(r[k]) if k == "source_id" else r[k])
             for k in ("source_id", "bp_rp", "dist_pc", "d_min_pc", "t_enc_myr",
                       "lum_class", "dest_score", "host_name", "has_hycean_candidate")
             if k in ranked.columns and pd.notna(r.get(k))}
            for _, r in ranked.head(20).iterrows()],
    }
    (out_dir / "targets_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("[panspermia-targets]", json.dumps({
        "target": target, "n_candidates": summary["n_candidates"],
        "n_main_sequence": summary["n_main_sequence"],
        "n_known_hosts": summary["n_known_hosts"],
        "n_hycean_candidate_hosts": summary["n_hycean_candidate_hosts"]}))
    return summary


_TARGET_GAIA_QUERY = """
SELECT source_id, ra, dec, parallax, pmra, pmdec, ruwe,
       astrometric_excess_noise, astrometric_excess_noise_sig,
       ipd_frac_multi_peak, non_single_star, phot_g_mean_mag, bp_rp,
       phot_variable_flag, radial_velocity, radial_velocity_error,
       phot_bp_rp_excess_factor, has_xp_sampled
FROM gaiadr3.gaia_source WHERE source_id = {source_id}
"""

_TARGET_WISE_QUERY = """
SELECT w.w1mpro, w.w2mpro, w.w3mpro, w.w4mpro,
       w.w1sigmpro, w.w2sigmpro, w.w3sigmpro, w.w4sigmpro
FROM gaiadr3.allwise_best_neighbour AS xm
JOIN gaiadr1.allwise_original_valid AS w
  ON w.designation = xm.original_ext_source_id
WHERE xm.source_id = {source_id}
"""


def _run_detectors(t, m, e) -> dict:
    """Run the dip/secular/glint detectors on one light curve -> dicts."""
    from dataclasses import asdict

    from ..dimming.dips import detect_dips
    from ..dimming.glint import detect_glints
    from ..dimming.secular import detect_secular_fade
    d, s, gl = detect_dips(t, m, e), detect_secular_fade(t, m, e), detect_glints(t, m, e)
    return {"dip": asdict(d) if d else None,
            "secular": asdict(s) if s else None,
            "glint": asdict(gl) if gl else None}


def _dossier_lightcurve(ra: float, dec: float) -> dict:
    """Fetch ZTF g+r light curves and run the detectors per band (achromatic vet)."""
    from ..dimming.acquire import fetch_ztf_lightcurve
    from .dossier import lightcurve_verdict

    bands, n_epochs = {}, {}
    for band in ("r", "g"):
        lc = fetch_ztf_lightcurve(ra, dec, band=band)
        n_epochs[band] = 0 if lc is None else len(lc)
        if lc is None or len(lc) < 30:
            continue
        bands[band] = _run_detectors(lc["mjd"].to_numpy(), lc["mag"].to_numpy(),
                                     lc["magerr"].to_numpy())
    verdict = lightcurve_verdict(bands)
    verdict["n_epochs"] = n_epochs
    verdict["source"] = "ZTF"
    return verdict


def _dossier_ir_variability(ra: float, dec: float) -> dict:
    """NEOWISE multi-epoch W1/W2 -> secular mid-IR trend flag."""
    from ..dimming.characterize import fetch_neowise
    from .dossier import ir_variability_verdict
    try:
        nw = fetch_neowise(ra, dec)
    except Exception as exc:  # noqa: BLE001
        return {"ir_variability_flag": False, "reasons": [f"NEOWISE failed: {exc!r}"]}
    return ir_variability_verdict(nw)


def _dossier_tess(ra: float, dec: float) -> dict:
    """TESS/K2 photometry via lightkurve -> the same detectors (best-effort).

    TESS gives continuous, high-precision photometry of these transiting-planet
    hosts -- far more sensitive to transit-shaped anomalies than ground-based ZTF.
    Optional: if lightkurve/MAST is unavailable the channel degrades to 'not
    available' rather than failing the run."""
    from .dossier import lightcurve_verdict
    try:
        import lightkurve as lk
        sr = lk.search_lightcurve(f"{ra} {dec}", mission=("TESS", "K2"))
        if sr is None or len(sr) == 0:
            return {"lightcurve_flag": False, "reasons": ["no TESS/K2 light curve"],
                    "source": "TESS/K2"}
        lc = sr[0].download().remove_nans().normalize()
        t = lc.time.value
        flux = lc.flux.value
        # Convert relative flux to magnitudes for the shared detectors.
        good = flux > 0
        t, flux = t[good], flux[good]
        mag = -2.5 * np.log10(flux / np.median(flux))
        merr = None
        bands = {"TESS": _run_detectors(t, mag, merr)}
        verdict = lightcurve_verdict(bands)   # single band -> needs_vetting only
        verdict["n_epochs"] = {"TESS": int(len(t))}
        verdict["source"] = "TESS/K2"
        # A confirmed dip in a single precise band is meaningful for space
        # photometry (no colour to cross-check), so surface it as needs_vetting.
        return verdict
    except Exception as exc:  # noqa: BLE001
        return {"lightcurve_flag": False, "reasons": [f"TESS unavailable: {exc!r}"],
                "source": "TESS/K2"}


def _dossier_xp(source_id: int) -> dict:
    """Fetch the Gaia XP sampled spectrum and scan for a narrow emission line."""
    from ..xp.acquire import fetch_xp_spectra
    from .dossier import narrow_feature_scan
    try:
        data = fetch_xp_spectra([int(source_id)])
        flux = data["flux"].get(int(source_id))
        wave = data.get("wave")
        if flux is None:
            return {"xp_feature_flag": False, "reasons": ["no XP spectrum"]}
        return narrow_feature_scan(wave, flux)
    except Exception as exc:  # noqa: BLE001
        return {"xp_feature_flag": False, "reasons": [f"XP fetch failed: {exc!r}"]}


def dossier_run(cfg: Config | None = None, targets: list | None = None) -> dict:
    """Exhaustive per-target signature sweep for the directed-travel candidates.

    For each target pulls Gaia astrometry, WISE photometry, ZTF light curves and
    the Gaia XP spectrum, runs every signature detector, and writes one dossier
    per target plus a combined summary.  Acquisition is runner-side; the scorers
    are unit-tested offline.
    """
    from .dossier import (
        TARGETS,
        companion_diagnostics,
        dossier_verdict,
        ir_color_excess,
    )

    cfg = cfg or load_config()
    targets = targets or TARGETS
    out_dir = cfg.root / "results" / "panspermia" / "dossier"
    out_dir.mkdir(parents=True, exist_ok=True)

    dossiers = []
    for tgt in targets:
        name, sid = tgt["name"], int(tgt["source_id"])
        print(f"[dossier] === {name} (Gaia DR3 {sid}) ===")
        # Gaia row.
        try:
            grow = _run_query(_TARGET_GAIA_QUERY.format(source_id=sid))
            row = grow.iloc[0].to_dict() if len(grow) else {}
        except Exception as exc:  # noqa: BLE001
            print(f"[dossier] Gaia query failed: {exc!r}")
            row = {}
        ra = float(row.get("ra", tgt["ra"]))
        dec = float(row.get("dec", tgt["dec"]))
        companion = companion_diagnostics(row)
        # WISE IR excess.
        try:
            wrow = _run_query(_TARGET_WISE_QUERY.format(source_id=sid))
            wise = wrow.iloc[0].to_dict() if len(wrow) else {}
        except Exception as exc:  # noqa: BLE001
            print(f"[dossier] WISE query failed: {exc!r}")
            wise = {}
        ir = ir_color_excess(wise)
        # ZTF (ground, 2-band) + TESS/K2 (space, precise) light curves; NEOWISE
        # mid-IR variability; Gaia XP narrow-line scan.
        lc = _dossier_lightcurve(ra, dec)
        tess = _dossier_tess(ra, dec)
        irvar = _dossier_ir_variability(ra, dec)
        xp = _dossier_xp(sid)

        parts = {"companion": companion, "ir_excess": ir, "ir_variability": irvar,
                 "lightcurve_ztf": lc, "lightcurve_tess": tess, "xp": xp}
        verdict = dossier_verdict(parts)
        coverage = {ch: ("data" if not any("failed" in r or "unavailable" in r
                                            or "no " in r for r in
                                            (v.get("reasons", []) if isinstance(v, dict)
                                             else []))
                         else "no_data")
                    for ch, v in parts.items()}
        coverage["not_covered"] = ["radio (SETI/VLA)", "high-res RV spectra "
                                   "(HARPS/ESPRESSO)", "X-ray"]
        dossier = {"name": name, "source_id": sid, "ra": ra, "dec": dec,
                   "gaia": {k: row.get(k) for k in
                            ("parallax", "phot_g_mean_mag", "bp_rp",
                             "phot_variable_flag", "radial_velocity")},
                   **parts, "coverage": coverage, "verdict": verdict}
        (out_dir / f"{name.replace(' ', '_')}.json").write_text(
            json.dumps(dossier, indent=2, default=str))
        dossiers.append(dossier)
        print(f"[dossier] {name}: {verdict['verdict']} "
              f"flags={[k for k, v in verdict['channel_flags'].items() if v]}")

    def _reasons(d):
        r = {}
        for ch in ("companion", "ir_excess", "ir_variability", "lightcurve_ztf",
                   "lightcurve_tess", "xp"):
            rs = d[ch].get("reasons", []) if isinstance(d.get(ch), dict) else []
            vet = d[ch].get("needs_vetting", []) if isinstance(d.get(ch), dict) else []
            if rs or vet:
                r[ch] = {"flag": rs, "needs_vetting": vet} if vet else rs
        return r

    summary = {
        "targets": [{"name": d["name"], "verdict": d["verdict"]["verdict"],
                     "flags": [k for k, v in d["verdict"]["channel_flags"].items() if v],
                     "channels": _reasons(d)}
                    for d in dossiers]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("[dossier]", json.dumps(summary, default=str))
    return summary


__all__ = ["panspermia_run", "targets_run", "dossier_run",
           "K2_18_SOURCE_ID", "K2_18_FALLBACK"]
