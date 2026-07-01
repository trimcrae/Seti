"""End-to-end astrometric-acceleration search: query Gaia, analyse, rank, save.

Pulls the highest-significance acceleration solutions from Gaia DR3
``nss_acceleration_astro`` (joined to ``gaia_source`` for distance and
photometry), computes the physical acceleration and implied companion mass,
rejects the ones already explained by a catalogued stellar/orbital companion, and
writes a ranked shortlist of *dark-companion* candidates to
``results/accel/``.
"""

from __future__ import annotations

import json

import pandas as pd

from ..config import Config, load_config
from .analyze import analyze_accelerations, rank_candidates
from .orbit import analyze_orbits, rank_dark_companions

_ORBIT_QUERY = """
SELECT TOP {limit}
       o.source_id, o.period, o.eccentricity,
       o.a_thiele_innes, o.b_thiele_innes, o.f_thiele_innes, o.g_thiele_innes,
       g.ra, g.dec, g.parallax, g.parallax_over_error,
       g.phot_g_mean_mag, g.bp_rp, g.ruwe
FROM gaiadr3.nss_two_body_orbit AS o
JOIN gaiadr3.gaia_source AS g USING (source_id)
WHERE g.parallax > {plx_min}
  AND o.nss_solution_type LIKE 'Orbital%'
  AND o.a_thiele_innes IS NOT NULL
ORDER BY g.parallax DESC
"""


def _fetch_orbits(limit: int, plx_min: float) -> pd.DataFrame:
    from astroquery.gaia import Gaia
    q = _ORBIT_QUERY.format(limit=int(limit), plx_min=plx_min)
    df = Gaia.launch_job_async(q).get_results().to_pandas()
    return df.rename(columns={c: c.lower() for c in df.columns})

_QUERY = """
SELECT TOP {limit}
       a.source_id, a.accel_ra, a.accel_dec,
       a.accel_ra_error, a.accel_dec_error, a.significance AS gaia_significance,
       g.ra, g.dec, g.parallax, g.parallax_over_error,
       g.phot_g_mean_mag, g.bp_rp, g.ruwe
FROM gaiadr3.nss_acceleration_astro AS a
JOIN gaiadr3.gaia_source AS g USING (source_id)
WHERE g.parallax > {plx_min}
  AND a.significance > {sig_min}
ORDER BY a.significance DESC
"""


def _fetch(limit: int, plx_min: float, sig_min: float) -> pd.DataFrame:
    from astroquery.gaia import Gaia
    q = _QUERY.format(limit=int(limit), plx_min=plx_min, sig_min=sig_min)
    job = Gaia.launch_job_async(q)
    df = job.get_results().to_pandas()
    return df.rename(columns={c: c.lower() for c in df.columns})


def _known_binary(source_ids: list[int]) -> set:
    """Source_ids that already have a Gaia NSS *orbital* solution --- their
    companion is characterised, so they are not an unexplained acceleration."""
    if not source_ids:
        return set()
    from astroquery.gaia import Gaia
    ids = ",".join(str(int(s)) for s in source_ids)
    q = (f"SELECT source_id FROM gaiadr3.nss_two_body_orbit "
         f"WHERE source_id IN ({ids})")
    try:
        return set(int(s) for s in
                   Gaia.launch_job_async(q).get_results()["source_id"])
    except Exception as exc:
        print(f"[accel] orbit cross-check failed: {exc!r}")
        return set()


def _characterize_dark(dark: pd.DataFrame) -> pd.DataFrame:
    """Add SIMBAD identity/type and a luminous-companion flag to the dark list."""
    out = dark.copy()
    # Gaia BP-RP excess factor: strongly elevated => a resolved/blended second
    # luminous source, i.e. the 'dark' mass function is contaminated.
    try:
        from astroquery.gaia import Gaia
        ids = ",".join(str(int(s)) for s in out["source_id"].head(100))
        q = (f"SELECT source_id, phot_bp_rp_excess_factor, non_single_star, "
             f"radial_velocity, rv_amplitude_robust FROM gaiadr3.gaia_source "
             f"WHERE source_id IN ({ids})")
        gd = Gaia.launch_job_async(q).get_results().to_pandas()
        gd = gd.rename(columns={c: c.lower() for c in gd.columns})
        out = out.merge(gd, on="source_id", how="left")
        out["bp_rp_excess"] = pd.to_numeric(out.get("phot_bp_rp_excess_factor"),
                                            errors="coerce")
        out["companion_luminous_flag"] = out["bp_rp_excess"] > 1.5
    except Exception as exc:
        print(f"[accel] Gaia detail fetch failed: {exc!r}")
    try:
        from ..acquire.science import fetch_simbad_context
        pos = out[["source_id", "ra", "dec"]].head(50)
        ctx = fetch_simbad_context(pos)
        if ctx is not None and len(ctx):
            keep = ["source_id"] + [c for c in ("simbad_id", "simbad_otype")
                                    if c in ctx.columns]
            out = out.merge(ctx[keep], on="source_id", how="left")
    except Exception as exc:
        print(f"[accel] SIMBAD characterisation failed: {exc!r}")
    return out


def accel_run(cfg: Config | None = None, limit: int = 6000, plx_min: float = 2.0,
              sig_min: float = 20.0, table: pd.DataFrame | None = None) -> dict:
    """Query, analyse and rank; write results.  ``table`` may be supplied for
    offline tests instead of querying Gaia."""
    cfg = cfg or load_config()
    raw = table if table is not None else _fetch(limit, plx_min, sig_min)
    n_searched = int(len(raw))
    analysed = analyze_accelerations(raw)
    ranked = rank_candidates(analysed)

    # Remove the ones whose companion is already characterised by an orbit.
    if len(ranked) and table is None:
        known = _known_binary(list(ranked["source_id"].head(300)))
        ranked = ranked[~ranked["source_id"].isin(known)].copy()
        ranked["orbit_catalogued"] = False

    out_dir = cfg.root / "results" / "accel"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ("source_id", "ra", "dec", "dist_pc", "parallax_over_error",
                        "phot_g_mean_mag", "bp_rp", "ruwe", "gaia_significance",
                        "accel_significance", "accel_total_mas_yr2", "accel_m_s2",
                        "implied_companion_msun", "abs_g", "dark_companion",
                        "rank_score") if c in ranked.columns]
    if len(ranked):
        ranked[cols].to_csv(out_dir / "accel_candidates.csv", index=False)

    summary = {
        "n_searched": n_searched,
        "n_candidates": int(len(ranked)),
        "selection": {"sig_min": sig_min, "plx_min": plx_min},
        "top_candidates": ranked[cols].head(25).to_dict("records") if len(ranked)
        else [],
    }
    # --- The decisive path: astrometric ORBITS give a companion MASS. ---
    # A full orbit + parallax yields the mass function and hence the companion
    # mass; a massive (>3 Msun) invisible companion is a dormant compact object.
    if table is None:
        try:
            # Distance-ordered + large limit => COMPLETE for the nearby volume,
            # not an arbitrary slice that could miss the massive-companion tail.
            orb = _fetch_orbits(limit=120000, plx_min=plx_min)
            dark = rank_dark_companions(analyze_orbits(orb))
            ocols = [c for c in ("source_id", "ra", "dec", "dist_pc", "period_yr",
                                 "a0_au", "mass_function", "m1_msun", "m2_msun",
                                 "phot_g_mean_mag", "bp_rp", "ruwe", "rank_score")
                     if c in dark.columns]
            # Characterise the (short) dark-companion list: SIMBAD identity/type
            # (known BH? nova? symbiotic?) and Gaia flags that reveal a *luminous*
            # companion faking the dark mass function (BP-RP excess, RV scatter).
            if len(dark):
                dark = _characterize_dark(dark)
                ocols += [c for c in ("simbad_id", "simbad_otype", "bp_rp_excess",
                                      "companion_luminous_flag") if c in dark.columns]
            if len(dark):
                dark[ocols].to_csv(out_dir / "dark_companion_candidates.csv",
                                   index=False)
            summary["n_orbits_searched"] = int(len(orb))
            summary["n_dark_companions"] = int(len(dark))
            summary["top_dark_companions"] = (dark[ocols].head(25).to_dict("records")
                                              if len(dark) else [])
            # --- Decisive novelty test: cross-match the compact-companion (AMRF
            # class >=2) shortlist against the *published* Gaia catalogues.  A
            # class-3 system absent from every list is the remarkable candidate. ---
            if len(dark):
                try:
                    from .crossmatch import crossmatch_candidates
                    from .orbit import KNOWN_GAIA_BH
                    xm = crossmatch_candidates(dark)
                    xm["known_gaia_bh"] = xm["source_id"].isin(KNOWN_GAIA_BH)
                    xm["novel_candidate"] = ((~xm["in_literature"])
                                             & (~xm["known_gaia_bh"]))
                    xcols = ocols + [c for c in ("lit_matches", "match_kind",
                                                 "in_literature", "known_gaia_bh",
                                                 "novel_candidate")
                                     if c in xm.columns]
                    xm[xcols].to_csv(out_dir / "literature_crossmatch.csv",
                                     index=False)
                    novel = xm[xm["novel_candidate"]]
                    summary["n_in_literature"] = int(xm["in_literature"].sum())
                    summary["n_novel_absent"] = int(len(novel))
                    summary["novel_source_ids"] = [int(s) for s in novel["source_id"]]
                    summary["novel_candidates"] = novel[xcols].to_dict("records")
                    print(f"[accel] literature cross-match: {len(xm)} candidates -> "
                          f"{summary['n_in_literature']} known, {len(novel)} ABSENT")
                    if len(novel):
                        print("[accel] NOVEL (absent from published catalogues):")
                        print(novel[xcols].to_string(index=False))
                except Exception as exc:  # noqa: BLE001
                    print(f"[accel] literature cross-match skipped: {exc!r}")
            print(f"[accel] orbits: {len(orb)} -> {len(dark)} massive dark-companion "
                  f"(>3 Msun) candidates")
            if len(dark):
                print(dark[ocols].head(15).to_string(index=False))
        except Exception as exc:
            print(f"[accel] orbit dark-companion search skipped: {exc!r}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[accel] {n_searched} acceleration stars -> "
          f"{len(ranked)} high-acceleration candidates")
    return summary


__all__ = ["accel_run"]
