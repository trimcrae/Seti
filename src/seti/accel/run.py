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
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[accel] {n_searched} acceleration stars -> "
          f"{len(ranked)} dark-companion candidates")
    if len(ranked):
        print(ranked[cols].head(15).to_string(index=False))
    return summary


__all__ = ["accel_run"]
