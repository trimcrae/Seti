"""Acquire 2MASS J/H/Ks photometry to anchor the photospheric SED.

Uses the precomputed Gaia DR3 crossmatch
(``gaiadr3.tmass_psc_xsc_best_neighbour`` joined to the 2MASS PSC) so no
positional matching code is needed for this leg.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..io import cached


def _fetch_twomass(source_ids: list[int]) -> pd.DataFrame:
    from astroquery.gaia import Gaia

    ids = ",".join(str(int(s)) for s in source_ids)
    query = f"""
        SELECT x.source_id,
               t.j_m  AS Jmag,  t.j_msigcom  AS e_Jmag,
               t.h_m  AS Hmag,  t.h_msigcom  AS e_Hmag,
               t.ks_m AS Ksmag, t.ks_msigcom AS e_Ksmag
        FROM gaiadr3.tmass_psc_xsc_best_neighbour AS x
        JOIN gaiadr3.tmass_psc_xsc_join AS j
          ON x.clean_tmass_psc_xsc_oid = j.clean_tmass_psc_xsc_oid
        JOIN gaiadr1.tmass_original_valid AS t
          ON j.original_psc_source_id = t.designation
        WHERE x.source_id IN ({ids})
    """
    job = Gaia.launch_job_async(query)
    return job.get_results().to_pandas()


def acquire_twomass(cache_dir: Path, source_ids, force: bool = False) -> pd.DataFrame:
    source_ids = list(source_ids)
    params = {"table": "gaiadr3.tmass_psc_xsc_best_neighbour", "n_in": len(source_ids)}
    return cached(
        cache_dir,
        "twomass",
        params,
        fetch=lambda: _fetch_twomass(source_ids),
        provenance={"source": "2MASS PSC via Gaia DR3 crossmatch"},
        force=force,
    )
