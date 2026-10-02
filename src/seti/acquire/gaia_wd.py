"""Acquire the Gaia EDR3 white-dwarf parent sample.

Primary source: Gentile Fusillo et al. 2021 (VizieR ``J/MNRAS/508/3877``),
optionally joined to ``gaiadr3.gaia_source`` for astrometric-quality columns
(RUWE, parallax_over_error, astrometric_excess_noise).  Results are memoised to
the parquet cache so subsequent (and offline) runs do not hit the network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ..io import cached

# Columns we rename from the VizieR catalogue to the pipeline's schema.
_VIZIER_RENAME = {
    "WD": "wd_name",
    "GaiaEDR3": "source_id",
    "RA_ICRS": "ra",
    "DE_ICRS": "dec",
    "Pwd": "pwd",
    "Teff": "teff",
    "logg": "logg",
    "Gmag": "Gmag",
    "pmRA": "pmra",
    "pmDE": "pmdec",
    "Plx": "parallax",
}


def _fetch_vizier(pwd_min: float, row_limit: int) -> pd.DataFrame:
    from astroquery.vizier import Vizier

    v = Vizier(columns=["**"], row_limit=row_limit)
    v.ROW_LIMIT = row_limit
    cats = v.get_catalogs("J/MNRAS/508/3877")
    df = cats[0].to_pandas()
    df = df.rename(columns={k: val for k, val in _VIZIER_RENAME.items() if k in df.columns})
    if "pwd" in df.columns:
        df = df[df["pwd"] >= pwd_min]
    return df.reset_index(drop=True)


def acquire_gaia_wd(
    cache_dir: Path,
    pwd_min: float = 0.90,
    row_limit: int = -1,
    force: bool = False,
) -> pd.DataFrame:
    """Return the high-confidence WD parent sample (cached)."""
    params = {"catalog": "J/MNRAS/508/3877", "pwd_min": pwd_min, "row_limit": row_limit}
    return cached(
        cache_dir,
        "gaia_wd",
        params,
        fetch=lambda: _fetch_vizier(pwd_min, row_limit),
        provenance={"source": "Gentile Fusillo et al. 2021, MNRAS 508, 3877"},
        force=force,
    )


def _fetch_gaia_astrometry(source_ids: list[int]) -> pd.DataFrame:
    """Pull astrometric-quality columns from gaiadr3.gaia_source by source_id."""
    from astroquery.gaia import Gaia

    ids = ",".join(str(int(s)) for s in source_ids)
    query = f"""
        SELECT source_id, ruwe, parallax_over_error, astrometric_excess_noise,
               pmra_error, pmdec_error, phot_bp_rp_excess_factor
        FROM gaiadr3.gaia_source
        WHERE source_id IN ({ids})
    """
    job = Gaia.launch_job_async(query)
    return job.get_results().to_pandas()


def acquire_gaia_astrometry(cache_dir: Path, source_ids, force: bool = False) -> pd.DataFrame:
    """Astrometric quality cached by the ordered native integer query IDs.

    Count-only legacy entries do not bind a target request and remain untouched.
    Native query integer coercion is retained; already rounded float inputs cannot be recovered.
    """
    source_ids = [int(s) for s in source_ids]
    identity = json.dumps(source_ids, separators=(",", ":")).encode("ascii")
    params = {
        "table": "gaiadr3.gaia_source",
        "n_in": len(source_ids),
        "source_ids_sha256": hashlib.sha256(identity).hexdigest(),
    }
    return cached(
        cache_dir,
        "gaia_astrometry",
        params,
        fetch=lambda: _fetch_gaia_astrometry(source_ids),
        provenance={"source": "gaiadr3.gaia_source astrometric quality"},
        force=force,
    )
