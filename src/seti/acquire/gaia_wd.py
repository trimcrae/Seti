"""Acquire the Gaia EDR3 white-dwarf parent sample.

Primary source: Gentile Fusillo et al. 2021 (VizieR ``J/MNRAS/508/3877``),
optionally joined to ``gaiadr3.gaia_source`` for astrometric-quality columns
(RUWE, parallax_over_error, astrometric_excess_noise).  Results are memoised to
the parquet cache so subsequent (and offline) runs do not hit the network.
"""

from __future__ import annotations

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
