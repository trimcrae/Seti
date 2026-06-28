"""Acquire WISE photometry and crowding metrics for the WD sample.

Strategy (most reproducible path):
  * CatWISE2020 (VizieR ``II/365``) via the CDS X-Match service -> primary W1/W2
    photometry + proper motions (for the co-movement test).
  * AllWISE neighbourhood counts (``gaiadr3.allwise_neighbourhood`` via the Gaia
    TAP) -> the multi-counterpart crowding flag.
  * unWISE (IRSA) forced photometry -> independent W1 cross-check (shortlist).

Each leg is memoised.  X-Match uploads are chunked to respect CDS limits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..io import cached

_CATWISE_RENAME = {
    "RA_ICRS": "ra_wise",
    "DE_ICRS": "dec_wise",
    "W1mproPM": "W1mag",
    "W2mproPM": "W2mag",
    "e_W1mproPM": "e_W1mag",
    "e_W2mproPM": "e_W2mag",
    "pmRA": "pmra_wise",
    "pmDE": "pmdec_wise",
    "e_pmRA": "e_pmra_wise",
    "e_pmDE": "e_pmdec_wise",
    "ccf": "cc_flags",
    "qph": "ph_qual",
}


def _xmatch_catwise(positions: pd.DataFrame, radius_arcsec: float) -> pd.DataFrame:
    from astropy import units as u
    from astroquery.xmatch import XMatch

    chunks = []
    n = len(positions)
    step = 80_000
    for start in range(0, n, step):
        sub = positions.iloc[start:start + step][["source_id", "ra", "dec"]]
        res = XMatch.query(
            cat1=sub,
            cat2="vizier:II/365/catwise",
            max_distance=radius_arcsec * u.arcsec,
            colRA1="ra",
            colDec1="dec",
        )
        chunks.append(res.to_pandas())
    out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return out.rename(columns={k: v for k, v in _CATWISE_RENAME.items() if k in out.columns})


def acquire_catwise(
    cache_dir: Path,
    positions: pd.DataFrame,
    radius_arcsec: float = 3.0,
    force: bool = False,
) -> pd.DataFrame:
    """CatWISE2020 cross-match for a table carrying ``source_id, ra, dec``."""
    params = {"catalog": "II/365", "radius_arcsec": radius_arcsec, "n_in": len(positions)}
    return cached(
        cache_dir,
        "catwise_xmatch",
        params,
        fetch=lambda: _xmatch_catwise(positions, radius_arcsec),
        provenance={"source": "CatWISE2020, Marocco et al. 2021 (VizieR II/365)"},
        force=force,
    )


def _fetch_allwise_neighbourhood(source_ids: list[int]) -> pd.DataFrame:
    """Count AllWISE counterparts per Gaia source via the Gaia TAP."""
    from astroquery.gaia import Gaia

    ids = ",".join(str(int(s)) for s in source_ids)
    query = f"""
        SELECT source_id, COUNT(*) AS n_wise_neighbours
        FROM gaiadr3.allwise_neighbourhood
        WHERE source_id IN ({ids})
        GROUP BY source_id
    """
    job = Gaia.launch_job_async(query)
    return job.get_results().to_pandas()


def acquire_allwise_neighbourhood(
    cache_dir: Path, source_ids, force: bool = False
) -> pd.DataFrame:
    source_ids = list(source_ids)
    params = {"table": "gaiadr3.allwise_neighbourhood", "n_in": len(source_ids)}
    return cached(
        cache_dir,
        "allwise_neighbourhood",
        params,
        fetch=lambda: _fetch_allwise_neighbourhood(source_ids),
        provenance={"source": "gaiadr3.allwise_neighbourhood (Gaia DR3 crossmatch)"},
        force=force,
    )


def add_w2snr(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a W2 signal-to-noise from the magnitude error if absent."""
    out = df.copy()
    if "w2snr" not in out.columns and "e_W2mag" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["w2snr"] = 1.0 / (0.4 * np.log(10.0) * out["e_W2mag"])
    return out
