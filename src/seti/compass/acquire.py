"""Gaia DR3 NSS acquisition for COMPASS (runner-side; sandbox blocks Gaia).

Chunked by random_index with an explicit maxrec and a truncation check on
every chunk (the MIDDEN 20k-cap lesson, applied preemptively), checkpointed
per chunk so a killed job resumes at the chunk boundary.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

_GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"
_MAXREC = 400_000
_N_CHUNKS = 8
_RANDOM_INDEX_MAX = 1_811_709_771          # DR3 gaia_source rows

_NSS_QUERY = """
SELECT n.source_id, n.nss_solution_type,
       n.a_thiele_innes, n.b_thiele_innes, n.f_thiele_innes, n.g_thiele_innes,
       n.period, n.eccentricity, n.significance,
       s.ra, s.dec, s.parallax, s.parallax_over_error,
       s.pmra, s.pmdec, s.radial_velocity, s.phot_g_mean_mag,
       s.mh_gspphot, s.teff_gspphot, s.random_index
FROM gaiadr3.nss_two_body_orbit AS n
JOIN gaiadr3.gaia_source AS s ON s.source_id = n.source_id
WHERE n.nss_solution_type IN ('Orbital', 'OrbitalTargetedSearch',
                              'OrbitalTargetedSearchValidated')
  AND n.significance >= {sig_min}
  AND s.parallax_over_error >= {poe_min}
  AND s.parallax > 0
  AND s.random_index BETWEEN {lo} AND {hi}
"""


def _retry(fn, retries=4, label="fetch"):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[compass] {label} attempt {attempt + 1}/{retries} "
                  f"failed: {exc!r}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{label} failed after {retries} attempts: {last!r}")


def fetch_nss(out_dir: Path, sig_min: float = 10.0,
              poe_min: float = 5.0) -> pd.DataFrame:
    """All qualifying DR3 astrometric orbits with Thiele-Innes coefficients."""
    out_path = out_dir / "nss_sample.parquet"
    if out_path.exists():
        print(f"[compass] sample checkpoint exists: {out_path}")
        return pd.read_parquet(out_path)
    import pyvo

    tap = pyvo.dal.TAPService(_GAIA_TAP)
    chunk_dir = out_dir / "nss_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(0, _RANDOM_INDEX_MAX, _N_CHUNKS + 1).astype(np.int64)
    frames = []
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
        part = chunk_dir / f"nss_{k:02d}.parquet"
        if part.exists():
            frames.append(pd.read_parquet(part))
            continue
        q = _NSS_QUERY.format(sig_min=sig_min, poe_min=poe_min,
                              lo=int(lo), hi=int(hi))

        def _go(q=q):
            res = tap.run_async(q, maxrec=_MAXREC)
            df = res.to_table().to_pandas()
            return df.rename(columns={c: c.lower() for c in df.columns})

        df = _retry(_go, label=f"nss chunk {k}")
        if len(df) >= _MAXREC:
            raise RuntimeError(
                f"[compass] chunk {k} hit maxrec={_MAXREC} — raise _N_CHUNKS")
        df.to_parquet(part, index=False)
        print(f"[compass] nss chunk {k + 1}/{_N_CHUNKS}: {len(df)} orbits")
        frames.append(df)
    nss = pd.concat(frames, ignore_index=True)
    nss = nss.sort_values("source_id").drop_duplicates("source_id")
    nss.to_parquet(out_path, index=False)
    print(f"[compass] NSS sample: {len(nss)} astrometric orbits")
    return nss
