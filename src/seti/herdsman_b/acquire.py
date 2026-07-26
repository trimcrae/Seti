"""Acquisition for HERDSMAN-B (runner only; every stage checkpoints to disk).

Three fetches, each resumable:

1. **Cluster membership** — Hunt & Reffert census via TAPVizieR (2024 update
   first, 2023 original as fallback). Column names are resolved dynamically
   from a TOP-1 probe so VizieR naming drift cannot break the pull.
2. **Member chemistry** — the member source_ids joined to gaiadr3.gaia_source
   through TAP uploads in chunks; each chunk lands in its own parquet and is
   skipped on re-run (checkpoint granularity = one chunk, ~2 min).
3. **Field baseline** — a random Gaia sample with GSP-Phot metallicities, from
   which the score stage builds the local field [M/H] spread as a function of
   Galactocentric radius (the thing an artificial assembly should mirror).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

# (catalog, members table, clusters known-name hints) — try in order.
_MEMBER_TABLES = ['"J/A+A/686/A42/members"', '"J/A+A/673/A114/members"']
_VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"

_ID_NAMES = ("gaiadr3", "source", "source_id", "gaia")
_CLUSTER_NAMES = ("cluster", "name")
_PROB_NAMES = ("prob", "proba", "pmem", "probability")

_CHEM_QUERY = """
SELECT u.source_id AS uid, g.source_id, g.ra, g.dec,
       g.parallax, g.parallax_over_error,
       g.phot_g_mean_mag, g.bp_rp, g.ruwe,
       g.mh_gspphot, g.mh_gspphot_lower, g.mh_gspphot_upper,
       g.teff_gspphot, g.logg_gspphot, g.ag_gspphot
FROM tap_upload.ids AS u
JOIN gaiadr3.gaia_source AS g ON g.source_id = u.source_id
"""

_FIELD_QUERY = """
SELECT TOP {top} source_id, ra, dec, parallax, parallax_over_error,
       phot_g_mean_mag, mh_gspphot, mh_gspphot_lower, mh_gspphot_upper,
       teff_gspphot
FROM gaiadr3.gaia_source
WHERE random_index < {rand_max}
  AND mh_gspphot IS NOT NULL
  AND parallax > 0.2 AND parallax_over_error > 5
  AND teff_gspphot BETWEEN {teff_lo} AND {teff_hi}
"""


def _resolve_columns(colnames) -> dict:
    """Map heterogeneous VizieR column names onto id/cluster/prob."""
    out = {}
    low = {c.lower(): c for c in colnames}
    for key, cands in (("id", _ID_NAMES), ("cluster", _CLUSTER_NAMES),
                       ("prob", _PROB_NAMES)):
        for c in cands:
            hit = next((low[x] for x in low if x == c or x.startswith(c)), None)
            if hit:
                out[key] = hit
                break
    return out


def fetch_membership(out_path: Path) -> pd.DataFrame:
    """Pull the cluster-membership table (checkpointed to parquet)."""
    if out_path.exists():
        print(f"[herdsman-b] membership checkpoint exists: {out_path}")
        return pd.read_parquet(out_path)
    import pyvo

    tap = pyvo.dal.TAPService(_VIZIER_TAP)
    last_err = None
    for table in _MEMBER_TABLES:
        try:
            probe = tap.run_sync(f"SELECT TOP 1 * FROM {table}").to_table()
            cols = _resolve_columns(probe.colnames)
            if len(cols) < 3:
                raise RuntimeError(f"{table}: unresolved columns {probe.colnames}")
            q = (f'SELECT "{cols["id"]}", "{cols["cluster"]}", "{cols["prob"]}" '
                 f"FROM {table}")
            print(f"[herdsman-b] pulling members from {table} "
                  f"(cols {cols}) ...")
            res = tap.run_async(q).to_table().to_pandas()
            res.columns = ["source_id", "cluster", "prob"]
            res["source_id"] = pd.to_numeric(res["source_id"], errors="coerce")
            res = res.dropna(subset=["source_id"])
            res["source_id"] = res["source_id"].astype(np.int64)
            res.to_parquet(out_path, index=False)
            print(f"[herdsman-b] {len(res)} members, "
                  f"{res['cluster'].nunique()} clusters -> {out_path}")
            return res
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[herdsman-b] {table} failed: {exc!r}")
    raise RuntimeError(f"membership fetch failed everywhere: {last_err!r}")


def fetch_chemistry(members: pd.DataFrame, out_dir: Path,
                    chunk: int = 120_000, retries: int = 3) -> pd.DataFrame:
    """Join member ids to gaia_source chemistry, one checkpointed chunk at a time."""
    from astropy.table import Table
    from astroquery.gaia import Gaia

    out_dir.mkdir(parents=True, exist_ok=True)
    ids = np.unique(members["source_id"].to_numpy(np.int64))
    n_chunks = int(np.ceil(len(ids) / chunk))
    frames = []
    for i in range(n_chunks):
        part = out_dir / f"chem_chunk_{i:03d}.parquet"
        if part.exists():
            frames.append(pd.read_parquet(part))
            continue
        sub = Table({"source_id": ids[i * chunk:(i + 1) * chunk]})
        last = None
        for attempt in range(retries):
            try:
                job = Gaia.launch_job_async(
                    _CHEM_QUERY, upload_resource=sub,
                    upload_table_name="ids")
                df = job.get_results().to_pandas()
                df = df.rename(columns={c: c.lower() for c in df.columns})
                df.to_parquet(part, index=False)          # checkpoint
                frames.append(df)
                print(f"[herdsman-b] chem chunk {i + 1}/{n_chunks}: "
                      f"{len(df)} rows")
                last = None
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                print(f"[herdsman-b] chem chunk {i} attempt "
                      f"{attempt + 1} failed: {exc!r}")
                time.sleep(3 * (attempt + 1))
        if last is not None:
            raise RuntimeError(f"chem chunk {i} failed: {last!r}")
    chem = pd.concat(frames, ignore_index=True)
    return chem


def fetch_field(out_path: Path, top: int = 2_000_000,
                rand_max: int = 80_000_000, teff_lo: float = 4000.0,
                teff_hi: float = 7500.0, retries: int = 4) -> pd.DataFrame:
    """Random Gaia field sample with metallicities (checkpointed, retried).

    The Gaia TAP server intermittently drops async results (HTTP 500 "cannot
    find result") — the failure that killed the first census run four hours
    in.  Same backoff discipline as the chemistry chunks.
    """
    if out_path.exists():
        print(f"[herdsman-b] field checkpoint exists: {out_path}")
        return pd.read_parquet(out_path)
    from astroquery.gaia import Gaia

    q = _FIELD_QUERY.format(top=top, rand_max=rand_max, teff_lo=teff_lo,
                            teff_hi=teff_hi)
    last = None
    for attempt in range(retries):
        try:
            job = Gaia.launch_job_async(q)
            df = job.get_results().to_pandas()
            df = df.rename(columns={c: c.lower() for c in df.columns})
            df.to_parquet(out_path, index=False)
            print(f"[herdsman-b] field sample: {len(df)} stars -> {out_path}")
            return df
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[herdsman-b] field attempt {attempt + 1}/{retries} "
                  f"failed: {exc!r}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"field fetch failed after {retries} attempts: {last!r}")


__all__ = ["fetch_membership", "fetch_chemistry", "fetch_field"]
