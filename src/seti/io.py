"""Parquet caching, query memoisation, and provenance.

Every external catalogue query is memoised to ``data/cache/<key>.parquet`` keyed
by a hash of the query parameters, with a sibling ``.provenance.json`` recording
what was fetched, when, and how many rows.  Re-runs hit the cache, so the whole
pipeline is reproducible and offline-friendly once populated.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd


def query_key(name: str, params: dict[str, Any]) -> str:
    """Stable cache key from a query name + parameters."""
    blob = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.sha1(blob.encode()).hexdigest()[:12]
    return f"{name}-{digest}"


def write_parquet(df: pd.DataFrame, path: Path, provenance: dict | None = None) -> Path:
    """Write a DataFrame to parquet and an optional provenance sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    if provenance is not None:
        prov = dict(provenance)
        prov.setdefault("n_rows", len(df))
        prov.setdefault("columns", list(df.columns))
        path.with_suffix(path.suffix + ".provenance.json").write_text(
            json.dumps(prov, indent=2, default=str)
        )
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def cached(
    cache_dir: Path,
    name: str,
    params: dict[str, Any],
    fetch: Callable[[], pd.DataFrame],
    *,
    provenance: dict | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Return cached query result if present, else call ``fetch`` and cache it.

    ``fetch`` is a zero-argument callable that performs the (possibly slow,
    network-dependent) query and returns a DataFrame.  This is the single choke
    point through which all network access flows, so offline runs simply never
    invoke ``fetch`` as long as the cache is warm.
    """
    cache_dir = Path(cache_dir)
    key = query_key(name, params)
    path = cache_dir / f"{key}.parquet"
    if path.exists() and not force:
        return read_parquet(path)
    df = fetch()
    prov = {"query": name, "params": params}
    if provenance:
        prov.update(provenance)
    write_parquet(df, path, provenance=prov)
    return df
