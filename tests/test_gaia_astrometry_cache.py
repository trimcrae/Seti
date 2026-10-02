"""Offline baseline for Gaia astrometry cache identity.

Only the catalogue fetch is replaced. Production memoisation, temporary
parquet persistence and provenance reads remain real; no network is allowed.
"""

import json

import pandas as pd

from seti.acquire import gaia_wd
from seti.io import query_key


def test_equal_size_distinct_targets_reuse_unbound_legacy_cache(tmp_path, monkeypatch):
    """Production cache witness, not an observed catalogue result."""
    calls = []

    def fake_fetch(source_ids):
        calls.append(list(source_ids))
        return pd.DataFrame({"source_id": source_ids, "ruwe": [1.0] * len(source_ids)})

    monkeypatch.setattr(gaia_wd, "_fetch_gaia_astrometry", fake_fetch)
    first = gaia_wd.acquire_gaia_astrometry(tmp_path, [111])
    legacy_key = query_key(
        "gaia_astrometry", {"table": "gaiadr3.gaia_source", "n_in": 1}
    )
    parquet = tmp_path / f"{legacy_key}.parquet"
    sidecar = parquet.with_suffix(".parquet.provenance.json")
    first_bytes = parquet.read_bytes()
    first_provenance = sidecar.read_bytes()

    second = gaia_wd.acquire_gaia_astrometry(tmp_path, [222])
    repeat = gaia_wd.acquire_gaia_astrometry(tmp_path, [111])

    assert first["source_id"].tolist() == [111]
    assert second["source_id"].tolist() == [111]  # Wrong target reused by baseline.
    assert repeat["source_id"].tolist() == [111]
    assert calls == [[111]]
    assert parquet.read_bytes() == first_bytes
    assert sidecar.read_bytes() == first_provenance
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        f"{legacy_key}.parquet", f"{legacy_key}.parquet.provenance.json"
    ]
    assert json.loads(first_provenance)["params"] == {
        "table": "gaiadr3.gaia_source", "n_in": 1
    }
