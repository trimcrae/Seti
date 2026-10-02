"""Offline request-identity controls for native Gaia astrometry memoisation.

Only the private catalogue fetch is replaced. Production query keys, temporary
parquet persistence/provenance and warm reads remain real and network-guarded.
Synthetic response order is a fixture property, not a Gaia service guarantee.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from seti.acquire import gaia_wd
from seti.io import query_key, write_parquet


def _stub_fetch(monkeypatch):
    calls = []

    def fake_fetch(source_ids):
        calls.append(list(source_ids))
        return pd.DataFrame({
            "source_id": source_ids,
            "ruwe": [1.0 + 0.1 * len(calls)] * len(source_ids),
        })

    monkeypatch.setattr(gaia_wd, "_fetch_gaia_astrometry", fake_fetch)
    return calls


def _snapshot(cache):
    return {p.name: p.read_bytes() for p in cache.iterdir()}


def _assert_preserved(cache, snapshot):
    for name, raw in snapshot.items():
        assert (cache / name).read_bytes() == raw


def _bound_receipts(cache):
    out = []
    for path in sorted(cache.glob("*.parquet.provenance.json")):
        receipt = json.loads(path.read_bytes())
        if "source_ids_sha256" in receipt["params"]:
            out.append((path, receipt))
    return out


def _seed_legacy(cache):
    params = {"table": "gaiadr3.gaia_source", "n_in": 1}
    key = query_key("gaia_astrometry", params)
    write_parquet(
        pd.DataFrame({"source_id": [333], "ruwe": [9.0]}),
        cache / f"{key}.parquet",
        provenance={"query": "gaia_astrometry", "params": params, "source": "synthetic"},
    )
    (cache / "unrelated.txt").write_bytes(b"Retain unrelated local bytes.\n")
    return _snapshot(cache)


@pytest.mark.parametrize("first_id,second_id", [
    (111, 222),
    (9007199254740993, 9007199254740994),
    (9223372036854775806, 9223372036854775807),
])
def test_distinct_same_size_requests_keep_exact_targets_and_warm_bytes(
    tmp_path, monkeypatch, first_id, second_id
):
    calls = _stub_fetch(monkeypatch)
    first = gaia_wd.acquire_gaia_astrometry(tmp_path, [first_id])
    first_bytes = _snapshot(tmp_path)
    second = gaia_wd.acquire_gaia_astrometry(tmp_path, [second_id])
    repeat = gaia_wd.acquire_gaia_astrometry(tmp_path, [first_id])

    assert first["source_id"].tolist() == [first_id]
    assert second["source_id"].tolist() == [second_id]
    assert repeat["source_id"].tolist() == [first_id]
    assert calls == [[first_id], [second_id]]
    _assert_preserved(tmp_path, first_bytes)
    receipts = _bound_receipts(tmp_path)
    assert len(receipts) == 2
    assert len(list(tmp_path.glob("*.parquet"))) == 2
    # Literal decimal byte vectors bind the opaque IDs independently of JSON generation.
    expected = {
        hashlib.sha256(f"[{first_id}]".encode("ascii")).hexdigest(),
        hashlib.sha256(f"[{second_id}]".encode("ascii")).hexdigest(),
    }
    assert {r["params"]["source_ids_sha256"] for _, r in receipts} == expected
    fixed_vectors = {
        111: "127309c873b897c7c23aeed40ea1a22676f3ef006fa5b853b03db31107e2f1a2",
        222: "f41676530b59bdeeb95fe413624204edb2a612fbe05f8dbef192c5d7f16ac74a",
        9007199254740993: "ee825a6b803b8c559f4ee311b23736dbc4b315351ce4b34ff75df0228b589b44",
        9007199254740994: "d83457813cf9989d82357c8e14d64e7b69f083b23a4ac3b798f1ccf90a4c910b",
    }
    if first_id in fixed_vectors:
        assert expected == {fixed_vectors[first_id], fixed_vectors[second_id]}
    assert all(r["params"]["n_in"] == 1 for _, r in receipts)
    assert all(r["params"]["table"] == "gaiadr3.gaia_source" for _, r in receipts)


@pytest.mark.parametrize("integer_id,text_id", [
    (111, "00111"),
    (9007199254740993, "9007199254740993"),
    (np.int64(9007199254740993), "9007199254740993"),
])
def test_same_native_integer_query_reuses_across_exact_type_forms(
    tmp_path, monkeypatch, integer_id, text_id
):
    calls = _stub_fetch(monkeypatch)
    first = gaia_wd.acquire_gaia_astrometry(tmp_path, [integer_id])
    original = _snapshot(tmp_path)
    same = gaia_wd.acquire_gaia_astrometry(tmp_path, [text_id])

    assert first["source_id"].tolist() == [int(integer_id)]
    assert same["source_id"].tolist() == [int(integer_id)]
    assert calls == [[int(integer_id)]]
    assert _snapshot(tmp_path) == original
    assert len(_bound_receipts(tmp_path)) == 1


def test_ordered_request_encoding_is_explicit_not_set_equivalence(tmp_path, monkeypatch):
    calls = _stub_fetch(monkeypatch)
    first = gaia_wd.acquire_gaia_astrometry(tmp_path, [111, 222])
    original = _snapshot(tmp_path)
    reordered = gaia_wd.acquire_gaia_astrometry(tmp_path, [222, 111])

    assert first["source_id"].tolist() == [111, 222]
    assert reordered["source_id"].tolist() == [222, 111]
    assert calls == [[111, 222], [222, 111]]
    _assert_preserved(tmp_path, original)
    expected = {
        hashlib.sha256(b"[111,222]").hexdigest(),
        hashlib.sha256(b"[222,111]").hexdigest(),
    }
    receipts = _bound_receipts(tmp_path)
    assert {r["params"]["source_ids_sha256"] for _, r in receipts} == expected
    assert all(r["params"]["n_in"] == 2 for _, r in receipts)


def test_force_refresh_changes_only_bound_entry_keeps_legacy_bytes(tmp_path, monkeypatch):
    legacy = _seed_legacy(tmp_path)
    calls = _stub_fetch(monkeypatch)
    first = gaia_wd.acquire_gaia_astrometry(tmp_path, [111])
    sidecar = _bound_receipts(tmp_path)[0][0]
    bound_path = tmp_path / sidecar.name.removesuffix(".provenance.json")
    original_bound = bound_path.read_bytes()
    refreshed = gaia_wd.acquire_gaia_astrometry(tmp_path, [111], force=True)
    warm = gaia_wd.acquire_gaia_astrometry(tmp_path, [111])

    assert first["source_id"].tolist() == [111]
    assert refreshed["source_id"].tolist() == [111]
    assert warm["source_id"].tolist() == [111]
    assert first["ruwe"].tolist() == [1.1]
    assert refreshed["ruwe"].tolist() == [1.2]
    assert warm["ruwe"].tolist() == [1.2]
    assert calls == [[111], [111]]
    assert bound_path.read_bytes() != original_bound
    _assert_preserved(tmp_path, legacy)
    assert len(_bound_receipts(tmp_path)) == 1
    assert len(list(tmp_path.glob("*.parquet"))) == 2


def test_failed_force_refresh_retains_bound_and_legacy_bytes(tmp_path, monkeypatch):
    legacy = _seed_legacy(tmp_path)
    calls = _stub_fetch(monkeypatch)
    gaia_wd.acquire_gaia_astrometry(tmp_path, [111])
    original = _snapshot(tmp_path)

    def failed_fetch(source_ids):
        calls.append(list(source_ids))
        raise RuntimeError("synthetic offline refusal")

    monkeypatch.setattr(gaia_wd, "_fetch_gaia_astrometry", failed_fetch)
    with pytest.raises(RuntimeError, match="synthetic offline refusal"):
        gaia_wd.acquire_gaia_astrometry(tmp_path, [111], force=True)
    warm = gaia_wd.acquire_gaia_astrometry(tmp_path, [111])

    assert warm["source_id"].tolist() == [111]
    assert calls == [[111], [111]]
    assert _snapshot(tmp_path) == original
    _assert_preserved(tmp_path, legacy)


def test_unbound_legacy_does_not_answer_failed_new_request(tmp_path, monkeypatch):
    original = _seed_legacy(tmp_path)
    calls = []

    def failed_fetch(source_ids):
        calls.append(list(source_ids))
        raise RuntimeError("synthetic offline refusal")

    monkeypatch.setattr(gaia_wd, "_fetch_gaia_astrometry", failed_fetch)
    with pytest.raises(RuntimeError, match="synthetic offline refusal"):
        gaia_wd.acquire_gaia_astrometry(tmp_path, [111])

    assert calls == [[111]]
    assert _snapshot(tmp_path) == original
    assert _bound_receipts(tmp_path) == []


def test_one_pass_generator_binds_the_same_native_integer_literals(tmp_path, monkeypatch):
    calls = _stub_fetch(monkeypatch)
    seen = []

    def targets():
        for raw in ("9007199254740993", "111"):
            seen.append(raw)
            yield raw

    result = gaia_wd.acquire_gaia_astrometry(tmp_path, targets())
    assert seen == ["9007199254740993", "111"]
    assert calls == [[9007199254740993, 111]]
    assert result["source_id"].tolist() == [9007199254740993, 111]
    receipt = _bound_receipts(tmp_path)[0][1]
    assert receipt["params"]["source_ids_sha256"] == hashlib.sha256(
        b"[9007199254740993,111]"
    ).hexdigest()
    assert receipt["params"]["n_in"] == 2
