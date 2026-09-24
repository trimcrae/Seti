"""The ARI mirror route and the sweep's ESA circuit breaker (2026-09-24).

Runs 35884646233 and 35923951760 added zero tiles in 11 h: every ESA parent
query timed out, the IRSA route's Gaia half is ALSO ESA, and each tile spent
its whole budget on ESA before VizieR could be asked.  These tests hold the
fix offline: an independent Gaia DR3 service (GAVO/ARI Heidelberg) for the
Gaia half, and a breaker that stops asking a dead ESA.
"""

import pandas as pd

from seti.ignition.run import stage_sweep
from seti.ignition.sample import (
    ROUTE_ARI,
    ROUTE_ESA,
    ROUTE_IRSA,
    SHAPE_GAIA_ONLY,
    ari_adql,
    build_query,
    fetch_parent,
    parent_columns,
)
from test_ignition import (
    _FIELD,
    _TIMEOUT_500,
    _asu_dead,
    _FakeIRSA,
    _gaia_only_from,
    _stars,
)
from test_ignition_scale import _cone_factory, _sweep_conf


class _DeadESA:
    def __init__(self):
        self.queries = []

    def __call__(self, adql):
        self.queries.append(adql)
        raise RuntimeError(_TIMEOUT_500)


def test_ari_adql_makes_the_variability_cut_null_safe_and_changes_nothing_else():
    q = build_query(field=_FIELD, cap=5, shape=SHAPE_GAIA_ONLY)
    a = ari_adql(q)
    assert "!=" not in a
    assert ("(g.phot_variable_flag IS NULL OR g.phot_variable_flag <> 'VARIABLE')") in a
    assert a.replace("(g.phot_variable_flag IS NULL OR g.phot_variable_flag <> 'VARIABLE')",
                     "g.phot_variable_flag != 'VARIABLE'") == q


def test_dead_esa_falls_through_to_ari_before_the_esa_backed_irsa_route():
    stars = _stars()
    esa = _DeadESA()
    got, rep = fetch_parent({"fields": [_FIELD]}, mode="fields", query_fn=esa,
                            irsa_fetch_fn=_FakeIRSA(stars), vizier_fetch_fn=_asu_dead,
                            ari_query_fn=_gaia_only_from(stars))
    assert rep["routes"] == {ROUTE_ARI: {"units": 1, "rows": 4}}
    assert rep["per_unit"][0]["route"] == ROUTE_ARI
    assert sorted(got["source_id"]) == ["14", "15"]           # the same science cuts
    assert list(got.columns)[:len(parent_columns())] == parent_columns() or \
        set(parent_columns()) <= set(got.columns)
    assert all(SHAPE_GAIA_ONLY not in q for q in esa.queries)  # IRSA route never reached
    assert (got["parent_route"] == ROUTE_ARI).all()


def test_esa_off_asks_neither_esa_nor_the_irsa_route():
    stars = _stars()
    esa = _DeadESA()
    got, rep = fetch_parent({"fields": [_FIELD]}, mode="fields", query_fn=esa,
                            irsa_fetch_fn=_FakeIRSA(stars), vizier_fetch_fn=_asu_dead,
                            ari_query_fn=_gaia_only_from(stars), esa=False)
    assert esa.queries == []
    assert rep["routes"] == {ROUTE_ARI: {"units": 1, "rows": 4}}
    assert ROUTE_IRSA not in rep["routes"] and ROUTE_ESA not in rep["routes"]


def test_an_injected_esa_transport_never_gains_a_live_ari_route():
    # no ari_query_fn given: ARI must stay off rather than reach the network
    got, rep = fetch_parent({"fields": [_FIELD]}, mode="fields", query_fn=_DeadESA(),
                            irsa_fetch_fn=_FakeIRSA(_stars()), vizier_fetch_fn=_asu_dead)
    assert ROUTE_ARI not in (rep.get("routes") or {})


def test_sweep_breaker_stops_asking_a_dead_esa_and_retries_it_periodically(tmp_path):
    conf = _sweep_conf()
    conf["sweep"].update({"esa_trip_after": 2, "esa_retry_every": 3, "sample_prefetch": 0})
    esa = _DeadESA()
    stars = _stars()
    ari_calls = []

    def ari(adql):
        ari_calls.append(adql)
        return _gaia_only_from(stars)(adql)

    rep = stage_sweep(conf, tmp_path, shard=0, n_shards=2, route="cone", query_fn=esa,
                      cone_fn=_cone_factory("constant"), asu_fetch_fn=_asu_dead,
                      irsa_fetch_fn=_FakeIRSA(stars), max_tiles=8, ari_query_fn=ari)
    assert rep["tiles_done"] == 8 and rep["tiles_failed"] == 0
    circles = {q[q.index("CIRCLE"):q.index(")", q.index("CIRCLE")) + 1] for q in esa.queries}
    # tiles 0,1 ask ESA (and trip it); then only every 3rd tile re-tries it: 2, 5
    assert len(circles) == 4
    assert rep["esa_breaker"]["trips"] == 1 and rep["esa_breaker"]["tiles_esa_skipped"] == 4
    assert len(ari_calls) == 8
    assert {t["parent_route"] for t in rep["tiles"]} <= {ROUTE_ARI, None}
    assert isinstance(pd.DataFrame(rep["tiles"]), pd.DataFrame)
