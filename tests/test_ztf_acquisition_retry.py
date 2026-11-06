"""Transient archive failures must not become empty coverage or a broker substitution."""

from types import SimpleNamespace

import pytest
import requests

from seti.tocsin.altfeeds import AltFeedError, ZtfIrsa


class Session:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.queries = []

    def get(self, endpoint, params):
        self.queries.append((endpoint, params.copy()))
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.parametrize("first", [requests.exceptions.ReadTimeout("unmeasured"),
                                 requests.exceptions.ConnectionError("reset"),
                                 SimpleNamespace(status_code=502),
                                 SimpleNamespace(status_code=503),
                                 SimpleNamespace(status_code=504)])
def test_one_transient_failure_retries_the_identical_archive_query(first):
    ok = SimpleNamespace(status_code=200, text="object-specific payload")
    client = ZtfIrsa()
    client._s = Session([first, ok])
    params = client._params(187.2779, 2.0524, mjd_lo=59000, mjd_hi=59365)
    assert client._get_lightcurve(params) is ok
    assert len(client._s.queries) == 2
    assert client._s.queries[0] == client._s.queries[1]


def test_repeated_timeouts_still_fail_without_an_empty_result():
    client = ZtfIrsa()
    client._s = Session([requests.exceptions.ReadTimeout("first"),
                         requests.exceptions.ReadTimeout("second")])
    with pytest.raises(requests.exceptions.ReadTimeout, match="second"):
        client._get_lightcurve({"ID": "123"})
    assert len(client._s.queries) == 2


@pytest.mark.parametrize("recovers", [True, False])
def test_target_walk_only_emits_measured_photometry(recovers):
    client = ZtfIrsa()
    payload = "mjd,mag,magerr,catflags,filtercode,oid\n59001,15,0.02,0,zg,123\n"
    second = (SimpleNamespace(status_code=200, text=payload) if recovers else
              requests.exceptions.ReadTimeout("still unmeasured"))
    client._s = Session([requests.exceptions.ReadTimeout("first"), second])
    curves = client.lightcurves([{"target_id": "star", "ra": 10, "dec": 2,
                                 "mjd_lo": 59000, "mjd_hi": 59002}])
    assert list(curves) == (["star"] if recovers else [])
    if recovers:
        assert curves["star"].mjd.tolist() == [59001]
    assert len(client._s.queries) == 2


def test_no_retry_after_budget_exhaustion():
    client = ZtfIrsa()
    client._s = Session([requests.exceptions.ReadTimeout("first")])
    with pytest.raises(AltFeedError, match="unmeasured"):
        client._get_lightcurve({}, deadline=10, clock=lambda: 10)
    assert len(client._s.queries) == 1


@pytest.mark.parametrize("status", [200, 400, 401, 404, 429])
def test_success_empty_incompatible_and_throttled_responses_are_not_retried(status):
    response = SimpleNamespace(status_code=status)
    client = ZtfIrsa()
    client._s = Session([response])
    assert client._get_lightcurve({}) is response
    assert len(client._s.queries) == 1
