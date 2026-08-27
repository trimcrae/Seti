"""Shared fixtures, and one guard that is not a fixture.

THE GUARD: NO TEST IN THIS SUITE MAY REACH THE NETWORK.

Not a style rule -- a bug that cost six consecutive red merges on `main`.
`test_the_probe_writes_a_record_even_when_nothing_is_reachable` stubbed two of
the probe's three survey clients; the third, added later, was left to call IRSA
for real.  In this sandbox that call fails (no egress), the probe returns
NO_FEED_REACHED and the test passes.  On the runner IRSA answers, the probe
returns PARTIAL, and the test fails.  So the local suite was green while CI was
red, and each of six merges looked verified.

A test whose result depends on whether the machine running it can reach the
internet is not testing this repository.  Every outbound connection from inside
a test now raises instead, naming the address, so the failure appears in the
environment where the test was WRITTEN rather than only in the one where it is
run.  Acquisition code that genuinely talks to a service is exercised on the
runner by the probe workflows, which is where a live answer means something.
"""

import socket

import pytest

from seti.config import load_config
from seti.sample import make_sample

_REAL_CONNECT = socket.socket.connect
_REAL_CREATE = socket.create_connection


class NetworkUseInTest(RuntimeError):
    """A test tried to open a socket to somewhere."""


def _is_local(address) -> bool:
    """Loopback is allowed: it is a fixture talking to itself, not a service."""
    try:
        host = address[0] if isinstance(address, tuple) else str(address)
    except Exception:                                          # noqa: BLE001
        return False
    return str(host) in ("127.0.0.1", "::1", "localhost", "")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch, request):
    """Fail any test that opens a socket, unless it asks not to be guarded.

    Opt out with ``@pytest.mark.allow_network`` -- and expect to justify it,
    because a test that needs a live service is a test whose result depends on
    that service being up.
    """
    if request.node.get_closest_marker("allow_network"):
        return

    def blocked_connect(self, address, *a, **k):
        if _is_local(address):
            return _REAL_CONNECT(self, address, *a, **k)
        raise NetworkUseInTest(
            f"a test tried to connect to {address!r}. Tests must not reach the "
            f"network: the result would then depend on whether the machine "
            f"running them has egress, which is how the suite went green here "
            f"while CI was red. Stub the client, or mark the test "
            f"@pytest.mark.allow_network and say why.")

    def blocked_create(address, *a, **k):
        if _is_local(address):
            return _REAL_CREATE(address, *a, **k)
        raise NetworkUseInTest(
            f"a test tried to open a connection to {address!r}; see "
            f"tests/conftest.py")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    monkeypatch.setattr(socket, "create_connection", blocked_create)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "allow_network: this test may open sockets to real services "
        "(justify it -- its result then depends on that service being up)")


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def sample():
    return make_sample(seed=7)
