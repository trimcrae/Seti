"""The mirror-vs-sky discriminator, exercised offline.

The network legs of ``scripts/rubin_outage_check.py`` cannot run in CI, but the
two parts that can silently return the WRONG ANSWER on live data both can:

* :func:`decide` --- calling a stalled mirror a stopped sky is the expensive
  mistake.  It means real Rubin nights go unscreened while the apparatus reports
  a clean null, which is the exact failure mode the frontier alerts exist for.
* :func:`_max_time_in` --- the payload scanner.  If it misses the time field of a
  broker that IS current, that broker reads as silent and the verdict flips to
  SKY_STOPPED for the same reason.
"""

import importlib.util
import pathlib

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "rubin_outage_check",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "rubin_outage_check.py")
roc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(roc)

FROZEN = 61235.41918          # the epoch both channels have been stuck on


def _d(alerce=None, fink=None, lasair=None):
    return roc.decide({"frontier_mjd": alerce}, {"frontier_mjd": fink},
                      {"frontier_mjd": lasair})


def test_a_broker_holding_newer_sky_means_the_mirror_stalled():
    d = _d(alerce=FROZEN, fink=FROZEN + 30.0)
    assert d["verdict"] == "MIRROR_STALLED"
    assert d["brokers_ahead_days"]["fink"] == pytest.approx(30.0, abs=1e-3)


def test_agreement_across_brokers_means_the_stream_stopped():
    d = _d(alerce=FROZEN, fink=FROZEN + 0.02)
    assert d["verdict"] == "SKY_STOPPED"
    assert not d["brokers_ahead_days"]


def test_one_broker_alone_can_never_return_sky_stopped():
    # The whole point of the check is that a single source cannot distinguish
    # the two causes; a confident verdict here would re-create the blind spot.
    d = _d(alerce=FROZEN)
    assert d["verdict"] == "UNDETERMINED_SINGLE_SOURCE"


def test_nothing_reachable_is_reported_as_our_failure_not_rubins():
    d = _d()
    assert d["verdict"] == "NO_BROKER_REACHED"


def test_sub_night_drift_is_not_evidence_of_a_stall():
    # Brokers ingest a night in batches, so their newest epochs differ by hours
    # while both are current.  Treating that as "ahead" would raise MIRROR_STALLED
    # every run and train the reader to ignore the one time it is real.
    assert _d(alerce=FROZEN, fink=FROZEN + 0.9)["verdict"] == "SKY_STOPPED"
    assert _d(alerce=FROZEN, fink=FROZEN + 1.5)["verdict"] == "MIRROR_STALLED"


def test_julian_dates_are_normalised_to_mjd():
    # Fink's ZTF API reports JD.  Left unconverted, 2.46e6 would be compared
    # against 6.1e4 and every run would read as MIRROR_STALLED by 2.4 million days.
    got = roc._max_time_in({"i:jd": 2461265.9})
    assert got == pytest.approx(61265.4, abs=0.1)


def test_the_scanner_finds_the_epoch_wherever_it_is_nested():
    payload = {"results": [{"d:tag": "x", "candidate": {"midpointMjdTai": 61240.5}},
                           {"candidate": {"midpointMjdTai": 61277.25}}]}
    assert roc._max_time_in(payload) == pytest.approx(61277.25)


def test_non_time_numbers_are_not_mistaken_for_epochs():
    # A count, a flux or a source id under a key containing "jd"/"time" must not
    # become a frontier: an invented newer epoch reads as MIRROR_STALLED.
    assert roc._max_time_in({"n_jd_rows": 12345.0, "exposure_time": 30.0}) is None
