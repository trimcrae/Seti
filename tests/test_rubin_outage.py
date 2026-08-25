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


def _d(alerce=None, fink=None, lasair=None, fink_ztf=None, alerce_other=None):
    return roc.decide({"frontier_mjd": alerce, "newest_any_survey_mjd": alerce_other},
                      {"frontier_mjd": fink,
                       "ztf_control": {"frontier_mjd": fink_ztf}},
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


# --- Fink's per-night statistics, the second independent histogram -----------

FINK_ROWS = [
    {"f:night": "20260712", "f:alerts": "71983"},
    {"f:night": "20260714", "f:alerts": "473344"},
    {"f:night": "20260713", "f:alerts": "744559"},
]


def test_fink_night_rows_are_parsed_and_ordered():
    nights = roc._nights_from_fink_stats(FINK_ROWS)
    assert [n["date"] for n in nights] == ["2026-07-12", "2026-07-13", "2026-07-14"]
    assert nights[-1]["n_alerts"] == pytest.approx(473344)


def test_a_night_label_lands_within_half_a_day_of_its_alerts():
    # f:night is a DATE, not an instant. A Chilean night's alerts run ~00:00-10:00
    # UT on the labelled date; placing the label at noon keeps every one of them
    # inside the one-night tolerance, so two brokers holding the same night can
    # never be scored as one ahead of the other.
    [n] = roc._nights_from_fink_stats([{"f:night": "20260714", "f:alerts": "1"}])
    assert n["mjd"] == pytest.approx(61235.5)
    assert abs(n["mjd"] - FROZEN) < roc.AHEAD_TOLERANCE_DAYS


def test_malformed_night_labels_are_dropped_not_guessed():
    # A short, non-numeric or impossible label must not become an epoch: an
    # invented night is either a phantom frontier or a phantom stall.
    assert roc._nights_from_fink_stats(
        [{"f:night": "2026071"}, {"f:night": "abcdefgh"},
         {"f:night": "20260732"}, {"f:alerts": "5"}]) == []


def test_a_live_control_corroborates_but_does_not_create_the_verdict():
    # Fink current on ZTF while both brokers' LSST feeds stop on the same night
    # is the strongest form of this result: the brokers are alive and simply have
    # no LSST alerts to serve.
    d = _d(alerce=FROZEN, fink=FROZEN + 0.1, fink_ztf=roc._now_mjd() - 1.0)
    assert d["verdict"] == "SKY_STOPPED"
    assert d["live_controls"] == ["fink_ztf"]
    assert "Corroborated" in d["why"]


def test_a_control_that_is_itself_stale_corroborates_nothing():
    d = _d(alerce=FROZEN, fink=FROZEN + 0.1, fink_ztf=FROZEN)
    assert d["verdict"] == "SKY_STOPPED"
    assert d["live_controls"] == []
    assert "Corroborated" not in d["why"]


def test_a_live_control_never_overrides_a_broker_that_is_ahead():
    # A current ZTF feed says nothing about LSST. If Fink's LSST side is ahead of
    # ALeRCE's, that is a stalled mirror no matter how healthy the control looks.
    d = _d(alerce=FROZEN, fink=FROZEN + 30.0, fink_ztf=roc._now_mjd())
    assert d["verdict"] == "MIRROR_STALLED"


# --- the observatory's own words (scripts/rubin_status_fetch.py) -------------

_SPEC2 = importlib.util.spec_from_file_location(
    "rubin_status_fetch",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "rubin_status_fetch.py")
rsf = importlib.util.module_from_spec(_SPEC2)
_SPEC2.loader.exec_module(rsf)


def test_block_tags_become_line_breaks_not_run_on_text():
    # The dated status lines are the whole point of the fetch; collapsing block
    # tags into spaces runs a heading into the paragraph below and buries them.
    got = rsf.strip_html("<h2>18 August</h2><p>Roads inspected.</p><p>Snow cleared.</p>")
    assert "18 August" in got
    assert got.count("\n") >= 2


def test_script_and_style_bodies_never_reach_the_text():
    got = rsf.strip_html(
        "<style>.a{color:red}</style><script>var closed=1;</script><p>on sky</p>")
    assert "color:red" not in got and "var closed" not in got
    assert "on sky" in got


def test_entities_are_decoded():
    assert "Cerro Pachón" in rsf.strip_html("<p>Cerro Pach&oacute;n</p>")


def test_phrases_are_counted_not_scored():
    # Reported side by side on purpose: a recovery post and a closure post share
    # most of their vocabulary, and a script that scored them would be guessing
    # where a human can simply read the dated post.
    text = "The road is open but the summit remains inaccessible without power."
    assert rsf.phrase_hits(text, rsf.RESUMED_PHRASES) == {"road is open": 1}
    assert set(rsf.phrase_hits(text, rsf.CLOSED_PHRASES)) == {
        "inaccessible", "without power"}
