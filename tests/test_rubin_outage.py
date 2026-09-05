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


# --- ZTF: is the other public stream alive? -----------------------------------

_LISTING = """
<tr><td><a href="ztf_public_20260901.tar.gz">ztf_public_20260901.tar.gz</a></td><td align="right">2026-09-02 06:13  </td><td align="right">812M</td></tr>
<a href="ztf_public_20260903.tar.gz">ztf_public_20260903.tar.gz</a>          2026-09-04 06:12  1.3G
<a href="ztf_public_20260904.tar.gz">ztf_public_20260904.tar.gz</a>          2026-09-05 06:13   45
<a href="ztf_public_2026090X.tar.gz">garbage</a>  9G
<a href="ztf_public_20260830.tar.gz">ztf_public_20260830.tar.gz</a>
"""


def test_the_archive_listing_is_parsed_by_name_and_size():
    entries = roc.parse_ztf_archive_listing(_LISTING)
    assert [e["date"] for e in entries] == ["2026-08-30", "2026-09-01", "2026-09-03",
                                            "2026-09-04"]
    by = {e["date"]: e for e in entries}
    assert by["2026-09-01"]["size_bytes"] == int(812 * 1024 ** 2)
    assert by["2026-09-03"]["size_bytes"] == int(1.3 * 1024 ** 3)
    assert by["2026-09-04"]["size_bytes"] == 45
    assert by["2026-08-30"]["size_bytes"] is None          # unknown, not empty
    # A night label sits at noon UT of its date, as Fink's does.
    assert by["2026-09-03"]["mjd"] == pytest.approx(61286.5)


def test_an_empty_tarball_does_not_move_the_frontier():
    fr = roc.ztf_archive_frontier(roc.parse_ztf_archive_listing(_LISTING))
    assert fr["last_night"] == "2026-09-03"               # the 45-byte night is skipped
    assert fr["last_tarball_any_size"] == "2026-09-04"
    assert fr["n_tarballs"] == 4
    assert fr["n_empty_recent"] == 1


def test_an_unknown_size_counts_as_a_night_with_alerts():
    fr = roc.ztf_archive_frontier([{"date": "2026-09-02", "mjd": 61285.5, "size_bytes": None}])
    assert fr["last_night"] == "2026-09-02"


def test_ztf_is_live_when_any_public_endpoint_is_current():
    now = 61290.0
    st = roc.ztf_status({"archive": {"frontier_mjd": now - 2.0},
                         "alerce_ztf": {"frontier_mjd": None},
                         "antares": {"frontier_mjd": now - 40.0}}, now=now)
    assert st["status"] == "LIVE"
    assert st["newest_source"] == "archive"
    assert st["days_behind_now"] == 2.0
    assert st["by_source"]["alerce_ztf"] is None


def test_ztf_dark_names_the_archive_when_it_was_reached():
    now = 61290.0
    st = roc.ztf_status({"archive": {"frontier_mjd": now - 60.0},
                         "antares": {"frontier_mjd": now - 65.0}}, now=now)
    assert st["status"] == "DARK_OR_UNSERVED"
    assert "own nightly alert archive" in st["why"]


def test_ztf_dark_from_brokers_alone_is_hedged():
    now = 61290.0
    st = roc.ztf_status({"archive": {"frontier_mjd": None, "error": "timeout"},
                         "antares": {"frontier_mjd": now - 65.0}}, now=now)
    assert st["status"] == "DARK_OR_UNSERVED"
    assert "not established" in st["why"]


def test_nothing_reached_is_unreached_not_dark():
    st = roc.ztf_status({"archive": {"frontier_mjd": None}}, now=61290.0)
    assert st["status"] == "UNREACHED"
    assert st["newest_mjd"] is None


def test_the_ztf_block_rides_along_with_the_rubin_verdict_without_changing_it():
    now = roc._now_mjd()
    ztf = {"status": roc.ztf_status({"archive": {"frontier_mjd": now - 1.0}}, now=now)}
    d = roc.decide({"frontier_mjd": FROZEN, "newest_any_survey_mjd": None},
                   {"frontier_mjd": FROZEN + 0.1, "ztf_control": {"frontier_mjd": None}},
                   {"frontier_mjd": None}, ztf)
    assert d["verdict"] == "SKY_STOPPED"
    assert d["ztf"]["status"] == "LIVE"
    assert "ztf_archive" in d["live_controls"]
    assert "Corroborated" in d["why"]
    # Without the block the decision is exactly what it was.
    d0 = roc.decide({"frontier_mjd": FROZEN, "newest_any_survey_mjd": None},
                    {"frontier_mjd": FROZEN + 0.1, "ztf_control": {"frontier_mjd": None}},
                    {"frontier_mjd": None})
    assert "ztf" not in d0 and d0["live_controls"] == []


def test_the_antares_and_alerce_shapes_are_read_by_the_scanner():
    # JSON:API from ANTARES, and ALeRCE's ZTF object page: neither key is
    # hard-coded, both are found by the shape-agnostic scanner.
    antares = {"data": [{"type": "locus", "attributes": {"properties": {
        "newest_alert_observation_time": 61280.25, "oldest_alert_observation_time": 58300.1,
        "num_alerts": 40}}}]}
    alerce = {"items": [{"oid": "ZTF26aaaaaaa", "firstmjd": 61200.1, "lastmjd": 61281.3,
                         "deltajd": 81.2, "ndet": 12}]}
    assert roc._max_time_in(antares) == 61280.25
    assert roc._max_time_in(alerce) == 61281.3
