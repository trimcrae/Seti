"""Offline tests for TOCSIN on the live ZTF stream (``seti.tocsin.ztf_live``).

No network.  The two services are documentation-derived until the runner's probe
records them, so what is pinned here is what can go wrong *given* a response of
the documented shape: the sign convention, the flux units, the quiescent flux
recovered from ``magpsf_corr``, the object-to-target match, the point-in-quadrant
denominator, the window/frontier logic, and the rule that decides which of a
matched object's historical events are folded.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from seti.tocsin import ztf_live as Z
from seti.tocsin.photometry import ab_to_njy, njy_to_ab

MJD0 = 61250.0


def _targets(n=3, ra0=150.0, dec0=30.0, pm=0.0, g=15.0):
    rows = []
    for k in range(n):
        rows.append({"source_id": f"t{k}", "ra": ra0 + 0.3 * k, "dec": dec0,
                     "ra_error": 0.02, "dec_error": 0.02, "pmra": pm, "pmdec": 0.0,
                     "pmra_error": 0.05, "pmdec_error": 0.05, "parallax": 25.0,
                     "phot_g_mean_mag": g - 0.4, "bp_rp": 2.4,
                     "u_sdss_mag": g + 1.6, "g_sdss_mag": g, "r_sdss_mag": g - 0.8,
                     "i_sdss_mag": g - 1.1, "z_sdss_mag": g - 1.3, "y_ps1_mag": g - 1.45})
    return pd.DataFrame(rows)


def _det(mjd, fid=2, magpsf=17.0, sigmapsf=0.05, isdiffpos="t", magpsf_corr=14.2,
         corrected=True, drb=0.95, rb=0.8, candid=None, ra=150.0, dec=30.0, dubious=False):
    return {"mjd": mjd, "fid": fid, "magpsf": magpsf, "sigmapsf": sigmapsf,
            "isdiffpos": isdiffpos, "magpsf_corr": magpsf_corr,
            "sigmapsf_corr": 0.01, "sigmapsf_corr_ext": 0.012, "corrected": corrected,
            "drb": drb, "rb": rb, "candid": candid or int(mjd * 1e6), "ra": ra, "dec": dec,
            "distnr": 0.3, "diffmaglim": 20.4, "dubious": dubious}


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
def test_isdiffpos_carries_the_sign_and_the_flux_is_ab_njy():
    alerts = Z.normalize_alerce_ztf_detections(
        "ZTF26aaaaaaa", [_det(MJD0, isdiffpos="t"), _det(MJD0 + 1, isdiffpos="f"),
                         _det(MJD0 + 2, isdiffpos=1), _det(MJD0 + 3, isdiffpos="-1")])
    assert [a.polarity for a in alerts] == ["flash", "dip", "flash", "dip"]
    assert alerts[0].dflux_njy == pytest.approx(float(ab_to_njy(17.0)))
    assert alerts[1].dflux_njy == pytest.approx(-float(ab_to_njy(17.0)))
    assert alerts[1].is_negative is True
    # sigma_mag -> flux error: dF = F * sigma * ln10/2.5
    assert alerts[0].dflux_err_njy == pytest.approx(alerts[0].dflux_njy * 0.05 * math.log(10) / 2.5)
    assert alerts[0].snr == pytest.approx(2.5 / (0.05 * math.log(10)))
    assert alerts[0].band == "r" and alerts[0].broker == "alerce-ztf"


def test_the_quiescent_flux_is_the_reference_recovered_from_magpsf_corr():
    a = Z.normalize_alerce_ztf_detections("o", [_det(MJD0, magpsf=17.0, magpsf_corr=14.2)])[0]
    total = float(ab_to_njy(14.2))
    assert a.template_flux_njy == pytest.approx(total - float(ab_to_njy(17.0)))
    assert a.template_flux_err_njy is not None and a.template_flux_err_njy > 0
    # A dip: total = ref - |dF|, so ref = total + |dF|.
    d = Z.normalize_alerce_ztf_detections("o", [_det(MJD0, isdiffpos="f", magpsf_corr=14.2)])[0]
    assert d.template_flux_njy == pytest.approx(total + float(ab_to_njy(17.0)))


def test_an_uncorrected_detection_leaves_the_baseline_to_the_gspc_fallback():
    a = Z.normalize_alerce_ztf_detections("o", [_det(MJD0, corrected=False)])[0]
    assert a.template_flux_njy is None
    b = Z.normalize_alerce_ztf_detections("o", [_det(MJD0, corrected="f")])[0]
    assert b.template_flux_njy is None
    c = Z.normalize_alerce_ztf_detections("o", [_det(MJD0, magpsf_corr=None)])[0]
    assert c.template_flux_njy is None


def test_drb_is_the_reliability_and_rb_the_fallback():
    a = Z.normalize_alerce_ztf_detections("o", [_det(MJD0, drb=0.97, rb=0.6)])[0]
    assert a.reliability == 0.97 and a.reliability_version == "drb"
    b = Z.normalize_alerce_ztf_detections("o", [_det(MJD0, drb=None, rb=0.6)])[0]
    assert b.reliability == 0.6 and b.reliability_version == "rb"
    c = Z.normalize_alerce_ztf_detections("o", [_det(MJD0, drb=None, rb=None)])[0]
    assert c.reliability is None and c.reliability_version is None


def test_dubious_is_the_pixel_flag_stand_in():
    a = Z.normalize_alerce_ztf_detections("o", [_det(MJD0, dubious=True)])[0]
    assert a.pixel_flag_bad is True
    b = Z.normalize_alerce_ztf_detections("o", [_det(MJD0, dubious=None)])[0]
    assert b.pixel_flag_bad is None


def test_unknown_filters_and_incomplete_rows_are_dropped_not_guessed():
    rows = [_det(MJD0, fid=9), {**_det(MJD0), "magpsf": None}, {**_det(MJD0), "isdiffpos": "?"},
            _det(MJD0, fid=1)]
    alerts = Z.normalize_alerce_ztf_detections("o", rows)
    assert len(alerts) == 1 and alerts[0].band == "g"


def test_upper_limits_carry_epoch_band_and_limit():
    ul = Z.upper_limits([{"mjd": MJD0, "fid": 1, "diffmaglim": 20.1},
                         {"mjd": MJD0 + 1, "fid": 2, "diffmaglim": None},
                         {"mjd": None, "fid": 2, "diffmaglim": 20.0}])
    assert ul[0] == (MJD0, "g", 20.1)
    assert ul[1][1] == "r" and math.isnan(ul[1][2])
    assert len(ul) == 2


# ---------------------------------------------------------------------------
# matching objects to targets
# ---------------------------------------------------------------------------def_th = None


def _th():
    from seti.tocsin.run import DEFAULTS, _thresholds
    return _thresholds(DEFAULTS)


def test_objects_on_targets_are_matched_and_others_are_not():
    t = _targets(n=3)
    objs = [{"oid": "on0", "meanra": 150.0, "meandec": 30.0},
            {"oid": "on2", "meanra": 150.6 + 0.3 / 3600.0, "meandec": 30.0},
            {"oid": "off", "meanra": 151.5, "meandec": 30.0},
            {"oid": "nan", "meanra": None, "meandec": 30.0}]
    m = Z.match_objects(objs, t, _th(), epoch_jyear=2026.6)
    assert m == {"on0": "t0", "on2": "t2"}


def test_matching_follows_the_propagated_position_of_a_fast_mover():
    # 2 arcsec/yr for ~10.6 yr since the Gaia epoch: the star has moved ~21".
    t = _targets(n=1, pm=2000.0)
    epoch = 2026.6
    from seti.tocsin.targets import propagate_pm
    p_ra, p_dec = propagate_pm(np.array([150.0]), np.array([30.0]), np.array([2000.0]),
                               np.array([0.0]), to_epoch=epoch)
    at_catalogue = [{"oid": "cat", "meanra": 150.0, "meandec": 30.0}]
    at_now = [{"oid": "now", "meanra": float(p_ra[0]), "meandec": float(p_dec[0])}]
    assert Z.match_objects(at_catalogue, t, _th(), epoch) == {}
    assert Z.match_objects(at_now, t, _th(), epoch) == {"now": "t0"}


# ---------------------------------------------------------------------------
# the quadrant footprint
# ---------------------------------------------------------------------------
def _quad(ra_c, dec_c, half=0.43, obsjd=MJD0 + 2400000.5 + 0.3, fid=2, maglimit=20.5):
    return {"obsjd": obsjd, "fid": fid, "field": 1, "ccdid": 1, "qid": 1,
            "ra": ra_c, "dec": dec_c,
            "ra1": ra_c - half, "dec1": dec_c - half, "ra2": ra_c + half, "dec2": dec_c - half,
            "ra3": ra_c + half, "dec3": dec_c + half, "ra4": ra_c - half, "dec4": dec_c + half,
            "maglimit": maglimit, "exptime": 30.0, "ipac_gid": 1}


def test_a_quadrant_covering_a_star_is_a_trial_with_its_own_limit():
    t = _targets(n=3)                       # t0 at 150.0, t1 at 150.3, t2 at 150.6
    exp = [_quad(150.1, 30.0), _quad(150.1, 30.0, obsjd=MJD0 + 2400000.5 + 0.34, fid=1,
                                     maglimit=20.9)]
    pairs, bands, limits, epochs, stats = Z.quadrant_footprint(exp, t, _th(), 2026.6)
    night = Z.night_id(MJD0 + 0.3)
    assert pairs == {("t0", night), ("t1", night)}
    assert bands[("t0", night)] == {"r", "g"}
    assert limits[("t0", night, "r")] == [20.5] and limits[("t0", night, "g")] == [20.9]
    assert len(epochs["t0"]) == 2 and "t2" not in epochs
    assert stats["quadrants"] == 2 and stats["footprint_star_nights"] == 2


def test_the_polygon_test_is_an_edge_test_not_a_radius():
    # A star 0.5 degrees from the centre along the diagonal is OUTSIDE a
    # 0.43-degree half-width square only if the corner is honoured... it is at
    # (0.35, 0.35): inside.  At (0.5, 0.0) it is outside the square but inside
    # the 0.75-degree candidate radius.
    t = pd.DataFrame({"source_id": ["in", "out"], "ra": [150.35, 150.5], "dec": [30.35, 30.0],
                      "pmra": [0.0, 0.0], "pmdec": [0.0, 0.0],
                      "pmra_error": [0.0, 0.0], "pmdec_error": [0.0, 0.0]})
    # ra is scaled by cos(dec) on the sky; build the quadrant in projected terms
    exp = [_quad(150.0, 30.0, half=0.43)]
    pairs, *_ = Z.quadrant_footprint(exp, t, _th(), 2026.6)
    tids = {p[0] for p in pairs}
    assert "out" not in tids


def test_an_unparsable_exposure_row_is_counted_and_skipped():
    t = _targets(n=1)
    exp = [{"obsjd": "bad"}, _quad(150.0, 30.0)]
    pairs, _b, _l, _e, stats = Z.quadrant_footprint(exp, t, _th(), 2026.6)
    assert stats["quadrant_rows_unparsable"] == 1 and len(pairs) == 1


# ---------------------------------------------------------------------------
# the objects sweep against a fake API
# ---------------------------------------------------------------------------
class FakeSession:
    def __init__(self, pages: dict, statuses: dict | None = None, first_slice_only=True):
        self.pages = pages          # page number -> items (served for the FIRST slice)
        self.statuses = statuses or {}
        self.requests: list = []
        self.first_slice_only = first_slice_only
        self._first_lo = None

    def get(self, url, params=None):
        self.requests.append((url, list(params) if params is not None else None))
        pmap = dict(params or [])
        page = int(pmap.get("page", 1))
        los = [v for k, v in (params or []) if k == "lastmjd"]
        if los:
            if self._first_lo is None:
                self._first_lo = los[0]
            if self.first_slice_only and los[0] != self._first_lo:
                class Empty:
                    status_code = 200
                    headers: dict = {}
                    text = ""

                    def json(self):
                        return {"items": [], "has_next": False, "page": None}
                return Empty()

        class R:
            status_code = 200
            headers: dict = {}
            text = ""

            def __init__(self, payload, status=200):
                self._p = payload
                self.status_code = status

            def json(self):
                return self._p

        if page in self.statuses:
            code = self.statuses.pop(page)
            return R({}, code)
        items = self.pages.get(page, [])
        # As measured with count=false: the service never numbers the page and
        # never says has_next; only a short page ends the walk.
        return R({"items": items, "has_next": False, "page": None, "total": None})


def test_the_sweep_passes_the_window_as_a_repeated_lastmjd_and_pages_until_a_short_page():
    """The first live run: `count=false` makes has_next always false, so the walk
    took one page of 1000 and stopped.  Only a short page ends it now."""
    api = Z.AlerceZtfAPI(sleep=lambda s: None, page_size=2, slice_days=1.0)
    api._s = FakeSession({1: [{"oid": "a"}, {"oid": "a2"}], 2: [{"oid": "b"}, {"oid": "b2"}],
                          3: [{"oid": "c"}]})
    rows, stats = api.objects_in_window(61000.0, 61001.0)
    assert [r["oid"] for r in rows] == ["a", "a2", "b", "b2", "c"]
    assert stats["pages"] == 3 and stats["slices"] == 1 and stats["truncated"] is False
    url, params = api._s.requests[0]
    assert url.endswith("/objects")
    assert [v for k, v in params if k == "lastmjd"] == ["61000.000000", "61001.000000"]
    assert ("count", "false") in params


def test_the_sweep_is_sliced_so_no_request_sits_deep_in_the_offset():
    """Run 33942793097: one three-night query, 27 minutes, HTTP 500."""
    api = Z.AlerceZtfAPI(sleep=lambda s: None, page_size=10, slice_days=0.5)
    api._s = FakeSession({1: [{"oid": "a"}]}, first_slice_only=False)
    rows, stats = api.objects_in_window(61000.0, 61002.0)
    assert stats["slices"] == 4 and stats["pages"] == 4
    los = [[v for k, v in p if k == "lastmjd"] for _u, p in api._s.requests]
    assert los[0] == ["61000.000000", "61000.500000"]
    assert los[-1] == ["61001.500000", "61002.000000"]


def test_a_slice_that_fails_after_retries_truncates_the_sweep_with_the_error():
    api = Z.AlerceZtfAPI(sleep=lambda s: None, page_size=10, slice_days=0.5)
    api._s = FakeSession({1: [{"oid": "a"}]}, statuses={1: 500}, first_slice_only=False)
    api.RETRY_WAITS_S = ()                              # no retries: fail at once
    rows, stats = api.objects_in_window(61000.0, 61001.0)
    assert stats["truncated"] is True and "500" in stats["error"]
    assert stats["slices"] == 1                          # stopped at the failing slice
    assert any("failed after retries" in n for n in api.notes)


def test_a_numbered_page_with_has_next_is_still_honoured():
    api = Z.AlerceZtfAPI(sleep=lambda s: None, page_size=5, slice_days=1.0)

    class Numbered(FakeSession):
        def get(self, url, params=None):
            r = super().get(url, params)
            page = int(dict(params or []).get("page", 1))
            r._p = {"items": self.pages.get(page, []), "has_next": page + 1 in self.pages,
                    "page": page}
            return r

    api._s = Numbered({1: [{"oid": "a"}], 2: [{"oid": "b"}]})
    rows, stats = api.objects_in_window(61000.0, 61001.0)
    assert [r["oid"] for r in rows] == ["a", "b"] and stats["pages"] == 2


def test_a_transient_server_error_is_retried_and_a_deadline_truncates():
    api = Z.AlerceZtfAPI(sleep=lambda s: None, page_size=1, slice_days=1.0)
    api._s = FakeSession({1: [{"oid": "a"}], 2: [{"oid": "b"}]}, statuses={1: 503})
    rows, stats = api.objects_in_window(61000.0, 61001.0)
    assert [r["oid"] for r in rows] == ["a", "b"]
    api2 = Z.AlerceZtfAPI(sleep=lambda s: None, page_size=1, slice_days=1.0)
    api2._s = FakeSession({1: [{"oid": "a"}], 2: [{"oid": "b"}]})
    import time as _t
    rows2, stats2 = api2.objects_in_window(61000.0, 61001.0, deadline=_t.monotonic() - 1)
    assert rows2 == [] and stats2["truncated"] is True


# ---------------------------------------------------------------------------
# the whole window, offline
# ---------------------------------------------------------------------------
class FakeApi:
    def __init__(self, objects, dets, nondets, frontier=None):
        self.objects = objects
        self.dets = dets
        self.nondets = nondets
        self._frontier = frontier
        self.calls = 0
        self.notes: list[str] = []

    def objects_in_window(self, lo, hi, deadline=None, max_pages=None):
        return [o for o in self.objects if lo <= o["lastmjd"] < hi], {"pages": 1,
                                                                        "truncated": False}

    def detections(self, oid):
        return self.dets.get(oid, [])

    def non_detections(self, oid):
        return self.nondets.get(oid, [])

    def frontier(self):
        return self._frontier


class FakeIrsa:
    def __init__(self, exposures, frontier):
        self._exp = exposures
        self._frontier = frontier
        self.calls = 0
        self.notes: list[str] = []

    def exposures(self, lo, hi, public_gid=1):
        return [e for e in self._exp if lo <= e["obsjd"] - 2400000.5 < hi]

    def frontier(self, public_gid=1):
        return self._frontier


def _prepare(tmp_path):
    t = _targets(n=3, g=15.0)
    tp = tmp_path / "targets.parquet"
    t.to_parquet(tp)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "tocsin.yaml").write_text("ztf:\n  backfill_start_mjd: 61250.0\n"
                                                     "  max_nights_per_run: 1.0\n")

    class Cfg:
        root = tmp_path
    return Cfg(), tp


def _nights_of_exposures(lo_mjd, n_nights, ra=150.1, dec=30.0):
    exp = []
    for k in range(n_nights):
        for j, fid in ((0.30, 2), (0.34, 1)):
            exp.append(_quad(ra, dec, obsjd=lo_mjd + k + j + 2400000.5, fid=fid))
    return exp


def test_a_window_screens_folds_and_advances_the_watermark(tmp_path):
    cfg, tp = _prepare(tmp_path)
    night_mjd = MJD0 + 0.3
    # Object on t0, alerted tonight, a grey-ish flash in g and r 30 min apart,
    # plus a long history of upper limits.
    dets = {"ZTFa": [_det(night_mjd, fid=1, magpsf=16.5, magpsf_corr=14.9, ra=150.0),
                     _det(night_mjd + 0.02, fid=2, magpsf=16.0, magpsf_corr=14.2, ra=150.0)]}
    nd = {"ZTFa": [{"mjd": MJD0 - 10 + k, "fid": 2, "diffmaglim": 20.3} for k in range(8)]}
    api = FakeApi([{"oid": "ZTFa", "meanra": 150.0, "meandec": 30.0, "lastmjd": night_mjd + 0.02},
                   {"oid": "ZTFfar", "meanra": 10.0, "meandec": 10.0, "lastmjd": night_mjd}],
                  dets, nd, frontier=MJD0 + 5.0)
    irsa = FakeIrsa(_nights_of_exposures(MJD0, 3), frontier=MJD0 + 2.9)
    out = tmp_path / "out"
    rec = Z.screen_window(cfg, targets_path=tp, out_dir=out, api=api, irsa=irsa)
    assert rec["verdict"] == "OK"
    assert rec["mjd_lo"] == 61250.0 and rec["mjd_hi"] == 61251.0        # one night
    assert rec["counts"]["objects_in_window"] == 2
    assert rec["counts"]["objects_matched"] == 1
    assert rec["counts"]["detections_pulled"] == 2
    assert rec["counts"]["upper_limits_pulled"] == 8
    # Two stars under the quadrant tonight = two trials; the one that alerted
    # is among them, so the denominator is the footprint.
    assert rec["counts"]["target_nights_from_footprint"] == 2
    assert rec["counts"]["target_nights_screened"] == 2
    assert rec["denominator"] in ("quadrant_footprint_exact", "detection_dominated_lower_bound")
    assert rec["watermark_mjd"] == 61251.0
    led = json.loads((out / "ledger.json").read_text())
    assert led["n_target_visits"] == 2
    assert led["last_mjd_screened"] == 61251.0
    if led["targets"]:
        t0 = led["targets"]["t0"]
        assert t0["visits_exact"] is True
        # Upper limits and quadrant epochs both reach the visit history.
        assert t0["n_visits"] >= 8
    assert (out / "summary.json").exists() and (out / "watchlist.csv").exists()

    # Second run: the next night, capped by the exposure-table frontier.
    rec2 = Z.screen_window(cfg, targets_path=tp, out_dir=out, api=api, irsa=irsa)
    assert rec2["mjd_lo"] == 61251.0 and rec2["mjd_hi"] == 61252.0
    rec3 = Z.screen_window(cfg, targets_path=tp, out_dir=out, api=api, irsa=irsa)
    assert rec3["mjd_hi"] == pytest.approx(MJD0 + 2.9)      # the frontier, not a full night
    rec4 = Z.screen_window(cfg, targets_path=tp, out_dir=out, api=api, irsa=irsa)
    assert rec4["verdict"] == "NO_NEW_DATA"


def test_an_event_on_an_unscreened_earlier_night_is_deferred_to_the_sweep(tmp_path):
    """The backfill rule: fold only what has trials in the ledger."""
    cfg, tp = _prepare(tmp_path)
    old = MJD0 - 30 + 0.3                     # a night the sweep has not reached
    now_night = MJD0 + 0.3
    dets = {"ZTFa": [_det(old, fid=2, magpsf=16.0, magpsf_corr=14.2, ra=150.0),
                     _det(now_night, fid=2, magpsf=16.0, magpsf_corr=14.2, ra=150.0)]}
    api = FakeApi([{"oid": "ZTFa", "meanra": 150.0, "meandec": 30.0, "lastmjd": now_night}],
                  dets, {}, frontier=MJD0 + 5.0)
    irsa = FakeIrsa(_nights_of_exposures(MJD0, 1), frontier=MJD0 + 5.0)
    rec = Z.screen_window(cfg, targets_path=tp, out_dir=tmp_path / "out", api=api, irsa=irsa)
    assert rec["counts"]["detections_pulled"] == 2
    kept = rec["counts"]["events_kept"]
    folded = rec["counts"]["events_folded"]
    assert kept - folded == rec["counts"]["events_deferred_to_sweep"]
    if kept:
        assert folded < kept, "the old night's event has no trials yet and must wait"


def test_a_truncated_sweep_folds_nothing_and_keeps_the_watermark(tmp_path):
    cfg, tp = _prepare(tmp_path)

    class Truncating(FakeApi):
        def objects_in_window(self, lo, hi, deadline=None, max_pages=None):
            return [], {"pages": 7, "slices": 2, "truncated": True,
                        "error": "GET objects: HTTP 500"}

    api = Truncating([], {}, {}, frontier=MJD0 + 5.0)
    irsa = FakeIrsa(_nights_of_exposures(MJD0, 1), frontier=MJD0 + 5.0)
    out = tmp_path / "out"
    rec = Z.screen_window(cfg, targets_path=tp, out_dir=out, api=api, irsa=irsa)
    assert rec["verdict"] == "SWEEP_TRUNCATED"
    assert rec["watermark_mjd"] is None
    assert "500" in rec["error"]
    assert not (out / "ledger.json").exists()
    assert any("TRUNCATED" in n for n in rec["notes"])


def test_an_unreachable_api_is_a_named_verdict(tmp_path):
    cfg, tp = _prepare(tmp_path)

    class Dead(FakeApi):
        def objects_in_window(self, lo, hi, deadline=None, max_pages=None):
            raise Z.ZtfLiveError("GET objects: HTTP 502")

    rec = Z.screen_window(cfg, targets_path=tp, out_dir=tmp_path / "out",
                          api=Dead([], {}, {}, frontier=MJD0 + 5.0),
                          irsa=FakeIrsa([], frontier=MJD0 + 5.0))
    assert rec["verdict"] == "NO_DATA_REACHED" and "502" in rec["error"]


def test_the_window_is_capped_by_the_exposure_table_frontier_not_the_clock(tmp_path):
    cfg, tp = _prepare(tmp_path)
    api = FakeApi([], {}, {}, frontier=MJD0 + 100.0)
    irsa = FakeIrsa([], frontier=MJD0 + 0.4)
    rec = Z.screen_window(cfg, targets_path=tp, out_dir=tmp_path / "out", api=api, irsa=irsa)
    assert rec["mjd_hi"] == pytest.approx(MJD0 + 0.4)
    assert rec["frontiers"]["irsa_exposures_mjd"] == MJD0 + 0.4


def test_the_config_block_overrides_the_defaults(tmp_path):
    cfg, _tp = _prepare(tmp_path)
    _conf, z = Z.ztf_config(cfg)
    assert z["backfill_start_mjd"] == 61250.0 and z["max_nights_per_run"] == 1.0
    assert z["public_gid"] == Z.DEFAULTS["public_gid"]


def test_the_repository_config_block_matches_the_module_defaults():
    from pathlib import Path

    import yaml
    doc = yaml.safe_load(Path("config/tocsin.yaml").read_text())
    z = doc.get("ztf") or {}
    for k, v in z.items():
        assert k in Z.DEFAULTS, f"config/tocsin.yaml ztf.{k} is not a known setting"
        assert v == Z.DEFAULTS[k], f"ztf.{k}: yaml {v!r} != DEFAULTS {Z.DEFAULTS[k]!r}"


def test_ab_round_trip_used_by_the_normaliser():
    assert float(njy_to_ab(float(ab_to_njy(17.0)))) == pytest.approx(17.0)


def test_ztf_detections_carry_the_measured_astrometric_floor():
    a = Z.normalize_alerce_ztf_detections("o", [_det(MJD0)])[0]
    assert a.ra_err_arcsec == Z.ZTF_ASTROMETRIC_FLOOR_ARCSEC
    # 1 arcsec is the 3-sigma line, so a 0.5 arcsec residual centroid passes.
    assert 0.5 / a.pos_err_arcsec < 3.0 < 1.1 / a.pos_err_arcsec


def test_the_votable_error_message_is_extracted_not_the_document_head():
    doc = ("<?xml version=\"1.0\"?><VOTABLE><RESOURCE type=\"results\"><INFO name=\"QUERY_STATUS\" "
           "value=\"ERROR\">Column programid not found in table ztf.ztf_current_meta_sci"
           "</INFO></RESOURCE></VOTABLE>")
    assert Z.votable_error(doc) == "Column programid not found in table ztf.ztf_current_meta_sci"
    assert Z.votable_error("obsjd,fid\n1,2\n") is None


def test_the_exposure_query_asks_for_ipac_gid_one_night_at_a_time():
    seen = []

    class Sess:
        def get(self, url, params=None):
            seen.append(params["QUERY"])

            class R:
                status_code = 200
                text = "obsjd,fid\n"
            return R()

    irsa = Z.IrsaZtfExposures()
    irsa._s = Sess()
    irsa.exposures(61000.0, 61002.5, 1)
    assert len(seen) == 3
    assert all("ipac_gid = 1" in q and "programid" not in q for q in seen)
    assert "obsjd >= 2461000.500000" in seen[0] and "obsjd < 2461001.500000" in seen[0]


def test_a_failed_footprint_folds_nothing_and_keeps_the_watermark(tmp_path):
    """The first live run folded four empty nights when IRSA errored."""
    cfg, tp = _prepare(tmp_path)

    class BrokenIrsa(FakeIrsa):
        def exposures(self, lo, hi, public_gid=1):
            raise Z.ZtfLiveError("IRSA TAP error: Column programid not found")

    api = FakeApi([], {}, {}, frontier=MJD0 + 5.0)
    out = tmp_path / "out"
    rec = Z.screen_window(cfg, targets_path=tp, out_dir=out, api=api,
                          irsa=BrokenIrsa([], frontier=MJD0 + 5.0))
    assert rec["verdict"] == "NO_DENOMINATOR"
    assert rec["watermark_mjd"] is None
    assert not (out / "ledger.json").exists()
    assert any("programid" in n for n in rec["notes"])


def test_the_frontier_query_is_bounded_and_widens_only_when_empty():
    seen = []

    class Sess:
        def get(self, url, params=None):
            seen.append(params["QUERY"])

            class R:
                status_code = 200
                text = ("obsjd_max\n\n" if len(seen) < 3 else
                        f"obsjd_max\n{61280.0 + 2400000.5}\n")
            return R()

    irsa = Z.IrsaZtfExposures()
    irsa._s = Sess()
    fr = irsa.frontier(1, now=61290.0)
    assert fr == pytest.approx(61280.0)
    assert len(seen) == 3
    assert all("obsjd >" in q and "ipac_gid = 1" in q for q in seen)
    # Never an unbounded MAX over the whole table.
    assert all("WHERE obsjd >" in q for q in seen)
