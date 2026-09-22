"""Offline test suite for CENTURY --- cessation, fade and rising scatter on
plates, with the Menzel gap.  No network anywhere.

The three tests that decide the channel:

* ``test_injected_cessation_is_recovered`` --- a period that stops in 1931 on
  a synthetic plate light curve (per-plate limits, exposure smear, the gap)
  is found, with the transition in the right decade.
* ``test_menzel_step_alone_is_not_a_fade`` --- a constant star with a 0.15 mag
  pre/post-gap offset: the naive line calls it a century fade; the step model
  does not.
* ``test_empty_api_gives_no_data_reached`` --- an assess over shards that
  fetched nothing says NO_DATA_REACHED, never a sky statement.

Plus a case for every rejection rule.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from seti.century.api import to_frame
from seti.century.cease import (
    analyze_century,
    censored_efficiency,
    detect_window,
    refine_reference,
    window_freqs,
)
from seti.century.fade import analyze_fade
from seti.century.flags import flagdefs_from_source, resolve_flagdefs
from seti.century.lightcurve import (
    CenturyLC,
    annual_table,
    calendar_blocks,
    from_api_frame,
    smear_factor,
    year_to_mjd,
)
from seti.century.run import (
    DEFAULTS,
    field_common_mode,
    load_shard_lcs,
    parse_shard,
    screen_star,
    stage_assess,
    stage_screen,
)
from seti.century.scatter import analyze_scatter, residualise_periodic
from seti.century.step import fit_step_slope, segment
from seti.century.vet import is_lpv_type, is_periodic_type, vet_row

GAP = (1954.0, 1970.0)


# ---------------------------------------------------------------------------
# Synthetic plate archive
# ---------------------------------------------------------------------------


def synth_plates(rng, *, n_per_year=25, years=(1890, 1992), gap=GAP, mag0=11.0, err=0.12,
                 lim_mean=13.5, lim_sd=0.8, period=0.0, amp=0.0, stop_year=None,
                 step=0.0, slope_per_century=0.0, scatter_growth_mmag_per_century=0.0,
                 exptime_min=45.0, series_switch_year=1930.0, blend_frac=0.0,
                 post_gap_lim_shift=0.0, post_gap_err_scale=1.0) -> CenturyLC:
    """A star on synthetic Harvard plates: per-plate limits, exposure smear, the
    Menzel gap, a series change, optional cessation / step / fade / growth."""
    ys = []
    for y in range(int(years[0]), int(years[1])):
        if gap[0] <= y < gap[1]:
            continue
        ys.append(y + rng.uniform(0.0, 1.0, size=int(n_per_year)))
    yr = np.sort(np.concatenate(ys))
    n = yr.size
    t = year_to_mjd(yr)
    post = yr >= gap[1]
    lim = rng.normal(lim_mean, lim_sd, size=n) + np.where(post, post_gap_lim_shift, 0.0)
    e = np.full(n, err) * np.where(post, post_gap_err_scale, 1.0)
    exp_ = np.full(n, float(exptime_min))
    base = mag0 + slope_per_century * (yr - years[0]) / 100.0 + np.where(post, step, 0.0)
    sig = np.zeros(n)
    if period > 0 and amp > 0:
        f = 1.0 / period
        on = np.ones(n, dtype=bool) if stop_year is None else (yr < stop_year)
        sig = amp * smear_factor(f, exp_) * np.sin(2 * np.pi * f * t + 0.3) * on
    grow = (scatter_growth_mmag_per_century * 1e-3) * (yr - years[0]) / 100.0
    noise = rng.normal(0.0, 1.0, size=n) * np.sqrt(e ** 2 + grow ** 2)
    mag = base + sig + noise
    det = mag <= lim
    series = np.where(yr < series_switch_year, "a", np.where(post, "dnb", "mc")).astype(object)
    blend = rng.random(n) < blend_frac
    lc = CenturyLC(
        t=t[det], mag=mag[det], err=e[det], lim=lim[det], series=series[det],
        exptime_min=exp_[det], blend=blend[det], reject=np.zeros(int(det.sum()), bool),
        plate=np.array([f"{s}_{i}" for i, s in enumerate(series[det])], dtype=object),
        t_nd=t[~det], lim_nd=lim[~det], series_nd=series[~det], exptime_nd=exp_[~det],
        flags_applied=True, n_raw=n, columns=["synthetic"], exptime_unit="minutes",
    )
    return lc


def _conf():
    c = json.loads(json.dumps(DEFAULTS))
    c["cease"].update({"n_trials": 60, "n_null_trial": 50, "n_null": 100, "pdm_null": 50,
                       "n_trials_confirm": 80, "n_null_trial_confirm": 60})
    return c


def _cess_kwargs(**over):
    kw = dict(block_years=2.0, min_epochs_block=15, min_blocks=4, n_null=100, n_trials=60,
              n_null_trial=50, pdm_null=50, gap=GAP)
    kw.update(over)
    return kw


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_step_segments_and_fit_recover_a_pure_step():
    rng = np.random.default_rng(1)
    yr = np.array([y + 0.5 for y in range(1890, 1954)] + [y + 0.5 for y in range(1970, 1992)])
    q = 11.0 + np.where(yr > 1960, 0.15, 0.0) + rng.normal(0, 0.01, yr.size)
    fit = fit_step_slope(yr, q, np.full(yr.size, 0.01), gap=GAP)
    assert fit.ok and fit.step_identifiable
    assert abs(fit.step - 0.15) < 0.02
    assert abs(fit.slope) * 100 < 0.02          # < 0.02 mag / century
    assert fit.slope_naive * 100 > 0.08         # the naive line calls it a fade
    assert (segment(np.array([1900, 1960, 1980]), GAP) == np.array([0, -1, 1])).all()


def test_step_fit_single_segment_is_honest():
    yr = np.arange(1890, 1950) + 0.5
    fit = fit_step_slope(yr, 11 + 0.001 * (yr - 1890), None, gap=GAP)
    assert fit.ok and not fit.step_identifiable and fit.step == 0.0
    assert abs(fit.slope - fit.slope_naive) < 1e-12


def test_window_freqs_and_smear():
    f = window_freqs(2.0, 730.0, half_width_frac=0.005)
    assert f.min() < 2.0 < f.max() and f.max() > 3.9
    assert np.isclose(smear_factor(1.0, 0.0)[()] if np.ndim(smear_factor(1.0, 0.0)) == 0
                      else smear_factor(1.0, [0.0])[0], 1.0)
    s = smear_factor(1.0 / 0.25, [45.0])[0]     # 45 min exposure at P = 6 h
    assert 0.9 < s < 1.0
    s2 = smear_factor(1.0 / 0.05, [60.0])[0]    # 1 h exposure at P = 1.2 h
    assert s2 < 0.7
    assert not np.isnan(smear_factor(1.0, [np.nan])[0])


def test_calendar_blocks_carry_nondetections_and_drop_thin_blocks():
    rng = np.random.default_rng(2)
    lc = synth_plates(rng, n_per_year=12, lim_mean=11.6, mag0=11.0)
    blocks = calendar_blocks(lc, block_years=2.0, min_epochs_block=10, min_blocks=4)
    assert len(blocks) >= 4
    assert all(b.n >= 10 for b in blocks)
    assert sum(b.n_nd for b in blocks) > 0           # shallow plates -> censored points ride along
    assert all(b.year_lo < b.year_hi for b in blocks)
    assert not any(b.year_lo < GAP[0] < b.year_hi - 2.0 for b in blocks)


def test_from_api_frame_normalises_a_plausible_answer_and_censors():
    df = pd.DataFrame({
        "date_jd": [2415000.5, 2416000.5, 2417000.5, 2418000.5],
        "magcal_magdep": [11.2, None, 11.4, 0.0],
        "magcal_local_rms": [0.1, None, 0.12, None],
        "limiting_mag_local": [13.0, 10.5, 13.5, 10.9],
        "series": ["a", "a", "mc", "mc"], "platenum": [1, 2, 3, 4],
        "exptime": [60.0, 60.0, 30.0, 30.0], "aflags": [0, 0, 0, 0], "bflags": [0, 0, 0, 0],
    })
    lc = from_api_frame(df)
    assert lc is not None and lc.n_det == 2 and lc.n_nd == 2
    assert lc.exptime_unit == "minutes"
    assert abs(lc.t[0] - 15000.0) < 1e-6
    assert set(lc.series) == {"a", "mc"}
    assert not lc.flags_applied
    df2 = df.assign(exptime=[3600.0, 3600.0, 1800.0, 1800.0])
    assert from_api_frame(df2).exptime_unit == "seconds"
    assert abs(from_api_frame(df2).exptime_min[0] - 60.0) < 1e-9


def test_to_frame_accepts_columnar_and_row_json():
    a = to_frame({"x": [1, 2], "y": [3, 4]})
    b = to_frame([{"x": 1, "y": 3}, {"x": 2, "y": 4}])
    c = to_frame({"data": [{"x": 1}], "status": "ok"})
    assert list(a.columns) == ["x", "y"] and len(b) == 2 and len(c) == 1
    assert len(to_frame(None)) == 0 and len(to_frame("junk")) == 0


def test_to_frame_reads_dr7s_array_of_csv_lines():
    """DR7's real wire format, verified on the runner (run 35738717013).

    Read as plain JSON this is one column of strings called ``value``, and the
    first probe duly reported 18,429 plates with not one usable column.
    """
    from seti.century.lightcurve import any_time_to_year
    from seti.century.targets import normalise_refcat, plate_density  # noqa: F401

    cat = ["ref_text,ref_number,gsc_bin_index,ra_deg,dec_deg,pos_epoch,stdmag,color,"
           "pm_ra_masyr,pm_dec_masyr",
           "N030330195374,11030330195374,141878532,291.355448,42.79008,2000.0,15.54,-1.48,0,0",
           "N030330195375,11030330195375,141878533,291.366310,42.78435,2000.0,7.60,0.30,3,4"]
    df = to_frame(cat)
    assert list(df.columns)[:5] == ["ref_text", "ref_number", "gsc_bin_index", "ra_deg",
                                    "dec_deg"]
    n = normalise_refcat(df)
    assert len(n) == 2 and int(n["gsc_bin_index"].iloc[1]) == 141878533
    assert abs(float(n["mag_cat"].iloc[1]) - 7.60) < 1e-9
    assert abs(float(n["pm_total_masyr"].iloc[1]) - 5.0) < 1e-9

    exps = ["series,platenum,scannum,exptime,expdate,epoch,limMagApass",
            "a,1,0,45.0,1899-07-04T00:00:00,1899.50,14.2",
            "mc,2,0,60.0,1975-07-04T00:00:00,1975.50,15.1"]
    e = to_frame(exps)
    assert "epoch" in e.columns and "limMagApass" in e.columns
    yr = any_time_to_year(e["epoch"].to_numpy(dtype=float))
    assert abs(yr[0] - 1899.5) < 1e-6 and abs(yr[1] - 1975.5) < 1e-6
    # A wrapper around the same lines is unwrapped first.
    assert list(to_frame({"data": cat}).columns) == list(df.columns)


def test_any_time_to_year_tells_jd_mjd_and_year_apart():
    from seti.century.lightcurve import any_time_to_year, year_to_mjd

    mjd = year_to_mjd(np.array([1899.5, 1975.5]))
    assert np.allclose(any_time_to_year(mjd), [1899.5, 1975.5], atol=1e-6)
    assert np.allclose(any_time_to_year(mjd + 2400000.5), [1899.5, 1975.5], atol=1e-6)
    assert np.allclose(any_time_to_year(np.array([1899.5, 1975.5])), [1899.5, 1975.5])
    # Something that is none of the three is NaN, not a plausible wrong century.
    assert not np.isfinite(any_time_to_year(np.array([3.0, 4.0]))).any()


class _FakeResponse:
    def __init__(self, status: int, payload=None, text: str = ""):
        self.status_code = status
        self._payload = payload
        self.text = text or json.dumps(payload or {})

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _FakeSession:
    """Answers by (method, frozenset of payload keys); records every call."""

    def __init__(self, table: dict):
        self.table = table
        self.calls: list[tuple[str, tuple]] = []

    def _answer(self, method, payload):
        key = tuple(sorted(payload))
        self.calls.append((method, key))
        return self.table.get((method, key), _FakeResponse(422, {"detail": "unknown field"}))

    def post(self, url, json=None, **kw):          # noqa: A002
        return self._answer("POST", json or {})

    def get(self, url, params=None, **kw):
        return self._answer("GET", params or {})


def test_payload_variants_advance_on_a_validation_error():
    """A 422 names the wrong key, so the next spelling is tried, not abandoned."""
    from seti.century.api import querycat

    ok = _FakeResponse(200, {"ra_deg": [1.0], "dec_deg": [2.0], "gsc_bin_index": [7],
                             "ref_number": [3]})
    sess = _FakeSession({("POST", ("dec_deg", "ra_deg", "radius_deg", "refcat")): ok})
    r = querycat(1.0, 2.0, 30.0, refcat="apass", session=sess, retries=1, backoff_s=0.0)
    assert r.ok and r.n_rows == 1 and r.method == "POST"
    assert "radius_deg" in r.payload and len(sess.calls) == 3


def test_a_wrong_verb_retries_the_whole_variant_list_as_get():
    """404/405/415 is a verb error, not a key error: the run must not be lost."""
    from seti.century.api import queryexps

    sess = _FakeSession({
        ("POST", ("dec_deg", "ra_deg")): _FakeResponse(405, None, "Method Not Allowed"),
        ("GET", ("dec_deg", "ra_deg")): _FakeResponse(200, [{"series": "a", "date_jd": 2.4e6}]),
    })
    r = queryexps(1.0, 2.0, session=sess, retries=1, backoff_s=0.0)
    assert r.ok and r.method == "GET" and r.n_rows == 1
    assert sess.calls[0][0] == "POST" and any(c[0] == "GET" for c in sess.calls)


def test_a_refusal_is_not_retried_as_a_shape_problem():
    """403/429 is the service saying no; renaming keys cannot cure it."""
    from seti.century.api import lightcurve

    sess = _FakeSession({})
    sess.table = {k: _FakeResponse(403, None, "forbidden") for k in
                  [("POST", ("gsc_bin_index", "ref_number", "refcat"))]}
    r = lightcurve("apass", 1, 2, session=sess, retries=1, backoff_s=0.0)
    assert not r.ok and r.status == 403 and len(sess.calls) == 1
    assert "403" in r.error


def test_flag_parsing_from_source_text_and_config_fallback():
    src = (
        "from enum import IntFlag\n"
        "class AFlags(IntFlag):\n    HIGH_BACKGROUND = 1 << 6\n    SXT_BLEND = 1 << 12\n"
        "    SUSPECTED_DEFECT = 1 << 15\n"
        "class BFlags(IntFlag):\n    NEIGHBORS = 1\n    BLEND = 2\n    SATURATED = 4\n"
    )
    a, b = flagdefs_from_source(src)
    assert a.bits["SXT_BLEND"] == 4096 and a.blend_mask == 4096
    assert a.reject_mask == 1 << 15
    assert b.blend_mask == 3 and b.reject_mask == 4
    a2, b2 = resolve_flagdefs({"flag_bits": {"aflags": {"X_BLEND": 8}, "bflags": {}}}, None)
    assert a2.source == "config" and a2.blend_mask == 8 and not b2.available
    a3, _ = resolve_flagdefs({"flag_bits": {"aflags": {}, "bflags": {}}}, None)
    assert not a3.available


def test_flag_bits_are_read_from_photometry_not_lightcurves():
    """The enums live in ``daschlab/photometry.py``; ``lightcurves.py`` imports them.

    The first probe (run 35738717013) fetched only ``lightcurves.py``, parsed
    nothing, and wrote an empty ``flag_bits.json`` --- i.e. the whole run would
    have applied NO blend and NO reject cut while reporting a plate light curve
    as clean.  So: the resolution takes a LIST of source texts, a definition
    found in any of them beats the config, and ``photometry.py`` is in the list
    the probe and the acquire stage fetch.
    """
    from seti.century.api import DASCHLAB_FILES, FLAG_SOURCE_FILES

    assert "photometry.py" in FLAG_SOURCE_FILES and "photometry.py" in DASCHLAB_FILES
    lc_src = "from .photometry import AFlags, BFlags\n\ndef _query_lc():\n    pass\n"
    phot_src = (
        "from enum import IntFlag\n"
        "class AFlags(IntFlag):\n    SXT_BLEND = 1 << 26\n    SUSPECTED_DEFECT = 1 << 25\n"
        "class BFlags(IntFlag):\n    NEIGHBORS = 1\n    BLEND = 2\n    SATURATED = 4\n"
    )
    # lightcurves.py alone yields nothing -- exactly the observed failure.
    a0, b0 = resolve_flagdefs(None, lc_src)
    assert not a0.available and not b0.available
    a, b = resolve_flagdefs(None, [lc_src, phot_src])
    assert a.source == "daschlab_source" and a.blend_mask == 1 << 26
    assert b.blend_mask == 3 and b.reject_mask == 4
    # A None in the list (a file that 404'd) is skipped, not fatal.
    a2, _ = resolve_flagdefs({"flag_bits": {"aflags": {"X_BLEND": 8}, "bflags": {}}},
                             [None, phot_src])
    assert a2.source == "daschlab_source" and a2.blend_mask == 1 << 26


def test_config_flag_bits_fallback_covers_the_blend_and_reject_cuts():
    """config/century.yaml must be able to stand in for the live source.

    Committed bits are the offline fallback; if they ever stop naming a blend
    bit or a reject bit the blending kill is silently off on any runner that
    cannot reach GitHub.
    """
    import pathlib

    import yaml

    conf = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parents[1] / "config" / "century.yaml").read_text())
    a, b = resolve_flagdefs(conf, None)
    assert a.source == "config" and b.source == "config"
    assert a.blend_mask > 0 and a.reject_mask > 0
    assert b.blend_mask > 0 and b.reject_mask > 0
    # The named bits the channel's kills depend on.
    assert "SXT_BLEND" in a.bits and "CASE_BC_BLEND" in a.bits
    assert "SUSPECTED_DEFECT" in a.bits and "BAD_PLATE_QUALITY" in a.bits
    assert "NEIGHBORS" in b.bits and "BLEND" in b.bits and "SATURATED" in b.bits


def test_error_column_is_the_one_that_belongs_to_the_magnitude():
    """DR7 serves several calibrations with their own rms; 99.0 means "none"."""
    df = pd.DataFrame({
        "date_jd": [2415020.5, 2415030.5, 2440000.5, 2440010.5],
        "magcal_magdep": [11.0, 11.1, 11.05, 11.2],
        "magcal_magdep_rms": [0.10, 99.0, 0.12, 0.11],
        "magcal_local": [12.0, 12.1, 12.05, 12.2],
        "magcal_local_rms": [0.50, 0.55, 0.52, 0.51],
        "limiting_mag_local": [14.0, 14.1, 15.0, 15.1],
        "series": ["a", "a", "mc", "mc"],
        "plate_number": [1, 2, 3, 4],
        "aflags": [0, 0, 0, 0], "bflags": [0, 0, 0, 0],
    })
    lc = from_api_frame(df, default_err=0.15)
    assert lc is not None and lc.n_det == 4
    # magcal_magdep was chosen, so magcal_magdep_rms is the error -- not the
    # five-times-larger local rms that pick_column used to reach first.
    assert abs(lc.err[0] - 0.10) < 1e-9 and abs(lc.err[3] - 0.11) < 1e-9
    # The 99.0 sentinel becomes the default error, never a 99-mag error bar.
    assert abs(lc.err[1] - 0.15) < 1e-9


def test_exposure_time_is_joined_in_from_queryexps():
    """DR7 light curves carry no exposure time; queryexps does, in minutes.

    The smear ``|sinc(f t_exp)|`` is what makes an injection in a post-gap
    block comparable to the pre-gap detection, and Harvard exposure lengths are
    a property of the plate series, which is clustered in calendar time.  A
    star whose late plates expose longer keeps less of a short-period signal,
    and an efficiency computed without that is optimistic exactly where the
    cessation claim is made.
    """
    from seti.century.lightcurve import attach_exptime
    from seti.century.targets import EXPOSURE_KEY_COLS, exposure_table

    exps = ["series,platenum,scannum,mosnum,expnum,solnum,exptime,epoch,limMagApass",
            "a,101,0,0,0,1,45.0,1899.50,14.2",
            "a,102,0,0,0,1,45.0,1901.50,14.1",
            "mc,900,0,0,0,1,75.0,1975.50,15.1",
            "mc,901,0,0,0,1,75.0,1977.50,15.0",
            # A duplicate key (a second solution of the same exposure) must not
            # multiply the rows it is joined onto.
            "mc,901,0,0,0,2,75.0,1977.50,15.0"]
    et = exposure_table(to_frame(exps))
    assert list(et.columns) == [*EXPOSURE_KEY_COLS, "exptime_min"]
    assert len(et) == 4 and float(et["exptime_min"].iloc[0]) == 45.0

    df = pd.DataFrame({
        "date_jd": [2415020.5, 2415750.5, 2442000.5, 2442730.5, 2442740.5],
        # The last row is a non-detection: no magnitude.
        "magcal_magdep": [11.0, 11.1, 11.05, 11.2, np.nan],
        "magcal_magdep_rms": [0.10, 0.11, 0.12, 0.11, np.nan],
        "limiting_mag_local": [14.2, 14.1, 15.1, 15.0, 15.0],
        "series": ["a", "a", "mc", "mc", "mc"],
        "plate_number": [101, 102, 900, 901, 901],
        "mosaic_number": [0, 0, 0, 0, 0],
        "exposure_number": [0, 0, 0, 0, 0],
        "aflags": [0, 0, 0, 0, 0], "bflags": [0, 0, 0, 0, 0],
    })
    lc = from_api_frame(df)
    assert lc is not None and lc.n_det == 4 and lc.n_nd == 1
    assert not np.isfinite(lc.exptime_min).any()      # nothing to take it from
    prov = attach_exptime(lc, et)
    assert prov["key"] == "series+platenum+mosnum+expnum"
    assert prov["matched_det"] == 4 and prov["matched_nd"] == 1
    assert lc.exptime_unit == "minutes"
    assert np.allclose(lc.exptime_min, [45.0, 45.0, 75.0, 75.0])
    assert np.allclose(lc.exptime_nd, [75.0])
    # The pre/post difference is reported: this is the case where leaving the
    # smear unmodelled would have mattered.
    assert prov["exptime_min_pre"] == 45.0 and prov["exptime_min_post"] == 75.0
    # And it is a real difference in what the plates could have kept.
    assert smear_factor(1 / 0.12, np.array([75.0]))[0] \
        < smear_factor(1 / 0.12, np.array([45.0]))[0]


def test_exposure_join_never_falls_back_to_the_series_alone():
    """A plate that is not in the table keeps NaN, not its series' typical time.

    Handing every unmatched plate the series' exposure would synthesise a smear
    history that follows the series --- i.e. calendar time --- which is exactly
    the confounder the channel exists to separate from a change in the star.
    """
    from seti.century.lightcurve import attach_exptime
    from seti.century.targets import exposure_table

    et = exposure_table(to_frame(["series,platenum,mosnum,expnum,exptime",
                                  "a,101,0,0,45.0"]))
    df = pd.DataFrame({
        "date_jd": [2415020.5, 2415750.5],
        "magcal_magdep": [11.0, 11.1], "magcal_magdep_rms": [0.1, 0.1],
        "limiting_mag_local": [14.0, 14.0], "series": ["a", "a"],
        "plate_number": [101, 999], "mosaic_number": [0, 0], "exposure_number": [0, 0],
        "aflags": [0, 0], "bflags": [0, 0],
    })
    lc = from_api_frame(df)
    prov = attach_exptime(lc, et)
    assert prov["matched_det"] == 1
    assert np.isfinite(lc.exptime_min[0]) and not np.isfinite(lc.exptime_min[1])
    # An empty or absent table is a degradation, not a crash or a guess.
    lc2 = from_api_frame(df)
    p2 = attach_exptime(lc2, pd.DataFrame())
    assert p2["key"] == "none" and p2["matched_det"] == 0
    assert not np.isfinite(lc2.exptime_min).any()


def test_queryexps_epoch_is_not_the_observation_date():
    """``epoch`` is the coordinate epoch, 2000.0 on every plate.

    Run 35748748365 reported all six fields as ``2000.0..2000.0``, every plate
    post-gap and none pre-gap, over 96,909 exposures, because the year was
    taken from ``epoch``.  The observation timestamp is ``expdate``, in
    DASCH's own dialect: hyphens in the time field as well as the date, and
    sometimes a ``:60.0`` seconds value that is not a valid time at all.
    """
    from seti.century.targets import exposure_years

    lines = ["series,platenum,scannum,mosnum,expnum,solnum,exptime,expdate,epoch,limMagApass",
             "a,101,0,0,0,1,45.0,1899-07-04T12-34-56,2000.0,14.2",
             "mc,900,0,0,0,1,60.0,1975-12-31T23-59-60.0,2000.0,15.1",
             "dnb,901,0,0,0,1,60.0,1953-06-15T01-02-03,2000.0,15.0",
             "x,902,0,0,0,1,60.0,,2000.0,15.0"]
    df = to_frame(lines)
    assert "epoch" in df.columns          # it is served, and it is still wrong
    yr, col = exposure_years(df)
    assert col == "expdate"
    assert abs(yr[0] - 1899.50) < 0.01
    assert abs(yr[1] - 1976.00) < 0.01    # the ":60.0" row still parses
    assert abs(yr[2] - 1953.45) < 0.01
    assert not np.isfinite(yr[3])         # an empty date is NaN, not year zero
    # And the gap split is then real, not "everything is post-gap".
    good = yr[np.isfinite(yr)]
    assert int(np.sum(good < GAP[0])) == 2 and int(np.sum(good >= GAP[1])) == 1


def test_the_exposure_join_picks_the_key_that_places_the_most_plates():
    """A key that matches a handful must not beat one that matches nearly all.

    The two endpoints need not spell ``mosnum``/``expnum`` the same way.  If
    the four-part key were kept merely because it matched *something*, the
    plates it missed would be silently unsmeared while the join reported
    success on the most specific key.
    """
    from seti.century.lightcurve import attach_exptime
    from seti.century.targets import exposure_table

    # The table knows plate a/101 under four different exposure numbers, but
    # only expnum 0 is the one the light curve names; every other row shares
    # (series, platenum) with a light-curve point whose expnum differs.
    lines = ["series,platenum,mosnum,expnum,exptime"]
    for i in range(6):
        lines.append(f"a,{200 + i},0,0,45.0")
    et = exposure_table(to_frame(lines))
    n = 6
    df = pd.DataFrame({
        "date_jd": [2415020.5 + 400 * i for i in range(n)],
        "magcal_magdep": [11.0 + 0.01 * i for i in range(n)],
        "magcal_magdep_rms": [0.1] * n,
        "limiting_mag_local": [14.0] * n,
        "series": ["a"] * n,
        "plate_number": [200 + i for i in range(n)],
        "mosaic_number": [0] * n,
        # Only the first point agrees with the table on expnum; the rest do not.
        "exposure_number": [0] + [7] * (n - 1),
        "aflags": [0] * n, "bflags": [0] * n,
    })
    lc = from_api_frame(df)
    prov = attach_exptime(lc, et)
    # (series, platenum) places all six; the four-part key places one.
    assert prov["key"] == "series+platenum"
    assert prov["matched_det"] == 6
    assert np.allclose(lc.exptime_min, 45.0)


def test_a_shard_rebuilds_the_exposure_table_when_the_artifact_did_not_arrive(monkeypatch):
    """plate_exptime.csv travels between jobs as an artifact, and the artifact
    list lives in the workflow file, which is fixed when a run is dispatched.

    A sweep launched from an older workflow, or a reduce-only re-run, would
    otherwise lose the exposure times silently and leave every smear
    unmodelled.  The shard asks for them itself: one ``queryexps`` per field it
    has stars in, at the median position of those stars, because exposure time
    is a property of the plate and not of the star.
    """
    from seti.century import run as crun
    from seti.century.api import ApiResponse

    calls: list[tuple[float, float]] = []

    def fake_queryexps(ra, dec, **kw):
        calls.append((round(float(ra), 3), round(float(dec), 3)))
        lines = ["series,platenum,scannum,mosnum,expnum,solnum,exptime,epoch,limMagApass",
                 f"a,{100 + len(calls)},0,0,0,1,45.0,1899.5,14.2",
                 f"mc,{900 + len(calls)},0,0,0,1,75.0,1975.5,15.1"]
        df = to_frame(lines)
        return ApiResponse("queryexps", 200, True, 0.1, {}, "", "", len(df),
                           [str(c) for c in df.columns], df, "POST")

    monkeypatch.setattr("seti.century.targets.queryexps", fake_queryexps)
    mine = pd.DataFrame([
        {"target_id": 0, "ra": 10.0, "dec": 20.0, "field": "f1"},
        {"target_id": 1, "ra": 10.2, "dec": 20.2, "field": "f1"},
        {"target_id": 2, "ra": 80.0, "dec": -1.0, "field": "f2"},
    ])
    log = __import__("seti.knell.acquire", fromlist=["AcquisitionLog"]).AcquisitionLog()
    conf = _conf()
    conf["acquire"]["pause_s"] = 0.0
    et, src = crun._shard_exposure_table(mine, conf, log)
    # One request per FIELD, at the median of that field's stars -- not one per
    # star, which would be an extra request for every light curve fetched.
    assert len(calls) == 2
    assert (10.1, 20.1) in calls and (80.0, -1.0) in calls
    assert src == "acquire_stage_rebuild" and len(et) == 4
    assert set(et["series"]) == {"a", "mc"}
    # Nothing answering is a degradation, not a guess.
    monkeypatch.setattr("seti.century.targets.queryexps",
                        lambda ra, dec, **kw: ApiResponse("queryexps", 503, False, 0.1, {},
                                                          "HTTP 503", "", 0, [], None, "POST"))
    et2, src2 = crun._shard_exposure_table(mine, conf, log)
    assert src2 == "none" and not len(et2)


def test_exposure_table_survives_the_csv_round_trip(tmp_path):
    """The table reaches the shards through plate_exptime.csv, not in memory.

    A CSV column with one empty cell reads back as float64 while the light
    curve's identifiers are Int64; merging those matches nothing, and the run
    would report a DASCH coverage gap when what it had was a dtype mismatch.
    """
    from seti.century.lightcurve import attach_exptime
    from seti.century.targets import exposure_table

    et = exposure_table(to_frame(["series,platenum,mosnum,expnum,exptime",
                                  "a,101,0,0,45.0",
                                  "mc,900,0,0,75.0"]))
    p = tmp_path / "plate_exptime.csv"
    et.to_csv(p, index=False)
    back = pd.read_csv(p)
    df = pd.DataFrame({
        "date_jd": [2415020.5, 2442000.5],
        "magcal_magdep": [11.0, 11.1], "magcal_magdep_rms": [0.1, 0.1],
        "limiting_mag_local": [14.0, 15.0], "series": ["a", "mc"],
        "plate_number": [101, 900], "mosaic_number": [0, 0], "exposure_number": [0, 0],
        "aflags": [0, 0], "bflags": [0, 0],
    })
    lc = from_api_frame(df)
    prov = attach_exptime(lc, back)
    assert prov["matched_det"] == 2
    assert np.allclose(lc.exptime_min, [45.0, 75.0])
    # And with a missing identifier somewhere in the file, which is what turns
    # the column into a float in the first place.
    back2 = back.copy()
    back2.loc[0, "expnum"] = np.nan
    lc2 = from_api_frame(df)
    prov2 = attach_exptime(lc2, back2)
    assert prov2["matched_det"] >= 1


def test_margin_mask_is_keyed_to_the_star_not_the_point():
    rng = np.random.default_rng(3)
    lc = synth_plates(rng, lim_mean=13.0, lim_sd=1.0)
    m = lc.margin_mask(1.5)
    ref = lc.median_mag()
    assert (lc.lim[m] - ref >= 1.5).all()
    # A faint noise excursion on a deep plate is kept; a bright one on a
    # shallow plate is dropped -- the selection never looks at lc.mag.
    assert m.sum() < lc.good.sum()


def test_annual_table_and_common_mode():
    rng = np.random.default_rng(4)
    lc = synth_plates(rng)
    tab = annual_table(lc, min_per_year=3)
    assert len(tab) > 50 and (tab["n"] >= 3).all()
    rows = []
    for i in range(12):
        lc_i = synth_plates(np.random.default_rng(100 + i), step=0.2, mag0=11.0 + 0.05 * i)
        t = annual_table(lc_i)
        rows.append({"field": "f", "median_mag": lc_i.median_mag(),
                     "annual": t.to_dict(orient="list")})
    cm = field_common_mode(rows, min_stars=8, mag_bin=2.0)
    assert "f" in cm
    ys = next(iter(cm["f"].values()))
    pre = np.median([v for y, v in ys.items() if y < 1954])
    post = np.median([v for y, v in ys.items() if y >= 1970])
    assert abs((post - pre) - 0.2) < 0.03


# ---------------------------------------------------------------------------
# Cessation
# ---------------------------------------------------------------------------


def test_fixed_window_detection_fires_on_signal_and_not_on_noise():
    rng = np.random.default_rng(5)
    lc = synth_plates(rng, period=0.57, amp=0.5, n_per_year=30)
    blocks = calendar_blocks(lc, block_years=2.0, min_epochs_block=15, min_blocks=4)
    f_ref, h, _ = refine_reference(lc.t, lc.mag, lc.err, 1 / 0.57)
    assert abs(f_ref * 0.57 / h - 1.0) < 0.002
    hits = 0
    for b in blocks[:8]:
        d = detect_window(b, window_freqs(f_ref, b.t_span), fap=0.01, n_null=100, rng=rng)
        hits += int(d.detected)
    assert hits >= 7
    lc0 = synth_plates(rng, n_per_year=30)
    blocks0 = calendar_blocks(lc0, block_years=2.0, min_epochs_block=15, min_blocks=4)
    false = sum(int(detect_window(b, window_freqs(f_ref, b.t_span), fap=0.01, n_null=100,
                                  rng=rng).detected) for b in blocks0[:20])
    assert false <= 2


def test_censored_efficiency_falls_when_plates_are_shallow():
    rng = np.random.default_rng(6)
    f = 1 / 0.57
    deep = synth_plates(rng, n_per_year=30, lim_mean=14.0, lim_sd=0.3)
    shallow = synth_plates(rng, n_per_year=30, lim_mean=11.2, lim_sd=0.3)
    bd = calendar_blocks(deep, block_years=2.0, min_epochs_block=12, min_blocks=4)[3]
    bs = calendar_blocks(shallow, block_years=2.0, min_epochs_block=12, min_blocks=4)[3]
    ed = censored_efficiency(bd, window_freqs(f, bd.t_span), f, 0.5, n_trials=40, n_null=50,
                             rng=rng)
    es = censored_efficiency(bs, window_freqs(f, bs.t_span), f, 0.5, n_trials=40, n_null=50,
                             rng=rng)
    assert ed.eta >= 0.9 and ed.censor_frac < 0.05
    assert es.censor_frac > 0.2
    assert es.eta <= ed.eta
    assert es.eta_uncensored >= es.eta - 0.05
    assert 0 < ed.miss_upper <= 1


def test_injected_cessation_is_recovered():
    # The series change is put well before the cessation ON PURPOSE: a plate
    # series that changes AT the transition is a confounder the channel is
    # required to kill (test_series_change_at_the_transition_is_not_a_cessation),
    # so injecting one here would test the wrong thing.
    rng = np.random.default_rng(7)
    lc = synth_plates(rng, period=0.57, amp=0.5, stop_year=1931.0, n_per_year=30,
                      lim_mean=14.0, lim_sd=0.4, series_switch_year=1910.0)
    res = analyze_century(lc, 0.57, **_cess_kwargs(), rng=rng)
    assert res.status == "cessation", (res.status, res.flags)
    assert 1927 < res.transition_year < 1935
    assert res.n_pre_detected >= 2 and res.n_post_informative >= 2
    assert res.eta_min_post >= 0.9
    assert res.p_persist_upper <= 0.01
    assert abs(res.mean_shift_mag) < 0.1
    assert not res.transition_at_gap


def test_persisting_signal_on_degrading_plates_is_not_a_cessation():
    """The channel's defining null: the clock never stops, the plates get shallow."""
    rng = np.random.default_rng(8)
    flagged = 0
    for k in range(4):
        lc = synth_plates(np.random.default_rng(80 + k), period=0.57, amp=0.5, n_per_year=30,
                          lim_mean=14.0, lim_sd=0.3, post_gap_lim_shift=-2.9,
                          post_gap_err_scale=1.6)
        res = analyze_century(lc, 0.57, **_cess_kwargs(), rng=rng)
        flagged += int(res.status == "cessation")
        assert res.status in ("still_periodic", "low_efficiency", "no_clean_transition",
                              "cessation"), res.status
        if res.status != "still_periodic":
            assert res.status != "cessation" or res.eta_min_post >= 0.9
    assert flagged == 0


def test_cessation_across_gap_is_flagged_and_mean_flux_deferred():
    rng = np.random.default_rng(9)
    lc = synth_plates(rng, period=0.57, amp=0.5, stop_year=1960.0, n_per_year=30,
                      lim_mean=14.0, lim_sd=0.4, step=0.3)
    res = analyze_century(lc, 0.57, **_cess_kwargs(), rng=rng)
    assert res.transition_at_gap
    assert "transition_at_menzel_gap" in res.flags
    assert res.mean_shift_across_gap
    assert abs(res.mean_shift_mag - 0.3) < 0.1
    assert "mean_flux_gap_uncorrected" in res.flags


def test_faded_star_is_not_a_cessation():
    rng = np.random.default_rng(10)
    lc = synth_plates(rng, period=0.57, amp=0.5, n_per_year=30, lim_mean=14.0, lim_sd=0.3)
    # Fade by 1.5 mag from 1935 on: the signal keeps going but sinks in noise.
    yr = lc.year
    lc.mag = lc.mag + np.where(yr > 1935, 1.5, 0.0)
    lc.err = lc.err * np.where(yr > 1935, 4.0, 1.0)
    res = analyze_century(lc, 0.57, **_cess_kwargs(), rng=rng)
    assert res.status != "cessation"
    assert res.status in ("low_efficiency", "faded_or_brightened", "no_clean_transition",
                          "still_periodic", "rejected", "variance_conserved")


def test_vanished_star_is_not_a_cessation():
    rng = np.random.default_rng(11)
    lc = synth_plates(rng, period=0.57, amp=0.5, n_per_year=30, lim_mean=14.0, lim_sd=0.3)
    # After 1931 the star is simply not detected although the plates are deep:
    # move its detections into the non-detection set.
    yr = lc.year
    gone = yr > 1931
    keep_some = gone & (rng.random(lc.n_det) < 0.08)
    drop = gone & ~keep_some
    lc2 = CenturyLC(
        t=lc.t[~drop], mag=lc.mag[~drop], err=lc.err[~drop], lim=lc.lim[~drop],
        series=lc.series[~drop], exptime_min=lc.exptime_min[~drop], blend=lc.blend[~drop],
        reject=lc.reject[~drop], plate=lc.plate[~drop],
        t_nd=np.concatenate([lc.t_nd, lc.t[drop]]), lim_nd=np.concatenate([lc.lim_nd, lc.lim[drop]]),
        series_nd=np.concatenate([lc.series_nd, lc.series[drop]]),
        exptime_nd=np.concatenate([lc.exptime_nd, lc.exptime_min[drop]]), flags_applied=True)
    res = analyze_century(lc2, 0.57, **_cess_kwargs(min_epochs_block=4), rng=rng)
    assert res.status != "cessation"
    assert res.status in ("vanished_not_ceased", "no_clean_transition", "insufficient_data",
                          "low_efficiency", "still_periodic", "rejected")


def test_mode_switch_and_blend_transition_are_closed():
    rng = np.random.default_rng(12)
    lc = synth_plates(rng, period=0.57, amp=0.5, n_per_year=30, lim_mean=14.0, lim_sd=0.3)
    yr = lc.year
    f2 = 1 / 0.41
    # Mode switch at 1931: same amplitude, new frequency.
    sw = yr > 1931
    lc.mag = lc.mag - (0.5 * smear_factor(1 / 0.57, lc.exptime_min)
                       * np.sin(2 * np.pi * lc.t / 0.57 + 0.3)) * sw \
        + 0.5 * np.sin(2 * np.pi * f2 * lc.t) * sw
    res = analyze_century(lc, 0.57, **_cess_kwargs(), rng=rng)
    assert res.status != "cessation"
    if res.status not in ("still_periodic", "no_clean_transition"):
        assert ("variance_conserved_mode_switch_like" in res.flags
                or "post_blind_periodic" in res.flags or "fail_pdm_post" in res.flags)
    # Blend transition: the pre blocks are clean and the post blocks 60% blended.
    lc2 = synth_plates(rng, period=0.57, amp=0.5, stop_year=1931.0, n_per_year=30,
                       lim_mean=14.0, lim_sd=0.3)
    lc2.blend = (lc2.year > 1931) & (rng.random(lc2.n_det) < 0.6)
    res2 = analyze_century(lc2, 0.57, **_cess_kwargs(), rng=rng)
    assert res2.status != "cessation" or "blend_transition" not in res2.flags
    assert np.isfinite(res2.blend_frac_post) and res2.blend_frac_post > 0.4


def test_series_change_at_the_transition_is_not_a_cessation():
    """Emulsion changes are series changes, and series are clustered in calendar
    time.  A "cessation" whose pre and post blocks share no plate series cannot
    be told from the plates changing, so it is not a cessation."""
    rng = np.random.default_rng(31)
    lc = synth_plates(rng, period=0.57, amp=0.5, stop_year=1932.0, n_per_year=30,
                      lim_mean=14.0, lim_sd=0.3, series_switch_year=1932.0)
    res = analyze_century(lc, 0.57, **_cess_kwargs(), rng=rng)
    assert res.status != "cessation", (res.status, res.flags)
    assert "series_disjoint" in res.flags
    assert res.series_overlap_frac == 0.0 or not np.isfinite(res.series_overlap_frac)
    # The same star with the series change moved off the transition IS one.
    lc2 = synth_plates(np.random.default_rng(31), period=0.57, amp=0.5, stop_year=1932.0,
                       n_per_year=30, lim_mean=14.0, lim_sd=0.3, series_switch_year=1910.0)
    res2 = analyze_century(lc2, 0.57, **_cess_kwargs(), rng=np.random.default_rng(32))
    assert "series_disjoint" not in res2.flags
    assert res2.series_overlap_frac > 0.0


def test_a_lone_post_gap_false_alarm_does_not_move_the_transition():
    """The Hippke/Lund failure mode with a periodogram in front of it.

    The per-block detector fires at rate `fap` by construction and the first
    block after the Menzel gap is the densest post-gap block in DASCH.  If one
    such false alarm can end the pre-segment, the transition moves across the
    gap, the gap's photometric step stops being deferred, and the plates' own
    offset is charged to the star.
    """
    from seti.century.cease import analyze_century as _ac

    rng = np.random.default_rng(9)
    lc = synth_plates(rng, period=0.57, amp=0.5, stop_year=1960.0, n_per_year=30,
                      lim_mean=14.0, lim_sd=0.4, step=0.3)
    res = _ac(lc, 0.57, **_cess_kwargs(), rng=rng)
    # The signal really stops at the gap: the transition has to be AT the gap
    # whether or not a post-gap block happened to fire.
    assert res.transition_at_gap, (res.transition_year, res.flags)
    assert res.last_detected_year < GAP[0] <= res.first_post_year
    assert res.mean_shift_across_gap and "mean_flux_gap_uncorrected" in res.flags
    # And the 0.3 mag step is measured, not absorbed into the star.
    assert abs(res.mean_shift_mag - 0.3) < 0.1


def test_catalogue_period_not_present_is_reported_not_claimed():
    rng = np.random.default_rng(13)
    lc = synth_plates(rng, n_per_year=30, lim_mean=14.0)
    res = analyze_century(lc, 0.57, **_cess_kwargs(), rng=rng)
    assert res.status == "period_not_recovered"
    assert "catalogue_period_not_seen_in_any_block" in res.flags


# ---------------------------------------------------------------------------
# Fade and scatter
# ---------------------------------------------------------------------------


def test_menzel_step_alone_is_not_a_fade():
    """The Hippke / Lund test.  A constant star with a 0.15 mag offset across
    the gap: the naive century slope is 'significant'; the step model is not."""
    naive_fades = fades = 0
    for k in range(6):
        lc = synth_plates(np.random.default_rng(200 + k), step=0.15, err=0.10)
        res, _ = analyze_fade(lc, margin=1.0, gap=GAP)
        assert res.status != "insufficient_data"
        naive_fades += int(res.slope_naive_sigma >= 5 and res.slope_naive_mag_per_century > 0)
        fades += int(res.is_fade)
        assert abs(res.step_mag - 0.15) < 0.04
    assert naive_fades >= 4
    assert fades == 0


def test_real_fade_is_recovered_and_pre_gap_segment_carries_it():
    rng = np.random.default_rng(14)
    lc = synth_plates(rng, slope_per_century=0.5, step=0.15, err=0.10)
    res, tab = analyze_fade(lc, margin=1.0, gap=GAP)
    assert res.is_fade, res.flags
    assert abs(res.slope_mag_per_century - 0.5) < 0.1
    assert res.slope_pre_sigma >= 3
    assert abs(res.step_mag - 0.15) < 0.05
    assert res.deep_consistent
    assert len(tab) >= 15


def test_slope_consistency_compares_errors_not_significances():
    """The units contract of the robustness refits.

    ``_consistent`` takes 1-sigma ERRORS.  Handing it significances makes the
    threshold tens of mag per century, so the deeper-plate and single-series
    refits can never disagree and two of the channel's three robustness guards
    are silently inert.
    """
    from seti.century.fade import _consistent

    assert not _consistent(1.0, 0.2, 0.05, 0.2)       # 3.4 sigma apart
    assert _consistent(1.0, 0.2, 0.9, 0.2)            # 0.35 sigma apart
    # An unusable error is a failure to ESTABLISH consistency, not consistency.
    assert not _consistent(1.0, float("nan"), 0.9, 0.2)
    assert not _consistent(1.0, 0.0, 0.9, 0.0)


def test_a_fade_carried_only_by_shallow_plates_is_rejected():
    """Near the plate limit only the bright excursions are measured, so a
    depth history that changes with calendar time makes a fade out of nothing.
    Here a spurious drift is put on the SHALLOW plates alone; the deeper-plate
    refit must disagree and the fade must not survive."""
    rng = np.random.default_rng(21)
    lc = synth_plates(rng, n_per_year=30, lim_mean=13.5, lim_sd=0.0, mag0=11.0, err=0.10)
    shallow = np.arange(lc.n_det) % 2 == 1
    lc.lim = np.where(shallow, 12.3, 16.0)
    yr = lc.year
    lc.mag = lc.mag + np.where(shallow, 1.2 * (yr - 1890.0) / 100.0, 0.0)
    res, _ = analyze_fade(lc, margin=1.0, deep_extra=1.0, gap=GAP)
    assert np.isfinite(res.slope_deep_mag_per_century), res.flags
    assert not res.deep_consistent, (res.slope_mag_per_century,
                                     res.slope_deep_mag_per_century, res.flags)
    assert "depends_on_shallow_plates" in res.flags
    assert not res.is_fade


def test_shallow_plate_censoring_bias_is_caught_by_the_deeper_margin():
    """A constant variable star on plates that get shallower late: the annual
    median of the *kept* plates brightens (only the bright phase is measured),
    and the deep-margin refit disagrees."""
    rng = np.random.default_rng(15)
    lc = synth_plates(rng, period=3.1, amp=0.8, n_per_year=30, lim_mean=12.6, lim_sd=0.6,
                      post_gap_lim_shift=-0.6)
    res, _ = analyze_fade(lc, amp_cat=0.0, margin=0.3, deep_extra=1.0, gap=GAP)
    assert not res.is_fade
    # With the amplitude-aware margin the kept plates are deep enough anyway.
    res2, _ = analyze_fade(lc, amp_cat=0.8, margin=1.0, gap=GAP)
    assert res2.margin_used >= 1.3
    assert not res2.is_fade


def test_rising_scatter_is_recovered_and_step_in_scatter_is_not():
    rng = np.random.default_rng(16)
    lc = synth_plates(rng, scatter_growth_mmag_per_century=250.0, err=0.08, n_per_year=30,
                      lim_mean=14.0)
    res, ss = analyze_scatter(lc, margin=1.0, gap=GAP)
    assert res.is_rust, res.flags
    assert res.pre_rank_p <= 0.01 and res.slope_var_per_century > 0
    # A post-gap emulsion with twice the scatter and no growth is a step.
    hits = 0
    for k in range(4):
        lc2 = synth_plates(np.random.default_rng(300 + k), err=0.08, n_per_year=30,
                           post_gap_err_scale=2.0, lim_mean=14.0)
        # The reported errors do NOT know about the emulsion change.
        lc2.err[:] = 0.08
        r2, _ = analyze_scatter(lc2, margin=1.0, gap=GAP)
        hits += int(r2.is_rust)
        assert not r2.is_rust or r2.pre_rank_p <= 0.01
    assert hits == 0


def test_periodic_residualisation_keeps_the_statistic_aperiodic():
    rng = np.random.default_rng(17)
    lc = synth_plates(rng, period=0.57, amp=0.6, err=0.08, n_per_year=40, lim_mean=14.0)
    f = 1 / 0.57
    y_res = residualise_periodic(lc.t, lc.mag, lc.err, f, n_harm=2)
    assert np.std(y_res) < 0.5 * np.std(lc.mag)
    res, _ = analyze_scatter(lc, f_ref=f, amp_cat=0.6, margin=1.0, gap=GAP)
    assert res.residualised
    assert not res.is_rust


# ---------------------------------------------------------------------------
# Vetting and orchestration
# ---------------------------------------------------------------------------


def test_type_regexes():
    assert is_lpv_type("M") and is_lpv_type("SRB") and is_lpv_type("L:") and is_lpv_type("SR/M")
    assert not is_lpv_type("RRAB") and not is_lpv_type("EA")
    assert is_periodic_type("RRAB") and is_periodic_type("EA/SD") and is_periodic_type("DCEP")
    assert not is_periodic_type("M") and not is_periodic_type("UG")


def test_vet_row_trips_every_kill():
    base = {"cess_status": "cessation", "cess_n_post_informative": 3, "cess_n_pre_detected": 4,
            "cess_flags": "", "vtype": "RRAB", "mag_cat": 11.0, "period_cat": 0.5}
    assert vet_row(dict(base))["verdict"] == "survivor"
    assert vet_row(dict(base, pm_total_masyr=120))["verdict"] == "killed:high_pm"
    assert vet_row(dict(base, vtype="M"))["verdict"] == "killed:long_period_giant"
    assert vet_row(dict(base, mag_cat=6.0))["verdict"] == "killed:too_bright"
    assert vet_row(dict(base, cess_flags="vanished_not_ceased"))["verdict"] == \
        "killed:vanished_not_ceased"
    assert vet_row(dict(base, cess_flags="blend_transition"))["verdict"] == \
        "killed:blend_transition"
    assert vet_row(dict(base, cess_flags="series_disjoint"))["verdict"] == "killed:series_disjoint"
    assert vet_row(dict(base, same_series_status="no_clean_transition"))["verdict"] == \
        "killed:same_series_fails"
    assert vet_row(dict(base, cess_mean_shift_corrected=0.4))["verdict"] == \
        "killed:mean_flux_changed"
    assert vet_row(dict(base, cess_n_post_informative=1))["verdict"] == \
        "killed:single_plate_evidence"
    r = vet_row(dict(base, cess_flags="transition_at_menzel_gap"))
    assert r["verdict"] == "survivor" and "menzel_gap_transition" in r["notes"]
    assert vet_row({"cess_status": "still_periodic"})["verdict"] == "not_candidate"
    assert vet_row({"fade_is_fade": True, "fade_n_years": 40})["verdict"] == "survivor"
    assert vet_row({"rust_is_rust": True, "rust_n_seasons": 5})["verdict"] == \
        "killed:single_plate_evidence"


def test_parse_shard():
    assert parse_shard("2/5") == (2, 5) and parse_shard(None) == (0, 1)
    with pytest.raises(ValueError):
        parse_shard("5/5")


def test_screen_star_runs_all_three_and_serialises():
    rng = np.random.default_rng(18)
    lc = synth_plates(rng, period=0.57, amp=0.5, stop_year=1931.0, n_per_year=30, lim_mean=14.0,
                      series_switch_year=1910.0)
    row = screen_star(lc, {"target_id": 1, "name": "x", "kind": "variable", "field": "f",
                           "period_cat": 0.57, "amp_cat": 0.5, "mag_cat": 11.0, "vtype": "RRAB"},
                      _conf(), rng=rng)
    assert row["usable"] and row["cess_status"] == "cessation"
    assert row["fade_status"] in ("no_secular_change", "rejected")
    assert row["rust_status"] in ("no_rise", "rejected", "insufficient_data")
    assert row["annual"]["year"] and row["season"]["season_t"]
    json.dumps(row, default=lambda o: None)


def _dr7_lightcurve_lines(lc: CenturyLC, exptimes: dict[str, float]) -> list[str]:
    """Render a synthetic star the way DR7 actually serves it.

    An array of CSV lines, header first, with the exact column spellings of
    ``_COLTYPES`` in ``daschlab/photometry.py``: ``date_jd`` on the JD scale,
    ``magcal_magdep`` with ``magcal_magdep_rms`` beside it (99.0 where the
    archive has no rms), ``limiting_mag_local``, ``series``/``plate_number``/
    ``mosaic_number``/``exposure_number``, ``aflags``/``bflags``, and rows with
    an EMPTY magnitude for the plates that did not detect the star.
    """
    cols = ["ref_number", "ra_deg", "dec_deg", "date_jd", "magcal_magdep",
            "magcal_magdep_rms", "magcal_local", "magcal_local_rms",
            "limiting_mag_local", "series", "plate_number", "mosaic_number",
            "exposure_number", "solution_number", "sextractor_number",
            "gsc_bin_index", "aflags", "bflags"]
    lines = [",".join(cols)]

    def _row(t, mag, err, lim, ser, pnum, detected):
        jd = t + 2400000.5
        return ",".join([
            "42", "10.0", "20.0", f"{jd:.5f}",
            f"{mag:.4f}" if detected else "",
            f"{err:.4f}" if detected else "99.0",
            f"{mag + 1.0:.4f}" if detected else "",
            "0.5" if detected else "99.0",
            f"{lim:.4f}", str(ser), str(pnum), "0", "0", "1",
            "7" if detected else "0", "1234" if detected else "0", "0", "0"])

    n = 0
    for i in range(lc.n_det):
        ser = str(lc.series[i])
        lines.append(_row(lc.t[i], lc.mag[i], lc.err[i], lc.lim[i], ser, 1000 + i, True))
        exptimes[f"{ser}:{1000 + i}"] = 45.0 if lc.year[i] < 1954.0 else 75.0
        n += 1
    for j in range(lc.n_nd):
        ser = str(lc.series_nd[j])
        lines.append(_row(lc.t_nd[j], 0.0, 0.0, lc.lim_nd[j], ser, 5000 + j, False))
        exptimes[f"{ser}:{5000 + j}"] = 45.0 if lc.year_nd[j] < 1954.0 else 75.0
        n += 1
    assert n == lc.n_det + lc.n_nd
    return lines


def test_dr7_wire_format_runs_end_to_end_into_the_three_statistics():
    """The column-name contract, proved from the wire format to the verdict.

    Every other test in this file starts from a ``CenturyLC`` built in Python.
    This one starts from the bytes DR7 actually sends --- an array of CSV lines
    with ``daschlab``'s column spellings --- and carries them through
    ``to_frame`` -> ``from_api_frame`` -> ``attach_exptime`` -> ``screen_star``.
    A renamed column, a misread time scale or a mispaired error would pass
    every synthetic test and fail only on the runner, an hour of queue later.
    """
    from seti.century.lightcurve import attach_exptime
    from seti.century.targets import exposure_table

    rng = np.random.default_rng(77)
    truth = synth_plates(rng, period=0.57, amp=0.5, stop_year=1931.0, n_per_year=30,
                         lim_mean=14.0, series_switch_year=1910.0)
    exptimes: dict[str, float] = {}
    lines = _dr7_lightcurve_lines(truth, exptimes)

    df = to_frame(lines)
    assert "magcal_magdep" in df.columns and "limiting_mag_local" in df.columns
    assert len(df) == truth.n_det + truth.n_nd

    lc = from_api_frame(df)
    assert lc is not None
    # The detections and non-detections come back split the way they went in.
    assert lc.n_det == truth.n_det and lc.n_nd == truth.n_nd
    assert np.allclose(np.sort(lc.t), np.sort(truth.t), atol=1e-3)
    # date_jd was read as a JD, not swallowed as an MJD two centuries early.
    assert 1885.0 < float(np.min(lc.year)) < 1900.0
    assert 1985.0 < float(np.max(lc.year)) < 1995.0
    # The error is magcal_magdep_rms, and the 99.0 sentinel never reaches a
    # detection because the 99.0 rows are the non-detections.
    assert np.allclose(np.sort(lc.err), np.sort(truth.err), atol=1e-3)
    assert float(np.max(lc.err)) < 1.0

    exps_lines = ["series,platenum,scannum,mosnum,expnum,solnum,exptime,epoch,limMagApass"]
    for key, et in exptimes.items():
        ser, pnum = key.split(":")
        exps_lines.append(f"{ser},{pnum},0,0,0,1,{et:.1f},1900.0,14.5")
    prov = attach_exptime(lc, exposure_table(to_frame(exps_lines)))
    assert prov["matched_det"] == lc.n_det and prov["matched_nd"] == lc.n_nd
    assert prov["exptime_min_pre"] == 45.0 and prov["exptime_min_post"] == 75.0

    row = screen_star(lc, {"target_id": 1, "name": "dr7", "kind": "variable", "field": "f",
                           "period_cat": 0.57, "amp_cat": 0.5, "mag_cat": 11.0,
                           "vtype": "RRAB"}, _conf(), rng=rng)
    assert row["usable"], row
    assert row["cess_status"] == "cessation", (row["cess_status"], row["cess_flags"])
    assert 1927 < row["cess_transition_year"] < 1935
    assert not row["cess_transition_at_gap"]
    json.dumps(row, default=lambda o: None)


def _fake_shard(tmp_path, rows_lc: dict[str, CenturyLC], targets: pd.DataFrame,
                acquire: dict) -> None:
    from seti.century.run import _save_shard_lcs
    sd = tmp_path / "shards" / "0_of_1"
    sd.mkdir(parents=True)
    store = {tid: lc.to_arrays() for tid, lc in rows_lc.items()}
    meta = {"shard": [0, 1], "stars": {tid: {"n_raw": lc.n_raw, "n_det": lc.n_det,
                                             "n_nd": lc.n_nd, "flags_applied": True,
                                             "columns": [], "exptime_unit": "minutes"}
                                       for tid, lc in rows_lc.items()},
            "flag_source": "config"}
    if store:
        _save_shard_lcs(sd / "lightcurves.npz", store, meta)
    (sd / "acquire_summary.json").write_text(json.dumps(acquire))
    targets.to_csv(tmp_path / "targets.csv", index=False)
    (tmp_path / "targets_summary.json").write_text(json.dumps({"n_targets": len(targets),
                                                               "fields": {}}))


def test_empty_api_gives_no_data_reached(tmp_path):
    targets = pd.DataFrame([{"target_id": 0, "name": "a", "ra": 1.0, "dec": 2.0,
                             "kind": "variable", "vtype": "RRAB", "period_cat": 0.5,
                             "mag_cat": 11.0, "amp_cat": 0.5, "source": "vsx", "field": "f",
                             "gsc_bin_index": np.nan, "ref_number": np.nan,
                             "pm_total_masyr": np.nan, "colour": np.nan, "n_det_cat": np.nan}])
    _fake_shard(tmp_path, {}, targets,
                {"stage": "acquire", "shard": [0, 1], "n_targets": 1, "n_fetched": 0,
                 "n_failed": 1, "n_empty": 0, "n_resumed": 0, "truncated": False,
                 "flag_source": "none", "service_probe_ok": False,
                 "verdict": "NO_DATA_REACHED"})
    stage_screen(_conf(), tmp_path, (0, 1))
    s = stage_assess(_conf(), tmp_path, confirm=False, gaia=False)
    assert s["verdict_code"] == "NO_DATA_REACHED"
    assert s["funnel"]["n_lc_fetched"] == 0 and s["funnel"]["n_candidates_any"] == 0
    assert (tmp_path / "summary.json").exists()


def test_no_shard_directory_is_not_a_statement_about_the_archive(tmp_path):
    """A reduce with no shard at all has not failed to reach DASCH.

    It happens when the sweep was cancelled or never ran, or when a
    reduce-only re-run is pointed at a run whose artifacts are gone.  Saying
    NO_LIGHTCURVES_RETURNED there would read as "DASCH returned nothing for
    these targets" — a claim about the archive that no request was ever made
    to support.  Observed on run 35748748365, cancelled between its targets
    and sweep jobs, whose `assess` job was still scheduled by ``if: always()``.
    """
    targets = pd.DataFrame([{"target_id": i, "name": f"s{i}", "ra": 10.0 + i, "dec": 20.0,
                             "kind": "bright", "vtype": "", "period_cat": np.nan,
                             "mag_cat": 11.0, "amp_cat": 0.0, "source": "apass",
                             "field": "f", "gsc_bin_index": 1, "ref_number": i,
                             "pm_total_masyr": np.nan, "colour": np.nan,
                             "n_det_cat": np.nan} for i in range(5)])
    targets.to_csv(tmp_path / "targets.csv", index=False)
    (tmp_path / "targets_summary.json").write_text(json.dumps({"n_targets": len(targets),
                                                               "fields": {}}))
    # No shards/ directory at all.
    s = stage_assess(_conf(), tmp_path, confirm=False, gaia=False)
    assert s["verdict_code"] == "NO_SHARDS_PRESENT", s["verdict_code"]
    # Targets were selected, so this is NOT "no targets"; and nothing was
    # fetched, but that is not the archive's answer.
    assert s["funnel"]["n_targets"] == 5
    assert s["funnel"]["n_lc_fetched"] == 0 and s["funnel"]["n_usable"] == 0
    assert (tmp_path / "summary.json").exists()


def test_shard_roundtrip_screen_and_assess_end_to_end(tmp_path):
    lcs, trows = {}, []
    for i in range(10):
        stop = 1931.0 if i == 0 else None
        lc = synth_plates(np.random.default_rng(400 + i), period=0.57 if i < 3 else 0.0,
                          amp=0.5 if i < 3 else 0.0, stop_year=stop, n_per_year=25,
                          lim_mean=14.0, step=0.12, mag0=11.0 + 0.05 * i,
                          series_switch_year=1910.0)
        lcs[str(i)] = lc
        trows.append({"target_id": i, "name": f"s{i}", "ra": 10.0 + i * 0.01, "dec": 20.0,
                      "kind": "variable" if i < 3 else "bright", "vtype": "RRAB" if i < 3 else "",
                      "period_cat": 0.57 if i < 3 else np.nan, "mag_cat": 11.0 + 0.05 * i,
                      "amp_cat": 0.5 if i < 3 else 0.0, "source": "vsx", "field": "f",
                      "gsc_bin_index": 1, "ref_number": i, "pm_total_masyr": np.nan,
                      "colour": np.nan, "n_det_cat": np.nan})
    _fake_shard(tmp_path, lcs, pd.DataFrame(trows),
                {"stage": "acquire", "shard": [0, 1], "n_targets": 10, "n_fetched": 10,
                 "n_failed": 0, "n_empty": 0, "n_resumed": 0, "truncated": False,
                 "flag_source": "config", "service_probe_ok": True,
                 "verdict": "LIGHTCURVES_FETCHED"})
    back, meta = load_shard_lcs(tmp_path / "shards" / "0_of_1")
    assert len(back) == 10 and back["0"].n_det == lcs["0"].n_det
    rep = stage_screen(_conf(), tmp_path, (0, 1))
    assert rep["n_screened_now"] == 10 and rep["n_cessation"] >= 1
    s = stage_assess(_conf(), tmp_path, confirm=True, gaia=False)
    f = s["funnel"]
    assert f["n_usable"] == 10 and f["n_with_period"] == 3
    assert f["n_cess_candidates_confirmed"] >= 1
    assert f["n_fade_candidates_corrected"] == 0
    assert s["ensemble"]["annual_common_mode_cells"] > 0
    assert s["verdict_code"] in ("SURVIVORS_FOR_FOLLOWUP", "CANDIDATES_ALL_TRACED")
    assert (tmp_path / "candidates.csv").exists()
    # Re-running the screen skips what is done (checkpoint semantics).
    rep2 = stage_screen(_conf(), tmp_path, (0, 1))
    assert rep2["n_screened_now"] == 0 and rep2["n_previously_done"] == 10


def test_screen_stops_on_its_wall_clock_and_resumes(tmp_path):
    """The sweep job shares one ``timeout-minutes`` between acquire and screen.

    A censored injection-efficiency measurement in every block of a
    century-long light curve costs of order a minute per periodic star, so a
    shard that drew more variables than expected must stop on its own clock
    and upload what it has --- not be killed by the runner with nothing
    written.  ``screen.jsonl`` is appended per star, so the next dispatch
    resumes instead of restarting.
    """
    lcs, trows = {}, []
    for i in range(6):
        lc = synth_plates(np.random.default_rng(600 + i), period=0.57, amp=0.5,
                          n_per_year=25, lim_mean=14.0, mag0=11.0 + 0.05 * i,
                          series_switch_year=1910.0)
        lcs[str(i)] = lc
        trows.append({"target_id": i, "name": f"s{i}", "ra": 10.0 + i * 0.01, "dec": 20.0,
                      "kind": "variable", "vtype": "RRAB", "period_cat": 0.57,
                      "mag_cat": 11.0 + 0.05 * i, "amp_cat": 0.5, "source": "vsx",
                      "field": "f", "gsc_bin_index": 1, "ref_number": i,
                      "pm_total_masyr": np.nan, "colour": np.nan, "n_det_cat": np.nan})
    _fake_shard(tmp_path, lcs, pd.DataFrame(trows),
                {"stage": "acquire", "shard": [0, 1], "n_targets": 6, "n_fetched": 6,
                 "n_failed": 0, "n_empty": 0, "n_resumed": 0, "truncated": False,
                 "flag_source": "config", "service_probe_ok": True,
                 "verdict": "LIGHTCURVES_FETCHED"})
    # A budget the first star alone exceeds.  One star is screened anyway ---
    # a clock that can stop a run before its first star is a shard that never
    # finishes, however many re-dispatches it is given.
    rep = stage_screen(_conf(), tmp_path, (0, 1), time_budget_s=1e-6)
    assert rep["n_screened_now"] == 1 and rep["truncated"] is True
    assert rep["n_not_screened"] == 5
    assert (tmp_path / "shards" / "0_of_1" / "screen.jsonl").exists()
    # The next dispatch resumes from the checkpoint rather than restarting.
    rep2 = stage_screen(_conf(), tmp_path, (0, 1), time_budget_s=0)
    assert rep2["n_previously_done"] == 1 and rep2["n_screened_now"] == 5
    assert rep2["truncated"] is False and rep2["n_not_screened"] == 0
