"""Offline tests for TOCSIN's alternative feeds (ASAS-SN Sky Patrol v2, ATLAS).

No network, per ``docs/channel-brief.md`` §5 --- and here that is not a style rule
but the operating condition: neither service is reachable from the sandbox at
all, so every column name in ``altfeeds.py`` is documentation-derived and the
live shapes are recorded on the runner by ``seti tocsin-altfeeds-probe``.  These
tests therefore pin the things that are *decidable* offline, which is most of
what can go wrong:

* **units** --- JD versus MJD, mJy versus uJy versus nJy.  Every one of these
  errors produces a plausible light curve rather than an exception, and a
  uniform flux-scale error is invisible to every ratio the channel computes.
* **the denominator** --- a quiet star must yield visits and no events.  If that
  ever inverts, the ledger's rate pins at 1.0 and no target can ever be
  promoted, which is the failure the Rubin path spent a whole backfill
  discovering (``docs/tocsin.md`` §3).
* **the band-label trap** --- native bands are carried under LSST labels so the
  shared schema accepts them, and there are exactly three places where the label
  taken literally would produce a wrong number.  One test per place.
* **proper motion** --- forced photometry at one fixed coordinate over a decade
  walks off a nearby star and manufactures dips.  Both the segmentation and the
  exclusion are pinned.
* **honest degradation** --- a missing token, a missing baseline, a missing target
  list and an unusable band each produce a named verdict, never a silent pass.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from seti.tocsin import altfeeds as A
from seti.tocsin.photometry import ab_to_njy, njy_to_ab
from seti.tocsin.schema import validate
from seti.tocsin.screen import BAND_ORDER

MJD0 = 60000.0          # arbitrary but inside both surveys' baselines
RNG = np.random.default_rng(20260825)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _targets(n=1, source_id="777", ra=150.0, dec=-30.0, pmra=0.0, pmdec=0.0,
             g=14.0, r=13.2, i=12.9):
    """A one-row Gaia-like target table with GSPC synthetic photometry."""
    rows = []
    for k in range(n):
        rows.append({
            "source_id": source_id if n == 1 else f"{source_id}{k}",
            "ra": ra + 0.5 * k, "dec": dec, "ra_error": 0.02, "dec_error": 0.02,
            "pmra": pmra, "pmdec": pmdec, "pmra_error": 0.05, "pmdec_error": 0.05,
            "parallax": 25.0, "phot_g_mean_mag": g - 0.4, "bp_rp": 2.4,
            "u_sdss_mag": g + 1.6, "g_sdss_mag": g, "r_sdss_mag": r,
            "i_sdss_mag": i, "z_sdss_mag": i - 0.2, "y_ps1_mag": i - 0.35,
        })
    return pd.DataFrame(rows)


def _asassn_rows(n=200, mag=14.0, band="g", noise_frac=0.01, events=None,
                 t0=MJD0, cadence=1.0, seed=1):
    """Synthetic Sky Patrol v2 rows: total flux in mJy, time in JD.

    ``events`` maps epoch index -> fractional amplitude, so a test can inject a
    grey dip or flash of a known size and demand it back.
    """
    rng = np.random.default_rng(seed)
    f_star_njy = float(ab_to_njy(mag))
    out = []
    for k in range(n):
        amp = float((events or {}).get(k, 0.0))
        f = f_star_njy * (1.0 + amp) * (1.0 + noise_frac * rng.normal())
        out.append({
            "jd": t0 + k * cadence + A.JD_MINUS_MJD,
            "flux": f / A.MJY_TO_NJY,
            "flux_err": f_star_njy * noise_frac / A.MJY_TO_NJY,
            "mag": float(njy_to_ab(f)),
            "mag_err": 1.0857 * noise_frac,
            "limit": mag + 3.0,
            "phot_filter": band,
            "quality": "G",
            "fwhm": 1.7,
        })
    return out


def _atlas_text(n=200, band="o", dflux_ujy=0.0, event_idx=None, event_ujy=0.0,
                noise_ujy=5.0, t0=MJD0, seed=2, extra_col=False):
    """A synthetic ATLAS forced-photometry result file, header included."""
    rng = np.random.default_rng(seed)
    cols = ["MJD", "m", "dm", "uJy", "duJy", "F", "err", "chi/N", "RA", "Dec",
            "x", "y", "maj", "min", "phi", "apfit", "mag5sig", "Sky", "Obs"]
    if extra_col:
        cols.insert(8, "newthing")
    lines = ["###" + "  ".join(cols)]
    for k in range(n):
        f = dflux_ujy + noise_ujy * rng.normal()
        if event_idx is not None and k == event_idx:
            f = event_ujy
        m = float(njy_to_ab(abs(f) * A.UJY_TO_NJY)) if f != 0 else 99.0
        vals = {
            "MJD": f"{t0 + k:.5f}", "m": f"{m:.4f}", "dm": "0.05",
            "uJy": f"{f:.3f}", "duJy": f"{noise_ujy:.3f}", "F": band,
            "err": "0", "chi/N": "1.02", "RA": "150.0", "Dec": "-30.0",
            "x": "100.0", "y": "100.0", "maj": "2.0", "min": "1.9",
            "phi": "10.0", "apfit": "-0.1", "mag5sig": "19.0", "Sky": "20.0",
            "Obs": f"o{k:05d}o", "newthing": "0",
        }
        lines.append(" ".join(vals[c] for c in cols))
    return "\n".join(lines) + "\n"


def _lc_from_asassn(rows, tid="777", ra=150.0, dec=-30.0):
    return A.asassn_rows_to_lightcurve(rows, tid, ra, dec)


# ---------------------------------------------------------------------------
# units --- every one of these fails silently rather than loudly
# ---------------------------------------------------------------------------
def test_asassn_julian_dates_are_converted_to_mjd():
    lc = _lc_from_asassn(_asassn_rows(n=30))
    # Left unconverted the epochs land 2.4 million days in the future, which the
    # ledger would accept as a perfectly valid and permanently disjoint set of
    # nights rather than reject.
    assert abs(float(lc.mjd[0]) - MJD0) < 1e-6
    assert float(lc.mjd[-1]) < 70000.0


def test_a_time_column_already_in_mjd_is_not_converted_twice():
    rows = _asassn_rows(n=30)
    for r in rows:
        r["mjd"] = r.pop("jd") - A.JD_MINUS_MJD
    lc = A.asassn_rows_to_lightcurve(rows, "777", 150.0, -30.0)
    assert abs(float(lc.mjd[0]) - MJD0) < 1e-6
    assert any("treated_as_mjd" in n for n in lc.notes)


def test_asassn_millijansky_scale_is_applied():
    mag = 14.0
    lc = _lc_from_asassn(_asassn_rows(n=60, mag=mag, noise_frac=0.0))
    assert abs(float(np.median(lc.flux_njy)) - float(ab_to_njy(mag))) / ab_to_njy(mag) < 1e-6


def test_atlas_microjansky_scale_is_applied():
    _cols, rows = A.parse_atlas_text(_atlas_text(n=40, dflux_ujy=100.0, noise_ujy=0.0))
    lc = A.atlas_rows_to_lightcurve(rows, "777", 150.0, -30.0)
    assert abs(float(np.median(lc.flux_njy)) - 100.0 * A.UJY_TO_NJY) < 1e-6


def test_a_wrong_flux_unit_is_detected_against_the_served_magnitudes():
    # A uniform scale error cancels in dF/F* and would never announce itself, so
    # the assumed unit is checked against the survey's own magnitude column.
    rows = _asassn_rows(n=60, noise_frac=0.0)
    for r in rows:
        r["flux"] = r["flux"] * 1000.0        # the service switched to uJy
    lc = A.asassn_rows_to_lightcurve(rows, "777", 150.0, -30.0)
    assert any(n.startswith("flux_unit_mismatch") for n in lc.notes)


def test_the_correct_flux_unit_raises_no_complaint():
    lc = _lc_from_asassn(_asassn_rows(n=60, noise_frac=0.0))
    assert not any(n.startswith("flux_unit_mismatch") for n in lc.notes)


def test_implied_flux_unit_returns_nan_when_it_cannot_decide():
    assert math.isnan(A.implied_flux_unit_njy(np.array([15.0]), np.array([1.0])))


# ---------------------------------------------------------------------------
# parsers --- name-keyed, so a re-ordering costs a field and not the file
# ---------------------------------------------------------------------------
def test_atlas_parser_reads_the_header_and_types_nothing():
    cols, rows = A.parse_atlas_text(_atlas_text(n=5))
    assert cols[:6] == ["mjd", "m", "dm", "ujy", "dujy", "f"]
    assert "chi_n" in cols                      # '/' is not a legal key
    assert len(rows) == 5
    assert isinstance(rows[0]["ujy"], str)


def test_atlas_parser_is_keyed_by_name_not_by_position():
    # An inserted column must not shift every field after it.  A position-keyed
    # parse would read RA into `x`, `maj` into `phi`, and the flux column would
    # still parse --- producing a light curve rather than an error.
    _c, rows = A.parse_atlas_text(_atlas_text(n=6, dflux_ujy=42.0, noise_ujy=0.0,
                                              extra_col=True))
    assert all(abs(float(r["ujy"]) - 42.0) < 1e-6 for r in rows)
    assert all(r["f"] == "o" for r in rows)
    assert all(abs(float(r["mag5sig"]) - 19.0) < 1e-6 for r in rows)


def test_atlas_rows_in_an_unknown_filter_are_dropped():
    text = _atlas_text(n=10, band="z")          # not an ATLAS filter
    _c, rows = A.parse_atlas_text(text)
    lc = A.atlas_rows_to_lightcurve(rows, "777", 150.0, -30.0)
    assert len(lc) == 0


def test_asassn_v_band_epochs_leave_both_the_numerator_and_the_denominator():
    rows = _asassn_rows(n=40, band="g") + _asassn_rows(n=40, band="V", seed=9)
    lc = A.asassn_rows_to_lightcurve(rows, "777", 150.0, -30.0)
    assert len(lc) == 40
    assert any("non_native_bands" in n for n in lc.notes)
    # Keeping V epochs as trials while they can never produce an event would
    # inflate the denominator, deflate the ensemble rate and make every
    # per-target p-value too small --- the anti-conservative direction.
    red = A.reduce_lightcurve(lc, A.ASASSN)
    assert len(red.visit_mjds) == 40


def test_a_zero_flux_epoch_is_not_read_as_a_missing_one():
    rows = _asassn_rows(n=30, noise_frac=0.0)
    rows[5]["flux"] = 0.0
    lc = A.asassn_rows_to_lightcurve(rows, "777", 150.0, -30.0)
    assert float(lc.flux_njy[5]) == 0.0
    assert np.isfinite(lc.flux_njy[5])


# ---------------------------------------------------------------------------
# the robust baseline
# ---------------------------------------------------------------------------
def test_robust_baseline_is_not_moved_by_an_injected_event():
    f = np.full(200, 1000.0) + RNG.normal(scale=10.0, size=200)
    clean = A.robust_baseline(f)
    f[17] = 5000.0                              # a 400 % flash
    f[93] = 100.0                               # a 90 % dip
    dirty = A.robust_baseline(f)
    assert abs(dirty.level - clean.level) < 5.0
    assert dirty.n_used < dirty.n_total


def test_clipping_is_symmetric_so_neither_polarity_is_favoured():
    base = np.full(200, 1000.0) + RNG.normal(scale=10.0, size=200)
    up, down = base.copy(), base.copy()
    up[:5] = 5000.0
    down[:5] = 100.0
    assert abs(A.robust_baseline(up).level - A.robust_baseline(down).level) < 6.0


def test_a_degenerate_lightcurve_is_refused_not_given_zero_scatter():
    # Zero MAD would make every epoch infinitely significant.
    b = A.robust_baseline(np.full(50, 7.0))
    assert not b.ok and b.reason == "degenerate_scatter"


def test_measured_scatter_beats_optimistic_formal_errors():
    # Formal errors 100x too small: the epoch-to-epoch scatter is the truth, and
    # believing the error bars instead would call ordinary systematics events.
    rows = _asassn_rows(n=200, noise_frac=0.02)
    for r in rows:
        r["flux_err"] = r["flux_err"] / 100.0
    red = A.reduce_lightcurve(A.asassn_rows_to_lightcurve(rows, "777", 150.0, -30.0),
                              A.ASASSN)
    assert red.bands["g"].usable
    assert red.bands["g"].n_events == 0


# ---------------------------------------------------------------------------
# the reduction: numerator and denominator from the same pass
# ---------------------------------------------------------------------------
def test_a_quiet_star_yields_visits_and_no_events():
    """The property the whole ledger rests on.

    Alerts exist only where there was a detection, which is why the Rubin path
    had to reconstruct its denominator from the observed footprint.  A
    forced-photometry feed answers directly --- and if it ever stopped doing so,
    the ensemble rate would pin at 1.0 and no target could ever be promoted.
    """
    red = A.reduce_lightcurve(_lc_from_asassn(_asassn_rows(n=300)), A.ASASSN)
    assert red.usable
    assert len(red.visit_mjds) == 300
    assert red.alerts == []


def test_an_injected_dip_is_recovered_with_the_right_amplitude():
    rows = _asassn_rows(n=300, noise_frac=0.01, events={120: -0.25})
    red = A.reduce_lightcurve(_lc_from_asassn(rows), A.ASASSN)
    assert len(red.alerts) == 1
    a = red.alerts[0]
    assert a.polarity == "dip"
    assert abs(a.dflux_njy / a.template_flux_njy + 0.25) < 0.05
    assert len(red.visit_mjds) == 300           # the trial is still counted


def test_an_injected_flash_is_recovered():
    rows = _asassn_rows(n=300, noise_frac=0.01, events={44: 0.30})
    red = A.reduce_lightcurve(_lc_from_asassn(rows), A.ASASSN)
    assert [x.polarity for x in red.alerts] == ["flash"]


def test_a_band_with_too_few_epochs_contributes_neither_events_nor_visits():
    red = A.reduce_lightcurve(_lc_from_asassn(_asassn_rows(n=5, events={2: 5.0})),
                              A.ASASSN)
    assert not red.usable
    assert red.alerts == [] and red.visit_mjds == []
    assert "fewer_than" in red.bands["g"].reason


def test_a_difference_feed_with_no_quiescent_flux_refuses_to_emit():
    """The band-label trap, closed at its source.

    ATLAS *c* is carried under the label *g*.  If an alert reached the funnel
    without a template flux, ``screen._baseline_flux`` would divide by the Gaia
    GSPC *g* magnitude --- a genuinely different passband, and on a red dwarf a
    substantially different flux.  Refusing is the only safe answer.
    """
    _c, rows = A.parse_atlas_text(_atlas_text(n=200, event_idx=50, event_ujy=800.0))
    lc = A.atlas_rows_to_lightcurve(rows, "777", 150.0, -30.0)
    red = A.reduce_lightcurve(lc, A.ATLAS, quiescent=None)
    assert red.alerts == [] and red.visit_mjds == []
    assert red.bands["o"].reason == "no_quiescent_flux"


def test_the_same_atlas_lightcurve_works_once_a_native_baseline_is_supplied():
    _c, rows = A.parse_atlas_text(_atlas_text(n=200, event_idx=50, event_ujy=800.0))
    lc = A.atlas_rows_to_lightcurve(rows, "777", 150.0, -30.0)
    q = {"o": A.Quiescent(4000.0 * A.UJY_TO_NJY, 40.0 * A.UJY_TO_NJY,
                          "atlas_reduced_images")}
    red = A.reduce_lightcurve(lc, A.ATLAS, quiescent=q)
    assert len(red.alerts) == 1
    a = red.alerts[0]
    assert a.raw["baseline_source"] == "atlas_reduced_images"
    assert a.template_flux_njy == pytest.approx(4000.0 * A.UJY_TO_NJY)
    # 800 uJy on a 4000 uJy star is a 20 % flash.
    assert a.dflux_njy / a.template_flux_njy == pytest.approx(0.2, abs=0.02)
    assert len(red.visit_mjds) == 200


def test_atlas_uses_the_exposures_own_five_sigma_limit_when_it_is_served():
    _c, rows = A.parse_atlas_text(_atlas_text(n=100))
    lc = A.atlas_rows_to_lightcurve(rows, "777", 150.0, -30.0)
    q = {"o": A.Quiescent(4000.0 * A.UJY_TO_NJY, 40.0 * A.UJY_TO_NJY, "x")}
    red = A.reduce_lightcurve(lc, A.ATLAS, quiescent=q)
    assert abs(red.bands["o"].limit_median_njy - float(ab_to_njy(19.0))) < 1.0


def test_a_bad_error_code_epoch_is_dropped():
    text = _atlas_text(n=100)
    lines = text.splitlines()
    parts = lines[10].split()
    parts[6] = "3"                              # err != 0
    lines[10] = " ".join(parts)
    _c, rows = A.parse_atlas_text("\n".join(lines))
    lc = A.atlas_rows_to_lightcurve(rows, "777", 150.0, -30.0)
    assert int(np.sum(~lc.good)) == 1


# ---------------------------------------------------------------------------
# band labels: the schema accepts them, and the label never lies
# ---------------------------------------------------------------------------
def test_native_bands_are_carried_under_labels_the_shared_schema_accepts():
    rows = _asassn_rows(n=300, events={10: 0.4})
    red = A.reduce_lightcurve(_lc_from_asassn(rows), A.ASASSN)
    assert red.alerts and validate(red.alerts[0]) == []
    assert red.alerts[0].band == "g"
    assert red.alerts[0].raw["native_band"] == "g"


def test_atlas_c_and_o_are_labelled_blue_then_red():
    assert A.ATLAS.band_label["c"] == "g" and A.ATLAS.band_label["o"] == "r"
    assert BAND_ORDER.index("g") < BAND_ORDER.index("r")
    assert A.ATLAS.band_wl_um["c"] < A.ATLAS.band_wl_um["o"]


def test_every_emitted_alert_carries_a_native_template_flux():
    """So ``screen._baseline_flux`` never reaches its GSPC-by-label branch."""
    rows = _asassn_rows(n=300, events={10: 0.4, 200: -0.3})
    red = A.reduce_lightcurve(_lc_from_asassn(rows), A.ASASSN)
    assert red.alerts
    for a in red.alerts:
        assert a.template_flux_njy is not None and a.template_flux_njy > 0


def test_label_inversion_recovers_the_native_band():
    assert A._labels_to_native(["g", "r"], A.ATLAS) == ["c", "o"]
    assert A._labels_to_native(["g"], A.ASASSN) == ["g"]


# ---------------------------------------------------------------------------
# native-passband physics
# ---------------------------------------------------------------------------
def test_native_colour_temperature_recovers_an_injected_blackbody():
    from seti.tocsin.photometry import planck_nu
    t_true = 3400.0
    bands = ["c", "o"]
    f = [1e4 * planck_nu(A.ATLAS.band_wl_um[b], t_true) /
         planck_nu(A.ATLAS.band_wl_um["o"], t_true) for b in bands]
    fit = A.native_colour_temperature(bands, A.ATLAS.band_wl_um, f,
                                      [x * 0.01 for x in f])
    assert fit.ok
    assert 0.7 * t_true < fit.temp_k < 1.4 * t_true


def test_the_lsst_label_wavelengths_give_a_different_answer():
    """Which is exactly why the fit is redone at the native wavelengths."""
    from seti.tocsin.photometry import blackbody_colour_temperature, planck_nu
    t_true = 3400.0
    f = [1e4 * planck_nu(A.ATLAS.band_wl_um[b], t_true) /
         planck_nu(A.ATLAS.band_wl_um["o"], t_true) for b in ("c", "o")]
    e = [x * 0.01 for x in f]
    native = A.native_colour_temperature(["c", "o"], A.ATLAS.band_wl_um, f, e)
    labelled = blackbody_colour_temperature(["g", "r"], f, e)
    assert native.ok and labelled.ok
    assert abs(labelled.temp_k - native.temp_k) / native.temp_k > 0.1


def test_negative_flux_has_no_emission_temperature():
    fit = A.native_colour_temperature(["c", "o"], A.ATLAS.band_wl_um,
                                      [-10.0, -12.0], [1.0, 1.0])
    assert not fit.ok and "negative_flux" in fit.reason


def test_native_nondetection_test_uses_the_other_bands_own_baseline():
    """The third place the label could lie, and the reason it is re-run here.

    A grey event has equal *fractional* amplitude in every band, so on a red star
    it is brighter in absolute flux in the redder band.  Feeding the test the
    detected band's baseline instead --- which is what ``screen`` does, because
    ``_baseline_flux`` ignores the band it is handed whenever a template flux is
    present --- changes the predicted flux by the star's own colour and therefore
    changes the verdict.
    """
    a_obs = 0.20
    f_blue, f_red = 1.0e4, 4.0e4               # a red star: F*_r = 4 F*_g
    limit = 1.0e3
    right = A.native_nondetection_test(a_obs, "o", f_red, limit)
    wrong = A.native_nondetection_test(a_obs, "o", f_blue, limit)
    assert right.tested and right.excluded     # 0.2 * 4e4 = 8000 >> 3 * 1000
    assert not wrong.excluded                  # 0.2 * 1e4 = 2000 < 3 * 1000
    assert wrong.reason == "prediction_below_detection_limit"


def test_a_marginal_prediction_is_untestable_rather_than_passed():
    gx = A.native_nondetection_test(0.05, "o", 1.0e4, 1.0e3)
    assert not gx.excluded and not gx.tested


# ---------------------------------------------------------------------------
# proper motion
# ---------------------------------------------------------------------------
def test_pm_segments_keep_the_aperture_on_a_fast_mover():
    # 1000 mas/yr over ten years is 10 arcsec, two ATLAS PSFs.
    segs = A.pm_segments(150.0, -30.0, 1000.0, 0.0, MJD0, MJD0 + 3652.5, A.ATLAS)
    assert len(segs) > 1
    allowance = A.LightCurveThresholds().max_drift_frac * A.ATLAS.psf_fwhm_arcsec
    assert all(s["drift_arcsec"] <= allowance + 1e-6 for s in segs)
    # and each segment is requested at its own propagated position
    assert segs[-1]["ra"] != pytest.approx(segs[0]["ra"])


def test_a_stationary_star_needs_only_one_segment():
    segs = A.pm_segments(150.0, -30.0, 0.0, 0.0, MJD0, MJD0 + 3652.5, A.ATLAS)
    assert len(segs) == 1


def test_a_fixed_aperture_survey_cannot_segment_and_excludes_the_fast_mover():
    # ASAS-SN photometers its own catalogued position, so segmentation is not
    # available and the honest response is to drop the target from BOTH the
    # numerator and the denominator.
    assert len(A.pm_segments(150.0, -30.0, 1000.0, 0.0, MJD0, MJD0 + 3652.5,
                             A.ASASSN)) == 1
    assert A.drift_excluded(1000.0, 0.0, MJD0, MJD0 + 3652.5, A.ASASSN)
    assert not A.drift_excluded(10.0, 0.0, MJD0, MJD0 + 3652.5, A.ASASSN)
    # ATLAS is never excluded on drift: it is segmented instead.
    assert not A.drift_excluded(1000.0, 0.0, MJD0, MJD0 + 3652.5, A.ATLAS)


def test_proper_motion_drift_uses_the_mu_alpha_star_convention():
    # pmra already carries cos(dec); treating it otherwise would understate the
    # drift for southern targets, which is most of this list.
    assert A.pm_drift_arcsec(3000.0, 4000.0, 1.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# depth: the number that decides whether any of this is worth running
# ---------------------------------------------------------------------------
def test_reachable_fraction_is_much_harsher_than_the_headline_depth():
    t = _targets(g=17.0, r=16.2, i=15.9)
    rec = A.reachable_fraction(t, A.ASASSN)
    band = rec["bands"]["g"]
    # g = 17 is inside the nominal g <~ 18, and still cannot carry a 10 % event.
    assert band["nominal_depth_5sigma"] == 18.0
    assert band["by_amplitude"]["0.1"]["mag_cut"] < 16.0
    assert band["by_amplitude"]["0.1"]["n_reachable"] == 0
    # A 100 % event on the same star is reachable.
    assert band["by_amplitude"]["1"]["n_reachable"] == 1
    assert band["n_saturated"] == 0


def test_the_magnitude_cut_follows_the_stated_formula():
    rec = A.reachable_fraction(_targets(), A.ASASSN, amplitudes=(0.1,), n_sigma=6.0)
    cut = rec["bands"]["g"]["by_amplitude"]["0.1"]["mag_cut"]
    assert cut == pytest.approx(18.0 - 2.5 * math.log10(6.0 / 0.5), abs=1e-3)


def test_targets_without_synthetic_photometry_are_counted_not_assumed():
    t = _targets()
    t = t.drop(columns=["g_sdss_mag"])
    rec = A.reachable_fraction(t, A.ASASSN)
    assert rec["bands"]["g"]["n_without_synthetic_photometry"] == 1
    assert rec["bands"]["g"]["by_amplitude"]["0.1"]["n_reachable"] == 0


def test_synthetic_native_mag_interpolates_between_the_bracketing_bands():
    t = _targets(g=15.0, r=14.0, i=13.5)
    c = float(A.synthetic_native_mag(t, A.ATLAS, "c")[0])
    o = float(A.synthetic_native_mag(t, A.ATLAS, "o")[0])
    assert 14.0 < c < 15.0          # c sits between SDSS g and r
    assert 13.5 < o < 14.0          # o sits between SDSS r and i
    # and the interpolation is monotone with the input colour
    redder = _targets(g=15.0, r=13.5, i=12.8)
    assert float(A.synthetic_native_mag(redder, A.ATLAS, "o")[0]) < o


def test_asassn_native_mag_is_the_gspc_column_itself():
    t = _targets(g=15.5)
    assert float(A.synthetic_native_mag(t, A.ASASSN, "g")[0]) == pytest.approx(15.5)


# ---------------------------------------------------------------------------
# blending: what replaces the astrometric discriminator
# ---------------------------------------------------------------------------
def test_a_comparable_neighbour_inside_the_aperture_is_flagged():
    # ASAS-SN's 16 arcsec aperture routinely contains other stars, and with no
    # centroid there is nothing to reject a flare on one of them.
    cat_ra = np.array([150.0, 150.0 + 5.0 / 3600.0, 150.0 + 60.0 / 3600.0])
    cat_dec = np.array([-30.0, -30.0, -30.0])
    cat_mag = np.array([14.0, 14.5, 12.0])
    rec = A.blend_neighbours([150.0], [-30.0], [14.0], cat_ra, cat_dec, cat_mag,
                             A.ASASSN.aperture_radius_arcsec)
    assert rec["n_with_any_neighbour"] == 1
    assert rec["blend_ratio"][0] > 0.5          # the 14.5 mag neighbour
    assert rec["n_with_blend_ratio_gt_0.1"] == 1


def test_an_isolated_star_has_no_blend():
    rec = A.blend_neighbours([150.0], [-30.0], [14.0],
                             np.array([150.0]), np.array([-30.0]),
                             np.array([14.0]), A.ATLAS.aperture_radius_arcsec)
    assert rec["n_with_any_neighbour"] == 0
    assert rec["blend_ratio"][0] == 0.0


# ---------------------------------------------------------------------------
# end to end through TOCSIN's own funnel
# ---------------------------------------------------------------------------
def _screen_one(events, targets=None, n=300, noise_frac=0.01):
    targets = _targets() if targets is None else targets
    rows = _asassn_rows(n=n, mag=14.0, noise_frac=noise_frac, events=events)
    lc = _lc_from_asassn(rows, tid="777",
                         ra=float(targets["ra"].iloc[0]),
                         dec=float(targets["dec"].iloc[0]))
    red = A.reduce_lightcurve(lc, A.ASASSN)
    th = A.funnel_thresholds(A.ASASSN)
    return A.screen_lightcurves([red], targets, A.ASASSN, th)


def test_the_funnel_runs_and_produces_events_over_a_real_denominator():
    v = _screen_one({30: 0.35, 150: -0.30})
    assert len(v.events) == 2
    assert len(v.star_night_pairs) == 300
    assert sum(v.trials_by_night.values()) == 300
    assert {e.polarity for e in v.events} == {"flash", "dip"}
    # The stratified null needs per-bin trials keyed by night, exactly as the
    # Rubin path builds them; empty here would silently fall back to the all-sky
    # rate the stratification exists to replace.
    assert v.bin_trials_by_night
    assert all(sum(b.values()) >= 1 for b in v.bin_trials_by_night.values())


def test_a_quiet_star_screens_to_zero_events_and_three_hundred_trials():
    v = _screen_one({})
    assert v.events == []
    assert sum(v.trials_by_night.values()) == 300


def test_every_event_names_the_discriminators_this_feed_cannot_supply():
    """A rule that did not fire must never read as a rule that passed."""
    v = _screen_one({30: 0.35})
    reasons = set(v.events[0].reasons)
    assert "feed_asassn" in reasons
    assert "astrometry_not_independent" in reasons
    assert "greyness_unavailable_single_band_survey" in reasons
    for missing in ("reliability", "isdipole", "glint_trail", "extendedness",
                    "ss_association"):
        assert f"{missing}_unavailable_in_asassn" in reasons


def test_the_visit_history_reaches_the_ledger_and_the_denominator_is_exact(tmp_path):
    v = _screen_one({30: 0.35, 150: 0.40})
    stats = A.fold(v, tmp_path / "ledger_asassn.json", targets_n=1)
    led = json.loads((tmp_path / "ledger_asassn.json").read_text())
    rec = led["targets"]["777"]
    assert rec["n_events"] == 2
    assert rec["visits_exact"] is True
    assert rec["n_visits"] == 300               # every epoch, not just the events
    assert stats["cumulative_target_visits"] == 300


def test_each_survey_gets_its_own_ledger(tmp_path):
    """Rubin star-nights and ASAS-SN star-nights are not the same trial.

    Pouring one survey's visits into another's denominator would produce a rate
    of nothing: different depth, different cadence, different systematics and a
    different minimum detectable amplitude.
    """
    v = _screen_one({30: 0.35})
    p1 = tmp_path / "ledger_asassn.json"
    p2 = tmp_path / "ledger_atlas.json"
    A.fold(v, p1, targets_n=1)
    assert p1.exists() and not p2.exists()


def test_two_bands_on_one_night_are_one_event_not_two():
    # The event unit is the star-night; ATLAS c and o on the same night are one
    # event measured twice, and counting both would corrupt the multiplicity.
    t = _targets()
    lc = A.LightCurve(
        target_id="777", ra=150.0, dec=-30.0, survey="atlas",
        mjd=np.concatenate([np.arange(200.0) + MJD0, np.arange(200.0) + MJD0]),
        flux_njy=np.concatenate([RNG.normal(0, 500, 200), RNG.normal(0, 500, 200)]),
        flux_err_njy=np.full(400, 500.0),
        band=np.array(["c"] * 200 + ["o"] * 200, dtype=object))
    lc.flux_njy[50] = 3.0e4
    lc.flux_njy[250] = 3.0e4                    # same night, the other filter
    q = {"c": A.Quiescent(1.0e5, 1.0e3, "atlas_reduced_images"),
         "o": A.Quiescent(1.0e5, 1.0e3, "atlas_reduced_images")}
    red = A.reduce_lightcurve(lc, A.ATLAS, quiescent=q)
    assert len(red.alerts) == 2
    v = A.screen_lightcurves([red], t, A.ATLAS, A.funnel_thresholds(A.ATLAS))
    assert len(v.events) == 1
    assert sorted(v.events[0].bands) == ["g", "r"]
    assert v.counts["star_nights_two_band"] == 200


def test_the_two_band_night_fraction_is_measured_not_assumed():
    v = _screen_one({})
    assert v.counts["two_band_night_fraction"] == 0.0    # ASAS-SN is single band


# ---------------------------------------------------------------------------
# honest degradation
# ---------------------------------------------------------------------------
def test_atlas_reports_no_token_rather_than_failing(monkeypatch):
    for k in ("ATLAS_TOKEN", "ATLAS_USERNAME", "ATLAS_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    client = A.AtlasForcedPhotometry()
    assert client.available is False
    assert client.ensure_token() is None
    with pytest.raises(A.AltFeedError):
        client.submit(150.0, -30.0)


def test_atlas_reads_its_token_from_the_environment(monkeypatch):
    monkeypatch.setenv("ATLAS_TOKEN", "deadbeef")
    assert A.AtlasForcedPhotometry().available is True
    assert A.ATLAS.auth_env == "ATLAS_TOKEN"


def test_asassn_needs_no_token_at_all():
    assert A.ASASSN.auth_env is None


def test_an_unknown_survey_is_an_error_not_a_default():
    # `ztf` was the example of an unknown survey here until 2026-08-26, when it
    # became a real one.  A feed that silently defaults to another feed's spec
    # would screen the wrong depths against the right targets.
    with pytest.raises(A.AltFeedError):
        A.run_survey("gaia-alerts")
    assert set(A.SURVEYS) == {"asassn", "atlas", "ztf"}


def test_a_missing_target_list_gives_a_named_verdict(tmp_path):
    rec = A.run_survey("asassn", targets_path=tmp_path / "nope.parquet",
                       out_dir=tmp_path / "out")
    assert rec["verdict"] == "NO_TARGET_LIST"
    assert (tmp_path / "out" / "summary.json").exists()


def test_the_census_says_how_to_get_the_target_list_when_it_is_absent(tmp_path):
    rec = A.census(targets_path=tmp_path / "nope.parquet", out_dir=tmp_path)
    assert rec["verdict"] == "NO_TARGET_LIST"
    assert "tocsin-targets" in rec["how_to_fix"]


def test_the_census_runs_offline_from_a_cached_target_list(tmp_path):
    t = _targets(n=3, g=14.5, r=13.9, i=13.6)
    p = tmp_path / "targets.parquet"
    t.to_parquet(p)
    rec = A.census(targets_path=p, out_dir=tmp_path)
    assert rec["verdict"] == "OK" and rec["n_targets"] == 3
    assert set(rec["surveys"]) == {"asassn", "atlas", "ztf"}
    assert rec["surveys"]["atlas"]["bands"]["o"]["by_amplitude"]["0.1"]["n_reachable"] == 3
    # These stars are brighter than Rubin's saturation, so they are exactly the
    # population the alert stream cannot screen at all.
    assert rec["surveys"]["atlas"]["bands"]["o"]["by_amplitude"]["0.1"][
        "n_reachable_and_rubin_saturated"] == 3
    assert (tmp_path / "census.json").exists()


def test_reach_is_a_window_so_a_saturating_star_is_not_counted_as_reachable():
    """Both feeds saturate, and the nearby-star list is richest at the bright end.

    Counting a star that saturates the detector as "reachable" would overstate
    the usable sample at precisely the end where a 100 pc catalogue has the most
    entries.
    """
    bright = _targets(g=9.0, r=8.4, i=8.1)
    rec = A.reachable_fraction(bright, A.ASASSN)
    band = rec["bands"]["g"]
    assert band["n_saturated"] == 1
    assert band["by_amplitude"]["0.1"]["n_reachable"] == 0
    assert band["by_amplitude"]["0.1"]["mag_window"][0] == A.ASASSN.saturation_mag["g"]


def test_the_complementary_population_is_counted_against_rubins_saturation():
    """The argument that makes these feeds more than a stopgap.

    Rubin saturates near r = 16 in a 30 s visit, and the 42 TOCSIN targets that
    produced events in the full Rubin walk have recovered quiescent magnitudes
    with NONE brighter than g = 16.5 --- the alert stream structurally cannot
    screen the bright half of a 100 pc sample.  Both alternative feeds can.
    """
    faint = _targets(g=18.0, r=17.4, i=17.1)      # Rubin's regime, not ATLAS's
    bright = _targets(g=14.5, r=13.9, i=13.6)     # inside ATLAS, saturating Rubin
    assert A.reachable_fraction(faint, A.ATLAS)["n_brighter_than_rubin_saturation"] == 0
    assert A.reachable_fraction(bright, A.ATLAS)["n_brighter_than_rubin_saturation"] == 1
    assert A.reachable_fraction(bright, A.ATLAS)["bands"]["o"]["by_amplitude"]["0.1"][
        "n_reachable_and_rubin_saturated"] == 1


def test_the_probe_writes_a_record_even_when_nothing_is_reachable(tmp_path, monkeypatch):
    """EVERY survey in the default set is stubbed, and that is the point.

    This test read the ambient network until 2026-08-27: adding ZTF to the
    probe's default surveys left its `describe` unpatched, so the test passed in
    a sandbox with no egress and FAILED on the runner, where IRSA answers and the
    verdict is PARTIAL rather than NO_FEED_REACHED.  It broke CI on `main` for
    six consecutive merges.  A test whose result depends on whether the machine
    running it can reach the internet is not testing this repository.
    """
    def boom(self):
        raise RuntimeError("no egress in the sandbox")
    for client in (A.AsasSnSkyPatrol, A.AtlasForcedPhotometry, A.ZtfIrsa):
        monkeypatch.setattr(client, "describe", boom)
    rec = A.probe(out_dir=tmp_path)
    assert rec["verdict"] == "NO_FEED_REACHED"
    assert (tmp_path / "probe.json").exists()
    assert "no egress" in rec["surveys"]["asassn"]["error"]
    assert set(rec["surveys"]) == set(A.SURVEYS), (
        "a survey missing from the probe's default set is a feed nobody probes")


def test_no_default_probe_survey_is_left_unstubbed():
    """The guard for the class, not the instance.

    A fourth feed added tomorrow would reintroduce exactly the same bug: the
    probe would reach a live service from inside the test suite, and the failure
    would appear only on a machine WITH network. Every client the probe can
    construct must be reachable from the test module by name.
    """
    for key in A.SURVEYS:
        assert key in {"asassn", "atlas", "ztf"}, (
            f"survey {key!r} is in SURVEYS but this test module does not know "
            f"how to stub its client; stub it above before adding it")


def test_the_signature_transfer_statement_is_carried_in_every_summary():
    st_atlas = A.signature_transfer(A.ATLAS)
    st_asassn = A.signature_transfer(A.ASASSN)
    # ATLAS's 30 s exposure equals a Rubin visit, so the sub-visit timescale
    # transfers; ASAS-SN's 3 x 90 s does not.
    assert st_atlas["sub_visit_timescale"].startswith("preserved")
    assert st_asassn["sub_visit_timescale"].startswith("degraded")
    assert st_asassn["achromaticity"].startswith("UNAVAILABLE")
    assert st_atlas["achromaticity"].startswith("available only")
    assert "PRESERVED" in st_atlas["recurrence_ledger"]
    assert st_atlas["astrometric_offset"].startswith("UNAVAILABLE")
    assert st_asassn["depth_penalty_mag_vs_rubin"] > 5.0


def test_json_writer_turns_non_finite_into_null_not_into_nan(tmp_path):
    A._write_json(tmp_path / "x.json", {"a": float("nan"), "b": True, "c": 3})
    rec = json.loads((tmp_path / "x.json").read_text())
    assert rec["a"] is None
    assert rec["b"] is True                     # bool must not become 1


def test_run_survey_drives_the_whole_stage_offline(tmp_path):
    """The orchestration, with the network stubbed out by supplying light curves.

    Exercises the path a runner takes: target list -> reduction -> funnel ->
    per-survey ledger -> committed artefacts, and checks that the artefacts say
    what the denominator was rather than leaving a reader to infer it.
    """
    t = _targets(g=13.0, r=12.4, i=12.1)
    tp = tmp_path / "targets.parquet"
    t.to_parquet(tp)
    lc = _lc_from_asassn(_asassn_rows(n=300, mag=13.0, events={40: 0.35, 200: 0.40}))
    rec = A.run_survey("asassn", targets_path=tp, out_dir=tmp_path / "out",
                       lightcurves={"777": lc})
    assert rec["verdict"] == "OK"
    assert rec["counts"]["target_nights_screened"] == 300
    assert rec["denominator"] == "forced_photometry_exact"
    assert rec["forced_coverage_fraction"] == 1.0
    assert rec["ledger"]["cumulative_target_visits"] == 300
    assert rec["signature_transfer"]["achromaticity"].startswith("UNAVAILABLE")
    assert rec["reachability"]["bands"]["g"]["nominal_depth_5sigma"] == 18.0
    out = tmp_path / "out"
    assert (out / "summary.json").exists()
    assert (out / "events.json").exists()
    assert (out / "ledger_asassn.json").exists()
    events = json.loads((out / "events.json").read_text())["events"]
    assert len(events) == 2
    assert all("astrometry_not_independent" in e["reasons"] for e in events)


def test_run_survey_records_the_reachability_census_even_with_no_data(tmp_path):
    t = _targets(g=20.0, r=19.5, i=19.2)          # far below either depth
    tp = tmp_path / "targets.parquet"
    t.to_parquet(tp)
    rec = A.run_survey("asassn", targets_path=tp, out_dir=tmp_path / "out",
                       lightcurves={})
    assert rec["verdict"] == "NO_USABLE_EPOCHS"
    assert rec["reachability"]["bands"]["g"]["by_amplitude"]["1"]["n_reachable"] == 0


def test_the_gspc_fallback_baseline_is_usable_and_flagged_as_a_transformation():
    """The documented fallback when ATLAS's reduced-image pass is unavailable.

    Without SOME F* the whole band is refused, so the alternative to this
    fallback is not a weaker result --- it is no result at all.  It is a
    cross-survey passband transformation and carries an explicit systematic, so
    the channel must never claim greyness tighter than it; what it must NOT do is
    fail the "is the star detected" test, which applies to a measured baseline
    and has no meaning for a catalogue-derived one.
    """
    _c, rows = A.parse_atlas_text(_atlas_text(n=200, event_idx=50, event_ujy=800.0))
    lc = A.atlas_rows_to_lightcurve(rows, "777", 150.0, -30.0)
    f = float(ab_to_njy(15.0))
    q = {"o": A.Quiescent(f, f * A.PASSBAND_INTERP_REL_ERR, "gspc_interpolated_native")}
    red = A.reduce_lightcurve(lc, A.ATLAS, quiescent=q)
    assert red.bands["o"].usable
    assert red.alerts and red.alerts[0].raw["baseline_source"] == "gspc_interpolated_native"
    # The systematic is carried into the amplitude error, not discarded.
    assert red.alerts[0].template_flux_err_njy == pytest.approx(
        f * A.PASSBAND_INTERP_REL_ERR)


def test_a_noise_dominated_band_leaves_the_denominator_as_well_as_the_numerator():
    """The consistency rule this module applies everywhere.

    The event threshold is 6x the star's OWN scatter, so a band scattering at
    more than ~50 % of the star's flux cannot register anything below a 300 %
    excursion.  Counting its epochs as trials would add trials with essentially
    no chance of an event, deflating the ensemble rate and making every OTHER
    target's binomial p-value too small.  Numerator and denominator leave
    together, or the ratio is not a rate.
    """
    rows = _asassn_rows(n=200, mag=14.0, noise_frac=3.0)     # scatter swamps the star
    red = A.reduce_lightcurve(_lc_from_asassn(rows), A.ASASSN)
    assert not red.bands["g"].usable
    assert "fractional_scatter" in red.bands["g"].reason
    assert red.visit_mjds == [] and red.alerts == []


def test_an_undetected_star_has_no_quiescent_flux_and_is_refused():
    rows = _asassn_rows(n=200, mag=14.0, noise_frac=0.0)
    f0 = rows[0]["flux"]
    for k, r in enumerate(rows):
        r["flux"] = 0.02 * f0 * ((-1) ** k)     # scattering about zero
        r["flux_err"] = 0.02 * f0
        r["mag"] = 99.0
    red = A.reduce_lightcurve(A.asassn_rows_to_lightcurve(rows, "777", 150.0, -30.0),
                              A.ASASSN)
    assert not red.bands["g"].usable
    assert red.bands["g"].reason == "no_quiescent_flux"


# --- the ASAS-SN endpoint the probe corrected -------------------------------
#
# The first probe run pointed at `asas-sn.ifa.hawaii.edu:80/skypatrol/` -- the
# human-facing web host, taken from documentation -- and got a connect timeout
# from the runner. That failure is indistinguishable from "the service is down",
# which is how a documentation-derived endpoint quietly becomes a null result.
# The vendor client's own source names the real API: a Flask service on
# asassn-lb01.ifa.hawaii.edu PORT 9006.

def test_the_asassn_endpoint_is_the_api_host_not_the_web_host():
    from seti.tocsin.altfeeds import ASASSN

    assert "asassn-lb01.ifa.hawaii.edu:9006" in ASASSN.endpoint
    assert "skypatrol" not in ASASSN.endpoint


def test_the_vendor_client_failure_names_the_pyarrow_pin():
    # pyasassn 0.6.4 decodes with pyarrow.deserialize, removed after pyarrow 4.x,
    # and this repository requires pyarrow>=12. Anyone who hits this should be
    # told why rather than concluding the package is merely missing.
    import pytest as _pytest

    from seti.tocsin.altfeeds import AltFeedError, AsasSnSkyPatrol

    c = AsasSnSkyPatrol()
    try:
        c._pyasassn()
    except AltFeedError as exc:
        assert "pyarrow" in str(exc)
    except Exception as exc:  # pragma: no cover - only if the client imports
        _pytest.skip(f"pyasassn importable in this environment: {exc}")


# ---------------------------------------------------------------------------
# ZTF through IRSA: the third feed, opened 2026-08-26 when ASAS-SN went down
# ---------------------------------------------------------------------------
ZTF_CSV = (
    "oid,expid,hjd,mjd,mag,magerr,catflags,filtercode,ra,dec,limitmag\n"
    "1,101,2458000.5,58000.0,15.00,0.010,0,zg,187.2779,2.0524,20.7\n"
    "1,102,2458002.5,58002.0,15.02,0.011,0,zg,187.2779,2.0524,20.6\n"
    "1,103,2458003.5,58003.0,15.90,0.030,32768,zg,187.2779,2.0524,20.1\n"
    "2,104,2458004.5,58004.0,14.60,0.009,0,zr,187.2779,2.0524,20.5\n"
)


def test_a_ztf_csv_is_parsed_by_its_own_header_names():
    header, rows = A.parse_csv_text(ZTF_CSV)
    assert header[:4] == ["oid", "expid", "hjd", "mjd"]
    assert len(rows) == 4
    assert rows[0]["filtercode"] == "zg"


def test_a_reordered_ztf_csv_reads_the_same_values():
    """The whole reason the parse is name-keyed.

    A positional parse of a re-ordered file reads the wrong number into every
    field after the change -- and a wrong magnitude column produces a light
    curve, not an error.
    """
    reordered = ("mjd,magerr,mag,filtercode,catflags,limitmag\n"
                 "58000.0,0.010,15.00,zg,0,20.7\n")
    _h, rows = A.parse_csv_text(reordered)
    lc = A.ztf_rows_to_lightcurve(rows, "t", 187.2779, 2.0524)
    straight = A.ztf_rows_to_lightcurve(A.parse_csv_text(ZTF_CSV)[1][:1], "t",
                                        187.2779, 2.0524)
    assert lc.flux_njy[0] == pytest.approx(straight.flux_njy[0])
    assert lc.mjd[0] == straight.mjd[0]


def test_ztf_magnitudes_become_nanojansky():
    """This is the one feed of the three that serves no flux column at all."""
    _h, rows = A.parse_csv_text(ZTF_CSV)
    lc = A.ztf_rows_to_lightcurve(rows, "t", 187.2779, 2.0524)
    # AB: m = 8.90 - 2.5 log10(F / 1 Jy); 15.00 mag is 3.63e6 nJy.
    assert lc.flux_njy[0] == pytest.approx(10 ** ((8.90 - 15.00) / 2.5) * 1e9, rel=1e-9)
    # A 0.010 mag error is 0.92 % in flux.
    assert lc.flux_err_njy[0] / lc.flux_njy[0] == pytest.approx(0.0092, abs=2e-4)
    assert lc.mjd[0] == 58000.0
    assert lc.survey == "ztf"


def test_a_flagged_ztf_epoch_is_kept_but_not_good():
    """A cut made server-side is a cut nobody can count."""
    _h, rows = A.parse_csv_text(ZTF_CSV)
    lc = A.ztf_rows_to_lightcurve(rows, "t", 187.2779, 2.0524)
    assert len(lc) == 4                              # nothing silently dropped
    assert list(lc.good) == [True, True, False, True]
    assert any("catflags" in n for n in lc.notes)


def test_ztf_bands_survive_both_spellings():
    _h, rows = A.parse_csv_text(
        "mjd,mag,magerr,fid\n58000.0,15.0,0.01,1\n58001.0,14.9,0.01,2\n")
    lc = A.ztf_rows_to_lightcurve(rows, "t", 0.0, 0.0)
    assert list(lc.band) == ["zg", "zr"]


def test_the_ztf_record_says_it_has_no_non_detections():
    """The honest cost of a matchfile feed, carried on the light curve itself."""
    _h, rows = A.parse_csv_text(ZTF_CSV)
    lc = A.ztf_rows_to_lightcurve(rows, "t", 187.2779, 2.0524)
    assert any("DETECTIONS ONLY" in n for n in lc.notes)


def test_an_empty_ztf_response_is_an_empty_curve_not_a_crash():
    lc = A.ztf_rows_to_lightcurve([], "t", 1.0, 2.0)
    assert len(lc) == 0 and "no rows served" in lc.notes[0]


def test_the_ztf_cone_is_small_enough_to_refuse_a_neighbour():
    """A generous radius returns a neighbour's light curve and says nothing."""
    c = A.ZtfIrsa()
    assert c.radius_arcsec <= 3.0
    params = c._params(10.0, 20.0)
    assert params["POS"].startswith("CIRCLE 10.000000 20.000000")
    # Checked in arcseconds, which is the unit the systematic lives in: the cone
    # that is sent must not be measurably wider than the cone that was asked for.
    sent_arcsec = float(params["POS"].split()[-1]) * 3600.0
    assert abs(sent_arcsec - 1.5) < 1e-3


def test_a_ztf_time_window_is_passed_as_mjd():
    p = A.ZtfIrsa()._params(1.0, 2.0, mjd_lo=58000.0, mjd_hi=59000.0)
    assert p["TIME"] == "58000.00000 59000.00000"


def test_ztf_needs_no_token_and_no_queue():
    assert A.ZTF.auth_env is None
    assert A.ZtfIrsa().available is True


def test_ztf_is_deeper_than_atlas_where_it_matters():
    """The reason for the third feed, as an assertion rather than a claim.

    Rubin saturates near 16; ATLAS runs out around 19.5 in the blue; ZTF reaches
    20.8.  The window 16 < m < 20.8 is what ZTF adds to this channel.
    """
    assert A.ZTF.depth_5sigma["zg"] > A.ATLAS.depth_5sigma["c"] + 1.0
    assert A.ZTF.saturation_mag["zg"] >= A.RUBIN_SATURATION_MAG - 4.0
    assert A.ZTF.exposure_s == 30.0                  # a Rubin visit, as ATLAS is


def test_the_band_bracket_reproduces_the_pairs_atlas_had_hard_coded():
    """The generalisation must not move ATLAS's numbers.

    `c` sits between SDSS g and r, `o` between r and i -- which is exactly what
    was written down by hand before the bracket was derived from wavelength.
    """
    import pandas as pd

    t = pd.DataFrame({"g_sdss_mag": [15.0], "r_sdss_mag": [14.5],
                      "i_sdss_mag": [14.3], "z_sdss_mag": [14.2],
                      "u_sdss_mag": [16.0]})
    c = A.synthetic_native_mag(t, A.ATLAS, "c")[0]
    o = A.synthetic_native_mag(t, A.ATLAS, "o")[0]
    assert 14.5 < c < 15.0          # between g and r, as its wavelength is
    assert 14.3 < o < 14.5          # between r and i


def test_every_ztf_band_is_predictable_from_the_target_list():
    """The failure this guards: a new feed whose census silently reaches zero.

    `synthetic_native_mag` used to return NaN for any survey but two, so adding
    ZTF produced a census in which all three of its bands reached no targets at
    all, while every other number in the record looked healthy.
    """
    import pandas as pd

    t = pd.DataFrame({"g_sdss_mag": [15.0], "r_sdss_mag": [14.5],
                      "i_sdss_mag": [14.3], "z_sdss_mag": [14.2],
                      "u_sdss_mag": [16.0]})
    for band in A.ZTF.native_bands:
        m = A.synthetic_native_mag(t, A.ZTF, band)[0]
        assert np.isfinite(m), f"{band} unpredictable"
    # zg is 2 nm from SDSS g, so it must land essentially on g.
    assert A.synthetic_native_mag(t, A.ZTF, "zg")[0] == pytest.approx(15.0, abs=0.02)
    # zr is redder than SDSS r and bluer than i.
    assert 14.3 < A.synthetic_native_mag(t, A.ZTF, "zr")[0] < 14.5


def test_a_band_outside_the_sdss_set_is_unknown_not_extrapolated():
    """A census that counts a band nobody can predict is worse than one that
    says it could not."""
    import dataclasses

    import pandas as pd

    t = pd.DataFrame({"g_sdss_mag": [15.0], "r_sdss_mag": [14.5],
                      "i_sdss_mag": [14.3], "z_sdss_mag": [14.2],
                      "u_sdss_mag": [16.0]})
    far_ir = dataclasses.replace(A.ZTF, band_wl_um={**A.ZTF.band_wl_um, "zk": 2.2},
                                 native_bands=("zk",))
    assert not np.isfinite(A.synthetic_native_mag(t, far_ir, "zk")[0])


def test_the_ztf_walk_stops_at_its_time_budget_and_says_so():
    """A per-request timeout does not bound a walk.

    200 targets at 60 s each is over three hours -- past the job's own timeout,
    and a job the runner kills commits nothing at all.  A short slice honestly
    reported beats a long one that never lands.
    """
    class SlowClock:
        def __init__(self):
            self.t = 0.0

        def monotonic(self):
            self.t += 100.0                  # every call burns 100 s
            return self.t

    c = A.ZtfIrsa()
    reqs = [{"target_id": str(i), "ra": 1.0 * i, "dec": 2.0} for i in range(50)]
    # The clock is INJECTED rather than the stdlib `time` module being swapped
    # out globally, which is what this test used to do.  That worked only while
    # `requests` happened to be imported already: the first import inside the
    # fake window asked for `time.time`, which the fake did not have, and the
    # test died in `requests.sessions` with an AttributeError having nothing to
    # do with budgets.
    out = c.lightcurves(reqs, max_seconds=250.0, clock=SlowClock().monotonic)
    assert out == {}
    assert any("time budget" in n for n in c.notes)


def test_the_ztf_probe_asks_a_bounded_question():
    """A diagnostic that asks for everything is a diagnostic that never answers.

    Probe 33022081059 sat in `describe()` for 25 minutes with a 120 s timeout
    configured: `requests` times out on socket reads, not on the request, so a
    slow trickle is unbounded. The probe now carries a wall-clock budget, a
    small cone and a one-season window.
    """
    c = A.ZtfIrsa()
    assert c.probe_timeout <= 30.0
    assert c.timeout <= 60.0


# The probe of 2026-08-26 recorded IRSA's real columns and one real hazard in
# them: some integer columns arrive in hex (`ccdid: "0x1"`).
ZTF_CSV_REAL = (
    "oid,expid,hjd,mjd,mag,magerr,catflags,filtercode,ra,dec,chi,sharp,"
    "filefracday,field,ccdid,qid,limitmag,magzp,magzprms,clrcoeff,clrcounc,"
    "exptime,airmass,programid\n"
    "1,45030090,2458204.8068,58204.3,13.121748,0.0144,0,zg,187.2779,2.0524,"
    "0.417,0.01,20180327300891,473,0x1,1,20.25,26.249,0.02,-0.057,3.8e-05,"
    "30,1.156,1\n"
    "1,45030091,2458206.8068,58206.3,13.130000,0.0150,0x8000,zg,187.2779,2.0524,"
    "0.430,0.01,20180329300891,473,0x1,1,20.10,26.240,0.02,-0.057,3.8e-05,"
    "30,1.200,1\n"
    "1,45030092,2458208.8068,58208.3,13.140000,0.0160,not-a-number,zg,187.2779,"
    "2.0524,0.440,0.01,20180331300891,473,0x1,1,20.00,26.230,0.02,-0.057,"
    "3.8e-05,30,1.210,1\n"
)


def test_a_hex_catflags_is_read_as_a_flag_not_dropped_to_clean():
    """The direction an error must never fall.

    IRSA serves some integer columns in hex, and `float("0x8000")` raises. Under
    a float parse that exception becomes NaN, NaN fails the finite test, and the
    epoch is treated as UNFLAGGED -- a bad epoch silently promoted to good.
    """
    _h, rows = A.parse_csv_text(ZTF_CSV_REAL)
    lc = A.ztf_rows_to_lightcurve(rows, "t", 187.2779, 2.0524)
    assert lc.good[0] is np.True_ or lc.good[0]        # clean epoch
    assert not lc.good[1], "0x8000 must be read as the bad-quality bit"


def test_an_unreadable_quality_word_is_not_a_clean_one():
    _h, rows = A.parse_csv_text(ZTF_CSV_REAL)
    lc = A.ztf_rows_to_lightcurve(rows, "t", 187.2779, 2.0524)
    assert not lc.good[2]
    assert any("could not read" in n for n in lc.notes)


def test_ztf_chi_reaches_the_artefact_gate():
    """`chi` transfers to ATLAS's `chi_n` slot without re-calibration.

    The gate is RELATIVE -- an epoch is rejected for being a wild outlier
    against this star's own median chi -- so it does not matter that ZTF's chi
    and ATLAS's reduced chi^2 are different statistics.
    """
    _h, rows = A.parse_csv_text(ZTF_CSV_REAL)
    lc = A.ztf_rows_to_lightcurve(rows, "t", 187.2779, 2.0524)
    assert lc.chi_n is not None
    assert lc.chi_n[0] == pytest.approx(0.417)


def test_the_real_irsa_columns_all_resolve():
    """Read against the column list the live service actually returned."""
    _h, rows = A.parse_csv_text(ZTF_CSV_REAL)
    lc = A.ztf_rows_to_lightcurve(rows, "t", 187.2779, 2.0524)
    assert lc.mjd[0] == 58204.3
    assert lc.flux_njy[0] == pytest.approx(10 ** ((8.90 - 13.121748) / 2.5) * 1e9)
    assert lc.limit_njy is not None                    # per-epoch limitmag
    assert list(lc.band) == ["zg", "zg", "zg"]
    assert any("sharp" in n for n in lc.notes)


def test_the_atlas_walk_is_bounded_by_wall_clock_too(tmp_path, monkeypatch):
    """The failure of 2026-08-26, in the feed it actually happened to.

    ATLAS is a queue: `collect` waits up to `max_wait_s` PER TASK and a target
    can need several. Nothing bounded the sum, so a 200-target slice could not
    finish inside the job's 240-minute timeout -- and a job the runner kills
    runs no commit step, so hours of throttled quota land nothing at all.
    """
    import pandas as pd

    targets = pd.DataFrame({
        "source_id": [f"{i}" for i in range(10)],
        "ra": [10.0 + i for i in range(10)], "dec": [2.0] * 10,
        "pmra": [0.0] * 10, "pmdec": [0.0] * 10,
        "g_sdss_mag": [14.0 + 0.1 * i for i in range(10)],
        "r_sdss_mag": [13.8 + 0.1 * i for i in range(10)],
        "i_sdss_mag": [13.6 + 0.1 * i for i in range(10)],
        "z_sdss_mag": [13.5 + 0.1 * i for i in range(10)],
        "u_sdss_mag": [15.0 + 0.1 * i for i in range(10)],
    })

    calls = {"n": 0}

    class SlowAtlas:
        available = True

        def __init__(self, *a, **k):
            pass

        def lightcurve(self, tid, ra, dec, *a, **k):
            calls["n"] += 1
            raise A.AltFeedError("would have taken 30 minutes")

    monkeypatch.setattr(A, "AtlasForcedPhotometry", SlowAtlas)
    monkeypatch.setenv("ALTFEEDS_FETCH_BUDGET_S", "0")     # budget already spent
    lcs, _q, notes = A._fetch(A.ATLAS, targets, A.LightCurveThresholds(),
                              max_targets=10, mjd_lo=58000.0, mjd_hi=59000.0)
    assert lcs == {}
    assert calls["n"] == 0, "the walk must stop before spending more quota"
    assert any("budget" in n for n in notes)


def test_a_normal_atlas_walk_reports_how_far_it_got(tmp_path, monkeypatch):
    import pandas as pd

    targets = pd.DataFrame({
        "source_id": ["a", "b"], "ra": [10.0, 11.0], "dec": [2.0, 2.0],
        "pmra": [0.0, 0.0], "pmdec": [0.0, 0.0],
        "g_sdss_mag": [14.0, 14.1], "r_sdss_mag": [13.8, 13.9],
        "i_sdss_mag": [13.6, 13.7], "z_sdss_mag": [13.5, 13.6],
        "u_sdss_mag": [15.0, 15.1],
    })

    class OkAtlas:
        available = True

        def __init__(self, *a, **k):
            pass

        def lightcurve(self, tid, ra, dec, *a, **k):
            return A.atlas_rows_to_lightcurve([], tid, ra, dec), {}

    monkeypatch.setattr(A, "AtlasForcedPhotometry", OkAtlas)
    _lcs, _q, notes = A._fetch(A.ATLAS, targets, A.LightCurveThresholds(),
                               max_targets=2, mjd_lo=58000.0, mjd_hi=59000.0)
    assert any("walked 2 of 2" in n for n in notes)


# ---------------------------------------------------------------------------
# What the first real ZTF slice taught (2026-08-26, NO_DATA_REACHED)
# ---------------------------------------------------------------------------
def _nearby_targets(n=6, bright=True):
    import pandas as pd

    # Bright end: 10-11 mag, far above ZTF's 12.5 saturation.  Faint end: 15-16,
    # squarely inside its usable range.
    base = 10.0 if bright else 15.0
    return pd.DataFrame({
        "source_id": [f"s{i}" for i in range(n)],
        "ra": [10.0 + i for i in range(n)], "dec": [20.0] * n,
        "pmra": [500.0] * n, "pmdec": [-300.0] * n,          # a real nearby star
        "g_sdss_mag": [base + 0.1 * i for i in range(n)],
        "r_sdss_mag": [base - 0.2 + 0.1 * i for i in range(n)],
        "i_sdss_mag": [base - 0.4 + 0.1 * i for i in range(n)],
        "z_sdss_mag": [base - 0.5 + 0.1 * i for i in range(n)],
        "u_sdss_mag": [base + 1.0 + 0.1 * i for i in range(n)],
    })


def test_a_saturated_target_is_never_requested(monkeypatch):
    """The bug behind the first real ZTF run's NO_DATA_REACHED.

    Brightest-first is right for a shallow feed and exactly wrong at the bright
    end: ZTF saturates near 12.5, the census counts 26,172 targets above it, and
    the slice spent every request on stars that cannot appear in a matchfile.
    """
    asked = []

    class Client:
        calls = 0
        notes: list[str] = []

        def __init__(self, *a, **k):
            pass

        def lightcurves(self, reqs, **kw):
            asked.extend(r["target_id"] for r in reqs)
            return {}

    monkeypatch.setattr(A, "ZtfIrsa", Client)
    _lc, _q, notes = A._fetch(A.ZTF, _nearby_targets(bright=True),
                              A.LightCurveThresholds(), max_targets=6,
                              mjd_lo=58194.0, mjd_hi=61300.0)
    assert asked == [], "saturated targets must not be requested at all"
    assert any("usable magnitude range" in n for n in notes)


def test_a_measurable_target_is_requested(monkeypatch):
    asked = []

    class Client:
        calls = 0
        notes: list[str] = []

        def __init__(self, *a, **k):
            pass

        def lightcurves(self, reqs, **kw):
            asked.extend(r["target_id"] for r in reqs)
            return {}

    monkeypatch.setattr(A, "ZtfIrsa", Client)
    A._fetch(A.ZTF, _nearby_targets(bright=False), A.LightCurveThresholds(),
             max_targets=6, mjd_lo=58194.0, mjd_hi=61300.0)
    assert len(asked) == 6


def test_high_proper_motion_is_segmented_not_refused(monkeypatch):
    """The other half of that run: 110 of 120 targets dropped on drift.

    Correct for ASAS-SN, whose aperture is pinned to its own source list. Wrong
    here: the SEARCH position is ours to choose and the service takes a time
    window in the same request, so the star is followed instead of abandoned.
    """
    seen: list[dict] = []

    class Session:
        def get(self, url, params=None, **kw):
            seen.append(params)

            class R:
                status_code = 200
                text = ("mjd,mag,magerr,catflags,filtercode,limitmag\n"
                        "58200.0,15.0,0.01,0,zg,20.5\n")
            return R()

    c = A.ZtfIrsa()
    c._s = Session()
    out = c.lightcurves([{"target_id": "fast", "ra": 10.0, "dec": 20.0,
                          "pmra": 3000.0, "pmdec": 0.0}],
                        mjd_lo=58194.0, mjd_hi=61300.0)
    assert "fast" in out, "a fast star must still produce a light curve"
    assert len(seen) > 1, "the window must be split into segments"
    # Each segment asks at a different position, and every window is inside the
    # requested baseline.
    positions = {p["POS"] for p in seen}
    assert len(positions) == len(seen)
    for p in seen:
        lo, hi = (float(x) for x in p["TIME"].split())
        assert 58194.0 <= lo < hi <= 61300.0
    assert any("pm_segments=" in n for n in out["fast"].notes)


def test_a_slow_star_is_asked_for_in_one_request(monkeypatch):
    seen = []

    class Session:
        def get(self, url, params=None, **kw):
            seen.append(params)

            class R:
                status_code = 200
                text = "mjd,mag,magerr,catflags,filtercode,limitmag\n58200.0,15.0,0.01,0,zg,20.5\n"
            return R()

    c = A.ZtfIrsa()
    c._s = Session()
    c.lightcurves([{"target_id": "slow", "ra": 10.0, "dec": 20.0,
                    "pmra": 1.0, "pmdec": 0.0}], mjd_lo=58194.0, mjd_hi=61300.0)
    assert len(seen) == 1


def test_the_segment_count_is_capped_so_one_star_cannot_eat_the_walk():
    """Barnard's Star would need ~170 requests; the cap says so out loud."""
    seen = []

    class Session:
        def get(self, url, params=None, **kw):
            seen.append(params)

            class R:
                status_code = 200
                text = ("mjd,mag,magerr,catflags,filtercode,limitmag\n"
                        "58200.0,15.0,0.01,0,zg,20.5\n")
            return R()

    c = A.ZtfIrsa()
    c._s = Session()
    out = c.lightcurves([{"target_id": "barnard", "ra": 269.45, "dec": 4.69,
                          "pmra": -800.0, "pmdec": 10300.0}],
                        mjd_lo=58194.0, mjd_hi=61300.0)
    assert len(seen) == A.ZtfIrsa.MAX_PM_SEGMENTS
    assert any("CAPPED" in n for n in out["barnard"].notes)


def test_a_target_half_walked_when_the_budget_ends_is_discarded():
    """A gappy curve assembled from some of a star's segments is not a light
    curve of that star -- the missing years look exactly like a dip."""
    class Session:
        def get(self, url, params=None, **kw):
            class R:
                status_code = 200
                text = "mjd,mag,magerr,catflags,filtercode,limitmag\n58200.0,15.0,0.01,0,zg,20.5\n"
            return R()

    c = A.ZtfIrsa()
    c._s = Session()
    out = c.lightcurves([{"target_id": "fast", "ra": 10.0, "dec": 20.0,
                          "pmra": 3000.0, "pmdec": 0.0}],
                        mjd_lo=58194.0, mjd_hi=61300.0, max_seconds=0.0)
    assert out == {}
