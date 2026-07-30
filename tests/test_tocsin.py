"""Offline tests for the TOCSIN channel (Rubin/LSST nightly alert screen).

Runs with no network, per ``docs/channel-brief.md`` §5.  The suite is organised
around the four requirements that document imposes:

* **recover an injected signal** --- a grey flash and a grey dip on a target star
  are found, with achromaticity confirmed;
* **return a clean null on the dominant confounder** --- an M-dwarf flare (blue),
  a subtraction dipole, a satellite glint trail, a mover and a reddened dip are
  each rejected, by name;
* **degrade honestly** --- an unreachable broker, an absent target list, a
  missing baseline flux and a missing visit history each produce an explicit
  verdict or an ``untestable`` marker, never a silent pass;
* **cover every rejection rule** with a case that trips it.

Two tests are load-bearing regressions rather than unit checks, and should not be
weakened:  :func:`test_match_fails_without_proper_motion_propagation` (the bug
class that has already cost this repository a whole run in another channel) and
:func:`test_timing_null_is_cadence_matched` (without which the survey's own
revisit cadence reads as a periodic beacon).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from seti.tocsin import ledger as L
from seti.tocsin import photometry as P
from seti.tocsin import targets as T
from seti.tocsin.brokers import (
    AlerceTAP,
    BrokerError,
    LasairLSST,
    normalize_alerce_rows,
    normalize_lasair_diasources,
)
from seti.tocsin.schema import NormalizedAlert, night_id, validate
from seti.tocsin.screen import Thresholds, screen_alerts

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
GAIA_EPOCH = T.GAIA_EPOCH
MJD_2026 = 61500.0          # ~mid-2026, inside the live LSST survey


def _target(source_id="4242", ra=150.0, dec=-30.0, pmra=0.0, pmdec=0.0,
            mag_g=18.0, mag_r=17.0, **kw):
    row = {
        "source_id": source_id, "ra": ra, "dec": dec,
        "ra_error": 0.02, "dec_error": 0.02,
        "pmra": pmra, "pmdec": pmdec, "pmra_error": 0.03, "pmdec_error": 0.03,
        "parallax": 20.0, "phot_g_mean_mag": 15.0, "bp_rp": 2.5,
        "u_sdss_mag": mag_g + 1.5, "g_sdss_mag": mag_g, "r_sdss_mag": mag_r,
        "i_sdss_mag": mag_r - 0.4, "z_sdss_mag": mag_r - 0.6,
        "y_ps1_mag": mag_r - 0.7,
    }
    row.update(kw)
    return row


def _targets(*rows):
    return pd.DataFrame(list(rows) or [_target()])


def _alert(band="g", dflux=500.0, dflux_err=20.0, mjd=MJD_2026, ra=150.0,
           dec=-30.0, template_flux=1000.0, **kw):
    kw.setdefault("alert_id", f"a-{band}-{mjd}")
    kw.setdefault("object_id", "obj1")
    return NormalizedAlert(
        mjd=mjd, band=band, ra=ra, dec=dec, dflux_njy=dflux,
        dflux_err_njy=dflux_err, template_flux_njy=template_flux,
        ra_err_arcsec=0.02, dec_err_arcsec=0.02, snr=dflux / dflux_err,
        reliability=0.9, extendedness=0.0, **kw)


# ---------------------------------------------------------------------------
# photometry
# ---------------------------------------------------------------------------
def test_ab_zeropoint_matches_rubin_convention():
    # 3631 Jy is AB zero; Rubin/Lasair use mag = 31.4 - 2.5 log10(F/nJy).
    assert P.njy_to_ab(3.631e12) == pytest.approx(0.0, abs=1e-3)
    assert P.njy_to_ab(3631.0) == pytest.approx(22.5, abs=1e-3)
    assert P.ab_to_njy(P.njy_to_ab(1234.0)) == pytest.approx(1234.0, rel=1e-9)


def test_negative_flux_has_no_magnitude():
    # The dip mode lives at negative flux; it must not silently become bright.
    assert math.isnan(P.njy_to_ab(-500.0))
    assert math.isnan(P.njy_to_ab(0.0))


def test_fractional_amplitude_and_error_propagation():
    amp = P.fractional_amplitude(100.0, 10.0, 1000.0, 30.0)
    assert amp.testable
    assert amp.a == pytest.approx(0.1)
    # relative error = hypot(10/100, 30/1000) = hypot(0.1, 0.03)
    assert amp.a_err == pytest.approx(0.1 * math.hypot(0.1, 0.03))


def test_missing_baseline_is_untestable_not_a_pass():
    for base in (None, 0.0, -5.0, float("nan")):
        amp = P.fractional_amplitude(100.0, 10.0, base)
        assert not amp.testable
        assert math.isnan(amp.a)
        assert amp.reason


def test_greyness_z_separates_grey_from_blue():
    # Grey: equal fractional amplitude in both bands.
    assert abs(P.greyness_z(0.50, 0.01, 0.50, 0.01)) < 1.0
    # Flare-like: much larger amplitude in the bluer band.
    assert P.greyness_z(0.50, 0.01, 0.05, 0.01) > 10.0


def test_reflection_predicts_unit_amplitude_ratio():
    # A reflector returns the stellar spectrum, so a_blue/a_red is exactly 1
    # whatever the star's temperature.  This identity is the channel's premise.
    for t_star in (3000.0, 5800.0, 9000.0):
        assert P.predicted_amplitude_ratio("g", "r", t_star, t_star) == \
            pytest.approx(1.0, rel=1e-9)


def test_flare_on_m_dwarf_predicts_large_amplitude_ratio():
    # A 9000 K flare continuum on a 3200 K dwarf gives a_g/a_r ~ 3.7 from the
    # blackbody ratio alone (real flares are bluer still, because of line
    # emission this model does not include).  Pinned as an inequality plus a
    # value check so a units or wavelength regression cannot pass quietly.
    ratio = P.predicted_amplitude_ratio("g", "r", 9000.0, 3200.0)
    assert ratio > 3.0
    assert ratio == pytest.approx(3.69, rel=0.02)


def test_colour_temperature_recovers_an_injected_blackbody():
    bands = ["g", "r", "i"]
    t_true = 9000.0
    flux = [1e4 * P.planck_nu(P.LSST_BAND_WL_UM[b], t_true) / 1e-20 for b in bands]
    fit = P.blackbody_colour_temperature(bands, flux, [f * 0.01 for f in flux])
    assert fit.ok
    assert fit.temp_k == pytest.approx(t_true, rel=0.15)
    assert fit.n_bands == 3


def test_colour_temperature_refuses_impossible_inputs():
    assert not P.blackbody_colour_temperature(["g"], [10.0], [1.0]).ok
    # A dip carries no emission temperature.
    bad = P.blackbody_colour_temperature(["g", "r"], [-10.0, -20.0], [1.0, 1.0])
    assert not bad.ok
    assert "negative_flux" in bad.reason


def test_two_band_colour_temperature_declares_zero_degrees_of_freedom():
    fit = P.blackbody_colour_temperature(["g", "r"], [100.0, 200.0], [1.0, 2.0])
    assert fit.ok and fit.reason == "two_band_zero_dof"


# ---------------------------------------------------------------------------
# targets and matching
# ---------------------------------------------------------------------------
def test_proper_motion_propagation_moves_a_high_pm_star():
    # 1000 mas/yr for 10 yr = 10 arcsec.
    ra1, dec1 = T.propagate_pm(150.0, 0.0, 1000.0, 0.0, 2016.0, 2026.0)
    assert (ra1 - 150.0) * 3600.0 == pytest.approx(10.0, rel=1e-6)
    _, dec2 = T.propagate_pm(150.0, 0.0, 0.0, 1000.0, 2016.0, 2026.0)
    assert (dec2 - 0.0) * 3600.0 == pytest.approx(10.0, rel=1e-6)


def test_proper_motion_includes_cos_dec_convention():
    # pmra is mu_alpha* (already includes cos dec), so at dec=60 the RA shift
    # is doubled relative to the equator.
    ra_eq, _ = T.propagate_pm(10.0, 0.0, 3600.0, 0.0, 2016.0, 2017.0)
    ra_hi, _ = T.propagate_pm(10.0, 60.0, 3600.0, 0.0, 2016.0, 2017.0)
    assert (ra_hi - 10.0) == pytest.approx(2.0 * (ra_eq - 10.0), rel=1e-6)


def test_match_fails_without_proper_motion_propagation():
    """Load-bearing regression: an un-propagated match returns a clean NULL.

    A nearby star with 1 arcsec/yr has moved 10 arcsec by 2026.  Matching the
    2026 alert against the 2016 catalogue position finds nothing --- and a null
    is far more dangerous than an exception, because it looks like a result.
    """
    ra0, dec0, pm = 150.0, -30.0, 1000.0
    ra_now, dec_now = T.propagate_pm(ra0, dec0, pm, 0.0, 2016.0, 2026.0)
    un = T.match_alerts_to_targets([ra_now], [dec_now], [ra0], [dec0],
                                   radius_arcsec=1.5)
    assert un.alert_index.size == 0          # the bug this test exists to catch
    good = T.match_alerts_to_targets([ra_now], [dec_now], [ra_now], [dec_now],
                                     radius_arcsec=1.5)
    assert good.alert_index.size == 1


def test_match_handles_ra_wraparound():
    m = T.match_alerts_to_targets([0.0001], [10.0], [359.9999], [10.0],
                                  radius_arcsec=5.0)
    assert m.alert_index.size == 1
    assert m.sep_arcsec[0] < 5.0


def test_match_keeps_only_the_nearest_target_per_alert():
    # Two targets inside the radius; an alert may consume only one trial.
    m = T.match_alerts_to_targets([150.0], [-30.0],
                                  [150.0, 150.00005], [-30.0, -30.0],
                                  radius_arcsec=5.0)
    assert m.alert_index.size == 1
    assert m.target_index[0] == 0


def test_missing_proper_motion_widens_the_error_not_the_sample():
    sig = T.position_uncertainty_arcsec([np.nan], [np.nan], dt_yr=10.0,
                                        pm_missing=np.array([True]),
                                        missing_pm_penalty_arcsec=2.0)
    assert sig[0] == pytest.approx(2.0)


def test_parallax_shells_cover_the_distance_limit():
    shells = T.parallax_shells(100.0, 6)
    assert len(shells) == 6
    assert min(lo for lo, _ in shells) == pytest.approx(10.0)
    for lo, hi in shells:
        assert hi > lo


def test_gspc_synthetic_magnitude_columns_use_the_real_gaia_names():
    """Regression: Gaia DR3 names these `<band>_<system>_mag`, not `mag_<band>_<system>`.

    Getting this backwards is silent, not loud: the JOIN errors, acquisition
    falls back to no synthetic photometry, and every fractional amplitude
    downstream becomes untestable — which disables the greyness test, i.e. the
    channel's core discriminator.  The first live run produced 62 events with
    exactly zero colour tests because of this.
    """
    assert T.GSPC_MAG_COLUMN["g"] == "g_sdss_mag"
    assert T.GSPC_MAG_COLUMN["r"] == "r_sdss_mag"
    assert T.GSPC_MAG_COLUMN["y"] == "y_ps1_mag"
    q = T.build_target_adql(10.0, 20.0)
    assert "s.g_sdss_mag AS g_sdss_mag" in q
    assert "mag_g_sdss" not in q
    # The screen must read the same names it asked Gaia for.
    from seti.tocsin.screen import BASELINE_COLUMN
    assert BASELINE_COLUMN == T.GSPC_MAG_COLUMN


def test_target_adql_has_the_quality_and_footprint_cuts():
    q = T.build_target_adql(10.0, 20.0, dec_max=15.0)
    assert "gaiadr3.gaia_source" in q
    assert "synthetic_photometry_gspc" in q
    assert "parallax_over_error" in q
    assert "BETWEEN -90.0 AND 15.0" in q


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------
def test_a_chilean_night_carries_one_label_across_utc_midnight():
    # 23:00 UTC and 02:00 UTC the next calendar day are the same observing night
    # at Cerro Pachon; a UTC-date split would cut the in-night filter pair apart.
    evening = 61500.0 + 23.0 / 24.0
    after_midnight = 61501.0 + 2.0 / 24.0
    assert night_id(evening) == night_id(after_midnight)
    # ... and the following night is a different label.
    assert night_id(evening) != night_id(evening + 1.0)


def test_validate_names_every_structural_problem():
    bad = NormalizedAlert(alert_id="x", object_id="o", mjd=float("nan"),
                          band="q", ra=1.0, dec=2.0, dflux_njy=0.0,
                          dflux_err_njy=-1.0)
    problems = validate(bad)
    assert "missing_mjd" in problems
    assert "unknown_band_q" in problems
    assert "nonpositive_flux_error" in problems
    assert "zero_difference_flux" in problems
    assert validate(_alert()) == []


def test_polarity_follows_the_sign_of_the_difference_flux():
    assert _alert(dflux=+500.0).polarity == "flash"
    assert _alert(dflux=-500.0).polarity == "dip"


def test_missing_astrometric_errors_fall_back_to_a_floor_not_to_zero():
    a = _alert()
    a.ra_err_arcsec = None
    a.dec_err_arcsec = None
    assert a.pos_err_arcsec == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# screen --- injected signal
# ---------------------------------------------------------------------------
def test_injected_grey_flash_is_recovered_and_confirmed_achromatic():
    tg = _targets(_target())
    # Equal fractional amplitude in g and r == the reflection hypothesis.
    alerts = [_alert(band="g", dflux=100.0, dflux_err=3.0, template_flux=1000.0),
              _alert(band="r", dflux=500.0, dflux_err=15.0, template_flux=5000.0,
                     mjd=MJD_2026 + 33.0 / 1440.0)]
    v = screen_alerts(alerts, tg, Thresholds())
    assert len(v.events) == 1                 # one star-NIGHT, not one per band
    ev = v.events[0]
    assert ev.polarity == "flash"
    assert ev.grey_tested and abs(ev.grey_z) < 3.0
    assert set(ev.bands) == {"g", "r"}
    assert ev.per_band["g"]["baseline_source"] == "rubin_template"


def test_injected_grey_dip_is_recovered():
    tg = _targets(_target())
    alerts = [_alert(band="g", dflux=-100.0, dflux_err=3.0, template_flux=1000.0),
              _alert(band="r", dflux=-500.0, dflux_err=15.0, template_flux=5000.0,
                     mjd=MJD_2026 + 33.0 / 1440.0)]
    v = screen_alerts(alerts, tg, Thresholds())
    assert len(v.events) == 1
    assert v.events[0].polarity == "dip"
    assert v.events[0].grey_tested and abs(v.events[0].grey_z) < 3.0


def test_two_bands_in_one_night_are_one_event_not_two():
    # The multiplicity that feeds the binomial p-value must count star-nights.
    tg = _targets(_target())
    alerts = [_alert(band="g", dflux=100.0, dflux_err=3.0, template_flux=1000.0),
              _alert(band="r", dflux=500.0, dflux_err=15.0, template_flux=5000.0)]
    assert len(screen_alerts(alerts, tg, Thresholds()).events) == 1


# ---------------------------------------------------------------------------
# screen --- confounders, each rejected by name
# ---------------------------------------------------------------------------
def test_m_dwarf_flare_is_rejected_as_chromatic():
    """The dominant astrophysical confounder: flares are blue, reflections are not."""
    tg = _targets(_target())
    # Red star (F*_g=1000, F*_r=10000); flare adds equal FLUX in both bands,
    # so the fractional amplitude is ~10x larger in g.
    alerts = [_alert(band="g", dflux=500.0, dflux_err=5.0, template_flux=1000.0),
              _alert(band="r", dflux=500.0, dflux_err=5.0, template_flux=10000.0)]
    v = screen_alerts(alerts, tg, Thresholds())
    assert v.events == []
    assert any(r["reason"] == "chromatic" for r in v.rejected)


def test_subtraction_dipole_is_rejected():
    tg = _targets(_target())
    a = _alert()
    a.is_dipole = True
    v = screen_alerts([a], tg, Thresholds())
    assert v.events == []
    assert v.rejected[0]["reason"] == "dipole"


def test_satellite_glint_trail_is_rejected():
    """A satellite glint is a brief achromatic reflection --- an exact mimic."""
    tg = _targets(_target())
    a = _alert()
    a.glint_trail = True
    v = screen_alerts([a], tg, Thresholds())
    assert v.events == []
    assert v.rejected[0]["reason"] == "glint_trail"


def test_moving_and_extended_and_flagged_sources_are_rejected():
    tg = _targets(_target())
    for field, value, reason in (
        ("extendedness", 0.9, "extended"),
        ("trail_length_arcsec", 2.0, "trailed"),
        ("ss_object_id", "asteroid-1", "solar_system"),
        ("pixel_flag_bad", True, "bad_pixels"),
    ):
        a = _alert()
        setattr(a, field, value)
        v = screen_alerts([a], tg, Thresholds())
        assert v.events == [], f"{reason} should have been rejected"
        assert v.rejected[0]["reason"] == reason


def test_low_significance_is_rejected():
    tg = _targets(_target())
    a = _alert(dflux=10.0, dflux_err=10.0)
    a.snr = 1.0
    v = screen_alerts([a], tg, Thresholds())
    assert v.rejected[0]["reason"] == "low_significance"


def test_astrometrically_offset_event_is_rejected():
    tg = _targets(_target(ra=150.0, dec=-30.0))
    # 0.8 arcsec away, with 0.02 arcsec errors: inside the match radius but far
    # outside the astrometric tolerance.
    v = screen_alerts([_alert(ra=150.0 + 0.8 / 3600.0)], tg, Thresholds())
    assert v.events == []
    assert v.rejected[0]["reason"] == "astrometric_offset"


def test_mixed_polarity_in_one_night_is_rejected_as_a_subtraction_artefact():
    tg = _targets(_target())
    alerts = [_alert(band="g", dflux=+300.0), _alert(band="r", dflux=-300.0)]
    v = screen_alerts(alerts, tg, Thresholds())
    assert v.events == []
    assert any(r["reason"] == "mixed_polarity_same_night" for r in v.rejected)


def test_high_proper_motion_star_still_matches_after_propagation():
    ra0, dec0, pm = 150.0, -30.0, 900.0
    tg = _targets(_target(ra=ra0, dec=dec0, pmra=pm))
    epoch = 2026.5
    ra_now, dec_now = T.propagate_pm(ra0, dec0, pm, 0.0, GAIA_EPOCH, epoch)
    v = screen_alerts([_alert(ra=float(ra_now), dec=float(dec_now))], tg,
                      Thresholds(), epoch_jyear=epoch)
    assert len(v.events) == 1


# ---------------------------------------------------------------------------
# screen --- honest degradation
# ---------------------------------------------------------------------------
def test_single_band_event_is_kept_but_marked_untested():
    tg = _targets(_target())
    v = screen_alerts([_alert(band="g")], tg, Thresholds())
    assert len(v.events) == 1
    ev = v.events[0]
    assert not ev.grey_tested
    assert "greyness_untested_single_band" in ev.reasons


def test_absent_baseline_flux_marks_the_amplitude_untestable():
    tg = _targets(_target())
    tg = tg.drop(columns=[c for c in tg.columns if c.endswith("_mag")])
    a = _alert()
    a.template_flux_njy = None
    v = screen_alerts([a], tg, Thresholds())
    assert len(v.events) == 1
    assert any("amplitude_" in r for r in v.events[0].reasons)
    assert math.isnan(v.events[0].a)


def test_gaia_synthetic_photometry_is_the_documented_fallback():
    tg = _targets(_target(mag_g=20.0, mag_r=19.0))
    a = _alert(band="g")
    a.template_flux_njy = None
    v = screen_alerts([a], tg, Thresholds())
    assert v.events[0].per_band["g"]["baseline_source"] == "gaia_gspc_synthetic"


def test_empty_inputs_degrade_with_a_named_note():
    assert "no_alerts_in" in screen_alerts([], _targets(), Thresholds()).notes
    assert "no_targets" in screen_alerts([_alert()], pd.DataFrame(), Thresholds()).notes


def test_malformed_alerts_are_counted_not_crashed_on():
    tg = _targets(_target())
    bad = _alert()
    bad.band = "q"
    v = screen_alerts([bad], tg, Thresholds())
    assert v.counts.get("rejected_malformed") == 1


def test_funnel_counts_are_recorded_at_every_stage():
    tg = _targets(_target())
    v = screen_alerts([_alert()], tg, Thresholds())
    for key in ("alerts_in", "structurally_valid", "quality_passed",
                "positionally_matched", "events_kept"):
        assert key in v.counts


# ---------------------------------------------------------------------------
# ledger --- statistics
# ---------------------------------------------------------------------------
def test_binomial_tail_is_correct():
    assert L.binomial_sf(1, 1, 0.25) == pytest.approx(0.25)
    assert L.binomial_sf(2, 2, 0.5) == pytest.approx(0.25)
    assert L.binomial_sf(1, 10, 0.1) == pytest.approx(1 - 0.9 ** 10, rel=1e-9)


def test_binomial_returns_no_evidence_for_a_degenerate_denominator():
    # A missing visit history must never look like significance.
    assert L.binomial_sf(3, 0, 0.01) == 1.0
    assert L.binomial_sf(3, 2, 0.01) == 1.0
    assert L.binomial_sf(1, 10, 0.0) == 1.0


def test_benjamini_hochberg_controls_the_false_discovery_rate():
    reject, thresh = L.benjamini_hochberg([1e-8, 0.02, 0.4, 0.9], alpha=0.05)
    assert reject[0] and not reject[2] and not reject[3]
    assert thresh <= 0.05


def test_timing_needs_three_events():
    p = L.timing_structure([1.0, 2.0], list(np.arange(0.0, 30.0)))[2]
    assert p == 1.0          # two points are always "periodic"


def test_timing_structure_finds_an_injected_period():
    visits = list(np.arange(0.0, 200.0, 1.0))
    events = [10.0, 30.0, 50.0, 70.0, 90.0]        # exactly 20 d apart
    period, r, p = L.timing_structure(events, visits, n_null=400)
    assert r > 0.99
    assert p < 0.05


def test_timing_null_is_cadence_matched():
    """Load-bearing: the survey's own cadence must not read as a beacon.

    Visits happen only every 4th day, so *every* possible event spacing is a
    multiple of 4 days and every event set is exactly commensurate with a 4-day
    period.  Against a uniform-time null that structure is wildly significant.
    Drawing the null from the star's own visit epochs cancels it exactly, so
    irregularly spaced events score as the noise they are.
    """
    visits = list(np.arange(0.0, 400.0, 4.0))
    events = [8.0, 36.0, 84.0, 180.0, 292.0]       # irregular, all on the cadence
    p = L.timing_structure(events, visits, n_null=600)[2]
    assert p > 0.05, "cadence alias leaked through as a timing detection"


def test_evenly_spaced_events_beat_the_cadence_matched_null():
    """The other side of the same test: real regularity must still be findable.

    Same coarse 4-day cadence, but the events are exactly evenly spaced.  The
    null (random draws from the same visits) rarely achieves that, so this is a
    genuine detection rather than a sampling artefact.
    """
    visits = list(np.arange(0.0, 400.0, 4.0))
    events = [8.0, 40.0, 72.0, 104.0, 136.0]       # every 32 d exactly
    period, r, p = L.timing_structure(events, visits, n_null=600)
    assert r > 0.99
    assert p < 0.05


def test_timing_p_value_respects_its_own_resolution():
    visits = list(np.arange(0.0, 200.0, 1.0))
    p = L.timing_structure([10.0, 30.0, 50.0, 70.0], visits, n_null=99)[2]
    assert p >= 1.0 / 100.0     # add-one estimator, never a bare zero


# ---------------------------------------------------------------------------
# ledger --- accumulation and promotion
# ---------------------------------------------------------------------------
def _event(tid="4242", night="n61500", mjd=61500.2, grey_z=0.4, grey=True,
           polarity="flash"):
    return L.Event(target_id=tid, night=night, mjd=mjd, polarity=polarity,
                   bands=["g", "r"], dflux_njy=500.0, dflux_err_njy=10.0,
                   strongest_band="r", a=0.1, a_err=0.005, sep_arcsec=0.05,
                   sep_sigma=1.0, grey_z=grey_z, grey_tested=grey,
                   colour_temp_k=3300.0, alert_ids=["a1", "a2"])


def test_ledger_accumulates_trials_across_nights():
    led = L.Ledger()
    led.add_night("n1", [_event(night="n1", mjd=61500.2)], target_visits=1000,
                  targets_in_footprint=900, alerts_seen=5000)
    led.add_night("n2", [_event(night="n2", mjd=61505.2)], target_visits=1200,
                  targets_in_footprint=900, alerts_seen=5200)
    assert led.n_target_visits == 2200
    assert led.n_events_kept == 2
    # Re-folding a night already recorded must not double-count its trials.
    led.add_night("n2", [], target_visits=1200, targets_in_footprint=900,
                  alerts_seen=5200)
    assert led.n_target_visits == 2200


def test_overlapping_run_windows_do_not_double_count_trials():
    """Anti-conservative bug: a re-seen night must not add its trials again.

    The screen pulls a 2-night window and runs daily, so consecutive runs
    overlap by a night.  Counting that night's trials twice inflates the
    denominator, deflates the ensemble rate, and makes every per-target
    binomial p-value too SMALL --- i.e. it manufactures significance.
    """
    led = L.Ledger()
    led.add_night("n61500", [], target_visits=1000, targets_in_footprint=900,
                  alerts_seen=5000)
    led.add_night("n61501", [], target_visits=1100, targets_in_footprint=900,
                  alerts_seen=5100)
    # Tomorrow's run re-covers n61501 and adds n61502.
    led.add_night("n61501", [], target_visits=1100, targets_in_footprint=900,
                  alerts_seen=5100)
    led.add_night("n61502", [], target_visits=1200, targets_in_footprint=900,
                  alerts_seen=5200)
    assert led.n_target_visits == 1000 + 1100 + 1200
    assert led.nights == ["n61500", "n61501", "n61502"]


def test_an_impossible_rate_is_refused_rather_than_published():
    """Regression from the first live run: 62 events against 8 star-nights.

    A per-star-night event *probability* cannot exceed 1.  When it does, the
    numerator and denominator were measured over different populations and the
    quotient is not a rate.  Publishing 7.75 into a committed artefact invites a
    reader to treat it as one.
    """
    led = L.Ledger()
    led.n_target_visits = 8
    led.n_events_kept = 62
    stats = led.assess()
    assert stats["ensemble_rate_per_target_visit"] is None or \
        math.isnan(stats["ensemble_rate_per_target_visit"])
    assert any("INVALID ensemble rate" in n for n in led.notes)


def test_ledger_deduplicates_identical_events():
    led = L.Ledger()
    ev = _event()
    led.add_night("n1", [ev, ev], target_visits=10, targets_in_footprint=5,
                  alerts_seen=10)
    assert led.targets["4242"]["n_events"] == 1


def test_visit_history_is_counted_in_nights_not_epochs():
    """The denominator must count the same unit the numerator does."""
    led = L.Ledger()
    led.add_night("n1", [_event()], target_visits=10, targets_in_footprint=5,
                  alerts_seen=10,
                  # Two epochs 33 min apart == ONE observing night.
                  visit_history={"4242": [61500.1, 61500.1 + 33.0 / 1440.0,
                                          61504.1, 61508.1]})
    assert led.targets["4242"]["n_visits"] == 3
    assert led.targets["4242"]["visits_exact"]


def test_a_first_night_detection_is_not_killed_by_its_own_duty_cycle():
    """Regression: one visit and one event is a duty cycle of 1.0 by arithmetic.

    Applying the subtraction-residual cut there would reject every new detection
    on the night it is found --- the one night that matters most.
    """
    led = L.Ledger()
    led.add_night("n1", [_event()], target_visits=1, targets_in_footprint=1,
                  alerts_seen=1, visit_history={"4242": [61500.2]})
    led.assess(max_duty_cycle=0.2, min_visits_for_rate=5)
    rec = led.targets["4242"]
    assert rec["duty_cycle"] == 1.0
    # `interest`, not `watch`: this event was colour-confirmed grey.  What
    # matters is that it is not `none` --- it survived to be looked at again.
    assert rec["tier"] == "interest"
    assert "duty_cycle_not_yet_testable" in rec["notes"]
    assert "rejected_high_duty_cycle" not in rec["notes"]


def test_approximate_denominator_cannot_reach_candidate_tier():
    led = L.Ledger()
    evs = [_event(night="n1", mjd=61500.2), _event(night="n2", mjd=61510.2)]
    led.add_night("n1", evs, target_visits=100, targets_in_footprint=50,
                  alerts_seen=100)                       # no visit_history
    led.assess()
    rec = led.targets["4242"]
    assert rec["tier"] in ("watch", "interest")
    assert "denominator_approximate" in rec["notes"]


def test_high_duty_cycle_star_is_rejected_as_a_subtraction_residual():
    """A proper-motion dipole alerts at every visit; a beacon does not."""
    led = L.Ledger()
    evs = [_event(night=f"n{i}", mjd=61500.0 + i) for i in range(8)]
    led.add_night("n1", evs, target_visits=10, targets_in_footprint=5,
                  alerts_seen=50,
                  visit_history={"4242": [61500.0 + i for i in range(10)]})
    led.assess(max_duty_cycle=0.2)
    rec = led.targets["4242"]
    assert rec["tier"] == "none"
    assert "rejected_high_duty_cycle" in rec["notes"]


def test_repeated_grey_events_on_a_quiet_star_reach_candidate_tier():
    led = L.Ledger()
    # A rare repeater: 2 events in 60 visited nights, against an ensemble rate
    # set by many quiet targets.
    led.n_target_visits = 200000
    led.n_events_kept = 200                    # ensemble rate = 1e-3 per visit
    evs = [_event(night="n1", mjd=61500.2), _event(night="n2", mjd=61530.2)]
    led.add_night("n1", evs, target_visits=0, targets_in_footprint=0,
                  alerts_seen=0,
                  visit_history={"4242": [61500.0 + 1.0 * i for i in range(60)]})
    led.assess(alpha_fdr=0.05, min_visits_for_rate=5)
    rec = led.targets["4242"]
    assert rec["visits_exact"]
    assert rec["p_binomial"] < 0.01
    assert rec["tier"] in ("candidate", "alarm")


def test_a_single_event_is_only_a_watch():
    led = L.Ledger()
    led.n_target_visits, led.n_events_kept = 100000, 100
    led.add_night("n1", [_event(grey=False)], target_visits=0,
                  targets_in_footprint=0, alerts_seen=0,
                  visit_history={"4242": [61500.0 + i for i in range(40)]})
    led.assess()
    assert led.targets["4242"]["tier"] == "watch"


def test_ledger_round_trips_through_json_without_nan(tmp_path):
    led = L.Ledger()
    led.add_night("n1", [_event(grey_z=float("nan"), grey=False)],
                  target_visits=10, targets_in_footprint=5, alerts_seen=10)
    led.assess()
    p = tmp_path / "ledger.json"
    led.save(p)
    raw = p.read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw)                                   # strict parse must succeed
    back = L.Ledger.load(p)
    assert back.n_events_kept == led.n_events_kept
    assert back.version == L.LEDGER_VERSION


def test_assess_survives_a_json_round_trip(tmp_path):
    """Regression: NaN is written as JSON null, so every re-read value may be None.

    ``float(None)`` raises, so an ``assess`` after a reload used to crash on any
    event whose greyness was untestable --- i.e. on every single-band event.
    """
    led = L.Ledger()
    led.add_night("n1", [_event(grey_z=float("nan"), grey=False)],
                  target_visits=10, targets_in_footprint=5, alerts_seen=10)
    led.assess()
    p = tmp_path / "ledger.json"
    led.save(p)
    reloaded = L.Ledger.load(p)
    assert reloaded.targets["4242"]["events"][0]["grey_z"] is None
    reloaded.assess()                       # must not raise
    assert reloaded.targets["4242"]["tier"] in L.TIERS


def test_a_perfectly_grey_event_is_not_discarded_by_a_truthiness_test(tmp_path):
    """Regression: ``grey_z == 0.0`` is the BEST possible event, not a missing one."""
    led = L.Ledger()
    led.n_target_visits, led.n_events_kept = 200000, 200
    evs = [_event(night="n1", mjd=61500.2, grey_z=0.0),
           _event(night="n2", mjd=61530.2, grey_z=0.0)]
    led.add_night("n1", evs, target_visits=0, targets_in_footprint=0,
                  alerts_seen=0,
                  visit_history={"4242": [61500.0 + i for i in range(60)]})
    led.assess()
    assert led.targets["4242"]["tier"] in ("candidate", "alarm")


def test_loading_an_absent_ledger_starts_a_fresh_one(tmp_path):
    led = L.Ledger.load(tmp_path / "nope.json")
    assert led.n_target_visits == 0 and led.targets == {}


def test_summary_reports_the_cumulative_trial_count():
    led = L.Ledger()
    led.add_night("n1", [_event()], target_visits=1234, targets_in_footprint=99,
                  alerts_seen=10)
    s = led.assess()
    assert s["cumulative_target_visits"] == 1234
    assert "tier_counts" in s and "ensemble_rate_per_target_visit" in s


# ---------------------------------------------------------------------------
# brokers (no network)
# ---------------------------------------------------------------------------
def test_alerce_band_integer_mapping_puts_u_at_six():
    """u = 6, not 0.  Getting this wrong silently relabels every u detection."""
    rows = [{"measurement_id": "1", "oid": "o", "mjd": 61500.0, "band": 6,
             "ra": 10.0, "dec": -20.0, "psfflux": 100.0, "psffluxerr": 5.0}]
    assert normalize_alerce_rows(rows)[0].band == "u"
    rows[0]["band"] = 1
    assert normalize_alerce_rows(rows)[0].band == "g"


def test_alerce_normalisation_tolerates_absent_optional_columns():
    rows = [{"measurement_id": "1", "oid": "o", "mjd": 61500.0, "band": 2,
             "ra": 10.0, "dec": -20.0, "psfflux": -100.0, "psffluxerr": 5.0}]
    a = normalize_alerce_rows(rows)[0]
    assert a.polarity == "dip"
    assert a.reliability is None and a.is_dipole is None
    assert a.template_flux_njy is None
    assert validate(a) == []


def test_alerce_normalisation_flags_bad_pixels():
    rows = [{"measurement_id": "1", "oid": "o", "mjd": 61500.0, "band": 2,
             "ra": 10.0, "dec": -20.0, "psfflux": 100.0, "psffluxerr": 5.0,
             "pixelflags_crcenter": True}]
    assert normalize_alerce_rows(rows)[0].pixel_flag_bad is True


def test_alerce_adql_carries_the_reductions_that_make_a_night_affordable():
    tap = AlerceTAP()
    captured = {}
    tap.query = lambda adql, maxrec=None: captured.setdefault("adql", adql) and []
    tap.night_detections(61500.0, 61501.0, parallax_min_mas=10.0)
    adql = captured["adql"]
    assert "alerce_tap.detection" in adql and "alerce_tap.lsst_detection" in adql
    assert "d.sid = 1" in adql               # excludes known solar-system objects
    assert "g.parallax > 10.0" in adql       # the nearby-star server-side cut
    assert "d.mjd >= 61500.0" in adql and "d.mjd < 61501.0" in adql


def test_gaia_join_uses_oid_catalog_on_both_sides():
    """Regression: ALeRCE's Gaia table has no `source_id` column.

    The first live probe failed with "column 'source_id' could not be located in
    table metadata", which broke both the nearby-star pre-cut and the
    forced-photometry denominator.  The join key is `oid_catalog` on both sides.
    """
    tap = AlerceTAP()
    captured = {}
    tap.query = lambda adql, maxrec=None, retries=4: (
        captured.setdefault("adql", adql) and [])
    tap.night_detections(61500.0, 61501.0, parallax_min_mas=10.0)
    adql = captured["adql"]
    assert "g.oid_catalog = x.oid_catalog" in adql
    # `g.source_id` is the column that does not exist; the output ALIAS
    # `gaia_source_id` is fine and must not trip this check.
    assert "g.source_id" not in adql


def test_sid_filter_is_optional_so_a_wrong_guess_cannot_empty_the_night():
    """`sid`'s meaning is unverified; the join to lsst_detection is what matters.

    If `sid` turns out to be a survey id rather than a source-table id, filtering
    on it would select ZTF and return a clean, plausible, wrong null.  Passing
    None must drop the clause entirely.
    """
    tap = AlerceTAP()
    captured = {}
    tap.query = lambda adql, maxrec=None, retries=4: (
        captured.setdefault("adql", adql) and [])
    tap.night_detections(61500.0, 61501.0, parallax_min_mas=None,
                         sid_diaobject=None)
    # The JOIN legitimately contains `d.sid = ld.sid`; what must be gone is the
    # literal WHERE filter on a specific sid value.
    assert "d.sid = 1" not in captured["adql"]
    # The LSST restriction survives regardless, via the inner join.
    assert "alerce_tap.lsst_detection" in captured["adql"]

    captured.clear()
    tap.forced_photometry_night(61500.0, 61501.0, parallax_min_mas=None,
                                sid_diaobject=None)
    assert "fp.sid = 1" not in captured["adql"]

    # ... and present when a value IS given, so the filter still works.
    captured.clear()
    tap.night_detections(61500.0, 61501.0, parallax_min_mas=None, sid_diaobject=1)
    assert "d.sid = 1" in captured["adql"]


def test_diagnostics_isolates_each_failure_instead_of_raising():
    """One failing diagnostic must not mask the others --- that is the whole point."""
    tap = AlerceTAP()
    calls = []

    def _flaky(adql, maxrec=None, retries=4):
        calls.append(adql)
        if "TAP_SCHEMA" in adql:
            raise BrokerError("schema unavailable")
        return [{"n": 7}]

    tap.query = _flaky
    diag = tap.diagnostics(61500.0)
    assert "error" in diag["tables"]                    # the failure is recorded
    assert diag["sample_detection"]["rows"] == 1        # ... and does not stop the rest
    assert "count_detection_last_30d" in diag
    assert all("adql" in v for v in diag.values())


def test_every_discriminator_column_is_actually_requested():
    """Regression: a discriminator with no column to read is silently inert.

    The funnel will happily run the glint, mover, dipole and artefact tests
    against fields that were never SELECTed — finding nothing to test, and
    passing everything.  That is worse than not having the tests, because the
    funnel counts still look like vetting happened.
    """
    tap = AlerceTAP()
    captured = {}
    tap.query = lambda adql, maxrec=None, retries=4: (
        captured.setdefault("adql", adql) and [])
    tap.night_detections(61230.0, 61235.0, parallax_min_mas=10.0)
    adql = captured["adql"]
    for col in ("templateflux", "templatefluxerr", "glint_trail", "traillength",
                "isdipole", "isnegative", "extendedness", "snr", "reliability",
                "raerr", "decerr", "pixelflags_cr", "pixelflags_streak",
                "pixelflags_injected", "pixelflags_saturated"):
        assert f"ld.{col}" in adql, f"{col} is never fetched"
    # ...and the column that breaks VOTable serialisation stays out of SELECT.
    assert "ld.ssobjectid" not in adql


def test_alerce_forced_photometry_query_is_the_denominator_not_the_numerator():
    tap = AlerceTAP()
    captured = {}
    tap.query = lambda adql, maxrec=None: captured.setdefault("adql", adql) and []
    tap.forced_photometry_night(61500.0, 61501.0, parallax_min_mas=10.0)
    assert "alerce_tap.forced_photometry" in captured["adql"]
    assert "fp.ra" in captured["adql"] and "fp.dec" in captured["adql"]


def test_lasair_normalisation_maps_the_vetting_fields():
    payload = {"diaObjectId": "170081276982722562", "diaSourcesList": [{
        "diaSourceId": "9", "midpointMjdTai": 61500.3, "band": "r",
        "ra": 10.0, "decl": -20.0, "raErr": 20.0, "decErr": 20.0,
        "psfFlux": 800.0, "psfFluxErr": 20.0, "templateFlux": 8000.0,
        "templateFluxErr": 40.0, "snr": 40.0, "reliability": 0.93,
        "isDipole": False, "isNegative": False, "extendedness": 0.0,
        "trailLength": 0.0, "glint_trail": False}]}
    a = normalize_lasair_diasources(payload)[0]
    assert a.dec == -20.0                     # Lasair spells it `decl`
    assert a.ra_err_arcsec == pytest.approx(0.02)   # mas -> arcsec
    assert a.template_flux_njy == 8000.0
    assert a.broker == "lasair-lsst"


def test_lasair_without_a_token_fails_loudly():
    with pytest.raises(BrokerError) as exc:
        LasairLSST(token="")._post("query", {})
    assert "LASAIR_TOKEN" in str(exc.value)


# ---------------------------------------------------------------------------
# run --- config and honest verdicts
# ---------------------------------------------------------------------------
def test_shipped_config_carries_the_measured_broker_encodings():
    """These are measurements from the live probe, not guesses --- pin them.

    Probe of 2026-07-30 (`results/tocsin/probe.json`):
      * `object_sid_tid` -> sid=0/tid=0 is ZTF, sid=1/tid=1 is LSST diaObject,
        sid=2/tid=1 is LSST ssObject.  So sid=1 selects LSST and excludes known
        solar-system objects.
      * `xmatch_catid`   -> catid 1 is Gaia DR3, catid 0 is AllWISE.
    """
    from seti.tocsin.run import _sid, load_tocsin_config
    conf = load_tocsin_config()
    assert _sid(conf["acquire"]) == 1
    assert int(conf["acquire"]["gaia_catid"]) == 1
    # None still drops the clause, which is what kept the first runs safe.
    assert _sid({"sid_diaobject": None}) is None


def test_config_defaults_are_complete_and_overridable():
    from seti.tocsin.run import DEFAULTS, load_tocsin_config
    conf = load_tocsin_config()
    for section in DEFAULTS:
        assert section in conf
    # The shipped config must keep the deliberate no-extra-reliability-cut.
    assert float(conf["screen"]["min_reliability"]) == 0.0


def test_nightly_screen_end_to_end_against_a_stubbed_broker(tmp_path, monkeypatch):
    """Full nightly path with the network stubbed: broker rows in, ledger out.

    The unit tests cover the pieces; this covers the plumbing between them --- the
    ALeRCE row -> normalisation -> proper-motion match -> funnel -> forced-photometry
    denominator -> ledger -> committed tables chain that the cron actually runs.
    """
    from seti.tocsin import run as R

    ra, dec, pm = 150.0, -30.0, 400.0
    epoch = R.mjd_to_jyear(61500.5) if hasattr(R, "mjd_to_jyear") else 2026.4
    from seti.tocsin.screen import mjd_to_jyear
    epoch = mjd_to_jyear(61500.5)
    ra_now, dec_now = T.propagate_pm(ra, dec, pm, 0.0, GAIA_EPOCH, epoch)

    tpath = tmp_path / "targets.parquet"
    pd.DataFrame([_target(source_id="777", ra=ra, dec=dec, pmra=pm)]).to_parquet(tpath)

    def _row(band_int, flux, mjd):
        return {"measurement_id": f"m{band_int}-{mjd}", "oid": "obj9", "sid": 1,
                "mjd": mjd, "ra": float(ra_now), "dec": float(dec_now),
                "band": band_int, "psfflux": flux, "psffluxerr": abs(flux) * 0.03,
                "snr": 33.0, "reliability": 0.95, "isdipole": False,
                "isnegative": False, "extendedness": 0.0,
                "templateflux": flux / 0.1, "templatefluxerr": flux / 0.1 * 0.01}

    class _StubTAP:
        def __init__(self, *a, **kw):
            pass

        def night_detections(self, lo, hi, **kw):
            from seti.tocsin.brokers import BrokerResult
            # A grey flash: 10% fractional amplitude in both g and r, the two
            # filters of one LSST in-night pair ~33 min apart.
            return BrokerResult(rows=[_row(1, 900.0, 61500.50),
                                      _row(2, 4000.0, 61500.50 + 33.0 / 1440.0)],
                                reached=True, verdict="OK", notes=["adql=stub"])

        def forced_photometry_night(self, lo, hi, **kw):
            from seti.tocsin.brokers import BrokerResult
            return BrokerResult(
                rows=[{"oid": "obj9", "mjd": 61500.50 + 0.001 * i,
                       "ra": float(ra_now), "dec": float(dec_now)}
                      for i in range(2)],
                reached=True, verdict="OK")

    monkeypatch.setattr(R, "AlerceTAP", _StubTAP)

    class _Cfg:
        root = tmp_path
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config" / "tocsin.yaml").write_text(
        "ledger:\n  path: results/tocsin/ledger.json\n")

    out = tmp_path / "results" / "tocsin"
    s = R.screen_night(_Cfg(), targets_path=tpath, out_dir=out,
                       mjd_lo=61500.0, mjd_hi=61501.0)

    assert s["verdict"] == "OK"
    assert s["counts"]["detections_pulled"] == 2
    assert s["counts"]["events_kept"] == 1          # one star-NIGHT, two bands
    assert s["denominator"] == "forced_photometry_exact"
    assert s["counts"]["target_nights_screened"] == 1
    # Trials are attributed per night, so a later overlapping run cannot
    # double-count them.
    assert s["trials_by_night"] == {"n61499": 1}
    # The denominator must always cover the numerator.
    assert s["counts"]["target_nights_screened"] >= s["counts"]["events_kept"]

    led = L.Ledger.load(tmp_path / "results" / "tocsin" / "ledger.json")
    rec = led.targets["777"]
    assert rec["n_events"] == 1
    assert rec["visits_exact"] and rec["n_visits"] == 1
    ev = rec["events"][0]
    assert ev["polarity"] == "flash"
    assert ev["grey_tested"] and abs(ev["grey_z"]) < 3.0
    assert sorted(ev["bands"]) == ["g", "r"]
    # A single grey-confirmed event is `interest`; only REPETITION can take a
    # target to `candidate`, however clean one night looks.
    assert rec["tier"] == "interest"

    # The committed artefacts the cron pushes back must all exist.
    for name in ("summary.json", "watchlist.csv", "events_latest.csv"):
        assert (out / name).exists(), name
    json.loads((out / "summary.json").read_text())
    assert len(pd.read_csv(out / "watchlist.csv")) == 1


def _stub_tap_factory(frontier, seen):
    """A stubbed AlerceTAP that records the window it was asked for."""
    from seti.tocsin.brokers import BrokerResult

    class _Stub:
        def __init__(self, *a, **kw):
            pass

        def max_available_mjd(self):
            return frontier

        def night_detections(self, lo, hi, **kw):
            seen.append((lo, hi))
            return BrokerResult(rows=[], reached=True,
                                verdict="NO_DETECTIONS_IN_WINDOW")

        def forced_photometry_night(self, lo, hi, **kw):
            return BrokerResult(rows=[], reached=True, verdict="OK")

    return _Stub


def test_window_is_anchored_to_the_broker_not_to_the_wall_clock(tmp_path, monkeypatch):
    """Regression: the broker's TAP mirror lags ~2 weeks behind real time.

    Asking for "the last two nights" against a mirror that is 15.6 days behind
    returns nothing every night, forever — and an empty result is
    indistinguishable from a genuine null.  The window must start from the
    newest epoch the broker actually holds.
    """
    from seti.tocsin import run as R

    frontier = 61235.4
    seen: list[tuple[float, float]] = []
    monkeypatch.setattr(R, "AlerceTAP", _stub_tap_factory(frontier, seen))
    tpath = tmp_path / "targets.parquet"
    pd.DataFrame([_target(source_id="777")]).to_parquet(tpath)

    class _Cfg:
        root = tmp_path
    R.screen_night(_Cfg(), targets_path=tpath, out_dir=tmp_path / "res")

    assert seen, "no query was issued"
    lo, hi = seen[0]
    # The window ends at the broker's frontier, never at "now": the mirror was
    # 15.6 days behind when measured, so a clock-anchored window is empty.
    assert hi <= frontier + 1e-6
    assert hi < R._now_mjd() - 5.0
    assert lo < hi


def test_the_watermark_advances_and_does_not_re_screen(tmp_path, monkeypatch):
    """Consecutive runs must be gapless AND non-overlapping."""
    from seti.tocsin import run as R

    frontier = 61235.4
    seen: list[tuple[float, float]] = []
    monkeypatch.setattr(R, "AlerceTAP", _stub_tap_factory(frontier, seen))
    tpath = tmp_path / "targets.parquet"
    pd.DataFrame([_target(source_id="777")]).to_parquet(tpath)

    class _Cfg:
        root = tmp_path
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config" / "tocsin.yaml").write_text(
        "acquire:\n  max_nights_per_run: 3.0\nledger:\n"
        "  path: results/tocsin/ledger.json\n")

    # Start well behind the frontier so the CHUNK CAP is the binding constraint
    # (this is the backfill case: ~262 nights of LSST data already exist).
    led = L.Ledger()
    led.last_mjd_screened = frontier - 10.0
    led.save(tmp_path / "results" / "tocsin" / "ledger.json")

    s1 = R.screen_night(_Cfg(), targets_path=tpath, out_dir=tmp_path / "res")
    s2 = R.screen_night(_Cfg(), targets_path=tpath, out_dir=tmp_path / "res")
    assert len(seen) == 2
    (lo1, hi1), (lo2, hi2) = seen
    assert lo1 == pytest.approx(frontier - 10.0)
    assert hi1 - lo1 == pytest.approx(3.0)      # chunk cap respected
    assert lo2 == pytest.approx(hi1)            # gapless, and no overlap
    assert hi2 - lo2 == pytest.approx(3.0)
    assert s1["watermark_mjd"] == pytest.approx(hi1)
    assert s2["watermark_mjd"] == pytest.approx(hi2)


def test_chunked_backfill_walks_forward_and_stops_when_caught_up(tmp_path,
                                                                 monkeypatch):
    """One dispatch should be able to walk several windows of the backlog.

    The broker holds ~262 nights already, so the early runs are a backfill; at
    one window per dispatch that would take weeks of wall clock for data that is
    already sitting there.
    """
    from seti.tocsin import run as R

    frontier = 61010.0
    seen: list[tuple[float, float]] = []
    monkeypatch.setattr(R, "AlerceTAP", _stub_tap_factory(frontier, seen))
    tpath = tmp_path / "targets.parquet"
    pd.DataFrame([_target(source_id="777")]).to_parquet(tpath)

    class _Cfg:
        root = tmp_path
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config" / "tocsin.yaml").write_text(
        "acquire:\n  max_nights_per_run: 4.0\n  backfill_start_mjd: 61000.0\n"
        "ledger:\n  path: results/tocsin/ledger.json\n")

    R.screen(_Cfg(), chunks=5, targets_path=tpath, out_dir=tmp_path / "res")
    # 61000 -> 61004 -> 61008 -> 61010 (frontier), then NO_NEW_DATA stops it.
    assert [round(lo, 1) for lo, _ in seen] == [61000.0, 61004.0, 61008.0]
    assert seen[-1][1] == pytest.approx(frontier)


def test_a_caught_up_watermark_reports_no_new_data(tmp_path, monkeypatch):
    """Reaching the frontier is a distinct verdict, not an empty candidate table."""
    from seti.tocsin import run as R

    frontier = 61235.4
    seen: list[tuple[float, float]] = []
    monkeypatch.setattr(R, "AlerceTAP", _stub_tap_factory(frontier, seen))
    tpath = tmp_path / "targets.parquet"
    pd.DataFrame([_target(source_id="777")]).to_parquet(tpath)

    class _Cfg:
        root = tmp_path
    led = L.Ledger()
    led.last_mjd_screened = frontier
    led.save(tmp_path / "results" / "tocsin" / "ledger.json")

    s = R.screen_night(_Cfg(), targets_path=tpath, out_dir=tmp_path / "res")
    assert s["verdict"] == "NO_NEW_DATA"
    assert seen == []                           # no pointless broker call


def test_an_explicit_window_does_not_move_the_watermark(tmp_path, monkeypatch):
    """A manual re-run of an old window must not skip forward over unseen data."""
    from seti.tocsin import run as R

    seen: list[tuple[float, float]] = []
    monkeypatch.setattr(R, "AlerceTAP", _stub_tap_factory(61235.4, seen))
    tpath = tmp_path / "targets.parquet"
    pd.DataFrame([_target(source_id="777")]).to_parquet(tpath)

    class _Cfg:
        root = tmp_path
    s = R.screen_night(_Cfg(), targets_path=tpath, out_dir=tmp_path / "res",
                       mjd_lo=61000.0, mjd_hi=61001.0)
    assert s["explicit_window"] is True
    assert seen[0] == (61000.0, 61001.0)
    led = L.Ledger.load(tmp_path / "results" / "tocsin" / "ledger.json")
    assert L._finite(led.last_mjd_screened) is None


def test_detections_without_forced_photometry_still_count_as_trials(tmp_path,
                                                                    monkeypatch):
    """A star-night that produced a detection was, by definition, observed.

    The first live run returned 62 events against only 8 forced-photometry
    star-nights, giving an "event rate" of 7.75 per star-night.  The union makes
    trials >= events by construction, and the poor forced coverage is reported
    rather than hidden — a rate near 1 promotes nobody, which is correct.
    """
    from seti.tocsin import run as R
    from seti.tocsin.brokers import BrokerResult

    ra, dec = 150.0, -30.0
    tpath = tmp_path / "targets.parquet"
    pd.DataFrame([_target(source_id="777", ra=ra, dec=dec)]).to_parquet(tpath)

    class _Stub:
        def __init__(self, *a, **kw):
            pass

        def max_available_mjd(self):
            return 61501.0

        def night_detections(self, lo, hi, **kw):
            return BrokerResult(rows=[{
                "measurement_id": "m1", "oid": "o1", "sid": 1, "mjd": 61500.5,
                "ra": ra, "dec": dec, "band": 1, "psfflux": 900.0,
                "psffluxerr": 27.0, "snr": 33.0, "reliability": 0.95,
                "templateflux": 9000.0, "templatefluxerr": 90.0,
                "extendedness": 0.0, "isdipole": False}],
                reached=True, verdict="OK", notes=["adql=stub"])

        def forced_photometry_night(self, lo, hi, **kw):
            return BrokerResult(rows=[], reached=True, verdict="OK")

    monkeypatch.setattr(R, "AlerceTAP", _Stub)

    class _Cfg:
        root = tmp_path
    s = R.screen_night(_Cfg(), targets_path=tpath, out_dir=tmp_path / "res")

    assert s["counts"]["events_kept"] == 1
    assert s["counts"]["target_nights_screened"] == 1        # from the detection
    assert s["counts"]["target_nights_with_forced_photometry"] == 0
    assert s["denominator"] == "detection_dominated_lower_bound"
    assert any("upper bound" in n.lower() or "UPPER bound" in n for n in s["notes"])
    # The rate is now a legal probability, and nobody is promoted on it.
    rate = s["ledger"]["ensemble_rate_per_target_visit"]
    assert rate is None or 0.0 <= rate <= 1.0
    assert s["ledger"]["tier_counts"]["candidate"] == 0


def test_screen_night_reports_an_unreachable_broker_as_such(tmp_path, monkeypatch):
    """An unreachable broker must never look like a clean null."""
    from seti.tocsin import run as R
    from seti.tocsin.brokers import BrokerError

    tpath = tmp_path / "targets.parquet"
    pd.DataFrame([_target(source_id="777")]).to_parquet(tpath)

    class _DeadTAP:
        def __init__(self, *a, **kw):
            pass

        def night_detections(self, *a, **kw):
            raise BrokerError("TAP query failed after 4 attempts: connection refused")

    monkeypatch.setattr(R, "AlerceTAP", _DeadTAP)

    class _Cfg:
        root = tmp_path
    out = tmp_path / "res"
    s = R.screen_night(_Cfg(), targets_path=tpath, out_dir=out,
                       mjd_lo=61500.0, mjd_hi=61501.0)
    assert s["verdict"] == "NO_DATA_REACHED"
    assert "connection refused" in s["error"]
    assert json.loads((out / "summary.json").read_text())["verdict"] == "NO_DATA_REACHED"


def test_screen_night_without_a_target_list_says_so(tmp_path):
    from seti.tocsin.run import screen_night
    out = tmp_path / "res"
    s = screen_night(targets_path=tmp_path / "absent.parquet", out_dir=out,
                     mjd_lo=61500.0, mjd_hi=61501.0)
    assert s["verdict"] == "NO_TARGET_LIST"
    assert json.loads((out / "summary.json").read_text())["verdict"] == "NO_TARGET_LIST"
