"""Offline suite for METRONOME's single-star vet (seti.metronome.vetstar).

The vet exists to kill one hypothesis: that a strict clock in a star's flare
times is an eclipsing binary (or any other low-amplitude periodicity) whose
crests or eclipses the flare detector is counting as events.  So the suite is
built the way the contract requires:

* an INJECTED eclipsing binary --- folded at half its orbital period, which is
  the classic trap --- must be recovered and named;
* a genuine flare clock on an otherwise quiet star must return a CLEAN NULL:
  no coherent oscillation survives masking the events, no narrow dip, no
  unequal minima;
* a low-amplitude sinusoid whose crests are detected as flares --- the
  dominant confounder, and the one the earlier ``period_is_photometric`` test
  could not see --- must be caught;
* an archive that cannot be reached must DEGRADE honestly: ``UNREACHED`` is
  never written as ``NOT_LISTED``, and the verdict says so;
* every rejection rule in :func:`vet_verdict` must be trippable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seti.metronome.vetstar import (
    STATUS_ABSENT,
    STATUS_OK,
    STATUS_UNREACHED,
    analyse_lightcurve,
    angular_separation_arcsec,
    catalogued_binary_hits,
    detrend_fractional,
    fold_amplitude_significance,
    fold_profile,
    gaia_neighbourhood,
    gaia_variability,
    harmonic_content,
    neighbour_context,
    odd_even_minima,
    periodogram_at,
    phase_separation,
    propagate_position,
    rayleigh,
    roll_season_test,
    vet_verdict,
    vizier_by_identifier,
    vizier_cone_report,
)

CADENCE = 0.0204340          # Kepler long cadence, days
PERIOD = 0.42328185409991    # the clock under test
PROT = 11.05                 # the star's rotation


# ---------------------------------------------------------------------------
# synthetic light curves
# ---------------------------------------------------------------------------
def _quarters(n_quarters: int = 6, days: float = 30.0, gap: float = 3.0):
    """Contiguous runs with Kepler-like inter-quarter gaps."""
    segs = []
    t0 = 130.0
    for q in range(n_quarters):
        t = np.arange(t0, t0 + days, CADENCE)
        segs.append({"sector": q + 1, "time": t, "exptime_s": CADENCE * 86400.0})
        t0 = t[-1] + gap
    return segs


def _finish(segs, rng, *, noise: float = 3e-4, rot_amp: float = 6e-3):
    """Add rotation and white noise, normalise to a flux scale."""
    for s in segs:
        t = s["time"]
        base = 1.0 + rot_amp * 0.5 * np.sin(2 * np.pi * t / PROT + 0.7)
        s["flux"] = (base + s.pop("signal", np.zeros(len(t)))) * 1.0e5 \
            * (1.0 + rng.normal(0.0, noise, len(t)))
    return segs


def eclipsing_binary_segments(seed=7, depth_primary=4e-3, depth_secondary=2e-3,
                              width=0.035):
    """A detached EB whose ORBITAL period is 2 * PERIOD, so a period search that
    does not distinguish the two minima lands on PERIOD."""
    rng = np.random.default_rng(seed)
    segs = _quarters()
    porb = 2.0 * PERIOD
    for s in segs:
        t = s["time"]
        ph = np.mod(t / porb, 1.0)
        sig = np.zeros(len(t))
        d1 = np.minimum(np.abs(ph - 0.25), 1.0 - np.abs(ph - 0.25))
        d2 = np.minimum(np.abs(ph - 0.75), 1.0 - np.abs(ph - 0.75))
        sig -= depth_primary * np.exp(-0.5 * (d1 / width) ** 2)
        sig -= depth_secondary * np.exp(-0.5 * (d2 / width) ** 2)
        s["signal"] = sig
    return _finish(segs, rng)


def flare_clock_segments(seed=11, amp=6e-3, jitter=0.0, occupancy=0.05):
    """A quiet star whose FLARES are on a strict clock at PERIOD: fast rise,
    exponential decay, and NO underlying periodicity at that period.

    ``occupancy`` is the fraction of cycles that actually carry a flare, and it
    is the crux.  The real candidate fires on 2.3% of its cycles over 3169 of
    them; a star that fired on EVERY cycle would be a periodic light curve, not
    a clock, and nothing could tell the two apart.  A sparse clock, masked,
    leaves a flat light curve behind — which is exactly what the vet measures.
    """
    rng = np.random.default_rng(seed)
    segs = _quarters()
    for s in segs:
        t = s["time"]
        sig = np.zeros(len(t))
        k0 = int(np.floor(t[0] / PERIOD)) + 1
        k1 = int(np.ceil(t[-1] / PERIOD))
        for k in range(k0, k1 + 1):
            if rng.uniform() > occupancy:
                continue
            tp = k * PERIOD + (rng.normal(0.0, jitter * PERIOD) if jitter else 0.0)
            dt = np.clip(t - tp, -50.0 * CADENCE, 50.0 * CADENCE)
            prof = np.where(dt < 0, np.exp(dt / (0.4 * CADENCE)),
                            np.exp(-dt / (1.3 * CADENCE)))
            sig += amp * prof
        s["signal"] = sig
    return _finish(segs, rng)


def sinusoid_crest_segments(seed=13, amp=2.2e-3):
    """A COHERENT low-amplitude sinusoid at PERIOD, small enough that the
    11-day rotation still dominates the raw periodogram.  Its crests are what
    the flare detector will call events."""
    rng = np.random.default_rng(seed)
    segs = _quarters()
    for s in segs:
        t = s["time"]
        ph = 2 * np.pi * t / PERIOD
        # peaked crests: a sinusoid raised to a power keeps one maximum per
        # cycle but narrows it, which is what actually fools the detector
        s["signal"] = amp * np.maximum(np.cos(ph), 0.0) ** 3
    return _finish(segs, rng, noise=2.0e-4)


# ---------------------------------------------------------------------------
# folding and shape
# ---------------------------------------------------------------------------
def test_detrend_keeps_the_short_period_and_removes_the_rotation():
    rng = np.random.default_rng(3)
    t = np.arange(0.0, 60.0, CADENCE)
    clock = 4e-3 * np.sin(2 * np.pi * t / PERIOD)
    rot = 2e-2 * np.sin(2 * np.pi * t / PROT)
    f = 1.0 + clock + rot + rng.normal(0, 1e-5, len(t))
    td, yd = detrend_fractional(t, f, cadence_days=CADENCE, window_days=2.0)
    ok = np.isfinite(yd)
    h = harmonic_content(td, yd, PERIOD)
    # the clock survives at most of its amplitude ...
    assert h["amplitudes"][0] == pytest.approx(4e-3, rel=0.25)
    # ... and the rotation is gone: the residual's own scatter is far below it
    assert float(np.std(yd[ok])) < 6e-3


def test_fold_of_a_sinusoid_is_wide_and_of_an_eclipse_is_narrow():
    t = np.arange(0.0, 80.0, CADENCE)
    sine = 1.0 + 3e-3 * np.sin(2 * np.pi * t / PERIOD)
    ph = np.mod(t / PERIOD, 1.0)
    d = np.minimum(np.abs(ph - 0.5), 1.0 - np.abs(ph - 0.5))
    ecl = 1.0 - 3e-3 * np.exp(-0.5 * (d / 0.03) ** 2)
    fs = fold_profile(t, sine - 1.0, PERIOD, n_bins=100)
    fe = fold_profile(t, ecl - 1.0, PERIOD, n_bins=100)
    assert fs["frac_below_half_depth"] == pytest.approx(1.0 / 3.0, abs=0.06)
    assert fe["frac_below_half_depth"] < 0.15
    assert fe["extremum_is_dip"] is True
    assert fe["depth"] > 5 * fe["height"]


def test_harmonic_content_of_a_pure_sinusoid_is_all_fundamental():
    t = np.arange(0.0, 50.0, CADENCE)
    y = 1e-3 * np.cos(2 * np.pi * t / PERIOD + 0.3)
    h = harmonic_content(t, y, PERIOD)
    assert h["fundamental_fraction"] > 0.99
    assert h["a2_over_a1"] < 0.05


def test_odd_even_separates_unequal_minima_and_passes_equal_ones():
    t = np.arange(0.0, 150.0, CADENCE)
    porb = 2.0 * PERIOD
    ph = np.mod(t / porb, 1.0)

    def eclipse(centre, depth, width=0.02):
        d = np.minimum(np.abs(ph - centre), 1.0 - np.abs(ph - centre))
        return -depth * np.exp(-0.5 * (d / width) ** 2)

    rng = np.random.default_rng(5)
    noise = rng.normal(0, 5e-5, len(t))
    unequal = eclipse(0.25, 4e-3) + eclipse(0.75, 1.5e-3) + noise
    equal = eclipse(0.25, 4e-3) + eclipse(0.75, 4e-3) + noise
    oe_u = odd_even_minima(t, unequal, porb, n_bins=100)
    oe_e = odd_even_minima(t, equal, porb, n_bins=100)
    assert oe_u["delta_sigma"] > 5.0
    assert oe_e["delta_sigma"] < 3.0
    assert oe_u["amplitude_sigma"] > 5.0


def test_periodogram_at_finds_a_buried_signal_and_ranks_it():
    rng = np.random.default_rng(9)
    t = np.arange(0.0, 200.0, CADENCE)
    y = 1e-3 * np.sin(2 * np.pi * t / PERIOD) + rng.normal(0, 3e-4, len(t))
    pg = periodogram_at(t, y, PERIOD, cadence_days=CADENCE)
    assert pg["rank_of_period"] == 1
    assert pg["amplitude_frac_at_period"] == pytest.approx(2e-3, rel=0.2)
    assert pg["fap_at_period"] < 1e-6
    # and on pure noise there is nothing there
    y2 = rng.normal(0, 3e-4, len(t))
    pg2 = periodogram_at(t, y2, PERIOD, cadence_days=CADENCE)
    assert pg2["rank_of_period"] > 1
    assert pg2["fap_at_period"] > 1e-3


# ---------------------------------------------------------------------------
# circular statistics
# ---------------------------------------------------------------------------
def test_rayleigh_uniform_versus_clustered():
    rng = np.random.default_rng(2)
    uni = rayleigh(rng.uniform(0, 1, 400))
    assert uni["p"] > 0.05
    clus = rayleigh(np.mod(rng.normal(0.3, 0.02, 60), 1.0))
    assert clus["p"] < 1e-20
    assert clus["mean_phase"] == pytest.approx(0.3, abs=0.02)


def test_phase_separation_wraps():
    assert phase_separation(0.95, 0.05) == pytest.approx(-0.1)
    assert phase_separation(0.05, 0.95) == pytest.approx(0.1)
    assert np.isnan(phase_separation(float("nan"), 0.1))


def test_proper_motion_propagation_and_separation():
    # 1000 mas/yr in RA for 16 years is 16 arcsec of great-circle motion at dec 0
    ra, dec = 290.0, 0.0
    r, d = propagate_position(ra, dec, 1000.0, 0.0, from_epoch=2016.0, to_epoch=2000.0)
    assert angular_separation_arcsec(ra, dec, r, d) == pytest.approx(16.0, rel=1e-3)
    # no proper motion, no motion
    r2, d2 = propagate_position(ra, dec, float("nan"), float("nan"),
                                from_epoch=2016.0, to_epoch=2000.0)
    assert (r2, d2) == (ra, dec)


# ---------------------------------------------------------------------------
# the three light-curve scenarios end to end
# ---------------------------------------------------------------------------
def test_injected_eclipsing_binary_is_recovered_and_named():
    rep = analyse_lightcurve(eclipsing_binary_segments(), PERIOD)
    rep["catalogued_binary_hits"] = []
    assert rep["status"] == "analysed"
    oe = rep["odd_even"]
    assert oe["delta_sigma"] > 3.0
    assert rep["fold_significance_2p"]["p_empirical"] <= 0.01
    assert rep["fold_significance_2p"]["z_control"] > 5.0
    f1 = rep["fold_at_period"]
    assert f1["extremum_is_dip"] is True
    assert f1["frac_below_half_depth"] < 0.25
    assert rep["fold_significance"]["z_control"] > 5.0
    verdict, _surviving = vet_verdict(rep)
    assert "UNEQUAL_MINIMA_AT_2P" in verdict
    assert "NARROW_DIP_AT_P" in verdict
    assert verdict.startswith("MUNDANE_EXPLANATION_FOUND")


def test_a_real_flare_clock_returns_a_clean_null():
    rep = analyse_lightcurve(flare_clock_segments(), PERIOD)
    rep["catalogued_binary_hits"] = []
    rep["gaia"] = {"status": STATUS_OK, "match": {"ruwe": 1.02, "non_single_star": 0}}
    rep["gaia_variability"] = {"gaiadr3.vari_summary": {"status": STATUS_ABSENT}}
    rep["vizier_cones"] = {"vsx": {"status": STATUS_ABSENT}}
    rep["vizier_by_id"] = {"kepler_eb_kirk2016": {"status": STATUS_ABSENT}}
    # the detector does find the injected flares ...
    assert rep["n_flares_redetected"] > 20
    # ... and the events cluster in phase at the clock period, by construction
    assert rep["event_phase_rayleigh_at_p"]["p"] < 1e-8
    # but nothing periodic survives masking them: the folded amplitude is no
    # bigger than the same fold at 200 control periods on the same light curve
    ctl = rep["fold_significance_flares_masked"]
    assert ctl["p_empirical"] > 0.01
    assert ctl["z_control"] < 5.0
    assert ctl["amplitude_ptp"] <= ctl["control_p95"]
    assert rep["odd_even"]["delta_sigma"] < 3.0
    verdict, surviving = vet_verdict(rep)
    assert verdict.startswith("NO_MUNDANE_EXPLANATION_FOUND"), verdict
    assert any("masking" in s for s in surviving)


def test_a_low_amplitude_oscillation_whose_crests_are_counted_as_flares_is_caught():
    rep = analyse_lightcurve(sinusoid_crest_segments(), PERIOD)
    rep["catalogued_binary_hits"] = []
    assert rep["n_flares_redetected"] > 20
    # the oscillation is NOT the dominant periodicity of the raw light curve
    # (the 11-day rotation is), which is what the earlier veto could not see
    ctl = rep["fold_significance_flares_masked"]
    assert ctl["p_empirical"] <= 0.01
    assert ctl["z_control"] > 5.0
    assert abs(rep["event_phase_offset_from_photometric_max"]) < 0.15
    # and it is NOT mistaken for an eclipsing binary: the two half-phase
    # minima of the 2P fold are equal
    assert rep["odd_even"]["delta_sigma"] < 3.0
    verdict, _ = vet_verdict(rep)
    assert "COHERENT_OSCILLATION_AT_P" in verdict
    assert "EVENTS_ON_THE_CREST" in verdict


def test_masking_the_events_is_what_separates_the_two():
    """The one measurement that distinguishes a flare clock from an
    oscillation whose crests are counted: the folded amplitude with the events
    removed."""
    clock = analyse_lightcurve(flare_clock_segments(), PERIOD)
    osc = analyse_lightcurve(sinusoid_crest_segments(), PERIOD)
    # unmasked, BOTH raise the folded amplitude above the per-bin standard
    # error --- which is exactly why that quantity is not a usable threshold
    assert clock["fold_at_period"]["amplitude_sigma"] > 5.0
    assert osc["fold_at_period"]["amplitude_sigma"] > 5.0
    # against control periods on the same light curve they separate cleanly,
    # masked or not
    assert osc["fold_significance"]["p_empirical"] <= 0.01
    assert clock["fold_significance"]["p_empirical"] > 0.05
    assert osc["fold_significance_flares_masked"]["z_control"] > 5.0
    assert clock["fold_significance_flares_masked"]["z_control"] < 5.0
    # and by a wide margin, not a threshold's worth
    assert (osc["fold_significance_flares_masked"]["amplitude_ptp"]
            > 5.0 * clock["fold_significance_flares_masked"]["amplitude_ptp"])


def test_per_segment_amplitudes_are_reported_per_quarter():
    rep = analyse_lightcurve(eclipsing_binary_segments(), PERIOD)
    seg = rep["per_segment"]
    assert len(seg) == 6
    amps = [r["amplitude_ptp"] for r in seg]
    assert all(np.isfinite(a) and a > 0 for a in amps)
    # an intrinsic signal has a stable amplitude between quarters
    assert max(amps) / min(amps) < 3.0


def test_blend_like_amplitude_swing_between_segments_is_flagged():
    segs = eclipsing_binary_segments()
    rep = analyse_lightcurve(segs, PERIOD)
    rep["catalogued_binary_hits"] = []
    rep["per_segment"] = [{"amplitude_ptp": a} for a in (1e-3, 1.1e-3, 9e-3, 1.2e-3, 1e-2)]
    verdict, _ = vet_verdict(rep)
    assert "AMPLITUDE_VARIES_BETWEEN_SEGMENTS" in verdict


def test_too_short_a_lightcurve_says_so_rather_than_guessing():
    segs = [{"sector": 1, "time": np.arange(0.0, 1.0, CADENCE),
             "flux": np.ones(len(np.arange(0.0, 1.0, CADENCE))) * 1e5,
             "exptime_s": CADENCE * 86400.0}]
    rep = analyse_lightcurve(segs, PERIOD)
    assert rep["status"] == "TOO_FEW_POINTS"


# ---------------------------------------------------------------------------
# archives: OK / NOT_LISTED / UNREACHED are three different things
# ---------------------------------------------------------------------------
def test_gaia_variability_separates_listed_absent_and_unreached():
    def qf(adql):
        if "vari_eclipsing_binary" in adql:
            return pd.DataFrame({"source_id": [123], "frequency": [2.3624]})
        if "vari_summary" in adql:
            return pd.DataFrame()
        raise RuntimeError("503 from gea.esac.esa.int")

    out = gaia_variability("123", query_fn=qf)
    assert out["gaiadr3.vari_eclipsing_binary"]["status"] == STATUS_OK
    assert out["gaiadr3.vari_eclipsing_binary"]["row"]["frequency"] == pytest.approx(2.3624)
    assert out["gaiadr3.vari_summary"]["status"] == STATUS_ABSENT
    assert out["gaiadr3.vari_rotation_modulation"]["status"] == STATUS_UNREACHED
    assert "503" in out["gaiadr3.vari_rotation_modulation"]["error"]


def test_gaia_neighbourhood_orders_by_separation_at_the_catalogue_epoch():
    # a high-proper-motion source 16 arcsec away in DR3 sits on top of the
    # target at epoch 2000; a static source 3 arcsec away does not move
    def qf(adql):
        return pd.DataFrame({
            "source_id": [1, 2],
            "ra": [290.0, 290.0], "dec": [45.0 + 16.0 / 3600.0, 45.0 + 3.0 / 3600.0],
            "pmra": [0.0, 0.0], "pmdec": [1000.0, 0.0], "ruwe": [1.1, 0.9],
            "non_single_star": [0, 0], "phot_g_mean_mag": [14.0, 17.0]})

    out = gaia_neighbourhood(290.0, 45.0, radius_arcsec=20.0, query_fn=qf)
    assert out["status"] == STATUS_OK
    assert out["n_sources"] == 2
    assert out["match"]["source_id"] == "1"
    assert out["match"]["sep_arcsec_at_epoch"] == pytest.approx(0.0, abs=1e-6)
    assert out["match"]["sep_arcsec_gaia_epoch"] == pytest.approx(16.0, rel=1e-3)
    assert len(out["neighbours"]) == 1


def test_gaia_neighbourhood_degrades_honestly_when_the_service_is_down():
    def qf(adql):
        raise RuntimeError("CONNECT tunnel failed, response 403")

    out = gaia_neighbourhood(290.0, 45.0, query_fn=qf)
    assert out["status"] == STATUS_UNREACHED
    assert "403" in out["error"]
    assert out["n_sources"] == 0


def test_gaia_neighbourhood_empty_cone_is_not_listed_not_unreached():
    out = gaia_neighbourhood(290.0, 45.0, query_fn=lambda a: pd.DataFrame())
    assert out["status"] == STATUS_ABSENT


def test_vizier_cone_report_records_rows_absences_and_failures():
    def cone(table, ra, dec, radius):
        if table.startswith("B/vsx"):
            return pd.DataFrame({"Name": ["V* AB Cyg"], "Type": ["EW"], "Period": [0.84656]})
        if table.startswith("I/358/veb"):
            return pd.DataFrame()
        raise RuntimeError("VizieR timed out")

    out = vizier_cone_report(290.0, 45.0, cone_fn=cone)
    assert out["vsx"]["status"] == STATUS_OK
    assert out["vsx"]["rows"][0]["Type"] == "EW"
    assert out["gaia_dr3_veb"]["status"] == STATUS_ABSENT
    assert out["ztf_chen2020"]["status"] == STATUS_UNREACHED


def test_vizier_by_identifier_uses_the_schema_it_read():
    seen = {}

    def qf(adql):
        if "TAP_SCHEMA.tables" in adql:
            return pd.DataFrame({"table_name": ['"J/AJ/151/68/table2"'],
                                 "description": ["Kepler eclipsing binaries"]})
        if "TAP_SCHEMA.columns" in adql:
            return pd.DataFrame({"column_name": ['"KIC"', '"Period"', '"pdepth"']})
        seen["query"] = adql
        return pd.DataFrame({"KIC": [5879574], "Period": [0.84656], "pdepth": [0.02]})

    out = vizier_by_identifier("5879574", "kepler", query_fn=qf)
    rec = out["kepler_eb_kirk2016"]
    assert rec["status"] == STATUS_OK
    assert rec["n_rows"] == 1
    assert '"KIC" = 5879574' in seen["query"]
    assert rec["rows"][0]["Period"] == pytest.approx(0.84656)


def test_vizier_by_identifier_absent_is_not_unreached():
    def qf(adql):
        if "TAP_SCHEMA.tables" in adql:
            return pd.DataFrame({"table_name": ['"J/AJ/151/68/table2"'], "description": [""]})
        if "TAP_SCHEMA.columns" in adql:
            return pd.DataFrame({"column_name": ['"KIC"']})
        return pd.DataFrame()

    out = vizier_by_identifier("5879574", "kepler", query_fn=qf)
    assert out["kepler_eb_kirk2016"]["status"] == STATUS_ABSENT

    def qf_down(adql):
        raise RuntimeError("no route to TAP_SCHEMA")

    out2 = vizier_by_identifier("5879574", "kepler", query_fn=qf_down)
    assert out2["kepler_eb_kirk2016"]["status"] == STATUS_UNREACHED


def test_tess_star_skips_the_kepler_only_identifier_tables():
    out = vizier_by_identifier("149573659", "tess", query_fn=lambda a: pd.DataFrame())
    assert out == {}


# ---------------------------------------------------------------------------
# catalogue interpretation and the verdict
# ---------------------------------------------------------------------------
def test_catalogued_binary_hits_matches_type_and_period_at_p_and_2p():
    report = {
        "vsx": {"table": "B/vsx/vsx",
                "rows": [{"Name": "V1", "Type": "EW", "Period": 0.84656}]},
        "rot": {"table": "J/ApJS/211/24/table1",
                "rows": [{"KIC": "5879574", "Prot": 11.107}]},
    }
    hits = catalogued_binary_hits(report, PERIOD)
    assert len(hits) == 1
    assert hits[0]["source"] == "vsx"
    assert hits[0]["typed_eclipsing"] is True
    assert hits[0]["periods_matching_clock"] == [pytest.approx(0.84656)]


def test_a_rotation_only_listing_is_not_a_binary_hit():
    report = {"vsx": {"table": "B/vsx/vsx",
                      "rows": [{"Name": "V2", "Type": "ROT", "Period": 11.107}]}}
    assert catalogued_binary_hits(report, PERIOD) == []


def test_catalogued_listing_alone_settles_it():
    rep = {"period": PERIOD,
           "catalogued_binary_hits": [{"source": "kepler_eb_kirk2016", "table": "x",
                                       "typed_eclipsing": True, "periods": [0.84656],
                                       "periods_matching_clock": [0.84656], "row": {}}]}
    verdict, _ = vet_verdict(rep)
    assert "CATALOGUED_ECLIPSING_BINARY:kepler_eb_kirk2016" in verdict
    assert f"P={PERIOD:.6f} d" in verdict


@pytest.mark.parametrize(("patch", "token"), [
    ({"gaia": {"status": STATUS_OK, "match": {"ruwe": 2.6}}}, "GAIA_RUWE=2.60"),
    ({"gaia": {"status": STATUS_OK, "match": {"non_single_star": 1}}},
     "GAIA_NON_SINGLE_STAR=1"),
    ({"gaia": {"status": STATUS_OK, "match": {"rv_amplitude_robust": 64.0}}},
     "GAIA_RV_AMPLITUDE=64.0km/s"),
    ({"odd_even": {"delta_sigma": 8.0},
      "fold_significance_2p": {"p_empirical": 0.005, "z_control": 40.0}},
     "UNEQUAL_MINIMA_AT_2P"),
    ({"fold_at_period": {"extremum_is_dip": True, "frac_below_half_depth": 0.08},
      "fold_significance": {"p_empirical": 0.005, "z_control": 40.0}}, "NARROW_DIP_AT_P"),
    ({"fold_at_period_flares_masked": {"amplitude_ptp": 2e-3},
      "fold_significance_flares_masked": {"p_empirical": 0.005, "z_control": 40.0}},
     "COHERENT_OSCILLATION_AT_P"),
    ({"event_phase_rayleigh_at_p": {"rbar": 0.2},
      "event_phase_rayleigh_at_2p": {"p": 1e-9, "rbar": 0.8}},
     "EVENTS_CLUSTER_MORE_TIGHTLY_AT_2P"),
])
def test_every_rejection_rule_is_trippable(patch, token):
    rep = {"period": PERIOD, "catalogued_binary_hits": []}
    rep.update(patch)
    verdict, _ = vet_verdict(rep)
    assert token in verdict
    assert verdict.startswith("MUNDANE_EXPLANATION_FOUND")


def test_unreached_archives_make_the_verdict_degraded_not_clean():
    rep = {"period": PERIOD, "catalogued_binary_hits": [],
           "gaia": {"status": STATUS_UNREACHED},
           "gaia_variability": {"gaiadr3.vari_summary": {"status": STATUS_UNREACHED}},
           "vizier_cones": {"vsx": {"status": STATUS_ABSENT},
                            "ztf_chen2020": {"status": STATUS_UNREACHED}},
           "vizier_by_id": {"kepler_eb_kirk2016": {"status": STATUS_OK}}}
    verdict, _ = vet_verdict(rep)
    assert verdict.startswith("DEGRADED(")
    assert "gaia:gaia_source" in verdict
    assert "vizier_cones:ztf_chen2020" in verdict
    assert "vizier_cones:vsx" not in verdict
    assert "NO_MUNDANE_EXPLANATION_FOUND" in verdict
    assert rep["unreached"]


def test_a_fully_reached_clean_star_is_not_degraded():
    rep = {"period": PERIOD, "catalogued_binary_hits": [],
           "gaia": {"status": STATUS_OK, "match": {"ruwe": 1.0, "non_single_star": 0}},
           "gaia_variability": {"gaiadr3.vari_summary": {"status": STATUS_ABSENT}},
           "vizier_cones": {"vsx": {"status": STATUS_ABSENT}},
           "vizier_by_id": {"kepler_eb_kirk2016": {"status": STATUS_ABSENT}}}
    verdict, surviving = vet_verdict(rep)
    assert verdict.startswith("NO_MUNDANE_EXPLANATION_FOUND")
    assert "DEGRADED" not in verdict
    assert surviving


def test_fold_significance_is_a_control_period_null_not_a_bin_error():
    """The per-bin standard error clears five sigma on almost anything once a
    fold has hundreds of cadences per bin; the control-period null does not."""
    rng = np.random.default_rng(21)
    t = np.arange(0.0, 180.0, CADENCE)
    y = rng.normal(0, 3e-4, len(t))
    ctl = fold_amplitude_significance(t, y, PERIOD, n_bins=100, n_control=100, rng=rng)
    assert ctl["n_control"] == 100
    assert ctl["p_empirical"] > 0.05
    assert abs(ctl["z_control"]) < 5.0
    # inject a real signal and it separates
    y2 = y + 1.5e-3 * np.sin(2 * np.pi * t / PERIOD)
    ctl2 = fold_amplitude_significance(t, y2, PERIOD, n_bins=100, n_control=100, rng=rng)
    assert ctl2["p_empirical"] <= 0.01
    assert ctl2["z_control"] > 10.0
    assert ctl2["amplitude_ptp"] > ctl2["control_max"]


def test_control_periods_exclude_the_aliases_that_would_inherit_the_signal():
    """No control period may sit on P, on a low-order rational multiple of it,
    or within the peak's own frequency width.

    Some leakage into the control TAIL is unavoidable and is left in on
    purpose: a per-bin median of a strong coherent signal folded at, say,
    ``5P/3`` still retains a fraction of its amplitude.  That makes the null
    harsher than the truth, never softer, which is the direction a
    contamination test is allowed to err in.
    """
    rng = np.random.default_rng(22)
    t = np.arange(0.0, 120.0, CADENCE)
    y = 1e-3 * np.sin(2 * np.pi * t / PERIOD) + rng.normal(0, 2e-4, len(t))
    ctl = fold_amplitude_significance(t, y, PERIOD, n_bins=80, n_control=150, rng=rng)
    assert ctl["exclusion_frac"] > 0.02          # short span: the width binds
    assert ctl["control_median"] < 0.25 * ctl["amplitude_ptp"]
    assert ctl["control_p95"] < 0.5 * ctl["amplitude_ptp"]
    assert ctl["control_max"] < ctl["amplitude_ptp"]
    assert ctl["p_empirical"] <= 1.0 / 150.0 + 1e-9
    assert ctl["z_control"] > 10.0


# ---------------------------------------------------------------------------
# the stage end to end, entirely offline
# ---------------------------------------------------------------------------
def _stage_conf():
    from seti.metronome.run import load_metronome_config

    conf = load_metronome_config()
    conf.setdefault("vetstar", {}).update(
        {"star_key": "kepler:5879574", "catalogue": "kepler_yang2019",
         "period_days": PERIOD, "budget_s": 120.0, "per_target_budget_s": 30.0,
         "max_quarters": 8})
    conf.setdefault("redetect", {})["retries"] = 1
    return conf


def test_stage_writes_a_verdict_and_a_fold_table(tmp_path):
    import json

    from seti.metronome.vetstar import stage_vetstar

    segs = eclipsing_binary_segments()

    def kep(kepid, **kw):
        assert str(kepid) == "5879574"
        return segs

    def qf(adql):
        if "TAP_SCHEMA.tables" in adql:
            return pd.DataFrame({"table_name": ['"J/AJ/151/68/table2"'], "description": [""]})
        if "TAP_SCHEMA.columns" in adql:
            return pd.DataFrame({"column_name": ['"KIC"', '"Period"']})
        return pd.DataFrame()

    def gq(adql):
        if "gaia_source" in adql:
            return pd.DataFrame({"source_id": [2050000000000000000], "ra": [290.0],
                                 "dec": [41.0], "pmra": [1.0], "pmdec": [-2.0],
                                 "ruwe": [1.05], "non_single_star": [0],
                                 "phot_g_mean_mag": [13.2]})
        return pd.DataFrame()

    rep = stage_vetstar(_stage_conf(), tmp_path, kepler_lc_fn=kep, query_fn=qf,
                        gaia_query_fn=gq, cone_fn=lambda *a: pd.DataFrame(),
                        position=(290.0, 41.0))
    assert rep["star_key"] == "kepler:5879574"
    assert rep["period"] == pytest.approx(PERIOD)
    assert rep["period_double"] == pytest.approx(2 * PERIOD)
    assert rep["status"] == "analysed"
    assert rep["verdict"].startswith("MUNDANE_EXPLANATION_FOUND")
    assert "UNEQUAL_MINIMA_AT_2P" in rep["verdict"]
    assert "DEGRADED" not in rep["verdict"]
    assert rep["gaia"]["match"]["source_id"] == "2050000000000000000"
    assert (tmp_path / "vetstar.json").exists()
    fold = pd.read_csv(tmp_path / "vetstar_fold.csv")
    assert set(fold["fold"]) == {"fold_at_period", "fold_at_period_flares_masked",
                                 "fold_at_2p", "fold_at_2p_flares_masked"}
    assert len(fold) == 400
    assert json.loads((tmp_path / "vetstar.json").read_text())["verdict"] == rep["verdict"]


def test_stage_says_no_lightcurve_rather_than_guessing(tmp_path):
    from seti.metronome.vetstar import stage_vetstar

    def dead(*a, **k):
        raise RuntimeError("MAST unreachable")

    def down(adql, *a, **k):
        raise RuntimeError("no route")

    rep = stage_vetstar(_stage_conf(), tmp_path, kepler_lc_fn=dead, query_fn=down,
                        gaia_query_fn=down, cone_fn=down, position=(290.0, 41.0))
    assert rep["status"] == "NO_LIGHTCURVE"
    assert rep["fetch_status"] != "OK"
    assert rep["verdict"].startswith("DEGRADED(")
    assert "NO_MUNDANE_EXPLANATION_FOUND" in rep["verdict"]
    assert "gaia:gaia_source" in rep["unreached"]
    assert any(u.startswith("vizier_cones:") for u in rep["unreached"])
    assert (tmp_path / "vetstar.json").exists()


def test_stage_reports_a_catalogued_binary_from_a_cone(tmp_path):
    from seti.metronome.vetstar import stage_vetstar

    def cone(table, ra, dec, radius):
        if table.startswith("B/vsx"):
            return pd.DataFrame({"Name": ["V* XY Lyr"], "Type": ["EW"],
                                 "Period": [2 * PERIOD]})
        return pd.DataFrame()

    rep = stage_vetstar(_stage_conf(), tmp_path, kepler_lc_fn=lambda *a, **k: [],
                        query_fn=lambda a: pd.DataFrame(),
                        gaia_query_fn=lambda a: pd.DataFrame(),
                        cone_fn=cone, position=(290.0, 41.0))
    hits = rep["catalogued_binary_hits"]
    assert [h["source"] for h in hits] == ["vsx"]
    assert "CATALOGUED_ECLIPSING_BINARY:vsx" in rep["verdict"]


# ---------------------------------------------------------------------------
# the spacecraft-roll contamination test
# ---------------------------------------------------------------------------
def _segments_from(amps):
    return [{"segment": i + 1, "amplitude_ptp": a} for i, a in enumerate(amps)]


def test_roll_season_catches_an_amplitude_locked_to_quarter_mod_four():
    """A neighbour's signal: the amplitude is a function of the aperture
    orientation, which repeats every four Kepler quarters."""
    rng = np.random.default_rng(31)
    base = {0: 6.6e-4, 1: 6.0e-4, 2: 1.41e-3, 3: 1.18e-3}
    amps = [base[(i + 1) % 4] * (1 + rng.normal(0, 0.12)) for i in range(17)]
    out = roll_season_test(_segments_from(amps), mission="kepler", n_perm=4000)
    assert out["status"] == STATUS_OK
    assert out["n_seasons"] == 4
    assert out["p_perm"] < 0.01
    assert out["ratio_max_min"] > 1.5
    assert out["f_stat"] > 1.0


def test_roll_season_passes_an_intrinsic_signal():
    """An intrinsic amplitude is diluted by crowding, which moves with the
    mask but carries no roll periodicity: the test must not fire."""
    rng = np.random.default_rng(32)
    amps = [1.0e-3 * (1 + rng.normal(0, 0.15)) for _ in range(17)]
    out = roll_season_test(_segments_from(amps), mission="kepler", n_perm=4000)
    assert out["status"] == STATUS_OK
    assert out["p_perm"] > 0.05


def test_roll_season_is_not_fooled_by_a_secular_trend():
    """Amplitude falling steadily across the mission is not a roll effect --
    a trend spreads itself evenly over the four seasons."""
    amps = [1.5e-3 - 5e-5 * i for i in range(17)]
    out = roll_season_test(_segments_from(amps), mission="kepler", n_perm=4000)
    assert out["status"] == STATUS_OK
    assert out["p_perm"] > 0.05


def test_roll_season_is_kepler_only_and_says_so():
    amps = [1e-3] * 17
    assert roll_season_test(_segments_from(amps), mission="tess")["status"] \
        == "NOT_APPLICABLE"
    assert roll_season_test(_segments_from([1e-3] * 3), mission="kepler")["status"] \
        == "TOO_FEW_SEGMENTS"
    assert roll_season_test([], mission="kepler")["status"] == "TOO_FEW_SEGMENTS"


def test_the_roll_rule_is_trippable_and_needs_both_p_and_ratio():
    rep = {"period": PERIOD, "catalogued_binary_hits": [],
           "roll_season": {"status": STATUS_OK, "p_perm": 0.0005, "f_stat": 3.07,
                           "ratio_max_min": 2.37}}
    verdict, _ = vet_verdict(rep)
    assert "AMPLITUDE_TRACKS_SPACECRAFT_ROLL" in verdict
    # a significant but tiny season difference is not a contamination claim
    rep["roll_season"] = {"status": STATUS_OK, "p_perm": 0.0005, "f_stat": 3.07,
                          "ratio_max_min": 1.05}
    assert "AMPLITUDE_TRACKS_SPACECRAFT_ROLL" not in vet_verdict(rep)[0]
    # nor is a large difference that a relabelling reproduces
    rep["roll_season"] = {"status": STATUS_OK, "p_perm": 0.4, "f_stat": 1.0,
                          "ratio_max_min": 3.0}
    assert "AMPLITUDE_TRACKS_SPACECRAFT_ROLL" not in vet_verdict(rep)[0]


# ---------------------------------------------------------------------------
# the neighbour census
# ---------------------------------------------------------------------------
def test_neighbour_context_asks_about_the_ones_that_could_be_the_source():
    gaia = {"neighbours": [
        {"source_id": "1", "sep_arcsec_at_epoch": 6.4, "phot_g_mean_mag": 18.0,
         "phot_variable_flag": "NOT_AVAILABLE", "ra": 292.9, "dec": 41.1},
        {"source_id": "2", "sep_arcsec_at_epoch": 13.3, "phot_g_mean_mag": 14.37,
         "phot_variable_flag": "VARIABLE", "ra": 292.86, "dec": 41.13},
        {"source_id": "3", "sep_arcsec_at_epoch": 40.0, "phot_g_mean_mag": 10.0,
         "phot_variable_flag": "VARIABLE", "ra": 292.0, "dec": 41.0},
    ]}
    asked = []

    def gq(adql):
        asked.append(adql)
        if "vari_summary" in adql:
            return pd.DataFrame({"source_id": [2], "num_selected_g_fov": [40]})
        return pd.DataFrame()

    def cone(table, ra, dec, radius):
        return pd.DataFrame()

    out = neighbour_context(gaia, g_target=14.79, query_fn=gq, cone_fn=cone,
                            max_sep_arcsec=20.0)
    # the faint non-variable one is not worth asking about; the 40-arcsec one
    # cannot be in a Kepler aperture
    assert [n["source_id"] for n in out] == ["2"]
    assert out[0]["why"] == "gaia_variable"
    assert out[0]["gaia_variability"]["gaiadr3.vari_summary"]["status"] == STATUS_OK
    assert out[0]["vizier_cones"]["vsx"]["status"] == STATUS_ABSENT
    assert any("vari_summary" in a for a in asked)


def test_neighbour_context_keeps_a_brighter_neighbour_even_when_not_flagged():
    gaia = {"neighbours": [
        {"source_id": "9", "sep_arcsec_at_epoch": 8.0, "phot_g_mean_mag": 12.0,
         "phot_variable_flag": "NOT_AVAILABLE", "ra": 1.0, "dec": 2.0}]}
    out = neighbour_context(gaia, g_target=14.79, query_fn=lambda a: pd.DataFrame(),
                            cone_fn=lambda *a: pd.DataFrame())
    assert [n["why"] for n in out] == ["brighter_than_target"]


def test_neighbour_context_is_empty_without_a_gaia_answer():
    assert neighbour_context({}, g_target=14.0) == []
    assert neighbour_context({"neighbours": []}, g_target=14.0) == []
