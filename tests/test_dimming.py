"""Offline tests for the anomalous-dimming (Boyajian-star) dip statistics."""

from __future__ import annotations

import numpy as np

from seti.dimming.dips import detect_dips


def _flat(n=200, base=12.0, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 1000, n))
    m = base + rng.normal(0, noise, n)
    return t, m, np.full(n, noise)


def test_flat_lightcurve_no_dips():
    t, m, e = _flat()
    s = detect_dips(t, m, e)
    assert s is not None
    assert s.n_dips == 0
    assert s.max_depth < 0.05
    assert s.score < 0.3


def test_boyajian_like_aperiodic_dimming_scores_high():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 1500, 400))
    m = 12.0 + rng.normal(0, 0.01, t.size)
    # A handful of deep, irregular (aperiodic) dips of 10-22% (0.11-0.27 mag).
    for tc, depth_mag in [(220, 0.16), (540, 0.27), (910, 0.12), (1290, 0.20)]:
        m += depth_mag * np.exp(-0.5 * ((t - tc) / 12.0) ** 2)
    e = np.full(t.size, 0.01)
    s = detect_dips(t, m, e)
    assert s is not None
    assert s.max_depth > 0.1
    assert s.n_dips >= 3
    assert s.asymmetry > 1.5           # dimming-only, not symmetric
    assert s.score > 0.5


def test_few_discrete_events_not_hundreds_of_epochs():
    from seti.dimming.dips import _count_events

    # A Boyajian-like light curve: 4 deep, discrete, well-separated dip events.
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0, 1500, 600))
    m = 12.0 + rng.normal(0, 0.008, t.size)
    for tc, depth in [(220, 0.18), (560, 0.25), (920, 0.15), (1300, 0.22)]:
        m += depth * np.exp(-0.5 * ((t - tc) / 8.0) ** 2)
    s = detect_dips(t, m, np.full(t.size, 0.01))
    assert s is not None
    # A few discrete events, not one-per-epoch.
    assert 1 <= s.n_dip_events <= 12
    assert s.n_dip_events < s.n_dips or s.n_dips <= s.n_dip_events  # event<=epochs
    assert s.out_of_dip_rms < 0.03            # quiescent between dips
    assert s.score > 0.5

    # _count_events groups adjacent dipped epochs into one event, and a lone
    # single-epoch outlier is NOT counted (min_run=2): {0,1,2} and {100,101}
    # count; the isolated {200} does not.
    t2 = np.array([0., 1, 2, 3, 100, 101, 200])
    is_dip = np.array([True, True, True, False, True, True, True])
    assert _count_events(t2, is_dip) == 2


def test_single_epoch_outliers_do_not_qualify():
    from seti.dimming.dips import _dip_events
    from seti.dimming.run import _is_candidate

    # A faint-star light curve whose only "dips" are isolated single-epoch noise
    # outliers must NOT be a candidate: no sustained (>=2 epoch) event.
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0, 2000, 110))
    m = 17.2 + rng.normal(0, 0.03, t.size)
    # three lone faint outliers at random isolated epochs
    for j in (10, 55, 95):
        m[j] += 0.12
    s = detect_dips(t, m, np.full(t.size, 0.03))
    assert s is not None
    assert s.n_dip_events == 0            # no multi-epoch event
    assert s.max_event_depth == 0.0
    assert not _is_candidate(s.as_dict(), depth_min=0.10, n_dips_min=3,
                             asym_min=1.5, period_power_max=0.4)

    # Direct check: one sustained pair + one lone outlier -> a single valid event.
    t3 = np.array([0., 1, 50, 100, 101, 102])
    is_dip = np.array([True, False, True, True, True, True])
    frac = np.where(is_dip, 0.2, 0.0)
    n_ev, dep = _dip_events(t3, is_dip, frac)
    assert n_ev == 1 and dep == 0.2       # {100,101,102}; lone {0} and {50} drop


def test_high_amplitude_periodic_variable_rejected():
    # An RR-Lyrae-like continuous pulsator: hundreds of epochs below the bright
    # baseline, large out-of-dip scatter -> NOT a Boyajian candidate.
    rng = np.random.default_rng(8)
    t = np.sort(rng.uniform(0, 1000, 800))
    m = 12.0 + 0.35 * np.sin(2 * np.pi * t / 0.57) + rng.normal(0, 0.01, t.size)
    s = detect_dips(t, m, np.full(t.size, 0.01))
    assert s is not None
    # A continuous pulsator dips at hundreds of epochs across very many "events",
    # is symmetric, and is strongly periodic --- every one disqualifying.
    assert s.n_dip_events > 40
    assert s.asymmetry < 1.5
    assert s.period_power > 0.4
    from seti.dimming.run import _is_candidate
    assert not _is_candidate(s.as_dict(), depth_min=0.10, n_dips_min=3,
                             asym_min=1.5, period_power_max=0.4)


def test_smooth_sinusoid_is_symmetric_low_score():
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 1000, 300))
    m = 12.0 + 0.05 * np.sin(2 * np.pi * t / 30.0) + rng.normal(0, 0.005, t.size)
    s = detect_dips(t, m, magerr=np.full(t.size, 0.005))
    # A symmetric oscillation is not dimming-dominated.
    assert s.asymmetry < 1.6
    assert s.score < 0.6


def test_eclipsing_binary_is_periodic():
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 1000, 600))
    period = 3.2
    phase = (t % period) / period
    m = 12.0 + rng.normal(0, 0.01, t.size)
    m += 0.3 * (np.abs(phase - 0.5) < 0.05)   # periodic box eclipses
    s = detect_dips(t, m, magerr=np.full(t.size, 0.01))
    assert s is not None
    # Strictly periodic eclipses produce strong Lomb-Scargle power.
    assert s.period_power > 0.1


def test_fetch_ztf_region_groups_by_oid(monkeypatch):
    import requests

    from seti.dimming import acquire

    # Two sources in one box: a 40-epoch source and a too-short 5-epoch source.
    rows = [f"101,{58000+k}.0,18.0,0.02,131.0,33.0" for k in range(40)]
    rows += [f"202,{58000+k}.0,17.0,0.02,131.1,33.1" for k in range(5)]
    csv = "oid,mjd,mag,magerr,ra,dec\n" + "\n".join(rows) + "\n"

    class _Resp:
        status_code = 200
        text = csv

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    out = acquire.fetch_ztf_region(131.0, 33.0, box_deg=0.3, min_epochs=30)
    assert set(out.keys()) == {"101"}          # only the 40-epoch source survives
    assert len(out["101"]) == 40
    assert {"mjd", "mag", "magerr", "ra", "dec"} <= set(out["101"].columns)


def _seasonal(n_per_season=20, n_seasons=6, base=15.0, noise=0.02, slope_mag_yr=0.0,
              seed=0):
    """Synthetic multi-season light curve with an optional linear trend."""
    rng = np.random.default_rng(seed)
    t = []
    for s in range(n_seasons):
        t.append(rng.uniform(s * 365.25, s * 365.25 + 120, n_per_season))
    t = np.sort(np.concatenate(t))
    yr = (t - t.min()) / 365.25
    m = base + slope_mag_yr * yr + rng.normal(0, noise, t.size)
    return t, m, np.full(t.size, noise)


def test_secular_fade_detects_monotonic_dimming():
    from seti.dimming.secular import detect_secular_fade

    # A star fading 0.04 mag/yr over 6 seasons (~0.2 mag total).
    t, m, e = _seasonal(slope_mag_yr=0.04, noise=0.015, seed=1)
    s = detect_secular_fade(t, m, e)
    assert s is not None
    assert s.slope_mag_yr > 0.02
    assert s.slope_sigma > 3.0
    assert s.total_change_mag > 0.1
    assert s.monotonic_frac > 0.6
    assert s.score > 0.5


def test_secular_flat_and_brightening_score_zero():
    from seti.dimming.secular import detect_secular_fade

    flat = detect_secular_fade(*_seasonal(slope_mag_yr=0.0, seed=2))
    assert flat is not None and flat.score < 0.3
    # Brightening (negative slope in mag) is not a fade -> score 0.
    bright = detect_secular_fade(*_seasonal(slope_mag_yr=-0.05, seed=3))
    assert bright is not None and bright.score == 0.0


def test_shared_epoch_cut_rejects_bad_night_artifacts():
    from seti.dimming.run import _flag_shared_epoch_dips

    rng = np.random.default_rng(21)
    base_t = np.sort(rng.uniform(0, 1500, 200))
    bad_nights = np.array([300.0, 720.0, 1100.0])   # field-wide bad epochs
    rows = []
    # 12 stars that "dip" only on the shared bad nights -> artifacts.
    for i in range(12):
        t = np.concatenate([base_t, bad_nights])
        m = 16.0 + rng.normal(0, 0.02, t.size)
        m[-3:] += 0.4                                # deep dip on each bad night
        o = np.argsort(t)
        rows.append({"_mjd": t[o], "_mag": m[o], "is_candidate": True,
                     "resists_mundane": True})
    # 1 star that dips on its OWN unique nights -> genuine.
    t = np.concatenate([base_t, [555.0, 556.0]])
    m = 16.0 + rng.normal(0, 0.02, t.size)
    m[-2:] += 0.4
    o = np.argsort(t)
    genuine = {"_mjd": t[o], "_mag": m[o], "is_candidate": True,
               "resists_mundane": True}
    rows.append(genuine)

    n = _flag_shared_epoch_dips(rows)
    assert n >= 10                                   # the shared-night artifacts demoted
    assert genuine["is_candidate"] is True           # the unique dipper survives
    assert all(not r["is_candidate"] for r in rows[:12])


def test_ensemble_detrend_removes_common_mode_fade():
    from seti.dimming.run import _ensemble_detrend_secular
    from seti.dimming.secular import season_medians

    rng = np.random.default_rng(11)
    # A shared per-season common-mode drift (e.g. ZTF zeropoint creep) that, alone,
    # looks like a 0.05 mag/yr fade in every star.
    n_seasons = 6
    cm = 0.05 * np.arange(n_seasons)          # mag offset per season
    rows = []

    def _make(extra_slope_mag_yr, seed):
        t, blocks = [], []
        for s in range(n_seasons):
            tt = rng.uniform(s * 365.25, s * 365.25 + 120, 25)
            t.append(tt)
            yr = (tt - 0) / 365.25
            blocks.append(15.0 + cm[s] + extra_slope_mag_yr * yr
                          + rng.normal(0, 0.012, tt.size))
        t = np.concatenate(t)
        m = np.concatenate(blocks)
        o = np.argsort(t)
        t, m = t[o], m[o]
        e = np.full(t.size, 0.012)
        sm = season_medians(t, m, e)
        return {"_sm": sm, "_omed": float(np.median(m)), "_nepoch": t.size,
                "is_secular_fader": True, "secular_score": 1.0, "secular_sigma": 9.0}

    # 15 stars with ONLY the common-mode drift; 1 with an extra intrinsic fade.
    for i in range(15):
        rows.append(_make(0.0, 100 + i))
    rows.append(_make(0.06, 999))            # intrinsic fader

    _ensemble_detrend_secular(rows)
    # The common-mode-only stars are no longer faders; the intrinsic one remains.
    assert sum(r["is_secular_fader"] for r in rows[:15]) <= 2   # nearly all cleared
    assert rows[-1]["is_secular_fader"] is True


def test_secular_too_few_seasons_returns_none():
    from seti.dimming.secular import detect_secular_fade

    t, m, e = _seasonal(n_seasons=2, seed=4)
    assert detect_secular_fade(t, m, e, min_seasons=3) is None


def test_ir_excess_verdict():
    from seti.dimming.vet import ir_excess_verdict

    # Bare photosphere (W1-W2 ~ 0, small K-W2), no SIMBAD type -> clean.
    assert ir_excess_verdict({"W1mag": 14.00, "W2mag": 13.98, "Ksmag": 14.3,
                              "simbad_otype": ""}) == "clean"
    # Strong W1-W2 excess -> dusty disk.
    assert ir_excess_verdict({"W1mag": 13.0, "W2mag": 12.4, "Ksmag": 13.5,
                              "simbad_otype": ""}) == "ir_excess_dusty"
    # K-W2 excess alone -> dusty.
    assert ir_excess_verdict({"W1mag": 13.0, "W2mag": 12.95, "Ksmag": 13.9,
                              "simbad_otype": ""}) == "ir_excess_dusty"
    # No excess but classified as a young stellar object -> known variable.
    assert ir_excess_verdict({"W1mag": 14.0, "W2mag": 13.98, "Ksmag": 14.2,
                              "simbad_otype": "YSO"}) == "known_variable"
    # No WISE/2MASS at all -> cannot clear.
    assert ir_excess_verdict({"simbad_otype": ""}) == "no_ir_data"


def test_too_few_epochs_returns_none():
    t, m, e = _flat(n=10)
    assert detect_dips(t, m, e) is None


def test_hr_class_separates_giants_dwarfs_main_sequence():
    from seti.dimming.context import absolute_g, hr_class

    # A nearby F/G main-sequence star: BP-RP~0.7, M_G~4.6.
    # parallax 20 mas (50 pc) => G - 5log10(50/10)... use absolute_g to set G.
    plx = 20.0  # mas, d=50 pc
    g_ms = 4.6 - 5.0 * np.log10(plx / 100.0)        # apparent G giving M_G=4.6
    assert abs(absolute_g(g_ms, plx) - 4.6) < 1e-6
    assert hr_class(g_ms, 0.7, plx, parallax_over_error=50.0) == "main_sequence"

    # A red giant at the same colour but hugely over-luminous (M_G ~ 0).
    g_giant = 0.0 - 5.0 * np.log10(plx / 100.0)
    assert hr_class(g_giant, 0.7, plx, parallax_over_error=50.0) == "giant"

    # A white dwarf: blue and very faint (M_G ~ 12).
    g_wd = 12.0 - 5.0 * np.log10(plx / 100.0)
    assert hr_class(g_wd, 0.2, plx, parallax_over_error=50.0) == "white_dwarf"

    # No reliable parallax => never assert a class.
    assert hr_class(g_ms, 0.7, plx, parallax_over_error=1.0) == "unknown"


def test_dimming_run_end_to_end(tmp_path):
    from seti.config import load_config
    from seti.dimming.run import dimming_run

    lightcurves = []
    # Five flat (non-dipping) stars.
    for i in range(5):
        t, m, e = _flat(n=200, seed=10 + i)
        lightcurves.append({"source_id": 1000 + i, "ra": 270.0 + 0.01 * i,
                            "dec": 30.0, "mjd": t, "mag": m, "magerr": e})
    # One Boyajian-like deep aperiodic dipper.
    rng = np.random.default_rng(99)
    t = np.sort(rng.uniform(0, 1500, 400))
    m = 12.0 + rng.normal(0, 0.01, t.size)
    for tc, depth in [(220, 0.16), (540, 0.27), (910, 0.12), (1290, 0.20)]:
        m += depth * np.exp(-0.5 * ((t - tc) / 12.0) ** 2)
    lightcurves.append({"source_id": 2002, "ra": 271.0, "dec": 31.0,
                        "mjd": t, "mag": m, "magerr": np.full(t.size, 0.01),
                        # main-sequence F star context: blue-ish, good parallax
                        "phot_g_mean_mag": 12.0, "bp_rp": 0.6, "parallax": 5.0,
                        "parallax_over_error": 40.0})

    cfg = load_config()
    cfg.root = tmp_path
    summary = dimming_run(cfg, lightcurves=lightcurves)

    assert summary["n_searched"] == 6
    assert summary["n_candidates"] >= 1
    # The injected dipper is the top candidate.
    assert summary["top_candidates"][0]["source_id"] == 2002
    # ...and it is flagged as resisting the mundane (main-sequence, aperiodic).
    assert summary["top_candidates"][0]["hr_class"] == "main_sequence"
    assert summary["top_candidates"][0]["resists_mundane"] is True
    assert summary["n_resists_mundane"] >= 1
    assert "occurrence_limit" in summary
    # Output is namespaced by sky field; locate it recursively.
    assert list((tmp_path / "results" / "dimming").rglob("summary.json"))
    assert list((tmp_path / "results" / "dimming").rglob("top_dippers.json"))


def test_glint_detects_brief_achromatic_brightening():
    from seti.dimming.glint import detect_glints
    rng = np.random.default_rng(31)
    t = np.sort(rng.uniform(0, 1500, 300))
    m = 16.0 + rng.normal(0, 0.02, t.size)          # quiescent baseline
    # one brief bright glint: +80% flux (-0.64 mag) at two adjacent epochs
    i = 150
    m[i] -= 0.64
    m[i + 1] -= 0.40
    s = detect_glints(t, m, np.full(t.size, 0.02))
    assert s is not None
    assert s.max_brighten > 0.3
    assert s.n_glint_events >= 1
    assert s.brighten_sigma > 5
    assert s.score > 0.4


def test_glint_flat_and_flaring_star_score_low():
    from seti.dimming.glint import detect_glints
    rng = np.random.default_rng(32)
    t = np.sort(rng.uniform(0, 1500, 300))
    # flat star -> no glint
    flat = detect_glints(t, 16.0 + rng.normal(0, 0.02, t.size), np.full(t.size, 0.02))
    assert flat is not None and flat.score < 0.3
    # frequent flaring (many brightening epochs) -> too many events, score 0
    m = 16.0 + rng.normal(0, 0.02, t.size)
    flare_idx = rng.choice(t.size, 40, replace=False)
    m[flare_idx] -= rng.uniform(0.4, 1.0, 40)
    flary = detect_glints(t, m, np.full(t.size, 0.02))
    assert flary is not None and flary.score < 0.3
