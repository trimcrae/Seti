"""S42 sub-exposure flash: offline tests of both stages on synthetic dq images and ramps.

Nothing here touches a Roman product.  The dq images and ramps are built by the
module's own synthesis helpers from the PSF model the detectors use (integrated
over pixels for the synthesis, sampled at pixel centres for the detection, so
the two are not the same function).  The suite covers the brief's four
requirements: recover an injected signal, reject the dominant confounders
(cosmic rays, snowballs, saturation, hot pixels, flares), degrade honestly on
missing flags, and trip every rejection rule at least once.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from seti.roman import flash as F
from seti.roman.schema import DQ_FLAGS_FALLBACK, DQCutout, Ramp, json_safe, load_roman_config

FLAGS = dict(DQ_FLAGS_FALLBACK)
JUMP = FLAGS["JUMP_DET"]
SAT = FLAGS["SATURATED"]
TIMES6 = np.linspace(3.0, 45.0, 6)


@pytest.fixture(scope="module")
def conf():
    return load_roman_config()


def _stars(rng, n=10, fwhm_jitter=0.4):
    return [{"star_id": f"S{i}", "x": 20.0 + 45.0 * (i % 5) + rng.uniform(-fwhm_jitter, fwhm_jitter),
             "y": 20.0 + 45.0 * (i // 5) + rng.uniform(-fwhm_jitter, fwhm_jitter)} for i in range(n)]


# --------------------------------------------------------------------------------------
# PSF model and geometry
# --------------------------------------------------------------------------------------

def test_gaussian_psf_is_normalised_and_centred():
    g = F.gaussian_psf(1.2, 7)
    assert g.shape == (7, 7)
    assert abs(g.sum() - 1.0) < 1e-12
    assert np.unravel_index(np.argmax(g), g.shape) == (3, 3)
    assert F.gaussian_psf(1.2, 6).shape == (7, 7)          # even sizes widened
    off = F.gaussian_psf(1.2, 7, center=(2.0, 4.0))
    assert np.unravel_index(np.argmax(off), off.shape) == (2, 4)


def test_psf_footprint_is_odd_at_least_five_and_contains_the_centre():
    for fwhm in (0.6, 1.1, 1.5, 3.0):
        fp = F.psf_footprint(fwhm, 0.10)
        assert fp.shape[0] == fp.shape[1] and fp.shape[0] % 2 == 1 and fp.shape[0] >= 5
        c = fp.shape[0] // 2
        assert fp[c, c]
    assert F.psf_footprint(3.0, 0.10).sum() > F.psf_footprint(1.1, 0.10).sum()


def test_footprint_pixels_clip_to_the_image_and_follow_the_star():
    fp = F.footprint_pixels((1.0, 0.0), 1.5, 0.10, shape=(5, 5))
    assert fp.shape[0] >= 3
    assert (fp >= 0).all() and (fp[:, 0] < 5).all() and (fp[:, 1] < 5).all()
    assert [0, 1] in fp.tolist()
    fp2 = F.footprint_pixels((10.0, 10.0), 1.5, 0.10)
    assert [10, 10] in fp2.tolist() and [10, 11] in fp2.tolist() and [9, 10] in fp2.tolist()


def test_elongation_of_lines_and_blobs():
    line = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]])
    plus = np.array([[1, 0], [0, 1], [1, 1], [2, 1], [1, 2]])
    assert F._elongation(line) > 4.0
    assert abs(F._elongation(plus) - 1.0) < 1e-9
    assert F._elongation(np.array([[3, 3]])) == 1.0


# --------------------------------------------------------------------------------------
# Stage 1: cluster finding and classification
# --------------------------------------------------------------------------------------

def test_cluster_finding_on_a_synthetic_dq_image(conf):
    rng = np.random.default_rng(3)
    fwhm = 1.2
    stars = _stars(rng)
    flash = ["S1", "S4", "S7", "S2"]
    dq = F.synthesise_dq_image((120, 240), stars, fwhm, flash, 200, 2, rng, FLAGS, saturated_stars=["S2"])
    assert dq.dtype == np.uint32
    cut = DQCutout(image_id="syn-1", dq=dq, stars=stars, detector="WFI07", band="F146", mjd=60100.5,
                   x0=1000, y0=2000, psf_fwhm_px=fwhm)
    out = F.screen_dq_cutout(cut, conf, FLAGS)
    assert out["status"] == "ok"
    labels = out["labels"]
    assert labels["psf_on_star"] == 3
    assert sorted(c["star_id"] for c in out["candidates"]) == ["S1", "S4", "S7"]
    assert labels["near_saturated"] == 1
    assert labels["snowball"] == 2
    assert labels.get("cosmic_ray_track", 0) >= 10          # the straight 4-8 px tracks
    assert labels.get("too_small", 0) >= 50                  # the 1-2 px hits
    assert set(labels) <= set(F.LABELS)
    assert sum(labels.values()) == out["n_clusters"]
    # candidate records are in detector coordinates and carry their pixels
    for c in out["candidates"]:
        star = next(s for s in stars if s["star_id"] == c["star_id"])
        assert abs(c["centroid_x"] - (star["x"] + 1000)) < 0.75 * fwhm
        assert abs(c["centroid_y"] - (star["y"] + 2000)) < 0.75 * fwhm
        assert c["coverage"] >= 0.6 and c["n_px"] >= 3 and c["elongation"] <= 1.6
        assert all(y >= 2000 and x >= 1000 for y, x in c["pixels"])
        assert c["tier"] == conf["flash"]["tiers"]["interest"]
        assert c["mjd"] == 60100.5 and c["band"] == "F146"
    f = out["funnel"]
    assert f["counts"]["psf_on_star"] == 3 and f["rejections"]["snowball"] == 2
    json_safe(out)


def test_every_size_and_shape_rule_trips(conf):
    star = {"star_id": "A", "x": 10.0, "y": 10.0}
    # a straight 5-px line off any star: a track
    dq = np.zeros((30, 30), np.uint32)
    dq[20, 3:8] |= JUMP
    # a 3-px L off any star: compact but on nothing
    dq[3, 20] |= JUMP
    dq[3, 21] |= JUMP
    dq[4, 20] |= JUMP
    # a 2-px hit: too small
    dq[25, 25] |= JUMP
    dq[25, 26] |= JUMP
    # a 7x7 block without a saturated core: too large
    dq[22:29, 10:17] |= JUMP
    cut = DQCutout(image_id="rules", dq=dq, stars=[star], psf_fwhm_px=1.2)
    out = F.screen_dq_cutout(cut, conf, FLAGS)
    assert out["labels"] == {"cosmic_ray_track": 1, "psf_off_star": 1, "too_small": 1, "too_large": 1}
    assert out["candidates"] == []


def test_footprint_incomplete_and_near_saturated(conf):
    fwhm = 2.0                                     # 9-px footprint
    star = {"star_id": "A", "x": 10.0, "y": 10.0}
    dq = np.zeros((21, 21), np.uint32)
    dq[10:12, 10:12] |= JUMP                       # a 2x2 block inside the footprint: 4 of 9
    cut = DQCutout(image_id="cov", dq=dq, stars=[star], psf_fwhm_px=fwhm)
    out = F.screen_dq_cutout(cut, conf, FLAGS)
    assert out["labels"] == {"footprint_incomplete": 1}
    # the full footprint is on-star ...
    fp = F.footprint_pixels((10.0, 10.0), fwhm, 0.10)
    dq2 = np.zeros((21, 21), np.uint32)
    dq2[fp[:, 0], fp[:, 1]] |= JUMP
    out2 = F.screen_dq_cutout(DQCutout(image_id="full", dq=dq2, stars=[star], psf_fwhm_px=fwhm), conf, FLAGS)
    assert out2["labels"] == {"psf_on_star": 1}
    # ... unless a SATURATED pixel sits within 3 px of the star
    dq3 = dq2.copy()
    dq3[12, 12] |= SAT
    out3 = F.screen_dq_cutout(DQCutout(image_id="sat", dq=dq3, stars=[star], psf_fwhm_px=fwhm), conf, FLAGS)
    assert out3["labels"] == {"near_saturated": 1}
    # ... or the catalogue says the star saturates
    out4 = F.screen_dq_cutout(DQCutout(image_id="cat", dq=dq2, stars=[dict(star, saturated=True)],
                                       psf_fwhm_px=fwhm), conf, FLAGS)
    assert out4["labels"] == {"near_saturated": 1}


def test_match_and_coverage_helpers():
    stars = [{"star_id": "A", "x": 5.0, "y": 5.0}, {"star_id": "B", "x": 20.0, "y": 20.0}]
    cl = {"centroid": (5.3, 5.2), "pixels": np.array([[5, 5], [5, 6], [4, 5], [6, 5], [5, 4]]), "n": 5}
    m = F.match_cluster_to_stars(cl, stars, 1.2, 0.75)
    assert m is not None and m["star"]["star_id"] == "A" and m["distance_px"] < 0.4
    assert F.match_cluster_to_stars({"centroid": (12.0, 12.0)}, stars, 1.2, 0.75) is None
    assert F.match_cluster_to_stars(cl, [], 1.2, 0.75) is None
    cov = F.footprint_coverage(cl["pixels"], (5.0, 5.0), 1.5, 0.10)
    assert cov == 1.0
    assert F.footprint_coverage(cl["pixels"][:2], (5.0, 5.0), 1.5, 0.10) == pytest.approx(0.4)


# --------------------------------------------------------------------------------------
# Hot-pixel ledger
# --------------------------------------------------------------------------------------

def _repeat_cutouts(n, fwhm=1.5):
    star = {"star_id": "H", "x": 10.0, "y": 10.0}
    fp = F.footprint_pixels((10.0, 10.0), fwhm, 0.10)
    cuts = []
    for i in range(n):
        dq = np.zeros((21, 21), np.uint32)
        dq[fp[:, 0], fp[:, 1]] |= JUMP
        cuts.append(DQCutout(image_id=f"img-{i}", dq=dq, stars=[star], detector="WFI03", mjd=60000.0 + i,
                             x0=500, y0=700, psf_fwhm_px=fwhm))
    return cuts


def test_hot_ledger_labels_the_third_exposure_and_round_trips(conf):
    ledger = F.HotPixelLedger()
    labels = [F.screen_dq_cutout(c, conf, FLAGS, ledger)["labels"] for c in _repeat_cutouts(3)]
    assert labels[0] == {"psf_on_star": 1}
    assert labels[1] == {"psf_on_star": 1}
    assert labels[2] == {"hot_pixel": 1}
    assert ledger.count("WFI03", 710, 510) == 3 and ledger.hot("WFI03", 710, 510, 3)
    assert not ledger.hot("WFI03", 710, 510, 4) and ledger.count("WFI03", 0, 0) == 0
    assert ledger.n_images("WFI03") == 3
    assert (710, 510, 3) in ledger.hot_pixels("WFI03", 3)
    # idempotent per image id
    assert ledger.add("img-0", "WFI03", [[710, 510]]) == 0
    back = F.HotPixelLedger.from_dict(json_safe(ledger.as_dict()))
    assert back.count("WFI03", 710, 510) == 3 and back.n_images("WFI03") == 3
    assert back.prune(min_count=4) > 0 and back.count("WFI03", 710, 510) == 0


def test_hot_pixels_are_vetoed_retroactively_in_assess(conf):
    recs = []
    for c in _repeat_cutouts(3):
        recs.extend(F.screen_dq_cutout(c, conf, FLAGS)["candidates"])     # no shared ledger
    assert len(recs) == 3
    out = F.assess_candidates(recs, conf)
    assert all(r["veto"] == "hot_pixel" for r in out["records"])
    assert out["n_vetoed"] == 3 and out["n_candidates"] == 0 and out["verdict"] == "NO_SURVIVORS"
    assert out["funnel"]["rejections"]["hot_pixel"] == 3
    assert len(out["hot_pixels"]) == 5
    # two exposures alone are not a defect, but are a recurrence on the star
    out2 = F.assess_candidates(recs[:2], conf)
    assert all(r["veto"] is None for r in out2["records"])
    assert out2["recurrent_stars"] == {"H": ["img-0", "img-1"]}
    assert all(r["recurrent_star"] and r["n_star_events"] == 2 for r in out2["records"])
    assert out2["verdict"] == conf["flash"]["tiers"]["interest"]


def test_assess_tiers(conf):
    tiers = conf["flash"]["tiers"]
    def rec(image_id, star_id="Z", **kw):
        # distinct pixels per record (the same pixels on 3+ records is the hot-pixel veto)
        j = 10 * len(image_id) + ord(image_id[-1])
        return dict({"star_id": star_id, "detector": "WFI01", "image_id": image_id,
                     "pixels": [[j, 1], [j, 2], [j + 1, 1]]}, **kw)

    recs = [
        rec("a", ramp={"verdict": "FLASH_CONSISTENT"}, next_exposure_residual_sigma=0.8),
        rec("b", ramp={"verdict": "FLASH_CONSISTENT"}),
        rec("c", "Y", ramp={"verdict": "SLOPE_CHANGE_FLARE_LIKE"}, next_exposure_residual_sigma=0.1),
        rec("d", "X", ramp={"verdict": "FLASH_CONSISTENT"}, next_exposure_residual_sigma=6.0),
        rec("e", "W"),
        rec("f", "V", ramp={"verdict": "INSUFFICIENT_RESULTANTS"}, next_exposure_residual_sigma=0.0),
    ]
    out = F.assess_candidates(recs, conf)
    by = {r["image_id"]: r for r in out["records"]}
    assert by["a"]["tier"] == tiers["candidate"] and by["a"]["veto"] is None
    assert by["b"]["tier"] == tiers["interest"] and by["b"]["pending"] == ["next_exposure"]
    assert by["c"]["veto"] == "ramp_slope_change_flare_like" and by["c"]["tier"] == "REJECTED"
    assert by["d"]["veto"] == "next_exposure_brightening"
    assert by["e"]["tier"] == tiers["interest"] and by["e"]["pending"] == ["ramp", "next_exposure"]
    assert by["f"]["tier"] == tiers["interest"] and by["f"]["pending"] == ["ramp_insufficient_resultants"]
    assert out["verdict"] == tiers["candidate"] and out["n_candidates"] == 1
    assert by["a"]["recurrent_star"] and by["a"]["n_star_events"] == 2     # a and b, both unvetoed
    json_safe(out)


# --------------------------------------------------------------------------------------
# Stage 2: ramp fits
# --------------------------------------------------------------------------------------

def test_fit_step_ramp_recovers_an_exact_step():
    t = TIMES6
    y = 2.0 + 3.0 * t + np.where(np.arange(6) >= 3, 40.0, 0.0)
    f = F.fit_step_ramp(y, t, 3, sigma=1.0)
    assert f["status"] == "ok" and abs(f["step"] - 40.0) < 1e-9
    assert abs(f["s_before"] - 3.0) < 1e-9 and abs(f["s_after"] - 3.0) < 1e-9
    assert not f["slope_after_constrained"] and abs(f["slope_change_sigma"]) < 1e-6
    assert f["sigma_source"] == "pooled" and 1.5 < f["step_err"] < 2.0   # ~1.7 sigma on 6 resultants
    best = F.best_step_index(y, t, 1.0)
    assert best["k"] == 3 and len(best["scan"]) == 5
    # the ends of the ramp can only be step tests: slope after is fixed to slope before
    f1 = F.fit_step_ramp(y, t, 1, sigma=1.0)
    f5 = F.fit_step_ramp(y, t, 5, sigma=1.0)
    assert f1["slope_after_constrained"] and f5["slope_after_constrained"]
    assert f1["slope_change_sigma"] == 0.0
    with pytest.raises(ValueError):
        F.fit_step_ramp(y, t, 0)
    with pytest.raises(ValueError):
        F.fit_step_ramp(y, t, 6)
    short = F.fit_step_ramp(y[:3], t[:3], 1, sigma=1.0)
    assert short["status"] == "insufficient_resultants" and math.isnan(short["step"])
    assert F.best_step_index(y[:3], t[:3], 1.0)["status"] == "insufficient_resultants"


def test_fit_step_ramp_sigma_rules():
    t = TIMES6
    rng = np.random.default_rng(0)
    y = 1.0 + 2.0 * t + rng.normal(0, 1.0, 6)
    assert F.fit_step_ramp(y, t, 3, sigma=1.0, single_slope=True)["sigma_source"] == "pooled"
    assert F.fit_step_ramp(y, t, 3, sigma=None, single_slope=True)["sigma_source"] == "residual"
    assert F.fit_step_ramp(y, t, 3, sigma=None)["sigma_source"] == "residual_low_dof"
    # a pooled sigma that the scatter contradicts is replaced
    noisy = 1.0 + 2.0 * t + rng.normal(0, 30.0, 6)
    f = F.fit_step_ramp(noisy, t, 3, sigma=1.0, single_slope=True)
    assert f["sigma_source"] == "residual_inflated" and f["sigma_used"] > 3.0
    # four resultants: never the residual
    assert F.fit_step_ramp(y[:4], t[:4], 2, sigma=1.0)["sigma_source"] == "pooled"


def test_fit_broken_ramp_follows_a_flare_not_a_step():
    t = TIMES6
    flare = 5.0 * t + np.where(np.arange(6) >= 3, 10.0 * (t - t[3]), 0.0)
    step = 5.0 * t + np.where(np.arange(6) >= 3, 100.0, 0.0)
    assert F.fit_broken_ramp(flare, t, 3)["rss"] < 1e-9
    assert F.fit_step_ramp(flare, t, 3, sigma=1.0, single_slope=True)["rss"] > 100.0
    assert F.fit_step_ramp(step, t, 3, sigma=1.0, single_slope=True)["rss"] < 1e-9
    assert min(F.fit_broken_ramp(step, t, k)["rss"] for k in range(1, 6)) > 100.0


def _ramp(rng, **kw):
    args = dict(box=(9, 9), times_s=TIMES6, star_xy=(4.0, 4.0), fwhm_px=1.5, rate_e_s=2.0,
                read_noise_e=12.0, frames_per_resultant=16, rng=rng)
    args.update(kw)
    return F.synthesise_ramp(**args)


def test_ramp_flash_is_recovered(conf):
    """A 500 e- pulse at resultant 3 on a faint star with quiet reads.

    The per-pixel step error on a 6-resultant ramp is ~1.7 sigma (three points
    a side barely pin the slope), so the ~60 e- wing pixels of a 500 e- pulse
    need a per-resultant read noise of ~3 e- to clear 5 sigma; 16 frames per
    resultant at 12 e- per frame gives that.  Recovery at these settings is
    99/100 seeds; the seed is fixed.
    """
    rng = np.random.default_rng(11)
    ramp = _ramp(rng, flash_e=500.0, flash_at_index=3)
    assert ramp.unit == "e-" and ramp.n_resultants == 6
    r = F.ramp_flash_test(ramp, (4.0, 4.0), 1.5, conf, 12.0)
    assert r["verdict"] == "FLASH_CONSISTENT"
    assert r["coincidence_frac"] >= 0.8
    assert r["psf_amplitude_corr"] >= 0.8
    assert r["modal_index"] == 3
    assert r["n_footprint_px"] == 5
    assert 250.0 < r["total_step_e"] < 600.0             # ~79 % of 500 e- lands in the footprint
    assert r["slope_change_sigma_max_over_pixels"] <= 3.0
    assert r["delta_chi2_step_minus_flare"] < 0.0        # the step model wins
    assert r["frames_source"] == "read_pattern" and r["gain_e_per_dn"] == 1.0
    assert len(r["per_pixel"]) == 5 and all(p["k"] == 3 for p in r["per_pixel"])
    json_safe(r)


def test_ramp_flash_gbtds_like_bright_pulse(conf):
    """A 5000 e- pulse on a 100 e-/s star with two frames per resultant: the harsher regime."""
    rng = np.random.default_rng(5)
    ramp = _ramp(rng, rate_e_s=100.0, frames_per_resultant=2, flash_e=5000.0, flash_at_index=2)
    r = F.ramp_flash_test(ramp, (4.0, 4.0), 1.5, conf, 12.0)
    assert r["verdict"] == "FLASH_CONSISTENT" and r["modal_index"] == 2
    assert r["coincidence_frac"] >= 0.8 and r["psf_amplitude_corr"] >= 0.8


def test_ramp_flare_is_slope_change_not_step(conf):
    rng = np.random.default_rng(2)
    ramp = _ramp(rng, rate_e_s=100.0, frames_per_resultant=2, flash_at_index=3, flare_rate_after=200.0)
    r = F.ramp_flash_test(ramp, (4.0, 4.0), 1.5, conf, 12.0)
    assert r["verdict"] == "SLOPE_CHANGE_FLARE_LIKE"
    assert r["delta_chi2_step_minus_flare"] > 9.0


def test_ramp_single_pixel_cosmic_ray_is_not_a_flash(conf):
    rng = np.random.default_rng(7)
    ramp = _ramp(rng, flash_e=500.0, flash_at_index=3, flash_pixels=[(4, 4)])
    r = F.ramp_flash_test(ramp, (4.0, 4.0), 1.5, conf, 12.0)
    assert r["verdict"] in ("NOT_PSF_SHAPED", "NOT_COINCIDENT")
    assert r["coincidence_frac"] < 0.8


def test_ramp_with_three_resultants_is_insufficient(conf):
    rng = np.random.default_rng(0)
    ramp = _ramp(rng, times_s=np.array([5.0, 20.0, 40.0]), flash_e=500.0, flash_at_index=1)
    r = F.ramp_flash_test(ramp, (4.0, 4.0), 1.5, conf, 12.0)
    assert r["verdict"] == "INSUFFICIENT_RESULTANTS" and r["per_pixel"] == []


def test_ramp_in_dn_uses_the_gain(conf):
    rng = np.random.default_rng(11)
    e = _ramp(rng, flash_e=500.0, flash_at_index=3)
    dn = Ramp(image_id="dn", resultants=e.resultants / 1.8, times_s=e.times_s, unit="DN",
              gain_e_per_dn=1.8, read_pattern=e.read_pattern)
    r_e = F.ramp_flash_test(e, (4.0, 4.0), 1.5, conf, 12.0)
    r_dn = F.ramp_flash_test(dn, (4.0, 4.0), 1.5, conf, 12.0)
    assert r_dn["verdict"] == r_e["verdict"] == "FLASH_CONSISTENT"
    assert r_dn["total_step_e"] == pytest.approx(r_e["total_step_e"], rel=1e-6)
    bare = Ramp(image_id="bare", resultants=e.resultants, times_s=e.times_s, unit="DN")
    r_b = F.ramp_flash_test(bare, (4.0, 4.0), 1.5, conf, 12.0)
    assert any("gain" in n for n in r_b["notes"]) and any("read pattern" in n for n in r_b["notes"])


def test_psf_amplitude_correlation_refines_the_centroid():
    pix = F.footprint_pixels((10.0, 10.0), 2.0, 0.10)
    true = F._psf_values_at(pix, (10.5, 10.0), 2.0)
    out = F.psf_amplitude_correlation(pix, true, (10.0, 10.0), 2.0)
    assert out["corr"] > 0.999 and out["dx"] == 0.5 and out["dy"] == 0.0
    flat = F.psf_amplitude_correlation(pix, np.ones(len(pix)), (10.0, 10.0), 2.0)
    assert flat["corr"] == 0.0
    assert math.isnan(F.psf_amplitude_correlation(pix[:2], true[:2], (10.0, 10.0), 2.0)["corr"])


# --------------------------------------------------------------------------------------
# Sensitivity numbers, next-exposure residual
# --------------------------------------------------------------------------------------

def test_fluence_threshold_matches_section_2_3(conf):
    wfi = conf["instruments"]["WFI"]
    fl = F.fluence_threshold(wfi["read_noise_e"], conf["flash"]["n_sigma_trigger"],
                             wfi["psf_core_fraction"]["F146"], rate_e_s=470.0)
    assert fl["n_e_peak_trigger"] == pytest.approx(60.0)
    assert fl["n_e_total_trigger"] == pytest.approx(300.0)
    assert fl["t_equivalent_s"] == pytest.approx(0.638, abs=0.005)
    assert F.fluence_threshold(12.0, 5.0, 0.20)["t_equivalent_s"] is None
    assert math.isnan(F.fluence_threshold(12.0, 5.0, 0.0)["n_e_total_trigger"])


def test_next_exposure_residual():
    assert F.next_exposure_residual(110.0, 3.0, 100.0, 4.0) == pytest.approx(2.0)
    assert F.next_exposure_residual(90.0, 3.0, 100.0, 4.0) == pytest.approx(-2.0)
    assert math.isnan(F.next_exposure_residual(110.0, 0.0, 100.0, 0.0))
    assert math.isnan(F.next_exposure_residual(float("nan"), 1.0, 100.0, 1.0))


# --------------------------------------------------------------------------------------
# Honest degradation and the selftest
# --------------------------------------------------------------------------------------

def test_honest_degradation(conf):
    star = {"star_id": "A", "x": 10.0, "y": 10.0}
    empty = DQCutout(image_id="empty", dq=np.zeros((21, 21), np.uint32), stars=[star], psf_fwhm_px=1.2)
    out = F.screen_dq_cutout(empty, conf, FLAGS)
    assert out["status"] == "ok" and out["n_clusters"] == 0 and out["candidates"] == [] and out["labels"] == {}
    # other bits set, the jump bit never: still nothing
    other = DQCutout(image_id="other", dq=np.full((21, 21), SAT | FLAGS["HOT"], np.uint32), stars=[star])
    assert F.screen_dq_cutout(other, conf, FLAGS)["n_clusters"] == 0
    # a flag table without JUMP_DET: the screen cannot run and says so
    no_jump = {k: v for k, v in FLAGS.items() if k != "JUMP_DET"}
    fp = F.footprint_pixels((10.0, 10.0), 1.2, 0.10)
    dq = np.zeros((21, 21), np.uint32)
    dq[fp[:, 0], fp[:, 1]] |= JUMP
    out2 = F.screen_dq_cutout(DQCutout(image_id="nojump", dq=dq, stars=[star], psf_fwhm_px=1.2), conf, no_jump)
    assert out2["status"] == "jump_flag_unavailable" and out2["candidates"] == [] and out2["n_clusters"] == 0
    assert out2["funnel"]["counts"]["cutouts_without_jump_flag"] == 1
    # no SATURATED flag: the screen runs, notes it, and the snowball/saturation rules are off
    no_sat = {k: v for k, v in FLAGS.items() if k != "SATURATED"}
    out3 = F.screen_dq_cutout(DQCutout(image_id="nosat", dq=dq, stars=[star], psf_fwhm_px=1.2), conf, no_sat)
    assert out3["status"] == "ok" and out3["labels"] == {"psf_on_star": 1}
    assert out3["funnel"]["counts"]["cutouts_without_saturated_flag"] == 1
    assert F.find_jump_clusters(np.zeros((4, 4), np.uint32), JUMP, 3, 40) == []
    assert F.assess_candidates([], conf)["verdict"] == "NO_SURVIVORS"


def test_flash_conf_accepts_block_or_whole_config(conf):
    whole = F.flash_conf(conf)
    block = F.flash_conf(conf["flash"])
    assert whole == block and whole["tiers"]["candidate"] == "SUB_EXPOSURE_FLASH_CANDIDATE"
    trimmed = F.flash_conf({"flash": {"elongation_max": 2.5}})
    assert trimmed["elongation_max"] == 2.5 and trimmed["min_cluster_px"] == 3


def test_selftest_passes(conf):
    out = F.selftest(conf)
    assert out["passed"] and out["synthetic"]
    assert out["psf_on_star_ids"] == ["S1", "S4", "S7"]
    assert out["flash_ramp"]["verdict"] == "FLASH_CONSISTENT"
    assert out["flare_ramp"]["verdict"] == "SLOPE_CHANGE_FLARE_LIKE"
    json_safe(out)
