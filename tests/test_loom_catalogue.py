"""Offline tests for the full-catalogue momentum-ceiling screen.

The live run happens on a GitHub runner because the sandbox has no egress to
JPL, so everything that can be checked without the network is checked here:
the efficiency arithmetic, the three-component magnitude and its signal-to-noise,
the reliability cuts and what they remove, the population statistics, the ranking
and the verdict — plus the fetch layer itself, driven through a fake transport, so
the self-repairing field request, the truncation guard and the chunk/resume logic
are covered without a single packet leaving the machine.

Four of these are load-bearing regressions rather than unit checks and should not
be weakened:

* :func:`test_standing_exceedances_reproduce_the_calibration_run` pins the screen
  to the two numbers this whole run exists to put in context.  If a refactor
  moves ``875163 (1998 SH2)`` off 1.578 or ``428209 (2006 VC)`` off 1.304, the
  suite says so rather than the tail quietly changing.
* :func:`test_vet_reduces_to_calibrate_for_a2_only_rows` pins the generalised
  reliability gate to ``calibrate.vet_exceedance`` on the rows where the two must
  agree — which is the overwhelming majority of asteroids.
* :func:`test_g_laws_are_normalised_at_one_au` guards the fact that makes a 1 au
  comparison law-independent.  If it were false, every comet in the screen would
  be compared against the wrong ceiling by a constant factor.
* :func:`test_screen_never_promotes_to_candidate` guards LOOM's central rule:
  magnitude alone never promotes, because "inactive body with an acceleration
  above Yarkovsky expectations" is a dark comet and the dark-comet population is
  incomplete.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from seti.loom import calibrate, catalogue, nongrav
from seti.loom.screen import Thresholds

# ---------------------------------------------------------------------------
# synthetic catalogue helpers
# ---------------------------------------------------------------------------
_BASE_ROW = {
    "spkid": "20000001", "pdes": "1", "full_name": "1 Testis (2000 AA)",
    "kind": "an", "class": "MBA", "H": 20.0, "diameter": None, "albedo": None,
    "a": 2.5, "e": 0.2, "i": 5.0, "rms": 0.3, "n_obs_used": 500.0,
    "data_arc": 9000.0, "condition_code": 0.0,
}


def make_row(**kw) -> dict:
    """One SBDB-shaped row, with a clean orbit unless a test spoils it."""
    row = dict(_BASE_ROW)
    row.update(kw)
    return row


def a2_for_epsilon(eps: float, h: float = 20.0, rho: float = 2000.0,
                   albedo: float = 0.14) -> float:
    """The ``A2`` an object of this ``H`` needs in order to realise ``eps``.

    Inverted through the same functions the screen uses, so a test that asks for
    ``eps = 3`` gets exactly ``eps = 3`` however the ceiling is implemented.
    """
    d = float(calibrate.diameter_m(np.array([h]), None, None, default_albedo=albedo)[0])
    unit = float(calibrate.epsilon_effective(1.0, d, rho_kg_m3=rho))
    return eps / unit


def population(n: int, *, mu: float = -1.13, sigma: float = 0.30, seed: int = 7,
               h: float = 20.0) -> list[dict]:
    """A log-normal ordinary-asteroid population, in realised efficiency.

    ``mu = -1.13`` is ``log10(0.074)`` — the median LOOM measured on the published
    ``A2`` population at rho = 2000 — and ``sigma = 0.30`` reproduces its p90 of
    0.143.  So the synthetic sample is not arbitrary: it is the real distribution
    this screen expects to find, which is what makes an injected excess a
    meaningful test rather than a tautology.
    """
    rng = np.random.default_rng(seed)
    eps = 10.0 ** rng.normal(mu, sigma, size=n)
    rows = []
    for i, e in enumerate(eps):
        a2 = a2_for_epsilon(float(e), h=h)
        rows.append(make_row(spkid=f"2000{i:04d}", pdes=str(1000 + i),
                             full_name=f"{1000 + i} Synth{i} (2000 A{i})",
                             A2=a2, A2_sigma=abs(a2) / 20.0))
    return rows


# ---------------------------------------------------------------------------
# 1. the physics: g(r), the ceiling, and where the comparison is made
# ---------------------------------------------------------------------------
def test_g_laws_are_normalised_at_one_au():
    """Both of JPL's g(r) laws equal 1 at 1 au — the fact the screen rests on.

    ``A1/A2/A3`` are coefficients of ``g(r)``, so "the acceleration at 1 au" is
    ``|A|`` only if ``g(1 au) = 1``.  The radiation law gives that for any
    exponent, which is why the ``d = 2`` vs ``d = 2.25`` ambiguity in JPL's
    Yarkovsky fits never reaches the ceiling comparison.  The sublimation law
    gives it only because ``alpha = 0.1112620426`` was chosen to make it so, and
    that is worth computing rather than believing.
    """
    g = catalogue.g_normalisation()
    assert g["g_radiation_d2_at_1au"] == pytest.approx(1.0, abs=1e-12)
    assert g["g_radiation_d225_at_1au"] == pytest.approx(1.0, abs=1e-12)
    assert g["g_comet_at_1au"] == pytest.approx(1.0, abs=1e-6)


def test_epsilon_at_distance_is_flat_for_radiation_and_falls_for_sublimation():
    """The ratio is distance-free under the radiation law and is not under the other.

    This is why ``A2`` is the natural quantity for a Yarkovsky screen, and why a
    comet whose perihelion is 3 au is being credited at 1 au with an efficiency it
    never realises anywhere.
    """
    for r in (0.5, 1.0, 2.0, 5.0):
        assert catalogue.epsilon_at_distance(0.4, r, law="radiation") == pytest.approx(0.4)
    assert catalogue.epsilon_at_distance(1.0, 1.0, law="sublimation") == pytest.approx(1.0, abs=1e-6)
    # Falls off steeply beyond the sublimation knee...
    assert catalogue.epsilon_at_distance(1.0, 3.0, law="sublimation") < 0.05
    # ...and rises modestly inside 1 au.
    assert 1.0 < catalogue.epsilon_at_distance(1.0, 0.5, law="sublimation") < 1.5
    assert math.isnan(catalogue.epsilon_at_distance(float("nan"), 1.0))
    assert math.isnan(catalogue.epsilon_at_distance(1.0, 0.0))
    with pytest.raises(ValueError):
        catalogue.epsilon_at_distance(1.0, 1.0, law="magic")


def test_epsilon_agrees_with_the_nongrav_ceiling_ratio():
    """The screen's efficiency and ``nongrav.ceiling_ratio`` are the same number.

    Two implementations of the momentum ceiling exist in this channel — the
    ``H``-only one the Rubin screen uses and the measured-diameter one the SBDB
    path uses — and they must coincide wherever their inputs coincide.  A silent
    disagreement between them would make the two screens' tiers incomparable.
    """
    h, a2 = 21.0, 4.0e-13
    row = make_row(H=h, A2=a2, A2_sigma=a2 / 10.0)
    entry = catalogue.screen_entry(row, rho_kg_m3=1000.0, default_albedo=0.25)
    expected = float(nongrav.ceiling_ratio(h, a2, albedo=0.25, rho_kg_m3=1000.0,
                                           epsilon=1.0))
    assert entry["epsilon_1au"] == pytest.approx(expected, rel=1e-9)


def test_standing_exceedances_reproduce_the_calibration_run():
    """The two objects this run exists to put in context, pinned to their numbers.

    Values from ``results/loom/calibration.json`` (the live 939-row ``A2|DF``
    pull, 2026-07-30).  Both were fitted with ``A2`` alone, so the
    three-component magnitude must reduce to ``|A2|`` and the efficiency must come
    back unchanged: 1.578 for ``875163 (1998 SH2)`` with a *measured* 383 m
    diameter, 1.304 for ``428209 (2006 VC)`` from ``H`` alone.
    """
    sh2 = catalogue.screen_entry(make_row(
        spkid="20875163", pdes="875163", full_name="875163 (1998 SH2)",
        H=20.88, **{"class": "AMO"}, diameter=0.383, albedo=0.058, a=2.743, e=0.7138, i=2.43,
        A2=-7e-13, A2_sigma=4.9e-14, rms=0.32627, n_obs_used=394.0, data_arc=9900.0,
        condition_code=0.0))
    vc = catalogue.screen_entry(make_row(
        spkid="20428209", pdes="428209", full_name="428209 (2006 VC)",
        H=20.09, diameter=None, albedo=None, a=1.941, e=0.4921, i=12.25,
        A2=6.5e-13, A2_sigma=1.7e-13, rms=0.44348, n_obs_used=175.0, data_arc=7113.0,
        condition_code=0.0))

    assert sh2["epsilon_1au"] == pytest.approx(1.5779627873813493, rel=1e-6)
    assert sh2["A_snr"] == pytest.approx(14.2857142857, rel=1e-6)
    assert sh2["components_fitted"] == ["A2"]
    assert sh2["reliable"] and sh2["fails"] == []
    assert sh2["diameter_measured"] is True
    assert sh2["tisserand_j"] == pytest.approx(2.9133106866, rel=1e-6)
    assert sh2["comet_like_dynamics"] is True
    assert sh2["standing_exceedance"] == "875163"

    assert vc["epsilon_1au"] == pytest.approx(1.3036897858, rel=1e-6)
    assert vc["A_snr"] == pytest.approx(3.8235294118, rel=1e-6)
    assert vc["del_vigna_R"] == pytest.approx(9.784446528, rel=1e-6)
    assert vc["comet_like_dynamics"] is False
    assert vc["standing_exceedance"] == "428209"
    # Both are above the hard ceiling but neither is a candidate: the momentum
    # ceiling cannot separate an engineered object from an unpublished dark comet.
    th = Thresholds()
    for entry in (sh2, vc):
        assert catalogue.record_from_entry(entry, th).tier == "interest"


# ---------------------------------------------------------------------------
# 2. the non-gravitational magnitude
# ---------------------------------------------------------------------------
def test_magnitude_combines_all_three_components():
    vec = catalogue.nongrav_vector({"A1": 3.0e-13, "A2": 4.0e-13,
                                    "A1_sigma": 1.0e-14, "A2_sigma": 1.0e-14})
    assert vec.magnitude == pytest.approx(5.0e-13)
    assert vec.components == ("A1", "A2")
    # sigma^2 = (0.6 s1)^2 + (0.8 s2)^2 with s1 = s2 = 1e-14  ->  1e-14
    assert vec.sigma == pytest.approx(1.0e-14, rel=1e-9)
    assert vec.snr == pytest.approx(50.0, rel=1e-9)


def test_magnitude_reduces_to_abs_a2_when_a2_is_all_there_is():
    vec = catalogue.nongrav_vector({"A2": -7e-13, "A2_sigma": 4.9e-14})
    assert vec.magnitude == pytest.approx(7e-13)
    assert vec.snr == pytest.approx(7e-13 / 4.9e-14)
    assert vec.components == ("A2",)


def test_a_component_without_an_uncertainty_poisons_the_snr():
    """A fitted parameter with no uncertainty has no signal-to-noise, and says so.

    Substituting a default would silently promote it, which is the single most
    expensive error available in a screen whose output is a shortlist.
    """
    vec = catalogue.nongrav_vector({"A1": 1e-12, "A2": 1e-12, "A2_sigma": 1e-14})
    assert vec.components == ("A1", "A2")
    assert vec.missing_sigma == ("A1",)
    assert math.isnan(vec.snr)
    vet = catalogue.vet_catalogue_row(make_row(A1=1e-12, A2=1e-12, A2_sigma=1e-14))
    assert not vet["reliable"]
    assert "no_uncertainty_for_A1" in vet["fails"]


def test_exact_zero_is_missing_not_a_measured_zero():
    """Zero is absence, never "we looked and it was fine".

    A *measured* non-gravitational term of zero is the strongest possible
    statement that an object is ordinary; an unfitted one means untestable, and
    the whole channel is built on not confusing the two.
    """
    vec = catalogue.nongrav_vector({"A1": 0.0, "A2": 0.0, "A3": None})
    assert vec.components == ()
    assert math.isnan(vec.magnitude)
    entry = catalogue.screen_entry(make_row(A2=0.0))
    assert entry["components_fitted"] == []
    assert "no_fitted_nongrav_parameter" in entry["fails"]


# ---------------------------------------------------------------------------
# 3. reliability, and its relationship to the calibration screen
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spoil", [
    {},
    {"rms": 2.0},
    {"data_arc": 100.0},
    {"data_arc": None},
    {"n_obs_used": 12.0},
    {"condition_code": 7.0},
    {"A2_sigma": None},
    {"A2": 1e-14, "A2_sigma": 1e-14},
    {"two_body": "Y"},
])
def test_vet_reduces_to_calibrate_for_a2_only_rows(spoil):
    """On an A2-only row this gate must be ``calibrate.vet_exceedance``, exactly.

    The generalisation to ``|A|`` is only allowed to change the *parameter*
    signal-to-noise, and only for objects with more than one fitted component.
    Every other gate — orbit RMS, arc length, observation count, condition code,
    two-body-only solutions — is reused rather than restated, and this test is
    what stops the two from drifting apart threshold by threshold.
    """
    row = make_row(**{"A2": -7e-13, "A2_sigma": 4.9e-14, **spoil})
    mine = catalogue.vet_catalogue_row(row)
    theirs = calibrate.vet_exceedance(row)
    assert mine["reliable"] == theirs["reliable"]
    # The SNR clause is renamed (`a_snr_` rather than `a2_snr_`) but must fire on
    # exactly the same rows; everything else must match word for word.
    def canon(reasons):
        return sorted(r.replace("a_snr_", "a2_snr_")
                       .replace("no_nongrav_uncertainty", "no_a2_uncertainty")
                       .replace("no_uncertainty_for_A2", "no_a2_uncertainty")
                      for r in reasons)

    assert canon(mine["fails"]) == canon(theirs["fails"])


def test_an_a1_only_object_is_not_rejected_for_lacking_an_a2():
    """The bug this generalisation exists to fix.

    ``calibrate.vet_exceedance`` gates on ``A2`` alone, so a comet fitted with a
    radial term and no transverse one comes back ``no_a2_uncertainty`` — rejected
    for lacking a parameter it never had.  That is not a reliability judgement, it
    is a category error, and it would have removed most of the comet catalogue.
    """
    row = make_row(A1=2.0e-9, A1_sigma=1.0e-10, A2=None, A2_sigma=None,
                   full_name="1P/Halley", **{"class": "HTC"})
    assert "no_a2_uncertainty" in calibrate.vet_exceedance(row)["fails"]
    mine = catalogue.vet_catalogue_row(row)
    assert mine["reliable"] is True
    assert mine["fails"] == []
    assert mine["components_fitted"] == ["A1"]


def test_orbit_quality_failures_reach_the_tier_reasons():
    """A badly determined orbit vetoes promotion and is named, not implied."""
    entry = catalogue.screen_entry(make_row(A2=a2_for_epsilon(5.0),
                                            A2_sigma=a2_for_epsilon(5.0) / 20.0,
                                            data_arc=200.0, n_obs_used=30.0))
    rec = catalogue.record_from_entry(entry, Thresholds())
    assert rec.tier == "interest"          # above the ceiling, but
    assert any(r.startswith("arc_") for r in rec.reasons)
    assert any("observations" in r for r in rec.reasons)
    assert not entry["reliable"]


def test_low_snr_is_untestable_not_ordinary():
    """The 'Oumuamua case: the most interesting number with nothing behind it.

    On the live run ``2002 AX51`` exceeded the realistic envelope by 6.5x at
    S/N 1.16 and was recorded ``untestable`` rather than promoted.  The same must
    happen here, and — critically — it must not come out ``ordinary``, which
    would fold "we could not look" in with "we looked and it was fine".
    """
    a2 = a2_for_epsilon(20.0)
    entry = catalogue.screen_entry(make_row(A2=a2, A2_sigma=a2 / 1.2))
    rec = catalogue.record_from_entry(entry, Thresholds())
    assert rec.tier == "untestable"
    assert any("snr" in r for r in rec.reasons)


def test_no_absolute_magnitude_is_untestable():
    entry = catalogue.screen_entry(make_row(H=None, diameter=None,
                                            A2=1e-12, A2_sigma=1e-14))
    assert math.isnan(entry["epsilon_1au"])
    rec = catalogue.record_from_entry(entry, Thresholds())
    assert rec.tier == "untestable"
    assert "no_absolute_magnitude" in rec.reasons


# ---------------------------------------------------------------------------
# 4. the cut ledger — what each gate removed
# ---------------------------------------------------------------------------
def test_cut_ledger_counts_marginally_and_sequentially():
    """Both columns, because they answer different questions.

    Marginal counts are order-free and answer "is this cut doing anything?";
    sequential counts answer "where did the population go?" and are the ones that
    sum to the total.  A marginal count can exceed a sequential one (the cut is
    redundant against an earlier one); the reverse is impossible.
    """
    good = [catalogue.screen_entry(r) for r in population(40, seed=3)]
    # Ten objects that fail arc length AND observation count — correlated, which
    # is the normal case for a small-body catalogue and the reason both columns
    # exist.
    bad = [catalogue.screen_entry(make_row(spkid=f"999{i}", pdes=f"999{i}",
                                           A2=1e-13, A2_sigma=1e-14,
                                           data_arc=90.0, n_obs_used=20.0))
           for i in range(10)]
    led = catalogue.cut_ledger(good + bad)
    by = {c["cut"]: c for c in led["cuts"]}
    assert led["n_entering"] == 50
    assert led["n_surviving"] == 40
    assert by["data_arc"]["n_failing_alone"] == 10
    assert by["n_obs_used"]["n_failing_alone"] == 10
    # `data_arc` comes first in the pipeline, so it takes all ten and
    # `n_obs_used` — which alone would also have taken ten — takes none.
    assert by["data_arc"]["n_removed_in_sequence"] == 10
    assert by["n_obs_used"]["n_removed_in_sequence"] == 0
    assert sum(c["n_removed_in_sequence"] for c in led["cuts"]) == 10
    for c in led["cuts"]:
        assert c["n_failing_alone"] >= c["n_removed_in_sequence"]


def test_cut_ledger_makes_a_catalogue_eating_cut_visible():
    """A cut that removes 90% of the population must show that on its own line.

    This is the reporting requirement, not a physics one: a screen whose
    denominator collapsed silently would report an exceedance fraction that means
    something completely different from what a reader assumes.
    """
    rows = population(100, seed=11)
    for r in rows[:90]:
        r["data_arc"] = 300.0
    entries = [catalogue.screen_entry(r) for r in rows]
    led = catalogue.cut_ledger(entries)
    arc = next(c for c in led["cuts"] if c["cut"] == "data_arc")
    assert arc["n_failing_alone"] == 90
    assert arc["fraction_failing_alone"] == pytest.approx(0.9)
    assert led["fraction_surviving"] == pytest.approx(0.1)


def test_cut_ledger_handles_an_empty_population():
    led = catalogue.cut_ledger([])
    assert led["n_entering"] == 0
    assert math.isnan(led["fraction_surviving"])


# ---------------------------------------------------------------------------
# 5. the distribution and its expected tail
# ---------------------------------------------------------------------------
def test_epsilon_distribution_reports_shape_not_just_quantiles():
    dist = catalogue.epsilon_distribution([0.05, 0.07, 0.1, 0.15, 0.4, 1.5, 3.0])
    assert dist["n"] == 7
    assert dist["quantiles"]["max"] == pytest.approx(3.0)
    # p99 and p99.9 must be DIFFERENT keys.  `int(0.999 * 100)` is 99, so a naive
    # label would report the 99.9th percentile under the 99th's name -- in exactly
    # the part of the distribution this screen is about.
    assert dist["quantiles"]["p999"] >= dist["quantiles"]["p99"]
    assert dist["quantiles"]["p99"] == pytest.approx(
        float(np.quantile([0.05, 0.07, 0.1, 0.15, 0.4, 1.5, 3.0], 0.99)))
    assert dist["n_above_hard_1"] == 2
    assert dist["n_above_specular_2"] == 1
    assert dist["fraction_above_hard_1"] == pytest.approx(2 / 7)
    hist = dist["histogram"]
    assert sum(hist["counts"]) == 7
    assert len(hist["log10_epsilon_edges"]) == len(hist["counts"]) + 1


def test_epsilon_distribution_degrades_on_an_empty_sample():
    dist = catalogue.epsilon_distribution([float("nan"), 0.0, -1.0])
    assert dist["n"] == 0
    assert "reason" in dist and "quantiles" not in dist


def test_tail_expectation_recovers_a_clean_lognormal():
    """With no excess, expected and observed agree and the Poisson p is unremarkable."""
    rng = np.random.default_rng(19)
    eps = 10.0 ** rng.normal(-1.13, 0.30, size=20000)
    out = catalogue.tail_expectation(eps)
    assert out["ok"]
    assert out["log10_mu"] == pytest.approx(-1.13, abs=0.02)
    assert out["log10_sigma"] == pytest.approx(0.30, abs=0.02)
    at_realistic = next(r for r in out["thresholds"]
                        if r["threshold"] == pytest.approx(nongrav.EPSILON_REALISTIC))
    assert at_realistic["expected_lognormal"] > 0
    assert at_realistic["poisson_p_at_least_observed"] > 0.01
    # And the model check at the sample's own quantiles must not already be broken,
    # or the extrapolation to eps = 1 would be worthless.
    for row in out["model_check"]:
        assert row["expected_lognormal"] == pytest.approx(row["observed"], rel=0.35)


def test_tail_expectation_flags_an_injected_excess():
    """An injected population above the ceiling shows up as an excess, plainly.

    This is the direction that matters: if the tail is denser than an ordinary
    thermal-recoil population predicts, that is a statement about the *tail* — most
    likely an unpublished dark-comet population — and it makes any individual
    exceedance in it unremarkable.  The screen has to be able to say so.
    """
    rng = np.random.default_rng(23)
    core = 10.0 ** rng.normal(-1.13, 0.30, size=5000)
    injected = np.concatenate([core, np.full(60, 3.0)])
    clean = catalogue.tail_expectation(core)
    out = catalogue.tail_expectation(injected)
    hard = next(r for r in out["thresholds"] if r["threshold"] == pytest.approx(1.0))
    baseline = next(r for r in clean["thresholds"] if r["threshold"] == pytest.approx(1.0))
    assert hard["observed"] == baseline["observed"] + 60
    assert hard["expected_lognormal"] < 1.0
    assert hard["excess"] > 50
    assert hard["poisson_p_at_least_observed"] < 1e-6


def test_tail_expectation_core_fit_is_immune_to_the_tails_values():
    """The objects under test cannot widen their own expectation by being extreme.

    Fitting the scale on the full sample is the classic way to make an excess
    disappear.  The lower-half estimator here sees nothing above the median, so
    moving the injected tail from ``eps = 5`` to ``eps = 5e4`` — four orders of
    magnitude — leaves ``mu`` and ``sigma`` bit-identical.
    """
    rng = np.random.default_rng(29)
    core = 10.0 ** rng.normal(-1.13, 0.30, size=4000)
    near = catalogue.tail_expectation(np.concatenate([core, np.full(40, 5.0)]))
    far = catalogue.tail_expectation(np.concatenate([core, np.full(40, 5.0e4)]))
    assert far["log10_sigma"] == near["log10_sigma"]
    assert far["log10_mu"] == near["log10_mu"]


def test_tail_expectation_core_fit_is_only_first_order_immune_to_the_tails_count():
    """And the residual sensitivity is bounded, measured, and in the safe direction.

    Every injected object shifts the quantile *positions* even when it cannot
    shift their values, so a tail holding 5% of the sample inflates ``sigma`` by a
    few percent.  That bias raises the expected tail count and therefore makes an
    excess harder to claim, never easier — and a sample with a 5% tail is a
    ``TAIL_DENSELY_POPULATED`` verdict in any case.
    """
    rng = np.random.default_rng(31)
    core = 10.0 ** rng.normal(-1.13, 0.30, size=4000)
    clean = catalogue.tail_expectation(core)
    heavy = catalogue.tail_expectation(np.concatenate([core, np.full(200, 50.0)]))
    assert heavy["log10_sigma"] > clean["log10_sigma"]           # conservative
    assert heavy["log10_sigma"] == pytest.approx(clean["log10_sigma"], rel=0.06)


def test_tail_expectation_refuses_a_sample_too_small_to_fit():
    out = catalogue.tail_expectation([0.1] * 5)
    assert out["ok"] is False
    assert "only 5 objects" in out["reason"]


def test_tail_expectation_refuses_a_degenerate_core():
    out = catalogue.tail_expectation([0.1] * 50)
    assert out["ok"] is False
    assert "zero width" in out["reason"]


def test_normal_quantile_and_poisson_helpers():
    assert catalogue._normal_quantile(0.5) == pytest.approx(0.0, abs=1e-9)
    assert catalogue._normal_quantile(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert catalogue._normal_quantile(0.025) == pytest.approx(-1.959964, abs=1e-5)
    assert math.isnan(catalogue._normal_quantile(1.0))
    assert catalogue._poisson_sf(0, 3.0) == pytest.approx(1.0)
    assert catalogue._poisson_sf(1, 2.0) == pytest.approx(1 - math.exp(-2.0), rel=1e-9)
    assert catalogue._poisson_sf(5, 0.0) == 0.0
    assert catalogue._poisson_sf(20, 0.5) < 1e-15


# ---------------------------------------------------------------------------
# 6. the whole screen
# ---------------------------------------------------------------------------
def test_screen_counts_its_denominator_and_deduplicates():
    """An object returned by two constrained queries is ONE object.

    ``A1|DF`` and ``A2|DF`` both return every object with both fitted, and
    counting it twice would inflate the denominator and — much worse — double any
    entry in the tail, which is a list a human reads.
    """
    rows = population(30, seed=5)
    scr = catalogue.screen_catalogue(rows + rows[:10])
    assert scr.n_rows == 40
    assert scr.n_unique == 30
    assert scr.populations["all"]["n_objects"] == 30
    assert scr.populations["asteroid"]["epsilon_all_screened"]["n"] == 30


def test_screen_merges_fields_across_duplicate_pulls():
    """The later pull's fields fill the earlier pull's gaps rather than being lost."""
    a = make_row(spkid="20001", A1=1e-12, A1_sigma=1e-14, A2=None, A2_sigma=None)
    b = make_row(spkid="20001", A1=1e-12, A1_sigma=1e-14, A2=2e-12, A2_sigma=1e-14)
    scr = catalogue.screen_catalogue([a, b])
    assert scr.n_unique == 1
    entry = scr.entries[0]
    assert entry["components_fitted"] == ["A1", "A2"]


def test_screen_separates_comets_from_asteroids():
    """Different physics, so a separate denominator — the point, not a refinement.

    A comet accelerates by shedding mass and is not bound by the radiation
    momentum budget at all, so pooling the two turns the ceiling separating them
    cleanly into a summary statistic that reads like a failure.
    """
    asteroids = population(25, seed=13)
    comets = [make_row(spkid=f"1000{i:03d}", pdes=f"{i}P", full_name=f"{i}P/Synth",
                       A1=1e-9, A1_sigma=1e-11, H=None, diameter=2.0,
                       **{"class": "JFc"}) for i in range(1, 9)]
    scr = catalogue.screen_catalogue(asteroids + comets)
    assert scr.populations["asteroid"]["n_objects"] == 25
    assert scr.populations["comet"]["n_objects"] == 8
    assert scr.populations["all"]["n_objects"] == 33
    # Every comet blows through the ceiling, which is the ceiling working.
    assert scr.populations["comet"]["epsilon_all_screened"]["n_above_hard_1"] == 8
    assert scr.populations["asteroid"]["epsilon_all_screened"]["n_above_hard_1"] == 0


def test_screen_never_promotes_to_candidate():
    """Magnitude alone never promotes, however extreme — LOOM's central rule.

    "Inactive small body with an acceleration above Yarkovsky expectations" is
    Seligman et al. (2023)'s dark comets, and that population is known to be
    incomplete, so a magnitude cut *will* keep finding unpublished members of it.
    Promotion requires an artificiality channel and SBDB's query fields carry
    none, so the ceiling tops out at ``interest``.
    """
    a2 = a2_for_epsilon(1.0e6)
    rows = population(20, seed=17) + [make_row(spkid="20999999", pdes="999999",
                                               full_name="999999 (2099 ZZ)",
                                               A2=a2, A2_sigma=a2 / 100.0)]
    scr = catalogue.screen_catalogue(rows)
    assert scr.populations["asteroid"]["tiers"]["candidate"] == 0
    assert scr.populations["asteroid"]["tiers"]["interest"] == 1
    assert scr.tail[0]["tier"] == "interest"
    assert scr.tail[0]["epsilon_1au"] == pytest.approx(1.0e6, rel=1e-6)


def test_tail_is_ranked_and_carries_its_assumption_grid():
    rows = population(20, seed=19)
    for eps, i in ((5.0, 0), (2.0, 1), (12.0, 2)):
        a2 = a2_for_epsilon(eps)
        rows[i]["A2"], rows[i]["A2_sigma"] = a2, a2 / 30.0
    scr = catalogue.screen_catalogue(rows)
    eps_seen = [t["epsilon_1au"] for t in scr.tail]
    assert eps_seen == sorted(eps_seen, reverse=True)
    assert [t["tail_rank"] for t in scr.tail] == [1, 2, 3]
    grid = scr.tail[0]["sensitivity"]
    assert grid["grid"] and "epsilon_min" in grid and "epsilon_max" in grid
    # An exceedance that survives every density and albedo in the grid is a
    # different claim from one that lives in a corner of it.
    assert grid["epsilon_min"] <= scr.tail[0]["epsilon_1au"] <= grid["epsilon_max"] * 1.0001
    # Both densities are attached to the entry, so the reader never has to join.
    assert "epsilon_1au_rho_1000" in scr.tail[0]
    assert scr.tail[0]["epsilon_1au_rho_1000"] == pytest.approx(
        scr.tail[0]["epsilon_1au"] / 2.0, rel=1e-9)


def test_survivors_exclude_the_unreliable_and_the_already_known():
    """Only what litcheck should be asked about: new, reliable, above the ceiling."""
    a2 = a2_for_epsilon(6.0)
    rows = population(20, seed=23) + [
        # reliable and unannotated -> a survivor
        make_row(spkid="20111111", pdes="111111", full_name="111111 (2011 AA)",
                 A2=a2, A2_sigma=a2 / 30.0),
        # above the ceiling but the arc is 200 days -> not reliable
        make_row(spkid="20222222", pdes="222222", full_name="222222 (2022 BB)",
                 A2=a2, A2_sigma=a2 / 30.0, data_arc=200.0),
        # above the ceiling and already known to be anomalous
        make_row(spkid="20003200", pdes="3200", full_name="3200 Phaethon (1983 TB)",
                 A2=a2, A2_sigma=a2 / 30.0),
    ]
    scr = catalogue.screen_catalogue(rows)
    names = [s["name"] for s in scr.survivors]
    assert names == ["111111 (2011 AA)"]
    known = [t for t in scr.tail if t["known_as"]]
    assert any("Phaethon" in t["known_as"] for t in known)
    # Every survivor carries the dynamics tests litcheck needs to read it.
    for s in scr.survivors:
        assert "tisserand_j" in s and "comet_like_dynamics" in s


def test_standing_report_places_both_objects_in_the_distribution():
    """The question the run was commissioned to answer, answered explicitly."""
    a2 = a2_for_epsilon(1.578, h=20.88)
    rows = population(300, seed=31) + [
        make_row(spkid="20875163", pdes="875163", full_name="875163 (1998 SH2)",
                 H=20.88, A2=a2, A2_sigma=a2 / 14.3),
        make_row(spkid="20428209", pdes="428209", full_name="428209 (2006 VC)",
                 H=20.09, A2=a2_for_epsilon(1.304, h=20.09),
                 A2_sigma=a2_for_epsilon(1.304, h=20.09) / 3.82),
        # Something bigger, so the standing pair are not automatically rank 1.
        make_row(spkid="20777777", pdes="777777", full_name="777777 (2077 CC)",
                 A2=a2_for_epsilon(40.0), A2_sigma=a2_for_epsilon(40.0) / 30.0),
    ]
    scr = catalogue.screen_catalogue(rows)
    sh2 = scr.standing["875163"]
    assert sh2["status"] == "SCREENED"
    assert sh2["epsilon_1au"] == pytest.approx(1.578, rel=1e-3)
    assert sh2["n_asteroids_above_it"] == 1
    assert sh2["tail_rank"] == 2
    assert 0.99 < sh2["percentile_among_screened_asteroids"] <= 1.0
    assert sh2["n_asteroids_above_hard_ceiling"] == 3
    assert sh2["n_comets_above_hard_ceiling"] == 0
    assert scr.standing["428209"]["tail_rank"] == 3


def test_standing_report_says_so_when_an_object_is_absent():
    """An absence is a finding with a name, never a blank.

    A refit since 2026-07-30 could remove either object's ``A2`` — and if it has,
    the exceedance is not reproduced and must not be quoted again until somebody
    finds out why.
    """
    scr = catalogue.screen_catalogue(population(30, seed=37))
    for key in catalogue.STANDING_EXCEEDANCES:
        assert scr.standing[key]["status"] == "NOT_IN_SCREENED_POPULATION"
        assert "not reproduced here" in scr.standing[key]["note"]


def test_verdict_calls_a_crowded_tail_what_it_is():
    """More than 5% above the ceiling is a population, not a shortlist.

    The failure mode this guards against is the opposite of a false null: a tail
    of hundreds reported as "N objects above the ceiling!" when what it means is
    that the dark-comet population is large and unpublished and the ceiling alone
    cannot separate it.
    """
    rows = population(100, seed=41)
    for r in rows[:20]:
        r["A2"] = a2_for_epsilon(4.0)
        r["A2_sigma"] = r["A2"] / 30.0
    scr = catalogue.screen_catalogue(rows)
    assert scr.verdict == "TAIL_DENSELY_POPULATED"
    assert "no individual exceedance" in scr.headline
    assert "20" in scr.headline


def test_verdict_when_nothing_exceeds_the_ceiling():
    scr = catalogue.screen_catalogue(population(60, seed=43))
    assert scr.verdict == "NOTHING_ABOVE_CEILING"
    assert "0 (0.00%) exceed it" in scr.headline


def test_verdict_when_every_exceedance_is_already_identified():
    a2 = a2_for_epsilon(9.0)
    rows = population(60, seed=47) + [
        make_row(spkid="20003200", pdes="3200", full_name="3200 Phaethon (1983 TB)",
                 A2=a2, A2_sigma=a2 / 30.0)]
    scr = catalogue.screen_catalogue(rows)
    assert scr.verdict == "ALL_EXCEEDANCES_ALREADY_IDENTIFIED"


def test_verdict_when_a_new_survivor_is_present():
    a2 = a2_for_epsilon(9.0)
    rows = population(60, seed=53) + [
        make_row(spkid="20121212", pdes="121212", full_name="121212 (2012 XX)",
                 A2=a2, A2_sigma=a2 / 30.0)]
    scr = catalogue.screen_catalogue(rows)
    assert scr.verdict == "TAIL_SPARSE_SURVIVORS_PRESENT"
    assert scr.survivors and scr.survivors[0]["name"] == "121212 (2012 XX)"


def test_verdict_when_nothing_is_screenable():
    scr = catalogue.screen_catalogue([make_row(A2=None, H=None)])
    assert scr.verdict == "NO_SCREENABLE_POPULATION"


def test_tail_listing_is_capped_and_says_so():
    rows = population(60, seed=59)
    for r in rows:
        r["A2"] = a2_for_epsilon(3.0)
        r["A2_sigma"] = r["A2"] / 30.0
    scr = catalogue.screen_catalogue(rows, max_tail=10)
    assert len(scr.tail) == 10
    assert any("60 asteroids exceed the hard ceiling" in n for n in scr.notes)


def test_screen_output_is_json_serialisable():
    """The record has to survive `_write_json`, which refuses NaN."""
    scr = catalogue.screen_catalogue(population(40, seed=61))
    from seti.loom.run import _clean

    json.dumps(_clean(scr.as_dict(max_entries=3)), allow_nan=False)


# ---------------------------------------------------------------------------
# 7. the census and the completeness check
# ---------------------------------------------------------------------------
def test_census_counts_each_component_and_the_union():
    rows = [{"spkid": "1", "A1": 1e-9, "A2": None, "A3": None},
            {"spkid": "2", "A1": None, "A2": 3e-13, "A3": None},
            {"spkid": "3", "A1": 1e-9, "A2": 2e-13, "A3": 1e-14},
            {"spkid": "4", "A1": 0.0, "A2": 0.0, "A3": None},
            {"spkid": "5"}]
    c = catalogue.census_counts(rows)
    assert c["n_rows"] == 5
    assert c["n_with_A1"] == 2 and c["n_with_A2"] == 2 and c["n_with_A3"] == 1
    # Object 4's zeros are ABSENCE, not a measured zero.
    assert c["n_with_any"] == 3
    assert c["fraction_with_any"] == pytest.approx(0.6)


def test_completeness_check_catches_a_constrained_query_that_missed_rows():
    """The check that makes a population fraction believable.

    A ``sb-cdata`` constraint that quietly misses rows is invisible from inside
    the constrained result — there is nothing to compare it against — so it is
    compared against the unconstrained census instead.
    """
    census = {"keys_with_any": ["1", "2", "3"]}
    ok = catalogue.completeness_check(census, ["1", "2", "3"])
    assert ok["verdict"] == "CONSTRAINT_COMPLETE"
    bad = catalogue.completeness_check(census, ["1", "2"])
    assert bad["verdict"] == "CONSTRAINT_INCOMPLETE"
    assert bad["missing_sample"] == ["3"]
    assert "lower bound" in bad["note"]
    none = catalogue.completeness_check({"keys_with_any": []}, ["1"])
    assert none["verdict"] == "NO_CENSUS"
    assert "ASSUMED" in none["note"]


# ---------------------------------------------------------------------------
# 8. the fetch layer, driven through a fake transport
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """A ``requests``-shaped transport that records every call it was given."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}),
                           "timeout": timeout})
        return self.handler(dict(params or {}), len(self.calls) - 1)


def test_sbdb_query_repairs_an_invalid_field_by_name():
    """One bad field name 400s the WHOLE query, so the request repairs itself.

    The API names the field it objected to, which is enough to drop it and retry
    — and it is how the correct spelling of ``A1_sigma`` gets discovered rather
    than assumed.  A guess left in the main list is what cost the second
    calibration run every field it asked for.
    """
    def handler(params, i):
        fields = params["fields"].split(",")
        for bad in ("A1_sigma", "A3_sigma"):
            if bad in fields:
                return FakeResponse(400, text=json.dumps(
                    {"code": "400", "message": f"invalid field specified: '{bad}'"}))
        return FakeResponse(200, {"fields": ["pdes", "A2"], "count": 2,
                                  "data": [["1", 1e-13], ["2", 2e-13]]})

    sess = FakeSession(handler)
    rec = catalogue.sbdb_query({"sb-kind": "a"},
                               ["pdes", "A2", "A1_sigma", "A3_sigma"], session=sess)
    assert rec["status"] == 200
    assert rec["dropped_fields"] == ["A1_sigma", "A3_sigma"]
    assert rec["fields_used"] == ["pdes", "A2"]
    assert rec["n_rows"] == 2
    assert rec["server_count"] == 2
    assert rec["rows"][0] == {"pdes": "1", "A2": 1e-13}


def test_sbdb_query_flags_a_truncated_response():
    """Exactly ``limit`` rows is a row CAP, not a row count.

    Read as a population it would silently understate the denominator, which is
    the one number this whole module exists to establish.
    """
    n = catalogue.SBDB_ROW_LIMIT
    payload = {"fields": ["pdes"], "count": n, "data": [[str(i)] for i in range(n)]}
    sess = FakeSession(lambda p, i: FakeResponse(200, payload))
    rec = catalogue.sbdb_query({"sb-kind": "a"}, ["pdes"], session=sess)
    assert rec["truncated"] is True
    assert "TRUNCATED" in rec["note"]


def test_sbdb_query_records_a_transport_failure_rather_than_returning_empty():
    class Boom:
        def get(self, *a, **kw):
            raise TimeoutError("read timed out")

    rec = catalogue.sbdb_query({"sb-kind": "a"}, ["pdes"], session=Boom())
    assert "TimeoutError" in rec["error"]
    assert rec["rows"] == [] and rec["n_rows"] == 0
    assert rec["status"] is None


def test_sbdb_query_records_a_non_400_http_error():
    sess = FakeSession(lambda p, i: FakeResponse(503, text="service unavailable"))
    rec = catalogue.sbdb_query({"sb-kind": "a"}, ["pdes"], session=sess)
    assert rec["status"] == 503
    assert rec["body"] == "service unavailable"
    assert rec["rows"] == []


def test_fetch_catalogue_issues_six_detail_chunks_then_the_census():
    """Six constrained pulls, two census pulls, checkpointed after every one.

    Ordering is deliberate and is tested because it is the difference between a
    cancelled job losing a denominator and a cancelled job losing everything: the
    detail pulls come first because they are what the screen needs, and the
    ~1.55-million-row census comes last because it only adds a denominator to a
    result already on disk.
    """
    def handler(params, i):
        if "sb-cdata" in params:
            comp = json.loads(params["sb-cdata"])["AND"][0].split("|")[0]
            return FakeResponse(200, {
                "fields": ["spkid", "pdes", comp],
                "data": [[f"{params['sb-kind']}{comp}", f"{params['sb-kind']}{comp}",
                          1e-12]]})
        if params.get("info") == "field":
            return FakeResponse(200, {"field": [{"name": f} for f in
                                                ("spkid", "pdes", "A2_sigma")]})
        return FakeResponse(200, {"fields": ["spkid", "A1", "A2", "A3"],
                                  "count": 2,
                                  "data": [["x", 1e-9, None, None],
                                           ["y", None, None, None]]})

    sess = FakeSession(handler)
    seen: list[str] = []
    out = catalogue.fetch_catalogue(session=sess, on_result=lambda n, v: seen.append(n))
    assert seen == ["detail:a:A1", "detail:a:A2", "detail:a:A3",
                    "detail:c:A1", "detail:c:A2", "detail:c:A3",
                    "census:a", "census:c"]
    assert out["verdict"] == "OK"
    assert out["n_rows"] == 6
    # Field discovery pruned the optional names the server does not advertise.
    dropped = out["field_discovery"]["optional_dropped_before_request"]
    assert "A1_sigma" in dropped and "A2_sigma" not in dropped
    # And the census was counted, not carried.
    assert out["census"]["asteroid"]["counts"]["n_with_any"] == 1
    assert "rows" not in out["census"]["asteroid"]


def test_fetch_catalogue_resumes_without_refetching_done_chunks():
    """A cancelled Actions job never runs its commit step, so nothing is re-paid.

    Chunks a previous run already committed are skipped and its rows are carried
    back in, which is the same discipline ``loom-probe`` adopted after a cancelled
    job cost TOCSIN a three-hour backfill.
    """
    sess = FakeSession(lambda p, i: FakeResponse(
        200, {"fields": ["spkid", "A2"], "data": [["z", 1e-12]]}))
    prior = [{"spkid": "earlier", "A2": 5e-13}]
    out = catalogue.fetch_catalogue(
        session=sess, do_census=False, resume_rows=prior,
        done_chunks=["detail:a:A1", "detail:a:A2", "detail:a:A3"])
    for name in ("detail:a:A1", "detail:a:A2", "detail:a:A3"):
        assert out["chunks"][name]["skipped"]
    assert out["n_rows"] == 1 + 3          # the carried row plus the three comet pulls
    assert out["rows"][0]["spkid"] == "earlier"


def test_fetch_catalogue_does_not_rerequest_a_field_the_server_rejected():
    """A field rejected once is rejected always; rediscovering it costs a retry each."""
    rejected: list[str] = []

    def handler(params, i):
        fields = params.get("fields", "").split(",")
        if params.get("info") == "field":
            return FakeResponse(200, {"note": "no field list here"})
        if "DT_sigma" in fields:
            rejected.append(params.get("sb-cdata", "census"))
            return FakeResponse(400, text=json.dumps(
                {"message": "invalid field specified: 'DT_sigma'"}))
        return FakeResponse(200, {"fields": ["spkid"], "data": [["q"]]})

    sess = FakeSession(handler)
    catalogue.fetch_catalogue(session=sess, do_census=False)
    # Rejected exactly once, on the first detail chunk, and never asked for again.
    assert len(rejected) == 1


def test_fetch_covariance_records_absence_rather_than_asserting_independence():
    """``nongrav_vector`` propagates without covariances and must say so.

    The per-object endpoint is the only place the covariance lives and is
    affordable only for the tail, so its presence — or absence — is recorded
    rather than assumed either way.
    """
    sess = FakeSession(lambda p, i: FakeResponse(200, {
        "orbit": {"covariance": {"data": [[1.0]]},
                  "model_pars": [{"name": "A1"}, {"name": "A2"}]}}))
    out = catalogue.fetch_covariance("20875163", session=sess)
    assert out["orbit_covariance_present"] is True
    assert out["model_pars"] == ["A1", "A2"]

    dead = catalogue.fetch_covariance("20875163", session=FakeSession(
        lambda p, i: FakeResponse(404, text="not found")))
    assert dead["status"] == 404
    assert "orbit_covariance_present" not in dead


def test_field_discovery_degrades_to_the_repair_loop():
    """Discovery is an optimisation, never a dependency."""
    sess = FakeSession(lambda p, i: FakeResponse(200, {"data": [["x"]]}))
    out = catalogue.fetch_field_names(session=sess)
    assert out["available"] is None
    assert out["notes"]


# ---------------------------------------------------------------------------
# 9. the CSV, which is what makes the distribution re-derivable offline
# ---------------------------------------------------------------------------
def test_objects_csv_has_one_row_per_screened_object(tmp_path):
    scr = catalogue.screen_catalogue(population(25, seed=67))
    path = tmp_path / "catalogue_objects.csv"
    catalogue._write_objects_csv(path, scr.entries)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 26                       # header plus 25 objects
    assert lines[0].startswith("key,name,pdes,")
    assert "epsilon_1au" in lines[0]
    # NaN must not reach the file as the string "nan": an empty cell is what a
    # reader and a spreadsheet both understand as "not measured".
    assert "nan" not in path.read_text().lower()


# ---------------------------------------------------------------------------
# 10. the orchestrator, with the network stubbed out
# ---------------------------------------------------------------------------
def test_run_catalogue_writes_checkpoints_and_can_resume(tmp_path, monkeypatch):
    """The stage writes after every chunk, and a second run does not re-pay for one.

    A cancelled GitHub Actions job never runs its commit step, so a stage that
    only writes at the end loses everything it learned.  This exercises the whole
    orchestration — fetch, screen, census, completeness, litcheck feed, CSV —
    with the network replaced by a stub, which is the only part of it that cannot
    run in the sandbox.
    """
    rows = population(40, seed=71) + [
        make_row(spkid="20131313", pdes="131313", full_name="131313 (2013 QQ)",
                 A2=a2_for_epsilon(7.0), A2_sigma=a2_for_epsilon(7.0) / 30.0)]
    census_keys = [r["spkid"] for r in rows]

    seen: list[str] = []

    def fake_fetch(**kw):
        for name in ("detail:a:A1", "detail:a:A2", "census:a"):
            if name in set(kw.get("done_chunks") or ()):
                continue
            seen.append(name)
            kw["on_result"](name, {"status": 200, "n_rows": len(rows), "rows": rows})
        return {"chunks": {}, "rows": list(kw.get("resume_rows") or []) + rows,
                "census": {"asteroid": {"counts": {
                    "n_rows": 1553263, "n_with_A1": 22, "n_with_A2": 589,
                    "n_with_A3": 11, "n_with_any": len(census_keys),
                    "keys_with_any": census_keys}}},
                "verdict": "OK", "field_discovery": {"available": None}}

    monkeypatch.setattr(catalogue, "fetch_catalogue", fake_fetch)
    rec = catalogue.run_catalogue(out_dir=tmp_path)

    assert rec["verdict"] == "TAIL_SPARSE_SURVIVORS_PRESENT"
    assert rec["litcheck_input"] == ["131313 (2013 QQ)"]
    assert rec["census"]["total"]["n_rows"] == 1553263
    assert rec["census"]["total"]["fraction_with_any"] == pytest.approx(
        len(census_keys) / 1553263)
    assert rec["completeness"]["verdict"] == "CONSTRAINT_COMPLETE"
    assert (tmp_path / "catalogue.json").exists()
    assert (tmp_path / "catalogue_objects.csv").exists()
    on_disk = json.loads((tmp_path / "catalogue.json").read_text())
    assert on_disk["completed_chunks"] == ["detail:a:A1", "detail:a:A2", "census:a"]
    # The census rows are counted in flight and never retained -- 1.55 million rows
    # in a committed JSON file is not a result, it is a repository problem.
    assert "keys_with_any" not in on_disk["census"]["asteroid"]["counts"]

    # Second run: the chunks the first one committed are not fetched again.
    seen.clear()
    catalogue.run_catalogue(out_dir=tmp_path)
    assert seen == []


def test_run_catalogue_calls_a_dead_fetch_a_dead_fetch(tmp_path, monkeypatch):
    """An unreachable service is never written out as an empty candidate table.

    "JPL did not answer" and "nothing exceeds the ceiling" are different
    statements about the solar system, and only one of them is a result.
    """
    monkeypatch.setattr(catalogue, "fetch_catalogue", lambda **kw: {
        "chunks": {}, "rows": [], "census": {}, "verdict": "NO_DATA_REACHED"})
    rec = catalogue.run_catalogue(out_dir=tmp_path)
    assert rec["verdict"] == "NO_DATA_REACHED"
    assert "DEAD FETCH" in rec["note"]
    assert "tail" not in rec


# ---------------------------------------------------------------------------
# 11. the asteroid/comet split, which decides what a "survivor" is
# ---------------------------------------------------------------------------
def test_tails_are_ranked_within_kind_not_across_kinds():
    """A comet at eps = 1e4 is not "more anomalous" than an asteroid at eps = 3.

    Every comet is above the ceiling because it accelerates by shedding mass, so a
    merged ranking would push every asteroid off a capped list with objects whose
    exceedance is expected of them — and would report the two standing objects as
    ranked several hundredth, which is arithmetic rather than a comparison.
    """
    a2 = a2_for_epsilon(4.0)
    rows = population(20, seed=79) + [
        make_row(spkid="20313131", pdes="313131", full_name="313131 (2031 AA)",
                 A2=a2, A2_sigma=a2 / 30.0),
    ] + [make_row(spkid=f"1000{i:03d}", pdes=f"{i}P", full_name=f"{i}P/Synth",
                  **{"class": "JFc"}, H=None, diameter=2.0, q=2.5,
                  A1=1e-9, A1_sigma=1e-11) for i in range(1, 12)]
    scr = catalogue.screen_catalogue(rows)
    assert scr.n_above_ceiling == {"asteroid": 1, "comet": 11}
    assert [t["name"] for t in scr.tail] == ["313131 (2031 AA)"]
    assert scr.tail[0]["tail_rank"] == 1
    assert len(scr.tail_comets) == 11
    assert scr.tail_comets[0]["tail_rank"] == 1


def test_survivors_never_include_comets():
    """A comet above the radiation ceiling is the ceiling WORKING, not a lead.

    All 81 comets in the 2026-07-30 calibration run were above it.  Feeding those
    to the literature search would bury the objects the search exists for.  Dark
    comets are unaffected: they are classified as asteroids, which is precisely
    why they are the contaminant this channel has to reject by argument rather
    than one it can filter out by kind.
    """
    rows = population(20, seed=83) + [
        make_row(spkid=f"1000{i:03d}", pdes=f"{i}P", full_name=f"{i}P/Synth",
                 **{"class": "JFc"}, H=None, diameter=2.0, q=2.5,
                 A1=1e-9, A1_sigma=1e-11) for i in range(1, 6)]
    scr = catalogue.screen_catalogue(rows)
    assert scr.tail_comets and scr.survivors == []
    assert scr.verdict == "NOTHING_ABOVE_CEILING"


def test_a_measured_diameter_is_a_size_even_without_an_absolute_magnitude():
    """``untestable`` must mean "we do not know how big it is", not "no H column".

    A comet with a measured 2 km diameter has a size and its efficiency was
    computed from that size; reporting it ``untestable`` beside a ratio of 1e4
    would be an internal contradiction.  ``H_effective`` is the H that diameter
    implies, and it is filled in ONLY where H is genuinely absent.
    """
    with_h = catalogue.screen_entry(make_row(H=20.0, diameter=0.383, albedo=0.058,
                                             A2=1e-13, A2_sigma=1e-15))
    assert with_h["H_effective"] == 20.0            # a real H is never overwritten

    no_h = catalogue.screen_entry(make_row(H=None, diameter=2.0, albedo=0.04,
                                           **{"class": "JFc"},
                                           full_name="12P/Synth",
                                           A1=1e-9, A1_sigma=1e-11))
    assert math.isnan(no_h["H"])
    assert no_h["H_effective"] == pytest.approx(
        catalogue.h_equivalent(2000.0, albedo=0.04))
    assert no_h["diameter_m"] == pytest.approx(2000.0)
    rec = catalogue.record_from_entry(no_h, Thresholds())
    assert rec.tier == "interest"
    assert "no_absolute_magnitude" not in rec.reasons


def test_h_equivalent_inverts_the_size_relation():
    d = float(nongrav.diameter_m_from_h(19.0, albedo=0.12))
    assert catalogue.h_equivalent(d, albedo=0.12) == pytest.approx(19.0, abs=1e-9)
    assert math.isnan(catalogue.h_equivalent(0.0))
    assert math.isnan(catalogue.h_equivalent(None))


def test_a_complete_run_does_not_commit_a_megabyte_of_raw_rows(tmp_path, monkeypatch):
    """A run with nothing left to resume carries no resume payload.

    The detail rows are ~1 MB at catalogue scale; committing them on every
    successful run would put a megabyte of duplicated input into git for no
    benefit, because ``catalogue_objects.csv`` already holds the screened
    quantities in a form anyone can re-derive the distribution from.
    """
    rows = population(30, seed=73)
    all_chunks = [f"detail:{k}:{c}" for k in ("a", "c")
                  for c in ("A1", "A2", "A3")] + ["census:a", "census:c"]

    def fake_fetch(**kw):
        for name in all_chunks:
            kw["on_result"](name, {"status": 200, "n_rows": len(rows)})
        return {"chunks": {}, "rows": rows, "census": {}, "verdict": "OK",
                "field_discovery": {"available": None}}

    monkeypatch.setattr(catalogue, "fetch_catalogue", fake_fetch)
    rec = catalogue.run_catalogue(out_dir=tmp_path)
    assert rec["incomplete_chunks"] == []
    assert "resume_rows" not in rec
    assert json.loads((tmp_path / "catalogue.json").read_text()).get("resume_rows") is None


def test_a_partial_run_says_its_fractions_are_over_a_partial_pull(tmp_path, monkeypatch):
    """An unfinished fetch must not be read as a population.

    Every fraction the screen reports is over what was fetched, so a run that
    lost two of its eight chunks has to say so on the record rather than leaving
    a reader to infer a denominator that was never reached.
    """
    rows = population(30, seed=89)

    def fake_fetch(**kw):
        kw["on_result"]("detail:a:A2", {"status": 200, "n_rows": len(rows)})
        return {"chunks": {}, "rows": rows, "census": {}, "verdict": "OK",
                "field_discovery": {"available": None}}

    monkeypatch.setattr(catalogue, "fetch_catalogue", fake_fetch)
    rec = catalogue.run_catalogue(out_dir=tmp_path)
    assert len(rec["incomplete_chunks"]) == 7
    assert rec["resume_rows"]
    assert any("PARTIAL pull" in n for n in rec["notes"])
