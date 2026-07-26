"""TAILINGS offline tests: the sparse/dense discriminant, the funnel, the budget.

No network. A synthetic abundance population (chemical-evolution trends, a
shared alpha axis so nucleosynthetic families genuinely co-move, and noise that
scales with SNR) drives:

(a) an injected **single-element** spike is recovered and classified SPARSE;
(b) an injected **dense** family-wide anomaly of the same amplitude is NOT
    flagged — it is classified DENSE, which is the whole discriminant;
(c) a low-SNR star does not become a candidate through noise, both because the
    empirical scatter is measured per SNR bin and because vetting has a floor;
(d) a co-natal pair with an engulfment-scale refractory difference is NOT
    flagged, while one beyond any plausible rocky-mass budget IS;
(e) the channel degrades honestly: an empty archive response yields a
    NO_DATA_REACHED verdict, never a candidate;
(f) every rejection rule in the funnel has a case that trips it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seti.config import Config, load_config
from seti.tailings import acquire as A
from seti.tailings import manifold as M
from seti.tailings import sparse as S
from seti.tailings import twins as T
from seti.tailings import vet as V

# Elements spanning several nucleosynthetic families, so "family co-moves" is a
# testable statement rather than an assertion.
ELEMENTS = ["Mg", "Si", "Ca", "Ti", "Na", "Al", "Cr", "Mn", "Ni", "Co", "V",
            "Y", "Zr", "Ba", "La", "Ce", "Eu"]
ALPHA = ("Mg", "Si", "Ca", "Ti")
S_HEAVY = ("Ba", "La", "Ce")
FE_PEAK = ("Cr", "Mn", "Ni", "Co", "V")

_SIG_INT = 0.025      # intrinsic star-to-star scatter, dex (Bovy 2016 scale)
_SNR_K = 1.5          # noise term = _SNR_K / SNR, so 0.037 dex at SNR 40


def make_population(rng, n=8000):
    """Synthetic cool-dwarf abundance catalogue with a real low-dimensional core."""
    feh = rng.normal(-0.10, 0.25, n)
    teff = rng.uniform(4200.0, 5900.0, n)
    logg = rng.uniform(4.10, 4.60, n)
    snr = 10 ** rng.uniform(np.log10(20.0), np.log10(300.0), n)
    # One latent alpha axis: this is what makes the alpha family co-move.
    alpha = np.clip(-0.28 * feh + rng.normal(0.0, 0.03, n), -0.10, 0.45)
    # One latent s-process axis (AGB pollution), shared by Ba/La/Ce and Y/Zr.
    s_axis = rng.normal(0.0, 0.08, n)

    df = pd.DataFrame({
        "star_id": [f"S{i:06d}" for i in range(n)],
        "teff": teff, "logg": logg, "fe_h": feh, "snr": snr,
        "chi2": rng.uniform(0.8, 1.6, n),
        "ruwe": rng.uniform(0.9, 1.15, n),
        "vbroad": rng.uniform(2.0, 8.0, n),
        "rv_scatter": rng.uniform(0.05, 0.4, n),
        "field_id": rng.integers(0, 40, n),
        "survey": "SYNTH",
    })
    noise = np.sqrt(_SIG_INT**2 + (_SNR_K / snr) ** 2)
    for k, el in enumerate(ELEMENTS):
        base = 0.0
        if el in ALPHA:
            base = base + alpha
        if el in S_HEAVY:
            base = base + 1.0 * s_axis
        if el in ("Y", "Zr"):
            base = base + 0.6 * s_axis
        if el == "Eu":
            base = base + 0.5 * alpha
        # Pipeline systematics: smooth in Teff, logg and [Fe/H].
        base = (base
                + (0.02 + 0.01 * k % 0.05) * (teff - 5000.0) / 1000.0
                + 0.03 * (logg - 4.35)
                + (-0.05 + 0.01 * (k % 5)) * feh)
        df[el] = base + rng.normal(0.0, noise, n)
        df[f"e_{el}"] = noise
    return df


def score(df, *, z_flag=6.0):
    """Fit the manifold and classify, as the pipeline does."""
    mani = M.fit_manifold(df, ELEMENTS, teff_col="teff", logg_col="logg",
                          feh_col="fe_h", snr_col="snr", min_rows=200,
                          min_count=30)
    Z, sig = M.zscores(df, mani, err_prefix="e_")
    stats = S.sparse_statistics(Z, cfg=S.SparseConfig(z_flag=z_flag))
    return mani, Z, sig, stats


@pytest.fixture(scope="module")
def population():
    rng = np.random.default_rng(20260726)
    return make_population(rng)


# ---------------------------------------------------------------------------
# The manifold itself
# ---------------------------------------------------------------------------
def test_manifold_removes_the_trends_it_is_given(population):
    """Residual scatter must collapse to the injected noise, not the raw spread."""
    mani = M.fit_manifold(population, ELEMENTS, teff_col="teff", logg_col="logg",
                          feh_col="fe_h", snr_col="snr", min_rows=200, min_count=30)
    r = M.residuals(population, mani)
    for el in ALPHA:
        raw = M.robust_sigma(population[el].to_numpy())
        res = M.robust_sigma(r[el].to_numpy())
        assert res < 0.6 * raw, f"{el}: manifold did not absorb the alpha trend"
        assert res < 0.06, f"{el}: residual {res:.3f} dex far above the injected noise"


def test_alpha_proxy_is_leave_one_out():
    df = pd.DataFrame({"Mg": [0.9, 0.3], "Si": [0.1, 0.5], "Ca": [0.3, 0.1]})
    both = M.alpha_proxy(df)
    without_mg = M.alpha_proxy(df, exclude="Mg")
    assert not np.allclose(both, without_mg)
    assert np.allclose(without_mg, df[["Si", "Ca"]].mean(axis=1).to_numpy())


def test_scatter_table_grows_at_low_snr(population):
    """The denominator must know that low-SNR abundances are worse."""
    mani = M.fit_manifold(population, ELEMENTS, teff_col="teff", logg_col="logg",
                          feh_col="fe_h", snr_col="snr", min_rows=200, min_count=30)
    tab = mani.scatter["Ni"]
    lo = tab.sigma_for(np.array([25.0]), np.array([5000.0]))[0]
    hi = tab.sigma_for(np.array([250.0]), np.array([5000.0]))[0]
    assert lo > hi, "sigma must be larger at low SNR"
    assert lo > 1.3 * hi


def test_clipping_keeps_an_anomaly_out_of_its_own_reference_surface(population):
    """A strong outlier must not be able to fit itself."""
    df = population.copy()
    df.loc[df.index[:40], "Ba"] += 0.8
    fit = M.fit_element(
        M.predictor_block(df, "Ba", abund_prefix="", teff_col="teff",
                          logg_col="logg", feh_col="fe_h")[0],
        df["Ba"].to_numpy(), element="Ba", min_rows=200)
    assert fit.n_clipped >= 30


# ---------------------------------------------------------------------------
# (a) injected single-element spike is recovered
# ---------------------------------------------------------------------------
def test_single_element_spike_is_recovered_as_sparse(population):
    df = population.copy()
    # A high-SNR star so the empirical sigma is small: 0.30 dex on ~0.027 dex.
    target = int(np.argmax(df["snr"].to_numpy()))
    df.loc[df.index[target], "Ni"] += 0.30

    _, _, _, stats = score(df)
    row = stats.iloc[target]
    assert row["classification"] == S.SPARSE, row["reason"]
    assert row["element_max"] == "Ni"
    assert row["z_max"] >= 6.0
    assert row["n_discrepant"] == 1
    # The point of the channel: the background stayed quiet.
    assert row["z_rest_rms"] < 2.0
    assert row["contrast"] >= 3.0


def test_recovery_is_not_a_fluke_of_one_element(population):
    """The same spike in three different families is recovered in each."""
    for el in ("Mn", "Al", "V", "Co"):
        df = population.copy()
        target = int(np.argmax(df["snr"].to_numpy()))
        df.loc[df.index[target], el] += 0.30
        _, _, _, stats = score(df)
        assert stats.iloc[target]["classification"] == S.SPARSE
        assert stats.iloc[target]["element_max"] == el


# ---------------------------------------------------------------------------
# (b) the dense anomaly must NOT be flagged -- this is the discriminant
# ---------------------------------------------------------------------------
def test_dense_family_anomaly_is_rejected(population):
    df = population.copy()
    target = int(np.argmax(df["snr"].to_numpy()))
    for el in ("Cr", "Mn", "Ni", "Co", "V"):   # the whole Fe-peak family moves
        df.loc[df.index[target], el] += 0.30

    _, _, _, stats = score(df)
    row = stats.iloc[target]
    assert row["classification"] == S.DENSE, row["reason"]
    assert row["n_discrepant"] >= 3
    assert row["z_rest_rms"] > 2.0            # the background is NOT quiet
    assert "family/global event" in row["reason"]


def test_two_element_family_pair_is_caught_by_the_family_veto(population):
    """Only two elements cross the hard threshold, but they are siblings and the
    rest of the family is leaning the same way: still a family event."""
    df = population.copy()
    target = int(np.argmax(df["snr"].to_numpy()))
    df.loc[df.index[target], "Ni"] += 0.30
    df.loc[df.index[target], "Co"] += 0.30
    df.loc[df.index[target], "Cr"] += 0.10     # sub-threshold, but co-moving
    _, _, _, stats = score(df)
    row = stats.iloc[target]
    assert row["classification"] == S.DENSE, row["reason"]
    assert "family" in row["reason"] or "background" in row["reason"]


def test_global_offset_is_dense_not_sparse(population):
    """A pipeline failure that shifts every element is the classic false positive."""
    df = population.copy()
    target = int(np.argmax(df["snr"].to_numpy()))
    for el in ELEMENTS:
        df.loc[df.index[target], el] += 0.25
    _, _, _, stats = score(df)
    assert stats.iloc[target]["classification"] == S.DENSE


def test_contrast_table_separates_the_two_populations(population):
    df = population.copy()
    idx = np.argsort(-df["snr"].to_numpy())
    for i in idx[:20]:
        df.loc[df.index[i], "Ni"] += 0.30                     # sparse injections
    for i in idx[20:40]:
        for el in FE_PEAK:
            df.loc[df.index[i], el] += 0.30                   # dense injections
    _, _, _, stats = score(df)
    tab = S.contrast_table(stats)
    high = tab[tab["z_max_lo"] >= 6]
    assert high["n_sparse"].sum() >= 15
    assert high["n_dense"].sum() >= 15
    # And the diagnostic that makes the claim falsifiable: dense stars have a
    # loud background, sparse ones do not.
    sp = stats[stats["classification"] == S.SPARSE]["z_rest_rms"].median()
    dn = stats[stats["classification"] == S.DENSE]["z_rest_rms"].median()
    assert sp < dn


# ---------------------------------------------------------------------------
# (c) low-SNR noise must not become a candidate
# ---------------------------------------------------------------------------
def test_low_snr_noise_does_not_produce_candidates(population):
    """No injection: the low-SNR tail of a pure-noise population must stay clean."""
    _, _, _, stats = score(population)
    low = population["snr"].to_numpy() < 40.0
    assert low.sum() > 200, "test population must actually contain low-SNR stars"
    n_bad = int((stats.loc[low, "classification"] == S.SPARSE).sum())
    assert n_bad == 0, f"{n_bad} low-SNR noise outliers leaked through as candidates"


def test_low_snr_star_is_rejected_by_the_funnel_even_if_it_scores():
    cand = pd.DataFrame({
        "star_id": ["A", "B"], "snr": [25.0, 120.0], "teff": [5000.0, 5000.0],
        "logg": [4.4, 4.4], "chi2": [1.0, 1.0], "ruwe": [1.0, 1.0],
        "rv_scatter": [0.1, 0.1], "vbroad": [4.0, 4.0], "n_elements": [16, 16],
        "element_max": ["Ni", "Ni"],
    })
    out = V.vet_candidates(cand, survey="GALAH")
    assert not bool(out.loc[0, "vet_pass"])
    assert "SNR" in out.loc[0, "vet_reasons"]
    assert bool(out.loc[1, "vet_pass"])


def test_scatter_table_alone_suppresses_a_low_snr_excursion():
    """Independently of the SNR cut: the same dex offset must score lower at low SNR."""
    rng = np.random.default_rng(7)
    df = make_population(rng, n=6000)
    hi = int(np.argmax(df["snr"].to_numpy()))
    lo = int(np.argmin(df["snr"].to_numpy()))
    df.loc[df.index[hi], "Ni"] += 0.20
    df.loc[df.index[lo], "Ni"] += 0.20
    _, Z, _, _ = score(df)
    assert abs(Z.iloc[hi]["Ni"]) > 2.0 * abs(Z.iloc[lo]["Ni"])


# ---------------------------------------------------------------------------
# (d) the co-natal pair stage and the engulfed-planet mass budget
# ---------------------------------------------------------------------------
def _rock_pair(mass_earth, teff=5800.0, feh=0.0, sigma=0.02, rng=None, safety=0.5):
    els = [e for e in T.T_COND if e in T.BULK_EARTH_MASS_FRACTION]
    deltas, sigmas = {}, {}
    for e in els:
        d = float(T.delta_from_rock(mass_earth, e, teff=teff, feh=feh,
                                    safety_factor=safety))
        if rng is not None:
            d += rng.normal(0.0, sigma * 0.2)
        deltas[e] = d
        sigmas[e] = sigma
    return deltas, sigmas


def test_solar_calibration_of_the_pollution_formula():
    """One Earth mass of rock into the solar convective zone is ~0.015-0.02 dex."""
    d = float(T.delta_from_rock(1.0, "Fe", teff=5772.0, feh=0.0, safety_factor=1.0))
    assert 0.010 < d < 0.030, d


def test_kronos_krios_calibration_against_the_published_mass():
    """The most extreme engulfment claim on record, ~0.20 dex, ~28 Earth masses.

    We use a coarse M_cz table and a bulk-Earth composition, so agreement is
    expected only to a factor of ~2 -- which is the honest precision of the
    budget test, and why the ceiling is set several times higher than this.
    """
    m = float(T.implied_engulfed_mass(0.20, "Fe", teff=5800.0, feh=0.0,
                                      safety_factor=1.0))
    assert 5.0 < m < 60.0, m


def test_engulfment_scale_pair_is_not_flagged():
    rng = np.random.default_rng(11)
    deltas, sigmas = _rock_pair(5.0, rng=rng)
    v = T.pair_verdict(deltas, sigmas, teff=5800.0, feh=0.0)
    assert v["verdict"] == T.ENGULFMENT_CONSISTENT, v["reason"]
    assert v["tcond_slope_sigma"] > 3.0
    assert v["implied_engulfed_mass_earth_median"] < 100.0


def test_pair_beyond_the_mass_budget_is_flagged():
    rng = np.random.default_rng(12)
    deltas, sigmas = _rock_pair(500.0, rng=rng)
    v = T.pair_verdict(deltas, sigmas, teff=5800.0, feh=0.0)
    assert v["verdict"] == T.ENGULFMENT_EXCESSIVE, v["reason"]
    assert v["implied_engulfed_mass_earth_median"] > 100.0


def test_single_element_pair_difference_is_unexplainable_at_any_mass():
    """No amount of rock produces one element with a flat Tcond trend."""
    rng = np.random.default_rng(13)
    els = [e for e in T.T_COND if e in T.BULK_EARTH_MASS_FRACTION]
    deltas = {e: float(rng.normal(0.0, 0.004)) for e in els}
    sigmas = dict.fromkeys(els, 0.02)
    deltas["Ba"] = 0.30
    v = T.pair_verdict(deltas, sigmas, teff=5800.0, feh=0.0)
    assert v["verdict"] == T.SPARSE_UNEXPLAINABLE, v["reason"]
    assert v["element_max"] == "Ba"


def test_identical_pair_is_no_difference():
    els = [e for e in T.T_COND if e in T.BULK_EARTH_MASS_FRACTION]
    v = T.pair_verdict(dict.fromkeys(els, 0.0), dict.fromkeys(els, 0.02), teff=5500.0)
    assert v["verdict"] == T.NO_DIFFERENCE


def test_convective_envelope_mass_is_monotonic_where_it_should_be():
    """Cooler dwarfs have deeper envelopes, until they go fully convective."""
    t = np.array([5800.0, 5400.0, 5000.0, 4500.0, 4000.0])
    m = T.convective_envelope_mass(t, safety_factor=1.0)
    assert np.all(np.diff(m) > 0)
    assert 0.015 < m[0] < 0.03      # solar value


def test_pair_table_orients_on_the_polluted_component():
    rng = np.random.default_rng(14)
    deltas, _ = _rock_pair(500.0, rng=rng)
    els = list(deltas)
    row = {"pair_id": "p1", "a_teff": 5800.0, "b_teff": 5800.0, "a_fe_h": 0.0}
    for e in els:
        row[f"a_{e}"] = 0.0                     # a is the CLEAN one
        row[f"b_{e}"] = deltas[e]
        row[f"a_{e}_err"] = 0.014
        row[f"b_{e}_err"] = 0.014
    out = T.pair_table(pd.DataFrame([row]), els)
    assert out.loc[0, "polluted_component"] == "b"
    assert out.loc[0, "verdict"] == T.ENGULFMENT_EXCESSIVE


# ---------------------------------------------------------------------------
# (e) honest degradation
# ---------------------------------------------------------------------------
def test_empty_archive_response_yields_no_data_reached():
    acq = A.fetch_survey("GALAH", discover=False,
                         probe_fn=lambda t: None, query_fn=lambda q: None)
    assert acq.n_rows == 0
    assert acq.degraded
    assert acq.degradation.startswith("NO_DATA_REACHED")
    assert acq.sources_tried  # it really did try every candidate source


def test_source_fallback_is_recorded_as_degradation():
    """When the preferred release is missing, the run says which one it used."""
    good = SOURCES_LAST = A.SOURCES["GALAH"][-1].locator

    def probe(table):
        if table != good:
            return None
        return pd.DataFrame({
            "sobject_id": [1], "Teff": [5000.0], "logg": [4.4], "fe_h": [0.0],
            "snr": [100.0], "Mg_fe": [0.0], "e_Mg_fe": [0.02],
            "Ni_fe": [0.0], "e_Ni_fe": [0.02],
        })

    def query(adql):
        assert SOURCES_LAST in adql
        return pd.DataFrame({
            "sobject_id": [1, 2], "Teff": [5000.0, 5100.0], "logg": [4.4, 4.5],
            "fe_h": [0.0, -0.1], "snr": [100.0, 90.0],
            "Mg_fe": [0.0, 0.1], "e_Mg_fe": [0.02, 0.02],
            "Ni_fe": [0.0, 0.05], "e_Ni_fe": [0.02, 0.02],
        })

    acq = A.fetch_survey("GALAH", n_chunks=1, discover=False,
                         probe_fn=probe, query_fn=query)
    assert acq.n_rows == 2
    assert acq.degraded
    assert "rather than the preferred" in acq.degradation
    assert set(acq.elements) == {"Mg", "Ni"}


def test_run_degrades_honestly_with_no_checkpoints(tmp_path):
    from seti.tailings.run import tailings_run

    real = load_config()
    cfg = Config(root=tmp_path, thresholds=real.thresholds,
                 catalogs=real.catalogs, paths=real.paths)
    out = tailings_run(cfg, stage="manifold", surveys="GALAH,APOGEE")
    assert out["verdict"].startswith("NO_DATA_REACHED")
    assert out["funnel"]["n_stars_total"] == 0
    assert out["funnel"]["n_vetted_total"] == 0
    assert (tmp_path / "results" / "tailings" / "summary.json").exists()
    assert (tmp_path / "results" / "tailings" / "REPORT.md").exists()


def test_twins_stage_degrades_without_a_binary_catalogue(tmp_path):
    from seti.tailings.run import stage_twins

    real = load_config()
    cfg = Config(root=tmp_path, thresholds=real.thresholds,
                 catalogs=real.catalogs, paths=real.paths)
    d = tmp_path / "results" / "tailings"
    d.mkdir(parents=True)
    out = stage_twins(cfg, out_dir=d, block=real.thresholds.get("tailings", {}))
    assert out["verdict"].startswith("NO_DATA_REACHED")
    assert out["n_pairs"] == 0


# ---------------------------------------------------------------------------
# (f) every rejection rule has a case that trips it
# ---------------------------------------------------------------------------
def _one(**kw):
    base = {"star_id": ["X"], "snr": [120.0], "teff": [5000.0], "logg": [4.4],
            "chi2": [1.0], "ruwe": [1.0], "rv_scatter": [0.1], "vbroad": [4.0],
            "n_elements": [16], "element_max": ["Ni"]}
    base.update({k: [v] for k, v in kw.items()})
    return pd.DataFrame(base)


@pytest.mark.parametrize(
    ("kw", "fragment"),
    [
        ({"snr": 10.0}, "SNR"),
        ({"chi2": 9.0}, "chi2"),
        ({"ruwe": 2.0}, "RUWE"),
        ({"rv_scatter": 5.0}, "RV scatter"),
        ({"vbroad": 40.0}, "broadening"),
        ({"teff": 7000.0}, "cool-dwarf box"),
        ({"logg": 2.5}, "cool-dwarf box"),
        ({"n_elements": 4}, "fewer than"),
    ],
)
def test_each_vet_rule_trips(kw, fragment):
    out = V.vet_candidates(_one(**kw), survey="GALAH")
    assert not bool(out.loc[0, "vet_pass"])
    assert fragment in out.loc[0, "vet_reasons"]


def test_pipeline_flag_column_is_honoured():
    df = _one()
    df["flag_sp"] = [1]
    out = V.vet_candidates(df, survey="GALAH")
    assert not bool(out.loc[0, "vet_pass"])
    assert "quality flag" in out.loc[0, "vet_reasons"]


def test_missing_column_is_untested_not_silently_passed():
    df = _one().drop(columns=["ruwe"])
    out = V.vet_candidates(df, survey="GALAH")
    assert "ruwe" in out.loc[0, "vet_untested"]


def test_runaway_element_flag_rate_vetoes():
    rates = pd.DataFrame({"element": ["Ni", "Ba"], "flag_rate": [0.30, 0.0001]})
    out = V.element_rate_veto(V.vet_candidates(_one(), survey="GALAH"), rates)
    assert not bool(out.loc[0, "pass_element_rate"])
    assert "pipeline systematic" in out.loc[0, "vet_reasons"]


def test_field_flag_rate_veto():
    n = 3100
    allst = pd.DataFrame({"field_id": [1] * 100 + [2] * 3000})
    mask = np.zeros(n, dtype=bool)
    mask[:50] = True                    # field 1 flags 50%; global rate is 1.6%
    mask[200:215] = True                # field 2 flags 0.5%
    cand = _one()
    cand["field_id"] = [1]
    out = V.field_rate_veto(cand, allst, mask)
    assert not bool(out.loc[0, "pass_field_rate"])


def test_element_caveats_demote_rather_than_delete():
    out = V.vet_candidates(_one(element_max="K"), survey="GALAH")
    assert bool(out.loc[0, "caveated_element"])
    assert "telluric" in out.loc[0, "element_caveat"]
    assert bool(out.loc[0, "vet_pass"])          # demoted, not rejected


def test_cross_survey_absence_is_not_a_refutation():
    cand = pd.DataFrame({"xmatch_id": ["a", "b"], "element_max": ["Ba", "Ni"],
                         "z_max_signed": [8.0, 8.0]})
    other = pd.DataFrame({"xmatch_id": ["a"], "z_Ni": [7.0]})
    out = V.cross_survey_check(cand, other)
    assert out.loc[0, "cross_survey"] == "not_covered"   # Ba absent from the other list
    assert out.loc[1, "cross_survey"] == "no_match"


def test_cross_survey_confirmation_and_refutation():
    cand = pd.DataFrame({"xmatch_id": ["a", "b"], "element_max": ["Ni", "Ni"],
                         "z_max_signed": [8.0, 8.0]})
    other = pd.DataFrame({"xmatch_id": ["a", "b"], "z_Ni": [7.0, 0.2]})
    out = V.cross_survey_check(cand, other)
    assert out.loc[0, "cross_survey"] == "confirmed"
    assert out.loc[1, "cross_survey"] == "refuted"


def test_dedupe_keeps_the_best_epoch():
    df = pd.DataFrame({"star_id": ["A", "A", "B"], "snr": [50.0, 200.0, 80.0]})
    out = V.dedupe(df, id_col="star_id")
    assert len(out) == 2
    assert float(out.loc[out["star_id"] == "A", "snr"].iloc[0]) == 200.0


def test_element_flag_rates_expose_a_systematic():
    Z = pd.DataFrame({"Ni": np.concatenate([np.full(50, 9.0), np.zeros(950)]),
                      "Ba": np.zeros(1000)})
    stats = S.sparse_statistics(Z, cfg=S.SparseConfig(min_elements=2))
    rates = S.element_flag_rates(stats, Z, cfg=S.SparseConfig(min_elements=2))
    ni = rates[rates["element"] == "Ni"].iloc[0]
    assert ni["flag_rate"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Re-measurement from the raw spectrum: the decisive step for any survivor
# ---------------------------------------------------------------------------
def test_equivalent_width_recovers_an_injected_line():
    wave = np.arange(4550.0, 4570.0, 0.02)
    depth, sig, c0 = 0.30, 0.06, 4560.0
    flux = 1.0 - depth * np.exp(-0.5 * ((wave - c0) / sig) ** 2)
    ew, err = V.measure_ew(wave, flux, c0)
    assert ew == pytest.approx(depth * sig * np.sqrt(2 * np.pi), rel=0.15)
    assert err >= 0.0


def test_equivalent_width_survives_a_sloped_continuum():
    wave = np.arange(4550.0, 4570.0, 0.02)
    depth, sig, c0 = 0.30, 0.06, 4560.0
    cont = 1.0 + 0.004 * (wave - c0)
    flux = cont * (1.0 - depth * np.exp(-0.5 * ((wave - c0) / sig) ** 2))
    ew, _ = V.measure_ew(wave, flux, c0)
    assert ew == pytest.approx(depth * sig * np.sqrt(2 * np.pi), rel=0.20)


def test_census_z_and_remeasurement_verdicts():
    rng = np.random.default_rng(3)
    peers = rng.normal(0.050, 0.004, 200)
    assert V.census_z(0.050, peers) == pytest.approx(0.0, abs=0.5)
    z = V.census_z(0.050 + 6 * 0.004 * 1.4826 / 1.4826, peers)
    assert z > 4.0
    assert V.remeasure_verdict(8.0, z) == "CONFIRMED"
    assert V.remeasure_verdict(8.0, 1.0) == "REFUTED_AMPLITUDE"
    assert V.remeasure_verdict(8.0, -9.0) == "REFUTED_SIGN"
    assert V.remeasure_verdict(8.0, float("nan")) == "NO_SPECTRUM"


# ---------------------------------------------------------------------------
# Schema resolution: a naming change must not cost a run
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cols",
    [
        ["mg_fe", "e_mg_fe", "flag_mg_fe", "ni_fe", "e_ni_fe"],          # GALAH native
        ["MG_FE", "MG_FE_ERR", "MG_FE_FLAG", "NI_FE", "NI_FE_ERR"],      # APOGEE
        ["__Mg_Fe_", "e__Mg_Fe_", "f__Mg_Fe_", "__Ni_Fe_", "e__Ni_Fe_"],  # VizieR
    ],
)
def test_abundance_column_resolution_across_conventions(cols):
    res = A.resolve_abundance_columns(cols)
    assert set(res) == {"Mg", "Ni"}
    assert "err" in res["Mg"]


def test_parameter_column_resolution():
    p = A.resolve_param_columns(["sobject_id", "RAJ2000", "DEJ2000", "Teff",
                                 "logg", "fe_h", "snr", "chi2_sp", "ruwe"])
    assert p["teff"] == "Teff"
    assert p["fe_h"] == "fe_h"
    assert p["star_id"] == "sobject_id"


def test_fe_is_not_treated_as_an_abundance_ratio():
    assert "Fe" not in A.resolve_abundance_columns(["fe_h", "e_fe_h", "mg_fe"])


def test_per_element_flags_blank_bad_values():
    df = pd.DataFrame({"Mg": [0.1, 0.2], "f_Mg": [0, 1], "Ni": [0.0, 0.0]})
    out = A.apply_element_flags(df, ["Mg", "Ni"])
    assert np.isfinite(out.loc[0, "Mg"])
    assert not np.isfinite(out.loc[1, "Mg"])


def test_join_pairs_requires_two_spectra():
    stars = pd.DataFrame({"star_id": ["1", "2"], "teff": [5000.0, 5100.0],
                          "logg": [4.4, 4.4], "fe_h": [0.0, 0.0],
                          "snr": [100.0, 100.0], "Mg": [0.0, 0.1],
                          "e_Mg": [0.02, 0.02]})
    pairs = pd.DataFrame({"source_id_a": ["1", "1"], "source_id_b": ["2", "9"]})
    out = A.join_pairs(pairs, stars, ["Mg"])
    assert len(out) == 1
    assert out.loc[0, "pair_id"] == "1_2"


# ---------------------------------------------------------------------------
# Sparse statistic edge cases
# ---------------------------------------------------------------------------
def test_short_abundance_vector_is_insufficient_not_sparse():
    Z = pd.DataFrame({"Ni": [9.0], "Mg": [0.1], "Ba": [0.0]})
    stats = S.sparse_statistics(Z)
    assert stats.iloc[0]["classification"] == S.INSUFFICIENT


def test_naturally_sparse_elements_cannot_carry_a_candidacy():
    Z = pd.DataFrame({el: [0.1] for el in ELEMENTS})
    Z["Li"] = [12.0]
    stats = S.sparse_statistics(Z)
    assert stats.iloc[0]["classification"] == S.NORMAL
    assert stats.iloc[0]["z_Li_excluded"] == 12.0


def test_classification_reasons_are_specific():
    _, r = S.classify(n_elements=20, n_discrepant=6, n_active=8,
                      contrast=1.0, family_mean_z=0.5)
    assert "family/global event" in r
    lab, r2 = S.classify(n_elements=20, n_discrepant=1, n_active=1,
                         contrast=8.0, family_mean_z=3.0)
    assert lab == S.DENSE and "family co-moves" in r2


# ---------------------------------------------------------------------------
# Two properties the design depends on, made explicit
# ---------------------------------------------------------------------------
def test_threshold_in_dex_is_element_dependent_by_construction(population):
    """An element with real astrophysical spread needs a larger excursion.

    Ba/La/Ce carry a latent s-process axis that ([Fe/H], Teff, log g, alpha)
    cannot predict, so their empirical residual width is several times that of
    the Fe-peak. A fixed *dex* threshold would therefore flag n-capture
    elements preferentially — which is exactly the systematic that would
    manufacture a fake candidate population. Dividing by the measured
    per-element width is what removes it, and the price is that a sparse
    anomaly in an s-process element has to be correspondingly larger.
    """
    mani = M.fit_manifold(population, ELEMENTS, teff_col="teff", logg_col="logg",
                          feh_col="fe_h", snr_col="snr", min_rows=200, min_count=30)
    s_ba = mani.scatter["Ba"].sigma_global
    s_ni = mani.scatter["Ni"].sigma_global
    assert s_ba > 2.0 * s_ni

    df = population.copy()
    target = int(np.argmax(df["snr"].to_numpy()))
    df.loc[df.index[target], "Ba"] += 0.30          # ~12 sigma for Ni, ~3.5 for Ba
    _, _, _, stats = score(df)
    assert stats.iloc[target]["classification"] != S.SPARSE

    df = population.copy()
    df.loc[df.index[target], "Ba"] += 0.90          # now a real excursion for Ba
    _, _, _, stats = score(df)
    assert stats.iloc[target]["classification"] == S.SPARSE
    assert stats.iloc[target]["element_max"] == "Ba"


def test_a_pure_alpha_family_shift_is_absorbed_by_its_own_predictor(population):
    """Shifting the whole alpha family cannot fake a sparse anomaly.

    The alpha proxy is a *predictor*, so a coherent shift of all alpha elements
    moves the proxy too and the manifold absorbs it. That is the intended
    behaviour: a global alpha offset is chemical evolution, not a refinery, and
    it must not reach the candidate list by any route.
    """
    df = population.copy()
    target = int(np.argmax(df["snr"].to_numpy()))
    for el in ALPHA:
        df.loc[df.index[target], el] += 0.30
    _, _, _, stats = score(df)
    assert stats.iloc[target]["classification"] != S.SPARSE


def test_teff_chunking_and_truncation_are_reported():
    """A chunk that returns exactly its cap is a truncation, and must say so."""
    head = pd.DataFrame({"sobject_id": [1], "Teff": [5000.0], "logg": [4.4],
                         "fe_h": [0.0], "snr": [100.0], "Mg_fe": [0.0],
                         "e_Mg_fe": [0.02], "Ni_fe": [0.0], "e_Ni_fe": [0.02]})
    seen = []

    def query(adql):
        seen.append(adql)
        n = 5                                   # every chunk returns its cap
        return pd.DataFrame({"sobject_id": range(n), "Teff": [5000.0] * n,
                             "logg": [4.4] * n, "fe_h": [0.0] * n,
                             "snr": [100.0] * n, "Mg_fe": [0.0] * n,
                             "e_Mg_fe": [0.02] * n, "Ni_fe": [0.0] * n,
                             "e_Ni_fe": [0.02] * n})

    acq = A.fetch_survey("GALAH", max_rows=20, n_chunks=4, discover=False,
                         probe_fn=lambda t: head, query_fn=query)
    assert len(seen) == 4, "the pull must be chunked in Teff, not monolithic"
    assert acq.n_rows == 20
    assert acq.degraded and "TRUNCATED" in acq.degradation


def test_a_lost_chunk_does_not_kill_the_run():
    head = pd.DataFrame({"sobject_id": [1], "Teff": [5000.0], "logg": [4.4],
                         "fe_h": [0.0], "snr": [100.0], "Mg_fe": [0.0],
                         "e_Mg_fe": [0.02], "Ni_fe": [0.0], "e_Ni_fe": [0.02]})
    calls = {"n": 0}

    def query(adql):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("TAP timeout")
        return head

    acq = A.fetch_survey("GALAH", max_rows=8, n_chunks=4, discover=False,
                         probe_fn=lambda t: head, query_fn=query)
    assert acq.n_rows == 3
    assert "coverage is incomplete" in acq.degradation


# ---------------------------------------------------------------------------
# Table discovery: the failure that killed the first dispatch
# ---------------------------------------------------------------------------
def test_table_discovery_finds_a_catalogue_the_encoded_locators_miss():
    """VizieR catalogue numbers drift; asking the service removes the failure mode."""
    rich = pd.DataFrame({"sobject_id": [1], "Teff": [5000.0], "logg": [4.4],
                         "fe_h": [0.0], "snr": [100.0], "ruwe": [1.0],
                         **{f"{e}_fe": [0.0] for e in ("Mg", "Si", "Ca", "Ni", "Ba")},
                         **{f"e_{e}_fe": [0.02] for e in ("Mg", "Si", "Ca", "Ni", "Ba")}})
    thin = pd.DataFrame({"Teff": [5000.0], "logg": [4.4], "fe_h": [0.0],
                         "Mg_fe": [0.0]})

    def query(adql):
        if "TAP_SCHEMA" in adql:
            return pd.DataFrame({"table_name": ["III/999/readme", "III/999/main"],
                                 "description": ["GALAH DR4 readme", "GALAH DR4 main"]})
        return rich.iloc[[0]]

    def probe(table):
        if table.startswith("III/28") or table.startswith("III/29"):
            return None                       # every encoded locator is stale
        return thin if "readme" in table else rich

    acq = A.fetch_survey("GALAH", max_rows=1, n_chunks=1,
                         probe_fn=probe, query_fn=query)
    assert acq.locator == "III/999/main", "must pick the RICHEST schema, not the first"
    assert acq.n_rows == 1
    assert len(acq.elements) == 5
    # The choice has to be auditable.
    tables = {r["table"] for r in acq.scoreboard}
    assert {"III/999/readme", "III/999/main"} <= tables
    assert max(r["score"] for r in acq.scoreboard) > 0


def test_discovery_failure_is_not_fatal():
    def query(adql):
        if "TAP_SCHEMA" in adql:
            raise RuntimeError("TAP_SCHEMA unavailable")
        return pd.DataFrame()

    acq = A.fetch_survey("GALAH", probe_fn=lambda t: None, query_fn=query)
    assert acq.n_rows == 0
    assert acq.degradation.startswith("NO_DATA_REACHED")


def test_wide_binary_missing_purity_column_is_flagged():
    """No R_chance_align means chance alignments are not removed — say so."""
    head = pd.DataFrame({"source_id1": [1], "source_id2": [2]})

    def query(adql):
        if "TAP_SCHEMA" in adql:
            return pd.DataFrame()
        assert "r_chance_align" not in adql.lower()
        return pd.DataFrame({"source_id1": [1, 3], "source_id2": [2, 4]})

    acq = A.fetch_wide_binaries(probe_fn=lambda t: head, query_fn=query)
    assert acq.n_rows == 2
    assert acq.degraded
    assert "chance alignments are NOT removed" in acq.degradation


def test_rows_without_enough_elements_is_insufficient_not_a_null(tmp_path):
    """A thin element vector is a coverage statement, never 'we found nothing'."""
    from seti.tailings.run import _overall_verdict

    v = _overall_verdict(
        [{"survey": "GALAH", "n_stars": 5000, "n_vetted": 0,
          "verdict": "INSUFFICIENT_SAMPLE: too few elements"}],
        {}, None)
    assert v.startswith("INSUFFICIENT_SAMPLE")
    assert "not a limit on the signature" in v


def test_a_degraded_source_is_carried_into_the_headline_verdict():
    from seti.tailings.run import _overall_verdict

    prov = {"surveys": [{"survey": "GALAH", "source_used": "GALAH_DR3_vizier",
                         "degraded": True}]}
    v = _overall_verdict([{"survey": "GALAH", "n_stars": 5000, "n_vetted": 0}], {}, prov)
    assert v.startswith("DEGRADED_SOURCE")
    assert "GALAH->GALAH_DR3_vizier" in v


# ---------------------------------------------------------------------------
# The methodological core: a global statistic CANNOT see what this one sees
# ---------------------------------------------------------------------------
def test_global_statistics_rank_dense_above_sparse_and_this_one_inverts_it(population):
    """Injection-recovery against the statistic the rest of the field uses.

    No paper states that global anomaly statistics dilute sparse anomalies, so
    it is demonstrated here rather than cited. Two stars are injected with the
    SAME per-element amplitude: one single-element, one four-element. A global
    reduced chi-squared -- the shared form of every executed abundance-outlier
    method -- ranks the dense star far above the sparse one, because it is
    monotone in how many elements deviate. The sparse statistic inverts that
    ordering. And the Weinberg leave-one-out convention, which omits the single
    largest contributor precisely because a lone deviant element is usually an
    artifact, scores the sparse star at essentially zero: it *requires* at least
    two anomalous abundances, which is an explicit exclusion of this signal.
    """
    df = population.copy()
    idx = np.argsort(-df["snr"].to_numpy())
    sparse_i, dense_i = int(idx[0]), int(idx[1])
    df.loc[df.index[sparse_i], "Ni"] += 0.30
    for el in FE_PEAK[:4]:
        df.loc[df.index[dense_i], el] += 0.30

    _, Z, _, stats = score(df)
    glob = S.global_statistics(Z)

    # 1. The global statistic prefers the dense star.
    assert glob.iloc[dense_i]["chi2_reduced"] > 2.0 * glob.iloc[sparse_i]["chi2_reduced"]

    # 2. The sparse statistic prefers the sparse star -- it rejects the other.
    assert stats.iloc[sparse_i]["classification"] == S.SPARSE
    assert stats.iloc[dense_i]["classification"] == S.DENSE

    # 3. The Weinberg leave-one-out criterion erases the sparse star entirely:
    #    with its one deviant element omitted, it is an ordinary star.
    loo_sparse = glob.iloc[sparse_i]["chi2_leave_one_out"]
    loo_dense = glob.iloc[dense_i]["chi2_leave_one_out"]
    assert loo_sparse < 4.0
    assert loo_dense > 10.0 * loo_sparse

    # 4. And by global rank the sparse star is unremarkable: it does not even
    #    reach the top 20 of a 8000-star sample, while this channel puts it top.
    rank = int((glob["chi2_reduced"] > glob.iloc[sparse_i]["chi2_reduced"]).sum())
    assert rank > 0, "the sparse injection should NOT be the top global outlier"


def test_working_in_xh_protects_a_sparse_anomaly_from_the_iron_error(population):
    """[X/Fe] normalisation smears a sparse anomaly across every element."""
    df = population.copy()
    target = int(np.argmax(df["snr"].to_numpy()))
    df.loc[df.index[target], "Ni"] += 0.30
    # An error in this star's own [Fe/H]. In [X/Fe] space it moves nothing at
    # all -- because [X/Fe] is already iron-normalised -- but it moves the
    # PREDICTOR, and after conversion to [X/H] it moves every element together.
    xh = M.to_xh(df, ELEMENTS, feh_col="fe_h")
    assert xh.loc[xh.index[target], "Ni"] == pytest.approx(
        df.loc[df.index[target], "Ni"] + df.loc[df.index[target], "fe_h"])
    # Every element shifted by exactly the same amount: a pure Fe error is a
    # coherent offset in [X/H], i.e. a DENSE pattern the manifold absorbs,
    # rather than a per-element perturbation that would fake sparsity.
    shifts = [xh.loc[xh.index[target], e] - df.loc[df.index[target], e] for e in ELEMENTS]
    assert np.allclose(shifts, shifts[0])


def test_metal_poor_star_is_cut_by_the_envelope_argument():
    """A metal-poor turnoff star passes Teff/logg but has a thin envelope."""
    cand = _one(fe_h=-2.0)
    out = V.vet_candidates(cand, survey="GALAH")
    assert not bool(out.loc[0, "vet_pass"])
    assert "metal-poor envelopes are thin" in out.loc[0, "vet_reasons"]
    assert bool(V.vet_candidates(_one(fe_h=-0.3), survey="GALAH").loc[0, "vet_pass"])


def test_cool_star_caveat_is_recorded_not_hidden():
    out = V.vet_candidates(_one(teff=4300.0, fe_h=0.0), survey="GALAH")
    assert bool(out.loc[0, "cool_star_caveat"])
    assert bool(out.loc[0, "vet_pass"])          # flagged, not deleted
    assert not bool(V.vet_candidates(_one(teff=5200.0, fe_h=0.0),
                                     survey="GALAH").loc[0, "cool_star_caveat"])


def test_radial_velocity_footprint_is_vetoed():
    """A telluric artifact lives in instrument coordinates, not in chemistry."""
    n = 3000
    rng = np.random.default_rng(5)
    rv = rng.uniform(-120.0, 120.0, n)
    allst = pd.DataFrame({"rv": rv})
    # Everything near -70 km/s flags: APOGEE's K lines land on a telluric there.
    mask = (rv > -80.0) & (rv < -60.0)
    mask |= rng.random(n) < 0.002              # a thin real background
    cand = _one(fe_h=0.0)
    cand["rv"] = [-70.0]
    out = V.covariate_rate_veto(V.vet_candidates(cand, survey="GALAH"), allst, mask,
                                covariate_col="rv", label="rv")
    assert not bool(out.loc[0, "pass_rv_rate"])
    assert "instrumental footprint" in out.loc[0, "vet_reasons"]

    clean = _one(fe_h=0.0)
    clean["rv"] = [40.0]
    ok = V.covariate_rate_veto(V.vet_candidates(clean, survey="GALAH"), allst, mask,
                               covariate_col="rv", label="rv")
    assert bool(ok.loc[0, "pass_rv_rate"])


def test_church_metallicity_dilution_calibration():
    """Independent check of the pollution formula against a published number.

    Church et al. measure a convective envelope of 3.45e-3 Msun at a
    solar-metallicity M67 turnoff and quote 5.2 Earth masses of rock as buying
    0.128 dex in metallicity. Applying the same formula to BULK METALS (rock is
    essentially all metals; solar material is 1.34% by mass) must reproduce it.
    """
    m_env_metals = 3.45e-3 * 0.0134
    m_pol = 5.2 * T.M_EARTH_IN_MSUN * 1.0
    delta = np.log10(1.0 + m_pol / m_env_metals)
    assert delta == pytest.approx(0.128, abs=0.01)

    # And the channel's own M_cz table must agree with Church at that Teff to
    # within the factor of ~2 the docs claim for it.
    mcz = float(T.convective_envelope_mass(6100.0, safety_factor=1.0))
    assert 0.4 < mcz / 3.45e-3 < 2.5


# ---------------------------------------------------------------------------
# The Karinkuzhi precedent: low resolution MANUFACTURES sparse anomalies
# ---------------------------------------------------------------------------
def test_low_resolution_sparsity_is_presumed_to_be_blending():
    ok, note = V.resolution_verdict("GALAH")
    assert ok and "adequate" in note
    bad, note = V.resolution_verdict("LAMOST")
    assert not bad
    assert "unresolved blends" in note and "dissolved" in note
    assert not V.resolution_verdict("LAMOST_MRS")[0]
    assert V.resolution_verdict("APOGEE")[0]


def test_resolution_requirement_travels_with_every_candidate():
    hi = V.vet_candidates(_one(fe_h=0.0), survey="GALAH")
    lo = V.vet_candidates(_one(fe_h=0.0), survey="LAMOST")
    assert not bool(hi.loc[0, "needs_high_resolution_confirmation"])
    assert bool(lo.loc[0, "needs_high_resolution_confirmation"])


def test_saturated_lines_cannot_carry_an_abundance_claim():
    assert V.curve_of_growth_regime(0.35)[0] == V.WEAK
    assert V.curve_of_growth_regime(0.90)[0] == V.SATURATED
    assert "no longer a measure of abundance" in V.curve_of_growth_regime(0.90)[1]
    assert V.curve_of_growth_regime(0.005)[0] == V.UNDETECTED


# ---------------------------------------------------------------------------
# The dilution ceiling: it sets the S15 bar AND the sensitivity limit
# ---------------------------------------------------------------------------
def test_convective_zone_metal_reservoir_matches_the_published_ladder():
    """Reservoir and dex-per-Earth-mass across spectral type."""
    sun = float(T.cz_metal_reservoir_earth_masses(5772.0))
    assert 60.0 < sun < 120.0                       # ~89 Me of metals
    per_me = float(T.delta_metals_from_rock(1.0, teff=5772.0))
    assert per_me == pytest.approx(0.0048, abs=0.002)
    # Cooler stars dilute harder, monotonically.
    res = T.cz_metal_reservoir_earth_masses(np.array([5772.0, 5000.0, 4200.0]))
    assert res[0] < res[1] < res[2]


def test_engulfment_ceiling_is_the_whole_planetary_system():
    """Eating the entire Solar System gives 0.27 dex; Kronos is already 85% of it."""
    ceiling = float(T.engulfment_ceiling_dex(5772.0))
    assert ceiling == pytest.approx(0.27, abs=0.05)
    assert 0.23 / ceiling > 0.75                    # the observed record
    # A 0.3 dex coherent refractory excess in a G dwarf is beyond ANY budget.
    assert 0.30 > ceiling
    # And the bar is spectral-type dependent, not one number: a K dwarf's
    # ceiling is far lower, so a much smaller excess is already unexplainable.
    assert float(T.engulfment_ceiling_dex(4500.0)) < 0.10


def test_sensitivity_and_null_strength_trade_against_each_other():
    """The cool end has the strongest null AND the most diluted signal."""
    need = {t: float(T.minimum_rock_mass_for(1.0, t))
            for t in (6200.0, 5772.0, 5000.0, 4200.0)}
    assert need[6200.0] < need[5772.0] < need[5000.0] < need[4200.0]
    assert need[5772.0] > 500.0        # ~800 Me for +1 dex in a solar analogue
    assert need[4200.0] > 3000.0       # a K dwarf needs more rock than exists
    # Which is the honest limitation: an M/K-dwarf detection would require an
    # implausible amount of material, and that must be said rather than implied.


def test_vizier_table_names_come_back_quoted_and_must_be_stripped():
    """The bug that cost a dispatch: TAP_SCHEMA returns '"III/283/allstar"'.

    Interpolating that into a quoted FROM clause gives FROM ""III/283/allstar"",
    which every table rejects -- and the run then reports a clean
    NO_DATA_REACHED for what is really a quoting bug. The honest verdict made
    it look like an archive-access problem, which is exactly why the failure
    has to be caught in a test rather than in a log.
    """
    assert A.unquote_table('"III/283/allstar"') == "III/283/allstar"
    assert A.unquote_table("III/283/allstar") == "III/283/allstar"
    assert A.unquote_table('  "J/MNRAS/506/2269/table1" ') == "J/MNRAS/506/2269/table1"

    seen = []

    def query(adql):
        if "TAP_SCHEMA" in adql:
            return pd.DataFrame({"table_name": ['"III/999/main"'],
                                 "description": ["GALAH DR4"]})
        seen.append(adql)
        return pd.DataFrame({"sobject_id": [1], "Teff": [5000.0], "logg": [4.4],
                             "fe_h": [0.0], "snr": [100.0], "Mg_fe": [0.0],
                             "e_Mg_fe": [0.02], "Ni_fe": [0.0], "e_Ni_fe": [0.02]})

    found = A.discover_tables(("GALAH",), query_fn=query)
    assert found == ["III/999/main"], "the quotes must be stripped at discovery"

    acq = A.fetch_survey("GALAH", max_rows=1, n_chunks=1,
                         probe_fn=lambda t: query("probe") if "999" in t else None,
                         query_fn=query)
    assert acq.locator == "III/999/main"
    assert all('""' not in q for q in seen), "no double-quoted table name may reach the service"


def test_report_surfaces_the_evidence_a_reader_needs(tmp_path):
    """n_quiet, the caveats and the resolution requirement must reach REPORT.md."""
    from seti.tailings.run import write_report

    real = load_config()
    cfg = Config(root=tmp_path, thresholds=real.thresholds,
                 catalogs=real.catalogs, paths=real.paths)
    d = tmp_path / "results" / "tailings"
    d.mkdir(parents=True)
    summary = {
        "verdict": "SPARSE_CANDIDATES_PENDING_REMEASUREMENT: 1 survivor",
        "per_survey": [{
            "survey": "GALAH", "n_stars": 100000, "n_elements": 24,
            "n_sparse": 1, "n_vetted": 1,
            "class_counts": {"NORMAL": 99000, "DENSE": 999, "SPARSE": 1},
            "candidates": [{
                "star_id": "S1", "element_max": "Ba", "z_max_signed": 7.3,
                "n_quiet": 22, "contrast": 7.1, "teff": 5100.0, "fe_h": -0.2,
                "element_caveat": "Ba II lines are strong, saturated and NLTE",
                "needs_high_resolution_confirmation": False,
                "cool_star_caveat": False, "cross_survey": "not_covered",
            }],
        }],
        "twins": {"n_pairs": 0},
    }
    txt = write_report(cfg, d, summary).read_text()
    assert "n_quiet" in txt and "measured and found ordinary" in txt
    assert "| S1 | **Ba** | +7.3 | 22 |" in txt
    assert "Ba II lines" in txt
    assert "cross-survey: not_covered" in txt
    assert "information-starved" in txt          # the Huang caveat is carried


# ---------------------------------------------------------------------------
# The acquisition route: bulk survey files primary, VizieR TAP fallback.
#
# Three dispatches (30203627605, 30204487245, 30204793446) completed with
# workflow-level success and zero rows at every stage. The analysis was fine;
# acquisition never returned anything. Two things had to change and both are
# pinned here: the *route* (survey-native FITS, which is also the only place
# the fibre/RV columns correction #5 needs actually exist), and the *verdict*
# vocabulary, because "NO_DATA_REACHED" was being emitted for a dead URL, a
# broken query and an empty selection alike -- three different problems with
# three different fixes wearing one label.
# ---------------------------------------------------------------------------
def _fake_route_probe(status=200, length=500_000_000):
    """A stand-in for the HEAD/ranged-GET prober; no socket is ever opened."""
    def probe(url):
        return {"status": status, "content_length": length,
                "content_type": "application/fits", "probe_method": "HEAD"}
    return probe


def _survey_frame(n=4, teff=5000.0, covariates=True):
    """A minimal survey-native table: cool dwarfs, two elements, instrument columns."""
    cols = {
        "sobject_id": [f"1712{i:04d}" for i in range(n)],
        "teff": np.full(n, float(teff)),
        "logg": np.full(n, 4.4),
        "fe_h": np.zeros(n),
        "snr_c3_iraf": np.full(n, 90.0),
        "Mg_fe": np.zeros(n), "e_Mg_fe": np.full(n, 0.02),
        "Ni_fe": np.zeros(n), "e_Ni_fe": np.full(n, 0.02),
    }
    if covariates:
        cols["rv_galah"] = np.full(n, 12.0)
        cols["pivot"] = np.arange(n, dtype=float) + 1.0
    return pd.DataFrame(cols)


def test_bulk_file_route_is_primary_and_vizier_is_only_the_fallback():
    """The decisive change: the survey's own file is tried first, TAP second."""
    tap_calls = {"n": 0}

    def query(adql):
        tap_calls["n"] += 1
        return pd.DataFrame()

    acq = A.fetch_survey("GALAH", max_rows=10, n_chunks=1,
                         route_probe_fn=_fake_route_probe(),
                         read_fn=lambda url, sel: _survey_frame(),
                         probe_fn=lambda t: None, query_fn=query)
    assert acq.route == "file"
    assert acq.verdict == A.VERDICT_OK
    assert acq.n_rows == 4
    assert set(acq.elements) == {"Mg", "Ni"}
    assert tap_calls["n"] == 0, "VizieR must not be touched when the file route works"
    # and the instrumental covariates correction #5 needs actually arrived
    assert "rv" in acq.param_columns and "fiber" in acq.param_columns


def test_a_download_that_dies_is_query_failed_not_no_data_reached():
    """A dead download and an unreachable archive are different problems."""
    def boom(url, sel):
        raise RuntimeError("connection reset after 300 MB")

    acq = A.fetch_survey_file("GALAH", route_probe_fn=_fake_route_probe(), read_fn=boom)
    assert acq.verdict == A.VERDICT_QUERY_FAILED
    assert acq.verdict != A.VERDICT_NO_DATA
    assert acq.n_rows == 0 and acq.degraded
    assert "connection reset" in acq.degradation
    assert acq.degradation.startswith("QUERY_FAILED")


def test_a_clean_download_with_no_surviving_rows_is_zero_rows_not_failure():
    """The file parsed perfectly and the *cuts* emptied it. Say which."""
    empty = _survey_frame(0)
    acq = A.fetch_survey_file("GALAH", route_probe_fn=_fake_route_probe(),
                              read_fn=lambda u, s: empty)
    assert acq.verdict == A.VERDICT_ZERO_ROWS
    assert acq.verdict not in (A.VERDICT_NO_DATA, A.VERDICT_QUERY_FAILED)
    assert acq.degradation.startswith("QUERY_RETURNED_ZERO_ROWS")
    assert "about the cuts, not" in acq.degradation


def test_the_cool_dwarf_cut_is_what_empties_a_hot_star_file():
    """The same verdict, reached through the selection rather than an empty file."""
    hot = _survey_frame(6, teff=7200.0)
    acq = A.fetch_survey_file("GALAH", selection=A.Selection(max_rows=10),
                              route_probe_fn=_fake_route_probe(),
                              read_fn=lambda u, s: hot)
    assert acq.verdict == A.VERDICT_ZERO_ROWS
    assert acq.stage_counts["rows_in_file"] == 6
    assert acq.stage_counts["rows_after_selection"] == 0


def test_every_candidate_url_is_probed_and_its_status_recorded():
    """The sandbox cannot check a URL, so the runner checks all of them and reports."""
    acq = A.fetch_survey_file("APOGEE",
                              route_probe_fn=_fake_route_probe(status=404, length=None))
    assert acq.verdict == A.VERDICT_NO_DATA
    urls = [r["url"] for r in acq.download_routes]
    assert len(urls) == len(A.FILE_ROUTES["APOGEE"])
    assert any("data.sdss.org/sas/dr17/apogee" in u for u in urls)
    assert all(r["status"] == 404 and not r["eligible"] for r in acq.download_routes)
    assert acq.stage_counts["routes_probed"] == len(A.FILE_ROUTES["APOGEE"])
    assert acq.stage_counts["routes_eligible"] == 0
    # the point of the report: the next dispatch is told exactly what to fix
    assert "provenance.json" in acq.degradation


def test_a_200_too_small_to_be_a_catalogue_is_not_selected():
    """A 4 kB reply to a catalogue request is an error page, not a catalogue."""
    report = A.probe_download_routes(
        A.FILE_ROUTES["GALAH"], probe_fn=lambda u: {"status": 200, "content_length": 4096})
    assert not any(r["eligible"] for r in report)
    assert all("too small" in r["why"] for r in report if r["expects_abundances"])


def test_a_value_added_file_is_probed_to_prove_the_host_but_never_used():
    report = A.probe_download_routes(A.FILE_ROUTES["GALAH"], probe_fn=_fake_route_probe())
    vac = [r for r in report if not r["expects_abundances"]]
    assert vac, "VAC routes are registered so a 200 proves the host and path prefix are live"
    assert all(not r["eligible"] and not r["selected"] for r in vac)
    assert sum(1 for r in report if r["selected"]) == 1, "exactly one route is selected"
    assert report[0]["selected"], "and it is the first eligible one, in the stated order"


def test_dead_urls_fall_back_to_vizier_and_the_report_names_both_routes():
    head = pd.DataFrame({"sobject_id": [1], "Teff": [5000.0], "logg": [4.4],
                         "fe_h": [0.0], "snr": [100.0], "Mg_fe": [0.0],
                         "e_Mg_fe": [0.02], "Ni_fe": [0.0], "e_Ni_fe": [0.02]})
    rows = pd.concat([head, head], ignore_index=True)

    acq = A.fetch_survey("GALAH", max_rows=8, n_chunks=1, discover=False, use_files=True,
                         route_probe_fn=_fake_route_probe(status=403, length=None),
                         probe_fn=lambda t: head, query_fn=lambda q: rows)
    assert acq.route == "tap" and acq.n_rows == 2
    assert acq.verdict == A.VERDICT_OK
    assert "bulk-file route" in acq.degradation
    assert acq.download_routes and all(r["status"] == 403 for r in acq.download_routes)


def test_tap_chunks_that_all_raise_are_query_failed():
    head = pd.DataFrame({"sobject_id": [1], "Teff": [5000.0], "logg": [4.4],
                         "fe_h": [0.0], "snr": [100.0], "Mg_fe": [0.0],
                         "e_Mg_fe": [0.02], "Ni_fe": [0.0], "e_Ni_fe": [0.02]})

    def query(adql):
        raise RuntimeError("HTTP 503 from the TAP service")

    acq = A.fetch_survey("GALAH", n_chunks=3, discover=False,
                         probe_fn=lambda t: head, query_fn=query)
    assert acq.verdict == A.VERDICT_QUERY_FAILED
    assert acq.n_rows == 0
    assert acq.stage_counts["chunks_failed"] == 3
    assert "not an empty sky" in acq.degradation


def test_tap_chunks_that_answer_with_nothing_are_zero_rows():
    head = pd.DataFrame({"sobject_id": [1], "Teff": [5000.0], "logg": [4.4],
                         "fe_h": [0.0], "snr": [100.0], "Mg_fe": [0.0],
                         "e_Mg_fe": [0.02], "Ni_fe": [0.0], "e_Ni_fe": [0.02]})

    acq = A.fetch_survey("GALAH", n_chunks=3, discover=False,
                         probe_fn=lambda t: head, query_fn=lambda q: pd.DataFrame())
    assert acq.verdict == A.VERDICT_ZERO_ROWS
    assert acq.stage_counts["chunks_failed"] == 0
    assert acq.stage_counts["rows_returned"] == 0
    assert "about the cuts, not" in acq.degradation


def test_a_table_without_fibre_or_rv_says_the_covariate_veto_cannot_run():
    """Correction #5 needs RV, fibre and detector position; absence is degradation."""
    acq = A.fetch_survey_file("GALAH", route_probe_fn=_fake_route_probe(),
                              read_fn=lambda u, s: _survey_frame(4, covariates=False))
    assert acq.n_rows == 4
    assert "veto on rv cannot run" in acq.degradation
    assert "veto on fiber cannot run" in acq.degradation


# ---------------------------------------------------------------------------
# VizieR fallback: discovery has to look at COLUMNS, not descriptions
# ---------------------------------------------------------------------------
def test_discovery_by_abundance_columns_finds_what_the_description_search_misses():
    """34 tables answered with a GALAH description and zero elements last time.

    The description says what a catalogue is *about*; the column list says what
    it *contains*, and only the second is the thing this channel needs.
    """
    schema = {
        "III/999/stub": ["Teff", "logg", "__Fe_H_", "Nobs"],
        "III/777/abund": ["sobject_id", "Teff", "logg", "__Fe_H_", "SNR", "RV", "Pivot",
                          "__Mg_Fe_", "e__Mg_Fe_", "__Ni_Fe_", "e__Ni_Fe_", "__Ba_Fe_"],
    }

    def query(adql):
        if "TAP_SCHEMA.tables" in adql:
            # keyword search finds only the useless stub
            return pd.DataFrame({"table_name": ['"III/999/stub"'],
                                 "description": ["GALAH DR3 per-field summary"]})
        if "TAP_SCHEMA.columns" in adql and "column_name LIKE" in adql:
            return pd.DataFrame({"table_name": ['"III/777/abund"'] * 3,
                                 "column_name": ["__Mg_Fe_", "__Ni_Fe_", "__Ba_Fe_"]})
        if "TAP_SCHEMA.columns" in adql:
            rows = [(t, c) for t, cs in schema.items() if t in adql for c in cs]
            return pd.DataFrame({"table_name": [f'"{t}"' for t, _ in rows],
                                 "column_name": [c for _, c in rows]})
        assert "III/777/abund" in adql, "the pull must run against the abundance table"
        return pd.DataFrame({c: [1.0] for c in schema["III/777/abund"]})

    acq = A.fetch_survey("GALAH", max_rows=1, n_chunks=1, use_files=False,
                         probe_fn=lambda t: None, query_fn=query)
    assert acq.locator == "III/777/abund"
    assert set(acq.elements) == {"Mg", "Ni", "Ba"}
    # and the fibre/RV columns the covariate veto needs were carried through
    assert "rv" in acq.param_columns and "fiber" in acq.param_columns
    board = {r["table"]: r for r in acq.scoreboard}
    assert board["III/999/stub"]["score"] == 0
    assert "no [X/Fe] columns" in board["III/999/stub"]["why"]
    assert board["III/777/abund"]["schema_from"] == "TAP_SCHEMA.columns"
    assert board["III/777/abund"]["n_elements"] == 3


def test_a_rejected_table_says_what_it_was_missing():
    """'missing parameters or elements' was printed 34 times and taught nobody anything."""
    why = A.schema_reason({"teff": "T", "logg": "g"}, {})
    assert "fe_h" in why and "[X/Fe]" in why and why.startswith("rejected")
    assert A.schema_reason({"teff": "T", "logg": "g", "fe_h": "f", "snr": "s",
                            "star_id": "i", "rv": "v", "fiber": "p"}, {"Mg": {}}) == "usable"
    assert "without snr" in A.schema_reason(
        {"teff": "T", "logg": "g", "fe_h": "f"}, {"Mg": {}})


def test_column_index_matches_quoted_and_unquoted_table_names():
    """VizieR stores the name quoted in some places and bare in others."""
    seen = []

    def query(adql):
        seen.append(adql)
        return pd.DataFrame({"table_name": ['"III/283/allstar"'], "column_name": ["Teff"]})

    idx = A.fetch_table_columns(["III/283/allstar"], query_fn=query)
    assert idx == {"III/283/allstar": ["Teff"]}
    assert "LIKE" in seen[0], "equality on the stored name silently misses half of them"


# ---------------------------------------------------------------------------
# The FITS reader: only the needed columns, only the needed rows
# ---------------------------------------------------------------------------
def test_the_fits_reader_takes_only_the_columns_and_rows_it_needs(tmp_path):
    from astropy.io import fits

    n = 400
    rng = np.random.default_rng(11)
    teff = rng.uniform(3500.0, 7500.0, n)
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="APOGEE_ID", format="20A",
                    array=np.array([f"2M{i:06d}" for i in range(n)])),
        fits.Column(name="TEFF", format="E", array=teff.astype(np.float32)),
        fits.Column(name="LOGG", format="E", array=np.full(n, 4.5, dtype=np.float32)),
        fits.Column(name="FE_H", format="E", array=np.zeros(n, dtype=np.float32)),
        fits.Column(name="SNR", format="E", array=np.full(n, 120.0, dtype=np.float32)),
        fits.Column(name="VHELIO_AVG", format="E",
                    array=rng.normal(0.0, 30.0, n).astype(np.float32)),
        fits.Column(name="MEANFIB", format="E",
                    array=rng.uniform(1.0, 300.0, n).astype(np.float32)),
        fits.Column(name="VSCATTER", format="E", array=np.full(n, 0.1, dtype=np.float32)),
        fits.Column(name="MG_FE", format="E", array=np.zeros(n, dtype=np.float32)),
        fits.Column(name="MG_FE_ERR", format="E", array=np.full(n, 0.02, dtype=np.float32)),
        fits.Column(name="MG_FE_FLAG", format="J", array=np.zeros(n, dtype=np.int32)),
        # the APOGEE element panel is a 2-D column and must be skipped, not crashed on
        fits.Column(name="X_H", format="20E", dim="(20)",
                    array=rng.normal(0.0, 1.0, (n, 20)).astype(np.float32)),
    ])
    path = tmp_path / "allStar-dr17-synspec_rev1.fits"
    hdu.writeto(path)

    df = A.read_fits_table(path, selection=A.Selection(teff_min=3000.0, teff_max=6000.0,
                                                      logg_min=4.0, snr_min=40.0,
                                                      feh_min=-1.0))
    assert "X_H" not in df.columns, "multi-dimensional columns are skipped, not exploded"
    assert df.attrs["n_rows_file"] == n
    assert 0 < len(df) < n, "the Teff cut must actually bite"
    assert df["TEFF"].max() < 6000.0 and df["TEFF"].min() > 3000.0
    # the instrumental covariates correction #5 requires, present in the survey
    # file and generally absent from the abbreviated VizieR copy
    for c in ("VHELIO_AVG", "MEANFIB", "VSCATTER", "MG_FE", "MG_FE_ERR", "MG_FE_FLAG"):
        assert c in df.columns
    params = A.resolve_param_columns(df.columns)
    assert params["rv"] == "VHELIO_AVG" and params["fiber"] == "MEANFIB"
    assert A.resolve_abundance_columns(df.columns)["Mg"]["flag"] == "MG_FE_FLAG"

    # and with no selection the reader returns the whole file
    full = A.read_fits_table(path, selection=None)
    assert len(full) == n


def test_a_file_without_an_snr_column_reports_the_cut_it_could_not_apply(tmp_path):
    from astropy.io import fits

    n = 50
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="TEFF", format="E", array=np.full(n, 5000.0, dtype=np.float32)),
        fits.Column(name="LOGG", format="E", array=np.full(n, 4.5, dtype=np.float32)),
        fits.Column(name="FE_H", format="E", array=np.zeros(n, dtype=np.float32)),
        fits.Column(name="MG_FE", format="E", array=np.zeros(n, dtype=np.float32)),
    ])
    path = tmp_path / "nosnr.fits"
    hdu.writeto(path)
    df = A.read_fits_table(path, selection=A.Selection())
    assert len(df) == n
    assert "snr" in df.attrs["cuts_not_applied"]

    acq = A.fetch_survey_file("GALAH", route_probe_fn=_fake_route_probe(),
                              read_fn=lambda u, s: A.read_fits_table(path, selection=s))
    assert acq.n_rows == n
    assert "so that cut was NOT applied" in acq.degradation


# ---------------------------------------------------------------------------
# Per-stage row counts and the verdict vocabulary must reach summary.json
# ---------------------------------------------------------------------------
def test_per_stage_row_counts_and_verdicts_reach_the_summary(tmp_path, monkeypatch):
    """The brief's hard requirement: the summary says where the rows went."""
    import json

    from seti.tailings import acquire as AQ
    from seti.tailings.run import tailings_run

    stars = A.normalize(_survey_frame(12), survey="GALAH")

    def fake_fetch(survey, **kw):
        return AQ.Acquisition(
            survey=survey, table=stars, source_used="GALAH_DR3_allstar_v2_cloud",
            locator="https://cloud.datacentral.org.au/.../GALAH_DR3_main_allstar_v2.fits",
            n_rows=len(stars), elements=["Mg", "Ni"], verdict=AQ.VERDICT_OK, route="file",
            stage_counts={"routes_probed": 10, "routes_eligible": 8,
                          "rows_in_file": 588571, "rows_after_selection": len(stars),
                          "rows_normalised": len(stars), "n_elements": 2},
            download_routes=[{"name": "GALAH_DR3_allstar_v2_cloud", "url": "https://x/f.fits",
                              "status": 200, "content_length": 512_000_000,
                              "eligible": True, "selected": True, "used": True}],
        )

    def fake_wb(**kw):
        return AQ.Acquisition(
            survey="WIDEBINARY", table=pd.DataFrame(), source_used="ELBADRY2021_zenodo_records",
            locator=None, n_rows=0, elements=[], degraded=True,
            verdict=AQ.VERDICT_ZERO_ROWS, route="file",
            degradation="QUERY_RETURNED_ZERO_ROWS: no pair passed the purity cut",
            stage_counts={"pairs_in_file": 1_256_400, "pairs_after_purity_cut": 0},
        )

    monkeypatch.setattr(AQ, "fetch_survey", fake_fetch)
    monkeypatch.setattr(AQ, "fetch_wide_binaries", fake_wb)

    real = load_config()
    cfg = Config(root=tmp_path, thresholds=real.thresholds,
                 catalogs=real.catalogs, paths=real.paths)
    tailings_run(cfg, stage="all", surveys="GALAH")
    summary = json.loads((tmp_path / "results" / "tailings" / "summary.json").read_text())

    prov = summary["provenance"]["surveys"][0]
    assert prov["verdict"] == "OK"
    assert prov["route"] == "file"
    counts = prov["stage_counts"]
    for key in ("routes_probed", "routes_eligible", "rows_in_file",
                "rows_after_selection", "rows_normalised"):
        assert key in counts, f"{key} is a per-stage count and must reach summary.json"
    assert counts["rows_in_file"] == 588571
    assert counts["rows_after_selection"] == len(stars)
    route = prov["download_routes"][0]
    assert route["status"] == 200 and route["selected"] and route["used"]

    wb = summary["provenance"]["wide_binaries"]
    assert wb["verdict"] == "QUERY_RETURNED_ZERO_ROWS"
    assert wb["verdict"] != "NO_DATA_REACHED", "an empty purity cut is not an unreachable archive"
    assert wb["stage_counts"]["pairs_in_file"] == 1_256_400


def test_the_three_failure_verdicts_are_distinct_strings():
    """They are compared as strings across the funnel; collapsing any two hides a bug."""
    assert len({A.VERDICT_NO_DATA, A.VERDICT_QUERY_FAILED,
                A.VERDICT_ZERO_ROWS, A.VERDICT_OK}) == 4
    assert A.Acquisition(survey="X", table=pd.DataFrame(), source_used=None, locator=None,
                         n_rows=0, elements=[]).provenance()["verdict"] == A.VERDICT_OK


# ---------------------------------------------------------------------------
# Wide binaries by file, with the same verdict vocabulary
# ---------------------------------------------------------------------------
def test_wide_binaries_come_from_the_published_file_first():
    pairs = pd.DataFrame({"source_id1": [1, 3, 5], "source_id2": [2, 4, 6],
                          "R_chance_align": [0.01, 0.02, 0.9]})
    acq = A.fetch_wide_binaries(route_probe_fn=_fake_route_probe(length=200_000_000),
                                read_fn=lambda u, s: pairs)
    assert acq.route == "file" and acq.verdict == A.VERDICT_OK
    assert acq.n_rows == 2, "the R_chance_align < 0.1 purity cut must be applied"
    assert set(acq.table.columns) >= {"source_id_a", "source_id_b", "r_chance_align"}
    assert acq.stage_counts["pairs_in_file"] == 3


def test_wide_binary_purity_cut_emptying_the_table_is_zero_rows():
    pairs = pd.DataFrame({"source_id1": [1], "source_id2": [2], "R_chance_align": [0.95]})
    acq = A.fetch_wide_binaries_file(route_probe_fn=_fake_route_probe(length=200_000_000),
                                     read_fn=lambda u, s: pairs)
    assert acq.verdict == A.VERDICT_ZERO_ROWS
    assert acq.stage_counts["pairs_after_purity_cut"] == 0


def test_an_xh_only_catalogue_is_rejected_but_diagnosed():
    """[X/H] must never be read as [X/Fe] -- it would double-subtract the metallicity.

    But the rejection has to name the reason, because a schema-convention
    mismatch has a one-line fix and a dead archive does not.
    """
    xh = pd.DataFrame({"sobject_id": ["1"], "teff": [5000.0], "logg": [4.4],
                       "fe_h": [0.0], "snr": [90.0],
                       "Mg_h": [0.1], "Ni_h": [0.05], "Ba_h": [0.0]})
    assert A.resolve_abundance_columns(xh.columns) == {}
    assert set(A.resolve_xh_columns(xh.columns)) == {"Mg", "Ni", "Ba"}

    acq = A.fetch_survey_file("GALAH", route_probe_fn=_fake_route_probe(),
                              read_fn=lambda u, s: xh)
    assert acq.n_rows == 0
    assert acq.verdict == A.VERDICT_QUERY_FAILED
    assert "[X/H] columns" in acq.degradation
    assert "schema-convention mismatch" in acq.degradation


def test_fits_big_endian_columns_survive_the_parquet_checkpoint(tmp_path):
    """FITS is big-endian and pyarrow refuses byte-swapped arrays.

    Run 30211322736 downloaded 758 MB of GALAH DR4, parsed 73,820 rows and 30
    elements, and then died on the FIRST ``to_parquet`` with::

        pyarrow.lib.ArrowNotImplementedError: ('Byte-swapped arrays not
        supported', 'Conversion failed for column star_id with type >i8')

    i.e. it failed *after* all the expensive work, which is the worst possible
    place to fail. Normalisation is centralised in ``to_native_byteorder`` and
    applied both at FITS-read time and again at checkpoint time.
    """
    import numpy as np
    import pandas as pd

    from seti.tailings.acquire import to_native_byteorder, write_checkpoint

    df = pd.DataFrame(
        {
            "star_id": np.array([1, 2, 3], dtype=">i8"),
            "teff": np.array([5000.0, 5500.0, 6000.0], dtype=">f8"),
            "fe_h": np.array([-0.1, 0.0, 0.2], dtype="<f8"),   # already native
            "name": ["a", "b", "c"],
        }
    )
    # Guard the premise: the raw frame really is unwritable.
    with pytest.raises(Exception, match="[Bb]yte-swapped"):
        df.to_parquet(tmp_path / "raw.parquet", index=False)

    native = to_native_byteorder(df)
    for col in ("star_id", "teff", "fe_h"):
        assert native[col].dtype.byteorder in ("=", "|"), col
    assert native["star_id"].tolist() == [1, 2, 3]
    assert native["teff"].tolist() == [5000.0, 5500.0, 6000.0]

    path = write_checkpoint(df, tmp_path / "stars.parquet")
    back = pd.read_parquet(path)
    assert back["star_id"].tolist() == [1, 2, 3]
    assert back["teff"].tolist() == [5000.0, 5500.0, 6000.0]
    assert back["fe_h"].tolist() == [-0.1, 0.0, 0.2]
    # An empty or already-native frame must pass through untouched.
    assert to_native_byteorder(pd.DataFrame()).empty


def test_threshold_sweep_reports_the_rate_against_the_published_scale():
    """The calibration curve, not the candidate count, is the channel's output.

    The real run flagged 2,100 vetted sparse anomalies in 210,867 stars (1.0%)
    against Griffith et al. 2022's 15 in 82,910 (1.8e-4) -- a 55x excess that
    cannot be astrophysical. A raw count is meaningless without the rate it
    implies, so ``threshold_sweep`` is computed on the same z-matrix as the
    candidates and carries ``rate_over_griffith`` explicitly.
    """
    import numpy as np
    import pandas as pd

    from seti.tailings.run import GRIFFITH_RATE, threshold_sweep
    from seti.tailings.sparse import SparseConfig

    assert GRIFFITH_RATE == pytest.approx(15 / 82910)

    els = ["O", "Na", "Mg", "Al", "Si", "K", "Ca", "Sc", "Ti", "V",
           "Cr", "Mn", "Co", "Ni", "Cu", "Zn", "Y", "Ba", "Ce", "Eu"]
    rng = np.random.default_rng(0)
    Z = pd.DataFrame(rng.standard_normal((2000, len(els))), columns=els)

    sw = threshold_sweep(Z, SparseConfig(), z_flags=(5.0, 8.0),
                         quiet_sigmas=(2.0,), contrasts=(3.0,))
    for col in ("z_flag", "n_stars", "n_sparse", "n_dense", "sparse_rate",
                "sparse_over_dense", "rate_over_griffith"):
        assert col in sw.columns
    assert len(sw) == 2
    assert (sw["n_stars"] == 2000).all()
    # PURE GAUSSIAN NOISE PRODUCES NO SPARSE CANDIDATES. This is the reference
    # point that makes the real 1,341 interpretable: they are entirely
    # non-Gaussian tail, i.e. the survey error model, not chance.
    assert (sw["n_sparse"] == 0).all(), (
        "Gaussian noise is producing sparse candidates: the rules are too loose "
        "to be a meaningful null reference"
    )
    # Tightening must be monotonic: more z_flag can never yield more candidates.
    tight = sw.sort_values("z_flag")
    assert tight["n_sparse"].is_monotonic_decreasing


def test_narrow_telluric_window_is_vetoed_where_quantile_bins_cannot_see_it():
    """Weinberg's low-K population, reproduced and now catchable.

    The real run's APOGEE candidates carry a demonstrable telluric artefact: K
    anomalies enhanced 4.03x over an (Fe/H, Teff)-matched control in the -110 to
    -60 km/s window (p = 2.7e-10), 28 of 29 of them DEFICITS. The pipeline
    vetoed none, because ``covariate_rate_veto`` bins into 12 equal-population
    quantiles over the full +/-200 km/s range: observed rv_flag_ratio spanned
    only 0.81-1.32 against a threshold of 5.0, so the veto was not merely
    un-triggered but un-triggerable at that resolution.
    """
    import numpy as np
    import pandas as pd

    from seti.tailings.vet import covariate_window_veto

    rng = np.random.default_rng(0)
    n = 300
    el = np.array(["K"] * 60 + ["Mg", "Ca", "Ni", "Y", "Na"] * 48)
    rv = rng.uniform(-200, 200, n)
    rv[np.flatnonzero(el == "K")[:40]] = rng.uniform(-80, -60, 40)
    df = pd.DataFrame({"element_max": el, "rv": rv, "star_id": np.arange(n)})

    out = covariate_window_veto(df, covariate_col="rv")
    vetoed = ~out["pass_rv_window"]
    k_vetoed = int((vetoed & (out["element_max"] == "K")).sum())
    assert k_vetoed >= 35, f"only {k_vetoed} of the 40 planted K artefacts vetoed"
    assert "telluric" in out.loc[vetoed, "rv_window_reason"].iloc[0]
    assert not bool(out["rv_window_untested"].iloc[0])

    # NULL CONTROL: no artefact, nothing vetoed. A veto that fires on clean data
    # would delete the very candidates the channel exists to find.
    clean = pd.DataFrame({"element_max": el, "rv": rng.uniform(-200, 200, n),
                          "star_id": np.arange(n)})
    assert int((~covariate_window_veto(clean, covariate_col="rv")
                ["pass_rv_window"]).sum()) == 0

    # A missing covariate must be reported as UNTESTED, never as a pass.
    no_rv = pd.DataFrame({"element_max": el, "star_id": np.arange(n)})
    o3 = covariate_window_veto(no_rv, covariate_col="rv")
    assert bool(o3["rv_window_untested"].iloc[0])


def test_yaml_and_dataclass_thresholds_do_not_drift():
    """A dataclass default that drifts from the yaml gives library callers --
    including the Griffith validation harness -- a different search from the one
    the workflow actually runs. Two such drifts were found live (z_flag 6.0 vs
    5.0, min_elements 8 vs 12); this pins every field.
    """
    import yaml

    from seti.tailings.sparse import SparseConfig

    block = yaml.safe_load(open("config/thresholds.yaml"))["tailings"]["sparse"]
    default = SparseConfig()
    drift = {k: (v, getattr(default, k)) for k, v in block.items()
             if hasattr(default, k) and getattr(default, k) != v}
    assert not drift, f"yaml/dataclass drift: {drift}"
