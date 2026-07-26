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
    acq = A.fetch_survey("GALAH", probe_fn=lambda t: None, query_fn=lambda q: None)
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

    acq = A.fetch_survey("GALAH", n_chunks=1, probe_fn=probe, query_fn=query)
    assert acq.n_rows == 2
    assert acq.degraded
    assert "fell back" in acq.degradation
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

    acq = A.fetch_survey("GALAH", max_rows=20, n_chunks=4,
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

    acq = A.fetch_survey("GALAH", max_rows=8, n_chunks=4,
                         probe_fn=lambda t: head, query_fn=query)
    assert acq.n_rows == 3
    assert "coverage is incomplete" in acq.degradation
