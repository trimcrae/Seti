"""FALLOUT offline tests: templates, classification, the pattern test, the funnel.

No network (tests/conftest.py enforces it). A synthetic GALAH-DR4-like cool
dwarf population -- with a real s-process axis, a real r-process axis,
Teff/logg/[Fe/H] pipeline trends and catalogue-style errors and flags -- drives:

(a) the physics tables: the chain yields sum to two fragments per fission and
    the three template vectors carry the predicted discriminant signs;
(b) the benchmarks: a synthetic Ba star is classified s, a synthetic r-II star
    r, a synthetic fission-polluted star fission;
(c) the single-element-driver test: a one-element Nd spike is NOT a pattern;
(d) every named veto is tripped by a case built to trip it;
(e) the peer residual removes an injected Teff trend that would otherwise
    manufacture raw-space "patterns";
(f) the shuffled-element null rarely makes the vector by accident, and an
    injected fission star at high amplitude is recovered end to end;
(g) the channel degrades honestly: dead routes give NO_DATA_REACHED and a
    fallback route gives a DEGRADED_SOURCE prefix, with the acquisition log,
    the columns found and the veto counters carried in summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seti.config import Config, load_config
from seti.fallout import acquire as FA
from seti.fallout import pattern as P
from seti.fallout import run as R
from seti.fallout import yields as Y

GALAH_ELEMENTS = ["Rb", "Sr", "Y", "Zr", "Mo", "Ru", "Ba", "La", "Ce", "Nd", "Sm", "Eu"]
ALPHA = ["Mg", "Si", "Ca", "Ti"]
#: catalogue-style per-element errors, dex (GALAH DR4 scale: Nd/Ce/La ~0.1-0.2)
ERRS = {"Rb": 0.18, "Sr": 0.08, "Y": 0.07, "Zr": 0.10, "Mo": 0.18, "Ru": 0.18, "Ba": 0.07,
        "La": 0.12, "Ce": 0.12, "Nd": 0.12, "Sm": 0.15, "Eu": 0.10}


@pytest.fixture(scope="module")
def templates() -> P.Templates:
    return P.build_templates(GALAH_ELEMENTS)


@pytest.fixture(scope="module")
def pcfg() -> P.PatternConfig:
    block = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "fallout.yaml").read_text())
    return R._pattern_config(block["fallout"])


@pytest.fixture(scope="module")
def block() -> dict:
    doc = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "fallout.yaml").read_text())
    return doc["fallout"]


def make_population(rng, n=4000, *, teff_trend=0.0, canonical=True):
    """Synthetic GALAH-like cool dwarfs in the canonical (post-normalize) schema.

    ``teff_trend`` adds a dex-per-1000K slope to the heavy-peak elements
    only -- a pipeline systematic that, uncorrected, aligns with the fission
    shape in raw space.
    """
    T = P.build_templates(GALAH_ELEMENTS)
    feh = rng.normal(-0.05, 0.2, n)
    teff = rng.uniform(4500.0, 6300.0, n)
    logg = rng.uniform(4.05, 4.6, n)
    snr = 10 ** rng.uniform(np.log10(40.0), np.log10(250.0), n)
    alpha = np.clip(-0.25 * feh + rng.normal(0, 0.03, n), -0.1, 0.4)
    a_s = np.abs(rng.normal(0.0, 0.25, n))          # AGB pollution axis
    a_r = np.abs(rng.normal(0.0, 0.25, n))          # r-process axis
    df = pd.DataFrame({
        "star_id": [f"G{i:07d}" for i in range(n)], "teff": teff, "logg": logg, "fe_h": feh,
        "snr": snr, "flag_sp": 0, "age": rng.uniform(1.5, 9.0, n), "survey": "SYNTH",
        "Li": rng.normal(0.0, 0.3, n) - 0.3,
    })
    for a in ALPHA:
        df[a] = alpha + rng.normal(0, 0.03, n)
        df[f"e_{a}"] = 0.03
        df[f"f_{a}"] = 0
    for k, el in enumerate(T.elements):
        base = np.log10(1 + a_s * T.S[k] + a_r * T.R[k])
        base = base - 0.06 * feh + 0.02 * (logg - 4.3)
        if el in ("Ba", "La", "Ce", "Nd", "Sm"):
            base = base + teff_trend * (teff - 5400.0) / 1000.0 * (1.5 if el == "Nd" else 1.0)
        e = ERRS[el] * np.sqrt(80.0 / snr)
        df[el] = base + rng.normal(0, 1, n) * e
        df[f"e_{el}"] = e
        df[f"f_{el}"] = 0
    if canonical:
        return df
    return df


def add_pattern(df, i, kind, amplitude, T=None):
    """Add a pure source to star ``i`` in place, in [X/Fe]."""
    T = T or P.build_templates(GALAH_ELEMENTS)
    pat = {"s": T.S, "r": T.R, "f": T.F}[kind]
    for k, el in enumerate(T.elements):
        df.loc[i, el] = df.loc[i, el] + np.log10(1 + amplitude * pat[k])


# ---------------------------------------------------------------------------
# (a) physics
# ---------------------------------------------------------------------------
def test_chain_yields_sum_to_two_fragments_per_fission():
    assert 195.0 < Y.total_chain_yield() < 205.0


def test_decay_horizon_moves_technetium_to_ruthenium():
    fresh = Y.element_yields(horizon_yr=100.0)
    old = Y.element_yields(horizon_yr=1.0e6)
    assert fresh.get("Tc", 0) > 5.0, "the A=99 chain sits on Tc-99 for 2e5 yr"
    assert old.get("Tc", 0) == 0.0, "MIDDEN's line has decayed at the FALLOUT horizon"
    assert old["Ru"] > fresh["Ru"] + 5.0
    # Cs-137 and Sr-90 are Ba and Zr on both horizons that matter here
    assert fresh["Ba"] > 12 and old["Ba"] > 12
    assert "Pb" not in old


def test_fission_template_has_the_predicted_discriminant_signs(templates):
    f = P.discriminant_ratios(dict(zip(templates.elements, np.log10(1 + templates.F), strict=True)))
    assert f["Nd/Ba"] > 0.2
    assert f["Ce/Ba"] > 0.05
    assert f["La/Ba"] > 0.05
    assert f["Mo/Zr"] > 0.05
    assert f["Ru/Zr"] > 0.05
    assert f["Eu/Nd"] < -0.1
    assert f["Sr/Nd"] < -0.2, "fission barely touches Sr against its huge solar abundance"


def test_s_and_r_templates_have_the_opposite_signs(templates):
    s = P.discriminant_ratios(dict(zip(templates.elements, np.log10(1 + templates.S), strict=True)))
    r = P.discriminant_ratios(dict(zip(templates.elements, np.log10(1 + templates.R), strict=True)))
    assert s["Nd/Ba"] < 0 and s["Mo/Zr"] < 0 and s["Sr/Nd"] > 0
    assert r["Eu/Nd"] > 0.1


def test_template_table_is_reproducible_and_cites_every_input():
    rows = Y.template_table(GALAH_ELEMENTS, amplitude=1.0)
    by = {r["element"]: r for r in rows}
    assert abs(by["Nd"]["fission_dex"] - np.log10(2.0)) < 1e-3, "a_f = 1 doubles the anchor"
    assert abs(by["Nd"]["F_over_Fanchor"] - 1.0) < 1e-9
    assert by["Eu"]["r_fraction"] > 0.9
    assert 0.0 <= by["Mo"]["p_fraction"] <= 0.3
    for r in rows:
        assert r["solar_logeps"] is not None and r["s_fraction"] is not None


# ---------------------------------------------------------------------------
# (b) benchmarks
# ---------------------------------------------------------------------------
def _sig(n):
    return np.tile(np.sqrt(np.array([ERRS[e] for e in GALAH_ELEMENTS]) ** 2 + 0.05 ** 2), (n, 1))


@pytest.mark.parametrize("kind,amp,expect", [
    ("s", 3.0, P.S_PROCESS),   # Ba star: [Ba/Fe] ~ +0.5, [Nd/Ba] < 0
    ("r", 6.0, P.R_PROCESS),   # r-II: [Eu/Fe] ~ +0.8, [Eu/Nd] > 0
    ("f", 5.0, P.FISSION),     # fission-polluted: Nd +0.78 dex
])
def test_synthetic_benchmark_is_classified_correctly(templates, pcfg, kind, amp, expect):
    rng = np.random.default_rng(3)
    pat = {"s": templates.S, "r": templates.R, "f": templates.F}[kind]
    n = 40
    obs = np.log10(1 + amp * pat)[None, :] + rng.normal(0, 1, (n, len(GALAH_ELEMENTS))) * _sig(n) * 0.7
    res = P.fit_patterns(obs, _sig(n), templates, pcfg)
    frac = (res["classification"] == expect).mean()
    assert frac >= 0.8, f"{kind}: {res['classification'].value_counts().to_dict()}"
    if kind == "f":
        loo = P.leave_one_out(obs, _sig(n), templates, pcfg)
        assert (loo["lr_loo_min"] >= pcfg.lr_min).mean() >= 0.7, "a real pattern survives leave-one-out"
    else:
        assert (res["fission_lr"] < 0).mean() >= 0.9, "natural mixtures must not look like fission"


def test_s_plus_r_mixture_is_not_mistaken_for_fission(templates, pcfg):
    rng = np.random.default_rng(11)
    n = 60
    a_s = rng.uniform(0.5, 4.0, n)
    a_r = rng.uniform(0.5, 4.0, n)
    obs = np.log10(1 + a_s[:, None] * templates.S[None, :] + a_r[:, None] * templates.R[None, :])
    obs = obs + rng.normal(0, 1, obs.shape) * _sig(n)
    res = P.fit_patterns(obs, _sig(n), templates, pcfg)
    assert (res["classification"] == P.FISSION).sum() == 0
    assert (res["fission_lr"] >= pcfg.lr_min).sum() == 0


def test_a_pure_fission_star_is_not_fitted_by_any_natural_mixture(templates, pcfg):
    obs = np.log10(1 + 8.0 * templates.F)[None, :]
    res = P.fit_patterns(obs, _sig(1), templates, pcfg)
    assert res["classification"].iloc[0] == P.FISSION
    assert res["fission_lr"].iloc[0] > 20
    assert res["chi2_f"].iloc[0] < 1e-3, "the refined amplitude must reproduce the input"
    assert abs(res["a_f"].iloc[0] - 8.0) / 8.0 < 0.05


# ---------------------------------------------------------------------------
# (c) the pattern test
# ---------------------------------------------------------------------------
def test_single_element_nd_spike_is_not_a_pattern(templates, pcfg):
    obs = np.zeros((1, len(GALAH_ELEMENTS)))
    obs[0, templates.index("Nd")] = 0.9
    res = P.fit_patterns(obs, _sig(1), templates, pcfg)
    loo = P.leave_one_out(obs, _sig(1), templates, pcfg)
    assert loo["lr_loo_driver"].iloc[0] == "Nd"
    assert loo["lr_loo_min"].iloc[0] < 1.0, "remove Nd and there is nothing left"
    cand = pd.concat([pd.DataFrame({"Nd": [0.9], "Ba": [0.0]}), res, loo], axis=1)
    cand["lr_noba"] = res["fission_lr"]
    cand["fission_lr_raw"] = res["fission_lr"]
    vet = P.apply_vetoes(cand, cfg=pcfg, lr_threshold=pcfg.lr_min)
    assert bool(vet["veto_single_element_driver"].iloc[0])
    assert not bool(vet["vet_pass"].iloc[0])


def test_two_element_heavy_peak_without_the_light_peak_is_flagged_by_loo(templates, pcfg):
    """Nd + Ce up, everything else solar: the LOO test names the driver."""
    obs = np.zeros((1, len(GALAH_ELEMENTS)))
    obs[0, templates.index("Nd")] = 0.6
    obs[0, templates.index("Ce")] = 0.3
    loo = P.leave_one_out(obs, _sig(1), templates, pcfg)
    assert loo["lr_loo_min"].iloc[0] < pcfg.lr_min


# ---------------------------------------------------------------------------
# (d) every veto tripped
# ---------------------------------------------------------------------------
def _passing_candidate(templates, pcfg) -> pd.DataFrame:
    """A row that passes every veto, to be broken one veto at a time."""
    obs = np.log10(1 + 6.0 * templates.F)[None, :]
    res = P.fit_patterns(obs, _sig(1), templates, pcfg)
    loo = P.leave_one_out(obs, _sig(1), templates, pcfg)
    row = {el: float(obs[0, k]) for k, el in enumerate(templates.elements)}
    row.update({f"peer_{el}": float(obs[0, k]) for k, el in enumerate(templates.elements)})
    row.update({f"sig_{el}": float(_sig(1)[0, k]) for k, el in enumerate(templates.elements)})
    row.update({"fe_h": 0.0, "snr": 120.0, "flag_sp": 0, "Li": -0.5, "age": 4.0, "binary_flag": 0,
                "sample": "dwarf", "teff": 5600.0})
    cand = pd.concat([pd.DataFrame([row]), res, loo], axis=1)
    cand["lr_noba"] = P.lr_without(obs, _sig(1), templates, pcfg, drop=("Ba",))
    cand["fission_lr_raw"] = res["fission_lr"]
    return cand


def test_the_reference_candidate_passes_every_veto(templates, pcfg):
    vet = P.apply_vetoes(_passing_candidate(templates, pcfg), cfg=pcfg, lr_threshold=pcfg.lr_min)
    assert bool(vet["vet_pass"].iloc[0]), vet["veto_reasons"].iloc[0]
    assert vet["first_veto"].iloc[0] == ""


@pytest.mark.parametrize("veto,mutate", [
    ("low_snr_or_flagged", {"flag_sp": 2}),
    ("low_snr_or_flagged", {"snr": 12.0}),
    ("s_process_star", {"Ba": 0.9, "Nd": 0.6}),                  # [Nd/Ba] < 0 with Ba high
    ("s_process_star", {"Ba": 0.9, "Y": 0.5, "Zr": 0.5}),          # whole s-process up
    ("s_process_star", {"binary_flag": 1}),
    ("r_process_star", {"Eu": 1.0, "Nd": 0.5}),
    ("young_ba_enhancement", {"Li": 2.0, "Ba": 0.4}),              # A(Li) = 2.0 + 0 + 0.96 > 2.3
    ("young_ba_enhancement", {"age": 0.3, "Ba": 0.4}),
    ("nlte_saturated_lines", {"lr_noba": 1.0}),
    ("single_element_driver", {"lr_loo_min": 2.0}),
    ("teff_peer_residual", {"fission_lr_raw": 0.5}),
    # the first real run's lessons
    ("unexplained_by_all_templates", {"reduced_chi2_best": 12.0}),      # chi2_f = 230 on 9 elements
    ("heavy_peak_incoherent", {"peer_Ce": 0.0, "peer_Nd": 0.0}),        # La alone carries the heavy peak
    ("la_cn_blend", {"sample": "giant", "teff": 4500.0, "lr_loo_driver": "La"}),
])
def test_each_veto_is_tripped_by_its_case(templates, pcfg, veto, mutate):
    cand = _passing_candidate(templates, pcfg)
    for k, v in mutate.items():
        cand.loc[0, k] = v
    vet = P.apply_vetoes(cand, cfg=pcfg, lr_threshold=pcfg.lr_min)
    assert bool(vet[f"veto_{veto}"].iloc[0]), vet.iloc[0].to_dict()
    assert not bool(vet["vet_pass"].iloc[0])
    assert veto in vet["veto_reasons"].iloc[0]
    counters = P.veto_counters(vet)
    assert counters[veto] == 1 and counters["n_pass"] == 0


def test_flagged_core_element_trips_the_quality_veto(templates, pcfg):
    cand = _passing_candidate(templates, pcfg)
    vet = P.apply_vetoes(cand, cfg=pcfg, lr_threshold=pcfg.lr_min, flagged_core=np.array([True]))
    assert bool(vet["veto_low_snr_or_flagged"].iloc[0])


def test_assemble_vectors_excludes_flagged_values_and_counts_them(pcfg):
    df = pd.DataFrame({"Nd": [0.5, 0.5], "e_Nd": [0.1, 0.1], "f_Nd": [0, 1],
                       "Ba": [0.1, 0.1], "e_Ba": [np.nan, 0.05]})
    obs, sig, flagged, info = P.assemble_vectors(df, ["Nd", "Ba", "Eu"], cfg=pcfg,
                                                 fallback_sigma={"Ba": 0.09})
    assert np.isnan(obs[1, 0]) and flagged[1, 0]
    assert not flagged[0, 0] and obs[0, 0] == 0.5
    assert abs(sig[0, 1] - np.sqrt(0.09 ** 2 + 0.05 ** 2)) < 1e-9, "missing error -> empirical scatter"
    assert "Eu" in info["missing"]
    assert np.isnan(obs[:, 2]).all()


# ---------------------------------------------------------------------------
# (e) peer residual removes an injected Teff trend
# ---------------------------------------------------------------------------
def test_peer_residual_removes_an_injected_teff_trend(templates, pcfg):
    rng = np.random.default_rng(5)
    df = make_population(rng, n=3000, teff_trend=0.35)
    raw = P.raw_vectors(df, GALAH_ELEMENTS)
    resid, scatter, notes = P.peer_residuals(df, GALAH_ELEMENTS, min_rows=200)
    teff = df["teff"].to_numpy()
    slope_raw = np.polyfit(teff / 1000.0, raw["Nd"].to_numpy(), 1)[0]
    slope_peer = np.polyfit(teff / 1000.0, resid["Nd"].to_numpy(), 1)[0]
    assert abs(slope_raw) > 0.3
    assert abs(slope_peer) < 0.05, "the Teff trend must be regressed out"
    assert notes["alpha_proxy"] is True
    # and the trend manufactured raw-space patterns that the peer residual does not carry
    frame = df.copy()
    for el in GALAH_ELEMENTS:
        frame[f"peer_{el}"] = resid[el].to_numpy()
        frame[f"raw_{el}"] = raw[el].to_numpy()
    o_raw, s_raw, _, _ = P.assemble_vectors(frame, GALAH_ELEMENTS, value_prefix="raw_", cfg=pcfg)
    o_peer, s_peer, _, _ = P.assemble_vectors(frame, GALAH_ELEMENTS, value_prefix="peer_", cfg=pcfg)
    lr_raw = P.fission_lr_only(o_raw, s_raw, templates, pcfg)
    lr_peer = P.fission_lr_only(o_peer, s_peer, templates, pcfg)
    assert (lr_raw >= pcfg.lr_min).sum() > (lr_peer >= pcfg.lr_min).sum()
    assert (lr_peer >= pcfg.lr_min).sum() <= 2


# ---------------------------------------------------------------------------
# (f) nulls and end-to-end recovery
# ---------------------------------------------------------------------------
def test_shuffled_null_rarely_makes_the_vector(templates, pcfg):
    rng = np.random.default_rng(9)
    df = make_population(rng, n=3000)
    resid, scatter, _ = P.peer_residuals(df, GALAH_ELEMENTS, min_rows=200)
    frame = df.copy()
    for el in GALAH_ELEMENTS:
        frame[f"peer_{el}"] = resid[el].to_numpy()
    obs, sig, _, _ = P.assemble_vectors(frame, GALAH_ELEMENTS, value_prefix="peer_", cfg=pcfg,
                                        fallback_sigma=scatter)
    null = P.shuffled_null(obs, sig, templates, pcfg, n_perm=2, max_rows=3000, rng=rng)
    assert null["n_rows"] == 3000 and null["n_perm"] == 2
    assert null["frac_above_lr_min"] < 0.01
    thr, why = P.derive_threshold(pcfg, null)
    assert thr >= pcfg.lr_min
    sens = P.sensitivity_curve(obs, sig, templates, pcfg, lr_threshold=thr, amplitudes=(1.0, 10.0),
                               n_inject=200, rng=rng)
    assert sens[0]["frac_lr_pass"] < sens[1]["frac_lr_pass"]
    assert sens[1]["frac_lr_and_loo_pass"] > 0.5, "a 10x Nd enrichment must be recoverable"


def _write_synthetic_checkpoint(out_dir: Path, df: pd.DataFrame, survey="GALAH"):
    from seti.tailings.acquire import write_checkpoint
    write_checkpoint(df, out_dir / f"stars_{survey.lower()}.parquet")
    (out_dir / "acquisition.json").write_text(json.dumps({
        "generated_utc": "2026-09-06T00:00:00Z",
        "surveys": [{"survey": survey, "verdict": "OK", "route": "file", "source_used": "SYNTH",
                     "n_rows": int(len(df)), "n_elements": len(GALAH_ELEMENTS),
                     "elements": GALAH_ELEMENTS, "degraded": False, "degradation": "",
                     "columns_found": {"elements": GALAH_ELEMENTS, "extras": {"flag_sp": "flag_sp"},
                                       "extras_absent": ["binary_flag"]},
                     "log": ["synthetic"]}]}))


def test_end_to_end_recovers_an_injected_fission_star_and_vetoes_the_ba_star(tmp_path, block):
    rng = np.random.default_rng(21)
    df = make_population(rng, n=2500)
    # a giant too, so the secondary sample exists
    df.loc[df.index[:300], "logg"] = rng.uniform(1.5, 3.0, 300)
    df.loc[df.index[:300], "teff"] = rng.uniform(4200.0, 5000.0, 300)
    add_pattern(df, 1000, "f", 8.0)      # the target: Nd +0.95 dex with the full shape
    add_pattern(df, 1001, "s", 4.0)      # a barium dwarf
    add_pattern(df, 1002, "r", 6.0)      # an r-II dwarf
    df.loc[1003, "Nd"] += 1.0            # a one-element spike
    _write_synthetic_checkpoint(tmp_path, df)

    cfg = load_config()
    b = json.loads(json.dumps(block))
    b["null"] = {"n_perm": 1, "max_rows": 2000, "seed": 1}
    b["sensitivity"] = {"amplitudes": [2.0, 10.0], "n_inject": 100}
    summary = R.fallout_run(cfg, stage="screen", out_dir=tmp_path, block=b)
    assert any(s["sample"] == "dwarf" for s in summary["samples"])
    summary = R.fallout_run(cfg, stage="assess", out_dir=tmp_path, block=b)

    assert summary["verdict_code"] == R.VERDICT_CANDIDATES, summary["verdict"]
    assert "generated_utc" in summary and summary["generated_utc"].endswith("Z")
    dwarf = next(s for s in summary["per_sample"] if s["sample"] == "dwarf")
    ids = [c["star_id"] for c in dwarf["survivors"]]
    assert "G0001000" in ids, dwarf
    assert "G0001001" not in ids and "G0001002" not in ids and "G0001003" not in ids
    assert dwarf["vetoes"]["n_pass"] >= 1
    cand = pd.read_csv(tmp_path / "candidates_galah_dwarf.csv")
    spike = cand[cand["star_id"] == "G0001003"]
    if len(spike):
        assert bool(spike["veto_single_element_driver"].iloc[0])
    assert (tmp_path / "REPORT.md").exists() and (tmp_path / "summary.json").exists()
    on_disk = json.loads((tmp_path / "summary.json").read_text())
    for key in ("generated_utc", "funnel", "vetoes", "acquisition", "columns_found", "templates",
                "thresholds"):
        assert key in on_disk, key
    assert on_disk["columns_found"]["GALAH"]["extras"]["flag_sp"] == "flag_sp"
    assert on_disk["funnel"]["n_survivors_dwarf"] >= 1
    assert any(s["sample"] == "giant" for s in on_disk["per_sample"])
    assert dwarf["sensitivity"][-1]["frac_lr_pass"] > 0.5


def test_a_clean_population_yields_no_fission_pattern(tmp_path, block):
    rng = np.random.default_rng(33)
    df = make_population(rng, n=1500)
    _write_synthetic_checkpoint(tmp_path, df)
    b = json.loads(json.dumps(block))
    b["null"] = {"n_perm": 1, "max_rows": 1500, "seed": 2}
    b["sensitivity"] = {"amplitudes": [3.0], "n_inject": 50}
    b["samples"] = {"dwarf": b["samples"]["dwarf"]}
    summary = R.fallout_run(load_config(), stage="all", out_dir=tmp_path, block=b,
                            inject={"route_probe_fn": lambda r: {"status": None, "error": "offline"},
                                    "probe_fn": lambda t: None, "query_fn": lambda q: None})
    # acquisition was dead, but a checkpoint from a prior run exists: the run must
    # still say the ARCHIVE gave nothing this time.
    assert summary["verdict_code"] == R.VERDICT_NO_DATA


# ---------------------------------------------------------------------------
# (g) honest degradation through the real acquisition path
# ---------------------------------------------------------------------------
def _raw_catalogue(rng, n=600):
    """A GALAH-DR4-shaped RAW table (pre-normalize): what the FITS reader returns."""
    df = make_population(rng, n=n)
    raw = pd.DataFrame({"sobject_id": np.arange(n), "teff": df["teff"], "logg": df["logg"],
                        "fe_h": df["fe_h"], "snr_px_ccd3": df["snr"], "flag_sp": 0,
                        "age_bstep": df["age"], "ruwe": 1.0})
    for el in GALAH_ELEMENTS + ALPHA:
        raw[f"{el.lower()}_fe"] = df[el]
        raw[f"e_{el.lower()}_fe"] = df[f"e_{el}"]
        raw[f"flag_{el.lower()}_fe"] = 0
    raw["li_fe"] = df["Li"]
    raw.attrs["n_rows_file"] = n
    return raw


def _probe(live: set[str]):
    def fn(url: str) -> dict:
        if url in live:
            return {"status": 200, "content_length": 700_000_000, "content_type": "image/fits",
                    "accept_ranges": "bytes", "final_url": url, "error": None}
        return {"status": 404, "content_length": None, "content_type": None, "error": "HTTP 404"}
    return fn


def test_dead_routes_give_no_data_reached(tmp_path, block):
    summary = R.fallout_run(load_config(), stage="all", out_dir=tmp_path, block=block,
                            inject={"route_probe_fn": _probe(set()),
                                    "probe_fn": lambda t: None, "query_fn": lambda q: None})
    assert summary["verdict_code"] == R.VERDICT_NO_DATA
    assert summary["verdict"].startswith(R.VERDICT_NO_DATA)
    assert summary["acquisition"]["surveys"][0]["verdict"] == "NO_DATA_REACHED"
    assert not (tmp_path / "stars_galah.parquet").exists()
    assert "generated_utc" in json.loads((tmp_path / "summary.json").read_text())


def test_fallback_route_is_reported_as_degraded_source_with_the_log(tmp_path, block):
    rng = np.random.default_rng(8)
    raw = _raw_catalogue(rng, n=600)
    routes = FA.routes_from_config(block, "GALAH")
    live = {routes[1].url}                       # the preferred URL is dead, the second answers
    b = json.loads(json.dumps(block))
    b["null"] = {"n_perm": 1, "max_rows": 600, "seed": 3}
    b["sensitivity"] = {"amplitudes": [5.0], "n_inject": 40}
    b["samples"] = {"dwarf": {**b["samples"]["dwarf"], "teff_min": 4400.0}}
    b["peer"] = {**b["peer"], "min_rows": 100}
    summary = R.fallout_run(load_config(), stage="all", out_dir=tmp_path, block=b, max_rows=10_000,
                            inject={"route_probe_fn": _probe(live),
                                    "read_fn": lambda url, sel: raw,
                                    "probe_fn": lambda t: None, "query_fn": lambda q: None})
    acq = summary["acquisition"]["surveys"][0]
    assert acq["verdict"] == "OK" and acq["degraded"]
    assert "rather than the preferred" in acq["degradation"]
    assert summary["verdict_code"] == R.VERDICT_DEGRADED
    assert summary["verdict"].startswith(R.VERDICT_DEGRADED)
    # what was found is on the record
    cf = summary["columns_found"]["GALAH"]
    assert set(GALAH_ELEMENTS) <= set(cf["elements"])
    assert cf["extras"]["flag_sp"] == "flag_sp" and cf["extras"]["age"] == "age_bstep"
    assert "binary_flag" in cf["extras_absent"]
    assert cf["flag_columns"]["Nd"] == "f_Nd"
    assert any("extra columns attached" in line for line in acq["log"])
    stars = pd.read_parquet(tmp_path / "stars_galah.parquet")
    assert "flag_sp" in stars.columns and "age" in stars.columns
    assert summary["funnel"]["n_rows_acquired"]["GALAH"] == 600
    assert "GALAH/dwarf" in summary["vetoes"]


def test_routes_prefer_the_runner_proven_cloud_url(block):
    routes = FA.routes_from_config(block, "GALAH")
    assert routes[0].url.startswith("https://cloud.datacentral.org.au/")
    assert "galah_dr4_allstar" in routes[0].url
    urls = [r.url for r in routes]
    assert len(urls) == len(set(urls)), "no duplicate route"
    assert any("GALAH_DR3" in r.url for r in routes), "TAILINGS' registry is appended as fallback"


def test_extra_column_discovery_reports_absence():
    found = FA.resolve_extra_columns(["sobject_id", "flag_sp", "AGE_BSTEP", "e_rv_comp_1"])
    assert found["flag_sp"] == "flag_sp" and found["age"] == "AGE_BSTEP"
    assert found["rv_err"] == "e_rv_comp_1" and found["binary_flag"] is None


def test_split_samples_applies_both_boxes(block):
    rng = np.random.default_rng(1)
    df = make_population(rng, n=200)
    df.loc[:49, "logg"] = 2.5
    df.loc[:49, "teff"] = 4700.0
    parts = FA.split_samples(df, block["samples"])
    assert set(parts) == {"dwarf", "giant"}
    assert len(parts["giant"]) == 50 and len(parts["dwarf"]) == 150
    assert (parts["dwarf"]["logg"] > 4.0).all() and (parts["giant"]["logg"] < 3.5).all()


# ---------------------------------------------------------------------------
# entry points, config and workflow
# ---------------------------------------------------------------------------
def test_entry_points_expose_the_four_stages():
    import argparse

    assert R.STAGES == ("probe", "acquire", "screen", "assess", "vet", "all")
    top = argparse.ArgumentParser()
    sub = top.add_subparsers()
    R.register(sub)
    args = top.parse_args(["fallout", "--stage", "probe", "--surveys", "GALAH"])
    assert args.stage == "probe" and args.func is R._cmd_fallout
    with pytest.raises(SystemExit):
        top.parse_args(["fallout", "--stage", "bogus"])
    with pytest.raises(ValueError):
        R.fallout_run(Config(root=Path("."), thresholds={}, catalogs={}, paths={}), stage="bogus")


def test_probe_stage_writes_the_route_report(tmp_path, block):
    rep = R.fallout_run(load_config(), stage="probe", out_dir=tmp_path, block=block,
                        inject={"route_probe_fn": _probe(set())})
    assert (tmp_path / "probe.json").exists()
    assert rep["surveys"]["GALAH"]["n_live"] == 0 and rep["surveys"]["GALAH"]["n_routes"] >= 3


def test_config_mirrors_the_pattern_config(block, pcfg):
    assert pcfg.lr_min == block["pattern"]["lr_min"]
    assert tuple(block["pattern"]["core_elements"]) == pcfg.core_elements
    assert block["samples"]["dwarf"]["logg_min"] >= 4.0
    assert 4500.0 <= block["samples"]["dwarf"]["teff_min"] < block["samples"]["dwarf"]["teff_max"] <= 6300.0
    assert block["samples"]["giant"]["weight"] == "secondary"


def test_workflow_is_dispatchable_and_commits_through_the_verified_script():
    path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "fallout.yml"
    doc = yaml.safe_load(path.read_text())
    triggers = doc.get(True, doc.get("on"))
    assert "workflow_dispatch" in triggers
    assert doc["permissions"]["contents"] == "write"
    text = path.read_text()
    assert "scripts/commit_results.sh" in text
    assert "scripts/falloutlit_fetch.py" in text
    assert "seti.fallout.run" in text
    assert "tests/test_fallout.py" in text
    assert "results/fallout/vet.json" in text


# ---------------------------------------------------------------------------
# What the first real GALAH DR4 run taught (2026-09-06)
# ---------------------------------------------------------------------------
def test_error_floor_inflates_underestimated_errors_and_is_recorded(pcfg):
    df = pd.DataFrame({"Nd": [0.3, 0.2], "e_Nd": [0.02, 0.03], "Ce": [0.1, 0.1], "e_Ce": [0.05, 0.05]})
    floors = P.error_floors(df, ["Nd", "Ce"], {"Nd": 0.16, "Ce": 0.28}, cfg=pcfg)
    assert floors["Nd"]["floor_dex"] == 0.16 and floors["Nd"]["median_quoted_dex"] == 0.025
    assert floors["Nd"]["inflation"] == 6.4 and floors["Ce"]["inflation"] == 5.6
    obs, sig, _, _ = P.assemble_vectors(df, ["Nd", "Ce"], cfg=pcfg,
                                        sigma_floor={el: d["floor_dex"] for el, d in floors.items()})
    assert abs(sig[0, 0] - np.sqrt(0.16 ** 2 + pcfg.systematic_floor_dex ** 2)) < 1e-9
    assert abs(sig[0, 1] - np.sqrt(0.28 ** 2 + pcfg.systematic_floor_dex ** 2)) < 1e-9
    # mode "none" leaves the quoted error alone
    off = P.PatternConfig(error_floor_mode="none")
    floors0 = P.error_floors(df, ["Nd"], {"Nd": 0.16}, cfg=off)
    assert floors0["Nd"]["floor_dex"] == 0.0


def test_underestimated_errors_do_not_make_a_candidate_but_a_real_fission_star_still_is(templates, pcfg):
    """A noise vector at 3 sigma of an under-quoted error looks like a pattern until
    the error is floored at the measured scatter; a genuine strong injection survives
    the flooring."""
    rng = np.random.default_rng(2026)
    K = len(GALAH_ELEMENTS)
    true_scatter = np.array([0.40, 0.34, 0.08, 0.23, 0.15, 0.31, 0.10, 0.22, 0.28, 0.16, 0.22, 0.32])
    quoted = true_scatter / 4.0                       # the real-run ratio
    n = 2000
    noise = rng.normal(0, 1, (n, K)) * true_scatter
    sig_quoted = np.tile(np.sqrt(quoted ** 2 + 0.05 ** 2), (n, 1))
    sig_floored = np.tile(np.sqrt(true_scatter ** 2 + 0.05 ** 2), (n, 1))
    lr_q = P.fission_lr_only(noise, sig_quoted, templates, pcfg)
    lr_f = P.fission_lr_only(noise, sig_floored, templates, pcfg)
    rate_q = (lr_q >= pcfg.lr_min).mean()
    rate_f = (lr_f >= pcfg.lr_min).mean()
    assert rate_q > 0.01, "under-quoted errors manufacture 'patterns' from pure noise"
    assert rate_f < 0.002, "floored errors do not"
    assert rate_q > 5 * max(rate_f, 1.0 / n)
    # the chi2 scale is what moved: the same vectors, the same shapes, 16x the statistic
    assert np.median(lr_q[lr_q > 0]) > 5 * np.median(lr_f[lr_f > 0])
    # the genuine star: fission at a_f = 20 (Nd +1.3 dex) on top of the same, dwarf-scale
    # noise. At a_f = 10 the honest completeness against 0.16-0.40 dex scatter is ~50%,
    # which is what the sensitivity curve is for; the test asks for the unambiguous case.
    inj = noise[:60] + np.log10(1 + 20.0 * templates.F)[None, :]
    fit = P.fit_patterns(inj, sig_floored[:60], templates, pcfg)
    cls = fit["classification"]
    # Against 0.3-0.4 dex scatter on half the elements, ~1 in 5 lands AMBIGUOUS
    # (fission best, ln LR below 8): that is the honest completeness, and the
    # sensitivity curve is where it is reported. The star must be PREFERRED as
    # fission almost always, and unambiguously so most of the time.
    assert ((cls == P.FISSION) | (cls == P.AMBIGUOUS)).mean() >= 0.9, cls.value_counts().to_dict()
    assert (cls == P.FISSION).mean() >= 0.7, cls.value_counts().to_dict()
    assert (fit["fission_lr"] > 0).mean() >= 0.9
    loo = P.leave_one_out(inj, sig_floored[:60], templates, pcfg)
    # Leave-one-out at the full threshold roughly halves completeness at this
    # noise level (measured ~47% here); the floor below is the honest one and
    # the per-sample curve in summary.json is where the real number lives.
    assert (loo["lr_loo_min"] >= pcfg.lr_min).mean() >= 0.35
    assert (loo["lr_loo_min"] > 0).mean() >= 0.8, "the preference does not COLLAPSE without any one element"


def test_a_vector_nothing_fits_is_unexplained_not_fission(templates, pcfg):
    """The real giant survivor: La +1.2, Sr -0.85, Sm negative, chi2_f = 230 on 9 elements."""
    obs = np.full((1, len(GALAH_ELEMENTS)), np.nan)
    for el, v in {"Sr": -1.0, "Y": -0.1, "Zr": 0.0, "Ba": -0.15, "La": 1.2, "Ce": 0.2,
                  "Nd": 0.3, "Sm": -0.45, "Eu": 0.0}.items():
        obs[0, templates.index(el)] = v
    sig = np.full_like(obs, np.sqrt(0.10 ** 2 + 0.05 ** 2))
    res = P.fit_patterns(obs, sig, templates, pcfg)
    assert res["fission_lr"].iloc[0] > pcfg.lr_min, "it 'wins' against the natural models"
    assert res["reduced_chi2_best"].iloc[0] > pcfg.max_reduced_chi2, "but nothing fits it"
    assert res["classification"].iloc[0] == P.UNEXPLAINED
    cand = pd.concat([pd.DataFrame([{"sample": "giant", "teff": 4780.0}]), res], axis=1)
    vet = P.apply_vetoes(cand, cfg=pcfg, lr_threshold=pcfg.lr_min)
    assert bool(vet["veto_unexplained_by_all_templates"].iloc[0])
    assert not bool(vet["vet_pass"].iloc[0])


def test_la_diagnostics_flags_a_cn_tracking_residual_and_clears_a_clean_one(pcfg):
    rng = np.random.default_rng(4)
    n = 2000
    N = rng.normal(0.0, 0.3, n)
    teff = rng.uniform(4000.0, 5200.0, n)
    tracking = pd.DataFrame({"peer_La": 0.4 * N + rng.normal(0, 0.1, n), "N": N, "C": rng.normal(0, 0.2, n),
                             "teff": teff, "logg": rng.uniform(1.5, 3.5, n), "vsini": rng.uniform(1, 8, n)})
    d = P.la_diagnostics(tracking, cfg=pcfg)
    assert d["la_cn_suspect"] and "N rho=" in d["reason"]
    assert abs(d["correlations"]["N"]) > 0.5
    clean = tracking.copy()
    clean["peer_La"] = rng.normal(0, 0.1, n)
    d2 = P.la_diagnostics(clean, cfg=pcfg)
    assert not d2["la_cn_suspect"]
    # not computable -> distrusted, and says why
    d3 = P.la_diagnostics(clean[["peer_La"]], cfg=pcfg)
    assert d3["la_cn_suspect"] and "no covariate" in d3["reason"]
    d4 = P.la_diagnostics(clean.head(20), cfg=pcfg)
    assert d4["la_cn_suspect"] and "need" in d4["reason"]


def test_la_cn_blend_veto_needs_giant_cool_la_carried_and_suspect(templates, pcfg):
    base = _passing_candidate(templates, pcfg)
    base.loc[0, "sample"] = "giant"
    base.loc[0, "teff"] = 4500.0
    base.loc[0, "lr_loo_driver"] = "La"
    assert bool(P.apply_vetoes(base, cfg=pcfg, lr_threshold=pcfg.lr_min)["veto_la_cn_blend"].iloc[0])
    # not suspect in this sample -> no veto
    assert not bool(P.apply_vetoes(base, cfg=pcfg, lr_threshold=pcfg.lr_min,
                                   la_cn_suspect=False)["veto_la_cn_blend"].iloc[0])
    # a warm giant, or a dwarf, is not vetoed
    warm = base.copy()
    warm.loc[0, "teff"] = 5100.0
    assert not bool(P.apply_vetoes(warm, cfg=pcfg, lr_threshold=pcfg.lr_min)["veto_la_cn_blend"].iloc[0])
    dwarf = base.copy()
    dwarf.loc[0, "sample"] = "dwarf"
    assert not bool(P.apply_vetoes(dwarf, cfg=pcfg, lr_threshold=pcfg.lr_min)["veto_la_cn_blend"].iloc[0])


def test_heavy_peak_coherence_counts_elements_individually(templates, pcfg):
    cand = _passing_candidate(templates, pcfg)
    vet = P.apply_vetoes(cand, cfg=pcfg, lr_threshold=pcfg.lr_min)
    assert int(vet["n_heavy_coherent"].iloc[0]) == 3 and not bool(vet["veto_heavy_peak_incoherent"].iloc[0])
    one = cand.copy()
    one.loc[0, "peer_Ce"] = 0.05
    one.loc[0, "peer_Nd"] = -0.05
    v1 = P.apply_vetoes(one, cfg=pcfg, lr_threshold=pcfg.lr_min)
    assert int(v1["n_heavy_coherent"].iloc[0]) == 1 and bool(v1["veto_heavy_peak_incoherent"].iloc[0])
    # no sigma information at all -> cannot be shown coherent -> vetoed, not waved through
    nosig = cand.drop(columns=[c for c in cand.columns if c.startswith("sig_")])
    assert bool(P.apply_vetoes(nosig, cfg=pcfg, lr_threshold=pcfg.lr_min)["veto_heavy_peak_incoherent"].iloc[0])


def test_sensitivity_is_conditioned_on_testable_stars(templates, pcfg):
    rng = np.random.default_rng(12)
    n = 400
    obs = rng.normal(0, 1, (n, len(GALAH_ELEMENTS))) * 0.1
    sig = np.full_like(obs, np.sqrt(0.1 ** 2 + 0.05 ** 2))
    # half the stars have only Y, Zr, Ba measured: untestable
    for el in GALAH_ELEMENTS:
        if el not in ("Y", "Zr", "Ba"):
            obs[: n // 2, templates.index(el)] = np.nan
    mask = P.testable_mask(obs, templates, pcfg)
    assert mask.sum() == n // 2 and not mask[: n // 2].any()
    sens = P.sensitivity_curve(obs, sig, templates, pcfg, lr_threshold=pcfg.lr_min,
                               amplitudes=(10.0,), n_inject=400, rng=rng, with_loo=False)
    r = sens[0]
    assert abs(r["testable_fraction"] - 0.5) < 1e-9 and r["n_testable"] == n // 2
    assert r["frac_lr_pass_testable"] > 0.9
    assert r["frac_lr_pass_all"] < 0.6, "the untestable half caps the all-star number"
    assert r["frac_lr_pass"] == r["frac_lr_pass_all"], "the legacy column is the all-star number"


def _real_shaped_candidates(templates, elements):
    """A candidates CSV shaped like the first real run's: peer_ columns, no sig_ columns,
    two La-driven 'survivors' and one genuine fission star."""
    rows = []
    def row(star_id, sample, teff, vec, chi2_f, chi2_nat, n_meas, vet_pass):
        r = {"star_id": star_id, "sample": sample, "teff": teff, "logg": 2.5 if sample == "giant" else 4.4,
             "fe_h": -0.1, "snr": 60.0, "flag_sp": 0, "n_measured": n_meas, "chi2_f": chi2_f,
             "chi2_natural": chi2_nat, "chi2_null": chi2_nat + 20, "fission_lr": 0.5 * (chi2_nat - chi2_f),
             "enrich_lr": 40.0, "lr_noba": 30.0, "lr_loo_min": 12.0, "lr_loo_driver": "La",
             "fission_lr_raw": 30.0, "a_f": 2.0, "classification": "FISSION", "natural_class": "S_PLUS_R",
             "vet_pass": vet_pass, "first_veto": "" if vet_pass else "single_element_driver",
             "veto_reasons": "", "flagged_core": False}
        for el in elements:
            r[el] = vec.get(el, np.nan)
            r[f"peer_{el}"] = vec.get(el, np.nan)
        for el in ("Rb", "Sr", "Y", "Zr", "Mo", "Ru", "Ba", "La", "Ce", "Nd", "Sm", "Eu"):
            r.setdefault(el, np.nan)
        return r
    rows.append(row("170203001601307", "giant", 4784.0,
                    {"Rb": -0.24, "Sr": -1.01, "Y": -0.12, "Zr": 0.02, "Ba": -0.16, "La": 0.89,
                     "Ce": 0.20, "Nd": 0.31, "Sm": -0.45, "Eu": -0.01}, 229.7, 334.6, 9, True))
    rows.append(row("230511003401363", "giant", 4451.0,
                    {"Y": -0.06, "Zr": -0.13, "Ba": 0.18, "La": 0.77, "Ce": 0.22, "Nd": 0.48,
                     "Sm": 0.23, "Eu": -0.14}, 52.5, 106.4, 8, True))
    real = {el: float(np.log10(1 + 8.0 * templates.F[templates.index(el)])) for el in elements}
    rows.append(row("G_FISSION", "dwarf", 5600.0, real, 1.0, 120.0, 12, True))
    return pd.DataFrame(rows)


def test_vet_stage_rebuilds_sigmas_from_recorded_floors_and_vetoes_the_la_driven_giants(tmp_path, block, templates):
    elements = GALAH_ELEMENTS
    cand = _real_shaped_candidates(templates, elements)
    floors = {"Rb": 0.17, "Sr": 0.14, "Y": 0.08, "Zr": 0.09, "Mo": 0.12, "Ru": 0.16, "Ba": 0.12,
              "La": 0.10, "Ce": 0.10, "Nd": 0.09, "Sm": 0.09, "Eu": 0.19}
    for sample in ("dwarf", "giant"):
        cand[cand["sample"] == sample].to_csv(tmp_path / f"candidates_galah_{sample}.csv", index=False)
    per_sample = []
    for sample in ("dwarf", "giant"):
        per_sample.append({"survey": "GALAH", "sample": sample, "n_stars": 1000, "n_elements": 12,
                           "elements": elements, "threshold": {"lr_used": 9.7, "lr_min_config": 8.0,
                                                              "enrich_min": 12.5},
                           "error_model": {"per_element": {el: {"floor_dex": v, "median_quoted_dex": v / 3,
                                                                "inflation": 3.0} for el, v in floors.items()}},
                           "la_diagnostics": {"la_cn_suspect": True, "reason": "La residual tracks N rho=+0.31"},
                           "n_survivors": 2 if sample == "giant" else 1, "n_pass_lr": 3, "vetoes": {}})
    summary = {"channel": "fallout", "generated_utc": "2026-09-06T13:42:27Z", "verdict_code": R.VERDICT_CANDIDATES,
               "verdict": "x", "per_sample": per_sample, "templates": {"templates": templates.to_dict(1.0)},
               "acquisition": {"surveys": [{"survey": "GALAH", "verdict": "OK", "n_rows": 1000,
                                            "degraded": False, "columns_found": {}}]}}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "templates.json").write_text(json.dumps({"templates": templates.to_dict(1.0)}))

    out = R.fallout_run(load_config(), stage="vet", out_dir=tmp_path, block=block)
    vet = json.loads((tmp_path / "vet.json").read_text())
    g = vet["samples"]["GALAH/giant"]
    assert "rebuilt from recorded" in g["sigma"]["sigma_source"] and g["sigma"]["refit"]
    assert g["n_survivors_screen"] == 2 and g["n_survivors_vetted"] == 0
    stars = {s["star_id"]: s for s in vet["stars"]}
    for sid in ("170203001601307", "230511003401363"):
        assert not stars[sid]["vet_pass"]
        assert any(v in stars[sid]["veto_reasons"] for v in
                   ("unexplained_by_all_templates", "heavy_peak_incoherent", "la_cn_blend",
                    "single_element_driver"))
    assert stars["G_FISSION"]["vet_pass"], stars["G_FISSION"]
    assert vet["samples"]["GALAH/dwarf"]["n_survivors_vetted"] == 1
    assert out["verdict_code"] == R.VERDICT_CANDIDATES
    assert out["funnel"]["n_survivors_vetted_giant"] == 0 and out["funnel"]["n_survivors_vetted_dwarf"] == 1
    assert "1 cool-dwarf and 0 giant" in out["verdict"]
    assert (tmp_path / "vetted_galah_giant.csv").exists() and (tmp_path / "REPORT.md").exists()
    assert "Vet stage" in (tmp_path / "REPORT.md").read_text()


def test_concept_scan_separates_the_target_from_its_decoys(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "falloutlit_fetch", Path(__file__).resolve().parents[1] / "scripts" / "falloutlit_fetch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def entry(aid, title, summary):
        return f"<entry><id>http://arxiv.org/abs/{aid}</id><title>{title}</title><summary>{summary}</summary></entry>"
    atom = "<feed>" + entry("1111.0001", "A target",
                            "We fit the fission product yield pattern to the photospheric abundances of a G dwarf.") \
        + entry("1111.0002", "A decoy", "Fission product yields of U-235 for reactor antineutrino "
                                        "spectra, with a stellar neutrino background.") \
        + entry("1111.0003", "Adjacent", "Fission fragment distributions shape the r-process abundance pattern in kilonova ejecta.") \
        + entry("1111.0004", "Unrelated", "The s-process in AGB stars and the solar abundance decomposition.") + "</feed>"
    (tmp_path / "arxiv_q_test.atom").write_text(atom)
    res = mod.scan(tmp_path)
    assert res["n_abstracts_scanned"] == 4 and res["n_target_regex_hits"] == 3
    assert [h["arxiv"] for h in res["decoy_free_hits"]] == ["http://arxiv.org/abs/1111.0001"]
    assert [h["arxiv"] for h in res["nucleosynthesis_adjacent_hits"]] == ["http://arxiv.org/abs/1111.0003"]
    assert (tmp_path / "concept_scan.json").exists()
