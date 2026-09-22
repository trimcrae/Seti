"""Offline suite for the ULINE asymmetric-top predictor and its literature ladder.

Nothing here opens a socket: every fetch takes an injected callable.  The
physics is checked against things that are true independently of this code ---
closed-form rigid-rotor energies, the exact symmetric-top limit, the dipole
selection rules, the line-strength sum rule, invariance under the choice of
representation --- and then against a catalogue the code did not generate.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import pytest

from seti.uline import litfetch as LF
from seti.uline import rotorpred as RP
from seti.uline.lines import CATDIR_TEMPS
from seti.uline.rotor import (
    DISTORTION_SCALE_REL,
    QUARTIC,
    RotorConstants,
    compare_with_cat,
    fit_constants,
    hamiltonian_matrix,
    hyperfine_width_mhz,
    levels,
    line_strengths,
    parse_jpl_doc,
    partition_function,
    predict_lines,
    rigid_rotor_energies_j1,
    spin_weight,
    wang_blocks,
)

# ---------------------------------------------------------------------------
# reference constants (only ever used to DRIVE the code, never as an answer)
# ---------------------------------------------------------------------------
SO2 = RotorConstants(A=60778.5522, B=10318.0736, C=8799.7023, representation="Ir",
                     DJ=0.0065766, DJK=-0.117054, DK=2.5898, dJ=0.0017017, dK=0.02534,
                     mu_b=1.633, spin_weights={"rule": "ka_kc_sum", "even": 1, "odd": 0},
                     name="SO2")
ASYM = RotorConstants(A=15000.0, B=9000.0, C=6000.0, representation="Ir",
                      mu_a=1.0, mu_b=1.0, mu_c=1.0, name="generic")


# ---------------------------------------------------------------------------
# the Hamiltonian
# ---------------------------------------------------------------------------
def test_j1_energies_match_the_closed_form():
    """E(1_01) = B+C, E(1_11) = A+C, E(1_10) = A+B — exactly, for any asymmetry."""
    lv = levels(ASYM, 1, rigid=True)
    want = rigid_rotor_energies_j1(ASYM)
    got = {f"1{ka}{kc}": e for ka, kc, e in zip(lv.ka, lv.kc, lv.energy, strict=True)}
    for k, v in want.items():
        assert got[k] == pytest.approx(v, abs=1e-9)


def test_j2_trace_is_the_rigid_rotor_trace():
    """Σ E(J=2) = 5·(A+B+C)·J(J+1)/... — the trace of H is basis-independent."""
    lv = levels(ASYM, 2, rigid=True)
    h = hamiltonian_matrix(ASYM, 2, rigid=True)
    assert float(lv.energy.sum()) == pytest.approx(float(np.trace(h)), rel=1e-12)


def test_exact_symmetric_top_limit_with_distortion():
    """A = B: E = B J(J+1) + (C−B)K² − ΔJ J²(J+1)² − ΔJK J(J+1)K² − ΔK K⁴."""
    c = RotorConstants(A=5000.0, B=5000.0, C=3000.0, representation="IIIr",
                       DJ=1e-3, DJK=2e-3, DK=3e-3, dJ=0.0, dK=0.0, mu_c=1.0)
    for j in (1, 3, 7):
        lv = levels(c, j)
        for ka, kc, e in zip(lv.ka, lv.kc, lv.energy, strict=True):
            k = kc                                   # z = c in IIIr
            jj = j * (j + 1)
            want = (5000.0 * jj + (3000.0 - 5000.0) * k * k
                    - 1e-3 * jj ** 2 - 2e-3 * jj * k * k - 3e-3 * k ** 4)
            assert float(e) == pytest.approx(want, abs=1e-6), (j, ka, kc)


def test_labels_obey_the_tau_ordering_theorem():
    """Within one J the RIGID-rotor energies rise monotonically with τ = Ka − Kc.

    Stated for the rigid rotor because that is where it is a theorem (King,
    Hainer & Cross 1943).  Centrifugal distortion can reorder the near-degenerate
    high-K doublets — SO₂ at J = 9 does, ΔK = 2.59 MHz moving the K = 9 pair by
    ~17 GHz — which is exactly why the labels come from the Wang-block symmetry
    and not from sorting the energies.
    """
    for j in (2, 5, 9):
        lv = levels(ASYM, j, rigid=True)
        assert list(lv.ka - lv.kc) == sorted(lv.ka - lv.kc)
        # every (Ka, Kc) of this J appears exactly once
        want = {(ka, j - ka + off) for ka in range(j + 1)
                for off in ((0,) if ka in (0,) else (0, 1)) if 0 <= j - ka + off <= j}
        assert set(zip(lv.ka, lv.kc, strict=True)) == want
        assert len(lv.ka) == 2 * j + 1


def test_high_k_labels_survive_distortion_reordering():
    """SO₂ with ΔK on: the label set is still complete and each appears once."""
    lv = levels(SO2, 9)
    assert len(set(zip(lv.ka, lv.kc, strict=True))) == 19
    assert all(0 <= ka <= 9 and 0 <= kc <= 9 and ka + kc in (9, 10)
               for ka, kc in zip(lv.ka, lv.kc, strict=True))


def test_wang_blocks_are_orthonormal_and_complete():
    for j in (0, 1, 4, 7):
        cols = np.concatenate([u for _, u in wang_blocks(j)], axis=1)
        assert cols.shape == (2 * j + 1, 2 * j + 1)
        assert np.allclose(cols.T @ cols, np.eye(2 * j + 1), atol=1e-12)


def test_representation_invariance():
    """The SPECTRUM does not know which axis was called z (rigid rotor)."""
    a = RotorConstants(A=15000.0, B=9000.0, C=6000.0, representation="Ir", mu_a=1.0)
    b = RotorConstants(A=15000.0, B=9000.0, C=6000.0, representation="IIIr", mu_a=1.0)
    for j in (1, 3, 6):
        ea = np.sort(levels(a, j, rigid=True).energy)
        eb = np.sort(levels(b, j, rigid=True).energy)
        assert np.allclose(ea, eb, atol=1e-6)


# ---------------------------------------------------------------------------
# dipole matrix elements
# ---------------------------------------------------------------------------
def test_line_strength_sum_rule():
    """Σ_{J'τ'} S_g(J'τ' ← Jτ) = 2J+1 over J' = J−1, J, J+1, for each g."""
    j = 4
    lo = levels(ASYM, j)
    tot = {"z": np.zeros(2 * j + 1), "x": np.zeros(2 * j + 1), "y": np.zeros(2 * j + 1)}
    for jp in (j - 1, j, j + 1):
        if jp < 0:
            continue
        s = line_strengths(lo, levels(ASYM, jp))
        for g in tot:
            tot[g] += s[g].sum(axis=0)
    for g, v in tot.items():
        assert np.allclose(v, 2 * j + 1, rtol=1e-10), g


def test_selection_rules_a_b_c():
    """a-type ee↔eo (ΔKa even, ΔKc odd), b-type oo, c-type oe — on the parities."""
    j = 3
    lo, up = levels(ASYM, j), levels(ASYM, j + 1)
    s = line_strengths(lo, up)
    axis = {"z": "a", "x": "b", "y": "c"}          # Ir: z=a, x=b, y=c
    rule = {"a": (0, 1), "b": (1, 1), "c": (1, 0)}
    for g, ax in axis.items():
        dka_par, dkc_par = rule[ax]
        iu, il = np.nonzero(s[g] > 1e-10)
        assert len(iu)
        for a_, b_ in zip(iu, il, strict=True):
            assert abs(int(up.ka[a_]) - int(lo.ka[b_])) % 2 == dka_par, (g, ax)
            assert abs(int(up.kc[a_]) - int(lo.kc[b_])) % 2 == dkc_par, (g, ax)


def test_symmetric_top_strength_matches_the_textbook():
    """S = ((J+1)² − K²)/(J+1) for a parallel-band R branch of a symmetric top."""
    c = RotorConstants(A=5000.0, B=5000.0, C=3000.0, representation="IIIr", mu_c=1.0)
    j = 5
    lo, up = levels(c, j), levels(c, j + 1)
    s = line_strengths(lo, up)["z"]
    for k in range(0, j + 1):
        il = int(np.flatnonzero(lo.kc == k)[0])
        iu = int(np.flatnonzero(up.kc == k)[0])
        want = ((j + 1) ** 2 - k * k) / (j + 1)
        # the K-doublet splits the strength between two degenerate levels
        got = s[:, il].sum() if k else s[iu, il]
        if k:
            got = s[np.flatnonzero(up.kc == k), il].sum()
        assert float(got) == pytest.approx(want, rel=1e-9), k


# ---------------------------------------------------------------------------
# spin weights, partition function, hyperfine width
# ---------------------------------------------------------------------------
def test_spin_weight_rules():
    assert spin_weight({"rule": "none"}, 3, 4) == 1.0
    assert spin_weight({"rule": "ka_kc_sum", "even": 10, "odd": 6}, 1, 3) == 10.0
    assert spin_weight({"rule": "ka_kc_sum", "even": 10, "odd": 6}, 1, 2) == 6.0
    assert spin_weight({"rule": "ka", "even": 1, "odd": 3}, 2, 5) == 1.0
    assert spin_weight({"rule": "k_mod3", "axis": "a", "multiple": 2.0, "other": 1.0}, 3, 1) == 2.0
    assert spin_weight({"rule": "k_mod3", "axis": "a", "multiple": 2.0, "other": 1.0}, 4, 1) == 1.0
    assert spin_weight({"rule": "k_mod3", "axis": "c", "multiple": 2.0, "other": 1.0}, 4, 6) == 2.0
    with pytest.raises(ValueError):
        spin_weight({"rule": "nonsense"}, 1, 1)


def test_partition_function_matches_the_classical_formula():
    """Q ≈ 5.3311e6 √(T³/ABC) / σ — the explicit sum must land within a few %."""
    qlog, _ = partition_function(SO2, [300.0])
    classical = 5.3311e6 * math.sqrt(300.0 ** 3 / (SO2.A * SO2.B * SO2.C)) / 2.0
    assert 10.0 ** qlog[0] == pytest.approx(classical, rel=0.05)


def test_hyperfine_width_vanishes_when_the_bracket_does_not_change():
    assert hyperfine_width_mhz(75.0, 2, 10, 0, 10, 0) == pytest.approx(0.0, abs=1e-12)
    assert hyperfine_width_mhz(75.0, 2, 5, 0, 5, 5) > 1.0


# ---------------------------------------------------------------------------
# the spectrum
# ---------------------------------------------------------------------------
def test_so2_anchor_line():
    """SO₂ 3(1,3)–2(0,2) at 104029.418 MHz — a frequency this code did not make.

    A single well-known laboratory anchor: the standard hot-core SO₂ line.  It
    catches a sign error, a wrong axis assignment or a units slip in one number.
    """
    df, _ = predict_lines(SO2, fmin_mhz=100000.0, fmax_mhz=110000.0, lgint_floor=-8.0)
    hit = df[(df.j_up == 3) & (df.ka_up == 1) & (df.kc_up == 3)
             & (df.j_lo == 2) & (df.ka_lo == 0) & (df.kc_lo == 2)]
    assert len(hit) == 1
    assert float(hit.freq_mhz.iloc[0]) == pytest.approx(104029.418, abs=0.5)


def test_predicted_table_has_the_cat_schema_and_is_sorted():
    df, ent = predict_lines(SO2, fmin_mhz=80000.0, fmax_mhz=300000.0, lgint_floor=-7.0)
    for col in ("freq_mhz", "err_mhz", "lgint_300", "elo_cm", "gup", "qn_up", "qn_lo"):
        assert col in df.columns
    assert list(df.freq_mhz) == sorted(df.freq_mhz)
    assert (df.err_mhz > 0).all()
    assert len(ent.qlog) == len(CATDIR_TEMPS)
    assert (df.elo_cm >= 0).all()


def test_spin_weight_zero_removes_the_level_entirely():
    """SO₂ has no Ka+Kc odd levels (¹⁶O spin 0): no line may touch one."""
    df, _ = predict_lines(SO2, fmin_mhz=80000.0, fmax_mhz=200000.0, lgint_floor=-7.0)
    assert ((df.ka_up + df.kc_up) % 2 == 0).all()
    assert ((df.ka_lo + df.kc_lo) % 2 == 0).all()


def test_unknown_quartic_constants_widen_the_error_enormously():
    """The point of the error model: no quartic ⇒ not searchable at survey precision.

    Checked on a **Q branch** as well as an R branch, because the ΔJ term
    cancels in a Q branch and an error model built on it alone called those
    lines accurate when they are not — the ΔJK and ΔK terms are what catch them.
    """
    blind = RotorConstants.from_dict({**SO2.as_dict(), **{k: None for k in QUARTIC}})
    for lo, hi in ((140000.0, 160000.0), (200000.0, 260000.0)):
        known, _ = predict_lines(SO2, fmin_mhz=lo, fmax_mhz=hi, lgint_floor=-7.0)
        unknown, _ = predict_lines(blind, fmin_mhz=lo, fmax_mhz=hi, lgint_floor=-7.0)
        assert len(known) and float(known.err_mhz.median()) < 10.0
        assert float(unknown.err_mhz.median()) > 50.0 * float(known.err_mhz.median())
    q = predict_lines(blind, fmin_mhz=140000.0, fmax_mhz=160000.0, lgint_floor=-7.0)[0]
    assert (q.j_up == q.j_lo).any()                    # these ARE Q-branch lines
    assert float(q.err_mhz.min()) > 100.0


def test_no_dipole_is_an_error_not_an_empty_table():
    with pytest.raises(ValueError):
        predict_lines(RotorConstants(A=5.0e3, B=4.0e3, C=3.0e3))


# ---------------------------------------------------------------------------
# catalogue comparison and refit — the validate stage's machinery
# ---------------------------------------------------------------------------
def _fake_catalogue(c: RotorConstants, **kw) -> pd.DataFrame:
    df, _ = predict_lines(c, **kw)
    return df


def test_compare_with_cat_is_zero_against_itself():
    cat = _fake_catalogue(SO2, fmin_mhz=80000.0, fmax_mhz=400000.0, lgint_floor=-7.0)
    rep = compare_with_cat(cat, cat)
    assert rep["n_matched"] == len(cat)
    assert rep["residual_mhz"]["max_abs"] == pytest.approx(0.0, abs=1e-9)


def test_refit_recovers_perturbed_constants():
    """Shift B by 3 MHz, fit it back: the Hamiltonian's own floor is ~0."""
    cat = _fake_catalogue(SO2, fmin_mhz=80000.0, fmax_mhz=300000.0, lgint_floor=-7.0)
    obs = [((int(r.j_up), int(r.ka_up), int(r.kc_up)),
            (int(r.j_lo), int(r.ka_lo), int(r.kc_lo)), float(r.freq_mhz))
           for r in cat.itertuples()]
    bad = RotorConstants.from_dict({**SO2.as_dict(), "B": SO2.B + 3.0})
    fitted, rep = fit_constants(bad, obs, fit=("A", "B", "C"), j_max=30)
    assert rep["rms_before_mhz"] > 10.0
    assert rep["rms_after_mhz"] < 0.5
    assert fitted.B == pytest.approx(SO2.B, abs=0.2)


def test_compare_with_cat_counts_unparsable_quantum_numbers():
    cat = _fake_catalogue(SO2, fmin_mhz=80000.0, fmax_mhz=200000.0, lgint_floor=-7.0).copy()
    cat.loc[cat.index[:3], "qn_up"] = "v=1 ??"
    rep = compare_with_cat(cat, cat)
    assert rep["n_unparsed_qn"] >= 3


def test_parse_jpl_doc_reads_both_layouts():
    text = ("  10000  60778.5522   1.0e-3\n"
            "    200      0.0065766\n"
            "Rotational constants: A = 60778.55 MHz, B = 10318.07 MHz\n")
    got = parse_jpl_doc(text)["constants"]
    assert got["A"] == pytest.approx(60778.5522)
    assert got["DJ"] == pytest.approx(-0.0065766)     # SPFIT lists −ΔJ
    assert got["B"] == pytest.approx(10318.07)


# ---------------------------------------------------------------------------
# rotorpred: assets → line lists
# ---------------------------------------------------------------------------
def test_assets_load_and_every_block_is_verify():
    a = RP.load_rotor_assets()
    assert a.get("species")
    for sp in RP.predictable_species(a):
        for iso in RP.species_isotopologues(a, sp):
            assert iso.constants.verify is True
            assert iso.constants.A >= iso.constants.B >= iso.constants.C > 0


def test_the_five_uncatalogued_species_are_all_present():
    a = RP.load_rotor_assets()
    assert set(RP.predictable_species(a, role="target")) == {
        "CF2Cl2", "CFCl3", "SO2F2", "CHClF2", "CF2"}
    assert set(RP.predictable_species(a, role="validation")) == {"SO2", "CH2F2", "COF2"}


def test_structure_scaling_moves_the_constants_the_right_way():
    """A heavier isotopologue has larger moments, so smaller B — and by ~1-2 %."""
    a = RP.load_rotor_assets()
    isos = {i.name: i for i in RP.species_isotopologues(a, "CF2Cl2")}
    parent, heavy = isos["CF2(35Cl)2"], isos["CF2(37Cl)2"]
    assert heavy.scaled_from == "CF2(35Cl)2"
    assert heavy.constants.B < parent.constants.B
    assert 0.001 < (parent.constants.B - heavy.constants.B) / parent.constants.B < 0.05


def test_symmetric_block_becomes_an_oblate_top():
    a = RP.load_rotor_assets()
    par = [i for i in RP.species_isotopologues(a, "CFCl3") if i.name == "CF(35Cl)3"][0]
    assert par.constants.A == par.constants.B > par.constants.C
    assert par.constants.representation == "IIIr"
    assert par.constants.mu_c > 0
    assert par.constants.kappa == pytest.approx(1.0)


def test_predict_species_lines_stacks_isotopologues_with_their_own_partition_functions():
    a = RP.load_rotor_assets()
    df, ents, meta = RP.predict_species_lines(a, "CF2Cl2", fmin_mhz=140000.0, fmax_mhz=160000.0,
                                              lgint_floor=-6.0)
    assert len(df) and set(df["isotopologue"]) == {i["name"] for i in meta["isotopologues"]
                                                   if i.get("n_lines")}
    assert len(ents) == len(set(df["entry_id"]))
    for e in ents.values():
        assert len(e["qlog"]) == len(CATDIR_TEMPS)
    assert meta["verify"] is True
    assert list(df.freq_mhz) == sorted(df.freq_mhz)


def test_no_target_is_searchable_while_its_quartic_constants_are_unknown():
    """The Limit-1 finding, as a test.

    All five uncatalogued species are now *predicted* — and none of them is
    *searchable* at a survey's linewidth, because the quartic constants of all
    five are unknown.  That is a sharper statement than "no catalogue entry",
    and it names the fix: obtain the constants.  The test also pins the reason,
    so a future run that reports a coincidence for one of these cannot do it
    silently on a wide tolerance.
    """
    a = RP.load_rotor_assets()
    for sp in ("CF2", "CF2Cl2", "CFCl3", "SO2F2", "CHClF2"):
        _, _, meta = RP.predict_species_lines(a, sp, fmin_mhz=128000.0, fmax_mhz=280000.0,
                                              lgint_floor=-6.0)
        st = RP.searchability(meta, tolerance_mhz=15.0)
        assert meta["n_lines"] > 0, sp
        assert not meta["quartic_known"], sp
        assert st["status"] in ("DEGRADED", "FREQUENCY_LIMITED"), (sp, st)
        assert "quartic" in st["remedy"], sp


def test_supplying_the_quartic_constants_makes_a_target_searchable():
    """The same species, with ΔJ/ΔJK/ΔK filled in, crosses back to SEARCHABLE."""
    a = RP.load_rotor_assets()
    merged, changes = RP.merge_fetched_constants(a, {"CF2": {"CF2 X1A1 v=0": {
        "DJ": 0.0174, "DJK": 0.3245, "DK": 2.8195, "dJ": 0.00266, "dK": 0.0561}}})
    assert len(changes) == 5
    _, _, meta = RP.predict_species_lines(merged, "CF2", fmin_mhz=128000.0, fmax_mhz=280000.0,
                                          lgint_floor=-6.0)
    assert meta["quartic_known"]
    assert RP.searchability(meta, tolerance_mhz=15.0)["status"] == "SEARCHABLE"


def test_merge_fetched_constants_reports_every_change_and_invents_none():
    a = RP.load_rotor_assets()
    merged, changes = RP.merge_fetched_constants(a, {"CF2Cl2": {"CF2(35Cl)2": {"DJ": 0.00046}}})
    assert [c["constant"] for c in changes] == ["DJ"]
    assert changes[0]["was"] is None and changes[0]["now"] == pytest.approx(0.00046)
    assert (a["species"]["CF2Cl2"]["isotopologues"][0]["constants"].get("DJ")) is None
    iso = merged["species"]["CF2Cl2"]["isotopologues"][0]
    assert iso["constants"]["DJ"] == pytest.approx(0.00046)
    _, none = RP.merge_fetched_constants(a, {"NotASpecies": {"x": {"DJ": 1.0}}})
    assert none == []


def test_quartic_constants_make_a_species_searchable_again():
    """The whole Limit-1 claim, as a test: add ΔJ and the error collapses."""
    a = RP.load_rotor_assets()
    _, _, before = RP.predict_species_lines(a, "SO2F2", fmin_mhz=140000.0, fmax_mhz=160000.0,
                                            lgint_floor=-6.0)
    merged, _ = RP.merge_fetched_constants(a, {"SO2F2": {"SO2F2 v=0": {
        "DJ": 1.2e-3, "DJK": -1.0e-3, "DK": 1.5e-3, "dJ": 1.0e-4, "dK": 1.0e-4}}})
    _, _, after = RP.predict_species_lines(merged, "SO2F2", fmin_mhz=140000.0, fmax_mhz=160000.0,
                                           lgint_floor=-6.0)
    assert after["quartic_known"] and not before["quartic_known"]
    assert after["err_mhz"]["median"] < 0.2 * before["err_mhz"]["median"]
    # what is left is the A/B/C uncertainty, 2·σ_ABC·(J+1) with σ_ABC = 0.5 MHz:
    # the remaining fix is a better *measurement* of A, B, C, not a better model
    assert after["err_mhz"]["median"] < 40.0


# ---------------------------------------------------------------------------
# litfetch
# ---------------------------------------------------------------------------
def test_scrape_constants_reads_every_spelling_and_the_units():
    t = ("A = 4118.90(12) MHz, B = 2638.70(9) MHz, C = 2233.72(7) MHz. "
         "DJ = 0.46 kHz, D_JK = -1.23 kHz, delta_J = 0.11 kHz, Delta_K = 3.4 kHz, "
         "δ_K = 0.02 kHz, Φ_J = 1.0e-3 Hz")
    got = LF.scrape_constants(t)["constants"]
    assert got["A"] == pytest.approx(4118.90)
    assert got["DJ"] == pytest.approx(0.00046)          # kHz → MHz
    assert got["DJK"] == pytest.approx(-0.00123)
    assert got["dJ"] == pytest.approx(0.00011)          # δ_J, not Δ_J
    assert got["DK"] == pytest.approx(0.0034)
    assert got["dK"] == pytest.approx(2.0e-5)
    assert got["HJ"] == pytest.approx(1.0e-9)


def test_scrape_constants_marks_an_assumed_unit():
    hits = LF.scrape_constants("B = 2466.0 and DJ = 0.46")["hits"]
    by = {h["name"]: h for h in hits}
    assert by["B"]["unit_assumed"] and by["B"]["value_mhz"] == pytest.approx(2466.0)
    assert by["DJ"]["unit_assumed"] and by["DJ"]["value_mhz"] == pytest.approx(0.00046)


def test_parse_measured_lines_reads_a_qn_table():
    html = ("<table><tr><td>3(1,3)</td><td>2(0,2)</td><td>104029.418</td></tr>"
            "<tr><td>10(1,9)</td><td>10(0,10)</td><td>104239.295</td></tr>"
            "<tr><td>junk</td><td>rows</td><td>are dropped</td></tr></table>")
    got = LF.parse_measured_lines(html)
    assert got == [{"up": (3, 1, 3), "lo": (2, 0, 2), "freq_mhz": pytest.approx(104029.418)},
                   {"up": (10, 1, 9), "lo": (10, 0, 10), "freq_mhz": pytest.approx(104239.295)}]


def test_biblio_text_inverts_an_openalex_abstract():
    body = json.dumps({"results": [{"title": "Millimetre spectrum of CF2Cl2",
                                    "abstract_inverted_index": {"We": [0], "find": [1],
                                                                "B": [2], "=": [3],
                                                                "2638.70": [4]}}]})
    txt = LF.biblio_text("openalex", body)
    assert "Millimetre spectrum" in txt and "B = 2638.70" in txt
    assert LF.scrape_constants(txt)["constants"]["B"] == pytest.approx(2638.70)


def test_fetch_one_records_a_failure_rather_than_raising():
    def boom(url, **kw):
        raise OSError("403 Forbidden")
    rec = LF.fetch_one({"name": "x", "species": "CF2", "kind": "text",
                        "url": "https://example.invalid/x"}, fetch_fn=boom)
    assert rec["status"] == "QUERY_FAILED"
    assert rec["error"] and rec["n_bytes"] == 0
    assert "constants_mhz" not in rec          # nothing is parsed out of nothing


def test_run_litfetch_ledgers_every_value_with_its_url():
    pages = {"https://a/1": "B = 2466.0 MHz DJ = 0.46 kHz",
             "https://a/2": "B = 2487.0 MHz"}

    def fake(url, **kw):
        if url not in pages:
            raise OSError("404")
        return pages[url]
    rep = LF.run_litfetch([{"name": "one", "species": "CFCl3", "kind": "text", "url": "https://a/1"},
                           {"name": "two", "species": "CFCl3", "kind": "text", "url": "https://a/2"},
                           {"name": "dead", "species": "CFCl3", "kind": "text",
                            "url": "https://a/3"}], fetch_fn=fake)
    assert rep["n_ok"] == 2 and rep["n_failed"] == 1
    blk = rep["by_species"]["CFCl3"]
    assert [x["value_mhz"] for x in blk["constants"]["B"]] == [2466.0, 2487.0]
    assert {x["url"] for x in blk["constants"]["B"]} == {"https://a/1", "https://a/2"}
    assert blk["routes_ok"] == 2 and blk["routes_failed"] == 1


def test_species_query_sources_covers_every_service_twice():
    srcs = LF.species_query_sources("CF2Cl2")
    assert {s["kind"] for s in srcs} == {"openalex", "europepmc", "crossref", "semanticscholar"}
    assert all("CF2Cl2" in s["url"] for s in srcs)
    assert any("centrifugal" in s["query"] for s in srcs)


def test_the_ladder_searches_the_species_real_names_not_only_its_formula():
    """A microwave paper on CF2Cl2 is indexed as "dichlorodifluoromethane" or
    "CFC-12", almost never as the formula, and it is the abstract that quotes
    the quartic constants this channel is missing."""
    from seti.uline.rotorpred import load_rotor_assets

    names = load_rotor_assets()["species"]["CF2Cl2"]["search_names"]
    assert "dichlorodifluoromethane" in names and "CFC-12" in names
    qs = LF.species_queries("CF2Cl2", names)
    assert "CF2Cl2 rotational spectrum" in qs
    assert any("dichlorodifluoromethane" in q and "centrifugal" in q for q in qs)
    srcs = LF.species_query_sources("CF2Cl2", names=names)
    # every service gets every phrasing, and the name reaches the URL
    assert len(srcs) == 4 * len(qs)
    assert any("dichlorodifluoromethane" in s["url"] for s in srcs)


def test_litfetch_stops_at_its_wall_clock_and_says_what_it_did_not_try():
    """One hanging service must not spend the job's whole budget — and a source
    that was never reached is NOT a route that answered with nothing."""
    def slow(url, **kw):
        time.sleep(0.05)
        return "B = 1.0 MHz"

    srcs = [{"name": f"s{i}", "species": "CF2", "kind": "text", "url": f"https://a/{i}"}
            for i in range(20)]
    rep = LF.run_litfetch(srcs, fetch_fn=slow, budget_s=0.12)
    assert rep["n_not_attempted"] >= 1
    assert rep["n_ok"] + rep["n_failed"] + rep["n_not_attempted"] == rep["n_sources"] == 20
    skipped = [r for r in rep["sources"] if r["status"] == "NOT_ATTEMPTED"]
    assert all("budget" in r["error"] for r in skipped)
    # a skipped source never counts as a route that failed to find constants
    assert all(not r.get("constants_mhz") for r in skipped)


# ---------------------------------------------------------------------------
# the validate stage, driven entirely offline
# ---------------------------------------------------------------------------
def test_validate_rotor_measures_constants_and_hamiltonian_separately(tmp_path):
    """A synthetic catalogue built from true constants, read with perturbed ones.

    The residual BEFORE the refit must show the constant error; the residual
    AFTER it must collapse to the Hamiltonian's own floor.  That is exactly the
    pair of numbers the runner reports for SO₂, CH₂F₂ and COF₂.
    """
    from seti.uline.run import validate_rotor
    cat = _fake_catalogue(SO2, fmin_mhz=80000.0, fmax_mhz=300000.0, lgint_floor=-7.0)
    assets = {"species": {"SO2": {
        "role": "validation",
        "validate_against": [{"database": "jpl", "tag": 64002}],
        "isotopologues": [{"name": "SO2 v=0", "abundance": 1.0, "quality": "lab",
                           "abc_uncertainty_mhz": 0.01,
                           "constants": {**{k: v for k, v in SO2.as_dict().items()
                                            if k in ("A", "C", "representation", *QUARTIC)},
                                         "B": SO2.B + 2.0, "mu_b": 1.633},
                           "spin_weights": {"rule": "ka_kc_sum", "even": 1, "odd": 0}}]}}}
    rep = validate_rotor(assets, {"jpl:64002": cat}, j_max=20, refit=True)
    rec = rep["species"]["jpl:64002"]
    assert rec["status"] == "OK"
    assert rec["comparison"]["n_matched"] > 20
    assert rec["comparison"]["residual_mhz"]["median_abs"] > 1.0      # the 2 MHz B error
    assert rec["refit"]["rms_after_mhz"] < 0.05                       # the Hamiltonian floor
    # switching the quartic terms off must move the lines by far more than the
    # Hamiltonian floor: that IS the distortion-truncation error
    assert (rec["distortion_truncation"]["residual_mhz"]["median_abs"]
            > 20.0 * rec["refit"]["rms_after_mhz"])
    assert rep["headline"]["n_species_validated"] == 1


def test_validate_rotor_says_no_catalogue_rather_than_guessing():
    from seti.uline.run import validate_rotor
    assets = {"species": {"SO2": {"role": "validation",
                                  "validate_against": [{"database": "jpl", "tag": 64002}],
                                  "isotopologues": []}}}
    rep = validate_rotor(assets, {}, j_max=10)
    assert rep["species"]["jpl:64002"]["status"] == "NO_CATALOGUE"
    assert rep["headline"]["n_species_validated"] == 0


def test_quartic_estimates_bracket_the_known_constants():
    """4X³/ω² per term must be ABOVE the true constant, and not absurdly so.

    An error bound that is below the truth is a lie; one that is 100× above it
    makes every species unsearchable for nothing.  Checked against the two
    validation species whose quartic constants are known.
    """
    from seti.uline.rotor import quartic_estimates
    ch2f2 = RotorConstants(A=49142.85, B=10604.51, C=9249.86, DJ=0.0103, DJK=0.0325, DK=0.50,
                           dJ=0.00105, dK=0.011, mu_b=1.978)
    for c in (SO2, ch2f2):
        est = quartic_estimates(c)
        for e, true in zip(est, (abs(c.DJ), abs(c.DJK), abs(c.DK)), strict=True):
            assert e > true, (c.name, e, true)
            assert e < 30.0 * true, (c.name, e, true)


def test_distortion_estimate_is_large_for_a_heavy_uncatalogued_rotor():
    assert DISTORTION_SCALE_REL == pytest.approx(3.0e-6)
    df, _ = predict_lines(
        RotorConstants(A=4118.90, B=2638.70, C=2233.72, mu_b=0.51, name="CF2Cl2"),
        fmin_mhz=149000.0, fmax_mhz=151000.0, lgint_floor=-6.0)
    assert 100.0 < float(df.err_mhz.median()) < 1.0e5
