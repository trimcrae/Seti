"""General rotational-spectrum predictor for ULINE: Watson A-reduced asymmetric top.

Why this exists.  Five of the most diagnostic S54 species --- CF₂Cl₂, CFCl₃,
SO₂F₂, CHClF₂ and the CF₂ radical --- have no entry in JPL or CDMS, so the
first ULINE dispatches reported them ``targets_unsearchable``.  Their
rotational constants are in the microwave literature, and a rotational
spectrum is a *computable* object: given the constants, every line frequency
and relative intensity follows.  This module computes them.

Physics
-------
Energy levels come from the Watson A-reduced Hamiltonian (Watson 1977, in
*Vibrational Spectra and Structure* vol. 6; Gordy & Cook 1984 §8.5) with
quartic and sextic centrifugal distortion,

.. math::

   H = B_z J_z^2 + B_x J_x^2 + B_y J_y^2
       - \\Delta_J J^4 - \\Delta_{JK} J^2 J_z^2 - \\Delta_K J_z^4
       - 2\\delta_J J^2 (J_x^2 - J_y^2)
       - \\delta_K [J_z^2 (J_x^2 - J_y^2) + (J_x^2 - J_y^2) J_z^2]
       + \\Phi_J J^6 + \\Phi_{JK} J^4 J_z^2 + \\Phi_{KJ} J^2 J_z^4 + \\Phi_K J_z^6
       + 2\\phi_J J^4 (J_x^2 - J_y^2)
       + \\phi_{JK} J^2 [J_z^2 (J_x^2 - J_y^2) + (J_x^2 - J_y^2) J_z^2]
       + \\phi_K [J_z^4 (J_x^2 - J_y^2) + (J_x^2 - J_y^2) J_z^4],

set up in the symmetric-top basis :math:`|J,K\\rangle` for each J and
block-diagonalised in the Wang basis
:math:`|J,K,\\pm\\rangle = (|J,K\\rangle \\pm |J,-K\\rangle)/\\sqrt 2`.  The
representation is a choice of which inertial axis is *z*: ``Ir`` (z = a,
x = b, y = c; the right choice for prolate-like tops) or ``IIIr`` (z = c,
x = a, y = b; for oblate-like tops).  Levels are labelled :math:`J_{K_a K_c}`
by the rigid-rotor ordering theorem (King, Hainer & Cross 1943: within one
J the energies increase monotonically with :math:`\\tau = K_a - K_c`) and,
inside each Wang block, by the block's fixed K-parity and fixed
:math:`K_a + K_c` sum, which is what makes the assignment robust to numerical
ties between near-degenerate K-doublets.

Line strengths are the rotational part of the transition moment summed over
lab orientations, computed from the eigenvectors and the Clebsch--Gordan
coefficients of the molecule-fixed dipole components (Gordy & Cook §7.3):

.. math::

   S_g(J'\\tau' \\leftarrow J\\tau) = (2J+1)\\,\\Big|\\sum_{K,K',q}
   c'_{K'}\\, c_K\\, u^g_q\\, \\langle J K\\, 1 q \\,|\\, J' K'\\rangle\\Big|^2 ,

with :math:`u^z_0 = 1`, :math:`u^x_{\\pm 1} = \\mp 1/\\sqrt 2`,
:math:`u^y_{\\pm 1} = i/\\sqrt 2`.  In the symmetric-top limit this is exactly
the textbook :math:`S = ((J+1)^2 - K^2)/(J+1)` for :math:`J \\to J+1`, and it
obeys :math:`\\sum_{J'\\tau'} S_g = 2J+1` for every dipole component, which the
offline suite checks together with the a-, b- and c-type selection rules.

Intensities follow the JPL convention (Pickett et al. 1998, JQSRT 60, 883):
:math:`I(300) = 4.16231\\times10^{-5}\\,\\nu\\,S\\,\\mu_g^2\\,g_{ns}
[e^{-E_l/kT} - e^{-E_u/kT}]/Q(T)` in nm² MHz with ν in MHz and μ in Debye,
:math:`Q = \\sum g_{ns}(2J+1)e^{-E/kT}` on the JPL ``catdir`` temperature
grid, so the output plugs into :func:`seti.uline.lines.rescale_lgint` like
a catalogue entry.  Nuclear-spin weights are a rule on the parities of
:math:`(K_a, K_c)`; nuclear quadrupole hyperfine structure is **not**
modelled --- its blend width is estimated into the frequency uncertainty
(:func:`hyperfine_width_mhz`) so a Cl-bearing line carries an honest error.

What is validated where
-----------------------
* offline (``tests/test_uline_rotor.py``): closed-form rigid-rotor energies
  at J = 1, 2; the exact symmetric-top limit including ΔJ, ΔJK, ΔK; the
  Wang-block labels against the τ-ordering theorem; the three selection
  rules; the strength sum rule; representation invariance; and the
  laboratory SO₂ lines whose frequencies are beyond doubt.
* on the runner (``python -m seti.uline.run --stage validate``): the full
  JPL/CDMS entries of SO₂, CH₂F₂ and COF₂ line by line, with the constants
  parsed from JPL's own documentation files, so the statement "reproduces the
  catalogue to X kHz" is measured, not asserted.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .lines import (
    C2_CM_K,
    CATDIR_TEMPS,
    JPL_INTENSITY_CONST,
    LINE_COLUMNS,
    MHZ_PER_CM,
    Entry,
)

#: h / (8 π² c) in amu Å² MHz: B[MHz] = ROT_CONST_AMU_A2 / I[amu Å²]  (CODATA 2018)
ROT_CONST_AMU_A2 = 505379.0096

#: Conservative estimator of an unmeasured ΔJ as a fraction of (B+C)/2.  The
#: rigid-rotor scaling ΔJ ≈ 4B̄³/ω² with a generic ω = 400 cm⁻¹ overshoots by
#: ~two orders of magnitude for the molecules of this channel; the ratio is
#: measured instead on the species whose ΔJ *is* known (SO₂ 6.4e-7, CH₂F₂
#: 9.7e-7, CF₂ ~1.4e-6 of (B+C)/2), and 3e-6 is a factor ~2–5 upper bound on
#: that set.  Used only as an ERROR TERM for a species whose quartic constants
#: are unknown — never as a distortion constant in the Hamiltonian.  The
#: ``validate`` stage re-measures the ratio on the catalogued species and
#: reports it next to this number.
DISTORTION_SCALE_REL = 3.0e-6

QUARTIC = ("DJ", "DJK", "DK", "dJ", "dK")
SEXTIC = ("HJ", "HJK", "HKJ", "HK", "hJ", "hJK", "hK")
REPRESENTATIONS = {"Ir": ("a", "b", "c"), "IIr": ("b", "c", "a"), "IIIr": ("c", "a", "b")}


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
@dataclass
class RotorConstants:
    """Spectroscopic constants of one isotopologue in one vibrational state.

    Rotational constants in MHz, A ≥ B ≥ C by convention (the a, b, c axes
    are defined by that ordering); the distortion constants are Watson
    A-reduction values *in the stated representation* (they are not
    representation-independent).  Unknown distortion constants are ``None``,
    which the error model treats differently from a known zero.
    """

    A: float
    B: float
    C: float
    representation: str = "Ir"
    DJ: float | None = None
    DJK: float | None = None
    DK: float | None = None
    dJ: float | None = None
    dK: float | None = None
    HJ: float | None = None
    HJK: float | None = None
    HKJ: float | None = None
    HK: float | None = None
    hJ: float | None = None
    hJK: float | None = None
    hK: float | None = None
    mu_a: float = 0.0
    mu_b: float = 0.0
    mu_c: float = 0.0
    #: nuclear-spin weight rule: {"rule": "ka_kc_sum" | "ka" | "kc" | "none",
    #: "even": w_even, "odd": w_odd}
    spin_weights: dict = field(default_factory=lambda: {"rule": "none"})
    name: str = "rotor"
    source: str = ""
    verify: bool = True

    def __post_init__(self):
        if self.representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be one of {sorted(REPRESENTATIONS)}")
        if not (self.A >= self.B >= self.C > 0):
            raise ValueError("rotational constants must satisfy A >= B >= C > 0 (MHz)")

    @property
    def quartic_known(self) -> bool:
        return all(getattr(self, k) is not None for k in QUARTIC)

    @property
    def any_quartic_known(self) -> bool:
        return any(getattr(self, k) is not None for k in QUARTIC)

    @property
    def sextic_known(self) -> bool:
        return any(getattr(self, k) is not None for k in SEXTIC)

    def value(self, key: str) -> float:
        v = getattr(self, key)
        return 0.0 if v is None else float(v)

    def axis_constants(self) -> tuple[float, float, float]:
        """(B_z, B_x, B_y) for the chosen representation."""
        by_axis = {"a": float(self.A), "b": float(self.B), "c": float(self.C)}
        z, x, y = REPRESENTATIONS[self.representation]
        return by_axis[z], by_axis[x], by_axis[y]

    def dipole_xyz(self) -> tuple[float, float, float]:
        """(μ_z, μ_x, μ_y) in Debye for the chosen representation."""
        by_axis = {"a": float(self.mu_a), "b": float(self.mu_b), "c": float(self.mu_c)}
        z, x, y = REPRESENTATIONS[self.representation]
        return by_axis[z], by_axis[x], by_axis[y]

    @property
    def kappa(self) -> float:
        """Ray's asymmetry parameter κ = (2B − A − C)/(A − C): −1 prolate, +1 oblate."""
        return (2.0 * self.B - self.A - self.C) / (self.A - self.C) if self.A != self.C else 0.0

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("A", "B", "C", "representation", *QUARTIC, *SEXTIC,
                                           "mu_a", "mu_b", "mu_c", "name", "source", "verify")}
        d["spin_weights"] = dict(self.spin_weights)
        d["kappa"] = self.kappa
        return d

    @classmethod
    def from_dict(cls, d: dict, **override) -> RotorConstants:
        keys = {"A", "B", "C", "representation", *QUARTIC, *SEXTIC, "mu_a", "mu_b", "mu_c",
                "spin_weights", "name", "source", "verify"}
        kw = {k: v for k, v in (d or {}).items() if k in keys}
        kw.update(override)
        return cls(**kw)


def spin_weight(rule: dict, ka: int, kc: int) -> float:
    """Nuclear-spin statistical weight of the level J_{Ka,Kc} under ``rule``.

    ``ka_kc_sum``: the C₂ axis is *b* (the exchange operation is C₂ᵇ, whose
    character is (−1)^{Ka+Kc}); ``ka``: the C₂ axis is *a* ((−1)^{Ka});
    ``kc``: the C₂ axis is *c* ((−1)^{Kc}); ``k_mod3``: a C₃ᵥ top with three
    equivalent nuclei, where the levels with K ≡ 0 (mod 3) carry the A
    spin weight and the rest the E weight (``axis`` names the symmetry axis,
    ``a`` for a prolate top and ``c`` for an oblate one); ``none``: every
    level weight 1.  A weight of 0 removes the level (SO₂: only Ka+Kc even
    exists).
    """
    r = str((rule or {}).get("rule", "none"))
    if r == "none":
        return 1.0
    if r == "k_mod3":
        k = ka if str(rule.get("axis", "a")) == "a" else kc
        return float(rule.get("multiple", 1.0)) if k % 3 == 0 else float(rule.get("other", 1.0))
    if r == "ka_kc_sum":
        par = (ka + kc) % 2
    elif r == "ka":
        par = ka % 2
    elif r == "kc":
        par = kc % 2
    else:
        raise ValueError(f"unknown spin-weight rule {r!r}")
    return float(rule.get("even", 1.0)) if par == 0 else float(rule.get("odd", 1.0))


# ---------------------------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------------------------
def _off_diagonal_f(j: int, k: np.ndarray) -> np.ndarray:
    """⟨J,K+2| (J_x² − J_y²) |J,K⟩ = ½ √[(J(J+1) − K(K+1)) (J(J+1) − (K+1)(K+2))]."""
    jj = j * (j + 1)
    return 0.5 * np.sqrt(np.maximum((jj - k * (k + 1)) * (jj - (k + 1) * (k + 2)), 0.0))


def hamiltonian_matrix(c: RotorConstants, j: int, *, rigid: bool = False) -> np.ndarray:
    """The (2J+1)×(2J+1) Hamiltonian in the |J,K⟩ basis, K = −J … J (index K+J), MHz."""
    bz, bx, by = c.axis_constants()
    n = 2 * j + 1
    k = np.arange(-j, j + 1, dtype=float)
    jj = float(j * (j + 1))
    h = np.zeros((n, n))
    diag = 0.5 * (bx + by) * jj + (bz - 0.5 * (bx + by)) * k ** 2
    if not rigid:
        diag += (-c.value("DJ") * jj ** 2 - c.value("DJK") * jj * k ** 2 - c.value("DK") * k ** 4
                 + c.value("HJ") * jj ** 3 + c.value("HJK") * jj ** 2 * k ** 2
                 + c.value("HKJ") * jj * k ** 4 + c.value("HK") * k ** 6)
    h[np.arange(n), np.arange(n)] = diag
    if j >= 1:
        kk = k[:-2]                                   # couples K with K+2
        f = _off_diagonal_f(j, kk)
        coef = np.full_like(kk, 0.5 * (bx - by))
        if not rigid:
            s2 = kk ** 2 + (kk + 2) ** 2
            s4 = kk ** 4 + (kk + 2) ** 4
            coef = coef + (-2.0 * c.value("dJ") * jj - c.value("dK") * s2
                           + 2.0 * c.value("hJ") * jj ** 2 + c.value("hJK") * jj * s2
                           + c.value("hK") * s4)
        off = f * coef
        idx = np.arange(n - 2)
        h[idx, idx + 2] = off
        h[idx + 2, idx] = off
    return h


def wang_blocks(j: int) -> list[tuple[str, np.ndarray]]:
    """The four Wang blocks of J as (label, transformation matrix U with columns
    = symmetrised basis vectors in the |J,K⟩ basis).  Labels ``E+ E- O+ O-``:
    K parity (even/odd) and the sign of the ± combination; K = 0 sits in E+."""
    n = 2 * j + 1
    out: list[tuple[str, np.ndarray]] = []
    for parity in ("E", "O"):
        for sign, s in (("+", 1.0), ("-", -1.0)):
            cols = []
            for k in range(0, j + 1):
                if (k % 2 == 0) != (parity == "E"):
                    continue
                v = np.zeros(n)
                if k == 0:
                    if sign == "-":
                        continue
                    v[j] = 1.0
                else:
                    v[j + k] = 1.0 / math.sqrt(2.0)
                    v[j - k] = s / math.sqrt(2.0)
                cols.append(v)
            if cols:
                out.append((parity + sign, np.stack(cols, axis=1)))
    return out


@dataclass
class Levels:
    """Eigen-decomposition of one J: energies (MHz), eigenvectors in the |J,K⟩
    basis (columns), labels (Ka, Kc), Wang block per level."""

    j: int
    energy: np.ndarray
    vectors: np.ndarray
    ka: np.ndarray
    kc: np.ndarray
    block: list[str]

    @property
    def tau(self) -> np.ndarray:
        return self.ka - self.kc


def _labels_for_block(j: int, block: str, n: int, sum_kakc: int, z_axis: str
                      ) -> list[tuple[int, int]]:
    """The (Ka, Kc) labels a Wang block holds, K_a ascending (energy ascending).

    The block's K parity is the parity of K_a (z = a) or of K_c (z = c) or of
    K_b (z = b, where K_b ≡ J − Ka − Kc + ... has no simple form: the IIr
    representation is labelled by the τ-ordering alone).  Within a block the
    sum Ka + Kc is fixed at J or J + 1.
    """
    parity = 0 if block[0] == "E" else 1
    labels = []
    for ka in range(0, j + 1):
        kc = sum_kakc - ka
        if not 0 <= kc <= j:
            continue
        if z_axis == "a" and ka % 2 != parity:
            continue
        if z_axis == "c" and kc % 2 != parity:
            continue
        labels.append((ka, kc))
    if len(labels) != n:
        raise RuntimeError(f"J={j} block {block}: {n} levels but {len(labels)} labels for "
                           f"Ka+Kc={sum_kakc}")
    return labels


def levels(c: RotorConstants, j: int, *, rigid: bool = False) -> Levels:
    """Energies, eigenvectors and J_{Ka,Kc} labels for one J."""
    h = hamiltonian_matrix(c, j, rigid=rigid)
    z_axis = REPRESENTATIONS[c.representation][0]
    pooled: list[tuple[float, str, int]] = []
    block_e: dict[str, np.ndarray] = {}
    block_v: dict[str, np.ndarray] = {}
    for label, u in wang_blocks(j):
        hb = u.T @ h @ u
        e, w = np.linalg.eigh(hb)
        block_e[label] = e
        block_v[label] = u @ w
        for i, ei in enumerate(e):
            pooled.append((float(ei), label, i))
    pooled.sort(key=lambda t: t[0])
    # τ-ordering labels on the pooled spectrum (the rigid-rotor theorem): the
    # labelling of last resort (IIr), and what the tests check the block
    # table against for every non-degenerate rotor.
    tent: dict[tuple[str, int], tuple[int, int]] = {}
    for rank, (_, label, i) in enumerate(pooled):
        tau = rank - j
        tent[(label, i)] = ((tau + j + 1) // 2, (j - tau + 1) // 2)
    n_tot = 2 * j + 1
    energy = np.zeros(n_tot)
    vectors = np.zeros((n_tot, n_tot))
    ka = np.zeros(n_tot, dtype=int)
    kc = np.zeros(n_tot, dtype=int)
    blocks: list[str] = [""] * n_tot
    pos = 0
    for label, e in block_e.items():
        if z_axis in ("a", "c"):
            # Each Wang block holds levels of ONE Ka+Kc sum, J or J+1, fixed by
            # the block's symmetry under the C₂ rotations (Gordy & Cook Table
            # 7.5): with z = a the "+" blocks carry Ka+Kc = J; with z = c the
            # sum is J when the sign equals (−1)^K.  Within a block the energy
            # rises with Ka, so the members are labelled Ka-ascending.  This
            # holds for exact ties (a symmetric top) where τ-ordering cannot.
            k_par = 0 if label[0] == "E" else 1
            plus = label[1] == "+"
            if z_axis == "a":
                s = j if plus else j + 1
            else:
                s = j if plus == (k_par == 0) else j + 1
            labs = _labels_for_block(j, label, len(e), s, z_axis)
        else:
            labs = [tent[(label, i)] for i in range(len(e))]
        for i in range(len(e)):
            energy[pos] = e[i]
            vectors[:, pos] = block_v[label][:, i]
            ka[pos], kc[pos] = labs[i]
            blocks[pos] = label
            pos += 1
    order = np.argsort(energy, kind="stable")
    return Levels(j=j, energy=energy[order], vectors=vectors[:, order], ka=ka[order],
                  kc=kc[order], block=[blocks[i] for i in order])


def rigid_rotor_energies_j1(c: RotorConstants) -> dict[str, float]:
    """Closed forms for J = 1 (used by the tests): E(1_01) = B+C, E(1_11) = A+C, E(1_10) = A+B."""
    return {"101": c.B + c.C, "111": c.A + c.C, "110": c.A + c.B}


# ---------------------------------------------------------------------------
# dipole matrix elements
# ---------------------------------------------------------------------------
def _cg_matrix(j: int, jp: int, q: int) -> np.ndarray:
    """⟨J K 1 q | J' K+q⟩ as a (2J'+1) × (2J+1) matrix (Condon–Shortley phases)."""
    out = np.zeros((2 * jp + 1, 2 * j + 1))
    for k in range(-j, j + 1):
        kp = k + q
        if abs(kp) > jp:
            continue
        out[kp + jp, k + j] = _cg_j1(j, k, q, jp)
    return out


def _cg_j1(j: int, k: int, q: int, jp: int) -> float:
    """Clebsch–Gordan ⟨j k; 1 q | j' k+q⟩ for j' = j+1, j, j−1 (Condon & Shortley Table)."""
    if jp == j + 1:
        if q == 1:
            return math.sqrt((j + k + 1) * (j + k + 2) / ((2 * j + 1) * (2 * j + 2)))
        if q == 0:
            return math.sqrt((j - k + 1) * (j + k + 1) / ((2 * j + 1) * (j + 1)))
        return math.sqrt((j - k + 1) * (j - k + 2) / ((2 * j + 1) * (2 * j + 2)))
    if jp == j:
        if j == 0:
            return 0.0
        if q == 1:
            return -math.sqrt((j + k + 1) * (j - k) / (2 * j * (j + 1)))
        if q == 0:
            return k / math.sqrt(j * (j + 1))
        return math.sqrt((j - k + 1) * (j + k) / (2 * j * (j + 1)))
    if jp == j - 1:
        if q == 1:
            return math.sqrt((j - k - 1) * (j - k) / (2 * j * (2 * j + 1)))
        if q == 0:
            return -math.sqrt((j - k) * (j + k) / (j * (2 * j + 1)))
        return math.sqrt((j + k - 1) * (j + k) / (2 * j * (2 * j + 1)))
    return 0.0


def line_strengths(lo: Levels, up: Levels) -> dict[str, np.ndarray]:
    """S_g(up ← lo) for g ∈ {z, x, y}, as (n_up × n_lo) arrays.

    ``S = (2J+1) |C_upᵀ (Σ_q u^g_q M_q) C_lo|²`` with M_q the Clebsch–Gordan
    matrices; each S obeys Σ_up S = 2J_lo+1 over a complete set of J' = J−1,
    J, J+1 final states (checked in the tests).
    """
    j, jp = lo.j, up.j
    if abs(jp - j) > 1:
        return {g: np.zeros((len(up.energy), len(lo.energy))) for g in ("z", "x", "y")}
    m0 = _cg_matrix(j, jp, 0)
    mp = _cg_matrix(j, jp, 1)
    mm = _cg_matrix(j, jp, -1)
    cu, cl = up.vectors, lo.vectors
    az = cu.T @ m0 @ cl
    ax = cu.T @ ((mm - mp) / math.sqrt(2.0)) @ cl
    ay = cu.T @ ((mm + mp) / math.sqrt(2.0)) @ cl          # × i: modulus unchanged
    w = 2 * j + 1
    return {"z": w * az ** 2, "x": w * ax ** 2, "y": w * ay ** 2}


# ---------------------------------------------------------------------------
# hyperfine blend width (Cl, N quadrupole), an error term not a model
# ---------------------------------------------------------------------------
def hyperfine_width_mhz(eqq_mhz: float, n_nuclei: int, j_lo: int, k_lo: int, j_up: int, k_up: int
                        ) -> float:
    """Estimated spread of the unmodelled quadrupole hyperfine pattern of a line.

    First-order quadrupole energies scale as ``−eQq [3K²/J(J+1) − 1] f(I,J,F)``
    with ``|f| ≤ 1/8`` for the strongest components; the pattern width of a
    transition is set by the change of that bracket between the two levels,
    times ``|eQq|/4``, times the number of equivalent nuclei (their patterns
    add in width, not in quadrature).  This is the blend width a survey would
    see, used as a frequency uncertainty; it is ~1 MHz for R-branch lines of
    CF₂Cl₂ at J ≈ 30 and tens of MHz for K_a-changing lines at low J.
    """
    def g(j: int, k: int) -> float:
        return 3.0 * k * k / (j * (j + 1)) - 1.0 if j > 0 else 0.0

    return abs(float(eqq_mhz)) / 4.0 * int(n_nuclei) * abs(g(j_up, k_up) - g(j_lo, k_lo))


# ---------------------------------------------------------------------------
# spectrum
# ---------------------------------------------------------------------------
def _q_is_converged(contrib: float, total: float) -> bool:
    return total > 0 and contrib / total < 1e-7


def partition_function(c: RotorConstants, temps=CATDIR_TEMPS, *, j_max: int = 250,
                       levels_cache: dict | None = None) -> tuple[list[float], int]:
    """log10 Q(T) on ``temps`` (explicit sum with spin weights); returns (qlog, J_max used)."""
    temps = [float(t) for t in temps]
    tot = np.zeros(len(temps))
    t_hot = max(temps)
    j_used = 0
    for j in range(0, j_max + 1):
        lv = (levels_cache or {}).get(j) or levels(c, j)
        if levels_cache is not None:
            levels_cache[j] = lv
        w = np.array([spin_weight(c.spin_weights, int(a), int(b)) for a, b in zip(lv.ka, lv.kc, strict=True)])
        e_cm = lv.energy / MHZ_PER_CM
        contrib = np.array([(2 * j + 1) * float(np.sum(w * np.exp(-C2_CM_K * e_cm / t)))
                            for t in temps])
        tot += contrib
        j_used = j
        hot = temps.index(t_hot)
        if j > 5 and _q_is_converged(contrib[hot], tot[hot]):
            break
    return [float(np.log10(x)) if x > 0 else float("-inf") for x in tot], j_used


def _qn_string(j: int, ka: int, kc: int) -> str:
    return f"{j:2d}{ka:2d}{kc:2d}"


def predict_lines(c: RotorConstants, *, fmin_mhz: float = 0.0, fmax_mhz: float = 2.0e6,
                  j_max: int | None = None, lgint_floor: float = -9.0, temps=CATDIR_TEMPS,
                  tag: int = 0, err_base_mhz: float = 0.5, err_rel: float = 1e-5,
                  err_per_j_mhz: float = 0.0,
                  hyperfine: dict | None = None, distortion_scale_mhz: float | None = None,
                  abundance: float = 1.0) -> tuple[pd.DataFrame, Entry]:
    """Every allowed rotational line of the species in ``[fmin, fmax]`` MHz.

    Returns a table in the ``.cat`` schema (:data:`seti.uline.lines.LINE_COLUMNS`
    plus ``j_up ka_up kc_up j_lo ka_lo kc_lo dipole strength gns``) and an
    :class:`Entry` carrying the explicit partition function on ``temps``.

    ``err_mhz`` is the model's *own* statement of how far the frequency is
    trusted: ``err_base + err_rel·ν + err_per_j·(J_up+1)`` (the last term
    propagates the uncertainty of A, B, C into an R-branch line, which grows
    linearly with J), plus — when the quartic constants are unknown — the size
    of the unmodelled distortion ``distortion_scale·[J'²(J'+1)² − J²(J+1)²]``
    (``distortion_scale`` defaults to
    :data:`DISTORTION_SCALE_REL`·(B+C)/2), plus the hyperfine blend width when
    ``hyperfine = {"eQq_mhz": …, "n_nuclei": …}``.  ``abundance`` scales the
    intensities (an isotopologue's fraction).

    That distortion term is not a formality.  For a heavy rotor with
    ΔJ ~ 10⁻² MHz an R-branch line at J ≈ 30 is displaced by ~4ΔJ(J+1)³ ≈ 10³
    MHz, which is two orders of magnitude wider than any survey's matching
    tolerance: **a species whose quartic constants are unknown is not
    searchable at survey precision, however well its A, B, C are known.**  The
    error column says so rather than hiding it.
    """
    mu_z, mu_x, mu_y = c.dipole_xyz()
    if mu_z == mu_x == mu_y == 0.0:
        raise ValueError("no dipole component: no rotational spectrum")
    bmin = min(c.axis_constants())
    if j_max is None:
        j_max = int(fmax_mhz / (2.0 * bmin)) + 4
    j_max = int(max(2, min(j_max, 99)))            # two-character QN fields, as in .cat
    cache: dict[int, Levels] = {}
    qlog, j_q = partition_function(c, temps, levels_cache=cache, j_max=max(j_max + 40, 60))
    for j in range(0, j_max + 2):
        if j not in cache:
            cache[j] = levels(c, j)
    q300 = 10.0 ** float(np.interp(np.log10(300.0), np.log10(np.asarray(temps, float))[::-1],
                                   np.asarray(qlog, float)[::-1]))
    if distortion_scale_mhz is None:
        distortion_scale_mhz = DISTORTION_SCALE_REL * 0.5 * (c.B + c.C)
    chunks: list[dict] = []
    z_axis = REPRESENTATIONS[c.representation][0]
    lg_min = float(lgint_floor)
    for j in range(0, j_max + 1):
        lo = cache[j]
        w_lo = np.array([spin_weight(c.spin_weights, int(a), int(b))
                         for a, b in zip(lo.ka, lo.kc, strict=True)])
        for jp in (j, j + 1):
            up = cache[jp]
            w_up = np.array([spin_weight(c.spin_weights, int(a), int(b))
                             for a, b in zip(up.ka, up.kc, strict=True)])
            s = line_strengths(lo, up)
            # ν, the lower-state energy and the Boltzmann factor are the same for
            # every dipole component; only S and μ change.  (n_up × n_lo) arrays.
            nu = up.energy[:, None] - lo.energy[None, :]
            el_cm = np.repeat((lo.energy / MHZ_PER_CM)[None, :], len(up.energy), axis=0)
            eu_cm = el_cm + nu / MHZ_PER_CM
            with np.errstate(over="ignore", under="ignore"):
                pop = (np.exp(-C2_CM_K * el_cm / 300.0) - np.exp(-C2_CM_K * eu_cm / 300.0))
            allowed = (nu > 0) & (nu >= fmin_mhz) & (nu <= fmax_mhz)
            allowed &= (w_up[:, None] > 0) & (w_lo[None, :] > 0)
            if jp == j:                       # each ΔJ = 0 pair once, upper above lower
                allowed &= np.greater.outer(np.arange(len(up.energy)), np.arange(len(lo.energy)))
            if not allowed.any():
                continue
            base = JPL_INTENSITY_CONST * nu * w_up[:, None] * pop / q300 * abundance
            for g, mu in (("z", mu_z), ("x", mu_x), ("y", mu_y)):
                if mu == 0.0:
                    continue
                sg = s[g]
                inten = base * sg * mu * mu
                keep = allowed & (sg >= 1e-12) & (inten > 0)
                if keep.any():
                    with np.errstate(divide="ignore", invalid="ignore"):
                        keep &= np.log10(np.where(keep, inten, 1.0)) >= lg_min
                if not keep.any():
                    continue
                iu, il = np.nonzero(keep)
                n = len(iu)
                f = nu[iu, il]
                err = err_base_mhz + err_rel * f + err_per_j_mhz * (jp + 1)
                if not c.quartic_known:
                    jj_u, jj_l = jp * (jp + 1), j * (j + 1)
                    err = err + float(distortion_scale_mhz) * abs(jj_u ** 2 - jj_l ** 2)
                if hyperfine:
                    k_lo = (lo.ka if z_axis != "c" else lo.kc)[il]
                    k_up = (up.ka if z_axis != "c" else up.kc)[iu]
                    hw = np.array([hyperfine_width_mhz(float(hyperfine.get("eQq_mhz", 0.0)),
                                                       int(hyperfine.get("n_nuclei", 1)),
                                                       j, int(a), jp, int(b))
                                   for a, b in zip(k_lo, k_up, strict=True)])
                    err = np.hypot(err, hw)
                gu = w_up[iu]
                chunks.append({
                    "freq_mhz": f, "err_mhz": err, "lgint_300": np.log10(inten[iu, il]),
                    "dr": np.full(n, 3), "elo_cm": el_cm[iu, il],
                    "gup": np.rint(gu * (2 * jp + 1)).astype(int), "tag": np.full(n, int(tag)),
                    "lab": np.zeros(n, dtype=bool), "qnfmt": np.full(n, 303),
                    "qn_up": [_qn_string(jp, int(a), int(b)) for a, b in zip(up.ka[iu], up.kc[iu], strict=True)],
                    "qn_lo": [_qn_string(j, int(a), int(b)) for a, b in zip(lo.ka[il], lo.kc[il], strict=True)],
                    "j_up": np.full(n, jp), "ka_up": up.ka[iu], "kc_up": up.kc[iu],
                    "j_lo": np.full(n, j), "ka_lo": lo.ka[il], "kc_lo": lo.kc[il],
                    "dipole": [g] * n, "strength": sg[iu, il], "gns": gu})
    cols = list(LINE_COLUMNS) + ["j_up", "ka_up", "kc_up", "j_lo", "ka_lo", "kc_lo", "dipole",
                                 "strength", "gns"]
    if chunks:
        df = pd.DataFrame({k: np.concatenate([np.asarray(ch[k]) for ch in chunks])
                           for k in chunks[0]}).reindex(columns=cols)
    else:
        df = pd.DataFrame(columns=cols)
    axis_of = dict(zip(("z", "x", "y"), REPRESENTATIONS[c.representation], strict=True))
    if len(df):
        df["dipole"] = df["dipole"].map(axis_of)
        df = df.sort_values("freq_mhz").reset_index(drop=True)
    entry = Entry(tag=int(tag), name=c.name, nlines=int(len(df)), temps=list(temps), qlog=qlog,
                  database="rotor")
    entry.version = f"rotor:{c.representation}:Jmax={j_max}:Q_Jmax={j_q}"
    return df, entry


# ---------------------------------------------------------------------------
# structure-scaled isotopologues
# ---------------------------------------------------------------------------
def moments_of_inertia(atoms: list[tuple[float, tuple[float, float, float]]]
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Principal moments (amu Å², ascending) and axes of a set of (mass, xyz)."""
    m = np.array([a[0] for a in atoms], dtype=float)
    r = np.array([a[1] for a in atoms], dtype=float)
    com = (m[:, None] * r).sum(0) / m.sum()
    r = r - com
    t = np.zeros((3, 3))
    for mi, ri in zip(m, r, strict=True):
        t += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
    w, v = np.linalg.eigh(t)
    return w, v


def isotopologue_constants(parent: RotorConstants, geometry: list[tuple[float, tuple]],
                           substituted: list[tuple[float, tuple]], **override) -> RotorConstants:
    """A, B, C of an isotopologue by scaling the parent's constants with the
    ratio of the moments of a common geometry.

    The scaling keeps the parent's measured constants exact and moves them by
    the *structural* isotopic shift, so the error is the geometry's error
    times the shift (∼1 % of a ∼2 % shift, i.e. ∼2×10⁻⁴ of B): a searchable
    but wide prediction, which the error model must and does reflect.
    """
    ip, _ = moments_of_inertia(geometry)
    iq, _ = moments_of_inertia(substituted)
    consts = sorted([parent.A, parent.B, parent.C], reverse=True)
    scaled = sorted([consts[i] * ip[i] / iq[i] for i in range(3)], reverse=True)
    d = parent.as_dict()
    d.update({"A": scaled[0], "B": scaled[1], "C": scaled[2]})
    d.update(override)
    return RotorConstants.from_dict(d)


# ---------------------------------------------------------------------------
# JPL documentation parser and catalogue comparison
# ---------------------------------------------------------------------------
#: SPFIT parameter codes (Pickett 1991) for an asymmetric top, A-reduction.
SPFIT_CODES: dict[int, tuple[str, float]] = {
    10000: ("A", 1.0), 20000: ("B", 1.0), 30000: ("C", 1.0),
    200: ("DJ", -1.0), 1100: ("DJK", -1.0), 2000: ("DK", -1.0),
    40100: ("dJ", -1.0), 41000: ("dK", -1.0),
    300: ("HJ", 1.0), 1200: ("HJK", 1.0), 2100: ("HKJ", 1.0), 3000: ("HK", 1.0),
    40200: ("hJ", 1.0), 41100: ("hJK", 1.0), 42000: ("hK", 1.0),
}
_DOC_PARAM_RE = re.compile(r"^\s*(\d{3,6})\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?:\(\s*\d*\s*\))?")
_DOC_NAMED_RE = re.compile(r"\b(A|B|C|D_?J|D_?JK|D_?K|d_?J|d_?K|DELTA_?J|DELTA_?JK|DELTA_?K|"
                           r"delta_?J|delta_?K)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(MHz|kHz|Hz)?")


def parse_jpl_doc(text: str) -> dict:
    """Constants from a JPL ``d<tag>.cat`` documentation file, tolerant of both
    layouts (SPFIT parameter-code lines, or prose ``A = 60778.55 MHz``).

    Returns ``{"constants": {name: MHz}, "codes": {code: value}, "lines":
    [raw lines used]}``; the SPFIT sign convention (``-DJ`` is listed) is
    undone so the constants are Watson's.  Anything not understood is left
    out, never guessed.
    """
    out: dict = {"constants": {}, "codes": {}, "lines": []}
    for line in (text or "").splitlines():
        m = _DOC_PARAM_RE.match(line)
        if m:
            code = int(m.group(1))
            if code in SPFIT_CODES:
                name, sign = SPFIT_CODES[code]
                try:
                    val = float(m.group(2)) * sign
                except ValueError:
                    continue
                out["codes"][code] = float(m.group(2))
                out["constants"].setdefault(name, val)
                out["lines"].append(line.rstrip()[:120])
                continue
        for m in _DOC_NAMED_RE.finditer(line):
            key = m.group(1).replace("_", "")
            key = {"DELTAJ": "DJ", "DELTAJK": "DJK", "DELTAK": "DK", "deltaJ": "dJ",
                   "deltaK": "dK"}.get(key, key)
            if key not in ("A", "B", "C", *QUARTIC):
                continue
            try:
                val = float(m.group(2))
            except ValueError:
                continue
            unit = (m.group(3) or "MHz").lower()
            val *= {"mhz": 1.0, "khz": 1e-3, "hz": 1e-6}[unit]
            out["constants"].setdefault(key, val)
            out["lines"].append(line.rstrip()[:120])
    return out


def _qn_triplet(s: str) -> tuple[int, int, int] | None:
    toks = str(s).split()
    if len(toks) < 3:
        # fixed 2-char fields without spaces: "12 3 9" vs "1239"
        raw = str(s)
        toks = [raw[i:i + 2] for i in range(0, min(len(raw), 6), 2)]
    try:
        return int(toks[0]), int(toks[1]), int(toks[2])
    except (ValueError, IndexError):
        return None


def compare_with_cat(pred: pd.DataFrame, cat: pd.DataFrame, *, j_max: int | None = None
                     ) -> dict:
    """Match predicted lines to catalogue lines by (J,Ka,Kc)_up/lo and report residuals.

    Returns the residual statistics (MHz) overall and per J bin, the intensity
    ratio statistics (dex), the unmatched counts, and the worst ten lines ---
    the numbers behind any statement that the predictor reproduces a
    catalogue.  Catalogue lines whose quantum numbers do not parse as three
    integers (a different QNFMT) are counted, not matched.
    """
    key_p = {(r.j_up, r.ka_up, r.kc_up, r.j_lo, r.ka_lo, r.kc_lo): (r.freq_mhz, r.lgint_300)
             for r in pred.itertuples()}
    n_unparsed = 0
    matched = []
    for r in cat.itertuples():
        qu, ql = _qn_triplet(r.qn_up), _qn_triplet(r.qn_lo)
        if qu is None or ql is None:
            n_unparsed += 1
            continue
        if j_max is not None and qu[0] > j_max:
            continue
        k = (*qu, *ql)
        if k in key_p:
            f, lg = key_p[k]
            matched.append((qu[0], qu[1], f - r.freq_mhz, lg - r.lgint_300, r.freq_mhz, k))
    n_cat = int(len(cat)) if j_max is None else int(sum(
        1 for r in cat.itertuples() if (_qn_triplet(r.qn_up) or (10 ** 6,))[0] <= j_max))
    if not matched:
        return {"n_catalogue": n_cat, "n_predicted": int(len(pred)), "n_matched": 0,
                "n_unparsed_qn": n_unparsed, "residual_mhz": None}
    m = pd.DataFrame(matched, columns=["j", "ka", "dnu", "dlg", "nu", "key"])
    absr = m["dnu"].abs()
    per_j = []
    for lo_j in range(0, int(m["j"].max()) + 1, 10):
        sub = m[(m["j"] >= lo_j) & (m["j"] < lo_j + 10)]
        if len(sub):
            per_j.append({"j_range": [lo_j, lo_j + 9], "n": int(len(sub)),
                          "median_abs_mhz": float(sub["dnu"].abs().median()),
                          "p95_abs_mhz": float(sub["dnu"].abs().quantile(0.95)),
                          "max_abs_mhz": float(sub["dnu"].abs().max())})
    worst = m.reindex(absr.sort_values(ascending=False).index).head(10)
    return {"n_catalogue": n_cat, "n_predicted": int(len(pred)), "n_matched": int(len(m)),
            "n_unparsed_qn": n_unparsed,
            "residual_mhz": {"median_abs": float(absr.median()), "p95_abs": float(absr.quantile(0.95)),
                             "max_abs": float(absr.max()), "mean": float(m["dnu"].mean()),
                             "rms": float(np.sqrt(np.mean(m["dnu"] ** 2)))},
            "intensity_dex": {"median": float(m["dlg"].median()),
                              "p95_abs_dev_from_median": float((m["dlg"] - m["dlg"].median()).abs().quantile(0.95)),
                              "std": float(m["dlg"].std())},
            "per_j": per_j,
            "worst": [{"qn": list(map(int, r.key)), "nu_cat_mhz": float(r.nu),
                       "dnu_mhz": float(r.dnu), "dlg": float(r.dlg)} for r in worst.itertuples()]}


def fit_constants(c0: RotorConstants, observed: list[tuple[tuple[int, int, int], tuple[int, int, int], float]],
                  fit: tuple[str, ...] = ("A", "B", "C", *QUARTIC), *, j_max: int | None = None,
                  max_nfev: int = 200) -> tuple[RotorConstants, dict]:
    """Least-squares refit of ``fit`` constants to ``observed`` = [(up, lo, ν_MHz)].

    Used by the runner validation to separate "the embedded constants are
    slightly off" from "the Hamiltonian is wrong": the residual after the
    refit is the Hamiltonian's own floor.  Returns the fitted constants and
    ``{"rms_before", "rms_after", "n_lines", "constants", "success"}``.
    """
    from scipy.optimize import least_squares

    if j_max is None:
        j_max = max(max(u[0], lo[0]) for u, lo, _ in observed)
    keys = [k for k in fit if getattr(c0, k) is not None or k in ("A", "B", "C")]
    x0 = np.array([c0.value(k) for k in keys], dtype=float)
    scale = np.array([max(abs(v), 1e-6) for v in x0])
    nu_obs = np.array([o[2] for o in observed], dtype=float)

    def predict(x: np.ndarray) -> np.ndarray:
        d = c0.as_dict()
        d.update({k: float(v) for k, v in zip(keys, x, strict=True)})
        try:
            cc = RotorConstants.from_dict(d)
        except ValueError:
            return np.full(len(observed), 1e6)
        lv = {j: levels(cc, j) for j in range(0, j_max + 1)}

        def e(j, ka, kc):
            L = lv[j]
            i = np.flatnonzero((L.ka == ka) & (L.kc == kc))
            return float(L.energy[i[0]]) if len(i) else np.nan

        return np.array([e(*u) - e(*lo) for u, lo, _ in observed])

    r0 = predict(x0) - nu_obs
    res = least_squares(lambda y: predict(y * scale) - nu_obs, x0 / scale, max_nfev=max_nfev,
                        x_scale="jac")
    x1 = res.x * scale
    r1 = predict(x1) - nu_obs
    d = c0.as_dict()
    d.update({k: float(v) for k, v in zip(keys, x1, strict=True)})
    fitted = RotorConstants.from_dict(d)
    rep = {"rms_before_mhz": float(np.sqrt(np.nanmean(r0 ** 2))),
           "rms_after_mhz": float(np.sqrt(np.nanmean(r1 ** 2))),
           "max_abs_after_mhz": float(np.nanmax(np.abs(r1))), "n_lines": int(len(observed)),
           "constants": {k: float(v) for k, v in zip(keys, x1, strict=True)}, "success": bool(res.success),
           "j_max": int(j_max)}
    return fitted, rep


__all__ = ["DISTORTION_SCALE_REL", "QUARTIC", "REPRESENTATIONS", "ROT_CONST_AMU_A2", "SEXTIC",
           "SPFIT_CODES", "Levels",
           "RotorConstants", "compare_with_cat", "fit_constants", "hamiltonian_matrix",
           "hyperfine_width_mhz", "isotopologue_constants", "levels", "line_strengths",
           "moments_of_inertia", "parse_jpl_doc", "partition_function", "predict_lines",
           "rigid_rotor_energies_j1", "spin_weight", "wang_blocks"]
