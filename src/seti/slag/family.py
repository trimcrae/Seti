"""The natural family: what a white dwarf can be fed by nature.

End-members (``data_assets/slag_natural_family.csv``): nine chondrite groups,
bulk Earth, core, bulk silicate Earth, continental crust, bulk silicate Moon,
bulk silicate Mars, a mean eucrite (Vesta's crust), water ice and a comet-like
CHON organic.  A natural accreted parcel is a point on the simplex spanned by
them, further moved by condensation-temperature fractionation (the volatile
lever) --- and the photosphere then applies the sinking lever (``sinking.py``).

Everything here is in NUMBER abundances (``ppm_by_mass / A``), because the
photosphere reports log(Z/H) or log(Z/He) by number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ASSET = Path(__file__).resolve().parents[1] / "data_assets" / "slag_natural_family.csv"

#: Columns of the asset that are not end-members.
_META = ("element", "A", "Tc50_K")

#: The asset writes 1e-4 ppm where a compilation says zero (lithophiles in the
#: core, everything but H/O in ice).  Below this an element is ABSENT from the
#: end-member: it never defines a natural ratio.
PRESENCE_FLOOR_PPM = 1e-2

#: 50% condensation temperatures that the fractionation lever must NOT use as
#: written.  Lodders' Tc(O) = 180 K is the temperature at which half of all
#: oxygen has condensed --- as water ice.  The oxygen of a ROCK condenses with
#: the silicates it is bound in (~1300 K), and a parcel with silicate Mg, Si,
#: Ca and Fe carries that oxygen whatever its volatile depletion; the ice-borne
#: excess is the water_ice end-member.  Without this the volatile lever could
#: strip the oxygen off a silicate and call a pure-Si parcel natural.
TC_FRACTIONATION_OVERRIDE_K = {"O": 1300.0}

#: Order matters nowhere in the physics; it fixes the parameter vector.
ENDMEMBER_GROUPS = {
    "chondrite": ["CI", "CM", "CO", "CV", "H", "L", "LL", "EH", "EL"],
    "differentiated": ["bulk_Earth", "core", "mantle_BSE", "crust_cont", "Moon_BSM",
                       "Mars_BSM", "Vesta_eucrite"],
    "volatile": ["water_ice", "CHON_organics"],
}


@dataclass
class NaturalFamily:
    """The end-member table in number units, with per-element metadata."""

    elements: list[str]
    endmembers: list[str]
    number: np.ndarray          # (n_elements, n_endmembers), number fraction per end-member
    mass_ppm: np.ndarray        # (n_elements, n_endmembers), as in the asset
    A: np.ndarray               # atomic masses
    Tc: np.ndarray              # 50% condensation temperatures (K) as in the asset
    present: np.ndarray = None  # (n_elements, n_endmembers): above the asset's write-in floor
    Tc_eff: np.ndarray = None   # what the fractionation lever uses (TC_FRACTIONATION_OVERRIDE_K)

    def __post_init__(self):
        if self.present is None:
            self.present = self.mass_ppm > PRESENCE_FLOOR_PPM
        if self.Tc_eff is None:
            self.Tc_eff = np.array(self.Tc, dtype=float)
            for el, t in TC_FRACTIONATION_OVERRIDE_K.items():
                if el in self.elements:
                    self.Tc_eff[self.elements.index(el)] = float(t)

    def index(self, element: str) -> int:
        return self.elements.index(element)

    def has(self, element: str, endmember: str) -> bool:
        return bool(self.present[self.index(element), self.endmembers.index(endmember)])

    def sub(self, elements: list[str]) -> np.ndarray:
        """Number abundances of ``elements`` (rows) for every end-member."""
        idx = [self.index(e) for e in elements]
        return self.number[idx, :]

    def tc(self, elements: list[str]) -> np.ndarray:
        """The fractionation-effective condensation temperatures of ``elements``."""
        return np.array([self.Tc_eff[self.index(e)] for e in elements], dtype=float)

    def column(self, endmember: str) -> np.ndarray:
        return self.number[:, self.endmembers.index(endmember)]


def load_family(path: Path | str | None = None) -> NaturalFamily:
    p = Path(path) if path is not None else ASSET
    df = pd.read_csv(p, comment="#")
    df = df.set_index("element")
    ends = [c for c in df.columns if c not in _META]
    mass = df[ends].to_numpy(dtype=float)
    A = df["A"].to_numpy(dtype=float)
    Tc = df["Tc50_K"].to_numpy(dtype=float)
    number = mass / A[:, None]
    number = number / number.sum(axis=0, keepdims=True)
    return NaturalFamily(elements=list(df.index), endmembers=ends, number=number,
                         mass_ppm=mass, A=A, Tc=Tc)


# ---------------------------------------------------------------------------
# the four levers, minus sinking (which lives in sinking.py)
# ---------------------------------------------------------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float) - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def fractionation_factor(Tc: np.ndarray, t_cut: float, depth_dex: float,
                         width_K: float = 100.0) -> np.ndarray:
    """Multiplicative depletion (or enrichment) of elements below ``t_cut``.

    ``depth_dex`` > 0 depletes elements whose 50% condensation temperature is
    below ``t_cut`` by up to that many dex, through a logistic of width
    ``width_K`` --- the volatile-depletion trend of the inner solar system
    (Earth, Vesta, the enstatite parent bodies) and of post-nebular heating.
    A negative depth enriches the volatiles instead (ice-rich accretion is
    also covered by the ice end-members).
    """
    s = 1.0 / (1.0 + np.exp((np.asarray(Tc, dtype=float) - float(t_cut)) / float(width_K)))
    return 10.0 ** (-float(depth_dex) * s)


def mixture(number_sub: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Number abundances of a weighted mixture of end-members (rows = elements)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return number_sub @ w


def natural_parcel(fam: NaturalFamily, elements: list[str], weights: np.ndarray,
                   t_cut: float, depth_dex: float, width_K: float = 100.0) -> np.ndarray:
    """The accreted parcel's number abundances of ``elements`` under the first three levers."""
    x = mixture(fam.sub(elements), weights)
    return x * fractionation_factor(fam.tc(elements), t_cut, depth_dex, width_K)


# ---------------------------------------------------------------------------
# natural envelopes of a ratio
# ---------------------------------------------------------------------------
def ratio_envelope(fam: NaturalFamily, num: str, den: str, *, endmembers=None,
                   fractionation_depths=(-1.0, 0.0, 1.0, 2.0), t_cuts=None,
                   width_K: float = 150.0, t_cut_max: float | None = None) -> dict:
    """Min / max of log10(num/den) over the natural family.

    A ratio of two linear mixtures is a weighted mean of the end-member ratios
    (weights ``w_i * den_i``), so it lies between the extreme end-members;
    the envelope is therefore the extreme over end-members, then widened by
    condensation fractionation over the listed depths and cut temperatures
    (``t_cut_max`` caps the cut: nature fractionates the moderately volatile
    elements, not the refractories among themselves).  End-members in which
    either element is absent (the core for a lithophile, ice for everything
    but H and O) do not define the envelope: a ratio against nothing is not a
    natural value.
    """
    ends = list(endmembers) if endmembers else list(fam.endmembers)
    i_n, i_d = fam.index(num), fam.index(den)
    Tc = fam.tc([num, den])
    if t_cuts is None:
        t_cuts = np.arange(0.0, 1801.0, 50.0)
    if t_cut_max is not None:
        t_cuts = [t for t in t_cuts if t <= float(t_cut_max)]
    logs, per_end = [], {}
    for e in ends:
        j = fam.endmembers.index(e)
        if not (fam.present[i_n, j] and fam.present[i_d, j]):
            continue
        n_, d_ = fam.number[i_n, j], fam.number[i_d, j]
        base = np.log10(n_ / d_)
        per_end[e] = float(base)
        for d in fractionation_depths:
            if d == 0.0:
                logs.append(base)
                continue
            for tc in t_cuts:
                f = fractionation_factor(Tc, tc, d, width_K)
                logs.append(base + np.log10(f[0] / f[1]))
    if not logs:
        return {"num": num, "den": den, "lo": np.nan, "hi": np.nan, "per_endmember": {},
                "width_dex": np.nan}
    lo, hi = float(np.min(logs)), float(np.max(logs))
    return {"num": num, "den": den, "lo": lo, "hi": hi, "per_endmember": per_end,
            "width_dex": hi - lo,
            "unfractionated_lo": float(min(per_end.values())),
            "unfractionated_hi": float(max(per_end.values()))}


__all__ = ["ASSET", "ENDMEMBER_GROUPS", "PRESENCE_FLOOR_PPM", "TC_FRACTIONATION_OVERRIDE_K",
           "NaturalFamily",
           "fractionation_factor", "load_family", "mixture", "natural_parcel", "ratio_envelope",
           "softmax"]
