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

import re
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


# ---------------------------------------------------------------------------
# the MEASURED family: PEWDD's meteorite compilation
# ---------------------------------------------------------------------------
#: A bulk meteorite never has a log element ratio beyond this.  Entries past it
#: are the compilation's zero / detection-limit sentinels (values near −36 dex
#: appear in the Mn column) and are dropped before any percentile is taken.
METEORITE_SANITY_DEX = 10.0

#: Percentiles that define the measured envelope.  With ~1,100 meteorites per
#: pair this leaves ~5 real rocks outside each end, which are reported by name
#: rather than hidden.
METEORITE_Q_LO = 0.5
METEORITE_Q_HI = 99.5

#: Minimum meteorites before the measured envelope is used at all.
METEORITE_MIN_N = 30


def load_meteorites(path: Path | str) -> pd.DataFrame | None:
    """PEWDD's meteorite compilation: one row per measured meteorite.

    Columns are ``[X/Mg]`` (or ``[X/Fe]`` / ``[X/Si]`` in the sibling files):
    base-10 logarithms of ABSOLUTE number ratios, not solar-referenced --- the
    class medians reproduce this module's end-member values to ~0.1 dex, which
    is what makes them directly comparable.
    """
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:                                             # noqa: BLE001
        return None
    if not any(str(c).startswith("[") for c in df.columns):
        return None
    return df


def _bracket_denominator(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        m = re.fullmatch(r"\[([A-Z][a-z]?)/([A-Z][a-z]?)\]", str(c))
        if m and m.group(1) == m.group(2):
            return m.group(2)
    for c in df.columns:
        m = re.fullmatch(r"\[[A-Z][a-z]?/([A-Z][a-z]?)\]", str(c))
        if m:
            return m.group(1)
    return None


def meteorite_ratio_envelope(df: pd.DataFrame | None, num: str, den: str, *,
                             q_lo: float = METEORITE_Q_LO, q_hi: float = METEORITE_Q_HI,
                             min_n: int = METEORITE_MIN_N,
                             model_envelope: tuple | None = None) -> dict | None:
    """The MEASURED natural envelope of log10(num/den) over real meteorites.

    Both elements are read against the file's own denominator and subtracted,
    so the denominator cancels and the result is the absolute log ratio.
    Returns ``None`` when the compilation does not constrain the pair.
    """
    if df is None or len(df) < min_n:
        return None
    dn = _bracket_denominator(df)
    if dn is None:
        return None
    ca, cb = f"[{num}/{dn}]", f"[{den}/{dn}]"
    if ca not in df.columns or cb not in df.columns:
        return None
    r = pd.to_numeric(df[ca], errors="coerce") - pd.to_numeric(df[cb], errors="coerce")
    raw = r[np.isfinite(r)]
    sane = raw[(raw > -METEORITE_SANITY_DEX) & (raw < METEORITE_SANITY_DEX)]
    if len(sane) < min_n:
        return None
    lo, hi = (float(x) for x in np.percentile(sane, [q_lo, q_hi]))
    names = df["Names"].astype(str) if "Names" in df.columns else pd.Series(
        [""] * len(df), index=df.index)
    classes = df["Class"].astype(str) if "Class" in df.columns else names
    ext_hi = sane[sane > hi].sort_values(ascending=False).index[:5]
    ext_lo = sane[sane < lo].sort_values().index[:5]
    frac_out = None
    if model_envelope is not None:
        mlo, mhi = (float(x) for x in model_envelope)
        if np.isfinite(mlo) and np.isfinite(mhi):
            frac_out = float(np.mean((sane < mlo) | (sane > mhi)))
    return {
        "fraction_outside_model": frac_out,
        "num": num, "den": den, "denominator_column": dn,
        "lo": lo, "hi": hi, "width_dex": hi - lo,
        "min": float(sane.min()), "max": float(sane.max()),
        "median": float(np.median(sane)),
        "n": int(len(sane)), "n_dropped_sentinel": int(len(raw) - len(sane)),
        "q_lo": q_lo, "q_hi": q_hi,
        "beyond_hi": [[str(names[i])[:24], str(classes[i]), round(float(sane[i]), 3)]
                      for i in ext_hi],
        "beyond_lo": [[str(names[i])[:24], str(classes[i]), round(float(sane[i]), 3)]
                      for i in ext_lo],
    }


def combined_ratio_envelope(fam: NaturalFamily, num: str, den: str, *,
                            meteorites: pd.DataFrame | None = None, **kw) -> dict:
    """The natural envelope of log10(num/den): end-member model UNION measurement.

    The model envelope alone is not the natural family.  Measured against
    PEWDD's 1,226-meteorite compilation it is *too narrow* for the pairs that
    matter --- 53 % of real meteorites fall outside the model Ni/Co envelope
    and 35 % outside the model Ti/Al envelope, because aubrites, ureilites,
    lodranites, eucrites and pallasites are real rocks that eighteen idealised
    compositions do not span.  It is also too wide for Mn/Cr (2.9 dex modelled
    against 1.5 dex measured), which needlessly weakens that test.  The
    envelope used is therefore the union, and both components, plus the
    fraction of real meteorites the model alone would have excluded, are
    carried in the record.
    """
    model = ratio_envelope(fam, num, den, **kw)
    met = meteorite_ratio_envelope(meteorites, num, den,
                                   model_envelope=(model["lo"], model["hi"]))
    out = dict(model)
    out["model_lo"], out["model_hi"] = model["lo"], model["hi"]
    out["meteorite"] = met
    if met is None:
        out["source"] = "endmember_model_only"
        return out
    out["lo"] = float(min(model["lo"], met["lo"])) if np.isfinite(model["lo"]) else met["lo"]
    out["hi"] = float(max(model["hi"], met["hi"])) if np.isfinite(model["hi"]) else met["hi"]
    out["width_dex"] = out["hi"] - out["lo"]
    out["source"] = "endmember_model_union_measured_meteorites"
    out["meteorites_outside_model_envelope_fraction"] = met.get("fraction_outside_model")
    return out


__all__ = ["ASSET", "ENDMEMBER_GROUPS", "METEORITE_MIN_N", "METEORITE_Q_HI", "METEORITE_Q_LO",
           "METEORITE_SANITY_DEX", "PRESENCE_FLOOR_PPM", "TC_FRACTIONATION_OVERRIDE_K",
           "NaturalFamily", "combined_ratio_envelope",
           "fractionation_factor", "load_family", "load_meteorites", "meteorite_ratio_envelope",
           "mixture", "natural_parcel", "ratio_envelope", "softmax"]
