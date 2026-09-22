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
    measured: MeasuredSuite | None = None  # measured meteorites, if they were reached

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
# the MEASURED meteorite suite (PEWDD's own compilation)
# ---------------------------------------------------------------------------
#: Column pattern of the compilations shipped with jamietwilliams/PEWDD:
#: ``[X/Si]`` etc., which is log10 of the NUMBER ratio X/reference (verified
#: against bulk Earth: [Mg/Si] = +0.028, [Fe/Si] = -0.072, [Al/Si] = -1.037).
_RATIO_COL = re.compile(r"^\[([A-Za-z]{1,2})/([A-Za-z]{1,2})\]$")

#: A log10 number ratio outside this is a placeholder, not a measurement.
SANE_LOG_RATIO = (-20.0, 6.0)


@dataclass
class MeasuredSuite:
    """Measured meteorites as log10 number ratios against one reference element.

    The end-member asset is eighteen compiled vectors.  This is the ~1,200
    individually analysed stones and irons that PEWDD itself compares against
    (``meteorite_database_*.csv`` in ``jamietwilliams/PEWDD``), and it is what
    the brief means by a meteorite mixture space.  It is used to WIDEN the
    Tier 2 envelopes to what nature has actually been measured to do, never to
    narrow them: a pair whose observed spread across real meteorites is larger
    than the compiled end-members suggest is a weaker test, not a wrong one,
    and the number belongs in the record rather than in a footnote.
    """

    reference: str
    elements: list[str]
    log_ratio: np.ndarray       # (n_meteorites, n_elements), log10(X/reference)
    names: list[str]
    classes: list[str]

    def pair_log_ratios(self, num: str, den: str) -> np.ndarray:
        """log10(num/den) for every meteorite that measures both."""
        if num not in self.elements or den not in self.elements:
            return np.array([])
        i, j = self.elements.index(num), self.elements.index(den)
        v = self.log_ratio[:, i] - self.log_ratio[:, j]
        return v[np.isfinite(v)]

    def pair_log_ratios_in_class(self, num: str, den: str, klass: str) -> np.ndarray:
        """log10(num/den) for the bodies of one class."""
        if num not in self.elements or den not in self.elements:
            return np.array([])
        i, j = self.elements.index(num), self.elements.index(den)
        sel = np.array([c == klass for c in self.classes])
        v = self.log_ratio[sel, i] - self.log_ratio[sel, j]
        return v[np.isfinite(v)]

    def pair_envelope(self, num: str, den: str, *, trim_percent: float = 0.5) -> dict:
        v = self.pair_log_ratios(num, den)
        if v.size == 0:
            return {"n": 0, "lo": np.nan, "hi": np.nan, "trim_lo": np.nan, "trim_hi": np.nan}
        return {"n": int(v.size), "lo": float(v.min()), "hi": float(v.max()),
                "trim_lo": float(np.percentile(v, trim_percent)),
                "trim_hi": float(np.percentile(v, 100.0 - trim_percent)),
                "median": float(np.median(v)),
                "width_dex": float(v.max() - v.min())}


def load_measured_meteorites(paths, *, report: dict | None = None) -> MeasuredSuite | None:
    """Read the ``[X/ref]`` compilations into ONE suite of measured bodies.

    Three of PEWDD's files are the same 1,226 analyses written against three
    different reference elements (Si, Fe, Mg; the name columns are identical
    row for row).  Reading them as three suites would treble-count every
    meteorite, so rows are merged on (name, class, occurrence) and an element
    missing from the reference file is filled from a converted one through
    log10(X/ref0) = log10(X/ref) + log10(ref/ref0).

    Values outside :data:`SANE_LOG_RATIO` are dropped as placeholders, not
    measurements: the compilations write about fifty +34 dex entries (all of
    them [Cr/*]) and a handful of -35 dex entries where a source reported
    zero.  Genuinely large ratios are kept --- [O/Fe] = +2.5 in a CI chondrite
    is real --- and the count of dropped cells is returned in ``report``.
    """
    rows: dict[tuple, dict] = {}
    n_cells = n_dropped = 0
    files: list[dict] = []
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as exc:                              # noqa: BLE001
            files.append({"path": str(p), "status": f"unreadable:{exc!r}"[:120]})
            continue
        cols, refs = {}, set()
        for c in df.columns:
            m = _RATIO_COL.match(str(c).strip())
            if m:
                cols[m.group(1)] = str(c)
                refs.add(m.group(2))
        if len(cols) < 3 or len(refs) != 1:
            files.append({"path": str(p), "status": "no single-reference ratio block"})
            continue
        frames.append((str(p), refs.pop(), cols, df))
        files.append({"path": str(p), "status": "OK", "n_rows": int(len(df)),
                      "reference": frames[-1][1], "n_elements": len(cols)})
    if not frames:
        if report is not None:
            report.update({"files": files})
        return None
    ref0 = max((f[1] for f in frames), key=lambda r: sum(1 for g in frames if g[1] == r))
    lo_ok, hi_ok = SANE_LOG_RATIO
    for _path, ref, cols, df in frames:
        names = df["Names"].astype(str) if "Names" in df else pd.Series([""] * len(df))
        klass = df["Class"].astype(str) if "Class" in df else pd.Series([""] * len(df))
        seen: dict[tuple, int] = {}
        vals = {el: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                for el, c in cols.items()}
        shift = vals.get(ref0)           # log10(ref0 / ref), zero when ref == ref0
        for i in range(len(df)):
            key0 = (names.iloc[i], klass.iloc[i])
            k = seen.get(key0, 0)
            seen[key0] = k + 1
            key = (names.iloc[i], klass.iloc[i], k)
            rec = rows.setdefault(key, {})
            if ref != ref0:
                if shift is None or not np.isfinite(shift[i]):
                    continue
                off = -float(shift[i])
            else:
                off = 0.0
            for el, arr in vals.items():
                v = arr[i]
                if not np.isfinite(v):
                    continue
                n_cells += 1
                if not (lo_ok <= v <= hi_ok):
                    n_dropped += 1
                    continue
                rec.setdefault(el, float(v) + off)
    keys = [k for k, v in rows.items() if v]
    els = sorted({e for k in keys for e in rows[k]})
    mat = np.full((len(keys), len(els)), np.nan)
    for r, k in enumerate(keys):
        for e, v in rows[k].items():
            mat[r, els.index(e)] = v
    if report is not None:
        report.update({"files": files, "reference": ref0, "n_bodies": len(keys),
                       "n_cells": n_cells, "n_cells_dropped_out_of_range": n_dropped,
                       "sane_log_ratio": list(SANE_LOG_RATIO)})
    return MeasuredSuite(reference=ref0, elements=els, log_ratio=mat,
                         names=[k[0] for k in keys], classes=[k[1] for k in keys])


def measured_suite_report(suite: MeasuredSuite | None, pairs) -> dict:
    """What the measured suite says about each pair, for the record."""
    if suite is None:
        return {"available": False}
    from collections import Counter
    out = {"available": True, "reference": suite.reference,
           "n_meteorites": int(suite.log_ratio.shape[0]),
           "elements": suite.elements,
           "classes": dict(Counter(suite.classes).most_common(15)),
           "pairs": {}}
    for a, b in pairs:
        env = suite.pair_envelope(a, b)
        # Per class as well as over the whole suite: the unconditional spread
        # of Ti/Al or Ca/Al is dominated by the metal-rich classes, in which
        # Al and Ti are trace.  Tier 2 uses the unconditional envelope on
        # purpose (it asks whether ANY measured body reaches the ratio), so
        # the per-class numbers are what says how much of the width is one
        # class and how much is the chemistry.
        by_class = {}
        for k in sorted(set(suite.classes)):
            v = suite.pair_log_ratios_in_class(a, b, k)
            if v.size >= MEASURED_MIN_BODIES:
                by_class[k] = {"n": int(v.size), "lo": float(v.min()), "hi": float(v.max()),
                               "median": float(np.median(v)),
                               "width_dex": float(v.max() - v.min())}
        env["by_class"] = by_class
        if by_class:
            env["narrowest_class_width_dex"] = min(c["width_dex"] for c in by_class.values())
            env["widest_class_width_dex"] = max(c["width_dex"] for c in by_class.values())
        out["pairs"][f"{a}/{b}"] = env
    return out


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
#: A measured pair envelope is only believed once this many bodies measure both.
MEASURED_MIN_BODIES = 5


def ratio_envelope(fam: NaturalFamily, num: str, den: str, *, endmembers=None,
                   fractionation_depths=(-1.0, 0.0, 1.0, 2.0), t_cuts=None,
                   width_K: float = 150.0, t_cut_max: float | None = None,
                   measured: MeasuredSuite | None | bool = True) -> dict:
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
    out = {"num": num, "den": den, "lo": lo, "hi": hi, "per_endmember": per_end,
           "width_dex": hi - lo,
           "endmember_lo": lo, "endmember_hi": hi,
           "unfractionated_lo": float(min(per_end.values())),
           "unfractionated_hi": float(max(per_end.values())),
           "measured_n": 0, "envelope_source": "endmembers"}
    # The MEASURED meteorites widen the envelope, never narrow it.  The
    # compiled end-members are eighteen averaged vectors; individual stones and
    # irons reach further (pallasite Ca/Al is +2.9 where no end-member exceeds
    # 0.0), and a Tier 2 exceedance against an envelope that real meteorites
    # already leave would be an artefact of the compilation, not a residual.
    suite = fam.measured if measured is True else (measured or None)
    if suite is not None:
        env = suite.pair_envelope(num, den)
        if env.get("n", 0) >= MEASURED_MIN_BODIES:
            out["measured_n"] = env["n"]
            out["measured_lo"], out["measured_hi"] = env["lo"], env["hi"]
            out["measured_trim_lo"], out["measured_trim_hi"] = env["trim_lo"], env["trim_hi"]
            if env["lo"] < out["lo"] or env["hi"] > out["hi"]:
                out["envelope_source"] = "endmembers+measured"
            out["lo"] = float(min(out["lo"], env["lo"]))
            out["hi"] = float(max(out["hi"], env["hi"]))
            out["width_dex"] = out["hi"] - out["lo"]
    return out


__all__ = ["ASSET", "ENDMEMBER_GROUPS", "PRESENCE_FLOOR_PPM", "TC_FRACTIONATION_OVERRIDE_K",
           "MeasuredSuite", "NaturalFamily", "SANE_LOG_RATIO", "load_measured_meteorites",
           "measured_suite_report",
           "fractionation_factor", "load_family", "mixture", "natural_parcel", "ratio_envelope",
           "softmax"]
