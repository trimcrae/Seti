"""Contamination rejection for DERELICT survivors.

Nothing that clears the screens is believed until it has been traced to a
systematic.  The systematics, in descending order of how much of the parameter
space they own:

1. **Human hardware.**  This is not a minor contaminant, it is *the* one.  JPL
   and the MPC already run exactly this area-to-mass test, reactively, to
   unmask spent upper stages that got asteroid designations -- 2020 SO
   (a Centaur from Surveyor 2), J002E3 (an Apollo 12 S-IVB), WT1190F.  A high
   implied area-to-mass on a low-eccentricity, low-inclination, ~1 au orbit is
   a rocket body until proven otherwise.  These objects are *also* the
   channel's positive controls: a pipeline that fails to flag them is broken.
2. **Outgassing.**  A comet with an undetected coma produces a genuine radial
   acceleration.  Cometary designation, a fitted DT, a cometary ``g(r)``, or a
   reported coma all remove an object.
3. **Short-arc fit artefacts.**  A spurious ``A1`` is overwhelmingly a
   badly-constrained orbit.  The negative-A1 census (screen 3) measures this
   rate empirically, which is why it doubles as the false-positive floor.
4. **Yarkovsky leakage.**  ``A1`` and ``A2`` are correlated in short-arc
   solutions, so a real transverse Yarkovsky signal can bleed into the radial
   term.  Requires the covariance matrix to test.
5. **Genuinely small natural bodies.**  A metre-scale rock really does have a
   measurable radiation-pressure signal.  The R statistic normalises this out
   by construction, but survivors below ~10 m are labelled so nothing is
   over-read.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .acquire import model_par_names, model_par_values
from .radiation import COMETARY_MODEL_PARS

# --- Verdicts -----------------------------------------------------------------
ARTIFICIAL_HUMAN = "ARTIFICIAL_HUMAN"
ARTIFICIAL_SUSPECT = "ARTIFICIAL_HUMAN_SUSPECT"
OUTGASSING = "OUTGASSING"
SHORT_ARC = "SHORT_ARC_ARTEFACT"
YARKOVSKY_LEAK = "YARKOVSKY_LEAKAGE"
NATURAL_SMALL = "NATURAL_SMALL_BODY"
NATURAL = "NATURAL_CONSISTENT"
INSUFFICIENT = "INSUFFICIENT_DATA"
UNEXPLAINED = "UNEXPLAINED"

#: Objects the literature has already identified as human hardware that received
#: (or nearly received) a minor-planet designation.  Used as LABELS and as
#: positive controls -- never as a silent delete.  Sources: Reddy et al. 2021
#: (2020 SO = Surveyor 2 Centaur, spectroscopic confirmation); Chodas & Chesley
#: 2002 (J002E3 = Apollo 12 S-IVB); ESA/NEOCC (WT1190F, lunar-origin debris);
#: MPEC retractions (2007 VN84 = Rosetta, 2015 HP116 = Gaia).
KNOWN_ARTIFICIAL: dict[str, str] = {
    "2020 SO": "Surveyor 2 Centaur upper stage (Reddy et al. 2021)",
    "J002E3": "Apollo 12 S-IVB third stage (Chodas & Chesley 2002)",
    "WT1190F": "lunar-origin artificial debris; impacted 2015",
    "6Q0B44E": "artificial, Earth-orbit escapee",
    "2007 VN84": "ESA Rosetta spacecraft; designation retracted",
    "2015 HP116": "ESA Gaia spacecraft; designation retracted",
    "2013 QW1": "probable artificial (rocket body)",
    "2010 KQ": "probable artificial (rocket body); Earth-like heliocentric orbit",
    "1991 VG": "long-debated; Apollo-era upper stage remains the leading explanation",
    "2018 AV2": "Falcon Heavy upper stage / Tesla Roadster",
    "2020 KZ2": "probable artificial",
}


@dataclass
class VetParams:
    earthlike_a_au_lo: float = 0.90
    earthlike_a_au_hi: float = 1.15
    earthlike_e_max: float = 0.20
    earthlike_i_deg_max: float = 10.0
    space_age_year: int = 1957
    small_body_diameter_m: float = 10.0
    #: |corr(A1, A2)| above which the radial term is not separable from Yarkovsky.
    a1_a2_corr_max: float = 0.90
    #: Tighter orbit-quality gate applied to survivors only.
    condition_code_max: float = 2.0
    data_arc_days_min: float = 180.0

    @classmethod
    def from_config(cls, d: dict) -> VetParams:
        v = (d or {}).get("vet", {})
        return cls(
            earthlike_a_au_lo=float(v.get("earthlike_a_au_lo", 0.90)),
            earthlike_a_au_hi=float(v.get("earthlike_a_au_hi", 1.15)),
            earthlike_e_max=float(v.get("earthlike_e_max", 0.20)),
            earthlike_i_deg_max=float(v.get("earthlike_i_deg_max", 10.0)),
            space_age_year=int(v.get("space_age_year", 1957)),
            small_body_diameter_m=float(v.get("small_body_diameter_m", 10.0)),
        )


@dataclass
class Vetting:
    """Per-object vetting outcome."""
    designation: str = ""
    verdict: str = INSUFFICIENT
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    detail_ok: bool = False

    def to_dict(self) -> dict:
        return {"designation": self.designation, "verdict": self.verdict,
                "flags": self.flags, "notes": self.notes, "detail_ok": self.detail_ok}


def _norm(name: str) -> str:
    """Normalise a designation for watchlist matching.

    SBDB ``full_name`` looks like ``"(2020 SO)"`` or ``"433 Eros (A898 PA)"``;
    strip parentheses, packed prefixes and collapse whitespace.
    """
    s = re.sub(r"[()]", " ", str(name or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s.upper()


def _matches_known_artificial(row: pd.Series) -> tuple[str, str] | None:
    hay = " ".join(_norm(row.get(c, "")) for c in ("full_name", "pdes", "name")
                   if c in row.index)
    for key, why in KNOWN_ARTIFICIAL.items():
        if _norm(key) in hay:
            return key, why
    return None


def _earthlike_orbit(row: pd.Series, p: VetParams) -> bool:
    """Near-Earth heliocentric orbit typical of an escaped upper stage."""
    try:
        a = float(row.get("a"))
        e = float(row.get("e"))
        i = float(row.get("i"))
    except (TypeError, ValueError):
        return False
    if not all(map(math.isfinite, (a, e, i))):
        return False
    return (p.earthlike_a_au_lo <= a <= p.earthlike_a_au_hi
            and e <= p.earthlike_e_max and i <= p.earthlike_i_deg_max)


def _a1_a2_correlation(detail: dict) -> float | None:
    """|corr(A1, A2)| from the SBDB covariance matrix, if both were fitted."""
    orbit = (detail or {}).get("orbit") or {}
    cov = orbit.get("covariance") or {}
    labels = cov.get("labels") or cov.get("elements")
    mat = cov.get("data") or cov.get("mat")
    if not labels or not mat:
        return None
    labels = [str(x).strip().upper() for x in labels]
    if "A1" not in labels or "A2" not in labels:
        return None
    ia, ib = labels.index("A1"), labels.index("A2")
    try:
        arr = np.array(mat, dtype=float)
        denom = math.sqrt(arr[ia, ia] * arr[ib, ib])
        if denom <= 0:
            return None
        return abs(float(arr[ia, ib]) / denom)
    except Exception:  # noqa: BLE001
        return None


def _first_obs_year(row: pd.Series) -> int | None:
    v = row.get("first_obs")
    m = re.match(r"\s*(\d{4})", str(v or ""))
    return int(m.group(1)) if m else None


def vet_object(row: pd.Series, detail: dict | None, p: VetParams) -> Vetting:
    """Trace one survivor to a systematic, or fail to.

    Returns :data:`UNEXPLAINED` **only** when every rejection route has been
    checked and none fires -- and even then the ``flags`` list records what
    could not be tested.
    """
    v = Vetting(designation=str(row.get("full_name") or row.get("pdes") or ""))
    v.detail_ok = bool(detail and detail.get("ok"))

    # --- 1. human hardware ---
    hit = _matches_known_artificial(row)
    if hit:
        v.flags.append("known_artificial")
        v.notes.append(f"{hit[0]}: {hit[1]}")
        v.verdict = ARTIFICIAL_HUMAN
        return v

    earthlike = _earthlike_orbit(row, p)
    year = _first_obs_year(row)
    if earthlike:
        v.flags.append("earthlike_orbit")
        v.notes.append("a~1 au, low e, low i: the orbit of an escaped upper stage")
    if year is not None and year >= p.space_age_year:
        v.flags.append("post_space_age_discovery")

    # --- 2. outgassing ---
    names = model_par_names(detail) if detail else []
    pars = model_par_values(detail) if detail else {}
    cometary = {n.strip().upper() for n in names} & COMETARY_MODEL_PARS
    if cometary:
        v.flags.append("cometary_nongrav_model")
        v.notes.append(f"fitted cometary g(r) parameters {sorted(cometary)}: "
                       "A1 is not a radiation-pressure coefficient")
        v.verdict = OUTGASSING
        return v
    dt = row.get("DT")
    if pd.notna(dt) and float(dt or 0) != 0:
        v.flags.append("fitted_DT")
        v.verdict = OUTGASSING
        v.notes.append("non-zero cometary delay parameter DT: outgassing model")
        return v
    if str(row.get("kind", "")).lower().startswith("c"):
        v.flags.append("comet_kind")
        v.verdict = OUTGASSING
        return v

    # --- 3. short-arc / fit quality (tighter than the screen) ---
    cc, arc = row.get("condition_code"), row.get("data_arc")
    if pd.isna(cc) or pd.isna(arc):
        v.flags.append("missing_orbit_quality")
    else:
        if float(cc) > p.condition_code_max:
            v.flags.append("condition_code_high")
        if float(arc) < p.data_arc_days_min:
            v.flags.append("short_arc")
    if {"condition_code_high", "short_arc"} & set(v.flags):
        v.verdict = SHORT_ARC
        v.notes.append("orbit too weakly constrained for the A1 fit to be trusted")
        return v

    # --- 4. Yarkovsky leakage ---
    corr = _a1_a2_correlation(detail) if detail else None
    if corr is None:
        v.flags.append("no_covariance")
        v.notes.append("A1/A2 correlation untested: covariance not retrieved")
    elif corr >= p.a1_a2_corr_max:
        v.flags.append("a1_a2_degenerate")
        v.notes.append(f"|corr(A1,A2)| = {corr:.2f}: radial and transverse terms "
                       "are not separable in this solution")
        v.verdict = YARKOVSKY_LEAK
        return v

    # --- 5. genuinely small natural body ---
    d_m = row.get("diameter_m")
    if pd.notna(d_m) and float(d_m) < p.small_body_diameter_m:
        v.flags.append("sub_10m_body")
        v.notes.append("metre-scale body: radiation pressure is expected and the "
                       "R statistic already normalises for it")

    # --- verdict ---
    if earthlike:
        v.verdict = ARTIFICIAL_SUSPECT
        v.notes.append("high implied area-to-mass on an Earth-like orbit is a "
                       "spent upper stage until spectroscopy says otherwise")
        return v

    r = row.get("R")
    if pd.isna(r):
        v.verdict = INSUFFICIENT
        v.notes.append("R could not be formed (no size or no usable A1)")
    elif float(r) < 10.0:
        v.verdict = NATURAL
        v.notes.append(f"R = {float(r):.2g}: consistent with an ordinary body of "
                       "the observed size")
    elif "sub_10m_body" in v.flags:
        v.verdict = NATURAL_SMALL
    else:
        v.verdict = UNEXPLAINED
        v.notes.append("no rejection route fires; NOT a detection claim -- see "
                       "the untested flags above")
    if pars:
        v.notes.append("fitted non-grav parameters: "
                       + ", ".join(sorted(pars)))
    return v


def vet_table(df: pd.DataFrame, details: dict[str, dict] | None, p: VetParams
              ) -> pd.DataFrame:
    """Vet every row; ``details`` maps designation -> ``sbdb.api`` record."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["designation", "verdict", "flags", "notes",
                                     "detail_ok"])
    details = details or {}
    out = []
    for _, row in df.iterrows():
        key = str(row.get("full_name") or row.get("pdes") or "")
        out.append(vet_object(row, details.get(key) or details.get(_norm(key)), p)
                   .to_dict())
    return pd.DataFrame(out, index=df.index)


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate objects (overlapping pulls inflate candidate counts).

    Uses the **composite** of every identifier column present, not the first
    one found: two rows are the same object only if all their identifiers
    agree.  Keying on a single column silently merges distinct objects whenever
    that column is degenerate or absent, which is the wrong direction to fail --
    it would delete real rows.
    """
    if df is None or len(df) == 0:
        return df
    keys = [k for k in ("spkid", "pdes", "full_name") if k in df.columns]
    if not keys:
        return df.drop_duplicates().reset_index(drop=True)
    return df.drop_duplicates(subset=keys).reset_index(drop=True)
