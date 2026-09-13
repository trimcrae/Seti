"""The contamination gauntlet for GROWTH: named vetoes with counters (pure).

Every veto is a mechanism that can produce an apparent Kepler -> TESS depth
change without anything being built:

``grazing``                     ``b > 1 - k``: the depth is set by the chord,
                                and any tiny precession moves it (KOI-120,
                                Kepler-13Ab, Kepler-47d are the named cases)
``ttv_system``                  known large-TTV systems (config list, from
                                Holczer et al. 2016 --- ``verify``): a folded
                                depth is smeared by an amount that depends on
                                the baseline and the cadence, differently in
                                each mission
``fpflag_*``                    the four KOI false-positive flags
                                (``nt`` not transit-like, ``ss`` stellar
                                eclipse, ``co`` centroid offset, ``ec``
                                ephemeris-match contamination)
``koi_false_positive``          ``koi_disposition`` / ``koi_pdisposition``
``toi_disposition_not_candidate`` ``tfopwg_disp`` outside {PC, CP, KP}
``period_alias``                the TESS period is an integer alias of the
                                Kepler one: the folded events are not the same
                                set of transits
``neighbours_not_checked``      the Gaia cone for this star failed: isolation
                                is unknown, not established
``neighbour_can_supply_change`` the growth does not survive the full range of
                                the dilution ambiguity --- if every Gaia
                                neighbour inside the search radius had already
                                been removed by the pipeline (SPOC's CROWDSAP
                                does exactly this), the corrected ratio would
                                be ``R_corr (1 - c_max)``; the candidate must
                                stay above ``n_candidate`` sigma there too

Report-only flags (never a veto): ``multi_sector_scatter:not_checked`` (the
per-sector TESS depths are not in the catalogue --- stage 2), and
``within_han2025_deficit_band`` (the raw ratio sits where Han et al. 2025's
~6 % radius / ~12 % depth deficit puts an ordinary planet).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .drift import (
    CLASS_CANDIDATE,
    CLASSES,
    DUR_FIXED_B,
    DUR_INCONCLUSIVE,
    DUR_INCONSISTENT,
    DUR_TRACKS_B,
    classify,
    population_offset,
)

FPFLAGS = ("koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec")
VETO_ORDER = ("grazing", "ttv_system", "fpflag_nt", "fpflag_ss", "fpflag_co", "fpflag_ec",
              "koi_false_positive", "toi_disposition_not_candidate", "period_alias",
              "neighbours_not_checked", "neighbour_can_supply_change")
REPORT_FLAGS = ("multi_sector_scatter:not_checked", "within_han2025_deficit_band",
                "duration_inconclusive", "ld_teff_missing")

_KOI_HOST_RE = re.compile(r"^(K\d+)\.\d+$", re.IGNORECASE)


@dataclass
class VetParams:
    ttv_systems: list[str] = field(default_factory=list)
    toi_candidate_dispositions: tuple[str, ...] = ("PC", "CP", "KP")
    koi_candidate_dispositions: tuple[str, ...] = ("CONFIRMED", "CANDIDATE")
    n_candidate: float = 5.0
    n_consistent: float = 3.0
    han2025_expected_deficit: float = 0.12
    subtract_population_median: bool = True

    @classmethod
    def from_config(cls, conf: dict) -> VetParams:
        v = conf.get("vet") or {}
        d = conf.get("drift") or {}
        return cls(ttv_systems=[str(s) for s in (v.get("ttv_systems") or [])],
                   toi_candidate_dispositions=tuple(v.get("toi_candidate_dispositions")
                                                    or ("PC", "CP", "KP")),
                   koi_candidate_dispositions=tuple(v.get("koi_candidate_dispositions")
                                                    or ("CONFIRMED", "CANDIDATE")),
                   n_candidate=float(d.get("n_candidate", 5.0)),
                   n_consistent=float(d.get("n_consistent", 3.0)),
                   han2025_expected_deficit=float(d.get("han2025_expected_deficit", 0.12)),
                   subtract_population_median=bool(d.get("subtract_population_median", True)))


def _norm(s) -> str:
    return " ".join(str(s).split()).strip().lower() if s is not None and s == s else ""


def host_names(row: dict) -> set[str]:
    """Every name under which this system might appear in a TTV list."""
    out = set()
    kn = _norm(row.get("kepler_name"))
    if kn:
        out.add(kn)
        out.add(re.sub(r"\s+[a-z]$", "", kn))                 # "kepler-9 b" -> "kepler-9"
    ko = str(row.get("kepoi_name") or "").strip()
    m = _KOI_HOST_RE.match(ko)
    if m:
        out.add(m.group(1).lower())                           # "K00142.01" -> "k00142"
        out.add(f"koi-{int(m.group(1)[1:])}")                 # -> "koi-142"
    kid = row.get("kepid")
    if kid is not None and kid == kid:
        try:
            out.add(f"kic {int(float(kid))}")
        except (TypeError, ValueError):
            pass
    return out


def grazing(b: float, k: float) -> bool:
    return bool(np.isfinite(b) and np.isfinite(k) and b > 1.0 - k)


def ttv_system(row: dict, systems) -> bool:
    names = {_norm(s) for s in systems}
    return bool(host_names(row) & names)


def neighbour_can_supply_change(ln_ratio_corr: float, sigma: float, contam_max: float, *,
                                n_candidate: float, offset: float = 0.0) -> bool:
    """True when ``ln R_corr + ln(1 - c_max)`` drops below the candidate threshold."""
    if not (np.isfinite(ln_ratio_corr) and np.isfinite(sigma)) or sigma <= 0:
        return False
    c = float(contam_max) if np.isfinite(contam_max) else 0.0
    if c <= 0:
        return False
    c = min(c, 0.999)
    return (ln_ratio_corr + math.log(1.0 - c) - offset) / sigma < n_candidate


def vet_planet(rec: dict, params: VetParams, *, offset: float = 0.0) -> dict:
    """Vetoes, flags and the class for one drift record (a dict)."""
    vetoes: list[str] = []
    flags: list[str] = ["multi_sector_scatter:not_checked"]
    b = float(rec.get("b_kepler", np.nan))
    k = float(rec.get("k_kepler", np.nan))
    if grazing(b, k):
        vetoes.append("grazing")
    if ttv_system(rec, params.ttv_systems):
        vetoes.append("ttv_system")
    for col in FPFLAGS:
        v = rec.get(col)
        try:
            if v is not None and v == v and int(float(v)) != 0:
                vetoes.append("fpflag_" + col.split("_")[-1])
        except (TypeError, ValueError):
            pass
    kd = str(rec.get("koi_disposition") or "").strip().upper()
    kpd = str(rec.get("koi_pdisposition") or "").strip().upper()
    if kd not in params.koi_candidate_dispositions or (kpd and kpd not in
                                                        params.koi_candidate_dispositions):
        vetoes.append("koi_false_positive")
    td = str(rec.get("tfopwg_disp") or "").strip().upper()
    if td not in params.toi_candidate_dispositions:
        vetoes.append("toi_disposition_not_candidate")
    alias = str(rec.get("period_alias") or "1")
    if alias != "1":
        vetoes.append("period_alias")
    nstat = str(rec.get("neighbours_status") or "")
    ln_corr = float(rec.get("ln_ratio_corr", np.nan))
    sig = float(rec.get("sigma_ln_ratio", np.nan))
    if nstat != "OK":
        vetoes.append("neighbours_not_checked")
    elif neighbour_can_supply_change(ln_corr, sig, float(rec.get("contam_max", 0.0)),
                                     n_candidate=params.n_candidate, offset=offset):
        vetoes.append("neighbour_can_supply_change")
    ln_raw = float(rec.get("ln_ratio_raw", np.nan))
    han = math.log(1.0 - params.han2025_expected_deficit)
    if np.isfinite(ln_raw) and np.isfinite(sig) and abs(ln_raw - han) < params.n_consistent * sig:
        flags.append("within_han2025_deficit_band")
    if rec.get("duration_verdict") == DUR_INCONCLUSIVE:
        flags.append("duration_inconclusive")
    if "teff_missing" in str(rec.get("ld_note") or ""):
        flags.append("ld_teff_missing")

    cls, z = classify(ln_corr, sig, str(rec.get("duration_verdict") or ""), vetoes,
                      n_consistent=params.n_consistent, n_candidate=params.n_candidate,
                      offset=offset)
    ordered = [v for v in VETO_ORDER if v in vetoes] + [v for v in vetoes if v not in VETO_ORDER]
    return {"class": cls, "z_ratio": z, "population_offset": float(offset),
            "vetoes": ";".join(ordered), "first_veto": ordered[0] if ordered else "",
            "flags": ";".join(flags),
            "would_be_candidate_without_vetoes": bool(
                np.isfinite(z) and z >= params.n_candidate
                and rec.get("duration_verdict") == DUR_FIXED_B and ordered)}


def vet_table(df: pd.DataFrame, params: VetParams) -> tuple[pd.DataFrame, dict]:
    """Vet every row; the population offset is measured on the same rows first."""
    if not len(df):
        return df.copy(), {"population_offset": 0.0, "n_measured": 0}
    offset, n_meas = population_offset(df.get("ln_ratio_corr", pd.Series(dtype=float)),
                                       enabled=params.subtract_population_median)
    recs = df.to_dict(orient="records")
    vet = [vet_planet(r, params, offset=offset) for r in recs]
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(vet)], axis=1)
    han = math.log(1.0 - params.han2025_expected_deficit)
    med_raw = float(np.nanmedian(pd.to_numeric(df["ln_ratio_raw"], errors="coerce"))) \
        if "ln_ratio_raw" in df else float("nan")
    return out, {"population_offset": float(offset), "n_measured": int(n_meas),
                 "population_median_ln_ratio_raw": med_raw,
                 "han2025_expected_ln_ratio": han,
                 "subtract_population_median": bool(params.subtract_population_median)}


def rejection_counters(df: pd.DataFrame) -> dict:
    """Class counts, first-veto counts, every veto raised, every flag raised."""
    classes = {c: 0 for c in CLASSES}
    first = {v: 0 for v in VETO_ORDER}
    raised = {v: 0 for v in VETO_ORDER}
    flags = {f: 0 for f in REPORT_FLAGS}
    dur = {DUR_FIXED_B: 0, DUR_TRACKS_B: 0, DUR_INCONSISTENT: 0, DUR_INCONCLUSIVE: 0}
    would = 0
    if len(df):
        for c in df.get("class", pd.Series(dtype=str)):
            classes[c] = classes.get(c, 0) + 1
        for v in df.get("first_veto", pd.Series(dtype=str)):
            if v:
                first[v] = first.get(v, 0) + 1
        for s in df.get("vetoes", pd.Series(dtype=str)):
            for v in str(s or "").split(";"):
                if v:
                    raised[v] = raised.get(v, 0) + 1
        for s in df.get("flags", pd.Series(dtype=str)):
            for f in str(s or "").split(";"):
                if f:
                    flags[f] = flags.get(f, 0) + 1
        for d in df.get("duration_verdict", pd.Series(dtype=str)):
            dur[d] = dur.get(d, 0) + 1
        would = int(df.get("would_be_candidate_without_vetoes",
                           pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    return {"classes": classes, "first_veto": first, "vetoes_raised": raised,
            "flags_raised": flags, "duration_verdicts": dur,
            "n_candidates": int(classes.get(CLASS_CANDIDATE, 0)),
            "n_would_be_candidate_without_vetoes": would}


__all__ = ["FPFLAGS", "REPORT_FLAGS", "VETO_ORDER", "VetParams", "grazing", "host_names",
           "neighbour_can_supply_change", "rejection_counters", "ttv_system", "vet_planet",
           "vet_table"]
