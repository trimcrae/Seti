"""GRAVE age stack: does a residue recur at one stratigraphic level across sections?

The K-Pg iridium is believed because it is at the same level in Gubbio,
Stevns Klint, Caravaca, Raton and a hundred more sections; a spike in one
sample of one core is a contamination hypothesis until a second section
shows it.  So every candidate is placed on the time axis and against the
catalogue of extinction / crisis boundaries, and the statistic asked is
whether candidates cluster at a boundary in *independent sections* beyond
the rate at which the whole population produces them.

Boundary ages are GTS2020 (Gradstein et al. 2020) unless noted; the window
half-widths are the age-model uncertainty typical of SGP interpreted ages
near each boundary, not the duration of the event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

#: (key, name, age Ma, default half-width Myr, note)
BOUNDARIES: tuple[dict, ...] = (
    {"key": "ediacaran_cambrian", "name": "Ediacaran-Cambrian", "age_ma": 538.8, "half_width_myr": 3.0,
     "note": "Ediacaran biota loss; base Cambrian"},
    {"key": "end_ordovician", "name": "End-Ordovician (Hirnantian)", "age_ma": 444.5, "half_width_myr": 2.0,
     "note": "first of the Big Five; contested trigger (glaciation; a GRB was proposed)"},
    {"key": "kellwasser", "name": "Late Devonian Kellwasser (F-F)", "age_ma": 372.2, "half_width_myr": 2.0,
     "note": "Frasnian-Famennian; a nearby supernova was proposed (Fields et al. 2020)"},
    {"key": "hangenberg", "name": "Hangenberg (D-C)", "age_ma": 358.9, "half_width_myr": 2.0,
     "note": "Devonian-Carboniferous"},
    {"key": "capitanian", "name": "End-Guadalupian (Capitanian)", "age_ma": 259.5, "half_width_myr": 2.0,
     "note": "Emeishan LIP"},
    {"key": "end_permian", "name": "End-Permian", "age_ma": 251.9, "half_width_myr": 2.0,
     "note": "Siberian Traps; the largest"},
    {"key": "carnian", "name": "Carnian Pluvial Episode", "age_ma": 233.0, "half_width_myr": 2.0,
     "note": "Wrangellia LIP; contested"},
    {"key": "end_triassic", "name": "End-Triassic", "age_ma": 201.4, "half_width_myr": 1.5,
     "note": "CAMP"},
    {"key": "toarcian", "name": "Toarcian OAE", "age_ma": 183.0, "half_width_myr": 1.5,
     "note": "Karoo-Ferrar"},
    {"key": "oae2", "name": "Cenomanian-Turonian OAE2", "age_ma": 93.9, "half_width_myr": 1.0,
     "note": "Caribbean LIP"},
    {"key": "k_pg", "name": "Cretaceous-Paleogene", "age_ma": 66.0, "half_width_myr": 1.0,
     "note": "Chicxulub + Deccan; the iridium positive control"},
    {"key": "petm", "name": "PETM", "age_ma": 56.0, "half_width_myr": 1.0,
     "note": "carbon isotope excursion; contested"},
    {"key": "eocene_oligocene", "name": "Eocene-Oligocene", "age_ma": 33.9, "half_width_myr": 1.0,
     "note": "Popigai / Chesapeake impacts, Antarctic glaciation"},
)


def boundary_table(conf: dict | None = None) -> list[dict]:
    b = (conf or {}).get("boundaries")
    return [dict(x) for x in (b or BOUNDARIES)]


def assign_boundary(age: float, min_age: float | None, max_age: float | None,
                    boundaries: list[dict], *, max_span_myr: float = 20.0,
                    width_scale: float = 1.0) -> str:
    """Which boundary window a sample falls in ('' = none).

    A sample is in a window if its interpreted age is within the half-width,
    or if its own [min_age, max_age] bracket (when narrower than
    ``max_span_myr``) overlaps the window.  The nearest boundary wins when two
    overlap.
    """
    if age is None or not np.isfinite(age):
        return ""
    best, best_d = "", np.inf
    for b in boundaries:
        w = float(b["half_width_myr"]) * width_scale
        lo, hi = b["age_ma"] - w, b["age_ma"] + w
        d = abs(age - b["age_ma"])
        inside = lo <= age <= hi
        if not inside and min_age is not None and max_age is not None and np.isfinite(min_age) \
                and np.isfinite(max_age) and 0 <= (max_age - min_age) <= max_span_myr:
            inside = not (max_age < lo or min_age > hi)
        if inside and d < best_d:
            best, best_d = b["key"], d
    return best


def section_key(row: dict) -> str:
    """An independent section: the named section, else the 0.1-degree site cell."""
    s = str(row.get("section") or "").strip()
    if s and s.lower() not in ("nan", "none"):
        return s
    lat, lon = row.get("lat"), row.get("lon")
    try:
        if lat is not None and lon is not None and np.isfinite(float(lat)) and np.isfinite(float(lon)):
            return f"cell_{round(float(lat), 1)}_{round(float(lon), 1)}"
    except (TypeError, ValueError):
        pass
    return str(row.get("sample_id") or "unknown")


def age_stack(df: pd.DataFrame, boundaries: list[dict], *, candidate_col: str = "is_candidate",
              boundary_col: str = "boundary", section_col: str = "section_key",
              cluster_p: float = 0.01) -> dict:
    """Per boundary: sections sampled, candidates, candidate sections, clustering p.

    The unit is the *section*, not the sample: a section with >= 1 candidate
    counts once however many analyses it contributed, because ten aliquots of
    one core are one opportunity for contamination, not ten.

    The test is conditional and therefore does not let the window contaminate
    its own null.  Given that the whole corpus produced ``C`` candidate
    sections out of ``S`` sections, and that ``S_w`` of those sections carry
    a sample inside the window, the count in the window under "candidates fall
    where sections are, regardless of stratigraphic level" is
    ``Hypergeometric(S, C, S_w)`` and ``p = P(X >= k)``.  (The earlier
    Poisson-with-global-rate form estimated the background from a rate the
    window's own candidates had already inflated, which is anticonservative
    for the rate and, worse, loses power exactly when every candidate sits at
    one level.)  A window with >= 2 candidate sections and ``p < cluster_p``
    is a *stratigraphic cluster* -- the only thing that promotes a per-sample
    candidate to a boundary-level claim.  One candidate section is always
    ``single_section``: that is the contamination hypothesis, not a find.
    """
    n_sec_total = int(df[section_col].nunique())
    cand_sec_all = set(df.loc[df[candidate_col].astype(bool), section_col].astype(str))
    cand_sec_total = len(cand_sec_all)
    rate = cand_sec_total / n_sec_total if n_sec_total else 0.0
    out = {"n_sections_total": n_sec_total, "n_candidate_sections_total": cand_sec_total,
           "section_candidate_rate": round(float(rate), 6), "test": "hypergeometric_on_sections",
           "cluster_p": float(cluster_p), "boundaries": {}}
    for b in boundaries:
        w = df[df[boundary_col] == b["key"]]
        n_sec = int(w[section_col].nunique())
        cand = w[w[candidate_col].astype(bool)]
        in_window = set(cand[section_col].astype(str))
        n_cand_sec = len(in_window)
        exp = (cand_sec_total * n_sec / n_sec_total) if n_sec_total else 0.0
        if n_cand_sec == 0:
            p = 1.0
        elif cand_sec_total <= 0 or n_sec <= 0:
            p = 1.0
        else:
            p = float(hypergeom.sf(n_cand_sec - 1, n_sec_total, cand_sec_total, n_sec))
        if n_cand_sec == 0:
            status = "no_candidate"
        elif n_cand_sec == 1:
            status = "single_section"
        elif p < cluster_p:
            status = "STRATIGRAPHIC_CLUSTER"
        else:
            status = "multi_section_at_background_rate"
        out["boundaries"][b["key"]] = {
            "name": b["name"], "age_ma": b["age_ma"], "half_width_myr": b["half_width_myr"],
            "n_samples": int(len(w)), "n_sections": n_sec, "n_candidates": int(len(cand)),
            "n_candidate_sections": n_cand_sec, "expected_candidate_sections": round(float(exp), 4),
            "n_candidate_sections_elsewhere": int(len(cand_sec_all - in_window)),
            "p_hypergeom": round(p, 6), "status": status,
            "candidate_sections": sorted(cand[section_col].astype(str).unique().tolist())[:25],
        }
    return out


def age_histogram(df: pd.DataFrame, *, age_col: str = "age", candidate_col: str = "is_candidate",
                  bin_myr: float = 10.0, max_age: float = 4000.0) -> list[dict]:
    a = pd.to_numeric(df[age_col], errors="coerce")
    ok = a.notna() & (a >= 0) & (a <= max_age)
    edges = np.arange(0, max_age + bin_myr, bin_myr)
    n_all, _ = np.histogram(a[ok], bins=edges)
    n_c, _ = np.histogram(a[ok & df[candidate_col].astype(bool)], bins=edges)
    return [{"age_lo": float(edges[i]), "age_hi": float(edges[i + 1]), "n": int(n_all[i]), "n_candidates": int(n_c[i])}
            for i in range(len(n_all)) if n_all[i] > 0]


__all__ = ["BOUNDARIES", "age_histogram", "age_stack", "assign_boundary", "boundary_table", "section_key"]
