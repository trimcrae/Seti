"""Positive controls: the thing no other channel in this repository has.

Every other search here — Dyson-sphere excess, laser lines, dimming, the
necrosignature taxonomy — has no confirmed positive.  There is no object known to
be a technosignature, so a screen's sensitivity can only be argued from injection
tests against a *model* of the signal, and if the model is wrong the argument is
worthless.

A solar-system artefact search is the exception.  Human beings have put
artificial objects into heliocentric orbit, several of them were catalogued as
minor planets before anyone realised, and in each case what gave them away is
exactly the observable this channel measures: an area-to-mass ratio no rock can
have.  ``J002E3`` is the Apollo 12 S-IVB third stage; ``WT1190F`` was a
lunar-origin rocket body; ``2020 SO`` was identified as the Centaur upper stage
from the 1966 Surveyor 2 launch and confirmed by near-infrared spectroscopy of
301 stainless steel.  These are real artificial objects, discovered by a survey,
mistaken for asteroids, and then correctly classified.

**If the screen does not recover objects like these, it does not work.**  That is
a falsifiable statement about the pipeline, obtainable from real data, and it is
the strongest methodological claim any channel in this repository can make.

Three control sets, three different jobs
----------------------------------------
``ARTIFICIAL``
    Known human-made objects with minor-planet designations.  These validate the
    *artificiality discriminant* (area-to-mass ratio).  A screen that misses them
    is not measuring what it claims to measure.
``NONGRAV_DETECTED``
    Natural objects with published, reliable Yarkovsky solutions.  These validate
    *sensitivity to a real non-gravitational acceleration*: they must be recovered
    as accelerating, and then correctly rejected as natural because their
    acceleration sits below the momentum ceiling.
``DARK_COMETS``
    The seven inactive bodies of Seligman et al. (2023) whose accelerations
    exceed Yarkovsky expectations.  These are the channel's hardest confusers and
    its specificity test: they *must* be flagged by a magnitude cut — that is what
    makes a magnitude cut insufficient — and they must be separated from an
    engineered object by :func:`~seti.loom.residuals.law_discrimination`
    preferring sublimation, and by their area-to-mass ratios being ordinary.

Provenance discipline
---------------------
Each entry carries the source of its number and a ``confidence`` field, and
values that could not be verified from a primary source are recorded as ``None``
with the reason, never filled in with a plausible figure.  Two consequences worth
stating: ``2020 SO``'s fitted area-to-mass ratio is not in any public source that
could be reached, so it is present as an identification without a number; and
``1991 VG``'s nature is genuinely disputed in the literature, so it is marked as
such and is excluded from the pass/fail arithmetic.

Most of these objects are also *not* currently observable — WT1190F impacted
Earth in November 2015, J002E3 and 2020 SO have left the geocentric neighbourhood
— so the honest expectation is that a Rubin-era sample contains none of them.
:func:`validate` therefore reports ``NO_CONTROLS_PRESENT`` as a first-class
outcome rather than silently passing, and the control set earns its place the
moment the survey catalogues one new artificial object, which it will.
"""

from __future__ import annotations

import math
import re

# ---------------------------------------------------------------------------
# Set 1: known artificial objects that carried minor-planet designations
# ---------------------------------------------------------------------------
ARTIFICIAL: tuple[dict, ...] = (
    {
        "designation": "J002E3",
        "identification": "Apollo 12 S-IVB third stage",
        "amr_m2_kg": 7.9e-3,
        "confidence": "confirmed",
        "source": "published AMR from radiation-pressure orbit fit",
        "note": "discovered 2002 in geocentric orbit; AMR ~26x the natural "
                "small-body locus, implying rho*D ~ 190 kg/m^2",
    },
    {
        "designation": "WT1190F",
        "identification": "lunar-origin rocket body",
        "amr_m2_kg": 1.18e-2,
        "amr_err_m2_kg": 5e-4,
        "confidence": "confirmed",
        "source": "published AMR from radiation-pressure orbit fit",
        "note": "impacted Earth 2015-11-13, so not observable; retained because "
                "its AMR is the best-measured artificial value available",
    },
    {
        "designation": "2020 SO",
        "identification": "Centaur upper stage, 1966 Surveyor 2 launch",
        "amr_m2_kg": None,
        "confidence": "confirmed",
        "source": "identification from orbit and near-infrared spectroscopy "
                  "(301 stainless steel); fitted AMR not in a reachable source",
        "note": "the cleanest end-to-end case: found by a survey as an asteroid, "
                "flagged by anomalous area-to-mass ratio, confirmed "
                "spectroscopically",
    },
    {
        "designation": "2010 KQ",
        "identification": "probable rocket body",
        "amr_m2_kg": None,
        "confidence": "probable",
        "source": "identified as artificial from its orbit and non-asteroidal "
                  "radiation-pressure response",
        "note": "heliocentric orbit almost identical to Earth's; artificial "
                "origin widely accepted but no spectroscopic confirmation",
    },
    {
        "designation": "6Q0B44E",
        "identification": "artificial, origin unidentified",
        "amr_m2_kg": None,
        "confidence": "probable",
        "source": "high area-to-mass ratio in geocentric orbit",
        "note": "artificial by dynamics; which launch it came from is unknown",
    },
    {
        "designation": "2007 VN84",
        "identification": "Rosetta spacecraft",
        "amr_m2_kg": None,
        "confidence": "confirmed",
        "source": "MPC designation retracted on identification with the "
                  "spacecraft during its Earth flyby",
        "note": "included because it is the canonical demonstration that a "
                "survey will designate a spacecraft as a minor planet",
    },
    {
        "designation": "1991 VG",
        "identification": "disputed: Apollo-era booster or natural body",
        "amr_m2_kg": None,
        "confidence": "disputed",
        "source": "argued both ways in the literature",
        "note": "EXCLUDED from pass/fail arithmetic; kept because a screen that "
                "flags it should say so and let a human weigh it",
    },
)

# ---------------------------------------------------------------------------
# Set 2: natural objects with reliable published Yarkovsky solutions
# ---------------------------------------------------------------------------
NONGRAV_DETECTED: tuple[dict, ...] = (
    {"designation": "101955", "name": "Bennu", "a2_au_day2": -4.62e-14,
     "diameter_m": 490.0, "h": 20.2,
     "source": "JPL fitted A2; da/dt = -19.0e-4 au/Myr", "confidence": "confirmed"},
    {"designation": "2009 BD", "name": None, "a2_au_day2": -1.14329e-12,
     "a2_unc_au_day2": 7.902e-14, "diameter_m": 4.0, "h": 28.2,
     "source": "Del Vigna et al. 2018", "confidence": "confirmed",
     "note": "also has a directly fitted AMR of (2.97 +- 0.33)e-4 m^2/kg -- the "
             "natural end of the AMR discriminant"},
    {"designation": "2005 ES70", "name": None, "a2_au_day2": -1.2848e-13,
     "diameter_m": 60.0, "h": 24.0,
     "source": "Del Vigna et al. 2018", "confidence": "confirmed"},
    {"designation": "1999 MN", "name": None, "a2_au_day2": 4.084e-14,
     "diameter_m": None, "h": None,
     "source": "Del Vigna et al. 2018", "confidence": "confirmed"},
    {"designation": "6489", "name": "Golevka", "a2_au_day2": None,
     "diameter_m": 530.0, "h": 19.2,
     "source": "first Yarkovsky detection (Chesley et al. 2003); A2 value not "
               "verified from a reachable primary source",
     "confidence": "confirmed_detection_unverified_value"},
    {"designation": "99942", "name": "Apophis", "a2_au_day2": None,
     "diameter_m": 340.0, "h": 19.7,
     "source": "Yarkovsky detection reported; value not verified here",
     "confidence": "confirmed_detection_unverified_value"},
    {"designation": "152563", "name": None, "a2_au_day2": None,
     "diameter_m": None, "h": None,
     "source": "early Yarkovsky detection (1992 BF); value not verified here",
     "confidence": "confirmed_detection_unverified_value"},
)

# ---------------------------------------------------------------------------
# Set 3: the dark comets --- the hardest confusers
# ---------------------------------------------------------------------------
DARK_COMETS: tuple[dict, ...] = (
    {"designation": "1998 KY26"},
    {"designation": "2005 VL1"},
    {"designation": "2016 NJ33"},
    {"designation": "2010 VL65"},
    {"designation": "2006 RH120"},
    {"designation": "2010 RF12"},
    {"designation": "2003 RM"},
)
DARK_COMET_SOURCE = ("Seligman et al. 2023, PSJ 4, 35 -- significant non-radial "
                     "non-gravitational accelerations in inactive objects, "
                     "attributed to hidden outgassing; extended by Farnocchia & "
                     "Seligman 2024 (PNAS) to two distinct populations")


# ---------------------------------------------------------------------------
# Designation matching
# ---------------------------------------------------------------------------
# A provisional designation begins with a four-digit year in this range.  Used to
# tell "2020 SO" (a provisional designation) from "101955 Bennu" (a permanent
# number followed by a name), which otherwise look structurally identical once
# whitespace is removed.
_PROVISIONAL_YEAR = re.compile(r"^(1[89]|20)\d{2}$")


def normalise_designation(value) -> str:
    """Canonical form for comparing designations across catalogues.

    The same object appears as ``2020 SO``, ``2020SO``, ``(101955)`` and
    ``101955 Bennu`` depending on which table it came from, so a naive string
    comparison silently finds nothing.

    **The trap this function exists to avoid, having fallen into it.**  An earlier
    version stripped whitespace and then matched a leading run of digits as the
    permanent number, dropping any trailing letters as a name.  That turns *every
    provisional designation into its discovery year*: ``2020 SO`` became ``2020``,
    and so did ``2020 AB12`` and every other object discovered that year.  The
    control index then matched hundreds of ordinary asteroids, 287 of them were
    forced into a shortlist as "positive controls", and a live run reported that
    ``2020 SO`` and the Rosetta spacecraft were present in the sample when neither
    was.  A matcher that is too permissive does not merely add noise here — it
    fabricates the one falsifiable check the channel has.

    So the permanent-number branch fires only when the digits stand alone, or when
    they are followed by a *name* (three or more letters) and are not a plausible
    discovery year.  Everything else keeps its letters.
    """
    s = str(value or "").strip()
    if not s:
        return ""
    # Strip an enclosing pair of parentheses, "(2020 SO)" -> "2020 SO", and the
    # MPC's leading "(101955)" number form alike.
    s = re.sub(r"^\(\s*(.*?)\s*\)$", r"\1", s)
    s = re.sub(r"^\(\s*(\d+)\s*\)\s*", r"\1 ", s).strip()
    m = re.fullmatch(r"(\d+)", s)
    if m:
        return str(int(m.group(1)))
    m = re.fullmatch(r"(\d+)\s+([A-Za-z]{3,})", s)      # 101955 Bennu
    if m and not _PROVISIONAL_YEAR.match(m.group(1)):
        return str(int(m.group(1)))
    return re.sub(r"[\s_\-]", "", s).upper()


def _index(entries) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for e in entries:
        out[normalise_designation(e["designation"])] = e
        nm = e.get("name")
        if nm:
            out[normalise_designation(nm)] = e
    return out


def control_index() -> dict[str, dict]:
    """Every control object, keyed by normalised designation, tagged with its set."""
    out: dict[str, dict] = {}
    for name, entries in (("artificial", ARTIFICIAL),
                          ("nongrav_detected", NONGRAV_DETECTED),
                          ("dark_comet", DARK_COMETS)):
        for key, e in _index(entries).items():
            out[key] = {**e, "control_set": name}
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(rows: list[dict], score_key: str, flagged_key: str = "flagged",
             designation_keys=("designation", "name", "ssobjectid")) -> dict:
    """Where did each control object land in the screen's own ranking?

    ``rows`` is the screened sample, ``score_key`` the column the screen ranks on
    (higher = more anomalous), ``flagged_key`` the boolean the screen promoted on.
    For every control present, the percentile of its score within the sample is
    reported, because "was it flagged" depends on a threshold whereas "where did
    it rank" does not — and a screen that puts an artificial object at the 99.7th
    percentile but below a 99.9th-percentile cut has a threshold problem, not a
    measurement problem, and those need different fixes.
    """
    idx = control_index()
    scored: list[tuple[float, dict]] = []
    for r in rows:
        v = _f(r.get(score_key))
        if math.isfinite(v):
            scored.append((v, r))
    scored.sort(key=lambda t: t[0])
    values = [v for v, _ in scored]
    n = len(values)

    found: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        keys = [normalise_designation(r.get(k)) for k in designation_keys
                if r.get(k) not in (None, "")]
        hit = next((idx[k] for k in keys if k in idx), None)
        if hit is None:
            continue
        key = normalise_designation(hit["designation"])
        if key in seen:
            continue
        seen.add(key)
        v = _f(r.get(score_key))
        measured = math.isfinite(v)
        pct = (float(sum(1 for x in values if x <= v)) / n * 100.0
               if n and measured else float("nan"))
        found.append({
            "designation": hit["designation"],
            "control_set": hit["control_set"],
            "identification": hit.get("identification") or hit.get("name"),
            "confidence": hit.get("confidence", "n/a"),
            "score": v,
            "percentile": pct,
            # THREE states, not two.  An object that is in the sample but was never
            # measured -- no acceleration measurement, or never shortlisted for a
            # residual pull -- has not been missed by the screen; it has not been
            # shown to the screen.  Conflating the two made the first working run
            # report SCREEN_INSENSITIVE over 2020 SO and the Rosetta spacecraft
            # purely because neither was in the shortlist, which says nothing at all
            # about sensitivity and would have discredited a working screen.
            "measured": measured,
            "flagged": bool(r.get(flagged_key)),
            "expected_amr_m2_kg": hit.get("amr_m2_kg"),
        })

    out: dict = {"n_sample": len(rows), "n_scored": n, "score_key": score_key,
                 "controls_found": found,
                 "n_artificial_available": sum(
                     1 for e in ARTIFICIAL if e["confidence"] != "disputed"),
                 "dark_comet_source": DARK_COMET_SOURCE}

    art = [c for c in found if c["control_set"] == "artificial"
           and c["confidence"] != "disputed"]
    ng = [c for c in found if c["control_set"] == "nongrav_detected"]
    dc = [c for c in found if c["control_set"] == "dark_comet"]
    art_measured = [c for c in art if c["measured"]]
    out["n_artificial_present"] = len(art)
    out["n_artificial_measured"] = len(art_measured)
    out["n_artificial_flagged"] = sum(1 for c in art if c["flagged"])
    out["n_nongrav_present"] = len(ng)
    out["n_nongrav_measured"] = sum(1 for c in ng if c["measured"])
    out["n_nongrav_flagged"] = sum(1 for c in ng if c["flagged"])
    out["n_dark_comet_present"] = len(dc)
    out["n_dark_comet_measured"] = sum(1 for c in dc if c["measured"])
    out["n_dark_comet_flagged"] = sum(1 for c in dc if c["flagged"])

    if not art:
        out["verdict"] = "NO_CONTROLS_PRESENT"
        out["note"] = ("no known artificial object is in this sample, which is the "
                       "expected outcome -- most have impacted, left the "
                       "neighbourhood, or predate the survey.  The control is "
                       "unexercised, NOT passed; sensitivity to the artificiality "
                       "discriminant is untested until one appears.")
    elif not art_measured:
        out["verdict"] = "CONTROLS_PRESENT_BUT_NOT_MEASURED"
        out["note"] = (
            f"{len(art)} known artificial object(s) are in the sample "
            f"({', '.join(c['designation'] for c in art)}) but none carries a "
            f"score: they were never shortlisted for a residual pull, or have no "
            f"usable acceleration measurement.  This says NOTHING about the "
            f"screen's sensitivity -- the control has not been exercised.  It is "
            f"also an instruction: a positive control that is present and not "
            f"measured is a wasted one, and it should be forced into the "
            f"shortlist.")
    elif sum(1 for c in art_measured if c["flagged"]) == len(art_measured):
        out["verdict"] = "SCREEN_VALIDATED"
        out["note"] = (f"all {len(art_measured)} measured artificial objects in the "
                       f"sample were recovered by the screen"
                       + (f" ({len(art) - len(art_measured)} more are present but "
                          f"unmeasured)" if len(art) > len(art_measured) else ""))
    else:
        missed = len(art_measured) - sum(1 for c in art_measured if c["flagged"])
        out["verdict"] = "SCREEN_INSENSITIVE"
        out["note"] = (f"{missed} of {len(art_measured)} MEASURED artificial objects "
                       f"were not flagged; the screen was shown the signal it is "
                       f"built to find and did not return it, so no null from it is "
                       f"interpretable")
    return out


def _f(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")
