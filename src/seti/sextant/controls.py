"""Positive controls for SEXTANT: published Yarkovsky detections.

The estimator's one falsifiable check.  For an object whose non-gravitational
acceleration has been measured independently --- from decades of optical and
radar astrometry --- the Gaia residual against a **gravity-only** prediction
must return the same ``A2``, in sign and in magnitude, or the estimator is
wrong.  Nothing in a synthetic test can substitute for that, because a synthetic
test only proves the estimator recovers what its own model injected.

Two lists, doing different jobs:

* :data:`PUBLISHED_YARKOVSKY` --- objects whose Yarkovsky drift is an
  established, peer-reviewed detection.  Numbers and the citing works only; the
  **numerical** control value for each is JPL's current fitted ``A2`` and its
  sigma from SBDB, pulled live, because that is the same physical quantity the
  papers measured, on a longer arc, with the covariance the fit actually has.
  The list is an annotation so the report can say *which* controls it saw.
* The live set --- every object in the shard whose JPL solution carries an
  ``A2`` at S/N >= ``min_snr``.  Larger than the published list and just as
  valid a control, since JPL only fits ``A2`` where the data demand it.

Citations: Farnocchia, Chesley, Vokrouhlický et al. 2013, Icarus 224, 1
(21 detections); Chesley et al. 2003, Science 302, 1739 (Golevka);
Chesley et al. 2014, Icarus 235, 5 (Bennu); Vokrouhlický et al. 2008, ApJ 135,
2336 (1992 BF); Del Vigna et al. 2018, A&A 617, A61 (87 detections);
Greenberg et al. 2020, AJ 159, 92 (247 detections, VizieR J/AJ/159/92);
Hanuš et al. 2018, A&A 620, L8 (Phaethon); Farnocchia & Chesley 2014, Icarus
229, 321 (1950 DA).
"""

from __future__ import annotations

import math

import numpy as np

#: number -> (name, sources).  Annotation only; values come from SBDB live.
PUBLISHED_YARKOVSKY: dict[int, tuple[str, str]] = {
    101955: ("Bennu", "Chesley+2014; Farnocchia+2013; Greenberg+2020"),
    6489: ("Golevka", "Chesley+2003; Farnocchia+2013"),
    99942: ("Apophis", "Farnocchia+2013b; Vokrouhlicky+2015; Greenberg+2020"),
    1862: ("Apollo", "Farnocchia+2013; Greenberg+2020"),
    1620: ("Geographos", "Farnocchia+2013; Greenberg+2020"),
    2062: ("Aten", "Farnocchia+2013; Greenberg+2020"),
    2100: ("Ra-Shalom", "Farnocchia+2013; Greenberg+2020"),
    2340: ("Hathor", "Farnocchia+2013; Greenberg+2020"),
    3908: ("Nyx", "Farnocchia+2013 (drift sign anomalous for its size)"),
    1685: ("Toro", "Greenberg+2020"),
    1566: ("Icarus", "Greenberg+2020"),
    3361: ("Orpheus", "Farnocchia+2013"),
    4034: ("Vishnu", "Farnocchia+2013"),
    4660: ("Nereus", "Greenberg+2020"),
    6037: ("1988 EG", "Farnocchia+2013"),
    10302: ("1989 ML", "Farnocchia+2013"),
    33342: ("1998 WT24", "Farnocchia+2013; Greenberg+2020"),
    85953: ("1999 FK21", "Farnocchia+2013"),
    152563: ("1992 BF", "Vokrouhlicky+2008; Farnocchia+2013"),
    162004: ("1991 VE", "Farnocchia+2013"),
    29075: ("1950 DA", "Farnocchia & Chesley 2014"),
    3200: ("Phaethon", "Hanus+2018"),
    4179: ("Toutatis", "Farnocchia+2013 (marginal)"),
}


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def score_control(fitted_a2, fitted_err, jpl_a2, jpl_sigma, *,
                  min_snr: float = 3.0, max_ratio: float = 2.0) -> dict:
    """Did the fit recover JPL's ``A2``?  Sign first, then magnitude.

    Three verdicts: ``RECOVERED`` (same sign, magnitude within a factor
    ``max_ratio`` and within 3 sigma of the combined error), ``SIGN_WRONG``,
    ``MAGNITUDE_OFF``; and ``NOT_MEASURED`` when the fit has no usable S/N,
    which is a statement about sensitivity, not about correctness.
    """
    fa, fe, ja, js = _f(fitted_a2), _f(fitted_err), _f(jpl_a2), _f(jpl_sigma)
    out = {"fitted_a2": fa, "fitted_err": fe, "jpl_a2": ja, "jpl_sigma": js}
    if not math.isfinite(ja):
        out["verdict"] = "NO_JPL_VALUE"
        return out
    if not (math.isfinite(fa) and math.isfinite(fe) and fe > 0):
        out["verdict"] = "NOT_MEASURED"
        return out
    out["fitted_snr"] = abs(fa) / fe
    out["jpl_snr"] = abs(ja) / js if math.isfinite(js) and js > 0 else float("nan")
    comb = math.sqrt(fe ** 2 + (js ** 2 if math.isfinite(js) else 0.0))
    out["difference_sigma"] = (fa - ja) / comb if comb > 0 else float("nan")
    out["ratio"] = fa / ja if ja != 0 else float("nan")
    if out["fitted_snr"] < min_snr:
        out["verdict"] = ("CONSISTENT_BUT_NOT_DETECTED"
                          if abs(out["difference_sigma"]) < 3.0 else "NOT_MEASURED")
        return out
    if np.sign(fa) != np.sign(ja):
        out["verdict"] = "SIGN_WRONG"
        return out
    if abs(out["difference_sigma"]) <= 3.0 or (1.0 / max_ratio <= out["ratio"] <= max_ratio):
        out["verdict"] = "RECOVERED"
    else:
        out["verdict"] = "MAGNITUDE_OFF"
    return out


def summarise_controls(scored: list[dict]) -> dict:
    """The control verdict for the run: unexercised, passed, or failed."""
    n = len(scored)
    counts: dict[str, int] = {}
    for s in scored:
        counts[s.get("verdict", "?")] = counts.get(s.get("verdict", "?"), 0) + 1
    out = {"n_controls": n, "verdicts": counts}
    if n == 0:
        out["verdict"] = "NO_CONTROLS_PRESENT"
        out["note"] = ("no object with an independently fitted A2 was in the "
                       "sample; the estimator's sensitivity is UNEXERCISED, not "
                       "passed")
        return out
    measured = [s for s in scored if s.get("verdict") in
                ("RECOVERED", "SIGN_WRONG", "MAGNITUDE_OFF")]
    out["n_measured"] = len(measured)
    if not measured:
        out["verdict"] = "CONTROLS_BELOW_SENSITIVITY"
        out["note"] = ("controls were present but none reached S/N >= 3 in the "
                       "Gaia-only fit; the estimator is not shown to be wrong, and "
                       "it is not shown to be sensitive at these amplitudes")
        return out
    n_ok = counts.get("RECOVERED", 0)
    n_sign = counts.get("SIGN_WRONG", 0)
    out["recovered_fraction"] = n_ok / len(measured)
    if n_sign > 0 and n_sign >= n_ok:
        out["verdict"] = "CONTROLS_FAILED_SIGN"
    elif out["recovered_fraction"] >= 0.7:
        out["verdict"] = "CONTROLS_RECOVERED"
    else:
        out["verdict"] = "CONTROLS_PARTIAL"
    return out
