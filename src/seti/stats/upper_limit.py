"""Occurrence-rate upper limits on white-dwarf Dyson-swarm-like excess.

If ``k`` candidates survive out of an effective sample of ``n_eff`` white dwarfs
to which the search was sensitive, the fraction ``f`` of white dwarfs hosting a
detectable structure has a one-sided upper limit from Poisson / binomial
statistics.  For the common null case (k=0) the 95% Poisson limit is the
familiar ``f < 3 / n_eff``.  Limits are reported as a function of the sensitivity
threshold (T_dust, tau) via the completeness map, mirroring the structure of the
null SETI-Ellipsoid result (Cabrales et al. 2024).
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats


@dataclass
class OccurrenceLimit:
    k: int
    n_eff: float
    confidence: float
    f_upper: float          # upper limit on the true fraction
    f_point: float          # point estimate k / n_eff


def poisson_upper_limit(k: int, confidence: float = 0.95) -> float:
    """One-sided upper limit on a Poisson mean given ``k`` observed events.

    Uses the chi-square relation mu_up = 0.5 * ppf(confidence; 2k+2).
    For k=0, confidence=0.95 this returns ~3.0.
    """
    return 0.5 * stats.chi2.ppf(confidence, 2 * k + 2)


def occurrence_upper_limit(
    k: int, n_eff: float, completeness: float = 1.0, confidence: float = 0.95
) -> OccurrenceLimit:
    """Upper limit on the occurrence fraction, corrected for completeness.

    ``completeness`` (0-1) is the fraction of true structures the pipeline would
    recover at the sensitivity threshold of interest; dividing by it inflates the
    limit to an intrinsic-occurrence limit.
    """
    if n_eff <= 0:
        raise ValueError("n_eff must be positive")
    comp = max(completeness, 1e-6)
    mu_up = poisson_upper_limit(k, confidence)
    f_upper = mu_up / (n_eff * comp)
    return OccurrenceLimit(
        k=int(k),
        n_eff=float(n_eff),
        confidence=float(confidence),
        f_upper=float(f_upper),
        f_point=float(k) / float(n_eff),
    )
