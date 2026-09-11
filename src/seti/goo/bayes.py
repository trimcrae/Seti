"""What the absence of galaxy-scale goo says about the rate at which civilisations
make it, and what that says about us.

Model.  Over a window W the Milky Way produced N ~ Poisson(Lambda) technological
civilisations at uniform times.  Each, independently, produced a self-replicator
that reaches other stars with probability p (the product of "made a goo" and
"the goo got out": p = p_goo * p_escape).  A goo that got out more than t_fill
ago has reached us.

Data.  D_own: we sit in a system whose belt is intact and whose biosphere is
alive.  D_ext: in N_gal external galaxies no galaxy shows the persistent
waste-heat signature such a goo would leave (G-hat III: none of ~1e5 with
gamma > 0.85).

Likelihoods.
  naive / SIA:  L_own(Lambda, p) = exp(-Lambda p f),  f = (W - t_fill) / W
  SSA (anthropic shadow): L_own = 1.  Whatever p is, an observer who exists
                finds an unconsumed system; our own galaxy's state is not data
                about p under self-sampling in a fixed reference class.
  external:     L_ext = exp(-N_gal Lambda_gal p q), q = fraction of the window
                during which the residue is visible (1 if the goo keeps
                replicating, ~t_res / W if it dies).  Not shadowed for galaxies
                beyond the reach of an intergalactic front, which for the
                passive branch is all of them.

Posteriors are computed on a grid over log10 p with a log-uniform prior, for a
grid of Lambda, and marginalised over a log-uniform Lambda prior.  The paper
reports the 95 % upper bound on p and the Bayes factor for "p > p_h" where p_h
is a human-elicited planet-scale accident probability, so the reader sees that
only the escaping branch is touched.
"""
from __future__ import annotations

import math

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


def loguniform_grid(lo: float, hi: float, n: int = 601) -> np.ndarray:
    return np.logspace(math.log10(lo), math.log10(hi), n)


def loglike_own(lam: float, p: np.ndarray, f: float, shadowed: bool) -> np.ndarray:
    if shadowed:
        return np.zeros_like(p)
    return -lam * p * f


def loglike_ext(lam: float, p: np.ndarray, n_gal: float, q: float) -> np.ndarray:
    # P(one galaxy shows nothing) = exp(-lam p q); N_gal independent galaxies
    return -n_gal * lam * p * q


def like_own(lam: float, p: np.ndarray, f: float, shadowed: bool) -> np.ndarray:
    return np.exp(loglike_own(lam, p, f, shadowed))


def like_ext(lam: float, p: np.ndarray, n_gal: float, q: float) -> np.ndarray:
    return np.exp(loglike_ext(lam, p, n_gal, q))


def _loglike(lam: float, p: np.ndarray, *, f_own: float, shadowed: bool,
             n_gal: float, q: float, use_ext: bool) -> np.ndarray:
    ll = loglike_own(lam, p, f_own, shadowed)
    if use_ext:
        ll = ll + loglike_ext(lam, p, n_gal, q)
    return ll


def posterior_p(lam: float, p: np.ndarray, *, f_own: float, shadowed: bool,
                n_gal: float, q: float, use_ext: bool) -> np.ndarray:
    """Posterior density in ln p (log-uniform prior), normalised on the grid.  The
    likelihood is exponentiated after subtracting its maximum so that a bound far
    below the grid's lower edge does not underflow to an all-zero posterior."""
    ll = _loglike(lam, p, f_own=f_own, shadowed=shadowed, n_gal=n_gal, q=q, use_ext=use_ext)
    post = np.exp(ll - ll.max())
    post /= _trapz(post, np.log(p))
    return post


def likelihood_bound(lam: float, f: float, n_gal: float = 1.0, q: float = 1.0, cl: float = 0.95) -> float:
    """Prior-free exclusion: the p at which the likelihood of the null datum falls to 1 - cl.
    For one galaxy (n_gal = 1, q = f) this is -ln(1 - cl) / (lam f) = 3.0 / (lam f)."""
    return -math.log(1.0 - cl) / (lam * n_gal * q * f)


def upper_bound(p: np.ndarray, post: np.ndarray, cl: float = 0.95) -> float:
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (post[1:] + post[:-1]) * np.diff(np.log(p)))])
    cdf /= cdf[-1]
    return float(np.interp(cl, cdf, p))


def prob_exceeds(p: np.ndarray, post: np.ndarray, p_h: float) -> float:
    mask = p >= p_h
    if not mask.any():
        return 0.0
    return float(_trapz(post[mask], np.log(p[mask])))


def bayes_factor_exceeds(p: np.ndarray, post: np.ndarray, p_h: float) -> float:
    """Posterior odds / prior odds that p > p_h under the log-uniform prior."""
    prior = np.ones_like(p)
    prior /= _trapz(prior, np.log(p))
    post_prob = prob_exceeds(p, post, p_h)
    prior_prob = prob_exceeds(p, prior, p_h)
    if prior_prob in (0.0, 1.0) or post_prob in (0.0, 1.0):
        return float("nan")
    return (post_prob / (1 - post_prob)) / (prior_prob / (1 - prior_prob))


def marginal_over_lambda(lams: np.ndarray, p: np.ndarray, **kw) -> np.ndarray:
    """Posterior on p with a log-uniform prior on Lambda over the supplied grid: the
    evidence for each Lambda weights its conditional posterior."""
    acc = np.zeros_like(p)
    for lam in lams:
        ll = _loglike(lam, p, **kw)
        acc += np.exp(ll)  # joint, unnormalised; marginalising Lambda with flat log prior
    acc /= _trapz(acc, np.log(p))
    return acc


def branch_update(p_goo_prior: float, p_escape: float, lr_escape: float) -> dict[str, float]:
    """Map an astronomical likelihood ratio on the escaping branch back onto the
    planet-scale accident probability p_goo.

    Prior: p_goo; of which a fraction p_escape reaches other stars.  The data
    multiply the likelihood of the escaping branch by lr_escape (< 1) and leave
    the planet-bound branch untouched.  Posterior on p_goo:

        p_goo' = p_goo [ (1 - p_escape) + p_escape lr ] / [ 1 - p_goo p_escape (1 - lr) ]
    """
    num = p_goo_prior * ((1.0 - p_escape) + p_escape * lr_escape)
    den = 1.0 - p_goo_prior * p_escape * (1.0 - lr_escape)
    return {"p_goo_prior": p_goo_prior, "p_escape": p_escape, "lr_escape": lr_escape,
            "p_goo_posterior": num / den, "ratio": (num / den) / p_goo_prior}
