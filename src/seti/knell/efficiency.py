"""Injection-measured per-block detection efficiency --- the load-bearing module.

This is the module that decides whether KNELL is a search for astrophysics or a
search for ZTF's observing calendar.

The problem
-----------
A "cessation" is a **non-detection** in a late block of a signal that *was*
detected in an early block.  A non-detection has two causes:

1. the signal stopped                                     (the target);
2. the block could not have detected it anyway            (the confounder).

Cause 2 is not rare, it is the default.  Between two blocks of the same survey
the epoch count changes, the seasonal window length changes, the alias comb
moves, the photometric errors change with depth and moon phase, and each of
those changes the probability of recovering a *fixed* signal.  A statistic that
compares peak significance early against peak significance late, without
correcting for this, ranks stars by how much the cadence degraded.  The sibling
RUST channel measured the analogous failure in the second moment: an uncorrected
version of its statistic flagged **46 of 60** injected confounders.

The fix
-------
Measure the detection probability directly, by injection, **in the block itself**:

    eta(P, A, block) = P( the blind block-level detector fires
                        | a signal of period P and amplitude A is present,
                          observed at *this block's* epochs, with *this block's*
                          noise, and scored against *this block's* own
                          permutation threshold )

and inject into the **observed magnitudes of that block**, not into synthetic
Gaussian noise.  By the cessation hypothesis the late block contains no signal,
so ``y_obs + A sin(2 pi f t + phi)`` is a light curve with exactly the block's
real sampling, real error distribution, real correlated systematics and real
outliers, plus a signal of known amplitude.  Nothing about the noise has to be
modelled, so nothing about the noise can be modelled wrongly.  (Gaussian and
residual-resampling modes exist for the case where only a sampling pattern is
available --- e.g. forecasting --- and are labelled as such in the output.)

The detection criterion used in the injection is **byte-for-byte the criterion
used on the data**: max GLS power over the same frequency grid, against the same
permutation threshold computed from the same block.  An efficiency measured with
a different criterion than the search uses would be worse than none.

What the number is then used for
--------------------------------
* **A gate.**  If ``eta`` in a post-transition block is low, that block carries
  no information about cessation and the star is not a candidate --- verdict
  ``low_efficiency``, not "ceased".  This single rule is what makes a degrading
  cadence unable to produce a candidate.
* **A p-value.**  Under the null "the signal persisted", the probability of the
  observed run of non-detections is ``prod_i (1 - eta_i)`` over post blocks.
  With a finite number of injection trials ``1 - eta`` is only bounded, never
  measured, once every trial is recovered; the module therefore reports a
  **Clopper-Pearson upper bound** and flags the result as pinned at the
  resolution floor, so the p-value is quoted as an inequality and never as a
  point estimate.
* **A sensitivity curve.**  ``efficiency_curve`` reports the amplitude at which
  each block reaches 50% and 90% efficiency, which is the honest statement of
  what that block could and could not have seen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blocks import Block, frequency_grid, gls_power_batch, permutation_threshold

# Recovery counts as a frequency match if the recovered peak lands within this
# fractional distance of the injected frequency or one of its low harmonics /
# sub-harmonics.  Eclipsing binaries routinely peak at 2/P_orb, and a genuinely
# recovered clock at the wrong harmonic is still a recovered clock.
HARMONIC_FACTORS = (0.5, 1.0, 2.0, 3.0)
FREQ_TOL_FRAC = 0.02


@dataclass
class Efficiency:
    """Injection-measured detection efficiency of one block at one (P, A)."""

    block_index: int
    period: float
    amplitude_mmag: float
    n_trials: int
    n_detected: int
    n_detected_freqmatch: int
    eta: float
    eta_freqmatch: float
    miss_upper: float          # Clopper-Pearson 95% upper bound on (1 - eta)
    at_resolution_floor: bool  # True when zero misses were observed
    noise_mode: str
    power_threshold: float
    n_epochs: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    """Upper ``conf`` confidence bound on a binomial rate from ``k`` of ``n``.

    For ``k = 0`` this is the rule of three generalised exactly:
    ``1 - (1 - conf)^(1/n)``.  Using it rather than ``k/n`` is what stops the
    channel from reporting ``p = 0`` when it merely ran out of trials.
    """
    n = int(max(n, 1))
    k = int(np.clip(k, 0, n))
    if k >= n:
        return 1.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(conf, k + 1, n - k))
    except Exception:                                    # noqa: BLE001
        if k == 0:
            return float(1.0 - (1.0 - conf) ** (1.0 / n))
        return float(min(1.0, (k + 2.0) / n))


def _injected_matrix(block: Block, freq: float, amp_mag: float, n_trials: int,
                     noise_mode: str, rng) -> np.ndarray:
    """``(n_trials, N)`` injected light curves on this block's exact epochs.

    In ``"data"`` mode every trial shares the **same** noise --- the block's one
    real realisation --- and differs only in the injected phase.  That is the
    right question ("would *this* block, as observed, have detected it?"), but it
    means the binomial error bar on ``eta`` covers phase only, not the sampling
    variance of the noise realisation itself.  ``"resample"`` bootstraps the
    block's own residuals to add that second source of variation when it is
    wanted; ``"gaussian"`` drops the real noise entirely and is the optimistic
    bound.
    """
    t = block.t
    N = len(t)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=int(n_trials))
    sig = amp_mag * np.sin(2.0 * np.pi * float(freq) * t[None, :] + phases[:, None])
    if noise_mode == "data":
        # By hypothesis this block holds no signal; use it verbatim as the noise.
        base = np.broadcast_to(block.y, (int(n_trials), N))
    elif noise_mode == "resample":
        resid = block.y - np.median(block.y)
        idx = rng.integers(0, N, size=(int(n_trials), N))
        base = np.median(block.y) + resid[idx]
    else:                                                # "gaussian"
        base = np.median(block.y) + rng.normal(0.0, block.e[None, :],
                                               size=(int(n_trials), N))
    return base + sig


def block_efficiency(
    block: Block,
    period: float,
    amplitude_mmag: float,
    *,
    freqs=None,
    power_threshold: float | None = None,
    n_trials: int = 200,
    n_null: int = 200,
    fap: float = 0.01,
    noise_mode: str = "data",
    min_period: float = 0.05,
    max_period: float = 100.0,
    oversample: float = 5.0,
    rng=None,
) -> Efficiency:
    """Measure ``eta`` for one block at one period and amplitude, by injection.

    The detection criterion is identical to the one the search applies to the
    real data: maximum GLS power over ``freqs`` at or above ``power_threshold``,
    where the threshold is this block's own permutation quantile at ``fap``.
    """
    rng = np.random.default_rng(rng)
    f = (frequency_grid(block.t, min_period=min_period, max_period=max_period,
                        oversample=oversample) if freqs is None
         else np.asarray(freqs, dtype=float))
    if power_threshold is None:
        power_threshold, _ = permutation_threshold(block.t, block.y, block.e, f,
                                                   fap=fap, n_null=n_null, rng=rng)

    amp_mag = float(amplitude_mmag) * 1e-3
    freq = 1.0 / float(period) if np.isfinite(period) and period > 0 else float("nan")
    if not np.isfinite(freq) or block.n < 8 or not np.isfinite(power_threshold):
        return Efficiency(block.index, float(period), float(amplitude_mmag), 0, 0, 0,
                          float("nan"), float("nan"), 1.0, False, noise_mode,
                          float(power_threshold if power_threshold is not None else np.nan),
                          block.n)

    Y = _injected_matrix(block, freq, amp_mag, int(n_trials), noise_mode, rng)
    P = gls_power_batch(block.t, Y, block.e, f)
    mx = P.max(axis=1)
    fbest = f[np.argmax(P, axis=1)]
    det = mx >= power_threshold
    match = np.zeros_like(det)
    for h in HARMONIC_FACTORS:
        match |= np.abs(fbest - h * freq) <= FREQ_TOL_FRAC * max(h * freq, 1e-9)
    det_fm = det & match

    n = int(n_trials)
    k_det = int(det.sum())
    eta = k_det / n
    n_miss = n - k_det
    return Efficiency(
        block_index=block.index, period=float(period),
        amplitude_mmag=float(amplitude_mmag), n_trials=n, n_detected=k_det,
        n_detected_freqmatch=int(det_fm.sum()), eta=float(eta),
        eta_freqmatch=float(int(det_fm.sum()) / n),
        miss_upper=float(clopper_pearson_upper(n_miss, n)),
        at_resolution_floor=bool(n_miss == 0), noise_mode=str(noise_mode),
        power_threshold=float(power_threshold), n_epochs=block.n,
    )


def efficiency_curve(block: Block, period: float, amplitudes_mmag, **kw) -> list[Efficiency]:
    """``eta`` versus injected amplitude for one block --- its sensitivity curve."""
    return [block_efficiency(block, period, float(a), **kw)
            for a in np.atleast_1d(amplitudes_mmag)]


def amplitude_at_efficiency(curve: list[Efficiency], target: float = 0.5) -> float:
    """Linear interpolation of the amplitude at which ``eta`` crosses ``target``.

    Returns NaN if the curve never crosses --- an honest "not bracketed" rather
    than an extrapolated number.
    """
    pts = sorted((c.amplitude_mmag, c.eta) for c in curve if np.isfinite(c.eta))
    for (a0, e0), (a1, e1) in zip(pts, pts[1:], strict=False):
        if (e0 - target) * (e1 - target) <= 0 and e1 != e0:
            return float(a0 + (target - e0) * (a1 - a0) / (e1 - e0))
    return float("nan")


def persistence_pvalue(effs: list[Efficiency]) -> dict:
    """P(all these blocks miss a signal that is still there), as a bound.

    The blocks are disjoint stretches of time with independent noise draws, so
    the miss probabilities multiply.  Every factor is the Clopper-Pearson
    **upper** bound on that block's miss rate, so the product is an upper bound
    on the p-value: the reported number can only overstate the chance that the
    clock is still running.

    ``pinned_at_floor`` is True when at least one block recovered every single
    injection, in which case the p-value must be quoted as an inequality
    (``p <= value``) and the number of trials escalated before the result is
    believed.  ``resolution_floor`` is the smallest p-value the trial count could
    ever resolve.
    """
    usable = [e for e in effs if np.isfinite(e.miss_upper) and e.n_trials > 0]
    if not usable:
        return {"p_persist_upper": float("nan"), "pinned_at_floor": False,
                "n_blocks": 0, "eta_min": float("nan"),
                "resolution_floor": float("nan"), "n_trials_min": 0}
    p = float(np.prod([e.miss_upper for e in usable]))
    floor = float(np.prod([clopper_pearson_upper(0, e.n_trials) for e in usable]))
    return {
        "p_persist_upper": p,
        "pinned_at_floor": bool(any(e.at_resolution_floor for e in usable)),
        "n_blocks": len(usable),
        "eta_min": float(min(e.eta for e in usable)),
        "eta_freqmatch_min": float(min(e.eta_freqmatch for e in usable)),
        "resolution_floor": floor,
        "n_trials_min": int(min(e.n_trials for e in usable)),
        "noise_mode": usable[0].noise_mode,
    }


def format_pvalue(p: float, pinned: bool, floor: float) -> str:
    """Render a p-value as an inequality when it sits at the resolution floor."""
    if not np.isfinite(p):
        return "undetermined"
    if pinned:
        return f"<= {max(p, floor):.3g} (injection-resolution limited)"
    return f"{p:.3g}"


__all__ = ["Efficiency", "amplitude_at_efficiency", "block_efficiency",
           "clopper_pearson_upper", "efficiency_curve", "format_pvalue",
           "persistence_pvalue"]
