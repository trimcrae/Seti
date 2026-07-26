"""Epoch blocking and per-block periodograms for KNELL.

Everything here is a pure function of arrays --- no network --- so the detector
is fully testable offline.

Three primitives:

``make_blocks``       split a light curve into epoch blocks (default: observing
                      seasons), keeping each block's own times *and* its own
                      per-epoch errors, because both are what the efficiency
                      module has to reproduce.
``gls_power_batch``   the generalised (floating-mean, error-weighted)
                      Lomb-Scargle periodogram of Zechmeister & Kurster 2009,
                      **batched over many light curves sharing one time
                      sampling**.  This is not a micro-optimisation: the
                      efficiency module needs hundreds of periodograms per block
                      per star, and a batched implementation is what makes an
                      injection-measured efficiency affordable at catalogue
                      scale rather than a survivor-only luxury.
``pdm_theta``         phase-dispersion minimisation, the independent
                      cross-check.  PDM is not a sinusoid fit, so it responds to
                      eclipse-shaped and sawtooth signals that a Fourier method
                      under-weights, and its systematics (bin occupancy) are
                      unrelated to the periodogram's (spectral window).  A
                      cessation asserted by one and denied by the other is an
                      artefact of the method, not a property of the star.

Detection threshold
-------------------
A block is called "periodic" when its maximum periodogram power exceeds a
threshold obtained from a **permutation null on that same block** --- the
magnitudes are shuffled against the times, which destroys any coherent signal
while preserving (a) the exact observing window, hence the exact spectral window
and alias structure, and (b) the exact magnitude distribution, hence any
non-Gaussian tail.  A single analytic false-alarm formula applied to every block
would do neither, and the difference between blocks *is* the systematic this
channel exists to control.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """One epoch block of a light curve."""

    index: int
    t: np.ndarray
    y: np.ndarray
    e: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.t))

    @property
    def t_mid(self) -> float:
        return float(np.median(self.t)) if self.n else float("nan")

    @property
    def t_span(self) -> float:
        return float(self.t.max() - self.t.min()) if self.n > 1 else 0.0

    @property
    def mean_mag(self) -> float:
        return float(np.median(self.y)) if self.n else float("nan")

    @property
    def median_err(self) -> float:
        e = self.e[np.isfinite(self.e)]
        return float(np.median(e)) if len(e) else float("nan")


def make_blocks(
    t,
    y,
    e=None,
    *,
    season_days: float = 365.25,
    min_epochs_block: int = 15,
    min_blocks: int = 4,
    mode: str = "fixed",
    gap_days: float = 90.0,
    default_err: float = 0.02,
) -> list[Block]:
    """Split a light curve into epoch blocks.

    ``mode="fixed"``  blocks are ``floor((t - t0) / season_days)`` --- deterministic,
    reproducible, and directly comparable between the two bands of the same star
    (which matters, since the two-band coincidence requirement compares block
    indices).  ``mode="gap"`` splits wherever the sampling gap exceeds
    ``gap_days``, which tracks a survey's real observing seasons when they are
    irregular.

    Blocks with fewer than ``min_epochs_block`` epochs are **dropped, not
    merged** --- merging a thin block into its neighbour would smear the
    transition this channel is trying to localise.  Their indices are preserved
    so a gap in the block sequence is visible downstream.

    Returns ``[]`` if fewer than ``min_blocks`` usable blocks survive, which is
    the honest "this star cannot be tested" outcome rather than a weak result.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    e = (np.full_like(t, float(default_err)) if e is None
         else np.asarray(e, dtype=float))
    ok = np.isfinite(t) & np.isfinite(y)
    t, y, e = t[ok], y[ok], e[ok]
    e = np.where(np.isfinite(e) & (e > 0), e, float(default_err))
    if len(t) == 0:
        return []
    order = np.argsort(t)
    t, y, e = t[order], y[order], e[order]

    if mode == "gap":
        gaps = np.diff(t)
        idx = np.concatenate([[0], np.cumsum(gaps > float(gap_days))]).astype(int)
    else:
        idx = np.floor((t - t[0]) / float(season_days)).astype(int)

    blocks: list[Block] = []
    for k in np.unique(idx):
        m = idx == k
        if int(m.sum()) < int(min_epochs_block):
            continue
        blocks.append(Block(index=int(k), t=t[m], y=y[m], e=e[m]))
    if len(blocks) < int(min_blocks):
        return []
    return blocks


# ---------------------------------------------------------------------------
# Periodogram
# ---------------------------------------------------------------------------


def frequency_grid(t, *, min_period: float = 0.05, max_period: float = 100.0,
                   oversample: float = 5.0, max_freqs: int = 40000) -> np.ndarray:
    """Uniform frequency grid, Nyquist-agnostic and baseline-matched.

    The spacing is ``1 / (oversample * T)`` with ``T`` the block baseline, the
    standard rule that guarantees no peak is stepped over.  ``max_period`` is
    capped at the baseline itself: a "period" longer than the block is a trend,
    not a clock, and admitting it lets a slow fade masquerade as periodicity.
    """
    t = np.asarray(t, dtype=float)
    T = float(t.max() - t.min()) if len(t) > 1 else 1.0
    T = max(T, 1e-6)
    max_period = min(float(max_period), T)
    f_lo = 1.0 / max(max_period, 1e-6)
    f_hi = 1.0 / max(float(min_period), 1e-6)
    if not np.isfinite(f_lo) or f_hi <= f_lo:
        return np.array([f_hi])
    df = 1.0 / (float(oversample) * T)
    n = int(np.ceil((f_hi - f_lo) / df)) + 1
    if n > int(max_freqs):
        n = int(max_freqs)
        df = (f_hi - f_lo) / (n - 1)
    return f_lo + df * np.arange(n)


def gls_power_batch(t, Y, e, freqs) -> np.ndarray:
    """Generalised Lomb-Scargle power for many light curves on one time grid.

    Zechmeister & Kurster (2009) formulation: floating mean, per-epoch weights
    ``w_i propto 1/e_i^2``, power normalised to ``[0, 1]`` (the fraction of the
    weighted variance explained by the sinusoid at that frequency).

    Parameters
    ----------
    t : (N,) times
    Y : (N,) or (M, N) magnitudes --- M independent light curves on the same ``t``
    e : (N,) per-epoch errors, shared by all M (they are the *same block*)
    freqs : (F,) frequencies

    Returns
    -------
    (F,) if ``Y`` was 1-D, else (M, F).
    """
    t = np.asarray(t, dtype=float)
    Y0 = np.asarray(Y, dtype=float)
    squeeze = Y0.ndim == 1
    Y = np.atleast_2d(Y0)
    e = np.asarray(e, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    N = len(t)
    if N < 4 or Y.shape[1] != N:
        z = np.zeros((Y.shape[0], len(freqs)))
        return z[0] if squeeze else z

    w = 1.0 / np.clip(e, 1e-6, None) ** 2
    w = w / w.sum()

    Yb = (Y * w).sum(axis=1)                        # (M,)
    YY = (Y * Y * w).sum(axis=1) - Yb ** 2          # (M,)
    YY = np.clip(YY, 1e-30, None)

    ph = 2.0 * np.pi * np.outer(freqs, t)           # (F, N)
    C_ = np.cos(ph)
    S_ = np.sin(ph)
    wc = C_ * w                                     # (F, N)
    ws = S_ * w

    C = wc.sum(axis=1)                              # (F,)
    S = ws.sum(axis=1)
    CC = (wc * C_).sum(axis=1) - C * C
    SS = (ws * S_).sum(axis=1) - S * S
    CS = (wc * S_).sum(axis=1) - C * S

    YC = Y @ wc.T - np.outer(Yb, C)                 # (M, F)
    YS = Y @ ws.T - np.outer(Yb, S)

    D = CC * SS - CS * CS                           # (F,)
    D = np.where(np.abs(D) < 1e-30, np.nan, D)
    num = SS * YC ** 2 + CC * YS ** 2 - 2.0 * CS * YC * YS
    p = num / (YY[:, None] * D[None, :])
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)
    return p[0] if squeeze else p


def gls_power(t, y, e, freqs) -> np.ndarray:
    """Single-light-curve wrapper around :func:`gls_power_batch`."""
    return gls_power_batch(t, np.asarray(y, dtype=float).ravel(), e, freqs)


def sine_amplitude(t, y, e, freq) -> tuple[float, float, float]:
    """Weighted least-squares sinusoid amplitude at a **fixed** frequency.

    Returns ``(amplitude, sigma_amplitude, mean_level)``.  Fitting at a frequency
    already fixed by the early blocks is far more sensitive than a blind
    periodogram search --- there is no trials factor --- so it is the statistic
    used to ask "how much signal is *left*", while the blind periodogram is what
    defines "detected".  Keeping the two separate is deliberate: the blind test
    supplies the yes/no that the efficiency module can reproduce by injection,
    the fixed-frequency fit supplies the amplitude with an error bar.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    e = np.asarray(e, dtype=float)
    if len(t) < 4 or not np.isfinite(freq) or freq <= 0:
        return float("nan"), float("nan"), float("nan")
    ph = 2.0 * np.pi * float(freq) * t
    X = np.column_stack([np.ones_like(t), np.cos(ph), np.sin(ph)])
    sig = np.clip(e, 1e-6, None)
    Xw = X / sig[:, None]
    yw = y / sig
    try:
        cov = np.linalg.inv(Xw.T @ Xw)
        beta = cov @ (Xw.T @ yw)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), float("nan")
    c, s = float(beta[1]), float(beta[2])
    amp = float(np.hypot(c, s))
    # Error propagation for A = sqrt(c^2 + s^2).  At amp -> 0 the Jacobian is
    # singular, so fall back to the mean marginal error, which is the right
    # scale for "consistent with zero" and does not blow up.
    vc, vs, vcs = float(cov[1, 1]), float(cov[2, 2]), float(cov[1, 2])
    if amp > 1e-9:
        var = (c * c * vc + s * s * vs + 2.0 * c * s * vcs) / (amp * amp)
    else:
        var = 0.5 * (vc + vs)
    return amp, float(np.sqrt(max(var, 0.0))), float(beta[0])


def pdm_theta(t, y, period: float, n_bins: int = 10, n_covers: int = 2) -> float:
    """Phase-dispersion-minimisation statistic ``Theta = s^2 / sigma^2``.

    ``Theta << 1`` means folding at ``period`` reduces the scatter --- a real
    signal of *any* shape.  ``Theta ~ 1`` means the fold explains nothing.
    Overlapping bin covers (Stellingwerf's multi-cover scheme) reduce sensitivity
    to the arbitrary bin phase origin.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 8 or not np.isfinite(period) or period <= 0:
        return float("nan")
    var = float(np.var(y, ddof=1))
    if var <= 0:
        return float("nan")
    phase = np.mod(t / float(period), 1.0)
    num = 0.0
    den = 0
    for c in range(int(n_covers)):
        shift = c / (n_covers * n_bins)
        b = np.floor(np.mod(phase + shift, 1.0) * n_bins).astype(int)
        for k in range(int(n_bins)):
            m = b == k
            nk = int(m.sum())
            if nk < 2:
                continue
            num += (nk - 1) * float(np.var(y[m], ddof=1))
            den += nk - 1
    if den <= 0:
        return float("nan")
    return float((num / den) / var)


def pdm_theta_pvalue(t, y, period: float, *, n_bins: int = 10, n_covers: int = 2,
                     n_null: int = 100, rng=None) -> tuple[float, float]:
    """``(Theta, p)`` where ``p`` is a permutation p-value for ``Theta`` being *low*.

    An absolute cut on ``Theta`` cannot be right: its null distribution depends on
    the epoch count, the bin occupancy and the phase coverage --- all of which
    differ between blocks, which is the very thing this channel must not let leak
    into the statistic.  Calibrating ``Theta`` against permutations of the same
    block's own magnitudes removes that dependence exactly the way the
    periodogram threshold does.

    Small ``p`` = folding at ``period`` reduces the scatter more than chance:
    a signal.  ``p`` of order unity = the fold explains nothing.
    """
    th = pdm_theta(t, y, period, n_bins=n_bins, n_covers=n_covers)
    if not np.isfinite(th):
        return float("nan"), float("nan")
    rng = np.random.default_rng(rng)
    y = np.asarray(y, dtype=float)
    k = 0
    n = int(max(n_null, 20))
    for _ in range(n):
        thn = pdm_theta(t, rng.permutation(y), period, n_bins=n_bins, n_covers=n_covers)
        if np.isfinite(thn) and thn <= th:
            k += 1
    return float(th), float((k + 1) / (n + 1))


def excess_variance(y, e) -> float:
    """Variance in excess of the reported photometric errors, in mag^2.

    Allowed to go negative: clipping at zero would rectify noise into a spurious
    positive and make every faint block look mildly variable.  This is the
    frequency-agnostic measure of "how much is this star doing", and it is what
    separates a clock that stopped (excess variance falls to zero) from a
    pulsator that switched modes (excess variance is **conserved**, it merely
    moved in frequency).
    """
    y = np.asarray(y, dtype=float)
    e = np.asarray(e, dtype=float)
    if len(y) < 4:
        return float("nan")
    resid = y - np.median(y)
    s2 = float(1.4826 * np.median(np.abs(resid))) ** 2
    return float(s2 - float(np.mean(np.clip(e, 1e-6, None) ** 2)))


def pdm_scan(t, y, periods, n_bins: int = 10) -> tuple[float, float]:
    """Best (minimum-``Theta``) period over a candidate list. Returns (period, theta)."""
    best_p, best_th = float("nan"), float("inf")
    for p in np.atleast_1d(periods):
        th = pdm_theta(t, y, float(p), n_bins=n_bins)
        if np.isfinite(th) and th < best_th:
            best_th, best_p = th, float(p)
    return best_p, (best_th if np.isfinite(best_th) else float("nan"))


# ---------------------------------------------------------------------------
# Per-block detection
# ---------------------------------------------------------------------------


@dataclass
class BlockPeriodogram:
    """The periodogram summary of one epoch block."""

    index: int
    n: int
    t_mid: float
    t_span: float
    best_freq: float
    best_period: float
    best_power: float
    power_threshold: float
    detected: bool
    mean_mag: float
    median_err: float
    scatter_mmag: float
    fap_empirical: float = float("nan")
    # Filled by seti.knell.cease once a reference frequency exists.
    amp_ref_mmag: float = float("nan")
    amp_ref_sigma_mmag: float = float("nan")
    pdm_theta_ref: float = float("nan")
    pdm_p_ref: float = float("nan")
    excess_var: float = float("nan")
    eta: float = float("nan")
    eta_freq: float = float("nan")
    miss_upper: float = float("nan")
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "extras"}
        d.update(self.extras)
        return d


def permutation_threshold(t, y, e, freqs, *, fap: float = 0.01, n_null: int = 200,
                          rng=None) -> tuple[float, np.ndarray]:
    """Max-power threshold from a permutation null **on this block's own data**.

    Shuffling ``y`` against ``t`` destroys coherence while preserving the exact
    observing window (hence the alias structure) and the exact magnitude
    distribution (hence any non-Gaussian tail).  Returns the ``1 - fap`` quantile
    of the null max-power distribution, and the null draws themselves so a caller
    can quote an empirical FAP.

    **One documented approximation.**  The magnitudes are permuted but the error
    vector is held in its original epoch order, because a per-draw weight vector
    cannot be expressed in the batched (matrix-product) form that makes this
    affordable per block per star.  In real photometry ``magerr`` is a
    deterministic function of ``mag``, so the null's weighting is slightly
    mismatched to its magnitudes.  The direction of the error is known: it
    under-weights the epochs that happen to carry the extreme magnitudes, which
    *widens* the null max-power distribution and therefore *raises* the
    threshold.  A raised threshold makes detection harder in both the data and
    the injections --- and since the efficiency is measured with this same
    threshold, the approximation cancels between the two rather than biasing the
    cessation statistic.
    """
    rng = np.random.default_rng(rng)
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    e = np.asarray(e, dtype=float)
    n_null = int(max(n_null, 20))
    N = len(t)
    if N < 8:
        return float("inf"), np.array([])
    # One permutation index matrix -> one batched periodogram call.
    perms = np.argsort(rng.random((n_null, N)), axis=1)
    Ynull = y[perms]
    # Weight with the *mean* weight vector: a per-draw weight vector cannot be
    # expressed in the batched form, and using the mean is conservative (it
    # slightly under-weights the best epochs, raising the null spread).
    P = gls_power_batch(t, Ynull, e, freqs)
    mx = P.max(axis=1)
    return float(np.quantile(mx, 1.0 - float(fap))), mx


def block_periodogram(block: Block, freqs=None, *, fap: float = 0.01,
                      n_null: int = 200, rng=None,
                      min_period: float = 0.05, max_period: float = 100.0,
                      oversample: float = 5.0) -> BlockPeriodogram:
    """Periodogram + permutation-calibrated detection verdict for one block."""
    rng = np.random.default_rng(rng)
    f = (frequency_grid(block.t, min_period=min_period, max_period=max_period,
                        oversample=oversample) if freqs is None
         else np.asarray(freqs, dtype=float))
    p = gls_power(block.t, block.y, block.e, f)
    i = int(np.argmax(p)) if len(p) else 0
    best_power = float(p[i]) if len(p) else 0.0
    best_freq = float(f[i]) if len(f) else float("nan")
    thr, nulls = permutation_threshold(block.t, block.y, block.e, f,
                                       fap=fap, n_null=n_null, rng=rng)
    fap_emp = (float((nulls >= best_power).sum() + 1) / float(len(nulls) + 1)
               if len(nulls) else float("nan"))
    resid = block.y - np.median(block.y)
    scat = float(1.4826 * np.median(np.abs(resid))) * 1e3 if block.n else float("nan")
    return BlockPeriodogram(
        index=block.index, n=block.n, t_mid=block.t_mid, t_span=block.t_span,
        best_freq=best_freq,
        best_period=(1.0 / best_freq if np.isfinite(best_freq) and best_freq > 0
                     else float("nan")),
        best_power=best_power, power_threshold=float(thr),
        detected=bool(np.isfinite(thr) and best_power >= thr),
        mean_mag=block.mean_mag, median_err=block.median_err,
        scatter_mmag=scat, fap_empirical=fap_emp,
    )


__all__ = ["Block", "BlockPeriodogram", "block_periodogram", "excess_variance",
           "frequency_grid", "gls_power", "gls_power_batch", "make_blocks",
           "pdm_scan", "pdm_theta", "pdm_theta_pvalue", "permutation_threshold",
           "sine_amplitude"]
