"""The cessation statistic: a clock that was running and then was not.

Given one band's light curve, :func:`analyze_band` answers a single question with
an auditable chain of intermediate numbers:

    Was there a coherent period in the early epoch blocks, is it absent in the
    late ones, **could the late ones have detected it**, and did the star's mean
    brightness stay put while that happened?

The order of those clauses is the design.  "Absent" is cheap and produced by a
hundred mundane causes; "could have detected it" is expensive, is measured by
injection into the late blocks themselves (:mod:`seti.knell.efficiency`), and is
the only clause that separates a stopped clock from a degraded observation.

Structure of the test
---------------------
1. **Block, then periodogram each block independently.**  Never a global
   periodogram --- a global one averages the cessation away, which is precisely
   how the effect has stayed uncatalogued.
2. **Reference frequency** ``f0`` from the detected blocks, refined on their
   union.  Every later statistic is evaluated at this *fixed* frequency, so
   there is no trials factor anywhere after step 2.
3. **Transition pattern.**  Detection must be a run of ``True`` followed by a run
   of ``False``, with at least ``min_pre_blocks`` and ``min_post_blocks`` of
   each.  This one requirement disposes of two named confounders for free:

   * a **mode-switching pulsator** moves power to a different frequency but
     remains detectable, so its late blocks are still ``detected = True`` and no
     transition exists;
   * a **Blazhko-like modulation** returns, so a later block detects again and
     the pattern is broken.

   The pattern is checked on the *blind* per-block detector for exactly this
   reason: it asks "is this star still periodic at all?", not "is it still
   periodic at ``f0``?".
4. **Efficiency gate and persistence p-value** over the post blocks.
5. **Constant mean flux.**  A star that faded until its signal sank into the
   noise is a fade, a different and mundane phenomenon.  Note this is *also*
   caught by step 4 --- injecting into fainter, noisier data lowers ``eta`` ---
   so the two guards are independent, which is the point of keeping both.
6. **PDM cross-check** at ``f0``, a non-Fourier estimator with unrelated
   systematics.
7. **Mechanism flags** for the known astrophysical ways a clock stops:
   a gradual pre-transition amplitude decline (the SS Lacertae third-body
   orbital-precession signature), pre-transition amplitude modulation
   (Blazhko-like), and an unstable inter-block frequency.

Nothing here touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .blocks import (
    Block,
    BlockPeriodogram,
    block_periodogram,
    excess_variance,
    frequency_grid,
    gls_power,
    make_blocks,
    pdm_theta_pvalue,
    sine_amplitude,
)
from .efficiency import block_efficiency, persistence_pvalue

# ---- defaults (config/knell.yaml overrides them; see run.load_knell_config) --
FAP = 0.01                  # per-block permutation false-alarm probability
MIN_PRE_BLOCKS = 2
MIN_POST_BLOCKS = 2
MIN_POST_SPAN_DAYS = 500.0  # longer than essentially every catalogued Blazhko cycle
ETA_MIN = 0.90              # a post block below this could not have seen it
P_PERSIST_MAX = 0.01
MEAN_SHIFT_MAX_MAG = 0.05
POST_AMP_SIGMA_MAX = 3.0    # residual amplitude must be consistent with zero
DROP_SIGMA_MIN = 5.0        # and the drop itself must be significant
PDM_P_PRE_MAX = 0.01        # folding at f0 must beat chance in the pre blocks
PDM_P_POST_MIN = 0.05       # ...and must NOT beat chance in the post blocks
VAR_DROP_FRAC = 0.35        # post excess variance, as a fraction of pre
FREQ_TOL_FRAC = 0.02
PRE_DECLINE_SIGMA_MAX = 3.0  # steeper than this = SS Lac-like precession, not a stop
AMP_MODULATION_MAX = 0.50    # sigma(A_pre) / mean(A_pre)


@dataclass
class CessationResult:
    """One band's verdict, with every number a reviewer would want to audit."""

    band: str = ""
    status: str = "insufficient_data"
    n_epochs: int = 0
    n_blocks: int = 0
    n_pre: int = 0
    n_post: int = 0
    n_detected: int = 0
    split_block: int = -1
    ref_period: float = float("nan")
    ref_freq: float = float("nan")
    freq_consistent: bool = False
    pattern_strict: bool = False
    amp_pre_mmag: float = float("nan")
    amp_pre_sigma_mmag: float = float("nan")
    amp_post_mmag: float = float("nan")
    amp_post_sigma_mmag: float = float("nan")
    amp_post_over_sigma: float = float("nan")
    drop_sigma: float = float("nan")
    amp_ratio: float = float("nan")
    mean_shift_mag: float = float("nan")
    mean_shift_sigma: float = float("nan")
    scatter_ratio: float = float("nan")
    eta_min_post: float = float("nan")
    eta_freqmatch_min_post: float = float("nan")
    p_persist_upper: float = float("nan")
    p_pinned_at_floor: bool = False
    p_resolution_floor: float = float("nan")
    eff_noise_mode: str = ""
    eff_n_trials: int = 0
    pdm_theta_pre: float = float("nan")
    pdm_theta_post: float = float("nan")
    pdm_p_pre: float = float("nan")
    pdm_p_post: float = float("nan")
    excess_var_pre: float = float("nan")
    excess_var_post: float = float("nan")
    excess_var_ratio: float = float("nan")
    pre_decline_sigma: float = float("nan")
    amp_modulation: float = float("nan")
    post_span_days: float = float("nan")
    t_transition_mjd: float = float("nan")
    is_cessation: bool = False
    flags: list[str] = field(default_factory=list)
    blocks: list[BlockPeriodogram] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "blocks"}
        d["flags"] = ";".join(self.flags)
        d["block_n"] = ",".join(str(b.n) for b in self.blocks)
        d["block_detected"] = ",".join("1" if b.detected else "0" for b in self.blocks)
        d["block_power"] = ",".join(f"{b.best_power:.4g}" for b in self.blocks)
        d["block_thresh"] = ",".join(f"{b.power_threshold:.4g}" for b in self.blocks)
        d["block_amp_mmag"] = ",".join(f"{b.amp_ref_mmag:.4g}" for b in self.blocks)
        d["block_eta"] = ",".join(
            ("" if not np.isfinite(b.eta) else f"{b.eta:.3f}") for b in self.blocks)
        return d


def _refine_frequency(blocks: list[Block], f0: float, *, half_width_frac: float = 0.02,
                      n_grid: int = 4001) -> float:
    """Refine ``f0`` on the union of the detected blocks.

    Starting from a peak that a single block already resolved bounds the alias
    risk of using the full (gap-ridden) baseline, while the long lever arm gives
    a frequency precise enough that the fixed-frequency amplitude fit in the late
    blocks does not lose coherence across a decade.
    """
    if not blocks or not np.isfinite(f0) or f0 <= 0:
        return float(f0)
    t = np.concatenate([b.t for b in blocks])
    y = np.concatenate([b.y for b in blocks])
    e = np.concatenate([b.e for b in blocks])
    lo, hi = f0 * (1.0 - half_width_frac), f0 * (1.0 + half_width_frac)
    grid = np.linspace(max(lo, 1e-9), hi, int(n_grid))
    p = gls_power(t, y, e, grid)
    return float(grid[int(np.argmax(p))])


def _weighted_mean(vals, sigs) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    sigs = np.asarray(sigs, dtype=float)
    m = np.isfinite(vals) & np.isfinite(sigs) & (sigs > 0)
    if not m.any():
        return float("nan"), float("nan")
    w = 1.0 / sigs[m] ** 2
    mu = float((vals[m] * w).sum() / w.sum())
    return mu, float(np.sqrt(1.0 / w.sum()))


def _linear_trend_sigma(x, y, s) -> float:
    """Significance of the slope of ``y`` against ``x`` (negative = declining)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(s) & (s > 0)
    if m.sum() < 3:
        return float("nan")
    x, y, s = x[m], y[m], s[m]
    w = 1.0 / s ** 2
    X = np.column_stack([np.ones_like(x), x - x.mean()])
    Xw = X * np.sqrt(w)[:, None]
    try:
        cov = np.linalg.inv(Xw.T @ Xw)
    except np.linalg.LinAlgError:
        return float("nan")
    beta = cov @ (Xw.T @ (y * np.sqrt(w)))
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    return float(beta[1] / se) if se > 0 else float("nan")


def analyze_band(
    t,
    y,
    e=None,
    *,
    band: str = "",
    season_days: float = 365.25,
    min_epochs_block: int = 15,
    min_blocks: int = 4,
    block_mode: str = "fixed",
    min_period: float = 0.05,
    max_period: float = 100.0,
    oversample: float = 5.0,
    fap: float = FAP,
    n_null: int = 200,
    n_trials: int = 200,
    eta_min: float = ETA_MIN,
    p_persist_max: float = P_PERSIST_MAX,
    min_pre_blocks: int = MIN_PRE_BLOCKS,
    min_post_blocks: int = MIN_POST_BLOCKS,
    min_post_span_days: float = MIN_POST_SPAN_DAYS,
    mean_shift_max_mag: float = MEAN_SHIFT_MAX_MAG,
    post_amp_sigma_max: float = POST_AMP_SIGMA_MAX,
    drop_sigma_min: float = DROP_SIGMA_MIN,
    pdm_p_pre_max: float = PDM_P_PRE_MAX,
    pdm_p_post_min: float = PDM_P_POST_MIN,
    var_drop_frac: float = VAR_DROP_FRAC,
    pdm_null: int = 100,
    noise_mode: str = "data",
    measure_efficiency: bool = True,
    rng=None,
) -> CessationResult:
    """Full single-band cessation test.  Pure function; no network."""
    rng = np.random.default_rng(rng)
    res = CessationResult(band=str(band))
    t = np.asarray(t, dtype=float)
    res.n_epochs = int(np.isfinite(t).sum())

    blocks = make_blocks(t, y, e, season_days=season_days,
                         min_epochs_block=min_epochs_block, min_blocks=min_blocks,
                         mode=block_mode)
    if not blocks:
        res.status = "insufficient_data"
        res.flags.append("too_few_usable_blocks")
        return res
    res.n_blocks = len(blocks)

    # -- 1. per-block blind periodograms, each on its own permutation threshold
    bps = [block_periodogram(b, fap=fap, n_null=n_null, rng=rng,
                             min_period=min_period, max_period=max_period,
                             oversample=oversample) for b in blocks]
    res.blocks = bps
    det = np.array([bp.detected for bp in bps], dtype=bool)
    res.n_detected = int(det.sum())

    if not det.any():
        res.status = "never_periodic"
        return res
    if det.all():
        res.status = "still_periodic"
        return res

    # -- 2. reference frequency from the detected blocks
    i_best = int(np.argmax([bp.best_power if bp.detected else -1 for bp in bps]))
    f0 = _refine_frequency([b for b, d in zip(blocks, det, strict=False) if d], bps[i_best].best_freq)
    res.ref_freq = float(f0)
    res.ref_period = float(1.0 / f0) if np.isfinite(f0) and f0 > 0 else float("nan")

    dfreq = [bp.best_freq for bp in bps if bp.detected]
    consistent = True
    for fb in dfreq:
        if not any(abs(fb - h * f0) <= FREQ_TOL_FRAC * max(h * f0, 1e-9)
                   for h in (0.5, 1.0, 2.0, 3.0)):
            consistent = False
    res.freq_consistent = bool(consistent)
    if not consistent:
        res.flags.append("frequency_unstable")

    # -- 3. transition pattern: a run of detections then a run of non-detections
    last_det = int(np.max(np.nonzero(det)[0]))
    s = last_det + 1
    res.split_block = int(bps[s].index) if s < len(bps) else -1
    res.n_pre, res.n_post = int(s), int(len(bps) - s)
    res.pattern_strict = bool(det[:s].all() and not det[s:].any())
    if not res.pattern_strict:
        res.flags.append("intermittent_detection")
    if res.n_post < 1:
        # The last block still detects.  Whatever happened in the middle, the
        # clock is running at the end of the baseline, so there is no cessation
        # to test and none of the post-transition statistics are defined.
        res.status = "no_clean_transition"
        res.flags.append("detected_in_final_block")
        return res
    res.t_transition_mjd = float(0.5 * (bps[s - 1].t_mid + bps[s].t_mid)) if s < len(bps) \
        else float("nan")
    post_blocks = blocks[s:]
    res.post_span_days = float(post_blocks[-1].t.max() - post_blocks[0].t.min()) \
        if post_blocks else 0.0

    # -- 4. fixed-frequency amplitudes and mean levels in every block
    for b, bp in zip(blocks, bps, strict=False):
        a, sa, _lvl = sine_amplitude(b.t, b.y, b.e, f0)
        bp.amp_ref_mmag = a * 1e3
        bp.amp_ref_sigma_mmag = sa * 1e3
        bp.pdm_theta_ref, bp.pdm_p_ref = pdm_theta_pvalue(
            b.t, b.y, res.ref_period, n_null=int(pdm_null), rng=rng)
        bp.excess_var = excess_variance(b.y, b.e)
    a_pre, sa_pre = _weighted_mean([bp.amp_ref_mmag for bp in bps[:s]],
                                   [bp.amp_ref_sigma_mmag for bp in bps[:s]])
    a_post, sa_post = _weighted_mean([bp.amp_ref_mmag for bp in bps[s:]],
                                     [bp.amp_ref_sigma_mmag for bp in bps[s:]])
    res.amp_pre_mmag, res.amp_pre_sigma_mmag = a_pre, sa_pre
    res.amp_post_mmag, res.amp_post_sigma_mmag = a_post, sa_post
    res.amp_post_over_sigma = (a_post / sa_post) if (np.isfinite(sa_post) and sa_post > 0) \
        else float("nan")
    denom = float(np.sqrt(np.nansum([sa_pre ** 2, sa_post ** 2])))
    res.drop_sigma = ((a_pre - a_post) / denom) if denom > 0 else float("nan")
    res.amp_ratio = (a_post / a_pre) if (np.isfinite(a_pre) and a_pre > 0) else float("nan")

    # -- 5. mean flux and noise level across the transition
    mpre = float(np.median([bp.mean_mag for bp in bps[:s]]))
    mpost = float(np.median([bp.mean_mag for bp in bps[s:]]))
    res.mean_shift_mag = float(mpost - mpre)
    spread = float(np.std([bp.mean_mag for bp in bps[:s]], ddof=1)) if s > 1 else float("nan")
    res.mean_shift_sigma = (abs(res.mean_shift_mag) / spread
                            if np.isfinite(spread) and spread > 0 else float("nan"))
    epre = float(np.median([bp.median_err for bp in bps[:s]]))
    epost = float(np.median([bp.median_err for bp in bps[s:]]))
    res.scatter_ratio = (epost / epre) if (np.isfinite(epre) and epre > 0) else float("nan")

    # -- 6. PDM cross-check at the reference period, permutation-calibrated
    res.pdm_theta_pre = float(np.nanmedian([bp.pdm_theta_ref for bp in bps[:s]]))
    res.pdm_theta_post = float(np.nanmin([bp.pdm_theta_ref for bp in bps[s:]]))
    res.pdm_p_pre = float(np.nanmedian([bp.pdm_p_ref for bp in bps[:s]]))
    # The post-block PDM veto asks "does ANY post block show the fold?", so the
    # minimum p-value over post blocks is a multiple comparison and must be
    # corrected --- otherwise the veto's own false-rejection rate rises with the
    # number of post blocks, i.e. with the length of the baseline, which would
    # reintroduce a cadence dependence through the back door.  Sidak.
    _pmin = float(np.nanmin([bp.pdm_p_ref for bp in bps[s:]]))
    res.pdm_p_post = float(1.0 - (1.0 - _pmin) ** max(res.n_post, 1))

    # -- 6b. frequency-agnostic variability budget.  A clock that stopped loses
    # its excess variance; a pulsator that switched modes KEEPS it and merely
    # moves it in frequency, so testing integrated variability -- not the power
    # at one frequency -- is what closes the mode-switching confounder even when
    # the new mode falls somewhere the periodogram handles badly.
    res.excess_var_pre = float(np.nanmedian([bp.excess_var for bp in bps[:s]]))
    res.excess_var_post = float(np.nanmedian([bp.excess_var for bp in bps[s:]]))
    if np.isfinite(res.excess_var_pre) and res.excess_var_pre > 0:
        res.excess_var_ratio = float(max(res.excess_var_post, 0.0) / res.excess_var_pre)

    # -- 7. mechanism flags on the pre-transition amplitude history
    res.pre_decline_sigma = _linear_trend_sigma(
        [bp.t_mid for bp in bps[:s]], [bp.amp_ref_mmag for bp in bps[:s]],
        [bp.amp_ref_sigma_mmag for bp in bps[:s]])
    amps_pre = np.array([bp.amp_ref_mmag for bp in bps[:s]], dtype=float)
    if np.isfinite(amps_pre).sum() >= 2 and np.nanmean(amps_pre) > 0:
        res.amp_modulation = float(np.nanstd(amps_pre, ddof=1) / np.nanmean(amps_pre))

    # -- 8. THE GATE: injection-measured efficiency in the post blocks
    if measure_efficiency and res.n_post >= 1:
        # Inject at the *conservative* end of the pre-transition amplitude, so a
        # marginal early detection cannot be converted into a confident late
        # non-detection by assuming the signal was stronger than measured.
        a_inj = max(a_pre - sa_pre, 0.0) if np.isfinite(sa_pre) else a_pre
        effs = []
        for b in post_blocks:
            f = frequency_grid(b.t, min_period=min_period, max_period=max_period,
                               oversample=oversample)
            effs.append(block_efficiency(
                b, res.ref_period, a_inj, freqs=f, n_trials=n_trials,
                n_null=n_null, fap=fap, noise_mode=noise_mode,
                min_period=min_period, max_period=max_period,
                oversample=oversample, rng=rng))
        pp = persistence_pvalue(effs)
        res.eta_min_post = pp.get("eta_min", float("nan"))
        res.eta_freqmatch_min_post = pp.get("eta_freqmatch_min", float("nan"))
        res.p_persist_upper = pp.get("p_persist_upper", float("nan"))
        res.p_pinned_at_floor = bool(pp.get("pinned_at_floor", False))
        res.p_resolution_floor = pp.get("resolution_floor", float("nan"))
        res.eff_noise_mode = str(pp.get("noise_mode", noise_mode))
        res.eff_n_trials = int(pp.get("n_trials_min", 0))
        for b, ef in zip(post_blocks, effs, strict=False):
            for bp in bps:
                if bp.index == b.index:
                    bp.eta, bp.eta_freq = ef.eta, ef.eta_freqmatch
                    bp.miss_upper = ef.miss_upper

    # -- 9. adjudicate
    checks = {
        "pattern": res.pattern_strict and res.n_pre >= min_pre_blocks
        and res.n_post >= min_post_blocks,
        "post_span": res.post_span_days >= float(min_post_span_days),
        "freq_consistent": res.freq_consistent,
        "efficiency": (np.isfinite(res.eta_min_post) and res.eta_min_post >= float(eta_min))
        if measure_efficiency else False,
        "p_persist": (np.isfinite(res.p_persist_upper)
                      and res.p_persist_upper <= float(p_persist_max))
        if measure_efficiency else False,
        "mean_flux": (np.isfinite(res.mean_shift_mag)
                      and abs(res.mean_shift_mag) <= float(mean_shift_max_mag)),
        "post_amp_zero": (np.isfinite(res.amp_post_over_sigma)
                          and res.amp_post_over_sigma <= float(post_amp_sigma_max)),
        "drop_significant": (np.isfinite(res.drop_sigma)
                             and res.drop_sigma >= float(drop_sigma_min)),
        "pdm_pre": (np.isfinite(res.pdm_p_pre) and res.pdm_p_pre <= float(pdm_p_pre_max)),
        "pdm_post": (np.isfinite(res.pdm_p_post)
                     and res.pdm_p_post >= float(pdm_p_post_min)),
        "variance_dropped": (np.isfinite(res.excess_var_ratio)
                             and res.excess_var_ratio <= float(var_drop_frac)),
    }
    res.flags.extend(f"fail_{k}" for k, v in checks.items() if not v)
    if np.isfinite(res.pre_decline_sigma) and res.pre_decline_sigma <= -PRE_DECLINE_SIGMA_MAX:
        res.flags.append("pre_decline_precession_like")
    if np.isfinite(res.amp_modulation) and res.amp_modulation > AMP_MODULATION_MAX:
        res.flags.append("pre_amplitude_modulated")
    if np.isfinite(res.mean_shift_mag) and abs(res.mean_shift_mag) > float(mean_shift_max_mag):
        res.flags.append("mean_flux_changed")
    if (measure_efficiency and np.isfinite(res.eta_min_post)
            and res.eta_min_post < float(eta_min)):
        res.flags.append("low_efficiency")
    if not checks["variance_dropped"] and checks["pattern"]:
        res.flags.append("variance_conserved_mode_switch_like")

    res.is_cessation = bool(all(checks.values()))
    if res.is_cessation:
        res.status = "cessation"
    elif not checks["efficiency"] and checks["pattern"]:
        res.status = "low_efficiency"
    elif not checks["mean_flux"]:
        res.status = "faded_or_brightened"
    elif not checks["variance_dropped"] and checks["pattern"]:
        res.status = "variance_conserved"
    elif not checks["pattern"]:
        res.status = "no_clean_transition"
    else:
        res.status = "rejected"
    return res


def combine_bands(res_g: CessationResult, res_r: CessationResult,
                  *, block_tol: int = 1) -> dict:
    """Two-band coincidence --- the repository ledger's first rule, at scoring time.

    A single-band cessation is an artefact until confirmed in a second band, and
    for this channel that is not a formality: a bad reference image, a
    filter-specific ghost, or a blend with a variable neighbour of one colour all
    stop a "period" in one band alone.  A clock that really stopped stopped in
    both, **and stopped at the same time** --- hence the block-index agreement,
    which a coincidence of two unrelated single-band artefacts would fail.
    """
    both = bool(res_g.is_cessation and res_r.is_cessation)
    same_block = (abs(res_g.split_block - res_r.split_block) <= int(block_tol)
                  if (res_g.split_block >= 0 and res_r.split_block >= 0) else False)
    same_period = False
    if np.isfinite(res_g.ref_period) and np.isfinite(res_r.ref_period) \
            and res_r.ref_period > 0:
        ratio = res_g.ref_period / res_r.ref_period
        same_period = any(abs(ratio - h) <= 0.05 for h in (0.5, 1.0, 2.0))
    n = int(res_g.is_cessation) + int(res_r.is_cessation)
    return {
        "n_bands_cessation": n,
        "two_band_cessation": bool(both and same_block and same_period),
        "same_transition_block": bool(same_block),
        "same_period_both_bands": bool(same_period),
        "split_block_g": int(res_g.split_block), "split_block_r": int(res_r.split_block),
        "period_g": float(res_g.ref_period), "period_r": float(res_r.ref_period),
        "eta_min_post": float(np.nanmin([res_g.eta_min_post, res_r.eta_min_post]))
        if np.isfinite([res_g.eta_min_post, res_r.eta_min_post]).any() else float("nan"),
        "p_persist_upper_joint": (res_g.p_persist_upper * res_r.p_persist_upper)
        if (np.isfinite(res_g.p_persist_upper) and np.isfinite(res_r.p_persist_upper))
        else float("nan"),
        "p_pinned_at_floor": bool(res_g.p_pinned_at_floor or res_r.p_pinned_at_floor),
        "status_g": res_g.status, "status_r": res_r.status,
    }


__all__ = ["CessationResult", "analyze_band", "combine_bands"]
