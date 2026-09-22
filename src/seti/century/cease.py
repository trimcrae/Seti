"""The century cessation statistic --- KNELL on plates, with censoring.

KNELL (``docs/knell.md`` section 3) established the governing rule: a
non-detection of a period is evidence of cessation only when normalised by the
**injection-measured efficiency of that block, in that block's own sampling
and noise**.  On CCD data the block's noise is its photometric scatter.  On
plates two more things vary block to block and both delete detections without
any change in the star:

1. **Per-plate limiting magnitude.**  A variable whose faint phase sinks below
   a plate's limit is simply not measured on that plate.  Shallow plates
   therefore see a *smaller* amplitude and fewer points, and plate depth is a
   property of the series --- of calendar time.  The injection here is
   **censored**: the injected light curve is evaluated on every plate of the
   block, detections and non-detections alike, and a point survives only if
   its injected magnitude is brighter than that plate's own limit.
2. **Exposure smearing.**  A patrol exposure of tens of minutes reduces a
   sinusoid of frequency ``f`` by ``|sinc(f t_exp)|``.  The injected amplitude
   is the intrinsic amplitude (pre-block measurement deconvolved by the
   pre-block mean smear) times each plate's own smear factor.

The detector is a **fixed-window** test: the catalogued period is known, so a
block is "detected" when its maximum generalised Lomb-Scargle power over a
narrow window around the reference frequency (and its first harmonic) exceeds
the ``1 - fap`` quantile of the same statistic on permutations of that block's
own magnitudes.  Fixed-window rather than fixed-frequency because periods
drift over a century; narrow rather than blind because the trials factor of a
blind search is what a photographic error budget cannot afford.  Because a
fixed-window periodogram is O(N) per frequency, every injected trial gets its
**own** permutation threshold on its own censored point set, so the censoring
changes the null exactly the way it changes the data.

Pure functions; no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..knell.blocks import (
    block_periodogram,
    excess_variance,
    gls_power,
    pdm_theta_pvalue,
    permutation_threshold,
    sine_amplitude,
)
from ..knell.efficiency import Efficiency, clopper_pearson_upper, persistence_pvalue
from .lightcurve import CenturyBlock, CenturyLC, calendar_blocks, mjd_to_year, smear_factor
from .step import MENZEL_GAP_END, MENZEL_GAP_START

HARMONICS = (1.0, 2.0)


# ---------------------------------------------------------------------------
# Frequency windows
# ---------------------------------------------------------------------------


def window_freqs(f_ref: float, t_span_days: float, *, half_width_frac: float = 0.005,
                 oversample: float = 5.0, harmonics=HARMONICS, max_per_window: int = 2000
                 ) -> np.ndarray:
    """Fine grids around ``h * f_ref`` for each harmonic ``h``."""
    T = max(float(t_span_days), 1.0)
    df = 1.0 / (float(oversample) * T)
    out = []
    for h in harmonics:
        fc = float(h) * float(f_ref)
        lo, hi = fc * (1.0 - half_width_frac), fc * (1.0 + half_width_frac)
        n = int(np.ceil((hi - lo) / df)) + 1
        n = int(np.clip(n, 3, max_per_window))
        out.append(np.linspace(lo, hi, n))
    return np.concatenate(out)


def refine_reference(t, y, e, f_cat: float, *, half_width_frac: float = 0.01,
                     n_grid: int = 4001, harmonics=HARMONICS) -> tuple[float, float, float]:
    """``(f_ref, harmonic, power)``: the harmonic of the catalogue frequency the
    data actually carry, refined on ``(t, y, e)``.

    An eclipsing binary catalogued at its orbital period peaks at ``2/P`` in a
    sinusoid-based periodogram; a pulsator at ``1/P``.  Taking the higher union
    power fixes the frequency every block statistic is then evaluated at.
    """
    t = np.asarray(t, dtype=float)
    if len(t) < 8 or not np.isfinite(f_cat) or f_cat <= 0:
        return float(f_cat), 1.0, float("nan")
    best = (float(f_cat), 1.0, -1.0)
    for h in harmonics:
        fc = float(h) * float(f_cat)
        grid = np.linspace(fc * (1 - half_width_frac), fc * (1 + half_width_frac), int(n_grid))
        p = gls_power(t, y, e, grid)
        i = int(np.argmax(p))
        if float(p[i]) > best[2]:
            best = (float(grid[i]), float(h), float(p[i]))
    return best


# ---------------------------------------------------------------------------
# Per-block detection
# ---------------------------------------------------------------------------


@dataclass
class BlockDetection:
    index: int
    n: int
    n_nd: int
    year_mid: float
    year_lo: float
    year_hi: float
    power: float
    threshold: float
    detected: bool
    best_freq: float
    fap_empirical: float
    mean_mag: float
    median_err: float
    median_lim: float
    blend_frac: float
    series: str
    amp_mmag: float = float("nan")
    amp_sigma_mmag: float = float("nan")
    excess_var: float = float("nan")
    eta: float = float("nan")
    eta_uncensored: float = float("nan")
    miss_upper: float = float("nan")
    censor_frac: float = float("nan")
    n_trials: int = 0
    role: str = ""            # pre_detected / pre_miss_explained / pre_miss_unexplained /
    #                            post_informative / post_uninformative / post_detected

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def detect_window(block: CenturyBlock, freqs: np.ndarray, *, fap: float = 0.01,
                  n_null: int = 200, rng=None) -> BlockDetection:
    """Fixed-window detection of one block against its own permutation null."""
    rng = np.random.default_rng(rng)
    t, y, e = block.t, block.y, block.e
    if block.n < 8:
        return BlockDetection(block.index, block.n, block.n_nd, block.year_mid, block.year_lo,
                              block.year_hi, 0.0, float("inf"), False, float("nan"),
                              float("nan"), block.mean_mag, block.median_err, block.median_lim,
                              block.blend_frac, block.dominant_series())
    p = gls_power(t, y, e, freqs)
    i = int(np.argmax(p))
    thr, nulls = permutation_threshold(t, y, e, freqs, fap=fap, n_null=n_null, rng=rng)
    fap_emp = (float((nulls >= p[i]).sum() + 1) / float(len(nulls) + 1)
               if len(nulls) else float("nan"))
    return BlockDetection(
        block.index, block.n, block.n_nd, block.year_mid, block.year_lo, block.year_hi,
        float(p[i]), float(thr), bool(np.isfinite(thr) and p[i] >= thr), float(freqs[i]),
        fap_emp, block.mean_mag, block.median_err, block.median_lim, block.blend_frac,
        block.dominant_series(),
    )


# ---------------------------------------------------------------------------
# Censoring-aware injection efficiency
# ---------------------------------------------------------------------------


@dataclass
class CensoredEfficiency:
    block_index: int
    n_trials: int
    n_detected: int
    eta: float
    miss_upper: float
    at_resolution_floor: bool
    eta_uncensored: float
    censor_frac: float          # mean fraction of the block's detections lost to censoring
    nd_gain_frac: float         # mean fraction of non-detection plates that would detect
    n_epochs: int
    n_nd: int
    amp_injected_mmag: float
    smear_modelled: bool

    def as_efficiency(self, period: float) -> Efficiency:
        return Efficiency(self.block_index, float(period), self.amp_injected_mmag,
                          self.n_trials, self.n_detected, self.n_detected, self.eta, self.eta,
                          self.miss_upper, self.at_resolution_floor, "data_censored",
                          float("nan"), self.n_epochs)


def censored_efficiency(block: CenturyBlock, freqs: np.ndarray, f_inj: float,
                        amp_int_mag: float, *, n_trials: int = 200, n_null: int = 100,
                        fap: float = 0.01, rng=None, smear: bool = True,
                        min_points: int = 8) -> CensoredEfficiency:
    """Probability that this block detects a signal of intrinsic amplitude
    ``amp_int_mag`` at ``f_inj``, given its plates, their limits and their
    exposure times.

    The base light curve is the block's own detections (data mode: real
    sampling, real errors, real outliers).  The block's non-detection plates
    are given the block's level plus a resampled residual, so a plate that did
    not detect the star can detect it when the injected phase is bright.  A
    point survives if its injected magnitude is brighter than its plate limit.
    Each trial is scored with a permutation threshold on its *own* surviving
    points, so a shallow block's smaller N and different window enter the null
    exactly as they enter the data.
    """
    rng = np.random.default_rng(rng)
    n_trials = int(max(n_trials, 1))
    N, M = block.n, block.n_nd
    if N < min_points or not np.isfinite(f_inj) or f_inj <= 0 or not np.isfinite(amp_int_mag):
        return CensoredEfficiency(block.index, 0, 0, float("nan"), 1.0, False, float("nan"),
                                  float("nan"), float("nan"), N, M, float(amp_int_mag) * 1e3,
                                  bool(smear))
    t_all = np.concatenate([block.t, block.t_nd])
    lim_all = np.concatenate([block.lim, block.lim_nd])
    exp_all = np.concatenate([block.exptime_min, block.exptime_nd])
    sm = smear_factor(f_inj, exp_all) if smear else np.ones_like(t_all)
    smear_modelled = bool(smear and np.isfinite(exp_all).any())
    resid = block.y - np.median(block.y)
    level = float(np.median(block.y))
    e_med = float(np.median(block.e))

    n_det = n_det_unc = 0
    censor_lost = []
    nd_gained = []
    for _ in range(n_trials):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        sig = amp_int_mag * sm * np.sin(2.0 * np.pi * f_inj * t_all + phase)
        base_nd = level + resid[rng.integers(0, N, size=M)] if M else np.array([], float)
        y_all = np.concatenate([block.y, base_nd]) + sig
        e_all = np.concatenate([block.e, np.full(M, e_med)])
        keep = ~np.isfinite(lim_all) | (y_all <= lim_all)
        censor_lost.append(float(np.mean(~keep[:N])))
        if M:
            nd_gained.append(float(np.mean(keep[N:])))
        # Uncensored: the block's own detections plus the signal, as KNELL does.
        y_unc = block.y + sig[:N]
        p_unc = gls_power(block.t, y_unc, block.e, freqs)
        thr_unc, _ = permutation_threshold(block.t, y_unc, block.e, freqs, fap=fap,
                                           n_null=n_null, rng=rng)
        if np.isfinite(thr_unc) and p_unc.max() >= thr_unc:
            n_det_unc += 1
        # Censored: the points that would actually have been measured.
        tk, yk, ek = t_all[keep], y_all[keep], e_all[keep]
        if len(tk) < min_points:
            continue
        p = gls_power(tk, yk, ek, freqs)
        thr, _ = permutation_threshold(tk, yk, ek, freqs, fap=fap, n_null=n_null, rng=rng)
        if np.isfinite(thr) and p.max() >= thr:
            n_det += 1
    n_miss = n_trials - n_det
    return CensoredEfficiency(
        block.index, n_trials, n_det, n_det / n_trials,
        float(clopper_pearson_upper(n_miss, n_trials)), bool(n_miss == 0),
        n_det_unc / n_trials, float(np.mean(censor_lost)) if censor_lost else float("nan"),
        float(np.mean(nd_gained)) if nd_gained else float("nan"), N, M,
        float(amp_int_mag) * 1e3, smear_modelled,
    )


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------


@dataclass
class CenturyCessation:
    status: str = "insufficient_data"
    n_det: int = 0
    n_blocks: int = 0
    n_detected_blocks: int = 0
    period_cat: float = float("nan")
    f_ref: float = float("nan")
    period_ref: float = float("nan")
    harmonic: float = 1.0
    union_power: float = float("nan")
    amp_pre_mmag: float = float("nan")
    amp_pre_sigma_mmag: float = float("nan")
    amp_post_mmag: float = float("nan")
    amp_post_sigma_mmag: float = float("nan")
    amp_post_sq_mmag2: float = float("nan")
    amp_post_sq_sigma_mmag2: float = float("nan")
    amp_post_over_sigma: float = float("nan")
    n_post_isolated_detections: int = 0
    drop_sigma: float = float("nan")
    amp_injected_mmag: float = float("nan")
    smear_pre: float = 1.0
    smear_modelled: bool = False
    n_pre: int = 0
    n_pre_detected: int = 0
    n_pre_miss_explained: int = 0
    n_pre_miss_unexplained: int = 0
    n_post: int = 0
    n_post_informative: int = 0
    n_post_uninformative: int = 0
    post_span_yr: float = float("nan")
    transition_year: float = float("nan")
    transition_at_gap: bool = False
    last_detected_year: float = float("nan")
    first_post_year: float = float("nan")
    eta_min_post: float = float("nan")
    eta_uncensored_min_post: float = float("nan")
    censor_frac_max_post: float = float("nan")
    p_persist_upper: float = float("nan")
    p_pinned_at_floor: bool = False
    p_resolution_floor: float = float("nan")
    mean_pre_mag: float = float("nan")
    mean_post_mag: float = float("nan")
    mean_shift_mag: float = float("nan")
    mean_shift_across_gap: bool = False
    excess_var_pre: float = float("nan")
    excess_var_post: float = float("nan")
    excess_var_ratio: float = float("nan")
    pdm_theta_pre: float = float("nan")
    pdm_p_pre: float = float("nan")
    pdm_theta_post: float = float("nan")
    pdm_p_post: float = float("nan")
    blend_frac_pre: float = float("nan")
    blend_frac_post: float = float("nan")
    series_pre: str = ""
    series_post: str = ""
    series_common: str = ""
    series_overlap_frac: float = float("nan")
    post_detect_rate_deep: float = float("nan")
    post_deep_plates: int = 0
    post_blind_periodic: int = -1
    is_cessation: bool = False
    flags: list[str] = field(default_factory=list)
    blocks: list[BlockDetection] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "blocks"}
        d["flags"] = ";".join(self.flags)
        d["block_year"] = ",".join(f"{b.year_mid:.1f}" for b in self.blocks)
        d["block_n"] = ",".join(str(b.n) for b in self.blocks)
        d["block_detected"] = ",".join("1" if b.detected else "0" for b in self.blocks)
        d["block_eta"] = ",".join("" if not np.isfinite(b.eta) else f"{b.eta:.3f}"
                                  for b in self.blocks)
        d["block_role"] = ",".join(b.role for b in self.blocks)
        d["block_lim"] = ",".join(f"{b.median_lim:.2f}" for b in self.blocks)
        return d


def _wmean(vals, sigs) -> tuple[float, float]:
    v = np.asarray(vals, dtype=float)
    s = np.asarray(sigs, dtype=float)
    m = np.isfinite(v) & np.isfinite(s) & (s > 0)
    if not m.any():
        return float("nan"), float("nan")
    w = 1.0 / s[m] ** 2
    return float((v[m] * w).sum() / w.sum()), float(np.sqrt(1.0 / w.sum()))


def analyze_century(
    lc: CenturyLC,
    period_cat: float,
    *,
    block_years: float = 2.0,
    origin_year: float = 1880.0,
    min_epochs_block: int = 20,
    min_blocks: int = 4,
    half_width_frac: float = 0.005,
    oversample: float = 5.0,
    fap: float = 0.01,
    n_null: int = 200,
    n_trials: int = 200,
    n_null_trial: int = 100,
    eta_min: float = 0.90,
    p_persist_max: float = 0.01,
    min_pre_detected: int = 2,
    min_post_informative: int = 2,
    min_post_span_yr: float = 10.0,
    mean_shift_max_mag: float = 0.10,
    post_amp_sigma_max: float = 3.0,
    drop_sigma_min: float = 5.0,
    pdm_p_pre_max: float = 0.01,
    pdm_p_post_min: float = 0.05,
    pdm_null: int = 100,
    var_drop_frac: float = 0.35,
    blend_jump_max: float = 0.30,
    vanish_margin_mag: float = 0.5,
    vanish_rate_min: float = 0.3,
    gap=(MENZEL_GAP_START, MENZEL_GAP_END),
    smear: bool = True,
    measure_efficiency: bool = True,
    blind_check: bool = True,
    rng=None,
) -> CenturyCessation:
    """The century cessation test for one star with a catalogued period."""
    rng = np.random.default_rng(rng)
    res = CenturyCessation(period_cat=float(period_cat))
    good = lc.good
    res.n_det = int(good.sum())
    if not (np.isfinite(period_cat) and period_cat > 0):
        res.status = "no_catalogue_period"
        return res
    blocks = calendar_blocks(lc, block_years=block_years, origin_year=origin_year,
                             min_epochs_block=min_epochs_block, min_blocks=min_blocks)
    if not blocks:
        res.status = "insufficient_data"
        res.flags.append("too_few_usable_blocks")
        return res
    res.n_blocks = len(blocks)

    # -- reference frequency on the pre-gap union (the long lever arm), else all
    yr = lc.year
    pre_mask = good & (yr < float(gap[0]))
    ref_mask = pre_mask if pre_mask.sum() >= 30 else good
    f_ref, h, up = refine_reference(lc.t[ref_mask], lc.mag[ref_mask], lc.err[ref_mask],
                                    1.0 / float(period_cat))
    res.f_ref, res.harmonic, res.union_power = f_ref, h, up
    res.period_ref = float(h / f_ref) if f_ref > 0 else float("nan")   # the fold period

    # -- per-block fixed-window detection
    dets: list[BlockDetection] = []
    for b in blocks:
        freqs = window_freqs(f_ref, b.t_span, half_width_frac=half_width_frac,
                             oversample=oversample)
        dets.append(detect_window(b, freqs, fap=fap, n_null=n_null, rng=rng))
    res.blocks = dets
    det = np.array([d.detected for d in dets], dtype=bool)
    res.n_detected_blocks = int(det.sum())

    # -- fixed-frequency amplitude and excess variance in every block
    for b, d in zip(blocks, dets, strict=True):
        a, sa, _ = sine_amplitude(b.t, b.y, b.e, f_ref)
        d.amp_mmag, d.amp_sigma_mmag = a * 1e3, sa * 1e3
        d.excess_var = excess_variance(b.y, b.e)

    if not det.any():
        res.status = "period_not_recovered"
        res.flags.append("catalogue_period_not_seen_in_any_block")
        if measure_efficiency:
            # Could the best three blocks have seen the catalogued signal?  Use
            # the union amplitude as the injected one.
            a_u, sa_u, _ = sine_amplitude(lc.t[ref_mask], lc.mag[ref_mask], lc.err[ref_mask],
                                          f_ref)
            a_inj = max(a_u - sa_u, 0.0)
            res.amp_injected_mmag = a_inj * 1e3
            best = np.argsort([-b.n for b in blocks])[:3]
            etas = []
            for i in best:
                b = blocks[i]
                freqs = window_freqs(f_ref, b.t_span, half_width_frac=half_width_frac,
                                     oversample=oversample)
                ce = censored_efficiency(b, freqs, f_ref, a_inj, n_trials=max(n_trials // 4, 25),
                                         n_null=n_null_trial, fap=fap, rng=rng, smear=smear)
                dets[i].eta, dets[i].miss_upper = ce.eta, ce.miss_upper
                dets[i].eta_uncensored, dets[i].censor_frac = ce.eta_uncensored, ce.censor_frac
                dets[i].n_trials = ce.n_trials
                etas.append(ce.eta)
            if etas and np.nanmax(etas) >= eta_min:
                res.flags.append("plates_could_have_seen_it_catalogue_suspect")
            else:
                res.flags.append("plates_insufficient_for_catalogue_amplitude")
        return res
    if det.all():
        res.status = "still_periodic"
        for d in dets:
            d.role = "pre_detected"
        return res

    # -- transition.  Over a century there can be forty blocks, so a strict
    # "the last detected block ends the pre segment" split lets a SINGLE false
    # alarm, which the per-block threshold produces at rate `fap` by
    # construction, push the transition arbitrarily late.  When that false
    # alarm lands in the first block after the Menzel gap --- the densest
    # post-gap block in DASCH, so the likeliest place for one --- the split
    # moves across the gap, the gap's photometric step stops being deferred,
    # and the star is charged with the plates' own 0.2--0.4 mag offset.  That
    # is the Hippke/Lund failure mode with a periodogram in front of it.
    #
    # So the split is chosen by a false-alarm test instead of by the last
    # detection: scan s upward over the detected blocks and take the EARLIEST
    # split whose later detections are consistent with noise --- at most
    # max(1, ceil(3 fap n_post)) of them and never two adjacent, since two
    # adjacent late detections are a clock that came back (Blazhko-like) and
    # not a false alarm.  s = s_last always satisfies the test (it leaves no
    # later detections at all), so the scan always terminates; taking the
    # earliest acceptable s is what makes an isolated post-gap false alarm
    # cost nothing.  Detections after the accepted split are excluded from the
    # post blocks rather than believed.
    nb = len(blocks)
    s_last = int(np.max(np.nonzero(det)[0]))

    def _late_ok(s_try: int) -> tuple[bool, list[int]]:
        late = [i for i in range(s_try + 1, nb) if det[i]]
        n_post_try = max(nb - s_try - 1, 1)
        max_fa = max(1, int(np.ceil(3.0 * float(fap) * n_post_try)))
        adjacent = any(b - a == 1 for a, b in zip(late, late[1:], strict=False))
        return (len(late) <= max_fa and not adjacent), late

    s, late_det = s_last, []
    for s_try in range(nb):
        if not det[s_try]:
            continue                     # a transition begins after a DETECTION
        ok, late = _late_ok(s_try)
        if ok:
            s, late_det = s_try, late
            break
    if late_det:
        res.flags.append("post_isolated_detection")
    res.n_post_isolated_detections = len(late_det)
    pre_i, post_i = list(range(0, s + 1)), [i for i in range(s + 1, nb) if i not in late_det]
    res.n_pre, res.n_post = len(pre_i), len(post_i)
    res.n_pre_detected = int(det[:s + 1].sum())
    res.last_detected_year = dets[s].year_mid
    if not post_i:
        res.status = "no_clean_transition"
        res.flags.append("detected_in_final_block")
        for i in pre_i:
            dets[i].role = "pre_detected" if det[i] else "pre_miss_untested"
        return res
    for i in late_det:
        dets[i].role = "post_isolated_detection"
    res.first_post_year = dets[post_i[0]].year_mid
    res.transition_year = 0.5 * (dets[s].year_hi + dets[post_i[0]].year_lo)
    res.transition_at_gap = bool(dets[s].year_hi <= float(gap[0]) + 1e-6
                                 and dets[post_i[0]].year_lo >= float(gap[1]) - 1e-6)

    # -- amplitudes
    a_pre, sa_pre = _wmean([dets[i].amp_mmag for i in pre_i if det[i]],
                           [dets[i].amp_sigma_mmag for i in pre_i if det[i]])
    res.amp_pre_mmag, res.amp_pre_sigma_mmag = a_pre, sa_pre
    # Intrinsic amplitude: deconvolve the pre-block mean smear.
    pre_exp = np.concatenate([blocks[i].exptime_min for i in pre_i if det[i]])
    sm_pre = smear_factor(f_ref, pre_exp) if smear else np.ones_like(pre_exp)
    res.smear_pre = float(np.mean(sm_pre)) if sm_pre.size else 1.0
    res.smear_modelled = bool(smear and np.isfinite(pre_exp).any())
    a_inj = max(a_pre - sa_pre, 0.0) * 1e-3 / max(res.smear_pre, 1e-3)
    res.amp_injected_mmag = a_inj * 1e3

    # -- efficiency in every non-detected block (pre misses and post blocks)
    effs_post: list[Efficiency] = []
    for i in pre_i:
        d = dets[i]
        if det[i]:
            d.role = "pre_detected"
            continue
        if measure_efficiency:
            b = blocks[i]
            freqs = window_freqs(f_ref, b.t_span, half_width_frac=half_width_frac,
                                 oversample=oversample)
            ce = censored_efficiency(b, freqs, f_ref, a_inj, n_trials=n_trials,
                                     n_null=n_null_trial, fap=fap, rng=rng, smear=smear)
            d.eta, d.miss_upper, d.n_trials = ce.eta, ce.miss_upper, ce.n_trials
            d.eta_uncensored, d.censor_frac = ce.eta_uncensored, ce.censor_frac
            if np.isfinite(ce.eta) and ce.eta >= eta_min:
                d.role = "pre_miss_unexplained"
                res.n_pre_miss_unexplained += 1
            else:
                d.role = "pre_miss_explained"
                res.n_pre_miss_explained += 1
        else:
            d.role = "pre_miss_untested"
    informative: list[int] = []
    for i in post_i:
        d = dets[i]
        if det[i]:
            d.role = "post_detected"        # cannot happen by construction (s is last)
            continue
        if not measure_efficiency:
            d.role = "post_untested"
            continue
        b = blocks[i]
        freqs = window_freqs(f_ref, b.t_span, half_width_frac=half_width_frac,
                             oversample=oversample)
        ce = censored_efficiency(b, freqs, f_ref, a_inj, n_trials=n_trials,
                                 n_null=n_null_trial, fap=fap, rng=rng, smear=smear)
        d.eta, d.miss_upper, d.n_trials = ce.eta, ce.miss_upper, ce.n_trials
        d.eta_uncensored, d.censor_frac = ce.eta_uncensored, ce.censor_frac
        if np.isfinite(ce.eta) and ce.eta >= eta_min:
            d.role = "post_informative"
            informative.append(i)
            effs_post.append(ce.as_efficiency(res.period_ref))
        else:
            d.role = "post_uninformative"
    res.n_post_informative = len(informative)
    res.n_post_uninformative = res.n_post - len(informative)
    if informative:
        res.post_span_yr = float(dets[informative[-1]].year_hi - dets[informative[0]].year_lo)
        res.eta_min_post = float(min(dets[i].eta for i in informative))
        res.eta_uncensored_min_post = float(np.nanmin([dets[i].eta_uncensored
                                                       for i in informative]))
        res.censor_frac_max_post = float(np.nanmax([dets[i].censor_frac for i in informative]))
        pp = persistence_pvalue(effs_post)
        res.p_persist_upper = pp["p_persist_upper"]
        res.p_pinned_at_floor = bool(pp["pinned_at_floor"])
        res.p_resolution_floor = pp["resolution_floor"]
        # A fitted sine amplitude is Rayleigh-distributed under the null, so an
        # average of many post-block amplitudes is biased positive by ~1.25
        # sigma_A per block and its "significance" grows as sqrt(n_post).
        # Average the DEBIASED squared amplitude instead: E[A^2] = A_true^2 +
        # 2 sigma_A^2, Var[A^2 | null] = 4 sigma_A^4.
        a2, sa2 = _wmean([dets[i].amp_mmag ** 2 - 2.0 * dets[i].amp_sigma_mmag ** 2
                          for i in informative],
                         [2.0 * dets[i].amp_sigma_mmag ** 2 for i in informative])
        res.amp_post_sq_mmag2, res.amp_post_sq_sigma_mmag2 = a2, sa2
        res.amp_post_mmag = float(np.sqrt(max(a2, 0.0))) if np.isfinite(a2) else float("nan")
        res.amp_post_sigma_mmag = (float(np.sqrt(sa2)) if np.isfinite(sa2) and sa2 > 0
                                   else float("nan"))
        res.amp_post_over_sigma = (a2 / sa2) if (np.isfinite(sa2) and sa2 > 0) else float("nan")
        # Drop significance on the squared amplitudes: sigma(A_pre^2) ~ 2 A_pre sigma_pre.
        den = float(np.sqrt(np.nansum([(2.0 * a_pre * sa_pre) ** 2, sa2 ** 2])))
        res.drop_sigma = ((a_pre ** 2 - a2) / den) if den > 0 else float("nan")
    cmp_post = informative if informative else post_i

    # -- mean flux, variance budget, PDM, blend and series histories
    # The mean flux of the pre-transition state is measured on ONE side of the
    # Menzel gap.  If the detected pre blocks straddle it, the minority side is
    # dropped: mixing them averages the plates' own pre/post offset into the
    # star's "mean magnitude before", which is the quantity the mean-shift test
    # then compares across the gap.
    pre_det_i = [i for i in pre_i if det[i]]
    pre_seg = np.array([0 if dets[i].year_mid < float(gap[0]) else 1 for i in pre_det_i])
    if pre_det_i and 0 < int(pre_seg.sum()) < len(pre_seg):
        major = int(np.argmax(np.bincount(pre_seg, minlength=2)))
        pre_mean_i = [i for i, sg in zip(pre_det_i, pre_seg, strict=False) if sg == major]
        res.flags.append("mean_pre_restricted_to_one_gap_segment")
    else:
        pre_mean_i = pre_det_i
    res.mean_pre_mag = float(np.median([dets[i].mean_mag for i in pre_mean_i]))
    res.mean_post_mag = float(np.median([dets[i].mean_mag for i in cmp_post]))
    res.mean_shift_mag = res.mean_post_mag - res.mean_pre_mag
    # Whether the comparison straddles the gap is a statement about WHERE the
    # two sets of plates are in time, not about a block index: the median of
    # the years that made mean_pre against the earliest year that made
    # mean_post.  A single late block among the pre blocks cannot hide the gap.
    pre_yr_used = [dets[i].year_mid for i in pre_mean_i]
    post_yr_used = [dets[i].year_mid for i in cmp_post]
    res.mean_shift_across_gap = bool(
        res.transition_at_gap
        or (pre_yr_used and post_yr_used
            and float(np.median(pre_yr_used)) < float(gap[0]) <= float(np.min(post_yr_used))))
    res.excess_var_pre = float(np.nanmedian([dets[i].excess_var for i in pre_i if det[i]]))
    res.excess_var_post = float(np.nanmedian([dets[i].excess_var for i in cmp_post]))
    if np.isfinite(res.excess_var_pre) and res.excess_var_pre > 0:
        res.excess_var_ratio = float(max(res.excess_var_post, 0.0) / res.excess_var_pre)
    t_pre = np.concatenate([blocks[i].t for i in pre_i if det[i]])
    y_pre = np.concatenate([blocks[i].y for i in pre_i if det[i]])
    t_post = np.concatenate([blocks[i].t for i in cmp_post])
    y_post = np.concatenate([blocks[i].y for i in cmp_post])
    res.pdm_theta_pre, res.pdm_p_pre = pdm_theta_pvalue(t_pre, y_pre, res.period_ref,
                                                        n_null=int(pdm_null), rng=rng)
    res.pdm_theta_post, res.pdm_p_post = pdm_theta_pvalue(t_post, y_post, res.period_ref,
                                                          n_null=int(pdm_null), rng=rng)
    res.blend_frac_pre = float(np.nanmean([dets[i].blend_frac for i in pre_i]))
    res.blend_frac_post = float(np.nanmean([dets[i].blend_frac for i in cmp_post]))
    ser_pre = set(np.unique(np.concatenate([blocks[i].series.astype(str) for i in pre_i])))
    ser_post = set(np.unique(np.concatenate([blocks[i].series.astype(str) for i in cmp_post])))
    common = ser_pre & ser_post
    res.series_pre = ",".join(sorted(ser_pre))[:200]
    res.series_post = ",".join(sorted(ser_post))[:200]
    res.series_common = ",".join(sorted(common))[:200]
    n_post_pts = sum(blocks[i].n for i in cmp_post)
    n_common_pts = sum(int(np.isin(blocks[i].series.astype(str), list(common)).sum())
                       for i in cmp_post)
    res.series_overlap_frac = (n_common_pts / n_post_pts) if n_post_pts else float("nan")

    # -- vanished?  A cessation star is still DETECTED at its mean brightness on
    # plates deep enough to show it.  A star that is not is a vanishing claim
    # (the contested VASCO-class phenomenon), which this channel does not make.
    deep_det = deep_all = 0
    ref = res.mean_pre_mag + float(vanish_margin_mag)
    for i in cmp_post:
        b = blocks[i]
        deep_det += int(np.sum(np.isfinite(b.lim) & (b.lim >= ref)))
        deep_all += int(np.sum(np.isfinite(b.lim) & (b.lim >= ref))
                        + np.sum(np.isfinite(b.lim_nd) & (b.lim_nd >= ref)))
    res.post_deep_plates = int(deep_all)
    res.post_detect_rate_deep = (deep_det / deep_all) if deep_all else float("nan")

    # -- blind mode-switch check on the informative post blocks (candidates only)
    pattern_ok = (res.n_pre_detected >= min_pre_detected
                  and res.n_pre_miss_unexplained == 0
                  and res.n_post_informative >= min_post_informative)
    if blind_check and pattern_ok and informative:
        n_blind = 0
        for i in informative:
            bp = block_periodogram(blocks[i].as_knell_block(), fap=fap, n_null=n_null, rng=rng,
                                   min_period=0.2, max_period=100.0, oversample=oversample)
            n_blind += int(bp.detected)
        res.post_blind_periodic = n_blind

    # -- adjudicate
    mean_flux_ok = (np.isfinite(res.mean_shift_mag)
                    and abs(res.mean_shift_mag) <= float(mean_shift_max_mag))
    checks = {
        "pattern": pattern_ok,
        "post_span": (np.isfinite(res.post_span_yr)
                      and res.post_span_yr >= float(min_post_span_yr)),
        "efficiency": (np.isfinite(res.eta_min_post) and res.eta_min_post >= float(eta_min)),
        "p_persist": (np.isfinite(res.p_persist_upper)
                      and res.p_persist_upper <= float(p_persist_max)),
        # Across the gap the mean-flux test needs the FIELD step; screen passes it
        # provisionally and assess re-evaluates against the ensemble.
        "mean_flux": (mean_flux_ok or res.mean_shift_across_gap),
        "post_amp_zero": (np.isfinite(res.amp_post_over_sigma)
                          and res.amp_post_over_sigma <= float(post_amp_sigma_max)),
        "drop_significant": (np.isfinite(res.drop_sigma)
                             and res.drop_sigma >= float(drop_sigma_min)),
        # The permutation p-value cannot resolve below 1/(n_null + 1).
        "pdm_pre": (np.isfinite(res.pdm_p_pre)
                    and res.pdm_p_pre <= max(float(pdm_p_pre_max), 1.0 / (int(pdm_null) + 1))),
        "pdm_post": (np.isfinite(res.pdm_p_post) and res.pdm_p_post >= float(pdm_p_post_min)),
        "variance_dropped": (np.isfinite(res.excess_var_ratio)
                             and res.excess_var_ratio <= float(var_drop_frac)),
        "not_vanished": (not np.isfinite(res.post_detect_rate_deep)
                         or res.post_detect_rate_deep >= float(vanish_rate_min)),
        "blend_stable": (not (np.isfinite(res.blend_frac_pre) and np.isfinite(res.blend_frac_post))
                         or abs(res.blend_frac_post - res.blend_frac_pre) <= float(blend_jump_max)),
        "series_overlap": (np.isfinite(res.series_overlap_frac) and res.series_overlap_frac > 0),
        "no_mode_switch": res.post_blind_periodic <= 0,
    }
    res.flags.extend(f"fail_{k}" for k, v in checks.items() if not v)
    if res.mean_shift_across_gap and not mean_flux_ok:
        res.flags.append("mean_flux_gap_uncorrected")
    if res.transition_at_gap:
        res.flags.append("transition_at_menzel_gap")
    if res.n_pre_miss_unexplained > 0:
        res.flags.append("intermittent_detection")
    if not checks["not_vanished"]:
        res.flags.append("vanished_not_ceased")
    if not checks["blend_stable"]:
        res.flags.append("blend_transition")
    if not checks["series_overlap"]:
        res.flags.append("series_disjoint")
    if not checks["variance_dropped"] and checks["pattern"]:
        res.flags.append("variance_conserved_mode_switch_like")
    if res.post_blind_periodic > 0:
        res.flags.append("post_blind_periodic")
    if np.isfinite(res.eta_uncensored_min_post) and np.isfinite(res.eta_min_post) \
            and res.eta_uncensored_min_post - res.eta_min_post > 0.2:
        res.flags.append("plate_limit_limited")

    res.is_cessation = bool(all(checks.values()))
    if res.is_cessation:
        res.status = "cessation"
    elif not checks["not_vanished"]:
        res.status = "vanished_not_ceased"
    elif checks["pattern"] and not checks["efficiency"]:
        res.status = "low_efficiency"
    elif not checks["mean_flux"]:
        res.status = "faded_or_brightened"
    elif checks["pattern"] and not checks["variance_dropped"]:
        res.status = "variance_conserved"
    elif not checks["pattern"]:
        res.status = ("low_efficiency" if res.n_post_informative < min_post_informative
                      and res.n_post > 0 and res.n_pre_miss_unexplained == 0
                      else "no_clean_transition")
    else:
        res.status = "rejected"
    return res


def same_series_check(lc: CenturyLC, res: CenturyCessation, *, min_epochs_block: int = 20,
                      block_years: float = 2.0, origin_year: float = 1880.0,
                      rng=None, **kw) -> dict:
    """Re-run the test inside the plate series the pre and post blocks share.

    Emulsion, plate scale and depth are properties of a series, so a cessation
    that survives only when the post blocks are on a different series from the
    pre blocks is a statement about two telescopes.  ``untestable`` when no
    common series has enough plates on both sides.
    """
    common = [s for s in res.series_common.split(",") if s]
    if not common:
        return {"same_series_status": "untestable", "same_series_name": ""}
    ser = lc.series.astype(str)
    best, best_n = "", 0
    yr = lc.year
    for s_ in common:
        m = lc.good & (ser == s_)
        n_pre = int((m & (yr < res.transition_year)).sum())
        n_post = int((m & (yr >= res.transition_year)).sum())
        if min(n_pre, n_post) > best_n:
            best, best_n = s_, min(n_pre, n_post)
    if best_n < 2 * min_epochs_block:
        return {"same_series_status": "untestable", "same_series_name": best}
    sub = lc.subset(ser == best)
    r2 = analyze_century(sub, res.period_cat, block_years=block_years, origin_year=origin_year,
                         min_epochs_block=min_epochs_block, min_blocks=3, rng=rng,
                         blind_check=False, **kw)
    return {"same_series_status": r2.status, "same_series_name": best,
            "same_series_n_pre_detected": r2.n_pre_detected,
            "same_series_n_post_informative": r2.n_post_informative,
            "same_series_transition_year": r2.transition_year}


def year_of(mjd: float) -> float:
    return float(mjd_to_year(mjd))


__all__ = ["BlockDetection", "CensoredEfficiency", "CenturyCessation", "HARMONICS",
           "analyze_century", "censored_efficiency", "detect_window", "refine_reference",
           "same_series_check", "window_freqs"]
