"""Artificial specular-glint detection --- a novel brightening technosignature.

Every technosignature search to date looks for a star getting *fainter* (dust,
megastructures, transits) or for emission/absorption lines.  A large *flat*
artificial surface --- a solar sail, a mirror, or a specular facet of a Dyson-swarm
element --- does the opposite: as its orientation sweeps through the
specular-reflection geometry it throws a brief, bright **glint** back at the
observer.  Two properties separate a glint from the mundane brightening a
ground-based survey sees:

* it is **brief** --- one or a few epochs, not a sustained outburst;
* it is **achromatic** --- a mirror reflects the star's own spectrum, so the flash
  brightens every band by the same factor, whereas a stellar flare (the dominant
  natural brightening) is strongly *blue* (chromatic), and a cataclysmic outburst
  is structured and recurrent.

This module scores the single-band light curve for brief high-amplitude
brightening; the achromaticity test (g vs r) is applied to the shortlist by the
vetting stage, exactly as multi-band achromaticity vets the dip channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GlintStats:
    n_epochs: int
    max_brighten: float       # largest fractional flux *increase* above baseline
    n_glint_epochs: int       # epochs brighter than baseline by > threshold & k-sigma
    n_glint_events: int       # discrete brightening events (contiguous epochs)
    brighten_sigma: float     # significance of the brightest glint epoch
    out_of_glint_rms: float   # robust scatter of the non-glint epochs (quiescence)
    score: float              # [0,1] specular-glint likeness

    def as_dict(self) -> dict:
        return {k: (int(v) if isinstance(v, int) else float(v))
                for k, v in self.__dict__.items()}


def detect_glints(time: np.ndarray, mag: np.ndarray, magerr: np.ndarray | None = None,
                  bright_min: float = 0.30, k_sigma: float = 5.0,
                  min_epochs: int = 30, max_events: int = 4,
                  merge_gap_d: float = 2.0) -> GlintStats | None:
    """Score a light curve for brief, high-amplitude brightening (specular glints).

    ``bright_min`` is the minimum fractional flux *increase* (0.30 = +30 %) and
    ``k_sigma`` the per-epoch significance above the baseline.  A glint is rare and
    brief, so the score rewards a high-amplitude brightening confined to a few
    epochs on an otherwise quiescent baseline, and is suppressed for a
    continuously variable (flaring) star.  Returns ``None`` if too few epochs.
    """
    t = np.asarray(time, dtype=float)
    m = np.asarray(mag, dtype=float)
    good = np.isfinite(t) & np.isfinite(m)
    t, m = t[good], m[good]
    if t.size < min_epochs:
        return None
    e = (np.asarray(magerr, dtype=float)[good] if magerr is not None
         else np.full(m.size, np.nanstd(m) or 0.02))
    e = np.where(np.isfinite(e) & (e > 0), e,
                 np.nanmedian(e[e > 0]) if np.any(e > 0) else 0.02)
    order = np.argsort(t)
    t, m, e = t[order], m[order], e[order]

    # A glint is brighter than the star's *typical* level, so reference the median
    # (not the bright-state baseline used for dips).
    med = float(np.median(m))
    dmag = med - m                        # >0 when brighter than the median
    frac_bright = 10.0 ** (0.4 * np.clip(dmag, 0, None)) - 1.0
    sig = dmag / (0.4 * np.log(10.0) * e + 1e-9)
    is_glint = (frac_bright >= bright_min) & (sig >= k_sigma)

    n_glint = int(is_glint.sum())
    max_brighten = float(np.nanmax(frac_bright)) if frac_bright.size else 0.0
    brighten_sigma = float(np.nanmax(sig)) if sig.size else 0.0

    # Discrete events (contiguous glint epochs, merged across a short gap).
    idx = np.flatnonzero(is_glint)
    n_events = 0
    if idx.size:
        n_events = 1
        for a, b in zip(idx[:-1], idx[1:], strict=False):
            if b != a + 1 or (t[b] - t[a]) > merge_gap_d:
                n_events += 1

    # Quiescence of the non-glint epochs (a flaring star is never quiet).
    q = m[~is_glint]
    if q.size >= 5:
        qmed = np.median(q)
        mad = np.median(np.abs(q - qmed)) * 1.4826
        out_rms = float(0.4 * np.log(10.0) * (mad if mad > 0 else np.std(q)))
    else:
        out_rms = 0.0

    # Score: high-amplitude + few discrete events + significant + quiescent baseline.
    amp_term = np.clip((max_brighten - bright_min) / 0.7, 0, 1)
    # peaks for 1-3 events, suppressed for none or many (flaring).
    ev_term = (np.clip(n_events / 1.0, 0, 1)
               * np.clip((max_events + 2 - n_events) / (max_events + 1), 0, 1))
    sig_term = np.clip((brighten_sigma - k_sigma) / 15.0, 0, 1)
    quiet_term = np.clip(1.0 - out_rms / 0.06, 0, 1)
    if n_events == 0 or n_events > max_events:
        score = 0.0
    else:
        score = float(np.clip(0.4 * amp_term + 0.2 * ev_term + 0.2 * sig_term
                              + 0.2 * quiet_term, 0, 1))
    return GlintStats(n_epochs=int(t.size), max_brighten=max_brighten,
                      n_glint_epochs=n_glint, n_glint_events=int(n_events),
                      brighten_sigma=brighten_sigma, out_of_glint_rms=out_rms,
                      score=score)


__all__ = ["GlintStats", "detect_glints"]
