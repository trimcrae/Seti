"""Astrophysical context for dimming candidates: the HR-diagram position.

A list of deep, aperiodic dippers is dominated by *mundane* astrophysics that
mimics the Boyajian signature:

* **R Coronae Borealis stars** --- evolved, carbon-rich supergiants that fade by
  several magnitudes at irregular intervals as they puff out sooty dust;
* **Mira / long-period variables** --- cool, luminous giants with large,
  semi-regular amplitude;
* **young stellar objects** (UX Ori, AA Tau "dippers") --- pre-main-sequence
  stars occulted by their own circumstellar disks.

What sets KIC 8462852 apart is that it is an ordinary *main-sequence* F star ---
not evolved, not a YSO --- with no infrared excess.  The single cheapest cut that
removes the dominant mimics is therefore the HR-diagram position: require the
candidate to sit on the main sequence, using the Gaia absolute magnitude (from
parallax) versus colour.  Giants and supergiants sit far above the main sequence;
white dwarfs far below; pre-main-sequence stars above-and-red.
"""

from __future__ import annotations

import numpy as np


def absolute_g(g_mag: float, parallax_mas: float) -> float:
    """Absolute Gaia G from apparent G and parallax (mas).  NaN if non-positive."""
    if not (np.isfinite(g_mag) and np.isfinite(parallax_mas)) or parallax_mas <= 0:
        return float("nan")
    return float(g_mag + 5.0 * np.log10(parallax_mas / 100.0))


def _main_sequence_mg(bp_rp: float) -> float:
    """Approximate main-sequence absolute G at a given Gaia BP-RP colour.

    A smooth empirical fit to the Gaia DR3 solar-neighbourhood main sequence over
    -0.2 < BP-RP < 4; used only to decide whether a star sits near the MS, well
    above it (giant), or well below (white dwarf), so modest accuracy suffices.
    """
    c = float(np.clip(bp_rp, -0.2, 4.0))
    # Cubic that tracks M_G ~ 1.5 at BP-RP=0 (A/F), ~5 at 0.8 (G/K), ~9 at 1.8 (M).
    return 2.0 + 4.3 * c - 0.55 * c**2 + 0.16 * c**3


def hr_class(g_mag: float, bp_rp: float, parallax_mas: float,
             parallax_over_error: float = 0.0, snr_min: float = 5.0) -> str:
    """Coarse HR-diagram class: main_sequence / giant / white_dwarf / unknown.

    ``unknown`` whenever the parallax is too uncertain (S/N below ``snr_min``) or a
    colour is missing --- we never assert a class we cannot support, so a candidate
    with no reliable distance is simply left unfiltered rather than wrongly cut.
    """
    if not np.isfinite(bp_rp) or parallax_over_error < snr_min:
        return "unknown"
    mg = absolute_g(g_mag, parallax_mas)
    if not np.isfinite(mg):
        return "unknown"
    ms = _main_sequence_mg(bp_rp)
    # Fainter (larger M_G) than the MS by >2.5 mag and blue => white dwarf branch.
    if mg > ms + 4.0 and bp_rp < 1.2:
        return "white_dwarf"
    # Brighter (smaller M_G) than the MS by >1.5 mag => giant/supergiant.
    if mg < ms - 1.5:
        return "giant"
    if abs(mg - ms) <= 1.5:
        return "main_sequence"
    # Above the MS and red (cool & over-luminous): subgiant / pre-MS / giant-ish.
    return "evolved_or_pms"


def resists_mundane(hr: str, period_power: float, period_power_max: float) -> bool:
    """A candidate 'resists' the mundane explanations when it is an aperiodic
    dimmer sitting on the *main sequence* --- i.e. not an evolved dusty giant
    (R CrB / Mira), not a white dwarf, and not a periodic eclipsing system."""
    return hr == "main_sequence" and period_power <= period_power_max


__all__ = ["absolute_g", "hr_class", "resists_mundane", "_main_sequence_mg"]
