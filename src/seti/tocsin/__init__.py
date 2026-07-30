"""TOCSIN --- the nightly alarm bell on the Rubin/LSST alert stream.

*A tocsin is the bell rung to raise an alarm.*  KNELL looks for clocks that
already stopped; TOCSIN watches the sky **tonight**.

Signature **S30** of ``docs/necrosignatures.md`` ("singularity flash" --- an
unclassified transient on a catalogued nearby dwarf, matching neither flare,
nova, nor microlensing) is the only event-residue signature in the taxonomy that
needs a *live* stream rather than an archive, and it was never built.  Rubin
alerts went world-public on 2026-02-24, which is the first time the signature is
reachable at all.

What TOCSIN adds that no broker filter can
------------------------------------------
A broker gives you a *filter*; it does not give you a *memory*.  Every published
alert-stream science filter is single-night and stateless.  The observable here
is not one flash --- a single flash is indistinguishable from a cosmic ray in one
band --- it is **coherence across nights at a fixed catalogued position**:
repetition, achromaticity, and timing structure accumulated over a persistent
per-star ledger.  That ledger is this channel's actual instrument.

Both flux polarities are screened by one funnel, because the difference-image
stream gives them symmetrically:

* ``flash`` (positive difference flux) --- S30, plus the specular-glint reading
  (a flat reflector returns the *stellar* spectrum, so the event is grey; a
  stellar flare is blue).  The ZTF glint channel died precisely because ZTF
  could rarely test achromaticity; Rubin's in-night filter pairs can.
* ``dip`` (negative difference flux) --- brief grey occultations, the
  short-timescale end that the ZTF dimming channel could not reach (it
  exhausted at the ~1.6-7.4% systematics floor).

The dominant confounders are named up front and each has a quantitative test:
flare chromaticity, proper-motion subtraction dipoles (which recur at *every*
visit --- a duty-cycle test kills them), solar-system movers, and cosmic rays
(single-visit, never repeating at one position).

Honest-accounting requirement
-----------------------------
A screen that runs every night is a candidate-generating machine unless the
**cumulative** number of trials is carried forward.  ``ledger.py`` tracks
stars x visits screened to date and every promotion threshold is stated against
that running total, never against one night's count.
"""

from .photometry import (
    AB_ZP_NJY,
    LSST_BAND_WL_UM,
    ab_to_njy,
    blackbody_colour_temperature,
    fractional_amplitude,
    greyness_z,
    njy_to_ab,
    predicted_amplitude_ratio,
)
from .screen import ScreenVerdict, screen_alerts
from .targets import match_alerts_to_targets, propagate_pm

__all__ = [
    "AB_ZP_NJY",
    "LSST_BAND_WL_UM",
    "ScreenVerdict",
    "ab_to_njy",
    "blackbody_colour_temperature",
    "fractional_amplitude",
    "greyness_z",
    "match_alerts_to_targets",
    "njy_to_ab",
    "predicted_amplitude_ratio",
    "propagate_pm",
    "screen_alerts",
]
