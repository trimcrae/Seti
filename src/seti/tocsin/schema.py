"""The broker-agnostic normalised alert row.

Seven community brokers carry the Rubin stream and each renames, flattens and
subsets the Avro packet differently.  Rather than let one broker's column names
leak into the physics, every adapter in ``brokers.py`` emits this structure and
nothing downstream knows which broker a row came from.

Every field that a discriminator needs is ``| None``-typed on purpose.  A broker
that does not expose (say) a reliability score must produce ``None``, and the
funnel then records ``reliability_unavailable`` and refuses to count that test as
passed.  The alternative --- defaulting a missing flag to "fine" --- is how a
screen quietly stops screening.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Boundary between observing nights, as a fraction of a day subtracted from MJD
# before flooring: 16:00 UTC is local noon at Cerro Pachon (UTC-4), so an
# entire Chilean night carries one label even though it straddles UTC midnight.
NIGHT_BOUNDARY_FRAC = 16.0 / 24.0


def night_id(mjd: float) -> str:
    """Observing-night label for an MJD, as ``"n<integer>"``.

    Grouping by night (not by UTC date) is load-bearing: the in-night visit pair
    is what supplies the second band for the achromaticity test, and a UTC-date
    split would cut many pairs in half.
    """
    if mjd is None or not math.isfinite(float(mjd)):
        return "unknown"
    return f"n{int(math.floor(float(mjd) - NIGHT_BOUNDARY_FRAC))}"


@dataclass
class NormalizedAlert:
    """One difference-image detection, in the units the physics wants.

    ``dflux_njy`` is **signed**: positive for a source brighter than the
    template, negative for fainter.  Both are screened; see the module docstring
    of ``seti.tocsin``.
    """

    alert_id: str
    object_id: str
    mjd: float
    band: str
    ra: float
    dec: float
    dflux_njy: float
    dflux_err_njy: float
    broker: str = ""

    # Astrometry
    ra_err_arcsec: float | None = None
    dec_err_arcsec: float | None = None

    # Reference photometry measured by Rubin itself, in the SAME band and the
    # SAME photometric system as `dflux_njy`.  `template_flux_njy` is the forced
    # PSF flux on the coadd template (`diaSource.templateFlux`) --- i.e. the
    # star's quiescent flux --- which makes the fractional amplitude dF/F* an
    # internally consistent measurement with no cross-survey passband
    # transformation.  This is strictly better than the Gaia synthetic-photometry
    # fallback and is preferred whenever present.
    template_flux_njy: float | None = None
    template_flux_err_njy: float | None = None
    science_flux_njy: float | None = None      # forced PSF on the direct image
    science_flux_err_njy: float | None = None

    # Quality / morphology.  None means "this broker did not tell us".
    snr: float | None = None
    reliability: float | None = None        # ML real-bogus score, higher = real
    reliability_version: str | None = None
    is_dipole: bool | None = None
    dipole_significance: float | None = None
    dipole_length_arcsec: float | None = None
    is_negative: bool | None = None        # detected as significantly negative
    extendedness: float | None = None      # 0 point-like, 1 extended
    trail_length_arcsec: float | None = None
    glint_trail: bool | None = None        # part of a satellite glint trail
    pixel_flag_bad: bool | None = None

    # Association / history
    ss_object_id: str | None = None         # known solar-system object
    n_prv_sources: int | None = None        # prior detections on this diaObject
    forced_mjds: list[float] = field(default_factory=list)
    visit: int | None = None
    detector: int | None = None
    raw: dict = field(default_factory=dict)

    @property
    def night(self) -> str:
        return night_id(self.mjd)

    @property
    def polarity(self) -> str:
        return "flash" if self.dflux_njy > 0 else "dip"

    @property
    def pos_err_arcsec(self) -> float:
        """Quadrature position error, with a Rubin single-visit systematic floor.

        Brokers routinely omit per-axis astrometric errors.  Rather than treat a
        missing error as zero (which would make every separation infinitely
        significant), fall back to the floor.
        """
        floor = 0.05
        parts = [e for e in (self.ra_err_arcsec, self.dec_err_arcsec)
                 if e is not None and math.isfinite(e)]
        if not parts:
            return floor
        return max(floor, math.hypot(*parts) if len(parts) == 2 else parts[0])


REQUIRED_FIELDS = ("alert_id", "mjd", "band", "ra", "dec",
                   "dflux_njy", "dflux_err_njy")


def validate(alert: NormalizedAlert) -> list[str]:
    """Return a list of structural problems; empty means usable.

    Called by every adapter so a broker schema change surfaces as an explicit
    per-row rejection with a reason, not as NaNs propagating into statistics.
    """
    problems = []
    for f in REQUIRED_FIELDS:
        v = getattr(alert, f, None)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            problems.append(f"missing_{f}")
    if alert.band and alert.band not in ("u", "g", "r", "i", "z", "y"):
        problems.append(f"unknown_band_{alert.band}")
    if isinstance(alert.dflux_err_njy, float) and alert.dflux_err_njy <= 0:
        problems.append("nonpositive_flux_error")
    if isinstance(alert.dflux_njy, float) and alert.dflux_njy == 0:
        problems.append("zero_difference_flux")
    return problems
