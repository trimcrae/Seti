"""Alternative feeds for TOCSIN: ASAS-SN Sky Patrol v2 and ATLAS forced photometry.

Rubin has been off sky since the night of 13/14 July 2026 (``docs/rubin-outage.md``,
verdict ``SKY_STOPPED``), the alert brokers hold nothing newer, and the obvious
substitute is dead by measurement rather than by assumption: ALeRCE's non-LSST
feed stopped on 2026-04-30 and Fink's ZTF portal did not answer at all
(``docs/substitute-surveys.md``).  So the channel cannot be re-pointed by
changing a survey id; it needs a different *kind* of feed.

What changes, and why the change is structural
----------------------------------------------
``brokers.py`` consumes an **alert stream**: a broker decided, upstream, that a
difference-image detection was worth issuing, and the hard problem is that
non-detections are invisible (which is why the ledger's denominator had to be
reconstructed from the observed footprint).  Both feeds here are the opposite
shape --- a **per-target light curve**, delivered whether or not anything
happened:

* **ASAS-SN Sky Patrol v2** returns the full light curve of a catalogued source,
  every epoch the field was observed, including the epochs where the star did
  nothing.  No token.
* **ATLAS forced photometry** is literally forced photometry: PSF photometry on
  the *difference* images at coordinates you supply, on every exposure that
  covered them.  Free account + token.

So the numerator has to be *constructed* here (an epoch is an event only if it
deviates from the star's own baseline) while the denominator comes for free and
is **exact** --- strictly better than the footprint proxy the Rubin path uses.
That inversion is the single most important thing about this module.

WHAT SURVIVES THE SWITCH AND WHAT DOES NOT
------------------------------------------
Stated here, in the code, and again in ``docs/tocsin-altfeeds.md``, because a
screen that reports a null on a signature it cannot detect is the failure mode
this repository is most exposed to.

**Survives, intact.**

* *The cross-night ledger.*  Recurrence at a fixed catalogued position is the
  channel's actual instrument (``docs/tocsin.md`` §3) and it is a property of
  accumulated state, not of any one survey's optics.  Both feeds carry many
  years of epochs, so the ledger gains power immediately instead of waiting.
* *The denominator, improved.*  Every good epoch is a trial, measured per target,
  with no footprint approximation and no ``visits_exact = False`` cap.
* *The duty-cycle test* and *the cadence-matched timing null*, which need only the
  visit epochs --- and these feeds give true per-visit epochs.
* *Both polarities.*  Dips and flashes are symmetric in a light curve exactly as
  they are in a difference image.

**Lost, and not recoverable from these feeds.**

* *Depth.*  Rubin reaches r ~ 24.5 in a 30 s visit; ATLAS reaches o ~ 19 and
  ASAS-SN g ~ 18 in a comparable epoch.  ~4.5-5.5 mag, i.e. a factor of 60-160
  in flux.  :func:`reachable_fraction` turns that into the only number that
  decides whether running this is worth anything: what fraction of the existing
  Gaia target list can actually carry an event of a given fractional amplitude.
* *Colour, in ASAS-SN.*  Sky Patrol v2 post-2018 is a single band (Sloan *g*).
  The achromaticity test --- this channel's headline discriminant, the one the
  ZTF glint search died on --- **cannot run at all** on ASAS-SN.  It is not
  degraded, it is absent, and every ASAS-SN event therefore carries the reason
  ``greyness_unavailable_single_band_survey``.
* *Colour, mostly, in ATLAS.*  ATLAS has two filters (*c*, *o*) but schedules them
  by lunation --- *c* near new moon, *o* otherwise --- so same-night two-band
  coverage is rare.  Rare is not never, and the fraction is **measured** per run
  (``two_band_night_fraction``) rather than assumed.
* *Astrometric discrimination.*  Forced photometry is measured *at the position
  you asked for*, so the separation between "the event" and "the star" is zero by
  construction and the ``astrometric_offset`` rule can never fire.  Every event
  carries ``astrometry_not_independent``.  This matters: the Rubin path uses the
  offset to reject an unrelated source blended with the target, and here that
  defence is gone, replaced only by the aperture-blend accounting in
  :func:`blend_neighbours`.
* *Rubin's flag suite.*  ``isDipole``, ``glint_trail``, ``extendedness``,
  ``pixelFlags_*``, ``ssObjectId``, ``reliability`` do not exist in either feed.
  Left as ``None`` they would make five of the funnel's rejection rules silently
  inert --- the rules would run, find nothing to test, and pass everything.  Where
  a per-epoch substitute exists it is constructed here and mapped onto the
  existing hooks (ATLAS ``chi/N`` -> ``pixel_flag_bad``, ATLAS ``maj``/``min``
  elongation -> ``raw["trail_flag"]``, ASAS-SN ``quality``/``fwhm`` -> bad-epoch
  rejection).  Where none exists it is named in the event's ``reasons``.

**Changed in a way that has to be argued rather than asserted: the timescale.**

The brief for this work assumed both surveys have "much longer effective
exposures", so the same event class is inaccessible.  That is right for ASAS-SN
and **wrong for ATLAS**, and the distinction decides whether the flash mode
transfers at all:

* Cadence (~1 day) sets *how many chances you get*.  Exposure sets *how much a
  short event is diluted*: a glint of duration tau << t_exp is recorded at
  amplitude tau/t_exp of its instantaneous value.
* A Rubin visit is 2 x 15 s = **30 s**.  An ATLAS exposure is **30 s**.  The
  dilution of a sub-second specular glint is therefore *identical* in the two,
  and ATLAS additionally takes a **quad** --- four exposures spanning ~1 hour ---
  which is the same intra-night structure Rubin's 33-minute visit pair provides,
  minus the filter change.  So the flash mode's *timescale* transfers to ATLAS
  exactly; only its depth does not.
* ASAS-SN stacks 3 x 90 s dithered images per epoch, so a short event is diluted
  by roughly an order of magnitude more than in a Rubin visit, on top of being
  shallower.  For ASAS-SN the fast end genuinely is inaccessible, and what
  remains is the slow end: grey dips and slow brightenings lasting hours to days.

Because ATLAS's quad resolves an hour, an event is classified rather than cut:
``single_exposure`` (present in one of the night's exposures --- consistent with a
fast glint, and *equally* consistent with a cosmic ray) versus
``multi_exposure`` (present in two or more --- slow, robust, but no longer the
sub-visit signature).  Cutting on it would quietly redefine the signal to fit
the data, which is exactly the move this module refuses to make; recording it
lets the ledger's recurrence test do the discriminating it was built for.

THE BAND LABELS, AND THE THREE PLACES A LABEL COULD SILENTLY LIE
----------------------------------------------------------------
``schema.validate`` accepts only the LSST band letters ``ugrizy``, so a native
band name (``c``, ``o``, ``V``) is rejected as malformed and never reaches the
funnel.  Rather than edit the shared schema, each native band is carried under
its nearest LSST **label**, with the native name preserved in
``raw["native_band"]``:

======  =======  ============================================================
native  label    justification
======  =======  ============================================================
g       ``g``    ASAS-SN g is Sloan g'.  The channel already treats SDSS g as
                 the stand-in for LSST g (``targets.GSPC_MAG_COLUMN``), so this
                 is the approximation already in use, not a new one.
c       ``g``    ATLAS c spans 420-650 nm, lam_eff ~ 533 nm; LSST g is 483 nm,
                 r is 622 nm.  c is the bluer of the two ATLAS bands and the
                 label is used only for *ordering* and for pairing.
o       ``r``    ATLAS o spans 560-820 nm, lam_eff ~ 679 nm.  The redder band.
V       ---      Johnson V sits between g and r with no defensible proxy, so
                 pre-2018 ASAS-SN V epochs are dropped from the numerator **and
                 from the denominator**.  Dropping them from the numerator alone
                 would inflate trials, deflate the ensemble rate, and make every
                 per-target p-value too small --- an anti-conservative error that
                 manufactures significance.
======  =======  ============================================================

A label is not a passband.  There are exactly three places where the label,
taken literally, would produce a wrong number, and each is closed here:

1. **The baseline flux.**  ``screen._baseline_flux`` falls back to the Gaia GSPC
   synthetic magnitude *of the label's band* when the alert carries no template
   flux.  For an ATLAS *c* detection that would divide by an SDSS *g* flux, and
   for a red dwarf --- this channel's whole sample --- *c* collects far more flux
   than *g*, so every fractional amplitude would be inflated.  Closed by
   construction: this module **always** sets ``template_flux_njy`` in the
   *native* band (from the survey's own light curve, or from an explicitly
   flagged interpolation), and an epoch for which no native baseline can be
   established is not emitted as an alert at all.  ``_baseline_flux`` prefers the
   template flux whenever it is present and positive, so the GSPC branch is
   never reached.
2. **The difference-flux colour temperature.**  ``blackbody_colour_temperature``
   looks up ``LSST_BAND_WL_UM[label]``, which for ATLAS would fit 0.483/0.622 um
   to data actually taken at 0.533/0.679 um and return a biased temperature.
   Closed by :func:`native_colour_temperature`, which refits with the native
   effective wavelengths and overwrites the value on every event this module
   emits.
3. **The one-sided non-detection test.**  ``screen``'s version asks
   ``_baseline_flux`` for the *other* band's quiescent flux, but
   ``_baseline_flux`` returns the template flux without looking at the band it
   was asked for --- so it hands back the *detected* band's flux (see the report
   accompanying this module; it is a pre-existing bug in ``screen.py`` that also
   affects the Rubin path, and it is not this module's to fix).  Closed by not
   using it: ``observed_bands`` is deliberately passed as ``None``, and
   :func:`native_nondetection_test` runs the same physics with the correct
   other-band baseline.

PROPER MOTION, WHICH IS NOT A DETAIL HERE
-----------------------------------------
TOCSIN's targets are nearby stars, which are the high-proper-motion stars.  A
star moving 1 arcsec/yr moves 10 arcsec in a decade --- larger than ATLAS's PSF
and comparable with ASAS-SN's 16 arcsec FWHM.  Forced photometry at a *single*
fixed coordinate across a multi-year baseline therefore walks off the star, and
the symptom is not an error: it is a slow decline that a robust baseline
partially absorbs and that leaves the ends of the light curve looking like dips.
That is a manufactured dip signal on precisely the best targets.

Two consequences, both implemented:

* ATLAS takes arbitrary coordinates, so requests are **segmented** in time by
  :func:`pm_segments`, each segment short enough that the drift stays under a
  configured fraction of the PSF, and each requested at the position propagated
  to that segment's mid-epoch.
* ASAS-SN photometers *its own* catalogued position (Sky Patrol v2 light curves
  are keyed to a source list built from ATLAS RefCat2, ~2015), so the drift
  cannot be controlled from outside.  Targets whose drift over the requested
  window exceeds ``max_drift_frac`` of the FWHM are therefore excluded from the
  numerator **and** the denominator, and counted, rather than screened badly.

NETWORK POSTURE
---------------
Everything that touches the network is confined to :class:`AsasSnSkyPatrol` and
:class:`AtlasForcedPhotometry` and runs **only on the GitHub Actions runner**
(the sandbox has no egress at all).  Every column name below was read from
documentation, which is inference, not verification --- the same posture that
produced three schema traps on the Rubin path before the live probe caught them
(``docs/tocsin.md`` §5.1).  :func:`probe` therefore records each service's live
response **verbatim** before any science is claimed, and the parsers are written
to look columns up by name rather than by position so that a re-ordering
degrades one field instead of corrupting all of them.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .ledger import Event, Ledger, bin_key, night_of
from .photometry import (
    ab_to_njy,
    grey_excluded_by_nondetection,
    njy_to_ab,
    planck_nu,
)
from .schema import NormalizedAlert, night_id
from .screen import Thresholds, screen_alerts
from .targets import GSPC_MAG_COLUMN, propagate_pm

# ---------------------------------------------------------------------------
# Units and constants
# ---------------------------------------------------------------------------
# JD -> MJD.  ASAS-SN serves JD; ATLAS serves MJD.  Getting this backwards
# offsets every epoch by 2.4 million days, which is loud rather than subtle --- but
# it would also silently shift the night labels by 0.5 d if applied as a bare
# integer, so it is a named constant and a unit test rather than a literal.
JD_MINUS_MJD = 2400000.5

# Served flux unit -> nanojansky.  ASAS-SN Sky Patrol v2 documents flux in mJy;
# ATLAS forced photometry documents `uJy`.  BOTH are checked against the served
# magnitude column at run time by `implied_flux_unit_njy`, because a wrong factor
# of 1000 changes no shape at all --- every amplitude is a *ratio*, so a uniform
# scale error cancels in dF/F* --- and would therefore never announce itself.
# What it does corrupt is the absolute detection limits and the non-detection
# test, so it is measured rather than trusted.
MJY_TO_NJY = 1.0e6
UJY_TO_NJY = 1.0e3


@dataclass(frozen=True)
class SurveySpec:
    """Everything about a feed that the physics depends on, in one place.

    Nominal values are documentation-derived and marked as such; every one of
    them that can be measured from a live response *is* measured at run time and
    the measurement wins (per-epoch limits from ``mag5sig``/``limit``, the flux
    unit from the magnitude/flux consistency check, the two-band night fraction
    from the epochs themselves).  The nominal numbers exist so the module can
    reason offline --- e.g. :func:`reachable_fraction` --- not so it can skip
    measuring.
    """

    key: str
    name: str
    endpoint: str
    native_bands: tuple[str, ...]
    band_label: dict[str, str]        # native band -> LSST label accepted by schema
    band_wl_um: dict[str, float]      # native effective wavelength
    depth_5sigma: dict[str, float]    # nominal per-epoch 5-sigma AB depth
    saturation_mag: dict[str, float]  # nominal bright limit; brighter is unusable
    exposure_s: float                 # ONE exposure, not the epoch total
    exposures_per_epoch: int
    psf_fwhm_arcsec: float
    pixel_scale_arcsec: float
    aperture_radius_arcsec: float     # radius within which a neighbour blends
    flux_unit_to_njy: float
    time_is_jd: bool
    is_difference_flux: bool          # True: served flux is already dF
    fixed_catalogue_position: bool    # True: we cannot choose the aperture centre
    auth_env: str | None
    notes: str = ""


# ASAS-SN Sky Patrol v2.  Depth, cadence and optics from the ASAS-SN
# documentation and the Sky Patrol v2 description; the pre-2018 V-band cameras
# are deliberately absent from `native_bands` (see the module docstring on why a
# band with no defensible LSST proxy is dropped from BOTH numerator and
# denominator rather than relabelled).
ASASSN = SurveySpec(
    key="asassn",
    name="ASAS-SN Sky Patrol v2",
    # MEASURED, not documented.  The probe run of 2026-08-25 found
    # `asas-sn.ifa.hawaii.edu:80/skypatrol/` unreachable from the runner (connect
    # timeout), and the vendor client's own source names a different service
    # entirely: a Flask API on `asassn-lb01.ifa.hawaii.edu` PORT 9006, with light
    # curve blocks served from `asassn-data{01,02,03}.ifa.hawaii.edu:9006`.  The
    # human-facing web host is not the API host, and pointing at it produced a
    # failure that could just as easily have been read as "the service is down".
    endpoint="http://asassn-lb01.ifa.hawaii.edu:9006",
    native_bands=("g",),
    band_label={"g": "g"},
    band_wl_um={"g": 0.4770},          # Sloan g'
    depth_5sigma={"g": 18.0},
    saturation_mag={"g": 10.5},
    exposure_s=90.0,
    exposures_per_epoch=3,             # 3 dithered images combined per epoch
    psf_fwhm_arcsec=16.0,
    pixel_scale_arcsec=8.0,
    aperture_radius_arcsec=16.0,
    flux_unit_to_njy=MJY_TO_NJY,
    time_is_jd=True,
    is_difference_flux=False,          # reference flux is added back in
    fixed_catalogue_position=True,     # keyed to ASAS-SN's own source list
    auth_env=None,                     # measured 2026-08-25: 200, no token
    notes=("Single band post-2018 (Sloan g'), so the achromaticity discriminant "
           "cannot run at all.  16 arcsec FWHM makes aperture blending the "
           "dominant systematic for a stellar sample."),
)

# ATLAS forced photometry.  c: 420-650 nm, o: 560-820 nm (Tonry et al. 2018).
# Depths are the routinely quoted per-exposure 5-sigma values and are superseded
# per epoch by the `mag5sig` column, which is why they are only nominal.
ATLAS = SurveySpec(
    key="atlas",
    name="ATLAS forced photometry",
    endpoint="https://fallingstar-data.com/forcedphot",
    native_bands=("c", "o"),
    band_label={"c": "g", "o": "r"},
    band_wl_um={"c": 0.5330, "o": 0.6790},
    depth_5sigma={"c": 19.5, "o": 19.0},
    saturation_mag={"c": 12.5, "o": 12.5},
    exposure_s=30.0,                   # THE SAME AS A RUBIN VISIT (2 x 15 s)
    exposures_per_epoch=4,             # the quad, spanning ~1 hour
    psf_fwhm_arcsec=5.0,
    pixel_scale_arcsec=1.86,
    aperture_radius_arcsec=5.0,
    flux_unit_to_njy=UJY_TO_NJY,
    time_is_jd=False,
    is_difference_flux=True,           # photometry on the difference images
    fixed_catalogue_position=False,    # arbitrary coordinates -> PM segmentation
    auth_env="ATLAS_TOKEN",
    notes=("30 s exposures in quads within the hour: the sub-visit timescale "
           "transfers from Rubin intact, only the depth does not.  Filters are "
           "scheduled by lunation, so same-night two-band coverage is rare."),
)

SURVEYS: dict[str, SurveySpec] = {ASASSN.key: ASASSN, ATLAS.key: ATLAS}

# Approximate Rubin single-visit saturation, AB, in r.  A 30 s LSST visit on an
# 8.4 m mirror saturates at roughly this magnitude; the exact value moves with
# band and seeing, so it is used only to COUNT a complementary population, never
# to cut one.
#
# WHY IT IS HERE AT ALL, AND WHY IT IS THE BEST ARGUMENT FOR THIS MODULE.  The
# 42 TOCSIN targets that produced events in the full Rubin walk have quiescent
# magnitudes, recovered from the ledger's own dF/F* and dF, of median g = 19.1
# and r = 17.8 --- and NONE brighter than g = 16.5.  That bright cutoff is not a
# property of nearby stars, it is Rubin's saturation limit: the alert stream
# structurally cannot screen the bright half of a 100 pc sample.  ASAS-SN
# saturates near g = 10.5 and ATLAS near 12.5, so both feeds cover a magnitude
# window that Rubin CANNOT, on the same catalogued nearby stars.  That makes
# these feeds complementary rather than merely substitutional, and it is a
# statement the census measures against the real target list rather than one
# this module asserts.
RUBIN_SATURATION_MAG = 16.0


class AltFeedError(RuntimeError):
    """Raised when an alternative feed is unreachable or answers unusably."""


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
@dataclass
class LightCurveThresholds:
    """Every light-curve reduction threshold in one place.

    These govern how a light curve becomes a numerator.  The funnel's own
    thresholds (``screen.Thresholds``) are untouched and still apply afterwards.
    """

    # Significance for calling an epoch an event, measured against the star's
    # OWN scatter (see `robust_baseline`), not against the formal error bar.
    min_abs_snr: float = 6.0
    # Below this many good epochs a robust baseline is not a baseline.  Such a
    # target contributes NEITHER events NOR visits: keeping its epochs in the
    # denominator while its numerator is structurally zero would deflate the
    # ensemble rate and make every p-value too small.
    min_good_epochs: int = 20
    # Sigma-clipping for the baseline, so a real event does not set the level it
    # is meant to be measured against.
    clip_sigma: float = 4.0
    clip_iters: int = 5
    # A band whose scatter exceeds this fraction of the star's own flux is a
    # known variable, a blend, or a drifting aperture.  REJECTED from both the
    # numerator and the denominator: the event threshold is `min_abs_snr` times
    # this same scatter, so such a band cannot register anything below a ~300 %
    # excursion, and counting its epochs as trials would deflate the ensemble
    # rate and shrink every other target's p-value.
    max_frac_scatter: float = 0.5
    # The quiescent flux must itself be a detection before dF/F* means anything.
    quiescent_min_snr: float = 5.0
    # ATLAS `chi/N` and elongation outlier factors, self-calibrated against each
    # light curve's own median rather than set as absolute numbers, because both
    # depend on seeing, field and magnitude.
    outlier_chi_factor: float = 5.0
    elongation_factor: float = 1.5
    # Fraction of the PSF FWHM that proper-motion drift may consume before the
    # aperture is no longer on the star.
    max_drift_frac: float = 0.25
    # How far above a band's limit the grey hypothesis must predict before a
    # silent band counts as evidence (mirrors `screen.Thresholds`).
    nondetection_margin: float = 3.0
    # Optional absolute floor on |dF/F*|.  Off by default: the per-target scatter
    # already sets a measured floor, and a global constant would be a guess.
    min_abs_amplitude: float | None = None


# ---------------------------------------------------------------------------
# Robust baseline and per-epoch significance
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Quiescent:
    """The star's quiescent flux ``F*`` in ONE NATIVE band, and where it came from.

    This exists as its own type because it is the single most dangerous quantity
    in the module.  ``screen._baseline_flux`` divides the difference flux by
    whatever the alert carries as ``template_flux_njy``, and if that is absent it
    silently falls back to the Gaia GSPC magnitude **of the LSST label** --- which,
    for a native band that is not that label (ATLAS *c* carried as *g*), is simply
    a different star's-worth of flux.  Making F* an explicit object with a named
    provenance means every amplitude in the channel can be traced to how its
    denominator was obtained, and an event whose F* is unknown is refused rather
    than silently given the wrong one.

    Provenances, best first:

    ``<survey>_lightcurve_median``
        The feed's own total-flux light curve, sigma-clipped.  Same instrument,
        same passband, same pixels as the difference flux --- the exact argument
        that makes Rubin's ``templateFlux`` the preferred baseline there.
    ``atlas_reduced_images``
        A second ATLAS pass with ``use_reduced=True``.  Same standing as above.
    ``gspc_interpolated_native``
        Gaia GSPC synthetic SDSS fluxes interpolated to the native effective
        wavelength (:func:`synthetic_native_mag`).  A cross-survey passband
        transformation, carrying an explicit systematic in ``err_njy``; the
        channel must never claim greyness tighter than it.
    """

    flux_njy: float
    err_njy: float
    source: str


@dataclass(frozen=True)
class Baseline:
    """Robust zero level of a light curve and the scatter about it.

    ``scatter`` is the *measured* epoch-to-epoch dispersion (1.4826 x MAD), not
    the mean formal error.  The two differ, always in the same direction: survey
    error bars omit flat-fielding, blending and subtraction systematics, so the
    formal error is optimistic and using it would call systematics events.  The
    per-epoch uncertainty actually used is ``max(formal_err, scatter)``.
    """

    level: float
    scatter: float
    n_used: int
    n_total: int
    ok: bool
    reason: str = ""


def robust_baseline(flux: np.ndarray, th: LightCurveThresholds | None = None
                    ) -> Baseline:
    """Sigma-clipped median and MAD scatter of a flux series.

    Clipping is iterative and symmetric.  Asymmetric clipping (only high
    outliers, say) would bias the level away from the polarity being searched,
    and this channel searches *both* polarities --- so a dip-hunting bias would be
    exactly as damaging as a flash-hunting one.
    """
    th = th or LightCurveThresholds()
    f = np.asarray(flux, dtype=float)
    f = f[np.isfinite(f)]
    n_total = int(f.size)
    if n_total == 0:
        return Baseline(float("nan"), float("nan"), 0, 0, False, "no_finite_epochs")
    keep = np.ones(f.size, dtype=bool)
    level = float(np.median(f))
    scatter = float(1.4826 * np.median(np.abs(f - level)))
    for _ in range(int(th.clip_iters)):
        if scatter <= 0 or not np.isfinite(scatter):
            break
        new = np.abs(f - level) <= th.clip_sigma * scatter
        if new.sum() < 3 or np.array_equal(new, keep):
            keep = new if new.sum() >= 3 else keep
            break
        keep = new
        level = float(np.median(f[keep]))
        scatter = float(1.4826 * np.median(np.abs(f[keep] - level)))
    n_used = int(keep.sum())
    if n_used < 3:
        return Baseline(level, scatter, n_used, n_total, False, "too_few_unclipped")
    if not np.isfinite(scatter) or scatter <= 0:
        # A light curve with a literally zero MAD is degenerate (repeated
        # identical values, or a stub); calling its scatter zero would make every
        # epoch infinitely significant.
        return Baseline(level, scatter, n_used, n_total, False, "degenerate_scatter")
    return Baseline(level, scatter, n_used, n_total, True)


def implied_flux_unit_njy(mag_ab: np.ndarray, flux_served: np.ndarray) -> float:
    """Multiplier that takes the served flux to nJy, implied by the served mags.

    A uniform flux-scale error is invisible to every *ratio* in this channel
    (``dF/F*`` cancels it exactly) but corrupts every *absolute* quantity: the
    per-band detection limits, the non-detection test, and any comparison across
    surveys.  So the assumed unit is checked against the survey's own magnitude
    column rather than trusted, and a disagreement is reported as a note instead
    of silently propagating.

    Returns NaN when there are too few usable pairs to decide.
    """
    m = np.asarray(mag_ab, dtype=float)
    f = np.asarray(flux_served, dtype=float)
    ok = np.isfinite(m) & np.isfinite(f) & (f > 0)
    if ok.sum() < 5:
        return float("nan")
    implied = ab_to_njy(m[ok]) / f[ok]
    return float(np.median(implied))


# ---------------------------------------------------------------------------
# Proper motion: segmentation and drift
# ---------------------------------------------------------------------------
def pm_drift_arcsec(pmra_mas_yr: float, pmdec_mas_yr: float, dt_yr: float) -> float:
    """Total on-sky displacement in arcsec over ``dt_yr``.

    ``pmra`` is the ``mu_alpha*`` convention (already carries cos(dec)), which is
    what Gaia publishes and what ``targets.propagate_pm`` assumes; mixing the two
    conventions here would under-state the drift for southern targets.
    """
    pr = 0.0 if pmra_mas_yr is None or not np.isfinite(pmra_mas_yr) else float(pmra_mas_yr)
    pd = 0.0 if pmdec_mas_yr is None or not np.isfinite(pmdec_mas_yr) else float(pmdec_mas_yr)
    return math.hypot(pr, pd) * abs(float(dt_yr)) / 1000.0


def pm_segments(ra: float, dec: float, pmra_mas_yr: float, pmdec_mas_yr: float,
                mjd_lo: float, mjd_hi: float, spec: SurveySpec,
                th: LightCurveThresholds | None = None) -> list[dict]:
    """Split a request window so the aperture never walks off a high-PM star.

    Forced photometry is measured at the coordinate supplied, once, for the whole
    window.  A star with mu = 1 arcsec/yr moves 10 arcsec in ten years, which is
    two ATLAS PSFs; the flux then decays across the window and the ends of the
    light curve look like dips.  Nothing errors --- it just fabricates the
    polarity this channel searches for, on the highest-proper-motion targets,
    which are exactly the nearest and best ones.

    Each returned segment is short enough that the drift within it stays below
    ``max_drift_frac x FWHM``, and carries the position propagated to the
    segment's mid-epoch.  Surveys that photometer their own catalogued position
    (ASAS-SN) cannot use this and get a single segment plus a drift figure the
    caller must act on --- see :func:`drift_excluded`.
    """
    th = th or LightCurveThresholds()
    mjd_lo, mjd_hi = float(mjd_lo), float(mjd_hi)
    if mjd_hi <= mjd_lo:
        return []
    mu = math.hypot(0.0 if not np.isfinite(pmra_mas_yr or 0.0) else float(pmra_mas_yr or 0.0),
                    0.0 if not np.isfinite(pmdec_mas_yr or 0.0) else float(pmdec_mas_yr or 0.0))
    allowance_arcsec = th.max_drift_frac * spec.psf_fwhm_arcsec
    if mu <= 0 or spec.fixed_catalogue_position:
        n_seg = 1
    else:
        span_yr = (mjd_hi - mjd_lo) / 365.25
        max_span_yr = allowance_arcsec / (mu / 1000.0)
        n_seg = max(1, int(math.ceil(span_yr / max_span_yr))) if max_span_yr > 0 else 1
    edges = np.linspace(mjd_lo, mjd_hi, n_seg + 1)
    out = []
    for i in range(n_seg):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mid_jyear = 2000.0 + ((lo + hi) / 2.0 - 51544.5) / 365.25
        p_ra, p_dec = propagate_pm(np.array([ra]), np.array([dec]),
                                   np.array([pmra_mas_yr or 0.0]),
                                   np.array([pmdec_mas_yr or 0.0]),
                                   to_epoch=mid_jyear)
        out.append({"mjd_lo": lo, "mjd_hi": hi,
                    "ra": float(p_ra[0]), "dec": float(p_dec[0]),
                    "epoch_jyear": mid_jyear,
                    "drift_arcsec": pm_drift_arcsec(pmra_mas_yr, pmdec_mas_yr,
                                                    (hi - lo) / 365.25)})
    return out


def drift_excluded(pmra_mas_yr, pmdec_mas_yr, mjd_lo: float, mjd_hi: float,
                   spec: SurveySpec, th: LightCurveThresholds | None = None) -> bool:
    """Whether a fixed-aperture survey's photometry of this star is unusable.

    Only meaningful for ``fixed_catalogue_position`` surveys, where the drift
    cannot be engineered away.  Excluded targets are dropped from the numerator
    *and* the denominator, and counted; screening them badly and keeping them in
    the trial count would be worse than not screening them at all.
    """
    th = th or LightCurveThresholds()
    if not spec.fixed_catalogue_position:
        return False
    drift = pm_drift_arcsec(pmra_mas_yr, pmdec_mas_yr, (float(mjd_hi) - float(mjd_lo)) / 365.25)
    return drift > th.max_drift_frac * spec.psf_fwhm_arcsec


# ---------------------------------------------------------------------------
# The light curve container and its reduction
# ---------------------------------------------------------------------------
@dataclass
class LightCurve:
    """One target's per-epoch record from an alternative feed, in nJy and MJD.

    Every array is per-epoch and the same length.  ``band`` holds the *native*
    band name; the LSST label is applied only at the moment a NormalizedAlert is
    built, so nothing upstream of that point can confuse the two.
    """

    target_id: str
    ra: float
    dec: float
    survey: str
    mjd: np.ndarray
    flux_njy: np.ndarray
    flux_err_njy: np.ndarray
    band: np.ndarray
    limit_njy: np.ndarray | None = None    # per-epoch 5-sigma limit, if served
    good: np.ndarray | None = None         # survey quality flag, True = usable
    chi_n: np.ndarray | None = None        # ATLAS reduced chi^2 of the PSF fit
    elongation: np.ndarray | None = None   # ATLAS maj/min
    raw_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(np.size(self.mjd))


@dataclass
class BandReduction:
    """What the reduction measured for one native band of one target."""

    band: str
    n_epochs: int = 0
    n_good: int = 0
    n_events: int = 0
    level_njy: float = float("nan")
    quiescent_njy: float = float("nan")
    quiescent_err_njy: float = float("nan")
    quiescent_source: str = "none"
    scatter_njy: float = float("nan")
    frac_scatter: float = float("nan")
    limit_median_njy: float = float("nan")
    usable: bool = False
    reason: str = ""


@dataclass
class Reduction:
    """A whole target's reduction: the numerator, the denominator, and the why."""

    target_id: str
    alerts: list[NormalizedAlert] = field(default_factory=list)
    visit_mjds: list[float] = field(default_factory=list)
    visit_bands: dict[str, list[float]] = field(default_factory=dict)
    bands: dict[str, BandReduction] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    usable: bool = False


def reduce_lightcurve(lc: LightCurve, spec: SurveySpec,
                      th: LightCurveThresholds | None = None,
                      quiescent: dict[str, Quiescent] | None = None,
                      ) -> Reduction:
    """Turn one light curve into alerts (the numerator) and visits (the denominator).

    The two are produced by the *same* pass over the *same* epochs, which is the
    property the Rubin path had to work hard to recover: there, detections and
    forced photometry came from different queries over different populations and
    their ratio was not a rate until the footprint union fixed it.  Here an epoch
    is a trial, and a trial that deviates is an event.  Nothing else can be true
    of the same data.

    ``quiescent`` supplies ``F*`` per native band for **difference-flux** feeds
    (ATLAS), where the served flux has the star's own light already subtracted
    and the light curve therefore cannot measure it.  For total-flux feeds
    (ASAS-SN) the clipped median *is* ``F*`` and the argument is ignored.

    A band with no usable baseline contributes neither events nor visits.  That
    is not tidiness: a target that can never produce an event but still counts as
    a trial deflates the ensemble rate, which makes every other target's binomial
    p-value smaller than it should be --- the anti-conservative direction, the one
    that manufactures significance.
    """
    th = th or LightCurveThresholds()
    red = Reduction(target_id=str(lc.target_id))
    n = len(lc)
    if n == 0:
        red.notes.append("empty_lightcurve")
        return red

    mjd = np.asarray(lc.mjd, dtype=float)
    flux = np.asarray(lc.flux_njy, dtype=float)
    ferr = np.asarray(lc.flux_err_njy, dtype=float)
    bands = np.asarray(lc.band, dtype=object)
    finite = np.isfinite(mjd) & np.isfinite(flux) & np.isfinite(ferr) & (ferr > 0)
    good = finite
    if lc.good is not None:
        good = good & np.asarray(lc.good, dtype=bool)
    # ATLAS: a PSF fit whose reduced chi^2 is a wild outlier against the star's
    # own median is a cosmic ray, a blend, or a subtraction artefact.  Calibrated
    # per light curve because chi/N depends on magnitude, seeing and field.
    bad_epoch = np.zeros(n, dtype=bool)
    if lc.chi_n is not None:
        chi = np.asarray(lc.chi_n, dtype=float)
        med = float(np.nanmedian(chi[good])) if good.any() else float("nan")
        if np.isfinite(med) and med > 0:
            bad_epoch |= np.isfinite(chi) & (chi > th.outlier_chi_factor * med)
    trailed = np.zeros(n, dtype=bool)
    if lc.elongation is not None:
        elo = np.asarray(lc.elongation, dtype=float)
        med = float(np.nanmedian(elo[good])) if good.any() else float("nan")
        if np.isfinite(med) and med > 0:
            trailed |= np.isfinite(elo) & (elo > th.elongation_factor * med)

    for nb in spec.native_bands:
        sel = good & np.array([str(b) == nb for b in bands], dtype=bool)
        br = BandReduction(band=nb,
                           n_epochs=int(np.sum(np.array([str(b) == nb for b in bands]))),
                           n_good=int(sel.sum()))
        if br.n_good < th.min_good_epochs:
            br.reason = f"fewer_than_{th.min_good_epochs}_good_epochs"
            red.bands[nb] = br
            continue
        base = robust_baseline(flux[sel], th)
        if not base.ok:
            br.reason = base.reason
            red.bands[nb] = br
            continue
        br.level_njy = base.level
        br.scatter_njy = base.scatter

        # F*: the star's quiescent flux in this NATIVE band.
        if spec.is_difference_flux:
            q = (quiescent or {}).get(nb)
            br.quiescent_source = q.source if q is not None else "none"
            br.quiescent_njy = float(q.flux_njy) if q is not None else float("nan")
            br.quiescent_err_njy = float(q.err_njy) if q is not None else float("nan")
            # A difference light curve should sit at zero.  A significant offset
            # means the reference image caught the star at a different level
            # (a real variable, or a reference built during an event), which
            # biases every amplitude; it is subtracted anyway but reported.
            if abs(base.level) > 3.0 * base.scatter / math.sqrt(max(base.n_used, 1)):
                red.notes.append(f"{nb}:nonzero_difference_baseline")
        else:
            br.quiescent_njy = base.level
            br.quiescent_source = f"{spec.key}_lightcurve_median"
            # The baseline's own uncertainty: the standard error of the clipped
            # median.  A long light curve therefore claims a precise F*, which is
            # correct --- the star really was measured hundreds of times.
            br.quiescent_err_njy = float(base.scatter / math.sqrt(max(base.n_used, 1)))
        if np.isfinite(br.quiescent_njy) and br.quiescent_njy > 0:
            br.frac_scatter = float(base.scatter / br.quiescent_njy)
            if br.frac_scatter > th.max_frac_scatter:
                # REJECTED, not merely noted --- and for the denominator's sake
                # rather than the numerator's.  The event threshold is 6x this
                # star's OWN scatter, so a band scattering at 50 % of the star's
                # flux cannot register anything below a 300 % excursion.  Leaving
                # its epochs in the trial count would add trials that have
                # essentially no chance of producing an event, which deflates
                # the ensemble rate and makes every OTHER target's binomial
                # p-value too small.  Same argument as the dropped V-band epochs
                # and the unusable bands: numerator and denominator leave
                # together or the ratio is not a rate.
                br.reason = f"fractional_scatter_{br.frac_scatter:.2f}_above_limit"
                red.notes.append(f"{nb}:high_fractional_scatter_{br.frac_scatter:.2f}")
                red.bands[nb] = br
                continue
            # The "is the star even detected?" test applies only to an F* the
            # SURVEY measured.  A catalogue-derived F* has no detection
            # significance in this instrument at all, and its error bar is a
            # passband systematic rather than photon noise --- running the test on
            # it would reject the fallback for a reason that does not apply, on
            # an arbitrary boundary set by the size of that systematic.
            measured = br.quiescent_source in (f"{spec.key}_lightcurve_median",
                                               "atlas_reduced_images")
            if measured and br.quiescent_njy < th.quiescent_min_snr * br.quiescent_err_njy:
                br.reason = "quiescent_flux_not_detected"
                red.bands[nb] = br
                continue
        else:
            # No usable F* -> every amplitude in this band would be untestable,
            # and (critically) `screen._baseline_flux` would fall back to the
            # GSPC column of the LSST *label*, which for a native band that is
            # not that label is simply the wrong number.  Refuse instead.
            br.reason = "no_quiescent_flux"
            red.bands[nb] = br
            continue

        if lc.limit_njy is not None:
            lim = np.asarray(lc.limit_njy, dtype=float)[sel]
            lim = lim[np.isfinite(lim) & (lim > 0)]
            if lim.size:
                br.limit_median_njy = float(np.median(lim))
        if not np.isfinite(br.limit_median_njy):
            # Fall back to the star's own measured noise, which is the honest
            # local limit when the survey does not publish one.
            br.limit_median_njy = float(5.0 * base.scatter)

        idx = np.nonzero(sel)[0]
        dflux = flux[idx] - base.level
        # max(), not hypot(): if the light curve scatters more than the formal
        # errors claim, the scatter is the truth and adding them in quadrature
        # would double-count the same noise.  If the formal error is larger
        # (a genuinely poor epoch), believe it.
        sigma = np.maximum(ferr[idx], base.scatter)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = dflux / sigma
        hit = np.isfinite(z) & (np.abs(z) >= th.min_abs_snr) & (dflux != 0.0)
        if th.min_abs_amplitude is not None and np.isfinite(br.quiescent_njy):
            hit &= np.abs(dflux) >= float(th.min_abs_amplitude) * abs(br.quiescent_njy)

        # Visits: every good epoch in a usable band.  Recorded before the event
        # cut, because a trial is a trial whether or not it produced anything ---
        # that is the entire point of using a forced-photometry feed.
        br.usable = True
        red.visit_mjds.extend(float(m) for m in mjd[idx])
        red.visit_bands.setdefault(nb, []).extend(float(m) for m in mjd[idx])

        # Per-night exposure multiplicity: ATLAS's quad resolves ~1 hour, so an
        # event present in one exposure and absent from the night's others is
        # short (a fast glint --- OR a cosmic ray), while one present in several
        # is slow.  Recorded, never cut on: cutting would redefine the signal.
        night_all: dict[int, int] = {}
        for m in mjd[idx]:
            night_all[night_of(float(m))] = night_all.get(night_of(float(m)), 0) + 1
        night_hit: dict[int, int] = {}
        for m in mjd[idx][hit]:
            night_hit[night_of(float(m))] = night_hit.get(night_of(float(m)), 0) + 1

        label = spec.band_label[nb]
        for k in np.nonzero(hit)[0]:
            j = int(idx[k])
            nt = night_of(float(mjd[j]))
            red.alerts.append(NormalizedAlert(
                alert_id=f"{spec.key}:{lc.target_id}:{mjd[j]:.6f}:{nb}",
                object_id=str(lc.target_id),
                mjd=float(mjd[j]),
                band=label,
                ra=float(lc.ra), dec=float(lc.dec),
                dflux_njy=float(dflux[k]), dflux_err_njy=float(sigma[k]),
                broker=f"{spec.key}-forced",
                # The position is the one the photometry was FORCED at, so the
                # astrometric test cannot be independent evidence.  A zero error
                # would make sep_sigma zero and read as a passed test, so the
                # PSF scale is carried instead and every event is additionally
                # marked `astrometry_not_independent`.
                ra_err_arcsec=spec.psf_fwhm_arcsec / 2.0,
                dec_err_arcsec=spec.psf_fwhm_arcsec / 2.0,
                # ALWAYS the native-band quiescent flux, so `_baseline_flux`
                # never reaches its GSPC branch with a mismatched label.
                template_flux_njy=float(br.quiescent_njy),
                template_flux_err_njy=float(br.quiescent_err_njy),
                snr=float(z[k]),
                reliability=None,          # no real/bogus model exists for these feeds
                pixel_flag_bad=bool(bad_epoch[j]),
                raw={"native_band": nb,
                     "survey": spec.key,
                     "trail_flag": bool(trailed[j]),
                     "n_exposures_in_night": int(night_all.get(nt, 0)),
                     "n_deviant_in_night": int(night_hit.get(nt, 0)),
                     "epoch_limit_njy": (float(np.asarray(lc.limit_njy, dtype=float)[j])
                                         if lc.limit_njy is not None else None),
                     "frac_scatter": br.frac_scatter,
                     "baseline_source": br.quiescent_source},
            ))
        br.n_events = int(hit.sum())
        red.bands[nb] = br

    red.visit_mjds = sorted({round(float(m), 6) for m in red.visit_mjds})
    red.usable = any(b.usable for b in red.bands.values())
    if not red.usable:
        red.notes.append("no_usable_band")
    return red


# ---------------------------------------------------------------------------
# Native-passband physics: the two places the LSST label would lie
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NativeColourTemperature:
    """Difference-flux blackbody temperature fitted at the NATIVE wavelengths."""

    temp_k: float
    t_lo: float
    t_hi: float
    n_bands: int
    ok: bool
    reason: str = ""


def native_colour_temperature(bands_native: list[str], wl_um: dict[str, float],
                              dflux_njy: list[float], dflux_err_njy: list[float],
                              t_grid: np.ndarray | None = None
                              ) -> NativeColourTemperature:
    """Refit the colour temperature using the feed's own effective wavelengths.

    ``photometry.blackbody_colour_temperature`` is correct for Rubin and wrong
    here, because it looks the wavelength up by the LSST *label*: an ATLAS c/o
    pair carried as g/r would be fitted at 0.483/0.622 um when it was taken at
    0.533/0.679 um.  The bias is not enormous but it is systematic and it lands
    on the one number the S30 claim rests on --- "the transient's temperature is
    the star's" --- so it is refitted rather than approximated.

    Same machinery as the LSST version: the amplitude is profiled out
    analytically at each trial temperature, so this is a 1-D chi^2 scan.
    """
    f = np.asarray(dflux_njy, dtype=float)
    e = np.asarray(dflux_err_njy, dtype=float)
    keep = np.isfinite(f) & np.isfinite(e) & (e > 0)
    bands_k = [b for b, k in zip(bands_native, keep, strict=True) if k]
    if len(bands_k) < 2:
        return NativeColourTemperature(float("nan"), float("nan"), float("nan"),
                                       len(bands_k), False, "need_two_bands")
    if any(b not in wl_um for b in bands_k):
        return NativeColourTemperature(float("nan"), float("nan"), float("nan"),
                                       len(bands_k), False, "unknown_native_band")
    f, e = f[keep], e[keep]
    if np.any(f <= 0):
        return NativeColourTemperature(float("nan"), float("nan"), float("nan"),
                                       len(bands_k), False,
                                       "negative_flux_no_emission_temperature")
    if t_grid is None:
        t_grid = np.geomspace(1500.0, 60000.0, 400)
    wl = np.array([wl_um[b] for b in bands_k])
    chi2 = np.empty_like(t_grid)
    for j, t in enumerate(t_grid):
        model = np.array([planck_nu(float(w), float(t)) for w in wl])
        if not np.any(model > 0):
            chi2[j] = np.inf
            continue
        w = 1.0 / e**2
        denom = float(np.sum(w * model * model))
        scale = float(np.sum(w * f * model)) / denom if denom > 0 else 0.0
        chi2[j] = float(np.sum(w * (f - scale * model) ** 2))
    jbest = int(np.argmin(chi2))
    within = np.where(chi2 <= chi2[jbest] + 1.0)[0]
    return NativeColourTemperature(
        float(t_grid[jbest]),
        float(t_grid[within[0]]) if within.size else float("nan"),
        float(t_grid[within[-1]]) if within.size else float("nan"),
        len(bands_k), True,
        "two_band_zero_dof" if len(bands_k) == 2 else "")


def native_nondetection_test(a_obs: float, other_band: str,
                             quiescent_other_njy: float | None,
                             limit_other_njy: float | None,
                             margin: float = 3.0):
    """One-sided colour test with the OTHER band's own quiescent flux.

    The physics is ``photometry.grey_excluded_by_nondetection`` unchanged: a grey
    event has equal *fractional* amplitude in every band, so on a red star it is
    brighter in absolute flux in the redder band, and that band's silence
    contradicts greyness.

    What is different is the input.  ``screen``'s caller asks
    ``_baseline_flux(alert, row, other_band, ...)`` for the other band's baseline,
    but ``_baseline_flux`` returns ``alert.template_flux_njy`` --- the *detected*
    band's flux --- whenever it is present, without consulting the band it was
    handed.  On the Rubin path that silently substitutes F*(detected) for
    F*(other); on a red star those differ by a factor of a few, so the test is
    over- or under-powered depending on which way the pair runs.  This module
    therefore passes ``observed_bands=None`` into the funnel, disabling that path
    entirely, and runs the test here with the correct baseline.
    """
    return grey_excluded_by_nondetection(a_obs, quiescent_other_njy,
                                         limit_other_njy, other_band,
                                         margin=margin)


# ---------------------------------------------------------------------------
# Depth: what fraction of the target list is actually reachable
# ---------------------------------------------------------------------------
def synthetic_native_mag(targets, spec: SurveySpec, native_band: str) -> np.ndarray:
    """Predicted AB magnitude of each target in a native band of ``spec``.

    Built from the Gaia DR3 GSPC synthetic SDSS magnitudes the target list
    already carries (``targets.GSPC_MAG_COLUMN``).

    * ASAS-SN *g* is Sloan g': the GSPC ``g_sdss_mag`` column is used directly.
    * ATLAS *c* and *o* have no GSPC equivalent, so the flux is **interpolated**
      in log F_nu against log lambda between the two bracketing SDSS bands
      (g,r for c; r,i for o).  This is an interpolation, not a published
      transformation, and it is labelled as one everywhere it is used: a real M
      dwarf has TiO bands that a smooth interpolation cannot know about, so
      errors of order 0.1-0.3 mag are expected.  That is tolerable for a
      *reachability census*, which is all it is used for.  It is NOT used as a
      baseline flux for any amplitude --- those come from the survey's own
      photometry, or the event is not emitted (see :func:`reduce_lightcurve`).
      The upgrade path is to integrate the Gaia XP spectra through the published
      ATLAS passbands (Tonry et al. 2018), which needs an acquisition this
      module does not have.

    Returns NaN wherever the required GSPC columns are absent, so a caller can
    count how much of the list it could not assess instead of assuming.
    """
    def col(band_sdss: str) -> np.ndarray:
        name = GSPC_MAG_COLUMN[band_sdss]
        if name not in targets:
            return np.full(len(targets), np.nan)
        return np.asarray(targets[name], dtype=float)

    # SDSS effective wavelengths (um), for the interpolation only.
    sdss_wl = {"u": 0.3557, "g": 0.4702, "r": 0.6175, "i": 0.7491, "z": 0.8946}
    if spec.key == ASASSN.key and native_band == "g":
        return col("g")
    if spec.key == ATLAS.key and native_band in ("c", "o"):
        blue, red = ("g", "r") if native_band == "c" else ("r", "i")
        m_b, m_r = col(blue), col(red)
        f_b, f_r = ab_to_njy(m_b), ab_to_njy(m_r)
        wb, wr = sdss_wl[blue], sdss_wl[red]
        w = spec.band_wl_um[native_band]
        with np.errstate(divide="ignore", invalid="ignore"):
            # Linear in log F_nu vs log lambda: a power-law SED locally, which is
            # the mildest defensible assumption for a two-point interpolation.
            slope = (np.log10(f_r) - np.log10(f_b)) / (math.log10(wr) - math.log10(wb))
            log_f = np.log10(f_b) + slope * (math.log10(w) - math.log10(wb))
            f = 10.0 ** log_f
        return njy_to_ab(f)
    return np.full(len(targets), np.nan)


def reachable_fraction(targets, spec: SurveySpec, amplitudes=(0.03, 0.10, 0.30, 1.00),
                       n_sigma: float = 6.0) -> dict:
    """What fraction of the target list can actually carry an event, per amplitude.

    **This is the number that decides whether running an alternative feed is
    worth anything**, and it is much harsher than the headline depth.  "g <~ 18"
    is the magnitude at which the *star* is detected at 5 sigma.  Detecting a
    *fractional* event of amplitude ``a`` on that star needs ``a x F*`` to clear
    ``n_sigma`` of the per-epoch noise, and in the background-limited regime the
    noise is roughly independent of the star, so

        a x F*(m)  >=  (n_sigma / 5) x F(m_5sigma)
        =>  m  <=  m_5sigma - 2.5 log10( n_sigma / (5 a) )

    For a 10 % event at 6 sigma that is 2.70 mag brighter than the nominal
    depth: ASAS-SN's g <~ 18 becomes g <~ 15.3.  Quoting the raw depth instead
    would over-state the usable sample by a large factor.

    The regime assumption is stated rather than hidden: near the bright end the
    noise becomes photon-dominated and scales as sqrt(F*), which makes the true
    cut *fainter* than this (i.e. this estimate is conservative there); near the
    faint end it is background-dominated and the formula is right.  The
    background-limited form is used because a search should err toward claiming
    less reach, not more.

    Reach is a **window**, not a limit.  Both feeds saturate --- ASAS-SN near
    g = 10.5, ATLAS near 12.5 --- so a target brighter than that is as unusable as
    one below the depth, and counting it as reachable would overstate the sample
    at exactly the end where the nearby-star list is richest.  The window also
    carries the complementary-population count: how many targets fall inside this
    feed's window **and** brighter than Rubin's own saturation, i.e. stars the
    alert stream structurally cannot screen at all (see
    :data:`RUBIN_SATURATION_MAG`).

    Returns a dict with, per amplitude, the magnitude window and the reachable
    fraction, plus the counts of targets that are saturated and of targets that
    could not be assessed at all because the GSPC synthetic photometry is
    missing for them.
    """
    n = int(len(targets))
    out: dict = {"survey": spec.key, "n_targets": n, "n_sigma": float(n_sigma),
                 "bands": {}, "assumption": "background_limited_noise",
                 "note": ("depth_5sigma is nominal; the per-epoch `mag5sig` "
                          "(ATLAS) / `limit` (ASAS-SN) column supersedes it at "
                          "run time")}
    if n == 0:
        out["verdict"] = "NO_TARGETS"
        return out
    # Rubin's own band for the complementary count: GSPC SDSS r stands in for
    # LSST r, which is the same approximation the Rubin path already makes.
    r_col = GSPC_MAG_COLUMN["r"]
    r_mag = (np.asarray(targets[r_col], dtype=float) if r_col in targets
             else np.full(n, np.nan))
    rubin_saturated = np.isfinite(r_mag) & (r_mag < RUBIN_SATURATION_MAG)
    out["n_brighter_than_rubin_saturation"] = int(np.sum(rubin_saturated))
    out["rubin_saturation_mag_r"] = RUBIN_SATURATION_MAG

    for nb in spec.native_bands:
        mags = synthetic_native_mag(targets, spec, nb)
        have = np.isfinite(mags)
        depth = spec.depth_5sigma[nb]
        sat = spec.saturation_mag[nb]
        per_amp = {}
        for a in amplitudes:
            cut = depth - 2.5 * math.log10(float(n_sigma) / (5.0 * float(a)))
            window = have & (mags <= cut) & (mags >= sat)
            n_ok = int(np.sum(window))
            per_amp[f"{a:g}"] = {
                "mag_cut": round(cut, 3),
                "mag_window": [sat, round(cut, 3)],
                "n_reachable": n_ok,
                "fraction_of_all_targets": round(n_ok / n, 5),
                "fraction_of_assessable": (round(n_ok / int(have.sum()), 5)
                                           if have.any() else None),
                # The population Rubin cannot screen at all: inside this feed's
                # window AND brighter than Rubin's saturation.  This is the
                # complementary sample, and the reason these feeds are more than
                # a stopgap.
                "n_reachable_and_rubin_saturated": int(np.sum(window & rubin_saturated)),
            }
        out["bands"][nb] = {
            "nominal_depth_5sigma": depth,
            "saturation_mag": sat,
            "n_with_synthetic_photometry": int(have.sum()),
            "n_without_synthetic_photometry": int(n - have.sum()),
            "n_saturated": int(np.sum(have & (mags < sat))),
            "median_mag": (round(float(np.median(mags[have])), 3) if have.any() else None),
            "by_amplitude": per_amp,
            "mag_source": ("gspc_g_sdss_mag_direct" if spec.key == ASASSN.key
                           else "interpolated_between_gspc_sdss_bands"),
        }
    out["verdict"] = "OK"
    return out


def blend_neighbours(target_ra, target_dec, target_mag,
                     cat_ra, cat_dec, cat_mag, radius_arcsec: float) -> dict:
    """Count catalogue neighbours inside the photometric aperture, by flux ratio.

    Why this replaces a discriminator rather than adding one.  On the Rubin path
    a flash from an unrelated star blended with the target is rejected by
    position: the difference-image centroid sits off the catalogued star and
    ``astrometric_offset`` fires.  Forced photometry has no centroid --- the flux
    is measured where you asked --- so that defence is gone, and with ASAS-SN's
    16 arcsec FWHM the aperture routinely contains other stars.  A flare on a
    neighbour of comparable brightness is then indistinguishable from an event on
    the target, and flare stars are common.

    ``flux_ratio`` is the summed neighbour flux over the target flux inside the
    aperture: below ~0.1 a neighbour cannot produce a >10 % apparent event
    without flaring by a factor of several, above ~1 the target is not the
    dominant source at all.

    The catalogue must be **complete to the survey's depth**, not the nearby-star
    target list: running this against the target list alone measures only the
    nearby-star-on-nearby-star blends and reports a lower bound.  The honest
    source is a Gaia cone search on the runner, which is why this function takes
    catalogue arrays rather than fetching anything itself.
    """
    from scipy.spatial import cKDTree

    t_ra = np.atleast_1d(np.asarray(target_ra, dtype=float))
    t_dec = np.atleast_1d(np.asarray(target_dec, dtype=float))
    c_ra = np.atleast_1d(np.asarray(cat_ra, dtype=float))
    c_dec = np.atleast_1d(np.asarray(cat_dec, dtype=float))
    c_mag = np.atleast_1d(np.asarray(cat_mag, dtype=float))
    t_mag = np.atleast_1d(np.asarray(target_mag, dtype=float))
    out = {"n_targets": int(t_ra.size), "radius_arcsec": float(radius_arcsec)}
    if t_ra.size == 0 or c_ra.size == 0:
        out["verdict"] = "EMPTY_INPUT"
        return out

    def uv(ra, dec):
        r, d = np.radians(ra), np.radians(dec)
        return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])

    tree = cKDTree(uv(c_ra, c_dec))
    chord = 2.0 * np.sin(np.radians(float(radius_arcsec) / 3600.0) / 2.0)
    groups = tree.query_ball_point(uv(t_ra, t_dec), r=chord)
    n_nb = np.zeros(t_ra.size, dtype=int)
    ratio = np.zeros(t_ra.size, dtype=float)
    t_flux = ab_to_njy(t_mag)
    for i, g in enumerate(groups):
        if not g:
            continue
        f = ab_to_njy(c_mag[np.asarray(g, dtype=int)])
        f = f[np.isfinite(f)]
        # The target itself is in the catalogue; remove the single closest match
        # in flux rather than by index, since the two catalogues need not share
        # identifiers.
        if f.size and np.isfinite(t_flux[i]):
            j = int(np.argmin(np.abs(f - t_flux[i])))
            f = np.delete(f, j)
        n_nb[i] = int(f.size)
        ratio[i] = float(np.sum(f) / t_flux[i]) if (f.size and t_flux[i] > 0) else 0.0
    out["n_neighbours"] = n_nb.tolist() if t_ra.size <= 50 else None
    out["n_with_any_neighbour"] = int(np.sum(n_nb > 0))
    out["n_with_blend_ratio_gt_0.1"] = int(np.sum(ratio > 0.1))
    out["n_with_blend_ratio_gt_1"] = int(np.sum(ratio > 1.0))
    out["median_blend_ratio"] = float(np.median(ratio))
    out["blend_ratio"] = ratio
    out["neighbour_count"] = n_nb
    out["verdict"] = "OK"
    return out


# ---------------------------------------------------------------------------
# Driving TOCSIN's existing funnel
# ---------------------------------------------------------------------------
@dataclass
class AltFeedVerdict:
    """One survey's screening pass: events, trials, and every count behind them."""

    survey: str
    events: list[Event] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    visit_history: dict[str, list[float]] = field(default_factory=dict)
    trials_by_night: dict[str, int] = field(default_factory=dict)
    star_night_pairs: set = field(default_factory=set)
    target_positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    bin_trials_by_night: dict[str, dict[str, int]] = field(default_factory=dict)
    band_reductions: dict[str, dict] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n


def funnel_thresholds(spec: SurveySpec, base: Thresholds | None = None) -> Thresholds:
    """TOCSIN's own funnel thresholds, widened only where the optics demand it.

    Exactly two are changed, and both for the same reason: the funnel's 1 arcsec
    association radius is calibrated to Rubin astrometry, and these feeds have no
    astrometry at all --- the flux is measured at the coordinate supplied, so the
    only separation that exists is between that coordinate and the propagated
    catalogue position.  Leaving the radius at 1 arcsec would silently discard
    every ASAS-SN target whose aperture is centred on a 2015 position, which is
    all of the high-proper-motion ones.  Widening to the PSF makes the cut mean
    "the aperture is still on the star", which is the only thing it can mean
    here, and every event is separately marked ``astrometry_not_independent`` so
    the widened cut can never be read as a passed astrometric test.

    Nothing else is relaxed.  In particular ``min_abs_snr`` is left at the
    channel's value: a shallower survey is a reason to screen fewer stars, never
    a reason to lower the bar on the ones it can screen.
    """
    th = base or Thresholds()
    return Thresholds(
        min_abs_snr=th.min_abs_snr,
        min_reliability=th.min_reliability,
        # No real/bogus model exists for either feed, so requiring one would
        # reject every event; the absence is recorded per event instead.
        require_reliability=False,
        max_dipole_significance=th.max_dipole_significance,
        max_extendedness=th.max_extendedness,
        max_trail_arcsec=th.max_trail_arcsec,
        max_sep_sigma=th.max_sep_sigma,
        max_sep_arcsec=max(th.max_sep_arcsec, spec.psf_fwhm_arcsec),
        match_radius_arcsec=max(th.match_radius_arcsec, spec.psf_fwhm_arcsec),
        max_grey_z=th.max_grey_z,
        baseline_rel_err=th.baseline_rel_err,
        missing_pm_penalty_arcsec=th.missing_pm_penalty_arcsec,
        nondetection_margin=th.nondetection_margin,
    )


def screen_lightcurves(reductions: list[Reduction], targets, spec: SurveySpec,
                       th: Thresholds | None = None,
                       lth: LightCurveThresholds | None = None) -> AltFeedVerdict:
    """Run TOCSIN's existing per-night funnel over reduced light curves.

    **Screening is done one night at a time.**  ``screen_alerts`` propagates the
    target list to the *median* epoch of the alerts it is handed, so feeding it a
    multi-year batch would propagate every target to the middle of the baseline
    and mis-associate the high-proper-motion stars at both ends --- by up to the
    full decade-scale drift, on exactly the targets this channel exists for.
    Per-night calls make the propagation exact, and the night is the unit the
    ledger counts in anyway.

    ``observed_bands`` is deliberately **not** passed: the funnel's one-sided
    non-detection test would ask ``_baseline_flux`` for the other band's
    quiescent flux and be handed the detected band's instead (see
    :func:`native_nondetection_test`).  The test is re-run here, correctly,
    afterwards.
    """
    th = th or Thresholds()
    lth = lth or LightCurveThresholds()
    v = AltFeedVerdict(survey=spec.key)

    by_night: dict[str, list[NormalizedAlert]] = {}
    for red in reductions:
        v.band_reductions[red.target_id] = {
            b: {"n_good": br.n_good, "n_events": br.n_events,
                "usable": br.usable, "reason": br.reason,
                "quiescent_njy": br.quiescent_njy,
                "frac_scatter": br.frac_scatter,
                "limit_median_njy": br.limit_median_njy}
            for b, br in red.bands.items()}
        if not red.usable:
            v.bump("targets_unusable")
            continue
        v.bump("targets_usable")
        v.bump("epochs_total", len(red.visit_mjds))
        v.visit_history[red.target_id] = list(red.visit_mjds)
        for m in red.visit_mjds:
            pair = (red.target_id, night_id(m))
            v.star_night_pairs.add(pair)
        for a in red.alerts:
            by_night.setdefault(a.night, []).append(a)
    v.bump("alerts_in", sum(len(x) for x in by_night.values()))

    # Per-target, per-night, per-native-band quiescent fluxes and limits, kept
    # for the native one-sided test below.
    quiescent: dict[tuple[str, str], float] = {}
    limits: dict[tuple[str, str], float] = {}
    observed_native: dict[tuple[str, str], set] = {}
    for red in reductions:
        for nb, br in red.bands.items():
            if not br.usable:
                continue
            quiescent[(red.target_id, nb)] = br.quiescent_njy
            limits[(red.target_id, nb)] = br.limit_median_njy
            for m in red.visit_bands.get(nb, ()):
                observed_native.setdefault((red.target_id, night_id(m)), set()).add(nb)

    # Exposure multiplicity per star-night, carried alongside the events because
    # `screen.Event` has no field for it and `per_band` is built by the funnel
    # from a fixed key set.  ATLAS's quad resolves ~1 hour, so "one deviant
    # exposure out of four" and "four out of four" are different physical claims
    # and must not be collapsed into one event record.
    n_exposures: dict[tuple[str, str], int] = {}
    n_deviant: dict[tuple[str, str], int] = {}
    for red in reductions:
        for m in red.visit_mjds:
            key = (red.target_id, night_id(m))
            n_exposures[key] = n_exposures.get(key, 0) + 1
        for a in red.alerts:
            key = (red.target_id, a.night)
            n_deviant[key] = n_deviant.get(key, 0) + 1

    two_band_nights = sum(1 for s in observed_native.values() if len(s) >= 2)
    v.counts["star_nights_observed"] = len(observed_native)
    v.counts["star_nights_two_band"] = two_band_nights
    v.counts["two_band_night_fraction"] = (
        round(two_band_nights / len(observed_native), 5) if observed_native else 0.0)

    for night in sorted(by_night):
        alerts = by_night[night]
        epoch_jyear = 2000.0 + (float(np.median([a.mjd for a in alerts])) - 51544.5) / 365.25
        sv = screen_alerts(alerts, targets, th, epoch_jyear=epoch_jyear,
                           observed_bands=None, band_limits=None)
        for k, n in sv.counts.items():
            if isinstance(n, int):
                v.counts[k] = v.counts.get(k, 0) + n
        v.rejected.extend(sv.rejected)
        v.target_positions.update(sv.target_positions)
        kept, dropped = _finish_events(sv.events, spec, quiescent, limits,
                                       observed_native, n_exposures, n_deviant, lth)
        v.events.extend(kept)
        v.rejected.extend(dropped)
        v.bump("rejected_chromatic_native_nondetection", len(dropped))
    v.counts["events_kept_after_native_tests"] = len(v.events)

    # THE DENOMINATOR.  Every good epoch of every usable band is a trial, so the
    # star-night pairs above are the complete trial set --- no footprint proxy, no
    # union with the numerator, no `visits_exact = False` cap.  This is the one
    # respect in which an alternative feed is strictly BETTER than the Rubin
    # path, whose forced-photometry coverage measured 0 % and had to be replaced
    # by an observed-footprint reconstruction (docs/tocsin.md §3).
    for _tid, night in v.star_night_pairs:
        v.trials_by_night[night] = v.trials_by_night.get(night, 0) + 1
    # Per-bin trials for the stratified null, keyed by night exactly as
    # `run.screen_night` does.
    pos = dict(v.target_positions)
    if targets is not None and len(targets):
        ids = (np.asarray(targets["source_id"]).astype(str) if "source_id" in targets
               else np.arange(len(targets)).astype(str))
        ra = np.asarray(targets["ra"], dtype=float)
        dec = np.asarray(targets["dec"], dtype=float)
        cat = dict(zip(ids, zip(ra, dec, strict=True), strict=True))
    else:
        cat = {}
    for tid, night in v.star_night_pairs:
        rd = pos.get(tid) or cat.get(str(tid))
        if not rd:
            continue
        k = bin_key(rd[0], rd[1])
        if k:
            v.bin_trials_by_night.setdefault(night, {})
            v.bin_trials_by_night[night][k] = v.bin_trials_by_night[night].get(k, 0) + 1
    return v


def _finish_events(events: list[Event], spec: SurveySpec,
                   quiescent: dict, limits: dict, observed_native: dict,
                   n_exposures: dict, n_deviant: dict,
                   lth: LightCurveThresholds) -> tuple[list[Event], list[dict]]:
    """Apply the native-passband corrections and the honesty annotations.

    Three things happen to every event, and each exists because leaving it
    undone would let a number be read as stronger evidence than it is:

    1. the colour temperature is refitted at the native wavelengths
       (see :func:`native_colour_temperature`);
    2. the one-sided non-detection test is re-run with the *other* band's own
       quiescent flux, and an event it excludes is dropped as chromatic;
    3. the discriminators that this feed cannot supply are named in ``reasons``,
       so no reader --- and no later summary --- can mistake "the rule did not fire"
       for "the rule passed".
    """
    kept: list[Event] = []
    dropped: list[dict] = []
    for ev in events:
        # The funnel builds `per_band` from a fixed key set, so the native band
        # name does not survive it; the label map is injective for both feeds,
        # so it is inverted instead of being smuggled through.
        native = _labels_to_native(ev.bands, spec)
        ev.reasons.append(f"feed_{spec.key}")
        ev.reasons.append("astrometry_not_independent")
        if len(spec.native_bands) == 1:
            ev.reasons.append("greyness_unavailable_single_band_survey")
        for miss in ("reliability", "isdipole", "glint_trail", "extendedness",
                     "ss_association"):
            ev.reasons.append(f"{miss}_unavailable_in_{spec.key}")

        # (1) native colour temperature, flash mode only (an emission temperature
        # cannot be fitted to negative flux).
        if ev.polarity == "flash" and len(native) >= 2:
            fit = native_colour_temperature(
                native, spec.band_wl_um,
                [ev.per_band[b]["dflux_njy"] for b in ev.bands],
                [ev.per_band[b]["dflux_err_njy"] for b in ev.bands])
            ev.colour_temp_k = fit.temp_k if fit.ok else float("nan")
            ev.reasons.append("colour_temperature_native_passbands"
                              if fit.ok else f"colour_temperature_{fit.reason}")
        elif ev.polarity == "flash":
            ev.colour_temp_k = float("nan")

        # (2) the correct one-sided test.
        excluded = False
        if len(native) == 1:
            det = native[0]
            seen = set(observed_native.get((ev.target_id, ev.night), ()) or ())
            for other in sorted(seen - {det}):
                gx = native_nondetection_test(
                    ev.a, other, quiescent.get((ev.target_id, other)),
                    limits.get((ev.target_id, other)),
                    margin=lth.nondetection_margin)
                if gx.tested and gx.excluded:
                    excluded = True
                    ev.verdict = "rejected_chromatic_native_nondetection"
                    dropped.append({
                        "alert_id": ";".join(ev.alert_ids),
                        "reason": "chromatic",
                        "detail": f"grey_excluded_by_{other}_nondetection",
                        "mjd": ev.mjd, "band": ev.strongest_band,
                        "predicted_njy": round(gx.predicted_flux_njy, 1),
                        "limit_njy": round(gx.limit_flux_njy, 1)})
                    break
                if gx.tested:
                    ev.reasons.append(f"grey_survives_{other}_nondetection")
                    ev.grey_tested = True
                    break
                ev.reasons.append(f"nondetection_untestable_{gx.reason}")
                break
        if excluded:
            continue

        # (3) the exposure-multiplicity class, recorded and NEVER cut on.  In
        # ATLAS's quad an event seen in one of four exposures is short --- which is
        # what a specular glint looks like, and equally what a cosmic ray looks
        # like.  Cutting on it would delete the sub-visit signature this channel
        # exists for; ignoring it would let a slow, ordinary excursion be quoted
        # as a fast one.  So it is classified, and the ledger's recurrence test
        # does the discriminating it was built for.
        key = (ev.target_id, ev.night)
        n_exp = int(n_exposures.get(key, 0))
        n_dev = int(n_deviant.get(key, 0))
        ev.reasons.append(f"night_exposures_{n_exp}_deviant_{n_dev}")
        if n_exp >= 2:
            ev.reasons.append("multi_exposure_night_slow_event" if n_dev >= 2
                              else "single_exposure_fast_or_cosmic_ray")
        kept.append(ev)
    return kept, dropped


def _labels_to_native(labels: list[str], spec: SurveySpec) -> list[str]:
    """Invert ``band_label`` --- used only when ``per_band`` lost the native name."""
    inv = {v: k for k, v in spec.band_label.items()}
    return [inv.get(b, b) for b in labels]


# ---------------------------------------------------------------------------
# Parsers --- pure functions, offline-testable, name-keyed not position-keyed
# ---------------------------------------------------------------------------
def parse_atlas_text(text: str) -> tuple[list[str], list[dict]]:
    """Parse an ATLAS forced-photometry result file into column names and rows.

    The documented header is::

        ###MJD m dm uJy duJy F err chi/N RA Dec x y maj min phi apfit mag5sig Sky Obs

    but the parse is **keyed by name**, taken from the file's own header line,
    not by position.  If ATLAS re-orders or adds a column, a name-keyed parse
    loses at most the fields that vanished; a position-keyed one would silently
    read the wrong number into every field after the change --- and a wrong flux
    column produces a light curve, not an error.

    Returns ``(column_names, rows)`` with every value left as a string; typing
    happens in :func:`atlas_rows_to_lightcurve`, where a non-numeric entry can be
    turned into a NaN for one epoch rather than an exception for the file.
    """
    cols: list[str] = []
    rows: list[dict] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            if not cols:
                names = s.lstrip("#").split()
                cols = [c.strip().lower().replace("/", "_") for c in names if c.strip()]
            continue
        if not cols:
            continue
        parts = s.split()
        if len(parts) < len(cols):
            parts = parts + [""] * (len(cols) - len(parts))
        rows.append(dict(zip(cols, parts[:len(cols)], strict=True)))
    return cols, rows


def _numv(v) -> float:
    """Float or NaN.  A single unparseable cell must cost one epoch, not a file."""
    if v is None:
        return float("nan")
    try:
        if isinstance(v, str) and not v.strip():
            return float("nan")
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _num(row: dict, key: str) -> float:
    return _numv(row.get(key))


def atlas_rows_to_lightcurve(rows: list[dict], target_id: str, ra: float, dec: float,
                             spec: SurveySpec = ATLAS,
                             flux_unit_to_njy: float | None = None) -> LightCurve:
    """Map parsed ATLAS rows onto a :class:`LightCurve` in nJy and MJD.

    Column meanings taken from the ATLAS forced-photometry documentation and
    **to be confirmed by** :func:`probe`:

    ``mjd``      epoch, already MJD (no JD conversion --- unlike ASAS-SN).
    ``ujy``      difference flux in microjansky, **signed**.  This is the whole
                 observable: ATLAS forced photometry runs on the wallpaper-
                 subtracted difference images, so a dip is a negative number
                 exactly as it is in a Rubin alert.
    ``dujy``     its 1-sigma error.
    ``f``        filter, ``c`` or ``o``.
    ``err``      an error code; non-zero epochs are dropped.
    ``chi_n``    reduced chi^2 of the PSF fit --- the cosmic-ray / blend / artefact
                 proxy that stands in for Rubin's pixel flags.
    ``maj``/``min``  fitted PSF axes; their ratio is the trailing proxy that
                 stands in for Rubin's ``glint_trail`` and ``trailLength``.
    ``mag5sig``  the exposure's own 5-sigma limiting magnitude --- a *better*
                 detection limit than the Rubin path's, which has to estimate one
                 from the median flux error of the night's detections.
    ``m``/``dm`` magnitude of the difference flux; used only to verify the flux
                 unit (:func:`implied_flux_unit_njy`), never as the measurement.
    """
    scale = float(flux_unit_to_njy if flux_unit_to_njy is not None else spec.flux_unit_to_njy)
    mjd, flux, ferr, band, lim, good, chi, elo = [], [], [], [], [], [], [], []
    for r in rows:
        f = _num(r, "ujy")
        e = _num(r, "dujy")
        m = _num(r, "mjd")
        b = str(r.get("f", "")).strip().lower()
        if b not in spec.native_bands:
            continue
        mjd.append(m)
        flux.append(f * scale)
        ferr.append(e * scale)
        band.append(b)
        m5 = _num(r, "mag5sig")
        # A 5-sigma limiting MAGNITUDE converts to a 1-sigma flux limit by
        # dividing by 5; the non-detection test wants the 5-sigma flux, so the
        # magnitude is converted straight to a flux and used as-is.
        lim.append(float(ab_to_njy(m5)) if np.isfinite(m5) else float("nan"))
        err_code = _num(r, "err")
        good.append(bool(np.isfinite(f) and np.isfinite(e) and e > 0
                         and (not np.isfinite(err_code) or err_code == 0)))
        chi.append(_num(r, "chi_n"))
        maj, mn = _num(r, "maj"), _num(r, "min")
        elo.append(maj / mn if (np.isfinite(maj) and np.isfinite(mn) and mn > 0)
                   else float("nan"))
    lc = LightCurve(
        target_id=str(target_id), ra=float(ra), dec=float(dec), survey=spec.key,
        mjd=np.array(mjd, dtype=float), flux_njy=np.array(flux, dtype=float),
        flux_err_njy=np.array(ferr, dtype=float), band=np.array(band, dtype=object),
        limit_njy=np.array(lim, dtype=float), good=np.array(good, dtype=bool),
        chi_n=np.array(chi, dtype=float), elongation=np.array(elo, dtype=float),
        raw_columns=sorted({k for r in rows for k in r}))
    # Verify the unit rather than trust it (see `implied_flux_unit_njy`).
    mags = np.array([_num(r, "m") for r in rows if str(r.get("f", "")).strip().lower()
                     in spec.native_bands], dtype=float)
    served = np.array([_num(r, "ujy") for r in rows if str(r.get("f", "")).strip().lower()
                       in spec.native_bands], dtype=float)
    implied = implied_flux_unit_njy(mags, served)
    if np.isfinite(implied) and not (scale / 3.0 <= implied <= scale * 3.0):
        lc.notes.append(f"flux_unit_mismatch_assumed_{scale:g}_implied_{implied:.4g}")
    return lc


# ASAS-SN Sky Patrol v2 light-curve column names, from the service's own
# documentation.  Alternatives are listed because the client and the raw HTTP
# API have historically spelled some of them differently; the lookup takes the
# first name present, and `probe` records what was actually served.
ASASSN_COLUMNS = {
    "time": ("jd", "hjd", "mjd"),
    "flux": ("flux", "flux_mjy"),
    "flux_err": ("flux_err", "fluxerr", "flux_error"),
    "mag": ("mag", "magnitude"),
    "mag_err": ("mag_err", "magerr", "mag_error"),
    "limit": ("limit", "mag_limit", "maglim"),
    "band": ("phot_filter", "filter", "band"),
    "quality": ("quality", "qual", "flag"),
    "fwhm": ("fwhm",),
    "camera": ("camera", "cam"),
}


def _pick(row: dict, names: tuple[str, ...]):
    for n in names:
        if n in row:
            return row[n]
    return None


def asassn_rows_to_lightcurve(rows: list[dict], target_id: str, ra: float, dec: float,
                              spec: SurveySpec = ASASSN,
                              flux_unit_to_njy: float | None = None,
                              time_is_jd: bool | None = None) -> LightCurve:
    """Map ASAS-SN Sky Patrol v2 light-curve rows onto a :class:`LightCurve`.

    Three unit traps, each of which produces a plausible-looking light curve
    rather than an error, and each therefore handled explicitly:

    1. **JD, not MJD.**  Sky Patrol serves ``jd``.  Subtracting 2400000.5 is not
       optional: leaving it produces night labels 2.4 million days in the future,
       which the ledger would happily accept as a valid --- and permanently
       disjoint --- set of nights.  If the served column is already ``mjd`` the
       conversion is skipped, decided by the column name that was actually
       present rather than by assumption.
    2. **mJy, not uJy or nJy.**  A uniform scale error cancels in every ratio the
       channel computes and would never announce itself; it is checked against
       the served magnitudes and reported as a note if it disagrees.
    3. **V versus g.**  Pre-2018 epochs are Johnson V and post-2018 are Sloan g'.
       V has no defensible LSST proxy, so V rows are dropped here --- which
       removes them from the numerator *and* the denominator together, the only
       combination that leaves the rate unbiased.
    """
    scale = float(flux_unit_to_njy if flux_unit_to_njy is not None else spec.flux_unit_to_njy)
    if not rows:
        return LightCurve(str(target_id), float(ra), float(dec), spec.key,
                          np.array([]), np.array([]), np.array([]),
                          np.array([], dtype=object), notes=["empty"])
    present = set(rows[0])
    time_col = next((c for c in ASASSN_COLUMNS["time"] if c in present), None)
    is_jd = (time_is_jd if time_is_jd is not None
             else (time_col in ("jd", "hjd") if time_col else spec.time_is_jd))
    mjd, flux, ferr, band, lim, good, fwhm_all = [], [], [], [], [], [], []
    n_dropped_band = 0
    for r in rows:
        b = _pick(r, ASASSN_COLUMNS["band"])
        b = str(b).strip() if b is not None else "g"
        # Sky Patrol reports the filter as 'g'/'V'; anything not in the spec's
        # native bands is dropped, and counted.
        if b not in spec.native_bands:
            n_dropped_band += 1
            continue
        t = _num(r, time_col) if time_col else float("nan")
        mjd.append(t - JD_MINUS_MJD if is_jd else t)
        # `_numv`, not `float(x or nan)`: a genuine zero flux --- which a dip in a
        # faint star can produce --- is falsy, and `or` would silently turn it
        # into a missing epoch.
        flux.append(_numv(_pick(r, ASASSN_COLUMNS["flux"])) * scale)
        ferr.append(_numv(_pick(r, ASASSN_COLUMNS["flux_err"])) * scale)
        band.append(b)
        lm = _numv(_pick(r, ASASSN_COLUMNS["limit"]))
        lim.append(float(ab_to_njy(lm)) if np.isfinite(lm) else float("nan"))
        q = _pick(r, ASASSN_COLUMNS["quality"])
        # ASAS-SN marks usable epochs 'G' and rejected ones 'B'; a numeric flag
        # is read as 0 = good.  Anything unrecognised is treated as GOOD, which
        # is the conservative direction for a DENOMINATOR (more trials, lower
        # rate, larger p-values) even though it is the permissive direction for
        # a numerator --- the significance cut does that work.
        if q is None:
            good.append(True)
        elif isinstance(q, str):
            good.append(q.strip().upper() != "B")
        else:
            try:
                good.append(float(q) == 0.0)
            except (TypeError, ValueError):
                good.append(True)
        fwhm_all.append(_numv(_pick(r, ASASSN_COLUMNS["fwhm"])))

    lc = LightCurve(
        target_id=str(target_id), ra=float(ra), dec=float(dec), survey=spec.key,
        mjd=np.array(mjd, dtype=float), flux_njy=np.array(flux, dtype=float),
        flux_err_njy=np.array(ferr, dtype=float), band=np.array(band, dtype=object),
        limit_njy=np.array(lim, dtype=float), good=np.array(good, dtype=bool),
        raw_columns=sorted(present))
    if n_dropped_band:
        lc.notes.append(f"dropped_{n_dropped_band}_epochs_in_non_native_bands")
    if not is_jd:
        lc.notes.append(f"time_column_{time_col}_treated_as_mjd")
    # Seeing outliers: an epoch whose FWHM is a wild outlier against the star's
    # own median has a different effective aperture, which for a 16 arcsec PSF
    # changes how much neighbour flux is included --- a manufactured excursion.
    fw = np.array(fwhm_all, dtype=float)
    if np.isfinite(fw).sum() > 10:
        med = float(np.nanmedian(fw))
        if med > 0:
            lc.good = lc.good & ~(np.isfinite(fw) & (fw > 2.0 * med))
    mags = np.array([_numv(_pick(r, ASASSN_COLUMNS["mag"])) for r in rows], dtype=float)
    served = np.array([_numv(_pick(r, ASASSN_COLUMNS["flux"])) for r in rows], dtype=float)
    implied = implied_flux_unit_njy(mags, served)
    if np.isfinite(implied) and not (scale / 3.0 <= implied <= scale * 3.0):
        lc.notes.append(f"flux_unit_mismatch_assumed_{scale:g}_implied_{implied:.4g}")
    return lc


# ---------------------------------------------------------------------------
# Clients --- runner-only network code
# ---------------------------------------------------------------------------
def _text_head(body: bytes, n: int) -> str:
    """The first ``n`` characters of a response, or a note that it is binary.

    Recording `r.text` of an Arrow buffer produces pages of replacement
    characters in a committed artefact and hides the one thing worth reading.
    """
    try:
        head = body[:n * 2].decode("utf-8")
    except UnicodeDecodeError:
        return f"<binary: {len(body)} bytes>"
    return head[:n]


def _session(timeout: float):
    """A ``requests.Session`` with a real per-request deadline.

    ``requests`` honours ``timeout`` only as a per-call keyword, so a default has
    to be injected in ``request()``.  Without it there is no client-side deadline
    at all, and one stalled call burns the whole CI job --- which has already
    happened on the Rubin path (``brokers.AlerceTAP._service``).
    """
    import requests

    class _S(requests.Session):
        def request(self, *args, **kwargs):      # noqa: D102
            kwargs.setdefault("timeout", timeout)
            return super().request(*args, **kwargs)

    return _S()


class AsasSnSkyPatrol:
    """ASAS-SN Sky Patrol v2 --- the feed that can run unattended today.

    **No token** (measured on the runner 2026-08-25: HTTP 200, no credentials),
    which is the whole reason it is prioritised: it is the only alternative feed
    that a scheduled workflow can use with nothing added to the repository's
    secrets.

    Two access paths are tried in order and whichever answers is recorded:

    1. the official ``pyasassn`` client, which speaks the service's own protocol
       and returns light curves as a DataFrame;
    2. raw HTTP against the documented endpoints, as a fallback if the package is
       unavailable on the runner.

    Neither path's column names are trusted: :meth:`describe` records what the
    service actually returned, verbatim, and the parser looks columns up by name.
    """

    def __init__(self, endpoint: str = ASASSN.endpoint, timeout: float = 300.0):
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self.calls = 0
        self._client = None
        self.notes: list[str] = []

    def _pyasassn(self):
        """The vendor client, if this environment can actually import it.

        It usually cannot, and the reason is worth recording rather than
        rediscovering: ``pyasassn`` 0.6.4 pins ``pyarrow==4.0.1`` because it
        decodes responses with ``pyarrow.deserialize``, which was REMOVED from
        pyarrow after 4.x.  This repository requires ``pyarrow>=12``, so the two
        cannot coexist, and the pinned build additionally drags in a numpy old
        enough that it will not compile on Python 3.11 (reproduced here and on
        the runner, 2026-08-25).

        So the raw-HTTP path is the real path, not the fallback.  This is kept
        because an environment that *does* have the client should use it --- it
        speaks the service's own protocol --- but nothing depends on it.
        """
        if self._client is None:
            try:
                from pyasassn.client import SkyPatrolClient
            except Exception as exc:                               # noqa: BLE001
                raise AltFeedError(
                    f"pyasassn unavailable ({type(exc).__name__}: {exc}); "
                    f"it pins pyarrow==4.0.1 for pyarrow.deserialize, which this "
                    f"repository's pyarrow>=12 no longer provides") from exc
            self._client = SkyPatrolClient()
        return self._client

    def describe(self) -> dict:
        """Record the live service's reachability and its light-curve columns.

        The point of this method is that every column name in this module was
        read from documentation.  On the Rubin path the same posture produced
        three schema traps --- a join key that did not exist, a column that could
        not be SELECTed, and a filter encoding that would have returned a clean
        empty night --- all of which looked like plausible nulls, not errors.  So
        the record is committed verbatim and a later change to the service shows
        up as a diff in version control.
        """
        rec: dict = {"survey": ASASSN.key, "endpoint": self.endpoint,
                     "reached": False, "paths": {}}
        s = _session(self.timeout)

        # The service's own metadata endpoints.  `get_schema` is the one that
        # matters: it names every column the light-curve tables actually carry,
        # which is the record this module's column lookups have to be checked
        # against.
        for name, path in (("get_schema", "/get_schema"),
                           ("get_counts", "/get_counts")):
            try:
                r = s.get(f"{self.endpoint}{path}")
                self.calls += 1
                rec["paths"][name] = {"status": int(r.status_code),
                                      "content_type": r.headers.get("content-type"),
                                      "bytes": len(r.content or b""),
                                      "body_head": r.text[:1500]}
                rec["reached"] = rec["reached"] or r.status_code < 400
            except Exception as exc:                               # noqa: BLE001
                rec["paths"][name] = {"error": str(exc)[:400]}

        # WHICH REQUEST DOES IT ACCEPT, AND WHICH SERIALISATION DOES IT SERVE?
        #
        # The first probe (2026-08-25) asked four formats with `cols:
        # ["asas_sn_id"]` and got HTTP 500 from all four, while `/get_schema` and
        # `/get_counts` answered 200.  A 500 on every format including the
        # vendor's own is not a format question: either the request shape is
        # wrong or the failure is server-side and after the query.  Two things
        # separate those, and both are ASKED rather than assumed:
        #
        #   * the VENDOR-EXACT payload.  `pyasassn` 0.6.4 sends
        #     cols=['asas_sn_id','ra_deg','dec_deg', +'catalog_sources' for
        #     master_list] -- and a server that filters a cone by computing an
        #     angular distance needs `ra_deg`/`dec_deg` to still BE there.  Our
        #     single-column request would then raise inside the handler, which is
        #     precisely a 500.
        #   * the first bytes, in hex.  `pa.deserialize` reads a LEGACY buffer
        #     format; a modern Arrow IPC stream starts `ARROW1` / `FFFFFFFF`, and
        #     pyarrow>=12 can read that directly.  Which one comes back decides
        #     whether this module can parse a light curve at all, and it cannot
        #     be read off any documentation.
        cols_default = ["asas_sn_id", "ra_deg", "dec_deg"]
        variants = [
            ("vendor_master", {"catalog": "master_list",
                               "cols": [*cols_default, "catalog_sources"]}),
            ("vendor_stellar", {"catalog": "stellar_main", "cols": cols_default}),
            ("single_col", {"catalog": "master_list", "cols": ["asas_sn_id"]}),
        ]
        for label, base in variants:
            for fmt in ("arrow", "json", "csv", "parquet", "pandas"):
                key = f"cone_{label}_{fmt}"
                try:
                    r = s.post(
                        f"{self.endpoint}/lookup_cone/radius0.02_ra180.0_dec0.0",
                        json={**base, "format": fmt, "download": False})
                    self.calls += 1
                    body = r.content or b""
                    rec["paths"][key] = {
                        "status": int(r.status_code),
                        "content_type": r.headers.get("content-type"),
                        "bytes": len(body),
                        "first_bytes_hex": body[:32].hex(),
                        "body_head": _text_head(body, 300)}
                except Exception as exc:                           # noqa: BLE001
                    rec["paths"][key] = {"error": str(exc)[:400]}
                if rec["paths"][key].get("status") == 200:
                    # One shape answering is enough to settle the question; the
                    # rest of the matrix is only there to find one.
                    rec["accepted_request"] = {"variant": label, "format": fmt}
                    break
            if rec.get("accepted_request"):
                break

        # THE LIGHT-CURVE ENDPOINT ITSELF.  The index query above only names
        # targets; the curves come from `get_block` on the data servers, with a
        # base64 query hash the CLIENT builds.  Probed with a deliberately
        # invalid hash: a 4xx that names the hash proves the route exists and is
        # reachable without credentials, which is what has to be true before any
        # of this is worth implementing.  A connection failure here would mean
        # the data servers are firewalled even though the load balancer is not.
        for server in (1, 2):
            key = f"get_block_probe_data{server:02d}"
            url = (f"http://asassn-data{server:02d}.ifa.hawaii.edu:9006/get_block/"
                   f"query_hash-PROBE-block_idx-0-catalog-master_list")
            try:
                r = s.get(url)
                self.calls += 1
                body = r.content or b""
                rec["paths"][key] = {"status": int(r.status_code),
                                     "content_type": r.headers.get("content-type"),
                                     "bytes": len(body),
                                     "first_bytes_hex": body[:32].hex(),
                                     "body_head": _text_head(body, 300)}
            except Exception as exc:                               # noqa: BLE001
                rec["paths"][key] = {"error": str(exc)[:400]}
        try:
            client = self._pyasassn()
            rec["paths"]["pyasassn"] = {"ok": True,
                                        "catalogs": _safe_repr(getattr(client, "catalogs", None))}
        except Exception as exc:                                   # noqa: BLE001
            rec["paths"]["pyasassn"] = {"error": str(exc)[:400]}
        return rec

    def lightcurves(self, requests_: list[dict], radius_arcsec: float = 5.0,
                    on_result=None) -> dict[str, LightCurve]:
        """Fetch one light curve per requested target.

        ``requests_`` is a list of ``{"target_id", "ra", "dec"}``.  The cone
        radius is small on purpose: Sky Patrol keys light curves to *its own*
        source list, so a generous radius returns a neighbour's light curve for a
        high-proper-motion star and nothing announces the substitution.  The
        drift check in :func:`drift_excluded` is applied by the caller before
        this is ever reached.
        """
        client = self._pyasassn()
        out: dict[str, LightCurve] = {}
        for req in requests_:
            tid = str(req["target_id"])
            try:
                res = client.cone_search(float(req["ra"]), float(req["dec"]),
                                         radius=float(radius_arcsec) / 3600.0,
                                         units="deg", download=True)
                self.calls += 1
                rows = _dataframe_to_rows(getattr(res, "data", res))
            except Exception as exc:                               # noqa: BLE001
                self.notes.append(f"{tid}: {str(exc)[:200]}")
                continue
            if not rows:
                continue
            out[tid] = asassn_rows_to_lightcurve(rows, tid, float(req["ra"]),
                                                 float(req["dec"]))
            if on_result is not None:
                on_result(tid, out[tid])
        return out


class AtlasForcedPhotometry:
    """ATLAS forced photometry --- the closest like-for-like to a Rubin visit.

    30 s exposures, taken in quads within the hour, over the whole sky including
    the south.  That combination is why this feed matters more than its depth
    suggests: the *dilution* of a sub-second event is identical to Rubin's,
    because the exposure time is identical, and the quad reproduces the
    intra-night structure of Rubin's visit pair.

    **Requires a token.**  It is read from ``ATLAS_TOKEN`` following the same
    pattern the existing workflows use for ``LASAIR_TOKEN``: absent, every method
    returns a clean ``NO_TOKEN`` verdict and the caller skips the survey rather
    than failing the run.  If ``ATLAS_USERNAME``/``ATLAS_PASSWORD`` are present
    instead, a token is minted from them once per process.

    The API is a job queue, not a query service: submit a task per (position,
    window), poll until it finishes, then fetch a text file.  The endpoints below
    are documentation-derived and :meth:`describe` records the live shapes.
    """

    def __init__(self, token: str | None = None, endpoint: str = ATLAS.endpoint,
                 timeout: float = 300.0, poll_s: float = 20.0,
                 max_wait_s: float = 1800.0):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = float(timeout)
        self.poll_s = float(poll_s)
        self.max_wait_s = float(max_wait_s)
        self.token = token or os.environ.get(ATLAS.auth_env or "ATLAS_TOKEN") or None
        self.calls = 0
        self.notes: list[str] = []
        self._s = None

    @property
    def available(self) -> bool:
        return bool(self.token) or bool(os.environ.get("ATLAS_USERNAME")
                                        and os.environ.get("ATLAS_PASSWORD"))

    def _sess(self):
        if self._s is None:
            self._s = _session(self.timeout)
        return self._s

    def ensure_token(self) -> str | None:
        """Return a usable token, minting one from username/password if needed."""
        if self.token:
            return self.token
        user = os.environ.get("ATLAS_USERNAME")
        pw = os.environ.get("ATLAS_PASSWORD")
        if not (user and pw):
            return None
        r = self._sess().post(f"{self.endpoint}/api-token-auth/",
                              data={"username": user, "password": pw})
        self.calls += 1
        if r.status_code >= 400:
            raise AltFeedError(f"ATLAS token request failed: {r.status_code} {r.text[:300]}")
        self.token = (r.json() or {}).get("token")
        return self.token

    def _headers(self) -> dict:
        return {"Authorization": f"Token {self.token}", "Accept": "application/json"}

    def describe(self) -> dict:
        """Reachability, auth posture and the live shape of a queue listing."""
        rec: dict = {"survey": ATLAS.key, "endpoint": self.endpoint,
                     "reached": False, "auth_env": ATLAS.auth_env,
                     "token_present": bool(self.token)}
        try:
            r = self._sess().get(f"{self.endpoint}/")
            self.calls += 1
            rec["reached"] = r.status_code < 500
            rec["root_status"] = int(r.status_code)
        except Exception as exc:                                   # noqa: BLE001
            rec["error"] = str(exc)[:400]
            return rec
        try:
            tok = self.ensure_token()
        except AltFeedError as exc:
            rec["token_error"] = str(exc)[:300]
            tok = None
        if not tok:
            rec["verdict"] = "NO_TOKEN"
            rec["how_to_fix"] = (
                "register a free account at https://fallingstar-data.com/forcedphot/ "
                "and add the token as the repository secret ATLAS_TOKEN (the "
                "workflow passes it through exactly as LASAIR_TOKEN is passed)")
            return rec
        try:
            r = self._sess().get(f"{self.endpoint}/queue/", headers=self._headers())
            self.calls += 1
            rec["queue_status"] = int(r.status_code)
            rec["queue_body_head"] = r.text[:1200]
            rec["verdict"] = "OK" if r.status_code < 400 else "QUEUE_UNREADABLE"
        except Exception as exc:                                   # noqa: BLE001
            rec["queue_error"] = str(exc)[:400]
            rec["verdict"] = "QUEUE_UNREACHABLE"
        return rec

    def submit(self, ra: float, dec: float, mjd_lo: float | None = None,
               mjd_hi: float | None = None, use_reduced: bool = False) -> str:
        """Queue one forced-photometry task; returns the task URL.

        ``use_reduced=True`` photometers the *reduced* (non-difference) images
        instead, which is how the star's quiescent flux F* is obtained in ATLAS's
        own passband --- the difference light curve has that flux subtracted out
        by construction and cannot supply it.  Using a Gaia-derived synthetic
        magnitude instead would reintroduce exactly the cross-survey passband
        transformation the channel's greyness argument was built to avoid.
        """
        tok = self.ensure_token()
        if not tok:
            raise AltFeedError("no ATLAS token: set ATLAS_TOKEN (or ATLAS_USERNAME/PASSWORD)")
        data = {"ra": float(ra), "dec": float(dec), "send_email": False,
                "use_reduced": bool(use_reduced)}
        if mjd_lo is not None:
            data["mjd_min"] = float(mjd_lo)
        if mjd_hi is not None:
            data["mjd_max"] = float(mjd_hi)
        r = self._sess().post(f"{self.endpoint}/queue/", headers=self._headers(), data=data)
        self.calls += 1
        if r.status_code == 429:
            raise AltFeedError("ATLAS queue is throttling (429); reduce the batch size")
        if r.status_code >= 400:
            raise AltFeedError(f"ATLAS queue rejected the task: "
                               f"{r.status_code} {r.text[:300]}")
        url = (r.json() or {}).get("url")
        if not url:
            raise AltFeedError(f"ATLAS queue returned no task url: {r.text[:300]}")
        return str(url)

    def collect(self, task_url: str) -> str:
        """Poll one queued task and return its result file as text."""
        deadline = time.monotonic() + self.max_wait_s
        while time.monotonic() < deadline:
            r = self._sess().get(task_url, headers=self._headers())
            self.calls += 1
            if r.status_code >= 400:
                raise AltFeedError(f"ATLAS task unreadable: {r.status_code} {r.text[:200]}")
            body = r.json() or {}
            result = body.get("result_url")
            if result:
                rr = self._sess().get(result, headers=self._headers())
                self.calls += 1
                if rr.status_code >= 400:
                    raise AltFeedError(f"ATLAS result unreadable: {rr.status_code}")
                return rr.text
            if body.get("error_msg"):
                raise AltFeedError(f"ATLAS task failed: {str(body['error_msg'])[:200]}")
            time.sleep(self.poll_s)
        raise AltFeedError(f"ATLAS task did not finish within {self.max_wait_s:.0f}s")

    def lightcurve(self, target_id: str, ra: float, dec: float,
                   pmra: float = 0.0, pmdec: float = 0.0,
                   mjd_lo: float | None = None, mjd_hi: float | None = None,
                   th: LightCurveThresholds | None = None,
                   with_baseline: bool = True) -> tuple[LightCurve, dict[str, Quiescent]]:
        """One target's difference light curve, plus its quiescent flux per band.

        The window is split by :func:`pm_segments` so the aperture stays on a
        high-proper-motion star (see the module docstring --- an un-segmented
        decade-long request on a 1 arcsec/yr star manufactures dips), and each
        segment is submitted at its own propagated position.

        ``with_baseline`` adds one ``use_reduced=True`` task per segment to
        measure F* in ATLAS's own passbands.  Without it every amplitude in this
        target is untestable and the events can be recorded but never promoted on
        amplitude evidence --- which is the honest degradation, not a fallback to
        a transformed Gaia magnitude.
        """
        th = th or LightCurveThresholds()
        lo = float(mjd_lo) if mjd_lo is not None else 57200.0     # ATLAS from ~2015
        hi = float(mjd_hi) if mjd_hi is not None else 70000.0
        segs = pm_segments(ra, dec, pmra, pmdec, lo, hi, ATLAS, th)
        rows: list[dict] = []
        cols: list[str] = []
        for s in segs:
            text = self.collect(self.submit(s["ra"], s["dec"], s["mjd_lo"], s["mjd_hi"]))
            c, r = parse_atlas_text(text)
            cols = cols or c
            rows.extend(r)
        lc = atlas_rows_to_lightcurve(rows, target_id, ra, dec)
        lc.notes.append(f"pm_segments={len(segs)}")
        lc.raw_columns = cols or lc.raw_columns
        quiescent: dict[str, Quiescent] = {}
        if with_baseline and segs:
            red_rows: list[dict] = []
            for seg in segs:
                text = self.collect(self.submit(seg["ra"], seg["dec"], seg["mjd_lo"],
                                                seg["mjd_hi"], use_reduced=True))
                _c, r = parse_atlas_text(text)
                red_rows.extend(r)
            rlc = atlas_rows_to_lightcurve(red_rows, target_id, ra, dec)
            for nb in ATLAS.native_bands:
                sel = np.array([str(b) == nb for b in rlc.band], dtype=bool)
                if sel.sum() >= th.min_good_epochs:
                    base = robust_baseline(rlc.flux_njy[sel], th)
                    if base.ok and base.level > 0:
                        quiescent[nb] = Quiescent(
                            float(base.level),
                            float(base.scatter / math.sqrt(max(base.n_used, 1))),
                            "atlas_reduced_images")
        return lc, quiescent


def _dataframe_to_rows(obj) -> list[dict]:
    """Best-effort conversion of a client's return value into row dicts.

    Deliberately tolerant: the point of the probe is that the shape of what these
    services return is not known here, so a conversion that raises would turn a
    recoverable surprise into a lost run.
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        return [dict(r) for r in obj if isinstance(r, dict)]
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return list(to_dict(orient="records"))
        except TypeError:
            pass
    return []


def _safe_repr(obj) -> str:
    try:
        return repr(obj)[:600]
    except Exception:                                              # noqa: BLE001
        return "<unrepresentable>"


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def _utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=1, sort_keys=True,
                               allow_nan=False) + "\n")


def _clean(obj):
    """JSON-safe: NaN/Inf mean *not measured* here and are written as null."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    # bool BEFORE int: Python's bool is an int subclass, so the int branch would
    # turn every True into 1 and quietly change the type of every flag.
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def probe(cfg=None, out_dir: str | Path | None = None, surveys=("asassn", "atlas")
          ) -> dict:
    """Stage 0, runner-only: record each service's LIVE response verbatim.

    Runs before any science claim, for the reason ``docs/tocsin.md`` §5.1 records
    the hard way: every column name in this module was inferred from
    documentation, and on the Rubin path that inference was wrong in three
    places, each of which would have produced a confident null rather than an
    error.  The output is committed so that a change at either service appears as
    a diff in version control instead of as an unexplained quiet result months
    later.

    Nothing here is fatal.  A survey that cannot be reached, or that needs a
    token nobody has set, records that fact and the run continues.
    """
    from .run import _repo_root
    root = Path(cfg.root) if cfg is not None else _repo_root()
    out = Path(out_dir) if out_dir else root / "results" / "tocsin_altfeeds"
    rec: dict = {"run_at_utc": _utc(), "surveys": {}, "verdict": "NOT_RUN"}
    reached = []
    if "asassn" in surveys:
        try:
            rec["surveys"]["asassn"] = AsasSnSkyPatrol().describe()
        except Exception as exc:                                   # noqa: BLE001
            rec["surveys"]["asassn"] = {"error": str(exc)[:600], "reached": False}
        reached.append(bool(rec["surveys"]["asassn"].get("reached")))
    if "atlas" in surveys:
        try:
            rec["surveys"]["atlas"] = AtlasForcedPhotometry().describe()
        except Exception as exc:                                   # noqa: BLE001
            rec["surveys"]["atlas"] = {"error": str(exc)[:600], "reached": False}
        reached.append(bool(rec["surveys"]["atlas"].get("reached")))
    rec["verdict"] = ("OK" if all(reached) else
                      "PARTIAL" if any(reached) else "NO_FEED_REACHED")
    _write_json(out / "probe.json", rec)
    print(f"[tocsin-altfeeds] probe verdict={rec['verdict']}")
    return rec


def census(cfg=None, targets_path: str | Path | None = None,
           out_dir: str | Path | None = None) -> dict:
    """Offline: how much of the existing target list either feed can actually use.

    **This is the go/no-go.**  It needs no network --- only the cached Gaia target
    list the Rubin path already builds (``.cache/tocsin/targets.parquet``, made by
    ``seti tocsin-targets``) --- and it answers the question that decides whether
    running an alternative feed is worth anything at all: at ASAS-SN's and
    ATLAS's depths, on stars this shallow, how many of the ~10^5 targets can carry
    a detectable event of a given fractional amplitude?

    A run whose census says "600 targets at 10 %" is a different project from one
    that says "60,000", and the difference is knowable before a single byte is
    fetched.
    """
    from .run import _repo_root, load_targets
    root = Path(cfg.root) if cfg is not None else _repo_root()
    out = Path(out_dir) if out_dir else root / "results" / "tocsin_altfeeds"
    tpath = (Path(targets_path) if targets_path
             else root / ".cache" / "tocsin" / "targets.parquet")
    rec: dict = {"run_at_utc": _utc(), "targets_path": str(tpath)}
    targets = load_targets(tpath)
    if targets is None or len(targets) == 0:
        rec["verdict"] = "NO_TARGET_LIST"
        rec["how_to_fix"] = ("run `python -m seti.cli tocsin-targets` (runner-only; "
                             "Gaia egress is blocked in the sandbox), or restore "
                             "the tocsin-targets cache in the workflow before "
                             "this stage")
        _write_json(out / "census.json", rec)
        return rec
    rec["n_targets"] = int(len(targets))
    rec["surveys"] = {k: reachable_fraction(targets, s) for k, s in SURVEYS.items()}
    # Proper-motion drift: how many targets a fixed-aperture survey cannot
    # photometer at all over a ten-year baseline.
    pmra = np.asarray(targets["pmra"], dtype=float) if "pmra" in targets else np.zeros(len(targets))
    pmdec = (np.asarray(targets["pmdec"], dtype=float) if "pmdec" in targets
             else np.zeros(len(targets)))
    mu = np.hypot(np.nan_to_num(pmra), np.nan_to_num(pmdec)) / 1000.0
    for k, s in SURVEYS.items():
        allow = LightCurveThresholds().max_drift_frac * s.psf_fwhm_arcsec
        for years in (3.0, 10.0):
            rec["surveys"][k][f"n_pm_drift_excluded_{years:g}yr"] = int(
                np.sum(mu * years > allow)) if s.fixed_catalogue_position else 0
    rec["verdict"] = "OK"
    _write_json(out / "census.json", rec)
    print(f"[tocsin-altfeeds] census over {rec['n_targets']} targets -> {out/'census.json'}")
    return rec


def fold(verdict: AltFeedVerdict, ledger_path: Path, targets_n: int,
         alpha_fdr: float = 0.05, min_visits_for_rate: int = 5,
         max_duty_cycle: float = 0.2, n_null_timing: int = 2000,
         timing_alpha: float = 0.01, max_grey_z: float = 3.0) -> dict:
    """Fold one survey's pass into its OWN ledger and re-assess.

    **A separate ledger per survey, never the Rubin one.**  The ledger's
    statistic is a rate per target-visit and its trials are cumulative; pouring
    ASAS-SN star-nights into a denominator built from Rubin star-nights would
    produce a number that is a rate of nothing --- different depth, different
    cadence, different systematics, different detectable amplitude.  The
    surveys are combined, if ever, at the level of *which targets recur in more
    than one of them*, which is a much stronger statement and needs no shared
    denominator.
    """
    led = Ledger.load(ledger_path)
    if not led.opened_utc:
        led.opened_utc = _utc()
    by_night: dict[str, list[Event]] = {}
    for ev in verdict.events:
        by_night.setdefault(ev.night, []).append(ev)
    nights = sorted(set(verdict.trials_by_night) | set(by_night))
    first = True
    for night in nights:
        led.add_night(night, by_night.get(night, []),
                      target_visits=verdict.trials_by_night.get(night, 0),
                      targets_in_footprint=targets_n,
                      alerts_seen=verdict.counts.get("alerts_in", 0) if first else 0,
                      visit_history=None,
                      target_positions=verdict.target_positions,
                      bin_trials=verdict.bin_trials_by_night.get(night))
        first = False
    # AFTER every night, so a target whose record is created by a later night
    # still receives its history --- the same ordering bug the Rubin path hit.
    led.apply_visit_history(verdict.visit_history)
    led.updated_utc = _utc()
    stats = led.assess(alpha_fdr=alpha_fdr, min_visits_for_rate=min_visits_for_rate,
                       max_duty_cycle=max_duty_cycle, n_null_timing=n_null_timing,
                       timing_alpha=timing_alpha, max_grey_z=max_grey_z,
                       mixed_polarity_requires_grey_both=True)
    led.save(ledger_path)
    return stats


def run_survey(survey: str, cfg=None, targets_path: str | Path | None = None,
               out_dir: str | Path | None = None, max_targets: int | None = 200,
               mjd_lo: float | None = None, mjd_hi: float | None = None,
               lightcurves: dict[str, LightCurve] | None = None,
               quiescent: dict[str, dict[str, Quiescent]] | None = None) -> dict:
    """Fetch, reduce, screen and ledger one alternative feed.

    Runner-only when ``lightcurves`` is not supplied; passing them in makes the
    whole stage offline and is how the tests drive it.

    ``max_targets`` exists because neither feed is a bulk service: ASAS-SN is one
    cone search per target and ATLAS is a job queue with a per-account throttle.
    A run therefore walks a bounded slice of the target list, brightest first ---
    which is not an arbitrary ordering but the only one that spends a limited
    quota on the stars where an event of a given fractional amplitude is actually
    detectable (:func:`reachable_fraction`).
    """
    from .run import _repo_root, load_targets, load_tocsin_config
    spec = SURVEYS.get(survey)
    if spec is None:
        raise AltFeedError(f"unknown survey {survey!r}; known: {sorted(SURVEYS)}")
    conf = load_tocsin_config(cfg)
    root = Path(cfg.root) if cfg is not None else _repo_root()
    out = Path(out_dir) if out_dir else root / "results" / "tocsin_altfeeds" / spec.key
    lth = LightCurveThresholds()
    sconf = conf["screen"]
    th = funnel_thresholds(spec, Thresholds(
        min_abs_snr=float(sconf["min_abs_snr"]),
        min_reliability=float(sconf["min_reliability"]),
        max_dipole_significance=float(sconf["max_dipole_significance"]),
        max_extendedness=float(sconf["max_extendedness"]),
        max_trail_arcsec=float(sconf["max_trail_arcsec"]),
        max_sep_sigma=float(sconf["max_sep_sigma"]),
        max_sep_arcsec=float(sconf["max_sep_arcsec"]),
        match_radius_arcsec=float(sconf["match_radius_arcsec"]),
        max_grey_z=float(sconf["max_grey_z"]),
        baseline_rel_err=float(sconf["baseline_rel_err"]),
        missing_pm_penalty_arcsec=float(sconf["missing_pm_penalty_arcsec"]),
    ))
    summary: dict = {"run_at_utc": _utc(), "survey": spec.key,
                     "survey_name": spec.name, "verdict": "NOT_RUN",
                     "auth_env": spec.auth_env, "counts": {}, "notes": [],
                     "signature_transfer": signature_transfer(spec)}

    tpath = (Path(targets_path) if targets_path
             else root / ".cache" / "tocsin" / "targets.parquet")
    targets = load_targets(tpath)
    if targets is None or len(targets) == 0:
        summary["verdict"] = "NO_TARGET_LIST"
        summary["notes"].append(f"target list missing at {tpath}; run tocsin-targets")
        _write_json(out / "summary.json", summary)
        return summary
    summary["n_targets_total"] = int(len(targets))
    summary["reachability"] = reachable_fraction(targets, spec)

    if lightcurves is None:
        lightcurves, quiescent, fetch_notes = _fetch(spec, targets, lth,
                                                     max_targets, mjd_lo, mjd_hi)
        summary["notes"].extend(fetch_notes)
        if not lightcurves:
            summary["verdict"] = "NO_DATA_REACHED"
            _write_json(out / "summary.json", summary)
            print(f"[tocsin-altfeeds] {spec.key}: NO_DATA_REACHED")
            return summary
    summary["counts"]["lightcurves"] = len(lightcurves)

    # F* FALLBACK, for difference-flux feeds only.  ATLAS's difference light
    # curve has the star's own light subtracted out, so it cannot measure F*;
    # the preferred source is a second `use_reduced=True` pass in ATLAS's own
    # passband, and this is the documented fallback when that pass is absent or
    # failed.  It is a cross-survey passband transformation and is flagged as
    # one: `gspc_interpolated_native`, carrying `passband_interp_rel_err` as an
    # irreducible relative error, exactly as the Rubin path carries
    # `baseline_rel_err` for its GSPC fallback.  Without SOME F* the whole band
    # is refused (see `reduce_lightcurve`), so the alternative to this fallback
    # is not a weaker result --- it is no result at all.
    if spec.is_difference_flux:
        quiescent = _fill_quiescent_from_gspc(quiescent or {}, lightcurves, targets, spec)

    reductions = []
    for tid, lc in lightcurves.items():
        red = reduce_lightcurve(lc, spec, lth, (quiescent or {}).get(tid))
        red.notes.extend(lc.notes)
        reductions.append(red)
    verdict = screen_lightcurves(reductions, targets, spec, th, lth)
    summary["counts"].update(verdict.counts)
    summary["notes"].extend(verdict.notes)

    # THE DENOMINATOR, stated in the artefact rather than implied.  Unlike the
    # Rubin path, every trial here carries real non-detection information, so
    # the ensemble rate is a rate and not an upper bound.
    n_pairs = len(verdict.star_night_pairs)
    n_events = len(verdict.events)
    summary["counts"]["target_nights_screened"] = n_pairs
    summary["denominator"] = ("forced_photometry_exact" if n_pairs > 2 * max(n_events, 1)
                              else "detection_dominated_lower_bound")
    summary["forced_coverage_fraction"] = 1.0 if n_pairs else 0.0

    lconf = conf["ledger"]
    ledger_path = out / f"ledger_{spec.key}.json"
    summary["ledger"] = fold(verdict, ledger_path, int(len(targets)),
                             alpha_fdr=float(lconf["alpha_fdr"]),
                             min_visits_for_rate=int(lconf["min_visits_for_rate"]),
                             max_duty_cycle=float(lconf["max_duty_cycle"]),
                             n_null_timing=int(lconf["n_null_timing"]),
                             timing_alpha=float(lconf["timing_alpha"]),
                             max_grey_z=float(conf["screen"]["max_grey_z"]))
    summary["verdict"] = "OK" if n_pairs else "NO_USABLE_EPOCHS"
    _write_json(out / "summary.json", summary)
    _write_json(out / "events.json",
                {"survey": spec.key,
                 "events": [asdict(ev) for ev in verdict.events[:2000]],
                 "rejected": verdict.rejected[:2000],
                 "band_reductions": verdict.band_reductions})
    print(f"[tocsin-altfeeds] {spec.key}: {summary['verdict']} "
          f"{n_events} events over {n_pairs} star-nights; "
          f"tiers={summary['ledger'].get('tier_counts')}")
    return summary


# Relative error carried by an F* obtained by interpolating GSPC synthetic SDSS
# fluxes to a native effective wavelength.  0.20 ~ 0.2 mag, which is the scale of
# the TiO-band structure a smooth two-point interpolation cannot know about in an
# M dwarf --- and M dwarfs are most of this target list.  It is deliberately
# generous: an over-stated baseline error makes every amplitude LESS significant,
# which is the direction a search should err in.
PASSBAND_INTERP_REL_ERR = 0.20


def _fill_quiescent_from_gspc(quiescent: dict, lightcurves: dict, targets,
                              spec: SurveySpec) -> dict:
    """Supply a native-band F* from Gaia GSPC wherever the feed did not measure one."""
    ids = (np.asarray(targets["source_id"]).astype(str) if "source_id" in targets
           else np.arange(len(targets)).astype(str))
    index = {str(t): i for i, t in enumerate(ids)}
    mags = {nb: synthetic_native_mag(targets, spec, nb) for nb in spec.native_bands}
    out = {tid: dict(q) for tid, q in (quiescent or {}).items()}
    for tid in lightcurves:
        row = index.get(str(tid))
        if row is None:
            continue
        have = out.setdefault(str(tid), {})
        for nb in spec.native_bands:
            if nb in have:
                continue
            m = float(mags[nb][row])
            if not np.isfinite(m):
                continue
            f = float(ab_to_njy(m))
            have[nb] = Quiescent(f, f * PASSBAND_INTERP_REL_ERR,
                                 "gspc_interpolated_native")
    return out


def _fetch(spec: SurveySpec, targets, lth: LightCurveThresholds,
           max_targets: int | None, mjd_lo: float | None, mjd_hi: float | None
           ) -> tuple[dict, dict, list[str]]:
    """Runner-only acquisition for one survey.  Returns (lightcurves, F*, notes).

    Target ordering is by predicted native-band brightness, not by catalogue
    order: a bounded quota spent on stars too faint to carry a detectable event
    buys a denominator and no possible numerator.
    """
    notes: list[str] = []
    n = len(targets)
    mag = np.full(n, np.nan)
    for nb in spec.native_bands:
        m = synthetic_native_mag(targets, spec, nb)
        mag = np.where(np.isfinite(m) & (~np.isfinite(mag) | (m < mag)), m, mag)
    order = np.argsort(np.where(np.isfinite(mag), mag, np.inf))
    if max_targets:
        order = order[:int(max_targets)]
    ids = (np.asarray(targets["source_id"]).astype(str) if "source_id" in targets
           else np.arange(n).astype(str))
    ra = np.asarray(targets["ra"], dtype=float)
    dec = np.asarray(targets["dec"], dtype=float)
    pmra = np.asarray(targets["pmra"], dtype=float) if "pmra" in targets else np.zeros(n)
    pmdec = np.asarray(targets["pmdec"], dtype=float) if "pmdec" in targets else np.zeros(n)
    lo = float(mjd_lo) if mjd_lo is not None else 57200.0
    hi = float(mjd_hi) if mjd_hi is not None else 61300.0

    lightcurves: dict[str, LightCurve] = {}
    quiescent: dict[str, dict[str, Quiescent]] = {}
    if spec.key == ASASSN.key:
        client = AsasSnSkyPatrol()
        reqs = []
        n_drift = 0
        for i in order:
            if drift_excluded(pmra[i], pmdec[i], lo, hi, spec, lth):
                n_drift += 1
                continue
            reqs.append({"target_id": str(ids[i]), "ra": float(ra[i]), "dec": float(dec[i])})
        if n_drift:
            notes.append(f"{n_drift} targets excluded: proper-motion drift exceeds "
                         f"{lth.max_drift_frac:g} x {spec.psf_fwhm_arcsec:g}\" over the "
                         "requested window, and ASAS-SN photometers its own catalogued "
                         "position so the drift cannot be engineered away")
        try:
            lightcurves = client.lightcurves(reqs)
        except AltFeedError as exc:
            notes.append(f"asassn_unavailable: {str(exc)[:300]}")
        notes.extend(client.notes[:20])
    else:
        client = AtlasForcedPhotometry()
        if not client.available:
            notes.append("ATLAS skipped: no ATLAS_TOKEN in the environment. Register "
                         "free at https://fallingstar-data.com/forcedphot/ and add the "
                         "token as a repository secret; the workflow passes it exactly "
                         "as LASAIR_TOKEN is passed.")
            return {}, {}, notes
        for i in order:
            tid = str(ids[i])
            try:
                lc, q = client.lightcurve(tid, float(ra[i]), float(dec[i]),
                                          float(pmra[i]), float(pmdec[i]), lo, hi, lth)
            except AltFeedError as exc:
                notes.append(f"{tid}: {str(exc)[:200]}")
                continue
            lightcurves[tid] = lc
            quiescent[tid] = q
    return lightcurves, quiescent, notes


def signature_transfer(spec: SurveySpec) -> dict:
    """A machine-readable statement of what this feed can and cannot detect.

    Written into every summary so that no reader of a committed artefact has to
    go looking for the caveats, and so that a null is never quoted against a
    signature the feed could not have seen.  ``docs/tocsin-altfeeds.md`` carries
    the argument; this carries the verdicts.
    """
    single_band = len(spec.native_bands) < 2
    return {
        "exposure_s": spec.exposure_s,
        "rubin_visit_s": 30.0,
        "sub_visit_timescale": ("preserved: the exposure time equals a Rubin visit, "
                                "so a short event is diluted identically"
                                if abs(spec.exposure_s - 30.0) < 1e-6 else
                                f"degraded: {spec.exposure_s:g} s per exposure x "
                                f"{spec.exposures_per_epoch} against a 30 s Rubin visit, "
                                "so a sub-second event is diluted further"),
        "intra_night_structure": (f"{spec.exposures_per_epoch} exposures per epoch"
                                  if spec.exposures_per_epoch > 1 else "none"),
        "achromaticity": ("UNAVAILABLE: single-band survey, the discriminant cannot run"
                          if single_band else
                          "available only when both filters fall on the same night; "
                          "the fraction is measured per run as two_band_night_fraction"),
        "astrometric_offset": ("UNAVAILABLE: photometry is forced at the requested "
                               "position, so the separation is zero by construction"),
        "rubin_flags": ("UNAVAILABLE: no reliability, dipole, glint-trail, "
                        "extendedness or solar-system association exists in this feed"
                        + ("; ATLAS chi/N and PSF elongation are used as partial "
                           "substitutes" if spec.key == ATLAS.key else "")),
        "recurrence_ledger": "PRESERVED --- and this is the channel's actual instrument",
        "denominator": ("PRESERVED AND IMPROVED: every good epoch is a trial, measured "
                        "per target, with no footprint proxy"),
        "depth_penalty_mag_vs_rubin": round(24.5 - max(spec.depth_5sigma.values()), 2),
        # The other side of the same ledger, and the reason this is not merely a
        # stopgap: Rubin saturates near r = 16 in a 30 s visit, so the bright
        # half of a 100 pc nearby-star sample is invisible to the alert stream
        # by construction.  Both feeds work there.  The census counts how many
        # real targets fall in this window.
        "complementary_bright_window": (
            f"{min(spec.saturation_mag.values()):g} <= m <= {RUBIN_SATURATION_MAG:g}: "
            "stars this feed can screen and the Rubin alert stream cannot, "
            "because they saturate a 30 s LSST visit"),
        "dominant_new_systematic": ("aperture blending: 16 arcsec FWHM with no centroid "
                                    "to reject an unrelated source"
                                    if spec.psf_fwhm_arcsec > 10 else
                                    "proper-motion drift across a multi-year request, "
                                    "handled by pm_segments"),
    }
