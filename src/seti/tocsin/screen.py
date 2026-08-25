"""The per-night funnel: alerts in, screened events out, every rejection named.

Order of operations is deliberate --- cheap structural cuts first, then quality
flags, then astrometric association, and only then the physics.  The funnel
counts at every stage are the channel's honesty record: a night that kept two
events out of nine million is only meaningful alongside *why* the other
8,999,998 went.

The rejection rules, and the systematic each one exists to kill
--------------------------------------------------------------
``low_significance``
    ``|dF| / sigma_dF`` below threshold.  Noise.
``low_reliability``
    The stream's own ML real/bogus score.  Cosmic rays, optical ghosts, cross-talk.
``dipole``
    A positive/negative pair from an astrometric mis-registration.  This is *the*
    systematic for a nearby-star sample: high-proper-motion stars are exactly the
    ones whose template position is wrong, so they subtract badly and alert
    spuriously.  Flagged dipoles go here; unflagged ones are caught downstream by
    the ledger's duty-cycle test, because a subtraction failure repeats at every
    visit while an event does not.
``extended``
    Resolved in the difference image: a host-galaxy transient or a nebular
    feature, not something on a 100 pc dwarf.
``trailed`` / ``solar_system``
    Streaked PSF, or association with a known minor planet.  An asteroid passing
    over a catalogued star is the most common way to get a "flash at a stellar
    position" that has nothing to do with the star.
``astrometric_offset``
    Position inconsistent with the proper-motion-propagated stellar position at
    more than ``max_sep_sigma``.  Rubin astrometry is good enough that this is a
    real discriminator, not a formality.
``chromatic``
    Two or more bands in the same night with fractional amplitudes that differ
    significantly: a stellar flare (flash mode) or reddening dust (dip mode).
    Grey events survive.  This is the test the ZTF glint channel could seldom
    apply and which killed all 15 of its candidates when it could.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ledger import Event
from .photometry import (
    ab_to_njy,
    blackbody_colour_temperature,
    fractional_amplitude,
    grey_excluded_by_nondetection,
    greyness_z,
)
from .schema import NormalizedAlert, validate
from .targets import (
    GSPC_MAG_COLUMN,
    match_alerts_to_targets,
    position_uncertainty_arcsec,
    propagate_pm,
)

# Gaia GSPC synthetic-photometry column for each LSST alert band.  SDSS ugriz
# stand in for LSST ugriz and PS1 y for LSST y; the resulting passband mismatch
# is carried as `baseline_rel_err`, never ignored.
BASELINE_COLUMN = dict(GSPC_MAG_COLUMN)

# Bands ordered blue -> red, so a "bluer band" is well defined when pairing.
BAND_ORDER = ("u", "g", "r", "i", "z", "y")


@dataclass
class Thresholds:
    """Every screening threshold in one place (loaded from ``config/tocsin.yaml``)."""

    # These defaults MUST mirror config/tocsin.yaml.  Two sources of truth
    # drifting apart is precisely how the Gaia column-name bug silently disabled
    # the greyness test, so they are pinned together by a test.
    min_abs_snr: float = 6.0
    min_reliability: float = 0.0        # the stream is already cut at 0.5 upstream
    require_reliability: bool = False
    max_dipole_significance: float = 3.0
    max_extendedness: float = 0.5
    max_trail_arcsec: float = 5.0       # backstop only; see config for why
    max_sep_sigma: float = 3.0
    max_sep_arcsec: float = 1.0
    match_radius_arcsec: float = 1.5
    max_grey_z: float = 3.0
    baseline_rel_err: float = 0.03
    missing_pm_penalty_arcsec: float = 2.0
    # How far above a band's detection limit the grey hypothesis must predict
    # before that band's silence counts as evidence.  Below it, untestable.
    nondetection_margin: float = 3.0


@dataclass
class ScreenVerdict:
    """Result of screening one night (or any batch) of alerts."""

    events: list[Event] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    visit_history: dict[str, list[float]] = field(default_factory=dict)
    target_positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    nights: list[str] = field(default_factory=list)

    def bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def reject(self, alert: NormalizedAlert, reason: str, **extra) -> None:
        self.rejected.append({"alert_id": alert.alert_id, "reason": reason,
                              "mjd": alert.mjd, "band": alert.band, **extra})
        self.bump(f"rejected_{reason}")


def mjd_to_jyear(mjd: float) -> float:
    """Julian year from MJD (for proper-motion propagation)."""
    return 2000.0 + (float(mjd) - 51544.5) / 365.25


def _baseline_flux(alert: NormalizedAlert, row, band: str, rel_err: float
                   ) -> tuple[float | None, float | None, str]:
    """Quiescent flux ``F*`` of the star in ``band`` (nJy), and where it came from.

    Priority matters for correctness, not just convenience:

    1. ``diaSource.templateFlux`` --- Rubin's own forced PSF flux on the coadd
       template, in the same band, same system, same pixels as the difference
       flux.  A ratio of two Rubin measurements has no passband transformation
       error at all, so the greyness test is limited only by photon noise.
    2. Gaia GSPC synthetic SDSS *ugriz* / PS1 *y*.  SDSS *griz* are close to but
       not identical with LSST *griz*, so this path carries the configured
       ``passband_systematic`` as an irreducible relative error and the channel
       never claims greyness tighter than it.
    3. Nothing --- the amplitude is marked untestable, and the event can be
       recorded but not promoted on amplitude evidence.
    """
    # THE TEMPLATE FLUX BELONGS TO THE ALERT'S OWN BAND, AND ONLY TO IT.
    # `alert.template_flux_njy` is Rubin's forced PSF flux on the coadd template
    # in `alert.band`; it says nothing about any other band.  Returning it for a
    # different `band` was a real bug with a quiet, one-sided failure mode: the
    # one-sided non-detection test asks what a grey event of amplitude `a` would
    # have produced in the band that stayed SILENT, and it was being handed the
    # DETECTED band's baseline.  On a red star those differ by a factor of a few,
    # so the test was under-powered when the detection was the bluer band and
    # over-powered when it was the redder one -- and this test rejects events
    # (docs/tocsin.md 3.2 records five), so an over-powered version discards real
    # candidates without leaving any trace that it did.
    tf, tfe = alert.template_flux_njy, alert.template_flux_err_njy
    if band == alert.band and tf is not None and np.isfinite(tf) and tf > 0:
        err = tfe if (tfe is not None and np.isfinite(tfe) and tfe > 0) else 0.0
        return float(tf), float(err), "rubin_template"
    col = BASELINE_COLUMN.get(band)
    if col is None:
        return None, None, "unknown_band"
    try:
        mag = float(row[col])
    except (KeyError, TypeError, ValueError):
        return None, None, "no_gaia_synthetic"
    if not np.isfinite(mag):
        return None, None, "no_gaia_synthetic"
    f = float(ab_to_njy(mag))
    return f, f * float(rel_err), "gaia_gspc_synthetic"


def _per_alert_flags(alert: NormalizedAlert, th: Thresholds) -> str | None:
    """First failing quality rule, or ``None`` if the alert passes them all."""
    snr = alert.snr
    if snr is None and alert.dflux_err_njy > 0:
        snr = alert.dflux_njy / alert.dflux_err_njy
    if snr is None or not np.isfinite(snr) or abs(snr) < th.min_abs_snr:
        return "low_significance"
    if alert.pixel_flag_bad:
        return "bad_pixels"
    # A satellite glint is itself a brief achromatic specular reflection --- the
    # exact mimic of this channel's flash signature --- so Rubin's own glint-trail
    # flag is treated as fatal, not advisory.  Untrailed single-point glints
    # survive this and are killed instead by the ledger's recurrence requirement:
    # a satellite does not return to the same catalogued star.
    if alert.glint_trail:
        return "glint_trail"
    if alert.reliability is None:
        if th.require_reliability:
            return "reliability_unavailable"
    elif alert.reliability < th.min_reliability:
        return "low_reliability"
    if alert.is_dipole:
        return "dipole"
    if (alert.dipole_significance is not None
            and np.isfinite(alert.dipole_significance)
            and alert.dipole_significance > th.max_dipole_significance):
        return "dipole"
    if (alert.extendedness is not None and np.isfinite(alert.extendedness)
            and alert.extendedness > th.max_extendedness):
        return "extended"
    # Rubin's own trail flag is the trustworthy signal.  The fitted
    # `trailLength` is NOT: a probe sample showed 1.38" on an ordinary,
    # unflagged, point-like star (extendedness 0.02), so the fit returns
    # non-trivial lengths for sources that are not trailed at all.  A tight cut
    # on the bare length would therefore throw away real events.  The bare
    # length is kept only as a backstop at a deliberately loose threshold; the
    # primary mover defences are the `sid` filter (which excludes known
    # solar-system objects) and Rubin's own upstream deletion of trails longer
    # than 10 deg/day.
    if alert.raw.get("trail_flag") is True:
        return "trailed"
    if (alert.trail_length_arcsec is not None
            and np.isfinite(alert.trail_length_arcsec)
            and alert.trail_length_arcsec > th.max_trail_arcsec):
        return "trailed"
    if alert.ss_object_id:
        return "solar_system"
    return None


def screen_alerts(alerts: list[NormalizedAlert], targets, th: Thresholds | None = None,
                  epoch_jyear: float | None = None,
                  observed_bands: dict | None = None,
                  band_limits: dict | None = None) -> ScreenVerdict:
    """Screen a batch of normalised alerts against the nearby-star target list.

    ``targets`` is a DataFrame-like with at least ``source_id, ra, dec`` and,
    where available, ``pmra, pmdec, pmra_error, pmdec_error`` and the GSPC
    baseline magnitude columns.  Target positions are propagated internally to
    the mean epoch of ``alerts``.

    Returns a :class:`ScreenVerdict` whose ``events`` are ready to fold into the
    ledger.  Nothing here promotes a candidate: promotion needs cross-night
    state, and that decision lives in ``ledger.assess``.
    """
    th = th or Thresholds()
    v = ScreenVerdict()
    v.bump("alerts_in", len(alerts))
    if not len(alerts):
        v.notes.append("no_alerts_in")
        return v
    if targets is None or len(targets) == 0:
        v.notes.append("no_targets")
        return v

    # -- 0. structural validation ------------------------------------------
    usable: list[NormalizedAlert] = []
    for a in alerts:
        problems = validate(a)
        if problems:
            v.reject(a, "malformed", detail=";".join(problems))
            continue
        usable.append(a)
    v.bump("structurally_valid", len(usable))
    if not usable:
        v.notes.append("all_alerts_malformed")
        return v
    v.nights = sorted({a.night for a in usable})

    # -- 1. quality flags ---------------------------------------------------
    passed: list[NormalizedAlert] = []
    for a in usable:
        why = _per_alert_flags(a, th)
        if why:
            v.reject(a, why)
        else:
            passed.append(a)
    v.bump("quality_passed", len(passed))
    if not passed:
        return v

    # -- 2. astrometric association to a catalogued nearby star -------------
    if epoch_jyear is None:
        epoch_jyear = mjd_to_jyear(float(np.median([a.mjd for a in passed])))
    t_ra = np.asarray(targets["ra"], dtype=float)
    t_dec = np.asarray(targets["dec"], dtype=float)
    pmra = np.asarray(targets["pmra"], dtype=float) if "pmra" in targets else np.zeros(t_ra.size)
    pmdec = (np.asarray(targets["pmdec"], dtype=float) if "pmdec" in targets
             else np.zeros(t_ra.size))
    pm_missing = ~(np.isfinite(pmra) & np.isfinite(pmdec))
    p_ra, p_dec = propagate_pm(t_ra, t_dec, pmra, pmdec, to_epoch=epoch_jyear)
    dt_yr = epoch_jyear - 2016.0
    t_sig = position_uncertainty_arcsec(
        targets["pmra_error"] if "pmra_error" in targets else np.zeros(t_ra.size),
        targets["pmdec_error"] if "pmdec_error" in targets else np.zeros(t_ra.size),
        targets["ra_error"] if "ra_error" in targets else np.zeros(t_ra.size),
        targets["dec_error"] if "dec_error" in targets else np.zeros(t_ra.size),
        dt_yr=dt_yr, pm_missing=pm_missing,
        missing_pm_penalty_arcsec=th.missing_pm_penalty_arcsec)

    m = match_alerts_to_targets(
        [a.ra for a in passed], [a.dec for a in passed], p_ra, p_dec,
        radius_arcsec=th.match_radius_arcsec,
        alert_pos_err_arcsec=[a.pos_err_arcsec for a in passed],
        target_pos_err_arcsec=t_sig)
    v.bump("positionally_matched", int(m.alert_index.size))
    v.notes.extend(m.notes)
    v.counts["epoch_jyear"] = round(float(epoch_jyear), 3)
    if m.alert_index.size == 0:
        return v

    ids = (np.asarray(targets["source_id"]).astype(str) if "source_id" in targets
           else np.arange(t_ra.size).astype(str))

    # -- 3. per-alert amplitude, grouped by (target, night) for the colour test
    groups: dict[tuple[str, str], list[dict]] = {}
    for ai, ti, sep, sig in zip(m.alert_index, m.target_index, m.sep_arcsec,
                                m.sep_sigma, strict=True):
        a = passed[int(ai)]
        if sig > th.max_sep_sigma or sep > th.max_sep_arcsec:
            v.reject(a, "astrometric_offset", sep_arcsec=float(sep),
                     sep_sigma=float(sig))
            continue
        row = targets.iloc[int(ti)] if hasattr(targets, "iloc") else targets[int(ti)]
        tid = str(ids[int(ti)])
        base, base_err, base_src = _baseline_flux(a, row, a.band, th.baseline_rel_err)
        amp = fractional_amplitude(a.dflux_njy, a.dflux_err_njy, base, base_err)
        groups.setdefault((tid, a.night), []).append(
            {"alert": a, "sep": float(sep), "sig": float(sig), "amp": amp,
             "baseline_source": base_src})
        v.target_positions[tid] = (float(p_ra[int(ti)]), float(p_dec[int(ti)]))
        if a.forced_mjds:
            hist = v.visit_history.setdefault(tid, [])
            hist.extend(float(x) for x in a.forced_mjds)
    v.bump("associated", sum(len(g) for g in groups.values()))

    # -- 4. the colour test, then emit events -------------------------------
    for (tid, night), rows in groups.items():
        # One event per (target, night, band): duplicate detections of the same
        # event in the same band are the same event, and counting them twice
        # would inflate both the multiplicity and the trial-corrected p-value.
        by_band: dict[str, dict] = {}
        for r in rows:
            b = r["alert"].band
            prev = by_band.get(b)
            if prev is None or abs(r["alert"].dflux_njy) > abs(prev["alert"].dflux_njy):
                by_band[b] = r
        polarity = {r["alert"].polarity for r in by_band.values()}
        if len(polarity) > 1:
            # Both signs in one night at one position is the signature of a
            # subtraction dipole or a mis-registered template, not of an event.
            for r in by_band.values():
                v.reject(r["alert"], "mixed_polarity_same_night")
            continue
        grey_z, grey_tested, chromatic = float("nan"), False, False
        bands_sorted = [b for b in BAND_ORDER if b in by_band]
        if len(bands_sorted) >= 2:
            blue, red = bands_sorted[0], bands_sorted[-1]
            ab, ar = by_band[blue]["amp"], by_band[red]["amp"]
            if ab.testable and ar.testable:
                grey_z = greyness_z(ab.a, ab.a_err, ar.a, ar.a_err)
                grey_tested = np.isfinite(grey_z)
                chromatic = grey_tested and abs(grey_z) > th.max_grey_z
        # ONE-SIDED COLOUR TEST.  Nearly every event is single-band (22 of 22 in
        # the first correct live run), so the two-band test almost never fires.
        # A band that was observed the same night and stayed silent is still
        # evidence: a grey event has equal FRACTIONAL amplitude in every band, so
        # on a red star it is brighter in absolute flux in the redder band, and
        # its absence there contradicts greyness.
        grey_excl = None
        if not grey_tested and len(bands_sorted) == 1 and observed_bands:
            seen_bands = set(observed_bands.get((tid, night), ()) or ())
            det_band = bands_sorted[0]
            r0 = by_band[det_band]
            for other in sorted(seen_bands - {det_band}):
                base_o, _e, _src = _baseline_flux(r0["alert"], row, other,
                                                  th.baseline_rel_err)
                lim_o = (band_limits or {}).get(other)
                gx = grey_excluded_by_nondetection(
                    r0["amp"].a, base_o, lim_o, other,
                    margin=th.nondetection_margin)
                if gx.tested:
                    grey_excl = gx
                    break
                if grey_excl is None:
                    # Remember why it was untestable, so the event records that
                    # the test was attempted rather than silently skipped.
                    grey_excl = gx
        if grey_excl is not None and grey_excl.excluded:
            for r in by_band.values():
                v.reject(r["alert"], "chromatic",
                         detail=f"grey_excluded_by_{grey_excl.other_band}_nondetection",
                         predicted_njy=round(grey_excl.predicted_flux_njy, 1),
                         limit_njy=round(grey_excl.limit_flux_njy, 1))
            continue

        ct = float("nan")
        if next(iter(polarity)) == "flash" and len(bands_sorted) >= 2:
            fit = blackbody_colour_temperature(
                bands_sorted, [by_band[b]["alert"].dflux_njy for b in bands_sorted],
                [by_band[b]["alert"].dflux_err_njy for b in bands_sorted])
            if fit.ok:
                ct = fit.temp_k
        if chromatic:
            for r in by_band.values():
                v.reject(r["alert"], "chromatic", grey_z=float(grey_z))
            continue

        # One event per star-night (see `ledger.Event`): the strongest band
        # carries the scalar summary, every band is kept in `per_band`.
        strongest = max(by_band.values(), key=lambda r: abs(r["alert"].dflux_njy))
        sa = strongest["alert"]
        reasons = []
        if not strongest["amp"].testable:
            reasons.append(f"amplitude_{strongest['amp'].reason}")
        if not grey_tested:
            reasons.append("greyness_untested_single_band" if len(bands_sorted) < 2
                           else "greyness_untestable")
        if sa.reliability is None:
            reasons.append("reliability_unavailable")
        if grey_excl is not None and grey_excl.tested and not grey_excl.excluded:
            reasons.append(f"grey_survives_{grey_excl.other_band}_nondetection")
        elif grey_excl is not None:
            reasons.append(f"nondetection_untestable_{grey_excl.reason}")
        if np.isnan(ct) and sa.polarity == "flash":
            reasons.append("colour_temperature_untestable")
        per_band = {
            b: {"mjd": float(by_band[b]["alert"].mjd),
                "dflux_njy": float(by_band[b]["alert"].dflux_njy),
                "dflux_err_njy": float(by_band[b]["alert"].dflux_err_njy),
                "a": float(by_band[b]["amp"].a),
                "a_err": float(by_band[b]["amp"].a_err),
                "baseline_source": by_band[b]["baseline_source"],
                "snr": by_band[b]["alert"].snr,
                "reliability": by_band[b]["alert"].reliability}
            for b in bands_sorted
        }
        v.events.append(Event(
            target_id=tid, night=night, mjd=float(sa.mjd), polarity=sa.polarity,
            bands=list(bands_sorted), dflux_njy=float(sa.dflux_njy),
            dflux_err_njy=float(sa.dflux_err_njy), strongest_band=sa.band,
            a=float(strongest["amp"].a), a_err=float(strongest["amp"].a_err),
            sep_arcsec=strongest["sep"], sep_sigma=strongest["sig"],
            grey_z=float(grey_z),
            grey_tested=bool(grey_tested or (grey_excl is not None and grey_excl.tested)),
            colour_temp_k=float(ct), verdict="kept",
            alert_ids=[by_band[b]["alert"].alert_id for b in bands_sorted],
            per_band=per_band, reasons=reasons))
    v.bump("events_kept", len(v.events))
    for tid, hist in v.visit_history.items():
        v.visit_history[tid] = sorted({round(float(x), 6) for x in hist})
    return v
