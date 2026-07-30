"""The anchored target list --- catalogued nearby stars, at the alert epoch.

Why the screen is anchored rather than blind
--------------------------------------------
Rubin issues of order 10^7 alerts a night and the overwhelming majority are
extragalactic transients, known variables, asteroids and subtraction residuals.
A blind anomaly hunt in that stream is a different (and heavily occupied)
problem.  S30 is specific: *an unclassified transient on a catalogued nearby
dwarf*.  Anchoring to a Gaia nearby-star list converts an unbounded anomaly
search into a bounded, statistically accountable one --- the number of trials is
``n_targets x n_visits``, which is knowable, so a p-value means something.

Two failure modes are designed out here
---------------------------------------
1. **Proper motion.**  Nearby stars are exactly the high-PM ones.  Gaia DR3
   positions are epoch 2016.0; a 1 arcsec/yr star has moved 10 arcsec by 2026.
   Matching a 2026 alert against an un-propagated 2016 position silently returns
   nothing --- the failure is a *null*, not an error, which is the most dangerous
   kind.  ``docs/channel-brief.md`` §2 records that this bug already cost this
   repository a whole run in another channel; here it is unit-tested.
2. **Baseline photometry is not free.**  Rubin *alerts* are world-public but the
   Rubin data releases (coadd catalogues, images) are data-rights restricted, so
   the quiescent flux ``F*`` that the fractional-amplitude statistic divides by
   cannot be taken from Rubin.  It comes from the Gaia DR3 synthetic photometry
   catalogue (GSPC, Gaia Collaboration/Montegriffo et al. 2023), which publishes
   standardised SDSS *ugriz* (+ PS1 *y*) synthetic magnitudes from the BP/RP
   spectra.  SDSS *griz* are close to but not identical with LSST *griz*, so the
   channel carries an explicit passband-mismatch systematic floor
   (``config/tocsin.yaml: baseline.passband_systematic``) and never claims
   greyness tighter than it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Gaia DR3 reference epoch (Julian year) for ra/dec.
GAIA_EPOCH = 2016.0

# GSPC synthetic band -> LSST alert band.  The u and y mappings are the loosest
# (SDSS u is bluer and much narrower than LSST u; PS1 y is used for LSST y), so
# `photometry` consumers should prefer g/r/i/z pairs when a choice exists.
# GSPC synthetic magnitude column for each LSST band.  The Gaia DR3 datamodel
# names these `<band>_<system>_mag` (g_sdss_mag, y_ps1_mag, ...) --- NOT
# `mag_g_sdss`.  Getting this backwards is not a loud failure: the JOIN simply
# errors, the acquisition falls back to no synthetic photometry, and every
# fractional amplitude downstream becomes untestable --- which silently disables
# the greyness test, i.e. this channel's core discriminator.  That is exactly
# what the first live run did.
GSPC_MAG_COLUMN = {
    "u": "u_sdss_mag",
    "g": "g_sdss_mag",
    "r": "r_sdss_mag",
    "i": "i_sdss_mag",
    "z": "z_sdss_mag",
    "y": "y_ps1_mag",
}

_MAS_PER_DEG = 3.6e6


def propagate_pm(ra, dec, pmra_mas_yr, pmdec_mas_yr,
                 from_epoch: float = GAIA_EPOCH, to_epoch: float = 2026.5):
    """Move positions from ``from_epoch`` to ``to_epoch`` (Julian years).

    ``pmra`` is the ``mu_alpha*`` convention (already includes ``cos(dec)``), as
    Gaia reports it.  Non-finite proper motions are treated as zero *and the
    caller must widen the match radius* --- silently dropping two-parameter
    sources would preferentially remove the faintest nearby dwarfs.

    Mirrors ``seti.vigil.acquire.propagate_pm`` deliberately: one convention for
    proper motion across the repository, because a sign or ``cos(dec)`` error
    here produces clean nulls rather than visible failures.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    pmra = np.nan_to_num(np.asarray(pmra_mas_yr, dtype=float))
    pmdec = np.nan_to_num(np.asarray(pmdec_mas_yr, dtype=float))
    dt = float(to_epoch - from_epoch)
    cosd = np.cos(np.radians(dec))
    cosd = np.where(np.abs(cosd) < 1e-6, 1e-6, cosd)
    ra_new = ra + (pmra * dt / _MAS_PER_DEG) / cosd
    dec_new = dec + pmdec * dt / _MAS_PER_DEG
    return ra_new % 360.0, dec_new


def position_uncertainty_arcsec(pmra_err_mas_yr, pmdec_err_mas_yr,
                                ra_err_mas=0.0, dec_err_mas=0.0,
                                dt_yr: float = 10.5,
                                pm_missing: np.ndarray | None = None,
                                missing_pm_penalty_arcsec: float = 2.0) -> np.ndarray:
    """1-sigma positional error at the alert epoch, in arcsec.

    Propagated proper-motion error dominates for nearby stars over a decade.
    Sources with no five-parameter solution get ``missing_pm_penalty_arcsec``
    instead of a fabricated small error, so they are matched loosely and flagged
    rather than dropped.
    """
    pmra_e = np.nan_to_num(np.asarray(pmra_err_mas_yr, dtype=float))
    pmdec_e = np.nan_to_num(np.asarray(pmdec_err_mas_yr, dtype=float))
    pos_e = np.hypot(np.nan_to_num(np.asarray(ra_err_mas, dtype=float)),
                     np.nan_to_num(np.asarray(dec_err_mas, dtype=float)))
    prop = np.hypot(pmra_e, pmdec_e) * abs(dt_yr)
    sigma = np.hypot(pos_e, prop) / 1000.0
    if pm_missing is not None:
        sigma = np.where(np.asarray(pm_missing, dtype=bool),
                         np.maximum(sigma, missing_pm_penalty_arcsec), sigma)
    return sigma


def _unit_vectors(ra_deg, dec_deg) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    cd = np.cos(dec)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)])


@dataclass
class MatchResult:
    """Alert-to-target association.

    ``sep_arcsec`` is the on-sky separation from the *propagated* target
    position; ``sep_sigma`` normalises it by the quadrature sum of the alert and
    propagated-target errors.  ``sep_sigma`` is the discriminator that matters:
    a real event on the star sits at the star, and Rubin astrometry is good
    enough that a 3-sigma offset is evidence against association.
    """

    alert_index: np.ndarray
    target_index: np.ndarray
    sep_arcsec: np.ndarray
    sep_sigma: np.ndarray
    n_alerts: int = 0
    n_targets: int = 0
    notes: list[str] = field(default_factory=list)


def match_alerts_to_targets(alert_ra, alert_dec, target_ra, target_dec,
                            radius_arcsec: float = 1.0,
                            alert_pos_err_arcsec=None,
                            target_pos_err_arcsec=None) -> MatchResult:
    """Positional cross-match on 3-D unit vectors (no RA-wrap pathology).

    Target positions must already be propagated to the alert epoch by
    :func:`propagate_pm`; this function deliberately does *not* do it, so a
    caller that forgets shows up as an obvious empty match in the tests rather
    than as a plausible-looking science result.

    The search radius is the *maximum* admitted separation; the returned
    ``sep_sigma`` is what the funnel actually cuts on.  Every alert may match at
    most one target (its nearest), because a duplicated alert would double-count
    a trial in the ledger.
    """
    a_ra = np.atleast_1d(np.asarray(alert_ra, dtype=float))
    a_dec = np.atleast_1d(np.asarray(alert_dec, dtype=float))
    t_ra = np.atleast_1d(np.asarray(target_ra, dtype=float))
    t_dec = np.atleast_1d(np.asarray(target_dec, dtype=float))
    empty = MatchResult(np.array([], dtype=int), np.array([], dtype=int),
                        np.array([]), np.array([]), a_ra.size, t_ra.size)
    if a_ra.size == 0 or t_ra.size == 0:
        empty.notes.append("empty_input")
        return empty

    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        raise

    tree = cKDTree(_unit_vectors(t_ra, t_dec))
    chord = 2.0 * np.sin(np.radians(radius_arcsec / 3600.0) / 2.0)
    dist, idx = tree.query(_unit_vectors(a_ra, a_dec), k=1,
                           distance_upper_bound=chord)
    hit = np.isfinite(dist) & (idx < t_ra.size)
    if not np.any(hit):
        empty.notes.append("no_positional_match")
        return empty
    ai = np.nonzero(hit)[0]
    ti = idx[hit].astype(int)
    # Chord length -> great-circle separation.
    sep_arcsec = np.degrees(2.0 * np.arcsin(np.clip(dist[hit] / 2.0, 0, 1))) * 3600.0

    sig_a = np.zeros(ai.size)
    if alert_pos_err_arcsec is not None:
        sig_a = np.atleast_1d(np.asarray(alert_pos_err_arcsec, dtype=float))[ai]
    sig_t = np.zeros(ti.size)
    if target_pos_err_arcsec is not None:
        sig_t = np.atleast_1d(np.asarray(target_pos_err_arcsec, dtype=float))[ti]
    sig = np.hypot(np.nan_to_num(sig_a), np.nan_to_num(sig_t))
    # A floor keeps sep_sigma finite when both catalogues claim perfect
    # astrometry; 50 mas is roughly Rubin's single-visit systematic floor.
    sig = np.where(sig > 0.05, sig, 0.05)
    return MatchResult(ai, ti, sep_arcsec, sep_arcsec / sig, a_ra.size, t_ra.size)


# ---------------------------------------------------------------------------
# Gaia acquisition (runner only --- the sandbox has no archive egress)
# ---------------------------------------------------------------------------
_TARGET_COLS = (
    "source_id", "ra", "dec", "ra_error", "dec_error", "parallax",
    "parallax_error", "pmra", "pmdec", "pmra_error", "pmdec_error",
    "phot_g_mean_mag", "bp_rp", "ruwe", "teff_gspphot",
    "phot_variable_flag", "non_single_star",
)


def build_target_adql(parallax_min_mas: float, parallax_max_mas: float,
                      dec_max: float = 15.0, dec_min: float = -90.0,
                      g_max: float = 21.0, max_rows: int = 500000,
                      require_synthetic: bool = True) -> str:
    """ADQL for one parallax shell of the nearby-star target list.

    Chunking by parallax shell is the pattern that works on the runner: a single
    monolithic Gaia query at >10^6 rows times out (``docs/channel-brief.md`` §2).

    The declination window defaults to the Rubin main-survey footprint; targets
    outside it would inflate the trial count with stars that can never produce
    an alert.  ``require_synthetic`` joins the GSPC synthetic photometry, which
    is the baseline-flux source, but is left optional so a coverage run can
    measure how many targets lack it rather than assuming.
    """
    cols = ", ".join(f"g.{c}" for c in _TARGET_COLS)
    if require_synthetic:
        syn = ", ".join(f"s.{c} AS {c}" for c in GSPC_MAG_COLUMN.values())
        join = ("JOIN gaiadr3.synthetic_photometry_gspc AS s "
                "ON s.source_id = g.source_id")
        select = f"SELECT TOP {max_rows} {cols}, {syn}"
    else:
        join = ""
        select = f"SELECT TOP {max_rows} {cols}"
    return (
        f"{select} FROM gaiadr3.gaia_source AS g {join} "
        f"WHERE g.parallax >= {parallax_min_mas} AND g.parallax < {parallax_max_mas} "
        f"AND g.parallax_over_error > 10 "
        f"AND g.dec BETWEEN {dec_min} AND {dec_max} "
        f"AND g.phot_g_mean_mag < {g_max}"
    )


def parallax_shells(d_max_pc: float, n_shells: int = 6) -> list[tuple[float, float]]:
    """Parallax shells (mas) covering ``d < d_max_pc``, roughly equal-count.

    Equal *volume* per shell keeps the row count per query roughly flat, which
    is what the archive timeout cares about.
    """
    plx_min = 1000.0 / float(d_max_pc)
    # Equal volume in distance -> shell edges at d_max * (k/n)^(1/3).
    edges_pc = [d_max_pc * ((k / n_shells) ** (1.0 / 3.0)) for k in range(1, n_shells + 1)]
    edges_plx = [1000.0 / d for d in edges_pc]
    shells = []
    hi = 1e6
    for plx in edges_plx:
        shells.append((max(plx, plx_min), hi))
        hi = max(plx, plx_min)
    return [(lo, hi) for lo, hi in shells if hi > lo]
