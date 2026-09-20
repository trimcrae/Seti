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
                      require_synthetic: bool = True,
                      g_min: float | None = None) -> str:
    """ADQL for one parallax shell of the nearby-star target list.

    Chunking by parallax shell is the pattern that works on the runner: a single
    monolithic Gaia query at >10^6 rows times out (``docs/channel-brief.md`` §2).

    The declination window defaults to the Rubin main-survey footprint; targets
    outside it would inflate the trial count with stars that can never produce
    an alert.  ``require_synthetic`` joins the GSPC synthetic photometry, which
    is the baseline-flux source, but is left optional so a coverage run can
    measure how many targets lack it rather than assuming.

    ``g_min`` is the survey's BRIGHT limit: a star brighter than the detector
    saturates cannot produce a valid difference-image alert, only a residual,
    and it is excluded from the list for the same reason a star outside the
    footprint is -- it is not a trial.  ``None`` applies no bright cut.
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
    bright = f" AND g.phot_g_mean_mag > {float(g_min)}" if g_min is not None else ""
    return (
        f"{select} FROM gaiadr3.gaia_source AS g {join} "
        f"WHERE g.parallax >= {parallax_min_mas} AND g.parallax < {parallax_max_mas} "
        f"AND g.parallax_over_error > 10 "
        f"AND g.dec BETWEEN {dec_min} AND {dec_max} "
        f"AND g.phot_g_mean_mag < {g_max}{bright}"
    )


def drop_saturated(df, bright_limit_mag: float | None, bands: tuple[str, ...] = ("g", "r")
                   ) -> tuple[object, int]:
    """Remove targets brighter than ``bright_limit_mag`` in ANY of ``bands``.

    The ADQL bright cut above is on Gaia G, which for a red dwarf sits between
    its g and r; saturation is per band, so the list is cut again on the GSPC
    synthetic magnitude of each band the survey observes in.  A star whose
    synthetic magnitude is missing is kept (the G cut already applied) -- a cut
    on an absent value would silently empty the list.  Returns
    ``(frame, n_removed)``.
    """
    if bright_limit_mag is None or df is None or len(df) == 0:
        return df, 0
    import numpy as np
    lim = float(bright_limit_mag)
    bright = np.zeros(len(df), dtype=bool)
    for band in bands:
        col = GSPC_MAG_COLUMN.get(band)
        if col is None or col not in df:
            continue
        m = np.asarray(df[col], dtype=float)
        bright |= np.isfinite(m) & (m < lim)
    if "phot_g_mean_mag" in df:
        g = np.asarray(df["phot_g_mean_mag"], dtype=float)
        bright |= np.isfinite(g) & (g < lim)
    n = int(bright.sum())
    if n:
        df = df.loc[~bright].reset_index(drop=True)
    return df, n


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


# ---------------------------------------------------------------------------
# Bright neighbours: the saturated star NEXT to a faint target
# ---------------------------------------------------------------------------
# The saturation rule (`drop_saturated`, `saturated_target`) looks only at the
# target's own magnitude.  A saturated star's residual lands wherever its PSF,
# halo and bleed do --- including on a faint catalogued star a few arcsec away,
# which then alerts with its bright neighbour's defect at its own position and
# is judged against its own faint baseline.  This rule cuts on what else is
# inside the aperture.  (It was written on 2026-09-20 while vetting issue #15,
# whose star turned out to be the OTHER reference systematic, a
# high-proper-motion star that had left its own reference image --- see
# docs/tocsin-ztf.md 8d and `ztf_live.rebaseline_persistent_residuals`; this
# rule is kept because the mechanism it closes is real and the target's own
# magnitude cannot see it.)
#
# The exclusion radius scales with the neighbour's flux: a saturated star's
# residual, halo and bleed all grow with brightness.  At the survey's saturation
# magnitude the radius is `r0` (5", a couple of PSF widths plus the 1.5" match
# radius and the ~1" centroid jitter of a residual); every 5 magnitudes brighter
# multiplies it by 10, capped at `cap`.  So G 11.5 -> 10", G 8 -> 50", G 5 ->
# 120".  Positions are compared at the Gaia epoch; over a decade a nearby star
# moves at most a few arcsec, which is inside the floor.
def bright_neighbour_radius_arcsec(neighbour_g, saturation_mag: float,
                                   r0_arcsec: float = 5.0, cap_arcsec: float = 120.0):
    """Exclusion radius around a star of magnitude ``neighbour_g`` (0 if not saturated)."""
    g = np.asarray(neighbour_g, dtype=float)
    r = r0_arcsec * np.power(10.0, 0.2 * (float(saturation_mag) - g))
    r = np.minimum(r, float(cap_arcsec))
    return np.where(np.isfinite(g) & (g < float(saturation_mag)), r, 0.0)


def bright_star_adql(ra_lo: float, ra_hi: float, dec_min: float, dec_max: float,
                     g_max: float, max_rows: int = 2_000_000) -> str:
    """Every Gaia DR3 source brighter than ``g_max`` in one RA stripe of the footprint."""
    return (f"SELECT TOP {int(max_rows)} source_id, ra, dec, pmra, pmdec, phot_g_mean_mag "
            f"FROM gaiadr3.gaia_source WHERE phot_g_mean_mag < {float(g_max)} "
            f"AND ra >= {float(ra_lo)} AND ra < {float(ra_hi)} "
            f"AND dec BETWEEN {float(dec_min)} AND {float(dec_max)}")


def flag_bright_neighbours(targets, bright, saturation_mag: float,
                           r0_arcsec: float = 5.0, cap_arcsec: float = 120.0):
    """Targets that lie inside a brighter-than-saturation star's exclusion radius.

    ``targets`` and ``bright`` are DataFrame-likes with ``source_id, ra, dec,
    phot_g_mean_mag``.  Returns a DataFrame with one row per flagged target
    (the CLOSEST-in-radius-units offending neighbour): ``source_id, reason,
    neighbour_source_id, neighbour_g, sep_arcsec, radius_arcsec``.  A star is
    never its own neighbour.
    """
    import pandas as pd
    from scipy.spatial import cKDTree

    cols = ["source_id", "reason", "neighbour_source_id", "neighbour_g",
            "sep_arcsec", "radius_arcsec"]
    if targets is None or bright is None or len(targets) == 0 or len(bright) == 0:
        return pd.DataFrame(columns=cols)
    b_g = np.asarray(bright["phot_g_mean_mag"], dtype=float)
    b_r = bright_neighbour_radius_arcsec(b_g, saturation_mag, r0_arcsec, cap_arcsec)
    keep = b_r > 0
    if not keep.any():
        return pd.DataFrame(columns=cols)
    b_ra = np.asarray(bright["ra"], dtype=float)[keep]
    b_dec = np.asarray(bright["dec"], dtype=float)[keep]
    b_id = np.asarray(bright["source_id"]).astype(str)[keep]
    b_g, b_r = b_g[keep], b_r[keep]
    t_ra = np.asarray(targets["ra"], dtype=float)
    t_dec = np.asarray(targets["dec"], dtype=float)
    t_id = np.asarray(targets["source_id"]).astype(str)
    tree = cKDTree(_unit_vectors(b_ra, b_dec))
    chord = 2.0 * np.sin(np.radians(float(np.max(b_r)) / 3600.0) / 2.0)
    hits = tree.query_ball_point(_unit_vectors(t_ra, t_dec), r=chord)
    rows = []
    for i, js in enumerate(hits):
        best = None
        for j in js:
            if b_id[j] == t_id[i]:
                continue
            # exact separation from the chord length
            d = np.linalg.norm(_unit_vectors(t_ra[i], t_dec[i])[0]
                               - _unit_vectors(b_ra[j], b_dec[j])[0])
            sep = np.degrees(2.0 * np.arcsin(min(1.0, d / 2.0))) * 3600.0
            if sep <= b_r[j]:
                score = sep / b_r[j]
                if best is None or score < best[0]:
                    best = (score, j, sep)
        if best is not None:
            _s, j, sep = best
            rows.append({"source_id": t_id[i], "reason": "bright_neighbour",
                         "neighbour_source_id": b_id[j], "neighbour_g": float(b_g[j]),
                         "sep_arcsec": float(sep), "radius_arcsec": float(b_r[j])})
    return pd.DataFrame(rows, columns=cols)
