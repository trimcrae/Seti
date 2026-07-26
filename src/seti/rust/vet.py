"""The contamination gauntlet.  RUST lives or dies here.

A rising second moment is a *weak* signature: almost everything that can go
wrong with difference photometry over a decade goes wrong in the direction of
more scatter.  The funnel below is ordered so the cheapest and most lethal tests
run first, and every verdict is a pure function of already-fetched numbers so it
can be unit-tested with no network.

The mandatory gate is **two-band coincidence**.  The repository ledger states it
flatly --- a single-band ZTF anomaly is an artefact until confirmed in a second
band --- and for this channel it is also the *physics* test.  Debris crossing a
star blocks starlight geometrically, so the induced magnitude excursion is the
same in g and r; the amplitude-growth ratio is therefore ~1.  Every mundane
mechanism that grows variability is chromatic in a known direction:

    accretion / flares / spots  ->  g >> r  (blue)
    line-of-sight dust growth   ->  A_g/A_r ~ 1.42  (R_V = 3.1)
    grey macroscopic occulters  ->  ratio ~ 1.0     <-- the RUST signature
    blend with a red variable   ->  ratio < 1

So the ratio is not a sanity check bolted on at the end.  It is the measurement.
"""

from __future__ import annotations

import numpy as np

from .trend import RustStats

# ZTF g (4746 A) and r (6366 A).  For a Cardelli/Wang-&-Chen R_V = 3.1 law,
# A_g/A_V ~ 1.21 and A_r/A_V ~ 0.85, so ordinary dust grows the g amplitude
# ~1.42x faster than the r amplitude.  A geometric (grey) occulter gives 1.00.
EXTINCTION_G_OVER_R = 1.42
GRAY_RATIO_LO, GRAY_RATIO_HI = 0.80, 1.20
REDDENING_RATIO_LO, REDDENING_RATIO_HI = 1.25, 1.70

# ZTF usable photometric range.  Below the bright limit the detector is
# non-linear and saturating, above the faint limit the scatter is dominated by
# the survey's own depth --- which changed over the decade, so both extremes
# manufacture exactly the trend this channel looks for.
ZTF_BRIGHT_LIMIT = 13.5
ZTF_FAINT_LIMIT = {"g": 20.3, "r": 20.1}

# A neighbour this close and this comparable in brightness leaks its own
# variability into the target aperture.
CROWD_RADIUS_ARCSEC = 5.0
CROWD_DELTA_G = 2.5

# SIMBAD object types whose *known* physics already produces evolving aperiodic
# variability.  Superset of the dimming channel's list: AGN are added because
# stochastic optical variability with a red-noise amplitude that grows with
# timescale is their defining property, and a decade-long window samples longer
# timescales as it lengthens --- the single most seductive false positive here.
MUNDANE_OTYPES = (
    "YSO", "Or*", "TT*", "Ae*", "out", "FU*",           # young / disk dippers
    "CV*", "No*", "DN*", "AM*", "XB*", "LXB", "HXB",    # accreting / cataclysmic
    "EB*", "Al*", "bL*", "WU*", "EClB", "SB*",          # eclipsing / spectroscopic
    "Mi*", "LP*", "sr*", "RG*", "AGB", "C*", "S*",      # evolved dusty giants
    "pA*", "RC*", "WR*", "Be*", "Em*",                  # post-AGB, R CrB, emission
    "AGN", "QSO", "Sy1", "Sy2", "BLL", "Bla", "G", "GiG", "rG",  # extragalactic
)


def two_band_verdict(g: RustStats | None, r: RustStats | None,
                     sigma_min: float = 3.0, loo_min: float = 2.0) -> dict:
    """The mandatory g/r gate, and the achromaticity measurement.

    Requires a significant *rising* excess-variance trend in **both** bands, then
    classifies the amplitude-growth ratio against the extinction law.

    Returns a dict with ``verdict`` in:

    ``insufficient_bands``   one or both bands unusable -- not a candidate.
    ``single_band``          only one band rises significantly -- artefact.
    ``chromatic_blue``       ratio > 1.70: flare/accretion-like, not geometric.
    ``reddening_law``        ratio 1.25-1.70: growing line-of-sight dust column.
    ``achromatic_gray``      ratio 0.80-1.20: grey occulter -- the RUST signature.
    ``chromatic_red``        ratio < 0.80: r rises faster; blend with a red variable.
    """
    out: dict = {"verdict": "insufficient_bands", "amp_growth_ratio": float("nan"),
                 "g_slope_sigma": float("nan"), "r_slope_sigma": float("nan"),
                 "n_bands_rising": 0}
    if g is None or r is None:
        return out
    out["g_slope_sigma"] = float(g.slope_sigma)
    out["r_slope_sigma"] = float(r.slope_sigma)

    def _rising(s: RustStats) -> bool:
        return bool(s.slope_var_yr > 0 and s.slope_sigma >= sigma_min
                    and s.slope_sigma_loo_min >= loo_min)

    rising = [_rising(g), _rising(r)]
    out["n_bands_rising"] = int(sum(rising))
    if not any(rising):
        out["verdict"] = "insufficient_bands"
        return out
    if not all(rising):
        out["verdict"] = "single_band"
        return out

    # Growth in amplitude (mag) across the baseline, per band, per year --- the
    # quantity an occulter's geometry fixes and an extinction law reddens.
    dg = (g.amp_last_mmag - g.amp_first_mmag) / max(g.baseline_yr, 1e-3)
    dr = (r.amp_last_mmag - r.amp_first_mmag) / max(r.baseline_yr, 1e-3)
    if not (np.isfinite(dg) and np.isfinite(dr)) or dr <= 0:
        out["verdict"] = "single_band" if dr <= 0 else "insufficient_bands"
        return out
    ratio = float(dg / dr)
    out["amp_growth_ratio"] = ratio
    out["g_amp_growth_mmag_yr"] = float(dg)
    out["r_amp_growth_mmag_yr"] = float(dr)
    if ratio < GRAY_RATIO_LO:
        out["verdict"] = "chromatic_red"
    elif ratio <= GRAY_RATIO_HI:
        out["verdict"] = "achromatic_gray"
    elif ratio <= REDDENING_RATIO_HI:
        out["verdict"] = "reddening_law"
    else:
        out["verdict"] = "chromatic_blue"
    return out


def periodic_fraction(time, mag, magerr=None, min_period_d: float = 0.05,
                      max_period_d: float = 300.0) -> float:
    """Fraction of the variance explained by the best single sinusoid.

    RUST claims *aperiodic* variability.  A pulsator whose amplitude is growing,
    or a spotted rotator whose active region is, produces a rising second moment
    for entirely stellar reasons --- and is periodic.  High returned power means
    "coherent", which means "not this channel".  Returns 0 if ``astropy`` is
    unavailable rather than silently passing the star.
    """
    try:
        from astropy.timeseries import LombScargle
    except Exception:                                   # noqa: BLE001
        return 0.0
    t = np.asarray(time, float)
    m = np.asarray(mag, float)
    ok = np.isfinite(t) & np.isfinite(m)
    t, m = t[ok], m[ok]
    if t.size < 30 or np.ptp(t) <= 0:
        return 0.0
    try:
        freq = np.linspace(1.0 / max_period_d, 1.0 / min_period_d, 20000)
        power = LombScargle(t, m).power(freq, normalization="standard")
        return float(np.nanmax(power))
    except Exception:                                   # noqa: BLE001
        return 0.0


def crowding_verdict(n_neighbors: float | None, brightest_dg: float | None) -> str:
    """Blending census verdict from the Gaia neighbour list inside the PSF."""
    if n_neighbors is None or not np.isfinite(float(n_neighbors)):
        return "crowding_unknown"
    if int(n_neighbors) == 0:
        return "isolated"
    dg = float(brightest_dg) if brightest_dg is not None else np.nan
    if np.isfinite(dg) and dg <= CROWD_DELTA_G:
        return "blended"
    return "faint_neighbors"


def photometric_range_verdict(mag_med: float, band: str) -> str:
    """Reject stars sitting on ZTF's bright or faint wall."""
    m = float(mag_med)
    if not np.isfinite(m):
        return "range_unknown"
    if m < ZTF_BRIGHT_LIMIT:
        return "saturated"
    if m > ZTF_FAINT_LIMIT.get(band, 20.1):
        return "near_faint_limit"
    return "in_range"


def known_class_verdict(otype: str | None) -> str:
    """SIMBAD classification verdict --- a known variable class explains itself."""
    o = str(otype or "").strip()
    if not o:
        return "unclassified"
    return "known_variable" if any(tok.lower() in o.lower()
                                   for tok in MUNDANE_OTYPES) else "unclassified"


def gaia_quality_verdict(ruwe: float | None, non_single_star: int | None,
                         excess_noise: float | None = None) -> str:
    """Astrometric-quality verdict: an unresolved companion is a blend in disguise."""
    r = float(ruwe) if ruwe is not None else np.nan
    if np.isfinite(r) and r > 1.4:
        return "high_ruwe_binary"
    if non_single_star:
        return "gaia_non_single_star"
    if excess_noise is not None and np.isfinite(float(excess_noise)) \
            and float(excess_noise) > 1.0:
        return "astrometric_excess_noise"
    return "astrometry_clean"


def ir_dust_production_verdict(neowise: dict | None, sig_min: float = 2.0) -> str:
    """What the mid-IR does while the optical scatter grows.

    Note the **inverted logic** relative to :mod:`seti.dimming`.  There, a mid-IR
    brightening killed a candidate (absorbed starlight reappearing as thermal
    dust emission = an ordinary enshrouding event).  Here, a collisional cascade
    grinding a swarm to fragments *should* produce dust and *should* brighten
    W1/W2, so a brightening is **corroboration**, not a veto.

    Its absence is informative rather than fatal: NEOWISE W1/W2 (3.4/4.6 um)
    only probe material hotter than ~600-850 K, so a cascade at >1 AU can be
    real and mid-IR-silent.  The verdict says which case we are in; it does not
    pretend the non-detection settles anything.
    """
    if not neowise:
        return "insufficient_ir"
    slopes = [(neowise.get(f"{b}_slope_mag_yr"), neowise.get(f"{b}_slope_sigma"))
              for b in ("w1", "w2")]
    slopes = [(s, g) for s, g in slopes if s is not None and g is not None]
    if not slopes:
        return "insufficient_ir"
    if any(s < 0 and g >= sig_min for s, g in slopes):
        return "ir_brightens_dust_production"
    if any(s > 0 and g >= sig_min for s, g in slopes):
        return "ir_fades_with_optical"
    return "ir_flat_no_warm_dust"


def rust_verdict(row: dict) -> str:
    """Single combined verdict for one candidate.

    Ordered so the most decisive rejection wins.  ``clean_gray`` is the only
    label that means "this survived everything"; ``clean_reddening`` survived the
    artefact tests but its colour dependence says growing dust, which is a real
    and interesting --- but astrophysically ordinary --- result.
    """
    band = row.get("two_band_verdict", "insufficient_bands")
    if band in ("insufficient_bands", "single_band"):
        return band
    for key, bad in (
        ("range_verdict_g", ("saturated", "near_faint_limit")),
        ("range_verdict_r", ("saturated", "near_faint_limit")),
        ("crowding_verdict", ("blended",)),
        ("class_verdict", ("known_variable",)),
        ("gaia_verdict", ("high_ruwe_binary", "gaia_non_single_star",
                          "astrometric_excess_noise")),
    ):
        v = row.get(key)
        if v in bad:
            return str(v)
    if float(row.get("periodic_power", 0.0) or 0.0) >= float(row.get("periodic_max", 0.35)):
        return "periodic_variable"
    if band == "chromatic_blue":
        return "chromatic_blue"
    if band == "chromatic_red":
        return "chromatic_red"
    if band == "reddening_law":
        return "clean_reddening"
    return "clean_gray"


def vet_row(stats_g: RustStats | None, stats_r: RustStats | None,
            context: dict | None = None, periodic_power: float = 0.0,
            periodic_max: float = 0.35) -> dict:
    """Run the whole gauntlet on one source and return every intermediate verdict."""
    ctx = dict(context or {})
    tb = two_band_verdict(stats_g, stats_r)
    row: dict = {
        "two_band_verdict": tb["verdict"],
        "amp_growth_ratio": tb["amp_growth_ratio"],
        "g_amp_growth_mmag_yr": tb.get("g_amp_growth_mmag_yr", float("nan")),
        "r_amp_growth_mmag_yr": tb.get("r_amp_growth_mmag_yr", float("nan")),
        "n_bands_rising": tb["n_bands_rising"],
        "range_verdict_g": photometric_range_verdict(
            stats_g.mag_med if stats_g else float("nan"), "g"),
        "range_verdict_r": photometric_range_verdict(
            stats_r.mag_med if stats_r else float("nan"), "r"),
        "crowding_verdict": crowding_verdict(ctx.get("n_neighbors_5as"),
                                             ctx.get("brightest_neighbor_dg")),
        "class_verdict": known_class_verdict(ctx.get("simbad_otype")),
        "gaia_verdict": gaia_quality_verdict(ctx.get("ruwe"),
                                             ctx.get("non_single_star"),
                                             ctx.get("astrometric_excess_noise")),
        "ir_verdict": ir_dust_production_verdict(ctx.get("neowise")),
        "periodic_power": float(periodic_power),
        "periodic_max": float(periodic_max),
    }
    row["rust_verdict"] = rust_verdict(row)
    return row


__all__ = [
    "CROWD_DELTA_G", "CROWD_RADIUS_ARCSEC", "EXTINCTION_G_OVER_R",
    "GRAY_RATIO_HI", "GRAY_RATIO_LO", "MUNDANE_OTYPES", "REDDENING_RATIO_HI",
    "REDDENING_RATIO_LO", "ZTF_BRIGHT_LIMIT", "ZTF_FAINT_LIMIT",
    "crowding_verdict", "gaia_quality_verdict", "ir_dust_production_verdict",
    "known_class_verdict", "periodic_fraction", "photometric_range_verdict",
    "rust_verdict", "two_band_verdict", "vet_row",
]
