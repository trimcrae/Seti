"""SHROUD population decomposition — subtract the mundane before believing anything.

The overwhelming majority of "optically vanished, infrared present" sources are
ordinary.  This module assigns every object to an explicit class so the
population breakdown is a *result*, not a caveat.  Nothing reaches the energy
budget until it has survived the cascade.

Classes, in the order they are tested (first match wins)
--------------------------------------------------------
``PLATE_DEFECT``          no counterpart at any wavelength.  A photographic
                          emulsion artefact has no infrared source behind it,
                          so *requiring a real IR detection is itself a strong
                          artifact filter* — this is the class that filter
                          removes, and it is expected to dominate the control
                          sample (Hambly & Blair 2024).
``ASTEROID``              near-ecliptic, no counterpart: a moving object caught
                          on one plate.  Solano+2022 already removed 189 of
                          these; the residual is the incompleteness of that cut.
``MODERN_OPTICAL_MATCH``  a modern optical source of comparable brightness sits
                          at the position — the object never vanished.
``HIGH_PM_STAR``          a modern source propagates back onto the POSS-I
                          position.  It moved; it did not disappear.
``VARIABLE_STAR``         a modern optical counterpart exists but is much
                          fainter — the star was caught bright on the plate.
``BLEND_CONFUSION``       more than one infrared source inside the WISE PSF.
``AGN_QSO``               red W1-W2 **and** a power-law mid-IR SED.  The colour
                          alone is not enough: a 350 K shroud has W1-W2 = 3.2,
                          redder than anything the Stern et al. 2012 wedge was
                          calibrated on, so the separating axis is SED *shape*.
``AGN_QSO_COLOUR_ONLY``   red W1-W2 with fewer than three infrared bands, where
                          a power law and a blackbody are not separable even in
                          principle.  The published ``vanish-neowise`` table
                          carries only W1/W2, so this class is the reason the
                          run must add AllWISE W3/W4.
``GALAXY``                flagged extended in the infrared catalogue.
``DUSTY_AGB``             bright, very red, rising through W3/W4 — an evolved
                          mass-losing giant, the single most abundant genuine
                          "optically faint, IR bright" stellar population.
``YSO``                   low galactic latitude with a rising mid-IR SED.
``RESIDUAL_UNEXPLAINED``  everything the cascade could not name.  This class,
                          and only this class, goes to the energy budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- coordinate transforms (vectorised, no astropy round-trip) --------------
_RA_NGP, _DEC_NGP, _L_NCP = 192.85948, 27.12825, 122.93192
_OBLIQ = 23.4392911


def galactic_latitude(ra_deg, dec_deg):
    """Galactic latitude b in degrees (IAU 1958 pole, J2000 equatorial input)."""
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    d_ngp, r_ngp = np.radians(_DEC_NGP), np.radians(_RA_NGP)
    sb = (np.sin(dec) * np.sin(d_ngp)
          + np.cos(dec) * np.cos(d_ngp) * np.cos(ra - r_ngp))
    return np.degrees(np.arcsin(np.clip(sb, -1.0, 1.0)))


def galactic_longitude(ra_deg, dec_deg):
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    d_ngp, r_ngp = np.radians(_DEC_NGP), np.radians(_RA_NGP)
    y = np.cos(dec) * np.sin(ra - r_ngp)
    x = (np.sin(dec) * np.cos(d_ngp)
         - np.cos(dec) * np.sin(d_ngp) * np.cos(ra - r_ngp))
    return np.degrees(np.radians(_L_NCP) - np.arctan2(y, x)) % 360.0


def ecliptic_latitude(ra_deg, dec_deg):
    """Ecliptic latitude beta in degrees (J2000 obliquity)."""
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    eps = np.radians(_OBLIQ)
    sb = np.sin(dec) * np.cos(eps) - np.cos(dec) * np.sin(eps) * np.sin(ra)
    return np.degrees(np.arcsin(np.clip(sb, -1.0, 1.0)))


# --- helpers ----------------------------------------------------------------
def _g(row, key, default=np.nan) -> float:
    v = row.get(key, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float(default) if default is not np.nan else np.nan
    return v


def _finite(x) -> bool:
    return x is not None and np.isfinite(x)


def _flag(row, key) -> bool:
    """Truthiness that treats NaN as False.

    Rows come from a merged DataFrame, so a column contributed by one object
    appears as NaN on every other — and ``bool(nan)`` is ``True``, which would
    silently mark every source as a recovered high-proper-motion star.
    """
    v = row.get(key, False)
    if v is None:
        return False
    if isinstance(v, float) and not np.isfinite(v):
        return False
    try:
        return bool(v)
    except (TypeError, ValueError):
        return False


def has_ir_detection(row, min_bands: int = 1) -> bool:
    n = sum(1 for b in ("2mass_j", "2mass_h", "2mass_ks", "w1", "w2", "w3", "w4")
            if _finite(_g(row, b)))
    return n >= min_bands


def n_ir_bands(row) -> int:
    return sum(1 for b in ("2mass_j", "2mass_h", "2mass_ks", "w1", "w2", "w3", "w4")
               if _finite(_g(row, b)))


def has_modern_optical(row) -> bool:
    return any(_finite(_g(row, b)) for b in
               ("gaia_g", "gaia_bp", "gaia_rp",
                "ps1_g", "ps1_r", "ps1_i", "ps1_z", "ps1_y"))


def modern_optical_mag(row) -> float:
    """A representative modern red-optical magnitude, for the fade comparison."""
    for b in ("ps1_r", "gaia_g", "ps1_i", "gaia_rp", "ps1_g", "gaia_bp"):
        v = _g(row, b)
        if _finite(v):
            return v
    return np.nan


def _ir_shape(row, cfg: dict) -> tuple[bool, dict]:
    """Power-law-vs-blackbody shape test on this row's infrared photometry."""
    from . import sed as sedmod
    from .vet import build_sed

    grid = cfg.get("sed", {}).get("tdust_grid_k", [120, 250, 500, 1000, 1800])
    return sedmod.ir_shape_prefers_powerlaw(build_sed(row), grid)


# --- the cascade ------------------------------------------------------------
def classify_source(row, cfg: dict) -> tuple[str, str]:
    """Return ``(class, reason)`` for one object."""
    c = cfg.get("classify", {})
    v = cfg.get("vet", {})

    ra, dec = _g(row, "ra_deg"), _g(row, "dec_deg")
    glat = float(galactic_latitude(ra, dec)) if _finite(ra) and _finite(dec) else np.nan
    ecl = float(ecliptic_latitude(ra, dec)) if _finite(ra) and _finite(dec) else np.nan

    w1, w2, w3, w4 = (_g(row, b) for b in ("w1", "w2", "w3", "w4"))
    ks = _g(row, "2mass_ks")
    plate = _g(row, "poss1_e")
    if not _finite(plate):
        plate = _g(row, "poss1_o")

    n_ir = n_ir_bands(row)
    n_neigh = _g(row, "n_ir_neighbours", 1.0)

    # 1. Nothing anywhere -> a plate defect, or a genuinely counterpart-free
    #    transient.  This is the control sample and never enters the budget.
    if n_ir == 0 and not has_modern_optical(row):
        if _finite(ecl) and abs(ecl) <= float(c.get("asteroid_abs_ecl_lat_max", 20.0)):
            return ("ASTEROID",
                    f"no counterpart at any wavelength, |ecliptic lat| = {abs(ecl):.1f} deg "
                    "<= the main-belt band: a moving object on a single plate")
        return ("PLATE_DEFECT",
                "no counterpart at any wavelength; an emulsion artefact has no "
                "infrared source behind it (Hambly & Blair 2024)")

    # 2. It never actually vanished.
    mod = modern_optical_mag(row)
    if _finite(mod) and _finite(plate):
        dmag = mod - plate
        if dmag < float(c.get("variable_delta_mag_min", 1.0)):
            return ("MODERN_OPTICAL_MATCH",
                    f"modern optical source at the position, {dmag:+.2f} mag "
                    "relative to the plate: not a disappearance")
        return ("VARIABLE_STAR",
                f"modern optical counterpart {dmag:+.2f} mag fainter than the "
                "plate: the star was caught bright on POSS-I")

    # 3. It moved (flag supplied by vet.epoch_propagation_check).
    if _flag(row, "pm_recovered"):
        sep = _g(row, "pm_back_propagated_sep_arcsec")
        mu = _g(row, "pm_total_mas_yr")
        return ("HIGH_PM_STAR",
                f"a modern source with mu = {mu:.0f} mas/yr propagates back to "
                f"{sep:.1f}\" of the POSS-I position")

    # 4. Confusion.
    if _finite(n_neigh) and n_neigh > float(v.get("max_ir_neighbours_in_psf", 1)):
        return ("BLEND_CONFUSION",
                f"{int(n_neigh)} infrared sources inside the "
                f"{v.get('wise_psf_arcsec', 6.0):g}\" WISE PSF")

    # 5. Extragalactic.
    #
    # A red W1-W2 alone CANNOT separate an AGN from an enshrouded star: a 350 K
    # shroud has W1-W2 = 3.2, far redder than any object the Stern et al. 2012
    # wedge was calibrated on, so applying the colour cut naively would delete
    # exactly the population this channel exists to find.  The separable axis is
    # SED *shape*: an AGN is a power law across 3-22 um, a shroud is a curved
    # single-temperature blackbody.  With fewer than three infrared bands the
    # two are not distinguishable even in principle, and the cut then falls back
    # to colour alone and says so.
    if _finite(w1) and _finite(w2):
        w1w2 = w1 - w2
        if w1w2 >= float(c.get("agn_w1w2_min", 0.80)):
            pl, info = _ir_shape(row, cfg)
            if not info["decidable"]:
                return ("AGN_QSO_COLOUR_ONLY",
                        f"W1-W2 = {w1w2:.2f} is in the AGN colour range but "
                        f"only {n_ir} IR band(s) are available: an AGN power law "
                        "and a warm blackbody are not separable here")
            if pl:
                return ("AGN_QSO",
                        f"W1-W2 = {w1w2:.2f} with a power-law mid-IR SED "
                        f"(beta = {info['beta']:.2f}, chi2 {info['chi2_powerlaw']:.1f} "
                        f"vs {info['chi2_ir_blackbody']:.1f} for a blackbody)")
    if float(_g(row, "ir_ext_flag", 0.0) or 0.0) > 0:
        return ("GALAXY", "flagged extended in the infrared catalogue")

    # 6. Dusty evolved stars — the dominant genuine optically-faint/IR-bright
    #    stellar population, and the class most easily mistaken for a shroud.
    if (_finite(w1) and _finite(w2) and _finite(w3) and _finite(w4)
            and (w1 - w2) >= float(c.get("agb_w1w2_min", 0.30))
            and (w3 - w4) >= float(c.get("agb_w3w4_min", 1.20))
            and _finite(ks) and ks <= float(c.get("agb_ks_max", 8.0))):
        return ("DUSTY_AGB",
                f"Ks = {ks:.2f} bright with W1-W2 = {w1 - w2:.2f} and "
                f"W3-W4 = {w3 - w4:.2f}: a mass-losing evolved giant")

    # 7. Young stellar objects.
    if (_finite(glat) and abs(glat) <= float(c.get("yso_abs_glat_max", 5.0))
            and _finite(w2) and _finite(w3)
            and (w2 - w3) >= float(c.get("yso_w2w3_min", 1.0))):
        return ("YSO",
                f"|b| = {abs(glat):.1f} deg with W2-W3 = {w2 - w3:.2f}: a rising "
                "mid-IR SED in the star-forming plane")

    return ("RESIDUAL_UNEXPLAINED",
            f"{n_ir} infrared band(s), no modern optical counterpart, and no "
            "mundane class fits")


def classify_table(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply the cascade to a whole table; adds ``class`` and ``class_reason``."""
    if not len(df):
        out = df.copy()
        out["class"] = pd.Series(dtype=object)
        out["class_reason"] = pd.Series(dtype=object)
        return out
    labels, reasons = [], []
    for _, row in df.iterrows():
        lab, why = classify_source(row, cfg)
        labels.append(lab)
        reasons.append(why)
    out = df.copy()
    out["class"] = labels
    out["class_reason"] = reasons
    if {"ra_deg", "dec_deg"} <= set(out.columns):
        out["glat_deg"] = galactic_latitude(out["ra_deg"], out["dec_deg"])
        out["glon_deg"] = galactic_longitude(out["ra_deg"], out["dec_deg"])
        out["ecl_lat_deg"] = ecliptic_latitude(out["ra_deg"], out["dec_deg"])
    return out


def population_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Counts and fractions per class — the first-ever census of this sample."""
    if not len(df) or "class" not in df.columns:
        return pd.DataFrame(columns=["class", "n", "fraction"])
    counts = df["class"].value_counts()
    out = counts.rename_axis("class").reset_index(name="n")
    out["fraction"] = out["n"] / float(len(df))
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def obscuration_vs_destruction_ratio(df: pd.DataFrame) -> dict:
    """The measurement the two samples were made for.

    ``R = N(optically vanished WITH an IR counterpart) /
          N(optically vanished with NO counterpart anywhere)``

    reported both raw and after the mundane classes are subtracted.  Solano+2022
    published both samples and never formed this ratio.
    """
    if not len(df):
        return {"n_with_ir": 0, "n_no_counterpart": 0, "ratio_raw": None,
                "n_residual_with_ir": 0, "ratio_after_subtraction": None}
    with_ir = int(df.apply(lambda r: n_ir_bands(r) > 0, axis=1).sum())
    none_any = int((df["class"] == "PLATE_DEFECT").sum()
                   + (df["class"] == "ASTEROID").sum()) if "class" in df else 0
    resid = int((df["class"] == "RESIDUAL_UNEXPLAINED").sum()) if "class" in df else 0
    return {
        "n_with_ir": with_ir,
        "n_no_counterpart": none_any,
        "ratio_raw": (with_ir / none_any) if none_any else None,
        "n_residual_with_ir": resid,
        "ratio_after_subtraction": (resid / none_any) if none_any else None,
    }


__all__ = [
    "classify_source", "classify_table", "ecliptic_latitude",
    "galactic_latitude", "galactic_longitude", "has_ir_detection",
    "has_modern_optical", "modern_optical_mag", "n_ir_bands",
    "obscuration_vs_destruction_ratio", "population_breakdown",
]
