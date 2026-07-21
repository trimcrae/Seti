"""Pure-logic scorers for the LHS 1140 signature sweep (unit-tested offline).

The per-target detectors themselves live in :mod:`seti.panspermia.dossier` and are
reused verbatim (companion astrometry, WISE IR-colour excess, NEOWISE mid-IR
variability, ZTF/TESS light-curve verdict, Gaia XP narrow-line scan).  This module
adds the two pieces specific to the LHS 1140 investigation:

* :func:`neighbor_ir_excess_scan` / :func:`neighbor_companion_scan` -- the same
  waste-heat and hidden-companion screens applied at *catalogue scale* to every
  Gaia source in the local volume, so the search covers the star's neighbours and
  not only the star itself; and
* :func:`inventory_summary` -- an honest accounting of which archival
  *spectroscopic* observations exist for the system (a molecular biosignature can
  only live in a transmission/emission spectrum, so this records whether such data
  have ever been taken, which instrument, and over what wavelengths).

Nothing here assumes a signature; every scorer records "clean"/"no data" where
that is the honest answer and flags only what genuinely exceeds a physical
threshold.
"""

from __future__ import annotations

import numpy as np

# Re-export the shared detectors so the runner imports them from one place.
from ..panspermia.dossier import (  # noqa: F401
    companion_diagnostics,
    dossier_verdict,
    ir_color_excess,
    ir_variability_verdict,
    lightcurve_verdict,
    narrow_feature_scan,
)

# --- LHS 1140 system anchor ------------------------------------------------
# ICRS position (J2000, SIMBAD); the runner resolves the live Gaia DR3 row by a
# proper-motion-tolerant cone and picks the nearest (highest-parallax) source, so
# these are a fallback only.  LHS 1140 is a very high proper-motion star
# (~0.7 arcsec/yr), which is why every survey cone must be PM-propagated.
LHS1140 = {
    "name": "LHS 1140",
    "aliases": ["GJ 3053", "LP 767-6", "TOI-256"],
    "ra": 11.24704, "dec": -15.27153,        # deg, ICRS J2000
    "parallax_mas": 66.70,                    # ~14.99 pc
    "pmra": 317.6, "pmdec": -596.6,           # mas/yr (high PM)
    # Gaia DR3 source_id is resolved on the runner; this literature value is a
    # fallback used only if the cone resolution fails.
    "source_id_fallback": 2371375962351331328,
}

# The two known transiting planets, for context in the biosignature inventory.
# (The transit signal of each is carried in the star's light curve, so they need
# no separate acquisition -- they are why the star's photometry matters.)
PLANETS = [
    {"name": "LHS 1140 b", "period_d": 24.737, "radius_rearth": 1.73,
     "insolation_searth": 0.43, "note": "temperate HZ rocky/water world; "
     "atmosphere reported (Cadieux+2024)"},
    {"name": "LHS 1140 c", "period_d": 3.778, "radius_rearth": 1.27,
     "insolation_searth": 8.4, "note": "hot super-Earth"},
]


# --- Neighbour-sweep scorers (catalogue-scale technosignature battery) ------
# WISE contamination guards (the AllWISE lesson, re-earned on the LHS 1140
# neighbours).  Two systematics dominate any catalogue-scale WISE excess screen:
#   (1) a W4-only "excess" -- W4 (22 um) is the shallowest, most confusion-limited
#       band, so for intrinsically 22-um-faint stars its flux is background cirrus
#       / a noise measurement, giving huge (up to 6 mag) formally-high-sigma
#       "excesses" with a photospheric W1-W2.  A real warm-dust / waste-heat SED is
#       bounded and lights up the shorter, star-dominated bands first, so a
#       W4-only excess is an artefact, not a detection.
#   (2) a negative W1-W2 -- a bare stellar photosphere has W1-W2 >= 0 (Vega); a
#       negative value means a blend / bad photometry in W1 or W2, not a star.
# A genuine excess must therefore appear in a star-dominated band (W1-W2, the hot
# ~1000 K band, or W1-W3, the ~300 K warm-dust band) with a physical W1-W2 >= 0.
_W1W2_BLEND_FLOOR = -0.05


def neighbor_ir_excess_scan(rows: list[dict]) -> dict:
    """Run the WISE IR-colour-excess screen over a list of neighbour rows.

    Each row must carry AllWISE ``w1mpro``..``w4mpro`` (+ their ``*sigmpro``).  The
    raw per-row colours come from :func:`ir_color_excess`; on top of it we apply the
    contamination guards above so a catalogue-scale sweep does not report the WISE
    W4 faint-source artefact (or a W1/W2 blend) as a waste-heat candidate.  A
    W4-only or negative-W1-W2 excess is recorded under ``needs_vetting`` rather than
    ``flagged``, keeping the coverage honest without manufacturing candidates.
    """
    flagged, needs_vetting, n_with_wise = [], [], 0
    for r in rows:
        ir = ir_color_excess(r)
        if ir.get("has_data"):
            n_with_wise += 1
        if not ir.get("ir_excess_flag"):
            continue
        w1w2 = ir.get("W1_W2")
        rec = {"source_id": r.get("source_id"), "ra": r.get("ra"),
               "dec": r.get("dec"), "reasons": ir["reasons"],
               **{k: ir.get(k) for k in ("W1_W2", "W1_W3", "W1_W4")}}
        # A star-dominated (W1-W2 or W1-W3) excess with a physical W1-W2 is a real
        # candidate; a W4-only excess or a negative W1-W2 is a known systematic.
        star_band = any(("W1_W2" in s or "W1_W3" in s) for s in ir["reasons"])
        blend = w1w2 is not None and np.isfinite(w1w2) and w1w2 < _W1W2_BLEND_FLOOR
        if star_band and not blend:
            flagged.append(rec)
        else:
            why = ("W1-W2<0 (blend/bad photometry)" if blend
                   else "W4-only excess (AllWISE W4 cirrus/faint-source artefact)")
            needs_vetting.append({**rec, "vetting": why})
    return {"n_sources": len(rows), "n_with_wise": n_with_wise,
            "n_needs_vetting": len(needs_vetting), "needs_vetting": needs_vetting,
            "n_ir_excess": len(flagged), "flagged": flagged}


def neighbor_companion_scan(rows: list[dict]) -> dict:
    """Run the Gaia hidden-companion astrometric screen over neighbour rows.

    Per-row logic is :func:`companion_diagnostics` (RUWE > 1.4, a real >=1 mas
    astrometric excess at high significance, IPD multi-peak, or an NSS solution).
    A flagged neighbour is a candidate unseen massive/dark companion -- worth a
    look, though the overwhelming prior is an ordinary binary.
    """
    flagged, n_with_astro = [], 0
    for r in rows:
        diag = companion_diagnostics(r)
        if np.isfinite(diag.get("ruwe", np.nan)):
            n_with_astro += 1
        if diag.get("companion_flag"):
            flagged.append({"source_id": r.get("source_id"),
                            "ra": r.get("ra"), "dec": r.get("dec"),
                            "reasons": diag["reasons"],
                            "ruwe": diag.get("ruwe")})
    return {"n_sources": len(rows), "n_with_astrometry": n_with_astro,
            "n_companion_flag": len(flagged), "flagged": flagged}


# --- Biosignature-observation inventory ------------------------------------
# A molecular biosignature (O2/O3, CH4, N2O, DMS, ...) can only be recovered from
# a *spectrum*, and for a transiting planet that means transmission or emission
# spectroscopy.  We cannot re-derive a biosignature at catalogue scale, but we can
# state honestly whether such data have ever been taken.  These are the archival
# collections whose presence makes a molecular search even possible.
_SPECTRO_INTENTS = {"spectrum", "spectroscopy", "spectra"}
# Instruments whose data can, in principle, carry an atmospheric molecular feature
# for a transiting terrestrial planet.
_ATMOSPHERE_CAPABLE = {
    "NIRISS", "NIRSPEC", "NIRCAM", "MIRI",        # JWST
    "STIS", "WFC3", "COS",                          # HST
    "ESPRESSO", "HARPS", "NIRPS", "IGRINS",         # ground high-res
}


def inventory_summary(records: list[dict]) -> dict:
    """Summarise a list of MAST/archive observation records into coverage.

    ``records`` are dicts with (case-insensitive) keys among ``instrument_name``,
    ``dataproduct_type``/``intentType``, ``t_exptime``, ``em_min``/``em_max``
    (metres), ``obs_collection``.  Returns per-instrument counts, whether any
    *spectroscopic, atmosphere-capable* observation exists (the pre-requisite for a
    molecular biosignature search), and the wavelength span covered.
    """
    def get(rec, *keys):
        for k in keys:
            for rk in rec:
                if rk.lower() == k.lower() and rec[rk] not in (None, ""):
                    return rec[rk]
        return None

    per_instrument: dict[str, int] = {}
    collections: dict[str, int] = {}
    n_spectro = 0
    atmosphere_capable = False
    em_lo, em_hi = np.inf, -np.inf
    total_exp = 0.0
    for rec in records:
        inst = str(get(rec, "instrument_name", "instrument") or "unknown").upper()
        base_inst = inst.split("/")[0].split("_")[0].strip()
        per_instrument[base_inst] = per_instrument.get(base_inst, 0) + 1
        coll = str(get(rec, "obs_collection", "collection") or "unknown").upper()
        collections[coll] = collections.get(coll, 0) + 1
        ptype = str(get(rec, "dataproduct_type", "intenttype", "intentType")
                    or "").lower()
        is_spectro = any(s in ptype for s in _SPECTRO_INTENTS)
        if is_spectro:
            n_spectro += 1
            if base_inst in _ATMOSPHERE_CAPABLE:
                atmosphere_capable = True
        exp = get(rec, "t_exptime", "exptime")
        try:
            total_exp += float(exp)
        except (TypeError, ValueError):
            pass
        for k in ("em_min", "em_max"):
            v = get(rec, k)
            try:
                v = float(v)
                em_lo, em_hi = min(em_lo, v), max(em_hi, v)
            except (TypeError, ValueError):
                pass
    span = None
    if np.isfinite(em_lo) and np.isfinite(em_hi):
        # MAST reports em_min/em_max in nanometres; convert to micron.
        span = {"min_um": em_lo / 1e3, "max_um": em_hi / 1e3}
    return {
        "n_observations": len(records),
        "n_spectroscopic": n_spectro,
        "atmosphere_capable_spectroscopy": atmosphere_capable,
        "per_instrument": dict(sorted(per_instrument.items(),
                                      key=lambda kv: -kv[1])),
        "collections": dict(sorted(collections.items(), key=lambda kv: -kv[1])),
        "wavelength_span": span,
        "total_exptime_s": total_exp,
        "note": ("atmosphere-capable spectroscopy present -- a molecular "
                 "biosignature search is possible on these data"
                 if atmosphere_capable else
                 "no atmosphere-capable spectroscopy found in the queried "
                 "archives (photometry/astrometry only)"),
    }


__all__ = [
    "LHS1140", "PLANETS",
    "companion_diagnostics", "ir_color_excess", "ir_variability_verdict",
    "lightcurve_verdict", "narrow_feature_scan", "dossier_verdict",
    "neighbor_ir_excess_scan", "neighbor_companion_scan", "inventory_summary",
]
