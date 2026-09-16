"""GROWTH stage 3 --- **is the transit on the target?**

Why this module exists
----------------------
Stage 2 (:mod:`seti.growth.stage2`) measured the TESS-era depth of Kepler-718 b
(K00897.01, KIC 7849854, TIC 268924036, TOI 4490.01) **from the light curve**:
30,614 +/- 1,195 ppm over sectors 41, 54, 55, 74, 75, 81 and 82, against a
Kepler-era catalogue depth of 13,884 ppm carried into the TESS band --- a factor
2.2 deeper.

**Dilution cannot produce that.**  Extra light in the aperture makes a transit
*shallower*, and TESS's ~21" pixels admit strictly more light than Kepler's
~4" ones, so on the same star TESS must read shallower, never deeper.  The one
reading that *does* produce it is that **the two apertures are not measuring the
same source**: if the dimming originates on a neighbour, an aperture centred
differently reports a different depth entirely, and "the depth grew" is a
statement about two different stars.

Three facts make that pressing for this target:

* stage 1 counted **7 Gaia sources within 21"** (``n_gaia_neighbours``);
* the out-of-transit scatter in the TESS photometry is **56,698 ppm --- 5.7 %**,
  nearly twice the transit depth, so the aperture is dominated by something
  noisy;
* stage 2 lists ``per_pixel_centroid_test`` in
  ``summary.json["checks_not_performed"]``.

The odd--even depth difference is 0.20 sigma, so an eclipsing binary at *twice*
the period is already disfavoured.  That does not exclude a source at the
**same** period on a different star.  This module answers the question with two
measurements instead of an assertion.

1. The neighbour census --- the cheapest decisive test
-----------------------------------------------------
:func:`neighbour_census` lists every Gaia DR3 source within a configurable
radius (default one TESS pixel, 21") with ``G``, separation and position angle,
computes each one's **flux fraction in the aperture**

.. code::

    frac_i = 10^(-0.4 (G_i - G_target)) w(d_i) / SUM_j 10^(-0.4 (G_j - G_target)) w(d_j)

(:func:`seti.growth.drift.aperture_weight` supplies ``w``, the same erfc
capture model stage 1 uses), and then the quantity that settles most cases
without touching a pixel: **the depth that neighbour would need in its own
light to produce the observed aperture depth**,

.. code::

    required_depth_i = observed_aperture_depth / frac_i .

A required depth above 100 % is **arithmetically impossible** --- a star cannot
fade by more than all of its light --- so that neighbour is excluded, and the
census says so per neighbour with the number that excluded it.  For a 30,614 ppm
aperture depth, a neighbour needs ``frac_i >= 0.0306``: roughly ``dG <= 3.7``
mag of the target once the target's own light is in the denominator.  Everything
fainter than that is excluded by arithmetic alone.

2. The difference image --- where the transit actually is
---------------------------------------------------------
:func:`difference_image_offset` builds, per sector, a mean **in-transit** image
and a mean **out-of-transit** image from the target pixel files, differences
them (``oot - in``, so the transit source is a *positive* peak), fits a centroid
to the difference image and to the out-of-transit image, and reports the
**offset between them** in pixels and arcseconds with an uncertainty.  An offset
consistent with zero says the transit is on whatever dominates the aperture; a
significant offset points somewhere else, and the census from step 1 says which
neighbour sits there.

Reported **per sector as well as combined**, deliberately: a consistent offset
across seven sectors is a very different fact from a scattered one, and
``sector_scatter_chi2_per_dof`` is the number that tells them apart.  Note that
the out-of-transit centroid is itself pulled toward a bright neighbour, so the
offset measured against it is a **lower bound** on the displacement from the
target; where a target pixel position is known (from the file's WCS and the
catalogue position) ``offset_from_target_*`` is reported beside it.

3. The verdict vocabulary
-------------------------
``TRANSIT_ON_TARGET``
    the offset is consistent with zero **and** the measurement was precise
    enough to have detected the offset that matters (the separation of the
    nearest neighbour the census could not exclude).
``TRANSIT_OFFSET_FROM_TARGET``
    the offset is significant at ``n_offset`` sigma.  The attributed neighbour,
    if the census has one at that separation, is named.
``OFFSET_UNRESOLVED``
    an offset was measured and its error bar spans zero **without** the
    precision to exclude the neighbour that matters, or no centroid could be
    fitted.  **This is never reported as ``TRANSIT_ON_TARGET``.**
``NO_DATA_REACHED``
    no pixels were reached at all.  ``reason`` keeps ``QUERY_FAILED`` (the
    archive errored) and ``QUERY_RETURNED_ZERO_ROWS`` (it answered with
    nothing) apart, and adds ``EPHEMERIS_UNAVAILABLE`` / ``NO_TRANSIT_CADENCES``
    / ``BUDGET_EXHAUSTED`` / ``NOT_ATTEMPTED``.

**A target whose pixels could not be fetched is never reported as on target.**

Access and budgets
------------------
Everything that touches a service takes an injectable callable (``cone_fn`` for
Gaia, ``query_fn`` for the Exoplanet Archive ephemerides, ``tpf_fn`` for the
target pixel files) and carries a **wall-clock budget**, because target pixel
files are large and an unbounded stage has already cost this channel a whole run
(``config/growth.yaml``, ``gaia.cone_budget_s``, and the reason recorded beside
it).  ``centroid.tpf.max_sectors`` is the per-target cap on how many sectors are
downloaded at all.  Nothing here reaches the network inside a test.
"""

from __future__ import annotations

import argparse
import json
import math
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .acquire import (
    GAIA_TAP,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_ZERO,
    AcquisitionLog,
    gaia_neighbours_cones,
    pyvo_sync_transport,
    tap_sync,
)
from .drift import aperture_weight
from .stage2 import (
    BTJD_OFFSET,
    Deadline,
    bkjd_to_btjd,
    fetch_koi_ephemerides,
    fold,
    propagate_epoch,
)
from .stage2 import (
    load_shortlist as stage2_shortlist,
)

# ---------------------------------------------------------------------------
# Verdict vocabulary.  Per target, and for the run (the same words).
# ---------------------------------------------------------------------------
ON_TARGET = "TRANSIT_ON_TARGET"
OFFSET_FROM_TARGET = "TRANSIT_OFFSET_FROM_TARGET"
OFFSET_UNRESOLVED = "OFFSET_UNRESOLVED"
NO_DATA = "NO_DATA_REACHED"
TARGET_VERDICTS = (ON_TARGET, OFFSET_FROM_TARGET, OFFSET_UNRESOLVED, NO_DATA)
RUN_VERDICTS = TARGET_VERDICTS

#: Why there is no usable offset.  ``QUERY_FAILED`` (the service errored) and
#: ``QUERY_RETURNED_ZERO_ROWS`` (it answered with nothing) are different facts
#: and are never collapsed into each other.
REASON_QUERY_FAILED = STATUS_FAILED
REASON_ZERO_ROWS = STATUS_ZERO
REASON_NO_EPHEMERIS = "EPHEMERIS_UNAVAILABLE"
REASON_NO_TRANSIT = "NO_TRANSIT_CADENCES"
REASON_BUDGET = "BUDGET_EXHAUSTED"
REASON_NOT_ATTEMPTED = "NOT_ATTEMPTED"
REASON_NO_CENTROID = "CENTROID_NOT_FITTED"
REASON_PRECISION = "PRECISION_INSUFFICIENT"
REASONS = (REASON_QUERY_FAILED, REASON_ZERO_ROWS, REASON_NO_EPHEMERIS, REASON_NO_TRANSIT,
           REASON_BUDGET, REASON_NOT_ATTEMPTED, REASON_NO_CENTROID, REASON_PRECISION)

STAGES = ("probe", "census", "difference", "assess")

#: Target-pixel-file authors, best cadence first --- the same order stage 2 uses
#: for light curves.  SPOC 2-minute target pixel files where they exist,
#: TESS-SPOC FFI cutouts otherwise.  QLP publishes no pixel product, so it is
#: absent here on purpose.
DEFAULT_TPF_AUTHORS: tuple[str, ...] = ("SPOC", "TESS-SPOC")

ROUTE_LIGHTKURVE = "lightkurve"
ROUTE_MAST_FITS = "astroquery_mast_fits"

#: TESS pixel scale, arcseconds per pixel (the number that makes a pixel offset
#: a sky offset).  ``verify`` only in the sense that a file's own WCS is used in
#: preference whenever one is available.
TESS_PIXEL_ARCSEC = 21.0

CENTROID_ERR_BOOTSTRAP = "bootstrap"
CENTROID_ERR_ANALYTIC = "analytic"


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class CensusParams:
    """The Gaia neighbour census and its arithmetic (``config/growth.yaml``)."""

    radius_arcsec: float = 21.0            # one TESS pixel
    aperture_radius_arcsec: float = 21.0   # the erfc capture model's aperture
    psf_sigma_arcsec: float = 10.5
    target_match_arcsec: float = 1.5       # the Gaia source that IS the target
    max_depth_fraction: float = 1.0        # a star cannot fade by more than its light
    cone_timeout_s: float = 120.0
    cone_retries: int = 2
    cone_budget_s: float = 600.0
    tap_url: str = GAIA_TAP

    @classmethod
    def from_config(cls, conf: dict | None) -> CensusParams:
        c = ((conf or {}).get("centroid") or {}).get("census") or {}
        d = cls()
        for k in d.__dataclass_fields__:
            if c.get(k) is not None:
                setattr(d, k, type(getattr(d, k))(c[k]))
        return d


@dataclass
class DiffParams:
    """The difference image, its centroids and the offset threshold."""

    in_transit_fraction: float = 0.7       # in-transit window = fraction * T14, centred
    oot_inner_durations: float = 0.75      # out-of-transit starts beyond this, in T14
    oot_outer_durations: float = 2.5       # ... and ends here, in T14
    min_in_cadences: int = 3
    min_oot_cadences: int = 6
    moment_threshold: float = 0.2          # pixels above this fraction of the peak
    min_pixels: int = 3
    subtract_background: bool = True       # a robust per-image pedestal
    bootstrap_draws: int = 400
    min_cadences_for_bootstrap: int = 8
    err_method: str = "auto"               # auto | bootstrap | analytic
    # `verify` --- ASSERTED FLOOR, not a measurement.  A thresholded first
    # moment on 21" pixels cannot be trusted below about an arcsecond: the PRF
    # is undersampled and position-dependent, the aperture's faint wings move
    # the direct-image centroid by a fraction of a pixel that the difference
    # image does not share, and the cadence bootstrap knows nothing about any
    # of it.  Without this floor a 0.1" offset measured to 0.03" comes out as a
    # 4-sigma "the transit is not on the target", which is arithmetic, not
    # astrometry.  The floor is added in quadrature per sector AND to the
    # combined error, because stacking sectors cannot beat a systematic.
    centroid_sys_floor_arcsec: float = 1.0
    n_offset: float = 3.0                  # |offset| / sigma at or above this is an offset
    on_target_max_offset_arcsec: float = 21.0   # the resolution ON_TARGET requires by default
    sector_scatter_chi2_per_dof: float = 3.0
    seed: int = 7

    @classmethod
    def from_config(cls, conf: dict | None) -> DiffParams:
        c = ((conf or {}).get("centroid") or {}).get("difference") or {}
        d = cls()
        for k in d.__dataclass_fields__:
            if c.get(k) is not None:
                setattr(d, k, type(getattr(d, k))(c[k]))
        return d


@dataclass
class TpfParams:
    """Wall-clock ceilings and the per-target sector cap on the pixel fetch.

    Target pixel files are **large** --- a 2-minute SPOC TPF is tens of
    megabytes per sector --- so the cap and the budget are not decoration.  The
    pattern and the reason are ``config/growth.yaml``'s ``gaia.cone_budget_s``:
    run 34789826297 sat three hours in an unbounded fetch and would have been
    killed at the workflow cap with nothing committed.  What the budget does not
    reach is ``NO_DATA_REACHED`` with reason ``BUDGET_EXHAUSTED``, never an
    offset and never an on-target verdict.
    """

    authors: tuple[str, ...] = DEFAULT_TPF_AUTHORS
    sectors: tuple[int, ...] = ()          # empty: whatever the archive has, capped
    max_sectors: int = 8                   # the PER-TARGET cap on sectors downloaded
    per_target_budget_s: float = 1800.0
    budget_s: float = 5400.0               # the WHOLE difference stage
    target_timeout_s: float = 900.0
    retries: int = 2
    retry_pause_s: float = 5.0
    quality_bitmask: str = "default"
    download_dir: str | None = None
    archive_timeout_s: float = 300.0       # the Exoplanet Archive ephemeris pulls
    archive_retries: int = 3

    @classmethod
    def from_config(cls, conf: dict | None) -> TpfParams:
        m = ((conf or {}).get("centroid") or {}).get("tpf") or {}
        d = cls()
        if m.get("authors"):
            d.authors = tuple(str(a) for a in m["authors"])
        if m.get("sectors"):
            d.sectors = tuple(int(s) for s in m["sectors"])
        for k in ("per_target_budget_s", "budget_s", "target_timeout_s", "retry_pause_s",
                  "archive_timeout_s"):
            if m.get(k) is not None:
                setattr(d, k, float(m[k]))
        for k in ("max_sectors", "retries", "archive_retries"):
            if m.get(k) is not None:
                setattr(d, k, int(m[k]))
        for k in ("quality_bitmask", "download_dir"):
            if m.get(k) is not None:
                setattr(d, k, str(m[k]))
        return d


# ---------------------------------------------------------------------------
# Spherical geometry
# ---------------------------------------------------------------------------
def sky_sep_arcsec(ra0: float, dec0: float, ra: float, dec: float) -> float:
    """Angular separation in arcseconds (the haversine form, stable at small d)."""
    if not all(np.isfinite([ra0, dec0, ra, dec])):
        return float("nan")
    p0, p1 = math.radians(float(dec0)), math.radians(float(dec))
    dphi = p1 - p0
    dlam = math.radians(float(ra) - float(ra0))
    h = (math.sin(dphi / 2.0) ** 2
         + math.cos(p0) * math.cos(p1) * math.sin(dlam / 2.0) ** 2)
    return float(math.degrees(2.0 * math.asin(min(1.0, math.sqrt(h)))) * 3600.0)


def position_angle_deg(ra0: float, dec0: float, ra: float, dec: float) -> float:
    """Position angle of ``(ra, dec)`` from ``(ra0, dec0)``, degrees **east of north**."""
    if not all(np.isfinite([ra0, dec0, ra, dec])):
        return float("nan")
    p0, p1 = math.radians(float(dec0)), math.radians(float(dec))
    dl = math.radians(float(ra) - float(ra0))
    y = math.sin(dl) * math.cos(p1)
    x = math.cos(p0) * math.sin(p1) - math.sin(p0) * math.cos(p1) * math.cos(dl)
    return float(math.degrees(math.atan2(y, x)) % 360.0)


# ---------------------------------------------------------------------------
# 1. The neighbour census
# ---------------------------------------------------------------------------
def neighbour_census(neighbours: pd.DataFrame | None, *, ra: float, dec: float,
                     observed_depth_ppm: float, params: CensusParams | None = None,
                     target_g_mag: float = float("nan"),
                     observed_depth_err_ppm: float = float("nan")
                     ) -> tuple[pd.DataFrame, dict]:
    """Every Gaia source in the aperture, its flux fraction, and the depth it would need.

    ``neighbours`` carries ``source_id``, ``ra``, ``dec``, ``phot_g_mean_mag``
    (the shape :func:`seti.growth.acquire.gaia_neighbours_cones` returns).  The
    **target** is the nearest source within ``target_match_arcsec``; when no
    source matches, ``target_g_mag`` (a Kepler or TESS magnitude, say) is used
    for the zero point and the summary says so.

    The arithmetic that can settle the question without a single pixel: a source
    supplying fraction ``frac_i`` of the aperture flux must itself fade by

    ``required_depth_i = observed_aperture_depth / frac_i``

    to produce the observed dip.  ``required_depth_i > max_depth_fraction``
    (1, i.e. 100 %) is impossible --- the source has no more light to lose ---
    and that neighbour is ``excluded_by_arithmetic`` with the number that
    excluded it in ``required_depth_ppm``.

    Returns ``(census, summary)``.
    """
    params = params or CensusParams()
    cols = ["source_id", "ra", "dec", "g_mag", "sep_arcsec", "pa_deg", "is_target",
            "delta_g_mag", "flux_ratio_to_target", "aperture_weight", "flux_fraction",
            "required_depth_ppm", "required_depth_percent", "excluded_by_arithmetic",
            "exclusion_note"]
    depth = float(observed_depth_ppm) * 1e-6 if np.isfinite(observed_depth_ppm) else float("nan")
    summary: dict = {
        "n_sources": 0, "n_neighbours": 0, "target_found_in_gaia": False,
        "target_g_mag": float(target_g_mag), "target_g_source": "",
        "target_sep_arcsec": float("nan"),
        "observed_depth_ppm": float(observed_depth_ppm),
        "observed_depth_err_ppm": float(observed_depth_err_ppm),
        "radius_arcsec": float(params.radius_arcsec),
        "contam_weighted": float("nan"), "contam_max": float("nan"),
        "target_flux_fraction": float("nan"),
        "n_neighbours_excluded": 0, "n_neighbours_not_excluded": 0,
        "census_excludes_all_neighbours": False,
        "min_delta_g_that_can_supply_depth": float("nan"),
        "nearest_unexcluded_sep_arcsec": float("nan"),
        "brightest_unexcluded_source_id": "", "brightest_unexcluded_delta_g": float("nan"),
        "all_neighbours_required_depth_min_ppm": float("nan"),
        "census_statement": "",
    }
    if neighbours is None or not len(neighbours):
        summary["census_statement"] = ("no Gaia source list: the census could not be run, so "
                                       "no neighbour is excluded and none is implicated")
        return pd.DataFrame(columns=cols), summary

    n = neighbours.rename(columns={c: str(c).strip().lower() for c in neighbours.columns}).copy()
    gcol = next((c for c in ("phot_g_mean_mag", "g_mag", "gmag", "g") if c in n.columns), None)
    n["_g"] = pd.to_numeric(n[gcol], errors="coerce") if gcol else np.nan
    n["_ra"] = pd.to_numeric(n.get("ra"), errors="coerce")
    n["_dec"] = pd.to_numeric(n.get("dec"), errors="coerce")
    if "sep_arcsec" in n.columns:
        sep = pd.to_numeric(n["sep_arcsec"], errors="coerce").to_numpy(float)
    else:
        sep = np.array([sky_sep_arcsec(ra, dec, a, d)
                        for a, d in zip(n["_ra"], n["_dec"], strict=True)], dtype=float)
    n["_sep"] = sep
    n["_pa"] = [position_angle_deg(ra, dec, a, d)
                for a, d in zip(n["_ra"], n["_dec"], strict=True)]
    n = n[np.isfinite(n["_sep"]) & (n["_sep"] <= float(params.radius_arcsec))].copy()
    n = n.sort_values("_sep", kind="stable").reset_index(drop=True)
    if not len(n):
        summary["census_statement"] = (f"no Gaia source within {params.radius_arcsec:.1f}\" of "
                                       "the target position")
        return pd.DataFrame(columns=cols), summary

    # The target: the nearest source inside the match radius.  Without one the
    # zero point is whatever magnitude the caller supplied, and that is stated.
    is_target = np.zeros(len(n), dtype=bool)
    tg = float(target_g_mag)
    cand = n[(n["_sep"] <= float(params.target_match_arcsec)) & np.isfinite(n["_g"])]
    if len(cand):
        i = int(cand.index[0])
        is_target[i] = True
        tg = float(n.loc[i, "_g"])
        summary.update({"target_found_in_gaia": True, "target_g_source": "gaia_dr3",
                        "target_sep_arcsec": float(n.loc[i, "_sep"])})
    else:
        summary["target_g_source"] = "caller_supplied" if np.isfinite(tg) else "unknown"
    summary["target_g_mag"] = tg

    g = n["_g"].to_numpy(float)
    dg = g - tg if np.isfinite(tg) else np.full(len(n), np.nan)
    flux_ratio = np.where(np.isfinite(dg), 10.0 ** (-0.4 * dg), np.nan)
    w = np.asarray(aperture_weight(n["_sep"].to_numpy(float), params.aperture_radius_arcsec,
                                   params.psf_sigma_arcsec), dtype=float)
    captured = flux_ratio * w
    total = float(np.nansum(captured))
    frac = captured / total if total > 0 else np.full(len(n), np.nan)
    required = np.where(np.isfinite(frac) & (frac > 0), depth / frac, np.inf)
    excluded = required > float(params.max_depth_fraction)
    notes = []
    for i in range(len(n)):
        if is_target[i]:
            notes.append("the target itself")
        elif not np.isfinite(required[i]):
            notes.append("no G magnitude: the arithmetic cannot be run for this source")
        elif excluded[i]:
            notes.append(f"would need a {required[i] * 100.0:.0f} % eclipse of its own light "
                         f"to make the observed {observed_depth_ppm:.0f} ppm --- impossible")
        else:
            notes.append(f"could supply the observed depth with a {required[i] * 100.0:.2f} % "
                         "eclipse of its own light --- NOT excluded")
    out = pd.DataFrame({
        "source_id": [str(v) for v in n.get("source_id", pd.Series([""] * len(n)))],
        "ra": n["_ra"].to_numpy(float), "dec": n["_dec"].to_numpy(float),
        "g_mag": g, "sep_arcsec": n["_sep"].to_numpy(float), "pa_deg": n["_pa"].to_numpy(float),
        "is_target": is_target, "delta_g_mag": dg, "flux_ratio_to_target": flux_ratio,
        "aperture_weight": w, "flux_fraction": frac,
        "required_depth_ppm": np.where(np.isfinite(required), required * 1e6, np.inf),
        "required_depth_percent": np.where(np.isfinite(required), required * 100.0, np.inf),
        "excluded_by_arithmetic": excluded & ~is_target,
        "exclusion_note": notes,
    })[cols]

    others = out[~out["is_target"]]
    unexcluded = others[~others["excluded_by_arithmetic"]]
    summary.update({
        "n_sources": int(len(out)), "n_neighbours": int(len(others)),
        "contam_weighted": float(np.nansum(captured[~is_target])),
        "contam_max": float(np.nansum(flux_ratio[~is_target])),
        "target_flux_fraction": float(frac[is_target][0]) if is_target.any() else float("nan"),
        "n_neighbours_excluded": int(others["excluded_by_arithmetic"].sum()),
        "n_neighbours_not_excluded": int(len(unexcluded)),
        "census_excludes_all_neighbours": bool(len(others) and not len(unexcluded)),
        "all_neighbours_required_depth_min_ppm": (float(others["required_depth_ppm"].min())
                                                  if len(others) else float("nan")),
    })
    # The magnitude gap at which a neighbour stops being able to do it at all:
    # frac >= depth  =>  10^(-0.4 dG) w >= depth * total.
    if np.isfinite(depth) and depth > 0 and total > 0:
        with np.errstate(divide="ignore"):
            summary["min_delta_g_that_can_supply_depth"] = float(
                -2.5 * math.log10(depth * total)) if depth * total > 0 else float("nan")
    if len(unexcluded):
        summary["nearest_unexcluded_sep_arcsec"] = float(unexcluded["sep_arcsec"].min())
        b = unexcluded.sort_values("delta_g_mag", kind="stable").iloc[0]
        summary["brightest_unexcluded_source_id"] = str(b["source_id"])
        summary["brightest_unexcluded_delta_g"] = float(b["delta_g_mag"])
    summary["census_statement"] = _census_statement(summary)
    return out, summary


def _census_statement(s: dict) -> str:
    """One plain sentence: what the arithmetic alone has and has not settled."""
    nn = int(s.get("n_neighbours", 0))
    ex = int(s.get("n_neighbours_excluded", 0))
    if not nn:
        return (f"no Gaia neighbour within {s.get('radius_arcsec')}\" --- there is no neighbour "
                "for the transit to originate on, as far as Gaia resolves")
    dgl = s.get("min_delta_g_that_can_supply_depth")
    tail = (f"; a source needs dG <= {dgl:.2f} mag of the target to be able to supply "
            f"{s.get('observed_depth_ppm'):.0f} ppm at all" if np.isfinite(dgl) else "")
    if ex == nn:
        return (f"all {nn} Gaia neighbours within {s.get('radius_arcsec')}\" are excluded by "
                f"arithmetic: each would need more than 100 % of its own light to make the "
                f"observed depth{tail}")
    return (f"{ex} of {nn} Gaia neighbours are excluded by arithmetic; "
            f"{nn - ex} could still supply the observed depth (nearest at "
            f"{s.get('nearest_unexcluded_sep_arcsec'):.1f}\", brightest at dG = "
            f"{s.get('brightest_unexcluded_delta_g'):.2f}){tail}")


def fetch_gaia_census(ra: float, dec: float, *, params: CensusParams | None = None,
                      cone_fn=None, log: AcquisitionLog | None = None,
                      key: str = "") -> tuple[pd.DataFrame, str]:
    """Gaia DR3 sources within ``params.radius_arcsec`` of one position.

    Reuses :func:`seti.growth.acquire.gaia_neighbours_cones` (and therefore
    :func:`seti.growth.acquire.pyvo_sync_transport`) rather than opening a new
    route to the same archive.  Returns ``(sources, status)``; a failure is
    ``QUERY_FAILED``, an empty answer is ``QUERY_RETURNED_ZERO_ROWS``, and
    neither is turned into the other.
    """
    params = params or CensusParams()
    log = log or AcquisitionLog(prefix="growth/centroid")
    label = f"gaia_census_{key or 'target'}"
    if not (np.isfinite(ra) and np.isfinite(dec)):
        log.record(label, "no position", error="target has no ra/dec")
        return pd.DataFrame(), STATUS_FAILED
    targets = pd.DataFrame({"key": [0], "ra": [float(ra)], "dec": [float(dec)]})
    attempts: list[dict] = []

    def _default_cone(t, r, **kw):                        # pragma: no cover - runner only
        return gaia_neighbours_cones(
            t, r, retries=int(params.cone_retries), tap_url=str(params.tap_url),
            transport=pyvo_sync_transport(str(params.tap_url),
                                          timeout_s=float(params.cone_timeout_s)),
            timeout_s=float(params.cone_timeout_s),
            budget_s=float(params.cone_budget_s), **kw)

    fn = cone_fn or _default_cone
    try:
        df, failed = fn(targets, float(params.radius_arcsec), attempts=attempts)
    except Exception as exc:                              # noqa: BLE001
        log.record(label, f"cone r={params.radius_arcsec}\"", error=repr(exc)[:400],
                   extra={"attempts": attempts})
        return pd.DataFrame(), STATUS_FAILED
    if failed:
        log.record(label, f"cone r={params.radius_arcsec}\"",
                   error="the cone did not answer for this target",
                   extra={"attempts": attempts})
        return pd.DataFrame(), STATUS_FAILED
    df = df if df is not None else pd.DataFrame()
    log.record(label, f"cone r={params.radius_arcsec}\"", rows=int(len(df)),
               extra={"attempts": attempts})
    return df, (STATUS_OK if len(df) else STATUS_ZERO)


# ---------------------------------------------------------------------------
# 2. The difference image
# ---------------------------------------------------------------------------
def transit_cadence_masks(time_btjd, *, period_days: float, t0_btjd: float,
                          duration_days: float, exptime_days: float = float("nan"),
                          params: DiffParams | None = None
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(in_transit, out_of_transit, dt)`` cadence masks around each predicted transit.

    In transit is the flat-bottom core ``|dt| <= f T14 / 2`` shrunk by half the
    exposure (a cadence whose integration straddles ingress is not in-transit
    flux); out of transit is the **local** annulus
    ``inner T14 <= |dt| <= outer T14``, so a slowly varying background enters
    both images the same way instead of being fitted out globally.
    """
    params = params or DiffParams()
    t = np.asarray(time_btjd, dtype=float)
    if not (np.isfinite(period_days) and period_days > 0 and np.isfinite(duration_days)
            and duration_days > 0 and np.isfinite(t0_btjd)):
        z = np.zeros(t.shape, dtype=bool)
        return z, z.copy(), np.full(t.shape, np.nan)
    _, dt = fold(t, float(period_days), float(t0_btjd))
    half = 0.5 * float(params.in_transit_fraction) * float(duration_days)
    e = float(exptime_days) if np.isfinite(exptime_days) and exptime_days > 0 else 0.0
    half_in = max(half - 0.5 * e, 1e-9)
    a = np.abs(dt)
    in_t = a <= half_in
    oot = (a >= float(params.oot_inner_durations) * float(duration_days)) & (
        a <= float(params.oot_outer_durations) * float(duration_days))
    return in_t, oot, dt


def _robust_background(image: np.ndarray) -> float:
    img = np.asarray(image, dtype=float)
    v = img[np.isfinite(img)]
    return float(np.median(v)) if v.size else 0.0


def image_centroid(image, *, err_image=None, params: DiffParams | None = None,
                   subtract_background: bool | None = None) -> dict:
    """Flux-weighted centroid of an image, in **pixel** coordinates, with an error.

    Pixels entering the first moment are those above
    ``moment_threshold * peak`` of the (background-subtracted) image, which
    keeps a 21"-pixel stamp's noisy corners out of the centroid.  The analytic
    error propagates ``err_image`` through the moment:
    ``dx/dw_i = (x_i - x) / sum(w)``.

    Returns ``x``, ``y`` (column, row; ``0`` is the first pixel), their errors,
    the number of pixels used, the peak and the total weight.
    """
    params = params or DiffParams()
    img = np.asarray(image, dtype=float)
    out = {"x": float("nan"), "y": float("nan"), "x_err": float("nan"), "y_err": float("nan"),
           "n_pixels": 0, "peak": float("nan"), "flux": float("nan"),
           "background": 0.0, "status": REASON_NO_CENTROID}
    if img.ndim != 2 or not np.isfinite(img).any():
        return out
    sub = (params.subtract_background if subtract_background is None else bool(subtract_background))
    bg = _robust_background(img) if sub else 0.0
    w = np.where(np.isfinite(img), img - bg, 0.0)
    peak = float(np.max(w)) if w.size else float("nan")
    out["background"], out["peak"] = float(bg), peak
    if not np.isfinite(peak) or peak <= 0:
        return out
    keep = w >= float(params.moment_threshold) * peak
    keep &= w > 0
    if int(keep.sum()) < int(params.min_pixels):
        out["n_pixels"] = int(keep.sum())
        return out
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ww = np.where(keep, w, 0.0)
    sw = float(ww.sum())
    if not np.isfinite(sw) or sw <= 0:
        return out
    x = float((ww * xx).sum() / sw)
    y = float((ww * yy).sum() / sw)
    out.update({"x": x, "y": y, "n_pixels": int(keep.sum()), "flux": sw, "status": STATUS_OK})
    if err_image is not None:
        e = np.asarray(err_image, dtype=float)
        e = np.where(np.isfinite(e) & keep, e, 0.0)
        out["x_err"] = float(math.sqrt(float((((xx - x) / sw) ** 2 * e**2).sum())))
        out["y_err"] = float(math.sqrt(float((((yy - y) / sw) ** 2 * e**2).sum())))
    return out


def _mean_and_sem(cube: np.ndarray, idx: np.ndarray, err_cube=None
                  ) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-pixel mean over the selected cadences and its standard error.

    The error is the **cadence-to-cadence scatter** of that pixel divided by
    ``sqrt(n)``, which carries the systematics (scattered light, pointing
    jitter) that a photon error does not.  When the file supplies ``FLUX_ERR``
    the propagated photon error is taken as a floor, because three cadences
    cannot estimate their own scatter reliably.
    """
    sel = cube[idx]
    n = int(sel.shape[0])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(sel, axis=0)
        sd = np.nanstd(sel, axis=0, ddof=1) if n > 1 else np.zeros(mean.shape)
    sem = sd / math.sqrt(max(n, 1))
    if err_cube is not None:
        e = np.asarray(err_cube, dtype=float)
        if e.shape == cube.shape:
            with np.errstate(invalid="ignore"):
                phot = np.sqrt(np.nanmean(e[idx] ** 2, axis=0) / max(n, 1))
            sem = np.where(np.isfinite(phot), np.maximum(sem, phot), sem)
    return mean, sem, n


def _magnitude_error(dx: float, dy: float, sx: float, sy: float) -> float:
    """Error on ``r = hypot(dx, dy)``, with a floor so a near-zero offset keeps a bar.

    The projected error ``sqrt((dx^2 sx^2 + dy^2 sy^2))/r`` collapses to
    nonsense as ``r -> 0``; the quadrature mean of the two component errors is
    the floor, which is the number that makes "the error bar spans zero" a
    statement rather than an artefact.
    """
    sx = float(sx) if np.isfinite(sx) else float("nan")
    sy = float(sy) if np.isfinite(sy) else float("nan")
    if not (np.isfinite(sx) and np.isfinite(sy)):
        return float("nan")
    floor = math.sqrt(0.5 * (sx * sx + sy * sy))
    r = math.hypot(float(dx), float(dy))
    if r <= 0:
        return floor
    proj = math.sqrt(dx * dx * sx * sx + dy * dy * sy * sy) / r
    return float(max(proj, floor))


def _debias_magnitude(dx: float, dy: float, sx: float, sy: float) -> float:
    """``sqrt(max(r^2 - sx^2 - sy^2, 0))`` --- the Rice-bias-corrected offset length.

    A length is **positive by construction**: two centroids measured with noise
    are never at exactly the same place, so ``hypot(dx, dy)`` is biased high and
    averaging several sectors' magnitudes shrinks the error without shrinking
    the bias --- which manufactures a significant offset out of pure noise.
    Subtracting the noise power in quadrature is the standard correction; where
    the raw length is already large it changes nothing, and where it is
    comparable with the noise it goes to zero, which is the honest answer.  The
    quoted uncertainty is left at the uncorrected value, so the correction can
    never make a verdict *more* confident.
    """
    r2 = float(dx) ** 2 + float(dy) ** 2
    n2 = (float(sx) ** 2 if np.isfinite(sx) else 0.0) + (float(sy) ** 2 if np.isfinite(sy)
                                                         else 0.0)
    return float(math.sqrt(max(r2 - n2, 0.0)))


def difference_image_offset(sector: dict, *, period_days: float, t0_btjd: float,
                            duration_days: float, params: DiffParams | None = None,
                            rng: np.random.Generator | None = None,
                            pixel_scale_arcsec: float = TESS_PIXEL_ARCSEC) -> dict:
    """One sector's difference image, its two centroids, and the offset between them.

    ``sector`` is a dict with ``time`` (BTJD, one entry per cadence), ``flux``
    (``n_cadence x ny x nx``), optionally ``flux_err``, ``quality``,
    ``exptime_s``, ``target_pixel`` and ``sky_fn`` (a callable
    ``(x, y) -> (ra, dec)`` built from the file's own WCS on the runner).

    The difference is **out-of-transit minus in-transit**, so the transit source
    is a positive peak.  Errors come from a bootstrap over cadences when there
    are enough of them and from the per-pixel standard error otherwise;
    ``centroid_err_method`` always says which.
    """
    params = params or DiffParams()
    rec: dict = {
        "sector": sector.get("sector"), "author": sector.get("author"),
        "exptime_s": sector.get("exptime_s"), "n_cadences": 0, "n_in_transit": 0,
        "n_out_of_transit": 0, "n_transits_covered": 0,
        "shape": "", "status": REASON_NO_TRANSIT, "reject_reason": "",
        "diff_x_px": float("nan"), "diff_y_px": float("nan"),
        "diff_x_err_px": float("nan"), "diff_y_err_px": float("nan"),
        "diff_peak": float("nan"), "diff_n_pixels": 0,
        "oot_x_px": float("nan"), "oot_y_px": float("nan"),
        "oot_x_err_px": float("nan"), "oot_y_err_px": float("nan"),
        "oot_n_pixels": 0,
        "offset_x_px": float("nan"), "offset_y_px": float("nan"),
        "offset_px": float("nan"), "offset_err_px": float("nan"),
        "offset_err_stat_px": float("nan"), "centroid_sys_floor_arcsec": float("nan"),
        "offset_debiased_px": float("nan"), "offset_debiased_arcsec": float("nan"),
        "offset_arcsec": float("nan"), "offset_err_arcsec": float("nan"),
        "offset_sigma": float("nan"), "centroid_err_method": "none",
        "offset_ra_arcsec": float("nan"), "offset_dec_arcsec": float("nan"),
        "offset_pa_deg": float("nan"),
        "offset_from_target_px": float("nan"), "offset_from_target_arcsec": float("nan"),
        "pixel_scale_arcsec": float(sector.get("pixel_scale_arcsec") or pixel_scale_arcsec),
    }
    t = np.asarray(sector.get("time"), dtype=float)
    cube = np.asarray(sector.get("flux"), dtype=float)
    if t.size == 0 or cube.ndim != 3 or cube.shape[0] != t.size:
        rec["reject_reason"] = "no pixel cube"
        return rec
    q = sector.get("quality")
    good = np.isfinite(t)
    if q is not None:
        good &= np.asarray(q).astype(float) == 0
    good &= np.isfinite(cube).any(axis=(1, 2))
    ecube = sector.get("flux_err")
    ecube = np.asarray(ecube, dtype=float) if ecube is not None else None
    if ecube is not None and ecube.shape == cube.shape:
        ecube = ecube[good]
    else:
        ecube = None
    t, cube = t[good], cube[good]
    rec["n_cadences"] = int(t.size)
    rec["shape"] = f"{cube.shape[1]}x{cube.shape[2]}" if cube.ndim == 3 else ""
    exptime_s = float(sector.get("exptime_s") or np.nan)
    exp_d = exptime_s / 86400.0 if np.isfinite(exptime_s) else float("nan")
    in_t, oot, _dt = transit_cadence_masks(t, period_days=period_days, t0_btjd=t0_btjd,
                                           duration_days=duration_days, exptime_days=exp_d,
                                           params=params)
    rec["n_in_transit"], rec["n_out_of_transit"] = int(in_t.sum()), int(oot.sum())
    if rec["n_in_transit"]:
        ep, _ = fold(t[in_t], float(period_days), float(t0_btjd))
        rec["n_transits_covered"] = int(len(np.unique(ep)))
    if rec["n_in_transit"] < int(params.min_in_cadences):
        rec["reject_reason"] = "too_few_in_transit_cadences"
        return rec
    if rec["n_out_of_transit"] < int(params.min_oot_cadences):
        rec["reject_reason"] = "too_few_out_of_transit_cadences"
        return rec

    i_idx = np.flatnonzero(in_t)
    o_idx = np.flatnonzero(oot)
    in_img, in_sem, n_in = _mean_and_sem(cube, i_idx, ecube)
    oot_img, oot_sem, n_oot = _mean_and_sem(cube, o_idx, ecube)
    diff = oot_img - in_img
    diff_sem = np.sqrt(in_sem**2 + oot_sem**2)

    c_diff = image_centroid(diff, err_image=diff_sem, params=params)
    c_oot = image_centroid(oot_img, err_image=oot_sem, params=params)
    rec.update({"diff_x_px": c_diff["x"], "diff_y_px": c_diff["y"],
                "diff_x_err_px": c_diff["x_err"], "diff_y_err_px": c_diff["y_err"],
                "diff_peak": c_diff["peak"], "diff_n_pixels": c_diff["n_pixels"],
                "oot_x_px": c_oot["x"], "oot_y_px": c_oot["y"],
                "oot_x_err_px": c_oot["x_err"], "oot_y_err_px": c_oot["y_err"],
                "oot_n_pixels": c_oot["n_pixels"]})
    if c_diff["status"] != STATUS_OK or c_oot["status"] != STATUS_OK:
        rec["status"] = REASON_NO_CENTROID
        rec["reject_reason"] = ("no positive peak in the difference image"
                                if c_diff["status"] != STATUS_OK
                                else "no centroid in the out-of-transit image")
        return rec

    dx = c_diff["x"] - c_oot["x"]
    dy = c_diff["y"] - c_oot["y"]
    sx = math.sqrt(_nan0(c_diff["x_err"]) ** 2 + _nan0(c_oot["x_err"]) ** 2)
    sy = math.sqrt(_nan0(c_diff["y_err"]) ** 2 + _nan0(c_oot["y_err"]) ** 2)
    method = CENTROID_ERR_ANALYTIC
    want = str(params.err_method).lower()
    enough = (n_in >= int(params.min_cadences_for_bootstrap)
              and n_oot >= int(params.min_cadences_for_bootstrap))
    if want == "bootstrap" or (want == "auto" and enough):
        b = _bootstrap_offset(cube, i_idx, o_idx, params=params, rng=rng)
        if np.isfinite(b["x_err"]) and np.isfinite(b["y_err"]):
            sx, sy = b["x_err"], b["y_err"]
            method = CENTROID_ERR_BOOTSTRAP
    rec["centroid_err_method"] = method
    scale = float(rec["pixel_scale_arcsec"])
    off = math.hypot(dx, dy)
    stat_err = _magnitude_error(dx, dy, sx, sy)
    deb = _debias_magnitude(dx, dy, sx, sy)
    # The systematic floor goes in HERE, in arcseconds, so a per-sector offset
    # is never quoted more precisely than the centroid method can be trusted.
    floor_as = float(params.centroid_sys_floor_arcsec)
    err_as = (math.hypot(stat_err * scale, floor_as) if np.isfinite(stat_err)
              else (floor_as if floor_as > 0 else float("nan")))
    off_err = err_as / scale if scale > 0 else float("nan")
    rec.update({"offset_x_px": float(dx), "offset_y_px": float(dy), "offset_px": float(off),
                "offset_err_px": float(off_err), "offset_err_stat_px": float(stat_err),
                "centroid_sys_floor_arcsec": floor_as,
                "offset_debiased_px": float(deb),
                "offset_debiased_arcsec": float(deb * scale),
                "offset_arcsec": float(off * scale),
                "offset_err_arcsec": float(err_as),
                "offset_sigma": float(off * scale / err_as) if np.isfinite(err_as)
                and err_as > 0 else float("nan"),
                "status": STATUS_OK})

    # The file's own WCS, when the fetch supplied one: a pixel offset becomes a
    # sky offset, which is the only form that can be combined across sectors
    # (two sectors observe the same star at different roll angles).
    sky_fn = sector.get("sky_fn")
    if callable(sky_fn):
        try:
            ra_d, dec_d = [float(v) for v in sky_fn(c_diff["x"], c_diff["y"])]
            ra_o, dec_o = [float(v) for v in sky_fn(c_oot["x"], c_oot["y"])]
            rec["offset_ra_arcsec"] = float((ra_d - ra_o) * math.cos(math.radians(dec_o)) * 3600.0)
            rec["offset_dec_arcsec"] = float((dec_d - dec_o) * 3600.0)
            rec["offset_pa_deg"] = position_angle_deg(ra_o, dec_o, ra_d, dec_d)
        except Exception:                                 # noqa: BLE001
            pass
    tp = sector.get("target_pixel")
    if tp is not None and len(tp) == 2 and all(np.isfinite([float(tp[0]), float(tp[1])])):
        d = math.hypot(c_diff["x"] - float(tp[0]), c_diff["y"] - float(tp[1]))
        rec["offset_from_target_px"] = float(d)
        rec["offset_from_target_arcsec"] = float(d * scale)
    return rec


def _nan0(v) -> float:
    x = float(v)
    return x if np.isfinite(x) else 0.0


def _bootstrap_offset(cube: np.ndarray, i_idx: np.ndarray, o_idx: np.ndarray, *,
                      params: DiffParams, rng: np.random.Generator | None = None) -> dict:
    """Resample **cadences** and re-measure: the error the photometry actually supports."""
    g = rng or np.random.default_rng(int(params.seed))
    draws = max(int(params.bootstrap_draws), 2)
    dxs, dys = [], []
    for _ in range(draws):
        ii = g.integers(0, i_idx.size, size=i_idx.size)
        oo = g.integers(0, o_idx.size, size=o_idx.size)
        in_img = np.nanmean(cube[i_idx[ii]], axis=0)
        oot_img = np.nanmean(cube[o_idx[oo]], axis=0)
        cd = image_centroid(oot_img - in_img, params=params)
        co = image_centroid(oot_img, params=params)
        if cd["status"] != STATUS_OK or co["status"] != STATUS_OK:
            continue
        dxs.append(cd["x"] - co["x"])
        dys.append(cd["y"] - co["y"])
    if len(dxs) < 3:
        return {"x_err": float("nan"), "y_err": float("nan"), "n_draws": len(dxs)}
    return {"x_err": float(np.std(dxs, ddof=1)), "y_err": float(np.std(dys, ddof=1)),
            "n_draws": len(dxs)}


def combine_sector_offsets(rows, *, params: DiffParams | None = None) -> dict:
    """Combine the per-sector offsets --- in the SKY frame when the WCS allows it.

    Two sectors observe the same star at different roll angles, so their pixel
    ``(dx, dy)`` are not the same quantity and may not be averaged as vectors.
    When every usable sector carries a sky offset the combination is a genuine
    inverse-variance **vector** mean in ``(dRA cos dec, dDec)``; otherwise the
    scalar magnitudes are combined, ``combined_from_magnitudes`` is set, and the
    **debiased** lengths are used --- a raw length is positive by construction,
    so averaging several of them shrinks the error without shrinking the bias
    and manufactures a significant offset out of noise.  The systematic floor
    is re-applied to the combined error for the same reason: stacking sectors
    averages down what is random, not what every sector shares.
    """
    params = params or DiffParams()
    rows = [r for r in (rows or []) if r.get("status") == STATUS_OK
            and np.isfinite(r.get("offset_px", np.nan))
            and np.isfinite(r.get("offset_err_px", np.nan)) and r["offset_err_px"] > 0]
    out = {"n_sectors_used": int(len(rows)), "combined_from_magnitudes": True,
           "offset_arcsec": float("nan"), "offset_err_arcsec": float("nan"),
           "offset_sigma": float("nan"), "offset_ra_arcsec": float("nan"),
           "offset_dec_arcsec": float("nan"), "offset_pa_deg": float("nan"),
           "sector_scatter_chi2": float("nan"), "sector_scatter_dof": 0,
           "sector_scatter_chi2_per_dof": float("nan"), "sector_offsets_consistent": False,
           "sectors_used": "",
           "centroid_sys_floor_arcsec": float(params.centroid_sys_floor_arcsec)}
    if not rows:
        return out
    out["sectors_used"] = ",".join(str(r.get("sector")) for r in rows)
    scale = float(rows[0].get("pixel_scale_arcsec") or TESS_PIXEL_ARCSEC)
    sky = all(np.isfinite(r.get("offset_ra_arcsec", np.nan))
              and np.isfinite(r.get("offset_dec_arcsec", np.nan)) for r in rows)
    e = np.array([float(r["offset_err_px"]) * float(r.get("pixel_scale_arcsec") or scale)
                  for r in rows], dtype=float)
    w = 1.0 / e**2
    #: Stacking sectors averages down the STATISTICAL error; it does not touch a
    #: systematic that every sector shares, so the floor is re-applied here.
    floor = float(params.centroid_sys_floor_arcsec)
    out["centroid_sys_floor_arcsec"] = floor
    if sky:
        out["combined_from_magnitudes"] = False
        ra = np.array([float(r["offset_ra_arcsec"]) for r in rows])
        de = np.array([float(r["offset_dec_arcsec"]) for r in rows])
        mra = float(np.sum(w * ra) / np.sum(w))
        mde = float(np.sum(w * de) / np.sum(w))
        serr = max(float(1.0 / math.sqrt(np.sum(w))), floor)
        off = math.hypot(mra, mde)
        out.update({"offset_ra_arcsec": mra, "offset_dec_arcsec": mde,
                    "offset_arcsec": off, "offset_err_arcsec": serr,
                    "offset_sigma": off / serr if serr > 0 else float("nan"),
                    "offset_pa_deg": float(math.degrees(math.atan2(mra, mde)) % 360.0)})
        resid = ((ra - mra) ** 2 + (de - mde) ** 2) / e**2
        chi2, dof = float(np.sum(resid)), int(2 * len(rows) - 2)
    else:
        # The DEBIASED magnitudes: averaging raw lengths shrinks the error
        # without shrinking the positive bias, which would manufacture a
        # significant offset from noise alone.
        d = np.array([float(r.get("offset_debiased_px", r["offset_px"]))
                      * float(r.get("pixel_scale_arcsec") or scale) for r in rows], dtype=float)
        mean = float(np.sum(w * d) / np.sum(w))
        serr = max(float(1.0 / math.sqrt(np.sum(w))), floor)
        out.update({"offset_arcsec": mean, "offset_err_arcsec": serr,
                    "offset_sigma": mean / serr if serr > 0 else float("nan")})
        chi2 = float(np.sum(((d - mean) / e) ** 2))
        dof = int(max(len(d) - 1, 0))
    if dof > 0:
        out.update({"sector_scatter_chi2": chi2, "sector_scatter_dof": dof,
                    "sector_scatter_chi2_per_dof": chi2 / dof,
                    "sector_offsets_consistent": bool(
                        chi2 / dof < float(params.sector_scatter_chi2_per_dof))})
    return out


# ---------------------------------------------------------------------------
# 3. The verdict
# ---------------------------------------------------------------------------
def attribute_offset(census: pd.DataFrame | None, offset_arcsec: float, offset_err_arcsec: float,
                     *, pa_deg: float = float("nan"), n_sigma: float = 3.0) -> dict:
    """Which census source, if any, sits where the difference image points."""
    out = {"attributed_source_id": "", "attributed_sep_arcsec": float("nan"),
           "attributed_delta_g": float("nan"), "attributed_required_depth_ppm": float("nan"),
           "attributed_pa_deg": float("nan"), "n_sources_at_that_separation": 0,
           "attribution_note": ""}
    if census is None or not len(census) or not np.isfinite(offset_arcsec):
        out["attribution_note"] = "no census to attribute the offset to"
        return out
    err = float(offset_err_arcsec) if np.isfinite(offset_err_arcsec) else 0.0
    tol = max(float(n_sigma) * err, 1e-9)
    c = census[~census["is_target"].astype(bool)].copy()
    if not len(c):
        out["attribution_note"] = "the census holds no neighbour to attribute the offset to"
        return out
    c["_dsep"] = (pd.to_numeric(c["sep_arcsec"], errors="coerce") - float(offset_arcsec)).abs()
    near = c[c["_dsep"] <= tol]
    out["n_sources_at_that_separation"] = int(len(near))
    if not len(near):
        out["attribution_note"] = (f"no Gaia source sits at {offset_arcsec:.1f} +/- "
                                   f"{tol:.1f}\" from the target: the offset does not land on a "
                                   "catalogued neighbour")
        return out
    if np.isfinite(pa_deg) and "pa_deg" in near:
        d = (pd.to_numeric(near["pa_deg"], errors="coerce") - float(pa_deg)).abs() % 360.0
        d = np.minimum(d, 360.0 - d)
        near = near.assign(_dpa=d).sort_values(["_dpa", "_dsep"], kind="stable")
    else:
        near = near.sort_values("_dsep", kind="stable")
    b = near.iloc[0]
    out.update({"attributed_source_id": str(b["source_id"]),
                "attributed_sep_arcsec": float(b["sep_arcsec"]),
                "attributed_delta_g": float(b["delta_g_mag"]),
                "attributed_required_depth_ppm": float(b["required_depth_ppm"]),
                "attributed_pa_deg": float(b["pa_deg"]),
                "attribution_note": (
                    "the offset lands on this Gaia source; its required depth says whether it "
                    "can actually supply the dip"
                    + (" --- it CANNOT (excluded by arithmetic)"
                       if bool(b["excluded_by_arithmetic"]) else ""))})
    return out


def centroid_verdict(offset: dict | None, census_summary: dict | None, *,
                     params: DiffParams | None = None, census: pd.DataFrame | None = None,
                     reason: str | None = None) -> dict:
    """``TRANSIT_ON_TARGET`` / ``TRANSIT_OFFSET_FROM_TARGET`` / ``OFFSET_UNRESOLVED`` / ``NO_DATA_REACHED``.

    The precision requirement is the point of this function.  An offset whose
    error bar spans zero says "on target" only if the measurement could have
    **seen** the offset that matters --- ``n_offset * sigma`` must be no larger
    than the separation of the nearest neighbour the census could not exclude
    (or ``on_target_max_offset_arcsec`` when the census named none).  Otherwise
    the answer is ``OFFSET_UNRESOLVED``, never ``TRANSIT_ON_TARGET``.
    """
    params = params or DiffParams()
    cs = census_summary or {}
    out: dict = {
        "verdict": NO_DATA, "reason": reason or REASON_NOT_ATTEMPTED,
        "offset_arcsec": float("nan"), "offset_err_arcsec": float("nan"),
        "offset_sigma": float("nan"), "n_offset": float(params.n_offset),
        "offset_detectable_arcsec": float("nan"),
        "resolution_required_arcsec": float("nan"),
        "n_sectors_used": int((offset or {}).get("n_sectors_used", 0) or 0),
        "combined_from_magnitudes": bool((offset or {}).get("combined_from_magnitudes", True)),
        "sector_scatter_chi2_per_dof": float((offset or {}).get("sector_scatter_chi2_per_dof",
                                                                float("nan"))),
        "sector_offsets_consistent": bool((offset or {}).get("sector_offsets_consistent", False)),
        "census_excludes_all_neighbours": bool(cs.get("census_excludes_all_neighbours", False)),
        "n_neighbours": int(cs.get("n_neighbours", 0) or 0),
        "n_neighbours_not_excluded": int(cs.get("n_neighbours_not_excluded", 0) or 0),
        "attributed_source_id": "", "attributed_sep_arcsec": float("nan"),
        "attributed_delta_g": float("nan"), "attributed_required_depth_ppm": float("nan"),
        "attribution_note": "", "n_sources_at_that_separation": 0,
        "attributed_pa_deg": float("nan"),
        "verdict_statement": "",
    }
    if reason:
        out["verdict_statement"] = (f"no offset was measured ({reason}); the transit is NOT "
                                    "reported as being on the target")
        return out
    o = offset or {}
    d = float(o.get("offset_arcsec", float("nan")))
    e = float(o.get("offset_err_arcsec", float("nan")))
    out.update({"offset_arcsec": d, "offset_err_arcsec": e,
                "offset_sigma": float(o.get("offset_sigma", float("nan")))})
    if not (np.isfinite(d) and np.isfinite(e) and e > 0):
        out.update({"verdict": OFFSET_UNRESOLVED, "reason": REASON_NO_CENTROID,
                    "verdict_statement": "no centroid offset could be fitted; unresolved, "
                                         "which is NOT an on-target verdict"})
        return out
    detect = float(params.n_offset) * e
    req = float(cs.get("nearest_unexcluded_sep_arcsec", float("nan")))
    if not np.isfinite(req):
        req = float(params.on_target_max_offset_arcsec)
    out.update({"offset_detectable_arcsec": detect, "resolution_required_arcsec": req,
                "reason": ""})
    z = d / e
    if z >= float(params.n_offset):
        out["verdict"] = OFFSET_FROM_TARGET
        out.update(attribute_offset(census, d, e, pa_deg=float(o.get("offset_pa_deg", np.nan)),
                                    n_sigma=float(params.n_offset)))
        out["verdict_statement"] = (
            f"the transit source sits {d:.2f} +/- {e:.2f}\" from the out-of-transit centroid "
            f"({z:.1f} sigma) over {out['n_sectors_used']} sector(s): it is NOT on the target"
            + (f"; the offset lands on Gaia {out['attributed_source_id']} at "
               f"{out['attributed_sep_arcsec']:.1f}\"" if out["attributed_source_id"] else ""))
        return out
    if detect <= req:
        out["verdict"] = ON_TARGET
        out["verdict_statement"] = (
            f"the offset is {d:.2f} +/- {e:.2f}\" ({z:.1f} sigma), consistent with zero, and the "
            f"measurement could have detected {detect:.2f}\" --- smaller than the {req:.2f}\" "
            "that matters, so the transit is on the target")
        return out
    out.update({"verdict": OFFSET_UNRESOLVED, "reason": REASON_PRECISION})
    out["verdict_statement"] = (
        f"the offset is {d:.2f} +/- {e:.2f}\", consistent with zero, but this measurement could "
        f"only have detected {detect:.2f}\" and the neighbour that matters is at {req:.2f}\": "
        "the question is UNRESOLVED, not answered on target")
    return out


def run_verdict(verdicts: pd.DataFrame) -> tuple[str, str]:
    """``(verdict, reason)`` for the whole stage from the per-target verdicts."""
    if verdicts is None or not len(verdicts) or "verdict" not in verdicts:
        return NO_DATA, "no_targets"
    v = verdicts["verdict"].fillna(NO_DATA).astype(str)
    measured = v[v != NO_DATA]
    if not len(measured):
        reasons = sorted({str(r) for r in verdicts.get("reason", pd.Series(dtype=str)).fillna("")
                          if str(r)})
        return NO_DATA, "no_target_measured:" + ",".join(reasons)
    if (measured == OFFSET_FROM_TARGET).any():
        n = int((measured == OFFSET_FROM_TARGET).sum())
        return OFFSET_FROM_TARGET, f"{n} target(s) have the transit off the target star"
    if (measured == ON_TARGET).all():
        return ON_TARGET, f"all {len(measured)} measured target(s) have the transit on target"
    return OFFSET_UNRESOLVED, (f"{int((measured == OFFSET_UNRESOLVED).sum())} unresolved, "
                               f"{int((measured == ON_TARGET).sum())} on target")


# ---------------------------------------------------------------------------
# Synthetic pixel stamps (the test gate, and the runner's self-check)
# ---------------------------------------------------------------------------
def gaussian_prf(shape, x0: float, y0: float, sigma_px: float, flux: float = 1.0) -> np.ndarray:
    """A normalised Gaussian stand-in for the TESS PRF, on a pixel grid."""
    ny, nx = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:ny, 0:nx]
    s = max(float(sigma_px), 1e-6)
    g = np.exp(-(((xx - float(x0)) ** 2 + (yy - float(y0)) ** 2) / (2.0 * s * s)))
    tot = float(g.sum())
    return float(flux) * (g / tot if tot > 0 else g)


def synth_tpf(*, sources, period_days: float, t0_btjd: float, duration_days: float,
              depth: float, transit_source: int = 0, target_index: int = 0,
              shape=(11, 11), prf_sigma_px: float = 0.9, exptime_s: float = 120.0,
              n_transits: int = 6, span_durations: float = 2.5, noise: float = 0.0,
              background: float = 0.0, sector: int = 1, author: str = "SPOC",
              seed: int = 13, pixel_scale_arcsec: float = TESS_PIXEL_ARCSEC,
              ingress_frac: float = 0.1) -> dict:
    """A synthetic target pixel file with the transit injected on a **known** pixel.

    ``sources`` is a list of ``{"x", "y", "flux"}`` in pixel coordinates;
    ``transit_source`` indexes the one that dims.  This is the gate the offline
    suite runs: with the transit on the target the measured offset must be
    consistent with zero, and with it on a neighbour three pixels away the
    difference-image centroid must land on that neighbour.
    """
    rng = np.random.default_rng(int(seed))
    exp_d = float(exptime_s) / 86400.0
    half_span = float(span_durations) * float(duration_days)
    times = []
    for k in range(int(n_transits)):
        centre = float(t0_btjd) + k * float(period_days)
        n = max(int(round(2 * half_span / exp_d)), 8)
        times.append(centre + np.linspace(-half_span, half_span, n))
    t = np.concatenate(times)
    _, dt = fold(t, float(period_days), float(t0_btjd))
    half = 0.5 * float(duration_days)
    tau = max(float(ingress_frac) * float(duration_days), 1e-12)
    a = np.abs(dt)
    shp = np.zeros_like(a)
    shp[a <= (half - tau)] = 1.0
    sl = (a > (half - tau)) & (a < half)
    shp[sl] = (half - a[sl]) / tau
    stamps = [gaussian_prf(shape, s["x"], s["y"], prf_sigma_px, float(s.get("flux", 1.0)))
              for s in sources]
    cube = np.empty((t.size, int(shape[0]), int(shape[1])), dtype=float)
    for i in range(t.size):
        img = np.full(shape, float(background), dtype=float)
        for j, st in enumerate(stamps):
            img = img + st * (1.0 - float(depth) * shp[i] if j == int(transit_source) else 1.0)
        cube[i] = img
    if float(noise) > 0:
        cube = cube + rng.normal(0.0, float(noise), size=cube.shape)
    err = np.full(cube.shape, float(noise) if noise > 0 else np.nan)
    tgt = sources[int(target_index)]
    return {"sector": int(sector), "author": str(author), "exptime_s": float(exptime_s),
            "time": t, "flux": cube, "flux_err": err,
            "quality": np.zeros(t.size, dtype=int),
            "column": 0, "row": 0, "pixel_scale_arcsec": float(pixel_scale_arcsec),
            "target_pixel": (float(tgt["x"]), float(tgt["y"])),
            "n_cadences": int(t.size)}


# ---------------------------------------------------------------------------
# MAST access (runner only; injectable everywhere)
# ---------------------------------------------------------------------------
def tpf_probe() -> dict:
    """Which route to TESS target pixel files this machine has.  Imports only."""
    rep = {"lightkurve": {"importable": False}, "astroquery_mast": {"importable": False},
           "astropy_io_fits": {"importable": False}, "astropy_wcs": {"importable": False},
           "pyvo": {"importable": False}, "route_preferred": None}
    try:
        import lightkurve as lk  # noqa: PLC0415

        rep["lightkurve"] = {"importable": True, "version": str(getattr(lk, "__version__", "?"))}
    except Exception as exc:                              # noqa: BLE001
        rep["lightkurve"] = {"importable": False, "error": repr(exc)[:300]}
    try:
        import astroquery  # noqa: PLC0415
        from astroquery.mast import Observations  # noqa: PLC0415, F401

        rep["astroquery_mast"] = {"importable": True,
                                  "version": str(getattr(astroquery, "__version__", "?"))}
    except Exception as exc:                              # noqa: BLE001
        rep["astroquery_mast"] = {"importable": False, "error": repr(exc)[:300]}
    for key, mod in (("astropy_io_fits", "astropy.io.fits"), ("astropy_wcs", "astropy.wcs"),
                     ("pyvo", "pyvo")):
        try:
            __import__(mod)
            rep[key] = {"importable": True}
        except Exception as exc:                          # noqa: BLE001
            rep[key] = {"importable": False, "error": repr(exc)[:300]}
    if rep["lightkurve"]["importable"]:
        rep["route_preferred"] = ROUTE_LIGHTKURVE
    elif rep["astroquery_mast"]["importable"] and rep["astropy_io_fits"]["importable"]:
        rep["route_preferred"] = ROUTE_MAST_FITS
    rep["note"] = ("import reachability only; whether MAST ANSWERS is established by the "
                   "difference stage and recorded per target as OK / QUERY_FAILED / "
                   "QUERY_RETURNED_ZERO_ROWS.  pyvo is the Gaia census route.")
    return rep


def _sky_fn_from_wcs(wcs):
    """``(x, y) -> (ra, dec)`` from an astropy WCS, or ``None`` if it cannot be built."""
    if wcs is None:
        return None

    def _fn(x, y):
        sky = wcs.all_pix2world([[float(x), float(y)]], 0)
        return float(sky[0][0]), float(sky[0][1])

    try:
        _fn(0.0, 0.0)
    except Exception:                                     # noqa: BLE001
        return None
    return _fn


def _target_pixel_from_wcs(wcs, ra: float, dec: float):
    if wcs is None or not (np.isfinite(ra) and np.isfinite(dec)):
        return None
    try:
        px = wcs.all_world2pix([[float(ra), float(dec)]], 0)
        return float(px[0][0]), float(px[0][1])
    except Exception:                                     # noqa: BLE001
        return None


def read_tess_tpf_fits(path, *, ra: float = float("nan"), dec: float = float("nan")) -> dict | None:
    """One TESS target pixel FITS product -> the sector dict this module consumes.

    ``TIME`` is carried through the file's own ``BJDREFI``/``BJDREFF`` rather
    than assumed to be BTJD (exactly as stage 2 reads a light curve), and the
    ``APERTURE`` extension's WCS --- when astropy can build one --- supplies the
    ``sky_fn`` and the target's pixel position, without which a pixel offset
    cannot be turned into a sky offset or attributed to a neighbour.
    """
    from astropy.io import fits  # noqa: PLC0415

    with fits.open(str(path), memmap=False) as hdul:
        hdu = None
        for h in hdul:
            try:
                names = [str(n).upper() for n in h.columns.names]
            except AttributeError:
                continue
            if "TIME" in names and "FLUX" in names:
                hdu = h
                break
        if hdu is None:
            return None
        data, hdr = hdu.data, hdu.header
        cols = [str(n).upper() for n in hdu.columns.names]
        bjdrefi = float(hdr.get("BJDREFI", hdul[0].header.get("BJDREFI", BTJD_OFFSET))
                        or BTJD_OFFSET)
        bjdreff = float(hdr.get("BJDREFF", hdul[0].header.get("BJDREFF", 0.0)) or 0.0)
        t = np.asarray(data["TIME"], dtype=float) + (bjdrefi + bjdreff) - BTJD_OFFSET
        cube = np.asarray(data["FLUX"], dtype=float)
        err = (np.asarray(data["FLUX_ERR"], dtype=float) if "FLUX_ERR" in cols else None)
        q = np.asarray(data["QUALITY"]) if "QUALITY" in cols else None
        exptime = hdr.get("TIMEDEL")
        exptime_s = (float(exptime) * 86400.0 if exptime and np.isfinite(float(exptime))
                     else float("nan"))
        if not np.isfinite(exptime_s):
            ft, nf = hdr.get("FRAMETIM"), hdr.get("NUM_FRM")
            if ft and nf:
                exptime_s = float(ft) * float(nf)
        wcs = None
        try:
            from astropy.wcs import WCS  # noqa: PLC0415

            for name in ("APERTURE", 2):
                try:
                    wcs = WCS(hdul[name].header)
                    break
                except Exception:                         # noqa: BLE001
                    continue
        except Exception:                                 # noqa: BLE001
            wcs = None
        rec = {"sector": (int(hdul[0].header["SECTOR"])
                          if hdul[0].header.get("SECTOR") is not None else None),
               "author": str(hdul[0].header.get("PROCVER", "")
                             or hdul[0].header.get("ORIGIN", "") or "unknown"),
               "exptime_s": exptime_s, "time": t, "flux": cube, "flux_err": err, "quality": q,
               "column": hdr.get("1CRV5P") or hdr.get("CRVAL1P"),
               "row": hdr.get("2CRV5P") or hdr.get("CRVAL2P"),
               "pixel_scale_arcsec": TESS_PIXEL_ARCSEC, "n_cadences": int(t.size)}
        fn = _sky_fn_from_wcs(wcs)
        if fn is not None:
            rec["sky_fn"] = fn
            tp = _target_pixel_from_wcs(wcs, ra, dec)
            if tp is not None:
                rec["target_pixel"] = tp
        return rec


def lightkurve_tpf_fn(tic_id, *, authors=DEFAULT_TPF_AUTHORS, sectors=(), max_sectors: int = 8,
                      download_dir: str | None = None, ra: float = float("nan"),
                      dec: float = float("nan"), **_kw) -> list[dict]:
    """Target pixel files for one TIC through ``lightkurve`` (runner only)."""
    import lightkurve as lk  # noqa: PLC0415

    sr = lk.search_targetpixelfile(f"TIC {int(tic_id)}", mission="TESS")
    if sr is None or len(sr) == 0:
        return []
    try:
        tbl = sr.table
        keep = np.isin(np.array([str(a) for a in tbl["author"]]), list(authors))
        if sectors:
            want = {int(s) for s in sectors}
            keep &= np.array([int(v) in want for v in tbl["sequence_number"]])
        if keep.any():
            sr = sr[keep]
    except Exception:                                     # noqa: BLE001
        pass
    out: list[dict] = []
    for i in range(min(len(sr), int(max_sectors))):
        try:
            tpf = sr[i].download(download_dir=download_dir)
        except Exception:                                 # noqa: BLE001
            continue
        if tpf is None:
            continue
        try:
            t = np.asarray(tpf.time.jd, dtype=float) - BTJD_OFFSET
        except Exception:                                 # noqa: BLE001
            t = np.asarray(tpf.time.value, dtype=float)
        cube = np.asarray(getattr(tpf.flux, "value", tpf.flux), dtype=float)
        err = np.asarray(getattr(getattr(tpf, "flux_err", None), "value",
                                 np.full(cube.shape, np.nan)), dtype=float)
        meta = dict(getattr(tpf, "meta", {}) or {})
        exptime = meta.get("TIMEDEL")
        exptime_s = float(exptime) * 86400.0 if exptime else float("nan")
        if not np.isfinite(exptime_s) and t.size > 2:
            exptime_s = float(np.median(np.diff(np.sort(t)))) * 86400.0
        rec = {"sector": int(meta.get("SECTOR")) if meta.get("SECTOR") else None,
               "author": str(meta.get("AUTHOR") or meta.get("ORIGIN") or "unknown"),
               "exptime_s": exptime_s, "time": t, "flux": cube, "flux_err": err,
               "quality": np.asarray(getattr(tpf, "quality", np.zeros(t.size))),
               "column": getattr(tpf, "column", None), "row": getattr(tpf, "row", None),
               "pixel_scale_arcsec": TESS_PIXEL_ARCSEC, "n_cadences": int(t.size)}
        wcs = getattr(tpf, "wcs", None)
        fn = _sky_fn_from_wcs(wcs)
        if fn is not None:
            rec["sky_fn"] = fn
            tp = _target_pixel_from_wcs(wcs, ra, dec)
            if tp is not None:
                rec["target_pixel"] = tp
        out.append(rec)
    return out


def mast_fits_tpf_fn(tic_id, *, authors=DEFAULT_TPF_AUTHORS, sectors=(), max_sectors: int = 8,
                     download_dir: str | None = None, ra: float = float("nan"),
                     dec: float = float("nan"), **_kw) -> list[dict]:
    """Target pixel files through ``astroquery.mast`` + FITS (the fallback route)."""
    from astroquery.mast import Observations  # noqa: PLC0415

    obs = Observations.query_criteria(obs_collection="TESS", dataproduct_type="timeseries",
                                      target_name=str(int(tic_id)))
    if obs is None or len(obs) == 0:
        return []
    prod = Observations.get_product_list(obs)
    if prod is None or len(prod) == 0:
        return []
    p = prod.to_pandas()
    if "productSubGroupDescription" in p.columns:
        p = p[p["productSubGroupDescription"].astype(str).str.upper() == "TP"]
    if "provenance_name" in p.columns and len(authors):
        want = {str(a).upper() for a in authors}
        sel = p["provenance_name"].astype(str).str.upper().isin(want)
        if sel.any():
            p = p[sel]
    p = p.head(int(max_sectors))
    root = Path(download_dir or ".") / "mast_tpf"
    root.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for _, r in p.iterrows():
        uri = str(r.get("dataURI") or "")
        if not uri:
            continue
        local = root / str(r.get("productFilename") or Path(uri).name)
        try:
            status, _msg, _url = Observations.download_file(uri, local_path=str(local),
                                                            cache=False)
            if str(status).upper() != "COMPLETE" or not local.exists():
                continue
            rec = read_tess_tpf_fits(local, ra=ra, dec=dec)
        except Exception:                                 # noqa: BLE001
            continue
        if rec is None:
            continue
        prov = str(r.get("provenance_name") or "").strip()
        if prov:
            rec["author"] = prov
        if sectors and rec.get("sector") is not None and int(rec["sector"]) not in {
                int(s) for s in sectors}:
            continue
        out.append(rec)
    return out


def default_tpf_fn(tic_id, **kw) -> list[dict]:
    """``lightkurve`` if it is installed, else ``astroquery.mast`` + FITS."""
    try:
        import lightkurve  # noqa: PLC0415, F401

        return lightkurve_tpf_fn(tic_id, **kw)
    except ImportError:
        return mast_fits_tpf_fn(tic_id, **kw)


def fetch_target_pixel_files(tic_id, *, tpf_fn=None, params: TpfParams | None = None,
                             log: AcquisitionLog | None = None,
                             deadline: Deadline | None = None, key: str = "",
                             ra: float = float("nan"), dec: float = float("nan")
                             ) -> tuple[list[dict], str, str]:
    """One target's TESS target pixel files --- **bounded, capped and recorded**.

    Returns ``(sectors, status, route)``; ``status`` is ``OK`` /
    ``QUERY_RETURNED_ZERO_ROWS`` / ``QUERY_FAILED`` and a failure is never
    turned into an empty-but-successful answer.  ``params.max_sectors`` caps how
    many sectors are downloaded at all (target pixel files are tens of MB each)
    and both the per-target and the stage-wide budgets stop the retries.
    """
    params = params or TpfParams()
    log = log or AcquisitionLog(prefix="growth/centroid")
    label = f"target_pixel_files_{key or tic_id}"
    if deadline is not None and deadline.expired():
        log.record(label, f"TIC {tic_id}", error="budget_exhausted_before_request")
        return [], STATUS_FAILED, ""
    route = "injected" if tpf_fn is not None else (tpf_probe().get("route_preferred") or "")
    fn = tpf_fn or default_tpf_fn
    own = Deadline(budget_s=float(params.per_target_budget_s))
    last = ""
    for attempt in range(max(int(params.retries), 1)):
        if own.expired() or (deadline is not None and deadline.expired()):
            last = "budget_exhausted"
            break
        if attempt and float(params.retry_pause_s) > 0:
            _time.sleep(min(float(params.retry_pause_s) * attempt, own.remaining(),
                            deadline.remaining() if deadline is not None else float("inf")))
        try:
            secs = fn(tic_id, authors=tuple(params.authors), sectors=tuple(params.sectors),
                      max_sectors=int(params.max_sectors), download_dir=params.download_dir,
                      ra=ra, dec=dec)
        except Exception as exc:                          # noqa: BLE001
            last = repr(exc)[:400]
            continue
        secs = [s for s in (secs or []) if s is not None
                and np.asarray(s.get("flux")).ndim == 3]
        secs = secs[:int(params.max_sectors)]
        if not secs:
            log.record(label, f"TIC {tic_id}", rows=0,
                       extra={"route": route, "attempt": attempt + 1})
            return [], STATUS_ZERO, route
        log.record(label, f"TIC {tic_id}",
                   rows=int(sum(int(np.asarray(s.get("time")).size) for s in secs)),
                   extra={"route": route, "n_sectors": len(secs), "attempt": attempt + 1,
                          "max_sectors": int(params.max_sectors)})
        return secs, STATUS_OK, route
    log.record(label, f"TIC {tic_id}", error=last or "unknown")
    return [], STATUS_FAILED, route


# ---------------------------------------------------------------------------
# Shortlist, depths and stage orchestration
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _s(v) -> str:
    """A CSV round trip turns an empty cell into NaN; ``str(nan)`` is ``"nan"``."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


def load_shortlist(conf: dict, *, stage2_dir: Path | None = None) -> pd.DataFrame:
    """The centroid shortlist: ``centroid.shortlist``, else stage 2's own.

    Keeping the fallback means the target stage 2 actually measured is the one
    whose transit gets located, without either module reaching into the other's
    results directory for its identifiers.
    """
    c = (conf or {}).get("centroid") or {}
    rows = [dict(e, shortlist_source="config") for e in (c.get("shortlist") or [])]
    if not rows and c.get("inherit_stage2_shortlist", True):
        cand = None
        if stage2_dir is not None:
            for p in (Path(stage2_dir) / "candidates.csv",
                      Path(stage2_dir).parent / "candidates.csv"):
                if p.exists():
                    cand = p
                    break
        s2 = stage2_shortlist(conf, candidates_csv=cand)
        rows = s2.to_dict(orient="records") if len(s2) else []
    if not rows:
        return pd.DataFrame(columns=["kepoi_name", "kepler_name", "kepid", "tic_id", "toi",
                                     "shortlist_source"])
    df = pd.DataFrame(rows)
    df["kepoi_name"] = df["kepoi_name"].map(str)
    return df.drop_duplicates(subset=["kepoi_name"], keep="first").reset_index(drop=True)


def load_measured_depths(path) -> dict:
    """``{kepoi_name: (depth_ppm, err_ppm)}`` from stage 2's ``measurements.csv``.

    The census needs the depth **the aperture actually shows**, which is stage
    2's fitted number (30,614 ppm for K00897.01), not a catalogue value: it is
    the aperture depth that a neighbour has to be able to supply.
    """
    p = Path(path)
    if not p.exists() or not p.stat().st_size:
        return {}
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception:                                     # noqa: BLE001
        return {}
    out: dict[str, tuple[float, float]] = {}
    for _, r in df.iterrows():
        k = str(r.get("kepoi_name"))
        d = _f(r.get("depth_measured_ppm"))
        if k and np.isfinite(d):
            out[k] = (d, _f(r.get("depth_measured_err_ppm")))
    return out


def centroid_probe(conf: dict, out: Path) -> dict:
    """Which routes to MAST and Gaia exist on this machine (imports only)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sl = load_shortlist(conf, stage2_dir=_stage2_dir(out))
    rep = {"stage": "probe", "generated_utc": _now(), "mast": tpf_probe(),
           "gaia_tap": CensusParams.from_config(conf).tap_url,
           "shortlist_size": int(len(sl)),
           "shortlist": sl.to_dict(orient="records")}
    _write(out / "probe.json", rep)
    print(f"[growth-centroid] probe: TPF route = {rep['mast'].get('route_preferred')}; "
          f"shortlist {rep['shortlist_size']}")
    return rep


def _stage2_dir(out: Path) -> Path:
    """Where stage 2 wrote its measurements (``results/growth/stage2`` by default)."""
    p = Path(out)
    for cand in (p.parent / "stage2", p / "stage2"):
        if cand.exists():
            return cand
    return p.parent / "stage2"


def _ephemerides(conf: dict, shortlist: pd.DataFrame, *, query_fn=None,
                 log: AcquisitionLog | None = None,
                 tpf: TpfParams | None = None) -> tuple[dict, str]:
    """KOI ``period`` / ``epoch`` / ``duration`` / ``ra`` / ``dec`` for the shortlist."""
    tpf = tpf or TpfParams()
    arch = (conf or {}).get("archive") or {}
    url = str(arch.get("exoarchive_tap")
              or "https://exoplanetarchive.ipac.caltech.edu/TAP/sync")
    qf = query_fn or (lambda adql: tap_sync(adql, url=url, retries=int(tpf.archive_retries),
                                            timeout=float(tpf.archive_timeout_s)))
    koi, status = fetch_koi_ephemerides(shortlist.get("kepoi_name", pd.Series(dtype=str)),
                                        query_fn=qf, log=log)
    by = {str(r.get("kepoi_name")): r for r in koi.to_dict(orient="records")} if len(koi) else {}
    return by, status


def centroid_census_stage(conf: dict, out: Path, *, cone_fn=None, query_fn=None,
                          shortlist: pd.DataFrame | None = None,
                          log: AcquisitionLog | None = None) -> dict:
    """The Gaia census for every shortlisted target --- the test that needs no pixels."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = log or AcquisitionLog(prefix="growth/centroid")
    cp = CensusParams.from_config(conf)
    tp = TpfParams.from_config(conf)
    if shortlist is None:
        shortlist = load_shortlist(conf, stage2_dir=_stage2_dir(out))
    koi_by, koi_status = _ephemerides(conf, shortlist, query_fn=query_fn, log=log, tpf=tp)
    depths = load_measured_depths(_stage2_dir(out) / "measurements.csv")
    frames, targets, summaries = [], [], []
    for entry in shortlist.to_dict(orient="records"):
        k = str(entry.get("kepoi_name"))
        koi = koi_by.get(k) or {}
        ra, dec = _f(koi.get("ra")), _f(koi.get("dec"))
        if not np.isfinite(ra):
            ra, dec = _f(entry.get("ra")), _f(entry.get("dec"))
        depth, derr = depths.get(k, (_f(entry.get("observed_depth_ppm")),
                                     _f(entry.get("observed_depth_err_ppm"))))
        src, status = fetch_gaia_census(ra, dec, params=cp, cone_fn=cone_fn, log=log, key=k)
        census, summary = neighbour_census(src if status == STATUS_OK else None, ra=ra, dec=dec,
                                           observed_depth_ppm=depth,
                                           observed_depth_err_ppm=derr, params=cp,
                                           target_g_mag=_f(entry.get("target_g_mag")))
        summary.update({"kepoi_name": k, "tic_id": entry.get("tic_id"),
                        "gaia_status": status, "ra": ra, "dec": dec})
        if len(census):
            census.insert(0, "kepoi_name", k)
            frames.append(census)
        summaries.append(summary)
        targets.append({"kepoi_name": k, "kepler_name": entry.get("kepler_name"),
                        "kepid": entry.get("kepid"), "tic_id": entry.get("tic_id"),
                        "toi": entry.get("toi"), "ra": ra, "dec": dec,
                        "period_days": _f(koi.get("koi_period")),
                        "t0_bkjd": _f(koi.get("koi_time0bk")),
                        "t0_bkjd_err": _f(koi.get("koi_time0bk_err1")),
                        "period_err": _f(koi.get("koi_period_err1")),
                        "duration_hours": _f(koi.get("koi_duration")),
                        "observed_depth_ppm": depth, "observed_depth_err_ppm": derr,
                        "koi_ephemeris_status": (STATUS_OK if koi else
                                                 (koi_status if koi_status != STATUS_OK
                                                  else STATUS_ZERO))})
        print(f"[growth-centroid] census {k}: gaia={status}; {summary['census_statement']}")
    cdf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    cdf.to_csv(out / "census.csv", index=False)
    tdf = pd.DataFrame(targets)
    tdf.to_csv(out / "targets.csv", index=False)
    rep = {"stage": "census", "generated_utc": _now(), "n_shortlist": int(len(shortlist)),
           "koi_ephemeris_status": koi_status, "census": summaries,
           "acquisition": log.as_dict()}
    _write(out / "census.json", rep)
    log.write(out / "acquisition_log.json")
    return rep


def centroid_difference_stage(conf: dict, out: Path, *, tpf_fn=None, query_fn=None,
                              shortlist: pd.DataFrame | None = None,
                              log: AcquisitionLog | None = None) -> dict:
    """Fetch the target pixel files and measure the per-sector difference-image offset."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = log or AcquisitionLog(prefix="growth/centroid")
    tp = TpfParams.from_config(conf)
    dp = DiffParams.from_config(conf)
    if shortlist is None:
        shortlist = load_shortlist(conf, stage2_dir=_stage2_dir(out))
    tpath = out / "targets.csv"
    tinfo: dict[str, dict] = {}
    if tpath.exists() and tpath.stat().st_size:
        for _, r in pd.read_csv(tpath).iterrows():
            tinfo[str(r.get("kepoi_name"))] = r.to_dict()
    if not tinfo:
        koi_by, _s = _ephemerides(conf, shortlist, query_fn=query_fn, log=log, tpf=tp)
        for k, r in koi_by.items():
            tinfo[k] = {"ra": _f(r.get("ra")), "dec": _f(r.get("dec")),
                        "period_days": _f(r.get("koi_period")),
                        "t0_bkjd": _f(r.get("koi_time0bk")),
                        "duration_hours": _f(r.get("koi_duration"))}
    deadline = Deadline(budget_s=float(tp.budget_s) if tp.budget_s else None)
    rng = np.random.default_rng(int(dp.seed))
    sec_rows, combined = [], []
    for entry in shortlist.to_dict(orient="records"):
        k = str(entry.get("kepoi_name"))
        info = tinfo.get(k, {})
        period = _f(info.get("period_days"))
        t0_bkjd = _f(info.get("t0_bkjd"))
        dur_h = _f(info.get("duration_hours"))
        rec = {"kepoi_name": k, "tic_id": entry.get("tic_id"), "tpf_status": REASON_NOT_ATTEMPTED,
               "tpf_route": "", "n_sectors": 0, "n_sectors_measured": 0, "sector_list": "",
               "reason": ""}
        if not (np.isfinite(period) and period > 0 and np.isfinite(t0_bkjd)
                and np.isfinite(dur_h) and dur_h > 0):
            rec.update({"reason": REASON_NO_EPHEMERIS, "tpf_status": "not_attempted"})
            combined.append({**rec, **combine_sector_offsets([], params=dp)})
            continue
        secs, status, route = fetch_target_pixel_files(
            entry.get("tic_id"), tpf_fn=tpf_fn, params=tp, log=log, deadline=deadline, key=k,
            ra=_f(info.get("ra")), dec=_f(info.get("dec")))
        rec.update({"tpf_status": status, "tpf_route": route, "n_sectors": int(len(secs))})
        if status != STATUS_OK:
            rec["reason"] = (REASON_BUDGET if deadline.expired() and status == STATUS_FAILED
                             else status)
            combined.append({**rec, **combine_sector_offsets([], params=dp)})
            continue
        t_all = np.concatenate([np.asarray(s["time"], dtype=float) for s in secs])
        prop = propagate_epoch(float(bkjd_to_btjd(t0_bkjd)), period,
                               float(np.nanmedian(t_all)),
                               t0_err_days=abs(_f(info.get("t0_bkjd_err"))),
                               period_err_days=abs(_f(info.get("period_err"))))
        rows = []
        for s in secs:
            r = difference_image_offset(s, period_days=period, t0_btjd=prop["t0_btjd"],
                                        duration_days=dur_h / 24.0, params=dp, rng=rng)
            r.update({"kepoi_name": k, "tic_id": entry.get("tic_id")})
            rows.append(r)
            sec_rows.append(r)
        used = [r for r in rows if r.get("status") == STATUS_OK]
        rec.update({"n_sectors_measured": int(len(used)),
                    "sector_list": ",".join(str(r.get("sector")) for r in rows),
                    "t0_btjd_used": prop["t0_btjd"],
                    "n_epochs_propagated": prop["n_epochs"],
                    "ephemeris_sigma_minutes": prop["sigma_minutes"],
                    "period_days": period, "duration_hours": dur_h})
        if not used:
            rec["reason"] = REASON_NO_TRANSIT
        combined.append({**rec, **combine_sector_offsets(rows, params=dp)})
        print(f"[growth-centroid] difference {k}: tpf={status} route={route} "
              f"sectors={rec['n_sectors']} measured={rec['n_sectors_measured']}")
    sdf = pd.DataFrame(sec_rows)
    sdf.to_csv(out / "sectors.csv", index=False)
    cdf = pd.DataFrame(combined)
    cdf.to_csv(out / "offsets.csv", index=False)
    rep = {"stage": "difference", "generated_utc": _now(), "n_shortlist": int(len(shortlist)),
           "n_sector_measurements": int(len(sdf)),
           "tpf_status_counts": (cdf["tpf_status"].map(str).value_counts().to_dict()
                                 if len(cdf) else {}),
           "tpf_routes": (cdf["tpf_route"].map(str).value_counts().to_dict()
                          if len(cdf) else {}),
           "budget_s": tp.budget_s, "elapsed_s": round(deadline.elapsed(), 1),
           "budget_exhausted": bool(deadline.expired()),
           "max_sectors_per_target": int(tp.max_sectors),
           "acquisition": log.as_dict()}
    _write(out / "acquire.json", rep)
    log.write(out / "acquisition_log.json")
    print(f"[growth-centroid] difference: {rep['n_sector_measurements']} sector measurement(s) "
          f"in {rep['elapsed_s']}s (budget {tp.budget_s}s)")
    return rep


def centroid_assess(conf: dict, out: Path) -> dict:
    """Per-target verdicts from the census and the offsets, then ``summary.json``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dp = DiffParams.from_config(conf)
    cp = CensusParams.from_config(conf)
    census = _read_csv(out / "census.csv")
    offsets = _read_csv(out / "offsets.csv")
    census_json = _read_json(out / "census.json")
    acquire = _read_json(out / "acquire.json")
    by_key = {str(s.get("kepoi_name")): s for s in (census_json.get("census") or [])}
    rows = []
    keys = list(dict.fromkeys(
        [str(k) for k in offsets.get("kepoi_name", pd.Series(dtype=str))]
        + list(by_key)))
    for k in keys:
        o = offsets[offsets["kepoi_name"].astype(str) == k] if len(offsets) else offsets
        orec = o.iloc[0].to_dict() if len(o) else {}
        cs = by_key.get(k, {})
        sub = (census[census["kepoi_name"].astype(str) == k]
               if len(census) and "kepoi_name" in census else None)
        reason = _s(orec.get("reason")) or (REASON_NOT_ATTEMPTED if not orec else "")
        v = centroid_verdict(orec if not reason else None, cs, params=dp, census=sub,
                             reason=reason or None)
        rows.append({"kepoi_name": k, "tic_id": orec.get("tic_id") or cs.get("tic_id"),
                     **v,
                     "tpf_status": orec.get("tpf_status", ""),
                     "tpf_route": orec.get("tpf_route", ""),
                     "n_sectors": orec.get("n_sectors", 0),
                     "n_sectors_measured": orec.get("n_sectors_measured", 0),
                     "sector_list": orec.get("sector_list", ""),
                     "gaia_status": cs.get("gaia_status", ""),
                     "observed_depth_ppm": cs.get("observed_depth_ppm", float("nan")),
                     "contam_weighted": cs.get("contam_weighted", float("nan")),
                     "contam_max": cs.get("contam_max", float("nan")),
                     "min_delta_g_that_can_supply_depth":
                         cs.get("min_delta_g_that_can_supply_depth", float("nan")),
                     "census_statement": cs.get("census_statement", "")})
    vdf = pd.DataFrame(rows)
    vdf.to_csv(out / "verdicts.csv", index=False)
    verdict, reason = run_verdict(vdf)
    counts = {w: int((vdf["verdict"].astype(str) == w).sum()) if len(vdf) else 0
              for w in TARGET_VERDICTS}
    summary = {
        "verdict": verdict, "reason": reason, "generated_utc": _now(),
        "n_targets": int(len(vdf)), "target_verdicts": counts,
        "reasons": (vdf.loc[vdf["verdict"] != ON_TARGET, "reason"].fillna("").astype(str)
                    .value_counts().to_dict() if len(vdf) else {}),
        "targets": vdf.to_dict(orient="records") if len(vdf) else [],
        "census": census_json.get("census") or [],
        "acquisition": {k: acquire.get(k) for k in
                        ("tpf_status_counts", "tpf_routes", "budget_s", "elapsed_s",
                         "budget_exhausted", "max_sectors_per_target")},
        "config": {"census": {k: getattr(cp, k) for k in cp.__dataclass_fields__},
                   "difference": {k: getattr(dp, k) for k in dp.__dataclass_fields__}},
        "checks_not_performed": [
            "prf_fit_photocentre (the centroid is a thresholded first moment, not a fitted "
            "TESS PRF; a PRF fit would tighten the offset error, not change its sign)",
            "kepler_era_pixel_centroid (only the TESS pixels are tested here)",
            "gaia_sources_below_the_catalogue_limit (an unresolved source inside the pixel "
            "is not in the census and is therefore neither implicated nor excluded)",
        ],
        "note": ("the census arithmetic alone can exclude a neighbour: a source supplying "
                 "fraction f of the aperture flux must fade by depth/f, and above 100 % that "
                 "is impossible.  NO_DATA_REACHED is not a null result and is not written up "
                 "(CLAUDE.md); OFFSET_UNRESOLVED is never TRANSIT_ON_TARGET"),
    }
    _write(out / "summary.json", summary)
    print(f"[growth-centroid] assess: {verdict} — {reason}; {counts}")
    return summary


def _read_csv(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or not p.stat().st_size:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:                                     # noqa: BLE001
        return pd.DataFrame()


def _read_json(path: Path) -> dict:
    p = Path(path)
    if not p.exists() or not p.stat().st_size:
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:                                     # noqa: BLE001
        return {}


def centroid_run(stage: str = "all", *, out_dir=None, conf: dict | None = None, query_fn=None,
                 cone_fn=None, tpf_fn=None, shortlist: pd.DataFrame | None = None) -> dict:
    """Run one stage, a comma list, or all.  Returns the last stage's report."""
    from .run import load_growth_config  # noqa: PLC0415

    conf = conf if conf is not None else load_growth_config()
    out = Path(out_dir) if out_dir else Path("results") / "growth" / "centroid"
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage in ("all", "", None) else tuple(s.strip() for s in stage.split(","))
    rep: dict = {}
    for s in stages:
        if s == "probe":
            rep = centroid_probe(conf, out)
        elif s == "census":
            rep = centroid_census_stage(conf, out, cone_fn=cone_fn, query_fn=query_fn,
                                        shortlist=shortlist)
        elif s == "difference":
            rep = centroid_difference_stage(conf, out, tpf_fn=tpf_fn, query_fn=query_fn,
                                            shortlist=shortlist)
        elif s == "assess":
            rep = centroid_assess(conf, out)
        else:
            raise SystemExit(f"unknown stage {s!r}; choose from {STAGES}")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="seti growth-centroid",
        description="GROWTH stage 3 (S57): is the transit ON THE TARGET?  A Gaia neighbour "
                    "census with the depth each neighbour would need, and a TESS "
                    "difference-image centroid offset")
    p.add_argument("--stage", default="all", choices=(*STAGES, "all"))
    p.add_argument("--out-dir", default="results/growth/centroid", help="results directory")
    a = p.parse_args(argv)
    rep = centroid_run(a.stage, out_dir=a.out_dir)
    v = rep.get("verdict") if isinstance(rep, dict) else None
    if v:
        print(f"[growth-centroid] verdict: {v}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CENTROID_ERR_ANALYTIC", "CENTROID_ERR_BOOTSTRAP", "DEFAULT_TPF_AUTHORS", "NO_DATA",
    "OFFSET_FROM_TARGET", "OFFSET_UNRESOLVED", "ON_TARGET", "REASONS", "REASON_BUDGET",
    "REASON_NOT_ATTEMPTED", "REASON_NO_CENTROID", "REASON_NO_EPHEMERIS", "REASON_NO_TRANSIT",
    "REASON_PRECISION", "REASON_QUERY_FAILED", "REASON_ZERO_ROWS", "ROUTE_LIGHTKURVE",
    "ROUTE_MAST_FITS", "RUN_VERDICTS", "STAGES", "TARGET_VERDICTS", "TESS_PIXEL_ARCSEC",
    "CensusParams", "DiffParams", "TpfParams", "attribute_offset", "centroid_assess",
    "centroid_census_stage", "centroid_difference_stage", "centroid_probe", "centroid_run",
    "centroid_verdict", "combine_sector_offsets", "default_tpf_fn", "difference_image_offset",
    "fetch_gaia_census", "fetch_target_pixel_files", "gaussian_prf", "image_centroid",
    "lightkurve_tpf_fn", "load_measured_depths", "load_shortlist", "main", "mast_fits_tpf_fn",
    "neighbour_census", "position_angle_deg", "read_tess_tpf_fits", "run_verdict",
    "sky_sep_arcsec", "synth_tpf", "tpf_probe", "transit_cadence_masks",
]
