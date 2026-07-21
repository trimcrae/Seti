"""Runner-side acquisition + orchestration for the JWST biosignature analysis of
LHS 1140 b.

The scorers in :mod:`seti.jwst_bio.spectrum` are pure and offline-tested; this
module does the parts that need archive egress (so it runs on the GitHub runner,
not the sandbox):

1. resolve LHS 1140 b's live ephemeris + parameters from the NASA Exoplanet
   Archive ``pscomppars`` table (period, mid-transit, duration, radii, Teff,
   semi-major axis), reusing the TAP pattern from :mod:`seti.lhs1140.run`;
2. query MAST for the JWST time-series spectroscopic products (NIRISS/NIRSpec)
   and the MIRI products, and download a bounded number of extracted-spectra
   ``x1dints`` files (with a total-size cap);
3. build the transmission spectrum from the downloaded integrations, run every
   detector (molecular features, disequilibrium, abiotic gate, laser scan), and
   run the MIRI eclipse discriminant if eclipse data are present; and
4. write ``results/jwst_bio/lhs1140b.json`` + ``summary.json`` with an explicit
   ``verdict`` and a ``limitations`` field.

The channel **degrades honestly**: if the MAST download is not feasible on the
runner (too large, unavailable, or blocked), it still records what coverage was
reachable and writes a verdict of ``no_data`` -- it never fabricates a spectrum.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config, load_config
from .spectrum import (
    abiotic_false_positive,
    build_transmission_spectrum,
    disequilibrium_biosignature,
    eclipse_brightness_temperature,
    laser_line_scan,
    molecular_feature_detect,
    transit_mask_from_ephemeris,
)

# LHS 1140 anchor (ICRS J2000) and planet-b literature parameters (Cadieux+2024),
# a fallback if the live NASA Exoplanet Archive fetch fails.  ``a_rs`` is a/Rs.
LHS1140 = {"name": "LHS 1140", "ra": 11.2487, "dec": -15.2742,
           "gaia_dr3": 2371032916186181760}
_LHS1140B_FALLBACK = {
    "pl_name": "LHS 1140 b", "pl_orbper": 24.7369, "pl_tranmid": 2458226.843,
    "pl_trandur": 2.055, "pl_rade": 1.730, "st_rad": 0.2159, "st_teff": 3096.0,
    "pl_orbsmax": 0.0946, "pl_ratror": 0.0730, "a_rs": 94.4,
}

# Total download budget for x1dints products (bytes) and the per-file cap.
_MAX_TOTAL_BYTES = 1.5e9      # ~1.5 GB
_MAX_FILES = 12


def _fetch_planet_params(name: str = "LHS 1140 b") -> dict:
    """Live LHS 1140 b ephemeris + parameters from the NASA Exoplanet Archive.

    Reuses the ``pscomppars`` TAP pattern from :mod:`seti.lhs1140.run`.  Derives
    ``a_rs = a/Rs`` from ``pl_orbsmax`` (AU) and ``st_rad`` (R_sun) when the direct
    ``pl_ratdor`` column is absent, and ``rp_rs`` from ``pl_ratror`` or the radii.
    """
    params = dict(_LHS1140B_FALLBACK)
    try:
        import io

        import requests
        q = ("select pl_name,pl_orbper,pl_tranmid,pl_trandur,pl_rade,pl_bmasse,"
             "st_rad,st_teff,pl_orbsmax,pl_ratror,pl_ratdor,pl_eqt "
             f"from pscomppars where pl_name='{name}'")
        r = requests.get("https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
                         params={"query": q, "format": "csv"}, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if len(df):
            row = df.iloc[0]

            def val(k, default):
                v = row.get(k)
                num = pd.to_numeric(v, errors="coerce")
                return float(num) if num is not None and np.isfinite(num) else default

            params.update({
                "pl_orbper": val("pl_orbper", params["pl_orbper"]),
                "pl_tranmid": val("pl_tranmid", params["pl_tranmid"]),
                "pl_trandur": val("pl_trandur", params["pl_trandur"]),
                "pl_rade": val("pl_rade", params["pl_rade"]),
                "st_rad": val("st_rad", params["st_rad"]),
                "st_teff": val("st_teff", params["st_teff"]),
                "pl_orbsmax": val("pl_orbsmax", params["pl_orbsmax"]),
                "pl_ratror": val("pl_ratror", params["pl_ratror"]),
            })
            # a/Rs: prefer the direct ratio, else derive from a [AU] and Rs [Rsun].
            a_rs = val("pl_ratdor", np.nan)
            if not np.isfinite(a_rs):
                _AU_PER_RSUN = 0.00465047
                a_rs = params["pl_orbsmax"] / (params["st_rad"] * _AU_PER_RSUN)
            params["a_rs"] = float(a_rs)
            print(f"[jwst_bio] resolved LHS 1140 b: P={params['pl_orbper']:.4f} d "
                  f"T0={params['pl_tranmid']:.3f} dur={params['pl_trandur']:.3f} h "
                  f"a/Rs={params['a_rs']:.1f} Teff={params['st_teff']:.0f} K")
    except Exception as exc:  # noqa: BLE001
        print(f"[jwst_bio] planet-param fetch failed ({exc!r}); using fallback")
    # rp_rs from the ratio, else from the radii.
    _RE_PER_RSUN = 6.371e6 / 6.957e8
    rp_rs = params.get("pl_ratror")
    if not (isinstance(rp_rs, float) and np.isfinite(rp_rs) and rp_rs > 0):
        rp_rs = params["pl_rade"] * _RE_PER_RSUN / params["st_rad"]
    params["rp_rs"] = float(rp_rs)
    return params


def _query_mast_products(ra: float, dec: float, radius_arcsec: float = 20.0):
    """Return (products DataFrame, observations DataFrame) of JWST spectra at LHS 1140.

    Uses the coordinate cone (robust to the star's high proper motion / name
    resolution), filtered to JWST spectroscopic time-series, then expands to the
    extracted-spectra products.  Returns empty frames on any failure so the caller
    can degrade honestly.
    """
    try:
        from astroquery.mast import Observations
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except Exception as exc:  # noqa: BLE001
        print(f"[jwst_bio] astroquery/astropy unavailable: {exc!r}")
        return pd.DataFrame(), pd.DataFrame()

    try:
        obs = Observations.query_region(SkyCoord(ra, dec, unit="deg"),
                                        radius=radius_arcsec * u.arcsec)
    except Exception as exc:  # noqa: BLE001
        print(f"[jwst_bio] MAST query_region failed: {exc!r}")
        try:
            obs = Observations.query_object("LHS 1140",
                                            radius=f"{radius_arcsec} arcsec")
        except Exception as exc2:  # noqa: BLE001
            print(f"[jwst_bio] MAST query_object failed: {exc2!r}")
            return pd.DataFrame(), pd.DataFrame()
    if obs is None or len(obs) == 0:
        return pd.DataFrame(), pd.DataFrame()

    obs_df = obs.to_pandas()
    jwst = obs_df[(obs_df.get("obs_collection", "").astype(str).str.upper() == "JWST")
                  & (obs_df.get("dataproduct_type", "").astype(str).str.lower()
                     == "spectrum")]
    if not len(jwst):
        return pd.DataFrame(), obs_df
    try:
        # Expand only the JWST spectroscopic observations to their products.
        prod = Observations.get_product_list(
            obs[np.array((obs_df["obs_collection"].astype(str).str.upper()
                          == "JWST")
                         & (obs_df["dataproduct_type"].astype(str).str.lower()
                            == "spectrum"))])
        prod_df = prod.to_pandas()
    except Exception as exc:  # noqa: BLE001
        print(f"[jwst_bio] get_product_list failed: {exc!r}")
        return pd.DataFrame(), obs_df
    return prod_df, obs_df


def _select_x1dints(prod_df: pd.DataFrame) -> pd.DataFrame:
    """Filter a product list to extracted time-series spectra (``x1dints``)."""
    if not len(prod_df):
        return prod_df
    fn = prod_df.get("productFilename", pd.Series([], dtype=str)).astype(str)
    sub = prod_df.get("productSubGroupDescription",
                      pd.Series([""] * len(prod_df))).astype(str).str.upper()
    is_x1d = fn.str.contains("x1dints", case=False) | (sub == "X1DINTS")
    return prod_df[is_x1d].reset_index(drop=True)


def _download_and_read(prod_df: pd.DataFrame, out_dir) -> list[dict]:
    """Download a bounded set of x1dints and read them into flux(wavelength) stacks.

    Honours the total-size and file-count caps; skips a file rather than exceeding
    the budget.  Each returned entry is a dict with ``instrument``, ``wavelength``,
    ``flux`` (n_integrations x n_wavelength), ``flux_err`` and ``times`` (BJD).
    Returns an empty list on any failure (the channel then degrades to no_data).
    """
    x1d = _select_x1dints(prod_df)
    if not len(x1d):
        print("[jwst_bio] no x1dints products found")
        return []
    # Prefer the smallest files first to fit the most within the budget.
    if "size" in x1d.columns:
        x1d = x1d.sort_values("size").reset_index(drop=True)
    try:
        from astroquery.mast import Observations
    except Exception as exc:  # noqa: BLE001
        print(f"[jwst_bio] astroquery unavailable for download: {exc!r}")
        return []

    stacks: list[dict] = []
    total = 0.0
    dl_dir = out_dir / "x1dints"
    dl_dir.mkdir(parents=True, exist_ok=True)
    for _, row in x1d.iterrows():
        if len(stacks) >= _MAX_FILES:
            break
        size = float(row.get("size", 0) or 0)
        if size and total + size > _MAX_TOTAL_BYTES:
            print(f"[jwst_bio] size cap reached ({total/1e9:.2f} GB); stopping")
            break
        try:
            man = Observations.download_products(
                row.to_frame().T if hasattr(row, "to_frame") else row,
                download_dir=str(dl_dir))
            local = None
            if man is not None and len(man):
                local = man["Local Path"][0]
            if not local:
                continue
            stack = _read_x1dints(local)
            if stack is not None:
                stacks.append(stack)
                total += size
        except Exception as exc:  # noqa: BLE001
            print(f"[jwst_bio] product download/read failed: {exc!r}")
            continue
    print(f"[jwst_bio] read {len(stacks)} x1dints stacks "
          f"({total/1e9:.2f} GB)")
    return stacks


def _read_x1dints(path: str) -> dict | None:
    """Read a JWST ``x1dints`` FITS file into a flux(wavelength) stack.

    x1dints stores one extracted spectrum per integration in successive EXTRACT1D
    extensions (or a single table with an integration axis).  We read WAVELENGTH /
    FLUX / FLUX_ERROR columns and the integration mid-times.  Returns None if the
    structure is not recognised (the caller then skips it).
    """
    try:
        from astropy.io import fits
    except Exception as exc:  # noqa: BLE001
        print(f"[jwst_bio] astropy.io.fits unavailable: {exc!r}")
        return None
    try:
        with fits.open(path) as hdul:
            inst = str(hdul[0].header.get("INSTRUME", "unknown")).upper()
            wl, flux_rows, err_rows = None, [], []
            for hdu in hdul:
                if getattr(hdu, "data", None) is None:
                    continue
                cols = getattr(getattr(hdu, "columns", None), "names", None)
                if not cols:
                    continue
                names = {c.upper(): c for c in cols}
                if "WAVELENGTH" not in names or "FLUX" not in names:
                    continue
                w = np.asarray(hdu.data[names["WAVELENGTH"]], float).ravel()
                f = np.asarray(hdu.data[names["FLUX"]], float).ravel()
                if wl is None:
                    wl = w
                if len(f) != len(wl):
                    continue
                flux_rows.append(f)
                if "FLUX_ERROR" in names:
                    err_rows.append(np.asarray(hdu.data[names["FLUX_ERROR"]],
                                               float).ravel())
            if wl is None or len(flux_rows) < 2:
                return None
            flux = np.vstack(flux_rows)
            err = np.vstack(err_rows) if len(err_rows) == len(flux_rows) else None
            # Integration mid-times: INT_TIMES extension if present, else index.
            times = np.arange(flux.shape[0], dtype=float)
            try:
                it = hdul["INT_TIMES"].data
                key = [c for c in it.columns.names
                       if "BJD" in c.upper() and "MID" in c.upper()]
                if key:
                    times = np.asarray(it[key[0]], float)[:flux.shape[0]]
            except Exception:  # noqa: BLE001
                pass
            return {"instrument": inst, "wavelength": wl, "flux": flux,
                    "flux_err": err, "times": times, "path": path}
    except Exception as exc:  # noqa: BLE001
        print(f"[jwst_bio] FITS read failed for {path}: {exc!r}")
        return None


def _analyse_stack(stack: dict, params: dict) -> dict:
    """Build the transmission spectrum for one stack and run all detectors."""
    times = np.asarray(stack["times"], float)
    # If we have real BJD mid-times, use the ephemeris; otherwise fall back to a
    # phase-agnostic split (first/last quarters as out, middle as in) so a stack
    # without INT_TIMES still yields a (clearly-labelled, lower-confidence) depth.
    ephemeris_used = False
    if np.nanmax(times) - np.nanmin(times) > 0.01:  # spans >~15 min in days
        # pl_tranmid is BJD; if times look like BJD (>2.4e6) use it directly.
        t0 = params["pl_tranmid"]
        if np.nanmedian(times) < 2.4e6:
            t0 = t0 % params["pl_orbper"]
        mask = transit_mask_from_ephemeris(times, t0, params["pl_orbper"],
                                           params["pl_trandur"] / 24.0,
                                           fraction=0.9)
        ephemeris_used = bool(mask.any() and (~mask).any())
    if not ephemeris_used:
        n = stack["flux"].shape[0]
        mask = np.zeros(n, bool)
        mask[n // 3: 2 * n // 3] = True   # crude middle-as-in-transit fallback

    spec = build_transmission_spectrum(stack["wavelength"], stack["flux"], mask,
                                       flux_err=stack.get("flux_err"))
    detections = molecular_feature_detect(spec["wavelength"], spec["depth"],
                                          spec["depth_err"])
    diseq = disequilibrium_biosignature(detections)
    abiotic = abiotic_false_positive(detections)
    # Laser scan on the out-of-transit stellar spectrum (mean over out points).
    out_spec = np.nanmean(stack["flux"][~mask], axis=0)
    laser = laser_line_scan(stack["wavelength"], out_spec)
    return {
        "instrument": stack["instrument"], "path": stack.get("path"),
        "n_integrations": int(stack["flux"].shape[0]),
        "n_in_transit": int(mask.sum()), "n_out_transit": int((~mask).sum()),
        "ephemeris_used": ephemeris_used,
        "wavelength_range_um": [float(np.nanmin(stack["wavelength"])),
                                float(np.nanmax(stack["wavelength"]))],
        "detections": detections, "disequilibrium": diseq,
        "abiotic_gate": abiotic, "laser_scan": laser,
    }


def _combine_verdict(analyses: list[dict], eclipse: dict | None) -> dict:
    """Roll the per-stack analyses + eclipse discriminant into one verdict."""
    any_bio = any(a["disequilibrium"]["is_biosignature"] for a in analyses)
    any_laser = any(a["laser_scan"]["laser_line_flag"] for a in analyses)
    # A biosignature claim is only robust if a pair passes AND (for oxygen pairs)
    # the abiotic gate does not fire.
    robust = False
    for a in analyses:
        d = a["disequilibrium"]
        if not d["is_biosignature"]:
            continue
        if d.get("oxygen_involved") and a["abiotic_gate"]["abiotic_flag"]:
            continue
        robust = True
    if not analyses:
        verdict = "no_data"
    elif robust:
        verdict = "BIOSIGNATURE_CANDIDATE_REVIEW"
    elif any_bio:
        verdict = "disequilibrium_pair_but_abiotic_gate_fired"
    elif any_laser:
        verdict = "LASER_LINE_CANDIDATE_REVIEW"
    else:
        verdict = "no_biosignature_detected"
    return {"verdict": verdict, "any_disequilibrium_pair": any_bio,
            "robust_biosignature": robust, "any_laser_line": any_laser,
            "eclipse_classification": (eclipse.get("classification")
                                       if eclipse else None)}


def _maybe_eclipse(obs_df: pd.DataFrame, params: dict) -> dict | None:
    """Run the MIRI eclipse discriminant if MIRI photometry/eclipse data exist.

    We do not download the MIRI light curve here (a full eclipse fit is out of
    scope); instead, if MIRI data are present we record that the discriminant is
    *applicable* and evaluate it on the literature/placeholder eclipse depth so the
    classification path is exercised.  A real run wires ``eclipse_depth_ppm`` from
    a measured MIRI secondary eclipse -- absent that, this is flagged as
    ``not_measured``.
    """
    if not len(obs_df):
        return None
    has_miri = obs_df.get("instrument_name", pd.Series([], dtype=str)) \
        .astype(str).str.upper().str.contains("MIRI").any()
    if not has_miri:
        return None
    # No measured eclipse depth is fetched here -> record applicability honestly.
    return {"miri_present": True, "eclipse_depth_ppm": None,
            "classification": "not_measured",
            "note": ("MIRI data present; secondary-eclipse depth not extracted in "
                     "this detection-level pass -- eclipse_brightness_temperature "
                     "is ready to classify once a measured depth is supplied "
                     f"(a/Rs={params['a_rs']:.1f}, Teff={params['st_teff']:.0f} K)")}


def jwst_bio_run(cfg: Config | None = None) -> dict:
    """Acquire LHS 1140 b JWST spectra and run the full biosignature battery."""
    cfg = cfg or load_config()
    out_dir = cfg.root / "results" / "jwst_bio"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[jwst_bio] === resolve LHS 1140 b parameters ===")
    params = _fetch_planet_params()

    print("[jwst_bio] === query MAST for JWST spectra ===")
    prod_df, obs_df = _query_mast_products(LHS1140["ra"], LHS1140["dec"])
    n_jwst_spectra = int(len(_select_x1dints(prod_df))) if len(prod_df) else 0
    print(f"[jwst_bio] MAST: {len(obs_df)} obs, {n_jwst_spectra} x1dints products")

    print("[jwst_bio] === download + read x1dints ===")
    stacks = _download_and_read(prod_df, out_dir) if n_jwst_spectra else []

    print("[jwst_bio] === analyse ===")
    analyses = [_analyse_stack(s, params) for s in stacks]
    eclipse = _maybe_eclipse(obs_df, params)
    verdict = _combine_verdict(analyses, eclipse)

    coverage = {
        "mast_observations": int(len(obs_df)),
        "x1dints_products_available": n_jwst_spectra,
        "x1dints_stacks_analysed": len(analyses),
        "instruments_analysed": sorted({a["instrument"] for a in analyses}),
        "miri_present": bool(eclipse and eclipse.get("miri_present")),
        "data_reached": bool(analyses),
    }
    result = {
        "planet": "LHS 1140 b", "target": LHS1140, "params": params,
        "coverage": coverage, "analyses": analyses, "eclipse": eclipse,
        **verdict,
        "limitations": (
            "This is a DETECTION-LEVEL transmission-spectrum pipeline, not a "
            "publication-grade atmospheric retrieval. Depths come from a simple "
            "in-/out-of-transit split (no limb-darkening or systematics/detrending "
            "model), features are measured in fixed matched windows (no line-by-line "
            "fit or multi-species retrieval), and the MIRI eclipse depth is not "
            "extracted here (the discriminant is wired but reports 'not_measured' "
            "without a supplied depth). A positive disequilibrium result is a "
            "trigger for a full retrieval, never a stand-alone claim; O2/O3 results "
            "are gated by the M-dwarf abiotic false-positive test. When MAST is "
            "unreachable on the runner the channel records coverage and returns "
            "'no_data' -- it never fabricates a spectrum."),
    }
    (out_dir / "lhs1140b.json").write_text(json.dumps(result, indent=2,
                                                      default=str))

    summary = {
        "target": LHS1140["name"], "planet": "LHS 1140 b",
        "verdict": verdict["verdict"],
        "robust_biosignature": verdict["robust_biosignature"],
        "any_disequilibrium_pair": verdict["any_disequilibrium_pair"],
        "any_laser_line": verdict["any_laser_line"],
        "eclipse_classification": verdict["eclipse_classification"],
        "coverage": coverage,
        "best_pairs": [a["disequilibrium"]["best_pair"] for a in analyses
                       if a["disequilibrium"]["best_pair"]],
        "limitations": result["limitations"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str))
    print("[jwst_bio]", json.dumps(summary, default=str))
    return summary


__all__ = ["jwst_bio_run", "LHS1140"]
