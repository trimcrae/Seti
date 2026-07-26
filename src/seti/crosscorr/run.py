"""Runner-side orchestration for the LHS 1140 b cross-correlation search.

Pipeline (writes under ``results/crosscorr/``):

1. **Ephemeris + Kp** -- resolve LHS 1140 b's orbital period, mid-transit epoch,
   transit duration, stellar mass, semi-major axis and inclination from the NASA
   Exoplanet Archive ``pscomppars`` TAP (same pattern as ``lhs1140/run.py``), and
   compute the planet radial-velocity semi-amplitude
   ``Kp = 2*pi*a*sin(i)/P`` -- the key quantity that sets the Doppler track the CCFs
   are stacked along.
2. **Archive lookup** -- attempt to locate archival high-resolution transit spectra
   of LHS 1140 (ESO Science Archive for ESPRESSO/HARPS/NIRPS; DACE).  Programmatic
   ESO/DACE access is frequently unauthenticated or unreachable on a runner; when
   it is, we **degrade honestly** -- we record a hard-coded, cited inventory of the
   published high-resolution datasets that exist, and never fabricate spectra.
3. **CCF pipeline** -- *only if spectra are actually retrieved*: build residuals
   (remove the stationary stellar + telluric component per wavelength, a SysRem-lite
   detrend), cross-correlate each in-transit residual against the O2 A-band and H2O
   templates, build the Kp-Vsys map and report the detection significance.

The pure engine is in :mod:`seti.crosscorr.xcorr` and is unit-tested offline; this
module supplies live parameters and does the acquisition, which needs archive
egress and runs on the GitHub runner.
"""

from __future__ import annotations

import json

import numpy as np

from ..config import Config, load_config
from .xcorr import (
    cross_correlation,
    h2o_template,
    kp_vsys_map,
    o2_a_band_template,
    planet_rv_track,
)

# --- constants -------------------------------------------------------------
_G = 6.67430e-11            # gravitational constant, SI
_MSUN = 1.98892e30         # kg
_AU = 1.495978707e11       # m
_DAY = 86400.0             # s

# LHS 1140 b literature ephemeris (Cadieux et al. 2024, ApJL 970 L2; stellar mass
# and systemic velocity from the same and Lillo-Box et al. 2020), a fallback if the
# live NASA Exoplanet Archive fetch fails.  ``vsys_kms`` is the barycentric
# systemic radial velocity of LHS 1140.
_LHS1140B_FALLBACK = {
    "pl_orbper_d": 24.73723, "pl_tranmid_bjd": 2458226.843, "pl_trandur_h": 2.055,
    "st_mass_msun": 0.184, "pl_orbsmax_au": 0.0946, "pl_orbincl_deg": 89.86,
    "vsys_kms": -13.23,
}

# Published high-resolution spectroscopic datasets of LHS 1140 (honest, cited
# inventory used when programmatic archive access is unavailable on the runner).
# The distinction that matters for this channel: these are predominantly
# radial-velocity *monitoring* sequences (out-of-transit epochs for the mass
# measurement), not dedicated *in-transit* high-resolution transmission sequences
# -- of which none are, to our knowledge, published for LHS 1140 b.  That is the
# core data-availability limitation this channel documents.
_PUBLISHED_HIRES_DATASETS = [
    {"instrument": "HARPS", "facility": "ESO 3.6 m (La Silla)",
     "resolution": 115000, "band": "optical 0.38-0.69 um",
     "reference": "Dittmann et al. 2017, Nature 544, 333 (discovery RVs)",
     "cadence": "RV monitoring", "in_transit_sequence": False},
    {"instrument": "HARPS", "facility": "ESO 3.6 m (La Silla)",
     "resolution": 115000, "band": "optical 0.38-0.69 um",
     "reference": "Ment et al. 2019, AJ 157, 32 (planet c; extended RVs)",
     "cadence": "RV monitoring", "in_transit_sequence": False},
    {"instrument": "ESPRESSO", "facility": "ESO VLT (Paranal)",
     "resolution": 140000, "band": "optical 0.38-0.79 um (covers O2 gamma/B; A-band at edge)",
     "reference": "Lillo-Box et al. 2020, A&A 642, A121 (mass refinement)",
     "cadence": "RV monitoring", "in_transit_sequence": False},
    {"instrument": "ESPRESSO", "facility": "ESO VLT (Paranal)",
     "resolution": 140000, "band": "optical 0.38-0.79 um",
     "reference": "Cadieux et al. 2024, ApJL 970, L2 (updated system params)",
     "cadence": "RV monitoring", "in_transit_sequence": False},
    {"instrument": "NIRPS", "facility": "ESO 3.6 m (La Silla)",
     "resolution": 90000, "band": "near-IR 0.98-1.8 um (H2O bands)",
     "reference": "NIRPS GTO temperate-M-dwarf program (commissioned 2023)",
     "cadence": "RV monitoring / potential transit sequences", "in_transit_sequence": False},
]


def _fetch_ephemeris(name: str = "LHS 1140 b") -> dict:
    """Live LHS 1140 b ephemeris + Kp from the NASA Exoplanet Archive (pscomppars).

    Fetches ``pl_orbper, pl_tranmid, pl_trandur, st_mass, pl_orbsmax, pl_orbincl``
    and the systemic RV ``st_radv``; falls back to the cited literature values on
    any failure.  Computes the planet radial-velocity semi-amplitude
    ``Kp = 2*pi*a*sin(i)/P`` (using the archived ``a`` when present, else Kepler's
    third law from the stellar mass).
    """
    p = dict(_LHS1140B_FALLBACK)
    try:
        import io

        import pandas as pd
        import requests
        q = ("select pl_orbper,pl_tranmid,pl_trandur,st_mass,pl_orbsmax,"
             f"pl_orbincl,st_radv from pscomppars where pl_name='{name}'")
        r = requests.get("https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
                         params={"query": q, "format": "csv"}, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if len(df):
            row = df.iloc[0]

            def val(col, default):
                v = row.get(col)
                num = pd.to_numeric(v, errors="coerce")
                return float(num) if num is not None and np.isfinite(num) else default

            p.update({
                "pl_orbper_d": val("pl_orbper", p["pl_orbper_d"]),
                "pl_tranmid_bjd": val("pl_tranmid", p["pl_tranmid_bjd"]),
                "pl_trandur_h": val("pl_trandur", p["pl_trandur_h"]),
                "st_mass_msun": val("st_mass", p["st_mass_msun"]),
                "pl_orbsmax_au": val("pl_orbsmax", p["pl_orbsmax_au"]),
                "pl_orbincl_deg": val("pl_orbincl", p["pl_orbincl_deg"]),
                "vsys_kms": val("st_radv", p["vsys_kms"]),
            })
            print(f"[crosscorr] resolved LHS 1140 b ephemeris from NASA archive: "
                  f"P={p['pl_orbper_d']:.5f} d T14={p['pl_trandur_h']:.2f} h "
                  f"a={p['pl_orbsmax_au']:.4f} AU i={p['pl_orbincl_deg']:.2f} deg")
    except Exception as exc:  # noqa: BLE001
        print(f"[crosscorr] ephemeris fetch failed ({exc!r}); using literature fallback")

    kp = _compute_kp(p)
    p["kp_kms"] = kp
    print(f"[crosscorr] planet RV semi-amplitude Kp = {kp:.2f} km/s "
          f"(Vsys = {p['vsys_kms']:.2f} km/s)")
    return p


def _compute_kp(p: dict) -> float:
    """Planet radial-velocity semi-amplitude ``Kp = 2*pi*a*sin(i)/P`` (km/s)."""
    period_s = p["pl_orbper_d"] * _DAY
    a_au = p.get("pl_orbsmax_au")
    if a_au and np.isfinite(a_au) and a_au > 0:
        a_m = a_au * _AU
    else:
        # Kepler III from the stellar mass (planet mass negligible for the semi).
        a_m = (_G * p["st_mass_msun"] * _MSUN * period_s ** 2
               / (4.0 * np.pi ** 2)) ** (1.0 / 3.0)
    sini = np.sin(np.radians(p.get("pl_orbincl_deg", 90.0)))
    v_orb = 2.0 * np.pi * a_m / period_s          # m/s
    return float(v_orb * sini / 1000.0)


def _locate_archival_spectra(name: str = "LHS 1140") -> dict:
    """Attempt to locate archival high-resolution transit spectra of LHS 1140.

    Tries the ESO Science Archive (astroquery.eso) for ESPRESSO/HARPS/NIRPS
    metadata.  Data *download* needs ESO authentication and is not attempted here;
    a successful metadata query still tells us what exists.  On any
    failure/unavailability we degrade honestly to the cited published inventory and
    return no spectra -- we never fabricate.
    """
    result = {
        "retrieved_spectra": [],           # list of {wavelength, flux, phase, ...}
        "archive_reachable": False,
        "eso_records": [],
        "published_inventory": _PUBLISHED_HIRES_DATASETS,
    }
    try:
        from astroquery.eso import Eso
        eso = Eso()
        eso.ROW_LIMIT = 200
        for inst in ("ESPRESSO", "HARPS", "NIRPS"):
            try:
                tbl = eso.query_instrument(inst.lower(), target=name)
            except Exception as exc:  # noqa: BLE001
                print(f"[crosscorr] ESO {inst} query failed: {exc!r}")
                continue
            n = 0 if tbl is None else len(tbl)
            print(f"[crosscorr] ESO {inst}: {n} archive records for {name}")
            if n:
                result["archive_reachable"] = True
                result["eso_records"].append({"instrument": inst, "n_records": int(n)})
    except Exception as exc:  # noqa: BLE001
        print(f"[crosscorr] ESO archive unreachable ({exc!r}); "
              f"degrading to published inventory")
    if not result["retrieved_spectra"]:
        print("[crosscorr] no in-transit high-resolution spectra retrieved "
              "(auth/egress); reporting published-dataset inventory only")
    return result


def build_residuals(wavelength: np.ndarray, flux_stack: np.ndarray,
                    n_sysrem: int = 1) -> np.ndarray:
    """Remove the stationary stellar+telluric component -- a SysRem-lite detrend.

    ``flux_stack`` is ``(n_exposure, n_wavelength)`` continuum-normalised in-transit
    spectra sharing ``wavelength``.  The star and Earth's telluric lines are fixed
    in wavelength across the sequence while the planet's lines march by ``v_p``, so
    dividing each column by its time median removes the bulk of both; ``n_sysrem``
    optional passes then subtract the leading systematic (a rank-1 SVD component)
    to mop up residual airmass/continuum trends.  Returns residuals with absorption
    as **negative** dips (the convention :func:`cross_correlation` expects).
    """
    stack = np.asarray(flux_stack, dtype=float)
    if stack.ndim != 2:
        raise ValueError("flux_stack must be 2-D (n_exposure, n_wavelength)")
    med = np.median(stack, axis=0)
    med = np.where(med == 0, 1.0, med)
    resid = stack / med - 1.0                      # dimensionless, ~0 continuum
    for _ in range(max(0, int(n_sysrem))):
        # Rank-1 removal of the dominant common-mode systematic.
        resid = np.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0)
        u, s, vt = np.linalg.svd(resid, full_matrices=False)
        if s.size == 0:
            break
        resid = resid - s[0] * np.outer(u[:, 0], vt[0])
    return resid


def _phases_from_times(times_bjd: np.ndarray, t0_bjd: float, period_d: float):
    """Orbital phase (cycles from mid-transit, wrapped to [-0.5, 0.5))."""
    ph = ((np.asarray(times_bjd, dtype=float) - t0_bjd) / period_d + 0.5) % 1.0 - 0.5
    return ph


def _run_ccf_pipeline(spectra: list[dict], ephem: dict) -> dict:
    """Full CCF pipeline on retrieved in-transit spectra: residuals -> Kp-Vsys map.

    Each entry of ``spectra`` needs ``wavelength`` (vacuum Angstrom), ``flux``
    (continuum-normalised) and ``time_bjd``.  Returns per-species detection maps.
    Called only when spectra are actually present.
    """
    wl = np.asarray(spectra[0]["wavelength"], dtype=float)
    flux_stack = np.array([s["flux"] for s in spectra], dtype=float)
    times = np.array([s["time_bjd"] for s in spectra], dtype=float)
    phases = _phases_from_times(times, ephem["pl_tranmid_bjd"], ephem["pl_orbper_d"])
    resid = build_residuals(wl, flux_stack)

    rv_grid = np.arange(-120.0, 120.01, 1.0)
    kp_grid = np.arange(0.0, 120.01, 1.0)
    vsys_grid = np.arange(-60.0, 60.01, 1.0)

    out = {"n_exposures": len(spectra), "species": {}}
    for species, tmpl in (("O2", o2_a_band_template()), ("H2O", h2o_template())):
        twl, tdepth = tmpl
        in_band = (twl >= wl.min()) & (twl <= wl.max())
        if np.count_nonzero(in_band) < 3:
            out["species"][species] = {"status": "template_out_of_band",
                                       "n_lines_in_band": int(np.count_nonzero(in_band))}
            continue
        ccfs = np.array([cross_correlation(wl, resid[e], twl[in_band],
                                           tdepth[in_band], rv_grid)
                         for e in range(len(spectra))])
        m = kp_vsys_map(ccfs, phases, rv_grid, kp_grid, vsys_grid)
        out["species"][species] = {
            "status": "computed",
            "n_lines_in_band": int(np.count_nonzero(in_band)),
            "kp_peak_kms": m["kp_peak"], "vsys_peak_kms": m["vsys_peak"],
            "significance": round(m["significance"], 2),
            "kp_expected_kms": round(ephem["kp_kms"], 2),
            "vsys_expected_kms": round(ephem["vsys_kms"], 2),
        }
        print(f"[crosscorr] {species}: peak Kp={m['kp_peak']:.0f} "
              f"Vsys={m['vsys_peak']:.0f} significance={m['significance']:.1f} "
              f"(expected Kp~{ephem['kp_kms']:.0f})")
    return out


def crosscorr_run(cfg: Config | None = None) -> dict:
    """LHS 1140 b high-resolution cross-correlation biosignature search.

    Resolves the ephemeris + Kp, attempts to locate archival high-resolution
    transit spectra, runs the full CCF pipeline on any retrieved, and writes
    ``results/crosscorr/lhs1140b.json`` + ``summary.json`` with an explicit verdict,
    ``limitations`` and ``data_availability``.
    """
    cfg = cfg or load_config()
    out_dir = cfg.root / "results" / "crosscorr"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[crosscorr] === ephemeris + Kp ===")
    ephem = _fetch_ephemeris()

    # Expected planet velocity excursion during the actual transit -- the physical
    # constraint on how well Kp can be separated from Vsys for THIS planet.
    half_dur_phase = (ephem["pl_trandur_h"] / 24.0) / ephem["pl_orbper_d"] / 2.0
    v_edge = float(planet_rv_track(half_dur_phase, ephem["kp_kms"], 0.0))
    dv_across_transit = 2.0 * abs(v_edge)

    print("[crosscorr] === archive lookup ===")
    arch = _locate_archival_spectra()

    print("[crosscorr] === CCF pipeline ===")
    spectra = arch.get("retrieved_spectra") or []
    if spectra:
        pipeline = _run_ccf_pipeline(spectra, ephem)
        best = max((v.get("significance", 0.0)
                    for v in pipeline["species"].values()
                    if isinstance(v, dict)), default=0.0)
        verdict = ("CCF_DETECTION_REVIEW" if best >= 5.0
                   else "NO_SIGNIFICANT_CCF_SIGNAL")
    else:
        pipeline = {"n_exposures": 0, "species": {},
                    "status": "no_spectra_retrieved"}
        verdict = "NO_ARCHIVAL_IN_TRANSIT_HIRES_SPECTRA_AVAILABLE"

    limitations = [
        "The O2 A-band (0.76 um) is telluric-O2 dominated; the method relies on the "
        "planet's Doppler shift to separate planetary from telluric O2 -- feasible "
        "only because the two are in different rest frames, but demanding on S/N and "
        "telluric modelling.",
        f"Kp only sweeps ~{dv_across_transit:.1f} km/s across LHS 1140 b's "
        f"{ephem['pl_trandur_h']:.1f} h transit, so within a single transit the "
        "Kp axis is weakly constrained (Kp-Vsys degeneracy); breaking it needs many "
        "transits combined.",
        "ESPRESSO/HARPS optical coverage reaches the O2 A-band only at its red edge; "
        "the strongest O2 A-band lines and the H2O bands are better matched by "
        "NIRPS/IGRINS near-IR spectra.",
        "The H2O template here is a small illustrative comb; a production search "
        "must use the full HITRAN line list.",
    ]
    data_availability = {
        "archive_reachable": arch.get("archive_reachable", False),
        "eso_records": arch.get("eso_records", []),
        "in_transit_hires_sequences_retrieved": len(spectra),
        "published_datasets": arch.get("published_inventory", []),
        "note": (
            "Published high-resolution LHS 1140 spectra are predominantly "
            "radial-velocity monitoring sequences (out of transit), not dedicated "
            "in-transit transmission sequences; to our knowledge no in-transit "
            "high-resolution cross-correlation dataset for LHS 1140 b is public. "
            "Programmatic ESO/DACE download also requires authentication not "
            "available on the runner. No spectra are fabricated: when none are "
            "retrieved the CCF pipeline is not run and this is reported honestly."),
    }

    result = {
        "planet": "LHS 1140 b",
        "gaia_dr3": "2371032916186181760",
        "ephemeris": ephem,
        "kp_kms": round(ephem["kp_kms"], 3),
        "vsys_kms": round(ephem["vsys_kms"], 3),
        "planet_velocity_swing_across_transit_kms": round(dv_across_transit, 3),
        "pipeline": pipeline,
        "verdict": verdict,
        "limitations": limitations,
        "data_availability": data_availability,
    }
    (out_dir / "lhs1140b.json").write_text(json.dumps(result, indent=2, default=str))

    summary = {
        "target": "LHS 1140 b",
        "method": "high-resolution transmission cross-correlation (O2 A-band + H2O)",
        "kp_kms": round(ephem["kp_kms"], 2),
        "vsys_kms": round(ephem["vsys_kms"], 2),
        "planet_velocity_swing_across_transit_kms": round(dv_across_transit, 2),
        "archive_reachable": arch.get("archive_reachable", False),
        "in_transit_hires_sequences_retrieved": len(spectra),
        "n_published_datasets_inventoried": len(arch.get("published_inventory", [])),
        "verdict": verdict,
        "data_availability_note": data_availability["note"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("[crosscorr]", json.dumps(summary, default=str))
    return summary


__all__ = ["crosscorr_run", "build_residuals"]
