"""CENOTAPH acquisition (runner only — the sandbox has no archive egress).

Three pulls, each chunked and checkpointed:

**1. The parent sample.** Gaia DR3 ``gaia_source`` × ``astrophysical_parameters``
(GSP-Spec), pulled in parallax shells. GSP-Spec is the primary spectroscopic
source rather than a LAMOST/APOGEE/GALAH crossmatch because it supplies all
four twin axes — Teff, log g, [M/H] **and [α/Fe]** — for ≈5.6 × 10⁶ stars from
a *single* pipeline, with no crossmatch attrition. A single pipeline is not a
convenience here: the whole twin estimator works by cancelling pipeline
systematics differentially, and that cancellation is exact only within one
pipeline. External surveys are supported as secondary samples via
:func:`fetch_vizier_spectro`.

**2. Photometry.** 2MASS JHKs and AllWISE W1–W4 through the Gaia archive's own
crossmatch tables, with the column names *probed at runtime* rather than
assumed — the Gaia mirrors of these catalogues have renamed error columns
between releases, and a wrong column name costs a whole runner job.

**3. The far-IR.** The AKARI/FIS Bright Source Catalogue (~427 k rows) and the
IRAS PSC/FSC are small enough to download *in their entirety* and crossmatch
locally with a KD-tree on unit vectors. This is strictly better than 10⁶ cone
searches and follows the pattern the ESO work in this repo settled on.

Epoch propagation
-----------------
IRAS observed in 1983.5, AKARI in 2006.5, 2MASS ~1999.5, WISE ~2010.5; Gaia DR3
positions are at 2016.0. A 200 mas/yr star has moved 6.6″ since AKARI and 6.6″
before that for IRAS — comparable to or larger than the match radius. Positions
are therefore propagated to each survey's epoch before matching. Skipping this
silently loses exactly the nearby, high-proper-motion stars that this search
most wants, and it has already cost this repository one run.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"

# Mean observation epochs (Julian year) used for proper-motion propagation.
SURVEY_EPOCH = {
    "gaia": 2016.0,
    "twomass": 1999.5,
    "wise": 2010.5,
    "akari": 2006.5,
    "iras": 1983.5,
    "galex": 2005.0,
}

# VizieR catalogue identifiers. Kept here (not in code paths) so that a
# correction is a one-line change; every one is probed for its real column
# names at runtime before use.
# Column names below are LIVE-VERIFIED against VizieR TAP, not guessed. Three
# traps that each cost a job if assumed:
#
#  * AKARI FIS quality flags are ``q_S65``…, **not** ``FQUAL65`` — the latter is
#    the JAXA-native name and does not exist in VizieR. Bit flags are ``f_S*``.
#  * IRAS PSC uses ``Fnu_12`` (with underscore); IRAS FSC uses ``Fnu12``
#    (without). A shared alias silently fails on one of the two.
#  * **The IRAS catalogues carry no J2000 columns at all** — TAP exposes only
#    the raw ``RA1950``/``DE1950``. VizieR's ``_RAJ2000`` is a web-interface
#    convenience that TAP does not serve. The B1950 positions must be precessed
#    (see ``_maybe_precess``), and at IRAS declinations the FK4→ICRS shift is
#    ~0.3-0.6 deg — a hundred times the match radius, so getting this wrong
#    does not degrade the match, it destroys it.
FAR_IR_CATALOGS = {
    "akari_fis": {
        "table": '"II/298/fis"',
        "epoch": "akari",
        "ra": ("raj2000", "radeg", "_raj2000", "ra"),
        "dec": ("dej2000", "dedeg", "_dej2000", "de", "dec"),
        "frame": "icrs",
        "fluxes": {"akari65": ("s65",), "akari90": ("s90",),
                   "akari140": ("s140",), "akari160": ("s160",)},
        "errors": {"akari65": ("e_s65",), "akari90": ("e_s90",),
                   "akari140": ("e_s140",), "akari160": ("e_s160",)},
        # q_S*: 3 = confirmed, flux reliable; 2 = confirmed, flux UNreliable;
        # 1 = not confirmed; 0 = not observed. Note this is NOT the IRAS
        # convention, where 1 means "upper limit".
        "quality": {"akari65": ("q_s65",), "akari90": ("q_s90",),
                    "akari140": ("q_s140",), "akari160": ("q_s160",)},
        # f_S* hex bits: 1 = CDS mode, 2 = flux too low, 8 = possible side-lobe.
        "bits": {"akari65": ("f_s65",), "akari90": ("f_s90",),
                 "akari140": ("f_s140",), "akari160": ("f_s160",)},
        "context": {"ndens": ("ndens",)},   # sources within 5 arcmin: crowding
        # The BSC is built from the All-Sky Survey, whose PSF is ~60" at
        # 65/90 um and ~90" at 140 um -- far broader than the slow-scan pointed
        # PSF (30-41") that the instrument papers quote. Matching at the pointed
        # PSF would lose real associations; the price is a larger chance-match
        # rate, which is why the closure ratio, not the position, is the
        # evidence.
        "match_radius_arcsec": 40.0,
        "beam_fwhm_arcsec": 60.0,
    },
    "iras_psc": {
        "table": '"II/125/main"',
        "epoch": "iras",
        "ra": ("_raj2000", "raj2000", "ra1950", "raj1950", "ra"),
        "dec": ("_dej2000", "dej2000", "de1950", "dej1950", "de", "dec"),
        "frame": "fk4_1950",
        "fluxes": {"iras12": ("fnu_12",), "iras25": ("fnu_25",),
                   "iras60": ("fnu_60",), "iras100": ("fnu_100",)},
        # e_Fnu_* are RELATIVE uncertainties in PERCENT, not Jy.
        "errors": {"iras12": ("e_fnu_12",), "iras25": ("e_fnu_25",),
                   "iras60": ("e_fnu_60",), "iras100": ("e_fnu_100",)},
        "errors_are_percent": True,
        "quality": {"iras12": ("q_fnu_12",), "iras25": ("q_fnu_25",),
                    "iras60": ("q_fnu_60",), "iras100": ("q_fnu_100",)},
        # Cirr3 is the total 100-um sky surface brightness in MJy/sr at the
        # source: the direct cirrus discriminant, and the single most useful
        # context column in the whole far-IR leg.
        "context": {"cirr1": ("cirr1",), "cirr2": ("cirr2",), "cirr3": ("cirr3",),
                    "confuse": ("confuse",), "var": ("var",)},
        "match_radius_arcsec": 60.0,
        "beam_fwhm_arcsec": 120.0,
    },
    "iras_fsc": {
        "table": '"II/156A/main"',
        "epoch": "iras",
        "ra": ("_raj2000", "raj2000", "ra1950", "raj1950", "ra"),
        "dec": ("_dej2000", "dej2000", "de1950", "dej1950", "de", "dec"),
        "frame": "fk4_1950",
        # FSC drops the underscore. |b| > 10 deg only, which suits this channel.
        "fluxes": {"iras12": ("fnu12",), "iras25": ("fnu25",),
                   "iras60": ("fnu60",), "iras100": ("fnu100",)},
        "errors": {},
        "quality": {"iras12": ("q_fnu12",), "iras25": ("q_fnu25",),
                    "iras60": ("q_fnu60",), "iras100": ("q_fnu100",)},
        "context": {"cir1": ("cir1",), "conf": ("conf",), "rel": ("rel",)},
        "match_radius_arcsec": 45.0,
        "beam_fwhm_arcsec": 90.0,
    },
}

# Quality codes that mean "this flux is a reliable detection".
AKARI_GOOD_QUALITY = (3,)
IRAS_GOOD_QUALITY = (3,)   # IRAS: 3 = high, 2 = moderate, 1 = upper limit

_GSPSPEC_QUERY = """
SELECT TOP {top}
       g.source_id, g.ra, g.dec, g.l, g.b, g.parallax, g.parallax_error,
       g.parallax_over_error, g.pmra, g.pmdec, g.ruwe,
       g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.bp_rp,
       g.phot_g_mean_flux_over_error, g.phot_bp_mean_flux_over_error,
       g.phot_rp_mean_flux_over_error, g.phot_g_n_obs,
       g.phot_bp_rp_excess_factor, g.phot_variable_flag,
       g.ipd_frac_multi_peak, g.astrometric_excess_noise_sig,
       g.non_single_star, g.nu_eff_used_in_astrometry, g.pseudocolour,
       g.ecl_lat, g.astrometric_params_solved,
       g.ag_gspphot, g.azero_gspphot, g.teff_gspphot, g.logg_gspphot,
       ap.teff_gspspec, ap.logg_gspspec, ap.mh_gspspec, ap.alphafe_gspspec,
       ap.teff_gspspec_upper, ap.teff_gspspec_lower,
       ap.logg_gspspec_upper, ap.logg_gspspec_lower,
       ap.mh_gspspec_upper, ap.mh_gspspec_lower,
       ap.flags_gspspec
FROM gaiadr3.gaia_source AS g
JOIN gaiadr3.astrophysical_parameters AS ap ON g.source_id = ap.source_id
WHERE g.parallax >= {plx_min} AND g.parallax < {plx_max}
  AND g.parallax_over_error > {poe_min}
  AND g.ruwe < {ruwe_max}
  AND ap.logg_gspspec > {logg_min}
  AND ap.teff_gspspec BETWEEN {teff_lo} AND {teff_hi}
  AND ap.mh_gspspec IS NOT NULL
  AND ap.alphafe_gspspec IS NOT NULL
"""

_SHELL_EDGES_MAS = [20.0, 12.0, 8.0, 6.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.6,
                    1.3, 1.1, 0.9, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0]


def _retry(fn, retries: int = 4, label: str = "query", base_sleep: float = 4.0):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[cenotaph] {label} attempt {attempt + 1}/{retries} failed: {exc!r}",
                  flush=True)
            time.sleep(base_sleep * (2**attempt))
    raise RuntimeError(f"{label} failed after {retries} attempts: {last!r}")


def _run_gaia(query: str, retries: int = 4) -> pd.DataFrame:
    """Async ADQL with backoff; the last attempt falls back to the sync endpoint."""
    from astroquery.gaia import Gaia

    state = {"n": 0}

    def _go():
        state["n"] += 1
        job = (Gaia.launch_job(query) if state["n"] >= retries
               else Gaia.launch_job_async(query))
        df = job.get_results().to_pandas()
        return df.rename(columns={c: c.lower() for c in df.columns})

    return _retry(_go, retries=retries, label="gaia")


def _run_vizier(query: str, retries: int = 4) -> pd.DataFrame:
    import pyvo

    def _go():
        svc = pyvo.dal.TAPService(VIZIER_TAP)
        df = svc.search(query).to_table().to_pandas()
        return df.rename(columns={c: c.lower() for c in df.columns})

    return _retry(_go, retries=retries, label="vizier")


def _probe_columns(table: str, service: str = "gaia") -> set[str]:
    """Return the real (lowercased) column names of ``table``."""
    q = f"SELECT TOP 1 * FROM {table}"
    df = _run_gaia(q) if service == "gaia" else _run_vizier(q)
    return {c.lower() for c in df.columns}


def _pick(candidates, available: set[str]) -> str | None:
    for c in candidates:
        if c in available:
            return c
    return None


# --------------------------------------------------------------------------
# 1. Parent sample
# --------------------------------------------------------------------------
def fetch_gspspec_sample(poe_min: float = 20.0, ruwe_max: float = 1.4,
                         logg_min: float = 3.8, teff_lo: float = 4000.0,
                         teff_hi: float = 7000.0, plx_min_mas: float = 1.0,
                         top_per_shell: int = 2_000_000,
                         checkpoint_dir: Path | None = None) -> pd.DataFrame:
    """Gaia DR3 GSP-Spec dwarfs in parallax shells, checkpointed per shell."""
    edges = [e for e in _SHELL_EDGES_MAS if e > plx_min_mas] + [plx_min_mas]
    edges = sorted(set(edges), reverse=True)
    frames = []
    for hi, lo in zip(edges[:-1], edges[1:], strict=False):
        ck = None
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ck = checkpoint_dir / f"shell_{lo:g}_{hi:g}.parquet"
            if ck.exists():
                df = pd.read_parquet(ck)
                print(f"[cenotaph] shell [{lo:g},{hi:g}) mas: {len(df)} (cached)")
                frames.append(df)
                continue
        q = _GSPSPEC_QUERY.format(top=int(top_per_shell), plx_min=lo, plx_max=hi,
                                  poe_min=poe_min, ruwe_max=ruwe_max,
                                  logg_min=logg_min, teff_lo=teff_lo, teff_hi=teff_hi)
        df = _run_gaia(q)
        if len(df) >= top_per_shell:
            print(f"[cenotaph] WARNING shell [{lo:g},{hi:g}) hit the row cap")
        print(f"[cenotaph] shell [{lo:g},{hi:g}) mas: {len(df)} stars", flush=True)
        if ck is not None:
            df.to_parquet(ck, index=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates("source_id")
    print(f"[cenotaph] parent sample: {len(out)} stars")
    return out.reset_index(drop=True)


def filter_gspspec_flags(df: pd.DataFrame, max_flag: int = 1) -> pd.DataFrame:
    """Apply the recommended GSP-Spec quality cut on ``flags_gspspec``.

    ``flags_gspspec`` is a 41-character string of per-diagnostic quality digits
    (Recio-Blanco et al. 2023). The first 13 characters cover the parameter
    determination itself; the recommended "best" sample keeps stars whose first
    13 flags are all ≤ 1. Parsed in Python rather than ADQL because string
    slicing across TAP implementations is not portable.
    """
    if "flags_gspspec" not in df.columns:
        print("[cenotaph] flags_gspspec absent; GSP-Spec quality cut NOT applied")
        out = df.copy()
        out["gspspec_flag_ok"] = pd.NA
        return out
    s = df["flags_gspspec"].astype("string").fillna("")
    ok = s.map(lambda v: bool(v) and len(v) >= 13
               and all(ch.isdigit() and int(ch) <= max_flag for ch in v[:13]))
    out = df.copy()
    out["gspspec_flag_ok"] = ok
    print(f"[cenotaph] GSP-Spec flags<= {max_flag}: {int(ok.sum())}/{len(out)} pass")
    return out


def apply_parallax_zero_point(df: pd.DataFrame) -> pd.DataFrame:
    """Lindegren et al. (2021) parallax zero-point, with an honest fallback.

    Uses the official ``gaiadr3-zeropoint`` package when it is installed (it
    ships the Z5/Z6 interpolation tables). If it is not, the global −17 µas
    offset is applied instead and the method is recorded in
    ``parallax_zp_method`` so the run never silently claims a correction it did
    not make. The distinction matters: the zero-point varies by ±30 µas with
    magnitude and colour, and at ϖ = 1 mas that is 0.065 mag of distance
    modulus — which is *exactly* a grey offset.
    """
    out = df.copy()
    method = "global_-17uas_fallback"
    zp = np.full(len(out), -17e-3)  # mas
    try:
        from zero_point import zpt  # type: ignore

        zpt.load_tables()
        need = ["phot_g_mean_mag", "nu_eff_used_in_astrometry", "pseudocolour",
                "ecl_lat", "astrometric_params_solved"]
        if all(c in out.columns for c in need):
            vals = zpt.get_zpt(
                out["phot_g_mean_mag"].to_numpy(float),
                out["nu_eff_used_in_astrometry"].to_numpy(float),
                out["pseudocolour"].to_numpy(float),
                out["ecl_lat"].to_numpy(float),
                out["astrometric_params_solved"].to_numpy(),
                _warnings=False,
            )
            zp = np.where(np.isfinite(vals), vals, -17e-3)
            method = "lindegren2021_Z5Z6"
    except Exception as exc:  # noqa: BLE001
        print(f"[cenotaph] gaiadr3-zeropoint unavailable ({exc!r}); "
              "falling back to the global -17 uas offset")
    out["parallax_raw"] = out["parallax"]
    out["parallax_zp_mas"] = zp
    out["parallax"] = out["parallax"] - zp
    out["parallax_zp_method"] = method
    print(f"[cenotaph] parallax zero-point: {method}, "
          f"median {np.nanmedian(zp) * 1e3:+.1f} uas")
    return out


# --------------------------------------------------------------------------
# 2. Photometry
# --------------------------------------------------------------------------
def _external_photometry_query(kind: str, plx_lo: float, plx_hi: float,
                               top: int, sel: dict, where: dict) -> str:
    cols = ", ".join(f"x.{v} AS {k}" for k, v in sel.items())
    if kind == "twomass":
        joins = (
            "JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS bn ON g.source_id = bn.source_id "
            "JOIN gaiadr3.tmass_psc_xsc_join AS tj "
            "  ON bn.clean_tmass_psc_xsc_oid = tj.clean_tmass_psc_xsc_oid "
            "JOIN gaiadr1.tmass_original_valid AS x "
            "  ON tj.original_psc_source_id = x.designation"
        )
    else:
        joins = (
            "JOIN gaiadr3.allwise_best_neighbour AS bn ON g.source_id = bn.source_id "
            "JOIN gaiadr1.allwise_original_valid AS x "
            "  ON bn.original_ext_source_id = x.designation"
        )
    return f"""
SELECT TOP {int(top)} g.source_id, bn.angular_distance AS xm_arcsec, {cols}
FROM gaiadr3.gaia_source AS g
JOIN gaiadr3.astrophysical_parameters AS ap ON g.source_id = ap.source_id
{joins}
WHERE g.parallax >= {plx_lo} AND g.parallax < {plx_hi}
  AND g.parallax_over_error > {where['poe_min']}
  AND g.ruwe < {where['ruwe_max']}
  AND ap.logg_gspspec > {where['logg_min']}
  AND ap.teff_gspspec BETWEEN {where['teff_lo']} AND {where['teff_hi']}
  AND ap.mh_gspspec IS NOT NULL
  AND ap.alphafe_gspspec IS NOT NULL
"""


_TWOMASS_WANT = {
    "j_mag": ("j_m", "jmag"), "j_mag_error": ("j_msigcom", "e_jmag", "j_cmsig"),
    "h_mag": ("h_m", "hmag"), "h_mag_error": ("h_msigcom", "e_hmag", "h_cmsig"),
    "ks_mag": ("ks_m", "kmag", "ksmag"),
    "ks_mag_error": ("ks_msigcom", "e_kmag", "ks_cmsig"),
    "tmass_ph_qual": ("ph_qual",),
}
_ALLWISE_WANT = {
    "w1_mag": ("w1mpro",), "w1_mag_error": ("w1mpro_error", "w1sigmpro"),
    "w2_mag": ("w2mpro",), "w2_mag_error": ("w2mpro_error", "w2sigmpro"),
    "w3_mag": ("w3mpro",), "w3_mag_error": ("w3mpro_error", "w3sigmpro"),
    "w4_mag": ("w4mpro",), "w4_mag_error": ("w4mpro_error", "w4sigmpro"),
    "wise_ph_qual": ("ph_qual",), "wise_cc_flags": ("cc_flags",),
    "wise_ext_flag": ("ext_flag", "ext_flg"),
    "wise_var_flag": ("var_flag", "var_flg"),
}


def fetch_external_photometry(kind: str, where: dict,
                              plx_min_mas: float = 1.0,
                              top_per_shell: int = 2_000_000,
                              checkpoint_dir: Path | None = None) -> pd.DataFrame:
    """2MASS (``kind='twomass'``) or AllWISE (``kind='allwise'``) via Gaia's crossmatch.

    Column names are *probed* first. The Gaia mirrors of these catalogues have
    used both ``w1mpro_error`` and ``w1sigmpro`` across releases; guessing costs
    a job, probing costs one row.
    """
    table = ("gaiadr1.tmass_original_valid" if kind == "twomass"
             else "gaiadr1.allwise_original_valid")
    want = _TWOMASS_WANT if kind == "twomass" else _ALLWISE_WANT
    available = _probe_columns(table)
    sel = {}
    for out_name, cands in want.items():
        got = _pick(cands, available)
        if got:
            sel[out_name] = got
        else:
            print(f"[cenotaph] {kind}: no column for {out_name} "
                  f"(tried {cands}); it will be missing")
    if not sel:
        raise RuntimeError(f"{kind}: probe found none of the wanted columns")

    edges = [e for e in _SHELL_EDGES_MAS if e > plx_min_mas] + [plx_min_mas]
    edges = sorted(set(edges), reverse=True)
    frames = []
    for hi, lo in zip(edges[:-1], edges[1:], strict=False):
        ck = None
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ck = checkpoint_dir / f"{kind}_{lo:g}_{hi:g}.parquet"
            if ck.exists():
                frames.append(pd.read_parquet(ck))
                continue
        q = _external_photometry_query(kind, lo, hi, top_per_shell, sel, where)
        df = _run_gaia(q)
        print(f"[cenotaph] {kind} shell [{lo:g},{hi:g}): {len(df)} rows", flush=True)
        if ck is not None:
            df.to_parquet(ck, index=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["source_id"])
    return (pd.concat(frames, ignore_index=True)
            .drop_duplicates("source_id").reset_index(drop=True))


# --------------------------------------------------------------------------
# 3. Far-infrared
# --------------------------------------------------------------------------

def _maybe_precess(df: pd.DataFrame, frame: str, name: str,
                   ra_col_used: str) -> pd.DataFrame:
    """Convert B1950/FK4 positions to ICRS when that is what the table serves.

    The IRAS catalogues expose only ``RA1950``/``DE1950`` over TAP. The FK4
    (B1950, epoch 1950) to ICRS shift is ~0.3-0.6 deg at IRAS declinations --
    roughly a hundred times the match radius -- so a missed precession does not
    merely degrade the crossmatch, it produces zero real matches and a pure
    chance-coincidence sample. If the resolved column was already a J2000 one,
    nothing is done.
    """
    if frame != "fk4_1950" or "1950" not in ra_col_used:
        return df
    try:
        import astropy.units as u
        from astropy.coordinates import FK4, SkyCoord

        c = SkyCoord(ra=df["ra"].to_numpy(float) * u.deg,
                     dec=df["dec"].to_numpy(float) * u.deg,
                     frame=FK4(equinox="B1950", obstime="B1950"))
        icrs = c.icrs
        out = df.copy()
        out["ra"] = icrs.ra.deg
        out["dec"] = icrs.dec.deg
        print(f"[cenotaph] {name}: precessed {len(out)} FK4/B1950 positions "
              f"to ICRS (median shift "
              f"{float(np.nanmedian(np.abs(out['dec'] - df['dec']))) * 60:.1f} arcmin)")
        return out
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"{name}: table serves B1950 positions ({ra_col_used}) but the "
            f"FK4->ICRS conversion failed ({exc!r}); refusing to crossmatch, "
            "because an unprecessed match is pure chance coincidence") from exc


def fetch_far_ir_catalog(name: str, checkpoint: Path | None = None) -> pd.DataFrame:
    """Download an entire far-IR catalogue (≤ ~0.5 M rows) for local matching.

    Cone-searching 10⁶ targets against VizieR is not a thing that finishes;
    downloading the 427 k-row AKARI/FIS BSC once and matching with a KD-tree is
    seconds of work.
    """
    if checkpoint is not None and checkpoint.exists():
        return pd.read_parquet(checkpoint)
    spec = FAR_IR_CATALOGS[name]
    table = spec["table"]
    available = _probe_columns(table, service="vizier")
    ra_col = _pick(spec["ra"], available)
    dec_col = _pick(spec["dec"], available)
    if ra_col is None or dec_col is None:
        raise RuntimeError(f"{name}: no RA/Dec column found in {table} "
                           f"(available: {sorted(available)[:40]})")
    sel = {"ra": ra_col, "dec": dec_col}
    for group in ("fluxes", "errors", "quality", "bits", "context"):
        for out_name, cands in spec.get(group, {}).items():
            got = _pick(cands, available)
            if got:
                key = out_name if group in ("fluxes", "context") \
                    else f"{out_name}_{group[:3]}"
                sel[key] = got
    cols = ", ".join(f"{v} AS {k}" for k, v in sel.items())
    df = _run_vizier(f"SELECT {cols} FROM {table}")
    df = _maybe_precess(df, spec.get("frame", "icrs"), name, ra_col)
    print(f"[cenotaph] {name}: {len(df)} rows, columns {sorted(sel)}")
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(checkpoint, index=False)
    return df


def propagate_position(ra_deg, dec_deg, pmra_masyr, pmdec_masyr,
                       from_epoch: float, to_epoch: float):
    """Linear proper-motion propagation. ``pmra`` is μ_α* (already × cos δ)."""
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    pmra = np.nan_to_num(np.asarray(pmra_masyr, dtype=float))
    pmdec = np.nan_to_num(np.asarray(pmdec_masyr, dtype=float))
    dt = to_epoch - from_epoch
    dec2 = dec + pmdec * dt / 3.6e6
    cosd = np.cos(np.radians(np.clip(dec, -89.999, 89.999)))
    ra2 = ra + (pmra * dt / 3.6e6) / np.maximum(cosd, 1e-6)
    return ra2, dec2


def _unit_vectors(ra_deg, dec_deg) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    return np.column_stack([np.cos(dec) * np.cos(ra),
                            np.cos(dec) * np.sin(ra),
                            np.sin(dec)])


def crossmatch_far_ir(targets: pd.DataFrame, cat: pd.DataFrame, name: str,
                      radius_arcsec: float | None = None) -> pd.DataFrame:
    """Nearest-neighbour match of ``targets`` to a far-IR catalogue.

    Targets are propagated from Gaia's 2016.0 to the survey epoch first.
    Returns one row per matched target with the catalogue fluxes, the
    separation, and — crucially — ``<name>_n_within``, the number of catalogue
    sources inside the radius. More than one means the association is
    ambiguous, which for a 40″ beam is common and is a rejection, not a detail.
    """
    from scipy.spatial import cKDTree

    spec = FAR_IR_CATALOGS[name]
    radius = radius_arcsec or spec["match_radius_arcsec"]
    epoch = SURVEY_EPOCH[spec["epoch"]]
    if len(cat) == 0 or len(targets) == 0:
        return pd.DataFrame(columns=["source_id"])

    ra_t, dec_t = propagate_position(
        targets["ra"], targets["dec"], targets.get("pmra"), targets.get("pmdec"),
        SURVEY_EPOCH["gaia"], epoch)
    vt = _unit_vectors(ra_t, dec_t)
    vc = _unit_vectors(cat["ra"], cat["dec"])
    chord = 2.0 * math.sin(math.radians(radius / 3600.0) / 2.0)

    tree = cKDTree(vc)
    neighbours = tree.query_ball_point(vt, r=chord, workers=-1)
    rows = []
    flux_cols = [c for c in cat.columns if c not in ("ra", "dec")]
    for i, nb in enumerate(neighbours):
        if not nb:
            continue
        d = np.linalg.norm(vc[nb] - vt[i], axis=1)
        j = nb[int(np.argmin(d))]
        sep = 2.0 * math.degrees(math.asin(min(float(d.min()) / 2.0, 1.0))) * 3600.0
        rec = {"source_id": targets["source_id"].iloc[i],
               f"{name}_sep_arcsec": sep,
               f"{name}_n_within": len(nb)}
        for c in flux_cols:
            rec[c] = cat[c].iloc[j]
        rows.append(rec)
    out = pd.DataFrame(rows)
    print(f"[cenotaph] {name}: {len(out)} of {len(targets)} targets matched "
          f"within {radius:.0f}\" (epoch {epoch})")
    return out


def fetch_vizier_spectro(table: str, colmap: dict[str, tuple[str, ...]],
                         top: int = 5_000_000) -> pd.DataFrame:
    """Secondary spectroscopic sample (LAMOST / APOGEE / GALAH) from VizieR.

    Kept generic and column-probed: these catalogues move between VizieR
    releases and the point of a secondary sample is to check that the result
    does not depend on GSP-Spec, so it must fail loudly rather than silently
    return the wrong column.
    """
    available = _probe_columns(table, service="vizier")
    sel = {}
    for out_name, cands in colmap.items():
        got = _pick(cands, available)
        if got:
            sel[out_name] = got
        else:
            print(f"[cenotaph] {table}: no column for {out_name} (tried {cands})")
    if "ra" not in sel or "dec" not in sel:
        raise RuntimeError(f"{table}: missing RA/Dec; available {sorted(available)[:40]}")
    cols = ", ".join(f"{v} AS {k}" for k, v in sel.items())
    return _run_vizier(f"SELECT TOP {int(top)} {cols} FROM {table}")


def count_beam_neighbours(ra_deg: float, dec_deg: float, radius_arcsec: float,
                          g_max: float = 18.0, exclude_source_id: int | None = None
                          ) -> int:
    """Gaia sources inside the far-IR beam other than the target itself.

    A far-IR flux attributed to a star that shares its beam with another
    catalogued source is not attributable to that star. Run only on the handful
    of stars that survive to the far-IR stage.
    """
    q = f"""
SELECT COUNT(*) AS n FROM gaiadr3.gaia_source
WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_arcsec / 3600.0}))
  AND phot_g_mean_mag < {g_max}
"""
    if exclude_source_id is not None:
        q += f"  AND source_id != {int(exclude_source_id)}\n"
    df = _run_gaia(q)
    return int(df.iloc[0, 0]) if len(df) else 0


__all__ = [
    "FAR_IR_CATALOGS",
    "SURVEY_EPOCH",
    "apply_parallax_zero_point",
    "count_beam_neighbours",
    "crossmatch_far_ir",
    "fetch_external_photometry",
    "fetch_far_ir_catalog",
    "fetch_gspspec_sample",
    "fetch_vizier_spectro",
    "filter_gspspec_flags",
    "propagate_position",
]
