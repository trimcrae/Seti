"""Runner-only archive acquisition for OSSUARY.

Everything here needs outbound network and therefore runs on the GitHub Actions
runner, never in the sandbox.  Each fetch checkpoints to its own parquet so a
killed job loses minutes rather than hours.

Why the whole pull is one Gaia-archive join
-------------------------------------------
The Gaia archive mirrors AllWISE (``gaiadr1.allwise_original_valid``) and 2MASS
(``gaiadr1.tmass_original_valid``) alongside the official pre-computed
cross-match tables (``gaiadr3.allwise_best_neighbour``,
``gaiadr3.tmass_psc_xsc_best_neighbour``).  Using them rather than a positional
cross-match buys two things that matter enormously for this sample:

* **The official cross-match already propagates proper motion** to the epoch of
  the external catalogue (Marrese et al. 2019).  A naive positional match on a
  halo sample -- mean proper motion of order 100 mas/yr, 5.5 yr of Gaia-to-AllWISE
  baseline -- silently returns nothing for exactly the fastest, most interesting
  stars.  That bug cost a previous channel a whole run.
* ``number_of_neighbours`` and ``number_of_mates`` come free with the join and are
  a direct, catalogue-level crowding measure.

The independent verification that the propagation actually happened is done in
``vet.astrometry_gate``, which re-derives the epoch-propagated offset from the raw
positions rather than trusting the archive.

Three sample tracks
-------------------
``spec``   Gaia GSP-Spec [M/H] < -1.  Bright (G_RVS < 12) so WISE photometry is
           high signal-to-noise and not confusion-limited, and every star has a
           radial velocity, hence a full UVW.  The gold sample.
``phot``   Gaia GSP-Phot [M/H] < -1 with the *upper* confidence bound still
           metal-poor.  Breadth at the cost of per-star metallicity reliability.
``halo``   Pure kinematics: high tangential or radial velocity, no metallicity
           required at all.  This track tests the halo leg of the claim without
           depending on any spectroscopic pipeline.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

# AllWISE / 2MASS mirror column names vary between archive releases; every one is
# resolved from a live TOP-1 probe rather than assumed.
_ALLWISE_WANT = {
    "designation": ["designation", "allwise", "source_id"],
    "ra_wise": ["ra", "raj2000", "ra_pm"],
    "dec_wise": ["dec", "dej2000", "de", "dec_pm"],
    "W1mag": ["w1mpro"], "e_W1mag": ["w1mpro_error", "w1sigmpro"],
    "W2mag": ["w2mpro"], "e_W2mag": ["w2mpro_error", "w2sigmpro"],
    "W3mag": ["w3mpro"], "e_W3mag": ["w3mpro_error", "w3sigmpro"],
    "W4mag": ["w4mpro"], "e_W4mag": ["w4mpro_error", "w4sigmpro"],
    "cc_flags": ["cc_flags", "ccf"],
    "ph_qual": ["ph_qual", "qph"],
    "ext_flag": ["ext_flag", "ext_flg", "extended_flag"],
    "var_flag": ["var_flag", "var_flg"],
}

_TMASS_WANT = {
    "tmass_designation": ["designation", "source_id", "tmass_oid"],
    "Jmag": ["j_m", "jmag"], "e_Jmag": ["j_msigcom", "e_jmag", "j_cmsig"],
    "Hmag": ["h_m", "hmag"], "e_Hmag": ["h_msigcom", "e_hmag", "h_cmsig"],
    "Ksmag": ["ks_m", "kmag", "ksmag"],
    "e_Ksmag": ["ks_msigcom", "e_kmag", "ks_cmsig"],
    "tmass_ph_qual": ["ph_qual"],
}

_TMASS_XMATCH_ID = ["original_psc_source_id", "original_ext_source_id"]

# Declination bands used to chunk every pull.  Complete by construction (unlike a
# random_index slice) and each band returns a tractable row count.
_DEC_EDGES = list(range(-90, 91, 15))


def _run_query(query: str, retries: int = 4, tag: str = "ossuary",
               upload=None, upload_name: str = "ids") -> pd.DataFrame:
    """Robust Gaia ADQL: async with exponential backoff, sync on the last try."""
    from astroquery.gaia import Gaia

    last = None
    for attempt in range(retries):
        try:
            if upload is not None:
                job = Gaia.launch_job_async(query, upload_resource=upload,
                                            upload_table_name=upload_name)
            elif attempt == retries - 1:
                job = Gaia.launch_job(query)
            else:
                job = Gaia.launch_job_async(query)
            df = job.get_results().to_pandas()
            return df.rename(columns={c: c.lower() for c in df.columns})
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[{tag}] query attempt {attempt + 1}/{retries} failed: {exc!r}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gaia query failed after {retries} attempts: {last!r}")


def probe_columns(table: str, want: dict, tag: str = "ossuary") -> dict:
    """Resolve logical column names against a live TOP-1 probe of ``table``.

    Returns ``{logical_name: actual_column}`` for everything that exists.  A
    catalogue mirror that renamed a column degrades that one field instead of
    breaking the whole pull.
    """
    try:
        probe = _run_query(f"SELECT TOP 1 * FROM {table}", retries=2, tag=tag)
    except Exception as exc:  # noqa: BLE001
        print(f"[{tag}] column probe failed for {table}: {exc!r}")
        return {}
    have = {c.lower(): c for c in probe.columns}
    out = {}
    for logical, cands in want.items():
        for c in cands:
            if c.lower() in have:
                out[logical] = have[c.lower()]
                break
    missing = [k for k in want if k not in out]
    print(f"[{tag}] {table}: resolved {len(out)}/{len(want)} columns"
          + (f"; missing {missing}" if missing else ""))
    return out


_TRACK_WHERE = {
    "spec": """
      ap.mh_gspspec IS NOT NULL AND ap.mh_gspspec < {feh_max}
      AND g.parallax_over_error > {poe_min}
      AND g.ruwe < {ruwe_max}
      AND g.phot_g_mean_mag < {g_max}
    """,
    "phot": """
      g.mh_gspphot IS NOT NULL AND g.mh_gspphot < {feh_max}
      AND g.mh_gspphot_upper < {feh_upper_max}
      AND g.parallax_over_error > {poe_min}
      AND g.ruwe < {ruwe_max}
      AND g.phot_g_mean_mag < {g_max}
    """,
    # Pure kinematics.  The ADQL pre-filter is a superset of the true halo cut
    # (heliocentric tangential OR radial speed above a floor); the exact LSR-frame
    # classification happens client-side in ``kinematics.classify``.
    "halo": """
      g.parallax_over_error > {poe_min}
      AND g.ruwe < {ruwe_max}
      AND g.phot_g_mean_mag < {g_max}
      AND (
        SQRT(POWER(4.740470446 * g.pmra / g.parallax, 2)
           + POWER(4.740470446 * g.pmdec / g.parallax, 2)) > {v_floor}
        OR ABS(g.radial_velocity) > {v_floor}
      )
    """,
}


def _build_query(track: str, dec_lo: float, dec_hi: float, wise_cols: dict,
                 tmass_cols: dict, tmass_id: str | None, *, feh_max: float,
                 poe_min: float, ruwe_max: float, g_max: float,
                 v_floor: float, limit: int) -> str:
    gaia_cols = """
       g.source_id, g.ra, g.dec, g.l, g.b,
       g.parallax, g.parallax_error, g.parallax_over_error,
       g.pmra, g.pmra_error, g.pmdec, g.pmdec_error,
       g.radial_velocity, g.radial_velocity_error,
       g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.bp_rp,
       g.ruwe, g.astrometric_excess_noise, g.phot_variable_flag,
       g.non_single_star, g.mh_gspphot, g.teff_gspphot, g.logg_gspphot,
       g.ag_gspphot,
       ap.mh_gspspec, ap.teff_gspspec, ap.logg_gspspec, ap.alphafe_gspspec,
       ap.flags_gspspec,
       xw.angular_distance AS wise_angdist,
       xw.number_of_neighbours, xw.number_of_mates
    """
    wsel = ", ".join(f"w.{a} AS {logical.lower()}"
                     for logical, a in wise_cols.items() if logical != "designation")
    tsel = ""
    tjoin = ""
    if tmass_cols and tmass_id:
        tsel = ", " + ", ".join(
            f"t.{a} AS {logical.lower()}" for logical, a in tmass_cols.items()
            if logical != "tmass_designation")
        tjoin = f"""
  LEFT OUTER JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xt
         ON xt.source_id = g.source_id
  LEFT OUTER JOIN gaiadr1.tmass_original_valid AS t
         ON t.designation = xt.{tmass_id}"""

    where = _TRACK_WHERE[track].format(
        feh_max=feh_max, feh_upper_max=feh_max + 0.3, poe_min=poe_min,
        ruwe_max=ruwe_max, g_max=g_max, v_floor=v_floor)

    return f"""
SELECT TOP {int(limit)}
{gaia_cols},
  {wsel}{tsel}
FROM gaiadr3.gaia_source AS g
  LEFT OUTER JOIN gaiadr3.astrophysical_parameters AS ap
         ON ap.source_id = g.source_id
  JOIN gaiadr3.allwise_best_neighbour AS xw ON xw.source_id = g.source_id
  JOIN gaiadr1.allwise_original_valid AS w
         ON w.{wise_cols['designation']} = xw.original_ext_source_id{tjoin}
WHERE g.dec >= {dec_lo} AND g.dec < {dec_hi}
  AND {where}
"""


def fetch_track(track: str, out_dir: Path, *, feh_max: float = -1.0,
                poe_min: float = 5.0, ruwe_max: float = 1.4,
                g_max: float = 18.0, v_floor: float = 150.0,
                limit_per_band: int = 400_000,
                wise_cols: dict | None = None, tmass_cols: dict | None = None,
                tmass_id: str | None = None) -> pd.DataFrame:
    """Pull one sample track, one declination band at a time, checkpointed.

    Falls back to a 2MASS-free query if the 2MASS join fails, recording the
    degradation on the returned frame's ``attrs`` so the photosphere anchor can
    switch from Ks to G and the report can say the anchor was degraded.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if wise_cols is None:
        wise_cols = probe_columns("gaiadr1.allwise_original_valid", _ALLWISE_WANT)
    if not wise_cols.get("designation") or not wise_cols.get("W1mag"):
        raise RuntimeError("AllWISE mirror unusable: no designation/W1 column")
    if tmass_cols is None:
        tmass_cols = probe_columns("gaiadr1.tmass_original_valid", _TMASS_WANT)
    if tmass_id is None:
        xt = probe_columns("gaiadr3.tmass_psc_xsc_best_neighbour",
                           {"id": _TMASS_XMATCH_ID})
        tmass_id = xt.get("id")

    frames, degraded = [], False
    for lo, hi in zip(_DEC_EDGES[:-1], _DEC_EDGES[1:], strict=False):
        part = out_dir / f"{track}_dec_{lo:+04d}_{hi:+04d}.parquet"
        if part.exists():
            frames.append(pd.read_parquet(part))
            continue
        kw = dict(feh_max=feh_max, poe_min=poe_min, ruwe_max=ruwe_max,
                  g_max=g_max, v_floor=v_floor, limit=limit_per_band)
        try:
            q = _build_query(track, lo, hi, wise_cols, tmass_cols, tmass_id, **kw)
            df = _run_query(q, tag=f"ossuary/{track}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ossuary/{track}] dec [{lo},{hi}) with 2MASS failed "
                  f"({exc!r}); retrying without the 2MASS join")
            degraded = True
            q = _build_query(track, lo, hi, wise_cols, {}, None, **kw)
            df = _run_query(q, tag=f"ossuary/{track}")
        if len(df) >= limit_per_band:
            print(f"[ossuary/{track}] WARNING dec [{lo},{hi}) hit the row limit "
                  f"({limit_per_band}); this band is incomplete")
        df["dec_band_lo"] = lo
        df["track"] = track
        df.to_parquet(part, index=False)
        print(f"[ossuary/{track}] dec [{lo:+d},{hi:+d}): {len(df)} stars")
        frames.append(df)

    if not frames:
        out = pd.DataFrame()
    else:
        out = pd.concat(frames, ignore_index=True).drop_duplicates("source_id")
    out.attrs["tmass_degraded"] = degraded
    print(f"[ossuary/{track}] total {len(out)} stars"
          + (" (2MASS anchor degraded)" if degraded else ""))
    return out


def harmonise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and build the single ``feh`` column with provenance.

    Metallicity provenance is carried per star, because the natural-null argument
    is only as strong as the metallicity behind it: a GSP-Phot [M/H] is a far
    weaker claim than an APOGEE one, and the report must be able to say which.
    """
    out = df.rename(columns={
        "w1mag": "W1mag", "e_w1mag": "e_W1mag",
        "w2mag": "W2mag", "e_w2mag": "e_W2mag",
        "w3mag": "W3mag", "e_w3mag": "e_W3mag",
        "w4mag": "W4mag", "e_w4mag": "e_W4mag",
        "jmag": "Jmag", "e_jmag": "e_Jmag",
        "hmag": "Hmag", "e_hmag": "e_Hmag",
        "ksmag": "Ksmag", "e_ksmag": "e_Ksmag",
    }).copy()

    feh = pd.Series(np.nan, index=out.index, dtype=float)
    prov = pd.Series("none", index=out.index, dtype=object)
    # Weakest first, so the strongest source present wins.
    for col, name in (("mh_gspphot", "gaia_gspphot"),
                      ("mh_gspspec", "gaia_gspspec"),
                      ("feh_lamost", "lamost"),
                      ("feh_segue", "segue"),
                      ("feh_galah", "galah"),
                      ("feh_apogee", "apogee")):
        if col in out.columns:
            v = pd.to_numeric(out[col], errors="coerce")
            take = v.notna()
            feh[take] = v[take]
            prov[take] = name
    out["feh"] = feh
    out["feh_provenance"] = prov
    out["feh_is_spectroscopic"] = prov.isin(
        ["gaia_gspspec", "lamost", "segue", "galah", "apogee"])

    for src, dst in (("teff_gspspec", "teff"), ("logg_gspspec", "logg")):
        if dst not in out.columns and src in out.columns:
            out[dst] = pd.to_numeric(out[src], errors="coerce")
    for src, dst in (("teff_gspphot", "teff"), ("logg_gspphot", "logg")):
        if src in out.columns:
            v = pd.to_numeric(out[src], errors="coerce")
            out[dst] = v if dst not in out.columns else out[dst].fillna(v)
    return out


def gspspec_quality_ok(flags: pd.Series, max_level: int = 1,
                       n_leading: int = 13) -> pd.Series:
    """Gaia GSP-Spec quality: the leading ``flags_gspspec`` characters <= level.

    The flag string encodes per-symptom severity 0 (best) to 9; the Gaia
    documentation recommends keeping only low values of the leading vbroad/vrad
    biases when using [M/H].  Absent flags are treated as unknown-but-kept, with
    the metallicity provenance already recording that these are GSP-Spec values.
    """
    s = flags.astype(str).fillna("")
    ok = pd.Series(True, index=flags.index)
    for i in range(n_leading):
        ch = s.str.slice(i, i + 1)
        num = pd.to_numeric(ch, errors="coerce")
        ok &= (num <= max_level) | num.isna()
    return ok


def fetch_ebv(positions: pd.DataFrame, tag: str = "ossuary") -> pd.DataFrame:
    """SFD E(B-V) per candidate (small lists only) for the cirrus gate.

    Tries the IRSA DUST service first and ``dustmaps`` second.  Returns whatever
    it got; a missing reddening leaves the cirrus gate marked untested rather than
    silently passed.
    """
    rows = []
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astroquery.ipac.irsa.irsa_dust import IrsaDust
    except Exception as exc:  # noqa: BLE001
        print(f"[{tag}] IRSA DUST unavailable: {exc!r}")
        IrsaDust = None  # noqa: N806

    for _, r in positions.iterrows():
        ebv = np.nan
        if IrsaDust is not None:
            try:
                coord = SkyCoord(float(r["ra"]) * u.deg, float(r["dec"]) * u.deg)
                tbl = IrsaDust.get_query_table(coord, section="ebv")
                for c in ("ext SFD mean", "ext SandF mean", "ext SFD ref"):
                    if c in tbl.colnames:
                        ebv = float(tbl[c][0])
                        break
            except Exception as exc:  # noqa: BLE001
                print(f"[{tag}] IRSA DUST failed for {r.get('source_id')}: {exc!r}")
        rows.append({"source_id": r["source_id"], "ebv_sfd": ebv})
    out = pd.DataFrame(rows)
    n = int(np.isfinite(out["ebv_sfd"]).sum()) if len(out) else 0
    print(f"[{tag}] SFD E(B-V) retrieved for {n}/{len(out)} candidates")
    return out


def fetch_beam_neighbours(ra: float, dec: float, radius_arcsec: float = 12.0):
    """Gaia DR3 cone for the beam-blend test (delegates to the shared helper)."""
    from ..discriminate.blend import fetch_neighbours

    return fetch_neighbours(ra, dec, radius_arcsec=radius_arcsec)


def fetch_simbad(positions: pd.DataFrame, radius_arcsec: float = 5.0) -> pd.DataFrame:
    """SIMBAD identity for the shortlist (RR Lyrae, known disks, AGN, binaries)."""
    from ..acquire.science import fetch_simbad_context

    return fetch_simbad_context(positions, radius_arcsec=radius_arcsec)


__all__ = ["probe_columns", "fetch_track", "harmonise", "gspspec_quality_ok",
           "fetch_ebv", "fetch_beam_neighbours", "fetch_simbad"]
