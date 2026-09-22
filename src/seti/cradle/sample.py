"""The CRADLE parent sample: Gaia DR3 FGK dwarfs x AllWISE (W3 AND W4 >= 5 sigma).

Selected **in the archive**, over HEALPix ``source_id`` ranges, with the in-archive
cross-matches.  Pure functions here; the runner-only transport is in
:mod:`seti.cradle.acquire` (which reuses IGNITION's ESA transport ladder).

Why ``source_id`` ranges and not cones
--------------------------------------
Gaia DR3 ``source_id`` carries the HEALPix level-12 NESTED index in its most
significant bits (``source_id // 2**35``; Gaia documentation, "source_id").  A
level-``L`` pixel ``k`` is therefore the contiguous range

    ``k * 4**(12-L) * 2**35  <=  source_id  <  (k+1) * 4**(12-L) * 2**35``

and ``gaia_source`` is keyed on ``source_id``, so a range predicate is served by
the primary key --- the cheapest possible sky chunk, complete by construction,
and trivially sharded (pixel ``k`` goes to shard ``k mod n``).  IGNITION's cones
worked at ~20 s each (run 35038504272) but a cone is not a tiling; 768 level-3
pixels ARE the whole sky, with no gaps and no double counting.

Why the join is written inner-first
-----------------------------------
What IGNITION learned (docs/ignition.md): a flat ``WHERE`` over the three-table
Gaia x AllWISE join lets the planner start from the 750-million-row AllWISE
mirror; putting every Gaia-only cut in an inner sub-select on ``gaia_source``
ALONE and joining AllWISE to that small result is what answered in 18 s.  The
same shape is used here, and the extra tables (``astrophysical_parameters`` for
FLAME/GSP-Spec ages, 2MASS for the K_s anchor, ``vari_rotation_modulation``
for a rotation period) hang off it as ``LEFT OUTER JOIN`` on ``source_id`` ---
primary-key lookups on the already-small intermediate.

The cuts, once, in :func:`gaia_predicates` / :func:`allwise_predicates`
--------------------------------------------------------------------------
* Gaia: ``G < g_max``, ``parallax > parallax_min_mas``, ``parallax_over_error >
  10``, ``|b| > 10``, ``ruwe < 1.4``, ``bp_rp`` in the FGK range, and the
  absolute-magnitude dwarf cut ``M_G > a + s (bp_rp - c0)``;
* AllWISE: **W3 AND W4 each >= 5 sigma**, spelled as ``wXmpro_error <
  1.0857/5 = 0.2171`` because the ESA mirror carries magnitude errors and no
  SNR column; ``ext_flag``, ``cc_flags``, the registration offset and
  ``K_s - W1 < 0.2`` are applied and COUNTED in :mod:`seti.cradle.vet`, not
  hidden in SQL.

``select_parent`` re-applies every Gaia cut locally (belt and braces, and the
only place the cuts live when a fallback route serves the Gaia half).
"""

from __future__ import annotations

import re
import textwrap

import numpy as np
import pandas as pd

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"

#: source_id = healpix_level12_nested * 2**35 + running number
L12_MULT = 2 ** 35
N_L12 = 12 * 4 ** 12

#: v_tan [km/s] = _K * mu [mas/yr] / parallax [mas]
_K = 4.740470446

#: 5-sigma in magnitudes: 2.5 / ln(10) / 5
SNR5_MAG_ERR = 2.5 / np.log(10.0) / 5.0

SHAPE_FULL = "full"                  # + astrophysical_parameters + 2MASS (direct) + rotation
SHAPE_FULL_VIA_JOIN = "full_via_join"  # 2MASS through tmass_psc_xsc_join (DR3's documented path)
SHAPE_LEAN = "lean"                  # + astrophysical_parameters only
SHAPE_GAIA_ONLY = "gaia_only"        # the Gaia half of the IRSA route
JOINED_SHAPES: tuple[str, ...] = (SHAPE_FULL, SHAPE_FULL_VIA_JOIN, SHAPE_LEAN)
SHAPES: tuple[str, ...] = JOINED_SHAPES + (SHAPE_GAIA_ONLY,)

ROUTE_ESA = "esa_gaia"
ROUTE_IRSA = "irsa_tap"

_ALIAS_RE = re.compile(r"\bg\.")
INNER_ALIAS = "gs"

DEFAULT_SAMPLE: dict = {
    "healpix_level": 3,             # 768 units over the whole sky
    "healpix_split_max_level": 6,   # a unit that times out is split into its 4 children
    "g_max": 13.5,
    "parallax_min_mas": 2.0,        # d < 500 pc
    "parallax_over_error_min": 10.0,
    "abs_b_min_deg": 10.0,
    "ruwe_max": 1.4,
    "bp_rp_min": 0.45,              # ~F2
    "bp_rp_max": 1.85,              # ~K7
    "dwarf_cut": {"intercept": 2.0, "slope": 3.3, "colour0": 0.6},
    "w3_snr_min": 5.0,
    "w4_snr_min": 5.0,
    "inner_top": 400000,            # bound on the inner sub-select; never on COUNT(*)
    "query_timeout_s": 1200.0,
    "count_parent": True,
    "query_shape": "auto",
    # Column spellings in the ESA mirrors.  IGNITION's ESA run used these exact
    # AllWISE names (846 rows, no route errors); the 2MASS spellings are the
    # Gaia DR1 mirror's and are re-resolved by the probe.
    "allwise_columns": {"designation": "designation", "ra": "ra", "dec": "dec",
                        "w1mpro": "w1mpro", "w1mpro_error": "w1mpro_error",
                        "w2mpro": "w2mpro", "w2mpro_error": "w2mpro_error",
                        "w3mpro": "w3mpro", "w3mpro_error": "w3mpro_error",
                        "w4mpro": "w4mpro", "w4mpro_error": "w4mpro_error",
                        "cc_flags": "cc_flags", "ext_flag": "ext_flag",
                        "var_flag": "var_flag", "ph_qual": "ph_qual"},
    "tmass_columns": {"designation": "designation", "j_m": "j_m", "j_msigcom": "j_msigcom",
                      "h_m": "h_m", "h_msigcom": "h_msigcom", "ks_m": "ks_m",
                      "ks_msigcom": "ks_msigcom", "ph_qual": "ph_qual"},
    "tmass_xmatch_id": "original_ext_source_id",
    "rotation_table": "gaiadr3.vari_rotation_modulation",
    "rotation_period_column": "best_rotation_period",
}

#: Known mature extreme debris disks, the channel's controls.  Positions are
#: ASSERTED (J2000, from SIMBAD as remembered) and re-resolved on the runner;
#: the record says which resolver answered.  TYC 4479-3-1 has no asserted
#: position and is resolved by name only.
CONTROLS: tuple[dict, ...] = (
    {"name": "BD+20 307", "aliases": ["HIP 8920", "BD+20 307"], "ra": 28.7098, "dec": 21.3063,
     "note": "~1 Gyr, T_dust ~ 400-450 K, f ~ 3e-2 (Song+2005, Weinberger+2011); WD companion"},
    {"name": "TYC 4479-3-1", "aliases": ["TYC 4479-3-1"], "ra": None, "dec": None,
     "note": "5 +/- 2 Gyr, ~400 K (Moor+2021 EDD catalogue)"},
    {"name": "HD 15407A", "aliases": ["HD 15407", "HIP 11696", "HD 15407A"], "ra": 37.7113,
     "dec": 55.5484,
     "note": "2.1 Gyr isochrone vs 80 Myr AB Dor membership: the mis-age warning; ~500-800 K"},
)

GAIA_COLS = (
    "g.source_id, g.ra, g.dec, g.l, g.b, g.parallax, g.parallax_error, g.parallax_over_error, "
    "g.pmra, g.pmra_error, g.pmdec, g.pmdec_error, g.radial_velocity, g.radial_velocity_error, "
    "g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.bp_rp, g.ruwe, "
    "g.astrometric_excess_noise, g.ipd_frac_multi_peak, g.phot_variable_flag, "
    "g.non_single_star, g.teff_gspphot, g.logg_gspphot, g.mh_gspphot, g.ag_gspphot, "
    "g.random_index"
)

AP_COLS = (
    "ap.age_flame, ap.age_flame_lower, ap.age_flame_upper, ap.mass_flame, ap.mass_flame_lower, "
    "ap.mass_flame_upper, ap.lum_flame, ap.lum_flame_lower, ap.lum_flame_upper, "
    "ap.radius_flame, ap.evolstage_flame, ap.flags_flame, ap.teff_gspspec, ap.logg_gspspec, "
    "ap.mh_gspspec, ap.alphafe_gspspec, ap.alphafe_gspspec_lower, ap.alphafe_gspspec_upper, "
    "ap.flags_gspspec, ap.activityindex_espcs, ap.activityindex_espcs_uncertainty, "
    "ap.spectraltype_esphs"
)


# ---------------------------------------------------------------------------
# HEALPix units
# ---------------------------------------------------------------------------
def unit_range(level: int, k: int) -> tuple[int, int]:
    """``[sid_lo, sid_hi)`` of NESTED pixel ``k`` at ``level`` in source_id space."""
    per = 4 ** (12 - int(level))
    lo = int(k) * per * L12_MULT
    hi = (int(k) + 1) * per * L12_MULT
    return lo, hi


def healpix_units(level: int) -> list[dict]:
    """Every pixel at ``level`` as a unit ``{level, k, sid_lo, sid_hi}``."""
    n = 12 * 4 ** int(level)
    out = []
    for k in range(n):
        lo, hi = unit_range(level, k)
        out.append({"level": int(level), "k": int(k), "sid_lo": lo, "sid_hi": hi})
    return out


def unit_children(unit: dict) -> list[dict]:
    """The four NESTED children of a unit (one level down)."""
    lvl = int(unit["level"]) + 1
    out = []
    for j in range(4):
        k = 4 * int(unit["k"]) + j
        lo, hi = unit_range(lvl, k)
        out.append({"level": lvl, "k": k, "sid_lo": lo, "sid_hi": hi})
    return out


def unit_label(unit: dict) -> str:
    return f"hp{int(unit['level'])}_{int(unit['k'])}"


def shard_units(units: list[dict], shard: int, n_shards: int) -> list[dict]:
    """Pixel ``k`` goes to shard ``k mod n``: balanced and reproducible."""
    n = max(int(n_shards), 1)
    return [u for u in units if int(u["k"]) % n == int(shard) % n]


# ---------------------------------------------------------------------------
# Predicates (one definition for every shape)
# ---------------------------------------------------------------------------
def gaia_predicates(conf: dict | None = None, *, sid_lo: int | None = None,
                    sid_hi: int | None = None, cone: dict | None = None) -> list[str]:
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    dc = {**DEFAULT_SAMPLE["dwarf_cut"], **(c.get("dwarf_cut") or {})}
    out: list[str] = []
    if sid_lo is not None and sid_hi is not None:
        out.append(f"g.source_id >= {int(sid_lo)} AND g.source_id < {int(sid_hi)}")
    if cone is not None:
        out.append(f"1 = CONTAINS(POINT('ICRS', g.ra, g.dec), "
                   f"CIRCLE('ICRS', {float(cone['ra'])}, {float(cone['dec'])}, "
                   f"{float(cone['radius_deg'])}))")
    out += [
        f"g.phot_g_mean_mag < {float(c['g_max'])}",
        f"g.parallax > {float(c['parallax_min_mas'])}",
        f"g.parallax_over_error > {float(c['parallax_over_error_min'])}",
        f"ABS(g.b) > {float(c['abs_b_min_deg'])}",
        f"g.ruwe < {float(c['ruwe_max'])}",
        f"g.bp_rp > {float(c['bp_rp_min'])} AND g.bp_rp < {float(c['bp_rp_max'])}",
        (f"g.phot_g_mean_mag + 5 * LOG10(g.parallax) - 10 > {float(dc['intercept'])} + "
         f"{float(dc['slope'])} * (g.bp_rp - {float(dc['colour0'])})"),
    ]
    return out


def allwise_predicates(conf: dict | None = None, *, controls: bool = False) -> list[str]:
    """W3 AND W4 >= 5 sigma (as magnitude errors).  Off for control cones."""
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    ac = {**DEFAULT_SAMPLE["allwise_columns"], **(c.get("allwise_columns") or {})}
    if controls:
        return []
    e3 = 2.5 / np.log(10.0) / float(c["w3_snr_min"])
    e4 = 2.5 / np.log(10.0) / float(c["w4_snr_min"])
    return [
        f"w.{ac['w3mpro_error']} IS NOT NULL AND w.{ac['w3mpro_error']} < {e3:.5f}",
        f"w.{ac['w4mpro_error']} IS NOT NULL AND w.{ac['w4mpro_error']} < {e4:.5f}",
    ]


def _wise_select(conf: dict) -> str:
    ac = {**DEFAULT_SAMPLE["allwise_columns"], **(conf.get("allwise_columns") or {})}
    return (f"w.{ac['designation']} AS allwise_designation, w.{ac['ra']} AS allwise_ra, "
            f"w.{ac['dec']} AS allwise_dec, "
            f"w.{ac['w1mpro']} AS w1mpro, w.{ac['w1mpro_error']} AS w1mpro_error, "
            f"w.{ac['w2mpro']} AS w2mpro, w.{ac['w2mpro_error']} AS w2mpro_error, "
            f"w.{ac['w3mpro']} AS w3mpro, w.{ac['w3mpro_error']} AS w3mpro_error, "
            f"w.{ac['w4mpro']} AS w4mpro, w.{ac['w4mpro_error']} AS w4mpro_error, "
            f"w.{ac['cc_flags']} AS cc_flags, w.{ac['ext_flag']} AS ext_flag, "
            f"w.{ac['var_flag']} AS var_flag, w.{ac['ph_qual']} AS ph_qual, "
            "xw.angular_distance AS allwise_sep_arcsec, "
            "xw.number_of_neighbours AS allwise_n_neighbours, "
            "xw.number_of_mates AS allwise_n_mates")


def _tmass_select(conf: dict) -> str:
    tc = {**DEFAULT_SAMPLE["tmass_columns"], **(conf.get("tmass_columns") or {})}
    return (f"t.{tc['designation']} AS tmass_designation, t.{tc['j_m']} AS j_m, "
            f"t.{tc['j_msigcom']} AS j_msigcom, t.{tc['h_m']} AS h_m, "
            f"t.{tc['h_msigcom']} AS h_msigcom, t.{tc['ks_m']} AS ks_m, "
            f"t.{tc['ks_msigcom']} AS ks_msigcom, t.{tc['ph_qual']} AS tmass_ph_qual, "
            "xt.angular_distance AS tmass_sep_arcsec, "
            "xt.number_of_neighbours AS tmass_n_neighbours")


def _rot_select(conf: dict) -> str:
    col = str(conf.get("rotation_period_column") or "best_rotation_period")
    return f"rot.{col} AS gaia_rot_period_d"


def _realias(text: str, alias: str = INNER_ALIAS) -> str:
    return _ALIAS_RE.sub(f"{alias}.", text)


def build_query(conf: dict | None = None, *, unit: dict | None = None, cone: dict | None = None,
                shape: str = SHAPE_FULL, count_only: bool = False,
                controls: bool = False, top: int | None = None) -> str:
    """ADQL for one HEALPix unit (or one control cone), in one of :data:`SHAPES`.

    ``count_only`` returns ``COUNT(*)`` of the Gaia-only selection (the
    denominator: how many FGK dwarfs the unit holds before the WISE cut) when
    ``shape == gaia_only``, or of the joined selection otherwise.
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    shape = str(shape or SHAPE_FULL)
    if shape not in SHAPES:
        raise ValueError(f"unknown query shape {shape!r}; choose from {SHAPES}")
    ac = {**DEFAULT_SAMPLE["allwise_columns"], **(c.get("allwise_columns") or {})}
    tc = {**DEFAULT_SAMPLE["tmass_columns"], **(c.get("tmass_columns") or {})}
    sid_lo = sid_hi = None
    if unit is not None:
        sid_lo, sid_hi = int(unit["sid_lo"]), int(unit["sid_hi"])
    gp = gaia_predicates(c, sid_lo=sid_lo, sid_hi=sid_hi, cone=cone)
    wp = allwise_predicates(c, controls=controls)
    itop = None if count_only else int(c.get("inner_top") or 0) or None

    if shape == SHAPE_GAIA_ONLY:
        where = "\n  AND ".join(gp)
        if count_only:
            return f"SELECT COUNT(*) AS n\nFROM gaiadr3.gaia_source AS g\nWHERE {where}"
        t = f"TOP {int(top)} " if top else ""
        return f"SELECT {t}{GAIA_COLS}\nFROM gaiadr3.gaia_source AS g\nWHERE {where}"

    inner = (f"SELECT {f'TOP {itop} ' if itop else ''}{_realias(GAIA_COLS)}\n"
             f"FROM gaiadr3.gaia_source AS {INNER_ALIAS}\n"
             "WHERE " + "\n  AND ".join(_realias(p) for p in gp))
    src = "(\n" + textwrap.indent(inner, "  ") + "\n) AS g"
    joins = [
        f"FROM {src}",
        "  JOIN gaiadr3.allwise_best_neighbour AS xw ON xw.source_id = g.source_id",
        "  JOIN gaiadr1.allwise_original_valid AS w",
        f"    ON w.{ac['designation']} = xw.original_ext_source_id",
        "  LEFT OUTER JOIN gaiadr3.astrophysical_parameters AS ap ON ap.source_id = g.source_id",
    ]
    sel = [GAIA_COLS, AP_COLS, _wise_select(c)]
    if shape in (SHAPE_FULL, SHAPE_FULL_VIA_JOIN):
        joins.append("  LEFT OUTER JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xt "
                     "ON xt.source_id = g.source_id")
        if shape == SHAPE_FULL:
            xid = str(c.get("tmass_xmatch_id") or "original_ext_source_id")
            joins.append(f"  LEFT OUTER JOIN gaiadr1.tmass_original_valid AS t "
                         f"ON t.{tc['designation']} = xt.{xid}")
        else:
            joins.append("  LEFT OUTER JOIN gaiadr3.tmass_psc_xsc_join AS tj "
                         "ON tj.clean_tmass_psc_xsc_oid = xt.clean_tmass_psc_xsc_oid")
            joins.append(f"  LEFT OUTER JOIN gaiadr1.tmass_original_valid AS t "
                         f"ON t.{tc['designation']} = tj.original_psc_source_id")
        joins.append(f"  LEFT OUTER JOIN {c['rotation_table']} AS rot "
                     "ON rot.source_id = g.source_id")
        sel += [_tmass_select(c), _rot_select(c)]
    tail = ("\nWHERE " + "\n  AND ".join(wp)) if wp else ""
    body = "\n".join(joins) + tail
    if count_only:
        return f"SELECT COUNT(*) AS n\n{body}"
    t = f"TOP {int(top)} " if top else ""
    return f"SELECT {t}" + ",\n  ".join(sel) + "\n" + body


def parent_columns() -> list[str]:
    """Every lowercase column name the ``full`` select produces (the frame contract)."""
    text = f"{GAIA_COLS}, {AP_COLS}, {_wise_select(DEFAULT_SAMPLE)}, " \
           f"{_tmass_select(DEFAULT_SAMPLE)}, {_rot_select(DEFAULT_SAMPLE)}"
    out: list[str] = []
    for item in text.split(","):
        piece = item.strip()
        if not piece:
            continue
        name = piece.split(" AS ")[-1] if " AS " in piece else piece.split(".")[-1]
        out.append(name.strip().lower())
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Local re-application (belt and braces; the only place the cuts live for a
# fallback route)
# ---------------------------------------------------------------------------
def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(np.nan, index=df.index)), errors="coerce")


def select_parent(df: pd.DataFrame, conf: dict | None = None, *,
                  apply_wise: bool = True) -> tuple[pd.DataFrame, dict]:
    """Re-apply the Gaia and the W3/W4 SNR cuts locally; add derived columns.

    Adds ``abs_g``, ``v_tan_kms``, ``w3_snr``, ``w4_snr``, ``ks_w1`` and the
    ``is_control`` column (kept if present).  Control rows are exempt from the
    WISE SNR cut (they are re-identified wherever they land) but not from the
    Gaia cuts, which are the definition of the population.
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    dc = {**DEFAULT_SAMPLE["dwarf_cut"], **(c.get("dwarf_cut") or {})}
    d = df.copy()
    d.columns = [str(x).lower() for x in d.columns]
    counters: dict = {"n_in": int(len(d))}
    if not len(d):
        counters["n_out"] = 0
        return d, counters
    d = d.reset_index(drop=True)
    if "is_control" not in d:
        d["is_control"] = False
    d["is_control"] = d["is_control"].fillna(False).astype(bool)
    plx = _num(d, "parallax")
    d["abs_g"] = _num(d, "phot_g_mean_mag") + 5.0 * np.log10(plx.clip(lower=1e-6)) - 10.0
    mu = np.hypot(_num(d, "pmra").fillna(0.0), _num(d, "pmdec").fillna(0.0))
    d["v_tan_kms"] = _K * mu / plx.clip(lower=1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["w3_snr"] = 1.0857 / _num(d, "w3mpro_error")
        d["w4_snr"] = 1.0857 / _num(d, "w4mpro_error")
    d["ks_w1"] = _num(d, "ks_m") - _num(d, "w1mpro")
    d["w1_w2"] = _num(d, "w1mpro") - _num(d, "w2mpro")

    ctrl = d["is_control"].to_numpy(bool)
    masks: list[tuple[str, pd.Series]] = [
        ("g_max", _num(d, "phot_g_mean_mag") < float(c["g_max"])),
        ("parallax", plx > float(c["parallax_min_mas"])),
        ("parallax_over_error", _num(d, "parallax_over_error") > float(c["parallax_over_error_min"])),
    ]
    if _num(d, "b").notna().any():
        masks.append(("galactic_latitude", _num(d, "b").abs() > float(c["abs_b_min_deg"])))
    masks.append(("ruwe", _num(d, "ruwe") < float(c["ruwe_max"])))
    masks.append(("colour", (_num(d, "bp_rp") > float(c["bp_rp_min"]))
                  & (_num(d, "bp_rp") < float(c["bp_rp_max"]))))
    masks.append(("dwarf", d["abs_g"] > float(dc["intercept"]) + float(dc["slope"])
                  * (_num(d, "bp_rp") - float(dc["colour0"]))))
    if apply_wise:
        snr_ok = ((d["w3_snr"] >= float(c["w3_snr_min"]))
                  & (d["w4_snr"] >= float(c["w4_snr_min"])))
        masks.append(("w3_w4_5sigma", snr_ok | pd.Series(ctrl, index=d.index)))
    alive = pd.Series(True, index=d.index)
    for name, keep in masks:
        keep = keep.fillna(False).astype(bool)
        counters[f"cut_{name}"] = int((alive & ~keep).sum())
        alive &= keep
    d = d[alive].reset_index(drop=True)
    counters["n_out"] = int(len(d))
    counters["n_controls"] = int(d["is_control"].sum())
    return d, counters


__all__ = ["AP_COLS", "CONTROLS", "DEFAULT_SAMPLE", "GAIA_COLS", "GAIA_TAP", "JOINED_SHAPES",
           "L12_MULT", "N_L12", "ROUTE_ESA", "ROUTE_IRSA", "SHAPES", "SHAPE_FULL",
           "SHAPE_FULL_VIA_JOIN", "SHAPE_GAIA_ONLY", "SHAPE_LEAN", "SNR5_MAG_ERR",
           "allwise_predicates", "build_query", "gaia_predicates", "healpix_units",
           "parent_columns", "select_parent", "shard_units", "unit_children", "unit_label",
           "unit_range"]
