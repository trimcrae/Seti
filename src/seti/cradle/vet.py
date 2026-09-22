"""The CRADLE kill list.  Every rule has a name, a counter, and a reason string.

The four contaminants the mission names, and what kills each:

* **Background dusty galaxies in the 6-12" WISE PSF** (every Project Hephaistos
  candidate): ``ext_flag``, AllWISE ``n_mates`` (another Gaia source shares the
  WISE source), Gaia beam neighbours from the enrichment cone, IRSA's profile-fit
  ``w3rchi2``/``w4rchi2``, the W1-W2 / W2-W3 galaxy colours, and SIMBAD.
* **Cirrus at W4**: both W3 AND W4 must be in excess at the same temperature
  (the SED fit), ``cc_flags`` in W3/W4 clean, and the SFD reddening gate.
* **Sco-Cen / rho Oph mis-aged young stars**: the region boxes and the
  moving-group veto in :mod:`seti.cradle.ages` (position + kinematics).
* **Wide companions** (Gaia within 1000 AU, parallax- and PM-consistent) and
  unresolved ones (``non_single_star``, RUWE) --- comet delivery.
* **Mantle-only impact debris** cannot be excluded photometrically; it is the
  S53 mineralogy's job and is listed among the systematics NOT excluded for
  every candidate without an IRS spectrum.

Rules that use an enrichment column (a Gaia cone, an IRSA row, SIMBAD, E(B-V))
are ``untested`` when that column is absent, never silently passed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_VET: dict = {
    "ks_w1_max": 0.2,
    "registration_max_arcsec": 1.0,
    "w3_saturation_mag": 3.8,
    "w4_saturation_mag": -0.5,
    "beam_neighbour_radius_arcsec": 6.5,     # W3 FWHM
    "beam_neighbour_dg_max": 4.0,
    "w4_beam_radius_arcsec": 12.0,           # W4 FWHM
    "w4_beam_dg_max": 2.0,
    "wide_companion_au": 1000.0,
    "wide_companion_plx_frac": 0.2,
    "wide_companion_pm_tol_mas_yr": 5.0,
    "wise_rchi2_max": 3.0,
    "agn_w1w2_min": 0.5,
    "galaxy_w2w3_min": 3.5,
    "galaxy_w1w2_min": 0.3,
    "ebv_kill": 0.2,
    "ebv_flag": 0.1,
    "bb_chi2_flag": 9.0,
    "simbad_kill_types": ["G", "AGN", "QSO", "Sy", "LIN", "BLL", "Bla", "YSO", "TTau", "TT*",
                          "Or*", "AGB", "Mi*", "LP*", "C*", "S*", "PN", "pA*", "RG*", "HII",
                          "Ae*", "Be*", "EmO", "Y*O", "Y*?", "IR", "sg*", "s*b", "s*r", "s*y"],
    "t_cell_k": [250.0, 350.0],
    "log_f_fmax_min": 3.0,
}

#: cc_flags characters that mean contamination (WISE Explanatory Supplement)
_CC_BAD = set("DPHOdpho")


def _num(d: pd.DataFrame, col: str) -> np.ndarray:
    if col not in d.columns:
        return np.full(len(d), np.nan, dtype=float)
    return pd.to_numeric(d[col], errors="coerce").to_numpy(float)


def _str(d: pd.DataFrame, col: str) -> np.ndarray:
    if col not in d.columns:
        return np.full(len(d), "", dtype=object)
    return d[col].fillna("").astype(str).str.strip().to_numpy(dtype=object)


def _cc(flags: np.ndarray) -> np.ndarray:
    """Normalise cc_flags to four characters (an integer 0 arrives as '0')."""
    out = []
    for f in flags:
        s = str(f).strip()
        if s.isdigit() and len(s) < 4:
            s = s.zfill(4)
        out.append(s)
    return np.asarray(out, dtype=object)


def apply_rules(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Every rule on every row.  Adds ``kill_reasons``, ``flags``, ``untested``,
    ``killed`` and per-rule boolean columns ``rule_<name>``; returns counters.
    """
    c = {**DEFAULT_VET, **(cfg or {})}
    n = len(df)
    out = df.copy()
    kills: dict[str, np.ndarray] = {}
    flags: dict[str, np.ndarray] = {}
    untested: dict[str, np.ndarray] = {}

    ks_w1 = _num(out, "ks_w1")
    if not np.isfinite(ks_w1).any() and "ks_m" in out:
        ks_w1 = _num(out, "ks_m") - _num(out, "w1mpro")
    ks = _num(out, "ks_m")
    kills["ks_missing"] = ~np.isfinite(ks)
    kills["ks_w1_not_photospheric"] = np.isfinite(ks_w1) & (ks_w1 >= float(c["ks_w1_max"]))
    sep = _num(out, "allwise_sep_arcsec")
    kills["registration_gt_1arcsec"] = np.isfinite(sep) & (sep > float(c["registration_max_arcsec"]))
    untested["registration_gt_1arcsec"] = ~np.isfinite(sep)
    ext = _num(out, "ext_flag")
    kills["ext_flag"] = np.isfinite(ext) & (ext > 0)
    cc = _cc(_str(out, "cc_flags"))
    w34_bad = np.array([len(s) >= 4 and (s[2] in _CC_BAD or s[3] in _CC_BAD) for s in cc])
    w12_bad = np.array([len(s) >= 2 and (s[0] in _CC_BAD or s[1] in _CC_BAD) for s in cc])
    kills["cc_flags_w3w4"] = w34_bad
    flags["cc_flags_w1w2"] = w12_bad
    kills["w3_saturated"] = _num(out, "w3mpro") < float(c["w3_saturation_mag"])
    kills["w4_saturated"] = _num(out, "w4mpro") < float(c["w4_saturation_mag"])
    mates = _num(out, "allwise_n_mates")
    kills["allwise_shared_by_gaia_mates"] = np.isfinite(mates) & (mates > 0)
    nnb = _num(out, "allwise_n_neighbours")
    flags["allwise_multiple_neighbours"] = np.isfinite(nnb) & (nnb > 1)
    # Gaia beam neighbours (enrichment)
    nb6 = _num(out, "n_gaia_beam_neighbours")
    kills["gaia_beam_neighbour"] = np.isfinite(nb6) & (nb6 > 0)
    untested["gaia_beam_neighbour"] = ~np.isfinite(nb6)
    nb12 = _num(out, "n_gaia_w4beam_neighbours")
    flags["gaia_w4_beam_neighbour"] = np.isfinite(nb12) & (nb12 > 0)
    wide = _num(out, "n_wide_companions")
    kills["wide_companion_1000au"] = np.isfinite(wide) & (wide > 0)
    untested["wide_companion_1000au"] = ~np.isfinite(wide)
    nss = _num(out, "non_single_star")
    kills["non_single_star"] = np.isfinite(nss) & (nss > 0)
    # IRSA profile fits (enrichment)
    r3, r4 = _num(out, "w3rchi2"), _num(out, "w4rchi2")
    kills["wise_profile_misfit"] = (np.isfinite(r3) & (r3 > float(c["wise_rchi2_max"]))) | \
        (np.isfinite(r4) & (r4 > float(c["wise_rchi2_max"])))
    untested["wise_profile_misfit"] = ~np.isfinite(r3) & ~np.isfinite(r4)
    # colours
    w1w2 = _num(out, "w1mpro") - _num(out, "w2mpro")
    w2w3 = _num(out, "w2mpro") - _num(out, "w3mpro")
    kills["agn_colour"] = np.isfinite(w1w2) & (w1w2 > float(c["agn_w1w2_min"]))
    kills["galaxy_colour"] = np.isfinite(w1w2) & np.isfinite(w2w3) & \
        (w1w2 > float(c["galaxy_w1w2_min"])) & (w2w3 > float(c["galaxy_w2w3_min"]))
    # SIMBAD (enrichment)
    st = _str(out, "simbad_otype")
    tested = np.array([s != "" for s in st])
    kill_types = tuple(str(x) for x in c["simbad_kill_types"])
    kills["simbad_type"] = np.array([any(s.startswith(t) for t in kill_types) for s in st])
    untested["simbad_type"] = ~tested if "simbad_otype" in out else np.ones(n, bool)
    # cirrus (enrichment)
    ebv = _num(out, "ebv_sfd")
    kills["cirrus_ebv"] = np.isfinite(ebv) & (ebv > float(c["ebv_kill"]))
    flags["cirrus_ebv_flag"] = np.isfinite(ebv) & (ebv > float(c["ebv_flag"])) & \
        (ebv <= float(c["ebv_kill"]))
    untested["cirrus_ebv"] = ~np.isfinite(ebv)
    # young (from ages)
    kills["young_group"] = np.array([s != "" for s in _str(out, "young_group")])
    kills["young_region"] = np.array([s != "" for s in _str(out, "young_region")])
    kills["young_indicator"] = _num(out, "n_young_indicators") > 0
    # SED shape and Gaia flags
    chi2 = _num(out, "bb_fit_chi2")
    flags["sed_not_single_blackbody"] = np.isfinite(chi2) & (chi2 > float(c["bb_chi2_flag"]))
    flags["gaia_variable"] = np.array([s.upper() == "VARIABLE" for s in _str(out, "phot_variable_flag")])
    ipd = _num(out, "ipd_frac_multi_peak")
    flags["ipd_multi_peak"] = np.isfinite(ipd) & (ipd > 10)
    flags["previously_catalogued_excess"] = _num(out, "n_known_disk_matches") > 0
    flags["neowise_variable"] = np.array([bool(x) for x in out.get("neowise_variable",
                                                                    pd.Series([False] * n))])

    reasons = [[] for _ in range(n)]
    for name, m in kills.items():
        m = np.asarray(m, bool)
        out[f"rule_{name}"] = m
        for i in np.nonzero(m)[0]:
            reasons[i].append(name)
    fl = [[] for _ in range(n)]
    for name, m in flags.items():
        m = np.asarray(m, bool)
        out[f"flag_{name}"] = m
        for i in np.nonzero(m)[0]:
            fl[i].append(name)
    ut = [[] for _ in range(n)]
    for name, m in untested.items():
        m = np.asarray(m, bool)
        for i in np.nonzero(m)[0]:
            ut[i].append(name)
    out["kill_reasons"] = [";".join(r) for r in reasons]
    out["flags"] = [";".join(r) for r in fl]
    out["untested"] = [";".join(r) for r in ut]
    out["killed"] = np.array([len(r) > 0 for r in reasons])
    counters = {"n_in": int(n), "kills": {k: int(np.asarray(v, bool).sum()) for k, v in kills.items()},
                "flags": {k: int(np.asarray(v, bool).sum()) for k, v in flags.items()},
                "untested": {k: int(np.asarray(v, bool).sum()) for k, v in untested.items()},
                "n_killed": int(out["killed"].sum())}
    return out, counters


def classify(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The final class per row, after :func:`apply_rules` and the age assessment.

    ``CANDIDATE``                in the cell, mature (>= 2 indicators), no kill
    ``IN_CELL_AGE_UNDETERMINED`` in the cell, one or zero old indicators, no kill
    ``IN_CELL_KILLED``           in the cell, at least one kill (``kill_reasons``)
    ``ABOVE_FMAX_HOT`` / ``_COLD`` log(f/f_max) > 3 but T_bb outside the cell
    ``BELOW_FMAX``               excess significant, log(f/f_max) <= 3
    ``NOT_SIGNIFICANT``          no >= 3-sigma excess in both W3 and W4
    ``KS_MISSING``               no 2MASS K_s anchor: not assessable
    ``CONTROL``                  prefix for the control stars (their landing is reported)
    """
    c = {**DEFAULT_VET, **(cfg or {})}
    n = len(df)
    out = df.copy()
    tlo, thi = (float(x) for x in c["t_cell_k"])
    t = _num(out, "t_bb_k")
    sig = out["excess_significant"].fillna(False).to_numpy(bool) if "excess_significant" in out \
        else np.zeros(n, bool)
    lff_lo = _num(out, "log_f_fmax_1gyr_lo")
    above = np.isfinite(lff_lo) & (lff_lo > float(c["log_f_fmax_min"]))
    ks = np.isfinite(_num(out, "ks_m"))
    killed = out["killed"].to_numpy(bool) if "killed" in out else np.zeros(n, bool)
    age = _str(out, "age_class")
    in_cell = sig & above & np.isfinite(t) & (t >= tlo) & (t <= thi)
    cls = np.full(n, "NOT_SIGNIFICANT", dtype=object)
    cls[~ks] = "KS_MISSING"
    cls[ks & sig & ~above] = "BELOW_FMAX"
    cls[ks & sig & above & np.isfinite(t) & (t > thi)] = "ABOVE_FMAX_HOT"
    cls[ks & sig & above & np.isfinite(t) & (t < tlo)] = "ABOVE_FMAX_COLD"
    cls[ks & in_cell & killed] = "IN_CELL_KILLED"
    cls[ks & in_cell & ~killed & (age != "MATURE_2PLUS")] = "IN_CELL_AGE_UNDETERMINED"
    cls[ks & in_cell & ~killed & (age == "MATURE_2PLUS")] = "CANDIDATE"
    ctrl = out["is_control"].fillna(False).to_numpy(bool) if "is_control" in out else np.zeros(n, bool)
    cls = np.where(ctrl, np.char.add("CONTROL:", cls.astype(str)), cls).astype(object)
    out["in_cell"] = in_cell & ks
    out["cradle_class"] = cls
    return out


def not_excluded(row: pd.Series) -> list[str]:
    """Systematics a surviving candidate has NOT yet excluded, by name."""
    out = []
    ut = str(row.get("untested", "") or "")
    for name in ut.split(";"):
        if name:
            out.append(f"untested:{name}")
    if not bool(row.get("irs_spectrum_scored", False)):
        out.append("mantle_only_impact_debris (no IRS spectrum: S53 not run)")
    if bool(row.get("flag_sed_not_single_blackbody", False)):
        out.append("multi-component SED (second ring or companion)")
    if not bool(row.get("neowise_measured", False)):
        out.append("NEOWISE variability not measured")
    out.append("natural late instability (the honest weakness stated in S52)")
    return out


__all__ = ["DEFAULT_VET", "apply_rules", "classify", "not_excluded"]
