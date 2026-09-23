"""The cessation population: catalogued PULSATORS, all sky, bright enough for DASCH.

Why a separate population (docs/century.md §6.3, §7).  The first target
selection (run 35748748365) drew its catalogued variables from six 1-degree
fields and got 24, dominated by eclipsing binaries (EA/EB/EW/E/ELL) plus RS and
ROT stars, with a single Cepheid.  An eclipse period is a geometric orbit: it
cannot stop without the system being destroyed, so "an eclipse that ceased" is a
statement about the photometry, never about the star.  The cessation question is
a question about oscillators that CAN stop --- pulsators --- and there are not
enough bright pulsators in six square degrees to ask it of.  DASCH DR7 covers
the whole sky, so this population is drawn from the whole of AAVSO VSX (GCVS as
the fallback), by type, not by field.

Every choice below is a pure function so it can be tested offline:

* :func:`pulsator_class` --- the class of a VSX/GCVS type string, or ``""``.
  Uncertain types (a ``:`` anywhere in the leading token) are refused: a star
  whose pulsation is itself uncertain cannot carry a claim that it stopped.
  Eclipsing, rotational and irregular types are never pulsators here.
* :func:`vsx_amplitude` --- the catalogued amplitude, NaN unless it is a real
  number in ONE passband.  The old selection let a NaN amplitude through the
  ``amp >= amp_min`` cut (every ``KID``/``KIC`` VSX row had one); a missing
  amplitude is now a rejection, not a pass.
* :func:`b_mean_estimate` --- an estimate of the MEAN B magnitude, the band the
  Harvard plates (blue-sensitive emulsions) and the DASCH APASS-B calibration
  work in.  VSX magnitudes are in whatever band the discoverer used; a V
  magnitude is moved to B with the class's typical B-V.  This is a SELECTION
  proxy only --- DASCH measures the star's own magnitude on every plate.
* :func:`filter_pulsators` --- the cuts, per class, with the reason every
  rejected row was rejected counted in a funnel.
* :func:`vsx_adql` --- the server-side prefilter (brightness, period range,
  type prefixes).  The local filter is the source of truth; the ADQL only keeps
  the transfer small.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Leading-token VSX/GCVS type -> pulsator class.  The token is the part of the
# type string before any "|", "+" or "/" (VSX writes multiple types that way).
# "(B)" suffixes (VSX's "in a binary" marker, e.g. "DCEP(B)", "RR(B)") are kept
# in the class, since a pulsator in a binary still pulsates.
_CLASS_TOKENS: dict[str, tuple[str, ...]] = {
    # Classical and type-II Cepheids, anomalous Cepheids, BL Her / W Vir.
    "cepheid": ("DCEP", "DCEPS", "DCEP(B)", "DCEPS(B)", "CEP", "CEP(B)", "ACEP", "ACEP(B)",
                "CW", "CWA", "CWB", "BLHER", "WVIR", "T2CEP", "BLBOO"),
    # RV Tauri: the catalogued period is the double-wave (formal) period.
    "rv_tauri": ("RV", "RVA", "RVB"),
    # RR Lyrae of every mode.
    "rr_lyrae": ("RR", "RRAB", "RRC", "RRD", "ARRD", "RR(B)", "RRAB(B)", "RRC(B)"),
    # High-amplitude delta Scuti / SX Phe (the amp_min cut keeps only the HADS
    # end of DSCT).
    "delta_scuti": ("DSCT", "DSCTC", "DSCT(B)", "HADS", "HADS(B)", "SXPHE", "SXPHE(B)"),
    # Miras: regular, large-amplitude long-period pulsators.
    "mira": ("M", "M(B)"),
    # Semi-regulars of the REGULAR kind only.  SRB/SRC/SRD/SRS and L are
    # excluded: their periodicity is poorly defined or multiple, so a period
    # that "stopped" in them is indistinguishable from their normal behaviour.
    "semiregular": ("SRA",),
}
TOKEN_TO_CLASS: dict[str, str] = {tok: cls for cls, toks in _CLASS_TOKENS.items() for tok in toks}
PULSATOR_CLASSES: tuple[str, ...] = tuple(_CLASS_TOKENS)
LPV_CLASSES: frozenset[str] = frozenset({"mira", "semiregular"})

# Typical B-V by class, for the selection proxy only (Cepheids ~0.4-1.1,
# RR Lyrae ~0.2-0.5, HADS ~0.2-0.4, RV Tau ~0.6-1.2, Miras/SRa ~1.3-1.8 at
# the bright phases that dominate the plates).
TYPICAL_B_MINUS_V: dict[str, float] = {"cepheid": 0.7, "rv_tauri": 0.9, "rr_lyrae": 0.35,
                                       "delta_scuti": 0.3, "mira": 1.5, "semiregular": 1.5}
# Passbands that are B-like (used as is), V-like (moved by B-V) or neither
# (refused: an R/I/NIR magnitude says little about a red star's blue brightness).
B_LIKE_BANDS: frozenset[str] = frozenset({"B", "P", "PG", "BT", "BJ"})
V_LIKE_BANDS: frozenset[str] = frozenset({"V", "VJ", "VT", "HP", "G", "CV", "TG", "G'", "GAIA"})

# Server-side type prefixes (ADQL LIKE).  Deliberately loose; the local filter
# decides.  "M%" also returns MISC, which pulsator_class refuses.
ADQL_TYPE_PREFIXES: tuple[str, ...] = ("RR", "DCEP", "CEP", "ACEP", "CW", "BLHER", "WVIR",
                                       "T2CEP", "BLBOO", "RV", "ARRD", "DSCT", "HADS",
                                       "SXPHE", "M", "SRA")

DEFAULT_CLASS_LIMITS: dict[str, dict] = {
    # period_min / period_max in days.  The lower limit of delta_scuti is set by
    # the exposure smear: a typical 60-minute patrol exposure keeps
    # |sinc(t/P)| >= 0.5 only for P >~ 0.07 d.
    "cepheid": {"period_min": 0.3, "period_max": 100.0, "max_n": 700},
    "rv_tauri": {"period_min": 20.0, "period_max": 200.0, "max_n": 120},
    "rr_lyrae": {"period_min": 0.2, "period_max": 1.2, "max_n": 900},
    "delta_scuti": {"period_min": 0.07, "period_max": 0.3, "max_n": 250},
    # The upper Mira/SRa limit keeps >= ~1.6 cycles in a 2-year cessation block.
    "mira": {"period_min": 80.0, "period_max": 450.0, "max_n": 500},
    "semiregular": {"period_min": 30.0, "period_max": 450.0, "max_n": 150},
}


def leading_token(vtype) -> str:
    """The first type in a VSX/GCVS type string, upper-cased and stripped."""
    s = "" if vtype is None or (isinstance(vtype, float) and np.isnan(vtype)) else str(vtype)
    return re.split(r"[|+/]", s.strip(), maxsplit=1)[0].strip().upper()


def pulsator_class(vtype) -> str:
    """The pulsator class of a catalogue type, or ``""`` when it is not one.

    Refuses uncertain classifications (``RRAB:``), eclipsers (``EA``),
    rotators, irregulars and the poorly periodic semi-regulars.
    """
    tok = leading_token(vtype)
    if not tok or ":" in tok:
        return ""
    return TOKEN_TO_CLASS.get(tok, "")


def _num(df: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce").to_numpy(dtype=float, copy=True)
    return np.full(len(df), np.nan)


def _str(df: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    for n in names:
        if n in df.columns:
            return df[n].fillna("").astype(str).str.strip().to_numpy()
    return np.full(len(df), "", dtype=object)


def vsx_amplitude(df: pd.DataFrame) -> np.ndarray:
    """Catalogued amplitude in one passband, NaN when it is not established.

    VSX ``min`` is itself an amplitude when ``f_min`` carries ``(``; otherwise
    the amplitude is ``min - max``, which is only meaningful when both are in
    the same band (``n_min`` empty or equal to ``n_max``) and ``min`` is not a
    fainter-than limit (``l_min == '<'``).
    """
    mx = _num(df, ("max", "mag_max", "magmax"))
    mn = _num(df, ("min", "mag_min", "min1", "magmin"))
    f_min = _str(df, ("f_min", "f_min1"))
    n_max = np.char.upper(_str(df, ("n_max", "n_magmax")).astype(str))
    n_min = np.char.upper(_str(df, ("n_min", "n_min1")).astype(str))
    l_min = _str(df, ("l_min", "l_min1")).astype(str)
    is_amp = np.array(["(" in s for s in f_min], dtype=bool)
    same_band = (n_min == "") | (n_min == n_max)
    limit = np.array([s.strip() in ("<", ">") for s in l_min], dtype=bool)
    amp = np.where(is_amp, mn, np.where(same_band & ~limit, mn - mx, np.nan))
    amp = np.where(np.isfinite(amp) & (amp > 0), amp, np.nan)
    return amp


def b_mean_estimate(mag_max: np.ndarray, amp: np.ndarray, band: np.ndarray,
                    cls: np.ndarray) -> np.ndarray:
    """Estimated mean B magnitude; NaN when the band cannot be moved to B."""
    mag_max = np.asarray(mag_max, dtype=float)
    amp = np.asarray(amp, dtype=float)
    band_u = np.char.upper(np.asarray(band, dtype=str))
    mean = mag_max + 0.5 * amp
    bv = np.array([TYPICAL_B_MINUS_V.get(str(c), np.nan) for c in cls], dtype=float)
    out = np.full(mean.shape, np.nan)
    is_b = np.isin(band_u, list(B_LIKE_BANDS))
    is_v = np.isin(band_u, list(V_LIKE_BANDS)) | (band_u == "")
    out[is_b] = mean[is_b]
    out[is_v] = mean[is_v] + bv[is_v]
    return out


def normalise_catalogue(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """VSX or GCVS rows -> ``name, ra, dec, vtype, period_cat, mag_max, band,
    amp_cat, var_flag, source``."""
    if df is None or not len(df):
        return pd.DataFrame(columns=["name", "ra", "dec", "vtype", "period_cat", "mag_max",
                                     "band", "amp_cat", "var_flag", "source"])
    d = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    out = pd.DataFrame({
        "name": _str(d, ("name", "gcvs", "designation")),
        "ra": _num(d, ("raj2000", "ra", "_raj2000", "ra_icrs")),
        "dec": _num(d, ("dej2000", "dec", "_dej2000", "de_icrs")),
        "vtype": _str(d, ("type", "vartype", "vtype")),
        "period_cat": _num(d, ("period",)),
        "mag_max": _num(d, ("max", "mag_max", "magmax")),
        "band": _str(d, ("n_max", "n_magmax")),
        "amp_cat": vsx_amplitude(d),
        # VSX "V": 0 variable, 1 suspected, 2 constant/non-existing; GCVS has none.
        "var_flag": _num(d, ("v", "var_flag")) if "v" in d.columns or "var_flag" in d.columns
        else np.zeros(len(d)),
        "source": source,
    })
    return out


def filter_pulsators(cat: pd.DataFrame, *, b_mean_max: float = 14.0,
                     b_mean_min: float = 8.0, amp_min: float = 0.3,
                     class_limits: dict | None = None,
                     max_total: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Apply the cessation-population cuts.  ``(selected, funnel)``.

    Order of the funnel is the order of the cuts; every row lands in exactly
    one bucket.  Within a class the brightest stars are kept first (plate
    depth is the binding constraint on the censored efficiency, §6.3).  When
    ``max_total`` binds, classes are drawn round-robin so no one class (the
    numerous Miras, say) crowds the others out.
    """
    limits = {k: dict(v) for k, v in DEFAULT_CLASS_LIMITS.items()}
    for k, v in (class_limits or {}).items():
        limits.setdefault(k, {}).update(v or {})
    funnel: dict = {"n_catalogue_rows": int(len(cat))}
    if not len(cat):
        funnel["n_selected"] = 0
        return cat.iloc[0:0].assign(pulsator_class=[], mag_cat=[]), funnel
    c = cat.copy()
    c["pulsator_class"] = [pulsator_class(v) for v in c["vtype"]]
    c["mag_cat"] = b_mean_estimate(c["mag_max"].to_numpy(float), c["amp_cat"].to_numpy(float),
                                   c["band"].to_numpy(str), c["pulsator_class"].to_numpy(str))
    reason = np.full(len(c), "", dtype=object)

    def cut(mask: np.ndarray, why: str) -> None:
        sel = (reason == "") & mask
        reason[sel] = why

    vf = c["var_flag"].to_numpy(float)
    cut(np.isfinite(vf) & (vf != 0), "not_a_confirmed_variable")
    cut(c["pulsator_class"].to_numpy(str) == "", "not_a_pulsator_type")
    cut(~(np.isfinite(c["ra"].to_numpy(float)) & np.isfinite(c["dec"].to_numpy(float))),
        "no_position")
    per = c["period_cat"].to_numpy(float)
    pmin = np.array([limits.get(k, {}).get("period_min", np.nan)
                     for k in c["pulsator_class"]], dtype=float)
    pmax = np.array([limits.get(k, {}).get("period_max", np.nan)
                     for k in c["pulsator_class"]], dtype=float)
    cut(~np.isfinite(per), "no_period")
    cut(~((per >= pmin) & (per <= pmax)), "period_outside_class_range")
    amp = c["amp_cat"].to_numpy(float)
    cut(~np.isfinite(amp), "amplitude_not_established")
    cut(amp < float(amp_min), "amplitude_below_plate_floor")
    bm = c["mag_cat"].to_numpy(float)
    cut(~np.isfinite(bm), "band_not_convertible_to_B")
    cut(bm > float(b_mean_max), "too_faint_for_plates")
    cut(bm < float(b_mean_min), "too_bright_saturates")
    c["reject_reason"] = reason
    for why, n in pd.Series(reason[reason != ""]).value_counts().items():
        funnel[f"rejected_{why}"] = int(n)
    keep = c[reason == ""].copy()
    # A star catalogued twice (VSX carries duplicates under different names)
    # must not be measured twice: collapse within 5 arcsec, brightest first.
    keep = keep.sort_values("mag_cat", kind="mergesort").reset_index(drop=True)
    if len(keep) > 1:
        ra = np.radians(keep["ra"].to_numpy(float))
        de = np.radians(keep["dec"].to_numpy(float))
        xyz = np.column_stack([np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)])
        dup = np.zeros(len(keep), dtype=bool)
        tol = np.radians(5.0 / 3600.0)
        order = np.argsort(keep["dec"].to_numpy(float), kind="mergesort")
        decs = keep["dec"].to_numpy(float)[order]
        for ii, i in enumerate(order):
            if dup[i]:
                continue
            j = ii + 1
            while j < len(order) and decs[j] - decs[ii] <= 5.0 / 3600.0:
                k = order[j]
                if not dup[k] and float(np.dot(xyz[i], xyz[k])) >= np.cos(tol):
                    # keep the brighter one (lower index after the sort)
                    dup[max(i, k)] = True
                j += 1
        funnel["rejected_duplicate_position"] = int(dup.sum())
        keep = keep[~dup].reset_index(drop=True)
    # Per-class caps, brightest first.
    capped = []
    for cls in PULSATOR_CLASSES:
        sub = keep[keep["pulsator_class"] == cls]
        cap = limits.get(cls, {}).get("max_n")
        n_before = len(sub)
        if cap is not None and int(cap) >= 0:
            sub = sub.head(int(cap))
        funnel[f"n_eligible_{cls}"] = int(n_before)
        funnel[f"n_after_class_cap_{cls}"] = int(len(sub))
        capped.append(sub)
    keep = pd.concat(capped, ignore_index=True) if capped else keep.iloc[0:0]
    if max_total is not None and int(max_total) >= 0 and len(keep) > int(max_total):
        keep = keep.assign(_rank=keep.groupby("pulsator_class").cumcount())
        keep = keep.sort_values(["_rank", "mag_cat"], kind="mergesort").head(int(max_total))
        keep = keep.drop(columns="_rank")
    keep = keep.sort_values(["pulsator_class", "mag_cat"], kind="mergesort").reset_index(drop=True)
    funnel["n_selected"] = int(len(keep))
    funnel["n_selected_by_class"] = {k: int(v) for k, v in
                                     keep["pulsator_class"].value_counts().items()}
    return keep, funnel


def vsx_adql(*, max_mag_max: float, period_min: float, period_max: float,
             table: str = "B/vsx/vsx", prefixes: tuple[str, ...] = ADQL_TYPE_PREFIXES) -> str:
    """The TAPVizieR prefilter.  ``max`` and ``min`` are ADQL reserved words
    and are quoted; so is the table name, which contains slashes."""
    like = " OR ".join(f"\"Type\" LIKE '{p}%'" for p in prefixes)
    return (f'SELECT "Name", "V", "RAJ2000", "DEJ2000", "Type", "l_max", "max", "n_max", '
            f'"f_min", "l_min", "min", "n_min", "Period" FROM "{table}" '
            f'WHERE "max" <= {float(max_mag_max):.3f} '
            f'AND "Period" >= {float(period_min):.5f} AND "Period" <= {float(period_max):.3f} '
            f"AND ({like})")


__all__ = ["ADQL_TYPE_PREFIXES", "B_LIKE_BANDS", "DEFAULT_CLASS_LIMITS", "LPV_CLASSES",
           "PULSATOR_CLASSES", "TOKEN_TO_CLASS", "TYPICAL_B_MINUS_V", "V_LIKE_BANDS",
           "b_mean_estimate", "filter_pulsators", "leading_token", "normalise_catalogue",
           "pulsator_class", "vsx_adql", "vsx_amplitude"]
