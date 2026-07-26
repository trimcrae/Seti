"""The four DERELICT screens plus the normalised R statistic.

Pure, offline-testable dataframe transforms.  Nothing here touches the network.

The screens
-----------
1. :func:`screen_a1_only` -- **the dark-comet complement**.  Significant radial
   ``A1`` with the non-radial ``A2``/``A3`` consistent with zero, no coma, and
   orbit-quality gates.  Seligman et al. selected the *opposite* population
   (large non-radial acceleration), which is precisely the signature that rules
   radiation pressure out.
2. :func:`add_r_statistic` -- ``R = AMR_implied / AMR_natural(D, rho)``.  This is
   the actual discriminant; the A1 cut is only a pre-filter.
3. :func:`screen_negative_a1` -- radiation pressure cannot push sunward, so a
   significant ``A1 < 0`` is a systematic or something very strange.
4. :func:`screen_albedo` -- geometric albedo > 0.7 is impossible for natural
   regolith and trivial for aluminised film.

Honesty rule that shapes the whole module
-----------------------------------------
"A2 was fitted and came out consistent with zero" and "A2 was never fitted" are
**different statements**, and only the first is evidence.  JPL's asteroid
non-grav solutions are overwhelmingly A2-only (Yarkovsky), so the set with all
three components fitted is small.  Every row therefore carries
``nonradial_constrained``: True only when A2 and A3 were actually fitted.  The
funnel reports the strict and permissive counts separately and never quietly
promotes an unconstrained object to "A2 and A3 are zero".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .radiation import COMETARY_MODEL_PARS, amr_from_a1, beta_from_a1, r_statistic

#: State of a fitted non-gravitational component.
NOT_FITTED = "not_fitted"
CONSISTENT_ZERO = "consistent_zero"
MARGINAL = "marginal"
SIGNIFICANT = "significant"


@dataclass
class ScreenParams:
    """Thresholds, mirrored from ``config/derelict.yaml``."""
    a1_snr_min: float = 3.0
    a2_snr_max: float = 1.0
    a3_snr_max: float = 1.0
    condition_code_max: float = 4.0
    data_arc_days_min: float = 30.0
    n_obs_min: int = 30
    require_no_coma: bool = True
    r_flag: float = 10.0
    r_strong: float = 100.0
    r_extreme: float = 1000.0
    neg_a1_snr_min: float = 3.0
    albedo_min: float = 0.70
    albedo_snr_min: float = 2.0
    q_pr: float = 1.0
    rho_natural_kg_m3: float = 1000.0
    albedo_assumed: float = 0.15
    albedo_lo: float = 0.05
    albedo_hi: float = 0.60
    #: Significance at which a non-radial component is called real.
    nonradial_significant_snr: float = 3.0

    @classmethod
    def from_config(cls, d: dict) -> ScreenParams:
        phys = (d or {}).get("physics", {})
        s1 = (d or {}).get("screen_a1_only", {})
        s2 = (d or {}).get("screen_r", {})
        s3 = (d or {}).get("screen_negative_a1", {})
        s4 = (d or {}).get("screen_albedo", {})
        return cls(
            a1_snr_min=float(s1.get("a1_snr_min", 3.0)),
            a2_snr_max=float(s1.get("a2_snr_max", 1.0)),
            a3_snr_max=float(s1.get("a3_snr_max", 1.0)),
            condition_code_max=float(s1.get("condition_code_max", 4.0)),
            data_arc_days_min=float(s1.get("data_arc_days_min", 30.0)),
            n_obs_min=int(s1.get("n_obs_min", 30)),
            require_no_coma=bool(s1.get("require_no_coma", True)),
            r_flag=float(s2.get("r_flag", 10.0)),
            r_strong=float(s2.get("r_strong", 100.0)),
            r_extreme=float(s2.get("r_extreme", 1000.0)),
            neg_a1_snr_min=float(s3.get("a1_snr_min", 3.0)),
            albedo_min=float(s4.get("albedo_min", 0.70)),
            albedo_snr_min=float(s4.get("albedo_snr_min", 2.0)),
            q_pr=float(phys.get("q_pr", 1.0)),
            rho_natural_kg_m3=float(phys.get("rho_natural_kg_m3", 1000.0)),
            albedo_assumed=float(phys.get("albedo_assumed", 0.15)),
            albedo_lo=float(phys.get("albedo_lo", 0.05)),
            albedo_hi=float(phys.get("albedo_hi", 0.60)),
        )


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric column or an all-NaN series of the right length."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _snr(value: pd.Series, sigma: pd.Series) -> pd.Series:
    """Signed significance ``value / sigma``; NaN where either is missing or
    sigma is non-positive."""
    s = sigma.where(sigma > 0)
    return value / s


def component_state(value: pd.Series, sigma: pd.Series, *,
                    zero_snr_max: float, significant_snr: float) -> pd.Series:
    """Classify each non-gravitational component as fitted/zero/marginal/significant."""
    snr = _snr(value, sigma).abs()
    out = pd.Series(NOT_FITTED, index=value.index, dtype=object)
    fitted = value.notna() & sigma.notna() & (sigma > 0)
    out[fitted & (snr < zero_snr_max)] = CONSISTENT_ZERO
    out[fitted & (snr >= zero_snr_max) & (snr < significant_snr)] = MARGINAL
    out[fitted & (snr >= significant_snr)] = SIGNIFICANT
    return out


def annotate(df: pd.DataFrame, p: ScreenParams) -> pd.DataFrame:
    """Add the derived physics + component-state columns every screen needs."""
    out = df.copy()
    a1, a2, a3 = _num(out, "A1"), _num(out, "A2"), _num(out, "A3")
    s1, s2, s3 = _num(out, "sigma_A1"), _num(out, "sigma_A2"), _num(out, "sigma_A3")

    out["a1_snr"] = _snr(a1, s1)
    out["a2_snr"] = _snr(a2, s2)
    out["a3_snr"] = _snr(a3, s3)
    out["a1_state"] = component_state(a1, s1, zero_snr_max=p.a2_snr_max,
                                      significant_snr=p.a1_snr_min)
    out["a2_state"] = component_state(a2, s2, zero_snr_max=p.a2_snr_max,
                                      significant_snr=p.nonradial_significant_snr)
    out["a3_state"] = component_state(a3, s3, zero_snr_max=p.a3_snr_max,
                                      significant_snr=p.nonradial_significant_snr)
    # Only a component that was actually FITTED constrains anything.
    out["nonradial_constrained"] = (
        out["a2_state"].ne(NOT_FITTED) & out["a3_state"].ne(NOT_FITTED))

    # A cometary g(r) invalidates the A1 -> beta conversion entirely.
    if "model_pars" in out.columns:
        out["nongrav_inverse_square"] = out["model_pars"].apply(
            lambda v: not ({str(x).strip().upper() for x in v} & COMETARY_MODEL_PARS)
            if isinstance(v, (list, tuple, set)) else True)
    else:
        out["nongrav_inverse_square"] = True
    # A fitted DT is the cometary delay parameter; it means outgassing.
    dt = _num(out, "DT")
    out.loc[dt.notna() & (dt != 0), "nongrav_inverse_square"] = False

    out["beta_implied"] = np.where(out["nongrav_inverse_square"],
                                   beta_from_a1(a1.to_numpy()), np.nan)
    out["amr_implied_m2_kg"] = np.where(
        out["nongrav_inverse_square"] & (a1 > 0),
        amr_from_a1(a1.to_numpy(), q_pr=p.q_pr), np.nan)
    return out


# --- screen 1: the A1-only complement ----------------------------------------
def screen_a1_only(df: pd.DataFrame, p: ScreenParams,
                   strict_nonradial: bool = False) -> pd.DataFrame:
    """Significant outward A1, non-radial components consistent with zero.

    ``strict_nonradial=True`` additionally demands that A2 and A3 were actually
    *fitted* -- the genuinely constrained subset.  With ``False`` (the default)
    an unfitted A2/A3 is allowed through but ``nonradial_constrained`` records
    that no constraint exists, and the funnel reports both counts.
    """
    d = df
    a1_ok = (d["a1_snr"] >= p.a1_snr_min) & (_num(d, "A1") > 0)
    a2_ok = d["a2_state"].isin([CONSISTENT_ZERO, NOT_FITTED])
    a3_ok = d["a3_state"].isin([CONSISTENT_ZERO, NOT_FITTED])
    if strict_nonradial:
        a2_ok &= d["a2_state"].eq(CONSISTENT_ZERO)
        a3_ok &= d["a3_state"].eq(CONSISTENT_ZERO)

    cc = _num(d, "condition_code")
    arc = _num(d, "data_arc")
    nobs = _num(d, "n_obs_used")
    # Missing quality metadata fails the gate: a short-arc artefact is the
    # single most likely origin of a spurious A1, so absence is not a pass.
    quality_ok = (cc.notna() & (cc <= p.condition_code_max)
                  & arc.notna() & (arc >= p.data_arc_days_min)
                  & nobs.notna() & (nobs >= p.n_obs_min))

    coma_ok = pd.Series(True, index=d.index)
    if p.require_no_coma:
        coma_ok = ~_coma_reported(d)

    law_ok = d["nongrav_inverse_square"].astype(bool)

    res = pd.DataFrame(index=d.index)
    res["s1_a1_significant"] = a1_ok.fillna(False)
    res["s1_a2_zero"] = a2_ok.fillna(False)
    res["s1_a3_zero"] = a3_ok.fillna(False)
    res["s1_quality"] = quality_ok.fillna(False)
    res["s1_no_coma"] = coma_ok.fillna(False)
    res["s1_inverse_square_law"] = law_ok.fillna(False)
    res["screen_a1_only"] = (res["s1_a1_significant"] & res["s1_a2_zero"]
                             & res["s1_a3_zero"] & res["s1_quality"]
                             & res["s1_no_coma"] & res["s1_inverse_square_law"])
    res["screen_a1_only_strict"] = res["screen_a1_only"] & d["nonradial_constrained"]
    return res


def _coma_reported(df: pd.DataFrame) -> pd.Series:
    """True where the object is a comet or a coma/activity is reported."""
    flag = pd.Series(False, index=df.index)
    if "kind" in df.columns:
        flag |= df["kind"].astype(str).str.lower().str.startswith("c")
    if "class" in df.columns:
        cls = df["class"].astype(str).str.upper()
        flag |= cls.isin({"COM", "CTC", "HTC", "JFC", "JFc", "ETC", "PAR", "HYP", "COM*"})
    for col in ("coma", "active", "is_comet"):
        if col in df.columns:
            flag |= df[col].fillna(False).astype(bool)
    if "full_name" in df.columns:
        # Cometary designations carry a P/ C/ D/ I/ prefix or a "P" suffix number.
        flag |= df["full_name"].astype(str).str.match(r"^\s*\d*[PCDXI]/", na=False)
    return flag


# --- screen 2: the normalised outlier statistic -------------------------------
def add_r_statistic(df: pd.DataFrame, p: ScreenParams) -> pd.DataFrame:
    """Attach ``R`` and its supporting quantities to every row."""
    recs = []
    for _, row in df.iterrows():
        a1 = row.get("A1")
        stat = r_statistic(
            float(a1) if pd.notna(a1) else None,
            diameter_m=(float(row["diameter"]) * 1000.0
                        if "diameter" in row.index and pd.notna(row.get("diameter"))
                        else None),
            h_mag=float(row["H"]) if "H" in row.index and pd.notna(row.get("H")) else None,
            albedo=(float(row["albedo"])
                    if "albedo" in row.index and pd.notna(row.get("albedo")) else None),
            albedo_assumed=p.albedo_assumed, albedo_lo=p.albedo_lo, albedo_hi=p.albedo_hi,
            rho_kg_m3=p.rho_natural_kg_m3, q_pr=p.q_pr,
            nongrav_law_is_inverse_square=bool(row.get("nongrav_inverse_square", True)),
        )
        recs.append(stat.to_dict())
    rdf = pd.DataFrame(recs, index=df.index)
    # ``annotate`` already emits a size-independent ``amr_implied_m2_kg`` (it is
    # defined even for rows where R cannot be formed, so it is the more complete
    # column).  Drop the R-statistic's copy so the concat cannot produce a
    # duplicate column name -- which silently turns row lookups into Series.
    rdf = rdf.drop(columns=[c for c in rdf.columns if c in df.columns])
    rdf["screen_r_flag"] = rdf["R_valid"] & (rdf["R"] >= p.r_flag)
    rdf["screen_r_strong"] = rdf["R_valid"] & (rdf["R"] >= p.r_strong)
    rdf["screen_r_extreme"] = rdf["R_valid"] & (rdf["R"] >= p.r_extreme)
    # The conservative reading: even at the albedo that MINIMISES R, is it high?
    rdf["screen_r_flag_conservative"] = rdf["R_valid"] & (rdf["R_lo"] >= p.r_flag)
    return rdf


# --- screen 3: the negative-A1 census -----------------------------------------
def screen_negative_a1(df: pd.DataFrame, p: ScreenParams) -> pd.DataFrame:
    """Significant *sunward* radial acceleration -- physically impossible for SRP.

    Cheap, never published, and a strong internal check: the negative tail
    measures the rate at which the astrometric fits manufacture a spurious A1,
    which is the empirical false-positive floor for screen 1.
    """
    a1 = _num(df, "A1")
    snr = df["a1_snr"] if "a1_snr" in df.columns else _snr(a1, _num(df, "sigma_A1"))
    res = pd.DataFrame(index=df.index)
    res["screen_negative_a1"] = ((a1 < 0) & (snr.abs() >= p.neg_a1_snr_min)).fillna(False)
    return res


# --- screen 4: impossible albedo ----------------------------------------------
def screen_albedo(df: pd.DataFrame, p: ScreenParams) -> pd.DataFrame:
    """Geometric albedo above the natural-regolith ceiling.

    Fresh ice and E-type enstatite reach ~0.6; a bare metallised film is >0.8.
    When an albedo uncertainty is available we require the *excess* to be
    significant, not just the point estimate.
    """
    alb = _num(df, "albedo")
    sig = _num(df, "albedo_sigma") if "albedo_sigma" in df.columns else pd.Series(
        np.nan, index=df.index)
    res = pd.DataFrame(index=df.index)
    point = (alb >= p.albedo_min).fillna(False)
    signif = ((alb - p.albedo_min) / sig >= p.albedo_snr_min).fillna(False)
    res["screen_albedo"] = point & (signif | sig.isna())
    res["albedo_significant"] = signif
    return res


# --- orchestration ------------------------------------------------------------
@dataclass
class ScreenResult:
    table: pd.DataFrame
    funnel: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def run_screens(df: pd.DataFrame, p: ScreenParams) -> ScreenResult:
    """Apply every screen and return the annotated table plus funnel counts.

    An empty or schema-incompatible input degrades to an empty table with a
    ``notes`` entry -- never to a fabricated candidate.
    """
    notes: list[str] = []
    if df is None or len(df) == 0:
        return ScreenResult(pd.DataFrame(), {"input": 0}, ["empty input table"])
    missing = [c for c in ("A1", "sigma_A1") if c not in df.columns]
    if missing:
        notes.append(f"input lacks required columns {missing}; screens degraded")
        if "A1" not in df.columns:
            return ScreenResult(pd.DataFrame(), {"input": len(df)},
                                notes + ["no A1 column: nothing to screen"])

    ann = annotate(df, p)
    s1 = screen_a1_only(ann, p)
    rs = add_r_statistic(ann, p)
    s3 = screen_negative_a1(ann, p)
    s4 = screen_albedo(ann, p)
    out = pd.concat([ann, s1, rs, s3, s4], axis=1)

    funnel = {
        "input": int(len(df)),
        "a1_fitted": int(_num(out, "A1").notna().sum()),
        "a1_positive": int((_num(out, "A1") > 0).sum()),
        "a1_significant": int(out["s1_a1_significant"].sum()),
        "nonradial_constrained": int(out["nonradial_constrained"].sum()),
        "screen1_a1_only": int(out["screen_a1_only"].sum()),
        "screen1_a1_only_strict": int(out["screen_a1_only_strict"].sum()),
        "r_computable": int(out["R_valid"].sum()),
        "screen2_r_flag": int((out["screen_a1_only"] & out["screen_r_flag"]).sum()),
        "screen2_r_strong": int((out["screen_a1_only"] & out["screen_r_strong"]).sum()),
        "screen2_r_extreme": int((out["screen_a1_only"] & out["screen_r_extreme"]).sum()),
        "screen3_negative_a1": int(out["screen_negative_a1"].sum()),
        "screen4_albedo": int(out["screen_albedo"].sum()),
        "cometary_law_excluded": int((~out["nongrav_inverse_square"]).sum()),
    }
    return ScreenResult(out, funnel, notes)
