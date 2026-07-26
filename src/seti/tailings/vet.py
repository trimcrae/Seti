"""The contamination funnel: everything that makes a fake sparse anomaly.

A single-element outlier in a survey catalogue is, on the prior, a bad
measurement. The channel is only worth running because the failure modes are
enumerable and each one has a discriminator that does not depend on believing
the catalogue.

Contamination ledger
--------------------
1. **Low SNR.** Abundance precision degrades steeply below SNR ~ 40, and the
   residual distribution grows heavy tails, not just a wider core. Handled
   twice: the empirical scatter is measured *in bins of SNR*, so a low-SNR star
   is divided by a large sigma; and a hard SNR floor removes the regime where
   the pipeline itself is unreliable.
2. **A bad spectral fit.** A poor global fit moves many elements — that is what
   the sparse statistic is for — but a locally bad fit can move one. Cut on the
   pipeline chi-squared and on the pipeline's own flags, both global and
   per-element.
3. **Fast rotation / broadened lines.** Above ~15-20 km/s the lines blend and
   individual abundances become meaningless; the pipeline often does not say
   so. Cut on vbroad / vsini.
4. **Binarity.** An unresolved companion adds a second spectrum with different
   parameters. It produces spurious abundances, and it is the failure mode that
   killed the one followed-up candidate of the nearest prior attenuation
   search. Cut on Gaia RUWE and on radial-velocity scatter.
5. **A systematic in one element.** If an element is flagged far more often
   than the others, that is the line list, not the Galaxy. Reported as a
   per-element flag rate and used as a veto.
6. **A systematic in one observation.** Surveys observe in fields/plates/visits
   with shared calibration. If a star's field flags the same element at an
   elevated rate, the anomaly belongs to the field. This is the abundance-space
   analogue of the ledger rule that a candidate wavelength recurring across
   unrelated sightlines is instrumental.
7. **A known-difficult line for that element.** Element-specific caveats are
   encoded per survey (telluric overlap, hyperfine structure, severe NLTE, one
   weak line only). A candidate resting on such an element is not rejected but
   is *demoted*: it may not be reported as clean without independent
   confirmation.
8. **Duplicate rows.** Repeat observations of the same star inflate candidate
   counts. Deduplicated on the survey ID, keeping the highest-SNR row.
9. **Single-survey anomaly.** The inherited ledger rule "a single-band anomaly
   is an artefact until confirmed in a second band" transfers directly: an
   anomaly seen in one survey's line list, in one wavelength region, with one
   pipeline, is a pipeline statement. Where the star is in both GALAH and
   APOGEE and the element is measured in both, the anomaly must reproduce.
   (Note the converse hazard: Manea et al. 2025 showed APOGEE doppelgangers
   with near-identical H-band abundances differ by up to ~0.4 dex in
   neutron-capture elements at higher resolution — so a *non*-confirmation in
   APOGEE for an n-capture element is weak evidence, and is recorded as
   ``not_covered`` rather than as a refutation.)
10. **The catalogue itself.** The decisive step for any survivor is to go back
    to the raw spectrum and re-measure the line, against Teff-matched peers
    from the same survey and instrument, so that blends and continuum
    structure common to the temperature slice cancel. That statistic is
    :func:`census_z` and it is what a survivor must pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Element caveats: known measurement difficulties, per survey. A candidate that
# rests on one of these is demoted, not deleted -- the caveat is reported with
# the candidate so a reader can weigh it.
# ---------------------------------------------------------------------------
ELEMENT_CAVEATS: dict[tuple[str, str], str] = {
    ("GALAH", "K"): "K I 7699 sits on a telluric O2 band and has an interstellar component",
    ("GALAH", "O"): "O I 7772-7775 triplet has large, Teff-dependent NLTE departures",
    ("GALAH", "Ba"): "Ba II lines are strong, saturated, hyperfine-split and strongly NLTE",
    ("GALAH", "Li"): "Li 6708 is blended with Fe I and varies naturally by orders of magnitude",
    ("GALAH", "Ce"): "few weak Ce II lines; low per-star significance",
    ("GALAH", "Nd"): "weak Nd II lines in a crowded region",
    ("GALAH", "Rb"): "single weak line, heavily blended",
    ("GALAH", "Sm"): "very weak lines; often an upper limit reported as a value",
    ("GALAH", "Zn"): "two lines only, one blended with a Fe I feature",
    ("GALAH", "Cu"): "hyperfine structure; strong NLTE",
    ("APOGEE", "Na"): "two weak H-band Na I lines; large scatter in dwarfs",
    ("APOGEE", "V"): "weak, blended V I lines; unreliable below solar metallicity",
    ("APOGEE", "Cu"): "marginal detection in most spectra",
    ("APOGEE", "Ce"): "single weak Ce II line; the only n-capture element in the H band",
    ("APOGEE", "Nd"): "marginal; frequently an upper limit",
    ("APOGEE", "P"): "very weak lines, strongly blended",
    ("APOGEE", "S"): "few usable lines in cool dwarfs",
    ("APOGEE", "K"): "two weak lines with significant blending",
    ("APOGEE", "C"): "molecular CO/CN features couple C to N, O and the atmosphere model",
    ("APOGEE", "N"): "from CN features; degenerate with C and O",
    ("LAMOST", "*"): "medium-resolution abundances have few elements and large systematics",
}


def element_caveat(survey: str, element: str) -> str | None:
    """Known measurement difficulty for ``element`` in ``survey``, if any."""
    return ELEMENT_CAVEATS.get((survey, element)) or ELEMENT_CAVEATS.get((survey, "*"))


@dataclass(frozen=True)
class VetConfig:
    """Cuts for the contamination funnel. Every default is a ledger rule."""

    snr_min: float = 40.0
    chi2_max: float = 4.0
    ruwe_max: float = 1.4
    rv_scatter_max_kms: float = 1.0
    vbroad_max_kms: float = 15.0
    teff_max: float = 6000.0
    teff_min: float = 3000.0
    logg_min: float = 4.0
    min_good_elements: int = 8
    max_element_flag_rate: float = 0.02
    """An element flagged in more than this fraction of the sample is a
    pipeline systematic, not a population of anomalies."""
    max_field_flag_rate_ratio: float = 5.0
    """A field flagging an element at more than this multiple of the global
    rate is a calibration problem in that field."""
    require_cross_survey: bool = False
    """Off by default because coverage is partial; the flag is always recorded."""
    flag_columns: tuple[str, ...] = field(
        default=("flag_sp", "flag_fe_h", "aspcapflag", "starflag")
    )


def _col(df: pd.DataFrame, name: str) -> np.ndarray | None:
    return df[name].to_numpy() if name in df.columns else None


def vet_candidates(
    cand: pd.DataFrame,
    *,
    cfg: VetConfig | None = None,
    survey: str = "GALAH",
    element_col: str = "element_max",
    snr_col: str = "snr",
    teff_col: str = "teff",
    logg_col: str = "logg",
    chi2_col: str = "chi2",
    ruwe_col: str = "ruwe",
    rv_scatter_col: str = "rv_scatter",
    vbroad_col: str = "vbroad",
    n_elements_col: str = "n_elements",
) -> pd.DataFrame:
    """Apply every catalogue-level cut, recording which ones each star fails.

    Missing columns are treated as *not testable*, never as passing silently:
    the per-check column is set to ``True`` but the check name is appended to
    ``vet_untested`` so the coverage is visible in the report.
    """
    cfg = cfg or VetConfig()
    n = len(cand)
    out = cand.copy()
    reasons: list[list[str]] = [[] for _ in range(n)]
    untested: list[list[str]] = [[] for _ in range(n)]

    def check(name: str, col: str, ok: np.ndarray | None, why: str) -> None:
        if ok is None:
            out[f"pass_{name}"] = True
            for i in range(n):
                untested[i].append(f"{name}(no {col})")
            return
        ok = np.asarray(ok, dtype=bool)
        out[f"pass_{name}"] = ok
        for i in np.flatnonzero(~ok):
            reasons[i].append(why)

    snr = _col(out, snr_col)
    check("snr", snr_col, None if snr is None else (snr.astype(float) >= cfg.snr_min),
          f"SNR < {cfg.snr_min}")

    chi2 = _col(out, chi2_col)
    check("chi2", chi2_col, None if chi2 is None else (chi2.astype(float) <= cfg.chi2_max),
          f"fit chi2 > {cfg.chi2_max}")

    ruwe = _col(out, ruwe_col)
    check("ruwe", ruwe_col, None if ruwe is None else (ruwe.astype(float) <= cfg.ruwe_max),
          f"RUWE > {cfg.ruwe_max} (astrometric binary)")

    rvs = _col(out, rv_scatter_col)
    check("rv_scatter", rv_scatter_col,
          None if rvs is None else (rvs.astype(float) <= cfg.rv_scatter_max_kms),
          f"RV scatter > {cfg.rv_scatter_max_kms} km/s (spectroscopic binary)")

    vb = _col(out, vbroad_col)
    check("vbroad", vbroad_col, None if vb is None else (vb.astype(float) <= cfg.vbroad_max_kms),
          f"line broadening > {cfg.vbroad_max_kms} km/s")

    teff = _col(out, teff_col)
    logg = _col(out, logg_col)
    if teff is None or logg is None:
        check("population", f"{teff_col}/{logg_col}", None, "")
    else:
        ok = ((teff.astype(float) < cfg.teff_max) & (teff.astype(float) > cfg.teff_min)
              & (logg.astype(float) > cfg.logg_min))
        check("population", teff_col, ok,
              f"outside the cool-dwarf box (Teff {cfg.teff_min}-{cfg.teff_max} K, "
              f"logg > {cfg.logg_min}) where diffusion is suppressed")

    nel = _col(out, n_elements_col)
    check("n_elements", n_elements_col,
          None if nel is None else (nel.astype(float) >= cfg.min_good_elements),
          f"fewer than {cfg.min_good_elements} measured elements")

    # Pipeline flags: any present flag column must be zero / empty.
    flag_ok = np.ones(n, dtype=bool)
    seen_flag = False
    for fc in cfg.flag_columns:
        if fc not in out.columns:
            continue
        seen_flag = True
        v = out[fc]
        if v.dtype == object:
            flag_ok &= v.fillna("").astype(str).str.strip().isin(["", "0", "0.0", "nan"]).to_numpy()
        else:
            flag_ok &= (v.fillna(0).to_numpy().astype(float) == 0)
    check("pipeline_flags", "flags", flag_ok if seen_flag else None, "pipeline quality flag set")

    # Per-element caveat (demotion, not rejection).
    if element_col in out.columns:
        cav = [element_caveat(survey, str(e)) for e in out[element_col]]
        out["element_caveat"] = cav
        out["caveated_element"] = [c is not None for c in cav]
    else:
        out["element_caveat"] = None
        out["caveated_element"] = False

    pass_cols = [c for c in out.columns if c.startswith("pass_")]
    out["vet_pass"] = out[pass_cols].all(axis=1) if pass_cols else True
    out["vet_reasons"] = ["; ".join(r) for r in reasons]
    out["vet_untested"] = ["; ".join(u) for u in untested]
    return out


def element_rate_veto(
    cand: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    cfg: VetConfig | None = None,
    element_col: str = "element_max",
) -> pd.DataFrame:
    """Veto candidates carried by an element with a runaway global flag rate."""
    cfg = cfg or VetConfig()
    lut = dict(zip(rates["element"], rates["flag_rate"], strict=False))
    out = cand.copy()
    if element_col not in out.columns:
        out["element_flag_rate"] = np.nan
        out["pass_element_rate"] = True
        return out
    r = np.array([float(lut.get(str(e), np.nan)) for e in out[element_col]])
    out["element_flag_rate"] = r
    bad = np.isfinite(r) & (r > cfg.max_element_flag_rate)
    out["pass_element_rate"] = ~bad
    out["vet_reasons"] = [
        (rs + ("; " if rs else "") + f"element flagged in {rr:.1%} of the sample "
         f"(> {cfg.max_element_flag_rate:.1%}): pipeline systematic")
        if b else rs
        for rs, rr, b in zip(out.get("vet_reasons", [""] * len(out)), r, bad, strict=False)
    ]
    out["vet_pass"] = out.get("vet_pass", True) & out["pass_element_rate"]
    return out


def field_rate_veto(
    cand: pd.DataFrame,
    all_stars: pd.DataFrame,
    flagged_mask: np.ndarray,
    *,
    field_col: str = "field_id",
    element_col: str = "element_max",
    cfg: VetConfig | None = None,
) -> pd.DataFrame:
    """Veto candidates from fields that flag their element at an elevated rate.

    Shared calibration within an observing field is the abundance-space version
    of "a wavelength recurring across unrelated sightlines is instrumental".
    """
    cfg = cfg or VetConfig()
    out = cand.copy()
    if field_col not in all_stars.columns or field_col not in out.columns:
        out["field_flag_ratio"] = np.nan
        out["pass_field_rate"] = True
        return out

    n_total = len(all_stars)
    global_rate = float(np.sum(flagged_mask)) / n_total if n_total else np.nan
    fields = all_stars[field_col].to_numpy()
    ratios = []
    for f in out[field_col].to_numpy():
        sel = fields == f
        n_f = int(sel.sum())
        if n_f < 20 or not np.isfinite(global_rate) or global_rate <= 0:
            ratios.append(np.nan)
            continue
        ratios.append((float(np.sum(flagged_mask[sel])) / n_f) / global_rate)
    r = np.array(ratios, dtype=float)
    out["field_flag_ratio"] = r
    bad = np.isfinite(r) & (r > cfg.max_field_flag_rate_ratio)
    out["pass_field_rate"] = ~bad
    out["vet_pass"] = out.get("vet_pass", True) & out["pass_field_rate"]
    return out


def cross_survey_check(
    cand: pd.DataFrame,
    other: pd.DataFrame,
    *,
    id_col: str = "xmatch_id",
    element_col: str = "element_max",
    z_prefix: str = "z_",
    z_confirm: float = 3.0,
) -> pd.DataFrame:
    """Does the anomaly reproduce in an independent survey's abundance vector?

    Emits ``cross_survey`` in {``confirmed``, ``refuted``, ``not_covered``,
    ``no_match``}. Absence of coverage is never scored as a refutation.
    """
    out = cand.copy()
    if other is None or len(other) == 0 or id_col not in out.columns:
        out["cross_survey"] = "no_match"
        out["cross_survey_z"] = np.nan
        return out

    idx = other.set_index(id_col) if id_col in other.columns else None
    verdicts, zs = [], []
    for _, r in out.iterrows():
        key = r.get(id_col)
        el = str(r.get(element_col))
        if idx is None or key is None or key not in idx.index:
            verdicts.append("no_match")
            zs.append(np.nan)
            continue
        row = idx.loc[key]
        row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
        zcol = f"{z_prefix}{el}"
        if zcol not in idx.columns or not np.isfinite(float(row.get(zcol, np.nan))):
            verdicts.append("not_covered")
            zs.append(np.nan)
            continue
        zv = float(row[zcol])
        zs.append(zv)
        same_sign = np.sign(zv) == np.sign(float(r.get("z_max_signed", zv)))
        verdicts.append("confirmed" if (abs(zv) >= z_confirm and same_sign) else "refuted")
    out["cross_survey"] = verdicts
    out["cross_survey_z"] = zs
    return out


def dedupe(df: pd.DataFrame, *, id_col: str, snr_col: str = "snr") -> pd.DataFrame:
    """Keep one row per star: the highest-SNR observation."""
    if id_col not in df.columns:
        return df
    if snr_col in df.columns:
        return (df.sort_values(snr_col, ascending=False)
                  .drop_duplicates(subset=[id_col], keep="first")
                  .reset_index(drop=True))
    return df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)


# ---------------------------------------------------------------------------
# The decisive step: re-measure the line from the raw spectrum.
# ---------------------------------------------------------------------------
def measure_ew(
    wave: np.ndarray,
    flux: np.ndarray,
    center: float,
    *,
    half_width: float = 0.35,
    cont_half: float = 3.0,
    cont_deg: int = 2,
) -> tuple[float, float]:
    """Pseudo-equivalent width of a line, with a robust local continuum.

    The continuum is a low-order polynomial over +/- ``cont_half`` angstroms
    with the line core excluded and with asymmetric clipping, so absorption
    elsewhere in the window cannot drag the fit down and manufacture a line.
    Returns ``(ew_angstrom, ew_error)``.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    win = (wave > center - cont_half) & (wave < center + cont_half) & np.isfinite(flux)
    if win.sum() < 10:
        return float("nan"), float("nan")
    core = np.abs(wave - center) < half_width
    cont_sel = win & ~core
    if cont_sel.sum() < 6:
        return float("nan"), float("nan")

    x = wave[cont_sel] - center
    y = flux[cont_sel]
    keep = np.ones(len(y), dtype=bool)
    coef = np.polyfit(x, y, cont_deg)
    for _ in range(3):
        model = np.polyval(coef, x)
        resid = y - model
        s = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if not np.isfinite(s) or s <= 0:
            break
        # Asymmetric: reject absorption hard, emission gently.
        keep = (resid > -2.0 * s) & (resid < 3.0 * s)
        if keep.sum() < max(6, cont_deg + 2):
            break
        coef = np.polyfit(x[keep], y[keep], cont_deg)

    cont_core = np.polyval(coef, wave[core] - center)
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = 1.0 - flux[core] / cont_core
    dlam = float(np.median(np.diff(wave[core]))) if core.sum() > 1 else float("nan")
    ew = float(np.nansum(depth) * dlam)
    scatter = 1.4826 * np.median(np.abs(y - np.polyval(coef, x)))
    err = float(scatter * dlam * np.sqrt(max(core.sum(), 1)))
    return ew, err


def census_z(value: float, peers: np.ndarray) -> float:
    """Rank a measurement against Teff-matched peers measured identically.

    Self-calibrating: blends, unresolved line forests, telluric residuals and
    blaze structure common to the temperature slice cancel exactly, because the
    comparison is at the same wavelength in the same instrument. No absolute
    spectral synthesis is required, which is what makes the re-measurement
    independent of the pipeline being tested.
    """
    p = np.asarray(peers, dtype=float)
    p = p[np.isfinite(p)]
    if p.size < 10 or not np.isfinite(value):
        return float("nan")
    med = np.median(p)
    mad = 1.4826 * np.median(np.abs(p - med))
    if not np.isfinite(mad) or mad <= 0:
        return float("nan")
    return float((value - med) / mad)


def remeasure_verdict(
    z_catalog: float,
    z_remeasured: float,
    *,
    min_z: float = 4.0,
    require_same_sign: bool = True,
) -> str:
    """Compare the catalogue anomaly with the independent re-measurement."""
    if not np.isfinite(z_remeasured):
        return "NO_SPECTRUM"
    if require_same_sign and np.isfinite(z_catalog) and np.sign(z_remeasured) != np.sign(z_catalog):
        return "REFUTED_SIGN"
    if abs(z_remeasured) < min_z:
        return "REFUTED_AMPLITUDE"
    return "CONFIRMED"


__all__ = [
    "ELEMENT_CAVEATS",
    "VetConfig",
    "census_z",
    "cross_survey_check",
    "dedupe",
    "element_caveat",
    "element_rate_veto",
    "field_rate_veto",
    "measure_ew",
    "remeasure_verdict",
    "vet_candidates",
]
