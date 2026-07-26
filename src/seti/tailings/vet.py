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
    teff_min: float = 4000.0
    """GALAH's own release notes state that cool stars carry systematic trends
    'that can reach values of 0.5 dex for some elements' and that 'dwarf stars
    are most affected at Teff < 4600 K'. The M-dwarf tail is therefore the least
    trustworthy part of any abundance sample and is bounded out by default
    rather than silently included; anything between 4000 and 4600 K that does
    survive is flagged with a cool-star caveat."""
    teff_caveat_below: float = 4600.0
    feh_min: float = -1.0
    """Convective protection depends on METALLICITY as well as Teff and log g.
    A metal-poor turnoff star of ~0.85 Msun can have an envelope below 1e-7 Msun
    -- four orders of magnitude thinner than a solar-metallicity dwarf -- and
    would pass a Teff/log g cut while retaining exactly the thin envelope that
    makes diffusive peculiarity possible (Matrozis et al.). Measured diffusion
    amplitudes run 0.3 dex at [Fe/H] = -2.3 falling to 0.1 dex at -1.1, so a
    floor at -1.0 caps the natural effect an order of magnitude below the
    signal. (Cross-channel note: OSSUARY *selects* the stars this excludes.)"""
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

    feh = _col(out, "fe_h")
    check("metallicity", "fe_h",
          None if feh is None else (feh.astype(float) >= cfg.feh_min),
          f"[Fe/H] < {cfg.feh_min}: metal-poor envelopes are thin enough for "
          "diffusion to act, so the convective-suppression argument does not hold")

    teff = _col(out, teff_col)
    logg = _col(out, logg_col)
    if teff is not None:
        out["cool_star_caveat"] = teff.astype(float) < cfg.teff_caveat_below
    else:
        out["cool_star_caveat"] = False
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

    trusted, note = resolution_verdict(survey)
    out["resolution_trusted"] = trusted
    out["resolution_note"] = note
    out["needs_high_resolution_confirmation"] = not trusted

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


def covariate_window_veto(
    cand: pd.DataFrame,
    *,
    covariate_col: str = "rv",
    element_col: str = "element_max",
    width: float = 20.0,
    min_n: int = 5,
    min_enhancement: float = 2.5,
    max_p: float = 1e-3,
    label: str | None = None,
) -> pd.DataFrame:
    """Veto a *narrow* over-density of one element in an instrumental covariate.

    ``covariate_rate_veto`` bins the covariate into equal-population quantiles.
    That is the right tool for a broad drift and the wrong one for a telluric,
    which is narrow: a ~20 km/s absorption window diluted into a quantile bin
    spanning tens of km/s of ordinary stars is not merely un-triggered, it is
    **un-triggerable**. Measured on the first real run, APOGEE's observed
    ``rv_flag_ratio`` spanned only 0.81-1.32 against a veto threshold of 5.0,
    while the artefact itself was sitting there at 4.03x over an (Fe/H, Teff)
    matched control in -110 to -60 km/s (p = 2.7e-10), 28 of its 29 K anomalies
    being *deficits* -- a telluric eating the 7699 A line. That is Weinberg's
    low-K population, reproduced, and the veto could not see it.

    So this scans fixed-width sliding windows *per element*, comparing each
    element's share of candidates inside the window with every other element's,
    and vetoes the element-window combinations that stand out. Working per
    element matters: the artefact is element-locked because the offending line
    is, and pooling elements averages it away.
    """
    name = label or covariate_col
    out = cand.copy()
    col_out, reason_out = f"pass_{name}_window", f"{name}_window_reason"
    out[col_out] = True
    out[reason_out] = ""
    if covariate_col not in out.columns or element_col not in out.columns or not len(out):
        out[f"{name}_window_untested"] = True
        return out
    v = pd.to_numeric(out[covariate_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(v).any():
        out[f"{name}_window_untested"] = True
        return out
    out[f"{name}_window_untested"] = False

    from math import comb

    els = out[element_col].astype(str).to_numpy()
    lo_edge, hi_edge = np.nanmin(v), np.nanmax(v)
    starts = np.arange(lo_edge, hi_edge, width / 2.0)     # 50% overlap
    for el in pd.unique(els):
        is_el = (els == el)
        n_el = int(is_el.sum())
        if n_el < min_n:
            continue
        for s0 in starts:
            win = np.isfinite(v) & (v >= s0) & (v < s0 + width)
            k = int((win & is_el).sum())
            if k < min_n:
                continue
            n_other = int((~is_el).sum())
            k_other = int((win & ~is_el).sum())
            f_el = k / n_el
            f_other = (k_other / n_other) if n_other else 0.0
            enh = (f_el / f_other) if f_other > 0 else float("inf")
            if enh < min_enhancement:
                continue
            # One-sided binomial tail for k of n_el at the other-element rate.
            p_null = max(f_other, 1e-9)
            p = sum(comb(n_el, i) * p_null**i * (1 - p_null) ** (n_el - i)
                    for i in range(k, n_el + 1))
            if p > max_p:
                continue
            hit = win & is_el
            out.loc[hit, col_out] = False
            out.loc[hit, reason_out] = (
                f"{el} over-dense in {name} [{s0:.0f}, {s0 + width:.0f}]: "
                f"{enh:.1f}x other elements, p={p:.1e} — instrumental "
                f"(telluric / bad-pixel) footprint, not a stellar anomaly"
            )
    return out


def covariate_rate_veto(
    cand: pd.DataFrame,
    all_stars: pd.DataFrame,
    flagged_mask: np.ndarray,
    *,
    covariate_col: str,
    n_bins: int = 12,
    cfg: VetConfig | None = None,
    label: str | None = None,
) -> pd.DataFrame:
    """Veto candidates sitting in an over-flagging bin of an *instrumental* covariate.

    This closes the failure mode that dominates real single-element outliers.
    Weinberg et al. traced two high-Ca APOGEE stars to bad pixels hit by one
    particular radial-velocity + fibre combination, and a whole *population* of
    low-K stars to a heliocentric velocity near -70 km/s that slid the K lines
    onto a telluric band. Their own conclusion is that a rare outlier and a rare
    reduction problem "are not always easy to tell one from the other".

    The discriminator is that an instrument leaves a footprint in instrument
    coordinates. A real anomaly has no reason to correlate with the star's
    radial velocity, its fibre number, or its position on the detector; a
    telluric or bad-pixel artifact does, sharply. Binning the flag rate in each
    such covariate and vetoing the outlying bins is the abundance-space form of
    the ledger rule that a feature recurring across unrelated sightlines is
    instrumental.
    """
    cfg = cfg or VetConfig()
    name = label or covariate_col
    out = cand.copy()
    # A VETO THAT CANNOT RUN MUST NOT REPORT "PASS".  On the first real run the
    # GALAH FITS route carried no rv and no fiber column, so rv_flag_ratio and
    # fiber_flag_ratio were 100% null for all 1,341 candidates while
    # pass_rv_rate and pass_fiber_rate read True -- True by default, not by
    # test.  GALAH's dominant element is K (39.4% of its candidates), the
    # 7699 A resonance doublet, which is precisely the Weinberg telluric failure
    # mode; the one veto that could have explained the pile-up silently reported
    # success.  ``<name>_rate_untested`` now records that, and the summary must
    # treat an untested veto as unknown rather than as a pass.
    if covariate_col not in all_stars.columns or covariate_col not in out.columns:
        out[f"{name}_flag_ratio"] = np.nan
        out[f"pass_{name}_rate"] = True
        out[f"{name}_rate_untested"] = True
        out[f"{name}_rate_untested_reason"] = f"no {covariate_col} column in this source"
        return out

    v_all = pd.to_numeric(all_stars[covariate_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(v_all)
    if finite.sum() < 100:
        out[f"{name}_flag_ratio"] = np.nan
        out[f"pass_{name}_rate"] = True
        out[f"{name}_rate_untested"] = True
        out[f"{name}_rate_untested_reason"] = (
            f"only {int(finite.sum())} finite {covariate_col} values")
        return out
    edges = np.quantile(v_all[finite], np.linspace(0.0, 1.0, int(n_bins) + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        out[f"{name}_flag_ratio"] = np.nan
        out[f"pass_{name}_rate"] = True
        out[f"{name}_rate_untested"] = True
        out[f"{name}_rate_untested_reason"] = f"{covariate_col} has no usable bin edges"
        return out

    idx = np.clip(np.digitize(v_all, edges) - 1, 0, edges.size - 2)
    global_rate = float(np.sum(flagged_mask)) / len(all_stars)
    ratios = np.full(edges.size - 1, np.nan)
    for b in range(edges.size - 1):
        sel = finite & (idx == b)
        if sel.sum() >= 20 and global_rate > 0:
            ratios[b] = (float(np.sum(flagged_mask[sel])) / sel.sum()) / global_rate

    v_c = pd.to_numeric(out[covariate_col], errors="coerce").to_numpy(dtype=float)
    b_c = np.clip(np.digitize(v_c, edges) - 1, 0, edges.size - 2)
    r = np.where(np.isfinite(v_c), ratios[b_c], np.nan)
    out[f"{name}_flag_ratio"] = r
    bad = np.isfinite(r) & (r > cfg.max_field_flag_rate_ratio)
    out[f"pass_{name}_rate"] = ~bad
    reasons = out["vet_reasons"] if "vet_reasons" in out.columns else pd.Series([""] * len(out))
    out["vet_reasons"] = [
        (rs + ("; " if rs else "")
         + f"{name} bin flags {rr:.1f}x the global rate: an instrumental footprint, "
           "not a chemical one")
        if b else rs
        for rs, rr, b in zip(reasons, r, bad, strict=False)
    ]
    out["vet_pass"] = (out["vet_pass"] if "vet_pass" in out.columns else True) & out[f"pass_{name}_rate"]
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
#: Nominal resolving power per survey. A "single-element" anomaly measured
#: below ~R = 20,000 is not a measurement of one element, it is a measurement
#: of a blend -- see :func:`resolution_verdict`.
SURVEY_RESOLUTION: dict[str, int] = {
    "GALAH": 28_000,
    "APOGEE": 22_500,
    "LAMOST_MRS": 7_500,
    "LAMOST": 1_800,
    "RAVE": 7_500,
    "GAIA_RVS": 11_500,
}

#: Below this resolving power a sparse anomaly cannot be believed without
#: high-resolution confirmation. See the Karinkuzhi precedent in
#: :func:`resolution_verdict`.
MIN_TRUSTED_RESOLUTION = 20_000


def resolution_verdict(survey: str, *, resolution: int | None = None) -> tuple[bool, str]:
    """Does this survey's resolving power support a single-element claim?

    There is a published precedent that says no, and it is the sharpest
    contamination result in this whole area. Karinkuzhi et al. re-observed at
    R ~ 86,000 the 15 brightest of 895 s-process candidates that a machine
    -learning pipeline had selected from LAMOST at R ~ 1,800 -- 13 classified
    "Sr-only" and 2 "Ba-only", i.e. exactly the sparse morphology this channel
    hunts. **Every one dissolved.** Four of the thirteen had no s-process
    overabundance at all, eight were mild barium stars, one was a strong barium
    star, and both "Ba-only" stars turned out to be strong (dwarf) barium
    stars -- a dense s-process family in every surviving case. Their conclusion
    is the rule adopted here: "blending effects and saturated lines have to be
    considered very carefully when using machine-learning techniques,
    especially on low-resolution spectra."

    So low-resolution sparsity is presumed to be unresolved blending until a
    high-resolution spectrum says otherwise. Returns ``(trusted, note)``.
    """
    r = resolution if resolution is not None else SURVEY_RESOLUTION.get(survey.upper(), 0)
    if r >= MIN_TRUSTED_RESOLUTION:
        return True, f"R~{r:,}: adequate to separate the line from its neighbours"
    return False, (
        f"R~{r:,} is below the R={MIN_TRUSTED_RESOLUTION:,} floor: apparent "
        "single-element anomalies at low resolution are unresolved blends until "
        "a high-resolution spectrum says otherwise (13/13 'Sr-only' and 2/2 "
        "'Ba-only' LAMOST candidates dissolved into dense barium stars at R~86,000)"
    )


WEAK = "linear"
SATURATED = "saturated"
UNDETECTED = "undetected"


def curve_of_growth_regime(
    central_depth: float,
    *,
    depth_saturated: float = 0.75,
    depth_min: float = 0.02,
) -> tuple[str, str]:
    """Which part of the curve of growth is this line on?

    An abundance is only recoverable from a line on the **linear** part of the
    curve of growth. A saturated core is insensitive to abundance -- its depth
    is set by the source function, not by how much of the element is present --
    so a saturated line can carry an enormous apparent abundance error in
    either direction, and a pipeline that fits it will happily report one. This
    is the second half of the Karinkuzhi warning and it is checked explicitly
    rather than assumed away.
    """
    d = float(central_depth)
    if not np.isfinite(d) or d < depth_min:
        return UNDETECTED, f"central depth {d:.3f} below the {depth_min} detection floor"
    if d >= depth_saturated:
        return SATURATED, (
            f"central depth {d:.2f} >= {depth_saturated}: the line core is saturated and "
            "its strength is no longer a measure of abundance"
        )
    return WEAK, f"central depth {d:.2f}: on the linear part of the curve of growth"


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
    "MIN_TRUSTED_RESOLUTION",
    "SATURATED",
    "SURVEY_RESOLUTION",
    "UNDETECTED",
    "WEAK",
    "VetConfig",
    "curve_of_growth_regime",
    "census_z",
    "covariate_rate_veto",
    "covariate_window_veto",
    "cross_survey_check",
    "dedupe",
    "element_caveat",
    "element_rate_veto",
    "field_rate_veto",
    "measure_ew",
    "remeasure_verdict",
    "resolution_verdict",
    "vet_candidates",
]
