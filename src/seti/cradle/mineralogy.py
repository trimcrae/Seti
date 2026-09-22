"""S53, slag not glass: the mineralogy score on a Spitzer/IRS low-resolution spectrum.

Hypervelocity collision debris (HD 172555, HD 23514, HD 15407A) shows silica
glass WITH crystalline forsterite/enstatite, FeS and SiO gas; smelter slag and
structural ceramics are CaO-Al2O3-SiO2 glass with the Fe, Mg and S phases
stripped.  Fe-depletion alone is not clean (mantle-only impact debris is
Fe-poor), so the target is a *featureless-to-silica-only* 8-13 um band: no
10 um amorphous olivine/pyroxene peak, no 11.3 um forsterite, no 23 um FeS.

What is measured (``score_spectrum``)
------------------------------------
A local linear continuum is fitted across anchor windows either side of each
feature and the feature contrast ``F / F_cont - 1`` is integrated:

* **10 um silicate**: anchors 7.4-7.9 and 12.7-13.4 um, feature 9.0-11.8 um;
  the peak wavelength and the 11.3 / 9.8 um ratio (crystalline forsterite),
  the 8.9-9.3 um contrast (silica);
* **18 um silicate**: anchors 15.0-16.0 and 21.0-22.5 um, feature 16.5-20 um;
* **23 um FeS**: anchors 21.0-22.5 and 25.5-27 um, feature 23-24.5 um.

Verdicts: ``SILICATE_FEATURED`` (10 um contrast > 0.10 at > 3 sigma),
``FEATURELESS`` (|contrast| < 0.05 with sigma < 0.05), ``AMBIGUOUS``, or
``NO_COVERAGE``.  A featureless band on a star with a strong W3/W4 excess is
``SLAG_CONSISTENT`` --- with the two caveats the doc must state: grains larger
than ~ 5 um suppress the features too, and no laboratory optical constants for
Ca-Al slag glass exist to fit against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MINERALOGY: dict = {
    "feat10": {"anchors": [[7.4, 7.9], [12.7, 13.4]], "band": [9.0, 11.8],
               "silica": [8.9, 9.3], "forsterite": [11.1, 11.5], "olivine": [9.6, 10.0]},
    "feat18": {"anchors": [[15.0, 16.0], [21.0, 22.5]], "band": [16.5, 20.0]},
    "feat23": {"anchors": [[21.0, 22.5], [25.5, 27.0]], "band": [23.0, 24.5]},
    "featured_min_contrast": 0.10,
    "featured_min_sigma": 3.0,
    "featureless_max_contrast": 0.05,
    "featureless_max_err": 0.05,
    "min_points_per_window": 3,
}


def _window(w: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (w >= lo) & (w <= hi)


def _feature(w: np.ndarray, f: np.ndarray, e: np.ndarray, spec: dict, c: dict) -> dict:
    """Contrast of one feature against a linear continuum through its anchors."""
    (a1, a2), (b1, b2) = spec["anchors"]
    lo, hi = spec["band"]
    m1, m2 = _window(w, a1, a2), _window(w, b1, b2)
    mb = _window(w, lo, hi)
    k = int(c["min_points_per_window"])
    out = {"covered": bool(m1.sum() >= k and m2.sum() >= k and mb.sum() >= k)}
    if not out["covered"]:
        out.update(contrast=np.nan, contrast_err=np.nan, peak_um=np.nan)
        return out
    x = np.concatenate([w[m1], w[m2]])
    y = np.concatenate([f[m1], f[m2]])
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    cont = coef[0] + coef[1] * w
    resid_anchor = y - (coef[0] + coef[1] * x)
    scatter = float(np.std(resid_anchor)) if len(x) > 2 else float(np.nanmedian(e[m1 | m2]))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = f / cont - 1.0
        err_pt = np.where(np.isfinite(e) & (e > 0), e, scatter) / np.abs(cont)
    contrast = float(np.nanmean(ratio[mb]))
    nb = int(np.isfinite(ratio[mb]).sum())
    # error: per-point noise averaged down, plus the continuum's own scatter
    err = float(np.sqrt(np.nanmean(err_pt[mb] ** 2) / max(nb, 1) + (scatter / np.nanmean(np.abs(cont[mb]))) ** 2))
    j = int(np.nanargmax(ratio[mb])) if nb else 0
    out.update(contrast=contrast, contrast_err=err, peak_um=float(w[mb][j]) if nb else np.nan,
               n_points=nb, continuum_slope=float(coef[1]))
    for name in ("silica", "forsterite", "olivine"):
        if name in spec:
            s1, s2 = spec[name]
            ms = _window(w, s1, s2)
            out[f"{name}_contrast"] = float(np.nanmean(ratio[ms])) if ms.any() else np.nan
    return out


def score_spectrum(wave_um, flux_jy, err_jy=None, cfg: dict | None = None) -> dict:
    """The feature-presence score of one spectrum (see the module docstring)."""
    c = {**DEFAULT_MINERALOGY, **(cfg or {})}
    w = np.asarray(wave_um, float)
    f = np.asarray(flux_jy, float)
    e = np.asarray(err_jy, float) if err_jy is not None else np.full_like(f, np.nan)
    ok = np.isfinite(w) & np.isfinite(f) & (w > 0)
    w, f, e = w[ok], f[ok], e[ok]
    order = np.argsort(w)
    w, f, e = w[order], f[order], e[order]
    rep: dict = {"n_points": int(len(w)),
                 "wave_min_um": float(w.min()) if len(w) else np.nan,
                 "wave_max_um": float(w.max()) if len(w) else np.nan}
    if len(w) < 10:
        rep.update(verdict="NO_COVERAGE", feat10={"covered": False})
        return rep
    f10 = _feature(w, f, e, c["feat10"], c)
    f18 = _feature(w, f, e, c["feat18"], c)
    f23 = _feature(w, f, e, c["feat23"], c)
    rep.update(feat10=f10, feat18=f18, feat23=f23)
    if not f10["covered"]:
        rep["verdict"] = "NO_COVERAGE"
        return rep
    con, err = f10["contrast"], f10["contrast_err"]
    snr = con / err if (np.isfinite(err) and err > 0) else np.nan
    rep["feat10_snr"] = float(snr) if np.isfinite(snr) else np.nan
    if con > float(c["featured_min_contrast"]) and np.isfinite(snr) and snr > float(c["featured_min_sigma"]):
        rep["verdict"] = "SILICATE_FEATURED"
    elif abs(con) < float(c["featureless_max_contrast"]) and np.isfinite(err) and err < float(c["featureless_max_err"]):
        rep["verdict"] = "FEATURELESS"
    else:
        rep["verdict"] = "AMBIGUOUS"
    fo, ol = f10.get("forsterite_contrast", np.nan), f10.get("olivine_contrast", np.nan)
    rep["crystalline_ratio_11p3_over_9p8"] = float((1 + fo) / (1 + ol)) \
        if np.isfinite(fo) and np.isfinite(ol) and (1 + ol) > 0 else np.nan
    rep["silica_contrast"] = f10.get("silica_contrast", np.nan)
    rep["fes_contrast"] = f23.get("contrast", np.nan)
    return rep


def slag_verdict(score: dict, excess_significant: bool) -> str:
    """Combine the feature score with the photometric excess."""
    v = str(score.get("verdict", "NO_COVERAGE"))
    if v == "NO_COVERAGE":
        return "NOT_SCORED"
    if not excess_significant:
        return "NO_EXCESS_TO_SCORE"
    if v == "SILICATE_FEATURED":
        cr = score.get("crystalline_ratio_11p3_over_9p8", np.nan)
        return "NATURAL_SILICATE_CRYSTALLINE" if (np.isfinite(cr) and cr > 1.05) \
            else "NATURAL_SILICATE_AMORPHOUS"
    if v == "FEATURELESS":
        return "SLAG_CONSISTENT (or large grains; no lab constants for Ca-Al glass)"
    return "AMBIGUOUS"


#: what an IPAC table writes where a value is absent
_IPAC_NULLS: frozenset[str] = frozenset({"", "null", "nan", "none", "-", "--", "n/a", "na", "*"})


def parse_ipac_table(text: str) -> pd.DataFrame:
    """IRSA/IPAC table text -> DataFrame (``|`` header rows, ``\\`` comments)."""
    names: list[str] = []
    rows: list[list[str]] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("\\"):
            continue
        if line.startswith("|"):
            if not names:
                names = [x.strip() for x in line.strip().strip("|").split("|")]
            continue
        rows.append(line.split())
    if not names:
        return pd.DataFrame()
    width = len(names)
    rows = [r[:width] + [""] * (width - len(r)) for r in rows]
    df = pd.DataFrame(rows, columns=names)
    for col in df.columns:
        # `errors="ignore"` is gone in pandas 3 -- it RAISES "invalid error
        # value specified" -- and the runner installs the current pandas while
        # the sandbox had 2.3, so run 35741356662's probe job died here, on the
        # offline gate, before it made a single archive call.  Same semantics,
        # spelled in the surviving API: a column becomes numeric only when every
        # value that is not an IPAC null token converts, so one object name
        # cannot silently turn a text column into NaNs, and one `null` in a
        # wavelength column cannot keep it as text.
        s = df[col]
        missing = s.astype(str).str.strip().str.lower().isin(_IPAC_NULLS)
        conv = pd.to_numeric(s.where(~missing), errors="coerce")
        if bool((conv.notna() | missing).all()) and bool(conv.notna().any()):
            df[col] = conv
    return df


def spectrum_from_table(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick wavelength / flux / error columns from an IRS product table."""
    cols = {c.lower(): c for c in df.columns}
    wcol = next((cols[k] for k in ("wavelength", "wave", "lambda", "wavelength_um") if k in cols), None)
    fcol = next((cols[k] for k in ("flux_density", "flux", "fnu", "flux_jy") if k in cols), None)
    ecol = next((cols[k] for k in ("error", "flux_error", "err", "flux_density_error", "sigma") if k in cols), None)
    if wcol is None or fcol is None:
        return np.array([]), np.array([]), np.array([])
    w = pd.to_numeric(df[wcol], errors="coerce").to_numpy(float)
    f = pd.to_numeric(df[fcol], errors="coerce").to_numpy(float)
    e = pd.to_numeric(df[ecol], errors="coerce").to_numpy(float) if ecol else np.full_like(f, np.nan)
    return w, f, e


__all__ = ["DEFAULT_MINERALOGY", "parse_ipac_table", "score_spectrum", "slag_verdict",
           "spectrum_from_table"]
