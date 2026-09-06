"""The empirical photospheric locus: X_b = Ks − W_b as a function of (J−Ks).

Pure numpy / pandas; fully offline.

The calibration sample is the uniform ``random_index`` subsample carried by
the ``deficit`` track (``is_locus_sample``), quality-cut to clean, unsaturated,
non-variable, well-measured photospheres.  It is split into luminosity classes
by a straight line in the CMD (``giant`` if M_G < giant_mg_max and
bp_rp > giant_bp_rp_min; ``blue`` pooled below blue_bp_rp_max; else ``dwarf``)
and, per class and band, a running median and robust scatter (1.4826·MAD) of
X_b in (J−Ks) bins.  A pooled ``all`` class is always fitted as the fallback.

Sign convention: ``resid = X_obs − X_locus``.  A star FAINTER than its
photosphere in W_b has a larger W_b magnitude, a smaller Ks − W_b, and hence a
NEGATIVE residual.  Negative = deficit.  The +tail is the natural warm-dust
population; the −tail is the control that should be empty.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = ("w1", "w2", "w3")
LUM_CLASSES = ("dwarf", "giant", "blue")
_MAD_TO_SIGMA = 1.4826

DEFAULT_LOCUS_CFG = {
    "tmass_ph_qual": "AAA", "wise_ph_qual_ok": ["A"], "cc_flags_ok": "0000",
    "ext_flag_max": 0, "w1_min_unsat": 8.0, "w2_min_unsat": 7.0,
    "rchi2_quantile": 0.95, "poe_min": 5.0,
    "giant_mg_max": 2.5, "giant_bp_rp_min": 0.9, "blue_bp_rp_max": 0.4,
    "jk_bin_width": 0.05, "jk_min": -0.2, "jk_max": 1.4, "min_per_bin": 25,
    "w3_snr_min": 5.0, "w3_err_max": 0.2, "tail_sigmas": [3.0, 5.0],
    # (G - Ks) vs (BP - RP) locus, per luminosity class and per Galactic zone
    "gks": {"bin_width": 0.05, "x_min": -0.5, "x_max": 4.5, "min_per_bin": 25,
            "plane_b_deg": 10.0},
}
GKS_BAND = "gks"


def _cfg(cfg: dict | None) -> dict:
    out = dict(DEFAULT_LOCUS_CFG)
    out["gks"] = dict(DEFAULT_LOCUS_CFG["gks"])
    for k, v in (cfg or {}).items():
        if k == "gks" and isinstance(v, dict):
            out["gks"].update(v)
        else:
            out[k] = v
    return out


def galactic_zone(df: pd.DataFrame, plane_b_deg: float = 10.0) -> np.ndarray:
    """'plane' (|b| < plane_b_deg) / 'offplane' / 'unknown' per row."""
    b = _num(df, "b")
    return np.where(~np.isfinite(b), "unknown",
                    np.where(np.abs(b) < float(plane_b_deg), "plane", "offplane")).astype(object)


def gks_keys(lum_class: str, zone: str) -> list[str]:
    """Locus-class keys to try, most specific first."""
    return [f"{lum_class}@{zone}", f"all@{zone}", f"{lum_class}@all", "all@all"]


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _str(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    return df[col].astype(object).where(df[col].notna(), "").astype(str).str.strip()


def absolute_g(g: np.ndarray, parallax_mas: np.ndarray) -> np.ndarray:
    """M_G = G + 5 log10(parallax_mas / 100); NaN where the parallax is not positive."""
    g = np.asarray(g, dtype=float)
    p = np.asarray(parallax_mas, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = g + 5.0 * np.log10(p / 100.0)
    out[~(p > 0)] = np.nan
    return out


def w3_usable(df: pd.DataFrame, cfg: dict | None = None) -> np.ndarray:
    """W3 is usable where ``w3snr > w3_snr_min``, OR — when the mirror serves no
    ``w3snr`` at all (the Gaia DR1 AllWISE mirror does not; run 34053752510
    returned NaN for every row) — where ``w3mpro_error < w3_err_max``."""
    c = _cfg(cfg)
    snr = _num(df, "w3snr")
    err = _num(df, "w3mpro_error")
    w3 = _num(df, "w3mpro")
    by_snr = snr > float(c["w3_snr_min"])
    by_err = ~np.isfinite(snr) & (err < float(c["w3_err_max"]))
    return np.isfinite(w3) & (by_snr | by_err)


def luminosity_class(df: pd.DataFrame, cfg: dict | None = None) -> pd.Series:
    """'giant' / 'blue' / 'dwarf' by a straight line in the CMD (configurable)."""
    c = _cfg(cfg)
    mg = absolute_g(_num(df, "phot_g_mean_mag"), _num(df, "parallax"))
    bprp = _num(df, "bp_rp")
    cls = np.full(len(df), "dwarf", dtype=object)
    blue = bprp < float(c["blue_bp_rp_max"])
    giant = (mg < float(c["giant_mg_max"])) & (bprp > float(c["giant_bp_rp_min"]))
    cls[giant] = "giant"
    cls[blue] = "blue"
    return pd.Series(cls, index=df.index, name="lum_class")


def wise_qual_letters(ph_qual: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(W1 letter, W2 letter) from the 4-character AllWISE ph_qual string."""
    s = ph_qual.astype(str).str.upper().str.ljust(4, "X")
    return s.str[0], s.str[1]


def locus_quality_mask(df: pd.DataFrame, cfg: dict | None = None) -> tuple[np.ndarray, dict]:
    """Boolean mask of locus-grade photospheres plus the derived rchi2 thresholds."""
    c = _cfg(cfg)
    n = len(df)
    ok = np.ones(n, dtype=bool)
    if "is_locus_sample" in df.columns:
        ok &= df["is_locus_sample"].fillna(False).astype(bool).to_numpy()
    ok &= (_str(df, "tmass_ph_qual").str.upper() == str(c["tmass_ph_qual"]).upper()).to_numpy()
    q1, q2 = wise_qual_letters(_str(df, "ph_qual"))
    good = [str(x).upper() for x in c["wise_ph_qual_ok"]]
    ok &= q1.isin(good).to_numpy() & q2.isin(good).to_numpy()
    ok &= (_str(df, "cc_flags") == str(c["cc_flags_ok"])).to_numpy()
    ext = _num(df, "ext_flag")
    ok &= np.nan_to_num(ext, nan=0.0) <= float(c["ext_flag_max"])
    w1, w2 = _num(df, "w1mpro"), _num(df, "w2mpro")
    ok &= (w1 > float(c["w1_min_unsat"])) & (w2 > float(c["w2_min_unsat"]))
    ok &= _str(df, "phot_variable_flag").str.upper() != "VARIABLE"
    ok &= _num(df, "parallax_over_error") > float(c["poe_min"])
    for col in ("j_m", "ks_m", "w1mpro", "w2mpro"):
        ok &= np.isfinite(_num(df, col))
    thresholds = {}
    q = float(c["rchi2_quantile"])
    for band in ("w1", "w2"):
        r = _num(df, f"{band}rchi2")
        base = r[ok & np.isfinite(r)]
        thr = float(np.quantile(base, q)) if len(base) >= 20 else np.inf
        thresholds[f"{band}rchi2_max"] = thr
        if np.isfinite(thr):
            ok &= ~(r > thr)
    return ok, thresholds


@dataclass
class Locus:
    """Per (lum_class, band) running median + robust scatter in (J−Ks) bins."""

    bins: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def has(self, lum_class: str, band: str) -> bool:
        return bool(self.bins.get(lum_class, {}).get(band, {}).get("centers"))

    def classes(self) -> list[str]:
        return list(self.bins)

    def predict(self, jk, lum_class="dwarf", band: str = "w1"):
        """(median, scatter) at ``jk`` with linear interpolation and edge clamping.

        ``lum_class`` may be a scalar or an array matched to ``jk``; a class
        without a fit for the band falls back to the pooled ``all`` locus, and
        NaN is returned where nothing is fitted.
        """
        jk = np.atleast_1d(np.asarray(jk, dtype=float))
        med = np.full(jk.shape, np.nan)
        sc = np.full(jk.shape, np.nan)
        classes = np.broadcast_to(np.asarray(lum_class, dtype=object), jk.shape)
        for cls in np.unique(classes.astype(str)):
            sel = classes.astype(str) == cls
            use = cls if self.has(cls, band) else ("all" if self.has("all", band) else None)
            if use is None:
                continue
            b = self.bins[use][band]
            x = np.asarray(b["centers"], dtype=float)
            m = np.asarray(b["median"], dtype=float)
            s = np.asarray(b["scatter"], dtype=float)
            jj = np.clip(jk[sel], x.min(), x.max())          # edge clamping
            med[sel] = np.interp(jj, x, m)
            sc[sel] = np.interp(jj, x, s)
        return med, sc

    def to_dict(self) -> dict:
        return {"bins": self.bins, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> Locus:
        return cls(bins=dict(d.get("bins") or {}), meta=dict(d.get("meta") or {}))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=float))

    @classmethod
    def load(cls, path: Path) -> Locus:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _running_median(jk: np.ndarray, x: np.ndarray, edges: np.ndarray, min_per_bin: int) -> dict:
    idx = np.digitize(jk, edges) - 1
    centers, med, sc, ns = [], [], [], []
    for i in range(len(edges) - 1):
        sel = (idx == i) & np.isfinite(x)
        n = int(sel.sum())
        if n < min_per_bin:
            continue
        v = x[sel]
        m = float(np.median(v))
        centers.append(float(0.5 * (edges[i] + edges[i + 1])))
        med.append(m)
        sc.append(float(_MAD_TO_SIGMA * np.median(np.abs(v - m))))
        ns.append(n)
    return {"centers": centers, "median": med, "scatter": sc, "n": ns}


def fit_locus(sample: pd.DataFrame, cfg: dict | None = None, *,
              apply_quality: bool = True) -> Locus:
    """Fit the locus on ``sample`` (quality-cut here unless ``apply_quality=False``)."""
    c = _cfg(cfg)
    if apply_quality:
        ok, thresholds = locus_quality_mask(sample, c)
        df = sample[ok]
    else:
        df, thresholds = sample, {}
    width = float(c["jk_bin_width"])
    edges = np.arange(float(c["jk_min"]), float(c["jk_max"]) + width / 2, width)
    jk = _num(df, "j_m") - _num(df, "ks_m")
    ks = _num(df, "ks_m")
    cls = luminosity_class(df, c).to_numpy().astype(str)
    w3ok = w3_usable(df, c)
    bins: dict = {}
    n_by_class: dict = {}
    for lc in list(LUM_CLASSES) + ["all"]:
        sel = np.ones(len(df), dtype=bool) if lc == "all" else (cls == lc)
        n_by_class[lc] = int(sel.sum())
        if not sel.any():
            continue
        per_band = {}
        for band in BANDS:
            x = ks - _num(df, f"{band}mpro")
            use = sel & (w3ok if band == "w3" else True)
            fit = _running_median(jk[use], x[use], edges, int(c["min_per_bin"]))
            if fit["centers"]:
                per_band[band] = fit
        if per_band:
            bins[lc] = per_band
    # (G - Ks) vs (BP - RP): a contaminated Ks moves a star off THIS locus by
    # the same amount it moves it off Ks - W1; a screen does not.  The reddening
    # vector runs close to the intrinsic locus, so the fit is per zone.
    g = c["gks"]
    gedges = np.arange(float(g["x_min"]), float(g["x_max"]) + float(g["bin_width"]) / 2,
                       float(g["bin_width"]))
    x = _num(df, "bp_rp")
    y = _num(df, "phot_g_mean_mag") - ks
    zone = galactic_zone(df, float(g["plane_b_deg"]))
    n_gks = {}
    for lc in list(LUM_CLASSES) + ["all"]:
        for z in ("plane", "offplane", "all"):
            sel = (np.ones(len(df), dtype=bool) if lc == "all" else (cls == lc)) \
                & (np.ones(len(df), dtype=bool) if z == "all" else (zone == z))
            if not sel.any():
                continue
            fit = _running_median(x[sel], y[sel], gedges, int(g["min_per_bin"]))
            if fit["centers"]:
                bins.setdefault(f"{lc}@{z}", {})[GKS_BAND] = fit
                n_gks[f"{lc}@{z}"] = int(sel.sum())
    meta = {"n_input": int(len(sample)), "n_locus": int(len(df)),
            "n_by_class": n_by_class, "n_gks_by_class_zone": n_gks,
            "rchi2_thresholds": thresholds,
            "jk_bin_width": width, "config": {k: c[k] for k in c}}
    return Locus(bins=bins, meta=meta)


def gks_residuals(df: pd.DataFrame, locus: Locus, cfg: dict | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """``resid_gks = (G - Ks) - locus(BP - RP)`` and ``sig_gks``; NaN without a fit.

    Positive = Ks too bright for the star's G (or G too faint).  A contaminated
    Ks gives ``resid_gks ~ -resid_w1``; a screen leaves ``resid_gks ~ 0``.
    """
    c = _cfg(cfg)
    n = len(df)
    x = _num(df, "bp_rp")
    y = _num(df, "phot_g_mean_mag") - _num(df, "ks_m")
    e_ks = np.nan_to_num(_num(df, "ks_msigcom"))
    zone = galactic_zone(df, float(c["gks"]["plane_b_deg"]))
    cls = luminosity_class(df, c).to_numpy().astype(str)
    med = np.full(n, np.nan)
    sc = np.full(n, np.nan)
    for lc in np.unique(cls):
        for z in np.unique(zone):
            sel = (cls == lc) & (zone == z)
            key = next((k for k in gks_keys(lc, z if z != "unknown" else "all")
                        if locus.has(k, GKS_BAND)), None)
            if key is None:
                continue
            m, s_ = locus.predict(x[sel], key, GKS_BAND)
            med[sel], sc[sel] = m, s_
    resid = y - med
    sig = resid / np.sqrt(np.square(sc) + np.square(e_ks) + 0.01 ** 2)
    return resid, sig


def residuals(df: pd.DataFrame, locus: Locus, cfg: dict | None = None) -> pd.DataFrame:
    """Add ``jk``, ``lum_class``, ``resid_w{1,2,3}``, ``sig_w{1,2,3}``.

    ``sig = resid / sqrt(scatter² + e_ks² + e_wb²)``; W3 is only evaluated
    where ``w3snr > w3_snr_min``.  Negative = deficit.
    """
    c = _cfg(cfg)
    out = df.copy()
    jk = _num(out, "j_m") - _num(out, "ks_m")
    ks, e_ks = _num(out, "ks_m"), _num(out, "ks_msigcom")
    cls = luminosity_class(out, c)
    out["jk"] = jk
    out["lum_class"] = cls
    e_ks = np.where(np.isfinite(e_ks), e_ks, 0.0)
    w3ok = w3_usable(out, c)
    for band in BANDS:
        w = _num(out, f"{band}mpro")
        e_w = _num(out, f"{band}mpro_error")
        e_w = np.where(np.isfinite(e_w), e_w, 0.0)
        med, sc = locus.predict(jk, cls.to_numpy().astype(str), band)
        resid = (ks - w) - med
        sig = resid / np.sqrt(np.square(sc) + np.square(e_ks) + np.square(e_w))
        if band == "w3":
            resid = np.where(w3ok, resid, np.nan)
            sig = np.where(w3ok, sig, np.nan)
        out[f"resid_{band}"] = resid
        out[f"sig_{band}"] = sig
    out["resid_gks"], out["sig_gks"] = gks_residuals(out, locus, c)
    return out


def tail_asymmetry(df: pd.DataFrame, cfg: dict | None = None) -> dict:
    """Counts beyond ±kσ per band: the +tail is dust, the −tail is the control."""
    c = _cfg(cfg)
    out = {}
    for band in BANDS:
        s = _num(df, f"sig_{band}")
        s = s[np.isfinite(s)]
        rep = {"n": int(len(s))}
        for k in c["tail_sigmas"]:
            k = float(k)
            rep[f"n_excess_gt_{k:g}sig"] = int((s > k).sum())
            rep[f"n_deficit_lt_-{k:g}sig"] = int((s < -k).sum())
        if len(s):
            rep["median_sig"] = float(np.median(s))
            rep["robust_sigma"] = float(_MAD_TO_SIGMA * np.median(np.abs(s - np.median(s))))
        out[band] = rep
    return out


__all__ = ["BANDS", "DEFAULT_LOCUS_CFG", "GKS_BAND", "LUM_CLASSES", "Locus", "absolute_g",
           "fit_locus", "galactic_zone", "gks_keys", "gks_residuals", "locus_quality_mask",
           "luminosity_class", "residuals", "tail_asymmetry", "w3_usable", "wise_qual_letters"]
