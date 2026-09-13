"""Pattern matching of predicted rotational lines against a U-line list.

Pure functions; the run stage feeds them tables and gets records back.

The unit of evidence is a **feature**, not a catalogue line: K-components of
one J→J+1 transition of a symmetric top (and any other lines closer together
than the source linewidth) are blended in the survey, so predicted lines are
merged within the matching tolerance before anything is counted — otherwise a
single U-line "coincides" with six K-components and the count is inflated
six-fold.

For one species × source × T_ex the test is:

* Doppler-correct the rest frequencies (radio convention) when the survey
  tabulates *sky* frequencies; surveys that already tabulate rest frequencies
  at their assumed v_LSR get no shift, and the v_LSR uncertainty between the
  source's velocity components widens the tolerance instead.
* Tolerance per pair ``= max(ν·FWHM/c, σ_pred, σ_U) + ν·Δv_unc/c``.
* Restrict to the survey's frequency coverage (a predicted line counts as
  observable only if the survey catalogued *some* line within
  ``coverage_window_mhz`` of it) and to the top-N predicted features by
  intensity at T_ex.
* One-to-one nearest-neighbour coincidences against the **clean** U-lines
  (U-lines within tolerance of a baseline/contaminant line are vetoed first
  and counted).
* LTE consistency: Spearman ρ between predicted and observed intensity over
  the coincident set, and the fraction of the five strongest predicted
  features (in coverage) that coincide with *any* catalogued line — a strong
  predicted line that is neither a U-line nor a blend with an identified line
  is *missing*, which no excitation model repairs.
* False-alarm probability: ``n_trials`` random rigid shifts of the predicted
  feature set by ``±U(shift_min, shift_max)`` MHz — the pattern's internal
  spacing is preserved, only its alignment with the list is destroyed —
  ``p_false = (k + 1)/(n_trials + 1)`` where ``k`` trials reach the observed
  coincidence count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .lines import rescale_lgint

C_KM_S = 299792.458

DEFAULT_MATCH: dict = {
    "tex_grid_k": [10.0, 30.0, 50.0, 100.0, 150.0, 200.0],
    "top_n_predicted": 40,
    "min_coincidences": 3,
    "spearman_min": 0.3,
    "min_top5_fraction": 0.6,
    "p_false_max": 0.01,
    "n_trials": 1000,
    "shift_min_mhz": 50.0,
    "shift_max_mhz": 500.0,
    "coverage_window_mhz": 1500.0,
    "veto_dynamic_range_dex": 4.0,
    "seed": 20260913,
}


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def doppler_sky(freq_rest_mhz, v_km_s: float) -> np.ndarray:
    """Rest → sky frequency, radio convention ν_sky = ν_rest (1 − v/c)."""
    return np.asarray(freq_rest_mhz, dtype=float) * (1.0 - float(v_km_s) / C_KM_S)


def linewidth_mhz(freq_mhz, fwhm_km_s: float) -> np.ndarray:
    return np.asarray(freq_mhz, dtype=float) * float(fwhm_km_s) / C_KM_S


def tolerance_mhz(freq_mhz, fwhm_km_s: float, err_mhz=0.0, v_unc_km_s: float = 0.0
                  ) -> np.ndarray:
    """``max(linewidth, catalogue error) + ν·Δv_unc/c`` (the U-line's own error
    is folded in per pair inside :func:`match_features`)."""
    f = np.asarray(freq_mhz, dtype=float)
    return (np.maximum(linewidth_mhz(f, fwhm_km_s), np.asarray(err_mhz, dtype=float))
            + f * float(v_unc_km_s) / C_KM_S)


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
@dataclass
class SourceLines:
    """A survey's line list, split into U-lines and everything catalogued."""

    name: str
    u_freq: np.ndarray                 # MHz, sorted
    u_int: np.ndarray                  # observed intensity (any consistent unit), NaN allowed
    u_err: np.ndarray                  # MHz
    all_freq: np.ndarray               # every catalogued line (identified + U), sorted
    fwhm_km_s: float
    v_lsr_km_s: float = 0.0
    frame: str = "rest"                # "rest" (no shift) | "sky" (shift predictions)
    v_unc_km_s: float = 0.0
    fmin_mhz: float | None = None
    fmax_mhz: float | None = None
    u_veto: np.ndarray | None = None   # bool per U-line: vetoed by a contaminant line
    u_veto_species: list[str] = field(default_factory=list)
    u_id: np.ndarray | None = None     # original row ids

    def __post_init__(self):
        order = np.argsort(self.u_freq)
        self.u_freq = np.asarray(self.u_freq, dtype=float)[order]
        self.u_int = np.asarray(self.u_int, dtype=float)[order]
        self.u_err = np.nan_to_num(np.asarray(self.u_err, dtype=float)[order], nan=0.0)
        if self.u_id is None:
            self.u_id = np.arange(len(order))
        self.u_id = np.asarray(self.u_id)[order]
        self.all_freq = np.sort(np.asarray(self.all_freq, dtype=float))
        if self.u_veto is None:
            self.u_veto = np.zeros(len(self.u_freq), dtype=bool)
        else:
            self.u_veto = np.asarray(self.u_veto, dtype=bool)[order]
        if self.fmin_mhz is None:
            self.fmin_mhz = float(self.all_freq.min()) if len(self.all_freq) else 0.0
        if self.fmax_mhz is None:
            self.fmax_mhz = float(self.all_freq.max()) if len(self.all_freq) else 0.0

    @property
    def n_ulines(self) -> int:
        return int(len(self.u_freq))

    @property
    def n_clean(self) -> int:
        return int((~self.u_veto).sum())


def apply_vetoes(src: SourceLines, contaminants: dict[str, tuple[np.ndarray, np.ndarray]],
                 *, dynamic_range_dex: float = 4.0) -> SourceLines:
    """Mark U-lines within tolerance of a baseline/contaminant line.

    ``contaminants`` maps species → (freq_mhz at the survey frame, lgint at the
    contaminant's T_ex).  Only lines within ``dynamic_range_dex`` of that
    species' strongest line inside the survey range are veto-capable, so a
    forest of 10⁻¹⁰-strength methanol lines does not veto the whole list.
    """
    veto = np.zeros(src.n_ulines, dtype=bool)
    names: list[list[str]] = [[] for _ in range(src.n_ulines)]
    tol_u = tolerance_mhz(src.u_freq, src.fwhm_km_s, src.u_err, src.v_unc_km_s)
    # a contaminant line just beyond the survey's last catalogued frequency can
    # still sit within tolerance of an edge U-line: pad the range by the widest
    # tolerance in the list
    pad = float(tol_u.max()) + 1.0 if len(tol_u) else 1.0
    for sp, (cf, cl) in contaminants.items():
        cf = np.asarray(cf, dtype=float)
        cl = np.asarray(cl, dtype=float)
        inr = (cf >= src.fmin_mhz - pad) & (cf <= src.fmax_mhz + pad) & np.isfinite(cl)
        if not inr.any():
            continue
        floor = float(cl[inr].max()) - float(dynamic_range_dex)
        keep = inr & (cl >= floor)
        f = np.sort(cf[keep])
        if not len(f):
            continue
        idx = np.searchsorted(f, src.u_freq)
        lo = f[np.clip(idx - 1, 0, len(f) - 1)]
        hi = f[np.clip(idx, 0, len(f) - 1)]
        sep = np.minimum(np.abs(src.u_freq - lo), np.abs(src.u_freq - hi))
        hit = sep <= tol_u
        for i in np.flatnonzero(hit):
            names[i].append(sp)
        veto |= hit
    out = SourceLines(src.name, src.u_freq, src.u_int, src.u_err, src.all_freq, src.fwhm_km_s,
                      src.v_lsr_km_s, src.frame, src.v_unc_km_s, src.fmin_mhz, src.fmax_mhz,
                      veto, [";".join(n) for n in names], src.u_id)
    return out


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def merge_blends(freq_mhz, lgint, err_mhz, labels, *, fwhm_km_s: float,
                 v_unc_km_s: float = 0.0) -> pd.DataFrame:
    """Merge predicted lines closer than the tolerance into single features."""
    f = np.asarray(freq_mhz, dtype=float)
    lg = np.asarray(lgint, dtype=float)
    er = np.asarray(err_mhz, dtype=float)
    lab = np.asarray(labels, dtype=object)
    ok = np.isfinite(f) & np.isfinite(lg)
    f, lg, er, lab = f[ok], lg[ok], er[ok], lab[ok]
    order = np.argsort(f)
    f, lg, er, lab = f[order], lg[order], er[order], lab[order]
    rows = []
    i = 0
    n = len(f)
    while i < n:
        j = i + 1
        tol = float(tolerance_mhz(f[i], fwhm_km_s, er[i], v_unc_km_s))
        while j < n and f[j] - f[j - 1] <= tol:
            tol = float(tolerance_mhz(f[j], fwhm_km_s, er[j], v_unc_km_s))
            j += 1
        w = 10.0 ** lg[i:j]
        tot = float(w.sum())
        fc = float(np.sum(f[i:j] * w) / tot) if tot > 0 else float(np.mean(f[i:j]))
        rows.append((fc, float(np.log10(tot)) if tot > 0 else -np.inf, float(er[i:j].max()),
                     int(j - i), "|".join(str(x) for x in lab[i:j])))
        i = j
    return pd.DataFrame(rows, columns=["freq_mhz", "lgint", "err_mhz", "n_members", "members"])


def coverage_mask(freq_mhz, all_freq_sorted, window_mhz: float) -> np.ndarray:
    """True where the survey catalogued some line within ``window_mhz``."""
    f = np.asarray(freq_mhz, dtype=float)
    a = np.asarray(all_freq_sorted, dtype=float)
    if not len(a):
        return np.zeros(len(f), dtype=bool)
    idx = np.searchsorted(a, f)
    lo = a[np.clip(idx - 1, 0, len(a) - 1)]
    hi = a[np.clip(idx, 0, len(a) - 1)]
    return np.minimum(np.abs(f - lo), np.abs(f - hi)) <= float(window_mhz)


def match_features(feat_freq, feat_tol, u_freq_sorted, u_err_sorted
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-to-one nearest-neighbour coincidences.

    Returns ``(feature_index, uline_index, separation_mhz)`` for the accepted
    pairs; each feature and each U-line is used at most once, closest pairs
    first.  A pair is accepted when ``sep ≤ max(feat_tol, u_err)``.
    """
    f = np.asarray(feat_freq, dtype=float)
    t = np.asarray(feat_tol, dtype=float)
    u = np.asarray(u_freq_sorted, dtype=float)
    ue = np.asarray(u_err_sorted, dtype=float)
    if not len(f) or not len(u):
        return np.array([], int), np.array([], int), np.array([], float)
    idx = np.searchsorted(u, f)
    cand_i, cand_j, cand_s = [], [], []
    for off in (-1, 0):
        jj = np.clip(idx + off, 0, len(u) - 1)
        sep = np.abs(f - u[jj])
        ok = sep <= np.maximum(t, ue[jj])
        cand_i.append(np.flatnonzero(ok))
        cand_j.append(jj[ok])
        cand_s.append(sep[ok])
    ci = np.concatenate(cand_i)
    cj = np.concatenate(cand_j)
    cs = np.concatenate(cand_s)
    order = np.argsort(cs, kind="stable")
    used_f: set[int] = set()
    used_u: set[int] = set()
    oi, oj, os_ = [], [], []
    for k in order:
        i, j = int(ci[k]), int(cj[k])
        if i in used_f or j in used_u:
            continue
        used_f.add(i)
        used_u.add(j)
        oi.append(i)
        oj.append(j)
        os_.append(float(cs[k]))
    return np.array(oi, int), np.array(oj, int), np.array(os_, float)


def count_matches(feat_freq, feat_tol, u_freq_sorted, u_err_sorted) -> int:
    """Number of distinct U-lines within tolerance of some feature (trial statistic)."""
    f = np.asarray(feat_freq, dtype=float)
    u = np.asarray(u_freq_sorted, dtype=float)
    if not len(f) or not len(u):
        return 0
    t = np.asarray(feat_tol, dtype=float)
    ue = np.asarray(u_err_sorted, dtype=float)
    idx = np.searchsorted(u, f)
    hit: list[np.ndarray] = []
    for off in (-1, 0):
        jj = np.clip(idx + off, 0, len(u) - 1)
        ok = np.abs(f - u[jj]) <= np.maximum(t, ue[jj])
        hit.append(jj[ok])
    return int(len(np.unique(np.concatenate(hit)))) if hit else 0


def features_matched(feat_freq, feat_tol, u_freq_sorted, u_err_sorted) -> np.ndarray:
    """Bool per feature: some catalogued line lies within tolerance (blends allowed)."""
    f = np.asarray(feat_freq, dtype=float)
    u = np.asarray(u_freq_sorted, dtype=float)
    if not len(f) or not len(u):
        return np.zeros(len(f), dtype=bool)
    t = np.asarray(feat_tol, dtype=float)
    ue = np.asarray(u_err_sorted, dtype=float)
    idx = np.searchsorted(u, f)
    out = np.zeros(len(f), dtype=bool)
    for off in (-1, 0):
        jj = np.clip(idx + off, 0, len(u) - 1)
        out |= np.abs(f - u[jj]) <= np.maximum(t, ue[jj])
    return out


def spearman(x, y) -> tuple[float, float]:
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    r = spearmanr(x[ok], y[ok])
    rho = float(r[0]) if np.isfinite(r[0]) else float("nan")
    p = float(r[1]) if np.isfinite(r[1]) else float("nan")
    return rho, p


# ---------------------------------------------------------------------------
# the test
# ---------------------------------------------------------------------------
def _predicted_at_tex(lines: pd.DataFrame, entries: dict, tex: float) -> np.ndarray:
    lg = np.full(len(lines), np.nan)
    for eid, grp in lines.groupby("entry_id"):
        ent = entries[eid]
        lg[grp.index.to_numpy()] = rescale_lgint(grp["lgint_300"], grp["elo_cm"],
                                                 grp["freq_mhz"], ent["temps"], ent["qlog"], tex)
    return lg


def false_alarm(feat_freq, feat_tol, src: SourceLines, n_obs: int, *, n_trials: int,
                rng: np.random.Generator, shift_min: float, shift_max: float,
                coverage_window: float, clean_mask: np.ndarray,
                min_coinc: int, top5_min: float) -> dict:
    """Rigid random shifts of the feature set; see the module docstring."""
    f = np.asarray(feat_freq, dtype=float)
    t = np.asarray(feat_tol, dtype=float)
    u = src.u_freq[clean_mask]
    ue = src.u_err[clean_mask]
    k_count = 0
    k_full = 0
    counts = np.zeros(int(n_trials), dtype=int)
    for i in range(int(n_trials)):
        d = rng.uniform(shift_min, shift_max) * (1.0 if rng.random() < 0.5 else -1.0)
        fs = f + d
        inr = (fs >= src.fmin_mhz) & (fs <= src.fmax_mhz)
        inr &= coverage_mask(fs, src.all_freq, coverage_window)
        c = count_matches(fs[inr], t[inr], u, ue)
        counts[i] = c
        if c >= n_obs:
            k_count += 1
        if c >= min_coinc:
            # the "missing strong line" test on the shifted set (features are
            # already ordered by intensity, so the first five in range are the top-5)
            top = np.flatnonzero(inr)[:5]
            if len(top):
                anyc = features_matched(fs[top], t[top], src.all_freq,
                                        np.zeros(len(src.all_freq))).sum()
                if anyc / len(top) >= top5_min:
                    k_full += 1
    n = int(n_trials)
    return {"n_trials": n, "k_ge_observed": int(k_count), "k_full_pattern": int(k_full),
            "p_false": (k_count + 1) / (n + 1), "p_false_full": (k_full + 1) / (n + 1),
            "trial_count_mean": float(counts.mean()) if n else float("nan"),
            "trial_count_max": int(counts.max()) if n else 0}


def evaluate_species(lines: pd.DataFrame, entries: dict, src: SourceLines, *,
                     species: str, line_source: str, conf: dict | None = None,
                     rng: np.random.Generator | None = None) -> list[dict]:
    """Run the pattern test on one species against one source over the T_ex grid.

    ``lines`` carries ``freq_mhz, err_mhz, lgint_300, elo_cm, entry_id`` (and
    optionally ``qn_up/qn_lo``); ``entries`` maps ``entry_id`` → dict with
    ``temps`` and ``qlog``.  One record per T_ex.
    """
    conf = {**DEFAULT_MATCH, **(conf or {})}
    rng = rng or np.random.default_rng(int(conf["seed"]))
    out: list[dict] = []
    clean = ~src.u_veto
    n_clean = int(clean.sum())
    lines = lines.reset_index(drop=True)
    labels = (lines["qn_up"].astype(str) + "<-" + lines["qn_lo"].astype(str)
              if "qn_up" in lines else pd.Series([""] * len(lines)))
    for tex in conf["tex_grid_k"]:
        rec = {"species": species, "source": src.name, "line_source": line_source,
               "tex_k": float(tex), "n_ulines": src.n_ulines, "n_ulines_clean": n_clean,
               "n_ulines_vetoed": int(src.n_ulines - n_clean)}
        if not len(lines) or src.n_ulines == 0:
            rec.update({"status": "NO_LINES" if not len(lines) else "NO_ULINES",
                        "n_features_in_coverage": 0, "n_coincident": 0, "pattern": False})
            out.append(rec)
            continue
        lg = _predicted_at_tex(lines, entries, float(tex))
        f_rest = lines["freq_mhz"].to_numpy(dtype=float)
        f_sky = doppler_sky(f_rest, src.v_lsr_km_s) if src.frame == "sky" else f_rest
        feats = merge_blends(f_sky, lg, lines["err_mhz"].to_numpy(dtype=float), labels.to_numpy(),
                             fwhm_km_s=src.fwhm_km_s, v_unc_km_s=src.v_unc_km_s)
        inr = (feats["freq_mhz"] >= src.fmin_mhz) & (feats["freq_mhz"] <= src.fmax_mhz)
        feats = feats[inr]
        cov = coverage_mask(feats["freq_mhz"], src.all_freq, conf["coverage_window_mhz"])
        n_in_range = int(len(feats))
        feats = feats[cov].sort_values("lgint", ascending=False).reset_index(drop=True)
        n_cov = int(len(feats))
        top = feats.head(int(conf["top_n_predicted"]))
        ff = top["freq_mhz"].to_numpy(dtype=float)
        ft = tolerance_mhz(ff, src.fwhm_km_s, top["err_mhz"].to_numpy(dtype=float),
                           src.v_unc_km_s)
        fi, uj_local, sep = match_features(ff, ft, src.u_freq[clean], src.u_err[clean])
        uj = np.flatnonzero(clean)[uj_local] if len(uj_local) else np.array([], int)
        n_coinc = int(len(fi))
        # coincidences with vetoed U-lines, for the ledger
        vi, _, _ = match_features(ff, ft, src.u_freq[~clean], src.u_err[~clean])
        # top-5 "missing strong line" tests
        t5 = min(5, len(ff))
        if t5:
            any5 = int(features_matched(ff[:t5], ft[:t5], src.all_freq,
                                        np.zeros(len(src.all_freq))).sum())
            u5 = int(np.sum(fi < t5))
            top5_any, top5_u = any5 / t5, u5 / t5
        else:
            top5_any = top5_u = float("nan")
        # LTE consistency over the coincident set
        if n_coinc >= 3:
            obs = src.u_int[uj]
            with np.errstate(invalid="ignore", divide="ignore"):
                lobs = np.where(obs > 0, np.log10(np.where(obs > 0, obs, 1.0)), np.nan)
            rho, p_rho = spearman(top["lgint"].to_numpy()[fi], lobs)
            n_lte = int(np.isfinite(lobs).sum())
        else:
            rho, p_rho, n_lte = float("nan"), float("nan"), 0
        lte_testable = bool(n_lte >= 3 and np.isfinite(rho))
        fap = false_alarm(ff, ft, src, n_coinc, n_trials=int(conf["n_trials"]), rng=rng,
                          shift_min=float(conf["shift_min_mhz"]),
                          shift_max=float(conf["shift_max_mhz"]),
                          coverage_window=float(conf["coverage_window_mhz"]),
                          clean_mask=clean, min_coinc=int(conf["min_coincidences"]),
                          top5_min=float(conf["min_top5_fraction"]))
        count_ok = n_coinc >= int(conf["min_coincidences"])
        lte_ok = (rho >= float(conf["spearman_min"])) if lte_testable else None
        top5_ok = bool(np.isfinite(top5_any) and top5_any >= float(conf["min_top5_fraction"]))
        fap_ok = fap["p_false"] <= float(conf["p_false_max"])
        pattern = bool(count_ok and (lte_ok is not False) and top5_ok and fap_ok)
        rec.update({
            "status": "OK",
            "n_lines_input": int(len(lines)), "n_features_in_range": n_in_range,
            "n_features_in_coverage": n_cov, "n_features_tested": int(len(top)),
            "n_coincident": n_coinc, "n_coincident_vetoed": int(len(vi)),
            "coincident_freq_mhz": [round(float(x), 3) for x in ff[fi]],
            "coincident_uline_freq_mhz": [round(float(x), 3) for x in src.u_freq[uj]],
            "coincident_sep_mhz": [round(float(x), 3) for x in sep],
            "coincident_members": [str(x) for x in top["members"].to_numpy()[fi]],
            "coincident_pred_lgint": [round(float(x), 3) for x in top["lgint"].to_numpy()[fi]],
            "coincident_obs_int": [None if not np.isfinite(x) else float(x)
                                   for x in src.u_int[uj]],
            "spearman_rho": rho, "spearman_p": p_rho, "n_lte_points": n_lte,
            "lte_testable": lte_testable, "lte_pass": lte_ok,
            "top5_fraction_any": top5_any, "top5_fraction_uline": top5_u,
            "tests": {"count": bool(count_ok), "lte": lte_ok, "top5": top5_ok,
                      "p_false": bool(fap_ok)},
            "pattern": pattern, **fap,
        })
        out.append(rec)
    return out


def best_record(records: list[dict]) -> dict | None:
    """The T_ex record to quote: a pattern if any, else the most coincidences."""
    ok = [r for r in records if r.get("status") == "OK"]
    if not ok:
        return records[0] if records else None
    pats = [r for r in ok if r.get("pattern")]
    pool = pats or ok
    return min(pool, key=lambda r: (-(r.get("n_coincident") or 0), r.get("p_false", 1.0)))


__all__ = ["C_KM_S", "DEFAULT_MATCH", "SourceLines", "apply_vetoes", "best_record",
           "count_matches", "coverage_mask", "doppler_sky", "evaluate_species", "false_alarm",
           "features_matched", "linewidth_mhz", "match_features", "merge_blends", "spearman",
           "tolerance_mhz"]
