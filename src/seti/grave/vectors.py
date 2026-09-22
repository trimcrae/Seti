"""GRAVE detector physics: fission and refined-particulate vectors in a sediment.

Everything here is a pure function of arrays and is exercised offline by
``tests/test_grave.py``.  Three ideas, each stated once:

**1. Fission product in a sedimentary matrix is a mass vector, and the
matrix decides which elements carry it.**  ``seti.fallout.yields`` gives the
U-235 chain yields per element at a decay horizon; multiplied by the atomic
mass they are a composition by weight.  Folded against PAAS (the shale the
sample is being compared with) the *relative* enrichment per element is
``(Phi_X / PAAS_X)`` and the ranking is nothing like the stellar one: the
crust is ~10^5 times poorer in Ru, Rh, Pd, Te than in Nd, so a fission
inventory that raises Nd by 10 % raises Ru by a factor of thousands.  Mo is
~30x more sensitive than Nd, Cs ~1.4x; among the REE the fission pattern is
Pr ~ Nd ~ Sm ~ Eu >> La ~ Ce (x 0.25) >> Gd (x 0.02) -- a mid-REE hump with a
cliff at Gd that apatite's smooth MREE bulge does not have -- and fission adds
NO Hf with its Zr (zircon does, at Zr/Hf ~ 40), NO Th with its LREE (monazite
carries 5 wt% Th), NO U or V with its Mo (anoxia does), NO Mn or Co with its
Mo (the Fe-Mn shuttle does).  Those *absent partners* are the discriminant;
:func:`fission_discriminants` tabulates them from the yields rather than
asserting them.

**2. The natural alternative is a non-negative mixture of reservoirs, not a
single baseline.**  Each sample's concentration vector (majors included, so
Al pins the detrital fraction, Ca the carbonate, P the apatite, Mn the oxide)
is fitted as ``c = sum_i f_i R_i`` over the reservoir family of
:mod:`seti.grave.references` with ``f_i >= 0`` (weighted NNLS in linear space,
refined in log space), and again with the fission vector as one extra
non-negative column.  ``fission_lr = 0.5 (chi2_natural - chi2_with_fission)``
is the statistic.  Because the natural family already contains every
process the brief lists as a kill, a sample that needs fission is a sample
none of those processes can build.

**3. Refined material is diagnosed by ratios, against chondrite and crust.**
An impact layer carries the PGE in CI proportions anchored on Ir; ultramafic
detritus carries the IPGE (Os, Ir, Ru) with Pd/Ir well below chondritic;
refined catalyst carries Pt-Pd-Rh with Ir and Os at crustal background;
fission carries Ru-Rh-Pd with neither Ir nor Pt.  Refined Ta has no Nb; a
tungsten alloy has no Sn/Mo/Bi.  :func:`classify_pge` and
:func:`classify_alloy` return a class and the ratios that decided it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

from ..fallout import yields as Y
from . import references as R

LN10 = float(np.log(10.0))

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
NORMAL = "NORMAL"
INSUFFICIENT = "INSUFFICIENT"
UNEXPLAINED = "UNEXPLAINED_BY_ALL_RESERVOIRS"
FISSION_CANDIDATE = "FISSION_CANDIDATE"
FISSION_AMBIGUOUS = "FISSION_AMBIGUOUS"
NATURAL = "NATURAL"

#: Named vetoes, in order of application; each is a counter in summary.json.
VETOES: tuple[str, ...] = (
    "insufficient_panel",
    "detection_limit_driver",
    "unexplained_by_all_reservoirs",
    "single_element_driver",
    "peak_incoherent",
    "redox_conditioned",
    "femn_shuttle",
    "heavy_mineral_zr_hf",
    "monazite_th",
    "hydrothermal_ba",
    "impact_pge",
)

#: PGE / alloy classes of the refined-particulate test.
PGE_INSUFFICIENT = "pge_insufficient"
PGE_BACKGROUND = "pge_background"
PGE_IMPACT = "impact"
PGE_ULTRAMAFIC = "ultramafic"
PGE_REFINED = "refined_pge"
PGE_FISSION_LIKE = "fission_like_pge"
PGE_FEMN = "femn_pt"
PGE_UNCLASSIFIED = "pge_anomalous_unclassified"

ALLOY_NONE = "alloy_none"
ALLOY_INSUFFICIENT = "alloy_insufficient"
ALLOY_REFINED_TA = "refined_ta"
ALLOY_REFINED_W = "refined_w"
ALLOY_NATURAL_TA = "natural_nb_ta"
ALLOY_NATURAL_W = "hydrothermal_w"


@dataclass(frozen=True)
class GraveConfig:
    """Thresholds; every value is mirrored in ``config/grave.yaml``."""

    horizon_yr: float = Y.DEFAULT_HORIZON_YR
    anchor: str = "Nd"
    # error model (dex): majors / trace / ultra-trace, plus a systematic floor
    sigma_major_dex: float = 0.05
    sigma_trace_dex: float = 0.10
    sigma_ultratrace_dex: float = 0.20
    systematic_dex: float = 0.05
    ultratrace_ppm: float = 0.05         # elements with PAAS below this are ultra-trace
    min_elements: int = 8
    min_fission_elements: int = 4
    # decision
    lr_min: float = 8.0
    ambiguity_margin: float = 2.0
    max_reduced_chi2: float = 4.0
    refine_lr: float = 2.0               # log-space refinement above this linear LR
    null_quantile: float = 0.999
    # peak coherence: >= n_heavy of the heavy set and >= n_light of the light set
    # individually >= peak_sigma above the natural model
    heavy_set: tuple[str, ...] = ("Cs", "Pr", "Nd", "Sm", "Eu")
    light_set: tuple[str, ...] = ("Zr", "Mo", "Ru", "Rh", "Pd", "Te")
    n_heavy: int = 2
    n_light: int = 1
    peak_sigma: float = 2.0
    # kills (enrichment factors are (X/Al)_sample / (X/Al)_PAAS)
    redox_ef_u: float = 3.0
    redox_ef_v: float = 3.0
    redox_toc_pct: float = 2.0
    femn_ef_mn: float = 5.0
    zr_hf_natural: tuple[float, float] = (25.0, 60.0)
    monazite_th_ef: float = 3.0
    hydro_ef_ba: float = 5.0
    # refined-particulate test
    pge_ef_anom: float = 10.0
    pge_ratio_tol: float = 3.0           # within x3 of chondritic counts as chondritic
    ipge_pd_ir_max: float = 0.3          # Pd/Ir below 0.3 x CI: ultramafic
    ta_ef_anom: float = 10.0
    nb_ta_natural: tuple[float, float] = (5.0, 40.0)
    w_ef_anom: float = 20.0
    w_partner_ef_max: float = 3.0
    # detection limits: a value repeated this often at a column's floor is a DL
    dl_repeat_min: int = 20
    dl_floor_fraction: float = 0.5       # ... and it is within this of the column minimum
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. The fission vector in mass space
# ---------------------------------------------------------------------------
def fission_mass_vector(cfg: GraveConfig | None = None, *, anchor_ppm: float | None = None
                        ) -> dict[str, float]:
    """ppm of each element that fission product adds at amplitude ``a = 1``.

    Normalised so that ``a = 1`` adds ``anchor_ppm`` (default: PAAS Nd,
    33.9 ppm) of the anchor element -- i.e. doubles the shale's Nd.  Elements
    are those on which a chain sits at the decay horizon (Tc has become Ru at
    1 Myr; Cs-135 is still Cs; I-129 is still I).
    """
    cfg = cfg or GraveConfig()
    y = Y.element_yields(cfg.horizon_yr)
    mass = {el: v * R.ATOMIC_MASS.get(el, np.nan) for el, v in y.items()}
    ref = mass.get(cfg.anchor)
    if not ref or not np.isfinite(ref):
        raise ValueError(f"anchor {cfg.anchor} carries no fission yield at this horizon")
    scale = float(anchor_ppm if anchor_ppm is not None else R.PAAS[cfg.anchor]) / ref
    return {el: float(m * scale) for el, m in mass.items() if np.isfinite(m)}


def fission_discriminants(cfg: GraveConfig | None = None) -> list[dict]:
    """Per element: relative enrichment per unit anchor doubling, against PAAS.

    ``rel_to_anchor = (Phi_X / PAAS_X) / (Phi_anchor / PAAS_anchor)``; its
    log10 is the ``[X/Nd]`` the fission vector imposes on a shale.  This is
    the table ``docs/grave.md`` quotes and ``summary.json`` carries.
    """
    cfg = cfg or GraveConfig()
    phi = fission_mass_vector(cfg)
    ref = phi[cfg.anchor] / R.PAAS[cfg.anchor]
    rows = []
    for el in sorted(phi, key=lambda e: R.ATOMIC_MASS.get(e, 0)):
        p = R.PAAS.get(el)
        if p is None or p <= 0:
            continue
        rel = (phi[el] / p) / ref
        rows.append({"element": el, "fission_ppm_at_a1": round(phi[el], 5), "paas_ppm": p,
                     "rel_to_anchor": round(rel, 5),
                     "dex_vs_anchor": round(float(np.log10(rel)), 3) if rel > 0 else None,
                     "paas_from_ucc": el in R.PAAS_FROM_UCC})
    return rows


# ---------------------------------------------------------------------------
# 2. The mixture fit
# ---------------------------------------------------------------------------
@dataclass
class Design:
    """Reservoir matrix over a fixed element order."""

    elements: list[str]
    reservoirs: list[str]
    A: np.ndarray            # (K, M) ppm: natural reservoirs
    phi: np.ndarray          # (K,) ppm: fission at a = 1
    sigma: np.ndarray        # (K,) dex: analytical + systematic (before floors)

    def to_dict(self) -> dict:
        return {"elements": list(self.elements), "reservoirs": list(self.reservoirs),
                "fission_ppm_at_a1": {e: round(float(v), 6) for e, v in zip(self.elements, self.phi, strict=True)},
                "sigma_dex": {e: round(float(s), 4) for e, s in zip(self.elements, self.sigma, strict=True)},
                "reservoir_sources": {r: R.RESERVOIR_SOURCES[r] for r in self.reservoirs}}


def default_sigma(el: str, cfg: GraveConfig) -> float:
    if el in R.MAJORS:
        s = cfg.sigma_major_dex
    elif R.PAAS.get(el, 1.0) < cfg.ultratrace_ppm:
        s = cfg.sigma_ultratrace_dex
    else:
        s = cfg.sigma_trace_dex
    return float(np.hypot(s, cfg.systematic_dex))


def build_design(elements, cfg: GraveConfig | None = None, *, reservoirs=None,
                 sigma_floor: dict[str, float] | None = None) -> Design:
    """Design over the elements a dataset actually carries (unknowns dropped)."""
    cfg = cfg or GraveConfig()
    names = list(reservoirs or R.RESERVOIRS.keys())
    phi_all = fission_mass_vector(cfg)
    keep = [e for e in elements if e in R.PAAS]
    A = np.array([[R.RESERVOIRS[r].get(e, 0.0) for r in names] for e in keep], dtype=float)
    phi = np.array([phi_all.get(e, 0.0) for e in keep], dtype=float)
    sig = np.array([max(default_sigma(e, cfg), float((sigma_floor or {}).get(e, 0.0)))
                    for e in keep], dtype=float)
    return Design(elements=keep, reservoirs=names, A=A, phi=phi, sigma=sig)


def _log_chi2(x, A, c, s):
    """chi2 in log space for amplitudes f = exp(x); returns (chi2, grad)."""
    f = np.exp(x)
    m = A @ f
    m = np.maximum(m, 1e-30)
    r = (np.log10(m) - np.log10(c)) / s
    chi2 = float((r * r).sum())
    # d chi2 / d x_i = sum_k 2 r_k / s_k * (1/ln10) * (A_ki f_i / m_k)
    g = (2.0 * r / s / LN10 / m) @ A * f
    return chi2, g


def _fit_one(c: np.ndarray, s: np.ndarray, A: np.ndarray, *, refine: bool) -> tuple[float, np.ndarray]:
    """Weighted NNLS in linear space (weights ~ 1/(sigma c ln10)), optional log refinement."""
    w = 1.0 / (s * LN10 * c)
    f, _ = nnls(A * w[:, None], c * w)
    m = np.maximum(A @ f, 1e-30)
    r = (np.log10(m) - np.log10(c)) / s
    chi2 = float((r * r).sum())
    if refine and A.shape[1] > 0:
        x0 = np.log(np.maximum(f, 1e-12))
        try:
            res = minimize(_log_chi2, x0, args=(A, c, s), jac=True, method="L-BFGS-B",
                           bounds=[(-40.0, 30.0)] * len(x0), options={"maxiter": 200})
            if res.fun < chi2:
                chi2, f = float(res.fun), np.exp(res.x)
        except Exception:                                  # noqa: BLE001
            pass
    return chi2, f


def fit_mixture(conc: np.ndarray, D: Design, cfg: GraveConfig | None = None, *,
                sigma: np.ndarray | None = None, refine: bool = True) -> pd.DataFrame:
    """Fit every row of ``conc`` (N x K ppm, NaN = unmeasured) natural-only and natural+fission.

    Returns per row: ``n_measured``, ``chi2_natural``, ``chi2_fission``,
    ``fission_lr`` (= 0.5 * (chi2_natural - chi2_fission), >= 0 by
    construction since the fission column is optional), ``a_fission`` (anchor
    doublings), ``reduced_chi2_natural`` and ``reduced_chi2_fission``, the
    dominant natural reservoir by *mass* share in the natural fit and its
    share, and ``f_<reservoir>`` columns (natural fit).
    """
    cfg = cfg or GraveConfig()
    conc = np.asarray(conc, dtype=float)
    n, k = conc.shape
    S = np.broadcast_to(D.sigma if sigma is None else np.asarray(sigma, dtype=float), (n, k))
    M = len(D.reservoirs)
    Afull = np.column_stack([D.A, D.phi])
    out = {"n_measured": np.zeros(n, int), "chi2_natural": np.full(n, np.nan),
           "chi2_fission": np.full(n, np.nan), "a_fission": np.zeros(n),
           "dominant_reservoir": np.full(n, "", dtype=object), "dominant_share": np.full(n, np.nan)}
    F = np.zeros((n, M))
    mass_w = D.A.sum(axis=0)                        # total ppm per reservoir column (mass weight)
    for i in range(n):
        c = conc[i]
        ok = np.isfinite(c) & (c > 0)
        nm = int(ok.sum())
        out["n_measured"][i] = nm
        if nm < cfg.min_elements:
            continue
        s = S[i][ok]
        chi_n, f_n = _fit_one(c[ok], s, D.A[ok], refine=False)
        chi_f, f_f = _fit_one(c[ok], s, Afull[ok], refine=False)
        if refine and 0.5 * (chi_n - chi_f) >= cfg.refine_lr:
            chi_n, f_n = _fit_one(c[ok], s, D.A[ok], refine=True)
            chi_f, f_f = _fit_one(c[ok], s, Afull[ok], refine=True)
        if chi_f > chi_n:                            # nested: fission can never fit worse
            chi_f, f_f = chi_n, np.append(f_n, 0.0)
        out["chi2_natural"][i] = chi_n
        out["chi2_fission"][i] = chi_f
        out["a_fission"][i] = float(f_f[-1])
        F[i] = f_n
        share = f_n * mass_w
        if share.sum() > 0:
            j = int(share.argmax())
            out["dominant_reservoir"][i] = D.reservoirs[j]
            out["dominant_share"][i] = float(share[j] / share.sum())
    df = pd.DataFrame(out)
    df["fission_lr"] = 0.5 * (df["chi2_natural"] - df["chi2_fission"])
    dof_n = np.maximum(df["n_measured"] - M, 1)
    df["reduced_chi2_natural"] = df["chi2_natural"] / dof_n
    df["reduced_chi2_fission"] = df["chi2_fission"] / np.maximum(df["n_measured"] - M - 1, 1)
    df["fission_nd_added_ppm"] = df["a_fission"] * R.PAAS[cfg.anchor]
    for j, r in enumerate(D.reservoirs):
        df[f"f_{r}"] = F[:, j]
    return df


def natural_model(conc_row: np.ndarray, D: Design, s: np.ndarray, *, refine: bool = True
                  ) -> tuple[np.ndarray, np.ndarray]:
    """(model ppm over all K elements, residual dex where measured) for one row."""
    ok = np.isfinite(conc_row) & (conc_row > 0)
    _, f = _fit_one(conc_row[ok], s[ok], D.A[ok], refine=refine)
    m = np.maximum(D.A @ f, 1e-30)
    resid = np.full(len(conc_row), np.nan)
    resid[ok] = np.log10(conc_row[ok] / m[ok])
    return m, resid


def leave_one_out(conc_row: np.ndarray, D: Design, cfg: GraveConfig, s: np.ndarray) -> dict:
    """``fission_lr`` with each measured element dropped in turn."""
    ok = np.isfinite(conc_row) & (conc_row > 0)
    Afull = np.column_stack([D.A, D.phi])
    out = {}
    for k in np.flatnonzero(ok):
        m = ok.copy()
        m[k] = False
        if m.sum() < cfg.min_elements:
            continue
        chi_n, _ = _fit_one(conc_row[m], s[m], D.A[m], refine=True)
        chi_f, _ = _fit_one(conc_row[m], s[m], Afull[m], refine=True)
        out[D.elements[k]] = 0.5 * (chi_n - min(chi_f, chi_n))
    if not out:
        return {"lr_loo_min": float("nan"), "lr_loo_driver": "", "lr_without": {}}
    drv = min(out, key=out.get)
    return {"lr_loo_min": float(out[drv]), "lr_loo_driver": drv,
            "lr_without": {e: round(float(v), 3) for e, v in out.items()}}


def peak_coherence(resid: np.ndarray, D: Design, cfg: GraveConfig, s: np.ndarray) -> dict:
    """How many heavy-set and light-set elements sit >= peak_sigma above the natural model."""
    z = resid / s
    idx = {e: i for i, e in enumerate(D.elements)}
    heavy = [e for e in cfg.heavy_set if e in idx and np.isfinite(z[idx[e]]) and z[idx[e]] >= cfg.peak_sigma]
    light = [e for e in cfg.light_set if e in idx and np.isfinite(z[idx[e]]) and z[idx[e]] >= cfg.peak_sigma]
    n_heavy_meas = sum(1 for e in cfg.heavy_set if e in idx and np.isfinite(z[idx[e]]))
    n_light_meas = sum(1 for e in cfg.light_set if e in idx and np.isfinite(z[idx[e]]))
    return {"heavy_up": heavy, "light_up": light, "n_heavy_measured": n_heavy_meas,
            "n_light_measured": n_light_meas,
            "coherent": len(heavy) >= cfg.n_heavy and len(light) >= cfg.n_light}


# ---------------------------------------------------------------------------
# Error floors and the shuffled null
# ---------------------------------------------------------------------------
def error_floors(conc: np.ndarray, D: Design, cfg: GraveConfig, *, max_rows: int = 4000,
                 seed: int = 0) -> dict[str, dict]:
    """Per-element robust scatter of log(obs / natural model) on a control population.

    The FALLOUT lesson (docs/fallout.md 4a): quoted errors are not the scatter.
    Here there are no quoted errors at all, so the floor *is* the error model:
    1.4826 x MAD of the natural-fit residual per element, over up to
    ``max_rows`` control samples.  Returns ``{el: {floor_dex, n, default_dex}}``.
    """
    rng = np.random.default_rng(seed)
    n = conc.shape[0]
    pick = np.arange(n) if n <= max_rows else rng.choice(n, max_rows, replace=False)
    res = np.full((len(pick), len(D.elements)), np.nan)
    for r, i in enumerate(pick):
        c = conc[i]
        if (np.isfinite(c) & (c > 0)).sum() < cfg.min_elements:
            continue
        _, res[r] = natural_model(c, D, D.sigma, refine=False)
    out = {}
    for k, el in enumerate(D.elements):
        v = res[:, k]
        v = v[np.isfinite(v)]
        if v.size >= 30:
            mad = 1.4826 * float(np.median(np.abs(v - np.median(v))))
        else:
            mad = float("nan")
        out[el] = {"floor_dex": round(mad, 4) if np.isfinite(mad) else None, "n": int(v.size),
                   "default_dex": round(float(D.sigma[k]), 4)}
    return out


def apply_floors(D: Design, floors: dict[str, dict], *, unmeasured_dex: float = 0.0) -> Design:
    """Widen each element's sigma to its measured scatter.

    An element whose floor could **not** be measured (too few control
    residuals -- which is the case for exactly the rare, decisive elements:
    Ru, Rh, Pd, Te, Ir) is given ``unmeasured_dex`` instead of its nominal
    class sigma, because an unmeasured scatter is not a small scatter.
    """
    sig = []
    for k, e in enumerate(D.elements):
        f = floors.get(e) or {}
        measured = f.get("floor_dex")
        floor = float(measured) if measured is not None else float(unmeasured_dex)
        sig.append(max(float(D.sigma[k]), floor))
    return Design(elements=D.elements, reservoirs=D.reservoirs, A=D.A, phi=D.phi,
                  sigma=np.array(sig, dtype=float))


def shuffled_null(conc: np.ndarray, D: Design, cfg: GraveConfig, *, max_rows: int = 2000,
                  seed: int = 1) -> dict:
    """``fission_lr`` on samples whose PAAS-normalised trace enrichments are permuted.

    Each sample's ``log10(c_X / PAAS_X)`` values over the *trace* elements are
    shuffled among the trace slots (majors are left in place so the reservoir
    fractions stay physical) and re-fitted.  Keeps the per-sample amplitude
    structure and the element count; destroys the alignment with any real
    pattern.  Reports the quantile the threshold is compared with.
    """
    rng = np.random.default_rng(seed)
    n = conc.shape[0]
    pick = np.arange(n) if n <= max_rows else rng.choice(n, max_rows, replace=False)
    paas = np.array([R.PAAS[e] for e in D.elements])
    trace = np.array([e not in R.MAJORS for e in D.elements])
    sh = conc[pick].copy()
    for i in range(len(pick)):
        row = sh[i]
        ok = np.isfinite(row) & (row > 0) & trace
        if ok.sum() < 2:
            continue
        ef = np.log10(row[ok] / paas[ok])
        rng.shuffle(ef)
        row[ok] = paas[ok] * 10.0 ** ef
    fit = fit_mixture(sh, D, cfg, refine=True)
    lr = fit["fission_lr"].to_numpy()
    lr = lr[np.isfinite(lr)]
    if lr.size == 0:
        return {"n": 0, "quantile": cfg.null_quantile, "lr_quantile": None, "frac_above_lr_min": None}
    return {"n": int(lr.size), "quantile": cfg.null_quantile,
            "lr_quantile": round(float(np.quantile(lr, cfg.null_quantile)), 3),
            "lr_max": round(float(lr.max()), 3),
            "frac_above_lr_min": round(float((lr >= cfg.lr_min).mean()), 6)}


# ---------------------------------------------------------------------------
# Enrichment factors, kills, refined-particulate classes
# ---------------------------------------------------------------------------
def enrichment_factor(row: dict, el: str, ref: dict | None = None) -> float:
    """(X/Al)_sample / (X/Al)_ref; NaN when X or Al is missing."""
    ref = ref or R.PAAS
    x, al = row.get(el), row.get("Al")
    if x is None or al is None or not (np.isfinite(x) and np.isfinite(al)) or x <= 0 or al <= 0:
        return float("nan")
    return float((x / al) / (ref[el] / ref["Al"]))


def redox_state(row: dict, cfg: GraveConfig) -> dict:
    ef_u, ef_v, ef_mo = (enrichment_factor(row, e) for e in ("U", "V", "Mo"))
    toc = row.get("TOC")
    anoxic = (np.isfinite(ef_u) and ef_u >= cfg.redox_ef_u) or (np.isfinite(ef_v) and ef_v >= cfg.redox_ef_v) \
        or (toc is not None and np.isfinite(toc) and toc >= cfg.redox_toc_pct)
    return {"ef_u": ef_u, "ef_v": ef_v, "ef_mo": ef_mo, "toc": toc, "anoxic": bool(anoxic)}


def classify_pge(row: dict, cfg: GraveConfig) -> dict:
    """Refined-particulate test on the PGE panel.

    Enrichment factors are against UCC (Al-normalised when Al exists, plain
    ratio otherwise -- PGE are so far below crustal that dilution is a minor
    correction).  Ratios to Ir are compared with CI.
    """
    ef = {}
    for e in R.PGE:
        v = row.get(e)
        if v is None or not np.isfinite(v) or v <= 0:
            continue
        al = row.get("Al")
        if al is not None and np.isfinite(al) and al > 0:
            ef[e] = float((v / al) / (R.UCC[e] / R.UCC["Al"]))
        else:
            ef[e] = float(v / R.UCC[e])
    ratios = {}
    ir = row.get("Ir")
    if ir is not None and np.isfinite(ir) and ir > 0:
        for e in ("Os", "Ru", "Rh", "Pt", "Pd"):
            v = row.get(e)
            if v is not None and np.isfinite(v) and v > 0:
                ratios[f"{e}/Ir_vs_CI"] = float((v / ir) / R.CI_PGE_TO_IR[e])
    out = {"pge_measured": sorted(ef), "pge_ef": {k: round(v, 3) for k, v in ef.items()},
           "pge_ratios_vs_ci": {k: round(v, 3) for k, v in ratios.items()}}
    if len(ef) < 2:
        out["pge_class"] = PGE_INSUFFICIENT
        return out
    anom = {e for e, v in ef.items() if v >= cfg.pge_ef_anom}
    if not anom:
        out["pge_class"] = PGE_BACKGROUND
        return out
    tol = cfg.pge_ratio_tol
    ir_up = "Ir" in anom or ("Os" in anom and "Ir" not in ef)
    ppge_up = anom & {"Pt", "Pd", "Rh"}
    if ir_up:
        # Ir-anchored.  Impact: every measured ratio chondritic, the PPGE (Pt,
        # Pd) included.  Ultramafic detritus: the IPGE (Os, Ru, Rh) chondritic
        # but Pt/Ir and Pd/Ir well below -- chromite / olivine hold the IPGE.
        ppge = {k: v for k, v in ratios.items() if k.startswith(("Pt", "Pd"))}
        chond_all = bool(ratios) and all(1.0 / tol <= v <= tol for v in ratios.values())
        ppge_low = bool(ppge) and all(v < cfg.ipge_pd_ir_max for v in ppge.values())
        if not ratios:
            out["pge_class"] = PGE_UNCLASSIFIED      # Ir (or Os) alone: no ratio to test
        elif ppge_low:
            out["pge_class"] = PGE_ULTRAMAFIC
        elif chond_all:
            out["pge_class"] = PGE_IMPACT
        else:
            out["pge_class"] = PGE_UNCLASSIFIED
        return out
    # Ir (and Os) at background: what carries the anomaly?
    ir_known = "Ir" in ef or "Os" in ef
    if not ir_known:
        out["pge_class"] = PGE_UNCLASSIFIED      # a Pt/Pd-only panel cannot exclude impact
        return out
    if "Pt" in anom and (anom - {"Pt"} <= {"Pd", "Rh"}):
        # Pt with Pd/Rh, no Ru: catalyst / refined; Pt alone with Mn up: Fe-Mn oxide
        if enrichment_factor(row, "Mn") >= cfg.femn_ef_mn and anom == {"Pt"}:
            out["pge_class"] = PGE_FEMN
        else:
            out["pge_class"] = PGE_REFINED
        return out
    if "Pt" not in anom and anom & {"Ru", "Rh", "Pd"}:
        out["pge_class"] = PGE_FISSION_LIKE
        return out
    out["pge_class"] = PGE_REFINED if ppge_up else PGE_UNCLASSIFIED
    return out


def classify_alloy(row: dict, cfg: GraveConfig) -> dict:
    """Ta without Nb; W without its hydrothermal partners (Sn, Mo, Bi)."""
    out = {"alloy_class": ALLOY_NONE, "alloy_ratios": {}}
    ta, nb, w = row.get("Ta"), row.get("Nb"), row.get("W")
    ef_ta, ef_w = enrichment_factor(row, "Ta"), enrichment_factor(row, "W")
    flags = []
    if ta is not None and np.isfinite(ta) and ta > 0 and np.isfinite(ef_ta) and ef_ta >= cfg.ta_ef_anom:
        if nb is not None and np.isfinite(nb) and nb > 0:
            r = float(nb / ta)
            out["alloy_ratios"]["Nb/Ta"] = round(r, 3)
            flags.append(ALLOY_REFINED_TA if r < cfg.nb_ta_natural[0] else ALLOY_NATURAL_TA)
        else:
            flags.append(ALLOY_INSUFFICIENT)
    if w is not None and np.isfinite(w) and w > 0 and np.isfinite(ef_w) and ef_w >= cfg.w_ef_anom:
        partners = {e: enrichment_factor(row, e) for e in ("Sn", "Mo", "Bi")}
        known = {e: v for e, v in partners.items() if np.isfinite(v)}
        out["alloy_ratios"].update({f"ef_{e}": round(v, 3) for e, v in known.items()})
        if not known:
            flags.append(ALLOY_INSUFFICIENT)
        elif all(v < cfg.w_partner_ef_max for v in known.values()):
            flags.append(ALLOY_REFINED_W)
        else:
            flags.append(ALLOY_NATURAL_W)
    out["alloy_ef"] = {"Ta": round(ef_ta, 3) if np.isfinite(ef_ta) else None,
                       "W": round(ef_w, 3) if np.isfinite(ef_w) else None}
    if flags:
        pref = [f for f in flags if f.startswith("refined")]
        out["alloy_class"] = pref[0] if pref else flags[0]
    return out


def detection_limit_mask(conc: pd.DataFrame, cfg: GraveConfig) -> tuple[pd.DataFrame, dict]:
    """Flag values that are a column's repeated floor (a reporting limit, not a measurement).

    Non-positive values are NaN (censored or absent).  A value that repeats
    at least ``dl_repeat_min`` times and lies within ``dl_floor_fraction`` of
    the column's positive minimum is treated as a detection limit: the value
    is kept (as a limit-level number) but the mask marks it, and a candidate
    whose fission preference rests on a masked element is vetoed.
    """
    mask = pd.DataFrame(False, index=conc.index, columns=conc.columns)
    ledger = {}
    for col in conc.columns:
        v = pd.to_numeric(conc[col], errors="coerce")
        pos = v[v > 0]
        if pos.empty:
            continue
        lo = float(pos.min())
        vc = pos.value_counts()
        dl = [float(x) for x, n in vc.items() if n >= cfg.dl_repeat_min and x <= lo * (1 + cfg.dl_floor_fraction)]
        if dl:
            m = v.isin(dl)
            mask[col] = m
            ledger[col] = {"values": sorted(dl)[:5], "n_flagged": int(m.sum())}
    return mask, ledger


# ---------------------------------------------------------------------------
# The per-sample verdict
# ---------------------------------------------------------------------------
def assess_row(row: dict, conc_row: np.ndarray, D: Design, cfg: GraveConfig, s: np.ndarray, *,
               fit: dict, threshold: float, dl_flags: set[str] | None = None) -> dict:
    """Run the kills on one sample that cleared the LR threshold.

    ``fit`` is that sample's row of :func:`fit_mixture`; the result carries
    the class, the first veto hit, every veto hit, the residual vector, the
    leave-one-out table, the peak coherence, the redox state and the two
    refined-particulate classes.
    """
    vetoes: list[str] = []
    lr = float(fit["fission_lr"])
    m, resid = natural_model(conc_row, D, s)
    loo = leave_one_out(conc_row, D, cfg, s)
    coh = peak_coherence(resid, D, cfg, s)
    red = redox_state(row, cfg)
    pge = classify_pge(row, cfg)
    alloy = classify_alloy(row, cfg)
    driver = loo.get("lr_loo_driver", "")

    if int(fit["n_measured"]) < cfg.min_elements:
        vetoes.append("insufficient_panel")
    if dl_flags and driver in dl_flags:
        vetoes.append("detection_limit_driver")
    if float(fit["reduced_chi2_fission"]) > cfg.max_reduced_chi2:
        vetoes.append("unexplained_by_all_reservoirs")
    if not (np.isfinite(loo["lr_loo_min"]) and loo["lr_loo_min"] >= threshold):
        vetoes.append("single_element_driver")
    if not coh["coherent"]:
        vetoes.append("peak_incoherent")
    if red["anoxic"] and driver in ("Mo", "U", "V", "Cd", "Ag", "Re", "Se", "Tl"):
        vetoes.append("redox_conditioned")
    ef_mn = enrichment_factor(row, "Mn")
    if np.isfinite(ef_mn) and ef_mn >= cfg.femn_ef_mn and driver in ("Mo", "Te", "Pt", "Ce", "Co", "Ni"):
        vetoes.append("femn_shuttle")
    zr, hf = row.get("Zr"), row.get("Hf")
    if driver == "Zr" and zr and hf and np.isfinite(zr) and np.isfinite(hf) and hf > 0 \
            and cfg.zr_hf_natural[0] <= zr / hf <= cfg.zr_hf_natural[1]:
        vetoes.append("heavy_mineral_zr_hf")
    ef_th = enrichment_factor(row, "Th")
    if driver in ("La", "Ce", "Pr", "Nd", "Sm") and np.isfinite(ef_th) and ef_th >= cfg.monazite_th_ef:
        vetoes.append("monazite_th")
    ef_ba = enrichment_factor(row, "Ba")
    if driver == "Ba" and np.isfinite(ef_ba) and ef_ba >= cfg.hydro_ef_ba:
        vetoes.append("hydrothermal_ba")
    if pge["pge_class"] in (PGE_IMPACT, PGE_ULTRAMAFIC) and driver in R.PGE:
        vetoes.append("impact_pge")

    if vetoes:
        cls = UNEXPLAINED if "unexplained_by_all_reservoirs" in vetoes else NATURAL
    else:
        cls = FISSION_CANDIDATE if lr >= threshold else FISSION_AMBIGUOUS
    return {
        "class": cls, "first_veto": vetoes[0] if vetoes else "", "vetoes": vetoes,
        "fission_lr": round(lr, 3), "lr_loo_min": round(float(loo["lr_loo_min"]), 3)
        if np.isfinite(loo["lr_loo_min"]) else None, "lr_loo_driver": driver,
        "lr_without": loo["lr_without"],
        "residual_dex": {e: round(float(resid[k]), 3) for k, e in enumerate(D.elements) if np.isfinite(resid[k])},
        "peak": coh, "redox": {k: (round(v, 3) if isinstance(v, float) and np.isfinite(v) else v)
                               for k, v in red.items()},
        "pge": pge, "alloy": alloy,
        "a_fission": round(float(fit["a_fission"]), 4),
        "fission_nd_added_ppm": round(float(fit["fission_nd_added_ppm"]), 3),
        "reduced_chi2_fission": round(float(fit["reduced_chi2_fission"]), 3),
        "dominant_reservoir": fit["dominant_reservoir"],
    }


__all__ = [
    "ALLOY_INSUFFICIENT", "ALLOY_NATURAL_TA", "ALLOY_NATURAL_W", "ALLOY_NONE", "ALLOY_REFINED_TA",
    "ALLOY_REFINED_W", "Design", "FISSION_AMBIGUOUS", "FISSION_CANDIDATE", "GraveConfig",
    "INSUFFICIENT", "NATURAL", "NORMAL", "PGE_BACKGROUND", "PGE_FEMN", "PGE_FISSION_LIKE",
    "PGE_IMPACT", "PGE_INSUFFICIENT", "PGE_REFINED", "PGE_ULTRAMAFIC", "PGE_UNCLASSIFIED",
    "UNEXPLAINED", "VETOES", "apply_floors", "assess_row", "build_design", "classify_alloy",
    "classify_pge", "default_sigma", "detection_limit_mask", "enrichment_factor",
    "error_floors", "fission_discriminants", "fission_mass_vector", "fit_mixture",
    "leave_one_out", "natural_model", "peak_coherence", "redox_state", "shuffled_null",
]
