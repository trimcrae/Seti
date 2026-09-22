"""Tier 1: is this abundance vector reachable by any natural parcel?

One panel (one white dwarf, one literature source) is a set of measured
log(Z/X) values with errors, plus upper limits.  The natural model is

    log(Z/X)_phot = log parcel_Z(w, T_cut, d)  +  phase_Z(tau_rel, t_acc, t_dec)  +  c

with ``w`` on the end-member simplex (``family.py``), condensation
fractionation ``(T_cut, d)``, the sinking phase ``(t_acc, t_dec)``
(``sinking.py``) and a free normalisation ``c`` (the reference element H or
He and the total accreted mass cancel).  The objective is the Gaussian
−2 ln L, i.e. sum of (residual/sigma_i)^2 + ln sigma_i^2, with
sigma_i^2 = sigma_meas^2 + sigma_sys^2 + (g_i sigma_tau)^2 where g_i is the
analytic sensitivity of the model to that element's timescale in the fitted
phase.  The ln sigma^2 term is not optional: with a bare chi-square the
optimiser "explains" any anomaly by choosing a deep declining phase whose
timescale sensitivity inflates the error bar.  Upper limits enter one-sided.

The misfit PROBABILITY is calibrated, not read off a chi-square table: the
best natural model is perturbed by the panel's own errors N_cal times, each
draw is refitted with exactly the same freedom, and p is the fraction of
draws whose minimum objective is at least the observed one.  A panel with a
small p is one that the natural family, with all its freedom, reproduces
worse than it reproduces its own draws --- the "unexplainable" list.

Every fit also exports its *acceptable phase set*: the (t_acc, t_dec) of every
parameter point within ``phase_delta`` of the optimum.  Tier 2 uses the set
from the REST of a panel to bound a pair's sinking shift, because the phase
is one number per object, not one per element.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .family import NaturalFamily, fractionation_factor, softmax
from .sinking import TimescaleModel, phase_label, phase_log_factor

_LN10 = np.log(10.0)


@dataclass
class Panel:
    """One white dwarf's abundance panel from one source."""

    name: str
    elements: list[str]
    values: np.ndarray
    errors: np.ndarray
    atmosphere: str = "He"
    teff: float = np.nan
    logg: float = np.nan
    reference: str = ""
    limit_elements: list[str] = field(default_factory=list)
    limit_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        self.errors = np.asarray(self.errors, dtype=float)
        self.limit_values = np.asarray(self.limit_values, dtype=float)

    @property
    def n_measured(self) -> int:
        return int(len(self.elements))

    def subset(self, drop: list[str]) -> Panel:
        keep = [i for i, e in enumerate(self.elements) if e not in drop]
        return Panel(name=self.name, elements=[self.elements[i] for i in keep],
                     values=self.values[keep], errors=self.errors[keep],
                     atmosphere=self.atmosphere, teff=self.teff, logg=self.logg,
                     reference=self.reference, limit_elements=list(self.limit_elements),
                     limit_values=np.array(self.limit_values, dtype=float), meta=dict(self.meta))


@dataclass
class FitSettings:
    sigma_sys: float = 0.15
    n_random: int = 400
    n_refine: int = 2
    refine_maxiter: int = 500
    t_cut_range: tuple = (0.0, 1800.0)
    depth_range: tuple = (-1.5, 4.0)
    t_acc_range: tuple = (0.01, 30.0)
    t_dec_range: tuple = (0.0, 3.0)
    fractionation_width_K: float = 100.0
    dirichlet_alpha: float = 0.3
    limit_sigma: float = 0.15
    phase_delta: float = 4.0
    seed: int = 12345

    def restricted(self, t_cut_max: float = 1400.0) -> FitSettings:
        """The same settings with the fractionation cut capped below the refractories."""
        d = dict(self.__dict__)
        d["t_cut_range"] = (float(self.t_cut_range[0]), float(t_cut_max))
        return FitSettings(**d)


@dataclass
class FitResult:
    nll2: float                   # the objective: sum res^2 + sum ln sigma^2
    chi2: float                   # sum res^2 at the optimum
    n_measured: int
    n_limits: int
    weights: np.ndarray
    t_cut: float
    depth: float
    t_acc: float
    t_dec: float
    norm: float
    model: np.ndarray             # fitted log abundances (with norm), measured elements
    residual_sigma: np.ndarray    # (obs - model) / sigma_eff
    sigma_eff: np.ndarray
    phase: str
    timescale_source: str
    limit_violation: float = 0.0
    dominant_endmember: str = ""
    dominant_weight: float = 0.0
    phase_set: list = field(default_factory=list)   # acceptable (t_acc, t_dec)

    def as_dict(self, endmembers: list[str]) -> dict:
        top = np.argsort(self.weights)[::-1][:3]
        ps = np.array(self.phase_set, dtype=float).reshape(-1, 2)
        return {
            "nll2": float(self.nll2), "chi2": float(self.chi2),
            "n_measured": int(self.n_measured), "n_limits": int(self.n_limits),
            "dof": int(max(self.n_measured - 1, 0)),
            "chi2_per_dof": float(self.chi2 / max(self.n_measured - 1, 1)),
            "t_cut_K": float(self.t_cut), "depth_dex": float(self.depth),
            "t_acc_over_tau_ref": float(self.t_acc), "t_dec_over_tau_ref": float(self.t_dec),
            "phase": self.phase, "timescale_source": self.timescale_source,
            "norm": float(self.norm), "limit_violation_chi2": float(self.limit_violation),
            "top_endmembers": {endmembers[i]: float(self.weights[i]) for i in top},
            "dominant_endmember": self.dominant_endmember,
            "dominant_weight": float(self.dominant_weight),
            "residual_sigma": [float(r) for r in self.residual_sigma],
            "max_abs_residual_sigma": float(np.max(np.abs(self.residual_sigma)))
            if len(self.residual_sigma) else 0.0,
            "n_acceptable_phases": int(len(ps)),
            "t_dec_acceptable_max": float(ps[:, 1].max()) if len(ps) else float(self.t_dec),
            "t_acc_acceptable_min": float(ps[:, 0].min()) if len(ps) else float(self.t_acc),
        }


# ---------------------------------------------------------------------------
# the model, vectorised over a batch of parameter sets
# ---------------------------------------------------------------------------
class PanelModel:
    """Everything about one panel's elements that does not depend on the parameters."""

    def __init__(self, fam: NaturalFamily, panel: Panel, tsm: TimescaleModel,
                 settings: FitSettings):
        self.fam, self.panel, self.s = fam, panel, settings
        self.elements = list(panel.elements)
        self.lim_elements = [e for e in panel.limit_elements if e in fam.elements]
        self.all_elements = self.elements + self.lim_elements
        self.X = fam.sub(self.all_elements)                         # (n_all, n_end)
        self.Tc = fam.tc(self.all_elements)
        lt, st, src = tsm.log_tau_rel(self.all_elements, panel.atmosphere, panel.teff,
                                      panel.logg, row_tau=panel.meta.get("sinking_times_s"))
        self.log_tau, self.sig_tau, self.timescale_source = lt, st, src
        self.n_meas = len(self.elements)
        self.y = np.asarray(panel.values, dtype=float)
        self.sig_meas2 = np.asarray(panel.errors, dtype=float) ** 2 + settings.sigma_sys ** 2
        keep = [i for i, e in enumerate(panel.limit_elements) if e in fam.elements]
        self.limits = np.asarray(panel.limit_values, dtype=float)[keep] if keep else np.zeros(0)
        self.n_end = self.X.shape[1]
        self.n_par = self.n_end + 4

    # parameter vector: [logits(n_end), t_cut, depth, log10 t_acc, t_dec]
    def unpack(self, theta: np.ndarray):
        theta = np.asarray(theta, dtype=float)
        w = softmax(theta[: self.n_end])
        t_cut = float(np.clip(theta[self.n_end], *self.s.t_cut_range))
        depth = float(np.clip(theta[self.n_end + 1], *self.s.depth_range))
        t_acc = float(10.0 ** np.clip(theta[self.n_end + 2], np.log10(self.s.t_acc_range[0]),
                                      np.log10(self.s.t_acc_range[1])))
        t_dec = float(np.clip(theta[self.n_end + 3], *self.s.t_dec_range))
        return w, t_cut, depth, t_acc, t_dec

    def predict(self, w, t_cut, depth, t_acc, t_dec) -> tuple[np.ndarray, np.ndarray]:
        """log parcel + phase term (no norm) for all elements, and the tau-sensitivity g."""
        x = np.maximum(self.X @ w, 1e-30)
        f = fractionation_factor(self.Tc, t_cut, depth, self.s.fractionation_width_K)
        ph = phase_log_factor(self.log_tau, t_acc, t_dec)
        tau = 10.0 ** self.log_tau
        u = max(t_acc, 1e-6) / tau
        g = 1.0 - u * np.exp(-u) / np.maximum(-np.expm1(-u), 1e-300) + t_dec / tau
        return np.log10(x) + np.log10(f) + ph, g

    def objective(self, theta: np.ndarray, *, y: np.ndarray | None = None
                  ) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
        """``(nll2, chi2, norm, model_with_norm, sigma_eff, residual_sigma)`` for one theta."""
        y = self.y if y is None else y
        w, t_cut, depth, t_acc, t_dec = self.unpack(theta)
        m, g = self.predict(w, t_cut, depth, t_acc, t_dec)
        sig2 = self.sig_meas2 + (g[: self.n_meas] * self.sig_tau[: self.n_meas]) ** 2
        wt = 1.0 / sig2
        c = float(np.sum(wt * (y - m[: self.n_meas])) / np.sum(wt))
        res = (y - m[: self.n_meas] - c) / np.sqrt(sig2)
        chi2 = float(np.sum(res ** 2))
        nll2 = chi2 + float(np.sum(np.log(sig2)))
        if len(self.limits):
            viol = m[self.n_meas:] + c - self.limits
            pen = float(np.sum(np.maximum(viol, 0.0) ** 2) / self.s.limit_sigma ** 2)
            chi2 += pen
            nll2 += pen
        return nll2, chi2, c, m[: self.n_meas] + c, np.sqrt(sig2), res

    def batch_objective(self, thetas: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """nll2 for a batch of thetas (rows) against ``y`` (1-D) or a batch of y (2-D)."""
        y = self.y if y is None else np.asarray(y, dtype=float)
        n = thetas.shape[0]
        logits = thetas[:, : self.n_end]
        z = logits - logits.max(axis=1, keepdims=True)
        W = np.exp(z)
        W /= W.sum(axis=1, keepdims=True)
        x = np.maximum(W @ self.X.T, 1e-30)                        # (n, n_all)
        t_cut = np.clip(thetas[:, self.n_end], *self.s.t_cut_range)
        depth = np.clip(thetas[:, self.n_end + 1], *self.s.depth_range)
        t_acc = 10.0 ** np.clip(thetas[:, self.n_end + 2], np.log10(self.s.t_acc_range[0]),
                                np.log10(self.s.t_acc_range[1]))
        t_dec = np.clip(thetas[:, self.n_end + 3], *self.s.t_dec_range)
        sfrac = 1.0 / (1.0 + np.exp((self.Tc[None, :] - t_cut[:, None])
                                    / self.s.fractionation_width_K))
        logf = -depth[:, None] * sfrac
        tau = 10.0 ** self.log_tau[None, :]
        u = np.maximum(t_acc[:, None], 1e-6) / tau
        build = np.maximum(-np.expm1(-u), 1e-300)
        ph = np.log10(tau) + np.log10(build) - t_dec[:, None] / tau / _LN10
        ph -= ph.mean(axis=1, keepdims=True)
        g = 1.0 - u * np.exp(-u) / build + t_dec[:, None] / tau
        m = np.log10(x) + logf + ph                                  # (n, n_all)
        mm = m[:, : self.n_meas]
        sig2 = self.sig_meas2[None, :] + (g[:, : self.n_meas] * self.sig_tau[None, : self.n_meas]) ** 2
        wt = 1.0 / sig2
        logdet = np.sum(np.log(sig2), axis=1)
        if y.ndim == 1:
            c = np.sum(wt * (y[None, :] - mm), axis=1) / np.sum(wt, axis=1)     # (n,)
            res2 = (y[None, :] - mm - c[:, None]) ** 2 / sig2
            out = res2.sum(axis=1) + logdet
            if len(self.limits):
                viol = m[:, self.n_meas:] + c[:, None] - self.limits[None, :]
                out = out + np.sum(np.maximum(viol, 0.0) ** 2, axis=1) / self.s.limit_sigma ** 2
            return out
        Y = y                                                        # (k, n_meas)
        num = wt @ Y.T - np.sum(wt * mm, axis=1)[:, None]            # (n, k)
        c = num / np.sum(wt, axis=1)[:, None]
        out = np.zeros((n, Y.shape[0]))
        for j in range(Y.shape[0]):
            res2 = (Y[j][None, :] - mm - c[:, j][:, None]) ** 2 / sig2
            out[:, j] = res2.sum(axis=1) + logdet
            if len(self.limits):
                viol = m[:, self.n_meas:] + c[:, j][:, None] - self.limits[None, :]
                out[:, j] += np.sum(np.maximum(viol, 0.0) ** 2, axis=1) / self.s.limit_sigma ** 2
        return out

    def random_thetas(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Random-search starting points: pure end-members, sparse and uniform mixtures."""
        s = self.s
        n_end = self.n_end
        n_rand = max(n - n_end, 0)
        logits = np.vstack([
            np.eye(n_end) * 12.0,
            np.log(rng.dirichlet(np.full(n_end, s.dirichlet_alpha), size=n_rand) + 1e-9),
        ])
        k = logits.shape[0]
        t_cut = rng.uniform(*s.t_cut_range, size=k)
        depth = rng.uniform(*s.depth_range, size=k)
        depth[rng.random(k) < 0.4] = 0.0
        lacc = rng.uniform(np.log10(s.t_acc_range[0]), np.log10(s.t_acc_range[1]), size=k)
        t_dec = rng.uniform(*s.t_dec_range, size=k)
        t_dec[rng.random(k) < 0.5] = 0.0
        return np.column_stack([logits, t_cut, depth, lacc, t_dec])


def fit_panel(fam: NaturalFamily, panel: Panel, tsm: TimescaleModel,
              settings: FitSettings | None = None, *, rng=None,
              y: np.ndarray | None = None, model: PanelModel | None = None) -> FitResult:
    """Minimum −2 ln L natural model of one panel (random search, then Nelder-Mead)."""
    s = settings or FitSettings()
    rng = np.random.default_rng(s.seed) if rng is None else rng
    pm = model or PanelModel(fam, panel, tsm, s)
    thetas = pm.random_thetas(rng, s.n_random)
    obj = pm.batch_objective(thetas, y)
    order = np.argsort(obj)[: max(int(s.n_refine), 1)]
    best_theta, best_obj = thetas[order[0]], float(obj[order[0]])
    refined = []
    for i in order:
        r = minimize(lambda th: pm.objective(th, y=y)[0], thetas[i], method="Nelder-Mead",
                     options={"maxiter": int(s.refine_maxiter), "xatol": 1e-3, "fatol": 1e-4,
                              "adaptive": True})
        refined.append((float(r.fun), np.asarray(r.x)))
        if float(r.fun) < best_obj:
            best_obj, best_theta = float(r.fun), np.asarray(r.x)
    nll2, chi2, c, m, sig, res = pm.objective(best_theta, y=y)
    w, t_cut, depth, t_acc, t_dec = pm.unpack(best_theta)
    top = int(np.argmax(w))
    lim_viol = 0.0
    if len(pm.limits):
        mfull, _ = pm.predict(w, t_cut, depth, t_acc, t_dec)
        viol = mfull[pm.n_meas:] + c - pm.limits
        lim_viol = float(np.sum(np.maximum(viol, 0.0) ** 2) / s.limit_sigma ** 2)
    # the acceptable phase set: every explored point within phase_delta of the optimum
    acc = [thetas[i] for i in np.flatnonzero(obj <= best_obj + s.phase_delta)]
    acc += [th for f, th in refined if f <= best_obj + s.phase_delta]
    acc.append(best_theta)
    phase_set = sorted({(round(pm.unpack(th)[3], 4), round(pm.unpack(th)[4], 4)) for th in acc})
    return FitResult(nll2=nll2, chi2=chi2, n_measured=pm.n_meas, n_limits=len(pm.limits),
                     weights=w, t_cut=t_cut, depth=depth, t_acc=t_acc, t_dec=t_dec, norm=c,
                     model=m, residual_sigma=res, sigma_eff=sig,
                     phase=phase_label(t_acc, t_dec), timescale_source=pm.timescale_source,
                     limit_violation=lim_viol, dominant_endmember=fam.endmembers[top],
                     dominant_weight=float(w[top]), phase_set=phase_set)


def calibrate_misfit(fam: NaturalFamily, panel: Panel, tsm: TimescaleModel, fit: FitResult,
                     settings: FitSettings | None = None, n_draws: int = 150, *,
                     rng=None, draw_mode: str = "posterior") -> dict:
    """Calibrated misfit probability from injected natural draws.

    ``posterior`` draws perturb the best natural model by the panel's
    effective errors (posterior-predictive); ``prior`` draws take a random
    natural parcel and phase instead (a harder null: any natural panel with
    these errors).  Each draw is refitted with the same freedom as the data,
    and p is the fraction of draws whose minimum objective is at least the
    data's.
    """
    s = settings or FitSettings()
    rng = np.random.default_rng(s.seed + 1) if rng is None else rng
    pm = PanelModel(fam, panel, tsm, s)
    n = int(n_draws)
    if draw_mode == "prior":
        thetas = pm.random_thetas(rng, n)
        base = np.zeros((n, pm.n_meas))
        for j in range(n):
            w, t_cut, depth, t_acc, t_dec = pm.unpack(thetas[j])
            m, _ = pm.predict(w, t_cut, depth, t_acc, t_dec)
            base[j] = m[: pm.n_meas] + fit.norm
    else:
        base = np.tile(fit.model, (n, 1))
    draws = base + rng.normal(size=(n, pm.n_meas)) * fit.sigma_eff[None, :]
    starts = pm.random_thetas(rng, s.n_random)
    obj_mat = pm.batch_objective(starts, draws)                               # (n_theta, n)
    obj_draw = np.zeros(n)
    for j in range(n):
        i0 = int(np.argmin(obj_mat[:, j]))
        r = minimize(lambda th, yy=draws[j]: pm.objective(th, y=yy)[0], starts[i0],
                     method="Nelder-Mead",
                     options={"maxiter": int(s.refine_maxiter), "xatol": 1e-3, "fatol": 1e-4,
                              "adaptive": True})
        obj_draw[j] = min(float(r.fun), float(obj_mat[i0, j]))
    p = (1.0 + float(np.sum(obj_draw >= fit.nll2))) / (n + 1.0)
    return {"p_misfit": float(p), "n_draws": n, "draw_mode": draw_mode,
            "nll2_obs": float(fit.nll2), "chi2_obs": float(fit.chi2),
            "nll2_draw_p50": float(np.median(obj_draw)),
            "nll2_draw_p95": float(np.percentile(obj_draw, 95)),
            "nll2_draw_max": float(np.max(obj_draw))}


def naive_p(chi2: float, dof: int) -> float:
    from scipy.stats import chi2 as _chi2
    if dof <= 0:
        return float("nan")
    return float(_chi2.sf(chi2, dof))


__all__ = ["FitResult", "FitSettings", "Panel", "PanelModel", "calibrate_misfit", "fit_panel",
           "naive_p"]
