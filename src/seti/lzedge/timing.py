"""Annual modulation at the kinematic edge and the likelihood of one event's date.

For a model with window rate R(t) (events per tonne-year at date t) and an
experiment with live-time density L(t), the probability density of the date of
a single signal event is p_S(t) = R(t) L(t) / ∫ R L dt; a background event is
uniform in live time, p_B(t) = L(t) / ∫ L dt.  Their ratio at the observed date,

    Λ_t = R(t_obs) / <R>_L,

is the evidence the *date alone* carries for the model against a constant-rate
background.  It is 1 for an unmodulated signal, grows with the modulation
amplitude, and can be zero: a model whose rate vanishes on t_obs is excluded by
the calendar whatever its cross section.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from .earth import date_grid, to_datetime, year_grid
from .rate import Efficiency, smeared_spectrum, window_rate


@dataclass
class LiveTime:
    """Live-time density as a list of (start, end, weight) intervals (weight = live fraction)."""
    intervals: list[tuple[dt.datetime, dt.datetime, float]]

    @classmethod
    def uniform(cls, start, end, live_days: float | None = None) -> LiveTime:
        a, b = to_datetime(start), to_datetime(end)
        span = (b - a).total_seconds() / 86400.0
        w = 1.0 if live_days is None else live_days / span
        return cls([(a, b, w)])

    @classmethod
    def from_rows(cls, rows) -> LiveTime:
        return cls([(to_datetime(s), to_datetime(e), float(w)) for s, e, w in rows])

    @property
    def start(self) -> dt.datetime:
        return min(s for s, _, _ in self.intervals)

    @property
    def end(self) -> dt.datetime:
        return max(e for _, e, _ in self.intervals)

    def density(self, t) -> float:
        d = to_datetime(t)
        return float(sum(w for s, e, w in self.intervals if s <= d < e))

    def live_days(self) -> float:
        return float(sum(w * (e - s).total_seconds() / 86400.0 for s, e, w in self.intervals))

    def grid(self, step_days: float = 2.0) -> tuple[list[dt.datetime], np.ndarray]:
        dates = date_grid(self.start, self.end, step_days)
        w = np.array([self.density(t) for t in dates])
        return dates, w


@dataclass
class EventModel:
    """A halo + particle model evaluated against one detector window."""
    halo: object
    m_chi_gev: float
    delta_keV: float
    efficiency: Efficiency
    v0_kms: float = 238.0
    sigma_n_cm2: float = 1e-45
    rho_gev_cm3: float = 0.3

    def eta_at(self, t):
        return self.halo.eta_at(t, self.v0_kms)

    def rate(self, t) -> float:
        return window_rate(self.eta_at(t), self.m_chi_gev, self.delta_keV, self.efficiency,
                           self.sigma_n_cm2, self.rho_gev_cm3)

    def rate_curve(self, dates) -> np.ndarray:
        return np.array([self.rate(t) for t in dates])

    def observed_energy_density(self, E_obs, t, sigma_fn) -> np.ndarray:
        return smeared_spectrum(E_obs, self.eta_at(t), self.m_chi_gev, self.delta_keV,
                                self.efficiency, sigma_fn, self.sigma_n_cm2, self.rho_gev_cm3)


def timing_bayes_factor(model: EventModel, livetime: LiveTime, t_obs, step_days: float = 2.0) -> dict:
    """Λ_t = R(t_obs) / <R>_L, plus the pieces."""
    dates, w = livetime.grid(step_days)
    r = model.rate_curve(dates)
    mean_rate = float(np.sum(r * w) / np.sum(w)) if np.sum(w) > 0 else 0.0
    r_obs = model.rate(t_obs)
    lam = r_obs / mean_rate if mean_rate > 0 else (np.inf if r_obs > 0 else 0.0)
    expected = mean_rate * livetime.live_days() / 365.25  # per tonne, for sigma_n as set
    return {"rate_at_event": r_obs, "mean_rate_livetime": mean_rate, "bayes_factor_timing": lam,
            "expected_per_tonne": expected, "n_grid": len(dates)}


def modulation_summary(model: EventModel, year: int = 2023, n: int = 73) -> dict:
    """Peak / trough dates, fractional modulation and the on-fraction of the year."""
    dates = year_grid(year, n)
    r = model.rate_curve(dates)
    rmax, rmin = float(r.max()), float(r.min())
    on = r > 1e-3 * rmax if rmax > 0 else np.zeros_like(r, dtype=bool)
    return {
        "peak_date": dates[int(np.argmax(r))].date().isoformat() if rmax > 0 else None,
        "trough_date": dates[int(np.argmin(r))].date().isoformat() if rmax > 0 else None,
        "rate_max": rmax, "rate_min": rmin,
        "fractional_modulation": (rmax - rmin) / (rmax + rmin) if rmax + rmin > 0 else None,
        "fraction_of_year_on": float(np.mean(on)),
        "curve": [(d.date().isoformat(), float(x)) for d, x in zip(dates, r, strict=True)],
    }


def energy_likelihood_ratio(model_a: EventModel, model_b: EventModel, E_obs: float, t_obs,
                            sigma_fn) -> float:
    """p_A(E_obs | t_obs) / p_B(E_obs | t_obs), each normalised to its own window rate."""
    ra, rb = model_a.rate(t_obs), model_b.rate(t_obs)
    if ra <= 0 or rb <= 0:
        return 0.0 if ra <= 0 else np.inf
    pa = float(model_a.observed_energy_density(E_obs, t_obs, sigma_fn)[0]) / ra
    pb = float(model_b.observed_energy_density(E_obs, t_obs, sigma_fn)[0]) / rb
    return pa / pb if pb > 0 else np.inf


def date_percentile_under_model(model: EventModel, livetime: LiveTime, t_obs,
                                step_days: float = 2.0) -> float:
    """P(rate density at a random signal date <= rate density at t_obs): a tail probability."""
    dates, w = livetime.grid(step_days)
    r = model.rate_curve(dates)
    p = r * w
    if p.sum() <= 0:
        return float("nan")
    p /= p.sum()
    r_obs = model.rate(t_obs)
    return float(np.sum(p[r <= r_obs]))


def seasonal_background_factor(t, amplitude: float = 0.02, peak_month_day=(7, 5)) -> float:
    """1 + A cos(2π (t - t_peak)/yr): the underground muon flux's seasonal
    modulation (per-cent amplitude, early-July maximum at northern mid-latitude
    sites).  The timing factor a muon-induced background could claim."""
    d = to_datetime(t)
    peak = dt.datetime(d.year, peak_month_day[0], peak_month_day[1], tzinfo=dt.timezone.utc)
    phase = 2.0 * np.pi * (d - peak).total_seconds() / (365.25 * 86400.0)
    return float(1.0 + amplitude * np.cos(phase))


def energy_density_marginal_sys(model: EventModel, E_obs: float, t, sigma_fn, sigma_sys_keV: float,
                                n: int = 9) -> float:
    """p(E_obs | t) with the energy-scale systematic marginalised as a Gaussian
    shift of the observed energy (Gauss–Hermite quadrature, ``n`` nodes)."""
    if sigma_sys_keV <= 0:
        return float(model.observed_energy_density(E_obs, t, sigma_fn)[0])
    x, w = np.polynomial.hermite_e.hermegauss(n)
    shifts = sigma_sys_keV * x
    dens = model.observed_energy_density(E_obs + shifts, t, sigma_fn)
    return float(np.sum(w * dens) / np.sum(w))


def profile_likelihood(model: EventModel, livetime: LiveTime, E_obs: float, t_obs, sigma_fn,
                       sigma_sys_keV: float = 0.0, step_days: float = 3.0) -> dict:
    """The single-event extended likelihood with the cross section profiled out.

    L = e^{-mu} mu p(E, t | model); the profile over mu (i.e. over sigma_n)
    sits at mu = 1, so up to a constant L_prof = [R(t_obs)/<R>_L] x p(E_obs | t_obs).
    Returns the two factors and their product, or zeros when the model gives
    no rate on the event date."""
    tb = timing_bayes_factor(model, livetime, t_obs, step_days)
    r_obs = tb["rate_at_event"]
    if r_obs <= 0 or tb["mean_rate_livetime"] <= 0:
        return {"timing": 0.0, "energy": 0.0, "profile": 0.0, **tb}
    pE = energy_density_marginal_sys(model, E_obs, t_obs, sigma_fn, sigma_sys_keV) / r_obs
    return {"timing": tb["bayes_factor_timing"], "energy": pE,
            "profile": tb["bayes_factor_timing"] * pE, **tb}
