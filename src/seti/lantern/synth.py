"""Synthetic JWST-like spectral time series for the offline tests and the
runner's injection self-check.  Deterministic given ``seed``."""

from __future__ import annotations

import numpy as np

from .phase import Ephemeris


def synthesise_timeseries(n_int: int = 300, n_wl: int = 600, wl_range=(2.9, 5.2),
                          period: float = 1.5, duration_h: float = 2.0,
                          window_h: float = 6.0, centre: str = "eclipse",
                          line_wl: float | None = 4.05, line_amp: float = 0.0,
                          line_vanishes: bool = True, line_sigma_samples: float = 0.9,
                          eclipse_depth: float = 0.002, transit_depth: float = 0.01,
                          noise: float = 2e-3, ramp_amp: float = 0.0, ramp_tau: float = 20.0,
                          line_ramp_amp: float = 0.0, centre_shift_h: float = 0.0,
                          cosmic_ray: tuple | None = None, rp_rs: float = 0.1,
                          seed: int = 1) -> dict:
    """Stellar continuum + noise (+ eclipse/transit) (+ optional planet line).

    ``centre`` = ``"eclipse"`` puts mid-eclipse at the middle of the window,
    ``"transit"`` mid-transit, ``"none"`` neither event.  A line with
    ``line_vanishes=True`` is zero while the planet is fully occulted and ramps
    through the contacts; a ``ramp_amp`` adds an exponential settling ramp to
    the whole spectrum and ``line_ramp_amp`` one to the line alone (a
    persistence-like artefact whose decay can mimic a drop when the eclipse
    sits late in the window -- shift it with ``centre_shift_h``);
    ``cosmic_ray=(integration, amplitude)`` adds a one-integration spike at the
    line wavelength.  Returns wavelength, flux (n_int, n_wl), flux_err, times
    (BJD_TDB) and the :class:`Ephemeris`.
    """
    rng = np.random.default_rng(seed)
    wl = np.linspace(*wl_range, n_wl)
    t0 = 2460000.0
    dur = duration_h / 24.0
    if centre == "eclipse":
        mid = t0 + 10 * period + 0.5 * period
    elif centre == "transit":
        mid = t0 + 10 * period
    else:
        mid = t0 + 10 * period + 0.25 * period
    times = mid + centre_shift_h / 24.0 + np.linspace(-0.5, 0.5, n_int) * window_h / 24.0
    # Smooth stellar continuum with a broad molecular band.
    # A broad band (CO2-like) plus a mild slope and a weak stellar line forest.
    cont = 1000.0 * (1.0 - 0.15 * np.exp(-0.5 * ((wl - 4.3) / 0.25) ** 2)) \
        * (1.0 - 0.02 * (wl - wl.mean()))
    forest = np.random.default_rng(99).uniform(size=n_wl) < 0.03
    depth = np.random.default_rng(98).uniform(0.005, 0.03, size=n_wl) * forest
    cont = cont * (1.0 - np.convolve(depth, np.exp(-0.5 * (np.arange(-3, 4) / 1.0) ** 2),
                                     mode="same"))
    phase_tr = ((times - t0) / period + 0.5) % 1.0 - 0.5
    phase_ec = ((times - t0) / period) % 1.0 - 0.5
    tau = dur * rp_rs
    vis = _visibility(phase_ec * period, dur, tau)          # 1 out of eclipse, 0 inside
    tr = _visibility(phase_tr * period, dur, tau)           # 1 out of transit, 0 inside
    flux = np.empty((n_int, n_wl))
    for i in range(n_int):
        scale = 1.0 - eclipse_depth * (1.0 - vis[i]) - transit_depth * (1.0 - tr[i])
        if ramp_amp:
            scale *= 1.0 + ramp_amp * np.exp(-i / ramp_tau)
        flux[i] = cont * scale
        if line_wl is not None and line_amp:
            j = np.argmin(np.abs(wl - line_wl))
            prof = np.exp(-0.5 * ((np.arange(n_wl) - j) / line_sigma_samples) ** 2)
            a = line_amp * (vis[i] if line_vanishes else scale)
            if line_ramp_amp:
                a *= 1.0 + line_ramp_amp * np.exp(-i / ramp_tau)
            flux[i] += a * cont[j] * prof
    if cosmic_ray is not None and line_wl is not None:
        k, amp = cosmic_ray
        j = np.argmin(np.abs(wl - line_wl))
        prof = np.exp(-0.5 * ((np.arange(n_wl) - j) / line_sigma_samples) ** 2)
        flux[int(k)] += amp * cont[j] * prof
    err = noise * cont[None, :] * np.ones((n_int, 1))
    flux = flux + rng.normal(0.0, 1.0, flux.shape) * err
    eph = Ephemeris(name="synth b", period=period, t0=t0, duration=dur,
                    period_err=1e-6, t0_err=1e-4, ecc=0.0, omega_deg=90.0, rp_rs=rp_rs)
    return {"wavelength": wl, "flux": flux, "flux_err": err, "times": times,
            "ephemeris": eph, "instrument": "NIRSPEC", "grating": "G395H"}


def _visibility(dt, dur, tau):
    """Planet visibility fraction vs time from mid-event: 1 outside T14, 0 inside T23."""
    a = np.abs(np.asarray(dt, float))
    half = 0.5 * dur
    v = np.ones_like(a)
    inside = a <= half - tau
    ramp = (a > half - tau) & (a <= half)
    v[inside] = 0.0
    v[ramp] = (a[ramp] - (half - tau)) / max(tau, 1e-9)
    return v


__all__ = ["synthesise_timeseries"]
