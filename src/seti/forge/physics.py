"""The physics of FORGE: excess models, the likelihood ratio, tiers.

Everything here is a pure function of numbers, so the offline tests can inject
a Planck swarm and a nano-grain disc and check what comes out.

Models
------
Every model is an excess *fraction* f(lambda) = F_excess / F_star, normalised
at a reference wavelength (K, 2.2 um).  The photosphere is a blackbody at Teff
(adequate from H to N for a main-sequence star at the few-percent level; the
cross-instrument calibration floor below is larger than the departure).

* grey body: f(l) = f_ref * [B(l, T) / B(l, Teff)] / [B(l_ref, T) / B(l_ref, Teff)]
  Two free parameters, T and f_ref.  A 1500 K body at 1 % in K gives ~7-8 %
  at 10 um for a solar-type star.
* nano-grain family: the same with an emissivity Q(l) = min(1, (2 pi a / l)^beta),
  a < 0.5 um, beta in {1, 2} -- the small-grain limit that makes the excess
  K-bright and N-faint (Kirchschlager et al. 2017).  Three free parameters
  (T, a, f_ref) at each beta.

Likelihood
----------
chi^2 over every measurement with a Gaussian error (a "null excess" of
0.5 +/- 0.3 % is a measurement, not a limit) plus a one-sided term for a
pure upper limit.  A per-band calibration floor is added in quadrature: the
K-vs-N comparison crosses instruments (FLUOR/PIONIER/JouFLU against LBTI/KIN)
and their absolute calibrations are not tied to each other.

The statistic per star is Delta = chi^2_nano(min) - chi^2_grey(min).  A
positive Delta says the Planck extrapolation fits the N band BETTER than any
small-grain emissivity can.  Tiers are assigned by :func:`assess_star`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

C2_UM_K = 14387.77          # hc/k in um K
WL_REF_UM = 2.2             # the K band, where the population excess is measured
BAND_WAVELENGTH_UM = {"H": 1.65, "K": 2.2, "L": 3.8, "N": 10.5, "N8": 8.5, "N11": 11.1,
                      "W1": 3.35, "W2": 4.6, "W3": 11.56, "W4": 22.1, "Ks": 2.16}

# Tier vocabulary (also the per-star ``status`` field).
TIER_CANDIDATE = "candidate"                  # Planck-consistent N band, nano rejected
TIER_INTEREST = "interest"                    # Planck-consistent but N not significant
TIER_NANO = "nano_preferred"                  # K-bright / N-faint: small grains
TIER_INCONCLUSIVE = "inconclusive"            # N measured, neither family preferred
TIER_N_UNTESTED = "N_UNTESTED"                # no N-band measurement at all
TIER_NO_NIR = "NO_NIR_EXCESS"                 # no H/K detection to extrapolate
TIER_COMPANION_T = "COMPANION_TEMPERATURE"    # grey T > 1800 K: a photosphere, not grains
TIER_KNOWN_COMPANION = "KNOWN_COMPANION"      # a companion at the 1 % level explains it
TIER_UNVERIFIED = "UNVERIFIED_INPUT"          # driving values not confirmed against an archive
TIER_BAD_FIT = "NEITHER_MODEL_FITS"           # both families rejected by the data

DEFAULT_PHYSICS = {
    # log grid for the grey / nano temperature; the free grey fit must reach a
    # companion photosphere as hot as the primary, hence the 7000 K ceiling
    "t_grid_k": [400.0, 7000.0, 110],
    "f_grid_pct": [0.02, 60.0, 90],          # log grid for f_ref (percent at K)
    "grain_sizes_um": [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
    "betas": [1.0, 2.0],
    "swarm_t_range_k": [700.0, 1800.0],      # grains (or collectors) survive; above is a photosphere
    # The nano-grain family is fitted TWICE: restricted to the temperatures the
    # published small-grain models occupy (grains at or just outside their
    # sublimation radius: ~1200-2000 K for carbon, Kirchschlager+2017,
    # Lebreton+2013 -- verify) and unrestricted.  A cool (~1000 K) nano-grain
    # population reproduces the K/N ratio of a 1500 K grey body exactly (the
    # lower temperature raises N/K by the factor the emissivity lowers it), so
    # the unrestricted fit is never decisive from K + N alone; it is reported
    # as the degeneracy every candidate carries until L/M-band data break it.
    # The ceiling is the sublimation temperature of the most refractory grain
    # material (carbon, ~2000 K): a hotter emitter is not a grain.
    "nano_t_range_k": [1500.0, 2000.0],
    # Delta chi2 is also reported at each of these nano-grain temperature
    # floors, so the reader sees how the verdict depends on the assumption.
    "nano_t_floors_k": [1000.0, 1200.0, 1500.0],
    "delta_chi2_min": 9.0,                   # nano rejected at >= 3 sigma
    "nir_detection_sigma": 3.0,              # a K/H excess must be this significant to extrapolate
    "n_detection_sigma": 3.0,                # and the N excess this significant for a candidate
    "planck_consistency_sigma": 2.0,         # |N_obs - N_pred(grey)| within this many sigma
    "grey_fit_p_min": 0.01,                  # chi^2 goodness of the grey fit
    # verify: cross-instrument absolute-calibration floors, in quadrature
    # (FLUOR / PIONIER / JouFLU visibilities against LBTI / KIN nulls)
    "calibration_floor_pct": {"H": 0.1, "K": 0.1, "L": 0.3, "N": 0.3, "N8": 0.3, "N11": 0.3},
    "upper_limit_sigma": 3.0,                # a quoted upper limit is taken as this many sigma
    "variability_sigma": 3.0,
}


# ---------------------------------------------------------------------------
# Planck and the two model families
# ---------------------------------------------------------------------------
def planck(wl_um, t_k):
    """B_lambda up to a constant (per unit wavelength)."""
    wl = np.asarray(wl_um, dtype=float)
    x = C2_UM_K / (wl * float(t_k))
    return wl ** -5 / np.expm1(x)


def emissivity(wl_um, a_um: float, beta: float):
    """Small-grain emissivity: unity below 2 pi a, (2 pi a / lambda)^beta beyond."""
    wl = np.asarray(wl_um, dtype=float)
    lam0 = 2.0 * math.pi * float(a_um)
    return np.minimum(1.0, (lam0 / wl) ** float(beta))


def excess_fraction(wl_um, t_k: float, f_ref: float, teff_k: float, *,
                    wl_ref_um: float = WL_REF_UM, a_um: float | None = None,
                    beta: float = 0.0):
    """f(lambda) for a grey body (``a_um`` None) or a nano-grain family member,
    normalised to ``f_ref`` at ``wl_ref_um``.  Same units in and out."""
    wl = np.asarray(wl_um, dtype=float)
    q = 1.0 if a_um is None else emissivity(wl, a_um, beta)
    q_ref = 1.0 if a_um is None else float(emissivity(wl_ref_um, a_um, beta))
    shape = q * planck(wl, t_k) / planck(wl, teff_k)
    shape_ref = q_ref * float(planck(wl_ref_um, t_k) / planck(wl_ref_um, teff_k))
    return float(f_ref) * shape / shape_ref


def planck_extrapolation(f_k_pct: float, teff_k: float, t_k: float = 1500.0,
                         wl_um: float = 10.5) -> float:
    """The N-band excess a grey body at ``t_k`` with ``f_k_pct`` in K produces."""
    return float(excess_fraction(wl_um, t_k, f_k_pct, teff_k))


# ---------------------------------------------------------------------------
# Measurements and chi^2
# ---------------------------------------------------------------------------
@dataclass
class Measurement:
    """One excess measurement: ``value_pct`` +/- ``err_pct`` at ``wl_um``.

    ``kind`` is ``meas`` (a value with a Gaussian error, whether or not it is
    a detection) or ``upper`` (``value_pct`` is the quoted upper limit).
    ``verified`` says whether the value was read from, or confirmed against,
    the archive on this run; an embedded value that was not is never allowed
    to drive a candidate.
    """

    band: str
    wl_um: float
    value_pct: float
    err_pct: float
    kind: str = "meas"
    instrument: str = ""
    epoch: str = ""
    source: str = ""
    verified: bool = False
    origin: str = "embedded"

    @property
    def significance(self) -> float:
        if self.kind != "meas" or not (self.err_pct > 0):
            return 0.0
        return float(self.value_pct / self.err_pct)

    def as_dict(self) -> dict:
        return {"band": self.band, "wl_um": self.wl_um, "value_pct": self.value_pct,
                "err_pct": self.err_pct, "kind": self.kind, "instrument": self.instrument,
                "epoch": self.epoch, "source": self.source, "verified": bool(self.verified),
                "origin": self.origin, "significance": round(self.significance, 2)}


def band_family(band: str) -> str:
    """``N8`` / ``N11`` / ``N`` -> ``N``; everything else is its own family."""
    b = str(band).strip()
    return "N" if b.upper().startswith("N") else b


def total_error(m: Measurement, floors: dict) -> float:
    fl = float(floors.get(m.band, floors.get(band_family(m.band), 0.0)))
    return math.sqrt(float(m.err_pct) ** 2 + fl ** 2)


def chi2_of(model_pct, meas: list[Measurement], floors: dict, upper_sigma: float = 3.0
            ) -> float:
    """chi^2 of ``model_pct`` (one value per measurement) against the set."""
    tot = 0.0
    for m, pred in zip(meas, np.atleast_1d(model_pct), strict=True):
        sig = total_error(m, floors)
        if m.kind == "upper":
            lim = float(m.value_pct)
            s = max(lim / float(upper_sigma), 1e-6)
            if pred > lim:
                tot += ((pred - lim) / s) ** 2
        else:
            tot += ((float(m.value_pct) - pred) / max(sig, 1e-6)) ** 2
    return float(tot)


def _log_grid(lo: float, hi: float, n: int) -> np.ndarray:
    return np.exp(np.linspace(math.log(lo), math.log(hi), int(n)))


def chi2_matrix(models_pct: np.ndarray, meas: list[Measurement], floors: dict,
                upper_sigma: float = 3.0) -> np.ndarray:
    """Vectorised :func:`chi2_of`: ``models_pct`` is (n_models, n_meas)."""
    models = np.atleast_2d(np.asarray(models_pct, dtype=float))
    tot = np.zeros(models.shape[0])
    for j, m in enumerate(meas):
        pred = models[:, j]
        if m.kind == "upper":
            lim = float(m.value_pct)
            s = max(lim / float(upper_sigma), 1e-6)
            tot += np.where(pred > lim, ((pred - lim) / s) ** 2, 0.0)
        else:
            sig = max(total_error(m, floors), 1e-6)
            tot += ((float(m.value_pct) - pred) / sig) ** 2
    return tot


def fit_family(meas: list[Measurement], teff_k: float, family: str = "grey",
               physics: dict | None = None, t_range: tuple | list | None = None) -> dict:
    """Grid minimum of chi^2 over (T, f_ref[, a, beta]) for one family.

    A grid rather than an optimiser: the surface has the classic T-f_ref
    degeneracy when only one band is measured, and the grid reports the
    minimum honestly instead of a spurious convergence.  ``t_range`` clips the
    temperature grid (the restricted nano-grain fit).
    """
    ph = {**DEFAULT_PHYSICS, **(physics or {})}
    floors = ph["calibration_floor_pct"]
    wl = np.array([m.wl_um for m in meas], dtype=float)
    t_lo, t_hi, t_n = ph["t_grid_k"]
    f_lo, f_hi, f_n = ph["f_grid_pct"]
    t_grid = _log_grid(t_lo, t_hi, t_n)
    if t_range is not None:
        lo, hi = float(t_range[0]), float(t_range[1])
        t_grid = np.concatenate([[lo], t_grid[(t_grid > lo) & (t_grid < hi)], [hi]])
    f_grid = _log_grid(f_lo, f_hi, f_n)
    if family == "grey":
        members = [(None, 0.0)]
    else:
        members = [(a, b) for b in ph["betas"] for a in ph["grain_sizes_um"]]
    best = {"chi2": float("inf"), "t_k": float("nan"), "f_ref_pct": float("nan"),
            "a_um": None, "beta": None, "family": family}
    for a_um, beta in members:
        for t in t_grid:
            shape = excess_fraction(wl, t, 1.0, teff_k, a_um=a_um, beta=beta)
            models = f_grid[:, None] * shape[None, :]
            c = chi2_matrix(models, meas, floors, ph["upper_limit_sigma"])
            i = int(np.argmin(c))
            if c[i] < best["chi2"]:
                best.update({"chi2": float(c[i]), "t_k": float(t), "f_ref_pct": float(f_grid[i]),
                             "a_um": a_um, "beta": beta})
    n_free = 2 if family == "grey" else 3
    best["dof"] = max(len(meas) - n_free, 0)
    best["n_meas"] = len(meas)
    if math.isfinite(best["t_k"]):
        best["prediction_pct"] = {m.band: float(v) for m, v in zip(
            meas, excess_fraction(wl, best["t_k"], best["f_ref_pct"], teff_k,
                                  a_um=best["a_um"], beta=best["beta"] or 0.0), strict=True)}
    return best


def _nano_n_ceiling(anchor: Measurement, teff_k: float, wl_n_um: float, ph: dict) -> float:
    """The brightest N-band excess the RESTRICTED nano-grain family can give
    for the anchor's NIR excess: max over (a, beta, T in nano_t_range) of
    f(N) with f(anchor band) fixed.  Anything measured well above it is
    outside what small grains at their sublimation temperature can do."""
    lo, hi = ph["nano_t_range_k"]
    t_grid = np.concatenate([[lo], _log_grid(lo, hi, 30), [hi]])
    best = 0.0
    for beta in ph["betas"]:
        for a in ph["grain_sizes_um"]:
            for t in t_grid:
                v = float(excess_fraction(wl_n_um, t, anchor.value_pct, teff_k,
                                          wl_ref_um=anchor.wl_um, a_um=a, beta=beta))
                best = max(best, v)
    return float(best)


def chi2_sf(chi2: float, dof: int) -> float:
    """Survival function of chi^2 (scipy if present; a series fallback)."""
    if dof <= 0:
        return 1.0
    try:
        from scipy.stats import chi2 as _c2
        return float(_c2.sf(chi2, dof))
    except Exception:                                     # noqa: BLE001
        # Wilson-Hilferty normal approximation
        z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
        return float(0.5 * math.erfc(z / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Per-star assessment
# ---------------------------------------------------------------------------
@dataclass
class StarContext:
    """What is known about the star besides its excesses."""

    key: str
    teff_k: float = 6000.0
    known_companion: bool = False
    companion_note: str = ""
    companion_assessed: bool = False
    polarimetry_limit_ppm: float | None = None
    notes: list[str] = field(default_factory=list)


def variability(meas: list[Measurement], physics: dict | None = None) -> dict:
    """Repeat epochs in one band family: the largest pairwise |f1 - f2| / sigma.

    A swarm being built or decaying changes; grains in equilibrium do not.
    Only kappa Tuc varied in the record (now a companion, Stuber et al. 2026),
    so the term is reported, never used as a kill.
    """
    ph = {**DEFAULT_PHYSICS, **(physics or {})}
    floors = ph["calibration_floor_pct"]
    by_band: dict[str, list[Measurement]] = {}
    for m in meas:
        if m.kind == "meas":
            by_band.setdefault(band_family(m.band), []).append(m)
    best = {"z_max": 0.0, "pairs": 0, "variable": False, "band": None}
    for band, ms in by_band.items():
        epochs = {}
        for m in ms:
            epochs.setdefault((m.instrument, m.epoch), m)
        ms = list(epochs.values())
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = ms[i], ms[j]
                s = math.sqrt(total_error(a, floors) ** 2 + total_error(b, floors) ** 2)
                z = abs(a.value_pct - b.value_pct) / max(s, 1e-6)
                best["pairs"] += 1
                if z > best["z_max"]:
                    best.update({"z_max": float(z), "band": band,
                                 "pair": [a.as_dict(), b.as_dict()]})
    best["variable"] = bool(best["z_max"] >= ph["variability_sigma"])
    return best


def assess_star(meas: list[Measurement], ctx: StarContext, physics: dict | None = None
                ) -> dict:
    """The per-star verdict.

    Order of the gates (each is a named field in the result):

    1. an H or K excess detected at >= ``nir_detection_sigma`` -- else NO_NIR_EXCESS;
    2. an N-band measurement (any kind) -- else N_UNTESTED, never a candidate;
    3. grey and nano fits; Delta = chi2_nano - chi2_grey;
    4. grey T above the swarm range -> COMPANION_TEMPERATURE;
    5. a known companion at the 1 % level -> KNOWN_COMPANION (closure phases do
       not exclude them, Tsishchankava et al. 2025);
    6. the driving values (the NIR detection and the N measurement) must be
       archive-read or archive-verified -> else UNVERIFIED_INPUT;
    7. tiers: candidate (Delta >= delta_chi2_min, N detected >= n_detection_sigma,
       N within planck_consistency_sigma of the grey prediction, grey fit
       acceptable), interest (Planck-consistent N but not significant),
       nano_preferred (Delta <= -delta_chi2_min), inconclusive otherwise.
    """
    ph = {**DEFAULT_PHYSICS, **(physics or {})}
    out: dict = {"key": ctx.key, "teff_k": float(ctx.teff_k), "n_meas": len(meas),
                 "measurements": [m.as_dict() for m in meas], "gates": {}}
    nir = [m for m in meas if band_family(m.band) in ("H", "K") and m.kind == "meas"]
    nir_det = [m for m in nir if m.significance >= ph["nir_detection_sigma"]]
    nband = [m for m in meas if band_family(m.band) == "N"]
    out["gates"]["nir_detected"] = bool(nir_det)
    out["gates"]["n_measured"] = bool(nband)
    out["variability"] = variability(meas, ph)
    if not nir_det:
        out["tier"] = TIER_NO_NIR
        out["reason"] = "no H/K excess at >= {:.1f} sigma to extrapolate".format(ph["nir_detection_sigma"])
        return out
    # the strongest NIR detection anchors the Planck extrapolation that is reported
    anchor = max(nir_det, key=lambda m: m.significance)
    out["anchor"] = anchor.as_dict()
    f_k_equiv = float(excess_fraction(WL_REF_UM, 1500.0, anchor.value_pct, ctx.teff_k,
                                      wl_ref_um=anchor.wl_um))
    out["planck_1500k_prediction_pct"] = {
        "K": f_k_equiv,
        "N": planck_extrapolation(f_k_equiv, ctx.teff_k, 1500.0, BAND_WAVELENGTH_UM["N"]),
        "L": planck_extrapolation(f_k_equiv, ctx.teff_k, 1500.0, BAND_WAVELENGTH_UM["L"]),
    }
    if not nband:
        out["tier"] = TIER_N_UNTESTED
        out["reason"] = ("NIR excess {:.2f} +/- {:.2f} % but no N-band measurement; the Planck "
                         "extrapolation predicts {:.1f} % at 10.5 um and nothing tests it".format(anchor.value_pct, anchor.err_pct,
                            out["planck_1500k_prediction_pct"]["N"]))
        return out
    t_lo, t_hi = ph["swarm_t_range_k"]
    grey_free = fit_family(meas, ctx.teff_k, "grey", ph)
    grey = fit_family(meas, ctx.teff_k, "grey", ph, t_range=(t_lo, t_hi))
    nano = fit_family(meas, ctx.teff_k, "nano", ph, t_range=ph["nano_t_range_k"])
    nano_free = fit_family(meas, ctx.teff_k, "nano", ph)
    out["grey"] = grey                       # the swarm hypothesis: T inside the swarm range
    out["grey_free"] = grey_free             # T free: a photosphere shows up here
    out["nano"] = nano                       # small grains at their sublimation temperatures
    out["nano_unrestricted"] = nano_free
    delta = float(nano["chi2"] - grey["chi2"])
    out["delta_chi2"] = delta
    out["delta_chi2_unrestricted"] = float(nano_free["chi2"] - grey["chi2"])
    out["delta_chi2_by_nano_t_floor"] = {
        str(int(tf)): float(fit_family(meas, ctx.teff_k, "nano", ph,
                                       t_range=(tf, ph["nano_t_range_k"][1]))["chi2"]
                            - grey["chi2"])
        for tf in ph["nano_t_floors_k"]}
    out["degenerate_with_cool_nano_grains"] = bool(
        delta >= ph["delta_chi2_min"] and out["delta_chi2_unrestricted"] < ph["delta_chi2_min"])
    out["grey_fit_p"] = chi2_sf(grey["chi2"], grey["dof"])
    out["nano_fit_p"] = chi2_sf(nano["chi2"], nano["dof"])
    out["grey_free_fit_p"] = chi2_sf(grey_free["chi2"], grey_free["dof"])
    n_best = max(nband, key=lambda m: (m.kind == "meas", m.significance))
    n_pred = float(grey["prediction_pct"].get(n_best.band, float("nan")))
    n_sig_tot = total_error(n_best, ph["calibration_floor_pct"])
    n_consistent = (n_best.kind == "meas"
                    and abs(n_best.value_pct - n_pred) <= ph["planck_consistency_sigma"] * n_sig_tot)
    # how far the N band sits from the brightest N the restricted nano family
    # can produce at the grey fit's K excess: the channel's discriminating power
    # on this star, in sigma
    nano_n_max = _nano_n_ceiling(anchor, ctx.teff_k, n_best.wl_um, ph)
    out["n_band"] = {"measurement": n_best.as_dict(), "grey_prediction_pct": n_pred,
                     "consistent_with_grey": bool(n_consistent),
                     "detected": bool(n_best.significance >= ph["n_detection_sigma"]),
                     "nano_ceiling_pct": nano_n_max,
                     "separation_sigma": float((n_best.value_pct - nano_n_max) / n_sig_tot)
                     if n_best.kind == "meas" else float("nan")}
    out["gates"]["grey_t_in_swarm_range"] = bool(t_lo <= grey_free["t_k"] <= t_hi)
    out["gates"]["companion_assessed"] = bool(ctx.companion_assessed)
    out["gates"]["known_companion"] = bool(ctx.known_companion)
    driving = [anchor, n_best]
    out["gates"]["driving_values_verified"] = all(m.verified for m in driving)
    grey_ok = out["grey_fit_p"] >= ph["grey_fit_p_min"]
    nano_ok = out["nano_fit_p"] >= ph["grey_fit_p_min"]
    # --- tiers -----------------------------------------------------------
    # A free grey body hotter than anything that survives, preferred over BOTH
    # the swarm-range grey body and the small grains, is a photosphere.
    hot_pref = (grey_free["t_k"] > t_hi
                and grey["chi2"] - grey_free["chi2"] >= ph["delta_chi2_min"]
                and nano["chi2"] - grey_free["chi2"] >= ph["delta_chi2_min"])
    if hot_pref:
        out["tier"] = TIER_COMPANION_T
        out["reason"] = ("a grey body at T = {:.0f} K > {:.0f} K is preferred over every "
                         "swarm-range and small-grain model: a companion photosphere".format(grey_free["t_k"], t_hi))
        return out
    if ctx.known_companion:
        out["tier"] = TIER_KNOWN_COMPANION
        out["reason"] = "known companion at the %% level: %s" % (ctx.companion_note or "catalogued")
        return out
    if delta >= ph["delta_chi2_min"] and grey_ok and n_consistent and out["n_band"]["detected"]:
        if not out["gates"]["driving_values_verified"]:
            out["tier"] = TIER_UNVERIFIED
            out["reason"] = ("Planck-consistent on embedded values that this run did not "
                             "confirm against an archive")
            return out
        out["tier"] = TIER_CANDIDATE
        out["reason"] = ("N-band excess {:.2f} +/- {:.2f} % matches the grey extrapolation {:.2f} % "
                         "(T = {:.0f} K); nano-grain family rejected, Delta chi2 = {:.1f}".format(n_best.value_pct, n_best.err_pct, n_pred, grey["t_k"], delta))
        return out
    if delta <= -ph["delta_chi2_min"] and nano_ok:
        out["tier"] = TIER_NANO
        out["reason"] = ("K-bright / N-faint: nano grains (a = {} um, beta = {}) preferred, "
                         "Delta chi2 = {:.1f}".format(nano["a_um"], nano["beta"], delta))
        return out
    if not grey_ok and not nano_ok:
        out["tier"] = TIER_BAD_FIT
        out["reason"] = "neither family fits (p_grey = {:.3g}, p_nano = {:.3g})".format(
            out["grey_fit_p"], out["nano_fit_p"])
        return out
    if n_consistent and grey_ok and out["gates"]["grey_t_in_swarm_range"]:
        out["tier"] = TIER_INTEREST if out["gates"]["driving_values_verified"] else TIER_UNVERIFIED
        out["reason"] = ("N band consistent with the grey extrapolation but not decisive "
                         f"(N at {n_best.significance:.1f} sigma, Delta chi2 = {delta:.1f})")
        return out
    out["tier"] = TIER_INCONCLUSIVE
    out["reason"] = f"N band measured; neither family preferred (Delta chi2 = {delta:.1f})"
    return out


# ---------------------------------------------------------------------------
# Broadband leg: what a hot component does to K-W1, W1-W2, W2-W3
# ---------------------------------------------------------------------------
def colour_shift(teff_k: float, f_k_pct: float, t_k: float = 1500.0) -> dict:
    """Magnitude shifts of K-W1, W1-W2 and W2-W3 for a grey ``t_k`` component
    at ``f_k_pct`` in K (positive = redder).  Monochromatic band centres; the
    photosphere is Rayleigh-Jeans from W1 on, so the approximation is good to
    a few percent of the shift."""
    f = {b: float(excess_fraction(BAND_WAVELENGTH_UM[b], t_k, f_k_pct, teff_k))
         for b in ("Ks", "W1", "W2", "W3")}
    dm = {b: -2.5 * math.log10(1.0 + v / 100.0) for b, v in f.items()}
    return {"d_k_w1": dm["Ks"] - dm["W1"], "d_w1_w2": dm["W1"] - dm["W2"],
            "d_w2_w3": dm["W2"] - dm["W3"], "excess_pct": f}


def f_k_for_shift(teff_k: float, target_d_k_w1: float, t_k: float = 1500.0) -> float:
    """The K-excess (percent) that produces a given K-W1 shift: the broadband
    leg's sensitivity in the interferometric unit."""
    lo, hi = 0.01, 500.0
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if colour_shift(teff_k, mid, t_k)["d_k_w1"] < target_d_k_w1:
            lo = mid
        else:
            hi = mid
    return float(math.sqrt(lo * hi))


def running_locus(x, y, n_bins: int = 25, min_per_bin: int = 8) -> dict:
    """Median and robust scatter of ``y`` in bins of ``x`` -- the empirical
    photospheric locus.  Returns interpolators as arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 * min_per_bin:
        return {"x": np.array([]), "med": np.array([]), "sig": np.array([]), "n": np.array([])}
    edges = np.quantile(x[ok], np.linspace(0, 1, int(n_bins) + 1))
    xs, meds, sigs, ns = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = ok & (x >= lo) & (x <= hi)
        if sel.sum() < min_per_bin:
            continue
        yy = y[sel]
        med = float(np.median(yy))
        mad = float(np.median(np.abs(yy - med))) * 1.4826
        xs.append(float(np.median(x[sel])))
        meds.append(med)
        sigs.append(max(mad, 0.01))
        ns.append(int(sel.sum()))
    return {"x": np.array(xs), "med": np.array(meds), "sig": np.array(sigs), "n": np.array(ns)}


def locus_residual(x, y, locus: dict) -> tuple[np.ndarray, np.ndarray]:
    """(y - median(x), sigma(x)) by linear interpolation along the locus."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(locus["x"]) < 2:
        nan = np.full(len(x), np.nan)
        return nan, nan
    med = np.interp(x, locus["x"], locus["med"])
    sig = np.interp(x, locus["x"], locus["sig"])
    return y - med, sig


BROADBAND_NORMAL = "NORMAL"
BROADBAND_HOT = "BROADBAND_HOT_EXCESS"
BROADBAND_W3_FAINT = "BROADBAND_W3_TOO_FAINT"
BROADBAND_W3_BRIGHT = "BROADBAND_W3_TOO_BRIGHT"
BROADBAND_SATURATED = "SATURATED"
BROADBAND_CONTAMINATED = "CONTAMINATED"
BROADBAND_BLEND = "NEGATIVE_W1_W2_BLEND"
BROADBAND_UNTESTED = "INSUFFICIENT_PHOTOMETRY"


def broadband_classify(teff_k: float, r_kw1: float, s_kw1: float, r_w12: float, s_w12: float,
                       r_w23: float | None, s_w23: float | None, *, saturated: bool,
                       contaminated: bool, sigma_min: float = 3.0, t_k: float = 1500.0) -> dict:
    """One star of the broadband leg.

    A hot grey component reddens K-W1 AND W1-W2 by amounts in a fixed ratio
    (for T = 1500 K roughly 1 : 0.8) and reddens W2-W3 by a further, larger,
    fixed amount.  Both NIR-side residuals must be positive at ``sigma_min``,
    the implied K excesses from the two colours must agree within a factor of
    two, and where W3 exists its residual must sit within 3 sigma of the
    extrapolation -- too faint is nano-like (or noise), too bright is cooler
    dust.  A negative W1-W2 residual beyond 3 sigma is a blend (the ledger).
    """
    out = {"class": BROADBAND_NORMAL, "f_k_from_k_w1_pct": float("nan"),
           "f_k_from_w1_w2_pct": float("nan"), "f_k_sensitivity_pct": float("nan")}
    if saturated:
        out["class"] = BROADBAND_SATURATED
        return out
    if contaminated:
        out["class"] = BROADBAND_CONTAMINATED
        return out
    if not all(np.isfinite([r_kw1, s_kw1, r_w12, s_w12])):
        out["class"] = BROADBAND_UNTESTED
        return out
    out["f_k_sensitivity_pct"] = f_k_for_shift(teff_k, sigma_min * s_kw1, t_k)
    if r_w12 < -sigma_min * s_w12:
        out["class"] = BROADBAND_BLEND
        return out
    z1, z2 = r_kw1 / s_kw1, r_w12 / s_w12
    out["z_k_w1"], out["z_w1_w2"] = float(z1), float(z2)
    if z1 < sigma_min or z2 < sigma_min:
        return out
    f1 = f_k_for_shift(teff_k, r_kw1, t_k)
    unit = colour_shift(teff_k, 1.0, t_k)
    # W1-W2 is linear in f at small f; solve by the same bisection on d_w1_w2
    lo, hi = 0.01, 500.0
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if colour_shift(teff_k, mid, t_k)["d_w1_w2"] < r_w12:
            lo = mid
        else:
            hi = mid
    f2 = math.sqrt(lo * hi)
    out["f_k_from_k_w1_pct"], out["f_k_from_w1_w2_pct"] = float(f1), float(f2)
    if not (0.5 <= f1 / f2 <= 2.0):
        out["class"] = BROADBAND_NORMAL
        out["note"] = f"K-W1 and W1-W2 imply different hot excesses ({f1:.1f} vs {f2:.1f} %)"
        return out
    f_est = math.sqrt(f1 * f2)
    out["f_k_est_pct"] = float(f_est)
    if r_w23 is not None and s_w23 is not None and np.isfinite(r_w23) and np.isfinite(s_w23):
        pred = colour_shift(teff_k, f_est, t_k)["d_w2_w3"]
        out["d_w2_w3_pred"] = float(pred)
        if r_w23 < pred - 3.0 * s_w23:
            out["class"] = BROADBAND_W3_FAINT
            return out
        if r_w23 > pred + 3.0 * s_w23:
            out["class"] = BROADBAND_W3_BRIGHT
            return out
    out["unit_shift"] = unit
    out["class"] = BROADBAND_HOT
    return out


__all__ = ["BAND_WAVELENGTH_UM", "BROADBAND_BLEND", "BROADBAND_CONTAMINATED", "BROADBAND_HOT",
           "BROADBAND_NORMAL", "BROADBAND_SATURATED", "BROADBAND_UNTESTED",
           "BROADBAND_W3_BRIGHT", "BROADBAND_W3_FAINT", "DEFAULT_PHYSICS", "Measurement",
           "StarContext", "TIER_BAD_FIT", "TIER_CANDIDATE", "TIER_COMPANION_T",
           "TIER_INCONCLUSIVE", "TIER_INTEREST", "TIER_KNOWN_COMPANION", "TIER_NANO",
           "TIER_NO_NIR", "TIER_N_UNTESTED", "TIER_UNVERIFIED", "WL_REF_UM", "assess_star",
           "band_family", "broadband_classify", "chi2_of", "colour_shift", "emissivity",
           "excess_fraction", "f_k_for_shift", "fit_family", "locus_residual", "planck",
           "planck_extrapolation", "running_locus", "variability"]
