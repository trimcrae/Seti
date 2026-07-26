"""Energy closure: where the intercepted luminosity has to reappear.

This module turns CENOTAPH from a "missing light" claim into an *energy
conservation test*, which is a far stronger thing to be able to state.

The three legs
--------------
1. A grey deficit ``g`` gives a covering fraction ``f_dim = 1 − 10^(−0.4 g)``.
2. Nothing in WISE W1–W4 bounds the re-radiation temperature from *above*:
   a warm shell would have been seen at 12/22 µm. This is exactly what makes
   such an object invisible to every executed Dyson search, all of which are
   capped by W4 at 22 µm and quote 100–1000 K.
3. The intercepted power ``f·L`` must reappear somewhere. If the occulter is
   isotropic, energy conservation forces

       f_ir ≡ L_IR / L_star  =  f_dim

   *independently of the geometry or optical depth*, because every photon the
   star emits either escapes or is thermalised and re-emitted.

The closure ratio ``ρ = f_ir / f_dim`` is therefore the single most diagnostic
number in the channel:

``ρ ≈ 1``      an isotropic occulter — the energy is found where nobody looked.
``ρ ≪ 1``      either an *anisotropic* occulter (an edge-on disk blocks the
               line of sight while intercepting only its own small solid angle
               — the #1 astrophysical confounder, and the reason leg 3 is
               mandatory rather than optional), or the extraordinary case in
               which the energy is not thermalised at all.
``ρ > 1``      the far-IR flux has been mis-attributed: cirrus, a background
               galaxy in the beam, or a blend.

Temperature ↔ radius
--------------------
An opaque shell element at radius ``r`` that absorbs on one face and radiates
from the same face sits at ``T = 393.6 K · (L/L_⊙)^(1/4) / √(r/AU)``. So

    100 K → 15.5 AU      50 K → 62 AU      30 K → 172 AU

and the Wien peak moves to 29 / 58 / 97 µm respectively. The first of those is
already past WISE's longest band. Ćirković & Bradbury (2006) argue on
Landauer/Brillouin grounds that computation is more efficient against a colder
reservoir, and that advanced civilisations therefore migrate toward cold
regions; they give no specific shell temperature (see ``docs/cenotaph.md`` §1.2
— an earlier draft of this repository misattributed a "50 K vs 300 K" quote to
them). The 30-100 K window here is set by what the far-IR all-sky catalogues
can actually reach.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

# --- physical constants (SI) -------------------------------------------------
H_PLANCK = 6.62607015e-34
K_BOLTZ = 1.380649e-23
C_LIGHT = 2.99792458e8
SIGMA_SB = 5.670374419e-8
L_SUN = 3.828e26          # W (IAU nominal)
AU_M = 1.495978707e11
PC_M = 3.0856775814913673e16
JY = 1.0e-26              # W m^-2 Hz^-1
WIEN_UM_K = 2897.771955   # µm·K, for the wavelength peak of B_lambda

T_EQ_1AU_SHELL = (L_SUN / (4.0 * math.pi * AU_M**2 * SIGMA_SB)) ** 0.25
"""≈393.6 K: one-sided (opaque shell) equilibrium temperature at 1 AU, L = L_⊙."""


@dataclass(frozen=True)
class FarIRBand:
    """A far-infrared band: wavelength, point-source sensitivity, beam."""

    name: str
    lambda_um: float
    limit_jy: float
    """Practical 5σ point-source detection limit of the all-sky catalogue (Jy)."""
    beam_fwhm_arcsec: float
    survey: str = ""


# Sensitivities are the catalogue-level practical limits, not the idealised
# instrumental ones; away from the Galactic plane they are set by cirrus
# confusion rather than by detector noise. They are used only to compute
# *decidability* — how far out the far-IR leg can actually rule on a star — and
# any real run reads the measured per-source flux, never these numbers.
FAR_IR_BANDS: tuple[FarIRBand, ...] = (
    FarIRBand("iras60", 60.0, 0.5, 120.0, "IRAS PSC"),
    FarIRBand("iras100", 100.0, 1.0, 180.0, "IRAS PSC"),
    # AKARI/FIS 5-sigma point-source limits from Kawada et al. (2007).
    # Beams are the ALL-SKY SURVEY PSF (~60" at 65/90 um, ~90" at 140/160),
    # not the much narrower slow-scan pointed PSF the instrument paper quotes;
    # the BSC is built from the survey mode.
    FarIRBand("akari65", 65.0, 2.4, 60.0, "AKARI/FIS BSC N60"),
    FarIRBand("akari90", 90.0, 0.55, 60.0, "AKARI/FIS BSC WIDE-S"),
    FarIRBand("akari140", 140.0, 1.4, 90.0, "AKARI/FIS BSC WIDE-L"),
    FarIRBand("akari160", 160.0, 6.3, 90.0, "AKARI/FIS BSC N160"),
)

MID_IR_BANDS: tuple[FarIRBand, ...] = (
    FarIRBand("w1", 3.35, 8.0e-5, 6.1, "AllWISE"),
    FarIRBand("w2", 4.60, 1.1e-4, 6.4, "AllWISE"),
    FarIRBand("w3", 11.56, 0.0018, 6.5, "AllWISE"),   # ~1.8 mJy 5-sigma
    FarIRBand("w4", 22.09, 0.0120, 12.0, "AllWISE"),  # ~12 mJy 5-sigma
)

BAND_BY_NAME = {b.name: b for b in FAR_IR_BANDS + MID_IR_BANDS}


def wise_temperature_ceilings() -> dict[str, float]:
    """Blackbody temperature whose Wien peak falls in each WISE band.

    This is the structural reason the mid-infrared route to the cold regime is
    closed, and it is worth stating as a computed fact rather than an opinion:

        W1 3.35 µm → 865 K     W2 4.60 µm → 630 K
        W3 11.56 µm → 251 K    W4 22.09 µm → 131 K

    A survey is only efficiently sensitive to re-radiation near its own Wien
    ceiling. The catalogues that have got *deeper* since 2010 — NEOWISE-R,
    CatWISE2020 (1.9 × 10⁹ sources, 1.7 mag deeper), the unWISE coadds — are
    **W1/W2 only**. W3 and W4 depth is frozen at the 2010 cryogenic phase of
    WISE and cannot improve until a new mid-IR all-sky survey flies. So the
    largest waste-heat searches ever run are the ones *least* able to reach
    cold temperatures, and improving them makes that worse, not better: deeper
    W1/W2 buys sensitivity at 630–865 K. Below ~130 K the mid-infrared route is
    closed by instrumentation, not by effort — which is the argument for
    attenuation plus far-IR recovery as the only way in.
    """
    return {b.name: WIEN_UM_K / b.lambda_um for b in MID_IR_BANDS}


# --- blackbody ---------------------------------------------------------------
def planck_bnu(lambda_um: float | np.ndarray, t_k: float | np.ndarray):
    """``B_ν(T)`` in W m⁻² Hz⁻¹ sr⁻¹ for a wavelength in µm."""
    lam = np.asarray(lambda_um, dtype=float) * 1e-6
    t = np.asarray(t_k, dtype=float)
    nu = C_LIGHT / lam
    x = H_PLANCK * nu / (K_BOLTZ * np.maximum(t, 1e-6))
    # expm1 keeps the Rayleigh-Jeans end from cancelling to zero.
    return 2.0 * H_PLANCK * nu**3 / C_LIGHT**2 / np.expm1(np.clip(x, 1e-12, 700.0))


def blackbody_shape_per_hz(lambda_um, t_k):
    """``π B_ν(T) / (σ T⁴)`` — the unit-normalised blackbody, in Hz⁻¹.

    Multiplying by a bolometric flux gives a flux density directly, with no
    solid-angle or emitting-area bookkeeping to get wrong.
    """
    t = np.asarray(t_k, dtype=float)
    return math.pi * planck_bnu(lambda_um, t) / (SIGMA_SB * np.maximum(t, 1e-6) ** 4)


def wien_peak_um(t_k: float | np.ndarray):
    """Wavelength of the ``B_λ`` peak (µm)."""
    return WIEN_UM_K / np.asarray(t_k, dtype=float)


def equilibrium_temperature(l_lsun: float, r_au: float, sides: int = 1) -> float:
    """Temperature of a shell element at ``r_au`` around an ``l_lsun`` star.

    ``sides=1``  opaque shell, absorbs and radiates through the same area
                 (the Wright 2023 "optical depth of several" case).
    ``sides=2``  thin flat collector radiating from both faces — colder by 2^¼.
    """
    denom = 4.0 * math.pi * (r_au * AU_M) ** 2 * SIGMA_SB * float(sides)
    return float((l_lsun * L_SUN / denom) ** 0.25)


def radius_for_temperature(l_lsun: float, t_k: float, sides: int = 1) -> float:
    """Inverse of :func:`equilibrium_temperature`, in AU."""
    denom = 4.0 * math.pi * SIGMA_SB * float(sides) * t_k**4
    return float(math.sqrt(l_lsun * L_SUN / denom) / AU_M)


def collector_area_m2(f: float, r_au: float) -> float:
    """Physical area of a covering-fraction-``f`` swarm at radius ``r_au``."""
    return f * 4.0 * math.pi * (r_au * AU_M) ** 2


def material_mass_kg(f: float, r_au: float, areal_density_kg_m2: float = 0.01) -> float:
    """Mass required, at a given areal density (0.01 kg/m² ≈ a 10-µm film).

    Included because it is the honest cost of going cold: area scales as r², so
    a 50 K shell needs ~16× the material of a 200 K one at the same ``f``. The
    trade Ćirković & Bradbury propose is thermodynamic efficiency bought with
    mass, and this makes that trade quantitative rather than rhetorical.
    """
    return collector_area_m2(f, r_au) * areal_density_kg_m2


# --- observables -------------------------------------------------------------
def bolometric_flux_wm2(f: float, l_lsun: float, d_pc: float) -> float:
    """Re-radiated bolometric flux at Earth, ``f·L / 4πd²`` (W m⁻²)."""
    return f * l_lsun * L_SUN / (4.0 * math.pi * (d_pc * PC_M) ** 2)


def predicted_flux_jy(f: float, l_lsun: float, d_pc: float, t_k: float,
                      lambda_um: float | np.ndarray):
    """Predicted flux density (Jy) of the re-radiating occulter.

    Assumes the occulter is an isotropic blackbody re-radiator of the full
    intercepted power ``f·L``. A greybody with emissivity ``ε(λ) < 1`` at these
    wavelengths would be *warmer* and its far-IR flux *lower*, so this is the
    optimistic case and the resulting sensitivity horizons are upper bounds.
    """
    f_bol = bolometric_flux_wm2(f, l_lsun, d_pc)
    return f_bol * blackbody_shape_per_hz(lambda_um, t_k) / JY


def f_ir_from_flux(flux_jy: float, lambda_um: float, t_k: float,
                   l_lsun: float, d_pc: float) -> float:
    """Invert a far-IR detection into ``L_IR/L_star`` at assumed ``T``.

    This is the quantity that must equal ``f_dim`` if the occulter is isotropic.
    """
    shape = float(blackbody_shape_per_hz(lambda_um, t_k))
    if shape <= 0:
        return float("nan")
    f_bol = flux_jy * JY / shape
    star_flux = l_lsun * L_SUN / (4.0 * math.pi * (d_pc * PC_M) ** 2)
    return f_bol / star_flux


def detection_horizon_pc(f: float, l_lsun: float, t_k: float,
                         band: FarIRBand | str) -> float:
    """Distance out to which ``band`` can detect the re-radiation (pc).

    ``F ∝ d⁻²``, so this is exact given the band limit. It is the number that
    decides whether the far-IR leg *rules* on a star or merely fails to rule.
    """
    b = BAND_BY_NAME[band] if isinstance(band, str) else band
    f_at_10pc = predicted_flux_jy(f, l_lsun, 10.0, t_k, b.lambda_um)
    if f_at_10pc <= 0 or not math.isfinite(f_at_10pc):
        return 0.0
    return 10.0 * math.sqrt(f_at_10pc / b.limit_jy)


def temperature_exclusion(f: float, l_lsun: float, d_pc: float,
                          limits_jy: dict[str, float],
                          t_grid: np.ndarray | None = None) -> dict:
    """Temperatures excluded by mid-IR *non-detections*, given ``f`` from leg 1.

    For a fixed bolometric budget ``f·L`` the flux in a *fixed* band is not
    monotonic in T — it peaks when the Wien peak crosses the band (≈132 K for
    W4 at 22 µm) and falls on both sides. A non-detection therefore excludes an
    *interval*, not a half-line, and reporting it as "T < T_max" would be wrong.
    This returns the excluded interval explicitly.

    Returns keys: ``excluded_lo``/``excluded_hi`` (K, the contiguous excluded
    band containing the maximum), ``t_max_cold`` (the largest temperature still
    allowed on the cold side — the actual CENOTAPH ceiling), ``binding_band``.
    """
    t_grid = np.geomspace(5.0, 2000.0, 800) if t_grid is None else np.asarray(t_grid)
    excluded = np.zeros(t_grid.shape, dtype=bool)
    binding = {}
    for name, lim in limits_jy.items():
        b = BAND_BY_NAME.get(name)
        if b is None or not math.isfinite(lim) or lim <= 0:
            continue
        pred = predicted_flux_jy(f, l_lsun, d_pc, t_grid, b.lambda_um)
        hit = pred > lim
        binding[name] = int(hit.sum())
        excluded |= hit

    if not excluded.any():
        return {"excluded_lo": None, "excluded_hi": None,
                "t_max_cold": float(t_grid[-1]), "binding_band": None,
                "n_excluded_grid": 0,
                "note": "no mid-IR limit is constraining at this f and distance"}

    idx = np.flatnonzero(excluded)
    lo, hi = float(t_grid[idx[0]]), float(t_grid[idx[-1]])
    allowed_cold = t_grid[(t_grid < lo) & ~excluded]
    t_max_cold = float(allowed_cold[-1]) if allowed_cold.size else float(t_grid[0])
    return {
        "excluded_lo": lo, "excluded_hi": hi, "t_max_cold": t_max_cold,
        "binding_band": max(binding, key=binding.get) if binding else None,
        "n_excluded_grid": int(excluded.sum()),
        "note": "mid-IR non-detection excludes a contiguous temperature interval",
    }


@dataclass
class Closure:
    """Result of the energy-conservation test for one star."""

    f_dim: float
    f_dim_err: float
    f_ir: float | None
    f_ir_err: float | None
    closure_ratio: float | None
    t_assumed_k: float | None
    far_ir_detected: bool
    far_ir_band: str | None
    far_ir_flux_jy: float | None
    horizon_pc: float | None
    """Distance to which the *best* far-IR band could have detected this f, L."""
    decidable: bool
    """True if the star is inside the horizon — i.e. a non-detection means something."""
    verdict: str
    """``closes`` | ``anisotropic_or_nonthermal`` | ``over_closure`` |
    ``far_ir_undecidable`` | ``no_far_ir_data``"""
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def close_budget(f_dim: float, f_dim_err: float, l_lsun: float, d_pc: float,
                 t_assumed_k: float = 50.0,
                 far_ir_fluxes_jy: dict[str, float] | None = None,
                 far_ir_flux_errs_jy: dict[str, float] | None = None,
                 closure_tol: float = 0.5) -> Closure:
    """Run the energy-conservation test on one star.

    ``far_ir_fluxes_jy`` maps band name -> measured flux density; absent or NaN
    entries mean "no measurement", which is *not* the same as "zero flux" and is
    reported as such.
    """
    notes: list[str] = []
    far_ir_fluxes_jy = far_ir_fluxes_jy or {}
    far_ir_flux_errs_jy = far_ir_flux_errs_jy or {}

    # Which band gives the deepest reach for this star's f, L, T?
    horizons = {b.name: detection_horizon_pc(f_dim, l_lsun, t_assumed_k, b)
                for b in FAR_IR_BANDS}
    best_band = max(horizons, key=horizons.get)
    horizon = horizons[best_band]
    decidable = bool(math.isfinite(horizon) and d_pc <= horizon)

    detections = {k: v for k, v in far_ir_fluxes_jy.items()
                  if v is not None and math.isfinite(v) and v > 0}
    if not detections:
        verdict = "no_far_ir_data" if not far_ir_fluxes_jy else "far_ir_undecidable"
        if decidable and far_ir_fluxes_jy:
            # We looked, we could have seen it, and we did not: the energy is
            # not thermalised at the assumed temperature. This is the
            # extraordinary branch — and also where an edge-on disk hides.
            verdict = "anisotropic_or_nonthermal"
            notes.append(
                f"non-detection inside the {best_band} horizon ({horizon:.0f} pc): "
                f"either the occulter is anisotropic (edge-on disk) or the "
                f"intercepted power is not re-radiated at ~{t_assumed_k:.0f} K"
            )
        elif far_ir_fluxes_jy:
            notes.append(
                f"star at {d_pc:.0f} pc is beyond the {best_band} horizon "
                f"({horizon:.0f} pc); the far-IR non-detection carries no "
                "information and no claim is made from it"
            )
        else:
            notes.append("no far-IR catalogue coverage retrieved for this position")
        return Closure(f_dim, f_dim_err, None, None, None, t_assumed_k, False,
                       None, None, horizon, decidable, verdict, notes)

    # Use the detection with the highest signal-to-limit ratio.
    def _snr(name: str) -> float:
        b = BAND_BY_NAME.get(name)
        return detections[name] / b.limit_jy if b else 0.0

    band = max(detections, key=_snr)
    flux = detections[band]
    ferr = far_ir_flux_errs_jy.get(band, 0.2 * flux)
    lam = BAND_BY_NAME[band].lambda_um
    f_ir = f_ir_from_flux(flux, lam, t_assumed_k, l_lsun, d_pc)
    f_ir_err = abs(f_ir) * (ferr / flux) if flux > 0 else float("nan")

    ratio = f_ir / f_dim if f_dim > 0 else float("nan")
    if not math.isfinite(ratio):
        verdict = "far_ir_undecidable"
    elif abs(math.log10(max(ratio, 1e-6))) <= closure_tol:
        verdict = "closes"
        notes.append(
            f"L_IR/L_* = {f_ir:.3g} matches the covering fraction {f_dim:.3g} "
            f"from the grey deficit to within {10**closure_tol:.1f}x: the "
            "intercepted power is accounted for by an isotropic cold re-radiator"
        )
    elif ratio < 1.0:
        verdict = "anisotropic_or_nonthermal"
        notes.append(
            f"L_IR/L_* = {f_ir:.3g} is {1 / max(ratio, 1e-9):.1f}x below the "
            f"covering fraction {f_dim:.3g}: consistent with an edge-on disk "
            "intercepting only its own solid angle"
        )
    else:
        verdict = "over_closure"
        notes.append(
            f"L_IR/L_* = {f_ir:.3g} exceeds the covering fraction {f_dim:.3g}: "
            "the far-IR flux is not all from this star (cirrus, blend, or a "
            "background galaxy in the beam)"
        )

    return Closure(f_dim, f_dim_err, f_ir, f_ir_err, ratio, t_assumed_k, True,
                   band, flux, horizon, decidable, verdict, notes)


def coverage_table(l_lsun: float = 1.0, f_values=(0.05, 0.1, 0.2, 0.5),
                   t_values=(30.0, 50.0, 80.0, 100.0)) -> list[dict]:
    """Grid of far-IR detection horizons — the channel's real reach.

    This is what makes the "unprobed T < 100 K regime" claim quantitative:
    for each (f, T) it says how far a survey can actually rule.
    """
    rows = []
    for f in f_values:
        for t in t_values:
            r_au = radius_for_temperature(l_lsun, t)
            row = {
                "f": f, "t_k": t, "l_lsun": l_lsun,
                "radius_au": r_au,
                "wien_peak_um": float(wien_peak_um(t)),
                "collector_area_m2": collector_area_m2(f, r_au),
                "mass_kg_at_10um_film": material_mass_kg(f, r_au),
            }
            for b in FAR_IR_BANDS:
                row[f"horizon_{b.name}_pc"] = detection_horizon_pc(f, l_lsun, t, b)
            rows.append(row)
    return rows


__all__ = [
    "FAR_IR_BANDS",
    "MID_IR_BANDS",
    "T_EQ_1AU_SHELL",
    "Closure",
    "FarIRBand",
    "blackbody_shape_per_hz",
    "close_budget",
    "collector_area_m2",
    "coverage_table",
    "detection_horizon_pc",
    "equilibrium_temperature",
    "f_ir_from_flux",
    "material_mass_kg",
    "planck_bnu",
    "predicted_flux_jy",
    "radius_for_temperature",
    "temperature_exclusion",
    "wien_peak_um",
]
