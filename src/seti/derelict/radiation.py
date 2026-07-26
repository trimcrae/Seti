"""Radiation-pressure physics: A1 -> beta -> area-to-mass, and the R statistic.

Pure functions, no I/O, no network.  Unit-tested offline against the published
numbers in ``docs/derelict.md``.

The chain
---------
JPL fits a non-gravitational acceleration to a small body's astrometry using the
Marsden, Sekanina & Yeomans (1973) decomposition

.. math:: \\mathbf{a}_{ng} = A_1\\,g(r)\\,\\hat{r} + A_2\\,g(r)\\,\\hat{t}
                             + A_3\\,g(r)\\,\\hat{n}

with :math:`A_1` the **radial** (sunward-positive) term.  For asteroids JPL's
default is the inverse-square law :math:`g(r) = (1\\,\\mathrm{au}/r)^2`, so
:math:`A_1` is literally the radial acceleration at 1 au in au/day².

Solar radiation pressure produces an acceleration that is radial and falls as
:math:`1/r^2` -- *the same functional form as gravity*.  So it is conventionally
written as a dimensionless ratio

.. math:: \\beta \\equiv \\frac{F_{rad}}{F_{grav}}
        = \\frac{L_\\odot Q_{pr}}{4\\pi c G M_\\odot}\\,\\frac{A}{m}

and the acceleration is :math:`\\beta\\,GM_\\odot/r^2`.  Equating that to
:math:`A_1 g(r)` at :math:`r = 1` au gives the two conversions this module
exists to provide:

* ``beta = A1 / GM_sun``   with :math:`GM_\\odot` in au³/day²  (= 3379.4 * A1)
* ``AMR  = beta * 4 pi c G M_sun / (L_sun Q_pr)``              (= 1306.1 * beta)

so ``AMR = 4.4137e6 * A1 / Q_pr`` m²/kg.

Why this is a technosignature
-----------------------------
A solid sphere of diameter *D* and bulk density *rho* has cross-section
:math:`\\pi D^2/4` and mass :math:`\\rho \\pi D^3/6`, hence

.. math:: (A/m)_{nat} = \\frac{3}{2 D \\rho}

which for any body big enough to be catalogued is *tiny*: 7.5e-6 m²/kg at
D = 100 m, rho = 2000.  A thin film is 8 orders of magnitude above that.  The
normalised statistic

.. math:: R = \\frac{(A/m)_{implied}}{(A/m)_{nat}(D, \\rho)}

is therefore ~1 for a natural body of the observed size and enormous for a
film.  ``R`` -- not ``A1`` -- is the discriminant, because it removes the size
dependence that makes a raw ``A1`` cut meaningless (a 3 m rock genuinely has a
detectable SRP signal; a 3 km one cannot).

Sign convention
---------------
``A1 > 0`` is **outward** (anti-sunward), the direction radiation pressure and
sublimation both push.  ``A1 < 0`` is sunward: radiation pressure *cannot*
produce it, so a significant negative A1 is a systematic or genuinely strange.
That asymmetry is screen 3.

Caveats that travel with every number
-------------------------------------
* ``AMR`` assumes the fitted ``A1`` really is radiation pressure with a
  :math:`1/r^2` law.  If JPL fitted a *cometary* g(r) (the Marsden
  :math:`\\alpha (r/r_0)^{-m}[1+(r/r_0)^n]^{-k}` form, signalled by the presence
  of ALN/NM/NN/NK/R0 model parameters or a non-zero DT), the coefficient means
  something else and the conversion is invalid.  ``nongrav_law_is_inverse_square``
  exists to gate on that and the run refuses to convert when it is False.
* ``Q_pr = 1`` (pure absorption) is the conservative choice: a *reflective*
  sail has ``Q_pr = 2`` and needs only **half** the area-to-mass for the same
  beta.  Reporting the Q_pr = 1 value makes the anomaly claim harder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- Physical constants (IAU 2015 nominal / CODATA) ---------------------------
#: Speed of light, m/s (exact).
C_LIGHT_M_S = 2.99792458e8
#: Solar mass parameter GM_sun, m^3/s^2 (IAU 2015 nominal, TDB-compatible).
GM_SUN_M3_S2 = 1.32712440018e20
#: Nominal solar luminosity, W (IAU 2015 Resolution B3).
L_SUN_W = 3.828e26
#: Astronomical unit, m (IAU 2012 definition, exact).
AU_M = 1.495978707e11
#: Seconds per day (exact).
DAY_S = 86400.0

#: Gauss's gravitational constant, au^(3/2)/day.  k^2 = GM_sun in au^3/day^2.
GAUSS_K = 0.01720209895
#: GM_sun expressed in au^3/day^2.  This is the ONLY constant needed for A1 -> beta.
GM_SUN_AU3_DAY2 = GAUSS_K**2  # = 2.9591220828559115e-4

#: beta per unit A1 [au/day^2].  = 1 / GM_SUN_AU3_DAY2 ~= 3379.38.
BETA_PER_A1 = 1.0 / GM_SUN_AU3_DAY2

#: area-to-mass [m^2/kg] per unit beta at Q_pr = 1.  = 4 pi c GM_sun / L_sun ~= 1306.1.
AMR_PER_BETA = 4.0 * math.pi * C_LIGHT_M_S * GM_SUN_M3_S2 / L_SUN_W

#: area-to-mass [m^2/kg] per unit A1 [au/day^2] at Q_pr = 1.  ~= 4.4137e6.
AMR_PER_A1 = BETA_PER_A1 * AMR_PER_BETA

#: 1 au/day^2 expressed in m/s^2.  ~= 20.04.
AU_DAY2_IN_M_S2 = AU_M / DAY_S**2

#: Standard IAU/Harris H-to-diameter relation constant, km.
H_DIAMETER_CONST_KM = 1329.0

#: Model-parameter names that signal JPL fitted a *cometary* g(r) rather than
#: the plain inverse-square law.  Presence of any of these (or a fitted DT)
#: invalidates the A1 -> beta conversion.
COMETARY_MODEL_PARS = frozenset({"ALN", "NM", "NN", "NK", "R0", "DT"})


# --- The conversions ----------------------------------------------------------
def beta_from_a1(a1_au_day2: float | np.ndarray) -> float | np.ndarray:
    """Radiation-pressure ``beta`` implied by a radial non-grav coefficient.

    Parameters
    ----------
    a1_au_day2
        JPL ``A1`` in au/day², i.e. the radial non-gravitational acceleration at
        1 au under the inverse-square law.  Positive = outward.

    Returns
    -------
    beta = F_rad / F_grav, dimensionless.  Preserves the sign of ``A1`` (a
    negative beta is unphysical for radiation pressure and is the point of
    screen 3).
    """
    return np.asarray(a1_au_day2, dtype=float) * BETA_PER_A1 if isinstance(
        a1_au_day2, np.ndarray) else a1_au_day2 * BETA_PER_A1


def a1_from_beta(beta: float | np.ndarray) -> float | np.ndarray:
    """Inverse of :func:`beta_from_a1`: ``A1`` in au/day² for a given beta."""
    return np.asarray(beta, dtype=float) * GM_SUN_AU3_DAY2 if isinstance(
        beta, np.ndarray) else beta * GM_SUN_AU3_DAY2


def amr_from_beta(beta: float | np.ndarray, q_pr: float = 1.0) -> float | np.ndarray:
    """Area-to-mass ratio (m²/kg) implied by ``beta``.

    ``q_pr`` is the radiation-pressure efficiency: 1 for perfect absorption,
    2 for perfect specular reflection (a sail).  A reflector needs only half
    the area-to-mass to reach the same beta, so the Q_pr = 1 value returned by
    default is the CONSERVATIVE (larger) one.
    """
    if q_pr <= 0:
        raise ValueError(f"q_pr must be positive, got {q_pr}")
    return np.asarray(beta, dtype=float) * AMR_PER_BETA / q_pr if isinstance(
        beta, np.ndarray) else beta * AMR_PER_BETA / q_pr


def beta_from_amr(amr_m2_kg: float | np.ndarray, q_pr: float = 1.0) -> float | np.ndarray:
    """Inverse of :func:`amr_from_beta`."""
    if q_pr <= 0:
        raise ValueError(f"q_pr must be positive, got {q_pr}")
    return np.asarray(amr_m2_kg, dtype=float) * q_pr / AMR_PER_BETA if isinstance(
        amr_m2_kg, np.ndarray) else amr_m2_kg * q_pr / AMR_PER_BETA


def amr_from_a1(a1_au_day2: float | np.ndarray, q_pr: float = 1.0) -> float | np.ndarray:
    """Area-to-mass ratio (m²/kg) implied directly by ``A1`` (au/day²).

    This is the whole channel in one line: ``AMR = 4.4137e6 * A1 / Q_pr``.
    Note it is **independent of the object's size and albedo** -- it depends
    only on the fitted acceleration.  That is what makes it a clean statistic.
    """
    return amr_from_beta(beta_from_a1(a1_au_day2), q_pr=q_pr)


def areal_density_from_a1(a1_au_day2: float, q_pr: float = 1.0) -> float:
    """Implied surface mass density m/A in kg/m² -- the Bialy & Loeb quantity.

    Simply ``1 / AMR``.  Bialy & Loeb 2018 quote 'Oumuamua at ~0.1 g/cm²
    (= 1 kg/m²); this function reproduces that from JPL's fitted A1.
    Returns ``inf`` for A1 = 0 and is undefined (nan) for A1 < 0.
    """
    amr = amr_from_a1(a1_au_day2, q_pr=q_pr)
    if amr <= 0:
        return float("nan") if amr < 0 else float("inf")
    return 1.0 / amr


def a1_to_m_s2(a1_au_day2: float | np.ndarray) -> float | np.ndarray:
    """Convert ``A1`` from au/day² to m/s² (the acceleration at 1 au)."""
    return np.asarray(a1_au_day2, dtype=float) * AU_DAY2_IN_M_S2 if isinstance(
        a1_au_day2, np.ndarray) else a1_au_day2 * AU_DAY2_IN_M_S2


# --- The natural-body comparison ---------------------------------------------
def amr_natural(diameter_m: float | np.ndarray,
                rho_kg_m3: float = 1000.0) -> float | np.ndarray:
    """Area-to-mass ratio of a solid sphere: ``3 / (2 D rho)`` m²/kg.

    A *low* assumed density raises this and therefore lowers R, so
    ``rho_kg_m3 = 1000`` (rubble pile / cometary) is the conservative default.
    """
    d = np.asarray(diameter_m, dtype=float) if isinstance(diameter_m, np.ndarray) \
        else float(diameter_m)
    if np.any(np.asarray(d) <= 0) or rho_kg_m3 <= 0:
        raise ValueError("diameter and density must be positive")
    return 3.0 / (2.0 * d * rho_kg_m3)


def diameter_from_h(h_mag: float | np.ndarray,
                    albedo: float = 0.15) -> float | np.ndarray:
    """Diameter in metres from absolute magnitude ``H`` and geometric albedo.

    The standard relation ``D[km] = 1329 / sqrt(p_V) * 10^(-H/5)``.  Note
    ``D ~ p^(-1/2)``, so a factor-4 albedo error is only a factor-2 diameter
    error -- and R is linear in D, so R is only weakly sensitive to the assumed
    albedo.  :func:`r_statistic` reports the bracketing interval explicitly.
    """
    if albedo <= 0 or albedo > 1:
        raise ValueError(f"albedo must be in (0, 1], got {albedo}")
    h = np.asarray(h_mag, dtype=float) if isinstance(h_mag, np.ndarray) else float(h_mag)
    return H_DIAMETER_CONST_KM / math.sqrt(albedo) * 10.0 ** (-h / 5.0) * 1000.0


# --- The discriminant ---------------------------------------------------------
@dataclass
class RStatistic:
    """Result of the normalised area-to-mass outlier test.

    Attributes
    ----------
    r
        ``AMR_implied / AMR_natural``.  ~1 for an ordinary body of the observed
        size; >> 1 cannot be one.
    r_lo, r_hi
        Bracketing values from the albedo interval, when the diameter came from
        H.  Equal to ``r`` when a measured diameter was used.
    amr_implied, amr_natural_
        The two area-to-mass ratios, m²/kg.
    beta
        The implied radiation-pressure beta.
    diameter_m
        Diameter actually used, metres.
    diameter_source
        ``"measured"`` (published diameter) or ``"H_albedo"`` (derived).
    areal_density_kg_m2
        Implied surface mass density, ``1 / amr_implied``.
    valid
        False when the statistic could not be formed (no A1, no size, or a
        cometary non-grav law).  ``reason`` says which.
    reason
        Human-readable degradation note.  Empty when ``valid``.
    """

    r: float = float("nan")
    r_lo: float = float("nan")
    r_hi: float = float("nan")
    amr_implied: float = float("nan")
    amr_natural_: float = float("nan")
    beta: float = float("nan")
    diameter_m: float = float("nan")
    diameter_source: str = ""
    areal_density_kg_m2: float = float("nan")
    valid: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "R": self.r, "R_lo": self.r_lo, "R_hi": self.r_hi,
            "amr_implied_m2_kg": self.amr_implied,
            "amr_natural_m2_kg": self.amr_natural_,
            "beta": self.beta,
            "diameter_m": self.diameter_m,
            "diameter_source": self.diameter_source,
            "areal_density_kg_m2": self.areal_density_kg_m2,
            "R_valid": self.valid, "R_reason": self.reason,
        }


def r_statistic(a1_au_day2: float | None,
                *,
                diameter_m: float | None = None,
                h_mag: float | None = None,
                albedo: float | None = None,
                albedo_assumed: float = 0.15,
                albedo_lo: float = 0.05,
                albedo_hi: float = 0.60,
                rho_kg_m3: float = 1000.0,
                q_pr: float = 1.0,
                nongrav_law_is_inverse_square: bool = True) -> RStatistic:
    """The normalised outlier statistic ``R = AMR_implied / AMR_natural(D, rho)``.

    Size precedence: a published ``diameter_m`` wins; else a measured ``albedo``
    with ``h_mag``; else ``h_mag`` with ``albedo_assumed``.  When the diameter
    had to be derived, ``r_lo``/``r_hi`` bracket the albedo range so the reader
    can see how much of R is assumption.

    Degrades honestly: returns ``valid=False`` with a ``reason`` rather than a
    fabricated number whenever an input is missing or the non-grav law is not
    the inverse-square one the conversion assumes.
    """
    if not nongrav_law_is_inverse_square:
        return RStatistic(valid=False,
                          reason="cometary g(r) fitted; A1 is not an SRP coefficient")
    if a1_au_day2 is None or not np.isfinite(a1_au_day2):
        return RStatistic(valid=False, reason="no fitted A1")
    if a1_au_day2 <= 0:
        # Radiation pressure cannot push sunward; R is meaningless.  Screen 3
        # handles these separately, so this is a clean, expected degradation.
        return RStatistic(beta=float(beta_from_a1(a1_au_day2)), valid=False,
                          reason="A1 <= 0; not attributable to radiation pressure")

    beta = float(beta_from_a1(a1_au_day2))
    amr_imp = float(amr_from_beta(beta, q_pr=q_pr))

    # --- size ---
    d_for_r_lo = d_for_r_hi = None
    if diameter_m is not None and np.isfinite(diameter_m) and diameter_m > 0:
        d = float(diameter_m)
        source = "measured"
    elif h_mag is not None and np.isfinite(h_mag):
        p = albedo if (albedo is not None and np.isfinite(albedo) and 0 < albedo <= 1) \
            else albedo_assumed
        source = "H_albedo_measured" if p is not albedo_assumed and albedo is not None \
            else "H_albedo_assumed"
        d = float(diameter_from_h(h_mag, albedo=p))
        if source == "H_albedo_assumed":
            # R = AMR_implied * 2 D rho / 3, so R is LINEAR IN D.  A darker
            # assumed albedo implies a LARGER body, which is MORE anomalous for
            # the same measured acceleration (more mass behind the same
            # cross-section).  So albedo_lo -> R_hi and albedo_hi -> R_lo.
            d_for_r_hi = float(diameter_from_h(h_mag, albedo=albedo_lo))
            d_for_r_lo = float(diameter_from_h(h_mag, albedo=albedo_hi))
    else:
        return RStatistic(beta=beta, amr_implied=amr_imp, valid=False,
                          reason="no diameter and no H; cannot normalise")

    amr_nat = float(amr_natural(d, rho_kg_m3=rho_kg_m3))
    r = amr_imp / amr_nat
    r_lo = (amr_imp / float(amr_natural(d_for_r_lo, rho_kg_m3=rho_kg_m3))
            if d_for_r_lo else r)
    r_hi = (amr_imp / float(amr_natural(d_for_r_hi, rho_kg_m3=rho_kg_m3))
            if d_for_r_hi else r)

    return RStatistic(r=r, r_lo=r_lo, r_hi=r_hi,
                      amr_implied=amr_imp, amr_natural_=amr_nat, beta=beta,
                      diameter_m=d, diameter_source=source,
                      areal_density_kg_m2=1.0 / amr_imp if amr_imp > 0 else float("inf"),
                      valid=True, reason="")


# --- Non-grav model interrogation --------------------------------------------
def nongrav_law_is_inverse_square(model_par_names) -> bool:
    """True when JPL's fitted non-grav model is the plain inverse-square law.

    ``model_par_names`` is the collection of parameter names JPL reports for the
    object (from ``sbdb.api``'s ``orbit.model_pars``).  Any of A1/A2/A3 alone
    means the default ``g(r) = (1 au / r)^2``.  The presence of ALN, NM, NN, NK,
    R0 or DT means a *cometary* g(r) was fitted, under which A1 is not a
    radiation-pressure coefficient and the conversion in this module is invalid.
    """
    if model_par_names is None:
        return True  # nothing reported -> assume JPL's asteroid default
    names = {str(n).strip().upper() for n in model_par_names}
    return not (names & COMETARY_MODEL_PARS)


# --- Reference table (the unit-test anchors) ---------------------------------
@dataclass(frozen=True)
class ReferenceObject:
    """A benchmark object with published or derived AMR/beta, for tests + docs."""
    name: str
    amr_m2_kg: float
    q_pr: float
    note: str
    beta: float = field(default=float("nan"))

    def __post_init__(self):
        object.__setattr__(self, "beta", float(beta_from_amr(self.amr_m2_kg, self.q_pr)))


#: The table in docs/derelict.md, as code.  Tests assert against these.
REFERENCE_OBJECTS: tuple[ReferenceObject, ...] = (
    ReferenceObject("natural_sphere_D100m_rho2000", amr_natural(100.0, 2000.0), 1.0,
                    "solid 100 m body, rho = 2000 kg/m^3"),
    ReferenceObject("natural_sphere_D10m_rho2000", amr_natural(10.0, 2000.0), 1.0,
                    "solid 10 m body, rho = 2000 kg/m^3"),
    ReferenceObject("oumuamua_if_pure_srp", 1.0837, 1.0,
                    "1I/'Oumuamua, A1 = 2.455e-7 au/d^2 (a = 4.92e-6 m/s^2 at 1 au, "
                    "Micheli et al. 2018); Bialy & Loeb 2018 quote m/A ~ 0.1 g/cm^2"),
    ReferenceObject("ikaros_sailcraft", 196.0 / 310.0, 2.0,
                    "JAXA IKAROS: 196 m^2 sail, 310 kg; specular reflector"),
    ReferenceObject("bare_mylar_1um", 1.0 / (1.0e-6 * 1400.0), 2.0,
                    "1 micron mylar film, rho = 1400 kg/m^3, reflective"),
)

#: 1I/'Oumuamua's radial non-grav coefficient in au/day^2, back-converted from
#: the acceleration Micheli et al. 2018 (Nature 559, 223) report at 1 au,
#: 4.92e-6 m/s^2.  Used as the calibration anchor for the whole conversion chain.
OUMUAMUA_A1_AU_DAY2 = 4.92e-6 / AU_DAY2_IN_M_S2
