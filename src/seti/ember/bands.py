"""Mid-infrared bandpasses and the cross-survey flux transformations EMBER needs.

EMBER compares an infrared *excess* measured by one instrument at one epoch with
the excess measured by a different instrument decades later. Doing that honestly
requires three things this module supplies:

1. **A response model per band.** Real relative system response (RSR) curves are
   used when they have been cached into ``seti/data_assets/rsr/`` (fetched from
   the SVO Filter Profile Service on the runner — the sandbox has no egress).
   When they are absent the module falls back to a documented **trapezoidal**
   approximation built from published 50%-response edges, and every result it
   produces is stamped ``rsr_source="trapezoid"`` so a reader knows which was
   used. It never silently pretends to precision it does not have.

2. **The quoting convention of each catalogue.** IRAS and AKARI quote
   monochromatic flux densities at a reference wavelength under the assumption
   that the source spectrum is ``nu*F_nu = const`` (``F_nu ~ nu^-1``); WISE
   magnitudes are zero-pointed assuming ``F_nu ~ nu^-2``; 2MASS is Vega-based.
   A catalogue flux is therefore **not** the flux at the reference wavelength
   unless the source happens to have the assumed spectrum. Ignoring this is the
   single easiest way to manufacture a spurious 9-to-12 micron "change".

3. **Band-to-band transfer for a given SED shape.** The photosphere transfers
   with one shape (a ~Rayleigh-Jeans stellar Planck function) and the excess
   transfers with another (a warm-dust Planck function). Those two transfer
   ratios are different, which is exactly why "9 vs 12 micron is not a null
   transformation".

Notation used throughout: ``Q(band, f)`` is the flux density a catalogue would
*quote* for a source whose true spectral shape is ``f(nu)``, normalised so that
``Q == f(nu_ref)`` when ``f`` equals the band's assumed spectrum. The ratio
``Q(b2, f) / Q(b1, f)`` is then the catalogue-to-catalogue transfer for a source
of shape ``f`` — the only quantity EMBER actually needs.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

# --- physical constants (CODATA / SI) --------------------------------------
H_PLANCK = 6.62607015e-34  # J s
K_BOLTZ = 1.380649e-23  # J / K
C_LIGHT = 2.99792458e8  # m / s

RSR_DIR = pathlib.Path(__file__).resolve().parents[1] / "data_assets" / "rsr"


@dataclass(frozen=True)
class Band:
    """A photometric band and the convention under which its catalogue quotes flux.

    Parameters
    ----------
    name
        EMBER's internal band key, e.g. ``"W3"`` or ``"S9W"``.
    lam_ref_um
        Reference / isophotal wavelength in microns, at which the catalogue
        quotes its monochromatic flux density.
    lam_lo_um, lam_hi_um
        Blue and red 50%-response edges in microns. Used only by the trapezoidal
        fallback response, and perturbed to estimate the bandpass systematic.
    zp_jy
        Vega zero-point flux density in Jy, for catalogues that publish
        magnitudes. ``None`` for catalogues (IRAS, AKARI) that publish Jy
        directly.
    alpha_assumed
        Exponent of the assumed source spectrum ``F_nu ~ nu**alpha`` used to
        define the quoted flux. ``-1`` is the ``nu*F_nu = const`` convention of
        IRAS and AKARI; ``-2`` is the WISE convention.
    vega_teff
        If set, the assumed spectrum is a Planck function at this temperature
        rather than a power law (the 2MASS/Vega case). Takes precedence over
        ``alpha_assumed``.
    sat_jy
        Approximate flux density at which the catalogue's photometry begins to
        saturate. ``None`` where saturation is not a practical concern.
    faint_5sig_jy
        Nominal 5-sigma point-source sensitivity, for funnel bookkeeping.
    beam_arcsec
        Characteristic beam / positional-confusion scale. For IRAS this is the
        *geometric mean* of the strongly elongated in-scan x cross-scan beam;
        ``beam_major_arcsec`` carries the long axis, which is what actually sets
        the blending radius.
    beam_major_arcsec
        Long axis of the beam in arcsec; defaults to ``beam_arcsec``.
    svo_id
        SVO Filter Profile Service identifier, used by the runner-side fetcher.
    """

    name: str
    lam_ref_um: float
    lam_lo_um: float
    lam_hi_um: float
    zp_jy: float | None = None
    alpha_assumed: float = -1.0
    vega_teff: float | None = None
    sat_jy: float | None = None
    faint_5sig_jy: float | None = None
    beam_arcsec: float = 6.0
    beam_major_arcsec: float | None = None
    svo_id: str = ""
    survey: str = ""
    epoch_year: float = 2000.0

    @property
    def blend_radius_arcsec(self) -> float:
        """Radius within which a second source contaminates this band's flux."""
        return 0.5 * (self.beam_major_arcsec or self.beam_arcsec)


# --------------------------------------------------------------------------
# Band table.
#
# Wavelengths, zero points and sensitivities are the published survey values;
# the 50%-response edges are the standard quoted band limits and are treated as
# uncertain at the +/-5% level (see ``EDGE_JITTER``). Beam sizes matter more
# than anything else in this table: the IRAS 12-micron beam is ~0.75' x 4.5',
# i.e. **two to three orders of magnitude more solid angle** than WISE W3, and
# that ratio is the dominant contamination term of the 27-year pair.
# --------------------------------------------------------------------------
BANDS: dict[str, Band] = {
    # --- 2MASS: the photospheric anchor (1997-2001) ------------------------
    "Ks": Band(
        name="Ks", lam_ref_um=2.159, lam_lo_um=2.028, lam_hi_um=2.320,
        zp_jy=666.7, vega_teff=9602.0, faint_5sig_jy=3.5e-4,
        beam_arcsec=2.5, svo_id="2MASS/2MASS.Ks", survey="2MASS", epoch_year=1999.5,
    ),
    "H": Band(
        name="H", lam_ref_um=1.662, lam_lo_um=1.513, lam_hi_um=1.798,
        zp_jy=1024.0, vega_teff=9602.0, faint_5sig_jy=3.0e-4,
        beam_arcsec=2.5, svo_id="2MASS/2MASS.H", survey="2MASS", epoch_year=1999.5,
    ),
    "J": Band(
        name="J", lam_ref_um=1.235, lam_lo_um=1.081, lam_hi_um=1.406,
        zp_jy=1594.0, vega_teff=9602.0, faint_5sig_jy=2.5e-4,
        beam_arcsec=2.5, svo_id="2MASS/2MASS.J", survey="2MASS", epoch_year=1999.5,
    ),
    # --- IRAS (1983): the 40-year lever arm --------------------------------
    # Beam: 0.75' x 4.5' at 12/25 um (45" x 270"). This is THE systematic.
    "I12": Band(
        name="I12", lam_ref_um=12.0, lam_lo_um=8.0, lam_hi_um=15.0,
        alpha_assumed=-1.0, faint_5sig_jy=0.4, sat_jy=None,
        beam_arcsec=110.0, beam_major_arcsec=270.0,
        svo_id="IRAS/IRAS.12mu", survey="IRAS", epoch_year=1983.5,
    ),
    "I25": Band(
        name="I25", lam_ref_um=25.0, lam_lo_um=19.0, lam_hi_um=30.0,
        alpha_assumed=-1.0, faint_5sig_jy=0.4, sat_jy=None,
        beam_arcsec=110.0, beam_major_arcsec=276.0,
        svo_id="IRAS/IRAS.25mu", survey="IRAS", epoch_year=1983.5,
    ),
    # --- AKARI/IRC (2006-07): the intermediate epoch -----------------------
    # Positions ~2"; unsaturated far above the WISE W3 saturation ceiling,
    # which is why AKARI is the arbiter for IRAS-bright sources.
    "S9W": Band(
        name="S9W", lam_ref_um=9.0, lam_lo_um=6.7, lam_hi_um=11.6,
        alpha_assumed=-1.0, faint_5sig_jy=0.05, sat_jy=180.0,
        beam_arcsec=5.5, svo_id="AKARI/IRC.S9W", survey="AKARI", epoch_year=2006.7,
    ),
    "L18W": Band(
        name="L18W", lam_ref_um=18.0, lam_lo_um=13.9, lam_hi_um=25.6,
        alpha_assumed=-1.0, faint_5sig_jy=0.09, sat_jy=90.0,
        beam_arcsec=5.7, svo_id="AKARI/IRC.L18W", survey="AKARI", epoch_year=2006.7,
    ),
    # --- WISE (2010 cryogenic): the only other epoch with 12/22 um ---------
    # NOTE: W3/W4 exist for the 2010 cryogenic phase ONLY. NEOWISE is W1/W2.
    "W1": Band(
        name="W1", lam_ref_um=3.3526, lam_lo_um=3.13, lam_hi_um=3.78,
        zp_jy=309.540, alpha_assumed=-2.0, sat_jy=0.175, faint_5sig_jy=5.4e-5,
        beam_arcsec=6.1, svo_id="WISE/WISE.W1", survey="WISE", epoch_year=2010.4,
    ),
    "W2": Band(
        name="W2", lam_ref_um=4.6028, lam_lo_um=4.02, lam_hi_um=5.19,
        zp_jy=171.787, alpha_assumed=-2.0, sat_jy=0.360, faint_5sig_jy=7.1e-5,
        beam_arcsec=6.4, svo_id="WISE/WISE.W2", survey="WISE", epoch_year=2010.4,
    ),
    "W3": Band(
        name="W3", lam_ref_um=11.5608, lam_lo_um=7.70, lam_hi_um=16.50,
        zp_jy=31.674, alpha_assumed=-2.0, sat_jy=0.957, faint_5sig_jy=7.3e-4,
        beam_arcsec=6.5, svo_id="WISE/WISE.W3", survey="WISE", epoch_year=2010.4,
    ),
    "W4": Band(
        name="W4", lam_ref_um=22.0883, lam_lo_um=19.60, lam_hi_um=23.20,
        zp_jy=8.363, alpha_assumed=-2.0, sat_jy=12.08, faint_5sig_jy=5.0e-3,
        beam_arcsec=12.0, svo_id="WISE/WISE.W4", survey="WISE", epoch_year=2010.4,
    ),
}

#: Fractional perturbation applied to the 50%-response edges when estimating the
#: bandpass systematic. The trapezoidal approximation is not exact; this bounds
#: how much the transfer ratio can move if the true response differs.
EDGE_JITTER = 0.05

#: The three epochs that carry 12-25 micron information. NEOWISE is deliberately
#: absent: it flies W1/W2 only, so it cannot see 100-300 K waste heat at all.
EPOCH_LADDER: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    ("IRAS", 1983.5, ("I12", "I25")),
    ("AKARI", 2006.7, ("S9W", "L18W")),
    ("WISE", 2010.4, ("W3", "W4")),
)


# --------------------------------------------------------------------------
# Spectral shapes
# --------------------------------------------------------------------------
def planck_fnu(lam_um: np.ndarray | float, temp_k: float) -> np.ndarray:
    """Planck ``B_nu`` in arbitrary units, as a function of wavelength in microns.

    Only *ratios* of this function are ever used, so the normalisation is
    irrelevant. Overflow in the Wien tail is clipped rather than allowed to
    produce ``inf``/``nan``.
    """
    lam = np.atleast_1d(np.asarray(lam_um, dtype=float)) * 1e-6
    nu = C_LIGHT / lam
    x = np.clip(H_PLANCK * nu / (K_BOLTZ * float(temp_k)), 1e-12, 700.0)
    return nu**3 / np.expm1(x)


def power_law_fnu(lam_um: np.ndarray | float, alpha: float) -> np.ndarray:
    """``F_nu ~ nu**alpha`` evaluated on a wavelength grid in microns."""
    lam = np.atleast_1d(np.asarray(lam_um, dtype=float))
    nu = C_LIGHT / (lam * 1e-6)
    return nu**alpha


#: Vega's effective temperature, used for the blackbody stand-in below.
VEGA_TEFF = 9602.0
#: The 2MASS Ks-band definition of Vega's monochromatic flux density
#: (Cohen et al. 2003): 666.7 Jy at the isophotal wavelength 2.159 micron.
#: This single number fixes the absolute normalisation of the Vega stand-in, so
#: the Vega-magnitude and Jy-native systems can be mixed in one transfer ratio.
VEGA_NORM_LAM_UM = 2.159
VEGA_NORM_JY = 666.7


def vega_fnu_jy(lam_um: np.ndarray | float) -> np.ndarray:
    """Vega's flux density in Jy, approximated by a normalised 9602 K blackbody.

    Accurate to a few percent through the near- and mid-infrared, which is all
    that is required here: the residual is measured by :func:`vega_consistency`
    and, more importantly, is absorbed entirely by the *empirical* photospheric
    colour locus that ``crossepoch`` fits from the data. The blackbody stand-in
    is used only for the dust-excess transfer, where no empirical locus exists.

    Vega's own 12-25 micron debris excess (~1% at 12 micron, larger at 25) is a
    known wrinkle in every mid-IR Vega system and is not corrected here.
    """
    scale = VEGA_NORM_JY / float(planck_fnu(VEGA_NORM_LAM_UM, VEGA_TEFF)[0])
    return planck_fnu(lam_um, VEGA_TEFF) * scale


def assumed_spectrum(band: Band, lam_um: np.ndarray) -> np.ndarray:
    """The spectral shape the catalogue assumed when it quoted its fluxes.

    For Vega-magnitude catalogues (2MASS, WISE) this is Vega itself **in Jy** —
    not a power law. Getting this wrong is a subtle but large error: treating
    WISE as if its zero point were literally defined on an ``F_nu ~ nu^-2``
    spectrum introduces a spurious ~1.5x colour term in W3, because W3 is broad
    enough that ``nu^2`` and ``nu^-2`` weightings differ substantially across it.
    The ``alpha_assumed`` convention applies only to the Jy-native catalogues
    (IRAS, AKARI), which really do quote ``nu*F_nu = const`` flux densities.
    """
    if band.zp_jy is not None:
        return vega_fnu_jy(lam_um)
    return power_law_fnu(lam_um, band.alpha_assumed)


# --------------------------------------------------------------------------
# Response curves
# --------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _cached_rsr(name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a real RSR curve from ``data_assets/rsr/<name>.csv`` if present.

    File format: two columns ``lam_um,response`` with a one-line header. The
    runner-side fetcher writes these from the SVO Filter Profile Service; the
    sandbox has no egress so they may legitimately be absent.
    """
    path = RSR_DIR / f"{name}.csv"
    if not path.exists():
        return None
    try:
        arr = np.genfromtxt(path, delimiter=",", names=True)
        lam = np.asarray(arr["lam_um"], dtype=float)
        resp = np.asarray(arr["response"], dtype=float)
    except Exception:  # noqa: BLE001 - a malformed cache must not be fatal
        return None
    good = np.isfinite(lam) & np.isfinite(resp) & (resp >= 0)
    if good.sum() < 4:
        return None
    order = np.argsort(lam[good])
    return lam[good][order], resp[good][order]


def response(band: Band, lam_um: np.ndarray, edge_scale: float = 1.0) -> np.ndarray:
    """Relative system response of ``band`` on the wavelength grid ``lam_um``.

    Uses the cached SVO curve when available, otherwise a trapezoid that is flat
    across the 50%-response range and falls linearly to zero over a 15% wing on
    each side. ``edge_scale`` widens (>1) or narrows (<1) the band about its
    centre and is used to propagate the bandpass systematic.
    """
    lam = np.atleast_1d(np.asarray(lam_um, dtype=float))
    real = _cached_rsr(band.name)
    if real is not None and edge_scale == 1.0:
        return np.interp(lam, real[0], real[1], left=0.0, right=0.0)

    centre = 0.5 * (band.lam_lo_um + band.lam_hi_um)
    half = 0.5 * (band.lam_hi_um - band.lam_lo_um) * edge_scale
    lo, hi = centre - half, centre + half
    wing = 0.15 * (hi - lo)
    out = np.zeros_like(lam)
    flat = (lam >= lo) & (lam <= hi)
    out[flat] = 1.0
    blue = (lam >= lo - wing) & (lam < lo)
    out[blue] = (lam[blue] - (lo - wing)) / wing
    red = (lam > hi) & (lam <= hi + wing)
    out[red] = ((hi + wing) - lam[red]) / wing
    return out


def rsr_source(band: Band) -> str:
    """``"svo"`` if a real response curve is in use, else ``"trapezoid"``."""
    return "svo" if _cached_rsr(band.name) is not None else "trapezoid"


def _grid(band: Band, edge_scale: float = 1.0, n: int = 512) -> np.ndarray:
    real = _cached_rsr(band.name)
    if real is not None and edge_scale == 1.0:
        lo, hi = float(real[0][0]), float(real[0][-1])
    else:
        centre = 0.5 * (band.lam_lo_um + band.lam_hi_um)
        half = 0.5 * (band.lam_hi_um - band.lam_lo_um) * edge_scale * 1.35
        lo, hi = centre - half, centre + half
    return np.linspace(max(lo, 0.05), hi, n)


def quoted_flux_ratio(band: Band, shape, edge_scale: float = 1.0) -> float:
    """``Q(band, f)``: what the catalogue quotes for a source of shape ``f``.

    ``shape`` is a callable mapping a wavelength grid in microns to relative
    ``F_nu``. The return value is normalised so that a source whose spectrum
    *equals* the band's assumed spectrum yields exactly ``shape(lam_ref)``.

    The integral is performed in frequency, following the IRAS Explanatory
    Supplement definition of a colour correction:

        Q = integral[ f(nu) R(nu) dnu ] / integral[ S(nu) R(nu) dnu ]

    with ``S`` the assumed spectrum normalised to unity at the reference
    wavelength.
    """
    lam = _grid(band, edge_scale)
    resp = response(band, lam, edge_scale)
    nu = C_LIGHT / (lam * 1e-6)
    order = np.argsort(nu)
    nu_s, resp_s = nu[order], resp[order]

    f_vals = np.asarray(shape(lam), dtype=float)[order]
    s_vals = assumed_spectrum(band, lam)[order]
    s_ref = float(assumed_spectrum(band, np.array([band.lam_ref_um]))[0])
    s_vals = s_vals / s_ref

    num = np.trapezoid(f_vals * resp_s, nu_s)
    den = np.trapezoid(s_vals * resp_s, nu_s)
    if den == 0 or not np.isfinite(den):
        return float("nan")
    return float(num / den)


@lru_cache(maxsize=65536)
def _q_planck(band_name: str, temp_k: float, edge_scale: float) -> float:
    """Cached ``Q(band, blackbody(T))``.

    The cessation statistic evaluates this on a fixed temperature grid for every
    source in the catalogue, so without memoisation the band integrals dominate
    the runtime by orders of magnitude. The grid points are identical floats
    across sources, which makes the cache hit essentially always.
    """
    band = BANDS[band_name]
    return quoted_flux_ratio(band, lambda lam: planck_fnu(lam, temp_k), edge_scale)


def transfer(band_from: Band, band_to: Band, temp_k: float,
             edge_scale: float = 1.0) -> float:
    """Catalogue-to-catalogue flux ratio for a blackbody of temperature ``temp_k``.

    Returns ``Q(band_to) / Q(band_from)``. Multiply a flux density quoted in
    ``band_from`` by this to obtain the flux density the ``band_to`` catalogue
    would quote for the *same physical source*, including both bands' colour
    corrections.

    For a hot (Rayleigh-Jeans) source this tends to the ratio of squared
    reference wavelengths; for a 200 K source it does not, which is exactly why
    the transformation cannot be treated as null.
    """
    q_from = _q_planck(band_from.name, float(temp_k), float(edge_scale))
    q_to = _q_planck(band_to.name, float(temp_k), float(edge_scale))
    if not np.isfinite(q_from) or q_from == 0:
        return float("nan")
    return float(q_to / q_from)


def transfer_with_systematic(band_from: Band, band_to: Band, temp_k: float
                             ) -> tuple[float, float]:
    """``transfer`` plus the bandpass systematic from perturbing the band edges.

    Returns ``(ratio, sigma_ratio)``. The systematic is the half-range of the
    ratio over edge scalings of ``1 -/+ EDGE_JITTER`` applied independently to
    the two bands (worst case of the four combinations), which is a deliberately
    conservative bound on the trapezoid's inadequacy.
    """
    nominal = transfer(band_from, band_to, temp_k)
    vals = []
    for s_from in (1 - EDGE_JITTER, 1 + EDGE_JITTER):
        for s_to in (1 - EDGE_JITTER, 1 + EDGE_JITTER):
            q_from = _q_planck(band_from.name, float(temp_k), float(s_from))
            q_to = _q_planck(band_to.name, float(temp_k), float(s_to))
            if np.isfinite(q_from) and q_from != 0:
                vals.append(q_to / q_from)
    if not vals:
        return nominal, float("nan")
    return nominal, float(0.5 * (max(vals) - min(vals)))


def photosphere_transfer(band_from: Band, band_to: Band, teff_k: float) -> float:
    """Band-to-band ratio for a stellar photosphere at ``teff_k``.

    A bare photosphere is Rayleigh-Jeans across the whole mid-infrared, so this
    is close to ``(lam_from / lam_to)**2`` and only weakly dependent on
    ``teff_k`` — which is what makes the anchor robust. Molecular opacity (SiO
    at 8 micron, CO) breaks the approximation at the 5-10% level for M dwarfs
    and giants; that is carried as ``PHOT_SYS_FRAC`` in ``crossepoch``, not
    modelled here.
    """
    return transfer(band_from, band_to, teff_k)


def mag_to_jy(band: Band, mag: float | np.ndarray) -> np.ndarray:
    """Vega magnitude to flux density in Jy. Raises for Jy-native catalogues."""
    if band.zp_jy is None:
        raise ValueError(f"band {band.name} publishes Jy directly, not magnitudes")
    return band.zp_jy * 10.0 ** (-0.4 * np.asarray(mag, dtype=float))


def magerr_to_jyerr(band: Band, mag: float | np.ndarray,
                    magerr: float | np.ndarray) -> np.ndarray:
    """Propagate a magnitude uncertainty to a flux-density uncertainty in Jy."""
    flux = mag_to_jy(band, mag)
    return flux * np.asarray(magerr, dtype=float) * np.log(10.0) / 2.5


def saturated(band: Band, flux_jy: float | np.ndarray) -> np.ndarray:
    """Boolean: is this flux at or above the band's saturation onset?"""
    if band.sat_jy is None:
        return np.zeros_like(np.atleast_1d(np.asarray(flux_jy, float)), dtype=bool)
    return np.atleast_1d(np.asarray(flux_jy, dtype=float)) >= band.sat_jy


@dataclass(frozen=True)
class PairAudit:
    """Quantitative verdict on one early-epoch to late-epoch band pair."""

    early: str
    late: str
    baseline_yr: float
    transfer_300k: float
    transfer_1000k: float
    transfer_spread: float
    bandpass_sys_frac: float
    beam_area_ratio: float
    usable_flux_lo_jy: float
    usable_flux_hi_jy: float
    verdict: str
    notes: str = field(default="")


def audit_pair(early_key: str, late_key: str) -> PairAudit:
    """Compute the systematics numbers that decide whether a pair is usable.

    The three quantities that actually kill epoch pairs, in order:

    * **beam_area_ratio** -- how many times more sky the early beam covers. IRAS
      at 12 micron covers ~1400x the solid angle of WISE W3, so an IRAS flux is
      the sum over everything in that footprint. The mitigation is the
      beam-summed comparison in ``crossepoch.beam_sum_consistency``, not a
      brighter cut.
    * **transfer_spread** -- how much the early-to-late flux ratio moves as the
      dust temperature runs over 150-1500 K. Near 1 the transformation is
      effectively null and the pair is well conditioned.
    * **the usable flux window** -- bounded below by the early survey's
      sensitivity and above by the late band's saturation. If the window is
      empty or inverted, the pair cannot be used at all.
    """
    b_e, b_l = BANDS[early_key], BANDS[late_key]
    t300 = transfer(b_e, b_l, 300.0)
    t1000 = transfer(b_e, b_l, 1000.0)
    grid = [transfer(b_e, b_l, t) for t in (150.0, 200.0, 300.0, 500.0, 1000.0, 1500.0)]
    grid = [g for g in grid if np.isfinite(g)]
    spread = (max(grid) / min(grid)) if grid else float("nan")
    _, sys_abs = transfer_with_systematic(b_e, b_l, 300.0)
    sys_frac = sys_abs / t300 if (np.isfinite(t300) and t300) else float("nan")

    area_ratio = (b_e.beam_arcsec**2) / (b_l.beam_arcsec**2)
    lo = b_e.faint_5sig_jy or 0.0
    # Saturation of the LATE band, expressed in EARLY-band flux units.
    hi = float("inf") if b_l.sat_jy is None else b_l.sat_jy / t300

    if not np.isfinite(hi) or hi > lo * 2.0:
        window = "usable"
    elif hi > lo:
        window = "narrow"
    else:
        window = "empty"

    if window == "empty":
        verdict = "REJECT: saturation ceiling below sensitivity floor"
    elif area_ratio > 100 and window == "narrow":
        verdict = "CONDITIONAL: severe blending AND a narrow flux window"
    elif area_ratio > 100:
        verdict = "CONDITIONAL: requires beam-summed late-epoch comparison"
    elif spread > 2.0:
        verdict = "CONDITIONAL: transfer strongly temperature-dependent"
    else:
        verdict = "USABLE"

    return PairAudit(
        early=early_key, late=late_key,
        baseline_yr=round(b_l.epoch_year - b_e.epoch_year, 1),
        transfer_300k=float(t300), transfer_1000k=float(t1000),
        transfer_spread=float(spread), bandpass_sys_frac=float(sys_frac),
        beam_area_ratio=float(area_ratio),
        usable_flux_lo_jy=float(lo), usable_flux_hi_jy=float(hi),
        verdict=verdict,
        notes=f"rsr={rsr_source(b_e)}/{rsr_source(b_l)}",
    )


#: The pairs EMBER evaluates. Ordered by scientific value of the baseline.
CANDIDATE_PAIRS: tuple[tuple[str, str], ...] = (
    ("I12", "W3"),    # 27 yr, near-identical bandpasses -- the enabling pair
    ("I25", "W4"),    # 27 yr, wider saturation headroom
    ("I12", "S9W"),   # 23 yr, AKARI arbitrates where W3 saturates
    ("I25", "L18W"),  # 23 yr
    ("S9W", "W3"),    # 4 yr, cleanest astrometry, shortest lever arm
    ("L18W", "W4"),   # 4 yr
)
