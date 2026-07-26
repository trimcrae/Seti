"""Contamination funnel for EMBER candidates.

The rules are ordered by the *measured* damage each contaminant does, taken from
the prior-art sweep rather than invented here. Anything that survives all of
them is a source whose mid-infrared excess was real in 1983 or 2006, is
demonstrably absent in 2010, and has no instrumental or ordinary astrophysical
explanation left.

Ranking, worst first:

1. **IRAS beam blending.** WISE has ~5x better angular resolution and ~1000x
   better sensitivity, so a single IRAS source routinely resolves into several
   WISE sources and the "excess" belongs to a neighbour. Handled upstream by
   ``crossepoch.beam_sum_consistency``; this module enforces its verdict.
2. **Cirrus.** The IRAS catalogues carry CIRR1/2/3 precisely because
   cirrus-knot contamination was endemic. Cut on them.
3. **Wrong optical association.** The IRAS error ellipse is tens of arcsec;
   the brightest optical source in it is often not the infrared source.
4. **Low-S/N spurious IRAS sources**, plus the flux-limited Eddington bias
   (corrected in ``crossepoch``, thresholded here).
5. **Genuine variables.** AGB/Mira/YSO stars vary hugely in the mid-infrared.
   Three independent handles: the IRAS catalogue's own VAR index (which Carrigan
   1983-era work already used as a filter), Gaia DR3 variability, and the
   position on the colour-magnitude diagram.
6. **WISE saturation.** A saturated W3/W4 measurement under-reports flux and
   manufactures exactly the signal being hunted. Bright IRAS sources are
   precisely the ones at risk, so this is a structural tension, not an edge case.
7. **Solar-system objects.** IRAS, AKARI and WISE all catalogued moving objects;
   an asteroid seen once and never again is a perfect false cessation.
8. **Background galaxies.** Hot dust-obscured galaxies and dusty starbursts
   destroyed *every* Project Hephaistos candidate -- JWST/MIRI resolved
   candidates D and E as a Hot DOG at z~0.9 and a starburst at z~0.4, both
   within ~1 arcsec. The defence is astrometric and decisive: a galaxy has no
   parallax and no proper motion.
9. **Youth.** This is the astrophysical discriminant that matters most. Every
   known analogue of the signature -- TYC 8241 2652 1 and the extreme debris
   disks, of which 14 of 17 varied at 3-5 micron between 2010 and 2019 -- sits
   around a *young* star with a collisional dust reservoir. A cessation around
   an old, kinematically-heated main-sequence star has no such reservoir and is
   the only version of this signature that is not already explained.

Every function is pure and takes a plain dict, so the whole funnel is testable
with no network.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- thresholds (mirrored into config/thresholds.yaml under `ember:`) -------
CIRR_MAX = 3  # IRAS CIRR2/CIRR3 index above which the field is cirrus-dominated
#: Kennedy & Wyatt (2012, arXiv:1207.0521) found ~8,000 of 180,000 stars with an
#: apparent IRAS excess (mostly at 12 micron) whose sky positions correlate with
#: the 100-micron background -- i.e. the excess is cirrus, not a disc. Cutting at
#: 100-micron surface brightness < 5 MJy/sr leaves 271 of 180,000. That single
#: number takes the per-star false-excess rate from 4.4e-2 to 1.5e-3, a factor of
#: ~30, and it is the most valuable published cut available to this channel.
IRAS_BACKGROUND_100UM_MAX = 5.0  # MJy/sr
IRAS_SNR_MIN = 7.0  # early-epoch flux S/N; also bounds the Eddington deboost
IRAS_QUAL_MIN = 3  # 3 = high quality, 2 = moderate, 1 = upper limit
PLX_SNR_MIN = 8.0  # parallax significance required to call it a star
PM_SNR_MIN = 5.0  # proper-motion significance; a galaxy has none
MATCH_RADIUS_ARCSEC = 2.0  # optical-to-infrared association at the WISE epoch
VAR_INDEX_MAX = 50  # IRAS VAR: percentage probability of variability
GAIA_VAR_AMP_MAX = 0.05  # mag; optical constancy required
#: NEOWISE W1 flatness after the drop. Every persistent natural variable class
#: exceeds this; the one known step-and-stay object does not.
NEOWISE_RMS_MAX = 0.08  # mag
NEOWISE_SLOPE_SIG_MAX = 3.0  # sigma on a secular W1 trend
#: Published empirical floor for the IRAS(1983)-to-WISE(2010) comparison:
#: HD 172555 was stable to within 4% over those 27 years (arXiv:1210.6258).
#: Any claimed fade must be large compared with this, not merely significant.
CROSS_EPOCH_FLOOR_FRAC = 0.04
F_CESS_MIN = 0.5  # a "cessation" must lose at least half the excess
TANGENTIAL_V_MIN_KMS = 30.0  # kinematic age proxy for the "old host" cut
YOUNG_NIR_EXCESS_MAX = 0.15  # mag; H-Ks excess flags a YSO/disc
W1W2_BLEND_FLOOR = -0.05  # negative W1-W2 is a blend, not a photosphere
ABS_G_GIANT_MAX = 3.5  # M_G brighter than this with a red colour = giant/AGB


@dataclass
class VetResult:
    """Outcome of the funnel for a single source."""

    source_id: str
    passed: bool
    stage_failed: str | None
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_run: int = 0
    checks_possible: int = 0
    coverage: dict = field(default_factory=dict)

    @property
    def coverage_str(self) -> str:
        return f"clean_in_{self.checks_run}_of_{self.checks_possible}_observed_channels"


def _f(row: dict, key: str, default: float = np.nan) -> float:
    v = row.get(key, default)
    try:
        out = float(v)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


# --------------------------------------------------------------------------
# Individual rules. Each returns (verdict, reason).
#   verdict True  -> passed this check
#   verdict False -> rejected
#   verdict None  -> not testable with the data present (counts against coverage)
# --------------------------------------------------------------------------
def check_beam_blending(row: dict) -> tuple[bool | None, str]:
    """Reject when the early flux is accounted for by the whole beam's contents."""
    if "beam_explained" not in row:
        return None, "no beam-sum test available"
    if bool(row["beam_explained"]):
        n = row.get("n_neighbours", "?")
        return False, (f"early flux explained by {n} late-epoch sources summed over "
                       "the early beam: this is resolution, not a fade")
    return True, "early flux exceeds the summed late-epoch flux in the beam"


def check_cirrus(row: dict) -> tuple[bool | None, str]:
    """Reject cirrus-dominated fields using the IRAS CIRR indices."""
    c2, c3 = row.get("cirr2"), row.get("cirr3")
    if c2 is None and c3 is None:
        return None, "no cirrus flags"
    worst = max([v for v in (c2, c3) if v is not None], default=0)
    if worst > CIRR_MAX:
        return False, f"cirrus index {worst} > {CIRR_MAX}"
    return True, f"cirrus index {worst} acceptable"


def check_far_ir_background(row: dict) -> tuple[bool | None, str]:
    """The single highest-yield cut available: IRAS 100-micron surface brightness.

    Kennedy & Wyatt (2012) showed that the ~8,000 apparent IRAS excesses among
    180,000 stars are positionally correlated with the 100-micron background, and
    that requiring that background to be below 5 MJy/sr leaves 271. The
    surviving number counts then match extragalactic counts, meaning even most
    of *those* are background sources rather than discs.

    This is a factor of ~30 on the dominant contaminant for one number, so it
    runs early and is not optional for any IRAS-based pair.
    """
    bkg = _f(row, "iras_100um_bkg_mjysr")
    if not np.isfinite(bkg):
        return None, "no 100-micron background measurement"
    if bkg > IRAS_BACKGROUND_100UM_MAX:
        return False, (f"IRAS 100 um background {bkg:.1f} MJy/sr > "
                       f"{IRAS_BACKGROUND_100UM_MAX}: the apparent excess is "
                       "cirrus (Kennedy & Wyatt 2012)")
    return True, f"IRAS 100 um background {bkg:.1f} MJy/sr: low-cirrus field"


def check_late_epoch_flat(row: dict) -> tuple[bool | None, str]:
    """Require the *late* epoch to be photometrically flat -- the class discriminant.

    This is the most discriminating rule in the funnel and it comes straight out
    of the confounder literature. Every natural class that varies in the
    mid-infrared varies *persistently*: 14 of 17 extreme debris disks changed at
    3-5 micron between 2010 and 2019 and 5 of 6 varied on sub-year timescales;
    R Coronae Borealis stars swing by factors of up to 10 between the IRAS,
    AKARI and WISE epochs; ~85% of white-dwarf discs vary on all timescales; YSOs
    are variable at the tens-of-percent level.

    None of them produce a single monotonic step down followed by a decade of
    flatness. TYC 8241 2652 1 is the sole known object that does -- its WISE
    fluxes showed no significant change between 2010 and 2019 after its earlier
    collapse -- and it remains unexplained fourteen years later.

    So NEOWISE earns its place here not as a cessation measurement -- it flies
    W1/W2 only and cannot see 100-300 K dust at all -- but as the stability
    requirement that separates a step from a wobble.
    """
    rms = _f(row, "neowise_w1_rms_mag")
    slope = _f(row, "neowise_w1_slope_mag_yr")
    slope_sig = _f(row, "neowise_w1_slope_sigma")
    n_ep = row.get("neowise_n_epochs")
    if not np.isfinite(rms) and not np.isfinite(slope):
        return None, "no NEOWISE light curve"
    if n_ep is not None and int(n_ep) < 20:
        return None, f"only {n_ep} NEOWISE epochs: too few to test flatness"
    if np.isfinite(rms) and rms > NEOWISE_RMS_MAX:
        return False, (f"NEOWISE W1 rms {rms:.3f} mag > {NEOWISE_RMS_MAX}: "
                       "persistently variable, like every natural confounder")
    if np.isfinite(slope) and np.isfinite(slope_sig) and slope_sig > 0:
        if abs(slope / slope_sig) > NEOWISE_SLOPE_SIG_MAX:
            return False, (f"NEOWISE W1 secular slope {slope:.4f} mag/yr at "
                           f"{abs(slope / slope_sig):.1f} sigma: still evolving, "
                           "not a completed step")
    return True, f"NEOWISE W1 flat (rms {rms:.3f} mag) after the drop: step-and-stay"


def check_early_quality(row: dict) -> tuple[bool | None, str]:
    """Require a high-quality, high-S/N early detection, not an upper limit."""
    qual = row.get("iras_qual")
    snr = _f(row, "early_snr")
    if qual is None and not np.isfinite(snr):
        return None, "no early-epoch quality information"
    if qual is not None and int(qual) < IRAS_QUAL_MIN:
        return False, f"early flux quality {qual} < {IRAS_QUAL_MIN} (upper limit or moderate)"
    if np.isfinite(snr) and snr < IRAS_SNR_MIN:
        return False, (f"early flux S/N {snr:.1f} < {IRAS_SNR_MIN}: inside the regime "
                       "where Eddington bias alone manufactures a fade")
    return True, f"early detection quality {qual}, S/N {snr:.1f}"


def check_variability(row: dict) -> tuple[bool | None, str]:
    """Reject known variables: IRAS VAR index, Gaia variability, optical amplitude.

    A mid-infrared excess that comes and goes on an AGB star is not a
    technosignature, it is a pulsating star with a dust shell. Optical
    constancy is required because dust-driven mid-IR variability is almost
    always accompanied by optical variability in these populations.
    """
    var = row.get("iras_var")
    gvar = row.get("gaia_variable")
    amp = _f(row, "optical_amp_mag")
    if var is None and gvar is None and not np.isfinite(amp):
        return None, "no variability information"
    if var is not None and float(var) > VAR_INDEX_MAX:
        return False, f"IRAS VAR index {var} > {VAR_INDEX_MAX}"
    if gvar is not None and bool(gvar):
        return False, "flagged variable in Gaia DR3"
    if np.isfinite(amp) and amp > GAIA_VAR_AMP_MAX:
        return False, (f"optical amplitude {amp:.3f} mag > {GAIA_VAR_AMP_MAX}: "
                       "mid-IR change accompanied by optical change")
    return True, "no variability flagged in any available channel"


def check_saturation(row: dict) -> tuple[bool | None, str]:
    """Reject when the *late* band is saturated -- it under-reports and fakes a fade."""
    late_sat = row.get("late_saturated")
    early_sat = row.get("early_saturated")
    if late_sat is None and early_sat is None:
        return None, "no saturation information"
    if late_sat:
        return False, "late-epoch band at or above saturation onset: flux under-reported"
    if early_sat:
        return False, "early-epoch band saturated: excess amplitude unreliable"
    return True, "neither epoch saturated"


def check_solar_system(row: dict) -> tuple[bool | None, str]:
    """Reject catalogued moving objects -- the perfect false cessation."""
    flag = row.get("solar_system_assoc")
    if flag is None:
        return None, "no solar-system association check"
    if bool(flag):
        return False, "associated with a catalogued solar-system object"
    return True, "not a known moving object"


def check_stellar_astrometry(row: dict) -> tuple[bool | None, str]:
    """Require parallax and proper motion -- the decisive background-galaxy veto.

    Hot DOGs and dusty starbursts accounted for every Project Hephaistos
    candidate that was followed up. They are also, without exception,
    astrometrically stationary and at zero parallax. Requiring a significant
    Gaia parallax *and* a significant proper motion removes the entire
    extragalactic contaminant class in one cut, at the cost of restricting the
    search to stars -- which is what the signature is about anyway.
    """
    plx_snr = _f(row, "parallax_over_error")
    pm_snr = _f(row, "pm_over_error")
    if not np.isfinite(plx_snr) and not np.isfinite(pm_snr):
        return None, "no Gaia astrometry"
    if np.isfinite(plx_snr) and plx_snr < PLX_SNR_MIN:
        return False, (f"parallax significance {plx_snr:.1f} < {PLX_SNR_MIN}: "
                       "not established as a star (background galaxy risk)")
    if np.isfinite(pm_snr) and pm_snr < PM_SNR_MIN:
        return False, f"proper-motion significance {pm_snr:.1f} < {PM_SNR_MIN}"
    return True, f"parallax {plx_snr:.0f} sigma, proper motion {pm_snr:.0f} sigma"


def check_association(row: dict) -> tuple[bool | None, str]:
    """Require the optical star to sit inside the infrared position at its epoch."""
    sep = _f(row, "sep_arcsec")
    n_opt = row.get("n_optical_in_beam")
    if not np.isfinite(sep):
        return None, "no positional separation"
    if sep > MATCH_RADIUS_ARCSEC:
        return False, f"optical-IR separation {sep:.2f}\" > {MATCH_RADIUS_ARCSEC}\""
    if n_opt is not None and int(n_opt) > 1:
        return False, (f"{n_opt} optical sources inside the infrared beam: "
                       "association ambiguous")
    return True, f"unique optical counterpart at {sep:.2f}\""


def check_blend(row: dict) -> tuple[bool | None, str]:
    """Negative W1-W2 is a blend, not a photosphere (inherited ledger rule)."""
    w1w2 = _f(row, "w1_w2")
    if not np.isfinite(w1w2):
        return None, "no W1-W2 colour"
    if w1w2 < W1W2_BLEND_FLOOR:
        return False, f"W1-W2 = {w1w2:.3f} < {W1W2_BLEND_FLOOR}: blended photometry"
    return True, f"W1-W2 = {w1w2:.3f} consistent with a photosphere"


def check_not_young(row: dict) -> tuple[bool | None, str]:
    """The astrophysical discriminant: require a mature, kinematically old host.

    Every known natural analogue of a vanishing mid-infrared excess is young.
    TYC 8241 2652 1 is a ~10 Myr pre-main-sequence star in Sco-Cen; the extreme
    debris disks whose 3-5 micron flux varied in 14 of 17 cases are 10-200 Myr
    A-G stars with collisional cascades. Those systems *have* a dust reservoir
    that can be created and destroyed on year timescales.

    An old thin- or thick-disk star does not. Requiring youth indicators to be
    absent -- no near-infrared excess, no chromospheric or X-ray activity, no
    young-moving-group membership, and a tangential velocity indicating
    kinematic heating -- removes the entire known analogue population while
    keeping exactly the regime in which no explanation exists.
    """
    reasons = []
    nir_exc = _f(row, "h_ks_excess")
    if np.isfinite(nir_exc) and nir_exc > YOUNG_NIR_EXCESS_MAX:
        reasons.append(f"H-Ks excess {nir_exc:.3f} mag indicates an inner disc")
    if row.get("young_moving_group"):
        reasons.append(f"member of young moving group {row['young_moving_group']}")
    if row.get("xray_active"):
        reasons.append("X-ray active (youth indicator)")
    if row.get("star_forming_region"):
        reasons.append("projected on a star-forming region")
    v_tan = _f(row, "v_tan_kms")
    if np.isfinite(v_tan) and v_tan < TANGENTIAL_V_MIN_KMS:
        reasons.append(f"tangential velocity {v_tan:.0f} km/s < {TANGENTIAL_V_MIN_KMS}: "
                       "kinematically young")
    tested = (np.isfinite(nir_exc) or np.isfinite(v_tan)
              or any(k in row for k in ("young_moving_group", "xray_active",
                                        "star_forming_region")))
    if not tested:
        return None, "no youth indicators available"
    if reasons:
        return False, "; ".join(reasons)
    return True, "no youth indicator triggered; kinematically mature host"


def check_not_evolved(row: dict) -> tuple[bool | None, str]:
    """Reject AGB stars and red giants, whose dust shells vary by construction."""
    abs_g = _f(row, "abs_g")
    bp_rp = _f(row, "bp_rp")
    if not np.isfinite(abs_g) or not np.isfinite(bp_rp):
        return None, "no colour-magnitude position"
    if abs_g < ABS_G_GIANT_MAX and bp_rp > 1.0:
        return False, (f"M_G = {abs_g:.2f}, BP-RP = {bp_rp:.2f}: luminous and red, "
                       "i.e. on the giant/AGB branch where dust shells vary")
    return True, f"M_G = {abs_g:.2f}, BP-RP = {bp_rp:.2f}: main sequence"


def check_fade_amplitude(row: dict) -> tuple[bool | None, str]:
    """Require the fade to be large compared with the published cross-epoch floor.

    HD 172555's mid-infrared flux was stable to within 4% between IRAS in 1983
    and WISE in 2010 -- the only explicit IRAS-to-WISE stability measurement in
    the literature, and therefore the empirical floor of this comparison. A
    statistically significant 10% fade sits barely above that floor and is not
    worth believing. TYC 8241 2652 1 dropped by a factor of ~30, which is far
    above it. Demanding a large *fraction* as well as a large *significance*
    keeps the search in the regime the systematics can support.
    """
    f_cess = _f(row, "f_cess")
    if not np.isfinite(f_cess):
        return None, "no cessation fraction"
    if f_cess < F_CESS_MIN:
        return False, f"only {100 * f_cess:.0f}% of the excess lost, < {100 * F_CESS_MIN:.0f}%"
    if f_cess < 3 * CROSS_EPOCH_FLOOR_FRAC:
        return False, (f"fade of {100 * f_cess:.0f}% is within 3x the published "
                       f"{100 * CROSS_EPOCH_FLOOR_FRAC:.0f}% cross-epoch stability floor")
    return True, f"{100 * f_cess:.0f}% of the excess is gone"


def check_not_eruptive(row: dict) -> tuple[bool | None, str]:
    """Veto R CrB stars, carbon stars and post-AGB objects.

    R Coronae Borealis stars puff out carbon dust and clear it again, producing
    factor-of-ten mid-infrared swings *between the very epochs this channel
    uses* -- the IRAS, AKARI and WISE comparison has already been run on them
    (Melis et al. 2023) and it fades exactly like the target signature. Post-AGB
    binaries and carbon stars behave similarly. Spectral class alone rejects
    the entire group.
    """
    spt = str(row.get("spectral_type", "") or "")
    known = str(row.get("known_class", "") or "").lower()
    eruptive = ("rcb", "r crb", "rcrb", "post-agb", "postagb", "carbon", "mira",
                "ysо", "yso", "herbig", "t tauri", "ttauri", "symbiotic")
    if not spt and not known:
        return None, "no spectral classification"
    hay = f"{spt} {known}".lower()
    for tag in eruptive:
        if tag in hay:
            return False, f"classified as an eruptive/dusty variable: '{spt or known}'"
    if spt.strip().upper().startswith(("C", "S")) and len(spt.strip()) > 1:
        return False, f"carbon/S-type star ({spt}): circumstellar dust varies"
    return True, f"spectral class '{spt or known}' is not an eruptive dust producer"


def check_ladder_coherent(row: dict) -> tuple[bool | None, str]:
    """Reject non-monotonic three-epoch behaviour and un-adjudicable two-epoch cases."""
    verdict = row.get("ladder_verdict")
    if verdict is None:
        return None, "no three-epoch adjudication"
    if verdict == "incoherent":
        return False, "excess rose and fell across three instruments: a systematic"
    if verdict == "no_mid_epoch":
        return False, ("no AKARI epoch: a 27-year IRAS-to-WISE fade cannot be "
                       "separated from an IRAS blend or spurious source")
    if verdict in ("constant", "rise"):
        return False, f"three-epoch verdict is '{verdict}', not a cessation"
    return True, f"three-epoch verdict '{verdict}'"


#: The funnel, in execution order. Ordered so the cheapest, most damaging
#: contaminants are removed first and the astrophysical discriminant last.
RULES: tuple[tuple[str, object], ...] = (
    ("ladder_coherent", check_ladder_coherent),
    ("fade_amplitude", check_fade_amplitude),
    ("beam_blending", check_beam_blending),
    ("far_ir_background", check_far_ir_background),
    ("saturation", check_saturation),
    ("early_quality", check_early_quality),
    ("cirrus", check_cirrus),
    ("association", check_association),
    ("stellar_astrometry", check_stellar_astrometry),
    ("solar_system", check_solar_system),
    ("blend", check_blend),
    ("variability", check_variability),
    ("late_epoch_flat", check_late_epoch_flat),
    ("not_eruptive", check_not_eruptive),
    ("not_evolved", check_not_evolved),
    ("not_young", check_not_young),
)


def vet_source(row: dict, require_all: bool = True) -> VetResult:
    """Run the whole funnel on one source.

    A rule that cannot be evaluated (its data are missing) does **not** count as
    a pass. It is recorded in ``coverage`` and reduces ``checks_run``, so a
    source that survives only because nothing could be tested is visibly
    distinguishable from one that survived real tests. With ``require_all`` the
    source must additionally have been testable on every rule.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    coverage: dict[str, str] = {}
    stage_failed: str | None = None
    n_run = 0

    for name, rule in RULES:
        verdict, reason = rule(row)  # type: ignore[operator]
        if verdict is None:
            coverage[name] = "not_tested"
            warnings.append(f"{name}: {reason}")
            continue
        n_run += 1
        coverage[name] = "pass" if verdict else "fail"
        if not verdict:
            stage_failed = name
            reasons.append(f"{name}: {reason}")
            break

    passed = stage_failed is None
    if passed and require_all and n_run < len(RULES):
        passed = False
        stage_failed = "coverage"
        reasons.append(f"only {n_run} of {len(RULES)} checks could be evaluated; "
                       "a candidate that survives untested checks is not a candidate")

    return VetResult(source_id=str(row.get("source_id", "?")), passed=passed,
                     stage_failed=stage_failed, reasons=reasons, warnings=warnings,
                     checks_run=n_run, checks_possible=len(RULES), coverage=coverage)


def vet_all(rows: list[dict], require_all: bool = True) -> dict:
    """Run the funnel over a list of sources and return the funnel counts."""
    results = [vet_source(r, require_all=require_all) for r in rows]
    stages = [name for name, _ in RULES] + ["coverage"]
    killed = dict.fromkeys(stages, 0)
    for r in results:
        if r.stage_failed:
            killed[r.stage_failed] = killed.get(r.stage_failed, 0) + 1
    survivors = [r for r in results if r.passed]
    return {
        "n_in": len(rows),
        "n_survivors": len(survivors),
        "killed_by_stage": killed,
        "survivors": [r.source_id for r in survivors],
        "results": results,
    }
