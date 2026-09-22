"""Line lists and the two wavelength vetoes for SPARK (pure; no I/O).

Every wavelength here is **vacuum microns**.  Literature NIR line lists are
mostly air wavelengths above 0.2 µm; :func:`air_to_vacuum_um` is the
conversion applied at definition time (``docs/channel-brief.md`` §4).  The
lists carry a ``src`` tag naming where each entry comes from, because a veto
list nobody can audit is a veto list nobody trusts.

Three lists
-----------
``STELLAR_LINES``   photospheric, chromospheric and circumstellar features a
                    *star* shows at low resolution: H I Paschen / Brackett /
                    Pfund series, He I, the strong neutral-metal lines of
                    cool stars (Mg I, Si I, Al I, Na I, Ca I, K I, Fe I),
                    Fe II / [Fe II] of active and YSO stars, the CO first-
                    and second-overtone bandheads, the H₂ shock lines, the
                    O I / Ca II / [S III] lines below 1 µm, the broad H₂O /
                    CH₄ / CO-fundamental molecular bands, and the YSO ice
                    absorptions (``kind: band`` entries have a half-width).
``GALAXY_LINES``    nebular / AGN lines whose redshifted images populate a
                    slitless survey — the classic single-line emitter
                    contamination — used by :func:`redshift_pattern`.
``INDUSTRIAL_BANDS`` Nd:YAG, Yb-fibre, Er-fibre, InGaAs telecom, Ti:sapphire
                    — DESCRIPTORS, never filters (``docs/roman.md`` §2.2).

Sources: NIST ASD (Kramida et al.) for H I / He I / metal lines; Rayner,
Cushing & Vacca 2009 (IRTF spectral library, their Tables 4–7) for the cool-
star H and K features; Meyer et al. 1998 and Wallace & Hinkle 1997 for the
CO bandheads; Black & van Dishoeck 1987 for the H₂ lines; Boogert et al.
2015 for the ice bands; van Dokkum / SDSS vacuum tables for the nebular set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Air <-> vacuum
# --------------------------------------------------------------------------


def air_to_vacuum_um(um):
    """Air → vacuum (micron), Edlén-type dispersion of standard air.

    ``n - 1 = 1e-8 (8342.13 + 2406030 / (130 - s²) + 15997 / (38.9 - s²))``,
    ``s = 1/λ_um``.  Nd:YAG 1064.1 nm (air) → 1064.4 nm (vacuum).
    """
    lam = np.asarray(um, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = (1.0 / lam) ** 2
        n = 1.0 + 1e-8 * (8342.13 + 2406030.0 / (130.0 - s2) + 15997.0 / (38.9 - s2))
    out = lam * n
    return float(out) if np.ndim(out) == 0 else out


def _air(um: float) -> float:
    return round(float(air_to_vacuum_um(um)), 5)


@dataclass(frozen=True)
class Line:
    name: str
    um: float              # vacuum micron (line centre, or band centre)
    kind: str = "line"     # line | band
    half_width_um: float = 0.0
    src: str = ""
    note: str = ""


# --------------------------------------------------------------------------
# Stellar features (what a STAR can show at R ~ 40-450)
# --------------------------------------------------------------------------
_H = "NIST ASD (vacuum)"
_R09 = "Rayner+2009 IRTF library (air->vac)"
_CO = "Wallace & Hinkle 1997; Meyer+1998 (vacuum)"
_H2 = "Black & van Dishoeck 1987 (vacuum)"
_ICE = "Boogert+2015 (band centre)"

STELLAR_LINES: tuple[Line, ...] = (
    # ---- H I Paschen (vacuum) ----
    Line("Pa-alpha", 1.87561, src=_H), Line("Pa-beta", 1.28216, src=_H),
    Line("Pa-gamma", 1.09411, src=_H), Line("Pa-delta", 1.00521, src=_H),
    Line("Pa-epsilon", 0.95486, src=_H), Line("Pa9", 0.92315, src=_H),
    Line("Pa10", 0.90174, src=_H), Line("Pa11", 0.88652, src=_H), Line("Pa12", 0.87529, src=_H),
    # ---- H I Brackett (vacuum); Br10-20 are the H-band absorption series of A/F stars ----
    Line("Br-alpha", 4.05226, src=_H), Line("Br-beta", 2.62587, src=_H),
    Line("Br-gamma", 2.16612, src=_H), Line("Br-delta", 1.94509, src=_H),
    Line("Br9", 1.81791, src=_H), Line("Br10", 1.73669, src=_H), Line("Br11", 1.68111, src=_H),
    Line("Br12", 1.64117, src=_H), Line("Br13", 1.61137, src=_H), Line("Br14", 1.58849, src=_H),
    Line("Br15", 1.57050, src=_H), Line("Br16", 1.55607, src=_H), Line("Br17", 1.54431, src=_H),
    Line("Br18", 1.53460, src=_H), Line("Br19", 1.52647, src=_H), Line("Br20", 1.51960, src=_H),
    Line("Br21", 1.51367, src=_H), Line("Br22", 1.50852, src=_H),
    # ---- H I Pfund (vacuum) ----
    Line("Pf-beta", 4.65378, src=_H), Line("Pf-gamma", 3.74056, src=_H),
    Line("Pf-delta", 3.29699, src=_H), Line("Pf-epsilon", 3.03920, src=_H),
    Line("Pf9", 2.87300, src=_H),
    # ---- He I (vacuum) ----
    Line("He I 1.083", 1.08332, src=_H, note="chromospheres, Be, YSO; 0.02 um from Nd:YAG at R=40"),
    Line("He I 2.058", 2.05869, src=_H), Line("He I 0.707", _air(0.70652), src=_H),
    Line("He I 1.279", _air(1.27850), src=_H), Line("He I 1.701", _air(1.70024), src=_H),
    Line("He I 0.588", _air(0.58756), src=_H),
    # ---- neutral metals of cool stars (air -> vacuum) ----
    Line("Mg I 1.503", _air(1.50249), src=_R09), Line("Mg I 1.504", _air(1.50401), src=_R09),
    Line("Mg I 1.575", _air(1.57488), src=_R09), Line("Mg I 1.711", _air(1.71090), src=_R09),
    Line("Mg I 1.183", _air(1.18280), src=_R09), Line("Mg I 1.488", _air(1.48786), src=_R09),
    Line("Si I 1.589", _air(1.58884), src=_R09), Line("Si I 1.596", _air(1.59640), src=_R09),
    Line("Si I 1.203", _air(1.20350), src=_R09), Line("Si I 1.210", _air(1.21030), src=_R09),
    Line("Al I 1.313", _air(1.31230), src=_R09), Line("Al I 1.672", _air(1.67190), src=_R09),
    Line("Al I 1.675", _air(1.67500), src=_R09), Line("Al I 2.110", _air(2.10990), src=_R09),
    Line("Na I 1.138", _air(1.13810), src=_R09), Line("Na I 1.140", _air(1.14040), src=_R09),
    Line("Na I 2.206", _air(2.20620), src=_R09), Line("Na I 2.209", _air(2.20900), src=_R09),
    Line("Ca I 1.978", _air(1.97820), src=_R09), Line("Ca I 2.261", _air(2.26140), src=_R09),
    Line("Ca I 2.263", _air(2.26310), src=_R09), Line("Ca I 2.266", _air(2.26570), src=_R09),
    Line("K I 0.767", _air(0.76649), src=_H), Line("K I 0.770", _air(0.76990), src=_H),
    Line("K I 1.169", _air(1.16900), src=_R09), Line("K I 1.177", _air(1.17700), src=_R09),
    Line("K I 1.243", _air(1.24320), src=_R09), Line("K I 1.252", _air(1.25220), src=_R09),
    Line("K I 1.516", _air(1.51630), src=_R09), Line("K I 1.517", _air(1.51680), src=_R09),
    Line("Fe I 1.189", _air(1.18860), src=_R09), Line("Fe I 1.198", _air(1.19760), src=_R09),
    Line("Fe I 1.530", _air(1.52950), src=_R09), Line("Fe I 1.163", _air(1.16300), src=_R09),
    Line("Ti I 0.970", _air(0.96950), src=_R09), Line("Ti I 2.190", _air(2.19000), src=_R09),
    Line("Ca II 0.850", _air(0.84982), src=_H), Line("Ca II 0.854", _air(0.85423), src=_H),
    Line("Ca II 0.866", _air(0.86645), src=_H),
    Line("O I 0.777", _air(0.77742), src=_H), Line("O I 0.845", _air(0.84463), src=_H),
    Line("O I 1.129", _air(1.12870), src=_H, note="Be / YSO / Herbig emission"),
    # ---- Fe II / [Fe II] (active, Be, YSO, outflows), vacuum ----
    Line("Fe II 1.257", 1.25702, src=_H), Line("[Fe II] 1.295", 1.29462, src=_H),
    Line("Fe II 1.321", 1.32092, src=_H), Line("[Fe II] 1.534", 1.53389, src=_H),
    Line("Fe II 1.644", 1.64400, src=_H), Line("[Fe II] 1.677", 1.67733, src=_H),
    Line("Fe II 1.687", 1.68778, src=_H), Line("Fe II 1.742", 1.74188, src=_H),
    Line("Fe II 1.688", 1.68811, src=_H), Line("Fe II 1.749", 1.74868, src=_H),
    # ---- forbidden lines of nebular envelopes below 1 um, vacuum ----
    Line("[S III] 0.907", 0.90714, src=_H), Line("[S III] 0.953", 0.95332, src=_H),
    Line("[C I] 0.985", 0.98532, src=_H), Line("[S II] 1.029", 1.02901, src=_H),
    Line("[S II] 1.032", 1.03202, src=_H), Line("[S II] 1.034", 1.03390, src=_H),
    Line("[N I] 1.040", 1.04010, src=_H),
    # ---- CO bandheads (vacuum): first overtone (K) and second overtone (H) ----
    Line("CO 3-0 bandhead", 1.55820, src=_CO), Line("CO 4-1 bandhead", 1.57800, src=_CO),
    Line("CO 5-2 bandhead", 1.59820, src=_CO), Line("CO 6-3 bandhead", 1.61890, src=_CO),
    Line("CO 7-4 bandhead", 1.63970, src=_CO), Line("CO 8-5 bandhead", 1.66100, src=_CO),
    Line("CO 9-6 bandhead", 1.68260, src=_CO),
    Line("CO 2-0 bandhead", 2.29353, src=_CO), Line("CO 3-1 bandhead", 2.32266, src=_CO),
    Line("CO 4-2 bandhead", 2.35246, src=_CO), Line("CO 5-3 bandhead", 2.38295, src=_CO),
    Line("CO 6-4 bandhead", 2.41420, src=_CO),
    Line("CO fundamental", 4.67, kind="band", half_width_um=0.25, src=_CO),
    Line("13CO 2-0 bandhead", 2.34483, src=_CO),
    # ---- H2 ro-vibrational (shocks, YSO), vacuum ----
    Line("H2 1-0 S(1)", 2.12183, src=_H2), Line("H2 1-0 S(0)", 2.22329, src=_H2),
    Line("H2 1-0 S(2)", 2.03376, src=_H2), Line("H2 1-0 S(3)", 1.95756, src=_H2),
    Line("H2 1-0 Q(1)", 2.40659, src=_H2), Line("H2 1-0 Q(3)", 2.42373, src=_H2),
    Line("H2 2-1 S(1)", 2.24772, src=_H2), Line("H2 1-0 S(7)", 1.74803, src=_H2),
    Line("H2 1-0 S(9)", 1.68772, src=_H2), Line("H2 0-0 S(9)", 4.69461, src=_H2),
    # ---- broad molecular bands of cool / young stars (band centres, half-widths) ----
    Line("H2O 1.4 band", 1.40, kind="band", half_width_um=0.08, src=_R09),
    Line("H2O 1.9 band", 1.90, kind="band", half_width_um=0.10, src=_R09),
    Line("H2O 2.7 band", 2.70, kind="band", half_width_um=0.25, src=_R09),
    Line("CH4 3.3 band", 3.30, kind="band", half_width_um=0.15, src=_R09),
    Line("PAH 3.3", 3.29, kind="band", half_width_um=0.04, src="Tokunaga+1991 (vacuum)"),
    Line("CN 1.10 band", 1.10, kind="band", half_width_um=0.03, src=_R09),
    Line("VO 1.05 band", 1.05, kind="band", half_width_um=0.03, src=_R09),
    Line("FeH 0.99 band", 0.99, kind="band", half_width_um=0.015, src=_R09),
    Line("TiO 0.85 band", 0.85, kind="band", half_width_um=0.02, src=_R09),
    Line("TiO 0.89 band", 0.89, kind="band", half_width_um=0.02, src=_R09),
    # ---- ice absorption (YSO / background stars behind clouds), band centres ----
    Line("H2O ice 3.05", 3.05, kind="band", half_width_um=0.20, src=_ICE),
    Line("CO2 ice 4.27", 4.27, kind="band", half_width_um=0.03, src=_ICE),
    Line("CO ice 4.67", 4.67, kind="band", half_width_um=0.02, src=_ICE),
    Line("CH3OH ice 3.53", 3.53, kind="band", half_width_um=0.04, src=_ICE),
)

# --------------------------------------------------------------------------
# Nebular / galaxy lines (rest vacuum micron) for the redshift-pattern veto
# --------------------------------------------------------------------------
_NEB = "SDSS/DESI vacuum rest wavelengths"
GALAXY_LINES: tuple[Line, ...] = (
    Line("Ly-alpha", 0.121567, src=_NEB), Line("N V 1240", 0.124081, src=_NEB),
    Line("Si IV 1397", 0.139761, src=_NEB), Line("C IV 1549", 0.154949, src=_NEB),
    Line("He II 1640", 0.164040, src=_NEB), Line("O III] 1666", 0.166615, src=_NEB),
    Line("C III] 1909", 0.190873, src=_NEB), Line("Mg II 2799", 0.279875, src=_NEB),
    Line("[Ne V] 3426", 0.342685, src=_NEB), Line("[O II] 3727", 0.372709, src=_NEB),
    Line("[O II] 3729", 0.372988, src=_NEB), Line("[Ne III] 3869", 0.386986, src=_NEB),
    Line("H-delta", 0.410289, src=_NEB), Line("H-gamma", 0.434169, src=_NEB),
    Line("[O III] 4363", 0.436444, src=_NEB), Line("H-beta", 0.486268, src=_NEB),
    Line("[O III] 4959", 0.496030, src=_NEB), Line("[O III] 5007", 0.500824, src=_NEB),
    Line("He I 5876", 0.587730, src=_NEB), Line("[O I] 6300", 0.630204, src=_NEB),
    Line("[N II] 6548", 0.654986, src=_NEB), Line("H-alpha", 0.656461, src=_NEB),
    Line("[N II] 6583", 0.658527, src=_NEB), Line("[S II] 6716", 0.671829, src=_NEB),
    Line("[S II] 6731", 0.673267, src=_NEB), Line("[Ar III] 7136", 0.713771, src=_NEB),
    Line("[S III] 9069", 0.907142, src=_NEB), Line("[S III] 9531", 0.953315, src=_NEB),
    Line("He I 1.083", 1.083320, src=_NEB), Line("Pa-gamma", 1.094114, src=_NEB),
    Line("[Fe II] 1.257", 1.257020, src=_NEB), Line("Pa-beta", 1.282159, src=_NEB),
    Line("[Fe II] 1.644", 1.644000, src=_NEB), Line("Pa-alpha", 1.875613, src=_NEB),
    Line("Br-gamma", 2.166120, src=_NEB), Line("H2 1-0 S(1)", 2.121834, src=_NEB),
    Line("PAH 3.3", 3.29, src=_NEB),
)

#: The strong single-line emitters a slitless survey mistakes for one another.
SINGLE_LINE_CANDIDATES: tuple[str, ...] = (
    "H-alpha", "[O III] 5007", "H-beta", "[O II] 3727", "Ly-alpha", "Pa-alpha",
    "Mg II 2799", "C IV 1549", "[S III] 9531", "Pa-beta",
)

# --------------------------------------------------------------------------
# Industrial / high-power laser bands (vacuum micron) — DESCRIPTORS
# --------------------------------------------------------------------------
INDUSTRIAL_BANDS: tuple[Line, ...] = (
    Line("Nd:YAG 1064", 1.06440, src="Koechner 2006 (air 1064.1 nm -> vac)"),
    Line("Nd:YAG 946", 0.94626, src="Koechner 2006"),
    Line("Nd:YAG 1319", 1.31910, src="Koechner 2006"),
    Line("Nd:YAG 1338", 1.33840, src="Koechner 2006"),
    Line("Yb fibre", 1.060, kind="band", half_width_um=0.030, src="1.03-1.09 um gain band"),
    Line("Er fibre C-band", 1.5480, kind="band", half_width_um=0.0175, src="1.5305-1.5655 um"),
    Line("Er fibre L-band", 1.5955, kind="band", half_width_um=0.0300, src="1.5655-1.6255 um"),
    Line("InGaAs telecom 1310", 1.310, kind="band", half_width_um=0.050, src="O-band 1.26-1.36 um"),
    Line("Tm fibre", 1.975, kind="band", half_width_um=0.075, src="1.90-2.05 um"),
    Line("Ho:YAG 2.1", 2.100, kind="band", half_width_um=0.020, src="2.08-2.12 um"),
    Line("Ti:sapphire", 0.850, kind="band", half_width_um=0.150, src="0.70-1.00 um"),
)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
def _in_band(lam: float, ln: Line, tol_um: float) -> bool:
    if ln.kind == "band":
        return abs(lam - ln.um) <= ln.half_width_um + tol_um
    return abs(lam - ln.um) <= tol_um


def stellar_line_match(lambda_um: float, tol_um: float,
                       lines: tuple[Line, ...] = STELLAR_LINES, *, include_bands: bool = True) -> Line | None:
    """The nearest stellar feature within ``tol_um``, else ``None``.

    A discrete line within tolerance always beats a band the wavelength merely
    sits inside.  ``include_bands=False`` ignores the broad molecular / ice
    bands — right for a resolved (R ≈ 450) table where the FWHM test already
    separates an unresolved feature from a band, wrong for R ≈ 40 where a band
    edge can leave a one-channel residual.
    """
    lam = float(lambda_um)
    best, best_key = None, (2, math.inf)
    for ln in lines:
        if ln.kind == "band" and not include_bands:
            continue
        if _in_band(lam, ln, tol_um):
            key = (1 if ln.kind == "band" else 0, abs(lam - ln.um))
            if key < best_key:
                best, best_key = ln, key
    return best


def industrial_flag(lambda_um: float, tol_um: float) -> str | None:
    """Every industrial line / band the wavelength falls in, lines first — a descriptor."""
    lam = float(lambda_um)
    hits = [ln for ln in INDUSTRIAL_BANDS if _in_band(lam, ln, tol_um)]
    if not hits:
        return None
    hits.sort(key=lambda ln: (ln.kind == "band", abs(lam - ln.um)))
    return "|".join(ln.name for ln in hits)


def single_line_interpretations(lambda_um: float, z_max: float = 12.0) -> list[dict]:
    """For a lone feature: the redshift each strong emitter would imply (descriptor)."""
    lam = float(lambda_um)
    by_name = {ln.name: ln for ln in GALAXY_LINES}
    out = []
    for nm in SINGLE_LINE_CANDIDATES:
        ln = by_name.get(nm)
        if ln is None:
            continue
        z = lam / ln.um - 1.0
        if -0.002 <= z <= z_max:
            out.append({"line": nm, "z": round(z, 4)})
    return out


def redshift_pattern(lambdas_um, *, z_tol: float = 0.003, z_min: float = -0.002,
                     z_max: float = 12.0, lines: tuple[Line, ...] = GALAXY_LINES,
                     min_lines: int = 2) -> dict:
    """Is there one redshift at which ≥ ``min_lines`` of these features are galaxy lines?

    Brute force over (feature, rest line) anchors: ``z = λ_obs/λ_rest − 1``;
    at that z every other feature is matched to the nearest *different* rest
    line within ``z_tol (1+z)`` in redshift.  Returns the best solution
    ``{"vetoed", "z", "n_lines", "ids"}`` (most lines, then smallest rms).
    Two features that are a locked pair at z≈0 (He I 1.083 + Pa-γ) are also
    caught — a z=0 nebular pattern is a star with an envelope, not a laser.
    """
    lams = [float(x) for x in np.atleast_1d(np.asarray(lambdas_um, dtype=float)) if np.isfinite(x)]
    best = {"vetoed": False, "z": None, "n_lines": 0, "ids": [], "rms": None}
    if len(lams) < min_lines:
        return best
    for i, li in enumerate(lams):
        for anchor in lines:
            z = li / anchor.um - 1.0
            if z < z_min or z > z_max:
                continue
            ids = [(li, anchor.name)]
            used = {anchor.name}
            resid = [0.0]
            for j, lj in enumerate(lams):
                if j == i:
                    continue
                cand, cd = None, math.inf
                for ln in lines:
                    if ln.name in used:
                        continue
                    zj = lj / ln.um - 1.0
                    d = abs(zj - z)
                    if d <= z_tol * (1.0 + z) and d < cd:
                        cand, cd = ln, d
                if cand is not None:
                    ids.append((lj, cand.name))
                    used.add(cand.name)
                    resid.append(cd)
            n = len(ids)
            rms = float(np.sqrt(np.mean(np.square(resid)))) if n else None
            if n > best["n_lines"] or (n == best["n_lines"] and best["rms"] is not None
                                       and rms is not None and rms < best["rms"]):
                best = {"vetoed": n >= min_lines, "z": round(float(z), 5), "n_lines": n,
                        "ids": [{"lambda_um": round(a, 5), "line": b} for a, b in ids],
                        "rms": rms}
    best["vetoed"] = bool(best["n_lines"] >= min_lines)
    return best


def line_table() -> list[dict]:
    """The embedded lists as rows (for the results directory and the doc)."""
    rows = []
    for tag, lst in (("stellar", STELLAR_LINES), ("galaxy", GALAXY_LINES),
                     ("industrial", INDUSTRIAL_BANDS)):
        for ln in lst:
            rows.append({"list": tag, "name": ln.name, "um": ln.um, "kind": ln.kind,
                         "half_width_um": ln.half_width_um, "src": ln.src, "note": ln.note})
    return rows
