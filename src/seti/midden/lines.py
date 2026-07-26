"""MIDDEN line list: target radionuclide lines, RV-registration lines, controls.

All wavelengths are **air Angstroms** (the HARPS/FEROS Phase-3 convention).
Provenance is recorded per line; the encoded values are *not* silently
trusted — ``verify_against_nist`` re-derives every non-control wavelength
from the NIST Atomic Spectra Database on the CI runner (which has egress) and
hard-fails the run on any mismatch > ``NIST_TOL_A``.

Roles
-----
radionuclide  The technosignature lines themselves (Tc I, U II, Th II).
rv_ref        Strong Fe I lines used to register the stellar radial velocity
              (they also absorb any global air/vacuum or wavelength-zero-point
              offset, which is velocity-like at the ~85 km/s level).
control       DUMMY wavelengths in quasi-clean continuum.  There is no real
              line there by construction; a star that shows "excess
              absorption" at a control wavelength is flagging on continuum /
              line-forest systematics and is vetoed.  Controls are not
              NIST-verifiable (they are deliberately not lines) — they are
              instead checked to sit > 3 A from every encoded real line.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

NIST_TOL_A = 0.05
_NIST_URL = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"


@dataclass(frozen=True)
class Line:
    species: str        # e.g. "Tc I"
    wavelength: float   # air Angstrom
    role: str           # "radionuclide" | "rv_ref" | "control"
    note: str


LINES: tuple[Line, ...] = (
    # --- Tc I resonance triplet (a6S_5/2 -> z6P) ------------------------------
    # Meggers & Scribner 1950 (J. Res. NBS 45, 476) laboratory wavelengths;
    # the lines Merrill 1952 used to discover stellar Tc in S stars, the
    # standard AGB Tc diagnostics (Little-Marenin & Little 1979; Lebzelter &
    # Hron 2003), and the lines Andrievsky et al. 2023 re-analysed (4297.06)
    # to refute the Przybylski's-Star Tc claim.
    Line("Tc I", 4238.19, "radionuclide", "resonance triplet z6P_3/2"),
    Line("Tc I", 4262.27, "radionuclide", "resonance triplet z6P_5/2"),
    Line("Tc I", 4297.06, "radionuclide", "resonance triplet z6P_7/2; Andrievsky+2023 line"),
    # --- actinides ------------------------------------------------------------
    # U II 3859.57: the uranium cosmochronometer line (Cayrel et al. 2001,
    # CS 31082-001).  Th II 4019.13: the classic thorium chronometer line
    # (Butcher 1987).  Overabundance WITHOUT the accompanying r-process
    # rare-earth pattern is the Whitmire-Wright actinide signature.
    Line("U II", 3859.57, "radionuclide", "U cosmochronometer line (Cayrel+2001)"),
    Line("Th II", 4019.13, "radionuclide", "Th cosmochronometer line (Butcher 1987)"),
    # --- RV registration: strong Fe I lines bracketing the target region ------
    # NIST/Moore air wavelengths; ubiquitous and strong in A5-F2 photospheres.
    Line("Fe I", 4045.81, "rv_ref", "Fe I multiplet 43, very strong"),
    Line("Fe I", 4063.59, "rv_ref", "Fe I multiplet 43"),
    Line("Fe I", 4071.74, "rv_ref", "Fe I multiplet 43"),
    Line("Fe I", 4132.06, "rv_ref", "Fe I"),
    Line("Fe I", 4143.87, "rv_ref", "Fe I"),
    Line("Fe I", 4202.03, "rv_ref", "Fe I"),
    Line("Fe I", 4250.79, "rv_ref", "Fe I"),
    Line("Fe I", 4271.76, "rv_ref", "Fe I multiplet 42"),
    Line("Fe I", 4325.76, "rv_ref", "Fe I multiplet 42"),
    Line("Fe I", 4383.55, "rv_ref", "Fe I multiplet 41, very strong"),
    Line("Fe I", 4404.75, "rv_ref", "Fe I multiplet 41"),
    # --- dummy controls: quasi-clean continuum, deliberately NOT lines --------
    # Chosen > 3 A from every encoded line and away from Balmer lines and
    # Ca I 4226.7.  A z>=4 "detection" here is a systematics veto.
    Line("DUMMY", 4152.30, "control", "clean-continuum false-positive control"),
    Line("DUMMY", 4222.10, "control", "clean-continuum false-positive control"),
    Line("DUMMY", 4288.40, "control", "clean-continuum false-positive control"),
)


def by_role(role: str) -> list[Line]:
    return [ln for ln in LINES if ln.role == role]


def radionuclide_lines() -> list[Line]:
    return by_role("radionuclide")


def tc_lines() -> list[Line]:
    return [ln for ln in LINES if ln.species == "Tc I"]


def rv_reference_wavelengths() -> list[float]:
    return [ln.wavelength for ln in by_role("rv_ref")]


def control_lines() -> list[Line]:
    return by_role("control")


def check_control_spacing(min_sep_a: float = 3.0) -> list[str]:
    """Offline invariant: every control sits > min_sep_a from every real line."""
    problems = []
    real = [ln for ln in LINES if ln.role != "control"]
    for c in control_lines():
        for r in real:
            if abs(c.wavelength - r.wavelength) < min_sep_a:
                problems.append(f"control {c.wavelength} within {min_sep_a} A "
                                f"of {r.species} {r.wavelength}")
    return problems


# ---------------------------------------------------------------------------
# NIST verification (runner-side; network required)
# ---------------------------------------------------------------------------

def _nist_wavelengths(species: str, lo: float, hi: float,
                      timeout: float = 60.0) -> list[float]:
    """All NIST ASD line wavelengths (air A) for one species in [lo, hi].

    Queries the ASD lines API in ASCII mode and parses every numeric token
    that lands in the requested window.  ASD returns *air* wavelengths for
    observed lines between 2000 A and 2 um, matching our convention.  The
    parameter set mirrors astroquery.nist; ``unit=0`` requests Angstroms, and
    a magnitude sanity-check rescales defensively if the service ever answers
    in nm.
    """
    import re

    import requests

    params = {
        "spectra": species, "low_w": f"{lo:.2f}", "upp_w": f"{hi:.2f}",
        "unit": 0, "format": 1, "line_out": 0, "en_unit": 0, "output": 0,
        "bibrefs": 1, "page_size": 15, "show_obs_wl": 1, "show_calc_wl": 1,
        "order_out": 0, "show_av": 2, "tsb_value": 0, "A_out": 0,
        "allowed_out": 1, "forbid_out": 1, "conf_out": "on", "term_out": "on",
        "enrg_out": "on", "J_out": "on",
    }
    r = requests.get(_NIST_URL, params=params, timeout=timeout)
    r.raise_for_status()
    toks = [float(t) for t in re.findall(r"\d{3,5}\.\d+", r.text)]
    if toks:
        med = sorted(toks)[len(toks) // 2]
        if lo / 10 - 1 < med < hi / 10 + 1:      # service answered in nm
            toks = [t * 10.0 for t in toks]
    return [t for t in toks if lo - 0.5 <= t <= hi + 0.5]


def _nist_wavelengths_astroquery(species: str, lo: float, hi: float) -> list[float]:
    """Fallback path through astroquery.nist (same service, maintained parser)."""
    import astropy.units as u
    from astroquery.nist import Nist

    tab = Nist.query(lo * u.AA, hi * u.AA, linename=species, wavelength_type="vac+air")
    out = []
    for col in ("Observed", "Ritz"):
        if col in tab.colnames:
            for v in tab[col]:
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    continue
    return [t for t in out if lo - 0.5 <= t <= hi + 0.5]


def verify_against_nist(out_path: Path | None = None,
                        tolerance: float = NIST_TOL_A) -> dict:
    """Verify every non-control wavelength against NIST ASD; raise on failure.

    This is the FIRST step of the CI workflow: the encoded line list is never
    silently trusted.  Controls are exempt from NIST matching (they are
    deliberately not lines) but their spacing invariant is enforced here too.
    """
    spacing = check_control_spacing()
    if spacing:
        raise RuntimeError(f"control-line spacing violated: {spacing}")

    results, failures = [], []
    species_lines: dict[str, list[Line]] = {}
    for ln in LINES:
        if ln.role != "control":
            species_lines.setdefault(ln.species, []).append(ln)

    for species, lns in species_lines.items():
        lo = min(ln.wavelength for ln in lns) - 2.0
        hi = max(ln.wavelength for ln in lns) + 2.0
        nist, source, err = [], None, None
        for fetch, name in ((_nist_wavelengths, "asd-api"),
                            (_nist_wavelengths_astroquery, "astroquery")):
            try:
                nist = fetch(species, lo, hi)
                if nist:
                    source = name
                    break
            except Exception as exc:  # noqa: BLE001 — try the next path
                err = exc
                print(f"[midden] NIST fetch via {name} failed for {species}: {exc!r}")
        if not nist:
            failures.append(f"{species}: no NIST lines retrievable ({err!r})")
            continue
        for ln in lns:
            best = min(abs(w - ln.wavelength) for w in nist)
            ok = best <= tolerance
            results.append({**asdict(ln), "nist_delta_a": best, "verified": ok,
                            "nist_source": source, "n_nist_lines": len(nist)})
            if not ok:
                failures.append(f"{ln.species} {ln.wavelength}: nearest NIST "
                                f"line is {best:.3f} A away (> {tolerance})")

    report = {"tolerance_a": tolerance, "n_lines_checked": len(results),
              "n_failures": len(failures), "failures": failures,
              "lines": results,
              "controls": [asdict(ln) for ln in control_lines()]}
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError(f"NIST line verification FAILED: {failures}")
    print(f"[midden] NIST verification passed: {len(results)} lines within "
          f"{tolerance} A")
    return report


__all__ = ["LINES", "Line", "NIST_TOL_A", "by_role", "check_control_spacing",
           "control_lines", "radionuclide_lines", "rv_reference_wavelengths",
           "tc_lines", "verify_against_nist"]
