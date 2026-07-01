"""Reject misclassified emission-line galaxies from the laser-line shortlist.

A background star-forming or active galaxy whose SDSS/DESI pipeline redshift is
wrong (or which is misclassified as ``STAR``) leaks its rest-frame nebular lines
into the search as high-significance "unresolved" candidates.  The observed-frame
known-line triage cannot catch these: it places the known lines using the
*catalogue* redshift, so at the wrong z the galaxy's Halpha sits nowhere near the
candidate wavelength.

The decisive test is *internal redshift consistency*: real nebular emission comes
as a family (Halpha, [N II], [O III], [S II], Hbeta ...) locked to one redshift.
If two or more of the surviving lines in a single spectrum are explained by the
galaxy emission-line grid at a common z, the "beacon" is a galaxy, not a
transmitter.  This is what unmasked the 7518/7542 A pair (Halpha + [N II] 6584 at
z = 0.1455) that had ranked first in the whole search.

Pure/offline; wavelengths are vacuum (matching SDSS/DESI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_C_KMS = 299792.458

# Strong nebular emission lines, vacuum wavelengths (Angstrom).  These are the
# lines that dominate emission-line-galaxy spectra and thus the leak.
GALAXY_LINES = {
    "[OII]3727": 3727.09,
    "[OII]3729": 3729.88,
    "[NeIII]3869": 3869.86,
    "Hg4341": 4341.68,
    "Hb4862": 4862.68,
    "[OIII]4959": 4960.30,
    "[OIII]5007": 5008.24,
    "[OI]6300": 6302.05,
    "[NII]6548": 6549.86,
    "Ha6563": 6564.61,
    "[NII]6584": 6585.27,
    "[SII]6716": 6718.29,
    "[SII]6731": 6732.68,
}


# Physically locked, tightly-spaced diagnostic pairs: both members always appear
# together in nebular emission, and their small rest separation makes a chance
# co-match improbable.  A galaxy call requires one of these (or >= 3 distinct
# lines at one z) --- not just any two grid lines, which a wide z-scan can fake.
_DIAGNOSTIC_PAIRS = [
    frozenset({"Ha6563", "[NII]6584"}),
    frozenset({"Ha6563", "[NII]6548"}),
    frozenset({"[NII]6548", "[NII]6584"}),
    frozenset({"[OIII]4959", "[OIII]5007"}),
    frozenset({"[SII]6716", "[SII]6731"}),
    frozenset({"Ha6563", "[SII]6716"}),
    frozenset({"Ha6563", "[SII]6731"}),
    frozenset({"[OII]3727", "[OII]3729"}),
]


def galaxy_redshift_match(obs_waves, tol_kms: float = 250.0,
                          z_max: float = 1.2) -> dict:
    """Find the redshift that explains the most observed lines as galaxy nebular
    emission.  Each observed wavelength is tried as an anchor against every grid
    line; at the implied z we count how many *other* observed lines fall on a grid
    line within ``tol_kms``.

    Returns ``{z, n_matched, rest_lines, matches, is_galaxy}``.  ``is_galaxy`` is
    True only when the matched lines contain a physically locked diagnostic pair
    (Halpha+[N II], the [O III] or [S II] doublet, ...) or >= 3 distinct rest
    lines at one z.  Requiring locked physics --- not just any two grid lines ---
    stops a wide redshift scan from faking a galaxy out of two unrelated peaks
    (e.g. an emission-line variable star)."""
    obs = sorted(float(w) for w in obs_waves if np.isfinite(w) and w > 0)
    best = {"z": np.nan, "n_matched": 0, "rest_lines": [], "matches": [],
            "is_galaxy": False}
    if len(obs) < 2:
        return best
    for anchor in obs:
        for arest in GALAXY_LINES.values():
            z = anchor / arest - 1.0
            if z < -0.003 or z > z_max:
                continue
            matches, rest_hit = [], set()
            for w in obs:
                best_dv, best_name = np.inf, None
                for name, rest in GALAXY_LINES.items():
                    pred = rest * (1 + z)
                    dv = abs(w - pred) / pred * _C_KMS
                    if dv < best_dv:
                        best_dv, best_name = dv, name
                if best_dv <= tol_kms:
                    matches.append((w, best_name, best_dv))
                    rest_hit.add(best_name)
            has_pair = any(p <= rest_hit for p in _DIAGNOSTIC_PAIRS)
            is_gal = (len(rest_hit) >= 3) or has_pair
            better = (len(rest_hit) > best["n_matched"]
                      or (is_gal and not best["is_galaxy"]))
            if better:
                best = {"z": z, "n_matched": len(rest_hit),
                        "rest_lines": sorted(rest_hit),
                        "matches": matches,
                        "is_galaxy": bool(is_gal)}
    return best


def flag_galaxy_spectra(candidates: pd.DataFrame, tol_kms: float = 250.0
                        ) -> pd.DataFrame:
    """Group candidate lines by ``spec_id`` and flag spectra whose surviving lines
    form a galaxy nebular family at a common redshift.

    Adds per-row columns ``galaxy_z``, ``galaxy_n_lines``, ``galaxy_rest_lines``,
    ``is_galaxy``.  A spectrum with < 2 surviving lines is left ``is_galaxy=False``
    (it cannot be tested this way and needs the raw-spectrum route instead).
    """
    out = candidates.copy()
    out["galaxy_z"] = np.nan
    out["galaxy_n_lines"] = 0
    out["galaxy_rest_lines"] = ""
    out["is_galaxy"] = False
    for _sid, grp in out.groupby("spec_id"):
        waves = pd.to_numeric(grp["wavelength"], errors="coerce").dropna().unique()
        res = galaxy_redshift_match(waves, tol_kms=tol_kms)
        if res["is_galaxy"]:
            idx = grp.index
            out.loc[idx, "galaxy_z"] = res["z"]
            out.loc[idx, "galaxy_n_lines"] = res["n_matched"]
            out.loc[idx, "galaxy_rest_lines"] = ",".join(res["rest_lines"])
            out.loc[idx, "is_galaxy"] = True
    return out


__all__ = ["GALAXY_LINES", "galaxy_redshift_match", "flag_galaxy_spectra"]
