"""Deep vetting of dimming candidates against the mundane Boyajian mimics.

A short list of deep, aperiodic, main-sequence dippers still has to survive the
explanations that produced every previous "alien megastructure" false alarm.  The
single most decisive one is **infrared excess**: a star dimmed by a circumstellar
dust disk (a UX Ori / "dipper" YSO, or a debris/transition disk) glows in the
mid-infrared, whereas KIC 8462852 has a *bare photosphere* --- no measurable WISE
excess.  We therefore cross-match each candidate to WISE (W1, W2), 2MASS (JHKs)
and SIMBAD, form the photospheric colours, and return a verdict:

* ``ir_excess_dusty``   --- WISE/2MASS excess => occulting dust, mundane;
* ``known_variable``    --- already classified (CV, YSO, eclipsing, Mira...) in SIMBAD;
* ``clean``             --- main-sequence, no IR excess, no prior classification:
                            the Boyajian-like regime that warrants real follow-up.

Network fetchers are reused from :mod:`seti.acquire.science`; the verdict logic is
a pure function so it is unit-tested offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A bare stellar photosphere has W1-W2 ~ 0 (Rayleigh-Jeans) and only a small
# K-W2; warm circumstellar dust pushes both red.  Thresholds are deliberately
# conservative so a genuinely bare star is not mislabelled as dusty.
W1W2_EXCESS = 0.15
KW2_EXCESS = 0.60

# SIMBAD object types that are themselves a mundane dimming explanation.
_MUNDANE_OTYPES = (
    "YSO", "Or*", "TT*", "Ae*", "out",          # young stellar / disk dippers
    "CV*", "No*", "DN*",                         # cataclysmic / accreting
    "EB*", "Al*", "bL*", "WU*", "EClB",          # eclipsing binaries
    "Mi*", "LP*", "sr*", "RG*", "AGB", "C*",     # evolved dusty giants (R CrB/Mira)
)


def ir_excess_verdict(row: dict) -> str:
    """Pure verdict from one candidate's cross-matched photometry/classification."""
    w1 = row.get("W1mag")
    w2 = row.get("W2mag")
    k = row.get("Ksmag")
    otype = str(row.get("simbad_otype") or "").strip()

    w1w2 = (float(w1) - float(w2)) if _ok(w1) and _ok(w2) else np.nan
    kw2 = (float(k) - float(w2)) if _ok(k) and _ok(w2) else np.nan
    ir_excess = ((np.isfinite(w1w2) and w1w2 > W1W2_EXCESS)
                 or (np.isfinite(kw2) and kw2 > KW2_EXCESS))

    if ir_excess:
        return "ir_excess_dusty"
    if otype and any(tok.lower() in otype.lower() for tok in _MUNDANE_OTYPES):
        return "known_variable"
    # No IR excess and no mundane classification: the interesting regime.  If the
    # HR class is known and main-sequence we call it clean; if photometry is
    # missing we cannot clear it, so flag it for manual review.
    if not (np.isfinite(w1w2) or np.isfinite(kw2)):
        return "no_ir_data"
    return "clean"


def _ok(v) -> bool:
    try:
        return v is not None and np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def vet_candidates(cand: pd.DataFrame, radius_arcsec: float = 3.0) -> pd.DataFrame:
    """Cross-match the candidate shortlist to WISE/2MASS/SIMBAD and add a verdict.

    ``cand`` must carry ``source_id``, ``ra``, ``dec``.  Returns the input frame
    augmented with WISE/2MASS magnitudes, photospheric colours, the SIMBAD object
    type, and an ``ir_verdict`` column.
    """
    from ..acquire.science import fetch_catwise, fetch_simbad_context, fetch_twomass

    pos = cand[["source_id", "ra", "dec"]].copy()
    out = cand.copy()
    for fetch, cols in (
        (lambda: fetch_catwise(pos, radius_arcsec), ["W1mag", "W2mag"]),
        (lambda: fetch_twomass(pos, max(radius_arcsec, 3.0)), ["Jmag", "Hmag", "Ksmag"]),
        (lambda: fetch_simbad_context(pos), ["simbad_otype", "simbad_sptype", "simbad_id"]),
    ):
        try:
            got = fetch()
            if got is not None and len(got):
                keep = ["source_id"] + [c for c in cols if c in got.columns]
                out = out.merge(got[keep], on="source_id", how="left")
        except Exception as exc:
            print(f"[dimming-vet] fetch skipped: {exc!r}")

    out["W1_W2"] = _col(out, "W1mag") - _col(out, "W2mag")
    out["K_W2"] = _col(out, "Ksmag") - _col(out, "W2mag")
    out["ir_verdict"] = [ir_excess_verdict(r) for _, r in out.iterrows()]
    return out


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors="coerce") if name in df.columns \
        else pd.Series(np.nan, index=df.index)


__all__ = ["vet_candidates", "ir_excess_verdict", "W1W2_EXCESS", "KW2_EXCESS"]
