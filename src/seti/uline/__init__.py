"""ULINE --- industrial fluorine molecules in public unidentified-line lists.

Signature **S54** (:doc:`docs/necrofrontier.md` §2, "the molecule life and
chemistry both refuse"): no interstellar molecule with more than one fluorine
atom has ever been detected; fluorine chemistry terminates at HF and CF⁺; and
terrestrial life makes no N–F, S–F or fully fluorinated bonds.  A rotational
line *pattern* of a polar industrial species (NF₃, CHF₃, CH₂F₂, CF₃Cl,
CF₂Cl₂, CFCl₃, CF₃CN, COF₂, SO₂F₂, CHClF₂, the CF₂ radical) in a published
unidentified-line (U-line) list is therefore a residue of chemistry, not of
astrochemistry.  CH₃Cl (detected: Fayolle et al. 2017) and CH₃F are the
natural organohalogen baseline and veto, not the target.

The record (necrofrontier sweep, §5 row g7): no ISM / circumstellar
line-survey search for artificial species exists; the CFC literature is
exoplanet atmospheres.  The public U-line inventories (Orion KL HIFI, Crockett
et al. 2014, ~1,730 U-lines; IRC+10216, He et al. 2008, 17 U-lines) and the
JPL / CDMS line lists are all reachable from a runner.

Modules
-------
``lines``     ``.cat`` and ``catdir`` parsers, 300 K → T_ex intensity rescaling,
              the symmetric-top predictor (pure)
``match``     blend merging, Doppler geometry, vetoes, the pattern test and its
              shift-trial false-alarm probability (pure)
``acquire``   runner-only JPL / CDMS / VizieR-TAP access with runtime discovery
``run``       stage orchestration -> ``results/uline/``
"""

from __future__ import annotations

__all__ = ["lines", "match"]
