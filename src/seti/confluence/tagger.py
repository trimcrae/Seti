"""Known-class tagger: which stars belong to classes that are EXPECTED to light up
several channels at once.

A young stellar object dips in ZTF, brightens and reddens in NEOWISE and
carries warm dust in W3/W4; an eclipsing binary is a dipper and a 'period';
a Be star has an infrared free-free excess and emission lines; a WD+M pair is
over-luminous in the red and blue at once; an AGB star is variable
everywhere.  If the confluence statistic is working, these classes must be
the leading overlaps (the positive control), and what is left after removing
them is the residual worth tracing.

Tags come from three independent labellers -- SIMBAD ``otype``, the AAVSO VSX
variability type and the Gaia DR3 variability classifier -- plus Gaia's own
multiplicity / variability flags.  Mapping is by explicit code lists, not by
substring luck; a code no list names becomes ``OTHER_VAR`` (for a variable
type) or nothing.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

FAMILIES = ("YSO", "EB", "CV_WDBIN", "WD", "BE_EM", "LPV_AGB", "PULSATOR", "ACTIVE_ROT",
            "GALAXY_QSO", "BINARY", "CHEM_PEC", "OTHER_VAR", "PN_NEB", "XRAY_BIN")

# SIMBAD object types (both the short codes of the current otype system and the
# legacy long names SIMBAD still serves in `otype_txt`).
SIMBAD = {
    "YSO": {"Y*O", "Y*?", "TT*", "TT?", "Or*", "Ae*", "Ae?", "HH", "pr*", "pr?", "out",
            "YSO", "YSO_Candidate", "TTauri*", "TTauri*_Candidate", "Herbig-Haro", "HerbigAe*",
            "Outflow", "OrionV*", "PartofCloud"},
    "EB": {"EB*", "EB?", "Al*", "bL*", "WU*", "EclBin", "EclBin_Candidate", "EllipVar"},
    "CV_WDBIN": {"CV*", "CV?", "DN*", "NL*", "No*", "No?", "Sy*", "Sy?", "CataclyV*",
                 "Symbiotic*", "Nova", "DwarfNova", "NovaLike"},
    "WD": {"WD*", "WD?", "WhiteDwarf", "WhiteDwarf_Candidate", "wd*"},
    "BE_EM": {"Be*", "Be?", "Em*", "BS*", "Emission-line*", "Be*_Candidate", "sg*", "s*b",
              "WR*", "LBV", "LB*"},
    "LPV_AGB": {"LP*", "LP?", "Mi*", "Mi?", "AB*", "AB?", "C*", "C*?", "S*", "S*?", "OH*",
                "OH?", "pA*", "pA?", "post-AGB*", "LongPeriodV*", "Mira", "AGB*", "RGB*",
                "SRV*"},
    "PULSATOR": {"RR*", "RR?", "Ce*", "Ce?", "cC*", "dS*", "bC*", "gD*", "sr*", "ZZ*",
                 "RRLyrae", "Cepheid", "delSctV*", "gammaDorV*", "BetaCepV*", "PulsV*",
                 "Pu*", "sdBV*", "RV*", "RV?", "WV*", "WV?"},
    "ACTIVE_ROT": {"RS*", "RS?", "BY*", "BY?", "Ro*", "Fl*", "fl*", "RSCVnV*", "BYDraV*",
                   "RotV*", "FlareStar", "Er*", "ErptV*"},
    "GALAXY_QSO": {"G", "QSO", "Q?", "AGN", "AG?", "Sy1", "Sy2", "SyG", "BLL", "Bla", "LIN",
                   "rG", "H2G", "EmG", "SBG", "GiG", "GiC", "ClG", "LSB", "bCG", "Galaxy",
                   "Seyfert1", "Seyfert2", "Blazar", "BLLac"},
    "BINARY": {"SB*", "**", "**?", "El*", "El?", "SB?", "Binary", "Spectroscopic Binary",
               "SB*_Candidate", "DoubleStar", "ellipsoidal"},
    "CHEM_PEC": {"CH*", "Ba*", "Ap*", "Ap?", "HgMn*", "Am*", "PM*"},
    "PN_NEB": {"PN", "PN?", "HII", "RNe", "DNe", "SNR", "Cld", "GNe", "ISM", "bub"},
    "XRAY_BIN": {"XB*", "LXB", "HXB", "XB?", "LX?", "HX?", "X", "gam", "Psr"},
    "OTHER_VAR": {"V*", "V*?", "Ir*", "Pu*", "VarStar", "Variable*", "Candidate_V*"},
}
# PM* is "high proper-motion star" in SIMBAD: NOT a peculiar class. Removed below.
SIMBAD["CHEM_PEC"].discard("PM*")

# VSX variability types: the leading code before ':' / '+' / '|'.
VSX = {
    "YSO": {"INS", "INSA", "INSB", "INST", "IN", "INA", "INB", "INT", "ISA", "ISB", "IS",
            "UXOR", "YSO", "TTS", "CTTS", "WTTS", "FUOR", "EXOR", "INSW", "DIP", "CTTS/ROT",
            "BY/TTS"},
    "EB": {"EA", "EB", "EW", "E", "EC", "ED", "ESD", "EP", "AR", "D", "DM", "DS", "DW", "K",
           "KE", "KW", "SD", "GS", "PN+E", "WD+E", "ELL"},
    "CV_WDBIN": {"UG", "UGSS", "UGSU", "UGZ", "UGWZ", "UGER", "NL", "NA", "NB", "NC", "NR",
                 "N", "ZAND", "AM", "IBWD", "CV", "DQ", "VY", "SS", "UX"},
    "WD": {"ZZ", "ZZA", "ZZB", "ZZO", "ZZLep", "V361HYA", "V1093HER", "DYPer"},
    "BE_EM": {"GCAS", "BE", "LERI", "WR", "SDOR", "LBV", "FKCOM", "BLBOO", "EXO"},
    "LPV_AGB": {"M", "SR", "SRA", "SRB", "SRC", "SRD", "SRS", "L", "LB", "LC", "LPV", "RCB",
                "RV", "RVA", "RVB", "PPN", "OSARG", "MISC"},
    "PULSATOR": {"RRAB", "RRC", "RRD", "RR", "DCEP", "DCEPS", "CEP", "CW", "CWA", "CWB",
                 "DSCT", "DSCTC", "HADS", "SXPHE", "GDOR", "BCEP", "BCEPS", "SPB", "ACYG",
                 "ACEP", "ROAP", "ROAM", "PVTEL", "BLAP", "V361HYA", "PULS"},
    "ACTIVE_ROT": {"RS", "BY", "ROT", "UV", "UVN", "FLARE", "ELL/RS", "SXARI", "ACV",
                   "ACVO", "CEP(B)"},
    "OTHER_VAR": {"VAR", "S", "I", "IA", "IB", "NSIN", "APER", "CST", "*", "GAL", "QSO",
                  "BLLAC", "AGN"},
}

GAIA_VARI = {
    "YSO": {"YSO"},
    "EB": {"ECL"},
    "CV_WDBIN": {"CV"},
    "WD": {"WD"},
    "BE_EM": {"BE|GCAS|SDOR|WR", "GCAS", "BE"},
    "LPV_AGB": {"LPV"},
    "PULSATOR": {"RR", "CEP", "DSCT|GDOR|SXPHE", "BCEP", "SPB", "ACV|CP|MCP|ROAM|ROAP|SXARI",
                 "RS"},
    "ACTIVE_ROT": {"SOLAR_LIKE", "RS", "ACV|CP|MCP|ROAM|ROAP|SXARI"},
    "GALAXY_QSO": {"AGN", "GALAXY", "QSO"},
    "OTHER_VAR": {"SN", "MICROLENSING", "S", "EP", "ELL", "AGN"},
}


def _leading_code(t: str) -> str:
    t = str(t).strip().upper()
    return re.split(r"[:+|/ ,()]", t, maxsplit=1)[0] if t else ""


def families_for(simbad_otype=None, vsx_type=None, gaia_class=None,
                 non_single_star=None, phot_variable_flag=None) -> set[str]:
    fam: set[str] = set()
    so = str(simbad_otype).strip() if simbad_otype is not None and \
        str(simbad_otype) not in ("nan", "None", "") else ""
    if so:
        for f, codes in SIMBAD.items():
            if so in codes:
                fam.add(f)
    vt = str(vsx_type).strip() if vsx_type is not None and \
        str(vsx_type) not in ("nan", "None", "") else ""
    if vt:
        codes_here = {c for c in re.split(r"[+|/,]", vt.upper()) if c}
        codes_here = {c.rstrip(":").strip() for c in codes_here}
        hit = False
        for f, codes in VSX.items():
            if codes_here & {c.upper() for c in codes}:
                fam.add(f)
                hit = True
        if not hit:
            fam.add("OTHER_VAR")   # a VSX entry of an unlisted type is still a variable
    gc = str(gaia_class).strip() if gaia_class is not None and \
        str(gaia_class) not in ("nan", "None", "") else ""
    if gc:
        hit = False
        for f, codes in GAIA_VARI.items():
            if gc in codes:
                fam.add(f)
                hit = True
        if not hit:
            fam.add("OTHER_VAR")
    try:
        if non_single_star is not None and int(non_single_star) > 0:
            fam.add("BINARY")
    except (TypeError, ValueError):
        pass
    if str(phot_variable_flag).upper() == "VARIABLE" and not fam:
        fam.add("OTHER_VAR")
    return fam


def tag_table(stars: pd.DataFrame) -> pd.DataFrame:
    """Add ``families`` (';'-joined) and ``known_class`` (bool) to a star table.

    Expects any of ``otype``, ``vsx_type``, ``vari_class``, ``non_single_star``,
    ``phot_variable_flag``; absent columns count as untested, and
    ``tag_coverage`` records which labellers were available for the row.
    """
    out = stars.copy()
    cols = {"otype": "simbad", "vsx_type": "vsx", "vari_class": "gaia_vari",
            "non_single_star": "gaia_nss"}
    fams, cov = [], []
    for _, r in out.iterrows():
        f = families_for(r.get("otype"), r.get("vsx_type"), r.get("vari_class"),
                         r.get("non_single_star"), r.get("phot_variable_flag"))
        fams.append(";".join(sorted(f)))
        cov.append(",".join(v for k, v in cols.items() if k in out.columns))
    out["families"] = fams
    out["known_class"] = np.array([bool(x) for x in fams])
    out["tag_coverage"] = cov
    return out
