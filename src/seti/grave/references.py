"""Reference reservoir vectors for GRAVE, in ppm by mass, with their sources.

Every number here is a *composition a natural process can deliver to a
sediment*, and the mixture fit in :mod:`seti.grave.vectors` is allowed to add
any non-negative amount of each.  The list is deliberately the list of the
brief's kills: detrital provenance (felsic, mafic), heavy minerals (zircon,
monazite), phosphate, carbonate, the authigenic redox enrichment of black
shales, Fe-Mn oxyhydroxide (the particulate shuttle that puts Mo, Te, Pt, Ce
up *without* U), hydrothermal sulfide/barite, and chondrite (impact).  Fission
product is then the one component *outside* that family, and the question per
sample is whether adding it explains the chemistry better than any mixture of
the natural family alone.

Precision.  The compilations differ from one another at the 10-30 % level for
most trace elements and by factors of a few for the ultra-trace ones (PGE,
Te, Ag, Cd, Re); the model's per-element error floors (measured on the control
population) are always wider than that, so refining these tables would change
nothing observable.  Where a compilation lacks an element, the value is taken
from the nearest analogue and the source column says so.

Sources
-------
PAAS      Post-Archaean Australian Shale, Taylor & McLennan 1985 (The
          Continental Crust: its Composition and Evolution, Blackwell) with
          the McLennan 2001 (G-cubed 2, 2000GC000109) revisions of Nb/Ta/Cs;
          elements PAAS does not list are UCC values (flagged ``ucc``).
UCC       upper continental crust, Rudnick & Gao 2003 (Treatise on
          Geochemistry 3, 1-64); PGE from Peucker-Ehrenbrink & Jahn 2001
          (G-cubed 2, 2001GC000172).
MORB      all-MORB mean, Gale et al. 2013 (G-cubed 14, 489); PGE Bezos et
          al. 2005 (GCA 69, 2613).
CI        CI chondrite, Lodders 2003 (ApJ 591, 1220) with McDonough & Sun 1995
          (Chem. Geol. 120, 223) for the lithophile trace elements.
BLACKSH   world black-shale "Clarke" values, Ketris & Yudovich 2009 (Int. J.
          Coal Geol. 78, 135).  The AUTHIGENIC vector is max(BLACKSH - PAAS, 0)
          for the redox-sensitive elements: what anoxia adds, not the shale.
FEMN      hydrogenetic Fe-Mn crust, Pacific prime zone mean, Hein &
          Koschinsky 2014 (Treatise on Geochemistry 2nd ed. 13, 273).
HYDRO     SEDEX / VMS metalliferous sediment, order-of-magnitude composite
          from Large et al. 2005 (Econ. Geol. 100th Anniv. 931) and
          Hannington 2014 (Treatise 13, 463): barite + sulfide with the
          As-Sb-Ag-Cd-Tl-Hg suite and a positive Eu anomaly (Michard 1989,
          GCA 53, 745, for the fluid REE).
ZIRCON    detrital zircon, Hoskin & Schaltegger 2003 (Rev. Mineral. Geochem.
          53, 27) and Belousova et al. 2002 (Contrib. Mineral. Petrol. 143,
          602); Zr/Hf ~ 40.
MONAZITE  detrital monazite, Williams, Jercinovic & Hetherington 2007
          (Annu. Rev. Earth Planet. Sci. 35, 137); Th 5 wt%.
APATITE   marine phosphorite / bioapatite, Altschuler 1980 (SEPM Spec. Publ.
          29, 19) and Emsbo et al. 2015 (Gondwana Res. 27, 776) for the REE.
CARB      average limestone, Turekian & Wedepohl 1961 (GSA Bull. 72, 175).
RHYOLITE  average rhyolite / distal tephra, Le Maitre 1976 (J. Petrol. 17,
          589) with the GEOROC rhyolite median for the trace elements; Nb/Ta,
          Zr/Hf crustal.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Atomic masses (IUPAC 2013 conventional), for chain-yield -> mass conversion
# ---------------------------------------------------------------------------
ATOMIC_MASS: dict[str, float] = {
    "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867,
    "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693,
    "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971,
    "Br": 79.904, "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Tc": 98.0, "Ru": 101.07, "Rh": 102.906, "Pd": 106.42,
    "Ag": 107.868, "Cd": 112.414, "In": 114.818, "Sn": 118.710, "Sb": 121.760,
    "Te": 127.60, "I": 126.904, "Xe": 131.293, "Cs": 132.905, "Ba": 137.327,
    "La": 138.905, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242, "Pm": 145.0,
    "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.925, "Dy": 162.500,
    "Ho": 164.930, "Er": 167.259, "Tm": 168.934, "Yb": 173.045, "Lu": 174.967,
    "Hf": 178.49, "Ta": 180.948, "W": 183.84, "Re": 186.207, "Os": 190.23,
    "Ir": 192.217, "Pt": 195.084, "Au": 196.967, "Hg": 200.592, "Tl": 204.38,
    "Pb": 207.2, "Bi": 208.980, "Th": 232.038, "U": 238.029,
}

#: Oxide -> (element, mass fraction of the element in the oxide).
OXIDE_TO_ELEMENT: dict[str, tuple[str, float]] = {
    "SiO2": ("Si", 0.4674), "TiO2": ("Ti", 0.5995), "Al2O3": ("Al", 0.5293),
    "Fe2O3": ("Fe", 0.6994), "Fe2O3T": ("Fe", 0.6994), "FeO": ("Fe", 0.7773),
    "FeOT": ("Fe", 0.7773), "MnO": ("Mn", 0.7745), "MgO": ("Mg", 0.6030),
    "CaO": ("Ca", 0.7147), "Na2O": ("Na", 0.7419), "K2O": ("K", 0.8301),
    "P2O5": ("P", 0.4364), "Cr2O3": ("Cr", 0.6842), "NiO": ("Ni", 0.7858),
    "BaO": ("Ba", 0.8957), "SrO": ("Sr", 0.8456), "ZrO2": ("Zr", 0.7403),
}

# ---------------------------------------------------------------------------
# The element universe of the channel
# ---------------------------------------------------------------------------
#: Major elements (used to pin the reservoir fractions: Al detrital, Ca
#: carbonate, P apatite, Mn/Fe oxides, Ti mafic/ash).
MAJORS: tuple[str, ...] = ("Al", "Ti", "Fe", "Mn", "Mg", "Ca", "Na", "K", "P")

#: Fission-peak elements a sedimentary analysis can carry (the light peak
#: Zr-Mo-Ru-Rh-Pd, the valley Ag-Cd-Sn-Sb-Te-I, the heavy peak Cs-Ba-La-Ce-Pr-
#: Nd-Sm-Eu, and the Gd cliff).
FISSION_ELEMENTS: tuple[str, ...] = (
    "Rb", "Sr", "Y", "Zr", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "Sn", "Sb", "Te", "I",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy",
)

#: Every element the mixture fit may use (majors + fission + the reservoir
#: tracers that pin each natural component).
FIT_ELEMENTS: tuple[str, ...] = MAJORS + FISSION_ELEMENTS + (
    "Sc", "V", "Cr", "Co", "Ni", "Cu", "Zn", "Ga", "As", "Se", "Nb", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Th", "U",
    "Ho", "Er", "Tm", "Yb", "Lu", "Li", "Be", "B",
)

#: The platinum-group + alloy panel of the refined-particulate test.
PGE: tuple[str, ...] = ("Os", "Ir", "Ru", "Rh", "Pt", "Pd")
ALLOY_PANEL: tuple[str, ...] = ("Nb", "Ta", "W", "Sn", "Mo", "Bi", "Cr", "Ni", "Co", "V", "Ti")

#: Redox proxies (authigenic enrichment in anoxic / euxinic water columns).
REDOX_PROXIES: tuple[str, ...] = ("U", "V", "Mo", "Re", "Ni", "Cu", "Zn", "Cd", "Se", "Tl")

# ---------------------------------------------------------------------------
# Reservoirs, ppm by mass (wt% x 10^4).  None = not tabulated in the source.
# ---------------------------------------------------------------------------
PAAS: dict[str, float] = {
    # majors: SiO2 62.8, TiO2 1.00, Al2O3 18.9, Fe2O3T 7.22, MnO 0.11, MgO 2.2,
    # CaO 1.3, Na2O 1.2, K2O 3.7, P2O5 0.16 (wt%)
    "Al": 100_000, "Ti": 6_000, "Fe": 50_500, "Mn": 850, "Mg": 13_300, "Ca": 9_300,
    "Na": 8_900, "K": 30_700, "P": 700,
    "Li": 75, "Be": 3.0, "B": 100, "Sc": 16, "V": 150, "Cr": 110, "Co": 23, "Ni": 55,
    "Cu": 50, "Zn": 85, "Ga": 20, "As": 4.8, "Se": 0.09,           # As, Se: ucc
    "Rb": 160, "Sr": 200, "Y": 27, "Zr": 210, "Nb": 19, "Mo": 1.0,
    "Ru": 0.00034, "Rh": 0.00006, "Pd": 0.00052,                    # ucc (PGE)
    "Ag": 0.053, "Cd": 0.09, "Sn": 4.0, "Sb": 0.4, "Te": 0.003, "I": 1.4,  # Ag/Cd/Sb/Te/I: ucc
    "Cs": 15, "Ba": 650, "La": 38.2, "Ce": 79.6, "Pr": 8.83, "Nd": 33.9, "Sm": 5.55,
    "Eu": 1.08, "Gd": 4.66, "Tb": 0.774, "Dy": 4.68, "Ho": 0.991, "Er": 2.85,
    "Tm": 0.405, "Yb": 2.82, "Lu": 0.433, "Hf": 5.0, "Ta": 1.28, "W": 2.7,
    "Re": 0.0002, "Os": 0.000031, "Ir": 0.000022, "Pt": 0.0005, "Au": 0.0015,  # ucc
    "Hg": 0.05, "Tl": 0.9, "Pb": 20, "Bi": 0.16, "Th": 14.6, "U": 3.1,
}
#: Which PAAS entries are borrowed from UCC (recorded in ``summary.json``).
PAAS_FROM_UCC: tuple[str, ...] = ("As", "Se", "Ru", "Rh", "Pd", "Ag", "Cd", "Sb", "Te", "I",
                                  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Bi")

UCC: dict[str, float] = {
    "Al": 81_500, "Ti": 3_840, "Fe": 39_200, "Mn": 774, "Mg": 15_000, "Ca": 25_700,
    "Na": 24_300, "K": 23_200, "P": 655,
    "Li": 21, "Be": 2.1, "B": 17, "Sc": 14, "V": 97, "Cr": 92, "Co": 17.3, "Ni": 47,
    "Cu": 28, "Zn": 67, "Ga": 17.5, "As": 4.8, "Se": 0.09, "Rb": 84, "Sr": 320, "Y": 21,
    "Zr": 193, "Nb": 12, "Mo": 1.1, "Ru": 0.00034, "Rh": 0.00006, "Pd": 0.00052,
    "Ag": 0.053, "Cd": 0.09, "Sn": 2.1, "Sb": 0.4, "Te": 0.003, "I": 1.4, "Cs": 4.9,
    "Ba": 624, "La": 31, "Ce": 63, "Pr": 7.1, "Nd": 27, "Sm": 4.7, "Eu": 1.0, "Gd": 4.0,
    "Tb": 0.7, "Dy": 3.9, "Ho": 0.83, "Er": 2.3, "Tm": 0.30, "Yb": 1.96, "Lu": 0.31,
    "Hf": 5.3, "Ta": 0.9, "W": 1.9, "Re": 0.0002, "Os": 0.000031, "Ir": 0.000022,
    "Pt": 0.0005, "Au": 0.0015, "Hg": 0.05, "Tl": 0.9, "Pb": 17, "Bi": 0.16,
    "Th": 10.5, "U": 2.7,
}

MORB: dict[str, float] = {
    "Al": 80_000, "Ti": 10_000, "Fe": 78_000, "Mn": 1_400, "Mg": 46_000, "Ca": 82_000,
    "Na": 20_000, "K": 1_500, "P": 600,
    "Li": 6.6, "Be": 0.6, "B": 1.5, "Sc": 38, "V": 300, "Cr": 250, "Co": 44, "Ni": 100,
    "Cu": 74, "Zn": 87, "Ga": 17, "As": 0.2, "Se": 0.2, "Rb": 4.05, "Sr": 138, "Y": 32.4,
    "Zr": 103, "Nb": 5.2, "Mo": 0.43, "Ru": 0.0002, "Rh": 0.0001, "Pd": 0.0005,
    "Ag": 0.02, "Cd": 0.1, "Sn": 1.2, "Sb": 0.02, "Te": 0.005, "I": 0.02, "Cs": 0.053,
    "Ba": 29.2, "La": 4.19, "Ce": 12.4, "Pr": 2.0, "Nd": 10.7, "Sm": 3.48, "Eu": 1.26,
    "Gd": 4.5, "Tb": 0.82, "Dy": 5.4, "Ho": 1.19, "Er": 3.4, "Tm": 0.51, "Yb": 3.3,
    "Lu": 0.49, "Hf": 2.62, "Ta": 0.31, "W": 0.05, "Re": 0.001, "Os": 0.00003,
    "Ir": 0.00003, "Pt": 0.0004, "Au": 0.001, "Hg": 0.01, "Tl": 0.01, "Pb": 0.57,
    "Bi": 0.01, "Th": 0.40, "U": 0.13,
}

CI_CHONDRITE: dict[str, float] = {
    "Al": 8_500, "Ti": 440, "Fe": 182_000, "Mn": 1_930, "Mg": 96_000, "Ca": 9_200,
    "Na": 5_000, "K": 550, "P": 1_000,
    "Li": 1.5, "Be": 0.025, "B": 0.9, "Sc": 5.8, "V": 55, "Cr": 2_650, "Co": 502,
    "Ni": 10_500, "Cu": 126, "Zn": 312, "Ga": 9.7, "As": 1.8, "Se": 21, "Rb": 2.3,
    "Sr": 7.8, "Y": 1.56, "Zr": 3.9, "Nb": 0.25, "Mo": 0.92, "Ru": 0.714, "Rh": 0.134,
    "Pd": 0.556, "Ag": 0.20, "Cd": 0.68, "Sn": 1.7, "Sb": 0.14, "Te": 2.3, "I": 0.43,
    "Cs": 0.19, "Ba": 2.4, "La": 0.24, "Ce": 0.61, "Pr": 0.093, "Nd": 0.46, "Sm": 0.15,
    "Eu": 0.056, "Gd": 0.20, "Tb": 0.037, "Dy": 0.25, "Ho": 0.056, "Er": 0.16,
    "Tm": 0.025, "Yb": 0.16, "Lu": 0.025, "Hf": 0.10, "Ta": 0.014, "W": 0.09,
    "Re": 0.037, "Os": 0.495, "Ir": 0.470, "Pt": 1.00, "Au": 0.146, "Hg": 0.3,
    "Tl": 0.14, "Pb": 2.6, "Bi": 0.11, "Th": 0.03, "U": 0.008,
}

#: Ketris & Yudovich 2009 black-shale Clarkes (whole rock).
BLACK_SHALE: dict[str, float] = {
    "Al": 88_000, "Ti": 4_500, "Fe": 40_000, "Mn": 400, "Mg": 12_000, "Ca": 15_000,
    "Na": 7_000, "K": 26_000, "P": 1_000,
    "Li": 30, "Be": 2.0, "B": 60, "Sc": 14, "V": 205, "Cr": 96, "Co": 19, "Ni": 70,
    "Cu": 70, "Zn": 130, "Ga": 18, "As": 30, "Se": 8.7, "Rb": 110, "Sr": 180, "Y": 26,
    "Zr": 180, "Nb": 13, "Mo": 20, "Ru": 0.0005, "Rh": 0.0001, "Pd": 0.002,
    "Ag": 1.0, "Cd": 5.0, "Sn": 4.0, "Sb": 5.0, "Te": 0.01, "I": 3.0, "Cs": 6.0,
    "Ba": 500, "La": 30, "Ce": 60, "Pr": 7.0, "Nd": 26, "Sm": 5.0, "Eu": 1.1, "Gd": 4.5,
    "Tb": 0.7, "Dy": 4.0, "Ho": 0.85, "Er": 2.5, "Tm": 0.38, "Yb": 2.6, "Lu": 0.4,
    "Hf": 4.0, "Ta": 1.0, "W": 2.7, "Re": 0.007, "Os": 0.0001, "Ir": 0.00005,
    "Pt": 0.003, "Au": 0.007, "Hg": 0.27, "Tl": 1.0, "Pb": 21, "Bi": 1.0, "Th": 11, "U": 8.5,
}

#: Hein & Koschinsky 2014, Pacific prime crust zone.
FEMN_CRUST: dict[str, float] = {
    "Al": 10_000, "Ti": 11_600, "Fe": 169_000, "Mn": 228_000, "Mg": 10_000, "Ca": 22_000,
    "Na": 15_000, "K": 4_000, "P": 4_000,
    "Li": 4.0, "Be": 5.0, "B": 200, "Sc": 10, "V": 640, "Cr": 20, "Co": 6_600, "Ni": 4_200,
    "Cu": 980, "Zn": 680, "Ga": 10, "As": 300, "Se": 1.0, "Rb": 20, "Sr": 1_600, "Y": 220,
    "Zr": 590, "Nb": 50, "Mo": 460, "Ru": 0.02, "Rh": 0.02, "Pd": 0.005, "Ag": 0.1,
    "Cd": 3.0, "Sn": 8.0, "Sb": 40, "Te": 60, "I": 50, "Cs": 1.0, "Ba": 1_600,
    "La": 300, "Ce": 1_300, "Pr": 60, "Nd": 250, "Sm": 55, "Eu": 13, "Gd": 60, "Tb": 9,
    "Dy": 50, "Ho": 10, "Er": 28, "Tm": 4, "Yb": 26, "Lu": 4, "Hf": 9.0, "Ta": 1.0,
    "W": 82, "Re": 0.001, "Os": 0.002, "Ir": 0.005, "Pt": 0.47, "Au": 0.003,
    "Hg": 0.1, "Tl": 150, "Pb": 1_600, "Bi": 30, "Th": 30, "U": 12,
}

#: SEDEX / VMS metalliferous sediment: barite + Fe-Zn-Pb-Cu sulfide with the
#: epithermal suite; order-of-magnitude composite (see module docstring).
HYDROTHERMAL: dict[str, float] = {
    "Al": 20_000, "Ti": 1_000, "Fe": 300_000, "Mn": 10_000, "Mg": 10_000, "Ca": 20_000,
    "Na": 3_000, "K": 5_000, "P": 500,
    "Li": 10, "Be": 1.0, "B": 50, "Sc": 3, "V": 50, "Cr": 20, "Co": 100, "Ni": 50,
    "Cu": 2_000, "Zn": 5_000, "Ga": 20, "As": 500, "Se": 20, "Rb": 20, "Sr": 500,
    "Y": 10, "Zr": 30, "Nb": 2, "Mo": 20, "Ru": 0.0005, "Rh": 0.0002, "Pd": 0.002,
    "Ag": 20, "Cd": 50, "Sn": 20, "Sb": 100, "Te": 1.0, "I": 1.0, "Cs": 2.0,
    "Ba": 50_000, "La": 5, "Ce": 8, "Pr": 1.0, "Nd": 4, "Sm": 0.8, "Eu": 0.8,   # Eu/Eu* ~ +5
    "Gd": 0.8, "Tb": 0.12, "Dy": 0.7, "Ho": 0.15, "Er": 0.4, "Tm": 0.06, "Yb": 0.4,
    "Lu": 0.06, "Hf": 0.8, "Ta": 0.1, "W": 10, "Re": 0.005, "Os": 0.0002, "Ir": 0.0001,
    "Pt": 0.005, "Au": 0.5, "Hg": 2.0, "Tl": 10, "Pb": 2_000, "Bi": 10, "Th": 2, "U": 3,
}

ZIRCON: dict[str, float] = {
    "Al": 100, "Ti": 10, "Fe": 500, "Mn": 20, "Mg": 20, "Ca": 100, "Na": 20, "K": 20,
    "P": 500, "Li": 1, "Be": 1, "B": 1, "Sc": 50, "V": 5, "Cr": 5, "Co": 1, "Ni": 2,
    "Cu": 2, "Zn": 5, "Ga": 1, "As": 0.5, "Se": 0.01, "Rb": 0.5, "Sr": 1, "Y": 2_000,
    "Zr": 497_000, "Nb": 20, "Mo": 0.5, "Ru": 0.0001, "Rh": 0.00002, "Pd": 0.0001,
    "Ag": 0.01, "Cd": 0.01, "Sn": 2, "Sb": 0.1, "Te": 0.001, "I": 0.1, "Cs": 0.1,
    "Ba": 2, "La": 0.5, "Ce": 10, "Pr": 0.5, "Nd": 3, "Sm": 5, "Eu": 1.5, "Gd": 25,
    "Tb": 8, "Dy": 100, "Ho": 40, "Er": 180, "Tm": 40, "Yb": 300, "Lu": 60,
    "Hf": 12_000, "Ta": 2, "W": 1, "Re": 0.0001, "Os": 0.00001, "Ir": 0.00001,
    "Pt": 0.0001, "Au": 0.001, "Hg": 0.01, "Tl": 0.05, "Pb": 20, "Bi": 0.1,
    "Th": 150, "U": 300,
}

MONAZITE: dict[str, float] = {
    "Al": 500, "Ti": 200, "Fe": 2_000, "Mn": 100, "Mg": 100, "Ca": 8_000, "Na": 100,
    "K": 100, "P": 125_000, "Li": 1, "Be": 1, "B": 1, "Sc": 10, "V": 10, "Cr": 10,
    "Co": 2, "Ni": 5, "Cu": 5, "Zn": 10, "Ga": 2, "As": 5, "Se": 0.1, "Rb": 1, "Sr": 300,
    "Y": 10_000, "Zr": 200, "Nb": 10, "Mo": 1, "Ru": 0.0002, "Rh": 0.00005,
    "Pd": 0.0002, "Ag": 0.05, "Cd": 0.1, "Sn": 5, "Sb": 0.5, "Te": 0.005, "I": 0.5,
    "Cs": 0.5, "Ba": 50, "La": 120_000, "Ce": 240_000, "Pr": 27_000, "Nd": 90_000,
    "Sm": 15_000, "Eu": 500, "Gd": 6_000, "Tb": 500, "Dy": 1_500, "Ho": 150, "Er": 250,
    "Tm": 20, "Yb": 80, "Lu": 8, "Hf": 20, "Ta": 2, "W": 5, "Re": 0.0005,
    "Os": 0.00005, "Ir": 0.00005, "Pt": 0.0005, "Au": 0.002, "Hg": 0.05, "Tl": 0.2,
    "Pb": 2_000, "Bi": 2, "Th": 50_000, "U": 2_000,
}

APATITE: dict[str, float] = {
    "Al": 5_000, "Ti": 500, "Fe": 10_000, "Mn": 300, "Mg": 5_000, "Ca": 340_000,
    "Na": 5_000, "K": 2_000, "P": 130_000, "Li": 5, "Be": 1, "B": 20, "Sc": 10, "V": 100,
    "Cr": 100, "Co": 5, "Ni": 50, "Cu": 30, "Zn": 200, "Ga": 5, "As": 20, "Se": 5,
    "Rb": 10, "Sr": 1_000, "Y": 300, "Zr": 50, "Nb": 5, "Mo": 5, "Ru": 0.0005,
    "Rh": 0.0001, "Pd": 0.001, "Ag": 2, "Cd": 20, "Sn": 2, "Sb": 3, "Te": 0.01,
    "I": 20, "Cs": 1, "Ba": 300, "La": 100, "Ce": 100, "Pr": 20, "Nd": 100, "Sm": 20,
    "Eu": 5, "Gd": 25, "Tb": 4, "Dy": 25, "Ho": 5, "Er": 15, "Tm": 2, "Yb": 12, "Lu": 2,
    "Hf": 1, "Ta": 0.3, "W": 2, "Re": 0.005, "Os": 0.0001, "Ir": 0.0001, "Pt": 0.002,
    "Au": 0.005, "Hg": 0.1, "Tl": 1, "Pb": 20, "Bi": 0.5, "Th": 5, "U": 100,
}

CARBONATE: dict[str, float] = {
    "Al": 4_200, "Ti": 400, "Fe": 3_800, "Mn": 1_100, "Mg": 47_000, "Ca": 302_000,
    "Na": 400, "K": 2_700, "P": 400, "Li": 5, "Be": 0.1, "B": 20, "Sc": 1, "V": 20,
    "Cr": 11, "Co": 0.1, "Ni": 20, "Cu": 4, "Zn": 20, "Ga": 4, "As": 1, "Se": 0.08,
    "Rb": 3, "Sr": 610, "Y": 30, "Zr": 19, "Nb": 0.3, "Mo": 0.4, "Ru": 0.00005,
    "Rh": 0.00001, "Pd": 0.0001, "Ag": 0.01, "Cd": 0.035, "Sn": 0.5, "Sb": 0.2,
    "Te": 0.001, "I": 1.0, "Cs": 0.5, "Ba": 10, "La": 1.0, "Ce": 1.5, "Pr": 0.2,
    "Nd": 0.9, "Sm": 0.2, "Eu": 0.04, "Gd": 0.2, "Tb": 0.03, "Dy": 0.2, "Ho": 0.04,
    "Er": 0.1, "Tm": 0.015, "Yb": 0.1, "Lu": 0.015, "Hf": 0.3, "Ta": 0.03, "W": 0.6,
    "Re": 0.0005, "Os": 0.00001, "Ir": 0.00001, "Pt": 0.0001, "Au": 0.002, "Hg": 0.04,
    "Tl": 0.05, "Pb": 9, "Bi": 0.05, "Th": 1.7, "U": 2.2,
}

RHYOLITE: dict[str, float] = {
    "Al": 70_000, "Ti": 2_000, "Fe": 15_000, "Mn": 500, "Mg": 3_000, "Ca": 8_000,
    "Na": 27_000, "K": 35_000, "P": 300, "Li": 30, "Be": 3, "B": 20, "Sc": 5, "V": 20,
    "Cr": 10, "Co": 3, "Ni": 5, "Cu": 8, "Zn": 60, "Ga": 18, "As": 3, "Se": 0.05,
    "Rb": 150, "Sr": 100, "Y": 40, "Zr": 250, "Nb": 20, "Mo": 2, "Ru": 0.0001,
    "Rh": 0.00003, "Pd": 0.0003, "Ag": 0.05, "Cd": 0.1, "Sn": 3, "Sb": 0.3,
    "Te": 0.002, "I": 0.5, "Cs": 5, "Ba": 600, "La": 40, "Ce": 80, "Pr": 9, "Nd": 35,
    "Sm": 7, "Eu": 0.7, "Gd": 6, "Tb": 1.0, "Dy": 6, "Ho": 1.3, "Er": 4, "Tm": 0.6,
    "Yb": 4, "Lu": 0.6, "Hf": 7, "Ta": 1.5, "W": 2, "Re": 0.0005, "Os": 0.00002,
    "Ir": 0.00002, "Pt": 0.0003, "Au": 0.001, "Hg": 0.03, "Tl": 1, "Pb": 20, "Bi": 0.2,
    "Th": 15, "U": 4,
}


def authigenic_vector() -> dict[str, float]:
    """What anoxia *adds*: max(black shale - PAAS, 0) on the redox-sensitive suite.

    Kept to the elements the authigenic literature actually attributes to
    water-column drawdown (Tribovillard et al. 2006, Chem. Geol. 232, 12;
    Algeo & Tribovillard 2009, Chem. Geol. 268, 211), so that the component
    cannot double as a second detrital shale.
    """
    keys = ("U", "V", "Mo", "Re", "Ni", "Cu", "Zn", "Cd", "Se", "Tl", "As", "Sb",
            "Ag", "Hg", "Au", "Cr", "Co", "Bi", "I", "Pd", "Pt")
    out: dict[str, float] = {}
    for el in keys:
        d = BLACK_SHALE.get(el, 0.0) - PAAS.get(el, 0.0)
        if d > 0:
            out[el] = float(d)
    # A pinch of organic-associated P and the "no majors" convention: the
    # vector has no Al, so it cannot absorb detrital chemistry.
    out["P"] = 300.0
    return out


#: The natural family.  Order matters only for reporting.
RESERVOIRS: dict[str, dict[str, float]] = {
    "paas": PAAS,
    "ucc_felsic": UCC,
    "morb_mafic": MORB,
    "zircon": ZIRCON,
    "monazite": MONAZITE,
    "apatite": APATITE,
    "carbonate": CARBONATE,
    "authigenic": authigenic_vector(),
    "femn_oxide": FEMN_CRUST,
    "hydrothermal": HYDROTHERMAL,
    "chondrite": CI_CHONDRITE,
    "rhyolite": RHYOLITE,
}

RESERVOIR_SOURCES: dict[str, str] = {
    "paas": "Taylor & McLennan 1985; McLennan 2001 (UCC where PAAS is silent: "
            + ", ".join(PAAS_FROM_UCC) + ")",
    "ucc_felsic": "Rudnick & Gao 2003; PGE Peucker-Ehrenbrink & Jahn 2001",
    "morb_mafic": "Gale et al. 2013; PGE Bezos et al. 2005",
    "zircon": "Hoskin & Schaltegger 2003; Belousova et al. 2002",
    "monazite": "Williams, Jercinovic & Hetherington 2007",
    "apatite": "Altschuler 1980; Emsbo et al. 2015",
    "carbonate": "Turekian & Wedepohl 1961",
    "authigenic": "max(Ketris & Yudovich 2009 black shale - PAAS, 0) on the Tribovillard 2006 suite",
    "femn_oxide": "Hein & Koschinsky 2014 (hydrogenetic crust, Pacific prime zone)",
    "hydrothermal": "Large et al. 2005; Hannington 2014; Michard 1989 (composite, order of magnitude)",
    "chondrite": "Lodders 2003; McDonough & Sun 1995",
    "rhyolite": "Le Maitre 1976; GEOROC rhyolite median (composite)",
}

#: Chondritic PGE ratios to Ir (CI, Lodders 2003): the impact fingerprint.
CI_PGE_TO_IR: dict[str, float] = {
    "Os": CI_CHONDRITE["Os"] / CI_CHONDRITE["Ir"],
    "Ru": CI_CHONDRITE["Ru"] / CI_CHONDRITE["Ir"],
    "Rh": CI_CHONDRITE["Rh"] / CI_CHONDRITE["Ir"],
    "Pt": CI_CHONDRITE["Pt"] / CI_CHONDRITE["Ir"],
    "Pd": CI_CHONDRITE["Pd"] / CI_CHONDRITE["Ir"],
}

__all__ = [
    "ALLOY_PANEL", "APATITE", "ATOMIC_MASS", "BLACK_SHALE", "CARBONATE", "CI_CHONDRITE",
    "CI_PGE_TO_IR", "FEMN_CRUST", "FISSION_ELEMENTS", "FIT_ELEMENTS", "HYDROTHERMAL",
    "MAJORS", "MONAZITE", "MORB", "OXIDE_TO_ELEMENT", "PAAS", "PAAS_FROM_UCC", "PGE",
    "REDOX_PROXIES", "RESERVOIRS", "RESERVOIR_SOURCES", "RHYOLITE", "UCC", "ZIRCON",
    "authigenic_vector",
]
