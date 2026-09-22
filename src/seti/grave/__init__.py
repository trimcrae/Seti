"""GRAVE — a technological extinction in Earth's own sedimentary record (S56).

The question (``docs/necrofrontier.md`` S56, ``docs/grave.md``): does any mass
extinction boundary carry a *technological* residue that impact, volcanism,
redox, hydrothermal input and detrital mineralogy cannot make?  Two residues
are searched, per sample, in the public whole-rock chemistry of sedimentary
rocks (SGP, EarthChem; GEOROC as the volcanic reference):

1. the **fission-product vector** — the two-humped fission-yield mass pattern
   (``seti.fallout.yields``) folded against a PAAS baseline and fitted as one
   extra non-negative component on top of the best natural reservoir mixture
   (felsic / mafic detritus, zircon, monazite, apatite, carbonate, authigenic
   redox, Fe-Mn oxyhydroxide, hydrothermal sulfide, chondrite);
2. the **refined-particulate vector** — platinum-group and alloy elements in
   ratios no natural concentration mechanism produces (Pt/Pd/Rh without Ir and
   Os; Ru/Rh/Pd without Pt; Ta without Nb; W without Sn/Mo/Bi), with the
   chondritic Ir-anchored pattern classed ``impact`` and IPGE-rich patterns
   classed ``ultramafic``.

Then an **age stack**: a residue that is real at a boundary recurs at the same
stratigraphic level in independent sections, the way the K-Pg iridium does; a
one-section one-sample outlier is contamination until shown otherwise.

Package layout: ``references`` (reservoir vectors with citations),
``vectors`` (the physics, pure functions), ``agestack`` (boundaries and the
clustering statistic), ``acquire`` (SGP / EarthChem / GEOROC clients with
runtime schema discovery), ``run`` (stages, ``results/grave/``).
"""

from __future__ import annotations

import os as _os

# Pin the BLAS thread pool before NumPy is imported by any submodule.  Every
# matrix in this channel is tiny (a design is of order 45 x 13), so a threaded
# BLAS spends its time synchronising workers: measured, one fit went from 7.0 s
# to 0.076 s for a bit-identical answer.  `setdefault` so an operator who sets
# these deliberately still wins.  `vectors.single_threaded` enforces the same
# thing at run time for callers that imported NumPy before this ran.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

__all__ = ["acquire", "agestack", "references", "run", "vectors"]
