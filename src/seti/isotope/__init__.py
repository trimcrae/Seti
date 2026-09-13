"""ISOTOPE — the impossible grain (S46, ``docs/necrofrontier.md`` §2.XII).

Isotopic *purity without a nucleosynthetic package* in the Presolar Grain
Database.  Natural solids carry isotope anomalies only as correlated
packages (a supernova X grain is ²⁸Si-rich AND carries ⁴⁴Ti, ²⁶Al, ¹⁵N, ¹²C)
or as per-mil mass-dependent fractionation.  Refined material is
mono-isotopic with *solar* partners.  The channel screens every grain for
²⁸Si (or ¹²C) purity beyond the database's own most extreme classified X
grain with every measured partner ratio consistent with solar.

Modules
-------
``acquire``  route ledger (human URL → local path → wustl crawl → EarthChem
             Library search → DataCite lookup), table parsing (csv / xlsx
             without openpyxl / zip), runtime header→role resolution.
``purity``   the detector: empirical X-grain envelope, partner tests, grain
             classes, the carbon analogue — pure functions.
``run``      ``main(argv)`` with ``--stage {probe,acquire,screen,assess,all}``;
             writes ``results/isotope/summary.json`` and ``candidates.csv``.

``main`` is resolved lazily so ``python -m seti.isotope.run`` does not import
the run module twice.
"""

from __future__ import annotations


def main(argv=None):
    from .run import main as _main
    return _main(argv)


__all__ = ["main"]
