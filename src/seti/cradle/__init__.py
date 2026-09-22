"""CRADLE --- the shattered cradle (S52) and slag, not glass (S53).

Warm debris at the habitable-zone radius of a MATURE star, above the
collisional steady-state maximum for the star's age.  A destroyed or
disassembled planet leaves warm dust at its orbit; nature's own extreme debris
disks are young or hot, and the cell **250 <= T_bb <= 350 K, log(f/f_max) > 3,
age > 1 Gyr** is empty in the literature (docs/necrofrontier.md S52).

Package layout::

    sample.py       the parent selection as ADQL over HEALPix source_id ranges
    acquire.py      runner-only archive calls (ESA Gaia join, IRSA, VizieR, NEOWISE, IRS)
    excess.py       photospheric locus, W3/W4 excess, T_bb, f = L_IR/L_*, Wyatt f_max
    ages.py         the independent age indicators and the young-group veto
    vet.py          the kill list, every rule with a counter
    mineralogy.py   the S53 feature score on a Spitzer/IRS spectrum
    run.py          stages: probe | acquire | screen | ages | assess | all

CLI: ``seti cradle --stage {probe,acquire,screen,ages,assess,all} --shard i/n``.
"""

from .excess import (
    blackbody_radius_au,
    disk_fraction,
    fit_disk,
    log_f_over_fmax,
    wyatt_fmax,
)

__all__ = ["blackbody_radius_au", "disk_fraction", "fit_disk", "log_f_over_fmax",
           "wyatt_fmax"]
