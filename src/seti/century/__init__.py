"""CENTURY --- cessation, secular fade and rising season-scatter on the DASCH
DR7 century baseline (1885--1992).  Signature S50 (``docs/necrofrontier.md``).

The three detectors are the repository's KNELL (a period that ceased), RUST (a
second moment that rises) and secular-fade statistics, each moved from ZTF's
six years to the Harvard plates' hundred, and each re-derived against the one
systematic that distinguishes a plate archive from a CCD survey: **every plate
has its own limiting magnitude, its own emulsion and its own blending
geometry**, and the archive has a sixteen-year hole (the Menzel gap, no plates
1954--1970) with a photometric offset across it that is modelled as a STEP and
never as a trend.

Layout::

    api.py         DASCH DR7 REST client (runner-only), schema-tolerant
    flags.py       AFLAGS / BFLAGS bit definitions, parsed from the daschlab
                   source at run time with a config fallback
    lightcurve.py  normalisation of an API light curve; plate-limit margins;
                   calendar blocks that carry their non-detections; annual tables
    step.py        the Menzel-gap step + slope model (pure)
    cease.py       fixed-window detection and censoring-aware injection
                   efficiency; the century cessation statistic (pure)
    fade.py        secular fade with the step, margin and series robustness (pure)
    scatter.py     RUST season scatter on plates, step-aware (pure)
    targets.py     VSX / GCVS periodic variables and bright stars per field (runner)
    vet.py         the contamination gauntlet (pure)
    run.py         stages: probe, targets, acquire, screen, assess, all
"""

from __future__ import annotations

__all__: list[str] = []
