"""PARALLAX4 --- Gaia DR4 intake.

Question: grey (achromatic across G/BP/RP) dips or brightenings in Gaia
per-transit photometry that come with NO simultaneous photocentre shift at the
sub-mas level -- proving an anomaly is ON the target, which kills the blend /
neighbour failure mode ~1000x more finely than a TESS pixel.  And the inverse
diagnostic for every other channel's shortlist: does the photocentre move with
the brightness (a blend) or not.

Gaia DR4 (scheduled 2026-12-02) publishes per-transit astrometry and
photometry for every source.  Until then this package runs the photometric
half on Gaia DR3 epoch photometry (the ~11.75 M sources with published light
curves), validates the epoch-astrometry reader on the real June-2026 DR4
prerelease, and keeps an inert DR4 stage that answers DR4_NOT_RELEASED.

Package layout::

    schema.py       run-time column discovery (roles -> spellings), array-cell parser
    epochs.py       readers -> normalised per-transit photometry / astrometry; the join
    greydip.py      the photometric detector (pure)
    photocentre.py  the joint photometry-astrometry blend test (pure)
    simulate.py     DR4-shaped synthetic scenes for the offline suite
    controls.py     positive / negative controls (gate the sweep and the DR4 verdicts)
    watchlist.py    every channel's shortlisted Gaia ids for release day
    acquire.py      runner-only archive access (CDN, DataLink, TAP, prerelease)
    run.py          stages: probe | controls | sweep | reduce | vet | watchlist | dr4 | summary

CLI: ``seti parallax4 --stage ...`` or ``python -m seti.parallax4.run --stage ...``.
"""

from .greydip import GreyConfig, detect_frame, detect_source, grey_test
from .photocentre import PhotocentreConfig, fit_photocentre

__all__ = ["GreyConfig", "PhotocentreConfig", "detect_frame", "detect_source", "fit_photocentre",
           "grey_test"]
