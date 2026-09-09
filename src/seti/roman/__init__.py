"""ROMAN -- intake of Nancy Grace Roman Space Telescope data, and the channels
only Roman can open (``docs/roman.md``).

Roman is not on sky yet (launch readiness no later than May 2027; the
observatory shipped to the launch site in 2026).  This package exists so that
the day its first products are public, every search in this repository can be
run on them the same afternoon -- and so that four signatures no earlier
facility could reach are already built, tested and waiting:

``S40 opaque lens``     (:mod:`~seti.roman.lens`)   a microlens whose physical
                        radius is a non-negligible fraction of its Einstein
                        radius.  No gravitationally bound natural body can be:
                        the implied mean density is orders of magnitude below
                        the least dense object known.  Needs the Galactic Bulge
                        Time Domain Survey's 12-minute cadence and space-based
                        precision on ~10^8 stars.
``S41 industrial line`` (:mod:`~seti.roman.lines`)  an unresolved emission line
                        on a *stellar* point source in the 1.0-1.93 um grism /
                        0.75-1.8 um prism -- the band that holds the two
                        wavelengths our own civilisation's high-power lasers
                        use (Nd:YAG 1.064 um, Er-fibre 1.53-1.57 um) and that
                        every optical laser search to date stops short of.
``S42 sub-exposure flash`` (:mod:`~seti.roman.flash`) a pulse of light shorter
                        than one up-the-ramp resultant, found in the
                        cosmic-ray jump flags as a PSF-shaped cluster centred
                        on a catalogued star.  Roman is the first wide-field
                        survey with non-destructive reads.
``S43 statite``         (:mod:`~seti.roman.statite`) a reflecting point source
                        near a nearby star, in the Coronagraph Instrument's
                        1e-8-contrast images, whose position does not obey
                        Kepler -- a hovering reflector, not a planet.

``bridge`` runs every existing light-curve and spectrum channel (dimming,
glint, secular fade, RUST, KNELL, METRONOME, narrow lines) on Roman products
with Roman-appropriate parameters; ``archive``/``products`` are the intake
(archive discovery, lazy ASDF reads, table adapters); ``run`` is the stage
orchestration; ``schema`` is the data model everything speaks.

Non-negotiables inherited from ``docs/channel-brief.md``: the sandbox has no
archive egress, every detector is pure and offline-tested on synthetic data,
nothing is fabricated, and a run that reaches no data says so in its verdict.
"""

from .schema import (  # noqa: F401
    DQCutout,
    Funnel,
    LightCurve,
    Ramp,
    RomanProduct,
    Spectrum,
    load_roman_config,
)

__all__ = ["DQCutout", "Funnel", "LightCurve", "Ramp", "RomanProduct", "Spectrum",
           "load_roman_config"]
