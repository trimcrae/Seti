"""IGNITION --- an infrared excess being born on an old star (S61).

The claim (docs/necrofrontier.md §2 XIV, "Transition residue"): a takeover's
resource-acquisition phase is exponential over decades, so its residue in a
decade-baseline archive is a **monotonic W1/W2 rise sustained for >= 5 yr** on
a Gaia-astrometric, kinematically old field star whose 2010 AllWISE colour was
photospheric and whose optical light curve stays flat.  It is the time-reverse
of EMBER (an excess that switched off).

The natural background differs in *time signature*, not amplitude: born
excesses on mature stars (giant impacts, comet swarms) rise in < 1 yr and then
decay or vary stochastically; obscured-AGN turn-ons (NGC 6447) rise for a
decade but have no parallax; YSOs, Miras, R CrB stars and novae all announce
themselves in the optical.  Every one of those is a named veto here.

Modules
-------
``sample``   Gaia DR3 x AllWISE parent sample (runner) + the pure selection
``acquire``  NEOWISE single-exposure photometry, PM-propagated (runner) + the
             pure frame cleaning and epoch binning
``rise``     the detector: sustained-rise statistics, scan-direction sinusoid,
             ramp-vs-impact shape test, the two-band rule --- pure
``vet``      the contaminant ladder --- pure, with an optical-series hook
``run``      stage orchestration, ``results/ignition/``
"""

from .rise import BandRise, StarVerdict, assess_band, assess_star

__all__ = ["BandRise", "StarVerdict", "assess_band", "assess_star"]
