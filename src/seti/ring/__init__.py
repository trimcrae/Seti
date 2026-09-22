"""RING (S63) -- rings around the dead: post-biological hosts by construction.

A technosignature at a host where biology is impossible has no biological
alternative.  The Osmanov (2016, 2018) ring-temperature reading -- a Dyson
ring at 300-700 K, a W1/W2 excess -- over four host classes:

* white dwarfs (Gaia EDR3 x AllWISE, the WD IR-excess acquisition with a
  different temperature prior),
* neutron stars / pulsars (ATNF x AllWISE / CatWISE2020, with offset-position
  controls), including the two pulsar-planet systems as a named vet,
* Y / late-T brown dwarfs (NEOWISE per-epoch W2: a Wien peak that switches
  is a duty cycle),
* young free-floating planetary-mass objects (hotter than cooling allows).

See ``docs/ring.md``.
"""

from .run import run

__all__ = ["run"]
