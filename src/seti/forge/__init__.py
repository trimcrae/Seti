"""FORGE (S47): hot exozodis read as ~1500 K swarm candidates.

Interferometry finds a ~1 % H/K-band excess around 10-30 % of nearby
main-sequence stars.  The standard reading is sub-micron ("nano") grains,
whose emissivity collapses beyond a few microns, so the excess is K-bright and
N-faint.  A grey Planck body at ~1500 K (the cheapest place to collect
stellar energy per unit collector mass) cannot do that: at 1 % in K it gives
~8 % at 10 um.  The observable is therefore the OUTLIER whose N-band excess IS
consistent with the Planck extrapolation of its K excess.

Modules
-------
``physics``   Planck / nano-grain excess models, the per-star likelihood
              ratio, tiers, the variability term and the broadband
              (K-W1, W1-W2, W2-W3) shift a hot component produces.
``tables``    the embedded published tables (``data_assets/forge_*.csv``),
              identifier canonicalisation and the per-star measurement set.
``acquire``   runner-only: VizieR discovery and fetch of every published
              excess table, archive verification of the embedded rows, star
              positions, Gaia / WDS companion context, the AllWISE / 2MASS
              broadband leg for the sample and for the < 30 pc population.
``run``       stage orchestration (probe / acquire / screen / assess / all);
              writes ``results/forge/``.
"""
