"""SPARK — a single-spectral-element excess on a stellar point source.

S48 (``docs/necrofrontier.md`` XVIII): SPHEREx QR2 spectral images, 0.75–5 µm
at R ≈ 35–130 — an unresolved line adds flux to ONE spectral element of a
star's 102-channel spectrum, repeating across independent sky passes and
detector positions.  S49: the Euclid Q1 NISP red-grism line-feature table
(1.21–1.89 µm, R ≈ 450) joined to Gaia DR3 stars, every stellar and every
redshifted galaxy line vetoed, the feature required in ≥ 2 dithers.

Modules
-------
``lines``    the line lists (stellar, nebular / galaxy, industrial bands),
             air→vacuum, the stellar-line veto, the redshift-pattern veto
``euclid``   S49: schema discovery, the chunked IRSA TAP pull, the pure
             screen over the joined feature table
``gaia``     Gaia DR3 seeds and the KD-tree join (IRSA's copy first, ESA next)
``spherex``  S48: obscore discovery, cutouts, the wavelength map, forced
             aperture photometry, the single-channel excess detector
``run``      stage orchestration; writes ``results/spark/``
"""
