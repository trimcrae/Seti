# PARALLAX4 test fixtures (real Gaia data, unmodified)

`gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip`
: The official ESA Gaia DR4 epoch-astrometry prerelease (12 sources, 1008 FoV
  transits, 37 columns, `release = "Gaia DR4_RC3"`).
  URL `https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_DR4/dr4-prerelease/gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip`,
  625,651 bytes, sha256 `07f0e8d9ac97a29ea376a0c7242de3124d2a08ad72aba0958d6575d94d35fa0b`
  (asserted by `tests/test_parallax4.py::test_prerelease_fixture_is_the_esa_file`).
  Obtained from the public redistribution in `vasilybelokurov/gaia-dr4-explorer`
  (tests/fixtures/prerelease.zip), whose sha256 matches the ESA file.

`dr3_epoch_photometry_1457486023639239296.vot`
: Gaia DR3 DataLink `EPOCH_PHOTOMETRY` product for Gaia-4
  (`source_id = 1457486023639239296`), retrieved unmodified 2026-09-22 (same
  redistribution). Used to test the long-form reader and the DR3→DR4
  transit_id join on real data.

Gaia data are © ESA/Gaia/DPAC, used under the Gaia data licence
(<https://www.cosmos.esa.int/web/gaia-users/license>); see the Gaia citation
instructions
(<https://gea.esac.esa.int/archive/documentation/GDR3/Miscellaneous/sec_credit_and_citation_instructions/>).
