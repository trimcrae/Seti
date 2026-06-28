"""Generate the committed *synthetic stand-in* for the Montreal/Bergeron WD
photometry table.

The real science table is the Montreal/Bergeron synthetic photometry
(https://www.astro.umontreal.ca/~bergeron/CoolingModels/), which provides
absolute magnitudes in Gaia/2MASS/WISE bands as a function of (Teff, logg).  It
is not redistributed here; this script builds a small blackbody-based stand-in
of the same schema so the ``BergeronModel`` code path is exercised offline.
Replace the asset with the real table for science runs (same columns).

Run:  python scripts/make_bergeron_asset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seti.photometry import band_freq_hz, flux_jy_to_mag, planck_bnu  # noqa: E402

BANDS = ("G", "BP", "RP", "J", "H", "Ks", "W1", "W2")


def main():
    teffs = np.arange(4000, 40001, 1000.0)
    loggs = np.arange(7.0, 9.51, 0.5)
    # Fixed reference solid angle (10 pc, R=0.0125 Rsun) -> absolute-ish mags.
    scale = (0.0125 * 6.957e8 / (10 * 3.086e16)) ** 2

    rows = []
    for teff in teffs:
        for logg in loggs:
            row = {"teff": float(teff), "logg": float(logg)}
            for b in BANDS:
                f_jy = scale * np.pi * planck_bnu(teff, band_freq_hz(b)) * 1e26
                row[f"M_{b}"] = float(flux_jy_to_mag(f_jy, b))
            rows.append(row)

    tbl = Table(rows)
    tbl.meta["origin"] = "SYNTHETIC blackbody stand-in (NOT the Montreal table)"
    tbl.meta["replace_with"] = "Montreal/Bergeron synthetic photometry"
    out = Path(__file__).resolve().parents[1] / "src/seti/data_assets/bergeron_synthetic_wise.ecsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tbl.write(out, format="ascii.ecsv", overwrite=True)
    print(f"wrote {len(tbl)} rows -> {out}")


if __name__ == "__main__":
    main()
