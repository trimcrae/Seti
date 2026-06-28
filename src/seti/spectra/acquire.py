"""SPARCL retrieval of public survey spectra (DESI / SDSS / BOSS), runner-side.

SPARCL (the NOIRLab Astro Data Lab SPectra Analysis and Retrievable Catalog) serves
~31 million 1-D spectra through one Python client.  GitHub-hosted runners have the
unrestricted egress this needs (the interactive sandbox does not), so retrieval
runs there, exactly like the white-dwarf catalogue acquisition.  This module is a
thin, defensive wrapper: it discovers a sample with ``find`` and pulls flux /
wavelength / inverse-variance / redshift with ``retrieve`` in bounded chunks.
"""

from __future__ import annotations

import numpy as np

# Nominal resolving power per data release (arm-averaged; the LSF is refined
# per-spectrum from the dispersion at search time).
NOMINAL_RESOLUTION = {
    "DESI-EDR": 3000.0,
    "SDSS-DR16": 2000.0,
    "BOSS-DR16": 2000.0,
}


def _records(obj):
    """SPARCL result objects expose ``.records``; fall back to the object itself."""
    return getattr(obj, "records", obj)


def fetch_spectra(
    n: int = 2000,
    dataset: str = "DESI-EDR",
    spectype: str | None = None,
    chunk: int = 500,
    client=None,
) -> list[dict]:
    """Retrieve up to ``n`` spectra from ``dataset`` as plain dicts.

    Each dict carries ``spec_id, wave, flux, ivar, redshift, resolution, meta``
    --- the schema :func:`seti.spectra.vet.search_spectra` consumes.  ``client``
    may be injected for testing; otherwise a real :class:`SparclClient` is built.
    """
    if client is None:
        from sparcl.client import SparclClient  # lazy: only on the runner
        client = SparclClient()

    constraints: dict = {"data_release": [dataset]}
    if spectype:
        constraints["spectype"] = [spectype]
    find_fields = ["sparcl_id", "ra", "dec", "redshift", "spectype", "data_release"]
    found = client.find(outfields=find_fields, constraints=constraints, limit=n)
    ids = [getattr(r, "sparcl_id", None) or getattr(r, "_dr", None)
           for r in _records(found)]
    ids = [i for i in ids if i]
    print(f"[spectra] SPARCL find: {len(ids)} ids from {dataset}"
          f"{'/' + spectype if spectype else ''}")

    inc = ["sparcl_id", "ra", "dec", "redshift", "spectype", "data_release",
           "wavelength", "flux", "ivar"]
    res_default = NOMINAL_RESOLUTION.get(dataset, 2000.0)
    out: list[dict] = []
    for start in range(0, len(ids), chunk):
        sub = ids[start:start + chunk]
        try:
            got = client.retrieve(uuid_list=sub, include=inc)
        except Exception as exc:
            print(f"[spectra] retrieve chunk {start} failed: {exc!r}")
            continue
        for r in _records(got):
            wave = np.asarray(getattr(r, "wavelength", []), dtype=float)
            flux = np.asarray(getattr(r, "flux", []), dtype=float)
            ivar = np.asarray(getattr(r, "ivar", []), dtype=float)
            if wave.size < 100 or flux.size != wave.size or ivar.size != wave.size:
                continue
            sid = getattr(r, "sparcl_id", None) or str(len(out))
            out.append({
                "spec_id": str(sid),
                "wave": wave, "flux": flux, "ivar": ivar,
                "redshift": float(getattr(r, "redshift", 0.0) or 0.0),
                "resolution": res_default,
                "meta": {
                    "ra": float(getattr(r, "ra", np.nan) or np.nan),
                    "dec": float(getattr(r, "dec", np.nan) or np.nan),
                    "spectype": str(getattr(r, "spectype", "")),
                    "data_release": str(getattr(r, "data_release", dataset)),
                },
            })
    print(f"[spectra] retrieved {len(out)} usable spectra")
    return out


__all__ = ["fetch_spectra", "NOMINAL_RESOLUTION"]
