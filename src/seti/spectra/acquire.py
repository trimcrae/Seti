"""SPARCL retrieval of public survey spectra (DESI / SDSS / BOSS), runner-side.

SPARCL (the NOIRLab Astro Data Lab SPectra Analysis and Retrievable Catalog) serves
~31 million 1-D spectra through one Python client.  GitHub-hosted runners have the
unrestricted egress this needs (the interactive sandbox does not), so retrieval
runs there, exactly like the white-dwarf catalogue acquisition.

SPARCL records are dict-like (they expose ``.keys()``); the discovery id field is
``id`` and the vector fields are ``wavelength``/``flux``/``ivar``.  Access is kept
defensive (dict item, ``.get`` or attribute) and self-diagnosing (the available
datasets and the first record's keys are logged) so the exact API surface is
visible in the CI logs if a release name or field differs.
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
    return getattr(obj, "records", obj)


def _rget(rec, key, default=None):
    """Read ``key`` from a SPARCL record by item, ``.get`` or attribute."""
    try:
        return rec[key]
    except (TypeError, KeyError, IndexError):
        pass
    if hasattr(rec, "get"):
        try:
            v = rec.get(key, None)
            if v is not None:
                return v
        except Exception:
            pass
    return getattr(rec, key, default)


def fetch_spectra(
    n: int = 2000,
    dataset: str = "DESI-EDR",
    spectype: str | None = None,
    chunk: int = 500,
    client=None,
) -> list[dict]:
    """Retrieve up to ``n`` spectra from ``dataset`` as the search-ready dict schema."""
    if client is None:
        from sparcl.client import SparclClient  # lazy: only on the runner
        client = SparclClient()

    # Diagnostics: surface the valid dataset labels so a wrong release is obvious.
    try:
        print(f"[spectra] SPARCL datasets: {getattr(client, 'all_datasets', '?')}")
    except Exception as exc:
        print(f"[spectra] could not list datasets: {exc!r}")

    constraints: dict = {"data_release": [dataset]}
    if spectype:
        constraints["spectype"] = [spectype]
    find_fields = ["id", "ra", "dec", "redshift", "spectype", "data_release"]
    found = client.find(outfields=find_fields, constraints=constraints, limit=n)
    recs = list(_records(found))
    if recs:
        try:
            print(f"[spectra] find record keys: {sorted(list(recs[0].keys()))}")
        except Exception:
            pass
    ids = [_rget(r, "id") for r in recs]
    ids = [i for i in ids if i]
    print(f"[spectra] SPARCL find: {len(ids)} ids from {dataset}"
          f"{'/' + spectype if spectype else ''}")
    if not ids:
        return []

    inc = ["id", "ra", "dec", "redshift", "spectype", "data_release",
           "wavelength", "flux", "ivar"]
    res_default = NOMINAL_RESOLUTION.get(dataset, 2000.0)
    out: list[dict] = []
    for start in range(0, len(ids), chunk):
        sub = ids[start:start + chunk]
        try:
            got = client.retrieve(uuid_list=sub, include=inc)
        except TypeError:
            got = client.retrieve(sub, include=inc)   # positional fallback
        except Exception as exc:
            print(f"[spectra] retrieve chunk {start} failed: {exc!r}")
            continue
        grecs = list(_records(got))
        if start == 0 and grecs:
            try:
                print(f"[spectra] retrieve record keys: {sorted(list(grecs[0].keys()))}")
            except Exception:
                pass
        for r in grecs:
            wave = np.asarray(_rget(r, "wavelength", []), dtype=float)
            flux = np.asarray(_rget(r, "flux", []), dtype=float)
            ivar = np.asarray(_rget(r, "ivar", []), dtype=float)
            if wave.size < 100 or flux.size != wave.size or ivar.size != wave.size:
                continue
            sid = _rget(r, "id") or str(len(out))
            out.append({
                "spec_id": str(sid),
                "wave": wave, "flux": flux, "ivar": ivar,
                "redshift": float(_rget(r, "redshift", 0.0) or 0.0),
                "resolution": res_default,
                "meta": {
                    "ra": float(_rget(r, "ra", np.nan) or np.nan),
                    "dec": float(_rget(r, "dec", np.nan) or np.nan),
                    "spectype": str(_rget(r, "spectype", "")),
                    "data_release": str(_rget(r, "data_release", dataset)),
                },
            })
    print(f"[spectra] retrieved {len(out)} usable spectra")
    return out


__all__ = ["fetch_spectra", "NOMINAL_RESOLUTION"]
