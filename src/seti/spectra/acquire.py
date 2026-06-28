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
    "DESI-DR1": 3000.0,
    "DESI-EDR": 3000.0,
    "SDSS-DR17": 2000.0,
    "BOSS-DR17": 2000.0,
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


_RETRIEVE_FIELDS = ["sparcl_id", "ra", "dec", "redshift", "spectype",
                    "data_release", "wavelength", "flux", "ivar", "mask"]


def find_ids(n: int, dataset: str, spectype: str | None = None, client=None) -> list:
    """Discover up to ``n`` SPARCL ids for ``dataset`` (and optional ``spectype``)."""
    if client is None:
        from sparcl.client import SparclClient
        client = SparclClient()
    try:
        print(f"[spectra] SPARCL datasets: {getattr(client, 'all_datasets', '?')}")
    except Exception as exc:
        print(f"[spectra] could not list datasets: {exc!r}")
    constraints: dict = {"data_release": [dataset]}
    if spectype:
        constraints["spectype"] = [spectype]
    # SPARCL's spectrum identifier field is ``sparcl_id`` (``id`` is silently
    # dropped).  Sort for deterministic, reproducible samples.
    fields = ["sparcl_id", "ra", "dec", "redshift", "spectype", "data_release"]
    found = client.find(outfields=fields, constraints=constraints,
                        sort="sparcl_id", limit=n)
    recs = list(_records(found))
    if recs:
        try:
            print(f"[spectra] find record keys: {sorted(list(recs[0].keys()))}")
        except Exception:
            pass
    ids = list(getattr(found, "ids", []) or [])
    if not ids:
        ids = [_rget(r, "sparcl_id") or _rget(r, "id") for r in recs]
        ids = [i for i in ids if i]
    print(f"[spectra] SPARCL find: {len(ids)} ids from {dataset}"
          f"{'/' + spectype if spectype else ''}")
    return ids


def _retrieve_dicts(client, ids_chunk, dataset, res_default, first=False) -> list[dict]:
    try:
        got = client.retrieve(uuid_list=ids_chunk, include=_RETRIEVE_FIELDS)
    except TypeError:
        got = client.retrieve(ids_chunk, include=_RETRIEVE_FIELDS)
    grecs = list(_records(got))
    if first and grecs:
        try:
            print(f"[spectra] retrieve record keys: {sorted(list(grecs[0].keys()))}")
        except Exception:
            pass
    out: list[dict] = []
    for r in grecs:
        wave = np.asarray(_rget(r, "wavelength", []), dtype=float)
        flux = np.asarray(_rget(r, "flux", []), dtype=float)
        ivar = np.asarray(_rget(r, "ivar", []), dtype=float)
        if wave.size < 100 or flux.size != wave.size or ivar.size != wave.size:
            continue
        mask = np.asarray(_rget(r, "mask", []), dtype=float)
        if mask.size != wave.size:
            mask = np.zeros_like(wave)
        sid = _rget(r, "sparcl_id") or _rget(r, "id") or str(len(out))
        out.append({
            "spec_id": str(sid), "wave": wave, "flux": flux, "ivar": ivar,
            "mask": mask, "redshift": float(_rget(r, "redshift", 0.0) or 0.0),
            "resolution": res_default,
            "meta": {"ra": float(_rget(r, "ra", np.nan) or np.nan),
                     "dec": float(_rget(r, "dec", np.nan) or np.nan),
                     "spectype": str(_rget(r, "spectype", "")),
                     "data_release": str(_rget(r, "data_release", dataset))},
        })
    return out


def iter_spectra(n: int = 2000, dataset: str = "DESI-DR1", spectype: str | None = None,
                 chunk: int = 500, client=None):
    """Yield retrieved spectra one chunk-list at a time (bounded memory for scale)."""
    if client is None:
        from sparcl.client import SparclClient
        client = SparclClient()
    ids = find_ids(n, dataset, spectype, client=client)
    res_default = NOMINAL_RESOLUTION.get(dataset, 2000.0)
    for start in range(0, len(ids), chunk):
        try:
            yield _retrieve_dicts(client, ids[start:start + chunk], dataset,
                                  res_default, first=(start == 0))
        except Exception as exc:
            print(f"[spectra] retrieve chunk {start} failed: {exc!r}")


def fetch_spectra(n: int = 2000, dataset: str = "DESI-DR1", spectype: str | None = None,
                  chunk: int = 500, client=None) -> list[dict]:
    """Retrieve up to ``n`` spectra as a single list (small n / offline tests)."""
    out: list[dict] = []
    for batch in iter_spectra(n, dataset, spectype, chunk=chunk, client=client):
        out.extend(batch)
    print(f"[spectra] retrieved {len(out)} usable spectra")
    return out


__all__ = ["fetch_spectra", "iter_spectra", "find_ids", "NOMINAL_RESOLUTION"]
