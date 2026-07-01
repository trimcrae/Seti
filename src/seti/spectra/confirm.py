"""Cross-instrument confirmation of laser-line candidates.

The decisive test a single coadded spectrum cannot provide: does the narrow line
appear in an *independent* spectrograph?  SPARCL serves DESI, SDSS and BOSS, so a
star observed by more than one survey can be checked directly.  A real persistent
emission line (a beacon, or a genuine astrophysical line) reproduces at the same
observed wavelength in the second instrument; a DESI-only cosmic ray, bad-pixel
or sky-subtraction residual does not.  This is the spectral analog of the dimming
search's multi-band achromaticity cut.

Everything runs runner-side (SPARCL egress).  For each candidate we query SPARCL
for any spectrum within a small cone in the *other* data releases, retrieve it,
and measure the flux excess at the candidate's observed wavelength.
"""

from __future__ import annotations

import time

import numpy as np

from .acquire import _records, _rget


def _make_client():
    """A SPARCL client with a sane read timeout (the default ~1.1 s times out
    almost every `find`).  Falls back to the bare client if the kwarg is rejected."""
    from sparcl.client import SparclClient
    for kw in ({"connect_timeout": 30.0}, {}):
        try:
            return SparclClient(**kw)
        except Exception:
            continue
    return SparclClient()


def _find_with_retry(fn, *, tries: int = 4, base_delay: float = 2.0):
    """Call a SPARCL query, retrying transient ReadTimeout / AccessNotAllowed
    (rate-limit) errors with exponential backoff."""
    last = None
    for k in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- SPARCL raises varied error types
            last = exc
            time.sleep(base_delay * (2 ** k))
    if last is not None:
        raise last
    return None

# Survey datasets to search for an independent observation, with their nominal
# resolution (used only for the local excess window).
_OTHER_DATASETS = ["DESI-DR1", "SDSS-DR17", "BOSS-DR17", "DESI-EDR",
                   "SDSS-DR16", "BOSS-DR16"]


def _find_overlap(client, ra: float, dec: float, exclude_release: str,
                  tol_arcsec: float = 2.0, exclude_id: str | None = None) -> list:
    """SPARCL ids of *independent* spectra within ``tol_arcsec`` of (ra, dec).

    Independence comes either from a different survey OR from a different
    observation of the same star in the *same* release: SPARCL labels SDSS-legacy
    and BOSS repeats under one release, and DESI/SDSS re-observe fields, so a
    second sparcl_id at the same sky position is a genuine independent exposure.
    We therefore search *all* datasets and drop only the candidate's own
    ``exclude_id`` (never the whole release), which is what makes repeat-visit
    confirmation possible.
    """
    d = tol_arcsec / 3600.0
    cosd = max(np.cos(np.radians(dec)), 1e-3)
    constraints = {
        "ra": [ra - d / cosd, ra + d / cosd],
        "dec": [dec - d, dec + d],
        "data_release": list(_OTHER_DATASETS),
    }
    fields = ["sparcl_id", "ra", "dec", "data_release"]
    try:
        found = _find_with_retry(
            lambda: client.find(outfields=fields, constraints=constraints, limit=20))
    except Exception as exc:
        print(f"[confirm] find failed at ({ra:.4f},{dec:.4f}): {exc!r}")
        return []
    ids = list(getattr(found, "ids", []) or [])
    if not ids:
        ids = [_rget(r, "sparcl_id") or _rget(r, "id") for r in _records(found)]
    # Keep every spectrum except the candidate's own coadd.
    return [i for i in ids if i and str(i) != str(exclude_id)]


def line_excess(wave: np.ndarray, flux: np.ndarray, ivar: np.ndarray,
                obs_wave: float, win: float = 4.0, cont_win: float = 60.0) -> dict:
    """Significance of an emission excess at ``obs_wave`` in a spectrum.

    Continuum is the median outside +/-``win`` but within +/-``cont_win`` of the
    line; the excess is the peak-minus-continuum in that window divided by the
    local noise.  Returns the significance and whether it clears 4 sigma.
    """
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    ivar = np.asarray(ivar, float)
    near = np.abs(wave - obs_wave) <= cont_win
    if near.sum() < 10:
        return {"sigma": float("nan"), "present": False, "n_pix": int(near.sum())}
    w, f = wave[near], flux[near]
    line = np.abs(w - obs_wave) <= win
    cont_mask = ~line
    if line.sum() < 1 or cont_mask.sum() < 5:
        return {"sigma": float("nan"), "present": False, "n_pix": int(near.sum())}
    cont = float(np.nanmedian(f[cont_mask]))
    noise = float(np.nanstd(f[cont_mask])) or 1e-9
    peak = float(np.nanmax(f[line]))
    sigma = (peak - cont) / noise
    return {"sigma": float(sigma), "present": bool(sigma >= 4.0),
            "cont": cont, "peak": peak, "n_pix": int(near.sum())}


def cross_confirm(candidates: list[dict], client=None, max_candidates: int = 40) -> list[dict]:
    """For each candidate, look for an independent spectrum and test the line.

    ``candidates`` are dicts with ``spec_id``, ``ra``, ``dec``, ``wavelength`` and
    ``data_release``.  Returns the input augmented with ``n_overlap`` (independent
    spectra found), ``confirm_sigma`` (best line significance in another survey),
    and ``cross_confirmed``.
    """
    if client is None:
        client = _make_client()
    out: list[dict] = []
    for idx, c in enumerate(candidates[:max_candidates]):
        if idx:
            time.sleep(0.5)   # throttle so rapid finds don't trip rate limits
        ra, dec = c.get("ra"), c.get("dec")
        obs = c.get("wavelength")
        rel = str(c.get("data_release", "DESI-DR1"))
        rec = dict(c)
        rec.update({"n_overlap": 0, "confirm_sigma": float("nan"),
                    "cross_confirmed": False})
        if not (np.isfinite(ra) and np.isfinite(dec) and np.isfinite(obs)):
            out.append(rec)
            continue
        ids = _find_overlap(client, float(ra), float(dec), rel,
                            exclude_id=c.get("spec_id"))
        rec["n_overlap"] = len(ids)
        if not ids:
            out.append(rec)
            continue
        inc = ["sparcl_id", "wavelength", "flux", "ivar", "data_release"]
        try:
            got = _find_with_retry(
                lambda ids=ids, inc=inc: client.retrieve(uuid_list=ids, include=inc))
        except TypeError:
            got = _find_with_retry(
                lambda ids=ids, inc=inc: client.retrieve(ids, include=inc))
        except Exception as exc:
            print(f"[confirm] retrieve failed for {str(c.get('spec_id'))[:8]}: {exc!r}")
            out.append(rec)
            continue
        best = float("nan")
        for r in _records(got):
            wave = np.asarray(_rget(r, "wavelength", []), float)
            flux = np.asarray(_rget(r, "flux", []), float)
            iv = np.asarray(_rget(r, "ivar", []), float)
            if wave.size < 50 or flux.size != wave.size:
                continue
            ex = line_excess(wave, flux, iv, float(obs))
            s = ex.get("sigma", float("nan"))
            if np.isfinite(s) and (not np.isfinite(best) or s > best):
                best = s
        rec["confirm_sigma"] = best
        rec["cross_confirmed"] = bool(np.isfinite(best) and best >= 4.0)
        out.append(rec)
        print(f"[confirm] {str(c.get('spec_id'))[:8]} lam={obs:.1f} "
              f"overlap={len(ids)} best_sigma={best:.1f} "
              f"confirmed={rec['cross_confirmed']}")
    return out


__all__ = ["cross_confirm", "line_excess"]
