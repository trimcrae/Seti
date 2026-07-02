"""Known-planet-host cross-match for the directed-travel destination ranking.

Destination quality from Gaia photometry is only a *host-star* prior.  The sharper
signal is: does a reachable neighbour already host a planet -- better, a temperate
(habitable-zone) one?  Those are the stars a K2-18 civilisation would actually aim
for.  This module pulls confirmed planets in the local volume from the NASA
Exoplanet Archive (runner-side; TAP) and cross-matches them onto the neighbour
list, by Gaia source_id where the archive provides it and by sky position
otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# NASA Exoplanet Archive TAP, Planetary Systems ("ps") table.
_EXO_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
_PS_QUERY = (
    "SELECT hostname, gaia_id, ra, dec, sy_dist, pl_name, pl_rade, pl_bmasse, "
    "pl_orbper, pl_eqt, pl_insol, st_teff "
    "FROM ps WHERE default_flag=1 AND sy_dist IS NOT NULL AND sy_dist < {max_pc}"
)

# Classical (rocky, Earth-analog) habitable zone: insolation in Earth units and,
# as a fallback, equilibrium temperature (K).
_HZ_INSOL = (0.20, 1.75)
_HZ_EQT = (180.0, 320.0)

# HYCEAN habitable zone (Madhusudhan et al. 2021).  The relevant target for a
# K2-18-evolved organism is not an Earth-analog but another hycean world: a
# sub-Neptune (~1-2.6 R_earth, K2-18 b itself is 2.6) with an H2 envelope over a
# liquid-water ocean.  The H2 greenhouse widens the temperate range enormously --
# "hot hycean" worlds stay habitable at high instellation and "cold hycean" worlds
# far out -- so the insolation window is broad, and cool (K/M) hosts are favoured.
_HYCEAN_RADE = (1.5, 2.6)          # R_earth: sub-Neptune band (above the radius valley)
_HYCEAN_INSOL = (0.01, 10.0)       # S_earth: cold hycean ... hot hycean
_HYCEAN_EQT = (150.0, 510.0)       # K, fallback when insolation is missing


def fetch_nearby_planets(max_pc: float = 80.0) -> pd.DataFrame:
    """Confirmed planets with a host distance under ``max_pc`` (runner-side)."""
    import io

    import requests
    q = _PS_QUERY.format(max_pc=float(max_pc))
    r = requests.get(_EXO_TAP, params={"query": q, "format": "csv"}, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return df.rename(columns={c: c.lower() for c in df.columns})


def _gaia_id_int(series: pd.Series) -> pd.Series:
    """Parse the trailing integer out of 'Gaia DR3 12345...' style ids."""
    return (series.astype(str).str.extract(r"(\d{5,})", expand=False)
            .astype("Float64").astype("Int64"))


def _is_temperate(planets: pd.DataFrame) -> np.ndarray:
    """Classical rocky-world (Earth-analog) habitable zone."""
    insol = pd.to_numeric(planets.get("pl_insol"), errors="coerce")
    eqt = pd.to_numeric(planets.get("pl_eqt"), errors="coerce")
    by_insol = insol.between(*_HZ_INSOL)
    by_eqt = eqt.between(*_HZ_EQT)
    # Prefer insolation; fall back to equilibrium temperature where insol is absent.
    return (by_insol | (insol.isna() & by_eqt)).to_numpy(bool)


def _is_hycean_candidate(planets: pd.DataFrame) -> np.ndarray:
    """Hycean-analog: a sub-Neptune in the (wide) hycean habitable zone -- the
    destination class a K2-18-evolved organism would actually find habitable."""
    rade = pd.to_numeric(planets.get("pl_rade"), errors="coerce")
    insol = pd.to_numeric(planets.get("pl_insol"), errors="coerce")
    eqt = pd.to_numeric(planets.get("pl_eqt"), errors="coerce")
    right_size = rade.between(*_HYCEAN_RADE)
    by_insol = insol.between(*_HYCEAN_INSOL)
    by_eqt = eqt.between(*_HYCEAN_EQT)
    in_hz = by_insol | (insol.isna() & by_eqt)
    # If neither insolation nor eqt is known, size alone still makes it a candidate
    # sub-Neptune (the archive often lacks flux for cool-host planets).
    no_flux = insol.isna() & eqt.isna()
    return (right_size & (in_hz | no_flux)).to_numpy(bool)


def crossmatch_hosts(neighbors: pd.DataFrame, planets: pd.DataFrame,
                     radius_arcsec: float = 3.0) -> pd.DataFrame:
    """Flag which neighbours are known planet hosts and whether any is temperate.

    Adds ``known_planet_host``, ``n_planets``, ``has_temperate_planet`` (classical
    HZ), ``has_hycean_candidate`` (the hycean-analog target class) and the host
    name to ``neighbors``.  Matches on Gaia source_id first (exact), then on sky
    position within ``radius_arcsec`` for archive rows lacking a Gaia id.
    """
    out = neighbors.copy()
    out["known_planet_host"] = False
    out["n_planets"] = 0
    out["has_temperate_planet"] = False
    out["has_hycean_candidate"] = False
    out["host_name"] = ""
    if not len(planets):
        return out

    planets = planets.copy()
    planets["_temperate"] = _is_temperate(planets)
    planets["_hycean"] = _is_hycean_candidate(planets)
    planets["_gaia_int"] = _gaia_id_int(planets.get("gaia_id", pd.Series(dtype=str)))

    nb_ids = pd.to_numeric(out.get("source_id"), errors="coerce").astype("Int64")
    nb_ra = pd.to_numeric(out.get("ra"), errors="coerce").to_numpy(float)
    nb_dec = pd.to_numeric(out.get("dec"), errors="coerce").to_numpy(float)
    r_deg = radius_arcsec / 3600.0

    for i in range(len(out)):
        sid = nb_ids.iloc[i]
        hit = planets[planets["_gaia_int"] == sid] if pd.notna(sid) else planets.iloc[:0]
        if not len(hit) and np.isfinite(nb_ra[i]):
            cosd = np.cos(np.radians(nb_dec[i]))
            dra = (pd.to_numeric(planets["ra"], errors="coerce").to_numpy(float)
                   - nb_ra[i]) * cosd
            ddec = pd.to_numeric(planets["dec"], errors="coerce").to_numpy(float) - nb_dec[i]
            hit = planets[np.hypot(dra, ddec) <= r_deg]
        if len(hit):
            out.iat[i, out.columns.get_loc("known_planet_host")] = True
            out.iat[i, out.columns.get_loc("n_planets")] = int(len(hit))
            out.iat[i, out.columns.get_loc("has_temperate_planet")] = bool(
                hit["_temperate"].any())
            out.iat[i, out.columns.get_loc("has_hycean_candidate")] = bool(
                hit["_hycean"].any())
            out.iat[i, out.columns.get_loc("host_name")] = str(hit["hostname"].iloc[0])
    return out


__all__ = ["fetch_nearby_planets", "crossmatch_hosts"]
