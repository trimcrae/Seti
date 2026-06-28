"""Synthetic white-dwarf sample generator for offline tests and CI.

Produces a labelled table exercising every branch of the funnel:

  * ``clean``         -- bare photospheres, no excess (the bulk);
  * ``known_disk``    -- warm-dust debris-disk excess inside the locus;
  * ``anomaly``       -- cool / swarm-like excess OUTSIDE the dust locus;
  * ``blend``         -- excess from a crowded WISE neighbour (crowding cut);
  * ``background``    -- non-co-moving IR source (co-movement cut);
  * ``agn``           -- very red W1-W2 extragalactic interloper.

The generator is deterministic given ``seed`` so the committed sample and the
tests are reproducible.  It is NOT a science input -- it only validates that the
pipeline behaves as designed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .photometry import (
    band_freq_hz,
    flux_jy_to_mag,
    planck_bnu,
)

ANCHOR = ("J", "H", "Ks")
PRED = ("W1", "W2")
GAIA_EPOCH = 2016.0
WISE_EPOCH = 2015.4


def _photosphere_mags(teff, scale, bands):
    """Vega mags of a blackbody photosphere of given Teff and solid-angle scale."""
    out = {}
    for b in bands:
        f_jy = scale * np.pi * planck_bnu(teff, band_freq_hz(b)) * 1e26
        out[b] = float(flux_jy_to_mag(f_jy, b))
    return out


def _add_excess(teff, scale, t_dust, tau):
    """Return W1/W2 mags of photosphere + blackbody excess of (T_dust, tau).

    The excess is normalised to a *bolometric* fractional luminosity ``tau``:
    a dust/structure solid angle ``Omega_dust = tau * scale * (T_WD/T_dust)^4``
    emitting as a blackbody at ``T_dust`` (consistent with ``sed.excess.fit_dust``).
    """
    out = {}
    omega_dust = tau * scale * (teff / t_dust) ** 4
    for b in PRED:
        nu = band_freq_hz(b)
        phot_jy = scale * np.pi * planck_bnu(teff, nu) * 1e26
        excess_jy = omega_dust * np.pi * planck_bnu(t_dust, nu) * 1e26
        out[b] = float(flux_jy_to_mag(phot_jy + excess_jy, b))
    return out


def make_sample(n_clean: int = 300, seed: int = 7) -> pd.DataFrame:
    """Generate the labelled synthetic WD sample."""
    rng = np.random.default_rng(seed)
    rows = []

    def base_row(teff, kind):
        # Nearby WD: large parallax, healthy proper motion.
        dist_pc = rng.uniform(20, 120)
        parallax = 1000.0 / dist_pc
        scale = (0.0125 * 6.957e8 / (dist_pc * 3.086e16)) ** 2  # (R_WD/d)^2, R~0.0125 Rsun
        anchors = _photosphere_mags(teff, scale, ANCHOR)
        wmags = _photosphere_mags(teff, scale, PRED)
        ra = rng.uniform(0, 360)
        dec = rng.uniform(-40, 80)
        pmra = rng.uniform(-200, 200)
        pmdec = rng.uniform(-200, 200)
        dt = WISE_EPOCH - GAIA_EPOCH
        cosd = np.cos(np.radians(dec))
        # Co-moving WISE position: Gaia propagated to WISE epoch (+ tiny noise).
        ra_wise = ra + (pmra * dt / 3.6e6) / max(cosd, 1e-3) + rng.normal(0, 5e-5)
        dec_wise = dec + (pmdec * dt / 3.6e6) + rng.normal(0, 5e-5)
        row = dict(
            source_id=rng.integers(1_000_000_000, 9_000_000_000),
            ra=ra, dec=dec, pmra=pmra, pmdec=pmdec,
            pmra_error=1.0, pmdec_error=1.0,
            parallax=parallax, parallax_over_error=parallax / 0.05,
            ruwe=rng.uniform(0.9, 1.2), astrometric_excess_noise=rng.uniform(0, 0.3),
            teff=teff, logg=8.0, pwd=rng.uniform(0.95, 1.0),
            ra_wise=ra_wise, dec_wise=dec_wise,
            pmra_wise=pmra + rng.normal(0, 15), pmdec_wise=pmdec + rng.normal(0, 15),
            e_pmra_wise=15.0, e_pmdec_wise=15.0,
            cc_flags="0000", ph_qual="AA", ext_flg=0,
            n_wise_neighbours=1, gaia_nn_arcsec=rng.uniform(8, 30),
            W1mag_unwise=np.nan,
            # known_disk stands in for a positive match to a published WD
            # debris-disk control catalogue (set during acquisition for science
            # runs; hard-coded here so the offline sample exercises subtraction).
            known_disk=(kind == "known_disk"),
            label=kind,
        )
        for b in ANCHOR:
            row[f"{b}mag"] = anchors[b] + rng.normal(0, 0.02)
            row[f"e_{b}mag"] = 0.03
        for b in PRED:
            row[f"{b}mag"] = wmags[b] + rng.normal(0, 0.03)
            row[f"e_{b}mag"] = 0.04
        row["w2snr"] = 1.0 / (0.4 * np.log(10.0) * row["e_W2mag"])
        return row, scale

    # Clean photospheres.
    for _ in range(n_clean):
        teff = rng.uniform(5000, 25000)
        row, _ = base_row(teff, "clean")
        rows.append(row)

    # Known debris disks: warm dust inside the locus.
    for _ in range(25):
        teff = rng.uniform(6000, 15000)
        row, scale = base_row(teff, "known_disk")
        t_dust = rng.uniform(1000, 1500)
        tau = 10 ** rng.uniform(-2.3, -1.3)
        new = _add_excess(teff, scale, t_dust, tau)
        for b in PRED:
            row[f"{b}mag"] = new[b]
        row["w2snr"] = 1.0 / (0.4 * np.log(10.0) * row["e_W2mag"])
        rows.append(row)

    # Anomalies: cool / swarm-like excess OUTSIDE the dust locus.
    for _ in range(6):
        teff = rng.uniform(6000, 20000)
        row, scale = base_row(teff, "anomaly")
        t_dust = rng.uniform(150, 400)            # too cool for sublimation dust
        tau = 10 ** rng.uniform(-1.0, -0.4)       # swarm-like covering fraction
        new = _add_excess(teff, scale, t_dust, tau)
        for b in PRED:
            row[f"{b}mag"] = new[b]
        row["w2snr"] = 1.0 / (0.4 * np.log(10.0) * row["e_W2mag"])
        rows.append(row)

    # Blends: strong excess but a crowded WISE neighbour + unWISE disagreement.
    for _ in range(8):
        teff = rng.uniform(6000, 15000)
        row, scale = base_row(teff, "blend")
        new = _add_excess(teff, scale, 800, 0.05)
        for b in PRED:
            row[f"{b}mag"] = new[b]
        row["n_wise_neighbours"] = 3
        row["gaia_nn_arcsec"] = 2.0
        row["W1mag_unwise"] = row["W1mag"] + 0.8   # disagrees with CatWISE
        row["w2snr"] = 1.0 / (0.4 * np.log(10.0) * row["e_W2mag"])
        rows.append(row)

    # Background: real-looking excess but the IR source is NOT co-moving.
    for _ in range(8):
        teff = rng.uniform(6000, 15000)
        row, scale = base_row(teff, "background")
        new = _add_excess(teff, scale, 600, 0.08)
        for b in PRED:
            row[f"{b}mag"] = new[b]
        # Put the WISE source at the *Gaia* epoch position (static -> fails test).
        row["ra_wise"] = row["ra"] + rng.normal(0, 5e-5)
        row["dec_wise"] = row["dec"] + rng.normal(0, 5e-5)
        row["pmra_wise"] = rng.normal(0, 5)
        row["pmdec_wise"] = rng.normal(0, 5)
        row["w2snr"] = 1.0 / (0.4 * np.log(10.0) * row["e_W2mag"])
        rows.append(row)

    # AGN: extragalactic, very red W1-W2.
    for _ in range(6):
        teff = rng.uniform(8000, 15000)
        row, scale = base_row(teff, "agn")
        row["W2mag"] = row["W1mag"] - 1.5          # W1-W2 ~ 1.5 (AGN wedge)
        row["w2snr"] = 1.0 / (0.4 * np.log(10.0) * row["e_W2mag"])
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


__all__ = ["make_sample"]
