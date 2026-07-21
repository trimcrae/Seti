"""Pure-logic technosignature-limit scorers (unit-tested offline).

The physics of a SETI sensitivity limit is a signal-to-noise problem with a small
number of measured ingredients, so it is exactly the kind of thing that can be
written as a transparent, benchmark-anchored scorer rather than an instrument
simulator.  This module turns telescope + observation metadata into the two
limits that matter for a targeted search, each expressed against a *standard
yardstick* so the number means something:

* :func:`eirp_limit` -- the minimum detectable Equivalent Isotropic Radiated
  Power (EIRP, W) of a narrowband radio beacon, from the radiometer equation
  ``sigma_S = SNR * SEFD / sqrt(npol * dnu * tau)`` and ``EIRP = 4 pi d^2 * S``.
  Reported relative to the **Arecibo planetary-radar beacon, ~2e13 W** -- the
  field's standard yardstick (Enriquez et al. 2017; Price et al. 2020) -- and, via
  :func:`beacon_capability`, as a fraction of a Kardashev-Type-I power budget.
* :func:`optical_seti_limit` -- the minimum detectable laser power (a photon-rate
  limit) for an optical SETI observation, with the classic yardstick that a
  diffraction-limited ~10 m transmitter driven by a ~MW-class laser radiates an
  EIRP of order 1e17 W that an existing telescope can see across the Galaxy
  (Howard et al. 2004; Wright et al. 2018).
* :func:`parse_observation_inventory` -- fold a list of archive observation
  records into a coverage table: which facilities/bands actually looked at the
  target and the best EIRP limit achieved per band.

Everything here is pure ``numpy``; nothing queries a network, nothing assumes a
detection, and every "no coverage" is reported as such.
"""

from __future__ import annotations

import numpy as np

# --- Physical constants and standard yardsticks ----------------------------
_PC_M = 3.0856775814913673e16      # metres per parsec
_JY = 1e-26                        # 1 Jansky in W / m^2 / Hz
_H = 6.62607015e-34                # Planck constant, J s
_C = 2.99792458e8                  # speed of light, m/s

# The Arecibo S-band planetary radar: EIRP ~ 2e13 W is the canonical SETI
# yardstick -- "could we detect us?" (Enriquez et al. 2017, ApJ 849, 104).
ARECIBO_EIRP_W = 2.0e13
# Order-of-magnitude Kardashev Type-I power budget: the ~1e16-1e17 W a
# civilisation commanding its planet's energy supply could in principle beam.
# We use 1e16 W as a round Type-I reference (Kardashev 1964; Sagan 1973).
KARDASHEV_I_W = 1.0e16
# Canonical narrowband beacon emitted bandwidth (a Drake-style ~1 Hz tone).  The
# radiometer channel bandwidth is a *search* parameter; the beacon itself is
# assumed to radiate all its power in this reference bandwidth.
_BEACON_BW_HZ = 1.0
# Optical yardstick: a diffraction-limited 10 m transmitter + MW-class laser.
OPTICAL_EIRP_YARDSTICK_W = 1.0e17


# --- Narrowband radio EIRP limit -------------------------------------------
def eirp_limit(sefd, bandwidth_hz, integration_s, distance_pc, snr=5.0, npol=2,
               beacon_bw_hz: float = _BEACON_BW_HZ) -> dict:
    """Minimum detectable narrowband-beacon EIRP (W) for a radio observation.

    The radiometer equation gives the flux-density sensitivity of a
    single-channel search::

        sigma_S = SNR * SEFD / sqrt(npol * bandwidth_hz * integration_s)   [Jy]

    A narrowband beacon radiating all its power in ``beacon_bw_hz`` (the canonical
    ~1 Hz Drake tone) is then detectable if its received flux exceeds
    ``sigma_S * beacon_bw_hz`` (W/m^2), and the transmitter's isotropic-equivalent
    power follows from the inverse-square law::

        EIRP_min = 4 * pi * d^2 * sigma_S * beacon_bw_hz.

    With ``beacon_bw_hz`` fixed, ``EIRP_min`` scales as ``d^2`` and as
    ``1 / sqrt(bandwidth_hz * integration_s)`` -- deeper (finer-channel,
    longer-integration, nearer) searches constrain fainter beacons.

    Parameters
    ----------
    sefd : float
        System-Equivalent Flux Density of the telescope (Jy).  Lower is more
        sensitive; e.g. GBT L-band ~10 Jy, MeerKAT L-band array ~7 Jy.
    bandwidth_hz : float
        Search channel bandwidth (Hz).  Breakthrough Listen uses ~3 Hz channels.
    integration_s : float
        On-source integration time (s).
    distance_pc : float
        Target distance (pc).  LHS 1140 is at ~14.96 pc.
    snr, npol : float, int
        Detection threshold (default 5 sigma) and number of summed polarisations.
    beacon_bw_hz : float
        Assumed emitted bandwidth of the beacon (default 1 Hz).

    Returns
    -------
    dict with ``eirp_w`` (the limit, W), ``eirp_arecibo_frac`` (relative to the
    ~2e13 W Arecibo radar), ``flux_density_sigma_jy`` (the radiometer sensitivity),
    ``flux_wm2`` (the received-flux limit), and the input assumptions echoed back.

    Honest yardstick: an EIRP limit below ``ARECIBO_EIRP_W`` means the search would
    have detected a transmitter no more powerful than humanity's strongest radar;
    below ``KARDASHEV_I_W`` it constrains only far more capable (Type-I) beacons.
    """
    sefd = float(sefd)
    bandwidth_hz = float(bandwidth_hz)
    integration_s = float(integration_s)
    distance_pc = float(distance_pc)
    if min(sefd, bandwidth_hz, integration_s, distance_pc, snr, npol,
           beacon_bw_hz) <= 0:
        raise ValueError("all eirp_limit inputs must be positive")

    sigma_jy = snr * sefd / np.sqrt(npol * bandwidth_hz * integration_s)
    flux_wm2 = sigma_jy * _JY * beacon_bw_hz
    d_m = distance_pc * _PC_M
    eirp_w = 4.0 * np.pi * d_m * d_m * flux_wm2
    return {
        "eirp_w": float(eirp_w),
        "eirp_arecibo_frac": float(eirp_w / ARECIBO_EIRP_W),
        "flux_density_sigma_jy": float(sigma_jy),
        "flux_wm2": float(flux_wm2),
        "sefd_jy": sefd,
        "bandwidth_hz": bandwidth_hz,
        "integration_s": integration_s,
        "distance_pc": distance_pc,
        "snr": float(snr),
        "npol": int(npol),
        "beacon_bw_hz": float(beacon_bw_hz),
    }


def beacon_capability(eirp_w: float) -> dict:
    """Express an EIRP limit as the *capability* of a beacon it can/can't rule out.

    A limit only means something against a yardstick.  This returns the limit as a
    multiple of the Arecibo planetary radar (~2e13 W) and of a Kardashev Type-I
    power budget (~1e16 W), and a one-line classification of what class of
    transmitter the search reaches:

    * ``below Arecibo`` -- constrains beacons no more powerful than our own radar;
    * ``Arecibo-to-Kardashev-I`` -- reaches only transmitters more capable than us
      but below a Type-I civilisation's budget;
    * ``above Kardashev-I`` -- reaches only implausibly powerful (>Type-I) beacons.
    """
    eirp_w = float(eirp_w)
    arecibo = eirp_w / ARECIBO_EIRP_W
    kardashev = eirp_w / KARDASHEV_I_W
    if eirp_w <= ARECIBO_EIRP_W:
        cls = "below Arecibo (constrains Arecibo-class or fainter beacons)"
    elif eirp_w <= KARDASHEV_I_W:
        cls = "Arecibo-to-Kardashev-I (only super-Arecibo beacons reachable)"
    else:
        cls = "above Kardashev-I (only >Type-I beacons reachable)"
    return {
        "eirp_w": eirp_w,
        "eirp_arecibo_frac": arecibo,
        "eirp_kardashev_I_frac": kardashev,
        "rules_out_arecibo_class": eirp_w <= ARECIBO_EIRP_W,
        "capability_class": cls,
    }


# --- Optical SETI photon-rate limit ----------------------------------------
def optical_seti_limit(aperture_m, distance_pc, wavelength_nm: float = 1064.0,
                       integration_s: float = 1.0, background_rate_hz: float = 100.0,
                       snr: float = 5.0, efficiency: float = 0.1,
                       transmitter_aperture_m: float = 10.0) -> dict:
    """Minimum detectable laser (CW/pulse-averaged) power for an optical SETI search.

    A simple photon-counting limit.  With detector quantum efficiency ``eta``,
    background count rate ``B`` (dark + sky + stellar-wing leakage, counts/s) and
    integration ``tau``, a source producing ``S`` detected counts is seen at

        SNR = S / sqrt(S + B*eta*tau)   (Poisson: signal + background noise),

    which inverts to the minimum detected signal counts

        S_min = (SNR^2 + SNR * sqrt(SNR^2 + 4 B eta tau)) / 2.

    That fixes the minimum photon rate incident on the collector,
    ``R_min = S_min / (eta * tau)``, hence a received power ``R_min * h c / lambda``
    and a received flux over the aperture area.  The isotropic-equivalent
    transmitter power is ``EIRP = 4 pi d^2 * flux``; the *actual* beamed laser power
    is smaller by the transmitter's diffraction-limited gain
    ``G_t = (pi D_t / lambda)^2``.

    Yardstick: a diffraction-limited ``transmitter_aperture_m`` ~10 m aperture
    driven by a ~MW laser radiates ``OPTICAL_EIRP_YARDSTICK_W`` ~1e17 W -- so a
    limit well below that means an existing telescope could catch a plausibly
    buildable beacon (Howard et al. 2004; Wright et al. 2018).

    Returns a dict with the minimum detectable received photon rate, received
    power, isotropic EIRP, and the equivalent beamed laser power.
    """
    aperture_m = float(aperture_m)
    distance_pc = float(distance_pc)
    lam_m = float(wavelength_nm) * 1e-9
    tau = float(integration_s)
    if min(aperture_m, distance_pc, lam_m, tau, snr, efficiency) <= 0:
        raise ValueError("all optical_seti_limit inputs must be positive")

    e_photon = _H * _C / lam_m                        # J per photon
    bkg_counts = float(background_rate_hz) * efficiency * tau
    s_min = 0.5 * (snr * snr + snr * np.sqrt(snr * snr + 4.0 * bkg_counts))
    rate_min = s_min / (efficiency * tau)             # incident photons/s at collector
    power_rec = rate_min * e_photon                   # W collected
    area = np.pi * (aperture_m / 2.0) ** 2
    flux_wm2 = power_rec / area
    d_m = distance_pc * _PC_M
    eirp_w = 4.0 * np.pi * d_m * d_m * flux_wm2
    gain_t = (np.pi * float(transmitter_aperture_m) / lam_m) ** 2
    laser_power_w = eirp_w / gain_t
    return {
        "eirp_w": float(eirp_w),
        "laser_power_w": float(laser_power_w),
        "min_photon_rate_hz": float(rate_min),
        "received_power_w": float(power_rec),
        "flux_wm2": float(flux_wm2),
        "eirp_optical_yardstick_frac": float(eirp_w / OPTICAL_EIRP_YARDSTICK_W),
        "aperture_m": aperture_m,
        "wavelength_nm": float(wavelength_nm),
        "integration_s": tau,
        "transmitter_aperture_m": float(transmitter_aperture_m),
        "detectable_below_yardstick": bool(eirp_w <= OPTICAL_EIRP_YARDSTICK_W),
    }


# --- Radio band assignment --------------------------------------------------
# IEEE-ish radio bands (GHz edges) used to bin observations for the coverage map.
_RADIO_BANDS = [
    ("UHF", 0.3, 1.0),
    ("L", 1.0, 2.0),
    ("S", 2.0, 4.0),
    ("C", 4.0, 8.0),
    ("X", 8.0, 12.0),
    ("Ku", 12.0, 18.0),
    ("K", 18.0, 27.0),
    ("Ka", 27.0, 40.0),
]


def radio_band(freq_ghz) -> str:
    """Name the radio band containing a centre frequency (GHz)."""
    try:
        f = float(freq_ghz)
    except (TypeError, ValueError):
        return "unknown"
    for name, lo, hi in _RADIO_BANDS:
        if lo <= f < hi:
            return name
    return "unknown"


def parse_observation_inventory(records: list[dict], distance_pc: float | None = None,
                                snr: float = 5.0) -> dict:
    """Summarise archive observation metadata into a coverage-and-limit map.

    ``records`` are dicts describing observations of the target.  Recognised
    (case-insensitive) keys:

    * ``telescope`` / ``facility`` -- facility name;
    * ``band`` -- explicit band name, else derived from frequency;
    * ``freq_min_mhz`` / ``freq_max_mhz`` (or ``center_freq_mhz``) -- to assign a
      band when ``band`` is absent;
    * ``sefd_jy``, ``channel_bw_hz``, ``integration_s`` -- to compute an EIRP limit
      via :func:`eirp_limit` (skipped for any record lacking them);
    * ``distance_pc`` -- per-record distance, else the ``distance_pc`` argument;
    * ``npol`` -- polarisations summed (default 2);
    * ``mjd``, ``target`` -- carried through for the table;
    * ``confirmed`` -- ``True`` if this is a real archived/published observation,
      ``False`` for a representative/what-if facility configuration.

    Returns per-facility counts, the set of bands observed, and -- for every band
    for which enough metadata was supplied -- the *best* (lowest) EIRP limit
    achieved and the facility that achieved it.  Records without SEFD/bandwidth/
    integration are counted as coverage but contribute no limit (honest: a
    detection of a pointing is not a sensitivity).
    """
    def get(rec, *keys):
        for k in keys:
            for rk in rec:
                if rk.lower() == k.lower() and rec[rk] not in (None, ""):
                    return rec[rk]
        return None

    def band_of(rec):
        b = get(rec, "band")
        if b:
            return str(b).upper() if len(str(b)) <= 3 else str(b)
        cf = get(rec, "center_freq_mhz", "centre_freq_mhz")
        if cf is None:
            lo = get(rec, "freq_min_mhz")
            hi = get(rec, "freq_max_mhz")
            try:
                cf = 0.5 * (float(lo) + float(hi))
            except (TypeError, ValueError):
                cf = None
        if cf is None:
            return "unknown"
        return radio_band(float(cf) / 1e3)

    per_facility: dict[str, int] = {}
    bands_observed: dict[str, int] = {}
    best_per_band: dict[str, dict] = {}
    n_confirmed = 0
    n_with_limit = 0
    limit_rows = []
    for rec in records:
        fac = str(get(rec, "telescope", "facility") or "unknown")
        per_facility[fac] = per_facility.get(fac, 0) + 1
        band = band_of(rec)
        bands_observed[band] = bands_observed.get(band, 0) + 1
        if bool(get(rec, "confirmed")):
            n_confirmed += 1

        sefd = get(rec, "sefd_jy")
        bw = get(rec, "channel_bw_hz", "bandwidth_hz")
        integ = get(rec, "integration_s", "t_exptime")
        dist = get(rec, "distance_pc") or distance_pc
        if None in (sefd, bw, integ, dist):
            continue
        try:
            lim = eirp_limit(float(sefd), float(bw), float(integ), float(dist),
                             snr=snr, npol=int(get(rec, "npol") or 2))
        except (TypeError, ValueError):
            continue
        n_with_limit += 1
        row = {"facility": fac, "band": band, "mjd": get(rec, "mjd"),
               "confirmed": bool(get(rec, "confirmed")),
               "eirp_w": lim["eirp_w"],
               "eirp_arecibo_frac": lim["eirp_arecibo_frac"]}
        limit_rows.append(row)
        cur = best_per_band.get(band)
        if cur is None or lim["eirp_w"] < cur["eirp_w"]:
            best_per_band[band] = row

    best_overall = None
    if best_per_band:
        best_overall = min(best_per_band.values(), key=lambda r: r["eirp_w"])
    return {
        "n_observations": len(records),
        "n_confirmed_observations": n_confirmed,
        "n_representative": len(records) - n_confirmed,
        "n_with_eirp_limit": n_with_limit,
        "facilities": dict(sorted(per_facility.items(), key=lambda kv: -kv[1])),
        "bands_observed": dict(sorted(bands_observed.items(),
                                      key=lambda kv: -kv[1])),
        "best_eirp_limit_per_band": best_per_band,
        "best_eirp_limit_overall": best_overall,
        "limit_rows": limit_rows,
    }


__all__ = [
    "ARECIBO_EIRP_W", "KARDASHEV_I_W", "OPTICAL_EIRP_YARDSTICK_W",
    "eirp_limit", "beacon_capability", "optical_seti_limit",
    "radio_band", "parse_observation_inventory",
]
