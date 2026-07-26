"""MIDDEN per-spectrum measurement and census scoring.

Per spectrum: read the Phase-3 1D FITS (binary-table and image-HDU forms both
handled), establish the stellar radial velocity (measured header CCF keyword
when present and consistent, else Fe I cross-correlation — which also absorbs
any global air/vacuum or zero-point offset, both velocity-like), shift every
line window to the stellar frame, and measure a local-continuum-normalized
central depth + equivalent-width proxy per target/control line.

Scoring is deliberately **self-calibrating** (same philosophy as
herdsman_b.score): no absolute synthesis is trusted.  Each star's depth at
each wavelength is ranked against the distribution of depths AT THE SAME
WAVELENGTH across all corpus stars within +-250 K Teff — the corpus is its
own template library, so blends, line forests and continuum systematics that
affect everyone cancel.  A candidate needs >= 2 radionuclide lines in excess
(or the coherent Tc-triplet path), clean controls, and epoch persistence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

C_KMS = 299792.458

# Detection thresholds (docs/midden.md section 4).
Z_LINE = 4.0            # per-line census z to flag a single line
Z_TC_EACH = 2.5         # coherent-triplet path: every Tc component above this
Z_TC_QUAD = 4.5         # ... and quadrature sum above this
DEPTH_SNR_MIN = 2.0     # measured depth must exceed 2x its own local error
RV_MAX_KMS = 400.0      # sanity bound on any adopted RV

_WAVE_COLS = ("wave", "wavelength", "lambda", "awav", "wavelength_air")
_FLUX_COLS = ("flux", "flux_reduced", "flux_calibrated", "normflux", "spectrum")
# Only *measured* CCF keywords are trusted; ESO TEL TARG RADVEL is the
# user-supplied catalog value (often 0.0) and is deliberately excluded.
_RV_KEYS = ("HIERARCH ESO DRS CCF RVC", "ESO DRS CCF RVC",
            "HIERARCH ESO DRS CCF RV", "ESO DRS CCF RV",
            "HIERARCH ESO QC CCF RV", "ESO QC CCF RV")


# ---------------------------------------------------------------------------
# Spectrum I/O
# ---------------------------------------------------------------------------

def read_spectrum(path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Wavelength (air A), flux, and merged header from a Phase-3 1D FITS.

    Handles both conventions defensively: a binary-table HDU with WAVE/FLUX
    array columns (HARPS/FEROS ADP standard: one row of vectors) and an image
    HDU with a CRVAL1/CDELT1(CD1_1) linear WCS.  Wavelength units are
    sanity-fixed (um/nm -> A) by magnitude.
    """
    from astropy.io import fits

    wave = flux = None
    header: dict = {}
    with fits.open(path, memmap=False) as hdul:
        header = dict(hdul[0].header)
        for hdu in hdul:
            if hdu.data is None or not hasattr(hdu, "columns"):
                continue
            names = {n.lower(): n for n in hdu.columns.names}
            wcol = next((names[c] for c in _WAVE_COLS if c in names), None)
            fcol = next((names[c] for c in _FLUX_COLS if c in names), None)
            if wcol and fcol:
                wave = np.asarray(hdu.data[wcol], float).ravel()
                flux = np.asarray(hdu.data[fcol], float).ravel()
                header.update(dict(hdu.header))
                break
        if wave is None:
            for hdu in hdul:
                data = getattr(hdu, "data", None)
                if data is None or hasattr(hdu, "columns"):
                    continue
                arr = np.squeeze(np.asarray(data, float))
                if arr.ndim != 1 or arr.size < 100:
                    continue
                h = hdu.header
                crval = h.get("CRVAL1")
                cdelt = h.get("CDELT1", h.get("CD1_1"))
                if crval is None or cdelt is None:
                    continue
                crpix = float(h.get("CRPIX1", 1.0))
                wave = float(crval) + (np.arange(arr.size) - (crpix - 1.0)) * float(cdelt)
                flux = arr
                header.update(dict(h))
                break
    if wave is None or flux is None:
        raise ValueError(f"no 1D spectrum found in {path}")

    med = float(np.nanmedian(wave))
    if med < 10.0:                 # micron
        wave = wave * 1e4
    elif med < 1000.0:             # nm
        wave = wave * 10.0
    ok = np.isfinite(wave) & np.isfinite(flux)
    wave, flux = wave[ok], flux[ok]
    order = np.argsort(wave)
    return wave[order], flux[order], header


def header_rv(header: dict) -> float | None:
    """Measured pipeline CCF RV from the header (km/s), if present and sane."""
    for key in _RV_KEYS:
        v = header.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and abs(v) < RV_MAX_KMS and v != 0.0:
            return v
    return None


# ---------------------------------------------------------------------------
# Continuum + line measurement
# ---------------------------------------------------------------------------

def _mad_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def local_continuum(wave, flux, lam0, half=3.0, exclude=0.25, deg=2):
    """Robust local continuum around lam0.

    Fits a degree-``deg`` polynomial over +-``half`` A excluding the central
    +-``exclude`` A, with asymmetric sigma-clipping (absorption pixels are
    low outliers and get rejected, so a sloped continuum is recovered
    correctly).  Returns (w, f, cont, sigma) or None if the window is empty.
    """
    sel = (wave >= lam0 - half) & (wave <= lam0 + half) \
        & np.isfinite(wave) & np.isfinite(flux)
    if sel.sum() < max(20, deg + 5):
        return None
    w, f = wave[sel], flux[sel]
    x = w - lam0
    fit_mask = np.abs(x) > exclude
    if fit_mask.sum() < deg + 3:
        return None
    m = fit_mask.copy()
    coeff = None
    for _ in range(3):
        if m.sum() < deg + 3:
            break
        coeff = np.polyfit(x[m], f[m], deg)
        r = f - np.polyval(coeff, x)
        s = _mad_std(r[m])
        if not np.isfinite(s) or s <= 0:
            break
        med = float(np.median(r[m]))
        m = fit_mask & (r > med - 2.0 * s) & (r < med + 3.0 * s)
    if coeff is None:
        return None
    cont = np.polyval(coeff, x)
    sigma = _mad_std((f - cont)[m])
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = max(_mad_std(f - cont), 1e-8)
    return w, f, cont, sigma


def measure_line(wave, flux, lam0, core=0.15, ew_half=0.75) -> dict:
    """Central depth + EW proxy of the feature at lam0 (stellar-frame A)."""
    nan = {"depth": np.nan, "depth_err": np.nan, "ew_ma": np.nan,
           "n_core": 0, "cont_snr": np.nan}
    loc = local_continuum(wave, flux, lam0)
    if loc is None:
        return nan
    w, f, cont, sigma = loc
    good = cont > 0
    if good.sum() < 10:
        return nan
    fn = np.where(good, f / np.where(good, cont, 1.0), np.nan)
    core_sel = (np.abs(w - lam0) <= core) & good
    n_core = int(core_sel.sum())
    if n_core < 2:
        return nan
    depth = float(np.nanmean(1.0 - fn[core_sel]))
    rel_sigma = sigma / max(float(np.median(cont[good])), 1e-12)
    depth_err = float(rel_sigma / np.sqrt(n_core))
    ew_sel = (np.abs(w - lam0) <= ew_half) & good
    dlam = float(np.median(np.diff(w))) if len(w) > 1 else np.nan
    ew_ma = float(np.nansum(1.0 - fn[ew_sel]) * dlam * 1000.0)  # milli-Angstrom
    return {"depth": depth, "depth_err": depth_err, "ew_ma": ew_ma,
            "n_core": n_core, "cont_snr": float(1.0 / max(rel_sigma, 1e-12))}


# ---------------------------------------------------------------------------
# Radial velocity by Fe I cross-correlation
# ---------------------------------------------------------------------------

def estimate_rv(wave, flux, ref_lines, v_max=250.0, dv=0.5,
                min_lines=3, min_mean_depth=0.02) -> dict:
    """RV (km/s) maximizing summed line depth at shifted Fe I rest wavelengths.

    Grid search +-v_max at dv resolution with parabolic refinement.  Because
    the score is evaluated at lam0*(1+v/c) for every line simultaneously, any
    global multiplicative wavelength offset (air/vacuum, zero point) is
    absorbed into the returned velocity — exactly what shifting the target
    windows needs.
    """
    v = np.arange(-v_max, v_max + dv, dv)
    score = np.zeros_like(v)
    n_used = 0
    for lam0 in ref_lines:
        loc = local_continuum(wave, flux, lam0, half=6.0, exclude=0.0, deg=2)
        if loc is None:
            continue
        w, f, cont, _ = loc
        good = cont > 0
        if good.sum() < 20:
            continue
        d = 1.0 - f[good] / cont[good]
        lam_v = lam0 * (1.0 + v / C_KMS)
        prof = np.interp(lam_v, w[good], d, left=0.0, right=0.0)
        if np.nanmax(prof) > 0.01:
            score += prof
            n_used += 1
    if n_used < min_lines:
        return {"rv_kms": np.nan, "n_lines": n_used, "mean_depth": np.nan}
    i = int(np.argmax(score))
    rv = float(v[i])
    if 0 < i < len(v) - 1:
        y0, y1, y2 = score[i - 1], score[i], score[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom < 0:
            rv += float(0.5 * (y0 - y2) / denom) * dv
    mean_depth = float(score[i] / n_used)
    if mean_depth < min_mean_depth:
        return {"rv_kms": np.nan, "n_lines": n_used, "mean_depth": mean_depth}
    return {"rv_kms": rv, "n_lines": n_used, "mean_depth": mean_depth}


def analyze_spectrum(path, meta: dict, line_set=None) -> list[dict]:
    """All per-line measurements for one FITS spectrum (stellar frame)."""
    wave, flux, header = read_spectrum(path)
    return analyze_arrays(wave, flux, header, meta, line_set=line_set)


def analyze_arrays(wave, flux, header, meta: dict, line_set=None) -> list[dict]:
    """Measurement core on in-memory arrays (offline-testable).

    ``meta`` (star id, teff, dp_id, ...) is copied into every output row so
    the checkpointed parquet is self-contained and re-scoring needs no
    re-download.
    """
    from .lines import LINES, rv_reference_wavelengths

    line_set = LINES if line_set is None else line_set
    rv_hdr = header_rv(header or {})
    est = estimate_rv(wave, flux, rv_reference_wavelengths())
    rv_ccf = est["rv_kms"]
    if rv_hdr is not None and np.isfinite(rv_ccf) and abs(rv_hdr - rv_ccf) <= 5.0:
        rv, rv_source = rv_hdr, "header"
    elif np.isfinite(rv_ccf):
        rv, rv_source = rv_ccf, "ccf"
    elif rv_hdr is not None:
        rv, rv_source = rv_hdr, "header_unconfirmed"
    else:
        rv, rv_source = 0.0, "none"

    rows = []
    for ln in line_set:
        lam_star = ln.wavelength * (1.0 + rv / C_KMS)
        m = measure_line(wave, flux, lam_star)
        rows.append({**meta, "species": ln.species, "wavelength": ln.wavelength,
                     "role": ln.role, "rv_kms": rv, "rv_source": rv_source,
                     "rv_ccf_nlines": est["n_lines"], **m})
    return rows


# ---------------------------------------------------------------------------
# Census scoring
# ---------------------------------------------------------------------------

def census_z(meas: pd.DataFrame, teff_window: float = 250.0,
             min_peers: int = 20) -> pd.DataFrame:
    """Attach the self-calibrating census z to every (spectrum, line) row.

    For each wavelength, each spectrum's depth is ranked against the depths
    at the SAME wavelength across all corpus spectra whose star Teff is
    within +-teff_window K (all spectra if Teff is missing or peers are too
    few).  Robust location/scale (median + MAD) keep a genuine outlier from
    polluting its own reference distribution.
    """
    meas = meas.copy()
    meas["z"] = np.nan
    for _, g in meas.groupby("wavelength", sort=False):
        d = g["depth"].to_numpy(float)
        teff = pd.to_numeric(g.get("teff"), errors="coerce").to_numpy(float) \
            if "teff" in g else np.full(len(g), np.nan)
        z = np.full(len(g), np.nan)
        finite_t = np.isfinite(teff)
        for i in range(len(g)):
            if not np.isfinite(d[i]):
                continue
            if finite_t[i]:
                peers = finite_t & (np.abs(teff - teff[i]) <= teff_window)
            else:
                peers = np.ones(len(g), bool)
            if peers.sum() < min_peers:
                peers = np.ones(len(g), bool)
            dd = d[peers & np.isfinite(d)]
            if len(dd) < 5:
                continue
            mu = float(np.median(dd))
            sd = max(_mad_std(dd), 1e-4)
            z[i] = (d[i] - mu) / sd
        meas.loc[g.index, "z"] = z
    return meas


def _epoch_verdict(g: pd.DataFrame) -> dict:
    """Candidate logic for one spectrum (one star epoch)."""
    radio = g[g["role"] == "radionuclide"]
    ctrl = g[g["role"] == "control"]
    tc = g[g["species"] == "Tc I"]

    def _flagged(rows):
        z = rows["z"].to_numpy(float)
        d = rows["depth"].to_numpy(float)
        e = rows["depth_err"].to_numpy(float)
        return np.isfinite(z) & (z >= Z_LINE) & (d >= DEPTH_SNR_MIN * e)

    n_radio_flagged = int(_flagged(radio).sum())
    tcz = tc["z"].to_numpy(float)
    tcd = tc["depth"].to_numpy(float)
    tce = tc["depth_err"].to_numpy(float)
    tc_ok = (len(tc) == 3 and np.all(np.isfinite(tcz))
             and bool(np.all(tcz >= Z_TC_EACH))
             and float(np.sqrt(np.sum(tcz ** 2))) >= Z_TC_QUAD
             and bool(np.all(tcd >= DEPTH_SNR_MIN * tce)))
    ctrl_z = ctrl["z"].to_numpy(float)
    control_veto = bool(np.any(np.isfinite(ctrl_z) & (ctrl_z >= Z_LINE)))
    epoch_pass = (n_radio_flagged >= 2) or tc_ok
    return {"n_radio_flagged": n_radio_flagged, "tc_coherent": tc_ok,
            "control_veto": control_veto,
            "epoch_candidate": bool(epoch_pass and not control_veto)}


def score_corpus(meas: pd.DataFrame, teff_window: float = 250.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-star summary, per-row measurements with z) for the whole corpus.

    Candidate rule: >= 2 radionuclide lines at census z >= 4 (or the coherent
    Tc-triplet path: all three components z >= 2.5 with quadrature >= 4.5),
    NO control line flagged, and — when a star has multiple epochs — the
    epoch verdict repeats in >= 2 epochs.
    """
    meas = census_z(meas, teff_window=teff_window)
    epoch_rows = []
    for (star, dp_id), g in meas.groupby(["star", "dp_id"], sort=False):
        v = _epoch_verdict(g)
        radio = g[g["role"] == "radionuclide"]
        zmap = {f"z_{r.species.replace(' ', '')}_{r.wavelength:.0f}": float(r.z)
                for r in radio.itertuples() if np.isfinite(r.z)}
        epoch_rows.append({"star": star, "dp_id": dp_id, **v, **zmap,
                           "z_control_max": float(np.nanmax(
                               g.loc[g["role"] == "control", "z"].to_numpy(float),
                               initial=-np.inf)),
                           "rv_kms": float(g["rv_kms"].iloc[0]),
                           "rv_source": g["rv_source"].iloc[0]})
    epochs = pd.DataFrame(epoch_rows)
    if not len(epochs):
        return pd.DataFrame(), meas

    star_rows = []
    for star, g in epochs.groupby("star", sort=False):
        n_ep = int(len(g))
        n_pass = int(g["epoch_candidate"].sum())
        candidate = bool(n_pass >= (2 if n_ep >= 2 else 1))
        zcols = [c for c in g.columns if c.startswith("z_")
                 and c != "z_control_max"]
        agg = {c: float(np.nanmedian(pd.to_numeric(g[c], errors="coerce")))
               for c in zcols}
        star_rows.append({"star": star, "n_epochs": n_ep,
                          "n_epochs_candidate": n_pass,
                          "candidate": candidate,
                          "any_control_veto": bool(g["control_veto"].any()),
                          "tc_coherent_any": bool(g["tc_coherent"].any()),
                          "max_radio_flagged": int(g["n_radio_flagged"].max()),
                          **agg})
    stars = pd.DataFrame(star_rows)
    order_col = next((c for c in stars.columns if c.startswith("z_TcI")), None)
    if order_col is not None:
        stars = stars.sort_values([order_col], ascending=False, na_position="last")
    return stars.reset_index(drop=True), meas


def line_flag_rates(meas: pd.DataFrame) -> pd.DataFrame:
    """Fraction of spectra with z >= Z_LINE per line — the report's honesty table."""
    rows = []
    for (species, lam, role), g in meas.groupby(["species", "wavelength", "role"]):
        z = g["z"].to_numpy(float)
        fin = np.isfinite(z)
        rows.append({"species": species, "wavelength": lam, "role": role,
                     "n_measured": int(fin.sum()),
                     "n_flagged": int((z[fin] >= Z_LINE).sum()),
                     "flag_rate": float((z[fin] >= Z_LINE).mean()) if fin.any() else np.nan})
    return pd.DataFrame(rows).sort_values("wavelength").reset_index(drop=True)


__all__ = ["C_KMS", "DEPTH_SNR_MIN", "Z_LINE", "Z_TC_EACH", "Z_TC_QUAD",
           "analyze_arrays", "analyze_spectrum", "census_z", "estimate_rv", "header_rv",
           "line_flag_rates", "local_continuum", "measure_line",
           "read_spectrum", "score_corpus"]
