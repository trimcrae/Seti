"""The data model every Roman module speaks, and the config loader.

Roman will publish Level 1 ramps, Level 2 calibrated exposures, Level 3
mosaics and Level 4 catalogues / light curves / extracted spectra, in ASDF and
in tables, from IRSA (with cloud copies on S3).  The exact file layouts are
only partly public before launch and will change between pipeline builds, so
nothing downstream of :mod:`seti.roman.products` is allowed to know what a
Roman file looks like.  Every adapter emits one of the four structures here
and every detector consumes only these:

``LightCurve``  one star, one band, in flux units with a per-epoch dq mask.
``Spectrum``    one source, one dispersive mode, wavelength in vacuum microns.
``DQCutout``    a data-quality (bit-mask) image cutout plus the catalogued
                stars inside it, in pixel coordinates.
``Ramp``        a Level 1 up-the-ramp cube (resultants x ny x nx) for one
                small pixel box, with the resultant mid-times.

Every field a discriminator needs is ``| None``-typed on purpose (the TOCSIN
lesson): a product that does not carry, say, a jump flag must yield ``None``,
and the funnel then records the test as *not run* rather than as passed.

``Funnel`` is the counter/rejection ledger every stage writes into
``summary.json`` -- the ``clean_in_N_of_M`` discipline of ``channel-brief.md``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

# Offline fallback for the WFI data-quality bit table.  ``products.dq_flags`` replaces
# it with ``roman_datamodels.dqflags.pixel`` whenever that package is importable, and
# records which of the two was used.  Bits follow the JWST/Roman convention; the ones
# this package tests (DO_NOT_USE, SATURATED, JUMP_DET, PERSISTENCE, DEAD, HOT) are the
# stable low bits shared with JWST.
DQ_FLAGS_FALLBACK: dict[str, int] = {
    "DO_NOT_USE": 1 << 0,
    "SATURATED": 1 << 1,
    "JUMP_DET": 1 << 2,
    "DROPOUT": 1 << 3,
    "GW_AFFECTED_DATA": 1 << 4,
    "PERSISTENCE": 1 << 5,
    "AD_FLOOR": 1 << 6,
    "OUTLIER": 1 << 7,
    "UNRELIABLE_ERROR": 1 << 8,
    "NON_SCIENCE": 1 << 9,
    "DEAD": 1 << 10,
    "HOT": 1 << 11,
    "WARM": 1 << 12,
    "LOW_QE": 1 << 13,
    "TELEGRAPH": 1 << 15,
    "NONLINEAR": 1 << 16,
    "BAD_REF_PIXEL": 1 << 17,
    "NO_FLAT_FIELD": 1 << 18,
    "NO_GAIN_VALUE": 1 << 19,
    "NO_LIN_CORR": 1 << 20,
    "NO_SAT_CHECK": 1 << 21,
    "UNRELIABLE_BIAS": 1 << 22,
    "UNRELIABLE_DARK": 1 << 23,
    "UNRELIABLE_SLOPE": 1 << 24,
    "UNRELIABLE_FLAT": 1 << 25,
    "REFERENCE_PIXEL": 1 << 31,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _deep_update(base: dict, extra: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_roman_config(root: Path | None = None, overrides: dict | None = None) -> dict:
    """``config/roman.yaml`` (the single source of every number), with overrides.

    A missing or unparsable file is reported, not hidden: the returned dict then
    carries ``config_status: "defaults"`` so a run can say it ran on defaults.
    """
    conf: dict = {"config_status": "loaded"}
    try:
        import yaml
        p = (root or repo_root()) / "config" / "roman.yaml"
        if p.exists():
            conf = _deep_update(conf, yaml.safe_load(p.read_text()) or {})
        else:
            conf["config_status"] = "defaults"
    except Exception as exc:  # noqa: BLE001
        conf["config_status"] = f"defaults ({exc!r})"
    if overrides:
        conf = _deep_update(conf, overrides)
    return conf


# --------------------------------------------------------------------------------------
# JSON helpers shared by every stage
# --------------------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_safe(o):
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "as_dict"):
        return json_safe(o.as_dict())
    return o


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(obj), indent=1, default=str))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return default


# --------------------------------------------------------------------------------------
# Funnel ledger
# --------------------------------------------------------------------------------------

@dataclass
class Funnel:
    """Stage counters and named rejections, serialised into every summary."""

    counts: dict[str, int] = field(default_factory=dict)
    rejections: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = int(self.counts.get(key, 0)) + int(n)

    def reject(self, reason: str, n: int = 1) -> None:
        self.rejections[reason] = int(self.rejections.get(reason, 0)) + int(n)

    def note(self, text: str) -> None:
        self.notes.append(str(text))

    def merge(self, other: Funnel) -> None:
        for k, v in other.counts.items():
            self.bump(k, v)
        for k, v in other.rejections.items():
            self.reject(k, v)
        self.notes.extend(other.notes)

    def as_dict(self) -> dict:
        return {"counts": dict(self.counts), "rejections": dict(self.rejections),
                "notes": list(self.notes)}


# --------------------------------------------------------------------------------------
# The four product structures
# --------------------------------------------------------------------------------------

def _arr(x, dtype=float) -> np.ndarray | None:
    if x is None:
        return None
    return np.asarray(x, dtype=dtype)


@dataclass
class LightCurve:
    """One star, one band.  Flux in the unit named by ``flux_unit``.

    ``mjd`` is the exposure mid-time (TDB if the product says so; ``time_system``
    records what it is).  ``dq`` is the per-epoch bit mask in the WFI convention
    (0 = clean); ``None`` when the product carries none.  ``flux_zp_ab`` is the AB
    magnitude of one flux unit, needed to express the curve in magnitudes for the
    bridged channels; ``None`` means the curve can only be used in relative flux.
    """

    star_id: str
    ra: float
    dec: float
    band: str
    mjd: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    survey: str = ""
    flux_unit: str = "e-/s"
    flux_zp_ab: float | None = None
    time_system: str = "unknown"
    dq: np.ndarray | None = None
    exposure_s: float | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mjd = _arr(self.mjd)
        self.flux = _arr(self.flux)
        self.flux_err = _arr(self.flux_err)
        self.dq = _arr(self.dq, dtype=np.int64) if self.dq is not None else None
        n = self.mjd.size
        if self.flux.size != n or self.flux_err.size != n or (self.dq is not None and self.dq.size != n):
            raise ValueError("LightCurve arrays must have one length")

    @property
    def n(self) -> int:
        return int(self.mjd.size)

    def good(self, dq_reject_mask: int | None = None) -> np.ndarray:
        """Epochs with finite time/flux/err, positive err, and no rejected dq bit."""
        g = np.isfinite(self.mjd) & np.isfinite(self.flux) & np.isfinite(self.flux_err) & (self.flux_err > 0)
        if self.dq is not None and dq_reject_mask:
            g &= (self.dq & int(dq_reject_mask)) == 0
        return g

    def sorted(self) -> LightCurve:
        o = np.argsort(self.mjd)
        return LightCurve(self.star_id, self.ra, self.dec, self.band, self.mjd[o], self.flux[o],
                          self.flux_err[o], self.survey, self.flux_unit, self.flux_zp_ab,
                          self.time_system, None if self.dq is None else self.dq[o],
                          self.exposure_s, dict(self.meta))

    def to_mag(self, dq_reject_mask: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """(mjd, mag, magerr) on good epochs with positive flux; ``None`` if no zero point.

        With no zero point the magnitudes are *relative* (zp 25.0) and the caller
        must not compare them with anything outside this curve; that is flagged by
        ``flux_zp_ab is None``.
        """
        g = self.good(dq_reject_mask) & (self.flux > 0)
        if not np.any(g):
            return None
        zp = 25.0 if self.flux_zp_ab is None else float(self.flux_zp_ab)
        f, e = self.flux[g], self.flux_err[g]
        mag = zp - 2.5 * np.log10(f)
        magerr = 2.5 / np.log(10.0) * e / f
        return self.mjd[g], mag, magerr

    def as_dict(self) -> dict:
        return {"star_id": self.star_id, "ra": self.ra, "dec": self.dec, "band": self.band,
                "survey": self.survey, "n": self.n, "flux_unit": self.flux_unit,
                "flux_zp_ab": self.flux_zp_ab, "time_system": self.time_system,
                "mjd_range": [float(np.nanmin(self.mjd)), float(np.nanmax(self.mjd))] if self.n else None,
                "exposure_s": self.exposure_s, "meta": dict(self.meta)}


@dataclass
class Spectrum:
    """One source, one dispersive element (``G150`` grism, ``P127`` prism, ``CGI_B3`` ...).

    ``wavelength_um`` is VACUUM microns (Roman calibrations are vacuum; the
    industrial line list in :mod:`seti.roman.lines` is converted to vacuum at
    definition time).  ``resolving_power`` may be a scalar or an array the same
    length as ``wavelength_um`` (the prism's R runs 80-180 across the band).
    ``contam`` is the slitless-contamination estimate per sample when the
    pipeline provides one; ``None`` means the overlap test cannot be run and the
    funnel must say so.
    """

    source_id: str
    ra: float
    dec: float
    mode: str
    wavelength_um: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    survey: str = ""
    flux_unit: str = "arbitrary"
    resolving_power: float | np.ndarray | None = None
    dq: np.ndarray | None = None
    contam: np.ndarray | None = None
    is_point_source: bool | None = None
    detector: str | None = None
    trace_pixel: np.ndarray | None = None      # detector pixel of each sample along the trace
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.wavelength_um = _arr(self.wavelength_um)
        self.flux = _arr(self.flux)
        self.flux_err = _arr(self.flux_err)
        n = self.wavelength_um.size
        if self.flux.size != n or self.flux_err.size != n:
            raise ValueError("Spectrum arrays must have one length")
        if isinstance(self.resolving_power, (list, tuple, np.ndarray)):
            self.resolving_power = _arr(self.resolving_power)
            if self.resolving_power.size != n:
                raise ValueError("resolving_power array must match wavelength")
        for name in ("dq", "contam", "trace_pixel"):
            v = getattr(self, name)
            if v is not None:
                v = _arr(v, dtype=np.int64 if name == "dq" else float)
                if v.size != n:
                    raise ValueError(f"{name} must match wavelength")
                setattr(self, name, v)

    @property
    def n(self) -> int:
        return int(self.wavelength_um.size)

    def r_at(self, wl_um: float) -> float | None:
        if self.resolving_power is None:
            return None
        if isinstance(self.resolving_power, np.ndarray):
            return float(np.interp(wl_um, self.wavelength_um, self.resolving_power))
        return float(self.resolving_power)

    def as_dict(self) -> dict:
        return {"source_id": self.source_id, "ra": self.ra, "dec": self.dec, "mode": self.mode,
                "survey": self.survey, "n": self.n, "flux_unit": self.flux_unit,
                "wavelength_range_um": [float(np.nanmin(self.wavelength_um)),
                                        float(np.nanmax(self.wavelength_um))] if self.n else None,
                "is_point_source": self.is_point_source, "detector": self.detector,
                "has_contam": self.contam is not None, "has_dq": self.dq is not None,
                "meta": dict(self.meta)}


@dataclass
class DQCutout:
    """A data-quality bit-mask cutout (uint32) and the catalogued stars inside it.

    ``stars`` is a list of dicts with at least ``star_id, x, y`` (pixel coordinates
    in the cutout frame, 0-based) and optionally ``mag`` and ``saturated``.
    ``psf_fwhm_px`` is the PSF FWHM at this filter (undersampled: ~1-1.4 px in
    F146).  ``x0, y0`` locate the cutout on the detector so that a per-pixel
    recurrence ledger (hot pixels) can be kept across exposures.
    """

    image_id: str
    dq: np.ndarray
    stars: list[dict]
    detector: str = ""
    band: str = ""
    mjd: float | None = None
    x0: int = 0
    y0: int = 0
    psf_fwhm_px: float = 1.2
    exposure_s: float | None = None
    n_resultants: int | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.dq = np.asarray(self.dq)
        if self.dq.ndim != 2:
            raise ValueError("dq must be 2-D")

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(s) for s in self.dq.shape)

    def as_dict(self) -> dict:
        return {"image_id": self.image_id, "detector": self.detector, "band": self.band,
                "mjd": self.mjd, "x0": self.x0, "y0": self.y0, "shape": list(self.shape),
                "n_stars": len(self.stars), "psf_fwhm_px": self.psf_fwhm_px,
                "exposure_s": self.exposure_s, "n_resultants": self.n_resultants,
                "meta": dict(self.meta)}


@dataclass
class Ramp:
    """A Level 1 up-the-ramp cube for one pixel box: ``resultants[k, y, x]``.

    ``times_s`` are the resultant mid-times from exposure start (each resultant
    is the average of the frames in its read-pattern group).  ``read_pattern``
    is the list of frame-index groups, as the MA table gives it.  ``dq`` is the
    per-resultant flag cube if the product carries one (Level 1 usually carries
    none: ``None``).  Units are DN unless ``unit`` says otherwise.
    """

    image_id: str
    resultants: np.ndarray
    times_s: np.ndarray
    x0: int = 0
    y0: int = 0
    read_pattern: list[list[int]] | None = None
    frame_time_s: float | None = None
    unit: str = "DN"
    gain_e_per_dn: float | None = None
    dq: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.resultants = np.asarray(self.resultants, dtype=float)
        self.times_s = _arr(self.times_s)
        if self.resultants.ndim != 3:
            raise ValueError("resultants must be (n_resultants, ny, nx)")
        if self.times_s.size != self.resultants.shape[0]:
            raise ValueError("times_s must have one entry per resultant")
        if self.dq is not None:
            self.dq = np.asarray(self.dq)
            if self.dq.shape != self.resultants.shape:
                raise ValueError("dq cube must match resultants")

    @property
    def n_resultants(self) -> int:
        return int(self.resultants.shape[0])

    def as_dict(self) -> dict:
        return {"image_id": self.image_id, "x0": self.x0, "y0": self.y0,
                "shape": [int(s) for s in self.resultants.shape],
                "times_s": self.times_s.tolist(), "frame_time_s": self.frame_time_s,
                "unit": self.unit, "gain_e_per_dn": self.gain_e_per_dn, "meta": dict(self.meta)}


@dataclass
class RomanProduct:
    """One archive product as the inventory records it (no data, only provenance).

    ``level`` in {"L1","L2","L3","L4","catalog","sim","unknown"}; ``kind`` in
    {"uncal","cal","coadd","lightcurve","spectrum_1d","spectrum_2d","catalog",
    "cgi","unknown"}; ``origin`` names the service (``irsa_tap``, ``irsa_s3``,
    ``mast``, ``local``); ``simulated`` is True for any pre-launch product.
    """

    uri: str
    level: str = "unknown"
    kind: str = "unknown"
    origin: str = ""
    survey: str = ""
    instrument: str = "WFI"
    band: str | None = None
    size_bytes: int | None = None
    simulated: bool | None = None
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["meta"] = dict(self.meta)
        return d


# --------------------------------------------------------------------------------------
# Survey descriptions from config
# --------------------------------------------------------------------------------------

def survey_table(conf: dict) -> dict[str, dict]:
    """The ``surveys:`` block, keyed by survey id, each row carrying ``verify`` notes."""
    return {str(k): dict(v) for k, v in (conf.get("surveys") or {}).items() if isinstance(v, dict)}


def filter_table(conf: dict) -> dict[str, dict]:
    return {str(k): dict(v) for k, v in
            ((conf.get("instruments") or {}).get("WFI", {}).get("filters") or {}).items()
            if isinstance(v, dict)}


def band_wavelength_um(conf: dict, band: str) -> tuple[float, float] | None:
    row = filter_table(conf).get(str(band).upper())
    if not row or "range_um" not in row:
        return None
    lo, hi = row["range_um"]
    return float(lo), float(hi)


def flag_value(conf: dict, name: str, table: dict[str, int] | None = None) -> int:
    """Bit value for a dq flag name; config ``dq_flags`` overrides the fallback."""
    tab = dict(DQ_FLAGS_FALLBACK)
    tab.update({str(k): int(v) for k, v in (conf.get("dq_flags") or {}).items()})
    if table:
        tab.update({str(k): int(v) for k, v in table.items()})
    return int(tab[name])


def summary_skeleton(channel: str, conf: dict) -> dict[str, Any]:
    """The common header of every ``results/roman/<channel>/summary.json``."""
    return {"channel": channel, "written_utc": utc_now(), "config_status": conf.get("config_status"),
            "verdict": "NO_DATA_REACHED", "funnel": Funnel().as_dict(), "data_state": None,
            "simulated_inputs": None}
