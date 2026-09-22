"""The published tables FORGE reads, and how a star is named across them.

Embedded assets (``src/seti/data_assets/forge_*.csv``)
------------------------------------------------------
``forge_targets.csv``      one row per star that any published interferometric
                           exozodi survey observed: HD, HIP, common name,
                           spectral type, which surveys (``samples``).
``forge_excess.csv``       excess measurements transcribed from the papers,
                           one per (star, survey, band, epoch).  Every numeric
                           value carries ``verify = unverified`` until the
                           runner's VizieR pass confirms it; a value the
                           archive contradicts is marked ``discrepant`` and the
                           archive value is used.
``forge_polarimetry.csv``  the polarimetric null (Marshall et al. 2016).
``forge_companions.csv``   catalogued companions at the percent level.

THE EMBEDDED VALUES ARE SEEDS, NOT DATA.  They were transcribed from the
publications without the archive in reach (the sandbox has no egress), so a
transcription error is possible and would be indistinguishable from a
measurement.  That is why :mod:`seti.forge.physics` refuses to promote a star
whose driving values are not archive-read or archive-verified: the tier is
``UNVERIFIED_INPUT`` instead of ``candidate``.

Identifiers
-----------
A star is keyed ``"HD <n>"``.  Every table names stars differently (``HD``
numbers as integers, ``HIP`` numbers, Bayer names in three spellings, proper
names), so :func:`canonical_key` maps any of those onto the HD key through
the alias table the targets asset builds.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .physics import BAND_WAVELENGTH_UM, Measurement

ASSET_DIR = Path(__file__).resolve().parents[1] / "data_assets"

GREEK = {
    "alp": "alpha", "alf": "alpha", "bet": "beta", "gam": "gamma", "del": "delta",
    "eps": "epsilon", "zet": "zeta", "eta": "eta", "the": "theta", "tet": "theta",
    "iot": "iota", "kap": "kappa", "lam": "lambda", "mu": "mu", "nu": "nu", "ksi": "xi",
    "xi": "xi", "omi": "omicron", "pi": "pi", "rho": "rho", "sig": "sigma", "tau": "tau",
    "ups": "upsilon", "phi": "phi", "chi": "chi", "psi": "psi", "ome": "omega",
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ζ": "zeta",
    "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu", "ν": "nu",
    "ξ": "xi", "ο": "omicron", "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau",
    "υ": "upsilon", "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
}

# Approximate main-sequence Teff by spectral type (Pecaut & Mamajek 2013 scale,
# rounded); the runner replaces these with the archive's value where it has one.
_TEFF_BY_TYPE = {
    "A0": 9700, "A1": 9200, "A2": 8840, "A3": 8550, "A4": 8270, "A5": 8080, "A6": 8000,
    "A7": 7800, "A8": 7500, "A9": 7440, "F0": 7220, "F1": 7020, "F2": 6820, "F3": 6750,
    "F4": 6670, "F5": 6550, "F6": 6350, "F7": 6280, "F8": 6180, "F9": 6050, "G0": 5930,
    "G1": 5860, "G2": 5770, "G3": 5720, "G4": 5680, "G5": 5660, "G6": 5600, "G7": 5550,
    "G8": 5480, "G9": 5380, "K0": 5270, "K1": 5170, "K2": 5100, "K3": 4830, "K4": 4600,
    "K5": 4440, "K6": 4300, "K7": 4100, "M0": 3850,
}


def teff_from_sptype(sptype: str | None, default: float = 6000.0) -> float:
    s = str(sptype or "").strip()
    m = re.match(r"^([ABFGKM])\s*(\d)?", s.upper())
    if not m:
        return float(default)
    letter, digit = m.group(1), m.group(2) or "5"
    return float(_TEFF_BY_TYPE.get(letter + digit, _TEFF_BY_TYPE.get(letter + "5", default)))


def _norm_token(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def normalise_name(name) -> str:
    """One spelling for ``HD 102647`` / ``HD102647`` / ``hd 102647.0`` and for
    Bayer names ``bet Leo`` / ``beta Leo`` / ``β Leo``; proper names are
    lower-cased with spaces squeezed."""
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    s = _norm_token(name)
    if not s or s in ("nan", "none", "--"):
        return ""
    m = re.match(r"^(hd|hip|hr|gj|gl|hic)\s*0*(\d+)(?:\.0)?\s*([a-z]?)$", s)
    if m:
        return f"{m.group(1)} {int(m.group(2))}{m.group(3)}".strip()
    if re.fullmatch(r"\d+(?:\.0)?", s):
        return f"hd {int(float(s))}"
    parts = s.split(" ")
    if len(parts) >= 2:
        g = parts[0]
        g2 = re.sub(r"\d+$", "", g)
        if g2 in GREEK:
            num = g[len(g2):]
            const = "".join(parts[1:])[:3]
            return f"{GREEK[g2]}{num} {const}"
        if g in ("v*", "*"):
            return normalise_name(" ".join(parts[1:]))
    return s


@dataclass
class TargetTable:
    """The targets asset plus the alias index it induces."""

    rows: pd.DataFrame
    alias: dict[str, str] = field(default_factory=dict)

    def key_for(self, name, id_kind: str | None = None) -> str | None:
        """The HD key for any spelling of the star, or ``None`` when unknown.
        ``id_kind`` (``hd`` / ``hip`` / ``name``) resolves bare integers."""
        s = normalise_name(name)
        if not s:
            return None
        if id_kind and re.fullmatch(r"hd \d+", s) and id_kind == "hip":
            s = "hip " + s.split(" ")[1]
        return self.alias.get(s)

    def teff(self, key: str) -> float:
        r = self.rows[self.rows["key"] == key]
        if not len(r):
            return 6000.0
        t = r.iloc[0].get("teff_k")
        try:
            t = float(t)
        except (TypeError, ValueError):
            t = float("nan")
        if math.isfinite(t) and t > 0:
            return t
        return teff_from_sptype(r.iloc[0].get("sptype"))

    def add(self, key: str, **fields) -> None:
        """A star the archive listed that the asset did not: append and index."""
        if key in set(self.rows["key"]):
            return
        row = {c: "" for c in self.rows.columns}
        row.update({"key": key, "hd": key.split(" ")[1] if key.startswith("HD ") else ""})
        row.update(fields)
        self.rows = pd.concat([self.rows, pd.DataFrame([row])], ignore_index=True)
        self._index_row(row)

    def _index_row(self, r) -> None:
        key = str(r["key"])
        for col in ("key", "name", "alt_names"):
            v = r.get(col, "")
            if isinstance(v, float) and math.isnan(v):
                continue
            for part in str(v).split("|"):
                s = normalise_name(part)
                if s:
                    self.alias.setdefault(s, key)
        for col, pref in (("hd", "hd"), ("hip", "hip")):
            v = r.get(col, "")
            try:
                n = int(float(v))
            except (TypeError, ValueError):
                continue
            self.alias.setdefault(f"{pref} {n}", key)


def load_targets(path: Path | None = None) -> TargetTable:
    p = Path(path) if path else ASSET_DIR / "forge_targets.csv"
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    tt = TargetTable(df)
    for _, r in df.iterrows():
        tt._index_row(r)
    return tt


def load_asset(name: str, path: Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else ASSET_DIR / f"forge_{name}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def _float(v, default=float("nan")) -> float:
    try:
        s = str(v).strip()
        if not s:
            return default
        return float(s)
    except (TypeError, ValueError):
        return default


def embedded_measurements(excess: pd.DataFrame, targets: TargetTable) -> dict[str, list[Measurement]]:
    """The embedded excess rows as :class:`Measurement` sets keyed by star.
    Rows without a numeric value are membership records and produce nothing."""
    out: dict[str, list[Measurement]] = {}
    for _, r in excess.iterrows():
        key = targets.key_for(r.get("key")) or (str(r.get("key")).strip() or None)
        if not key:
            continue
        val = _float(r.get("value_pct"))
        if not math.isfinite(val):
            continue
        err = _float(r.get("err_pct"), 0.0)
        kind = str(r.get("kind") or "meas").strip() or "meas"
        band = str(r.get("band") or "K").strip()
        wl = _float(r.get("wl_um"))
        if not math.isfinite(wl):
            wl = float(BAND_WAVELENGTH_UM.get(band, 2.2))
        verified = str(r.get("verify", "")).strip().lower() == "verified"
        out.setdefault(key, []).append(Measurement(
            band=band, wl_um=wl, value_pct=val, err_pct=err, kind=kind,
            instrument=str(r.get("instrument", "")), epoch=str(r.get("epoch", "")),
            source=str(r.get("source", "")), verified=verified, origin="embedded"))
    return out


def merge_measurements(embedded: dict[str, list[Measurement]],
                       archive: dict[str, list[Measurement]]) -> dict[str, list[Measurement]]:
    """Archive rows win over embedded rows for the same (survey, band, epoch);
    embedded rows with no archive counterpart are kept, unverified."""
    out: dict[str, list[Measurement]] = {}
    keys = set(embedded) | set(archive)
    for k in keys:
        got: dict[tuple, Measurement] = {}
        for m in embedded.get(k, []):
            got[(m.source, m.band, m.epoch)] = m
        for m in archive.get(k, []):
            got[(m.source, m.band, m.epoch)] = m
        out[k] = list(got.values())
    return out


def write_asset_with_verification(path: Path, rows: list[dict]) -> None:
    """Rewrite an asset with the runner's verification column filled in
    (the file stays a plain CSV, one row per measurement)."""
    if not rows:
        return
    cols = list(rows[0].keys())
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


__all__ = ["ASSET_DIR", "TargetTable", "embedded_measurements", "load_asset", "load_targets",
           "merge_measurements", "normalise_name", "teff_from_sptype",
           "write_asset_with_verification"]
