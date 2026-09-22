"""Photospheric sinking: the fourth lever.

Koester 2009 (A&A 498, 517) gives the photospheric mass of an element being
accreted at a constant rate into a convection zone from which it diffuses
with timescale tau_Z:

    X_Z(t) ∝ Mdot_Z · tau_Z · (1 − exp(−t/tau_Z))            while accreting,
    X_Z(t) ∝ X_Z(t_stop) · exp(−(t − t_stop)/tau_Z)             after it stops.

Three phases follow for a RATIO of two elements:

* **early** (t ≪ tau): the photosphere is the accreted parcel;
* **steady state** (t ≫ tau): the ratio is multiplied by tau_1/tau_2;
* **declining** (accretion over): the ratio is further multiplied by
  exp(−t_dec (1/tau_1 − 1/tau_2)), which can grow without bound.

Only RELATIVE timescales enter a ratio, and the overall photospheric level is
a free normalisation in every fit, so this module carries log10(tau_Z/tau_ref).

Where the relative timescales come from
---------------------------------------
The runner tries to fetch the real tables (Koester's web tables; the copies
PyllutedWD ships) --- ``acquire.discover_timescale_tables`` --- and when a
table parses, :class:`TimescaleModel` interpolates it in Teff.  Until then,
and offline, the model is the mass-scaling parametrisation
``log10(tau_Z/tau_ref) = −beta · log10(A_Z/A_ref)`` with beta = 0.45 (He
atmospheres) and 0.60 (H atmospheres) and a per-element scatter of 0.10 /
0.15 dex that every fit marginalises over.  ``source`` records which one was
used; nothing in a result reads as a measurement of the timescales.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .family import NaturalFamily

PHASE_EARLY = "early"
PHASE_STEADY = "steady_state"
PHASE_DECLINING = "declining"

SOURCE_SCALING = "embedded_mass_scaling"
SOURCE_TABLE = "fetched_table"

#: Mass-scaling exponents and their per-element scatter, by atmosphere.
#: ``sigma_dex`` is for the block Koester's tables cover well (``TABULATED``);
#: ``sigma_dex_other`` for the trace elements the tables extrapolate to.
#: The scatter matters most in the declining phase, where it is amplified by
#: t_dec/tau: a generous scatter there lets the natural model absorb any
#: single-element excess, so it is set to what the tables actually leave open.
SCALING = {
    "He": {"beta": 0.45, "sigma_dex": 0.05, "sigma_dex_other": 0.12},
    "H": {"beta": 0.60, "sigma_dex": 0.08, "sigma_dex_other": 0.15},
}
TABULATED = {"C", "N", "O", "Na", "Mg", "Al", "Si", "P", "S", "K", "Ca", "Ti", "Cr", "Mn",
             "Fe", "Ni"}

_LN10 = np.log(10.0)


@dataclass
class TimescaleModel:
    """log10(tau_Z / tau_ref) per element, per atmosphere, with an uncertainty."""

    fam: NaturalFamily
    reference: str = "Ca"
    source: str = SOURCE_SCALING
    tables: dict = field(default_factory=dict)   # {"H": DataFrame, "He": DataFrame}
    scaling: dict = field(default_factory=lambda: {k: dict(v) for k, v in SCALING.items()})

    def log_tau_rel(self, elements: list[str], atmosphere: str, teff: float | None = None,
                    logg: float | None = None) -> tuple[np.ndarray, np.ndarray, str]:
        """``(log10 tau_rel, sigma_dex, source_used)`` for ``elements``."""
        atm = "H" if str(atmosphere).upper().startswith("H") and str(atmosphere).upper() != "HE" \
            else "He"
        tab = self.tables.get(atm)
        if tab is not None and teff is not None and np.isfinite(teff):
            vals = _interp_table(tab, elements, self.reference, float(teff), logg)
            if vals is not None:
                sig = np.full(len(elements), 0.05)
                return vals, sig, SOURCE_TABLE
        par = self.scaling[atm]
        A = np.array([self.fam.A[self.fam.index(e)] for e in elements], dtype=float)
        A_ref = self.fam.A[self.fam.index(self.reference)]
        vals = -float(par["beta"]) * np.log10(A / A_ref)
        sig = np.array([float(par["sigma_dex"]) if e in TABULATED
                        else float(par.get("sigma_dex_other", par["sigma_dex"]))
                        for e in elements])
        sig[[e == self.reference for e in elements]] = 0.0
        return vals, sig, SOURCE_SCALING


def _interp_table(tab: pd.DataFrame, elements: list[str], reference: str, teff: float,
                  logg: float | None) -> np.ndarray | None:
    """Interpolate a parsed timescale table in Teff (nearest logg); ``None`` if it lacks an element."""
    cols = {c.strip(): c for c in tab.columns}
    if reference not in cols or "Teff" not in cols:
        return None
    if any(e not in cols for e in elements):
        return None
    t = tab.copy()
    if "logg" in cols and logg is not None and np.isfinite(logg):
        gs = t[cols["logg"]].astype(float)
        t = t[np.abs(gs - float(logg)) == np.abs(gs - float(logg)).min()]
    t = t.sort_values(cols["Teff"])
    x = t[cols["Teff"]].astype(float).to_numpy()
    if len(x) < 2:
        return None
    ref = np.log10(t[cols[reference]].astype(float).to_numpy())
    out = []
    for e in elements:
        y = np.log10(t[cols[e]].astype(float).to_numpy()) - ref
        out.append(float(np.interp(teff, x, y)))
    return np.array(out, dtype=float)


def parse_timescale_table(text: str) -> pd.DataFrame | None:
    """Best-effort parse of a whitespace/CSV table whose header names elements and Teff.

    Returns ``None`` unless the header carries a Teff column and at least
    three element symbols; the caller records the failure with the head of
    the text so the next dispatch can see the real format.
    """
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    for i, ln in enumerate(lines[:40]):
        raw = ln.lstrip("#").strip()
        sep = "," if raw.count(",") >= 3 else None
        cells = [c.strip() for c in (raw.split(sep) if sep else raw.split())]
        low = [c.lower() for c in cells]
        if not any(c.startswith("teff") or c in ("t", "temp") for c in low):
            continue
        els = [c for c in cells if re.fullmatch(r"[A-Z][a-z]?", c)]
        if len(els) < 3:
            continue
        rows = []
        for ln2 in lines[i + 1:]:
            if ln2.lstrip().startswith("#"):
                continue
            parts = [p.strip() for p in (ln2.split(sep) if sep else ln2.split())]
            if len(parts) != len(cells):
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=cells)
        ren = {}
        for c in df.columns:
            if c.lower().startswith("teff") or c.lower() in ("t", "temp"):
                ren[c] = "Teff"
            elif c.lower() in ("logg", "log g", "log_g", "g"):
                ren[c] = "logg"
        return df.rename(columns=ren)
    return None


# ---------------------------------------------------------------------------
# the three phases
# ---------------------------------------------------------------------------
def phase_log_factor(log_tau_rel: np.ndarray, t_acc: float, t_dec: float) -> np.ndarray:
    """log10 of the photospheric / accreted abundance ratio, per element, up to a constant.

    Times are in units of the reference element's timescale.  ``t_acc`` is
    how long accretion has been (or was) running; ``t_dec`` is how long ago it
    stopped (0 while it runs).  ``t_acc → 0`` recovers the early phase (an
    element-independent factor), ``t_acc → ∞`` the steady state
    (``+ log tau_rel``), and ``t_dec > 0`` the declining phase.
    """
    tau = 10.0 ** np.asarray(log_tau_rel, dtype=float)
    t_acc = max(float(t_acc), 1e-6)
    build = -np.expm1(-t_acc / tau)                     # 1 - exp(-t/tau), stable at small t
    out = np.log10(tau) + np.log10(build) - float(t_dec) / tau / _LN10
    return out - out.mean()


def phase_label(t_acc: float, t_dec: float, early_below: float = 0.3,
                steady_above: float = 3.0) -> str:
    if float(t_dec) > 0.05:
        return PHASE_DECLINING
    if float(t_acc) < early_below:
        return PHASE_EARLY
    if float(t_acc) >= steady_above:
        return PHASE_STEADY
    return "building"


def phase_grid(t_acc_range=(0.01, 30.0), t_dec_range=(0.0, 5.0), n_acc: int = 7,
               n_dec: int = 6) -> list[tuple[float, float]]:
    """A coarse (t_acc, t_dec) grid that covers all three phases."""
    accs = np.geomspace(t_acc_range[0], t_acc_range[1], n_acc)
    decs = np.linspace(t_dec_range[0], t_dec_range[1], n_dec)
    return [(float(a), float(d)) for a in accs for d in decs]


__all__ = ["PHASE_DECLINING", "PHASE_EARLY", "PHASE_STEADY", "SCALING", "SOURCE_SCALING",
           "SOURCE_TABLE", "TABULATED", "TimescaleModel", "parse_timescale_table", "phase_grid",
           "phase_label", "phase_log_factor"]
