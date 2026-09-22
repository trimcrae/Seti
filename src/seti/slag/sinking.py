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
#: PEWDD publishes the diffusion timescale of every element it tabulates, for
#: that star's own Teff and log g (``SinTimeCa`` / ``Sinking_time_Ca``).  When
#: a row carries them for the whole panel they are the timescales, per object.
SOURCE_ROW = "pewdd_row_timescales"

#: Dex left open when a row's own timescales are used.  It is not zero: the
#: tabulated timescales still depend on the convective-overshoot treatment
#: (PyllutedWD ships ov0 and ov1 grids that differ), on the adopted Teff and
#: log g, and on the opacity data.  0.05 dex is the spread between the two
#: overshoot prescriptions at fixed Teff in those grids.
SIGMA_ROW_DEX = 0.05

#: The relative timescales averaged over every PEWDD row that publishes them.
#: PEWDD tabulates tau_Z per star for ~95 objects; log10(tau_Z/tau_Ca) is
#: nearly constant across them (interquartile widths of 0.03--0.15 dex, no
#: significant Teff trend, H and He agreeing to ~0.01 dex), so those rows
#: calibrate the lever for the rows that lack it.  Built at run time by
#: :func:`relative_timescale_library`; ``SOURCE_LIBRARY`` records its use.
SOURCE_LIBRARY = "pewdd_relative_library"

#: Floor on a library element's scatter: no element is pinned better than the
#: overshoot prescription allows.
SIGMA_LIBRARY_FLOOR_DEX = 0.05
#: Minimum rows before an element enters the library at all.
LIBRARY_MIN_ROWS = 5

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
    #: {element: {"median": dex, "sigma": dex, "n": rows}} from the catalogue's
    #: own published timescales; empty until :func:`relative_timescale_library`
    #: has been run on the acquired table.
    library: dict = field(default_factory=dict)
    #: Mass-scaling exponent refitted to ``library`` (used for the elements the
    #: library is too thin to cover).  ``None`` keeps the literature default.
    library_beta: float | None = None
    #: The full library record (counts, per-atmosphere medians, the beta fit).
    library_meta: dict = field(default_factory=dict)

    def log_tau_rel(self, elements: list[str], atmosphere: str, teff: float | None = None,
                    logg: float | None = None, row_tau: dict | None = None
                    ) -> tuple[np.ndarray, np.ndarray, str]:
        """``(log10 tau_rel, sigma_dex, source_used)`` for ``elements``.

        ``row_tau`` is the object's own tabulated timescales (seconds, any
        common unit --- only ratios are used).  It is preferred over every
        other source when it covers the whole element list AND the reference,
        because it is the one source computed for this star's structure.
        """
        atm = "H" if str(atmosphere).upper().startswith("H") and str(atmosphere).upper() != "HE" \
            else "He"
        if row_tau and self.reference in row_tau and all(e in row_tau for e in elements):
            ref = float(row_tau[self.reference])
            if ref > 0 and all(float(row_tau[e]) > 0 for e in elements):
                vals = np.array([np.log10(float(row_tau[e]) / ref) for e in elements])
                sig = np.full(len(elements), float(SIGMA_ROW_DEX))
                sig[[e == self.reference for e in elements]] = 0.0
                return vals, sig, SOURCE_ROW
        tab = self.tables.get(atm)
        if tab is not None and teff is not None and np.isfinite(teff):
            vals = _interp_table(tab, elements, self.reference, float(teff), logg)
            if vals is not None:
                sig = np.full(len(elements), 0.05)
                return vals, sig, SOURCE_TABLE
        par = self.scaling[atm]
        A = np.array([self.fam.A[self.fam.index(e)] for e in elements], dtype=float)
        A_ref = self.fam.A[self.fam.index(self.reference)]
        beta = float(self.library_beta) if self.library_beta is not None else float(par["beta"])
        vals = -beta * np.log10(A / A_ref)
        sig = np.array([float(par["sigma_dex"]) if e in TABULATED
                        else float(par.get("sigma_dex_other", par["sigma_dex"]))
                        for e in elements])
        used_library = False
        if self.library:
            for k, e in enumerate(elements):
                rec = self.library.get(e)
                if rec and int(rec.get("n", 0)) >= LIBRARY_MIN_ROWS:
                    vals[k] = float(rec["median"])
                    sig[k] = max(float(rec["sigma"]), SIGMA_LIBRARY_FLOOR_DEX)
                    used_library = True
        sig[[e == self.reference for e in elements]] = 0.0
        return vals, sig, (SOURCE_LIBRARY if used_library else SOURCE_SCALING)


def relative_timescale_library(df, sinking_columns: dict, reference: str = "Ca",
                               atmosphere_column: str | None = None,
                               min_rows: int = LIBRARY_MIN_ROWS) -> dict:
    """log10(tau_Z / tau_ref) per element from every row that publishes both.

    The catalogue's own timescale columns are the only in-catalogue statement
    of the sinking lever.  This reduces them to one robust number and one
    robust scatter per element (median and 0.7413 x IQR), plus the
    mass-scaling exponent that best reproduces those medians --- which is what
    the elements with too few rows fall back on, so that the fallback is
    anchored to the same tables rather than to a literature guess.
    """
    ref_col = sinking_columns.get(reference)
    if df is None or not ref_col or ref_col not in getattr(df, "columns", []):
        return {"reference": reference, "elements": {}, "beta": None, "n_rows_with_reference": 0}
    ref = pd.to_numeric(df[ref_col], errors="coerce")
    ok_ref = np.isfinite(ref) & (ref > 0)
    out: dict = {}
    for el, col in sinking_columns.items():
        if el == reference or col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        m = ok_ref & np.isfinite(v) & (v > 0)
        if int(m.sum()) < min_rows:
            continue
        r = np.log10(v[m].to_numpy() / ref[m].to_numpy())
        q1, q3 = np.percentile(r, [25, 75])
        rec = {"median": float(np.median(r)), "sigma": float(max(0.7413 * (q3 - q1), 0.0)),
               "n": int(m.sum()), "p05": float(np.percentile(r, 5)),
               "p95": float(np.percentile(r, 95))}
        if atmosphere_column and atmosphere_column in df.columns:
            per = {}
            for atm in ("H", "He"):
                k = m & (df[atmosphere_column].astype(str).str.strip() == atm)
                if int(k.sum()) >= 3:
                    per[atm] = float(np.median(np.log10(v[k].to_numpy() / ref[k].to_numpy())))
            rec["per_atmosphere_median"] = per
        out[el] = rec
    return {"reference": reference, "elements": out, "beta": _fit_beta(out, reference),
            "n_rows_with_reference": int(ok_ref.sum())}


def _fit_beta(library: dict, reference: str) -> float | None:
    """The mass-scaling exponent that best reproduces the library's medians.

    ``log10(tau_Z/tau_ref) = -beta log10(A_Z/A_ref)``, least squares through
    the origin over the elements the library covers.  Atomic masses come from
    the natural-family asset via a small local table so the fit needs no
    family object.
    """
    xs, ys = [], []
    a_ref = _ATOMIC_MASS.get(reference)
    if not a_ref:
        return None
    for el, rec in library.items():
        a = _ATOMIC_MASS.get(el)
        if not a or int(rec.get("n", 0)) < LIBRARY_MIN_ROWS:
            continue
        xs.append(np.log10(a / a_ref))
        ys.append(float(rec["median"]))
    if len(xs) < 3:
        return None
    x = np.asarray(xs)
    y = np.asarray(ys)
    denom = float(np.sum(x * x))
    if denom <= 0:
        return None
    return float(-np.sum(x * y) / denom)


#: Atomic masses for the beta refit only (the family asset carries the set the
#: physics uses; this avoids a circular import at module scope).
_ATOMIC_MASS = {
    "H": 1.008, "He": 4.003, "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867,
    "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693,
    "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "Sr": 87.62, "Sn": 118.71,
    "Ba": 137.33, "Pb": 207.2,
}


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


def _parse_transposed(text: str) -> pd.DataFrame | None:
    """``T:,5000,5250,...`` / ``Ca:,...`` grids (PyllutedWD's ``data/timescales_*.csv``)."""
    rows: dict[str, list[float]] = {}
    grid: list[float] | None = None
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        cells = [c.strip() for c in s.split(",")]
        if len(cells) < 4:
            continue
        key = cells[0].rstrip(":").strip()
        if not key:
            continue
        try:
            vals = [float(c) for c in cells[1:] if c not in ("", "--")]
        except ValueError:
            continue
        if len(vals) < 3:
            continue
        if grid is None and re.fullmatch(r"(?i)t|teff|temp", key):
            grid = vals
            continue
        if re.fullmatch(r"[A-Z][a-z]?", key):
            rows[key] = vals
    if grid is None or len(rows) < 3:
        return None
    n = len(grid)
    out: dict[str, list[float]] = {"Teff": list(grid)}
    for el, vals in rows.items():
        if len(vals) != n:
            continue
        arr = np.asarray(vals, dtype=float)
        if np.nanmax(np.abs(arr)) < _LOG_VALUE_CEILING:     # already log10
            arr = 10.0 ** arr
        out[el] = arr.tolist()
    if len(out) < 4:
        return None
    return pd.DataFrame(out)


def describe_timescale_text(text: str, *, max_keys: int = 25) -> dict:
    """Why a timescale file did not parse, in terms of what it actually holds.

    A file that is fetched and then silently ignored is the worst kind of
    degradation: the run looks healthy and quietly uses a different timescale
    source.  This reports the row keys, the row widths and the temperature
    grid it found, which is enough to see the real layout from the committed
    JSON without refetching anything.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    keys, widths, grid_n = [], [], None
    n_numeric_rows = 0
    for ln in lines:
        if ln.startswith("#"):
            continue
        cells = [c.strip() for c in ln.split(",")]
        if len(cells) < 4:
            continue
        key = cells[0].rstrip(":").strip()
        try:
            vals = [float(c) for c in cells[1:] if c not in ("", "--")]
        except ValueError:
            keys.append(key + " (non-numeric)")
            continue
        n_numeric_rows += 1
        if key and re.fullmatch(r"(?i)t|teff|temp", key) and grid_n is None:
            grid_n = len(vals)
        keys.append(key)
        widths.append(len(vals))
    element_keys = [k for k in keys if re.fullmatch(r"[A-Z][a-z]?", k)]
    return {"n_lines": len(lines), "n_numeric_rows": n_numeric_rows,
            "row_keys": keys[:max_keys], "n_row_keys": len(keys),
            "element_like_keys": element_keys[:max_keys],
            "n_element_like_keys": len(element_keys),
            "temperature_grid_n": grid_n,
            "distinct_row_widths": sorted(set(widths))[:10],
            "reason": ("no T:/Teff row found" if grid_n is None else
                       "fewer than 3 element-named rows" if len(element_keys) < 3 else
                       "element rows do not match the temperature grid width"
                       if all(w != grid_n for w in widths) else
                       "recognised rows, but fewer than 3 matched the grid width")}


#: Above this an element's tabulated value is a timescale in seconds or years;
#: below it, it is already a base-10 logarithm.  Koester's grids run from
#: ~1e4 s to ~1e15 s, so no linear timescale is ever this small and no log one
#: is ever this large.
_LOG_VALUE_CEILING = 100.0


def parse_timescale_table(text: str) -> pd.DataFrame | None:
    """Best-effort parse of a timescale grid in either orientation.

    Two layouts are handled.  Column-per-element: a header naming ``Teff`` and
    at least three element symbols, rows one per model.  Row-per-element (the
    layout PyllutedWD actually ships, seen on run 35737893922): a first line
    ``T:,5000,5250,...`` giving the temperature grid, then one line per
    quantity, ``Ca:,<value per temperature>``.  Values that are base-10
    logarithms are exponentiated so the returned frame is always in linear
    timescale units --- only ratios are ever used, so the unit itself does not
    matter, but mixing logs and linears would.

    Returns ``None`` when neither layout is recognised; the caller records the
    head of the text so the next dispatch can see the real format.
    """
    tab = _parse_transposed(text)
    if tab is not None:
        return tab
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


__all__ = ["PHASE_DECLINING", "PHASE_EARLY", "PHASE_STEADY", "SCALING", "SIGMA_ROW_DEX",
           "SOURCE_ROW", "SOURCE_SCALING", "SOURCE_TABLE", "TABULATED", "TimescaleModel",
           "describe_timescale_text", "parse_timescale_table", "phase_grid",
           "phase_label", "phase_log_factor"]
