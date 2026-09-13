"""Line-list physics for ULINE --- pure functions, offline-testable.

Three things live here and nothing else:

1. **Parsers for the JPL/CDMS ``.cat`` fixed format** and for the JPL
   ``catdir.cat`` species directory (CDMS publishes the same ``.cat`` format
   and a partition-function table with the same log-Q grid plus extra
   temperatures).
2. **Intensity rescaling from 300 K to an excitation temperature** using the
   catalogue's own partition function.  With ``LGINT`` the base-10 log of the
   integrated intensity at 300 K (nm² MHz), ``E_l`` the lower-state energy
   (cm⁻¹), ``E_u = E_l + ν/c`` and ``c₂ = hc/k = 1.438777 cm K`` (Pickett et
   al. 1998, JQSRT 60, 883, eq. 3 rearranged):

   .. math::

      I(T) = I(300)\\,\\frac{Q(300)}{Q(T)}\\,
             \\frac{e^{-c_2 E_l/T} - e^{-c_2 E_u/T}}
                  {e^{-c_2 E_l/300} - e^{-c_2 E_u/300}}

   ``log Q`` is interpolated linearly in ``log T`` between the tabulated grid
   points (300, 225, 150, 75, 37.5, 18.75, 9.375 K for JPL; CDMS adds 1000,
   500, 5 and 2.725 K) and extrapolated linearly at the ends.  The difference
   of exponentials is evaluated in log space so a line at ``E_l = 1000`` cm⁻¹
   at 10 K underflows to ``-inf`` rather than to a NaN.
3. **A symmetric-top predictor** for species with no catalogue entry:

   .. math::

      \\nu(J \\to J+1, K) = 2B(J+1) - 4D_J(J+1)^3 - 2D_{JK}(J+1)K^2

   with an approximate intensity: line strength ``S = ((J+1)² − K²)/(J+1)``
   (Gordy & Cook), K-degeneracy ``g_K = 1`` (K = 0) or 2, an optional
   nuclear-spin weight for ``K = 3n`` levels (C₃ᵥ tops with three equivalent
   I = ½ nuclei: 2:1), lower-state energy ``E(J,K) = B J(J+1) + (X − B) K²``
   where ``X`` is the axial constant (A for a prolate top, C for an oblate
   top; when unknown the K ladder is taken as Boltzmann-flat and the run says
   so), a partition function summed explicitly over the same levels, and the
   JPL intensity constant ``4.16231e-5 nm² MHz / (MHz D²)`` so the result is
   on the same (approximate) scale as a catalogue ``LGINT``.  Only the
   *relative* intensities matter downstream; the absolute scale of a
   predicted species is approximate at the factor-of-a-few level and is
   labelled ``predicted`` everywhere it appears.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: JPL catdir.cat partition-function temperature grid (K), in file order.
CATDIR_TEMPS: tuple[float, ...] = (300.0, 225.0, 150.0, 75.0, 37.5, 18.75, 9.375)
#: CDMS partition_function.html grid (K), in file order.
CDMS_TEMPS: tuple[float, ...] = (1000.0, 500.0, 300.0, 225.0, 150.0, 75.0, 37.5, 18.75,
                                 9.375, 5.0, 2.725)
#: Second radiation constant hc/k in cm K (CODATA 2018).
C2_CM_K = 1.438776877
#: MHz per cm⁻¹ (c in units of 10⁴ m s⁻¹ ... i.e. 29979.2458 MHz cm).
MHZ_PER_CM = 29979.2458
#: JPL intensity constant: I [nm² MHz] = 4.16231e-5 ν[MHz] S μ²[D²] (…)/Q.
JPL_INTENSITY_CONST = 4.16231e-5

LINE_COLUMNS = ("freq_mhz", "err_mhz", "lgint_300", "dr", "elo_cm", "gup", "tag", "lab",
                "qnfmt", "qn_up", "qn_lo")


# ---------------------------------------------------------------------------
# species directory / partition-function tables
# ---------------------------------------------------------------------------
@dataclass
class Entry:
    """One catalogue entry (a species tag) with its partition-function grid."""

    tag: int
    name: str
    nlines: int
    temps: list[float] = field(default_factory=lambda: list(CATDIR_TEMPS))
    qlog: list[float] = field(default_factory=list)
    database: str = "jpl"
    version: str = ""

    def as_dict(self) -> dict:
        return {"tag": int(self.tag), "name": self.name, "nlines": int(self.nlines),
                "temps": [float(t) for t in self.temps],
                "qlog": [None if not np.isfinite(q) else float(q) for q in self.qlog],
                "database": self.database, "version": self.version}


_FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
_CATDIR_RE = re.compile(
    rf"^\s*(\d+)\s+(.*?)\s+(\d+)\s+((?:{_FLOAT}\s+){{6}}{_FLOAT})(?:\s+(-?\d+))?\s*$")


def parse_catdir(text: str) -> list[Entry]:
    """Parse JPL ``catdir.cat``: ``tag name nlines Q(300)…Q(9.375) version``.

    Lines that do not fit the format are skipped and counted on the returned
    list's ``n_unparsed`` attribute (via :func:`catdir_report`).
    """
    out: list[Entry] = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        m = _CATDIR_RE.match(line)
        if not m:
            continue
        q = [float(x) for x in m.group(4).split()]
        out.append(Entry(tag=int(m.group(1)), name=m.group(2).strip(), nlines=int(m.group(3)),
                         temps=list(CATDIR_TEMPS), qlog=q, database="jpl",
                         version=(m.group(5) or "").strip()))
    return out


def count_unparsed_catdir(text: str) -> int:
    return sum(1 for line in (text or "").splitlines()
               if line.strip() and not _CATDIR_RE.match(line))


_QHDR_RE = re.compile(r"lg\s*\(\s*Q\s*\(\s*([0-9.]+)\s*\)\s*\)", re.IGNORECASE)
_TAG_RE = re.compile(r"^\s*(\d{4,6})\s+(.*\S)\s*$")


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", text or "")
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return (text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
            .replace("&gt;", ">"))


def parse_partition_table(text: str, *, temps=None, database: str = "cdms") -> list[Entry]:
    """Parse a CDMS-style partition-function table (HTML or plain text).

    The temperature grid is read from a header line containing ``lg(Q(T))``
    tokens; if none is present the CDMS default grid is assumed.  Each data
    line is ``tag name… nlines q q q …`` and is parsed from the right so a
    species name may contain spaces (``"CH3OH, vt=0-2"``).  Missing Q values
    (``---``) become NaN.
    """
    plain = _strip_html(text)
    grid = list(temps) if temps else None
    if grid is None:
        for line in plain.splitlines():
            hits = _QHDR_RE.findall(line)
            if len(hits) >= 3:
                grid = [float(h) for h in hits]
                break
    if grid is None:
        grid = list(CDMS_TEMPS)
    n_t = len(grid)
    out: list[Entry] = []
    for line in plain.splitlines():
        m = _TAG_RE.match(line)
        if not m:
            continue
        toks = m.group(2).split()
        if len(toks) < n_t + 2:
            continue
        qtoks = toks[-n_t:]
        try:
            q = [float("nan") if re.fullmatch(r"-+|NaN|nan", t) else float(t) for t in qtoks]
            nl = int(toks[-n_t - 1])
        except ValueError:
            continue
        name = " ".join(toks[:-n_t - 1]).strip()
        if not name:
            continue
        out.append(Entry(tag=int(m.group(1)), name=name, nlines=nl, temps=list(grid), qlog=q,
                         database=database))
    return out


def find_species(entries: list[Entry], patterns) -> list[Entry]:
    """Entries whose name matches any of the regexes (``re.search``)."""
    rx = [re.compile(p) for p in patterns]
    return [e for e in entries if any(r.search(e.name) for r in rx)]


# ---------------------------------------------------------------------------
# .cat line parser
# ---------------------------------------------------------------------------
def _gup(s: str) -> int:
    """GUP is I3 with a letter overflow: ``A05`` = 1005 (JPL convention)."""
    s = s.strip()
    if not s:
        return 0
    if s[0].isalpha():
        return (ord(s[0].upper()) - ord("A") + 10) * 100 + int(s[1:] or 0)
    return int(s)


def parse_cat(text: str, *, tag: int | None = None) -> pd.DataFrame:
    """Parse a JPL/CDMS ``.cat`` file (Pickett fixed format, 80 columns).

    Columns 1–13 FREQ (MHz), 14–21 ERR (MHz), 22–29 LGINT (log10 nm² MHz at
    300 K), 30–31 DR, 32–41 ELO (cm⁻¹), 42–44 GUP, 45–51 TAG (negative when
    the frequency is a laboratory measurement), 52–55 QNFMT, 56–67 upper QNs,
    68–79 lower QNs.  A whitespace-split fallback handles the occasional
    non-fixed-width line; anything else is skipped.
    """
    rows = []
    for line in (text or "").splitlines():
        if len(line.strip()) < 30:
            continue
        rec = None
        try:
            rec = (float(line[0:13]), float(line[13:21]), float(line[21:29]),
                   int(line[29:31]), float(line[31:41]), _gup(line[41:44]),
                   int(line[44:51]), int(line[51:55] or 0),
                   line[55:67].strip(), line[67:79].strip())
        except ValueError:
            toks = line.split()
            try:
                rec = (float(toks[0]), float(toks[1]), float(toks[2]), int(toks[3]),
                       float(toks[4]), _gup(toks[5]), int(toks[6]), int(toks[7]),
                       " ".join(toks[8:8 + 3]), " ".join(toks[11:14]))
            except (ValueError, IndexError):
                continue
        freq, err, lgint, dr, elo, gup, ltag, qnfmt, qu, ql = rec
        rows.append((freq, abs(err), lgint, dr, elo, gup, abs(ltag), ltag < 0, qnfmt, qu, ql))
    df = pd.DataFrame(rows, columns=list(LINE_COLUMNS))
    if tag is not None:
        df["tag"] = int(tag)
    return df


# ---------------------------------------------------------------------------
# intensity rescaling
# ---------------------------------------------------------------------------
def interp_log_q(temps, qlogs, t: float) -> float:
    """log10 Q(T): linear in log T between grid points, linear extrapolation."""
    temps = np.asarray(temps, dtype=float)
    q = np.asarray(qlogs, dtype=float)
    ok = np.isfinite(q) & np.isfinite(temps) & (temps > 0)
    if not ok.any():
        raise ValueError("partition function grid has no finite values")
    order = np.argsort(temps[ok])
    lt = np.log10(temps[ok][order])
    lq = q[ok][order]
    if lt.size == 1:
        return float(lq[0])
    x = float(np.log10(t))
    if x <= lt[0]:
        return float(lq[0] + (lq[1] - lq[0]) / (lt[1] - lt[0]) * (x - lt[0]))
    if x >= lt[-1]:
        return float(lq[-1] + (lq[-1] - lq[-2]) / (lt[-1] - lt[-2]) * (x - lt[-1]))
    return float(np.interp(x, lt, lq))


def _log10_boltzmann_difference(elo_cm, eup_cm, t: float) -> np.ndarray:
    """log10( exp(-c2 E_l/T) - exp(-c2 E_u/T) ), stable for large E/T."""
    a = C2_CM_K * np.asarray(elo_cm, dtype=float) / float(t)
    d = C2_CM_K * (np.asarray(eup_cm, dtype=float) - np.asarray(elo_cm, dtype=float)) / float(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (-a + np.log1p(-np.exp(-d))) / np.log(10.0)


def rescale_lgint(lgint_300, elo_cm, freq_mhz, temps, qlogs, t: float) -> np.ndarray:
    """LGINT at temperature ``t`` from LGINT at 300 K (formula in the module doc)."""
    el = np.asarray(elo_cm, dtype=float)
    eu = el + np.asarray(freq_mhz, dtype=float) / MHZ_PER_CM
    q300 = interp_log_q(temps, qlogs, 300.0)
    qt = interp_log_q(temps, qlogs, float(t))
    return (np.asarray(lgint_300, dtype=float) + q300 - qt
            + _log10_boltzmann_difference(el, eu, t) - _log10_boltzmann_difference(el, eu, 300.0))


# ---------------------------------------------------------------------------
# symmetric-top predictor
# ---------------------------------------------------------------------------
def symmetric_top_lines(b_mhz: float, dj_mhz: float = 0.0, djk_mhz: float = 0.0, *,
                        axial_mhz: float | None = None, mu_debye: float = 1.0,
                        k3_weight: float = 1.0, fmax_mhz: float = 2.0e6,
                        j_max: int | None = None, temps=CATDIR_TEMPS,
                        tag: int = 0, name: str = "predicted",
                        err_base_mhz: float = 0.5, err_rel: float = 1e-5
                        ) -> tuple[pd.DataFrame, Entry]:
    """R-branch (J→J+1, K) lines of a symmetric top in the ``.cat`` schema.

    Returns the line table (with extra ``j`` and ``k`` columns) and an
    :class:`Entry` whose ``qlog`` is the explicit level sum on the same
    ``temps`` grid, so :func:`rescale_lgint` treats it like a catalogue entry.
    ``err_mhz`` is ``err_base + err_rel·ν``, the caller's statement of how far
    the placeholder constants are trusted.
    """
    b = float(b_mhz)
    dj = float(dj_mhz or 0.0)
    djk = float(djk_mhz or 0.0)
    x = b if axial_mhz is None else float(axial_mhz)
    if j_max is None:
        j_max = int(fmax_mhz / (2.0 * b)) + 2
    j_max = int(max(j_max, 1))

    def energy_cm(j: int, k: int) -> float:
        return (b * j * (j + 1) + (x - b) * k * k) / MHZ_PER_CM

    def weight(k: int) -> float:
        gk = 1.0 if k == 0 else 2.0
        gns = float(k3_weight) if k % 3 == 0 else 1.0
        return gk * gns

    # Partition function: explicit sum over the same model levels.  Sum high
    # enough in J that the 300 K value has converged (E(J) ≫ kT).
    jq = max(j_max, int(np.sqrt(300.0 * MHZ_PER_CM / (C2_CM_K * b)) * 6) + 5)
    jj = np.arange(0, jq + 1)
    qlog = []
    for t in temps:
        tot = 0.0
        for j in jj:
            ks = np.arange(0, j + 1)
            w = np.where(ks == 0, 1.0, 2.0) * np.where(ks % 3 == 0, float(k3_weight), 1.0)
            e = (b * j * (j + 1) + (x - b) * ks * ks) / MHZ_PER_CM
            tot += (2 * j + 1) * float(np.sum(w * np.exp(-C2_CM_K * e / t)))
        qlog.append(float(np.log10(tot)))
    entry = Entry(tag=int(tag), name=name, nlines=0, temps=list(temps), qlog=qlog,
                  database="predicted")
    q300 = 10.0 ** interp_log_q(temps, qlog, 300.0)

    rows = []
    for j in range(0, j_max + 1):
        for k in range(0, j + 1):
            nu = 2.0 * b * (j + 1) - 4.0 * dj * (j + 1) ** 3 - 2.0 * djk * (j + 1) * k * k
            if nu <= 0 or nu > fmax_mhz:
                continue
            el = energy_cm(j, k)
            eu = el + nu / MHZ_PER_CM
            s = ((j + 1) ** 2 - k * k) / (j + 1)
            w = weight(k)
            pop = np.exp(-C2_CM_K * el / 300.0) - np.exp(-C2_CM_K * eu / 300.0)
            inten = JPL_INTENSITY_CONST * nu * s * mu_debye ** 2 * w * pop / q300
            lg = float(np.log10(inten)) if inten > 0 else -np.inf
            gup = (2 * (j + 1) + 1) * w
            rows.append((nu, err_base_mhz + err_rel * nu, lg, 3, el, int(round(gup)),
                         int(tag), False, 2, f"{j + 1:2d}{k:2d}", f"{j:2d}{k:2d}", j, k))
    df = pd.DataFrame(rows, columns=list(LINE_COLUMNS) + ["j", "k"])
    entry.nlines = int(len(df))
    return df, entry


__all__ = ["C2_CM_K", "CATDIR_TEMPS", "CDMS_TEMPS", "Entry", "JPL_INTENSITY_CONST",
           "LINE_COLUMNS", "MHZ_PER_CM", "count_unparsed_catdir", "find_species",
           "interp_log_q", "parse_cat", "parse_catdir", "parse_partition_table",
           "rescale_lgint", "symmetric_top_lines"]
