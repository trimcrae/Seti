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


#: A partition-function token: ``-1.2345``, ``---`` (missing) or ``NaN``.
_Q_TOKEN_RE = re.compile(r"^(?:[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?|-{2,}|[Nn][Aa][Nn])$")
_INT_TOKEN_RE = re.compile(r"^[-+]?\d+$")


def _q_value(tok: str) -> float:
    if re.fullmatch(r"-{2,}|[Nn][Aa][Nn]", tok):
        return float("nan")
    return float(tok)


def parse_catdir_report(text: str) -> tuple[list[Entry], list[str]]:
    """Parse JPL ``catdir.cat`` tolerantly; return ``(entries, unparsed lines)``.

    The published layout is ``tag  name  nlines  lg Q(300)…lg Q(9.375)  version``
    but *none* of those fields is fixed-width in practice:

    * the NAME may contain spaces, commas and punctuation
      (``H2O v2,2v2,v``, ``N-atom-D-st``, ``CH3OH, vt=0-2``);
    * the VERSION may be ``1``, ``2*``, ``1* `` (with trailing blanks) or
      absent altogether — the ``*`` is what made the old rigid regex drop 171
      of 403 lines on run 34787988880;
    * extra trailing columns may appear after the version.

    So the line is parsed by *field position from the right*: the trailing run
    of decimal-point (or ``---``/``NaN``) tokens is the Q grid, the integer
    immediately to its left is the line count, everything between the tag and
    that integer is the name, and whatever follows the Q grid is the version
    plus any extra columns.  A line that still will not parse is returned
    verbatim in the second element — counted, never silently dropped.
    """
    out: list[Entry] = []
    unparsed: list[str] = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        toks = line.split()
        if len(toks) < 3 or not _INT_TOKEN_RE.match(toks[0]):
            unparsed.append(line)
            continue
        # rightmost contiguous run of Q-like tokens
        end = None
        for i in range(len(toks) - 1, 0, -1):
            if _Q_TOKEN_RE.match(toks[i]):
                end = i
                break
        if end is None:
            unparsed.append(line)
            continue
        start = end
        while start - 1 >= 1 and _Q_TOKEN_RE.match(toks[start - 1]):
            start -= 1
        q_toks = toks[start:end + 1]
        if len(q_toks) > len(CATDIR_TEMPS):                    # extra leading Q columns: keep the grid
            q_toks = q_toks[-len(CATDIR_TEMPS):]
            start = end + 1 - len(q_toks)
        # the integer just left of the Q grid is the line count
        if start - 1 < 1 or not _INT_TOKEN_RE.match(toks[start - 1]):
            unparsed.append(line)
            continue
        name = " ".join(toks[1:start - 1]).strip()
        if not name:
            unparsed.append(line)
            continue
        try:
            q = [_q_value(t) for t in q_toks]
            entry = Entry(tag=int(toks[0]), name=name, nlines=int(toks[start - 1]),
                          temps=list(CATDIR_TEMPS[:len(q)]), qlog=q, database="jpl",
                          version=" ".join(toks[end + 1:]).strip())
        except ValueError:
            unparsed.append(line)
            continue
        out.append(entry)
    return out, unparsed


def parse_catdir(text: str) -> list[Entry]:
    """Entries of a JPL ``catdir.cat`` (see :func:`parse_catdir_report`)."""
    return parse_catdir_report(text)[0]


def count_unparsed_catdir(text: str) -> int:
    return len(parse_catdir_report(text)[1])


def unparsed_catdir_lines(text: str, limit: int = 20) -> list[str]:
    """The first ``limit`` lines of ``catdir.cat`` that would not parse, verbatim."""
    return [ln[:300] for ln in parse_catdir_report(text)[1][:int(limit)]]


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
# normalised-formula species matching
# ---------------------------------------------------------------------------
#: Element symbols, plus D and T kept distinct from H (an isotopologue of a
#: *target* must not silently become the target).
_ELEMENTS: tuple[str, ...] = tuple("""
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb
Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm D T
""".split())
_ELEMENT_SET = frozenset(_ELEMENTS)
_ELEMENT_BY_LOWER: dict[str, str] = {s.lower(): s for s in _ELEMENTS}

#: Vibrational / torsional state tags that JPL and CDMS append to a name.
_STATE_RE = re.compile(r"(?i)\b(?:v|vt|vib|nu|n)\s*\d*\s*=\s*[^\s,;]*")
_STATE_SUFFIX_RE = re.compile(r"(?i)[\s_-]+v[a-z]?\d*$")
_ISOTOPE_SUFFIX_RE = re.compile(r"-\d{1,3}$")
_PAREN_MASS_RE = re.compile(r"\(\s*\d{1,3}\s*\)")


def normalise_name(name: str) -> str:
    """Case-fold and drop everything that is not a letter or a digit.

    ``"CH3-35Cl"``, ``"CH3Cl, v=0"`` and ``"CH3CL"`` reduce to ``ch335cl``,
    ``ch3clv0`` and ``ch3cl``: comparable strings in which the target formula
    ``ch3cl`` is either the whole string, the string after a leading isotope
    mass, or a substring (a *near miss*, reported so the next run can be
    pointed at the catalogue's actual spelling).
    """
    return re.sub(r"[^a-z0-9]+", "", str(name or "").casefold())


_LEADING_MASS_RE = re.compile(r"^\d{1,3}")


def _strip_state(name: str) -> str:
    s = str(name or "")
    s = s.split(",")[0].split(";")[0]              # "CH3Cl, v=0" -> "CH3Cl"
    s = _STATE_RE.sub(" ", s)                      # "H2CO v=0"   -> "H2CO"
    s = _PAREN_MASS_RE.sub(" ", s)                 # "CH3(35)Cl"  -> "CH3 Cl"
    prev = None
    while prev != s:
        prev = s
        s = _STATE_SUFFIX_RE.sub("", s.strip())    # "CH2F2-v4"   -> "CH2F2"
        s = _ISOTOPE_SUFFIX_RE.sub("", s.strip())  # "C2H5CN-15"  -> "C2H5CN"
    return s.strip()


def _tokenise_formula(s: str) -> dict[str, int] | None:
    """Atom counts of a chemical formula, or ``None`` if it is not one.

    Backtracking so that both ``CO`` (carbon + oxygen) and ``Co`` (cobalt)
    read correctly and an all-caps ``CH3CL`` still resolves to ``Cl``; a digit
    run is a subscript when it follows an element and an isotope mass (ignored)
    when it does not.
    """
    counts: dict[str, int] = {}

    def walk(i: int, last: str | None) -> bool:
        if i >= len(s):
            return True
        ch = s[i]
        if ch == "|":
            return walk(i + 1, None)
        if ch.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            if last is not None:
                counts[last] += int(s[i:j]) - 1
                if walk(j, None):
                    return True
                counts[last] -= int(s[i:j]) - 1
                return False
            return walk(j, None)               # leading isotope mass: ignored
        two, one = s[i:i + 2], s[i:i + 1]
        # exact case first (so "CO" is carbon+oxygen and "Co" is cobalt), then
        # case-insensitively (so an all-caps "CH3CL" still resolves to Cl)
        cands = [(two, two if two in _ELEMENT_SET else None),
                 (one, one if one in _ELEMENT_SET else None),
                 (two, _ELEMENT_BY_LOWER.get(two.lower())),
                 (one, _ELEMENT_BY_LOWER.get(one.lower()))]
        seen: set[tuple[int, str]] = set()
        for cand, sym in cands:
            if sym is None or (len(cand), sym) in seen:
                continue
            seen.add((len(cand), sym))
            counts[sym] = counts.get(sym, 0) + 1
            if walk(i + len(cand), sym):
                return True
            counts[sym] -= 1
            if counts[sym] == 0:
                del counts[sym]
        return False

    return counts if s and walk(0, None) else None


def formula_key(name: str) -> str | None:
    """Canonical ``element+count`` key of a species name, or ``None``.

    Atom counting is what makes the catalogue's own spellings comparable to a
    plain formula: ``HCCCN`` and ``HC3N`` both give ``C3H1N1``, ``SiCC`` and
    ``SiC2`` both give ``C2Si1``, ``F2CO``/``CF2O``/``COF2`` all give
    ``C1F2O1``, and isotope masses (``CH3-35Cl``) and state tags
    (``CH3Cl, v=0``) drop out first.  Charge is kept, so ``CF+`` never
    matches ``CF``.
    """
    s = _strip_state(name)
    if not s:
        return None
    charge = ""
    m = re.search(r"([-+]+)\s*$", s)
    if m:
        charge = "+" if m.group(1)[0] == "+" else "-"
        s = s[:m.start()]
    # whitespace inside a formula is an artefact of HTML stripping
    # ("CH<sub>3</sub>OH" -> "CH 3 OH") and is closed up; other punctuation is
    # a real separator, so a digit after it is an isotope mass, not a subscript
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Za-z0-9]+", "|", s).strip("|")
    counts = _tokenise_formula(s)
    if not counts:
        return None
    return "".join(f"{k}{counts[k]}" for k in sorted(counts)) + charge


def species_match_route(cat_name: str, formula: str, rx=()) -> str | None:
    """Why (and whether) a catalogue entry name is the target ``formula``.

    Routes, in the order tried: ``normalised`` (the normalised catalogue name
    equals the normalised formula), ``isotopologue`` (equal after a leading
    isotope mass) and ``regex`` (a configured pattern).

    **``atom_counts`` is NOT a route, and removing it is a correctness fix.**
    Identical atom counts do not identify a molecule --- they identify an
    *empirical formula*, which isomers share. Probe run 35039720676 accepted,
    through that route alone:

    ======================  ==============================  ==============
    target                  accepted                        actually
    ======================  ==============================  ==============
    ``HCOOCH3``             ``HCOCH2OH`` (JPL 60006)        glycolaldehyde
    ``CH3CN``               ``CH3NC``    (JPL 41009)        methyl isocyanide
    ``HC3N``                ``HCCNC``, ``HNCCC``            two isomers
    ======================  ==============================  ==============

    Methyl formate and glycolaldehyde are both C2H4O2 and have entirely
    different rotational spectra. These entries' frequencies were being used to
    VETO unidentified lines *under the target's name*, so a veto could be
    justified by a molecule that is not the one named and may not even be
    present in the source. Over-vetoing discards exactly what this search is
    looking for, so the route is gone.

    Nothing legitimate is lost: ``config/uline.yaml`` already carries an
    explicit pattern for every real name variant (``^CH3OCHO`` for the JPL
    spelling of methyl formate, ``^HCCCN\\b``, ``^SiCC\\b``, ``^CCCCH\\b``).
    A same-formula entry is now recorded as a near miss with its reason, which
    is how the next run learns of an alias that ought to be configured.
    """
    cat_n, want_n = normalise_name(cat_name), normalise_name(formula)
    if want_n and cat_n == want_n:
        return "normalised"
    if want_n and _LEADING_MASS_RE.sub("", cat_n, count=1) == want_n != cat_n:
        return "isotopologue"
    if want_n and normalise_name(undecorate(cat_name)) == want_n:
        return "decorated"
    if any(r.search(str(cat_name)) for r in rx):
        return "regex"
    return None


#: An isotope mass written INSIDE a name: ``CH3-35Cl``, ``CH3(35)Cl``,
#: ``C-13-H3OH``.  Removed only where it sits immediately before an element
#: symbol, so the ``3`` of ``CH3`` is never touched.
_EMBEDDED_MASS_RE = re.compile(r"[-(]\s*\d{1,3}\s*\)?-?(?=[A-Z])")


def undecorate(name: str) -> str:
    """A catalogue name with its STATE and ISOTOPE decorations removed.

    ``"CH3-35Cl, v=0"`` -> ``"CH3Cl"``, ``"C-13-H3OH"`` -> ``"CH3OH"``,
    ``"CH2F2-v4"`` -> ``"CH2F2"``.  This is a transformation of the NAME, so it
    can never turn one molecule into another the way an atom-count key can:
    ``HCOCH2OH`` and ``HCOOCH3`` stay different strings, as they must.
    """
    return _EMBEDDED_MASS_RE.sub("", _strip_state(name))


def same_formula_not_matched(cat_name: str, formula: str) -> bool:
    """Same empirical formula, but not accepted as the species: an isomer, or an
    alias nobody has configured yet.  Recorded, never matched."""
    want_k = formula_key(formula)
    return bool(want_k and formula_key(cat_name) == want_k)


@dataclass
class SpeciesMatch:
    """Matched catalogue entries for one species, plus the near misses."""

    formula: str
    entries: list[Entry] = field(default_factory=list)
    matched_names: list[dict] = field(default_factory=list)
    near_miss_names: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"formula": self.formula, "matched_names": self.matched_names,
                "near_miss_names": self.near_miss_names}


def match_species(entries: list[Entry], formula: str, patterns=None, *,
                  near_miss_limit: int = 40) -> SpeciesMatch:
    """Entries that *are* ``formula``, and entries that merely mention it.

    A ``near_miss`` is an entry whose normalised name **contains** the
    normalised formula without matching it (``CF2Cl2`` for target ``CF2``,
    or a spelling the matcher still does not understand).  Recording them is
    the point: a species with no matches and a populated ``near_miss_names``
    tells the next run the catalogue's exact spelling instead of leaving it
    blind, which is what the first dispatch did.
    """
    rx = [re.compile(p) for p in (patterns or [])]
    want_n = normalise_name(formula)
    out = SpeciesMatch(formula=str(formula))
    for e in entries:
        route = species_match_route(e.name, formula, rx)
        if route is not None:
            out.entries.append(e)
            out.matched_names.append({"tag": int(e.tag), "name": e.name, "route": route})
            continue
        if len(out.near_miss_names) >= int(near_miss_limit):
            continue
        if same_formula_not_matched(e.name, formula):
            # An ISOMER, or an alias nobody has configured. Recorded with its
            # reason and NEVER matched: this is where HCOCH2OH (glycolaldehyde)
            # was being accepted as HCOOCH3 (methyl formate), and CH3NC as
            # CH3CN, and their frequencies were vetoing unidentified lines under
            # the target's name.
            out.near_miss_names.append(f"{e.name} [same formula {formula_key(formula)}: "
                                       "isomer or unconfigured alias, NOT matched]")
        elif want_n and want_n in normalise_name(e.name):
            out.near_miss_names.append(e.name)
    return out


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
           "LINE_COLUMNS", "MHZ_PER_CM", "SpeciesMatch", "count_unparsed_catdir",
           "find_species", "formula_key", "interp_log_q", "match_species", "normalise_name",
           "same_formula_not_matched",
           "parse_cat", "parse_catdir", "parse_catdir_report", "parse_partition_table",
           "rescale_lgint", "species_match_route", "symmetric_top_lines",
           "unparsed_catdir_lines"]
