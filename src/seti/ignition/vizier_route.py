"""IGNITION's SECOND route to the parent sample: VizieR's non-TAP ASU interface.

Why this module exists
----------------------
``results/ignition/probe.json`` (generated 2026-09-14T00:22Z, committed on
``main``) reads ``verdict NO_DATA_REACHED``, ``gaia_shape_working: null`` and

    inner_cone             TIMED_OUT  QueryTimeout('no answer within 480 s')
    inner_cone_postfilter  TIMED_OUT  QueryTimeout('no answer within 480 s')
    flat                   TIMED_OUT  QueryTimeout('no answer within 420 s')

with the run before it recording ``Error 500: canceling statement due to
statement timeout`` and ``Error 503: Server overloaded: synchronous job
rejected due to maximum number of synchronous queued jobs (150) reached``.
All three query shapes and both queues are blocked **at the ESA archive
itself**, so no shape change and no queue change can reach the parent sample.
A channel with no parent sample has nothing downstream to run.

The transport that demonstrably worked the same night is **VizieR's non-TAP
ASU interface** (``https://vizier.cds.unistra.fr/viz-bin/asu-tsv``), which
served ULINE's IRC+10216 line list while TAPVizieR *itself* was answering 503.
It is a plain HTTP GET returning tab-separated text: no TAP queue, no ADQL
planner, no statement timeout.  This module builds the SAME parent selection
over that transport.

What is identical to the ESA route, and what is not
---------------------------------------------------
Identical (this is the whole point --- only the transport differs):

* the Gaia cuts.  Every numeric cut that ASU can express is sent as a column
  constraint (``Gmag=<14.5``, ``Plx=>3``, ``RPlx=>10``, ``RUWE=<1.4``,
  ``BP-RP=0.6..2.5``); everything else --- ``|b| > 15``, the absolute-magnitude
  dwarf cut, the ``phot_variable_flag`` cut --- is applied by
  :func:`seti.ignition.sample.select_parent`, which the ESA route already runs
  over every frame it pulls.  A server-side constraint here is therefore an
  *optimisation*, never the only place a cut lives;
* the AllWISE cuts.  :func:`apply_allwise_predicates` is the pandas spelling of
  :func:`seti.ignition.sample.allwise_predicates` --- the photospheric 2010
  colour ``-0.1 < W1-W2 < 0.15``, ``ext_flag = 0`` and ``cc_flags = '0000'`` ---
  and it is applied to every chunk this route returns, so the rows handed back
  have passed exactly the predicates the ``inner_cone`` shape puts in SQL;
* the column set.  :func:`to_parent_frame` emits exactly
  :func:`seti.ignition.sample.parent_columns`, in that order, so the two routes
  are indistinguishable to every stage downstream of ``sample``.

Not identical, and recorded as such:

* **the cross-match**.  The ESA route uses the archive's own
  ``gaiadr3.allwise_best_neighbour``.  There is no ASU equivalent, so this
  route does a positional match of the chunk's Gaia rows against
  ``II/328/allwise``, propagating Gaia's 2016.0 positions back to the AllWISE
  mean epoch with the channel's own :func:`propagate_pm`.  ``allwise_sep_arcsec``
  and ``allwise_n_neighbours`` are measured from that match rather than read
  from the archive's table;
* **``-out.max``**.  ASU has no ``COUNT(*)``; it has a row cap.  A chunk that
  comes back with exactly ``-out.max`` rows is recorded ``capped: true`` and its
  row count is reported as *what was obtained*, never as a complete selection;
* **allsky mode**.  A parallax shell is not a sky chunk, so there is no cone to
  fetch the AllWISE side over.  Rather than fabricate a match, a shell unit is
  recorded ``NOT_SUPPORTED_FOR_MODE`` and contributes nothing.

Every catalogue id, column name and ASU parameter spelling below is
**asserted and unverified** (``config/ignition.yaml`` -> ``sample.vizier``
carries the same note).  A wrong id or a missing column is a recorded status
--- ``CATALOGUE_NOT_FOUND``, ``COLUMNS_UNRESOLVED`` --- not a crash and never a
fabricated row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..metronome.acquire import (
    VIZIER_ASU,
    VIZIER_ASU_MIRRORS,
    VizierRouteError,
    asu_constraint_ladder,
    asu_position_ladder,
    asu_position_spellings,
    asu_rows,
    asu_url,
    ladder_verdict,
)
from ..shroud.classify import galactic_latitude
from ..vigil.acquire import GAIA_EPOCH, propagate_pm
from .sample import (  # ROUTE_* live there so this module can import them without a cycle
    DEFAULT_SAMPLE,
    ROUTE_ESA,
    ROUTE_VIZIER,
    parent_columns,
)

#: AllWISE source positions are quoted at the survey's mean epoch, ~2010.5
#: (verify: the exact mean epoch is per-source; 2010.5 is the catalogue mean and
#: costs at most ~0.1 arcsec of match radius for a 100 mas/yr star).
ALLWISE_EPOCH = 2010.5

STATUS_OK = "OK"
STATUS_ZERO = "QUERY_RETURNED_ZERO_ROWS"
STATUS_FAILED = "QUERY_FAILED"
STATUS_NOT_FOUND = "CATALOGUE_NOT_FOUND"
STATUS_COLUMNS = "COLUMNS_UNRESOLVED"
STATUS_UNSUPPORTED = "NOT_SUPPORTED_FOR_MODE"
STATUS_DISABLED = "DISABLED"

#: Gaia columns this route cannot do without: every one of them either carries a
#: cut or is needed by the NEOWISE acquisition downstream.
REQUIRED_GAIA = ("source_id", "ra", "dec", "parallax", "parallax_over_error",
                 "pmra", "pmdec", "ruwe", "phot_g_mean_mag", "bp_rp")
#: AllWISE columns this route cannot do without (the 2010 colour).
REQUIRED_ALLWISE = ("allwise_designation", "ra", "dec", "w1mpro", "w2mpro")
#: Columns that CARRY A CUT but have a named, recorded consequence when absent:
#: the cut is not applied, the unit is degraded, and the record says which.
CUT_GAIA = {"phot_variable_flag": "gaia_variable"}
CUT_ALLWISE = {"ext_flag": "ext_flag", "ph_qual": "ph_qual", "cc_flags": "cc_flags"}

DEFAULT_VIZIER: dict = {
    "enabled": True,
    # --- asserted, UNVERIFIED (no network in the sandbox; the probe checks) ---
    "gaia_catalogue": "I/355/gaiadr3",
    "gaia_params_catalogue": "I/355/paramp",
    "allwise_catalogue": "II/328/allwise",
    # --- ASU request shape ---
    "out_max": 150000,             # the row cap; reported, never called complete
    "allwise_out_max": 150000,
    "out_add": [],                 # extra computed columns, e.g. ["_r"]
    "server_side_constraints": True,
    "cone_pad_arcmin": 1.0,        # AllWISE cone is the Gaia cone plus this pad
    # Which spelling of the cone VizieR is sent (asu_position_spellings). The
    # probe's position ladder MEASURES which one the service honours; this is
    # the one used until a ladder says otherwise.
    "cone_spelling": "decimal_signed",
    "match_radius_arcsec": 3.0,    # Gaia(2016 -> 2010.5) against AllWISE
    "timeout_s": 300.0,
    "bases": [],                   # empty -> metronome's VIZIER_ASU_MIRRORS
    # --- column names in the VizieR mirrors, first match wins (UNVERIFIED) ---
    "gaia_columns": {
        "source_id": ["Source", "DR3Name", "GaiaDR3"],
        "ra": ["RA_ICRS", "RAJ2000", "_RAJ2000"],
        "dec": ["DE_ICRS", "DEJ2000", "_DEJ2000"],
        "parallax": ["Plx"],
        "parallax_over_error": ["RPlx"],
        "pmra": ["pmRA"],
        "pmdec": ["pmDE"],
        "ruwe": ["RUWE"],
        "phot_g_mean_mag": ["Gmag"],
        "bp_rp": ["BP-RP", "BPRP"],
        "phot_variable_flag": ["VarFlag", "Var"],
        "non_single_star": ["NSS"],
        "teff_gspphot": ["Teff"],
        "random_index": ["RandomI", "Rand"],
    },
    "allwise_columns": {
        "allwise_designation": ["AllWISE", "AllWISEid", "designation"],
        "ra": ["RAJ2000", "RA_ICRS", "_RAJ2000"],
        "dec": ["DEJ2000", "DE_ICRS", "_DEJ2000"],
        "w1mpro": ["W1mag", "w1mpro"],
        "w1mpro_error": ["e_W1mag", "w1sigmpro"],
        "w2mpro": ["W2mag", "w2mpro"],
        "w2mpro_error": ["e_W2mag", "w2sigmpro"],
        "w3mpro": ["W3mag", "w3mpro"],
        "w3mpro_error": ["e_W3mag", "w3sigmpro"],
        "cc_flags": ["ccf", "cc_flags"],
        "ph_qual": ["qph", "ph_qual"],
        "ext_flag": ["ex", "ext_flg", "ext_flag"],
    },
    # Which Gaia cut goes to the server as an ASU column constraint.  Each is an
    # OPTIMISATION: select_parent re-applies all of them locally, so a wrong
    # spelling costs rows that are re-fetched, never a silently different cut.
    "constraint_columns": {
        "phot_g_mean_mag": "Gmag",
        "parallax": "Plx",
        "parallax_over_error": "RPlx",
        "ruwe": "RUWE",
        "bp_rp": "BP-RP",
    },
}


def vizier_conf(conf: dict | None = None) -> dict:
    """``sample.vizier`` merged over :data:`DEFAULT_VIZIER` (one level deep)."""
    got = dict((conf or {}).get("vizier") or {})
    out = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
           for k, v in DEFAULT_VIZIER.items()}
    for k, v in got.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def enabled(conf: dict | None = None) -> bool:
    return bool(vizier_conf(conf).get("enabled", True))


def bases_for(conf: dict | None = None) -> tuple[str, ...]:
    b = [str(x) for x in (vizier_conf(conf).get("bases") or []) if str(x).strip()]
    return tuple(b) if b else tuple(VIZIER_ASU_MIRRORS)


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------
def _fmt(x: float) -> str:
    """A number as ASU wants to read it (no exponent, no trailing noise)."""
    v = float(x)
    return str(int(v)) if v == int(v) else f"{v:g}"


def cone_constraints(conf: dict | None = None, field: dict | None = None, *,
                     pad_deg: float = 0.0) -> dict:
    """The ``-c`` / radius / equinox trio for one cone, in the configured spelling.

    ``sample.vizier.cone_spelling`` picks one of
    :func:`seti.metronome.acquire.asu_position_spellings`; the default is
    ``decimal_signed``.  The spelling is not cosmetic: ``-c=<ra>+<dec>`` put a
    literal plus on the wire, which a query string decodes to a SPACE, so
    VizieR received the unsigned dotless pair ``266 65``, failed to read it as
    a position and answered with an empty resource for BOTH ``I/355/gaiadr3``
    and ``II/328/allwise`` (probe run 34799195807).  ``probe_route`` runs the
    whole spelling ladder when a cone returns nothing, so the record says which
    spelling the service actually honours rather than asserting one.
    """
    if field is None:
        return {}
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    want = str(vizier_conf(c).get("cone_spelling") or "decimal_signed")
    radius = float(field["radius_deg"]) + float(pad_deg or 0.0)
    spellings = dict(asu_position_spellings(float(field["ra"]), float(field["dec"]), radius))
    return dict(spellings.get(want) or next(iter(spellings.values())))


def asu_constraints(conf: dict | None = None, *, field: dict | None = None,
                    plx_lo: float | None = None, plx_hi: float | None = None) -> dict:
    """The ASU ``<col>=<constraint>`` pairs for one chunk of the sweep.

    The chunking is the sample stage's own: a cone for ``mode=fields``
    (``-c=<ra> <dec>`` with ``-c.rd=<deg>``), a parallax shell for
    ``mode=allsky`` (``Plx=lo..hi``).  ``random_index`` slicing has no ASU
    spelling (there is no MOD), so a shard of a chunk is taken locally.
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    vz = vizier_conf(c)
    cols = dict(vz.get("constraint_columns") or {})
    out: dict[str, str] = {}
    if field is not None:
        # The cone spelling is the shared, MEASURED one (see
        # asu_position_spellings): a decimal pair with an explicit declination
        # sign. The old "<ra>+<dec>" form reached VizieR as "266 65" — the
        # literal plus decodes to a space — and it answered with an empty
        # resource because it could not read that as a position at all.
        out.update(cone_constraints(c, field))
    if not vz.get("server_side_constraints", True):
        return out
    if cols.get("phot_g_mean_mag"):
        out[cols["phot_g_mean_mag"]] = f"<{_fmt(c['g_max'])}"
    if cols.get("parallax"):
        if plx_lo is not None and plx_hi is not None:
            out[cols["parallax"]] = f"{_fmt(plx_lo)}..{_fmt(plx_hi)}"
        else:
            out[cols["parallax"]] = f">{_fmt(c['parallax_min_mas'])}"
    if cols.get("parallax_over_error"):
        out[cols["parallax_over_error"]] = f">{_fmt(c['parallax_over_error_min'])}"
    if cols.get("ruwe"):
        out[cols["ruwe"]] = f"<{_fmt(c['ruwe_max'])}"
    if cols.get("bp_rp"):
        out[cols["bp_rp"]] = f"{_fmt(c['bp_rp_min'])}..{_fmt(c['bp_rp_max'])}"
    return out


def _wanted(mapping: dict) -> list[str]:
    """The first candidate spelling of every logical column, request order."""
    out: list[str] = []
    for cands in mapping.values():
        for cnd in (cands if isinstance(cands, (list, tuple)) else [cands]):
            if str(cnd).strip():
                out.append(str(cnd))
                break
    return list(dict.fromkeys(out))


def gaia_asu_url(conf: dict | None = None, *, field: dict | None = None,
                 plx_lo: float | None = None, plx_hi: float | None = None,
                 base: str = VIZIER_ASU, max_rows: int | None = None) -> str:
    """The ASU URL for one chunk of the Gaia mirror."""
    vz = vizier_conf({**DEFAULT_SAMPLE, **(conf or {})})
    cons = asu_constraints(conf, field=field, plx_lo=plx_lo, plx_hi=plx_hi)
    add = [str(a) for a in (vz.get("out_add") or []) if str(a).strip()]
    if add:
        cons = {"-out.add": ",".join(add), **cons}
    return asu_url(str(vz["gaia_catalogue"]), base=base,
                   columns=_wanted(vz["gaia_columns"]),
                   max_rows=int(max_rows or vz["out_max"]), constraints=cons)


def allwise_asu_url(conf: dict | None = None, *, field: dict, base: str = VIZIER_ASU,
                    max_rows: int | None = None) -> str:
    """The ASU URL for the AllWISE side of one chunk (the cone, padded)."""
    vz = vizier_conf({**DEFAULT_SAMPLE, **(conf or {})})
    pad = float(vz.get("cone_pad_arcmin") or 0.0) / 60.0
    cons = cone_constraints({**DEFAULT_SAMPLE, **(conf or {})}, field, pad_deg=pad)
    return asu_url(str(vz["allwise_catalogue"]), base=base,
                   columns=_wanted(vz["allwise_columns"]),
                   max_rows=int(max_rows or vz["allwise_out_max"]), constraints=cons)


# ---------------------------------------------------------------------------
# The response
# ---------------------------------------------------------------------------
def resolve_columns(columns, mapping: dict) -> tuple[dict, list[str]]:
    """``(logical -> actual)`` for the columns that are there, and what is not.

    Case-insensitive, because VizieR's TSV header capitalises as the ReadMe
    does and the ReadMe is not the archive.
    """
    have = {str(c).strip().lower(): str(c) for c in columns}
    got: dict[str, str] = {}
    missing: list[str] = []
    for logical, cands in mapping.items():
        for cnd in (cands if isinstance(cands, (list, tuple)) else [cands]):
            key = str(cnd).strip().lower()
            if key in have:
                got[logical] = have[key]
                break
        else:
            missing.append(logical)
    return got, missing


def _norm_flags(series: pd.Series, width: int = 4) -> pd.Series:
    """``cc_flags`` as a string, whatever the TSV parser made of it.

    ``parse_asu_tsv`` converts a column to numbers when every value parses, so
    the AllWISE ``ccf`` column of a clean chunk arrives as the integer ``0``,
    not the string ``'0000'``.  Comparing that to ``'0000'`` would reject every
    clean source --- the exact opposite of the archive's ``cc_flags = '0000'``.
    """
    s = series.copy()
    num = pd.to_numeric(s, errors="coerce")
    out = s.astype(str).str.strip()
    fix = num.notna()
    if fix.any():
        out.loc[fix] = num[fix].astype("int64").astype(str).str.zfill(width)
    return out


def apply_allwise_predicates(df: pd.DataFrame, conf: dict | None = None,
                             *, applied: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """The pandas spelling of :func:`seti.ignition.sample.allwise_predicates`.

    The ESA route puts these four in SQL (the ``inner_cone`` and ``flat``
    shapes); ``select_parent`` re-applies the colour, ``ext_flag`` and
    ``ph_qual`` but NOT ``cc_flags``, so this route applies them here and the
    two routes end up with the same rows.  ``applied`` names which columns were
    actually resolved; a cut whose column is missing is reported in
    ``counters["cuts_not_applied"]`` rather than silently skipped.
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    have = set(applied or {k: k for k in df.columns})
    counters: dict = {"n_in": int(len(df)), "cuts_not_applied": []}
    if not len(df):
        counters["n_out"] = 0
        return df, counters
    d = df.reset_index(drop=True)
    alive = pd.Series(True, index=d.index)

    def _cut(name: str, keep: pd.Series) -> None:
        nonlocal alive
        keep = keep.fillna(False).astype(bool)
        counters[f"cut_{name}"] = int((alive & ~keep).sum())
        alive &= keep

    w1 = pd.to_numeric(d.get("w1mpro"), errors="coerce")
    w2 = pd.to_numeric(d.get("w2mpro"), errors="coerce")
    col = w1 - w2
    _cut("w1w2_photospheric",
         (col < float(c["w1w2_max"])) & (col > float(c["w1w2_min"])))
    if "ext_flag" in have and "ext_flag" in d:
        _cut("ext_flag", pd.to_numeric(d["ext_flag"], errors="coerce").fillna(0) == 0)
    else:
        counters["cuts_not_applied"].append("ext_flag")
    if "cc_flags" in have and "cc_flags" in d:
        _cut("cc_flags", _norm_flags(d["cc_flags"]) == "0000")
    else:
        counters["cuts_not_applied"].append("cc_flags")
    out = d[alive].reset_index(drop=True)
    counters["n_out"] = int(len(out))
    return out, counters


def to_parent_frame(gaia: pd.DataFrame, conf: dict | None = None) -> pd.DataFrame:
    """Reindex onto exactly :func:`seti.ignition.sample.parent_columns`, in order.

    Columns the mirror does not carry become NaN; nothing is invented.  ``b`` is
    computed from the mirror's own RA/Dec (IAU 1958 pole) rather than requested,
    because the ESA route selects on ``gaia_source.b`` and a computed latitude is
    exact where an unverified ``-out.add=_Glat`` spelling is not.
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    want = parent_columns(c)
    d = gaia.copy()
    if "b" not in d and {"ra", "dec"} <= set(d.columns):
        d["b"] = galactic_latitude(pd.to_numeric(d["ra"], errors="coerce"),
                                   pd.to_numeric(d["dec"], errors="coerce"))
    for col in want:
        if col not in d:
            d[col] = np.nan
    return d[want]


# ---------------------------------------------------------------------------
# The cross-match (no allwise_best_neighbour outside the archive)
# ---------------------------------------------------------------------------
def _unit_vectors(ra, dec) -> np.ndarray:
    r = np.radians(np.asarray(ra, dtype=float))
    d = np.radians(np.asarray(dec, dtype=float))
    return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


def match_allwise(gaia: pd.DataFrame, wise: pd.DataFrame, conf: dict | None = None
                  ) -> tuple[pd.DataFrame, dict]:
    """Positional Gaia x AllWISE match, PM-propagated 2016.0 -> 2010.5.

    The channel already propagates proper motion to the survey epoch before it
    matches (``acquire.positions_at_neowise_epoch``); this is the same operation
    against a different epoch.  Nearest neighbour inside
    ``match_radius_arcsec`` wins, ``allwise_n_neighbours`` counts every AllWISE
    source inside that radius (so a blend is visible downstream exactly as the
    archive's own neighbour table makes it visible), and an unmatched Gaia row
    is DROPPED --- the archive's inner join drops it too.
    """
    from scipy.spatial import cKDTree

    vz = vizier_conf({**DEFAULT_SAMPLE, **(conf or {})})
    tol = float(vz["match_radius_arcsec"])
    rec = {"n_gaia": int(len(gaia)), "n_allwise": int(len(wise)),
           "match_radius_arcsec": tol, "from_epoch": GAIA_EPOCH, "to_epoch": ALLWISE_EPOCH}
    wise_cols = [c for c in wise.columns if c not in ("ra", "dec")]
    if not len(gaia) or not len(wise):
        rec["n_matched"] = 0
        return gaia.iloc[0:0].assign(**{c: pd.Series(dtype="object") for c in wise_cols}), rec
    pmra = pd.to_numeric(gaia.get("pmra"), errors="coerce").fillna(0.0).to_numpy(float)
    pmdec = pd.to_numeric(gaia.get("pmdec"), errors="coerce").fillna(0.0).to_numpy(float)
    ra_m, dec_m = propagate_pm(pd.to_numeric(gaia["ra"], errors="coerce").to_numpy(float),
                               pd.to_numeric(gaia["dec"], errors="coerce").to_numpy(float),
                               pmra, pmdec, GAIA_EPOCH, ALLWISE_EPOCH)
    tree = cKDTree(_unit_vectors(pd.to_numeric(wise["ra"], errors="coerce"),
                                 pd.to_numeric(wise["dec"], errors="coerce")))
    pts = _unit_vectors(ra_m, dec_m)
    chord = 2.0 * np.sin(np.radians(tol / 3600.0) / 2.0)
    dist, idx = tree.query(pts, k=1, distance_upper_bound=chord)
    hit = np.isfinite(dist) & (idx < len(wise))
    n_near = np.array([len(tree.query_ball_point(p, chord)) for p in pts], dtype=int)
    sep = np.degrees(2.0 * np.arcsin(np.clip(np.where(hit, dist, np.nan) / 2.0, 0.0, 1.0))) * 3600.0
    out = gaia.reset_index(drop=True).loc[hit].reset_index(drop=True)
    picked = wise.reset_index(drop=True).iloc[idx[hit]].reset_index(drop=True)
    for col in wise_cols:
        out[col] = picked[col].to_numpy()
    out["allwise_sep_arcsec"] = sep[hit]
    out["allwise_n_neighbours"] = n_near[hit]
    rec["n_matched"] = int(len(out))
    rec["n_unmatched"] = int(rec["n_gaia"] - rec["n_matched"])
    return out, rec


# ---------------------------------------------------------------------------
# One chunk, end to end
# ---------------------------------------------------------------------------
def _classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    for marker in ("not found", "no such", "unknown catalog", "unknown table",
                   "****", "no known catalog"):
        if marker in text:
            return STATUS_NOT_FOUND
    return STATUS_FAILED


def _fetch(catalogue: str, conf: dict, *, fetch_fn, max_rows: int,
           columns: list[str], constraints: dict) -> tuple[pd.DataFrame, list[dict]]:
    """One ASU pull through the shared helper, mirror by mirror.

    ``asu_rows``/``parse_asu_tsv`` are metronome's --- the parser that served
    ULINE's table tonight --- so this route inherits its error-line handling
    (an error page must never read as zero rows) rather than re-deriving it.
    """
    return asu_rows(catalogue, columns=columns, max_rows=int(max_rows), fetch_fn=fetch_fn,
                    bases=list(bases_for(conf)), constraints=constraints)


def fetch_gaia_chunk(conf: dict | None = None, *, field: dict | None = None,
                     plx_lo: float | None = None, plx_hi: float | None = None,
                     cap: int | None = None, fetch_fn=None) -> tuple[pd.DataFrame, dict]:
    """One chunk of the Gaia mirror over ASU, mapped onto the ESA column names."""
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    vz = vizier_conf(c)
    max_rows = int(cap or vz["out_max"])
    cons = asu_constraints(c, field=field, plx_lo=plx_lo, plx_hi=plx_hi)
    add = [str(a) for a in (vz.get("out_add") or []) if str(a).strip()]
    if add:
        cons = {"-out.add": ",".join(add), **cons}
    rec: dict = {"catalogue": str(vz["gaia_catalogue"]), "out_max": max_rows,
                 "url": gaia_asu_url(c, field=field, plx_lo=plx_lo, plx_hi=plx_hi,
                                     max_rows=max_rows),
                 "constraints": cons, "attempts": [], "errors": []}
    try:
        raw, attempts = _fetch(str(vz["gaia_catalogue"]), c, fetch_fn=fetch_fn,
                               max_rows=max_rows, columns=_wanted(vz["gaia_columns"]),
                               constraints=cons)
    except VizierRouteError as exc:
        rec["attempts"] = list(exc.attempts or [])
        rec["errors"] = [str(a.get("error") or a.get("status")) for a in rec["attempts"]]
        rec["status"] = _classify_error(exc)
        rec["error"] = str(exc)
        return pd.DataFrame(), rec
    except Exception as exc:                               # noqa: BLE001
        rec.update(status=STATUS_FAILED, error=repr(exc))
        return pd.DataFrame(), rec
    rec["attempts"] = list(attempts)
    rec["endpoint"] = raw.attrs.get("endpoint")
    got, missing = resolve_columns(raw.columns, vz["gaia_columns"])
    rec["columns_resolved"] = got
    rec["columns_missing"] = missing
    hard = [k for k in REQUIRED_GAIA if k in missing]
    if hard:
        rec.update(status=STATUS_COLUMNS, error=f"required Gaia columns missing: {hard}",
                   n_rows=int(len(raw)))
        return pd.DataFrame(), rec
    out = pd.DataFrame({logical: raw[actual] for logical, actual in got.items()})
    if "source_id" in out:
        out["source_id"] = out["source_id"].astype(str).str.strip()
    rec["cuts_not_applied"] = [CUT_GAIA[k] for k in CUT_GAIA if k in missing]
    rec["n_rows"] = int(len(out))
    rec["capped"] = bool(len(out) >= max_rows)
    rec["status"] = STATUS_OK if len(out) else STATUS_ZERO
    return out, rec


def fetch_allwise_chunk(conf: dict | None = None, *, field: dict, fetch_fn=None
                        ) -> tuple[pd.DataFrame, dict]:
    """The AllWISE side of one cone, mapped onto the ESA column names."""
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    vz = vizier_conf(c)
    max_rows = int(vz["allwise_out_max"])
    pad = float(vz.get("cone_pad_arcmin") or 0.0) / 60.0
    cons = cone_constraints(c, field, pad_deg=pad)
    rec: dict = {"catalogue": str(vz["allwise_catalogue"]), "out_max": max_rows,
                 "url": allwise_asu_url(c, field=field, max_rows=max_rows),
                 "constraints": cons, "attempts": [], "errors": []}
    try:
        raw, attempts = _fetch(str(vz["allwise_catalogue"]), c, fetch_fn=fetch_fn,
                               max_rows=max_rows, columns=_wanted(vz["allwise_columns"]),
                               constraints=cons)
    except VizierRouteError as exc:
        rec["attempts"] = list(exc.attempts or [])
        rec["errors"] = [str(a.get("error") or a.get("status")) for a in rec["attempts"]]
        rec["status"] = _classify_error(exc)
        rec["error"] = str(exc)
        return pd.DataFrame(), rec
    except Exception as exc:                               # noqa: BLE001
        rec.update(status=STATUS_FAILED, error=repr(exc))
        return pd.DataFrame(), rec
    rec["attempts"] = list(attempts)
    rec["endpoint"] = raw.attrs.get("endpoint")
    got, missing = resolve_columns(raw.columns, vz["allwise_columns"])
    rec["columns_resolved"] = got
    rec["columns_missing"] = missing
    hard = [k for k in REQUIRED_ALLWISE if k in missing]
    if hard:
        rec.update(status=STATUS_COLUMNS, error=f"required AllWISE columns missing: {hard}",
                   n_rows=int(len(raw)))
        return pd.DataFrame(), rec
    out = pd.DataFrame({logical: raw[actual] for logical, actual in got.items()})
    rec["cuts_not_applied"] = [CUT_ALLWISE[k] for k in CUT_ALLWISE if k in missing]
    rec["n_rows"] = int(len(out))
    rec["capped"] = bool(len(out) >= max_rows)
    rec["status"] = STATUS_OK if len(out) else STATUS_ZERO
    return out, rec


def fetch_unit(conf: dict | None = None, unit: dict | None = None, *, cap: int | None = None,
               fetch_fn=None, label: str = "") -> tuple[pd.DataFrame, dict]:
    """One chunk of the parent sample over the VizieR route.

    Returns ``(rows, record)``.  ``rows`` carries exactly
    :func:`seti.ignition.sample.parent_columns` and has already passed the
    AllWISE predicates the ESA route writes in SQL; the Gaia-only cuts are left
    to ``select_parent``, which the caller runs over every route's frame alike.
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    unit = dict(unit or {})
    field = unit.get("field")
    rec: dict = {"route": ROUTE_VIZIER, "label": label, "unit": unit,
                 "endpoints": list(bases_for(c))}
    if not enabled(c):
        rec.update(status=STATUS_DISABLED, reason="config sample.vizier.enabled is false")
        return pd.DataFrame(), rec
    if field is None:
        rec.update(status=STATUS_UNSUPPORTED, n_rows=0,
                   reason=("a parallax shell is not a sky chunk, so there is no cone to pull "
                           "the AllWISE side over; the ASU route serves mode=fields only and "
                           "will not fabricate a cross-match for mode=allsky"))
        return pd.DataFrame(), rec

    gaia, grec = fetch_gaia_chunk(c, field=field, cap=cap, fetch_fn=fetch_fn)
    rec["gaia"] = grec
    if grec["status"] not in (STATUS_OK,):
        rec.update(status=grec["status"], n_rows=0,
                   error=grec.get("error") or grec.get("status"))
        return pd.DataFrame(), rec

    wise, wrec = fetch_allwise_chunk(c, field=field, fetch_fn=fetch_fn)
    rec["allwise"] = wrec
    if wrec["status"] not in (STATUS_OK,):
        rec.update(status=wrec["status"], n_rows=0,
                   error=wrec.get("error") or wrec.get("status"))
        return pd.DataFrame(), rec

    joined, mrec = match_allwise(gaia, wise, c)
    rec["match"] = mrec
    kept, counters = apply_allwise_predicates(
        joined, c, applied={**(grec.get("columns_resolved") or {}),
                            **(wrec.get("columns_resolved") or {})})
    rec["allwise_cut_counters"] = counters
    out = to_parent_frame(kept, c)
    rec["n_rows"] = int(len(out))
    rec["capped"] = bool(grec.get("capped") or wrec.get("capped"))
    rec["out_max"] = {"gaia": grec.get("out_max"), "allwise": wrec.get("out_max")}
    rec["cuts_not_applied"] = sorted(set(list(grec.get("cuts_not_applied") or [])
                                         + list(wrec.get("cuts_not_applied") or [])
                                         + list(counters.get("cuts_not_applied") or [])))
    rec["status"] = STATUS_OK if len(out) else STATUS_ZERO
    rec["row_count_note"] = (
        "n_rows is what the ASU -out.max cap actually returned for this chunk; it is not a "
        "COUNT(*) and must never be reported as a complete selection")
    return out, rec


def diagnose_zero(catalogue: str, *, columns, constraints: dict, conf: dict | None = None,
                  fetch_fn=None, cap: int = 5, field: dict | None = None) -> dict:
    """Why did this ASU request come back empty?  The ladder, and its verdict.

    Run 34796722335 reported ``QUERY_RETURNED_ZERO_ROWS`` for a bare one-degree
    cone on ``II/328/allwise`` --- a catalogue of 750 million sources, where an
    empty degree is not a possible sky.  A zero-row response is therefore a
    request defect until the ladder proves otherwise, and this is what proves
    it: the bare ``-source`` request first, then the columns, then one
    constraint at a time, with VizieR's own words recorded at the step that
    dies.

    When that constraint turns out to be the CONE (run 34799195807: rows
    survived the column list and died on ``-c``), the position ladder follows,
    sending every candidate spelling of the same cone until one returns rows.
    The two together take the record from "zero rows" to "this parameter, in
    this spelling, is what the service will accept".
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    try:
        steps = asu_constraint_ladder(str(catalogue), columns=_wanted(columns),
                                      constraints=dict(constraints or {}),
                                      max_rows=int(cap), fetch_fn=fetch_fn,
                                      bases=bases_for(c))
    except Exception as exc:                                # noqa: BLE001
        return {"status": STATUS_FAILED, "error": repr(exc), "steps": []}
    out = {"catalogue": str(catalogue), "steps": steps, "verdict": ladder_verdict(steps)}
    culprit = str(out["verdict"].get("culprit") or "")
    if field is not None and culprit.startswith(("+-c", "+-c.")):
        try:
            out["position_ladder"] = asu_position_ladder(
                str(catalogue), ra=float(field["ra"]), dec=float(field["dec"]),
                radius_deg=float(field["radius_deg"]), columns=_wanted(columns),
                max_rows=int(cap), fetch_fn=fetch_fn, bases=bases_for(c))
        except Exception as exc:                            # noqa: BLE001
            out["position_ladder"] = {"status": STATUS_FAILED, "error": repr(exc), "steps": []}
    return out


def probe_route(conf: dict | None = None, *, fetch_fn=None, cap: int = 5
                ) -> tuple[dict, pd.DataFrame]:
    """One minimal ASU call per catalogue, so ``probe.json`` names both routes.

    Asks each asserted catalogue id for a handful of rows of the first
    configured field.  A wrong id comes back ``CATALOGUE_NOT_FOUND`` with the
    URL and VizieR's own error text --- which is how the next run learns the id
    without anyone guessing again.  Returns ``(report, rows)``; the rows let the
    probe test the NEOWISE routes on a real star even when the ESA archive
    never answered.
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    vz = vizier_conf(c)
    fields = list(c.get("fields") or [])
    field = fields[0] if fields else {"ra": 266.0, "dec": 65.0, "radius_deg": 0.1}
    rep: dict = {"enabled": bool(vz.get("enabled", True)),
                 "endpoints": list(bases_for(c)),
                 "catalogues": {"gaia": str(vz["gaia_catalogue"]),
                                "gaia_params": str(vz["gaia_params_catalogue"]),
                                "allwise": str(vz["allwise_catalogue"])},
                 "asserted_unverified": ["gaia_catalogue", "gaia_params_catalogue",
                                         "allwise_catalogue", "gaia_columns",
                                         "allwise_columns", "constraint_columns"],
                 "field": field}
    if not rep["enabled"]:
        rep.update(status=STATUS_DISABLED, usable=False, n_rows=0)
        return rep, pd.DataFrame()
    _g, grec = fetch_gaia_chunk(c, field=field, cap=int(cap), fetch_fn=fetch_fn)
    rep["gaia"] = grec
    _w, wrec = fetch_allwise_chunk(c, field=field, fetch_fn=fetch_fn)
    rep["allwise"] = wrec
    # A request that returned nothing is diagnosed in the SAME run: a probe that
    # only says "zero rows" costs a whole dispatch cycle to learn nothing.
    vzc = vizier_conf(c)
    if str(grec.get("status")) in (STATUS_ZERO, STATUS_COLUMNS) or not grec.get("n_rows"):
        rep["gaia"]["ladder"] = diagnose_zero(
            vzc["gaia_catalogue"], columns=vzc["gaia_columns"],
            constraints=grec.get("constraints") or {}, conf=c, fetch_fn=fetch_fn,
            cap=int(cap), field=field)
    if str(wrec.get("status")) in (STATUS_ZERO, STATUS_COLUMNS) or not wrec.get("n_rows"):
        rep["allwise"]["ladder"] = diagnose_zero(
            vzc["allwise_catalogue"], columns=vzc["allwise_columns"],
            constraints=wrec.get("constraints") or {}, conf=c, fetch_fn=fetch_fn,
            cap=int(cap), field=field)
    rows, urec = fetch_unit(c, {"field": field}, cap=int(cap), fetch_fn=fetch_fn,
                            label="probe")
    rep["parent_chunk"] = urec
    rep["n_rows"] = int(len(rows))
    rep["status"] = urec.get("status", STATUS_FAILED)
    rep["usable"] = rep["status"] in (STATUS_OK, STATUS_ZERO)
    return rep, rows


__all__ = ["ALLWISE_EPOCH", "CUT_ALLWISE", "CUT_GAIA", "DEFAULT_VIZIER", "REQUIRED_ALLWISE",
           "REQUIRED_GAIA", "ROUTE_ESA", "ROUTE_VIZIER", "STATUS_COLUMNS", "STATUS_DISABLED",
           "STATUS_FAILED", "STATUS_NOT_FOUND", "STATUS_OK", "STATUS_UNSUPPORTED", "STATUS_ZERO",
           "allwise_asu_url", "apply_allwise_predicates", "asu_constraints", "bases_for",
           "cone_constraints", "diagnose_zero",
           "enabled", "fetch_allwise_chunk", "fetch_gaia_chunk", "fetch_unit", "gaia_asu_url",
           "match_allwise", "probe_route", "resolve_columns", "to_parent_frame", "vizier_conf"]
